"""Diretório de agentes A2A: descoberta pelo Agent Card, com cache.

Para cada agente habilitado, o diretório busca o card, escolhe a interface
JSON-RPC 1.x, confere que o endpoint fica no mesmo host da URL cadastrada
(um card malicioso não consegue apontar o roteador para outro lugar — a menos
que ``allow_cross_origin`` esteja ligado) e lê os termos de contrato de cada
skill, se o agente declarar a extensão ``urn:switchboard:a2a:contract:v1``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit

from ..config import AgentSpec
from ..contracts.terms import CONTRACT_EXTENSION_URI, SkillTerms, skills_from_extension
from ..discovery import DiscoveryCache, describe_error
from ..errors import AgentError
from ..secrets import resolve_env
from .card import AgentCard
from .client import A2AClient

AgentStatus = Literal["online", "offline"]


@dataclass(frozen=True)
class SkillInfo:
    id: str
    name: str
    description: str
    examples: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    terms: SkillTerms | None = None
    problems: tuple[str, ...] = ()

    @property
    def contract(self) -> str:
        """``completo`` (schemas declarados pelo agente) ou ``basico`` (texto)."""
        return "completo" if self.terms is not None else "basico"

    @property
    def usable(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "examples": list(self.examples),
            "tags": list(self.tags),
            "contract": self.contract,
            "input_schema": self.terms.input_schema if self.terms else None,
            "output_schema": self.terms.output_schema if self.terms else None,
            "input_hash": self.terms.input_hash if self.terms else None,
            "output_hash": self.terms.output_hash if self.terms else None,
            "max_duration_s": self.terms.max_duration_s if self.terms else None,
            "problems": list(self.problems),
        }


@dataclass
class AgentInfo:
    name: str
    description: str
    url: str
    status: AgentStatus
    rpc_url: str | None = None
    protocol_version: str | None = None
    push: bool = False
    streaming: bool = False
    contract_extension: bool = False
    skills: list[SkillInfo] = field(default_factory=list)
    card_name: str | None = None
    card_version: str | None = None
    error: str | None = None
    latency_ms: float = 0.0
    checked_at: float = field(default_factory=time.time)

    def skill(self, skill_id: str) -> SkillInfo | None:
        return next((s for s in self.skills if s.id == skill_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "status": self.status,
            "rpc_url": self.rpc_url,
            "protocol_version": self.protocol_version,
            "push": self.push,
            "streaming": self.streaming,
            "contract_extension": self.contract_extension,
            "card": {"name": self.card_name, "version": self.card_version},
            "skills": [s.to_dict() for s in self.skills],
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
        }


# nomes da própria máquina: "localhost" e "127.0.0.1" na mesma porta são o mesmo servidor
LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    port = parts.port or {"http": 80, "https": 443}.get(parts.scheme)
    host = (parts.hostname or "").lower()
    return parts.scheme, "loopback" if host in LOOPBACK else host, port


def resolve_rpc_url(spec: AgentSpec, card: AgentCard) -> tuple[str, str]:
    """Endpoint JSON-RPC 1.x do card, validado contra a URL cadastrada."""
    interface = card.jsonrpc_interface()
    if interface is None:
        offered = ", ".join(f"{i.binding} {i.version}" for i in card.interfaces) or "nenhuma"
        raise AgentError(
            f"o agente não oferece interface JSON-RPC 1.x (interfaces do card: {offered})"
        )
    base = spec.url if spec.url.endswith((".json", "/")) else spec.url + "/"
    rpc_url = urljoin(base, interface.url)
    if not spec.allow_cross_origin and _origin(rpc_url) != _origin(spec.url):
        raise AgentError(
            f"o card aponta o JSON-RPC para {rpc_url}, fora do host cadastrado ({spec.url}); "
            "ligue 'allow_cross_origin' no agente se isso for esperado"
        )
    return rpc_url, interface.version


def agent_info_from_card(spec: AgentSpec, card: AgentCard, latency_ms: float = 0.0) -> AgentInfo:
    rpc_url, version = resolve_rpc_url(spec, card)
    extension = card.extension(CONTRACT_EXTENSION_URI)
    terms, problems = skills_from_extension(extension.get("params") if extension else None)
    skills = []
    for skill in card.skills:
        if spec.allowed_skills and skill.id not in spec.allowed_skills:
            continue
        skills.append(
            SkillInfo(
                id=skill.id,
                name=skill.name,
                description=skill.description,
                examples=skill.examples,
                tags=skill.tags,
                terms=terms.get(skill.id),
                problems=tuple(problems.get(skill.id, ())),
            )
        )
    return AgentInfo(
        name=spec.name,
        description=spec.description.strip() or card.description,
        url=spec.url,
        status="online",
        rpc_url=rpc_url,
        protocol_version=version,
        push=card.push,
        streaming=card.streaming,
        contract_extension=extension is not None,
        skills=skills,
        card_name=card.name,
        card_version=card.version,
        latency_ms=latency_ms,
    )


class AgentDirectory:
    def __init__(
        self,
        *,
        client: A2AClient | None = None,
        ttl_s: float = 30.0,
        offline_ttl_s: float = 15.0,
        discovery_timeout_s: float = 5.0,
        resolve_secret: Callable[[str | None], str | None] = resolve_env,
    ):
        self.client = client or A2AClient()
        self.discovery_timeout_s = discovery_timeout_s
        self._resolve = resolve_secret
        self._cache: DiscoveryCache[AgentSpec, AgentInfo] = DiscoveryCache(
            self._probe,
            online=lambda info: info.status == "online",
            name_of=lambda info: info.name,
            ttl_s=ttl_s,
            offline_ttl_s=offline_ttl_s,
        )

    def token(self, spec: AgentSpec) -> str | None:
        return self._resolve(spec.auth_token)

    def invalidate(self, name: str | None = None) -> None:
        self._cache.invalidate(name)

    async def describe(self, spec: AgentSpec, *, refresh: bool = False) -> AgentInfo:
        return await self._cache.get(spec, refresh=refresh)

    async def discover(
        self, agents: Sequence[AgentSpec], *, refresh: bool = False
    ) -> list[AgentInfo]:
        enabled = [a for a in agents if a.enabled]
        if not enabled:
            return []
        return list(await asyncio.gather(*(self.describe(a, refresh=refresh) for a in enabled)))

    async def _probe(self, spec: AgentSpec) -> AgentInfo:
        started = time.perf_counter()
        try:
            timeout = min(spec.timeout_s, self.discovery_timeout_s)
            async with asyncio.timeout(timeout):
                data = await self.client.fetch_card(
                    spec.url, token=self.token(spec), timeout_s=timeout
                )
            card = AgentCard.parse(data)
            return agent_info_from_card(spec, card, (time.perf_counter() - started) * 1000)
        except Exception as exc:  # qualquer falha = agente offline, com o motivo
            message = str(exc) if isinstance(exc, AgentError) else describe_error(exc)
            return AgentInfo(
                name=spec.name,
                description=spec.description,
                url=spec.url,
                status="offline",
                error=message[:400],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

    async def aclose(self) -> None:
        await self.client.aclose()
