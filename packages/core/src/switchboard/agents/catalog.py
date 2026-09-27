"""Descoberta e acionamento de agentes via MCP.

Cada agente registrado é um servidor MCP: as *tools* que ele expõe são as
capacidades que o roteador pode delegar. O catálogo:

1. conecta em cada agente (Streamable HTTP ou SSE), faz o handshake e chama
   ``tools/list`` — o agente fica ``online`` com suas tools, ou ``offline``
   com o erro;
2. guarda o resultado por ``ttl_s`` segundos (agentes offline por menos
   tempo, para voltarem rápido);
3. aplica a allowlist de tools do agente, se houver;
4. executa ``tools/call`` quando o roteador decide delegar.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from ..config import AgentSpec
from ..errors import AgentError
from ..secrets import resolve_env

AgentStatus = Literal["online", "offline"]


@dataclass(frozen=True)
class ToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class AgentInfo:
    name: str
    description: str
    url: str
    status: AgentStatus
    tools: list[ToolInfo] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0
    server_name: str | None = None
    checked_at: float = field(default_factory=time.time)

    def tool(self, name: str) -> ToolInfo | None:
        return next((t for t in self.tools if t.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "status": self.status,
            "tools": [t.to_dict() for t in self.tools],
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
            "server_name": self.server_name,
        }


@dataclass
class ToolCallResult:
    text: str
    is_error: bool
    structured: dict[str, Any] | None
    latency_ms: float


# Uma "conexão" é um context manager assíncrono que entrega um mcp.Client já
# inicializado. O padrão abre HTTP; os testes injetam servidores em processo.
Connector = Callable[[AgentSpec, str | None], AbstractAsyncContextManager[Any]]


@asynccontextmanager
async def http_connector(spec: AgentSpec, token: str | None) -> AsyncIterator[Any]:
    import httpx2
    from mcp import Client
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamable_http_client

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if spec.transport == "sse":
        transport = sse_client(
            spec.url, headers=headers, timeout=spec.timeout_s, sse_read_timeout=spec.timeout_s
        )
        async with Client(transport, read_timeout_seconds=spec.timeout_s) as client:
            yield client
        return
    http = httpx2.AsyncClient(
        headers=headers, timeout=httpx2.Timeout(spec.timeout_s, read=spec.timeout_s)
    )
    async with http:
        transport = streamable_http_client(spec.url, http_client=http)
        async with Client(transport, read_timeout_seconds=spec.timeout_s) as client:
            yield client


def _describe_error(exc: BaseException) -> str:
    # ExceptionGroup (anyio/TaskGroup) esconde a causa real lá dentro.
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    text = str(exc).strip() or exc.__class__.__name__
    return f"{exc.__class__.__name__}: {text}"[:400]


def _dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=True, exclude_none=True, mode="json")
    return dict(model)


def result_text(result: Any) -> tuple[str, bool, dict[str, Any] | None]:
    """Texto legível de um CallToolResult (texto das partes ou o JSON estruturado)."""
    data = _dump(result)
    parts = []
    for item in data.get("content") or []:
        if item.get("type") == "text" and item.get("text"):
            parts.append(item["text"])
        elif item.get("type") == "resource":
            resource = item.get("resource") or {}
            if resource.get("text"):
                parts.append(resource["text"])
    structured = data.get("structuredContent")
    text = "\n".join(parts).strip()
    if not text and structured is not None:
        text = json.dumps(structured, ensure_ascii=False, indent=2)
    return text, bool(data.get("isError")), structured


class AgentCatalog:
    def __init__(
        self,
        *,
        ttl_s: float = 30.0,
        offline_ttl_s: float = 5.0,
        connector: Connector = http_connector,
        resolve_secret: Callable[[str | None], str | None] = resolve_env,
    ):
        self.ttl_s = ttl_s
        self.offline_ttl_s = offline_ttl_s
        self._connector = connector
        self._resolve = resolve_secret
        self._cache: dict[str, tuple[float, AgentInfo]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _key(spec: AgentSpec) -> str:
        raw = json.dumps(spec.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def invalidate(self, name: str | None = None) -> None:
        if name is None:
            self._cache.clear()
            return
        for key, (_, info) in list(self._cache.items()):
            if info.name == name:
                self._cache.pop(key, None)

    async def describe(self, spec: AgentSpec, *, refresh: bool = False) -> AgentInfo:
        key = self._key(spec)
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached and not refresh and cached[0] > now:
            return cached[1]
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._cache.get(key)
            if cached and not refresh and cached[0] > time.monotonic():
                return cached[1]
            info = await self._probe(spec)
            ttl = self.ttl_s if info.status == "online" else self.offline_ttl_s
            self._cache[key] = (time.monotonic() + ttl, info)
            return info

    async def _probe(self, spec: AgentSpec) -> AgentInfo:
        started = time.perf_counter()
        try:
            token = self._resolve(spec.auth_token)
            async with asyncio.timeout(spec.timeout_s):
                async with self._connector(spec, token) as client:
                    listing = await client.list_tools()
                    server = client.server_info
                    instructions = client.instructions
        except Exception as exc:  # qualquer falha = agente offline, com o motivo
            return AgentInfo(
                name=spec.name,
                description=spec.description,
                url=spec.url,
                status="offline",
                error=_describe_error(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        tools = []
        for tool in listing.tools:
            data = _dump(tool)
            name = data.get("name", "")
            if spec.allowed_tools and name not in spec.allowed_tools:
                continue
            tools.append(
                ToolInfo(
                    name=name,
                    description=(data.get("description") or data.get("title") or "").strip(),
                    input_schema=data.get("inputSchema") or {"type": "object", "properties": {}},
                )
            )
        description = spec.description.strip() or (instructions or "").strip()
        if not description and server is not None:
            description = (
                getattr(server, "description", None) or getattr(server, "title", None) or ""
            ).strip()
        return AgentInfo(
            name=spec.name,
            description=description,
            url=spec.url,
            status="online",
            tools=tools,
            latency_ms=(time.perf_counter() - started) * 1000,
            server_name=getattr(server, "name", None) if server is not None else None,
        )

    async def discover(
        self, agents: Sequence[AgentSpec], *, refresh: bool = False
    ) -> list[AgentInfo]:
        """Descobre todos os agentes habilitados em paralelo."""
        enabled = [a for a in agents if a.enabled]
        if not enabled:
            return []
        return list(await asyncio.gather(*(self.describe(a, refresh=refresh) for a in enabled)))

    async def call(self, spec: AgentSpec, tool: str, arguments: dict[str, Any]) -> ToolCallResult:
        if spec.allowed_tools and tool not in spec.allowed_tools:
            raise AgentError(f"a tool '{tool}' não está liberada para o agente '{spec.name}'")
        started = time.perf_counter()
        try:
            token = self._resolve(spec.auth_token)
            async with asyncio.timeout(spec.timeout_s):
                async with self._connector(spec, token) as client:
                    result = await client.call_tool(
                        tool, arguments, read_timeout_seconds=spec.timeout_s
                    )
        except Exception as exc:
            self.invalidate(spec.name)
            raise AgentError(
                f"falha ao acionar {spec.name}/{tool}: {_describe_error(exc)}"
            ) from exc
        text, is_error, structured = result_text(result)
        return ToolCallResult(text, is_error, structured, (time.perf_counter() - started) * 1000)
