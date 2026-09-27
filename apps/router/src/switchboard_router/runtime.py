"""De onde o router tira a configuração: banco (console) ou YAML.

O router nunca precisa reiniciar para pegar mudanças feitas no console: cada
perfil é relido do banco a cada ``config_ttl_s`` segundos e o motor só é
recriado quando a configuração efetiva muda (comparação por fingerprint).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
from typing import Protocol

import anyio.to_thread

from switchboard.agents import AgentCatalog
from switchboard.app import Switchboard
from switchboard.config import ModelSpec
from switchboard.errors import ConfigError
from switchboard.llm import build_chat_model
from switchboard.llm.base import ChatModel
from switchboard.routing import ResolvedProfile, RouterEngine, RouterResult
from switchboard.secrets import SecretBox
from switchboard.storage import Database, EmbedderCache, KnowledgeService, repo

log = logging.getLogger("switchboard.router")


class ProfileNotFound(ConfigError):
    pass


class Runtime(Protocol):
    catalog: AgentCatalog
    source: str

    async def engine(self, profile: str) -> RouterEngine: ...

    async def profiles(self) -> list[str]: ...

    async def record(self, result: RouterResult) -> None: ...

    def ready(self) -> bool: ...

    async def aclose(self) -> None: ...


def _fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class DbRuntime:
    source = "database"

    def __init__(
        self,
        db: Database,
        box: SecretBox,
        *,
        config_ttl_s: float = 5.0,
        agents_ttl_s: float = 30.0,
        catalog: AgentCatalog | None = None,
    ):
        self.db = db
        self.box = box
        self.config_ttl_s = config_ttl_s
        self.catalog = catalog or AgentCatalog(ttl_s=agents_ttl_s, resolve_secret=box.open)
        self.knowledge = KnowledgeService(db, EmbedderCache(box.open))
        self._profiles: dict[str, tuple[float, ResolvedProfile | None]] = {}
        self._models: dict[str, ChatModel] = {}
        self._engines: dict[str, tuple[str, RouterEngine]] = {}
        self._lock = asyncio.Lock()

    async def _resolved(self, name: str) -> ResolvedProfile | None:
        cached = self._profiles.get(name)
        if cached and cached[0] > time.monotonic():
            return cached[1]

        def load() -> ResolvedProfile | None:
            with self.db.session() as session:
                return repo.resolve_profile(session, name)

        resolved = await anyio.to_thread.run_sync(load)
        self._profiles[name] = (time.monotonic() + self.config_ttl_s, resolved)
        return resolved

    def _chat(self, spec: ModelSpec) -> ChatModel:
        key = _fingerprint(spec.model_dump(mode="json"))
        if key not in self._models:
            self._models[key] = build_chat_model(spec, resolve_secret=self.box.open)
        return self._models[key]

    async def engine(self, profile: str) -> RouterEngine:
        resolved = await self._resolved(profile)
        if resolved is None:
            raise ProfileNotFound(f"roteador '{profile}' não existe ou está desabilitado")
        fingerprint = _fingerprint(
            [
                resolved.spec.model_dump(mode="json"),
                resolved.model.model_dump(mode="json"),
                [a.model_dump(mode="json") for a in resolved.agents],
            ]
        )
        async with self._lock:
            cached = self._engines.get(profile)
            if cached and cached[0] == fingerprint:
                return cached[1]
            engine = RouterEngine(
                resolved,
                chat=self._chat(resolved.model),
                catalog=self.catalog,
                retriever=self.knowledge,
            )
            self._engines[profile] = (fingerprint, engine)
            self._retire_unused_models()
            log.info("config do roteador '%s' carregada (modelo %s)", profile, engine.chat.label)
            return engine

    def _retire_unused_models(self) -> None:
        in_use = {id(engine.chat) for _, engine in self._engines.values()}
        for key, model in list(self._models.items()):
            if id(model) not in in_use:
                self._models.pop(key)
                asyncio.get_running_loop().create_task(self._close_later(model))

    @staticmethod
    async def _close_later(model: ChatModel, delay: float = 30.0) -> None:
        await asyncio.sleep(delay)  # deixa terminar pedidos em andamento
        with contextlib.suppress(Exception):
            await model.aclose()

    async def profiles(self) -> list[str]:
        def load() -> list[str]:
            with self.db.session() as session:
                return repo.enabled_profile_names(session)

        return await anyio.to_thread.run_sync(load)

    async def record(self, result: RouterResult) -> None:
        def save() -> None:
            with self.db.session() as session:
                repo.save_trace(session, result)

        try:
            await anyio.to_thread.run_sync(save)
        except Exception:  # trace perdido não pode derrubar o atendimento
            log.exception("falha ao gravar o trace %s", result.trace_id)

    def ready(self) -> bool:
        return self.db.ping()

    async def aclose(self) -> None:
        for model in self._models.values():
            with contextlib.suppress(Exception):
                await model.aclose()
        await self.knowledge.aclose()
        self.db.dispose()


class FileRuntime:
    """Modo framework: perfis vindos de um switchboard.yaml (sem banco)."""

    source = "yaml"

    def __init__(self, switchboard: Switchboard):
        self.switchboard = switchboard
        self.catalog = switchboard.catalog

    async def engine(self, profile: str) -> RouterEngine:
        await self.switchboard.load()
        try:
            return self.switchboard.engine(profile)
        except ConfigError as exc:
            raise ProfileNotFound(str(exc)) from exc

    async def profiles(self) -> list[str]:
        return [p.name for p in self.switchboard.spec.profiles]

    async def record(self, result: RouterResult) -> None:
        return None  # no modo YAML o trace sai só no log

    def ready(self) -> bool:
        return True

    async def aclose(self) -> None:
        await self.switchboard.aclose()
