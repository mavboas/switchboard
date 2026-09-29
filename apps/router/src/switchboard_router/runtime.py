"""De onde o router tira a configuração (banco do console ou YAML) e onde guarda as execuções.

O router nunca precisa reiniciar para pegar mudanças feitas no console: cada
perfil é relido do banco a cada ``config_ttl_s`` segundos e o motor só é
recriado quando a configuração efetiva muda (comparação por fingerprint).

O runtime também é dono do :class:`~switchboard.contracts.ContractManager`
(contratos com agentes A2A): liga o supervisor na subida, recebe as push
notifications e consolida as execuções com o motor do perfil de cada uma.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
from typing import Any, Protocol

import anyio.to_thread
import httpx

from switchboard.a2a import A2AClient, AgentDirectory
from switchboard.app import Switchboard
from switchboard.config import AgentSpec, ModelSpec
from switchboard.connectors import ConnectorCatalog
from switchboard.contracts import (
    Consolidation,
    ContractManager,
    ContractRecord,
    RunRecord,
    default_consolidation,
)
from switchboard.errors import ConfigError, SwitchboardError
from switchboard.jev import JevClient, build_decision_model
from switchboard.llm import OfflineChat, build_chat_model
from switchboard.llm.base import ChatModel
from switchboard.routing import ResolvedProfile, RouterEngine, RouterResult
from switchboard.secrets import SecretBox
from switchboard.storage import Database, EmbedderCache, KnowledgeService, SqlContractStore, repo
from switchboard.tracing import Span

log = logging.getLogger("switchboard.router")


class ProfileNotFound(ConfigError):
    pass


class Runtime(Protocol):
    connectors: ConnectorCatalog
    agents: AgentDirectory
    contracts: ContractManager
    source: str

    async def engine(self, profile: str) -> RouterEngine: ...

    async def profiles(self) -> list[str]: ...

    async def record(self, result: RouterResult) -> None: ...

    async def run_details(self, run_id: str) -> dict[str, Any] | None: ...

    async def ready(self) -> bool: ...

    async def start(self) -> None: ...

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
        public_url: str | None = None,
        supervisor_tick_s: float = 1.0,
        connectors: ConnectorCatalog | None = None,
        agents: AgentDirectory | None = None,
        http: httpx.AsyncClient | None = None,
    ):
        self.db = db
        self.box = box
        self.config_ttl_s = config_ttl_s
        self.connectors = connectors or ConnectorCatalog(
            ttl_s=agents_ttl_s, resolve_secret=box.open
        )
        self.agents = agents or AgentDirectory(
            client=A2AClient(http=http), ttl_s=agents_ttl_s, resolve_secret=box.open
        )
        self.contracts = ContractManager(
            SqlContractStore(db),
            client=self.agents.client,
            resolve_secret=box.open,
            agent_resolver=self._agent_spec,
            public_url=public_url,
            tick_s=supervisor_tick_s,
        )
        self.contracts.consolidator = self.consolidate
        self.knowledge = KnowledgeService(db, EmbedderCache(box.open))
        self._profiles: dict[str, tuple[float, ResolvedProfile | None]] = {}
        self._models: dict[str, ChatModel] = {}
        self._deciders: dict[str, JevClient] = {}
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

    async def _agent_spec(self, name: str) -> AgentSpec | None:
        def load() -> AgentSpec | None:
            with self.db.session() as session:
                return repo.find_agent_spec(session, name)

        return await anyio.to_thread.run_sync(load)

    def _chat(self, spec: ModelSpec) -> ChatModel:
        key = _fingerprint(spec.model_dump(mode="json"))
        if key not in self._models:
            self._models[key] = build_chat_model(spec, resolve_secret=self.box.open)
        return self._models[key]

    def _decider(self, spec: ModelSpec) -> JevClient:
        key = _fingerprint(spec.model_dump(mode="json"))
        if key not in self._deciders:
            self._deciders[key] = build_decision_model(spec, resolve_secret=self.box.open)
        return self._deciders[key]

    async def engine(self, profile: str) -> RouterEngine:
        resolved = await self._resolved(profile)
        if resolved is None:
            raise ProfileNotFound(f"roteador '{profile}' não existe ou está desabilitado")
        fingerprint = _fingerprint(
            [
                resolved.spec.model_dump(mode="json"),
                resolved.model.model_dump(mode="json"),
                resolved.decision_model.model_dump(mode="json")
                if resolved.decision_model
                else None,
                [c.model_dump(mode="json") for c in resolved.connectors],
                [a.model_dump(mode="json") for a in resolved.agents],
            ]
        )
        async with self._lock:
            cached = self._engines.get(profile)
            if cached and cached[0] == fingerprint:
                return cached[1]
            warnings: list[str] = []
            try:
                chat = self._chat(resolved.model)
            except SwitchboardError as exc:
                # segredo ilegível ou config inválida: atende em modo offline e avisa em cada trace
                log.error("modelo '%s' indisponível: %s", resolved.model.name, exc)
                chat = OfflineChat(label=f"offline (fallback de {resolved.model.name})")
                warnings.append(
                    f"modelo '{resolved.model.name}' indisponível, usando o modo offline: {exc}"
                )
            jev = None
            if resolved.decision_model is not None:
                try:
                    jev = self._decider(resolved.decision_model)
                except SwitchboardError as exc:
                    log.error(
                        "modelo de decisão '%s' indisponível: %s", resolved.decision_model.name, exc
                    )
                    warnings.append(
                        f"modelo de decisão '{resolved.decision_model.name}' indisponível, decidindo com o LLM: {exc}"
                    )
            engine = RouterEngine(
                resolved,
                chat=chat,
                connectors=self.connectors,
                agents=self.agents,
                contracts=self.contracts,
                retriever=self.knowledge,
                jev=jev,
                warnings=warnings,
            )
            self._engines[profile] = (fingerprint, engine)
            self._retire_unused()
            log.info(
                "config do roteador '%s' carregada (decisão: %s)", profile, engine.decision_label
            )
            return engine

    def _retire_unused(self) -> None:
        in_use = {id(engine.chat) for _, engine in self._engines.values()}
        in_use |= {id(engine.jev) for _, engine in self._engines.values() if engine.jev is not None}
        for cache in (self._models, self._deciders):
            for key, client in list(cache.items()):
                if id(client) not in in_use:
                    cache.pop(key)
                    asyncio.get_running_loop().create_task(self._close_later(client))

    @staticmethod
    async def _close_later(client: Any, delay: float = 300.0) -> None:
        # deixa terminar pedidos em andamento; se algum ainda usar o cliente depois
        # disso, o adaptador devolve erro e o motor cai para o plano B
        await asyncio.sleep(delay)
        with contextlib.suppress(Exception):
            await client.aclose()

    async def consolidate(self, run: RunRecord, contracts: list[ContractRecord]) -> Consolidation:
        try:
            engine = await self.engine(run.profile)
        except SwitchboardError:  # roteador removido ou desabilitado: consolida sem LLM
            return Consolidation(default_consolidation(run, contracts))
        return await engine.consolidate(run, contracts)

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

    async def run_details(self, run_id: str) -> dict[str, Any] | None:
        def load():
            with self.db.session() as session:
                return repo.run_details(session, run_id)

        return await anyio.to_thread.run_sync(load)

    async def ready(self) -> bool:
        return await anyio.to_thread.run_sync(self.db.ping)

    async def start(self) -> None:
        self.contracts.start()

    async def aclose(self) -> None:
        await self.contracts.stop()
        for client in [*self._models.values(), *self._deciders.values()]:
            with contextlib.suppress(Exception):
                await client.aclose()
        await self.agents.aclose()
        await self.knowledge.aclose()
        self.db.dispose()


class FileRuntime:
    """Modo framework: perfis vindos de um switchboard.yaml (sem banco; execuções em memória)."""

    source = "yaml"

    def __init__(self, switchboard: Switchboard, *, public_url: str | None = None):
        self.switchboard = switchboard
        self.connectors = switchboard.connectors
        self.agents = switchboard.agents
        self.contracts = switchboard.contracts
        if public_url:
            self.contracts.public_url = public_url.rstrip("/")

    async def engine(self, profile: str) -> RouterEngine:
        await self.switchboard.load()
        try:
            return self.switchboard.engine(profile)
        except ConfigError as exc:
            raise ProfileNotFound(str(exc)) from exc

    async def profiles(self) -> list[str]:
        return [p.name for p in self.switchboard.spec.profiles]

    async def record(self, result: RouterResult) -> None:
        # no modo YAML o trace sai no log; só as execuções com agentes guardam spans (em memória)
        if result.route == "delegated" and result.spans:
            await self.contracts.store.add_spans([Span.from_dict(s) for s in result.spans])

    async def run_details(self, run_id: str) -> dict[str, Any] | None:
        store = self.contracts.store
        run = await store.get_run(run_id)
        if run is None:
            return None
        contracts = []
        for c in await store.run_contracts(run_id):
            contracts.append(c.to_dict(await store.contract_events(c.id)))
        spans = await store.run_spans(run_id) if hasattr(store, "run_spans") else []
        contract_spans = repo.contract_spans(contracts, run_id)
        all_spans = sorted([*spans, *contract_spans], key=lambda s: s.started_at)
        return {
            **run.to_dict(),
            "trace_id": run.id,
            "route": "delegated",
            "contracts": contracts,
            "spans": [s.to_dict() for s in all_spans],
        }

    async def ready(self) -> bool:
        return True

    async def start(self) -> None:
        await self.switchboard.load()
        self.contracts.start()

    async def aclose(self) -> None:
        await self.switchboard.aclose()
