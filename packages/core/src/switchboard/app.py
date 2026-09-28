"""Modo framework: o roteador como biblioteca, configurado por YAML.

Exemplo::

    from switchboard import Switchboard

    async with Switchboard.from_yaml("switchboard.yaml") as sb:
        result = await sb.ask("Quanto fica a parcela de 50 mil em 24 meses a 1,5%?")
        print(result.route, result.status, result.answer)
        if result.status == "pending":  # delegação a agente A2A ainda em andamento
            run = await sb.wait(result.run_id, timeout_s=120)
            print(run.answer)

Contratos com agentes ficam em memória (sem push notifications: o
acompanhamento é por polling).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from .a2a import AgentDirectory, AgentInfo
from .config import AgentSpec, SwitchboardSpec, load_yaml
from .connectors import ConnectorCatalog, ConnectorInfo
from .contracts import (
    Consolidation,
    ContractManager,
    ContractRecord,
    MemoryContractStore,
    RunRecord,
    default_consolidation,
)
from .embeddings import build_embedder
from .jev import JevClient, build_decision_model
from .llm.base import ChatModel, Message
from .llm.factory import build_chat_model
from .rag.retriever import MemoryRetriever
from .routing.engine import RouterEngine
from .routing.types import ResolvedProfile, RouterResult


class Switchboard:
    def __init__(
        self,
        spec: SwitchboardSpec,
        *,
        base_dir: str | Path | None = None,
        connectors: ConnectorCatalog | None = None,
        agents: AgentDirectory | None = None,
        contracts: ContractManager | None = None,
    ):
        self.spec = spec
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.connectors = connectors or ConnectorCatalog()
        self.agents = agents or AgentDirectory()
        self.contracts = contracts or ContractManager(
            MemoryContractStore(), client=self.agents.client, agent_resolver=self._agent_spec
        )
        self.contracts.consolidator = self._consolidate
        self.retriever = MemoryRetriever()
        self._models: dict[str, ChatModel] = {}
        self._decision_models: dict[str, JevClient] = {}
        self._engines: dict[str, RouterEngine] = {}
        self._loaded = False
        self._lock = asyncio.Lock()

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs) -> Switchboard:
        path = Path(path)
        return cls(load_yaml(path), base_dir=path.parent, **kwargs)

    async def __aenter__(self) -> Switchboard:
        await self.load()
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.aclose()

    async def load(self) -> None:
        """Indexa os arquivos das bases de conhecimento (uma vez) e liga o supervisor."""
        async with self._lock:
            if self._loaded:
                return
            for kb in self.spec.knowledge_bases:
                connection = (
                    self.spec.model(kb.embedder.connection) if kb.embedder.kind == "model" else None
                )
                self.retriever.register(kb, build_embedder(kb.embedder, connection))
                if kb.paths:
                    await self.retriever.load_paths(kb.name, self.base_dir)
            if self.spec.agents:
                self.contracts.start()
            self._loaded = True

    async def _agent_spec(self, name: str) -> AgentSpec | None:
        return next((a for a in self.spec.agents if a.name == name), None)

    def resolve(self, profile: str | None = None) -> ResolvedProfile:
        spec = self.spec.profile(profile)
        return ResolvedProfile(
            spec=spec,
            model=self.spec.model(spec.model),
            connectors=[c for c in self.spec.connectors if c.name in spec.connectors],
            agents=[a for a in self.spec.agents if a.name in spec.agents],
            decision_model=self.spec.model(spec.decision_model) if spec.decision_model else None,
        )

    def engine(self, profile: str | None = None) -> RouterEngine:
        resolved = self.resolve(profile)
        if resolved.name not in self._engines:
            model = self._models.get(resolved.model.name)
            if model is None:
                model = self._models[resolved.model.name] = build_chat_model(resolved.model)
            jev = None
            if resolved.decision_model is not None:
                jev = self._decision_models.get(resolved.decision_model.name)
                if jev is None:
                    jev = self._decision_models[resolved.decision_model.name] = (
                        build_decision_model(resolved.decision_model)
                    )
            self._engines[resolved.name] = RouterEngine(
                resolved,
                chat=model,
                connectors=self.connectors,
                agents=self.agents,
                contracts=self.contracts,
                retriever=self.retriever,
                jev=jev,
            )
        return self._engines[resolved.name]

    async def ask(
        self,
        message: str | Sequence[Message] | Sequence[dict],
        *,
        profile: str | None = None,
        wait_s: float | None = None,
    ) -> RouterResult:
        await self.load()
        return await self.engine(profile).handle(message, wait_s=wait_s)

    async def reply(
        self, run_id: str, message: str, *, profile: str | None = None, wait_s: float | None = None
    ) -> RouterResult:
        """Responde ao agente de uma execução que está aguardando entrada."""
        await self.load()
        run = await self.contracts.store.get_run(run_id)
        name = run.profile if run else profile
        return await self.engine(name).resume(run_id, message, wait_s=wait_s)

    async def run(self, run_id: str) -> RunRecord | None:
        return await self.contracts.store.get_run(run_id)

    async def run_contracts(self, run_id: str) -> list[ContractRecord]:
        return await self.contracts.store.run_contracts(run_id)

    async def wait(self, run_id: str, timeout_s: float = 60.0) -> RunRecord | None:
        return await self.contracts.wait_run(run_id, timeout_s)

    async def _consolidate(self, run: RunRecord, contracts: list[ContractRecord]) -> Consolidation:
        try:
            engine = self.engine(run.profile)
        except Exception:  # perfil removido: consolida sem LLM
            return Consolidation(default_consolidation(run, contracts))
        return await engine.consolidate(run, contracts)

    async def agents_status(
        self, profile: str | None = None, *, refresh: bool = False
    ) -> list[AgentInfo]:
        return await self.agents.discover(self.resolve(profile).agents, refresh=refresh)

    async def connectors_status(
        self, profile: str | None = None, *, refresh: bool = False
    ) -> list[ConnectorInfo]:
        return await self.connectors.discover(self.resolve(profile).connectors, refresh=refresh)

    async def aclose(self) -> None:
        await self.contracts.stop()
        for model in self._models.values():
            await model.aclose()
        for jev in self._decision_models.values():
            await jev.aclose()
        await self.retriever.aclose()
        await self.agents.aclose()
        self._models.clear()
        self._decision_models.clear()
        self._engines.clear()
