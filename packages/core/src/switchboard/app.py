"""Modo framework: o roteador como biblioteca, configurado por YAML.

Exemplo::

    from switchboard import Switchboard

    async with Switchboard.from_yaml("switchboard.yaml") as sb:
        result = await sb.ask("Quanto fica a parcela de 50 mil em 24 meses a 1,5%?")
        print(result.route, result.answer)
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from .agents.catalog import AgentCatalog, AgentInfo
from .config import SwitchboardSpec, load_yaml
from .embeddings import build_embedder
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
        catalog: AgentCatalog | None = None,
    ):
        self.spec = spec
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.catalog = catalog or AgentCatalog()
        self.retriever = MemoryRetriever()
        self._models: dict[str, ChatModel] = {}
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
        """Indexa os arquivos das bases de conhecimento (uma vez)."""
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
            self._loaded = True

    def resolve(self, profile: str | None = None) -> ResolvedProfile:
        spec = self.spec.profile(profile)
        agents = [a for a in self.spec.agents if a.name in spec.agents]
        return ResolvedProfile(spec=spec, model=self.spec.model(spec.model), agents=agents)

    def engine(self, profile: str | None = None) -> RouterEngine:
        resolved = self.resolve(profile)
        if resolved.name not in self._engines:
            model = self._models.get(resolved.model.name)
            if model is None:
                model = self._models[resolved.model.name] = build_chat_model(resolved.model)
            self._engines[resolved.name] = RouterEngine(
                resolved, chat=model, catalog=self.catalog, retriever=self.retriever
            )
        return self._engines[resolved.name]

    async def ask(
        self, message: str | Sequence[Message] | Sequence[dict], *, profile: str | None = None
    ) -> RouterResult:
        await self.load()
        return await self.engine(profile).handle(message)

    async def agents(self, profile: str | None = None, *, refresh: bool = False) -> list[AgentInfo]:
        return await self.catalog.discover(self.resolve(profile).agents, refresh=refresh)

    async def aclose(self) -> None:
        for model in self._models.values():
            await model.aclose()
        await self.retriever.aclose()
        self._models.clear()
        self._engines.clear()
