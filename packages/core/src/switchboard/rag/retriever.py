"""Contrato de busca do RAG e a implementação em memória (modo framework)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import anyio.to_thread

from ..config import KnowledgeBaseSpec
from ..embeddings import Embedder, cosine
from ..errors import ConfigError
from .chunking import split_text
from .loaders import extract_text, guess_title, iter_files


@dataclass(frozen=True)
class Hit:
    """Um trecho recuperado da base de conhecimento."""

    chunk_id: str
    kb: str
    document: str
    content: str
    score: float
    section: str | None = None


class Retriever(Protocol):
    async def search(
        self,
        query: str,
        knowledge_bases: Sequence[str],
        *,
        top_k: int,
        min_score: float,
    ) -> list[Hit]: ...


def index_text(title: str, section: str | None, content: str) -> str:
    """Texto que efetivamente vira vetor: título e seção ajudam a busca."""
    header = f"{title} — {section}" if section else title
    return f"{header}\n{content}"


@dataclass
class _MemoryChunk:
    chunk_id: str
    document: str
    section: str | None
    content: str
    vector: list[float]


@dataclass
class _MemoryKB:
    spec: KnowledgeBaseSpec
    embedder: Embedder
    chunks: list[_MemoryChunk] = field(default_factory=list)


class MemoryRetriever:
    """Bases de conhecimento em memória, carregadas de arquivos (modo YAML)."""

    def __init__(self) -> None:
        self._kbs: dict[str, _MemoryKB] = {}

    def register(self, spec: KnowledgeBaseSpec, embedder: Embedder) -> None:
        self._kbs[spec.name] = _MemoryKB(spec, embedder)

    async def add_text(self, kb: str, title: str, text: str) -> int:
        entry = self._kbs.get(kb)
        if entry is None:
            raise ConfigError(f"base '{kb}' não registrada")
        pieces = split_text(text, entry.spec.chunk_size, entry.spec.chunk_overlap)
        if not pieces:
            return 0
        vectors = await entry.embedder.embed(
            [index_text(title, p.section, p.content) for p in pieces]
        )
        for i, (piece, vector) in enumerate(zip(pieces, vectors, strict=True)):
            digest = hashlib.sha1(f"{kb}/{title}/{i}".encode()).hexdigest()[:12]
            entry.chunks.append(_MemoryChunk(digest, title, piece.section, piece.content, vector))
        return len(pieces)

    async def load_paths(self, kb: str, base_dir: Path | None = None) -> int:
        entry = self._kbs.get(kb)
        if entry is None:
            raise ConfigError(f"base '{kb}' não registrada")

        def read_all() -> list[tuple[str, str]]:
            docs = []
            for path in iter_files(entry.spec.paths, base_dir):
                text = extract_text(path.name, path.read_bytes())
                docs.append((guess_title(path.name, text), text))
            return docs

        total = 0
        for title, text in await anyio.to_thread.run_sync(read_all):
            total += await self.add_text(kb, title, text)
        return total

    def stats(self) -> dict[str, int]:
        return {name: len(kb.chunks) for name, kb in self._kbs.items()}

    async def search(
        self,
        query: str,
        knowledge_bases: Sequence[str],
        *,
        top_k: int,
        min_score: float,
    ) -> list[Hit]:
        hits: list[Hit] = []
        for name in knowledge_bases:
            entry = self._kbs.get(name)
            if entry is None or not entry.chunks:
                continue
            [qvec] = await entry.embedder.embed([query])
            for chunk in entry.chunks:
                score = cosine(qvec, chunk.vector)
                if score >= min_score:
                    hits.append(
                        Hit(
                            chunk.chunk_id,
                            name,
                            chunk.document,
                            chunk.content,
                            score,
                            chunk.section,
                        )
                    )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    async def aclose(self) -> None:
        for entry in self._kbs.values():
            await entry.embedder.aclose()
