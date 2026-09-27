"""RAG sobre o banco: indexação de documentos e busca vetorial.

No PostgreSQL com pgvector a similaridade de cosseno é calculada pelo banco
(operador ``<=>``); sem pgvector (ou no SQLite) os vetores são lidos e
comparados na aplicação.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import anyio.to_thread
from sqlalchemy import bindparam, delete, select, text

from ..config import EmbedderSpec, ModelSpec
from ..embeddings import Embedder, build_embedder, cosine
from ..errors import IngestError
from ..rag.chunking import split_text
from ..rag.retriever import Hit, index_text
from ..secrets import resolve_env
from .db import Database
from .orm import Chunk, Document, KnowledgeBase
from .repo import embedder_of

SecretResolver = Callable[[str | None], str | None]


class EmbedderCache:
    """Reaproveita embedders (e suas conexões HTTP) entre chamadas."""

    def __init__(self, resolve_secret: SecretResolver = resolve_env):
        self._resolve = resolve_secret
        self._items: dict[str, Embedder] = {}

    def get(self, spec: EmbedderSpec, connection: ModelSpec | None) -> Embedder:
        key = json.dumps(
            [
                spec.model_dump(mode="json"),
                connection.model_dump(mode="json") if connection else None,
            ],
            sort_keys=True,
        )
        if key not in self._items:
            self._items[key] = build_embedder(spec, connection, resolve_secret=self._resolve)
        return self._items[key]

    async def aclose(self) -> None:
        for embedder in self._items.values():
            await embedder.aclose()
        self._items.clear()


@dataclass
class _KBInfo:
    id: int
    name: str
    embedder: EmbedderSpec
    connection: ModelSpec | None


def _content_hash(text_value: str) -> str:
    return hashlib.sha256(text_value.strip().encode("utf-8")).hexdigest()


class KnowledgeService:
    def __init__(self, db: Database, embedders: EmbedderCache | None = None):
        self.db = db
        self.embedders = embedders or EmbedderCache()

    # -- leitura ---------------------------------------------------------------

    def _kb_infos(self, names: Sequence[str]) -> list[_KBInfo]:
        with self.db.session() as session:
            rows = session.scalars(
                select(KnowledgeBase).where(KnowledgeBase.name.in_(list(names)))
            ).all()
            infos = []
            for row in rows:
                embedder, connection = embedder_of(row)
                infos.append(_KBInfo(row.id, row.name, embedder, connection))
            return infos

    def embedder_for(self, kb_id: int) -> tuple[Embedder, str]:
        with self.db.session() as session:
            row = session.get(KnowledgeBase, kb_id)
            if row is None:
                raise IngestError("base de conhecimento não encontrada")
            spec, connection = embedder_of(row)
        return self.embedders.get(spec, connection), spec.label

    def _search_sql(self, kb_ids: list[int], vector: list[float], top_k: int) -> list[tuple]:
        if self.db.pgvector:
            literal = "[" + ",".join(repr(float(x)) for x in vector) + "]"
            sql = text(
                """
                SELECT c.id, kb.name, d.title, c.content, c.section,
                       1 - (c.embedding <=> CAST(:q AS vector)) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                JOIN knowledge_bases kb ON kb.id = c.kb_id
                WHERE c.kb_id IN :kb_ids AND c.embedding_dim = :dim
                ORDER BY c.embedding <=> CAST(:q AS vector)
                LIMIT :k
                """
            ).bindparams(bindparam("kb_ids", expanding=True))
            with self.db.engine.connect() as conn:
                rows = conn.execute(
                    sql, {"q": literal, "kb_ids": kb_ids, "dim": len(vector), "k": top_k}
                ).all()
            return [tuple(r) for r in rows]
        # sem pgvector: similaridade na aplicação
        with self.db.session() as session:
            rows = session.execute(
                select(
                    Chunk.id,
                    KnowledgeBase.name,
                    Document.title,
                    Chunk.content,
                    Chunk.section,
                    Chunk.embedding,
                )
                .join(Document, Document.id == Chunk.document_id)
                .join(KnowledgeBase, KnowledgeBase.id == Chunk.kb_id)
                .where(Chunk.kb_id.in_(kb_ids), Chunk.embedding_dim == len(vector))
            ).all()
        scored = [(r[0], r[1], r[2], r[3], r[4], cosine(vector, r[5])) for r in rows]
        scored.sort(key=lambda r: r[5], reverse=True)
        return scored[:top_k]

    async def search(
        self,
        query: str,
        knowledge_bases: Sequence[str],
        *,
        top_k: int,
        min_score: float,
    ) -> list[Hit]:
        if not query.strip() or not knowledge_bases:
            return []
        infos = await anyio.to_thread.run_sync(self._kb_infos, knowledge_bases)
        groups: dict[str, list[_KBInfo]] = {}
        for info in infos:
            groups.setdefault(info.embedder.label + "|" + str(info.connection), []).append(info)
        hits: list[Hit] = []
        for group in groups.values():
            embedder = self.embedders.get(group[0].embedder, group[0].connection)
            [vector] = await embedder.embed([query])
            rows = await anyio.to_thread.run_sync(
                self._search_sql, [g.id for g in group], vector, top_k
            )
            for chunk_id, kb_name, title, content, section, score in rows:
                if score is not None and float(score) >= min_score:
                    hits.append(Hit(str(chunk_id), kb_name, title, content, float(score), section))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    # -- escrita ---------------------------------------------------------------

    async def add_document(
        self, kb_id: int, *, title: str, content: str, source: str = "texto"
    ) -> tuple[int, bool]:
        """Indexa um documento; devolve (id, criado?). Conteúdo repetido é ignorado."""
        title = title.strip() or "sem título"
        content = content.strip()
        if not content:
            raise IngestError("o documento está vazio")
        digest = _content_hash(content)

        def prepare():
            with self.db.session() as session:
                kb = session.get(KnowledgeBase, kb_id)
                if kb is None:
                    raise IngestError("base de conhecimento não encontrada")
                existing = session.scalar(
                    select(Document.id).where(
                        Document.kb_id == kb_id, Document.content_hash == digest
                    )
                )
                return kb.chunk_size, kb.chunk_overlap, existing

        chunk_size, overlap, existing = await anyio.to_thread.run_sync(prepare)
        if existing is not None:
            return int(existing), False
        pieces = split_text(content, chunk_size, overlap)
        embedder, label = await anyio.to_thread.run_sync(self.embedder_for, kb_id)
        vectors = (
            await embedder.embed([index_text(title, p.section, p.content) for p in pieces])
            if pieces
            else []
        )

        def store() -> int:
            with self.db.session() as session:
                doc = Document(
                    kb_id=kb_id,
                    title=title[:300],
                    source=source[:300],
                    content=content,
                    content_hash=digest,
                    chunk_count=len(pieces),
                    embedder=label,
                )
                session.add(doc)
                session.flush()
                for i, (piece, vector) in enumerate(zip(pieces, vectors, strict=True)):
                    session.add(
                        Chunk(
                            kb_id=kb_id,
                            document_id=doc.id,
                            ordinal=i,
                            section=(piece.section or None) and piece.section[:500],
                            content=piece.content,
                            embedding=vector,
                            embedding_dim=len(vector),
                        )
                    )
                return doc.id

        doc_id = await anyio.to_thread.run_sync(store)
        return doc_id, True

    async def delete_document(self, kb_id: int, document_id: int) -> None:
        def run():
            with self.db.session() as session:
                doc = session.get(Document, document_id)
                if doc is None or doc.kb_id != kb_id:
                    raise IngestError("documento não encontrado")
                session.delete(doc)

        await anyio.to_thread.run_sync(run)

    async def reindex(self, kb_id: int) -> int:
        """Recalcula trechos e vetores de todos os documentos (após trocar o embedder)."""

        def load():
            with self.db.session() as session:
                kb = session.get(KnowledgeBase, kb_id)
                if kb is None:
                    raise IngestError("base de conhecimento não encontrada")
                docs = session.scalars(select(Document).where(Document.kb_id == kb_id)).all()
                return kb.chunk_size, kb.chunk_overlap, [(d.id, d.title, d.content) for d in docs]

        chunk_size, overlap, docs = await anyio.to_thread.run_sync(load)
        embedder, label = await anyio.to_thread.run_sync(self.embedder_for, kb_id)
        total = 0
        for doc_id, title, content in docs:
            pieces = split_text(content, chunk_size, overlap)
            vectors = (
                await embedder.embed([index_text(title, p.section, p.content) for p in pieces])
                if pieces
                else []
            )

            def store(doc_id=doc_id, pieces=pieces, vectors=vectors):
                with self.db.session() as session:
                    session.execute(delete(Chunk).where(Chunk.document_id == doc_id))
                    for i, (piece, vector) in enumerate(zip(pieces, vectors, strict=True)):
                        session.add(
                            Chunk(
                                kb_id=kb_id,
                                document_id=doc_id,
                                ordinal=i,
                                section=(piece.section or None) and piece.section[:500],
                                content=piece.content,
                                embedding=vector,
                                embedding_dim=len(vector),
                            )
                        )
                    doc = session.get(Document, doc_id)
                    if doc is not None:
                        doc.chunk_count = len(pieces)
                        doc.embedder = label

            await anyio.to_thread.run_sync(store)
            total += len(pieces)
        return total

    async def aclose(self) -> None:
        await self.embedders.aclose()
