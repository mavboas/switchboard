"""Tabelas do Switchboard (SQLAlchemy 2)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    cast,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator, UserDefinedType


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class _PgVector(UserDefinedType):
    """Coluna ``vector`` do pgvector sem dimensão fixa (cada base pode ter a sua)."""

    cache_ok = True

    def get_col_spec(self, **_kw: Any) -> str:
        return "VECTOR"

    def bind_expression(self, bindvalue):
        return cast(bindvalue, self)

    def bind_processor(self, _dialect):
        def process(value):
            if value is None:
                return None
            return "[" + ",".join(repr(float(x)) for x in value) + "]"

        return process

    def result_processor(self, _dialect, _coltype):
        def process(value):
            if value is None:
                return None
            if isinstance(value, str):
                return [float(x) for x in value.strip("[]").split(",") if x]
            return [float(x) for x in value]

        return process


class EmbeddingType(TypeDecorator):
    """pgvector quando disponível; JSON nos demais casos (ver Database.init)."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if getattr(dialect, "_switchboard_pgvector", False):
            return dialect.type_descriptor(_PgVector())
        return dialect.type_descriptor(JSON())


class Timestamped:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


profile_agents = Table(
    "profile_agents",
    Base.metadata,
    Column("profile_id", ForeignKey("router_profiles.id", ondelete="CASCADE"), primary_key=True),
    Column("agent_id", ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True),
)

profile_knowledge_bases = Table(
    "profile_knowledge_bases",
    Base.metadata,
    Column("profile_id", ForeignKey("router_profiles.id", ondelete="CASCADE"), primary_key=True),
    Column(
        "knowledge_base_id", ForeignKey("knowledge_bases.id", ondelete="CASCADE"), primary_key=True
    ),
)


class LlmModel(Timestamped, Base):
    __tablename__ = "llm_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(63), unique=True)
    preset: Mapped[str | None] = mapped_column(String(40))
    provider: Mapped[str] = mapped_column(String(20))
    model: Mapped[str] = mapped_column(String(200), default="")
    base_url: Mapped[str | None] = mapped_column(String(500))
    api_key: Mapped[str | None] = mapped_column(Text)
    api_key_header: Mapped[str] = mapped_column(String(100), default="Authorization")
    extra_headers: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    temperature: Mapped[float | None] = mapped_column(Float, default=0.2)
    max_tokens: Mapped[int | None] = mapped_column(Integer, default=1024)
    timeout_s: Mapped[float] = mapped_column(Float, default=60.0)
    json_mode: Mapped[bool] = mapped_column(Boolean, default=True)


class Agent(Timestamped, Base):
    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(63), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    url: Mapped[str] = mapped_column(String(500))
    transport: Mapped[str] = mapped_column(String(20), default="streamable-http")
    auth_token: Mapped[str | None] = mapped_column(Text)
    allowed_tools: Mapped[list[str]] = mapped_column(JSON, default=list)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    timeout_s: Mapped[float] = mapped_column(Float, default=30.0)


class KnowledgeBase(Timestamped, Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(63), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    embedder_kind: Mapped[str] = mapped_column(String(20), default="hashing")
    embedder_dim: Mapped[int] = mapped_column(Integer, default=512)
    embedding_model_id: Mapped[int | None] = mapped_column(
        ForeignKey("llm_models.id", ondelete="SET NULL")
    )
    embedding_model: Mapped[str | None] = mapped_column(String(200))
    chunk_size: Mapped[int] = mapped_column(Integer, default=800)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=120)

    embedding_connection: Mapped[LlmModel | None] = relationship(lazy="joined")
    documents: Mapped[list[Document]] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Document.created_at.desc()",
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("kb_id", "content_hash", name="uq_documents_kb_hash"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    kb_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(300))
    source: Mapped[str] = mapped_column(String(300), default="texto")
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embedder: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="documents")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    kb_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(EmbeddingType)
    embedding_dim: Mapped[int] = mapped_column(Integer)


class RouterProfile(Timestamped, Base):
    __tablename__ = "router_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(63), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    model_id: Mapped[int] = mapped_column(ForeignKey("llm_models.id", ondelete="RESTRICT"))
    system_prompt: Mapped[str] = mapped_column(Text)
    top_k: Mapped[int] = mapped_column(Integer, default=4)
    min_score: Mapped[float] = mapped_column(Float, default=0.2)
    synthesize: Mapped[bool] = mapped_column(Boolean, default=True)
    allow_clarify: Mapped[bool] = mapped_column(Boolean, default=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    model: Mapped[LlmModel] = relationship(lazy="joined")
    agents: Mapped[list[Agent]] = relationship(secondary=profile_agents, order_by=Agent.name)
    knowledge_bases: Mapped[list[KnowledgeBase]] = relationship(
        secondary=profile_knowledge_bases, order_by=KnowledgeBase.name
    )


class Trace(Base):
    __tablename__ = "traces"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    profile: Mapped[str] = mapped_column(String(63), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    route: Mapped[str] = mapped_column(String(20), index=True)
    model: Mapped[str] = mapped_column(String(250), default="")
    agent: Mapped[str | None] = mapped_column(String(63))
    tool: Mapped[str | None] = mapped_column(String(200))
    arguments: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    reason: Mapped[str] = mapped_column(Text, default="")
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    steps: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
