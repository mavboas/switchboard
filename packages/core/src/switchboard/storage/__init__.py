"""Persistência plugável: PostgreSQL (pgvector) por padrão, SQLite opcional."""

from .contracts import SqlContractStore
from .db import Database, normalize_url, redact_url
from .knowledge import EmbedderCache, KnowledgeService
from .orm import (
    Agent,
    Base,
    Chunk,
    Connector,
    Contract,
    ContractEventRow,
    Document,
    KnowledgeBase,
    LlmModel,
    RouterProfile,
    SpanRow,
    Trace,
)
from .seed import seed_demo

__all__ = [
    "Agent",
    "Base",
    "Chunk",
    "Connector",
    "Contract",
    "ContractEventRow",
    "Database",
    "Document",
    "EmbedderCache",
    "KnowledgeBase",
    "KnowledgeService",
    "LlmModel",
    "RouterProfile",
    "SpanRow",
    "SqlContractStore",
    "Trace",
    "normalize_url",
    "redact_url",
    "seed_demo",
]
