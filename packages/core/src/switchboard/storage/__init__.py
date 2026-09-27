"""Persistência plugável: PostgreSQL (pgvector) por padrão, SQLite opcional."""

from .db import Database, normalize_url, redact_url
from .knowledge import EmbedderCache, KnowledgeService
from .orm import Agent, Base, Chunk, Document, KnowledgeBase, LlmModel, RouterProfile, Trace
from .seed import seed_demo

__all__ = [
    "Agent",
    "Base",
    "Chunk",
    "Database",
    "Document",
    "EmbedderCache",
    "KnowledgeBase",
    "KnowledgeService",
    "LlmModel",
    "RouterProfile",
    "Trace",
    "normalize_url",
    "redact_url",
    "seed_demo",
]
