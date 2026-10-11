"""Conexão com o banco escolhido (PostgreSQL por padrão, SQLite opcional).

A URL segue o padrão do SQLAlchemy; ``postgresql://`` e ``postgres://`` são
convertidas para o driver psycopg 3. No PostgreSQL, os vetores do RAG usam a
extensão pgvector quando ela existe (busca feita pelo próprio banco); sem ela,
e no SQLite, os vetores ficam em JSON e a similaridade é calculada na
aplicação — funciona igual, só escala menos.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..errors import ConfigError
from .migrate import upgrade
from .orm import Base

log = logging.getLogger(__name__)

VectorBackend = Literal["auto", "pgvector", "json"]
_INIT_LOCK_KEY = 7_042_026


def normalize_url(url: str) -> str:
    url = url.strip()
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def redact_url(url: str) -> str:
    """URL sem a senha, para logs e para a UI."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.rsplit("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


class Database:
    def __init__(self, url: str, *, vector_backend: VectorBackend = "auto", echo: bool = False):
        self.url = normalize_url(url)
        self.vector_backend = vector_backend
        self.pgvector = False
        kwargs: dict = {"echo": echo, "future": True}
        if self.url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if self.url in ("sqlite://", "sqlite:///:memory:"):
                kwargs["poolclass"] = StaticPool
        else:
            kwargs["pool_pre_ping"] = True
            if self.url.startswith("postgresql+psycopg"):
                # garante texto como str mesmo em bancos criados com SQL_ASCII
                kwargs["connect_args"] = {"client_encoding": "utf8"}
        self.engine = create_engine(self.url, **kwargs)
        if self.dialect == "sqlite":

            @event.listens_for(self.engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                # console e router podem abrir o mesmo arquivo: WAL + espera evitam "database is locked"
                cursor.execute("PRAGMA busy_timeout=5000")
                if self.url not in ("sqlite://", "sqlite:///:memory:"):
                    cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()

        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    @property
    def dialect(self) -> str:
        return self.engine.dialect.name

    @property
    def safe_url(self) -> str:
        return redact_url(self.url)

    @property
    def vector_mode(self) -> str:
        return "pgvector" if self.pgvector else "json"

    def init(self) -> None:
        """Cria extensão e tabelas (idempotente e seguro com vários processos)."""
        if self.dialect == "postgresql":
            self._init_postgres()
        else:
            if self.vector_backend == "pgvector":
                raise ConfigError("vector_backend=pgvector exige PostgreSQL")
            self.pgvector = False
            self.engine.dialect._switchboard_pgvector = False  # type: ignore[attr-defined]
            with self.engine.begin() as conn:
                Base.metadata.create_all(conn)
                upgrade(conn)

    def _init_postgres(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _INIT_LOCK_KEY})
            has_extension = False
            if self.vector_backend in ("auto", "pgvector"):
                try:
                    with conn.begin_nested():
                        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    has_extension = True
                except DBAPIError as exc:
                    if self.vector_backend == "pgvector":
                        raise ConfigError(
                            f"não consegui habilitar a extensão pgvector: {exc}"
                        ) from exc
                    log.warning(
                        "pgvector indisponível; vetores ficarão em JSON (%s)",
                        exc.__class__.__name__,
                    )
            existing = conn.execute(
                text(
                    "SELECT udt_name FROM information_schema.columns "
                    "WHERE table_schema = current_schema() AND table_name = 'chunks' AND column_name = 'embedding'"
                )
            ).scalar()
            self.pgvector = (existing == "vector") if existing else has_extension
            self.engine.dialect._switchboard_pgvector = self.pgvector  # type: ignore[attr-defined]
            Base.metadata.create_all(conn)
            upgrade(conn)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def dispose(self) -> None:
        self.engine.dispose()
