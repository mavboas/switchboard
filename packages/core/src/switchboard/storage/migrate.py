"""Migrações do schema (aditivas e idempotentes), aplicadas na subida.

O MVP (v0.1) criava as tabelas com ``create_all`` e não tinha migração. A
partir da v0.2:

1. ``create_all`` cria as tabelas novas;
2. colunas que faltam em tabelas existentes são adicionadas (sempre
   anuláveis; os valores padrão são preenchidos com ``UPDATE``);
3. dados legados são movidos: no MVP, "agentes" eram servidores MCP (tabela
   ``agents``); eles viram **conectores MCP** (``mcp_connectors``), mantendo os
   vínculos com os roteadores, e as tabelas antigas são removidas; as execuções
   antigas com rota ``delegated`` (tool MCP na v0.1) passam a ``tool``;
4. a versão do schema fica em ``switchboard_meta``.

Tudo roda na mesma transação do ``Database.init`` (no PostgreSQL, sob o
advisory lock que já protege a subida simultânea de console e router).
"""

from __future__ import annotations

import logging
import warnings

from sqlalchemy import Connection, inspect, text
from sqlalchemy.exc import SAWarning

from .orm import Base

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2

# valores padrão para as colunas novas em linhas que já existiam
_BACKFILL = {
    ("traces", "status"): "'completed'",
    ("router_profiles", "decision_threshold"): "0.6",
    ("router_profiles", "wait_s"): "8.0",
    ("router_profiles", "max_parallel"): "3",
    ("router_profiles", "deadline_s"): "600.0",
}

_LEGACY_CONNECTOR_COLUMNS = (
    "name",
    "description",
    "url",
    "transport",
    "auth_token",
    "allowed_tools",
    "enabled",
    "timeout_s",
    "created_at",
    "updated_at",
)


def _columns(inspector, table: str) -> set[str]:
    with warnings.catch_warnings():  # o inspector não conhece o tipo "vector" do pgvector
        warnings.filterwarnings("ignore", "Did not recognize type", SAWarning)
        return {c["name"] for c in inspector.get_columns(table)}


def _add_missing_columns(conn: Connection) -> list[str]:
    inspector = inspect(conn)
    existing_tables = set(inspector.get_table_names())
    added = []
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        existing = _columns(inspector, table.name)
        for column in table.columns:
            if column.name in existing:
                continue
            ddl_type = column.type.compile(dialect=conn.dialect)
            conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{column.name}" {ddl_type}'))
            default = _BACKFILL.get((table.name, column.name))
            if default is not None:
                conn.execute(
                    text(
                        f'UPDATE {table.name} SET "{column.name}" = {default} WHERE "{column.name}" IS NULL'
                    )
                )
            added.append(f"{table.name}.{column.name}")
        if added:
            for index in table.indexes:  # índices das colunas novas
                index.create(conn, checkfirst=True)
    return added


def _migrate_legacy_agents(conn: Connection) -> int:
    """MVP: ``agents`` (servidores MCP) → ``mcp_connectors``; vínculos junto."""
    inspector = inspect(conn)
    tables = set(inspector.get_table_names())
    if "agents" not in tables:
        return 0
    columns = _columns(inspector, "agents")
    if "transport" not in columns:  # não é a tabela legada
        return 0
    # cópia feita pelo próprio banco (tipos JSON/boolean passam sem conversão)
    cols = ", ".join(c for c in _LEGACY_CONNECTOR_COLUMNS if c in columns)
    moved = conn.execute(
        text(
            f"INSERT INTO mcp_connectors ({cols}) SELECT {cols} FROM agents a "
            "WHERE NOT EXISTS (SELECT 1 FROM mcp_connectors c WHERE c.name = a.name)"
        )
    ).rowcount
    if "profile_agents" in tables:
        conn.execute(
            text(
                "INSERT INTO profile_mcp_connectors (profile_id, connector_id) "
                "SELECT pa.profile_id, c.id FROM profile_agents pa "
                "JOIN agents a ON a.id = pa.agent_id "
                "JOIN mcp_connectors c ON c.name = a.name "
                "WHERE NOT EXISTS (SELECT 1 FROM profile_mcp_connectors x "
                "WHERE x.profile_id = pa.profile_id AND x.connector_id = c.id)"
            )
        )
        conn.execute(text("DROP TABLE profile_agents"))
    conn.execute(text("DROP TABLE agents"))
    return max(moved or 0, 0)


def upgrade(conn: Connection) -> None:
    """Deixa o schema na versão atual (chame depois do ``create_all``)."""
    added = _add_missing_columns(conn)
    moved = _migrate_legacy_agents(conn)
    current = conn.execute(
        text("SELECT value FROM switchboard_meta WHERE key = 'schema_version'")
    ).scalar()
    if current is None or int(current) < 2:
        # na v0.1, "delegated" era uma chamada de tool MCP; na v0.2 essa rota é
        # "tool" e "delegated" passou a ser delegação a agente A2A
        conn.execute(text("UPDATE traces SET route = 'tool' WHERE route = 'delegated'"))
    if current is None:
        conn.execute(
            text("INSERT INTO switchboard_meta (key, value) VALUES ('schema_version', :v)"),
            {"v": str(SCHEMA_VERSION)},
        )
    elif int(current) < SCHEMA_VERSION:
        conn.execute(
            text("UPDATE switchboard_meta SET value = :v WHERE key = 'schema_version'"),
            {"v": str(SCHEMA_VERSION)},
        )
    if added or moved:
        log.info(
            "schema atualizado para a v%s (colunas novas: %s; agentes MCP migrados para conectores: %s)",
            SCHEMA_VERSION,
            ", ".join(added) or "nenhuma",
            moved,
        )
