from __future__ import annotations

import os
import uuid
from typing import Annotated, Literal

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field
from sqlalchemy import create_engine, text

from switchboard.agents import AgentCatalog
from switchboard.storage import Database
from switchboard.testing import inproc_connector

# --------------------------------------------------------------------------
# agentes MCP em processo (sem rede)


def make_calc_server() -> MCPServer:
    server = MCPServer("calc", instructions="Calculadora de testes.")

    @server.tool(structured_output=False)
    def somar(
        a: Annotated[float, Field(description="Primeira parcela da soma")],
        b: Annotated[float, Field(description="Segunda parcela da soma")],
    ) -> str:
        """Soma dois números."""
        return f"resultado: {a + b:g}"

    @server.tool(structured_output=False)
    def converter_moeda(
        valor: Annotated[float, Field(description="Valor em reais")],
        moeda: Annotated[Literal["usd", "eur"], Field(description="Moeda de destino")] = "usd",
    ) -> str:
        """Converte um valor em reais para dólar ou euro."""
        taxa = {"usd": 5.0, "eur": 6.0}[moeda]
        return f"{valor / taxa:.2f} {moeda}"

    @server.tool(structured_output=False)
    def falhar(motivo: str) -> str:
        """Sempre falha (para testar erro de tool)."""
        raise ToolError(f"falhei: {motivo}")

    return server


def make_echo_server() -> MCPServer:
    server = MCPServer("echo", instructions="Repete mensagens.")

    @server.tool(structured_output=False)
    def repetir(mensagem: str) -> str:
        """Repete a mensagem recebida."""
        return mensagem

    return server


@pytest.fixture
def servers() -> dict[str, MCPServer]:
    return {"calc": make_calc_server(), "echo": make_echo_server()}


@pytest.fixture
def catalog(servers) -> AgentCatalog:
    return AgentCatalog(connector=inproc_connector(servers))


# --------------------------------------------------------------------------
# bancos: SQLite sempre; PostgreSQL quando SWITCHBOARD_TEST_DATABASE_URL existir

PG_URL = os.environ.get("SWITCHBOARD_TEST_DATABASE_URL")


def _pg_schema_url(base: str) -> tuple[str, str]:
    schema = "t_" + uuid.uuid4().hex[:10]
    sep = "&" if "?" in base else "?"
    return schema, f"{base}{sep}options=-csearch_path%3D{schema},public"


@pytest.fixture(params=["sqlite", "postgres"])
def db(request, tmp_path):
    if request.param == "sqlite":
        database = Database(f"sqlite:///{tmp_path / 'switchboard.db'}")
        database.init()
        yield database
        database.dispose()
        return
    if not PG_URL:
        pytest.skip("defina SWITCHBOARD_TEST_DATABASE_URL para testar no PostgreSQL")
    schema, url = _pg_schema_url(PG_URL)
    admin = create_engine(Database(PG_URL).url)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    database = Database(url)
    database.init()
    try:
        yield database
    finally:
        database.dispose()
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()
