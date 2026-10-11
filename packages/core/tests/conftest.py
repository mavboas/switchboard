from __future__ import annotations

import os
import uuid
from typing import Annotated, Literal

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field
from sqlalchemy import create_engine, text

from switchboard.connectors import ConnectorCatalog
from switchboard.plugins import pack
from switchboard.storage import Database
from switchboard.testing import FakeA2AAgent, FakeNetwork, inproc_connector

# --------------------------------------------------------------------------
# conectores MCP em processo (sem rede)


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
def catalog(servers) -> ConnectorCatalog:
    return ConnectorCatalog(connector=inproc_connector(servers))


@pytest.fixture
def risk_agent() -> FakeA2AAgent:
    return FakeA2AAgent()


@pytest.fixture
def network(risk_agent) -> FakeNetwork:
    return FakeNetwork(risk_agent)


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


@pytest.fixture(params=["sqlite", "postgres"])
def empty_db_url(request, tmp_path):
    """URL de um banco vazio (sem init), para testar migrações."""
    if request.param == "sqlite":
        yield f"sqlite:///{tmp_path / 'legado.db'}"
        return
    if not PG_URL:
        pytest.skip("defina SWITCHBOARD_TEST_DATABASE_URL para testar no PostgreSQL")
    schema, url = _pg_schema_url(PG_URL)
    admin = create_engine(Database(PG_URL).url)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    try:
        yield url
    finally:
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


# --------------------------------------------------------------------------
# pacotes de plugin em memória

PLUGIN_MANIFEST = """\
name: __NAME__
version: __VERSION__
description: Plugin de teste com um conector, um agente e uma base de conhecimento.
config:
  CALC_URL:
    description: Endereço do servidor MCP de cálculo
  RISCO_URL:
    description: Endereço do agente A2A de risco
  RISCO_TOKEN:
    description: Token do agente de risco
    sensitive: true
    required: false
connectors:
  - name: calc
    description: Calculadora de testes.
    url: ${config.CALC_URL}
    allowed_tools: [somar]
agents:
  - name: risco
    description: Avalia o risco de fraude de uma operação.
    url: ${config.RISCO_URL}
    auth_token: ${config.RISCO_TOKEN}
knowledge_bases:
  - name: faq
    description: Perguntas frequentes do plugin.
playbooks:
  - skill: usar-calculo
    applies_to: [calc]
    triggers: [somar, soma]
routing_examples:
  calc/somar:
    - Quanto é 2 mais 3?
"""

PLUGIN_SKILL = """\
---
name: usar-calculo
description: Como usar a calculadora para somas e contas simples.
---

# Cálculo

Use `calc/somar` quando o usuário pedir uma soma. Confirme os números antes de somar.
"""

PLUGIN_DOC = "# FAQ\n\nO prazo de entrega do pedido é de 5 dias úteis.\n"


def make_plugin_files(
    *,
    name: str = "acme-teste",
    version: str = "1.0.0",
    manifest: str | None = None,
    skill: str | None = None,
    doc: str | None = None,
    extra: dict[str, str | bytes] | None = None,
    remove: tuple[str, ...] = (),
) -> dict[str, bytes]:
    """Os arquivos de um plugin válido (sem achados); ``extra``/``remove`` ajustam o pacote."""
    text = (manifest if manifest is not None else PLUGIN_MANIFEST).replace("__NAME__", name)
    files: dict[str, str | bytes] = {
        "plugin.yaml": text.replace("__VERSION__", version),
        "skills/usar-calculo/SKILL.md": skill if skill is not None else PLUGIN_SKILL,
        "knowledge/faq.md": doc if doc is not None else PLUGIN_DOC,
    }
    files.update(extra or {})
    for path in remove:
        files.pop(path, None)
    return {p: v.encode("utf-8") if isinstance(v, str) else v for p, v in files.items()}


@pytest.fixture
def plugin_files():
    """Fábrica dos arquivos de um plugin de teste (``plugin_files(version="2.0.0", ...)``)."""
    return make_plugin_files


@pytest.fixture
def plugin_zip():
    """Fábrica dos bytes do ``.zip`` de um plugin de teste."""
    return lambda **kw: pack(make_plugin_files(**kw))
