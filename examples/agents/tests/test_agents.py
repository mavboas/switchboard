from __future__ import annotations

import pytest
from mcp import Client

from switchboard_agents import chamados, credito


def test_amortization_math():
    p = credito.price(50_000, 1.5, 24)
    assert round(p["primeira"], 2) == 2496.21 and round(p["juros"], 2) == 9908.92
    s = credito.sac(100_000, 2, 36)
    assert round(s["primeira"], 2) == 4777.78
    assert round(s["ultima"], 2) == 2833.33
    assert round(s["juros"], 2) == 37000.00
    assert credito.brl(1234567.891) == "R$ 1.234.567,89"


async def test_credito_tools_over_mcp():
    async with Client(credito.server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert names == {"simular_financiamento", "comparar_sistemas"}
        ok = await client.call_tool(
            "simular_financiamento",
            {"valor": 10000, "taxa_mensal_percentual": 1, "prazo_meses": 12, "sistema": "sac"},
        )
        assert not ok.is_error and "SAC" in ok.content[0].text
        bad = await client.call_tool(
            "simular_financiamento", {"valor": 0, "taxa_mensal_percentual": 1, "prazo_meses": 12}
        )
        assert bad.is_error and "valor" in bad.content[0].text


@pytest.fixture(autouse=True)
def _reset():
    chamados.reset()


async def test_chamados_lifecycle_over_mcp():
    async with Client(chamados.server) as client:
        empty = await client.call_tool("listar_chamados", {})
        assert "Nenhum chamado" in empty.content[0].text
        opened = await client.call_tool(
            "abrir_chamado",
            {"titulo": "Erro no login", "descricao": "App fecha", "prioridade": "alta"},
        )
        assert "CH-1001" in opened.content[0].text and "4 horas úteis" in opened.content[0].text
        status = await client.call_tool("consultar_chamado", {"protocolo": "1001"})
        assert "está aberto" in status.content[0].text
        missing = await client.call_tool("consultar_chamado", {"protocolo": "CH-9"})
        assert missing.is_error and "não encontrado" in missing.content[0].text
