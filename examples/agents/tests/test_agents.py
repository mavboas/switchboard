from __future__ import annotations

import pytest
from google.protobuf.json_format import MessageToDict
from mcp import Client

from switchboard.a2a import AgentCard
from switchboard.contracts import (
    CONTRACT_EXTENSION_URI,
    schema_hash,
    skills_from_extension,
    validate,
)
from switchboard_agents import analise, chamados, credito, risco


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


# --------------------------------------------------------------------------
# agentes A2A (lógica de negócio e termos de contrato publicados no card)


def _analise(valor, prazo, renda, **kw):
    dados = {"cliente": "Maria Souza", "valor": valor, "prazo_meses": prazo, "renda_mensal": renda}
    return analise.analisar(dados, **kw)


def test_credit_analysis_policy_and_output_contract():
    ok = _analise(80_000, 24, 12_000, garantia=False)
    assert ok["decisao"] == "aprovado" and ok["limite_aprovado"] == 80_000
    assert ok["garantia"] == "sem garantia" and ok["taxa_mensal_percentual"] == 1.69
    adjusted = _analise(300_000, 24, 10_000, garantia=False)
    assert adjusted["decisao"] == "aprovado_com_ajuste" and adjusted["limite_aprovado"] < 300_000
    assert adjusted["comprometimento_renda_percentual"] <= analise.COMPROMETIMENTO_MAXIMO
    denied = _analise(900_000, 12, 3_000, garantia=False)
    assert denied["decisao"] == "reprovado" and denied["limite_aprovado"] == 0
    secured = _analise(800_000, 120, 60_000, garantia=True)
    assert secured["taxa_mensal_percentual"] == 1.19 and secured["garantia"] == "imóvel"
    for result in (ok, adjusted, denied, secured):
        assert validate(result, analise.OUTPUT_SCHEMA) == []  # o agente cumpre o próprio contrato
    assert (
        analise._sim("Sim, com o apartamento")
        and analise._sim("tem imóvel")
        and analise._sim("não") is False
    )


def test_risk_score_is_deterministic_and_valid():
    base = risco.avaliar({"cliente": "Maria Souza", "valor": 80_000})
    assert base == risco.avaliar({"cliente": " maria souza ", "valor": 80_000.0})
    phone = risco.avaliar({"cliente": "Maria Souza", "valor": 80_000, "canal": "telefone"})
    assert (
        phone["score"] == max(0, base["score"] - 60) and "engenharia social" in phone["sinais"][0]
    )
    big = risco.avaliar({"cliente": "Maria Souza", "valor": 250_000})
    assert "valor acima do padrão do segmento" in big["sinais"]
    for result in (base, phone, big):
        assert validate(result, risco.OUTPUT_SCHEMA) == []


@pytest.mark.parametrize(
    ("module", "skill", "max_duration"),
    [(analise, "analisar_proposta", 900), (risco, "avaliar_risco", 120)],
)
def test_cards_publish_complete_contract_terms(module, skill, max_duration):
    card = AgentCard.parse(MessageToDict(module.build("http://localhost:8201").card()))
    interface = card.jsonrpc_interface()
    assert interface.url == "http://localhost:8201/" and interface.version == "1.0"
    extension = card.extension(CONTRACT_EXTENSION_URI)
    terms, problems = skills_from_extension(extension.get("params"))
    assert problems == {}
    declared = terms[skill]
    assert declared.complete and declared.max_duration_s == max_duration
    assert declared.input_schema["required"] == module.INPUT_SCHEMA["required"]
    assert declared.input_hash == schema_hash(
        module.INPUT_SCHEMA
    )  # números do protobuf não mudam o hash
    assert [s.id for s in card.skills] == [skill]
