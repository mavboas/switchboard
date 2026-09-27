from __future__ import annotations

import pytest

from switchboard.agents import AgentInfo, ToolInfo
from switchboard.routing.args import (
    extract_arguments,
    find_mentions,
    parse_number,
    validate_arguments,
)
from switchboard.routing.decision import (
    DecisionError,
    clarify_for_missing,
    extract_json_object,
    parse_decision,
    validate_decision,
)

LOAN_SCHEMA = {
    "type": "object",
    "properties": {
        "valor": {"type": "number", "description": "Valor financiado em reais"},
        "taxa_mensal_percentual": {
            "type": "number",
            "description": "Taxa de juros mensal em percentual",
        },
        "prazo_meses": {"type": "integer", "description": "Prazo em meses"},
        "sistema": {"type": "string", "enum": ["price", "sac"], "default": "price"},
    },
    "required": ["valor", "taxa_mensal_percentual", "prazo_meses"],
}

TICKET_SCHEMA = {
    "type": "object",
    "properties": {
        "titulo": {"type": "string", "description": "Resumo curto do problema"},
        "descricao": {
            "type": "string",
            "description": "Descrição do problema de crédito ou acesso",
        },
        "prioridade": {"type": "string", "enum": ["baixa", "media", "alta"]},
    },
    "required": ["titulo", "descricao"],
}


@pytest.mark.parametrize(
    "raw,expected",
    [("100.000,50", 100000.5), ("1,5", 1.5), ("2.5", 2.5), ("1.500", 1500), ("36", 36)],
)
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


def test_find_mentions_units():
    kinds = [(m.kind, m.value) for m in find_mentions("R$ 50 mil em 2 anos a 1,5% a.m. e mais 7")]
    assert ("money", 50000) in kinds
    assert ("years", 2) in kinds
    assert ("percent", 1.5) in kinds
    assert ("plain", 7) in kinds


def test_extract_arguments_loan():
    args = extract_arguments(LOAN_SCHEMA, "simular 100 mil em 3 anos, taxa de 2% ao mês, pelo SAC")
    assert args == {
        "valor": 100000,
        "prazo_meses": 36,
        "taxa_mensal_percentual": 2.0,
        "sistema": "sac",
    }


def test_extract_arguments_plain_money_falls_back_to_largest_number():
    args = extract_arguments(LOAN_SCHEMA, "empréstimo de 20000 em 12 meses a 1,9%")
    assert (
        args["valor"] == 20000
        and args["prazo_meses"] == 12
        and args["taxa_mensal_percentual"] == 1.9
    )


def test_extract_arguments_text_fields_are_not_confused_with_money():
    text = "Não consigo acessar o aplicativo, erro no login. Prioridade alta"
    args = extract_arguments(TICKET_SCHEMA, text)
    assert args["prioridade"] == "alta"
    assert args["descricao"] == text
    assert args["titulo"].startswith("Não consigo acessar")


def test_validate_arguments_coerces_and_reports():
    args, errors, missing = validate_arguments(
        LOAN_SCHEMA,
        {"valor": "50.000", "taxa_mensal_percentual": "1,5", "prazo_meses": "24", "sistema": "SAC"},
    )
    assert args == {
        "valor": 50000.0,
        "taxa_mensal_percentual": 1.5,
        "prazo_meses": 24,
        "sistema": "sac",
    }
    assert errors == [] and missing == []
    _, errors, missing = validate_arguments(LOAN_SCHEMA, {"valor": "abc", "sistema": "outro"})
    assert any("valor" in e for e in errors) and any("sistema" in e for e in errors)
    assert missing == ["valor", "taxa_mensal_percentual", "prazo_meses"]


def test_extract_json_object_variants():
    assert extract_json_object('```json\n{"action": "answer"}\n```') == {"action": "answer"}
    assert extract_json_object('Claro! {"a": {"b": 1}} fim') == {"a": {"b": 1}}
    with pytest.raises(DecisionError):
        extract_json_object("sem json aqui")


def test_parse_decision_aliases_and_fields():
    d = parse_decision(
        '{"acao": "delegar", "agente": "calc", "ferramenta": "somar", "arguments": "{\\"a\\": 1}"}'
    )
    assert (d.action, d.agent, d.tool, d.arguments) == ("delegate", "calc", "somar", {"a": 1})
    d = parse_decision('{"action": "answer", "answer": "ok", "sources": ["[2]", 1]}')
    assert d.sources == [2, 1]
    with pytest.raises(DecisionError):
        parse_decision('{"action": "dançar"}')


def _agents():
    loan = ToolInfo("simular", "Simula empréstimo", LOAN_SCHEMA)
    return [AgentInfo("credito", "Crédito", "http://x", "online", tools=[loan])]


def test_validate_decision_delegate_paths():
    agents = _agents()
    ok = parse_decision(
        '{"action": "delegate", "agent": "CREDITO", "tool": "simular", '
        '"arguments": {"valor": 1000, "taxa_mensal_percentual": 1, "prazo_meses": 10}}'
    )
    assert validate_decision(ok, agents, allow_clarify=True, n_sources=0).agent == "credito"
    # agente inferido pelo nome da tool ("agente/tool" também vale)
    inferred = parse_decision(
        '{"action": "delegate", "tool": "credito/simular", '
        '"arguments": {"valor": 1000, "taxa_mensal_percentual": 1, "prazo_meses": 10}}'
    )
    assert validate_decision(inferred, agents, allow_clarify=True, n_sources=0).tool == "simular"
    missing = parse_decision(
        '{"action": "delegate", "agent": "credito", "tool": "simular", "arguments": {"valor": 5}}'
    )
    with pytest.raises(DecisionError) as err:
        validate_decision(missing, agents, allow_clarify=True, n_sources=0)
    assert err.value.missing == ["taxa_mensal_percentual", "prazo_meses"]
    unknown = parse_decision('{"action": "delegate", "agent": "rh", "tool": "ferias"}')
    with pytest.raises(DecisionError, match="opções: credito"):
        validate_decision(unknown, agents, allow_clarify=True, n_sources=0)


def test_validate_decision_answer_and_clarify():
    agents = _agents()
    d = validate_decision(
        parse_decision('{"action":"answer","answer":"x","sources":[1,9]}'),
        agents,
        allow_clarify=True,
        n_sources=2,
    )
    assert d.sources == [1]
    with pytest.raises(DecisionError):
        validate_decision(
            parse_decision('{"action":"clarify","question":"?"}'),
            agents,
            allow_clarify=False,
            n_sources=0,
        )
    with pytest.raises(DecisionError):
        validate_decision(
            parse_decision('{"action":"answer"}'), agents, allow_clarify=True, n_sources=0
        )


def test_clarify_for_missing_lists_field_descriptions():
    tool = ToolInfo("simular", "Simula empréstimo", LOAN_SCHEMA)
    text = clarify_for_missing(tool, ["valor", "prazo_meses"])
    assert "valor financiado em reais e prazo em meses" in text
