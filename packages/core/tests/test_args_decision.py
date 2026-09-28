from __future__ import annotations

import pytest

from switchboard.a2a import AgentInfo, SkillInfo
from switchboard.config import ProfileSpec
from switchboard.connectors import ConnectorInfo, ToolInfo
from switchboard.contracts import SkillTerms
from switchboard.llm.base import Message
from switchboard.rag.retriever import Hit
from switchboard.routing.args import (
    extract_arguments,
    find_mentions,
    find_person_name,
    parse_number,
    validate_arguments,
)
from switchboard.routing.capabilities import INSTRUCTION_SCHEMA, capabilities
from switchboard.routing.deciders import HeuristicDecider, is_question
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

CREDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "cliente": {"type": "string", "description": "Nome do cliente"},
        "valor": {"type": "number", "description": "Valor do crédito solicitado, em reais"},
        "prazo_meses": {"type": "integer", "description": "Prazo em meses"},
        "renda_mensal": {"type": "number", "description": "Renda mensal do cliente, em reais"},
    },
    "required": ["cliente", "valor", "prazo_meses", "renda_mensal"],
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


@pytest.mark.parametrize(
    ("text", "valor", "renda"),
    [
        # a renda vem antes do valor pedido: a palavra colada ao número decide
        (
            "Analise a proposta de Maria Souza: renda de 8 mil, pede 40 mil em 24 meses",
            40_000,
            8_000,
        ),
        (
            "Analise um crédito de R$ 80 mil para Maria Souza em 24 meses, renda de R$ 12 mil",
            80_000,
            12_000,
        ),
        ("Maria Souza ganha 25 mil por mês e quer 300 mil em 24 meses", 300_000, 25_000),
        ("Maria Souza quer 300 mil em 24 meses e ganha 25 mil", 300_000, 25_000),
        # sem deixa nenhuma, vale a ordem: primeiro o valor, depois a renda
        ("Proposta de Maria Souza: 50 mil em 24 meses, 9 mil", 50_000, 9_000),
    ],
)
def test_extract_arguments_uses_the_words_next_to_each_number(text, valor, renda):
    args = extract_arguments(CREDIT_SCHEMA, text)
    assert args == {
        "cliente": "Maria Souza",
        "valor": valor,
        "prazo_meses": 24,
        "renda_mensal": renda,
    }


def test_cued_plain_numbers_fill_term_and_rate():
    args = extract_arguments(LOAN_SCHEMA, "Simule 50 mil, prazo de 24, taxa de 1,5")
    assert args == {"valor": 50_000, "prazo_meses": 24, "taxa_mensal_percentual": 1.5}


def test_extract_arguments_text_fields_are_not_confused_with_money():
    text = "Não consigo acessar o aplicativo, erro no login. Prioridade alta"
    args = extract_arguments(TICKET_SCHEMA, text)
    assert args["prioridade"] == "alta"
    assert args["descricao"] == text
    assert args["titulo"].startswith("Não consigo acessar")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Analise uma proposta de 800 mil para João Lima em 120 meses", "João Lima"),
        ("Avalie o risco da operação de R$ 80 mil do cliente Maria Souza", "Maria Souza"),
        ("Maria da Silva quer 50 mil", "Maria da Silva"),
        ("Proposta de Maria Souza: 50 mil em 24 meses", "Maria Souza"),
        ("Crédito para Ana Lima", "Ana Lima"),
        ("Olá! Ana Paula Reis aqui", "Ana Paula Reis"),
        ("quero um crédito para João", "João"),
        ("Analise uma proposta de 50 mil", None),
        ("Compare Price e SAC para 100 mil", None),
        ("Quero Abrir Um Chamado", None),
    ],
)
def test_find_person_name(text, expected):
    assert find_person_name(text) == expected


def test_name_fields_get_a_name_or_stay_missing():
    schema = {
        "type": "object",
        "properties": {
            "cliente": {"type": "string", "description": "Nome do cliente"},
            "valor": {"type": "number", "description": "Valor da operação, em reais"},
        },
        "required": ["cliente", "valor"],
    }
    found = extract_arguments(schema, "Avalie o risco de R$ 80 mil do cliente Maria Souza")
    assert found == {"cliente": "Maria Souza", "valor": 80000.0}
    # sem nome no texto, o campo fica faltando (vira pergunta), em vez de receber o pedido inteiro
    assert "cliente" not in extract_arguments(schema, "Avalie o risco de uma operação de 80 mil")


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
        '{"acao": "acionar", "conector": "calc", "ferramenta": "somar", "arguments": "{\\"a\\": 1}"}'
    )
    assert (d.action, d.owner, d.target, d.arguments) == ("tool", "calc", "somar", {"a": 1})
    d = parse_decision('{"action": "answer", "answer": "ok", "sources": ["[2]", 1]}')
    assert d.sources == [2, 1]
    d = parse_decision(
        '{"action": "delegar", "tasks": [{"agent": "a", "skill": "s", "arguments": {}}, {"agent": "b", "skill": "t"}]}'
    )
    assert d.action == "delegate" and [t.key for t in d.tasks] == ["a/s", "b/t"]
    d = parse_decision('{"action": "delegate", "agent": "a", "skill": "s", "arguments": {"x": 1}}')
    assert [(t.kind, t.key, t.arguments) for t in d.tasks] == [("skill", "a/s", {"x": 1})]
    with pytest.raises(DecisionError):
        parse_decision('{"action": "dançar"}')


RISK_INPUT = {
    "type": "object",
    "properties": {
        "cliente": {"type": "string", "minLength": 2},
        "valor": {"type": "number", "exclusiveMinimum": 0},
    },
    "required": ["cliente", "valor"],
}
RISK_OUTPUT = {"type": "object", "properties": {"risco": {"type": "string"}}, "required": ["risco"]}


def _caps():
    loan = ToolInfo("simular", "Simula empréstimo", LOAN_SCHEMA)
    connectors = [ConnectorInfo("credito", "Crédito", "http://x", "online", tools=[loan])]
    agents = [
        AgentInfo(
            "risco",
            "Prevenção a fraude",
            "http://r",
            "online",
            skills=[
                SkillInfo(
                    "avaliar",
                    "Avaliar risco",
                    "Avalia o risco",
                    terms=SkillTerms(RISK_INPUT, RISK_OUTPUT),
                ),
                SkillInfo("broken", "Quebrada", "x", problems=("input_schema: inválido",)),
            ],
        ),
        AgentInfo(
            "livre",
            "Agente sem contrato",
            "http://l",
            "online",
            skills=[SkillInfo("resumir", "Resumir", "Resume textos")],
        ),
        AgentInfo("fora", "", "http://f", "offline", error="caiu"),
    ]
    return capabilities(connectors, agents)


def test_capabilities_merge_tools_and_skills():
    caps = {c.key: c for c in _caps()}
    assert set(caps) == {
        "credito/simular",
        "risco/avaliar",
        "livre/resumir",
    }  # quebrada e offline ficam de fora
    assert caps["credito/simular"].kind == "tool"
    assert (
        caps["risco/avaliar"].contract == "completo"
        and caps["risco/avaliar"].input_schema == RISK_INPUT
    )
    # skill sem contrato declarado recebe uma instrução em texto
    assert (
        caps["livre/resumir"].contract == "basico"
        and caps["livre/resumir"].arguments_schema == INSTRUCTION_SCHEMA
    )


def test_validate_decision_tool_paths():
    caps = _caps()
    ok = parse_decision(
        '{"action": "tool", "connector": "CREDITO", "tool": "simular", '
        '"arguments": {"valor": 1000, "taxa_mensal_percentual": 1, "prazo_meses": 10}}'
    )
    decided = validate_decision(ok, caps, allow_clarify=True, n_sources=0)
    assert (decided.action, decided.owner, decided.target) == ("tool", "credito", "simular")
    # dono inferido pelo nome ("dono/nome" também vale) e o tipo vem da capacidade real
    inferred = parse_decision(
        '{"action": "delegate", "tool": "credito/simular", '
        '"arguments": {"valor": 1000, "taxa_mensal_percentual": 1, "prazo_meses": 10}}'
    )
    assert validate_decision(inferred, caps, allow_clarify=True, n_sources=0).action == "tool"
    missing = parse_decision(
        '{"action": "tool", "connector": "credito", "tool": "simular", "arguments": {"valor": 5}}'
    )
    with pytest.raises(DecisionError) as err:
        validate_decision(missing, caps, allow_clarify=True, n_sources=0)
    assert err.value.missing == ["taxa_mensal_percentual", "prazo_meses"]
    assert err.value.capability.key == "credito/simular"
    unknown = parse_decision('{"action": "tool", "connector": "rh", "tool": "ferias"}')
    with pytest.raises(DecisionError, match="opções: credito/simular"):
        validate_decision(unknown, caps, allow_clarify=True, n_sources=0)


def test_validate_decision_delegate_paths():
    caps = _caps()
    both = parse_decision(
        '{"action": "delegate", "tasks": ['
        '{"agent": "risco", "skill": "avaliar", "arguments": {"cliente": "Ana", "valor": "10 mil"}},'
        '{"agent": "livre", "skill": "resumir", "arguments": {"instrucao": "resuma o contrato"}}]}'
    )
    decided = validate_decision(both, caps, allow_clarify=True, n_sources=0)
    assert decided.action == "delegate"
    assert [(t.kind, t.key) for t in decided.tasks] == [
        ("skill", "risco/avaliar"),
        ("skill", "livre/resumir"),
    ]
    assert decided.tasks[0].arguments == {"cliente": "Ana", "valor": 10000.0}
    assert decided.tasks[1].instruction == "resuma o contrato"
    # o contrato é estrito: valor fora do schema não passa
    bad = parse_decision(
        '{"action": "delegate", "agent": "risco", "skill": "avaliar", "arguments": {"cliente": "A", "valor": 5}}'
    )
    with pytest.raises(DecisionError, match="minLength|short"):
        validate_decision(bad, caps, allow_clarify=True, n_sources=0)
    mixed = parse_decision(
        '{"action": "delegate", "tasks": [{"agent": "risco", "skill": "avaliar", "arguments": {"cliente": "Ana", "valor": 1}},'
        '{"connector": "credito", "tool": "simular", "arguments": {}}]}'
    )
    with pytest.raises(DecisionError, match="não misture"):
        validate_decision(mixed, caps, allow_clarify=True, n_sources=0)
    too_many = parse_decision(
        '{"action": "delegate", "tasks": [{"agent": "risco", "skill": "avaliar", "arguments": {"cliente": "Ana", "valor": 1}},'
        '{"agent": "livre", "skill": "resumir", "arguments": {"instrucao": "x"}}]}'
    )
    with pytest.raises(DecisionError, match="no máximo 1"):
        validate_decision(too_many, caps, allow_clarify=True, n_sources=0, max_parallel=1)


def test_validate_decision_answer_and_clarify():
    caps = _caps()
    d = validate_decision(
        parse_decision('{"action":"answer","answer":"x","sources":[1,9]}'),
        caps,
        allow_clarify=True,
        n_sources=2,
    )
    assert d.sources == [1]
    with pytest.raises(DecisionError):
        validate_decision(
            parse_decision('{"action":"clarify","question":"?"}'),
            caps,
            allow_clarify=False,
            n_sources=0,
        )
    with pytest.raises(DecisionError):
        validate_decision(
            parse_decision('{"action":"answer"}'), caps, allow_clarify=True, n_sources=0
        )


def test_clarify_for_missing_lists_field_descriptions():
    cap = next(c for c in _caps() if c.key == "credito/simular")
    text = clarify_for_missing(cap, ["valor", "prazo_meses"])
    assert "valor financiado em reais e prazo em meses" in text


CREDIT_OUTPUT = {"type": "object", "properties": {"decisao": {"type": "string"}}}


@pytest.mark.parametrize(
    ("text", "action", "owner"),
    [
        # cita o crédito (uma palavra do título da skill), mas é uma pergunta: a base responde
        ("Quais documentos preciso para pedir crédito?", "answer", None),
        # pedido de ação com a mesma palavra: vai para a skill (e pergunta o que falta)
        ("Quero pedir um crédito", "clarify", "analise-credito"),
        # pergunta com casamento forte continua indo para a capacidade
        ("Qual o risco de fraude da operação de 80 mil da Maria Souza?", "delegate", "risco"),
        (
            "Analise a proposta de crédito de 80 mil da Maria Souza em 24 meses, renda de 12 mil",
            "delegate",
            "analise-credito",
        ),
    ],
)
async def test_heuristic_questions_with_weak_matches_go_to_the_knowledge_base(text, action, owner):
    credit_skill = SkillInfo(
        "analisar_proposta",
        "Analisar proposta de crédito",
        "Analisa uma proposta de crédito (cliente, valor, prazo e renda).",
        terms=SkillTerms(CREDIT_SCHEMA, CREDIT_OUTPUT),
    )
    risk_skill = SkillInfo(
        "avaliar_risco",
        "Avaliar risco de fraude",
        "Avalia o risco de fraude de uma operação.",
        terms=SkillTerms(RISK_INPUT, RISK_OUTPUT),
    )
    agents = [
        AgentInfo(
            "analise-credito", "Análise de crédito", "http://a", "online", skills=[credit_skill]
        ),
        AgentInfo("risco", "Prevenção a fraude", "http://r", "online", skills=[risk_skill]),
    ]
    hits = [
        Hit(
            "c1",
            "manual",
            "Manual",
            "Para pedir crédito, envie RG, CPF e comprovante de renda.",
            0.3,
        )
    ]
    decision, calls = await HeuristicDecider().decide(
        profile=ProfileSpec(model="offline"),
        messages=[Message("user", text)],
        caps=capabilities([], agents),
        hits=hits,
    )
    assert calls == [] and decision.action == action, decision.reason
    assert (decision.tasks[0].owner if decision.tasks else None) == owner
    if action == "answer":
        assert "comprovante de renda" in decision.answer and decision.sources == [1]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Quais documentos preciso?", True),
        ("como faço para abrir um chamado", True),
        ("Essa transação é segura?", True),
        ("Simule 50 mil em 24 meses", False),
        ("Quero abrir um chamado", False),
    ],
)
def test_is_question(text, expected):
    assert is_question(text) is expected
