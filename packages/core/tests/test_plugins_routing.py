"""Plugins no roteamento: playbooks nos prompts, uso por plugin/versão e o modo YAML."""

from __future__ import annotations

import pytest

from switchboard import ConnectorCatalog, Switchboard
from switchboard.a2a import AgentInfo, SkillInfo
from switchboard.config import (
    AgentSpec,
    ConnectorSpec,
    EmbedderSpec,
    KnowledgeBaseSpec,
    ModelSpec,
    ProfileSpec,
)
from switchboard.connectors import ConnectorInfo, ToolInfo
from switchboard.contracts import ContractManager, MemoryContractStore, states
from switchboard.embeddings import HashingEmbedder
from switchboard.llm import OfflineChat
from switchboard.llm.base import Message
from switchboard.plugins import Playbook, pack
from switchboard.rag import MemoryRetriever
from switchboard.rag.retriever import Hit
from switchboard.routing import ResolvedProfile, RouterEngine
from switchboard.routing.capabilities import capabilities
from switchboard.routing.playbooks import (
    GUIDANCE_CHARS,
    HEADER,
    MAX_SELECTED,
    PER_PLAYBOOK_CHARS,
    GuidedProfile,
    guidance_for,
    guided,
    plugin_uses,
    render,
    select_playbooks,
    with_guidance,
)
from switchboard.routing.prompts import (
    answer_messages,
    arguments_messages,
    consolidation_messages,
    decision_system_prompt,
    synthesis_messages,
)
from switchboard.testing import FakeA2AAgent, FakeNetwork, ScriptedChat, inproc_connector

CALC = "acme-teste.calc"
RISCO = "acme-teste.risco"
LLM = ModelSpec(name="llm", model="fake")


def playbook(
    skill: str = "simular-credito",
    *,
    description: str = "Simulação de financiamento, empréstimo e parcelas.",
    body: str = "Pergunte o prazo antes de simular.",
    applies_to: tuple[str, ...] = (),
    triggers: tuple[str, ...] = (),
    always: bool = False,
    plugin: str = "acme-teste",
) -> Playbook:
    return Playbook(
        plugin=plugin,
        version="1.0.0",
        skill=skill,
        description=description,
        body=body,
        applies_to=applies_to,
        triggers=triggers,
        always=always,
    )


def users(*texts: str) -> list[Message]:
    return [Message("user", t) for t in texts]


def keys(picks) -> list[str]:
    return [p.playbook.skill for p in picks]


# --------------------------------------------------------------------------
# seleção de playbooks


def test_triggers_match_whole_phrases_without_accents_or_plurals():
    pb = playbook(
        "atender-clientes",
        description="Atendimento geral.",
        triggers=("simulação", "limite de crédito"),
    )
    [pick] = select_playbooks([pb], users("Quero uma SIMULACAO de 50 mil"), owners=[])
    assert pick.playbook is pb and pick.reason == "gatilho 'simulação'" and pick.score == 3.0
    assert keys(select_playbooks([pb], users("Qual o meu limite de crédito?"), owners=[])) == [
        "atender-clientes"
    ]
    plural = playbook("atender-clientes", description="Atendimento geral.", triggers=("parcela",))
    assert keys(select_playbooks([plural], users("quais as parcelas?"), owners=[])) == [
        "atender-clientes"
    ]
    # o gatilho é uma expressão inteira: pedaço de palavra não vale
    assert select_playbooks([plural], users("a parcelamento"), owners=[]) == []
    assert (
        select_playbooks([playbook(triggers=("ab",))], users("ab"), owners=[]) == []
    )  # curto demais


def test_name_and_description_words_score_without_triggers():
    pb = playbook(
        "simular-credito", description="Como atender pedidos de financiamento imobiliário."
    )
    # as duas palavras do nome (2 pontos cada) passam do mínimo
    assert keys(select_playbooks([pb], users("quero simular um crédito"), owners=[])) == [
        "simular-credito"
    ]
    # uma palavra do nome (2) mais duas da descrição (1+1) também
    assert keys(select_playbooks([pb], users("financiamento imobiliário, simular"), owners=[])) == [
        "simular-credito"
    ]
    # só uma palavra do nome: pouco
    assert select_playbooks([pb], users("crédito"), owners=[]) == []
    assert select_playbooks([pb], users("qual o horário de atendimento?"), owners=[]) == []
    # números do pedido não casam com nada
    assert select_playbooks([pb], users("50 24 12"), owners=[]) == []


def test_always_playbooks_enter_every_request_and_do_not_use_the_limit():
    always = playbook("sempre", always=True)
    many = [playbook(f"credito-{i}", triggers=("credito",)) for i in range(5)]
    picks = select_playbooks([always, *many], users("quero credito"), owners=[])
    assert keys(picks) == ["sempre", "credito-0", "credito-1", "credito-2"]
    assert len(picks) == MAX_SELECTED + 1 and picks[0].reason.startswith("sempre")
    assert keys(select_playbooks([always], users("oi"), owners=[])) == ["sempre"]


def test_best_scores_win_and_ties_follow_the_key():
    weak = playbook("fraco", description="Algo.", triggers=("credito",))
    strong = playbook("forte", description="Algo.", triggers=("credito", "empréstimo"))
    picks = select_playbooks([weak, strong], users("credito e emprestimo"), owners=[])
    assert keys(picks) == ["forte", "fraco"] and picks[0].score == 6.0
    a = playbook("b-skill", description="Algo.", triggers=("x1x",))
    b = playbook("a-skill", description="Algo.", triggers=("x1x",))
    assert keys(select_playbooks([a, b], users("x1x"), owners=[])) == ["a-skill", "b-skill"]


def test_a_playbook_needs_the_components_it_refers_to():
    pb = playbook(applies_to=(CALC,), triggers=("credito",))
    assert select_playbooks([pb], users("credito"), owners=["outro.coisa"]) == []
    assert keys(select_playbooks([pb], users("credito"), owners=["outro.coisa", CALC])) != []
    anywhere = playbook(triggers=("credito",))  # applies_to vazio: vale para o plugin todo
    assert keys(select_playbooks([anywhere], users("credito"), owners=[])) != []


def test_the_request_and_the_previous_user_turn_are_considered():
    pb = playbook(triggers=("financiamento",))
    convo = [
        Message("user", "quero um financiamento"),
        Message("assistant", "de quanto?"),
        Message("user", "de 50 mil"),
    ]
    assert keys(select_playbooks([pb], convo, owners=[])) == ["simular-credito"]
    older = [Message("user", "financiamento"), *users("oi", "tudo bem?")]
    assert select_playbooks([pb], older, owners=[]) == []
    # o que o assistente escreveu não ativa playbook algum
    assert select_playbooks([pb], [Message("assistant", "financiamento")], owners=[]) == []


# --------------------------------------------------------------------------
# texto das orientações


def test_render_wraps_playbooks_under_a_third_party_header():
    assert render([]) == ""
    text = render([playbook(body="Faça A.\nFaça B.")])
    assert text.startswith(HEADER) and "não muda as regras acima" in HEADER
    assert "### acme-teste · simular-credito\nFaça A.\nFaça B." in text


def test_render_respects_the_budgets():
    big = "palavra " * 2000
    text = render([playbook("a", body=big)])
    assert "[…texto cortado]" in text and len(text) < len(HEADER) + PER_PLAYBOOK_CHARS + 200
    many = render([playbook(f"s{i}", body=big) for i in range(10)])
    assert len(many) <= len(HEADER) + GUIDANCE_CHARS + 400
    assert "### acme-teste · s0" in many and "### acme-teste · s9" not in many


def test_guidance_for_scopes_by_request_or_by_component():
    spec = ProfileSpec(name="p", model="llm")
    assert guidance_for(spec) == "" and guidance_for(spec, [CALC]) == ""
    credit = playbook("credito", body="TEXTO-CREDITO", applies_to=(CALC,))
    risk = playbook("risco", body="TEXTO-RISCO", applies_to=(RISCO,))
    whole = playbook("geral", body="TEXTO-GERAL")
    profile = guided(spec, [credit, risk, whole], selected=["acme-teste/risco"])
    assert isinstance(profile, GuidedProfile) and isinstance(profile, ProfileSpec)
    # sem componente: só o que foi escolhido para o pedido
    text = guidance_for(profile)
    assert "TEXTO-RISCO" in text and "TEXTO-CREDITO" not in text and "TEXTO-GERAL" not in text
    # com componente: o que cobre aquele componente
    text = guidance_for(profile, [CALC])
    assert "TEXTO-CREDITO" in text and "TEXTO-GERAL" in text and "TEXTO-RISCO" not in text
    assert guidance_for(profile, ["fora.de-plugin"]) == ""
    assert with_guidance("base", profile, [RISCO]).startswith("base\n\n" + HEADER)
    assert with_guidance("base", spec, [RISCO]) == "base"


def test_guided_profile_is_a_plain_profile_for_everything_else():
    spec = ProfileSpec(name="p", model="llm", connectors=[CALC], plugins=["acme-teste"])
    assert guided(spec, []) is spec
    profile = guided(spec, [playbook(body="segredo do playbook")], ["acme-teste/simular-credito"])
    assert profile.name == "p" and profile.connectors == [CALC]
    # os playbooks nunca saem numa serialização do perfil (traces, API)
    assert "segredo do playbook" not in repr(profile.model_dump())
    assert set(profile.model_dump()) == set(spec.model_dump())


# --------------------------------------------------------------------------
# prompts


def make_caps(examples=None):
    tool = ToolInfo("somar", "Soma dois números.", {"type": "object", "properties": {}})
    connectors = [ConnectorInfo(CALC, "Calculadora", "http://c", "online", tools=[tool])]
    agents = [
        AgentInfo(
            RISCO,
            "Risco",
            "http://r",
            "online",
            skills=[
                SkillInfo(
                    "avaliar",
                    "Avaliar risco",
                    "Avalia o risco",
                    examples=("Avalie o risco desta operação",),
                )
            ],
        )
    ]
    return capabilities(connectors, agents, examples)


def test_routing_examples_from_plugins_come_before_the_agents_own():
    caps = {
        c.key: c
        for c in make_caps(
            {
                f"{CALC}/somar": ["Quanto é 2 mais 3?"],
                f"{RISCO}/avaliar": ["Avalie a fraude de 80 mil", "Avalie o risco desta operação"],
            }
        )
    }
    assert caps[f"{CALC}/somar"].examples == ("Quanto é 2 mais 3?",)
    assert caps[f"{RISCO}/avaliar"].examples == (
        "Avalie a fraude de 80 mil",
        "Avalie o risco desta operação",  # sem repetir o que o agente já traz
    )
    assert make_caps()[1].examples == ("Avalie o risco desta operação",)


def test_decision_prompt_carries_only_the_chosen_playbooks_after_the_catalog():
    chosen = playbook("escolhido", body="TEXTO-ESCOLHIDO")
    other = playbook("outro", body="TEXTO-OUTRO")
    spec = guided(
        ProfileSpec(name="p", model="llm", system_prompt="Você é o atendente da Acme."),
        [chosen, other],
        ["acme-teste/escolhido"],
    )
    hits = [
        Hit(
            chunk_id="c1",
            kb="acme-teste.faq",
            document="faq.md",
            content="Prazo de 5 dias.",
            score=0.9,
        )
    ]
    prompt = decision_system_prompt(spec, make_caps(), hits)
    assert prompt.startswith("Você é o atendente da Acme.")
    assert "TEXTO-ESCOLHIDO" in prompt and "TEXTO-OUTRO" not in prompt
    order = [
        prompt.index("## CAPACIDADES DISPONÍVEIS"),
        prompt.index("## ORIENTAÇÕES DOS PLUGINS"),
        prompt.index("## TRECHOS DE CONHECIMENTO"),
    ]
    assert order == sorted(order)
    plain = decision_system_prompt(ProfileSpec(name="p", model="llm"), make_caps(), hits)
    assert "ORIENTAÇÕES" not in plain


def test_answer_arguments_synthesis_and_consolidation_prompts_use_the_covering_playbooks():
    credit = playbook("credito", body="TEXTO-CREDITO", applies_to=(CALC,))
    risk = playbook("risco", body="TEXTO-RISCO", applies_to=(RISCO,))
    spec = guided(ProfileSpec(name="p", model="llm"), [credit, risk], ["acme-teste/credito"])
    cap_calc, cap_risk = make_caps()
    answer = answer_messages(spec, users("oi"), [])[0].content
    assert "TEXTO-CREDITO" in answer and "TEXTO-RISCO" not in answer  # os escolhidos do pedido
    args = arguments_messages(spec, users("some"), cap_calc)[0].content
    assert "TEXTO-CREDITO" in args and "TEXTO-RISCO" not in args
    args = arguments_messages(spec, users("avalie"), cap_risk)[0].content
    assert "TEXTO-RISCO" in args and "TEXTO-CREDITO" not in args
    synth = synthesis_messages(spec, "some", f"{CALC}/somar", {}, "5", False)[0].content
    assert "TEXTO-CREDITO" in synth and "TEXTO-RISCO" not in synth
    consolidation = consolidation_messages(
        spec, "avalie", [], [{"agente": RISCO, "estado": "Concluído"}]
    )[0].content
    assert "TEXTO-RISCO" in consolidation and "TEXTO-CREDITO" not in consolidation
    # sem plugins, os prompts são os de sempre
    bare = ProfileSpec(name="p", model="llm")
    assert "ORIENTAÇÕES" not in arguments_messages(bare, users("x"), cap_calc)[0].content
    assert (
        "ORIENTAÇÕES"
        not in synthesis_messages(bare, "x", f"{CALC}/somar", {}, "5", False)[0].content
    )


# --------------------------------------------------------------------------
# uso por plugin e versão


def test_plugin_uses_are_deduplicated_and_limited_to_active_plugins():
    versions = {"acme-teste": "1.2.0"}
    uses = plugin_uses(
        versions,
        picks=[playbook("a"), playbook("a"), playbook("b", plugin="outro")],
        tasks=[
            ("tool", CALC, "somar", True),
            ("tool", CALC, "somar", True),
            ("skill", RISCO, "avaliar", None),
            ("tool", "manual", "x", True),  # conector cadastrado à mão: sem plugin
            ("tool", "outro.coisa", "x", True),  # plugin que não está ativo no perfil
        ],
        knowledge=["acme-teste.faq", "faq-manual", "acme-teste.faq"],
    )
    assert [(u["kind"], u["component"]) for u in uses] == [
        ("playbook", "a"),
        ("tool", f"{CALC}/somar"),
        ("skill", f"{RISCO}/avaliar"),
        ("knowledge", "acme-teste.faq"),
    ]
    assert {u["version"] for u in uses} == {"1.2.0"}
    assert [u["ok"] for u in uses] == [None, True, None, None]


# --------------------------------------------------------------------------
# motor com plugin


def plugin_profile(*, playbooks=None, kbs=(), agents=(), connectors=True, **kw) -> ResolvedProfile:
    conns = [ConnectorSpec(name=CALC, url="http://calc.local/mcp")] if connectors else []
    spec = ProfileSpec(
        name="teste",
        model=LLM.name,
        connectors=[c.name for c in conns],
        agents=[a.name for a in agents],
        knowledge_bases=list(kbs),
        plugins=["acme-teste"],
        min_score=0.05,
        **kw,
    )
    return ResolvedProfile(
        spec=spec,
        model=LLM,
        connectors=conns,
        agents=list(agents),
        plugins={"acme-teste": "1.0.0"},
        playbooks=playbooks
        if playbooks is not None
        else [
            playbook(
                "usar-calculo",
                body="REGRA-DO-PLUGIN: confirme os números antes de somar.",
                applies_to=(CALC,),
                triggers=("somar",),
            )
        ],
        examples={f"{CALC}/somar": ["Quanto é 2 mais 3?"]},
    )


@pytest.fixture
def plugin_catalog(servers) -> ConnectorCatalog:
    return ConnectorCatalog(connector=inproc_connector({CALC: servers["calc"]}))


def tool_decision(reason: str = "conta") -> str:
    return (
        f'{{"action": "tool", "connector": "{CALC}", "tool": "somar", '
        f'"arguments": {{"a": 2, "b": 3}}, "reason": "{reason}"}}'
    )


async def test_engine_guides_the_llm_and_attributes_the_tool_to_the_plugin(plugin_catalog):
    chat = ScriptedChat([tool_decision(), "A soma dá 5."])
    engine = RouterEngine(plugin_profile(), chat=chat, connectors=plugin_catalog)
    result = await engine.handle("Preciso somar 2 e 3")
    assert (result.route, result.agent, result.tool) == ("tool", CALC, "somar")
    assert result.answer == "A soma dá 5."
    decision, synthesis = chat.calls[0][0].content, chat.calls[1][0].content
    assert "REGRA-DO-PLUGIN" in decision and "Quanto é 2 mais 3?" in decision
    assert decision.index("## CAPACIDADES DISPONÍVEIS") < decision.index("ORIENTAÇÕES DOS PLUGINS")
    assert "REGRA-DO-PLUGIN" in synthesis  # o playbook cobre o conector usado
    assert [
        (u["plugin"], u["version"], u["kind"], u["component"], u["ok"]) for u in result.plugin_uses
    ] == [
        ("acme-teste", "1.0.0", "playbook", "usar-calculo", None),
        ("acme-teste", "1.0.0", "tool", f"{CALC}/somar", True),
    ]
    span = next(s for s in result.spans if s["name"] == "playbooks")
    assert span["attributes"]["disponiveis"] == 1
    [picked] = span["attributes"]["escolhidos"]
    assert picked["playbook"] == "usar-calculo" and picked["versao"] == "1.0.0"
    assert result.to_dict()["plugins"] == result.plugin_uses


async def test_a_playbook_the_request_did_not_trigger_still_guides_the_tool_it_covers(
    plugin_catalog,
):
    chat = ScriptedChat([tool_decision(), "5."])
    engine = RouterEngine(plugin_profile(), chat=chat, connectors=plugin_catalog)
    result = await engine.handle("quanto dá dois mais três?")
    assert "REGRA-DO-PLUGIN" not in chat.calls[0][0].content  # a decisão não a viu
    assert "REGRA-DO-PLUGIN" in chat.calls[1][0].content  # a síntese, sim
    assert [u["kind"] for u in result.plugin_uses] == ["playbook", "tool"]
    span = next(s for s in result.spans if s["name"] == "playbooks")
    assert span["attributes"]["escolhidos"] == []


async def test_a_failing_tool_is_recorded_as_a_plugin_error(plugin_catalog):
    reply = f'{{"action": "tool", "connector": "{CALC}", "tool": "falhar", "arguments": {{"motivo": "x"}}}}'
    chat = ScriptedChat([reply, "Não consegui."])
    engine = RouterEngine(plugin_profile(), chat=chat, connectors=plugin_catalog)
    result = await engine.handle("preciso falhar")
    tool = next(u for u in result.plugin_uses if u["kind"] == "tool")
    assert tool["component"] == f"{CALC}/falhar" and tool["ok"] is False


async def test_direct_answers_credit_the_plugin_knowledge_base(plugin_catalog):
    retriever = MemoryRetriever()
    retriever.register(
        KnowledgeBaseSpec(name="acme-teste.faq", embedder=EmbedderSpec()), HashingEmbedder()
    )
    await retriever.add_text("acme-teste.faq", "faq.md", "## Prazo\n\nA entrega leva 5 dias úteis.")
    chat = ScriptedChat(['{"action": "answer", "answer": "5 dias úteis.", "sources": [1]}'])
    engine = RouterEngine(
        plugin_profile(kbs=("acme-teste.faq",)),
        chat=chat,
        connectors=plugin_catalog,
        retriever=retriever,
    )
    result = await engine.handle("Qual o prazo de entrega do pedido?")
    assert result.route == "direct"
    assert [(u["kind"], u["component"]) for u in result.plugin_uses] == [
        ("knowledge", "acme-teste.faq")
    ]


async def test_requests_without_plugin_components_leave_no_usage(plugin_catalog):
    chat = ScriptedChat(['{"action": "answer", "answer": "Olá!"}'])
    engine = RouterEngine(plugin_profile(), chat=chat, connectors=plugin_catalog)
    result = await engine.handle("oi")
    assert result.route == "direct" and result.plugin_uses == []


async def test_profiles_without_plugins_are_untouched(catalog):
    chat = ScriptedChat([tool_decision().replace(CALC, "calc"), "5."])
    spec = ProfileSpec(name="p", model="llm", connectors=["calc"])
    profile = ResolvedProfile(
        spec=spec, model=LLM, connectors=[ConnectorSpec(name="calc", url="http://calc.local/mcp")]
    )
    result = await RouterEngine(profile, chat=chat, connectors=catalog).handle("some 2 e 3")
    assert result.plugin_uses == [] and "playbooks" not in [s["name"] for s in result.spans]
    assert "ORIENTAÇÕES" not in chat.calls[0][0].content


async def test_profile_warnings_reach_every_result(plugin_catalog):
    profile = plugin_profile()
    profile.warnings = ["plugin acme-teste 1.0.0 não está ativo: aguardando aprovação"]
    engine = RouterEngine(profile, chat=OfflineChat(), connectors=plugin_catalog)
    result = await engine.handle("oi")
    assert "aguardando aprovação" in " ".join(result.warnings)


async def test_offline_engine_still_records_playbooks_and_usage(plugin_catalog):
    engine = RouterEngine(plugin_profile(), chat=OfflineChat(), connectors=plugin_catalog)
    result = await engine.handle("preciso somar 10 e 32")
    assert (result.route, result.tool, result.answer) == ("tool", "somar", "resultado: 42")
    assert [u["kind"] for u in result.plugin_uses] == ["playbook", "tool"]


# --------------------------------------------------------------------------
# delegação a agentes de plugin


def delegating_engine(agent: FakeA2AAgent, chat, playbooks):
    network = FakeNetwork(agent)
    directory = network.directory()
    manager = ContractManager(
        MemoryContractStore(), client=directory.client, poll_min_s=0.01, tick_s=0.02
    )
    profile = plugin_profile(
        playbooks=playbooks, connectors=False, agents=[AgentSpec(name=RISCO, url=agent.url)]
    )
    engine = RouterEngine(
        profile, chat=chat, connectors=ConnectorCatalog(), agents=directory, contracts=manager
    )
    manager.consolidator = engine.consolidate
    return engine, manager


async def test_delegation_to_a_plugin_agent_is_attributed_and_consolidated_with_the_playbook():
    agent = FakeA2AAgent()
    agent.on_send = lambda params: {
        "status": {"state": "TASK_STATE_COMPLETED"},
        "artifacts": [{"artifactId": "a", "parts": [{"data": {"risco": "baixo", "score": 800}}]}],
    }
    chat = ScriptedChat(
        [
            f'{{"action": "delegate", "tasks": [{{"agent": "{RISCO}", "skill": "avaliar_risco", '
            '"arguments": {"cliente": "Ana Lima", "valor": "80 mil"}}], "reason": "fraude"}',
            "O risco é baixo.",
        ]
    )
    whole_plugin = playbook("atender", body="REGRA-DE-ATENDIMENTO", triggers=("fraude",))
    engine, manager = delegating_engine(agent, chat, [whole_plugin])
    result = await engine.handle("verifique o risco de fraude de 80 mil da Ana Lima")
    assert (result.route, result.status, result.answer) == (
        "delegated",
        "completed",
        "O risco é baixo.",
    )
    assert result.contracts[0]["state"] == states.COMPLETED
    assert "REGRA-DE-ATENDIMENTO" in chat.calls[0][0].content  # decisão
    assert "REGRA-DE-ATENDIMENTO" in chat.calls[1][0].content  # consolidação
    assert [(u["kind"], u["component"], u["ok"]) for u in result.plugin_uses] == [
        ("playbook", "atender", None),
        ("skill", f"{RISCO}/avaliar_risco", True),  # o contrato terminou COMPLETED
    ]
    spans = await manager.store.run_spans(result.run_id)
    consolidation = next(s for s in spans if s.name == "consolidacao")
    assert consolidation.attributes["playbooks"] == ["acme-teste/atender"]


# --------------------------------------------------------------------------
# modo YAML

CONFIG_YAML = """
models:
  - name: offline
    provider: offline
plugins:
  - name: acme-teste
    path: plugins/acme-teste
    config:
      CALC_URL: http://calc.local/mcp
      RISCO_URL: http://risco.test
profiles:
  - name: default
    model: offline
    plugins: [acme-teste]
    min_score: 0.05
    wait_s: 1
  - name: sem-plugin
    model: offline
"""


def yaml_switchboard(tmp_path, plugin_files, servers, network, *, yaml_text=CONFIG_YAML, **kw):
    for path, data in plugin_files(**kw).items():
        target = tmp_path / "plugins" / "acme-teste" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    config = tmp_path / "switchboard.yaml"
    config.write_text(yaml_text, encoding="utf-8")
    directory = network.directory()
    manager = ContractManager(
        MemoryContractStore(), client=directory.client, poll_min_s=0.05, tick_s=0.05
    )
    return Switchboard.from_yaml(
        config,
        connectors=ConnectorCatalog(connector=inproc_connector({CALC: servers["calc"]})),
        agents=directory,
        contracts=manager,
    )


async def test_yaml_mode_merges_the_plugin_into_the_profile(
    tmp_path, plugin_files, servers, network
):
    sb = yaml_switchboard(tmp_path, plugin_files, servers, network)
    resolved = sb.resolve("default")
    assert [c.name for c in resolved.connectors] == [CALC]
    assert [a.name for a in resolved.agents] == [RISCO]
    assert resolved.spec.knowledge_bases == ["acme-teste.faq"]
    assert resolved.spec.connectors == [CALC] and resolved.spec.plugins == ["acme-teste"]
    assert resolved.plugins == {"acme-teste": "1.0.0"}
    assert [p.key for p in resolved.playbooks] == ["acme-teste/usar-calculo"]
    assert resolved.examples == {f"{CALC}/somar": ["Quanto é 2 mais 3?"]}
    # o perfil sem plugin continua igual
    bare = sb.resolve("sem-plugin")
    assert bare.plugins == {} and bare.connectors == [] and bare.playbooks == []


async def test_yaml_mode_indexes_the_plugin_documents_and_answers_with_them(
    tmp_path, plugin_files, servers, network
):
    async with yaml_switchboard(tmp_path, plugin_files, servers, network) as sb:
        assert sb.retriever.stats()["acme-teste.faq"] >= 1
        result = await sb.ask("Qual o prazo de entrega do pedido?")
        assert any(s.kb == "acme-teste.faq" for s in result.sources)
        assert [(u["plugin"], u["kind"]) for u in result.plugin_uses][-1] == (
            "acme-teste",
            "knowledge",
        )


async def test_yaml_mode_routes_to_the_plugin_connector_and_uses_its_examples(
    tmp_path, plugin_files, servers, network
):
    async with yaml_switchboard(tmp_path, plugin_files, servers, network) as sb:
        result = await sb.ask("preciso somar 10 e 32")
        assert (result.route, result.agent, result.tool) == ("tool", CALC, "somar")
        assert result.answer == "resultado: 42"
        assert {"playbook", "tool"} <= {u["kind"] for u in result.plugin_uses}
        statuses = {c.name: c.status for c in await sb.connectors_status("default")}
        assert statuses == {CALC: "online"}
        agents = {a.name: a.status for a in await sb.agents_status("default")}
        assert agents == {RISCO: "online"}


async def test_yaml_mode_delegates_to_the_plugin_agent(
    tmp_path, plugin_files, servers, network, risk_agent
):
    risk_agent.on_send = lambda params: {
        "status": {"state": "TASK_STATE_COMPLETED"},
        "artifacts": [{"artifactId": "a", "parts": [{"text": "Risco baixo."}]}],
    }
    async with yaml_switchboard(tmp_path, plugin_files, servers, network) as sb:
        result = await sb.ask("avaliar risco de fraude da operação de 12 mil do cliente Bruno Dias")
        assert (result.route, result.status) == ("delegated", "completed")
        [contract] = await sb.run_contracts(result.run_id)
        assert contract.agent == RISCO
        assert any(
            u["kind"] == "skill" and u["component"].startswith(RISCO) for u in result.plugin_uses
        )


async def test_yaml_mode_disabled_plugin_is_left_out_with_a_warning(
    tmp_path, plugin_files, servers, network
):
    text = CONFIG_YAML.replace(
        "    path: plugins/acme-teste\n", "    path: plugins/acme-teste\n    enabled: false\n"
    )
    sb = yaml_switchboard(tmp_path, plugin_files, servers, network, yaml_text=text)
    assert sb.plugins == []
    resolved = sb.resolve("default")
    assert resolved.connectors == [] and resolved.plugins == {}
    assert any("desativado" in w for w in resolved.warnings)
    async with sb:
        result = await sb.ask("oi")
        assert any("desativado" in w for w in result.warnings)


async def test_yaml_mode_fails_early_when_a_plugin_is_missing_config(
    tmp_path, plugin_files, servers, network
):
    from switchboard.errors import ConfigError

    text = CONFIG_YAML.replace("      RISCO_URL: http://risco.test\n", "")
    with pytest.raises(ConfigError, match="falta configurar RISCO_URL"):
        yaml_switchboard(tmp_path, plugin_files, servers, network, yaml_text=text)


def test_yaml_zip_plugins_work_too(tmp_path, plugin_files, servers, network):
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins" / "acme-teste").write_bytes(pack(plugin_files()))
    config = tmp_path / "switchboard.yaml"
    config.write_text(CONFIG_YAML, encoding="utf-8")
    sb = Switchboard.from_yaml(config)
    assert [p.name for p in sb.plugins] == ["acme-teste"]
    assert sb.resolve("default").plugins == {"acme-teste": "1.0.0"}
