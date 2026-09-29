"""Jev (System One): cliente HTTP e decisor híbrido com o LLM (System Two)."""

from __future__ import annotations

import json

import httpx
import pytest

from switchboard.config import (
    ConnectorSpec,
    EmbedderSpec,
    KnowledgeBaseSpec,
    ModelSpec,
    ProfileSpec,
)
from switchboard.embeddings import HashingEmbedder
from switchboard.errors import ConfigError, DecisionModelError
from switchboard.jev import JevClient, build_decision_model, choice, endpoint_url, noul, score
from switchboard.llm import OfflineChat, build_chat_model
from switchboard.rag import MemoryRetriever
from switchboard.routing import ResolvedProfile, RouterEngine
from switchboard.testing import ScriptedChat, jev_choice, jev_noul, scripted_jev

# --------------------------------------------------------------------------
# cliente


def test_question_builders_and_endpoint():
    q = choice("Qual?", {"a": "opção a", "b": {"what": "opção b", "examples": ["x"]}})
    assert q == {
        "type": "choice",
        "instructions": "Qual?",
        "criteria": {"a": "opção a", "b": {"what": "opção b", "examples": ["x"]}},
    }
    assert noul("É sim?", true="sim", false="não")["criteria"] == {"true": "sim", "false": "não"}
    assert "criteria" not in noul("É sim?")
    assert score("Quão grave?", ["leve", "grave"])["criteria"] == ["leve", "grave"]
    with pytest.raises(ValueError):
        choice("x", {"só": "uma"})
    with pytest.raises(ValueError):
        score("x", ["1"])
    assert endpoint_url(None) == "https://api.typesafe.ai/v1/systemone"
    assert endpoint_url("https://openrouter.ai/api") == "https://openrouter.ai/api/v1/systemone"
    assert endpoint_url("https://proxy.local/v1/") == "https://proxy.local/v1/systemone"


async def test_client_request_shape_and_parse():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "rota": {
                        "type": "choice",
                        "choice": "b",
                        "confidence": 0.8,
                        "probabilities": {"a": 0.1, "b": 0.9},
                    },
                    "sim": {"type": "noul", "noul": 0.93},
                    "grav": {
                        "type": "score",
                        "score": 1.43,
                        "probabilities": {"0": 0, "1": 0.57, "2": 0.43},
                        "confidence": 0.5,
                    },
                },
                "usage": {"input_tokens": 360, "output_tokens": 39},
            },
        )

    client = JevClient(model="jev-1.13.0", api_key="k", transport=httpx.MockTransport(handler))
    result = await client.evaluate(
        {"mensagem": "oi"}, {"rota": choice("?", {"a": "A", "b": "B"}), "sim": noul("?")}
    )
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone" and seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "jev-1.13.0" and seen["body"]["state"] == {"mensagem": "oi"}
    rota = result.get("rota")
    assert (rota.choice, rota.confidence) == ("b", 0.8)
    assert rota.ranked() == [("b", 0.9), ("a", 0.1)]
    assert result.get("sim").noul == 0.93 and result.get("grav").score == 1.43
    assert result.usage.input_tokens == 360 and result.to_dict()["answers"]["sim"] == {
        "type": "noul",
        "noul": 0.93,
    }
    await client.aclose()


async def test_client_retries_rate_limits_and_reports_errors():
    calls = []
    waits = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                429, headers={"retry-after": "1"}, json={"error": {"message": "calma"}}
            )
        if len(calls) == 2:
            return httpx.Response(529, json={"error": {"message": "sobrecarregado"}})
        return httpx.Response(
            200, json={"model": "m", "answers": {"q": {"type": "noul", "noul": 0.1}}}
        )

    client = JevClient(
        api_key="k", transport=httpx.MockTransport(handler), max_retries=2, sleep=fake_sleep
    )
    assert (await client.evaluate("s", {"q": noul("?")})).get("q").noul == 0.1
    assert len(calls) == 3 and waits == [1.0, 1.0]

    bad = JevClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(422, json={"error": {"message": "questions inválidas"}})
        ),
        max_retries=0,
    )
    with pytest.raises(DecisionModelError, match="HTTP 422 - questions inválidas") as err:
        await bad.evaluate("s", {"q": noul("?")})
    assert err.value.status == 422

    def boom(request):
        raise httpx.ConnectError("recusada")

    down = JevClient(transport=httpx.MockTransport(boom), max_retries=1, sleep=fake_sleep)
    with pytest.raises(DecisionModelError, match="falha de rede"):
        await down.evaluate("s", {"q": noul("?")})

    weird = JevClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"answers": {"q": {"type": "noul"}}})
        )
    )
    with pytest.raises(DecisionModelError, match="sem o campo"):
        await weird.evaluate("s", {"q": noul("?")})


def test_factories_keep_chat_and_decision_models_apart():
    jev_spec = ModelSpec(name="jev", provider="typesafe", model="jev-1.13.0", api_key="k")
    client = build_decision_model(jev_spec)
    assert client.label == "jev:jev-1.13.0" and client.url.endswith("/v1/systemone")
    with pytest.raises(ConfigError, match="modelo de decisão"):
        build_chat_model(jev_spec)
    with pytest.raises(ConfigError, match="modelo de chat"):
        build_decision_model(ModelSpec(name="llm", model="gpt"))


# --------------------------------------------------------------------------
# decisor híbrido no motor

CALC = ConnectorSpec(name="calc", url="http://calc.local/mcp")
LLM = ModelSpec(name="llm", model="fake")


def _profile(**kw) -> ResolvedProfile:
    spec = ProfileSpec(
        name="teste",
        model="llm",
        decision_model="jev",
        connectors=["calc"],
        knowledge_bases=["faq"],
        min_score=0.05,
        **kw,
    )
    return ResolvedProfile(
        spec=spec,
        model=LLM,
        connectors=[CALC],
        decision_model=ModelSpec(name="jev", provider="typesafe", model="jev-test"),
    )


async def _retriever() -> MemoryRetriever:
    retriever = MemoryRetriever()
    retriever.register(KnowledgeBaseSpec(name="faq", embedder=EmbedderSpec()), HashingEmbedder())
    await retriever.add_text(
        "faq", "horarios.md", "## Horários\n\nAtendemos aos sábados das 9h às 14h."
    )
    return retriever


def route_script(route: str, confidence: float = 0.95, **answers):
    """Roteiro do Jev: rota fixa; 'kb' e 'stated_<j>' opcionais."""

    def script(state, questions):
        out = {}
        for key, question in questions.items():
            if key == "route":
                assert question["type"] == "choice"
                out[key] = jev_choice(route, confidence)
            elif key == "kb_answers":
                out[key] = jev_noul(answers.get("kb", 0.9))
            elif key.startswith("stated_"):
                out[key] = jev_noul(answers.get(key, 0.95))
            elif key.startswith("needs_"):
                out[key] = jev_noul(answers.get(key, 0.1))
        return out

    return script


async def test_jev_routes_to_a_tool_and_llm_fills_arguments(catalog):
    calls: list = []
    jev = scripted_jev(route_script("calc/somar"), calls=calls)
    chat = ScriptedChat(['{"arguments": {"a": 20, "b": "22"}}', "A soma dá 42."])
    engine = RouterEngine(
        _profile(), chat=chat, connectors=catalog, retriever=await _retriever(), jev=jev
    )
    result = await engine.handle("some 20 com 22")
    assert (result.route, result.decided_by, result.agent, result.tool) == (
        "tool",
        "jev",
        "calc",
        "somar",
    )
    assert result.arguments == {"a": 20.0, "b": 22.0} and result.answer == "A soma dá 42."
    assert result.confidence == 0.95
    # 1ª chamada ao Jev: rota com as capacidades como opções (critérios estruturados)
    route_q = next(c for c in calls if "route" in c["questions"])["questions"]["route"]
    assert {"knowledge", "out_of_scope", "calc/somar", "calc/converter_moeda"} <= set(
        route_q["criteria"]
    )
    assert "what" in route_q["criteria"]["calc/somar"]
    assert route_q["instructions"].startswith("Decide who should handle")  # instruções em inglês
    # 2ª: conferência dos parâmetros obrigatórios (padrão "stated")
    stated = next(c for c in calls if any(k.startswith("stated_") for k in c["questions"]))
    assert (
        len(stated["questions"]) == 2 and stated["state"]["latest_user_message"] == "some 20 com 22"
    )
    # o LLM só extraiu argumentos (JSON) e sintetizou; não decidiu a rota
    assert chat.json_flags == [True, False]
    assert "Schema dos argumentos" in chat.calls[0][0].content
    names = [s["name"] for s in result.spans]
    assert "jev: rota" in names and "jev: parâmetros de calc/somar" in names
    assert "llm: argumentos de calc/somar" in names and "tool: calc/somar" in names


async def test_jev_knowledge_answer_uses_llm_with_citations(catalog):
    jev = scripted_jev(route_script("knowledge", kb=0.92))
    chat = ScriptedChat(["Abrimos aos sábados, das 9h às 14h [1]."])
    engine = RouterEngine(
        _profile(), chat=chat, connectors=catalog, retriever=await _retriever(), jev=jev
    )
    result = await engine.handle("vocês abrem no sábado?")
    assert (result.route, result.decided_by) == ("direct", "jev")
    assert [s.document for s in result.sources] == ["horarios.md"]
    assert "[1] (base: faq" in chat.calls[0][0].content
    assert "trechos respondem: 0.92" in result.reason


async def test_jev_says_passages_do_not_answer(catalog):
    jev = scripted_jev(route_script("knowledge", kb=0.05))
    engine = RouterEngine(
        _profile(), chat=OfflineChat(), connectors=catalog, retriever=await _retriever(), jev=jev
    )
    result = await engine.handle("qual o cnpj da empresa?")
    assert result.route == "direct" and "Não encontrei" in result.answer and result.sources == []


async def test_jev_missing_parameter_becomes_clarify_without_llm_invention(catalog):
    # o LLM "inventa" b; o Jev diz que o usuário não informou: vira pergunta
    jev = scripted_jev(route_script("calc/somar", stated_1=0.03))
    chat = ScriptedChat(['{"arguments": {"a": 3, "b": 7}}'])
    engine = RouterEngine(_profile(), chat=chat, connectors=catalog, jev=jev)
    result = await engine.handle("some 3 com outro número")
    assert result.route == "clarify" and result.decided_by == "jev"
    assert "segunda parcela da soma" in result.answer
    assert result.arguments == {"a": 3.0}
    assert any("descartei b" in w for w in result.warnings)


async def test_guarded_parameter_is_not_reinvented_when_the_profile_does_not_ask(catalog):
    # sem perguntas no perfil, escalar para o LLM traria b de volta (inventado);
    # o roteador diz o que falta e não chama a tool
    jev = scripted_jev(route_script("calc/somar", stated_1=0.03))
    chat = ScriptedChat(
        [
            '{"arguments": {"a": 3, "b": 7}}',
            '{"action": "tool", "connector": "calc", "tool": "somar", "arguments": {"a": 3, "b": 7}}',
        ]
    )
    engine = RouterEngine(_profile(allow_clarify=False), chat=chat, connectors=catalog, jev=jev)
    result = await engine.handle("some 3 com outro número")
    assert result.route == "direct" and result.decided_by == "jev"
    assert result.answer.startswith("Para seguir com isso, preciso de:")
    assert "segunda parcela da soma" in result.answer
    assert result.tool is None and len(chat.calls) == 1  # o LLM só extraiu os argumentos


async def test_low_confidence_escalates_to_llm(catalog):
    jev = scripted_jev(route_script("calc/somar", confidence=0.3))
    chat = ScriptedChat(['{"action": "answer", "answer": "Posso ajudar com somas e câmbio."}'])
    engine = RouterEngine(_profile(decision_threshold=0.6), chat=chat, connectors=catalog, jev=jev)
    result = await engine.handle("hmm")
    assert result.route == "direct" and result.decided_by == "llm"
    assert "escalado: confiança do Jev 0.30 < 0.60" in result.reason
    assert any("decisão escalada" in w for w in result.warnings)
    assert any(s["name"] == "escalada" for s in result.spans)


async def test_jev_outage_escalates_and_offline_still_answers(catalog):
    def down(state, questions):
        raise RuntimeError("fora do ar")

    jev = scripted_jev(down)
    engine = RouterEngine(
        _profile(), chat=OfflineChat(), connectors=catalog, retriever=await _retriever(), jev=jev
    )
    result = await engine.handle("preciso somar 10 e 32")
    assert (result.route, result.decided_by, result.answer) == (
        "tool",
        "heuristic",
        "resultado: 42",
    )
    assert any("Jev indisponível" in w for w in result.warnings)


async def test_jev_out_of_scope(catalog):
    jev = scripted_jev(route_script("out_of_scope"))
    chat = ScriptedChat(["Não consigo ajudar com isso por aqui."])
    engine = RouterEngine(_profile(), chat=chat, connectors=catalog, jev=jev)
    result = await engine.handle("me conta uma piada")
    assert result.route == "direct" and "fora do escopo" in result.reason
    assert "fora do que este assistente atende" in chat.calls[0][0].content
