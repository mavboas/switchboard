from __future__ import annotations

import json

import pytest

from switchboard.agents import AgentCatalog
from switchboard.config import AgentSpec, EmbedderSpec, KnowledgeBaseSpec, ModelSpec, ProfileSpec
from switchboard.embeddings import HashingEmbedder
from switchboard.errors import AgentError, LLMError
from switchboard.llm import OfflineChat
from switchboard.rag import MemoryRetriever
from switchboard.routing import ResolvedProfile, RouterEngine
from switchboard.testing import ScriptedChat as FakeChat
from switchboard.testing import inproc_connector

CALC = AgentSpec(name="calc", url="http://calc.local/mcp")
ECHO = AgentSpec(name="echo", url="http://echo.local/mcp", allowed_tools=["repetir"])
GHOST = AgentSpec(name="fantasma", url="http://fantasma.local/mcp")


# --------------------------------------------------------------------------
# catálogo MCP


async def test_discover_lists_tools_and_marks_offline(catalog):
    infos = {i.name: i for i in await catalog.discover([CALC, GHOST])}
    calc = infos["calc"]
    assert calc.status == "online"
    assert {t.name for t in calc.tools} == {"somar", "converter_moeda", "falhar"}
    assert calc.description == "Calculadora de testes."  # veio das instructions do servidor MCP
    somar = calc.tool("somar")
    assert somar.input_schema["required"] == ["a", "b"]
    assert infos["fantasma"].status == "offline" and "fora do ar" in infos["fantasma"].error


async def test_discover_skips_disabled_and_caches(servers):
    calls = []
    base = inproc_connector(servers)

    def counting(spec, token):
        calls.append(spec.name)
        return base(spec, token)

    catalog = AgentCatalog(connector=counting, ttl_s=60)
    disabled = CALC.model_copy(update={"enabled": False})
    assert await catalog.discover([disabled]) == []
    await catalog.discover([CALC])
    await catalog.discover([CALC])
    assert calls == ["calc"]
    await catalog.discover([CALC], refresh=True)
    assert calls == ["calc", "calc"]
    catalog.invalidate("calc")
    await catalog.discover([CALC])
    assert calls == ["calc", "calc", "calc"]


async def test_allowlist_filters_tools_and_blocks_calls(catalog):
    [info] = await catalog.discover([ECHO])
    assert [t.name for t in info.tools] == ["repetir"]
    restricted = CALC.model_copy(update={"allowed_tools": ["somar"]})
    with pytest.raises(AgentError, match="não está liberada"):
        await catalog.call(restricted, "falhar", {"motivo": "x"})


async def test_call_tool_success_and_tool_error(catalog):
    ok = await catalog.call(CALC, "somar", {"a": 2, "b": 3})
    assert ok.text == "resultado: 5" and not ok.is_error
    bad = await catalog.call(CALC, "falhar", {"motivo": "teste"})
    assert bad.is_error and "falhei: teste" in bad.text
    with pytest.raises(AgentError, match="fantasma"):
        await catalog.call(GHOST, "x", {})


# --------------------------------------------------------------------------
# motor


def _profile(model: ModelSpec, agents=(CALC,), kbs=("faq",), **kw) -> ResolvedProfile:
    spec = ProfileSpec(
        name="teste",
        model=model.name,
        agents=[a.name for a in agents],
        knowledge_bases=list(kbs),
        min_score=0.05,
        **kw,
    )
    return ResolvedProfile(spec=spec, model=model, agents=list(agents))


async def _retriever() -> MemoryRetriever:
    retriever = MemoryRetriever()
    retriever.register(KnowledgeBaseSpec(name="faq", embedder=EmbedderSpec()), HashingEmbedder())
    await retriever.add_text(
        "faq", "horarios.md", "## Horários\n\nAtendemos aos sábados das 9h às 14h."
    )
    await retriever.add_text("faq", "boletos.md", "## Boletos\n\nO boleto vence todo dia 10.")
    return retriever


LLM = ModelSpec(name="llm", model="fake")


async def test_engine_direct_answer_with_sources(catalog):
    chat = FakeChat(
        ['{"action": "answer", "answer": "Sim, das 9h às 14h.", "sources": [1], "reason": "faq"}']
    )
    engine = RouterEngine(_profile(LLM), chat=chat, catalog=catalog, retriever=await _retriever())
    result = await engine.handle("Vocês abrem no sábado?")
    assert result.route == "direct"
    assert result.answer == "Sim, das 9h às 14h."
    assert [s.document for s in result.sources] == ["horarios.md"]
    assert [s.name for s in result.steps] == ["rag", "descoberta_mcp", "decisao"]
    # o prompt levou o catálogo real e os trechos numerados
    system = chat.calls[0][0].content
    assert '"tool": "somar"' in system and "[1] (base: faq, documento: horarios.md" in system
    assert chat.json_flags == [True]
    assert result.usage.input_tokens == 10


async def test_engine_delegates_and_synthesizes(catalog):
    chat = FakeChat(
        [
            '{"action": "delegate", "agent": "calc", "tool": "somar", "arguments": {"a": "2", "b": 40}, "reason": "conta"}',
            "A soma dá 42.",
        ]
    )
    engine = RouterEngine(_profile(LLM, kbs=()), chat=chat, catalog=catalog)
    result = await engine.handle([{"role": "user", "content": "quanto é 2 + 40?"}])
    assert (result.route, result.agent, result.tool) == ("delegated", "calc", "somar")
    assert result.arguments == {"a": 2.0, "b": 40.0}
    assert result.answer == "A soma dá 42."
    delegation = next(s for s in result.steps if s.name == "delegacao_mcp")
    assert delegation.detail["resultado"] == "resultado: 42"
    synthesis_prompt = chat.calls[1][1].content
    assert "resultado: 42" in synthesis_prompt and "quanto é 2 + 40?" in synthesis_prompt


async def test_engine_without_synthesis_returns_agent_text(catalog):
    chat = FakeChat(
        ['{"action": "delegate", "agent": "calc", "tool": "somar", "arguments": {"a": 1, "b": 1}}']
    )
    engine = RouterEngine(_profile(LLM, kbs=(), synthesize=False), chat=chat, catalog=catalog)
    result = await engine.handle("1+1")
    assert result.answer == "resultado: 2" and len(chat.calls) == 1


async def test_engine_repairs_invalid_decision(catalog):
    chat = FakeChat(
        [
            '{"action": "delegate", "agent": "calc", "tool": "multiplicar", "arguments": {}}',
            '{"action": "delegate", "agent": "calc", "tool": "somar", "arguments": {"a": 3, "b": 4}}',
            "Deu 7.",
        ]
    )
    engine = RouterEngine(_profile(LLM, kbs=()), chat=chat, catalog=catalog)
    result = await engine.handle("3 vezes 4? quer dizer, some")
    assert result.route == "delegated" and result.answer == "Deu 7."
    repair = chat.calls[1][-1].content
    assert "multiplicar" in repair and "opções" in repair


async def test_engine_missing_arguments_become_clarify(catalog):
    reply = '{"action": "delegate", "agent": "calc", "tool": "somar", "arguments": {"a": 3}}'
    chat = FakeChat([reply, reply])
    engine = RouterEngine(_profile(LLM, kbs=()), chat=chat, catalog=catalog)
    result = await engine.handle("some 3 com outro número")
    assert result.route == "clarify"
    assert "segunda parcela da soma" in result.answer


async def test_engine_llm_failure_falls_back_to_offline(catalog):
    chat = FakeChat([LLMError("provedor fora do ar")])
    engine = RouterEngine(_profile(LLM), chat=chat, catalog=catalog, retriever=await _retriever())
    result = await engine.handle("Vocês abrem no sábado?")
    assert result.route == "direct"
    assert "sábados" in result.answer
    assert any("decisor offline" in w for w in result.warnings)
    assert any(s.name == "fallback_offline" for s in result.steps)


async def test_engine_tool_error_is_reported(catalog):
    chat = FakeChat(
        [
            '{"action": "delegate", "agent": "calc", "tool": "falhar", "arguments": {"motivo": "x"}}',
            "Não deu certo, tente de novo.",
        ]
    )
    engine = RouterEngine(_profile(LLM, kbs=()), chat=chat, catalog=catalog)
    result = await engine.handle("teste de falha")
    assert result.route == "delegated"
    assert any("retornou erro" in w for w in result.warnings)
    assert "Status: ERRO" in chat.calls[1][1].content


async def test_engine_offline_agent_is_left_out_of_prompt(catalog):
    chat = FakeChat(['{"action": "answer", "answer": "ok"}'])
    engine = RouterEngine(_profile(LLM, agents=(CALC, GHOST), kbs=()), chat=chat, catalog=catalog)
    result = await engine.handle("oi")
    assert "fantasma" not in chat.calls[0][0].content
    assert any("fantasma indisponível" in w for w in result.warnings)


async def test_engine_offline_mode_end_to_end(catalog):
    offline = ModelSpec(name="off", provider="offline")
    engine = RouterEngine(
        _profile(offline), chat=OfflineChat(), catalog=catalog, retriever=await _retriever()
    )
    delegated = await engine.handle("preciso somar 10 e 32")
    assert (delegated.route, delegated.tool, delegated.answer) == (
        "delegated",
        "somar",
        "resultado: 42",
    )
    clarify = await engine.handle("quero converter moeda")
    assert clarify.route == "clarify" and "valor em reais" in clarify.answer
    follow_up = await engine.handle(
        [
            {"role": "user", "content": "quero converter moeda"},
            {"role": "assistant", "content": clarify.answer},
            {"role": "user", "content": "500 reais para eur"},
        ]
    )
    assert follow_up.route == "delegated" and follow_up.arguments == {
        "valor": 500.0,
        "moeda": "eur",
    }
    knowledge = await engine.handle("vocês abrem no sábado?")
    assert knowledge.route == "direct" and knowledge.sources[0].document == "horarios.md"
    assert json.loads(json.dumps(knowledge.to_dict()))["route"] == "direct"


async def test_engine_requires_user_message(catalog):
    engine = RouterEngine(_profile(LLM, kbs=()), chat=FakeChat([]), catalog=catalog)
    result = await engine.handle([{"role": "system", "content": "x"}])
    assert result.route == "error" and result.error
