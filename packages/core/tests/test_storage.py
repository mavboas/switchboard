from __future__ import annotations

from pathlib import Path

import pytest

from switchboard.errors import ConfigError, IngestError, SecretError
from switchboard.llm.base import Usage
from switchboard.routing.types import RouterResult, SourceRef
from switchboard.secrets import SecretBox
from switchboard.storage import KnowledgeService, repo, seed_demo
from switchboard.storage.db import normalize_url, redact_url
from switchboard.storage.orm import Agent, Connector, LlmModel
from switchboard.tracing import Span, utcnow

BOX = SecretBox("teste")
KNOWLEDGE_DIR = Path(__file__).resolve().parents[3] / "examples" / "knowledge"


def test_url_helpers():
    assert normalize_url("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert (
        redact_url("postgresql+psycopg://u:segredo@h:5432/db")
        == "postgresql+psycopg://u:***@h:5432/db"
    )


def test_models_connectors_agents_profiles_crud(db):
    with db.session() as s:
        model = repo.save_model(
            s,
            {
                "name": "gpt",
                "provider": "openai",
                "model": "gpt-x",
                "api_key": "sk-1",
                "preset": "openai",
            },
            box=BOX,
        )
        assert model.api_key.startswith("enc:")
        jev = repo.save_model(
            s,
            {
                "name": "jev",
                "provider": "typesafe",
                "model": "jev-1.13.0",
                "api_key": "env:TYPESAFE_API_KEY",
            },
            box=BOX,
        )
        connector = repo.save_connector(
            s,
            {
                "name": "calc",
                "url": "http://calc/mcp",
                "allowed_tools": "somar, subtrair",
                "auth_token": "env:MCP_CALC_TOKEN",
            },
            box=BOX,
        )
        assert (
            connector.allowed_tools == ["somar", "subtrair"]
            and connector.auth_token == "env:MCP_CALC_TOKEN"
        )
        agent = repo.save_agent(
            s,
            {
                "name": "risco",
                "url": "http://risco:8202",
                "allowed_skills": "avaliar_risco",
                "auth_token": "segredo",
                "deadline_s": 90,
                "push": False,
            },
            box=BOX,
        )
        assert agent.auth_token.startswith("enc:") and agent.allowed_skills == ["avaliar_risco"]
        kb = repo.save_knowledge_base(s, {"name": "faq"})
        profile = repo.save_profile(
            s,
            {
                "name": "default",
                "model_id": model.id,
                "decision_model_id": jev.id,
                "decision_threshold": 0.7,
                "connector_ids": [connector.id],
                "agent_ids": [agent.id],
                "kb_ids": [kb.id],
                "top_k": 3,
                "wait_s": 2,
                "max_parallel": 2,
                "deadline_s": 120,
            },
        )
        ids = (model.id, connector.id, profile.id, agent.id, jev.id)
        # o LLM do roteador não pode ser um modelo de decisão, e vice-versa
        with pytest.raises(ConfigError, match="modelo de decisão"):
            repo.save_profile(s, {"name": "x", "model_id": jev.id})
        with pytest.raises(ConfigError, match="não é um modelo de decisão"):
            repo.save_profile(s, {"name": "x", "model_id": model.id, "decision_model_id": model.id})

    with db.session() as s:
        resolved = repo.resolve_profile(s, "default")
        assert resolved.model.name == "gpt" and BOX.open(resolved.model.api_key) == "sk-1"
        assert resolved.decision_model.provider == "typesafe"
        assert resolved.spec.decision_model == "jev" and resolved.spec.decision_threshold == 0.7
        assert [c.name for c in resolved.connectors] == ["calc"]
        [a2a] = resolved.agents
        assert (a2a.name, a2a.deadline_s, a2a.push) == ("risco", 90, False)
        assert BOX.open(a2a.auth_token) == "segredo"
        assert (resolved.spec.wait_s, resolved.spec.max_parallel, resolved.spec.deadline_s) == (
            2,
            2,
            120,
        )
        assert resolved.spec.knowledge_bases == ["faq"] and resolved.spec.top_k == 3
        assert repo.find_agent_spec(s, "risco").url == "http://risco:8202"
        # editar sem mandar a chave mantém a atual; clear remove
        repo.save_model(
            s, {"name": "gpt", "provider": "openai", "model": "gpt-y"}, box=BOX, model_id=ids[0]
        )
        assert BOX.open(s.get(LlmModel, ids[0]).api_key) == "sk-1"
        repo.save_model(
            s,
            {"name": "gpt", "provider": "openai", "model": "gpt-y", "clear_api_key": True},
            box=BOX,
            model_id=ids[0],
        )
        assert s.get(LlmModel, ids[0]).api_key is None
        # trocar o tipo de um modelo em uso quebraria o roteador
        with pytest.raises(ConfigError, match="LLM dos roteadores"):
            repo.save_model(
                s, {"name": "gpt", "provider": "typesafe", "model": "jev"}, box=BOX, model_id=ids[0]
            )
        # conector e agente desabilitados somem do perfil resolvido
        s.get(Connector, ids[1]).enabled = False
        s.get(Agent, ids[3]).enabled = False

    with db.session() as s:
        resolved = repo.resolve_profile(s, "default")
        assert resolved.connectors == [] and resolved.agents == []
        assert repo.enabled_profile_names(s) == ["default"]
        with pytest.raises(ConfigError, match="em uso"):
            repo.delete_model(s, ids[0])
        with pytest.raises(ConfigError, match="decide"):
            repo.delete_model(s, ids[4])
        with pytest.raises(ConfigError, match="já existe"):
            repo.save_connector(s, {"name": "calc", "url": "http://outro/mcp"}, box=BOX)
        with pytest.raises(ConfigError, match="dados inválidos"):
            repo.save_connector(s, {"name": "Nome Ruim", "url": "ftp://x"}, box=BOX)
        with pytest.raises(ConfigError, match="dados inválidos"):
            repo.save_agent(
                s, {"name": "a", "url": "http://x", "timeout_s": 0.0001, "deadline_s": -1}, box=BOX
            )
        with pytest.raises(SecretError):
            repo.save_model(s, {"name": "x", "model": "y", "api_key": "sk"}, box=SecretBox(None))

    with db.session() as s:
        repo.delete_profile(s, ids[2])
        repo.delete_model(s, ids[0])
        repo.delete_model(s, ids[4])
        repo.delete_connector(s, ids[1])
        repo.delete_agent(s, ids[3])
        assert repo.resolve_profile(s, "default") is None


async def test_knowledge_ingest_search_dedupe_reindex(db):
    with db.session() as s:
        kb_id = repo.save_knowledge_base(
            s, {"name": "faq", "chunk_size": 300, "chunk_overlap": 40}
        ).id
    service = KnowledgeService(db)
    try:
        doc_id, created = await service.add_document(
            kb_id, title="Horários", content="## Sábado\n\nAtendemos aos sábados das 9h às 14h."
        )
        assert created
        _, created_again = await service.add_document(
            kb_id, title="Cópia", content="## Sábado\n\nAtendemos aos sábados das 9h às 14h."
        )
        assert not created_again
        await service.add_document(
            kb_id, title="Boletos", content="O boleto vence todo dia 10 de cada mês."
        )
        hits = await service.search("abre no sábado?", ["faq"], top_k=2, min_score=0.0)
        assert hits[0].document == "Horários" and hits[0].section == "Sábado"
        assert hits[0].score > hits[-1].score
        assert await service.search("abre no sábado?", ["faq"], top_k=2, min_score=0.99) == []

        with db.session() as s:
            stats = repo.kb_stats(s)[kb_id]
            assert stats == {"documents": 2, "chunks": 2, "stale": 0}
            # trocar o embedder deixa os documentos "velhos" até reindexar
            repo.save_knowledge_base(s, {"name": "faq", "embedder_dim": 128}, kb_id=kb_id)
        with db.session() as s:
            assert repo.kb_stats(s)[kb_id]["stale"] == 2
        assert (
            await service.search("sábado", ["faq"], top_k=2, min_score=0.0) == []
        )  # dimensões diferentes
        assert await service.reindex(kb_id) == 2
        with db.session() as s:
            assert repo.kb_stats(s)[kb_id]["stale"] == 0
        assert (await service.search("sábado", ["faq"], top_k=1, min_score=0.0))[
            0
        ].document == "Horários"

        await service.delete_document(kb_id, doc_id)
        with pytest.raises(IngestError):
            await service.delete_document(kb_id, doc_id)
        with pytest.raises(IngestError):
            await service.add_document(kb_id, title="vazio", content="   ")
    finally:
        await service.aclose()


def test_traces_roundtrip(db):
    result = RouterResult(
        trace_id="abc123",
        profile="default",
        question="oi?",
        answer="olá",
        route="direct",
        model="offline",
        sources=[SourceRef(1, "faq", "doc", "7", 0.5, "trecho")],
        spans=[
            Span("a" * 16, "abc123", None, "pedido", "pedido", utcnow(), utcnow(), "ok").to_dict(),
            Span(
                "b" * 16, "abc123", "a" * 16, "rag", "rag", utcnow(), utcnow(), "ok", {"trechos": 1}
            ).to_dict(),
        ],
        usage=Usage(3, 4),
        warnings=["aviso"],
    )
    with db.session() as s:
        repo.save_trace(s, result)
    with db.session() as s:
        [row] = repo.query_traces(s, profile="default")
        data = repo.trace_to_dict(row)
        assert data["sources"][0]["document"] == "doc" and data["usage"] == {
            "input_tokens": 3,
            "output_tokens": 4,
        }
        assert data["status"] == "completed" and data["steps"][0]["name"] == "rag"
        assert repo.query_traces(s, route="delegated") == []
        assert repo.route_counts(s) == {"direct": 1}
        details = repo.run_details(s, "abc123")
        assert [sp["kind"] for sp in details["spans"]] == ["pedido", "rag"]
        assert details["contracts"] == []


async def test_seed_demo_is_idempotent(db):
    connectors = {"credito": "http://localhost:8101/mcp", "chamados": "http://localhost:8102/mcp"}
    agents = {"analise-credito": "http://localhost:8201", "risco": "http://localhost:8202"}
    assert await seed_demo(
        db, knowledge_dir=KNOWLEDGE_DIR, connector_urls=connectors, agent_urls=agents
    )
    assert not await seed_demo(
        db, knowledge_dir=KNOWLEDGE_DIR, connector_urls=connectors, agent_urls=agents
    )
    with db.session() as s:
        resolved = repo.resolve_profile(s, "default")
        assert resolved.model.provider == "offline"
        assert sorted(c.name for c in resolved.connectors) == ["chamados", "credito"]
        assert sorted(a.name for a in resolved.agents) == ["analise-credito", "risco"]
        stats = list(repo.kb_stats(s).values())[0]
        assert stats["documents"] == 3 and stats["chunks"] > 5
    service = KnowledgeService(db)
    try:
        hits = await service.search(
            "quais documentos preciso para pedir crédito?",
            ["manual-atendimento"],
            top_k=1,
            min_score=0.1,
        )
        assert "comprovante de renda" in hits[0].content
    finally:
        await service.aclose()


def test_database_mode(db):
    assert db.ping()
    if db.dialect == "postgresql":
        assert db.vector_mode == "pgvector"
    else:
        assert db.vector_mode == "json"


def test_extra_headers_are_sealed_and_masked(db):
    from switchboard.secrets import MASK

    with db.session() as s:
        m = repo.save_model(
            s,
            {
                "name": "h",
                "model": "x",
                "extra_headers": {"X-Chave": "segredo", "X-Env": "env:ACME_API_KEY"},
            },
            box=BOX,
        )
        model_id = m.id
        assert (
            m.extra_headers["X-Chave"].startswith("enc:")
            and m.extra_headers["X-Env"] == "env:ACME_API_KEY"
        )
    with db.session() as s:
        # a UI devolve a máscara no lugar do valor cifrado: o valor gravado é mantido
        repo.save_model(
            s,
            {"name": "h", "model": "x", "extra_headers": {"X-Chave": MASK}},
            box=BOX,
            model_id=model_id,
        )
        assert BOX.open(s.get(LlmModel, model_id).extra_headers["X-Chave"]) == "segredo"
    with db.session() as s:
        # sem chave mestra, cabeçalhos comuns continuam salváveis (em texto)
        m = repo.save_model(
            s,
            {"name": "sem", "model": "x", "extra_headers": {"HTTP-Referer": "https://app"}},
            box=SecretBox(None),
        )
        assert m.extra_headers == {"HTTP-Referer": "https://app"}
