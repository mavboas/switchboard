from __future__ import annotations

from pathlib import Path

import pytest

from switchboard.errors import ConfigError, IngestError, SecretError
from switchboard.llm.base import Usage
from switchboard.routing.types import RouterResult, SourceRef, TraceStep
from switchboard.secrets import SecretBox
from switchboard.storage import KnowledgeService, repo, seed_demo
from switchboard.storage.db import normalize_url, redact_url
from switchboard.storage.orm import Agent, LlmModel

BOX = SecretBox("teste")
KNOWLEDGE_DIR = Path(__file__).resolve().parents[3] / "examples" / "knowledge"


def test_url_helpers():
    assert normalize_url("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert (
        redact_url("postgresql+psycopg://u:segredo@h:5432/db")
        == "postgresql+psycopg://u:***@h:5432/db"
    )


def test_models_agents_profiles_crud(db):
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
        agent = repo.save_agent(
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
            agent.allowed_tools == ["somar", "subtrair"]
            and agent.auth_token == "env:MCP_CALC_TOKEN"
        )
        kb = repo.save_knowledge_base(s, {"name": "faq"})
        profile = repo.save_profile(
            s,
            {
                "name": "default",
                "model_id": model.id,
                "agent_ids": [agent.id],
                "kb_ids": [kb.id],
                "top_k": 3,
            },
        )
        ids = (model.id, agent.id, profile.id)

    with db.session() as s:
        resolved = repo.resolve_profile(s, "default")
        assert resolved.model.name == "gpt" and BOX.open(resolved.model.api_key) == "sk-1"
        assert [a.name for a in resolved.agents] == ["calc"]
        assert resolved.spec.knowledge_bases == ["faq"] and resolved.spec.top_k == 3
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
        # agente desabilitado some do perfil resolvido
        s.get(Agent, ids[1]).enabled = False

    with db.session() as s:
        assert repo.resolve_profile(s, "default").agents == []
        assert repo.enabled_profile_names(s) == ["default"]
        with pytest.raises(ConfigError, match="em uso"):
            repo.delete_model(s, ids[0])
        with pytest.raises(ConfigError, match="já existe"):
            repo.save_agent(s, {"name": "calc", "url": "http://outro/mcp"}, box=BOX)
        with pytest.raises(ConfigError, match="dados inválidos"):
            repo.save_agent(s, {"name": "Nome Ruim", "url": "ftp://x"}, box=BOX)
        with pytest.raises(SecretError):
            repo.save_model(s, {"name": "x", "model": "y", "api_key": "sk"}, box=SecretBox(None))

    with db.session() as s:
        repo.delete_profile(s, ids[2])
        repo.delete_model(s, ids[0])
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
        steps=[TraceStep("rag", 1.5, {"trechos": 1})],
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
        assert repo.query_traces(s, route="delegated") == []
        assert repo.route_counts(s) == {"direct": 1}


async def test_seed_demo_is_idempotent(db):
    urls = {"credito": "http://localhost:8101/mcp", "chamados": "http://localhost:8102/mcp"}
    assert await seed_demo(db, knowledge_dir=KNOWLEDGE_DIR, agent_urls=urls)
    assert not await seed_demo(db, knowledge_dir=KNOWLEDGE_DIR, agent_urls=urls)
    with db.session() as s:
        resolved = repo.resolve_profile(s, "default")
        assert resolved.model.provider == "offline"
        assert sorted(a.name for a in resolved.agents) == ["chamados", "credito"]
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
