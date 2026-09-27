from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest
from fastapi.testclient import TestClient

from switchboard.agents import AgentCatalog
from switchboard.secrets import SecretBox
from switchboard.storage import Database, repo, seed_demo
from switchboard.storage.orm import RouterProfile
from switchboard.testing import inproc_connector
from switchboard_agents import chamados, credito
from switchboard_router.app import create_app
from switchboard_router.runtime import DbRuntime
from switchboard_router.settings import Settings

KNOWLEDGE_DIR = Path(__file__).resolve().parents[3] / "examples" / "knowledge"
SERVERS = {"credito": credito.server, "chamados": chamados.server}


@pytest.fixture
def database(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'router.db'}")
    db.init()
    anyio.run(
        lambda: seed_demo(
            db,
            knowledge_dir=KNOWLEDGE_DIR,
            agent_urls={"credito": "http://credito/mcp", "chamados": "http://chamados/mcp"},
        )
    )
    chamados.reset()
    yield db


def make_client(db: Database, **settings_kwargs) -> TestClient:
    settings = Settings(database_url=db.url, config_ttl_s=0, **settings_kwargs)
    runtime = DbRuntime(
        db,
        SecretBox(None),
        config_ttl_s=0,
        catalog=AgentCatalog(connector=inproc_connector(SERVERS)),
    )
    return TestClient(create_app(settings, runtime=runtime))


def test_health_and_models(database):
    with make_client(database) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").status_code == 200
        data = client.get("/v1/models").json()
        assert data["object"] == "list" and [m["id"] for m in data["data"]] == ["default"]


def test_chat_direct_answer_and_trace_is_recorded(database):
    with make_client(database) as client:
        resp = client.post(
            "/v1/chat", json={"message": "Qual o horário de atendimento aos sábados?"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["route"] == "direct" and "9h às 14h" in body["answer"]
        assert body["sources"][0]["kb"] == "manual-atendimento"
        assert [s["name"] for s in body["steps"]] == ["rag", "descoberta_mcp", "decisao"]
    with database.session() as s:
        [trace] = repo.query_traces(s)
        assert trace.id == body["trace_id"] and trace.route == "direct"


def test_chat_delegates_to_mcp_agent(database):
    with make_client(database) as client:
        body = client.post(
            "/v1/chat",
            json={
                "profile": "default",
                "message": "Simule um empréstimo de R$ 30 mil em 12 meses a 2% ao mês",
            },
        ).json()
    assert (body["route"], body["agent"], body["tool"]) == (
        "delegated",
        "credito",
        "simular_financiamento",
    )
    assert body["arguments"] == {"valor": 30000.0, "prazo_meses": 12, "taxa_mensal_percentual": 2.0}
    assert "Parcela mensal: R$ 2.836,79" in body["answer"]


def test_chat_multi_turn_clarify_then_delegate(database):
    with make_client(database) as client:
        first = client.post("/v1/chat", json={"message": "quero abrir um chamado"}).json()
        assert first["route"] == "clarify"
        history = [
            {"role": "user", "content": "quero abrir um chamado"},
            {"role": "assistant", "content": first["answer"]},
            {
                "role": "user",
                "content": "o aplicativo fecha sozinho ao abrir o extrato, prioridade alta",
            },
        ]
        second = client.post("/v1/chat", json={"messages": history}).json()
    assert second["route"] == "delegated" and second["tool"] == "abrir_chamado"
    assert second["arguments"]["prioridade"] == "alta"
    assert "CH-1001" in second["answer"]


def test_openai_compatible_endpoint(database):
    with make_client(database) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "switchboard/default",
                "messages": [
                    {"role": "user", "content": "Qual o prazo máximo do empréstimo pessoal?"}
                ],
            },
        )
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert "60 meses" in body["choices"][0]["message"]["content"]
        assert body["switchboard"]["route"] == "direct"

        missing = client.post(
            "/v1/chat/completions",
            json={"model": "nao-existe", "messages": [{"role": "user", "content": "oi"}]},
        )
        assert missing.status_code == 404 and missing.json()["error"]["code"] == "model_not_found"

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "default",
                "stream": True,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "oi"}]}],
            },
        ) as stream:
            lines = [line for line in stream.iter_lines() if line]
    assert lines[-1] == "data: [DONE]"
    first = json.loads(lines[0].removeprefix("data: "))
    assert (
        first["object"] == "chat.completion.chunk"
        and "Olá" in first["choices"][0]["delta"]["content"]
    )
    assert json.loads(lines[1].removeprefix("data: "))["choices"][0]["finish_reason"] == "stop"


def test_agents_endpoint_reports_discovery(database):
    with make_client(database) as client:
        body = client.get("/v1/agents").json()
    agents = {a["name"]: a for a in body["agents"]}
    assert agents["credito"]["status"] == "online"
    assert {t["name"] for t in agents["chamados"]["tools"]} == {
        "abrir_chamado",
        "consultar_chamado",
        "listar_chamados",
    }


def test_api_keys_are_enforced(database):
    with make_client(database, api_keys="chave-1, chave-2") as client:
        assert client.post("/v1/chat", json={"message": "oi"}).status_code == 401
        assert (
            client.post(
                "/v1/chat", json={"message": "oi"}, headers={"Authorization": "Bearer chave-2"}
            ).status_code
            == 200
        )
        assert client.get("/v1/models", headers={"x-api-key": "chave-1"}).status_code == 200
        assert client.get("/healthz").status_code == 200  # health continua aberto


def test_config_changes_are_picked_up_without_restart(database):
    with make_client(database) as client:
        assert client.post("/v1/chat", json={"message": "oi"}).status_code == 200
        with database.session() as s:
            profile = s.query(RouterProfile).one()
            profile.enabled = False
        resp = client.post("/v1/chat", json={"message": "oi"})
        assert resp.status_code == 404 and "desabilitado" in resp.json()["detail"]


def test_validation_errors(database):
    with make_client(database) as client:
        assert client.post("/v1/chat", json={}).status_code == 422
        assert (
            client.post("/v1/chat", json={"profile": "outro", "message": "oi"}).status_code == 404
        )


def test_yaml_mode(tmp_path):
    config = tmp_path / "switchboard.yaml"
    config.write_text(
        f"""
models: [{{name: offline, provider: offline}}]
knowledge_bases: [{{name: manual, paths: ["{KNOWLEDGE_DIR.as_posix()}"]}}]
profiles: [{{name: default, model: offline, knowledge_bases: [manual], min_score: 0.1}}]
""",
        encoding="utf-8",
    )
    with TestClient(create_app(Settings(config_file=str(config)))) as client:
        body = client.post("/v1/chat", json={"message": "Qual o e-mail da ouvidoria?"}).json()
        assert body["route"] == "direct" and "ouvidoria@acme.example" in body["answer"]
        assert client.get("/v1/models").json()["data"][0]["id"] == "default"
