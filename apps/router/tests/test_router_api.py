from __future__ import annotations

import json
import time
from pathlib import Path

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient

from switchboard.connectors import ConnectorCatalog
from switchboard.secrets import SecretBox
from switchboard.storage import Database, repo, seed_demo
from switchboard.storage.orm import LlmModel, RouterProfile
from switchboard.testing import HostRoutingTransport, inproc_connector
from switchboard_agents import analise, chamados, credito, risco
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
            connector_urls={"credito": "http://credito/mcp", "chamados": "http://chamados/mcp"},
            agent_urls={"analise-credito": "http://analise.test", "risco": "http://risco.test"},
        )
    )
    chamados.reset()
    yield db


def make_client(
    db: Database, *, analise_delay: float = 0.05, risco_delay: float = 0.05, **settings_kwargs
) -> TestClient:
    """Router com os agentes A2A de exemplo (SDK oficial) em processo.

    Um transporte httpx roteia por host: o router fala com ``analise.test`` e
    ``risco.test``, e os agentes mandam push notifications para ``router.test``
    — que é o próprio app do router.
    """
    settings_kwargs.setdefault("allowed_hosts", "testserver")
    settings_kwargs.setdefault("public_url", "http://router.test")
    settings = Settings(
        database_url=db.url, config_ttl_s=0, supervisor_tick_s=0.05, **settings_kwargs
    )
    transport = HostRoutingTransport()
    http = httpx.AsyncClient(transport=transport)
    for host, module, delay in (
        ("analise.test", analise, analise_delay),
        ("risco.test", risco, risco_delay),
    ):
        agent = module.build(f"http://{host}", delay_s=delay)
        agent.http = http
        transport.mount(host, agent.build_app())
    runtime = DbRuntime(
        db,
        SecretBox(None),
        config_ttl_s=0,
        public_url=settings.public_url,
        supervisor_tick_s=0.05,
        connectors=ConnectorCatalog(connector=inproc_connector(SERVERS)),
        http=http,
    )
    runtime.contracts.poll_min_s = 0.05
    app = create_app(settings, runtime=runtime)
    app.state.test_transport = transport  # os testes leem o que foi enviado aos agentes
    transport.mount("router.test", app)
    return TestClient(app)


def wait_run(
    client: TestClient, run_id: str, status: str = "completed", timeout: float = 5.0
) -> dict:
    end = time.monotonic() + timeout
    while True:
        body = client.get(f"/v1/runs/{run_id}").json()
        if body["status"] == status or time.monotonic() > end:
            return body
        time.sleep(0.05)


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
        assert body["status"] == "completed" and body["decided_by"] == "heuristic"
        assert body["sources"][0]["kb"] == "manual-atendimento"
        assert [s["name"] for s in body["steps"]] == ["rag", "descoberta", "decisao"]
        assert body["spans"][0]["kind"] == "pedido" and body["spans"][0]["parent_id"] is None
    with database.session() as s:
        [trace] = repo.query_traces(s)
        assert (
            trace.id == body["trace_id"] and trace.route == "direct" and trace.status == "completed"
        )
        details = repo.run_details(s, trace.id)
        assert {sp["kind"] for sp in details["spans"]} >= {"pedido", "rag", "descoberta", "decisao"}


def test_chat_calls_mcp_tool(database):
    with make_client(database) as client:
        body = client.post(
            "/v1/chat",
            json={
                "profile": "default",
                "message": "Simule um empréstimo de R$ 30 mil em 12 meses a 2% ao mês",
            },
        ).json()
    assert (body["route"], body["agent"], body["tool"]) == (
        "tool",
        "credito",
        "simular_financiamento",
    )
    assert body["arguments"] == {"valor": 30000.0, "prazo_meses": 12, "taxa_mensal_percentual": 2.0}
    assert "Parcela mensal: R$ 2.836,79" in body["answer"]
    assert "links" not in body and body["contracts"] == []


def test_chat_multi_turn_clarify_then_tool(database):
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
    assert second["route"] == "tool" and second["tool"] == "abrir_chamado"
    assert second["arguments"]["prioridade"] == "alta"
    assert "CH-1001" in second["answer"]


def test_delegation_completes_within_the_wait(database):
    with make_client(database) as client:
        resp = client.post(
            "/v1/chat",
            json={
                "message": "Avalie o risco de fraude da operação de R$ 80 mil do cliente Maria Souza"
            },
        )
        assert resp.status_code == 200, resp.json()
        body = resp.json()
        assert (body["route"], body["status"], body["agent"], body["tool"]) == (
            "delegated",
            "completed",
            "risco",
            "avaliar_risco",
        )
        assert "risco (avaliar_risco)" in body["answer"] and "Risco " in body["answer"]
        [contract] = body["contracts"]
        assert contract["state"] == "concluido" and contract["kind"] == "completo"
        assert body["links"]["run"] == f"/v1/runs/{body['run_id']}"
        details = wait_run(client, body["run_id"])
        [full] = details["contracts"]
        assert full["reply_mode"] == "push" and full["output"]["risco"] in (
            "baixo",
            "medio",
            "alto",
        )
        sources = {e["source"] for e in full["events"]}
        assert "push" in sources  # o agente (SDK) avisou o router
        kinds = {sp["kind"] for sp in details["spans"]}
        assert {"pedido", "delegacao", "espera", "contrato", "consolidacao"} <= kinds
        contract_span = next(sp for sp in details["spans"] if sp["kind"] == "contrato")
        delegation = next(sp for sp in details["spans"] if sp["kind"] == "delegacao")
        assert contract_span["parent_id"] == delegation["id"] and contract_span["status"] == "ok"


def test_long_task_returns_202_and_finishes_in_background(database):
    with make_client(database, analise_delay=0.6) as client:
        resp = client.post(
            "/v1/chat",
            json={
                "message": "Analise uma proposta de crédito de 80 mil para Maria Souza em 24 meses com renda de 12 mil",
                "wait_s": 0.05,
            },
        )
        assert resp.status_code == 202, resp.json()
        body = resp.json()
        assert body["status"] == "pending" and body["agent"] == "analise-credito"
        assert "Encaminhei seu pedido" in body["answer"]
        run_id = body["run_id"]
        with client.stream("GET", f"/v1/runs/{run_id}/events") as stream:
            events = []
            for line in stream.iter_lines():
                if line.startswith("event: "):
                    events.append(line.removeprefix("event: "))
                if line.startswith("data: ") and events and events[-1] == "done":
                    break
        assert events[0] == "run" and events[-1] == "done"
        details = wait_run(client, run_id)
        assert details["status"] == "completed" and "Proposta aprovada" in details["answer"]
        assert details["contracts"][0]["output"]["decisao"] == "aprovado"
    with database.session() as s:
        row = repo.trace_to_dict(
            s.get(__import__("switchboard.storage.orm", fromlist=["Trace"]).Trace, run_id)
        )
        assert row["status"] == "completed" and row["route"] == "delegated"


def test_agent_asks_for_input_and_user_reply_continues_the_contract(database):
    with make_client(database) as client:
        body = client.post(
            "/v1/chat",
            json={
                "message": "Analise uma proposta de crédito de 800 mil para João Lima em 120 meses com renda de 60 mil",
                "wait_s": 3,
            },
        ).json()
        assert body["status"] == "needs_input", body
        assert "garantia" in body["answer"]
        run_id = body["run_id"]
        reply = client.post(
            "/v1/chat", json={"run_id": run_id, "message": "sim, imóvel quitado", "wait_s": 3}
        )
        assert reply.status_code == 200, reply.json()
        final = reply.json()
        assert final["run_id"] == run_id and final["status"] == "completed", final
        details = wait_run(client, run_id)
        [contract] = details["contracts"]
        assert contract["output"]["garantia"] == "imóvel"
        states = [e["state"] for e in contract["events"] if e["kind"] == "state"]
        assert (
            states[:3] == ["proposto", "ativo", "aguardando_entrada"] and states[-1] == "concluido"
        )
        assert (
            client.post("/v1/chat", json={"run_id": "0" * 32, "message": "oi"}).status_code == 404
        )


def test_cancel_run(database):
    with make_client(database, analise_delay=5.0) as client:
        body = client.post(
            "/v1/chat",
            json={
                "message": "Analise uma proposta de crédito de 50 mil para Ana Costa em 12 meses com renda de 9 mil",
                "wait_s": 0,
            },
        ).json()
        assert body["status"] == "pending"
        canceled = client.post(f"/v1/runs/{body['run_id']}/cancel").json()
        assert [c["state"] for c in canceled["canceled"]] == ["cancelado"]
        details = wait_run(client, body["run_id"])
        assert details["status"] == "completed" and "cancelado" in details["answer"]
        assert client.post("/v1/runs/nao-existe/cancel").status_code == 404


def test_push_endpoint_requires_the_contract_token(database):
    with make_client(database, analise_delay=5.0) as client:
        body = client.post(
            "/v1/chat",
            json={
                "message": "Analise uma proposta de crédito de 50 mil para Ana Costa em 12 meses com renda de 9 mil",
                "wait_s": 0,
            },
        ).json()
        contract_id = body["contracts"][0]["id"]
        event = {"statusUpdate": {"taskId": "x", "status": {"state": "TASK_STATE_COMPLETED"}}}
        assert client.post(f"/a2a/push/{contract_id}", json=event).status_code == 401
        assert (
            client.post(
                f"/a2a/push/{contract_id}",
                json=event,
                headers={"X-A2A-Notification-Token": "chute"},
            ).status_code
            == 401
        )
        assert client.post("/a2a/push/ctr_nao_existe", json=event).status_code == 404
        # sem o token, o corpo nem é lido: 401 antes do parse e do limite de tamanho
        big = b'{"x": "' + b"a" * 2_000_000 + b'"}'
        assert client.post(f"/a2a/push/{contract_id}", content=big).status_code == 401
        # com o token do contrato (o que o router mandou ao agente no SendMessage)
        sent = [
            json.loads(r.content)
            for r in client.app.state.test_transport.sent
            if r.url.host == "analise.test" and r.method == "POST"
        ]
        [config] = [
            m["params"]["configuration"]["taskPushNotificationConfig"]
            for m in sent
            if m.get("method") == "SendMessage"
        ]
        auth = {"X-A2A-Notification-Token": config["token"]}
        assert (
            client.post(
                f"/a2a/push/{contract_id}",
                content=b"{",
                headers={**auth, "content-type": "application/json"},
            ).status_code
            == 400
        )
        assert client.post(f"/a2a/push/{contract_id}", content=big, headers=auth).status_code == 413

        def chunked():  # sem Content-Length: o teto vale durante a leitura
            for _ in range(40):
                yield b"a" * 65536

        assert (
            client.post(f"/a2a/push/{contract_id}", content=chunked(), headers=auth).status_code
            == 413
        )
        client.post(f"/v1/runs/{body['run_id']}/cancel")


def test_push_endpoint_is_open_to_agents_even_with_api_keys(database):
    with make_client(database, api_keys="k1") as client:
        # sem a chave da API: a rota de push não pede chave (autentica pelo token do contrato)
        assert client.post("/a2a/push/ctr_x", json={}).status_code == 404
        assert client.post("/v1/chat", json={"message": "oi"}).status_code == 401


def test_callbacks_need_allowed_hosts(database):
    with make_client(database) as client:
        resp = client.post(
            "/v1/chat", json={"message": "oi", "callback_url": "http://cliente/hook"}
        )
        assert resp.status_code == 422 and "desligados" in resp.json()["detail"]
    with make_client(database, callback_hosts="cliente") as client:
        bad = client.post("/v1/chat", json={"message": "oi", "callback_url": "http://outro/hook"})
        assert bad.status_code == 422
        ok = client.post("/v1/chat", json={"message": "oi", "callback_url": "http://cliente/hook"})
        assert ok.status_code == 200


def test_openai_compatible_endpoint(database):
    with make_client(database, analise_delay=0.5) as client:
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
        assert (
            body["switchboard"]["route"] == "direct"
            and body["switchboard"]["status"] == "completed"
        )

        pending = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": "Analise uma proposta de crédito de 80 mil para Maria Souza em 24 meses com renda de 12 mil",
                    }
                ],
                "switchboard": {"wait_s": 0},
            },
        ).json()
        assert pending["switchboard"]["status"] == "pending"
        assert "Encaminhei" in pending["choices"][0]["message"]["content"]
        assert pending["switchboard"]["links"]["events"].endswith("/events")

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
        wait_run(client, pending["switchboard"]["run_id"])
    assert lines[-1] == "data: [DONE]"
    first = json.loads(lines[0].removeprefix("data: "))
    assert (
        first["object"] == "chat.completion.chunk"
        and "Olá" in first["choices"][0]["delta"]["content"]
    )
    assert json.loads(lines[1].removeprefix("data: "))["choices"][0]["finish_reason"] == "stop"


def test_connectors_and_agents_endpoints(database):
    with make_client(database) as client:
        connectors = {c["name"]: c for c in client.get("/v1/connectors").json()["connectors"]}
        agents = {a["name"]: a for a in client.get("/v1/agents?refresh=true").json()["agents"]}
    assert connectors["credito"]["status"] == "online"
    assert {t["name"] for t in connectors["chamados"]["tools"]} == {
        "abrir_chamado",
        "consultar_chamado",
        "listar_chamados",
    }
    assert agents["analise-credito"]["status"] == "online" and agents["analise-credito"]["push"]
    [skill] = agents["risco"]["skills"]
    assert (
        skill["id"] == "avaliar_risco" and skill["contract"] == "completo" and skill["input_hash"]
    )


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
        assert client.get("/v1/runs/x").status_code == 401
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
        assert client.post("/v1/chat", json={"message": "oi", "wait_s": -1}).status_code == 422
        assert (
            client.post("/v1/chat", json={"profile": "outro", "message": "oi"}).status_code == 404
        )
        assert client.get("/v1/runs/nao-existe").status_code == 404
        assert client.get("/v1/runs/nao-existe/events").status_code == 404


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
    with TestClient(
        create_app(Settings(config_file=str(config), allowed_hosts="testserver"))
    ) as client:
        body = client.post("/v1/chat", json={"message": "Qual o e-mail da ouvidoria?"}).json()
        assert body["route"] == "direct" and "ouvidoria@acme.example" in body["answer"]
        assert client.get("/v1/models").json()["data"][0]["id"] == "default"
        assert (
            client.get(f"/v1/runs/{body['run_id']}").status_code == 404
        )  # sem delegação, sem execução guardada


def test_unreadable_model_secret_falls_back_to_offline(database):
    """Chave cifrada com outra chave mestra: o router atende em modo offline e avisa."""
    with database.session() as s:
        model = s.query(LlmModel).one()
        model.provider, model.model = "openai", "gpt-x"
        model.api_key = SecretBox("outra-chave-mestra").seal("sk-123")
    with make_client(database) as client:
        body = client.post(
            "/v1/chat", json={"message": "Qual o horário de atendimento aos sábados?"}
        ).json()
    assert body["route"] == "direct" and "9h às 14h" in body["answer"]
    assert any("modo offline" in w for w in body["warnings"])


def test_open_api_only_answers_known_hosts(database):
    """Sem chaves de API, o router só atende hosts conhecidos (contra DNS rebinding)."""
    with make_client(database, allowed_hosts="localhost,router") as client:
        evil = client.post("/v1/chat", json={"message": "oi"}, headers={"Host": "evil.example"})
        assert evil.status_code == 400
        assert (
            client.post(
                "/v1/chat", json={"message": "oi"}, headers={"Host": "localhost:8080"}
            ).status_code
            == 200
        )
        # o host da URL pública (push dos agentes) entra na lista sozinho
        assert (
            client.post("/a2a/push/ctr_x", json={}, headers={"Host": "router.test"}).status_code
            == 404
        )
        assert client.get("/healthz", headers={"Host": "10.0.0.7:8080"}).status_code == 200
    with make_client(database, allowed_hosts="localhost", api_keys="k1") as client:
        # com chave, qualquer host serve (a chave é a proteção)
        ok = client.post(
            "/v1/chat",
            json={"message": "oi"},
            headers={"Host": "api.empresa.com", "Authorization": "Bearer k1"},
        )
        assert ok.status_code == 200
