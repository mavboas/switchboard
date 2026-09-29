from __future__ import annotations

import base64
import json
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from switchboard.a2a import A2AClient, AgentDirectory
from switchboard.connectors import ConnectorCatalog
from switchboard.secrets import SecretBox
from switchboard.storage import Database
from switchboard.storage.orm import LlmModel
from switchboard.testing import HostRoutingTransport, inproc_connector
from switchboard_agents import analise, chamados, credito, risco
from switchboard_console.app import create_app
from switchboard_console.ops import waterfall
from switchboard_console.settings import Settings
from switchboard_router.app import create_app as create_router
from switchboard_router.runtime import DbRuntime
from switchboard_router.settings import Settings as RouterSettings

KNOWLEDGE_DIR = Path(__file__).resolve().parents[3] / "examples" / "knowledge"
SERVERS = {"credito": credito.server, "chamados": chamados.server}
SECRET = "segredo-de-teste"


@pytest.fixture
def db_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'console.db'}"


def jev_transport(calls: list | None = None) -> httpx.MockTransport:
    """Um Jev falso: responde noul alto para qualquer pergunta."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if calls is not None:
            calls.append(
                {"url": str(request.url), "auth": request.headers.get("authorization"), **payload}
            )
        answers = {key: {"type": "noul", "noul": 0.97} for key in payload["questions"]}
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "answers": answers,
                "usage": {"input_tokens": 50, "output_tokens": 1},
            },
        )

    return httpx.MockTransport(handler)


class Network:
    """Router, agentes A2A de exemplo (SDK oficial) e conectores MCP, todos em processo.

    Um único transporte httpx roteia por host: o console fala com o router
    (``router``) e com os agentes (``analise.test``/``risco.test``); o router
    fala com os agentes, e os agentes mandam push para ``router.test``.
    """

    def __init__(self, db_url: str, *, analise_delay: float = 0.05):
        self.transport = HostRoutingTransport()
        self.http = httpx.AsyncClient(transport=self.transport)
        for host, module, delay in (
            ("analise.test", analise, analise_delay),
            ("risco.test", risco, 0.05),
        ):
            agent = module.build(f"http://{host}", delay_s=delay)
            agent.http = self.http
            self.transport.mount(host, agent.build_app())
        db = Database(db_url)
        db.init()
        self.runtime = DbRuntime(
            db,
            SecretBox(SECRET),
            config_ttl_s=0,
            public_url="http://router.test",
            supervisor_tick_s=0.05,
            connectors=ConnectorCatalog(connector=inproc_connector(SERVERS)),
            http=self.http,
        )
        self.runtime.contracts.poll_min_s = 0.05
        router_app = create_router(
            RouterSettings(
                database_url=db_url,
                allowed_hosts="router,router.test",
                public_url="http://router.test",
            ),
            runtime=self.runtime,
        )
        router_app.state.runtime = self.runtime  # o lifespan do router não roda via transporte
        self.transport.mount("router", router_app)
        self.transport.mount("router.test", router_app)


@contextmanager
def make_client(
    db_url: str, *, analise_delay: float = 0.05, jev_calls: list | None = None, **overrides
) -> Iterator[TestClient]:
    net = Network(db_url, analise_delay=analise_delay)
    settings = Settings(
        database_url=db_url,
        secret_key=SECRET,
        router_url="http://router",
        demo_knowledge_dir=str(KNOWLEDGE_DIR),
        demo_credito_url="http://credito/mcp",
        demo_chamados_url="http://chamados/mcp",
        demo_analise_url="http://analise.test",
        demo_risco_url="http://risco.test",
        **{"console_allowed_hosts": "console,localhost", **overrides},
    )
    app = create_app(
        settings,
        connectors=ConnectorCatalog(connector=inproc_connector(SERVERS)),
        agents=AgentDirectory(client=A2AClient(http=net.http)),
        http=net.http,
        jev_transport=jev_transport(jev_calls),
    )
    with TestClient(app, base_url="http://console") as client:
        # o supervisor de contratos roda no mesmo event loop do console (e dos agentes)
        client.portal.call(net.runtime.start)
        try:
            yield client
        finally:
            client.portal.call(net.runtime.contracts.stop)


def _ids(html: str, prefix: str) -> list[int]:
    return [int(x) for x in re.findall(rf'href="/{prefix}/(\d+)"', html)]


def _wait_run(
    client: TestClient, run_id: str, statuses: tuple[str, ...], timeout: float = 8.0
) -> dict:
    end = time.monotonic() + timeout
    while True:
        body = client.get(f"/playground/runs/{run_id}").json()
        if body.get("status") in statuses or time.monotonic() > end:
            return body
        time.sleep(0.05)


def test_all_pages_render_with_demo_data(db_url):
    chamados.reset()
    with make_client(db_url) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "Primeiros passos" in home.text and "pgvector" not in home.text  # sqlite usa json
        assert (
            "Conectores MCP" in home.text
            and "Agentes A2A" in home.text
            and "Contratos" in home.text
        )
        model_id = _ids(client.get("/models").text, "models")[0]
        connector_id = _ids(client.get("/connectors").text, "connectors")[0]
        agent_id = _ids(client.get("/agents").text, "agents")[0]
        kb_id = _ids(client.get("/knowledge").text, "knowledge")[0]
        profile_id = _ids(client.get("/profiles").text, "profiles")[0]
        pages = [
            "/models/new?preset=anthropic",
            "/models/new?preset=typesafe",
            f"/models/{model_id}",
            "/connectors/new",
            f"/connectors/{connector_id}",
            f"/connectors/{connector_id}/edit",
            "/agents/new",
            f"/agents/{agent_id}",
            f"/agents/{agent_id}/edit",
            "/knowledge/new",
            f"/knowledge/{kb_id}",
            f"/knowledge/{kb_id}/edit",
            "/profiles/new",
            f"/profiles/{profile_id}",
            "/playground",
            "/traces",
            "/traces?status=pending&route=delegated",
            "/contracts",
            "/contracts?state=concluido&agent=risco",
            "/docs",
        ]
        for page in pages:
            resp = client.get(page)
            assert resp.status_code == 200, page
        connectors_page = client.get("/connectors").text
        assert "simular_financiamento" in connectors_page and "online" in connectors_page
        agents_page = client.get("/agents").text
        assert "analisar_proposta" in agents_page and "avaliar_risco" in agents_page
        assert "contrato completo" in agents_page and "online" in agents_page


def test_model_crud_encrypts_key_and_tests_connection(db_url):
    with make_client(db_url) as client:
        resp = client.post(
            "/models",
            data={
                "preset": "openai",
                "name": "gpt-teste",
                "provider": "openai",
                "model": "gpt-x",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-super-secreta",
                "api_key_header": "Authorization",
                "temperature": "0,3",
                "max_tokens": "512",
                "timeout_s": "30",
                "json_mode": "on",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303 and "ok=" in resp.headers["location"]
        page = client.get("/models").text
        assert "gpt-teste" in page and "sk-super-secreta" not in page and "cifrada" in page

        invalid = client.post(
            "/models", data={"name": "Nome Inválido", "provider": "openai", "model": "x"}
        )
        assert invalid.status_code == 200 and "letras minúsculas" in invalid.text

        offline_id = _ids(client.get("/models").text, "models")
        tested = client.post(f"/models/{offline_id[-1]}/test", follow_redirects=True)
        assert "respondeu em" in tested.text

    db = Database(db_url)
    with db.session() as s:
        row = s.query(LlmModel).filter_by(name="gpt-teste").one()
        assert row.api_key.startswith("enc:") and row.temperature == 0.3
        assert SecretBox(SECRET).open(row.api_key) == "sk-super-secreta"


def test_decision_model_is_tested_with_a_jev_question(db_url):
    calls: list = []
    with make_client(db_url, seed_demo=False, jev_calls=calls) as client:
        form = client.get("/models/new?preset=typesafe").text
        assert 'value="typesafe" selected' in form and "jev-1.13.0" in form
        created = client.post(
            "/models",
            data={
                "preset": "typesafe",
                "name": "jev",
                "provider": "typesafe",
                "model": "jev-1.13.0",
                "base_url": "https://api.typesafe.ai",
                "api_key": "ts-segredo",
                "api_key_header": "Authorization",
                "timeout_s": "10",
            },
        )
        assert created.status_code == 200 and "jev" in created.text and "decisão" in created.text
        model_id = _ids(created.text, "models")[0]
        tested = client.post(f"/models/{model_id}/test", follow_redirects=True)
        assert "noul = 0.97" in tested.text
    [call] = calls
    assert (
        call["url"] == "https://api.typesafe.ai/v1/systemone"
        and call["auth"] == "Bearer ts-segredo"
    )
    assert call["model"] == "jev-1.13.0" and call["questions"]["greets"]["type"] == "noul"


def test_connector_registration_discovers_tools_and_calls_one(db_url):
    chamados.reset()
    with make_client(db_url, seed_demo=False) as client:
        resp = client.post(
            "/connectors",
            data={
                "name": "chamados",
                "url": "http://chamados/mcp",
                "transport": "streamable-http",
                "enabled": "on",
                "timeout_s": "10",
            },
        )
        assert resp.status_code == 200  # segue o redirect para o detalhe
        assert "abrir_chamado" in resp.text and "online" in resp.text
        connector_id = int(re.search(r"/connectors/(\d+)/call", resp.text).group(1))
        called = client.post(
            f"/connectors/{connector_id}/call",
            data={
                "tool": "abrir_chamado",
                "arguments": '{"titulo": "Erro no login", "descricao": "App fecha ao entrar"}',
            },
        )
        assert "CH-1001" in called.text
        bad = client.post(
            f"/connectors/{connector_id}/call",
            data={"tool": "abrir_chamado", "arguments": "não é json"},
        )
        assert "erro" in bad.text.lower()


def test_a2a_agent_registration_reads_card_and_contract_terms(db_url):
    with make_client(db_url, seed_demo=False) as client:
        resp = client.post(
            "/agents",
            data={
                "name": "risco",
                "url": "http://risco.test",
                "enabled": "on",
                "push": "on",
                "timeout_s": "10",
                "deadline_s": "90",
            },
        )
        assert resp.status_code == 200
        page = resp.text
        assert "avaliar_risco" in page and "contrato completo" in page and "online" in page
        assert "A2A 1.0" in page and "urn:switchboard:a2a:contract:v1" in page
        assert "sha256:" in page  # hash dos schemas declarados
        assert "Valor da operação" in page  # tabela do input schema
        agent_id = int(re.search(r'href="/agents/(\d+)/edit"', page).group(1))
        edit = client.get(f"/agents/{agent_id}/edit").text
        assert 'value="90.0"' in edit or 'value="90"' in edit

        offline = client.post(
            "/agents", data={"name": "fantasma", "url": "http://nao-existe.test", "enabled": "on"}
        )
        assert offline.status_code == 200 and "offline" in offline.text

        invalid = client.post("/agents", data={"name": "x", "url": "ftp://agente", "enabled": "on"})
        assert invalid.status_code == 200 and "http" in invalid.text


def test_profile_form_saves_decision_model_connectors_and_agents(db_url):
    with make_client(db_url) as client:
        client.post(
            "/models",
            data={
                "preset": "typesafe",
                "name": "jev",
                "provider": "typesafe",
                "model": "jev-1.13.0",
            },
        )
        form = client.get("/profiles/new").text
        model_id = re.search(r'<option value="(\d+)"[^>]*>offline', form).group(1)
        decision_id = re.search(r'<option value="(\d+)"[^>]*>jev', form).group(1)
        # o Jev não aparece como LLM principal, e o offline não aparece como modelo de decisão
        main_select = form.split('name="model_id"')[1].split("</select>")[0]
        decision_select = form.split('name="decision_model_id"')[1].split("</select>")[0]
        assert ">jev" not in main_select and ">offline" not in decision_select
        connector_ids = re.findall(r'name="connector_ids" value="(\d+)"', form)
        agent_ids = re.findall(r'name="agent_ids" value="(\d+)"', form)
        kb_ids = re.findall(r'name="kb_ids" value="(\d+)"', form)
        assert len(connector_ids) == 2 and len(agent_ids) == 2
        resp = client.post(
            "/profiles",
            data={
                "name": "hibrido",
                "model_id": model_id,
                "decision_model_id": decision_id,
                "decision_threshold": "0,7",
                "connector_ids": connector_ids[:1],
                "agent_ids": agent_ids,
                "kb_ids": kb_ids,
                "top_k": "3",
                "min_score": "0,15",
                "wait_s": "5",
                "max_parallel": "2",
                "deadline_s": "300",
                "synthesize": "on",
                "allow_clarify": "on",
                "enabled": "on",
            },
        )
        assert resp.status_code == 200 and "Roteador criado" in resp.text
        profiles = {p["name"]: p for p in client.get("/api/profiles").json()}
        hybrid = profiles["hibrido"]
        assert hybrid["decision_model"] == "jev" and hybrid["decision_threshold"] == 0.7
        assert len(hybrid["connectors"]) == 1 and sorted(hybrid["agents"]) == [
            "analise-credito",
            "risco",
        ]
        assert (hybrid["wait_s"], hybrid["max_parallel"], hybrid["deadline_s"]) == (5.0, 2, 300.0)
        listing = client.get("/profiles").text
        assert "decisão" in listing and "analise-credito" in listing

        swapped = client.post(
            "/profiles",
            data={"name": "errado", "model_id": decision_id, "enabled": "on"},
        )
        assert "modelo de decisão" in swapped.text


def test_playground_tool_call_and_trace(db_url):
    chamados.reset()
    with make_client(db_url) as client:
        answer = client.post(
            "/playground/send",
            json={
                "profile": "default",
                "messages": [{"role": "user", "content": "Simule 20 mil em 10 meses a 1% ao mês"}],
            },
        ).json()
        assert answer["route"] == "tool" and answer["tool"] == "simular_financiamento"
        assert answer["status"] == "completed" and answer["decided_by"] == "heuristic"

        missing = client.post(
            "/playground/send",
            json={"profile": "nao-existe", "messages": [{"role": "user", "content": "oi"}]},
        )
        assert missing.status_code == 404

        traces = client.get("/traces?route=tool").text
        assert "credito/simular_financiamento" in traces
        trace_id = re.search(r'href="/traces/([0-9a-f]{32})"', traces).group(1)
        detail = client.get(f"/traces/{trace_id}").text
        assert "tool: credito/simular_financiamento" in detail and "Simulação Price" in detail
        assert (
            'class="wf-row' in detail and "sb-autorefresh" not in detail
        )  # concluída: sem auto refresh


def test_playground_delegation_runs_in_background_and_is_consolidated(db_url):
    with make_client(db_url, analise_delay=0.4) as client:
        first = client.post(
            "/playground/send",
            json={
                "profile": "default",
                "wait_s": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": "Analise uma proposta de crédito de 80 mil para Maria Souza em 24 meses com renda de 12 mil",
                    }
                ],
            },
        )
        assert first.status_code == 202
        body = first.json()
        assert body["status"] == "pending" and body["route"] == "delegated"
        run_id = body["run_id"]

        live = client.get(f"/traces/{run_id}").text
        # em andamento: a página se atualiza sozinha (pausa com um detalhe aberto)
        assert 'name="sb-autorefresh" content="3"' in live and "app.js" in live
        assert "analise-credito/analisar_proposta" in live

        done = _wait_run(client, run_id, ("completed", "failed"))
        assert done["status"] == "completed" and "analise-credito" in done["answer"]
        [contract] = done["contracts"]
        assert contract["state"] == "concluido" and contract["kind"] == "completo"
        assert contract["output"]["decisao"] in ("aprovado", "aprovado_com_ajuste", "reprovado")
        kinds = {s["kind"] for s in done["spans"]}
        assert {"pedido", "decisao", "delegacao", "contrato", "consolidacao"} <= kinds

        page = client.get(f"/traces/{run_id}").text
        assert (
            "sb-autorefresh" not in page and "contrato: analise-credito/analisar_proposta" in page
        )
        assert "consolidacao" in page and "Eventos de" in page

        contracts = client.get("/contracts").text
        assert contract["id"] in contracts and "concluído" in contracts
        contract_page = client.get(f"/contracts/{contract['id']}").text
        assert "Linha do tempo" in contract_page and "limite_aprovado" in contract_page
        assert contract["input_hash"] in contract_page

        api = client.get(f"/api/contracts/{contract['id']}").json()
        assert api["state"] == "concluido" and api["input_schema"]["required"]
        run = client.get(f"/api/traces/{run_id}").json()
        assert run["status"] == "completed" and run["contracts"][0]["events"]


def test_playground_answers_an_agent_that_needs_input(db_url):
    with make_client(db_url, analise_delay=0.1) as client:
        question = "Analise uma proposta de crédito de 800 mil para João Lima em 120 meses com renda de 60 mil"
        history = [{"role": "user", "content": question}]
        first = client.post(
            "/playground/send", json={"profile": "default", "wait_s": 0, "messages": history}
        ).json()
        run_id = first["run_id"]
        asked = _wait_run(client, run_id, ("needs_input", "completed", "failed"))
        assert asked["status"] == "needs_input" and "garantia" in asked["answer"]
        # o painel mostra a execução aguardando entrada, e os links levam a listas que a incluem
        dashboard = client.get("/").text
        assert "/contracts?state=abertos" in dashboard and "/traces?status=abertas" in dashboard
        assert run_id in client.get("/traces?status=abertas").text
        assert "aguardando entrada" in client.get("/contracts?state=abertos").text
        waiting_page = client.get(f"/traces/{run_id}").text  # esperando o usuário: devagar
        assert 'name="sb-autorefresh" content="15"' in waiting_page

        history += [
            {"role": "assistant", "content": asked["answer"]},
            {"role": "user", "content": "sim, o cliente oferece o apartamento em garantia"},
        ]
        reply = client.post(
            "/playground/send",
            json={"profile": "default", "run_id": run_id, "wait_s": 5, "messages": history},
        ).json()
        final = (
            reply
            if reply["status"] == "completed"
            else _wait_run(client, run_id, ("completed", "failed"))
        )
        assert final["status"] == "completed"
        [contract] = client.get(f"/api/traces/{run_id}").json()["contracts"]
        assert contract["state"] == "concluido" and "imóvel" in contract["output"]["garantia"]


def test_cancel_run_from_trace_page(db_url):
    with make_client(db_url, analise_delay=30.0) as client:
        body = client.post(
            "/playground/send",
            json={
                "profile": "default",
                "wait_s": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": "Analise uma proposta de crédito de 50 mil para Ana Costa em 12 meses com renda de 9 mil",
                    }
                ],
            },
        ).json()
        run_id = body["run_id"]
        page = client.get(f"/traces/{run_id}").text
        assert f"/traces/{run_id}/cancel" in page
        canceled = client.post(f"/traces/{run_id}/cancel", follow_redirects=True)
        assert "1 contrato(s) cancelado(s)" in canceled.text
        done = _wait_run(client, run_id, ("completed", "failed"))
        assert done["status"] in ("completed", "failed")
        assert done["contracts"][0]["state"] == "cancelado"


def test_basic_auth_and_origin_check(db_url):
    with make_client(db_url, seed_demo=False, console_password="s3nha") as client:
        assert client.get("/").status_code == 401
        assert client.get("/healthz").status_code == 200
        token = base64.b64encode(b"admin:s3nha").decode()
        auth = {"Authorization": f"Basic {token}"}
        assert client.get("/", headers=auth).status_code == 200
        forged = client.post(
            "/models", data={"name": "x"}, headers={**auth, "Origin": "http://malicioso.example"}
        )
        assert forged.status_code == 403


def test_admin_json_api(db_url):
    with make_client(db_url, seed_demo=False) as client:
        model = client.post("/api/models", json={"name": "local", "provider": "offline"}).json()
        assert model["name"] == "local" and model["api_key"] is None
        jev = client.post(
            "/api/models", json={"name": "jev", "provider": "typesafe", "model": "jev-1.13.0"}
        ).json()
        connector = client.post(
            "/api/connectors",
            json={
                "name": "credito",
                "url": "http://credito/mcp",
                "allowed_tools": ["simular_financiamento"],
            },
        ).json()
        discovered = client.get(f"/api/connectors/{connector['id']}/discover").json()
        assert discovered["status"] == "online" and [t["name"] for t in discovered["tools"]] == [
            "simular_financiamento"
        ]
        called = client.post(
            f"/api/connectors/{connector['id']}/call",
            json={
                "tool": "simular_financiamento",
                "arguments": {"valor": 10000, "taxa_mensal_percentual": 1, "prazo_meses": 10},
            },
        ).json()
        assert called["is_error"] is False and "Parcela mensal" in called["text"]

        # /api/agents agora é A2A: um cadastro no formato MCP é recusado com explicação
        legacy = client.post(
            "/api/agents", json={"name": "velho", "url": "http://credito/mcp", "transport": "sse"}
        )
        assert legacy.status_code == 400 and "/api/connectors" in legacy.json()["detail"]
        agent = client.post(
            "/api/agents", json={"name": "risco", "url": "http://risco.test", "deadline_s": 60}
        ).json()
        assert agent["deadline_s"] == 60 and agent["push"] is True
        card = client.get(f"/api/agents/{agent['id']}/discover").json()
        assert card["status"] == "online" and card["contract_extension"] is True
        [skill] = card["skills"]
        assert skill["id"] == "avaliar_risco" and skill["contract"] == "completo"

        kb = client.post("/api/knowledge-bases", json={"name": "faq"}).json()
        doc = client.post(
            f"/api/knowledge-bases/{kb['id']}/documents",
            json={"title": "Pix", "content": "O Pix funciona 24 horas por dia."},
        ).json()
        assert doc["created"] is True
        upload = client.post(
            f"/api/knowledge-bases/{kb['id']}/files",
            files={"file": ("ted.txt", b"TED so em dias uteis.", "text/plain")},
        )
        assert upload.status_code == 201
        hits = client.post(
            f"/api/knowledge-bases/{kb['id']}/search", json={"query": "pix funciona de madrugada?"}
        ).json()
        assert hits[0]["document"] == "Pix"
        profile = client.post(
            "/api/profiles",
            json={
                "name": "api",
                "model": "local",
                "decision_model": "jev",
                "connectors": ["credito"],
                "agents": ["risco"],
                "knowledge_bases": ["faq"],
                "wait_s": 3,
            },
        ).json()
        assert profile["connectors"] == ["credito"] and profile["agents"] == ["risco"]
        assert profile["decision_model"] == "jev" and profile["wait_s"] == 3.0
        bad = client.post(
            "/api/profiles", json={"name": "x", "model": "local", "agents": ["fantasma"]}
        )
        assert bad.status_code == 400 and "fantasma" in bad.json()["detail"]
        wrong = client.post("/api/profiles", json={"name": "y", "model": "jev"})
        assert wrong.status_code == 400 and "modelo de decisão" in wrong.json()["detail"]
        assert client.delete(f"/api/models/{model['id']}").status_code == 409  # em uso
        assert client.delete(f"/api/models/{jev['id']}").status_code == 409  # decide pelo 'api'
        assert [p["name"] for p in client.get("/api/profiles").json()] == ["api"]
        assert "typesafe" in [p["key"] for p in client.get("/api/presets").json()]
        assert client.get("/api/traces").json() == []
        assert client.get("/api/contracts").json() == []
        assert client.get("/api/contracts/nao-existe").status_code == 404


def test_unexpected_host_is_rejected(db_url):
    """Proteção contra DNS rebinding: só os nomes configurados são atendidos."""
    with make_client(db_url, seed_demo=False) as client:
        assert client.get("/", headers={"Host": "evil.example"}).status_code == 400
        assert client.get("/healthz", headers={"Host": "localhost:8000"}).status_code == 200
        # health check de orquestrador chega pelo IP do contêiner
        assert client.get("/healthz", headers={"Host": "10.1.2.3:8000"}).status_code == 200


def test_non_ascii_password_does_not_crash(db_url):
    with make_client(db_url, seed_demo=False, console_password="senhaçã") as client:
        assert client.get("/").status_code == 401
        token = base64.b64encode("admin:senhaçã".encode()).decode()
        assert client.get("/", headers={"Authorization": f"Basic {token}"}).status_code == 200
        wrong = base64.b64encode("admin:outra-ç".encode()).decode()
        assert client.get("/", headers={"Authorization": f"Basic {wrong}"}).status_code == 401
        raw = {"Authorization": "Basic ç".encode("latin-1")}  # navegadores podem mandar latin-1
        assert client.get("/", headers=raw).status_code == 401


def test_console_restarts_after_demo_router_is_deleted(db_url):
    with make_client(db_url) as client:
        profile_id = _ids(client.get("/profiles").text, "profiles")[0]
        assert client.post(f"/profiles/{profile_id}/delete").status_code == 200
    with make_client(db_url) as client:  # segunda subida: não recria a demo nem quebra
        assert client.get("/healthz").status_code == 200
        assert _ids(client.get("/profiles").text, "profiles") == []
        assert "offline" in client.get("/models").text


def test_forbidden_env_reference_is_explained(db_url):
    with make_client(db_url, seed_demo=False) as client:
        resp = client.post(
            "/connectors",
            data={
                "name": "x",
                "url": "https://externo.example/mcp",
                "auth_token": "env:SWITCHBOARD_SECRET_KEY",
                "enabled": "on",
            },
        )
        assert resp.status_code == 200 and "não está liberada" in resp.text
        agent = client.post(
            "/agents",
            data={
                "name": "y",
                "url": "https://externo.example",
                "auth_token": "env:PATH",
                "enabled": "on",
            },
        )
        assert agent.status_code == 200 and "não está liberada" in agent.text
        api = client.post("/api/models", json={"name": "m", "model": "x", "api_key": "env:PATH"})
        assert api.status_code == 400


def test_upload_limit(db_url):
    with make_client(db_url, seed_demo=False, max_upload_mb=0) as client:
        kb = client.post("/api/knowledge-bases", json={"name": "faq"}).json()
        too_big = client.post(
            f"/api/knowledge-bases/{kb['id']}/files",
            files={"file": ("a.txt", b"conteudo", "text/plain")},
        )
        assert too_big.status_code == 413
        page = client.post(
            f"/knowledge/{kb['id']}/documents",
            files={"files": ("a.txt", b"conteudo", "text/plain")},
        )
        assert "maior que 0 MB" in page.text


def test_waterfall_orders_spans_as_a_tree():
    spans = [
        {
            "id": "b",
            "parent_id": "a",
            "kind": "rag",
            "name": "rag",
            "status": "ok",
            "started_at": "2026-01-01T00:00:00.100+00:00",
            "ended_at": "2026-01-01T00:00:00.300+00:00",
        },
        {
            "id": "a",
            "parent_id": None,
            "kind": "pedido",
            "name": "pedido",
            "status": "ok",
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": "2026-01-01T00:00:01+00:00",
        },
        {
            "id": "c",
            "parent_id": "b",
            "kind": "llm",
            "name": "llm",
            "status": "ok",
            "started_at": "2026-01-01T00:00:00.150+00:00",
            "ended_at": "2026-01-01T00:00:00.200+00:00",
        },
        {
            "id": "d",
            "parent_id": "sumiu",
            "kind": "contrato",
            "name": "contrato: x/y",
            "status": "open",
            "started_at": "2026-01-01T00:00:00.500+00:00",
            "ended_at": None,
        },
    ]
    from datetime import UTC, datetime

    wf = waterfall(spans, now=datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC))
    assert [(r["id"], r["depth"]) for r in wf["rows"]] == [("a", 0), ("b", 1), ("c", 2), ("d", 0)]
    assert wf["total_ms"] == 2000.0
    contract = wf["rows"][-1]
    assert contract["open"] and contract["elapsed_ms"] == 1500.0 and contract["left"] == 25.0
    assert waterfall([]) == {"rows": [], "total_ms": 0.0}
