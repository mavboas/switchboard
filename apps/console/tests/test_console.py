from __future__ import annotations

import base64
import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from switchboard.agents import AgentCatalog
from switchboard.secrets import SecretBox
from switchboard.storage import Database
from switchboard.storage.orm import LlmModel
from switchboard.testing import inproc_connector
from switchboard_agents import chamados, credito
from switchboard_console.app import create_app
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


def _router_http(db_url: str) -> httpx.AsyncClient:
    """Cliente HTTP do console apontando para um router em processo (ASGI)."""
    db = Database(db_url)
    db.init()
    catalog = AgentCatalog(connector=inproc_connector(SERVERS))
    runtime = DbRuntime(db, SecretBox(SECRET), config_ttl_s=0, catalog=catalog)
    router_app = create_router(RouterSettings(database_url=db_url), runtime=runtime)
    router_app.state.runtime = runtime
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=router_app), base_url="http://router"
    )


def make_client(db_url: str, **overrides) -> TestClient:
    settings = Settings(
        database_url=db_url,
        secret_key=SECRET,
        router_url="http://router",
        demo_knowledge_dir=str(KNOWLEDGE_DIR),
        demo_credito_url="http://credito/mcp",
        demo_chamados_url="http://chamados/mcp",
        **{"console_allowed_hosts": "console,localhost", **overrides},
    )
    app = create_app(
        settings,
        catalog=AgentCatalog(connector=inproc_connector(SERVERS)),
        http=_router_http(db_url),
    )
    return TestClient(app, base_url="http://console")


def _ids(html: str, prefix: str) -> list[int]:
    return [int(x) for x in re.findall(rf'href="/{prefix}/(\d+)"', html)]


def test_all_pages_render_with_demo_data(db_url):
    chamados.reset()
    with make_client(db_url) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "Primeiros passos" in home.text and "pgvector" not in home.text  # sqlite usa json
        model_id = _ids(client.get("/models").text, "models")[0]
        agent_id = _ids(client.get("/agents").text, "agents")[0]
        kb_id = _ids(client.get("/knowledge").text, "knowledge")[0]
        profile_id = _ids(client.get("/profiles").text, "profiles")[0]
        pages = [
            "/models/new?preset=anthropic",
            f"/models/{model_id}",
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
            "/docs",
        ]
        for page in pages:
            resp = client.get(page)
            assert resp.status_code == 200, page
        agents_page = client.get("/agents").text
        assert "simular_financiamento" in agents_page and "online" in agents_page


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


def test_agent_registration_discovers_tools_and_calls_one(db_url):
    chamados.reset()
    with make_client(db_url, seed_demo=False) as client:
        resp = client.post(
            "/agents",
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
        agent_id = int(re.search(r"/agents/(\d+)/call", resp.text).group(1))
        called = client.post(
            f"/agents/{agent_id}/call",
            data={
                "tool": "abrir_chamado",
                "arguments": '{"titulo": "Erro no login", "descricao": "App fecha ao entrar"}',
            },
        )
        assert "CH-1001" in called.text
        bad = client.post(
            f"/agents/{agent_id}/call", data={"tool": "abrir_chamado", "arguments": "não é json"}
        )
        assert "erro" in bad.text.lower()


def test_knowledge_flow(db_url):
    with make_client(db_url, seed_demo=False) as client:
        resp = client.post(
            "/knowledge",
            data={
                "name": "faq",
                "embedder_kind": "hashing",
                "embedder_dim": "512",
                "chunk_size": "600",
                "chunk_overlap": "80",
            },
        )
        kb_id = int(re.search(r'action="/knowledge/(\d+)/documents"', resp.text).group(1))
        added = client.post(
            f"/knowledge/{kb_id}/documents",
            data={
                "title": "Horários",
                "content": "## Sábado\n\nAbrimos aos sábados das 9h às 14h.",
            },
            files={
                "files": (
                    "boletos.md",
                    b"# Boletos\n\nO boleto vence todo dia 10.",
                    "text/markdown",
                )
            },
        )
        assert "2 documento(s) indexado(s)" in added.text
        again = client.post(
            f"/knowledge/{kb_id}/documents",
            data={"title": "x", "content": "O boleto vence todo dia 10."},
        )
        assert again.status_code == 200
        bad = client.post(
            f"/knowledge/{kb_id}/documents",
            files={"files": ("virus.exe", b"MZ", "application/octet-stream")},
        )
        assert "não suportado" in bad.text
        found = client.post(f"/knowledge/{kb_id}/search", data={"query": "vocês abrem no sábado?"})
        assert "Horários" in found.text and "Sábado" in found.text
        client.post(
            f"/knowledge/{kb_id}",
            data={
                "name": "faq",
                "embedder_kind": "hashing",
                "embedder_dim": "256",
                "chunk_size": "600",
                "chunk_overlap": "80",
            },
        )
        assert "reindexar" in client.get("/knowledge").text.lower()
        reindexed = client.post(f"/knowledge/{kb_id}/reindex")
        assert "Reindexação concluída" in reindexed.text
        doc_ids = re.findall(rf"/knowledge/{kb_id}/documents/(\d+)/delete", reindexed.text)
        removed = client.post(f"/knowledge/{kb_id}/documents/{doc_ids[0]}/delete")
        assert "Documento removido" in removed.text


def test_profile_form_and_playground_roundtrip(db_url):
    chamados.reset()
    with make_client(db_url) as client:
        form = client.get("/profiles/new").text
        model_id = re.search(r'<option value="(\d+)"[^>]*>offline', form).group(1)
        agent_ids = re.findall(r'name="agent_ids" value="(\d+)"', form)
        kb_ids = re.findall(r'name="kb_ids" value="(\d+)"', form)
        resp = client.post(
            "/profiles",
            data={
                "name": "atendimento",
                "model_id": model_id,
                "agent_ids": agent_ids,
                "kb_ids": kb_ids,
                "system_prompt": "Seja breve.",
                "top_k": "3",
                "min_score": "0,15",
                "synthesize": "on",
                "allow_clarify": "on",
                "enabled": "on",
            },
        )
        assert resp.status_code == 200 and "Roteador criado" in resp.text

        answer = client.post(
            "/playground/send",
            json={
                "profile": "atendimento",
                "messages": [{"role": "user", "content": "Simule 20 mil em 10 meses a 1% ao mês"}],
            },
        ).json()
        assert answer["route"] == "delegated" and answer["tool"] == "simular_financiamento"

        missing = client.post(
            "/playground/send",
            json={"profile": "nao-existe", "messages": [{"role": "user", "content": "oi"}]},
        )
        assert missing.status_code == 404

        traces = client.get("/traces?route=delegated").text
        assert "credito/simular_financiamento" in traces
        trace_id = re.search(r'href="/traces/([0-9a-f]{32})"', traces).group(1)
        detail = client.get(f"/traces/{trace_id}").text
        assert "delegacao_mcp" in detail and "Simulação Price" in detail


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
        agent = client.post(
            "/api/agents",
            json={
                "name": "credito",
                "url": "http://credito/mcp",
                "allowed_tools": ["simular_financiamento"],
            },
        ).json()
        discovered = client.get(f"/api/agents/{agent['id']}/discover").json()
        assert discovered["status"] == "online" and [t["name"] for t in discovered["tools"]] == [
            "simular_financiamento"
        ]
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
                "agents": ["credito"],
                "knowledge_bases": ["faq"],
            },
        ).json()
        assert profile["agents"] == ["credito"] and profile["knowledge_bases"] == ["faq"]
        bad = client.post(
            "/api/profiles", json={"name": "x", "model": "local", "agents": ["fantasma"]}
        )
        assert bad.status_code == 400 and "fantasma" in bad.json()["detail"]
        assert client.delete(f"/api/models/{model['id']}").status_code == 409  # em uso
        assert [p["name"] for p in client.get("/api/profiles").json()] == ["api"]
        assert client.get("/api/presets").json()[0]["key"] == "openai"
        assert client.get("/api/traces").json() == []


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
            "/agents",
            data={
                "name": "x",
                "url": "https://externo.example/mcp",
                "auth_token": "env:SWITCHBOARD_SECRET_KEY",
                "enabled": "on",
            },
        )
        assert resp.status_code == 200 and "não está liberada" in resp.text
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
