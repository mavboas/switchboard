"""Objetos compartilhados pelas rotas do console."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import anyio.to_thread
import httpx
from fastapi import Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from switchboard import __version__
from switchboard.a2a import A2AClient, AgentDirectory
from switchboard.connectors import ConnectorCatalog
from switchboard.contracts import states
from switchboard.llm import PRESETS
from switchboard.secrets import SecretBox, load_master_key
from switchboard.storage import Database, EmbedderCache, KnowledgeService

from .settings import Settings

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

ROUTE_LABELS = {
    "direct": "Resposta direta",
    "tool": "Tool MCP",
    "delegated": "Agente A2A",
    "clarify": "Esclarecimento",
    "error": "Erro",
}

RUN_STATUS_LABELS = {
    "completed": "concluída",
    "pending": "em andamento",
    "needs_input": "aguardando entrada",
    "consolidating": "consolidando",
    "failed": "falhou",
}

# classe de cor (badge) de cada estado de contrato e de execução
STATE_TONES = {
    states.PROPOSED: "",
    states.ACTIVE: "info",
    states.INPUT_REQUIRED: "warn",
    states.COMPLETED: "ok",
    states.FAILED: "danger",
    states.REJECTED: "danger",
    states.CANCELED: "",
    states.EXPIRED: "danger",
    states.BREACHED: "danger",
    "completed": "ok",
    "pending": "info",
    "needs_input": "warn",
    "consolidating": "info",
    "failed": "danger",
}


def _as_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _fmt_datetime(value) -> str:
    value = _as_datetime(value)
    if value is None:
        return "—"
    return value.strftime("%d/%m/%Y %H:%M:%S")


def _fmt_ms(value) -> str:
    if value is None:
        return "—"
    value = float(value)
    if value >= 60_000:
        return f"{value / 60_000:.1f} min"
    if value >= 1000:
        return f"{value / 1000:.1f} s"
    return f"{value:.0f} ms"


def build_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    env = templates.env
    env.globals.update(
        version=__version__,
        route_labels=ROUTE_LABELS,
        run_labels=RUN_STATUS_LABELS,
        contract_labels=states.LABELS,
        contract_states=[
            states.PROPOSED,
            states.ACTIVE,
            states.INPUT_REQUIRED,
            states.COMPLETED,
            states.FAILED,
            states.REJECTED,
            states.CANCELED,
            states.EXPIRED,
            states.BREACHED,
        ],
        open_states=states.OPEN,
        tones=STATE_TONES,
        presets=PRESETS,
    )
    env.filters["datetime"] = _fmt_datetime
    env.filters["ms"] = _fmt_ms
    env.filters["secret"] = SecretBox.describe
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False, "sort_keys": False}
    return templates


@dataclass
class Console:
    settings: Settings
    db: Database
    box: SecretBox
    knowledge: KnowledgeService
    connectors: ConnectorCatalog
    agents: AgentDirectory
    templates: Jinja2Templates
    http: httpx.AsyncClient
    # transporte HTTP do teste de modelos de decisão (Jev); None = rede de verdade
    jev_transport: httpx.AsyncBaseTransport | None = None

    @classmethod
    def create(
        cls,
        settings: Settings,
        *,
        connectors: ConnectorCatalog | None = None,
        agents: AgentDirectory | None = None,
        http: httpx.AsyncClient | None = None,
        jev_transport: httpx.AsyncBaseTransport | None = None,
    ) -> Console:
        db = Database(settings.database_url, vector_backend=settings.vector_backend)
        db.init()
        box = SecretBox(
            load_master_key(settings.secret_key, settings.secret_key_file),
            settings.allowed_env_secrets,
        )
        return cls(
            settings=settings,
            db=db,
            box=box,
            knowledge=KnowledgeService(db, EmbedderCache(box.open)),
            connectors=connectors or ConnectorCatalog(ttl_s=15.0, resolve_secret=box.open),
            agents=agents
            or AgentDirectory(client=A2AClient(), ttl_s=15.0, resolve_secret=box.open),
            templates=build_templates(),
            http=http or httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)),
            jev_transport=jev_transport,
        )

    async def aclose(self) -> None:
        await self.knowledge.aclose()
        await self.agents.aclose()
        await self.http.aclose()
        self.db.dispose()

    # -- helpers --------------------------------------------------------------

    async def run_db(self, fn, *args):
        """Executa uma função que usa sessão do banco fora do event loop."""

        def call():
            with self.db.session() as session:
                return fn(session, *args)

        return await anyio.to_thread.run_sync(call)

    def render(self, request: Request, name: str, **context: Any):
        context.setdefault("nav", "")
        context["flash_ok"] = request.query_params.get("ok")
        context["flash_error"] = request.query_params.get("erro")
        context["settings"] = self.settings
        return self.templates.TemplateResponse(request, name, context)

    async def router_request(self, method: str, path: str, **kwargs) -> httpx.Response:
        headers = kwargs.pop("headers", {})
        if self.settings.router_api_key:
            headers["Authorization"] = f"Bearer {self.settings.router_api_key}"
        url = self.settings.router_url.rstrip("/") + path
        return await self.http.request(method, url, headers=headers, **kwargs)


def get_console(request: Request) -> Console:
    return request.app.state.console


def redirect(url: str, *, ok: str | None = None, erro: str | None = None) -> RedirectResponse:
    params = {k: v for k, v in (("ok", ok), ("erro", erro)) if v}
    if params:
        url += ("&" if "?" in url else "?") + urlencode(params)
    return RedirectResponse(url, status_code=303)
