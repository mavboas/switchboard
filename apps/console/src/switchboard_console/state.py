"""Objetos compartilhados pelas rotas do console."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import anyio.to_thread
import httpx
from fastapi import Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from switchboard import __version__
from switchboard.agents import AgentCatalog
from switchboard.llm import PRESETS
from switchboard.secrets import SecretBox, load_master_key
from switchboard.storage import Database, EmbedderCache, KnowledgeService

from .settings import Settings

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

ROUTE_LABELS = {
    "direct": "Resposta direta",
    "delegated": "Delegado",
    "clarify": "Esclarecimento",
    "error": "Erro",
}


def _fmt_datetime(value) -> str:
    if value is None:
        return "—"
    return value.strftime("%d/%m/%Y %H:%M:%S")


def build_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    env = templates.env
    env.globals.update(version=__version__, route_labels=ROUTE_LABELS, presets=PRESETS)
    env.filters["datetime"] = _fmt_datetime
    env.filters["secret"] = SecretBox.describe
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False, "sort_keys": False}
    return templates


@dataclass
class Console:
    settings: Settings
    db: Database
    box: SecretBox
    knowledge: KnowledgeService
    catalog: AgentCatalog
    templates: Jinja2Templates
    http: httpx.AsyncClient

    @classmethod
    def create(
        cls,
        settings: Settings,
        *,
        catalog: AgentCatalog | None = None,
        http: httpx.AsyncClient | None = None,
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
            catalog=catalog or AgentCatalog(ttl_s=15.0, resolve_secret=box.open),
            templates=build_templates(),
            http=http or httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)),
        )

    async def aclose(self) -> None:
        await self.knowledge.aclose()
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
