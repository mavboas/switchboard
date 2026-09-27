"""Console do Switchboard: monólito com UI e API admin (control plane)."""

from __future__ import annotations

import base64
import binascii
import hmac
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from switchboard import __version__
from switchboard.agents import AgentCatalog
from switchboard.storage import seed_demo

from .api import api
from .settings import Settings
from .state import STATIC_DIR, Console
from .ui import router as ui_router

log = logging.getLogger("switchboard.console")

OPEN_PATHS = ("/healthz", "/static/")
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _basic_ok(header: str | None, user: str, password: str) -> bool:
    if not header or not header.lower().startswith("basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:]).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    given_user, _, given_password = decoded.partition(":")
    return hmac.compare_digest(given_user, user) and hmac.compare_digest(given_password, password)


def create_app(
    settings: Settings | None = None,
    *,
    catalog: AgentCatalog | None = None,
    http: httpx.AsyncClient | None = None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        console = Console.create(settings, catalog=catalog, http=http)
        app.state.console = console
        if settings.seed_demo:
            seeded = await seed_demo(
                console.db,
                knowledge_dir=Path(settings.demo_knowledge_dir),
                agent_urls={
                    "credito": settings.demo_credito_url,
                    "chamados": settings.demo_chamados_url,
                },
                box=console.box,
            )
            if seeded:
                log.info("banco vazio: configuração de demonstração criada (roteador 'default')")
        log.info(
            "console pronto (banco %s, vetores %s)", console.db.safe_url, console.db.vector_mode
        )
        try:
            yield
        finally:
            await console.aclose()

    app = FastAPI(
        title="Switchboard Console",
        version=__version__,
        description="Configuração de modelos, agentes MCP, bases de conhecimento e roteadores.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def guard(request: Request, call_next) -> Response:
        path = request.url.path
        needs_auth = bool(settings.console_password) and not path.startswith(OPEN_PATHS)
        if needs_auth and not _basic_ok(
            request.headers.get("authorization"),
            settings.console_user,
            settings.console_password or "",
        ):
            return PlainTextResponse(
                "autenticação necessária",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="switchboard", charset="UTF-8"'},
            )
        if request.method in UNSAFE_METHODS:
            # formulários só podem vir do próprio console (proteção simples contra CSRF)
            origin = request.headers.get("origin") or request.headers.get("referer")
            if origin and urlsplit(origin).netloc != request.url.netloc:
                return JSONResponse({"detail": "origem não permitida"}, status_code=403)
        return await call_next(request)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(api)
    app.include_router(ui_router)
    return app
