"""Console do Switchboard: monólito com UI e API admin (control plane)."""

from __future__ import annotations

import base64
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
from switchboard.a2a import AgentDirectory
from switchboard.connectors import ConnectorCatalog
from switchboard.net import host_allowed
from switchboard.storage import seed_demo

from .api import api
from .ops import router as ops_router
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
    except (ValueError, UnicodeDecodeError):  # base64 inválido ou com caracteres fora do ASCII
        return False
    given_user, _, given_password = decoded.partition(":")
    # compare_digest com bytes: str com caracteres fora do ASCII levantaria TypeError
    user_ok = hmac.compare_digest(given_user.encode("utf-8"), user.encode("utf-8"))
    password_ok = hmac.compare_digest(given_password.encode("utf-8"), password.encode("utf-8"))
    return user_ok and password_ok


def create_app(
    settings: Settings | None = None,
    *,
    connectors: ConnectorCatalog | None = None,
    agents: AgentDirectory | None = None,
    http: httpx.AsyncClient | None = None,
    jev_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        console = Console.create(
            settings, connectors=connectors, agents=agents, http=http, jev_transport=jev_transport
        )
        app.state.console = console
        if settings.seed_demo:
            try:
                seeded = await seed_demo(
                    console.db,
                    knowledge_dir=Path(settings.demo_knowledge_dir),
                    connector_urls={
                        "credito": settings.demo_credito_url,
                        "chamados": settings.demo_chamados_url,
                    },
                    agent_urls={
                        "analise-credito": settings.demo_analise_url,
                        "risco": settings.demo_risco_url,
                    },
                    box=console.box,
                )
            except Exception:  # a demo nunca pode impedir o console de subir
                log.exception("falha ao criar a configuração de demonstração")
                seeded = False
            if seeded:
                log.info("banco vazio: configuração de demonstração criada (roteador 'default')")
        if not settings.console_password:
            log.warning(
                "console sem senha: defina SWITCHBOARD_CONSOLE_PASSWORD antes de expor fora da sua máquina"
            )
        if not console.box.enabled:
            log.warning("sem chave mestra: só dá para usar env:NOME nos campos de segredo")
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
        description=(
            "Configuração de modelos, conectores MCP, agentes A2A, bases de conhecimento e "
            "roteadores; execuções, spans e contratos."
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def guard(request: Request, call_next) -> Response:
        path = request.url.path
        open_path = path.startswith(OPEN_PATHS)  # health checks podem vir pelo IP do contêiner
        if not open_path and not host_allowed(request.url.hostname, settings.allowed_hosts):
            # protege contra DNS rebinding: só atende os nomes esperados
            return PlainTextResponse(
                "host não permitido (ajuste SWITCHBOARD_CONSOLE_ALLOWED_HOSTS)", status_code=400
            )
        needs_auth = bool(settings.console_password) and not open_path
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
    app.include_router(ops_router)
    return app
