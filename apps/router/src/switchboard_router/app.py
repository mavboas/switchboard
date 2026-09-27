"""API HTTP do router (data plane).

Endpoints:

* ``POST /v1/chat`` — API nativa: devolve resposta, rota, agente/tool, fontes e trace;
* ``POST /v1/chat/completions`` — compatível com OpenAI (inclusive ``stream``);
  o campo ``model`` escolhe o roteador (``default``, ``switchboard/default``…);
* ``GET /v1/models`` — lista os roteadores como "modelos";
* ``GET /v1/agents`` — agentes do roteador e o status da descoberta via MCP;
* ``GET /healthz`` e ``GET /readyz``.
"""

from __future__ import annotations

import hmac
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from switchboard import __version__
from switchboard.app import Switchboard
from switchboard.net import host_allowed
from switchboard.routing import RouterResult
from switchboard.secrets import SecretBox, load_master_key
from switchboard.storage import Database

from .runtime import DbRuntime, FileRuntime, ProfileNotFound, Runtime
from .settings import Settings

log = logging.getLogger("switchboard.router")


class ChatMessage(BaseModel):
    role: str = "user"
    content: str | list[dict[str, Any]] | None = ""


class ChatRequest(BaseModel):
    profile: str | None = Field(
        default=None, description="roteador (padrão: SWITCHBOARD_DEFAULT_PROFILE)"
    )
    message: str | None = Field(
        default=None, description="atalho para uma única mensagem de usuário"
    )
    messages: list[ChatMessage] = Field(
        default_factory=list, description="histórico no formato role/content"
    )


class CompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    stream: bool = False

    model_config = {"extra": "allow"}


def build_runtime(settings: Settings) -> Runtime:
    if settings.config_file:
        return FileRuntime(Switchboard.from_yaml(Path(settings.config_file)))
    db = Database(settings.database_url, vector_backend=settings.vector_backend)
    db.init()
    return DbRuntime(
        db,
        SecretBox(
            load_master_key(settings.secret_key, settings.secret_key_file),
            settings.allowed_env_secrets,
        ),
        config_ttl_s=settings.config_ttl_s,
        agents_ttl_s=settings.agents_ttl_s,
    )


def _profile_from_model(model: str, default: str) -> str:
    model = model.strip()
    for prefix in ("switchboard/", "switchboard:"):
        if model.startswith(prefix):
            model = model[len(prefix) :]
    return model if model and model != "switchboard" else default


def _openai_error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "invalid_request_error", "code": code}},
    )


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = runtime or build_runtime(settings)
        log.info("router pronto (config: %s)", app.state.runtime.source)
        try:
            yield
        finally:
            await app.state.runtime.aclose()

    app = FastAPI(
        title="Switchboard Router",
        version=__version__,
        description="Roteador de agentes: responde, delega via MCP ou esclarece.",
        lifespan=lifespan,
    )

    keys = settings.api_key_list
    allowed_hosts = settings.allowed_host_list

    @app.middleware("http")
    async def host_guard(request: Request, call_next):
        # API aberta (sem chaves) = uso local: só atende hosts conhecidos (contra DNS rebinding)
        if (
            not keys
            and request.url.path not in ("/healthz", "/readyz")
            and not host_allowed(request.url.hostname, allowed_hosts)
        ):
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "host não permitido (defina SWITCHBOARD_API_KEYS ou SWITCHBOARD_ALLOWED_HOSTS)"
                },
            )
        return await call_next(request)

    def require_key(request: Request) -> None:
        if not keys:
            return
        header = request.headers.get("authorization", "")
        token = (
            header[7:].strip()
            if header.lower().startswith("bearer ")
            else request.headers.get("x-api-key", "")
        )
        given = token.encode("utf-8")
        if not any(hmac.compare_digest(given, k.encode("utf-8")) for k in keys):
            raise HTTPException(status_code=401, detail="chave de API inválida ou ausente")

    def rt(request: Request) -> Runtime:
        return request.app.state.runtime

    async def run(
        runtime: Runtime, profile: str, messages: list[dict[str, Any]], tasks: BackgroundTasks
    ) -> RouterResult:
        engine = await runtime.engine(profile)
        result = await engine.handle(messages)
        log.info(
            json.dumps(
                {
                    "evento": "pedido",
                    "trace_id": result.trace_id,
                    "roteador": result.profile,
                    "rota": result.route,
                    "agente": result.agent,
                    "tool": result.tool,
                    "latencia_ms": round(result.latency_ms, 1),
                    "avisos": len(result.warnings),
                },
                ensure_ascii=False,
            )
        )
        tasks.add_task(runtime.record, result)
        return result

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        return {"servico": "switchboard-router", "versao": __version__, "docs": "/docs"}

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(runtime: Runtime = Depends(rt)) -> JSONResponse:
        ok = await runtime.ready()
        return JSONResponse(
            status_code=200 if ok else 503, content={"status": "ok" if ok else "banco indisponível"}
        )

    @app.get("/v1/models", dependencies=[Depends(require_key)])
    async def models(runtime: Runtime = Depends(rt)) -> dict[str, Any]:
        names = await runtime.profiles()
        return {
            "object": "list",
            "data": [
                {"id": n, "object": "model", "created": 0, "owned_by": "switchboard"} for n in names
            ],
        }

    @app.get("/v1/agents", dependencies=[Depends(require_key)])
    async def agents(
        profile: str | None = None, refresh: bool = False, runtime: Runtime = Depends(rt)
    ):
        try:
            engine = await runtime.engine(profile or settings.default_profile)
        except ProfileNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        infos = await runtime.catalog.discover(engine.profile.agents, refresh=refresh)
        return {"profile": engine.profile.name, "agents": [i.to_dict() for i in infos]}

    @app.post("/v1/chat", dependencies=[Depends(require_key)])
    async def chat(
        body: ChatRequest, tasks: BackgroundTasks, runtime: Runtime = Depends(rt)
    ) -> dict[str, Any]:
        messages = [m.model_dump() for m in body.messages]
        if body.message:
            messages.append({"role": "user", "content": body.message})
        if not messages:
            raise HTTPException(status_code=422, detail="envie 'message' ou 'messages'")
        try:
            result = await run(runtime, body.profile or settings.default_profile, messages, tasks)
        except ProfileNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return result.to_dict()

    @app.post("/v1/chat/completions", dependencies=[Depends(require_key)])
    async def completions(
        body: CompletionRequest, tasks: BackgroundTasks, runtime: Runtime = Depends(rt)
    ):
        profile = _profile_from_model(body.model, settings.default_profile)
        try:
            result = await run(runtime, profile, [m.model_dump() for m in body.messages], tasks)
        except ProfileNotFound as exc:
            return _openai_error(404, str(exc), "model_not_found")
        created = int(time.time())
        completion_id = f"chatcmpl-{result.trace_id}"
        extra = {
            "trace_id": result.trace_id,
            "route": result.route,
            "agent": result.agent,
            "tool": result.tool,
            "sources": [s.to_dict() for s in result.sources],
        }
        usage = {
            "prompt_tokens": result.usage.input_tokens,
            "completion_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.input_tokens + result.usage.output_tokens,
        }
        if not body.stream:
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": profile,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
                "switchboard": extra,
            }

        def chunk(delta: dict[str, Any], finish: str | None, **more: Any) -> str:
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": profile,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                **more,
            }
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        async def events() -> AsyncIterator[str]:
            yield chunk({"role": "assistant", "content": result.answer}, None)
            yield chunk({}, "stop", usage=usage, switchboard=extra)
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
