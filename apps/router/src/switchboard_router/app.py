"""API HTTP do router (data plane).

Endpoints:

* ``POST /v1/chat`` — API nativa: resposta, rota, decisão, tarefas, contratos e
  spans. Com delegação a agentes que não terminam em ``wait_s``, responde
  **202** com ``status: "pending"`` e o ``run_id``; ``run_id`` no corpo continua
  uma execução que aguarda entrada do usuário;
* ``POST /v1/chat/completions`` — compatível com OpenAI (inclusive ``stream``);
  o campo ``model`` escolhe o roteador e o campo extra ``switchboard`` traz o
  estado da execução;
* ``GET /v1/runs/{id}`` — execução completa (spans, contratos e eventos);
* ``GET /v1/runs/{id}/events`` — SSE com o andamento até a resposta final;
* ``POST /v1/runs/{id}/cancel`` — cancela os contratos abertos;
* ``POST /a2a/push/{contract_id}`` — push notifications dos agentes A2A
  (autenticadas pelo token do contrato, não pela chave da API);
* ``GET /v1/models``, ``GET /v1/connectors``, ``GET /v1/agents``,
  ``GET /healthz`` e ``GET /readyz``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from switchboard import __version__
from switchboard.a2a import NOTIFICATION_TOKEN_HEADER
from switchboard.app import Switchboard
from switchboard.contracts import RUN_FINAL, PushRejected
from switchboard.net import host_allowed
from switchboard.routing import RouterResult
from switchboard.secrets import SecretBox, load_master_key
from switchboard.storage import Database

from .runtime import DbRuntime, FileRuntime, ProfileNotFound, Runtime
from .settings import Settings

log = logging.getLogger("switchboard.router")

OPEN_PATHS = ("/healthz", "/readyz")
SSE_MAX_S = 15 * 60
SSE_HEARTBEAT_S = 15.0


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
    run_id: str | None = Field(
        default=None,
        description="continua uma execução em 'needs_input': a mensagem vai ao agente, sob o mesmo contrato",
    )
    wait_s: float | None = Field(
        default=None,
        ge=0,
        description="quanto esperar os agentes antes de responder 'pending' (padrão: o do roteador)",
    )
    callback_url: str | None = Field(
        default=None,
        description="webhook chamado com o resultado final (só hosts de SWITCHBOARD_CALLBACK_HOSTS)",
    )


class CompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    stream: bool = False

    model_config = {"extra": "allow"}


def build_runtime(settings: Settings) -> Runtime:
    if settings.config_file:
        return FileRuntime(
            Switchboard.from_yaml(Path(settings.config_file)), public_url=settings.public_url
        )
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
        public_url=settings.public_url,
        supervisor_tick_s=settings.supervisor_tick_s,
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


def _links(run_id: str) -> dict[str, str]:
    return {
        "run": f"/v1/runs/{run_id}",
        "events": f"/v1/runs/{run_id}/events",
        "cancel": f"/v1/runs/{run_id}/cancel",
    }


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = runtime or build_runtime(settings)
        await app.state.runtime.start()
        log.info(
            "router pronto (config: %s; push de agentes: %s)",
            app.state.runtime.source,
            settings.public_url or "desligado (só polling)",
        )
        try:
            yield
        finally:
            await app.state.runtime.aclose()

    app = FastAPI(
        title="Switchboard Router",
        version=__version__,
        description=(
            "Roteador de agentes: responde com RAG, aciona tools via MCP e delega tarefas a "
            "agentes via A2A, cada uma sob um contrato."
        ),
        lifespan=lifespan,
    )

    keys = settings.api_key_list
    allowed_hosts = settings.allowed_host_list

    @app.middleware("http")
    async def host_guard(request: Request, call_next):
        # API aberta (sem chaves) = uso local: só atende hosts conhecidos (contra DNS rebinding)
        if (
            not keys
            and request.url.path not in OPEN_PATHS
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

    def check_callback(url: str | None) -> str | None:
        if not url:
            return None
        allowed = settings.callback_host_list
        parts = urlsplit(url)
        if not allowed:
            raise HTTPException(
                status_code=422, detail="callbacks desligados (defina SWITCHBOARD_CALLBACK_HOSTS)"
            )
        if parts.scheme not in ("http", "https") or (parts.hostname or "").lower() not in allowed:
            raise HTTPException(status_code=422, detail="callback_url fora dos hosts liberados")
        return url

    def wait_budget(wait_s: float | None) -> float | None:
        if wait_s is None or not math.isfinite(wait_s):  # NaN/inf (JSON aceita NaN) = padrão
            return None
        return min(max(wait_s, 0.0), settings.max_wait_s)

    def log_result(result: RouterResult) -> None:
        log.info(
            json.dumps(
                {
                    "evento": "pedido",
                    "trace_id": result.trace_id,
                    "roteador": result.profile,
                    "rota": result.route,
                    "status": result.status,
                    "decidido_por": result.decided_by,
                    "agente": result.agent,
                    "tool": result.tool,
                    "contratos": len(result.contracts),
                    "latencia_ms": round(result.latency_ms, 1),
                    "avisos": len(result.warnings),
                },
                ensure_ascii=False,
            )
        )

    async def run(
        runtime: Runtime,
        profile: str,
        messages: list[dict[str, Any]],
        tasks: BackgroundTasks,
        *,
        run_id: str | None = None,
        wait_s: float | None = None,
        callback_url: str | None = None,
    ) -> RouterResult:
        if run_id:
            existing = await runtime.contracts.store.get_run(run_id)
            if existing is None:
                raise HTTPException(status_code=404, detail=f"execução {run_id} não encontrada")
            engine = await runtime.engine(existing.profile)
            result = await engine.resume(run_id, messages, wait_s=wait_s)
        else:
            engine = await runtime.engine(profile)
            result = await engine.handle(messages, wait_s=wait_s, callback_url=callback_url)
        log_result(result)
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

    async def _engine_or_404(runtime: Runtime, profile: str | None):
        try:
            return await runtime.engine(profile or settings.default_profile)
        except ProfileNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/connectors", dependencies=[Depends(require_key)])
    async def connectors(
        profile: str | None = None, refresh: bool = False, runtime: Runtime = Depends(rt)
    ):
        engine = await _engine_or_404(runtime, profile)
        infos = await runtime.connectors.discover(engine.profile.connectors, refresh=refresh)
        return {"profile": engine.profile.name, "connectors": [i.to_dict() for i in infos]}

    @app.get("/v1/agents", dependencies=[Depends(require_key)])
    async def agents(
        profile: str | None = None, refresh: bool = False, runtime: Runtime = Depends(rt)
    ):
        engine = await _engine_or_404(runtime, profile)
        infos = await runtime.agents.discover(engine.profile.agents, refresh=refresh)
        return {"profile": engine.profile.name, "agents": [i.to_dict() for i in infos]}

    @app.post("/v1/chat", dependencies=[Depends(require_key)])
    async def chat(
        body: ChatRequest, tasks: BackgroundTasks, runtime: Runtime = Depends(rt)
    ) -> JSONResponse:
        messages = [m.model_dump() for m in body.messages]
        if body.message:
            messages.append({"role": "user", "content": body.message})
        if not messages:
            raise HTTPException(status_code=422, detail="envie 'message' ou 'messages'")
        try:
            result = await run(
                runtime,
                body.profile or settings.default_profile,
                messages,
                tasks,
                run_id=body.run_id,
                wait_s=wait_budget(body.wait_s),
                callback_url=check_callback(body.callback_url),
            )
        except ProfileNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        data = result.to_dict()
        if result.route == "delegated":
            data["links"] = _links(result.run_id)
        return JSONResponse(data, status_code=202 if result.status == "pending" else 200)

    @app.post("/v1/chat/completions", dependencies=[Depends(require_key)])
    async def completions(
        body: CompletionRequest, tasks: BackgroundTasks, runtime: Runtime = Depends(rt)
    ):
        profile = _profile_from_model(body.model, settings.default_profile)
        extra = body.model_extra or {}
        options = extra.get("switchboard") if isinstance(extra.get("switchboard"), dict) else {}
        run_id = options.get("run_id") or extra.get("run_id")
        wait_s = options.get("wait_s", extra.get("wait_s"))
        try:
            result = await run(
                runtime,
                profile,
                [m.model_dump() for m in body.messages],
                tasks,
                run_id=str(run_id) if run_id else None,
                wait_s=wait_budget(float(wait_s)) if isinstance(wait_s, (int, float)) else None,
            )
        except ProfileNotFound as exc:
            return _openai_error(404, str(exc), "model_not_found")
        except HTTPException as exc:
            return _openai_error(exc.status_code, str(exc.detail), "run_not_found")
        created = int(time.time())
        completion_id = f"chatcmpl-{result.trace_id}"
        extra_out: dict[str, Any] = {
            "trace_id": result.trace_id,
            "run_id": result.run_id,
            "status": result.status,
            "route": result.route,
            "decided_by": result.decided_by,
            "agent": result.agent,
            "tool": result.tool,
            "sources": [s.to_dict() for s in result.sources],
            "contracts": result.contracts,
        }
        if result.route == "delegated":
            extra_out["links"] = _links(result.run_id)
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
                "switchboard": extra_out,
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
            yield chunk({}, "stop", usage=usage, switchboard=extra_out)
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    # -- execuções assíncronas ------------------------------------------------------

    @app.get("/v1/runs/{run_id}", dependencies=[Depends(require_key)])
    async def get_run(run_id: str, runtime: Runtime = Depends(rt)) -> dict[str, Any]:
        details = await runtime.run_details(run_id)
        if details is None:
            raise HTTPException(status_code=404, detail="execução não encontrada")
        details["links"] = _links(run_id)
        return details

    async def _snapshot(runtime: Runtime, run_id: str) -> dict[str, Any] | None:
        run = await runtime.contracts.store.get_run(run_id)
        if run is None:
            return None
        contracts = await runtime.contracts.store.run_contracts(run_id)
        return {**run.to_dict(), "contracts": [c.summary() for c in contracts]}

    @app.get("/v1/runs/{run_id}/events", dependencies=[Depends(require_key)])
    async def run_events(run_id: str, request: Request, runtime: Runtime = Depends(rt)):
        first = await _snapshot(runtime, run_id)
        if first is None:
            raise HTTPException(status_code=404, detail="execução não encontrada")

        def sse(event: str, data: dict[str, Any]) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

        async def stream() -> AsyncIterator[str]:
            loop = asyncio.get_running_loop()
            end = loop.time() + SSE_MAX_S
            last_beat = loop.time()
            snapshot: dict[str, Any] | None = first
            sent: str | None = None
            while True:
                if snapshot is None:
                    yield sse("error", {"detail": "execução removida"})
                    return
                key = json.dumps(
                    [snapshot["status"], snapshot["answer"], snapshot["contracts"]],
                    default=str,
                    sort_keys=True,
                )
                if key != sent:
                    sent = key
                    yield sse("run", snapshot)
                    last_beat = loop.time()
                if snapshot["status"] in RUN_FINAL:
                    yield sse("done", {"run_id": run_id, "status": snapshot["status"]})
                    return
                if loop.time() >= end:
                    yield sse("timeout", {"run_id": run_id})
                    return
                if await request.is_disconnected():
                    return
                if loop.time() - last_beat >= SSE_HEARTBEAT_S:
                    yield ": ping\n\n"
                    last_beat = loop.time()
                if snapshot["status"] == "pending":
                    with contextlib.suppress(Exception):  # acorda cedo quando consolida
                        await runtime.contracts.wait_run(run_id, 1.0)
                else:  # aguardando entrada do usuário: só acompanha
                    await asyncio.sleep(1.0)
                snapshot = await _snapshot(runtime, run_id)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/runs/{run_id}/cancel", dependencies=[Depends(require_key)])
    async def cancel_run(run_id: str, runtime: Runtime = Depends(rt)) -> dict[str, Any]:
        if await runtime.contracts.store.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="execução não encontrada")
        canceled = await runtime.contracts.cancel_run(run_id)
        return {"run_id": run_id, "canceled": [c.summary() for c in canceled]}

    # -- push notifications dos agentes A2A ----------------------------------------

    @app.post("/a2a/push/{contract_id}", tags=["a2a"])
    async def a2a_push(contract_id: str, request: Request, runtime: Runtime = Depends(rt)):
        """Recebe um ``StreamResponse`` A2A; autenticado pelo token exclusivo do contrato."""
        token = request.headers.get(NOTIFICATION_TOKEN_HEADER)
        try:
            # o token é conferido antes de ler o corpo: sem ele, nada é bufferizado
            await runtime.contracts.authorize_push(contract_id, token)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="contrato não encontrado") from exc
        except PushRejected as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_push_bytes:
            raise HTTPException(status_code=413, detail="notificação grande demais")
        raw = bytearray()
        async for chunk in request.stream():  # corpo sem Content-Length (chunked) também tem teto
            raw.extend(chunk)
            if len(raw) > settings.max_push_bytes:
                raise HTTPException(status_code=413, detail="notificação grande demais")
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="corpo não é JSON") from exc
        try:
            contract = await runtime.contracts.apply_push(contract_id, token, payload)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="contrato não encontrado") from exc
        except PushRejected as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "state": contract.state}

    return app
