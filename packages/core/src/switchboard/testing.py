"""Utilitários para testar roteadores sem rede nem modelos de verdade.

* :class:`ScriptedChat` — um "LLM" que devolve respostas roteirizadas;
* :func:`scripted_jev` — um Jev (System One) de mentira: uma função recebe o
  estado e as perguntas e devolve as respostas;
* :func:`inproc_connector` — conecta o catálogo MCP a servidores
  ``MCPServer`` no mesmo processo (sem HTTP);
* :class:`HostRoutingTransport` — transporte httpx que entrega cada host a um
  app ASGI em processo (agentes A2A e o próprio router, com push
  notifications), sem abrir portas;
* :class:`FakeA2AAgent` / :class:`FakeNetwork` — agente A2A 1.0 mínimo,
  controlado pelo teste (estados, artefatos, erros JSON-RPC, queda de rede).

Exemplo::

    catalog = ConnectorCatalog(connector=inproc_connector({"calc": meu_servidor}))
    chat = ScriptedChat(['{"action": "answer", "answer": "ok"}'])
    jev = scripted_jev(lambda state, questions: {"route": {"type": "choice", "choice": "knowledge", "confidence": 0.9}})
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx

from .a2a import A2AClient, AgentDirectory
from .config import ConnectorSpec
from .contracts import CONTRACT_EXTENSION_URI
from .jev import JevClient
from .llm.base import ChatResult, Message, Usage


class ScriptedChat:
    """Devolve respostas pré-definidas (ou lança exceções) e registra as chamadas."""

    offline = False

    def __init__(self, replies: Sequence[str | Exception], label: str = "scripted:model"):
        self.replies = list(replies)
        self.label = label
        self.calls: list[list[Message]] = []
        self.json_flags: list[bool] = []

    async def chat(
        self, messages: Sequence[Message], *, json_mode: bool = False, max_tokens: int | None = None
    ) -> ChatResult:
        self.calls.append(list(messages))
        self.json_flags.append(json_mode)
        if not self.replies:
            raise AssertionError("ScriptedChat sem respostas restantes")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(text=reply, model=self.label, usage=Usage(10, 5))

    async def aclose(self) -> None:
        return None


JevScript = Callable[[Any, dict[str, Any]], Mapping[str, Mapping[str, Any]]]


def scripted_jev(
    script: JevScript, *, model: str = "jev-test", calls: list | None = None
) -> JevClient:
    """Um :class:`JevClient` cujas respostas vêm de ``script(state, questions)``.

    ``script`` devolve ``{id_da_pergunta: resposta}`` no formato da API (por
    exemplo ``{"type": "noul", "noul": 0.9}``); perguntas sem resposta ficam
    de fora, e uma exceção vira HTTP 500. ``calls`` (opcional) recebe cada
    payload enviado.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if calls is not None:
            calls.append(payload)
        try:
            answers = dict(script(payload["state"], payload["questions"]))
        except Exception as exc:
            return httpx.Response(500, json={"error": {"message": str(exc)}})
        return httpx.Response(
            200,
            json={
                "model": model,
                "answers": answers,
                "usage": {"input_tokens": 100, "output_tokens": 5},
            },
        )

    return JevClient(
        model=model, api_key="teste", transport=httpx.MockTransport(handler), max_retries=0
    )


def jev_choice(
    key: str, confidence: float = 0.95, probabilities: dict[str, float] | None = None
) -> dict[str, Any]:
    return {
        "type": "choice",
        "choice": key,
        "confidence": confidence,
        "probabilities": probabilities or {key: confidence},
    }


def jev_noul(value: float) -> dict[str, Any]:
    return {"type": "noul", "noul": value}


def inproc_connector(servers: dict[str, Any]):
    """Connector do :class:`~switchboard.connectors.ConnectorCatalog` para servidores em processo."""
    from mcp import Client

    @asynccontextmanager
    async def connect(spec: ConnectorSpec, _token: str | None) -> AsyncIterator[Any]:
        if spec.name not in servers:
            raise ConnectionError(f"conector {spec.name} fora do ar")
        async with Client(servers[spec.name]) as client:
            yield client

    return connect


class HostRoutingTransport(httpx.AsyncBaseTransport):
    """Entrega cada requisição ao app ASGI registrado para o host (``nome:porta`` ou ``nome``)."""

    def __init__(self, apps: Mapping[str, Any] | None = None):
        self._apps: dict[str, httpx.ASGITransport] = {}
        for host, app in (apps or {}).items():
            self.mount(host, app)

    def mount(self, host: str, app: Any) -> None:
        self._apps[host.lower()] = httpx.ASGITransport(app=app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host.lower()
        port = request.url.port
        transport = self._apps.get(f"{host}:{port}") if port else None
        transport = transport or self._apps.get(host)
        if transport is None:
            raise httpx.ConnectError(
                f"nenhum app em processo para {request.url.host}", request=request
            )
        return await transport.handle_async_request(request)


# --------------------------------------------------------------------------
# agente A2A falso (JSON-RPC 1.0 sobre httpx.MockTransport)

RISK_INPUT = {
    "type": "object",
    "properties": {
        "cliente": {"type": "string", "minLength": 2, "description": "Nome do cliente"},
        "valor": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "Valor da operação em reais",
        },
    },
    "required": ["cliente", "valor"],
    "additionalProperties": False,
}
RISK_OUTPUT = {
    "type": "object",
    "properties": {
        "risco": {"type": "string", "enum": ["baixo", "medio", "alto"]},
        "score": {"type": "integer"},
    },
    "required": ["risco", "score"],
}


class FakeA2AAgent:
    """Agente A2A mínimo e controlável pelo teste.

    Por padrão, ``SendMessage`` cria a tarefa em ``WORKING``; o teste avança o
    estado com :meth:`complete`, :meth:`ask` etc. (o roteador vê por polling).
    ``on_send`` pode responder na hora (tarefa concluída ou ``message``).
    """

    def __init__(
        self,
        *,
        name: str = "risco",
        host: str = "risco.test",
        skills: dict[str, dict[str, Any]] | None = None,
        push: bool = False,
        extension: bool = True,
    ):
        self.name = name
        self.host = host
        self.skills = (
            skills
            if skills is not None
            else {
                "avaliar_risco": {
                    "description": "Avalia o risco de fraude de uma operação",
                    "terms": {
                        "input_schema": RISK_INPUT,
                        "output_schema": RISK_OUTPUT,
                        "max_duration_s": 120,
                    },
                    "examples": ["Verifique o risco de fraude de uma operação de 80 mil"],
                }
            }
        )
        self.push = push
        self.extension = extension
        self.tasks: dict[str, dict[str, Any]] = {}
        self.sent: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.canceled: list[str] = []
        self.errors: dict[str, dict[str, Any]] = {}
        self.on_send: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None
        self.up = True

    @property
    def url(self) -> str:
        return f"http://{self.host}"

    def card(self) -> dict[str, Any]:
        capabilities: dict[str, Any] = {"pushNotifications": self.push}
        if self.extension:
            capabilities["extensions"] = [
                {
                    "uri": CONTRACT_EXTENSION_URI,
                    "params": {
                        "skills": {k: v["terms"] for k, v in self.skills.items() if v.get("terms")}
                    },
                }
            ]
        return {
            "name": self.name,
            "description": f"agente {self.name} de teste",
            "version": "1.0.0",
            "supportedInterfaces": [
                {"url": f"{self.url}/", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}
            ],
            "capabilities": capabilities,
            "skills": [
                {
                    "id": k,
                    "name": k.replace("_", " "),
                    "description": v.get("description", ""),
                    "examples": v.get("examples", []),
                }
                for k, v in self.skills.items()
            ],
        }

    # -- controle do teste ---------------------------------------------------

    def status(self, task_id: str, state: str, text: str = "") -> None:
        status: dict[str, Any] = {"state": state, "timestamp": _now()}
        if text:
            status["message"] = {
                "messageId": uuid.uuid4().hex,
                "role": "ROLE_AGENT",
                "parts": [{"text": text}],
            }
        self.tasks[task_id]["status"] = status

    def complete(self, task_id: str, data: Any = None, text: str = "") -> None:
        parts = []
        if data is not None:
            parts.append({"data": data, "mediaType": "application/json"})
        if text:
            parts.append({"text": text})
        self.tasks[task_id]["artifacts"] = [
            {"artifactId": "a1", "name": "resultado", "parts": parts}
        ]
        self.status(task_id, "TASK_STATE_COMPLETED")

    def ask(self, task_id: str, question: str) -> None:
        self.status(task_id, "TASK_STATE_INPUT_REQUIRED", question)

    @property
    def last_task_id(self) -> str:
        return list(self.tasks)[-1]

    # -- servidor ------------------------------------------------------------

    def handler(self, request: httpx.Request) -> httpx.Response:
        if not self.up:
            raise httpx.ConnectError("agente fora do ar", request=request)
        if request.method == "GET" and request.url.path.endswith("/agent-card.json"):
            return httpx.Response(200, json=self.card())
        body = json.loads(request.content)
        self.headers.append(dict(request.headers))
        method, params, rid = body["method"], body.get("params") or {}, body["id"]
        if request.headers.get("A2A-Version") != "1.0":
            return self._error(rid, -32009, "versão", "VERSION_NOT_SUPPORTED")
        if method in self.errors:
            err = self.errors[method]
            return self._error(
                rid, err.get("code", -32603), err.get("message", "erro"), err.get("reason")
            )
        if method == "SendMessage":
            self.sent.append(params)
            message = params["message"]
            task_id = message.get("taskId") or uuid.uuid4().hex
            if task_id not in self.tasks:
                self.tasks[task_id] = {
                    "id": task_id,
                    "contextId": message.get("contextId") or uuid.uuid4().hex,
                    "status": {"state": "TASK_STATE_WORKING", "timestamp": _now()},
                    "history": [message],
                }
            else:
                self.tasks[task_id]["history"].append(message)
            override = self.on_send(params) if self.on_send else None
            if override and "message" in override:
                return self._ok(rid, override)
            if override:
                self.tasks[task_id].update(override)
            return self._ok(rid, {"task": self.tasks[task_id]})
        if method == "GetTask":
            task = self.tasks.get(params["id"])
            if task is None:
                return self._error(rid, -32001, "Task not found", "TASK_NOT_FOUND")
            return self._ok(rid, task)
        if method == "CancelTask":
            self.canceled.append(params["id"])
            if params["id"] in self.tasks:
                self.status(params["id"], "TASK_STATE_CANCELED")
            return self._ok(rid, self.tasks.get(params["id"], {}))
        return self._error(rid, -32601, "método desconhecido", None)

    @staticmethod
    def _ok(rid: str, result: Any) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": rid, "result": result})

    @staticmethod
    def _error(rid: str, code: int, message: str, reason: str | None) -> httpx.Response:
        error: dict[str, Any] = {"code": code, "message": message}
        if reason:
            error["data"] = [
                {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason}
            ]
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": rid, "error": error})


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class FakeNetwork:
    """Roteia requisições para agentes falsos pelo host."""

    def __init__(self, *agents: FakeA2AAgent):
        self.agents = {a.host: a for a in agents}

    def add(self, agent: FakeA2AAgent) -> FakeA2AAgent:
        self.agents[agent.host] = agent
        return agent

    def handler(self, request: httpx.Request) -> httpx.Response:
        agent = self.agents.get(request.url.host)
        if agent is None:
            raise httpx.ConnectError(f"{request.url.host} fora do ar", request=request)
        return agent.handler(request)

    def client(self) -> A2AClient:
        return A2AClient(http=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)))

    def directory(self, **kw) -> AgentDirectory:
        return AgentDirectory(client=self.client(), **kw)
