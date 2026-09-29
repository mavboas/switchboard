"""Agente A2A com o contrato fechado do Switchboard, sobre o ``a2a-sdk`` oficial.

Exemplo::

    agent = ContractAgent(
        name="analise-credito",
        description="Análise de propostas de crédito",
        url="http://localhost:8201",
    )

    @agent.skill(
        "analisar_proposta",
        name="Analisar proposta de crédito",
        description="Aplica a política de crédito e sugere limite e taxa.",
        input_schema={...},
        output_schema={...},
        max_duration_s=600,
    )
    async def analisar(ctx: SkillContext, dados: dict) -> SkillResult:
        await ctx.progress("consultando bureau")
        if dados["valor"] > 500_000 and not ctx.replies:
            ctx.require_input("Há imóvel em garantia? (sim/não)")
        return SkillResult({"decisao": "aprovado", ...}, text="Aprovado com limite de ...")

    app = agent.build_app()  # Starlette: card em /.well-known/agent-card.json e JSON-RPC em /

O lado do agente do contrato:

1. o Agent Card declara os schemas de cada skill na extensão
   ``urn:switchboard:a2a:contract:v1``;
2. ao receber uma tarefa, confere os termos (skill existe, hash dos schemas
   igual ao publicado, prazo não vencido) e valida a entrada — senão
   ``TASK_STATE_REJECTED`` com o motivo;
3. a saída da skill é validada contra o ``output_schema`` antes de sair: o
   agente falha em vez de entregar algo fora do contrato;
4. ``ctx.require_input(pergunta)`` põe a tarefa em ``INPUT_REQUIRED``; quando a
   resposta chega (mesma tarefa), a skill roda de novo com ``ctx.replies``.
"""

from __future__ import annotations

import hmac
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx
from a2a.helpers.proto_helpers import new_data_part, new_task_from_user_message, new_text_part
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks.base_push_notification_sender import BasePushNotificationSender
from a2a.server.tasks.inmemory_push_notification_config_store import (
    InMemoryPushNotificationConfigStore,
)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types.a2a_pb2 import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Role,
    TaskState,
)
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from switchboard.a2a.protocol import parts_data, parts_text
from switchboard.contracts.terms import (
    CONTRACT_EXTENSION_URI,
    SkillTerms,
    coerce_to_schema,
    normalize_numbers,
    schema_problems,
    validate,
)

log = logging.getLogger("switchboard.agentkit")

JSON = "application/json"
TEXT = "text/plain"


class InputRequired(Exception):  # noqa: N818 - é um sinal de controle, não um erro
    """Levantada por ``ctx.require_input``: a tarefa espera uma resposta do usuário."""

    def __init__(self, question: str):
        super().__init__(question)
        self.question = question


class SkillFailed(Exception):
    """Falha de negócio: a tarefa termina como ``FAILED`` com esta mensagem."""


@dataclass
class SkillResult:
    data: Any
    text: str = ""


@dataclass
class SkillContext:
    """O que a skill enxerga de uma tarefa."""

    task_id: str
    context_id: str
    skill: str
    terms: dict[str, Any]
    instruction: str
    replies: list[str]
    traceparent: str | None
    _updater: TaskUpdater = field(repr=False)

    async def progress(self, text: str) -> None:
        """Publica uma mensagem de progresso (estado ``WORKING``)."""
        await self._updater.update_status(
            TaskState.TASK_STATE_WORKING, self._updater.new_agent_message([new_text_part(text)])
        )

    def require_input(self, question: str) -> None:
        raise InputRequired(question)


SkillFn = Callable[[SkillContext, dict[str, Any]], Awaitable[SkillResult | Mapping[str, Any] | str]]


@dataclass
class Skill:
    id: str
    name: str
    description: str
    fn: SkillFn
    terms: SkillTerms
    examples: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


def _message_dict(message: Any) -> dict[str, Any]:
    return MessageToDict(message) if message is not None else {}


def _parse_deadline(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class ContractExecutor(AgentExecutor):
    def __init__(self, agent: ContractAgent):
        self.agent = agent

    async def execute(self, context, event_queue) -> None:  # noqa: C901 - fluxo linear do contrato
        incoming = context.message
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(incoming)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        # a primeira mensagem do usuário carrega os termos e a entrada; as
        # seguintes são respostas a pedidos de entrada
        history = [_message_dict(m) for m in task.history]
        current = _message_dict(incoming)
        seen = {m.get("messageId") for m in history}
        messages = history + ([current] if current.get("messageId") not in seen else [])
        user_messages = [m for m in messages if m.get("role") == "ROLE_USER"]
        first = user_messages[0] if user_messages else current
        replies = [parts_text(m.get("parts")) for m in user_messages[1:]]
        terms = dict((first.get("metadata") or {}).get(CONTRACT_EXTENSION_URI) or {})
        request_meta = dict(context.metadata or {})

        async def reject(reason: str) -> None:
            log.info("tarefa %s rejeitada: %s", task.id, reason)
            await updater.reject(updater.new_agent_message([new_text_part(reason)]))

        skill_id = str(terms.get("skill") or "")
        if not skill_id and len(self.agent.skills) == 1:
            skill_id = next(iter(self.agent.skills))
        skill = self.agent.skills.get(skill_id)
        if skill is None:
            await reject(f"skill desconhecida: {skill_id or '(não informada)'}")
            return
        if not terms and self.agent.require_contract:
            await reject(
                f"esta skill exige o contrato {CONTRACT_EXTENSION_URI} (termos na metadata da mensagem)"
            )
            return
        if terms:
            for key, expected in (
                ("input_schema_sha256", skill.terms.input_hash),
                ("output_schema_sha256", skill.terms.output_hash),
            ):
                given = terms.get(key)
                if expected and given != expected:
                    await reject(
                        f"contrato desatualizado: {key} não confere (esperado {expected}, veio {given}); "
                        "refaça a descoberta do agente"
                    )
                    return
            deadline = _parse_deadline(terms.get("deadline"))
            if deadline is not None and deadline <= datetime.now(UTC):
                await reject("o prazo do contrato já venceu")
                return
        data_items = parts_data(first.get("parts"))
        data = normalize_numbers(data_items[0]) if data_items else {}
        if not isinstance(data, dict):
            await reject("a entrada precisa ser um objeto JSON (parte 'data')")
            return
        if skill.terms.input_schema is not None:
            data = coerce_to_schema(data, skill.terms.input_schema)
            errors = validate(data, skill.terms.input_schema)
            if errors:
                await reject("entrada fora do contrato: " + "; ".join(errors[:5]))
                return

        ctx = SkillContext(
            task_id=task.id,
            context_id=task.context_id,
            skill=skill.id,
            terms=terms,
            instruction=parts_text(first.get("parts")),
            replies=replies,
            traceparent=request_meta.get("traceparent"),
            _updater=updater,
        )
        await updater.start_work(
            updater.new_agent_message([new_text_part(f"{skill.name}: iniciado")])
        )
        try:
            result = await skill.fn(ctx, data)
        except InputRequired as ask:
            await updater.requires_input(updater.new_agent_message([new_text_part(ask.question)]))
            return
        except SkillFailed as exc:
            await updater.failed(updater.new_agent_message([new_text_part(str(exc))]))
            return
        except Exception as exc:  # erro inesperado da skill: falha com o motivo
            log.exception("skill %s falhou", skill.id)
            await updater.failed(
                updater.new_agent_message([new_text_part(f"erro interno do agente: {exc}")])
            )
            return

        if isinstance(result, SkillResult):
            output, text = result.data, result.text
        elif isinstance(result, str):
            output, text = None, result
        else:
            output, text = dict(result), ""
        parts = []
        if output is not None:
            output = normalize_numbers(output)
            if skill.terms.output_schema is not None:
                errors = validate(output, skill.terms.output_schema)
                if errors:
                    # o agente não entrega algo fora do contrato
                    await updater.failed(
                        updater.new_agent_message(
                            [
                                new_text_part(
                                    "saída fora do contrato (erro do agente): "
                                    + "; ".join(errors[:5])
                                )
                            ]
                        )
                    )
                    return
            parts.append(new_data_part(output, media_type=JSON))
        elif skill.terms.output_schema is not None:
            await updater.failed(
                updater.new_agent_message(
                    [new_text_part("a skill não produziu a saída do contrato")]
                )
            )
            return
        if text:
            parts.append(new_text_part(text))
        await updater.add_artifact(parts, name="resultado")
        await updater.complete()

    async def cancel(self, context, event_queue) -> None:
        task = context.current_task
        if task is None:
            return
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.cancel(updater.new_agent_message([new_text_part("tarefa cancelada")]))


class _BearerAuth:
    """Exige ``Authorization: Bearer <token>`` no JSON-RPC; o Agent Card continua público."""

    def __init__(self, app: ASGIApp, tokens: Iterable[str]):
        self.app = app
        self.tokens = [t.encode("utf-8") for t in tokens]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and not scope["path"].startswith("/.well-known/"):
            header = dict(scope.get("headers") or []).get(b"authorization", b"")
            given = header[7:].strip() if header[:7].lower() == b"bearer " else b""
            if not any(hmac.compare_digest(given, t) for t in self.tokens):
                response = JSONResponse({"error": "token ausente ou inválido"}, status_code=401)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


class ContractAgent:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        url: str,
        version: str = "1.0.0",
        organization: str | None = None,
        require_contract: bool = True,
        push_hosts: Iterable[str] | None = None,
        auth_tokens: Iterable[str] | None = None,
        http: httpx.AsyncClient | None = None,
    ):
        """
        Args:
            url: URL pública do agente (vai no card como endpoint JSON-RPC).
            require_contract: rejeita tarefas sem os termos do contrato.
            push_hosts: destinos aceitos nas URLs de push notification: ``host``
                (qualquer porta) ou ``host:porta``; ``None`` aceita qualquer um —
                só para desenvolvimento.
            auth_tokens: tokens aceitos em ``Authorization: Bearer`` no JSON-RPC
                (o do cadastro do agente no roteador); vazio = sem autenticação.
            http: cliente usado para enviar as push notifications.
        """
        self.name = name
        self.description = description
        self.url = url.rstrip("/") + "/"
        self.version = version
        self.organization = organization
        self.require_contract = require_contract
        self.push_hosts = {h.lower() for h in push_hosts} if push_hosts is not None else None
        self.auth_tokens = [t for t in (auth_tokens or ()) if t]
        self.http = http
        self.skills: dict[str, Skill] = {}

    def skill(
        self,
        skill_id: str,
        *,
        name: str,
        description: str,
        input_schema: Mapping[str, Any],
        output_schema: Mapping[str, Any],
        max_duration_s: float | None = None,
        examples: Iterable[str] = (),
        tags: Iterable[str] = (),
    ) -> Callable[[SkillFn], SkillFn]:
        for label, schema in (("input_schema", input_schema), ("output_schema", output_schema)):
            problems = schema_problems(schema)
            if problems:
                raise ValueError(f"skill {skill_id}: {label} inválido: {'; '.join(problems)}")
        terms = SkillTerms(
            normalize_numbers(dict(input_schema)),
            normalize_numbers(dict(output_schema)),
            max_duration_s,
        )

        def register(fn: SkillFn) -> SkillFn:
            self.skills[skill_id] = Skill(
                skill_id, name, description, fn, terms, tuple(examples), tuple(tags)
            )
            return fn

        return register

    def card(self) -> AgentCard:
        params = Struct()
        params.update(
            {
                "version": "1",
                "skills": {s.id: s.terms.to_params() for s in self.skills.values()},
            }
        )
        extra: dict[str, Any] = {}
        if self.organization:
            extra["provider"] = AgentProvider(organization=self.organization, url=self.url)
        return AgentCard(
            name=self.name,
            description=self.description,
            version=self.version,
            **extra,
            supported_interfaces=[
                AgentInterface(url=self.url, protocol_binding="JSONRPC", protocol_version="1.0")
            ],
            capabilities=AgentCapabilities(
                streaming=False,
                push_notifications=True,
                extensions=[
                    AgentExtension(
                        uri=CONTRACT_EXTENSION_URI,
                        description="Contrato fechado do Switchboard: schemas de entrada e saída por skill",
                        required=self.require_contract,
                        params=params,
                    )
                ],
            ),
            default_input_modes=[JSON, TEXT],
            default_output_modes=[JSON, TEXT],
            skills=[
                AgentSkill(
                    id=s.id,
                    name=s.name,
                    description=s.description,
                    tags=list(s.tags),
                    examples=list(s.examples),
                    input_modes=[JSON],
                    output_modes=[JSON, TEXT],
                )
                for s in self.skills.values()
            ],
        )

    async def _push_url_ok(self, url: str) -> bool:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            return False
        if self.push_hosts is None:
            return True
        host = (parts.hostname or "").lower()
        port = parts.port or {"http": 80, "https": 443}[parts.scheme]
        return host in self.push_hosts or f"{host}:{port}" in self.push_hosts

    def build_app(self) -> Starlette:
        card = self.card()
        tasks = InMemoryTaskStore()
        push_store = InMemoryPushNotificationConfigStore()
        http = self.http or httpx.AsyncClient(timeout=10.0)
        handler = DefaultRequestHandlerV2(
            agent_executor=ContractExecutor(self),
            task_store=tasks,
            agent_card=card,
            push_config_store=push_store,
            push_sender=BasePushNotificationSender(
                http, push_store, push_url_validator=self._push_url_ok
            ),
            push_url_validator=self._push_url_ok,
        )
        routes = create_agent_card_routes(card) + create_jsonrpc_routes(handler, rpc_url="/")
        app = Starlette(routes=routes)
        app.state.agent = self
        if self.auth_tokens:
            app.add_middleware(_BearerAuth, tokens=self.auth_tokens)
        return app

    def run(self, host: str = "127.0.0.1", port: int = 8201) -> None:  # pragma: no cover
        import uvicorn

        uvicorn.run(self.build_app(), host=host, port=port, log_level="info")


__all__ = [
    "ContractAgent",
    "ContractExecutor",
    "InputRequired",
    "Role",
    "Skill",
    "SkillContext",
    "SkillFailed",
    "SkillResult",
]
