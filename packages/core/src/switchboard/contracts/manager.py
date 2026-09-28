"""Ciclo de vida dos contratos com agentes A2A.

O :class:`ContractManager` é o único que fala com os agentes depois que o
roteador decide delegar:

1. **abrir** — valida a entrada contra o schema da skill, grava o contrato
   (``proposto``) *antes* de enviar (a push notification pode chegar antes da
   resposta) e faz o ``SendMessage`` com os termos na ``metadata``;
2. **acompanhar** — aplica push notifications (token exclusivo do contrato) e,
   como reserva, faz polling com ``GetTask`` (backoff); a saída é validada
   contra o ``output_schema`` e, se não bater, o contrato fica ``violado``;
3. **encerrar** — prazo vencido vira ``expirado`` (com ``CancelTask``);
4. **consolidar** — quando todos os contratos de uma execução terminam, uma
   única consolidação acontece (``update_run`` condicional) e quem espera é
   avisado; se sobrar só contrato aguardando entrada, a execução fica
   ``needs_input`` com a pergunta do agente.

O supervisor (``start``/``tick``) roda no router e retoma sozinho, depois de
um restart, tudo o que estava aberto — o agendamento fica no próprio contrato.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from ..a2a.client import A2AClient
from ..a2a.protocol import (
    StreamEvent,
    data_part,
    merge_artifact,
    new_message,
    parse_stream_event,
    parse_timestamp,
    parts_data,
    parts_text,
    text_part,
)
from ..config import AgentSpec
from ..errors import AgentError, ContractError
from ..secrets import resolve_env
from ..tracing import Span, new_span_id, traceparent, utcnow
from . import states
from .models import (
    RUN_COMPLETED,
    RUN_CONSOLIDATING,
    RUN_FAILED,
    RUN_FINAL,
    RUN_NEEDS_INPUT,
    RUN_PENDING,
    ContractEvent,
    ContractRecord,
    RunRecord,
    new_contract_id,
    new_push_token,
    token_hash,
    token_matches,
)
from .store import ContractStore
from .terms import (
    CONTRACT_EXTENSION_URI,
    CONTRACT_VERSION,
    coerce_to_schema,
    normalize_numbers,
    validate,
)

if TYPE_CHECKING:  # evita import circular (o diretório lê os termos daqui)
    from ..a2a.directory import AgentInfo, SkillInfo

log = logging.getLogger("switchboard.contracts")

ACCEPTED_OUTPUT_MODES = ["application/json", "text/plain"]


@dataclass
class OpenRequest:
    """O que o motor pede ao abrir um contrato com a skill de um agente."""

    run_id: str
    profile: str
    spec: AgentSpec
    agent: AgentInfo
    skill: SkillInfo
    arguments: dict[str, Any]
    instruction: str
    deadline_s: float
    parent_span_id: str | None = None


@dataclass
class Consolidation:
    answer: str
    spans: list[Span] = field(default_factory=list)


Consolidator = Callable[[RunRecord, list[ContractRecord]], Awaitable[Consolidation]]
AgentResolver = Callable[[str], Awaitable[AgentSpec | None]]


class PushRejected(Exception):
    """Push notification recusada (token inválido ou contrato sem push)."""


def contract_terms(contract: ContractRecord, deadline: datetime | None) -> dict[str, Any]:
    """Os termos que viajam em ``message.metadata[URI]``."""
    terms: dict[str, Any] = {
        "contract_id": contract.id,
        "version": CONTRACT_VERSION,
        "skill": contract.skill,
        "kind": contract.kind,
        "deadline": deadline.isoformat() if deadline else None,
        "caller": {"system": "switchboard", "profile": contract.profile, "run_id": contract.run_id},
    }
    if contract.input_hash:
        terms["input_schema_sha256"] = contract.input_hash
    if contract.output_hash:
        terms["output_schema_sha256"] = contract.output_hash
    return terms


def default_consolidation(run: RunRecord, contracts: list[ContractRecord]) -> str:
    """Consolidação sem LLM: um parágrafo por contrato."""
    lines = []
    for c in contracts:
        who = f"{c.agent} ({c.skill})"
        if c.state == states.COMPLETED:
            body = c.output_text.strip()
            if not body and c.output is not None:
                body = json.dumps(c.output, ensure_ascii=False, indent=2)
            lines.append(f"{who}:\n{body or 'concluído, sem detalhes.'}")
        else:
            label = states.LABELS.get(c.state, c.state)
            lines.append(f"{who}: {label}" + (f" — {c.error}" if c.error else "."))
    return "\n\n".join(lines) or "Nenhum agente respondeu."


def input_question(contracts: list[ContractRecord]) -> str:
    asks = [c for c in contracts if c.state == states.INPUT_REQUIRED]
    if len(asks) == 1:
        c = asks[0]
        return c.last_message or f"O agente {c.agent} precisa de mais informações para continuar."
    lines = ["Os agentes precisam de mais informações para continuar:"]
    lines += [f"- {c.agent}: {c.last_message or 'sem detalhes'}" for c in asks]
    return "\n".join(lines)


class ContractManager:
    def __init__(
        self,
        store: ContractStore,
        *,
        client: A2AClient | None = None,
        resolve_secret: Callable[[str | None], str | None] = resolve_env,
        agent_resolver: AgentResolver | None = None,
        public_url: str | None = None,
        clock: Callable[[], datetime] = utcnow,
        tick_s: float = 1.0,
        push_check_s: float = 20.0,
        poll_min_s: float = 1.0,
        poll_max_s: float = 15.0,
        lease_s: float = 30.0,
        proposed_timeout_s: float = 60.0,
        callback_http: httpx.AsyncClient | None = None,
    ):
        self.store = store
        self.client = client or A2AClient()
        self._resolve = resolve_secret
        self._agent_resolver = agent_resolver
        self.public_url = public_url.rstrip("/") if public_url else None
        self._clock = clock
        self.tick_s = tick_s
        self.push_check_s = push_check_s
        self.poll_min_s = poll_min_s
        self.poll_max_s = poll_max_s
        self.lease_s = lease_s
        self.proposed_timeout_s = proposed_timeout_s
        self.consolidator: Consolidator | None = None
        self._callback_http = callback_http
        self._run_events: dict[str, asyncio.Event] = {}
        self._inflight: set[str] = set()
        self._background: set[asyncio.Task[Any]] = set()
        self._supervisor: asyncio.Task[None] | None = None
        self._semaphore = asyncio.Semaphore(8)

    # ------------------------------------------------------------------ helpers

    def _spawn(self, coro: Awaitable[Any]) -> None:
        task = asyncio.get_running_loop().create_task(coro)  # type: ignore[arg-type]
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _spec(self, name: str) -> AgentSpec | None:
        if self._agent_resolver is None:
            return None
        try:
            return await self._agent_resolver(name)
        except Exception:  # agente removido, banco fora: segue sem token
            log.exception("não consegui resolver o agente %s", name)
            return None

    def _token(self, spec: AgentSpec | None) -> str | None:
        return self._resolve(spec.auth_token) if spec is not None else None

    def _run_event(self, run_id: str) -> asyncio.Event:
        return self._run_events.setdefault(run_id, asyncio.Event())

    def _signal(self, run_id: str) -> None:
        self._run_event(run_id).set()

    # --------------------------------------------------------------- execuções

    async def begin_run(self, run: RunRecord) -> None:
        await self.store.create_run(run)
        self._run_events[run.id] = asyncio.Event()

    async def wait_run(self, run_id: str, timeout_s: float) -> RunRecord | None:
        """Espera a execução terminar (ou pedir entrada) por até ``timeout_s``.

        Devolve o registro no estado em que estiver ao fim da espera.
        """
        loop = asyncio.get_running_loop()
        end = loop.time() + max(timeout_s, 0.0)
        event = self._run_event(run_id)
        while True:
            run = await self.store.get_run(run_id)
            if run is None or run.status in RUN_FINAL or run.status == RUN_NEEDS_INPUT:
                return run
            remaining = end - loop.time()
            if remaining <= 0:
                return run
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(event.wait(), timeout=min(remaining, 0.5))

    # --------------------------------------------------------------- abertura

    async def open(self, req: OpenRequest) -> ContractRecord:
        """Abre um contrato: valida, grava, envia e aplica a resposta do agente."""
        now = self._clock()
        terms = req.skill.terms
        arguments = dict(req.arguments or {})
        if terms is not None and terms.input_schema is not None:
            arguments = coerce_to_schema(arguments, terms.input_schema)
            errors = validate(arguments, terms.input_schema)
            if errors:
                raise ContractError("entrada fora do contrato: " + "; ".join(errors))
        if not req.agent.rpc_url:
            raise ContractError(f"o agente {req.spec.name} não tem endpoint JSON-RPC")
        # prazo: o fixado no cadastro do agente; senão o menor entre o compromisso
        # da skill (max_duration_s do card) e o prazo do perfil
        limits = [d for d in ((terms.max_duration_s if terms else None), req.deadline_s) if d]
        deadline_s = req.spec.deadline_s or (min(limits) if limits else 600.0)
        use_push = bool(req.spec.push and req.agent.push and self.public_url)
        token = new_push_token() if use_push else None
        contract = ContractRecord(
            id=new_contract_id(),
            run_id=req.run_id,
            profile=req.profile,
            agent=req.spec.name,
            skill=req.skill.id,
            kind=req.skill.contract,
            rpc_url=req.agent.rpc_url,
            input=arguments,
            input_text=req.instruction,
            input_schema=terms.input_schema if terms else None,
            output_schema=terms.output_schema if terms else None,
            input_hash=terms.input_hash if terms else None,
            output_hash=terms.output_hash if terms else None,
            deadline_at=now + timedelta(seconds=deadline_s),
            reply_mode="push" if use_push else "poll",
            push_token_hash=token_hash(token) if token else None,
            created_at=now,
            updated_at=now,
            next_check_at=now
            + timedelta(seconds=self.push_check_s if use_push else self.poll_min_s),
            span_id=new_span_id(),
            parent_span_id=req.parent_span_id,
        )
        await self.store.add_contract(
            contract,
            [
                ContractEvent(
                    contract.id,
                    "state",
                    "router",
                    states.PROPOSED,
                    {
                        "contrato": contract.kind,
                        "prazo": contract.deadline_at.isoformat() if contract.deadline_at else None,
                        "retorno": contract.reply_mode,
                        "input_schema_sha256": contract.input_hash,
                        "output_schema_sha256": contract.output_hash,
                    },
                    at=now,
                )
            ],
        )
        extensions = [CONTRACT_EXTENSION_URI] if contract.kind == "completo" else None
        parts = []
        if contract.kind == "completo":
            parts.append(data_part(arguments))
        if req.instruction.strip():
            parts.append(text_part(req.instruction.strip()))
        if not parts:
            parts.append(text_part("(sem instruções)"))
        message = new_message(
            parts,
            metadata={CONTRACT_EXTENSION_URI: contract_terms(contract, contract.deadline_at)},
            extensions=extensions,
        )
        configuration: dict[str, Any] = {
            "acceptedOutputModes": ACCEPTED_OUTPUT_MODES,
            "returnImmediately": True,
        }
        if use_push:
            configuration["taskPushNotificationConfig"] = {
                "url": f"{self.public_url}/a2a/push/{contract.id}",
                "token": token,
            }
        parent = traceparent(req.run_id, contract.span_id)
        try:
            result = await self.client.send_message(
                req.agent.rpc_url,
                message,
                configuration=configuration,
                metadata={
                    "traceparent": parent,
                    "switchboard": {"run_id": req.run_id, "contract_id": contract.id},
                },
                token=self._token(req.spec),
                extensions=extensions,
                headers={"traceparent": parent},
                timeout_s=req.spec.timeout_s,
            )
        except AgentError as exc:
            # erro JSON-RPC = o agente recusou; erro de rede = falha
            target = states.REJECTED if exc.code is not None else states.FAILED
            await self._transition(contract.id, target, "response", error=str(exc))
            return await self._reload(contract.id, contract)
        try:
            event = parse_stream_event(result)
        except ValueError as exc:
            await self._transition(contract.id, states.BREACHED, "response", error=str(exc))
            return await self._reload(contract.id, contract)
        await self.apply_event(contract.id, event, source="response")
        return await self._reload(contract.id, contract)

    async def _reload(self, contract_id: str, fallback: ContractRecord) -> ContractRecord:
        return await self.store.get_contract(contract_id) or fallback

    # ------------------------------------------------------ aplicação de eventos

    async def apply_push(
        self, contract_id: str, token: str | None, payload: Mapping[str, Any]
    ) -> ContractRecord:
        contract = await self.store.get_contract(contract_id)
        if contract is None:
            raise LookupError(f"contrato {contract_id} não existe")
        if contract.reply_mode != "push" or not token_matches(token, contract.push_token_hash):
            raise PushRejected("token de notificação inválido para este contrato")
        event = parse_stream_event(payload)  # ValueError = payload inválido
        return await self.apply_event(contract_id, event, source="push") or contract

    async def apply_event(
        self,
        contract_id: str,
        event: StreamEvent,
        *,
        source: str,
        reschedule_s: float | None = None,
    ) -> ContractRecord | None:
        event = await self._complete_snapshot(contract_id, event, source)
        for _attempt in range(6):
            contract = await self.store.get_contract(contract_id)
            if contract is None:
                return None
            before = contract.state
            events = self._reduce(contract, event, source, reschedule_s)
            if events is None:
                return contract
            if await self.store.save_contract(contract, events):
                if contract.state != before and (
                    contract.terminal or contract.state == states.INPUT_REQUIRED
                ):
                    await self._check_run(contract.run_id)
                return contract
        log.warning("contrato %s: conflito ao aplicar evento de %s", contract_id, source)
        return None

    async def _complete_snapshot(
        self, contract_id: str, event: StreamEvent, source: str
    ) -> StreamEvent:
        """Status COMPLETED sem artefatos conhecidos: busca a tarefa inteira antes."""
        if event.kind != "status" or event.state is None:
            return event
        if states.from_task_state(event.state) != states.COMPLETED:
            return event
        contract = await self.store.get_contract(contract_id)
        if (
            contract is None
            or contract.artifacts
            or not contract.remote_task_id
            or not contract.rpc_url
        ):
            return event
        spec = await self._spec(contract.agent)
        try:
            task = await self.client.get_task(
                contract.rpc_url,
                contract.remote_task_id,
                token=self._token(spec),
                timeout_s=spec.timeout_s if spec else None,
            )
            return parse_stream_event({"task": task})
        except (AgentError, ValueError) as exc:
            log.warning("contrato %s: não consegui buscar os artefatos: %s", contract_id, exc)
            return event

    def _reduce(
        self,
        c: ContractRecord,
        event: StreamEvent,
        source: str,
        reschedule_s: float | None,
    ) -> list[ContractEvent] | None:
        now = self._clock()
        out: list[ContractEvent] = []
        changed = False
        if event.task_id and c.remote_task_id and event.task_id != c.remote_task_id:
            out.append(
                ContractEvent(
                    c.id,
                    "violation",
                    source,
                    c.state,
                    {"motivo": "evento de outra tarefa ignorado", "task_id": event.task_id},
                    at=now,
                )
            )
            return out
        if c.terminal:
            if source == "push":
                out.append(
                    ContractEvent(
                        c.id,
                        "note",
                        source,
                        c.state,
                        {"motivo": "evento depois do encerramento ignorado", "a2a": event.state},
                        at=now,
                    )
                )
            return out or None
        if event.task_id and not c.remote_task_id:
            c.remote_task_id, changed = event.task_id, True
        if event.context_id and not c.remote_context_id:
            c.remote_context_id, changed = event.context_id, True
        if self._stale_input_request(c, event, source):
            # retrato de antes da entrada chegar ao agente: não é um pedido novo
            if reschedule_s is not None:
                c.next_check_at = now + timedelta(seconds=reschedule_s)
                c.checks += 1
                changed = True
            return out if (out or changed) else None
        if event.kind == "artifact" and event.artifact:
            c.artifacts = merge_artifact(c.artifacts, event.artifact, append=event.append)
            changed = True
            out.append(
                ContractEvent(
                    c.id,
                    "artifact",
                    source,
                    c.state,
                    {
                        "nome": event.artifact.get("name"),
                        "partes": len(event.artifact.get("parts") or []),
                        "append": event.append,
                    },
                    at=now,
                )
            )
        if event.kind == "task" and event.task is not None and event.task.artifacts:
            c.artifacts = [dict(a) for a in event.task.artifacts]  # a tarefa inteira manda
            changed = True
        if event.status_text and event.status_text != c.last_message:
            c.last_message = event.status_text[:2000]
            changed = True
            out.append(
                ContractEvent(
                    c.id, "message", source, c.state, {"texto": event.status_text[:2000]}, at=now
                )
            )
        if event.kind == "message" and event.message is not None:
            c.artifacts = merge_artifact(
                c.artifacts,
                {"artifactId": "mensagem", "parts": event.message.get("parts") or []},
                append=False,
            )
            target: str | None = states.COMPLETED
        else:
            target = states.from_task_state(event.state) if event.state else None
        if target is not None and target != c.state:
            error: str | None = None
            if target == states.COMPLETED:
                target, error = self._finish_output(c)
                if target == states.COMPLETED and c.output_text.strip():
                    # a última palavra do agente é o resultado, não o último progresso
                    c.last_message = c.output_text.strip()[:2000]
            if not states.can_move(c.state, target):
                out.append(
                    ContractEvent(
                        c.id,
                        "violation",
                        source,
                        c.state,
                        {"motivo": f"transição proibida {c.state} → {target}"},
                        at=now,
                    )
                )
            elif target != c.state:
                self._move(c, target, now, error=error)
                changed = True
                out.append(
                    ContractEvent(
                        c.id,
                        "state",
                        source,
                        target,
                        {"a2a": event.state, **({"erro": error} if error else {})},
                        at=now,
                    )
                )
        if not c.terminal:
            if reschedule_s is not None:
                c.next_check_at = now + timedelta(seconds=reschedule_s)
                c.checks += 1
                changed = True
            elif source == "push":
                c.next_check_at = now + timedelta(seconds=self.push_check_s)
                changed = True
        return out if (out or changed) else None

    @staticmethod
    def _stale_input_request(c: ContractRecord, event: StreamEvent, source: str) -> bool:
        """``INPUT_REQUIRED`` que só repete o estado de antes da resposta do usuário.

        Com ``returnImmediately``, a resposta ao envio da entrada é o retrato da
        tarefa antes de o agente processá-la; um polling logo depois também
        pode vê-lo. Um pedido de entrada novo tem carimbo de tempo posterior.
        """
        if c.input_sent_at is None or event.state is None:
            return False
        if states.from_task_state(event.state) != states.INPUT_REQUIRED:
            return False
        if source == "response":
            return True
        stamp = parse_timestamp(event.timestamp)
        return stamp is not None and stamp <= c.input_sent_at

    def _move(self, c: ContractRecord, target: str, now: datetime, *, error: str | None) -> None:
        c.state = target
        if target in (states.ACTIVE, states.INPUT_REQUIRED) and c.accepted_at is None:
            c.accepted_at = now
        if target in states.TERMINAL:
            c.finished_at = now
            c.next_check_at = None
            if error:
                c.error = error[:2000]
            elif target != states.COMPLETED and not c.error:
                c.error = c.last_message or f"agente encerrou como {states.LABELS[target]}"

    def _finish_output(self, c: ContractRecord) -> tuple[str, str | None]:
        """Extrai e valida a saída; devolve o estado final e o erro, se houver."""
        data_items: list[Any] = []
        texts: list[str] = []
        for artifact in c.artifacts:
            parts = artifact.get("parts") or []
            data_items.extend(parts_data(parts))
            text = parts_text(parts)
            if text:
                texts.append(text)
        c.output_text = ("\n\n".join(texts) or c.last_message or "")[:20000]
        output = normalize_numbers(data_items[-1]) if data_items else None
        if c.output_schema is not None:
            if output is None:
                c.output = None
                return states.BREACHED, "o agente não devolveu a parte 'data' exigida pelo contrato"
            output = coerce_to_schema(output, c.output_schema)
            c.output = output
            errors = validate(output, c.output_schema)
            if errors:
                return states.BREACHED, "saída fora do contrato: " + "; ".join(errors[:5])
            return states.COMPLETED, None
        c.output = output
        return states.COMPLETED, None

    async def _transition(
        self,
        contract_id: str,
        target: str,
        source: str,
        *,
        error: str | None = None,
        detail: dict | None = None,
        input_sent: bool = False,
    ) -> ContractRecord | None:
        for _attempt in range(6):
            c = await self.store.get_contract(contract_id)
            if c is None or c.terminal:
                return c
            if not states.can_move(c.state, target):
                return c
            now = self._clock()
            self._move(c, target, now, error=error)
            if input_sent:
                c.input_sent_at = now
                c.next_check_at = now + timedelta(
                    seconds=self.push_check_s if c.reply_mode == "push" else self.poll_min_s
                )
                c.checks = 0
            event = ContractEvent(
                c.id,
                "state",
                source,
                target,
                {**(detail or {}), **({"erro": error} if error else {})},
                at=now,
            )
            if await self.store.save_contract(c, [event]):
                if c.terminal or c.state == states.INPUT_REQUIRED:
                    await self._check_run(c.run_id)
                return c
        return None

    async def _note(
        self,
        contract_id: str,
        source: str,
        detail: dict[str, Any],
        *,
        reschedule_s: float | None = None,
    ) -> None:
        for _attempt in range(6):
            c = await self.store.get_contract(contract_id)
            if c is None:
                return
            now = self._clock()
            if reschedule_s is not None and not c.terminal:
                c.next_check_at = now + timedelta(seconds=reschedule_s)
                c.checks += 1
            if await self.store.save_contract(
                c, [ContractEvent(c.id, "note", source, c.state, detail, at=now)]
            ):
                return

    # ------------------------------------------------------ polling e prazos

    def _poll_interval(self, c: ContractRecord) -> float:
        if c.reply_mode == "push":
            return min(self.push_check_s * (2 ** min(c.checks, 3)), 120.0)
        return min(self.poll_min_s * (1.5 ** min(c.checks, 12)), self.poll_max_s)

    async def check(self, contract: ContractRecord) -> None:
        """Polling de reserva: consulta a tarefa e aplica o que mudou."""
        now = self._clock()
        if contract.deadline_at is not None and now >= contract.deadline_at:
            await self.expire(contract)
            return
        if not contract.remote_task_id or not contract.rpc_url:
            age = (now - contract.created_at).total_seconds()
            if contract.state == states.PROPOSED and age >= self.proposed_timeout_s:
                await self._transition(
                    contract.id,
                    states.FAILED,
                    "poll",
                    error="o agente não confirmou o recebimento da tarefa",
                )
            else:
                await self._note(
                    contract.id,
                    "poll",
                    {"motivo": "aguardando o id da tarefa"},
                    reschedule_s=self._poll_interval(contract),
                )
            return
        spec = await self._spec(contract.agent)
        try:
            task = await self.client.get_task(
                contract.rpc_url,
                contract.remote_task_id,
                token=self._token(spec),
                timeout_s=spec.timeout_s if spec else None,
            )
            event = parse_stream_event({"task": task})
        except (AgentError, ValueError) as exc:
            if isinstance(exc, AgentError) and exc.reason == "TASK_NOT_FOUND":
                await self._transition(
                    contract.id, states.FAILED, "poll", error="o agente não encontra mais a tarefa"
                )
                return
            await self._note(
                contract.id,
                "poll",
                {"motivo": "falha no polling", "erro": str(exc)[:300]},
                reschedule_s=self._poll_interval(contract),
            )
            return
        await self.apply_event(
            contract.id, event, source="poll", reschedule_s=self._poll_interval(contract)
        )

    async def expire(self, contract: ContractRecord) -> None:
        c = await self._transition(
            contract.id, states.EXPIRED, "router", error="o prazo do contrato venceu"
        )
        if c is not None and c.state == states.EXPIRED:
            await self._cancel_remote(c)

    async def _cancel_remote(self, c: ContractRecord) -> None:
        if not c.remote_task_id or not c.rpc_url:
            return
        spec = await self._spec(c.agent)
        try:
            await self.client.cancel_task(
                c.rpc_url,
                c.remote_task_id,
                token=self._token(spec),
                timeout_s=spec.timeout_s if spec else 5.0,
            )
        except AgentError as exc:
            await self._note(
                c.id, "router", {"motivo": "CancelTask falhou", "erro": str(exc)[:300]}
            )

    async def cancel_run(
        self, run_id: str, reason: str = "cancelado a pedido do cliente"
    ) -> list[ContractRecord]:
        out = []
        for c in await self.store.run_contracts(run_id):
            if c.terminal:
                continue
            moved = await self._transition(c.id, states.CANCELED, "router", error=reason)
            if moved is not None and moved.state == states.CANCELED:
                await self._cancel_remote(moved)
                out.append(moved)
        return out

    async def provide_input(self, run_id: str, text: str) -> list[ContractRecord]:
        """Leva a resposta do usuário aos contratos que aguardam entrada."""
        waiting = [
            c for c in await self.store.run_contracts(run_id) if c.state == states.INPUT_REQUIRED
        ]
        if not waiting:
            return []
        self._run_events[run_id] = asyncio.Event()
        agents = ", ".join(dict.fromkeys(c.agent for c in waiting))
        await self.store.update_run(
            run_id,
            expect={RUN_NEEDS_INPUT},
            status=RUN_PENDING,
            # a pergunta do agente já foi respondida: não pode continuar como resposta provisória
            answer=(
                f"Recebi sua resposta e repassei para {agents}. Assim que houver resultado, "
                "eu consolido para você."
            ),
        )
        out = []
        for c in waiting:
            spec = await self._spec(c.agent)
            extensions = [CONTRACT_EXTENSION_URI] if c.kind == "completo" else None
            message = new_message(
                [text_part(text)],
                task_id=c.remote_task_id,
                context_id=c.remote_context_id,
                metadata={CONTRACT_EXTENSION_URI: contract_terms(c, c.deadline_at)},
                extensions=extensions,
            )
            # volta para "ativo" antes da resposta: se o agente pedir outra coisa, a
            # mudança para aguardando_entrada é vista e a execução volta a needs_input
            await self._transition(
                c.id,
                states.ACTIVE,
                "router",
                detail={"motivo": "entrada do usuário enviada", "texto": text[:2000]},
                input_sent=True,
            )
            parent = traceparent(run_id, c.span_id)
            try:
                result = await self.client.send_message(
                    c.rpc_url or "",
                    message,
                    configuration={
                        "acceptedOutputModes": ACCEPTED_OUTPUT_MODES,
                        "returnImmediately": True,
                    },
                    metadata={
                        "traceparent": parent,
                        "switchboard": {"run_id": run_id, "contract_id": c.id},
                    },
                    token=self._token(spec),
                    extensions=extensions,
                    headers={"traceparent": parent},
                    timeout_s=spec.timeout_s if spec else None,
                )
                await self.apply_event(c.id, parse_stream_event(result), source="response")
            except (AgentError, ValueError) as exc:
                await self._transition(
                    c.id, states.FAILED, "router", error=f"falha ao enviar a entrada: {exc}"
                )
            out.append(await self._reload(c.id, c))
        return out

    # -------------------------------------------------------------- consolidação

    async def _check_run(self, run_id: str) -> None:
        contracts = await self.store.run_contracts(run_id)
        if not contracts:
            return
        open_ = [c for c in contracts if not c.terminal]
        if open_:
            if all(c.state == states.INPUT_REQUIRED for c in open_):
                question = input_question(open_)
                if await self.store.update_run(
                    run_id,
                    expect={RUN_PENDING, RUN_NEEDS_INPUT},
                    status=RUN_NEEDS_INPUT,
                    answer=question,
                ):
                    self._signal(run_id)
                    self._spawn(self._notify_callback(run_id))
            return
        if not await self.store.update_run(
            run_id, expect={RUN_PENDING, RUN_NEEDS_INPUT}, status=RUN_CONSOLIDATING
        ):
            return  # outro evento (ou processo) já está consolidando
        run = await self.store.get_run(run_id)
        if run is None:
            return
        try:
            if self.consolidator is not None:
                result = await self.consolidator(run, contracts)
            else:
                result = Consolidation(default_consolidation(run, contracts))
            await self.store.update_run(
                run_id, status=RUN_COMPLETED, answer=result.answer, finished_at=self._clock()
            )
            if result.spans:
                await self.store.add_spans(result.spans)
        except Exception as exc:
            log.exception("falha ao consolidar a execução %s", run_id)
            await self.store.update_run(
                run_id,
                status=RUN_FAILED,
                answer=default_consolidation(run, contracts),
                error=f"falha na consolidação: {exc}"[:500],
                finished_at=self._clock(),
            )
        self._signal(run_id)
        self._spawn(self._notify_callback(run_id))

    async def _notify_callback(self, run_id: str) -> None:
        run = await self.store.get_run(run_id)
        if run is None or not run.callback_url:
            return
        contracts = await self.store.run_contracts(run_id)
        payload = {**run.to_dict(), "contracts": [c.summary() for c in contracts]}
        http = self._callback_http or self.client.http
        for attempt in range(2):
            try:
                resp = await http.post(run.callback_url, json=payload, timeout=5.0)
                if resp.status_code < 500:
                    return
            except httpx.HTTPError as exc:
                log.warning("callback da execução %s falhou (%s): %s", run_id, attempt + 1, exc)
            await asyncio.sleep(1.0)

    async def recover(self) -> None:
        """Depois de um restart: consolida execuções cujos contratos já terminaram."""
        runs = getattr(self.store, "open_runs", None)
        if runs is None:
            return
        now = self._clock()
        for run in await runs():
            if run.status == RUN_CONSOLIDATING:
                if (now - run.updated_at).total_seconds() < 120:
                    continue
                await self.store.update_run(run.id, expect={RUN_CONSOLIDATING}, status=RUN_PENDING)
            await self._check_run(run.id)

    # ------------------------------------------------------------------ supervisor

    async def tick(self) -> int:
        """Uma rodada do supervisor: polling e prazos do que está vencido."""
        now = self._clock()
        due = await self.store.due_contracts(now)
        started = 0
        for contract in due:
            if contract.id in self._inflight:
                continue
            if not await self.store.claim(contract.id, now, now + timedelta(seconds=self.lease_s)):
                continue
            self._inflight.add(contract.id)
            started += 1
            self._spawn(self._process(contract))
        return started

    async def _process(self, contract: ContractRecord) -> None:
        try:
            async with self._semaphore:
                await self.check(contract)
        except Exception:
            log.exception("falha ao acompanhar o contrato %s", contract.id)
        finally:
            self._inflight.discard(contract.id)
            with contextlib.suppress(Exception):
                await self.store.release(contract.id)

    async def drain(self) -> None:
        """Espera as tarefas em segundo plano (testes)."""
        while self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)

    async def _loop(self) -> None:
        with contextlib.suppress(Exception):
            await self.recover()
        while True:
            try:
                await self.tick()
            except Exception:
                log.exception("supervisor de contratos: falha numa rodada")
            await asyncio.sleep(self.tick_s)

    def start(self) -> None:
        if self._supervisor is None or self._supervisor.done():
            self._supervisor = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        if self._supervisor is not None:
            self._supervisor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._supervisor
            self._supervisor = None
        for task in list(self._background):
            task.cancel()
        await asyncio.gather(*list(self._background), return_exceptions=True)

    async def aclose(self) -> None:
        await self.stop()
        await self.client.aclose()
