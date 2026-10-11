"""Persistência SQL de execuções assíncronas, contratos, eventos e spans.

Implementa :class:`switchboard.contracts.store.ContractStore` sobre o mesmo
banco do console. As operações que decidem corrida são condicionais no
próprio SQL — ``UPDATE ... WHERE version = ?`` (contratos),
``UPDATE ... WHERE status IN (...)`` (execuções) e ``lease_until`` (quem faz o
polling) — então vários processos do router podem receber push, fazer
polling e consolidar sem duplicar trabalho.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any, TypeVar

import anyio.to_thread
from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from ..contracts import states
from ..contracts.models import ContractEvent, ContractRecord, RunRecord
from ..contracts.store import OPEN_RUN_STATUSES
from ..tracing import Span, utcnow
from .db import Database
from .orm import Contract, ContractEventRow, SpanRow, Trace

T = TypeVar("T")

_CONTRACT_FIELDS = (
    "state",
    "rpc_url",
    "remote_task_id",
    "remote_context_id",
    "input",
    "input_text",
    "output",
    "output_text",
    "artifacts",
    "input_schema",
    "output_schema",
    "input_hash",
    "output_hash",
    "deadline_at",
    "reply_mode",
    "push_token_hash",
    "accepted_at",
    "finished_at",
    "next_check_at",
    "input_sent_at",
    "question",
    "answered_questions",
    "checks",
    "last_message",
    "error",
    "span_id",
    "parent_span_id",
)

_RUN_FIELDS = {"status", "answer", "error", "finished_at", "callback_url", "context"}


def aware(value: datetime | None) -> datetime | None:
    """SQLite devolve datetimes sem fuso: tudo aqui é UTC."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def row_to_contract(row: Contract) -> ContractRecord:
    return ContractRecord(
        id=row.id,
        run_id=row.run_id,
        profile=row.profile,
        agent=row.agent,
        skill=row.skill,
        kind=row.kind,
        state=row.state,
        rpc_url=row.rpc_url,
        remote_task_id=row.remote_task_id,
        remote_context_id=row.remote_context_id,
        input=dict(row.input or {}),
        input_text=row.input_text or "",
        output=row.output,
        output_text=row.output_text or "",
        artifacts=list(row.artifacts or []),
        input_schema=row.input_schema,
        output_schema=row.output_schema,
        input_hash=row.input_hash,
        output_hash=row.output_hash,
        deadline_at=aware(row.deadline_at),
        reply_mode=row.reply_mode or "poll",
        push_token_hash=row.push_token_hash,
        created_at=aware(row.created_at) or utcnow(),
        updated_at=aware(row.updated_at) or utcnow(),
        accepted_at=aware(row.accepted_at),
        finished_at=aware(row.finished_at),
        next_check_at=aware(row.next_check_at),
        input_sent_at=aware(row.input_sent_at),
        question=row.question,
        answered_questions=list(row.answered_questions or []),
        checks=row.checks or 0,
        last_message=row.last_message or "",
        error=row.error,
        span_id=row.span_id or "",
        parent_span_id=row.parent_span_id,
        version=row.version or 0,
    )


def row_to_event(row: ContractEventRow) -> ContractEvent:
    return ContractEvent(
        contract_id=row.contract_id,
        kind=row.kind,
        source=row.source,
        state=row.state,
        detail=dict(row.detail or {}),
        at=aware(row.at) or utcnow(),
        id=row.id,
    )


def row_to_run(row: Trace) -> RunRecord:
    return RunRecord(
        id=row.id,
        profile=row.profile,
        question=row.question,
        status=row.status or "completed",
        answer=row.answer or "",
        context=list(row.context or []),
        callback_url=row.callback_url,
        error=row.error,
        created_at=aware(row.created_at) or utcnow(),
        updated_at=aware(row.updated_at) or aware(row.created_at) or utcnow(),
        finished_at=aware(row.finished_at),
    )


def row_to_span(row: SpanRow) -> Span:
    return Span(
        id=row.id,
        trace_id=row.trace_id,
        parent_id=row.parent_id,
        kind=row.kind,
        name=row.name,
        started_at=aware(row.started_at) or utcnow(),
        ended_at=aware(row.ended_at),
        status=row.status or "ok",
        attributes=dict(row.attributes or {}),
    )


def span_rows(spans: Iterable[Span]) -> list[SpanRow]:
    return [
        SpanRow(
            id=s.id,
            trace_id=s.trace_id,
            parent_id=s.parent_id,
            kind=s.kind,
            name=s.name[:200],
            started_at=s.started_at,
            ended_at=s.ended_at,
            status=s.status,
            attributes=s.attributes,
        )
        for s in spans
    ]


def _event_rows(events: Iterable[ContractEvent]) -> list[ContractEventRow]:
    return [
        ContractEventRow(
            contract_id=e.contract_id,
            at=e.at,
            kind=e.kind,
            source=e.source,
            state=e.state,
            detail=e.detail,
        )
        for e in events
    ]


class SqlContractStore:
    def __init__(self, db: Database):
        self.db = db

    async def _run(self, fn: Callable[[Session], T]) -> T:
        def call() -> T:
            with self.db.session() as session:
                return fn(session)

        return await anyio.to_thread.run_sync(call)

    # -- execuções --------------------------------------------------------------

    async def create_run(self, run: RunRecord) -> None:
        extra = run.extra or {}

        def save(s: Session) -> None:
            s.add(
                Trace(
                    id=run.id,
                    created_at=run.created_at,
                    updated_at=run.updated_at,
                    profile=run.profile,
                    question=run.question,
                    answer=run.answer,
                    route=str(extra.get("route") or "delegated"),
                    model=str(extra.get("model") or ""),
                    agent=extra.get("agent"),
                    tool=extra.get("tool"),
                    arguments=extra.get("arguments"),
                    reason=str(extra.get("reason") or ""),
                    decided_by=extra.get("decided_by"),
                    confidence=extra.get("confidence"),
                    tasks=extra.get("tasks"),
                    sources=[],
                    steps=[],
                    warnings=[],
                    status=run.status,
                    context=run.context,
                    callback_url=run.callback_url,
                )
            )

        await self._run(save)

    async def get_run(self, run_id: str) -> RunRecord | None:
        return await self._run(lambda s: (row := s.get(Trace, run_id)) and row_to_run(row))

    async def update_run(
        self, run_id: str, *, expect: Iterable[str] | None = None, **fields: Any
    ) -> bool:
        values = {k: v for k, v in fields.items() if k in _RUN_FIELDS}
        values["updated_at"] = utcnow()
        expected = list(expect) if expect is not None else None

        def apply(s: Session) -> bool:
            stmt = update(Trace).where(Trace.id == run_id)
            if expected is not None:
                stmt = stmt.where(Trace.status.in_(expected))
            return s.execute(stmt.values(**values)).rowcount == 1

        return await self._run(apply)

    async def open_runs(self) -> list[RunRecord]:
        return await self._run(
            lambda s: [
                row_to_run(r)
                for r in s.scalars(select(Trace).where(Trace.status.in_(list(OPEN_RUN_STATUSES))))
            ]
        )

    # -- contratos --------------------------------------------------------------

    async def add_contract(self, contract: ContractRecord, events: list[ContractEvent]) -> None:
        def save(s: Session) -> None:
            row = Contract(
                id=contract.id,
                run_id=contract.run_id,
                profile=contract.profile,
                agent=contract.agent,
                skill=contract.skill,
                kind=contract.kind,
                created_at=contract.created_at,
                updated_at=contract.updated_at,
                version=contract.version,
            )
            for name in _CONTRACT_FIELDS:
                setattr(row, name, getattr(contract, name))
            s.add(row)
            s.flush()
            s.add_all(_event_rows(events))

        await self._run(save)

    async def get_contract(self, contract_id: str) -> ContractRecord | None:
        return await self._run(
            lambda s: (row := s.get(Contract, contract_id)) and row_to_contract(row)
        )

    async def save_contract(self, contract: ContractRecord, events: list[ContractEvent]) -> bool:
        values = {name: getattr(contract, name) for name in _CONTRACT_FIELDS}
        now = utcnow()

        def apply(s: Session) -> bool:
            result = s.execute(
                update(Contract)
                .where(Contract.id == contract.id, Contract.version == contract.version)
                .values(**values, updated_at=now, version=contract.version + 1)
            )
            if result.rowcount != 1:
                return False
            s.add_all(_event_rows(events))
            return True

        ok = await self._run(apply)
        if ok:
            contract.version += 1
            contract.updated_at = now
        return ok

    async def run_contracts(self, run_id: str) -> list[ContractRecord]:
        return await self._run(
            lambda s: [
                row_to_contract(r)
                for r in s.scalars(
                    select(Contract).where(Contract.run_id == run_id).order_by(Contract.created_at)
                )
            ]
        )

    async def contract_events(self, contract_id: str) -> list[ContractEvent]:
        return await self._run(
            lambda s: [
                row_to_event(r)
                for r in s.scalars(
                    select(ContractEventRow)
                    .where(ContractEventRow.contract_id == contract_id)
                    .order_by(ContractEventRow.id)
                )
            ]
        )

    async def due_contracts(self, now: datetime, limit: int = 50) -> list[ContractRecord]:
        def load(s: Session) -> list[ContractRecord]:
            query = (
                select(Contract)
                .where(
                    Contract.state.in_(list(states.OPEN)),
                    or_(Contract.next_check_at <= now, Contract.deadline_at <= now),
                    or_(Contract.lease_until.is_(None), Contract.lease_until < now),
                )
                .order_by(Contract.next_check_at)
                .limit(limit)
            )
            return [row_to_contract(r) for r in s.scalars(query)]

        return await self._run(load)

    async def claim(self, contract_id: str, now: datetime, until: datetime) -> bool:
        def apply(s: Session) -> bool:
            return (
                s.execute(
                    update(Contract)
                    .where(
                        Contract.id == contract_id,
                        or_(Contract.lease_until.is_(None), Contract.lease_until < now),
                    )
                    .values(lease_until=until)
                ).rowcount
                == 1
            )

        return await self._run(apply)

    async def release(self, contract_id: str) -> None:
        await self._run(
            lambda s: s.execute(
                update(Contract).where(Contract.id == contract_id).values(lease_until=None)
            )
        )

    # -- spans --------------------------------------------------------------------

    async def add_spans(self, spans: list[Span]) -> None:
        if spans:
            await self._run(lambda s: s.add_all(span_rows(spans)))

    async def run_spans(self, run_id: str) -> list[Span]:
        return await self._run(
            lambda s: [
                row_to_span(r)
                for r in s.scalars(
                    select(SpanRow)
                    .where(SpanRow.trace_id == run_id)
                    .order_by(SpanRow.started_at, SpanRow.pk)
                )
            ]
        )
