"""Registros de execuções assíncronas, contratos e eventos."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..tracing import utcnow
from . import states

# status de uma execução (run)
RUN_PENDING = "pending"  # esperando contratos
RUN_NEEDS_INPUT = "needs_input"  # algum agente pediu informação ao usuário
RUN_CONSOLIDATING = "consolidating"
RUN_COMPLETED = "completed"
RUN_FAILED = "failed"
RUN_FINAL = frozenset({RUN_COMPLETED, RUN_FAILED})


def new_contract_id() -> str:
    return "ctr_" + secrets.token_hex(10)


def new_push_token() -> str:
    return secrets.token_urlsafe(24)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_matches(token: str | None, expected_hash: str | None) -> bool:
    if not token or not expected_hash:
        return False
    return hmac.compare_digest(token_hash(token), expected_hash)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


@dataclass
class ContractEvent:
    contract_id: str
    kind: str  # state | message | artifact | note | violation
    source: str  # router | response | push | poll
    state: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    at: datetime = field(default_factory=utcnow)
    id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "at": _iso(self.at),
            "kind": self.kind,
            "source": self.source,
            "state": self.state,
            "detail": self.detail,
        }


@dataclass
class ContractRecord:
    id: str
    run_id: str
    profile: str
    agent: str
    skill: str
    kind: str  # completo | basico
    state: str = states.PROPOSED
    rpc_url: str | None = None
    remote_task_id: str | None = None
    remote_context_id: str | None = None
    input: dict[str, Any] = field(default_factory=dict)
    input_text: str = ""
    output: Any = None
    output_text: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    deadline_at: datetime | None = None
    reply_mode: str = "poll"  # push | poll
    push_token_hash: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    accepted_at: datetime | None = None
    finished_at: datetime | None = None
    next_check_at: datetime | None = None
    input_sent_at: datetime | None = None
    checks: int = 0
    last_message: str = ""
    error: str | None = None
    span_id: str = ""
    parent_span_id: str | None = None
    version: int = 0

    @property
    def terminal(self) -> bool:
        return self.state in states.TERMINAL

    @property
    def duration_ms(self) -> float | None:
        end = self.finished_at
        return (end - self.created_at).total_seconds() * 1000 if end else None

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent": self.agent,
            "skill": self.skill,
            "kind": self.kind,
            "state": self.state,
            "deadline_at": _iso(self.deadline_at),
            "last_message": self.last_message,
            "error": self.error,
        }

    def to_dict(self, events: list[ContractEvent] | None = None) -> dict[str, Any]:
        data = {
            **self.summary(),
            "run_id": self.run_id,
            "profile": self.profile,
            "rpc_url": self.rpc_url,
            "remote_task_id": self.remote_task_id,
            "remote_context_id": self.remote_context_id,
            "input": self.input,
            "input_text": self.input_text,
            "output": self.output,
            "output_text": self.output_text,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "reply_mode": self.reply_mode,
            "created_at": _iso(self.created_at),
            "accepted_at": _iso(self.accepted_at),
            "finished_at": _iso(self.finished_at),
            "duration_ms": round(self.duration_ms, 1) if self.duration_ms is not None else None,
            "checks": self.checks,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
        }
        if events is not None:
            data["events"] = [e.to_dict() for e in events]
        return data


@dataclass
class RunRecord:
    """A parte assíncrona de uma execução: o que é preciso para consolidar depois."""

    id: str
    profile: str
    question: str
    status: str = RUN_PENDING
    answer: str = ""
    context: list[dict[str, str]] = field(default_factory=list)
    callback_url: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    finished_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # colunas do trace na criação

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.id,
            "profile": self.profile,
            "question": self.question,
            "status": self.status,
            "answer": self.answer,
            "error": self.error,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "finished_at": _iso(self.finished_at),
        }
