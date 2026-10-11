"""Spans de uma execução (formato compatível com W3C Trace Context).

Cada execução do roteador é uma árvore de spans: ``pedido`` na raiz e, abaixo,
``rag``, ``descoberta``, ``decisao`` (com ``jev`` e ``llm``), ``tool_mcp``,
``consolidacao``… Os contratos com agentes (os *spawns*) também são spans,
derivados dos próprios registros de contrato, que ficam abertos até o estado
terminal.

O ``trace_id`` tem 32 dígitos hexadecimais e os ids de span 16, como no W3C:
o ``traceparent`` vai para os agentes A2A e liga o tracing dos dois lados.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_trace_id() -> str:
    return secrets.token_hex(16)


def new_span_id() -> str:
    return secrets.token_hex(8)


def traceparent(trace_id: str, span_id: str) -> str:
    return f"00-{trace_id}-{span_id}-01"


@dataclass
class Span:
    id: str
    trace_id: str
    parent_id: str | None
    kind: str
    name: str
    started_at: datetime
    ended_at: datetime | None = None
    status: str = "open"  # open | ok | warn | error
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at).total_seconds() * 1000

    def finish(self, status: str = "ok", at: datetime | None = None, **attributes: Any) -> Span:
        self.attributes.update(attributes)
        if self.ended_at is None:
            self.ended_at = at or utcnow()
        self.status = status
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "name": self.name,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_ms": round(self.duration_ms, 1),
            "status": self.status,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Span:
        def when(value: Any) -> datetime | None:
            if not value:
                return None
            return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))

        return cls(
            id=str(data["id"]),
            trace_id=str(data["trace_id"]),
            parent_id=data.get("parent_id"),
            kind=str(data.get("kind") or ""),
            name=str(data.get("name") or ""),
            started_at=when(data.get("started_at")) or utcnow(),
            ended_at=when(data.get("ended_at")),
            status=str(data.get("status") or "ok"),
            attributes=dict(data.get("attributes") or {}),
        )


class SpanRecorder:
    """Registra os spans de uma execução (a raiz é criada na construção)."""

    def __init__(
        self,
        trace_id: str,
        *,
        root_name: str = "pedido",
        clock: Callable[[], datetime] = utcnow,
        **root_attributes: Any,
    ):
        self.trace_id = trace_id
        self._clock = clock
        self.spans: list[Span] = []
        self.root = self.start("pedido", root_name, parent=None, **root_attributes)

    def start(
        self, kind: str, name: str, *, parent: Span | None | str = "root", **attributes: Any
    ) -> Span:
        if parent == "root":
            parent_id = self.root.id if self.spans else None
        elif isinstance(parent, Span):
            parent_id = parent.id
        else:
            parent_id = parent
        span = Span(
            id=new_span_id(),
            trace_id=self.trace_id,
            parent_id=parent_id,
            kind=kind,
            name=name,
            started_at=self._clock(),
            attributes=dict(attributes),
        )
        self.spans.append(span)
        return span

    @contextmanager
    def span(
        self, kind: str, name: str, *, parent: Span | None | str = "root", **attributes: Any
    ) -> Iterator[Span]:
        span = self.start(kind, name, parent=parent, **attributes)
        try:
            yield span
        except BaseException as exc:
            span.finish("error", at=self._clock(), erro=f"{exc.__class__.__name__}: {exc}"[:300])
            raise
        else:
            if span.ended_at is None:
                span.finish(span.status if span.status != "open" else "ok", at=self._clock())

    def close(self, status: str = "ok", **attributes: Any) -> None:
        self.root.finish(status, at=self._clock(), **attributes)

    def traceparent(self, span: Span | None = None) -> str:
        return traceparent(self.trace_id, (span or self.root).id)

    def to_list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.spans]
