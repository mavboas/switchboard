"""Constantes e formatos do A2A v1.0 (binding JSON-RPC, JSON no formato ProtoJSON).

Formatos conferidos contra o SDK oficial (``a2a-sdk`` 1.x):

* partes: ``{"text": "..."}`` ou ``{"data": {...}, "mediaType": "application/json"}``;
* papéis: ``ROLE_USER`` / ``ROLE_AGENT``; estados: ``TASK_STATE_*``;
* push notification: POST com um ``StreamResponse`` (``task``, ``message``,
  ``statusUpdate`` ou ``artifactUpdate``) e o cabeçalho
  ``X-A2A-Notification-Token``.

A leitura é tolerante com o formato 0.3 (estados em minúsculas, partes com
``kind``), mas as chamadas saem sempre no formato 1.0.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

A2A_VERSION = "1.0"
CARD_PATH = "/.well-known/agent-card.json"
VERSION_HEADER = "A2A-Version"
EXTENSIONS_HEADER = "A2A-Extensions"
NOTIFICATION_TOKEN_HEADER = "X-A2A-Notification-Token"
JSONRPC_BINDING = "JSONRPC"

ROLE_USER = "ROLE_USER"
ROLE_AGENT = "ROLE_AGENT"


class TaskState:
    UNSPECIFIED = "TASK_STATE_UNSPECIFIED"
    SUBMITTED = "TASK_STATE_SUBMITTED"
    WORKING = "TASK_STATE_WORKING"
    COMPLETED = "TASK_STATE_COMPLETED"
    FAILED = "TASK_STATE_FAILED"
    CANCELED = "TASK_STATE_CANCELED"
    INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
    REJECTED = "TASK_STATE_REJECTED"
    AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"


TERMINAL_TASK_STATES = frozenset(
    {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED, TaskState.REJECTED}
)

_LEGACY_STATES = {
    "submitted": TaskState.SUBMITTED,
    "working": TaskState.WORKING,
    "completed": TaskState.COMPLETED,
    "failed": TaskState.FAILED,
    "canceled": TaskState.CANCELED,
    "cancelled": TaskState.CANCELED,
    "input-required": TaskState.INPUT_REQUIRED,
    "input_required": TaskState.INPUT_REQUIRED,
    "rejected": TaskState.REJECTED,
    "auth-required": TaskState.AUTH_REQUIRED,
    "auth_required": TaskState.AUTH_REQUIRED,
    "unknown": TaskState.UNSPECIFIED,
}


def normalize_state(value: Any) -> str:
    text = str(value or "").strip()
    if text.upper().startswith("TASK_STATE_"):
        return text.upper()
    return _LEGACY_STATES.get(text.lower(), TaskState.UNSPECIFIED)


# -- partes e mensagens ----------------------------------------------------------


def text_part(text: str) -> dict[str, Any]:
    return {"text": text}


def data_part(data: Any, media_type: str = "application/json") -> dict[str, Any]:
    return {"data": data, "mediaType": media_type}


def parts_text(parts: Iterable[Mapping[str, Any]] | None, sep: str = "\n") -> str:
    texts = [str(p["text"]) for p in parts or () if isinstance(p, Mapping) and p.get("text")]
    return sep.join(t.strip() for t in texts if t.strip())


def parts_data(parts: Iterable[Mapping[str, Any]] | None) -> list[Any]:
    return [p["data"] for p in parts or () if isinstance(p, Mapping) and "data" in p]


def new_message(
    parts: Sequence[Mapping[str, Any]],
    *,
    role: str = ROLE_USER,
    message_id: str | None = None,
    context_id: str | None = None,
    task_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    extensions: Sequence[str] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "messageId": message_id or uuid.uuid4().hex,
        "role": role,
        "parts": [dict(p) for p in parts],
    }
    if context_id:
        message["contextId"] = context_id
    if task_id:
        message["taskId"] = task_id
    if metadata:
        message["metadata"] = dict(metadata)
    if extensions:
        message["extensions"] = list(extensions)
    return message


# -- tarefas e eventos -------------------------------------------------------------


@dataclass
class TaskView:
    """Uma ``Task`` A2A lida de forma tolerante."""

    id: str
    context_id: str | None
    state: str
    status_text: str = ""
    timestamp: str | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def parse(cls, data: Mapping[str, Any]) -> TaskView:
        status = data.get("status") or {}
        message = status.get("message") if isinstance(status, Mapping) else None
        return cls(
            id=str(data.get("id") or ""),
            context_id=data.get("contextId") or data.get("context_id"),
            state=normalize_state(status.get("state") if isinstance(status, Mapping) else None),
            status_text=parts_text((message or {}).get("parts"))
            if isinstance(message, Mapping)
            else "",
            timestamp=status.get("timestamp") if isinstance(status, Mapping) else None,
            artifacts=[a for a in data.get("artifacts") or [] if isinstance(a, Mapping)],
        )


@dataclass
class StreamEvent:
    """Um ``StreamResponse`` (resposta, push notification ou item de stream)."""

    kind: str  # task | status | artifact | message
    task_id: str | None = None
    context_id: str | None = None
    state: str | None = None
    status_text: str = ""
    timestamp: str | None = None
    task: TaskView | None = None
    artifact: dict[str, Any] | None = None
    append: bool = False
    last_chunk: bool = False
    message: dict[str, Any] | None = None


def parse_stream_event(payload: Mapping[str, Any]) -> StreamEvent:
    """Interpreta um ``StreamResponse``/``SendMessageResponse`` (ou uma ``Task`` pura)."""
    if not isinstance(payload, Mapping):
        raise ValueError("evento A2A precisa ser um objeto JSON")
    if isinstance(payload.get("task"), Mapping):
        task = TaskView.parse(payload["task"])
        return StreamEvent(
            "task",
            task_id=task.id,
            context_id=task.context_id,
            state=task.state,
            status_text=task.status_text,
            timestamp=task.timestamp,
            task=task,
        )
    update = payload.get("statusUpdate") or payload.get("status_update")
    if isinstance(update, Mapping):
        status = update.get("status") or {}
        message = status.get("message") if isinstance(status, Mapping) else None
        return StreamEvent(
            "status",
            task_id=update.get("taskId"),
            context_id=update.get("contextId"),
            state=normalize_state(status.get("state") if isinstance(status, Mapping) else None),
            status_text=parts_text(message.get("parts")) if isinstance(message, Mapping) else "",
            timestamp=status.get("timestamp") if isinstance(status, Mapping) else None,
        )
    update = payload.get("artifactUpdate") or payload.get("artifact_update")
    if isinstance(update, Mapping):
        return StreamEvent(
            "artifact",
            task_id=update.get("taskId"),
            context_id=update.get("contextId"),
            artifact=dict(update.get("artifact") or {}),
            append=bool(update.get("append")),
            last_chunk=bool(update.get("lastChunk")),
        )
    if isinstance(payload.get("message"), Mapping):
        message = dict(payload["message"])
        return StreamEvent(
            "message",
            task_id=message.get("taskId"),
            context_id=message.get("contextId"),
            status_text=parts_text(message.get("parts")),
            message=message,
        )
    if "status" in payload and "id" in payload:  # uma Task sem o envelope
        return parse_stream_event({"task": payload})
    if payload.get("kind") in ("status-update", "artifact-update"):  # formato 0.3
        key = "statusUpdate" if payload["kind"] == "status-update" else "artifactUpdate"
        return parse_stream_event({key: payload})
    raise ValueError(
        "evento A2A desconhecido (esperava task, statusUpdate, artifactUpdate ou message)"
    )


def parse_timestamp(value: str | None):
    """Carimbo de tempo A2A (RFC 3339) como datetime com fuso; ``None`` se inválido."""
    from datetime import UTC, datetime

    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def merge_artifact(
    artifacts: list[dict[str, Any]], artifact: Mapping[str, Any], *, append: bool
) -> list[dict[str, Any]]:
    """Acumula artefatos recebidos aos pedaços (``append`` junta partes do mesmo id)."""
    out = [dict(a) for a in artifacts]
    artifact_id = artifact.get("artifactId")
    for i, existing in enumerate(out):
        if artifact_id and existing.get("artifactId") == artifact_id:
            if append:
                merged = dict(existing)
                merged["parts"] = list(existing.get("parts") or []) + list(
                    artifact.get("parts") or []
                )
                out[i] = merged
            else:
                out[i] = dict(artifact)
            return out
    out.append(dict(artifact))
    return out
