"""Tipos do motor de roteamento: decisão, trace e resultado."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..config import AgentSpec, ConnectorSpec, ModelSpec, ProfileSpec
from ..llm.base import Usage

Action = Literal["answer", "tool", "delegate", "clarify"]
Route = Literal["direct", "tool", "delegated", "clarify", "error"]
RunStatus = Literal["completed", "pending", "needs_input", "failed"]
DecidedBy = Literal["jev", "llm", "heuristic", "rule"]


@dataclass
class TaskRequest:
    """Uma unidade de trabalho escolhida pelo decisor.

    ``kind="tool"``: ``owner`` é o conector MCP e ``name`` a tool;
    ``kind="skill"``: ``owner`` é o agente A2A e ``name`` a skill.
    """

    kind: Literal["tool", "skill"]
    owner: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    instruction: str = ""

    @property
    def key(self) -> str:
        return f"{self.owner}/{self.name}"

    def to_dict(self) -> dict[str, Any]:
        out = {
            "kind": self.kind,
            "owner": self.owner,
            "name": self.name,
            "arguments": self.arguments,
        }
        if self.instruction:
            out["instruction"] = self.instruction
        return out


@dataclass
class Decision:
    """O que o roteador decidiu fazer com o pedido."""

    action: Action
    reason: str = ""
    answer: str | None = None
    sources: list[int] = field(default_factory=list)
    question: str | None = None
    tasks: list[TaskRequest] = field(default_factory=list)
    decided_by: DecidedBy = "llm"
    confidence: float | None = None

    # atalhos para a primeira tarefa (compatibilidade com o formato de uma tarefa só)
    @property
    def task(self) -> TaskRequest | None:
        return self.tasks[0] if self.tasks else None

    @property
    def owner(self) -> str | None:
        return self.task.owner if self.task else None

    @property
    def target(self) -> str | None:
        return self.task.name if self.task else None

    @property
    def arguments(self) -> dict[str, Any]:
        return self.task.arguments if self.task else {}


@dataclass
class SourceRef:
    index: int
    kb: str
    document: str
    chunk_id: str
    score: float
    snippet: str
    section: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kb": self.kb,
            "document": self.document,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "snippet": self.snippet,
        }


@dataclass
class TraceStep:
    """Visão plana (compatível com o MVP) de um span."""

    name: str
    duration_ms: float
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "duration_ms": round(self.duration_ms, 1), "detail": self.detail}


@dataclass
class RouterResult:
    trace_id: str
    profile: str
    question: str
    answer: str
    route: Route
    model: str
    status: RunStatus = "completed"
    decided_by: str | None = None
    confidence: float | None = None
    agent: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    tasks: list[dict[str, Any]] = field(default_factory=list)
    contracts: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    sources: list[SourceRef] = field(default_factory=list)
    spans: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    usage: Usage = field(default_factory=Usage)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def run_id(self) -> str:
        return self.trace_id

    @property
    def steps(self) -> list[TraceStep]:
        """Spans como passos planos (sem a raiz), na ordem em que começaram."""
        return [
            TraceStep(s["name"], s.get("duration_ms") or 0.0, s.get("attributes") or {})
            for s in self.spans
            if s.get("parent_id")
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "run_id": self.trace_id,
            "status": self.status,
            "profile": self.profile,
            "question": self.question,
            "answer": self.answer,
            "route": self.route,
            "model": self.model,
            "decided_by": self.decided_by,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "agent": self.agent,
            "tool": self.tool,
            "arguments": self.arguments,
            "tasks": self.tasks,
            "contracts": self.contracts,
            "reason": self.reason,
            "sources": [s.to_dict() for s in self.sources],
            "steps": [s.to_dict() for s in self.steps],
            "spans": self.spans,
            "latency_ms": round(self.latency_ms, 1),
            "usage": self.usage.to_dict(),
            "warnings": self.warnings,
            "error": self.error,
        }


@dataclass
class ResolvedProfile:
    """Perfil pronto para uso: a spec mais modelos, conectores e agentes resolvidos."""

    spec: ProfileSpec
    model: ModelSpec
    connectors: list[ConnectorSpec] = field(default_factory=list)
    agents: list[AgentSpec] = field(default_factory=list)
    decision_model: ModelSpec | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    def connector(self, name: str) -> ConnectorSpec | None:
        return next((c for c in self.connectors if c.name == name), None)

    def agent(self, name: str) -> AgentSpec | None:
        return next((a for a in self.agents if a.name == name), None)
