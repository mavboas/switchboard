"""Tipos do motor de roteamento: decisão, trace e resultado."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..config import AgentSpec, ModelSpec, ProfileSpec
from ..llm.base import Usage

Action = Literal["answer", "delegate", "clarify"]
Route = Literal["direct", "delegated", "clarify", "error"]


@dataclass
class Decision:
    """O que o roteador decidiu fazer com o pedido."""

    action: Action
    reason: str = ""
    answer: str | None = None
    sources: list[int] = field(default_factory=list)
    agent: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    question: str | None = None


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
    agent: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    reason: str = ""
    sources: list[SourceRef] = field(default_factory=list)
    steps: list[TraceStep] = field(default_factory=list)
    latency_ms: float = 0.0
    usage: Usage = field(default_factory=Usage)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "profile": self.profile,
            "question": self.question,
            "answer": self.answer,
            "route": self.route,
            "model": self.model,
            "agent": self.agent,
            "tool": self.tool,
            "arguments": self.arguments,
            "reason": self.reason,
            "sources": [s.to_dict() for s in self.sources],
            "steps": [s.to_dict() for s in self.steps],
            "latency_ms": round(self.latency_ms, 1),
            "usage": self.usage.to_dict(),
            "warnings": self.warnings,
            "error": self.error,
        }


@dataclass
class ResolvedProfile:
    """Perfil pronto para uso: a spec mais o modelo e os agentes já resolvidos."""

    spec: ProfileSpec
    model: ModelSpec
    agents: list[AgentSpec]

    @property
    def name(self) -> str:
        return self.spec.name

    def agent(self, name: str) -> AgentSpec | None:
        return next((a for a in self.agents if a.name == name), None)
