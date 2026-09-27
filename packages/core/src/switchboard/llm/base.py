"""Contrato comum dos provedores de LLM."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


@dataclass
class ChatResult:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0


@runtime_checkable
class ChatModel(Protocol):
    """Qualquer provedor de chat: OpenAI-compatível, Anthropic ou offline."""

    label: str
    offline: bool

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult: ...

    async def aclose(self) -> None: ...


def error_text(resp: httpx.Response, limit: int = 500) -> str:
    """Extrai a mensagem de erro de uma resposta HTTP de provedor."""
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:limit]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:limit]
        if isinstance(err, str):
            return err[:limit]
        if data.get("message"):
            return str(data["message"])[:limit]
    return str(data)[:limit]
