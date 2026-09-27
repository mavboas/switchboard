"""Utilitários para testar roteadores sem rede nem LLM de verdade.

* :class:`ScriptedChat` — um "LLM" que devolve respostas roteirizadas;
* :func:`inproc_connector` — conecta o catálogo MCP a servidores ``MCPServer``
  no mesmo processo (sem HTTP).

Exemplo::

    catalog = AgentCatalog(connector=inproc_connector({"calc": meu_servidor}))
    chat = ScriptedChat(['{"action": "answer", "answer": "ok"}'])
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from .config import AgentSpec
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


def inproc_connector(servers: dict[str, Any]):
    """Connector do :class:`~switchboard.agents.AgentCatalog` para servidores em processo."""
    from mcp import Client

    @asynccontextmanager
    async def connect(spec: AgentSpec, _token: str | None) -> AsyncIterator[Any]:
        if spec.name not in servers:
            raise ConnectionError(f"agente {spec.name} fora do ar")
        async with Client(servers[spec.name]) as client:
            yield client

    return connect
