"""Modelo "offline": nenhum LLM é chamado.

Com ele o roteador usa o decisor heurístico (palavras-chave + extração de
argumentos pelo schema da tool) e respostas extrativas da base de
conhecimento. Serve para rodar a demo sem chave de API e para testes
determinísticos; em produção, configure um provedor de verdade.
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import ChatResult, Message


class OfflineChat:
    offline = True

    def __init__(self, label: str = "offline"):
        self.label = label

    async def aclose(self) -> None:
        return None

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        text = (
            "Modelo offline ativo (heurístico, sem LLM). "
            f"Recebi: {last[:200]!r}. Configure um provedor para respostas geradas."
        )
        return ChatResult(text=text, model="offline")
