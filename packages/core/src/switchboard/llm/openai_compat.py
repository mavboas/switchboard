"""Provedor compatível com a API de Chat Completions da OpenAI.

Serve para OpenAI, Azure OpenAI (API v1), Google Gemini (endpoint
compatível), Ollama, OpenRouter, Groq, vLLM, LM Studio e qualquer outro
servidor que fale ``POST {base_url}/chat/completions``.

Modelos diferentes aceitam parâmetros diferentes (alguns rejeitam
``temperature``, outros exigem ``max_completion_tokens`` no lugar de
``max_tokens``, nem todo servidor tem ``response_format``). Em vez de exigir
que você saiba disso, o adaptador ajusta a chamada quando o servidor recusa um
parâmetro com HTTP 400 e lembra do ajuste nas próximas chamadas.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx

from ..errors import LLMError
from .base import ChatResult, Message, Usage, error_text

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def build_headers(
    api_key: str | None, api_key_header: str, extra_headers: dict[str, str] | None
) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        if api_key_header.lower() == "authorization":
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers[api_key_header] = api_key
    headers.update(extra_headers or {})
    return headers


class OpenAICompatibleChat:
    offline = False

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_header: str = "Authorization",
        extra_headers: dict[str, str] | None = None,
        temperature: float | None = 0.2,
        max_tokens: int | None = 4096,
        timeout_s: float = 60.0,
        json_mode: bool = True,
        label: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.label = label or f"openai:{model}"
        self._max_tokens_key = "max_tokens"
        self._dropped: set[str] = set()
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=build_headers(api_key, api_key_header, extra_headers),
            timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 10.0)),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(
        self, messages: Sequence[Message], json_mode: bool, max_tokens: int | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if self.temperature is not None and "temperature" not in self._dropped:
            payload["temperature"] = self.temperature
        limit = max_tokens or self.max_tokens
        if limit and "max_tokens" not in self._dropped:
            payload[self._max_tokens_key] = limit
        if json_mode and self.json_mode and "response_format" not in self._dropped:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _adapt(self, payload: dict[str, Any], message: str) -> bool:
        """Ajusta o payload depois de um HTTP 400; devolve True se mudou algo."""
        msg = message.lower()
        if "response_format" in payload and ("response_format" in msg or "json" in msg):
            payload.pop("response_format")
            self._dropped.add("response_format")
            return True
        if "max_tokens" in payload and "max_completion_tokens" in msg:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
            self._max_tokens_key = "max_completion_tokens"
            return True
        if "temperature" in payload and "temperature" in msg:
            payload.pop("temperature")
            self._dropped.add("temperature")
            return True
        key = self._max_tokens_key
        context_error = any(
            t in msg for t in ("context length", "context window", "maximum context")
        )
        if key in payload and (key in msg or context_error):
            # limite maior que o contexto do servidor (ex.: vLLM com max-model-len pequeno)
            payload.pop(key)
            self._dropped.add("max_tokens")
            return True
        return False

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        payload = self._payload(messages, json_mode, max_tokens)
        started = time.perf_counter()
        for _attempt in range(5):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
            except (httpx.HTTPError, RuntimeError) as exc:  # RuntimeError: cliente já fechado
                raise LLMError(
                    f"{self.label}: falha de rede ao chamar {self.base_url} ({exc.__class__.__name__}: {exc})",
                    provider=self.label,
                ) from exc
            if resp.status_code == 400 and self._adapt(payload, error_text(resp)):
                continue
            break
        if resp.status_code >= 400:
            raise LLMError(
                f"{self.label}: HTTP {resp.status_code} - {error_text(resp)}",
                provider=self.label,
                status=resp.status_code,
            )
        try:
            data = resp.json()
            message = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"{self.label}: resposta inesperada do provedor: {resp.text[:300]}"
            ) from exc
        content = message.get("content")
        if isinstance(content, list):  # alguns servidores devolvem partes
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        finish = data["choices"][0].get("finish_reason")
        if not (content or "").strip() and finish == "length":
            raise LLMError(
                f"{self.label}: a resposta veio vazia porque bateu no limite de tokens "
                "(modelos com raciocínio gastam tokens antes de responder; aumente max_tokens)",
                provider=self.label,
            )
        usage = data.get("usage") or {}
        return ChatResult(
            text=content or "",
            model=str(data.get("model") or self.model),
            usage=Usage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
