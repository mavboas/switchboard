"""Provedor nativo da Anthropic (Messages API)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx

from ..errors import LLMError
from .base import ChatResult, Message, Usage, error_text

DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"


def _normalize_base(url: str | None) -> str:
    url = (url or DEFAULT_BASE_URL).rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def to_anthropic_messages(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Separa o system prompt e garante a alternância user/assistant exigida pela API."""
    system_parts = [m.content for m in messages if m.role == "system" and m.content.strip()]
    convo: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue
        role = "assistant" if m.role == "assistant" else "user"
        if convo and convo[-1]["role"] == role:
            convo[-1]["content"] += "\n\n" + m.content
        else:
            convo.append({"role": role, "content": m.content})
    if not convo or convo[0]["role"] != "user":
        convo.insert(0, {"role": "user", "content": "(início da conversa)"})
    return "\n\n".join(system_parts), convo


class AnthropicChat:
    offline = False

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        temperature: float | None = 0.2,
        max_tokens: int | None = 4096,
        timeout_s: float = 60.0,
        label: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model = model
        self.base_url = _normalize_base(base_url)
        self.temperature = temperature
        self.max_tokens = max_tokens or 4096
        self.label = label or f"anthropic:{model}"
        self._dropped: set[str] = set()
        headers = {
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        if api_key:
            headers["x-api-key"] = api_key
        headers.update(extra_headers or {})
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 10.0)),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        system, convo = to_anthropic_messages(messages)
        if json_mode:
            system = (system + "\n\n" if system else "") + (
                "Responda somente com um único objeto JSON válido, sem texto antes ou depois."
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": convo,
        }
        if system:
            payload["system"] = system
        if self.temperature is not None and "temperature" not in self._dropped:
            payload["temperature"] = self.temperature

        started = time.perf_counter()
        for _attempt in range(3):
            try:
                resp = await self._client.post("/v1/messages", json=payload)
            except (httpx.HTTPError, RuntimeError) as exc:  # RuntimeError: cliente já fechado
                raise LLMError(
                    f"{self.label}: falha de rede ao chamar {self.base_url} ({exc.__class__.__name__}: {exc})",
                    provider=self.label,
                ) from exc
            if (
                resp.status_code == 400
                and "temperature" in payload
                and "temperature" in error_text(resp).lower()
            ):
                payload.pop("temperature")
                self._dropped.add("temperature")
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
            blocks = data["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError(
                f"{self.label}: resposta inesperada do provedor: {resp.text[:300]}"
            ) from exc
        text = "".join(
            b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
        )
        if not text.strip() and data.get("stop_reason") == "max_tokens":
            raise LLMError(
                f"{self.label}: a resposta veio vazia porque bateu no limite de tokens (aumente max_tokens)",
                provider=self.label,
            )
        usage = data.get("usage") or {}
        return ChatResult(
            text=text,
            model=str(data.get("model") or self.model),
            usage=Usage(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
