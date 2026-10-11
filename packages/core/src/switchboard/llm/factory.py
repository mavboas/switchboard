"""Monta o provedor certo a partir de um :class:`~switchboard.config.ModelSpec`."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from ..config import ModelSpec
from ..errors import ConfigError
from ..secrets import resolve_env
from .anthropic import AnthropicChat
from .base import ChatModel
from .offline import OfflineChat
from .openai_compat import OpenAICompatibleChat

SecretResolver = Callable[[str | None], str | None]


def build_chat_model(
    spec: ModelSpec,
    *,
    resolve_secret: SecretResolver = resolve_env,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ChatModel:
    if spec.is_decision_model:
        raise ConfigError(
            f"'{spec.name}' é um modelo de decisão ({spec.provider}) e não conversa; "
            "use-o como modelo de decisão do roteador e escolha um LLM como modelo principal"
        )
    api_key = resolve_secret(spec.api_key)
    extra = {k: (resolve_secret(v) or "") for k, v in spec.extra_headers.items()}
    if spec.provider == "offline":
        return OfflineChat(label=f"offline:{spec.name}")
    if spec.provider == "anthropic":
        return AnthropicChat(
            model=spec.model,
            base_url=spec.base_url,
            api_key=api_key,
            extra_headers=extra,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            timeout_s=spec.timeout_s,
            label=f"anthropic:{spec.model}",
            transport=transport,
        )
    return OpenAICompatibleChat(
        model=spec.model,
        base_url=spec.base_url,
        api_key=api_key,
        api_key_header=spec.api_key_header,
        extra_headers=extra,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        timeout_s=spec.timeout_s,
        json_mode=spec.json_mode,
        label=f"{spec.name}:{spec.model}",
        transport=transport,
    )
