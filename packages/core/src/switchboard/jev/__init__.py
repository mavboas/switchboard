"""Modelo de decisão (System One): cliente do Jev, da TypeSafe."""

from collections.abc import Callable

import httpx

from ..config import ModelSpec
from ..errors import ConfigError
from ..secrets import resolve_env
from .client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    JevAnswer,
    JevClient,
    JevResult,
    choice,
    endpoint_url,
    noul,
    score,
)


def build_decision_model(
    spec: ModelSpec,
    *,
    resolve_secret: Callable[[str | None], str | None] = resolve_env,
    transport: httpx.AsyncBaseTransport | None = None,
) -> JevClient:
    """Cliente do modelo de decisão a partir de uma conexão ``provider: typesafe``."""
    if not spec.is_decision_model:
        raise ConfigError(
            f"'{spec.name}' é um modelo de chat ({spec.provider}); o modelo de decisão precisa "
            "de uma conexão typesafe (Jev)"
        )
    extra = {k: (resolve_secret(v) or "") for k, v in spec.extra_headers.items()}
    return JevClient(
        model=spec.model,
        base_url=spec.base_url,
        api_key=resolve_secret(spec.api_key),
        api_key_header=spec.api_key_header,
        extra_headers=extra,
        timeout_s=min(spec.timeout_s, 30.0),
        label=f"{spec.name}:{spec.model}",
        transport=transport,
    )


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "JevAnswer",
    "JevClient",
    "JevResult",
    "build_decision_model",
    "choice",
    "endpoint_url",
    "noul",
    "score",
]
