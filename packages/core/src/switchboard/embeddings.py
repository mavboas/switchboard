"""Embedders: transformam texto em vetores para a busca do RAG.

* :class:`HashingEmbedder` — offline e determinístico (feature hashing de
  palavras, prefixos e bigramas). Não entende sinônimos, mas funciona sem
  nenhuma dependência ou chave e é estável entre processos.
* :class:`OpenAICompatibleEmbedder` — qualquer endpoint ``/embeddings``
  compatível com OpenAI (OpenAI, Azure, Ollama, vLLM, Gemini…).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import httpx

from .config import EmbedderSpec, ModelSpec
from .errors import ConfigError, LLMError
from .llm.base import error_text
from .llm.openai_compat import DEFAULT_BASE_URL, build_headers
from .secrets import resolve_env
from .text import content_tokens, stem


@runtime_checkable
class Embedder(Protocol):
    label: str

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def aclose(self) -> None: ...


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        if dim < 16:
            raise ConfigError("dim do embedder hashing precisa ser >= 16")
        self.dim = dim
        self.label = f"hashing-{dim}"

    async def aclose(self) -> None:
        return None

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def _slot(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest[:4], "little") % self.dim, (1.0 if digest[4] & 1 else -1.0)

    def embed_one(self, text: str) -> list[float]:
        tokens = content_tokens(text)
        stems = [stem(t) for t in tokens]
        # o stem é a feature principal ("horario"/"horarios" casam por inteiro);
        # a palavra exata e os bigramas refinam o ranking sem dominar.
        weights: dict[str, float] = {}
        for tok, st in zip(tokens, stems, strict=True):
            weights["s:" + st] = weights.get("s:" + st, 0.0) + 1.0
            if st != tok:
                weights["w:" + tok] = weights.get("w:" + tok, 0.0) + 0.35
        for a, b in zip(stems, stems[1:], strict=False):
            key = f"b:{a}_{b}"
            weights[key] = weights.get(key, 0.0) + 0.35
        vec = [0.0] * self.dim
        for feature, weight in weights.items():
            index, sign = self._slot(feature)
            vec[index] += sign * math.log1p(weight)  # TF sublinear: repetição satura
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_header: str = "Authorization",
        extra_headers: dict[str, str] | None = None,
        timeout_s: float = 60.0,
        batch_size: int = 64,
        label: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.label = label or f"embeddings:{model}"
        self._client = httpx.AsyncClient(
            base_url=(base_url or DEFAULT_BASE_URL).rstrip("/"),
            headers=build_headers(api_key, api_key_header, extra_headers),
            timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 10.0)),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                resp = await self._client.post(
                    "/embeddings", json={"model": self.model, "input": batch}
                )
            except httpx.HTTPError as exc:
                raise LLMError(f"{self.label}: falha de rede ({exc})", provider=self.label) from exc
            if resp.status_code >= 400:
                raise LLMError(
                    f"{self.label}: HTTP {resp.status_code} - {error_text(resp)}",
                    provider=self.label,
                    status=resp.status_code,
                )
            try:
                items = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
                vectors.extend([[float(x) for x in item["embedding"]] for item in items])
            except (ValueError, KeyError, TypeError) as exc:
                raise LLMError(f"{self.label}: resposta inesperada: {resp.text[:300]}") from exc
        if len(vectors) != len(texts):
            raise LLMError(f"{self.label}: pedi {len(texts)} embeddings e recebi {len(vectors)}")
        return vectors


def build_embedder(
    spec: EmbedderSpec,
    connection: ModelSpec | None = None,
    *,
    resolve_secret: Callable[[str | None], str | None] = resolve_env,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Embedder:
    """Cria o embedder de uma base; ``connection`` é o modelo que fornece URL e chave."""
    if spec.kind == "hashing":
        return HashingEmbedder(spec.dim)
    if connection is None:
        raise ConfigError(f"embedder '{spec.label}' precisa de uma conexão de modelo")
    if connection.provider != "openai":
        raise ConfigError(
            f"embeddings exigem um provedor compatível com OpenAI; '{connection.name}' é {connection.provider}"
        )
    return OpenAICompatibleEmbedder(
        model=spec.model or "",
        base_url=connection.base_url,
        api_key=resolve_secret(connection.api_key),
        api_key_header=connection.api_key_header,
        extra_headers={k: (resolve_secret(v) or "") for k, v in connection.extra_headers.items()},
        timeout_s=connection.timeout_s,
        label=f"{connection.name}:{spec.model}",
        transport=transport,
    )


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
