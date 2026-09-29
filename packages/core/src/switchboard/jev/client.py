"""Cliente do Jev (TypeSafe System One): decisões tipadas com probabilidades.

O Jev não gera texto. Ele recebe um *estado* (texto, objeto ou lista) e
perguntas tipadas, e devolve para cada uma:

* ``noul`` — probabilidade (0–1) de a resposta ser "sim";
* ``choice`` — a opção mais provável entre as dadas, a distribuição
  (``probabilities``) e a ``confidence`` (concentração da distribuição);
* ``score`` — posição ponderada numa escala ordenada de 2 a 10 níveis.

API: ``POST {base}/v1/systemone`` com ``Authorization: Bearer <chave>``.
Respostas 429/529 (e 5xx) são repetidas com backoff exponencial.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..errors import DecisionModelError
from ..llm.base import Usage, error_text

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
MAX_CHOICE_OPTIONS = 255
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})


# -- construtores de perguntas -------------------------------------------------


def choice(instructions: str, criteria: Mapping[str, Any]) -> dict[str, Any]:
    """Escolha entre opções; ``criteria`` mapeia a chave da opção à descrição.

    A descrição pode ser texto ou um objeto estruturado (``what``,
    ``not_for``, ``examples``) para separar opções parecidas.
    """
    if not 2 <= len(criteria) <= MAX_CHOICE_OPTIONS:
        raise ValueError(
            f"choice precisa de 2 a {MAX_CHOICE_OPTIONS} opções (veio {len(criteria)})"
        )
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def noul(instructions: str, *, true: str | None = None, false: str | None = None) -> dict[str, Any]:
    """Pergunta sim/não; ``true``/``false`` deixam explícito o que conta como cada um."""
    question: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true or false:
        question["criteria"] = {"true": true or "yes", "false": false or "no"}
    return question


def score(instructions: str, levels: Sequence[Any]) -> dict[str, Any]:
    """Escala ordenada (do menor para o maior), de 2 a 10 níveis."""
    if not 2 <= len(levels) <= 10:
        raise ValueError(f"score precisa de 2 a 10 níveis (veio {len(levels)})")
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


# -- respostas -----------------------------------------------------------------


@dataclass
class JevAnswer:
    type: str
    choice: str | None = None
    noul: float | None = None
    score: float | None = None
    probabilities: Any = None
    confidence: float | None = None
    legend: Any = None

    @classmethod
    def parse(cls, data: Mapping[str, Any]) -> JevAnswer:
        kind = str(data.get("type") or "")
        if kind not in ("choice", "noul", "score"):
            raise DecisionModelError(f"Jev: tipo de resposta desconhecido: {kind!r}")

        def number(key: str) -> float | None:
            value = data.get(key)
            return float(value) if isinstance(value, (int, float)) else None

        answer = cls(
            type=kind,
            choice=str(data["choice"]) if data.get("choice") is not None else None,
            noul=number("noul"),
            score=number("score"),
            probabilities=data.get("probabilities"),
            confidence=number("confidence"),
            legend=data.get("legend"),
        )
        missing = {"choice": answer.choice, "noul": answer.noul, "score": answer.score}[kind]
        if missing is None:
            raise DecisionModelError(f"Jev: resposta '{kind}' sem o campo '{kind}'")
        return answer

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        for key in ("choice", "noul", "score", "confidence"):
            value = getattr(self, key)
            if value is not None:
                out[key] = round(value, 4) if isinstance(value, float) else value
        if isinstance(self.probabilities, dict):
            out["probabilities"] = {
                k: round(float(v), 4)
                for k, v in sorted(self.probabilities.items(), key=lambda kv: -float(kv[1]))
            }
        elif self.probabilities is not None:
            out["probabilities"] = self.probabilities
        return out

    def ranked(self) -> list[tuple[str, float]]:
        """Opções de um ``choice`` da mais para a menos provável."""
        if not isinstance(self.probabilities, dict):
            return [(self.choice, 1.0)] if self.choice is not None else []
        return sorted(
            ((str(k), float(v)) for k, v in self.probabilities.items()), key=lambda kv: -kv[1]
        )


@dataclass
class JevResult:
    answers: dict[str, JevAnswer]
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0

    def get(self, key: str) -> JevAnswer | None:
        return self.answers.get(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "answers": {k: a.to_dict() for k, a in self.answers.items()},
            "usage": self.usage.to_dict(),
            "latency_ms": round(self.latency_ms, 1),
        }


def endpoint_url(base_url: str | None) -> str:
    base = (base_url or DEFAULT_BASE_URL).rstrip("/")
    return f"{base}/systemone" if base.endswith("/v1") else f"{base}/v1/systemone"


class JevClient:
    """Cliente assíncrono do endpoint System One."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_header: str = "Authorization",
        extra_headers: Mapping[str, str] | None = None,
        timeout_s: float = 10.0,
        max_retries: int = 2,
        label: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.model = model or DEFAULT_MODEL
        self.url = endpoint_url(base_url)
        self.max_retries = max(0, max_retries)
        self.label = label or f"typesafe:{self.model}"
        self._sleep = sleep
        headers = {"Content-Type": "application/json"}
        if api_key:
            if api_key_header.lower() == "authorization":
                headers["Authorization"] = f"Bearer {api_key}"
            else:
                headers[api_key_header] = api_key
        headers.update(extra_headers or {})
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 5.0)),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _backoff(self, attempt: int, resp: httpx.Response | None) -> float:
        if resp is not None:
            retry_after = resp.headers.get("retry-after")
            try:
                if retry_after is not None:
                    return min(max(float(retry_after), 0.0), 5.0)
            except ValueError:
                pass
        return min(0.5 * (2**attempt), 4.0)

    async def evaluate(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> JevResult:
        if not questions:
            raise ValueError("envie ao menos uma pergunta")
        payload = {"model": self.model, "state": state, "questions": dict(questions)}
        started = time.perf_counter()
        resp: httpx.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(self.url, json=payload)
            except (httpx.HTTPError, RuntimeError) as exc:  # RuntimeError: cliente já fechado
                if attempt < self.max_retries and not isinstance(exc, RuntimeError):
                    await self._sleep(self._backoff(attempt, None))
                    continue
                raise DecisionModelError(
                    f"{self.label}: falha de rede ao chamar {self.url} ({exc.__class__.__name__}: {exc})"
                ) from exc
            if resp.status_code in RETRY_STATUSES and attempt < self.max_retries:
                await self._sleep(self._backoff(attempt, resp))
                continue
            break
        assert resp is not None
        if resp.status_code >= 400:
            raise DecisionModelError(
                f"{self.label}: HTTP {resp.status_code} - {error_text(resp)}",
                status=resp.status_code,
            )
        try:
            data = resp.json()
            raw_answers = data["answers"]
            if not isinstance(raw_answers, dict):
                raise TypeError("answers não é um objeto")
            answers = {str(k): JevAnswer.parse(v) for k, v in raw_answers.items()}
        except DecisionModelError:
            raise
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise DecisionModelError(
                f"{self.label}: resposta inesperada: {resp.text[:300]}"
            ) from exc
        usage = data.get("usage") or {}
        return JevResult(
            answers=answers,
            model=str(data.get("model") or self.model),
            usage=Usage(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
