"""Cliente A2A v1.0 (binding JSON-RPC 2.0 sobre HTTP), sem dependência do SDK.

Só o que o roteador usa: buscar o Agent Card, ``SendMessage`` (sempre com
``returnImmediately``: quem espera é o roteador, não a conexão),
``GetTask`` e ``CancelTask``. Erros JSON-RPC viram :class:`AgentError` com o
código e o ``reason`` do ``google.rpc.ErrorInfo``.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from ..errors import AgentError
from .protocol import A2A_VERSION, CARD_PATH, EXTENSIONS_HEADER, VERSION_HEADER


def card_url(url: str) -> str:
    """URL do Agent Card a partir da URL cadastrada (base do agente ou o próprio card)."""
    url = url.strip()
    if url.endswith(".json"):
        return url
    return url.rstrip("/") + CARD_PATH


def _headers(
    token: str | None,
    extensions: Sequence[str] | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    headers = {VERSION_HEADER: A2A_VERSION}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extensions:
        headers[EXTENSIONS_HEADER] = ", ".join(extensions)
    headers.update(extra or {})
    return headers


def _error_reason(error: Mapping[str, Any]) -> str | None:
    for detail in error.get("data") or []:
        if isinstance(detail, Mapping) and detail.get("reason"):
            return str(detail["reason"])
    return None


class A2AClient:
    def __init__(
        self,
        *,
        http: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = 15.0,
    ):
        self._own = http is None
        self.http = http or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 5.0)), transport=transport
        )

    async def aclose(self) -> None:
        if self._own:
            await self.http.aclose()

    async def fetch_card(
        self, url: str, *, token: str | None = None, timeout_s: float | None = None
    ) -> dict[str, Any]:
        target = card_url(url)
        try:
            resp = await self.http.get(
                target,
                headers=_headers(token),
                timeout=timeout_s if timeout_s is not None else httpx.USE_CLIENT_DEFAULT,
            )
        except httpx.HTTPError as exc:
            raise AgentError(
                f"não consegui buscar o Agent Card em {target}: {exc.__class__.__name__}: {exc}"
            ) from exc
        if resp.status_code != 200:
            raise AgentError(f"Agent Card em {target} respondeu HTTP {resp.status_code}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise AgentError(f"Agent Card em {target} não é JSON") from exc
        if not isinstance(data, dict):
            raise AgentError(f"Agent Card em {target} não é um objeto JSON")
        return data

    async def call(
        self,
        endpoint: str,
        method: str,
        params: Mapping[str, Any],
        *,
        token: str | None = None,
        extensions: Sequence[str] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        request_id = uuid.uuid4().hex
        body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params)}
        try:
            resp = await self.http.post(
                endpoint,
                json=body,
                headers=_headers(token, extensions, headers),
                timeout=timeout_s if timeout_s is not None else httpx.USE_CLIENT_DEFAULT,
            )
        except httpx.HTTPError as exc:
            raise AgentError(
                f"{method} em {endpoint} falhou: {exc.__class__.__name__}: {exc}"
            ) from exc
        try:
            data = resp.json()
        except ValueError as exc:
            raise AgentError(
                f"{method} em {endpoint}: HTTP {resp.status_code}, resposta não é JSON"
            ) from exc
        if not isinstance(data, Mapping):
            raise AgentError(f"{method} em {endpoint}: resposta JSON-RPC inválida")
        error = data.get("error")
        if isinstance(error, Mapping):
            reason = _error_reason(error)
            code = error.get("code")
            raise AgentError(
                f"{method} recusado pelo agente ({code}{', ' + reason if reason else ''}): "
                f"{error.get('message') or 'erro sem mensagem'}",
                code=code if isinstance(code, int) else None,
                reason=reason,
            )
        if resp.status_code >= 400:
            raise AgentError(f"{method} em {endpoint}: HTTP {resp.status_code}")
        if "result" not in data:
            raise AgentError(f"{method} em {endpoint}: resposta JSON-RPC sem 'result'")
        return data["result"]

    async def send_message(
        self,
        endpoint: str,
        message: Mapping[str, Any],
        *,
        configuration: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"message": dict(message)}
        if configuration:
            params["configuration"] = dict(configuration)
        if metadata:
            params["metadata"] = dict(metadata)
        result = await self.call(endpoint, "SendMessage", params, **kwargs)
        if not isinstance(result, Mapping) or not (
            isinstance(result.get("task"), Mapping) or isinstance(result.get("message"), Mapping)
        ):
            raise AgentError("SendMessage: o agente não devolveu nem 'task' nem 'message'")
        return dict(result)

    async def get_task(
        self, endpoint: str, task_id: str, *, history_length: int = 0, **kwargs: Any
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"id": task_id}
        if history_length:
            params["historyLength"] = history_length
        result = await self.call(endpoint, "GetTask", params, **kwargs)
        if not isinstance(result, Mapping):
            raise AgentError("GetTask: resposta sem a tarefa")
        return dict(result)

    async def cancel_task(self, endpoint: str, task_id: str, **kwargs: Any) -> dict[str, Any]:
        result = await self.call(endpoint, "CancelTask", {"id": task_id}, **kwargs)
        return dict(result) if isinstance(result, Mapping) else {}
