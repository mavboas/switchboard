"""Conectores MCP: descoberta (``tools/list``) e acionamento (``tools/call``) de tools.

Cada conector registrado é um servidor MCP: as *tools* que ele expõe são
ações rápidas e síncronas que o roteador pode executar dentro do pedido
(consultar, calcular, abrir um chamado). Tarefas longas ou autônomas não
passam por aqui: vão para agentes A2A, sob contrato
(:mod:`switchboard.a2a`, :mod:`switchboard.contracts`).

O catálogo:

1. conecta em cada conector (Streamable HTTP ou SSE), faz o handshake e chama
   ``tools/list`` — o conector fica ``online`` com suas tools, ou ``offline``
   com o erro;
2. guarda o resultado com cache "stale-while-revalidate";
3. aplica a allowlist de tools do conector, se houver;
4. executa ``tools/call`` quando o roteador decide usar uma tool.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from ..config import ConnectorSpec
from ..discovery import DiscoveryCache, describe_error
from ..errors import ConnectorError
from ..secrets import resolve_env

ConnectorStatus = Literal["online", "offline"]


@dataclass(frozen=True)
class ToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ConnectorInfo:
    name: str
    description: str
    url: str
    status: ConnectorStatus
    tools: list[ToolInfo] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0
    server_name: str | None = None
    checked_at: float = field(default_factory=time.time)

    def tool(self, name: str) -> ToolInfo | None:
        return next((t for t in self.tools if t.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "status": self.status,
            "tools": [t.to_dict() for t in self.tools],
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
            "server_name": self.server_name,
        }


@dataclass
class ToolCallResult:
    text: str
    is_error: bool
    structured: dict[str, Any] | None
    latency_ms: float


# Uma "conexão" é um context manager assíncrono que entrega um mcp.Client já
# inicializado. O padrão abre HTTP; os testes injetam servidores em processo.
Connector = Callable[[ConnectorSpec, str | None], AbstractAsyncContextManager[Any]]


@asynccontextmanager
async def http_connector(spec: ConnectorSpec, token: str | None) -> AsyncIterator[Any]:
    import httpx2
    from mcp import Client
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamable_http_client

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if spec.transport == "sse":
        transport = sse_client(
            spec.url, headers=headers, timeout=spec.timeout_s, sse_read_timeout=spec.timeout_s
        )
        async with Client(transport, read_timeout_seconds=spec.timeout_s) as client:
            yield client
        return
    http = httpx2.AsyncClient(
        headers=headers, timeout=httpx2.Timeout(spec.timeout_s, read=spec.timeout_s)
    )
    async with http:
        transport = streamable_http_client(spec.url, http_client=http)
        async with Client(transport, read_timeout_seconds=spec.timeout_s) as client:
            yield client


def _dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=True, exclude_none=True, mode="json")
    return dict(model)


def result_text(result: Any) -> tuple[str, bool, dict[str, Any] | None]:
    """Texto legível de um CallToolResult (texto das partes ou o JSON estruturado)."""
    data = _dump(result)
    parts = []
    for item in data.get("content") or []:
        if item.get("type") == "text" and item.get("text"):
            parts.append(item["text"])
        elif item.get("type") == "resource":
            resource = item.get("resource") or {}
            if resource.get("text"):
                parts.append(resource["text"])
    structured = data.get("structuredContent")
    text = "\n".join(parts).strip()
    if not text and structured is not None:
        text = json.dumps(structured, ensure_ascii=False, indent=2)
    return text, bool(data.get("isError")), structured


class ConnectorCatalog:
    """Catálogo de conectores MCP com cache "stale-while-revalidate".

    Só o primeiro pedido para um conector ainda desconhecido espera a
    descoberta, limitada a ``discovery_timeout_s``.
    """

    def __init__(
        self,
        *,
        ttl_s: float = 30.0,
        offline_ttl_s: float = 15.0,
        discovery_timeout_s: float = 5.0,
        connector: Connector = http_connector,
        resolve_secret: Callable[[str | None], str | None] = resolve_env,
    ):
        self.discovery_timeout_s = discovery_timeout_s
        self._connector = connector
        self._resolve = resolve_secret
        self._cache: DiscoveryCache[ConnectorSpec, ConnectorInfo] = DiscoveryCache(
            self._probe,
            online=lambda info: info.status == "online",
            name_of=lambda info: info.name,
            ttl_s=ttl_s,
            offline_ttl_s=offline_ttl_s,
        )

    @property
    def ttl_s(self) -> float:
        return self._cache.ttl_s

    def invalidate(self, name: str | None = None) -> None:
        self._cache.invalidate(name)

    async def describe(self, spec: ConnectorSpec, *, refresh: bool = False) -> ConnectorInfo:
        return await self._cache.get(spec, refresh=refresh)

    async def _list_all_tools(self, client: Any) -> list[Any]:
        tools: list[Any] = []
        cursor = None
        for _page in range(50):  # tools/list é paginado
            listing = await (client.list_tools(cursor=cursor) if cursor else client.list_tools())
            tools.extend(listing.tools)
            cursor = getattr(listing, "next_cursor", None) or getattr(listing, "nextCursor", None)
            if not cursor:
                break
        return tools

    async def _probe(self, spec: ConnectorSpec) -> ConnectorInfo:
        started = time.perf_counter()
        try:
            token = self._resolve(spec.auth_token)
            async with asyncio.timeout(min(spec.timeout_s, self.discovery_timeout_s)):
                async with self._connector(spec, token) as client:
                    listed = await self._list_all_tools(client)
                    server = client.server_info
                    instructions = client.instructions
        except Exception as exc:  # qualquer falha = conector offline, com o motivo
            return ConnectorInfo(
                name=spec.name,
                description=spec.description,
                url=spec.url,
                status="offline",
                error=describe_error(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        tools = []
        for tool in listed:
            data = _dump(tool)
            name = data.get("name", "")
            if spec.allowed_tools and name not in spec.allowed_tools:
                continue
            tools.append(
                ToolInfo(
                    name=name,
                    description=(data.get("description") or data.get("title") or "").strip(),
                    input_schema=data.get("inputSchema") or {"type": "object", "properties": {}},
                )
            )
        description = spec.description.strip() or (instructions or "").strip()
        if not description and server is not None:
            description = (
                getattr(server, "description", None) or getattr(server, "title", None) or ""
            ).strip()
        return ConnectorInfo(
            name=spec.name,
            description=description,
            url=spec.url,
            status="online",
            tools=tools,
            latency_ms=(time.perf_counter() - started) * 1000,
            server_name=getattr(server, "name", None) if server is not None else None,
        )

    async def discover(
        self, connectors: Sequence[ConnectorSpec], *, refresh: bool = False
    ) -> list[ConnectorInfo]:
        """Descobre todos os conectores habilitados em paralelo."""
        enabled = [c for c in connectors if c.enabled]
        if not enabled:
            return []
        return list(await asyncio.gather(*(self.describe(c, refresh=refresh) for c in enabled)))

    async def call(
        self, spec: ConnectorSpec, tool: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        if spec.allowed_tools and tool not in spec.allowed_tools:
            raise ConnectorError(f"a tool '{tool}' não está liberada para o conector '{spec.name}'")
        started = time.perf_counter()
        try:
            token = self._resolve(spec.auth_token)
            async with asyncio.timeout(spec.timeout_s):
                async with self._connector(spec, token) as client:
                    result = await client.call_tool(
                        tool, arguments, read_timeout_seconds=spec.timeout_s
                    )
        except Exception as exc:
            self.invalidate(spec.name)
            raise ConnectorError(
                f"falha ao acionar {spec.name}/{tool}: {describe_error(exc)}"
            ) from exc
        text, is_error, structured = result_text(result)
        return ToolCallResult(text, is_error, structured, (time.perf_counter() - started) * 1000)
