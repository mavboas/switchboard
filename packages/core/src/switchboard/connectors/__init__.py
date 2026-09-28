"""Conectores MCP: descoberta (tools/list) e acionamento síncrono de tools (tools/call)."""

from .catalog import (
    ConnectorCatalog,
    ConnectorInfo,
    ToolCallResult,
    ToolInfo,
    http_connector,
    result_text,
)

__all__ = [
    "ConnectorCatalog",
    "ConnectorInfo",
    "ToolCallResult",
    "ToolInfo",
    "http_connector",
    "result_text",
]
