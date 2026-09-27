"""Agentes acionáveis via MCP: descoberta (tools/list) e delegação (tools/call)."""

from .catalog import AgentCatalog, AgentInfo, ToolCallResult, ToolInfo, http_connector, result_text

__all__ = [
    "AgentCatalog",
    "AgentInfo",
    "ToolCallResult",
    "ToolInfo",
    "http_connector",
    "result_text",
]
