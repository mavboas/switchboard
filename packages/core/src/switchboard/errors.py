"""Exceções do Switchboard.

Toda falha esperada (configuração inválida, provedor fora do ar, agente MCP
inalcançável, segredo ausente) vira uma subclasse de ``SwitchboardError`` com
mensagem em português, pronta para aparecer na UI ou no trace de execução.
"""

from __future__ import annotations


class SwitchboardError(Exception):
    """Erro base do Switchboard."""


class ConfigError(SwitchboardError):
    """Configuração inválida ou incompleta (YAML, banco ou formulário)."""


class SecretError(SwitchboardError):
    """Segredo (chave de API, token) ausente ou impossível de decifrar."""


class LLMError(SwitchboardError):
    """Falha ao chamar um provedor de LLM ou de embeddings."""

    def __init__(self, message: str, *, provider: str | None = None, status: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status = status


class AgentError(SwitchboardError):
    """Falha ao descobrir ou acionar um agente via MCP."""


class IngestError(SwitchboardError):
    """Falha ao extrair texto ou indexar um documento na base de conhecimento."""
