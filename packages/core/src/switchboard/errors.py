"""Exceções do Switchboard.

Toda falha esperada (configuração inválida, provedor fora do ar, conector MCP
ou agente A2A inalcançável, segredo ausente, contrato violado) vira uma
subclasse de ``SwitchboardError`` com mensagem em português, pronta para
aparecer na UI ou no trace de execução.
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


class DecisionModelError(SwitchboardError):
    """Falha ao consultar o modelo de decisão (Jev / TypeSafe System One)."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


class ConnectorError(SwitchboardError):
    """Falha ao descobrir ou acionar um conector MCP (tools)."""


class AgentError(SwitchboardError):
    """Falha ao descobrir ou conversar com um agente A2A."""

    def __init__(self, message: str, *, code: int | None = None, reason: str | None = None):
        super().__init__(message)
        self.code = code
        self.reason = reason


class ContractError(SwitchboardError):
    """Contrato inválido: entrada fora do schema, termos divergentes ou transição proibida."""


class IngestError(SwitchboardError):
    """Falha ao extrair texto ou indexar um documento na base de conhecimento."""
