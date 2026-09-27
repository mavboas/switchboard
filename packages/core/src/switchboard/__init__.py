"""Switchboard: roteador de agentes.

Recebe o pedido, responde o que sabe (com RAG) e transfere o resto para o
agente certo, descoberto e acionado via MCP.
"""

from .agents import AgentCatalog, AgentInfo, ToolInfo
from .app import Switchboard
from .config import (
    AgentSpec,
    EmbedderSpec,
    KnowledgeBaseSpec,
    ModelSpec,
    ProfileSpec,
    SwitchboardSpec,
    load_yaml,
)
from .errors import AgentError, ConfigError, IngestError, LLMError, SecretError, SwitchboardError
from .llm import Message, build_chat_model
from .routing import ResolvedProfile, RouterEngine, RouterResult

__version__ = "0.1.0"

__all__ = [
    "AgentCatalog",
    "AgentError",
    "AgentInfo",
    "AgentSpec",
    "ConfigError",
    "EmbedderSpec",
    "IngestError",
    "KnowledgeBaseSpec",
    "LLMError",
    "Message",
    "ModelSpec",
    "ProfileSpec",
    "ResolvedProfile",
    "RouterEngine",
    "RouterResult",
    "SecretError",
    "Switchboard",
    "SwitchboardError",
    "SwitchboardSpec",
    "ToolInfo",
    "__version__",
    "build_chat_model",
    "load_yaml",
]
