"""Switchboard: roteador de agentes.

Recebe o pedido, responde o que sabe (com RAG), aciona tools rápidas via MCP
e delega tarefas a agentes via A2A — cada delegação sob um contrato fechado.
A decisão pode ficar com um modelo de decisão (Jev, System One) e a redação
com um LLM (System Two).
"""

from .a2a import AgentDirectory, AgentInfo, SkillInfo
from .app import Switchboard
from .config import (
    AgentSpec,
    ConnectorSpec,
    EmbedderSpec,
    KnowledgeBaseSpec,
    ModelSpec,
    ProfileSpec,
    SwitchboardSpec,
    load_yaml,
)
from .connectors import ConnectorCatalog, ConnectorInfo, ToolInfo
from .contracts import ContractManager, ContractRecord, MemoryContractStore
from .errors import (
    AgentError,
    ConfigError,
    ConnectorError,
    ContractError,
    DecisionModelError,
    IngestError,
    LLMError,
    SecretError,
    SwitchboardError,
)
from .jev import JevClient, build_decision_model
from .llm import Message, build_chat_model
from .routing import ResolvedProfile, RouterEngine, RouterResult

__version__ = "0.2.0"

__all__ = [
    "AgentDirectory",
    "AgentError",
    "AgentInfo",
    "AgentSpec",
    "ConfigError",
    "ConnectorCatalog",
    "ConnectorError",
    "ConnectorInfo",
    "ConnectorSpec",
    "ContractError",
    "ContractManager",
    "ContractRecord",
    "DecisionModelError",
    "EmbedderSpec",
    "IngestError",
    "JevClient",
    "KnowledgeBaseSpec",
    "LLMError",
    "MemoryContractStore",
    "Message",
    "ModelSpec",
    "ProfileSpec",
    "ResolvedProfile",
    "RouterEngine",
    "RouterResult",
    "SecretError",
    "SkillInfo",
    "Switchboard",
    "SwitchboardError",
    "SwitchboardSpec",
    "ToolInfo",
    "__version__",
    "build_chat_model",
    "build_decision_model",
    "load_yaml",
]
