"""Agentes A2A (v1.0): Agent Card, cliente JSON-RPC e diretório com cache."""

from .card import AgentCard, AgentInterface, SkillCard
from .client import A2AClient, card_url
from .directory import AgentDirectory, AgentInfo, SkillInfo, agent_info_from_card
from .protocol import (
    A2A_VERSION,
    CARD_PATH,
    NOTIFICATION_TOKEN_HEADER,
    ROLE_AGENT,
    ROLE_USER,
    TERMINAL_TASK_STATES,
    StreamEvent,
    TaskState,
    TaskView,
    data_part,
    new_message,
    parse_stream_event,
    parts_data,
    parts_text,
    text_part,
)

__all__ = [
    "A2A_VERSION",
    "CARD_PATH",
    "NOTIFICATION_TOKEN_HEADER",
    "ROLE_AGENT",
    "ROLE_USER",
    "TERMINAL_TASK_STATES",
    "A2AClient",
    "AgentCard",
    "AgentDirectory",
    "AgentInfo",
    "AgentInterface",
    "SkillCard",
    "SkillInfo",
    "StreamEvent",
    "TaskState",
    "TaskView",
    "agent_info_from_card",
    "card_url",
    "data_part",
    "new_message",
    "parse_stream_event",
    "parts_data",
    "parts_text",
    "text_part",
]
