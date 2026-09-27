"""Provedores de LLM plugáveis."""

from .anthropic import AnthropicChat
from .base import ChatModel, ChatResult, Message, Usage
from .factory import build_chat_model
from .offline import OfflineChat
from .openai_compat import OpenAICompatibleChat
from .presets import PRESETS, Preset

__all__ = [
    "PRESETS",
    "AnthropicChat",
    "ChatModel",
    "ChatResult",
    "Message",
    "OfflineChat",
    "OpenAICompatibleChat",
    "Preset",
    "Usage",
    "build_chat_model",
]
