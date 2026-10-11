"""Motor de roteamento: responder, acionar uma tool MCP, delegar a agentes A2A ou esclarecer."""

from .args import extract_arguments, validate_arguments
from .capabilities import Capability, capabilities
from .deciders import (
    ExtractiveAnswerer,
    HeuristicArgumentFiller,
    HeuristicDecider,
    LLMAnswerer,
    LLMArgumentFiller,
    LLMComposer,
    LLMDecider,
    TemplateComposer,
)
from .decision import DecisionError, parse_decision, validate_decision
from .engine import RouterEngine, normalize_messages
from .jev_decider import DecisionContext, JevDecider, JevSettings
from .types import Decision, ResolvedProfile, RouterResult, SourceRef, TaskRequest, TraceStep

__all__ = [
    "Capability",
    "Decision",
    "DecisionContext",
    "DecisionError",
    "ExtractiveAnswerer",
    "HeuristicArgumentFiller",
    "HeuristicDecider",
    "JevDecider",
    "JevSettings",
    "LLMAnswerer",
    "LLMArgumentFiller",
    "LLMComposer",
    "LLMDecider",
    "ResolvedProfile",
    "RouterEngine",
    "RouterResult",
    "SourceRef",
    "TaskRequest",
    "TemplateComposer",
    "TraceStep",
    "capabilities",
    "extract_arguments",
    "normalize_messages",
    "parse_decision",
    "validate_arguments",
    "validate_decision",
]
