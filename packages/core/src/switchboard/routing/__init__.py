"""Motor de roteamento: responder, delegar via MCP ou esclarecer."""

from .args import extract_arguments, validate_arguments
from .deciders import HeuristicDecider, LLMComposer, LLMDecider, TemplateComposer
from .decision import DecisionError, parse_decision, validate_decision
from .engine import RouterEngine, normalize_messages
from .types import Decision, ResolvedProfile, RouterResult, SourceRef, TraceStep

__all__ = [
    "Decision",
    "DecisionError",
    "HeuristicDecider",
    "LLMComposer",
    "LLMDecider",
    "ResolvedProfile",
    "RouterEngine",
    "RouterResult",
    "SourceRef",
    "TemplateComposer",
    "TraceStep",
    "extract_arguments",
    "normalize_messages",
    "parse_decision",
    "validate_arguments",
    "validate_decision",
]
