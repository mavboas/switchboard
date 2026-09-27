"""Leitura tolerante e validação da decisão devolvida pelo LLM."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from ..agents.catalog import AgentInfo, ToolInfo
from ..text import fold
from .args import describe_fields, validate_arguments
from .types import Decision

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

ACTION_ALIASES = {
    "answer": "answer",
    "respond": "answer",
    "reply": "answer",
    "direct": "answer",
    "responder": "answer",
    "resposta": "answer",
    "delegate": "delegate",
    "delegar": "delegate",
    "tool": "delegate",
    "call_tool": "delegate",
    "acionar": "delegate",
    "clarify": "clarify",
    "clarification": "clarify",
    "ask": "clarify",
    "esclarecer": "clarify",
    "perguntar": "clarify",
}


class DecisionError(ValueError):
    """Decisão ilegível ou inválida; a mensagem vai no prompt de reparo."""

    def __init__(
        self,
        message: str,
        *,
        agent: AgentInfo | None = None,
        tool: ToolInfo | None = None,
        missing: list[str] | None = None,
        arguments: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.agent = agent
        self.tool = tool
        self.missing = missing or []
        self.arguments = arguments or {}


def extract_json_object(text: str) -> dict[str, Any]:
    """Acha o primeiro objeto JSON do texto (com ou sem bloco ```json)."""
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        start = candidate.find("{")
        while start != -1:
            try:
                obj, _ = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                start = candidate.find("{", start + 1)
                continue
            if isinstance(obj, dict):
                return obj
            start = candidate.find("{", start + 1)
    raise DecisionError("a resposta não contém um objeto JSON")


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    out = []
    for item in value:
        match = re.search(r"\d+", str(item))
        if match:
            out.append(int(match.group()))
    return out


def parse_decision(text: str) -> Decision:
    data = extract_json_object(text)
    raw_action = str(data.get("action") or data.get("acao") or "").strip().lower()
    action = ACTION_ALIASES.get(raw_action)
    if action is None:
        raise DecisionError(
            f"campo 'action' inválido ({raw_action!r}); use answer, delegate ou clarify"
        )
    arguments = data.get("arguments", data.get("args", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise DecisionError("'arguments' precisa ser um objeto JSON") from exc
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise DecisionError("'arguments' precisa ser um objeto JSON")

    def text_field(*names: str) -> str | None:
        for name in names:
            value = data.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    return Decision(
        action=action,  # type: ignore[arg-type]
        reason=text_field("reason", "motivo") or "",
        answer=text_field("answer", "resposta"),
        sources=_as_int_list(data.get("sources", data.get("fontes"))),
        agent=text_field("agent", "agente"),
        tool=text_field("tool", "ferramenta"),
        arguments=arguments,
        question=text_field("question", "pergunta"),
    )


def _find_agent(agents: Sequence[AgentInfo], name: str | None) -> AgentInfo | None:
    if not name:
        return None
    wanted = fold(name).strip()
    return next((a for a in agents if fold(a.name) == wanted), None)


def validate_decision(
    decision: Decision,
    agents: Sequence[AgentInfo],
    *,
    allow_clarify: bool,
    n_sources: int,
) -> Decision:
    if decision.action == "answer":
        if not decision.answer:
            raise DecisionError("action=answer exige o campo 'answer' preenchido")
        decision.sources = sorted({s for s in decision.sources if 1 <= s <= n_sources})
        return decision

    if decision.action == "clarify":
        if not allow_clarify:
            raise DecisionError(
                "este roteador não faz perguntas de esclarecimento; responda ou delegue"
            )
        if not decision.question:
            raise DecisionError("action=clarify exige o campo 'question' preenchido")
        return decision

    # delegate
    if not agents:
        raise DecisionError("não há agentes disponíveis; use action=answer")
    tool_name = decision.tool or ""
    agent = _find_agent(agents, decision.agent)
    if agent is None and tool_name:
        # aceita "agente/tool" ou "agente.tool" no campo tool
        for sep in ("/", ".", ":"):
            if sep in tool_name:
                prefix, suffix = tool_name.split(sep, 1)
                if _find_agent(agents, prefix):
                    agent, tool_name = _find_agent(agents, prefix), suffix
                    break
    if agent is None and tool_name:
        owners = [a for a in agents if a.tool(tool_name)]
        if len(owners) == 1:
            agent = owners[0]
    if agent is None:
        names = ", ".join(a.name for a in agents)
        raise DecisionError(f"agente {decision.agent!r} não existe; opções: {names}")
    tool = agent.tool(tool_name)
    if tool is None:
        names = ", ".join(t.name for t in agent.tools)
        raise DecisionError(
            f"a tool {tool_name!r} não existe no agente {agent.name}; opções: {names}", agent=agent
        )
    arguments, errors, missing = validate_arguments(tool.input_schema, decision.arguments)
    if errors:
        raise DecisionError(
            f"argumentos inválidos para {agent.name}/{tool.name}: " + "; ".join(errors),
            agent=agent,
            tool=tool,
            missing=missing,
            arguments=arguments,
        )
    if missing:
        labels = ", ".join(describe_fields(tool.input_schema, missing))
        raise DecisionError(
            f"faltam argumentos obrigatórios para {agent.name}/{tool.name}: {', '.join(missing)} ({labels})",
            agent=agent,
            tool=tool,
            missing=missing,
            arguments=arguments,
        )
    decision.agent, decision.tool, decision.arguments = agent.name, tool.name, arguments
    return decision


def clarify_for_missing(tool: ToolInfo, missing: list[str], agent: AgentInfo | None = None) -> str:
    labels = describe_fields(tool.input_schema, missing)
    what = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " e " + labels[-1]
    return f"Consigo ajudar com isso, mas preciso de: {what}. Pode me informar?"
