"""Leitura tolerante e validação da decisão devolvida pelo LLM.

Formatos aceitos (o LLM responde um objeto JSON)::

    {"action": "answer",   "answer": "...", "sources": [1, 2], "reason": "..."}
    {"action": "tool",     "connector": "credito", "tool": "simular_financiamento", "arguments": {...}}
    {"action": "delegate", "tasks": [{"agent": "analise-credito", "skill": "analisar_proposta", "arguments": {...}}]}
    {"action": "clarify",  "question": "...", "reason": "..."}

``delegate`` também aceita uma tarefa só no nível de cima (``agent``,
``skill``, ``arguments``). A validação confere cada tarefa contra as
capacidades reais (tool MCP ou skill A2A) e o tipo final vem da capacidade
encontrada — não do rótulo que o LLM usou.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from ..contracts.terms import validate as validate_schema
from ..text import fold
from .args import describe_fields, validate_arguments
from .capabilities import Capability, find
from .types import Decision, TaskRequest

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

ACTION_ALIASES = {
    "answer": "answer",
    "respond": "answer",
    "reply": "answer",
    "direct": "answer",
    "responder": "answer",
    "resposta": "answer",
    "tool": "tool",
    "call_tool": "tool",
    "use_tool": "tool",
    "ferramenta": "tool",
    "acionar": "tool",
    "delegate": "delegate",
    "delegar": "delegate",
    "agent": "delegate",
    "agente": "delegate",
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
        capability: Capability | None = None,
        missing: list[str] | None = None,
        arguments: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.capability = capability
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


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise DecisionError("'arguments' precisa ser um objeto JSON") from exc
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise DecisionError("'arguments' precisa ser um objeto JSON")
    return value


def _text(data: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = data.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _task(data: dict[str, Any], default_kind: str) -> TaskRequest:
    owner = _text(data, "agent", "agente", "connector", "conector", "server", "owner") or ""
    name = _text(data, "skill", "tool", "ferramenta", "name") or ""
    kind = (
        "skill" if (data.get("skill") or data.get("agent") or data.get("agente")) else default_kind
    )
    return TaskRequest(
        kind=kind,  # type: ignore[arg-type]
        owner=owner,
        name=name,
        arguments=_arguments(data.get("arguments", data.get("args", {}))),
    )


def parse_decision(text: str) -> Decision:
    data = extract_json_object(text)
    raw_action = str(data.get("action") or data.get("acao") or "").strip().lower()
    action = ACTION_ALIASES.get(raw_action)
    if action is None:
        raise DecisionError(
            f"campo 'action' inválido ({raw_action!r}); use answer, tool, delegate ou clarify"
        )
    tasks: list[TaskRequest] = []
    if action in ("tool", "delegate"):
        raw_tasks = data.get("tasks") or data.get("tarefas")
        if isinstance(raw_tasks, list) and raw_tasks:
            for item in raw_tasks:
                if not isinstance(item, dict):
                    raise DecisionError("cada item de 'tasks' precisa ser um objeto")
                tasks.append(_task(item, "skill" if action == "delegate" else "tool"))
        else:
            tasks.append(_task(data, "skill" if action == "delegate" else "tool"))
    return Decision(
        action=action,  # type: ignore[arg-type]
        reason=_text(data, "reason", "motivo") or "",
        answer=_text(data, "answer", "resposta"),
        sources=_as_int_list(data.get("sources", data.get("fontes"))),
        question=_text(data, "question", "pergunta"),
        tasks=tasks,
        decided_by="llm",
    )


def resolve_capability(task: TaskRequest, caps: Sequence[Capability]) -> Capability:
    """Acha a capacidade de uma tarefa (aceita ``dono/nome`` no nome e nome único)."""
    name, owner = task.name, task.owner or None
    for sep in ("/", ".", ":"):
        if not owner and sep in name:
            prefix, suffix = name.split(sep, 1)
            if any(fold(c.owner) == fold(prefix) for c in caps):
                owner, name = prefix, suffix
                break
    found = find(caps, owner, name) or (find(caps, None, name) if owner else None)
    if found is None:
        options = ", ".join(c.key for c in caps) or "nenhuma"
        raise DecisionError(
            f"capacidade {task.owner + '/' if task.owner else ''}{task.name!r} não existe; opções: {options}"
        )
    return found


def check_arguments(cap: Capability, arguments: dict[str, Any]) -> dict[str, Any]:
    """Converte e confere os argumentos; faltas e erros viram :class:`DecisionError`."""
    schema = cap.arguments_schema
    converted, errors, missing = validate_arguments(schema, arguments)
    if not errors and not missing and cap.kind == "skill" and cap.input_schema is not None:
        errors = validate_schema(converted, cap.input_schema)  # o contrato é estrito
    if errors:
        raise DecisionError(
            f"argumentos inválidos para {cap.key}: " + "; ".join(errors),
            capability=cap,
            missing=missing,
            arguments=converted,
        )
    if missing:
        labels = ", ".join(describe_fields(schema, missing))
        raise DecisionError(
            f"faltam argumentos obrigatórios para {cap.key}: {', '.join(missing)} ({labels})",
            capability=cap,
            missing=missing,
            arguments=converted,
        )
    return converted


def validate_decision(
    decision: Decision,
    caps: Sequence[Capability],
    *,
    allow_clarify: bool,
    n_sources: int,
    max_parallel: int = 3,
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

    if not caps:
        raise DecisionError("não há tools nem agentes disponíveis; use action=answer")
    if not decision.tasks:
        raise DecisionError("informe a tool ou as tarefas a delegar")
    resolved: list[tuple[TaskRequest, Capability]] = []
    for task in decision.tasks:
        cap = resolve_capability(task, caps)
        if any(c.key == cap.key for _, c in resolved):
            continue  # tarefa repetida
        resolved.append((task, cap))
    kinds = {cap.kind for _, cap in resolved}
    if kinds == {"tool"}:
        if len(resolved) > 1:
            raise DecisionError("acione uma tool por vez (use delegate só para agentes)")
        decision.action = "tool"
    elif kinds == {"skill"}:
        if len(resolved) > max_parallel:
            raise DecisionError(f"no máximo {max_parallel} tarefas delegadas por pedido")
        decision.action = "delegate"
    else:
        raise DecisionError("não misture tool MCP com delegação a agente no mesmo pedido")
    tasks = []
    for task, cap in resolved:
        arguments = check_arguments(cap, task.arguments)
        instruction = str(arguments.get("instrucao") or "") if cap.input_schema is None else ""
        tasks.append(
            TaskRequest(cap.kind, cap.owner, cap.name, arguments, instruction or task.instruction)
        )
    decision.tasks = tasks
    return decision


def clarify_for_missing(cap: Capability, missing: list[str]) -> str:
    labels = describe_fields(cap.arguments_schema, missing)
    what = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " e " + labels[-1]
    return f"Consigo ajudar com isso, mas preciso de: {what}. Pode me informar?"
