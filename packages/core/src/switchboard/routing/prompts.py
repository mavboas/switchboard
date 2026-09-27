"""Prompts do roteador (decisão, reparo e síntese)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ..agents.catalog import AgentInfo
from ..config import ProfileSpec
from ..llm.base import Message
from ..rag.retriever import Hit
from ..text import truncate

HISTORY_LIMIT = 12
SNIPPET_LIMIT = 900


_SCHEMA_MAPS = {"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"}
_SCHEMA_LISTS = {"anyOf", "oneOf", "allOf", "prefixItems"}
_SCHEMA_SINGLE = {
    "items",
    "additionalProperties",
    "additionalItems",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "not",
    "if",
    "then",
    "else",
}


def compact_schema(schema: Any) -> Any:
    """Remove só o ruído (``title`` gerado, ``$schema``) e preserva o resto.

    Objetos aninhados, ``$defs``/``$ref``, limites e formatos continuam lá —
    o LLM precisa deles para montar argumentos de tools mais ricas. Nomes de
    propriedades nunca são tocados (uma propriedade pode se chamar "title").
    """
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key in ("title", "$schema"):
            continue
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            out[key] = {name: compact_schema(sub) for name, sub in value.items()}
        elif key in _SCHEMA_LISTS and isinstance(value, list):
            out[key] = [compact_schema(sub) for sub in value]
        elif key in _SCHEMA_SINGLE:
            out[key] = compact_schema(value)
        else:
            out[key] = value
    return out


def catalog_text(agents: Sequence[AgentInfo]) -> str:
    if not agents:
        return "(nenhum agente disponível agora)"
    items = []
    for agent in agents:
        items.append(
            {
                "agent": agent.name,
                "description": agent.description,
                "tools": [
                    {
                        "tool": t.name,
                        "description": t.description,
                        "arguments": compact_schema(t.input_schema),
                    }
                    for t in agent.tools
                ],
            }
        )
    return json.dumps(items, ensure_ascii=False, indent=1)


def knowledge_text(hits: Sequence[Hit]) -> str:
    if not hits:
        return "(nenhum trecho relevante encontrado)"
    lines = []
    for i, hit in enumerate(hits, start=1):
        origin = f"base: {hit.kb}, documento: {hit.document}"
        if hit.section:
            origin += f", seção: {hit.section}"
        lines.append(f"[{i}] ({origin})\n{truncate(hit.content, SNIPPET_LIMIT)}")
    return "\n\n".join(lines)


def _formats(has_agents: bool, allow_clarify: bool) -> str:
    formats = [
        '{"action": "answer", "answer": "<resposta final ao usuário>", "sources": [<números dos trechos usados>], "reason": "<motivo curto>"}'
    ]
    if has_agents:
        formats.append(
            '{"action": "delegate", "agent": "<nome do agente>", "tool": "<nome da tool>", "arguments": {<argumentos conforme o schema da tool>}, "reason": "<motivo curto>"}'
        )
    if allow_clarify:
        formats.append(
            '{"action": "clarify", "question": "<uma pergunta objetiva ao usuário>", "reason": "<motivo curto>"}'
        )
    return "\n".join(formats)


def decision_system_prompt(
    profile: ProfileSpec, agents: Sequence[AgentInfo], hits: Sequence[Hit]
) -> str:
    has_agents = bool(agents)
    options = [
        '- "answer": o pedido é simples e dá para responder com os TRECHOS DE CONHECIMENTO ou com conhecimento geral seguro.'
    ]
    if has_agents:
        options.append(
            '- "delegate": o pedido precisa de uma capacidade de um dos AGENTES DISPONÍVEIS (cálculo, consulta ou ação em sistema). '
            'Escolha o agente e a tool e preencha "arguments" seguindo o schema da tool.'
        )
    if profile.allow_clarify:
        options.append(
            '- "clarify": falta um dado obrigatório para delegar, ou o pedido é ambíguo. Faça UMA pergunta curta e objetiva.'
        )
    rules = [
        "Use somente agentes e tools listados abaixo; nunca invente nomes."
        if has_agents
        else "Não há agentes disponíveis: não delegue.",
        'Se os trechos de conhecimento respondem o pedido, prefira "answer" e cite os números dos trechos em "sources".',
        "Nunca invente dados, números ou políticas. Sem base e sem agente adequado, diga que não tem a informação.",
        "Não peça dados que já estão na conversa; reaproveite o que o usuário já informou.",
        'Escreva "answer" e "question" em português do Brasil, prontos para o usuário final.',
        "Responda SOMENTE com um objeto JSON válido, sem nenhum texto fora dele.",
    ]
    return (
        f"{profile.system_prompt.strip()}\n\n"
        "## Seu papel\n"
        "Você é o roteador do Switchboard. Para cada pedido, escolha UMA ação:\n"
        + "\n".join(options)
        + "\n\n## Regras\n"
        + "\n".join(f"- {r}" for r in rules)
        + "\n\n## Formato (JSON)\n"
        + _formats(has_agents, profile.allow_clarify)
        + "\n\n## AGENTES DISPONÍVEIS\n"
        + catalog_text(agents)
        + "\n\n## TRECHOS DE CONHECIMENTO\n"
        + knowledge_text(hits)
    )


def history(messages: Sequence[Message], limit: int = HISTORY_LIMIT) -> list[Message]:
    convo = [m for m in messages if m.role in ("user", "assistant") and m.content.strip()]
    return convo[-limit:]


def decision_messages(
    profile: ProfileSpec,
    messages: Sequence[Message],
    agents: Sequence[AgentInfo],
    hits: Sequence[Hit],
) -> list[Message]:
    return [Message("system", decision_system_prompt(profile, agents, hits)), *history(messages)]


def repair_message(error: str) -> Message:
    return Message(
        "user",
        f"Sua resposta anterior não pôde ser usada: {error}\n"
        "Responda novamente com SOMENTE o objeto JSON válido, seguindo o formato e as regras.",
    )


def synthesis_messages(
    profile: ProfileSpec,
    question: str,
    agent: str,
    tool: str,
    arguments: dict[str, Any],
    result_text: str,
    is_error: bool,
) -> list[Message]:
    system = (
        f"{profile.system_prompt.strip()}\n\n"
        "Você acionou um agente especialista para atender o pedido do usuário. Redija a resposta final "
        "com base no resultado do agente: seja fiel aos números e fatos, não acrescente informação que "
        "não esteja no resultado e não mencione JSON, tools ou detalhes internos. Se o agente retornou "
        "erro, explique de forma simples o que deu errado ou o que falta e sugira o próximo passo."
    )
    status = "ERRO" if is_error else "OK"
    user = (
        f"Pedido do usuário:\n{question}\n\n"
        f"Agente: {agent} · tool: {tool}\n"
        f"Argumentos: {json.dumps(arguments, ensure_ascii=False)}\n"
        f"Status: {status}\n"
        f"Resultado do agente:\n{truncate(result_text, 6000)}"
    )
    return [Message("system", system), Message("user", user)]
