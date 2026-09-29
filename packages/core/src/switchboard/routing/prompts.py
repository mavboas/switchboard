"""Prompts do LLM (System Two): decisão completa, resposta, argumentos, síntese e consolidação."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ..config import ProfileSpec
from ..llm.base import Message
from ..rag.retriever import Hit
from ..text import truncate
from .capabilities import Capability

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
    o LLM precisa deles para montar argumentos mais ricos. Nomes de
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


def catalog_text(caps: Sequence[Capability]) -> str:
    tools = [c for c in caps if c.kind == "tool"]
    skills = [c for c in caps if c.kind == "skill"]
    sections = []
    if tools:
        items = [
            {
                "connector": c.owner,
                "tool": c.name,
                "description": c.summary(),
                "arguments": compact_schema(c.arguments_schema),
            }
            for c in tools
        ]
        sections.append(
            "### TOOLS (conectores MCP: ações rápidas, executadas na hora)\n"
            + json.dumps(items, ensure_ascii=False, indent=1)
        )
    if skills:
        items = []
        for c in skills:
            item: dict[str, Any] = {
                "agent": c.owner,
                "skill": c.name,
                "description": c.summary(),
                "arguments": compact_schema(c.arguments_schema),
            }
            if c.examples:
                item["examples"] = list(c.examples[:3])
            items.append(item)
        sections.append(
            "### AGENTES (A2A: tarefas delegadas sob contrato; podem levar minutos)\n"
            + json.dumps(items, ensure_ascii=False, indent=1)
        )
    return "\n\n".join(sections) or "(nenhuma tool ou agente disponível agora)"


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


def _formats(has_tools: bool, has_agents: bool, allow_clarify: bool, max_parallel: int) -> str:
    formats = [
        '{"action": "answer", "answer": "<resposta final ao usuário>", "sources": [<números dos trechos usados>], "reason": "<motivo curto>"}'
    ]
    if has_tools:
        formats.append(
            '{"action": "tool", "connector": "<conector>", "tool": "<tool>", "arguments": {<conforme o schema>}, "reason": "<motivo curto>"}'
        )
    if has_agents:
        formats.append(
            '{"action": "delegate", "tasks": [{"agent": "<agente>", "skill": "<skill>", "arguments": {<conforme o schema>}}], "reason": "<motivo curto>"}'
            + (f"  (até {max_parallel} tarefas independentes)" if max_parallel > 1 else "")
        )
    if allow_clarify:
        formats.append(
            '{"action": "clarify", "question": "<uma pergunta objetiva ao usuário>", "reason": "<motivo curto>"}'
        )
    return "\n".join(formats)


def decision_system_prompt(
    profile: ProfileSpec, caps: Sequence[Capability], hits: Sequence[Hit]
) -> str:
    has_tools = any(c.kind == "tool" for c in caps)
    has_agents = any(c.kind == "skill" for c in caps)
    options = [
        '- "answer": o pedido é simples e dá para responder com os TRECHOS DE CONHECIMENTO ou com conhecimento geral seguro.'
    ]
    if has_tools:
        options.append(
            '- "tool": o pedido precisa de uma ação rápida de uma TOOL (consulta, cálculo, registro). '
            'Escolha o conector e a tool e preencha "arguments" seguindo o schema.'
        )
    if has_agents:
        options.append(
            '- "delegate": o pedido é uma tarefa para um AGENTE (análise, processo com várias etapas). '
            "Cada tarefa vira um contrato com o agente; se o pedido tiver tarefas independentes para "
            'agentes diferentes, liste todas em "tasks".'
        )
    if profile.allow_clarify:
        options.append(
            '- "clarify": falta um dado obrigatório ou o pedido é ambíguo. Faça UMA pergunta curta e objetiva.'
        )
    rules = [
        "Use somente tools e agentes listados abaixo; nunca invente nomes."
        if caps
        else "Não há tools nem agentes disponíveis: não use tool nem delegate.",
        'Se os trechos de conhecimento respondem o pedido, prefira "answer" e cite os números dos trechos em "sources".',
        "Nunca invente dados, números ou políticas. Sem base e sem capacidade adequada, diga que não tem a informação.",
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
        + _formats(has_tools, has_agents, profile.allow_clarify, profile.max_parallel)
        + "\n\n## CAPACIDADES DISPONÍVEIS\n"
        + catalog_text(caps)
        + "\n\n## TRECHOS DE CONHECIMENTO\n"
        + knowledge_text(hits)
    )


def history(messages: Sequence[Message], limit: int = HISTORY_LIMIT) -> list[Message]:
    convo = [m for m in messages if m.role in ("user", "assistant") and m.content.strip()]
    return convo[-limit:]


def decision_messages(
    profile: ProfileSpec,
    messages: Sequence[Message],
    caps: Sequence[Capability],
    hits: Sequence[Hit],
) -> list[Message]:
    return [Message("system", decision_system_prompt(profile, caps, hits)), *history(messages)]


def repair_message(error: str) -> Message:
    return Message(
        "user",
        f"Sua resposta anterior não pôde ser usada: {error}\n"
        "Responda novamente com SOMENTE o objeto JSON válido, seguindo o formato e as regras.",
    )


def answer_messages(
    profile: ProfileSpec, messages: Sequence[Message], hits: Sequence[Hit], *, in_scope: bool = True
) -> list[Message]:
    """Resposta direta (a decisão já foi tomada): texto final com citações [n]."""
    rules = [
        "Responda em português do Brasil, de forma direta e cordial.",
        "Use os TRECHOS DE CONHECIMENTO quando forem relevantes e cite-os no texto como [1], [2].",
        "Nunca invente dados, números ou políticas; se a informação não estiver nos trechos, diga que não a tem.",
    ]
    if not in_scope:
        rules.append(
            "O pedido está fora do que este assistente atende: diga isso com gentileza e ofereça o que você pode fazer."
        )
    system = (
        f"{profile.system_prompt.strip()}\n\n## Regras\n"
        + "\n".join(f"- {r}" for r in rules)
        + "\n\n## TRECHOS DE CONHECIMENTO\n"
        + knowledge_text(hits)
    )
    return [Message("system", system), *history(messages)]


def arguments_messages(
    profile: ProfileSpec, messages: Sequence[Message], cap: Capability
) -> list[Message]:
    """Extração dos argumentos de UMA capacidade já escolhida (JSON)."""
    kind = "tool" if cap.kind == "tool" else "tarefa para o agente"
    system = (
        "Você extrai argumentos para acionar uma capacidade já escolhida pelo roteador.\n"
        f"Capacidade ({kind}): {cap.key}\nDescrição: {cap.summary()}\n"
        "Schema dos argumentos (JSON Schema):\n"
        + json.dumps(compact_schema(cap.arguments_schema), ensure_ascii=False, indent=1)
        + "\n\n## Regras\n"
        "- Use SOMENTE valores que o usuário informou na conversa; nunca invente nem suponha valores.\n"
        "- Omita campos que o usuário não informou (mesmo os obrigatórios).\n"
        '- Converta números e unidades para o que o schema pede (ex.: "50 mil" -> 50000; "2 anos" -> 24 se o campo for em meses).\n'
        '- Se o schema tiver o campo "instrucao", escreva a tarefa de forma autocontida, com todos os dados que o usuário já deu.\n'
        '- Responda SOMENTE com um objeto JSON: {"arguments": {...}}'
    )
    return [Message("system", system), *history(messages)]


def synthesis_messages(
    profile: ProfileSpec,
    question: str,
    capability: str,
    arguments: dict[str, Any],
    result_text: str,
    is_error: bool,
) -> list[Message]:
    system = (
        f"{profile.system_prompt.strip()}\n\n"
        "Você acionou uma ferramenta para atender o pedido do usuário. Redija a resposta final "
        "com base no resultado: seja fiel aos números e fatos, não acrescente informação que "
        "não esteja no resultado e não mencione JSON, tools ou detalhes internos. Se a ferramenta "
        "retornou erro, explique de forma simples o que deu errado ou o que falta e sugira o próximo passo."
    )
    status = "ERRO" if is_error else "OK"
    user = (
        f"Pedido do usuário:\n{question}\n\n"
        f"Ferramenta: {capability}\n"
        f"Argumentos: {json.dumps(arguments, ensure_ascii=False)}\n"
        f"Status: {status}\n"
        f"Resultado:\n{truncate(result_text, 6000)}"
    )
    return [Message("system", system), Message("user", user)]


def consolidation_messages(
    profile: ProfileSpec,
    question: str,
    context: Sequence[dict[str, str]],
    results: Sequence[dict[str, Any]],
) -> list[Message]:
    system = (
        f"{profile.system_prompt.strip()}\n\n"
        "Você delegou tarefas a agentes especialistas, cada uma sob um contrato. Agora consolide "
        "os resultados numa única resposta ao usuário:\n"
        "- seja fiel aos dados de cada agente; não invente nem arredonde números por conta própria;\n"
        "- se algum contrato não foi concluído (falhou, expirou, foi rejeitado ou violado), diga "
        "o que ficou pendente e o próximo passo;\n"
        "- não mencione JSON, contratos, ids ou detalhes internos;\n"
        "- responda em português do Brasil."
    )
    convo = "\n".join(
        f"{m.get('role', 'user')}: {truncate(str(m.get('content', '')), 500)}" for m in context[-6:]
    )
    user = (
        f"Conversa recente:\n{convo or '(sem histórico)'}\n\n"
        f"Pedido atual:\n{question}\n\n"
        "Resultados dos agentes:\n"
        + json.dumps(list(results), ensure_ascii=False, indent=1)[:12000]
    )
    return [Message("system", system), Message("user", user)]
