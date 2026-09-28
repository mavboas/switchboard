"""Catálogo unificado de capacidades: tools MCP e skills de agentes A2A.

Para decidir, o roteador enxerga uma lista só: cada tool de um conector MCP
(ação rápida, síncrona) e cada skill de um agente A2A (tarefa delegada, sob
contrato). A chave ``dono/nome`` identifica a capacidade no prompt do LLM e
nas opções do Jev.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..a2a.directory import AgentInfo, SkillInfo
from ..connectors.catalog import ConnectorInfo, ToolInfo

INSTRUCTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "instrucao": {
            "type": "string",
            "description": "a tarefa, autocontida, com todos os dados que o usuário já informou",
        }
    },
    "required": ["instrucao"],
}


@dataclass(frozen=True)
class Capability:
    kind: Literal["tool", "skill"]
    owner: str
    name: str
    title: str
    description: str
    owner_description: str = ""
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    examples: tuple[str, ...] = ()
    contract: str | None = None  # completo | basico (só skills)
    tool: ToolInfo | None = field(default=None, compare=False, repr=False)
    skill: SkillInfo | None = field(default=None, compare=False, repr=False)

    @property
    def key(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def arguments_schema(self) -> dict[str, Any]:
        """Schema dos argumentos que o decisor precisa preencher.

        Skills sem schema (contrato básico) recebem uma instrução em texto.
        """
        if self.input_schema is not None:
            return self.input_schema
        if self.kind == "skill":
            return INSTRUCTION_SCHEMA
        return {"type": "object", "properties": {}}

    @property
    def label(self) -> str:
        return f"{self.owner}/{self.name}"

    def summary(self) -> str:
        text = self.description or self.title
        if self.owner_description and self.owner_description not in text:
            text = f"{text} ({self.owner_description})"
        return text


def capabilities(
    connectors: Sequence[ConnectorInfo], agents: Sequence[AgentInfo]
) -> list[Capability]:
    out: list[Capability] = []
    for connector in connectors:
        if connector.status != "online":
            continue
        for tool in connector.tools:
            out.append(
                Capability(
                    kind="tool",
                    owner=connector.name,
                    name=tool.name,
                    title=tool.name,
                    description=tool.description,
                    owner_description=connector.description,
                    input_schema=tool.input_schema,
                    tool=tool,
                )
            )
    for agent in agents:
        if agent.status != "online":
            continue
        for skill in agent.skills:
            if not skill.usable:
                continue
            out.append(
                Capability(
                    kind="skill",
                    owner=agent.name,
                    name=skill.id,
                    title=skill.name,
                    description=skill.description or skill.name,
                    owner_description=agent.description,
                    input_schema=skill.terms.input_schema if skill.terms else None,
                    output_schema=skill.terms.output_schema if skill.terms else None,
                    examples=skill.examples,
                    contract=skill.contract,
                    skill=skill,
                )
            )
    return out


def find(caps: Sequence[Capability], owner: str | None, name: str | None) -> Capability | None:
    from ..text import fold

    if not name:
        return None
    wanted_name = fold(name).strip()
    wanted_owner = fold(owner).strip() if owner else None
    matches = [
        c
        for c in caps
        if fold(c.name) == wanted_name and (wanted_owner is None or fold(c.owner) == wanted_owner)
    ]
    return matches[0] if len(matches) == 1 else None


def by_key(caps: Sequence[Capability], key: str) -> Capability | None:
    return next((c for c in caps if c.key == key), None)
