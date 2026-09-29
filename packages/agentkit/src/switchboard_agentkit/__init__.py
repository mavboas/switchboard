"""Kit para agentes A2A que falam o contrato fechado do Switchboard.

Construído sobre o SDK oficial (``a2a-sdk``): o card declara os schemas de
cada skill na extensão ``urn:switchboard:a2a:contract:v1`` e o executor
confere termos, entrada e saída de cada tarefa.
"""

from .agent import (
    ContractAgent,
    ContractExecutor,
    InputRequired,
    Skill,
    SkillContext,
    SkillFailed,
    SkillResult,
)

__all__ = [
    "ContractAgent",
    "ContractExecutor",
    "InputRequired",
    "Skill",
    "SkillContext",
    "SkillFailed",
    "SkillResult",
]
