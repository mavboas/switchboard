"""Máquina de estados do contrato.

::

    proposto ──► ativo ◄──► aguardando_entrada
       │           │              │
       └───────────┴──────────────┴──► concluido | falhou | rejeitado |
                                       cancelado | expirado | violado

* ``proposto`` — gravado pelo roteador antes do envio (push pode chegar antes
  da resposta do ``SendMessage``);
* ``ativo`` — o agente aceitou (tarefa ``SUBMITTED``/``WORKING``);
* ``aguardando_entrada`` — o agente pediu informação (``INPUT_REQUIRED``);
* terminais: ``concluido`` (saída válida), ``falhou``, ``rejeitado``,
  ``cancelado``, ``expirado`` (prazo vencido) e ``violado`` (o agente quebrou
  o contrato: saída fora do schema ou protocolo inválido).

Nada sai de um estado terminal: eventos atrasados são registrados e ignorados.
"""

from __future__ import annotations

from ..a2a.protocol import TaskState

PROPOSED = "proposto"
ACTIVE = "ativo"
INPUT_REQUIRED = "aguardando_entrada"
COMPLETED = "concluido"
FAILED = "falhou"
REJECTED = "rejeitado"
CANCELED = "cancelado"
EXPIRED = "expirado"
BREACHED = "violado"

TERMINAL = frozenset({COMPLETED, FAILED, REJECTED, CANCELED, EXPIRED, BREACHED})
OPEN = frozenset({PROPOSED, ACTIVE, INPUT_REQUIRED})
ALL = OPEN | TERMINAL

LABELS = {
    PROPOSED: "proposto",
    ACTIVE: "em andamento",
    INPUT_REQUIRED: "aguardando entrada",
    COMPLETED: "concluído",
    FAILED: "falhou",
    REJECTED: "rejeitado",
    CANCELED: "cancelado",
    EXPIRED: "expirado",
    BREACHED: "violado",
}

_ALLOWED = {
    PROPOSED: {ACTIVE, INPUT_REQUIRED} | TERMINAL,
    ACTIVE: {ACTIVE, INPUT_REQUIRED} | TERMINAL,
    INPUT_REQUIRED: {ACTIVE, INPUT_REQUIRED} | TERMINAL,
}


def can_move(current: str, target: str) -> bool:
    return target in _ALLOWED.get(current, set())


def from_task_state(state: str) -> str | None:
    """Estado do contrato para um estado de tarefa A2A (``None`` = não muda nada)."""
    return {
        TaskState.SUBMITTED: ACTIVE,
        TaskState.WORKING: ACTIVE,
        TaskState.INPUT_REQUIRED: INPUT_REQUIRED,
        TaskState.COMPLETED: COMPLETED,
        TaskState.FAILED: FAILED,
        TaskState.CANCELED: CANCELED,
        TaskState.REJECTED: REJECTED,
        TaskState.AUTH_REQUIRED: FAILED,
    }.get(state)
