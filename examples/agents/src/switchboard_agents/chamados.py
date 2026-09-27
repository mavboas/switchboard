"""Agente MCP de exemplo: central de chamados de suporte (em memória)."""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

server = MCPServer(
    "chamados",
    instructions="Abertura, consulta e listagem de chamados de suporte técnico.",
)

Prioridade = Literal["baixa", "media", "alta"]
PRAZOS = {"alta": "4 horas úteis", "media": "1 dia útil", "baixa": "3 dias úteis"}
NOMES = {"alta": "alta", "media": "média", "baixa": "baixa"}


@dataclass
class Chamado:
    protocolo: str
    titulo: str
    descricao: str
    prioridade: str
    status: str = "aberto"
    criado_em: datetime = field(default_factory=lambda: datetime.now(UTC))


_lock = threading.Lock()
_seq = itertools.count(1001)
_chamados: dict[str, Chamado] = {}


def reset() -> None:
    """Limpa o armazenamento (usado nos testes)."""
    global _seq
    with _lock:
        _chamados.clear()
        _seq = itertools.count(1001)


@server.tool(structured_output=False)
def abrir_chamado(
    titulo: Annotated[
        str, Field(description="Resumo curto do problema", min_length=3, max_length=120)
    ],
    descricao: Annotated[
        str, Field(description="Descrição do problema com os detalhes informados", min_length=3)
    ],
    prioridade: Annotated[
        Prioridade, Field(description="Prioridade: baixa, media ou alta")
    ] = "media",
) -> str:
    """Abre um chamado de suporte técnico e devolve o número do protocolo."""
    with _lock:
        protocolo = f"CH-{next(_seq)}"
        _chamados[protocolo] = Chamado(protocolo, titulo.strip(), descricao.strip(), prioridade)
    return (
        f"Chamado {protocolo} aberto com prioridade {NOMES[prioridade]}: {titulo.strip()}. "
        f"Prazo de primeiro atendimento: {PRAZOS[prioridade]}."
    )


@server.tool(structured_output=False)
def consultar_chamado(
    protocolo: Annotated[str, Field(description="Número do protocolo do chamado, ex.: CH-1001")],
) -> str:
    """Consulta o status de um chamado pelo protocolo."""
    chave = protocolo.strip().upper()
    if chave.isdigit():
        chave = f"CH-{chave}"
    chamado = _chamados.get(chave)
    if chamado is None:
        raise ToolError(f"Chamado {chave} não encontrado.")
    return (
        f"Chamado {chamado.protocolo} ({chamado.titulo}) está {chamado.status}, prioridade "
        f"{NOMES[chamado.prioridade]}, aberto em {chamado.criado_em:%d/%m/%Y %H:%M} UTC."
    )


@server.tool(structured_output=False)
def listar_chamados() -> str:
    """Lista os chamados abertos nesta sessão do agente."""
    if not _chamados:
        return "Nenhum chamado aberto até agora."
    linhas = [f"{len(_chamados)} chamado(s):"]
    linhas += [
        f"- {c.protocolo} [{c.status}, {NOMES[c.prioridade]}] {c.titulo}"
        for c in _chamados.values()
    ]
    return "\n".join(linhas)
