"""Agente MCP de exemplo: simulador de crédito (tabelas Price e SAC).

É um servidor MCP comum: o Switchboard descobre as tools via ``tools/list`` e
aciona via ``tools/call`` quando o pedido do usuário for uma simulação.
"""

from __future__ import annotations

from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

server = MCPServer(
    "credito",
    instructions="Simulações de financiamento e empréstimo: parcelas, juros totais e comparação Price x SAC.",
)

Valor = Annotated[float, Field(description="Valor financiado em reais", gt=0, le=100_000_000)]
Taxa = Annotated[
    float,
    Field(
        description="Taxa de juros mensal em percentual (ex.: 1.5 para 1,5% ao mês)", gt=0, le=20
    ),
]
Prazo = Annotated[int, Field(description="Prazo em meses", ge=1, le=480)]


def brl(value: float) -> str:
    text = f"{value:,.2f}"
    return "R$ " + text.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(value: float) -> str:
    return f"{value:.2f}".replace(".", ",") + "%"


def price(valor: float, taxa_percentual: float, prazo: int) -> dict[str, float]:
    i = taxa_percentual / 100
    parcela = valor * i / (1 - (1 + i) ** -prazo)
    total = parcela * prazo
    return {"primeira": parcela, "ultima": parcela, "total": total, "juros": total - valor}


def sac(valor: float, taxa_percentual: float, prazo: int) -> dict[str, float]:
    i = taxa_percentual / 100
    amortizacao = valor / prazo
    juros = i * amortizacao * prazo * (prazo + 1) / 2
    return {
        "primeira": amortizacao + valor * i,
        "ultima": amortizacao * (1 + i),
        "total": valor + juros,
        "juros": juros,
    }


@server.tool(structured_output=False)
def simular_financiamento(
    valor: Valor,
    taxa_mensal_percentual: Taxa,
    prazo_meses: Prazo,
    sistema: Annotated[
        Literal["price", "sac"],
        Field(description="Sistema de amortização: price (parcelas fixas) ou sac"),
    ] = "price",
) -> str:
    """Simula um financiamento ou empréstimo e calcula parcelas, total pago e juros."""
    calc = price if sistema == "price" else sac
    r = calc(valor, taxa_mensal_percentual, prazo_meses)
    nome = "Price (parcelas fixas)" if sistema == "price" else "SAC (parcelas decrescentes)"
    linhas = [
        f"Simulação {nome}: {brl(valor)} em {prazo_meses} meses a {pct(taxa_mensal_percentual)} ao mês.",
    ]
    if sistema == "price":
        linhas.append(f"- Parcela mensal: {brl(r['primeira'])}")
    else:
        linhas.append(f"- Primeira parcela: {brl(r['primeira'])}")
        linhas.append(f"- Última parcela: {brl(r['ultima'])}")
    linhas += [
        f"- Total pago: {brl(r['total'])}",
        f"- Juros totais: {brl(r['juros'])}",
        "Valores estimados, sem IOF, seguros ou tarifas.",
    ]
    return "\n".join(linhas)


@server.tool(structured_output=False)
def comparar_sistemas(valor: Valor, taxa_mensal_percentual: Taxa, prazo_meses: Prazo) -> str:
    """Compara as tabelas Price e SAC para o mesmo valor, taxa e prazo."""
    p = price(valor, taxa_mensal_percentual, prazo_meses)
    s = sac(valor, taxa_mensal_percentual, prazo_meses)
    economia = p["juros"] - s["juros"]
    return "\n".join(
        [
            f"Comparação para {brl(valor)} em {prazo_meses} meses a {pct(taxa_mensal_percentual)} ao mês:",
            f"- Price: parcela fixa de {brl(p['primeira'])}; juros totais {brl(p['juros'])}.",
            f"- SAC: começa em {brl(s['primeira'])} e termina em {brl(s['ultima'])}; juros totais {brl(s['juros'])}.",
            f"- O SAC economiza {brl(economia)} em juros, mas exige parcelas iniciais maiores.",
        ]
    )
