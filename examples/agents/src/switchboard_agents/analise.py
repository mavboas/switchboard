"""Agente A2A de exemplo: análise de proposta de crédito (tarefa longa, sob contrato).

Diferente do conector MCP ``credito`` (uma calculadora síncrona), este agente
faz um processo com etapas — consulta, política, cálculo — que leva alguns
segundos (``ANALISE_DELAY_S``, padrão 12 s), publica o progresso e, para
propostas acima de R$ 500 mil, pede uma informação ao usuário no meio do
caminho (``INPUT_REQUIRED``).
"""

from __future__ import annotations

import asyncio
import os

from switchboard_agentkit import ContractAgent, SkillContext, SkillResult

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "cliente": {"type": "string", "minLength": 2, "description": "Nome do cliente"},
        "valor": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": 100_000_000,
            "description": "Valor do crédito solicitado, em reais",
        },
        "prazo_meses": {
            "type": "integer",
            "minimum": 1,
            "maximum": 480,
            "description": "Prazo em meses",
        },
        "renda_mensal": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "Renda mensal do cliente, em reais",
        },
    },
    "required": ["cliente", "valor", "prazo_meses", "renda_mensal"],
    "additionalProperties": False,
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "decisao": {"type": "string", "enum": ["aprovado", "aprovado_com_ajuste", "reprovado"]},
        "limite_aprovado": {"type": "number", "minimum": 0},
        "prazo_meses": {"type": "integer", "minimum": 1},
        "taxa_mensal_percentual": {"type": "number", "minimum": 0},
        "parcela_estimada": {"type": "number", "minimum": 0},
        "comprometimento_renda_percentual": {"type": "number", "minimum": 0},
        "garantia": {"type": "string"},
        "observacoes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "decisao",
        "limite_aprovado",
        "prazo_meses",
        "taxa_mensal_percentual",
        "parcela_estimada",
        "observacoes",
    ],
    "additionalProperties": False,
}

GARANTIA_ACIMA_DE = 500_000
COMPROMETIMENTO_MAXIMO = 35.0


def brl(value: float) -> str:
    text = f"{value:,.2f}"
    return "R$ " + text.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",") + "%"


def parcela_price(valor: float, taxa_percentual: float, prazo: int) -> float:
    i = taxa_percentual / 100
    return valor * i / (1 - (1 + i) ** -prazo)


def valor_presente(parcela: float, taxa_percentual: float, prazo: int) -> float:
    i = taxa_percentual / 100
    return parcela * (1 - (1 + i) ** -prazo) / i


def _sim(text: str) -> bool | None:
    folded = text.strip().lower()
    if folded.startswith(("s", "y")) or "imóvel" in folded or "imovel" in folded:
        return True
    if folded.startswith("n"):
        return False
    return None


def analisar(dados: dict, *, garantia: bool) -> dict:
    valor, prazo, renda = (
        float(dados["valor"]),
        int(dados["prazo_meses"]),
        float(dados["renda_mensal"]),
    )
    taxa = 1.19 if garantia else 1.69
    if prazo > 120:
        taxa += 0.1
    observacoes = []
    parcela = parcela_price(valor, taxa, prazo)
    comprometimento = parcela / renda * 100
    decisao, limite = "aprovado", valor
    if comprometimento > COMPROMETIMENTO_MAXIMO:
        limite = valor_presente(renda * COMPROMETIMENTO_MAXIMO / 100, taxa, prazo)
        if limite < valor * 0.2:
            decisao, limite = "reprovado", 0.0
            observacoes.append(
                f"A parcela comprometeria {pct(comprometimento, 1)} da renda (máximo {pct(COMPROMETIMENTO_MAXIMO, 0)})."
            )
        else:
            decisao = "aprovado_com_ajuste"
            observacoes.append(
                f"Valor ajustado para caber em {pct(COMPROMETIMENTO_MAXIMO, 0)} da renda mensal."
            )
        parcela = parcela_price(limite, taxa, prazo) if limite else 0.0
        comprometimento = parcela / renda * 100
    if garantia:
        observacoes.append("Taxa reduzida pela garantia de imóvel.")
    observacoes.append(
        "Análise demonstrativa (empresa fictícia); sujeita a confirmação documental."
    )
    return {
        "decisao": decisao,
        "limite_aprovado": round(limite, 2),
        "prazo_meses": prazo,
        "taxa_mensal_percentual": round(taxa, 2),
        "parcela_estimada": round(parcela, 2),
        "comprometimento_renda_percentual": round(comprometimento, 1),
        "garantia": "imóvel" if garantia else "sem garantia",
        "observacoes": observacoes,
    }


def build(
    url: str, *, push_hosts: list[str] | None = None, delay_s: float | None = None
) -> ContractAgent:
    delay = float(os.environ.get("ANALISE_DELAY_S", "12")) if delay_s is None else delay_s
    agent = ContractAgent(
        name="analise-credito",
        description="Análise de propostas de crédito da Acme (fictícia): política, capacidade de pagamento, limite e taxa.",
        url=url,
        organization="Acme Serviços Financeiros (demonstração)",
        push_hosts=push_hosts,
    )

    @agent.skill(
        "analisar_proposta",
        name="Analisar proposta de crédito",
        description=(
            "Analisa uma proposta de crédito (cliente, valor, prazo e renda): aplica a política, "
            "calcula o comprometimento de renda e devolve a decisão com limite, taxa e parcela."
        ),
        input_schema=INPUT_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
        max_duration_s=900,
        examples=[
            "Analise um crédito de R$ 80 mil para Maria Souza em 24 meses, renda de R$ 12 mil",
            "Quero a aprovação de um empréstimo de 300 mil em 60 meses para João, que ganha 25 mil por mês",
        ],
        tags=["credito", "analise", "aprovacao"],
    )
    async def analisar_proposta(ctx: SkillContext, dados: dict) -> SkillResult:
        garantia = False
        if float(dados["valor"]) > GARANTIA_ACIMA_DE:
            if not ctx.replies:
                ctx.require_input(
                    f"Propostas acima de {brl(GARANTIA_ACIMA_DE)} exigem garantia real. "
                    "O cliente oferece um imóvel em garantia? (sim/não)"
                )
            garantia = bool(_sim(ctx.replies[-1]))
        steps = [
            "consultando histórico e bureau",
            "aplicando a política de crédito",
            "calculando limite e taxa",
        ]
        for step in steps:
            await ctx.progress(step)
            await asyncio.sleep(delay / len(steps))
        resultado = analisar(dados, garantia=garantia)
        texto = {
            "aprovado": f"Proposta aprovada: {brl(resultado['limite_aprovado'])}",
            "aprovado_com_ajuste": f"Aprovada com ajuste: limite de {brl(resultado['limite_aprovado'])}",
            "reprovado": "Proposta reprovada",
        }[resultado["decisao"]]
        if resultado["parcela_estimada"]:
            texto += (
                f" em {resultado['prazo_meses']} meses a {pct(resultado['taxa_mensal_percentual'])} ao mês "
                f"(parcela de {brl(resultado['parcela_estimada'])})"
            )
        return SkillResult(resultado, text=texto + ".")

    return agent
