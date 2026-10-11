"""Agente A2A de exemplo: prevenção a fraude (avaliação de risco rápida, sob contrato)."""

from __future__ import annotations

import asyncio
import hashlib
import os

from switchboard_agentkit import ContractAgent, SkillContext, SkillResult

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "cliente": {"type": "string", "minLength": 2, "description": "Nome do cliente"},
        "valor": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "Valor da operação, em reais",
        },
        "canal": {
            "type": "string",
            "enum": ["app", "agencia", "internet", "telefone"],
            "description": "Canal da operação",
        },
    },
    "required": ["cliente", "valor"],
    "additionalProperties": False,
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "risco": {"type": "string", "enum": ["baixo", "medio", "alto"]},
        "score": {"type": "integer", "minimum": 0, "maximum": 1000},
        "recomendacao": {"type": "string"},
        "sinais": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["risco", "score", "recomendacao", "sinais"],
    "additionalProperties": False,
}


def avaliar(dados: dict) -> dict:
    """Score determinístico (demonstração): mesmo cliente e valor, mesmo resultado."""
    seed = hashlib.sha256(
        f"{dados['cliente'].strip().lower()}|{float(dados['valor']):.0f}".encode()
    )
    score = 350 + int.from_bytes(seed.digest()[:2], "big") % 600
    sinais = []
    if float(dados["valor"]) > 200_000:
        score -= 120
        sinais.append("valor acima do padrão do segmento")
    if dados.get("canal") == "telefone":
        score -= 60
        sinais.append("canal com maior incidência de engenharia social")
    score = max(0, min(1000, score))
    if score >= 650:
        risco, recomendacao = "baixo", "seguir com a operação"
    elif score >= 450:
        risco, recomendacao = "medio", "confirmar identidade por um segundo fator antes de liberar"
    else:
        risco, recomendacao = "alto", "bloquear e encaminhar para a mesa de prevenção"
    return {"risco": risco, "score": score, "recomendacao": recomendacao, "sinais": sinais}


def build(
    url: str, *, push_hosts: list[str] | None = None, delay_s: float | None = None
) -> ContractAgent:
    delay = float(os.environ.get("RISCO_DELAY_S", "2")) if delay_s is None else delay_s
    agent = ContractAgent(
        name="risco",
        description="Prevenção a fraude da Acme (fictícia): avalia o risco de uma operação ou cliente.",
        url=url,
        organization="Acme Serviços Financeiros (demonstração)",
        push_hosts=push_hosts,
    )

    @agent.skill(
        "avaliar_risco",
        name="Avaliar risco de fraude",
        description="Avalia o risco de fraude de uma operação (cliente, valor e canal) e recomenda o próximo passo.",
        input_schema=INPUT_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
        max_duration_s=120,
        examples=[
            "Verifique o risco de fraude de uma operação de R$ 80 mil para Maria Souza",
            "Essa transação de 5 mil pelo telefone do cliente João é segura?",
        ],
        tags=["fraude", "risco", "seguranca"],
    )
    async def avaliar_risco(ctx: SkillContext, dados: dict) -> SkillResult:
        await ctx.progress("cruzando sinais de dispositivo, histórico e listas restritivas")
        await asyncio.sleep(delay)
        resultado = avaliar(dados)
        texto = (
            f"Risco {resultado['risco']} (score {resultado['score']}): {resultado['recomendacao']}."
        )
        return SkillResult(resultado, text=texto)

    return agent
