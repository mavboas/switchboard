# switchboard-agentkit

Kit para escrever agentes [A2A 1.0](https://a2a-protocol.org) que falam o **contrato fechado** do Switchboard, sobre o SDK oficial (`a2a-sdk`).

- Publica o Agent Card com a extensão `urn:switchboard:a2a:contract:v1`: os schemas de entrada e saída de cada skill e a duração máxima.
- Confere os termos de cada tarefa (skill conhecida, hash dos schemas igual ao publicado, prazo não vencido) e valida a entrada; se algo não bate, rejeita com o motivo.
- Valida a própria saída antes de entregar: o agente falha em vez de entregar algo fora do contrato.
- Suporta progresso, pedidos de entrada no meio da tarefa (`INPUT_REQUIRED`), cancelamento e push notifications (com allowlist de hosts).

```python
from switchboard_agentkit import ContractAgent, SkillContext, SkillResult

agent = ContractAgent(name="cambio", description="Cotações de câmbio.", url="http://localhost:8301")


@agent.skill(
    "cotar",
    name="Cotar câmbio",
    description="Cota a conversão de reais para uma moeda estrangeira.",
    input_schema={
        "type": "object",
        "properties": {"valor": {"type": "number"}, "moeda": {"enum": ["usd", "eur"]}},
        "required": ["valor", "moeda"],
    },
    output_schema={
        "type": "object",
        "properties": {"convertido": {"type": "number"}},
        "required": ["convertido"],
    },
    max_duration_s=120,
)
async def cotar(ctx: SkillContext, dados: dict) -> SkillResult:
    await ctx.progress("consultando a mesa de câmbio")
    if dados["valor"] > 50_000 and not ctx.replies:
        ctx.require_input("Acima de R$ 50 mil a cotação é da mesa. Confirma? (sim/não)")
    convertido = dados["valor"] / {"usd": 5.0, "eur": 6.0}[dados["moeda"]]
    return SkillResult(
        {"convertido": convertido}, text=f"{convertido:.2f} {dados['moeda'].upper()}"
    )


agent.run(port=8301)  # ou app = agent.build_app() (Starlette), para servir como quiser
```

O formato do contrato, a máquina de estados e como testar estão em [docs/a2a-contrato.md](../../docs/a2a-contrato.md). Os agentes `analise-credito` e `risco` em [examples/agents](../../examples/agents/src/switchboard_agents) são exemplos completos.
