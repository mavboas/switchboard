# switchboard-core

O framework do Switchboard (pacote `switchboard`): recebe um pedido, busca contexto nas bases de conhecimento, descobre as tools dos conectores MCP e as skills dos agentes A2A e decide entre **responder**, acionar uma **tool**, **delegar** a agentes (sob contrato) ou **esclarecer**.

```python
from switchboard import Switchboard

async with Switchboard.from_yaml("switchboard.yaml") as sb:
    resultado = await sb.ask("Simule 50 mil em 24 meses a 1,5% ao mês")
    print(resultado.route, resultado.agent, resultado.tool, resultado.answer)
    if resultado.status == "pending":  # delegação a um agente A2A ainda em andamento
        run = await sb.wait(resultado.run_id, 60)
        print(run.status, run.answer)
```

Módulos:

| Módulo | Conteúdo |
|---|---|
| `switchboard.llm` | `ChatModel` e adaptadores OpenAI-compatível, Anthropic e offline; presets de provedores |
| `switchboard.jev` | cliente do Jev (TypeSafe System One): perguntas `choice`, `noul` e `score` com probabilidades |
| `switchboard.embeddings` | embedder `hashing` (offline) e compatível com OpenAI |
| `switchboard.rag` | extração de texto, quebra em trechos, busca em memória |
| `switchboard.connectors` | `ConnectorCatalog`: descoberta (`tools/list`) e chamada (`tools/call`) de conectores MCP |
| `switchboard.a2a` | cliente A2A 1.0 (JSON-RPC) e `AgentDirectory`: Agent Card, skills e termos do contrato |
| `switchboard.contracts` | termos, máquina de estados e `ContractManager` (abertura, push, polling, prazos, consolidação) |
| `switchboard.routing` | `RouterEngine`, decisores (Jev, LLM e heurístico), catálogo de capacidades, validação de argumentos |
| `switchboard.tracing` | spans de cada execução (um por spawn de agente) e `traceparent` W3C |
| `switchboard.storage` | SQLAlchemy (PostgreSQL/pgvector ou SQLite), migrações, indexação, execuções, contratos e spans |
| `switchboard.testing` | `ScriptedChat`, `scripted_jev`, `inproc_connector`, `FakeA2AAgent` e transportes em processo para testes sem rede |
| `switchboard.cli` | `switchboard ask | connectors | agents | check` |

Documentação completa no [README do monorepo](../../README.md), em [docs/arquitetura.md](../../docs/arquitetura.md) e em [docs/a2a-contrato.md](../../docs/a2a-contrato.md).
