# switchboard-core

O framework do Switchboard (pacote `switchboard`): recebe um pedido, busca contexto nas bases de conhecimento, descobre agentes via MCP e decide entre **responder**, **delegar** a uma tool de agente ou **esclarecer**.

```python
from switchboard import Switchboard

async with Switchboard.from_yaml("switchboard.yaml") as sb:
    resultado = await sb.ask("Simule 50 mil em 24 meses a 1,5% ao mês")
    print(resultado.route, resultado.agent, resultado.tool, resultado.answer)
```

Módulos:

| Módulo | Conteúdo |
|---|---|
| `switchboard.llm` | `ChatModel` e adaptadores OpenAI-compatível, Anthropic e offline; presets de provedores |
| `switchboard.embeddings` | embedder `hashing` (offline) e compatível com OpenAI |
| `switchboard.rag` | extração de texto, quebra em trechos, busca em memória |
| `switchboard.agents` | `AgentCatalog`: descoberta (`tools/list`) e chamada (`tools/call`) via MCP |
| `switchboard.routing` | `RouterEngine`, decisores (LLM e heurístico), validação de argumentos |
| `switchboard.storage` | SQLAlchemy (PostgreSQL/pgvector ou SQLite), indexação e busca no banco, traces |
| `switchboard.testing` | `ScriptedChat` e `inproc_connector` para testes sem rede |
| `switchboard.cli` | `switchboard ask | agents | check` |

Documentação completa no [README do monorepo](../../README.md) e em [docs/arquitetura.md](../../docs/arquitetura.md).
