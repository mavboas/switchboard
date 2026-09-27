# API

Os dois serviços publicam OpenAPI: router em `http://localhost:8080/docs`, console em `http://localhost:8000/docs`.

## Router (`:8080`)

Autenticação opcional: com `SWITCHBOARD_API_KEYS` definida, envie `Authorization: Bearer <chave>` (ou `X-API-Key`). Sem chaves, a API é para uso local e só atende os hosts de `SWITCHBOARD_ALLOWED_HOSTS`. `/healthz` e `/readyz` ficam sempre abertos.

### `POST /v1/chat`

```json
{
  "profile": "default",
  "message": "Simule um empréstimo de R$ 50 mil em 24 meses a 1,5% ao mês",
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
}
```

`message` é atalho para um turno de usuário; `messages` leva o histórico (os dois podem ser combinados: `message` entra no fim). `profile` é opcional (padrão: `SWITCHBOARD_DEFAULT_PROFILE`).

Resposta:

```json
{
  "trace_id": "5c0f…",
  "profile": "default",
  "question": "Simule um empréstimo de R$ 50 mil em 24 meses a 1,5% ao mês",
  "answer": "Simulação Price (parcelas fixas): R$ 50.000,00 em 24 meses a 1,50% ao mês…",
  "route": "delegated",
  "model": "offline:offline",
  "agent": "credito",
  "tool": "simular_financiamento",
  "arguments": {"valor": 50000.0, "prazo_meses": 24, "taxa_mensal_percentual": 1.5},
  "reason": "palavras-chave casaram com credito/simular_financiamento (score 3)",
  "sources": [],
  "steps": [
    {"name": "rag", "duration_ms": 5.1, "detail": {"bases": ["manual-atendimento"], "trechos": 2, "scores": [0.31, 0.22]}},
    {"name": "descoberta_mcp", "duration_ms": 0.1, "detail": {"online": {"credito": ["simular_financiamento", "comparar_sistemas"]}, "offline": {}}},
    {"name": "decisao", "duration_ms": 0.4, "detail": {"acao": "delegate", "agente": "credito", "tool": "simular_financiamento"}},
    {"name": "delegacao_mcp", "duration_ms": 61.8, "detail": {"erro": false, "resultado": "Simulação Price…"}},
    {"name": "sintese", "duration_ms": 0.0, "detail": {"llm": false}}
  ],
  "latency_ms": 68.2,
  "usage": {"input_tokens": 0, "output_tokens": 0},
  "warnings": [],
  "error": null
}
```

`route` é `direct`, `delegated`, `clarify` ou `error`. Em `sources`, cada item tem `index`, `kb`, `document`, `section`, `score` e `snippet`.

### `POST /v1/chat/completions`

Compatível com a API de Chat Completions da OpenAI. `model` é o nome do roteador (aceita `switchboard/<nome>`); `stream: true` devolve SSE (`chat.completion.chunk` e `[DONE]`). A resposta inclui o campo extra `switchboard` com `trace_id`, `route`, `agent`, `tool` e `sources`. Roteador inexistente: HTTP 404 com `{"error": {"code": "model_not_found", ...}}`.

### Outros

- `GET /v1/models` — roteadores ativos no formato de lista de modelos da OpenAI.
- `GET /v1/agents?profile=default&refresh=true` — agentes do roteador com status, latência, tools e schemas.
- `GET /healthz` — processo no ar. `GET /readyz` — banco acessível (503 se não).

## Console — API admin (`:8000/api`)

Mesma configuração da UI, para automação e config-as-code. Com `SWITCHBOARD_CONSOLE_PASSWORD` definida, use HTTP Basic (`admin:<senha>`). Segredos (`api_key`, `auth_token`) são só de escrita: omita para manter o valor atual, use `clear_*: true` para remover.

| Método e caminho | Descrição |
|---|---|
| `GET /api/presets` | atalhos de provedores |
| `GET·POST /api/models` · `PUT·DELETE /api/models/{id}` | modelos (DELETE devolve 409 se estiver em uso) |
| `GET·POST /api/agents` · `PUT·DELETE /api/agents/{id}` | agentes MCP |
| `GET /api/agents/{id}/discover` | handshake + `tools/list` agora |
| `GET·POST /api/knowledge-bases` · `PUT·DELETE /api/knowledge-bases/{id}` | bases (`embedding_connection` = nome do modelo) |
| `GET·POST /api/knowledge-bases/{id}/documents` | lista ou indexa `{"title", "content"}` |
| `POST /api/knowledge-bases/{id}/files` | upload multipart (campo `file`) |
| `DELETE /api/knowledge-bases/{id}/documents/{doc}` | remove documento |
| `POST /api/knowledge-bases/{id}/search` | `{"query", "top_k", "min_score"}` |
| `POST /api/knowledge-bases/{id}/reindex` | recalcula vetores |
| `GET·POST /api/profiles` · `PUT·DELETE /api/profiles/{id}` | roteadores, referenciando modelo, agentes e bases **pelo nome** |
| `GET /api/traces?profile=&route=&limit=&offset=` · `GET /api/traces/{id}` | execuções |

Exemplo — criar um roteador com Claude:

```bash
curl -s -X POST localhost:8000/api/models -H 'Content-Type: application/json' \
  -d '{"name": "claude", "provider": "anthropic", "model": "claude-sonnet-4-5", "api_key": "env:ANTHROPIC_API_KEY"}'

curl -s -X POST localhost:8000/api/profiles -H 'Content-Type: application/json' \
  -d '{"name": "atendimento", "model": "claude", "agents": ["credito", "chamados"], "knowledge_bases": ["manual-atendimento"]}'
```

## Variáveis de ambiente

| Variável | Serviço | Padrão | Descrição |
|---|---|---|---|
| `SWITCHBOARD_DATABASE_URL` | ambos | `postgresql://switchboard:switchboard@localhost:5432/switchboard` | banco (PostgreSQL ou SQLite) |
| `SWITCHBOARD_SECRET_KEY` | ambos | vazio | chave mestra que cifra segredos no banco (mesma nos dois) |
| `SWITCHBOARD_SECRET_KEY_FILE` | ambos | vazio | sem a variável acima, a chave mestra é lida deste arquivo (e gerada nele na primeira subida) |
| `SWITCHBOARD_ALLOWED_ENV_SECRETS` | ambos | `*_API_KEY,MCP_*` | padrões de variáveis aceitas em `env:NOME` (`SWITCHBOARD_*` nunca) |
| `SWITCHBOARD_VECTOR_BACKEND` | ambos | `auto` | `auto`, `pgvector` ou `json` |
| `SWITCHBOARD_LOG_LEVEL` | ambos | `info` | nível de log |
| `SWITCHBOARD_API_KEYS` | router | vazio | chaves aceitas (vírgula); vazio = aberta (uso local) |
| `SWITCHBOARD_ALLOWED_HOSTS` | router | `localhost,127.0.0.1,[::1],router` | com a API aberta, só estes hosts são atendidos (`*` libera) |
| `SWITCHBOARD_DEFAULT_PROFILE` | router | `default` | roteador quando o pedido não informa |
| `SWITCHBOARD_CONFIG_TTL_S` | router | `5` | segundos de cache da configuração |
| `SWITCHBOARD_AGENTS_TTL_S` | router | `30` | segundos de cache da descoberta MCP |
| `SWITCHBOARD_CONFIG_FILE` | router | vazio | YAML do modo framework (dispensa o banco) |
| `SWITCHBOARD_PORT` / `SWITCHBOARD_HOST` | ambos | `8080`/`8000`, `127.0.0.1` (`0.0.0.0` na imagem Docker) | onde escutar |
| `SWITCHBOARD_ROUTER_URL` | console | `http://localhost:8080` | router usado pelo playground |
| `SWITCHBOARD_ROUTER_PUBLIC_URL` | console | = `ROUTER_URL` | URL mostrada nos exemplos |
| `SWITCHBOARD_ROUTER_API_KEY` | console | vazio | chave do router para o playground |
| `SWITCHBOARD_CONSOLE_USER` / `_PASSWORD` | console | `admin` / vazio | HTTP Basic; sem senha = aberto |
| `SWITCHBOARD_CONSOLE_ALLOWED_HOSTS` | console | `localhost,127.0.0.1,[::1]` | nomes de host aceitos (`*.dominio` e `*` valem) |
| `SWITCHBOARD_SEED_DEMO` | console | `true` | cria a demonstração num banco vazio |
| `SWITCHBOARD_DEMO_CREDITO_URL` / `_CHAMADOS_URL` | console | `http://localhost:8101/mcp` / `:8102` | agentes da demonstração |
| `SWITCHBOARD_MAX_UPLOAD_MB` | console | `20` | limite por arquivo enviado |

No `docker-compose.yml` também valem `CONSOLE_BIND`, `ROUTER_BIND` e `AGENTS_BIND` (padrão `127.0.0.1`) e `CONSOLE_PORT`/`ROUTER_PORT`.
