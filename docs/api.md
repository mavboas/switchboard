# API

Os dois serviços publicam OpenAPI: router em `http://localhost:8080/docs`, console em `http://localhost:8000/docs`.

## Router (`:8080`)

Autenticação opcional: com `SWITCHBOARD_API_KEYS` definida, envie `Authorization: Bearer <chave>` (ou `X-API-Key`). Sem chaves, a API é para uso local e só atende os hosts de `SWITCHBOARD_ALLOWED_HOSTS`. `/healthz` e `/readyz` ficam sempre abertos; `/a2a/push/{contrato}` é autenticado pelo token do próprio contrato.

### `POST /v1/chat`

```json
{
  "profile": "default",
  "message": "Simule um empréstimo de R$ 50 mil em 24 meses a 1,5% ao mês",
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "wait_s": 8,
  "run_id": null,
  "callback_url": null
}
```

| Campo | Descrição |
|---|---|
| `message` / `messages` | um turno de usuário e/ou o histórico (os dois podem ser combinados: `message` entra no fim) |
| `profile` | roteador (padrão: `SWITCHBOARD_DEFAULT_PROFILE`) |
| `wait_s` | quanto esperar os agentes A2A antes de responder `pending` (padrão: o do roteador; teto `SWITCHBOARD_MAX_WAIT_S`) |
| `run_id` | continua uma execução em `needs_input`: a mensagem vai para o agente que perguntou, na mesma tarefa e sob o mesmo contrato |
| `callback_url` | webhook chamado (POST) com o resultado final de uma execução em segundo plano; só hosts de `SWITCHBOARD_CALLBACK_HOSTS` |

Resposta de uma **tool MCP** (`200`):

```json
{
  "trace_id": "74e87b54611bab518d8b466c812b08ff",
  "run_id": "74e87b54611bab518d8b466c812b08ff",
  "status": "completed",
  "profile": "default",
  "question": "Simule um empréstimo de R$ 50 mil em 24 meses a 1,5% ao mês",
  "answer": "Simulação Price (parcelas fixas): R$ 50.000,00 em 24 meses a 1,50% ao mês.\n- Parcela mensal: R$ 2.496,21…",
  "route": "tool",
  "model": "offline:offline",
  "decided_by": "heuristic",
  "confidence": null,
  "agent": "credito",
  "tool": "simular_financiamento",
  "arguments": {"valor": 50000, "taxa_mensal_percentual": 1.5, "prazo_meses": 24},
  "tasks": [{"kind": "tool", "owner": "credito", "name": "simular_financiamento", "arguments": {"...": "..."}}],
  "contracts": [],
  "reason": "palavras-chave casaram com credito/simular_financiamento (score 3)",
  "sources": [],
  "steps": [{"name": "rag", "duration_ms": 121.7, "detail": {"bases": ["manual-atendimento"], "trechos": 1}}, "…"],
  "spans": [{"id": "ede2adfe56e803f1", "parent_id": null, "kind": "pedido", "name": "pedido", "duration_ms": 191.8, "status": "ok", "attributes": {"rota": "tool"}}, "…"],
  "latency_ms": 191.8,
  "usage": {"input_tokens": 0, "output_tokens": 0},
  "warnings": [],
  "error": null
}
```

Resposta de uma **delegação** que passou da espera (`202`):

```json
{
  "run_id": "0ba143339a0b4f229178202034526389",
  "status": "pending",
  "route": "delegated",
  "agent": "analise-credito",
  "tool": "analisar_proposta",
  "answer": "Encaminhei seu pedido para analise-credito (Analisar proposta de crédito). Assim que houver resposta, eu consolido o resultado para você.",
  "contracts": [
    {"id": "ctr_093f12f4c111aa4e710c", "agent": "analise-credito", "skill": "analisar_proposta", "kind": "completo",
     "state": "ativo", "deadline_at": "2026-09-28T12:04:23+00:00", "last_message": "consultando histórico e bureau", "error": null}
  ],
  "links": {"run": "/v1/runs/0ba1…", "events": "/v1/runs/0ba1…/events", "cancel": "/v1/runs/0ba1…/cancel"},
  "…": "os demais campos, como acima"
}
```

- `route`: `direct` (RAG), `tool` (conector MCP), `delegated` (agentes A2A), `clarify` ou `error`.
- `status`: `completed`, `pending` (agentes ainda trabalhando), `needs_input` (um agente pediu informação: `answer` traz a pergunta), `consolidating` ou `failed`.
- `decided_by`: `jev`, `llm` ou `heuristic`; `confidence` é a probabilidade da escolha quando quem decidiu foi o Jev.
- `tasks`: o que foi acionado (`kind` `tool` ou `skill`, dono, nome e argumentos). `agent`/`tool`/`arguments` repetem a primeira tarefa, por compatibilidade.
- `contracts`: um resumo por contrato (id, agente, skill, tipo, estado, prazo, última mensagem, erro).
- `sources`: cada item tem `index`, `kb`, `document`, `section`, `chunk_id`, `score` e `snippet`.
- `spans`: a árvore completa da execução (ver [arquitetura](arquitetura.md#spans)); `steps` é um resumo por etapa, mantido por compatibilidade com a v0.1.

### Execuções: `/v1/runs/{run_id}`

| Método e caminho | Descrição |
|---|---|
| `GET /v1/runs/{id}` | a execução completa: estado, resposta, tarefas, spans e os contratos com termos, entrada, saída e eventos |
| `GET /v1/runs/{id}/events` | SSE: `event: run` a cada mudança (estado, resposta ou contratos), `event: done` no fim, `event: timeout` depois de 15 min; comentários `: ping` mantêm a conexão viva |
| `POST /v1/runs/{id}/cancel` | cancela os contratos abertos (`CancelTask` nos agentes) e devolve os cancelados |

Fluxo típico de um cliente:

```bash
RUN=$(curl -s localhost:8080/v1/chat -H 'Content-Type: application/json' \
  -d '{"message": "Analise uma proposta de crédito de 800 mil para João Lima em 120 meses com renda de 60 mil"}' \
  | python3 -c 'import sys, json; print(json.load(sys.stdin)["run_id"])')

curl -N localhost:8080/v1/runs/$RUN/events            # acompanha até needs_input ou done

# o agente perguntou (needs_input): a resposta segue para ele, sob o mesmo contrato
curl -s localhost:8080/v1/chat -H 'Content-Type: application/json' \
  -d "{\"run_id\": \"$RUN\", \"message\": \"sim, um imóvel\"}"
```

### `POST /v1/chat/completions`

Compatível com a API de Chat Completions da OpenAI. `model` é o nome do roteador (aceita `switchboard/<nome>`); `stream: true` devolve SSE (`chat.completion.chunk` e `[DONE]`). A resposta inclui o campo extra `switchboard` com `trace_id`, `run_id`, `status`, `route`, `decided_by`, `agent`, `tool`, `sources`, `contracts` e, numa delegação, `links`. Para retomar uma execução em `needs_input` ou ajustar a espera, envie `"switchboard": {"run_id": "...", "wait_s": 0}` no corpo. Roteador inexistente: HTTP 404 com `{"error": {"code": "model_not_found", ...}}`.

### Outros

- `GET /v1/models` — roteadores ativos no formato de lista de modelos da OpenAI.
- `GET /v1/connectors?profile=default&refresh=true` — conectores MCP do roteador com status, latência, tools e schemas.
- `GET /v1/agents?profile=default&refresh=true` — agentes A2A do roteador com status, versão do protocolo, push, skills e termos do contrato.
- `POST /a2a/push/{contract_id}` — recebe as push notifications dos agentes (`StreamResponse` A2A). Autenticado pelo token exclusivo do contrato no cabeçalho `X-A2A-Notification-Token`; token errado recebe 401.
- `GET /healthz` — processo no ar. `GET /readyz` — banco acessível (503 se não).

## Console — API admin (`:8000/api`)

Mesma configuração da UI, para automação e config-as-code. Com `SWITCHBOARD_CONSOLE_PASSWORD` definida, use HTTP Basic (`admin:<senha>`). Segredos (`api_key`, `auth_token`) são só de escrita: omita para manter o valor atual, use `clear_*: true` para remover.

| Método e caminho | Descrição |
|---|---|
| `GET /api/presets` | atalhos de provedores (inclusive o TypeSafe Jev) |
| `GET·POST /api/models` · `PUT·DELETE /api/models/{id}` | LLMs e modelos de decisão (DELETE devolve 409 se estiver em uso) |
| `GET·POST /api/connectors` · `PUT·DELETE /api/connectors/{id}` | conectores MCP |
| `GET /api/connectors/{id}/discover` | handshake + `tools/list` agora |
| `POST /api/connectors/{id}/call` | chama uma tool: `{"tool": "...", "arguments": {...}}` |
| `GET·POST /api/agents` · `PUT·DELETE /api/agents/{id}` | agentes A2A (`url`, `allowed_skills`, `timeout_s`, `deadline_s`, `push`, `allow_cross_origin`) |
| `GET /api/agents/{id}/discover` | lê o Agent Card agora: skills, tipo de contrato e problemas |
| `GET·POST /api/knowledge-bases` · `PUT·DELETE /api/knowledge-bases/{id}` | bases (`embedding_connection` = nome do modelo) |
| `GET·POST /api/knowledge-bases/{id}/documents` | lista ou indexa `{"title", "content"}` |
| `POST /api/knowledge-bases/{id}/files` | upload multipart (campo `file`) |
| `DELETE /api/knowledge-bases/{id}/documents/{doc}` | remove documento |
| `POST /api/knowledge-bases/{id}/search` | `{"query", "top_k", "min_score"}` |
| `POST /api/knowledge-bases/{id}/reindex` | recalcula vetores |
| `GET·POST /api/profiles` · `PUT·DELETE /api/profiles/{id}` | roteadores, referenciando modelos, conectores, agentes e bases **pelo nome** |
| `GET /api/traces?profile=&route=&status=&limit=&offset=` · `GET /api/traces/{id}` | execuções (o detalhe traz spans e contratos com eventos) |
| `GET /api/contracts?state=&agent=&profile=&limit=&offset=` · `GET /api/contracts/{id}` | contratos (o detalhe traz termos, entrada, saída e a linha do tempo) |

Mandar um servidor MCP para `/api/agents` (com `transport` ou `allowed_tools`) devolve 400 indicando `/api/connectors`.

Exemplo — um roteador com Claude redigindo e o Jev decidindo:

```bash
curl -s -X POST localhost:8000/api/models -H 'Content-Type: application/json' \
  -d '{"name": "claude", "provider": "anthropic", "model": "claude-sonnet-4-5", "api_key": "env:ANTHROPIC_API_KEY"}'

curl -s -X POST localhost:8000/api/models -H 'Content-Type: application/json' \
  -d '{"name": "jev", "provider": "typesafe", "model": "jev-1.13.0", "api_key": "env:TYPESAFE_API_KEY"}'

curl -s -X POST localhost:8000/api/profiles -H 'Content-Type: application/json' -d '{
  "name": "atendimento", "model": "claude", "decision_model": "jev", "decision_threshold": 0.6,
  "connectors": ["credito", "chamados"], "agents": ["analise-credito", "risco"],
  "knowledge_bases": ["manual-atendimento"], "wait_s": 8, "max_parallel": 3, "deadline_s": 600}'
```

## Variáveis de ambiente

| Variável | Serviço | Padrão | Descrição |
|---|---|---|---|
| `SWITCHBOARD_DATABASE_URL` | ambos | `postgresql://switchboard:switchboard@localhost:5432/switchboard` | banco (PostgreSQL ou SQLite) |
| `SWITCHBOARD_SECRET_KEY` | ambos | vazio | chave mestra que cifra segredos no banco (mesma nos dois) |
| `SWITCHBOARD_SECRET_KEY_FILE` | ambos | vazio | sem a variável acima, a chave mestra é lida deste arquivo (e gerada nele na primeira subida) |
| `SWITCHBOARD_ALLOWED_ENV_SECRETS` | ambos | `*_API_KEY,MCP_*,A2A_*` | padrões de variáveis aceitas em `env:NOME` (`SWITCHBOARD_*` nunca) |
| `SWITCHBOARD_VECTOR_BACKEND` | ambos | `auto` | `auto`, `pgvector` ou `json` |
| `SWITCHBOARD_LOG_LEVEL` | ambos | `info` | nível de log |
| `SWITCHBOARD_API_KEYS` | router | vazio | chaves aceitas (vírgula); vazio = aberta (uso local) |
| `SWITCHBOARD_ALLOWED_HOSTS` | router | `localhost,127.0.0.1,[::1],router` | com a API aberta, só estes hosts são atendidos (`*` libera); o host de `SWITCHBOARD_PUBLIC_URL` entra sozinho |
| `SWITCHBOARD_DEFAULT_PROFILE` | router | `default` | roteador quando o pedido não informa |
| `SWITCHBOARD_CONFIG_TTL_S` | router | `5` | segundos de cache da configuração |
| `SWITCHBOARD_AGENTS_TTL_S` | router | `30` | segundos de cache da descoberta (tools MCP e Agent Cards) |
| `SWITCHBOARD_PUBLIC_URL` | router | vazio | URL pela qual os agentes A2A alcançam o router (push notifications); vazio = contratos só por polling |
| `SWITCHBOARD_MAX_WAIT_S` | router | `60` | teto para o `wait_s` pedido pelo cliente |
| `SWITCHBOARD_SUPERVISOR_TICK_S` | router | `1` | frequência do supervisor de contratos (polling e prazos) |
| `SWITCHBOARD_CALLBACK_HOSTS` | router | vazio | hosts aceitos em `callback_url`; vazio = callbacks desligados |
| `SWITCHBOARD_MAX_PUSH_BYTES` | router | `1000000` | tamanho máximo de uma push notification |
| `SWITCHBOARD_CONFIG_FILE` | router | vazio | YAML do modo framework (dispensa o banco) |
| `SWITCHBOARD_PORT` / `SWITCHBOARD_HOST` | ambos | `8080`/`8000`, `127.0.0.1` (`0.0.0.0` na imagem Docker) | onde escutar |
| `SWITCHBOARD_ROUTER_URL` | console | `http://localhost:8080` | router usado pelo playground |
| `SWITCHBOARD_ROUTER_PUBLIC_URL` | console | = `ROUTER_URL` | URL mostrada nos exemplos |
| `SWITCHBOARD_ROUTER_API_KEY` | console | vazio | chave do router para o playground |
| `SWITCHBOARD_CONSOLE_USER` / `_PASSWORD` | console | `admin` / vazio | HTTP Basic; sem senha = aberto |
| `SWITCHBOARD_CONSOLE_ALLOWED_HOSTS` | console | `localhost,127.0.0.1,[::1]` | nomes de host aceitos (`*.dominio` e `*` valem) |
| `SWITCHBOARD_SEED_DEMO` | console | `true` | cria a demonstração num banco vazio |
| `SWITCHBOARD_DEMO_CREDITO_URL` / `_CHAMADOS_URL` | console | `http://localhost:8101/mcp` / `:8102` | conectores MCP da demonstração |
| `SWITCHBOARD_DEMO_ANALISE_URL` / `_RISCO_URL` | console | `http://localhost:8201` / `:8202` | agentes A2A da demonstração |
| `SWITCHBOARD_MAX_UPLOAD_MB` | console | `20` | limite por arquivo enviado |
| `TYPESAFE_API_KEY`, `OPENAI_API_KEY`… | ambos | vazio | chaves dos provedores, referenciadas como `env:NOME` no cadastro dos modelos |

Agentes de exemplo (`switchboard-agent`): `AGENT_PUBLIC_URL` (URL que vai no Agent Card), `AGENT_PUSH_HOSTS` (hosts aceitos nas URLs de push, separados por vírgula), `ANALISE_DELAY_S` (padrão 12) e `RISCO_DELAY_S` (padrão 2).

No `docker-compose.yml` também valem `CONSOLE_BIND`, `ROUTER_BIND` e `AGENTS_BIND` (padrão `127.0.0.1`) e `CONSOLE_PORT`/`ROUTER_PORT`.
