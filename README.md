# Switchboard

**Roteador de agentes: responde com RAG, aciona ferramentas via MCP e delega tarefas a outros agentes via A2A, sob contrato.** Como uma mesa telefônica: o Switchboard atende o pedido, responde o que sabe com as bases de conhecimento, resolve na hora o que uma ferramenta resolve e transfere o resto para o ramal certo — um agente que pode demorar, pedir informação no meio do caminho e responder depois.

A cada pedido o roteador escolhe uma rota:

| Rota | Quando | O que acontece |
|---|---|---|
| **Responder** (`direct`) | a resposta está nas bases | responde com os trechos recuperados (RAG), citando as fontes |
| **Tool** (`tool`) | uma ação rápida resolve: calcular, consultar, abrir chamado | chama a tool de um **conector MCP** (`tools/call`) e redige a resposta |
| **Delegar** (`delegated`) | é trabalho de outro agente, que pode levar minutos | abre um **contrato** com o agente A2A, espera um pouco e, se ele não terminar, responde "em andamento" e consolida o resultado quando ele responder |
| **Esclarecer** (`clarify`) | falta um dado obrigatório | faz uma pergunta objetiva em vez de chutar |

**Quem decide:** o que é decisão — qual rota, quais tarefas, se cada parâmetro obrigatório foi informado — vai para o [Jev](https://typesafe.ai) (TypeSafe, System One), que responde com probabilidades em milissegundos; o que é geração — redigir, extrair valores, consolidar — fica com o LLM (System Two). Sem Jev, o LLM decide tudo; sem LLM (modelo `offline`), um decisor heurístico assume.

Tudo é configurável sem mexer em código: **qual LLM** (OpenAI, Anthropic, Azure OpenAI, Gemini, Ollama, OpenRouter, Groq ou qualquer API compatível com OpenAI), **qual modelo de decisão**, **quais conectores MCP**, **quais agentes A2A** e **quais bases de conhecimento** cada roteador usa. A configuração fica num banco (PostgreSQL por padrão) e é editada por um console web; o router aplica mudanças em segundos, sem reiniciar.

![Playground do console: delegação a um agente A2A que termina depois da resposta provisória](docs/img/playground.png)

## Arquitetura

```mermaid
flowchart LR
    U[Usuário ou aplicação] -->|"/v1/chat · /v1/chat/completions · /v1/runs"| R(Router)
    C["Console<br/>UI + API admin"] -->|grava configuração| DB[("PostgreSQL<br/>+ pgvector")]
    R -->|"configuração · RAG · execuções · contratos · spans"| DB
    R -->|"decisão (System One)"| J[["Jev<br/>TypeSafe"]]
    R -->|"texto (System Two)"| L[["LLM<br/>OpenAI · Anthropic · Ollama…"]]
    R -->|"MCP: tools/list · tools/call"| M["Conectores MCP<br/>crédito · chamados"]
    R <-->|"A2A: SendMessage · GetTask · push"| A["Agentes A2A<br/>análise de crédito · risco"]
    C -->|playground| R
```

É um monorepo com [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/):

```
switchboard/
├── packages/core/        switchboard-core — o framework: LLMs, Jev, RAG, conectores MCP, cliente A2A,
│                         contratos, motor de roteamento, spans e storage
├── packages/agentkit/    switchboard-agentkit — para escrever agentes A2A que falam o contrato
├── apps/router/          switchboard-router — API de chat (data plane), compatível com OpenAI
├── apps/console/         switchboard-console — UI e API admin (control plane)
├── examples/agents/      conectores MCP (crédito, chamados) e agentes A2A (análise de crédito, risco) de exemplo
├── examples/knowledge/   documentos da base de demonstração (empresa fictícia)
├── examples/switchboard.yaml   configuração do modo framework (sem banco)
├── scripts/              dev.sh / dev.ps1 (stack sem Docker) e smoke.py (teste ponta a ponta)
├── docs/                 arquitetura, API, contrato A2A e ADRs
├── Dockerfile            imagem única para os serviços Python
└── docker-compose.yml    stack completa
```

Detalhes do fluxo e do modelo de dados em [docs/arquitetura.md](docs/arquitetura.md); endpoints em [docs/api.md](docs/api.md); o contrato com agentes em [docs/a2a-contrato.md](docs/a2a-contrato.md); as decisões de desenho em [docs/adr](docs/adr/0001-decisao-hibrida-e-contratos-a2a.md).

## Subir em dois minutos (Docker)

Pré-requisito: Docker com Compose.

```bash
cp .env.example .env          # no PowerShell: copy .env.example .env
docker compose up -d --build
```

Isso sobe PostgreSQL com pgvector, o console, o router, dois conectores MCP e dois agentes A2A de exemplo. No primeiro start o console cria uma **configuração de demonstração** que roda sem nenhuma chave de API: o modelo `offline`, os conectores `credito` e `chamados`, os agentes `analise-credito` (uma análise que leva ~12 s e, acima de R$ 500 mil, pede uma informação no meio) e `risco`, a base `manual-atendimento` e o roteador `default`.

- Console: <http://localhost:8000> — abra o **Playground** e teste as sugestões.
- Router: <http://localhost:8080/docs> (Swagger).

![Painel do console](docs/img/painel.png)

Uma tool MCP responde na hora:

```bash
curl -s http://localhost:8080/v1/chat -H 'Content-Type: application/json' \
  -d '{"message": "Simule um empréstimo de R$ 50 mil em 24 meses a 1,5% ao mês"}'
```

A resposta traz a rota (`tool`), o conector e a tool (`credito/simular_financiamento`), os argumentos extraídos, as fontes do RAG e os spans de cada etapa. Já uma tarefa delegada a um agente que demora volta como **HTTP 202** com `status: "pending"` e um `run_id`:

```bash
curl -s http://localhost:8080/v1/chat -H 'Content-Type: application/json' \
  -d '{"message": "Analise uma proposta de crédito de 80 mil para Maria Souza em 24 meses com renda de 12 mil"}'
curl -s http://localhost:8080/v1/runs/<run_id>          # ou acompanhe por SSE em /v1/runs/<run_id>/events
```

> O modelo `offline` é um decisor heurístico (palavras-chave + extração de argumentos pelo schema). Serve para demonstração e testes; para conversas de verdade, configure um LLM e, se quiser, o Jev (próximas seções). Com `make` disponível, `make up` sobe a stack e `make smoke` testa todas as rotas. As portas ficam presas a `127.0.0.1`; para expor na rede, veja [Segurança](#segurança).

## Configurando os modelos

1. Coloque a chave no `.env` (ex.: `OPENAI_API_KEY=sk-...`) e rode `docker compose up -d` de novo.
2. No console, **Modelos → Novo modelo**, escolha o atalho do provedor e preencha o campo da chave com `env:OPENAI_API_KEY`.
3. Clique em **Testar** para validar a conexão.
4. Em **Roteadores**, troque o modelo do `default` (ou crie outro roteador). O router pega a mudança em até 5 segundos.

Você também pode colar a chave direto no formulário: ela é cifrada com a chave mestra antes de ir para o banco e nunca volta para a tela. A chave mestra é gerada sozinha na primeira subida (fica no volume `secrets`, compartilhado por console e router) ou vem de `SWITCHBOARD_SECRET_KEY`.

| Atalho | Tipo | URL base |
|---|---|---|
| OpenAI | LLM compatível com OpenAI | `https://api.openai.com/v1` |
| Anthropic | LLM (Messages API) | `https://api.anthropic.com` |
| Azure OpenAI (API v1) | LLM compatível com OpenAI, cabeçalho `api-key` | `https://SEU-RECURSO.openai.azure.com/openai/v1` |
| Google Gemini | LLM compatível com OpenAI | `https://generativelanguage.googleapis.com/v1beta/openai` |
| Ollama | LLM compatível com OpenAI | `http://localhost:11434/v1` (no Docker: `http://host.docker.internal:11434/v1`) |
| OpenRouter, Groq, vLLM, LM Studio… | LLM compatível com OpenAI | a do provedor |
| TypeSafe Jev | **modelo de decisão** (System One) | `https://api.typesafe.ai` |

O adaptador compatível com OpenAI se ajusta sozinho quando o modelo recusa um parâmetro (`max_tokens` → `max_completion_tokens`, `temperature`, `response_format`) e lembra do ajuste. Se o provedor cair, o router decide com o heurístico e registra o aviso no trace.

### Decisão com o Jev

Cadastre o atalho **TypeSafe Jev** com a chave `env:TYPESAFE_API_KEY` (o botão **Testar** faz uma pergunta de verdade ao Jev) e escolha-o como **modelo de decisão** do roteador, ao lado do LLM. A cada pedido, o Jev responde perguntas tipadas:

- **qual capacidade atende** o pedido — uma das tools MCP, uma das skills dos agentes, a base de conhecimento ou "fora do escopo" (`choice`);
- se o pedido **também pede outra tarefa** de agente, para delegar em paralelo (`noul` por skill);
- se os trechos do RAG **respondem** o pedido (`noul`);
- se **cada parâmetro obrigatório** da capacidade escolhida já foi informado (`noul` por parâmetro): o que falta vira pergunta sem gastar LLM, e um valor que o LLM tenha preenchido para um parâmetro não informado é descartado.

Abaixo do **limiar de confiança** do roteador (padrão 0,6), ou com o Jev fora do ar, a decisão sobe para o LLM e o trace registra o porquê. Fixe a versão do modelo (ex.: `jev-1.13.0`) se for calibrar o limiar.

## Conectores MCP e agentes A2A

Conectar uma ferramenta e conversar com outro agente são coisas diferentes, e o Switchboard trata as duas separadamente — no YAML (`connectors` e `agents`), no console (páginas próprias) e no banco.

| | Conector MCP | Agente A2A |
|---|---|---|
| O que é | um servidor MCP: cada **tool** é uma ação rápida | outro agente: cada **skill** é uma tarefa delegável |
| Descoberta | `tools/list` | Agent Card (`/.well-known/agent-card.json`) |
| Chamada | `tools/call`, síncrona, dentro do pedido | `SendMessage` + push notifications / `GetTask`, pode levar minutos |
| Termos | `inputSchema` da tool | **contrato fechado**: schemas de entrada e saída, prazo, estados |
| Pode pedir informação no meio? | não | sim (`INPUT_REQUIRED`), e a resposta volta ao mesmo agente |

### Conectores MCP

Qualquer servidor MCP com transporte Streamable HTTP (ou SSE): em **Conectores MCP → Registrar conector**, informe a URL do endpoint (ex.: `http://meu-conector:9000/mcp`). O console faz o handshake, lista as tools com seus argumentos e deixa você chamar uma tool na hora para testar. Há allowlist de tools por conector e token de acesso opcional (`Authorization: Bearer`).

```python
from mcp.server.mcpserver import MCPServer

server = MCPServer("cambio", instructions="Cotações e conversão de moedas.")


@server.tool()
def converter(valor: float, moeda: str = "usd") -> str:
    """Converte um valor em reais para a moeda informada."""
    taxa = {"usd": 5.0, "eur": 6.0}[moeda]
    return f"R$ {valor:.2f} = {valor / taxa:.2f} {moeda.upper()}"


server.run("streamable-http", host="0.0.0.0", port=9000, stateless_http=True)
```

### Agentes A2A, sob contrato

Qualquer agente [A2A 1.0](https://a2a-protocol.org) (binding JSON-RPC): em **Agentes A2A → Registrar agente**, informe a URL base. O console lê o Agent Card e mostra as skills, se cada uma tem **contrato completo** (o agente declara os schemas de entrada e saída na extensão `urn:switchboard:a2a:contract:v1`) ou **básico** (instrução e resposta em texto) e os agentes que suportam push notifications.

![Agente A2A com os termos do contrato de cada skill](docs/img/agente.png)

Toda delegação abre um contrato: a entrada é validada contra o schema da skill **antes** do envio, o agente confere os termos (hash dos schemas e prazo) e o roteador valida a saída **ao receber** — saída fora do schema deixa o contrato `violado`. Os estados vão de `proposto` e `ativo` (ou `aguardando_entrada`) até `concluido`, `falhou`, `rejeitado`, `cancelado`, `expirado` ou `violado`. O pacote `switchboard-agentkit` implementa o lado do agente sobre o SDK oficial:

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


agent.run(port=8301)
```

O formato da extensão, o que viaja em cada mensagem, a máquina de estados e como testar à mão estão em [docs/a2a-contrato.md](docs/a2a-contrato.md).

### Espera, segundo plano e consolidação

O roteador espera os agentes por até `wait_s` (padrão 8 s, por roteador e por pedido). Se todos terminam a tempo, a resposta consolidada volta na hora. Senão, a API responde `pending` com o `run_id`, os contratos seguem em segundo plano e, quando o último termina, o roteador **consolida** os resultados (com o LLM, ou por modelo de texto no modo offline). O resultado chega por `GET /v1/runs/{id}`, por SSE (`/v1/runs/{id}/events`) ou por webhook (`callback_url`, chamado também quando um agente pede informação). Se um agente pede informação, a execução fica `needs_input` com a pergunta dele; a resposta do usuário, enviada com o mesmo `run_id`, vai para o mesmo agente, na mesma tarefa e sob o mesmo contrato. Um supervisor no router retoma contratos abertos depois de um restart, faz polling de reserva e expira quem passou do prazo (com `CancelTask`).

## Observabilidade

Cada execução é uma árvore de **spans** — `pedido` → `rag`, `descoberta`, `decisao` (com as perguntas ao Jev e as chamadas ao LLM), `tool_mcp` ou `delegacao` → um span `contrato: agente/skill` **por spawn**, `espera`, `consolidacao` e, quando o usuário responde a um agente, `entrada_do_usuario`. O span do contrato fica aberto até o estado final e aponta para o contrato, cuja linha do tempo (aceite, progresso, artefatos, pedidos de entrada) aparece junto na execução.

![Execução com um contrato por spawn e a cascata de spans](docs/img/execucao.png)

No console, **Execuções** mostra a cascata de cada pedido (e se atualiza sozinha enquanto há contratos abertos), **Contratos** lista todos os spawns com estado, termos e prazo, e o painel mostra o que está em andamento. O `trace_id` vai para os agentes no cabeçalho `traceparent` (W3C) e na `metadata` da mensagem A2A, para correlacionar com o tracing do lado deles; os spans também saem em `GET /v1/runs/{id}`.

## Bases de conhecimento (RAG)

Em **Conhecimento**, crie uma base e envie arquivos (`.md`, `.txt`, `.pdf`, `.html`, `.csv`, `.json`…) ou cole texto. Os documentos são quebrados em trechos (títulos Markdown viram seções) e indexados. A tela da base tem uma busca de teste com o score de cada trecho.

- **Embedder `hashing`** (padrão): offline, determinístico, sem chave — bom para começar.
- **Embedder de modelo**: qualquer endpoint `/embeddings` compatível com OpenAI (ex.: `text-embedding-3-small`, `nomic-embed-text` no Ollama), reaproveitando a conexão de um modelo cadastrado. Trocar o embedder pede reindexação (um botão na tela da base).

No PostgreSQL com pgvector a similaridade é calculada pelo banco; sem a extensão (ou no SQLite) os vetores ficam em JSON e a conta é feita na aplicação.

## Usando o router

| Endpoint | Para quê |
|---|---|
| `POST /v1/chat` | API nativa: `{"profile": "default", "message": "..."}` ou `messages` com histórico; devolve resposta, rota, tool ou contratos, fontes e spans. `202` + `run_id` quando a delegação continua em segundo plano |
| `POST /v1/chat/completions` | compatível com OpenAI (inclusive `stream: true`); o campo `model` é o nome do roteador |
| `GET /v1/runs/{id}` · `/events` · `POST /cancel` | acompanha (polling ou SSE) ou cancela uma execução com agentes |
| `GET /v1/models` | lista os roteadores ativos |
| `GET /v1/connectors` · `GET /v1/agents` | conectores MCP e agentes A2A do roteador, com o resultado da descoberta |

Com qualquer SDK da OpenAI:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="qualquer-coisa")
resposta = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Compare Price e SAC para 100 mil em 36 meses a 2% ao mês"}
    ],
)
print(resposta.choices[0].message.content)
```

No endpoint compatível com OpenAI, uma delegação pendente volta com a resposta provisória e o campo extra `switchboard` (`status`, `run_id`, `links`). Para exigir chave na API, defina `SWITCHBOARD_API_KEYS` (lista separada por vírgula) e `SWITCHBOARD_ROUTER_API_KEY` (a chave que o console usa no playground). Todo pedido vira uma execução visível em **Execuções** no console.

## Modo framework (sem console nem banco)

O núcleo funciona como biblioteca, configurado por YAML — veja [examples/switchboard.yaml](examples/switchboard.yaml), que tem os quatro servidores de exemplo e um roteador com Jev:

```bash
uv sync
uv run switchboard -c examples/switchboard.yaml check
uv run switchboard -c examples/switchboard.yaml connectors   # tools dos conectores MCP
uv run switchboard -c examples/switchboard.yaml agents       # skills e contratos dos agentes A2A
uv run switchboard -c examples/switchboard.yaml ask "Qual o horário de atendimento aos sábados?"
SWITCHBOARD_CONFIG_FILE=examples/switchboard.yaml uv run switchboard-router   # router lendo o YAML
```

```python
from switchboard import Switchboard

async with Switchboard.from_yaml("examples/switchboard.yaml") as sb:
    resultado = await sb.ask(
        "Analise uma proposta de crédito de 80 mil para Maria Souza em 24 meses com renda de 12 mil"
    )
    if resultado.status == "pending":  # o agente ainda está trabalhando
        run = await sb.wait(resultado.run_id, 60)
        print(run.status, run.answer)
```

Na biblioteca e na CLI os contratos ficam em memória e são acompanhados por polling; o router lendo o YAML usa push notifications quando `SWITCHBOARD_PUBLIC_URL` está definida.

## Banco de dados

A URL segue o padrão do SQLAlchemy em `SWITCHBOARD_DATABASE_URL`:

- **PostgreSQL** (padrão): `postgresql://usuario:senha@host:5432/banco` — habilita a extensão `vector` sozinho quando tem permissão.
- **SQLite** (desenvolvimento): `sqlite:///./data/switchboard.db`.

As tabelas são criadas na subida e as migrações são aditivas e automáticas: um banco da v0.1 é atualizado na primeira subida da v0.2 (os antigos "agentes" MCP viram conectores, com os vínculos preservados). O acesso a dados passa por uma camada única (`switchboard.storage`), então outros bancos suportados pelo SQLAlchemy entram sem mudar o resto do código.

## Desenvolvimento

Pré-requisito: [uv](https://docs.astral.sh/uv/). O Python 3.12 é baixado pelo uv se preciso.

```bash
uv sync                 # instala todos os pacotes do workspace + ferramentas de dev
./scripts/dev.sh        # sobe conectores, agentes, console e router sem Docker (SQLite em ./data)
                        # no Windows: powershell -ExecutionPolicy Bypass -File scripts/dev.ps1
make smoke              # com a stack no ar: RAG, tool MCP, agentes A2A e pedido de entrada, ponta a ponta
uv run pytest -q        # testes (SQLite)
make test-pg            # inclui PostgreSQL: precisa de um pgvector em localhost:5432
make lint               # ruff
```

A suíte cobre provedores (com transporte HTTP simulado), o cliente do Jev, RAG, conectores MCP com servidores em processo, agentes A2A construídos com o SDK oficial (contratos completos e básicos, push, polling, prazos, violações e pedidos de entrada), o motor de roteamento com um LLM roteirizado, as APIs do router e do console e o storage em SQLite e PostgreSQL. O CI roda lint, testes com PostgreSQL e o `scripts/smoke.py` contra a stack inteira via Docker Compose.

## Segurança

- **Segredos**: chaves de API e tokens são `env:NOME` (lidos do ambiente) ou cifrados no banco com a chave mestra (gerada na primeira subida ou `SWITCHBOARD_SECRET_KEY`); nunca aparecem na UI nem na API admin. Só variáveis liberadas em `SWITCHBOARD_ALLOWED_ENV_SECRETS` (padrão `*_API_KEY,MCP_*,A2A_*` — dê aos tokens nomes como `MCP_CRM_TOKEN` ou `A2A_RISCO_TOKEN`; `SWITCHBOARD_*` nunca) podem ser referenciadas.
- **Exposição**: fora do Docker, console e router escutam só em `127.0.0.1`; o compose publica tudo só em `127.0.0.1`. Com a API do router aberta (sem `SWITCHBOARD_API_KEYS`), ele só atende os hosts de `SWITCHBOARD_ALLOWED_HOSTS`. Para expor, ajuste `CONSOLE_BIND`/`ROUTER_BIND` no `.env` **e** defina `SWITCHBOARD_CONSOLE_PASSWORD`, `SWITCHBOARD_API_KEYS` e `SWITCHBOARD_CONSOLE_ALLOWED_HOSTS`.
- **Agentes A2A**: o endpoint JSON-RPC do Agent Card precisa estar na mesma origem da URL cadastrada (senão, `allow_cross_origin` explícito); cada contrato tem um token de push exclusivo (o banco guarda só o hash) e o endpoint de push recusa token errado; `callback_url` só aceita hosts de `SWITCHBOARD_CALLBACK_HOSTS`; os agentes de exemplo só mandam push para os destinos de `AGENT_PUSH_HOSTS` (padrão: a própria máquina) e o `switchboard-agentkit` pode exigir o token do roteador (`auth_tokens`) no JSON-RPC. Entrada e saída são validadas contra os schemas do contrato dos dois lados.
- **Console**: senha opcional (HTTP Basic) e bloqueio de formulários vindos de outra origem.
- **Router**: chaves de API opcionais; allowlist de tools por conector e de skills por agente; argumentos validados contra o schema antes do `tools/call` ou do `SendMessage`.
- **Resiliência**: conector ou agente lento não trava os pedidos (descoberta com timeout curto e cache servido enquanto revalida); Jev fora → decide o LLM; LLM fora ou resposta inválida → decide o heurístico, com aviso no trace; contratos têm prazo e são retomados depois de um restart.
- Ainda não tem: login com OIDC/SSO, perfis de acesso no console, rate limiting e migrações versionadas com Alembic.

## Roadmap

- Streaming (`SendStreamingMessage`) como alternativa ao push e respostas parciais durante a delegação.
- Exportação dos spans via OpenTelemetry (OTLP) e métricas por rota, agente e modelo.
- Calibração dos limiares do Jev com dados rotulados do domínio.
- Compatibilidade com agentes A2A 0.3 (`message/send`).
- Políticas de acesso por agente/usuário (OPA) e autenticação OIDC no console e no router.
- Migrações com Alembic e índices ANN (HNSW) por dimensão no pgvector.

## Licença

[Apache 2.0](LICENSE).
