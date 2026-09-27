# Switchboard

**Roteador de agentes com RAG e delegação via MCP.** Como uma mesa telefônica: o Switchboard atende o pedido, responde o que sabe com as bases de conhecimento e transfere o resto para o ramal certo — um agente que ele descobre e aciona pelo [Model Context Protocol](https://modelcontextprotocol.io).

A cada pedido o roteador escolhe uma de três rotas:

| Rota | Quando | O que acontece |
|---|---|---|
| **Responder** | o pedido é simples | responde com o conhecimento das bases (RAG), citando as fontes |
| **Delegar** | precisa de cálculo, consulta ou ação em sistema | escolhe o agente e a tool (descobertos via `tools/list`), aciona com `tools/call` e sintetiza a resposta |
| **Esclarecer** | falta um dado obrigatório | faz uma pergunta objetiva em vez de chutar |

Tudo é configurável sem mexer em código: **qual modelo** (OpenAI, Anthropic, Azure OpenAI, Gemini, Ollama, OpenRouter, Groq ou qualquer API compatível com OpenAI), **quais agentes** e **quais bases de conhecimento** cada roteador usa. A configuração fica num banco (PostgreSQL por padrão) e é editada por um console web; o router consome essa configuração e aplica mudanças em segundos, sem reiniciar.

![Playground do console](docs/img/playground.png)

## Arquitetura

```mermaid
flowchart LR
    U[Usuário ou aplicação] -->|"/v1/chat · /v1/chat/completions"| R(Router)
    C["Console<br/>UI + API admin"] -->|grava configuração| DB[("PostgreSQL<br/>+ pgvector")]
    R -->|lê configuração| DB
    R -->|RAG| DB
    R -->|"decisão e síntese"| L[["LLM configurado<br/>OpenAI · Anthropic · Azure · Gemini · Ollama…"]]
    R -->|"MCP: tools/list e tools/call"| A1["Agente MCP<br/>crédito"]
    R -->|MCP| A2["Agente MCP<br/>chamados"]
    C -->|playground| R
```

É um monorepo com [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/):

```
switchboard/
├── packages/core/        switchboard-core — o framework: provedores de LLM, RAG, cliente MCP, motor de roteamento, storage
├── apps/router/          switchboard-router — API de chat (data plane), compatível com OpenAI
├── apps/console/         switchboard-console — monólito com UI e API admin (control plane)
├── examples/agents/      dois agentes MCP de exemplo: simulador de crédito e central de chamados
├── examples/knowledge/   documentos da base de demonstração (empresa fictícia)
├── examples/switchboard.yaml   configuração do modo framework (sem banco)
├── docs/                 arquitetura e referência da API
├── Dockerfile            imagem única para os três serviços Python
└── docker-compose.yml    stack completa: postgres, console, router e agentes
```

Detalhes do fluxo, do contrato de decisão e do modelo de dados em [docs/arquitetura.md](docs/arquitetura.md); endpoints em [docs/api.md](docs/api.md).

## Subir em dois minutos (Docker)

Pré-requisito: Docker com Compose.

```bash
cp .env.example .env          # no PowerShell: copy .env.example .env
docker compose up -d --build
```

Isso sobe PostgreSQL com pgvector, o console, o router e os dois agentes MCP de exemplo. No primeiro start o console cria uma **configuração de demonstração** que roda sem nenhuma chave de API: o modelo `offline`, os agentes `credito` e `chamados`, a base `manual-atendimento` e o roteador `default`.

- Console: <http://localhost:8000> — abra o **Playground** e teste as sugestões.
- Router: <http://localhost:8080/docs> (Swagger).

![Painel do console](docs/img/painel.png)

```bash
curl -s http://localhost:8080/v1/chat -H 'Content-Type: application/json' \
  -d '{"message": "Simule um empréstimo de R$ 50 mil em 24 meses a 1,5% ao mês"}'
```

A resposta traz a rota (`delegated`), o agente e a tool acionados (`credito/simular_financiamento`), os argumentos extraídos, as fontes do RAG e o trace de cada etapa com o tempo gasto.

> O modelo `offline` é um decisor heurístico (palavras-chave + extração de argumentos pelo schema da tool). Serve para demonstração e testes; para conversas de verdade, configure um LLM (próxima seção). Com `make` disponível, `make up` faz o mesmo. As portas ficam presas a `127.0.0.1`; para expor na rede, veja [Segurança](#segurança-estado-do-mvp).

## Configurando um modelo de verdade

1. Coloque a chave no `.env` (ex.: `OPENAI_API_KEY=sk-...`) e rode `docker compose up -d` de novo.
2. No console, **Modelos → Novo modelo**, escolha o atalho do provedor e preencha o campo da chave com `env:OPENAI_API_KEY`.
3. Clique em **Testar** para validar a conexão.
4. Em **Roteadores**, troque o modelo do `default` (ou crie outro roteador). O router pega a mudança em até 5 segundos.

Você também pode colar a chave direto no formulário: ela é cifrada com a chave mestra antes de ir para o banco e nunca volta para a tela. A chave mestra é gerada sozinha na primeira subida (fica no volume `secrets`, compartilhado por console e router) ou vem de `SWITCHBOARD_SECRET_KEY`.

| Atalho | Tipo de API | URL base |
|---|---|---|
| OpenAI | compatível com OpenAI | `https://api.openai.com/v1` |
| Anthropic | Messages API | `https://api.anthropic.com` |
| Azure OpenAI (API v1) | compatível com OpenAI, cabeçalho `api-key` | `https://SEU-RECURSO.openai.azure.com/openai/v1` |
| Google Gemini | compatível com OpenAI | `https://generativelanguage.googleapis.com/v1beta/openai` |
| Ollama | compatível com OpenAI | `http://localhost:11434/v1` (no Docker: `http://host.docker.internal:11434/v1`) |
| OpenRouter, Groq, vLLM, LM Studio… | compatível com OpenAI | a do provedor |

O adaptador compatível com OpenAI se ajusta sozinho quando o modelo recusa um parâmetro (`max_tokens` → `max_completion_tokens`, `temperature`, `response_format`) e lembra do ajuste. Se o provedor cair, o router responde com o decisor offline e registra o aviso no trace.

## Registrando seus agentes (MCP)

Qualquer servidor MCP com transporte Streamable HTTP (ou SSE) vira um agente: em **Agentes MCP → Registrar agente**, informe a URL do endpoint (ex.: `http://meu-agente:9000/mcp`). O console faz o handshake, lista as tools com seus argumentos e deixa você chamar uma tool na hora para testar. Há allowlist de tools por agente e token de acesso opcional (enviado como `Authorization: Bearer`).

![Agente descoberto via MCP](docs/img/agente.png)

Um agente mínimo com o SDK oficial do MCP:

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

O roteador usa a descrição do agente (ou as `instructions` do servidor) e a descrição e o schema de cada tool para decidir quando delegar e como preencher os argumentos. Argumentos são validados contra o `inputSchema` antes da chamada; se falta algo obrigatório, o roteador pergunta ao usuário.

## Bases de conhecimento (RAG)

Em **Conhecimento**, crie uma base e envie arquivos (`.md`, `.txt`, `.pdf`, `.html`, `.csv`, `.json`…) ou cole texto. Os documentos são quebrados em trechos (títulos Markdown viram seções) e indexados. A tela da base tem uma busca de teste com o score de cada trecho.

- **Embedder `hashing`** (padrão): offline, determinístico, sem chave — bom para começar.
- **Embedder de modelo**: qualquer endpoint `/embeddings` compatível com OpenAI (ex.: `text-embedding-3-small`, `nomic-embed-text` no Ollama), reaproveitando a conexão de um modelo cadastrado. Trocar o embedder pede reindexação (um botão na tela da base).

No PostgreSQL com pgvector a similaridade é calculada pelo banco; sem a extensão (ou no SQLite) os vetores ficam em JSON e a conta é feita na aplicação.

## Usando o router

| Endpoint | Para quê |
|---|---|
| `POST /v1/chat` | API nativa: `{"profile": "default", "message": "..."}` ou `messages` com histórico; devolve resposta, rota, agente/tool, fontes e trace |
| `POST /v1/chat/completions` | compatível com OpenAI (inclusive `stream: true`); o campo `model` é o nome do roteador |
| `GET /v1/models` | lista os roteadores ativos |
| `GET /v1/agents?profile=default&refresh=true` | agentes do roteador e o resultado da descoberta via MCP |

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

Para exigir chave na API, defina `SWITCHBOARD_API_KEYS` (lista separada por vírgula) e `SWITCHBOARD_ROUTER_API_KEY` (a chave que o console usa no playground). Todo pedido vira uma execução visível em **Execuções** no console.

## Modo framework (sem console nem banco)

O núcleo funciona como biblioteca, configurado por YAML — veja [examples/switchboard.yaml](examples/switchboard.yaml):

```bash
uv sync
uv run switchboard -c examples/switchboard.yaml ask "Qual o horário de atendimento aos sábados?"
uv run switchboard -c examples/switchboard.yaml agents     # descobre os agentes via MCP
SWITCHBOARD_CONFIG_FILE=examples/switchboard.yaml uv run switchboard-router   # router lendo o YAML
```

```python
from switchboard import Switchboard

async with Switchboard.from_yaml("examples/switchboard.yaml") as sb:
    resultado = await sb.ask("Simule 50 mil em 24 meses a 1,5% ao mês")
    print(resultado.route, resultado.agent, resultado.answer)
```

## Banco de dados

A URL segue o padrão do SQLAlchemy em `SWITCHBOARD_DATABASE_URL`:

- **PostgreSQL** (padrão): `postgresql://usuario:senha@host:5432/banco` — habilita a extensão `vector` sozinho quando tem permissão.
- **SQLite** (desenvolvimento): `sqlite:///./data/switchboard.db`.

As tabelas são criadas na subida. O acesso a dados passa por uma camada única (`switchboard.storage`), então outros bancos suportados pelo SQLAlchemy entram sem mudar o resto do código.

## Desenvolvimento

Pré-requisito: [uv](https://docs.astral.sh/uv/). O Python 3.12 é baixado pelo uv se preciso.

```bash
uv sync                 # instala todos os pacotes do workspace + ferramentas de dev
./scripts/dev.sh        # sobe agentes, console e router sem Docker (SQLite em ./data)
                        # no Windows: powershell -ExecutionPolicy Bypass -File scripts/dev.ps1
uv run pytest -q        # testes (SQLite)
make test-pg            # inclui PostgreSQL: precisa de um pgvector em localhost:5432
make lint               # ruff
```

A suíte cobre provedores (com transporte HTTP simulado), RAG, descoberta e chamada MCP com servidores em processo, o motor de roteamento com um LLM roteirizado, as APIs do router e do console e o storage em SQLite e PostgreSQL. O CI roda lint, testes com PostgreSQL e um smoke test da stack inteira via Docker Compose.

## Segurança (estado do MVP)

- **Segredos**: chaves de API e tokens de agentes são `env:NOME` (lidos do ambiente) ou cifrados no banco com a chave mestra (gerada na primeira subida ou `SWITCHBOARD_SECRET_KEY`); nunca aparecem na UI nem na API admin. Só variáveis liberadas em `SWITCHBOARD_ALLOWED_ENV_SECRETS` (padrão `*_API_KEY,MCP_*` — dê aos tokens de agentes nomes como `MCP_CRM_TOKEN`; `SWITCHBOARD_*` nunca) podem ser referenciadas — quem edita a configuração não consegue ler outras variáveis do processo.
- **Exposição**: fora do Docker, console e router escutam só em `127.0.0.1`; o compose publica console, router e agentes só em `127.0.0.1`. Com a API do router aberta (sem `SWITCHBOARD_API_KEYS`), ele só atende os hosts de `SWITCHBOARD_ALLOWED_HOSTS` (padrão: localhost e o nome `router` do compose). Para expor, ajuste `CONSOLE_BIND`/`ROUTER_BIND` no `.env` **e** defina `SWITCHBOARD_CONSOLE_PASSWORD`, `SWITCHBOARD_API_KEYS` e `SWITCHBOARD_CONSOLE_ALLOWED_HOSTS` (nomes de host aceitos pelo console, contra DNS rebinding).
- **Console**: senha opcional (HTTP Basic) e bloqueio de formulários vindos de outra origem.
- **Router**: chaves de API opcionais; allowlist de tools por agente; argumentos validados contra o schema antes do `tools/call`.
- **Resiliência**: agente lento ou fora do ar não trava os pedidos (descoberta com timeout curto e cache servido enquanto revalida); provedor de LLM fora do ar ou resposta inválida caem para o decisor offline com aviso no trace.
- Ainda não tem: login com OIDC/SSO, perfis de acesso no console, rate limiting e migrações versionadas (Alembic). Estão no roadmap.

## Roadmap

- Streaming token a token do LLM e respostas parciais durante a delegação.
- Delegação em mais de um passo (vários agentes por pedido) e memória de conversa no servidor.
- Observabilidade com OpenTelemetry e métricas por rota, agente e modelo.
- Políticas de acesso por agente/usuário (OPA) e autenticação OIDC no console e no router.
- Migrações com Alembic e índices ANN (HNSW) por dimensão no pgvector.
- Agentes via A2A, além de MCP.

## Licença

[Apache 2.0](LICENSE).
