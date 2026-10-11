# ADR 0001 — Decisão híbrida (Jev + LLM), agentes via A2A com contrato fechado e spawns observáveis

- **Status:** aceita
- **Data:** 2026-09-27
- **Substitui:** o desenho do MVP em que "agente" era um servidor MCP e o LLM decidia tudo

## Contexto

No MVP, o roteador fazia três coisas com o mesmo mecanismo: decidia a rota com o LLM (resposta em JSON), tratava todo "agente" como um servidor MCP e executava a delegação de forma síncrona (`tools/call`) dentro do pedido HTTP. Três problemas apareceram ao evoluir a visão:

1. **Decisão cara e instável.** Escolher entre responder, acionar uma tool, delegar ou esclarecer é um julgamento fechado (a resposta é uma de N opções). Gerar texto com um LLM para isso é lento, caro e varia de uma chamada para outra.
2. **MCP não é comunicação entre agentes.** Conectar ferramentas (MCP) e conversar com outro agente autônomo são coisas diferentes: um agente pode demorar minutos, pedir informação no meio do caminho e responder depois. Com MCP síncrono, o pedido do usuário ficava preso até o timeout.
3. **Sem contrato, sem governança.** A delegação não tinha termos explícitos (o que entra, o que sai, até quando, em que estados) nem registro de cada tarefa delegada.

## Decisões

### 1. Decisão em dois sistemas: Jev (System One) decide, LLM (System Two) gera

- O **Jev** (modelo de decisão da TypeSafe, `POST /v1/systemone`) responde perguntas tipadas — `choice`, `noul` (sim/não com probabilidade) e `score` — com probabilidades calibradas e `confidence`, em dezenas a centenas de milissegundos.
- Tudo o que é **nível de decisão** vai para o Jev:
  - qual capacidade atende o pedido (`choice` entre as tools MCP, as skills dos agentes A2A, `knowledge` e `out_of_scope`), com critérios estruturados (`what` + `examples`);
  - se há mais de uma tarefa independente no pedido (um `noul` por skill de agente → *fan-out*);
  - se os trechos recuperados pelo RAG respondem o pedido (`noul`);
  - se cada parâmetro obrigatório da capacidade escolhida já foi informado (`noul` por parâmetro, padrão "stated" da própria TypeSafe) — o que falta vira pergunta de esclarecimento sem gastar LLM.
- O **LLM** fica com o que é geração: redigir a resposta com RAG, extrair os valores dos argumentos (números, datas, texto livre — o Jev não extrai valores), escrever a síntese de uma tool e consolidar os resultados dos agentes.
- **Portão de confiança:** abaixo de `decision_threshold` (padrão 0,6) ou com o Jev indisponível, a decisão sobe para o LLM (decisor completo, como no MVP), e o trace registra o porquê. Sem LLM, o decisor heurístico continua sendo o último recurso.
- O Jev também atua como **guarda do LLM**: um argumento que o LLM preencheu mas que o Jev diz não ter sido informado (probabilidade < 0,25) é descartado e vira pergunta, em vez de um valor inventado.
- As instruções das perguntas ao Jev são escritas em inglês (idioma em que ele é mais preciso); o estado (a conversa) segue no idioma do usuário.

### 2. MCP só para tools; agentes conversam por A2A v1.0

- **Conectores MCP**: servidores MCP cujas tools são ações rápidas e síncronas (consultar, calcular, abrir chamado). Continuam com descoberta via `tools/list` e chamada via `tools/call`.
- **Agentes A2A**: agentes autônomos descobertos pelo Agent Card (`/.well-known/agent-card.json`) e acionados por JSON-RPC (`SendMessage`, `GetTask`, `CancelTask`) com `A2A-Version: 1.0`. Cada skill do card é uma capacidade delegável.
- A configuração separa os dois conceitos (`connectors` e `agents` no YAML, páginas próprias no console, tabelas `mcp_connectors` e `a2a_agents`). Bancos do MVP são migrados na subida: os antigos "agentes" viram conectores MCP.
- O cliente A2A do core é enxuto (httpx, sem dependência do SDK) e é testado contra um agente construído com o SDK oficial (`a2a-sdk` 1.x), para garantir interoperabilidade.

### 3. Toda delegação a agente é um contrato fechado

Ao delegar, o roteador cria um **contrato** e só conversa com aquele agente, sobre aquela tarefa, através dele:

| Termo | Conteúdo |
|---|---|
| partes | roteador/perfil/execução (chamador) e agente/skill (executor) |
| entrada | argumentos validados contra o `input_schema` da skill **antes** do envio |
| saída | resultado validado contra o `output_schema` **ao receber** |
| versão | hash SHA-256 da forma canônica de cada schema — se o agente mudou o schema, rejeita o contrato |
| prazo | `deadline` absoluto: o `deadline_s` do cadastro do agente, quando definido; senão o menor entre o `max_duration_s` da skill e o `deadline_s` do perfil |
| retorno | push notification com token exclusivo do contrato; polling de reserva |

Estados: `proposto → ativo → (aguardando_entrada ↔ ativo) → concluido | falhou | rejeitado | cancelado | expirado | violado`. Transições fora da máquina são ignoradas e registradas; saída fora do schema vira `violado`.

Os termos viajam como extensão A2A (`urn:switchboard:a2a:contract:v1`): o agente declara os schemas no Agent Card (`capabilities.extensions[].params.skills`) e recebe os termos na `metadata` da mensagem. Agentes A2A que não declaram a extensão continuam utilizáveis com o **contrato básico** (entrada e saída em texto, prazo e estados impostos pelo roteador). O pacote `switchboard-agentkit` implementa o lado do agente (validação da entrada, checagem dos hashes, validação da saída) sobre o SDK oficial.

### 4. Delegação assíncrona: espera curta + segundo plano

- O pedido abre os contratos e espera até `wait_s` (padrão 8 s, configurável por perfil e por pedido). Se todos terminam a tempo, a resposta consolidada volta na hora, como antes.
- Senão, volta `status: "pending"` (HTTP 202 na API nativa; 200 com o campo `switchboard.status` no endpoint compatível com OpenAI) com uma resposta provisória e o `run_id`. O resultado final chega por `GET /v1/runs/{id}`, por SSE (`GET /v1/runs/{id}/events`) ou por webhook (`callback_url`, restrito a hosts liberados e chamado também quando a execução para em `needs_input`).
- Quando o último contrato de uma execução termina (push, polling ou prazo), o roteador **consolida** os resultados com o LLM (ou por modelo de texto, se offline) e grava a resposta final. A consolidação é atômica no banco, então vários processos do router podem receber os eventos.
- Um supervisor por processo retoma contratos abertos depois de reiniciar, faz polling de quem não suporta push (ou ficou em silêncio), expira contratos vencidos com `CancelTask` e revisita periodicamente as execuções abertas (consolidações perdidas ou interrompidas).
- Se o agente pede informação (`TASK_STATE_INPUT_REQUIRED`), a execução fica `needs_input` e a pergunta volta ao usuário; a resposta dele segue para o mesmo agente, na mesma tarefa e sob o mesmo contrato (`run_id` no pedido seguinte).

### 5. Observabilidade: cada spawn é um span

- Cada execução vira uma árvore de **spans** (`pedido` → `rag`, `descoberta`, `decisao` → `jev`/`llm`, `tool_mcp`, `delegacao` → `contrato: agente/skill` (um por spawn), `espera`, `consolidacao`; e `entrada_do_usuario` quando o usuário responde a um agente), com início, fim, status e atributos. O span do contrato fica aberto até o estado terminal e aponta para o contrato, que guarda a linha do tempo dos eventos do agente (aceite, progresso, artefatos, pedidos de entrada).
- O console mostra a cascata (waterfall) de cada execução, a página **Contratos** lista todos os spawns com estado e prazo, e as execuções pendentes se atualizam sozinhas.
- O `trace_id` segue o formato W3C e vai para os agentes no cabeçalho `traceparent` (e na `metadata` do pedido A2A), para correlacionar com o tracing do lado deles.

## Consequências

- **Positivas:** decisões rápidas, baratas e auditáveis (probabilidades no trace); tarefas longas não prendem conexões; cada delegação tem termos verificáveis e histórico; interoperabilidade com qualquer agente A2A.
- **Negativas / custos:** mais estado no banco (execuções, contratos, eventos, spans); um componente em segundo plano no router; o Jev é um serviço pago e externo (sem chave, o roteador segue decidindo com o LLM); o Jev é mais preciso em inglês — mitigado com instruções em inglês e com o portão de confiança.
- **Compatibilidade:** a rota de tool MCP passa a se chamar `tool` (antes `delegated`); `delegated` agora significa delegação a agente A2A. No YAML, servidores MCP saem de `agents` e vão para `connectors`.

## Alternativas consideradas

- **Só LLM para decidir** (como no MVP): mais simples, mas lento e sem probabilidades calibradas para decidir quando escalar.
- **Protocolo próprio entre agentes:** mais simples de implementar, mas isolaria o Switchboard do ecossistema A2A.
- **MCP com tasks assíncronas** (MCP 2025-11): resolveria a espera, mas mantém a confusão entre ferramenta e agente e não traz Agent Card nem skills.
- **Sempre síncrono / sempre assíncrono:** o primeiro trava conexões em tarefas longas; o segundo piora a experiência nas tarefas curtas, que são a maioria.

## Em aberto

- Streaming (`SendStreamingMessage`) como alternativa ao push para clientes que mantêm conexão.
- Exportação dos spans via OpenTelemetry (OTLP).
- Compatibilidade com agentes A2A 0.3 (métodos `message/send` e partes com `kind`).
- Calibração dos limiares do Jev com dados rotulados do domínio.
