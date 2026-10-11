# ADR 0002 — Plugins declarativos: o roteador recebe pacotes de capacidades e de orientação, nunca código

- **Status:** aceita
- **Data:** 2026-10-02
- **Relaciona-se com:** [ADR 0001](0001-decisao-hibrida-e-contratos-a2a.md) (decisão híbrida e contratos). Nada do que ela decidiu muda: o plugin entra *antes* da decisão (o que existe e o que se sabe sobre isso) e passa *pelos mesmos* contratos, schemas e allowlists depois dela.

## Contexto

O Switchboard é um roteador: cada pedido é respondido, vira uma tool MCP, é delegado a um agente A2A sob contrato ou volta como pergunta. A pergunta desta decisão é se **poder subir plugins** ajuda esse papel, sem transformá-lo em outra coisa. O ponto de partida foi o modelo de plugins do Claude (um pacote versionado com skills e servidores MCP, instalável com configuração do usuário), e a restrição foi explícita: **não pivotar a ideia do roteador**. O Switchboard continua sendo quem decide e contrata; ele não vira um plugin nem um servidor MCP do Claude.

Hoje, colocar um domínio novo (crédito, chamados, RH) no roteador custa quatro passos manuais, sem unidade de distribuição:

1. **Um domínio é um conjunto espalhado.** Conector MCP, agente A2A, base de conhecimento e a explicação de como usá-los são cadastrados separadamente e ligados roteador a roteador. Não há um "pacote" para levar de um ambiente a outro nem uma versão para apontar.
2. **O "como usar" vive no `system_prompt`.** Ele só cresce, vale para todo pedido (custo de tokens e distração) e não tem dono: quando o conector sai, a orientação sobre ele continua lá.
3. **Atualizar é às cegas.** Sem prévia do que muda, sem histórico de quem mudou, sem saber se a versão nova se comporta melhor ou pior que a anterior.
4. **Texto e endereços de terceiros entram sem triagem.** Quem adota a integração de um parceiro copia o que ele mandou direto para o prompt e para a lista de conectores.

Do modelo de plugins do Claude servem a um roteador o pacote versionado, as skills no formato aberto Agent Skills, o MCP remoto e a configuração do usuário. Não servem hooks, scripts, comandos de slash e subagentes: o roteador é um serviço de longa duração que guarda segredos e fala com terceiros, e executar código de um pacote ali é exatamente o risco a evitar.

## Decisões

### 1. Plugin é um pacote declarativo: dado, nunca código

Um `.zip` (ou pasta) com `plugin.yaml` (`switchboard.plugin/v1`) e arquivos de texto. O manifesto declara **conectores MCP**, **agentes A2A**, **bases de conhecimento** (documentos do próprio pacote, indexados na instalação), **playbooks** e **frases de exemplo** de roteamento. Só entram extensões de texto (`.yaml`, `.json`, `.md`, `.txt`, `.csv`, `.rst`, `.html`…); script ou binário recusa o pacote. O pacote é lido em memória, com limites (arquivo comprimido, total, por arquivo, número de arquivos e razão de compressão), sem nunca ir para o disco.

O que um plugin aciona (tools, skills, agentes) passa pelas mesmas allowlists, validações de schema e contratos fechados do ADR 0001. Ele **soma** capacidades; não cria um caminho novo de execução.

### 2. Playbooks são Agent Skills, escolhidos por relevância e dentro de um orçamento

A orientação de uso é uma skill no formato aberto (`skills/<nome>/SKILL.md`, com `name` e `description`), o que a torna portátil para outras ferramentas e legível para quem escreve. O roteador **não** põe tudo no prompt:

- **na decisão e na resposta direta**, entram os playbooks relevantes ao pedido (e ao turno anterior): gatilho do manifesto, palavras do nome e da descrição da skill, com pontuação mínima e no máximo três por pedido;
- **na execução** (argumentos, síntese, consolidação), entram os playbooks que cobrem o componente escolhido (`applies_to`);
- `always: true` existe, mas é retido pela varredura para aprovação, porque é o caso em que um pacote de terceiros fala em todo pedido.

O texto entra sempre **sob um cabeçalho** que o identifica como de terceiros e o proíbe de mudar as regras do roteador, dentro de um orçamento por playbook e por prompt, e **nunca no lugar do `system_prompt`**. A seleção é lexical e determinística (testável, grátis, funciona sem LLM); embeddings ou o Jev podem substituí-la depois sem mudar o formato do pacote.

### 3. Endereços e segredos são de quem instala

O pacote declara chaves em `config` e as usa como `${config.CHAVE}` em `url` e `auth_token`. Token escrito no manifesto bloqueia o pacote; só uma chave marcada `sensitive` alimenta `auth_token`, e nunca tem `default`. Os valores sensíveis são cifrados como todo segredo do Switchboard (`enc:`/`env:`) e nunca voltam por tela nem API. O mesmo `.zip` serve em ambientes diferentes, e a configuração sobrevive às atualizações.

### 4. Componentes pertencem ao plugin: `plugin.componente`

Conectores, agentes e bases de um plugin ganham o nome `<plugin>.<componente>`. O **ponto é reservado**: um cadastro manual não pode usá-lo, então nada manual colide com um plugin nem se passa por ele. Os componentes ficam somente leitura (409 na API) e mudam só pela configuração, por uma nova versão ou ao remover o plugin. Liga-se o **plugin** ao roteador (`profile_plugins`), não os itens um a um, o que torna instalar, atualizar e remover operações atômicas. O nome do próprio plugin não pode começar por `switchboard`, `anthropic`, `claude`, `typesafe` ou `jev`, nem ser um nome de sistema (`core`, `admin`, `router`…).

### 5. Varredura em quatro níveis; a aprovação segue o achado

Todo pacote passa por regras que geram achados com nível: **bloqueia** (pacote recusado: caminho que escapa do zip, bomba de compressão, segredo no texto, credencial no endereço ou na query, endpoint de metadados, Unicode de direção do texto ou de tags), **exige aprovação** (texto que dá ordens ao roteador, endereço de rede interna, host público fixo, texto escondido, como caracteres de formato invisíveis e comentários HTML), **aviso** e **informação**. Quem opera revisa e aprova no console (ou aceita por código, no YAML). A aprovação é guardada pela **impressão digital do achado** (código, arquivo e hash do trecho), então uma versão nova não repete o que já foi lido e muda de mão só o que mudou.

A varredura é **triagem, não prova**: um texto malicioso bem escrito pode passar. A barreira real é o desenho (sem código, endereços e tokens de quem instala, texto sob cabeçalho e orçamento, tudo passando pelos contratos). A tela e a documentação dizem isso, para que "sem achados" não seja lido como "seguro".

### 6. Um plugin só entra no roteamento quando está pronto

Pronto é **ligado, sem achado pendente de aprovação, com a configuração obrigatória completa** e ligado ao roteador. Um roteador ligado a um plugin que ainda não está pronto segue funcionando sem ele e diz o motivo em `warnings` (resposta e trace): falhar o pedido por causa de um plugin esquecido seria pior do que atender com o que existe.

### 7. Ciclo de vida versionado e auditável

- O pacote guardado é a forma **normalizada e determinística** (`bundle.pack()`: mesma ordem e datas, mesmos bytes), com o `sha256` do conteúdo como identidade da versão.
- A prévia **valida sem gravar** e diz se é primeira instalação, atualização (quantos achados aprovados continuam valendo), repetição ou downgrade; downgrade exige confirmação.
- Remover apaga o plugin e seus componentes, e **mantém** o histórico (`plugin_events`) e o uso (`plugin_usage`), para que a auditoria sobreviva ao que ela audita. Contratos em aberto com agentes do plugin pedem confirmação (`force`).

### 8. Uso por plugin **e por versão**

Cada pedido registra o que veio de plugins: o playbook que entrou, a tool ou skill acionada (com o resultado) e a base citada. A execução mostra isso (`plugins` em `GET /v1/runs/{id}`, cartão no console), e a página do plugin compara versões, que era o que faltava para saber se a atualização melhorou. O router publica o que cada roteador recebeu em `GET /v1/plugins`.

### 9. Dois modos, a mesma regra

Com banco, a aprovação é um ato humano no console. No modo framework (YAML) não há console, então um achado retido **interrompe o carregamento** até ser aceito por código (`allow: [código]` no plugin), em vez de ser aprovado em silêncio. O `check` e o CI validam os plugins do arquivo.

### 10. Importação do formato Claude, só do declarativo

Um pacote com `.claude-plugin/plugin.json` é convertido na hora: `skills/` viram playbooks, servidores MCP **remotos** viram conectores e `userConfig` vira `config`. O resto (servidores locais por stdio, hooks, comandos, subagentes, scripts) **não entra**, e cada omissão é um achado que explica por quê. O que fica instalado é sempre o pacote nativo gerado, de modo que reler não passa pelo conversor. Assim se reaproveita o ecossistema de skills sem trazer execução de código.

## Consequências

- **Positivas:** um domínio vira uma unidade versionada que se instala, configura, atualiza e remove de uma vez; a orientação só gasta prompt nos pedidos a que se refere; atualizações têm prévia, histórico e comparação de uso por versão; a superfície de risco é pequena por construção.
- **Negativas / custos:** o router passa a receber arquivos de terceiros (por isso os limites e a leitura em memória); há um scanner para manter, com falsos positivos que geram trabalho de revisão e falsos negativos que a documentação assume; texto de terceiros entra no prompt do LLM, o que **reduz mas não elimina** o risco de injeção indireta; a seleção lexical de playbooks erra com sinônimos (a saída é escrever bons `triggers`); mais estado no banco (`plugins`, `plugin_events`, `plugin_usage`, `profile_plugins` e `plugin_id` nas tabelas de componentes).
- **Compatibilidade:** sem plugins, nada muda. As tabelas e colunas novas são aditivas e criadas na subida. A regra de nomes dos cadastros manuais nunca aceitou ponto, então reservá-lo não afeta nada que já exista.

## Alternativas consideradas

- **Fazer do Switchboard um plugin ou servidor MCP do Claude:** é outro produto (o roteador viraria ferramenta de outro agente) e foi descartado pela restrição de não pivotar.
- **Plugins com código** (scripts, hooks, entrypoints Python): máxima flexibilidade, mas é executar código de terceiros num serviço que guarda segredos e alcança a rede interna; exigiria sandbox, assinatura e revisão de cada versão. Fica fora até haver essa infraestrutura.
- **Exportar e importar a configuração (JSON/YAML):** resolve a distribuição, mas sem playbooks, varredura, versão nem uso; é um backup, não um pacote.
- **Playbooks sempre no prompt:** mais simples, mas repete o problema do `system_prompt` que só cresce e dá a terceiros voz em todo pedido.
- **Selecionar playbooks com LLM ou embeddings:** mais recall, porém com custo, latência e variação por chamada, e inútil no modo offline. Em aberto como melhoria.
- **Varredura por LLM:** o scanner seria ele mesmo alvo de injeção e não é determinístico. Pode complementar como revisão sugerida, nunca como portão.
- **Catálogo (marketplace) desde já:** sem assinatura de pacotes e sem confiança no publicador, um catálogo só facilita a distribuição do que ainda não sabemos verificar.

## Em aberto

- Assinatura de pacotes e catálogo, com a política de confiança que isso exige.
- Seleção de playbooks por embeddings ou pelo Jev (`score`), mantendo o formato do pacote.
- Carregar `references/` e `assets/` das skills sob demanda (hoje são aceitos e varridos, mas não carregados).
- Dependências entre plugins e atualização automática por canal.
- Qualidade além do uso: ligar o desfecho dos contratos e o retorno do usuário à versão do plugin.
- Limites por roteador (por exemplo, quantos plugins e quantos caracteres de playbook).
