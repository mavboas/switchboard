/* Switchboard Console — comportamento das páginas (JS puro, sem dependências). */
(function () {
  "use strict";

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([key, value]) => {
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else if (value !== undefined && value !== null) node.setAttribute(key, value);
    });
    (children || []).forEach((child) => {
      if (child === null || child === undefined) return;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    });
    return node;
  }

  const ROUTES = {
    direct: "Resposta direta",
    tool: "Tool MCP",
    delegated: "Agente A2A",
    clarify: "Esclarecimento",
    error: "Erro",
  };

  const RUN_STATUS = {
    completed: ["concluída", "ok"],
    pending: ["em andamento", "info"],
    consolidating: ["consolidando", "info"],
    needs_input: ["aguardando entrada", "warn"],
    failed: ["falhou", "danger"],
  };

  const CONTRACT_STATES = {
    proposto: ["proposto", ""],
    ativo: ["em andamento", "info"],
    aguardando_entrada: ["aguardando entrada", "warn"],
    concluido: ["concluído", "ok"],
    falhou: ["falhou", "danger"],
    rejeitado: ["rejeitado", "danger"],
    cancelado: ["cancelado", ""],
    expirado: ["expirado", "danger"],
    violado: ["violado", "danger"],
  };

  const OPEN_RUN = ["pending", "consolidating"];

  function routeBadge(route) {
    return el("span", { class: "badge route-" + route }, [el("span", { class: "dot" }), ROUTES[route] || route]);
  }

  function toneBadge(table, key) {
    const [label, tone] = table[key] || [key, ""];
    return el("span", { class: "badge " + tone }, [el("span", { class: "dot" }), label]);
  }

  function ms(value) {
    if (value === null || value === undefined) return "—";
    if (value >= 60000) return (value / 60000).toFixed(1) + " min";
    if (value >= 1000) return (value / 1000).toFixed(1) + " s";
    return Math.round(value) + " ms";
  }

  // spans em árvore, com a posição de cada barra (mesma regra da página da execução)
  function waterfall(spans) {
    const now = Date.now();
    const items = (spans || []).filter((s) => s.started_at);
    if (!items.length) return { rows: [], total: 0 };
    const start = (s) => Date.parse(s.started_at);
    const end = (s) => (s.ended_at ? Date.parse(s.ended_at) : null);
    const ids = new Set(items.map((s) => s.id));
    const t0 = Math.min(...items.map(start));
    const t1 = Math.max(t0, ...items.map((s) => end(s) || now));
    const total = Math.max(t1 - t0, 1);
    const children = new Map();
    items.forEach((s) => {
      const parent = s.parent_id && ids.has(s.parent_id) ? s.parent_id : null;
      if (!children.has(parent)) children.set(parent, []);
      children.get(parent).push(s);
    });
    children.forEach((list) => list.sort((a, b) => start(a) - start(b)));
    const rows = [];
    const seen = new Set();
    function visit(s, depth) {
      if (seen.has(s.id)) return;
      seen.add(s.id);
      const stop = end(s) || now;
      rows.push({
        span: s,
        depth: depth,
        open: !s.ended_at,
        elapsed: Math.max(stop - start(s), 0),
        left: ((start(s) - t0) / total) * 100,
        width: Math.max(((stop - start(s)) / total) * 100, 0.4),
      });
      (children.get(s.id) || []).forEach((child) => visit(child, depth + 1));
    }
    (children.get(null) || []).forEach((root) => visit(root, 0));
    return { rows: rows, total: total };
  }

  function renderWaterfall(spans) {
    const wf = waterfall(spans);
    const box = el("div", { class: "wf compact" });
    wf.rows.forEach((row) => {
      const s = row.span;
      box.appendChild(
        el("details", { class: "wf-row status-" + s.status + " kind-" + s.kind }, [
          el("summary", {}, [
            el("span", { class: "wf-name", style: "padding-left:" + row.depth * 12 + "px", title: s.name }, [
              // o tipo só aparece quando o nome não diz ("entrada_do_usuario" é um pedido)
              s.name.indexOf(s.kind) === 0 ? null : el("span", { class: "wf-kind", text: s.kind }),
              s.name,
            ]),
            el("span", { class: "wf-track" }, [
              el("span", {
                class: "wf-bar" + (row.open ? " open" : ""),
                style: "left:" + row.left.toFixed(2) + "%;width:" + row.width.toFixed(2) + "%",
              }),
            ]),
            el("span", { class: "wf-ms", text: ms(row.elapsed) + (row.open ? " …" : "") }),
          ]),
          el("pre", { class: "block", text: pretty(s.attributes || {}) }),
        ])
      );
    });
    return box;
  }

  function pretty(value) {
    return JSON.stringify(value, null, 2);
  }

  // --- formulário de modelo --------------------------------------------------

  function modelForm() {
    const presets = window.SWITCHBOARD_PRESETS || {};
    const select = document.getElementById("f-preset");
    if (!select) return;
    const provider = document.getElementById("f-provider");
    const baseUrl = document.getElementById("f-base_url");
    const header = document.getElementById("f-api_key_header");
    const model = document.getElementById("f-model");
    const apiKey = document.getElementById("f-api_key");
    const notes = document.getElementById("preset-notes");

    function apply(overwrite) {
      const p = presets[select.value];
      if (!p) return;
      notes.textContent = p.notes || "";
      if (p.model_hint) model.placeholder = "ex.: " + p.model_hint;
      if (apiKey && !apiKey.placeholder.startsWith("deixe")) {
        apiKey.placeholder = p.key_hint ? "sk-... ou " + p.key_hint : "sem chave";
      }
      if (overwrite) {
        provider.value = p.provider;
        baseUrl.value = p.base_url;
        header.value = p.api_key_header || "Authorization";
      }
    }

    // campos que não se aplicam ao tipo de API escolhido ficam escondidos
    const onlyFor = {
      api_key_header: ["openai", "typesafe"],
      json_mode: ["openai"],
      model: ["openai", "anthropic", "typesafe"],
      base_url: ["openai", "anthropic", "typesafe"],
      api_key: ["openai", "anthropic", "typesafe"],
      extra_headers: ["openai", "anthropic", "typesafe"],
      temperature: ["openai", "anthropic"],
      max_tokens: ["openai", "anthropic"],
    };
    function toggle() {
      Object.entries(onlyFor).forEach(([name, kinds]) => {
        const input = document.querySelector('[name="' + name + '"]');
        const box = input && (input.closest(".field") || input.closest(".check"));
        if (box) box.classList.toggle("hidden", !kinds.includes(provider.value));
      });
    }

    select.addEventListener("change", () => {
      apply(true);
      toggle();
    });
    provider.addEventListener("change", toggle);
    apply(false);
    toggle();
  }

  // --- playground ------------------------------------------------------------

  function playground() {
    const root = document.getElementById("playground");
    if (!root) return;
    const messagesBox = document.getElementById("messages");
    const input = document.getElementById("composer-input");
    const sendButton = document.getElementById("composer-send");
    const profileSelect = document.getElementById("profile-select");
    const resetButton = document.getElementById("reset");
    const tracePanel = document.getElementById("trace");
    const suggestions = document.getElementById("suggestions");
    let history = [];
    let busy = false;
    let awaitingRun = null; // run_id de um agente que pediu informação
    let generation = 0; // invalida acompanhamentos de uma conversa anterior
    let selectedRun = null;
    const latest = new Map(); // run_id -> dados mais recentes

    function scroll() {
      messagesBox.scrollTop = messagesBox.scrollHeight;
    }

    function setPlaceholder() {
      input.placeholder = awaitingRun
        ? "O agente pediu uma informação: responda aqui para continuar"
        : "Escreva uma mensagem (Enter envia, Shift+Enter quebra linha)";
    }

    function reset() {
      generation += 1;
      history = [];
      awaitingRun = null;
      selectedRun = null;
      latest.clear();
      messagesBox.innerHTML = "";
      renderTrace(null);
      setPlaceholder();
      if (suggestions) suggestions.style.display = "";
      input.focus();
    }

    function addBubble(role, text, extraClass) {
      const bubble = el("div", { class: "msg " + role + (extraClass ? " " + extraClass : ""), text: text });
      messagesBox.appendChild(bubble);
      scroll();
      return bubble;
    }

    function select(bubble, runId) {
      messagesBox.querySelectorAll(".msg.selected").forEach((m) => m.classList.remove("selected"));
      bubble.classList.add("selected");
      selectedRun = runId;
      renderTrace(latest.get(runId));
    }

    function meta(data) {
      const box = el("div", { class: "meta" }, [routeBadge(data.route)]);
      if (data.status && data.status !== "completed") box.appendChild(toneBadge(RUN_STATUS, data.status));
      if (data.decided_by) {
        const conf = typeof data.confidence === "number" ? " " + Math.round(data.confidence * 100) + "%" : "";
        box.appendChild(el("span", { class: "badge", text: data.decided_by + conf }));
      }
      const tasks = data.tasks || [];
      if (tasks.length) tasks.forEach((t) => box.appendChild(el("span", { class: "badge", text: t.owner + "/" + t.name })));
      else if (data.agent) box.appendChild(el("span", { class: "badge", text: data.agent + "/" + data.tool }));
      if (data.sources && data.sources.length) box.appendChild(el("span", { class: "badge", text: data.sources.length + " fonte(s)" }));
      if (typeof data.latency_ms === "number") box.appendChild(el("span", { class: "badge", text: ms(data.latency_ms) }));
      return box;
    }

    function assistantBubble(data, text, extraClass) {
      const bubble = addBubble("assistant", text, extraClass);
      bubble.appendChild(meta(data));
      bubble.addEventListener("click", () => select(bubble, data.run_id || data.trace_id));
      return bubble;
    }

    function renderTrace(result) {
      tracePanel.innerHTML = "";
      if (!result) {
        tracePanel.appendChild(
          el("div", { class: "empty" }, [
            el("strong", { text: "Trace da resposta" }),
            "Envie uma mensagem: aqui aparecem a rota e quem decidiu, as tools MCP chamadas, os contratos abertos com agentes A2A, as fontes do RAG e os spans de cada etapa.",
          ])
        );
        return;
      }
      const runId = result.run_id || result.trace_id;
      const actions = el("div", { class: "actions" }, [
        el("a", { class: "btn small", href: "/traces/" + runId, text: "abrir" }),
      ]);
      const open = OPEN_RUN.includes(result.status) || result.status === "needs_input";
      if (open && (result.contracts || []).length) {
        actions.prepend(
          el("button", {
            class: "btn small danger",
            type: "button",
            text: "cancelar",
            onclick: async () => {
              await fetch("/playground/runs/" + runId + "/cancel", { method: "POST" });
              refresh(runId, generation, true);
            },
          })
        );
      }
      tracePanel.appendChild(el("div", { class: "card-head" }, [el("h2", { text: "Trace" }), actions]));

      const kv = el("dl", { class: "kv" });
      function row(label, value) {
        kv.appendChild(el("dt", { text: label }));
        kv.appendChild(el("dd", {}, [value]));
      }
      row("Rota", routeBadge(result.route));
      if (result.status) row("Situação", toneBadge(RUN_STATUS, result.status));
      if (result.decided_by) {
        const conf = typeof result.confidence === "number" ? " · confiança " + Math.round(result.confidence * 100) + "%" : "";
        row("Decidiu", result.decided_by + conf);
      }
      row("Motivo", result.reason || "—");
      row("Modelo", result.model || "—");
      if (typeof result.latency_ms === "number") row("Latência", ms(result.latency_ms));
      const usage = result.usage || {};
      if (usage.input_tokens || usage.output_tokens) row("Tokens", (usage.input_tokens || 0) + " entrada · " + (usage.output_tokens || 0) + " saída");
      tracePanel.appendChild(kv);

      const tasks = result.tasks || [];
      if (tasks.length) {
        tracePanel.appendChild(el("h2", { text: tasks.length > 1 ? "Tarefas" : "Tarefa", style: "margin-top:16px" }));
        tasks.forEach((t) => {
          tracePanel.appendChild(el("div", { class: "small mono", text: (t.kind === "tool" ? "tool " : "skill ") + t.owner + "/" + t.name }));
          if (t.arguments && Object.keys(t.arguments).length) tracePanel.appendChild(el("pre", { class: "block", text: pretty(t.arguments) }));
        });
      } else if (result.arguments && Object.keys(result.arguments).length) {
        tracePanel.appendChild(el("h2", { text: "Argumentos", style: "margin-top:16px" }));
        tracePanel.appendChild(el("pre", { class: "block", text: pretty(result.arguments) }));
      }

      const contracts = result.contracts || [];
      if (contracts.length) {
        tracePanel.appendChild(el("h2", { text: "Contratos", style: "margin-top:16px" }));
        contracts.forEach((c) => {
          tracePanel.appendChild(
            el("div", { class: "source" }, [
              el("div", { class: "head" }, [
                el("a", { href: "/contracts/" + c.id, class: "mono", text: c.agent + "/" + c.skill }),
                toneBadge(CONTRACT_STATES, c.state),
              ]),
              el("div", { class: "small", text: c.error || c.last_message || "—" }),
            ])
          );
        });
      }
      if (result.warnings && result.warnings.length) {
        tracePanel.appendChild(el("h2", { text: "Avisos", style: "margin-top:16px" }));
        result.warnings.forEach((w) => tracePanel.appendChild(el("div", { class: "flash warn", text: w })));
      }
      if (result.sources && result.sources.length) {
        tracePanel.appendChild(el("h2", { text: "Fontes", style: "margin-top:16px" }));
        result.sources.forEach((s) => {
          tracePanel.appendChild(
            el("div", { class: "source" }, [
              el("div", { class: "head" }, [
                el("span", { text: "[" + s.index + "] " + s.document + (s.section ? " › " + s.section : "") }),
                el("span", { class: "mono", text: s.score.toFixed(3) }),
              ]),
              el("div", { class: "small", text: s.snippet }),
            ])
          );
        });
      }
      if (result.spans && result.spans.length) {
        tracePanel.appendChild(el("h2", { text: "Spans", style: "margin-top:16px" }));
        tracePanel.appendChild(renderWaterfall(result.spans));
      }
    }

    function remember(data) {
      const runId = data.run_id || data.trace_id;
      latest.set(runId, Object.assign({}, latest.get(runId) || {}, data));
      if (selectedRun === runId) renderTrace(latest.get(runId));
    }

    // acompanha uma execução em andamento até concluir, falhar ou um agente pedir informação
    async function refresh(runId, gen, once) {
      let delay = 1000;
      while (gen === generation) {
        let data;
        try {
          const resp = await fetch("/playground/runs/" + runId);
          data = await resp.json();
          if (!resp.ok) throw new Error(typeof data.error === "string" ? data.error : pretty(data.error));
        } catch (err) {
          if (once) return;
          await new Promise((r) => setTimeout(r, Math.min(delay * 2, 5000)));
          continue;
        }
        if (gen !== generation) return;
        remember(data);
        if (data.status === "completed" || data.status === "failed") {
          const extra = data.status === "failed" ? "error" : "";
          assistantBubble(data, data.answer || "(sem resposta)", extra);
          history.push({ role: "assistant", content: data.answer || "" });
          return;
        }
        if (data.status === "needs_input") {
          if (awaitingRun !== runId) {
            awaitingRun = runId;
            setPlaceholder();
            assistantBubble(data, data.answer || "O agente precisa de uma informação para continuar.", "asks");
            history.push({ role: "assistant", content: data.answer || "" });
          }
          return;
        }
        if (once) return;
        await new Promise((r) => setTimeout(r, delay));
        delay = Math.min(delay * 1.3, 3000);
      }
    }

    async function send(text) {
      text = (text || "").trim();
      if (!text || busy) return;
      if (!profileSelect.value) {
        addBubble("assistant", "Crie um roteador antes de testar.", "error");
        return;
      }
      busy = true;
      sendButton.disabled = true;
      if (suggestions) suggestions.style.display = "none";
      history.push({ role: "user", content: text });
      addBubble("user", text);
      input.value = "";
      const pending = addBubble("assistant", awaitingRun ? "enviando ao agente…" : "pensando…", "pending");
      const body = { profile: profileSelect.value, messages: history };
      const resuming = awaitingRun;
      if (resuming) body.run_id = resuming;
      const gen = generation;
      try {
        const resp = await fetch("/playground/send", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(typeof data.error === "string" ? data.error : pretty(data.error));
        if (gen !== generation) return;
        if (resuming) {
          awaitingRun = null;
          setPlaceholder();
        }
        const runId = data.run_id || data.trace_id;
        remember(data);
        pending.remove();
        if (data.status === "needs_input") {
          awaitingRun = runId;
          setPlaceholder();
        }
        const bubble = assistantBubble(data, data.answer, data.status === "needs_input" ? "asks" : data.status === "failed" ? "error" : "");
        history.push({ role: "assistant", content: data.answer });
        select(bubble, runId);
        if (OPEN_RUN.includes(data.status)) refresh(runId, gen, false);
      } catch (err) {
        history.pop();
        pending.className = "msg assistant error";
        pending.textContent = "Erro: " + err.message;
      } finally {
        busy = false;
        sendButton.disabled = false;
        scroll();
        input.focus();
      }
    }

    sendButton.addEventListener("click", () => send(input.value));
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        send(input.value);
      }
    });
    resetButton.addEventListener("click", reset);
    profileSelect.addEventListener("change", () => {
      const url = new URL(window.location.href);
      url.searchParams.set("profile", profileSelect.value);
      window.history.replaceState(null, "", url);
      reset();
    });
    if (suggestions) {
      suggestions.querySelectorAll("button").forEach((button) => {
        button.addEventListener("click", () => send(button.dataset.text));
      });
    }
    renderTrace(null);
    setPlaceholder();
    input.focus();
  }

  window.SwitchboardUI = { modelForm: modelForm, playground: playground };
})();
