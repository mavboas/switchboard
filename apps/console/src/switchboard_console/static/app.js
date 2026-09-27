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
    delegated: "Delegado",
    clarify: "Esclarecimento",
    error: "Erro",
  };

  function routeBadge(route) {
    return el("span", { class: "badge route-" + route }, [el("span", { class: "dot" }), ROUTES[route] || route]);
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
      api_key_header: ["openai"],
      json_mode: ["openai"],
      model: ["openai", "anthropic"],
      base_url: ["openai", "anthropic"],
      api_key: ["openai", "anthropic"],
      extra_headers: ["openai", "anthropic"],
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

    function scroll() {
      messagesBox.scrollTop = messagesBox.scrollHeight;
    }

    function reset() {
      history = [];
      messagesBox.innerHTML = "";
      renderTrace(null);
      if (suggestions) suggestions.style.display = "";
      input.focus();
    }

    function addBubble(role, text, extraClass) {
      const bubble = el("div", { class: "msg " + role + (extraClass ? " " + extraClass : ""), text: text });
      messagesBox.appendChild(bubble);
      scroll();
      return bubble;
    }

    function renderTrace(result) {
      tracePanel.innerHTML = "";
      if (!result) {
        tracePanel.appendChild(
          el("div", { class: "empty" }, [
            el("strong", { text: "Trace da resposta" }),
            "Envie uma mensagem: aqui aparecem a rota escolhida, o agente e a tool acionados via MCP, as fontes do RAG e cada etapa com o tempo gasto.",
          ])
        );
        return;
      }
      const head = el("div", { class: "card-head" }, [
        el("h2", { text: "Trace" }),
        el("a", { class: "btn small", href: "/traces/" + result.trace_id, text: "abrir" }),
      ]);
      tracePanel.appendChild(head);

      const kv = el("dl", { class: "kv" });
      function row(label, value) {
        kv.appendChild(el("dt", { text: label }));
        kv.appendChild(el("dd", {}, [value]));
      }
      row("Rota", routeBadge(result.route));
      row("Motivo", result.reason || "—");
      row("Modelo", result.model);
      if (result.agent) row("Agente", result.agent + " / " + result.tool);
      row("Latência", Math.round(result.latency_ms) + " ms");
      const usage = result.usage || {};
      if (usage.input_tokens || usage.output_tokens) row("Tokens", usage.input_tokens + " entrada · " + usage.output_tokens + " saída");
      tracePanel.appendChild(kv);

      if (result.arguments && Object.keys(result.arguments).length) {
        tracePanel.appendChild(el("h2", { text: "Argumentos da tool", style: "margin-top:16px" }));
        tracePanel.appendChild(el("pre", { class: "block", text: pretty(result.arguments) }));
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
      tracePanel.appendChild(el("h2", { text: "Etapas", style: "margin-top:16px" }));
      const timeline = el("ul", { class: "timeline" });
      (result.steps || []).forEach((step) => {
        timeline.appendChild(
          el("li", {}, [
            el("span", { class: "step-name", text: step.name }),
            el("span", { class: "step-ms", text: Math.round(step.duration_ms) + " ms" }),
            el("pre", { class: "block", text: pretty(step.detail) }),
          ])
        );
      });
      tracePanel.appendChild(timeline);
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
      const pending = addBubble("assistant", "pensando…", "pending");
      try {
        const resp = await fetch("/playground/send", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ profile: profileSelect.value, messages: history }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(typeof data.error === "string" ? data.error : pretty(data.error));
        pending.className = "msg assistant";
        pending.textContent = data.answer;
        const meta = el("div", { class: "meta" }, [routeBadge(data.route)]);
        if (data.agent) meta.appendChild(el("span", { class: "badge", text: data.agent + "/" + data.tool }));
        if (data.sources && data.sources.length) meta.appendChild(el("span", { class: "badge", text: data.sources.length + " fonte(s)" }));
        meta.appendChild(el("span", { class: "badge", text: Math.round(data.latency_ms) + " ms" }));
        pending.appendChild(meta);
        pending.addEventListener("click", () => {
          messagesBox.querySelectorAll(".msg.selected").forEach((m) => m.classList.remove("selected"));
          pending.classList.add("selected");
          renderTrace(data);
        });
        messagesBox.querySelectorAll(".msg.selected").forEach((m) => m.classList.remove("selected"));
        pending.classList.add("selected");
        history.push({ role: "assistant", content: data.answer });
        renderTrace(data);
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
    input.focus();
  }

  window.SwitchboardUI = { modelForm: modelForm, playground: playground };
})();
