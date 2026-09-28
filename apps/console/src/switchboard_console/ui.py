"""Páginas de configuração do console (HTML renderizado no servidor).

Painel, modelos, conectores MCP, agentes A2A, bases de conhecimento e
roteadores. As páginas de operação (playground, execuções e contratos) ficam
em :mod:`.ops`.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import anyio.to_thread
import httpx
from fastapi import APIRouter, Depends, Request
from sqlalchemy import select

from switchboard.config import DEFAULT_SYSTEM_PROMPT
from switchboard.contracts import states
from switchboard.errors import SwitchboardError
from switchboard.jev import build_decision_model, noul
from switchboard.llm import PRESETS, Message, build_chat_model
from switchboard.rag import SUPPORTED_EXTENSIONS, extract_text, guess_title
from switchboard.secrets import MASK
from switchboard.storage import repo
from switchboard.storage.orm import (
    Agent,
    Connector,
    Contract,
    Document,
    KnowledgeBase,
    LlmModel,
    RouterProfile,
    Trace,
)

from . import forms
from .ops import trace_row
from .state import Console, get_console, redirect

router = APIRouter()


# ---------------------------------------------------------------------------
# helpers de leitura (sempre devolvem dados simples, nunca objetos da sessão)


def _model_dict(row: LlmModel, used_by: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "preset": row.preset,
        "provider": row.provider,
        "model": row.model,
        "base_url": row.base_url or "",
        "api_key": row.api_key,
        "api_key_header": row.api_key_header,
        "extra_headers": forms.headers_text(
            {
                k: (MASK if str(v).startswith("enc:") else v)
                for k, v in (row.extra_headers or {}).items()
            }
        ),
        "temperature": row.temperature,
        "max_tokens": row.max_tokens,
        "timeout_s": row.timeout_s,
        "json_mode": row.json_mode,
        "used_by": used_by or [],
    }


def _connector_dict(row: Connector) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "url": row.url,
        "transport": row.transport,
        "auth_token": row.auth_token,
        "allowed_tools": ", ".join(row.allowed_tools or []),
        "enabled": row.enabled,
        "timeout_s": row.timeout_s,
    }


def _agent_dict(row: Agent) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "url": row.url,
        "auth_token": row.auth_token,
        "allowed_skills": ", ".join(row.allowed_skills or []),
        "enabled": row.enabled,
        "timeout_s": row.timeout_s,
        "deadline_s": row.deadline_s,
        "push": True if row.push is None else row.push,
        "allow_cross_origin": bool(row.allow_cross_origin),
    }


def _kb_dict(row: KnowledgeBase, stats: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "embedder_kind": row.embedder_kind,
        "embedder_dim": row.embedder_dim,
        "embedding_model_id": row.embedding_model_id,
        "embedding_model": row.embedding_model or "",
        "chunk_size": row.chunk_size,
        "chunk_overlap": row.chunk_overlap,
        "label": repo.kb_label(row),
        "stats": stats or {"documents": 0, "chunks": 0, "stale": 0},
    }


def _profile_dict(row: RouterProfile) -> dict[str, Any]:
    spec = repo.profile_to_spec(row, only_enabled=False)
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "model_id": row.model_id,
        "model_name": row.model.name,
        "model_label": f"{row.model.provider}:{row.model.model or row.model.name}",
        "decision_model_id": row.decision_model_id,
        "decision_model_name": row.decision_model.name if row.decision_model else None,
        "decision_threshold": spec.decision_threshold,
        "system_prompt": row.system_prompt,
        "connector_ids": [c.id for c in row.connectors],
        "connectors": [c.name for c in row.connectors],
        "agent_ids": [a.id for a in row.agents],
        "agents": [a.name for a in row.agents],
        "kb_ids": [k.id for k in row.knowledge_bases],
        "kbs": [k.name for k in row.knowledge_bases],
        "top_k": row.top_k,
        "min_score": row.min_score,
        "synthesize": row.synthesize,
        "allow_clarify": row.allow_clarify,
        "wait_s": spec.wait_s,
        "max_parallel": spec.max_parallel,
        "deadline_s": spec.deadline_s,
        "enabled": row.enabled,
    }


# ---------------------------------------------------------------------------
# painel


async def _router_health(console: Console) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        resp = await console.router_request("GET", "/healthz", timeout=2.0)
        ok = resp.status_code == 200
        detail = "no ar" if ok else f"HTTP {resp.status_code}"
    except httpx.HTTPError as exc:
        ok, detail = False, f"inacessível ({exc.__class__.__name__})"
    return {"ok": ok, "detail": detail, "ms": (time.perf_counter() - started) * 1000}


@router.get("/")
async def dashboard(request: Request, console: Console = Depends(get_console)):
    def load(s):
        stats = repo.kb_stats(s)
        return {
            "counts": {
                "models": repo.count(s, LlmModel),
                "connectors": repo.count(s, Connector),
                "agents": repo.count(s, Agent),
                "kbs": repo.count(s, KnowledgeBase),
                "profiles": repo.count(s, RouterProfile),
                "traces": repo.count(s, Trace),
                "contracts": repo.count(s, Contract),
                "documents": sum(v["documents"] for v in stats.values()),
                "chunks": sum(v["chunks"] for v in stats.values()),
            },
            "routes": repo.route_counts(s),
            "contract_states": repo.contract_state_counts(s),
            "open_runs": [
                trace_row(t)
                for status in ("needs_input", "pending", "consolidating")
                for t in repo.query_traces(s, status=status, limit=8)
            ],
            "recent": [trace_row(t) for t in repo.query_traces(s, limit=8)],
            "profiles": [_profile_dict(p) for p in repo.list_profiles(s)],
            "connector_specs": [
                repo.connector_to_spec(c)
                for c in s.scalars(select(Connector).order_by(Connector.name))
            ],
            "agent_specs": [
                repo.agent_to_spec(a) for a in s.scalars(select(Agent).order_by(Agent.name))
            ],
        }

    data = await console.run_db(load)
    health, connectors, agents = await asyncio.gather(
        _router_health(console),
        console.connectors.discover(data["connector_specs"]),
        console.agents.discover(data["agent_specs"]),
    )
    counts = data["counts"]
    steps = [
        ("Cadastre um modelo (LLM)", counts["models"] > 0, "/models/new"),
        ("Registre conectores MCP (tools rápidas)", counts["connectors"] > 0, "/connectors/new"),
        ("Registre agentes A2A (tarefas delegadas)", counts["agents"] > 0, "/agents/new"),
        ("Crie uma base de conhecimento", counts["documents"] > 0, "/knowledge/new"),
        ("Monte um roteador", counts["profiles"] > 0, "/profiles/new"),
        ("Teste no playground", counts["traces"] > 0, "/playground"),
    ]
    open_contracts = sum(n for st, n in data["contract_states"].items() if st in states.OPEN)
    return console.render(
        request,
        "dashboard.html",
        nav="dashboard",
        data=data,
        health=health,
        connectors=connectors,
        agents=agents,
        open_contracts=open_contracts,
        steps=steps,
        db_info={
            "dialect": console.db.dialect,
            "vector": console.db.vector_mode,
            "url": console.db.safe_url,
        },
    )


# ---------------------------------------------------------------------------
# modelos


@router.get("/models")
async def models_list(request: Request, console: Console = Depends(get_console)):
    def load(s):
        usage: dict[int, list[str]] = {}
        for p in s.scalars(select(RouterProfile)):
            usage.setdefault(p.model_id, []).append(p.name)
            if p.decision_model_id:
                usage.setdefault(p.decision_model_id, []).append(f"{p.name} (decisão)")
        rows = s.scalars(select(LlmModel).order_by(LlmModel.name))
        return [_model_dict(m, usage.get(m.id)) for m in rows]

    return console.render(
        request, "models/list.html", nav="models", models=await console.run_db(load)
    )


def _model_form(
    console: Console, request: Request, values: dict[str, Any], error: str | None = None
):
    return console.render(
        request,
        "models/form.html",
        nav="models",
        values=values,
        error=error,
        presets_json=json.dumps({k: p.to_dict() for k, p in PRESETS.items()}, ensure_ascii=False),
        secrets_enabled=console.box.enabled,
    )


@router.get("/models/new")
async def models_new(
    request: Request, preset: str = "openai", console: Console = Depends(get_console)
):
    p = PRESETS.get(preset) or PRESETS["openai"]
    decision = p.provider == "typesafe"
    values = {
        "preset": p.key,
        "provider": p.provider,
        "base_url": p.base_url,
        "api_key_header": p.api_key_header,
        "model": p.model_hint if decision else "",
        "temperature": None if decision else 0.2,
        "max_tokens": None if decision else 4096,
        "timeout_s": 10 if decision else 60,
        "json_mode": not decision,
    }
    return _model_form(console, request, values)


@router.post("/models")
async def models_create(request: Request, console: Console = Depends(get_console)):
    form = await request.form()
    data = forms.model_data(form)
    try:
        row = await console.run_db(lambda s: repo.save_model(s, data, box=console.box).name)
    except SwitchboardError as exc:
        return _model_form(console, request, {**data, "api_key": None}, str(exc))
    return redirect("/models", ok=f"Modelo '{row}' criado. Use 'Testar' para validar a conexão.")


@router.get("/models/{model_id}")
async def models_edit(request: Request, model_id: int, console: Console = Depends(get_console)):
    row = await console.run_db(lambda s: (m := s.get(LlmModel, model_id)) and _model_dict(m))
    if row is None:
        return redirect("/models", erro="Modelo não encontrado.")
    return _model_form(console, request, row)


@router.post("/models/{model_id}")
async def models_update(request: Request, model_id: int, console: Console = Depends(get_console)):
    form = await request.form()
    data = forms.model_data(form)
    try:
        await console.run_db(lambda s: repo.save_model(s, data, box=console.box, model_id=model_id))
    except SwitchboardError as exc:
        current = await console.run_db(
            lambda s: (m := s.get(LlmModel, model_id)) and _model_dict(m)
        )
        return _model_form(
            console,
            request,
            {**data, "id": model_id, "api_key": (current or {}).get("api_key")},
            str(exc),
        )
    return redirect("/models", ok="Modelo atualizado.")


@router.post("/models/{model_id}/delete")
async def models_delete(model_id: int, console: Console = Depends(get_console)):
    try:
        await console.run_db(lambda s: repo.delete_model(s, model_id))
    except SwitchboardError as exc:
        return redirect("/models", erro=str(exc))
    return redirect("/models", ok="Modelo removido.")


@router.post("/models/{model_id}/test")
async def models_test(model_id: int, console: Console = Depends(get_console)):
    spec = await console.run_db(
        lambda s: (m := s.get(LlmModel, model_id)) and repo.model_to_spec(m)
    )
    if spec is None:
        return redirect("/models", erro="Modelo não encontrado.")
    if spec.is_decision_model:
        return await _test_decision_model(console, spec)
    try:
        chat = build_chat_model(spec, resolve_secret=console.box.open)
        try:
            # sem limite baixo de tokens: modelos com raciocínio gastam tokens antes de responder
            result = await chat.chat(
                [Message("user", "Teste de conexão do Switchboard. Responda apenas: ok")]
            )
        finally:
            await chat.aclose()
    except SwitchboardError as exc:
        return redirect("/models", erro=f"Falha no teste de '{spec.name}': {exc}")
    reply = result.text.strip().replace("\n", " ")[:120]
    return redirect("/models", ok=f"'{spec.name}' respondeu em {result.latency_ms:.0f} ms: {reply}")


async def _test_decision_model(console: Console, spec) -> Any:
    """Uma pergunta sim/não óbvia ao Jev: confere URL, chave, modelo e formato da resposta."""
    try:
        client = build_decision_model(
            spec, resolve_secret=console.box.open, transport=console.jev_transport
        )
        try:
            result = await client.evaluate(
                {"message": "Hi! I would like to check my account balance, please."},
                {
                    "greets": noul(
                        "Does the message contain a greeting?",
                        true="the message greets someone",
                        false="there is no greeting in the message",
                    )
                },
            )
        finally:
            await client.aclose()
    except SwitchboardError as exc:
        return redirect("/models", erro=f"Falha no teste de '{spec.name}': {exc}")
    answer = result.get("greets")
    value = f"{answer.noul:.2f}" if answer and answer.noul is not None else "?"
    return redirect(
        "/models",
        ok=(
            f"'{spec.name}' ({result.model}) respondeu em {result.latency_ms:.0f} ms: "
            f"noul = {value} para uma saudação óbvia (quanto mais perto de 1, melhor)."
        ),
    )


# ---------------------------------------------------------------------------
# conectores MCP


def _connector_row(s, connector_id: int):
    row = s.get(Connector, connector_id)
    return (_connector_dict(row), repo.connector_to_spec(row)) if row is not None else None


@router.get("/connectors")
async def connectors_list(request: Request, console: Console = Depends(get_console)):
    def load(s):
        rows = s.scalars(select(Connector).order_by(Connector.name)).all()
        return [(_connector_dict(c), repo.connector_to_spec(c)) for c in rows]

    rows = await console.run_db(load)
    infos = await asyncio.gather(
        *(console.connectors.describe(spec) for _, spec in rows if spec.enabled)
    )
    by_name = {i.name: i for i in infos}
    items = [{**data, "info": by_name.get(data["name"])} for data, _ in rows]
    return console.render(request, "connectors/list.html", nav="connectors", connectors=items)


def _connector_form(
    console: Console, request: Request, values: dict[str, Any], error: str | None = None
):
    return console.render(
        request,
        "connectors/form.html",
        nav="connectors",
        values=values,
        error=error,
        secrets_enabled=console.box.enabled,
    )


@router.get("/connectors/new")
async def connectors_new(request: Request, console: Console = Depends(get_console)):
    return _connector_form(
        console, request, {"transport": "streamable-http", "enabled": True, "timeout_s": 30}
    )


@router.post("/connectors")
async def connectors_create(request: Request, console: Console = Depends(get_console)):
    data = forms.connector_data(await request.form())
    try:
        connector_id = await console.run_db(
            lambda s: repo.save_connector(s, data, box=console.box).id
        )
    except SwitchboardError as exc:
        return _connector_form(console, request, {**data, "auth_token": None}, str(exc))
    return redirect(
        f"/connectors/{connector_id}",
        ok="Conector registrado. Veja abaixo as tools que a descoberta via MCP encontrou.",
    )


async def _connector_detail(
    console: Console,
    request: Request,
    connector_id: int,
    *,
    refresh: bool = False,
    call: dict | None = None,
):
    row = await console.run_db(_connector_row, connector_id)
    if row is None:
        return redirect("/connectors", erro="Conector não encontrado.")
    data, spec = row
    info = await console.connectors.describe(spec, refresh=refresh)
    return console.render(
        request, "connectors/detail.html", nav="connectors", connector=data, info=info, call=call
    )


@router.get("/connectors/{connector_id}")
async def connectors_detail(
    request: Request,
    connector_id: int,
    refresh: bool = False,
    console: Console = Depends(get_console),
):
    return await _connector_detail(console, request, connector_id, refresh=refresh)


@router.get("/connectors/{connector_id}/edit")
async def connectors_edit(
    request: Request, connector_id: int, console: Console = Depends(get_console)
):
    row = await console.run_db(
        lambda s: (c := s.get(Connector, connector_id)) and _connector_dict(c)
    )
    if row is None:
        return redirect("/connectors", erro="Conector não encontrado.")
    return _connector_form(console, request, row)


@router.post("/connectors/{connector_id}")
async def connectors_update(
    request: Request, connector_id: int, console: Console = Depends(get_console)
):
    data = forms.connector_data(await request.form())
    try:
        await console.run_db(
            lambda s: repo.save_connector(s, data, box=console.box, connector_id=connector_id)
        )
    except SwitchboardError as exc:
        current = await console.run_db(
            lambda s: (c := s.get(Connector, connector_id)) and _connector_dict(c)
        )
        return _connector_form(
            console,
            request,
            {**data, "id": connector_id, "auth_token": (current or {}).get("auth_token")},
            str(exc),
        )
    console.connectors.invalidate(data["name"])
    return redirect(f"/connectors/{connector_id}", ok="Conector atualizado.")


@router.post("/connectors/{connector_id}/delete")
async def connectors_delete(connector_id: int, console: Console = Depends(get_console)):
    try:
        await console.run_db(lambda s: repo.delete_connector(s, connector_id))
    except SwitchboardError as exc:
        return redirect("/connectors", erro=str(exc))
    return redirect("/connectors", ok="Conector removido.")


@router.post("/connectors/{connector_id}/call")
async def connectors_call(
    request: Request, connector_id: int, console: Console = Depends(get_console)
):
    form = await request.form()
    tool = forms.text(form, "tool")
    raw = forms.text(form, "arguments") or "{}"
    call: dict[str, Any] = {"tool": tool, "arguments": raw}
    row = await console.run_db(_connector_row, connector_id)
    if row is None:
        return redirect("/connectors", erro="Conector não encontrado.")
    _, spec = row
    try:
        arguments = json.loads(raw)
        if not isinstance(arguments, dict):
            raise ValueError("os argumentos precisam ser um objeto JSON")
        outcome = await console.connectors.call(spec, tool, arguments)
        call.update(result=outcome.text, is_error=outcome.is_error, ms=outcome.latency_ms)
    except (ValueError, SwitchboardError) as exc:
        call.update(result=str(exc), is_error=True, ms=0)
    return await _connector_detail(console, request, connector_id, call=call)


# ---------------------------------------------------------------------------
# agentes A2A


def _agent_row(s, agent_id: int):
    row = s.get(Agent, agent_id)
    return (_agent_dict(row), repo.agent_to_spec(row)) if row is not None else None


@router.get("/agents")
async def agents_list(request: Request, console: Console = Depends(get_console)):
    def load(s):
        rows = s.scalars(select(Agent).order_by(Agent.name)).all()
        return [(_agent_dict(a), repo.agent_to_spec(a)) for a in rows], repo.contract_agent_counts(
            s
        )

    rows, counts = await console.run_db(load)
    infos = await asyncio.gather(
        *(console.agents.describe(spec) for _, spec in rows if spec.enabled)
    )
    by_name = {i.name: i for i in infos}
    items = [
        {**data, "info": by_name.get(data["name"]), "contracts": counts.get(data["name"], {})}
        for data, _ in rows
    ]
    return console.render(request, "agents/list.html", nav="agents", agents=items)


def _agent_form(
    console: Console, request: Request, values: dict[str, Any], error: str | None = None
):
    return console.render(
        request,
        "agents/form.html",
        nav="agents",
        values=values,
        error=error,
        secrets_enabled=console.box.enabled,
    )


@router.get("/agents/new")
async def agents_new(request: Request, console: Console = Depends(get_console)):
    return _agent_form(console, request, {"enabled": True, "timeout_s": 15, "push": True})


@router.post("/agents")
async def agents_create(request: Request, console: Console = Depends(get_console)):
    data = forms.agent_data(await request.form())
    try:
        agent_id = await console.run_db(lambda s: repo.save_agent(s, data, box=console.box).id)
    except SwitchboardError as exc:
        return _agent_form(console, request, {**data, "auth_token": None}, str(exc))
    return redirect(
        f"/agents/{agent_id}",
        ok="Agente registrado. Veja abaixo o Agent Card e os termos de contrato de cada skill.",
    )


@router.get("/agents/{agent_id}")
async def agents_detail(
    request: Request, agent_id: int, refresh: bool = False, console: Console = Depends(get_console)
):
    def load(s):
        row = _agent_row(s, agent_id)
        if row is None:
            return None
        recent = repo.query_contracts(s, agent=row[0]["name"], limit=10)
        return row, recent

    loaded = await console.run_db(load)
    if loaded is None:
        return redirect("/agents", erro="Agente não encontrado.")
    (data, spec), recent = loaded
    info = await console.agents.describe(spec, refresh=refresh)
    return console.render(
        request, "agents/detail.html", nav="agents", agent=data, info=info, contracts=recent
    )


@router.get("/agents/{agent_id}/edit")
async def agents_edit(request: Request, agent_id: int, console: Console = Depends(get_console)):
    row = await console.run_db(lambda s: (a := s.get(Agent, agent_id)) and _agent_dict(a))
    if row is None:
        return redirect("/agents", erro="Agente não encontrado.")
    return _agent_form(console, request, row)


@router.post("/agents/{agent_id}")
async def agents_update(request: Request, agent_id: int, console: Console = Depends(get_console)):
    data = forms.agent_data(await request.form())
    try:
        await console.run_db(lambda s: repo.save_agent(s, data, box=console.box, agent_id=agent_id))
    except SwitchboardError as exc:
        current = await console.run_db(lambda s: (a := s.get(Agent, agent_id)) and _agent_dict(a))
        return _agent_form(
            console,
            request,
            {**data, "id": agent_id, "auth_token": (current or {}).get("auth_token")},
            str(exc),
        )
    console.agents.invalidate(data["name"])
    return redirect(f"/agents/{agent_id}", ok="Agente atualizado.")


@router.post("/agents/{agent_id}/delete")
async def agents_delete(agent_id: int, console: Console = Depends(get_console)):
    try:
        await console.run_db(lambda s: repo.delete_agent(s, agent_id))
    except SwitchboardError as exc:
        return redirect("/agents", erro=str(exc))
    return redirect(
        "/agents", ok="Agente removido (o histórico de contratos continua em Contratos)."
    )


# ---------------------------------------------------------------------------
# bases de conhecimento


def _connections(s) -> list[dict[str, Any]]:
    rows = s.scalars(select(LlmModel).where(LlmModel.provider == "openai").order_by(LlmModel.name))
    return [{"id": m.id, "name": m.name, "base_url": m.base_url or ""} for m in rows]


@router.get("/knowledge")
async def knowledge_list(request: Request, console: Console = Depends(get_console)):
    def load(s):
        stats = repo.kb_stats(s)
        rows = s.scalars(select(KnowledgeBase).order_by(KnowledgeBase.name))
        return [_kb_dict(k, stats.get(k.id)) for k in rows]

    return console.render(
        request, "knowledge/list.html", nav="knowledge", kbs=await console.run_db(load)
    )


async def _kb_form(
    console: Console, request: Request, values: dict[str, Any], error: str | None = None
):
    return console.render(
        request,
        "knowledge/form.html",
        nav="knowledge",
        values=values,
        error=error,
        connections=await console.run_db(_connections),
    )


@router.get("/knowledge/new")
async def knowledge_new(request: Request, console: Console = Depends(get_console)):
    return await _kb_form(
        console,
        request,
        {"embedder_kind": "hashing", "embedder_dim": 512, "chunk_size": 800, "chunk_overlap": 120},
    )


@router.post("/knowledge")
async def knowledge_create(request: Request, console: Console = Depends(get_console)):
    data = forms.knowledge_data(await request.form())
    try:
        kb_id = await console.run_db(lambda s: repo.save_knowledge_base(s, data).id)
    except SwitchboardError as exc:
        return await _kb_form(console, request, data, str(exc))
    return redirect(f"/knowledge/{kb_id}", ok="Base criada. Agora adicione documentos.")


async def _kb_detail(console: Console, request: Request, kb_id: int, *, search: dict | None = None):
    def load(s):
        kb = s.get(KnowledgeBase, kb_id)
        if kb is None:
            return None
        docs = s.scalars(
            select(Document).where(Document.kb_id == kb_id).order_by(Document.created_at.desc())
        )
        documents = [
            {
                "id": d.id,
                "title": d.title,
                "source": d.source,
                "chunks": d.chunk_count,
                "embedder": d.embedder,
                "created": d.created_at,
                "size": len(d.content),
            }
            for d in docs
        ]
        profiles = [
            p.name for p in repo.list_profiles(s) if kb_id in [k.id for k in p.knowledge_bases]
        ]
        return _kb_dict(kb, repo.kb_stats(s, [kb_id]).get(kb_id)), documents, profiles

    loaded = await console.run_db(load)
    if loaded is None:
        return redirect("/knowledge", erro="Base não encontrada.")
    kb, documents, profiles = loaded
    return console.render(
        request,
        "knowledge/detail.html",
        nav="knowledge",
        kb=kb,
        documents=documents,
        profiles=profiles,
        search=search,
        extensions=", ".join(sorted(SUPPORTED_EXTENSIONS)),
    )


@router.get("/knowledge/{kb_id}")
async def knowledge_detail(request: Request, kb_id: int, console: Console = Depends(get_console)):
    return await _kb_detail(console, request, kb_id)


@router.get("/knowledge/{kb_id}/edit")
async def knowledge_edit(request: Request, kb_id: int, console: Console = Depends(get_console)):
    row = await console.run_db(lambda s: (k := s.get(KnowledgeBase, kb_id)) and _kb_dict(k))
    if row is None:
        return redirect("/knowledge", erro="Base não encontrada.")
    return await _kb_form(console, request, row)


@router.post("/knowledge/{kb_id}")
async def knowledge_update(request: Request, kb_id: int, console: Console = Depends(get_console)):
    data = forms.knowledge_data(await request.form())
    try:
        await console.run_db(lambda s: repo.save_knowledge_base(s, data, kb_id=kb_id))
    except SwitchboardError as exc:
        return await _kb_form(console, request, {**data, "id": kb_id}, str(exc))
    return redirect(f"/knowledge/{kb_id}", ok="Base atualizada.")


@router.post("/knowledge/{kb_id}/delete")
async def knowledge_delete(kb_id: int, console: Console = Depends(get_console)):
    try:
        await console.run_db(lambda s: repo.delete_knowledge_base(s, kb_id))
    except SwitchboardError as exc:
        return redirect("/knowledge", erro=str(exc))
    return redirect("/knowledge", ok="Base removida (com todos os documentos).")


@router.post("/knowledge/{kb_id}/documents")
async def knowledge_add(request: Request, kb_id: int, console: Console = Depends(get_console)):
    form = await request.form()
    items: list[tuple[str, str, str]] = []
    errors: list[str] = []
    limit = console.settings.max_upload_mb * 1024 * 1024
    for upload in form.getlist("files"):
        if not getattr(upload, "filename", None):
            continue
        too_big = f"{upload.filename}: maior que {console.settings.max_upload_mb} MB"
        if (getattr(upload, "size", None) or 0) > limit:
            errors.append(too_big)
            continue
        data = await upload.read(limit + 1)
        if len(data) > limit:
            errors.append(too_big)
            continue
        try:
            # extrair texto (PDF, HTML…) é CPU: roda fora do event loop
            text = await anyio.to_thread.run_sync(extract_text, upload.filename, data)
            items.append((guess_title(upload.filename, text), upload.filename, text))
        except SwitchboardError as exc:
            errors.append(f"{upload.filename}: {exc}")
    content = forms.text(form, "content")
    if content:
        items.append((forms.text(form, "title") or "Texto colado", "texto", content))
    if not items and not errors:
        return redirect(f"/knowledge/{kb_id}", erro="Envie um arquivo ou cole um texto.")
    created, repeated = 0, 0
    for title, source, text in items:
        try:
            _, was_created = await console.knowledge.add_document(
                kb_id, title=title, content=text, source=source
            )
        except SwitchboardError as exc:
            errors.append(f"{source}: {exc}")
            continue
        created += int(was_created)
        repeated += int(not was_created)
    parts = [f"{created} documento(s) indexado(s)"]
    if repeated:
        parts.append(f"{repeated} ignorado(s) por conteúdo repetido")
    ok = "; ".join(parts) + "."
    return redirect(f"/knowledge/{kb_id}", ok=ok, erro="; ".join(errors) or None)


@router.post("/knowledge/{kb_id}/documents/{doc_id}/delete")
async def knowledge_doc_delete(kb_id: int, doc_id: int, console: Console = Depends(get_console)):
    try:
        await console.knowledge.delete_document(kb_id, doc_id)
    except SwitchboardError as exc:
        return redirect(f"/knowledge/{kb_id}", erro=str(exc))
    return redirect(f"/knowledge/{kb_id}", ok="Documento removido.")


@router.post("/knowledge/{kb_id}/search")
async def knowledge_search(request: Request, kb_id: int, console: Console = Depends(get_console)):
    form = await request.form()
    query = forms.text(form, "query")
    name = await console.run_db(lambda s: (k := s.get(KnowledgeBase, kb_id)) and k.name)
    if name is None:
        return redirect("/knowledge", erro="Base não encontrada.")
    search: dict[str, Any] = {"query": query, "hits": []}
    if query:
        try:
            search["hits"] = await console.knowledge.search(query, [name], top_k=5, min_score=0.0)
        except SwitchboardError as exc:
            search["error"] = str(exc)
    return await _kb_detail(console, request, kb_id, search=search)


@router.post("/knowledge/{kb_id}/reindex")
async def knowledge_reindex(kb_id: int, console: Console = Depends(get_console)):
    try:
        total = await console.knowledge.reindex(kb_id)
    except SwitchboardError as exc:
        return redirect(f"/knowledge/{kb_id}", erro=str(exc))
    return redirect(f"/knowledge/{kb_id}", ok=f"Reindexação concluída: {total} trecho(s).")


# ---------------------------------------------------------------------------
# roteadores (perfis)


@router.get("/profiles")
async def profiles_list(request: Request, console: Console = Depends(get_console)):
    rows = await console.run_db(lambda s: [_profile_dict(p) for p in repo.list_profiles(s)])
    return console.render(request, "profiles/list.html", nav="profiles", profiles=rows)


async def _profile_form(
    console: Console, request: Request, values: dict[str, Any], error: str | None = None
):
    def load(s):
        models = [
            {
                "id": m.id,
                "name": m.name,
                "label": f"{m.provider}:{m.model or m.name}",
                "decision": m.provider == "typesafe",
            }
            for m in s.scalars(select(LlmModel).order_by(LlmModel.name))
        ]
        return (
            [m for m in models if not m["decision"]],
            [m for m in models if m["decision"]],
            [
                {"id": c.id, "name": c.name, "description": c.description, "enabled": c.enabled}
                for c in s.scalars(select(Connector).order_by(Connector.name))
            ],
            [
                {"id": a.id, "name": a.name, "description": a.description, "enabled": a.enabled}
                for a in s.scalars(select(Agent).order_by(Agent.name))
            ],
            [
                {"id": k.id, "name": k.name, "description": k.description}
                for k in s.scalars(select(KnowledgeBase).order_by(KnowledgeBase.name))
            ],
        )

    models, deciders, connectors, agents, kbs = await console.run_db(load)
    return console.render(
        request,
        "profiles/form.html",
        nav="profiles",
        values=values,
        error=error,
        models=models,
        deciders=deciders,
        connectors=connectors,
        agents=agents,
        kbs=kbs,
    )


@router.get("/profiles/new")
async def profiles_new(request: Request, console: Console = Depends(get_console)):
    values = {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "decision_threshold": 0.6,
        "top_k": 4,
        "min_score": 0.2,
        "synthesize": True,
        "allow_clarify": True,
        "wait_s": 8,
        "max_parallel": 3,
        "deadline_s": 600,
        "enabled": True,
        "connector_ids": [],
        "agent_ids": [],
        "kb_ids": [],
    }
    return await _profile_form(console, request, values)


@router.post("/profiles")
async def profiles_create(request: Request, console: Console = Depends(get_console)):
    data = forms.profile_data(await request.form())
    try:
        profile_id = await console.run_db(lambda s: repo.save_profile(s, data).id)
    except SwitchboardError as exc:
        return await _profile_form(console, request, data, str(exc))
    return redirect(
        f"/profiles/{profile_id}",
        ok="Roteador criado. O router aplica a mudança em poucos segundos.",
    )


@router.get("/profiles/{profile_id}")
async def profiles_edit(request: Request, profile_id: int, console: Console = Depends(get_console)):
    row = await console.run_db(
        lambda s: (p := repo.get_profile(s, profile_id)) and _profile_dict(p)
    )
    if row is None:
        return redirect("/profiles", erro="Roteador não encontrado.")
    return await _profile_form(console, request, row)


@router.post("/profiles/{profile_id}")
async def profiles_update(
    request: Request, profile_id: int, console: Console = Depends(get_console)
):
    data = forms.profile_data(await request.form())
    try:
        await console.run_db(lambda s: repo.save_profile(s, data, profile_id=profile_id))
    except SwitchboardError as exc:
        return await _profile_form(console, request, {**data, "id": profile_id}, str(exc))
    return redirect(
        f"/profiles/{profile_id}",
        ok="Roteador atualizado. O router aplica a mudança em poucos segundos.",
    )


@router.post("/profiles/{profile_id}/delete")
async def profiles_delete(profile_id: int, console: Console = Depends(get_console)):
    try:
        await console.run_db(lambda s: repo.delete_profile(s, profile_id))
    except SwitchboardError as exc:
        return redirect("/profiles", erro=str(exc))
    return redirect("/profiles", ok="Roteador removido.")
