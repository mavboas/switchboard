"""Páginas do console (HTML renderizado no servidor)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy import select

from switchboard.config import DEFAULT_SYSTEM_PROMPT
from switchboard.errors import SwitchboardError
from switchboard.llm import PRESETS, Message, build_chat_model
from switchboard.rag import SUPPORTED_EXTENSIONS, extract_text, guess_title
from switchboard.storage import repo
from switchboard.storage.orm import Agent, Document, KnowledgeBase, LlmModel, RouterProfile, Trace

from . import forms
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
        "extra_headers": forms.headers_text(row.extra_headers),
        "temperature": row.temperature,
        "max_tokens": row.max_tokens,
        "timeout_s": row.timeout_s,
        "json_mode": row.json_mode,
        "used_by": used_by or [],
    }


def _agent_dict(row: Agent) -> dict[str, Any]:
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
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "model_id": row.model_id,
        "model_name": row.model.name,
        "model_label": f"{row.model.provider}:{row.model.model or row.model.name}",
        "system_prompt": row.system_prompt,
        "agent_ids": [a.id for a in row.agents],
        "agents": [a.name for a in row.agents],
        "kb_ids": [k.id for k in row.knowledge_bases],
        "kbs": [k.name for k in row.knowledge_bases],
        "top_k": row.top_k,
        "min_score": row.min_score,
        "synthesize": row.synthesize,
        "allow_clarify": row.allow_clarify,
        "enabled": row.enabled,
    }


def _trace_row(row: Trace) -> dict[str, Any]:
    data = repo.trace_to_dict(row)
    data["created"] = row.created_at
    return data


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
                "agents": repo.count(s, Agent),
                "kbs": repo.count(s, KnowledgeBase),
                "profiles": repo.count(s, RouterProfile),
                "traces": repo.count(s, Trace),
                "documents": sum(v["documents"] for v in stats.values()),
                "chunks": sum(v["chunks"] for v in stats.values()),
            },
            "routes": repo.route_counts(s),
            "recent": [_trace_row(t) for t in repo.query_traces(s, limit=8)],
            "profiles": [_profile_dict(p) for p in repo.list_profiles(s)],
            "agent_specs": [
                repo.agent_to_spec(a) for a in s.scalars(select(Agent).order_by(Agent.name))
            ],
        }

    data = await console.run_db(load)
    health, infos = await asyncio.gather(
        _router_health(console), console.catalog.discover(data["agent_specs"])
    )
    counts = data["counts"]
    steps = [
        ("Cadastre um modelo (LLM)", counts["models"] > 0, "/models/new"),
        ("Registre os agentes MCP", counts["agents"] > 0, "/agents/new"),
        ("Crie uma base de conhecimento", counts["documents"] > 0, "/knowledge/new"),
        ("Monte um roteador", counts["profiles"] > 0, "/profiles/new"),
        ("Teste no playground", counts["traces"] > 0, "/playground"),
    ]
    return console.render(
        request,
        "dashboard.html",
        nav="dashboard",
        data=data,
        health=health,
        agents=infos,
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
    values = {
        "preset": p.key,
        "provider": p.provider,
        "base_url": p.base_url,
        "api_key_header": p.api_key_header,
        "temperature": 0.2,
        "max_tokens": 1024,
        "timeout_s": 60,
        "json_mode": True,
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
    try:
        chat = build_chat_model(spec, resolve_secret=console.box.open)
        try:
            result = await chat.chat(
                [Message("user", "Teste de conexão do Switchboard. Responda apenas: ok")],
                max_tokens=20,
            )
        finally:
            await chat.aclose()
    except SwitchboardError as exc:
        return redirect("/models", erro=f"Falha no teste de '{spec.name}': {exc}")
    reply = result.text.strip().replace("\n", " ")[:120]
    return redirect("/models", ok=f"'{spec.name}' respondeu em {result.latency_ms:.0f} ms: {reply}")


# ---------------------------------------------------------------------------
# agentes


@router.get("/agents")
async def agents_list(request: Request, console: Console = Depends(get_console)):
    def load(s):
        rows = s.scalars(select(Agent).order_by(Agent.name)).all()
        return [(_agent_dict(a), repo.agent_to_spec(a)) for a in rows]

    rows = await console.run_db(load)
    infos = await asyncio.gather(
        *(console.catalog.describe(spec) for _, spec in rows if spec.enabled)
    )
    by_name = {i.name: i for i in infos}
    agents = [{**data, "info": by_name.get(data["name"])} for data, _ in rows]
    return console.render(request, "agents/list.html", nav="agents", agents=agents)


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
    return _agent_form(
        console, request, {"transport": "streamable-http", "enabled": True, "timeout_s": 30}
    )


@router.post("/agents")
async def agents_create(request: Request, console: Console = Depends(get_console)):
    data = forms.agent_data(await request.form())
    try:
        agent_id = await console.run_db(lambda s: repo.save_agent(s, data, box=console.box).id)
    except SwitchboardError as exc:
        return _agent_form(console, request, {**data, "auth_token": None}, str(exc))
    return redirect(
        f"/agents/{agent_id}",
        ok="Agente registrado. Veja abaixo o que a descoberta via MCP encontrou.",
    )


async def _agent_detail(
    console: Console,
    request: Request,
    agent_id: int,
    *,
    refresh: bool = False,
    call: dict | None = None,
):
    row = await console.run_db(
        lambda s: (a := s.get(Agent, agent_id)) and (_agent_dict(a), repo.agent_to_spec(a))
    )
    if row is None:
        return redirect("/agents", erro="Agente não encontrado.")
    data, spec = row
    info = await console.catalog.describe(spec, refresh=refresh)
    return console.render(
        request, "agents/detail.html", nav="agents", agent=data, info=info, call=call
    )


@router.get("/agents/{agent_id}")
async def agents_detail(
    request: Request, agent_id: int, refresh: bool = False, console: Console = Depends(get_console)
):
    return await _agent_detail(console, request, agent_id, refresh=refresh)


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
    console.catalog.invalidate(data["name"])
    return redirect(f"/agents/{agent_id}", ok="Agente atualizado.")


@router.post("/agents/{agent_id}/delete")
async def agents_delete(agent_id: int, console: Console = Depends(get_console)):
    try:
        await console.run_db(lambda s: repo.delete_agent(s, agent_id))
    except SwitchboardError as exc:
        return redirect("/agents", erro=str(exc))
    return redirect("/agents", ok="Agente removido.")


@router.post("/agents/{agent_id}/call")
async def agents_call(request: Request, agent_id: int, console: Console = Depends(get_console)):
    form = await request.form()
    tool = forms.text(form, "tool")
    raw = forms.text(form, "arguments") or "{}"
    call: dict[str, Any] = {"tool": tool, "arguments": raw}
    spec = await console.run_db(lambda s: (a := s.get(Agent, agent_id)) and repo.agent_to_spec(a))
    if spec is None:
        return redirect("/agents", erro="Agente não encontrado.")
    try:
        arguments = json.loads(raw)
        if not isinstance(arguments, dict):
            raise ValueError("os argumentos precisam ser um objeto JSON")
        outcome = await console.catalog.call(spec, tool, arguments)
        call.update(result=outcome.text, is_error=outcome.is_error, ms=outcome.latency_ms)
    except (ValueError, SwitchboardError) as exc:
        call.update(result=str(exc), is_error=True, ms=0)
    return await _agent_detail(console, request, agent_id, call=call)


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
        data = await upload.read()
        if len(data) > limit:
            errors.append(f"{upload.filename}: maior que {console.settings.max_upload_mb} MB")
            continue
        try:
            text = extract_text(upload.filename, data)
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
        return (
            [
                {"id": m.id, "name": m.name, "label": f"{m.provider}:{m.model or m.name}"}
                for m in s.scalars(select(LlmModel).order_by(LlmModel.name))
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

    models, agents, kbs = await console.run_db(load)
    return console.render(
        request,
        "profiles/form.html",
        nav="profiles",
        values=values,
        error=error,
        models=models,
        agents=agents,
        kbs=kbs,
    )


@router.get("/profiles/new")
async def profiles_new(request: Request, console: Console = Depends(get_console)):
    values = {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "top_k": 4,
        "min_score": 0.2,
        "synthesize": True,
        "allow_clarify": True,
        "enabled": True,
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


# ---------------------------------------------------------------------------
# playground


@router.get("/playground")
async def playground(
    request: Request, profile: str | None = None, console: Console = Depends(get_console)
):
    names = await console.run_db(repo.enabled_profile_names)
    selected = profile if profile in names else (names[0] if names else None)
    return console.render(
        request, "playground.html", nav="playground", profiles=names, selected=selected
    )


@router.post("/playground/send")
async def playground_send(request: Request, console: Console = Depends(get_console)) -> Response:
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": "JSON inválido"}, status_code=400)
    payload = {"profile": body.get("profile"), "messages": body.get("messages") or []}
    try:
        resp = await console.router_request("POST", "/v1/chat", json=payload)
    except httpx.HTTPError as exc:
        return JSONResponse(
            {
                "error": f"router inacessível em {console.settings.router_url} ({exc.__class__.__name__})"
            },
            status_code=502,
        )
    try:
        data = resp.json()
    except ValueError:
        data = {"error": resp.text[:500]}
    if resp.status_code >= 400:
        detail = data.get("detail") if isinstance(data, dict) else None
        return JSONResponse({"error": detail or data}, status_code=resp.status_code)
    return JSONResponse(data)


# ---------------------------------------------------------------------------
# execuções


@router.get("/traces")
async def traces_list(
    request: Request,
    profile: str | None = None,
    route: str | None = None,
    page: int = 1,
    console: Console = Depends(get_console),
):
    page = max(page, 1)
    size = 25

    def load(s):
        rows = repo.query_traces(
            s,
            profile=profile or None,
            route=route or None,
            limit=size + 1,
            offset=(page - 1) * size,
        )
        return [_trace_row(t) for t in rows], repo.enabled_profile_names(s)

    rows, names = await console.run_db(load)
    return console.render(
        request,
        "traces/list.html",
        nav="traces",
        traces=rows[:size],
        has_next=len(rows) > size,
        page=page,
        profiles=names,
        filters={"profile": profile or "", "route": route or ""},
    )


@router.get("/traces/{trace_id}")
async def traces_detail(request: Request, trace_id: str, console: Console = Depends(get_console)):
    row = await console.run_db(lambda s: (t := s.get(Trace, trace_id)) and _trace_row(t))
    if row is None:
        return redirect("/traces", erro="Execução não encontrada.")
    return console.render(request, "traces/detail.html", nav="traces", trace=row)
