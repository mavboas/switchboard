"""API admin em JSON (a mesma configuração da UI, para automação e config-as-code).

Os segredos (chave de API, token) são só de escrita: as respostas mostram
apenas se há valor e de que tipo (``env:NOME`` ou cifrado).

Recursos: ``/api/models``, ``/api/connectors`` (servidores MCP),
``/api/agents`` (agentes A2A), ``/api/knowledge-bases``, ``/api/profiles``,
``/api/traces`` (execuções, com spans e contratos) e ``/api/contracts``.
"""

from __future__ import annotations

from typing import Any

import anyio.to_thread
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select

from switchboard.errors import SwitchboardError
from switchboard.llm import PRESETS
from switchboard.rag import extract_text, guess_title
from switchboard.secrets import SecretBox
from switchboard.storage import repo
from switchboard.storage.orm import (
    Agent,
    Connector,
    Document,
    KnowledgeBase,
    LlmModel,
    RouterProfile,
)

from .state import Console, get_console

api = APIRouter(prefix="/api", tags=["admin"])


def _fail(exc: Exception, status: int = 400) -> HTTPException:
    return HTTPException(status_code=status, detail=str(exc))


def _model_out(m: LlmModel) -> dict[str, Any]:
    return {
        "id": m.id,
        "name": m.name,
        "preset": m.preset,
        "provider": m.provider,
        "model": m.model,
        "base_url": m.base_url,
        "api_key": SecretBox.describe(m.api_key) if m.api_key else None,
        "api_key_header": m.api_key_header,
        "extra_headers": {k: "•••" for k in (m.extra_headers or {})},
        "temperature": m.temperature,
        "max_tokens": m.max_tokens,
        "timeout_s": m.timeout_s,
        "json_mode": m.json_mode,
    }


def _connector_out(c: Connector) -> dict[str, Any]:
    return {
        "id": c.id,
        "name": c.name,
        "description": c.description,
        "url": c.url,
        "transport": c.transport,
        "auth_token": SecretBox.describe(c.auth_token) if c.auth_token else None,
        "allowed_tools": c.allowed_tools or [],
        "enabled": c.enabled,
        "timeout_s": c.timeout_s,
    }


def _agent_out(a: Agent) -> dict[str, Any]:
    return {
        "id": a.id,
        "name": a.name,
        "description": a.description,
        "url": a.url,
        "auth_token": SecretBox.describe(a.auth_token) if a.auth_token else None,
        "allowed_skills": a.allowed_skills or [],
        "enabled": a.enabled,
        "timeout_s": a.timeout_s,
        "deadline_s": a.deadline_s,
        "push": True if a.push is None else a.push,
        "allow_cross_origin": bool(a.allow_cross_origin),
    }


def _kb_out(k: KnowledgeBase, stats: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        "id": k.id,
        "name": k.name,
        "description": k.description,
        "embedder": repo.kb_label(k),
        "embedder_kind": k.embedder_kind,
        "embedder_dim": k.embedder_dim,
        "embedding_connection": k.embedding_connection.name if k.embedding_connection else None,
        "embedding_model": k.embedding_model,
        "chunk_size": k.chunk_size,
        "chunk_overlap": k.chunk_overlap,
        "stats": stats,
    }


def _profile_out(p: RouterProfile) -> dict[str, Any]:
    spec = repo.profile_to_spec(p, only_enabled=False)
    return {
        "id": p.id,
        "name": p.name,
        "description": p.description,
        "model": p.model.name,
        "decision_model": p.decision_model.name if p.decision_model else None,
        "decision_threshold": spec.decision_threshold,
        "system_prompt": p.system_prompt,
        "connectors": [c.name for c in p.connectors],
        "agents": [a.name for a in p.agents],
        "knowledge_bases": [k.name for k in p.knowledge_bases],
        "top_k": p.top_k,
        "min_score": p.min_score,
        "synthesize": p.synthesize,
        "allow_clarify": p.allow_clarify,
        "wait_s": spec.wait_s,
        "max_parallel": spec.max_parallel,
        "deadline_s": spec.deadline_s,
        "enabled": p.enabled,
    }


# -- modelos ------------------------------------------------------------------


class ModelIn(BaseModel):
    name: str
    provider: str = "openai"
    preset: str | None = None
    model: str = ""
    base_url: str | None = None
    api_key: str | None = Field(
        default=None, description="texto (será cifrado) ou env:NOME; omita para manter"
    )
    clear_api_key: bool = False
    api_key_header: str = "Authorization"
    extra_headers: dict[str, str] = Field(default_factory=dict)
    temperature: float | None = 0.2
    max_tokens: int | None = 4096
    timeout_s: float = 60.0
    json_mode: bool = True


@api.get("/presets")
async def presets() -> list[dict[str, str]]:
    return [p.to_dict() for p in PRESETS.values()]


@api.get("/models")
async def list_models(console: Console = Depends(get_console)):
    return await console.run_db(
        lambda s: [_model_out(m) for m in s.scalars(select(LlmModel).order_by(LlmModel.name))]
    )


@api.post("/models", status_code=201)
async def create_model(body: ModelIn, console: Console = Depends(get_console)):
    try:
        return await console.run_db(
            lambda s: _model_out(repo.save_model(s, body.model_dump(), box=console.box))
        )
    except SwitchboardError as exc:
        raise _fail(exc) from exc


@api.put("/models/{model_id}")
async def update_model(model_id: int, body: ModelIn, console: Console = Depends(get_console)):
    try:
        return await console.run_db(
            lambda s: _model_out(
                repo.save_model(s, body.model_dump(), box=console.box, model_id=model_id)
            )
        )
    except SwitchboardError as exc:
        raise _fail(exc) from exc


@api.delete("/models/{model_id}", status_code=204)
async def delete_model(model_id: int, console: Console = Depends(get_console)):
    try:
        await console.run_db(lambda s: repo.delete_model(s, model_id))
    except SwitchboardError as exc:
        raise _fail(exc, 409) from exc


# -- conectores MCP -------------------------------------------------------------


class ConnectorIn(BaseModel):
    name: str
    description: str = ""
    url: str
    transport: str = "streamable-http"
    auth_token: str | None = None
    clear_auth_token: bool = False
    allowed_tools: list[str] = Field(default_factory=list)
    enabled: bool = True
    timeout_s: float = 30.0


class ToolCallIn(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


def _connector_spec(s, connector_id: int):
    row = s.get(Connector, connector_id)
    return repo.connector_to_spec(row) if row is not None else None


@api.get("/connectors")
async def list_connectors(console: Console = Depends(get_console)):
    return await console.run_db(
        lambda s: [_connector_out(c) for c in s.scalars(select(Connector).order_by(Connector.name))]
    )


@api.post("/connectors", status_code=201)
async def create_connector(body: ConnectorIn, console: Console = Depends(get_console)):
    try:
        return await console.run_db(
            lambda s: _connector_out(repo.save_connector(s, body.model_dump(), box=console.box))
        )
    except SwitchboardError as exc:
        raise _fail(exc) from exc


@api.put("/connectors/{connector_id}")
async def update_connector(
    connector_id: int, body: ConnectorIn, console: Console = Depends(get_console)
):
    try:
        out = await console.run_db(
            lambda s: _connector_out(
                repo.save_connector(
                    s, body.model_dump(), box=console.box, connector_id=connector_id
                )
            )
        )
    except SwitchboardError as exc:
        raise _fail(exc) from exc
    console.connectors.invalidate(out["name"])
    return out


@api.delete("/connectors/{connector_id}", status_code=204)
async def delete_connector(connector_id: int, console: Console = Depends(get_console)):
    try:
        await console.run_db(lambda s: repo.delete_connector(s, connector_id))
    except SwitchboardError as exc:
        raise _fail(exc, 404) from exc


@api.get("/connectors/{connector_id}/discover")
async def discover_connector(
    connector_id: int, refresh: bool = True, console: Console = Depends(get_console)
):
    spec = await console.run_db(_connector_spec, connector_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="conector não encontrado")
    return (await console.connectors.describe(spec, refresh=refresh)).to_dict()


@api.post("/connectors/{connector_id}/call")
async def call_tool(connector_id: int, body: ToolCallIn, console: Console = Depends(get_console)):
    spec = await console.run_db(_connector_spec, connector_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="conector não encontrado")
    try:
        result = await console.connectors.call(spec, body.tool, body.arguments)
    except SwitchboardError as exc:
        raise _fail(exc, 502) from exc
    return {
        "text": result.text,
        "is_error": result.is_error,
        "structured": result.structured,
        "latency_ms": round(result.latency_ms, 1),
    }


# -- agentes A2A -------------------------------------------------------------------


class AgentIn(BaseModel):
    model_config = {"extra": "allow"}

    name: str
    description: str = ""
    url: str = Field(description="URL base do agente (onde fica /.well-known/agent-card.json)")
    auth_token: str | None = None
    clear_auth_token: bool = False
    allowed_skills: list[str] = Field(default_factory=list)
    enabled: bool = True
    timeout_s: float = 15.0
    deadline_s: float | None = Field(
        default=None, description="prazo padrão dos contratos, em segundos"
    )
    push: bool = True
    allow_cross_origin: bool = False

    def payload(self) -> dict[str, Any]:
        extra = set(self.model_extra or {})
        if extra & {"transport", "allowed_tools"}:
            # cliente da API antiga: agentes MCP viraram conectores
            raise SwitchboardError(
                "isto parece um servidor MCP: cadastre-o em /api/connectors "
                "(/api/agents agora é para agentes A2A)"
            )
        return self.model_dump(exclude=extra)


def _agent_spec(s, agent_id: int):
    row = s.get(Agent, agent_id)
    return repo.agent_to_spec(row) if row is not None else None


@api.get("/agents")
async def list_agents(console: Console = Depends(get_console)):
    return await console.run_db(
        lambda s: [_agent_out(a) for a in s.scalars(select(Agent).order_by(Agent.name))]
    )


@api.post("/agents", status_code=201)
async def create_agent(body: AgentIn, console: Console = Depends(get_console)):
    try:
        data = body.payload()
        return await console.run_db(lambda s: _agent_out(repo.save_agent(s, data, box=console.box)))
    except SwitchboardError as exc:
        raise _fail(exc) from exc


@api.put("/agents/{agent_id}")
async def update_agent(agent_id: int, body: AgentIn, console: Console = Depends(get_console)):
    try:
        data = body.payload()
        out = await console.run_db(
            lambda s: _agent_out(repo.save_agent(s, data, box=console.box, agent_id=agent_id))
        )
    except SwitchboardError as exc:
        raise _fail(exc) from exc
    console.agents.invalidate(out["name"])
    return out


@api.delete("/agents/{agent_id}", status_code=204)
async def delete_agent(agent_id: int, console: Console = Depends(get_console)):
    try:
        await console.run_db(lambda s: repo.delete_agent(s, agent_id))
    except SwitchboardError as exc:
        raise _fail(exc, 404) from exc


@api.get("/agents/{agent_id}/discover")
async def discover_agent(
    agent_id: int, refresh: bool = True, console: Console = Depends(get_console)
):
    """Busca o Agent Card e mostra skills, contrato (completo/básico) e problemas."""
    spec = await console.run_db(_agent_spec, agent_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="agente não encontrado")
    return (await console.agents.describe(spec, refresh=refresh)).to_dict()


# -- bases de conhecimento ------------------------------------------------------


class KnowledgeIn(BaseModel):
    name: str
    description: str = ""
    embedder_kind: str = "hashing"
    embedder_dim: int = 512
    embedding_connection: str | None = Field(
        default=None, description="nome do modelo que gera os embeddings"
    )
    embedding_model: str | None = None
    chunk_size: int = 800
    chunk_overlap: int = 120


class DocumentIn(BaseModel):
    title: str
    content: str
    source: str = "api"


class SearchIn(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.0


def _kb_payload(s, body: KnowledgeIn) -> dict[str, Any]:
    data = body.model_dump()
    connection = data.pop("embedding_connection")
    if connection:
        model = s.scalars(select(LlmModel).where(LlmModel.name == connection)).first()
        if model is None:
            raise SwitchboardError(f"modelo '{connection}' não existe")
        data["embedding_model_id"] = model.id
    return data


@api.get("/knowledge-bases")
async def list_kbs(console: Console = Depends(get_console)):
    def load(s):
        stats = repo.kb_stats(s)
        return [
            _kb_out(k, stats.get(k.id))
            for k in s.scalars(select(KnowledgeBase).order_by(KnowledgeBase.name))
        ]

    return await console.run_db(load)


@api.post("/knowledge-bases", status_code=201)
async def create_kb(body: KnowledgeIn, console: Console = Depends(get_console)):
    try:
        return await console.run_db(
            lambda s: _kb_out(repo.save_knowledge_base(s, _kb_payload(s, body)))
        )
    except SwitchboardError as exc:
        raise _fail(exc) from exc


@api.put("/knowledge-bases/{kb_id}")
async def update_kb(kb_id: int, body: KnowledgeIn, console: Console = Depends(get_console)):
    try:
        return await console.run_db(
            lambda s: _kb_out(repo.save_knowledge_base(s, _kb_payload(s, body), kb_id=kb_id))
        )
    except SwitchboardError as exc:
        raise _fail(exc) from exc


@api.delete("/knowledge-bases/{kb_id}", status_code=204)
async def delete_kb(kb_id: int, console: Console = Depends(get_console)):
    try:
        await console.run_db(lambda s: repo.delete_knowledge_base(s, kb_id))
    except SwitchboardError as exc:
        raise _fail(exc, 404) from exc


@api.get("/knowledge-bases/{kb_id}/documents")
async def list_documents(kb_id: int, console: Console = Depends(get_console)):
    def load(s):
        docs = s.scalars(
            select(Document).where(Document.kb_id == kb_id).order_by(Document.created_at.desc())
        )
        return [
            {
                "id": d.id,
                "title": d.title,
                "source": d.source,
                "chunks": d.chunk_count,
                "embedder": d.embedder,
            }
            for d in docs
        ]

    return await console.run_db(load)


@api.post("/knowledge-bases/{kb_id}/documents", status_code=201)
async def add_document(kb_id: int, body: DocumentIn, console: Console = Depends(get_console)):
    try:
        doc_id, created = await console.knowledge.add_document(
            kb_id, title=body.title, content=body.content, source=body.source
        )
    except SwitchboardError as exc:
        raise _fail(exc) from exc
    return {"id": doc_id, "created": created}


@api.post("/knowledge-bases/{kb_id}/files", status_code=201)
async def upload_file(
    kb_id: int, file: UploadFile = File(...), console: Console = Depends(get_console)
):
    limit = console.settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if (file.size or 0) > limit or len(data) > limit:
        raise HTTPException(
            status_code=413, detail=f"arquivo maior que {console.settings.max_upload_mb} MB"
        )
    try:
        text = await anyio.to_thread.run_sync(extract_text, file.filename or "arquivo.txt", data)
        doc_id, created = await console.knowledge.add_document(
            kb_id,
            title=guess_title(file.filename or "arquivo", text),
            content=text,
            source=file.filename or "upload",
        )
    except SwitchboardError as exc:
        raise _fail(exc) from exc
    return {"id": doc_id, "created": created}


@api.delete("/knowledge-bases/{kb_id}/documents/{doc_id}", status_code=204)
async def delete_document(kb_id: int, doc_id: int, console: Console = Depends(get_console)):
    try:
        await console.knowledge.delete_document(kb_id, doc_id)
    except SwitchboardError as exc:
        raise _fail(exc, 404) from exc


@api.post("/knowledge-bases/{kb_id}/search")
async def search(kb_id: int, body: SearchIn, console: Console = Depends(get_console)):
    name = await console.run_db(lambda s: (k := s.get(KnowledgeBase, kb_id)) and k.name)
    if name is None:
        raise HTTPException(status_code=404, detail="base não encontrada")
    hits = await console.knowledge.search(
        body.query, [name], top_k=body.top_k, min_score=body.min_score
    )
    return [
        {
            "document": h.document,
            "section": h.section,
            "score": round(h.score, 4),
            "content": h.content,
        }
        for h in hits
    ]


@api.post("/knowledge-bases/{kb_id}/reindex")
async def reindex(kb_id: int, console: Console = Depends(get_console)):
    try:
        return {"chunks": await console.knowledge.reindex(kb_id)}
    except SwitchboardError as exc:
        raise _fail(exc) from exc


# -- roteadores ---------------------------------------------------------------


class ProfileIn(BaseModel):
    name: str
    description: str = ""
    model: str = Field(description="nome do modelo (LLM)")
    decision_model: str | None = Field(
        default=None, description="nome do modelo de decisão (TypeSafe Jev); vazio = o LLM decide"
    )
    decision_threshold: float = 0.6
    system_prompt: str = ""
    connectors: list[str] = Field(default_factory=list, description="conectores MCP")
    agents: list[str] = Field(default_factory=list, description="agentes A2A")
    knowledge_bases: list[str] = Field(default_factory=list)
    top_k: int = 4
    min_score: float = 0.2
    synthesize: bool = True
    allow_clarify: bool = True
    wait_s: float = 8.0
    max_parallel: int = 3
    deadline_s: float = 600.0
    enabled: bool = True


def _profile_payload(s, body: ProfileIn) -> dict[str, Any]:
    def ids(cls, names: list[str], label: str) -> list[int]:
        rows = (
            {r.name: r.id for r in s.scalars(select(cls).where(cls.name.in_(names)))}
            if names
            else {}
        )
        missing = [n for n in names if n not in rows]
        if missing:
            raise SwitchboardError(f"{label} inexistente(s): {', '.join(missing)}")
        return [rows[n] for n in names]

    [model_id] = ids(LlmModel, [body.model], "modelo")
    decision_ids = (
        ids(LlmModel, [body.decision_model], "modelo de decisão") if body.decision_model else []
    )
    data = body.model_dump(
        exclude={"model", "decision_model", "connectors", "agents", "knowledge_bases"}
    )
    data.update(
        model_id=model_id,
        decision_model_id=decision_ids[0] if decision_ids else None,
        connector_ids=ids(Connector, body.connectors, "conector"),
        agent_ids=ids(Agent, body.agents, "agente"),
        kb_ids=ids(KnowledgeBase, body.knowledge_bases, "base"),
    )
    return data


@api.get("/profiles")
async def list_profiles(console: Console = Depends(get_console)):
    return await console.run_db(lambda s: [_profile_out(p) for p in repo.list_profiles(s)])


@api.post("/profiles", status_code=201)
async def create_profile(body: ProfileIn, console: Console = Depends(get_console)):
    try:
        return await console.run_db(
            lambda s: _profile_out(repo.save_profile(s, _profile_payload(s, body)))
        )
    except SwitchboardError as exc:
        raise _fail(exc) from exc


@api.put("/profiles/{profile_id}")
async def update_profile(profile_id: int, body: ProfileIn, console: Console = Depends(get_console)):
    try:
        return await console.run_db(
            lambda s: _profile_out(
                repo.save_profile(s, _profile_payload(s, body), profile_id=profile_id)
            )
        )
    except SwitchboardError as exc:
        raise _fail(exc) from exc


@api.delete("/profiles/{profile_id}", status_code=204)
async def delete_profile(profile_id: int, console: Console = Depends(get_console)):
    try:
        await console.run_db(lambda s: repo.delete_profile(s, profile_id))
    except SwitchboardError as exc:
        raise _fail(exc, 404) from exc


# -- execuções e contratos ------------------------------------------------------------


@api.get("/traces")
async def list_traces(
    profile: str | None = None,
    route: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    console: Console = Depends(get_console),
):
    limit = max(1, min(limit, 500))
    return await console.run_db(
        lambda s: [
            repo.trace_to_dict(t)
            for t in repo.query_traces(
                s, profile=profile, route=route, status=status, limit=limit, offset=offset
            )
        ]
    )


@api.get("/traces/{trace_id}")
async def get_trace(trace_id: str, console: Console = Depends(get_console)):
    """A execução completa: decisão, spans (inclusive um por contrato) e contratos com eventos."""
    row = await console.run_db(repo.run_details, trace_id)
    if row is None:
        raise HTTPException(status_code=404, detail="execução não encontrada")
    return row


@api.get("/contracts")
async def list_contracts(
    state: str | None = None,
    agent: str | None = None,
    profile: str | None = None,
    limit: int = 50,
    offset: int = 0,
    console: Console = Depends(get_console),
):
    limit = max(1, min(limit, 500))
    return await console.run_db(
        lambda s: repo.query_contracts(
            s, state=state, agent=agent, profile=profile, limit=limit, offset=offset
        )
    )


@api.get("/contracts/{contract_id}")
async def get_contract(contract_id: str, console: Console = Depends(get_console)):
    """Termos (schemas e hashes), entrada, saída e a linha do tempo de eventos do contrato."""
    row = await console.run_db(repo.contract_details, contract_id)
    if row is None:
        raise HTTPException(status_code=404, detail="contrato não encontrado")
    return row
