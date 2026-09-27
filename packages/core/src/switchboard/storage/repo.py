"""Leitura e escrita da configuração no banco.

Toda escrita passa pelas mesmas specs Pydantic do modo YAML (``ModelSpec``,
``AgentSpec``…), então as regras de validação são idênticas nos dois modos.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from ..config import (
    DEFAULT_SYSTEM_PROMPT,
    AgentSpec,
    EmbedderSpec,
    KnowledgeBaseSpec,
    ModelSpec,
    ProfileSpec,
    format_validation_error,
)
from ..errors import ConfigError
from ..routing.types import ResolvedProfile, RouterResult
from ..secrets import ENC_PREFIX, ENV_PREFIX, MASK, SecretBox
from .orm import Agent, Chunk, Document, KnowledgeBase, LlmModel, RouterProfile, Trace

# --------------------------------------------------------------------------
# linhas -> specs


def model_to_spec(row: LlmModel) -> ModelSpec:
    return ModelSpec(
        name=row.name,
        provider=row.provider,  # type: ignore[arg-type]
        model=row.model or "",
        base_url=row.base_url or None,
        api_key=row.api_key,
        api_key_header=row.api_key_header or "Authorization",
        extra_headers=dict(row.extra_headers or {}),
        temperature=row.temperature,
        max_tokens=row.max_tokens,
        timeout_s=row.timeout_s,
        json_mode=row.json_mode,
    )


def agent_to_spec(row: Agent) -> AgentSpec:
    return AgentSpec(
        name=row.name,
        description=row.description or "",
        url=row.url,
        transport=row.transport,  # type: ignore[arg-type]
        auth_token=row.auth_token,
        allowed_tools=list(row.allowed_tools or []),
        enabled=row.enabled,
        timeout_s=row.timeout_s,
    )


def embedder_of(row: KnowledgeBase) -> tuple[EmbedderSpec, ModelSpec | None]:
    if row.embedder_kind == "model" and row.embedding_connection is not None:
        spec = EmbedderSpec(
            kind="model", connection=row.embedding_connection.name, model=row.embedding_model
        )
        return spec, model_to_spec(row.embedding_connection)
    return EmbedderSpec(kind="hashing", dim=row.embedder_dim or 512), None


def kb_to_spec(row: KnowledgeBase) -> KnowledgeBaseSpec:
    embedder, _ = embedder_of(row)
    return KnowledgeBaseSpec(
        name=row.name,
        description=row.description or "",
        embedder=embedder,
        chunk_size=row.chunk_size,
        chunk_overlap=row.chunk_overlap,
    )


def profile_to_spec(row: RouterProfile, *, only_enabled_agents: bool = True) -> ProfileSpec:
    agents = [a.name for a in row.agents if a.enabled or not only_enabled_agents]
    return ProfileSpec(
        name=row.name,
        description=row.description or "",
        model=row.model.name,
        system_prompt=row.system_prompt or DEFAULT_SYSTEM_PROMPT,
        agents=agents,
        knowledge_bases=[k.name for k in row.knowledge_bases],
        top_k=row.top_k,
        min_score=row.min_score,
        synthesize=row.synthesize,
        allow_clarify=row.allow_clarify,
    )


def _profile_query():
    return select(RouterProfile).options(
        selectinload(RouterProfile.agents), selectinload(RouterProfile.knowledge_bases)
    )


def resolve_profile(session: Session, name: str) -> ResolvedProfile | None:
    row = session.scalars(_profile_query().where(RouterProfile.name == name)).first()
    if row is None or not row.enabled:
        return None
    return ResolvedProfile(
        spec=profile_to_spec(row),
        model=model_to_spec(row.model),
        agents=[agent_to_spec(a) for a in row.agents if a.enabled],
    )


def enabled_profile_names(session: Session) -> list[str]:
    return list(
        session.scalars(
            select(RouterProfile.name)
            .where(RouterProfile.enabled.is_(True))
            .order_by(RouterProfile.name)
        )
    )


def list_profiles(session: Session) -> list[RouterProfile]:
    return list(session.scalars(_profile_query().order_by(RouterProfile.name)))


def get_profile(session: Session, profile_id: int) -> RouterProfile | None:
    return session.scalars(_profile_query().where(RouterProfile.id == profile_id)).first()


# --------------------------------------------------------------------------
# escrita


def _validate(spec_cls, data: dict[str, Any]):
    try:
        return spec_cls.model_validate(data)
    except ValidationError as exc:
        raise ConfigError("dados inválidos:\n" + format_validation_error(exc)) from exc


def _ensure_unique(session: Session, cls, name: str, current_id: int | None) -> None:
    existing = session.scalars(select(cls).where(cls.name == name)).first()
    if existing is not None and existing.id != current_id:
        raise ConfigError(f"já existe um registro chamado '{name}'")


def _secret(box: SecretBox, new_value: str | None, current: str | None, clear: bool) -> str | None:
    if clear:
        return None
    if new_value is None or not str(new_value).strip():
        return current
    return box.seal(str(new_value))


def _headers(box: SecretBox, new: dict[str, str], current: dict[str, str]) -> dict[str, str]:
    """Cabeçalhos extras: ``env:`` validado; texto cifrado quando há chave mestra."""
    out: dict[str, str] = {}
    for name, value in new.items():
        value = (value or "").strip()
        if value == MASK and name in current:
            out[name] = current[name]  # a UI mostrou a máscara: mantém o valor gravado
        elif not value:
            out[name] = ""
        elif value.startswith((ENV_PREFIX, ENC_PREFIX)) or box.enabled:
            out[name] = box.seal(value) or ""
        else:
            out[name] = value  # sem chave mestra, cabeçalhos comuns ficam em texto
    return out


def save_model(
    session: Session, data: dict[str, Any], *, box: SecretBox, model_id: int | None = None
) -> LlmModel:
    row = session.get(LlmModel, model_id) if model_id else LlmModel()
    if row is None:
        raise ConfigError("modelo não encontrado")
    api_key = _secret(box, data.get("api_key"), row.api_key, bool(data.get("clear_api_key")))
    fields = {
        "name": str(data.get("name") or "").strip(),
        "provider": data.get("provider") or "openai",
        "model": str(data.get("model") or "").strip(),
        "base_url": (str(data.get("base_url") or "").strip() or None),
        "api_key": api_key,
        "api_key_header": str(data.get("api_key_header") or "Authorization").strip(),
        "extra_headers": _headers(box, data.get("extra_headers") or {}, row.extra_headers or {}),
        "temperature": data.get("temperature"),
        "max_tokens": data.get("max_tokens"),
        "timeout_s": data.get("timeout_s") or 60.0,
        "json_mode": bool(data.get("json_mode", True)),
    }
    spec = _validate(ModelSpec, fields)
    _ensure_unique(session, LlmModel, spec.name, row.id)
    for key, value in spec.model_dump().items():
        setattr(row, key, value)
    row.preset = data.get("preset") or None
    session.add(row)
    session.flush()
    return row


def delete_model(session: Session, model_id: int) -> None:
    row = session.get(LlmModel, model_id)
    if row is None:
        raise ConfigError("modelo não encontrado")
    profiles = list(
        session.scalars(select(RouterProfile.name).where(RouterProfile.model_id == model_id))
    )
    if profiles:
        raise ConfigError(f"o modelo está em uso pelos roteadores: {', '.join(profiles)}")
    kbs = list(
        session.scalars(
            select(KnowledgeBase.name).where(KnowledgeBase.embedding_model_id == model_id)
        )
    )
    if kbs:
        raise ConfigError(f"o modelo é a conexão de embeddings das bases: {', '.join(kbs)}")
    session.delete(row)


def save_agent(
    session: Session, data: dict[str, Any], *, box: SecretBox, agent_id: int | None = None
) -> Agent:
    row = session.get(Agent, agent_id) if agent_id else Agent()
    if row is None:
        raise ConfigError("agente não encontrado")
    token = _secret(box, data.get("auth_token"), row.auth_token, bool(data.get("clear_auth_token")))
    tools = data.get("allowed_tools") or []
    if isinstance(tools, str):
        tools = [t.strip() for t in tools.replace("\n", ",").split(",") if t.strip()]
    fields = {
        "name": str(data.get("name") or "").strip(),
        "description": str(data.get("description") or "").strip(),
        "url": str(data.get("url") or "").strip(),
        "transport": data.get("transport") or "streamable-http",
        "auth_token": token,
        "allowed_tools": tools,
        "enabled": bool(data.get("enabled", True)),
        "timeout_s": data.get("timeout_s") or 30.0,
    }
    spec = _validate(AgentSpec, fields)
    _ensure_unique(session, Agent, spec.name, row.id)
    for key, value in spec.model_dump().items():
        setattr(row, key, value)
    session.add(row)
    session.flush()
    return row


def delete_agent(session: Session, agent_id: int) -> None:
    row = session.get(Agent, agent_id)
    if row is None:
        raise ConfigError("agente não encontrado")
    session.delete(row)


def save_knowledge_base(
    session: Session, data: dict[str, Any], kb_id: int | None = None
) -> KnowledgeBase:
    row = session.get(KnowledgeBase, kb_id) if kb_id else KnowledgeBase()
    if row is None:
        raise ConfigError("base de conhecimento não encontrada")
    kind = data.get("embedder_kind") or "hashing"
    connection: LlmModel | None = None
    if kind == "model":
        connection_id = data.get("embedding_model_id")
        connection = session.get(LlmModel, int(connection_id)) if connection_id else None
        if connection is None:
            raise ConfigError("escolha o modelo (conexão) que vai gerar os embeddings")
    embedder = {"kind": kind, "dim": int(data.get("embedder_dim") or 512)}
    if connection is not None:
        embedder.update(
            {"connection": connection.name, "model": str(data.get("embedding_model") or "").strip()}
        )
    spec = _validate(
        KnowledgeBaseSpec,
        {
            "name": str(data.get("name") or "").strip(),
            "description": str(data.get("description") or "").strip(),
            "embedder": embedder,
            "chunk_size": data.get("chunk_size") or 800,
            "chunk_overlap": data.get("chunk_overlap")
            if data.get("chunk_overlap") is not None
            else 120,
        },
    )
    _ensure_unique(session, KnowledgeBase, spec.name, row.id)
    row.name, row.description = spec.name, spec.description
    row.embedder_kind, row.embedder_dim = spec.embedder.kind, spec.embedder.dim
    row.embedding_model_id = connection.id if connection else None
    row.embedding_model = spec.embedder.model if connection else None
    row.chunk_size, row.chunk_overlap = spec.chunk_size, spec.chunk_overlap
    session.add(row)
    session.flush()
    session.refresh(row)
    return row


def delete_knowledge_base(session: Session, kb_id: int) -> None:
    row = session.get(KnowledgeBase, kb_id)
    if row is None:
        raise ConfigError("base de conhecimento não encontrada")
    session.delete(row)


def kb_label(row: KnowledgeBase) -> str:
    embedder, _ = embedder_of(row)
    return embedder.label


def kb_stats(session: Session, kb_ids: Iterable[int] | None = None) -> dict[int, dict[str, int]]:
    """Documentos, trechos e documentos indexados com embedder antigo, por base."""
    stats: dict[int, dict[str, int]] = {}
    kb_query = select(KnowledgeBase)
    if kb_ids is not None:
        kb_query = kb_query.where(KnowledgeBase.id.in_(list(kb_ids)))
    for kb in session.scalars(kb_query):
        docs = session.execute(
            select(func.count(Document.id), func.coalesce(func.sum(Document.chunk_count), 0)).where(
                Document.kb_id == kb.id
            )
        ).one()
        stale = session.scalar(
            select(func.count(Document.id)).where(
                Document.kb_id == kb.id, Document.embedder != kb_label(kb)
            )
        )
        stats[kb.id] = {"documents": int(docs[0]), "chunks": int(docs[1]), "stale": int(stale or 0)}
    return stats


def save_profile(
    session: Session, data: dict[str, Any], profile_id: int | None = None
) -> RouterProfile:
    row = get_profile(session, profile_id) if profile_id else RouterProfile()
    if row is None:
        raise ConfigError("roteador não encontrado")
    model_id = data.get("model_id")
    model = session.get(LlmModel, int(model_id)) if model_id else None
    if model is None:
        raise ConfigError("escolha o modelo do roteador")
    agent_ids = [int(x) for x in data.get("agent_ids") or []]
    kb_ids = [int(x) for x in data.get("kb_ids") or []]
    agents = (
        list(session.scalars(select(Agent).where(Agent.id.in_(agent_ids)))) if agent_ids else []
    )
    kbs = (
        list(session.scalars(select(KnowledgeBase).where(KnowledgeBase.id.in_(kb_ids))))
        if kb_ids
        else []
    )
    spec = _validate(
        ProfileSpec,
        {
            "name": str(data.get("name") or "").strip(),
            "description": str(data.get("description") or "").strip(),
            "model": model.name,
            "system_prompt": (
                str(data.get("system_prompt") or "").strip() or DEFAULT_SYSTEM_PROMPT
            ),
            "agents": [a.name for a in agents],
            "knowledge_bases": [k.name for k in kbs],
            "top_k": data.get("top_k") or 4,
            "min_score": data.get("min_score") if data.get("min_score") is not None else 0.2,
            "synthesize": bool(data.get("synthesize", True)),
            "allow_clarify": bool(data.get("allow_clarify", True)),
        },
    )
    _ensure_unique(session, RouterProfile, spec.name, row.id)
    row.name, row.description, row.system_prompt = spec.name, spec.description, spec.system_prompt
    row.model_id, row.model = model.id, model
    row.agents, row.knowledge_bases = agents, kbs
    row.top_k, row.min_score = spec.top_k, spec.min_score
    row.synthesize, row.allow_clarify = spec.synthesize, spec.allow_clarify
    row.enabled = bool(data.get("enabled", True))
    session.add(row)
    session.flush()
    return row


def delete_profile(session: Session, profile_id: int) -> None:
    row = session.get(RouterProfile, profile_id)
    if row is None:
        raise ConfigError("roteador não encontrado")
    session.delete(row)


# --------------------------------------------------------------------------
# execuções (traces)


def save_trace(session: Session, result: RouterResult) -> Trace:
    row = Trace(
        id=result.trace_id,
        profile=result.profile,
        question=result.question,
        answer=result.answer,
        route=result.route,
        model=result.model,
        agent=result.agent,
        tool=result.tool,
        arguments=result.arguments,
        reason=result.reason,
        sources=[s.to_dict() for s in result.sources],
        steps=[s.to_dict() for s in result.steps],
        warnings=list(result.warnings),
        latency_ms=result.latency_ms,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        error=result.error,
    )
    session.add(row)
    return row


def query_traces(
    session: Session,
    *,
    profile: str | None = None,
    route: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Trace]:
    query = select(Trace).order_by(Trace.created_at.desc())
    if profile:
        query = query.where(Trace.profile == profile)
    if route:
        query = query.where(Trace.route == route)
    return list(session.scalars(query.limit(limit).offset(offset)))


def trace_to_dict(row: Trace) -> dict[str, Any]:
    return {
        "trace_id": row.id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "profile": row.profile,
        "question": row.question,
        "answer": row.answer,
        "route": row.route,
        "model": row.model,
        "agent": row.agent,
        "tool": row.tool,
        "arguments": row.arguments,
        "reason": row.reason,
        "sources": row.sources or [],
        "steps": row.steps or [],
        "warnings": row.warnings or [],
        "latency_ms": row.latency_ms,
        "usage": {"input_tokens": row.input_tokens, "output_tokens": row.output_tokens},
        "error": row.error,
    }


def route_counts(session: Session) -> dict[str, int]:
    rows = session.execute(select(Trace.route, func.count(Trace.id)).group_by(Trace.route)).all()
    return {route: int(count) for route, count in rows}


def count(session: Session, cls) -> int:
    return int(session.scalar(select(func.count()).select_from(cls)) or 0)


__all__ = [
    "Chunk",
    "agent_to_spec",
    "count",
    "delete_agent",
    "delete_knowledge_base",
    "delete_model",
    "delete_profile",
    "embedder_of",
    "enabled_profile_names",
    "get_profile",
    "kb_label",
    "kb_stats",
    "kb_to_spec",
    "list_profiles",
    "model_to_spec",
    "profile_to_spec",
    "query_traces",
    "resolve_profile",
    "route_counts",
    "save_agent",
    "save_knowledge_base",
    "save_model",
    "save_profile",
    "save_trace",
    "trace_to_dict",
]
