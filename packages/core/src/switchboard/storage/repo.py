"""Leitura e escrita da configuração e das execuções no banco.

Toda escrita de configuração passa pelas mesmas specs Pydantic do modo YAML
(``ModelSpec``, ``ConnectorSpec``, ``AgentSpec``…), então as regras de
validação são idênticas nos dois modos.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from ..config import (
    DEFAULT_SYSTEM_PROMPT,
    AgentSpec,
    ConnectorSpec,
    EmbedderSpec,
    KnowledgeBaseSpec,
    ModelSpec,
    ProfileSpec,
    format_validation_error,
)
from ..errors import ConfigError
from ..routing.types import ResolvedProfile, RouterResult
from ..secrets import ENC_PREFIX, ENV_PREFIX, MASK, SecretBox
from ..tracing import Span
from .contracts import aware, row_to_contract, row_to_event, row_to_span, span_rows
from .orm import (
    Agent,
    Chunk,
    Connector,
    Contract,
    ContractEventRow,
    Document,
    KnowledgeBase,
    LlmModel,
    RouterProfile,
    SpanRow,
    Trace,
)

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


def connector_to_spec(row: Connector) -> ConnectorSpec:
    return ConnectorSpec(
        name=row.name,
        description=row.description or "",
        url=row.url,
        transport=row.transport,  # type: ignore[arg-type]
        auth_token=row.auth_token,
        allowed_tools=list(row.allowed_tools or []),
        enabled=row.enabled,
        timeout_s=row.timeout_s,
    )


def agent_to_spec(row: Agent) -> AgentSpec:
    return AgentSpec(
        name=row.name,
        description=row.description or "",
        url=row.url,
        auth_token=row.auth_token,
        allowed_skills=list(row.allowed_skills or []),
        enabled=row.enabled,
        timeout_s=row.timeout_s or 15.0,
        deadline_s=row.deadline_s,
        push=True if row.push is None else row.push,
        allow_cross_origin=bool(row.allow_cross_origin),
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


def _or(value: Any, default: Any) -> Any:
    return default if value is None else value


def profile_to_spec(row: RouterProfile, *, only_enabled: bool = True) -> ProfileSpec:
    return ProfileSpec(
        name=row.name,
        description=row.description or "",
        model=row.model.name,
        decision_model=row.decision_model.name if row.decision_model else None,
        decision_threshold=_or(row.decision_threshold, 0.6),
        system_prompt=row.system_prompt or DEFAULT_SYSTEM_PROMPT,
        connectors=[c.name for c in row.connectors if c.enabled or not only_enabled],
        agents=[a.name for a in row.agents if a.enabled or not only_enabled],
        knowledge_bases=[k.name for k in row.knowledge_bases],
        top_k=row.top_k,
        min_score=row.min_score,
        synthesize=row.synthesize,
        allow_clarify=row.allow_clarify,
        wait_s=_or(row.wait_s, 8.0),
        max_parallel=_or(row.max_parallel, 3),
        deadline_s=_or(row.deadline_s, 600.0),
    )


def _profile_query():
    return select(RouterProfile).options(
        selectinload(RouterProfile.connectors),
        selectinload(RouterProfile.agents),
        selectinload(RouterProfile.knowledge_bases),
    )


def resolve_profile(session: Session, name: str) -> ResolvedProfile | None:
    row = session.scalars(_profile_query().where(RouterProfile.name == name)).first()
    if row is None or not row.enabled:
        return None
    return ResolvedProfile(
        spec=profile_to_spec(row),
        model=model_to_spec(row.model),
        connectors=[connector_to_spec(c) for c in row.connectors if c.enabled],
        agents=[agent_to_spec(a) for a in row.agents if a.enabled],
        decision_model=model_to_spec(row.decision_model) if row.decision_model else None,
    )


def find_agent_spec(session: Session, name: str) -> AgentSpec | None:
    row = session.scalars(select(Agent).where(Agent.name == name)).first()
    return agent_to_spec(row) if row is not None else None


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


def _names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [t.strip() for t in value.replace("\n", ",").split(",") if t.strip()]
    return [str(v).strip() for v in value or [] if str(v).strip()]


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
    if row.id is not None:
        # trocar o tipo de um modelo em uso quebraria os roteadores que dependem dele
        chat_users = list(
            session.scalars(select(RouterProfile.name).where(RouterProfile.model_id == row.id))
        )
        decision_users = list(
            session.scalars(
                select(RouterProfile.name).where(RouterProfile.decision_model_id == row.id)
            )
        )
        if spec.is_decision_model and chat_users:
            raise ConfigError(
                f"o modelo é o LLM dos roteadores {', '.join(chat_users)}: não pode virar modelo de decisão"
            )
        if not spec.is_decision_model and decision_users:
            raise ConfigError(
                f"o modelo é o modelo de decisão dos roteadores {', '.join(decision_users)}: "
                "mantenha o provedor typesafe"
            )
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
    deciders = list(
        session.scalars(
            select(RouterProfile.name).where(RouterProfile.decision_model_id == model_id)
        )
    )
    if deciders:
        raise ConfigError(f"o modelo decide pelos roteadores: {', '.join(deciders)}")
    kbs = list(
        session.scalars(
            select(KnowledgeBase.name).where(KnowledgeBase.embedding_model_id == model_id)
        )
    )
    if kbs:
        raise ConfigError(f"o modelo é a conexão de embeddings das bases: {', '.join(kbs)}")
    session.delete(row)


def save_connector(
    session: Session, data: dict[str, Any], *, box: SecretBox, connector_id: int | None = None
) -> Connector:
    row = session.get(Connector, connector_id) if connector_id else Connector()
    if row is None:
        raise ConfigError("conector não encontrado")
    token = _secret(box, data.get("auth_token"), row.auth_token, bool(data.get("clear_auth_token")))
    fields = {
        "name": str(data.get("name") or "").strip(),
        "description": str(data.get("description") or "").strip(),
        "url": str(data.get("url") or "").strip(),
        "transport": data.get("transport") or "streamable-http",
        "auth_token": token,
        "allowed_tools": _names(data.get("allowed_tools")),
        "enabled": bool(data.get("enabled", True)),
        "timeout_s": data.get("timeout_s") or 30.0,
    }
    spec = _validate(ConnectorSpec, fields)
    _ensure_unique(session, Connector, spec.name, row.id)
    for key, value in spec.model_dump().items():
        setattr(row, key, value)
    session.add(row)
    session.flush()
    return row


def delete_connector(session: Session, connector_id: int) -> None:
    row = session.get(Connector, connector_id)
    if row is None:
        raise ConfigError("conector não encontrado")
    session.delete(row)


def save_agent(
    session: Session, data: dict[str, Any], *, box: SecretBox, agent_id: int | None = None
) -> Agent:
    row = session.get(Agent, agent_id) if agent_id else Agent()
    if row is None:
        raise ConfigError("agente não encontrado")
    token = _secret(box, data.get("auth_token"), row.auth_token, bool(data.get("clear_auth_token")))
    fields = {
        "name": str(data.get("name") or "").strip(),
        "description": str(data.get("description") or "").strip(),
        "url": str(data.get("url") or "").strip(),
        "auth_token": token,
        "allowed_skills": _names(data.get("allowed_skills")),
        "enabled": bool(data.get("enabled", True)),
        "timeout_s": data.get("timeout_s") or 15.0,
        "deadline_s": data.get("deadline_s") or None,
        "push": bool(data.get("push", True)),
        "allow_cross_origin": bool(data.get("allow_cross_origin", False)),
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


def _number(data: dict[str, Any], key: str, default: Any) -> Any:
    value = data.get(key)
    return default if value is None or value == "" else value


def save_profile(
    session: Session, data: dict[str, Any], profile_id: int | None = None
) -> RouterProfile:
    row = get_profile(session, profile_id) if profile_id else RouterProfile()
    if row is None:
        raise ConfigError("roteador não encontrado")
    model_id = data.get("model_id")
    model = session.get(LlmModel, int(model_id)) if model_id else None
    if model is None:
        raise ConfigError("escolha o modelo (LLM) do roteador")
    decision_id = data.get("decision_model_id")
    decision = session.get(LlmModel, int(decision_id)) if decision_id else None
    if decision_id and decision is None:
        raise ConfigError("modelo de decisão não encontrado")
    connector_ids = [int(x) for x in data.get("connector_ids") or []]
    agent_ids = [int(x) for x in data.get("agent_ids") or []]
    kb_ids = [int(x) for x in data.get("kb_ids") or []]
    connectors = (
        list(session.scalars(select(Connector).where(Connector.id.in_(connector_ids))))
        if connector_ids
        else []
    )
    agents = (
        list(session.scalars(select(Agent).where(Agent.id.in_(agent_ids)))) if agent_ids else []
    )
    kbs = (
        list(session.scalars(select(KnowledgeBase).where(KnowledgeBase.id.in_(kb_ids))))
        if kb_ids
        else []
    )
    if model.provider == "typesafe":
        raise ConfigError(
            f"'{model.name}' é um modelo de decisão; escolha-o em 'modelo de decisão' e use um LLM como modelo principal"
        )
    if decision is not None and decision.provider != "typesafe":
        raise ConfigError(
            f"'{decision.name}' não é um modelo de decisão (precisa ser TypeSafe Jev)"
        )
    spec = _validate(
        ProfileSpec,
        {
            "name": str(data.get("name") or "").strip(),
            "description": str(data.get("description") or "").strip(),
            "model": model.name,
            "decision_model": decision.name if decision else None,
            "decision_threshold": _number(data, "decision_threshold", 0.6),
            "system_prompt": (
                str(data.get("system_prompt") or "").strip() or DEFAULT_SYSTEM_PROMPT
            ),
            "connectors": [c.name for c in connectors],
            "agents": [a.name for a in agents],
            "knowledge_bases": [k.name for k in kbs],
            "top_k": data.get("top_k") or 4,
            "min_score": _number(data, "min_score", 0.2),
            "synthesize": bool(data.get("synthesize", True)),
            "allow_clarify": bool(data.get("allow_clarify", True)),
            "wait_s": _number(data, "wait_s", 8.0),
            "max_parallel": _number(data, "max_parallel", 3),
            "deadline_s": _number(data, "deadline_s", 600.0),
        },
    )
    _ensure_unique(session, RouterProfile, spec.name, row.id)
    row.name, row.description, row.system_prompt = spec.name, spec.description, spec.system_prompt
    row.model_id, row.model = model.id, model
    row.decision_model_id, row.decision_model = (decision.id if decision else None), decision
    row.decision_threshold = spec.decision_threshold
    row.connectors, row.agents, row.knowledge_bases = connectors, agents, kbs
    row.top_k, row.min_score = spec.top_k, spec.min_score
    row.synthesize, row.allow_clarify = spec.synthesize, spec.allow_clarify
    row.wait_s, row.max_parallel, row.deadline_s = spec.wait_s, spec.max_parallel, spec.deadline_s
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
# execuções (traces), spans e contratos


def save_trace(session: Session, result: RouterResult) -> Trace:
    """Grava a execução; se ela já existe (delegação a agentes), completa sem sobrescrever.

    Numa delegação, a linha é criada pelo gerenciador de contratos antes do
    envio, e a resposta/estado passam a ser dele (a consolidação pode acontecer
    antes deste registro). Aqui só entram os dados do pedido: spans, avisos,
    tokens, latência e fontes.
    """
    now = datetime.now(UTC)
    row = session.get(Trace, result.trace_id)
    spans = [Span.from_dict(s) for s in result.spans]
    if row is None:
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
            status=result.status,
            decided_by=result.decided_by,
            confidence=result.confidence,
            tasks=result.tasks or None,
            updated_at=now,
            finished_at=now if result.status in ("completed", "failed") else None,
        )
        session.add(row)
    else:
        if result.question and not row.question:
            row.question = result.question
        row.steps = list(row.steps or []) + [s.to_dict() for s in result.steps]
        row.warnings = list(dict.fromkeys(list(row.warnings or []) + list(result.warnings)))
        row.latency_ms = max(row.latency_ms or 0.0, result.latency_ms)
        row.input_tokens = (row.input_tokens or 0) + result.usage.input_tokens
        row.output_tokens = (row.output_tokens or 0) + result.usage.output_tokens
        if result.sources and not row.sources:
            row.sources = [s.to_dict() for s in result.sources]
        if result.error and not row.error:
            row.error = result.error
        row.updated_at = now
    session.flush()
    if spans:
        session.add_all(span_rows(spans))
    return row


OPEN_RUNS = "abertas"  # filtro: execuções em andamento (pending, needs_input, consolidating)
OPEN_CONTRACTS = "abertos"  # filtro: contratos em aberto (proposto, ativo, aguardando_entrada)


def query_traces(
    session: Session,
    *,
    profile: str | None = None,
    route: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Trace]:
    query = select(Trace).order_by(Trace.created_at.desc())
    if profile:
        query = query.where(Trace.profile == profile)
    if route:
        query = query.where(Trace.route == route)
    if status == OPEN_RUNS:  # todas as que ainda não terminaram
        query = query.where(Trace.status.in_(["pending", "needs_input", "consolidating"]))
    elif status:
        query = query.where(Trace.status == status)
    return list(session.scalars(query.limit(limit).offset(offset)))


def _iso(value: datetime | None) -> str | None:
    value = aware(value)
    return value.isoformat() if value else None


def trace_to_dict(row: Trace) -> dict[str, Any]:
    return {
        "trace_id": row.id,
        "run_id": row.id,
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
        "finished_at": _iso(row.finished_at),
        "status": row.status or "completed",
        "profile": row.profile,
        "question": row.question,
        "answer": row.answer,
        "route": row.route,
        "model": row.model,
        "decided_by": row.decided_by,
        "confidence": row.confidence,
        "agent": row.agent,
        "tool": row.tool,
        "arguments": row.arguments,
        "tasks": row.tasks or [],
        "reason": row.reason,
        "sources": row.sources or [],
        "steps": row.steps or [],
        "warnings": row.warnings or [],
        "latency_ms": row.latency_ms,
        "usage": {"input_tokens": row.input_tokens, "output_tokens": row.output_tokens},
        "error": row.error,
    }


def run_spans(session: Session, run_id: str) -> list[Span]:
    return [
        row_to_span(r)
        for r in session.scalars(
            select(SpanRow)
            .where(SpanRow.trace_id == run_id)
            .order_by(SpanRow.started_at, SpanRow.pk)
        )
    ]


def run_contracts(session: Session, run_id: str, *, events: bool = True) -> list[dict[str, Any]]:
    out = []
    for row in session.scalars(
        select(Contract).where(Contract.run_id == run_id).order_by(Contract.created_at)
    ):
        record = row_to_contract(row)
        evs = (
            [
                row_to_event(e)
                for e in session.scalars(
                    select(ContractEventRow)
                    .where(ContractEventRow.contract_id == row.id)
                    .order_by(ContractEventRow.id)
                )
            ]
            if events
            else None
        )
        out.append(record.to_dict(evs))
    return out


def contract_spans(contracts: list[dict[str, Any]], trace_id: str) -> list[Span]:
    """Cada contrato (spawn) visto como um span, aberto até o estado terminal."""
    from ..contracts import states

    spans = []
    for c in contracts:
        started = datetime.fromisoformat(c["created_at"])
        ended = datetime.fromisoformat(c["finished_at"]) if c.get("finished_at") else None
        status = (
            "ok"
            if c["state"] == states.COMPLETED
            else "open"
            if c["state"] in states.OPEN
            else "error"
        )
        spans.append(
            Span(
                id=c.get("span_id") or c["id"][-16:],
                trace_id=trace_id,
                parent_id=c.get("parent_span_id"),
                kind="contrato",
                name=f"contrato: {c['agent']}/{c['skill']}",
                started_at=started,
                ended_at=ended,
                status=status,
                attributes={"contract_id": c["id"], "estado": c["state"], "tipo": c["kind"]},
            )
        )
    return spans


def run_details(session: Session, run_id: str) -> dict[str, Any] | None:
    """Execução completa: trace, spans (inclusive os contratos) e contratos com eventos."""
    row = session.get(Trace, run_id)
    if row is None:
        return None
    data = trace_to_dict(row)
    contracts = run_contracts(session, run_id)
    spans = run_spans(session, run_id) + contract_spans(contracts, run_id)
    spans.sort(key=lambda s: s.started_at)
    data["contracts"] = contracts
    data["spans"] = [s.to_dict() for s in spans]
    return data


def query_contracts(
    session: Session,
    *,
    state: str | None = None,
    agent: str | None = None,
    profile: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    from ..contracts import states

    query = select(Contract).order_by(Contract.created_at.desc())
    if state == OPEN_CONTRACTS:
        query = query.where(Contract.state.in_(sorted(states.OPEN)))
    elif state:
        query = query.where(Contract.state == state)
    if agent:
        query = query.where(Contract.agent == agent)
    if profile:
        query = query.where(Contract.profile == profile)
    return [
        row_to_contract(r).to_dict() for r in session.scalars(query.limit(limit).offset(offset))
    ]


def contract_details(session: Session, contract_id: str) -> dict[str, Any] | None:
    row = session.get(Contract, contract_id)
    if row is None:
        return None
    events = [
        row_to_event(e)
        for e in session.scalars(
            select(ContractEventRow)
            .where(ContractEventRow.contract_id == contract_id)
            .order_by(ContractEventRow.id)
        )
    ]
    data = row_to_contract(row).to_dict(events)
    data["input_schema"] = row.input_schema
    data["output_schema"] = row.output_schema
    return data


def contract_state_counts(session: Session) -> dict[str, int]:
    rows = session.execute(
        select(Contract.state, func.count(Contract.id)).group_by(Contract.state)
    ).all()
    return {state: int(n) for state, n in rows}


def contract_agent_counts(session: Session) -> dict[str, dict[str, int]]:
    """Contratos por agente e estado: ``{"risco": {"concluido": 3, "ativo": 1}}``."""
    rows = session.execute(
        select(Contract.agent, Contract.state, func.count(Contract.id)).group_by(
            Contract.agent, Contract.state
        )
    ).all()
    out: dict[str, dict[str, int]] = {}
    for agent, state, n in rows:
        out.setdefault(agent, {})[state] = int(n)
    return out


def route_counts(session: Session) -> dict[str, int]:
    rows = session.execute(select(Trace.route, func.count(Trace.id)).group_by(Trace.route)).all()
    return {route: int(count) for route, count in rows}


def count(session: Session, cls) -> int:
    return int(session.scalar(select(func.count()).select_from(cls)) or 0)


__all__ = [
    "Chunk",
    "agent_to_spec",
    "connector_to_spec",
    "contract_agent_counts",
    "contract_details",
    "contract_spans",
    "contract_state_counts",
    "count",
    "delete_agent",
    "delete_connector",
    "delete_knowledge_base",
    "delete_model",
    "delete_profile",
    "embedder_of",
    "enabled_profile_names",
    "find_agent_spec",
    "get_profile",
    "kb_label",
    "kb_stats",
    "kb_to_spec",
    "list_profiles",
    "model_to_spec",
    "profile_to_spec",
    "query_contracts",
    "query_traces",
    "resolve_profile",
    "route_counts",
    "run_contracts",
    "run_details",
    "run_spans",
    "save_agent",
    "save_connector",
    "save_knowledge_base",
    "save_model",
    "save_profile",
    "save_trace",
    "trace_to_dict",
]
