"""Plugins no banco: instalar, atualizar, aprovar, configurar, ligar a roteadores e medir o uso.

O que é guardado de um plugin:

* a linha em ``plugins`` com o pacote normalizado (zip), o manifesto, os achados da
  varredura, os playbooks, os exemplos de roteamento, a configuração e a aprovação;
* os conectores, agentes e bases do plugin como linhas das tabelas de sempre, com
  ``plugin_id`` e nome ``plugin.componente`` (somente leitura no console);
* os documentos das bases do plugin, indexados pelo :class:`PluginService`;
* o histórico (``plugin_events``) e o uso por versão (``plugin_usage``).

Um plugin só participa do roteamento quando está **pronto**: ligado, sem achados
``held`` pendentes e com a configuração obrigatória completa (:func:`plugin_state`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError
from sqlalchemy import case, func, select, update
from sqlalchemy.orm import Session

from ..contracts import states
from ..errors import PluginConflict, PluginError
from ..plugins import Finding, Playbook, PluginBundle, PluginManifest, counts, load_plugin
from ..plugins.bundle import check_config_value, resolve_config
from ..plugins.findings import sorted_findings
from ..plugins.semver import compare_versions
from ..secrets import ENC_PREFIX, ENV_PREFIX, MASK, SecretBox
from .contracts import aware
from .orm import (
    Agent,
    Connector,
    Contract,
    KnowledgeBase,
    Plugin,
    PluginEvent,
    PluginUse,
    RouterProfile,
    profile_plugins,
)

STATE_READY = "pronto"
STATE_DISABLED = "desativado"
STATE_PENDING = "aguardando_aprovacao"
STATE_CONFIG = "config_incompleta"
STATE_INVALID = "invalido"

STATE_LABELS = {
    STATE_READY: "Ativo",
    STATE_DISABLED: "Desativado",
    STATE_PENDING: "Aguardando aprovação",
    STATE_CONFIG: "Configuração incompleta",
    STATE_INVALID: "Inválido (reenvie o pacote)",
}


def _iso(value: datetime | None) -> str | None:
    value = aware(value)
    return value.isoformat() if value else None


# --------------------------------------------------------------------------
# estado de um plugin instalado


def findings_of(row: Plugin) -> list[Finding]:
    return [Finding.from_dict(d) for d in row.findings or []]


def pending_findings(row: Plugin) -> list[Finding]:
    """Achados ``held`` que ainda ninguém aprovou."""
    approved = set(row.approved_fingerprints or [])
    return [f for f in findings_of(row) if f.level == "held" and f.fingerprint not in approved]


def manifest_of(row: Plugin) -> PluginManifest | None:
    try:
        return PluginManifest.model_validate(row.manifest)
    except ValidationError:
        return None


def missing_config(row: Plugin) -> list[str]:
    manifest = manifest_of(row)
    if manifest is None:
        return []
    try:
        return resolve_config(manifest, row.config or {})[1]
    except PluginError:
        return []


def plugin_state(row: Plugin) -> str:
    if manifest_of(row) is None:
        return STATE_INVALID
    if any(f.get("level") == "block" for f in row.findings or []):
        return STATE_INVALID  # a varredura refeita (regras ou versão novas) passou a bloquear
    if not row.enabled:
        return STATE_DISABLED
    if pending_findings(row):
        return STATE_PENDING
    if missing_config(row):
        return STATE_CONFIG
    return STATE_READY


def plugin_ready(row: Plugin) -> bool:
    return plugin_state(row) == STATE_READY


def playbook_to_dict(pb: Playbook) -> dict[str, Any]:
    return {
        "plugin": pb.plugin,
        "version": pb.version,
        "skill": pb.skill,
        "description": pb.description,
        "body": pb.body,
        "applies_to": list(pb.applies_to),
        "triggers": list(pb.triggers),
        "always": pb.always,
    }


def playbook_from_dict(data: Mapping[str, Any]) -> Playbook:
    return Playbook(
        plugin=str(data["plugin"]),
        version=str(data["version"]),
        skill=str(data["skill"]),
        description=str(data.get("description") or ""),
        body=str(data.get("body") or ""),
        applies_to=tuple(data.get("applies_to") or ()),
        triggers=tuple(data.get("triggers") or ()),
        always=bool(data.get("always")),
    )


def playbooks_of(row: Plugin) -> list[Playbook]:
    return [playbook_from_dict(d) for d in row.playbooks or []]


# --------------------------------------------------------------------------
# leitura


def list_plugins(session: Session) -> list[Plugin]:
    return list(session.scalars(select(Plugin).order_by(Plugin.name)))


def get_plugin(session: Session, key: int | str) -> Plugin | None:
    if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
        return session.get(Plugin, int(key))
    return session.scalars(select(Plugin).where(Plugin.name == key)).first()


def plugin_owned(session: Session, kind: type, plugin_id: int | None) -> list[Any]:
    if plugin_id is None:
        return []
    return list(
        session.scalars(select(kind).where(kind.plugin_id == plugin_id).order_by(kind.name))
    )


def plugin_names_by_id(session: Session) -> dict[int, str]:
    return {pid: name for pid, name in session.execute(select(Plugin.id, Plugin.name))}


def profiles_using(session: Session, plugin_id: int) -> list[str]:
    return list(
        session.scalars(
            select(RouterProfile.name)
            .join(profile_plugins, profile_plugins.c.profile_id == RouterProfile.id)
            .where(profile_plugins.c.plugin_id == plugin_id)
            .order_by(RouterProfile.name)
        )
    )


def open_contracts(session: Session, plugin_name: str) -> int:
    """Contratos em aberto com agentes deste plugin."""
    return int(
        session.scalar(
            select(func.count(Contract.id)).where(
                Contract.agent.like(f"{plugin_name}.%"), Contract.state.in_(sorted(states.OPEN))
            )
        )
        or 0
    )


# --------------------------------------------------------------------------
# histórico


def record_event(
    session: Session, plugin: str, action: str, *, version: str = "", actor: str = "", **detail: Any
) -> None:
    session.add(
        PluginEvent(plugin=plugin, action=action, version=version, actor=actor[:100], detail=detail)
    )


def plugin_events(
    session: Session, name: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Histórico, do mais novo para o mais antigo (de um plugin ou, sem nome, de todos).

    O histórico sobrevive à remoção do plugin: é a trilha de quem instalou, aprovou e removeu.
    """
    query = select(PluginEvent).order_by(PluginEvent.at.desc(), PluginEvent.id.desc()).limit(limit)
    if name:
        query = query.where(PluginEvent.plugin == name)
    return [
        {
            "plugin": r.plugin,
            "at": _iso(r.at),
            "action": r.action,
            "version": r.version,
            "actor": r.actor,
            "detail": r.detail or {},
        }
        for r in session.scalars(query)
    ]


# --------------------------------------------------------------------------
# instalar e atualizar


@dataclass
class InstallResult:
    plugin: Plugin
    action: str  # install | upgrade | unchanged
    previous_version: str | None = None
    pending: list[Finding] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def state(self) -> str:
        return plugin_state(self.plugin)


def merge_config(
    manifest: PluginManifest,
    current: Mapping[str, str],
    given: Mapping[str, Any] | None,
    box: SecretBox,
    clear: Iterable[str] = (),
) -> dict[str, str]:
    """Aplica os valores informados sobre os atuais, respeitando o que é sensível.

    Valores sensíveis são cifrados (ou ``env:NOME``) e nunca saem do banco; vazio ou a
    máscara da UI mantêm o valor atual, e ``clear`` remove.
    """
    given = {k: v for k, v in (given or {}).items() if v is not None}
    unknown = sorted((set(given) | set(clear)) - set(manifest.config))
    if unknown:
        known = ", ".join(manifest.config) or "nenhuma"
        raise PluginError(
            f"config desconhecida no plugin '{manifest.name}': {', '.join(unknown)} (declaradas: {known})"
        )
    removed = set(clear)
    out: dict[str, str] = {}
    for key, spec in manifest.config.items():
        keep = current.get(key)
        # a chave mudou de natureza numa versão nova: o que estava guardado deixa de servir
        if keep and spec.sensitive:
            keep = box.seal(keep)  # texto puro de uma versão antiga (não sensível) passa a cifrado
        elif keep and keep.startswith((ENC_PREFIX, ENV_PREFIX)):
            keep = None  # segredo numa chave que agora é pública: não vira endereço
        value: str | None
        if key in removed:
            value = None
        elif key in given:
            raw = str(given[key]).strip()
            if raw and raw != MASK:
                check_config_value(key, raw)  # o que a pessoa digitou (o cifrado é bem maior)
            if spec.sensitive:
                if raw.startswith(ENC_PREFIX):
                    raise PluginError(
                        f"config {key}: informe o segredo em texto (ele é cifrado aqui) ou "
                        "env:NOME; um valor já cifrado não é aceito"
                    )
                value = keep if raw in ("", MASK) else box.seal(raw)
            else:
                if raw.startswith((ENC_PREFIX, ENV_PREFIX)):
                    raise PluginError(
                        f"config {key}: a chave não é sensível, então o valor é usado como está "
                        "(env:/enc: só valem para segredos)"
                    )
                value = raw or None
        else:
            value = keep
        if value:
            out[key] = value
    return out


def _sync_components(session: Session, row: Plugin, bundle: PluginBundle) -> None:
    """Faz as linhas de conectores, agentes e bases refletirem o plugin com a configuração atual."""
    expanded = bundle.expand(row.config)
    session.flush()  # row.id

    existing_c = {c.name: c for c in plugin_owned(session, Connector, row.id)}
    for spec in expanded.connectors:
        conn = existing_c.pop(spec.name, None) or Connector(plugin_id=row.id)
        for key, value in spec.model_dump().items():
            setattr(conn, key, value)
        session.add(conn)
    for gone in existing_c.values():
        session.delete(gone)

    existing_a = {a.name: a for a in plugin_owned(session, Agent, row.id)}
    for spec in expanded.agents:
        agent = existing_a.pop(spec.name, None) or Agent(plugin_id=row.id)
        for key, value in spec.model_dump().items():
            setattr(agent, key, value)
        session.add(agent)
    busy = {
        name: n
        for name, n in session.execute(
            select(Contract.agent, func.count(Contract.id))
            .where(Contract.agent.in_(list(existing_a)), Contract.state.in_(sorted(states.OPEN)))
            .group_by(Contract.agent)
        )
        if n
    }
    if busy:
        listed = ", ".join(f"{name} ({n})" for name, n in sorted(busy.items()))
        raise PluginError(
            f"esta mudança remove agente(s) com contratos em aberto: {listed}. Espere os "
            "contratos terminarem (ou cancele-os) e tente de novo"
        )
    for gone in existing_a.values():
        session.delete(gone)

    existing_k = {k.name: k for k in plugin_owned(session, KnowledgeBase, row.id)}
    for spec in expanded.knowledge_bases:
        kb = existing_k.pop(spec.name, None)
        if kb is None:
            kb = KnowledgeBase(
                plugin_id=row.id,
                name=spec.name,
                embedder_kind=spec.embedder.kind,
                embedder_dim=spec.embedder.dim,
            )
        # o embedder é escolha de quem opera; o resto vem do pacote
        kb.description = spec.description
        kb.chunk_size, kb.chunk_overlap = spec.chunk_size, spec.chunk_overlap
        session.add(kb)
    for gone in existing_k.values():
        session.delete(gone)
    session.flush()
    _apply_readiness(session, row)


def _claim(session: Session, row: Plugin) -> None:
    """Garante que ninguém mudou o plugin entre a leitura e esta escrita.

    A atualização condicional espera (ou falha) se outro envio do mesmo plugin está em
    andamento e não casa se ele já gravou outro conteúdo: aí o segundo envio é recusado em
    vez de sobrescrever o primeiro com base numa leitura velha (um downgrade silencioso).
    """
    claimed = session.execute(
        update(Plugin)
        .where(Plugin.id == row.id, Plugin.digest == row.digest, Plugin.version == row.version)
        .values(updated_at=datetime.now(UTC))
        .execution_options(synchronize_session=False)
    ).rowcount
    if claimed != 1:
        raise PluginConflict(
            f"o plugin '{row.name}' mudou durante este envio (outro envio ao mesmo tempo?): "
            "confira a versão instalada e envie de novo"
        )
    session.refresh(row)


def _apply_readiness(session: Session, row: Plugin) -> None:
    """Conectores e agentes de um plugin só ficam ligados enquanto ele está pronto.

    Assim nada os contata (descoberta no console, roteamento) enquanto o plugin espera
    aprovação, está desativado, inválido ou com a configuração incompleta: o endereço e o
    token só são usados depois que alguém aprovou e ligou o plugin.
    """
    ready = plugin_ready(row)
    for kind in (Connector, Agent):
        for owned in plugin_owned(session, kind, row.id):
            owned.enabled = ready


def _store_bundle(row: Plugin, bundle: PluginBundle) -> None:
    manifest = bundle.manifest
    assert manifest is not None
    row.name = manifest.name
    row.version = manifest.version
    row.description = manifest.description
    row.format = bundle.format
    row.origin = bundle.origin[:200]
    row.digest = bundle.digest
    row.package = bundle.pack()
    row.manifest = manifest.to_dict()
    row.findings = [f.to_dict() for f in sorted_findings(bundle.findings)]
    row.playbooks = [playbook_to_dict(p) for p in bundle.playbooks]
    row.examples = bundle.routing_examples()


def install_plugin(
    session: Session,
    bundle: PluginBundle,
    *,
    box: SecretBox,
    config: Mapping[str, Any] | None = None,
    clear: Iterable[str] = (),
    actor: str = "console",
    allow_downgrade: bool = False,
) -> InstallResult:
    """Instala o pacote (ou atualiza o plugin de mesmo nome).

    Recusa o que tem achado ``block``. Achados ``held`` não impedem a instalação: o plugin
    fica **aguardando aprovação** e só entra no roteamento depois de aprovado. A aprovação
    de versões anteriores continua valendo para os achados que não mudaram.
    """
    bundle.require_ok()
    manifest = bundle.manifest
    assert manifest is not None
    row = session.scalars(select(Plugin).where(Plugin.name == manifest.name)).first()
    notes: list[str] = []
    previous: str | None = None
    if row is None:
        row = Plugin(name=manifest.name, config={}, approved_fingerprints=[], enabled=True)
        action = "install"
        changed = True
    else:
        _claim(session, row)
        previous = row.version
        order = compare_versions(manifest.version, row.version)
        if order < 0 and not allow_downgrade:
            raise PluginError(
                f"o plugin '{manifest.name}' já está na versão {row.version}, mais nova que a "
                f"{manifest.version} enviada; confirme o downgrade para continuar"
            )
        changed = row.digest != bundle.digest
        action = "upgrade" if changed else "unchanged"
        if changed and order == 0:
            notes.append(
                f"a versão {manifest.version} já estava instalada com outro conteúdo: "
                "mude a versão no manifesto a cada alteração, para o uso ser separado por versão"
            )
        if order < 0:
            notes.append(f"downgrade de {row.version} para {manifest.version}")
    # mesmo sem mudança de conteúdo, regrava o que se lê do pacote: conserta um plugin
    # que ficou inválido (por exemplo, guardado por uma versão anterior do Switchboard)
    _store_bundle(row, bundle)
    before = dict(row.config or {})
    row.config = merge_config(manifest, before, config, box, clear)
    session.add(row)
    session.flush()
    _sync_components(session, row, bundle)
    pending = pending_findings(row)
    missing = missing_config(row)
    if action != "unchanged" or row.config != before:
        record_event(
            session,
            manifest.name,
            action if action != "unchanged" else "config",
            version=manifest.version,
            actor=actor,
            previous=previous,
            digest=bundle.digest[:12],
            pending=[f.code for f in pending],
            missing=missing,
        )
    return InstallResult(row, action, previous, pending, missing, notes)


# --------------------------------------------------------------------------
# operar


def require_plugin(session: Session, key: int | str) -> Plugin:
    row = get_plugin(session, key)
    if row is None:
        raise PluginError("plugin não encontrado")
    return row


def approve_plugin(
    session: Session, key: int | str, *, actor: str = "console", digest: str | None = None
) -> Plugin:
    """Aprova os achados ``held`` atuais (guarda quem aprovou e quando).

    ``digest`` é o conteúdo que a pessoa revisou: se outro pacote foi instalado depois que
    ela abriu a página, a aprovação é recusada em vez de aprovar o que ninguém viu.
    """
    row = require_plugin(session, key)
    if digest is not None and digest.strip() and not row.digest.startswith(digest.strip()):
        raise PluginConflict(
            "o pacote mudou depois que você abriu a página (outra versão foi enviada): "
            "revise os achados de novo antes de aprovar"
        )
    pending = pending_findings(row)
    if not pending:
        return row
    approved = list(row.approved_fingerprints or [])
    approved.extend(f.fingerprint for f in pending if f.fingerprint not in approved)
    row.approved_fingerprints = approved
    row.approved_by = actor[:100]
    row.approved_at = datetime.now(UTC)
    record_event(
        session,
        row.name,
        "approve",
        version=row.version,
        actor=actor,
        codes=sorted({f.code for f in pending}),
        count=len(pending),
    )
    _apply_readiness(session, row)
    return row


def set_plugin_enabled(
    session: Session, key: int | str, enabled: bool, *, actor: str = "console"
) -> Plugin:
    row = require_plugin(session, key)
    if row.enabled != enabled:
        row.enabled = enabled
        record_event(
            session, row.name, "enable" if enabled else "disable", version=row.version, actor=actor
        )
    _apply_readiness(session, row)
    return row


def set_plugin_config(
    session: Session,
    key: int | str,
    config: Mapping[str, Any],
    *,
    box: SecretBox,
    clear: Iterable[str] = (),
    actor: str = "console",
) -> Plugin:
    """Troca valores da configuração e refaz os conectores/agentes/bases do plugin."""
    row = require_plugin(session, key)
    manifest = manifest_of(row)
    if manifest is None:
        raise PluginError("o manifesto guardado não é mais válido: reenvie o pacote")
    new_config = merge_config(manifest, row.config or {}, config, box, clear)
    if new_config == (row.config or {}):
        return row
    row.config = new_config
    bundle = load_plugin(row.package, switchboard_version="")
    _sync_components(session, row, bundle.require_ok())
    record_event(
        session,
        row.name,
        "config",
        version=row.version,
        actor=actor,
        keys=sorted(set(config) | set(clear)),
    )
    return row


def rescan_plugin(session: Session, key: int | str, *, actor: str = "console") -> InstallResult:
    """Refaz a varredura do pacote guardado (por exemplo, depois de atualizar o Switchboard)."""
    row = require_plugin(session, key)
    bundle = load_plugin(row.package)
    before = {f.fingerprint for f in findings_of(row)}
    row.findings = [f.to_dict() for f in sorted_findings(bundle.findings)]
    blocked = [f for f in bundle.findings if f.level == "block"]
    if bundle.manifest is not None and not blocked:
        row.manifest = bundle.manifest.to_dict()
        row.playbooks = [playbook_to_dict(p) for p in bundle.playbooks]
        row.examples = bundle.routing_examples()
        _sync_components(session, row, bundle)
    else:
        # passou a bloquear (regras novas, versão do Switchboard fora da faixa): o plugin
        # fica inválido e sai do roteamento, com os achados à vista
        _apply_readiness(session, row)
    after = {f.fingerprint for f in findings_of(row)}
    pending = pending_findings(row)
    record_event(
        session,
        row.name,
        "rescan",
        version=row.version,
        actor=actor,
        new=len(after - before),
        pending=[f.code for f in pending],
        blocked=sorted({f.code for f in blocked}),
    )
    return InstallResult(row, "rescan", row.version, pending, missing_config(row))


def uninstall_plugin(
    session: Session, key: int | str, *, actor: str = "console", force: bool = False
) -> str:
    """Remove o plugin, seus conectores, agentes, bases (com os documentos) e as ligações.

    O histórico e o uso por versão ficam. Com contratos em aberto com agentes do plugin,
    exige ``force``: o router não conseguiria mais acompanhá-los.
    """
    row = require_plugin(session, key)
    open_n = open_contracts(session, row.name)
    if open_n and not force:
        raise PluginError(
            f"há {open_n} contrato(s) em aberto com agentes do plugin '{row.name}': espere "
            "terminarem (ou confirme a remoção)"
        )
    for kind in (Connector, Agent, KnowledgeBase):
        for owned in plugin_owned(session, kind, row.id):
            session.delete(owned)
    session.flush()  # os componentes saem antes do plugin, sem depender da cascata do banco
    name, version = row.name, row.version
    record_event(session, name, "uninstall", version=version, actor=actor, open_contracts=open_n)
    session.delete(row)
    session.flush()
    return name


# --------------------------------------------------------------------------
# ligação com os roteadores


def set_profile_plugins(
    session: Session, profile: RouterProfile, plugin_ids: Iterable[int], *, actor: str = "console"
) -> None:
    """Os plugins de um roteador (pela tela ou API do roteador); cada mudança vai ao histórico."""
    ids = [int(i) for i in plugin_ids]
    rows = list(session.scalars(select(Plugin).where(Plugin.id.in_(ids)))) if ids else []
    if len(rows) != len(set(ids)):
        raise PluginError("plugin não encontrado")
    before = {p.id: p for p in profile.plugins} if profile.id is not None else {}
    profile.plugins = sorted(rows, key=lambda p: p.name)
    after = {p.id: p for p in rows}
    for pid in sorted(set(before) ^ set(after)):
        plugin = after.get(pid) or before[pid]
        record_event(
            session,
            plugin.name,
            "profiles",
            version=plugin.version,
            actor=actor,
            profile=profile.name,
            attached=pid in after,
        )


def set_plugin_profiles(
    session: Session, key: int | str, profile_ids: Iterable[int], *, actor: str = "console"
) -> list[str]:
    """Liga o plugin exatamente aos roteadores indicados (e o desliga dos demais)."""
    row = require_plugin(session, key)
    wanted = {int(i) for i in profile_ids}
    profiles = list(session.scalars(select(RouterProfile).order_by(RouterProfile.name)))
    if wanted - {p.id for p in profiles}:
        raise PluginError("roteador não encontrado")
    before = [p.name for p in profiles if row in p.plugins]
    for profile in profiles:
        attached = row in profile.plugins
        if profile.id in wanted and not attached:
            profile.plugins = sorted([*profile.plugins, row], key=lambda x: x.name)
        elif profile.id not in wanted and attached:
            profile.plugins = [x for x in profile.plugins if x.id != row.id]
    after = [p.name for p in profiles if p.id in wanted]
    if before != after:
        record_event(
            session,
            row.name,
            "profiles",
            version=row.version,
            actor=actor,
            before=before,
            after=after,
        )
    return after


# --------------------------------------------------------------------------
# uso


def record_usage(
    session: Session, trace_id: str, profile: str, uses: Iterable[Mapping[str, Any]]
) -> None:
    for use in uses:
        session.add(
            PluginUse(
                trace_id=trace_id,
                plugin=str(use["plugin"])[:30],
                version=str(use["version"])[:40],
                profile=profile[:63],
                kind=str(use["kind"])[:12],
                component=str(use["component"])[:120],
                ok=use.get("ok"),
            )
        )


def trace_plugins(session: Session, trace_id: str) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(PluginUse).where(PluginUse.trace_id == trace_id).order_by(PluginUse.id)
    )
    return [
        {
            "plugin": r.plugin,
            "version": r.version,
            "kind": r.kind,
            "component": r.component,
            "ok": r.ok,
        }
        for r in rows
    ]


def usage_by_version(session: Session, plugin: str | None = None) -> list[dict[str, Any]]:
    """Uso por plugin e versão: pedidos, usos por tipo, erros e último uso."""
    query = (
        select(
            PluginUse.plugin,
            PluginUse.version,
            func.count(func.distinct(PluginUse.trace_id)),
            func.count(PluginUse.id),
            func.sum(case((PluginUse.ok.is_(False), 1), else_=0)),
            func.max(PluginUse.at),
        )
        .group_by(PluginUse.plugin, PluginUse.version)
        .order_by(PluginUse.plugin, PluginUse.version)
    )
    if plugin:
        query = query.where(PluginUse.plugin == plugin)
    kinds: dict[tuple[str, str], dict[str, int]] = {}
    kind_query = select(
        PluginUse.plugin, PluginUse.version, PluginUse.kind, func.count(PluginUse.id)
    ).group_by(PluginUse.plugin, PluginUse.version, PluginUse.kind)
    if plugin:
        kind_query = kind_query.where(PluginUse.plugin == plugin)
    for name, version, kind, n in session.execute(kind_query):
        kinds.setdefault((name, version), {})[kind] = int(n)
    out = []
    for name, version, requests, uses, errors, last in session.execute(query):
        out.append(
            {
                "plugin": name,
                "version": version,
                "requests": int(requests),
                "uses": int(uses),
                "errors": int(errors or 0),
                "kinds": kinds.get((name, version), {}),
                "last_at": _iso(last),
            }
        )
    return out


def usage_by_component(session: Session, plugin: str) -> list[dict[str, Any]]:
    rows = session.execute(
        select(
            PluginUse.kind,
            PluginUse.component,
            func.count(PluginUse.id),
            func.sum(case((PluginUse.ok.is_(False), 1), else_=0)),
            func.max(PluginUse.at),
        )
        .where(PluginUse.plugin == plugin)
        .group_by(PluginUse.kind, PluginUse.component)
        .order_by(func.count(PluginUse.id).desc())
    )
    return [
        {
            "kind": kind,
            "component": component,
            "uses": int(n),
            "errors": int(errors or 0),
            "last_at": _iso(last),
        }
        for kind, component, n, errors, last in rows
    ]


# --------------------------------------------------------------------------
# pré-visualização (nada é gravado)


def plugin_preview(session: Session, bundle: PluginBundle) -> dict[str, Any]:
    """O que aconteceria ao instalar este pacote agora: ação, versão instalada e o que pede aprovação.

    Os achados ``held`` que já foram aprovados na versão instalada e não mudaram continuam
    aprovados (``carried``); os demais ficam pendentes (``pending``).
    """
    manifest = bundle.manifest
    row = (
        session.scalars(select(Plugin).where(Plugin.name == manifest.name)).first()
        if manifest is not None
        else None
    )
    action, installed = "install", None
    approved: set[str] = set()
    if row is not None and manifest is not None:
        if row.digest == bundle.digest:
            action = "unchanged"
        else:
            action = (
                "downgrade" if compare_versions(manifest.version, row.version) < 0 else "upgrade"
            )
        state = plugin_state(row)
        current = manifest_of(row)
        installed = {
            "version": row.version,
            "state": state,
            "state_label": STATE_LABELS[state],
            # valores atuais da configuração (os sensíveis só dizem se existem)
            "config": {
                c["key"]: {"set": c["set"], "value": c["value"]}
                for c in (_config_view(row, current) if current else [])
            },
        }
        approved = set(row.approved_fingerprints or [])
    held = bundle.held
    return {
        "action": action,
        "installed": installed,
        "pending": [f.to_dict() for f in held if f.fingerprint not in approved],
        "carried": sum(1 for f in held if f.fingerprint in approved),
        "findings": [
            {**f.to_dict(), "approved": f.level == "held" and f.fingerprint in approved}
            for f in sorted_findings(bundle.findings)
        ],
    }


# --------------------------------------------------------------------------
# visões para API e console


def _config_view(row: Plugin, manifest: PluginManifest) -> list[dict[str, Any]]:
    stored = row.config or {}
    out = []
    for key, spec in manifest.config.items():
        value = stored.get(key)
        out.append(
            {
                "key": key,
                "description": spec.description,
                "sensitive": spec.sensitive,
                "required": spec.required,
                "default": spec.default,
                "set": bool(value),
                # o valor de um segredo nunca sai daqui, nem cifrado
                "value": None if spec.sensitive else value,
                "display": SecretBox.describe(value) if spec.sensitive else value,
            }
        )
    return out


def plugin_to_dict(session: Session, row: Plugin, *, detail: bool = False) -> dict[str, Any]:
    manifest = manifest_of(row)
    findings = sorted_findings(findings_of(row))
    approved = set(row.approved_fingerprints or [])
    state = plugin_state(row)
    data: dict[str, Any] = {
        "id": row.id,
        "name": row.name,
        "version": row.version,
        "description": row.description,
        "format": row.format,
        "origin": row.origin,
        "digest": row.digest,
        "state": state,
        "state_label": STATE_LABELS[state],
        "enabled": row.enabled,
        "installed_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
        "approved_by": row.approved_by,
        "approved_at": _iso(row.approved_at),
        "counts": counts(findings),
        "pending": len(pending_findings(row)),
        "missing": missing_config(row),
        "profiles": profiles_using(session, row.id),
    }
    if manifest is not None:
        data.update(
            author=manifest.author.model_dump(exclude_none=True) if manifest.author else None,
            license=manifest.license,
            homepage=manifest.homepage,
            keywords=manifest.keywords,
            requires=manifest.requires.model_dump(exclude_none=True),
            component_counts={
                "connectors": len(manifest.connectors),
                "agents": len(manifest.agents),
                "knowledge_bases": len(manifest.knowledge_bases),
                "playbooks": len(row.playbooks or []),
            },
        )
    if not detail:
        return data
    data["config"] = _config_view(row, manifest) if manifest else []
    data["findings"] = [
        {**f.to_dict(), "approved": f.level == "held" and f.fingerprint in approved}
        for f in findings
    ]
    data["components"] = {
        "connectors": [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "url": c.url,
                "allowed_tools": list(c.allowed_tools or []),
            }
            for c in plugin_owned(session, Connector, row.id)
        ],
        "agents": [
            {
                "id": a.id,
                "name": a.name,
                "description": a.description,
                "url": a.url,
                "allowed_skills": list(a.allowed_skills or []),
            }
            for a in plugin_owned(session, Agent, row.id)
        ],
        "knowledge_bases": [
            {"name": k.name, "description": k.description, "id": k.id}
            for k in plugin_owned(session, KnowledgeBase, row.id)
        ],
    }
    data["playbooks"] = [
        {
            "skill": p.skill,
            "description": p.description,
            "applies_to": list(p.applies_to),
            "triggers": list(p.triggers),
            "always": p.always,
            "body": p.body,
        }
        for p in playbooks_of(row)
    ]
    data["events"] = plugin_events(session, row.name)
    data["usage"] = usage_by_version(session, row.name)
    data["components_usage"] = usage_by_component(session, row.name)
    data["open_contracts"] = open_contracts(session, row.name)
    return data
