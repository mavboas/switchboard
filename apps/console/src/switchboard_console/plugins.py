"""Plugins no console: subir um pacote, conferir o que ele traz, aprovar, configurar e acompanhar.

Um plugin é um pacote **declarativo** (conectores MCP, agentes A2A, bases de conhecimento e
playbooks no formato Agent Skills). O console lê o pacote com limites, mostra o que ele traz e o
que a varredura de segurança achou, e só instala quando o administrador confirma. Nada de um
plugin é executado. Um plugin só entra no roteamento quando está **ativo**: ligado, sem achado
pendente de aprovação e com a configuração obrigatória completa; e só os roteadores a que foi
ligado o usam.

Páginas em ``/plugins`` e API JSON em ``/api/plugins`` (as mesmas operações, para automação).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import anyio.to_thread
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field
from starlette.datastructures import FormData

from switchboard.errors import PluginConflict, PluginError, SwitchboardError
from switchboard.plugins import (
    DEFAULT_LIMITS,
    PluginBundle,
    PluginRejected,
    compare_versions,
    load_plugin,
)
from switchboard.storage import plugins as plugin_repo
from switchboard.storage import repo
from switchboard.storage.orm import Agent, Connector

from . import forms
from .state import Console, get_console, redirect

router = APIRouter()
api = APIRouter(prefix="/api/plugins", tags=["plugins"])

MiB = 1024 * 1024
EVENTS_ON_LIST = 12


# ---------------------------------------------------------------------------
# helpers


def _actor(console: Console) -> str:
    """Quem fica no histórico: o usuário do console (ou "console", se não há senha)."""
    settings = console.settings
    return settings.console_user if settings.console_password else "console"


def _limits() -> dict[str, int]:
    return {
        "archive_mb": DEFAULT_LIMITS.max_archive_bytes // MiB,
        "file_mb": DEFAULT_LIMITS.max_file_bytes // MiB,
        "files": DEFAULT_LIMITS.max_files,
    }


async def _load_upload(upload: Any) -> PluginBundle:
    """Lê o .zip enviado (com limite de tamanho) e analisa o pacote fora do event loop."""
    filename = getattr(upload, "filename", None)
    if not filename:
        raise PluginError("escolha o arquivo .zip do plugin")
    limit = DEFAULT_LIMITS.max_archive_bytes
    size = getattr(upload, "size", None) or 0
    if size > limit:
        raise PluginError(f"o arquivo tem {size / MiB:.1f} MB; o limite é {limit // MiB} MB")
    data = await upload.read(limit + 1)  # um byte a mais: o leitor do pacote recusa o excesso
    if not data:
        raise PluginError("o arquivo enviado está vazio")
    bundle = await anyio.to_thread.run_sync(load_plugin, data)
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    bundle.origin = f"upload {name}"[:200]
    return bundle


def _config_from_form(form: FormData) -> tuple[dict[str, str], list[str]]:
    """Campos ``config.CHAVE`` (valores) e ``clear.CHAVE`` (caixas de "remover")."""
    config: dict[str, str] = {}
    clear: list[str] = []
    for key, value in form.multi_items():
        if not isinstance(value, str):
            continue
        if key.startswith("config."):
            config[key.removeprefix("config.")] = value
        elif key.startswith("clear.") and value in ("on", "true", "1", "sim"):
            clear.append(key.removeprefix("clear."))
    return config, clear


def _component_names(session, plugin_id: int) -> list[str]:
    """Nomes dos conectores e agentes do plugin (para esquecer o que o console tinha em cache)."""
    return [
        item.name
        for kind in (Connector, Agent)
        for item in plugin_repo.plugin_owned(session, kind, plugin_id)
    ]


def _forget(console: Console, names: Iterable[str]) -> None:
    for name in names:
        console.connectors.invalidate(name)
        console.agents.invalidate(name)


def _list_items(session) -> list[dict[str, Any]]:
    usage: dict[str, list[dict[str, Any]]] = {}
    for item in plugin_repo.usage_by_version(session):
        usage.setdefault(item["plugin"], []).append(item)
    items = []
    for row in plugin_repo.list_plugins(session):
        data = plugin_repo.plugin_to_dict(session, row)
        versions = usage.get(row.name, [])
        data["usage"] = {
            "requests": sum(v["requests"] for v in versions),
            "errors": sum(v["errors"] for v in versions),
            "current": next((v for v in versions if v["version"] == row.version), None),
            "versions": len(versions),
        }
        items.append(data)
    return items


def event_summary(event: dict[str, Any]) -> str:
    """O detalhe guardado no histórico, em uma frase."""
    detail = event.get("detail") or {}
    action = event.get("action")
    bits: list[str] = []
    if action in ("install", "upgrade", "config") and not detail.get("keys"):
        if detail.get("previous") and action != "config":
            bits.append(f"a partir da versão {detail['previous']}")
        if detail.get("pending"):
            bits.append("aguardando aprovação: " + ", ".join(detail["pending"]))
        if detail.get("missing"):
            bits.append("falta configurar " + ", ".join(detail["missing"]))
        if detail.get("digest") and action != "config":
            bits.append(f"pacote {detail['digest']}")
    elif action == "config":
        bits.append("chaves: " + ", ".join(detail.get("keys") or []))
    elif action == "approve":
        bits.append(f"{detail.get('count', 0)} achado(s): " + ", ".join(detail.get("codes") or []))
    elif action == "rescan":
        if detail.get("new"):
            bits.append(f"{detail['new']} achado(s) novo(s)")
        if detail.get("pending"):
            bits.append("aguardando aprovação: " + ", ".join(detail["pending"]))
    elif action == "profiles":
        before = ", ".join(detail.get("before") or []) or "nenhum"
        after = ", ".join(detail.get("after") or []) or "nenhum"
        bits.append(f"{before} → {after}")
    elif action == "uninstall" and detail.get("open_contracts"):
        bits.append(f"{detail['open_contracts']} contrato(s) em aberto no momento")
    return "; ".join(bits)


def _with_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**e, "summary": event_summary(e)} for e in events]


def _detail(session, key: int | str) -> dict[str, Any] | None:
    """Tudo o que a página (e a API) de um plugin mostra."""
    row = plugin_repo.get_plugin(session, key)
    if row is None:
        return None
    data = plugin_repo.plugin_to_dict(session, row, detail=True)
    kb_ids = [k["id"] for k in data["components"]["knowledge_bases"]]
    stats = repo.kb_stats(session, kb_ids) if kb_ids else {}
    for kb in data["components"]["knowledge_bases"]:
        kb["stats"] = stats.get(kb["id"], {"documents": 0, "chunks": 0, "stale": 0})
    data["events"] = _with_summary(data["events"])
    return data


def _profiles_view(session, plugin_id: int) -> list[dict[str, Any]]:
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "enabled": p.enabled,
            "attached": any(pl.id == plugin_id for pl in p.plugins),
        }
        for p in repo.list_profiles(session)
    ]


async def _preview(console: Console, bundle: PluginBundle) -> dict[str, Any]:
    """O que o pacote traz e o que instalar faria agora (nada é gravado)."""
    info = await console.run_db(plugin_repo.plugin_preview, bundle)
    summary = bundle.summary()
    summary["findings"] = info.pop(
        "findings"
    )  # com a marca de "já aprovado" nas versões anteriores
    summary["preview"] = info
    return summary


def _installed_message(result: plugin_repo.InstallResult, profiles: list[str]) -> str:
    plugin = result.plugin
    if result.action == "install":
        head = f"Plugin {plugin.name} {plugin.version} instalado."
    elif result.action == "upgrade":
        verb = (
            "voltou da versão"
            if compare_versions(plugin.version, result.previous_version) < 0
            else "atualizado de"
        )
        head = f"Plugin {plugin.name} {verb} {result.previous_version} para {plugin.version}."
    else:
        head = (
            f"Plugin {plugin.name} {plugin.version}: o conteúdo é o mesmo que já estava instalado."
        )
    state = result.state
    if state == plugin_repo.STATE_PENDING:
        step = f"Falta aprovar {len(result.pending)} achado(s) da varredura antes de ele entrar em uso."
    elif state == plugin_repo.STATE_CONFIG:
        step = f"Falta configurar {', '.join(result.missing)}."
    elif state == plugin_repo.STATE_DISABLED:
        step = "Ele está desativado."
    elif profiles:
        step = f"Ele está ativo nos roteadores: {', '.join(profiles)}."
    else:
        step = "Ele está ativo: ligue-o a um roteador para o roteamento passar a usá-lo."
    return " ".join([head, step, *result.notes])


# ---------------------------------------------------------------------------
# páginas


@router.get("/plugins")
async def plugins_list(request: Request, console: Console = Depends(get_console)):
    def load(s):
        return _list_items(s), _with_summary(plugin_repo.plugin_events(s, limit=EVENTS_ON_LIST))

    items, events = await console.run_db(load)
    installed = {p["name"]: p["id"] for p in items}
    for event in events:
        event["plugin_id"] = installed.get(event["plugin"])
    return console.render(request, "plugins/list.html", nav="plugins", plugins=items, events=events)


def _new_page(
    console: Console,
    request: Request,
    *,
    preview: dict[str, Any] | None = None,
    error: str | None = None,
):
    return console.render(
        request,
        "plugins/new.html",
        nav="plugins",
        preview=preview,
        error=error,
        limits=_limits(),
        secrets_enabled=console.box.enabled,
    )


@router.get("/plugins/new")
async def plugins_new(request: Request, console: Console = Depends(get_console)):
    return _new_page(console, request)


@router.post("/plugins/validate")
async def plugins_validate(request: Request, console: Console = Depends(get_console)):
    """Lê e varre o pacote sem instalar nada: o que ele traz, os achados e o que mudaria."""
    form = await request.form()
    try:
        bundle = await _load_upload(form.get("file"))
    except PluginError as exc:
        return _new_page(console, request, error=str(exc))
    return _new_page(console, request, preview=await _preview(console, bundle))


@router.post("/plugins")
async def plugins_install(request: Request, console: Console = Depends(get_console)):
    form = await request.form()
    try:
        bundle = await _load_upload(form.get("file"))
    except PluginError as exc:
        return _new_page(console, request, error=str(exc))
    config, clear = _config_from_form(form)
    try:
        result = await console.plugins.install_bundle(
            bundle,
            config=config,
            clear=clear,
            actor=_actor(console),
            allow_downgrade=forms.checkbox(form, "allow_downgrade"),
        )
    except SwitchboardError as exc:  # inclui o pacote recusado, o downgrade e a config inválida
        message = (
            "O pacote tem achados que bloqueiam a instalação: corrija e envie de novo."
            if isinstance(exc, PluginRejected)
            else str(exc)
        )
        return _new_page(console, request, preview=await _preview(console, bundle), error=message)
    plugin = result.plugin
    names, profiles = await console.run_db(
        lambda s: (_component_names(s, plugin.id), plugin_repo.profiles_using(s, plugin.id))
    )
    _forget(console, names)
    return redirect(f"/plugins/{plugin.id}", ok=_installed_message(result, profiles))


@router.get("/plugins/{plugin_id}")
async def plugins_detail(request: Request, plugin_id: int, console: Console = Depends(get_console)):
    def load(s):
        data = _detail(s, plugin_id)
        return None if data is None else (data, _profiles_view(s, plugin_id))

    loaded = await console.run_db(load)
    if loaded is None:
        return redirect("/plugins", erro="Plugin não encontrado.")
    plugin, profiles = loaded
    return console.render(
        request,
        "plugins/detail.html",
        nav="plugins",
        plugin=plugin,
        profiles=profiles,
        secrets_enabled=console.box.enabled,
    )


@router.get("/plugins/{plugin_id}/package.zip")
async def plugins_package(plugin_id: int, console: Console = Depends(get_console)):
    found = await console.run_db(
        lambda s: (
            (p := plugin_repo.get_plugin(s, plugin_id)) and (p.name, p.version, bytes(p.package))
        )
    )
    if not found:
        return redirect("/plugins", erro="Plugin não encontrado.")
    name, version, package = found
    return Response(
        package,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}-{version}.zip"'},
    )


@router.post("/plugins/{plugin_id}/approve")
async def plugins_approve(
    plugin_id: int, digest: str = Form(""), console: Console = Depends(get_console)
):
    """O digest (da página que a pessoa revisou) faz a aprovação recusar um pacote trocado."""

    def run(s):
        row = plugin_repo.require_plugin(s, plugin_id)
        pending = len(plugin_repo.pending_findings(row))
        plugin_repo.approve_plugin(s, plugin_id, actor=_actor(console), digest=digest)
        return pending, plugin_repo.plugin_state(row)

    try:
        pending, state = await console.run_db(run)
    except SwitchboardError as exc:
        return redirect(f"/plugins/{plugin_id}", erro=str(exc))
    if not pending:
        return redirect(f"/plugins/{plugin_id}", ok="Não havia nada pendente de aprovação.")
    tail = (
        " O plugin está ativo."
        if state == plugin_repo.STATE_READY
        else f" Situação agora: {plugin_repo.STATE_LABELS[state].lower()}."
    )
    return redirect(f"/plugins/{plugin_id}", ok=f"{pending} achado(s) aprovado(s).{tail}")


@router.post("/plugins/{plugin_id}/enable")
async def plugins_enable(plugin_id: int, console: Console = Depends(get_console)):
    return await _set_enabled(console, plugin_id, True)


@router.post("/plugins/{plugin_id}/disable")
async def plugins_disable(plugin_id: int, console: Console = Depends(get_console)):
    return await _set_enabled(console, plugin_id, False)


async def _set_enabled(console: Console, plugin_id: int, enabled: bool):
    try:
        await console.run_db(
            lambda s: plugin_repo.set_plugin_enabled(s, plugin_id, enabled, actor=_actor(console))
        )
    except SwitchboardError as exc:
        return redirect(f"/plugins/{plugin_id}", erro=str(exc))
    message = (
        "Plugin ativado."
        if enabled
        else "Plugin desativado: os roteadores deixam de usá-lo em poucos segundos."
    )
    return redirect(f"/plugins/{plugin_id}", ok=message)


@router.post("/plugins/{plugin_id}/config")
async def plugins_config(request: Request, plugin_id: int, console: Console = Depends(get_console)):
    config, clear = _config_from_form(await request.form())

    def run(s):
        row = plugin_repo.require_plugin(s, plugin_id)
        before = _component_names(s, row.id)
        plugin_repo.set_plugin_config(
            s, plugin_id, config, box=console.box, clear=clear, actor=_actor(console)
        )
        return before + _component_names(s, row.id), plugin_repo.plugin_state(row)

    try:
        names, state = await console.run_db(run)
    except SwitchboardError as exc:
        return redirect(f"/plugins/{plugin_id}", erro=str(exc))
    _forget(console, names)
    tail = (
        ""
        if state == plugin_repo.STATE_READY
        else f" Situação: {plugin_repo.STATE_LABELS[state].lower()}."
    )
    return redirect(f"/plugins/{plugin_id}", ok=f"Configuração salva.{tail}")


@router.post("/plugins/{plugin_id}/profiles")
async def plugins_profiles(
    request: Request, plugin_id: int, console: Console = Depends(get_console)
):
    ids = forms.id_list(await request.form(), "profile_ids")
    try:
        attached = await console.run_db(
            lambda s: plugin_repo.set_plugin_profiles(s, plugin_id, ids, actor=_actor(console))
        )
    except SwitchboardError as exc:
        return redirect(f"/plugins/{plugin_id}", erro=str(exc))
    where = ", ".join(attached) if attached else "nenhum roteador"
    return redirect(
        f"/plugins/{plugin_id}",
        ok=f"Plugin ligado a: {where}. O router aplica a mudança em poucos segundos.",
    )


@router.post("/plugins/{plugin_id}/rescan")
async def plugins_rescan(plugin_id: int, console: Console = Depends(get_console)):
    try:
        result = await console.run_db(
            lambda s: _rescan(s, plugin_id, _actor(console)),
        )
    except SwitchboardError as exc:
        return redirect(f"/plugins/{plugin_id}", erro=str(exc))
    pending = len(result)
    message = "Varredura refeita."
    if pending:
        message += f" {pending} achado(s) aguardando aprovação."
    return redirect(f"/plugins/{plugin_id}", ok=message)


def _rescan(session, key: int | str, actor: str) -> list[str]:
    result = plugin_repo.rescan_plugin(session, key, actor=actor)
    return [f.code for f in result.pending]


@router.post("/plugins/{plugin_id}/sync")
async def plugins_sync(plugin_id: int, console: Console = Depends(get_console)):
    """Deixa as bases do plugin com exatamente os documentos do pacote guardado."""
    name = await console.run_db(lambda s: (p := plugin_repo.get_plugin(s, plugin_id)) and p.name)
    if not name:
        return redirect("/plugins", erro="Plugin não encontrado.")
    try:
        summary = await console.plugins.ingest(name)
    except SwitchboardError as exc:
        return redirect(f"/plugins/{plugin_id}", erro=str(exc))
    added = sum(v["added"] for v in summary.values())
    removed = sum(v["removed"] for v in summary.values())
    return redirect(
        f"/plugins/{plugin_id}",
        ok=f"Documentos sincronizados com o pacote: {added} indexado(s), {removed} removido(s).",
    )


@router.post("/plugins/{plugin_id}/delete")
async def plugins_delete(request: Request, plugin_id: int, console: Console = Depends(get_console)):
    force = forms.checkbox(await request.form(), "force")

    def run(s):
        names = _component_names(s, plugin_id)
        name = plugin_repo.uninstall_plugin(s, plugin_id, actor=_actor(console), force=force)
        return name, names

    try:
        name, names = await console.run_db(run)
    except SwitchboardError as exc:
        return redirect(f"/plugins/{plugin_id}", erro=str(exc))
    _forget(console, names)
    return redirect(
        "/plugins",
        ok=(
            f"Plugin {name} removido, com os conectores, agentes e bases dele. "
            "O histórico e o uso por versão continuam registrados."
        ),
    )


# ---------------------------------------------------------------------------
# API JSON


def _fail(exc: Exception, status: int = 400) -> HTTPException:
    if isinstance(exc, PluginConflict):
        return HTTPException(status_code=409, detail=str(exc))  # outra versão chegou antes
    if isinstance(exc, PluginRejected):
        return HTTPException(
            status_code=422,
            detail={
                "message": "plugin recusado: há achados que bloqueiam a instalação",
                "findings": [f.to_dict() for f in exc.findings],
            },
        )
    return HTTPException(status_code=status, detail=str(exc))


def _missing() -> HTTPException:
    return HTTPException(status_code=404, detail="plugin não encontrado")


def _require(session, ref: str):
    row = plugin_repo.get_plugin(session, ref)
    if row is None:
        raise _missing()
    return row


class ConfigIn(BaseModel):
    config: dict[str, str | None] = Field(
        default_factory=dict,
        description=("chave -> valor. Segredos: texto (cifrado) ou env:NOME; vazio mantém o atual"),
    )
    clear: list[str] = Field(default_factory=list, description="chaves cujo valor será removido")


class ProfilesIn(BaseModel):
    profiles: list[str] = Field(
        default_factory=list,
        description="nomes dos roteadores que usam o plugin (os demais deixam de usar)",
    )


@api.get("")
async def api_list(console: Console = Depends(get_console)):
    """Plugins instalados: situação, o que cada um traz e o uso."""
    return await console.run_db(_list_items)


@api.get("/usage")
async def api_usage(plugin: str | None = None, console: Console = Depends(get_console)):
    """Uso por plugin e versão: pedidos, usos por tipo (playbook, tool, skill, base), erros."""
    return await console.run_db(lambda s: plugin_repo.usage_by_version(s, plugin))


@api.get("/events")
async def api_events(
    plugin: str | None = None, limit: int = 50, console: Console = Depends(get_console)
):
    """Histórico (instalações, aprovações, mudanças); sobrevive à remoção do plugin."""
    limit = max(1, min(limit, 500))
    return await console.run_db(lambda s: plugin_repo.plugin_events(s, plugin, limit))


@api.post("/validate")
async def api_validate(
    file: UploadFile = File(..., description="o pacote .zip"),
    console: Console = Depends(get_console),
):
    """Lê e varre o pacote sem instalar: o que ele traz, os achados e o que mudaria."""
    try:
        bundle = await _load_upload(file)
    except PluginError as exc:
        raise _fail(exc) from exc
    return await _preview(console, bundle)


@api.post("", status_code=201)
async def api_install(
    response: Response,
    file: UploadFile = File(..., description="o pacote .zip"),
    config: str = Form("", description='JSON: {"CHAVE": "valor"}'),
    clear: str = Form("", description="chaves a remover, separadas por vírgula"),
    allow_downgrade: bool = Form(False),
    console: Console = Depends(get_console),
):
    """Instala (ou atualiza) um plugin. Pacote com achado que bloqueia: 422 com os achados."""
    try:
        values = json.loads(config) if config.strip() else {}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="config precisa ser um JSON válido") from exc
    if not isinstance(values, dict):
        raise HTTPException(status_code=400, detail="config precisa ser um objeto JSON")
    keys = [k.strip() for k in clear.split(",") if k.strip()]
    try:
        bundle = await _load_upload(file)
        result = await console.plugins.install_bundle(
            bundle,
            config={str(k): v for k, v in values.items()},
            clear=keys,
            actor=_actor(console),
            allow_downgrade=allow_downgrade,
        )
    except SwitchboardError as exc:
        raise _fail(exc) from exc
    name = result.plugin.name

    def load(s):
        data = _detail(s, name)
        return data, _component_names(s, result.plugin.id)

    detail, names = await console.run_db(load)
    _forget(console, names)
    if result.action != "install":
        response.status_code = 200
    return {
        "action": result.action,
        "previous_version": result.previous_version,
        "pending": [f.to_dict() for f in result.pending],
        "missing": result.missing,
        "notes": result.notes,
        "plugin": detail,
    }


@api.get("/{ref}")
async def api_get(ref: str, console: Console = Depends(get_console)):
    """Um plugin: achados, configuração (sem segredos), componentes, playbooks, uso e histórico."""
    data = await console.run_db(_detail, ref)
    if data is None:
        raise _missing()
    return data


@api.get("/{ref}/package")
async def api_package(ref: str, console: Console = Depends(get_console)):
    """O pacote normalizado (.zip determinístico) que está guardado."""
    found = await console.run_db(
        lambda s: (p := plugin_repo.get_plugin(s, ref)) and (p.name, p.version, bytes(p.package))
    )
    if not found:
        raise _missing()
    name, version, package = found
    return Response(
        package,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}-{version}.zip"'},
    )


@api.put("/{ref}/config")
async def api_config(ref: str, body: ConfigIn, console: Console = Depends(get_console)):
    def run(s):
        row = _require(s, ref)
        before = _component_names(s, row.id)
        try:
            plugin_repo.set_plugin_config(
                s,
                row.id,
                {k: v for k, v in body.config.items() if v is not None},
                box=console.box,
                clear=body.clear,
                actor=_actor(console),
            )
        except SwitchboardError as exc:
            raise _fail(exc) from exc
        return _detail(s, row.id), before + _component_names(s, row.id)

    data, names = await console.run_db(run)
    _forget(console, names)
    return data


class ApproveIn(BaseModel):
    digest: str | None = Field(
        default=None,
        description="digest do pacote que você revisou; se o pacote mudou, a aprovação é recusada",
    )


@api.post("/{ref}/approve")
async def api_approve(
    ref: str, body: ApproveIn | None = None, console: Console = Depends(get_console)
):
    """Aprova os achados que exigem aprovação (guarda quem aprovou e quando)."""

    def run(s):
        row = _require(s, ref)
        try:
            plugin_repo.approve_plugin(
                s, row.id, actor=_actor(console), digest=body.digest if body else None
            )
        except SwitchboardError as exc:
            raise _fail(exc) from exc
        return _detail(s, row.id)

    return await console.run_db(run)


@api.post("/{ref}/enable")
async def api_enable(ref: str, console: Console = Depends(get_console)):
    return await _api_enabled(console, ref, True)


@api.post("/{ref}/disable")
async def api_disable(ref: str, console: Console = Depends(get_console)):
    return await _api_enabled(console, ref, False)


async def _api_enabled(console: Console, ref: str, enabled: bool):
    def run(s):
        row = _require(s, ref)
        plugin_repo.set_plugin_enabled(s, row.id, enabled, actor=_actor(console))
        return _detail(s, row.id)

    return await console.run_db(run)


@api.put("/{ref}/profiles")
async def api_profiles(ref: str, body: ProfilesIn, console: Console = Depends(get_console)):
    """Liga o plugin exatamente aos roteadores indicados (por nome)."""

    def run(s):
        row = _require(s, ref)
        known = {p.name: p.id for p in repo.list_profiles(s)}
        missing = [n for n in body.profiles if n not in known]
        if missing:
            raise _fail(PluginError(f"roteador inexistente: {', '.join(missing)}"))
        plugin_repo.set_plugin_profiles(
            s, row.id, [known[n] for n in body.profiles], actor=_actor(console)
        )
        return _detail(s, row.id)

    return await console.run_db(run)


@api.post("/{ref}/rescan")
async def api_rescan(ref: str, console: Console = Depends(get_console)):
    """Refaz a varredura do pacote guardado (por exemplo, depois de atualizar o Switchboard)."""

    def run(s):
        row = _require(s, ref)
        try:
            plugin_repo.rescan_plugin(s, row.id, actor=_actor(console))
        except SwitchboardError as exc:
            raise _fail(exc) from exc
        return _detail(s, row.id)

    return await console.run_db(run)


@api.post("/{ref}/sync")
async def api_sync(ref: str, console: Console = Depends(get_console)):
    """Deixa as bases do plugin com exatamente os documentos do pacote guardado."""
    name = await console.run_db(lambda s: _require(s, ref).name)
    try:
        return await console.plugins.ingest(name)
    except SwitchboardError as exc:
        raise _fail(exc) from exc


@api.delete("/{ref}", status_code=204)
async def api_delete(ref: str, force: bool = False, console: Console = Depends(get_console)):
    """Remove o plugin, com os conectores, agentes e bases dele. Com contratos em aberto: 409."""

    def run(s):
        row = _require(s, ref)
        names = _component_names(s, row.id)
        try:
            plugin_repo.uninstall_plugin(s, row.id, actor=_actor(console), force=force)
        except SwitchboardError as exc:
            raise _fail(exc, 409) from exc
        return names

    _forget(console, await console.run_db(run))
