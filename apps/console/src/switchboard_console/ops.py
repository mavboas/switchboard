"""Páginas de operação: playground, execuções (com spans e contratos) e contratos.

O console lê execuções e contratos direto do banco, mas quem manda nos
contratos é o router (ele recebe os push e roda o supervisor). Por isso as
ações que mexem numa execução — enviar mensagem, responder a um agente,
cancelar — passam pela API do router.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from switchboard.contracts import states
from switchboard.storage import repo

from .state import RUN_STATUS_LABELS, Console, get_console, redirect

router = APIRouter()

OPEN_RUN_STATUSES = ("pending", "needs_input", "consolidating")


# ---------------------------------------------------------------------------
# waterfall de spans


def _when(value: Any) -> datetime | None:
    if not value:
        return None
    moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def waterfall(spans: list[dict[str, Any]], *, now: datetime | None = None) -> dict[str, Any]:
    """Spans em árvore (pai antes dos filhos) com a posição de cada barra na linha do tempo.

    Spans ainda abertos (um contrato em andamento, por exemplo) vão até ``now``.
    """
    now = now or datetime.now(UTC)
    items = [s for s in spans if _when(s.get("started_at")) is not None]
    if not items:
        return {"rows": [], "total_ms": 0.0}
    ids = {s["id"] for s in items}
    starts = {s["id"]: _when(s["started_at"]) for s in items}
    ends = {s["id"]: _when(s.get("ended_at")) for s in items}
    t0 = min(starts.values())
    t1 = max(max((ends[i] or now) for i in ids), t0)
    total_ms = max((t1 - t0).total_seconds() * 1000, 1.0)

    children: dict[str | None, list[dict[str, Any]]] = {}
    for span in items:
        parent = span.get("parent_id") if span.get("parent_id") in ids else None
        children.setdefault(parent, []).append(span)
    for group in children.values():
        group.sort(key=lambda s: starts[s["id"]])

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(span: dict[str, Any], depth: int) -> None:
        if span["id"] in seen:
            return
        seen.add(span["id"])
        start, end = starts[span["id"]], ends[span["id"]]
        stop = end or now
        duration = max((stop - start).total_seconds() * 1000, 0.0)
        rows.append(
            {
                **span,
                "depth": depth,
                "open": end is None,
                "offset_ms": (start - t0).total_seconds() * 1000,
                "elapsed_ms": duration,
                "left": round((start - t0).total_seconds() * 1000 / total_ms * 100, 3),
                "width": round(max(duration / total_ms * 100, 0.4), 3),
            }
        )
        for child in children.get(span["id"], []):
            visit(child, depth + 1)

    for root in children.get(None, []):
        visit(root, 0)
    return {"rows": rows, "total_ms": total_ms}


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


async def _proxy(console: Console, method: str, path: str, **kwargs) -> Response:
    try:
        resp = await console.router_request(method, path, **kwargs)
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
    return JSONResponse(data, status_code=resp.status_code)


@router.post("/playground/send")
async def playground_send(request: Request, console: Console = Depends(get_console)) -> Response:
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": "JSON inválido"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON inválido"}, status_code=400)
    payload: dict[str, Any] = {
        "profile": body.get("profile"),
        "messages": body.get("messages") or [],
    }
    if body.get("run_id"):
        payload["run_id"] = str(body["run_id"])  # resposta a um agente que pediu informação
    if isinstance(body.get("wait_s"), (int, float)):
        payload["wait_s"] = float(body["wait_s"])
    return await _proxy(console, "POST", "/v1/chat", json=payload)


@router.get("/playground/runs/{run_id}")
async def playground_run(run_id: str, console: Console = Depends(get_console)) -> Response:
    return await _proxy(console, "GET", f"/v1/runs/{run_id}")


@router.post("/playground/runs/{run_id}/cancel")
async def playground_cancel(run_id: str, console: Console = Depends(get_console)) -> Response:
    return await _proxy(console, "POST", f"/v1/runs/{run_id}/cancel")


# ---------------------------------------------------------------------------
# execuções


def trace_row(row) -> dict[str, Any]:
    data = repo.trace_to_dict(row)
    data["created"] = row.created_at
    return data


@router.get("/traces")
async def traces_list(
    request: Request,
    profile: str | None = None,
    route: str | None = None,
    status: str | None = None,
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
            status=status or None,
            limit=size + 1,
            offset=(page - 1) * size,
        )
        return [trace_row(t) for t in rows], repo.enabled_profile_names(s)

    rows, names = await console.run_db(load)
    return console.render(
        request,
        "traces/list.html",
        nav="traces",
        traces=rows[:size],
        has_next=len(rows) > size,
        page=page,
        profiles=names,
        statuses=RUN_STATUS_LABELS,
        filters={"profile": profile or "", "route": route or "", "status": status or ""},
    )


def _refresh_s(running: bool, *, waiting: bool) -> int | None:
    """De quanto em quanto a página se atualiza: rápido com o agente trabalhando,
    devagar esperando o usuário (pode levar até o prazo do contrato)."""
    if not running:
        return None
    return 15 if waiting else 3


@router.get("/traces/{trace_id}")
async def traces_detail(request: Request, trace_id: str, console: Console = Depends(get_console)):
    trace = await console.run_db(repo.run_details, trace_id)
    if trace is None:
        return redirect("/traces", erro="Execução não encontrada.")
    running = trace["status"] in OPEN_RUN_STATUSES or any(
        c["state"] in states.OPEN for c in trace["contracts"]
    )
    return console.render(
        request,
        "traces/detail.html",
        nav="traces",
        trace=trace,
        running=running,
        refresh_s=_refresh_s(running, waiting=trace["status"] == "needs_input"),
        spans=waterfall(trace["spans"]),
    )


@router.post("/traces/{trace_id}/cancel")
async def traces_cancel(trace_id: str, console: Console = Depends(get_console)):
    """Cancela os contratos em aberto da execução (quem cancela é o router)."""
    try:
        resp = await console.router_request("POST", f"/v1/runs/{trace_id}/cancel")
    except httpx.HTTPError as exc:
        return redirect(
            f"/traces/{trace_id}",
            erro=f"router inacessível em {console.settings.router_url} ({exc.__class__.__name__})",
        )
    if resp.status_code >= 400:
        return redirect(
            f"/traces/{trace_id}", erro=f"o router recusou o cancelamento (HTTP {resp.status_code})"
        )
    canceled = resp.json().get("canceled") or []
    return redirect(f"/traces/{trace_id}", ok=f"{len(canceled)} contrato(s) cancelado(s).")


# ---------------------------------------------------------------------------
# contratos


@router.get("/contracts")
async def contracts_list(
    request: Request,
    state: str | None = None,
    agent: str | None = None,
    profile: str | None = None,
    page: int = 1,
    console: Console = Depends(get_console),
):
    page = max(page, 1)
    size = 25

    def load(s):
        rows = repo.query_contracts(
            s,
            state=state or None,
            agent=agent or None,
            profile=profile or None,
            limit=size + 1,
            offset=(page - 1) * size,
        )
        return rows, repo.contract_state_counts(s), sorted(repo.contract_agent_counts(s))

    rows, counts, agents = await console.run_db(load)
    return console.render(
        request,
        "contracts/list.html",
        nav="contracts",
        contracts=rows[:size],
        has_next=len(rows) > size,
        page=page,
        counts=counts,
        agents=agents,
        filters={"state": state or "", "agent": agent or "", "profile": profile or ""},
    )


@router.get("/contracts/{contract_id}")
async def contracts_detail(
    request: Request, contract_id: str, console: Console = Depends(get_console)
):
    contract = await console.run_db(repo.contract_details, contract_id)
    if contract is None:
        return redirect("/contracts", erro="Contrato não encontrado.")
    running = contract["state"] in states.OPEN
    return console.render(
        request,
        "contracts/detail.html",
        nav="contracts",
        contract=contract,
        running=running,
        refresh_s=_refresh_s(running, waiting=contract["state"] == states.INPUT_REQUIRED),
    )
