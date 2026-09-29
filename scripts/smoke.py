#!/usr/bin/env python3
"""Smoke ponta a ponta da stack de demonstração (só biblioteca padrão).

Exercita os quatro caminhos do roteador contra a stack no ar (docker compose
ou scripts/dev.sh) e sai com código 1 no primeiro que falhar:

1. resposta direta com RAG;
2. tool de um conector MCP (síncrono);
3. delegação a um agente A2A que responde dentro da espera do pedido;
4. delegação longa: "pending" na hora, acompanhamento por /v1/runs e consolidação;
5. contrato que pede entrada no meio (needs_input) e é retomado pelo run_id.

    python3 scripts/smoke.py                           # http://localhost:8080 e :8000
    python3 scripts/smoke.py --router http://localhost:8080 --console ""

Com SWITCHBOARD_API_KEY no ambiente, envia ``Authorization: Bearer``; com
SWITCHBOARD_CONSOLE_PASSWORD, entra no console com o usuário ``admin``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

FINAL = {"completed", "failed"}


class SmokeError(Exception):
    pass


def _request(
    url: str, *, data: dict[str, Any] | None = None, headers: dict[str, str] | None = None
) -> tuple[int, Any]:
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method="POST" if body else "GET")
    req.add_header("Accept", "application/json")
    if body:
        req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    # a stack é local: não passa pelo proxy do ambiente
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=60) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read(), exc.code
    text = raw.decode("utf-8", "replace")
    try:
        return status, json.loads(text)
    except ValueError:
        return status, text


class Smoke:
    def __init__(self, router: str, console: str | None, timeout: float):
        self.router = router.rstrip("/")
        self.console = console.rstrip("/") if console else None
        self.timeout = timeout
        key = os.environ.get("SWITCHBOARD_API_KEY")
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        password = os.environ.get("SWITCHBOARD_CONSOLE_PASSWORD")
        token = base64.b64encode(f"admin:{password}".encode()).decode() if password else None
        self.console_headers = {"Authorization": f"Basic {token}"} if token else {}
        self.last_run: str | None = None  # uma execução já gravada, para abrir no console

    def chat(self, message: str, **extra: Any) -> tuple[int, dict[str, Any]]:
        status, data = _request(
            f"{self.router}/v1/chat", data={"message": message, **extra}, headers=self.headers
        )
        if not isinstance(data, dict) or status >= 400:
            raise SmokeError(f"/v1/chat respondeu {status}: {data}")
        return status, data

    def run(self, run_id: str) -> dict[str, Any]:
        status, data = _request(f"{self.router}/v1/runs/{run_id}", headers=self.headers)
        if status != 200 or not isinstance(data, dict):
            raise SmokeError(f"/v1/runs/{run_id} respondeu {status}: {data}")
        return data

    def wait(self, run_id: str, statuses: set[str]) -> dict[str, Any]:
        end = time.monotonic() + self.timeout
        while True:
            data = self.run(run_id)
            if data["status"] in statuses:
                return data
            if data["status"] in FINAL or time.monotonic() > end:
                raise SmokeError(
                    f"execução {run_id} ficou em {data['status']!r} (esperava {sorted(statuses)}): "
                    f"{data.get('answer') or data.get('error')}"
                )
            time.sleep(1)


def expect(condition: bool, message: str, data: Any = None) -> None:
    if not condition:
        detail = json.dumps(data, ensure_ascii=False, default=str)[:800] if data is not None else ""
        raise SmokeError(f"{message} {detail}".strip())


def contracts_of(data: dict[str, Any]) -> list[dict[str, Any]]:
    return list(data.get("contracts") or [])


def check_direct(s: Smoke) -> str:
    _, d = s.chat("Qual o horário de atendimento aos sábados?")
    expect(d["route"] == "direct" and d["status"] == "completed", "RAG: rota errada", d)
    expect(bool(d.get("sources")), "RAG: resposta sem fontes", d)
    return f"direta com {len(d['sources'])} fonte(s)"


def check_tool(s: Smoke) -> str:
    _, d = s.chat("Simule um empréstimo de R$ 50 mil em 24 meses a 1,5% ao mês")
    expect(d["route"] == "tool", "MCP: esperava a rota 'tool'", d)
    expect((d["agent"], d["tool"]) == ("credito", "simular_financiamento"), "MCP: tool errada", d)
    expect(d["arguments"].get("valor") == 50000, "MCP: argumentos errados", d["arguments"])
    expect("Parcela" in d["answer"], "MCP: resposta sem a simulação", d["answer"])
    return f"tool {d['agent']}/{d['tool']}"


def check_quick_delegation(s: Smoke) -> str:
    code, d = s.chat("Verifique o risco de fraude de uma operação de R$ 80 mil para Maria Souza")
    expect(d["route"] == "delegated", "A2A: esperava delegação", d)
    if d["status"] != "completed":  # agente lento na máquina de CI: acompanha pela execução
        d = s.wait(d["run_id"], {"completed"})
    [contract] = contracts_of(d)
    expect(contract["agent"] == "risco" and contract["state"] == "concluido", "A2A: contrato", d)
    return f"risco em {code} · contrato {contract['id']} concluído"


def check_long_delegation(s: Smoke) -> str:
    code, d = s.chat(
        "Analise um crédito de R$ 80 mil para Maria Souza em 24 meses, renda de R$ 12 mil",
        wait_s=0,
    )
    expect(code == 202 and d["status"] == "pending", "A2A: esperava 202 + pending", d)
    expect(d["links"]["run"] == f"/v1/runs/{d['run_id']}", "A2A: links da execução", d)
    run = s.wait(d["run_id"], {"completed"})
    [contract] = contracts_of(run)
    expect(contract["state"] == "concluido", "A2A: contrato não concluído", contract)
    expect(
        contract["input"]
        == {"cliente": "Maria Souza", "valor": 80000, "prazo_meses": 24, "renda_mensal": 12000},
        "A2A: entrada do contrato",
        contract["input"],
    )
    expect(contract["output"]["decisao"].startswith("aprovado"), "A2A: saída", contract["output"])
    spans = [sp["name"] for sp in run.get("spans") or []]
    for name in ("contrato: analise-credito/analisar_proposta", "consolidacao"):
        expect(name in spans, f"observabilidade: falta o span {name!r}", spans)
    s.last_run = d["run_id"]
    return f"pending → completed · {len(spans)} spans · decisão {contract['output']['decisao']}"


def check_needs_input(s: Smoke) -> str:
    _, d = s.chat(
        "Analise um crédito de R$ 800 mil para João Lima em 120 meses, renda de R$ 60 mil",
        wait_s=2,
    )
    run = d if d["status"] == "needs_input" else s.wait(d["run_id"], {"needs_input"})
    expect("garantia" in run["answer"], "needs_input: sem a pergunta do agente", run)
    _, resumed = s.chat("sim, um imóvel", run_id=d["run_id"], wait_s=0)
    expect(resumed["run_id"] == d["run_id"], "needs_input: retomou outra execução", resumed)
    final = s.wait(d["run_id"], {"completed"})
    [contract] = contracts_of(final)
    expect(contract["state"] == "concluido", "needs_input: contrato", contract)
    expect(contract["output"]["garantia"] == "imóvel", "needs_input: garantia", contract["output"])
    return f"needs_input → retomado → completed ({contract['output']['decisao']})"


def check_console(s: Smoke) -> str:
    assert s.console
    pages = ["/", "/connectors", "/agents", "/contracts"]
    if s.last_run:  # trace com os spans de cada contrato
        pages.append(f"/traces/{s.last_run}")
    for page in pages:
        status, _ = _request(f"{s.console}{page}", headers=s.console_headers)
        expect(status == 200, f"console: {page} respondeu {status}")
    return f"{len(pages)} páginas"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--router", default="http://localhost:8080")
    parser.add_argument("--console", default="http://localhost:8000", help='"" para pular')
    parser.add_argument("--timeout", type=float, default=90.0, help="espera máxima por execução")
    args = parser.parse_args(argv)
    s = Smoke(args.router, args.console or None, args.timeout)
    checks = [
        ("resposta direta (RAG)", check_direct),
        ("tool MCP", check_tool),
        ("agente A2A rápido", check_quick_delegation),
        ("agente A2A demorado", check_long_delegation),
        ("contrato com pergunta", check_needs_input),
    ]
    for label, check in checks:
        started = time.monotonic()
        try:
            detail = check(s)
        except (SmokeError, KeyError, ValueError, TypeError) as exc:
            print(f"FALHOU {label}: {exc}", file=sys.stderr)
            return 1
        print(f"ok  {label}: {detail} ({time.monotonic() - started:.1f} s)")
    if s.console:
        try:
            print(f"ok  console: {check_console(s)}")
        except SmokeError as exc:
            print(f"FALHOU console: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
