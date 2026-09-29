"""CLI do modo framework: ``switchboard ask | connectors | agents | check``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .app import Switchboard
from .config import load_yaml
from .errors import SwitchboardError

ROUTE_LABELS = {
    "direct": "resposta direta",
    "tool": "tool MCP",
    "delegated": "delegado a agente (A2A)",
    "clarify": "pergunta de esclarecimento",
    "error": "erro",
}


def _config_path(value: str | None) -> Path:
    return Path(value or os.environ.get("SWITCHBOARD_CONFIG", "switchboard.yaml"))


async def _ask(args: argparse.Namespace) -> int:
    async with Switchboard.from_yaml(_config_path(args.config)) as sb:
        result = await sb.ask(args.message, profile=args.profile)
        final_answer = None
        contracts = []
        if result.status == "pending" and args.wait > 0:
            if not args.json:
                print(
                    f"em andamento (execução {result.run_id}); aguardando até {args.wait:g} s…",
                    file=sys.stderr,
                )
            run = await sb.wait(result.run_id, args.wait)
            if run is not None:
                result.status = {"completed": "completed", "needs_input": "needs_input"}.get(
                    run.status, result.status
                )
                final_answer = run.answer
        if result.route == "delegated":
            contracts = [c.to_dict() for c in await sb.run_contracts(result.run_id)]
    if final_answer is not None:
        result.answer = final_answer
    if args.json:
        data = result.to_dict()
        if contracts:
            data["contracts"] = contracts
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        return 0 if result.route != "error" else 1
    decided = f" · decidido por {result.decided_by}" if result.decided_by else ""
    if result.confidence is not None:
        decided += f" ({result.confidence:.2f})"
    print(
        f"rota: {ROUTE_LABELS.get(result.route, result.route)} · status: {result.status}{decided} · "
        f"{result.latency_ms:.0f} ms"
    )
    if result.route == "tool" and result.agent:
        print(
            f"tool: {result.agent}/{result.tool}  argumentos: {json.dumps(result.arguments, ensure_ascii=False)}"
        )
    for c in contracts:
        print(f"contrato {c['id']}: {c['agent']}/{c['skill']} ({c['kind']}) → {c['state']}")
    if result.reason:
        print(f"motivo: {result.reason}")
    print()
    print(result.answer)
    if result.sources:
        print("\nfontes:")
        for s in result.sources:
            section = f" › {s.section}" if s.section else ""
            print(f"  [{s.index}] {s.kb} · {s.document}{section} (score {s.score:.2f})")
    for warning in result.warnings:
        print(f"\naviso: {warning}", file=sys.stderr)
    return 0 if result.route != "error" else 1


async def _connectors(args: argparse.Namespace) -> int:
    async with Switchboard.from_yaml(_config_path(args.config)) as sb:
        infos = await sb.connectors_status(args.profile, refresh=True)
    if not infos:
        print("nenhum conector MCP configurado neste perfil")
        return 0
    for info in infos:
        status = "ONLINE " if info.status == "online" else "OFFLINE"
        print(f"{status} {info.name}  {info.url}  ({info.latency_ms:.0f} ms)")
        if info.error:
            print(f"        erro: {info.error}")
        for tool in info.tools:
            required = ", ".join(tool.input_schema.get("required") or [])
            print(f"        - {tool.name}({required}): {tool.description}")
    return 0 if all(i.status == "online" for i in infos) else 1


async def _agents(args: argparse.Namespace) -> int:
    async with Switchboard.from_yaml(_config_path(args.config)) as sb:
        infos = await sb.agents_status(args.profile, refresh=True)
    if not infos:
        print("nenhum agente A2A configurado neste perfil")
        return 0
    for info in infos:
        status = "ONLINE " if info.status == "online" else "OFFLINE"
        extra = (
            f"A2A {info.protocol_version} · push={'sim' if info.push else 'não'}"
            if info.status == "online"
            else ""
        )
        print(f"{status} {info.name}  {info.url}  ({info.latency_ms:.0f} ms) {extra}")
        if info.error:
            print(f"        erro: {info.error}")
        for skill in info.skills:
            flag = "contrato completo" if skill.contract == "completo" else "contrato básico"
            print(f"        - {skill.id} [{flag}]: {skill.description}")
            for problem in skill.problems:
                print(f"          problema: {problem}")
    return 0 if all(i.status == "online" for i in infos) else 1


def _check(args: argparse.Namespace) -> int:
    spec = load_yaml(_config_path(args.config))
    print(
        f"ok: {len(spec.models)} modelo(s), {len(spec.connectors)} conector(es) MCP, "
        f"{len(spec.agents)} agente(s) A2A, {len(spec.knowledge_bases)} base(s), "
        f"{len(spec.profiles)} perfil(is)"
    )
    for p in spec.profiles:
        decision = f" decisão={p.decision_model}" if p.decision_model else ""
        print(
            f"  - {p.name}: modelo={p.model}{decision} conectores={p.connectors} "
            f"agentes={p.agents} bases={p.knowledge_bases}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="switchboard", description="Switchboard em modo framework (YAML)"
    )
    parser.add_argument(
        "-c", "--config", help="arquivo YAML (padrão: $SWITCHBOARD_CONFIG ou switchboard.yaml)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="envia uma pergunta ao roteador")
    ask.add_argument("message")
    ask.add_argument("-p", "--profile", help="perfil (padrão: o primeiro do YAML)")
    ask.add_argument("--json", action="store_true", help="saída completa em JSON (inclui o trace)")
    ask.add_argument(
        "--wait",
        type=float,
        default=120.0,
        help="segundos para esperar agentes A2A depois da espera do perfil (0 = não esperar)",
    )

    connectors = sub.add_parser("connectors", help="descobre os conectores MCP e lista as tools")
    connectors.add_argument("-p", "--profile")

    agents = sub.add_parser("agents", help="descobre os agentes A2A (Agent Card) e lista as skills")
    agents.add_argument("-p", "--profile")

    sub.add_parser("check", help="valida o YAML")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "ask":
            return asyncio.run(_ask(args))
        if args.command == "connectors":
            return asyncio.run(_connectors(args))
        if args.command == "agents":
            return asyncio.run(_agents(args))
        return _check(args)
    except SwitchboardError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
