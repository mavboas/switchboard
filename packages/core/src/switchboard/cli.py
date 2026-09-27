"""CLI do modo framework: ``switchboard ask | agents | check``."""

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
    "delegated": "delegado a agente",
    "clarify": "pergunta de esclarecimento",
    "error": "erro",
}


def _config_path(value: str | None) -> Path:
    return Path(value or os.environ.get("SWITCHBOARD_CONFIG", "switchboard.yaml"))


async def _ask(args: argparse.Namespace) -> int:
    async with Switchboard.from_yaml(_config_path(args.config)) as sb:
        result = await sb.ask(args.message, profile=args.profile)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0 if result.route != "error" else 1
    print(
        f"rota: {ROUTE_LABELS.get(result.route, result.route)}  ·  modelo: {result.model}  ·  {result.latency_ms:.0f} ms"
    )
    if result.agent:
        print(
            f"agente: {result.agent} / {result.tool}  argumentos: {json.dumps(result.arguments, ensure_ascii=False)}"
        )
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


async def _agents(args: argparse.Namespace) -> int:
    async with Switchboard.from_yaml(_config_path(args.config)) as sb:
        infos = await sb.agents(args.profile, refresh=True)
    if not infos:
        print("nenhum agente configurado neste perfil")
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


def _check(args: argparse.Namespace) -> int:
    spec = load_yaml(_config_path(args.config))
    print(
        f"ok: {len(spec.models)} modelo(s), {len(spec.agents)} agente(s), "
        f"{len(spec.knowledge_bases)} base(s), {len(spec.profiles)} perfil(is)"
    )
    for p in spec.profiles:
        print(f"  - {p.name}: modelo={p.model} agentes={p.agents} bases={p.knowledge_bases}")
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

    agents = sub.add_parser("agents", help="descobre os agentes via MCP e lista as tools")
    agents.add_argument("-p", "--profile")

    sub.add_parser("check", help="valida o YAML")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "ask":
            return asyncio.run(_ask(args))
        if args.command == "agents":
            return asyncio.run(_agents(args))
        return _check(args)
    except SwitchboardError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
