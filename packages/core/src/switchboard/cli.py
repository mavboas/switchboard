"""CLI do modo framework: ``switchboard ask | connectors | agents | check | plugin``."""

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
from .plugins import LEVELS, PluginBundle, format_findings, load_plugin, load_plugins

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
    path = _config_path(args.config)
    spec = load_yaml(path)
    print(
        f"ok: {len(spec.models)} modelo(s), {len(spec.connectors)} conector(es) MCP, "
        f"{len(spec.agents)} agente(s) A2A, {len(spec.knowledge_bases)} base(s), "
        f"{len(spec.plugins)} plugin(s), {len(spec.profiles)} perfil(is)"
    )
    for p in spec.profiles:
        decision = f" decisão={p.decision_model}" if p.decision_model else ""
        plugins = f" plugins={p.plugins}" if p.plugins else ""
        print(
            f"  - {p.name}: modelo={p.model}{decision} conectores={p.connectors} "
            f"agentes={p.agents} bases={p.knowledge_bases}{plugins}"
        )
    # os plugins são lidos e varridos aqui também: quem faz o check vê o que a subida veria
    loaded = {p.name: p for p in load_plugins(spec.plugins, path.parent)}
    for ref in spec.plugins:
        plugin = loaded.get(ref.name)
        if plugin is None:
            print(f"  plugin {ref.name}: desativado (enabled: false)")
            continue
        parts = _components(plugin.bundle)
        accepted = f"; aceitos por allow: {', '.join(sorted({f.code for f in plugin.accepted}))}"
        print(f"  plugin {ref.name} {plugin.version}: {parts}{accepted if plugin.accepted else ''}")
    return 0


# --------------------------------------------------------------------------
# plugin: validate | inspect | pack


def _components(bundle: PluginBundle) -> str:
    m = bundle.manifest
    if m is None:
        return "sem manifesto válido"
    documents = sum(len(docs) for docs in bundle.documents.values())
    return (
        f"{len(m.connectors)} conector(es), {len(m.agents)} agente(s), "
        f"{len(m.knowledge_bases)} base(s) ({documents} documento(s)), "
        f"{len(bundle.playbooks)} playbook(s)"
    )


COUNT_LABELS = {
    "block": "bloqueio(s)",
    "held": "exige(m) aprovação",
    "warning": "aviso(s)",
    "note": "informação(ões)",
}


def _count_line(bundle: PluginBundle) -> str:
    counts = dict.fromkeys(LEVELS, 0)
    for f in bundle.findings:
        counts[f.level] += 1
    return ", ".join(f"{counts[level]} {COUNT_LABELS[level]}" for level in LEVELS)


def _plugin_validate(args: argparse.Namespace) -> int:
    bundle = load_plugin(Path(args.path))
    held = bundle.held
    failed = (not bundle.ok) or (args.strict and bool(held))
    if args.json:
        print(
            json.dumps(
                {**bundle.summary(), "strict": args.strict, "passed": not failed},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1 if failed else 0
    if bundle.manifest is None:
        print(f"plugin inválido ({args.path}):")
    else:
        print(f"plugin {bundle.name} {bundle.version} ({bundle.format}): {_components(bundle)}")
    print(f"achados: {_count_line(bundle)}")
    if bundle.findings:
        print(format_findings(bundle.findings))
    if not bundle.ok:
        print("\nrecusado: corrija os achados que bloqueiam a instalação")
    elif held and args.strict:
        print("\nrecusado (--strict): há achados que exigiriam aprovação de quem instala")
    elif held:
        print("\nok, com ressalvas: quem instala terá de aprovar os achados 'exige aprovação'")
    else:
        print("\nok")
    return 1 if failed else 0


def _plugin_inspect(args: argparse.Namespace) -> int:
    bundle = load_plugin(Path(args.path))
    info = bundle.summary()
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0 if bundle.ok else 1
    if bundle.manifest is None:
        print(f"plugin inválido ({args.path}):")
        print(format_findings(bundle.findings))
        return 1
    print(f"{info['name']} {info['version']}  ({info['format']}, sha256 {info['digest'][:12]})")
    if info.get("description"):
        print(f"  {info['description']}")
    meta = []
    if info.get("author"):
        meta.append(f"autor: {info['author'].get('name', '')}")
    if info.get("license"):
        meta.append(f"licença: {info['license']}")
    if info.get("requires", {}).get("switchboard"):
        meta.append(f"requer switchboard {info['requires']['switchboard']}")
    if meta:
        print("  " + " · ".join(meta))
    if info["config"]:
        print("\nconfiguração (quem instala informa):")
        for c in info["config"]:
            flags = ["obrigatória" if c["required"] and c["default"] is None else "opcional"]
            if c["sensitive"]:
                flags.append("sensível")
            if c["default"] is not None:
                flags.append(f"padrão: {c['default']}")
            print(f"  {c['key']}  ({', '.join(flags)})  {c['description']}")
    if info["connectors"]:
        print("\nconectores MCP:")
        for c in info["connectors"]:
            tools = f"  tools: {', '.join(c['allowed_tools'])}" if c["allowed_tools"] else ""
            print(f"  {c['name']} → {c['url']}{tools}")
    if info["agents"]:
        print("\nagentes A2A:")
        for a in info["agents"]:
            skills = f"  skills: {', '.join(a['allowed_skills'])}" if a["allowed_skills"] else ""
            print(f"  {a['name']} → {a['url']}{skills}")
    if info["knowledge_bases"]:
        print("\nbases de conhecimento:")
        for k in info["knowledge_bases"]:
            print(f"  {k['name']}  ({k['documents']} documento(s))")
    if info["playbooks"]:
        print("\nplaybooks (skills que orientam o roteador):")
        for pb in info["playbooks"]:
            when = "sempre" if pb["always"] else f"gatilhos: {', '.join(pb['triggers']) or '—'}"
            scope = f"  aplica-se a: {', '.join(pb['applies_to'])}" if pb["applies_to"] else ""
            print(f"  {pb['skill']}  ({pb['chars']} caracteres; {when}){scope}")
    print(f"\nachados: {_count_line(bundle)}")
    if bundle.findings:
        print(format_findings(bundle.findings))
    return 0 if bundle.ok else 1


def _plugin_pack(args: argparse.Namespace) -> int:
    source = Path(args.path)
    bundle = load_plugin(source)
    bundle.require_ok()
    out = Path(args.output or f"{bundle.name}-{bundle.version}.zip")
    if source.is_dir() and out.resolve().is_relative_to(source.resolve()):
        print("erro: escolha um destino fora do diretório do plugin", file=sys.stderr)
        return 2
    data = bundle.pack()
    out.write_bytes(data)
    print(f"ok: {out} ({len(data)} bytes, sha256 {bundle.digest[:12]})")
    if bundle.held:
        print(
            f"aviso: {len(bundle.held)} achado(s) exigem aprovação de quem instalar; "
            f"veja: switchboard plugin validate {args.path}",
            file=sys.stderr,
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

    sub.add_parser("check", help="valida o YAML (e lê e varre os plugins ligados)")

    plugin = sub.add_parser("plugin", help="valida, inspeciona e empacota plugins")
    plugin_sub = plugin.add_subparsers(dest="plugin_command", required=True)
    validate = plugin_sub.add_parser(
        "validate", help="valida o manifesto e varre o conteúdo (diretório ou .zip)"
    )
    validate.add_argument("path", help="diretório ou .zip do plugin")
    validate.add_argument(
        "--strict",
        action="store_true",
        help="falha também com achados que exigem aprovação (instalável sem revisão humana)",
    )
    validate.add_argument("--json", action="store_true", help="resultado completo em JSON")
    inspect_ = plugin_sub.add_parser(
        "inspect", help="mostra o que o plugin traz: configuração, componentes e playbooks"
    )
    inspect_.add_argument("path", help="diretório ou .zip do plugin")
    inspect_.add_argument("--json", action="store_true", help="resumo completo em JSON")
    pack_ = plugin_sub.add_parser(
        "pack", help="gera o .zip normalizado (o mesmo conteúdo, sempre os mesmos bytes)"
    )
    pack_.add_argument("path", help="diretório do plugin")
    pack_.add_argument("-o", "--output", help="arquivo de saída (padrão: <nome>-<versão>.zip)")
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
        if args.command == "plugin":
            return {
                "validate": _plugin_validate,
                "inspect": _plugin_inspect,
                "pack": _plugin_pack,
            }[args.plugin_command](args)
        return _check(args)
    except SwitchboardError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
