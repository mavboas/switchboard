"""Sobe um conector MCP ou um agente A2A de exemplo.

Conectores MCP (tools síncronas; endpoint em ``http://HOST:PORTA/mcp``)::

    switchboard-agent credito --port 8101
    switchboard-agent chamados --port 8102

Agentes A2A (card em ``/.well-known/agent-card.json``, JSON-RPC em ``/``)::

    switchboard-agent analise-credito --port 8201 --public-url http://localhost:8201
    switchboard-agent risco --port 8202 --push-hosts localhost,router
"""

from __future__ import annotations

import argparse
import os

from . import analise, chamados, credito, risco

MCP = {"credito": (credito.server, 8101), "chamados": (chamados.server, 8102)}
A2A = {"analise-credito": (analise.build, 8201), "risco": (risco.build, 8202)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="switchboard-agent", description=__doc__)
    parser.add_argument("agent", choices=sorted(MCP) + sorted(A2A))
    parser.add_argument("--host", default=os.environ.get("AGENT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--public-url",
        default=os.environ.get("AGENT_PUBLIC_URL"),
        help="URL pública do agente A2A (vai no Agent Card); padrão http://HOST:PORTA",
    )
    parser.add_argument(
        "--push-hosts",
        default=os.environ.get("AGENT_PUSH_HOSTS"),
        help="hosts aceitos nas URLs de push notification (vírgula); vazio = qualquer um",
    )
    args = parser.parse_args(argv)
    if args.agent in MCP:
        server, default_port = MCP[args.agent]
        port = args.port or int(os.environ.get("AGENT_PORT", default_port))
        server.run(
            "streamable-http", host=args.host, port=port, stateless_http=True, json_response=True
        )
        return
    build, default_port = A2A[args.agent]
    port = args.port or int(os.environ.get("AGENT_PORT", default_port))
    loopback = args.host in ("0.0.0.0", "::", "127.0.0.1", "::1")
    public = args.public_url or f"http://{'localhost' if loopback else args.host}:{port}"
    push_hosts = (
        [h.strip() for h in args.push_hosts.split(",") if h.strip()] if args.push_hosts else None
    )
    build(public, push_hosts=push_hosts).run(host=args.host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
