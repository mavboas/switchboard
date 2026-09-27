"""Sobe um agente MCP de exemplo via Streamable HTTP.

Uso: ``switchboard-agent credito --port 8101`` (ou ``chamados``).
O endpoint MCP fica em ``http://HOST:PORTA/mcp``.
"""

from __future__ import annotations

import argparse
import os

from . import chamados, credito

AGENTS = {"credito": (credito.server, 8101), "chamados": (chamados.server, 8102)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="switchboard-agent", description=__doc__)
    parser.add_argument("agent", choices=sorted(AGENTS))
    parser.add_argument("--host", default=os.environ.get("AGENT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    server, default_port = AGENTS[args.agent]
    port = args.port or int(os.environ.get("AGENT_PORT", default_port))
    server.run(
        "streamable-http", host=args.host, port=port, stateless_http=True, json_response=True
    )


if __name__ == "__main__":  # pragma: no cover
    main()
