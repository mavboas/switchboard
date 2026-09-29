#!/usr/bin/env bash
# Sobe a stack local sem Docker: conectores MCP e agentes A2A de exemplo, console e router.
# Usa SQLite em ./data por padrão; para PostgreSQL, exporte SWITCHBOARD_DATABASE_URL antes.
set -euo pipefail
cd "$(dirname "$0")/.."

export SWITCHBOARD_DATABASE_URL="${SWITCHBOARD_DATABASE_URL:-sqlite:///./data/switchboard.db}"
export SWITCHBOARD_SECRET_KEY="${SWITCHBOARD_SECRET_KEY:-dev-local-troque-em-producao}"
export SWITCHBOARD_ROUTER_URL="${SWITCHBOARD_ROUTER_URL:-http://127.0.0.1:8080}"
# URL pela qual os agentes A2A mandam push notifications ao router
export SWITCHBOARD_PUBLIC_URL="${SWITCHBOARD_PUBLIC_URL:-http://127.0.0.1:8080}"
mkdir -p data

trap 'kill 0' EXIT INT TERM
uv run switchboard-agent credito --port 8101 &
uv run switchboard-agent chamados --port 8102 &
uv run switchboard-agent analise-credito --port 8201 &
uv run switchboard-agent risco --port 8202 &
SWITCHBOARD_PORT=8000 uv run switchboard-console &
sleep 3
SWITCHBOARD_PORT=8080 uv run switchboard-router &

echo
echo "  console: http://localhost:8000   router: http://localhost:8080/docs"
echo "  Ctrl+C encerra tudo."
echo
wait
