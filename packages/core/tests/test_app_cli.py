from __future__ import annotations

import json
from pathlib import Path

from switchboard import AgentCatalog, Switchboard
from switchboard.cli import main
from switchboard.testing import inproc_connector

KNOWLEDGE_DIR = Path(__file__).resolve().parents[3] / "examples" / "knowledge"

YAML = f"""
models:
  - name: offline
    provider: offline
agents:
  - name: calc
    url: http://calc.local/mcp
knowledge_bases:
  - name: manual
    paths: ["{KNOWLEDGE_DIR.as_posix()}"]
profiles:
  - name: default
    model: offline
    agents: [calc]
    knowledge_bases: [manual]
    min_score: 0.1
"""


async def test_switchboard_from_yaml(tmp_path, servers):
    config = tmp_path / "switchboard.yaml"
    config.write_text(YAML, encoding="utf-8")
    catalog = AgentCatalog(connector=inproc_connector(servers))
    async with Switchboard.from_yaml(config, catalog=catalog) as sb:
        assert sb.retriever.stats()["manual"] > 5
        answer = await sb.ask("Qual o horário de atendimento aos sábados?")
        assert answer.route == "direct" and "9h às 14h" in answer.answer
        delegated = await sb.ask("somar 20 e 22")
        assert delegated.route == "delegated" and delegated.answer == "resultado: 42"
        [info] = await sb.agents()
        assert info.status == "online"


def test_cli_check_and_ask(tmp_path, capsys):
    config = tmp_path / "switchboard.yaml"
    config.write_text(YAML.replace("agents: [calc]", "agents: []"), encoding="utf-8")
    assert main(["-c", str(config), "check"]) == 0
    assert "1 perfil" in capsys.readouterr().out
    assert main(["-c", str(config), "ask", "Qual o prazo máximo do empréstimo pessoal?"]) == 0
    out = capsys.readouterr().out
    assert "rota: resposta direta" in out and "60 meses" in out
    assert (
        main(["-c", str(config), "ask", "--json", "quais documentos preciso para pedir crédito?"])
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert data["route"] == "direct" and data["sources"]


def test_cli_reports_config_errors(tmp_path, capsys):
    assert main(["-c", str(tmp_path / "nao-existe.yaml"), "check"]) == 2
    assert "erro:" in capsys.readouterr().err
