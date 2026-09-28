from __future__ import annotations

import json
from pathlib import Path

import pytest

from switchboard import ConnectorCatalog, Switchboard
from switchboard.cli import main
from switchboard.config import load_yaml
from switchboard.contracts import ContractManager, MemoryContractStore
from switchboard.errors import ConfigError
from switchboard.testing import inproc_connector

KNOWLEDGE_DIR = Path(__file__).resolve().parents[3] / "examples" / "knowledge"

YAML = f"""
models:
  - name: offline
    provider: offline
connectors:
  - name: calc
    url: http://calc.local/mcp
agents:
  - name: risco
    url: http://risco.test
    push: false
knowledge_bases:
  - name: manual
    paths: ["{KNOWLEDGE_DIR.as_posix()}"]
profiles:
  - name: default
    model: offline
    connectors: [calc]
    agents: [risco]
    knowledge_bases: [manual]
    min_score: 0.1
    wait_s: 1
"""


def _switchboard(config, servers, network):
    directory = network.directory()
    manager = ContractManager(
        MemoryContractStore(), client=directory.client, poll_min_s=0.05, tick_s=0.05
    )
    return Switchboard.from_yaml(
        config,
        connectors=ConnectorCatalog(connector=inproc_connector(servers)),
        agents=directory,
        contracts=manager,
    )


async def test_switchboard_from_yaml(tmp_path, servers, network, risk_agent):
    config = tmp_path / "switchboard.yaml"
    config.write_text(YAML, encoding="utf-8")
    async with _switchboard(config, servers, network) as sb:
        assert sb.retriever.stats()["manual"] > 5
        answer = await sb.ask("Qual o horário de atendimento aos sábados?")
        assert answer.route == "direct" and "9h às 14h" in answer.answer
        tool = await sb.ask("somar 20 e 22")
        assert tool.route == "tool" and tool.answer == "resultado: 42"
        [connector] = await sb.connectors_status()
        assert connector.status == "online"
        [agent] = await sb.agents_status()
        assert agent.status == "online" and agent.skill("avaliar_risco").contract == "completo"

        # delegação ao agente A2A: a resposta sai quando o agente conclui (polling)
        risk_agent.on_send = lambda params: {
            "status": {"state": "TASK_STATE_COMPLETED"},
            "artifacts": [
                {
                    "artifactId": "a",
                    "parts": [{"data": {"risco": "baixo", "score": 800}}, {"text": "Risco baixo."}],
                }
            ],
        }
        delegated = await sb.ask(
            "avaliar o risco de fraude da operação de R$ 80 mil do cliente Maria Souza"
        )
        assert delegated.route == "delegated", delegated.to_dict()
        assert delegated.status == "completed" and "Risco baixo." in delegated.answer
        [contract] = await sb.run_contracts(delegated.run_id)
        assert contract.state == "concluido" and contract.input["valor"] == 80000

        # tarefa longa: volta "pending" e termina em segundo plano
        risk_agent.on_send = None
        pending = await sb.ask(
            "avaliar o risco de fraude da operação de R$ 90 mil do cliente João Lima", wait_s=0.1
        )
        assert pending.status == "pending" and "Encaminhei" in pending.answer
        risk_agent.complete(risk_agent.last_task_id, {"risco": "alto", "score": 120}, "Risco alto.")
        run = await sb.wait(pending.run_id, 5)
        assert run.status == "completed" and "Risco alto." in run.answer


def test_yaml_separates_connectors_from_agents(tmp_path):
    config = tmp_path / "old.yaml"
    config.write_text(
        "models: [{name: m, provider: offline}]\n"
        "agents: [{name: calc, url: 'http://calc/mcp', transport: streamable-http}]\n"
        "profiles: [{model: m}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="connectors"):
        load_yaml(config)
    config.write_text(
        "models: [{name: m, provider: offline}]\n"
        "connectors: [{name: calc, url: 'http://calc/mcp'}]\n"
        "profiles: [{model: m, agents: [calc]}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="está em connectors"):
        load_yaml(config)
    config.write_text(
        "models: [{name: m, provider: offline}, {name: j, provider: typesafe, model: jev-1.13.0}]\n"
        "profiles: [{model: j}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="modelo de decisão"):
        load_yaml(config)
    config.write_text(
        "models: [{name: m, provider: offline}, {name: j, provider: typesafe, model: jev-1.13.0}]\n"
        "profiles: [{model: m, decision_model: m}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="provider typesafe"):
        load_yaml(config)


def test_example_yaml_matches_the_schema():
    spec = load_yaml(KNOWLEDGE_DIR.parent / "switchboard.yaml")
    assert [c.name for c in spec.connectors] == ["credito", "chamados"]
    assert [a.name for a in spec.agents] == ["analise-credito", "risco"]
    default, hybrid = spec.profile("default"), spec.profile("hibrido")
    assert default.decision_model is None and spec.model(default.model).provider == "offline"
    assert spec.model(hybrid.decision_model).is_decision_model
    assert hybrid.connectors == ["credito", "chamados"]
    assert hybrid.agents == ["analise-credito", "risco"]


def test_cli_check_and_ask(tmp_path, capsys):
    config = tmp_path / "switchboard.yaml"
    config.write_text(
        YAML.replace("connectors: [calc]", "connectors: []").replace(
            "agents: [risco]", "agents: []"
        ),
        encoding="utf-8",
    )
    assert main(["-c", str(config), "check"]) == 0
    out = capsys.readouterr().out
    assert "1 perfil" in out and "1 conector(es) MCP" in out and "1 agente(s) A2A" in out
    assert main(["-c", str(config), "ask", "Qual o prazo máximo do empréstimo pessoal?"]) == 0
    out = capsys.readouterr().out
    assert "rota: resposta direta" in out and "60 meses" in out
    assert (
        main(["-c", str(config), "ask", "--json", "quais documentos preciso para pedir crédito?"])
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert data["route"] == "direct" and data["sources"] and data["status"] == "completed"


def test_cli_reports_config_errors(tmp_path, capsys):
    assert main(["-c", str(tmp_path / "nao-existe.yaml"), "check"]) == 2
    assert "erro:" in capsys.readouterr().err
