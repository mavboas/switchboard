"""A2A (protocolo, card, diretório) e contratos (termos, estados, gerenciador, motor)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from switchboard.a2a import AgentCard, parse_stream_event
from switchboard.a2a.protocol import TaskState, merge_artifact, normalize_state
from switchboard.config import AgentSpec, ModelSpec, ProfileSpec
from switchboard.contracts import (
    CONTRACT_EXTENSION_URI,
    Consolidation,
    ContractManager,
    MemoryContractStore,
    OpenRequest,
    PushRejected,
    RunRecord,
    coerce_to_schema,
    schema_hash,
    skills_from_extension,
    states,
)
from switchboard.errors import ContractError
from switchboard.llm import OfflineChat
from switchboard.routing import ResolvedProfile, RouterEngine
from switchboard.testing import FakeA2AAgent, FakeNetwork, ScriptedChat
from switchboard.tracing import new_trace_id

# --------------------------------------------------------------------------
# protocolo


def test_stream_events_v1_and_legacy_shapes():
    task = parse_stream_event(
        {
            "task": {
                "id": "t",
                "contextId": "c",
                "status": {"state": "TASK_STATE_WORKING", "message": {"parts": [{"text": "oi"}]}},
            }
        }
    )
    assert (task.kind, task.task_id, task.state, task.status_text) == (
        "task",
        "t",
        TaskState.WORKING,
        "oi",
    )
    status = parse_stream_event(
        {
            "statusUpdate": {
                "taskId": "t",
                "status": {"state": "TASK_STATE_COMPLETED", "timestamp": "2026-01-01T00:00:00Z"},
            }
        }
    )
    assert (status.kind, status.state, status.timestamp) == (
        "status",
        TaskState.COMPLETED,
        "2026-01-01T00:00:00Z",
    )
    art = parse_stream_event(
        {
            "artifactUpdate": {
                "taskId": "t",
                "artifact": {"artifactId": "a", "parts": [{"data": {"x": 1}}]},
                "append": True,
            }
        }
    )
    assert art.kind == "artifact" and art.append
    msg = parse_stream_event(
        {"message": {"messageId": "m", "role": "ROLE_AGENT", "parts": [{"text": "pronto"}]}}
    )
    assert (msg.kind, msg.status_text) == ("message", "pronto")
    # 0.3: estado em minúsculas, evento com "kind"
    assert normalize_state("input-required") == TaskState.INPUT_REQUIRED
    legacy = parse_stream_event(
        {"kind": "status-update", "taskId": "t", "status": {"state": "completed"}}
    )
    assert legacy.state == TaskState.COMPLETED
    with pytest.raises(ValueError):
        parse_stream_event({"nada": 1})
    merged = merge_artifact(
        [{"artifactId": "a", "parts": [{"text": "1"}]}],
        {"artifactId": "a", "parts": [{"text": "2"}]},
        append=True,
    )
    assert [p["text"] for p in merged[0]["parts"]] == ["1", "2"]


def test_card_parsing_and_rpc_origin_check():
    from switchboard.a2a import agent_info_from_card
    from switchboard.errors import AgentError

    agent = FakeA2AAgent()
    card = AgentCard.parse(agent.card())
    assert card.jsonrpc_interface().url == "http://risco.test/" and card.extension(
        CONTRACT_EXTENSION_URI
    )
    info = agent_info_from_card(AgentSpec(name="risco", url="http://risco.test"), card)
    assert (
        info.rpc_url == "http://risco.test/" and info.skill("avaliar_risco").contract == "completo"
    )
    # o card não pode desviar o roteador para outro host
    hostile = dict(
        agent.card(),
        supportedInterfaces=[
            {
                "url": "http://169.254.169.254/",
                "protocolBinding": "JSONRPC",
                "protocolVersion": "1.0",
            }
        ],
    )
    with pytest.raises(AgentError, match="fora do host cadastrado"):
        agent_info_from_card(
            AgentSpec(name="risco", url="http://risco.test"), AgentCard.parse(hostile)
        )
    allowed = agent_info_from_card(
        AgentSpec(name="risco", url="http://risco.test", allow_cross_origin=True),
        AgentCard.parse(hostile),
    )
    assert allowed.rpc_url == "http://169.254.169.254/"
    # localhost e 127.0.0.1 na mesma porta são a mesma máquina; outra porta não
    local = dict(
        agent.card(),
        supportedInterfaces=[
            {
                "url": "http://127.0.0.1:8201/",
                "protocolBinding": "JSONRPC",
                "protocolVersion": "1.0",
            }
        ],
    )
    same = agent_info_from_card(
        AgentSpec(name="risco", url="http://localhost:8201"), AgentCard.parse(local)
    )
    assert same.rpc_url == "http://127.0.0.1:8201/"
    with pytest.raises(AgentError, match="fora do host cadastrado"):
        agent_info_from_card(
            AgentSpec(name="risco", url="http://localhost:9999"), AgentCard.parse(local)
        )
    only_grpc = dict(
        agent.card(),
        supportedInterfaces=[
            {"url": "http://risco.test", "protocolBinding": "GRPC", "protocolVersion": "1.0"}
        ],
    )
    with pytest.raises(AgentError, match="JSON-RPC 1.x"):
        agent_info_from_card(
            AgentSpec(name="risco", url="http://risco.test"), AgentCard.parse(only_grpc)
        )
    # allowlist de skills
    filtered = agent_info_from_card(
        AgentSpec(name="risco", url="http://risco.test", allowed_skills=["outra"]), card
    )
    assert filtered.skills == []


def test_terms_hash_is_stable_across_protobuf_numbers():
    schema = {"type": "object", "properties": {"n": {"type": "integer", "maximum": 480}}}
    via_struct = {"properties": {"n": {"maximum": 480.0, "type": "integer"}}, "type": "object"}
    assert schema_hash(schema) == schema_hash(via_struct)
    assert coerce_to_schema({"n": 24.0, "outro": 2.0}, schema) == {"n": 24, "outro": 2.0}
    terms, problems = skills_from_extension(
        {
            "skills": {
                "ok": {"input_schema": schema, "output_schema": schema},
                "ruim": {"input_schema": {"type": "banana"}},
                "vazia": {},
            }
        }
    )
    assert set(terms) == {"ok"} and terms["ok"].complete
    assert "ruim" in problems and "vazia" in problems


# --------------------------------------------------------------------------
# gerenciador de contratos com o agente falso


class Harness:
    def __init__(self, agent: FakeA2AAgent | None = None, **manager_kw):
        self.agent = agent or FakeA2AAgent()
        self.network = FakeNetwork(self.agent)
        self.directory = self.network.directory()
        self.store = MemoryContractStore()
        self.spec = AgentSpec(name=self.agent.name, url=self.agent.url, push=True)
        self.manager = ContractManager(
            self.store,
            client=self.directory.client,
            agent_resolver=self._resolve,
            poll_min_s=0.01,
            **manager_kw,
        )

    async def _resolve(self, name):
        return self.spec

    async def open(self, arguments=None, *, skill="avaliar_risco", deadline_s=60.0):
        info = await self.directory.describe(self.spec, refresh=True)
        run_id = new_trace_id()
        await self.manager.begin_run(RunRecord(id=run_id, profile="p", question="q"))
        contract = await self.manager.open(
            OpenRequest(
                run_id,
                "p",
                self.spec,
                info,
                info.skill(skill),
                arguments or {"cliente": "Ana Lima", "valor": 1000},
                "avalie",
                deadline_s,
            )
        )
        return run_id, contract


async def test_open_sends_terms_and_validates_input():
    h = Harness()
    run_id, contract = await h.open()
    sent = h.agent.sent[0]
    message = sent["message"]
    terms = message["metadata"][CONTRACT_EXTENSION_URI]
    assert terms["contract_id"] == contract.id and terms["skill"] == "avaliar_risco"
    assert terms["input_schema_sha256"] == contract.input_hash and terms["deadline"]
    assert message["extensions"] == [CONTRACT_EXTENSION_URI]
    assert message["parts"][0] == {
        "data": {"cliente": "Ana Lima", "valor": 1000},
        "mediaType": "application/json",
    }
    assert sent["configuration"]["returnImmediately"] is True
    assert "taskPushNotificationConfig" not in sent["configuration"]  # sem public_url: só polling
    assert sent["metadata"]["traceparent"] == f"00-{run_id}-{contract.span_id}-01"
    headers = h.agent.headers[0]
    assert headers["a2a-extensions"] == CONTRACT_EXTENSION_URI and headers[
        "traceparent"
    ].startswith("00-")
    # prazo: o menor entre o compromisso da skill (120 s) e o do perfil (60 s)
    assert 59 <= (contract.deadline_at - contract.created_at).total_seconds() <= 60
    with pytest.raises(ContractError, match="entrada fora do contrato"):
        await h.open({"cliente": "A", "valor": -1})


async def test_invalid_output_breaches_the_contract():
    h = Harness()
    run_id, contract = await h.open()
    h.agent.complete(contract.remote_task_id, {"risco": "catastrófico", "score": "x"})
    await h.manager.check(await h.store.get_contract(contract.id))
    breached = await h.store.get_contract(contract.id)
    assert breached.state == states.BREACHED and "saída fora do contrato" in breached.error
    run = await h.manager.wait_run(run_id, 1)
    assert run.status == "completed" and "violado" in run.answer

    run_id, contract = await h.open()
    h.agent.complete(contract.remote_task_id, None, "só texto")
    await h.manager.check(await h.store.get_contract(contract.id))
    assert "parte 'data'" in (await h.store.get_contract(contract.id)).error


async def test_direct_message_reply_and_rpc_errors():
    h = Harness()
    h.agent.on_send = lambda params: {
        "message": {
            "messageId": "m",
            "role": "ROLE_AGENT",
            "parts": [{"data": {"risco": "baixo", "score": 900}}],
        }
    }
    _, contract = await h.open()
    assert contract.state == states.COMPLETED and contract.output == {
        "risco": "baixo",
        "score": 900,
    }

    h.agent.on_send = None
    h.agent.errors["SendMessage"] = {
        "code": -32602,
        "message": "parâmetros inválidos",
        "reason": "INVALID_PARAMS",
    }
    _, contract = await h.open()
    assert contract.state == states.REJECTED and "INVALID_PARAMS" in contract.error

    h.agent.errors.clear()
    info = await h.directory.describe(h.spec)  # card descoberto antes da queda
    h.agent.up = False
    run_id = new_trace_id()
    await h.manager.begin_run(RunRecord(id=run_id, profile="p", question="q"))
    contract = await h.manager.open(
        OpenRequest(
            run_id,
            "p",
            h.spec,
            info,
            info.skill("avaliar_risco"),
            {"cliente": "Ana", "valor": 1},
            "",
            60,
        )
    )
    assert contract.state == states.FAILED and "ConnectError" in contract.error
    run = await h.manager.wait_run(run_id, 1)
    assert run.status == "completed" and "falhou" in run.answer


async def test_push_token_is_required_and_late_events_are_ignored():
    h = Harness(public_url="http://router.test")
    h.agent.push = True
    _, contract = await h.open()
    config = h.agent.sent[-1]["configuration"]["taskPushNotificationConfig"]
    assert config["url"] == f"http://router.test/a2a/push/{contract.id}"
    token = config["token"]
    task_id = contract.remote_task_id
    with pytest.raises(PushRejected):
        await h.manager.apply_push(
            contract.id,
            "token-errado",
            {"statusUpdate": {"taskId": task_id, "status": {"state": "TASK_STATE_COMPLETED"}}},
        )
    with pytest.raises(LookupError):
        await h.manager.apply_push("ctr_nao_existe", token, {})
    # evento de outra tarefa: registrado como violação, sem mudar o estado
    await h.manager.apply_push(
        contract.id,
        token,
        {"statusUpdate": {"taskId": "outra", "status": {"state": "TASK_STATE_FAILED"}}},
    )
    assert (await h.store.get_contract(contract.id)).state == states.ACTIVE
    await h.manager.apply_push(
        contract.id,
        token,
        {
            "artifactUpdate": {
                "taskId": task_id,
                "artifact": {
                    "artifactId": "r",
                    "parts": [{"data": {"risco": "medio", "score": 500}}],
                },
            }
        },
    )
    await h.manager.apply_push(
        contract.id,
        token,
        {"statusUpdate": {"taskId": task_id, "status": {"state": "TASK_STATE_COMPLETED"}}},
    )
    done = await h.store.get_contract(contract.id)
    assert done.state == states.COMPLETED and done.output["risco"] == "medio"
    await h.manager.apply_push(
        contract.id,
        token,
        {"statusUpdate": {"taskId": task_id, "status": {"state": "TASK_STATE_FAILED"}}},
    )
    assert (await h.store.get_contract(contract.id)).state == states.COMPLETED
    kinds = [e.kind for e in await h.store.contract_events(contract.id)]
    assert "violation" in kinds and kinds[-1] == "note"


async def test_consolidation_happens_once_and_callback_is_called():
    posted = []

    def callback(request: httpx.Request) -> httpx.Response:
        posted.append(request)
        return httpx.Response(204)

    calls = []

    async def consolidator(run, contracts):
        calls.append(run.id)
        await asyncio.sleep(0.01)
        return Consolidation("consolidado: " + ", ".join(c.state for c in contracts))

    h = Harness(callback_http=httpx.AsyncClient(transport=httpx.MockTransport(callback)))
    h.manager.consolidator = consolidator
    info = await h.directory.describe(h.spec)
    run_id = new_trace_id()
    await h.manager.begin_run(
        RunRecord(id=run_id, profile="p", question="q", callback_url="http://cliente.test/hook")
    )
    contracts = []
    for valor in (1000, 2000):
        contracts.append(
            await h.manager.open(
                OpenRequest(
                    run_id,
                    "p",
                    h.spec,
                    info,
                    info.skill("avaliar_risco"),
                    {"cliente": "Ana", "valor": valor},
                    "",
                    60,
                )
            )
        )
    for c in contracts:
        h.agent.complete(c.remote_task_id, {"risco": "baixo", "score": 700})
    current = [await h.store.get_contract(c.id) for c in contracts]
    await asyncio.gather(*(h.manager.check(c) for c in current))
    run = await h.manager.wait_run(run_id, 1)
    await h.manager.drain()
    assert run.answer == "consolidado: concluido, concluido" and calls == [run_id]
    assert len(posted) == 1
    import json

    body = json.loads(posted[0].content)
    assert body["status"] == "completed" and len(body["contracts"]) == 2


async def test_cancel_run_and_supervisor_expiry():
    h = Harness(tick_s=0.02)
    run_id, contract = await h.open()
    canceled = await h.manager.cancel_run(run_id)
    assert [c.state for c in canceled] == [states.CANCELED]
    assert h.agent.canceled == [contract.remote_task_id]
    run = await h.manager.wait_run(run_id, 1)
    assert run.status == "completed"

    run_id, contract = await h.open(deadline_s=0.05)
    await asyncio.sleep(0.08)
    assert await h.manager.tick() == 1
    await h.manager.drain()
    expired = await h.store.get_contract(contract.id)
    assert expired.state == states.EXPIRED and h.agent.canceled[-1] == contract.remote_task_id


# --------------------------------------------------------------------------
# motor delegando a agentes A2A

LLM = ModelSpec(name="llm", model="fake")


def _profile(agent: FakeA2AAgent, **kw) -> ResolvedProfile:
    spec = ProfileSpec(name="teste", model="llm", agents=[agent.name], **kw)
    return ResolvedProfile(spec=spec, model=LLM, agents=[AgentSpec(name=agent.name, url=agent.url)])


def _engine(agent: FakeA2AAgent, chat, **kw):
    network = FakeNetwork(agent)
    directory = network.directory()
    manager = ContractManager(
        MemoryContractStore(), client=directory.client, poll_min_s=0.01, tick_s=0.02
    )
    engine = RouterEngine(
        _profile(agent, **kw),
        chat=chat,
        connectors=_NoConnectors(),
        agents=directory,
        contracts=manager,
    )
    manager.consolidator = engine.consolidate
    return engine, manager


class _NoConnectors:
    async def discover(self, specs, refresh=False):
        return []


async def test_engine_delegates_waits_and_consolidates_with_llm():
    agent = FakeA2AAgent()
    agent.on_send = lambda params: {
        "status": {"state": "TASK_STATE_COMPLETED"},
        "artifacts": [{"artifactId": "a", "parts": [{"data": {"risco": "baixo", "score": 800}}]}],
    }
    chat = ScriptedChat(
        [
            '{"action": "delegate", "tasks": [{"agent": "risco", "skill": "avaliar_risco", "arguments": {"cliente": "Ana Lima", "valor": "80 mil"}}], "reason": "fraude"}',
            "O risco da operação é baixo (score 800).",
        ]
    )
    engine, _ = _engine(agent, chat)
    result = await engine.handle("verifique o risco de fraude de 80 mil da Ana Lima")
    assert (result.route, result.status, result.decided_by) == ("delegated", "completed", "llm")
    assert result.answer == "O risco da operação é baixo (score 800)."
    assert result.contracts[0]["state"] == states.COMPLETED
    assert agent.sent[0]["message"]["parts"][0]["data"] == {"cliente": "Ana Lima", "valor": 80000.0}
    consolidation_prompt = chat.calls[1][1].content
    assert '"score": 800' in consolidation_prompt and "Ana Lima" in consolidation_prompt
    names = [s["name"] for s in result.spans]
    assert "delegacao" in names and "espera" in names


async def test_engine_pending_then_needs_input_then_resume():
    agent = FakeA2AAgent()
    chat = ScriptedChat(
        [
            '{"action": "delegate", "agent": "risco", "skill": "avaliar_risco", "arguments": {"cliente": "Ana Lima", "valor": 5000}}'
        ]
    )
    engine, manager = _engine(agent, chat, wait_s=0.05)
    first = await engine.handle("avaliar risco de fraude de 5 mil da Ana Lima")
    assert first.status == "pending" and "Encaminhei seu pedido para risco" in first.answer
    task_id = agent.last_task_id
    agent.ask(task_id, "A operação foi feita pelo telefone?")
    await manager.check((await manager.store.run_contracts(first.run_id))[0])
    run = await manager.wait_run(first.run_id, 1)
    assert run.status == "needs_input" and "telefone" in run.answer

    agent.on_send = lambda params: {"status": {"state": "TASK_STATE_WORKING"}}
    resumed = await engine.resume(first.run_id, "sim, pelo telefone", wait_s=0.05)
    assert resumed.status == "pending" and resumed.run_id == first.run_id
    # a pergunta já respondida não volta como resposta provisória
    assert (
        "telefone?" not in resumed.answer
        and "Recebi sua resposta e repassei para risco" in resumed.answer
    )
    follow_up = agent.sent[-1]["message"]
    assert follow_up["taskId"] == task_id and follow_up["parts"] == [{"text": "sim, pelo telefone"}]
    assert follow_up["metadata"][CONTRACT_EXTENSION_URI]["contract_id"]
    agent.complete(task_id, {"risco": "alto", "score": 200}, "Risco alto: bloquear.")
    await manager.check((await manager.store.run_contracts(first.run_id))[0])
    run = await manager.wait_run(first.run_id, 1)
    # sem LLM de consolidação (ScriptedChat sem respostas): cai para o modelo de texto
    assert run.status == "completed" and "Risco alto: bloquear." in run.answer


async def test_engine_offline_delegation_and_unknown_resume():
    agent = FakeA2AAgent()
    agent.on_send = lambda params: {
        "status": {"state": "TASK_STATE_COMPLETED"},
        "artifacts": [
            {
                "artifactId": "a",
                "parts": [{"data": {"risco": "medio", "score": 510}}, {"text": "Risco médio."}],
            }
        ],
    }
    engine, _ = _engine(agent, OfflineChat())
    result = await engine.handle(
        "avaliar risco de fraude da operação de 12 mil do cliente Bruno Dias"
    )
    assert (result.route, result.status, result.decided_by) == (
        "delegated",
        "completed",
        "heuristic",
    )
    assert "Risco médio." in result.answer
    missing = await engine.resume("0" * 32, "oi")
    assert missing.route == "error" and "não encontrada" in missing.error
