"""Contrato de ponta a ponta: gerenciador do Switchboard ↔ agente feito com o SDK oficial.

Tudo em processo: o agente (Starlette do ``a2a-sdk``) e um receptor de push
ficam atrás de um transporte httpx que roteia por host, então as push
notifications do SDK chegam de verdade ao gerenciador de contratos.
"""

from __future__ import annotations

import asyncio
import dataclasses

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from switchboard.a2a import NOTIFICATION_TOKEN_HEADER, A2AClient, AgentDirectory
from switchboard.config import AgentSpec
from switchboard.contracts import (
    ContractManager,
    MemoryContractStore,
    OpenRequest,
    RunRecord,
    SkillTerms,
    states,
)
from switchboard.testing import HostRoutingTransport
from switchboard.tracing import new_trace_id
from switchboard_agentkit import ContractAgent, SkillContext, SkillResult
from switchboard_agents import analise, risco

PROPOSTA = {"cliente": "Maria Souza", "valor": 80000, "prazo_meses": 24, "renda_mensal": 12000}


class Harness:
    def __init__(
        self, *, delay_s: float = 0.05, push: bool = True, agent: ContractAgent | None = None
    ):
        self.transport = HostRoutingTransport()
        self.http = httpx.AsyncClient(transport=self.transport)
        self.agent = agent or analise.build("http://analise.test", delay_s=delay_s)
        self.agent.http = self.http  # push notifications do SDK passam pelo mesmo transporte
        self.transport.mount("analise.test", self.agent.build_app())
        self.directory = AgentDirectory(client=A2AClient(http=self.http))
        self.spec = AgentSpec(name="analise-credito", url="http://analise.test", push=push)
        self.store = MemoryContractStore()
        self.manager = ContractManager(
            self.store,
            client=self.directory.client,
            public_url="http://router.test",
            agent_resolver=self._resolve,
            tick_s=0.05,
            poll_min_s=0.05,
            poll_max_s=0.2,
            push_check_s=0.3,
        )
        self.pushes: list[dict] = []

        async def receive(request: Request):
            payload = await request.json()
            self.pushes.append(payload)
            try:
                await self.manager.apply_push(
                    request.path_params["cid"],
                    request.headers.get(NOTIFICATION_TOKEN_HEADER),
                    payload,
                )
            except Exception as exc:  # o teste olha os estados; aqui só registra
                return JSONResponse({"erro": str(exc)}, status_code=403)
            return JSONResponse({"ok": True})

        self.transport.mount(
            "router.test", Starlette(routes=[Route("/a2a/push/{cid}", receive, methods=["POST"])])
        )

    async def _resolve(self, name: str):
        return self.spec if name == self.spec.name else None

    async def open(
        self,
        arguments: dict,
        *,
        deadline_s: float = 60.0,
        skill_id: str = "analisar_proposta",
        info=None,
    ):
        info = info or await self.directory.describe(self.spec, refresh=True)
        assert info.status == "online", info.error
        run_id = new_trace_id()
        await self.manager.begin_run(RunRecord(id=run_id, profile="p", question="q"))
        contract = await self.manager.open(
            OpenRequest(
                run_id=run_id,
                profile="p",
                spec=self.spec,
                agent=info,
                skill=info.skill(skill_id),
                arguments=arguments,
                instruction="analise a proposta",
                deadline_s=deadline_s,
            )
        )
        return run_id, contract

    async def aclose(self):
        await self.manager.stop()
        # tarefas do SDK (produtor/consumidor de cada tarefa A2A) que ainda estejam vivas
        me = asyncio.current_task()
        lingering = [t for t in asyncio.all_tasks() if t is not me and not t.done()]
        for task in lingering:
            task.cancel()
        if lingering:
            await asyncio.wait(lingering, timeout=3)
        await self.http.aclose()


@pytest.fixture
async def harness():
    h = Harness()
    yield h
    await h.aclose()


async def test_card_declares_contract_and_directory_reads_it(harness):
    info = await harness.directory.describe(harness.spec)
    assert info.status == "online" and info.protocol_version == "1.0" and info.push
    skill = info.skill("analisar_proposta")
    assert skill.contract == "completo" and skill.terms.max_duration_s == 900
    assert skill.terms.input_schema["required"] == [
        "cliente",
        "valor",
        "prazo_meses",
        "renda_mensal",
    ]


async def test_push_flow_completes_and_validates_output(harness):
    run_id, contract = await harness.open(PROPOSTA)
    assert contract.state in (states.ACTIVE, states.COMPLETED)
    assert contract.reply_mode == "push" and contract.remote_task_id
    run = await harness.manager.wait_run(run_id, 5)
    assert run.status == "completed"
    [done] = await harness.store.run_contracts(run_id)
    assert done.state == states.COMPLETED
    assert done.output["decisao"] == "aprovado" and done.output["prazo_meses"] == 24
    assert isinstance(done.output["prazo_meses"], int)  # 24.0 do protobuf voltou a ser inteiro
    assert "Proposta aprovada" in done.output_text
    kinds = [(e.kind, e.state, e.source) for e in await harness.store.contract_events(contract.id)]
    assert ("state", states.PROPOSED, "router") in kinds
    assert ("state", states.COMPLETED, "push") in kinds
    assert any(k == "artifact" for k, _, _ in kinds)
    assert harness.pushes and all(
        "statusUpdate" in p or "artifactUpdate" in p or "task" in p for p in harness.pushes
    )
    # consolidação padrão (sem LLM) cita o agente e o resultado
    assert "analise-credito" in run.answer and "Proposta aprovada" in run.answer


async def test_input_required_round_trip(harness):
    run_id, contract = await harness.open({**PROPOSTA, "valor": 800000, "renda_mensal": 60000})
    run = await harness.manager.wait_run(run_id, 5)
    assert run.status == "needs_input"
    assert "garantia" in run.answer.lower()
    [waiting] = await harness.store.run_contracts(run_id)
    assert waiting.state == states.INPUT_REQUIRED

    await harness.manager.provide_input(run_id, "sim, imóvel quitado")
    run = await harness.manager.wait_run(run_id, 5)
    assert run.status == "completed", run
    [done] = await harness.store.run_contracts(run_id)
    assert done.state == states.COMPLETED
    assert done.output["garantia"] == "imóvel" and done.output["taxa_mensal_percentual"] == 1.19


async def test_poll_only_contract_is_driven_by_the_supervisor():
    h = Harness(push=False)
    try:
        run_id, contract = await h.open(PROPOSTA)
        assert contract.reply_mode == "poll"
        h.manager.start()
        run = await h.manager.wait_run(run_id, 5)
        assert run.status == "completed"
        [done] = await h.store.run_contracts(run_id)
        assert done.state == states.COMPLETED and done.checks >= 1
        assert not h.pushes
    finally:
        await h.aclose()


async def test_schema_hash_mismatch_is_rejected_by_the_agent(harness):
    info = await harness.directory.describe(harness.spec)
    skill = info.skill("analisar_proposta")
    stale_input = {**skill.terms.input_schema, "description": "versão antiga"}
    stale = dataclasses.replace(
        skill, terms=SkillTerms(stale_input, skill.terms.output_schema, 900)
    )
    info = dataclasses.replace(info, skills=[stale])
    run_id, contract = await harness.open(PROPOSTA, info=info)
    run = await harness.manager.wait_run(run_id, 5)
    [rejected] = await harness.store.run_contracts(run_id)
    assert rejected.state == states.REJECTED
    assert "contrato desatualizado" in (rejected.error or "")
    assert run.status == "completed"  # consolidou explicando a recusa
    assert "rejeitado" in run.answer


async def test_deadline_expires_and_cancels_the_task():
    h = Harness(delay_s=3.0)
    try:
        run_id, contract = await h.open(PROPOSTA, deadline_s=0.4)
        h.manager.start()
        run = await h.manager.wait_run(run_id, 5)
        assert run.status == "completed"
        [expired] = await h.store.run_contracts(run_id)
        assert expired.state == states.EXPIRED
        # o agente recebeu o CancelTask (o SDK cancela a execução de forma assíncrona)
        for _ in range(40):
            task = await h.directory.client.get_task(expired.rpc_url, expired.remote_task_id)
            if task["status"]["state"] != "TASK_STATE_WORKING":
                break
            await asyncio.sleep(0.05)
        assert task["status"]["state"] == "TASK_STATE_CANCELED"
        # evento tardio do agente depois do encerramento não muda nada
        assert (await h.store.get_contract(expired.id)).state == states.EXPIRED
    finally:
        await h.aclose()


async def test_agent_refuses_tasks_without_contract_terms(harness):
    info = await harness.directory.describe(harness.spec)
    message = {
        "messageId": "m1",
        "role": "ROLE_USER",
        "parts": [{"data": PROPOSTA, "mediaType": "application/json"}],
    }
    result = await harness.directory.client.send_message(info.rpc_url, message)
    task = result["task"]
    for _ in range(50):
        if task["status"]["state"] != "TASK_STATE_SUBMITTED":
            break
        await asyncio.sleep(0.02)
        task = await harness.directory.client.get_task(info.rpc_url, task["id"])
    assert task["status"]["state"] == "TASK_STATE_REJECTED"


async def test_agent_validates_its_own_output():
    agent = ContractAgent(name="quebrado", description="d", url="http://analise.test")

    @agent.skill(
        "s",
        name="s",
        description="d",
        input_schema={"type": "object"},
        output_schema={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
    )
    async def skill(ctx: SkillContext, dados: dict) -> SkillResult:
        return SkillResult({"n": "não é número"})

    h = Harness(agent=agent)
    h.spec = AgentSpec(name="quebrado", url="http://analise.test")
    try:
        run_id, _ = await h.open({}, skill_id="s")
        await h.manager.wait_run(run_id, 5)
        [c] = await h.store.run_contracts(run_id)
        assert c.state == states.FAILED and "saída fora do contrato" in (c.error or "")
    finally:
        await h.aclose()


async def test_risk_agent_fan_out_contracts_finish_independently():
    h = Harness()
    risk = risco.build("http://risco.test", delay_s=0.05)
    risk.http = h.http
    h.transport.mount("risco.test", risk.build_app())
    risk_spec = AgentSpec(name="risco", url="http://risco.test")
    try:
        info_a = await h.directory.describe(h.spec)
        info_r = await h.directory.describe(risk_spec)
        run_id = new_trace_id()
        await h.manager.begin_run(RunRecord(id=run_id, profile="p", question="q"))
        specs = {"analise-credito": h.spec, "risco": risk_spec}
        h.manager._agent_resolver = lambda name: asyncio.sleep(0, result=specs.get(name))
        await asyncio.gather(
            h.manager.open(
                OpenRequest(
                    run_id, "p", h.spec, info_a, info_a.skill("analisar_proposta"), PROPOSTA, "", 60
                )
            ),
            h.manager.open(
                OpenRequest(
                    run_id,
                    "p",
                    risk_spec,
                    info_r,
                    info_r.skill("avaliar_risco"),
                    {"cliente": "Maria Souza", "valor": 80000},
                    "",
                    60,
                )
            ),
        )
        run = await h.manager.wait_run(run_id, 5)
        assert run.status == "completed"
        contracts = await h.store.run_contracts(run_id)
        assert sorted(c.agent for c in contracts) == ["analise-credito", "risco"]
        assert all(c.state == states.COMPLETED for c in contracts)
    finally:
        await h.aclose()
