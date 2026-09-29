"""Motor de roteamento: recebe o pedido e decide o caminho.

Fluxo de :meth:`RouterEngine.handle`:

1. **RAG** e **descoberta** em paralelo — trechos relevantes das bases do
   perfil; tools dos conectores MCP e skills dos agentes A2A que estão no ar;
2. **decisão** — o Jev (System One), se o perfil tiver modelo de decisão, ou o
   LLM, ou o decisor heurístico (offline): ``answer``, ``tool``, ``delegate``
   ou ``clarify``;
3. **execução**:

   * ``answer``/``clarify``: devolve o texto;
   * ``tool``: chama a tool MCP e sintetiza a resposta (síncrono);
   * ``delegate``: abre um contrato por tarefa com os agentes A2A, espera até
     ``wait_s`` e devolve a resposta consolidada — ou ``pending`` com o
     ``run_id``; a consolidação continua em segundo plano quando o último
     contrato termina.

Cada etapa vira um span (:mod:`switchboard.tracing`). Falhas de provedor não
derrubam o atendimento: Jev fora → decide o LLM; LLM fora → decide o
heurístico; síntese falhou → devolve o resultado da tool como veio.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

from ..a2a.directory import AgentDirectory, AgentInfo
from ..connectors.catalog import ConnectorCatalog, ConnectorInfo, ToolCallResult
from ..contracts import (
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_NEEDS_INPUT,
    RUN_PENDING,
    Consolidation,
    ContractManager,
    ContractRecord,
    OpenRequest,
    RunRecord,
    default_consolidation,
    states,
)
from ..errors import ConnectorError, ContractError, DecisionModelError, LLMError
from ..jev import JevClient
from ..llm.base import ChatModel, Message, Usage
from ..rag.retriever import Hit, Retriever
from ..text import content_tokens, truncate
from ..tracing import Span, SpanRecorder, new_span_id, new_trace_id, utcnow
from .capabilities import Capability, capabilities
from .deciders import (
    Composer,
    Decider,
    ExtractiveAnswerer,
    HeuristicArgumentFiller,
    HeuristicDecider,
    LLMAnswerer,
    LLMArgumentFiller,
    LLMComposer,
    LLMDecider,
    TemplateComposer,
    user_turns,
)
from .decision import DecisionError
from .jev_decider import DecisionContext, JevDecider
from .prompts import consolidation_messages
from .types import Decision, ResolvedProfile, RouterResult, SourceRef

CONTEXT_LIMIT = 12


def normalize_messages(messages: str | Sequence[Message] | Sequence[dict]) -> list[Message]:
    if isinstance(messages, str):
        return [Message("user", messages)]
    out: list[Message] = []
    for m in messages:
        if isinstance(m, Message):
            out.append(m)
        else:
            content = m.get("content", "")
            if isinstance(content, list):  # partes estilo OpenAI
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            out.append(Message(str(m.get("role", "user")), str(content or "")))
    return out


def search_query(messages: Sequence[Message]) -> str:
    """Pergunta usada no RAG; pedidos muito curtos herdam o turno anterior."""
    turns = user_turns(messages)
    if not turns:
        return ""
    if len(turns) >= 2 and len(content_tokens(turns[-1])) < 4:
        return f"{turns[-2]}\n{turns[-1]}"
    return turns[-1]


class _Clock:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000


SORRY = "Desculpe, não consegui processar seu pedido agora. Tente novamente em instantes."


class RouterEngine:
    def __init__(
        self,
        profile: ResolvedProfile,
        *,
        chat: ChatModel,
        connectors: ConnectorCatalog,
        agents: AgentDirectory | None = None,
        contracts: ContractManager | None = None,
        retriever: Retriever | None = None,
        jev: JevClient | None = None,
        decider: Decider | None = None,
        composer: Composer | None = None,
        warnings: Sequence[str] = (),
    ):
        self.profile = profile
        self.static_warnings = list(warnings)
        self.chat = chat
        self.connectors = connectors
        self.agents = agents
        self.contracts = contracts
        self.retriever = retriever
        self.jev = jev
        fallback: Decider = HeuristicDecider() if chat.offline else LLMDecider(chat)
        if decider is not None:
            self.decider = decider
        elif jev is not None:
            self.decider = JevDecider(
                jev,
                answerer=ExtractiveAnswerer() if chat.offline else LLMAnswerer(chat),
                filler=HeuristicArgumentFiller() if chat.offline else LLMArgumentFiller(chat),
                escalate=fallback,
            )
        else:
            self.decider = fallback
        self.composer = composer or (TemplateComposer() if chat.offline else LLMComposer(chat))
        self._fallback_decider = HeuristicDecider()
        self._fallback_composer = TemplateComposer()

    @property
    def decision_label(self) -> str:
        if isinstance(self.decider, JevDecider):
            return f"{self.decider.jev.label} → {self.chat.label}"
        return self.chat.label

    # ------------------------------------------------------------------ entrada

    async def handle(
        self,
        messages: str | Sequence[Message] | Sequence[dict],
        *,
        wait_s: float | None = None,
        callback_url: str | None = None,
    ) -> RouterResult:
        clock = _Clock()
        spec = self.profile.spec
        msgs = normalize_messages(messages)
        turns = user_turns(msgs)
        question = turns[-1] if turns else ""
        trace_id = new_trace_id()
        recorder = SpanRecorder(trace_id, roteador=spec.name)
        result = RouterResult(
            trace_id=trace_id,
            profile=spec.name,
            question=question,
            answer="",
            route="error",
            model=self.chat.label,
        )
        result.warnings.extend(self.static_warnings)
        if not question:
            result.answer = "Envie uma mensagem de usuário para eu poder ajudar."
            result.error = "nenhuma mensagem de usuário no pedido"
        else:
            try:
                await self._run(msgs, question, result, recorder, wait_s, callback_url)
            except Exception as exc:  # nunca derruba o pedido: vira uma resposta de erro com trace
                result.route = "error"
                result.status = "failed"
                result.error = f"{exc.__class__.__name__}: {exc}"[:500]
                result.answer = SORRY
        recorder.close(
            "error" if result.route == "error" else "ok", rota=result.route, execucao=result.status
        )
        result.spans = recorder.to_list()
        result.latency_ms = clock.ms()
        return result

    async def _run(
        self,
        msgs: list[Message],
        question: str,
        result: RouterResult,
        recorder: SpanRecorder,
        wait_s: float | None,
        callback_url: str | None,
    ) -> None:
        hits, (connectors, agents) = await asyncio.gather(
            self._rag(msgs, result, recorder), self._discover(result, recorder)
        )
        caps = capabilities(connectors, agents)
        decision = await self._decide(msgs, caps, hits, result, recorder)
        if decision is None:
            return
        result.reason = decision.reason
        result.decided_by = decision.decided_by
        result.confidence = decision.confidence
        result.tasks = [t.to_dict() for t in decision.tasks]
        if decision.action == "answer":
            result.route = "direct"
            result.answer = decision.answer or ""
            result.sources = [
                self._source(i, hits[i - 1]) for i in decision.sources if 1 <= i <= len(hits)
            ]
        elif decision.action == "clarify":
            result.route = "clarify"
            result.answer = decision.question or ""
            if decision.task:
                result.agent, result.tool = decision.task.owner, decision.task.name
                result.arguments = decision.task.arguments or None
        elif decision.action == "tool":
            await self._call_tool(decision, question, result, recorder)
        else:
            await self._delegate(
                decision, msgs, question, agents, result, recorder, wait_s, callback_url
            )

    # -------------------------------------------------------------------- etapas

    async def _rag(
        self, msgs: list[Message], result: RouterResult, recorder: SpanRecorder
    ) -> list[Hit]:
        spec = self.profile.spec
        if self.retriever is None or not spec.knowledge_bases:
            return []
        with recorder.span("rag", "rag", bases=spec.knowledge_bases) as span:
            try:
                hits = await self.retriever.search(
                    search_query(msgs),
                    spec.knowledge_bases,
                    top_k=spec.top_k,
                    min_score=spec.min_score,
                )
            except Exception as exc:  # RAG fora do ar não derruba o atendimento
                result.warnings.append(f"busca na base de conhecimento falhou: {exc}")
                span.finish("warn", erro=str(exc)[:300])
                return []
            span.attributes.update(trechos=len(hits), scores=[round(h.score, 3) for h in hits])
            return hits

    async def _discover(
        self, result: RouterResult, recorder: SpanRecorder
    ) -> tuple[list[ConnectorInfo], list[AgentInfo]]:
        profile = self.profile
        if not profile.connectors and not (profile.agents and self.agents is not None):
            return [], []
        with recorder.span("descoberta", "descoberta") as span:
            connectors, agents = await asyncio.gather(
                self.connectors.discover(profile.connectors),
                self.agents.discover(profile.agents)
                if self.agents is not None
                else asyncio.sleep(0, result=[]),
            )
            span.attributes.update(
                conectores={
                    c.name: [t.name for t in c.tools] for c in connectors if c.status == "online"
                },
                agentes={
                    a.name: [f"{s.id} ({s.contract})" for s in a.skills]
                    for a in agents
                    if a.status == "online"
                },
                offline={
                    **{c.name: c.error for c in connectors if c.status != "online"},
                    **{a.name: a.error for a in agents if a.status != "online"},
                },
            )
        for c in connectors:
            if c.status != "online":
                result.warnings.append(f"conector {c.name} indisponível: {c.error}")
        for a in agents:
            if a.status != "online":
                result.warnings.append(f"agente {a.name} indisponível: {a.error}")
            for skill in a.skills:
                if skill.problems:
                    result.warnings.append(
                        f"skill {a.name}/{skill.id} ignorada: contrato inválido ({'; '.join(skill.problems)})"
                    )
        return connectors, agents

    async def _decide(
        self,
        msgs: list[Message],
        caps: list[Capability],
        hits: list[Hit],
        result: RouterResult,
        recorder: SpanRecorder,
    ) -> Decision | None:
        spec = self.profile.spec
        with recorder.span("decisao", "decisao", capacidades=len(caps)) as span:
            ctx = DecisionContext(recorder, parent=span)
            try:
                if isinstance(self.decider, JevDecider):
                    decision, calls = await self.decider.decide(
                        profile=spec, messages=msgs, caps=caps, hits=hits, ctx=ctx
                    )
                else:
                    decision, calls = await self.decider.decide(
                        profile=spec, messages=msgs, caps=caps, hits=hits
                    )
            except (LLMError, DecisionError, DecisionModelError) as exc:
                if isinstance(exc, (LLMError, DecisionModelError)):
                    result.warnings.append(f"modelo indisponível, usei o decisor offline: {exc}")
                else:
                    result.warnings.append(
                        f"resposta do modelo fora do formato mesmo após reparo, usei o decisor offline: {exc}"
                    )
                with recorder.span("decisao", "fallback_offline", parent=span, erro=str(exc)[:300]):
                    decision, calls = await self._fallback_decider.decide(
                        profile=spec, messages=msgs, caps=caps, hits=hits
                    )
            except Exception as exc:
                result.error = f"falha ao decidir a rota: {exc}"
                result.answer = SORRY
                result.status = "failed"
                span.finish("error", erro=str(exc)[:300])
                return None
            result.warnings.extend(ctx.warnings)
            usage = Usage()
            for call in calls:
                usage.add(call.usage)
            result.usage.add(usage)
            span.attributes.update(
                acao=decision.action,
                decidido_por=decision.decided_by,
                confianca=round(decision.confidence, 4)
                if decision.confidence is not None
                else None,
                motivo=decision.reason,
                tarefas=[t.to_dict() for t in decision.tasks],
                chamadas_llm=len(calls),
            )
            if calls:
                span.attributes["tokens"] = usage.to_dict()
        return decision

    async def _call_tool(
        self, decision: Decision, question: str, result: RouterResult, recorder: SpanRecorder
    ) -> None:
        spec = self.profile.spec
        task = decision.task
        assert task is not None
        result.route = "tool"
        result.agent, result.tool, result.arguments = task.owner, task.name, task.arguments
        connector = self.profile.connector(task.owner)
        with recorder.span(
            "tool_mcp",
            f"tool: {task.key}",
            conector=task.owner,
            tool=task.name,
            argumentos=task.arguments,
        ) as span:
            if connector is None:  # só acontece com decisores customizados
                outcome = ToolCallResult(
                    f"conector {task.owner} não pertence a este perfil", True, None, 0.0
                )
            else:
                try:
                    outcome = await self.connectors.call(connector, task.name, task.arguments)
                except ConnectorError as exc:
                    outcome = ToolCallResult(str(exc), True, None, 0.0)
            span.attributes.update(erro=outcome.is_error, resultado=truncate(outcome.text, 800))
            if outcome.is_error:
                span.status = "warn"
        if outcome.is_error:
            result.warnings.append(f"a tool {task.key} retornou erro")
        if not spec.synthesize:
            result.answer = (
                outcome.text
                if not outcome.is_error
                else f"A ferramenta retornou erro: {outcome.text}"
            )
            return
        with recorder.span("sintese", "sintese") as span:
            try:
                answer, call = await self.composer.compose(
                    profile=spec,
                    question=question,
                    capability=task.key,
                    arguments=task.arguments,
                    result_text=outcome.text,
                    is_error=outcome.is_error,
                )
            except Exception as exc:  # síntese é enfeite: sem ela, devolve o resultado
                result.warnings.append(
                    f"síntese pelo modelo falhou, devolvi o resultado da tool: {exc}"
                )
                answer, call = await self._fallback_composer.compose(
                    profile=spec,
                    question=question,
                    capability=task.key,
                    arguments=task.arguments,
                    result_text=outcome.text,
                    is_error=outcome.is_error,
                )
            span.attributes["llm"] = call is not None
        if call is not None:
            result.usage.add(call.usage)
        result.answer = answer

    # ---------------------------------------------------------------- delegação

    async def _delegate(
        self,
        decision: Decision,
        msgs: list[Message],
        question: str,
        agents: list[AgentInfo],
        result: RouterResult,
        recorder: SpanRecorder,
        wait_s: float | None,
        callback_url: str | None,
    ) -> None:
        spec = self.profile.spec
        result.route = "delegated"
        first = decision.task
        assert first is not None
        result.agent, result.tool, result.arguments = first.owner, first.name, first.arguments
        if self.contracts is None:
            result.status = "failed"
            result.error = "delegação a agentes indisponível (sem gerenciador de contratos)"
            result.answer = SORRY
            return
        by_name = {a.name: a for a in agents}
        names = ", ".join(
            f"{t.owner} ({(by_name[t.owner].skill(t.name).name if by_name.get(t.owner) and by_name[t.owner].skill(t.name) else t.name)})"
            for t in decision.tasks
        )
        interim = (
            f"Encaminhei seu pedido para {names}. Assim que houver resposta, eu consolido o "
            "resultado para você."
        )
        context = [
            {"role": m.role, "content": truncate(m.content, 2000)}
            for m in msgs
            if m.role in ("user", "assistant")
        ][-CONTEXT_LIMIT:]
        run = RunRecord(
            id=result.trace_id,
            profile=spec.name,
            question=question,
            status=RUN_PENDING,
            answer=interim,
            context=context,
            callback_url=callback_url,
            extra={
                "route": "delegated",
                "model": result.model,
                "reason": result.reason,
                "decided_by": result.decided_by,
                "confidence": result.confidence,
                "agent": result.agent,
                "tool": result.tool,
                "arguments": result.arguments,
                "tasks": result.tasks,
            },
        )
        await self.contracts.begin_run(run)
        with recorder.span("delegacao", "delegacao", tarefas=len(decision.tasks)) as span:
            problems: list[str] = []
            requests: list[tuple[str, OpenRequest]] = []
            for task in decision.tasks:
                agent_spec = self.profile.agent(task.owner)
                info = by_name.get(task.owner)
                skill = info.skill(task.name) if info else None
                if agent_spec is None or info is None or skill is None:
                    problems.append(f"{task.key}: agente ou skill fora do perfil")
                    continue
                requests.append(
                    (
                        task.key,
                        OpenRequest(
                            run_id=result.trace_id,
                            profile=spec.name,
                            spec=agent_spec,
                            agent=info,
                            skill=skill,
                            arguments=task.arguments,
                            instruction=task.instruction or question,
                            deadline_s=spec.deadline_s,
                            parent_span_id=span.id,
                        ),
                    )
                )
            # todos os contratos são gravados antes do primeiro envio (fan-out sem corrida)
            opened = await self.contracts.open_many([req for _, req in requests])
            contracts: list[ContractRecord] = []
            for (key, _), item in zip(requests, opened, strict=True):
                if isinstance(item, ContractError):
                    problems.append(f"{key}: {item}")
                else:
                    contracts.append(item)
            span.attributes["contratos"] = [c.summary() for c in contracts]
            if problems:
                span.status = "warn"
                span.attributes["problemas"] = problems
                result.warnings.extend(problems)
        if not contracts:
            message = "Não consegui encaminhar o pedido aos agentes: " + "; ".join(problems)
            await self.contracts.store.update_run(
                result.trace_id,
                status=RUN_FAILED,
                answer=message,
                error=message,
                finished_at=utcnow(),
            )
            result.status = "failed"
            result.answer = message
            result.error = message
            return
        await self._await_run(result, recorder, wait_s, interim)

    async def _await_run(
        self, result: RouterResult, recorder: SpanRecorder, wait_s: float | None, interim: str
    ) -> None:
        assert self.contracts is not None
        budget = self.profile.spec.wait_s if wait_s is None else min(max(wait_s, 0.0), 120.0)
        with recorder.span("espera", "espera", limite_s=budget) as span:
            run = await self.contracts.wait_run(result.trace_id, budget)
            span.attributes["status"] = run.status if run else None
        contracts = await self.contracts.store.run_contracts(result.trace_id)
        result.contracts = [c.summary() for c in contracts]
        if run is not None and run.status in (RUN_COMPLETED, RUN_FAILED):
            result.status = "completed" if run.status == RUN_COMPLETED else "failed"
            result.answer = run.answer
            if run.error:
                result.warnings.append(run.error)
        elif run is not None and run.status == RUN_NEEDS_INPUT:
            result.status = "needs_input"
            result.answer = run.answer
        else:
            result.status = "pending"
            result.answer = run.answer if run and run.answer else interim

    async def resume(
        self,
        run_id: str,
        messages: str | Sequence[Message] | Sequence[dict],
        *,
        wait_s: float | None = None,
    ) -> RouterResult:
        """Leva a resposta do usuário ao(s) agente(s) de uma execução em ``needs_input``."""
        clock = _Clock()
        msgs = normalize_messages(messages)
        turns = user_turns(msgs)
        text = turns[-1] if turns else ""
        recorder = SpanRecorder(run_id, root_name="entrada_do_usuario", roteador=self.profile.name)
        result = RouterResult(
            trace_id=run_id,
            profile=self.profile.name,
            question=text,
            answer="",
            route="delegated",
            model=self.chat.label,
        )
        run = await self.contracts.store.get_run(run_id) if self.contracts else None
        if run is None or self.contracts is None:
            result.route, result.status = "error", "failed"
            result.error = f"execução {run_id} não encontrada"
            result.answer = "Não encontrei essa conversa com os agentes para continuar."
        elif not text:
            result.route, result.status = "error", "failed"
            result.error = "nenhuma mensagem de usuário no pedido"
            result.answer = "Envie a informação pedida pelo agente."
        elif run.status != RUN_NEEDS_INPUT:
            await self._await_run(result, recorder, 0.0, run.answer)
            result.warnings.append(f"a execução está '{run.status}', não aguardando entrada")
        else:
            with recorder.span("entrada", "entrada", texto=truncate(text, 500)) as span:
                updated = await self.contracts.provide_input(run_id, text)
                span.attributes["contratos"] = [c.summary() for c in updated]
                if not updated:
                    span.status = "warn"
                    result.warnings.append(
                        "nenhum contrato aguardava esta resposta (já entregue por outro pedido?)"
                    )
            await self._await_run(result, recorder, wait_s, run.answer)
        recorder.close("error" if result.route == "error" else "ok", execucao=result.status)
        result.spans = recorder.to_list()
        result.latency_ms = clock.ms()
        return result

    # -------------------------------------------------------------- consolidação

    async def consolidate(self, run: RunRecord, contracts: list[ContractRecord]) -> Consolidation:
        spec = self.profile.spec
        parent = next((c.parent_span_id for c in contracts if c.parent_span_id), None)
        span = Span(
            id=new_span_id(),
            trace_id=run.id,
            parent_id=parent,
            kind="consolidacao",
            name="consolidacao",
            started_at=utcnow(),
            attributes={"contratos": {c.id: c.state for c in contracts}},
        )
        results = [
            {
                "agente": c.agent,
                "skill": c.skill,
                "estado": states.LABELS.get(c.state, c.state),
                "resultado": c.output if c.output is not None else (c.output_text or None),
                "detalhes": c.output_text if c.output is not None and c.output_text else None,
                "erro": c.error,
            }
            for c in contracts
        ]
        answer: str | None = None
        if spec.synthesize and not self.chat.offline:
            try:
                reply = await self.chat.chat(
                    consolidation_messages(spec, run.question, run.context, results)
                )
                answer = reply.text.strip() or None
                span.attributes.update(llm=True, tokens=reply.usage.to_dict())
            except Exception as exc:
                span.attributes.update(llm=False, erro=str(exc)[:300])
                span.status = "warn"
        if answer is None:
            if (
                len(contracts) == 1
                and contracts[0].state == states.COMPLETED
                and not spec.synthesize
            ):
                c = contracts[0]
                answer = c.output_text or default_consolidation(run, contracts)
            else:
                answer = default_consolidation(run, contracts)
            span.attributes.setdefault("llm", False)
        span.finish("ok" if span.status == "open" else span.status)
        return Consolidation(answer=answer, spans=[span])

    @staticmethod
    def _source(index: int, hit: Hit) -> SourceRef:
        return SourceRef(
            index=index,
            kb=hit.kb,
            document=hit.document,
            chunk_id=hit.chunk_id,
            score=hit.score,
            snippet=truncate(hit.content, 280),
            section=hit.section,
        )


def contracts_summary(contracts: Sequence[ContractRecord]) -> list[dict[str, Any]]:
    return [c.summary() for c in contracts]
