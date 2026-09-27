"""Motor de roteamento: recebe o pedido e decide o caminho.

Fluxo de :meth:`RouterEngine.handle`:

1. **RAG** — busca trechos relevantes nas bases do perfil;
2. **descoberta** — pergunta via MCP quais agentes estão no ar e que tools têm;
3. **decisão** — o LLM (ou o decisor offline) escolhe ``answer``, ``delegate``
   ou ``clarify``;
4. **execução** — responde direto, pergunta, ou aciona a tool do agente via
   MCP e sintetiza a resposta final.

Cada etapa vira um :class:`TraceStep` com duração e detalhes, devolvido no
resultado (e gravado pelo serviço, se houver persistência). Falha no provedor
de LLM não derruba o atendimento: o motor cai para o decisor offline e
registra o aviso.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence

from ..agents.catalog import AgentCatalog, ToolCallResult
from ..errors import AgentError, LLMError
from ..llm.base import ChatModel, Message, Usage
from ..rag.retriever import Hit, Retriever
from ..text import content_tokens, truncate
from .deciders import (
    Composer,
    Decider,
    HeuristicDecider,
    LLMComposer,
    LLMDecider,
    TemplateComposer,
    user_turns,
)
from .types import Decision, ResolvedProfile, RouterResult, SourceRef, TraceStep


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


class RouterEngine:
    def __init__(
        self,
        profile: ResolvedProfile,
        *,
        chat: ChatModel,
        catalog: AgentCatalog,
        retriever: Retriever | None = None,
        decider: Decider | None = None,
        composer: Composer | None = None,
    ):
        self.profile = profile
        self.chat = chat
        self.catalog = catalog
        self.retriever = retriever
        self.decider = decider or (HeuristicDecider() if chat.offline else LLMDecider(chat))
        self.composer = composer or (TemplateComposer() if chat.offline else LLMComposer(chat))
        self._fallback_decider = HeuristicDecider()
        self._fallback_composer = TemplateComposer()

    async def handle(self, messages: str | Sequence[Message] | Sequence[dict]) -> RouterResult:
        clock = _Clock()
        spec = self.profile.spec
        msgs = normalize_messages(messages)
        turns = user_turns(msgs)
        question = turns[-1] if turns else ""
        result = RouterResult(
            trace_id=uuid.uuid4().hex,
            profile=spec.name,
            question=question,
            answer="",
            route="error",
            model=self.chat.label,
        )
        if not question:
            result.answer = "Envie uma mensagem de usuário para eu poder ajudar."
            result.error = "nenhuma mensagem de usuário no pedido"
            result.latency_ms = clock.ms()
            return result

        hits = await self._rag(msgs, result)
        agents = await self._discover(result)
        decision = await self._decide(msgs, agents, hits, result)
        if decision is None:
            result.latency_ms = clock.ms()
            return result

        result.reason = decision.reason
        if decision.action == "answer":
            result.route = "direct"
            result.answer = decision.answer or ""
            result.sources = [
                self._source(i, hits[i - 1]) for i in decision.sources if 1 <= i <= len(hits)
            ]
        elif decision.action == "clarify":
            result.route = "clarify"
            result.answer = decision.question or ""
            result.agent, result.tool = decision.agent, decision.tool
            result.arguments = decision.arguments or None
        else:
            await self._delegate(decision, question, result)
        result.latency_ms = clock.ms()
        return result

    # -- etapas ---------------------------------------------------------------

    async def _rag(self, msgs: list[Message], result: RouterResult) -> list[Hit]:
        spec = self.profile.spec
        if self.retriever is None or not spec.knowledge_bases:
            return []
        clock = _Clock()
        try:
            hits = await self.retriever.search(
                search_query(msgs), spec.knowledge_bases, top_k=spec.top_k, min_score=spec.min_score
            )
        except Exception as exc:  # RAG fora do ar não derruba o atendimento
            result.warnings.append(f"busca na base de conhecimento falhou: {exc}")
            result.steps.append(TraceStep("rag", clock.ms(), {"erro": str(exc)[:300]}))
            return []
        result.steps.append(
            TraceStep(
                "rag",
                clock.ms(),
                {
                    "bases": spec.knowledge_bases,
                    "trechos": len(hits),
                    "scores": [round(h.score, 3) for h in hits],
                },
            )
        )
        return hits

    async def _discover(self, result: RouterResult):
        if not self.profile.agents:
            return []
        clock = _Clock()
        infos = await self.catalog.discover(self.profile.agents)
        online = [a for a in infos if a.status == "online" and a.tools]
        result.steps.append(
            TraceStep(
                "descoberta_mcp",
                clock.ms(),
                {
                    "online": {a.name: [t.name for t in a.tools] for a in online},
                    "offline": {a.name: a.error for a in infos if a.status != "online"},
                },
            )
        )
        for info in infos:
            if info.status != "online":
                result.warnings.append(f"agente {info.name} indisponível: {info.error}")
        return online

    async def _decide(self, msgs, agents, hits, result: RouterResult) -> Decision | None:
        spec = self.profile.spec
        clock = _Clock()
        try:
            decision, calls = await self.decider.decide(
                profile=spec, messages=msgs, agents=agents, hits=hits
            )
        except LLMError as exc:
            result.warnings.append(f"modelo indisponível, usei o decisor offline: {exc}")
            decision, calls = await self._fallback_decider.decide(
                profile=spec, messages=msgs, agents=agents, hits=hits
            )
            result.steps.append(TraceStep("fallback_offline", 0.0, {"erro": str(exc)[:300]}))
        except Exception as exc:
            result.error = f"falha ao decidir a rota: {exc}"
            result.answer = (
                "Desculpe, não consegui processar seu pedido agora. Tente novamente em instantes."
            )
            result.steps.append(TraceStep("decisao", clock.ms(), {"erro": str(exc)[:300]}))
            return None
        usage = Usage()
        for call in calls:
            usage.add(call.usage)
        result.usage.add(usage)
        detail = {"acao": decision.action, "motivo": decision.reason, "chamadas_llm": len(calls)}
        if decision.action == "delegate":
            detail.update(
                {"agente": decision.agent, "tool": decision.tool, "argumentos": decision.arguments}
            )
        if calls:
            detail["tokens"] = usage.to_dict()
        result.steps.append(TraceStep("decisao", clock.ms(), detail))
        return decision

    async def _delegate(self, decision: Decision, question: str, result: RouterResult) -> None:
        spec = self.profile.spec
        agent_spec = self.profile.agent(decision.agent or "")
        result.route = "delegated"
        result.agent, result.tool, result.arguments = (
            decision.agent,
            decision.tool,
            decision.arguments,
        )
        clock = _Clock()
        if agent_spec is None:  # só acontece com decisores customizados
            outcome = ToolCallResult(
                f"agente {decision.agent} não pertence a este perfil", True, None, 0.0
            )
        else:
            try:
                outcome = await self.catalog.call(
                    agent_spec, decision.tool or "", decision.arguments
                )
            except AgentError as exc:
                outcome = ToolCallResult(str(exc), True, None, clock.ms())
        result.steps.append(
            TraceStep(
                "delegacao_mcp",
                clock.ms(),
                {
                    "agente": decision.agent,
                    "tool": decision.tool,
                    "erro": outcome.is_error,
                    "resultado": truncate(outcome.text, 800),
                },
            )
        )
        if outcome.is_error:
            result.warnings.append(f"o agente {decision.agent} retornou erro")
        if not spec.synthesize:
            result.answer = (
                outcome.text if not outcome.is_error else f"O agente retornou erro: {outcome.text}"
            )
            return
        clock = _Clock()
        try:
            answer, call = await self.composer.compose(
                profile=spec,
                question=question,
                decision=decision,
                result_text=outcome.text,
                is_error=outcome.is_error,
            )
        except LLMError as exc:
            result.warnings.append(
                f"síntese pelo modelo falhou, devolvi o resultado do agente: {exc}"
            )
            answer, call = await self._fallback_composer.compose(
                profile=spec,
                question=question,
                decision=decision,
                result_text=outcome.text,
                is_error=outcome.is_error,
            )
        if call is not None:
            result.usage.add(call.usage)
        result.steps.append(TraceStep("sintese", clock.ms(), {"llm": call is not None}))
        result.answer = answer

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
