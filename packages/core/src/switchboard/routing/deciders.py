"""Quem decide entre responder, delegar ou esclarecer.

* :class:`LLMDecider` — pergunta ao modelo configurado (JSON), valida contra o
  catálogo real de agentes e, se a resposta vier torta, pede um reparo.
* :class:`HeuristicDecider` — sem LLM: casa palavras-chave do pedido com o
  nome/descrição das tools, extrai argumentos do texto e responde de forma
  extrativa com a base de conhecimento. É o modo offline e o plano B quando o
  provedor de LLM falha.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..agents.catalog import AgentInfo, ToolInfo
from ..config import ProfileSpec
from ..llm.base import ChatModel, ChatResult, Message
from ..rag.retriever import Hit
from ..text import content_tokens, fold, keywords, truncate
from .args import extract_arguments, validate_arguments
from .decision import DecisionError, clarify_for_missing, parse_decision, validate_decision
from .prompts import decision_messages, repair_message, synthesis_messages
from .types import Decision


class Decider(Protocol):
    async def decide(
        self,
        *,
        profile: ProfileSpec,
        messages: Sequence[Message],
        agents: Sequence[AgentInfo],
        hits: Sequence[Hit],
    ) -> tuple[Decision, list[ChatResult]]: ...


def user_turns(messages: Sequence[Message]) -> list[str]:
    return [m.content for m in messages if m.role == "user" and m.content.strip()]


class LLMDecider:
    def __init__(self, chat: ChatModel, *, max_repairs: int = 1):
        self.chat = chat
        self.max_repairs = max_repairs

    async def decide(
        self,
        *,
        profile: ProfileSpec,
        messages: Sequence[Message],
        agents: Sequence[AgentInfo],
        hits: Sequence[Hit],
    ) -> tuple[Decision, list[ChatResult]]:
        prompt = decision_messages(profile, messages, agents, hits)
        calls: list[ChatResult] = []
        result = await self.chat.chat(prompt, json_mode=True)
        calls.append(result)
        for attempt in range(self.max_repairs + 1):
            try:
                decision = parse_decision(result.text)
                return (
                    validate_decision(
                        decision, agents, allow_clarify=profile.allow_clarify, n_sources=len(hits)
                    ),
                    calls,
                )
            except DecisionError as err:
                if attempt < self.max_repairs:
                    prompt = [*prompt, Message("assistant", result.text), repair_message(str(err))]
                    result = await self.chat.chat(prompt, json_mode=True)
                    calls.append(result)
                    continue
                return self._fallback(err, result.text, profile), calls
        raise AssertionError("inalcançável")  # pragma: no cover

    @staticmethod
    def _fallback(err: DecisionError, raw: str, profile: ProfileSpec) -> Decision:
        if err.tool is not None and err.missing and profile.allow_clarify:
            return Decision(
                action="clarify",
                question=clarify_for_missing(err.tool, err.missing, err.agent),
                reason=f"faltaram argumentos obrigatórios ({', '.join(err.missing)})",
            )
        text = raw.strip()
        if text and "{" not in text:
            # o modelo ignorou o formato mas respondeu em texto: aproveita
            return Decision(action="answer", answer=text, reason="resposta fora do formato JSON")
        raise err


@dataclass
class _ToolMatch:
    agent: AgentInfo
    tool: ToolInfo
    score: float


GREETINGS = {"oi", "ola", "bom dia", "boa tarde", "boa noite", "hello", "hi", "e ai", "hey", "opa"}


class HeuristicDecider:
    min_tool_score = 2.0

    @staticmethod
    def _tool_score(query: set[str], agent: AgentInfo, tool: ToolInfo) -> float:
        name_kw = keywords(tool.name.replace("_", " ").replace("-", " "))
        desc_kw = keywords(tool.description)
        agent_kw = keywords(f"{agent.name.replace('-', ' ')} {agent.description}")
        score = 0.0
        for kw in query:
            if kw in name_kw:
                score += 2.0
            elif kw in desc_kw:
                score += 1.0
            elif kw in agent_kw:
                score += 0.5
        return score

    def _best_tool(self, text: str, agents: Sequence[AgentInfo]) -> _ToolMatch | None:
        query = keywords(text)
        best: _ToolMatch | None = None
        for agent in agents:
            for tool in agent.tools:
                score = self._tool_score(query, agent, tool)
                if best is None or score > best.score:
                    best = _ToolMatch(agent, tool, score)
        return best if best and best.score >= self.min_tool_score else None

    @staticmethod
    def _greeting(question: str, agents: Sequence[AgentInfo]) -> Decision | None:
        folded = fold(question).strip(" !?.,")
        if folded not in GREETINGS and content_tokens(question):
            return None
        lines = [
            "Olá! Sou o assistente virtual. Posso responder dúvidas com base na nossa base de conhecimento"
        ]
        if agents:
            caps = [t.description.rstrip(".") or t.name for a in agents for t in a.tools]
            lines[0] += " e também acionar especialistas para:"
            lines.extend(f"- {c[0].lower() + c[1:]}" for c in caps[:8])
        else:
            lines[0] += "."
        lines.append("Como posso ajudar?")
        return Decision(action="answer", answer="\n".join(lines), reason="saudação")

    async def decide(
        self,
        *,
        profile: ProfileSpec,
        messages: Sequence[Message],
        agents: Sequence[AgentInfo],
        hits: Sequence[Hit],
    ) -> tuple[Decision, list[ChatResult]]:
        turns = user_turns(messages)
        question = turns[-1] if turns else ""
        greeting = self._greeting(question, agents)
        if greeting:
            return greeting, []

        # 1) alguma tool combina com o pedido? (tenta também juntar com o turno
        #    anterior, para responder a uma pergunta de esclarecimento)
        match, context = self._best_tool(question, agents), question
        if match is None and len(turns) >= 2:
            joined = f"{turns[-2]}\n{question}"
            match = self._best_tool(joined, agents)
            context = joined
        if match is not None:
            ignore = keywords(
                f"{match.tool.name.replace('_', ' ')} {match.tool.description} {match.agent.name}"
            )
            arguments = extract_arguments(
                match.tool.input_schema, context, focus=question, ignore=ignore
            )
            arguments, _errors, missing = validate_arguments(match.tool.input_schema, arguments)
            reason = f"palavras-chave casaram com {match.agent.name}/{match.tool.name} (score {match.score:g})"
            if missing and profile.allow_clarify:
                return (
                    Decision(
                        action="clarify",
                        question=clarify_for_missing(match.tool, missing, match.agent),
                        reason=reason + f"; faltam {', '.join(missing)}",
                        agent=match.agent.name,
                        tool=match.tool.name,
                        arguments=arguments,
                    ),
                    [],
                )
            if not missing:
                return (
                    Decision(
                        action="delegate",
                        agent=match.agent.name,
                        tool=match.tool.name,
                        arguments=arguments,
                        reason=reason,
                    ),
                    [],
                )

        # 2) responde de forma extrativa com a base de conhecimento
        if hits:
            best = hits[0]
            used = [1]
            parts = [truncate(best.content, 900)]
            if (
                len(hits) > 1
                and hits[1].document != best.document
                and hits[1].score >= best.score * 0.85
            ):
                parts.append(truncate(hits[1].content, 500))
                used.append(2)
            answer = "Encontrei isto na base de conhecimento:\n\n" + "\n\n".join(parts)
            return (
                Decision(
                    action="answer",
                    answer=answer,
                    sources=used,
                    reason=f"trecho mais próximo com score {best.score:.2f}",
                ),
                [],
            )
        answer = (
            "Não encontrei essa informação nas bases de conhecimento e nenhum agente disponível "
            "atende esse pedido. Pode reformular ou dar mais detalhes?"
        )
        return Decision(
            action="answer", answer=answer, reason="sem trecho relevante e sem tool compatível"
        ), []


class Composer(Protocol):
    async def compose(
        self,
        *,
        profile: ProfileSpec,
        question: str,
        decision: Decision,
        result_text: str,
        is_error: bool,
    ) -> tuple[str, ChatResult | None]: ...


class LLMComposer:
    def __init__(self, chat: ChatModel):
        self.chat = chat

    async def compose(
        self,
        *,
        profile: ProfileSpec,
        question: str,
        decision: Decision,
        result_text: str,
        is_error: bool,
    ) -> tuple[str, ChatResult | None]:
        messages = synthesis_messages(
            profile,
            question,
            decision.agent or "",
            decision.tool or "",
            decision.arguments,
            result_text,
            is_error,
        )
        result = await self.chat.chat(messages)
        return result.text.strip() or result_text, result


class TemplateComposer:
    async def compose(
        self,
        *,
        profile: ProfileSpec,
        question: str,
        decision: Decision,
        result_text: str,
        is_error: bool,
    ) -> tuple[str, ChatResult | None]:
        if is_error:
            return (
                f"Não consegui concluir com o agente {decision.agent}: {truncate(result_text, 600)}",
                None,
            )
        return result_text.strip() or "O agente concluiu a tarefa, mas não retornou detalhes.", None
