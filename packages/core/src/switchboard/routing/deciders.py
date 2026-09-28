"""Quem decide e quem redige.

Decisores (escolhem a ação):

* :class:`LLMDecider` — o LLM decide tudo num JSON, validado contra as
  capacidades reais (com uma tentativa de reparo). É o decisor quando o perfil
  não tem modelo de decisão, e o plano B do Jev (confiança baixa ou falha);
* :class:`HeuristicDecider` — sem LLM: casa palavras-chave com tools e skills,
  extrai argumentos do texto e responde de forma extrativa. Modo offline e
  último recurso.

O decisor com Jev (System One) fica em :mod:`.jev_decider`; ele usa as peças
de redação daqui (System Two):

* :class:`LLMAnswerer` / :class:`ExtractiveAnswerer` — resposta direta com RAG;
* :class:`LLMArgumentFiller` / :class:`HeuristicArgumentFiller` — valores dos
  argumentos de uma capacidade já escolhida;
* :class:`LLMComposer` / :class:`TemplateComposer` — síntese do resultado de
  uma tool MCP.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import ProfileSpec
from ..llm.base import ChatModel, ChatResult, Message
from ..rag.retriever import Hit
from ..text import content_tokens, fold, keywords, truncate
from .args import extract_arguments, validate_arguments
from .capabilities import Capability
from .decision import (
    DecisionError,
    clarify_for_missing,
    extract_json_object,
    parse_decision,
    validate_decision,
)
from .prompts import (
    answer_messages,
    arguments_messages,
    decision_messages,
    repair_message,
    synthesis_messages,
)
from .types import Decision, TaskRequest


class Decider(Protocol):
    async def decide(
        self,
        *,
        profile: ProfileSpec,
        messages: Sequence[Message],
        caps: Sequence[Capability],
        hits: Sequence[Hit],
    ) -> tuple[Decision, list[ChatResult]]: ...


def user_turns(messages: Sequence[Message]) -> list[str]:
    return [m.content for m in messages if m.role == "user" and m.content.strip()]


# ---------------------------------------------------------------------------
# decisor LLM (System Two decidindo tudo)


class LLMDecider:
    def __init__(self, chat: ChatModel, *, max_repairs: int = 1):
        self.chat = chat
        self.max_repairs = max_repairs

    async def decide(
        self,
        *,
        profile: ProfileSpec,
        messages: Sequence[Message],
        caps: Sequence[Capability],
        hits: Sequence[Hit],
    ) -> tuple[Decision, list[ChatResult]]:
        prompt = decision_messages(profile, messages, caps, hits)
        calls: list[ChatResult] = []
        result = await self.chat.chat(prompt, json_mode=True)
        calls.append(result)
        for attempt in range(self.max_repairs + 1):
            try:
                decision = parse_decision(result.text)
                decision = validate_decision(
                    decision,
                    caps,
                    allow_clarify=profile.allow_clarify,
                    n_sources=len(hits),
                    max_parallel=profile.max_parallel,
                )
                decision.decided_by = "llm"
                return decision, calls
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
        if err.capability is not None and err.missing and profile.allow_clarify:
            return Decision(
                action="clarify",
                question=clarify_for_missing(err.capability, err.missing),
                reason=f"faltaram argumentos obrigatórios ({', '.join(err.missing)})",
                tasks=[
                    TaskRequest(
                        err.capability.kind,
                        err.capability.owner,
                        err.capability.name,
                        err.arguments,
                    )
                ],
                decided_by="llm",
            )
        text = raw.strip()
        if text and "{" not in text:
            # o modelo ignorou o formato mas respondeu em texto: aproveita
            return Decision(
                action="answer",
                answer=text,
                reason="resposta fora do formato JSON",
                decided_by="llm",
            )
        raise err


# ---------------------------------------------------------------------------
# decisor heurístico (offline)


@dataclass
class _Match:
    cap: Capability
    score: float


GREETINGS = {"oi", "ola", "bom dia", "boa tarde", "boa noite", "hello", "hi", "e ai", "hey", "opa"}
INTERROGATIVES = frozenset(
    "qual quais quanto quanta quantos quantas como onde quando quem porque existe existem".split()
)


def is_question(text: str) -> bool:
    """Pergunta ("Quais documentos…?", "Como faço…") e não pedido de ação ("Simule…")."""
    words = fold(text).split()
    return text.rstrip().endswith("?") or (bool(words) and words[0] in INTERROGATIVES)


# unidades e grandezas: aparecem em qualquer pedido com valores ("30 mil", "12 meses",
# "R$ 5 milhões") e não dizem nada sobre qual capacidade usar
UNIT_WORDS = frozenset(
    keywords(
        "mil milhao milhoes bilhao bilhoes reais real dolar dolares mes meses ano anos "
        "semana semanas dias hora horas minuto minutos vez vezes"
    )
)


def query_keywords(text: str) -> set[str]:
    """Palavras-chave do pedido que ajudam a escolher uma capacidade (sem números e unidades)."""
    return {kw for kw in keywords(text) if not kw.isdigit() and kw not in UNIT_WORDS}


def _instruction(messages: Sequence[Message], limit: int = 3) -> str:
    turns = user_turns(messages)[-limit:]
    return "\n".join(t.strip() for t in turns if t.strip())


class HeuristicDecider:
    min_score = 2.0
    # abaixo disto o casamento é fraco (uma palavra do nome, ou só da descrição):
    # numa pergunta que a base de conhecimento cobre, a base responde melhor
    strong_score = 3.0

    @staticmethod
    def _score(query: set[str], cap: Capability) -> float:
        name_kw = keywords(f"{cap.name} {cap.title}".replace("_", " ").replace("-", " "))
        desc_kw = keywords(cap.description)
        owner_kw = keywords(f"{cap.owner.replace('-', ' ')} {cap.owner_description}")
        # exemplos trazem nomes, valores e palavras de ocasião: contam pouco
        example_kw = keywords(" ".join(cap.examples))
        score = 0.0
        for kw in query:
            if kw in name_kw:
                score += 2.0
            elif kw in desc_kw:
                score += 1.0
            elif kw in owner_kw or kw in example_kw:
                score += 0.5
        return score

    def _best(self, text: str, caps: Sequence[Capability]) -> _Match | None:
        query = query_keywords(text)
        best: _Match | None = None
        for cap in caps:
            score = self._score(query, cap)
            if best is None or score > best.score:
                best = _Match(cap, score)
        return best if best and best.score >= self.min_score else None

    @staticmethod
    def _greeting(question: str, caps: Sequence[Capability]) -> Decision | None:
        folded = fold(question).strip(" !?.,")
        if folded not in GREETINGS and content_tokens(question):
            return None
        lines = [
            "Olá! Sou o assistente virtual. Posso responder dúvidas com base na nossa base de conhecimento"
        ]
        if caps:
            items = [c.description.rstrip(".") or c.title for c in caps]
            lines[0] += " e também acionar especialistas para:"
            lines.extend(f"- {c[0].lower() + c[1:]}" for c in items[:8] if c)
        else:
            lines[0] += "."
        lines.append("Como posso ajudar?")
        return Decision(
            action="answer", answer="\n".join(lines), reason="saudação", decided_by="heuristic"
        )

    async def decide(
        self,
        *,
        profile: ProfileSpec,
        messages: Sequence[Message],
        caps: Sequence[Capability],
        hits: Sequence[Hit],
    ) -> tuple[Decision, list[ChatResult]]:
        turns = user_turns(messages)
        question = turns[-1] if turns else ""
        greeting = self._greeting(question, caps)
        if greeting:
            return greeting, []

        # 1) alguma capacidade combina com o pedido? (tenta também juntar com o
        #    turno anterior, para responder a uma pergunta de esclarecimento)
        match, context = self._best(question, caps), question
        if match is not None and match.score < self.strong_score and hits and is_question(question):
            # "Quais documentos preciso para pedir crédito?" cita o crédito, mas não pede a análise
            return self._from_knowledge(hits, f"pergunta com casamento fraco ({match.cap.key})")
        if match is None and len(turns) >= 2:
            joined = f"{turns[-2]}\n{question}"
            match = self._best(joined, caps)
            context = joined
        if match is not None:
            cap = match.cap
            arguments = heuristic_arguments(cap, messages, context=context, focus=question)
            arguments, _errors, missing = validate_arguments(cap.arguments_schema, arguments)
            reason = f"palavras-chave casaram com {cap.key} (score {match.score:g})"
            task = TaskRequest(
                cap.kind,
                cap.owner,
                cap.name,
                arguments,
                str(arguments.get("instrucao") or "") if cap.input_schema is None else "",
            )
            if missing and profile.allow_clarify:
                return (
                    Decision(
                        action="clarify",
                        question=clarify_for_missing(cap, missing),
                        reason=reason + f"; faltam {', '.join(missing)}",
                        tasks=[task],
                        decided_by="heuristic",
                    ),
                    [],
                )
            if not missing:
                return (
                    Decision(
                        action="tool" if cap.kind == "tool" else "delegate",
                        tasks=[task],
                        reason=reason,
                        decided_by="heuristic",
                    ),
                    [],
                )

        # 2) responde de forma extrativa com a base de conhecimento
        return self._from_knowledge(hits)

    @staticmethod
    def _from_knowledge(
        hits: Sequence[Hit], why: str | None = None
    ) -> tuple[Decision, list[ChatResult]]:
        answer, sources = extractive_answer(hits)
        reason = (
            f"trecho mais próximo com score {hits[0].score:.2f}"
            if hits
            else "sem trecho relevante e sem capacidade compatível"
        )
        return (
            Decision(
                action="answer",
                answer=answer,
                sources=sources,
                reason=f"{why}; {reason}" if why else reason,
                decided_by="heuristic",
            ),
            [],
        )


def heuristic_arguments(
    cap: Capability,
    messages: Sequence[Message],
    *,
    context: str | None = None,
    focus: str | None = None,
) -> dict[str, Any]:
    if cap.kind == "skill" and cap.input_schema is None:
        return {"instrucao": _instruction(messages)}
    turns = user_turns(messages)
    text = context if context is not None else "\n".join(turns[-2:])
    ignore = keywords(f"{cap.name.replace('_', ' ')} {cap.description} {cap.owner}")
    return extract_arguments(
        cap.arguments_schema,
        text,
        focus=focus if focus is not None else (turns[-1] if turns else ""),
        ignore=ignore,
    )


NOT_FOUND = (
    "Não encontrei essa informação nas bases de conhecimento e nenhuma ferramenta ou agente "
    "disponível atende esse pedido. Pode reformular ou dar mais detalhes?"
)


def extractive_answer(hits: Sequence[Hit]) -> tuple[str, list[int]]:
    if not hits:
        return NOT_FOUND, []
    best = hits[0]
    used = [1]
    parts = [truncate(best.content, 900)]
    if len(hits) > 1 and hits[1].document != best.document and hits[1].score >= best.score * 0.85:
        parts.append(truncate(hits[1].content, 500))
        used.append(2)
    return "Encontrei isto na base de conhecimento:\n\n" + "\n\n".join(parts), used


# ---------------------------------------------------------------------------
# redação (System Two)


_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


class Answerer(Protocol):
    async def answer(
        self,
        *,
        profile: ProfileSpec,
        messages: Sequence[Message],
        hits: Sequence[Hit],
        in_scope: bool = True,
    ) -> tuple[str, list[int], ChatResult | None]: ...


class LLMAnswerer:
    def __init__(self, chat: ChatModel):
        self.chat = chat

    async def answer(
        self,
        *,
        profile: ProfileSpec,
        messages: Sequence[Message],
        hits: Sequence[Hit],
        in_scope: bool = True,
    ) -> tuple[str, list[int], ChatResult | None]:
        result = await self.chat.chat(answer_messages(profile, messages, hits, in_scope=in_scope))
        text = result.text.strip()
        if not text:
            raise DecisionError("o modelo devolveu uma resposta vazia")
        cited = sorted({int(n) for n in _CITATION_RE.findall(text) if 1 <= int(n) <= len(hits)})
        return text, cited, result


class ExtractiveAnswerer:
    async def answer(
        self,
        *,
        profile: ProfileSpec,
        messages: Sequence[Message],
        hits: Sequence[Hit],
        in_scope: bool = True,
    ) -> tuple[str, list[int], ChatResult | None]:
        if not in_scope:
            return (
                "Esse pedido está fora do que eu consigo atender por aqui. Posso ajudar com dúvidas "
                "da nossa base de conhecimento ou acionar os especialistas disponíveis.",
                [],
                None,
            )
        answer, sources = extractive_answer(hits)
        return answer, sources, None


class ArgumentFiller(Protocol):
    async def fill(
        self, *, profile: ProfileSpec, messages: Sequence[Message], cap: Capability
    ) -> tuple[dict[str, Any], ChatResult | None]: ...


class LLMArgumentFiller:
    def __init__(self, chat: ChatModel):
        self.chat = chat

    async def fill(
        self, *, profile: ProfileSpec, messages: Sequence[Message], cap: Capability
    ) -> tuple[dict[str, Any], ChatResult | None]:
        result = await self.chat.chat(arguments_messages(profile, messages, cap), json_mode=True)
        data = extract_json_object(result.text)
        arguments = data.get("arguments", data.get("argumentos", data))
        if not isinstance(arguments, dict):
            raise DecisionError("'arguments' precisa ser um objeto JSON")
        return arguments, result


class HeuristicArgumentFiller:
    async def fill(
        self, *, profile: ProfileSpec, messages: Sequence[Message], cap: Capability
    ) -> tuple[dict[str, Any], ChatResult | None]:
        return heuristic_arguments(cap, messages), None


class Composer(Protocol):
    async def compose(
        self,
        *,
        profile: ProfileSpec,
        question: str,
        capability: str,
        arguments: dict[str, Any],
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
        capability: str,
        arguments: dict[str, Any],
        result_text: str,
        is_error: bool,
    ) -> tuple[str, ChatResult | None]:
        messages = synthesis_messages(
            profile, question, capability, arguments, result_text, is_error
        )
        result = await self.chat.chat(messages)
        return result.text.strip() or result_text, result


class TemplateComposer:
    async def compose(
        self,
        *,
        profile: ProfileSpec,
        question: str,
        capability: str,
        arguments: dict[str, Any],
        result_text: str,
        is_error: bool,
    ) -> tuple[str, ChatResult | None]:
        if is_error:
            return f"Não consegui concluir com {capability}: {truncate(result_text, 600)}", None
        return result_text.strip() or "A ferramenta concluiu, mas não retornou detalhes.", None
