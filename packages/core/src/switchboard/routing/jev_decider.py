"""Decisão em dois sistemas: o Jev (System One) decide, o LLM (System Two) redige.

Perguntas feitas ao Jev (instruções em inglês, idioma em que ele é mais
preciso; a conversa vai como está):

1. ``route`` (choice) — quem atende a última mensagem: uma das capacidades
   (tool MCP ou skill de agente A2A), ``knowledge`` (responder com a base /
   conversa) ou ``out_of_scope``;
2. ``needs_<i>`` (noul, uma por skill de agente) — o pedido também pede esta
   tarefa? Serve para delegar tarefas independentes em paralelo (*fan-out*);
3. ``kb_answers`` (noul, em paralelo, só com trechos do RAG) — os trechos
   respondem o pedido?
4. ``stated_<j>`` (noul, depois da escolha) — cada parâmetro obrigatório já foi
   informado pelo usuário? O que não foi vira pergunta de esclarecimento, e um
   valor que o LLM tenha preenchido para um parâmetro não informado é
   descartado (guarda contra alucinação).

Confiança do ``route`` abaixo de ``decision_threshold`` (do perfil), ou Jev
indisponível: a decisão sobe para o decisor de reserva (o LLM decidindo tudo,
ou o heurístico no modo offline) e o trace registra o motivo.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import ProfileSpec
from ..contracts.terms import validate as validate_schema
from ..errors import DecisionModelError, LLMError
from ..jev import JevClient, JevResult, choice, noul
from ..llm.base import ChatResult, Message
from ..rag.retriever import Hit
from ..text import truncate
from ..tracing import Span, SpanRecorder, new_trace_id
from .args import describe_fields, validate_arguments
from .capabilities import Capability, by_key
from .deciders import Answerer, ArgumentFiller, Decider, user_turns
from .decision import DecisionError, clarify_for_missing
from .types import Decision, TaskRequest

KNOWLEDGE = "knowledge"
OUT_OF_SCOPE = "out_of_scope"

ROUTE_INSTRUCTIONS = (
    "Decide who should handle the user's latest message, considering the whole conversation. "
    "Pick a capability only when the user is asking for what it does. Pick 'knowledge' for "
    "questions about policies, procedures, opening hours, products, or for small talk "
    "(greetings, thanks). Pick 'out_of_scope' when the request is unrelated to this assistant "
    "and no capability fits."
)


@dataclass
class JevSettings:
    kb_threshold: float = 0.35  # abaixo: os trechos não respondem o pedido
    fanout_threshold: float = 0.8  # noul mínimo para uma skill extra entrar no fan-out
    stated_threshold: float = 0.25  # abaixo: o parâmetro não foi informado


@dataclass
class DecisionContext:
    """Onde o decisor registra spans e avisos (o motor cria um por pedido)."""

    recorder: SpanRecorder = field(default_factory=lambda: SpanRecorder(new_trace_id()))
    parent: Span | None = None
    warnings: list[str] = field(default_factory=list)

    def span(self, kind: str, name: str, **attributes: Any):
        return self.recorder.span(kind, name, parent=self.parent or "root", **attributes)


def _criteria(cap: Capability) -> dict[str, Any]:
    kind = (
        "Quick action executed immediately (tool)"
        if cap.kind == "tool"
        else "Task delegated to a specialist agent"
    )
    item: dict[str, Any] = {"what": f"{kind}: {cap.summary()}"}
    if cap.examples:
        item["examples"] = list(cap.examples[:3])
    return item


def conversation_state(profile: ProfileSpec, messages: Sequence[Message]) -> dict[str, Any]:
    convo = [
        {"role": m.role, "content": truncate(m.content, 800)}
        for m in messages
        if m.role in ("user", "assistant") and m.content.strip()
    ][-6:]
    turns = user_turns(messages)
    return {
        "assistant_scope": truncate(profile.description or profile.system_prompt, 300),
        "conversation": convo,
        "latest_user_message": turns[-1] if turns else "",
    }


def _param_label(schema: dict[str, Any], name: str) -> str:
    return describe_fields(schema, [name])[0]


class JevDecider:
    def __init__(
        self,
        jev: JevClient,
        *,
        answerer: Answerer,
        filler: ArgumentFiller,
        escalate: Decider,
        settings: JevSettings | None = None,
    ):
        self.jev = jev
        self.answerer = answerer
        self.filler = filler
        self.escalate = escalate
        self.settings = settings or JevSettings()

    async def _ask(
        self, ctx: DecisionContext, label: str, state: Any, questions: dict[str, Any]
    ) -> JevResult:
        with ctx.span("jev", f"jev: {label}", perguntas=list(questions)) as span:
            result = await self.jev.evaluate(state, questions)
            span.attributes.update(
                modelo=result.model,
                respostas={k: a.to_dict() for k, a in result.answers.items()},
                tokens=result.usage.to_dict(),
            )
            return result

    async def decide(
        self,
        *,
        profile: ProfileSpec,
        messages: Sequence[Message],
        caps: Sequence[Capability],
        hits: Sequence[Hit],
        ctx: DecisionContext | None = None,
    ) -> tuple[Decision, list[ChatResult]]:
        ctx = ctx or DecisionContext()
        state = conversation_state(profile, messages)
        options: dict[str, Any] = {
            KNOWLEDGE: {
                "what": "Answer directly from the company's knowledge base, or small talk (greetings, thanks).",
                "not_for": "Requests that need a calculation, a lookup or action in a system, or a specialist agent.",
            },
            OUT_OF_SCOPE: {
                "what": "Unrelated to this assistant's services, and no listed capability can fulfill it."
            },
        }
        usable = list(caps)[:253]
        for cap in usable:
            options[cap.key] = _criteria(cap)
        questions: dict[str, Any] = {"route": choice(ROUTE_INSTRUCTIONS, options)}
        skills = [c for c in usable if c.kind == "skill"]
        if len(skills) >= 2 and profile.max_parallel > 1:
            for i, cap in enumerate(skills):
                questions[f"needs_{i}"] = noul(
                    f"Does the user's latest message ask for this task? {cap.summary()}"
                )
        kb_call = None
        if hits:
            kb_state = {
                "latest_user_message": state["latest_user_message"],
                "passages": [
                    {"n": i, "document": h.document, "text": truncate(h.content, 700)}
                    for i, h in enumerate(hits, start=1)
                ],
            }
            kb_call = self._ask(
                ctx,
                "conhecimento",
                kb_state,
                {
                    "kb_answers": noul(
                        "Do the passages contain the information needed to answer the user's latest message?"
                    )
                },
            )
        try:
            route_res, kb_res = await asyncio.gather(
                self._ask(ctx, "rota", state, questions),
                kb_call if kb_call is not None else asyncio.sleep(0, result=None),
            )
        except DecisionModelError as exc:
            return await self._escalate(
                profile, messages, caps, hits, ctx, f"Jev indisponível: {exc}"
            )

        route = route_res.get("route")
        if route is None or route.choice is None:
            return await self._escalate(
                profile, messages, caps, hits, ctx, "Jev não respondeu a rota"
            )
        confidence = (
            route.confidence
            if route.confidence is not None
            else max((p for _, p in route.ranked()), default=0.0)
        )
        if confidence < profile.decision_threshold:
            top = ", ".join(f"{k} {p:.2f}" for k, p in route.ranked()[:3])
            return await self._escalate(
                profile,
                messages,
                caps,
                hits,
                ctx,
                f"confiança do Jev {confidence:.2f} < {profile.decision_threshold:.2f} ({top})",
            )

        if route.choice in (KNOWLEDGE, OUT_OF_SCOPE) or by_key(usable, route.choice) is None:
            in_scope = route.choice != OUT_OF_SCOPE
            relevant = list(hits)
            kb_answer = kb_res.get("kb_answers") if kb_res is not None else None
            if (
                kb_answer is not None
                and kb_answer.noul is not None
                and kb_answer.noul < self.settings.kb_threshold
            ):
                relevant = []  # os trechos não respondem: melhor dizer que não sabe
            with ctx.span("llm", "llm: resposta") as span:
                try:
                    text, sources, call = await self.answerer.answer(
                        profile=profile, messages=messages, hits=relevant, in_scope=in_scope
                    )
                except (LLMError, DecisionError) as exc:
                    span.finish("warn", erro=str(exc)[:300])
                    return await self._escalate(
                        profile, messages, caps, hits, ctx, f"falha ao redigir a resposta: {exc}"
                    )
                span.attributes.update(llm=call is not None, trechos=len(relevant))
            index = {id(h): i for i, h in enumerate(hits, start=1)}
            mapped = [index[id(relevant[s - 1])] for s in sources if 1 <= s <= len(relevant)]
            reason = "Jev: " + ("responder com a base" if in_scope else "fora do escopo")
            if kb_answer is not None and kb_answer.noul is not None:
                reason += f" (trechos respondem: {kb_answer.noul:.2f})"
            return (
                Decision(
                    action="answer",
                    answer=text,
                    sources=mapped,
                    reason=reason,
                    decided_by="jev",
                    confidence=confidence,
                ),
                [call] if call else [],
            )

        primary = by_key(usable, route.choice)
        assert primary is not None
        selected = [primary]
        if primary.kind == "skill":
            for i, cap in enumerate(skills):
                answer = route_res.get(f"needs_{i}")
                if (
                    cap.key != primary.key
                    and answer is not None
                    and (answer.noul or 0.0) >= self.settings.fanout_threshold
                    and len(selected) < profile.max_parallel
                ):
                    selected.append(cap)

        try:
            filled = await asyncio.gather(
                *(self._arguments(profile, messages, state, cap, ctx) for cap in selected)
            )
        except DecisionModelError as exc:
            return await self._escalate(
                profile, messages, caps, hits, ctx, f"Jev indisponível: {exc}"
            )
        except (LLMError, DecisionError) as exc:
            return await self._escalate(
                profile, messages, caps, hits, ctx, f"falha ao extrair os argumentos: {exc}"
            )
        calls = [c for _, _, c in filled if c is not None]
        missing_questions = []
        tasks = []
        for cap, (arguments, missing, _call) in zip(selected, filled, strict=True):
            if missing:
                missing_questions.append((cap, missing))
            tasks.append(
                TaskRequest(
                    cap.kind,
                    cap.owner,
                    cap.name,
                    arguments,
                    str(arguments.get("instrucao") or "") if cap.input_schema is None else "",
                )
            )
        reason = f"Jev: {primary.key} (confiança {confidence:.2f})"
        if len(selected) > 1:
            reason += "; também: " + ", ".join(c.key for c in selected[1:])
        if missing_questions:
            if not profile.allow_clarify:
                return await self._escalate(
                    profile,
                    messages,
                    caps,
                    hits,
                    ctx,
                    "faltam parâmetros e o perfil não faz perguntas",
                )
            if len(missing_questions) == 1:
                cap, missing = missing_questions[0]
                question = clarify_for_missing(cap, missing)
            else:
                labels = [
                    label
                    for cap, missing in missing_questions
                    for label in describe_fields(cap.arguments_schema, missing)
                ]
                question = f"Consigo ajudar com isso, mas preciso de: {', '.join(dict.fromkeys(labels))}. Pode me informar?"
            return (
                Decision(
                    action="clarify",
                    question=question,
                    reason=reason
                    + "; faltam "
                    + ", ".join(f"{cap.key}: {', '.join(m)}" for cap, m in missing_questions),
                    tasks=tasks,
                    decided_by="jev",
                    confidence=confidence,
                ),
                calls,
            )
        return (
            Decision(
                action="tool" if primary.kind == "tool" else "delegate",
                tasks=tasks,
                reason=reason,
                decided_by="jev",
                confidence=confidence,
            ),
            calls,
        )

    async def _arguments(
        self,
        profile: ProfileSpec,
        messages: Sequence[Message],
        state: dict[str, Any],
        cap: Capability,
        ctx: DecisionContext,
    ) -> tuple[dict[str, Any], list[str], ChatResult | None]:
        """Valores (System Two) + conferência do que foi informado (System One)."""
        schema = cap.arguments_schema
        required = [str(r) for r in schema.get("required") or [] if r != "instrucao"]
        stated_call = None
        if required:
            stated_questions = {
                f"stated_{j}": noul(
                    f"Has the user already provided this information in the conversation: "
                    f"{_param_label(schema, name)}?",
                    true="The value is explicitly given or can be read directly from the messages.",
                    false="The value is not mentioned anywhere in the conversation.",
                )
                for j, name in enumerate(required)
            }
            stated_call = self._ask(ctx, f"parâmetros de {cap.key}", state, stated_questions)
        with ctx.span("llm", f"llm: argumentos de {cap.key}") as span:
            (arguments, call), stated = await asyncio.gather(
                self.filler.fill(profile=profile, messages=messages, cap=cap),
                stated_call if stated_call is not None else asyncio.sleep(0, result=None),
            )
            span.attributes.update(llm=call is not None, argumentos=arguments)
        dropped = []
        if stated is not None:
            for j, name in enumerate(required):
                answer = stated.get(f"stated_{j}")
                if (
                    answer is not None
                    and answer.noul is not None
                    and answer.noul < self.settings.stated_threshold
                    and name in arguments
                ):
                    arguments.pop(name)  # o Jev diz que o usuário não informou: não inventa
                    dropped.append(name)
        converted, errors, missing = validate_arguments(schema, arguments)
        if errors:
            bad = {e.split(":", 1)[0] for e in errors}
            for name in bad:
                converted.pop(name, None)
            missing = sorted(
                set(missing) | ({n for n in bad if n in (schema.get("required") or [])})
            )
        if not missing and cap.kind == "skill" and cap.input_schema is not None:
            problems = validate_schema(converted, cap.input_schema)
            if problems:
                raise DecisionError(
                    f"argumentos fora do contrato de {cap.key}: " + "; ".join(problems)
                )
        if dropped:
            ctx.warnings.append(
                f"{cap.key}: descartei {', '.join(dropped)} (o Jev indica que o usuário não informou)"
            )
        return converted, missing, call

    async def _escalate(
        self,
        profile: ProfileSpec,
        messages: Sequence[Message],
        caps: Sequence[Capability],
        hits: Sequence[Hit],
        ctx: DecisionContext,
        reason: str,
    ) -> tuple[Decision, list[ChatResult]]:
        ctx.warnings.append(f"decisão escalada: {reason}")
        with ctx.span("decisao", "escalada", motivo=reason):
            decision, calls = await self.escalate.decide(
                profile=profile, messages=messages, caps=caps, hits=hits
            )
        decision.reason = f"{decision.reason} [escalado: {reason}]".strip()
        return decision, calls
