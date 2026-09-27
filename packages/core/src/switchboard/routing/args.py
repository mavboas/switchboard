"""Argumentos de tools: validação contra o JSON Schema e extração heurística.

``validate_arguments`` confere o que o LLM (ou o decisor heurístico) propôs
contra o ``inputSchema`` da tool MCP antes de acionar o agente: converte tipos
simples ("100000" -> 100000), confere ``enum`` e aponta campos obrigatórios
ausentes — que viram uma pergunta de esclarecimento em vez de uma chamada
fadada ao erro.

``extract_arguments`` é usado só no modo offline: acha valores no texto do
usuário (R$, mil, %, meses, anos…) e casa com as propriedades do schema pelo
nome e pela descrição.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..text import fold, keywords, truncate

_NUMBER = r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?"
_MULT = {"mil": 1_000, "k": 1_000, "milhao": 1_000_000, "milhoes": 1_000_000, "mi": 1_000_000}

_MONEY_RE = re.compile(
    rf"(?:r\$\s*(?P<a>{_NUMBER})\s*(?P<am>mil|milhao|milhoes|mi|k)?)|(?:(?P<b>{_NUMBER})\s*(?P<bm>mil|milhao|milhoes|mi|k)\b(?:\s*reais)?)|(?:(?P<c>{_NUMBER})\s*reais)"
)
_PERCENT_RE = re.compile(rf"(?P<n>{_NUMBER})\s*(?:%|por\s*cento|a\.\s?m\.?(?!\w)|ao\s+mes\b)")
_MONTHS_RE = re.compile(rf"(?P<n>{_NUMBER})\s*(?:meses|mes|parcelas|prestacoes|x\b)")
_YEARS_RE = re.compile(rf"(?P<n>{_NUMBER})\s*anos?\b")
_PLAIN_RE = re.compile(rf"(?P<n>{_NUMBER})")

_ROLE_WORDS = {
    "percent": ("taxa", "juros", "percent", "rate", "%"),
    "term": ("prazo", "meses", "parcelas", "month", "term", "periodo", "duracao"),
    "money": (
        "valor",
        "montante",
        "principal",
        "amount",
        "quantia",
        "credito",
        "emprestimo",
        "financiado",
    ),
    "id": ("codigo", "protocolo", "ticket", "numero do chamado", "identificador"),
    "title": ("titulo", "assunto", "title", "subject", "resumo"),
    "text": (
        "descricao",
        "description",
        "detalhe",
        "mensagem",
        "texto",
        "pergunta",
        "query",
        "consulta",
        "task",
        "tarefa",
        "pedido",
        "problema",
    ),
}


def parse_number(raw: str) -> float:
    """Número em formato brasileiro ou internacional ("100.000,50", "1,5", "2.5")."""
    raw = raw.strip()
    if "," in raw and "." in raw:
        raw = raw.replace(".", "").replace(",", ".")
    elif "," in raw:
        raw = raw.replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", raw):
        raw = raw.replace(".", "")
    return float(raw)


@dataclass
class _Mention:
    kind: str  # money | percent | months | years | plain
    value: float
    start: int


def find_mentions(text: str) -> list[_Mention]:
    folded = fold(text)
    mentions: list[_Mention] = []
    taken: list[tuple[int, int]] = []

    def free(span: tuple[int, int]) -> bool:
        return all(span[1] <= s or span[0] >= e for s, e in taken)

    for kind, regex in (("percent", _PERCENT_RE), ("months", _MONTHS_RE), ("years", _YEARS_RE)):
        for m in regex.finditer(folded):
            if free(m.span()):
                mentions.append(_Mention(kind, parse_number(m.group("n")), m.start()))
                taken.append(m.span())
    for m in _MONEY_RE.finditer(folded):
        if not free(m.span()):
            continue
        num = m.group("a") or m.group("b") or m.group("c")
        mult = _MULT.get(m.group("am") or m.group("bm") or "", 1)
        mentions.append(_Mention("money", parse_number(num) * mult, m.start()))
        taken.append(m.span())
    for m in _PLAIN_RE.finditer(folded):
        if free(m.span()):
            mentions.append(_Mention("plain", parse_number(m.group("n")), m.start()))
            taken.append(m.span())
    mentions.sort(key=lambda x: x.start)
    return mentions


def _types(prop: dict[str, Any]) -> list[str]:
    t = prop.get("type")
    if isinstance(t, list):
        return [x for x in t if x != "null"]
    if isinstance(t, str):
        return [t]
    for key in ("anyOf", "oneOf"):
        if isinstance(prop.get(key), list):
            found: list[str] = []
            for option in prop[key]:
                found.extend(_types(option))
            return found
    return []


def _enum(prop: dict[str, Any]) -> list[Any] | None:
    if isinstance(prop.get("enum"), list):
        return prop["enum"]
    for key in ("anyOf", "oneOf"):
        for option in prop.get(key) or []:
            if isinstance(option, dict) and isinstance(option.get("enum"), list):
                return option["enum"]
    return None


def _is_numeric(prop: dict[str, Any]) -> bool:
    types = _types(prop)
    return "number" in types or "integer" in types


def _role(name: str, prop: dict[str, Any]) -> str | None:
    """Papel semântico de uma propriedade; o nome pesa mais que a descrição."""
    numeric = _is_numeric(prop)
    roles = ("percent", "term", "money") if numeric else ("id", "title", "text")
    name_text = fold(name.replace("_", " ")).strip()
    if not numeric and (
        name_text == "id" or name_text.endswith(" id") or name_text.startswith("id ")
    ):
        return "id"
    description = fold(f"{prop.get('title', '')} {prop.get('description', '')}")
    for source in (name_text, description):
        for role in roles:
            if any(word in source for word in _ROLE_WORDS[role]):
                return role
    return None


def _coerce(value: Any, prop: dict[str, Any]) -> tuple[Any, str | None]:
    enum = _enum(prop)
    types = _types(prop)
    if value is None:
        return None, None
    if enum is not None:
        for option in enum:
            if value == option or (
                isinstance(option, str) and fold(str(value)).strip() == fold(option)
            ):
                return option, None
        return value, f"valor {value!r} fora das opções {enum}"
    if not types:
        return value, None
    if "integer" in types:
        try:
            number = parse_number(str(value)) if isinstance(value, str) else float(value)
            if number.is_integer():
                return int(number), None
            if "number" in types:
                return number, None
            return value, f"esperava inteiro, veio {value!r}"
        except (TypeError, ValueError):
            if "string" not in types:
                return value, f"esperava inteiro, veio {value!r}"
    if "number" in types:
        try:
            return (parse_number(value) if isinstance(value, str) else float(value)), None
        except (TypeError, ValueError):
            if "string" not in types:
                return value, f"esperava número, veio {value!r}"
    if "boolean" in types:
        if isinstance(value, bool):
            return value, None
        folded = fold(str(value)).strip()
        if folded in {"true", "sim", "s", "yes", "1"}:
            return True, None
        if folded in {"false", "nao", "n", "no", "0"}:
            return False, None
        if "string" not in types:
            return value, f"esperava sim/não, veio {value!r}"
    if "string" in types:
        if isinstance(value, (dict, list)):
            return value, f"esperava texto, veio {type(value).__name__}"
        return str(value), None
    if "array" in types and not isinstance(value, list):
        return [value], None
    return value, None


def validate_arguments(
    schema: dict[str, Any], args: dict[str, Any] | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Devolve (argumentos convertidos, erros, obrigatórios ausentes)."""
    props: dict[str, Any] = schema.get("properties") or {}
    required: list[str] = list(schema.get("required") or [])
    out: dict[str, Any] = {}
    errors: list[str] = []
    for name, value in (args or {}).items():
        prop = props.get(name)
        if prop is None:
            if schema.get("additionalProperties") is False:
                errors.append(f"campo desconhecido '{name}'")
            else:
                out[name] = value
            continue
        coerced, err = _coerce(value, prop)
        if err:
            errors.append(f"{name}: {err}")
        elif coerced is not None:
            out[name] = coerced
    missing = [n for n in required if n not in out or out[n] in ("", None)]
    return out, errors, missing


def describe_fields(schema: dict[str, Any], names: list[str]) -> list[str]:
    props = schema.get("properties") or {}
    labels = []
    for name in names:
        prop = props.get(name) or {}
        label = (prop.get("description") or prop.get("title") or name.replace("_", " ")).strip()
        labels.append(label[0].lower() + label[1:] if label else name)
    return labels


_ID_RES = (
    re.compile(r"\b([A-Z]{2,6}-\d{1,8})\b"),
    re.compile(r"#\s?(\d{1,10})\b"),
    re.compile(r"\b(?:chamado|protocolo|ticket|n[ºo°]?)\s*(\d{1,10})\b", re.IGNORECASE),
)


def extract_arguments(
    schema: dict[str, Any],
    text: str,
    *,
    focus: str | None = None,
    ignore: frozenset[str] | set[str] = frozenset(),
) -> dict[str, Any]:
    """Preenche argumentos a partir de texto livre (decisor offline).

    Números e opções saem de ``text`` (a conversa recente). Campos de texto
    livre (título, descrição) preferem ``focus`` (a última mensagem) e só são
    preenchidos se o texto disser algo além do próprio pedido — "quero abrir
    um chamado" não é uma descrição de problema; as palavras em ``ignore``
    (stems do nome/descrição da tool) não contam.
    """

    def informative(candidate: str | None) -> bool:
        return bool(candidate) and len(keywords(candidate) - set(ignore)) >= 2

    free_text = next((c for c in (focus, text) if informative(c)), None)
    props: dict[str, Any] = schema.get("properties") or {}
    folded = fold(text)
    mentions = find_mentions(text)
    used: set[int] = set()
    out: dict[str, Any] = {}

    def take(kinds: tuple[str, ...]) -> _Mention | None:
        for kind in kinds:
            for i, m in enumerate(mentions):
                if i not in used and m.kind == kind:
                    used.add(i)
                    return m
        return None

    # enums primeiro (não consomem números)
    for name, prop in props.items():
        enum = _enum(prop)
        if not enum:
            continue
        for option in enum:
            if isinstance(option, str) and re.search(rf"\b{re.escape(fold(option))}\b", folded):
                out[name] = option
                break

    ordered = sorted(
        (n for n in props if n not in out),
        key=lambda n: {"percent": 0, "term": 1, "money": 2}.get(_role(n, props[n]) or "", 3),
    )
    for name in ordered:
        prop = props[name]
        types = _types(prop)
        role = _role(name, prop)
        if _enum(prop):
            continue
        if "number" in types or "integer" in types:
            if role == "percent":
                m = take(("percent",))
            elif role == "term":
                m = take(("months", "years"))
            elif role == "money":
                m = take(("money",))
                if m is None:
                    plain = [
                        (i, x)
                        for i, x in enumerate(mentions)
                        if i not in used and x.kind == "plain"
                    ]
                    if plain:
                        i, m = max(plain, key=lambda p: p[1].value)
                        used.add(i)
            else:
                m = take(("plain", "money"))
            if m is None:
                continue
            value = m.value * 12 if (role == "term" and m.kind == "years") else m.value
            out[name] = int(value) if ("integer" in types and float(value).is_integer()) else value
        elif "string" in types or not types:
            if role == "id":
                for regex in _ID_RES:
                    found = regex.search(text)
                    if found:
                        out[name] = found.group(1)
                        break
            elif free_text is None:
                continue
            elif role == "title":
                out[name] = truncate(free_text, 80)
            elif role == "text" or (role is None and name in (schema.get("required") or [])):
                out[name] = free_text.strip()
    return out
