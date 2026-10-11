"""O playbook: a parte "como usar" de um plugin, que o roteador junta ao prompt do LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from ..config import split_qualified
from ..text import STOPWORDS, fold


def broad_trigger(trigger: str) -> bool:
    """Gatilho que casaria com quase todo pedido ("que", "para", "o"): não conta pontos."""
    words = fold(trigger).split()
    return not words or len("".join(words)) < 3 or all(w in STOPWORDS for w in words)


def _word_pattern(word: str) -> str:
    """Uma palavra (sem acento) e os plurais regulares dela em português e inglês."""
    if word.endswith("ao"):  # simulacao -> simulacoes, pao -> paes, cidadao -> cidadaos
        return re.escape(word[:-2]) + "(?:ao|oes|aes|aos)"
    if word.endswith(("al", "el", "ol", "ul")):  # anual -> anuais, papel -> papeis
        return re.escape(word[:-1]) + "(?:l|is)"
    if word.endswith("il"):  # fuzil -> fuzis
        return re.escape(word[:-1]) + "(?:l|s)"
    if word.endswith("m"):  # item -> itens
        return re.escape(word[:-1]) + "(?:m|ns)"
    if word.endswith(("r", "z")):  # valor -> valores, taxa -> taxas
        return re.escape(word) + "(?:es)?"
    return re.escape(word) + "(?:s|es)?"


@lru_cache(maxsize=4096)
def trigger_pattern(trigger: str) -> re.Pattern[str] | None:
    """Expressão que acha o gatilho num texto já dobrado (sem acento, minúsculo), aceitando
    plural em cada palavra ("limite de crédito" casa "limites de crédito")."""
    if broad_trigger(trigger):
        return None
    words = fold(trigger).split()
    body = r"\s+".join(_word_pattern(w) for w in words)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


@dataclass(frozen=True)
class Playbook:
    """Uma skill de plugin (``skills/<nome>/SKILL.md``) pronta para o roteamento.

    ``applies_to`` traz os nomes *qualificados* (``plugin.componente``) dos conectores e
    agentes a que o playbook se refere: se nenhum deles está no ar, o playbook não entra.
    Vazio quer dizer "vale para o plugin todo".
    """

    plugin: str
    version: str
    skill: str
    description: str
    body: str
    applies_to: tuple[str, ...] = ()
    triggers: tuple[str, ...] = ()
    always: bool = False

    @property
    def key(self) -> str:
        """Identificador estável do playbook (usado em spans e no uso por plugin)."""
        return f"{self.plugin}/{self.skill}"

    def covers(self, owner: str) -> bool:
        """O playbook orienta o uso deste conector/agente (nome qualificado)?"""
        if self.applies_to:
            return owner in self.applies_to
        return split_qualified(owner)[0] == self.plugin
