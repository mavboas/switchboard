"""Utilitários de texto compartilhados pelo RAG offline e pelo decisor heurístico.

Tudo aqui é determinístico e sem dependências externas: normalização
(minúsculas, sem acento), tokenização e um "stemming" leve por prefixo, bom o
suficiente para casar "simulação"/"simular"/"simule" ou "chamado"/"chamados"
sem precisar de um stemmer de verdade.
"""

from __future__ import annotations

import re
import unicodedata

_WORD_RE = re.compile(r"[a-z0-9]+")

STEM_SIZE = 5

STOPWORDS: frozenset[str] = frozenset(
    """
    a o as os um uma uns umas de do da dos das em no na nos nas num numa por pelo pela
    pelos pelas para pra pro com sem sob sobre entre ate e ou mas que se ao aos eu tu
    ele ela nos vos eles elas me te lhe mim meu minha meus minhas seu sua seus suas
    nosso nossa isso isto esse essa este esta aquele aquela qual quais quem como
    quando onde porque pois ja nao sim mais menos muito muita muitos muitas tem ter
    tenho ser estar estou esta estao foi era sao ha voce voces gostaria quero queria
    preciso precisava poderia pode posso podem favor ola oi bom boa dia tarde noite
    obrigado obrigada tudo bem entao agora ai la aqui ali cada todo toda todos todas
    algum alguma alguns algumas outro outra sera seria fazer faz faca fale diga
    the an of to in on for and or is are be with me my i you it this that what how
    can please do does
    """.split()
)


def fold(text: str) -> str:
    """Minúsculas e sem acentos ("Simulação" -> "simulacao")."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).lower()


def tokenize(text: str) -> list[str]:
    """Palavras e números, já normalizados por :func:`fold`."""
    return _WORD_RE.findall(fold(text))


def stem(token: str) -> str:
    """Stemming leve por prefixo; números ficam intactos."""
    if token.isdigit() or len(token) <= STEM_SIZE:
        return token
    return token[:STEM_SIZE]


def content_tokens(text: str) -> list[str]:
    """Tokens relevantes (sem stopwords e sem letras soltas)."""
    return [t for t in tokenize(text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def keywords(text: str) -> set[str]:
    """Conjunto de stems relevantes de um texto."""
    return {stem(t) for t in content_tokens(text)}


def truncate(text: str, limit: int, *, suffix: str = "…") -> str:
    """Corta ``text`` em ``limit`` caracteres sem quebrar palavra no meio."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",.;:")
    return cut + suffix
