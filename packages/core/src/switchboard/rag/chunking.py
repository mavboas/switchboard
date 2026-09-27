"""Quebra de documentos em trechos (chunks) para indexação.

Estratégia: parágrafos são agrupados até ``chunk_size`` caracteres; parágrafos
grandes são divididos por frases e, em último caso, por palavras. Cada trecho
novo recebe o final do anterior (``overlap``) para não perder contexto na
fronteira. Títulos Markdown viram a "seção" do trecho, que também entra no
texto indexado e melhora a busca; o primeiro ``# Título`` do arquivo é o
título do documento, não uma seção.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+")


@dataclass(frozen=True)
class TextChunk:
    content: str
    section: str | None = None


def _split_long(text: str, size: int) -> list[str]:
    """Divide um bloco maior que ``size`` por frases e, se preciso, por palavras."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > size:
            if current:
                pieces.append(current)
                current = ""
            words, buf = sentence.split(), ""
            for word in words:
                if buf and len(buf) + 1 + len(word) > size:
                    pieces.append(buf)
                    buf = word
                else:
                    buf = f"{buf} {word}".strip()
            if buf:
                pieces.append(buf)
            continue
        if current and len(current) + 1 + len(sentence) > size:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[TextChunk]:
    if chunk_size < 50:
        raise ValueError("chunk_size muito pequeno")
    overlap = max(0, min(overlap, chunk_size // 2))
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1) blocos (parágrafos) com a seção Markdown vigente
    blocks: list[tuple[str | None, str]] = []
    headings: list[tuple[int, str]] = []
    paragraph: list[str] = []

    def section_name() -> str | None:
        return " > ".join(h for _, h in headings) if headings else None

    def flush() -> None:
        if paragraph:
            body = " ".join(line.strip() for line in paragraph).strip()
            if body:
                blocks.append((section_name(), body))
            paragraph.clear()

    first_heading = True
    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2).strip()
            if first_heading and level == 1:
                # o primeiro "# Título" é o título do documento, não uma seção
                first_heading = False
                continue
            first_heading = False
            headings[:] = [h for h in headings if h[0] < level] + [(level, title)]
        elif not line.strip():
            flush()
        else:
            paragraph.append(line)
    flush()

    # 2) agrupa blocos da mesma seção até o tamanho alvo
    chunks: list[TextChunk] = []
    current, current_section = "", None

    def emit() -> None:
        nonlocal current
        if current.strip():
            chunks.append(TextChunk(current.strip(), current_section))
        current = ""

    for section, body in blocks:
        parts = [body] if len(body) <= chunk_size else _split_long(body, chunk_size)
        for part in parts:
            if current and (
                section != current_section or len(current) + 2 + len(part) > chunk_size
            ):
                previous = current
                same_section = section == current_section
                emit()
                if same_section:
                    tail = _tail(previous, overlap)
                    if tail and len(tail) + 2 + len(part) <= chunk_size:
                        current = tail
            current_section = section
            current = f"{current}\n\n{part}" if current else part
    emit()
    return chunks
