"""Extração de texto de arquivos enviados para a base de conhecimento."""

from __future__ import annotations

import html
import io
import re
from pathlib import Path

from ..errors import IngestError

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".rst", ".log"}
HTML_EXTENSIONS = {".html", ".htm"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | HTML_EXTENSIONS | {".pdf"}

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_BLOCK_RE = re.compile(r"</?(p|div|br|li|h[1-6]|tr|section|article)[^>]*>", re.IGNORECASE)


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader  # import tardio: só quem usa PDF paga o custo

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf lança vários tipos para PDF corrompido
        raise IngestError(f"não consegui ler o PDF: {exc}") from exc
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        raise IngestError("o PDF não tem texto extraível (talvez seja imagem escaneada)")
    return text


def _html_text(raw: str) -> str:
    raw = _SCRIPT_RE.sub(" ", raw)
    raw = _BLOCK_RE.sub("\n\n", raw)
    return html.unescape(_TAG_RE.sub(" ", raw))


def extract_text(filename: str, data: bytes) -> str:
    """Devolve o texto de um arquivo pelo nome/extensão."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        text = _pdf_text(data)
    elif ext in HTML_EXTENSIONS:
        text = _html_text(_decode(data))
    elif ext in TEXT_EXTENSIONS or not ext:
        text = _decode(data)
    else:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise IngestError(f"formato {ext} não suportado (aceitos: {supported})")
    text = text.replace("\x00", "").strip()
    if not text:
        raise IngestError(f"{filename}: arquivo sem texto")
    return text


_H1_RE = re.compile(r"^\s{0,3}#\s+(.+?)\s*#*\s*$", re.MULTILINE)


def guess_title(filename: str, text: str) -> str:
    """Título do documento: o primeiro título ``# ...`` do Markdown ou o nome do arquivo."""
    match = _H1_RE.search(text[:4000])
    if match:
        return match.group(1).strip()[:300]
    stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    return (stem[:1].upper() + stem[1:]) if stem else "sem título"


def iter_files(paths: list[str], base_dir: Path | None = None) -> list[Path]:
    """Expande arquivos e diretórios (recursivo) com extensões suportadas."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        if path.is_dir():
            found.extend(
                sorted(
                    p
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                )
            )
        elif path.is_file():
            found.append(path)
        else:
            raise IngestError(f"caminho da base de conhecimento não encontrado: {path}")
    return found
