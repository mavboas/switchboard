"""RAG: quebra em trechos, extração de texto e busca por similaridade."""

from .chunking import TextChunk, split_text
from .loaders import SUPPORTED_EXTENSIONS, extract_text, guess_title, iter_files
from .retriever import Hit, MemoryRetriever, Retriever, index_text

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "Hit",
    "MemoryRetriever",
    "Retriever",
    "TextChunk",
    "extract_text",
    "guess_title",
    "index_text",
    "iter_files",
    "split_text",
]
