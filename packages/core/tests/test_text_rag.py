from __future__ import annotations

import pytest

from switchboard.config import KnowledgeBaseSpec
from switchboard.embeddings import HashingEmbedder, cosine
from switchboard.errors import IngestError
from switchboard.rag import MemoryRetriever, extract_text, split_text
from switchboard.text import fold, keywords, stem, tokenize, truncate


def test_fold_tokenize_keywords():
    assert fold("Simulação de Crédito") == "simulacao de credito"
    assert tokenize("R$ 1.500,00 em 12x!") == ["r", "1", "500", "00", "em", "12x"]
    assert keywords("Quero simular um empréstimo") == {"simul", "empre"}
    assert stem("chamados") == stem("chamado") == "chama"
    assert stem("2024") == "2024"


def test_truncate_keeps_words():
    assert truncate("uma frase bem curta", 50) == "uma frase bem curta"
    out = truncate("uma frase um pouco mais comprida que o limite", 20)
    assert out.endswith("…") and len(out) <= 21 and " " in out


async def test_hashing_embedder_is_deterministic_and_normalized():
    emb = HashingEmbedder(256)
    [a1], [a2] = (
        await emb.embed(["horário de atendimento"]),
        await emb.embed(["horário de atendimento"]),
    )
    assert a1 == a2
    assert abs(sum(x * x for x in a1) - 1.0) < 1e-9
    [b] = await emb.embed(["horários do atendimento aos sábados"])
    [c] = await emb.embed(["taxa de juros do financiamento"])
    assert cosine(a1, b) > cosine(a1, c)


def test_split_text_sections_and_sizes():
    doc = "# Manual\n\nIntro curta.\n\n## Horários\n\n" + ("Atendemos de segunda a sexta. " * 40)
    chunks = split_text(doc, chunk_size=300, overlap=60)
    assert chunks[0].section == "Manual"
    assert all(c.section == "Manual > Horários" for c in chunks[1:])
    assert all(len(c.content) <= 300 for c in chunks)
    # sobreposição: o fim de um trecho reaparece no começo do seguinte
    assert chunks[2].content[:20] in chunks[1].content


def test_split_text_rejects_tiny_chunk_size():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10)


def test_extract_text_formats():
    assert extract_text("a.md", "# Título\nconteúdo".encode()) == "# Título\nconteúdo"
    html = b"<html><style>x{}</style><body><h1>Oi</h1><p>mundo &amp; cia</p></body></html>"
    assert "mundo & cia" in extract_text("a.html", html)
    assert "x{}" not in extract_text("a.html", html)
    assert extract_text("a.txt", "acentuação".encode("cp1252")) == "acentuação"
    with pytest.raises(IngestError):
        extract_text("a.exe", b"MZ")
    with pytest.raises(IngestError):
        extract_text("vazio.txt", b"   ")


async def test_memory_retriever_ranks_relevant_chunk_first(tmp_path):
    (tmp_path / "faq.md").write_text(
        "# FAQ\n\n## Horários\n\nAtendemos aos sábados das 9h às 14h.\n\n"
        "## Pagamentos\n\nO boleto vence todo dia 10 e pode ser pago no aplicativo.\n",
        encoding="utf-8",
    )
    retriever = MemoryRetriever()
    spec = KnowledgeBaseSpec(name="faq", paths=[str(tmp_path)])
    retriever.register(spec, HashingEmbedder())
    assert await retriever.load_paths("faq") == 2
    hits = await retriever.search("vocês abrem no sábado?", ["faq"], top_k=2, min_score=0.0)
    assert hits[0].section == "FAQ > Horários"
    assert hits[0].document == "FAQ"  # título vem do H1 do Markdown
    assert await retriever.search("abrem no sábado?", ["outra"], top_k=2, min_score=0.0) == []


def test_guess_title():
    from switchboard.rag import guess_title

    assert guess_title("x.md", "intro\n# Política de crédito\n\ntexto") == "Política de crédito"
    assert guess_title("manual_de-uso.txt", "sem título markdown") == "Manual de uso"
