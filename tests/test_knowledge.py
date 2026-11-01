"""Busca de contexto: documentos em pedaços com embedding, em RAM e no Postgres com pgvector."""

import pytest

from sentinel.llm.embeddings import FakeEmbedder
from sentinel.memory import (
    Chunk,
    InMemoryKnowledgeStore,
    chunk_markdown,
    index_files,
)

DOC = """# Guia

Introdução geral do guia.

## Queda de conversão

Quando a conversão cai no Android depois de um deploy, confira a versão do app.

## Estoque

Produto sem estoque tem conversão zerada. Avise o time de compras.

### Detalhes

Repor o estoque do watch resolve a queda.
"""


@pytest.fixture(params=["ram", pytest.param("pg", marks=pytest.mark.pg)])
def kstore(request):
    embedder = FakeEmbedder(dim=64)
    if request.param == "ram":
        yield InMemoryKnowledgeStore(embedder)
    else:
        yield request.getfixturevalue("pg_memory").knowledge(embedder)


# --- chunk_markdown ---------------------------------------------------------------------------


def test_chunks_follow_the_headings_and_keep_the_heading_path():
    chunks = chunk_markdown(DOC, "guia.md")
    headings = [c.heading for c in chunks]
    assert headings == [
        "Guia",
        "Guia > Queda de conversão",
        "Guia > Estoque",
        "Guia > Estoque > Detalhes",
    ]
    assert all(c.source == "guia.md" for c in chunks)
    assert "versão do app" in chunks[1].content
    assert "Introdução" in chunks[0].content


def test_a_hash_symbol_inside_a_code_block_is_not_a_heading():
    text = "# Título\n\n```bash\n# comentário do shell\nsentinel detect\n```\n"
    [chunk] = chunk_markdown(text, "x.md")
    assert chunk.heading == "Título"
    assert "# comentário do shell" in chunk.content


def test_long_sections_are_split_without_losing_text_and_empty_ones_are_dropped():
    paragraph = "palavra " * 60  # ~480 caracteres
    text = "# Longo\n\n" + "\n\n".join([paragraph] * 6) + "\n\n# Vazio\n"
    chunks = chunk_markdown(text, "x.md", max_chars=1000)
    assert len(chunks) >= 3
    assert all(len(c.content) <= 1000 for c in chunks)
    assert {c.heading for c in chunks} == {"Longo"}
    assert sum(c.content.count("palavra") for c in chunks) == 360


def test_one_giant_paragraph_is_hard_split():
    chunks = chunk_markdown("# T\n\n" + "x" * 2500, "x.md", max_chars=1000)
    assert [len(c.content) for c in chunks] == [1000, 1000, 500]


def test_chunk_hash_is_stable_and_depends_on_source_heading_and_content():
    a = Chunk(source="a.md", heading="H", content="texto")
    assert a.content_hash == Chunk(source="a.md", heading="H", content="texto").content_hash
    assert a.content_hash != Chunk(source="b.md", heading="H", content="texto").content_hash
    assert a.content_hash != Chunk(source="a.md", heading="H", content="outro").content_hash


# --- contrato dos stores ----------------------------------------------------------------------


def test_indexing_is_idempotent(kstore):
    chunks = chunk_markdown(DOC, "guia.md")
    first = kstore.index(chunks)
    second = kstore.index(chunks)
    assert (first.added, first.kept, first.removed) == (4, 0, 0)
    assert (second.added, second.kept, second.removed) == (0, 4, 0)
    assert kstore.count() == 4


def test_reindexing_an_edited_source_replaces_its_old_chunks(kstore):
    kstore.index(chunk_markdown(DOC, "guia.md"))
    kstore.index(chunk_markdown("# Outro\n\nOutra fonte.", "outro.md"))
    edited = DOC.replace("Repor o estoque do watch resolve a queda.", "Peça reposição.")
    report = kstore.index(chunk_markdown(edited, "guia.md"))
    assert (report.added, report.kept, report.removed) == (1, 3, 1)
    assert kstore.count() == 5  # a outra fonte não foi tocada
    hits = kstore.search("watch", k=5, min_similarity=0.0)
    assert not any("Repor o estoque" in h.chunk.content for h in hits)


def test_search_finds_the_relevant_chunk_first(kstore):
    kstore.index(chunk_markdown(DOC, "guia.md"))
    hits = kstore.search("conversão caiu no android depois do deploy", k=2, min_similarity=0.0)
    assert hits[0].chunk.heading == "Guia > Queda de conversão"
    assert len(hits) == 2
    assert hits[0].similarity >= hits[1].similarity
    assert kstore.search("conversão caiu no android", k=4, min_similarity=0.999) == []


def test_search_on_an_empty_store(kstore):
    assert kstore.search("qualquer coisa", k=3, min_similarity=0.0) == []
    assert kstore.count() == 0


def test_chunks_with_secrets_are_never_indexed(kstore):
    leaked = "postgresql://postgres:" + "pw" + "8q8q@localhost/db"
    chunks = [
        Chunk(source="a.md", heading="Ok", content="Conteúdo seguro."),
        Chunk(source="a.md", heading="Ruim", content=f"use {leaked} para conectar"),
    ]
    report = kstore.index(chunks)
    assert report.added == 1
    assert report.skipped == ["a.md > Ruim (possível segredo)"]
    assert kstore.count() == 1
    assert "8q8q" not in repr(report)


# --- index_files ------------------------------------------------------------------------------


def test_index_files_reads_markdown_and_reports_what_it_skipped(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guia.md").write_text(DOC, encoding="utf-8")
    (tmp_path / "notas.txt").write_text("Texto simples sobre estoque.", encoding="utf-8")
    (tmp_path / "codigo.py").write_text("print('x')", encoding="utf-8")
    (tmp_path / ".env.local").write_text("TOKEN=abc", encoding="utf-8")
    (tmp_path / "enorme.md").write_text("x" * 2_000_000, encoding="utf-8")
    (tmp_path / "binario.md").write_bytes(b"\xff\xfe\x00\x01")

    store = InMemoryKnowledgeStore(FakeEmbedder(dim=64))
    report = index_files(
        store,
        [
            tmp_path / "docs" / "guia.md",
            tmp_path / "notas.txt",
            tmp_path / "codigo.py",
            tmp_path / ".env.local",
            tmp_path / "enorme.md",
            tmp_path / "binario.md",
            tmp_path / "nao-existe.md",
        ],
        root=tmp_path,
    )
    assert report.added == 5  # 4 do guia + 1 das notas
    assert {item.split(" (")[0] for item in report.skipped} == {
        "codigo.py",
        ".env.local",
        "enorme.md",
        "binario.md",
        "nao-existe.md",
    }
    sources = {h.chunk.source for h in store.search("estoque", k=10, min_similarity=0.0)}
    assert sources == {"docs/guia.md", "notas.txt"}  # caminho relativo, nunca absoluto


def test_index_files_never_stores_absolute_or_personal_paths(tmp_path):
    outside = tmp_path / "fora"
    outside.mkdir()
    file = outside / "x.md"
    file.write_text("# A\n\ntexto", encoding="utf-8")
    store = InMemoryKnowledgeStore(FakeEmbedder(dim=64))
    index_files(store, [file], root=tmp_path / "projeto")
    [hit] = store.search("texto", k=1, min_similarity=0.0)
    assert hit.chunk.source == "x.md"
