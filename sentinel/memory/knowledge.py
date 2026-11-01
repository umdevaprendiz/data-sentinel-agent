"""Busca de contexto: documentos em pedaços com embedding (RAM aqui, Postgres em `pg.py`).

Regras de segurança: só entra texto simples (`.md`/`.txt`), nunca arquivo `.env`, e nenhum pedaço
com segredo ou caminho pessoal é indexado. A fonte guardada é sempre um caminho relativo.
"""

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

from sentinel.llm.embeddings import Embedder
from sentinel.memory.incidents import check_vectors, normalized
from sentinel.security import clean_text, find_secrets

MAX_CHUNK = 1200
MAX_FILE_BYTES = 1_000_000
ALLOWED_SUFFIXES = {".md", ".txt"}


class Chunk(BaseModel):
    source: str  # caminho relativo ao projeto
    heading: str  # "Título > Subtítulo"
    content: str

    @property
    def content_hash(self) -> str:
        raw = f"{self.source}\0{self.heading}\0{self.content}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def embedding_text(self) -> str:
        return f"{self.heading}\n{self.content}"

    @property
    def label(self) -> str:
        return f"{self.source} > {self.heading}" if self.heading else self.source


class KnowledgeHit(BaseModel):
    chunk: Chunk
    similarity: float


class IndexReport(BaseModel):
    added: int = 0
    kept: int = 0
    removed: int = 0
    skipped: list[str] = Field(default_factory=list)  # "origem (motivo)": nunca o conteúdo

    def merge(self, other: "IndexReport") -> "IndexReport":
        return IndexReport(
            added=self.added + other.added,
            kept=self.kept + other.kept,
            removed=self.removed + other.removed,
            skipped=[*self.skipped, *other.skipped],
        )


class KnowledgeStore(Protocol):
    def index(self, chunks: list[Chunk]) -> IndexReport:
        """Indexa os pedaços; as fontes presentes são substituídas, as demais ficam como estão."""
        ...

    def search(self, query: str, k: int, min_similarity: float) -> list[KnowledgeHit]: ...

    def count(self) -> int: ...


# --- Pedaços ----------------------------------------------------------------------------------


def _split_text(text: str, limit: int) -> list[str]:
    """Junta parágrafos até o limite; um parágrafo maior que o limite é cortado."""
    pieces: list[str] = []
    current = ""
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        while len(paragraph) > limit:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(paragraph[:limit])
            paragraph = paragraph[limit:]
        joined = f"{current}\n\n{paragraph}" if current else paragraph
        if len(joined) > limit:
            pieces.append(current)
            current = paragraph
        else:
            current = joined
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(text: str, source: str, max_chars: int = MAX_CHUNK) -> list[Chunk]:
    """Um pedaço por seção (com o caminho dos títulos); seções longas são divididas."""
    base = Path(source).name
    stack: list[str] = []
    sections: list[tuple[str, list[str]]] = [(base, [])]
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        title = line.lstrip("#")
        level = len(line) - len(title)
        if not in_code and 1 <= level <= 6 and title.startswith(" "):
            stack = [*stack[: level - 1], title.strip()]
            sections.append((" > ".join(stack), []))
        else:
            sections[-1][1].append(line)

    chunks = []
    for heading, lines in sections:
        for piece in _split_text("\n".join(lines), max_chars):
            chunks.append(Chunk(source=source, heading=heading, content=piece))
    return chunks


def screen(chunks: list[Chunk]) -> tuple[list[Chunk], list[str]]:
    """Separa os pedaços seguros dos que têm segredo ou caminho pessoal (só os rótulos voltam)."""
    safe, skipped = [], []
    for chunk in chunks:
        if find_secrets(f"{chunk.heading}\n{chunk.content}"):
            skipped.append(f"{chunk.label} (possível segredo)")
        else:
            safe.append(chunk)
    return safe, skipped


class InMemoryKnowledgeStore:
    """Mesmo contrato do Postgres, só em RAM: testes e uso sem `MEMORY_DSN` (não persiste)."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._rows: dict[str, tuple[Chunk, np.ndarray]] = {}

    def index(self, chunks: list[Chunk]) -> IndexReport:
        safe, skipped = screen(chunks)
        report = IndexReport(skipped=skipped)
        wanted = {c.content_hash: c for c in safe}
        sources = {c.source for c in chunks}

        stale = [h for h, (c, _) in self._rows.items() if c.source in sources and h not in wanted]
        for digest in stale:
            del self._rows[digest]
        report.removed = len(stale)

        new = [c for h, c in wanted.items() if h not in self._rows]
        report.kept = len(wanted) - len(new)
        if new:
            vectors = self._embedder.embed([c.embedding_text() for c in new])
            check_vectors(self._embedder, vectors, len(new))
            for chunk, vector in zip(new, vectors, strict=True):
                self._rows[chunk.content_hash] = (chunk, normalized(vector))
        report.added = len(new)
        return report

    def search(self, query: str, k: int, min_similarity: float) -> list[KnowledgeHit]:
        if not self._rows:
            return []
        vectors = self._embedder.embed([query])
        check_vectors(self._embedder, vectors, 1)
        vector = normalized(vectors[0])
        hits = [
            KnowledgeHit(chunk=chunk, similarity=float(np.clip(row @ vector, 0.0, 1.0)))
            for chunk, row in self._rows.values()
        ]
        hits = [h for h in hits if h.similarity >= min_similarity]
        return sorted(hits, key=lambda h: h.similarity, reverse=True)[:k]

    def count(self) -> int:
        return len(self._rows)


# --- Arquivos ---------------------------------------------------------------------------------


def _label(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def index_files(
    store: KnowledgeStore, paths: Iterable[Path], root: Path | None = None
) -> IndexReport:
    """Indexa arquivos `.md`/`.txt`. O que não puder entrar volta em `skipped`, com o motivo."""
    report = IndexReport()
    for path in paths:
        label = _label(path, root)
        reason = None
        text = ""
        if path.name.startswith(".env"):
            reason = "arquivo privado"
        elif path.suffix.lower() not in ALLOWED_SUFFIXES:
            reason = "extensão não suportada"
        elif not path.is_file():
            reason = "não encontrado"
        elif path.stat().st_size > MAX_FILE_BYTES:
            reason = "grande demais"
        else:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                reason = "não é texto UTF-8"
        if reason:
            report.skipped.append(f"{clean_text(label, 120)} ({reason})")
            continue
        report = report.merge(store.index(chunk_markdown(text, label)))
    return report
