"""Escolhe o backend da memória: Postgres com pgvector se houver `MEMORY_DSN`, senão RAM."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sentinel.config import Settings
from sentinel.llm.embeddings import Embedder, OllamaEmbedder
from sentinel.memory.incidents import IncidentStore, InMemoryIncidentStore
from sentinel.memory.knowledge import InMemoryKnowledgeStore, KnowledgeStore

if TYPE_CHECKING:
    from sentinel.memory.pg import PgMemory


@dataclass
class Memory:
    incidents: IncidentStore
    knowledge: KnowledgeStore
    backend: str  # "postgres" | "ram"
    pg: "PgMemory | None" = None

    def close(self) -> None:
        if self.pg is not None:
            self.pg.close()


def open_memory(settings: Settings, embedder: Embedder | None = None) -> Memory:
    """Abre a memória. `embedder` só é passado nos testes; no uso real vem do Ollama."""
    embedder = embedder or OllamaEmbedder(
        settings.ollama_url, settings.embed_model, settings.embed_dim
    )
    dsn = settings.memory_dsn.get_secret_value()
    if not dsn:
        return Memory(
            InMemoryIncidentStore(embedder), InMemoryKnowledgeStore(embedder), backend="ram"
        )
    from sentinel.memory.pg import PgMemory  # import tardio: RAM não precisa do psycopg

    pg = PgMemory.connect(dsn, dim=embedder.dim)
    return Memory(pg.incidents(embedder), pg.knowledge(embedder), backend="postgres", pg=pg)
