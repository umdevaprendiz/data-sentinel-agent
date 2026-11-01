"""Memória do agente: incidentes passados e contexto (documentos), com busca por similaridade."""

from sentinel.memory.incidents import (
    ACTIVE_STATES,
    STATES,
    DecisionStats,
    Incident,
    IncidentStore,
    InMemoryIncidentStore,
    MemoryStoreError,
    SimilarIncident,
    State,
    cause_key,
    signature_text,
)
from sentinel.memory.knowledge import (
    Chunk,
    IndexReport,
    InMemoryKnowledgeStore,
    KnowledgeHit,
    KnowledgeStore,
    chunk_markdown,
    index_files,
)

__all__ = [
    "ACTIVE_STATES",
    "STATES",
    "Chunk",
    "DecisionStats",
    "InMemoryIncidentStore",
    "InMemoryKnowledgeStore",
    "Incident",
    "IncidentStore",
    "IndexReport",
    "KnowledgeHit",
    "KnowledgeStore",
    "MemoryStoreError",
    "SimilarIncident",
    "State",
    "cause_key",
    "chunk_markdown",
    "index_files",
    "signature_text",
]
