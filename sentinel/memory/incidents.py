"""Memória de incidentes (seção 11 do CLAUDE.md): modelo, contrato e store em RAM."""

import re
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol, Self

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from sentinel.llm.embeddings import Embedder
from sentinel.security import clean_text, redact

State = Literal["open", "pending", "approved", "ignored", "expired", "resolved"]
NAME_PATTERN = r"^[a-z][a-z0-9_]{0,40}$"  # métrica ou dimensão: o agente as descobre nos dados
MAX_VALUE = 80
STATES: tuple[str, ...] = ("open", "pending", "approved", "ignored", "expired", "resolved")
ACTIVE_STATES: tuple[str, ...] = ("open", "pending")  # contam para a deduplicação
MAX_CAUSES = 2
MAX_SUMMARY = 2000


class MemoryStoreError(RuntimeError):
    """Falha da memória, com mensagem sem dados sensíveis (nunca DSN nem texto do banco)."""


def utcnow() -> datetime:
    return datetime.now(UTC)


def cause_key(causes: list[dict[str, str]]) -> str:
    """Chave canônica das causas: não depende da ordem das causas nem das dimensões."""
    parts = (",".join(f"{k}={v}" for k, v in sorted(cause.items())) for cause in causes)
    return "|".join(sorted(parts))


def signature_text(metric: str, direction: str, causes: list[dict[str, str]]) -> str:
    """Texto curto que descreve o incidente (é o que mais pesa no embedding)."""
    kind = "queda" if direction == "drop" else "pico"
    where = " | ".join(" ".join(f"{k}={v}" for k, v in c.items()) for c in causes)
    return f"{kind} em {metric}; causa: {where or 'sem causa identificada'}"


class Incident(BaseModel):
    incident_id: str = Field(min_length=1, max_length=64)
    metric: str = Field(pattern=NAME_PATTERN)
    direction: Literal["drop", "spike"]
    status: Literal["identified", "inconclusive"]
    causes: list[dict[str, str]] = Field(default_factory=list)  # 0 a 2 causas confirmadas
    summary: str  # relatório em texto (limpo e sem segredos antes de ser guardado)
    state: State = "open"
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("causes")
    @classmethod
    def _valid_causes(cls, value: list[dict[str, str]]) -> list[dict[str, str]]:
        if len(value) > MAX_CAUSES:
            raise ValueError(f"no máximo {MAX_CAUSES} causas")
        for cause in value:
            for name, option in cause.items():
                if not re.match(NAME_PATTERN, name):
                    raise ValueError(f"nome de dimensão inválido: {name!r}")
                if not option or len(option) > MAX_VALUE or clean_text(option) != option:
                    raise ValueError(f"valor inválido para {name}")
        return value

    @model_validator(mode="after")
    def _aware(self) -> Self:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at precisa ter fuso horário")
        return self

    @property
    def signature(self) -> str:
        return signature_text(self.metric, self.direction, self.causes)

    def embedding_text(self) -> str:
        return f"{self.signature}. {self.summary}"

    def sanitized(self) -> "Incident":
        """Cópia com o resumo limpo (sem controle de terminal) e sem segredos."""
        return self.model_copy(update={"summary": clean_text(redact(self.summary), MAX_SUMMARY)})


class SimilarIncident(BaseModel):
    incident: Incident
    similarity: float  # cosseno, 0..1


class DecisionStats(BaseModel):
    """Decisões humanas (Aprovar/Ignorar): alimentam a precisão histórica da policy."""

    approved: int = 0
    ignored: int = 0

    @property
    def total(self) -> int:
        return self.approved + self.ignored

    @property
    def precision(self) -> float | None:
        return self.approved / self.total if self.total else None


class IncidentStore(Protocol):
    def add(self, incident: Incident) -> None: ...

    def get(self, incident_id: str) -> Incident | None: ...

    def set_state(self, incident_id: str, state: State) -> None: ...

    def find_open(
        self,
        metric: str,
        direction: str,
        causes: list[dict[str, str]],
        within_hours: float,
        now: datetime | None = None,
    ) -> Incident | None:
        """Incidente aberto ou pendente com a mesma assinatura dentro do cooldown."""
        ...

    def similar(
        self, text: str, k: int, min_similarity: float, exclude: str | None = None
    ) -> list[SimilarIncident]: ...

    def decision_stats(self) -> DecisionStats: ...


def check_vectors(embedder: Embedder, vectors: list[list[float]], expected: int) -> None:
    """O embedder precisa devolver um vetor por texto e na dimensão configurada."""
    if len(vectors) != expected or any(len(v) != embedder.dim for v in vectors):
        raise MemoryStoreError(f"o embedder devolveu vetores fora da dimensão {embedder.dim}")


def validate_state(state: str) -> None:
    if state not in STATES:
        raise MemoryStoreError(f"estado inválido: {state!r}")


def normalized(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(array)
    return array / norm if norm else array


class InMemoryIncidentStore:
    """Mesmo contrato do Postgres, só em RAM: testes e uso sem `MEMORY_DSN` (não persiste)."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._rows: dict[str, tuple[Incident, np.ndarray]] = {}

    def add(self, incident: Incident) -> None:
        clean = incident.sanitized()
        if clean.incident_id in self._rows:
            raise MemoryStoreError("o incidente já existe")
        vectors = self._embedder.embed([clean.embedding_text()])
        check_vectors(self._embedder, vectors, 1)
        self._rows[clean.incident_id] = (clean, normalized(vectors[0]))

    def get(self, incident_id: str) -> Incident | None:
        row = self._rows.get(incident_id)
        return row[0] if row else None

    def set_state(self, incident_id: str, state: State) -> None:
        validate_state(state)
        row = self._rows.get(incident_id)
        if row is None:
            raise MemoryStoreError("o incidente não existe")
        self._rows[incident_id] = (row[0].model_copy(update={"state": state}), row[1])

    def find_open(
        self,
        metric: str,
        direction: str,
        causes: list[dict[str, str]],
        within_hours: float,
        now: datetime | None = None,
    ) -> Incident | None:
        now = now or utcnow()
        since = now - timedelta(hours=within_hours)
        key = cause_key(causes)
        matches = [
            incident
            for incident, _ in self._rows.values()
            if incident.metric == metric
            and incident.direction == direction
            and cause_key(incident.causes) == key
            and incident.state in ACTIVE_STATES
            and since <= incident.created_at <= now
        ]
        return max(matches, key=lambda i: i.created_at, default=None)

    def similar(
        self, text: str, k: int, min_similarity: float, exclude: str | None = None
    ) -> list[SimilarIncident]:
        if not self._rows:
            return []
        vectors = self._embedder.embed([text])
        check_vectors(self._embedder, vectors, 1)
        query = normalized(vectors[0])
        scored = [
            SimilarIncident(incident=incident, similarity=float(np.clip(vector @ query, 0.0, 1.0)))
            for incident_id, (incident, vector) in self._rows.items()
            if incident_id != exclude
        ]
        scored = [hit for hit in scored if hit.similarity >= min_similarity]
        return sorted(scored, key=lambda hit: hit.similarity, reverse=True)[:k]

    def decision_stats(self) -> DecisionStats:
        states = [incident.state for incident, _ in self._rows.values()]
        return DecisionStats(approved=states.count("approved"), ignored=states.count("ignored"))
