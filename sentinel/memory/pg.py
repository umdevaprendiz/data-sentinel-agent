"""Memória no Postgres com pgvector: incidentes e conhecimento, busca por cosseno.

Segurança: todo valor entra como parâmetro (nunca em texto SQL); só identificadores validados
(nome de schema) e a dimensão (inteiro) entram na DDL, via `psycopg.sql`. Nenhuma mensagem de erro
repete o DSN nem o texto do banco: só a família e o tipo da exceção.
"""

import functools
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import errors, sql
from psycopg.types.json import Jsonb

from sentinel.llm.embeddings import Embedder
from sentinel.memory.incidents import (
    ACTIVE_STATES,
    DecisionStats,
    Incident,
    MemoryStoreError,
    SimilarIncident,
    State,
    cause_key,
    check_vectors,
    utcnow,
    validate_state,
)
from sentinel.memory.knowledge import Chunk, IndexReport, KnowledgeHit, screen
from sentinel.security import register_secrets

SCHEMA_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,40}$")

_INCIDENT_COLUMNS = "incident_id, metric, direction, status, causes, summary, state, created_at"


def _dsn_passwords(dsn: str) -> list[str]:
    """Senha do DSN em URL (crua e decodificada), para a redação de logs e erros."""
    try:
        password = urlsplit(dsn).password
    except ValueError:
        return []
    return [password, unquote(password)] if password else []


def _vector(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _guard[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Falha do Postgres vira `MemoryStoreError` sem a mensagem original (pode ter dados)."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except psycopg.Error as error:
            raise MemoryStoreError(
                f"falha no Postgres da memória ({type(error).__name__})"
            ) from None

    return wrapper


def _incident_from(row: tuple[Any, ...]) -> Incident:
    incident_id, metric, direction, status, causes, summary, state, created_at = row
    return Incident(
        incident_id=incident_id,
        metric=metric,
        direction=direction,
        status=status,
        causes=causes,
        summary=summary,
        state=state,
        created_at=created_at,
    )


class PgMemory:
    """Conexão única com as duas tabelas (`incidents` e `knowledge_chunks`)."""

    def __init__(self, con: psycopg.Connection, dim: int, schema: str | None) -> None:
        self._con = con
        self.dim = dim
        self.schema = schema

    @classmethod
    def connect(
        cls, dsn: str, dim: int, schema: str | None = None, timeout_s: int = 10
    ) -> "PgMemory":
        """Conecta, garante a extensão `vector` e cria as tabelas se faltarem."""
        register_secrets(dsn, *_dsn_passwords(dsn))
        if dim < 1:
            raise MemoryStoreError("a dimensão do embedding precisa ser positiva")
        if schema is not None and not SCHEMA_NAME.match(schema):
            raise MemoryStoreError("nome de schema inválido")
        try:
            con = psycopg.connect(dsn, autocommit=True, connect_timeout=timeout_s)
        except psycopg.Error as error:
            raise MemoryStoreError(
                f"não consegui conectar ao Postgres ({type(error).__name__}); confira o MEMORY_DSN"
            ) from None
        try:
            return cls._prepare(con, dim, schema)
        except BaseException:
            con.close()
            raise

    @classmethod
    def _prepare(cls, con: psycopg.Connection, dim: int, schema: str | None) -> "PgMemory":
        try:
            con.execute("CREATE EXTENSION IF NOT EXISTS vector")  # em `public`, antes do schema
        except psycopg.Error as error:
            raise MemoryStoreError(
                "a extensão pgvector não está disponível neste servidor Postgres, ou o usuário "
                f"não pode criá-la ({type(error).__name__}); veja o README"
            ) from None
        try:
            if schema is not None:
                con.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
                )
                con.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
            register_vector(con)
            memory = cls(con, dim, schema)
            memory._create_tables()
            memory._check_dimension()
        except psycopg.Error as error:
            raise MemoryStoreError(
                f"falha ao preparar as tabelas da memória ({type(error).__name__})"
            ) from None
        return memory

    def _create_tables(self) -> None:
        vector = sql.SQL("vector({})").format(sql.Literal(self.dim))
        statements = [
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id text PRIMARY KEY,
                    metric text NOT NULL,
                    direction text NOT NULL CHECK (direction IN ('drop', 'spike')),
                    status text NOT NULL CHECK (status IN ('identified', 'inconclusive')),
                    causes jsonb NOT NULL,
                    cause_key text NOT NULL,
                    summary text NOT NULL,
                    state text NOT NULL CHECK (
                        state IN ('open', 'pending', 'approved', 'ignored', 'expired', 'resolved')
                    ),
                    created_at timestamptz NOT NULL,
                    embedding {} NOT NULL
                )
                """
            ).format(vector),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS incidents_signature_idx "
                "ON incidents (metric, direction, cause_key, created_at)"
            ),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS incidents_embedding_idx "
                "ON incidents USING hnsw (embedding vector_cosine_ops)"
            ),
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id bigserial PRIMARY KEY,
                    source text NOT NULL,
                    heading text NOT NULL,
                    content text NOT NULL,
                    content_hash text NOT NULL UNIQUE,
                    embedding {} NOT NULL
                )
                """
            ).format(vector),
            sql.SQL("CREATE INDEX IF NOT EXISTS knowledge_source_idx ON knowledge_chunks (source)"),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS knowledge_embedding_idx "
                "ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)"
            ),
        ]
        for statement in statements:
            self._con.execute(statement)

    def _check_dimension(self) -> None:
        for table in ("incidents", "knowledge_chunks"):
            row = self._con.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'",
                (table,),
            ).fetchone()
            if row is not None and row[0] != self.dim:
                raise MemoryStoreError(
                    f"a tabela {table} usa vetores de dimensão {row[0]}, mas EMBED_DIM é "
                    f"{self.dim}; use outro banco ou schema, ou recrie a tabela"
                )

    @_guard
    def status(self) -> dict[str, str]:
        """Dados de diagnóstico para `sentinel memory check` (nunca o DSN)."""
        database, user, version = self._con.execute(
            "SELECT current_database(), current_user, split_part(version(), ',', 1)"
        ).fetchone()  # type: ignore[misc]
        extension = self._con.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        return {
            "banco": database,
            "usuario": user,
            "servidor": version,
            "pgvector": extension[0] if extension else "ausente",
        }

    def incidents(self, embedder: Embedder) -> "PgIncidentStore":
        return PgIncidentStore(self._con, embedder)

    def knowledge(self, embedder: Embedder) -> "PgKnowledgeStore":
        return PgKnowledgeStore(self._con, embedder)

    @_guard
    def drop_schema(self) -> None:
        """Remove o schema criado por esta conexão (uso dos testes)."""
        if self.schema is None:
            raise MemoryStoreError("só é possível remover um schema nomeado")
        self._con.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(self.schema))
        )

    def close(self) -> None:
        self._con.close()


class PgIncidentStore:
    def __init__(self, con: psycopg.Connection, embedder: Embedder) -> None:
        self._con = con
        self._embedder = embedder

    @_guard
    def add(self, incident: Incident) -> None:
        clean = incident.sanitized()
        vectors = self._embedder.embed([clean.embedding_text()])
        check_vectors(self._embedder, vectors, 1)
        try:
            self._con.execute(
                "INSERT INTO incidents (incident_id, metric, direction, status, causes, cause_key, "
                "summary, state, created_at, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    clean.incident_id,
                    clean.metric,
                    clean.direction,
                    clean.status,
                    Jsonb(clean.causes),
                    cause_key(clean.causes),
                    clean.summary,
                    clean.state,
                    clean.created_at,
                    _vector(vectors[0]),
                ),
            )
        except errors.UniqueViolation:
            raise MemoryStoreError("o incidente já existe") from None

    @_guard
    def get(self, incident_id: str) -> Incident | None:
        row = self._con.execute(
            f"SELECT {_INCIDENT_COLUMNS} FROM incidents WHERE incident_id = %s", (incident_id,)
        ).fetchone()
        return _incident_from(row) if row else None

    @_guard
    def set_state(self, incident_id: str, state: State) -> None:
        validate_state(state)
        cursor = self._con.execute(
            "UPDATE incidents SET state = %s WHERE incident_id = %s", (state, incident_id)
        )
        if cursor.rowcount == 0:
            raise MemoryStoreError("o incidente não existe")

    @_guard
    def find_open(
        self,
        metric: str,
        direction: str,
        causes: list[dict[str, str]],
        within_hours: float,
        now: datetime | None = None,
    ) -> Incident | None:
        now = now or utcnow()
        row = self._con.execute(
            f"SELECT {_INCIDENT_COLUMNS} FROM incidents "
            "WHERE metric = %s AND direction = %s AND cause_key = %s AND state = ANY(%s) "
            "AND created_at <= %s AND created_at >= %s::timestamptz - make_interval(secs => %s) "
            "ORDER BY created_at DESC LIMIT 1",
            (
                metric,
                direction,
                cause_key(causes),
                list(ACTIVE_STATES),
                now,
                now,
                within_hours * 3600,
            ),
        ).fetchone()
        return _incident_from(row) if row else None

    @_guard
    def similar(
        self, text: str, k: int, min_similarity: float, exclude: str | None = None
    ) -> list[SimilarIncident]:
        vectors = self._embedder.embed([text])
        check_vectors(self._embedder, vectors, 1)
        query = _vector(vectors[0])
        rows = self._con.execute(
            f"SELECT {_INCIDENT_COLUMNS}, 1 - (embedding <=> %s) AS similarity FROM incidents "
            "WHERE (%s::text IS NULL OR incident_id <> %s) ORDER BY embedding <=> %s LIMIT %s",
            (query, exclude, exclude, query, k),
        ).fetchall()
        hits = [
            SimilarIncident(
                incident=_incident_from(row[:-1]), similarity=float(np.clip(row[-1], 0, 1))
            )
            for row in rows
        ]
        return [hit for hit in hits if hit.similarity >= min_similarity]

    @_guard
    def decision_stats(self) -> DecisionStats:
        rows = self._con.execute(
            "SELECT state, count(*) FROM incidents WHERE state IN ('approved', 'ignored') "
            "GROUP BY state"
        ).fetchall()
        counts = dict(rows)
        return DecisionStats(
            approved=int(counts.get("approved", 0)), ignored=int(counts.get("ignored", 0))
        )


class PgKnowledgeStore:
    def __init__(self, con: psycopg.Connection, embedder: Embedder) -> None:
        self._con = con
        self._embedder = embedder

    @_guard
    def index(self, chunks: list[Chunk]) -> IndexReport:
        safe, skipped = screen(chunks)
        report = IndexReport(skipped=skipped)
        wanted = {c.content_hash: c for c in safe}
        sources = sorted({c.source for c in chunks})
        with self._con.transaction():
            existing = {
                digest
                for (digest,) in self._con.execute(
                    "SELECT content_hash FROM knowledge_chunks WHERE source = ANY(%s)", (sources,)
                )
            }
            stale = sorted(existing - wanted.keys())
            if stale:
                self._con.execute(
                    "DELETE FROM knowledge_chunks WHERE content_hash = ANY(%s)", (stale,)
                )
            new = [c for h, c in wanted.items() if h not in existing]
            if new:
                vectors = self._embedder.embed([c.embedding_text() for c in new])
                check_vectors(self._embedder, vectors, len(new))
                with self._con.cursor() as cursor:
                    cursor.executemany(
                        "INSERT INTO knowledge_chunks (source, heading, content, content_hash, "
                        "embedding) VALUES (%s, %s, %s, %s, %s) "
                        "ON CONFLICT (content_hash) DO NOTHING",
                        [
                            (c.source, c.heading, c.content, c.content_hash, _vector(v))
                            for c, v in zip(new, vectors, strict=True)
                        ],
                    )
        report.removed = len(stale)
        report.added = len(new)
        report.kept = len(wanted) - len(new)
        return report

    @_guard
    def search(self, query: str, k: int, min_similarity: float) -> list[KnowledgeHit]:
        vectors = self._embedder.embed([query])
        check_vectors(self._embedder, vectors, 1)
        vector = _vector(vectors[0])
        rows = self._con.execute(
            "SELECT source, heading, content, 1 - (embedding <=> %s) FROM knowledge_chunks "
            "ORDER BY embedding <=> %s LIMIT %s",
            (vector, vector, k),
        ).fetchall()
        hits = [
            KnowledgeHit(
                chunk=Chunk(source=source, heading=heading, content=content),
                similarity=float(np.clip(similarity, 0, 1)),
            )
            for source, heading, content, similarity in rows
        ]
        return [hit for hit in hits if hit.similarity >= min_similarity]

    @_guard
    def count(self) -> int:
        (total,) = self._con.execute("SELECT count(*) FROM knowledge_chunks").fetchone()  # type: ignore[misc]
        return int(total)
