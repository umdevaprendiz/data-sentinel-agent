"""Fábrica da memória e erros do Postgres: sem servidor, sem Ollama, sem vazar o DSN."""

import pytest
from pydantic import SecretStr

from sentinel.config import Settings
from sentinel.llm.embeddings import FakeEmbedder
from sentinel.memory import MemoryStoreError
from sentinel.memory.factory import open_memory
from sentinel.memory.pg import PgMemory
from sentinel.security import redact


def settings(**values) -> Settings:
    return Settings(_env_file=None, **values)


def test_without_a_dsn_the_memory_lives_in_ram():
    memory = open_memory(settings(), FakeEmbedder(dim=64))
    assert memory.backend == "ram" and memory.pg is None
    memory.close()  # não falha sem conexão


def test_a_unreachable_server_gives_a_safe_error():
    password = "pw" + "x9y8z7"
    dsn = f"postgresql://usr:{password}@127.0.0.1:1/db"  # porta 1: ninguém escuta
    with pytest.raises(MemoryStoreError) as caught:
        open_memory(settings(memory_dsn=SecretStr(dsn)), FakeEmbedder(dim=64))
    message = str(caught.value)
    assert password not in message and "127.0.0.1" not in message and "usr" not in message
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


def test_connect_registers_the_dsn_as_a_secret_for_redaction():
    password = "pw" + "q1w2e3"
    with pytest.raises(MemoryStoreError):
        PgMemory.connect(f"postgresql://u:{password}@127.0.0.1:1/db", dim=8)
    assert password not in redact(f"falhou com {password}")


def test_invalid_schema_and_dimension_are_refused_before_connecting():
    dsn = "postgresql://" + "u:" + "pw1234" + "@127.0.0.1:1/db"
    with pytest.raises(MemoryStoreError, match="schema"):
        PgMemory.connect(dsn, dim=8, schema="x; DROP TABLE y")
    with pytest.raises(MemoryStoreError, match="dimensão"):
        PgMemory.connect(dsn, dim=0)
