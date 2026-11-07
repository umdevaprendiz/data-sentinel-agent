import uuid
from pathlib import Path

import pytest
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from sentinel.config import Settings
from sentinel.data.db import connect_readonly
from sentinel.data.generator import generate_events, load_scenarios, write_database
from sentinel.detection.detector import DetectorParams, daily_series, detect, select_root
from sentinel.investigation.cube import Cube
from sentinel.investigation.plan import investigate
from sentinel.investigation.tools import ToolContext
from sentinel.memory import MemoryStoreError
from sentinel.memory.pg import PgMemory

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_FILE = ROOT / "scenarios" / "scenarios.jsonl"
SEED = 42


@pytest.fixture(scope="session")
def scenarios():
    return {s.id: s for s in load_scenarios(SCENARIOS_FILE)}


@pytest.fixture(scope="session")
def seed() -> int:
    return SEED


@pytest.fixture(scope="session")
def events_of(scenarios, seed):
    """Eventos de um cenário; `with_causes=False` gera o controle (mesma semente, sem causa)."""
    cache = {}

    def get(scenario_id: str, with_causes: bool = True):
        key = (scenario_id, with_causes)
        if key not in cache:
            scenario = scenarios[scenario_id]
            if not with_causes:
                scenario = scenario.model_copy(update={"causes": []})
            cache[key] = generate_events(scenario, seed)
        return cache[key]

    return get


@pytest.fixture(scope="session")
def cube_of(events_of):
    """Cubo em memória dos eventos de um cenário (cacheado)."""
    cache = {}

    def get(scenario_id: str) -> Cube:
        if scenario_id not in cache:
            cache[scenario_id] = Cube.from_frame(events_of(scenario_id))
        return cache[scenario_id]

    return get


@pytest.fixture(scope="session")
def ctx_of(scenarios, seed, tmp_path_factory):
    """`ToolContext` novo, sobre um banco somente leitura do cenário (o banco é cacheado)."""
    folder = tmp_path_factory.mktemp("dbs")
    connections = []

    def get(scenario_id: str) -> ToolContext:
        path = folder / f"{scenario_id}.db"
        if not path.exists():
            write_database(scenarios[scenario_id], seed, path)
        con = connect_readonly(path)
        connections.append(con)
        return ToolContext(con, Settings(_env_file=None))

    yield get
    for con in connections:
        con.close()


@pytest.fixture(scope="session")
def plan_of(ctx_of, events_of):
    """(anomalia, hipóteses, findings) do plano determinístico sobre um cenário (cacheado)."""
    cache = {}

    def get(scenario_id: str):
        if scenario_id not in cache:
            anomaly = select_root(detect(daily_series(events_of(scenario_id)), DetectorParams()))
            ctx = ctx_of(scenario_id)
            hypotheses = investigate(ctx, anomaly)
            cache[scenario_id] = (anomaly, hypotheses, list(ctx.findings))
        return cache[scenario_id]

    return get


# --- Postgres com pgvector (testes marcados com `pg`) -------------------------------------------


class _PgTestSettings(BaseSettings):
    """DSN do banco de testes (`TEST_MEMORY_DSN` no `.env`): nunca o banco de uso real."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    test_memory_dsn: SecretStr = SecretStr("")


@pytest.fixture
def pg_memory():
    """`PgMemory` num schema próprio e descartável. Sem `TEST_MEMORY_DSN`, o teste é pulado."""
    dsn = _PgTestSettings().test_memory_dsn.get_secret_value()
    if not dsn:
        pytest.skip("TEST_MEMORY_DSN não configurado")
    schema = f"t_{uuid.uuid4().hex[:12]}"
    try:
        memory = PgMemory.connect(dsn, dim=64, schema=schema)
    except MemoryStoreError as error:  # configurado mas inutilizável: falha, não pula
        pytest.fail(f"banco de testes inutilizável: {error}", pytrace=False)
    yield memory
    memory.drop_schema()
    memory.close()


@pytest.fixture(scope="session")
def db_of(scenarios, seed, tmp_path_factory):
    """Caminho de um banco gerado para o cenário (cacheado; abrir sempre como somente leitura)."""
    folder = tmp_path_factory.mktemp("graph_dbs")

    def get(scenario_id: str) -> Path:
        path = folder / f"{scenario_id}.db"
        if not path.exists():
            write_database(scenarios[scenario_id], seed, path)
        return path

    return get
