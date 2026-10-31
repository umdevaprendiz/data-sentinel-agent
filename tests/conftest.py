from pathlib import Path

import pytest

from sentinel.data.generator import generate_events, load_scenarios

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
