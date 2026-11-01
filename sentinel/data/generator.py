"""Gerador de dados sintéticos com causa injetada (seção 7 do CLAUDE.md).

Regra central: todo o ruído é sorteado antes e sem olhar o cenário. A causa só multiplica valores
nas linhas que casam com a fatia e com o período. Por isso, fora disso os dados são idênticos aos
de um controle sem causa gerado com a mesma semente.
"""

import json
import sqlite3
import zlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.schemas import DIMENSION_VALUES, Cause, Scenario

START_DATE = date(2026, 1, 1)
N_DAYS = 90
BASE_DAILY_SESSIONS = 400_000

SESSION_SHARE: dict[str, dict[str, float]] = {
    "region": {"norte": 0.08, "nordeste": 0.20, "centro_oeste": 0.10, "sudeste": 0.42, "sul": 0.20},
    "platform": {"android": 0.60, "ios": 0.40},
    "app_version": {"3.0": 0.20, "3.1": 0.30, "3.2": 0.50},
    "product": {
        "laptop": 0.12,
        "phone": 0.25,
        "tablet": 0.15,
        "watch": 0.12,
        "headset": 0.18,
        "charger": 0.18,
    },
    "channel": {"organic": 0.40, "paid_search": 0.25, "social": 0.20, "email": 0.15},
}

# Taxa de conversão = taxa do produto × modificadores de região, plataforma e canal.
PRODUCT_RATE = {
    "laptop": 0.020,
    "phone": 0.025,
    "tablet": 0.025,
    "watch": 0.030,
    "headset": 0.035,
    "charger": 0.050,
}
RATE_MODIFIER: dict[str, dict[str, float]] = {
    "region": {"norte": 0.90, "nordeste": 0.95, "centro_oeste": 1.0, "sudeste": 1.08, "sul": 1.05},
    "platform": {"android": 0.92, "ios": 1.14},
    "channel": {"organic": 1.0, "paid_search": 1.10, "social": 0.75, "email": 1.45},
}
PRODUCT_PRICE = {
    "laptop": 3500.0,
    "phone": 2000.0,
    "tablet": 1500.0,
    "watch": 800.0,
    "headset": 300.0,
    "charger": 60.0,
}

# Desvio do dia da semana (segunda a domingo), na amplitude padrão de 0,15.
WEEKDAY_PATTERN = np.array([0.08, 0.10, 0.10, 0.06, 0.0, -0.18, -0.20])
DEFAULT_AMPLITUDE = 0.15

# Ruído (multiplicativo), escalado por `noise_level`.
SD_SESSIONS_DAY, SD_SESSIONS_SLICE, SD_SESSIONS_ROW = 0.02, 0.010, 0.05
SD_RATE_DAY, SD_RATE_SLICE = 0.012, 0.008
SD_REVENUE_ROW = 0.04

EVENT_COLUMNS = ["day", *DIMENSION_VALUES, "sessions", "conversions", "revenue"]


def load_scenarios(path: str | Path) -> list[Scenario]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Scenario.model_validate_json(line) for line in lines if line.strip()]


def _day_strings() -> np.ndarray:
    return np.array([(START_DATE + timedelta(days=i)).isoformat() for i in range(N_DAYS)])


def cause_weights(cause: Cause) -> np.ndarray:
    """Intensidade do efeito por dia (0 = fora do período, 1 = efeito pleno)."""
    days = np.arange(N_DAYS)
    first = (cause.start - START_DATE).days
    last = N_DAYS - 1 if cause.end is None else (cause.end - START_DATE).days
    active = (days >= first) & (days <= last)
    ramp = (
        np.clip((days - first + 1) / cause.ramp_days, 0, 1) if cause.ramp_days else np.ones(N_DAYS)
    )
    return np.where(active, ramp, 0.0)


def cause_mask(events: pd.DataFrame, cause: Cause) -> pd.Series:
    """Linhas da fatia da causa, nos dias em que ela está ativa."""
    last = cause.end or START_DATE + timedelta(days=N_DAYS - 1)
    mask = events["day"].between(cause.start.isoformat(), last.isoformat())
    for name, value in cause.dimensions.items():
        mask &= events[name] == value
    return mask


def generate_events(scenario: Scenario, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng([seed, zlib.crc32(scenario.id.encode())])
    level = scenario.noise_level

    shape = (N_DAYS, *(len(values) for values in DIMENSION_VALUES.values()))
    grid = np.indices(shape).reshape(len(shape), -1)
    n = grid.shape[1]
    day_idx = grid[0]
    dim_idx = dict(zip(DIMENSION_VALUES, grid[1:], strict=True))

    # Sorteios: ordem fixa e independente da causa.
    shock_s_day = rng.normal(0, SD_SESSIONS_DAY, N_DAYS)
    shock_r_day = rng.normal(0, SD_RATE_DAY, N_DAYS)
    shock_s_slice = {
        d: rng.normal(0, SD_SESSIONS_SLICE, (N_DAYS, len(v))) for d, v in DIMENSION_VALUES.items()
    }
    shock_r_slice = {
        d: rng.normal(0, SD_RATE_SLICE, (N_DAYS, len(v))) for d, v in DIMENSION_VALUES.items()
    }
    noise_sessions = rng.normal(0, SD_SESSIONS_ROW, n)
    z_conversions = rng.normal(0, 1, n)
    noise_revenue = rng.normal(0, SD_REVENUE_ROW, n)

    weekdays = np.array([(START_DATE + timedelta(days=i)).weekday() for i in range(N_DAYS)])
    amplitude = scenario.weekly_amplitude / DEFAULT_AMPLITUDE
    week_sessions = 1 + amplitude * WEEKDAY_PATTERN[weekdays]
    week_rate = 1 + 0.25 * amplitude * WEEKDAY_PATTERN[weekdays]

    # Efeitos das causas (multiplicadores; exatamente 1.0 fora da fatia e do período).
    sessions_mult, conversion_mult, price_mult = np.ones(n), np.ones(n), np.ones(n)
    for cause in scenario.causes:
        in_slice = np.ones(n, dtype=bool)
        for name, value in cause.dimensions.items():
            in_slice &= dim_idx[name] == DIMENSION_VALUES[name].index(value)
        gate = cause_weights(cause)[day_idx] * in_slice
        sessions_mult *= 1 + (cause.sessions_factor - 1) * gate
        conversion_mult *= 1 + (cause.conversion_factor - 1) * gate
        price_mult *= 1 + (cause.price_factor - 1) * gate

    # Sessões.
    share = np.ones(n)
    for name, values in DIMENSION_VALUES.items():
        share *= np.array([SESSION_SHARE[name][v] for v in values])[dim_idx[name]]
    session_noise = shock_s_day[day_idx] + noise_sessions
    for name in DIMENSION_VALUES:
        session_noise = session_noise + shock_s_slice[name][day_idx, dim_idx[name]]
    sessions_base = (
        BASE_DAILY_SESSIONS
        * share
        * week_sessions[day_idx]
        * np.maximum(0.2, 1 + level * session_noise)
    )
    sessions = np.rint(sessions_base * sessions_mult).astype(np.int64)

    # Conversões.
    product_values = DIMENSION_VALUES["product"]
    rate = np.array([PRODUCT_RATE[v] for v in product_values])[dim_idx["product"]]
    for name, modifiers in RATE_MODIFIER.items():
        rate = rate * np.array([modifiers[v] for v in DIMENSION_VALUES[name]])[dim_idx[name]]
    rate_noise = shock_r_day[day_idx]
    for name in DIMENSION_VALUES:
        rate_noise = rate_noise + shock_r_slice[name][day_idx, dim_idx[name]]
    p = np.clip(
        rate * week_rate[day_idx] * np.maximum(0.2, 1 + level * rate_noise) * conversion_mult,
        0,
        0.95,
    )
    expected = sessions * p
    conversions = np.clip(
        np.rint(expected + z_conversions * np.sqrt(expected * (1 - p))), 0, sessions
    ).astype(np.int64)

    # Receita.
    price = np.array([PRODUCT_PRICE[v] for v in product_values])[dim_idx["product"]] * price_mult
    revenue = np.round(conversions * price * np.maximum(0.5, 1 + level * noise_revenue), 2)

    columns: dict[str, np.ndarray] = {"day": _day_strings()[day_idx]}
    for name, values in DIMENSION_VALUES.items():
        columns[name] = np.array(values)[dim_idx[name]]
    columns.update(sessions=sessions, conversions=conversions, revenue=revenue)
    return pd.DataFrame(columns)[EVENT_COLUMNS]


def generate_changes(scenario: Scenario) -> pd.DataFrame:
    rows = [
        {
            "ts": c.ts.isoformat(),
            "type": c.type,
            "description": c.description,
            "scope": json.dumps(c.scope, sort_keys=True),
        }
        for c in sorted(scenario.changes, key=lambda c: c.ts)
    ]
    return pd.DataFrame(rows, columns=["ts", "type", "description", "scope"])


def write_database(scenario: Scenario, seed: int, db_path: str | Path) -> None:
    """Grava `events` e `changes`. O gabarito NÃO vai para o banco, para o agente não enxergá-lo."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    events, changes = generate_events(scenario, seed), generate_changes(scenario)
    with sqlite3.connect(path) as con:
        events.to_sql("events", con, index=False)
        changes.to_sql("changes", con, index=False)
        con.execute("CREATE INDEX idx_events_day ON events(day)")
    # `with` não fecha a conexão no sqlite3; o arquivo precisa ficar livre no Windows.
    con.close()


def write_ground_truth(scenario: Scenario, seed: int, path: str | Path) -> None:
    """Gabarito em arquivo separado do banco."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "scenario": json.loads(scenario.model_dump_json())}
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
