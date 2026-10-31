import json
import sqlite3
from math import prod

import numpy as np
import pandas as pd
import pytest

from sentinel.data.db import connect_readonly
from sentinel.data.generator import (
    N_DAYS,
    cause_mask,
    generate_changes,
    generate_events,
    load_scenarios,
    write_database,
)
from sentinel.schemas import DIMENSION_VALUES

from .conftest import SCENARIOS_FILE

ALL = load_scenarios(SCENARIOS_FILE)
CAUSED = [s for s in ALL if s.causes]
CAUSE_PARAMS = [(s.id, i) for s in CAUSED for i in range(len(s.causes))]
ROWS_PER_DAY = prod(len(v) for v in DIMENSION_VALUES.values())
NUM_COLS = ["sessions", "conversions", "revenue"]
CAT_COLS = ["day", *DIMENSION_VALUES]
LAST_DAYS = ["2026-03-30", "2026-03-31"]  # todas as causas já estão em efeito pleno


def test_shape_and_columns(events_of):
    df = events_of("s15")
    assert len(df) == N_DAYS * ROWS_PER_DAY == 90 * 720
    assert list(df.columns) == [*CAT_COLS, *NUM_COLS]
    assert df["day"].nunique() == 90
    assert df["day"].min() == "2026-01-01"
    assert df["day"].max() == "2026-03-31"
    for dim, values in DIMENSION_VALUES.items():
        assert set(df[dim]) == set(values)


def test_same_seed_is_deterministic(scenarios, seed):
    a = generate_events(scenarios["s01"], seed)
    b = generate_events(scenarios["s01"], seed)
    pd.testing.assert_frame_equal(a, b)


def test_other_seed_changes_the_noise(scenarios, seed):
    a = generate_events(scenarios["s15"], seed)
    b = generate_events(scenarios["s15"], seed + 1)
    assert not a["sessions"].equals(b["sessions"])


def test_values_are_valid(events_of):
    df = events_of("s07")
    assert (df["sessions"] >= 0).all()
    assert (df["conversions"] >= 0).all()
    assert (df["conversions"] <= df["sessions"]).all()
    assert (df["revenue"] >= 0).all()


@pytest.mark.parametrize("scenario_id", [s.id for s in CAUSED])
def test_cause_is_isolated_to_its_slice(scenario_id, scenarios, events_of):
    """Fora da fatia e do período da causa, os dados são idênticos aos do controle."""
    scenario = scenarios[scenario_id]
    df, control = events_of(scenario_id), events_of(scenario_id, with_causes=False)
    assert df[CAT_COLS].equals(control[CAT_COLS])

    inside = np.zeros(len(df), dtype=bool)
    for cause in scenario.causes:
        inside |= cause_mask(df, cause).to_numpy()
    assert inside.any()

    outside_same = df.loc[~inside, NUM_COLS].to_numpy() == control.loc[~inside, NUM_COLS].to_numpy()
    assert outside_same.all()
    inside_diff = df.loc[inside, NUM_COLS].to_numpy() != control.loc[inside, NUM_COLS].to_numpy()
    assert inside_diff.any()


@pytest.mark.parametrize(("scenario_id", "cause_index"), CAUSE_PARAMS)
def test_effect_matches_the_injected_factors(scenario_id, cause_index, scenarios, events_of):
    scenario = scenarios[scenario_id]
    cause = scenario.causes[cause_index]
    df, control = events_of(scenario_id), events_of(scenario_id, with_causes=False)

    mask = cause_mask(df, cause)
    for other in scenario.causes:
        if other is not cause:
            mask &= ~cause_mask(df, other)
    mask &= df["day"].isin(LAST_DAYS)
    assert mask.any()

    got, base = df.loc[mask, NUM_COLS].sum(), control.loc[mask, NUM_COLS].sum()
    assert got["sessions"] / base["sessions"] == pytest.approx(cause.sessions_factor, abs=0.02)
    conv_ratio = got["conversions"] / base["conversions"]
    assert conv_ratio == pytest.approx(cause.sessions_factor * cause.conversion_factor, abs=0.1)
    if got["conversions"] > 0:
        price_ratio = (got["revenue"] / got["conversions"]) / (
            base["revenue"] / base["conversions"]
        )
        assert price_ratio == pytest.approx(cause.price_factor, abs=0.03)


def test_pair_cause_touches_only_the_two_dimensions(scenarios, events_of):
    df = events_of("s01")
    cause = scenarios["s01"].causes[0]
    mask = cause_mask(df, cause)
    assert (df.loc[mask, "platform"] == "android").all()
    assert (df.loc[mask, "app_version"] == "3.2").all()
    assert mask.sum() == 6 * (ROWS_PER_DAY // (2 * 3))  # 6 dias de efeito


@pytest.mark.parametrize("scenario_id", ["s06", "s07"])
def test_stock_out_zeroes_conversions(scenario_id, scenarios, events_of):
    df = events_of(scenario_id)
    mask = cause_mask(df, scenarios[scenario_id].causes[0])
    assert mask.any()
    assert (df.loc[mask, "conversions"] == 0).all()
    assert (df.loc[mask, "revenue"] == 0).all()
    assert (df.loc[mask, "sessions"] > 0).all()


def test_drift_is_gradual(scenarios, events_of):
    df, control = events_of("s20"), events_of("s20", with_causes=False)
    mask = cause_mask(df, scenarios["s20"].causes[0])
    days = df.loc[mask, "day"]
    ratios = []
    for week in range(4):
        start = pd.Timestamp("2026-03-03") + pd.Timedelta(days=7 * week)
        in_week = days.between(
            start.strftime("%Y-%m-%d"), (start + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        )
        idx = in_week[in_week].index
        ratios.append(df.loc[idx, "conversions"].sum() / control.loc[idx, "conversions"].sum())
    assert ratios[0] > 0.9
    assert all(a > b for a, b in zip(ratios, ratios[1:], strict=False))
    assert ratios[-1] < 0.7


def test_diffuse_drop_hits_every_slice(scenarios, events_of):
    df, control = events_of("s19"), events_of("s19", with_causes=False)
    cause = scenarios["s19"].causes[0]
    assert cause.dimensions == {}
    mask = cause_mask(df, cause)
    assert mask.sum() == 6 * ROWS_PER_DAY
    for region in DIMENSION_VALUES["region"]:
        in_region = mask & (df["region"] == region)
        assert (
            df.loc[in_region, "conversions"].sum()
            < 0.95 * control.loc[in_region, "conversions"].sum()
        )


def test_strong_seasonality_scenario_has_deeper_weekends(events_of):
    def weekend_ratio(df: pd.DataFrame) -> float:
        daily = df.groupby("day")["sessions"].sum()
        weekend = pd.to_datetime(daily.index).day_of_week >= 5
        return daily[weekend].mean() / daily[~weekend].mean()

    assert weekend_ratio(events_of("s18")) < weekend_ratio(events_of("s15")) - 0.1


@pytest.mark.parametrize("scenario", ALL, ids=lambda s: s.id)
def test_changes_are_what_the_scenario_declares(scenario):
    changes = generate_changes(scenario)
    assert list(changes.columns) == ["ts", "type", "description", "scope"]
    assert len(changes) == len(scenario.changes)


def test_unlogged_and_decoy_scenarios(scenarios):
    assert generate_changes(scenarios["s13"]).empty
    decoy = generate_changes(scenarios["s03"])
    assert len(decoy) == 2
    cause_dims = scenarios["s03"].causes[0].dimensions
    scopes = [json.loads(s) for s in decoy["scope"]]
    assert sum(scope == cause_dims for scope in scopes) == 1


def test_database_is_read_only_and_has_no_ground_truth(scenarios, seed, tmp_path):
    path = tmp_path / "sentinel.db"
    write_database(scenarios["s03"], seed, path)

    con = connect_readonly(path)
    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"events", "changes"}
    assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == N_DAYS * ROWS_PER_DAY
    assert con.execute("SELECT COUNT(*) FROM changes").fetchone()[0] == 2
    with pytest.raises(sqlite3.OperationalError):
        con.execute("INSERT INTO changes VALUES ('2026-01-01', 'deploy', 'x', '{}')")
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM events")
    con.close()
