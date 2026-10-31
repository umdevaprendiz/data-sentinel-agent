from datetime import date

import numpy as np
import pandas as pd
import pytest

from sentinel.data.db import connect_readonly
from sentinel.data.generator import generate_events, load_scenarios, write_database
from sentinel.detection.detector import (
    METRIC_PRIORITY,
    DetectorParams,
    daily_series,
    detect,
    expected_series,
    load_daily_series,
    select_root,
)
from sentinel.schemas import Anomaly

from .conftest import SCENARIOS_FILE

WEEKDAY_FACTOR = np.array([1.0, 1.05, 1.05, 1.02, 1.0, 0.8, 0.75])
PARAMS = DetectorParams()
NORMAL_IDS = ["s15", "s16", "s17", "s18"]
STEP_IDS = [f"s{i:02d}" for i in [*range(1, 15), 19]]
ALL = {s.id: s for s in load_scenarios(SCENARIOS_FILE)}


def make_series(
    seed: int = 0,
    days: int = 90,
    noise: float = 0.015,
    effects: dict[str, tuple[int, float]] | None = None,
) -> pd.DataFrame:
    """Série sintética semanal. `effects[metric] = (primeiro_dia, fator)` multiplica até o fim."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=days, freq="D")
    columns = {}
    for metric in METRIC_PRIORITY:
        values = 1000 * WEEKDAY_FACTOR[index.day_of_week] * (1 + rng.normal(0, noise, days))
        if effects and metric in effects:
            first, factor = effects[metric]
            values[first:] *= factor
        columns[metric] = values
    return pd.DataFrame(columns, index=index)


def anomaly(metric: str) -> Anomaly:
    return Anomaly(
        metric=metric,  # type: ignore[arg-type]
        window_start=date(2026, 3, 26),
        window_end=date(2026, 3, 31),
        onset=date(2026, 3, 26),
        observed=1.0,
        expected=2.0,
        z_score=-5.0,
        direction="drop",
    )


# --- Núcleo, em séries sintéticas -------------------------------------------------------------


def test_expected_is_the_median_of_the_same_weekday():
    values = pd.Series(np.arange(70, dtype=float), index=pd.date_range("2026-01-01", periods=70))
    expected = expected_series(values, weeks=4)
    assert expected.iloc[:28].isna().all()
    # lags 7, 14, 21, 28 do dia 30 → 23, 16, 9, 2; mediana = 12,5
    assert expected.iloc[30] == pytest.approx(12.5)


@pytest.mark.parametrize("seed", range(20))
def test_clean_series_has_no_anomaly(seed):
    assert detect(make_series(seed), PARAMS) == []


def test_weekly_pattern_without_noise_is_not_an_anomaly():
    assert detect(make_series(noise=0.0), PARAMS) == []


def test_step_drop_is_detected_with_direction_onset_and_window():
    series = make_series(effects={"sessions": (85, 0.7)})
    [found] = detect(series, PARAMS)
    assert found.metric == "sessions"
    assert found.direction == "drop"
    assert found.onset == date(2026, 3, 27)  # dia 85 (0-based) = 27/03
    assert found.window_end == date(2026, 3, 31)
    assert found.observed < found.expected
    assert found.z_score < -PARAMS.z_threshold


def test_spike_direction():
    [found] = detect(make_series(effects={"revenue": (84, 1.5)}), PARAMS)
    assert (found.metric, found.direction) == ("revenue", "spike")
    assert found.observed > found.expected


def test_a_single_day_is_not_enough():
    assert detect(make_series(effects={"sessions": (89, 0.6)}), PARAMS) == []
    assert len(detect(make_series(effects={"sessions": (88, 0.6)}), PARAMS)) == 1


def test_min_consecutive_days_is_configurable():
    one_day = make_series(effects={"sessions": (89, 0.6)})
    assert len(detect(one_day, DetectorParams(min_consecutive_days=1))) == 1


def test_small_persistent_shift_is_caught_by_the_sustained_rule():
    """~2 desvios por dia: nenhum dia chega ao limiar sozinho, mas a janela inteira desloca."""
    series = make_series(noise=0.01, effects={"conversions": (83, 0.975)})
    assert detect(series, DetectorParams(sustained_z_threshold=1e9)) == []
    [found] = detect(series, PARAMS)
    assert (found.metric, found.direction) == ("conversions", "drop")


def test_not_enough_history_is_an_error():
    with pytest.raises(ValueError, match="histórico insuficiente"):
        detect(make_series(days=60), PARAMS)


def test_gaps_in_the_series_are_an_error():
    series = make_series().drop(pd.Timestamp("2026-02-10"))
    with pytest.raises(ValueError, match="sem buracos"):
        detect(series, PARAMS)


def test_as_of_ignores_what_happens_later():
    series = make_series(effects={"sessions": (85, 0.7)})
    assert detect(series, PARAMS, as_of=date(2026, 3, 20)) == []
    assert len(detect(series, PARAMS)) == 1


# --- Métrica-raiz -----------------------------------------------------------------------------


def test_root_follows_the_metric_tree():
    found = [
        anomaly("revenue"),
        anomaly("conversions"),
        anomaly("conversion_rate"),
        anomaly("sessions"),
    ]
    assert select_root(found).metric == "sessions"
    assert select_root(found[:3]).metric == "conversion_rate"
    assert select_root(found[:2]).metric == "conversions"
    assert select_root([]) is None


def test_results_come_in_priority_order():
    series = make_series(effects={"revenue": (85, 0.7), "sessions": (85, 0.7)})
    assert [a.metric for a in detect(series, PARAMS)] == ["sessions", "revenue"]


# --- Séries diárias ---------------------------------------------------------------------------


def test_daily_series_derives_ratios(events_of):
    series = daily_series(events_of("s15"))
    assert list(series.columns) == list(METRIC_PRIORITY)
    assert len(series) == 90
    assert series["conversion_rate"].iloc[0] == pytest.approx(
        series["conversions"].iloc[0] / series["sessions"].iloc[0]
    )
    assert series["avg_ticket"].iloc[0] == pytest.approx(
        series["revenue"].iloc[0] / series["conversions"].iloc[0]
    )


def test_series_from_the_database_matches_the_one_from_events(scenarios, seed, tmp_path):
    write_database(scenarios["s05"], seed, tmp_path / "s.db")
    con = connect_readonly(tmp_path / "s.db")
    from_db = load_daily_series(con)
    con.close()
    from_events = daily_series(generate_events(scenarios["s05"], seed))
    pd.testing.assert_frame_equal(from_db, from_events, check_freq=False)


# --- Cenários do projeto ----------------------------------------------------------------------


@pytest.mark.parametrize("scenario_id", STEP_IDS)
def test_every_expected_anomaly_is_found(scenario_id, events_of):
    found = detect(daily_series(events_of(scenario_id)), PARAMS)
    assert found, f"{scenario_id}: nenhuma anomalia detectada"
    root = select_root(found)
    cause = ALL[scenario_id].causes[0]
    if scenario_id != "s19":
        assert abs((root.onset - cause.start).days) <= 1


@pytest.mark.parametrize(
    ("scenario_id", "metric", "direction"),
    [
        ("s05", "sessions", "drop"),
        ("s08", "sessions", "drop"),
        ("s09", "sessions", "drop"),
        ("s14", "sessions", "spike"),
        ("s06", "conversion_rate", "drop"),
        ("s07", "conversion_rate", "drop"),
        ("s13", "conversion_rate", "drop"),
        ("s19", "conversion_rate", "drop"),
    ],
)
def test_root_metric_and_direction(scenario_id, metric, direction, events_of):
    root = select_root(detect(daily_series(events_of(scenario_id)), PARAMS))
    assert (root.metric, root.direction) == (metric, direction)


@pytest.mark.parametrize("scenario_id", NORMAL_IDS)
def test_normal_scenarios_raise_no_alarm(scenario_id, events_of):
    assert detect(daily_series(events_of(scenario_id)), PARAMS) == []


@pytest.mark.parametrize("scenario_id", NORMAL_IDS)
def test_no_false_alarm_over_20_seeds(scenario_id):
    false_alarms = [
        seed
        for seed in range(20)
        if detect(daily_series(generate_events(ALL[scenario_id], seed)), PARAMS)
    ]
    assert false_alarms == []


@pytest.mark.slow
@pytest.mark.parametrize("scenario_id", NORMAL_IDS)
def test_no_false_alarm_over_200_seeds(scenario_id):
    """Critério de aceitação da etapa 2. Sementes de teste (0–199), separadas das de calibração."""
    false_alarms = [
        seed
        for seed in range(200)
        if detect(daily_series(generate_events(ALL[scenario_id], seed)), PARAMS)
    ]
    assert false_alarms == []


@pytest.mark.slow
def test_recall_over_40_seeds():
    """Cenários com degrau são sempre achados; a deriva lenta (s20) é mais difícil e é só medida."""
    for scenario_id in STEP_IDS:
        missed = [
            seed
            for seed in range(40)
            if not detect(daily_series(generate_events(ALL[scenario_id], seed)), PARAMS)
        ]
        assert missed == [], scenario_id

    drift_hits = sum(
        bool(detect(daily_series(generate_events(ALL["s20"], seed)), PARAMS)) for seed in range(40)
    )
    assert drift_hits / 40 >= 0.5
