"""Perfil descoberto nos dados (decisão 21): métricas e dimensões não são listas fixas."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from sentinel.data.db import connect_readonly
from sentinel.data.profile import (
    DEFAULT_PROFILE,
    ProfileError,
    discover_profile,
    profile_from_frame,
    propose_dimension,
    propose_ratio,
)
from sentinel.detection.detector import METRIC_PRIORITY, daily_series, detect, select_root
from sentinel.investigation.cube import Cube
from sentinel.investigation.drilldown import Window, find_causes


def test_the_generated_data_gives_the_default_profile(events_of):
    profile = profile_from_frame(events_of("s01"))
    assert profile == DEFAULT_PROFILE
    assert profile.metrics == METRIC_PRIORITY
    assert profile.volume == "sessions"
    assert profile.rates == ("conversion_rate",)
    assert profile.chain("revenue") == ["sessions", "conversion_rate", "avg_ticket"]
    assert profile.deepest_measure == "revenue"
    assert profile.count_measures == ("sessions", "conversions")


def test_discovery_reads_the_database(scenarios, seed, tmp_path):
    from sentinel.data.generator import write_database

    write_database(scenarios["s05"], seed, tmp_path / "s.db")
    con = connect_readonly(tmp_path / "s.db")
    try:
        assert discover_profile(con) == DEFAULT_PROFILE
    finally:
        con.close()


def _shop(rng: np.random.Generator, days: int = 84) -> pd.DataFrame:
    """Outro negócio: lojas, cliques, pedidos e valor (nenhum nome do gerador)."""
    rows = []
    for d in range(days):
        day = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=d)).date().isoformat()
        for store in ("centro", "praia", "shopping"):
            for kind in ("fisica", "online"):
                clicks = int(rng.integers(900, 1100))
                orders = int(rng.binomial(clicks, 0.1))
                rows.append((day, store, kind, clicks, orders, round(orders * 50.0, 2)))
    return pd.DataFrame(rows, columns=["dt", "store", "kind", "clicks", "orders", "amount"])


def test_another_business_is_profiled_without_code_changes():
    profile = profile_from_frame(_shop(np.random.default_rng(1)))
    assert profile.time_column == "dt"
    assert profile.dimensions == {
        "store": ("centro", "praia", "shopping"),
        "kind": ("fisica", "online"),
    }
    assert profile.measures == ("clicks", "orders", "amount")
    # pedidos <= cliques em todas as linhas e os dois são contagens: é uma taxa descoberta
    assert profile.ratios["orders_per_clicks"].is_rate
    assert profile.ratios["orders_per_clicks"].source == "discovered"
    assert "amount_per_orders" not in profile.ratios  # valor não é contagem
    assert profile.metrics[0] == "clicks"
    assert profile.volume == "clicks"


def test_the_detector_and_the_drilldown_run_on_another_business():
    rng = np.random.default_rng(7)
    frame = _shop(rng)
    hit = (frame["store"] == "praia") & (frame["dt"] >= "2026-03-20")
    frame.loc[hit, "orders"] = (frame.loc[hit, "orders"] * 0.4).round().astype(int)
    frame["amount"] = frame["orders"] * 50.0
    profile = profile_from_frame(frame)
    series = daily_series(frame, profile)
    root = select_root(detect(series), profile.metrics)
    assert root is not None and root.metric == "orders_per_clicks"
    cube = Cube.from_frame(frame, profile)
    [cause] = find_causes(cube, root.metric, Window(root.window_start, root.window_end))
    assert cause.dimensions == {"store": "praia"}


def test_high_cardinality_text_is_not_a_dimension_until_proposed():
    frame = _shop(np.random.default_rng(2))
    frame["order_code"] = [f"c{i % 300}" for i in range(len(frame))]
    profile = profile_from_frame(frame)
    assert "order_code" not in profile.dimensions
    [skipped] = profile.skipped
    assert (skipped.column, skipped.reason, skipped.distinct) == (
        "order_code",
        "cardinalidade alta",
        300,
    )
    with pytest.raises(ProfileError, match="valores demais"):
        propose_dimension(profile, frame, "order_code")

    frame["order_code"] = [f"c{i % 80}" for i in range(len(frame))]
    profile = profile_from_frame(frame)
    widened = propose_dimension(profile, frame, "order_code")
    assert len(widened.dimensions["order_code"]) == 80
    assert widened.fingerprint != profile.fingerprint


def test_a_proposed_ratio_is_validated_by_the_code(events_of):
    frame = events_of("s01")
    profile = propose_ratio(DEFAULT_PROFILE, frame, "revenue_per_session", "revenue", "sessions")
    assert profile.ratios["revenue_per_session"].source == "proposed"
    assert not profile.ratios["revenue_per_session"].is_rate
    assert "revenue_per_session" in profile.metrics
    with pytest.raises(ProfileError, match="já existe"):
        propose_ratio(DEFAULT_PROFILE, frame, "conversion_rate", "conversions", "sessions")
    with pytest.raises(ProfileError, match="não é uma medida"):
        propose_ratio(DEFAULT_PROFILE, frame, "x_rate", "region", "sessions")
    with pytest.raises(ProfileError, match="iguais"):
        propose_ratio(DEFAULT_PROFILE, frame, "same", "sessions", "sessions")


def test_fingerprint_and_diff_track_structure_changes(events_of):
    frame = events_of("s01").copy()
    before = profile_from_frame(frame)
    frame.loc[frame.index[:10], "region"] = "exterior"
    after = profile_from_frame(frame)
    assert before.fingerprint != after.fingerprint
    assert after.diff(before.structure()) == ["region: valor(es) novo(s) exterior"]
    assert before.diff(before.structure()) == []


@pytest.mark.parametrize(
    ("frame", "fragment"),
    [
        (pd.DataFrame(), "vazia"),
        (pd.DataFrame({"x": ["a"], "n": [1]}), "data"),
        (pd.DataFrame({"Day": ["2026-01-01"], "n": [1]}), "padrão"),
    ],
)
def test_unusable_tables_are_refused(frame, fragment):
    with pytest.raises(ProfileError, match=fragment):
        profile_from_frame(frame)


def test_values_with_control_characters_are_not_dimension_values():
    frame = pd.DataFrame(
        {"day": ["2026-01-01", "2026-01-02"], "tag": ["ok", "\x1b[31mmal"], "n": [1, 2]}
    )
    profile = profile_from_frame(frame)
    assert "tag" not in profile.dimensions
    assert profile.skipped[0].reason == "valores vazios ou inválidos"


def test_unknown_metric_and_dimension_are_refused():
    with pytest.raises(ProfileError, match="métrica desconhecida"):
        DEFAULT_PROFILE.check_metric("profit")
    with pytest.raises(ProfileError, match="dimensão desconhecida"):
        DEFAULT_PROFILE.check_dimensions({"color": "red"})
    with pytest.raises(ProfileError, match="valor inválido"):
        DEFAULT_PROFILE.check_dimensions({"region": "marte"})


def test_the_events_table_name_is_never_taken_from_outside(tmp_path):
    from sentinel.data.profile import load_events

    path = tmp_path / "x.db"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE events (day TEXT, n INTEGER)")
    con.close()
    con = connect_readonly(path)
    try:
        with pytest.raises(ProfileError, match="tabela"):
            load_events(con, 'events"; DROP TABLE events; --')
    finally:
        con.close()
