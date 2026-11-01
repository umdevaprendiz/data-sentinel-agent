import sqlite3

import pandas as pd
import pytest

from sentinel.config import Settings
from sentinel.data.db import connect_readonly
from sentinel.investigation.tools import TOOLS, ToolContext, tool_catalog

WINDOW = {"window_start": "2026-03-26", "window_end": "2026-03-31"}
EXPECTED_TOOLS = {
    "compare_periods",
    "split_by",
    "top_contributors",
    "test_hypothesis",
    "decompose_metric",
    "check_recent_changes",
    "check_data_quality",
}


def test_menu_is_closed_and_described():
    assert set(TOOLS) == EXPECTED_TOOLS
    catalog = tool_catalog()
    assert {t["name"] for t in catalog} == EXPECTED_TOOLS
    for entry in catalog:
        assert entry["description"]
        assert entry["parameters"]["type"] == "object"


def test_findings_are_numbered_and_keep_their_params(ctx_of):
    ctx = ctx_of("s05")
    first = ctx.run("compare_periods", {**WINDOW, "metric": "sessions"})
    second = ctx.run("decompose_metric", WINDOW)
    assert (first.id, second.id) == ("f001", "f002")
    assert ctx.findings == [first, second]
    assert first.tool == "compare_periods"
    assert first.params["window_start"] == "2026-03-26"  # datas viram texto: o Finding é JSON
    assert first.params["metric"] == "sessions"


def test_compare_periods_matches_the_raw_data(ctx_of, events_of):
    result = ctx_of("s05").run("compare_periods", {**WINDOW, "metric": "sessions"}).result
    df = events_of("s05")
    observed = df[df["day"].between("2026-03-26", "2026-03-31")]["sessions"].sum()
    assert result["observed"] == pytest.approx(observed)
    assert result["direction"] == "drop"
    assert result["change_pct"] == pytest.approx(-0.12, abs=0.03)
    assert result["z"] < -5


def test_split_by_ranks_slices_by_impact(ctx_of):
    result = (
        ctx_of("s05")
        .run("split_by", {**WINDOW, "metric": "sessions", "dimension": "region"})
        .result
    )
    slices = result["slices"]
    assert slices[0]["dimensions"] == {"region": "sul"}
    assert slices[0]["explained_share"] == pytest.approx(1.0, abs=0.2)  # o ruído das outras fatias
    assert {s["dimensions"]["region"] for s in slices} == {
        "norte",
        "nordeste",
        "centro_oeste",
        "sudeste",
        "sul",
    }
    impacts = [abs(s["impact"]) for s in slices]
    assert impacts == sorted(impacts, reverse=True)
    assert sum(s["impact"] for s in slices) == pytest.approx(result["total_change"])


def test_top_contributors_with_two_dimensions(ctx_of):
    request = {
        **WINDOW,
        "metric": "conversion_rate",
        "dimensions": ["platform", "app_version"],
        "k": 3,
    }
    result = ctx_of("s01").run("top_contributors", request).result
    assert len(result["slices"]) == 3
    assert result["slices"][0]["dimensions"] == {"platform": "android", "app_version": "3.2"}
    assert result["slices"][0]["explained_share"] > 0.9


def test_test_hypothesis_accepts_the_cause_and_rejects_the_rest(ctx_of):
    ctx = ctx_of("s01")
    base = {**WINDOW, "metric": "conversion_rate"}
    true = ctx.run(
        "test_hypothesis", {**base, "dimensions": {"platform": "android", "app_version": "3.2"}}
    )
    wrong = ctx.run("test_hypothesis", {**base, "dimensions": {"platform": "ios"}})
    superset = ctx.run("test_hypothesis", {**base, "dimensions": {"platform": "android"}})
    assert true.result["accepted"] is True
    assert wrong.result["accepted"] is False
    assert superset.result["accepted"] is True
    assert superset.result["refinements"][0]["dimensions"] == {
        "platform": "android",
        "app_version": "3.2",
    }


def test_two_independent_causes_are_accepted_together_but_not_alone(ctx_of):
    ctx = ctx_of("s12")
    base = {**WINDOW, "metric": "conversion_rate"}
    sul, phone = {"region": "sul"}, {"product": "phone"}
    alone = ctx.run("test_hypothesis", {**base, "dimensions": sul}).result
    joint = ctx.run("test_hypothesis", {**base, "dimensions": sul, "together_with": phone}).result
    wrong = ctx.run(
        "test_hypothesis", {**base, "dimensions": sul, "together_with": {"product": "tablet"}}
    ).result
    assert alone["accepted"] is False  # sozinha, explica menos que o limiar de causa única
    assert joint["accepted"] is True
    assert joint["together_with"] == phone
    assert joint["explained_share"] >= joint["threshold"]
    assert [m["dimensions"] for m in joint["members"]] == [sul, phone]
    assert all(m["explained_share"] >= 0.3 for m in joint["members"])
    assert wrong["accepted"] is False


def test_a_hypothesis_cannot_be_tested_together_with_itself(ctx_of):
    request = {
        **WINDOW,
        "metric": "conversion_rate",
        "dimensions": {"region": "sul"},
        "together_with": {"region": "sul"},
    }
    finding = ctx_of("s12").run("test_hypothesis", request)
    assert "together_with" in finding.result["error"]


def test_decompose_metric_names_the_factor_that_moved(ctx_of):
    sessions = ctx_of("s05").run("decompose_metric", WINDOW).result
    rate = ctx_of("s06").run("decompose_metric", WINDOW).result
    assert sessions["dominant_factor"] == "sessions"
    assert rate["dominant_factor"] == "conversion_rate"
    for result in (sessions, rate):
        total = sum(f["log_share"] for f in result["factors"].values())
        assert total == pytest.approx(1.0, abs=1e-4)  # revenue = sessões × taxa × ticket, em log


def test_check_recent_changes_filters_by_scope(ctx_of):
    ctx = ctx_of("s03")
    everything = ctx.run("check_recent_changes", WINDOW).result
    assert everything["count"] == 2  # a causa e a isca
    scoped = ctx.run(
        "check_recent_changes", {**WINDOW, "scope": {"platform": "android", "app_version": "3.2"}}
    ).result
    assert [c["description"] for c in scoped["changes"]] == ["Release 3.2 no Android"]
    assert scoped["changes"][0]["days_from_window_start"] == 0


def test_scope_needs_a_dimension_in_common(ctx_of):
    """O deploy do Sul não é evidência para a hipótese 'phone': escopos sem dimensão em comum."""
    ctx = ctx_of("s12")
    sul = ctx.run("check_recent_changes", {**WINDOW, "scope": {"region": "sul"}}).result
    phone = ctx.run("check_recent_changes", {**WINDOW, "scope": {"product": "phone"}}).result
    assert sul["count"] == 1
    assert phone["count"] == 0


def test_check_recent_changes_with_nothing_registered(ctx_of):
    assert ctx_of("s13").run("check_recent_changes", WINDOW).result["count"] == 0


def test_check_recent_changes_lookback(ctx_of):
    narrow = ctx_of("s03").run("check_recent_changes", {**WINDOW, "lookback_days": 0}).result
    assert narrow["count"] == 1  # o hotfix de 25/03 fica de fora


def test_data_quality_on_clean_and_zeroed_data(ctx_of):
    clean = ctx_of("s15").run("check_data_quality", WINDOW).result
    assert clean == {
        "missing_days": [],
        "days_with_fewer_rows": [],
        "duplicate_rows": 0,
        "null_values": 0,
        "zero_session_rows": 0,
        "zeroed_slices": [],
    }
    stock_out = ctx_of("s06").run("check_data_quality", WINDOW).result
    assert stock_out["zeroed_slices"] == [
        {"dimension": "product", "value": "watch", "metric": "conversions"}
    ]


def test_data_quality_finds_missing_duplicated_and_null_data(events_of, tmp_path):
    df = events_of("s15").copy()
    df = df[df["day"] != "2026-03-28"]  # um dia some
    df = pd.concat([df, df.iloc[-5:]])  # 5 linhas duplicadas
    df.loc[df.index[0], "revenue"] = None
    df["day"] = df["day"].astype(str)
    path = tmp_path / "broken.db"
    with sqlite3.connect(path) as con:
        df.to_sql("events", con, index=False)
        con.execute("CREATE TABLE changes (ts TEXT, type TEXT, description TEXT, scope TEXT)")
    con.close()

    ctx = ToolContext(connect_readonly(path), Settings(_env_file=None))
    result = ctx.run("check_data_quality", WINDOW).result
    ctx.con.close()
    assert result["missing_days"] == ["2026-03-28"]
    assert result["duplicate_rows"] == 5
    assert result["null_values"] == 1


@pytest.mark.parametrize(
    ("tool", "params", "fragment"),
    [
        ("compare_periods", {**WINDOW, "metric": "profit"}, "metric"),
        ("compare_periods", {**WINDOW, "metric": "sessions", "extra": 1}, "extra"),
        (
            "compare_periods",
            {"window_start": "2026-03-31", "window_end": "2026-03-26", "metric": "sessions"},
            "window_end",
        ),
        (
            "compare_periods",
            {"window_start": "2026-04-01", "window_end": "2026-04-03", "metric": "sessions"},
            "fora dos dados",
        ),
        (
            "compare_periods",
            {"window_start": "2026-01-05", "window_end": "2026-01-10", "metric": "sessions"},
            "histórico insuficiente",
        ),
        ("split_by", {**WINDOW, "metric": "sessions", "dimension": "color"}, "dimension"),
        (
            "top_contributors",
            {**WINDOW, "metric": "sessions", "dimensions": ["region", "region"]},
            "repetidas",
        ),
        (
            "top_contributors",
            {**WINDOW, "metric": "sessions", "dimensions": ["region", "platform", "channel"]},
            "dimensions",
        ),
        (
            "test_hypothesis",
            {**WINDOW, "metric": "sessions", "dimensions": {"app_version": "9.9"}},
            "9.9",
        ),
        ("check_recent_changes", {**WINDOW, "scope": {"region": "marte"}}, "marte"),
        ("rm_rf", {}, "ferramenta desconhecida"),
    ],
)
def test_bad_requests_become_error_findings(tool, params, fragment, ctx_of):
    ctx = ctx_of("s01")
    finding = ctx.run(tool, params)
    assert finding.id == "f001"
    assert fragment in finding.result["error"]
    assert list(finding.result) == ["error"]


def test_a_bad_request_does_not_stop_the_next_one(ctx_of):
    ctx = ctx_of("s01")
    ctx.run("compare_periods", {**WINDOW, "metric": "profit"})
    ok = ctx.run("compare_periods", {**WINDOW, "metric": "sessions"})
    assert ok.id == "f002" and "error" not in ok.result


def test_tools_cannot_change_the_data(ctx_of):
    ctx = ctx_of("s01")
    ctx.run("decompose_metric", WINDOW)
    with pytest.raises(sqlite3.OperationalError):
        ctx.con.execute("DELETE FROM events")
    count = ctx.con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert count == 90 * 720


def test_cube_is_loaded_once_per_context(ctx_of):
    ctx = ctx_of("s01")
    ctx.run("compare_periods", {**WINDOW, "metric": "sessions"})
    cube = ctx.cube
    ctx.run("decompose_metric", WINDOW)
    assert ctx.cube is cube


def test_window_can_use_a_different_baseline(ctx_of):
    short = (
        ctx_of("s05")
        .run("compare_periods", {**WINDOW, "metric": "sessions", "baseline_weeks": 4})
        .result
    )
    default = ctx_of("s05").run("compare_periods", {**WINDOW, "metric": "sessions"}).result
    assert short["expected"] != default["expected"]
    assert short["change_pct"] == pytest.approx(default["change_pct"], abs=0.03)
