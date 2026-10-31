from datetime import date

import numpy as np
import pytest

from sentinel.data.generator import generate_events, load_scenarios
from sentinel.detection.detector import DetectorParams, daily_series, detect, select_root
from sentinel.investigation.cube import Cube
from sentinel.investigation.drilldown import (
    DrillParams,
    Window,
    dims_table,
    evaluate_hypothesis,
    find_causes,
    refinement_candidates,
    single_dim_candidates,
    total_is_significant,
)

from .conftest import SCENARIOS_FILE

PARAMS = DrillParams()
ALL = {s.id: s for s in load_scenarios(SCENARIOS_FILE)}
METRICS = ("sessions", "conversions", "conversion_rate", "avg_ticket", "revenue")
ADDITIVE = ("sessions", "conversions", "revenue")
SINGLE_CAUSE_IDS = [f"s{i:02d}" for i in [*range(1, 12), *range(13, 15), 20]]
NORMAL_IDS = ["s15", "s16", "s17", "s18"]
# A métrica em que a causa de cada cenário se mostra (a raiz que o detector costuma escolher).
METRIC_OF = {"s05": "sessions", "s08": "sessions", "s09": "sessions", "s14": "sessions"}


def window_of(scenario_id: str) -> Window:
    return Window(*ALL[scenario_id].window)


def metric_of(scenario_id: str) -> str:
    return METRIC_OF.get(scenario_id, "conversion_rate")


def as_set(dimension_dicts) -> set[frozenset]:
    return {frozenset(d.items()) for d in dimension_dicts}


# --- Contas do poder de explicação -----------------------------------------------------------


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("dims", [("region",), ("platform", "app_version"), ("product", "channel")])
def test_impacts_add_up_to_the_total_change(metric, dims, cube_of):
    table = dims_table(cube_of("s10"), metric, dims, window_of("s10"), PARAMS)
    assert table.impact.sum() == pytest.approx(table.total_change, rel=1e-9, abs=1e-12)
    if metric in ADDITIVE:
        assert table.ep.sum() == pytest.approx(1.0)


def test_ratio_totals_are_the_ratio_of_the_sums(cube_of, events_of):
    table = dims_table(cube_of("s06"), "conversion_rate", ("product",), window_of("s06"), PARAMS)
    df = events_of("s06")
    inside = df[df["day"].between("2026-03-26", "2026-03-31")]
    assert table.total_obs == pytest.approx(inside["conversions"].sum() / inside["sessions"].sum())


def test_mix_effect_is_part_of_the_impact(cube_of):
    """s08: um canal perde sessões; as taxas por canal quase não mudam, mas a taxa total muda."""
    table = dims_table(cube_of("s08"), "conversion_rate", ("channel",), window_of("s08"), PARAMS)
    paid = table.labels.index(("paid_search",))
    assert abs(table.change_pct[paid]) < 0.05  # a taxa do canal praticamente não mudou
    assert table.impact[paid] != 0
    assert table.impact.sum() == pytest.approx(table.total_change)
    assert np.sign(table.impact.sum()) == np.sign(table.impact[paid])  # a mistura puxa a taxa


def test_slice_z_is_calibrated_in_pure_noise(cube_of):
    """Em dados sem causa, o z das fatias é ~N(0, 1): quase nada passa de 3,5."""
    zs = []
    for dim in ("region", "platform", "app_version", "product", "channel"):
        table = dims_table(cube_of("s15"), "conversion_rate", (dim,), window_of("s15"), PARAMS)
        zs.extend(table.z)
    assert np.abs(zs).max() < 3.5
    assert 0.5 < np.std(zs) < 1.6


def test_window_needs_enough_history(cube_of):
    with pytest.raises(ValueError, match="histórico insuficiente"):
        dims_table(
            cube_of("s01"),
            "sessions",
            ("region",),
            Window(date(2026, 2, 1), date(2026, 2, 5)),
            PARAMS,
        )


def test_window_outside_the_data_is_an_error(cube_of):
    with pytest.raises(ValueError, match="fora dos dados"):
        dims_table(
            cube_of("s01"),
            "sessions",
            ("region",),
            Window(date(2026, 4, 1), date(2026, 4, 3)),
            PARAMS,
        )


# --- Busca de causas -------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_id", SINGLE_CAUSE_IDS)
def test_finds_the_injected_cause(scenario_id, cube_of):
    causes = find_causes(
        cube_of(scenario_id), metric_of(scenario_id), window_of(scenario_id), PARAMS
    )
    truth = [c.dimensions for c in ALL[scenario_id].causes]
    assert as_set(c.dimensions for c in causes) == as_set(truth)
    assert all(0.5 <= c.explained_share <= 1.0 for c in causes)


@pytest.mark.parametrize("metric", ["conversion_rate", "avg_ticket", "revenue"])
def test_price_change_is_found_through_any_of_its_metrics(metric, cube_of):
    causes = find_causes(cube_of("s10"), metric, window_of("s10"), PARAMS)
    assert as_set(c.dimensions for c in causes) == {frozenset({("product", "laptop")})}


def test_two_independent_causes_are_both_returned(cube_of):
    causes = find_causes(cube_of("s12"), "conversion_rate", window_of("s12"), PARAMS)
    assert as_set(c.dimensions for c in causes) == {
        frozenset({("region", "sul")}),
        frozenset({("product", "phone")}),
    }
    assert all(c.explained_share >= PARAMS.ep_multi_each for c in causes)


def test_diffuse_drop_is_inconclusive(cube_of):
    assert find_causes(cube_of("s19"), "conversion_rate", window_of("s19"), PARAMS) == []


@pytest.mark.parametrize("scenario_id", NORMAL_IDS)
@pytest.mark.parametrize("metric", METRICS)
def test_noise_never_becomes_a_cause(scenario_id, metric, cube_of):
    assert find_causes(cube_of(scenario_id), metric, window_of(scenario_id), PARAMS) == []


def test_total_change_must_leave_the_noise(cube_of):
    assert total_is_significant(cube_of("s01"), "conversion_rate", window_of("s01"), PARAMS)
    assert not total_is_significant(cube_of("s15"), "conversion_rate", window_of("s15"), PARAMS)


def test_most_specific_hypothesis_wins(cube_of):
    cube, window = cube_of("s01"), window_of("s01")
    candidates = single_dim_candidates(cube, "conversion_rate", window, PARAMS)
    assert as_set(c.dimensions for c in candidates) == {
        frozenset({("platform", "android")}),
        frozenset({("app_version", "3.2")}),
    }
    refined = refinement_candidates(
        cube, "conversion_rate", window, PARAMS, {"platform": "android"}, 1.0
    )
    assert as_set(c.dimensions for c in refined) == {
        frozenset({("platform", "android"), ("app_version", "3.2")})
    }
    # Sem refinamento possível, fica a hipótese de uma dimensão.
    strict = DrillParams(refine_keep=5.0)
    [cause] = find_causes(cube, "conversion_rate", window, strict)
    assert len(cause.dimensions) == 1


def test_tiny_slices_are_never_candidates(cube_of):
    huge = DrillParams(min_slice_sessions=10**9)
    assert find_causes(cube_of("s01"), "conversion_rate", window_of("s01"), huge) == []


def test_one_valued_dimension_is_not_a_cause(cube_of):
    """Depois de filtrar para um valor só, essa dimensão explica 100% por definição: não vale."""
    cube = cube_of("s19").subset(cube_of("s19").mask({"platform": "ios"}))
    candidates = single_dim_candidates(cube, "conversion_rate", window_of("s19"), PARAMS)
    assert all("platform" not in c.dimensions for c in candidates)


# --- Avaliar uma hipótese dada ---------------------------------------------------------------


def test_true_hypothesis_is_accepted(cube_of):
    ev = evaluate_hypothesis(
        cube_of("s01"),
        "conversion_rate",
        window_of("s01"),
        PARAMS,
        {"platform": "android", "app_version": "3.2"},
    )
    assert ev.accepted and ev.others_stable and ev.total_significant
    assert ev.explained_share == pytest.approx(1.0, abs=0.1)
    assert ev.inside.change_pct < -0.3
    assert abs(ev.outside.change_pct) < 0.02


def test_superset_is_accepted_but_points_to_the_refinement(cube_of):
    ev = evaluate_hypothesis(
        cube_of("s01"), "conversion_rate", window_of("s01"), PARAMS, {"platform": "android"}
    )
    assert ev.accepted
    assert [r.dimensions for r in ev.refinements] == [{"platform": "android", "app_version": "3.2"}]


@pytest.mark.parametrize(
    "dimensions",
    [{"platform": "ios"}, {"region": "sul"}, {"product": "watch", "channel": "email"}],
)
def test_wrong_hypotheses_are_rejected(dimensions, cube_of):
    ev = evaluate_hypothesis(
        cube_of("s01"), "conversion_rate", window_of("s01"), PARAMS, dimensions
    )
    assert not ev.accepted


def test_nothing_is_accepted_when_the_total_did_not_move(cube_of):
    ev = evaluate_hypothesis(
        cube_of("s15"), "conversion_rate", window_of("s15"), PARAMS, {"platform": "ios"}
    )
    assert not ev.total_significant and not ev.accepted


# --- Ponta a ponta com o detector ------------------------------------------------------------


@pytest.mark.parametrize("scenario_id", [s for s in SINGLE_CAUSE_IDS if s != "s20"])
def test_detector_then_drilldown_finds_the_cause(scenario_id, events_of, cube_of):
    root = select_root(detect(daily_series(events_of(scenario_id)), DetectorParams()))
    assert root is not None
    window = Window(root.window_start, root.window_end)
    causes = find_causes(cube_of(scenario_id), root.metric, window, PARAMS)
    truth = [c.dimensions for c in ALL[scenario_id].causes]
    assert as_set(c.dimensions for c in causes) == as_set(truth)


@pytest.mark.slow
def test_accuracy_over_40_seeds():
    """Sementes de teste 0–39: a causa certa em quase todas e nunca uma causa inventada."""
    right = total = 0
    for scenario_id in SINGLE_CAUSE_IDS + ["s12"]:
        if scenario_id == "s20":
            continue  # a deriva nem sempre é detectada; é medida no benchmark
        for seed in range(40):
            events = generate_events(ALL[scenario_id], seed)
            root = select_root(detect(daily_series(events), DetectorParams()))
            total += 1
            if root is None:
                continue
            causes = find_causes(
                Cube.from_frame(events), root.metric, Window(root.window_start, root.window_end)
            )
            right += as_set(c.dimensions for c in causes) == as_set(
                c.dimensions for c in ALL[scenario_id].causes
            )
    assert right / total >= 0.97

    spurious = 0
    for scenario_id in [*NORMAL_IDS, "s19"]:
        for seed in range(40):
            cube = Cube.from_frame(generate_events(ALL[scenario_id], seed))
            spurious += bool(find_causes(cube, "conversion_rate", window_of(scenario_id)))
    assert spurious == 0
