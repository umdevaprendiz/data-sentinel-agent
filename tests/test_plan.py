import pytest

from sentinel.detection.detector import DetectorParams, daily_series, detect, select_root
from sentinel.investigation.plan import investigate


def root_of(events):
    return select_root(detect(daily_series(events), DetectorParams()))


def test_plan_confirms_the_cause_and_leaves_the_evidence(ctx_of, events_of, scenarios):
    ctx = ctx_of("s01")
    [hypothesis] = investigate(ctx, root_of(events_of("s01")))
    assert hypothesis.dimensions == scenarios["s01"].causes[0].dimensions
    assert hypothesis.explained_share > 0.9

    by_id = {f.id: f for f in ctx.findings}
    assert set(hypothesis.finding_ids) <= set(by_id)
    assert [by_id[i].tool for i in hypothesis.finding_ids] == [
        "test_hypothesis",
        "check_recent_changes",
    ]
    assert by_id[hypothesis.finding_ids[0]].result["accepted"] is True
    assert by_id[hypothesis.finding_ids[1]].result["count"] == 1  # o deploy registrado


def test_plan_with_two_causes(ctx_of, events_of):
    ctx = ctx_of("s12")
    hypotheses = investigate(ctx, root_of(events_of("s12")))
    assert {frozenset(h.dimensions.items()) for h in hypotheses} == {
        frozenset({("region", "sul")}),
        frozenset({("product", "phone")}),
    }
    by_id = {f.id: f for f in ctx.findings}
    proofs = {h.finding_ids[0] for h in hypotheses}
    assert len(proofs) == 1  # duas causas independentes se provam juntas
    [proof] = (by_id[i] for i in proofs)
    assert proof.result["accepted"] is True
    assert {frozenset(m["dimensions"].items()) for m in proof.result["members"]} == {
        frozenset({("region", "sul")}),
        frozenset({("product", "phone")}),
    }
    assert len({i for h in hypotheses for i in h.finding_ids}) == 3
    registered = {
        h.dimensions and next(iter(h.dimensions.values())): by_id[h.finding_ids[1]].result["count"]
        for h in hypotheses
    }
    assert registered == {"sul": 1, "phone": 0}  # só a causa regional tem mudança registrada


def test_plan_is_inconclusive_when_nothing_explains_the_change(ctx_of, events_of):
    ctx = ctx_of("s19")
    assert investigate(ctx, root_of(events_of("s19"))) == []
    assert [f.tool for f in ctx.findings][-1] == "check_recent_changes"


def test_every_finding_of_the_plan_is_valid(ctx_of, events_of):
    ctx = ctx_of("s06")
    investigate(ctx, root_of(events_of("s06")))
    assert ctx.findings
    assert all("error" not in f.result for f in ctx.findings)
    assert [f.id for f in ctx.findings] == [f"f{i:03d}" for i in range(1, len(ctx.findings) + 1)]


@pytest.mark.parametrize("scenario_id", ["s05", "s08", "s14"])
def test_plan_works_for_session_metrics(scenario_id, ctx_of, events_of, scenarios):
    anomaly = root_of(events_of(scenario_id))
    assert anomaly.metric == "sessions"
    [hypothesis] = investigate(ctx_of(scenario_id), anomaly)
    assert hypothesis.dimensions == scenarios[scenario_id].causes[0].dimensions
