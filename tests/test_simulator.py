"""Catálogo de ações e simulador de operações (etapa 8)."""

import pytest

from sentinel.investigation.actions_catalog import CATALOG, proposal_for
from sentinel.ops.simulator import OpsSimulator, SimulatorError
from sentinel.schemas import ActionProposal


def action(action_id: str, **params: str) -> ActionProposal:
    spec = CATALOG[action_id]
    return ActionProposal(
        action_id=action_id, params=params, reversible=spec.reversible, risk=spec.risk
    )


@pytest.fixture
def world_of(scenarios, seed, tmp_path):
    def get(scenario_id: str) -> OpsSimulator:
        sim = OpsSimulator.load(tmp_path / scenario_id)
        sim.reset(scenarios[scenario_id], seed)
        return sim

    return get


def test_the_catalog_is_closed_and_every_action_is_reversible():
    assert set(CATALOG) == {
        "rollback_deploy",
        "restore_price",
        "resume_campaign",
        "restock_alert",
        "open_ticket",
    }
    assert all(spec.reversible for spec in CATALOG.values())


@pytest.mark.parametrize(
    ("scenario_id", "action_id", "params"),
    [
        ("s01", "rollback_deploy", {"platform": "android", "app_version": "3.2"}),
        ("s03", "rollback_deploy", {"platform": "android", "app_version": "3.2"}),
        ("s05", "rollback_deploy", {"region": "sul"}),
        ("s06", "restock_alert", {"product": "watch"}),
        ("s08", "resume_campaign", {"channel": "paid_search"}),
        ("s10", "restore_price", {"product": "laptop"}),
        ("s12", "rollback_deploy", {"region": "sul"}),
        ("s13", "open_ticket", {"product": "headset"}),
    ],
)
def test_the_plan_proposes_the_catalog_action_for_the_cause(
    scenario_id, action_id, params, plan_of
):
    anomaly, hypotheses, findings = plan_of(scenario_id)
    proposal = proposal_for(hypotheses, findings, anomaly.direction)
    assert proposal is not None
    assert (proposal.action_id, proposal.params) == (action_id, params)


def test_spikes_and_inconclusive_cases_get_no_action(plan_of):
    for scenario_id in ("s14", "s19"):
        anomaly, hypotheses, findings = plan_of(scenario_id)
        assert proposal_for(hypotheses, findings, anomaly.direction) is None


@pytest.mark.parametrize(
    ("scenario_id", "proposal", "outcome"),
    [
        ("s01", action("rollback_deploy", platform="android", app_version="3.2"), "correct"),
        ("s03", action("rollback_deploy", platform="ios", app_version="3.0"), "harmful"),
        ("s17", action("rollback_deploy", platform="ios", app_version="3.1"), "harmful"),
        ("s06", action("restock_alert", product="watch"), "correct"),
        ("s06", action("restock_alert", product="phone"), "innocuous"),
        ("s08", action("resume_campaign", channel="paid_search"), "correct"),
        ("s08", action("resume_campaign", channel="email"), "innocuous"),
        ("s10", action("restore_price", product="laptop"), "correct"),
        ("s10", action("restore_price", product="tablet"), "innocuous"),
        ("s13", action("open_ticket", product="headset"), "innocuous"),
        ("s01", action("rollback_deploy", region="norte"), "innocuous"),
    ],
)
def test_every_action_is_classified_against_the_injected_cause(
    scenario_id, proposal, outcome, world_of
):
    assert world_of(scenario_id).classify(proposal)[0] == outcome


def test_execution_changes_the_state_and_can_be_reverted(world_of):
    sim = world_of("s01")
    rollback = action("rollback_deploy", platform="android", app_version="3.2")
    execution = sim.execute("inc-x", rollback, dry_run=False, world_fingerprint=sim.fingerprint)
    assert execution.applied and execution.outcome == "correct"
    assert sim.world.deploys[0].status == "rolled_back"
    reloaded = OpsSimulator.load(sim.path.parent)
    assert reloaded.world.deploys[0].status == "rolled_back"
    reloaded.revert(execution.execution_id)
    assert reloaded.world.deploys[0].status == "live"
    with pytest.raises(SimulatorError, match="nada a reverter"):
        reloaded.revert(execution.execution_id)


def test_dry_run_only_registers(world_of):
    sim = world_of("s06")
    execution = sim.execute(
        "inc-x", action("restock_alert", product="watch"), dry_run=True, world_fingerprint=None
    )
    assert not execution.applied
    assert sim.world.in_stock["watch"] is False
    assert "dry-run" in execution.effect


def test_an_action_on_a_changed_world_is_refused(world_of, scenarios, seed):
    sim = world_of("s01")
    old = sim.fingerprint
    sim.reset(scenarios["s06"], seed)
    with pytest.raises(SimulatorError, match="mundo simulado mudou"):
        sim.execute(
            "inc-x",
            action("rollback_deploy", platform="android", app_version="3.2"),
            dry_run=False,
            world_fingerprint=old,
        )


def test_a_simulator_without_world_asks_for_generate(tmp_path):
    with pytest.raises(SimulatorError, match="sentinel generate"):
        OpsSimulator.load(tmp_path).classify(action("open_ticket"))
