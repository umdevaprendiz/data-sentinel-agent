"""Policy (etapa 8): cada condição de L3 tem um teste; qualquer dúvida cai para ask_human."""

import pytest

from sentinel.agent.policy import PolicyConfig, decide
from sentinel.records import History
from sentinel.schemas import ActionProposal, Hypothesis, Report, Statement

ROLLBACK = ActionProposal(
    action_id="rollback_deploy",
    params={"platform": "android", "app_version": "3.2"},
    reversible=True,
    risk="low",
)
GOOD_HISTORY = History(approved=19, ignored=1)
L3 = PolicyConfig(autonomy_level=3, dry_run=False)


def report(share: float = 0.95, status: str = "identified") -> Report:
    causes = (
        [Hypothesis(dimensions={"platform": "android"}, explained_share=share, finding_ids=["f1"])]
        if status == "identified"
        else []
    )
    return Report(
        status=status,
        facts=[Statement(text="fato", finding_id="f1")],
        causes=causes,
        opinions=[],
        evidence=["f1"],
    )


def test_all_conditions_met_executes_alone():
    decision = decide(report(), ROLLBACK, GOOD_HISTORY, L3)
    assert decision.kind == "auto_execute"
    assert decision.dry_run is False


def test_dry_run_still_decides_auto_but_only_registers():
    decision = decide(report(), ROLLBACK, GOOD_HISTORY, L3.model_copy(update={"dry_run": True}))
    assert decision.kind == "auto_execute"
    assert decision.dry_run is True


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"report": report(share=0.7)}, "poder de explicação"),
        ({"verified": False}, "verificador"),
        ({"history": History(approved=5)}, "histórico curto"),
        ({"history": History(approved=8, ignored=4)}, "precisão histórica"),
        ({"history": GOOD_HISTORY.model_copy(update={"auto_last_hour": 2})}, "por hora"),
        (
            {
                "proposal": ActionProposal(
                    action_id="restore_price",
                    params={"product": "laptop"},
                    reversible=True,
                    risk="medium",
                )
            },
            "risco medium",
        ),
    ],
)
def test_each_failed_condition_falls_back_to_a_human(change, fragment):
    args = {"report": report(), "proposal": ROLLBACK, "history": GOOD_HISTORY, "verified": True}
    args.update(change)
    decision = decide(
        args["report"], args["proposal"], args["history"], L3, verified=args["verified"]
    )
    assert decision.kind == "ask_human"
    assert any(fragment in reason for reason in decision.reasons)


def test_levels_one_and_two():
    l1 = decide(report(), ROLLBACK, GOOD_HISTORY, PolicyConfig(autonomy_level=1))
    l2 = decide(report(), ROLLBACK, GOOD_HISTORY, PolicyConfig(autonomy_level=2))
    assert (l1.kind, l2.kind) == ("report_only", "ask_human")


def test_kill_switch_blocks_even_a_perfect_case():
    config = L3.model_copy(update={"kill_switch": True})
    assert decide(report(), ROLLBACK, GOOD_HISTORY, config).kind == "block"


def test_no_action_or_inconclusive_is_report_only():
    assert decide(report(), None, GOOD_HISTORY, L3).kind == "report_only"
    assert decide(report(status="inconclusive"), ROLLBACK, GOOD_HISTORY, L3).kind == "report_only"


def test_an_action_outside_the_catalog_or_tampered_is_blocked():
    rogue = ActionProposal(action_id="drop_database", params={}, reversible=True, risk="low")
    tampered = ROLLBACK.model_copy(update={"risk": "low", "reversible": False})
    assert decide(report(), rogue, GOOD_HISTORY, L3).kind == "block"
    assert decide(report(), tampered, GOOD_HISTORY, L3).kind == "block"


def test_a_duplicate_is_only_reported():
    decision = decide(report(), ROLLBACK, GOOD_HISTORY, L3, duplicate_of="inc-1")
    assert decision.kind == "report_only"
    assert "inc-1" in decision.reasons[0]
