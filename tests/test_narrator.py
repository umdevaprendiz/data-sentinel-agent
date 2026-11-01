import pytest

from sentinel.agent.narrator import narrate
from sentinel.agent.render import render_report
from sentinel.agent.verifier import verify


@pytest.mark.parametrize("scenario_id", ["s01", "s03", "s06", "s08", "s12", "s19"])
def test_the_template_report_always_passes_the_verifier(scenario_id, plan_of):
    anomaly, hypotheses, findings = plan_of(scenario_id)
    report = narrate(anomaly, hypotheses, findings)
    assert verify(report, findings).ok
    assert report.opinions == []
    assert report.proposed_action is None
    assert report.causes == hypotheses
    assert report.status == ("identified" if hypotheses else "inconclusive")


def test_identified_report_names_the_cause_and_the_registered_change(plan_of):
    anomaly, hypotheses, findings = plan_of("s01")
    text = render_report(narrate(anomaly, hypotheses, findings), findings)
    assert "Status: causa identificada" in text
    assert "platform = android, app_version = 3.2" in text
    assert "Mudança registrada: Release 3.2 no Android em 2026-03-26." in text
    assert "{" not in text and "}" not in text
    assert "Opinião" not in text


def test_two_causes_proved_together_get_one_joint_fact_and_a_labeled_change_each(plan_of):
    anomaly, hypotheses, findings = plan_of("s12")
    report = narrate(anomaly, hypotheses, findings)
    text = render_report(report, findings)
    [joint] = [f.text for f in report.facts if "em conjunto" in f.text]
    assert "product = phone e region = sul" in joint or "region = sul e product = phone" in joint
    assert sum("Causa" in f.text for f in report.facts) == 1  # o mesmo número não é repetido
    assert "Para product = phone, não há mudança registrada." in text
    assert "Para region = sul, mudança registrada: Mudança de infraestrutura" in text
    assert "sozinhas," in text and "Juntas," in text


def test_a_decoy_change_in_another_scope_is_not_reported_as_evidence(plan_of):
    anomaly, hypotheses, findings = plan_of("s03")
    text = render_report(narrate(anomaly, hypotheses, findings), findings)
    assert "Release 3.2 no Android" in text
    assert "Hotfix" not in text


def test_inconclusive_report_says_so_and_cites_no_slice(plan_of):
    anomaly, hypotheses, findings = plan_of("s19")
    assert hypotheses == []
    report = narrate(anomaly, hypotheses, findings)
    text = render_report(report, findings)
    assert "Status: inconclusivo" in text
    assert "a causa não foi identificada" in text
    assert "Causa confirmada" not in text
    assert report.causes == []


def test_narrate_needs_something_to_narrate(plan_of):
    anomaly, hypotheses, _ = plan_of("s01")
    with pytest.raises(ValueError, match="Finding"):
        narrate(anomaly, hypotheses, [])
