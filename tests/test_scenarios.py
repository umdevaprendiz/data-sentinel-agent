from datetime import date

import pytest

from sentinel.data.generator import N_DAYS, START_DATE, load_scenarios
from sentinel.schemas import Cause, Scenario

from .conftest import SCENARIOS_FILE

ALL = load_scenarios(SCENARIOS_FILE)
BY_ID = {s.id: s for s in ALL}
LAST_DAY = date.fromordinal(START_DATE.toordinal() + N_DAYS - 1)


def test_there_are_twenty_unique_scenarios():
    assert [s.id for s in ALL] == [f"s{i:02d}" for i in range(1, 21)]


@pytest.mark.parametrize("scenario", ALL, ids=lambda s: s.id)
def test_dates_fall_inside_the_data(scenario):
    start, end = scenario.window
    assert START_DATE <= start <= end <= LAST_DAY
    for cause in scenario.causes:
        assert START_DATE <= cause.start <= LAST_DAY
    for change in scenario.changes:
        assert START_DATE <= change.ts <= LAST_DAY


def test_normal_scenarios_have_no_cause_and_no_anomaly():
    for sid in ("s15", "s16", "s17", "s18"):
        scenario = BY_ID[sid]
        assert scenario.causes == []
        assert scenario.anomaly_expected is False
        assert scenario.expected_status == "none"


def test_decoy_change_in_a_normal_scenario():
    assert BY_ID["s17"].causes == []
    assert len(BY_ID["s17"].changes) == 1


def test_diffuse_drop_expects_inconclusive():
    scenario = BY_ID["s19"]
    assert scenario.anomaly_expected is True
    assert scenario.expected_status == "inconclusive"
    assert scenario.causes[0].dimensions == {}


def test_two_causes_scenario():
    assert len(BY_ID["s12"].causes) == 2


def test_unlogged_scenarios_have_no_changes():
    assert BY_ID["s13"].changes == []
    assert BY_ID["s20"].changes == []


def test_the_rest_expect_an_identified_cause():
    identified = [s.id for s in ALL if s.expected_status == "identified"]
    assert identified == [f"s{i:02d}" for i in [*range(1, 15), 20]]
    assert all(BY_ID[sid].anomaly_expected for sid in identified)


def test_drift_has_a_ramp():
    assert BY_ID["s20"].causes[0].ramp_days == 28


def test_schema_rejects_unknown_dimension_values():
    with pytest.raises(ValueError, match="app_version"):
        Cause(dimensions={"app_version": "9.9"}, start=date(2026, 3, 26), conversion_factor=0.5)
    with pytest.raises(ValueError, match="dimens"):
        Cause(dimensions={"color": "red"}, start=date(2026, 3, 26), conversion_factor=0.5)


def test_schema_rejects_inconsistent_expectations():
    base = BY_ID["s15"].model_dump()
    with pytest.raises(ValueError, match="expected_status"):
        Scenario.model_validate({**base, "anomaly_expected": True})
