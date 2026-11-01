import pytest

from sentinel.agent.placeholders import (
    PlaceholderError,
    fill,
    finding_ids_in,
    flatten,
    format_value,
    resolve,
    strip_placeholders,
    tokens,
)
from sentinel.schemas import Finding

COMPARE = Finding(
    id="f001",
    tool="compare_periods",
    params={"metric": "conversion_rate", "window_start": "2026-03-26"},
    result={
        "metric": "conversion_rate",
        "observed": 0.03012,
        "expected": 0.0352,
        "change": -0.00508,
        "change_pct": -0.1443,
        "z": -8.24,
        "direction": "drop",
    },
)
SESSIONS = Finding(
    id="f002",
    tool="compare_periods",
    params={"metric": "sessions"},
    result={
        "metric": "sessions",
        "observed": 352_410.4,
        "expected": 400_000.0,
        "change_pct": -0.119,
    },
)
CHANGES = Finding(
    id="f003",
    tool="check_recent_changes",
    params={},
    result={
        "count": 2,
        "changes": [
            {"ts": "2026-03-26", "description": "Release 3.2 no Android"},
            {"ts": "2026-03-25", "description": "Hotfix"},
        ],
    },
)
BY_ID = {f.id: f for f in (COMPARE, SESSIONS, CHANGES)}


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("f001.change_pct", "-14,4%"),
        ("f001.z", "-8,2"),
        ("f001.observed", "3,01%"),  # taxa: vira porcentagem
        ("f001.change", "-0,51 p.p."),
        ("f001.metric", "conversion_rate"),
        ("f002.observed", "352.410"),
        ("f002.change_pct", "-11,9%"),
        ("f003.count", "2"),
        ("f003.changes.0.description", "Release 3.2 no Android"),
        ("f003.changes.1.ts", "2026-03-25"),
        ("f001.params.window_start", "2026-03-26"),
    ],
)
def test_resolve_formats_in_portuguese(token, expected):
    assert resolve(token, BY_ID) == expected


def test_fill_replaces_every_placeholder():
    text = "Caiu {f001.change_pct} ({f001.observed} contra {f001.expected})."
    assert fill(text, BY_ID.values()) == "Caiu -14,4% (3,01% contra 3,52%)."


@pytest.mark.parametrize(
    ("token", "fragment"),
    [
        ("f009.z", "f009"),
        ("f001.nada", "nada"),
        ("f003.changes.5.ts", "5"),
        ("f003.changes", "valor simples"),
        ("f003.changes.0", "valor simples"),
        ("f001", "malformado"),
        ("ID.CAMPO", "malformado"),
        ("", "malformado"),
        ("f001.z extra", "malformado"),
    ],
)
def test_resolve_refuses_what_does_not_exist(token, fragment):
    with pytest.raises(PlaceholderError, match=fragment):
        resolve(token, BY_ID)


def test_text_helpers():
    text = "A {f001.z} e {f003.count} e {ID.CAMPO}."
    assert tokens(text) == ["f001.z", "f003.count", "ID.CAMPO"]
    assert finding_ids_in(text) == ["f001", "f003"]
    assert strip_placeholders(text).count("{") == 0


def test_format_value_rules():
    assert format_value("flag", True) == "sim"
    assert format_value("n", 1234567) == "1.234.567"
    assert format_value("explained_share", 0.974) == "97%"
    assert format_value("threshold", 0.6) == "60%"
    assert format_value("avg_ticket_value", 12.3456) == "12,35"
    assert format_value("tiny", 0.01234) == "0,0123"
    assert format_value("big", 1234.56) == "1.235"


def test_flatten_lists_every_citable_path_and_each_one_resolves():
    paths = flatten(CHANGES)
    assert paths["count"] == "2"
    assert paths["changes.0.description"] == "Release 3.2 no Android"
    for finding in BY_ID.values():
        for path, value in flatten(finding).items():
            assert resolve(f"{finding.id}.{path}", BY_ID) == value


def test_flatten_keeps_only_the_first_items_of_long_lists():
    many = Finding(id="f004", tool="split_by", params={}, result={"slices": list(range(10))})
    assert sorted(flatten(many)) == ["slices.0", "slices.1", "slices.2"]


def test_flatten_uses_the_metric_of_nested_factors():
    decomposed = Finding(
        id="f005",
        tool="decompose_metric",
        params={},
        result={
            "factors": {"conversion_rate": {"observed": 0.03}, "sessions": {"observed": 5000.0}}
        },
    )
    assert flatten(decomposed) == {
        "factors.conversion_rate.observed": "3,00%",
        "factors.sessions.observed": "5.000",
    }
