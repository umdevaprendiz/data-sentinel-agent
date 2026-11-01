import pytest

from sentinel.agent.verifier import verify
from sentinel.schemas import Finding, Hypothesis, Opinion, Report, Statement


def compare(id_: str = "f001", direction: str = "drop") -> Finding:
    return Finding(
        id=id_,
        tool="compare_periods",
        params={"metric": "conversion_rate"},
        result={
            "metric": "conversion_rate",
            "observed": 0.0301,
            "expected": 0.0352,
            "change": -0.0051,
            "change_pct": -0.145,
            "z": -8.2,
            "direction": direction,
        },
    )


def hypothesis_finding(id_: str = "f002", dimensions: dict | None = None, accepted: bool = True):
    return Finding(
        id=id_,
        tool="test_hypothesis",
        params={},
        result={
            "dimensions": dimensions or {"platform": "android", "app_version": "3.2"},
            "explained_share": 0.97,
            "accepted": accepted,
            "inside": {"change_pct": -0.41},
        },
    )


FINDINGS = [compare(), hypothesis_finding()]
CAUSE = Hypothesis(
    dimensions={"platform": "android", "app_version": "3.2"},
    explained_share=0.97,
    finding_ids=["f002"],
)


def report(**changes) -> Report:
    base = {
        "status": "identified",
        "facts": [
            Statement(text="A taxa variou {f001.change_pct}.", finding_id="f001"),
            Statement(
                text="A causa é platform = android, app_version = 3.2 ({f002.explained_share}).",
                finding_id="f002",
            ),
        ],
        "causes": [CAUSE],
        "opinions": [],
        "evidence": ["f001", "f002"],
    }
    return Report(**{**base, **changes})


def codes(result) -> list[str]:
    return [p.code for p in result.problems]


def test_a_clean_report_passes():
    result = verify(report(), FINDINGS)
    assert result.ok
    assert result.problems == []


def test_inconclusive_report_passes():
    inconclusive = report(
        status="inconclusive",
        causes=[],
        facts=[Statement(text="Nenhuma fatia explica a variação.", finding_id="f001")],
        evidence=["f001"],
    )
    assert verify(inconclusive, FINDINGS).ok


# --- números soltos ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["A taxa caiu 14%.", "Foram 3 causas.", "Caiu para 0,03."])
def test_loose_number_in_a_fact_is_refused(text):
    bad = report(facts=[Statement(text=text, finding_id="f001")])
    assert codes(verify(bad, FINDINGS)) == ["loose_number"]


def test_loose_number_in_an_opinion_is_refused():
    opinion = Opinion(
        kind="cause", text="Talvez 30% seja bug.", confidence=0.4, finding_ids=["f001"]
    )
    assert codes(verify(report(opinions=[opinion]), FINDINGS)) == ["loose_number"]


def test_a_number_inside_a_placeholder_is_fine_and_known_versions_are_ignored():
    opinion = Opinion(
        kind="recommendation",
        text="Reverter a versão 3.2 no Android reduz a perda de {f001.change_pct}.",
        confidence=0.6,
        finding_ids=["f001", "f002"],
    )
    assert verify(report(opinions=[opinion]), FINDINGS).ok


# --- placeholders -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("Taxa {f009.change_pct}.", "f009"),
        ("Taxa {f001.profit}.", "profit"),
        ("Taxa {f002.inside}.", "valor simples"),
        ("Taxa {ID.CAMPO}.", "ID.CAMPO"),
        ("Taxa {}.", "{}"),
        ("Taxa {f001}.", "f001"),
    ],
)
def test_bad_placeholders_are_refused(text, fragment):
    bad = report(facts=[Statement(text=text, finding_id="f001")])
    result = verify(bad, FINDINGS)
    assert codes(result) == ["bad_placeholder"]
    assert fragment in result.problems[0].message


def test_placeholder_to_a_finding_that_failed_is_refused():
    failed = Finding(id="f003", tool="split_by", params={}, result={"error": "dimension inválida"})
    bad = report(facts=[Statement(text="Taxa {f003.total_change}.", finding_id="f001")])
    assert codes(verify(bad, [*FINDINGS, failed])) == ["bad_placeholder"]


# --- ids inventados ---------------------------------------------------------------------------


def test_invented_ids_are_refused_everywhere():
    bad = report(
        facts=[Statement(text="Algo.", finding_id="f777")],
        opinions=[Opinion(kind="caveat", text="Ver.", confidence=0.5, finding_ids=["f888"])],
        evidence=["f001", "f999"],
        causes=[CAUSE.model_copy(update={"finding_ids": ["f002", "f555"]})],
    )
    result = verify(bad, FINDINGS)
    assert codes(result).count("unknown_finding") == 4
    messages = " ".join(p.message for p in result.problems)
    assert all(i in messages for i in ("f777", "f888", "f999", "f555"))


# --- dimensões --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["A queda é no ios.", "A versão 3.1 também caiu.", "Vem do canal social."]
)
def test_a_fact_cannot_cite_a_slice_that_is_not_a_confirmed_cause(text):
    bad = report(facts=[Statement(text=text, finding_id="f001")])
    assert codes(verify(bad, FINDINGS)) == ["foreign_dimension"]


def test_a_fact_cannot_cite_a_dimension_name_outside_the_causes():
    bad = report(facts=[Statement(text="Veja o region.", finding_id="f001")])
    assert codes(verify(bad, FINDINGS)) == ["foreign_dimension"]


def test_an_inconclusive_fact_cannot_cite_any_slice():
    bad = report(
        status="inconclusive",
        causes=[],
        facts=[Statement(text="Parece vir do android.", finding_id="f001")],
        evidence=["f001"],
    )
    assert codes(verify(bad, FINDINGS)) == ["foreign_dimension"]


def test_an_opinion_may_name_an_unconfirmed_hypothesis():
    opinion = Opinion(
        kind="cause",
        text="Pode ser um problema no ios, mas o código não confirmou.",
        confidence=0.2,
        finding_ids=["f001"],
    )
    assert verify(report(opinions=[opinion]), FINDINGS).ok


# --- status e causas --------------------------------------------------------------------------


def test_identified_without_causes_is_incoherent():
    bare = report(causes=[], facts=[Statement(text="Algo variou.", finding_id="f001")])
    assert codes(verify(bare, FINDINGS)) == ["incoherent_status"]


def test_inconclusive_with_causes_is_incoherent():
    assert codes(verify(report(status="inconclusive"), FINDINGS)) == ["incoherent_status"]


def test_more_than_two_causes_is_incoherent():
    three = [CAUSE, CAUSE, CAUSE]
    assert "incoherent_status" in codes(verify(report(causes=three), FINDINGS))


def test_a_report_without_facts_is_incoherent():
    assert codes(verify(report(facts=[]), FINDINGS)) == ["incoherent_status"]


def test_a_cause_needs_an_accepted_test_hypothesis_behind_it():
    only_compare = CAUSE.model_copy(update={"finding_ids": ["f001"]})
    assert codes(verify(report(causes=[only_compare]), FINDINGS)) == ["unconfirmed_cause"]

    rejected = [compare(), hypothesis_finding(dimensions={"platform": "ios"}, accepted=False)]
    ios = Hypothesis(dimensions={"platform": "ios"}, explained_share=0.9, finding_ids=["f002"])
    bad = report(
        causes=[ios],
        facts=[Statement(text="A causa é platform = ios.", finding_id="f002")],
    )
    assert codes(verify(bad, rejected)) == ["unconfirmed_cause"]


def test_the_accepted_test_must_be_about_the_same_slice():
    other = [compare(), hypothesis_finding(dimensions={"platform": "android"})]
    assert codes(verify(report(), other)) == ["unconfirmed_cause"]


def test_two_causes_can_share_one_accepted_joint_test():
    sul, phone = {"region": "sul"}, {"product": "phone"}
    joint = Finding(
        id="f002",
        tool="test_hypothesis",
        params={},
        result={"dimensions": sul, "together_with": phone, "accepted": True},
    )
    causes = [
        Hypothesis(dimensions=sul, explained_share=0.5, finding_ids=["f002"]),
        Hypothesis(dimensions=phone, explained_share=0.5, finding_ids=["f002"]),
    ]
    facts = [Statement(text="Causas: region = sul e product = phone.", finding_id="f002")]
    both = report(causes=causes, facts=facts)
    assert verify(both, [compare(), joint]).ok
    third = Hypothesis(dimensions={"channel": "email"}, explained_share=0.5, finding_ids=["f002"])
    assert "unconfirmed_cause" in codes(
        verify(report(causes=[third], facts=facts), [compare(), joint])
    )


# --- direção ----------------------------------------------------------------------------------


def test_a_fact_that_contradicts_the_direction_of_its_finding_is_refused():
    bad = report(facts=[Statement(text="A taxa subiu {f001.change_pct}.", finding_id="f001")])
    assert codes(verify(bad, FINDINGS)) == ["contradiction"]


def test_an_opinion_cannot_contradict_the_cited_finding():
    opinion = Opinion(
        kind="caveat", text="A conversão aumentou no período.", confidence=0.5, finding_ids=["f001"]
    )
    assert codes(verify(report(opinions=[opinion]), FINDINGS)) == ["contradiction"]


def test_an_opinion_without_citations_is_checked_against_all_the_findings():
    opinion = Opinion(kind="caveat", text="A taxa subiu.", confidence=0.5, finding_ids=[])
    assert codes(verify(report(opinions=[opinion]), FINDINGS)) == ["contradiction"]


def test_mixed_directions_do_not_trigger_a_contradiction_without_citation():
    ticket = Finding(
        id="f003",
        tool="compare_periods",
        params={},
        result={"metric": "avg_ticket", "direction": "spike", "change_pct": 0.1},
    )
    opinion = Opinion(kind="caveat", text="O ticket subiu.", confidence=0.5, finding_ids=[])
    assert verify(report(opinions=[opinion]), [*FINDINGS, ticket]).ok
    cited = opinion.model_copy(update={"finding_ids": ["f001"]})
    assert codes(verify(report(opinions=[cited]), [*FINDINGS, ticket])) == ["contradiction"]


def test_the_right_direction_passes():
    ok = report(facts=[Statement(text="A taxa caiu {f001.change_pct}.", finding_id="f001")])
    assert verify(ok, FINDINGS).ok


def test_every_problem_is_reported():
    bad = report(
        facts=[Statement(text="Caiu 14% no ios.", finding_id="f001")],
        evidence=["f404"],
    )
    assert sorted(codes(verify(bad, FINDINGS))) == [
        "foreign_dimension",
        "loose_number",
        "unknown_finding",
    ]
