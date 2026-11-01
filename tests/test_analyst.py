import pytest

from sentinel.agent.analyst import (
    AnalystError,
    analyze,
    build_prompt,
    response_schema,
    write_report,
)
from sentinel.agent.render import render_report
from sentinel.llm.base import FakeLLM, LLMError
from sentinel.schemas import Finding

GOOD = {
    "facts": [
        {"text": "A taxa de conversão variou {f001.change_pct}.", "finding_id": "f001"},
        {
            "text": "A causa é platform = android, app_version = 3.2, que explica "
            "{f003.explained_share} da variação.",
            "finding_id": "f003",
        },
    ],
    "opinions": [
        {
            "kind": "cause",
            "text": "Parece um defeito introduzido pela release; é opinião, não medição.",
            "confidence": 0.7,
            "finding_ids": ["f003", "f004"],
        },
        {
            "kind": "recommendation",
            "text": "Vale conferir o funil de pagamento da versão antes de decidir um rollback.",
            "confidence": 0.5,
            "finding_ids": ["f004"],
        },
    ],
}


def edited(**changes):
    return {**GOOD, **changes}


def opinion(text="Talvez seja um bug.", **fields):
    base = {"kind": "caveat", "text": text, "confidence": 0.4, "finding_ids": ["f001"]}
    return {**base, **fields}


@pytest.fixture
def s01(plan_of):
    anomaly, hypotheses, findings = plan_of("s01")
    assert [f.tool for f in findings] == [
        "compare_periods",
        "decompose_metric",
        "test_hypothesis",
        "check_recent_changes",
    ]
    return anomaly, hypotheses, findings


def run(llm, s01, **kwargs):
    anomaly, hypotheses, findings = s01
    return write_report(llm, anomaly, hypotheses, findings, **kwargs)


# --- rascunho ---------------------------------------------------------------------------------


def test_a_valid_answer_becomes_a_verified_report(s01):
    llm = FakeLLM([GOOD])
    outcome = run(llm, s01)
    assert outcome.source == "llm"
    assert outcome.rejection is None and outcome.error is None
    report = outcome.report
    assert report.status == "identified"
    assert report.causes == s01[1]  # as causas vêm do código, não do modelo
    assert [f.finding_id for f in report.facts] == ["f001", "f003"]
    assert [o.kind for o in report.opinions] == ["cause", "recommendation"]
    assert report.evidence == ["f003", "f004", "f001"]  # causas primeiro, depois o que o texto cita
    assert len(llm.calls) == 1


def test_the_rendered_report_separates_fact_from_opinion_and_fills_the_numbers(s01):
    outcome = run(FakeLLM([GOOD]), s01)
    text = render_report(outcome.report, s01[2])
    facts, opinions = text.split("Opinião do agente")
    assert "Fatos (verificados pelo código)" in facts
    assert "{" not in text
    assert "A taxa de conversão variou -" in facts and "%." in facts
    assert "[causa provável, confiança 70%]" in opinions
    assert "[recomendação, confiança 50%]" in opinions


def test_the_status_is_decided_by_the_code_even_if_the_model_talks_about_no_cause(plan_of):
    anomaly, hypotheses, findings = plan_of("s19")
    answer = {
        "facts": [{"text": "A taxa variou {f001.change_pct}.", "finding_id": "f001"}],
        "opinions": [opinion("Pode ser uma queda difusa de demanda.")],
    }
    outcome = write_report(FakeLLM([answer]), anomaly, hypotheses, findings)
    assert outcome.source == "llm"
    assert outcome.report.status == "inconclusive"
    assert outcome.report.causes == []


def test_the_schema_only_accepts_existing_ids_and_closes_the_object(s01):
    llm = FakeLLM([GOOD])
    run(llm, s01)
    schema = llm.calls[0].schema
    ids = [f.id for f in s01[2]]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["facts", "opinions"]
    fact = schema["properties"]["facts"]["items"]["properties"]
    assert fact["finding_id"]["enum"] == ids
    opinion_props = schema["properties"]["opinions"]["items"]["properties"]
    assert opinion_props["finding_ids"]["items"]["enum"] == ids
    assert opinion_props["kind"]["enum"] == ["cause", "recommendation", "caveat"]
    assert response_schema(["f001"]) == response_schema(["f001"])


def test_findings_with_errors_are_hidden_from_the_model(s01):
    anomaly, hypotheses, findings = s01
    bad = Finding(
        id="f099", tool="compare_periods", params={}, result={"error": "métrica inválida"}
    )
    llm = FakeLLM([GOOD])
    write_report(llm, anomaly, hypotheses, [*findings, bad])
    fact_ids = llm.calls[0].schema["properties"]["facts"]["items"]["properties"]["finding_id"]
    assert "f099" not in fact_ids["enum"]
    assert "f099" not in llm.calls[0].prompt


# --- prompt -----------------------------------------------------------------------------------


def test_the_prompt_is_short_and_has_no_ready_made_answer(s01):
    anomaly, hypotheses, findings = s01
    llm = FakeLLM([GOOD])
    run(llm, s01)
    call = llm.calls[0]
    assert "platform=android, app_version=3.2" in call.prompt
    assert "Causas confirmadas pelo código" in call.prompt
    assert "f001 compare_periods:" in call.prompt
    assert "f003 test_hypothesis:" in call.prompt
    assert len(call.prompt) < 3500  # resumos, nunca tabelas inteiras
    assert "Nunca escreva dígitos" in call.system
    assert "kind" not in call.system and "confidence" not in call.system  # o schema já diz
    assert "A causa é" not in call.system + call.prompt  # nada para o modelo copiar


def test_the_prompt_lists_unconfirmed_hypotheses_apart(s01):
    anomaly, hypotheses, findings = s01
    prompt = build_prompt(anomaly, hypotheses, findings, unconfirmed=[{"platform": "ios"}])
    assert "NÃO confirmadas" in prompt
    assert "- platform=ios" in prompt
    assert "NÃO confirmadas" not in build_prompt(anomaly, hypotheses, findings)


def test_the_prompt_without_causes_says_so(plan_of):
    anomaly, hypotheses, findings = plan_of("s19")
    assert "Nenhuma causa foi confirmada pelo código." in build_prompt(
        anomaly, hypotheses, findings
    )


# --- o verificador reprova -> narrador de template --------------------------------------------


@pytest.mark.parametrize(
    ("answer", "code"),
    [
        (
            edited(facts=[{"text": "A taxa caiu 14%.", "finding_id": "f001"}]),
            "loose_number",
        ),
        (
            edited(facts=[{"text": "A taxa variou.", "finding_id": "f777"}]),
            "unknown_finding",
        ),
        (
            edited(facts=[{"text": "A taxa variou {f001.nada}.", "finding_id": "f001"}]),
            "bad_placeholder",
        ),
        (
            edited(facts=[{"text": "A queda é no ios.", "finding_id": "f001"}]),
            "foreign_dimension",
        ),
        (
            edited(facts=[{"text": "A taxa subiu {f001.change_pct}.", "finding_id": "f001"}]),
            "contradiction",
        ),
        (edited(opinions=[opinion("Foram 30% das sessões.")]), "loose_number"),
        (edited(opinions=[opinion("Talvez.", finding_ids=["f999"])]), "unknown_finding"),
        (edited(opinions=[opinion("A conversão aumentou.")]), "contradiction"),
    ],
)
def test_a_rejected_draft_falls_back_to_the_template_and_drops_the_opinion(answer, code, s01):
    outcome = run(FakeLLM([answer]), s01)
    assert outcome.source == "template"
    assert code in [p.code for p in outcome.rejection.problems]
    assert outcome.error is None
    assert outcome.report.opinions == []
    assert outcome.report.causes == s01[1]
    assert "Causa confirmada" in render_report(outcome.report, s01[2])


def test_an_unconfirmed_hypothesis_may_live_in_the_opinion_but_never_in_a_fact(s01):
    ok = edited(opinions=[opinion("Talvez o problema esteja no ios.", kind="cause")])
    assert run(FakeLLM([ok]), s01, unconfirmed=[{"platform": "ios"}]).source == "llm"
    bad = edited(facts=[{"text": "A causa está no ios.", "finding_id": "f001"}])
    assert run(FakeLLM([bad]), s01, unconfirmed=[{"platform": "ios"}]).source == "template"


# --- o LLM falha ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        {},
        {"facts": []},
        {"facts": "texto", "opinions": []},
        edited(extra="não pedi"),
        edited(opinions=[opinion(confidence=1.5)]),
        edited(opinions=[opinion(kind="palpite")]),
        edited(opinions=[opinion() for _ in range(4)]),
        edited(facts=[GOOD["facts"][0]] * 7),
        edited(facts=[{"text": "sem id"}]),
    ],
)
def test_an_answer_outside_the_schema_falls_back_to_the_template(reply, s01):
    outcome = run(FakeLLM([reply]), s01)
    assert outcome.source == "template"
    assert outcome.rejection is None
    assert "fora do schema" in outcome.error


def test_an_llm_that_raises_falls_back_to_the_template(s01):
    outcome = run(FakeLLM([LLMError("tempo esgotado")]), s01)
    assert outcome.source == "template"
    assert outcome.error == "tempo esgotado"
    assert "Causa confirmada" in render_report(outcome.report, s01[2])


def test_an_exhausted_script_is_an_llm_error(s01):
    assert run(FakeLLM([]), s01).error == "FakeLLM sem resposta no roteiro"


def test_without_an_llm_the_template_is_used_and_no_call_is_made(s01):
    outcome = run(None, s01)
    assert outcome.source == "template"
    assert outcome.error is None and outcome.rejection is None


def test_analyze_needs_usable_findings(plan_of):
    anomaly, hypotheses, _ = plan_of("s01")
    with pytest.raises(AnalystError, match="utilizável"):
        analyze(FakeLLM([GOOD]), anomaly, hypotheses, [])
