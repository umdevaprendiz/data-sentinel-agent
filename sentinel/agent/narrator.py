"""Narrador de template: relatório sem LLM (`PLAN_MODE=deterministic` e fallback do analista).

Só escreve fatos, com placeholders; não tem opinião nem ação proposta.
"""

from sentinel.schemas import Anomaly, Finding, Hypothesis, Report, Statement

MAX_CHANGES = 2


def _ok(findings: list[Finding], tool: str) -> list[Finding]:
    return [f for f in findings if f.tool == tool and "error" not in f.result]


def _where(dimensions: dict[str, str]) -> str:
    return ", ".join(f"{name} = {value}" for name, value in dimensions.items())


def _changes_fact(finding: Finding, subject: str | None, labeled: bool) -> Statement:
    """Mudanças registradas. `subject` = fatia da causa (None = período, sem causa)."""
    i = finding.id
    count = finding.result["count"]
    lead = f"Para {subject}, " if subject and labeled else ""
    if count == 0:
        if subject is None:
            return Statement(text="Não há mudança registrada no período.", finding_id=i)
        if labeled:
            return Statement(text=f"{lead}não há mudança registrada.", finding_id=i)
        return Statement(text="Não há mudança registrada relacionada a essa fatia.", finding_id=i)
    items = "; ".join(
        f"{{{i}.changes.{n}.description}} em {{{i}.changes.{n}.ts}}"
        for n in range(min(count, MAX_CHANGES))
    )
    if subject is None:
        return Statement(text=f"Mudanças registradas, nenhuma confirmada: {items}.", finding_id=i)
    kind = "mudança registrada" if labeled else "Mudança registrada"
    return Statement(text=f"{lead}{kind}: {items}.", finding_id=i)


def _cause_facts(proof: Finding, causes: list[Hypothesis]) -> Statement:
    i = proof.id
    if "together_with" in proof.result:  # duas causas independentes, provadas juntas
        first, second = _where(proof.result["dimensions"]), _where(proof.result["together_with"])
        return Statement(
            text=(
                f"Causas confirmadas em conjunto: {first} e {second}. Juntas, explicam "
                f"{{{i}.explained_share}} da variação; sozinhas, {{{i}.members.0.explained_share}} "
                f"e {{{i}.members.1.explained_share}}."
            ),
            finding_id=i,
        )
    return Statement(
        text=(
            f"Causa confirmada: {_where(causes[0].dimensions)}. Dentro da fatia a métrica "
            f"variou {{{i}.inside.change_pct}}, fora dela {{{i}.outside.change_pct}}; "
            f"a fatia explica {{{i}.explained_share}} da variação."
        ),
        finding_id=i,
    )


def narrate(anomaly: Anomaly, hypotheses: list[Hypothesis], findings: list[Finding]) -> Report:
    """Relatório de template. `hypotheses` são as causas já confirmadas pelo código."""
    if not findings:
        raise ValueError("sem Finding não há o que narrar")
    facts: list[Statement] = []

    compare = [f for f in _ok(findings, "compare_periods") if f.result["metric"] == anomaly.metric]
    if compare:
        i = compare[0].id
        facts.append(
            Statement(
                text=(
                    f"A métrica {{{i}.metric}} ficou em {{{i}.observed}} contra {{{i}.expected}} "
                    f"esperado ({{{i}.change_pct}}), com z de {{{i}.z}}."
                ),
                finding_id=i,
            )
        )

    by_id = {f.id: f for f in findings}
    proofs = {}  # a prova de cada causa; duas causas provadas juntas dividem a mesma
    for cause in hypotheses:
        proof = next(by_id[n] for n in cause.finding_ids if by_id[n].tool == "test_hypothesis")
        proofs.setdefault(proof.id, (proof, []))[1].append(cause)
    for proof, causes in proofs.values():
        facts.append(_cause_facts(proof, causes))
    for cause in hypotheses:
        changes = [by_id[n] for n in cause.finding_ids if by_id[n].tool == "check_recent_changes"]
        if changes:
            facts.append(
                _changes_fact(changes[0], _where(cause.dimensions), labeled=len(hypotheses) > 1)
            )

    if not hypotheses:
        anchor = compare[0] if compare else findings[0]
        facts.append(
            Statement(
                text="Nenhuma fatia explica a variação de forma consistente; a causa não foi "
                "identificada.",
                finding_id=anchor.id,
            )
        )
        changes = _ok(findings, "check_recent_changes")
        if changes:
            facts.append(_changes_fact(changes[-1], None, labeled=False))

    cited = [f.finding_id for f in facts] + [n for h in hypotheses for n in h.finding_ids]
    evidence = list(dict.fromkeys(cited))
    return Report(
        status="identified" if hypotheses else "inconclusive",
        facts=facts,
        causes=hypotheses,
        opinions=[],
        evidence=evidence,
    )
