"""Texto final do relatório: o código preenche os placeholders e separa fato de opinião."""

from sentinel.agent.placeholders import fill
from sentinel.data.profile import SchemaProfile
from sentinel.schemas import Finding, Report

KIND_LABELS = {"cause": "causa provável", "recommendation": "recomendação", "caveat": "ressalva"}


def render_report(
    report: Report, findings: list[Finding], profile: SchemaProfile | None = None
) -> str:
    """Relatório em texto, com os números vindos dos `Finding`s."""
    status = "causa identificada" if report.status == "identified" else "inconclusivo"
    lines = [f"Status: {status}", "", "Fatos (verificados pelo código):"]
    lines += [f"- {fill(fact.text, findings, profile)}" for fact in report.facts]
    if report.opinions:
        lines += ["", "Opinião do agente (não verificada; a confiança é declarada pelo modelo):"]
        for opinion in report.opinions:
            label = f"{KIND_LABELS[opinion.kind]}, confiança {opinion.confidence:.0%}"
            lines.append(f"- [{label}] {fill(opinion.text, findings, profile)}")
    return "\n".join(lines)
