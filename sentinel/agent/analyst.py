"""Analista (seção 9.5 do CLAUDE.md): o LLM escreve fatos com placeholders e a opinião.

O LLM não decide status, causas nem evidências: isso vem do código. Depois do rascunho, o
verificador confere tudo; se reprovar (ou o modelo falhar), sai o relatório do narrador de template.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sentinel.agent.narrator import narrate
from sentinel.agent.placeholders import finding_ids_in, flatten
from sentinel.agent.verifier import Verification, verify
from sentinel.llm.base import JsonObject, LLMClient, LLMError
from sentinel.schemas import Anomaly, Finding, Hypothesis, Opinion, Report, Statement

MAX_FACTS = 6
MAX_OPINIONS = 3

SYSTEM = (
    "Você é o analista do DataSentinel, um agente que investiga quedas e picos em métricas de "
    "negócio. Escreva em português do Brasil, em frases curtas. "
    "Nunca escreva dígitos: para citar um valor, escreva {ID.CAMPO} com um id de fato e um campo "
    "listados abaixo. "
    "Em 'facts' diga só o que as causas confirmadas e os fatos mostram; não cite outra fatia. "
    "Em 'opinions' dê a sua leitura (causa provável, recomendação ou ressalva), deixando claro que "
    "é opinião, não fato. Cite apenas ids que existem."
)


class AnalystError(RuntimeError):
    """O modelo respondeu, mas não no formato pedido."""


class _Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[Statement] = Field(min_length=1, max_length=MAX_FACTS)
    opinions: list[Opinion] = Field(max_length=MAX_OPINIONS)


class ReportOutcome(BaseModel):
    report: Report
    source: Literal["llm", "template"]
    rejection: Verification | None = None  # por que o rascunho do LLM foi reprovado
    error: str | None = None  # por que o LLM não entregou um rascunho


def usable(findings: Sequence[Finding]) -> list[Finding]:
    return [f for f in findings if "error" not in f.result]


def response_schema(finding_ids: Sequence[str]) -> JsonObject:
    """JSON Schema da resposta; o `finding_id` só aceita ids que existem."""
    finding_id = {"type": "string", "enum": list(finding_ids)}
    return {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_FACTS,
                "items": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}, "finding_id": finding_id},
                    "required": ["text", "finding_id"],
                    "additionalProperties": False,
                },
            },
            "opinions": {
                "type": "array",
                "maxItems": MAX_OPINIONS,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["cause", "recommendation", "caveat"]},
                        "text": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "finding_ids": {"type": "array", "items": finding_id},
                    },
                    "required": ["kind", "text", "confidence", "finding_ids"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["facts", "opinions"],
        "additionalProperties": False,
    }


def _slice(dimensions: dict[str, str]) -> str:
    return ", ".join(f"{name}={value}" for name, value in dimensions.items())


def build_prompt(
    anomaly: Anomaly,
    hypotheses: Sequence[Hypothesis],
    findings: Sequence[Finding],
    unconfirmed: Sequence[dict[str, str]] = (),
) -> str:
    """Resumo compacto: o LLM vê os valores já formatados, não as tabelas."""
    direction = "queda" if anomaly.direction == "drop" else "pico"
    lines = [
        f"Anomalia: {direction} em {anomaly.metric}, de {anomaly.window_start} a "
        f"{anomaly.window_end} (início estimado {anomaly.onset}).",
    ]
    if hypotheses:
        lines.append("Causas confirmadas pelo código:")
        lines += [
            f"- {_slice(h.dimensions)} (explica {h.explained_share:.0%}; "
            f"provas {', '.join(h.finding_ids)})"
            for h in hypotheses
        ]
    else:
        lines.append("Nenhuma causa foi confirmada pelo código.")
    if unconfirmed:
        lines.append("Hipóteses NÃO confirmadas (só podem aparecer em 'opinions'):")
        lines += [f"- {_slice(d)}" for d in unconfirmed]
    lines.append("Fatos (id, ferramenta, campos citáveis):")
    for finding in usable(findings):
        fields = "; ".join(f"{path}={value}" for path, value in flatten(finding).items())
        lines.append(f"{finding.id} {finding.tool}: {fields}")
    return "\n".join(lines)


def analyze(
    llm: LLMClient,
    anomaly: Anomaly,
    hypotheses: Sequence[Hypothesis],
    findings: Sequence[Finding],
    unconfirmed: Sequence[dict[str, str]] = (),
) -> Report:
    """Rascunho do relatório. Status, causas e evidências vêm do código, não do modelo."""
    ids = [f.id for f in usable(findings)]
    if not ids:
        raise AnalystError("nenhum Finding utilizável")
    raw = llm.generate_json(
        SYSTEM, build_prompt(anomaly, hypotheses, findings, unconfirmed), response_schema(ids)
    )
    try:
        draft = _Draft.model_validate(raw)
    except ValidationError as error:
        raise AnalystError(f"resposta fora do schema: {error.error_count()} erro(s)") from error

    cited = [h_id for h in hypotheses for h_id in h.finding_ids]
    for fact in draft.facts:
        cited += [fact.finding_id, *finding_ids_in(fact.text)]
    for opinion in draft.opinions:
        cited += [*opinion.finding_ids, *finding_ids_in(opinion.text)]
    return Report(
        status="identified" if hypotheses else "inconclusive",
        facts=draft.facts,
        causes=list(hypotheses),
        opinions=draft.opinions,
        evidence=list(dict.fromkeys(cited)),
    )


def _template(
    anomaly: Anomaly,
    hypotheses: Sequence[Hypothesis],
    findings: Sequence[Finding],
    *,
    rejection: Verification | None = None,
    error: str | None = None,
) -> ReportOutcome:
    report = narrate(anomaly, list(hypotheses), list(findings))
    check = verify(report, list(findings))
    if not check.ok:  # o narrador é código nosso: reprovar aqui é bug, não dado ruim
        raise RuntimeError(f"relatório de template reprovado: {check.problems}")
    return ReportOutcome(report=report, source="template", rejection=rejection, error=error)


def write_report(
    llm: LLMClient | None,
    anomaly: Anomaly,
    hypotheses: Sequence[Hypothesis],
    findings: Sequence[Finding],
    unconfirmed: Sequence[dict[str, str]] = (),
) -> ReportOutcome:
    """Relatório verificado: do LLM se o verificador aprovar, senão do narrador de template."""
    if llm is None:
        return _template(anomaly, hypotheses, findings)
    try:
        draft = analyze(llm, anomaly, hypotheses, findings, unconfirmed)
    except (LLMError, AnalystError) as error:
        return _template(anomaly, hypotheses, findings, error=str(error))
    verification = verify(draft, list(findings))
    if verification.ok:
        return ReportOutcome(report=draft, source="llm")
    return _template(anomaly, hypotheses, findings, rejection=verification)
