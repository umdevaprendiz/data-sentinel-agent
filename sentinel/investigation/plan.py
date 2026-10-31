"""Plano determinístico de investigação (`PLAN_MODE=deterministic`).

Roda as ferramentas do menu em ordem fixa e registra cada passo como `Finding`. É a linha de
base da ablação: o modo `llm` escolhe as mesmas ferramentas, mas por conta própria.
"""

from sentinel.investigation.drilldown import Window, find_causes
from sentinel.investigation.tools import ToolContext
from sentinel.schemas import Anomaly, Hypothesis


def investigate(ctx: ToolContext, anomaly: Anomaly) -> list[Hypothesis]:
    """Causas confirmadas pelo código (vazio = inconclusivo). Os fatos ficam em `ctx.findings`."""
    window = {
        "window_start": anomaly.window_start.isoformat(),
        "window_end": anomaly.window_end.isoformat(),
    }
    ctx.run("compare_periods", {**window, "metric": anomaly.metric})
    ctx.run("decompose_metric", window)

    causes = find_causes(
        ctx.cube,
        anomaly.metric,
        Window(anomaly.window_start, anomaly.window_end),
        ctx.drill_params(),
    )
    if not causes:
        ctx.run("check_recent_changes", window)  # contexto para o relatório inconclusivo
        return []

    hypotheses = []
    for cause in causes:
        proof = ctx.run(
            "test_hypothesis", {**window, "metric": anomaly.metric, "dimensions": cause.dimensions}
        )
        changes = ctx.run("check_recent_changes", {**window, "scope": cause.dimensions})
        hypotheses.append(
            Hypothesis(
                dimensions=cause.dimensions,
                explained_share=cause.explained_share,
                finding_ids=[proof.id, changes.id],
            )
        )
    return hypotheses
