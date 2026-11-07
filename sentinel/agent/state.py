"""Estado do grafo (seção 5 do CLAUDE.md).

Só dados JSON simples: o estado vai para o checkpoint (SQLite) quando o grafo pausa esperando
aprovação e precisa ser retomado em outro processo. Conexões e caches ficam no `Runtime`.
"""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    incident_id: str
    as_of: str | None
    started_at: float
    profile: dict[str, Any]
    anomaly: dict[str, Any] | None
    anomalies: list[dict[str, Any]]
    # investigação
    rounds: int
    explore_done: bool
    candidates: list[dict[str, Any]]  # hipóteses nomeadas pelo LLM (ainda não confirmadas)
    hypotheses: list[dict[str, Any]]  # confirmadas pelo código
    unconfirmed: list[dict[str, str]]
    refutation: dict[str, Any] | None
    investigation: dict[str, Any]
    findings: list[dict[str, Any]]
    # conclusão e relatório
    duplicate_of: str | None
    proposal: dict[str, Any] | None
    report: dict[str, Any] | None
    report_source: str
    report_problems: list[dict[str, Any]]
    report_error: str | None
    rendered: str
    verified: bool
    # ação
    decision: dict[str, Any]
    world: str | None
    approval: dict[str, Any] | None
    execution: dict[str, Any] | None
    final_state: str
