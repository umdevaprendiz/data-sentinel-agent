"""Catálogo fechado de ações (seção 9.5 e 9.8 do CLAUDE.md).

A ação sugerida vem daqui, por tipo de causa, nunca do LLM (ele só justifica a escolha). O tipo
da causa sai do que o agente pode ver: a mudança registrada em `changes` para a fatia da causa
(um `Finding` de `check_recent_changes`). Sem registro, a ação é abrir um chamado.

Risco e reversibilidade são fixos por ação. A policy só deixa executar sozinho (L3) o que é
reversível e de baixo risco; o resto pede um humano.
"""

from dataclasses import dataclass
from typing import Literal

from sentinel.schemas import ActionProposal, Finding, Hypothesis

Risk = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    description: str
    reversible: bool
    risk: Risk
    change_type: str | None  # tipo de mudança registrada que essa ação desfaz


CATALOG: dict[str, ActionSpec] = {
    spec.action_id: spec
    for spec in (
        ActionSpec(
            "rollback_deploy",
            "Reverter o deploy registrado para a fatia (versão anterior em produção).",
            reversible=True,
            risk="low",
            change_type="deploy",
        ),
        ActionSpec(
            "restore_price",
            "Voltar o preço do produto ao valor anterior à mudança registrada.",
            reversible=True,
            risk="medium",
            change_type="price_change",
        ),
        ActionSpec(
            "resume_campaign",
            "Reativar a campanha encerrada no canal (volta a gastar verba).",
            reversible=True,
            risk="medium",
            change_type="campaign_end",
        ),
        ActionSpec(
            "restock_alert",
            "Avisar a equipe de estoque para repor o produto.",
            reversible=True,
            risk="low",
            change_type="stock_out",
        ),
        ActionSpec(
            "open_ticket",
            "Abrir um chamado para a equipe responsável investigar a fatia.",
            reversible=True,
            risk="low",
            change_type=None,
        ),
    )
}

# Pico causado por campanha nova é o efeito esperado: não há o que corrigir.
NO_ACTION_CHANGES = frozenset({"campaign_start"})

ACTION_LABELS = {
    "rollback_deploy": "reverter deploy",
    "restore_price": "restaurar preço",
    "resume_campaign": "reativar campanha",
    "restock_alert": "alerta de reposição",
    "open_ticket": "abrir chamado",
}


def _matching_change(cause: Hypothesis, findings: dict[str, Finding]) -> dict | None:
    """Mudança registrada que fala exatamente da fatia (ou de parte dela) da causa."""
    for finding_id in cause.finding_ids:
        finding = findings.get(finding_id)
        if finding is None or finding.tool != "check_recent_changes":
            continue
        if "error" in finding.result:
            continue
        candidates = [
            change
            for change in finding.result.get("changes", [])
            if change.get("scope")
            and all(cause.dimensions.get(k) == v for k, v in change["scope"].items())
        ]
        if candidates:
            # a mais específica e, entre iguais, a mais recente
            return max(candidates, key=lambda c: (len(c["scope"]), c["ts"]))
    return None


def proposal_for(
    causes: list[Hypothesis], findings: list[Finding], direction: str
) -> ActionProposal | None:
    """Ação do catálogo para as causas (None = nada a fazer).

    Uma causa com mudança registrada tem preferência: desfazer a mudança ataca a causa. Sem
    registro, a ação é abrir um chamado. Pico (alta) não pede correção.
    """
    if direction == "spike":
        return None
    by_id = {f.id: f for f in findings}
    for cause in causes:
        change = _matching_change(cause, by_id)
        if change is None or change["type"] in NO_ACTION_CHANGES:
            continue
        spec = next((s for s in CATALOG.values() if s.change_type == change["type"]), None)
        if spec is not None:
            params = {str(k): str(v) for k, v in change["scope"].items()}
            return _proposal(spec, params)
    if causes:
        params = {str(k): str(v) for k, v in causes[0].dimensions.items()}
        return _proposal(CATALOG["open_ticket"], params)
    return None


def _proposal(spec: ActionSpec, params: dict[str, str]) -> ActionProposal:
    return ActionProposal(
        action_id=spec.action_id, params=params, reversible=spec.reversible, risk=spec.risk
    )


def describe(proposal: ActionProposal) -> str:
    where = ", ".join(f"{k} = {v}" for k, v in proposal.params.items())
    label = ACTION_LABELS.get(proposal.action_id, proposal.action_id)
    return f"{label} ({where})" if where else label
