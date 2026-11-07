"""Policy de autonomia (seção 9.7 do CLAUDE.md): decide o que pode ser feito sem humano.

Função pura. Só vira `auto_execute` se todas as condições forem verdadeiras; qualquer dúvida cai
para `ask_human`. `block` = há uma ação, mas nada pode ser executado (kill switch, ação fora do
catálogo). Com `dry_run`, a ação automática é só registrada.
"""

from typing import Literal

from pydantic import BaseModel

from sentinel.config import Settings
from sentinel.investigation.actions_catalog import CATALOG
from sentinel.records import History
from sentinel.schemas import ActionProposal, Report

DecisionKind = Literal["report_only", "ask_human", "auto_execute", "block"]


class PolicyConfig(BaseModel):
    autonomy_level: int = 2
    kill_switch: bool = False
    dry_run: bool = True
    auto_min_ep: float = 0.8
    auto_min_precision: float = 0.9
    auto_min_history: int = 10
    auto_max_per_hour: int = 2

    @classmethod
    def from_settings(cls, settings: Settings, kill_switch: bool | None = None) -> "PolicyConfig":
        return cls(
            autonomy_level=settings.autonomy_level,
            kill_switch=settings.kill_switch if kill_switch is None else kill_switch,
            dry_run=settings.dry_run,
            auto_min_ep=settings.auto_min_ep,
            auto_min_precision=settings.auto_min_precision,
            auto_min_history=settings.auto_min_history,
            auto_max_per_hour=settings.auto_max_per_hour,
        )


class Decision(BaseModel):
    kind: DecisionKind
    reasons: list[str]  # por que a decisão não foi mais autônoma (ou por que foi)
    dry_run: bool = False


def _auto_blockers(
    report: Report,
    proposal: ActionProposal,
    history: History,
    config: PolicyConfig,
    verified: bool,
) -> list[str]:
    """Cada condição de L3 que falhou (lista vazia = pode executar sozinho)."""
    problems = []
    if report.status != "identified":
        problems.append("a causa não foi identificada")
    if not verified:
        problems.append("o verificador reprovou o rascunho do relatório")
    shares = [c.explained_share for c in report.causes]
    if not shares or min(shares) < config.auto_min_ep:
        problems.append(f"poder de explicação abaixo de {config.auto_min_ep:.0%}")
    spec = CATALOG[proposal.action_id]
    if not spec.reversible:
        problems.append("a ação não é reversível")
    if spec.risk != "low":
        problems.append(f"risco {spec.risk}")
    if history.total < config.auto_min_history:
        problems.append(
            f"histórico curto: {history.total} decisões (mínimo {config.auto_min_history})"
        )
    elif (history.precision or 0.0) < config.auto_min_precision:
        problems.append(f"precisão histórica abaixo de {config.auto_min_precision:.0%}")
    if history.auto_last_hour >= config.auto_max_per_hour:
        problems.append(f"limite de {config.auto_max_per_hour} ações automáticas por hora")
    return problems


def decide(
    report: Report,
    proposal: ActionProposal | None,
    history: History,
    config: PolicyConfig,
    *,
    verified: bool = True,
    duplicate_of: str | None = None,
) -> Decision:
    if duplicate_of is not None:
        return Decision(kind="report_only", reasons=[f"duplicado de {duplicate_of} (cooldown)"])
    if proposal is None or report.status != "identified":
        return Decision(kind="report_only", reasons=["nenhuma ação do catálogo se aplica"])
    spec = CATALOG.get(proposal.action_id)
    if spec is None or spec.reversible != proposal.reversible or spec.risk != proposal.risk:
        return Decision(kind="block", reasons=["ação fora do catálogo ou adulterada"])
    if config.kill_switch:
        return Decision(kind="block", reasons=["kill switch ligado: nenhuma ação é executada"])
    if config.autonomy_level <= 1:
        return Decision(kind="report_only", reasons=["autonomia L1: só relatório"])
    if config.autonomy_level == 2:
        return Decision(kind="ask_human", reasons=["autonomia L2: ação só com aprovação"])
    blockers = _auto_blockers(report, proposal, history, config, verified)
    if blockers:
        return Decision(kind="ask_human", reasons=blockers)
    return Decision(
        kind="auto_execute",
        reasons=["todas as condições de L3 atendidas"],
        dry_run=config.dry_run,
    )
