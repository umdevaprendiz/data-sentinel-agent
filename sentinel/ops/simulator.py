"""Simulador de operações (seção 9.8 do CLAUDE.md): um sistema FALSO, com estado.

Guarda o que estaria em produção (deploys, campanhas, preços, estoque) a partir do cenário gerado
e aplica as ações do catálogo sobre esse estado. Ele conhece a causa injetada (o agente não) e
classifica cada ação:

- **correta**: desfaz a causa (reverter o deploy que quebrou a fatia, repor o produto zerado…);
- **inócua**: não muda nada que importe (abrir chamado, alerta de estoque de produto em dia…);
- **prejudicial**: mexe em algo saudável (reverter um deploy-isca, reativar campanha sem motivo…).

Nenhuma ação sai deste arquivo: não há rede nem sistema real envolvido. Toda execução é
reversível (`revert`). O estado fica em `logs/ops_state.json`.
"""

import hashlib
import json
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from sentinel.data.generator import PRODUCT_PRICE
from sentinel.investigation.actions_catalog import CATALOG
from sentinel.schemas import ActionProposal, Scenario

Outcome = Literal["correct", "innocuous", "harmful"]
STATE_FILE = "ops_state.json"
OUTCOME_LABELS = {"correct": "correta", "innocuous": "inócua", "harmful": "prejudicial"}


class SimulatorError(RuntimeError):
    """Pedido que o simulador recusa (mundo mudou, execução inexistente…)."""


class Deploy(BaseModel):
    scope: dict[str, str]
    ts: str
    description: str
    status: Literal["live", "rolled_back"] = "live"


class TruthCause(BaseModel):
    dimensions: dict[str, str]
    mechanism: str  # deploy, price_change, campaign_end, stock_out, campaign_start, unknown


class Execution(BaseModel):
    execution_id: str
    incident_id: str
    action_id: str
    params: dict[str, str]
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    applied: bool  # False = dry-run (só registrado)
    outcome: Outcome
    effect: str
    reverted: bool = False


class World(BaseModel):
    scenario_id: str
    seed: int
    truth: list[TruthCause]
    deploys: list[Deploy] = Field(default_factory=list)
    campaigns_active: dict[str, bool] = Field(default_factory=dict)
    prices: dict[str, float] = Field(default_factory=dict)
    previous_prices: dict[str, float] = Field(default_factory=dict)
    in_stock: dict[str, bool] = Field(default_factory=dict)
    alerts: list[str] = Field(default_factory=list)
    tickets: list[dict[str, str]] = Field(default_factory=list)
    executions: list[Execution] = Field(default_factory=list)
    undo: dict[str, dict] = Field(default_factory=dict)  # execution_id -> estado anterior

    @property
    def fingerprint(self) -> str:
        text = f"{self.scenario_id}:{self.seed}"
        return hashlib.sha256(text.encode()).hexdigest()[:12]


def _mechanism(cause, scenario: Scenario) -> str:
    for change in scenario.changes:
        if change.scope == cause.dimensions:
            return change.type
    if cause.conversion_factor == 0:
        return "stock_out"
    if cause.price_factor != 1:
        return "price_change"
    if cause.sessions_factor > 1 and "channel" in cause.dimensions:
        return "campaign_start"
    if cause.sessions_factor < 1 and "channel" in cause.dimensions:
        return "campaign_end"
    return "unknown"


def world_from(scenario: Scenario, seed: int) -> World:
    """Estado de produção coerente com o que o cenário injetou e registrou."""
    channels = ("organic", "paid_search", "social", "email")
    world = World(
        scenario_id=scenario.id,
        seed=seed,
        truth=[
            TruthCause(dimensions=c.dimensions, mechanism=_mechanism(c, scenario))
            for c in scenario.causes
        ],
        campaigns_active={c: True for c in channels},
        prices=dict(PRODUCT_PRICE),
        in_stock={p: True for p in PRODUCT_PRICE},
    )
    for change in scenario.changes:
        if change.type == "deploy":
            world.deploys.append(
                Deploy(scope=change.scope, ts=change.ts.isoformat(), description=change.description)
            )
        elif change.type == "campaign_end" and "channel" in change.scope:
            world.campaigns_active[change.scope["channel"]] = False
        elif change.type == "stock_out" and "product" in change.scope:
            world.in_stock[change.scope["product"]] = False
        elif change.type == "price_change" and "product" in change.scope:
            product = change.scope["product"]
            factor = next(
                (c.price_factor for c in scenario.causes if c.dimensions == change.scope), 1.1
            )
            world.previous_prices[product] = world.prices[product]
            world.prices[product] = round(world.prices[product] * factor, 2)
    for cause in world.truth:  # causa sem registro também existe no mundo
        product = cause.dimensions.get("product")
        if cause.mechanism == "stock_out" and product:
            world.in_stock[product] = False
    return world


class OpsSimulator:
    def __init__(self, world: World | None, path: Path | None = None) -> None:
        self.world = world
        self.path = path

    # --- Persistência ----------------------------------------------------------------------

    @classmethod
    def load(cls, logs_dir: Path | None) -> "OpsSimulator":
        if logs_dir is None:
            return cls(None, None)
        path = Path(logs_dir) / STATE_FILE
        if not path.exists():
            return cls(None, path)
        try:
            world = World.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            world = None
        return cls(world, path)

    def save(self) -> None:
        if self.path is None or self.world is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(self.world.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def reset(self, scenario: Scenario, seed: int) -> World:
        self.world = world_from(scenario, seed)
        self.save()
        return self.world

    @property
    def fingerprint(self) -> str | None:
        return None if self.world is None else self.world.fingerprint

    # --- Classificação ---------------------------------------------------------------------

    def classify(self, proposal: ActionProposal) -> tuple[Outcome, str]:
        world = self._require()
        action, params = proposal.action_id, proposal.params
        causes = {json.dumps(c.dimensions, sort_keys=True): c for c in world.truth}
        cause = causes.get(json.dumps(params, sort_keys=True))

        if action == "open_ticket":
            return "innocuous", "chamado aberto; nada muda em produção"
        if action == "rollback_deploy":
            deploy = self._live_deploy(params)
            if deploy is None:
                return "innocuous", "não há deploy ativo com esse escopo; nada a reverter"
            if cause is not None and cause.mechanism == "deploy":
                return "correct", f"deploy revertido: {deploy.description}"
            return "harmful", f"reverteu um deploy saudável: {deploy.description}"
        if action == "restore_price":
            product = params.get("product", "")
            if product not in world.previous_prices:
                return "innocuous", "o preço não tinha mudado; nada a restaurar"
            if cause is not None and cause.mechanism == "price_change":
                return "correct", f"preço de {product} restaurado"
            return "harmful", f"desfez uma mudança de preço que não era a causa ({product})"
        if action == "resume_campaign":
            channel = params.get("channel", "")
            if world.campaigns_active.get(channel, True):
                return "innocuous", "a campanha já estava ativa"
            if cause is not None and cause.mechanism == "campaign_end":
                return "correct", f"campanha de {channel} reativada"
            return "harmful", f"reativou a campanha de {channel} sem relação com a queda"
        if action == "restock_alert":
            product = params.get("product", "")
            if cause is not None and cause.mechanism == "stock_out":
                return "correct", f"reposição de {product} acionada"
            return "innocuous", f"alerta de estoque para {product}, que não estava em falta"
        raise SimulatorError("ação fora do catálogo")

    # --- Execução --------------------------------------------------------------------------

    def execute(
        self,
        incident_id: str,
        proposal: ActionProposal,
        *,
        dry_run: bool,
        world_fingerprint: str | None,
    ) -> Execution:
        world = self._require()
        if proposal.action_id not in CATALOG:
            raise SimulatorError("ação fora do catálogo")
        if world_fingerprint is not None and world_fingerprint != world.fingerprint:
            raise SimulatorError(
                "o mundo simulado mudou desde o incidente (outro cenário foi gerado)"
            )
        outcome, effect = self.classify(proposal)
        execution = Execution(
            execution_id=f"ex-{secrets.token_hex(4)}",
            incident_id=incident_id,
            action_id=proposal.action_id,
            params=dict(proposal.params),
            applied=not dry_run,
            outcome=outcome,
            effect=effect if not dry_run else f"(dry-run, nada aplicado) {effect}",
        )
        if not dry_run:
            world.undo[execution.execution_id] = self._snapshot()
            self._apply(proposal)
        world.executions.append(execution)
        self.save()
        return execution

    def revert(self, execution_id: str) -> Execution:
        world = self._require()
        execution = next((e for e in world.executions if e.execution_id == execution_id), None)
        if execution is None:
            raise SimulatorError("execução não encontrada")
        if execution.reverted or not execution.applied:
            raise SimulatorError("nada a reverter")
        snapshot = world.undo.pop(execution_id, None)
        if snapshot is None:
            raise SimulatorError("sem estado anterior guardado")
        data = json.loads(world.model_dump_json())
        data.update(snapshot)
        self.world = world = World.model_validate(data)
        execution = next(e for e in world.executions if e.execution_id == execution_id)
        execution.reverted = True
        self.save()
        return execution

    def _apply(self, proposal: ActionProposal) -> None:
        world = self._require()
        params = proposal.params
        if proposal.action_id == "rollback_deploy":
            deploy = self._live_deploy(params)
            if deploy is not None:
                deploy.status = "rolled_back"
        elif proposal.action_id == "restore_price":
            product = params.get("product", "")
            if product in world.previous_prices:
                world.prices[product] = world.previous_prices.pop(product)
        elif proposal.action_id == "resume_campaign":
            world.campaigns_active[params.get("channel", "")] = True
        elif proposal.action_id == "restock_alert":
            product = params.get("product", "")
            world.alerts.append(f"repor {product}")
            if product in world.in_stock:
                world.in_stock[product] = True
        elif proposal.action_id == "open_ticket":
            world.tickets.append(dict(params))

    def _snapshot(self) -> dict:
        """Partes do estado que uma ação pode mudar, como JSON simples (para o `revert`)."""
        fields = {"deploys", "campaigns_active", "prices", "previous_prices", "in_stock"}
        return json.loads(self._require().model_dump_json(include=fields | {"alerts", "tickets"}))

    def _live_deploy(self, scope: dict[str, str]) -> Deploy | None:
        world = self._require()
        return next((d for d in world.deploys if d.scope == scope and d.status == "live"), None)

    def _require(self) -> World:
        if self.world is None:
            raise SimulatorError("simulador sem mundo: rode `sentinel generate` antes")
        return self.world
