"""Grafo do agente em LangGraph (seção 5 do CLAUDE.md).

observar → detectar → memória → investigar (⟲ rodadas) → testar hipóteses → refutar → concluir →
analisar → verificar → policy → (relatório | pedir aprovação | executar no simulador) → registrar.

- `PLAN_MODE=deterministic`: investigar roda o plano fixo; testar e refutar não usam LLM; o
  relatório sai do narrador de template. É a linha de base.
- `PLAN_MODE=llm`: o LLM escolhe as consultas, nomeia hipóteses e contesta a líder; o código
  executa, mede e confirma (ou não).

O grafo não bloqueia esperando o humano: o nó de aprovação usa `interrupt` e o estado fica no
checkpoint (`logs/checkpoints.db`). `resume_incident` retoma com a decisão (Telegram, web, CLI ou
expiração). Cada nó grava uma linha no audit.
"""

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from sentinel.agent.analyst import write_report
from sentinel.agent.placeholders import fill
from sentinel.agent.policy import Decision, PolicyConfig, decide
from sentinel.agent.render import KIND_LABELS, render_report
from sentinel.agent.state import AgentState
from sentinel.agent.verifier import verify
from sentinel.audit import AuditLog
from sentinel.config import Settings
from sentinel.data.db import connect_readonly
from sentinel.data.profile import SchemaProfile
from sentinel.detection.detector import DetectorParams, detect, load_daily_series, select_root
from sentinel.investigation.actions_catalog import describe, proposal_for
from sentinel.investigation.plan import investigate as investigate_plan
from sentinel.investigation.tools import ToolContext
from sentinel.llm.base import LLMClient
from sentinel.memory.factory import Memory
from sentinel.memory.incidents import Incident, MemoryStoreError
from sentinel.notify.base import Notifier, approval_message
from sentinel.ops.simulator import OpsSimulator, SimulatorError
from sentinel.records import (
    ACTIVE,
    DecisionEntry,
    DecisionLog,
    History,
    IncidentRecord,
    RecordStore,
    check_incident_id,
    dump,
    history_from,
    new_incident_id,
)
from sentinel.schemas import ActionProposal, Anomaly, Finding, Hypothesis, Report
from sentinel.security import describe_error

CHECKPOINT_FILE = "checkpoints.db"
PROFILE_FILE = "profile.json"
KILL_SWITCH_FILE = "KILL_SWITCH"

ApprovalDecision = Literal["approve", "ignore", "expired"]


class AgentError(RuntimeError):
    """Pedido que o grafo recusa (incidente que não está pendente, banco ausente…)."""


@dataclass
class Runtime:
    """O que os nós usam e não vai para o checkpoint (conexões, clientes, caches)."""

    settings: Settings
    db_path: Path | None = None
    llm: LLMClient | None = None
    memory: Memory | None = None
    notifier: Notifier | None = None
    simulator: OpsSimulator | None = None
    audit: AuditLog = field(default_factory=lambda: AuditLog(None))
    records: RecordStore | None = None
    decisions: DecisionLog | None = None
    origin: dict[str, Any] = field(default_factory=dict)
    history: History | None = None  # fixa a história (benchmark); None = log de decisões
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    ctx: ToolContext | None = None
    con: sqlite3.Connection | None = None

    @property
    def logs_dir(self) -> Path | None:
        return None if self.records is None else self.records.dir.parent

    @property
    def mode(self) -> str:
        return self.settings.plan_mode

    def kill_switch(self) -> bool:
        if self.settings.kill_switch:
            return True
        return self.logs_dir is not None and (self.logs_dir / KILL_SWITCH_FILE).exists()

    def open(self) -> ToolContext:
        if self.ctx is None:
            if self.db_path is None or not Path(self.db_path).exists():
                raise AgentError("banco de eventos não encontrado; rode `sentinel generate`")
            self.con = connect_readonly(self.db_path)
            store = self.memory.incidents if self.memory is not None else None
            self.ctx = ToolContext(self.con, self.settings, memory=store)
        return self.ctx

    def close(self) -> None:
        if self.con is not None:
            self.con.close()
            self.con = None
        self.ctx = None

    def current_history(self) -> History:
        if self.history is not None:
            return self.history
        if self.decisions is None:
            return History()
        return history_from(self.decisions.entries(), self.clock())


# --- Ajudantes -------------------------------------------------------------------------------


def _anomaly(state: AgentState) -> Anomaly:
    return Anomaly.model_validate(state["anomaly"])


def _hypotheses(items: list[dict] | None) -> list[Hypothesis]:
    return [Hypothesis.model_validate(h) for h in items or []]


def _findings(rt: Runtime, state: AgentState) -> list[Finding]:
    if rt.ctx is not None:
        return list(rt.ctx.findings)
    return [Finding.model_validate(f) for f in state.get("findings", [])]


def _profile(rt: Runtime) -> SchemaProfile | None:
    return None if rt.ctx is None else rt.ctx.profile


def _where(dimensions: dict[str, str]) -> str:
    return ", ".join(f"{k} = {v}" for k, v in dimensions.items()) or "todas as fatias"


def _track_profile(rt: Runtime, profile: SchemaProfile) -> list[str]:
    """Compara com o último perfil visto (logs/profile.json) e guarda o atual."""
    if rt.logs_dir is None:
        return []
    path = rt.logs_dir / PROFILE_FILE
    previous = None
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            previous = None
    changes: list[str] = []
    if previous and previous.get("fingerprint") != profile.fingerprint:
        changes = profile.diff(previous.get("structure", {})) or ["estrutura mudou"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"fingerprint": profile.fingerprint, "structure": profile.structure()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return changes


def _memory_call(rt: Runtime, incident_id: str, what: str, call: Callable[[], Any]) -> None:
    """A memória ajuda, mas não é obrigatória: falha vira linha no audit, não erro."""
    if rt.memory is None:
        return
    try:
        call()
    except (MemoryStoreError, RuntimeError, ValueError) as error:
        rt.audit.write(
            incident_id, "memory", fallback=True, detail=what, error=describe_error(error)
        )


def _remember(rt: Runtime, record: IncidentRecord) -> None:
    if rt.memory is None or not record.anomaly or record.report is None:
        return
    store = rt.memory.incidents

    def add_or_update() -> None:
        if store.get(record.incident_id) is None:
            store.add(
                Incident(
                    incident_id=record.incident_id,
                    metric=record.anomaly["metric"],
                    direction=record.anomaly["direction"],
                    status=record.report["status"],
                    causes=[h["dimensions"] for h in record.hypotheses],
                    summary=record.rendered or "",
                    state=record.state,
                    created_at=record.created_at,
                )
            )
        else:
            store.set_state(record.incident_id, record.state)

    _memory_call(rt, record.incident_id, "guardar incidente", add_or_update)


# --- Construção do grafo ---------------------------------------------------------------------


def build_graph(rt: Runtime, checkpointer: Any) -> Any:
    s = rt.settings
    audit = rt.audit

    def observe(state: AgentState) -> dict:
        iid = state["incident_id"]
        with audit.step(iid, "observe") as extra:
            ctx = rt.open()

            def listen(finding: Finding, ms: float) -> None:
                audit.write(
                    iid,
                    "tool",
                    tool=finding.tool,
                    params=finding.params,
                    duration_ms=ms,
                    finding_id=finding.id,
                    finding=dump(finding),
                )

            ctx.listener = listen
            profile = ctx.profile
            changes = _track_profile(rt, profile)
            extra.update(
                profile=profile.fingerprint, profile_changes=changes, profile_full=dump(profile)
            )
        return {
            "profile": {"fingerprint": profile.fingerprint, **profile.structure()},
            "started_at": time.perf_counter(),
            "rounds": 0,
            "candidates": [],
            "unconfirmed": [],
            "investigation": {"llm_calls": 0, "invalid": 0, "blocked": 0, "fallbacks": 0},
        }

    def detect_node(state: AgentState) -> dict:
        iid = state["incident_id"]
        ctx = rt.open()
        with audit.step(iid, "detect") as extra:
            series = load_daily_series(ctx.con, ctx.profile)
            as_of = date.fromisoformat(state["as_of"]) if state.get("as_of") else None
            found = detect(series, DetectorParams.from_settings(s), as_of)
            root = select_root(found, ctx.profile.metrics)
            extra.update(
                anomalies=[a.metric for a in found], root=None if root is None else root.metric
            )
        _resolve_recovered(rt, iid, {a.metric for a in found})
        return {
            "anomaly": None if root is None else dump(root),
            "anomalies": [dump(a) for a in found],
        }

    def recall(state: AgentState) -> dict:
        iid, anomaly = state["incident_id"], _anomaly(state)
        with audit.step(iid, "memory") as extra:
            finding = rt.open().run(
                "find_similar_incidents",
                {"metric": anomaly.metric, "direction": anomaly.direction},
            )
            extra.update(finding_id=finding.id, similar=finding.result.get("count", 0))
        return {}

    def investigate(state: AgentState) -> dict:
        iid, anomaly, ctx = state["incident_id"], _anomaly(state), rt.open()
        if rt.mode != "llm" or rt.llm is None:
            with audit.step(iid, "investigate", plan="deterministic") as extra:
                hypotheses = investigate_plan(ctx, anomaly)
                extra.update(hypotheses=len(hypotheses))
            return {"hypotheses": dump(hypotheses), "explore_done": True, "rounds": 1}

        from sentinel.agent.investigator import explore_round

        rounds = state.get("rounds", 0)
        elapsed = time.perf_counter() - state.get("started_at", time.perf_counter())
        result = explore_round(
            ctx,
            anomaly,
            rt.llm,
            candidates=state.get("candidates", []),
            round_number=rounds + 1,
            max_rounds=s.max_rounds,
            time_left_s=s.time_budget_s - elapsed,
        )
        stats = dict(state.get("investigation", {}))
        for key, value in result.stats.items():
            stats[key] = stats.get(key, 0) + value
        audit.write(
            iid,
            "investigate",
            llm=True,
            fallback=result.fallback,
            round=rounds + 1,
            decision=result.decision,
            reasoning=result.reasoning,
            finding_id=result.finding_id,
            candidates=result.candidates,
        )
        return {
            "rounds": rounds + 1,
            "explore_done": result.done,
            "candidates": result.candidates,
            "investigation": stats,
        }

    def after_investigate(state: AgentState) -> str:
        if state.get("explore_done"):
            return "test"
        elapsed = time.perf_counter() - state.get("started_at", time.perf_counter())
        if state.get("rounds", 0) >= s.max_rounds or elapsed >= s.time_budget_s:
            return "test"
        return "investigate"

    def test(state: AgentState) -> dict:
        if rt.mode != "llm" or rt.llm is None:
            return {}  # o plano determinístico já testou e confirmou
        from sentinel.agent.investigator import confirm

        iid, anomaly, ctx = state["incident_id"], _anomaly(state), rt.open()
        with audit.step(iid, "test_hypotheses") as extra:
            hypotheses, unconfirmed = confirm(ctx, anomaly, state.get("candidates", []))
            extra.update(confirmed=len(hypotheses), unconfirmed=len(unconfirmed))
        return {"hypotheses": dump(hypotheses), "unconfirmed": unconfirmed}

    def refute(state: AgentState) -> dict:
        hypotheses = _hypotheses(state.get("hypotheses"))
        if rt.mode != "llm" or rt.llm is None or not hypotheses:
            return {"refutation": None}
        from sentinel.agent.investigator import refute_leader

        iid, anomaly, ctx = state["incident_id"], _anomaly(state), rt.open()
        start = time.perf_counter()
        outcome = refute_leader(ctx, anomaly, rt.llm, hypotheses)
        stats = dict(state.get("investigation", {}))
        for key, value in outcome.stats.items():
            stats[key] = stats.get(key, 0) + value
        audit.write(
            iid,
            "refute",
            llm=True,
            duration_ms=(time.perf_counter() - start) * 1000,
            fallback=outcome.fallback,
            objection=outcome.objection,
            finding_id=outcome.finding_id,
            upheld=outcome.upheld,
        )
        unconfirmed = list(state.get("unconfirmed", []))
        if outcome.upheld:  # a objeção se sustentou: a causa é rebaixada
            unconfirmed += [h.dimensions for h in hypotheses]
            hypotheses = outcome.remaining
        return {
            "hypotheses": dump(hypotheses),
            "unconfirmed": unconfirmed,
            "refutation": outcome.summary(),
            "investigation": stats,
        }

    def conclude(state: AgentState) -> dict:
        iid, anomaly = state["incident_id"], _anomaly(state)
        hypotheses = _hypotheses(state.get("hypotheses"))
        findings = _findings(rt, state)
        with audit.step(iid, "conclude") as extra:
            proposal = proposal_for(hypotheses, findings, anomaly.direction)
            duplicate = None
            if rt.records is not None:
                signature = (
                    anomaly.metric,
                    anomaly.direction,
                    "|".join(
                        sorted(
                            ",".join(f"{k}={v}" for k, v in sorted(h.dimensions.items()))
                            for h in hypotheses
                        )
                    ),
                )
                previous = rt.records.find_active(signature, s.cooldown_h, rt.clock())
                duplicate = None if previous is None else previous.incident_id
            extra.update(
                status="identified" if hypotheses else "inconclusive",
                causes=[h.dimensions for h in hypotheses],
                proposal=None if proposal is None else proposal.action_id,
                duplicate_of=duplicate,
            )
        return {"proposal": dump(proposal), "duplicate_of": duplicate}

    def analyze(state: AgentState) -> dict:
        iid, anomaly = state["incident_id"], _anomaly(state)
        hypotheses = _hypotheses(state.get("hypotheses"))
        findings = _findings(rt, state)
        llm = rt.llm if rt.mode == "llm" else None
        proposal = state.get("proposal")
        unconfirmed = list(state.get("unconfirmed", []))
        with audit.step(iid, "analyze", llm=llm is not None) as extra:
            outcome = write_report(
                llm,
                anomaly,
                hypotheses,
                findings,
                unconfirmed,
                _profile(rt),
                proposal=None if proposal is None else describe(ActionProposal(**proposal)),
            )
            extra.update(
                source=outcome.source,
                fallback=llm is not None and outcome.source == "template",
                rejection=None if outcome.rejection is None else dump(outcome.rejection.problems),
                error=outcome.error,
            )
        return {
            "report": dump(outcome.report),
            "report_source": outcome.source,
            "report_problems": []
            if outcome.rejection is None
            else dump(outcome.rejection.problems),
            "report_error": outcome.error,
            "findings": dump(findings),
        }

    def verify_node(state: AgentState) -> dict:
        iid = state["incident_id"]
        findings = _findings(rt, state)
        report = Report.model_validate(state["report"])
        proposal = state.get("proposal")
        if proposal is not None and report.status == "identified":
            report = report.model_copy(update={"proposed_action": ActionProposal(**proposal)})
        with audit.step(iid, "verify") as extra:
            check = verify(report, findings, _profile(rt))
            if not check.ok:  # não deveria acontecer: o analista já verificou
                report = report.model_copy(update={"opinions": [], "proposed_action": None})
            rendered = render_report(report, findings, _profile(rt))
            extra.update(ok=check.ok, report=dump(report), rendered=rendered)
        return {
            "report": dump(report),
            "rendered": rendered,
            "verified": not state.get("report_problems"),
        }

    def policy(state: AgentState) -> dict:
        iid = state["incident_id"]
        report = Report.model_validate(state["report"])
        proposal = report.proposed_action
        history = rt.current_history()
        config = PolicyConfig.from_settings(s, kill_switch=rt.kill_switch())
        with audit.step(iid, "policy") as extra:
            decision = decide(
                report,
                proposal,
                history,
                config,
                verified=state.get("verified", True),
                duplicate_of=state.get("duplicate_of"),
            )
            extra.update(
                decision=decision.kind,
                reasons=decision.reasons,
                history={"approved": history.approved, "ignored": history.ignored},
            )
        world = None if rt.simulator is None else rt.simulator.fingerprint
        return {"decision": dump(decision), "world": world}

    def route_policy(state: AgentState) -> str:
        kind = state["decision"]["kind"]
        return {"ask_human": "notify", "auto_execute": "execute"}.get(kind, "register")

    def notify(state: AgentState) -> dict:
        iid = state["incident_id"]
        record = _record(rt, state, "pending")
        if rt.records is not None:
            record = rt.records.save(record)
        _remember(rt, record)
        with audit.step(iid, "notify") as extra:
            sent = False
            if rt.notifier is not None:
                try:
                    rt.notifier.request_approval(iid, _message(rt, state))
                    sent = True
                except Exception as error:  # rede fora do ar não derruba o incidente
                    extra.update(fallback=True, error=describe_error(error))
            extra.update(sent=sent)
        return {}

    def approval(state: AgentState) -> dict:
        answer = interrupt({"incident_id": state["incident_id"], "kind": "approval"})
        return {"approval": answer}

    def route_approval(state: AgentState) -> str:
        answer = state.get("approval") or {}
        return "execute" if answer.get("decision") == "approve" else "register"

    def execute(state: AgentState) -> dict:
        iid = state["incident_id"]
        report = Report.model_validate(state["report"])
        proposal = report.proposed_action
        decision = Decision.model_validate(state["decision"])
        automatic = decision.kind == "auto_execute"
        dry_run = s.dry_run
        with audit.step(iid, "execute") as extra:
            if proposal is None or rt.simulator is None:
                extra.update(executed=False, reason="sem ação ou sem simulador")
                return {"execution": None}
            if rt.kill_switch():
                extra.update(executed=False, reason="kill switch ligado")
                return {"execution": {"blocked": "kill switch ligado"}}
            try:
                execution = rt.simulator.execute(
                    iid, proposal, dry_run=dry_run, world_fingerprint=state.get("world")
                )
            except SimulatorError as error:
                extra.update(executed=False, reason=str(error))
                return {"execution": {"blocked": str(error)}}
            extra.update(
                executed=execution.applied,
                dry_run=dry_run,
                outcome=execution.outcome,
                action=proposal.action_id,
            )
        if automatic and rt.decisions is not None:
            rt.decisions.append(
                DecisionEntry(
                    incident_id=iid,
                    decision="dry_run" if dry_run else "auto_execute",
                    via="policy",
                    action_id=proposal.action_id,
                    outcome=execution.outcome,
                )
            )
        return {"execution": dump(execution)}

    def register(state: AgentState) -> dict:
        iid = state["incident_id"]
        decision = state["decision"]["kind"]
        answer = state.get("approval")
        if answer is not None:
            final = {"approve": "approved", "ignore": "ignored"}.get(answer["decision"], "expired")
        elif decision == "auto_execute":
            final = "approved"
        else:
            final = "open"
        record = _record(rt, state, final)
        if rt.records is not None:
            existing = rt.records.get(iid)
            if existing is not None and existing.state == "resolved":
                record = record.model_copy(update={"state": "resolved"})
            record = rt.records.save(record)
        _remember(rt, record)
        audit.write(iid, "register", state=record.state, decision=decision)
        return {"final_state": record.state}

    graph = StateGraph(AgentState)
    for name, node in (
        ("observe", observe),
        ("detect", detect_node),
        ("recall", recall),
        ("investigate", investigate),
        ("test", test),
        ("refute", refute),
        ("conclude", conclude),
        ("analyze", analyze),
        ("verify", verify_node),
        ("policy", policy),
        ("notify", notify),
        ("approval", approval),
        ("execute", execute),
        ("register", register),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "observe")
    graph.add_edge("observe", "detect")
    graph.add_conditional_edges(
        "detect", lambda st: "recall" if st.get("anomaly") else END, ["recall", END]
    )
    graph.add_edge("recall", "investigate")
    graph.add_conditional_edges("investigate", after_investigate, ["investigate", "test"])
    graph.add_edge("test", "refute")
    graph.add_edge("refute", "conclude")
    graph.add_edge("conclude", "analyze")
    graph.add_edge("analyze", "verify")
    graph.add_edge("verify", "policy")
    graph.add_conditional_edges("policy", route_policy, ["notify", "execute", "register"])
    graph.add_edge("notify", "approval")
    graph.add_conditional_edges("approval", route_approval, ["execute", "register"])
    graph.add_edge("execute", "register")
    graph.add_edge("register", END)
    return graph.compile(checkpointer=checkpointer)


def _record(rt: Runtime, state: AgentState, final_state: str) -> IncidentRecord:
    iid = state["incident_id"]
    existing = rt.records.get(iid) if rt.records is not None else None
    started = state.get("started_at")
    record = existing or IncidentRecord(
        incident_id=iid,
        mode=rt.mode,  # type: ignore[arg-type]
        llm_model=rt.llm.model if rt.llm is not None and rt.mode == "llm" else None,
        autonomy_level=rt.settings.autonomy_level,
        dry_run=rt.settings.dry_run,
        origin=rt.origin,
        created_at=rt.clock(),
    )
    findings = dump(rt.ctx.findings) if rt.ctx is not None else state.get("findings", [])
    return record.model_copy(
        update={
            "state": final_state,
            "profile": state.get("profile", record.profile),
            "anomaly": state.get("anomaly"),
            "anomalies": state.get("anomalies", []),
            "findings": findings,
            "hypotheses": state.get("hypotheses", []),
            "unconfirmed": state.get("unconfirmed", []),
            "refutation": state.get("refutation"),
            "report": state.get("report"),
            "report_source": state.get("report_source"),
            "report_problems": state.get("report_problems", []),
            "report_error": state.get("report_error"),
            "rendered": state.get("rendered"),
            "proposal": (state.get("report") or {}).get("proposed_action"),
            "decision": state.get("decision"),
            "approval": state.get("approval"),
            "execution": state.get("execution"),
            "duplicate_of": state.get("duplicate_of"),
            "investigation": {**state.get("investigation", {}), "rounds": state.get("rounds", 0)},
            "duration_ms": record.duration_ms
            or (round((time.perf_counter() - started) * 1000, 1) if started else 0.0),
        }
    )


def _message(rt: Runtime, state: AgentState) -> str:
    findings = _findings(rt, state)
    report = Report.model_validate(state["report"])
    profile = _profile(rt)
    facts = [fill(f.text, findings, profile) for f in report.facts]
    opinions = [
        f"[{KIND_LABELS[o.kind]}, confiança {o.confidence:.0%}] {fill(o.text, findings, profile)}"
        for o in report.opinions
    ]
    causes = [f"{_where(c.dimensions)} (explica {c.explained_share:.0%})" for c in report.causes]
    action = None if report.proposed_action is None else describe(report.proposed_action)
    return approval_message(state["incident_id"], facts, causes, opinions, report.evidence, action)


def _resolve_recovered(rt: Runtime, run_id: str, anomalous: set[str]) -> None:
    """Incidente ativo cuja métrica voltou ao normal passa a `resolved`."""
    if rt.records is None:
        return
    for record in rt.records.list():
        if record.state not in ACTIVE or not record.anomaly:
            continue
        if record.anomaly["metric"] in anomalous:
            continue
        updated = rt.records.update(record.incident_id, state="resolved", resolved_by=run_id)
        _remember(rt, updated)
        rt.audit.write(record.incident_id, "resolve", state="resolved", by=run_id)


# --- Execução ---------------------------------------------------------------------------------


@dataclass
class RunResult:
    incident_id: str
    status: Literal["quiet", "pending", "done"]
    state: dict[str, Any]
    record: IncidentRecord | None


def _checkpointer(rt: Runtime) -> tuple[Any, sqlite3.Connection | None]:
    if rt.logs_dir is None:
        return InMemorySaver(), None
    from langgraph.checkpoint.sqlite import SqliteSaver

    rt.logs_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(rt.logs_dir / CHECKPOINT_FILE, check_same_thread=False)
    return SqliteSaver(con), con


def _config(incident_id: str) -> dict:
    return {"configurable": {"thread_id": incident_id}, "recursion_limit": 100}


def _result(rt: Runtime, incident_id: str, out: dict) -> RunResult:
    if not out.get("anomaly"):
        return RunResult(incident_id, "quiet", out, None)
    record = rt.records.get(incident_id) if rt.records is not None else None
    if record is None:
        record = _record(rt, out, out.get("final_state", "open"))
    status = "pending" if "__interrupt__" in out else "done"
    return RunResult(incident_id, status, out, record)


def run_incident(
    rt: Runtime,
    as_of: date | None = None,
    incident_id: str | None = None,
    checkpointer: Any = None,
) -> RunResult:
    """Uma passada do agente sobre o banco. `pending` = esperando aprovação humana."""
    incident_id = incident_id or new_incident_id(rt.clock())
    own, con = (checkpointer, None) if checkpointer is not None else _checkpointer(rt)
    try:
        graph = build_graph(rt, own)
        state: AgentState = {
            "incident_id": incident_id,
            "as_of": None if as_of is None else as_of.isoformat(),
        }
        out = graph.invoke(state, _config(incident_id))
        return _result(rt, incident_id, out)
    finally:
        rt.close()
        if con is not None:
            con.close()


def pending_approval(rt: Runtime, incident_id: str, checkpointer: Any = None) -> bool:
    own, con = (checkpointer, None) if checkpointer is not None else _checkpointer(rt)
    try:
        snapshot = build_graph(rt, own).get_state(_config(check_incident_id(incident_id)))
        return "approval" in (snapshot.next or ())
    finally:
        if con is not None:
            con.close()


def resume_incident(
    rt: Runtime,
    incident_id: str,
    decision: ApprovalDecision,
    via: str,
    checkpointer: Any = None,
) -> RunResult:
    """Retoma um incidente pendente com a decisão humana (ou a expiração)."""
    check_incident_id(incident_id)
    own, con = (checkpointer, None) if checkpointer is not None else _checkpointer(rt)
    try:
        graph = build_graph(rt, own)
        config = _config(incident_id)
        snapshot = graph.get_state(config)
        if "approval" not in (snapshot.next or ()):
            raise AgentError("o incidente não está esperando aprovação")
        answer = {"decision": decision, "via": via, "at": rt.clock().isoformat()}
        rt.audit.write(incident_id, "approval", decision=decision, via=via)
        if rt.decisions is not None:
            report = snapshot.values.get("report") or {}
            action = (report.get("proposed_action") or {}).get("action_id")
            rt.decisions.append(
                DecisionEntry(incident_id=incident_id, decision=decision, via=via, action_id=action)
            )
        out = graph.invoke(Command(resume=answer), config)
        return _result(rt, incident_id, out)
    finally:
        rt.close()
        if con is not None:
            con.close()


def expire_overdue(rt: Runtime, checkpointer: Any = None) -> list[str]:
    """Pendentes há mais de `APPROVAL_TIMEOUT_S` viram `expired`."""
    if rt.records is None:
        return []
    limit = rt.clock() - timedelta(seconds=rt.settings.approval_timeout_s)
    expired = []
    for record in rt.records.list():
        if record.state == "pending" and record.created_at <= limit:
            try:
                resume_incident(rt, record.incident_id, "expired", "timeout", checkpointer)
            except AgentError:
                rt.records.update(record.incident_id, state="expired")
            expired.append(record.incident_id)
    return expired


# --- Replay -----------------------------------------------------------------------------------


@dataclass
class Replay:
    incident_id: str
    rendered: str
    matches: bool  # o texto reconstruído é igual ao gravado
    ok: bool  # o verificador aprova o relatório reconstruído
    findings: int


def replay(entries: list[dict], incident_id: str) -> Replay:
    """Reconstrói o relatório só a partir do audit (determinístico: sem banco nem LLM)."""
    findings: list[Finding] = []
    report: Report | None = None
    recorded: str | None = None
    profile: SchemaProfile | None = None
    for entry in entries:
        if entry.get("incident_id") != incident_id:
            continue
        if entry.get("node") == "observe" and entry.get("profile_full"):
            profile = SchemaProfile.model_validate(entry["profile_full"])
        if entry.get("node") == "tool" and entry.get("finding"):
            findings.append(Finding.model_validate(entry["finding"]))
        if entry.get("node") == "verify" and entry.get("report"):
            report = Report.model_validate(entry["report"])
            recorded = entry.get("rendered")
    if report is None:
        raise AgentError("o audit não tem o relatório desse incidente")
    rendered = render_report(report, findings, profile)
    check = verify(report, findings, profile)
    return Replay(incident_id, rendered, rendered == recorded, check.ok, len(findings))
