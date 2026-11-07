"""Grafo no modo determinístico (etapa 5): fecha em identified e inconclusive, replay, ciclo do
incidente (pendente, aprovado, ignorado, expirado, resolvido), deduplicação e audit seguro."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from sentinel.agent.graph import (
    AgentError,
    Runtime,
    expire_overdue,
    pending_approval,
    replay,
    resume_incident,
    run_incident,
)
from sentinel.audit import AuditLog, read_audit
from sentinel.config import Settings
from sentinel.llm.embeddings import FakeEmbedder
from sentinel.memory.factory import Memory
from sentinel.memory.incidents import InMemoryIncidentStore
from sentinel.memory.knowledge import InMemoryKnowledgeStore
from sentinel.notify.base import FakeNotifier
from sentinel.ops.simulator import OpsSimulator
from sentinel.records import DecisionLog, RecordStore
from sentinel.security import register_secrets


@pytest.fixture
def make_runtime(tmp_path, db_of, scenarios, seed):
    logs = tmp_path / "logs"

    def make(scenario_id: str, *, memory: Memory | None = None, **overrides) -> Runtime:
        settings = Settings(_env_file=None, logs_dir=logs, **overrides)
        simulator = OpsSimulator.load(logs)
        if simulator.world is None or simulator.world.scenario_id != scenario_id:
            simulator.reset(scenarios[scenario_id], seed)
        return Runtime(
            settings=settings,
            db_path=db_of(scenario_id),
            audit=AuditLog.in_dir(logs),
            records=RecordStore(logs),
            decisions=DecisionLog(logs),
            simulator=simulator,
            notifier=FakeNotifier(),
            memory=memory,
            origin={"scenario": scenario_id},
        )

    make.logs = logs  # type: ignore[attr-defined]
    return make


def test_identified_incident_waits_for_approval(make_runtime):
    rt = make_runtime("s01")
    result = run_incident(rt)
    assert result.status == "pending"
    record = result.record
    assert record.state == "pending"
    assert record.report["status"] == "identified"
    assert [h["dimensions"] for h in record.hypotheses] == [
        {"platform": "android", "app_version": "3.2"}
    ]
    assert record.proposal["action_id"] == "rollback_deploy"
    assert record.decision["kind"] == "ask_human"
    [(incident_id, text)] = rt.notifier.approvals
    assert incident_id == record.incident_id
    assert "android" in text and "NÃO verificada" not in text  # sem opinião no modo determinístico
    assert pending_approval(make_runtime("s01"), record.incident_id)


def test_approval_resumes_in_another_runtime_and_runs_the_simulator(make_runtime):
    first = run_incident(make_runtime("s01"))
    resumed = resume_incident(make_runtime("s01"), first.incident_id, "approve", "cli")
    assert resumed.status == "done"
    assert resumed.record.state == "approved"
    assert resumed.record.approval["via"] == "cli"
    assert resumed.record.execution["outcome"] == "correct"
    assert resumed.record.execution["applied"] is False  # DRY_RUN=true por padrão
    entries = DecisionLog(make_runtime.logs).entries()
    assert [(e.decision, e.via) for e in entries] == [("approve", "cli")]
    with pytest.raises(AgentError, match="não está esperando"):
        resume_incident(make_runtime("s01"), first.incident_id, "approve", "cli")


def test_ignored_incident_executes_nothing(make_runtime):
    first = run_incident(make_runtime("s06"))
    resumed = resume_incident(make_runtime("s06"), first.incident_id, "ignore", "web")
    assert resumed.record.state == "ignored"
    assert resumed.record.execution is None


def test_inconclusive_incident_is_only_reported(make_runtime):
    result = run_incident(make_runtime("s19"))
    assert result.status == "done"
    assert result.record.report["status"] == "inconclusive"
    assert result.record.report["causes"] == []
    assert result.record.decision["kind"] == "report_only"
    assert result.record.state == "open"


def test_quiet_data_creates_no_incident(make_runtime):
    result = run_incident(make_runtime("s15"))
    assert result.status == "quiet"
    assert result.record is None
    assert RecordStore(make_runtime.logs).list() == []


def test_l1_reports_and_l3_without_history_still_asks(make_runtime):
    l1 = run_incident(make_runtime("s01", autonomy_level=1))
    assert l1.record.decision["kind"] == "report_only"
    l3 = run_incident(make_runtime("s06", autonomy_level=3))
    assert l3.record.decision["kind"] == "ask_human"
    assert any("histórico curto" in r for r in l3.record.decision["reasons"])


def test_kill_switch_file_blocks_actions(make_runtime):
    make_runtime.logs.mkdir(parents=True, exist_ok=True)
    (make_runtime.logs / "KILL_SWITCH").write_text("", encoding="utf-8")
    result = run_incident(make_runtime("s01"))
    assert result.record.decision["kind"] == "block"
    assert result.status == "done"


def test_replay_rebuilds_the_report_from_the_audit(make_runtime):
    for scenario_id in ("s01", "s19", "s12"):
        result = run_incident(make_runtime(scenario_id))
        entries = read_audit(make_runtime.logs / "audit.jsonl")
        rebuilt = replay(entries, result.incident_id)
        assert rebuilt.matches and rebuilt.ok
        assert rebuilt.rendered == result.record.rendered


def test_every_node_writes_an_audit_line(make_runtime):
    result = run_incident(make_runtime("s01"))
    nodes = [e["node"] for e in read_audit(make_runtime.logs / "audit.jsonl", result.incident_id)]
    for node in ("observe", "detect", "memory", "investigate", "conclude", "analyze", "verify"):
        assert node in nodes
    assert nodes.count("tool") >= 4
    assert nodes[-1] == "notify"


def test_same_open_incident_is_not_alerted_twice(make_runtime):
    first = run_incident(make_runtime("s01"))
    rt = make_runtime("s01")
    second = run_incident(rt)
    assert second.record.duplicate_of == first.incident_id
    assert second.record.decision["kind"] == "report_only"
    assert rt.notifier.approvals == []


def test_incident_resolves_when_the_metric_is_back_to_normal(make_runtime):
    first = run_incident(make_runtime("s01"))
    run_incident(make_runtime("s15"))  # dados sem anomalia
    record = RecordStore(make_runtime.logs).get(first.incident_id)
    assert record.state == "resolved"
    assert record.resolved_by is not None


def test_overdue_approvals_expire(make_runtime):
    first = run_incident(make_runtime("s01"))
    later = make_runtime("s01")
    later.clock = lambda: datetime.now(UTC) + timedelta(hours=2)
    assert expire_overdue(later) == [first.incident_id]
    assert RecordStore(make_runtime.logs).get(first.incident_id).state == "expired"


def test_memory_receives_the_incident_and_its_decision(make_runtime):
    embedder = FakeEmbedder()
    memory = Memory(InMemoryIncidentStore(embedder), InMemoryKnowledgeStore(embedder), "ram")
    first = run_incident(make_runtime("s01", memory=memory))
    assert memory.incidents.get(first.incident_id).state == "pending"
    resume_incident(make_runtime("s01", memory=memory), first.incident_id, "approve", "cli")
    assert memory.incidents.get(first.incident_id).state == "approved"
    assert memory.incidents.decision_stats().approved == 1


def test_audit_and_records_never_keep_secrets(make_runtime):
    secret = "segredo-do-teste-123456"
    register_secrets(secret)
    rt = make_runtime("s01")
    rt.origin = {"scenario": "s01", "note": f"token={secret}"}
    result = run_incident(rt)
    audit_text = (make_runtime.logs / "audit.jsonl").read_text(encoding="utf-8")
    record_text = json.dumps(RecordStore(make_runtime.logs).get(result.incident_id).origin)
    assert secret not in audit_text
    assert secret not in record_text


def test_a_missing_database_is_a_clear_error(tmp_path):
    rt = Runtime(settings=Settings(_env_file=None), db_path=tmp_path / "nada.db")
    with pytest.raises(AgentError, match="sentinel generate"):
        run_incident(rt)
