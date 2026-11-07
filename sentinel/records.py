"""Registro dos incidentes em disco (`logs/incidents/`) e log de decisões (`logs/decisions.jsonl`).

O registro é o que o painel web e o `sentinel incidents` mostram: anomalia, fatos, relatório,
decisão da policy, aprovação e resultado no simulador. Tudo passa por `redact` antes de gravar.
O log de decisões alimenta a precisão histórica da policy e o limite de ações por hora.
"""

import json
import os
import re
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from sentinel.memory.incidents import State
from sentinel.security import redact_obj

INCIDENT_ID = re.compile(r"^inc-\d{8}-\d{6}-[0-9a-f]{4}$")
INCIDENTS_DIR = "incidents"
DECISIONS_FILE = "decisions.jsonl"
ACTIVE = ("open", "pending", "approved")  # ainda não resolvidos

Decision = Literal["approve", "ignore", "expired", "auto_execute", "dry_run", "blocked"]


def new_incident_id(now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    return f"inc-{now:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def check_incident_id(incident_id: str) -> str:
    """Só ids no formato do agente: o id vira nome de arquivo e vem de fora (web, Telegram)."""
    if not INCIDENT_ID.match(incident_id):
        raise ValueError("id de incidente inválido")
    return incident_id


def utcnow() -> datetime:
    return datetime.now(UTC)


class IncidentRecord(BaseModel):
    incident_id: str = Field(pattern=INCIDENT_ID.pattern)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    state: State = "open"
    mode: Literal["deterministic", "llm"] = "deterministic"
    llm_model: str | None = None
    autonomy_level: int = 2
    dry_run: bool = True
    origin: dict[str, Any] = Field(default_factory=dict)  # cenário e semente (vêm do CLI)
    profile: dict[str, Any] = Field(default_factory=dict)
    anomaly: dict[str, Any] | None = None
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    unconfirmed: list[dict[str, Any]] = Field(default_factory=list)
    refutation: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    report_source: str | None = None
    report_problems: list[dict[str, Any]] = Field(default_factory=list)
    report_error: str | None = None
    rendered: str | None = None
    proposal: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    approval: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    duplicate_of: str | None = None
    resolved_by: str | None = None
    investigation: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def signature(self) -> tuple[str, str, str] | None:
        if not self.anomaly:
            return None
        causes = sorted(
            ",".join(f"{k}={v}" for k, v in sorted(h["dimensions"].items()))
            for h in self.hypotheses
        )
        return (self.anomaly["metric"], self.anomaly["direction"], "|".join(causes))


class RecordStore:
    """Um arquivo JSON por incidente. Escrita atômica (arquivo temporário + troca)."""

    def __init__(self, logs_dir: Path) -> None:
        self.dir = Path(logs_dir) / INCIDENTS_DIR

    def _path(self, incident_id: str) -> Path:
        return self.dir / f"{check_incident_id(incident_id)}.json"

    def save(self, record: IncidentRecord) -> IncidentRecord:
        record = record.model_copy(update={"updated_at": utcnow()})
        self.dir.mkdir(parents=True, exist_ok=True)
        clean = IncidentRecord.model_validate(redact_obj(json.loads(record.model_dump_json())))
        path = self._path(clean.incident_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(clean.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return clean

    def get(self, incident_id: str) -> IncidentRecord | None:
        path = self._path(incident_id)
        if not path.exists():
            return None
        return IncidentRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def update(self, incident_id: str, **fields: Any) -> IncidentRecord:
        record = self.get(incident_id)
        if record is None:
            raise KeyError("incidente não encontrado")
        return self.save(record.model_copy(update=fields))

    def list(self) -> list[IncidentRecord]:
        if not self.dir.exists():
            return []
        records = []
        for path in self.dir.glob("inc-*.json"):
            try:
                records.append(IncidentRecord.model_validate_json(path.read_text("utf-8")))
            except ValueError:
                continue  # arquivo corrompido não derruba a listagem
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def find_active(
        self, signature: tuple[str, str, str], within_hours: float, now: datetime | None = None
    ) -> IncidentRecord | None:
        """Incidente aberto ou pendente com a mesma assinatura dentro do cooldown."""
        now = now or utcnow()
        since = now - timedelta(hours=within_hours)
        for record in self.list():
            if (
                record.state in ("open", "pending")
                and record.signature == signature
                and since <= record.created_at <= now
                and record.duplicate_of is None
            ):
                return record
        return None


class DecisionEntry(BaseModel):
    incident_id: str
    decision: Decision
    via: str  # telegram | web | cli | timeout | policy
    at: datetime = Field(default_factory=utcnow)
    action_id: str | None = None
    outcome: str | None = None  # correct | innocuous | harmful (simulador)


class DecisionLog:
    def __init__(self, logs_dir: Path) -> None:
        self.path = Path(logs_dir) / DECISIONS_FILE

    def append(self, entry: DecisionEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            clean = DecisionEntry.model_validate(redact_obj(json.loads(entry.model_dump_json())))
            handle.write(clean.model_dump_json() + "\n")

    def entries(self) -> list[DecisionEntry]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(DecisionEntry.model_validate_json(line))
            except ValueError:
                continue
        return out


class History(BaseModel):
    """O que a policy sabe do passado: decisões humanas e ações automáticas recentes."""

    approved: int = 0
    ignored: int = 0
    auto_last_hour: int = 0

    @property
    def total(self) -> int:
        return self.approved + self.ignored

    @property
    def precision(self) -> float | None:
        return self.approved / self.total if self.total else None


def history_from(entries: list[DecisionEntry], now: datetime | None = None) -> History:
    now = now or utcnow()
    hour_ago = now - timedelta(hours=1)
    return History(
        approved=sum(e.decision == "approve" for e in entries),
        ignored=sum(e.decision == "ignore" for e in entries),
        auto_last_hour=sum(
            e.decision in ("auto_execute", "dry_run") and e.via == "policy" and e.at >= hour_ago
            for e in entries
        ),
    )


def dump(value: Any) -> Any:
    """Modelo Pydantic (ou lista deles) como JSON simples."""
    if isinstance(value, BaseModel):
        return json.loads(value.model_dump_json())
    if isinstance(value, list):
        return [dump(v) for v in value]
    return value
