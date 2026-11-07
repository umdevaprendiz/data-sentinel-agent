"""Log de auditoria (seção 12 do CLAUDE.md): uma linha JSON por passo em `logs/audit.jsonl`.

Cada linha leva o nó, a ferramenta, os parâmetros, a duração, o `Finding` produzido (com o
resultado, para o `replay`), o modelo e se houve fallback. A linha inteira passa por `redact`
antes de ir para o disco: nenhum segredo, DSN, token ou caminho pessoal entra no arquivo.
"""

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentinel.security import redact_obj

AUDIT_FILE = "audit.jsonl"


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass
class AuditLog:
    """Escreve no arquivo (ou só em memória, com `path=None`, nos testes e no benchmark)."""

    path: Path | None
    llm_model: str | None = None
    lines: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def in_dir(cls, logs_dir: Path, llm_model: str | None = None) -> "AuditLog":
        return cls(Path(logs_dir) / AUDIT_FILE, llm_model)

    def write(
        self,
        incident_id: str,
        node: str,
        *,
        tool: str | None = None,
        params: dict | None = None,
        duration_ms: float = 0.0,
        finding_id: str | None = None,
        fallback: bool = False,
        llm: bool = False,
        **extra: Any,
    ) -> dict:
        entry = {
            "ts": now_iso(),
            "incident_id": incident_id,
            "node": node,
            "tool": tool,
            "params": params or {},
            "duration_ms": round(duration_ms, 1),
            "finding_id": finding_id,
            "llm_model": self.llm_model if llm else None,
            "fallback": fallback,
            **extra,
        }
        clean = redact_obj(json.loads(json.dumps(entry, ensure_ascii=False, default=str)))
        line = json.dumps(clean, ensure_ascii=False)
        with self._lock:
            self.lines.append(clean)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        return clean

    @contextmanager
    def step(self, incident_id: str, node: str, **fields: Any) -> Iterator[dict]:
        """Mede a duração de um passo; o chamador completa `extra` antes de sair."""
        extra: dict = {}
        start = time.perf_counter()
        try:
            yield extra
        finally:
            self.write(
                incident_id,
                node,
                duration_ms=(time.perf_counter() - start) * 1000,
                **{**fields, **extra},
            )


def read_audit(path: Path, incident_id: str | None = None) -> list[dict]:
    """Linhas do audit (de um incidente, se pedido). Linha corrompida é ignorada."""
    if not path.exists():
        return []
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if incident_id is None or entry.get("incident_id") == incident_id:
                entries.append(entry)
    return entries
