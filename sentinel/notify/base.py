"""Contrato do notificador (seção 10 do CLAUDE.md) e um dublê para os testes."""

from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from sentinel.security import clean_text

MAX_MESSAGE = 3500  # o Telegram aceita 4096 caracteres; sobra espaço para o rodapé
MAX_EVIDENCE = 3


class Callback(BaseModel):
    """Resposta de um humano a um pedido de aprovação."""

    incident_id: str
    decision: Literal["approve", "ignore"]
    chat_id: str
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Notifier(Protocol):
    def request_approval(self, incident_id: str, text: str) -> None:
        """Manda a mensagem com os botões Aprovar e Ignorar."""
        ...

    def send(self, text: str) -> None:
        """Mensagem informativa (sem botões)."""
        ...

    def poll(self, timeout_s: int = 0) -> list[Callback]:
        """Respostas recebidas do chat configurado (as de outros chats são descartadas)."""
        ...


class FakeNotifier:
    """Guarda o que seria enviado e devolve as respostas enfileiradas pelo teste."""

    def __init__(self) -> None:
        self.approvals: list[tuple[str, str]] = []
        self.messages: list[str] = []
        self._queue: list[Callback] = []

    def request_approval(self, incident_id: str, text: str) -> None:
        self.approvals.append((incident_id, text))

    def send(self, text: str) -> None:
        self.messages.append(text)

    def answer(self, incident_id: str, decision: Literal["approve", "ignore"]) -> None:
        self._queue.append(Callback(incident_id=incident_id, decision=decision, chat_id="fake"))

    def poll(self, timeout_s: int = 0) -> list[Callback]:
        queued, self._queue = self._queue, []
        return queued


def approval_message(
    incident_id: str,
    facts: list[str],
    causes: list[str],
    opinions: list[str],
    evidence: list[str],
    action: str | None,
) -> str:
    """Resumo para o humano decidir: fatos, causa, opinião rotulada, três evidências, ação."""
    lines = [f"DataSentinel: incidente {incident_id}", ""]
    if causes:
        lines.append("Causa confirmada pelo código: " + "; ".join(causes))
    lines.append("Fatos (verificados):")
    lines += [f"- {clean_text(f, 400)}" for f in facts]
    if opinions:
        lines += ["", "Opinião do agente (NÃO verificada):"]
        lines += [f"- {clean_text(o, 400)}" for o in opinions]
    if evidence:
        lines += ["", "Evidências: " + ", ".join(evidence[:MAX_EVIDENCE])]
    if action:
        lines += ["", f"Ação proposta (simulador): {action}"]
    text = "\n".join(lines)
    return text if len(text) <= MAX_MESSAGE else text[: MAX_MESSAGE - 1] + "…"
