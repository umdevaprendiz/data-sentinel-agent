"""Contrato do LLM e um dublê para os testes (seção 13 do CLAUDE.md)."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

JsonObject = dict[str, Any]


class LLMError(RuntimeError):
    """O modelo falhou, estourou o tempo ou devolveu algo que não é JSON."""


class LLMClient(Protocol):
    model: str

    def generate_json(self, system: str, prompt: str, schema: JsonObject) -> JsonObject:
        """Resposta do modelo, restrita ao JSON Schema. Levanta `LLMError` se falhar."""
        ...


@dataclass(frozen=True)
class LLMCall:
    system: str
    prompt: str
    schema: JsonObject


Reply = JsonObject | Exception | Callable[[LLMCall], JsonObject]


class FakeLLM:
    """Responde na ordem do roteiro e guarda as chamadas. Sem roteiro, levanta `LLMError`."""

    model = "fake"

    def __init__(self, script: Iterable[Reply] = ()) -> None:
        self._script = list(script)
        self.calls: list[LLMCall] = []

    def generate_json(self, system: str, prompt: str, schema: JsonObject) -> JsonObject:
        call = LLMCall(system, prompt, schema)
        self.calls.append(call)
        if len(self.calls) > len(self._script):
            raise LLMError("FakeLLM sem resposta no roteiro")
        reply = self._script[len(self.calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply(call) if callable(reply) else reply
