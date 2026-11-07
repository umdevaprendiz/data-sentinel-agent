"""Cache das respostas do LLM por (modelo, semente, prompt, schema) — seção 13 do CLAUDE.md.

No benchmark, a mesma pergunta não vai duas vezes ao modelo (CPU lenta). Só respostas válidas
entram no cache; falhas não. Os arquivos ficam em `data/llm_cache/` (fora do Git).
"""

import hashlib
import json
from pathlib import Path

from sentinel.llm.base import JsonObject, LLMClient


class CachedLLM:
    def __init__(self, inner: LLMClient, folder: Path, seed: int = 0) -> None:
        self.inner = inner
        self.folder = Path(folder)
        self.seed = seed
        self.hits = 0
        self.misses = 0

    @property
    def model(self) -> str:
        return self.inner.model

    def _key(self, system: str, prompt: str, schema: JsonObject) -> str:
        text = json.dumps(
            [self.model, self.seed, system, prompt, schema], sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(text.encode()).hexdigest()

    def generate_json(self, system: str, prompt: str, schema: JsonObject) -> JsonObject:
        key = self._key(system, prompt, schema)
        path = self.folder / key[:2] / f"{key}.json"
        if path.exists():
            try:
                answer = json.loads(path.read_text(encoding="utf-8"))
                self.hits += 1
                return answer
            except ValueError:
                path.unlink(missing_ok=True)
        answer = self.inner.generate_json(system, prompt, schema)
        self.misses += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(answer, ensure_ascii=False), encoding="utf-8")
        return answer
