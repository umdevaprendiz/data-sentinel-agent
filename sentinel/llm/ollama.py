"""Cliente do Ollama local (`POST /api/chat`) com saída restrita a um JSON Schema.

Temperatura 0 e semente fixa: a mesma pergunta tende à mesma resposta. Nada sai da máquina: o
endereço padrão é `localhost`. Erros não repetem a mensagem do `httpx` (ela pode trazer a URL).
"""

import json

import httpx

from sentinel.llm.base import JsonObject, LLMError

THINKING_FAMILIES = ("qwen3", "deepseek-r1", "gpt-oss")  # modelos com "pensamento" desligável


class OllamaClient:
    def __init__(
        self,
        url: str,
        model: str,
        timeout_s: float = 300,
        seed: int = 0,
        num_ctx: int = 8192,
        client: httpx.Client | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.seed = seed
        self.num_ctx = num_ctx
        self._client = client or httpx.Client(timeout=timeout_s)
        self.last_usage: dict[str, float] = {}

    def generate_json(self, system: str, prompt: str, schema: JsonObject) -> JsonObject:
        body: dict = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "seed": self.seed, "num_ctx": self.num_ctx},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if self.model.startswith(THINKING_FAMILIES):
            body["think"] = False
        try:
            response = self._client.post(f"{self.url}/api/chat", json=body)
        except httpx.TimeoutException:
            raise LLMError("o Ollama estourou o tempo (LLM_TIMEOUT_S)") from None
        except httpx.HTTPError as error:
            raise LLMError(f"Ollama indisponível ({type(error).__name__})") from None
        if response.status_code != 200:
            raise LLMError(f"Ollama respondeu {response.status_code}")
        try:
            payload = response.json()
            content = payload["message"]["content"]
            answer = json.loads(content)
        except (ValueError, KeyError, TypeError):
            raise LLMError("o modelo não devolveu JSON") from None
        if not isinstance(answer, dict):
            raise LLMError("o modelo devolveu JSON que não é um objeto")
        self.last_usage = {
            "prompt_tokens": payload.get("prompt_eval_count", 0),
            "cached_tokens": payload.get("prompt_eval_cached_count", 0),
            "output_tokens": payload.get("eval_count", 0),
            "seconds": payload.get("total_duration", 0) / 1e9,
        }
        return answer

    def available(self) -> bool:
        """O modelo está instalado no Ollama local?"""
        try:
            response = self._client.get(f"{self.url}/api/tags")
            names = {m.get("name") for m in response.json().get("models", [])}
        except (httpx.HTTPError, ValueError, AttributeError):
            return False
        return self.model in names or f"{self.model}:latest" in names
