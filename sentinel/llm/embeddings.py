"""Embeddings locais (Ollama) e um dublê determinístico para os testes."""

import hashlib
import math
import re
from typing import Protocol

import httpx

Vector = list[float]


class EmbeddingError(RuntimeError):
    """O modelo de embedding falhou ou devolveu algo inesperado."""


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str]) -> list[Vector]:
        """Um vetor de `dim` posições por texto, na mesma ordem."""
        ...


def cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class FakeEmbedder:
    """Saco de palavras com hash estável: textos com palavras em comum ficam próximos."""

    model = "fake-embed"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[Vector]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> Vector:
        vector = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            vector[0], norm = 1.0, 1.0
        return [x / norm for x in vector]


class OllamaEmbedder:
    """Cliente de `POST /api/embed` do Ollama local."""

    def __init__(
        self,
        url: str,
        model: str,
        dim: int,
        timeout_s: float = 60,
        client: httpx.Client | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.dim = dim
        self._client = client or httpx.Client(timeout=timeout_s)

    def embed(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        try:
            response = self._client.post(
                f"{self.url}/api/embed", json={"model": self.model, "input": texts}
            )
        except httpx.HTTPError as error:
            raise EmbeddingError(f"Ollama indisponível ({type(error).__name__})") from None
        if response.status_code != 200:
            raise EmbeddingError(f"Ollama respondeu {response.status_code}")
        try:
            vectors = response.json()["embeddings"]
        except (ValueError, KeyError, TypeError):
            raise EmbeddingError("resposta do Ollama sem JSON com 'embeddings'") from None
        if len(vectors) != len(texts):
            raise EmbeddingError(f"esperava {len(texts)} vetor(es), vieram {len(vectors)}")
        if any(len(v) != self.dim for v in vectors):
            raise EmbeddingError(f"dimensão diferente de {self.dim} (EMBED_DIM)")
        return [[float(x) for x in v] for v in vectors]
