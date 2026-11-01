import json

import httpx
import numpy as np
import pytest

from sentinel.llm.embeddings import EmbeddingError, FakeEmbedder, OllamaEmbedder, cosine


def test_fake_embedder_is_deterministic_and_normalized():
    embedder = FakeEmbedder(dim=32)
    first, again = embedder.embed(["queda no android"]), embedder.embed(["queda no android"])
    assert first == again
    assert len(first[0]) == 32
    assert np.linalg.norm(first[0]) == pytest.approx(1.0)


def test_fake_embedder_ranks_similar_text_above_unrelated_text():
    embedder = FakeEmbedder()
    query, near, far = embedder.embed(
        ["queda de conversão no android versão", "conversão caiu no android", "estoque do watch"]
    )
    assert cosine(query, near) > cosine(query, far)
    assert cosine(query, query) == pytest.approx(1.0)


def test_fake_embedder_handles_empty_text():
    [vector] = FakeEmbedder(dim=8).embed([""])
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def ollama(handler, dim=3) -> OllamaEmbedder:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaEmbedder("http://ollama.test", "nomic-embed-text", dim, client=client)


def test_ollama_embedder_posts_the_texts_and_returns_the_vectors():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [[1, 0, 0], [0, 1, 0]]})

    vectors = ollama(handler).embed(["a", "b"])
    assert seen["url"] == "http://ollama.test/api/embed"
    assert seen["body"] == {"model": "nomic-embed-text", "input": ["a", "b"]}
    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_ollama_embedder_with_no_texts_makes_no_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("não devia chamar o Ollama")

    assert ollama(handler).embed([]) == []


@pytest.mark.parametrize(
    ("handler", "fragment"),
    [
        (lambda r: httpx.Response(500, text="boom"), "500"),
        (lambda r: httpx.Response(200, text="não é json"), "JSON"),
        (lambda r: httpx.Response(200, json={}), "embeddings"),
        (lambda r: httpx.Response(200, json={"embeddings": [[1, 0, 0]]}), "vieram 1"),
        (lambda r: httpx.Response(200, json={"embeddings": [[1, 0], [0, 1]]}), "dimensão"),
    ],
)
def test_ollama_embedder_failures_become_embedding_errors(handler, fragment):
    with pytest.raises(EmbeddingError, match=fragment):
        ollama(handler).embed(["a", "b"])


def test_ollama_embedder_turns_network_errors_into_embedding_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("recusou", request=request)

    with pytest.raises(EmbeddingError, match="Ollama"):
        ollama(handler).embed(["a"])


def test_network_error_messages_do_not_carry_the_url_or_its_credentials():
    secret = "pw" + "9x9"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"falhou em http://admin:{secret}@ollama.test", request=request)

    with pytest.raises(EmbeddingError) as raised:
        ollama(handler).embed(["a"])
    assert secret not in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__suppress_context__
