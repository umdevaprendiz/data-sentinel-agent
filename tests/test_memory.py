"""Contrato da memória de incidentes: o mesmo teste roda em RAM e no Postgres com pgvector."""

from datetime import UTC, datetime, timedelta

import pytest

from sentinel.llm.base import LLMError  # noqa: F401  (garante a importação do pacote llm)
from sentinel.llm.embeddings import EmbeddingError, FakeEmbedder
from sentinel.memory import (
    DecisionStats,
    Incident,
    InMemoryIncidentStore,
    MemoryStoreError,
    cause_key,
    signature_text,
)

NOW = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)


def incident(
    incident_id: str = "inc-1",
    *,
    metric: str = "conversion_rate",
    direction: str = "drop",
    causes: list[dict[str, str]] | None = None,
    summary: str = "A conversão caiu no Android 3.2 depois do deploy.",
    state: str = "open",
    created_at: datetime = NOW,
) -> Incident:
    causes = [{"platform": "android", "app_version": "3.2"}] if causes is None else causes
    return Incident(
        incident_id=incident_id,
        metric=metric,
        direction=direction,
        status="identified" if causes else "inconclusive",
        causes=causes,
        summary=summary,
        state=state,
        created_at=created_at,
    )


@pytest.fixture(params=["ram", pytest.param("pg", marks=pytest.mark.pg)])
def store(request):
    """O store de incidentes em cada backend, com o mesmo embedder determinístico."""
    embedder = FakeEmbedder(dim=64)
    if request.param == "ram":
        yield InMemoryIncidentStore(embedder)
    else:
        pg_memory = request.getfixturevalue("pg_memory")
        yield pg_memory.incidents(embedder)


# --- modelo -----------------------------------------------------------------------------------


def test_signature_and_cause_key_ignore_the_order_of_the_causes():
    a = [{"region": "sul"}, {"product": "phone"}]
    b = [{"product": "phone"}, {"region": "sul"}]
    assert cause_key(a) == cause_key(b)
    assert cause_key(a) != cause_key([{"region": "sul"}])
    text = signature_text(
        "conversion_rate", "drop", [{"platform": "android", "app_version": "3.2"}]
    )
    assert "queda" in text and "conversion_rate" in text and "platform=android" in text
    assert "sem causa" in signature_text("sessions", "spike", [])


def test_incident_accepts_any_well_formed_metric_and_dimension():
    """Métricas e dimensões são descobertas nos dados: a memória só confere o formato."""
    found = incident(metric="refund_rate", causes=[{"warehouse": "campinas"}])
    assert found.metric == "refund_rate" and found.causes == [{"warehouse": "campinas"}]


def test_incident_rejects_malformed_names_values_and_naive_dates():
    with pytest.raises(ValueError, match="dimensão"):
        incident(causes=[{"Cor Azul!": "azul"}])
    with pytest.raises(ValueError, match="valor"):
        incident(causes=[{"platform": ""}])
    with pytest.raises(ValueError, match="valor"):
        incident(causes=[{"platform": "x" * 200}])
    with pytest.raises(ValueError):
        incident(metric="DROP TABLE; --")
    with pytest.raises(ValueError, match="fuso"):
        incident(created_at=datetime(2026, 4, 1, 12, 0))
    with pytest.raises(ValueError, match="no máximo 2"):
        incident(causes=[{"region": "sul"}, {"product": "phone"}, {"channel": "email"}])


def test_decision_stats_precision():
    assert DecisionStats(approved=0, ignored=0).precision is None
    stats = DecisionStats(approved=9, ignored=1)
    assert stats.total == 10 and stats.precision == pytest.approx(0.9)


# --- contrato dos stores ----------------------------------------------------------------------


def test_add_and_get_roundtrip(store):
    original = incident(causes=[{"region": "sul"}, {"product": "phone"}])
    store.add(original)
    loaded = store.get("inc-1")
    assert loaded == original
    assert loaded.created_at.tzinfo is not None
    assert store.get("nao-existe") is None


def test_adding_the_same_incident_twice_is_an_error_and_keeps_the_first(store):
    store.add(incident(summary="primeiro"))
    with pytest.raises(MemoryStoreError, match="já existe"):
        store.add(incident(summary="segundo"))
    assert store.get("inc-1").summary == "primeiro"


def test_set_state(store):
    store.add(incident())
    store.set_state("inc-1", "approved")
    assert store.get("inc-1").state == "approved"
    with pytest.raises(MemoryStoreError, match="não existe"):
        store.set_state("nao-existe", "resolved")
    with pytest.raises(MemoryStoreError, match="estado"):
        store.set_state("inc-1", "inventado")


def test_similar_ranks_by_meaning_and_honours_k_threshold_and_exclusion(store):
    store.add(incident("android", summary="Conversão caiu no Android versão 3.2 após o deploy."))
    store.add(
        incident(
            "sul",
            metric="sessions",
            causes=[{"region": "sul"}],
            summary="Sessões caíram na região sul por falha de infraestrutura.",
        )
    )
    store.add(
        incident(
            "watch",
            causes=[{"product": "watch"}],
            summary="Produto watch sem estoque, conversão zerada.",
        )
    )
    query = signature_text(
        "conversion_rate", "drop", [{"platform": "android", "app_version": "3.2"}]
    )

    ranked = store.similar(query + " conversão caiu android versão", k=3, min_similarity=0.0)
    assert [hit.incident.incident_id for hit in ranked][0] == "android"
    similarities = [hit.similarity for hit in ranked]
    assert similarities == sorted(similarities, reverse=True)
    assert all(0.0 <= s <= 1.0 + 1e-6 for s in similarities)

    assert len(store.similar(query, k=1, min_similarity=0.0)) == 1
    assert store.similar(query, k=3, min_similarity=0.999) == []
    others = store.similar(query, k=3, min_similarity=0.0, exclude="android")
    assert "android" not in [hit.incident.incident_id for hit in others]


def test_similar_on_an_empty_store(store):
    assert store.similar("qualquer coisa", k=3, min_similarity=0.0) == []


def test_find_open_deduplicates_within_the_cooldown(store):
    store.add(incident("a", created_at=NOW))
    same = {"metric": "conversion_rate", "direction": "drop"}
    causes = [{"platform": "android", "app_version": "3.2"}]
    inside = NOW + timedelta(hours=3)
    assert store.find_open(**same, causes=causes, within_hours=24, now=inside).incident_id == "a"
    assert (
        store.find_open(**same, causes=causes, within_hours=24, now=NOW + timedelta(hours=25))
        is None
    )
    assert store.find_open(**same, causes=[{"region": "sul"}], within_hours=24, now=inside) is None
    assert store.find_open("sessions", "drop", causes, within_hours=24, now=inside) is None, (
        "outra métrica não é o mesmo incidente"
    )
    assert (
        store.find_open(**same, causes=causes, within_hours=24, now=NOW - timedelta(hours=1))
        is None
    )


def test_find_open_only_sees_open_or_pending_incidents(store):
    store.add(incident("a", created_at=NOW))
    args = {
        "metric": "conversion_rate",
        "direction": "drop",
        "causes": [{"platform": "android", "app_version": "3.2"}],
        "within_hours": 24,
        "now": NOW + timedelta(hours=1),
    }
    store.set_state("a", "pending")
    assert store.find_open(**args) is not None
    for state in ("approved", "ignored", "expired", "resolved"):
        store.set_state("a", state)
        assert store.find_open(**args) is None


def test_find_open_matches_two_causes_in_any_order(store):
    store.add(incident("duas", causes=[{"region": "sul"}, {"product": "phone"}], created_at=NOW))
    found = store.find_open(
        "conversion_rate",
        "drop",
        [{"product": "phone"}, {"region": "sul"}],
        within_hours=24,
        now=NOW + timedelta(hours=1),
    )
    assert found is not None and found.incident_id == "duas"


def test_decision_stats_count_only_human_decisions(store):
    for number, state in enumerate(
        ["approved", "approved", "ignored", "open", "expired", "resolved"]
    ):
        store.add(incident(f"i{number}", state=state))
    stats = store.decision_stats()
    assert (stats.approved, stats.ignored) == (2, 1)
    assert stats.precision == pytest.approx(2 / 3)


def test_the_summary_is_cleaned_and_redacted_before_it_is_stored(store):
    secret = "pw" + "4k4k"
    dirty = "ok" + chr(27) + "[2J" + chr(10) + f"postgresql://u:{secret}@h/db" + "x" * 5000
    store.add(incident(summary=dirty))
    saved = store.get("inc-1").summary
    assert chr(27) not in saved and chr(10) not in saved
    assert secret not in saved
    assert len(saved) <= 2000


def test_a_failing_embedder_leaves_the_store_unchanged():
    class Broken(FakeEmbedder):
        def embed(self, texts):
            raise EmbeddingError("fora do ar")

    ram = InMemoryIncidentStore(Broken(dim=64))
    with pytest.raises(EmbeddingError):
        ram.add(incident())
    assert ram.get("inc-1") is None


def test_the_embedder_dimension_must_match():
    class Wrong(FakeEmbedder):
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    bad = InMemoryIncidentStore(Wrong(dim=64))
    with pytest.raises(MemoryStoreError, match="dimensão"):
        bad.add(incident())
