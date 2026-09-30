"""Embedding provider-selection and dim-probe invariants.

Two failure modes this locks down, both of which only show up in production:

1. An invalid EMBEDDING_PROVIDER must raise at first use. A silent default
   would mean ingesting a corpus with one model and querying with another.
2. A cloud probe failure with EMBEDDING_ALLOW_LOCAL_FALLBACK=false must RAISE.
   The deploy image deliberately omits torch/sentence-transformers, so "helpfully"
   falling back to local on a 512 MB instance means loading a ~700 MB model and
   getting OOM-killed. Failing loudly is the whole point of that env var.

Every test here mocks the provider. None of them touch a live API or download a
model — the suite must run offline and in well under a second.
"""

import pytest

import app.services.retrieval.embedding as emb
from app.config import settings


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Keep the process-wide lazy cache from leaking between tests."""
    saved = (emb._active_model, emb._model_type, emb._active_model_id,
             emb._active_dim, emb._last_probe_error)
    emb._active_model = emb._model_type = emb._active_model_id = None
    emb._active_dim = emb._last_probe_error = None
    yield
    (emb._active_model, emb._model_type, emb._active_model_id,
     emb._active_dim, emb._last_probe_error) = saved


class _FakeCloudModel:
    """Stands in for OpenAIEmbeddings — returns a fixed-width vector."""

    def __init__(self, dim: int = 3072):
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]


def test_invalid_provider_raises(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "gpt-4", raising=False)
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        emb._init()


def test_valid_providers_are_accepted(monkeypatch):
    """Both documented providers must pass the validator."""
    for provider in emb._VALID_PROVIDERS:
        monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", provider, raising=False)
        monkeypatch.setattr(
            emb, "_probe_cloud", lambda: (_FakeCloudModel(3072), "fake", 3072)
        )
        monkeypatch.setattr(emb, "_build_local", lambda: (_FakeCloudModel(768), "fake-local"))
        monkeypatch.setattr(emb, "_probe_local", lambda m: 768)
        emb._init()
        assert emb._active_model is not None, provider


def test_dim_is_probed_not_hard_coded(monkeypatch):
    """get_embedding_dim() must report what the provider actually returned.

    Changing AICREDITS_EMBEDDING_MODEL changes the dim, and the whole index is
    built off this number — a hard-coded 3072 would silently write mismatched
    vectors into Qdrant.
    """
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "aicredits", raising=False)
    for dim in (768, 1536, 3072):
        emb._active_model = None  # force a re-init at the new dim
        monkeypatch.setattr(
            emb, "_probe_cloud", lambda d=dim: (_FakeCloudModel(d), "fake", d)
        )
        assert emb.get_embedding_dim() == dim


def test_cloud_probe_failure_raises_when_fallback_disabled(monkeypatch):
    """The deploy-image path: fail loudly instead of loading ~700 MB.

    This is exactly what EMBEDDING_ALLOW_LOCAL_FALLBACK=false in render.yaml
    buys. If this test ever fails because the code stopped raising, the
    free-tier instance OOMs on the next AICredits outage.
    """
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "aicredits", raising=False)
    monkeypatch.setattr(settings, "EMBEDDING_ALLOW_LOCAL_FALLBACK", False, raising=False)

    def _boom():
        raise ConnectionError("simulated AICredits outage")

    monkeypatch.setattr(emb, "_probe_cloud", _boom)

    def _should_never_run():
        raise AssertionError(
            "local fallback was attempted with EMBEDDING_ALLOW_LOCAL_FALLBACK=false"
        )

    monkeypatch.setattr(emb, "_build_local", _should_never_run)

    with pytest.raises(RuntimeError, match="failed its probe"):
        emb._init()


def test_cloud_probe_failure_falls_back_to_local_when_allowed(monkeypatch):
    """Local dev keeps the friendly fallback — and records that it happened."""
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "aicredits", raising=False)
    monkeypatch.setattr(settings, "EMBEDDING_ALLOW_LOCAL_FALLBACK", True, raising=False)
    monkeypatch.setattr(
        emb, "_probe_cloud", lambda: (_ for _ in ()).throw(ConnectionError("down"))
    )
    monkeypatch.setattr(emb, "_build_local", lambda: (_FakeCloudModel(768), "all-mpnet-base-v2"))
    monkeypatch.setattr(emb, "_probe_local", lambda m: 768)

    emb._init()
    info = emb.active_embedding_info()
    assert info["provider"] == "local"
    assert info["requested_provider"] == "aicredits"
    assert info["fallback"] is True, "the fallback must be visible in the fingerprint"
    assert info["probe_error"], "probe_error must be recorded, not swallowed"


def test_local_provider_never_calls_the_cloud(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "local", raising=False)
    monkeypatch.setattr(emb, "_build_local", lambda: (_FakeCloudModel(768), "all-mpnet-base-v2"))
    monkeypatch.setattr(emb, "_probe_local", lambda m: 768)

    def _no_network():
        raise AssertionError("EMBEDDING_PROVIDER=local must not touch the network")

    monkeypatch.setattr(emb, "_probe_cloud", _no_network)
    assert emb.get_embedding_dim() == 768


def test_aicredits_without_a_key_raises_before_any_network_call(monkeypatch):
    """Missing key must fail fast, not as a confusing 401 from the gateway."""
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "aicredits", raising=False)
    monkeypatch.setattr(settings, "AICREDITS_API_KEY", "", raising=False)
    with pytest.raises(RuntimeError, match="AICREDITS_API_KEY"):
        emb._build_aicredits()


def test_embed_texts_splits_on_the_configured_batch_size(monkeypatch):
    """Batching must respect EMBEDDING_BATCH_SIZE so the gateway never 413s."""
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_SIZE", 2, raising=False)
    monkeypatch.setattr(emb, "_active_model", _FakeCloudModel(4))
    monkeypatch.setattr(emb, "_model_type", "aicredits")

    calls = []

    def _record(texts):
        calls.append(list(texts))
        return [[0.1] * 4 for _ in texts]

    monkeypatch.setattr(emb._active_model, "embed_documents", _record)
    out = emb.embed_texts([f"t{i}" for i in range(5)])
    assert len(out) == 5
    assert [len(c) for c in calls] == [2, 2, 1], calls


def test_embed_texts_short_circuits_on_empty_input():
    """An empty corpus must not make a provider call at all."""
    assert emb.embed_texts([]) == []
