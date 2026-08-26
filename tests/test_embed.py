"""Phase 5 retrieval-core: OptionalEmbeddingService — disabled zero-call
behavior, content-addressed SHA1 cache, batching, and fail-closed
validation/degradation at the service boundary."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from pretender.embed import EmbeddingCache, EmbeddingResult, OptionalEmbeddingService


def run(coro):
    return asyncio.run(coro)


class FakeEmbedder:
    """A scriptable Embedder that records every call."""

    def __init__(self, vectors=None, fail=None):
        self.calls: list[list[str]] = []
        self.vectors = vectors  # callable(text) -> list[float], or None
        self.fail = fail  # exception to raise, or None

    async def embed(self, texts):
        self.calls.append(list(texts))
        if self.fail is not None:
            raise self.fail
        if self.vectors is None:
            return [[0.1, 0.2, 0.3] for _ in texts]
        return [self.vectors(t) for t in texts]


def vec(*values):
    return list(values)


# ── Disabled / unconfigured ─────────────────────────────────────────────────

def test_disabled_embed_returns_unavailable_and_never_calls():
    calls = []

    class NeverCalled:
        async def embed(self, texts):
            calls.append(texts)
            return []

    svc = OptionalEmbeddingService(NeverCalled())
    # Force the disabled path by constructing with no embedder.
    svc2 = OptionalEmbeddingService()
    res = run(svc2.embed(["hello"]))
    assert res.status == "unavailable"
    assert res.vectors == ()
    assert res.reason is not None
    assert calls == []  # zero provider calls


def test_disabled_service_enabled_flag():
    assert OptionalEmbeddingService().enabled is False
    assert OptionalEmbeddingService(FakeEmbedder()).enabled is True


# ── Batching ────────────────────────────────────────────────────────────────

def test_embed_batches_by_batch_size():
    emb = FakeEmbedder()
    svc = OptionalEmbeddingService(emb, batch_size=2)
    res = run(svc.embed(["a", "b", "c", "d", "e"]))
    assert res.status == "ok"
    assert len(res.vectors) == 5
    # 5 texts, batch_size 2 -> 3 provider calls: [a,b], [c,d], [e].
    assert emb.calls == [["a", "b"], ["c", "d"], ["e"]]


def test_embed_empty_input_ok_without_calls():
    emb = FakeEmbedder()
    svc = OptionalEmbeddingService(emb)
    res = run(svc.embed([]))
    assert res.status == "ok"
    assert res.vectors == ()
    assert emb.calls == []


# ── Content-addressed cache ─────────────────────────────────────────────────

def test_cache_serves_repeat_text_without_provider_call():
    emb = FakeEmbedder()
    svc = OptionalEmbeddingService(emb)
    res1 = run(svc.embed(["same text"]))
    res2 = run(svc.embed(["same text"]))
    assert res1.status == "ok" and res2.status == "ok"
    assert len(emb.calls) == 1  # second call served from cache


def test_cache_dedupes_within_one_batch():
    emb = FakeEmbedder()
    svc = OptionalEmbeddingService(emb)
    res = run(svc.embed(["x", "x", "y"]))
    assert res.status == "ok"
    assert len(res.vectors) == 3
    # Only the unique texts are sent to the provider.
    assert emb.calls == [["x", "y"]]


def test_cache_persists_to_disk(tmp_path):
    emb = FakeEmbedder()
    svc = OptionalEmbeddingService(emb, cache_path=tmp_path / "cache")
    assert run(svc.embed(["persisted"])).status == "ok"
    assert len(emb.calls) == 1
    # A fresh service over the same disk cache serves without the provider.
    emb2 = FakeEmbedder()
    svc2 = OptionalEmbeddingService(emb2, cache_path=tmp_path / "cache")
    assert run(svc2.embed(["persisted"])).status == "ok"
    assert emb2.calls == []


def test_cache_clear_forces_reembed():
    emb = FakeEmbedder()
    cache = EmbeddingCache()
    svc = OptionalEmbeddingService(emb, cache=cache)
    run(svc.embed(["t"]))
    cache.clear()
    run(svc.embed(["t"]))
    assert len(emb.calls) == 2


# ── Gate 5: (space_id, text_hash) cache identity + corrupt-cache reject ──────

def test_cache_same_text_distinct_spaces_are_misses():
    """The cache identity is (space_id, text_hash): the same text under a
    different revision space is a distinct miss."""
    emb = FakeEmbedder()
    svc1 = OptionalEmbeddingService(emb, space_id="m@r1")
    svc2 = OptionalEmbeddingService(emb, space_id="m@r2")
    assert run(svc1.embed(["same text"])).status == "ok"
    assert run(svc2.embed(["same text"])).status == "ok"
    assert len(emb.calls) == 2  # distinct spaces -> distinct misses
    # The same space serves from cache.
    assert run(svc1.embed(["same text"])).status == "ok"
    assert len(emb.calls) == 2


def test_disk_cache_hashes_unsafe_model_space_id(tmp_path):
    """Provider model identifiers must never become path components."""
    cache = EmbeddingCache(tmp_path / "cache")
    svc = OptionalEmbeddingService(FakeEmbedder(), cache=cache, space_id="BAAI/bge-m3@r1")
    assert run(svc.embed(["persist"])).status == "ok"
    assert list((tmp_path / "cache").iterdir())
    assert not (tmp_path / "cache" / "BAAI").exists()
    restarted = OptionalEmbeddingService(FakeEmbedder(), cache=cache, space_id="BAAI/bge-m3@r1")
    assert run(restarted.embed(["persist"])).status == "ok"


def test_cache_rejects_corrupt_cached_vector():
    """A corrupt cached vector (NaN) is treated as a miss on every read and
    never served."""
    emb = FakeEmbedder()
    cache = EmbeddingCache()
    svc = OptionalEmbeddingService(emb, cache=cache, space_id="s", dim=3)
    assert run(svc.embed(["t"])).status == "ok"
    # Corrupt the in-memory cache entry with a NaN vector.
    cache._mem[("s", cache._key("s", "t"))] = np.array([float("nan"), 0.0, 0.0])
    assert run(svc.embed(["t"])).status == "ok"  # re-embedded, not served
    assert len(emb.calls) == 2


def test_cache_rejects_wrong_dim_cached_vector():
    """A cached vector with the wrong dimension is a miss on every read."""
    emb = FakeEmbedder()
    cache = EmbeddingCache()
    svc = OptionalEmbeddingService(emb, cache=cache, space_id="s", dim=3)
    assert run(svc.embed(["t"])).status == "ok"
    cache._mem[("s", cache._key("s", "t"))] = np.array([0.1, 0.2])  # dim 2
    assert run(svc.embed(["t"])).status == "ok"
    assert len(emb.calls) == 2


# ── Fail-closed validation / degradation ────────────────────────────────────

def test_embed_degrades_on_provider_exception():
    emb = FakeEmbedder(fail=RuntimeError("boom"))
    svc = OptionalEmbeddingService(emb)
    res = run(svc.embed(["a"]))
    assert res.status == "degraded"
    assert res.reason is not None
    # The service is now degraded: subsequent calls do not hit the provider.
    assert run(svc.embed(["b"])).status == "degraded"
    assert len(emb.calls) == 1


def test_embed_degrades_on_wrong_batch_size():
    class BadSize:
        async def embed(self, texts):
            return [[0.1]]  # wrong count

    svc = OptionalEmbeddingService(BadSize())
    assert run(svc.embed(["a", "b"])).status == "degraded"


def test_embed_degrades_on_non_finite_vector():
    emb = FakeEmbedder(vectors=lambda t: [float("nan"), 0.0, 0.0])
    svc = OptionalEmbeddingService(emb)
    assert run(svc.embed(["a"])).status == "degraded"


def test_embed_degrades_on_zero_vector():
    emb = FakeEmbedder(vectors=lambda t: [0.0, 0.0, 0.0])
    svc = OptionalEmbeddingService(emb)
    assert run(svc.embed(["a"])).status == "degraded"


def test_embed_degrades_on_dim_mismatch():
    emb = FakeEmbedder(vectors=lambda t: [0.1, 0.2])  # dim 2
    svc = OptionalEmbeddingService(emb, dim=3)
    assert run(svc.embed(["a"])).status == "degraded"


def test_embed_degrades_on_dim_change_across_batches():
    """One embedding dimension per service/space: a provider that changes
    dimension between batches degrades (never silently mixes)."""
    class DimChanging:
        def __init__(self):
            self.calls = 0

        async def embed(self, texts):
            self.calls += 1
            if self.calls == 1:
                return [[0.1, 0.2, 0.3] for _ in texts]  # dim 3
            return [[0.1, 0.2] for _ in texts]  # dim 2

    svc = OptionalEmbeddingService(DimChanging(), batch_size=1)
    assert run(svc.embed(["a"])).status == "ok"  # locks dim 3
    assert run(svc.embed(["b"])).status == "degraded"  # dim changed


def test_embed_cache_read_enforces_locked_dim():
    """Once the service locks a dimension, a cached vector of a DIFFERENT
    dimension is a miss on every read (never served)."""
    emb = FakeEmbedder()
    cache = EmbeddingCache()
    svc = OptionalEmbeddingService(emb, cache=cache, space_id="s")
    assert run(svc.embed(["t"])).status == "ok"  # locks dim 3
    # A dim-2 cached vector for another text must be a miss.
    cache._mem[("s", cache._key("s", "other"))] = np.array([0.1, 0.2])
    assert svc.cached(["other"]) == []  # zero-budget hit check: miss
    res = run(svc.embed(["other"]))
    assert res.status == "ok"  # re-embedded, not served the dim-2 vector
    assert res.vectors[0].shape == (3,)


def test_embed_ok_vectors_are_float32_ndarray():
    emb = FakeEmbedder()
    svc = OptionalEmbeddingService(emb)
    res = run(svc.embed(["a"]))
    assert res.status == "ok"
    assert res.vectors[0].dtype == np.float32
    assert res.vectors[0].shape == (3,)


def test_embedding_result_validates_status():
    with pytest.raises(ValueError):
        EmbeddingResult(status="bogus")
