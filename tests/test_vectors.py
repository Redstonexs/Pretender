"""Phase 5 retrieval-core: float32 vector index — ndarray <-> BLOB
round-trip, malformed-blob fail-closed, brute-force-equivalent cosine top-k
with stable ties, resident cache invalidation on write/delete, and restart
reload."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from pretender.types import ChatKey, VectorRow
from pretender.vectors import (
    VectorCache,
    VectorIndex,
    blob_to_ndarray,
    cosine_top_k,
    ndarray_to_blob,
    normalize,
)
from tests.durable_helpers import CK, open_repo_with_chat, run
from tests.knowledge_helpers import f32, make_vector


# ── ndarray <-> BLOB conversion ─────────────────────────────────────────────

def test_ndarray_blob_roundtrip():
    arr = np.array([0.1, -0.2, 0.3, 1.0], dtype=np.float32)
    blob = ndarray_to_blob(arr)
    assert len(blob) == 4 * 4  # little-endian float32
    back = blob_to_ndarray(blob, 4)
    assert back.dtype == np.float32
    np.testing.assert_allclose(back, arr)


def test_blob_is_explicit_little_endian():
    # 1.0 as float32 little-endian is 00 00 80 3f.
    assert ndarray_to_blob(np.array([1.0], dtype=np.float32)) == b"\x00\x00\x80\x3f"


def test_blob_to_ndarray_fails_closed_on_malformed():
    with pytest.raises(ValueError):
        blob_to_ndarray(f32(0.1, 0.2), 3)  # length != dim*4
    with pytest.raises(ValueError):
        blob_to_ndarray(f32(float("nan"), 0.2), 2)  # NaN
    with pytest.raises(ValueError):
        blob_to_ndarray(f32(float("inf"), 0.2), 2)  # inf


def test_normalize_fails_closed_on_zero_and_nonfinite():
    with pytest.raises(ValueError):
        normalize(np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        normalize(np.array([float("nan"), 1.0]))
    np.testing.assert_allclose(normalize(np.array([3.0, 4.0])), [0.6, 0.8])


# ── cosine top-k: brute-force goldens and stable ties ───────────────────────

def _brute_force(query, vectors, k):
    q = query / np.linalg.norm(query)
    scored = []
    for owner_id, vec in vectors.items():
        v = vec / np.linalg.norm(vec)
        scored.append((owner_id, float(np.dot(q, v))))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:k]


def test_cosine_top_k_matches_brute_force():
    rng = np.random.default_rng(42)
    query = rng.standard_normal(8).astype(np.float32)
    vectors = {i: rng.standard_normal(8).astype(np.float32) for i in range(50)}
    got = cosine_top_k(query, vectors, 10)
    expected = _brute_force(query, vectors, 10)
    assert [(h.owner_id, h.score) for h in got] == expected


def test_cosine_top_k_stable_ties_by_owner_id():
    # All vectors identical -> all scores equal; ties break by owner_id ASC.
    vectors = {5: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0]),
               9: np.array([1.0, 0.0])}
    got = cosine_top_k(np.array([1.0, 0.0]), vectors, 3)
    assert [h.owner_id for h in got] == [2, 5, 9]


def test_cosine_top_k_skips_corrupt_vectors():
    vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.0, 0.0]),  # zero norm -> skipped
        3: np.array([float("nan"), 1.0]),  # NaN -> skipped
        4: np.array([1.0, 0.0, 0.0]),  # wrong dim -> skipped
    }
    got = cosine_top_k(np.array([1.0, 0.0]), vectors, 10)
    assert [h.owner_id for h in got] == [1]


def test_cosine_top_k_rejects_bad_k():
    with pytest.raises(ValueError):
        cosine_top_k(np.array([1.0]), {1: np.array([1.0])}, 0)


# ── resident cache + VectorIndex over the repository ────────────────────────

async def _seed_memory(repo, chat_key: ChatKey, text: str) -> int:
    from pretender.types import MemoryRecord, MemoryWriteRequest, MessageRowId

    max_id = await repo._db.read(
        lambda c: c.execute(
            "SELECT COALESCE(MAX(id), 0) FROM messages WHERE chat_key = ?",
            (chat_key,),
        ).fetchone()[0]
    )
    batch = await repo.read_memory_source_batch(
        chat_key, through_msg_id=MessageRowId(max_id), tail=100
    )
    assert batch is not None
    rec = MemoryRecord(
        chat_key=chat_key, text=text,
        source_first_msg_id=batch.first_msg_id,
        source_last_msg_id=batch.last_msg_id,
        source_hash=batch.source_hash,
    )
    expected = await repo.get_memory_watermark(chat_key)
    assert await repo.commit_memory_source(
        MemoryWriteRequest(
            chat_key=chat_key, batch=batch, records=(rec,),
            expected_through_msg_id=expected,
        )
    ) is True
    return await repo._db.read(
        lambda c: c.execute(
            "SELECT id FROM memories WHERE chat_key = ? ORDER BY id DESC LIMIT 1",
            (chat_key,),
        ).fetchone()[0]
    )


def test_vector_index_search_and_cache_invalidation(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            None,
            __import__("tests.durable_helpers", fromlist=["make_message"]).make_message(
                chat_key=CK, msg_id="m1", text="src 1", recv_ts=1_700_000_000.0
            ),
        )
        mid = await _seed_memory(repo, CK, "alpha")
        g = await repo.create_embedding_generation("m1", 2)
        assert g.id is not None
        await repo.activate_embedding_generation(g.id)
        idx = VectorIndex(repo)
        await idx.upsert(CK, make_vector(owner_id=mid, generation=g.id, values=(1.0, 0.0)))
        hits = await idx.search(CK, np.array([1.0, 0.0]), "m1", g.id, 5)
        assert [h.owner_id for h in hits] == [mid]
        # Overwrite the vector: the cache must reflect the new value.
        await idx.upsert(CK, make_vector(owner_id=mid, generation=g.id, values=(0.0, 1.0)))
        hits2 = await idx.search(CK, np.array([1.0, 0.0]), "m1", g.id, 5)
        # Now orthogonal -> score ~0 (old cached value would have been 1.0).
        assert hits2[0].owner_id == mid
        assert abs(hits2[0].score) < 1e-6
        # Delete invalidates too.
        await idx.delete(CK, "memories", mid, "m1", g.id)
        assert await idx.search(CK, np.array([0.0, 1.0]), "m1", g.id, 5) == []
        await repo.close()

    run(scenario())


def test_vector_index_restart_reloads_from_repo(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            None,
            __import__("tests.durable_helpers", fromlist=["make_message"]).make_message(
                chat_key=CK, msg_id="m1", text="src 1", recv_ts=1_700_000_000.0
            ),
        )
        mid = await _seed_memory(repo, CK, "alpha")
        g = await repo.create_embedding_generation("m1", 2)
        assert g.id is not None
        await repo.upsert_vector(CK, make_vector(owner_id=mid, generation=g.id, values=(1.0, 0.0)))
        # A fresh index (no shared cache) reloads from the repository.
        idx = VectorIndex(repo)
        hits = await idx.search(CK, np.array([1.0, 0.0]), "m1", g.id, 5)
        assert [h.owner_id for h in hits] == [mid]
        await repo.close()

    run(scenario())


def test_vector_index_skips_corrupt_rows_and_does_not_cache(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            None,
            __import__("tests.durable_helpers", fromlist=["make_message"]).make_message(
                chat_key=CK, msg_id="m1", text="src 1", recv_ts=1_700_000_000.0
            ),
        )
        mid = await _seed_memory(repo, CK, "alpha")
        g = await repo.create_embedding_generation("m1", 2)
        assert g.id is not None
        # A zero-norm vector (valid as a row, but un-normalizable) is skipped.
        await repo.upsert_vector(
            CK, make_vector(owner_id=mid, generation=g.id, values=(0.0, 0.0))
        )
        idx = VectorIndex(repo)
        # Corrupt row is skipped: no hit, and the cache holds no poisoned entry.
        assert await idx.search(CK, np.array([1.0, 0.0]), "m1", g.id, 5) == []
        gen = await repo.get_embedding_generation(g.id)
        cached = idx._cache.get(CK, "m1", g.id, gen.vector_revision)
        assert cached is not None
        assert mid not in cached
        await repo.close()

    run(scenario())


def test_vector_cache_keyed_by_chat_model_generation():
    cache = VectorCache()
    cache.set(CK, "m1", 1, 0, {1: np.array([1.0])})
    # Different generation / model / chat / revision are distinct keys.
    assert cache.get(CK, "m1", 2, 0) is None
    assert cache.get(CK, "m2", 1, 0) is None
    assert cache.get(ChatKey("qq:group:other"), "m1", 1, 0) is None
    assert cache.get(CK, "m1", 1, 1) is None  # a bumped revision is a miss
    assert cache.get(CK, "m1", 1, 0) == {1: np.array([1.0])}
    cache.invalidate(CK, "m1", 1)
    assert cache.get(CK, "m1", 1, 0) is None


def test_direct_repo_vector_write_visible_on_next_search(tmp_path):
    """A DIRECT repo vector mutation (bypassing VectorIndex) is visible on
    the next search: the durable vector_revision cache key changes."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            None,
            __import__("tests.durable_helpers", fromlist=["make_message"]).make_message(
                chat_key=CK, msg_id="m1", text="src 1", recv_ts=1_700_000_000.0
            ),
        )
        mid = await _seed_memory(repo, CK, "alpha")
        g = await repo.create_embedding_generation("m1", 2)
        assert g.id is not None
        await repo.activate_embedding_generation(g.id)
        idx = VectorIndex(repo)
        # Direct repo write (not through the index).
        await repo.upsert_vector(
            CK, make_vector(owner_id=mid, generation=g.id, values=(1.0, 0.0))
        )
        hits = await idx.search(CK, np.array([1.0, 0.0]), "m1", g.id, 5)
        assert [h.owner_id for h in hits] == [mid]
        assert abs(hits[0].score - 1.0) < 1e-6
        # Direct repo overwrite bumps the durable revision: next search sees
        # the new vector, never the stale cached one.
        await repo.upsert_vector(
            CK, make_vector(owner_id=mid, generation=g.id, values=(0.0, 1.0))
        )
        hits2 = await idx.search(CK, np.array([1.0, 0.0]), "m1", g.id, 5)
        assert [h.owner_id for h in hits2] == [mid]
        assert abs(hits2[0].score) < 1e-6  # orthogonal now, not the stale 1.0
        await repo.close()

    run(scenario())


def test_vector_cache_poisoned_entry_never_poisons_result():
    """A poisoned cache entry (wrong dim) is skipped by cosine_top_k — it
    never poisons the result."""
    cache = VectorCache()
    cache.set(CK, "m1", 1, 0, {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0, 0.0]),  # wrong dim -> skipped
    })
    hits = cosine_top_k(np.array([1.0, 0.0]), cache.get(CK, "m1", 1, 0), 5)
    assert [h.owner_id for h in hits] == [1]
