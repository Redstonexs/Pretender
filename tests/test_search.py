"""Phase 5 retrieval-core: deterministic hybrid memory retrieval — strict
CJK/ASCII query normalization, fixed-RRF fusion with deterministic ties,
FTS-only parity when semantic is disabled, active-generation selection, and
typed recall hits with source/strength facts."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from pretender.embed import OptionalEmbeddingService
from pretender.search import MemoryRecallHit, MemorySearch, normalize_query, rrf_merge
from pretender.types import (
    ChatKey,
    LexicalHit,
    MemoryRecord,
    MemoryWriteRequest,
    MessageRowId,
)
from pretender.vectors import VectorHit, VectorIndex
from tests.durable_helpers import CK, make_identity, make_message, open_repo_with_chat, run
from tests.knowledge_helpers import OTHER, make_vector


def run(coro):
    return asyncio.run(coro)


class FixedEmbedder:
    """Returns a fixed vector for every text; records call count."""

    def __init__(self, vector):
        self.vector = vector
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return [self.vector for _ in texts]


def embed_service(vector, *, space_id="m1@default"):
    """An OptionalEmbeddingService bound to a canonical space (the default
    revision space of ``create_embedding_generation("m1", ...)``)."""
    return OptionalEmbeddingService(FixedEmbedder(vector), space_id=space_id)


async def seed_memories(repo, chat_key, texts, strengths=None, prefix="m"):
    """Seed one message per text and CAS-commit one memory each; returns the
    memory ids in order."""
    ids = []
    for i, text in enumerate(texts, start=1):
        await repo.ingest_message(
            None,
            make_message(
                chat_key=chat_key, msg_id=f"{prefix}{i}", text=f"src {i}",
                recv_ts=1_700_000_000.0 + i,
            ),
        )
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
        strength = strengths[i - 1] if strengths else 1.0
        rec = MemoryRecord(
            chat_key=chat_key, text=text, strength=strength,
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
        mid = await repo._db.read(
            lambda c: c.execute(
                "SELECT id FROM memories WHERE chat_key = ? ORDER BY id DESC LIMIT 1",
                (chat_key,),
            ).fetchone()[0]
        )
        ids.append(mid)
    return ids


# ── Query normalization ─────────────────────────────────────────────────────

def test_normalize_query_strips_punctuation_and_caps_terms():
    # CJK one-char and two-char survive; punctuation is stripped.
    assert normalize_query("你") == "你"
    assert normalize_query("你好") == "你好"
    assert normalize_query("你好，世界！") == "你好 世界"
    # ASCII punctuation stripped, runs collapsed.
    assert normalize_query("Hello,  world!!!") == "Hello world"
    # Term cap: only the first max_terms survive.
    assert normalize_query("a b c d e", max_terms=3) == "a b c"
    # Empty / punctuation-only -> "".
    assert normalize_query("!!!") == ""
    assert normalize_query("") == ""


def test_normalize_query_rejects_bad_max_terms():
    with pytest.raises(ValueError):
        normalize_query("x", max_terms=0)


# ── RRF fusion ──────────────────────────────────────────────────────────────

def test_rrf_merge_deterministic_and_stable_ties():
    lex = [LexicalHit(chat_key=CK, memory_id=1, text="a", score=1.0),
           LexicalHit(chat_key=CK, memory_id=2, text="b", score=0.5)]
    sem = [VectorHit(owner_id=2, score=0.9),
           VectorHit(owner_id=3, score=0.8)]
    out = rrf_merge(lex, sem, limit=10)
    # memory 1: lexical only -> source lexical; 2: both -> hybrid; 3: semantic.
    by_id = {mid: (score, src) for mid, score, src in out}
    assert by_id[1][1] == "lexical"
    assert by_id[2][1] == "hybrid"
    assert by_id[3][1] == "semantic"
    # Deterministic: identical input -> identical output.
    assert rrf_merge(lex, sem, limit=10) == out


def test_rrf_merge_single_list_preserves_order():
    lex = [LexicalHit(chat_key=CK, memory_id=3, text="a", score=2.0),
           LexicalHit(chat_key=CK, memory_id=1, text="b", score=1.0),
           LexicalHit(chat_key=CK, memory_id=2, text="c", score=0.5)]
    out = rrf_merge(lex, [], limit=10)
    # RRF over one list reproduces its order exactly.
    assert [mid for mid, _, _ in out] == [3, 1, 2]
    assert all(src == "lexical" for _, _, src in out)


def test_rrf_merge_ties_break_by_memory_id():
    # Equal RRF scores -> deterministic memory_id ASC.
    lex = [LexicalHit(chat_key=CK, memory_id=5, text="a", score=1.0)]
    sem = [VectorHit(owner_id=2, score=0.9)]
    # memory 5: 1/61; memory 2: 1/61 -> tie -> 2 before 5.
    out = rrf_merge(lex, sem, limit=10)
    assert [mid for mid, _, _ in out] == [2, 5]


# ── Search integration ──────────────────────────────────────────────────────

def test_search_no_embed_lexical_parity(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_memories(repo, CK, ["apple pie", "banana split", "apple cider"])
        # No embedder at all.
        s1 = MemorySearch(repo, embed=None)
        hits1 = await s1.search(CK, "apple", limit=10)
        # Embedder present but no active generation -> semantic unavailable.
        embed = embed_service([1.0, 0.0, 0.0])
        s2 = MemorySearch(repo, embed=embed)
        hits2 = await s2.search(CK, "apple", limit=10)
        await repo.close()
        return hits1, hits2

    hits1, hits2 = run(scenario())
    assert [h.memory_id for h in hits1] == [h.memory_id for h in hits2]
    assert all(h.source == "lexical" for h in hits1)
    assert len(hits1) == 2  # the two "apple" memories


def test_search_hybrid_fusion_and_strength_facts(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        ids = await seed_memories(
            repo, CK,
            ["apple pie recipe", "unrelated stuff", "apple orchard"],
            strengths=[1.0, 0.5, 1.0],
        )
        g = await repo.create_embedding_generation("m1", 3)
        assert g.id is not None
        await repo.activate_embedding_generation(g.id)
        # Query vector [1,0,0]; memory 1 closest, memory 2 semantic-only.
        await repo.upsert_vector(
            CK, make_vector(owner_id=ids[0], generation=g.id, dim=3, values=(1.0, 0.0, 0.0))
        )
        await repo.upsert_vector(
            CK, make_vector(owner_id=ids[1], generation=g.id, dim=3, values=(0.9, 0.1, 0.0))
        )
        await repo.upsert_vector(
            CK, make_vector(owner_id=ids[2], generation=g.id, dim=3, values=(0.5, 0.5, 0.0))
        )
        embed = embed_service([1.0, 0.0, 0.0])
        search = MemorySearch(repo, embed=embed)
        hits = await search.search(CK, "apple", limit=10)
        hits2 = await search.search(CK, "apple", limit=10)  # deterministic
        await repo.close()
        return hits, hits2, ids

    hits, hits2, ids = run(scenario())
    assert [h.memory_id for h in hits] == [h.memory_id for h in hits2]  # deterministic
    by_id = {h.memory_id: h for h in hits}
    # memory 1 (apple pie) is in both lists -> hybrid, top result.
    assert hits[0].memory_id == ids[0]
    assert by_id[ids[0]].source == "hybrid"
    # memory 2 (unrelated) is semantic-only -> source semantic, strength 0.5.
    assert by_id[ids[1]].source == "semantic"
    assert by_id[ids[1]].strength == 0.5
    assert by_id[ids[1]].semantic_score is not None
    # memory 3 (apple orchard) is in both -> hybrid.
    assert by_id[ids[2]].source == "hybrid"


def test_search_active_generation_only(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        ids = await seed_memories(repo, CK, ["apple pie", "unrelated stuff"])
        # g1 inactive with vectors; g2 active but empty.
        g1 = await repo.create_embedding_generation("m1", 3)
        g2 = await repo.create_embedding_generation("m2", 3)
        assert g1.id is not None and g2.id is not None
        await repo.activate_embedding_generation(g2.id)
        # Vectors only under the INACTIVE generation g1.
        await repo.upsert_vector(
            CK, make_vector(owner_id=ids[1], generation=g1.id, dim=3, values=(1.0, 0.0, 0.0))
        )
        # The embed service matches the ACTIVE generation's space (m2@default).
        embed = embed_service([1.0, 0.0, 0.0], space_id="m2@default")
        search = MemorySearch(repo, embed=embed)
        hits = await search.search(CK, "apple", limit=10)
        await repo.close()
        return hits

    hits = run(scenario())
    # Inactive generation's vectors never mix in: memory 2 stays absent.
    assert [h.memory_id for h in hits] == [1]
    assert hits[0].source == "lexical"


def test_search_mismatched_dim_falls_back_to_fts(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        ids = await seed_memories(repo, CK, ["apple pie", "unrelated stuff"])
        g = await repo.create_embedding_generation("m1", 5)  # active dim 5
        assert g.id is not None
        await repo.activate_embedding_generation(g.id)
        await repo.upsert_vector(
            CK, make_vector(owner_id=ids[1], generation=g.id, dim=5, values=(1.0, 0.0, 0.0, 0.0, 0.0))
        )
        # Embedder returns a dim-3 vector -> dim mismatch -> semantic skipped.
        embed = embed_service([1.0, 0.0, 0.0])
        search = MemorySearch(repo, embed=embed)
        hits = await search.search(CK, "apple", limit=10)
        await repo.close()
        return hits

    hits = run(scenario())
    assert [h.memory_id for h in hits] == [1]
    assert hits[0].source == "lexical"


def test_search_cross_chat_isolation(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        await seed_memories(repo, CK, ["apple pie"])
        await seed_memories(repo, OTHER, ["apple secret"], prefix="o")
        search = MemorySearch(repo, embed=None)
        hits = await search.search(CK, "apple", limit=10)
        await repo.close()
        return hits

    hits = run(scenario())
    assert [h.memory_id for h in hits] == [1]
    assert all(h.chat_key == CK for h in hits)


def test_search_corrupt_vector_ignored_semantic_partial(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        ids = await seed_memories(repo, CK, ["apple pie", "unrelated stuff"])
        g = await repo.create_embedding_generation("m1", 3)
        assert g.id is not None
        await repo.activate_embedding_generation(g.id)
        # memory 1 has a zero-norm (un-normalizable) vector; memory 2 good.
        await repo.upsert_vector(
            CK, make_vector(owner_id=ids[0], generation=g.id, dim=3, values=(0.0, 0.0, 0.0))
        )
        await repo.upsert_vector(
            CK, make_vector(owner_id=ids[1], generation=g.id, dim=3, values=(1.0, 0.0, 0.0))
        )
        embed = embed_service([1.0, 0.0, 0.0])
        search = MemorySearch(repo, embed=embed)
        hits = await search.search(CK, "apple", limit=10)
        await repo.close()
        return hits

    hits = run(scenario())
    # Corrupt vector is skipped (fail closed): memory 1 still found lexically,
    # memory 2 found semantically; no crash, no bad result.
    by_id = {h.memory_id: h for h in hits}
    assert by_id[1].source == "lexical"
    assert by_id[2].source == "semantic"


def test_search_cjk_query(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_memories(repo, CK, ["火锅很好吃", "今天天气不错", "火锅店在哪"])
        search = MemorySearch(repo, embed=None)
        hits = await search.search(CK, "火锅", limit=10)
        await repo.close()
        return hits

    hits = run(scenario())
    # Two-character CJK query matches the two "火锅" memories.
    assert {h.memory_id for h in hits} == {1, 3}


# ── Gate 5: deterministic ties + active-space matching before embed ──────────

def test_search_lexical_tie_deterministic_and_restart_stable(tmp_path):
    """Equal BM25 scores tie-break by memory_id ASC, and the order is stable
    across an FTS rebuild (a restart reproduces the same index)."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        # Two memories with identical text -> identical BM25 score.
        await seed_memories(repo, CK, ["apple pie", "apple pie"])
        search = MemorySearch(repo, embed=None)
        hits1 = await search.search(CK, "apple", limit=10)
        await repo.rebuild_memory_fts(CK)
        hits2 = await search.search(CK, "apple", limit=10)
        await repo.close()
        return hits1, hits2

    hits1, hits2 = run(scenario())
    assert [h.memory_id for h in hits1] == [h.memory_id for h in hits2]
    assert [h.memory_id for h in hits1] == sorted(h.memory_id for h in hits1)
    assert len(hits1) == 2


def test_search_lexical_scores_independent_of_other_chats(tmp_path):
    """Lexical scores AND order for one chat are independent of other chats:
    no global-BM25 influence after the chat filter (chat-local ranking)."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        await seed_memories(repo, CK, ["apple pie", "apple cider", "banana"])
        search = MemorySearch(repo, embed=None)
        before = await search.search(CK, "apple", limit=10)
        before_scores = [(h.memory_id, h.score) for h in before]
        # Add a second chat with many different memories: global BM25 stats
        # (avg doc length, df) would shift, but chat-local ranking must not.
        await seed_memories(
            repo, OTHER,
            ["orange juice", "grape soda", "lemon water", "mint tea"],
            prefix="o",
        )
        after = await search.search(CK, "apple", limit=10)
        after_scores = [(h.memory_id, h.score) for h in after]
        await repo.close()
        return before_scores, after_scores

    before, after = run(scenario())
    assert before == after  # identical scores AND order


def test_search_active_space_mismatch_zero_embed_calls(tmp_path):
    """When the active generation's space does not match the configured
    embed space, semantic is skipped BEFORE any query embedding call —
    ZERO provider calls, FTS-only."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_memories(repo, CK, ["apple pie"])
        g = await repo.create_embedding_generation("m1", 3)
        assert g.id is not None
        await repo.activate_embedding_generation(g.id)
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder, space_id="other@space")
        search = MemorySearch(repo, embed=embed)
        hits = await search.search(CK, "apple", limit=10)
        await repo.close()
        return hits, embedder.calls

    hits, calls = run(scenario())
    assert calls == 0  # ZERO query embedding calls
    assert [h.memory_id for h in hits] == [1]
    assert hits[0].source == "lexical"


def test_search_missing_embed_revision_zero_calls(tmp_path):
    """An embed service with no space_id (missing revision) performs ZERO
    query embedding calls — FTS-only."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_memories(repo, CK, ["apple pie"])
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder)  # no space_id
        search = MemorySearch(repo, embed=embed)
        hits = await search.search(CK, "apple", limit=10)
        await repo.close()
        return hits, embedder.calls

    hits, calls = run(scenario())
    assert calls == 0
    assert [h.memory_id for h in hits] == [1]
    assert hits[0].source == "lexical"


def test_search_empty_query_returns_nothing(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_memories(repo, CK, ["apple pie"])
        search = MemorySearch(repo, embed=None)
        hits = await search.search(CK, "!!!", limit=10)
        await repo.close()
        return hits

    assert run(scenario()) == []


def test_memory_recall_hit_validates():
    with pytest.raises(ValueError):
        MemoryRecallHit(chat_key=CK, memory_id=1, text="x", score=float("nan"),
                        source="lexical", strength=1.0)
    with pytest.raises(ValueError):
        MemoryRecallHit(chat_key=CK, memory_id=1, text="x", score=1.0,
                        source="bogus", strength=1.0)
    with pytest.raises(ValueError):
        MemoryRecallHit(chat_key=CK, memory_id=1, text="x", score=1.0,
                        source="lexical", strength=-1.0)
