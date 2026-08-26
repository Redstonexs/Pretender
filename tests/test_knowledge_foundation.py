"""Phase 5 knowledge foundation: the durable source-bounded memory
contract, canonical CJK-bigram FTS documents, per-chat person identity with
a CAS profile cursor, embedding generations, and chat-scoped vector rows —
all over the v7 schema via the KnowledgeRepository surface."""

from __future__ import annotations

import pytest

from pretender.errors import RepoError
from pretender.types import (
    ChatKey,
    EmbeddingGeneration,
    LexicalHit,
    MemoryRecord,
    MemorySourceBatch,
    MemoryWriteRequest,
    MessageRowId,
    PersonProfile,
    SenderId,
    VectorRow,
)
from tests.durable_helpers import CK, make_identity, make_message, open_repo_with_chat, run
from tests.knowledge_helpers import (
    OTHER,
    f32,
    make_memory,
    make_person,
    make_vector,
    read_and_commit,
    seed_messages,
    source_hash,
)


# ── Boundary type validation (fail closed) ──────────────────────────────────

def test_memory_record_validates_strength_timestamps_and_source_range():
    with pytest.raises(ValueError):
        MemoryRecord(chat_key=CK, text="x", strength=float("nan"))
    with pytest.raises(ValueError):
        MemoryRecord(chat_key=CK, text="x", strength=-1.0)
    with pytest.raises(ValueError):
        MemoryRecord(chat_key=CK, text="x", created_ts=float("inf"))
    with pytest.raises(ValueError):
        MemoryRecord(chat_key=CK, text="x", last_hit_ts=float("-inf"))
    # Source range: both bounds or neither, first <= last.
    with pytest.raises(ValueError):
        MemoryRecord(chat_key=CK, text="x", source_first_msg_id=MessageRowId(1))
    with pytest.raises(ValueError):
        MemoryRecord(chat_key=CK, text="x", source_last_msg_id=MessageRowId(2))
    with pytest.raises(ValueError):
        MemoryRecord(
            chat_key=CK, text="x",
            source_first_msg_id=MessageRowId(5), source_last_msg_id=MessageRowId(2),
        )
    ok = MemoryRecord(
        chat_key=CK, text="x",
        source_first_msg_id=MessageRowId(1), source_last_msg_id=MessageRowId(2),
    )
    assert ok.source_hash is None


def test_memory_source_batch_validates_range_and_hash():
    with pytest.raises(ValueError):
        MemorySourceBatch(
            chat_key=CK, first_msg_id=MessageRowId(5),
            last_msg_id=MessageRowId(2), source_hash="h",
        )
    with pytest.raises(ValueError):
        MemorySourceBatch(
            chat_key=CK, first_msg_id=MessageRowId(1),
            last_msg_id=MessageRowId(2), source_hash="",
        )


def test_memory_write_request_rejects_cross_chat():
    batch = MemorySourceBatch(
        chat_key=CK, first_msg_id=MessageRowId(1),
        last_msg_id=MessageRowId(2), source_hash="h", texts=("a", "b"),
    )
    # A batch from another chat cannot be committed under this chat.
    with pytest.raises(ValueError):
        MemoryWriteRequest(chat_key=OTHER, batch=batch)
    # A record from another chat cannot ride along.
    rec = make_memory(chat_key=OTHER, text="x")
    with pytest.raises(ValueError):
        MemoryWriteRequest(chat_key=CK, batch=batch, records=(rec,))


def test_person_profile_validates_timestamp():
    with pytest.raises(ValueError):
        PersonProfile(chat_key=CK, platform_uid=SenderId("u1"), updated_ts=float("inf"))


def test_embedding_generation_validates_dim_state_and_timestamp():
    with pytest.raises(ValueError):
        EmbeddingGeneration(model="m", dim=0)
    with pytest.raises(ValueError):
        EmbeddingGeneration(model="m", dim=2, state="bogus")
    with pytest.raises(ValueError):
        EmbeddingGeneration(model="m", dim=2, created_ts=float("nan"))


def test_vector_row_validates_dim_blob_and_finite_values():
    with pytest.raises(ValueError):
        VectorRow(owner_table="memories", owner_id=1, dim=0, model="m",
                  generation=1, blob=f32(0.1))
    with pytest.raises(ValueError):
        VectorRow(owner_table="memories", owner_id=1, dim=2, model="m",
                  generation=1, blob=f32(0.1))  # length != dim * 4
    with pytest.raises(ValueError):
        VectorRow(owner_table="memories", owner_id=1, dim=2, model="m",
                  generation=1, blob=f32(float("nan"), 0.2))
    with pytest.raises(ValueError):
        VectorRow(owner_table="memories", owner_id=1, dim=2, model="m",
                  generation=1, blob=f32(float("inf"), 0.2))
    with pytest.raises(ValueError):
        VectorRow(owner_table="memories", owner_id=1, dim=2, model="m",
                  generation=-1, blob=f32(0.1, 0.2))
    ok = VectorRow(owner_table="memories", owner_id=1, dim=2, model="m",
                   generation=1, blob=f32(0.1, 0.2))
    assert ok.dim == 2


def test_lexical_hit_validates_score():
    with pytest.raises(ValueError):
        LexicalHit(chat_key=CK, memory_id=1, text="x", score=float("nan"))


# ── Memory watermark and source batches ─────────────────────────────────────

def test_memory_watermark_starts_none_and_reads_back(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        assert await repo.get_memory_watermark(CK) is None
        await seed_messages(repo, n=2)
        assert await read_and_commit(
            repo, through_msg_id=MessageRowId(2), text="s1"
        ) is True
        wm = await repo.get_memory_watermark(CK)
        await repo.close()
        return wm

    assert run(scenario()) == MessageRowId(2)


def test_memory_watermark_unknown_chat_returns_none(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        wm = await repo.get_memory_watermark(OTHER)
        await repo.close()
        return wm

    assert run(scenario()) is None


def test_observed_memory_watermark_recorded_at_commit(tmp_path):
    """The observed memory watermark (the snapshot the batch was read
    against) is recorded durably at commit."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        assert await repo.get_memory_observed_watermark(CK) is None
        await seed_messages(repo, n=3)
        batch = await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(3), tail=100
        )
        assert batch is not None
        assert batch.observed_watermark == MessageRowId(0)
        assert await read_and_commit(
            repo, through_msg_id=MessageRowId(3), text="s1"
        ) is True
        observed = await repo.get_memory_observed_watermark(CK)
        wm = await repo.get_memory_watermark(CK)
        await repo.close()
        return observed, wm

    observed, wm = run(scenario())
    assert observed == MessageRowId(0)  # the snapshot observed at read time
    assert wm == MessageRowId(3)  # the summarized watermark advanced


def test_source_batch_row_fences_and_retained_tail(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=5)
        # Watermark 0, terminal cursor 5, tail 3: the OLDEST bounded
        # unsummarized chunk [1, 3] — rows 4-5 wait for the next read, and
        # nothing past the cursor is included. No source rows are skipped.
        batch = await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(5), tail=3
        )
        assert batch is not None
        assert batch.first_msg_id == MessageRowId(1)
        assert batch.last_msg_id == MessageRowId(3)
        assert batch.texts == ("msg 1", "msg 2", "msg 3")
        assert batch.observed_watermark == MessageRowId(0)
        # Commit the range: the watermark advances to the batch's last row.
        rec = make_memory(
            chat_key=CK, text="s",
            source_first_msg_id=batch.first_msg_id,
            source_last_msg_id=batch.last_msg_id,
            source_hash=batch.source_hash,
        )
        assert await repo.commit_memory_source(
            MemoryWriteRequest(chat_key=CK, batch=batch, records=(rec,))
        ) is True
        assert await repo.get_memory_watermark(CK) == MessageRowId(3)
        # The NEXT read picks up the remaining rows [4, 5] — no gaps.
        batch2 = await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(5), tail=3
        )
        assert batch2 is not None
        assert batch2.first_msg_id == MessageRowId(4)
        assert batch2.last_msg_id == MessageRowId(5)
        assert batch2.observed_watermark == MessageRowId(3)
        # Rows past the terminal cursor are excluded even when the watermark
        # is behind: through 4 after the watermark advanced to 3 leaves only
        # row 4.
        batch3 = await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(4), tail=10
        )
        assert batch3 is not None
        assert batch3.first_msg_id == MessageRowId(4)
        assert batch3.last_msg_id == MessageRowId(4)
        await repo.close()

    run(scenario())


def test_source_batch_none_for_unknown_chat_and_no_messages(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        assert await repo.read_memory_source_batch(
            OTHER, through_msg_id=MessageRowId(5), tail=10
        ) is None
        assert await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(5), tail=10
        ) is None
        await repo.close()

    run(scenario())


def test_source_batch_hash_is_deterministic(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        b1 = await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(2), tail=10
        )
        b2 = await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(2), tail=10
        )
        await repo.close()
        return b1, b2

    b1, b2 = run(scenario())
    assert b1.source_hash == b2.source_hash
    assert b1.source_hash == source_hash(("msg 1", "msg 2"))


# ── Memory CAS commit ───────────────────────────────────────────────────────

def test_memory_cas_commits_exactly_once(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=3)
        batch = await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(3), tail=100
        )
        assert batch is not None
        rec = make_memory(
            chat_key=CK, text="s1",
            source_first_msg_id=batch.first_msg_id,
            source_last_msg_id=batch.last_msg_id,
            source_hash=batch.source_hash,
        )
        # First commit succeeds and advances the watermark.
        assert await repo.commit_memory_source(
            MemoryWriteRequest(chat_key=CK, batch=batch, records=(rec,))
        ) is True
        assert await repo.get_memory_watermark(CK) == MessageRowId(3)
        # The same range is no longer readable as new work.
        assert await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(3), tail=100
        ) is None
        # A retry with the ORIGINAL expected watermark is a stale CAS loser.
        assert await repo.commit_memory_source(
            MemoryWriteRequest(
                chat_key=CK, batch=batch, records=(rec,),
                expected_through_msg_id=MessageRowId(0),
            )
        ) is False
        count = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        )
        await repo.close()
        return count

    assert run(scenario()) == 1  # exactly one memory, never duplicated


def test_memory_cas_stale_loser_changes_nothing(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=3)
        batch = await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(3), tail=100
        )
        rec = make_memory(
            chat_key=CK, text="s1",
            source_first_msg_id=batch.first_msg_id,
            source_last_msg_id=batch.last_msg_id,
            source_hash=batch.source_hash,
        )
        assert await repo.commit_memory_source(
            MemoryWriteRequest(chat_key=CK, batch=batch, records=(rec,))
        ) is True
        # A second summarizer with a stale expected watermark loses.
        rec2 = make_memory(
            chat_key=CK, text="s2",
            source_first_msg_id=batch.first_msg_id,
            source_last_msg_id=batch.last_msg_id,
            source_hash=batch.source_hash,
        )
        assert await repo.commit_memory_source(
            MemoryWriteRequest(
                chat_key=CK, batch=batch, records=(rec2,),
                expected_through_msg_id=MessageRowId(0),
            )
        ) is False
        # Nothing changed: watermark intact, only the first memory exists.
        assert await repo.get_memory_watermark(CK) == MessageRowId(3)
        rows = await repo._db.read(
            lambda c: c.execute(
                "SELECT text FROM memories ORDER BY id"
            ).fetchall()
        )
        await repo.close()
        return rows

    assert run(scenario()) == [("s1",)]


def test_memory_cas_rejects_bad_hash(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=3)
        batch = await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(3), tail=100
        )
        rec = make_memory(
            chat_key=CK, text="s1",
            source_first_msg_id=batch.first_msg_id,
            source_last_msg_id=batch.last_msg_id,
            source_hash=batch.source_hash,
        )
        # A tampered hash must fail closed: the messages no longer match.
        bad_batch = MemorySourceBatch(
            chat_key=CK, first_msg_id=batch.first_msg_id,
            last_msg_id=batch.last_msg_id, source_hash="deadbeef",
            texts=batch.texts,
        )
        with pytest.raises(RepoError):
            await repo.commit_memory_source(
                MemoryWriteRequest(chat_key=CK, batch=bad_batch, records=(rec,))
            )
        assert await repo.get_memory_watermark(CK) is None
        await repo.close()

    run(scenario())


def test_memory_cas_rejects_overlapping_range(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=3)
        batch = await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(3), tail=100
        )
        rec = make_memory(
            chat_key=CK, text="s1",
            source_first_msg_id=batch.first_msg_id,
            source_last_msg_id=batch.last_msg_id,
            source_hash=batch.source_hash,
        )
        assert await repo.commit_memory_source(
            MemoryWriteRequest(chat_key=CK, batch=batch, records=(rec,))
        ) is True
        # A batch whose range overlaps the advanced watermark is rejected.
        stale_batch = MemorySourceBatch(
            chat_key=CK, first_msg_id=MessageRowId(1),
            last_msg_id=MessageRowId(3), source_hash=batch.source_hash,
            texts=batch.texts,
        )
        with pytest.raises(RepoError):
            await repo.commit_memory_source(
                MemoryWriteRequest(
                    chat_key=CK, batch=stale_batch, records=(rec,),
                    expected_through_msg_id=MessageRowId(3),
                )
            )
        await repo.close()

    run(scenario())


def test_memory_cas_unknown_chat_fails_closed(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        batch = MemorySourceBatch(
            chat_key=OTHER, first_msg_id=MessageRowId(1),
            last_msg_id=MessageRowId(1), source_hash="h", texts=("x",),
        )
        rec = make_memory(chat_key=OTHER, text="s")
        with pytest.raises(RepoError):
            await repo.commit_memory_source(
                MemoryWriteRequest(chat_key=OTHER, batch=batch, records=(rec,))
            )
        await repo.close()

    run(scenario())


# ── Canonical memory FTS (rebuild reproduces exactly) ───────────────────────

def test_memory_fts_rebuild_reproduces_exactly(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        assert await read_and_commit(
            repo, through_msg_id=MessageRowId(2), text="火锅好吃"
        ) is True
        before = await repo.query_memory(CK, "火锅", limit=5)
        assert len(before) == 1
        assert before[0].text == "火锅好吃"
        # A rebuild from the canonical token documents reproduces the same
        # index: same hit, same memory, same score.
        await repo.rebuild_memory_fts(CK)
        after = await repo.query_memory(CK, "火锅", limit=5)
        assert len(after) == 1
        assert after[0].memory_id == before[0].memory_id
        assert after[0].text == "火锅好吃"
        assert after[0].score == before[0].score
        await repo.close()

    run(scenario())


def test_memory_fts_rebuild_backfills_legacy_raw_memories(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        # A legacy memory row with raw text (no canonical doc yet).
        await repo._db.write(
            lambda c: c.execute(
                "INSERT INTO memories(chat_key, kind, text, strength)"
                " VALUES (?, 'memory', ?, 1.0)",
                (CK, "火锅好吃"),
            )
        )
        # No docs yet: the query finds nothing.
        assert await repo.query_memory(CK, "火锅", limit=5) == []
        # The explicit local rebuild/backfill tokenizes the raw text and
        # rebuilds the index transactionally.
        await repo.rebuild_memory_fts(CK)
        hits = await repo.query_memory(CK, "火锅", limit=5)
        assert len(hits) == 1
        assert hits[0].text == "火锅好吃"
        await repo.close()

    run(scenario())


def test_memory_fts_query_is_chat_safe(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        await seed_messages(repo, n=1)
        await seed_messages(repo, chat_key=OTHER, n=1, prefix="other")
        await _seed_memory(repo, text="火锅好吃")
        await _seed_memory(repo, chat_key=OTHER, text="banana")
        # A chat-scoped query never leaks another chat's memories.
        hits = await repo.query_memory(CK, "火锅", limit=10)
        assert len(hits) == 1
        assert hits[0].chat_key == CK
        assert await repo.query_memory(OTHER, "火锅", limit=10) == []
        assert await repo.query_memory(CK, "banana", limit=10) == []
        await repo.close()

    run(scenario())


def test_memory_fts_query_is_bounded(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=3)
        for i in range(1, 4):
            assert await read_and_commit(
                repo, through_msg_id=MessageRowId(i), text=f"火锅{i}"
            ) is True
        hits = await repo.query_memory(CK, "火锅", limit=2)
        await repo.close()
        return hits

    hits = run(scenario())
    assert len(hits) == 2


def test_memory_fts_bootstrap_state_is_idempotent(tmp_path):
    """The canonical memory FTS bootstrap/backlog state is idempotent:
    rebuild marks the chat bootstrapped and clears the backlog, re-running
    reproduces the same index, and marking a backlog re-marks the chat
    UNBOOTSTRAPPED so it is re-enumerated for bootstrap (the backlog state
    genuinely drives enumeration, never dead)."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        assert await read_and_commit(
            repo, through_msg_id=MessageRowId(2), text="火锅好吃"
        ) is True
        assert await repo.get_memory_fts_state(CK) is None
        await repo.rebuild_memory_fts(CK)
        state = await repo.get_memory_fts_state(CK)
        assert state is not None and state[0] is True
        # Idempotent: re-running reproduces the same index.
        before = await repo.query_memory(CK, "火锅", limit=5)
        await repo.rebuild_memory_fts(CK)
        after = await repo.query_memory(CK, "火锅", limit=5)
        assert [h.memory_id for h in after] == [h.memory_id for h in before]
        assert [h.score for h in after] == [h.score for h in before]
        # Marking a backlog re-marks the chat UNBOOTSTRAPPED (drives
        # re-enumeration); a rebuild clears the backlog.
        await repo.mark_memory_fts_backlog(CK, MessageRowId(2))
        await repo.mark_memory_fts_backlog(CK, MessageRowId(2))
        state2 = await repo.get_memory_fts_state(CK)
        assert state2 is not None
        assert state2[0] is False  # re-marked unbootstrapped
        assert state2[1] == MessageRowId(2)
        assert await repo.list_memory_fts_unbootstrapped_chats() == [CK]
        await repo.rebuild_memory_fts(CK)
        state3 = await repo.get_memory_fts_state(CK)
        assert state3 == (True, None)  # bootstrapped, backlog cleared
        await repo.close()

    run(scenario())


# ── Person identity and CAS profile cursor ──────────────────────────────────

def test_person_upsert_and_get_roundtrip(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_person(
            make_person(names=("alice", "小爱"), profile="likes tea")
        )
        p = await repo.get_person(CK, SenderId("u1"))
        await repo.close()
        return p

    p = run(scenario())
    assert p is not None
    assert p.names == ("alice", "小爱")
    assert p.profile == "likes tea"
    assert p.person_key is not None


def test_person_unique_by_chat_and_uid(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        await repo.upsert_person(make_person(names=("alice",)))
        await repo.upsert_person(make_person(uid="u2", names=("bob",)))
        await repo.upsert_person(
            make_person(chat_key=OTHER, uid="u1", names=("alice-other",))
        )
        # Same (chat, uid) upsert updates the identity in place.
        await repo.upsert_person(make_person(names=("alice2",)))
        a = await repo.get_person(CK, SenderId("u1"))
        b = await repo.get_person(CK, SenderId("u2"))
        c = await repo.get_person(OTHER, SenderId("u1"))
        await repo.close()
        return a, b, c

    a, b, c = run(scenario())
    assert a.names == ("alice2",)
    assert b.names == ("bob",)
    assert c.names == ("alice-other",)
    # The same platform uid in different chats is a DIFFERENT person — no
    # global nickname matching.
    assert a.person_key != c.person_key


def test_person_profile_cas_success_and_stale(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_person(make_person())
        # CAS with the expected cursor (None == 0) succeeds.
        assert await repo.cas_person_profile(
            CK, SenderId("u1"), None,
            make_person(profile="p1", profile_through_msg_id=MessageRowId(5)),
        ) is True
        p = await repo.get_person(CK, SenderId("u1"))
        assert p.profile == "p1"
        assert p.profile_through_msg_id == MessageRowId(5)
        # A stale CAS (expected 0, actual 5) fails and changes nothing.
        assert await repo.cas_person_profile(
            CK, SenderId("u1"), None,
            make_person(profile="p2", profile_through_msg_id=MessageRowId(9)),
        ) is False
        p2 = await repo.get_person(CK, SenderId("u1"))
        assert p2.profile == "p1"
        assert p2.profile_through_msg_id == MessageRowId(5)
        await repo.close()

    run(scenario())


def test_person_profile_cas_unknown_person_fails(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        ok = await repo.cas_person_profile(
            CK, SenderId("nobody"), None,
            make_person(uid="nobody", profile="x"),
        )
        await repo.close()
        return ok

    assert run(scenario()) is False


# ── Embedding generations ───────────────────────────────────────────────────

def test_embedding_generation_create_read_and_idempotent(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        g1 = await repo.create_embedding_generation("m1", 1024)
        g2 = await repo.create_embedding_generation("m1", 1024)  # idempotent
        g3 = await repo.create_embedding_generation("m2", 512)
        got = await repo.get_embedding_generation(g1.id)
        all_g = await repo.list_embedding_generations()
        await repo.close()
        return g1, g2, g3, got, all_g

    g1, g2, g3, got, all_g = run(scenario())
    assert g1.id == g2.id  # same (model, dim) returns the existing generation
    assert g1.state == "inactive"
    assert g3.id != g1.id
    assert got == g1
    assert len(all_g) == 2


def test_embedding_generation_activate_flips_and_deactivates_previous(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        g1 = await repo.create_embedding_generation("m1", 1024)
        g2 = await repo.create_embedding_generation("m2", 1024)
        assert await repo.activate_embedding_generation(g1.id) is True
        assert (await repo.get_embedding_generation(g1.id)).state == "active"
        assert (await repo.get_embedding_generation(g2.id)).state == "inactive"
        # Activating g2 atomically deactivates g1: never two active.
        assert await repo.activate_embedding_generation(g2.id) is True
        assert (await repo.get_embedding_generation(g1.id)).state == "inactive"
        assert (await repo.get_embedding_generation(g2.id)).state == "active"
        # Unknown generation activation fails closed.
        assert await repo.activate_embedding_generation(999) is False
        await repo.close()

    run(scenario())


# ── Chat-scoped vector rows ─────────────────────────────────────────────────

async def _seed_memory(repo, chat_key: ChatKey = CK, text: str = "x"):
    """Commit one memory for the chat's newest message; returns its row id."""
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
    rec = make_memory(
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
    return 1


def test_vector_upsert_get_roundtrip(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=1)
        await _seed_memory(repo)
        g = await repo.create_embedding_generation("m1", 2)
        row = make_vector(owner_id=1, generation=g.id, source_hash="h1")
        await repo.upsert_vector(CK, row)
        got = await repo.get_vector(CK, "memories", 1, "m1", g.id)
        await repo.close()
        return got

    got = run(scenario())
    assert got is not None
    assert got.dim == 2
    assert got.blob == f32(0.1, 0.2)
    assert got.source_hash == "h1"


def test_vector_generation_coexistence(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=1)
        await _seed_memory(repo)
        g1 = await repo.create_embedding_generation("m1", 2)
        g2 = await repo.create_embedding_generation("m2", 4)
        await repo.upsert_vector(
            CK, make_vector(owner_id=1, generation=g1.id, model="m1", dim=2,
                            values=(0.1, 0.2))
        )
        await repo.upsert_vector(
            CK, make_vector(owner_id=1, generation=g2.id, model="m2", dim=4,
                            values=(0.1, 0.2, 0.3, 0.4))
        )
        v1 = await repo.get_vector(CK, "memories", 1, "m1", g1.id)
        v2 = await repo.get_vector(CK, "memories", 1, "m2", g2.id)
        l1 = await repo.list_vectors(CK, "m1", g1.id)
        l2 = await repo.list_vectors(CK, "m2", g2.id)
        await repo.close()
        return v1, v2, l1, l2

    v1, v2, l1, l2 = run(scenario())
    assert v1 is not None and v2 is not None
    assert v1.dim == 2 and v2.dim == 4
    assert len(l1) == 1 and len(l2) == 1


def test_vector_cross_chat_isolation(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        await seed_messages(repo, n=1)
        await seed_messages(repo, chat_key=OTHER, n=1, prefix="other")
        await _seed_memory(repo)  # memory id 1 in CK
        await _seed_memory(repo, chat_key=OTHER)  # memory id 2 in OTHER
        g = await repo.create_embedding_generation("m1", 2)
        await repo.upsert_vector(CK, make_vector(owner_id=1, generation=g.id))
        await repo.upsert_vector(OTHER, make_vector(owner_id=2, generation=g.id))
        # Each chat lists only its own vectors.
        ck_rows = await repo.list_vectors(CK, "m1", g.id)
        other_rows = await repo.list_vectors(OTHER, "m1", g.id)
        assert [r.owner_id for r in ck_rows] == [1]
        assert [r.owner_id for r in other_rows] == [2]
        # Wrong-chat ownership fails closed on read and delete.
        with pytest.raises(RepoError):
            await repo.get_vector(OTHER, "memories", 1, "m1", g.id)
        with pytest.raises(RepoError):
            await repo.delete_vector(OTHER, "memories", 1, "m1", g.id)
        # The CK vector is untouched.
        assert await repo.get_vector(CK, "memories", 1, "m1", g.id) is not None
        await repo.close()

    run(scenario())


def test_vector_unknown_generation_rejected(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=1)
        await _seed_memory(repo)
        with pytest.raises(RepoError):
            await repo.upsert_vector(CK, make_vector(owner_id=1, generation=999))
        await repo.close()

    run(scenario())


def test_vector_model_dim_must_match_generation(tmp_path):
    """A vector write must match the generation's model AND dim — no
    generation/model/dimension mixing."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=1)
        await _seed_memory(repo)
        g = await repo.create_embedding_generation("m1", 2)
        # Wrong model for the generation.
        with pytest.raises(RepoError):
            await repo.upsert_vector(
                CK, make_vector(owner_id=1, generation=g.id, model="other", dim=2)
            )
        # Wrong dim for the generation.
        with pytest.raises(RepoError):
            await repo.upsert_vector(
                CK, make_vector(owner_id=1, generation=g.id, model="m1", dim=4,
                                values=(0.1, 0.2, 0.3, 0.4))
            )
        # Matching model/dim succeeds.
        await repo.upsert_vector(
            CK, make_vector(owner_id=1, generation=g.id, model="m1", dim=2)
        )
        assert await repo.get_vector(CK, "memories", 1, "m1", g.id) is not None
        await repo.close()

    run(scenario())


def test_generation_space_id_and_revision(tmp_path):
    """Generations carry a canonical space_id (model + explicit revision);
    creation is idempotent per space, and distinct revisions are distinct
    generations. A same-space generation with an INCOMPATIBLE dim is
    rejected, never silently returned as a mismatched row."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        g1 = await repo.create_embedding_generation("m1", 1024, revision="r1")
        g2 = await repo.create_embedding_generation("m1", 1024, revision="r1")
        g3 = await repo.create_embedding_generation("m1", 1024, revision="r2")
        # Same space with a DIFFERENT dim is rejected (fail closed).
        try:
            await repo.create_embedding_generation("m1", 512, revision="r1")
            raised = False
        except RepoError:
            raised = True
        await repo.close()
        return g1, g2, g3, raised

    g1, g2, g3, raised = run(scenario())
    assert g1.id == g2.id  # same space -> same generation
    assert g1.space_id == "m1@r1"
    assert g1.revision == "r1"
    assert g3.id != g1.id  # distinct revision -> distinct generation
    assert g3.space_id == "m1@r2"
    assert raised is True  # incompatible dim rejected, not silently returned


def test_vector_unknown_owner_rejected(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        g = await repo.create_embedding_generation("m1", 2)
        with pytest.raises(RepoError):
            await repo.upsert_vector(CK, make_vector(owner_id=999, generation=g.id))
        await repo.close()

    run(scenario())


def test_vector_delete(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=1)
        await _seed_memory(repo)
        g = await repo.create_embedding_generation("m1", 2)
        await repo.upsert_vector(CK, make_vector(owner_id=1, generation=g.id))
        assert await repo.delete_vector(CK, "memories", 1, "m1", g.id) is True
        assert await repo.get_vector(CK, "memories", 1, "m1", g.id) is None
        assert await repo.delete_vector(CK, "memories", 1, "m1", g.id) is False
        await repo.close()

    run(scenario())


# ── Chat-safe bulk memory lookup ────────────────────────────────────────────

def test_get_memories_returns_chat_scoped_records_in_id_order(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        await repo.ingest_message(
            None, make_message(chat_key=CK, msg_id="a1", text="s1",
                               recv_ts=1_700_000_000.0 + 1)
        )
        await _seed_memory(repo, text="alpha")  # memory id 1 in CK
        await repo.ingest_message(
            None, make_message(chat_key=CK, msg_id="a2", text="s2",
                               recv_ts=1_700_000_000.0 + 2)
        )
        await _seed_memory(repo, text="beta")  # memory id 2 in CK
        await repo.ingest_message(
            None, make_message(chat_key=OTHER, msg_id="b1", text="s3",
                               recv_ts=1_700_000_000.0 + 3)
        )
        await _seed_memory(repo, chat_key=OTHER, text="gamma")  # id 3 in OTHER
        got = await repo.get_memories(CK, [3, 1, 2, 999])
        await repo.close()
        return got

    got = run(scenario())
    # Deterministic id order; cross-chat and unknown ids are absent.
    assert [m.id for m in got] == [1, 2]
    assert [m.text for m in got] == ["alpha", "beta"]
    assert all(m.chat_key == CK for m in got)
    assert got[0].strength == 1.0


def test_get_memories_empty_and_unknown(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=1)
        await _seed_memory(repo)
        empty = await repo.get_memories(CK, [])
        unknown = await repo.get_memories(CK, [999])
        await repo.close()
        return empty, unknown

    empty, unknown = run(scenario())
    assert empty == []
    assert unknown == []


# ── Stats compatibility ─────────────────────────────────────────────────────

def test_stats_counts_knowledge_tables(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=1)
        await _seed_memory(repo)
        await repo.upsert_person(make_person())
        await repo.create_embedding_generation("m1", 2)
        stats = await repo.stats()
        await repo.close()
        return stats

    stats = run(scenario())
    assert stats["memories"] == 1
    assert stats["persons"] == 1
    assert stats["vectors"] == 0
    assert stats["embedding_generations"] == 1
    assert stats["memory_search_docs"] == 1
    assert stats["user_version"] == 15
