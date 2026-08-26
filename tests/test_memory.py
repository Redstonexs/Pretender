"""Phase 5 durable memory service: source batch -> injected compressor ->
CAS commit (winner/stale/unavailable), and lexical-first recall with
no-embed zero-call and chat isolation."""

from __future__ import annotations

from pretender.errors import RepoError
from pretender.memory import MemoryService, SummarizeResult, default_capsule_summarizer
from pretender.search import MemorySearch
from pretender.types import (
    ChatKey,
    MemoryRecord,
    MemoryWriteRequest,
    MessageRowId,
)
from tests.durable_helpers import CK, make_identity, open_repo_with_chat, run
from tests.knowledge_helpers import OTHER, make_memory, seed_messages


def _summarizer(text: str = "summary"):
    """An injected async summarizer: one memory record from the batch."""

    async def summarize(batch):
        return (
            make_memory(
                chat_key=batch.chat_key,
                text=text,
                source_first_msg_id=batch.first_msg_id,
                source_last_msg_id=batch.last_msg_id,
                source_hash=batch.source_hash,
            ),
        )

    return summarize


def _service(repo, *, summarizer=None, **kw):
    return MemoryService(repo, summarizer=summarizer, **kw)


# ── summarize: winner / stale / unavailable / no_work ────────────────────────

def test_summarize_commits_winner(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=3)
        svc = _service(repo, summarizer=_summarizer("s1"))
        result = await svc.summarize(CK, through_msg_id=MessageRowId(3))
        wm = await repo.get_memory_watermark(CK)
        hits = await repo.query_memory(CK, "s1", limit=5)
        await repo.close()
        return result, wm, hits

    result, wm, hits = run(scenario())
    assert result.status == "ok"
    assert result.committed is True
    assert len(result.records) == 1
    assert wm == MessageRowId(3)
    assert len(hits) == 1 and hits[0].text == "s1"


def test_summarize_stale_loser_changes_nothing(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=3)

        # A racing summarizer that first commits the range itself (advancing
        # the watermark), then returns records — so the service's own CAS
        # commit is stale and must change nothing.
        async def racing(batch):
            winner = make_memory(
                chat_key=batch.chat_key, text="winner",
                source_first_msg_id=batch.first_msg_id,
                source_last_msg_id=batch.last_msg_id,
                source_hash=batch.source_hash,
            )
            await repo.commit_memory_source(
                MemoryWriteRequest(chat_key=batch.chat_key, batch=batch, records=(winner,))
            )
            return (
                make_memory(
                    chat_key=batch.chat_key, text="loser",
                    source_first_msg_id=batch.first_msg_id,
                    source_last_msg_id=batch.last_msg_id,
                    source_hash=batch.source_hash,
                ),
            )

        svc = _service(repo, summarizer=racing)
        result = await svc.summarize(CK, through_msg_id=MessageRowId(3))
        wm = await repo.get_memory_watermark(CK)
        rows = await repo._db.read(
            lambda c: c.execute("SELECT text FROM memories ORDER BY id").fetchall()
        )
        await repo.close()
        return result, wm, rows

    result, wm, rows = run(scenario())
    assert result.status == "stale"
    assert result.committed is False
    assert wm == MessageRowId(3)  # unchanged
    assert rows == [("winner",)]  # only the racing winner's memory


def test_summarize_unavailable_without_compressor(tmp_path):
    """No injected compressor -> an explicit unavailable result, never an
    error, and no network call."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        svc = _service(repo)  # no summarizer
        result = await svc.summarize(CK, through_msg_id=MessageRowId(2))
        wm = await repo.get_memory_watermark(CK)
        await repo.close()
        return result, wm

    result, wm = run(scenario())
    assert result.status == "unavailable"
    assert result.committed is False
    assert result.reason == "no summarizer configured"
    assert wm is None  # nothing committed


def test_summarize_no_work(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        svc = _service(repo, summarizer=_summarizer())
        # First summarize commits and advances the watermark.
        assert (await svc.summarize(CK, through_msg_id=MessageRowId(2))).status == "ok"
        # Nothing beyond the watermark now.
        result = await svc.summarize(CK, through_msg_id=MessageRowId(2))
        await repo.close()
        return result

    result = run(scenario())
    assert result.status == "no_work"
    assert result.committed is False


def test_summarize_rejects_cross_chat_record(tmp_path):
    """A summarizer that emits a cross-chat record fails closed before any
    transaction."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)

        async def bad(batch):
            return (make_memory(chat_key=OTHER, text="x"),)

        svc = _service(repo, summarizer=bad)
        try:
            await svc.summarize(CK, through_msg_id=MessageRowId(2))
            raised = False
        except ValueError:
            raised = True
        wm = await repo.get_memory_watermark(CK)
        await repo.close()
        return raised, wm

    raised, wm = run(scenario())
    assert raised is True
    assert wm is None  # nothing committed


# ── Gate 5: sequential batches (no gaps) + concurrent CAS + one-record rule ──

def test_summarize_second_batch_no_gaps(tmp_path):
    """Sequential summarizes process the OLDEST bounded chunk first; the
    second summarize picks up the remaining rows — no source gaps."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=5)
        svc = _service(repo, summarizer=_summarizer("s1"), tail=3)
        r1 = await svc.summarize(CK, through_msg_id=MessageRowId(5))
        r2 = await svc.summarize(CK, through_msg_id=MessageRowId(5))
        wm = await repo.get_memory_watermark(CK)
        rows = await repo._db.read(
            lambda c: c.execute(
                "SELECT text, source_first_msg_id, source_last_msg_id"
                " FROM memories ORDER BY id"
            ).fetchall()
        )
        await repo.close()
        return r1, r2, wm, rows

    r1, r2, wm, rows = run(scenario())
    assert r1.status == "ok" and r2.status == "ok"
    assert wm == MessageRowId(5)
    # First batch covered rows 1-3, second covered rows 4-5: no gaps.
    assert rows == [("s1", 1, 3), ("s1", 4, 5)]


def test_summarize_concurrent_cas_winner_and_stale_loser(tmp_path):
    """Two summarizers reading the same batch (same observed watermark): the
    first CAS wins, the second is stale and changes nothing — exactly one
    memory per source range."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=3)
        b1 = await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(3), tail=100
        )
        b2 = await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(3), tail=100
        )
        assert b1 is not None and b2 is not None
        assert b1.observed_watermark == b2.observed_watermark

        def req(batch, text):
            return MemoryWriteRequest(
                chat_key=CK, batch=batch,
                records=(
                    make_memory(
                        chat_key=CK, text=text,
                        source_first_msg_id=batch.first_msg_id,
                        source_last_msg_id=batch.last_msg_id,
                        source_hash=batch.source_hash,
                    ),
                ),
                expected_through_msg_id=batch.observed_watermark,
            )

        ok1 = await repo.commit_memory_source(req(b1, "winner"))
        ok2 = await repo.commit_memory_source(req(b2, "loser"))
        wm = await repo.get_memory_watermark(CK)
        rows = await repo._db.read(
            lambda c: c.execute("SELECT text FROM memories ORDER BY id").fetchall()
        )
        await repo.close()
        return ok1, ok2, wm, rows

    ok1, ok2, wm, rows = run(scenario())
    assert ok1 is True and ok2 is False
    assert wm == MessageRowId(3)
    assert rows == [("winner",)]  # exactly one memory, never duplicated


def test_commit_rejects_multi_record_batch_atomically(tmp_path):
    """A batch producing 2+ memory records is rejected atomically: nothing
    is inserted and the watermark never moves."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=3)
        batch = await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(3), tail=100
        )
        assert batch is not None
        rec1 = make_memory(
            chat_key=CK, text="a",
            source_first_msg_id=batch.first_msg_id,
            source_last_msg_id=batch.last_msg_id,
            source_hash=batch.source_hash,
        )
        rec2 = make_memory(
            chat_key=CK, text="b",
            source_first_msg_id=batch.first_msg_id,
            source_last_msg_id=batch.last_msg_id,
            source_hash=batch.source_hash,
        )
        try:
            await repo.commit_memory_source(
                MemoryWriteRequest(
                    chat_key=CK, batch=batch, records=(rec1, rec2),
                    expected_through_msg_id=batch.observed_watermark,
                )
            )
            raised = False
        except RepoError:
            raised = True
        wm = await repo.get_memory_watermark(CK)
        count = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        )
        await repo.close()
        return raised, wm, count

    raised, wm, count = run(scenario())
    assert raised is True
    assert wm is None  # watermark never moved
    assert count == 0  # nothing inserted


def test_summarize_rejects_multi_record_batch(tmp_path):
    """A summarizer producing 2+ records fails closed at the SERVICE layer
    (before any transaction) — one source batch -> exactly one record."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)

        async def bad(batch):
            return (
                make_memory(
                    chat_key=batch.chat_key, text="a",
                    source_first_msg_id=batch.first_msg_id,
                    source_last_msg_id=batch.last_msg_id,
                    source_hash=batch.source_hash,
                ),
                make_memory(
                    chat_key=batch.chat_key, text="b",
                    source_first_msg_id=batch.first_msg_id,
                    source_last_msg_id=batch.last_msg_id,
                    source_hash=batch.source_hash,
                ),
            )

        svc = _service(repo, summarizer=bad)
        try:
            await svc.summarize(CK, through_msg_id=MessageRowId(2))
            raised = False
        except ValueError:
            raised = True
        wm = await repo.get_memory_watermark(CK)
        await repo.close()
        return raised, wm

    raised, wm = run(scenario())
    assert raised is True
    assert wm is None  # nothing committed


def test_default_capsule_summarizer_is_deterministic_and_local(tmp_path):
    """The default deterministic local capsule writer produces exactly ONE
    memory record per batch, deterministically, with no provider call."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        svc = _service(repo, summarizer=default_capsule_summarizer())
        r1 = await svc.summarize(CK, through_msg_id=MessageRowId(2))
        r2 = await svc.summarize(CK, through_msg_id=MessageRowId(2))
        await repo.close()
        return r1, r2

    r1, r2 = run(scenario())
    assert r1.status == "ok" and r1.committed is True
    assert len(r1.records) == 1
    assert r1.records[0].text == "msg 1\nmsg 2"
    assert r1.records[0].source_first_msg_id == MessageRowId(1)
    assert r1.records[0].source_last_msg_id == MessageRowId(2)
    assert r2.status == "no_work"  # nothing beyond the watermark


# ── recall: lexical / no-embed zero-call / chat isolation ────────────────────

def test_recall_lexical_no_embed_zero_call(tmp_path):
    """With no embedding service, recall is FTS-only and performs ZERO
    provider calls."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        svc = _service(repo, summarizer=_summarizer("火锅好吃"))
        await svc.summarize(CK, through_msg_id=MessageRowId(2))
        hits = await svc.recall(CK, "火锅", limit=5)
        await repo.close()
        return hits

    hits = run(scenario())
    assert len(hits) == 1
    assert hits[0].text == "火锅好吃"
    assert hits[0].source == "lexical"


def test_recall_chat_isolation(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        await seed_messages(repo, n=1)
        await seed_messages(repo, chat_key=OTHER, n=1, prefix="other")
        svc = _service(repo, summarizer=_summarizer("火锅好吃"))
        await svc.summarize(CK, through_msg_id=MessageRowId(1))
        svc_other = _service(repo, summarizer=_summarizer("banana"))
        await svc_other.summarize(OTHER, through_msg_id=MessageRowId(1))
        ck_hits = await svc.recall(CK, "火锅", limit=5)
        other_hits = await svc.recall(OTHER, "火锅", limit=5)
        await repo.close()
        return ck_hits, other_hits

    ck_hits, other_hits = run(scenario())
    assert len(ck_hits) == 1 and ck_hits[0].chat_key == CK
    assert other_hits == []  # no cross-chat leak


def test_recall_empty_query_returns_empty(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=1)
        svc = _service(repo, summarizer=_summarizer("火锅好吃"))
        await svc.summarize(CK, through_msg_id=MessageRowId(1))
        hits = await svc.recall(CK, "!!!", limit=5)
        await repo.close()
        return hits

    assert run(scenario()) == []
