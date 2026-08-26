"""Repository: the async protocol over SQLite — dedupe, identity
derivation, CJK FTS, the fenced claim lifecycle (boundary, expiry, atomic
finish), and at-most-once outbox semantics with transition fences."""

from __future__ import annotations

import asyncio
import dataclasses
import struct

import pytest

from pretender.adapters.console import ConsoleAdapter
from pretender.errors import ClaimError, RepoError
from pretender.outbox import OutboxDriver
from pretender.pacing import ewma_interval
from pretender.repo import SqliteRepository, bigram_tokenize
from pretender.seams import Repository
from pretender.types import (
    ChatKey,
    ChatState,
    ClaimBusy,
    ClaimGrant,
    CycleId,
    MemoryRecord,
    MemoryWriteRequest,
    MessageRowId,
    OutboxItem,
    Outgoing,
    Record,
    Segment,
)
from tests.durable_helpers import (
    CK,
    finish_batch,
    make_claim,
    make_finish,
    make_identity,
    make_message,
    open_repo,
    open_repo_with_chat,
    run,
)
from tests.knowledge_helpers import make_vector

OTHER = ChatKey("qq:group:other")


def item(text="hi", idem_key="k1", chat_key=CK, **kw) -> OutboxItem:
    return OutboxItem(chat_key=chat_key, text=text, idem_key=idem_key, **kw)


# ── chats: identity and runtime state ───────────────────────────────────────

def test_chat_identity_roundtrip(tmp_path):
    async def scenario():
        _db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(title="Test Group"))
        chat = await repo.get_chat(CK)
        await repo.close()
        return chat

    chat = run(scenario())
    assert chat is not None
    assert chat.platform == "qq"
    assert chat.self_id == "bot-1"
    assert chat.kind == "group"
    assert chat.title == "Test Group"


def test_upsert_chat_updates_identity(tmp_path):
    async def scenario():
        _db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(title="Old"))
        await repo.upsert_chat(make_identity(title="New"))
        chat = await repo.get_chat(CK)
        await repo.close()
        return chat

    assert run(scenario()).title == "New"


def test_chat_state_roundtrip(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat_state(
            ChatState(CK, focus_until=123.0, avg_interval=42.0,
                      cfg_json='{"a": 1}')
        )
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    state = run(scenario())
    assert state is not None
    assert state.focus_until == 123.0
    assert state.avg_interval == 42.0
    assert state.idle_streak == 0  # never written by upsert_chat_state
    assert state.cfg_json == '{"a": 1}'


def test_upsert_chat_state_on_unknown_chat_is_noop(tmp_path):
    async def scenario():
        _db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat_state(ChatState(CK, focus_until=1.0))
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    assert run(scenario()) is None


def test_upsert_chat_state_cannot_move_cursor(tmp_path):
    """Omission attempt: a ChatState carrying a cursor must not move it."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.upsert_chat_state(ChatState(CK, cursor_msg_id=MessageRowId(99)))
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    state = run(scenario())
    assert state.cursor_msg_id is None  # untouched


def test_upsert_chat_state_cannot_write_hold_until(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat_state(ChatState(CK, hold_until=MessageRowId(99)))  # type: ignore[arg-type]
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    assert run(scenario()).hold_until is None


def test_upsert_chat_state_cannot_write_idle_streak(tmp_path):
    """The idle streak is materialized ONLY by finish_cycle: a later
    save_session must never reintroduce a crash gap for it."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat_state(ChatState(CK, idle_streak=7))
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    assert run(scenario()).idle_streak == 0  # untouched


# ── messages: atomic ingest, dedupe, identity derivation ────────────────────

def test_ingest_message_returns_row_id_and_inserted(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        result = await repo.ingest_message(
            make_identity(), make_message()
        )
        msg = await repo.get_message(CK, "m1")
        await repo.close()
        return result, msg

    result, msg = run(scenario())
    assert (result.row_id, result.inserted) == (1, True)
    assert result.echo_status == "not_applicable"
    assert msg is not None
    assert msg.row_id == result.row_id
    assert msg.text == "hello"
    assert msg.sender_id == "u1"
    assert msg.recv_ts == 1_700_000_000.0


def test_duplicate_message_dedupes_to_same_row(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        first = await repo.ingest_message(
            make_identity(), make_message()
        )
        second = await repo.ingest_message(
            make_identity(), make_message(text="changed")
        )
        count = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        )
        await repo.close()
        return first, second, count

    first, second, count = run(scenario())
    assert (first.row_id, first.inserted) == (1, True)
    assert (second.row_id, second.inserted) == (1, False)  # dedupe status surfaced
    assert count == 1  # ON CONFLICT DO NOTHING: one duplicate never poisons a batch


def test_ingest_message_commits_identity_and_message_atomically(tmp_path):
    async def scenario():
        _db, repo = await open_repo(tmp_path / "t.db")
        result = await repo.ingest_message(
            make_identity(), make_message()
        )
        chat = await repo.get_chat(CK)
        await repo.close()
        return result, chat

    result, chat = run(scenario())
    assert (result.row_id, result.inserted) == (1, True)
    assert chat is not None and chat.platform == "qq"


def test_platform_identity_derives_from_chat(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message())
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT platform, self_id FROM messages WHERE id = 1"
            ).fetchone()
        )
        await repo.close()
        return row

    assert run(scenario()) == ("qq", "bot-1")


def test_ingest_message_without_chat_identity_raises(tmp_path):
    async def scenario():
        _db, repo = await open_repo(tmp_path / "t.db")
        with pytest.raises(RepoError):
            await repo.ingest_message(None, make_message())
        await repo.close()

    run(scenario())


def test_message_without_platform_id_is_not_deduped(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        a = await repo.ingest_message(make_identity(), make_message(msg_id=None))
        b = await repo.ingest_message(make_identity(), make_message(msg_id=None))
        await repo.close()
        return a, b

    a, b = run(scenario())
    assert a.row_id == 1 and b.row_id == 2  # NULL platform ids never conflict


def test_forwarded_segment_payload_never_persists(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        msg = make_message(
            text="[转发消息]",
            segments=(
                Segment("forward", {"id": "fwd-1", "content": "secret forwarded text"}),
                Segment("text", {"text": "hi"}),
            ),
        )
        await repo.ingest_message(make_identity(), msg)
        stored = await repo.get_message(CK, "m1")
        await repo.close()
        return stored

    stored = run(scenario())
    assert stored is not None
    fwd = stored.segments[0]
    assert fwd.kind == "forward"
    assert fwd.data == {}  # content dropped
    assert fwd.raw is None
    assert stored.segments[1].data == {"text": "hi"}


# ── ingest pending count: atomic current pending non-self count ─────────────

def test_ingest_pending_count_counts_beyond_cursor(tmp_path):
    """A newly inserted non-self message reports the atomic CURRENT
    pending count (non-self beyond the durable cursor, itself included);
    self and duplicate inserts report None."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        first = await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", recv_ts=100.0)
        )
        second = await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", recv_ts=110.0)
        )
        self_result = await repo.ingest_message(
            make_identity(),
            make_message(msg_id="self1", is_self=True, recv_ts=115.0),
        )
        dup = await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", recv_ts=999.0)
        )
        await repo.close()
        return first, second, self_result, dup

    first, second, self_result, dup = run(scenario())
    assert first.pending_count == 1
    assert second.pending_count == 2
    assert self_result.pending_count is None  # self is never pending
    assert dup.pending_count is None  # noninserted: no atomic count


def test_ingest_pending_count_is_relative_to_durable_cursor(tmp_path):
    """The count is relative to the durable cursor: after finish_cycle
    advances it, only arrivals beyond the cursor count."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", recv_ts=100.0)
        )
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", recv_ts=110.0)
        )
        grant = await repo.claim_cycle(make_claim())
        await repo.finish_cycle(make_finish(), [], now=200.0)
        assert grant.through_msg_id == 2  # cursor now at 2
        after = await repo.ingest_message(
            make_identity(), make_message(msg_id="m3", recv_ts=120.0)
        )
        # A duplicate of a consumed message stays None (its row is behind
        # the cursor; no atomic count is reported).
        dup = await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", recv_ts=999.0)
        )
        await repo.close()
        return after, dup

    after, dup = run(scenario())
    assert after.pending_count == 1  # only m3 is beyond the cursor
    assert dup.pending_count is None


def test_ingest_pending_count_survives_restart(tmp_path):
    """The count is durable: after a restart the same cursor-relative
    count is reported for a new insert."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", recv_ts=100.0)
        )
        await repo.close()
        _db2, repo2 = await open_repo_with_chat(tmp_path / "t.db")
        result = await repo2.ingest_message(
            make_identity(), make_message(msg_id="m2", recv_ts=110.0)
        )
        await repo2.close()
        return result

    result = run(scenario())
    assert result.pending_count == 2


def test_ingest_pending_count_matches_claim_pending(tmp_path):
    """The reported count equals the pending tuple a subsequent claim
    grants (the same non-self-beyond-cursor predicate)."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        r1 = await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", recv_ts=100.0)
        )
        r2 = await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", recv_ts=110.0)
        )
        await repo.ingest_message(
            make_identity(),
            make_message(msg_id="s1", is_self=True, recv_ts=115.0),
        )
        grant = await repo.claim_cycle(make_claim())
        await repo.close()
        return r1, r2, grant

    r1, r2, grant = run(scenario())
    assert r1.pending_count == 1
    assert r2.pending_count == 2
    assert len(grant.pending) == 2  # self excluded, same predicate


def test_ingest_pending_count_echo_and_avg_unchanged(tmp_path):
    """The pending count rides alongside the existing trusted-echo and
    EWMA behavior: a reconciled self echo reports None and the durable
    average is untouched."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", recv_ts=100.0)
        )
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", recv_ts=130.0)
        )
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        echo = await repo.ingest_message(
            make_identity(), _echo(), self_echo_delivery_key="cy-1:0"
        )
        state = await repo.get_chat_state(CK)
        await repo.close()
        return echo, state

    echo, state = run(scenario())
    assert echo.inserted is True
    assert echo.echo_status == "reconciled"
    assert echo.pending_count is None  # self echo: never pending
    assert state is not None and state.avg_interval == pytest.approx(30.0)


# ── recent snapshot: the claim-bounded gate read ────────────────────────────

def test_recent_snapshot_returns_limited_list_and_full_window_counts(tmp_path):
    """The rendered list is limited; the FULL-window counts (self messages
    included) never change with a small limit."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        for i in range(1, 6):
            await repo.ingest_message(
                make_identity(),
                make_message(msg_id=f"m{i}", text=f"msg{i}", recv_ts=100.0 + i),
            )
        await repo.ingest_message(
            make_identity(),
            make_message(msg_id="self1", text="self", is_self=True, recv_ts=106.0),
        )
        full = await repo.get_recent_snapshot(CK, MessageRowId(6), 0.0, 100)
        limited = await repo.get_recent_snapshot(CK, MessageRowId(6), 0.0, 3)
        await repo.close()
        return full, limited

    full, limited = run(scenario())
    assert full.window_count == 6  # self included
    assert full.self_count == 1
    assert len(full.messages) == 6
    assert limited.window_count == 6  # full-window counts, unchanged
    assert limited.self_count == 1
    assert len(limited.messages) == 3  # the LIMITED rendered list
    assert [m.text for m in limited.messages] == ["self", "msg5", "msg4"]
    assert limited.last_nonself_ts == 105.0
    assert limited.since_ts == 0.0
    assert limited.through_row_id == MessageRowId(6)


def test_recent_snapshot_fences_time_and_row_bounds(tmp_path):
    """Only rows with recv_ts >= since_ts AND id <= through_row_id appear;
    the since_ts boundary is inclusive."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            make_identity(), make_message(msg_id="old", text="old", recv_ts=50.0)
        )
        await repo.ingest_message(
            make_identity(), make_message(msg_id="in1", text="in1", recv_ts=100.0)
        )
        await repo.ingest_message(
            make_identity(), make_message(msg_id="in2", text="in2", recv_ts=150.0)
        )
        await repo.ingest_message(
            make_identity(), make_message(msg_id="late", text="late", recv_ts=200.0)
        )
        snap = await repo.get_recent_snapshot(CK, MessageRowId(3), 100.0, 10)
        await repo.close()
        return snap

    snap = run(scenario())
    assert [m.text for m in snap.messages] == ["in2", "in1"]  # row 4 fenced out
    assert snap.window_count == 2
    assert snap.self_count == 0
    assert snap.last_nonself_ts == 150.0


def test_recent_snapshot_excludes_late_arrivals_after_boundary(tmp_path):
    """Post-boundary rows must never appear even if inserted after the
    snapshot boundary was fixed."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", text="m1", recv_ts=100.0)
        )
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", text="m2", recv_ts=110.0)
        )
        before = await repo.get_recent_snapshot(CK, MessageRowId(2), 100.0, 10)
        # Late arrivals: a row past the row boundary, and a row inside the
        # row boundary but before the time boundary.
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m3", text="m3", recv_ts=120.0)
        )
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m4", text="m4", recv_ts=90.0)
        )
        after = await repo.get_recent_snapshot(CK, MessageRowId(2), 100.0, 10)
        await repo.close()
        return before, after

    before, after = run(scenario())
    assert [m.text for m in before.messages] == ["m2", "m1"]
    assert [m.text for m in after.messages] == ["m2", "m1"]  # unchanged
    assert after.window_count == 2
    assert after.self_count == 0


def test_recent_snapshot_empty_and_only_self_windows(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        empty = await repo.get_recent_snapshot(CK, MessageRowId(0), 0.0, 10)
        await repo.ingest_message(
            make_identity(),
            make_message(msg_id="s1", text="self", is_self=True, recv_ts=100.0),
        )
        only_self = await repo.get_recent_snapshot(CK, MessageRowId(1), 0.0, 10)
        await repo.close()
        return empty, only_self

    empty, only_self = run(scenario())
    assert empty.messages == ()
    assert empty.window_count == 0
    assert empty.self_count == 0
    assert empty.last_nonself_ts is None
    assert only_self.window_count == 1
    assert only_self.self_count == 1
    assert [m.text for m in only_self.messages] == ["self"]
    assert only_self.last_nonself_ts is None  # no non-self message in window


def test_recent_snapshot_rejects_non_positive_limit(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        with pytest.raises(ValueError, match="limit"):
            await repo.get_recent_snapshot(CK, MessageRowId(0), 0.0, 0)
        with pytest.raises(ValueError, match="limit"):
            await repo.get_recent_snapshot(CK, MessageRowId(0), 0.0, -1)
        await repo.close()

    run(scenario())


def test_recent_snapshot_maps_message_rows_correctly(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            make_identity(),
            make_message(msg_id="m1", text="hi", sender_id="u1", recv_ts=100.0),
        )
        snap = await repo.get_recent_snapshot(CK, MessageRowId(1), 0.0, 10)
        await repo.close()
        return snap

    snap = run(scenario())
    msg = snap.messages[0]
    assert msg.row_id == MessageRowId(1)
    assert msg.chat_key == CK
    assert msg.text == "hi"
    assert msg.sender_id == "u1"
    assert msg.is_self is False
    assert msg.recv_ts == 100.0


def test_sqlite_repository_satisfies_repository_protocol(tmp_path):
    """The concrete repository is structurally a Repository (runtime
    protocol validation of the full async surface)."""

    async def scenario():
        _db, repo = await open_repo(tmp_path / "t.db")
        ok = isinstance(repo, Repository)
        await repo.close()
        return ok

    assert run(scenario()) is True


# ── CJK bigram FTS ──────────────────────────────────────────────────────────

async def _fts_rowids(db, query: str) -> list[int]:
    tokens = bigram_tokenize(query)
    match = " OR ".join(f'"{t}"' for t in tokens)
    return await db.read(
        lambda c: [
            r[0]
            for r in c.execute(
                "SELECT rowid FROM message_fts WHERE message_fts MATCH ?", (match,)
            ).fetchall()
        ]
    )


def test_two_character_chinese_search_works(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1", text="火锅真好吃"))
        await repo.ingest_message(make_identity(), make_message(msg_id="m2", text="今天天气不错"))
        hits = await _fts_rowids(db, "火锅")
        hits2 = await _fts_rowids(db, "好吃")
        hits3 = await _fts_rowids(db, "天气")
        await repo.close()
        return hits, hits2, hits3

    hits, hits2, hits3 = run(scenario())
    assert hits == [1]
    assert hits2 == [1]
    assert hits3 == [2]


def test_fts_update_is_in_same_transaction_as_insert(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1", text="火锅"))
        hits = await _fts_rowids(db, "火锅")
        await repo.close()
        return hits

    assert run(scenario()) == [1]


def test_fts_matches_ascii_and_mixed_text(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1", text="hello world"))
        await repo.ingest_message(make_identity(), make_message(msg_id="m2", text="hello火锅"))
        hits = await _fts_rowids(db, "hello")
        hits2 = await _fts_rowids(db, "火锅")
        await repo.close()
        return hits, hits2

    hits, hits2 = run(scenario())
    assert hits == [1, 2]
    assert hits2 == [2]


def test_bigram_tokenize_units():
    assert bigram_tokenize("火锅好吃") == ["火锅", "锅好", "好吃"]
    assert bigram_tokenize("火") == ["火"]
    assert bigram_tokenize("hello火锅world") == ["hello", "火锅", "world"]


# ── claims: bounded grant, CAS, lease, recovery ─────────────────────────────

def test_claim_returns_grant_with_boundary_and_pending(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1", text="a"))
        await repo.ingest_message(make_identity(), make_message(msg_id="m2", text="b"))
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m3", text="self", is_self=True)
        )
        grant = await repo.claim_cycle(make_claim())
        await repo.close()
        return grant

    grant = run(scenario())
    assert grant is not None
    assert grant.start_msg_id == 0
    assert grant.through_msg_id == 3
    # Pending excludes is_self and is bounded by the claim boundary.
    assert [m.text for m in grant.pending] == ["a", "b"]


def test_claim_is_compare_and_swap(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        first = await repo.claim_cycle(make_claim())
        second = await repo.claim_cycle(
            make_claim(cycle_id="cy-2", started_ts=110.0)
        )
        await repo.upsert_chat(make_identity(chat_key="qq:group:other"))
        other = await repo.claim_cycle(
            make_claim(chat_key="qq:group:other", cycle_id="cy-3")
        )
        await repo.close()
        return first, second, other

    first, second, other = run(scenario())
    assert first is not None
    assert isinstance(second, ClaimBusy)  # live, unexpired claim blocks
    assert second.busy_until == 500.0
    assert other is not None  # different chat


def test_claim_unknown_chat_returns_none(tmp_path):
    async def scenario():
        _db, repo = await open_repo(tmp_path / "t.db")
        grant = await repo.claim_cycle(make_claim())
        await repo.close()
        return grant

    assert run(scenario()) is None


def test_claim_boundary_is_fixed_at_claim_time(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.ingest_message(make_identity(), make_message(msg_id="m2"))
        await repo.ingest_message(make_identity(), make_message(msg_id="m3"))
        grant = await repo.claim_cycle(make_claim())
        # New arrivals after the claim stay for the next claim.
        await repo.ingest_message(make_identity(), make_message(msg_id="m4"))
        await repo.close()
        return grant

    grant = run(scenario())
    assert grant.through_msg_id == 3
    assert len(grant.pending) == 3


def test_renew_cycle_extends_lease(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim())
        ok = await repo.renew_cycle(CK, "cy-1", 999.0, now=200.0)
        wrong = await repo.renew_cycle(CK, "cy-other", 999.0, now=200.0)
        states = await repo._db.read(
            lambda c: c.execute(
                "SELECT expires_at FROM claims WHERE cycle_id = 'cy-1'"
            ).fetchone()
        )
        await repo.close()
        return ok, wrong, states

    ok, wrong, states = run(scenario())
    assert ok and not wrong
    assert states[0] == 999.0


def test_expired_owner_cannot_renew(tmp_path):
    """An expired owner cannot renew even before another claimant acts."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim(started_ts=50.0, expires_at=100.0))
        renewed = await repo.renew_cycle(CK, "cy-1", 999.0, now=150.0)
        # A non-finite lease is also rejected.
        bad_lease = await repo.renew_cycle(CK, "cy-1", 100.0, now=150.0)
        await repo.close()
        return renewed, bad_lease

    assert run(scenario()) == (False, False)


def test_release_cycle_does_not_move_cursor(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.claim_cycle(make_claim())
        await repo.release_cycle(CK, "cy-1")
        state = await repo.get_chat_state(CK)
        again = await repo.claim_cycle(make_claim(cycle_id="cy-2", started_ts=120.0))
        await repo.close()
        return state, again

    state, again = run(scenario())
    assert state.cursor_msg_id is None
    assert again is not None


def test_expired_claim_is_recovered(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim(started_ts=50.0, expires_at=100.0))
        # New claim starts at 200: the old lease (100) has expired.
        recovered = await repo.claim_cycle(
            make_claim(cycle_id="cy-2", started_ts=200.0)
        )
        states = await repo._db.read(
            lambda c: [
                r[0] for r in c.execute("SELECT state FROM claims ORDER BY id").fetchall()
            ]
        )
        await repo.close()
        return recovered, states

    recovered, states = run(scenario())
    assert recovered is not None
    assert recovered.claim.cycle_id == "cy-2"
    assert states == ["expired", "live"]


def test_live_claim_blocks_until_expiry(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim(expires_at=500.0))
        blocked = await repo.claim_cycle(
            make_claim(cycle_id="cy-2", started_ts=200.0)
        )
        await repo.close()
        return blocked

    blocked = run(scenario())
    assert isinstance(blocked, ClaimBusy)
    assert blocked.busy_until == 500.0


# ── claim busy: typed unexpired-owner result ────────────────────────────────

def test_claim_busy_reports_exact_busy_until(tmp_path):
    """A live, unexpired claim is reported as a typed ClaimBusy carrying
    the active owner's exact busy_until — never a bare None."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim(expires_at=500.0))
        busy = await repo.claim_cycle(
            make_claim(cycle_id="cy-2", started_ts=200.0, expires_at=900.0)
        )
        await repo.close()
        return busy

    busy = run(scenario())
    assert isinstance(busy, ClaimBusy)
    assert busy.chat_key == CK
    assert busy.cycle_id == CycleId("cy-1")  # the active owner
    assert busy.busy_until == 500.0  # the owner's exact expiry, not the new claim's


def test_claim_busy_until_tracks_renewed_lease(tmp_path):
    """busy_until is the exact STORED expiry: a renewed lease is reported
    at its renewed value."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim(expires_at=500.0))
        await repo.renew_cycle(CK, CycleId("cy-1"), 999.0, now=200.0)
        busy = await repo.claim_cycle(
            make_claim(cycle_id="cy-2", started_ts=200.0)
        )
        await repo.close()
        return busy

    busy = run(scenario())
    assert isinstance(busy, ClaimBusy)
    assert busy.busy_until == 999.0


def test_claim_grant_busy_none_distinction(tmp_path):
    """The three outcomes are distinct: None for an unknown chat, ClaimBusy
    for a live unexpired owner, ClaimGrant after expiry recovery."""

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        unknown = await repo.claim_cycle(make_claim())
        await repo.upsert_chat(make_identity())
        await repo.claim_cycle(make_claim(started_ts=50.0, expires_at=100.0))
        busy = await repo.claim_cycle(
            make_claim(cycle_id="cy-2", started_ts=80.0)
        )
        recovered = await repo.claim_cycle(
            make_claim(cycle_id="cy-3", started_ts=200.0)
        )
        await repo.close()
        return unknown, busy, recovered

    unknown, busy, recovered = run(scenario())
    assert unknown is None
    assert isinstance(busy, ClaimBusy)
    assert busy.busy_until == 100.0
    assert isinstance(recovered, ClaimGrant)
    assert recovered.claim.cycle_id == "cy-3"


def test_claim_busy_does_not_expose_raw_rows(tmp_path):
    """ClaimBusy is a typed frozen result: no raw claim row attributes
    (id, state, started_ts) leak to consumers."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim())
        busy = await repo.claim_cycle(
            make_claim(cycle_id="cy-2", started_ts=200.0)
        )
        await repo.close()
        return busy

    busy = run(scenario())
    assert isinstance(busy, ClaimBusy)
    assert not hasattr(busy, "id")
    assert not hasattr(busy, "state")
    assert not hasattr(busy, "started_ts")
    with pytest.raises(dataclasses.FrozenInstanceError):
        busy.busy_until = 1.0  # type: ignore[misc]


# ── claim leases: finiteness and <= expiry semantics ────────────────────────

def test_claim_lease_rejects_non_finite_timestamps():
    with pytest.raises(ValueError, match="finite"):
        make_claim(started_ts=float("inf"), expires_at=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        make_claim(started_ts=0.0, expires_at=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        make_claim(started_ts=0.0, expires_at=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        make_claim(started_ts=float("nan"), expires_at=100.0)


def test_claim_lease_rejects_equality():
    with pytest.raises(ValueError, match="finite"):
        make_claim(started_ts=100.0, expires_at=100.0)


def test_recovery_uses_le_expiry_semantics(tmp_path):
    """Recovery treats a lease expiring exactly at the new claim's start as
    expired (same <= semantics as finish and renew)."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim(started_ts=50.0, expires_at=100.0))
        # New claim starts exactly at the old lease's expiry: recovered.
        recovered = await repo.claim_cycle(
            make_claim(cycle_id="cy-2", started_ts=100.0)
        )
        states = await repo._db.read(
            lambda c: [r[0] for r in c.execute("SELECT state FROM claims ORDER BY id")]
        )
        await repo.close()
        return recovered, states

    recovered, states = run(scenario())
    assert recovered is not None
    assert recovered.claim.cycle_id == "cy-2"
    assert states == ["expired", "live"]


def test_renew_rejects_non_finite_lease(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim())
        inf_ok = await repo.renew_cycle(CK, "cy-1", float("inf"), now=200.0)
        nan_ok = await repo.renew_cycle(CK, "cy-1", float("nan"), now=200.0)
        past_ok = await repo.renew_cycle(CK, "cy-1", 100.0, now=200.0)
        await repo.close()
        return inf_ok, nan_ok, past_ok

    assert run(scenario()) == (False, False, False)


def test_finish_rejects_non_finite_stored_lease(tmp_path):
    """A stored lease corrupted to a non-finite value is treated as
    expired by finish (same <= semantics as renew and recovery)."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim())
        await repo._db.write(
            lambda c: c.execute(
                "UPDATE claims SET expires_at = ? WHERE cycle_id = 'cy-1'",
                (float("inf"),),
            )
        )
        with pytest.raises(ClaimError, match="expired"):
            await repo.finish_cycle(make_finish(), [], now=200.0)
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    state = run(scenario())
    assert state.cursor_msg_id is None  # nothing moved


# ── finish_cycle: fences, cursor derivation, atomic batch ───────────────────

def test_finish_derives_cursor_from_claim_boundary(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.ingest_message(make_identity(), make_message(msg_id="m2"))
        await repo.ingest_message(make_identity(), make_message(msg_id="m3"))
        grant = await repo.claim_cycle(make_claim())
        await repo.finish_cycle(make_finish(), [], now=200.0)
        state = await repo.get_chat_state(CK)
        await repo.close()
        return grant, state

    grant, state = run(scenario())
    assert grant.through_msg_id == 3
    assert state.cursor_msg_id == 3  # derived from the claim boundary


def test_finish_does_not_consume_post_claim_arrivals(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.ingest_message(make_identity(), make_message(msg_id="m2"))
        await repo.claim_cycle(make_claim())
        await repo.ingest_message(make_identity(), make_message(msg_id="m3"))
        await repo.finish_cycle(make_finish(), [], now=200.0)
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    state = run(scenario())
    assert state.cursor_msg_id == 2  # boundary, not MAX(id) at finish time


def test_finish_persists_cycle_and_releases_claim(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim())
        await repo.finish_cycle(make_finish(), [], now=200.0)
        cycle = await repo._db.read(
            lambda c: c.execute(
                "SELECT end_reason, trace_json, tokens_in, tokens_out, started_ts"
                " FROM cycles"
            ).fetchone()
        )
        states = await repo._db.read(
            lambda c: [
                r[0] for r in c.execute("SELECT state FROM claims").fetchall()
            ]
        )
        await repo.close()
        return cycle, states

    cycle, states = run(scenario())
    assert cycle[0] == "completed"
    assert cycle[1] == '{"t": 1}'
    assert cycle[2] == 10 and cycle[3] == 20
    assert cycle[4] == 100.0  # started_ts from the claim
    assert states == ["finished"]


def test_finish_hold_until_separate_from_focus_until(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim())
        await repo.finish_cycle(
            make_finish(end_reason="held", hold_until=1_700_000_600.0), [], now=200.0
        )
        state = await repo.get_chat_state(CK)
        # Focus mode writes focus_until only; hold_until survives.
        await repo.upsert_chat_state(ChatState(CK, focus_until=1_700_000_700.0))
        state2 = await repo.get_chat_state(CK)
        await repo.close()
        return state, state2

    state, state2 = run(scenario())
    assert state.hold_until == 1_700_000_600.0
    assert state.focus_until is None  # distinct columns
    assert state2.focus_until == 1_700_000_700.0
    assert state2.hold_until == 1_700_000_600.0  # untouched by state updates


def test_finish_materializes_idle_streak_after_atomically(tmp_path):
    """Idle backoff is materialized transactionally at terminal completion
    as idle_streak_after — in the SAME transaction as the cursor advance."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim())
        await repo.finish_cycle(
            make_finish(end_reason="planner_no_tool_end", idle_streak_after=3),
            [], now=200.0,
        )
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    state = run(scenario())
    assert state.idle_streak == 3
    assert state.hold_until is None  # no hold on this outcome


def test_finish_terminal_reset_clears_hold_and_resets_streak(tmp_path):
    """Skip / dry-run trigger terminally finish with an empty outbox: the
    same transaction clears the hold and resets the idle streak."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        # Cycle 1: a held backoff outcome materializes hold + streak.
        await repo.claim_cycle(make_claim())
        await repo.finish_cycle(
            make_finish(end_reason="planner_no_tool_end", hold_until=1_700_000_600.0,
                        idle_streak_after=4),
            [], now=200.0,
        )
        # Cycle 2: a terminal reset (skip) clears both.
        await repo.claim_cycle(make_claim(cycle_id="cy-2", started_ts=300.0))
        await repo.finish_cycle(
            make_finish(cycle_id="cy-2", end_reason="skip", hold_until=None,
                        idle_streak_after=0),
            [], now=400.0,
        )
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    state = run(scenario())
    assert state.hold_until is None  # cleared by the terminal reset
    assert state.idle_streak == 0  # reset by the terminal reset


def test_finish_clears_previous_hold_when_not_held(tmp_path):
    """A non-held finish ALWAYS writes hold_until (None clears a stale
    hold) — the hold window is never left behind by a later outcome."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim())
        await repo.finish_cycle(
            make_finish(end_reason="held", hold_until=1_700_000_600.0), [], now=200.0
        )
        await repo.claim_cycle(make_claim(cycle_id="cy-2", started_ts=300.0))
        await repo.finish_cycle(
            make_finish(cycle_id="cy-2", end_reason="completed"), [], now=400.0
        )
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    state = run(scenario())
    assert state.hold_until is None  # the stale hold was cleared


def test_finish_failure_does_not_touch_hold_or_streak(tmp_path):
    """A fenced-out finish changes nothing — including the durable hold
    window and idle streak."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim())
        await repo.finish_cycle(
            make_finish(end_reason="held", hold_until=1_700_000_600.0,
                        idle_streak_after=2),
            [], now=200.0,
        )
        # A stale finish attempt must not clobber the materialized state.
        with pytest.raises(ClaimError):
            await repo.finish_cycle(
                make_finish(cycle_id="cy-other", hold_until=None, idle_streak_after=0),
                [], now=400.0,
            )
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    state = run(scenario())
    assert state.hold_until == 1_700_000_600.0
    assert state.idle_streak == 2


def test_finish_inserts_outbox_batch_with_cycle_provenance(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim())
        batch = [
            item("part 1", idem_key="g1:0", group_id="g1", seq=0),
            item("part 2", idem_key="g1:1", group_id="g1", seq=1),
        ]
        await repo.finish_cycle(make_finish(), batch, now=200.0)
        rows = await repo._db.read(
            lambda c: c.execute(
                "SELECT chat_key, cycle_id, group_id, seq, text, state"
                " FROM outbox ORDER BY seq"
            ).fetchall()
        )
        cycle_id = await repo._db.read(
            lambda c: c.execute("SELECT id FROM cycles").fetchone()[0]
        )
        await repo.close()
        return rows, cycle_id

    rows, cycle_id = run(scenario())
    assert rows == [
        (CK, cycle_id, "g1", 0, "part 1", "pending"),
        (CK, cycle_id, "g1", 1, "part 2", "pending"),
    ]


def test_finish_rejects_cross_chat_outbox_item(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key="qq:group:other"))
        await repo.claim_cycle(make_claim())
        with pytest.raises(RepoError, match="cross-chat"):
            await repo.finish_cycle(
                make_finish(), [item(chat_key=OTHER)], now=200.0
            )
        state = await repo.get_chat_state(CK)
        states = await repo._db.read(
            lambda c: [r[0] for r in c.execute("SELECT state FROM claims")]
        )
        cycles = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
        )
        await repo.close()
        return state, states, cycles

    state, states, cycles = run(scenario())
    assert state.cursor_msg_id is None  # nothing moved
    assert states == ["live"]  # claim still live
    assert cycles == 0


def test_stale_finalization_fails(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim())
        with pytest.raises(ClaimError):
            await repo.finish_cycle(make_finish(cycle_id="cy-other"), [], now=200.0)
        await repo.release_cycle(CK, "cy-1")
        with pytest.raises(ClaimError):
            await repo.finish_cycle(make_finish(), [], now=200.0)
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    state = run(scenario())
    assert state.cursor_msg_id is None  # nothing moved


def test_expired_owner_cannot_finish(tmp_path):
    """An expired owner cannot finish even before another claimant acts."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim(started_ts=50.0, expires_at=100.0))
        with pytest.raises(ClaimError, match="expired"):
            await repo.finish_cycle(make_finish(), [], now=150.0)
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    state = run(scenario())
    assert state.cursor_msg_id is None


def test_finish_rejected_when_cursor_moved_past_claim_start(tmp_path):
    """Backwards/forward cursor tampering: the start-cursor fence fails."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.claim_cycle(make_claim())
        # An external write moves the cursor past the claim's start.
        await repo._db.write(
            lambda c: c.execute(
                "UPDATE chats SET cursor_msg_id = 5 WHERE chat_key = ?", (CK,)
            )
        )
        with pytest.raises(ClaimError, match="start boundary"):
            await repo.finish_cycle(make_finish(), [], now=200.0)
        states = await repo._db.read(
            lambda c: [r[0] for r in c.execute("SELECT state FROM claims")]
        )
        await repo.close()
        return states

    assert run(scenario()) == ["live"]  # claim not released by the failed finish


def test_finish_rejected_when_cursor_moved_backwards(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.ingest_message(make_identity(), make_message(msg_id="m2"))
        await repo.claim_cycle(make_claim())
        await repo.finish_cycle(make_finish(), [], now=200.0)  # cursor -> 2
        await repo.claim_cycle(make_claim(cycle_id="cy-2", started_ts=300.0))
        # Cursor moved BACKWARDS to 1: the second claim's start (2) no
        # longer matches.
        await repo._db.write(
            lambda c: c.execute(
                "UPDATE chats SET cursor_msg_id = 1 WHERE chat_key = ?", (CK,)
            )
        )
        with pytest.raises(ClaimError, match="start boundary"):
            await repo.finish_cycle(
                make_finish(cycle_id="cy-2"), [], now=400.0
            )
        await repo.close()

    run(scenario())


def test_finish_failure_injection_changes_nothing(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.claim_cycle(make_claim())

        def boom(conn, items, chat_key, cycle_id):
            raise RuntimeError("injected outbox failure")

        repo._insert_outbox_batch = boom  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            await repo.finish_cycle(
                make_finish(), [item("x", idem_key="k1")], now=200.0
            )
        state = await repo.get_chat_state(CK)
        states = await repo._db.read(
            lambda c: [r[0] for r in c.execute("SELECT state FROM claims")]
        )
        cycles = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
        )
        outbox = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        await repo.close()
        return state, states, cycles, outbox

    state, states, cycles, outbox = run(scenario())
    assert state.cursor_msg_id is None  # cursor not advanced
    assert states == ["live"]  # claim still live
    assert cycles == 0  # no terminal cycle
    assert outbox == 0  # no outbox rows


def test_finish_then_finish_again_is_stale(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim())
        await repo.finish_cycle(make_finish(), [], now=200.0)
        with pytest.raises(ClaimError):
            await repo.finish_cycle(make_finish(), [], now=200.0)
        await repo.close()

    run(scenario())


def test_concurrent_stale_finish_plus_inbound_commit(tmp_path):
    """A stale finish (ClaimError) in the same writer batch as an inbound
    ingest commit must not roll the ingest back (savepoint isolation)."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim())

        async def stale_finish() -> None:
            await repo.finish_cycle(make_finish(cycle_id="cy-other"), [], now=200.0)

        async def inbound() -> None:
            await repo.ingest_message(
                make_identity(), make_message(msg_id="m1", text="inbound")
            )

        results = await asyncio.gather(
            asyncio.create_task(stale_finish()),
            asyncio.create_task(inbound()),
            return_exceptions=True,
        )
        assert isinstance(results[0], ClaimError)
        assert results[1] is None
        msg = await repo.get_message(CK, "m1")
        await repo.close()
        return msg

    msg = run(scenario())
    assert msg is not None and msg.text == "inbound"  # committed despite the fence


# ── outbox: transition fences, at-most-once, reconciliation ─────────────────

def test_list_ready_outbox_respects_pacing(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(
            repo,
            [
                item("now", idem_key="k1"),
                item("later", idem_key="k2", send_after_ts=500.0),
            ],
        )
        ready = await repo.list_ready_outbox(CK, now=100.0)
        later = await repo.list_ready_outbox(CK, now=600.0)
        await repo.close()
        return [i.text for i in ready], [i.text for i in later]

    ready, later = run(scenario())
    assert ready == ["now"]
    assert later == ["now", "later"]


def test_next_due_outbox_schedules_worker_wake(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        # Nothing pending: no wake needed.
        none = await repo.next_due_outbox(CK, now=100.0)
        await finish_batch(
            repo,
            [
                item("now", idem_key="k1"),
                item("later", idem_key="k2", send_after_ts=500.0),
            ],
        )
        # A due row (NULL send_after_ts) counts as due now.
        due_now = await repo.next_due_outbox(CK, now=100.0)
        # Only the future row remains after the due one is attempted.
        await repo.attempt_outbox(1, 100.0)
        future = await repo.next_due_outbox(CK, now=100.0)
        await repo.close()
        return none, due_now, future

    none, due_now, future = run(scenario())
    assert none is None
    assert due_now == 0.0  # something is ready right now
    assert future == 500.0  # the earliest future send_after_ts


def test_same_text_sends_in_different_cycles(tmp_path):
    """Idempotency keys represent delivery intent: the same text may send
    in different cycles (distinct keys), while retrying the same cycle
    produces the same keys."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        driver = OutboxDriver(repo, ConsoleAdapter())
        out = Outgoing(chat_key=CK, text="hi", idem_key="k1")
        await finish_batch(
            repo, driver.to_items(out, "cy-1"), cycle_id="cy-1"
        )
        await finish_batch(
            repo,
            driver.to_items(out, "cy-2"),
            cycle_id="cy-2", started_ts=300.0, now=400.0,
        )
        rows = await repo._db.read(
            lambda c: c.execute(
                "SELECT text, idem_key, cycle_id FROM outbox ORDER BY id"
            ).fetchall()
        )
        await repo.close()
        return rows

    rows = run(scenario())
    assert len(rows) == 2  # same text, two deliveries
    assert rows[0][0] == rows[1][0] == "hi"
    assert rows[0][1] != rows[1][1]  # distinct keys (cycle-scoped)
    assert rows[0][2] != rows[1][2]  # distinct cycle provenance


def test_attempt_outbox_is_a_cas(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="k1")])
        first = await repo.attempt_outbox(1, 100.0)
        second = await repo.attempt_outbox(1, 101.0)
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT state, attempt_started_ts FROM outbox WHERE id = 1"
            ).fetchone()
        )
        await repo.close()
        return first, second, row

    first, second, row = run(scenario())
    assert first and not second
    assert row == ("in_flight", 100.0)


def test_in_flight_items_never_listed_after_restart(tmp_path):
    path = tmp_path / "t.db"

    async def scenario():
        _db, repo = await open_repo_with_chat(path)
        await finish_batch(repo, [item("hi", idem_key="k1")])
        await repo.attempt_outbox(1, 100.0)
        await repo.close()
        _db2, repo2 = await open_repo_with_chat(path)
        ready = await repo2.list_ready_outbox(CK, now=999.0)
        await repo2.close()
        return ready

    assert run(scenario()) == []  # in_flight is never auto-retried


def test_mark_outbox_sent_only_from_in_flight(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("bot says hi", idem_key="k1")])
        # Pending -> sent is NOT a legal transition.
        blocked = await repo.mark_outbox_sent(1, "console:out:1", 101.0)
        state = await repo._db.read(
            lambda c: c.execute("SELECT state FROM outbox WHERE id = 1").fetchone()[0]
        )
        await repo.attempt_outbox(1, 100.0)
        ok = await repo.mark_outbox_sent(1, "console:out:1", 101.0)
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT state, sent_ts, platform_msg_id FROM outbox WHERE id = 1"
            ).fetchone()
        )
        echo = await repo.get_message(CK, "console:out:1")
        await repo.close()
        return blocked, state, ok, row, echo

    blocked, state, ok, row, echo = run(scenario())
    assert blocked is False
    assert state == "pending"  # untouched
    assert ok is True
    assert row == ("sent", 101.0, "console:out:1")
    assert echo is not None
    assert echo.is_self
    assert echo.text == "bot says hi"
    assert echo.sender_id == "bot-1"  # the chat's self id
    assert echo.recv_ts == 101.0


def test_mark_outbox_sent_is_idempotent(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="k1")])
        await repo.attempt_outbox(1, 100.0)
        first = await repo.mark_outbox_sent(1, "console:out:1", 101.0)
        second = await repo.mark_outbox_sent(1, "console:out:1", 102.0)
        count = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        )
        await repo.close()
        return first, second, count

    first, second, count = run(scenario())
    assert first is True and second is False
    assert count == 1  # one echo only


def test_mark_outbox_sent_without_platform_id_uses_local_fallback(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="k1")])
        await repo.attempt_outbox(1, 100.0)
        await repo.mark_outbox_sent(1, None, 101.0)
        echo = await repo.get_message(CK, "local:1")
        await repo.close()
        return echo

    echo = run(scenario())
    assert echo is not None and echo.is_self


def test_drop_outbox_only_from_pending(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(
            repo,
            [
                item("a", idem_key="k1"),
                item("b", idem_key="k2"),
                item("c", idem_key="k3"),
            ],
        )
        await repo.attempt_outbox(2, 100.0)  # in_flight
        await repo.attempt_outbox(3, 100.0)
        await repo.mark_outbox_sent(3, "console:out:3", 101.0)  # sent
        drop_pending = await repo.drop_outbox(1)
        drop_in_flight = await repo.drop_outbox(2)
        drop_sent = await repo.drop_outbox(3)
        states = await repo._db.read(
            lambda c: [
                r[0] for r in c.execute("SELECT state FROM outbox ORDER BY id")
            ]
        )
        await repo.close()
        return drop_pending, drop_in_flight, drop_sent, states

    dp, di, ds, states = run(scenario())
    assert dp is True
    assert di is False and ds is False  # staleness drop never rewrites these
    assert states == ["dropped", "in_flight", "sent"]


def test_no_untrusted_reconcile_outbox_surface(tmp_path):
    """The ONLY reconciliation path is the trusted self-echo key flow
    inside ingest_message: no item-id reconciliation method may exist on
    the concrete repository or the public seam."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.close()
        return repo

    repo = run(scenario())
    assert not hasattr(SqliteRepository, "reconcile_outbox")
    assert not hasattr(Repository, "reconcile_outbox")
    assert not hasattr(repo, "reconcile_outbox")


def test_real_echo_dedupes_against_synthetic_echo(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="k1")])
        await repo.attempt_outbox(1, 100.0)
        await repo.mark_outbox_sent(1, "console:out:1", 101.0)
        result = await repo.ingest_message(
            make_identity(), make_message(msg_id="console:out:1", is_self=True)
        )
        count = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        )
        await repo.close()
        return result, count

    result, count = run(scenario())
    assert result.inserted is False  # reconciled, not duplicated
    assert result.echo_status == "unproven"  # no trusted key: never matched
    assert count == 1


# ── startup recovery: pending chats beyond the durable cursor ───────────────

def test_list_pending_chats_selects_chats_with_pending_nonself(tmp_path):
    """A chat is pending exactly when it holds a non-self message beyond
    its durable cursor; self messages and cursor-covered rows never make
    it pending, and other chats are never included."""

    async def scenario():
        _db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())  # CK
        await repo.upsert_chat(make_identity(chat_key="qq:group:other"))
        await repo.upsert_chat(make_identity(chat_key="qq:group:empty"))
        # CK: two non-self messages, then a self message beyond them.
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", recv_ts=100.0)
        )
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", recv_ts=110.0)
        )
        await repo.ingest_message(
            make_identity(),
            make_message(msg_id="self1", is_self=True, recv_ts=120.0),
        )
        # other: one non-self message, then a finished cycle covers it
        # (the cursor advances ONLY through finish_cycle).
        other = ChatKey("qq:group:other")
        await repo.ingest_message(
            make_identity(chat_key="qq:group:other"),
            make_message(chat_key="qq:group:other", msg_id="o1", recv_ts=100.0),
        )
        grant = await repo.claim_cycle(
            make_claim(chat_key="qq:group:other", cycle_id="cy-other")
        )
        assert grant is not None
        await repo.finish_cycle(
            make_finish(chat_key="qq:group:other", cycle_id="cy-other"),
            [],
            now=200.0,
        )
        # empty: no messages at all.
        before = await repo.list_pending_chats()
        await repo.close()
        return before

    before = run(scenario())
    assert before == [CK]  # deterministic order; other/empty excluded


def test_list_pending_chats_after_cursor_advance_and_restart(tmp_path):
    """finish_cycle advances the cursor; the covered chat drops out of the
    pending set, and the surviving set is identical after close/reopen."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", recv_ts=100.0)
        )
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", recv_ts=110.0)
        )
        # Claim and finish: cursor advances to the through boundary.
        grant = await repo.claim_cycle(make_claim())
        assert grant is not None
        await repo.finish_cycle(make_finish(), [], now=200.0)
        covered = await repo.list_pending_chats()
        # A new non-self message arrives after the finish: pending again.
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m3", recv_ts=130.0)
        )
        pending = await repo.list_pending_chats()
        await repo.close()
        return covered, pending

    covered, pending = run(scenario())
    assert covered == []
    assert pending == [CK]

    # Restart: the same durable state answers identically.
    async def reopened():
        _db, repo = await open_repo(tmp_path / "t.db")
        result = await repo.list_pending_chats()
        await repo.close()
        return result

    assert run(reopened()) == [CK]


def test_list_pending_chats_self_only_chat_is_not_pending(tmp_path):
    """A chat whose only beyond-cursor rows are self messages has no
    inbound work: it must never be woken at startup."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            make_identity(),
            make_message(msg_id="self1", is_self=True, recv_ts=100.0),
        )
        result = await repo.list_pending_chats()
        await repo.close()
        return result

    assert run(scenario()) == []


# ── durable EWMA interval: atomic ingest updates ────────────────────────────

def test_ingest_updates_avg_interval_atomically(tmp_path):
    """Every newly inserted non-self message folds into the chat's durable
    avg_interval in the same transaction as the insert."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", recv_ts=100.0)
        )
        first = await repo.get_chat_state(CK)
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", recv_ts=130.0)
        )
        second = await repo.get_chat_state(CK)
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m3", recv_ts=160.0)
        )
        third = await repo.get_chat_state(CK)
        await repo.close()
        return first, second, third

    first, second, third = run(scenario())
    # First non-self message: no prior sample, average stays unseeded.
    assert first is not None and first.avg_interval is None
    # Second: seeds with the gap (130 - 100 = 30).
    assert second is not None and second.avg_interval == pytest.approx(30.0)
    # Third: EWMA over the seeded average (0.5*30 + 0.5*30 = 30).
    assert third is not None and third.avg_interval == pytest.approx(30.0)


def test_ingest_avg_interval_parity_with_pacing_reducer(tmp_path):
    """The durable average is EXACTLY what the dependency-neutral
    pacing.ewma_interval reducer produces over the same prior durable
    data — the repository and the runtime session layer share one
    reducer."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        samples = [(100.0, "m1"), (130.0, "m2"), (145.0, "m3"), (200.0, "m4")]
        for ts, mid in samples:
            await repo.ingest_message(
                make_identity(), make_message(msg_id=mid, recv_ts=ts)
            )
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    state = run(scenario())
    assert state is not None
    expected = None
    prev_ts = None
    for ts, _mid in [(100.0, "m1"), (130.0, "m2"), (145.0, "m3"), (200.0, "m4")]:
        expected = ewma_interval(expected, prev_ts, ts)
        prev_ts = ts
    assert state.avg_interval == pytest.approx(expected)


def test_ingest_avg_interval_survives_restart(tmp_path):
    """The durable average survives close/reopen, and the terminal-owned
    cursor/hold/idle fields are preserved untouched by ingest."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", recv_ts=100.0)
        )
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", recv_ts=130.0)
        )
        # Terminal-owned state, written by finish_cycle.
        grant = await repo.claim_cycle(make_claim())
        assert grant is not None
        await repo.finish_cycle(
            make_finish(hold_until=500.0, idle_streak_after=2), [], now=200.0
        )
        # A new non-self message after the finish: ingest updates the
        # average but must not touch cursor/hold/idle.
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m3", recv_ts=160.0)
        )
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    state = run(scenario())
    assert state is not None
    assert state.avg_interval == pytest.approx(30.0)
    # finish_cycle advanced the cursor to the claim's through boundary (2);
    # ingest never touches it.
    assert state.cursor_msg_id == MessageRowId(2)
    assert state.hold_until == 500.0
    assert state.idle_streak == 2

    # Restart: the same durable state answers identically.
    async def reopened():
        _db, repo = await open_repo(tmp_path / "t.db")
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    state2 = run(reopened())
    assert state2 is not None
    assert state2.avg_interval == pytest.approx(30.0)
    assert state2.cursor_msg_id == MessageRowId(2)
    assert state2.hold_until == 500.0
    assert state2.idle_streak == 2


def test_ingest_self_and_duplicate_do_not_change_avg_interval(tmp_path):
    """Self messages and duplicate (platform, self_id, platform_msg_id)
    rows never spuriously change the durable average."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", recv_ts=100.0)
        )
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", recv_ts=130.0)
        )
        before = await repo.get_chat_state(CK)
        # Self message between the samples: must not change the average.
        await repo.ingest_message(
            make_identity(),
            make_message(msg_id="self1", is_self=True, recv_ts=115.0),
        )
        # Duplicate of m2: must not change the average.
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", recv_ts=999.0)
        )
        after = await repo.get_chat_state(CK)
        await repo.close()
        return before, after

    before, after = run(scenario())
    assert before is not None and after is not None
    assert after.avg_interval == before.avg_interval == pytest.approx(30.0)


def test_ingest_avg_interval_ignores_invalid_time_samples(tmp_path):
    """A missing timestamp or a non-positive gap (clock skew,
    same-timestamp batch) carries no pacing information: the average is
    left untouched."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", recv_ts=100.0)
        )
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", recv_ts=130.0)
        )
        # Same-timestamp batch: gap 0.
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m3", recv_ts=130.0)
        )
        # Clock skew: negative gap.
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m4", recv_ts=120.0)
        )
        # Missing timestamp: no sample at all.
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m5", recv_ts=None)
        )
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    state = run(scenario())
    assert state is not None and state.avg_interval == pytest.approx(30.0)


# ── latest terminal end reason: the gate's only history input ───────────────

def test_latest_terminal_end_reason_none_without_cycles(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        reason = await repo.get_latest_terminal_end_reason(CK)
        await repo.close()
        return reason

    assert run(scenario()) is None


def test_latest_terminal_end_reason_returns_latest_finish(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim())
        await repo.finish_cycle(
            make_finish(end_reason="planner_no_tool_end"), [], now=200.0
        )
        first = await repo.get_latest_terminal_end_reason(CK)
        await repo.claim_cycle(make_claim(cycle_id="cy-2", started_ts=300.0))
        await repo.finish_cycle(
            make_finish(cycle_id="cy-2", end_reason="skip"), [], now=400.0
        )
        latest = await repo.get_latest_terminal_end_reason(CK)
        await repo.close()
        return first, latest

    first, latest = run(scenario())
    assert first == "planner_no_tool_end"
    assert latest == "skip"  # the LATEST terminal outcome


def test_latest_terminal_end_reason_ignores_released_and_expired_claims(tmp_path):
    """Releases and expired claims are not terminal outcomes: they never
    affect the latest terminal end reason."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.claim_cycle(make_claim())
        await repo.release_cycle(CK, "cy-1")
        after_release = await repo.get_latest_terminal_end_reason(CK)
        # An expired claim is recovered, never finished.
        await repo.claim_cycle(make_claim(cycle_id="cy-2", started_ts=50.0,
                                          expires_at=100.0))
        await repo.claim_cycle(make_claim(cycle_id="cy-3", started_ts=200.0))
        after_expiry = await repo.get_latest_terminal_end_reason(CK)
        await repo.close()
        return after_release, after_expiry

    after_release, after_expiry = run(scenario())
    assert after_release is None
    assert after_expiry is None  # still no terminal cycle


def test_latest_terminal_end_reason_is_per_chat(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key="qq:group:other"))
        await repo.claim_cycle(make_claim())
        await repo.finish_cycle(make_finish(end_reason="completed"), [], now=200.0)
        other = await repo.get_latest_terminal_end_reason(ChatKey("qq:group:other"))
        await repo.close()
        return other

    assert run(scenario()) is None  # the other chat has no terminal cycle


# ── trusted-key self-echo reconciliation ────────────────────────────────────

def _echo(msg_id="echo:1", text="hi", sender_id="bot-1", **kw) -> Message:
    """A real self echo matching the standard outbox item (text "hi")."""
    return make_message(
        msg_id=msg_id, text=text, sender_id=sender_id, is_self=True,
        recv_ts=150.0, **kw,
    )


def test_self_echo_reconciles_in_flight_with_trusted_key(tmp_path):
    """A verified self echo with the trusted delivery key atomically
    transitions exactly one in-flight row to sent with the real platform
    id/timestamp — and never creates a synthetic echo."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)  # ambiguous send: in_flight
        result = await repo.ingest_message(
            make_identity(), _echo(), self_echo_delivery_key="cy-1:0"
        )
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT state, sent_ts, platform_msg_id FROM outbox WHERE id = 1"
            ).fetchone()
        )
        count = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        )
        await repo.close()
        return result, row, count

    result, row, count = run(scenario())
    assert result.inserted is True
    assert result.echo_status == "reconciled"
    assert row == ("sent", 150.0, "echo:1")  # real platform id/timestamp
    assert count == 1  # the real echo only — NO synthetic echo


def test_duplicate_echo_reconciliation_is_idempotent(tmp_path):
    """The same echo event twice: the first reconciles, the second is
    already_reconciled — no second transition, no duplicate row."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        first = await repo.ingest_message(
            make_identity(), _echo(), self_echo_delivery_key="cy-1:0"
        )
        second = await repo.ingest_message(
            make_identity(), _echo(), self_echo_delivery_key="cy-1:0"
        )
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT state, platform_msg_id FROM outbox WHERE id = 1"
            ).fetchone()
        )
        count = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        )
        await repo.close()
        return first, second, row, count

    first, second, row, count = run(scenario())
    assert first.echo_status == "reconciled"
    assert second.echo_status == "already_reconciled"  # idempotent
    assert second.inserted is False
    assert row == ("sent", "echo:1")
    assert count == 1


def test_self_echo_without_key_is_unproven_and_never_moves_outbox(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        result = await repo.ingest_message(make_identity(), _echo())
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT state FROM outbox WHERE id = 1"
            ).fetchone()[0]
        )
        await repo.close()
        return result, row

    result, row = run(scenario())
    assert result.echo_status == "unproven"
    assert row == "in_flight"  # never transitioned


def test_echo_with_unknown_key_is_conflict(tmp_path):
    """A trusted key that matches no outbox row is a mismatch, not a proof."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        result = await repo.ingest_message(
            make_identity(), _echo(), self_echo_delivery_key="no-such-key"
        )
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT state FROM outbox WHERE id = 1"
            ).fetchone()[0]
        )
        await repo.close()
        return result, row

    result, row = run(scenario())
    assert result.echo_status == "conflict"
    assert row == "in_flight"


def test_echo_with_wrong_sender_is_conflict(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        result = await repo.ingest_message(
            make_identity(), _echo(sender_id="someone-else"),
            self_echo_delivery_key="cy-1:0",
        )
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT state FROM outbox WHERE id = 1"
            ).fetchone()[0]
        )
        await repo.close()
        return result, row

    result, row = run(scenario())
    assert result.echo_status == "conflict"  # sender != chat self_id
    assert row == "in_flight"


def test_echo_with_mismatched_text_is_conflict(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        result = await repo.ingest_message(
            make_identity(), _echo(text="different text"),
            self_echo_delivery_key="cy-1:0",
        )
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT state FROM outbox WHERE id = 1"
            ).fetchone()[0]
        )
        await repo.close()
        return result, row

    result, row = run(scenario())
    assert result.echo_status == "conflict"
    assert row == "in_flight"


def test_echo_with_mismatched_segments_is_conflict(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        result = await repo.ingest_message(
            make_identity(),
            _echo(segments=(Segment("face", {"id": 1}),)),
            self_echo_delivery_key="cy-1:0",
        )
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT state FROM outbox WHERE id = 1"
            ).fetchone()[0]
        )
        await repo.close()
        return result, row

    result, row = run(scenario())
    assert result.echo_status == "conflict"  # canonical segments differ
    assert row == "in_flight"


def test_echo_with_mismatched_reply_is_conflict(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0", reply_to="9")])
        await repo.attempt_outbox(1, 100.0)
        result = await repo.ingest_message(
            make_identity(), _echo(reply_to="10"),
            self_echo_delivery_key="cy-1:0",
        )
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT state FROM outbox WHERE id = 1"
            ).fetchone()[0]
        )
        await repo.close()
        return result, row

    result, row = run(scenario())
    assert result.echo_status == "conflict"
    assert row == "in_flight"


def test_echo_for_pending_row_is_conflict(tmp_path):
    """A pending row must go through attempt_outbox first; an echo can
    never transition it directly."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        result = await repo.ingest_message(
            make_identity(), _echo(), self_echo_delivery_key="cy-1:0"
        )
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT state FROM outbox WHERE id = 1"
            ).fetchone()[0]
        )
        await repo.close()
        return result, row

    result, row = run(scenario())
    assert result.echo_status == "conflict"  # wrong state
    assert row == "pending"


def test_echo_for_dropped_row_is_conflict(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.drop_outbox(1)
        result = await repo.ingest_message(
            make_identity(), _echo(), self_echo_delivery_key="cy-1:0"
        )
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT state FROM outbox WHERE id = 1"
            ).fetchone()[0]
        )
        await repo.close()
        return result, row

    result, row = run(scenario())
    assert result.echo_status == "conflict"  # terminal dropped row
    assert row == "dropped"


def test_echo_for_already_sent_row_is_already_reconciled(tmp_path):
    """A duplicate echo whose row was already reconciled (e.g. by the
    synthetic-echo path) is idempotent — never a second transition."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        await repo.mark_outbox_sent(1, "echo:1", 101.0)
        result = await repo.ingest_message(
            make_identity(), _echo(), self_echo_delivery_key="cy-1:0"
        )
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT state, platform_msg_id FROM outbox WHERE id = 1"
            ).fetchone()
        )
        await repo.close()
        return result, row

    result, row = run(scenario())
    assert result.echo_status == "already_reconciled"
    assert row == ("sent", "echo:1")  # untouched


def test_echo_for_sent_row_with_mismatch_is_conflict(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        await repo.mark_outbox_sent(1, "echo:1", 101.0)
        result = await repo.ingest_message(
            make_identity(), _echo(text="different"),
            self_echo_delivery_key="cy-1:0",
        )
        await repo.close()
        return result

    result = run(scenario())
    assert result.echo_status == "conflict"


def test_echo_without_platform_id_is_conflict(tmp_path):
    """No real platform id: the send cannot be proven to have landed."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        result = await repo.ingest_message(
            make_identity(), _echo(msg_id=None),
            self_echo_delivery_key="cy-1:0",
        )
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT state FROM outbox WHERE id = 1"
            ).fetchone()[0]
        )
        await repo.close()
        return result, row

    result, row = run(scenario())
    assert result.echo_status == "conflict"
    assert row == "in_flight"


def test_echo_reconciliation_never_creates_synthetic_echo(tmp_path):
    """The real echo insertion and the reconciliation share one
    transaction; no second synthetic echo is ever generated."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        await repo.ingest_message(
            make_identity(), _echo(), self_echo_delivery_key="cy-1:0"
        )
        rows = await repo._db.read(
            lambda c: c.execute(
                "SELECT platform_msg_id, is_self FROM messages"
            ).fetchall()
        )
        await repo.close()
        return rows

    rows = run(scenario())
    assert rows == [("echo:1", 1)]  # exactly the real echo, once


def test_conflict_still_commits_the_message(tmp_path):
    """A rejected reconciliation never moves the outbox, but the real echo
    message itself still commits (shared transaction, insert side)."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        result = await repo.ingest_message(
            make_identity(), _echo(text="different"),
            self_echo_delivery_key="cy-1:0",
        )
        msg = await repo.get_message(CK, "echo:1")
        await repo.close()
        return result, msg

    result, msg = run(scenario())
    assert result.echo_status == "conflict"
    assert msg is not None and msg.text == "different"  # committed


# ── fallback send → later real echo: synthetic-row reconciliation ───────────

def test_fallback_send_then_real_echo_reconciles_synthetic_row(tmp_path):
    """mark_outbox_sent with no platform id writes a synthetic ``local:``
    echo; a later real echo with the trusted key updates that durable row
    to the real platform id instead of inserting a duplicate context
    message."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        await repo.mark_outbox_sent(1, None, 101.0)  # fallback: local:1
        result = await repo.ingest_message(
            make_identity(), _echo(), self_echo_delivery_key="cy-1:0"
        )
        rows = await repo._db.read(
            lambda c: c.execute(
                "SELECT platform_msg_id, is_self, text FROM messages ORDER BY id"
            ).fetchall()
        )
        outbox = await repo._db.read(
            lambda c: c.execute(
                "SELECT state, platform_msg_id FROM outbox WHERE id = 1"
            ).fetchone()
        )
        commits = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM inbound_commits").fetchone()[0]
        )
        await repo.close()
        return result, rows, outbox, commits

    result, rows, outbox, commits = run(scenario())
    assert result.inserted is False  # no duplicate context message
    assert result.echo_status == "already_reconciled"
    assert rows == [("echo:1", 1, "hi")]  # synthetic row updated, singular
    assert outbox == ("sent", "echo:1")  # real platform id recorded
    assert commits == 0  # no scheduler wake / no new commit


def test_fallback_send_then_repeated_real_echo_is_idempotent(tmp_path):
    """After the first real echo reconciles the synthetic row, a repeated
    real echo is already_reconciled and never duplicates the context."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        await repo.mark_outbox_sent(1, None, 101.0)
        first = await repo.ingest_message(
            make_identity(), _echo(), self_echo_delivery_key="cy-1:0"
        )
        second = await repo.ingest_message(
            make_identity(), _echo(), self_echo_delivery_key="cy-1:0"
        )
        rows = await repo._db.read(
            lambda c: c.execute(
                "SELECT platform_msg_id FROM messages ORDER BY id"
            ).fetchall()
        )
        await repo.close()
        return first, second, rows

    first, second, rows = run(scenario())
    assert first.echo_status == "already_reconciled"
    assert second.echo_status == "already_reconciled"
    assert second.inserted is False
    assert rows == [("echo:1",)]  # still singular


def test_fallback_send_then_real_echo_different_payload_not_merged(tmp_path):
    """A real echo whose payload differs from the outbox item is NOT merged
    into the synthetic row: it commits as its own context message and the
    synthetic row is left untouched (fail-closed)."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        await repo.mark_outbox_sent(1, None, 101.0)
        result = await repo.ingest_message(
            make_identity(), _echo(text="different"),
            self_echo_delivery_key="cy-1:0",
        )
        rows = await repo._db.read(
            lambda c: c.execute(
                "SELECT platform_msg_id, text FROM messages ORDER BY id"
            ).fetchall()
        )
        await repo.close()
        return result, rows

    result, rows = run(scenario())
    assert result.echo_status == "conflict"  # not merged
    assert rows == [("local:1", "hi"), ("echo:1", "different")]  # both kept


def test_fallback_send_then_real_echo_without_key_is_unproven(tmp_path):
    """Without the trusted delivery key the real echo is unproven: the
    synthetic row is NOT reconciled and the real echo commits separately
    (never heuristically matched)."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        await repo.mark_outbox_sent(1, None, 101.0)
        result = await repo.ingest_message(make_identity(), _echo())
        rows = await repo._db.read(
            lambda c: c.execute(
                "SELECT platform_msg_id FROM messages ORDER BY id"
            ).fetchall()
        )
        await repo.close()
        return result, rows

    result, rows = run(scenario())
    assert result.echo_status == "unproven"
    assert rows == [("local:1",), ("echo:1",)]  # synthetic untouched, real added


def test_fallback_send_then_real_echo_wrong_sender_is_conflict(tmp_path):
    """A real echo from a non-self sender never reconciles the synthetic
    row (fail-closed)."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        await repo.mark_outbox_sent(1, None, 101.0)
        result = await repo.ingest_message(
            make_identity(), _echo(sender_id="someone-else"),
            self_echo_delivery_key="cy-1:0",
        )
        rows = await repo._db.read(
            lambda c: c.execute(
                "SELECT platform_msg_id FROM messages ORDER BY id"
            ).fetchall()
        )
        await repo.close()
        return result, rows

    result, rows = run(scenario())
    assert result.echo_status == "conflict"
    assert rows == [("local:1",), ("echo:1",)]  # not merged


def test_fallback_send_then_real_echo_without_platform_id_is_conflict(tmp_path):
    """A real echo with no platform id cannot reconcile the synthetic row:
    there is nothing to update it to (fail-closed — the synthetic row is
    left untouched)."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        await repo.mark_outbox_sent(1, None, 101.0)
        result = await repo.ingest_message(
            make_identity(), _echo(msg_id=None),
            self_echo_delivery_key="cy-1:0",
        )
        rows = await repo._db.read(
            lambda c: c.execute(
                "SELECT platform_msg_id FROM messages ORDER BY id"
            ).fetchall()
        )
        outbox = await repo._db.read(
            lambda c: c.execute(
                "SELECT platform_msg_id FROM outbox WHERE id = 1"
            ).fetchone()[0]
        )
        await repo.close()
        return result, rows, outbox

    result, rows, outbox = run(scenario())
    assert result.echo_status == "already_reconciled"  # sent-row semantics
    assert rows[0] == ("local:1",)  # synthetic row NOT updated
    assert outbox is None  # outbox still has no real platform id


def test_normal_platform_id_path_still_dedupes(tmp_path):
    """The normal path — mark_outbox_sent with a real platform id — still
    dedupes the later real echo on the UNIQUE constraint and never touches
    the synthetic-reconciliation branch."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        await repo.mark_outbox_sent(1, "echo:1", 101.0)  # real id
        result = await repo.ingest_message(
            make_identity(), _echo(), self_echo_delivery_key="cy-1:0"
        )
        rows = await repo._db.read(
            lambda c: c.execute(
                "SELECT platform_msg_id FROM messages ORDER BY id"
            ).fetchall()
        )
        await repo.close()
        return result, rows

    result, rows = run(scenario())
    assert result.inserted is False
    assert result.echo_status == "already_reconciled"
    assert rows == [("echo:1",)]  # singular, no local: row ever created


def test_fallback_marker_replay_self_context_remains_singular(tmp_path):
    """Marker/replay self context stays singular: after the fallback send
    and a real echo, exactly one self message row exists for the send."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("hi", idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        await repo.mark_outbox_sent(1, None, 101.0)
        await repo.ingest_message(
            make_identity(), _echo(), self_echo_delivery_key="cy-1:0"
        )
        # A replay of the same real echo (marker replay) is idempotent.
        replay = await repo.ingest_message(
            make_identity(), _echo(), self_echo_delivery_key="cy-1:0"
        )
        count = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        )
        await repo.close()
        return replay, count

    replay, count = run(scenario())
    assert replay.echo_status == "already_reconciled"
    assert count == 1  # singular self context


def test_idem_key_collision_with_identical_payload_hydrates(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("same", idem_key="k1")])
        # A later cycle reuses the key with an identical payload: hydrate.
        await finish_batch(repo, [item("same", idem_key="k1")], cycle_id="cy-2",
                           started_ts=300.0, now=400.0)
        count = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        await repo.close()
        return count

    assert run(scenario()) == 1  # no silent duplicate, no error


def test_idem_key_collision_with_conflicting_payload_rejects(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await finish_batch(repo, [item("original", idem_key="k1")])
        await repo.claim_cycle(make_claim(cycle_id="cy-2", started_ts=300.0))
        with pytest.raises(RepoError, match="collision"):
            await repo.finish_cycle(
                make_finish(cycle_id="cy-2"), [item("conflicting", idem_key="k1")],
                now=400.0,
            )
        count = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        states = await repo._db.read(
            lambda c: [r[0] for r in c.execute("SELECT state FROM claims ORDER BY id")]
        )
        await repo.close()
        return count, states

    count, states = run(scenario())
    assert count == 1  # the conflicting part was NOT silently lost/added
    assert states == ["finished", "live"]  # second finish aborted


# ── records / kv / stats ────────────────────────────────────────────────────

def test_add_record_returns_id(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        rec_id = await repo.add_record(
            Record(learner="expression", payload={"situation": "x"}, chat_key=CK)
        )
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT learner, payload_json, weight, uses FROM records WHERE id = ?",
                (rec_id,),
            ).fetchone()
        )
        await repo.close()
        return rec_id, row

    rec_id, row = run(scenario())
    assert rec_id == 1
    assert row == ("expression", '{"situation":"x"}', 1.0, 0)


def test_kv_roundtrip_and_upsert(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.set_kv("embed_dim", "1024")
        first = await repo.get_kv("embed_dim")
        await repo.set_kv("embed_dim", "2048")
        second = await repo.get_kv("embed_dim")
        missing = await repo.get_kv("nope")
        await repo.close()
        return first, second, missing

    assert run(scenario()) == ("1024", "2048", None)


def test_stats_counts(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await finish_batch(repo, [item("hi", idem_key="k1")])
        stats = await repo.stats()
        await repo.close()
        return stats

    stats = run(scenario())
    assert stats["messages"] == 1
    assert stats["outbox_pending"] == 1
    assert stats["cycles"] == 1
    assert stats["claims"] == 1
    assert stats["user_version"] == 15


# ── restart persistence ─────────────────────────────────────────────────────

def test_repository_data_survives_restart(tmp_path):
    path = tmp_path / "t.db"

    async def scenario():
        _db, repo = await open_repo_with_chat(path)
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.set_kv("k", "v")
        await repo.claim_cycle(make_claim())
        await repo.finish_cycle(make_finish(), [], now=200.0)
        await repo.close()
        _db2, repo2 = await open_repo_with_chat(path)
        msg = await repo2.get_message(CK, "m1")
        state = await repo2.get_chat_state(CK)
        kv = await repo2.get_kv("k")
        await repo2.close()
        return msg, state, kv

    msg, state, kv = run(scenario())
    assert msg is not None and msg.row_id == 1
    assert state.cursor_msg_id == 1
    assert kv == "v"


# ── embedding generation state persistence ───────────────────────────────────

def test_create_embedding_generation_persists_requested_state(tmp_path):
    """create_embedding_generation persists the requested ``state``: the
    default stays ``inactive`` (backward compatible), ``building`` is
    persisted literally in the DB, and invalid states are rejected."""
    async def scenario():
        _db, repo = await open_repo(tmp_path / "t.db")
        manual = await repo.create_embedding_generation("m1", 2)
        building = await repo.create_embedding_generation(
            "m2", 4, revision="r1", state="building"
        )
        rows = await repo._db.read(
            lambda c: c.execute(
                "SELECT space_id, state FROM embedding_generations ORDER BY id"
            ).fetchall()
        )
        try:
            await repo.create_embedding_generation("m3", 8, state="active")
            invalid = False
        except RepoError:
            invalid = True
        await repo.close()
        return manual, building, rows, invalid

    manual, building, rows, invalid = run(scenario())
    assert manual.state == "inactive"  # backward-compatible default
    assert building.state == "building"  # requested building persisted
    assert dict(rows)["m1@default"] == "inactive"
    assert dict(rows)["m2@r1"] == "building"  # literally building in the DB
    assert invalid is True  # 'active' is not settable at create


# ── Gate 5 remediation: keyset-paged enumeration + source-fenced activation ──

async def _seed_memory(repo, chat_key, text, prefix="m"):
    """Insert one message and CAS-commit one memory; returns
    ``(memory_id, source_hash)``."""
    await repo.ingest_message(
        None,
        make_message(
            chat_key=chat_key, msg_id=f"{prefix}{text}", text=f"src {text}",
            recv_ts=1_700_000_000.0,
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
    mid = await repo._db.read(
        lambda c: c.execute(
            "SELECT id FROM memories WHERE chat_key = ? ORDER BY id DESC LIMIT 1",
            (chat_key,),
        ).fetchone()[0]
    )
    return mid, batch.source_hash


def test_list_memory_chats_after_keyset_pages(tmp_path):
    """list_memory_chats_after returns bounded deterministic keyset pages
    (``chat_key > after``), covering every chat across pages."""
    async def scenario():
        _db, repo = await open_repo(tmp_path / "t.db")
        chats = [CK, OTHER, ChatKey("qq:group:zzz")]
        for chat in chats:
            await repo.upsert_chat(make_identity(chat_key=str(chat)))
            await _seed_memory(repo, chat, f"text {chat}")
        page1 = await repo.list_memory_chats_after("", limit=2)
        page2 = await repo.list_memory_chats_after(page1[-1], limit=2)
        page3 = await repo.list_memory_chats_after(page2[-1], limit=2)
        try:
            await repo.list_memory_chats_after("", limit=0)
            invalid = False
        except RepoError:
            invalid = True
        await repo.close()
        return page1, page2, page3, invalid

    page1, page2, page3, invalid = run(scenario())
    assert len(page1) == 2 and len(page2) == 1 and page3 == []
    all_chats = page1 + page2
    assert all_chats == sorted(all_chats)  # deterministic keyset order
    assert set(all_chats) == {CK, OTHER, ChatKey("qq:group:zzz")}
    assert invalid is True  # non-positive limit rejected


def test_list_vectors_for_memories_is_chat_scoped_and_bounded(tmp_path):
    """list_vectors_for_memories returns only the chat's vectors for the
    requested memory ids — cross-chat ids are absent, empty input is empty."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=str(OTHER)))
        id_ck, hash_ck = await _seed_memory(repo, CK, "apple")
        id_other, hash_other = await _seed_memory(repo, OTHER, "secret")
        gen = await repo.create_embedding_generation("m1", 2, revision="r1")
        assert gen.id is not None
        await repo.upsert_vector(
            CK, make_vector(owner_id=id_ck, generation=gen.id, model="m1", dim=2,
                            source_hash=hash_ck)
        )
        await repo.upsert_vector(
            OTHER, make_vector(owner_id=id_other, generation=gen.id, model="m1",
                               dim=2, source_hash=hash_other)
        )
        ck_rows = await repo.list_vectors_for_memories(
            CK, "m1", gen.id, [id_ck, id_other]
        )
        other_rows = await repo.list_vectors_for_memories(
            OTHER, "m1", gen.id, [id_ck, id_other]
        )
        empty = await repo.list_vectors_for_memories(CK, "m1", gen.id, [])
        await repo.close()
        return id_ck, id_other, ck_rows, other_rows, empty

    id_ck, id_other, ck_rows, other_rows, empty = run(scenario())
    assert [r.owner_id for r in ck_rows] == [id_ck]  # cross-chat id absent
    assert [r.owner_id for r in other_rows] == [id_other]
    assert empty == []


def test_activate_embedding_generation_if_complete_source_fenced(tmp_path):
    """The source-fenced activation activates ONLY when every nonempty
    memory has a matching vector; a committed memory makes it fail with the
    deterministic repair set; the previous active generation is preserved
    until the building generation completes."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=str(OTHER)))
        id1, hash1 = await _seed_memory(repo, CK, "apple")
        id2, hash2 = await _seed_memory(repo, OTHER, "banana")
        # Old ACTIVE generation G1 (preserved until the build completes).
        g1 = await repo.create_embedding_generation("m1", 2, revision="r1")
        assert g1.id is not None
        await repo.activate_embedding_generation(g1.id)
        # Building generation G2 with a vector for CK only.
        g2 = await repo.create_embedding_generation(
            "m1", 2, revision="r2", state="building"
        )
        assert g2.id is not None
        await repo.upsert_vector(
            CK, make_vector(owner_id=id1, generation=g2.id, model="m1", dim=2,
                            source_hash=hash1)
        )
        # Incomplete: OTHER's memory has no vector -> repair set, stays
        # building, and the old active generation is preserved.
        repair = await repo.activate_embedding_generation_if_complete(g2.id)
        g2_after = await repo.get_embedding_generation(g2.id)
        g1_after = await repo.get_embedding_generation(g1.id)
        # Complete the coverage: write OTHER's vector, then activation
        # succeeds and deactivates the old active generation.
        await repo.upsert_vector(
            OTHER, make_vector(owner_id=id2, generation=g2.id, model="m1", dim=2,
                               source_hash=hash2)
        )
        activated = await repo.activate_embedding_generation_if_complete(g2.id)
        g2_final = await repo.get_embedding_generation(g2.id)
        g1_final = await repo.get_embedding_generation(g1.id)
        await repo.close()
        return repair, g2_after, g1_after, activated, g2_final, g1_final

    repair, g2_after, g1_after, activated, g2_final, g1_final = run(scenario())
    assert repair == [OTHER]  # deterministic repair set
    assert g2_after.state == "building"  # never activated while incomplete
    assert g1_after.state == "active"  # old generation preserved
    assert activated is None  # activated once coverage is complete
    assert g2_final.state == "active"
    assert g1_final.state == "inactive"  # at-most-one-active preserved


def test_activate_embedding_generation_if_complete_fails_closed(tmp_path):
    """Missing or non-building generations fail closed ([]); a stale
    source_hash is not a valid match."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        id1, _hash1 = await _seed_memory(repo, CK, "apple")
        missing = await repo.activate_embedding_generation_if_complete(999)
        inactive = await repo.create_embedding_generation("m1", 2, revision="r1")
        assert inactive.id is not None
        not_building = await repo.activate_embedding_generation_if_complete(
            inactive.id
        )
        # A building generation with a STALE source_hash vector: not a match.
        g = await repo.create_embedding_generation(
            "m1", 2, revision="r2", state="building"
        )
        assert g.id is not None
        await repo.upsert_vector(
            CK, make_vector(owner_id=id1, generation=g.id, model="m1", dim=2,
                            source_hash="stale")
        )
        stale = await repo.activate_embedding_generation_if_complete(g.id)
        g_after = await repo.get_embedding_generation(g.id)
        await repo.close()
        return missing, not_building, stale, g_after

    missing, not_building, stale, g_after = run(scenario())
    assert missing == []
    assert not_building == []
    assert stale == [CK]  # stale source_hash is not a valid match
    assert g_after.state == "building"


def test_activate_embedding_generation_if_complete_null_source_hash_matches(tmp_path):
    """A legacy memory with a NULL source_hash matches a vector with a NULL
    source_hash (NULL-safe comparison), so legacy rows are not stuck."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        # A legacy memory row with NULL source_hash (pre-Gate-5 row).
        await repo._db.write(
            lambda c: c.execute(
                "INSERT INTO memories(chat_key, kind, text, cues_json, strength,"
                " created_ts, last_hit_ts, source_first_msg_id, source_last_msg_id,"
                " source_hash) VALUES (?, 'fact', 'legacy', '[]', 1.0, 0.0, 0.0,"
                " NULL, NULL, NULL)",
                (CK,),
            )
        )
        mid = await repo._db.read(
            lambda c: c.execute(
                "SELECT id FROM memories WHERE chat_key = ? ORDER BY id DESC LIMIT 1",
                (CK,),
            ).fetchone()[0]
        )
        g = await repo.create_embedding_generation(
            "m1", 2, revision="r1", state="building"
        )
        assert g.id is not None
        # A vector with NULL source_hash matches the NULL-source_hash memory.
        await repo.upsert_vector(
            CK, make_vector(owner_id=mid, generation=g.id, model="m1", dim=2,
                            source_hash=None)
        )
        activated = await repo.activate_embedding_generation_if_complete(g.id)
        g_after = await repo.get_embedding_generation(g.id)
        await repo.close()
        return activated, g_after

    activated, g_after = run(scenario())
    assert activated is None  # NULL-safe match: activated
    assert g_after.state == "active"


# ── Phase 6 vector ownership widening (records + memories) ──────────────────

async def _seed_adaptive_record(repo, text: str = "fact") -> int:
    """Commit one adaptive record via the learner CAS path; returns its id."""
    from pretender.types import LearnerDraft, LearnerRunRequest

    grant = await repo.acquire_learner_run(
        LearnerRunRequest(chat_key=CK, learner="personality",
                          started_ts=100.0, expires_at=500.0, now=100.0)
    )
    assert grant is not None
    batch = await repo.read_learner_source_batch(
        CK, "personality", through_msg_id=grant.through_msg_id, tail=100
    )
    assert batch is not None
    await repo.commit_learner_source(
        LearnerDraft(
            chat_key=CK, learner="personality", batch=batch,
            records=(Record(learner="personality", payload={"text": text},
                            chat_key=CK),),
        ),
        now=200.0,
    )
    records = await repo.list_learner_records(CK, "personality")
    assert records
    rid = records[0].id
    assert rid is not None
    return rid


def _vector(owner_table: str, owner_id: int, generation: int, **kw) -> VectorRow:
    """A VectorRow with a configurable owner table (make_vector hardcodes
    ``memories``)."""
    from pretender.types import VectorRow

    return VectorRow(
        owner_table=owner_table,
        owner_id=owner_id,
        dim=kw.pop("dim", 2),
        model=kw.pop("model", "m1"),
        generation=generation,
        blob=kw.pop("blob", struct.pack("<2f", 0.1, 0.2)),
        **kw,
    )


def test_record_vector_upsert_get_delete_roundtrip(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            None, make_message(chat_key=CK, msg_id="a1", text="s1",
                               recv_ts=1_700_000_000.0 + 1)
        )
        rid = await _seed_adaptive_record(repo, text="likes tea")
        g = await repo.create_embedding_generation("m1", 2)
        rec = await repo.list_learner_records(CK, "personality")
        assert rec[0].content_hash is not None
        row = _vector("records", rid, g.id,
                          source_hash=rec[0].content_hash)
        await repo.upsert_vector(CK, row)
        got = await repo.get_vector(CK, "records", rid, "m1", g.id)
        assert got is not None
        assert got.owner_table == "records"
        assert got.source_hash == rec[0].content_hash
        assert await repo.delete_vector(CK, "records", rid, "m1", g.id) is True
        assert await repo.get_vector(CK, "records", rid, "m1", g.id) is None
        await repo.close()

    run(scenario())


def test_record_vector_requires_matching_content_hash(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            None, make_message(chat_key=CK, msg_id="a1", text="s1",
                               recv_ts=1_700_000_000.0 + 1)
        )
        rid = await _seed_adaptive_record(repo, text="x")
        g = await repo.create_embedding_generation("m1", 2)
        # A mismatched source_hash is rejected (content_hash source identity).
        with pytest.raises(RepoError, match="content_hash"):
            await repo.upsert_vector(
                CK, _vector("records", rid, g.id, source_hash="deadbeef")
            )
        # A legacy record (no content_hash) cannot own a vector at all.
        legacy_id = await repo.add_record(
            Record(learner="personality", payload={"text": "legacy"}, chat_key=CK)
        )
        with pytest.raises(RepoError, match="content_hash"):
            await repo.upsert_vector(
                CK, _vector("records", legacy_id, g.id, source_hash="h")
            )
        await repo.close()

    run(scenario())


def test_record_and_memory_same_numeric_vector_ids_coexist(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            None, make_message(chat_key=CK, msg_id="a1", text="s1",
                               recv_ts=1_700_000_000.0 + 1)
        )
        # A memory with id 1 and a record with id 1 (both numeric id 1).
        await repo.ingest_message(
            None, make_message(chat_key=CK, msg_id="a2", text="s2",
                               recv_ts=1_700_000_000.0 + 2)
        )
        batch = await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(2), tail=100
        )
        assert batch is not None
        await repo.commit_memory_source(
            MemoryWriteRequest(
                chat_key=CK, batch=batch,
                records=(MemoryRecord(
                    chat_key=CK, text="memory one",
                    source_first_msg_id=batch.first_msg_id,
                    source_last_msg_id=batch.last_msg_id,
                    source_hash=batch.source_hash,
                ),),
            )
        )
        rid = await _seed_adaptive_record(repo, text="record one")
        assert rid == 1  # the record shares the numeric id with the memory
        g = await repo.create_embedding_generation("m1", 2)
        rec = await repo.list_learner_records(CK, "personality")
        await repo.upsert_vector(
            CK, make_vector(owner_id=1, generation=g.id, source_hash="mh")
        )
        await repo.upsert_vector(
            CK, _vector("records", 1, g.id,
                            source_hash=rec[0].content_hash)
        )
        # Lookup identity includes owner_table: the two rows coexist and are
        # individually addressable.
        mv = await repo.get_vector(CK, "memories", 1, "m1", g.id)
        rv = await repo.get_vector(CK, "records", 1, "m1", g.id)
        assert mv is not None and rv is not None
        assert mv.owner_table == "memories"
        assert rv.owner_table == "records"
        # The memory index (list_vectors) stays memory-only: the record
        # vector never leaks into the memory search index.
        listed = await repo.list_vectors(CK, "m1", g.id)
        assert [r.owner_table for r in listed] == ["memories"]
        await repo.close()

    run(scenario())


def test_record_vector_does_not_affect_memory_generation_activation(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            None, make_message(chat_key=CK, msg_id="a1", text="s1",
                               recv_ts=1_700_000_000.0 + 1)
        )
        # A memory source exists (so coverage is required)...
        batch = await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(1), tail=100
        )
        assert batch is not None
        await repo.commit_memory_source(
            MemoryWriteRequest(
                chat_key=CK, batch=batch,
                records=(MemoryRecord(
                    chat_key=CK, text="memory",
                    source_first_msg_id=batch.first_msg_id,
                    source_last_msg_id=batch.last_msg_id,
                    source_hash=batch.source_hash,
                ),),
            )
        )
        # ...and a record vector exists for the building generation.
        rid = await _seed_adaptive_record(repo, text="record")
        g = await repo.create_embedding_generation(
            "m1", 2, revision="r1", state="building"
        )
        rec = await repo.list_learner_records(CK, "personality")
        await repo.upsert_vector(
            CK, _vector("records", rid, g.id,
                            model="m1", dim=2, source_hash=rec[0].content_hash)
        )
        # The record vector does NOT count as memory coverage: activation
        # returns the repair set (the memory is still missing its vector).
        repair = await repo.activate_embedding_generation_if_complete(g.id)
        g_after = await repo.get_embedding_generation(g.id)
        await repo.close()
        return repair, g_after

    repair, g_after = run(scenario())
    assert repair == [CK]  # the memory is missing a matching vector
    assert g_after.state == "building"  # never activated by a record vector


# ── Phase 6 P6.5 media catalog (MediaRepository surface on the repo) ─────────

def test_repo_implements_media_catalog_surface(tmp_path):
    """SqliteRepository is the ONE concrete implementation of the media
    catalog seam: submit -> approve -> select -> use round-trips through the
    same writer/reader lanes as every other durable surface."""
    from pretender.types import MediaAssetCandidate, MediaKind

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        cid = await repo.submit_media_candidate(
            MediaAssetCandidate(
                chat_key=CK, kind=MediaKind.STICKER,
                cache_key="c" * 64, sha256="a" * 64, mime="image/gif",
            ),
            now=200.0,
        )
        asset = await repo.approve_media_candidate(CK, cid, capacity=4, now=300.0)
        assert asset is not None and asset.safety_status == "approved"
        selected = await repo.select_media_assets(
            CK, MediaKind.STICKER, limit=5, cooldown_s=0.0, now=400.0
        )
        assert [a.id for a in selected] == [asset.id]
        assert await repo.use_media_asset(CK, asset.id, now=500.0) is True
        stats = await repo.stats()
        await repo.close()
        return stats

    stats = run(scenario())
    assert stats["media_assets"] == 1
