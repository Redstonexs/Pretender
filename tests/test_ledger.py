"""Durable dispatch ledger (frozen Oracle advisory): typed boundary types,
ingest event/commit/wake data, atomic begin/settle semantics, at-least-once
marker export, crash-point recovery, and the preserved trusted-echo/average
regressions. The legacy claim_cycle/finish_cycle surface stays untouched —
the next integration lane switches live use to this ledger."""

from __future__ import annotations

import json

import pytest

from pretender.clock import VirtualClock
from pretender.config import Config
from pretender.cycle import CycleRunner
from pretender.gate import Gate
from pretender.ingest import Ingest
from pretender.record import Recorder, export_marker, export_unexported, read_markers
from pretender.repo import SqliteRepository
from pretender.types import (
    AdapterEvent,
    ChatKey,
    ChatState,
    ClaimBusy,
    ClaimGrant,
    CommitSeq,
    CorpusMarker,
    CycleId,
    DispatchCause,
    DispatchDeferred,
    DispatchGrant,
    DispatchId,
    DispatchRequest,
    DispatchSettle,
    EchoStatus,
    EventId,
    IngestResult,
    MessageRowId,
    OutboxItem,
    WakeKind,
)
from tests.durable_helpers import (
    CK,
    FakeRepo,
    make_identity,
    make_message,
    open_repo,
    open_repo_with_chat,
    run,
)


def make_request(
    chat_key: str = "qq:group:123456",
    cause: str = DispatchCause.INBOUND,
    cycle_id: str = "cy-1",
    started_ts: float = 100.0,
    expires_at: float = 500.0,
    now: float = 100.0,
    **kw,
) -> DispatchRequest:
    return DispatchRequest(
        chat_key=ChatKey(chat_key),
        cause=cause,
        cycle_id=CycleId(cycle_id),
        started_ts=started_ts,
        expires_at=expires_at,
        now=now,
        **kw,
    )


def make_settle(
    dispatch_id: int,
    outcome: str = "finish",
    cycle_id: str = "cy-1",
    chat_key: str = "qq:group:123456",
    end_reason: str | None = "completed",
    **kw,
) -> DispatchSettle:
    return DispatchSettle(
        chat_key=ChatKey(chat_key),
        dispatch_id=DispatchId(dispatch_id),
        cycle_id=CycleId(cycle_id),
        outcome=outcome,
        end_reason=end_reason,
        **kw,
    )


def item(text="hi", idem_key="k1", chat_key=CK) -> OutboxItem:
    return OutboxItem(chat_key=chat_key, text=text, idem_key=idem_key)


def read_lines(path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def test_ingest_persists_priority_and_commit_message_identity(tmp_path):
    """Direct/quote priority and the real messages.id are frozen with the
    inbound commit, so Scheduler and replay never infer either later."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        result = await repo.ingest_message(
            make_identity(),
            make_message(msg_id="m1", mentions=("bot",)),
            structural_priority=True,
            pending_threshold=8,
            event_id=EventId("priority-1"),
        )
        row = await db.read(
            lambda c: c.execute(
                "SELECT message_id, priority FROM inbound_commits WHERE id = ?",
                (result.commit_seq,),
            ).fetchone()
        )
        marker = (await repo.list_unexported_commits())[0]
        await repo.close()
        return result, row, marker

    result, row, marker = run(scenario())
    assert result.priority is True
    assert row == (result.row_id, 1)
    assert marker.message_row_id == result.row_id
    assert marker.priority is True


def test_ledger_pending_chats_includes_live_prepared_dispatch(tmp_path):
    """Startup must notify a chat whose only work is attached to a live
    prepared dispatch, so ClaimBusy can re-arm at its lease horizon."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        pending = await repo.list_ledger_pending_chats()
        await repo.close()
        return pending

    assert run(scenario()) == [CK]


def test_dispatch_boundary_includes_trailing_self_commit(tmp_path):
    """A self echo never becomes pending, but it still belongs inside the
    frozen message boundary so presence/context sees it consistently."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        user = await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", recv_ts=100.0)
        )
        self_echo = await repo.ingest_message(
            make_identity(),
            make_message(msg_id="m2", is_self=True, sender_id="bot", recv_ts=101.0),
        )
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        await repo.close()
        return user, self_echo, grant

    user, self_echo, grant = run(scenario())
    assert len(grant.pending) == 1
    assert grant.pending[0].id == "m1"
    assert grant.pending[0].row_id == user.row_id
    assert grant.through_msg_id == self_echo.row_id
    assert grant.attached == (CommitSeq(1),)


def test_settled_dispatch_marker_carries_frozen_snapshot_input(tmp_path):
    """The v5 marker exports the evaluation timestamp/input rather than
    asking replay to reconstruct focus/hold/config from later state."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        await repo.settle_dispatch(
            make_settle(
                grant.dispatch_id,
                trace_json='{"trace":true}',
                evaluated_ts=111.0,
                snapshot_json='{"frozen":true}',
            ),
            [],
            now=120.0,
        )
        marker = (await repo.list_unexported_dispatches())[0]
        await repo.close()
        return marker

    marker = run(scenario())
    assert marker.evaluated_ts == 111.0
    assert marker.snapshot_json == '{"frozen":true}'


# ── ingest: event/commit/wake data and record-before-commit ─────────────────

def test_ingest_generates_event_id_before_recording(tmp_path):
    """The stable EventId is generated BEFORE the recorder write: the
    corpus event line carries it, and the repository receives the SAME id
    for the commit metadata."""

    async def scenario():
        fake = FakeRepo()
        fake.ingest_result = IngestResult(
            row_id=MessageRowId(1), inserted=True, pending_count=1,
            event_id=EventId("ev-1"), commit_seq=CommitSeq(1),
            wake_kind=WakeKind.INBOUND,
        )
        recorder = Recorder(tmp_path / "events.jsonl")
        ingest = Ingest(
            fake, recorder, identity=lambda ck: make_identity(),
        )
        result = await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=1.0)
        )
        recorder.close()
        return result, fake.calls, read_lines(tmp_path / "events.jsonl")

    result, calls, lines = run(scenario())
    assert result.event_id == EventId("ev-1")
    assert result.commit_seq == CommitSeq(1)
    assert result.wake_kind == WakeKind.INBOUND
    # The event line was written BEFORE the repo call and carries the SAME
    # stable id that reached the repository.
    assert lines[0]["type"] == "message"
    assert calls[0][0] == "ingest_message"
    assert lines[0]["event_id"] == calls[0][4]


def test_event_recorded_before_db_commit(tmp_path):
    """The recorder write happens BEFORE the durable commit: the repo fake
    observes the flushed event line while ingest_message is still running."""

    async def scenario():
        seen: list[dict] = []

        class _Probe(FakeRepo):
            async def ingest_message(
                self,
                identity,
                msg,
                *,
                self_echo_delivery_key=None,
                event_id=None,
                structural_priority=False,
                pending_threshold=None,
            ):
                seen.append(read_lines(tmp_path / "events.jsonl")[0])
                return IngestResult(
                    row_id=MessageRowId(1), inserted=True, pending_count=1,
                    event_id=event_id, commit_seq=CommitSeq(1),
                    wake_kind=WakeKind.INBOUND,
                )

        ingest = Ingest(
            _Probe(), Recorder(tmp_path / "events.jsonl"),
            identity=lambda ck: make_identity(),
        )
        await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=1.0)
        )
        return seen

    seen = run(scenario())
    assert len(seen) == 1
    assert seen[0]["type"] == "message"
    assert seen[0]["text"] == "hello"


def test_ingest_exports_commit_marker_after_commit(tmp_path):
    """The commit marker is appended AFTER the durable commit and marked
    exported: the corpus holds the event line then the marker line, and
    the marker carries the event/commit/wake data."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        recorder = Recorder(tmp_path / "events.jsonl")
        ingest = Ingest(repo, recorder, identity=lambda ck: make_identity())
        result = await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=1.0)
        )
        recorder.close()
        markers = read_markers(tmp_path / "events.jsonl")
        unexported = await repo.list_unexported_commits()
        await repo.close()
        return result, markers, unexported

    result, markers, unexported = run(scenario())
    assert result.commit_seq == CommitSeq(1)
    assert result.wake_kind == WakeKind.INBOUND
    assert len(markers) == 1
    assert markers[0].record_type == "commit"
    assert markers[0].sequence == 1
    assert markers[0].chat_key == CK
    assert markers[0].event_id == result.event_id
    assert markers[0].wake_kind == WakeKind.INBOUND
    assert unexported == []  # marked exported right after the append


def test_ingest_duplicate_commits_no_row_and_no_marker(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        recorder = Recorder(tmp_path / "events.jsonl")
        ingest = Ingest(repo, recorder, identity=lambda ck: make_identity())
        event = AdapterEvent(type="message", payload=make_message(), ts=1.0)
        first = await ingest.handle(event)
        second = await ingest.handle(event)
        recorder.close()
        markers = read_markers(tmp_path / "events.jsonl")
        unassigned = await repo.list_unassigned_commits(CK)
        await repo.close()
        return first, second, markers, unassigned

    first, second, markers, unassigned = run(scenario())
    assert first.inserted is True and first.commit_seq == CommitSeq(1)
    assert second.inserted is False and second.commit_seq is None
    assert second.wake_kind is None
    assert len(markers) == 1  # one commit, one marker
    assert unassigned == [CommitSeq(1)]


def test_ingest_self_commit_uses_wake_kind_none(tmp_path):
    """A committed self echo is ledger-complete (commit row + marker) but
    wake_kind none: it is never attached to a dispatch."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        recorder = Recorder(tmp_path / "events.jsonl")
        ingest = Ingest(repo, recorder, identity=lambda ck: make_identity())
        result = await ingest.handle(
            AdapterEvent(
                type="message",
                payload=make_message(msg_id="echo:1", is_self=True),
                ts=1.0,
            )
        )
        recorder.close()
        markers = read_markers(tmp_path / "events.jsonl")
        grant = await repo.begin_dispatch(make_request())
        await repo.close()
        return result, markers, grant

    result, markers, grant = run(scenario())
    assert result.inserted is True
    assert result.wake_kind == WakeKind.NONE
    assert result.commit_seq == CommitSeq(1)
    assert len(markers) == 1 and markers[0].wake_kind == WakeKind.NONE
    # The self commit is not eligible: an inbound dispatch finds no work.
    assert grant is None


# ── begin_dispatch: writer order, boundary, priority OR, busy/recovery ──────

def test_commit_first_joins_dispatch(tmp_path):
    """A commit that writes first joins the dispatch: the grant attaches
    it and carries its message as pending."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        result = await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", text="first")
        )
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        await repo.close()
        return result, grant

    result, grant = run(scenario())
    assert result.commit_seq == CommitSeq(1)
    assert isinstance(grant, DispatchGrant)
    assert grant.attached == (CommitSeq(1),)
    assert len(grant.pending) == 1
    assert grant.pending[0].text == "first"
    assert grant.pending[0].row_id == MessageRowId(1)
    assert grant.start_msg_id == MessageRowId(0)
    assert grant.through_msg_id == MessageRowId(1)


def test_dispatch_first_excludes_later_commit(tmp_path):
    """A timer dispatch that writes first excludes the later commit: the
    grant attaches nothing, and the commit stays unassigned for the next
    dispatch (durable writer order resolves the tie)."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        grant = await repo.begin_dispatch(
            make_request(cause=DispatchCause.TIMER, scheduled_ts=200.0)
        )
        result = await repo.ingest_message(
            make_identity(), make_message(msg_id="m1")
        )
        unassigned = await repo.list_unassigned_commits(CK)
        await repo.close()
        return grant, result, unassigned

    grant, result, unassigned = run(scenario())
    assert isinstance(grant, DispatchGrant)
    assert grant.attached == ()  # the timer wrote first: nothing attached
    assert result.commit_seq == CommitSeq(1)
    assert unassigned == [CommitSeq(1)]  # still pending for the next dispatch


def test_boundary_attachment_before_vs_after(tmp_path):
    """Commits within the frozen boundary attach; a commit after the
    dispatch began stays unassigned."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.ingest_message(make_identity(), make_message(msg_id="m2"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        await repo.ingest_message(make_identity(), make_message(msg_id="m3"))
        unassigned = await repo.list_unassigned_commits(CK)
        await repo.close()
        return grant, unassigned

    grant, unassigned = run(scenario())
    assert isinstance(grant, DispatchGrant)
    assert grant.attached == (CommitSeq(1), CommitSeq(2))
    assert [m.row_id for m in grant.pending] == [MessageRowId(1), MessageRowId(2)]
    assert unassigned == [CommitSeq(3)]  # arrived after the frozen boundary


def test_begin_dispatch_freezes_and_stores_boundary(tmp_path):
    """begin_dispatch freezes the max inbound commit sequence and the
    scheduled time in the SAME transaction that creates the prepared
    dispatch: the grant carries the exact boundary and scheduled_for."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.ingest_message(make_identity(), make_message(msg_id="m2"))
        grant = await repo.begin_dispatch(
            make_request(cause=DispatchCause.TIMER, scheduled_ts=200.0)
        )
        assert isinstance(grant, DispatchGrant)
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT commit_boundary, scheduled_ts FROM dispatches WHERE id = ?",
                (grant.dispatch_id,),
            ).fetchone()
        )
        await repo.close()
        return grant, row

    grant, row = run(scenario())
    assert isinstance(grant, DispatchGrant)
    assert grant.attached == (CommitSeq(1), CommitSeq(2))
    assert grant.commit_boundary == CommitSeq(2)  # the frozen max commit seq
    assert grant.scheduled_for == 200.0
    # The boundary and scheduled time are stored on the durable row.
    assert row == (2, 200.0)


def test_begin_dispatch_persists_attached_membership_and_grant_metadata(tmp_path):
    """begin_dispatch persists the exact attached CommitSeq tuple
    (attached_json) in the SAME transaction that creates the prepared
    dispatch, and the grant carries the cause and claimed_ts — the frozen
    membership a later released/detached dispatch stays replayable with."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.ingest_message(make_identity(), make_message(msg_id="m2"))
        grant = await repo.begin_dispatch(
            make_request(cause=DispatchCause.TIMER, scheduled_ts=200.0, now=100.0)
        )
        assert isinstance(grant, DispatchGrant)
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT attached_json, claimed_ts, cause FROM dispatches WHERE id = ?",
                (grant.dispatch_id,),
            ).fetchone()
        )
        await repo.close()
        return grant, row

    grant, row = run(scenario())
    assert isinstance(grant, DispatchGrant)
    assert grant.attached == (CommitSeq(1), CommitSeq(2))
    assert grant.cause == DispatchCause.TIMER
    assert grant.claimed_ts == 100.0
    # The exact membership is frozen on the durable row in the same
    # transaction as the prepared dispatch.
    assert row == ("[1,2]", 100.0, "timer")


def test_begin_dispatch_boundary_zero_when_no_commits(tmp_path):
    """A priority wake with no prior commits freezes boundary 0 and no
    scheduled time — the wake itself is the work."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        grant = await repo.begin_dispatch(make_request(cause=DispatchCause.STARTUP))
        await repo.close()
        return grant

    grant = run(scenario())
    assert isinstance(grant, DispatchGrant)
    assert grant.attached == ()
    assert grant.commit_boundary == CommitSeq(0)
    assert grant.scheduled_for is None


def test_priority_wake_creates_dispatch_with_zero_commits(tmp_path):
    """Priority OR: timer/startup/busy_recovery wakes always create a
    dispatch even with no eligible commits — the wake itself is the work."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        timer = await repo.begin_dispatch(
            make_request(cause=DispatchCause.TIMER, scheduled_ts=200.0)
        )
        assert isinstance(timer, DispatchGrant)
        await repo.settle_dispatch(
            make_settle(timer.dispatch_id, outcome="release"), [], now=100.0
        )
        startup = await repo.begin_dispatch(
            make_request(cause=DispatchCause.STARTUP, cycle_id="cy-2")
        )
        assert isinstance(startup, DispatchGrant)
        await repo.settle_dispatch(
            make_settle(startup.dispatch_id, outcome="release", cycle_id="cy-2"),
            [], now=100.0,
        )
        busy = await repo.begin_dispatch(
            make_request(cause=DispatchCause.BUSY_RECOVERY, cycle_id="cy-3")
        )
        await repo.close()
        return timer, startup, busy

    timer, startup, busy = run(scenario())
    for grant in (timer, startup, busy):
        assert isinstance(grant, DispatchGrant)
        assert grant.attached == ()
        assert grant.pending == ()


def test_inbound_dispatch_with_no_work_returns_none(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        grant = await repo.begin_dispatch(make_request())
        await repo.close()
        return grant

    assert run(scenario()) is None


def test_begin_dispatch_unknown_chat_returns_none(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        grant = await repo.begin_dispatch(make_request())
        await repo.close()
        return grant

    assert run(scenario()) is None


def test_begin_dispatch_busy_returns_claim_busy(tmp_path):
    """A live, unexpired prepared dispatch reports the exact busy_until —
    never a raw dispatch row."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        first = await repo.begin_dispatch(make_request())
        assert isinstance(first, DispatchGrant)
        second = await repo.begin_dispatch(make_request(cycle_id="cy-2"))
        await repo.close()
        return first, second

    first, second = run(scenario())
    assert isinstance(first, DispatchGrant)
    assert isinstance(second, ClaimBusy)
    assert second.cycle_id == CycleId("cy-1")  # the active owner
    assert second.busy_until == 500.0  # the exact lease expiry


def test_begin_dispatch_recovers_expired_prepared(tmp_path):
    """An expired prepared dispatch is recovered: marked expired and
    replaced by a fresh grant."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        # A timer dispatch claims nothing (no commits exist yet).
        first = await repo.begin_dispatch(
            make_request(cause=DispatchCause.TIMER, started_ts=100.0,
                         expires_at=150.0, now=100.0)
        )
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        # The first dispatch's lease expired long ago: recovered, and the
        # still-unassigned commit attaches to the fresh dispatch.
        second = await repo.begin_dispatch(
            make_request(cycle_id="cy-2", started_ts=300.0, expires_at=500.0, now=300.0)
        )
        await repo.close()
        return first, second

    first, second = run(scenario())
    assert isinstance(first, DispatchGrant)
    assert isinstance(second, DispatchGrant)
    assert second.claim.cycle_id == CycleId("cy-2")
    assert second.attached == (CommitSeq(1),)  # the commit was still unassigned


def test_expired_prepared_dispatch_reattaches_attached_commits(tmp_path):
    """The core gap: an expired prepared dispatch's attached commits are
    detached in the SAME transaction, so the next begin_dispatch reattaches
    exactly those commits and can terminally settle them — nothing strands
    after a crash."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.ingest_message(make_identity(), make_message(msg_id="m2"))
        # A prepared dispatch attaches both commits, then its lease expires.
        first = await repo.begin_dispatch(
            make_request(started_ts=100.0, expires_at=150.0, now=100.0)
        )
        assert isinstance(first, DispatchGrant)
        assert first.attached == (CommitSeq(1), CommitSeq(2))
        # The lease expired long ago: recovery detaches and reattaches.
        second = await repo.begin_dispatch(
            make_request(cycle_id="cy-2", started_ts=300.0, expires_at=500.0, now=300.0)
        )
        assert isinstance(second, DispatchGrant)
        assert second.attached == (CommitSeq(1), CommitSeq(2))
        # The reattached commits can be terminally settled.
        await repo.settle_dispatch(
            make_settle(second.dispatch_id, outcome="finish", cycle_id="cy-2"),
            [item(idem_key="k1"), item(text="second", idem_key="k2")],
            now=300.0,
        )
        state = await repo.get_chat_state(CK)
        outbox = await repo.list_ready_outbox(CK, now=999.0)
        unassigned = await repo.list_unassigned_commits(CK)
        states = await repo._db.read(
            lambda c: c.execute(
                "SELECT state FROM dispatches WHERE id IN (?, ?) ORDER BY id",
                (first.dispatch_id, second.dispatch_id),
            ).fetchall()
        )
        await repo.close()
        return second, state, outbox, unassigned, states

    second, state, outbox, unassigned, states = run(scenario())
    assert isinstance(second, DispatchGrant)
    assert state is not None and state.cursor_msg_id == second.through_msg_id
    assert [o.idem_key for o in outbox] == ["k1", "k2"]
    assert unassigned == []  # the reattached commits were consumed
    assert [s[0] for s in states] == ["expired", "completed"]


def test_expired_priority_dispatch_reattaches_attached_commits(tmp_path):
    """A priority (timer) dispatch that attached commits recovers the same
    way: the next begin_dispatch reattaches exactly those commits and can
    terminally settle them."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        first = await repo.begin_dispatch(
            make_request(cause=DispatchCause.TIMER, scheduled_ts=200.0,
                         started_ts=100.0, expires_at=150.0, now=100.0)
        )
        assert isinstance(first, DispatchGrant)
        assert first.attached == (CommitSeq(1),)
        # The lease expired: the priority dispatch recovers and reattaches.
        second = await repo.begin_dispatch(
            make_request(cause=DispatchCause.TIMER, cycle_id="cy-2",
                         scheduled_ts=400.0, started_ts=300.0,
                         expires_at=500.0, now=300.0)
        )
        assert isinstance(second, DispatchGrant)
        assert second.attached == (CommitSeq(1),)
        await repo.settle_dispatch(
            make_settle(second.dispatch_id, outcome="finish", cycle_id="cy-2"),
            [item(idem_key="k1")], now=300.0,
        )
        state = await repo.get_chat_state(CK)
        await repo.close()
        return second, state

    second, state = run(scenario())
    assert isinstance(second, DispatchGrant)
    assert state is not None and state.cursor_msg_id == second.through_msg_id


def test_live_busy_dispatch_keeps_commits_attached(tmp_path):
    """A live, unexpired prepared dispatch is NOT detached: the second
    begin_dispatch reports ClaimBusy and the attached commits stay on the
    live dispatch (still terminally settleable)."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        first = await repo.begin_dispatch(make_request())
        assert isinstance(first, DispatchGrant)
        assert first.attached == (CommitSeq(1),)
        second = await repo.begin_dispatch(make_request(cycle_id="cy-2"))
        assert isinstance(second, ClaimBusy)
        # The commit is still attached to the live dispatch: not unassigned.
        unassigned = await repo.list_unassigned_commits(CK)
        # The live dispatch can still terminally settle its attached commit.
        await repo.settle_dispatch(
            make_settle(first.dispatch_id, outcome="finish"),
            [item(idem_key="k1")], now=100.0,
        )
        state = await repo.get_chat_state(CK)
        await repo.close()
        return first, second, unassigned, state

    first, second, unassigned, state = run(scenario())
    assert isinstance(first, DispatchGrant)
    assert isinstance(second, ClaimBusy)
    assert unassigned == []  # still attached to the live dispatch
    assert state is not None and state.cursor_msg_id == first.through_msg_id


def test_completed_dispatch_keeps_commits_attached(tmp_path):
    """A terminally completed dispatch's commits are consumed, not
    detached: they stay attached (dispatch_id set) and never become
    unassigned — the expiry-recovery detach never touches them."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        await repo.settle_dispatch(
            make_settle(grant.dispatch_id, outcome="finish"),
            [item(idem_key="k1")], now=100.0,
        )
        unassigned = await repo.list_unassigned_commits(CK)
        # The commit still points at the completed dispatch.
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT dispatch_id FROM inbound_commits WHERE id = 1"
            ).fetchone()
        )
        await repo.close()
        return unassigned, row, grant.dispatch_id

    unassigned, row, dispatch_id = run(scenario())
    assert unassigned == []
    assert row[0] == dispatch_id  # still attached to the completed dispatch


def test_released_dispatch_commits_reattach_on_next_begin(tmp_path):
    """A released dispatch's commits were detached by settlement (existing
    behavior); the expiry-recovery detach never touches a released row, and
    the next begin_dispatch reattaches the still-pending commits."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        await repo.settle_dispatch(
            make_settle(grant.dispatch_id, outcome="release"), [], now=100.0
        )
        # Released: the commit was detached by settlement.
        unassigned = await repo.list_unassigned_commits(CK)
        assert unassigned == [CommitSeq(1)]
        # The released dispatch row is untouched by the recovery detach.
        state = await repo._db.read(
            lambda c: c.execute(
                "SELECT state FROM dispatches WHERE id = ?", (grant.dispatch_id,)
            ).fetchone()
        )
        # A fresh dispatch reattaches the still-pending commit.
        again = await repo.begin_dispatch(make_request(cycle_id="cy-2"))
        assert isinstance(again, DispatchGrant)
        assert again.attached == (CommitSeq(1),)
        await repo.close()
        return state, again

    state, again = run(scenario())
    assert state[0] == "released"
    assert isinstance(again, DispatchGrant)
    assert again.attached == (CommitSeq(1),)


def test_expired_recovery_export_markers_still_correct(tmp_path):
    """Crash/export behavior under the settled-only contract: after an
    expired prepared dispatch is recovered, the startup export appends the
    unexported COMMIT marker but NEVER a prepared or expired dispatch
    marker — only settled (completed/released) dispatches are exported, so
    an expired dispatch produces no fake evaluation marker."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        first = await repo.begin_dispatch(
            make_request(started_ts=100.0, expires_at=150.0, now=100.0)
        )
        assert isinstance(first, DispatchGrant)
        # Crash: the lease expires, then a fresh dispatch recovers it.
        second = await repo.begin_dispatch(
            make_request(cycle_id="cy-2", started_ts=300.0, expires_at=500.0, now=300.0)
        )
        assert isinstance(second, DispatchGrant)
        assert second.attached == (CommitSeq(1),)
        # Restart: export every unexported marker. The commit marker
        # exports; the expired (first) and prepared (second) dispatch rows
        # are NOT settled, so they are NOT exported.
        recorder = Recorder(tmp_path / "events.jsonl")
        await export_unexported(recorder, repo)
        recorder.close()
        markers = read_markers(tmp_path / "events.jsonl")
        unexported_commits = await repo.list_unexported_commits()
        unexported_dispatches = await repo.list_unexported_dispatches()
        states = await repo._db.read(
            lambda c: c.execute(
                "SELECT state, exported FROM dispatches WHERE id IN (?, ?) ORDER BY id",
                (first.dispatch_id, second.dispatch_id),
            ).fetchall()
        )
        await repo.close()
        return (markers, unexported_commits, unexported_dispatches,
                first.dispatch_id, second.dispatch_id, states)

    markers, unexported_commits, unexported_dispatches, first_id, second_id, states = run(scenario())
    # Only the commit marker is exported; neither dispatch marker is.
    assert [(m.record_type, m.sequence) for m in markers] == [("commit", 1)]
    assert unexported_commits == []
    # list_unexported_dispatches returns only SETTLED dispatches: the
    # expired and prepared rows are never exported and never marked
    # exported (their durable exported flag stays 0).
    assert unexported_dispatches == []
    assert [(s[0], s[1]) for s in states] == [("expired", 0), ("prepared", 0)]


# ── settle_dispatch: release/delay vs terminal finish ───────────────────────

def test_delay_releases_without_cursor_or_outbox_movement(tmp_path):
    """Ordinary delay releases the claim: no cursor advance, no outbox
    rows, no terminal cycle — the trace is recorded on the released row."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        await repo.settle_dispatch(
            make_settle(grant.dispatch_id, outcome="delay", trace_json='{"d":1}'),
            [item()],
            now=100.0,
        )
        state = await repo.get_chat_state(CK)
        reason = await repo.get_latest_terminal_end_reason(CK)
        outbox = await repo.list_ready_outbox(CK, now=999.0)
        unassigned = await repo.list_unassigned_commits(CK)
        # A fresh dispatch may now claim the still-pending commit.
        again = await repo.begin_dispatch(make_request(cycle_id="cy-2"))
        assert isinstance(again, DispatchGrant)
        await repo.close()
        return state, reason, outbox, unassigned, again

    state, reason, outbox, unassigned, again = run(scenario())
    assert state is not None and state.cursor_msg_id is None  # cursor unmoved
    assert reason is None  # no terminal cycle
    assert outbox == []  # no outbox rows
    assert unassigned == [CommitSeq(1)]  # still pending
    assert isinstance(again, DispatchGrant)  # the claim was released


def test_release_gives_claim_back_without_movement(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        await repo.settle_dispatch(
            make_settle(grant.dispatch_id, outcome="release"), [], now=100.0
        )
        state = await repo.get_chat_state(CK)
        again = await repo.begin_dispatch(make_request(cycle_id="cy-2"))
        assert isinstance(again, DispatchGrant)
        await repo.close()
        return state, again

    state, again = run(scenario())
    assert state is not None and state.cursor_msg_id is None
    assert isinstance(again, DispatchGrant)


def test_terminal_finish_advances_cursor_and_creates_outbox(tmp_path):
    """Terminal settlement is the ONLY cursor/outbox path: cursor advances
    to the dispatch's stored through boundary, the outbox batch lands with
    cycle provenance, the durable hold/idle state materializes, and the
    dispatch completes with its trace."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.ingest_message(make_identity(), make_message(msg_id="m2"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        await repo.settle_dispatch(
            make_settle(
                grant.dispatch_id, outcome="finish", end_reason="completed",
                hold_until=600.0, idle_streak_after=2, trace_json='{"t":1}',
                tokens_in=10, tokens_out=20,
            ),
            [item(idem_key="k1"), item(text="second", idem_key="k2")],
            now=100.0,
        )
        state = await repo.get_chat_state(CK)
        reason = await repo.get_latest_terminal_end_reason(CK)
        outbox = await repo.list_ready_outbox(CK, now=999.0)
        unassigned = await repo.list_unassigned_commits(CK)
        # The attached commits are no longer unassigned.
        await repo.close()
        return grant, state, reason, outbox, unassigned

    grant, state, reason, outbox, unassigned = run(scenario())
    assert isinstance(grant, DispatchGrant)
    assert state is not None
    assert state.cursor_msg_id == grant.through_msg_id  # advanced to the boundary
    assert state.hold_until == 600.0
    assert state.idle_streak == 2
    assert reason == "completed"
    assert [o.idem_key for o in outbox] == ["k1", "k2"]
    assert unassigned == []


def test_settle_fences_reject_stale_or_expired(tmp_path):
    """Settlement fences: wrong owner, expired lease, or a moved cursor
    raise ClaimError and change nothing."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        errors = []
        for settle in (
            make_settle(grant.dispatch_id, cycle_id="cy-wrong"),
            make_settle(grant.dispatch_id, outcome="delay", cycle_id="cy-wrong"),
        ):
            try:
                await repo.settle_dispatch(settle, [], now=100.0)
            except Exception as e:  # noqa: BLE001
                errors.append(type(e).__name__)
        # Expired lease: settle with now past expires_at.
        try:
            await repo.settle_dispatch(
                make_settle(grant.dispatch_id, outcome="release"), [], now=9999.0
            )
        except Exception as e:  # noqa: BLE001
            errors.append(type(e).__name__)
        # The dispatch is still prepared and settleable at a valid time.
        await repo.settle_dispatch(
            make_settle(grant.dispatch_id, outcome="release"), [], now=100.0
        )
        await repo.close()
        return errors

    errors = run(scenario())
    assert errors == ["ClaimError", "ClaimError", "ClaimError"]


def test_finish_rejects_moved_cursor(tmp_path):
    """A terminal finish whose chat cursor moved past the dispatch's start
    boundary raises ClaimError — the cursor can never be double-advanced."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        # A legacy cycle finishes first and moves the cursor.
        from tests.durable_helpers import make_claim, make_finish

        legacy = await repo.claim_cycle(make_claim())
        assert legacy is not None
        await repo.finish_cycle(make_finish(), [], now=100.0)
        try:
            await repo.settle_dispatch(
                make_settle(grant.dispatch_id, outcome="finish"), [], now=100.0
            )
        except Exception as e:  # noqa: BLE001
            return type(e).__name__
        return None

    assert run(scenario()) == "ClaimError"


# ── at-least-once export and crash points ───────────────────────────────────

def test_export_unexported_appends_and_marks(tmp_path):
    """The startup export appends every unexported commit/dispatch marker
    and marks it exported — protocol-only (FakeRepo)."""

    async def scenario():
        fake = FakeRepo()
        fake.unexported_commits = [
            CorpusMarker(
                record_type="commit", sequence=CommitSeq(1), chat_key=CK,
                event_id=EventId("ev-1"), wake_kind=WakeKind.INBOUND,
            )
        ]
        fake.unexported_dispatches = [
            CorpusMarker(
                record_type="dispatch", sequence=DispatchId(7), chat_key=CK,
                cause=DispatchCause.TIMER,
            )
        ]
        recorder = Recorder(tmp_path / "events.jsonl")
        await export_unexported(recorder, fake)
        recorder.close()
        return read_markers(tmp_path / "events.jsonl"), fake.calls

    markers, calls = run(scenario())
    assert [m.sequence for m in markers] == [1, 7]
    kinds = [c[0] for c in calls]
    assert kinds == [
        "list_unexported_commits", "mark_commit_exported",
        "list_unexported_dispatches", "mark_dispatch_exported",
    ]


def test_crash_event_no_commit(tmp_path):
    """Crash point 1 — event recorded, commit never happened (unknown
    chat): no commit row, no marker, nothing to export or dispatch."""

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        recorder = Recorder(tmp_path / "events.jsonl")
        ingest = Ingest(repo, recorder, identity=lambda ck: None)
        await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=1.0)
        )
        recorder.close()
        # Restart: export + recovery scans find nothing.
        await export_unexported(recorder, repo)
        unassigned = await repo.list_unassigned_commits(CK)
        pending = await repo.list_ledger_pending_chats()
        await repo.close()
        return read_markers(tmp_path / "events.jsonl"), unassigned, pending

    markers, unassigned, pending = run(scenario())
    assert markers == []  # no commit marker
    assert unassigned == []
    assert pending == []


def test_crash_commit_no_notification_or_export(tmp_path):
    """Crash point 2 — the commit landed but the marker was never exported
    and the scheduler never notified: the startup export re-appends the
    marker, and the recovery scan finds the unassigned commit for a fresh
    dispatch."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        # Commit directly (the legacy path): no marker, no wake.
        result = await repo.ingest_message(
            make_identity(), make_message(msg_id="m1")
        )
        recorder = Recorder(tmp_path / "events.jsonl")
        # Restart: export the unexported commit marker, then dispatch.
        await export_unexported(recorder, repo)
        markers = read_markers(tmp_path / "events.jsonl")
        unassigned = await repo.list_unassigned_commits(CK)
        pending = await repo.list_ledger_pending_chats()
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        recorder.close()
        await repo.close()
        return result, markers, unassigned, pending, grant

    result, markers, unassigned, pending, grant = run(scenario())
    assert result.commit_seq == CommitSeq(1)
    assert len(markers) == 1
    assert markers[0].record_type == "commit"
    assert markers[0].sequence == 1
    assert markers[0].event_id == result.event_id
    assert unassigned == [CommitSeq(1)]
    assert pending == [CK]
    assert isinstance(grant, DispatchGrant)
    assert grant.attached == (CommitSeq(1),)


def test_crash_between_marker_append_and_export_mark(tmp_path):
    """Crash point 2b — the marker was appended but the export mark never
    committed: the startup export re-appends it; readers deduplicate by
    (record_type, sequence)."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")

        class _CrashAfterAppend(SqliteRepository):
            async def mark_commit_exported(self, commit_seq):
                raise RuntimeError("crash before the export mark commits")

        recorder = Recorder(tmp_path / "events.jsonl")
        ingest = Ingest(
            _CrashAfterAppend(db), recorder, identity=lambda ck: make_identity()
        )
        with pytest.raises(RuntimeError, match="crash"):
            await ingest.handle(
                AdapterEvent(type="message", payload=make_message(), ts=1.0)
            )
        # The marker line was appended before the crash.
        assert len(read_markers(tmp_path / "events.jsonl")) == 1
        # Restart: the startup export re-appends (duplicate) and marks.
        await export_unexported(recorder, repo)
        recorder.close()
        markers = read_markers(tmp_path / "events.jsonl")
        unexported = await repo.list_unexported_commits()
        await repo.close()
        return markers, unexported

    markers, unexported = run(scenario())
    assert len(markers) == 1  # deduplicated
    assert markers[0].sequence == 1
    assert unexported == []  # now marked exported


def test_crash_prepared_dispatch_survives_restart(tmp_path):
    """Crash point 3 — a prepared dispatch survives a restart: it is still
    settleable, blocks a new claim while live, and is recovered after its
    lease expires."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        await repo.close()
        # Restart on the same file.
        db2, repo2 = await open_repo_with_chat(tmp_path / "t.db")
        busy = await repo2.begin_dispatch(make_request(cycle_id="cy-2"))
        # The durable prepared dispatch is still settleable after restart.
        await repo2.settle_dispatch(
            make_settle(grant.dispatch_id, outcome="finish"), [item()], now=100.0
        )
        state = await repo2.get_chat_state(CK)
        await repo2.close()
        return grant, busy, state

    grant, busy, state = run(scenario())
    assert isinstance(grant, DispatchGrant)
    assert isinstance(busy, ClaimBusy)
    assert busy.cycle_id == CycleId("cy-1")
    assert state is not None and state.cursor_msg_id == grant.through_msg_id


def test_crash_settlement_before_export(tmp_path):
    """Crash point 4 — the dispatch settled (terminal finish) but its
    marker was never exported: the startup export appends the dispatch
    marker."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        await repo.settle_dispatch(
            make_settle(grant.dispatch_id, outcome="finish"), [], now=100.0
        )
        recorder = Recorder(tmp_path / "events.jsonl")
        # Restart: export the unexported dispatch marker.
        await export_unexported(recorder, repo)
        recorder.close()
        markers = read_markers(tmp_path / "events.jsonl")
        unexported = await repo.list_unexported_dispatches()
        await repo.close()
        return markers, unexported, grant.dispatch_id

    markers, unexported, dispatch_id = run(scenario())
    # The commit marker (never exported either — the commit went through
    # the direct repo path) and the dispatch marker both export at startup.
    assert [(m.record_type, m.sequence) for m in markers] == [
        ("commit", 1), ("dispatch", dispatch_id),
    ]
    assert markers[1].cause == DispatchCause.INBOUND
    assert unexported == []


def test_marker_dedupe_by_record_type_and_sequence(tmp_path):
    """Readers deduplicate by (record_type, sequence): a re-appended
    duplicate after a crash is tolerated; the same sequence in the OTHER
    record_type is a distinct marker."""

    async def scenario():
        recorder = Recorder(tmp_path / "events.jsonl")
        commit = CorpusMarker(
            record_type="commit", sequence=CommitSeq(1), chat_key=CK,
            event_id=EventId("ev-1"), wake_kind=WakeKind.INBOUND,
        )
        dispatch = CorpusMarker(
            record_type="dispatch", sequence=DispatchId(1), chat_key=CK,
            cause=DispatchCause.TIMER,
        )
        recorder.append_marker(commit)
        recorder.append_marker(commit)  # crash duplicate
        recorder.append_marker(dispatch)  # same sequence, other type
        recorder.append_marker(dispatch)  # crash duplicate
        recorder.close()
        return read_markers(tmp_path / "events.jsonl")

    markers = run(scenario())
    assert [(m.record_type, m.sequence) for m in markers] == [
        ("commit", 1), ("dispatch", 1),
    ]


def test_dispatch_before_commit_reversed_export_order(tmp_path):
    """The core gap: a timer dispatch that writes first excludes a later
    commit. When BOTH markers are unexported after a crash, the generic
    export order (commits first, then dispatches) reverses the live writer
    order in the JSONL — yet the dispatch marker's frozen commit_boundary
    lets replay reconstruct that the live dispatch excluded the commit."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        # Timer dispatch writes FIRST: boundary 0, nothing attached.
        grant = await repo.begin_dispatch(
            make_request(cause=DispatchCause.TIMER, scheduled_ts=200.0)
        )
        assert isinstance(grant, DispatchGrant)
        # The inbound commit lands AFTER the dispatch began.
        result = await repo.ingest_message(
            make_identity(), make_message(msg_id="m1")
        )
        # The dispatch settles (release: no cursor/outbox movement) so it
        # becomes a replayable released dispatch — only settled dispatches
        # are ever exported.
        await repo.settle_dispatch(
            make_settle(grant.dispatch_id, outcome="release"), [], now=100.0
        )
        # Crash before any export: both markers are unexported.
        recorder = Recorder(tmp_path / "events.jsonl")
        # Startup export: commits first, then dispatches — the REVERSED
        # order relative to the live writer order.
        await export_unexported(recorder, repo)
        recorder.close()
        markers = read_markers(tmp_path / "events.jsonl")
        await repo.close()
        return grant, result, markers

    grant, result, markers = run(scenario())
    assert isinstance(grant, DispatchGrant)
    assert grant.attached == ()  # the timer wrote first: nothing attached
    assert grant.commit_boundary == CommitSeq(0)
    assert grant.scheduled_for == 200.0
    assert result.commit_seq == CommitSeq(1)
    # The commit marker precedes the dispatch marker in the file (generic
    # export order), yet the dispatch marker's frozen boundary proves the
    # live dispatch excluded the commit.
    assert [(m.record_type, m.sequence) for m in markers] == [
        ("commit", 1), ("dispatch", grant.dispatch_id),
    ]
    dispatch_marker = markers[1]
    assert dispatch_marker.commit_boundary == CommitSeq(0)
    assert dispatch_marker.scheduled_for == 200.0
    # Replay reconstruction: the commit (seq 1) is BEYOND the dispatch's
    # frozen boundary (0), so it was NOT included in the live dispatch.
    assert result.commit_seq is not None
    assert result.commit_seq > CommitSeq(0)


def test_dispatch_after_commit_inclusion_boundary(tmp_path):
    """A commit that writes first joins the dispatch: the dispatch marker's
    frozen boundary includes it, so replay reconstructs attachment even
    when the dispatch marker is exported BEFORE the commit marker."""

    from pretender.record import export_marker

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        result = await repo.ingest_message(
            make_identity(), make_message(msg_id="m1")
        )
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        # Settle the dispatch (release) so it becomes a replayable released
        # dispatch — only settled dispatches are ever exported.
        await repo.settle_dispatch(
            make_settle(grant.dispatch_id, outcome="release"), [], now=100.0
        )
        recorder = Recorder(tmp_path / "events.jsonl")
        # Export the dispatch marker FIRST, then the commit marker — the
        # reversed order relative to the generic startup export.
        dispatch_marker = (await repo.list_unexported_dispatches())[0]
        commit_marker = (await repo.list_unexported_commits())[0]
        await export_marker(recorder, repo, dispatch_marker)
        await export_marker(recorder, repo, commit_marker)
        recorder.close()
        markers = read_markers(tmp_path / "events.jsonl")
        await repo.close()
        return grant, result, markers

    grant, result, markers = run(scenario())
    assert isinstance(grant, DispatchGrant)
    assert grant.attached == (CommitSeq(1),)
    assert grant.commit_boundary == CommitSeq(1)
    # The dispatch marker precedes the commit marker in the file.
    assert [(m.record_type, m.sequence) for m in markers] == [
        ("dispatch", grant.dispatch_id), ("commit", 1),
    ]
    dispatch_marker = markers[0]
    assert dispatch_marker.commit_boundary == CommitSeq(1)
    # Replay reconstruction: the commit (seq 1) is WITHIN the dispatch's
    # frozen boundary (1), so it WAS included in the live dispatch.
    assert result.commit_seq is not None
    assert dispatch_marker.commit_boundary is not None
    assert result.commit_seq <= dispatch_marker.commit_boundary


def test_old_dispatch_marker_without_metadata_reads_back(tmp_path):
    """Backward compatibility: a v2 dispatch marker (no commit_boundary /
    scheduled_for fields) reads back with None metadata and still dedupes
    by (record_type, sequence)."""

    async def scenario():
        recorder = Recorder(tmp_path / "events.jsonl")
        # A v2-style dispatch marker line, as the old exporter wrote it.
        recorder.write(
            {
                "record_type": "dispatch",
                "sequence": 7,
                "chat_key": CK,
                "cause": DispatchCause.TIMER,
            }
        )
        recorder.append_marker(
            CorpusMarker(
                record_type="dispatch", sequence=DispatchId(7), chat_key=CK,
                cause=DispatchCause.TIMER,
            )
        )  # crash duplicate, same (type, sequence)
        recorder.close()
        return read_markers(tmp_path / "events.jsonl")

    markers = run(scenario())
    assert len(markers) == 1  # deduplicated
    assert markers[0].sequence == 7
    assert markers[0].cause == DispatchCause.TIMER
    assert markers[0].commit_boundary is None  # old marker: no metadata
    assert markers[0].scheduled_for is None


def test_legacy_claim_finish_unaffected_by_boundary_metadata(tmp_path):
    """The additive boundary metadata has no effect on the legacy
    claim_cycle/finish_cycle surface: a legacy terminal cycle still claims,
    finishes, advances the cursor, and creates outbox rows on the v3
    schema."""

    from tests.durable_helpers import make_claim, make_finish

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.claim_cycle(make_claim())
        assert isinstance(grant, ClaimGrant)
        await repo.finish_cycle(
            make_finish(end_reason="completed"), [item(idem_key="k1")], now=100.0
        )
        state = await repo.get_chat_state(CK)
        outbox = await repo.list_ready_outbox(CK, now=999.0)
        await repo.close()
        return grant, state, outbox

    grant, state, outbox = run(scenario())
    assert isinstance(grant, ClaimGrant)
    assert state is not None and state.cursor_msg_id == grant.through_msg_id
    assert [o.idem_key for o in outbox] == ["k1"]


def test_marker_lines_are_skipped_by_event_reader(tmp_path):
    """The event reader skips marker lines: read_corpus round-trips only
    events, so replay is unaffected by the ledger markers."""

    from pretender.record import read_corpus

    async def scenario():
        recorder = Recorder(tmp_path / "events.jsonl")
        recorder.write_event(
            AdapterEvent(type="message", payload=make_message(), ts=1.0),
            event_id=EventId("ev-1"),
        )
        recorder.append_marker(
            CorpusMarker(
                record_type="commit", sequence=CommitSeq(1), chat_key=CK,
                event_id=EventId("ev-1"), wake_kind=WakeKind.INBOUND,
            )
        )
        recorder.close()
        return read_corpus(tmp_path / "events.jsonl")

    events = run(scenario())
    assert len(events) == 1
    assert events[0].type == "message"
    assert events[0].payload.text == "hello"


# ── replayable settled-dispatch marker contract (v4) ────────────────────────
# Only settled (completed/released) dispatches are ever exported; each
# marker carries the full frozen evaluation metadata from the durable row,
# and the exact attached membership is frozen at begin_dispatch so a later
# released/detached dispatch stays replayable.

def test_prepared_dispatch_excluded_from_export(tmp_path):
    """A prepared (unevaluated) dispatch is NEVER exported on startup:
    list_unexported_dispatches returns only settled dispatches, so a
    prepared dispatch produces no marker."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        # Prepared: not settled, so not listed for export.
        unexported = await repo.list_unexported_dispatches()
        recorder = Recorder(tmp_path / "events.jsonl")
        await export_unexported(recorder, repo)
        recorder.close()
        markers = read_markers(tmp_path / "events.jsonl")
        # The commit marker exports; the prepared dispatch does not.
        assert [(m.record_type, m.sequence) for m in markers] == [("commit", 1)]
        assert unexported == []
        await repo.close()
        return markers

    markers = run(scenario())
    assert len(markers) == 1 and markers[0].record_type == "commit"


def test_completed_marker_carries_full_settled_metadata(tmp_path):
    """A terminally completed dispatch marker carries the FULL frozen
    evaluation metadata: settled state, evaluation timestamp, message
    boundaries, exact attached membership, trace, cause, boundary, and
    scheduled time."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.ingest_message(make_identity(), make_message(msg_id="m2"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        await repo.settle_dispatch(
            make_settle(
                grant.dispatch_id, outcome="finish", end_reason="completed",
                trace_json='{"t":1}',
            ),
            [], now=150.0,
        )
        recorder = Recorder(tmp_path / "events.jsonl")
        await export_unexported(recorder, repo)
        recorder.close()
        markers = read_markers(tmp_path / "events.jsonl")
        dispatch = [m for m in markers if m.record_type == "dispatch"][0]
        await repo.close()
        return grant, dispatch

    grant, dispatch = run(scenario())
    assert dispatch.state == "completed"
    assert dispatch.settled_ts == 150.0
    assert dispatch.start_msg_id == grant.start_msg_id
    assert dispatch.through_msg_id == grant.through_msg_id
    assert dispatch.attached == grant.attached
    assert dispatch.trace_json == '{"t":1}'
    assert dispatch.cause == DispatchCause.INBOUND
    assert dispatch.commit_boundary == grant.commit_boundary
    assert dispatch.scheduled_for is None


def test_released_dispatch_retains_membership_after_detach(tmp_path):
    """A released dispatch's live commit rows are detached (dispatch_id =
    NULL) by settlement, but the frozen attached membership is PRESERVED
    on the row and exported: the released dispatch stays replayable with
    its exact membership."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.ingest_message(make_identity(), make_message(msg_id="m2"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        assert grant.attached == (CommitSeq(1), CommitSeq(2))
        await repo.settle_dispatch(
            make_settle(grant.dispatch_id, outcome="release"), [], now=120.0
        )
        # The live commit rows were detached by settlement.
        unassigned = await repo.list_unassigned_commits(CK)
        assert unassigned == [CommitSeq(1), CommitSeq(2)]
        recorder = Recorder(tmp_path / "events.jsonl")
        await export_unexported(recorder, repo)
        recorder.close()
        markers = read_markers(tmp_path / "events.jsonl")
        dispatch = [m for m in markers if m.record_type == "dispatch"][0]
        await repo.close()
        return dispatch, unassigned

    dispatch, unassigned = run(scenario())
    assert dispatch.state == "released"
    assert dispatch.settled_ts == 120.0
    assert dispatch.attached == (CommitSeq(1), CommitSeq(2))
    assert dispatch.trace_json is None  # release carries no trace


def test_prepared_expiry_detaches_rows_but_no_fake_marker(tmp_path):
    """An expired prepared dispatch detaches its live commit rows (so they
    reattach to the next dispatch) but produces NO evaluation marker: it is
    not settled, so list_unexported_dispatches never exports it — and its
    frozen attached membership is preserved on the expired row."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        first = await repo.begin_dispatch(
            make_request(started_ts=100.0, expires_at=150.0, now=100.0)
        )
        assert isinstance(first, DispatchGrant)
        assert first.attached == (CommitSeq(1),)
        # The lease expires; a fresh dispatch recovers it and detaches the
        # live commit rows from the expired dispatch.
        second = await repo.begin_dispatch(
            make_request(cycle_id="cy-2", started_ts=300.0, expires_at=500.0, now=300.0)
        )
        assert isinstance(second, DispatchGrant)
        assert second.attached == (CommitSeq(1),)
        # The expired dispatch's live membership was detached and the
        # commit reattached to the fresh dispatch.
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT dispatch_id FROM inbound_commits WHERE id = 1"
            ).fetchone()
        )
        assert row[0] == second.dispatch_id
        # The expired dispatch's frozen membership is preserved on the row.
        expired = await repo._db.read(
            lambda c: c.execute(
                "SELECT attached_json FROM dispatches WHERE id = ?",
                (first.dispatch_id,),
            ).fetchone()
        )
        assert expired[0] == "[1]"
        # The expired dispatch is NOT settled: no marker is exported for it.
        recorder = Recorder(tmp_path / "events.jsonl")
        await export_unexported(recorder, repo)
        recorder.close()
        markers = read_markers(tmp_path / "events.jsonl")
        dispatch_markers = [m for m in markers if m.record_type == "dispatch"]
        await repo.close()
        return dispatch_markers, row, expired, second.dispatch_id

    dispatch_markers, row, expired, second_id = run(scenario())
    assert dispatch_markers == []  # no fake evaluation marker
    assert row[0] == second_id
    assert expired[0] == "[1]"


def test_export_retry_dedupe_with_full_fields(tmp_path):
    """At-least-once export with the full v4 fields: a crash between the
    marker append and the export mark leaves the marker unexported; the
    startup export re-appends it (duplicate) and readers deduplicate by
    (record_type, sequence) — the full fields survive the retry."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        await repo.settle_dispatch(
            make_settle(
                grant.dispatch_id, outcome="finish", end_reason="completed",
                trace_json='{"t":1}',
            ),
            [], now=150.0,
        )
        recorder = Recorder(tmp_path / "events.jsonl")
        # Crash between the append and the export mark.
        marker = (await repo.list_unexported_dispatches())[0]
        recorder.append_marker(marker)
        # Restart: the startup export re-appends (duplicate) and marks.
        await export_unexported(recorder, repo)
        recorder.close()
        markers = read_markers(tmp_path / "events.jsonl")
        dispatch = [m for m in markers if m.record_type == "dispatch"]
        unexported = await repo.list_unexported_dispatches()
        await repo.close()
        return dispatch, unexported

    dispatch, unexported = run(scenario())
    assert len(dispatch) == 1  # deduplicated by (record_type, sequence)
    assert dispatch[0].state == "completed"
    assert dispatch[0].settled_ts == 150.0
    assert dispatch[0].attached == (CommitSeq(1),)
    assert dispatch[0].trace_json == '{"t":1}'
    assert unexported == []  # now marked exported


def test_old_v2_v3_dispatch_markers_read_back(tmp_path):
    """Backward compatibility: a v2 dispatch marker (no boundary/scheduled)
    and a v3 dispatch marker (boundary/scheduled but no settled fields)
    both read back with None/empty settled metadata and still dedupe."""

    async def scenario():
        recorder = Recorder(tmp_path / "events.jsonl")
        # v2-style marker: no boundary, no scheduled, no settled fields.
        recorder.write(
            {
                "record_type": "dispatch", "sequence": 5, "chat_key": CK,
                "cause": DispatchCause.INBOUND,
            }
        )
        # v3-style marker: boundary + scheduled, but no settled fields.
        recorder.write(
            {
                "record_type": "dispatch", "sequence": 6, "chat_key": CK,
                "cause": DispatchCause.TIMER, "commit_boundary": 3,
                "scheduled_for": 200.0,
            }
        )
        recorder.close()
        return read_markers(tmp_path / "events.jsonl")

    markers = run(scenario())
    assert len(markers) == 2
    v2, v3 = markers
    assert v2.commit_boundary is None and v2.scheduled_for is None
    assert v2.state is None and v2.settled_ts is None
    assert v2.attached == () and v2.trace_json is None
    assert v2.start_msg_id is None and v2.through_msg_id is None
    assert v3.commit_boundary == CommitSeq(3) and v3.scheduled_for == 200.0
    assert v3.state is None and v3.settled_ts is None
    assert v3.attached == () and v3.trace_json is None
    assert v3.start_msg_id is None and v3.through_msg_id is None


def test_legacy_paths_unaffected_by_settled_marker_columns(tmp_path):
    """The additive v4 columns have no effect on the legacy
    claim_cycle/finish_cycle surface: a legacy terminal cycle still claims,
    finishes, advances the cursor, and creates outbox rows on the v4
    schema, and no dispatch row is created by the legacy path."""

    from tests.durable_helpers import make_claim, make_finish

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.claim_cycle(make_claim())
        assert isinstance(grant, ClaimGrant)
        await repo.finish_cycle(
            make_finish(end_reason="completed"), [item(idem_key="k1")], now=100.0
        )
        state = await repo.get_chat_state(CK)
        outbox = await repo.list_ready_outbox(CK, now=999.0)
        dispatch_count = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM dispatches").fetchone()[0]
        )
        await repo.close()
        return grant, state, outbox, dispatch_count

    grant, state, outbox, dispatch_count = run(scenario())
    assert isinstance(grant, ClaimGrant)
    assert state is not None and state.cursor_msg_id == grant.through_msg_id
    assert [o.idem_key for o in outbox] == ["k1"]
    assert dispatch_count == 0  # the legacy path creates no dispatch rows


# ── preserved regressions: trusted echo and average ─────────────────────────

def test_trusted_echo_reconciliation_regression(tmp_path):
    """The trusted self-echo flow still reconciles exactly one in-flight
    row — and the echo's commit row carries wake_kind none."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        from tests.durable_helpers import finish_batch

        await finish_batch(repo, [item(idem_key="cy-1:0")])
        await repo.attempt_outbox(1, 100.0)
        result = await repo.ingest_message(
            make_identity(),
            make_message(
                msg_id="real:1", is_self=True, text="hi",
                sender_id="bot-1", recv_ts=150.0,
            ),
            self_echo_delivery_key="cy-1:0",
        )
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT wake_kind FROM inbound_commits WHERE id = ?",
                (result.commit_seq,),
            ).fetchone()
        )
        sent = await repo._db.read(
            lambda c: c.execute(
                "SELECT state, platform_msg_id FROM outbox WHERE id = 1"
            ).fetchone()
        )
        await repo.close()
        return result, row, sent

    result, row, sent = run(scenario())
    assert result.echo_status == EchoStatus.RECONCILED
    assert result.wake_kind == WakeKind.NONE
    assert row[0] == WakeKind.NONE
    assert sent == ("sent", "real:1")


def test_avg_interval_regression(tmp_path):
    """The durable EWMA average still folds in newly inserted non-self
    messages — the ledger commit row does not disturb it."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", recv_ts=100.0)
        )
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", recv_ts=130.0)
        )
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state

    state = run(scenario())
    assert state is not None and state.avg_interval == pytest.approx(30.0)


# ── typed boundary validation ───────────────────────────────────────────────

def test_dispatch_request_validates_lease_and_cause():
    with pytest.raises(ValueError, match="cause"):
        make_request(cause="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        make_request(expires_at=100.0, started_ts=100.0)  # not > started
    with pytest.raises(ValueError, match="finite"):
        make_request(now=float("inf"))


def test_dispatch_settle_requires_end_reason_for_finish():
    with pytest.raises(ValueError, match="end_reason"):
        make_settle(1, outcome="finish", end_reason=None)
    with pytest.raises(ValueError, match="outcome"):
        make_settle(1, outcome="bogus")  # type: ignore[arg-type]


def test_corpus_marker_validates_shape():
    with pytest.raises(ValueError, match="record_type"):
        CorpusMarker(record_type="x", sequence=1, chat_key=CK)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="event_id"):
        CorpusMarker(
            record_type="commit", sequence=1, chat_key=CK, wake_kind=WakeKind.INBOUND
        )
    with pytest.raises(ValueError, match="cause"):
        CorpusMarker(record_type="dispatch", sequence=1, chat_key=CK)


def test_dispatch_marker_validates_boundary_and_scheduled_for():
    with pytest.raises(ValueError, match="commit_boundary"):
        CorpusMarker(
            record_type="dispatch", sequence=1, chat_key=CK,
            cause=DispatchCause.TIMER, commit_boundary=CommitSeq(-1),
        )
    with pytest.raises(ValueError, match="commit_boundary"):
        CorpusMarker(
            record_type="dispatch", sequence=1, chat_key=CK,
            cause=DispatchCause.TIMER, commit_boundary=CommitSeq(True),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="scheduled_for"):
        CorpusMarker(
            record_type="dispatch", sequence=1, chat_key=CK,
            cause=DispatchCause.TIMER, scheduled_for=float("inf"),
        )
    # Valid: boundary 0 and no scheduled time are fine.
    marker = CorpusMarker(
        record_type="dispatch", sequence=1, chat_key=CK,
        cause=DispatchCause.STARTUP, commit_boundary=CommitSeq(0),
        scheduled_for=None,
    )
    assert marker.commit_boundary == CommitSeq(0)
    assert marker.scheduled_for is None


def test_dispatch_marker_validates_settled_state_and_metadata():
    """The v4 settled-state fields validate: state must be completed or
    released, settled_ts must be finite, boundaries nonnegative, and the
    attached membership a tuple of nonnegative sequences."""
    with pytest.raises(ValueError, match="state"):
        CorpusMarker(
            record_type="dispatch", sequence=1, chat_key=CK,
            cause=DispatchCause.INBOUND, state="prepared",
        )
    with pytest.raises(ValueError, match="settled_ts"):
        CorpusMarker(
            record_type="dispatch", sequence=1, chat_key=CK,
            cause=DispatchCause.INBOUND, settled_ts=float("inf"),
        )
    with pytest.raises(ValueError, match="start_msg_id"):
        CorpusMarker(
            record_type="dispatch", sequence=1, chat_key=CK,
            cause=DispatchCause.INBOUND, start_msg_id=MessageRowId(-1),
        )
    with pytest.raises(ValueError, match="through_msg_id"):
        CorpusMarker(
            record_type="dispatch", sequence=1, chat_key=CK,
            cause=DispatchCause.INBOUND, through_msg_id=MessageRowId(-1),
        )
    with pytest.raises(ValueError, match="attached"):
        CorpusMarker(
            record_type="dispatch", sequence=1, chat_key=CK,
            cause=DispatchCause.INBOUND, attached=(CommitSeq(-1),),
        )
    # Valid: a fully settled marker with all metadata.
    marker = CorpusMarker(
        record_type="dispatch", sequence=1, chat_key=CK,
        cause=DispatchCause.INBOUND, state="completed", settled_ts=150.0,
        start_msg_id=MessageRowId(0), through_msg_id=MessageRowId(2),
        attached=(CommitSeq(1), CommitSeq(2)), trace_json='{"t":1}',
    )
    assert marker.state == "completed"
    assert marker.attached == (CommitSeq(1), CommitSeq(2))


def test_ingest_result_validates_wake_kind_and_commit_seq():
    with pytest.raises(ValueError, match="wake kind"):
        IngestResult(wake_kind="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="commit_seq"):
        IngestResult(commit_seq=CommitSeq(-1))


# ── CycleRunner.run_dispatch: at-least-once dispatch marker export ──────────

def _dispatch_runner(repo, recorder, *, clock_epoch: float = 200.0, **kw):
    return CycleRunner(
        repo, Gate(), Config(), clock=VirtualClock(epoch=clock_epoch),
        marker_exporter=lambda marker: export_marker(recorder, repo, marker),
        **kw,
    )


def test_run_dispatch_exports_dispatch_marker(tmp_path):
    """The runner arranges the at-least-once dispatch marker export through
    the injected exporter (wired to record.export_marker): the marker
    carries the frozen boundary and scheduled time, and the dispatch is
    marked exported."""
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1", recv_ts=100.0))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        recorder = Recorder(tmp_path / "events.jsonl")
        runner = _dispatch_runner(repo, recorder)
        decision = await runner.run_dispatch(grant)
        recorder.close()
        markers = read_markers(tmp_path / "events.jsonl")
        unexported = await repo.list_unexported_dispatches()
        await repo.close()
        return decision, markers, unexported, grant.dispatch_id

    decision, markers, unexported, dispatch_id = run(scenario())
    assert decision.action == "delay"
    assert len(markers) == 1
    marker = markers[0]
    assert marker.record_type == "dispatch"
    assert marker.sequence == dispatch_id
    assert marker.chat_key == CK
    assert marker.cause == DispatchCause.INBOUND
    assert marker.commit_boundary == CommitSeq(1)
    assert marker.scheduled_for is None
    assert unexported == []  # marked exported right after the append


def test_run_dispatch_timer_marker_carries_boundary_and_scheduled_for(tmp_path):
    """A timer dispatch marker carries the frozen commit boundary and the
    scheduled deadline so replay reconstructs the exact attachment
    boundary."""
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1", recv_ts=100.0))
        grant = await repo.begin_dispatch(
            make_request(cause=DispatchCause.TIMER, scheduled_ts=250.0)
        )
        assert isinstance(grant, DispatchGrant)
        recorder = Recorder(tmp_path / "events.jsonl")
        runner = _dispatch_runner(repo, recorder)
        decision = await runner.run_dispatch(grant)
        recorder.close()
        markers = read_markers(tmp_path / "events.jsonl")
        await repo.close()
        return decision, markers

    decision, markers = run(scenario())
    assert len(markers) == 1
    marker = markers[0]
    assert marker.cause == DispatchCause.TIMER
    assert marker.commit_boundary == CommitSeq(1)
    assert marker.scheduled_for == 250.0


def test_run_dispatch_marker_duplicate_recovery(tmp_path):
    """At-least-once export: a crash between the marker append and the
    export mark leaves the marker unexported; the startup export re-appends
    it (duplicate) and readers deduplicate by (record_type, sequence)."""
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1", recv_ts=100.0))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        recorder = Recorder(tmp_path / "events.jsonl")

        async def crash_exporter(marker):
            recorder.append_marker(marker)
            raise RuntimeError("crash before the export mark commits")

        runner = CycleRunner(
            repo, Gate(), Config(), clock=VirtualClock(epoch=200.0),
            marker_exporter=crash_exporter,
        )
        with pytest.raises(RuntimeError, match="crash"):
            await runner.run_dispatch(grant)
        # The marker line was appended before the crash; the dispatch is
        # already settled (the settlement committed before the export).
        assert len(read_markers(tmp_path / "events.jsonl")) == 1
        # Restart: the startup export re-appends (duplicate) and marks.
        await export_unexported(recorder, repo)
        recorder.close()
        markers = read_markers(tmp_path / "events.jsonl")
        unexported = await repo.list_unexported_dispatches()
        await repo.close()
        return markers, unexported, grant.dispatch_id

    markers, unexported, dispatch_id = run(scenario())
    # The startup export also re-exports the never-exported commit marker;
    # the dispatch marker deduplicates to exactly one.
    dispatch_markers = [m for m in markers if m.record_type == "dispatch"]
    assert len(dispatch_markers) == 1  # deduplicated
    assert dispatch_markers[0].sequence == dispatch_id
    assert unexported == []  # now marked exported


# ── durable agent barrier: defer / wait streak / renew_dispatch ─────────────

def test_begin_dispatch_defers_while_barrier_active(tmp_path):
    """An active agent barrier (agent_resume_at > now) makes begin_dispatch
    return DispatchDeferred instead of creating/attaching a dispatch."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        # Record a barrier in the future.
        await repo._db.write(
            lambda c: c.execute(
                "UPDATE chats SET agent_resume_at = ? WHERE chat_key = ?",
                (300.0, CK),
            )
        )
        deferred = await repo.begin_dispatch(make_request(now=100.0))
        # The commit stays unassigned: nothing was attached.
        unassigned = await repo.list_unassigned_commits(CK)
        await repo.close()
        return deferred, unassigned

    deferred, unassigned = run(scenario())
    assert isinstance(deferred, DispatchDeferred)
    assert deferred.chat_key == CK
    assert deferred.resume_at == 300.0
    assert deferred.defer_kind == "retry"
    assert unassigned == [CommitSeq(1)]  # nothing attached


def test_begin_dispatch_clears_expired_barrier_and_grants(tmp_path):
    """Once the barrier expires (agent_resume_at <= now), begin_dispatch
    clears it and grants normally, attaching the pending commit."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo._db.write(
            lambda c: c.execute(
                "UPDATE chats SET agent_resume_at = ? WHERE chat_key = ?",
                (150.0, CK),
            )
        )
        grant = await repo.begin_dispatch(make_request(now=200.0))
        state = await repo.get_chat_state(CK)
        await repo.close()
        return grant, state

    grant, state = run(scenario())
    assert isinstance(grant, DispatchGrant)
    assert grant.attached == (CommitSeq(1),)
    assert state is not None and state.agent_resume_at is None  # cleared


def test_defer_barrier_survives_restart(tmp_path):
    """The barrier is durable: after a restart it still defers begin_dispatch
    until it expires."""

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        await repo.settle_dispatch(
            make_settle(
                (await repo.begin_dispatch(make_request())).dispatch_id,
                outcome="defer", resume_at=500.0, defer_kind="wait",
            ),
            [], now=100.0,
        )
        await repo.close()
        # Restart on the same file.
        _db2, repo2 = await open_repo_with_chat(tmp_path / "t.db")
        deferred = await repo2.begin_dispatch(make_request(cycle_id="cy-2", now=200.0))
        state = await repo2.get_chat_state(CK)
        await repo2.close()
        return deferred, state

    deferred, state = run(scenario())
    assert isinstance(deferred, DispatchDeferred)
    assert deferred.resume_at == 500.0
    assert state is not None and state.agent_resume_at == 500.0


def test_wait_defer_increments_streak_retry_does_not(tmp_path):
    """A wait defer increments the wait streak; a retry defer does not."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        # First a wait defer: streak 0 -> 1.
        wait_grant = await repo.begin_dispatch(make_request())
        assert isinstance(wait_grant, DispatchGrant)
        await repo.settle_dispatch(
            make_settle(wait_grant.dispatch_id, outcome="defer",
                        resume_at=300.0, defer_kind="wait"),
            [], now=100.0,
        )
        state1 = await repo.get_chat_state(CK)
        # Then a retry defer: streak stays 1.
        retry_grant = await repo.begin_dispatch(make_request(cycle_id="cy-2", now=400.0))
        assert isinstance(retry_grant, DispatchGrant)
        await repo.settle_dispatch(
            make_settle(retry_grant.dispatch_id, outcome="defer", cycle_id="cy-2",
                        resume_at=700.0, defer_kind="retry"),
            [], now=400.0,
        )
        state2 = await repo.get_chat_state(CK)
        await repo.close()
        return state1, state2

    state1, state2 = run(scenario())
    assert state1 is not None and state1.wait_streak == 1
    assert state1.agent_resume_at == 300.0
    assert state2 is not None and state2.wait_streak == 1  # retry: no increment
    assert state2.agent_resume_at == 700.0


def test_terminal_finish_clears_barrier_and_resets_streak(tmp_path):
    """A terminal finish clears the agent barrier and resets the wait streak
    in the same transaction as the cursor advance."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        # Set a barrier and a streak directly.
        await repo._db.write(
            lambda c: c.execute(
                "UPDATE chats SET agent_resume_at = ?, wait_streak = 2"
                " WHERE chat_key = ?",
                (400.0, CK),
            )
        )
        grant = await repo.begin_dispatch(
            make_request(now=600.0, started_ts=600.0, expires_at=1000.0)
        )
        assert isinstance(grant, DispatchGrant)
        await repo.settle_dispatch(
            make_settle(grant.dispatch_id, outcome="finish", end_reason="completed"),
            [item(idem_key="k1")], now=600.0,
        )
        state = await repo.get_chat_state(CK)
        await repo.close()
        return state, grant

    state, grant = run(scenario())
    assert state is not None
    assert state.agent_resume_at is None  # barrier cleared
    assert state.wait_streak == 0  # streak reset
    assert state.cursor_msg_id == grant.through_msg_id  # cursor advanced


def test_defer_detaches_commits_and_records_barrier(tmp_path):
    """A defer atomically detaches the attached commits (they stay pending)
    and records the barrier; the released dispatch stays replayable."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        assert grant.attached == (CommitSeq(1),)
        await repo.settle_dispatch(
            make_settle(grant.dispatch_id, outcome="defer",
                        resume_at=300.0, defer_kind="wait"),
            [], now=100.0,
        )
        unassigned = await repo.list_unassigned_commits(CK)
        state = await repo.get_chat_state(CK)
        dispatch_state = await repo._db.read(
            lambda c: c.execute(
                "SELECT state FROM dispatches WHERE id = ?", (grant.dispatch_id,)
            ).fetchone()
        )
        await repo.close()
        return unassigned, state, dispatch_state

    unassigned, state, dispatch_state = run(scenario())
    assert unassigned == [CommitSeq(1)]  # detached, still pending
    assert state is not None and state.agent_resume_at == 300.0
    assert state.wait_streak == 1
    assert dispatch_state[0] == "released"  # replayable released dispatch


def test_renew_dispatch_extends_lease_fenced(tmp_path):
    """renew_dispatch extends a prepared dispatch's lease, fenced to the
    same unexpired prepared owner with a finite forward extension."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.begin_dispatch(make_request())
        assert isinstance(grant, DispatchGrant)
        ok = await repo.renew_dispatch(CK, grant.dispatch_id, "cy-1", 999.0, now=200.0)
        wrong_owner = await repo.renew_dispatch(
            CK, grant.dispatch_id, "cy-other", 999.0, now=200.0
        )
        wrong_id = await repo.renew_dispatch(
            CK, DispatchId(999), "cy-1", 999.0, now=200.0
        )
        expires = await repo._db.read(
            lambda c: c.execute(
                "SELECT expires_at FROM dispatches WHERE id = ?", (grant.dispatch_id,)
            ).fetchone()
        )
        await repo.close()
        return ok, wrong_owner, wrong_id, expires

    ok, wrong_owner, wrong_id, expires = run(scenario())
    assert ok and not wrong_owner and not wrong_id
    assert expires[0] == 999.0


def test_renew_dispatch_rejects_stale_or_nonfinite(tmp_path):
    """An expired owner cannot renew even before another claimant acts, and
    a non-finite or non-forward extension is rejected."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), make_message(msg_id="m1"))
        grant = await repo.begin_dispatch(
            make_request(started_ts=50.0, expires_at=100.0, now=50.0)
        )
        assert isinstance(grant, DispatchGrant)
        # Expired owner: cannot renew.
        expired = await repo.renew_dispatch(CK, grant.dispatch_id, "cy-1", 999.0, now=150.0)
        # Non-finite / non-forward extensions are rejected.
        inf = await repo.renew_dispatch(CK, grant.dispatch_id, "cy-1", float("inf"), now=150.0)
        nan = await repo.renew_dispatch(CK, grant.dispatch_id, "cy-1", float("nan"), now=150.0)
        past = await repo.renew_dispatch(CK, grant.dispatch_id, "cy-1", 100.0, now=150.0)
        await repo.close()
        return expired, inf, nan, past

    assert run(scenario()) == (False, False, False, False)
