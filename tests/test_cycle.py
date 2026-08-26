"""CycleRunner + snapshot assembler: deterministic claim->snapshot->
disposition with real temp SQLite and VirtualClock.

Covers the frozen Phase 2 lifecycle: claim-before-read with finally
release (exceptions, cancellation, lease expiry); the pure Gate applying
hold precedence after claiming (direct @ / quote, focus, and high pending
bypass an active hold; an expired hold never regenerates); ordinary delay
releases with no cursor/session mutation; refusal skip and dry-run trigger
terminally finish with an empty outbox (cursor consumed, hold cleared,
idle reset, trace persisted, hook emitted); a trigger claim is never
retained without an agent; durable quote-target resolution beyond the
rendered window; timing/UUID injectability; and the pure direct-address /
self-ratio / idle / backoff-config derivations.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest

from pretender.clock import VirtualClock
from pretender.config import Config
from pretender.cycle import CycleRunner, assemble_snapshot, replay_corpus
from pretender.errors import ClaimError
from pretender.gate import Gate
from pretender.registry import HookBus
from pretender.types import (
    AdapterEvent,
    ChatState,
    ClaimGrant,
    CommitSeq,
    CycleClaim,
    CycleId,
    DispatchCause,
    DispatchGrant,
    DispatchId,
    DispatchRequest,
    DispatchSettle,
    GateSnapshot,
    Message,
    MessageId,
    MessageRowId,
    Reason,
    RecentSnapshot,
    SelfId,
    SenderId,
)
from tests.durable_helpers import (
    CK,
    finish_batch,
    make_claim,
    make_identity,
    make_message,
    open_repo,
    run,
)

SELF = SelfId("bot-1")
SELF_SENDER = SenderId("bot-1")


def _msg(
    text: str = "hi",
    *,
    msg_id: str = "m1",
    sender_id: str = "u1",
    is_self: bool = False,
    recv_ts: float = 150.0,
    reply_to: str | None = None,
    mentions: tuple[str, ...] = (),
) -> Message:
    return Message(
        chat_key=CK,
        sender_id=SenderId(sender_id),
        sender_name="user",
        is_self=is_self,
        text=text,
        id=MessageId(msg_id),
        reply_to=MessageId(reply_to) if reply_to else None,
        mentions=tuple(SenderId(m) for m in mentions),
        recv_ts=recv_ts,
    )


def _grant(pending: tuple[Message, ...] = (), *, through: int = 1) -> ClaimGrant:
    return ClaimGrant(
        claim=CycleClaim(CK, CycleId("c1"), started_ts=100.0, expires_at=200.0),
        start_msg_id=MessageRowId(0),
        through_msg_id=MessageRowId(through),
        pending=pending,
    )


def _recent(
    messages: tuple[Message, ...] = (),
    *,
    window_count: int | None = None,
    self_count: int = 0,
    last_nonself_ts: float | None = 150.0,
) -> RecentSnapshot:
    return RecentSnapshot(
        chat_key=CK,
        messages=messages,
        window_count=window_count if window_count is not None else len(messages),
        self_count=self_count,
        last_nonself_ts=last_nonself_ts,
        since_ts=0.0,
        through_row_id=MessageRowId(1),
    )


def _snap(**kw: Any) -> GateSnapshot:
    """Assemble a snapshot with defaults; keyword overrides pass through."""
    base: dict[str, Any] = dict(
        grant=_grant(),
        identity=make_identity(self_id="bot-1"),
        state=ChatState(chat_key=CK),
        recent=_recent(),
        cfg=Config().for_chat(CK),
        now=200.0,
        self_name="麦麦",
        previous_end_reason=None,
    )
    base.update(kw)
    return cast(GateSnapshot, assemble_snapshot(**base))


def make_runner(repo, *, clock=None, hooks=None, dry_run=True, uuid_fn=None, **kw):
    return CycleRunner(
        repo,
        Gate(),
        Config(),
        clock=clock if clock is not None else VirtualClock(),
        hooks=hooks,
        dry_run=dry_run,
        uuid_fn=uuid_fn,
        **kw,
    )


# ── snapshot assembler: structured derivations ──────────────────────────────

def test_assemble_direct_at_from_mentions():
    snap = _snap(grant=_grant(pending=(_msg(mentions=("bot-1",)),)))
    assert snap.has_direct_at is True
    assert snap.self_id == SELF


def test_assemble_no_direct_at_for_other_mentions():
    snap = _snap(grant=_grant(pending=(_msg(mentions=("someone-else",)),)))
    assert snap.has_direct_at is False


def test_assemble_quote_to_self_resolves_reply_target():
    self_msg = _msg(text="bot reply", msg_id="s1", is_self=True)
    reply = _msg(text="thanks", msg_id="m2", reply_to="s1")
    snap = _snap(
        grant=_grant(pending=(reply,), through=2),
        recent=_recent((reply, self_msg), window_count=2, self_count=1),
    )
    assert snap.has_quote_to_self is True


def test_assemble_quote_to_unresolvable_target_is_not_self():
    reply = _msg(text="thanks", msg_id="m2", reply_to="gone")
    snap = _snap(grant=_grant(pending=(reply,), through=2), recent=_recent((reply,)))
    assert snap.has_quote_to_self is False


def test_assemble_other_assistant_through_signals():
    snap = _snap(grant=_grant(pending=(_msg(text="DeepSeek，你好"),)))
    assert snap.has_other_assistant is True


def test_assemble_self_name_from_bot_config():
    snap = _snap(grant=_grant(pending=(_msg(text="麦麦 在吗"),)))
    assert snap.self_name == "麦麦"


def test_assemble_exact_full_window_self_ratio():
    snap = _snap(
        grant=_grant(pending=(_msg(),), through=4),
        recent=_recent(
            (_msg(msg_id="m4"), _msg(msg_id="s1", is_self=True)),
            window_count=4,
            self_count=1,
        ),
    )
    assert snap.self_ratio == 0.25
    assert snap.window_count == 4
    assert snap.self_count == 1


def test_assemble_empty_window_ratio_is_zero():
    snap = _snap(recent=_recent((), window_count=0, self_count=0, last_nonself_ts=None))
    assert snap.self_ratio == 0.0


def test_assemble_idle_seconds_from_last_nonself():
    snap = _snap(now=200.0, recent=_recent(last_nonself_ts=150.0))
    assert snap.idle_seconds == 50.0


def test_assemble_idle_seconds_without_nonself_is_window():
    snap = _snap(now=200.0, recent=_recent((), window_count=0, last_nonself_ts=None))
    assert snap.idle_seconds == 300.0  # idle for at least the whole window


def test_assemble_fixed_bounds_and_config():
    cfg = Config.from_dict(
        {"gate": {"mode": "frequency", "threshold": 12, "trigger_score": 60,
                  "frequency": 0.5}}
    )
    snap = _snap(grant=_grant(through=7), cfg=cfg.for_chat(CK), now=250.0)
    assert snap.start_msg_id == 0
    assert snap.through_msg_id == 7
    assert snap.evaluated_ts == 250.0
    assert snap.cycle_id == "c1"
    assert snap.mode == "frequency"
    assert snap.threshold == 12
    assert snap.trigger_score == 60
    assert snap.frequency == 0.5


def test_assemble_focus_and_hold_facts():
    state = ChatState(chat_key=CK, focus_until=300.0, hold_until=400.0, idle_streak=3)
    snap = _snap(state=state, now=200.0)
    assert snap.is_focused is True
    assert snap.hold_until == 400.0
    assert snap.idle_streak == 3


def test_assemble_previous_end_reason_and_avg():
    state = ChatState(chat_key=CK, avg_interval=60.0)
    snap = _snap(state=state, previous_end_reason="skip")
    assert snap.previous_end_reason == "skip"
    assert snap.recent_average_interval == 60.0


def test_assemble_pending_count_matches_claimed_tuple():
    snap = _snap(grant=_grant(pending=(_msg(), _msg(msg_id="m2"))))
    assert snap.pending == 2
    assert len(snap.pending_messages) == 2


def test_assemble_is_group_from_identity_kind():
    snap = _snap(identity=make_identity(kind="private"))
    assert snap.is_group is False


# ── CycleRunner: dispositions against real SQLite ───────────────────────────

def test_cycle_delay_releases_claim_without_cursor_mutation(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        clock = VirtualClock(epoch=200.0)
        runner = make_runner(repo, clock=clock, uuid_fn=lambda: "cy-1")
        decision = await runner(CK)
        cursor = await db.read(
            lambda c: c.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?", (CK,)
            ).fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
        )
        claims = await db.read(
            lambda c: [r[0] for r in c.execute("SELECT state FROM claims")]
        )
        await repo.close()
        return decision, cursor, cycles, claims

    decision, cursor, cycles, claims = run(scenario())
    assert decision.action == "delay"
    assert decision.reason == Reason.DELAY
    assert decision.delay_seconds is None  # event-only: no average yet
    assert cursor is None  # no cursor movement (never set)
    assert cycles == 0  # no terminal cycle
    assert claims == ["released"]  # claim released, nothing else


def test_cycle_refusal_skip_finishes_atomically(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(
            make_identity(), make_message(text="DeepSeek，你好", recv_ts=100.0)
        )
        clock = VirtualClock(epoch=200.0)
        hooks = HookBus()
        seen: list[tuple] = []

        @hooks.on_cycle_end
        def hook(chat_key, trace, end_reason):
            seen.append((chat_key, trace.decision.action, end_reason))

        runner = make_runner(
            repo, clock=clock, hooks=hooks, dry_run=False, uuid_fn=lambda: "cy-1"
        )
        decision = await runner(CK)
        cursor = await db.read(
            lambda c: c.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?", (CK,)
            ).fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason, trace_json FROM cycles").fetchall()
        )
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        state = await repo.get_chat_state(CK)
        await repo.close()
        assert state is not None
        return decision, cursor, cycles, outbox, state, seen

    decision, cursor, cycles, outbox, state, seen = run(scenario())
    assert decision.action == "skip"
    assert decision.reason == Reason.REFUSAL
    assert cursor == 1  # boundary consumed
    assert cycles[0][0] == "skip"
    assert json.loads(cycles[0][1])["decision"]["reason"] == "refusal"
    assert outbox == 0  # empty outbox
    assert state.idle_streak == 0  # idle reset
    assert state.hold_until is None  # hold cleared
    assert seen == [(CK, "skip", "skip")]  # hook after terminal completion (LIVE)


def test_cycle_dry_run_trigger_finishes_with_empty_outbox(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        msg = Message(
            chat_key=CK,
            sender_id=SenderId("u1"),
            sender_name="user",
            is_self=False,
            text="hi",
            id=MessageId("m1"),
            mentions=(SELF_SENDER,),
            recv_ts=100.0,
        )
        await repo.ingest_message(make_identity(), msg)
        clock = VirtualClock(epoch=200.0)
        hooks = HookBus()
        seen: list[tuple] = []

        @hooks.on_cycle_end
        def hook(chat_key, trace, end_reason):
            seen.append((chat_key, trace.decision.action, end_reason))

        runner = make_runner(repo, clock=clock, hooks=hooks, uuid_fn=lambda: "cy-1")
        decision = await runner(CK)
        cursor = await db.read(
            lambda c: c.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?", (CK,)
            ).fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        await repo.close()
        return decision, cursor, cycles, outbox, seen

    decision, cursor, cycles, outbox, seen = run(scenario())
    assert decision.action == "trigger"
    assert decision.reason == Reason.TRIGGER
    assert decision.score >= 100  # hard trigger floor
    assert cursor == 1
    assert cycles == [("dry_run_trigger",)]
    assert outbox == 0
    # Phase 6 P6.6: hooks NEVER run in dry-run — the dry-run lane is
    # deterministic and plugin-free.
    assert seen == []


def test_cycle_held_chat_claims_and_gate_applies_remaining_hold(tmp_path):
    """An active durable hold no longer short-circuits the runner: the
    cycle CLAIMS, the pure Gate applies the hold precedence (delay for
    the remaining duration), and the claim is released — so direct @ /
    quote, focus, and high pending can bypass the hold."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        # A durable hold from a previous terminal cycle (hold_until=500).
        await finish_batch(repo, [], chat_key=CK, hold_until=500.0, now=200.0)
        clock = VirtualClock(epoch=300.0)
        runner = make_runner(repo, clock=clock, uuid_fn=lambda: "cy-2")
        decision = await runner(CK)
        claims = await db.read(
            lambda c: [tuple(r) for r in c.execute("SELECT cycle_id, state FROM claims")]
        )
        await repo.close()
        return decision, claims

    decision, claims = run(scenario())
    assert decision.action == "delay"
    assert decision.delay_seconds == 200.0  # remaining hold: 500 - 300
    assert decision.reason == Reason.BACKOFF
    assert claims == [("cy-1", "finished"), ("cy-2", "released")]


def test_cycle_trigger_without_agent_releases_claim(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        msg = Message(
            chat_key=CK,
            sender_id=SenderId("u1"),
            sender_name="user",
            is_self=False,
            text="hi",
            id=MessageId("m1"),
            mentions=(SELF_SENDER,),
            recv_ts=100.0,
        )
        await repo.ingest_message(make_identity(), msg)
        clock = VirtualClock(epoch=200.0)
        runner = make_runner(repo, clock=clock, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner(CK)
        claims = await db.read(
            lambda c: [r[0] for r in c.execute("SELECT state FROM claims")]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
        )
        await repo.close()
        return decision, claims, cycles

    decision, claims, cycles = run(scenario())
    assert decision.action == "trigger"
    assert claims == ["released"]  # never retained without an agent
    assert cycles == 0  # no terminal completion


def test_cycle_unknown_chat_returns_skip_without_claim(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")  # no chat row
        runner = make_runner(repo, uuid_fn=lambda: "cy-1")
        decision = await runner(CK)
        claims = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        )
        await repo.close()
        return decision, claims

    decision, claims = run(scenario())
    assert decision.action == "skip"
    assert decision.reason == Reason.SKIP
    assert claims == 0


def test_cycle_injectable_uuid_and_clock(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        clock = VirtualClock(epoch=200.0)
        runner = make_runner(repo, clock=clock, uuid_fn=lambda: "fixed-id")
        decision = await runner(CK)
        claim = await db.read(
            lambda c: c.execute(
                "SELECT cycle_id, started_ts, expires_at FROM claims"
            ).fetchone()
        )
        await repo.close()
        return decision, claim

    decision, claim = run(scenario())
    assert decision.action == "delay"
    assert claim[0] == "fixed-id"  # injected cycle id
    assert claim[1] == 200.0  # injected clock: started at now
    assert claim[2] == 260.0  # finite lease: now + 60


def test_cycle_trace_persists_full_facts(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(
            make_identity(), make_message(text="DeepSeek，你好", recv_ts=100.0)
        )
        clock = VirtualClock(epoch=200.0)
        runner = make_runner(repo, clock=clock, uuid_fn=lambda: "cy-1")
        await runner(CK)
        trace_json = await db.read(
            lambda c: c.execute("SELECT trace_json FROM cycles").fetchone()[0]
        )
        await repo.close()
        return json.loads(trace_json)

    trace = run(scenario())
    assert trace["chat_key"] == CK
    assert trace["mode"] == "reply_necessity"
    assert trace["threshold"] == 8
    assert trace["pending"] == 1
    assert trace["decision"]["reason"] == "refusal"
    assert trace["snapshot_facts"]["has_other_assistant"] is True
    assert trace["snapshot_facts"]["through_msg_id"] == 1
    assert trace["config"]["trigger_score"] == 80
    assert trace["aggregates"]["score"] == trace["decision"]["score"]
    assert trace["backoff"]["bypass_reason"] == "non_idle"


def test_cycle_returns_decision_suitable_for_rearming(tmp_path):
    """The returned Decision is exactly what the scheduler re-arms on: a
    session-maintained average yields a timed delay, otherwise the delay
    is event-only; terminal outcomes are event-only."""

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=150.0))
        # A session-maintained average (the agent phase updates it; the
        # runner only reads it).
        await repo.upsert_chat_state(ChatState(chat_key=CK, avg_interval=60.0))
        clock = VirtualClock(epoch=200.0)
        runner = make_runner(repo, clock=clock, uuid_fn=lambda: "cy-1")
        decision = await runner(CK)
        await repo.close()
        return decision

    decision = run(scenario())
    assert decision.action == "delay"
    assert decision.delay_seconds == 10.0  # timed: avg 60 - idle 50

# ── claim-before-read / finally-release ─────────────────────────────────────

class RecordingRepo:
    """A minimal protocol-shaped repo recording call order with scripted
    results for the runner's reads (claim-order and claim-race tests)."""

    def __init__(
        self,
        *,
        claim_result: ClaimGrant | None = None,
        identity: Any = None,
        state: ChatState | None = None,
        recent: RecentSnapshot | None = None,
        end_reason: str | None = None,
        messages: dict[str, Message] | None = None,
    ) -> None:
        self.calls: list[tuple] = []
        self.claim_result = claim_result
        self.identity = identity
        self.state = state
        self.recent = recent
        self.end_reason = end_reason
        self.messages = messages or {}

    async def claim_cycle(self, claim: CycleClaim):
        self.calls.append(("claim_cycle", claim))
        return self.claim_result

    async def get_chat(self, chat_key):
        self.calls.append(("get_chat", chat_key))
        return self.identity

    async def get_chat_state(self, chat_key):
        self.calls.append(("get_chat_state", chat_key))
        return self.state

    async def get_latest_terminal_end_reason(self, chat_key):
        self.calls.append(("get_latest_terminal_end_reason", chat_key))
        return self.end_reason

    async def get_recent_snapshot(self, chat_key, through_row_id, since_ts, limit):
        self.calls.append(
            ("get_recent_snapshot", chat_key, through_row_id, since_ts, limit)
        )
        return self.recent

    async def get_message(self, chat_key, msg_id):
        self.calls.append(("get_message", chat_key, msg_id))
        return self.messages.get(msg_id)

    async def release_cycle(self, chat_key, cycle_id):
        self.calls.append(("release_cycle", chat_key, cycle_id))

    async def finish_cycle(self, finish, outbox, *, now):
        self.calls.append(("finish_cycle", finish, outbox, now))


def test_cycle_claims_before_reading_any_state():
    """The claim comes FIRST: identity/state/recent/history are read only
    after the claim succeeded, so the snapshot is claim-bounded."""
    repo = RecordingRepo(
        claim_result=_grant(pending=(_msg(),)),
        identity=make_identity(),
        state=ChatState(chat_key=CK),
        recent=_recent(),
    )
    runner = make_runner(repo, uuid_fn=lambda: "cy-1")
    decision = run(runner(CK))
    kinds = [c[0] for c in repo.calls]
    assert kinds[0] == "claim_cycle"
    assert kinds.index("claim_cycle") < kinds.index("get_chat")
    assert kinds.index("claim_cycle") < kinds.index("get_chat_state")
    assert kinds.index("claim_cycle") < kinds.index("get_recent_snapshot")
    assert kinds.index("claim_cycle") < kinds.index("get_latest_terminal_end_reason")
    assert decision.action == "delay"
    assert ("release_cycle", CK, "cy-1") in repo.calls


def test_cycle_claim_race_returns_skip_without_state_reads():
    """A racing claim (another live cycle owns the chat) is detected
    BEFORE any state is read: the runner returns skip immediately."""
    repo = RecordingRepo(claim_result=None)
    runner = make_runner(repo, uuid_fn=lambda: "cy-1")
    decision = run(runner(CK))
    assert decision.action == "skip"
    assert decision.reason == Reason.SKIP
    assert [c[0] for c in repo.calls] == ["claim_cycle"]


# ── hold precedence: direct @ / quote / focus / high pending ────────────────

def test_cycle_direct_at_bypasses_active_hold(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        # A durable hold from a previous terminal cycle (hold_until=500).
        await finish_batch(repo, [], chat_key=CK, hold_until=500.0, now=200.0)
        # A direct @ arrives during the hold.
        msg = Message(
            chat_key=CK, sender_id=SenderId("u1"), sender_name="user",
            is_self=False, text="hi", id=MessageId("m2"),
            mentions=(SELF_SENDER,), recv_ts=300.0,
        )
        await repo.ingest_message(make_identity(), msg)
        clock = VirtualClock(epoch=300.0)
        runner = make_runner(repo, clock=clock, uuid_fn=lambda: "cy-2")
        decision = await runner(CK)
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        await repo.close()
        return decision, cycles

    decision, cycles = run(scenario())
    assert decision.action == "trigger"
    assert decision.reason == Reason.TRIGGER
    assert decision.score >= 100  # hard trigger floor
    assert cycles == [("completed",), ("dry_run_trigger",)]  # hold did not block


def test_cycle_older_quote_to_self_bypasses_active_hold(tmp_path):
    """A quote of an OLD self message (outside the rendered window) still
    triggers during an active hold: the runner resolves every distinct
    pending reply target through the durable repository."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        # An OLD self message (outside the 300 s window) the quote targets.
        await repo.ingest_message(
            make_identity(), make_message(text="bot reply", msg_id="s1",
                                          is_self=True, recv_ts=100.0)
        )
        await repo.ingest_message(make_identity(), make_message(recv_ts=150.0))
        await finish_batch(repo, [], chat_key=CK, hold_until=500.0, now=200.0)
        # A quote of the OLD self message arrives during the hold.
        msg = Message(
            chat_key=CK, sender_id=SenderId("u1"), sender_name="user",
            is_self=False, text="thanks", id=MessageId("m3"),
            reply_to=MessageId("s1"), recv_ts=450.0,
        )
        await repo.ingest_message(make_identity(), msg)
        clock = VirtualClock(epoch=450.0)
        runner = make_runner(repo, clock=clock, uuid_fn=lambda: "cy-2")
        decision = await runner(CK)
        await repo.close()
        return decision

    decision = run(scenario())
    assert decision.action == "trigger"
    assert decision.reason == Reason.TRIGGER
    assert decision.score >= 100


def test_cycle_focus_bypasses_active_hold(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        await finish_batch(repo, [], chat_key=CK, hold_until=500.0, now=200.0)
        # A pending message plus an active focus window during the hold.
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", recv_ts=250.0)
        )
        await repo.upsert_chat_state(ChatState(chat_key=CK, focus_until=400.0))
        clock = VirtualClock(epoch=300.0)
        traces: list = []
        runner = make_runner(repo, clock=clock, uuid_fn=lambda: "cy-2",
                             trace_sink=traces.append)
        decision = await runner(CK)
        await repo.close()
        return decision, traces[0].backoff

    decision, backoff = run(scenario())
    assert backoff.bypass_reason == "focus"
    assert decision.action == "delay"
    assert decision.reason == Reason.DELAY  # mode selection, not backoff


def test_cycle_high_pending_bypasses_active_hold(tmp_path):
    cfg = Config.from_dict({"gate": {"threshold": 2}})

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        await finish_batch(repo, [], chat_key=CK, hold_until=500.0, now=200.0)
        for i in range(2):
            await repo.ingest_message(
                make_identity(), make_message(msg_id=f"m{i + 2}", recv_ts=250.0 + i)
            )
        clock = VirtualClock(epoch=300.0)
        traces: list = []
        runner = CycleRunner(repo, Gate(), cfg, clock=clock, uuid_fn=lambda: "cy-2",
                             trace_sink=traces.append)
        decision = await runner(CK)
        await repo.close()
        return decision, traces[0].backoff

    decision, backoff = run(scenario())
    assert backoff.bypass_reason == "high_pending"
    assert decision.action == "delay"
    assert decision.reason == Reason.DELAY  # mode selection, not backoff


def test_cycle_expired_hold_never_regenerates_fresh_hold(tmp_path):
    """An EXPIRED durable hold is ignored: the backoff controller is never
    consulted, so no fresh hold/delay is regenerated from the previous
    idle end reason and idle streak."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        # An EXPIRED hold + idle history that WOULD regenerate a fresh
        # hold if the controller were consulted.
        await finish_batch(
            repo, [], chat_key=CK, hold_until=200.0, now=200.0,
            end_reason="planner_no_tool_end", idle_streak_after=2,
        )
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", recv_ts=250.0)
        )
        clock = VirtualClock(epoch=300.0)
        traces: list = []
        runner = make_runner(repo, clock=clock, uuid_fn=lambda: "cy-2",
                             trace_sink=traces.append)
        decision = await runner(CK)
        await repo.close()
        return decision, traces[0].backoff

    decision, backoff = run(scenario())
    assert backoff.bypass_reason == "expired_hold"
    assert decision.action == "delay"
    assert decision.reason == Reason.DELAY  # never BACKOFF
    # The delay is the MODE delay (avg 150 - idle 50), never a fresh hold.


# ── exception / cancellation / lease-expiry release ─────────────────────────

class _ExplodingRepo:
    """Delegates everything to a real repo but raises in one method."""

    def __init__(self, repo, *, explode_in: str) -> None:
        self._repo = repo
        self._explode_in = explode_in

    def __getattr__(self, name: str) -> Any:
        return getattr(self._repo, name)

    async def get_recent_snapshot(self, chat_key, through_row_id, since_ts, limit):
        raise RuntimeError("boom")


def test_cycle_exception_after_claim_releases_and_recovers(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        clock = VirtualClock(epoch=200.0)
        runner = make_runner(_ExplodingRepo(repo, explode_in="get_recent_snapshot"),
                             clock=clock, uuid_fn=lambda: "cy-1")
        with pytest.raises(RuntimeError, match="boom"):
            await runner(CK)
        claims = await db.read(
            lambda c: [r[0] for r in c.execute("SELECT state FROM claims")]
        )
        # The unfinished claim was released, not stranded.
        assert claims == ["released"]
        # A later cycle recovers: the same chat claims and evaluates fine.
        runner2 = make_runner(repo, clock=clock, uuid_fn=lambda: "cy-2")
        decision = await runner2(CK)
        await repo.close()
        return decision

    decision = run(scenario())
    assert decision.action == "delay"


def test_cycle_cancellation_after_claim_releases_and_recovers(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        clock = VirtualClock(epoch=200.0)
        started = asyncio.Event()
        release = asyncio.Event()

        class _HoldingRepo:
            def __init__(self, repo: Any) -> None:
                self._repo = repo

            def __getattr__(self, name: str) -> Any:
                return getattr(self._repo, name)

            async def get_recent_snapshot(self, chat_key, through_row_id, since_ts, limit):
                started.set()
                await release.wait()
                return await self._repo.get_recent_snapshot(
                    chat_key, through_row_id, since_ts, limit
                )

        runner = make_runner(_HoldingRepo(repo), clock=clock, uuid_fn=lambda: "cy-1")
        task = asyncio.create_task(runner(CK))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        claims = await db.read(
            lambda c: [r[0] for r in c.execute("SELECT state FROM claims")]
        )
        assert claims == ["released"]  # cancellation never strands the claim
        # Later recovery: a fresh cycle claims and evaluates.
        runner2 = make_runner(repo, clock=clock, uuid_fn=lambda: "cy-2")
        decision = await runner2(CK)
        await repo.close()
        return decision

    decision = run(scenario())
    assert decision.action == "delay"


def test_cycle_finalization_after_lease_expiry_fails_and_releases(tmp_path):
    """Terminal completion fences against a FRESH clock timestamp: a lease
    that expired while the cycle ran rejects the finish (ClaimError), the
    claim is released by the finally, and a later cycle recovers."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        msg = Message(
            chat_key=CK, sender_id=SenderId("u1"), sender_name="user",
            is_self=False, text="hi", id=MessageId("m1"),
            mentions=(SELF_SENDER,), recv_ts=100.0,
        )
        await repo.ingest_message(make_identity(), msg)
        clock = VirtualClock(epoch=200.0)

        class _SlowRepo:
            """Advances the clock past the lease mid-cycle (before the
            runner's fresh finish timestamp is read)."""

            def __init__(self, repo: Any, clock: VirtualClock) -> None:
                self._repo = repo
                self._clock = clock

            def __getattr__(self, name: str) -> Any:
                return getattr(self._repo, name)

            async def get_recent_snapshot(self, chat_key, through_row_id, since_ts, limit):
                self._clock.advance(120.0)  # the 60 s lease is now expired
                return await self._repo.get_recent_snapshot(
                    chat_key, through_row_id, since_ts, limit
                )

        runner = make_runner(_SlowRepo(repo, clock), clock=clock,
                             uuid_fn=lambda: "cy-1")
        with pytest.raises(ClaimError):
            await runner(CK)
        claims = await db.read(
            lambda c: [r[0] for r in c.execute("SELECT state FROM claims")]
        )
        assert claims == ["released"]  # not stranded
        # Recovery: a later cycle claims and finishes fine.
        runner2 = make_runner(repo, clock=clock, uuid_fn=lambda: "cy-2")
        decision = await runner2(CK)
        await repo.close()
        return decision

    decision = run(scenario())
    assert decision.action == "trigger"


# ── exact per-chat backoff configuration ────────────────────────────────────

def test_assemble_backoff_facts_from_per_chat_config():
    cfg = Config.from_dict(
        {"gate": {"backoff": {"base_s": 30.0, "cap_s": 600.0, "start_count": 3}}}
    )
    snap = _snap(cfg=cfg.for_chat(CK))
    assert snap.backoff_base_s == 30.0
    assert snap.backoff_cap_s == 600.0
    assert snap.backoff_start_count == 3


def test_cycle_trace_carries_exact_per_chat_backoff_config(tmp_path):
    cfg = Config.from_dict(
        {"gate": {"backoff": {"base_s": 30.0, "cap_s": 600.0, "start_count": 3}}}
    )

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        clock = VirtualClock(epoch=200.0)
        traces: list = []
        runner = CycleRunner(repo, Gate(), cfg, clock=clock, uuid_fn=lambda: "cy-1",
                             trace_sink=traces.append)
        await runner(CK)
        await repo.close()
        return traces[0].config

    config = run(scenario())
    assert config["backoff"] == {
        "base_s": 30.0, "cap_s": 600.0, "start_count": 3, "threshold": 8,
    }


# ── ClaimBusy: the busy-horizon timed delay ─────────────────────────────────

def test_cycle_busy_claim_returns_timed_delay_without_trace(tmp_path):
    """A live, unexpired claim (e.g. a crash mid-cycle) maps to a TIMED
    delay at the exact busy_until horizon: the scheduler re-arms, the
    next wake recovers the expired claim. Never a terminal skip, never a
    false trace, and the owner's claim is untouched."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        # A live claim from a crashed cycle (lease expires at 500).
        grant = await repo.claim_cycle(
            make_claim(cycle_id="crash-1", started_ts=100.0, expires_at=500.0)
        )
        assert grant is not None
        clock = VirtualClock(epoch=300.0)
        traces: list = []
        runner = make_runner(repo, clock=clock, uuid_fn=lambda: "cy-2",
                             trace_sink=traces.append)
        decision = await runner(CK)
        claims = await db.read(
            lambda c: [tuple(r) for r in c.execute("SELECT cycle_id, state FROM claims")]
        )
        await repo.close()
        return decision, traces, claims

    decision, traces, claims = run(scenario())
    assert decision.action == "delay"
    assert decision.reason == Reason.DELAY
    assert decision.delay_seconds == 200.0  # busy_until 500 - now 300
    assert traces == []  # no false trace
    assert claims == [("crash-1", "live")]  # the owner's claim untouched


def test_cycle_busy_claim_recovers_after_horizon_without_new_input(tmp_path):
    """At/after the busy_until horizon the expired claim is recovered and
    the cycle gets a grant — no new input required."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        await repo.claim_cycle(
            make_claim(cycle_id="crash-1", started_ts=100.0, expires_at=500.0)
        )
        clock = VirtualClock(epoch=300.0)
        runner = make_runner(repo, clock=clock, uuid_fn=lambda: "cy-2")
        first = await runner(CK)  # busy: timed delay to the horizon
        clock.advance(200.0)  # now 500: the lease expired
        second = await runner(CK)  # recovered: a grant, evaluated
        claims = await db.read(
            lambda c: [tuple(r) for r in c.execute("SELECT cycle_id, state FROM claims")]
        )
        await repo.close()
        return first, second, claims

    first, second, claims = run(scenario())
    assert first.action == "delay"
    assert first.delay_seconds == 200.0
    assert second.action == "delay"  # the recovered grant evaluated
    assert claims == [("crash-1", "expired"), ("cy-2", "released")]


def test_replay_file_order_matches_durable_claim_boundary(tmp_path):
    """The replay's file-order evaluation mirrors the live claim-bounded
    grant for the same corpus: row ids follow commit order, so a receding
    timestamp commits at the current time and the claim's through boundary
    covers every committed row (never a recv_ts reorder)."""
    clock = VirtualClock(auto_advance=False)
    cfg = Config.from_dict({"storage": {"db_path": str(tmp_path / "p.db")}})

    async def scenario():
        db, repo = await open_repo(tmp_path / "p.db")
        await repo.upsert_chat(make_identity())
        # Commit in FILE order with a receding timestamp: row ids 1, 2.
        m1 = make_message(text="a", msg_id="m1", recv_ts=clock.now())
        await repo.ingest_message(make_identity(), m1)
        m2 = make_message(text="b", msg_id="m2", recv_ts=clock.now() - 50.0)
        await repo.ingest_message(make_identity(), m2)
        grant = await repo.claim_cycle(
            make_claim(started_ts=clock.now(), expires_at=clock.now() + 60.0)
        )
        await repo.close()
        assert isinstance(grant, ClaimGrant)
        return grant, m1, m2

    grant, m1, m2 = run(scenario())
    events = [
        AdapterEvent(type="message", payload=m1, ts=m1.recv_ts),
        AdapterEvent(type="message", payload=m2, ts=m2.recv_ts),
    ]
    result = replay_corpus(events, chat_key=CK, identity=make_identity(), cfg=cfg)
    # The live claim covers BOTH committed rows (through = 2) even though
    # m2's recv_ts recedes; the replay's single evaluation at the current
    # time has the same boundary and pending set.
    assert grant.through_msg_id == MessageRowId(2)
    assert len(grant.pending) == 2
    assert result.decisions == 1
    facts = result.traces[0].snapshot_facts
    assert facts["through_msg_id"] == 2
    assert facts["pending"] == 2


# ── run_dispatch: the dispatch-ledger cycle (LedgerScheduler handler) ───────

def _dispatch_grant(
    pending: tuple[Message, ...] = (),
    *,
    dispatch_id: int = 1,
    through: int = 1,
    started_ts: float = 100.0,
    expires_at: float = 500.0,
    commit_boundary: int = 1,
    scheduled_for: float | None = None,
) -> DispatchGrant:
    return DispatchGrant(
        dispatch_id=DispatchId(dispatch_id),
        claim=CycleClaim(CK, CycleId("c1"), started_ts, expires_at),
        start_msg_id=MessageRowId(0),
        through_msg_id=MessageRowId(through),
        pending=pending,
        commit_boundary=CommitSeq(commit_boundary),
        scheduled_for=scheduled_for,
    )


def make_dispatch_runner(
    repo, *, clock=None, hooks=None, dry_run=True, uuid_fn=None,
    marker_exporter=None, **kw,
):
    return CycleRunner(
        repo, Gate(), Config(),
        clock=clock if clock is not None else VirtualClock(epoch=200.0),
        hooks=hooks, dry_run=dry_run, uuid_fn=uuid_fn,
        marker_exporter=marker_exporter, **kw,
    )


async def _begin_dispatch(
    repo,
    *,
    cause: str = DispatchCause.INBOUND,
    cycle_id: str = "cy-1",
    started_ts: float = 200.0,
    expires_at: float = 500.0,
    now: float = 200.0,
    scheduled_ts: float | None = None,
) -> DispatchGrant:
    grant = await repo.begin_dispatch(
        DispatchRequest(
            chat_key=CK, cause=cause, cycle_id=CycleId(cycle_id),
            started_ts=started_ts, expires_at=expires_at, now=now,
            scheduled_ts=scheduled_ts,
        )
    )
    assert isinstance(grant, DispatchGrant)
    return grant


class DispatchRecordingRepo:
    """Records every call; raises if the legacy claim surface is touched."""

    def __init__(
        self,
        *,
        identity: Any = None,
        state: ChatState | None = None,
        recent: RecentSnapshot | None = None,
        end_reason: str | None = None,
        messages: dict[str, Message] | None = None,
    ) -> None:
        self.calls: list[tuple] = []
        self.settles: list[DispatchSettle] = []
        self.identity = identity
        self.state = state
        self.recent = recent
        self.end_reason = end_reason
        self.messages = messages or {}

    async def claim_cycle(self, claim):
        raise AssertionError("legacy claim_cycle must not be called")

    async def renew_cycle(self, *a, **k):
        raise AssertionError("legacy renew_cycle must not be called")

    async def release_cycle(self, *a, **k):
        raise AssertionError("legacy release_cycle must not be called")

    async def finish_cycle(self, *a, **k):
        raise AssertionError("legacy finish_cycle must not be called")

    async def get_chat(self, chat_key):
        self.calls.append(("get_chat", chat_key))
        return self.identity

    async def get_chat_state(self, chat_key):
        self.calls.append(("get_chat_state", chat_key))
        return self.state

    async def get_latest_terminal_end_reason(self, chat_key):
        self.calls.append(("get_latest_terminal_end_reason", chat_key))
        return self.end_reason

    async def get_recent_snapshot(self, chat_key, through_row_id, since_ts, limit):
        self.calls.append(
            ("get_recent_snapshot", chat_key, through_row_id, since_ts, limit)
        )
        return self.recent

    async def get_message(self, chat_key, msg_id):
        self.calls.append(("get_message", chat_key, msg_id))
        return self.messages.get(msg_id)

    async def settle_dispatch(self, settle, outbox, *, now):
        self.calls.append(("settle_dispatch", settle, outbox, now))
        self.settles.append(settle)


def test_run_dispatch_never_calls_legacy_claim_surface():
    """run_dispatch consumes the grant and settles through the ledger: the
    legacy claim/renew/release/finish surface is never touched, and
    settle_dispatch is called exactly once."""
    repo = DispatchRecordingRepo(
        identity=make_identity(),
        state=ChatState(chat_key=CK),
        recent=_recent(),
    )
    grant = _dispatch_grant(pending=(_msg(),))
    runner = make_dispatch_runner(repo, uuid_fn=lambda: "cy-1")
    decision = run(runner.run_dispatch(grant))
    kinds = [c[0] for c in repo.calls]
    assert "claim_cycle" not in kinds
    assert "renew_cycle" not in kinds
    assert "release_cycle" not in kinds
    assert "finish_cycle" not in kinds
    assert kinds.count("settle_dispatch") == 1
    settle = repo.settles[0]
    assert settle.outcome == "delay"
    assert settle.dispatch_id == grant.dispatch_id
    assert settle.cycle_id == grant.claim.cycle_id
    assert decision.action == "delay"


def test_run_dispatch_grant_boundary_honored(tmp_path):
    """The snapshot and the terminal settlement use the grant's frozen
    boundary: the recent read is bounded by grant.through_msg_id and the
    cursor advances exactly to it — never re-read from the repository."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m1", text="DeepSeek，你好", recv_ts=100.0)
        )
        await repo.ingest_message(
            make_identity(), make_message(msg_id="m2", text="DeepSeek，你好", recv_ts=110.0)
        )
        grant = await _begin_dispatch(repo)
        assert grant.through_msg_id == MessageRowId(2)
        seen: list = []

        class _Probe:
            def __init__(self, repo: Any) -> None:
                self._repo = repo

            def __getattr__(self, name: str) -> Any:
                return getattr(self._repo, name)

            async def get_recent_snapshot(self, chat_key, through_row_id, since_ts, limit):
                seen.append(through_row_id)
                return await self._repo.get_recent_snapshot(
                    chat_key, through_row_id, since_ts, limit
                )

        runner = make_dispatch_runner(_Probe(repo), uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        cursor = await db.read(
            lambda c: c.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?", (CK,)
            ).fetchone()[0]
        )
        await repo.close()
        return decision, seen, cursor

    decision, seen, cursor = run(scenario())
    assert decision.action == "skip"
    assert seen == [MessageRowId(2)]  # recent read bounded by the grant
    assert cursor == 2  # advanced exactly to the grant's through boundary


def test_run_dispatch_delay_detaches_commits_cursor_unchanged(tmp_path):
    """Ordinary delay releases the claim: no cursor/outbox movement, the
    attached commits are DETACHED and stay pending, and a fresh dispatch
    re-attaches them."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        grant = await _begin_dispatch(repo)
        assert grant.attached == (CommitSeq(1),)
        runner = make_dispatch_runner(repo, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        state = await repo.get_chat_state(CK)
        cycles = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
        )
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        unassigned = await repo.list_unassigned_commits(CK)
        again = await repo.begin_dispatch(
            DispatchRequest(
                chat_key=CK, cause=DispatchCause.INBOUND, cycle_id=CycleId("cy-2"),
                started_ts=200.0, expires_at=500.0, now=200.0,
            )
        )
        await repo.close()
        return decision, state, cycles, outbox, unassigned, again

    decision, state, cycles, outbox, unassigned, again = run(scenario())
    assert decision.action == "delay"
    assert state is not None and state.cursor_msg_id is None  # cursor unchanged
    assert cycles == 0  # no terminal cycle
    assert outbox == 0  # no outbox rows
    assert unassigned == [CommitSeq(1)]  # detached: still pending
    assert isinstance(again, DispatchGrant)  # the claim was released
    assert again.attached == (CommitSeq(1),)  # re-attached by the next dispatch


def test_run_dispatch_skip_terminal_cursor_empty_outbox_trace(tmp_path):
    """Refusal skip is TERMINAL: cursor advances to the grant's through
    boundary, outbox stays empty, the trace is persisted, the hold is
    cleared, and the idle streak resets."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(
            make_identity(), make_message(text="DeepSeek，你好", recv_ts=100.0)
        )
        grant = await _begin_dispatch(repo)
        runner = make_dispatch_runner(repo, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        cursor = await db.read(
            lambda c: c.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?", (CK,)
            ).fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason, trace_json FROM cycles").fetchall()
        )
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        state = await repo.get_chat_state(CK)
        dispatch_state = await db.read(
            lambda c: c.execute(
                "SELECT state FROM dispatches WHERE id = ?", (grant.dispatch_id,)
            ).fetchone()[0]
        )
        await repo.close()
        return decision, cursor, cycles, outbox, state, dispatch_state

    decision, cursor, cycles, outbox, state, dispatch_state = run(scenario())
    assert decision.action == "skip"
    assert decision.reason == Reason.REFUSAL
    assert cursor == 1  # advanced to the grant's through boundary
    assert cycles[0][0] == "skip"
    assert json.loads(cycles[0][1])["decision"]["reason"] == "refusal"
    assert outbox == 0  # empty outbox
    assert state is not None and state.idle_streak == 0 and state.hold_until is None
    assert dispatch_state == "completed"


def test_run_dispatch_dry_run_trigger_terminal_empty_outbox(tmp_path):
    """Dry-run trigger is TERMINAL with an EMPTY outbox: cursor consumed,
    end reason recorded, no agent/send path."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        msg = Message(
            chat_key=CK, sender_id=SenderId("u1"), sender_name="user",
            is_self=False, text="hi", id=MessageId("m1"),
            mentions=(SELF_SENDER,), recv_ts=100.0,
        )
        await repo.ingest_message(make_identity(), msg)
        grant = await _begin_dispatch(repo)
        runner = make_dispatch_runner(repo, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        cursor = await db.read(
            lambda c: c.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?", (CK,)
            ).fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        await repo.close()
        return decision, cursor, cycles, outbox

    decision, cursor, cycles, outbox = run(scenario())
    assert decision.action == "trigger"
    assert decision.reason == Reason.TRIGGER
    assert cursor == 1
    assert cycles == [("dry_run_trigger",)]
    assert outbox == 0


def test_run_dispatch_trigger_without_agent_releases(tmp_path):
    """A trigger dispatch is NEVER retained without an agent: outside
    dry-run the runner releases the claim (no cursor/outbox movement) and
    returns the trigger decision event-only."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        msg = Message(
            chat_key=CK, sender_id=SenderId("u1"), sender_name="user",
            is_self=False, text="hi", id=MessageId("m1"),
            mentions=(SELF_SENDER,), recv_ts=100.0,
        )
        await repo.ingest_message(make_identity(), msg)
        grant = await _begin_dispatch(repo)
        runner = make_dispatch_runner(repo, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        dispatch_state = await db.read(
            lambda c: c.execute(
                "SELECT state FROM dispatches WHERE id = ?", (grant.dispatch_id,)
            ).fetchone()[0]
        )
        cursor = await db.read(
            lambda c: c.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?", (CK,)
            ).fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
        )
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        unassigned = await repo.list_unassigned_commits(CK)
        await repo.close()
        return decision, dispatch_state, cursor, cycles, outbox, unassigned

    decision, dispatch_state, cursor, cycles, outbox, unassigned = run(scenario())
    assert decision.action == "trigger"
    assert dispatch_state == "released"  # never retained without an agent
    assert cursor is None  # no cursor movement
    assert cycles == 0  # no terminal completion
    assert outbox == 0
    assert unassigned == [CommitSeq(1)]  # still pending for the next dispatch


def test_run_dispatch_hook_emission(tmp_path):
    """The on_cycle_end hook fires after terminal completion only."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(
            make_identity(), make_message(text="DeepSeek，你好", recv_ts=100.0)
        )
        grant = await _begin_dispatch(repo)
        hooks = HookBus()
        seen: list[tuple] = []

        @hooks.on_cycle_end
        def hook(chat_key, trace, end_reason):
            seen.append((chat_key, trace.decision.action, end_reason))

        runner = make_dispatch_runner(
            repo, hooks=hooks, dry_run=False, uuid_fn=lambda: "cy-1"
        )
        decision = await runner.run_dispatch(grant)
        await repo.close()
        return decision, seen

    decision, seen = run(scenario())
    assert decision.action == "skip"
    assert seen == [(CK, "skip", "skip")]  # hook after terminal completion (LIVE)


def test_run_dispatch_delay_emits_no_hook(tmp_path):
    """A non-terminal delay release emits no on_cycle_end hook."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        grant = await _begin_dispatch(repo)
        hooks = HookBus()
        seen: list[tuple] = []

        @hooks.on_cycle_end
        def hook(chat_key, trace, end_reason):
            seen.append((chat_key, trace.decision.action, end_reason))

        runner = make_dispatch_runner(repo, hooks=hooks, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        await repo.close()
        return decision, seen

    decision, seen = run(scenario())
    assert decision.action == "delay"
    assert seen == []  # no hook for a non-terminal release


def test_run_dispatch_trace_sink_fires(tmp_path):
    """The trace sink fires for every evaluation, exactly like the legacy
    cycle path."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        grant = await _begin_dispatch(repo)
        traces: list = []
        runner = make_dispatch_runner(
            repo, uuid_fn=lambda: "cy-1", trace_sink=traces.append
        )
        decision = await runner.run_dispatch(grant)
        await repo.close()
        return decision, traces

    decision, traces = run(scenario())
    assert decision.action == "delay"
    assert len(traces) == 1
    assert traces[0].decision.action == "delay"


def test_run_dispatch_returns_decision_suitable_for_rearming(tmp_path):
    """The returned Decision is exactly what the LedgerScheduler re-arms
    on: a session-maintained average yields a timed delay, otherwise the
    delay is event-only; terminal outcomes are event-only."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message(recv_ts=150.0))
        await repo.upsert_chat_state(ChatState(chat_key=CK, avg_interval=60.0))
        grant = await _begin_dispatch(repo)
        runner = make_dispatch_runner(repo, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        await repo.close()
        return decision

    decision = run(scenario())
    assert decision.action == "delay"
    assert decision.delay_seconds == 10.0  # timed: avg 60 - idle 50


def test_run_dispatch_fresh_timestamp_fencing(tmp_path):
    """Terminal settlement fences against a FRESH clock timestamp: a lease
    that expired while the cycle ran rejects the settle (ClaimError), the
    prepared dispatch stays recoverable by its lease, and a later
    begin_dispatch recovers it (attaching the still-unassigned commit)."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        # A timer dispatch claims nothing (no commits exist yet); the
        # message lands AFTER, so its commit stays unassigned for recovery.
        grant = await _begin_dispatch(
            repo, cause=DispatchCause.TIMER, scheduled_ts=250.0,
            started_ts=200.0, expires_at=260.0,
        )
        assert grant.attached == ()
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        clock = VirtualClock(epoch=200.0)

        class _SlowRepo:
            """Advances the clock past the lease mid-cycle (before the
            runner's fresh settlement timestamp is read)."""

            def __init__(self, repo: Any, clock: VirtualClock) -> None:
                self._repo = repo
                self._clock = clock

            def __getattr__(self, name: str) -> Any:
                return getattr(self._repo, name)

            async def get_recent_snapshot(self, chat_key, through_row_id, since_ts, limit):
                self._clock.advance(120.0)  # now 320: the 60 s lease expired
                return await self._repo.get_recent_snapshot(
                    chat_key, through_row_id, since_ts, limit
                )

        runner = make_dispatch_runner(_SlowRepo(repo, clock), clock=clock,
                                      uuid_fn=lambda: "cy-1")
        with pytest.raises(ClaimError):
            await runner.run_dispatch(grant)
        state = await db.read(
            lambda c: c.execute(
                "SELECT state FROM dispatches WHERE id = ?", (grant.dispatch_id,)
            ).fetchone()[0]
        )
        assert state == "prepared"  # recoverable by its lease, not released
        # A later begin_dispatch at/after the horizon recovers it and
        # attaches the still-unassigned commit.
        recovered = await repo.begin_dispatch(
            DispatchRequest(
                chat_key=CK, cause=DispatchCause.INBOUND, cycle_id=CycleId("cy-2"),
                started_ts=320.0, expires_at=500.0, now=320.0,
            )
        )
        await repo.close()
        return recovered

    recovered = run(scenario())
    assert isinstance(recovered, DispatchGrant)
    assert recovered.claim.cycle_id == CycleId("cy-2")
    assert recovered.attached == (CommitSeq(1),)  # the unassigned commit attaches


def test_run_dispatch_prepared_cancellation_recovery(tmp_path):
    """A cancellation mid-cycle leaves the prepared dispatch recoverable by
    its lease (no unsafe legacy release); a later begin_dispatch recovers
    it after the lease expires."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        grant = await _begin_dispatch(
            repo, cause=DispatchCause.TIMER, scheduled_ts=250.0,
            started_ts=200.0, expires_at=260.0,
        )
        assert grant.attached == ()
        await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
        clock = VirtualClock(epoch=200.0)
        started = asyncio.Event()
        release = asyncio.Event()

        class _HoldingRepo:
            def __init__(self, repo: Any) -> None:
                self._repo = repo

            def __getattr__(self, name: str) -> Any:
                return getattr(self._repo, name)

            async def get_recent_snapshot(self, chat_key, through_row_id, since_ts, limit):
                started.set()
                await release.wait()
                return await self._repo.get_recent_snapshot(
                    chat_key, through_row_id, since_ts, limit
                )

        runner = make_dispatch_runner(_HoldingRepo(repo), clock=clock,
                                      uuid_fn=lambda: "cy-1")
        task = asyncio.create_task(runner.run_dispatch(grant))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        state = await db.read(
            lambda c: c.execute(
                "SELECT state FROM dispatches WHERE id = ?", (grant.dispatch_id,)
            ).fetchone()[0]
        )
        assert state == "prepared"  # not released, not finished
        # After the lease expires, a fresh begin_dispatch recovers it.
        clock.advance(120.0)  # now 320: lease expired
        recovered = await repo.begin_dispatch(
            DispatchRequest(
                chat_key=CK, cause=DispatchCause.INBOUND, cycle_id=CycleId("cy-2"),
                started_ts=320.0, expires_at=500.0, now=320.0,
            )
        )
        await repo.close()
        return recovered

    recovered = run(scenario())
    assert isinstance(recovered, DispatchGrant)
    assert recovered.claim.cycle_id == CycleId("cy-2")
    assert recovered.attached == (CommitSeq(1),)  # the unassigned commit attaches
