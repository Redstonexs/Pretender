"""Replay: the deterministic corpus reader, re-scoring through the SAME
snapshot assembler + Gate.evaluate path, and the sweep that varies gate
constants through a RuntimeOverlay. No database/outbox/adapter operations
anywhere in the replay path — plus live-vs-replay parity for burst and
timed-delay scheduler scenarios.

The replay processes the corpus in FILE (commit) order with a monotonic
virtual clock: receding timestamps never reorder committed rows, commits
at the same commit time coalesce into one wake (the App's next-turn
flush), and the dispatcher tie rule is commit-before-timer (a timed wake
due at exactly a commit time fires after the commit and sees it)."""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import threading
from typing import Any

import pytest

from pretender.app import App
from pretender.adapters.console import ConsoleAdapter
from pretender.clock import VirtualClock
from pretender.config import Config, RuntimeOverlay
from pretender.cycle import (
    CycleRunner,
    ReplayResult,
    _composition_fingerprint,
    replay_corpus,
    replay_marker_schedule,
    sweep_corpus,
    sweep_marker_schedule,
)
from pretender.gate import Gate
from pretender.record import CorpusView, Recorder, read_corpus, read_corpus_view
from pretender.scheduler import Scheduler
from pretender.types import (
    AdapterEvent,
    ChatKey,
    CommitSeq,
    CorpusMarker,
    Decision,
    DecisionTrace,
    DispatchCause,
    DispatchId,
    EventId,
    Message,
    MessageId,
    MessageRowId,
    Reason,
    SenderId,
    WakeKind,
)
from tests.durable_helpers import CK, make_identity, make_message, open_repo, run


def _decision(trace: DecisionTrace) -> Decision:
    """The trace's decision, narrowed (the gate always sets it)."""
    assert trace.decision is not None
    return trace.decision


def _event(msg: Message, ts: float) -> AdapterEvent:
    return AdapterEvent(type="message", payload=msg, ts=ts)


def _msg(
    text: str = "hi",
    *,
    msg_id: str = "m1",
    is_self: bool = False,
    recv_ts: float = 100.0,
    reply_to: str | None = None,
    mentions: tuple[str, ...] = (),
) -> Message:
    return Message(
        chat_key=CK,
        sender_id=SenderId("u1"),
        sender_name="user",
        is_self=is_self,
        text=text,
        id=MessageId(msg_id),
        reply_to=MessageId(reply_to) if reply_to else None,
        mentions=tuple(SenderId(m) for m in mentions),
        recv_ts=recv_ts,
    )


def _replay(events, **kw) -> ReplayResult:
    kw.setdefault("chat_key", CK)
    kw.setdefault("identity", make_identity())
    kw.setdefault("cfg", Config())
    return replay_corpus(events, **kw)


# ── determinism ─────────────────────────────────────────────────────────────

def test_replay_is_deterministic():
    events = [
        _event(_msg(text="hello"), 100.0),
        _event(_msg(text="DeepSeek，你好", msg_id="m2", recv_ts=110.0), 110.0),
        _event(_msg(text="x", msg_id="m3", recv_ts=120.0), 120.0),
    ]
    first = _replay(events)
    second = _replay(events)
    assert first == second
    assert first.traces == second.traces
    # hello: event-only delay; DeepSeek: refusal skip; x: timed delay 10;
    # the scheduler's timed wake at t+130 fires after the last arrival
    # (idle bonus activates) — the live scheduler would do the same.
    assert first.decisions == 4


def test_replay_ignores_other_chats_and_non_messages():
    other = make_message(chat_key="qq:group:999", text="other chat")
    notice = AdapterEvent(type="notice", payload={"kind": "poke"}, ts=1.0)
    result = _replay([_event(_msg(), 100.0), _event(other, 100.0), notice])
    assert result.decisions == 1
    assert result.traces[0].chat_key == CK


# ── would-have-spoken ───────────────────────────────────────────────────────

def test_replay_direct_at_would_have_spoken():
    msg = _msg(text="hi", mentions=("bot-1",))
    result = _replay([_event(msg, 100.0)])
    assert result.decisions == 1
    assert result.would_have_spoken == 1
    assert result.rate == 1.0
    assert _decision(result.traces[0]).action == "trigger"
    assert _decision(result.traces[0]).reason == Reason.TRIGGER


def test_replay_refusal_skips():
    result = _replay([_event(_msg(text="DeepSeek，你好"), 100.0)])
    assert result.would_have_spoken == 0
    assert _decision(result.traces[0]).action == "skip"
    assert _decision(result.traces[0]).reason == Reason.REFUSAL


def test_replay_self_messages_never_trigger_cycles():
    events = [
        _event(_msg(text="self reply", msg_id="s1", is_self=True, recv_ts=90.0), 90.0),
        _event(_msg(text="hello", recv_ts=100.0), 100.0),
    ]
    result = _replay(events)
    assert result.decisions == 1  # only the non-self message
    assert result.traces[0].snapshot_facts["window_count"] == 2  # self counted


def test_replay_quote_to_self_resolves():
    self_msg = _msg(text="bot reply", msg_id="s1", is_self=True, recv_ts=90.0)
    reply = _msg(text="thanks", msg_id="m2", recv_ts=100.0, reply_to="s1")
    result = _replay([_event(self_msg, 90.0), _event(reply, 100.0)])
    assert result.traces[0].snapshot_facts["has_quote_to_self"] is True
    assert _decision(result.traces[0]).action == "trigger"


def test_replay_full_window_self_ratio():
    events = [
        _event(_msg(text="a"), 100.0),
        _event(_msg(text="b", msg_id="m2", recv_ts=110.0), 110.0),
        _event(_msg(text="self", msg_id="s1", is_self=True, recv_ts=120.0), 120.0),
        _event(_msg(text="c", msg_id="m3", recv_ts=130.0), 130.0),
    ]
    result = _replay(events)
    facts = result.traces[-1].snapshot_facts
    assert facts["window_count"] == 4
    assert facts["self_count"] == 1
    assert facts["self_ratio"] == 0.25


def test_replay_window_bounds_are_fixed():
    """Messages older than the 300 s window leave the counts but the
    limited list and the window facts stay consistent."""
    events = [
        _event(_msg(text="old", recv_ts=-400.0), -400.0),
        _event(_msg(text="new", msg_id="m2", recv_ts=100.0), 100.0),
    ]
    result = _replay(events)
    # The evaluation at t=100 (the new message's wake): the old message is
    # outside the window, the new one is the last non-self message.
    facts = result.traces[1].snapshot_facts
    assert facts["window_count"] == 1  # the old message is outside the window
    assert facts["idle_seconds"] == 0.0  # last non-self is the new message
    # The tail evaluation (the timed wake at t+600) has an EMPTY window:
    # both messages are outside the 300 s window by then.
    assert result.traces[-1].snapshot_facts["window_count"] == 0


def test_replay_cursor_advances_on_terminal_only():
    """A delay leaves the pending set intact; a later trigger consumes
    everything up to it."""
    events = [
        _event(_msg(text="hello"), 100.0),  # delay (score below trigger)
        _event(_msg(text="hi", msg_id="m2", mentions=("bot-1",), recv_ts=110.0), 110.0),
    ]
    result = _replay(events)
    assert result.decisions == 2
    assert _decision(result.traces[0]).action == "delay"
    assert _decision(result.traces[1]).action == "trigger"
    assert result.traces[1].snapshot_facts["pending"] == 2  # both still pending


# ── sweep ───────────────────────────────────────────────────────────────────

def test_replay_sweep_varies_constants_through_overlay():
    events = [
        _event(_msg(text="x", msg_id=f"m{i}", recv_ts=100.0 + i), 100.0 + i)
        for i in range(10)
    ]
    base = _replay(events)
    overlay = RuntimeOverlay()
    overlay.set("gate.threshold", 2)
    overlay.set("gate.trigger_score", 40)
    swept = _replay(events, overlay=overlay)
    assert base.would_have_spoken == 0  # 10 ambient messages stay below 80
    assert swept.would_have_spoken > 0  # threshold 2 crosses the trigger
    assert swept.traces[0].threshold == 2
    assert swept.traces[0].trigger_score == 40


def test_sweep_corpus_reports_every_combination():
    events = [
        _event(_msg(text="x", msg_id=f"m{i}", recv_ts=100.0 + i), 100.0 + i)
        for i in range(10)
    ]
    rows = sweep_corpus(events, chat_key=CK, identity=make_identity(), cfg=Config())
    assert len(rows) == 16  # 4 thresholds x 4 trigger scores
    assert rows[0].threshold == 2 and rows[0].trigger_score == 40
    assert rows[-1].threshold == 12 and rows[-1].trigger_score == 100
    # 10 arrivals + at most one tail timed wake after the last one (the
    # tail fires only when the last evaluation re-arms a timed delay).
    assert all(row.decisions in (10, 11) for row in rows)
    assert any(row.would_have_spoken != rows[0].would_have_spoken for row in rows)


def test_replay_empty_corpus():
    result = _replay([])
    assert result.decisions == 0
    assert result.would_have_spoken == 0
    assert result.rate == 0.0
    assert result.traces == ()


# ── recorder reader ─────────────────────────────────────────────────────────

def test_recorder_read_events_roundtrip(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [
        _event(_msg(text="你好", recv_ts=100.0), 100.0),
        AdapterEvent(type="notice", payload={"kind": "poke"}, raw={"x": 1}, ts=2.0),
    ]
    with Recorder(path) as rec:
        for e in events:
            rec.write_event(e)
    with Recorder(path) as rec:
        read = rec.read_events()
    assert read == events


def test_recorder_reader_skips_malformed_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"ts": 1.0, "type": "message", "chat_key": "qq:group:123456",'
        ' "sender_id": "u1", "sender_name": "user", "is_self": false,'
        ' "text": "hi", "id": "m1", "reply_to": null, "mentions": [],'
        ' "segments": [], "raw": null}\n'
        "not json at all\n"
        '{"ts": 2.0, "type": "notice", "payload": {"kind": "poke"}, "raw": null}\n',
        encoding="utf-8",
    )
    with Recorder(path) as rec:
        events = rec.read_events()
    assert len(events) == 2
    assert events[0].payload.text == "hi"
    assert events[1].type == "notice"


def test_read_corpus_missing_file_returns_empty(tmp_path):
    assert read_corpus(tmp_path / "nope.jsonl") == []


def test_replay_from_recorded_corpus(tmp_path):
    """The full loop: record events, read them back, replay them."""
    path = tmp_path / "events.jsonl"
    with Recorder(path) as rec:
        rec.write_event(_event(_msg(text="DeepSeek，你好"), 100.0))
    events = read_corpus(path)
    result = _replay(events)
    assert result.decisions == 1
    assert _decision(result.traces[0]).reason == Reason.REFUSAL

# ── scheduler-equivalent wake/batch semantics ───────────────────────────────

def test_replay_same_time_burst_coalesces_into_one_evaluation():
    """A same-time burst is claimed as ONE batch by the live cycle: the
    replay coalesces it into one evaluation with the full pending set."""
    events = [
        _event(_msg(text="a", msg_id="m1", recv_ts=100.0), 100.0),
        _event(_msg(text="b", msg_id="m2", recv_ts=100.0), 100.0),
        _event(_msg(text="c", msg_id="m3", recv_ts=100.0), 100.0),
    ]
    result = _replay(events)
    assert result.decisions == 1  # one evaluation for the whole burst
    assert result.traces[0].snapshot_facts["pending"] == 3
    assert result.traces[0].snapshot_facts["window_count"] == 3


def test_replay_timed_delay_holds_later_ordinary_arrivals():
    """A timed delay re-arms the next evaluation at now + delay; ordinary
    arrivals during the delay stay pending and are seen at the scheduled
    evaluation — they never override it."""
    events = [
        _event(_msg(text="a", msg_id="m1", recv_ts=100.0), 100.0),
        _event(_msg(text="b", msg_id="m2", recv_ts=160.0), 160.0),  # avg = 60
        _event(_msg(text="c", msg_id="m3", recv_ts=190.0), 190.0),  # during delay
    ]
    result = _replay(events)
    # m1: event-only delay (no average). m2: timed delay 60 (avg 60, idle 0).
    # m3: held — no evaluation at 190.
    assert _decision(result.traces[0]).action == "delay"
    assert _decision(result.traces[0]).delay_seconds is None
    assert _decision(result.traces[1]).action == "delay"
    assert _decision(result.traces[1]).delay_seconds == 60.0
    assert result.traces[1].snapshot_facts["pending"] == 2
    # The scheduled evaluation at t+220 sees the held arrival (pending 3);
    # its timed delay 15 fires at t+235, where the idle bonus activates
    # (idle 45 >= avg 45) and the chain ends event-only.
    assert result.traces[2].snapshot_facts["evaluated_ts"] == 220.0
    assert result.traces[2].snapshot_facts["pending"] == 3
    assert result.decisions == 4


def test_replay_timed_wake_fires_after_last_arrival():
    """The scheduler keeps firing the timed wake after the last arrival:
    the idle bonus activates there (idle >= avg), so the evaluation
    happens even with no further corpus input."""
    events = [
        _event(_msg(text="a", msg_id="m1", recv_ts=100.0), 100.0),
        _event(_msg(text="b", msg_id="m2", recv_ts=160.0), 160.0),
    ]
    result = _replay(events)
    assert result.decisions == 3  # t=100, t=160, timed wake at t=220
    tail = result.traces[-1]
    assert tail.snapshot_facts["evaluated_ts"] == 220.0
    assert tail.snapshot_facts["idle_seconds"] == 60.0  # 220 - 160
    assert tail.snapshot_facts["pending"] == 2  # the held batch


def test_replay_priority_direct_overrides_scheduled_delay():
    """A structurally recognized direct @ during a scheduled delay
    re-evaluates immediately (the priority wake path); the gate applies
    the exact precedence."""
    events = [
        _event(_msg(text="a", msg_id="m1", recv_ts=100.0), 100.0),
        _event(_msg(text="b", msg_id="m2", recv_ts=160.0), 160.0),  # timed delay 60
        _event(_msg(text="hi", msg_id="m3", recv_ts=190.0,
                    mentions=("bot-1",)), 190.0),  # direct @ during the delay
    ]
    result = _replay(events)
    assert result.decisions == 3  # the direct @ overrode the delay
    assert _decision(result.traces[2]).action == "trigger"
    assert _decision(result.traces[2]).reason == Reason.TRIGGER
    assert result.traces[2].snapshot_facts["pending"] == 3


def test_replay_evolves_ewma_avg_from_non_self_messages():
    """Replay evolves the SAME EWMA average the live ingest path persists
    (the central pacing reducer): self messages never participate and a
    first non-self message carries no prior sample."""
    events = [
        _event(_msg(text="self", msg_id="s1", is_self=True, recv_ts=100.0), 100.0),
        _event(_msg(text="a", msg_id="m1", recv_ts=160.0), 160.0),
        _event(_msg(text="b", msg_id="m2", recv_ts=190.0), 190.0),
    ]
    result = _replay(events)
    # a@160: no prior non-self sample -> no average. b@190: seeds 30.
    assert result.traces[0].snapshot_facts["recent_average_interval"] == 0.0
    assert result.traces[1].snapshot_facts["recent_average_interval"] == 30.0


def test_replay_quote_resolves_against_whole_corpus():
    """A quote of an OLD self message (outside the 300 s window) still
    triggers: the corpus is the durable store, not the rendered window."""
    events = [
        _event(_msg(text="bot reply", msg_id="s1", is_self=True, recv_ts=100.0), 100.0),
        _event(_msg(text="a", msg_id="m1", recv_ts=150.0), 150.0),
        _event(_msg(text="thanks", msg_id="m2", recv_ts=500.0,
                    reply_to="s1"), 500.0),
    ]
    result = _replay(events)
    assert result.traces[-1].snapshot_facts["has_quote_to_self"] is True
    assert _decision(result.traces[-1]).action == "trigger"


def test_replay_pending_excludes_self_messages():
    """Self messages are retained in the recent window and the full-window
    counts but EXCLUDED from every pending batch: a self echo can never
    inflate pending/pending_messages (live claim parity)."""
    events = [
        _event(_msg(text="a", msg_id="m1", recv_ts=100.0), 100.0),
        _event(_msg(text="self", msg_id="s1", is_self=True, recv_ts=110.0), 110.0),
        _event(_msg(text="b", msg_id="m2", recv_ts=120.0), 120.0),
    ]
    result = _replay(events)
    # The evaluation at 120 sees m1 + m2 pending (the self message is
    # excluded) while the window still counts it.
    trace = result.traces[-1]
    assert trace.snapshot_facts["pending"] == 2
    assert trace.pending == 2
    assert trace.snapshot_facts["window_count"] == 3
    assert trace.snapshot_facts["self_count"] == 1
    assert trace.snapshot_facts["self_ratio"] == pytest.approx(1 / 3)


def test_replay_self_echo_never_inflates_pending_after_terminal():
    """A self echo between two non-self messages stays out of pending even
    when the cursor advances past it (the cursor is a row boundary; the
    pending set is self-free)."""
    events = [
        _event(_msg(text="hi", msg_id="m1", mentions=("bot-1",), recv_ts=100.0), 100.0),
        _event(_msg(text="self", msg_id="s1", is_self=True, recv_ts=110.0), 110.0),
        _event(_msg(text="b", msg_id="m2", recv_ts=120.0), 120.0),
    ]
    result = _replay(events)
    # m1 triggers at 100 (cursor -> 1); the self echo at 110 never wakes;
    # m2 at 120 is the only pending message (its timed delay fires a tail
    # wake at 260, which sees the same self-free pending set).
    assert result.decisions == 3
    assert result.traces[1].snapshot_facts["pending"] == 1
    assert result.traces[1].snapshot_facts["window_count"] == 3
    assert result.traces[2].snapshot_facts["pending"] == 1


def test_replay_high_pending_priority_overrides_scheduled_delay():
    """A batch whose pending count reaches the gate threshold takes the
    App's priority wake path: it overrides a scheduled delay and
    re-evaluates immediately (high pending may bypass a hold/delay)."""
    cfg = Config.from_dict({"gate": {"threshold": 2, "trigger_score": 40}})
    events = [
        _event(_msg(text="a", msg_id="m1", recv_ts=100.0), 100.0),
        _event(_msg(text="b", msg_id="m2", recv_ts=160.0), 160.0),  # trigger
        _event(_msg(text="c", msg_id="m3", recv_ts=190.0), 190.0),  # timed delay 60
        _event(_msg(text="d", msg_id="m4", recv_ts=220.0), 220.0),  # during delay
    ]
    result = replay_corpus(events, chat_key=CK, identity=make_identity(), cfg=cfg)
    # m4's batch: pending 2 >= threshold 2 -> priority -> evaluated at 220
    # (without the priority rule it would be held until the 250 wake).
    assert result.decisions == 4
    assert result.traces[3].snapshot_facts["evaluated_ts"] == 220.0
    assert result.traces[3].snapshot_facts["pending"] == 2
    assert _decision(result.traces[3]).action == "trigger"


def test_replay_high_pending_priority_never_fires_for_self_only_batch():
    """A self-only batch never wakes, so it can never priority-wake even
    when the pending count is at/above the threshold."""
    cfg = Config.from_dict({"gate": {"threshold": 2, "trigger_score": 40}})
    events = [
        _event(_msg(text="a", msg_id="m1", recv_ts=100.0), 100.0),
        _event(_msg(text="b", msg_id="m2", recv_ts=160.0), 160.0),  # trigger
        _event(_msg(text="c", msg_id="m3", recv_ts=190.0), 190.0),  # timed delay 60
        _event(_msg(text="self", msg_id="s1", is_self=True, recv_ts=220.0), 220.0),
    ]
    result = replay_corpus(events, chat_key=CK, identity=make_identity(), cfg=cfg)
    # The self-only batch at 220 never wakes: no evaluation at 220; the
    # timed wake at 235 fires with only m3 pending (self excluded).
    assert result.decisions == 4
    assert result.traces[3].snapshot_facts["evaluated_ts"] == 235.0
    assert result.traces[3].snapshot_facts["pending"] == 1


# ── file/commit order (no global recv_ts sort) ──────────────────────────────

def test_replay_preserves_file_commit_order_with_receding_timestamps():
    """A corpus whose timestamps recede is processed in FILE order: row
    ids follow the file, the virtual clock never moves backwards, and the
    receding message coalesces into the evaluation at the current time —
    never reordered by recv_ts."""
    events = [
        _event(_msg(text="a", msg_id="m1", recv_ts=200.0), 200.0),
        _event(_msg(text="b", msg_id="m2", recv_ts=100.0), 100.0),
    ]
    result = _replay(events)
    # ONE evaluation at t=200 (both commits coalesce at the current time):
    # a global recv_ts sort would evaluate b alone at t=100 first.
    assert result.decisions == 1
    trace = result.traces[0]
    assert trace.snapshot_facts["evaluated_ts"] == 200.0
    assert trace.snapshot_facts["through_msg_id"] == 2  # row ids follow the file
    assert trace.snapshot_facts["pending"] == 2
    assert trace.snapshot_facts["window_count"] == 2


def test_replay_sawtooth_timestamps_never_reorder_rows():
    """A sawtooth corpus (300, 100, 200) commits in file order: every row
    commits at the current time (t=300) as ONE coalesced evaluation with
    row ids 1..3 in file order. A global sort would produce three
    evaluations with reordered row ids."""
    events = [
        _event(_msg(text="a", msg_id="m1", recv_ts=300.0), 300.0),
        _event(_msg(text="b", msg_id="m2", recv_ts=100.0), 100.0),
        _event(_msg(text="c", msg_id="m3", recv_ts=200.0), 200.0),
    ]
    result = _replay(events)
    assert result.decisions == 1
    trace = result.traces[0]
    assert trace.snapshot_facts["evaluated_ts"] == 300.0
    assert trace.snapshot_facts["through_msg_id"] == 3
    assert trace.snapshot_facts["pending"] == 3
    assert trace.snapshot_facts["window_count"] == 3


def test_replay_receding_timestamp_stays_out_of_window():
    """A receding timestamp commits at the current time but keeps its
    recv_ts: the row is committed (pending, within the claim) yet outside
    the 300 s window when its recv_ts is older than now - window."""
    events = [
        _event(_msg(text="a", msg_id="m1", recv_ts=100.0), 100.0),
        _event(_msg(text="old", msg_id="m2", recv_ts=-400.0), -400.0),
    ]
    result = _replay(events)
    # Both commit at t=100 (the clock never moves backwards): ONE
    # evaluation; the receding row is pending but outside the window.
    assert result.decisions == 1
    trace = result.traces[0]
    assert trace.snapshot_facts["evaluated_ts"] == 100.0
    assert trace.snapshot_facts["through_msg_id"] == 2
    assert trace.snapshot_facts["pending"] == 2
    assert trace.snapshot_facts["window_count"] == 1
    assert trace.snapshot_facts["last_nonself_ts"] == 100.0


def test_replay_ewma_evolves_in_commit_order_not_timestamp_order():
    """The durable EWMA folds non-self messages in ROW-ID (commit) order
    with the SAME reducer + prior-sample semantics as the repository: the
    prior sample is the MAX recv_ts among the prior non-self rows, so a
    receding timestamp is a non-positive gap that carries no pacing
    information and never drags the average."""
    events = [
        _event(_msg(text="a", msg_id="m1", recv_ts=200.0), 200.0),
        _event(_msg(text="b", msg_id="m2", recv_ts=100.0), 100.0),  # recedes
        _event(_msg(text="hi", msg_id="m3", recv_ts=250.0,
                    mentions=("bot-1",)), 250.0),
    ]
    result = _replay(events)
    # m1+m2 commit at t=200 (one coalesced evaluation, avg None); m3
    # commits at t=250: its gap is 250 - MAX(200, 100) = 50, NOT
    # 250 - 100 = 150 (the receding row never drags the average).
    assert result.decisions == 2
    assert result.traces[0].snapshot_facts["recent_average_interval"] == 0.0
    assert result.traces[1].snapshot_facts["recent_average_interval"] == 50.0


# ── dispatcher tie rule (commit-before-timer) ───────────────────────────────

def test_replay_timer_due_at_commit_time_sees_the_committed_event():
    """Tie rule (commit-before-timer): a timed wake due at exactly an
    event's commit time fires AFTER the commit — the committed event's
    wake metadata coalesces into that evaluation (the live claim includes
    every row committed before the wake), and the event's own wake is
    subsumed: exactly ONE evaluation at that time."""
    events = [
        _event(_msg(text="a", msg_id="m1", recv_ts=100.0), 100.0),
        _event(_msg(text="b", msg_id="m2", recv_ts=160.0), 160.0),  # timed delay 60
        _event(_msg(text="c", msg_id="m3", recv_ts=220.0), 220.0),  # timer due at 220
    ]
    result = _replay(events)
    at_220 = [t for t in result.traces if t.snapshot_facts["evaluated_ts"] == 220.0]
    assert len(at_220) == 1  # the timer's evaluation subsumes the event's wake
    assert at_220[0].snapshot_facts["pending"] == 3  # the committed event is seen


def test_replay_timer_due_before_commit_time_does_not_see_the_event():
    """The other side of the tie rule: a timed wake due strictly BEFORE an
    event's commit time fires during the clock advance, BEFORE the commit,
    and does not see the event."""
    events = [
        _event(_msg(text="a", msg_id="m1", recv_ts=100.0), 100.0),
        _event(_msg(text="b", msg_id="m2", recv_ts=160.0), 160.0),  # timed delay 60
        _event(_msg(text="c", msg_id="m3", recv_ts=250.0), 250.0),  # timer due at 220
    ]
    result = _replay(events)
    at_220 = [t for t in result.traces if t.snapshot_facts["evaluated_ts"] == 220.0]
    assert len(at_220) == 1
    assert at_220[0].snapshot_facts["pending"] == 2  # m3 is NOT yet committed
    at_250 = [t for t in result.traces if t.snapshot_facts["evaluated_ts"] == 250.0]
    assert len(at_250) == 1
    assert at_250[0].snapshot_facts["pending"] == 3  # m3's own wake fires


# ── dispatcher-turn coalescing ──────────────────────────────────────────────

def test_replay_commits_within_one_turn_coalesce_into_one_wake():
    """Commits at the same commit time (the App's next-turn flush) coalesce
    into ONE evaluation with the full pending set; a later commit gets its
    own evaluation."""
    events = [
        _event(_msg(text="a", msg_id="m1", recv_ts=100.0), 100.0),
        _event(_msg(text="b", msg_id="m2", recv_ts=100.0), 100.0),
        _event(_msg(text="hi", msg_id="m3", recv_ts=150.0,
                    mentions=("bot-1",)), 150.0),
    ]
    result = _replay(events)
    assert result.decisions == 2  # one wake for the burst, one for m3
    assert result.traces[0].snapshot_facts["evaluated_ts"] == 100.0
    assert result.traces[0].snapshot_facts["pending"] == 2
    assert result.traces[1].snapshot_facts["evaluated_ts"] == 150.0
    assert result.traces[1].snapshot_facts["pending"] == 3


def test_replay_self_echo_in_coalesced_group_never_wakes_or_inflates():
    """A self echo committed in the same turn as a non-self message never
    wakes and never joins the pending batch (it stays in the window)."""
    events = [
        _event(_msg(text="a", msg_id="m1", recv_ts=100.0), 100.0),
        _event(_msg(text="self", msg_id="s1", is_self=True, recv_ts=100.0), 100.0),
        _event(_msg(text="hi", msg_id="m2", recv_ts=150.0,
                    mentions=("bot-1",)), 150.0),
    ]
    result = _replay(events)
    assert result.decisions == 2  # the self echo never wakes
    first = result.traces[0]
    assert first.snapshot_facts["evaluated_ts"] == 100.0
    assert first.snapshot_facts["pending"] == 1  # self excluded
    assert first.snapshot_facts["window_count"] == 2  # self counted in the window
    assert first.snapshot_facts["self_count"] == 1


# ── live-vs-replay parity ───────────────────────────────────────────────────

def _assert_traces_match(live: DecisionTrace, replay: DecisionTrace) -> None:
    """The live cycle and the replay produce the SAME decision, aggregates,
    backoff facts, config, and snapshot facts (cycle ids differ by
    construction)."""
    assert live.decision == replay.decision
    assert live.mode == replay.mode
    assert live.threshold == replay.threshold
    assert live.trigger_score == replay.trigger_score
    assert live.pending == replay.pending
    assert live.aggregates == replay.aggregates
    assert live.backoff == replay.backoff
    assert live.config == replay.config
    lf = {k: v for k, v in live.snapshot_facts.items() if k != "cycle_id"}
    rf = {k: v for k, v in replay.snapshot_facts.items() if k != "cycle_id"}
    assert lf == rf


async def _settle(scheduler: Scheduler, traces: list, count: int) -> None:
    """Yield until ``count`` traces were produced AND the scheduler's
    lease is free (the cycle completed and the re-arm happened).
    Deterministic: pure event-loop yielding."""
    for _ in range(20_000):
        if len(traces) >= count and not scheduler.is_leased(CK):
            return
        await asyncio.sleep(0)
    raise AssertionError(f"scheduler did not produce {count} traces")


def test_live_replay_parity_same_time_burst(tmp_path):
    """A same-time burst committed before the wake is claimed as ONE batch
    by the live cycle; replay coalesces it into one identical evaluation."""
    clock = VirtualClock(auto_advance=False)
    cfg = Config.from_dict({"storage": {"db_path": str(tmp_path / "p.db")}})
    corpus_path = tmp_path / "p.jsonl"

    async def scenario():
        db, repo = await open_repo(tmp_path / "p.db")
        await repo.upsert_chat(make_identity())
        traces: list[DecisionTrace] = []
        runner = CycleRunner(repo, Gate(), cfg, clock=clock, dry_run=True,
                             trace_sink=traces.append)
        scheduler = Scheduler(clock, runner)
        scheduler.start()
        rec = Recorder(corpus_path)
        for i in range(3):
            msg = make_message(text=f"m{i}", msg_id=f"m{i}", recv_ts=clock.now())
            await repo.ingest_message(make_identity(), msg)
            rec.write_event(AdapterEvent(type="message", payload=msg, ts=msg.recv_ts))
        await scheduler.wake(CK)
        await _settle(scheduler, traces, 1)
        await scheduler.stop()
        rec.close()
        await repo.close()
        return traces

    live = run(scenario())
    events = read_corpus(corpus_path)
    replay = replay_corpus(events, chat_key=CK, identity=make_identity(), cfg=cfg)
    assert len(live) == 1  # one cycle for the whole burst
    assert replay.decisions == 1
    assert live[0].decision is not None
    assert replay.would_have_spoken == (
        1 if live[0].decision.action == "trigger" else 0
    )
    _assert_traces_match(live[0], replay.traces[0])


def test_live_replay_parity_timed_delay(tmp_path):
    """The live scheduler's timed-delay wake/batch semantics — including
    the held ordinary arrival and the post-corpus tail wake — match the
    replay evaluation for evaluation."""
    clock = VirtualClock(auto_advance=False)
    cfg = Config.from_dict({"storage": {"db_path": str(tmp_path / "p.db")}})
    corpus_path = tmp_path / "p.jsonl"

    async def scenario():
        db, repo = await open_repo(tmp_path / "p.db")
        await repo.upsert_chat(make_identity())
        traces: list[DecisionTrace] = []
        runner = CycleRunner(repo, Gate(), cfg, clock=clock, dry_run=True,
                             trace_sink=traces.append)
        scheduler = Scheduler(clock, runner)
        scheduler.start()
        rec = Recorder(corpus_path)

        async def ingest(text: str, msg_id: str) -> None:
            msg = make_message(text=text, msg_id=msg_id, recv_ts=clock.now())
            await repo.ingest_message(make_identity(), msg)
            rec.write_event(AdapterEvent(type="message", payload=msg, ts=msg.recv_ts))
            await scheduler.wake(CK)

        await ingest("a", "m1")  # t=0: event-only delay (no average)
        await _settle(scheduler, traces, 1)
        clock.advance(60.0)
        await ingest("b", "m2")  # t=60: timed delay 60 (avg 60, idle 0)
        await _settle(scheduler, traces, 2)
        clock.advance(30.0)
        await ingest("c", "m3")  # t=90: held by the delay — no evaluation
        await asyncio.sleep(0)
        assert len(traces) == 2
        clock.advance(30.0)  # t=120: the timed wake fires
        await _settle(scheduler, traces, 3)
        clock.advance(15.0)  # t=135: the next timed wake fires (idle bonus)
        await _settle(scheduler, traces, 4)
        clock.advance(45.0)  # t=180
        await ingest("d", "m4")  # t=180: its own evaluation
        await _settle(scheduler, traces, 5)
        # The tail: the scheduler keeps firing the timed wake after the
        # last arrival (idle bonus activates) — replay models the same.
        clock.advance(67.5)  # t=247.5
        await _settle(scheduler, traces, 6)
        await scheduler.stop()
        rec.close()
        await repo.close()
        return traces

    live = run(scenario())
    events = read_corpus(corpus_path)
    replay = replay_corpus(events, chat_key=CK, identity=make_identity(), cfg=cfg)
    assert len(live) == replay.decisions == 6
    for lt, rt in zip(live, replay.traces):
        _assert_traces_match(lt, rt)


def test_live_replay_parity_quiet_single_message(tmp_path):
    """One quiet message: the live cycle commits it, wakes once, and the
    replay produces the identical single trace."""
    clock = VirtualClock(auto_advance=False)
    cfg = Config.from_dict({"storage": {"db_path": str(tmp_path / "p.db")}})
    corpus_path = tmp_path / "p.jsonl"

    async def scenario():
        db, repo = await open_repo(tmp_path / "p.db")
        await repo.upsert_chat(make_identity())
        traces: list[DecisionTrace] = []
        runner = CycleRunner(repo, Gate(), cfg, clock=clock, dry_run=True,
                             trace_sink=traces.append)
        scheduler = Scheduler(clock, runner)
        scheduler.start()
        rec = Recorder(corpus_path)
        msg = make_message(text="quiet", msg_id="m1", recv_ts=clock.now())
        await repo.ingest_message(make_identity(), msg)
        rec.write_event(AdapterEvent(type="message", payload=msg, ts=msg.recv_ts))
        await scheduler.wake(CK)
        await _settle(scheduler, traces, 1)
        await scheduler.stop()
        rec.close()
        await repo.close()
        return traces

    live = run(scenario())
    events = read_corpus(corpus_path)
    replay = replay_corpus(events, chat_key=CK, identity=make_identity(), cfg=cfg)
    assert len(live) == replay.decisions == 1
    _assert_traces_match(live[0], replay.traces[0])


def test_live_replay_parity_receding_timestamps_dispatcher_scenario(tmp_path):
    """The documented dispatcher scenario with receding timestamps: the
    live cycle commits every event immediately (row ids in file order),
    same-turn commits coalesce into one wake, and the replay — processing
    the corpus in file order with a monotonic clock — produces the
    identical traces."""
    clock = VirtualClock(auto_advance=False)
    cfg = Config.from_dict({"storage": {"db_path": str(tmp_path / "p.db")}})
    corpus_path = tmp_path / "p.jsonl"

    async def scenario():
        db, repo = await open_repo(tmp_path / "p.db")
        await repo.upsert_chat(make_identity())
        traces: list[DecisionTrace] = []
        runner = CycleRunner(repo, Gate(), cfg, clock=clock, dry_run=True,
                             trace_sink=traces.append)
        scheduler = Scheduler(clock, runner)
        scheduler.start()
        rec = Recorder(corpus_path)

        async def commit(text: str, msg_id: str, recv_ts: float,
                         mentions: tuple = ()) -> None:
            msg = make_message(text=text, msg_id=msg_id, recv_ts=recv_ts,
                               mentions=mentions)
            await repo.ingest_message(make_identity(), msg)
            rec.write_event(AdapterEvent(type="message", payload=msg, ts=msg.recv_ts))

        # t=0: m1 (recv_ts 0) and m2 (recv_ts -50, receding) commit in the
        # SAME turn: the next-turn flush coalesces them into ONE wake.
        await commit("a", "m1", clock.now())
        await commit("b", "m2", clock.now() - 50.0)
        await scheduler.wake(CK)
        await _settle(scheduler, traces, 1)
        # t=60: m3 commits; its own wake fires (timed delay 110).
        clock.advance(60.0)
        await commit("c", "m3", clock.now())
        await scheduler.wake(CK)
        await _settle(scheduler, traces, 2)
        # t=90: m4 (recv_ts 90, direct @) commits; the priority wake
        # overrides the scheduled delay and triggers.
        clock.advance(30.0)
        await commit("hi", "m4", clock.now(), mentions=("bot-1",))
        await scheduler.wake_priority(CK)
        await _settle(scheduler, traces, 3)
        await scheduler.stop()
        rec.close()
        await repo.close()
        return traces

    live = run(scenario())
    events = read_corpus(corpus_path)
    replay = replay_corpus(events, chat_key=CK, identity=make_identity(), cfg=cfg)
    assert len(live) == replay.decisions == 3
    for lt, rt in zip(live, replay.traces):
        _assert_traces_match(lt, rt)


def test_durable_ewma_matches_replay(tmp_path):
    """The durable avg_interval the live ingest path persists equals the
    EWMA the replay evolves from the same corpus (the SAME reducer)."""
    clock = VirtualClock(auto_advance=False)
    cfg = Config.from_dict({"storage": {"db_path": str(tmp_path / "p.db")}})
    corpus_path = tmp_path / "p.jsonl"

    async def scenario():
        db, repo = await open_repo(tmp_path / "p.db")
        await repo.upsert_chat(make_identity())
        rec = Recorder(corpus_path)
        for i, gap in enumerate((0.0, 60.0, 30.0)):
            if i:
                clock.advance(gap)
            msg = make_message(text=f"m{i}", msg_id=f"m{i}", recv_ts=clock.now())
            await repo.ingest_message(make_identity(), msg)
            rec.write_event(AdapterEvent(type="message", payload=msg, ts=msg.recv_ts))
        state = await repo.get_chat_state(CK)
        rec.close()
        await repo.close()
        assert state is not None
        return state.avg_interval

    durable_avg = run(scenario())
    # EWMA: 60 seeds, then 0.5*30 + 0.5*60 = 45.
    assert durable_avg == pytest.approx(45.0)
    events = read_corpus(corpus_path)
    result = replay_corpus(events, chat_key=CK, identity=make_identity(), cfg=cfg)
    # The replay's last evaluation carries the same evolved average.
    assert result.traces[-1].snapshot_facts["recent_average_interval"] == pytest.approx(45.0)


# ── marker-driven replay: the v4 recorded dispatch schedule ─────────────────

def _marker_view(
    messages,
    *,
    message_row_ids=None,
    chat_key=CK,
    schedule=None,
    settled_ts_base=100.0,
    causes=None,
    scheduled_fors=None,
) -> CorpusView:
    """Build a ``CorpusView`` from messages + a dispatch schedule.

    ``messages`` is a list of ``Message``; each gets a commit marker in
    CommitSeq order (event ids ``ev-1``..). ``schedule`` is a list of
    ``(attached, state)`` tuples, one per dispatch in DispatchId order,
    where ``attached`` is a tuple of CommitSeq ints; defaults to one
    completed dispatch per message attaching just that message's commit.
    ``causes``/``scheduled_fors`` optionally override the per-dispatch
    cause/scheduled time."""
    events: dict[EventId, AdapterEvent] = {}
    commits: list[CorpusMarker] = []
    for i, msg in enumerate(messages, start=1):
        ev = EventId(f"ev-{i}")
        events[ev] = AdapterEvent(type="message", payload=msg, ts=msg.recv_ts)
        commits.append(
            CorpusMarker(
                record_type="commit",
                sequence=CommitSeq(i),
                chat_key=chat_key,
                event_id=ev,
                wake_kind=WakeKind.INBOUND,
                message_row_id=(
                    MessageRowId(message_row_ids[i - 1])
                    if message_row_ids is not None
                    else None
                ),
            )
        )
    if schedule is None:
        schedule = [((i,), "completed") for i in range(1, len(messages) + 1)]
    fingerprint = _composition_fingerprint(Gate())
    dispatches: list[CorpusMarker] = []
    for i, (attached, state) in enumerate(schedule, start=1):
        through = (
            max(
                message_row_ids[s - 1] if message_row_ids is not None else s
                for s in attached
            )
            if attached
            else 0
        )
        dispatches.append(
            CorpusMarker(
                record_type="dispatch",
                sequence=DispatchId(i),
                chat_key=chat_key,
                cause=(
                    causes[i - 1]
                    if causes is not None
                    else DispatchCause.INBOUND
                ),
                commit_boundary=CommitSeq(through),
                scheduled_for=(
                    scheduled_fors[i - 1]
                    if scheduled_fors is not None
                    else None
                ),
                state=state,
                settled_ts=settled_ts_base + i,
                start_msg_id=MessageRowId(0),
                through_msg_id=MessageRowId(through),
                attached=tuple(CommitSeq(s) for s in attached),
                trace_json=json.dumps({"config": fingerprint}),
            )
        )
    return CorpusView(
        events_by_event_id=events,
        commits=tuple(commits),
        dispatches=tuple(dispatches),
    )


def _marker_replay(view, **kw) -> ReplayResult:
    kw.setdefault("chat_key", CK)
    kw.setdefault("identity", make_identity())
    kw.setdefault("cfg", Config())
    # Synthetic marker fixtures model a recorded default composition. Exact
    # replay production paths still fail closed when a corpus omits this proof.
    return replay_marker_schedule(view, **kw)


def test_marker_replay_reconstructs_exact_attached_pending():
    """A settled dispatch marker's frozen ``attached`` CommitSeqs
    reconstruct the exact pending messages — self messages stay in the
    recent/presence history, never pending."""
    view = _marker_view(
        [
            _msg(text="a", msg_id="m1", recv_ts=100.0),
            _msg(text="self", msg_id="s1", is_self=True, recv_ts=110.0),
            _msg(text="hi", msg_id="m2", mentions=("bot-1",), recv_ts=120.0),
        ],
        schedule=[((1, 2, 3), "completed")],
        settled_ts_base=120.0,
    )
    result = _marker_replay(view)
    assert result.decisions == 1
    trace = result.traces[0]
    assert trace.snapshot_facts["cycle_id"] == "dispatch:1"
    assert trace.snapshot_facts["pending"] == 2  # self excluded
    assert trace.snapshot_facts["window_count"] == 3  # self counted
    assert trace.snapshot_facts["self_count"] == 1
    assert _decision(trace).action == "trigger"


def test_marker_replay_keeps_message_row_id_distinct_from_commit_seq():
    view = _marker_view(
        [_msg(text="row-id", msg_id="m-row", recv_ts=100.0)],
        message_row_ids=[41],
        schedule=[((1,), "completed")],
    )
    result = _marker_replay(view)
    assert result.traces[0].snapshot_facts["through_msg_id"] == 41


def test_marker_replay_raw_uncommitted_events_ignored():
    """A raw event WITHOUT a durable commit marker is ignored: it never
    enters the message timeline and never joins a pending batch."""
    view = _marker_view(
        [_msg(text="hi", msg_id="m1", mentions=("bot-1",), recv_ts=100.0)],
        schedule=[((1,), "completed")],
        settled_ts_base=100.0,
    )
    # An extra raw event with NO commit marker (e.g. a crash before the
    # commit, or a non-message event): it must be ignored.
    view = CorpusView(
        events_by_event_id={
            **view.events_by_event_id,
            EventId("ev-ghost"): AdapterEvent(
                type="message",
                payload=_msg(text="ghost", msg_id="g1", recv_ts=100.0),
                ts=100.0,
            ),
        },
        commits=view.commits,
        dispatches=view.dispatches,
    )
    result = _marker_replay(view)
    assert result.decisions == 1
    assert result.traces[0].snapshot_facts["pending"] == 1  # only m1
    assert result.traces[0].snapshot_facts["window_count"] == 1  # ghost ignored


def test_marker_replay_self_echo_recent_but_not_pending():
    """A committed self echo (wake_kind none) is preserved in the recent /
    presence history but never pending — even when it is the only attached
    commit of a dispatch."""
    view = _marker_view(
        [_msg(text="self", msg_id="s1", is_self=True, recv_ts=100.0)],
        schedule=[((1,), "released")],
        settled_ts_base=100.0,
    )
    result = _marker_replay(view)
    assert result.decisions == 1
    trace = result.traces[0]
    assert trace.snapshot_facts["pending"] == 0  # self never pending
    assert trace.snapshot_facts["window_count"] == 1  # but in the window
    assert trace.snapshot_facts["self_count"] == 1
    assert _decision(trace).action == "delay"


def test_marker_replay_release_vs_completed_dispatch():
    """A released (delay) dispatch leaves the cursor/session unchanged; a
    completed (terminal) dispatch advances the cursor and sets the
    previous end reason — reconstructed from the marker schedule."""
    view = _marker_view(
        [
            _msg(text="a", msg_id="m1", recv_ts=100.0),
            _msg(text="hi", msg_id="m2", mentions=("bot-1",), recv_ts=110.0),
            _msg(text="b", msg_id="m3", recv_ts=120.0),
        ],
        schedule=[((1,), "released"), ((2,), "completed"), ((3,), "released")],
        settled_ts_base=120.0,
    )
    result = _marker_replay(view)
    assert result.decisions == 3
    first, second, third = result.traces
    # Released dispatch: delay, no cursor advance, no terminal reason.
    assert _decision(first).action == "delay"
    assert first.snapshot_facts["previous_end_reason"] is None
    # Completed dispatch: terminal trigger, cursor advances.
    assert _decision(second).action == "trigger"
    assert second.snapshot_facts["previous_end_reason"] is None  # prior was delay
    assert second.snapshot_facts["pending"] == 1  # only m2 (m1 consumed)
    # The terminal reason is reconstructed for the NEXT dispatch.
    assert _decision(third).action == "delay"
    assert third.snapshot_facts["previous_end_reason"] == "dry_run_trigger"
    assert third.snapshot_facts["pending"] == 1  # only m3 (m1, m2 consumed)


def test_marker_replay_receding_timestamps_no_effect_on_dispatch_order():
    """Dispatches are re-scored in DispatchId order — receding timestamps
    never reorder them (the durable sequence is the only order)."""
    view = _marker_view(
        [
            _msg(text="a", msg_id="m1", recv_ts=200.0),
            _msg(text="b", msg_id="m2", recv_ts=100.0),  # recedes
        ],
        schedule=[((1,), "released"), ((2,), "released")],
        settled_ts_base=200.0,
    )
    result = _marker_replay(view)
    assert result.decisions == 2
    # Dispatch 1 evaluates m1 alone; dispatch 2 evaluates m2 alone — the
    # receding timestamp never reorders the dispatches.
    assert result.traces[0].snapshot_facts["cycle_id"] == "dispatch:1"
    assert result.traces[0].snapshot_facts["pending"] == 1
    assert result.traces[1].snapshot_facts["cycle_id"] == "dispatch:2"
    assert result.traces[1].snapshot_facts["pending"] == 1


def test_marker_replay_timer_before_inbound_reversed_export_order(tmp_path):
    """The core writer-order gap: a timer dispatch that writes first
    excludes a later commit, while a commit that writes first joins the
    dispatch. Even when the markers are physically exported in the
    REVERSED order, the frozen ``attached`` membership reconstructs the
    exact differing boundaries."""
    # Scenario A: timer writes first — boundary 0, nothing attached.
    timer_first = _marker_view(
        [_msg(text="m1", msg_id="m1", recv_ts=100.0)],
        schedule=[((), "released")],
        causes=[DispatchCause.TIMER],
        scheduled_fors=[200.0],
        settled_ts_base=100.0,
    )
    # Scenario B: commit writes first — the inbound dispatch attaches it.
    commit_first = _marker_view(
        [_msg(text="m1", msg_id="m1", recv_ts=100.0)],
        schedule=[((1,), "released")],
        settled_ts_base=100.0,
    )
    timer_result = _marker_replay(timer_first)
    commit_result = _marker_replay(commit_first)
    assert timer_result.traces[0].snapshot_facts["pending"] == 0
    assert commit_result.traces[0].snapshot_facts["pending"] == 1
    # Reversed export order: the dispatch marker physically precedes the
    # commit marker in the file, yet the attached membership still
    # reconstructs the exact boundary.
    path = _write_reversed_markers(commit_first, tmp_path / "reversed.jsonl")
    view = read_corpus_view(path)
    result = _marker_replay(view)
    assert result.traces[0].snapshot_facts["pending"] == 1


def _write_reversed_markers(view: CorpusView, path) -> str:
    """Write a corpus with the dispatch marker BEFORE the commit marker
    (the reversed export order) and return its path."""
    from pretender.record import Recorder

    with Recorder(path) as rec:
        for dispatch in view.dispatches:
            rec.append_marker(dispatch)
        for commit in view.commits:
            rec.append_marker(commit)
        for event_id, event in view.events_by_event_id.items():
            rec.write_event(event, event_id=event_id)
    return str(path)


def test_marker_replay_duplicate_marker_dedupe(tmp_path):
    """Duplicate markers (a crash re-append) are deduplicated by
    (record_type, sequence) — the corpus reader sees exactly one dispatch,
    so the replay re-scores it once."""
    view = _marker_view(
        [_msg(text="hi", msg_id="m1", mentions=("bot-1",), recv_ts=100.0)],
        schedule=[((1,), "completed")],
        settled_ts_base=100.0,
    )
    path = tmp_path / "dup.jsonl"
    from pretender.record import Recorder

    with Recorder(path) as rec:
        for event_id, event in view.events_by_event_id.items():
            rec.write_event(event, event_id=event_id)
        for commit in view.commits:
            rec.append_marker(commit)
        rec.append_marker(view.dispatches[0])
        rec.append_marker(view.dispatches[0])  # crash duplicate
    read = read_corpus_view(path)
    assert len(read.dispatches) == 1  # deduplicated
    result = _marker_replay(read)
    assert result.decisions == 1
    assert result.traces[0].snapshot_facts["cycle_id"] == "dispatch:1"


def test_marker_replay_skips_v2_v3_dispatch_markers():
    """A v2/v3 dispatch marker (no settled state / settled_ts) is skipped
    gracefully — old corpora stay readable, but only v4 markers replay."""
    view = _marker_view(
        [_msg(text="hi", msg_id="m1", mentions=("bot-1",), recv_ts=100.0)],
        schedule=[((1,), "completed")],
        settled_ts_base=100.0,
    )
    old = CorpusMarker(
        record_type="dispatch",
        sequence=DispatchId(9),
        chat_key=CK,
        cause=DispatchCause.INBOUND,
        commit_boundary=CommitSeq(1),
        scheduled_for=None,
        state=None,  # v2/v3: no settled metadata
        settled_ts=None,
    )
    view = CorpusView(
        events_by_event_id=view.events_by_event_id,
        commits=view.commits,
        dispatches=view.dispatches + (old,),
    )
    result = _marker_replay(view)
    assert result.decisions == 1  # only the v4 dispatch replayed


# ── Gate 5 remediation: exact replay fails closed, never omits decisions ─────

def test_marker_replay_fails_closed_on_settled_state_missing_settled_ts():
    """A dispatch marker with a settled state but no settled timestamp is
    invalid: exact replay fails closed instead of silently omitting it."""
    view = _marker_view(
        [_msg(text="hi", msg_id="m1", mentions=("bot-1",), recv_ts=100.0)],
        schedule=[((1,), "completed")],
        settled_ts_base=100.0,
    )
    dispatch = dataclasses.replace(view.dispatches[0], settled_ts=None)
    view = CorpusView(
        events_by_event_id=view.events_by_event_id,
        commits=view.commits,
        dispatches=(dispatch,),
    )
    with pytest.raises(ValueError, match="settled_ts"):
        _marker_replay(view)


def test_marker_replay_fails_closed_on_settled_ts_without_state():
    """A dispatch marker carrying a settled timestamp but no settled state
    is inconsistent: exact replay fails closed."""
    view = _marker_view(
        [_msg(text="hi", msg_id="m1", mentions=("bot-1",), recv_ts=100.0)],
        schedule=[((1,), None)],  # state None but settled_ts set
        settled_ts_base=100.0,
    )
    with pytest.raises(ValueError, match="invalid dispatch marker"):
        _marker_replay(view)


def test_marker_replay_fails_closed_on_unresolved_attached_commit():
    """A dispatch marker whose frozen attached membership references a
    commit that does not exist in the corpus is a corrupted ledger: exact
    replay fails closed instead of silently dropping the member."""
    view = _marker_view(
        [_msg(text="hi", msg_id="m1", mentions=("bot-1",), recv_ts=100.0)],
        schedule=[((1, 2), "completed")],  # commit 2 does not exist
        settled_ts_base=100.0,
    )
    with pytest.raises(ValueError, match="not found in corpus"):
        _marker_replay(view)


def _snapshot_view(
    messages,
    *,
    chat_key=CK,
    attached=(1,),
    state="completed",
    settled_ts=100.0,
    evaluated_ts=100.0,
    snapshot_overrides=None,
) -> CorpusView:
    """A CorpusView whose single dispatch marker carries a frozen
    ``snapshot_json`` (the v5 exact-replay shape) consistent with the
    marker by default; ``snapshot_overrides`` corrupts specific snapshot
    fields for the inconsistency tests."""
    events: dict[EventId, AdapterEvent] = {}
    commits: list[CorpusMarker] = []
    for i, msg in enumerate(messages, start=1):
        ev = EventId(f"ev-{i}")
        events[ev] = AdapterEvent(type="message", payload=msg, ts=msg.recv_ts)
        commits.append(
            CorpusMarker(
                record_type="commit",
                sequence=CommitSeq(i),
                chat_key=chat_key,
                event_id=ev,
                wake_kind=WakeKind.INBOUND,
            )
        )
    through = max(attached) if attached else 0
    pending = [
        dataclasses.replace(messages[seq - 1], row_id=MessageRowId(seq))
        for seq in attached
    ]
    snapshot: dict[str, Any] = {
        "chat_key": chat_key,
        "cycle_id": "dispatch:1",
        "start_msg_id": 0,
        "through_msg_id": through,
        "evaluated_ts": evaluated_ts,
        "self_id": "bot-1",
        "mode": "reply_necessity",
        "threshold": 8,
        "trigger_score": 80,
        "frequency": 1.0,
        "pending": len(pending),
        "pending_messages": [dataclasses.asdict(m) for m in pending],
        "recent": [dataclasses.asdict(m) for m in pending],
        "window_count": len(pending),
        "self_count": 0,
        "last_nonself_ts": None,
        "idle_seconds": 0.0,
        "recent_average_interval": 0.0,
        "self_ratio": 0.0,
        "is_group": True,
        "is_focused": False,
        "last_message": dataclasses.asdict(pending[-1]) if pending else None,
        "self_name": "麦麦",
        "has_direct_at": False,
        "has_quote_to_self": False,
        "has_other_assistant": False,
        "hold_until": None,
        "idle_streak": 0,
        "previous_end_reason": None,
        "backoff_base_s": 15.0,
        "backoff_cap_s": 300.0,
        "backoff_start_count": 2,
    }
    if snapshot_overrides:
        snapshot.update(snapshot_overrides)
    dispatch = CorpusMarker(
        record_type="dispatch",
        sequence=DispatchId(1),
        chat_key=chat_key,
        cause=DispatchCause.INBOUND,
        commit_boundary=CommitSeq(through),
        scheduled_for=None,
        state=state,
        settled_ts=settled_ts,
        start_msg_id=MessageRowId(0),
        through_msg_id=MessageRowId(through),
        attached=tuple(CommitSeq(s) for s in attached),
        # Snapshot fixtures are settled dispatch witnesses too; carry the
        # same default composition proof as the non-snapshot helper.
        trace_json=json.dumps({"config": _composition_fingerprint(Gate())}),
        evaluated_ts=evaluated_ts,
        snapshot_json=json.dumps(snapshot, default=str),
    )
    return CorpusView(
        events_by_event_id=events,
        commits=tuple(commits),
        dispatches=(dispatch,),
    )


def test_marker_replay_frozen_snapshot_consistent_replays():
    """A frozen snapshot consistent with its marker (chat, boundary,
    evaluated_ts, attached membership) re-scores exactly once."""
    view = _snapshot_view(
        [_msg(text="hi", msg_id="m1", mentions=("bot-1",), recv_ts=100.0)],
        attached=(1,),
        settled_ts=100.0,
        evaluated_ts=100.0,
    )
    result = _marker_replay(view)
    assert result.decisions == 1
    assert result.traces[0].snapshot_facts["cycle_id"] == "dispatch:1"
    assert result.traces[0].snapshot_facts["pending"] == 1


def test_marker_replay_fails_closed_on_frozen_snapshot_chat_mismatch():
    """A frozen snapshot whose chat_key does not match its marker is an
    inconsistency: exact replay fails closed instead of silently skipping
    the settled decision."""
    view = _snapshot_view(
        [_msg(text="hi", msg_id="m1", mentions=("bot-1",), recv_ts=100.0)],
        attached=(1,),
        settled_ts=100.0,
        evaluated_ts=100.0,
        snapshot_overrides={"chat_key": "qq:group:other"},
    )
    with pytest.raises(ValueError, match="chat mismatch"):
        _marker_replay(view)


def test_marker_replay_fails_closed_on_frozen_snapshot_cycle_mismatch():
    view = _snapshot_view(
        [_msg(text="hi", msg_id="m1", mentions=("bot-1",), recv_ts=100.0)],
        snapshot_overrides={"cycle_id": "dispatch:99"},
    )
    with pytest.raises(ValueError, match="cycle_id mismatch"):
        _marker_replay(view)


def test_marker_replay_fails_closed_on_frozen_snapshot_boundary_mismatch():
    """A frozen snapshot whose start/through boundary does not match its
    marker is an inconsistency: exact replay fails closed."""
    view = _snapshot_view(
        [_msg(text="hi", msg_id="m1", mentions=("bot-1",), recv_ts=100.0)],
        attached=(1,),
        settled_ts=100.0,
        evaluated_ts=100.0,
        snapshot_overrides={"through_msg_id": 5},
    )
    with pytest.raises(ValueError, match="through boundary mismatch"):
        _marker_replay(view)


def test_marker_replay_fails_closed_on_frozen_snapshot_evaluated_ts_mismatch():
    """A frozen snapshot whose evaluated_ts does not match its marker's
    frozen evaluation timestamp is an inconsistency: exact replay fails
    closed."""
    view = _snapshot_view(
        [_msg(text="hi", msg_id="m1", mentions=("bot-1",), recv_ts=100.0)],
        attached=(1,),
        settled_ts=100.0,
        evaluated_ts=100.0,
        snapshot_overrides={"evaluated_ts": 999.0},
    )
    with pytest.raises(ValueError, match="evaluated_ts mismatch"):
        _marker_replay(view)


def test_marker_replay_fails_closed_on_frozen_snapshot_membership_mismatch():
    """A frozen snapshot whose pending membership does not match the
    marker's frozen attached set is an inconsistency: exact replay fails
    closed."""
    view = _snapshot_view(
        [
            _msg(text="a", msg_id="m1", recv_ts=100.0),
            _msg(text="b", msg_id="m2", recv_ts=110.0),
        ],
        attached=(1, 2),
        settled_ts=110.0,
        evaluated_ts=110.0,
        snapshot_overrides={
            "pending": 1,
            "pending_messages": [
                dataclasses.asdict(
                    dataclasses.replace(
                        _msg(text="a", msg_id="m1", recv_ts=100.0),
                        row_id=MessageRowId(1),
                    )
                )
            ]
        },
    )
    with pytest.raises(ValueError, match="membership mismatch"):
        _marker_replay(view)


def test_marker_sweep_rescores_fixed_recorded_schedule():
    """Sweep re-scores the FIXED recorded dispatch schedule under each
    RuntimeOverlay combination — no counterfactual future timer events are
    invented, and the dispatch schedule (attached/boundary/time) is
    unchanged."""
    messages = [
        _msg(text="x", msg_id=f"m{i}", recv_ts=100.0 + i) for i in range(1, 11)
    ]
    # Every dispatch attaches ALL accumulated commits (delay dispatches
    # release the claim, so commits stay pending).
    schedule = [tuple(range(1, i + 1)) for i in range(1, 11)]
    view = _marker_view(
        messages,
        schedule=[(attached, "released") for attached in schedule],
        settled_ts_base=100.0,
    )
    rows = sweep_marker_schedule(view, chat_key=CK, identity=make_identity(), cfg=Config())
    assert len(rows) == 16  # 4 thresholds x 4 trigger scores
    assert rows[0].threshold == 2 and rows[0].trigger_score == 40
    assert rows[-1].threshold == 12 and rows[-1].trigger_score == 100
    assert all(row.decisions == 10 for row in rows)  # the fixed schedule
    # threshold 2 triggers every 2nd accumulated batch (5/10); the default
    # threshold 8 never crosses the trigger score for ambient messages.
    assert rows[0].would_have_spoken == 5
    assert any(row.would_have_spoken != rows[0].would_have_spoken for row in rows)


def test_marker_replay_zero_sends_and_no_side_effects(tmp_path):
    """Marker replay is storage-free: it never creates outbox rows, never
    invokes an adapter send, and never mutates the corpus."""
    view = _marker_view(
        [_msg(text="hi", msg_id="m1", mentions=("bot-1",), recv_ts=100.0)],
        schedule=[((1,), "completed")],
        settled_ts_base=100.0,
    )
    path = _write_reversed_markers(view, tmp_path / "zero.jsonl")
    before = open(path, encoding="utf-8").read()
    result = _marker_replay(view)
    assert result.decisions == 1
    assert result.would_have_spoken == 1
    # The corpus file is untouched (no marker/event appended).
    assert open(path, encoding="utf-8").read() == before


# ── full real SQLite dry-run App: markers then marker replay equality ───────

class _BlockingStream:
    """A REPL input stream that blocks in readline until ``release`` is
    called, then returns EOF — lets a test observe the ledger scheduler's
    dispatch before the run loop ends."""

    def __init__(self) -> None:
        self._release = threading.Event()

    def readline(self) -> str:
        self._release.wait()
        return ""

    def release(self) -> None:
        self._release.set()


def _make_app(tmp_path, **kw):
    cfg = Config.from_dict({"storage": {"db_path": str(tmp_path / "data" / "app.db")}})
    clock = kw.pop("clock", VirtualClock(auto_advance=False))
    input_stream = kw.pop("input_stream", io.StringIO(""))
    adapter = ConsoleAdapter(
        clock=clock, input_stream=input_stream, output_stream=io.StringIO()
    )
    return App.build(cfg, clock=clock, adapter=adapter, **kw)


async def _wait_traces(traces: list, n: int) -> None:
    """Yield with real-time sleeps until ``n`` traces were produced (the
    writer's coalescing window is wall-clock)."""
    for _ in range(2000):
        if len(traces) >= n:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"only {len(traces)} traces after 2000 polls")


def test_marker_replay_full_sqlite_app_trace_equality_by_dispatch_id(tmp_path):
    """The full loop: a real SQLite dry-run App produces event + commit +
    settled dispatch markers, and the marker-driven replay re-scores the
    recorded dispatch schedule with trace equality by DispatchId — and
    zero sends (empty outbox, no adapter send)."""
    clock = VirtualClock(auto_advance=False)
    traces: list[DecisionTrace] = []
    trace_event = asyncio.Event()
    stream = _BlockingStream()
    app = _make_app(
        tmp_path,
        clock=clock,
        input_stream=stream,
        dry_run=True,
        trace_sink=lambda t: (traces.append(t), trace_event.set()),
    )

    async def scenario():
        await app.start()
        run_task = asyncio.create_task(app.run())

        async def feed(text: str, *, mentions: tuple = ()) -> None:
            msg = make_message(
                chat_key="console:group:demo",
                text=text,
                msg_id=None,
                mentions=mentions,
                recv_ts=clock.now(),
            )
            await app._ingest_batched(
                AdapterEvent(type="message", payload=msg, ts=msg.recv_ts)
            )

        await feed("hi", mentions=("bot",))  # direct @ -> trigger (completed)
        await _wait_traces(traces, 1)
        await feed("hello")  # quiet -> delay (released)
        await _wait_traces(traces, 2)
        stream.release()
        await run_task
        view = read_corpus_view(tmp_path / "data" / "app.jsonl")
        return view

    view = run(scenario())
    # The App produced event + commit + settled dispatch markers.
    assert len(view.commits) == 2
    assert len(view.dispatches) == 2
    assert [m.state for m in view.dispatches] == ["completed", "released"]
    assert [m.attached for m in view.dispatches] == [(CommitSeq(1),), (CommitSeq(2),)]
    # Zero sends: the dry-run App never drained the outbox or sent.
    assert app.adapter is not None
    assert app.adapter.sent == []
    # Marker replay: trace equality by DispatchId.
    identity = make_identity(
        chat_key="console:group:demo", platform="console", self_id="bot"
    )
    result = replay_marker_schedule(
        view, chat_key=view.dispatches[0].chat_key, identity=identity, cfg=app.cfg
    )
    assert result.decisions == len(traces) == 2
    for live, replay in zip(traces, result.traces):
        assert live.decision == replay.decision
        assert live.aggregates == replay.aggregates
        assert live.backoff == replay.backoff
        lf = {k: v for k, v in live.snapshot_facts.items() if k != "cycle_id"}
        rf = {k: v for k, v in replay.snapshot_facts.items() if k != "cycle_id"}
        assert lf == rf
    # The replay traces identify their DispatchId.
    assert [t.snapshot_facts["cycle_id"] for t in result.traces] == [
        "dispatch:1",
        "dispatch:2",
    ]
