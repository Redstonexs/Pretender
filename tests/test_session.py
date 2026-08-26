"""Session state helpers: EWMA interval, self-ratio ring, focus/hold,
wait streak cap, idle/backoff transitions, and persistence.

Async tests run via asyncio.run() so the test extra stays at just pytest.
"""

from __future__ import annotations

import asyncio

import pytest

import pretender.session as session_module
from pretender.session import (
    RATIO_WINDOW_S,
    WAIT_STREAK_CAP,
    SelfRatioRing,
    Session,
    backoff_seconds,
    clear_focus,
    is_focused,
    is_held,
    load_session,
    save_session,
    set_focus,
    update_avg_interval,
)
from pretender.types import ChatKey, ChatState, MessageRowId
from tests.durable_helpers import FakeRepo

CK = ChatKey("qq:group:123456")
EPOCH = 1_700_000_000.0


def run(coro):
    return asyncio.run(coro)


# ── EWMA message interval ───────────────────────────────────────────────────

def test_ewma_seeded_by_first_sample():
    state = ChatState(chat_key=CK)
    updated = update_avg_interval(state, EPOCH, EPOCH + 30.0)
    assert updated.avg_interval == pytest.approx(30.0)


def test_ewma_alpha_weighting():
    state = ChatState(chat_key=CK)
    state = update_avg_interval(state, EPOCH, EPOCH + 10.0)  # seed 10
    state = update_avg_interval(state, EPOCH + 10.0, EPOCH + 30.0)  # gap 20
    # 0.5 * 20 + 0.5 * 10 = 15
    assert state.avg_interval == pytest.approx(15.0)


def test_ewma_converges_toward_steady_state():
    state = ChatState(chat_key=CK)
    state = update_avg_interval(state, EPOCH, EPOCH + 10.0)  # seed 10
    for i in range(200):
        state = update_avg_interval(state, EPOCH + i * 10.0, EPOCH + i * 10.0 + 20.0)
    assert state.avg_interval == pytest.approx(20.0, abs=1e-6)


def test_ewma_ignores_non_positive_gaps():
    state = ChatState(chat_key=CK, avg_interval=30.0)
    # zero gap (same-timestamp batch) and negative gap (clock skew) carry
    # no pacing information and must not drag the average toward 0.
    assert update_avg_interval(state, EPOCH + 50.0, EPOCH + 50.0).avg_interval == 30.0
    assert update_avg_interval(state, EPOCH + 60.0, EPOCH + 50.0).avg_interval == 30.0


def test_ewma_never_zero_once_seeded():
    # Idle compensation divides by the average; it must stay positive.
    state = ChatState(chat_key=CK)
    state = update_avg_interval(state, EPOCH, EPOCH + 1.0)
    for i in range(100):
        state = update_avg_interval(state, EPOCH + i, EPOCH + i + 1.0)
    assert state.avg_interval is not None and state.avg_interval > 0


def test_session_reducer_is_the_centralized_pacing_reducer():
    # The session wrapper and the dependency-neutral pacing reducer are
    # ONE reducer: identical inputs produce identical averages, and the
    # smoothing factor is shared (the repository's atomic ingest path
    # consumes the same module, so durable and in-memory averages can
    # never drift apart).
    from pretender.pacing import EWMA_ALPHA as PACING_ALPHA
    from pretender.pacing import ewma_interval

    assert session_module.EWMA_ALPHA == PACING_ALPHA == 0.5
    state = ChatState(chat_key=CK)
    prev_ts = EPOCH
    for ts in (EPOCH + 10.0, EPOCH + 30.0, EPOCH + 45.0):
        prior = state.avg_interval
        state = update_avg_interval(state, prev_ts, ts)
        expected = ewma_interval(prior, prev_ts, ts)
        assert state.avg_interval == pytest.approx(expected)
        prev_ts = ts
    # Non-positive gaps behave identically in both: the wrapper leaves the
    # state untouched, the reducer reports "no sample" (None).
    unchanged = update_avg_interval(state, prev_ts, prev_ts)
    assert unchanged.avg_interval == state.avg_interval
    assert ewma_interval(state.avg_interval, prev_ts, prev_ts) is None


# ── Self-ratio ring over the 300-second window ──────────────────────────────

def test_ring_empty_window_ratio_zero():
    ring = SelfRatioRing()
    assert ring.counts(EPOCH) == (0, 0)
    assert ring.self_ratio(EPOCH) == 0.0


def test_ring_ratio_includes_self_messages():
    ring = SelfRatioRing()
    ring.push(EPOCH - 100.0, False)
    ring.push(EPOCH - 50.0, True)
    ring.push(EPOCH - 10.0, True)
    assert ring.counts(EPOCH) == (3, 2)
    assert ring.self_ratio(EPOCH) == pytest.approx(2 / 3)


def test_ring_all_self_ratio_one():
    ring = SelfRatioRing()
    for i in range(5):
        ring.push(EPOCH - float(i), True)
    assert ring.self_ratio(EPOCH) == 1.0


def test_ring_prunes_at_window_boundary():
    ring = SelfRatioRing()
    ring.push(EPOCH - RATIO_WINDOW_S - 1.0, False)  # strictly older: dropped
    ring.push(EPOCH - RATIO_WINDOW_S, True)  # exactly at the cutoff: stays
    ring.push(EPOCH - 10.0, False)
    assert ring.counts(EPOCH) == (2, 1)
    assert ring.self_ratio(EPOCH) == pytest.approx(0.5)


def test_ring_capacity_bounds_memory():
    ring = SelfRatioRing(capacity=4)
    for i in range(10):
        ring.push(EPOCH + float(i), i % 2 == 0)
    # Only the newest 4 survive; the oldest 6 are dropped.
    assert ring.counts(EPOCH + 100.0) == (4, 2)


# ── Focus / hold windows ────────────────────────────────────────────────────

def test_focus_helpers():
    state = ChatState(chat_key=CK)
    assert not is_focused(state, EPOCH)
    state = set_focus(state, EPOCH + 600.0)
    assert is_focused(state, EPOCH)
    assert is_focused(state, EPOCH + 599.999)
    assert not is_focused(state, EPOCH + 600.0)  # expired at exactly now
    state = clear_focus(state)
    assert not is_focused(state, EPOCH)


def test_hold_is_terminal_owned_read_only_view():
    # The hold window is written ONLY by finish_cycle: the session layer
    # exposes no mutator for it — is_held is a read-only view over the
    # durable state.
    state = ChatState(chat_key=CK, hold_until=EPOCH + 300.0)
    assert is_held(state, EPOCH + 299.0)
    assert not is_held(state, EPOCH + 300.0)
    assert not is_held(ChatState(chat_key=CK), EPOCH)
    # No session mutator exists: a save can never move the hold window.
    assert not hasattr(Session, "set_hold")
    assert not hasattr(Session, "clear_hold")
    assert not hasattr(session_module, "set_hold")
    assert not hasattr(session_module, "clear_hold")


# ── Wait streak cap ─────────────────────────────────────────────────────────

def test_wait_streak_caps_at_three():
    s = Session(ChatState(chat_key=CK))
    assert s.note_wait() == 1
    assert s.note_wait() == 2
    assert s.note_wait() == 3
    assert s.wait_capped
    assert s.note_wait() == 3  # capped: no further growth
    assert s.wait_streak == WAIT_STREAK_CAP
    s.reset_wait()
    assert s.wait_streak == 0
    assert not s.wait_capped


# ── Idle streak / backoff are terminal-owned ────────────────────────────────

def test_idle_streak_is_terminal_owned():
    # The idle streak is written ONLY by finish_cycle (CycleFinish
    # idle_streak_after): the session layer exposes no mutator for it.
    s = Session(ChatState(chat_key=CK))
    assert s.state.idle_streak == 0
    assert not hasattr(Session, "note_idle_cycle")
    assert not hasattr(Session, "reset_idle")
    # The read-only view still reflects the durable state.
    held = Session(ChatState(chat_key=CK, idle_streak=3))
    assert held.state.idle_streak == 3


def test_backoff_seconds_boundaries():
    # Before start_count: no backoff.
    assert backoff_seconds(0) == 0.0
    assert backoff_seconds(1) == 0.0
    # start_count: base, then doubling.
    assert backoff_seconds(2) == 15.0
    assert backoff_seconds(3) == 30.0
    assert backoff_seconds(4) == 60.0
    assert backoff_seconds(5) == 120.0
    assert backoff_seconds(6) == 240.0
    # Capped at cap_s.
    assert backoff_seconds(7) == 300.0
    assert backoff_seconds(100) == 300.0
    # Custom parameters.
    assert backoff_seconds(1, base_s=10.0, cap_s=40.0, start_count=1) == 10.0
    assert backoff_seconds(3, base_s=10.0, cap_s=40.0, start_count=1) == 40.0


# ── Session: durable + in-memory facts together ─────────────────────────────

def test_session_combines_durable_and_in_memory_facts():
    s = Session(ChatState(chat_key=CK))
    s.update_interval(EPOCH, EPOCH + 30.0)
    s.note_message(EPOCH + 30.0, False)
    s.note_message(EPOCH + 60.0, True)
    assert s.state.avg_interval == pytest.approx(30.0)
    assert s.window_counts(EPOCH + 60.0) == (2, 1)
    assert s.self_ratio(EPOCH + 60.0) == pytest.approx(0.5)
    # The idle streak is terminal-owned: the session never grows it, and
    # the pure backoff formula stays available for the persistence lane.
    assert s.state.idle_streak == 0
    assert backoff_seconds(2) == 15.0


def test_session_focus():
    s = Session(ChatState(chat_key=CK))
    s.set_focus(EPOCH + 600.0)
    assert s.is_focused(EPOCH)
    assert not s.is_focused(EPOCH + 600.0)
    s.clear_focus()
    assert not s.is_focused(EPOCH)
    # is_held is a read-only view of the terminal-owned hold window.
    s2 = Session(ChatState(chat_key=CK, hold_until=EPOCH + 300.0))
    assert s2.is_held(EPOCH)
    assert not s2.is_held(EPOCH + 300.0)


# ── Persistence (typed against the Repository seam) ─────────────────────────

class StateRepo(FakeRepo):
    """Protocol-complete Repository fake with real chat-state storage:
    get/upsert round-trip, everything else inherited from FakeRepo."""

    def __init__(self) -> None:
        super().__init__()
        self.states: dict[ChatKey, ChatState] = {}
        self.upserts: list[ChatState] = []

    async def get_chat_state(self, chat_key: ChatKey):  # type: ignore[override]
        return self.states.get(chat_key)

    async def upsert_chat_state(self, state: ChatState) -> None:
        self.states[state.chat_key] = state
        self.upserts.append(state)


def test_load_session_fresh_default_when_no_row():
    repo = StateRepo()

    async def scenario():
        s = await load_session(repo, CK)
        assert s.state == ChatState(chat_key=CK)
        assert s.wait_streak == 0

    run(scenario())


def test_session_persistence_truth_focus_and_pacing_only():
    """The concrete persistence truth: ``save_session`` persists focus and
    pacing (EWMA interval) — never the cursor, the hold window, or the
    idle streak, which are terminal-owned (written only by
    ``finish_cycle``)."""
    repo = StateRepo()
    repo.states[CK] = ChatState(
        chat_key=CK,
        cursor_msg_id=MessageRowId(42),
        hold_until=EPOCH + 300.0,
        idle_streak=3,
        avg_interval=30.0,
    )

    async def scenario():
        s = await load_session(repo, CK)
        # The session mutates ONLY session-owned facts.
        s.update_interval(EPOCH, EPOCH + 60.0)
        s.set_focus(EPOCH + 600.0)
        await save_session(repo, s)
        s2 = await load_session(repo, CK)
        # Session-owned facts persisted.
        assert s2.state.focus_until == EPOCH + 600.0
        assert s2.state.avg_interval == pytest.approx(45.0)
        # Terminal-owned facts are untouched by the save: the session
        # layer never mutated them, so upsert_chat_state cannot move them.
        assert s2.state.cursor_msg_id == MessageRowId(42)
        assert s2.state.hold_until == EPOCH + 300.0
        assert s2.state.idle_streak == 3
        assert s2.is_held(EPOCH)

    run(scenario())


def test_session_preserves_cursor_view():
    # The cursor is a read-only view owned by finish_cycle; the session
    # layer must never clobber it.
    repo = StateRepo()
    repo.states[CK] = ChatState(
        chat_key=CK, cursor_msg_id=MessageRowId(42), avg_interval=30.0
    )

    async def scenario():
        s = await load_session(repo, CK)
        s.update_interval(EPOCH, EPOCH + 60.0)
        s.set_focus(EPOCH + 600.0)
        await save_session(repo, s)
        s2 = await load_session(repo, CK)
        assert s2.state.cursor_msg_id == MessageRowId(42)
        assert s2.state.avg_interval == pytest.approx(45.0)

    run(scenario())


# ── durable agent barrier is defer/terminal-owned ───────────────────────────

def test_session_seeds_wait_streak_from_durable_state():
    # The in-memory wait streak is seeded from the durable state (a
    # read-only view); the session never persists it.
    s = Session(ChatState(chat_key=CK, wait_streak=2))
    assert s.wait_streak == 2
    assert not s.wait_capped  # 2 < 3
    s2 = Session(ChatState(chat_key=CK, wait_streak=3))
    assert s2.wait_capped


def test_agent_barrier_is_defer_owned_read_only_view():
    # agent_resume_at / wait_streak are defer/terminal-owned (written only
    # by settle_dispatch / finish): the session layer exposes no mutators
    # for them, so a save can never move them.
    state = ChatState(chat_key=CK, agent_resume_at=EPOCH + 300.0, wait_streak=2)
    s = Session(state)
    assert s.state.agent_resume_at == EPOCH + 300.0
    assert s.state.wait_streak == 2
    assert not hasattr(Session, "set_agent_resume")
    assert not hasattr(Session, "set_wait_streak")
    assert not hasattr(session_module, "set_agent_resume")
    assert not hasattr(session_module, "set_wait_streak")