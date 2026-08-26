"""Durable session state helpers for one chat (PLAN.md §2, RUNTIME).

Everything here is CURSOR-INDEPENDENT: the per-chat cursor is a read-only
view written only by ``finish_cycle`` (frozen decision #2), so the session
layer never touches it. The durable ``ChatState`` (focus window, EWMA
message interval, config overlay) is immutable; every transition returns a
NEW ``ChatState`` via ``dataclasses.replace``, and the repository swaps
whole states.

The hold window (``hold_until``) and the idle streak (``idle_streak``) are
TERMINAL-OWNED: they are written only by ``finish_cycle``, transactionally
with the terminal outcome (``CycleFinish.hold_until`` /
``idle_streak_after``). The session layer therefore exposes NO mutators for
them — ``is_held`` is a read-only view, and ``upsert_chat_state`` never
persists them. The pure ``update_avg_interval`` EWMA helper stays
importable here (a thin wrapper over the dependency-neutral
``pacing.ewma_interval`` reducer) so the persistence lane can use the SAME
reducer without importing the runtime ``Session``.

Two session facts have no durable column and live in memory alongside the
state:

  - the 300-second self-ratio ring (``SelfRatioRing``): fed by messages as
    they arrive, read at gate time. Self messages ARE included in the
    window counts — the presence penalty reads the full window (PLAN.md
    §1.B).
  - the wait streak: the ``wait`` tool's consecutive-pause counter, capped
    at 3; hitting the cap forces a rest (PLAN.md §1.B).

``Session`` wraps one chat's durable state plus those in-memory facts. It
is MUTABLE (the working copy the cycle lane loads once, transitions over
the cycle, and persists via ``save_session``); the underlying ChatState
stays immutable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

from pretender.pacing import EWMA_ALPHA, ewma_interval
from pretender.seams import Repository
from pretender.types import ChatKey, ChatState

# EWMA smoothing factor for the message-interval average (re-exported from
# the dependency-neutral pacing module — the SAME reducer the repository's
# atomic ingest path uses, so the durable chats.avg_interval and any
# in-memory session average can never drift apart).
EWMA_ALPHA = EWMA_ALPHA
# The presence-penalty window: self ratio over the last 300 s (PLAN.md §1.B).
RATIO_WINDOW_S = 300.0
# Ring capacity: 512 entries is ~1.7/s sustained over the window — far
# beyond any human chat rate, and it bounds memory for pathological groups.
RING_CAPACITY = 512
# Consecutive `wait` pauses cap at 3; hitting the cap forces a rest.
WAIT_STREAK_CAP = 3


# ── EWMA message interval ───────────────────────────────────────────────────

def update_avg_interval(state: ChatState, prev_ts: float, now: float) -> ChatState:
    """EWMA of the inter-message interval, seeded by the first sample.

    Thin wrapper over the dependency-neutral ``pacing.ewma_interval``
    reducer — the SAME reducer the repository's atomic ingest path uses,
    so the durable ``chats.avg_interval`` and the in-memory session
    average can never drift apart. Non-positive gaps (clock skew,
    same-timestamp batches) are ignored: they carry no pacing information
    and would drag the average toward 0. The average is always positive
    once seeded, so idle compensation
    (``idle_seconds / recent_average_interval``) never divides by zero.
    """
    avg = ewma_interval(state.avg_interval, prev_ts, now)
    if avg is None:
        return state
    return replace(state, avg_interval=avg)


# ── Self-ratio ring over the 300-second window ──────────────────────────────

class SelfRatioRing:
    """The presence window: ``(timestamp, is_self)`` entries over the last
    ``window_s`` seconds.

    ``push`` appends a message as it arrives (oldest dropped beyond
    capacity); reads prune entries older than the window lazily.
    ``self_ratio`` is ``self_count / window_count`` over the window — self
    messages INCLUDED in both counts. An empty window reads 0.0.
    """

    def __init__(
        self, window_s: float = RATIO_WINDOW_S, capacity: int = RING_CAPACITY
    ) -> None:
        self._window_s = window_s
        self._capacity = capacity
        self._entries: deque[tuple[float, bool]] = deque()

    def push(self, ts: float, is_self: bool) -> None:
        self._entries.append((ts, is_self))
        if len(self._entries) > self._capacity:
            self._entries.popleft()

    def prune(self, now: float) -> None:
        """Drop entries strictly older than ``now - window_s``; an entry at
        exactly the cutoff stays in the window (inclusive bound)."""
        cutoff = now - self._window_s
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()

    def counts(self, now: float) -> tuple[int, int]:
        """``(window_count, self_count)`` over the window at ``now``."""
        self.prune(now)
        self_count = sum(1 for _ts, is_self in self._entries if is_self)
        return len(self._entries), self_count

    def self_ratio(self, now: float) -> float:
        window_count, self_count = self.counts(now)
        if window_count == 0:
            return 0.0
        return self_count / window_count


# ── Focus / hold windows ────────────────────────────────────────────────────

def is_focused(state: ChatState, now: float) -> bool:
    """Focus is active while ``focus_until`` is strictly in the future; a
    window that expired at exactly ``now`` is not focused."""
    return state.focus_until is not None and now < state.focus_until


def is_held(state: ChatState, now: float) -> bool:
    """A held (backoff) outcome is active while ``hold_until`` is strictly
    in the future. READ-ONLY view: ``hold_until`` is written ONLY by
    ``finish_cycle`` (types.py), never by the session layer — there is no
    session mutator for it."""
    return state.hold_until is not None and now < state.hold_until


def set_focus(state: ChatState, until: float) -> ChatState:
    return replace(state, focus_until=until)


def clear_focus(state: ChatState) -> ChatState:
    return replace(state, focus_until=None)


# ── Idle backoff ────────────────────────────────────────────────────────────

def backoff_seconds(
    streak: int,
    *,
    base_s: float = 15.0,
    cap_s: float = 300.0,
    start_count: int = 2,
) -> float:
    """Idle backoff for ``streak`` consecutive idle cycles (PLAN.md §1.B):
    ``min(cap, base * 2**(streak - start))`` once ``streak >= start``,
    else 0. Defaults: base 15 s, cap 300 s, start 2 — so the second
    consecutive idle cycle backs off 15 s, then 30, 60, 120, 240, capped
    at 300.
    """
    if streak < start_count:
        return 0.0
    return min(cap_s, base_s * 2.0 ** (streak - start_count))


# ── Session: durable state + in-memory facts ────────────────────────────────

class Session:
    """One chat's runtime session: the durable ``ChatState`` plus the
    in-memory facts that have no durable column (the 300 s self-ratio ring
    and the wait streak).

    MUTABLE working copy: transitions swap ``self.state`` in place (the
    ChatState itself stays immutable); the cycle lane loads once, mutates
    over the cycle, and persists via ``save_session``.
    """

    def __init__(self, state: ChatState) -> None:
        self._state = state
        self._ring = SelfRatioRing()
        # Seed the in-memory wait streak from the durable state: the
        # durable ``wait_streak`` is DEFER/TERMINAL OWNED (written only by
        # settle_dispatch / finish), so the session reads it as a view and
        # never persists it (save_session cannot move it).
        self._wait_streak = state.wait_streak

    @property
    def state(self) -> ChatState:
        return self._state

    @property
    def wait_streak(self) -> int:
        return self._wait_streak

    @property
    def wait_capped(self) -> bool:
        """True once the streak reached the cap: a rest is forced."""
        return self._wait_streak >= WAIT_STREAK_CAP

    # ── pacing facts ──

    def update_interval(self, prev_ts: float, now: float) -> None:
        self._state = update_avg_interval(self._state, prev_ts, now)

    def note_message(self, ts: float, is_self: bool) -> None:
        """Feed the presence ring with one message (self included)."""
        self._ring.push(ts, is_self)

    def self_ratio(self, now: float) -> float:
        return self._ring.self_ratio(now)

    def window_counts(self, now: float) -> tuple[int, int]:
        return self._ring.counts(now)

    # ── wait streak ──

    def note_wait(self) -> int:
        """Record one ``wait`` pause. Returns the new streak; the streak is
        capped at ``WAIT_STREAK_CAP`` — hitting the cap forces a rest."""
        self._wait_streak = min(self._wait_streak + 1, WAIT_STREAK_CAP)
        return self._wait_streak

    def reset_wait(self) -> None:
        self._wait_streak = 0

    # ── focus window (session-owned; hold/idle are terminal-owned) ──

    def set_focus(self, until: float) -> None:
        self._state = replace(self._state, focus_until=until)

    def clear_focus(self) -> None:
        self._state = replace(self._state, focus_until=None)

    def is_focused(self, now: float) -> bool:
        return is_focused(self._state, now)

    def is_held(self, now: float) -> bool:
        """Read-only view of the terminal-owned hold window."""
        return is_held(self._state, now)


# ── Persistence (typed against the Repository seam) ─────────────────────────

async def load_session(repo: Repository, chat_key: ChatKey) -> Session:
    """Load the durable state for one chat; a fresh default state when the
    chat has no row yet. Never touches the cursor."""
    state = await repo.get_chat_state(chat_key)
    return Session(state if state is not None else ChatState(chat_key=chat_key))


async def save_session(repo: Repository, session: Session) -> None:
    """Persist the durable part of a session. ``upsert_chat_state`` writes
    focus/pacing/config only — never the cursor, the hold window, the idle
    streak, or the agent barrier (``agent_resume_at`` / ``wait_streak``):
    those are terminal/defer-owned (written only by ``finish_cycle`` /
    ``settle_dispatch``, transactionally with the terminal or defer
    outcome). The session layer exposes no mutators for them, so a save can
    never reintroduce a crash gap."""
    await repo.upsert_chat_state(session.state)