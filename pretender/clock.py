"""Time sources: RealClock for production, VirtualClock for deterministic tests.

Invariant from PLAN.md §4: persisted timestamps are absolute epoch seconds.
RealClock corrects intra-process drift with ``time.monotonic()`` — the
``epoch_at``/``monotonic_at`` pair converts between the two using an offset
calibrated at construction, so a monotonic reading taken now maps to the
epoch second it actually corresponds to.

VirtualClock never touches the wall clock: ``sleep`` jumps virtual time
forward and completes immediately (after yielding once), and ``advance``
wakes any sleepers whose deadline has passed. A 6-hour scheduling scenario
runs in milliseconds, with no busy polling.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from typing import Any

from pretender.seams import Clock


class RealClock:
    """The production clock: wall-clock epoch seconds + monotonic seconds.

    ``now`` is DERIVED from the epoch/monotonic calibration: the offset is
    measured once at construction, and every reading goes through
    ``epoch_at(monotonic())`` so ``now()`` and ``epoch_at``/``monotonic_at``
    are exactly consistent with each other (no drift between the two
    views of time).
    """

    def __init__(self) -> None:
        # Calibrate the epoch↔monotonic offset once; both reads happen
        # back-to-back so the offset error is sub-millisecond.
        self._epoch_offset = time.time() - time.monotonic()

    def now(self) -> float:
        """Absolute epoch seconds (UTC), via the calibrated offset. This is
        what gets persisted."""
        return self.epoch_at(self.monotonic())

    def monotonic(self) -> float:
        """Monotonic seconds since an arbitrary boot-relative origin."""
        return time.monotonic()

    def epoch_at(self, monotonic_ts: float) -> float:
        """Convert a monotonic reading to the epoch second it corresponds to."""
        return monotonic_ts + self._epoch_offset

    def monotonic_at(self, epoch_ts: float) -> float:
        """Inverse of ``epoch_at``."""
        return epoch_ts - self._epoch_offset

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class VirtualClock:
    """A deterministic clock for tests and replay.

    Virtual time starts at ``epoch`` (default: a fixed 2023-11-14 epoch so
    results are reproducible) and advances only when something sleeps or
    ``advance`` is called.

    Two modes, selected by ``auto_advance``:

    - ``auto_advance=True`` (default): ``await clock.sleep(x)`` jumps virtual
      time forward by ``x`` and completes immediately. Code that sleeps in
      production runs at full speed under test — a 6-hour scheduling
      scenario finishes in milliseconds. Nothing can hang.
    - ``auto_advance=False``: ``sleep`` registers a sleeper and waits;
      ``advance(seconds)`` moves time forward and wakes every sleeper whose
      deadline has passed. This is the granular pattern for observing
      intermediate states: drive the clock, assert, drive again — never
      busy-polling.

    ``advance`` is meaningful in both modes (it moves time even when no
    sleeper is waiting); in auto-advance mode sleepers wake themselves.
    """

    def __init__(self, epoch: float = 1_700_000_000.0, auto_advance: bool = True) -> None:
        self._epoch = float(epoch)
        self._auto_advance = auto_advance
        self._mono = 0.0
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = 0

    # ── Clock protocol ──────────────────────────────────────────────────────

    def now(self) -> float:
        return self._epoch + self._mono

    def monotonic(self) -> float:
        return self._mono

    def epoch_at(self, monotonic_ts: float) -> float:
        return self._epoch + monotonic_ts

    def monotonic_at(self, epoch_ts: float) -> float:
        return epoch_ts - self._epoch

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            # Yield once so other tasks get a turn, but do not advance time.
            await asyncio.sleep(0)
            return
        if self._auto_advance:
            # Register a sleeper at the SCHEDULED DEADLINE. ONLY the
            # earliest scheduled sleeper may advance the clock — to its own
            # deadline — so the clock reaches deadlines in order,
            # independent of task creation order: a long sleep created
            # first can never advance past a shorter one (it is not the
            # earliest), and concurrent sleeps complete at their own
            # deadlines (t+10, t+20), never at summed delays (t+30).
            fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._seq += 1
            seq = self._seq
            heapq.heappush(self._sleepers, (self._mono + seconds, seq, fut))
            try:
                while True:
                    await asyncio.sleep(0)
                    if self._sleepers and self._sleepers[0][1] == seq:
                        # I am the earliest scheduled sleeper: advance the
                        # clock to MY deadline and wake.
                        self._advance_to(self._sleepers[0][0])
                        return
                    if fut.done():
                        # Someone else advanced to my deadline (same-deadline
                        # sleepers wake together).
                        return
            finally:
                if not fut.done():
                    self._drop_sleeper(seq)
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._seq += 1
        heapq.heappush(self._sleepers, (self._mono + seconds, self._seq, fut))
        await fut

    # ── Test control ────────────────────────────────────────────────────────

    def advance(self, seconds: float) -> None:
        """Move virtual time forward by ``seconds``, waking due sleepers."""
        if seconds < 0:
            raise ValueError(f"advance must be >= 0, got {seconds}")
        self._advance_to(self._mono + seconds)

    def _advance_to(self, target: float) -> None:
        if target <= self._mono:
            return
        self._mono = target
        while self._sleepers and self._sleepers[0][0] <= self._mono:
            _, _, fut = heapq.heappop(self._sleepers)
            if not fut.done():
                fut.set_result(None)

    def _drop_sleeper(self, seq: int) -> None:
        """Remove a still-pending sleeper (its task was cancelled before
        the clock reached its deadline)."""
        for i, (_deadline, s, _fut) in enumerate(self._sleepers):
            if s == seq:
                del self._sleepers[i]
                heapq.heapify(self._sleepers)
                return

    def __repr__(self) -> str:
        return f"VirtualClock(now={self.now():.3f}, sleepers={len(self._sleepers)})"


def make_clock(virtual: bool = False, **kwargs: Any) -> Clock:
    """Factory used by tests and the app: ``virtual=True`` for determinism."""
    return VirtualClock(**kwargs) if virtual else RealClock()