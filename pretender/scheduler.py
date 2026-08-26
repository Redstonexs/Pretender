"""The async wake scheduler: one heap, lazy invalidation, one lease per chat.

PLAN.md §4: ``heapq`` has no decrease-key, so the heap of ``(wake_ts,
chat_key)`` entries is paired with a ``next_wake`` map holding the CURRENT
scheduled wake per chat (a generation counter makes re-arms unambiguous).
A popped entry whose generation no longer matches the map is stale and
discarded. Without this, a ``delay`` for t+300 is silently overridden by
the next immediate wake.

Semantics:

  - ``wake(chat_key)`` is the ingest commit-then-wake call. It pushes an
    immediate heap entry; the ``next_wake`` map decides whether that entry
    is LIVE (the chat had no scheduled wake — evaluate now) or STALE (a
    gate delay is scheduled — the entry is discarded on pop and the delay
    stands: new messages are seen at the scheduled evaluation, never
    yanking the bot out of a deliberate pause).
  - ``wake_priority(chat_key)`` is the narrowly-scoped priority wake for
    structurally recognized direct @/quote input: it OVERRIDES a scheduled
    delay (the chat re-evaluates now, and the gate applies the exact
    precedence — direct @/quote, focus, and high pending bypass an active
    hold), while ordinary input never overrides a scheduled delay.
  - One lease per chat: while the injected cycle executor runs for a chat,
    no second cycle may start for it. The durable claim (``claim_cycle``)
    is the cycle's job; this lease is the scheduler's in-memory
    single-flight guard. A wake during a lease sets a re-evaluate flag.
  - Release always re-arms: the re-evaluate flag wins (a wake arrived while
    leased → immediate re-run), else a timed wake from
    ``Decision.delay_seconds``, else event-only (no timed wake — the chat
    wakes only on the next ``wake()``). A non-positive delay is treated as
    event-only (defensive: a 0 delay would busy-loop).
  - The loop sleeps on the injected Clock — never ``time.time``, never busy
    polling. A sleep races a change event, so a wake for ANOTHER chat is
    processed promptly instead of waiting out the sleeping chat's delay.
    VirtualClock makes a 6-hour scenario run in milliseconds.
  - ``stop()`` is cancellation-safe: it wakes the loop, cancels the task,
    and suppresses the cancellation; the loop's ``finally`` releases any
    lease held at cancellation, and the scheduler state is cleared.

``LedgerScheduler`` (the durable dispatch-ledger lane, frozen Oracle
advisory) drives the SAME heap machinery through
``Repository.begin_dispatch`` instead of a bare cycle call. It owns NO
ordering of its own: every wake — inbound commit, timer deadline, startup,
busy-horizon retry — becomes a typed ``DispatchRequest``, and
``begin_dispatch`` is the only authority for what attaches to a dispatch
(durable writer order). The scheduler never batches commits locally and
never claims or settles outside the ledger methods; settlement is the
injected handler's job.
"""

from __future__ import annotations

import asyncio
from collections import deque
import heapq
import logging
import uuid
from typing import Awaitable, Callable, Iterable

from pretender.seams import Clock, Repository
from pretender.types import (
    ChatKey,
    ClaimBusy,
    CommitSeq,
    CycleId,
    Decision,
    DispatchCause,
    DispatchDeferred,
    DispatchGrant,
    DispatchRequest,
)

log = logging.getLogger("pretender.scheduler")

# The injected async cycle executor: one gate→reply saga per call, returning
# the gate's Decision. The scheduler only reads ``delay_seconds`` (timed vs
# event-only); action/reason are the cycle's business.
CycleFn = Callable[[ChatKey], Awaitable[Decision]]


class Scheduler:
    """The per-chat wake scheduler. Start it with ``start()``, feed it with
    ``wake()``, stop it with ``stop()``."""

    def __init__(self, clock: Clock, cycle: CycleFn) -> None:
        self._clock = clock
        self._cycle = cycle
        # (wake_ts, chat_key, seq): seq is the per-chat generation counter
        # that makes lazy invalidation unambiguous (two entries for the same
        # chat at the same ts can otherwise collide).
        self._heap: list[tuple[float, ChatKey, int]] = []
        # chat_key -> (ts, seq): the CURRENT scheduled wake per chat. An
        # entry whose (ts, seq) does not match is stale and discarded.
        self._next_wake: dict[ChatKey, tuple[float, int]] = {}
        self._seq: dict[ChatKey, int] = {}
        self._leases: set[ChatKey] = set()
        self._re_evaluate: set[ChatKey] = set()
        self._stopped = False
        self._changed = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> asyncio.Task[None]:
        """Create the loop task (idempotent)."""
        if self._task is not None:
            return self._task
        self._task = asyncio.create_task(self._run(), name="pretender-scheduler")
        return self._task

    async def stop(self) -> None:
        """Cancellation-safe shutdown: wake the loop, cancel the task,
        suppress the cancellation, clear the scheduler state. Safe to call
        when idle, sleeping, or mid-cycle (the lease is released by the
        loop's ``finally``)."""
        self._stopped = True
        self._changed.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._heap.clear()
        self._next_wake.clear()
        self._seq.clear()
        self._leases.clear()
        self._re_evaluate.clear()

    # ── public API ──────────────────────────────────────────────────────────

    async def wake(self, chat_key: ChatKey) -> None:
        """Commit-then-wake entry point (ingest calls this after the durable
        commit). Pushes an immediate entry; the ``next_wake`` map marks it
        stale when a delay is scheduled, so the delay is never overridden.
        A wake during a lease sets the re-evaluate flag."""
        if self._stopped:
            return
        if chat_key in self._leases:
            self._re_evaluate.add(chat_key)
            return
        self._push(chat_key, self._clock.now(), live=chat_key not in self._next_wake)

    async def wake_priority(self, chat_key: ChatKey) -> None:
        """Priority wake: a structurally recognized direct @/quote may
        re-evaluate DURING a scheduled delay — the delay is overridden —
        while ordinary input never overrides a scheduled delay (``wake``).
        During a lease it sets the re-evaluate flag, exactly like
        ``wake``. The gate applies the exact precedence at evaluation."""
        if self._stopped:
            return
        if chat_key in self._leases:
            self._re_evaluate.add(chat_key)
            return
        self._push(chat_key, self._clock.now(), live=True)

    def is_leased(self, chat_key: ChatKey) -> bool:
        return chat_key in self._leases

    def next_wake(self, chat_key: ChatKey) -> float | None:
        """The currently scheduled wake time for a chat, or None when the
        chat is event-only (no timed wake scheduled)."""
        entry = self._next_wake.get(chat_key)
        return entry[0] if entry is not None else None

    def pending_wakes(self) -> int:
        """Heap size including stale entries (diagnostics/tests)."""
        return len(self._heap)

    # ── internals ───────────────────────────────────────────────────────────

    def _push(self, chat_key: ChatKey, ts: float, *, live: bool) -> None:
        seq = self._seq.get(chat_key, 0) + 1
        self._seq[chat_key] = seq
        if live:
            self._next_wake[chat_key] = (ts, seq)
        heapq.heappush(self._heap, (ts, chat_key, seq))
        self._changed.set()

    async def _sleep_until(self, ts: float) -> None:
        """Sleep on the injected clock until the absolute deadline ``ts``.
        The remaining delay is computed at execution time, so the deadline
        is exact no matter when the task first runs."""
        delay = ts - self._clock.now()
        if delay > 0:
            await self._clock.sleep(delay)

    async def _run(self) -> None:
        while not self._stopped:
            if not self._heap:
                # Nothing scheduled: wait for a wake() — never busy-poll.
                await self._changed.wait()
                self._changed.clear()
                continue
            ts, chat_key, seq = self._heap[0]
            now = self._clock.now()
            if ts > now:
                # Sleep until the deadline, but wake early when a new wake
                # arrives (it may be for another chat, or a stale entry to
                # discard). The clock sleep and the change event race; the
                # loser is cancelled. The sleep computes its remaining delay
                # at EXECUTION time from the absolute deadline, so a clock
                # advance between task creation and execution can never
                # shift the registered deadline.
                sleep_task = asyncio.create_task(self._sleep_until(ts))
                changed_task = asyncio.create_task(self._changed.wait())
                try:
                    done, pending = await asyncio.wait(
                        {sleep_task, changed_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                except asyncio.CancelledError:
                    for t in (sleep_task, changed_task):
                        t.cancel()
                    raise
                for t in pending:
                    t.cancel()
                for t in pending:
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
                if changed_task in done:
                    self._changed.clear()
                    continue  # a wake arrived: re-read the heap
                # deadline reached: fall through and process the heap top
            heapq.heappop(self._heap)
            if self._next_wake.get(chat_key) != (ts, seq):
                continue  # stale entry: a newer wake superseded it
            del self._next_wake[chat_key]
            if chat_key in self._leases:
                self._re_evaluate.add(chat_key)
                continue
            self._leases.add(chat_key)
            try:
                decision = await self._cycle(chat_key)
            except Exception:
                log.exception("cycle executor failed for %s", chat_key)
                decision = None
            finally:
                self._leases.discard(chat_key)
            self._rearm(chat_key, decision)

    def _rearm(self, chat_key: ChatKey, decision: Decision | None) -> None:
        """Release always re-arms: the re-evaluate flag wins (a wake arrived
        while leased), else a timed wake from the decision, else event-only
        (no timed wake — the chat wakes only on the next ``wake()``)."""
        if chat_key in self._re_evaluate:
            self._re_evaluate.discard(chat_key)
            self._push(chat_key, self._clock.now(), live=True)
            return
        if decision is not None and decision.delay_seconds is not None:
            if decision.delay_seconds > 0:
                self._push(chat_key, self._clock.now() + decision.delay_seconds, live=True)
                return
            # Non-positive delay: defensive event-only (a 0 delay would
            # busy-loop the scheduler).
        # Event-only: no timed wake; the chat stays wakeable by events.


# The injected async dispatch handler: one gate→reply saga per granted
# dispatch, returning the Decision the scheduler re-arms on. The handler
# owns ALL settlement (``settle_dispatch``); the scheduler never settles.
DispatchFn = Callable[[DispatchGrant], Awaitable[Decision]]


class LedgerScheduler:
    """The durable dispatch-ledger scheduler (frozen Oracle advisory).

    Owns dispatch ORDERING for the ledger lane: every wake becomes a typed
    ``DispatchRequest`` and ``Repository.begin_dispatch`` is the only
    authority for what attaches to a dispatch — the scheduler never infers
    batching from event-loop order or timestamps, and never coalesces
    commits locally. Ordinary/priority data survives through the
    request/ledger: a commit that arrives after a dispatch's frozen
    boundary stays unassigned and is attached by the next wake.

    Feed it with ``notify_commit`` (after a durable inbound commit),
    ``notify_startup`` / ``recover`` (startup recovery from caller-provided
    pending chat keys), and ``notify_busy_recovery`` (explicit busy retry).
    On a wake it calls ``begin_dispatch``:

      - ``DispatchGrant`` → the injected handler runs (it settles).
      - ``ClaimBusy`` → a precise busy-horizon retry is scheduled at the
        active owner's exact ``busy_until`` (a ``busy_recovery`` wake).
      - ``None`` → safe no-op (unknown chat, or an inbound wake with no
        eligible commits).

    Timed Decisions schedule typed ``timer`` requests (``scheduled_ts`` is
    the exact deadline); event-only decisions schedule no timed wake. One
    in-memory lease per chat guards single-flight; a wake during a lease
    sets a re-evaluate flag so the release re-wakes with the recorded
    cause. Start/stop are cancellation-safe and never busy-poll.
    """

    def __init__(
        self,
        repo: Repository,
        clock: Clock,
        dispatch: DispatchFn,
        *,
        dispatch_lease_s: float = 60.0,
        lease_for_chat: Callable[[ChatKey], float] | None = None,
        uuid_fn: Callable[[], str] | None = None,
    ) -> None:
        self._repo = repo
        self._clock = clock
        self._handler = dispatch
        self._dispatch_lease_s = dispatch_lease_s
        self._lease_for_chat = lease_for_chat
        self._uuid = uuid_fn if uuid_fn is not None else lambda: str(uuid.uuid4())
        # (wake_ts, chat_key, seq, cause): seq is the per-chat generation
        # counter that makes lazy invalidation unambiguous; cause is the
        # DispatchCause this wake must begin with.
        self._heap: list[tuple[float, ChatKey, int, str]] = []
        # chat_key -> (ts, seq): the CURRENT scheduled wake per chat. An
        # entry whose (ts, seq) does not match is stale and discarded.
        self._next_wake: dict[ChatKey, tuple[float, int]] = {}
        self._seq: dict[ChatKey, int] = {}
        self._leases: set[ChatKey] = set()
        # chat_key -> cause: a wake arrived while leased; the release
        # re-wakes with the recorded cause.
        self._re_evaluate: dict[ChatKey, str] = {}
        # Commit notifications can arrive while begin_dispatch is awaiting
        # the single writer. Keep them by sequence until the returned grant
        # proves which side of its frozen boundary they landed on. Values are
        # the durable priority bit; only genuinely post-boundary priority
        # commits may supersede a later timer.
        self._after_lease: dict[ChatKey, dict[CommitSeq, bool]] = {}
        # chat_key -> last durable commit sequence notified (diagnostics
        # only — ordering is delegated to begin_dispatch, never inferred).
        self._last_commit_seq: dict[ChatKey, CommitSeq] = {}
        # Commit notifications are delivered after their DB transaction and
        # can race a timer's begin_dispatch. Once a grant has attached a
        # sequence in this process, a late notification for that same commit
        # must not create a duplicate release/re-evaluation.
        self._handled_commits: set[CommitSeq] = set()
        self._handled_order: deque[CommitSeq] = deque()
        self._handled_limit = 4096
        # chat_key -> resume_at: the durable agent barrier is active; no
        # wake (priority included) may invoke the agent before resume_at.
        self._deferred_until: dict[ChatKey, float] = {}
        self._stopped = False
        self._changed = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> asyncio.Task[None]:
        """Create the loop task (idempotent)."""
        if self._task is not None:
            return self._task
        self._task = asyncio.create_task(
            self._run(), name="pretender-ledger-scheduler"
        )
        return self._task

    async def stop(self) -> None:
        """Cancellation-safe shutdown: wake the loop, cancel the task,
        suppress the cancellation, clear the scheduler state. Safe to call
        when idle, sleeping, or mid-dispatch (the lease is released by the
        loop's ``finally``)."""
        self._stopped = True
        self._changed.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._heap.clear()
        self._next_wake.clear()
        self._seq.clear()
        self._leases.clear()
        self._re_evaluate.clear()
        self._after_lease.clear()
        self._last_commit_seq.clear()
        self._handled_commits.clear()
        self._handled_order.clear()
        self._deferred_until.clear()

    # ── public API ──────────────────────────────────────────────────────────

    async def notify_commit(
        self, chat_key: ChatKey, commit_seq: CommitSeq, *, priority: bool = False
    ) -> None:
        """Submit one durable inbound commit to the ledger dispatcher.

        ``commit_seq`` is diagnostics-only: durable writer order in
        ``begin_dispatch`` decides attachment. The durable ``priority`` bit,
        however, controls local timer precedence: ordinary commits remain
        behind an already scheduled delay, while a direct/quote/high-pending
        commit may supersede it. A non-priority commit during a live dispatch
        is attached by writer order or left for the next timer; it does not
        spuriously force an immediate re-evaluation.
        """
        if self._stopped:
            return
        if commit_seq in self._handled_commits:
            return
        self._last_commit_seq[chat_key] = commit_seq
        if self._defer_guard(chat_key):
            return
        if chat_key in self._leases:
            self._after_lease.setdefault(chat_key, {})[commit_seq] = priority
            return
        if priority or chat_key not in self._next_wake:
            self._push(chat_key, self._clock.now(), DispatchCause.INBOUND)

    async def notify_startup(self, chat_key: ChatKey) -> None:
        """Explicit startup notification: a ``startup`` dispatch (a
        priority wake — always creates a dispatch even with zero attached
        commits)."""
        if self._stopped:
            return
        if self._defer_guard(chat_key):
            return
        if chat_key in self._leases:
            self._re_evaluate[chat_key] = DispatchCause.STARTUP
            return
        self._push(chat_key, self._clock.now(), DispatchCause.STARTUP)

    async def notify_busy_recovery(self, chat_key: ChatKey) -> None:
        """Explicit busy-recovery notification: a ``busy_recovery`` dispatch
        (a priority wake). The scheduler also schedules these internally
        when ``begin_dispatch`` reports ``ClaimBusy``."""
        if self._stopped:
            return
        if self._defer_guard(chat_key):
            return
        if chat_key in self._leases:
            self._re_evaluate[chat_key] = DispatchCause.BUSY_RECOVERY
            return
        self._push(chat_key, self._clock.now(), DispatchCause.BUSY_RECOVERY)

    async def recover(self, chat_keys: Iterable[ChatKey]) -> None:
        """Startup recovery: schedule a ``startup`` dispatch for every
        durable pending chat key the CALLER provides (e.g. from
        ``Repository.list_ledger_pending_chats``). No App implementation
        lives here — the caller owns the recovery read."""
        for chat_key in chat_keys:
            await self.notify_startup(chat_key)

    def is_leased(self, chat_key: ChatKey) -> bool:
        return chat_key in self._leases

    def next_wake(self, chat_key: ChatKey) -> float | None:
        """The currently scheduled wake time for a chat, or None when the
        chat is event-only (no timed wake scheduled)."""
        entry = self._next_wake.get(chat_key)
        return entry[0] if entry is not None else None

    def pending_wakes(self) -> int:
        """Heap size including stale entries (diagnostics/tests)."""
        return len(self._heap)

    def last_commit_seq(self, chat_key: ChatKey) -> CommitSeq | None:
        """The last durable commit sequence notified for a chat
        (diagnostics only — never used for ordering)."""
        return self._last_commit_seq.get(chat_key)

    # ── internals ───────────────────────────────────────────────────────────

    def _push(self, chat_key: ChatKey, ts: float, cause: str) -> None:
        seq = self._seq.get(chat_key, 0) + 1
        self._seq[chat_key] = seq
        self._next_wake[chat_key] = (ts, seq)
        heapq.heappush(self._heap, (ts, chat_key, seq, cause))
        self._changed.set()

    async def _sleep_until(self, ts: float) -> None:
        """Sleep on the injected clock until the absolute deadline ``ts``.
        The remaining delay is computed at execution time, so the deadline
        is exact no matter when the task first runs."""
        delay = ts - self._clock.now()
        if delay > 0:
            await self._clock.sleep(delay)

    async def _run(self) -> None:
        while not self._stopped:
            if not self._heap:
                # Nothing scheduled: wait for a notification — never
                # busy-poll.
                await self._changed.wait()
                self._changed.clear()
                continue
            ts, chat_key, seq, cause = self._heap[0]
            now = self._clock.now()
            if ts > now:
                # Sleep until the deadline, but wake early when a new wake
                # arrives (it may be for another chat, or a stale entry to
                # discard). The clock sleep and the change event race; the
                # loser is cancelled.
                sleep_task = asyncio.create_task(self._sleep_until(ts))
                changed_task = asyncio.create_task(self._changed.wait())
                try:
                    done, pending = await asyncio.wait(
                        {sleep_task, changed_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                except asyncio.CancelledError:
                    for t in (sleep_task, changed_task):
                        t.cancel()
                    raise
                for t in pending:
                    t.cancel()
                for t in pending:
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
                if changed_task in done:
                    self._changed.clear()
                    continue  # a wake arrived: re-read the heap
                # deadline reached: fall through and process the heap top
            heapq.heappop(self._heap)
            if self._next_wake.get(chat_key) != (ts, seq):
                continue  # stale entry: a newer wake superseded it
            del self._next_wake[chat_key]
            if chat_key in self._leases:
                self._re_evaluate[chat_key] = cause
                continue
            self._leases.add(chat_key)
            try:
                decision = await self._begin(chat_key, cause, ts)
            except Exception:
                log.exception("dispatch failed for %s", chat_key)
                decision = None
            finally:
                self._leases.discard(chat_key)
            self._rearm(chat_key, decision)

    async def _begin(
        self, chat_key: ChatKey, cause: str, ts: float
    ) -> Decision | None:
        """One typed ``begin_dispatch`` attempt: build the DispatchRequest,
        call the repository, and route the outcome. Returns the handler's
        Decision on a grant, None otherwise (no-op / busy re-armed /
        deferred)."""
        self._deferred_until.pop(chat_key, None)
        now = self._clock.now()
        lease_s = (
            self._lease_for_chat(chat_key)
            if self._lease_for_chat is not None
            else self._dispatch_lease_s
        )
        if not isinstance(lease_s, (int, float)) or lease_s <= 0:
            raise ValueError(f"invalid dispatch lease for {chat_key!r}: {lease_s!r}")
        request = DispatchRequest(
            chat_key=chat_key,
            cause=cause,
            cycle_id=CycleId(self._uuid()),
            started_ts=now,
            expires_at=now + float(lease_s),
            now=now,
            wake_kind=cause,
            scheduled_ts=ts if cause == DispatchCause.TIMER else None,
        )
        result = await self._repo.begin_dispatch(request)
        if result is None:
            return None  # unknown chat, or an inbound wake with no work
        if isinstance(result, ClaimBusy):
            # A live, unexpired prepared dispatch owns the chat: re-arm at
            # the exact busy horizon so the next wake finds the lease
            # expired, recovers it, and gets a grant — without new input.
            self._push(
                chat_key, max(result.busy_until, now), DispatchCause.BUSY_RECOVERY
            )
            return None
        if isinstance(result, DispatchDeferred):
            # The durable agent barrier is active: re-arm exactly at
            # resume_at. No agent runs; priority input cannot invoke it
            # early (the notify guards keep the resume wake).
            self._deferred_until[chat_key] = result.resume_at
            self._push(chat_key, result.resume_at, DispatchCause.TIMER)
            return None
        self._remember_attached(result.attached)
        after = self._after_lease.get(chat_key)
        if after is not None:
            for sequence in result.attached:
                after.pop(sequence, None)
            if not after:
                self._after_lease.pop(chat_key, None)
        try:
            return await self._handler(result)
        except Exception:
            log.exception("dispatch handler failed for %s", chat_key)
            # The prepared dispatch is still live: schedule busy recovery
            # no later than the grant expiry so it is recovered/settled
            # without new input.
            self._push(
                chat_key,
                max(result.claim.expires_at, now),
                DispatchCause.BUSY_RECOVERY,
            )
            return None

    def _defer_guard(self, chat_key: ChatKey) -> bool:
        """While the chat's durable agent barrier is active (a defer), no
        wake — priority included — may invoke the agent early: keep the
        resume wake scheduled and return True so the caller does not push
        an early wake. Once the barrier has expired, clear the guard and
        let the wake through (begin_dispatch clears the durable barrier and
        grants normally)."""
        deferred = self._deferred_until.get(chat_key)
        if deferred is None:
            return False
        if self._clock.now() < deferred:
            if self._next_wake.get(chat_key) is None:
                self._push(chat_key, deferred, DispatchCause.TIMER)
            return True
        self._deferred_until.pop(chat_key, None)
        return False

    def _remember_attached(self, attached: Iterable[CommitSeq]) -> None:
        """Bounded stale-notification suppression for this process."""
        for sequence in attached:
            if sequence in self._handled_commits:
                continue
            if len(self._handled_order) >= self._handled_limit:
                oldest = self._handled_order.popleft()
                self._handled_commits.discard(oldest)
            self._handled_order.append(sequence)
            self._handled_commits.add(sequence)

    def _rearm(self, chat_key: ChatKey, decision: Decision | None) -> None:
        """Release always re-arms: the re-evaluate flag wins (a wake arrived
        while leased → immediate re-run with its cause), else a timed wake
        from ``Decision.delay_seconds`` (a typed ``timer`` request), else
        event-only (no timed wake — the chat wakes only on the next
        notification). A non-positive delay is treated as event-only
        (defensive: a 0 delay would busy-loop)."""
        if chat_key in self._re_evaluate:
            cause = self._re_evaluate.pop(chat_key)
            self._after_lease.pop(chat_key, None)
            self._push(chat_key, self._clock.now(), cause)
            return
        after = self._after_lease.pop(chat_key, {})
        if any(after.values()):
            self._push(chat_key, self._clock.now(), DispatchCause.INBOUND)
            return
        if decision is not None and decision.delay_seconds is not None:
            if decision.delay_seconds > 0:
                deadline = self._clock.now() + decision.delay_seconds
                self._push(chat_key, deadline, DispatchCause.TIMER)
                return
            # Non-positive delay: defensive event-only (a 0 delay would
            # busy-loop the scheduler).
        if after:
            self._push(chat_key, self._clock.now(), DispatchCause.INBOUND)
            return
        # Event-only: no timed wake; the chat stays wakeable by events.
