"""Scheduler: lazy-invalidated wake heap, one lease per chat, re-arm on
release, re-evaluate while leased, timed vs event-only decisions, and
cancellation-safe stop — all under VirtualClock, no busy polling.

Async tests run via asyncio.run() so the test extra stays at just pytest.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from pretender.clock import VirtualClock
from pretender.scheduler import LedgerScheduler, Scheduler
from pretender.session import backoff_seconds
from pretender.types import (
    ChatKey,
    ClaimBusy,
    CommitSeq,
    CycleClaim,
    CycleId,
    Decision,
    DispatchCause,
    DispatchDeferred,
    DispatchGrant,
    DispatchId,
    DispatchRequest,
    MessageRowId,
    Reason,
)
from tests.durable_helpers import FakeRepo

CK_A = ChatKey("qq:group:111")
CK_B = ChatKey("qq:group:222")
EPOCH = 1_700_000_000.0


def run(coro):
    return asyncio.run(coro)


async def drain_until(predicate, *, yields: int = 20_000) -> None:
    """Yield to the scheduler task until ``predicate`` holds. Deterministic:
    pure event-loop yielding, never real-time polling."""
    for _ in range(yields):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("drain_until: condition not reached")


class FakeCycle:
    """Records ``(chat_key, clock.now())`` per call; returns scripted
    decisions (then ``default``). When ``hold`` is set, the executor waits
    on it — so a test can observe the chat mid-lease."""

    def __init__(
        self,
        clock: VirtualClock,
        decisions: list[Decision] | None = None,
        default: Decision | None = None,
        hold: asyncio.Event | None = None,
    ) -> None:
        self.clock = clock
        self.calls: list[tuple[ChatKey, float]] = []
        self.decisions = list(decisions or [])
        self.default = default if default is not None else Decision(action="delay")
        self.hold = hold

    async def __call__(self, chat_key: ChatKey) -> Decision:
        self.calls.append((chat_key, self.clock.now()))
        if self.hold is not None:
            await self.hold.wait()
        if self.decisions:
            return self.decisions.pop(0)
        return self.default


# ── basic wake → immediate evaluation ───────────────────────────────────────

def test_wake_schedules_immediate_evaluation():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    cycle = FakeCycle(clock)
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.wake(CK_A)
        await drain_until(lambda: len(cycle.calls) == 1)
        assert cycle.calls == [(CK_A, EPOCH)]
        assert sched.next_wake(CK_A) is None  # event-only default: no re-arm
        await sched.stop()

    run(scenario())


# ── release re-arms with a timed wake ───────────────────────────────────────

def test_release_rearms_timed_delay():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    cycle = FakeCycle(clock, decisions=[Decision(action="delay", delay_seconds=300.0)])
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.wake(CK_A)
        await drain_until(lambda: len(cycle.calls) == 1)
        assert cycle.calls == [(CK_A, EPOCH)]
        # release re-armed the chat at t+300
        assert sched.next_wake(CK_A) == EPOCH + 300.0
        clock.advance(300.0)
        await drain_until(lambda: len(cycle.calls) == 2)
        assert cycle.calls[1] == (CK_A, EPOCH + 300.0)
        await sched.stop()

    run(scenario())


# ── t+300 delay is not overridden by an immediate wake ──────────────────────

def test_t300_delay_not_overridden_by_immediate_wake():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    cycle = FakeCycle(clock, decisions=[Decision(action="delay", delay_seconds=300.0)])
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.wake(CK_A)
        await drain_until(lambda: len(cycle.calls) == 1)
        assert sched.next_wake(CK_A) == EPOCH + 300.0
        # a new message at t+10 must NOT yank the bot out of the delay
        await sched.wake(CK_A)
        clock.advance(10.0)
        await asyncio.sleep(0)
        assert len(cycle.calls) == 1  # still not evaluated
        assert sched.next_wake(CK_A) == EPOCH + 300.0  # delay stands
        clock.advance(290.0)
        await drain_until(lambda: len(cycle.calls) == 2)
        assert cycle.calls[1] == (CK_A, EPOCH + 300.0)
        await sched.stop()

    run(scenario())


# ── stale heap entries are discarded (lazy invalidation) ────────────────────

def test_stale_heap_entries_discarded():
    """A wake pushed while a delay is scheduled is a STALE heap entry: it
    must be discarded on pop, never evaluated (PLAN.md §4 lazy
    invalidation)."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    cycle = FakeCycle(clock, decisions=[Decision(action="delay", delay_seconds=300.0)])
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.wake(CK_A)
        await drain_until(lambda: len(cycle.calls) == 1)
        assert sched.next_wake(CK_A) == EPOCH + 300.0
        # an immediate wake during the delay pushes a stale entry
        await sched.wake(CK_A)
        assert sched.pending_wakes() == 2  # stale (now) + live (t+300)
        clock.advance(300.0)
        await drain_until(lambda: len(cycle.calls) == 2)
        # exactly one evaluation at t+300 — the stale entry was discarded
        assert cycle.calls[1] == (CK_A, EPOCH + 300.0)
        await asyncio.sleep(0)
        assert len(cycle.calls) == 2
        await sched.stop()

    run(scenario())


# ── wake during a lease sets a re-evaluate flag ─────────────────────────────

def test_wake_during_lease_sets_re_evaluate():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    hold = asyncio.Event()
    cycle = FakeCycle(
        clock, decisions=[Decision(action="delay", delay_seconds=300.0)], hold=hold
    )
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.wake(CK_A)
        await drain_until(lambda: sched.is_leased(CK_A))
        await sched.wake(CK_A)  # during the lease
        hold.set()
        # release re-arms: the re-evaluate flag wins over the delay decision
        await drain_until(lambda: len(cycle.calls) == 2)
        assert cycle.calls[1] == (CK_A, EPOCH)  # immediate re-run, not t+300
        await sched.stop()

    run(scenario())


# ── event-only decisions schedule no timed wake ─────────────────────────────

def test_event_only_decision_no_timed_wake():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    cycle = FakeCycle(clock, default=Decision(action="delay", delay_seconds=None))
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.wake(CK_A)
        await drain_until(lambda: len(cycle.calls) == 1)
        assert sched.next_wake(CK_A) is None  # event-only: no timed wake
        # the next event wakes the chat immediately
        await sched.wake(CK_A)
        await drain_until(lambda: len(cycle.calls) == 2)
        assert cycle.calls[1] == (CK_A, EPOCH)
        await sched.stop()

    run(scenario())


def test_non_positive_delay_treated_as_event_only():
    # Defensive: a 0 delay would busy-loop the scheduler.
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    cycle = FakeCycle(clock, decisions=[Decision(action="delay", delay_seconds=0.0)])
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.wake(CK_A)
        await drain_until(lambda: len(cycle.calls) == 1)
        assert sched.next_wake(CK_A) is None
        await sched.stop()

    run(scenario())


# ── concurrent chats ────────────────────────────────────────────────────────

def test_concurrent_chats_independent_wakes_and_delays():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    cycle = FakeCycle(
        clock,
        decisions=[
            Decision(action="delay", delay_seconds=100.0),
            Decision(action="delay", delay_seconds=50.0),
        ],
    )
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.wake(CK_A)
        await sched.wake(CK_B)
        await drain_until(lambda: len(cycle.calls) == 2)
        assert cycle.calls == [(CK_A, EPOCH), (CK_B, EPOCH)]
        assert sched.next_wake(CK_A) == EPOCH + 100.0
        assert sched.next_wake(CK_B) == EPOCH + 50.0
        clock.advance(50.0)
        await drain_until(lambda: len(cycle.calls) == 3)
        assert cycle.calls[2] == (CK_B, EPOCH + 50.0)
        assert len(cycle.calls) == 3  # A's delay still stands
        clock.advance(50.0)
        await drain_until(lambda: len(cycle.calls) == 4)
        assert cycle.calls[3] == (CK_A, EPOCH + 100.0)
        await sched.stop()

    run(scenario())


# ── wake during another chat's sleep is processed promptly ──────────────────

def test_wake_for_other_chat_during_sleep_is_prompt():
    """A wake for an event-only chat arriving while the loop sleeps toward
    another chat's delay must be evaluated at the wake, not at the sleeping
    chat's deadline."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    cycle = FakeCycle(clock, decisions=[Decision(action="delay", delay_seconds=300.0)])
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.wake(CK_A)
        await drain_until(lambda: len(cycle.calls) == 1)
        assert sched.next_wake(CK_A) == EPOCH + 300.0
        # B has no scheduled wake: its wake is LIVE and must not wait for A
        await sched.wake(CK_B)
        await drain_until(lambda: len(cycle.calls) == 2)
        assert cycle.calls[1] == (CK_B, EPOCH)  # evaluated immediately
        assert sched.next_wake(CK_A) == EPOCH + 300.0  # A's delay stands
        clock.advance(300.0)
        await drain_until(lambda: len(cycle.calls) == 3)
        assert cycle.calls[2] == (CK_A, EPOCH + 300.0)
        await sched.stop()

    run(scenario())


# ── compressed 6-hour scenario ──────────────────────────────────────────────

def test_six_hour_backoff_scenario_runs_in_milliseconds():
    """Idle backoff growth over 6 hours of virtual time: 15, 30, 60, 120,
    240, then capped at 300 s — the whole run in well under a second of
    wall time (PLAN.md §8)."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=True)
    calls: list[tuple[ChatKey, float, float]] = []  # (chat, ts, delay)

    async def backoff_cycle(chat_key: ChatKey) -> Decision:
        streak = len(calls) + 2  # first call is the 2nd consecutive idle cycle
        delay = backoff_seconds(streak)
        calls.append((chat_key, clock.now(), delay))
        return Decision(action="delay", delay_seconds=delay)

    sched = Scheduler(clock, backoff_cycle)
    wall_start = time.monotonic()

    async def scenario():
        sched.start()
        await sched.wake(CK_A)
        # 6 hours = 21600 s; delays sum 15+30+60+120+240 = 465, then 300 each:
        # call 77 lands at 465 + 71*300 = 21765 > 21600.
        await drain_until(lambda: len(calls) == 77)
        await sched.stop()

    run(scenario())
    assert time.monotonic() - wall_start < 1.0
    assert clock.now() - EPOCH == pytest.approx(21765.0)
    # backoff growth: 15, 30, 60, 120, 240, then capped at 300
    assert [d for _c, _t, d in calls[:7]] == [15.0, 30.0, 60.0, 120.0, 240.0, 300.0, 300.0]
    assert all(d == 300.0 for _c, _t, d in calls[6:])
    # every evaluation happened at its scheduled wake time
    expected_ts = EPOCH
    for i, (_c, ts, delay) in enumerate(calls):
        assert ts == pytest.approx(expected_ts)
        expected_ts += delay


# ── cancellation-safe stop ──────────────────────────────────────────────────

def test_clean_shutdown_when_idle():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    cycle = FakeCycle(clock)
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.stop()  # no exception, no hang

    run(scenario())
    assert sched.pending_wakes() == 0
    assert not sched.is_leased(CK_A)


def test_clean_shutdown_while_sleeping():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    cycle = FakeCycle(clock, decisions=[Decision(action="delay", delay_seconds=300.0)])
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.wake(CK_A)
        await drain_until(lambda: len(cycle.calls) == 1)
        assert sched.next_wake(CK_A) == EPOCH + 300.0
        await sched.stop()  # cancels the pending sleep

    run(scenario())
    # the cancelled sleeper must not resurrect anything
    clock.advance(1000.0)
    assert len(cycle.calls) == 1
    assert sched.pending_wakes() == 0


def test_clean_shutdown_while_leased():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    hold = asyncio.Event()
    cycle = FakeCycle(clock, hold=hold)
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.wake(CK_A)
        await drain_until(lambda: sched.is_leased(CK_A))
        await sched.stop()  # cancels the mid-cycle executor

    run(scenario())
    assert not sched.is_leased(CK_A)  # the loop's finally released the lease
    assert cycle.calls == [(CK_A, EPOCH)]


def test_wake_after_stop_is_noop():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    cycle = FakeCycle(clock)
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.stop()
        await sched.wake(CK_A)
        await asyncio.sleep(0)
        assert cycle.calls == []

    run(scenario())


def test_stop_is_idempotent():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    cycle = FakeCycle(clock)
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.stop()
        await sched.stop()  # second stop is a no-op

    run(scenario())

# ── priority wake (direct @/quote) vs ordinary wake ─────────────────────────

def test_priority_wake_overrides_scheduled_delay():
    """A priority wake (structurally recognized direct @/quote) OVERRIDES
    a scheduled delay: the chat re-evaluates immediately, while ordinary
    input never overrides a scheduled delay."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    cycle = FakeCycle(clock, decisions=[Decision(action="delay", delay_seconds=300.0)])
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.wake(CK_A)
        await drain_until(lambda: len(cycle.calls) == 1)
        assert sched.next_wake(CK_A) == EPOCH + 300.0
        # Ordinary input during the delay: held (the delay stands).
        await sched.wake(CK_A)
        await asyncio.sleep(0)
        assert len(cycle.calls) == 1
        assert sched.next_wake(CK_A) == EPOCH + 300.0
        # Priority input during the delay: the delay is overridden.
        await sched.wake_priority(CK_A)
        await drain_until(lambda: len(cycle.calls) == 2)
        assert cycle.calls[1] == (CK_A, EPOCH)  # evaluated immediately
        assert sched.next_wake(CK_A) is None  # event-only default: no re-arm
        await sched.stop()

    run(scenario())


def test_priority_wake_without_scheduled_delay_evaluates_immediately():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    cycle = FakeCycle(clock)
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.wake_priority(CK_A)
        await drain_until(lambda: len(cycle.calls) == 1)
        assert cycle.calls == [(CK_A, EPOCH)]
        await sched.stop()

    run(scenario())


def test_priority_wake_during_lease_sets_re_evaluate():
    """During a lease the priority wake sets the re-evaluate flag, exactly
    like an ordinary wake: the release re-arms immediately."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    hold = asyncio.Event()
    cycle = FakeCycle(
        clock, decisions=[Decision(action="delay", delay_seconds=300.0)], hold=hold
    )
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.wake(CK_A)
        await drain_until(lambda: sched.is_leased(CK_A))
        await sched.wake_priority(CK_A)  # during the lease
        hold.set()
        await drain_until(lambda: len(cycle.calls) == 2)
        assert cycle.calls[1] == (CK_A, EPOCH)  # immediate re-run, not t+300
        await sched.stop()

    run(scenario())


def test_priority_wake_after_stop_is_noop():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    cycle = FakeCycle(clock)
    sched = Scheduler(clock, cycle)

    async def scenario():
        sched.start()
        await sched.stop()
        await sched.wake_priority(CK_A)
        await asyncio.sleep(0)
        assert cycle.calls == []

    run(scenario())


# ── busy-horizon re-arm: recovery without new input ─────────────────────────

def test_scheduler_rearms_at_busy_horizon_and_recovers_without_input():
    """A cycle executor returning the busy-horizon delay (ClaimBusy ->
    timed delay) re-arms the scheduler at the exact horizon; the wake
    fires WITHOUT new input and the cycle runs again (the recovered-grant
    path)."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    calls: list[float] = []

    async def busy_cycle(chat_key: ChatKey) -> Decision:
        calls.append(clock.now())
        if len(calls) == 1:
            return Decision(action="delay", delay_seconds=200.0)  # busy horizon
        return Decision(action="skip", reason=Reason.SKIP)

    sched = Scheduler(clock, busy_cycle)

    async def scenario():
        sched.start()
        await sched.wake(CK_A)
        await drain_until(lambda: len(calls) == 1)
        assert sched.next_wake(CK_A) == EPOCH + 200.0  # re-armed at the horizon
        clock.advance(200.0)
        await drain_until(lambda: len(calls) == 2)  # fires without new input
        assert calls[1] == EPOCH + 200.0
        await sched.stop()

    run(scenario())


# ── LedgerScheduler: durable dispatch-ledger lane ────────────────────────────
# The frozen Oracle advisory lane: every wake becomes a typed
# DispatchRequest and Repository.begin_dispatch is the only ordering
# authority. Protocol fakes + VirtualClock, no busy polling.

def make_grant(
    chat_key=CK_A,
    dispatch_id=1,
    cycle_id="cy-1",
    started_ts=EPOCH,
    expires_at=EPOCH + 60.0,
    attached=(),
    pending=(),
    commit_boundary=0,
    scheduled_for=None,
) -> DispatchGrant:
    return DispatchGrant(
        dispatch_id=DispatchId(dispatch_id),
        claim=CycleClaim(chat_key, CycleId(cycle_id), started_ts, expires_at),
        start_msg_id=MessageRowId(0),
        through_msg_id=MessageRowId(0),
        attached=tuple(attached),
        pending=tuple(pending),
        commit_boundary=CommitSeq(commit_boundary),
        scheduled_for=scheduled_for,
    )


def make_busy(chat_key=CK_A, cycle_id="cy-1", busy_until=EPOCH + 200.0) -> ClaimBusy:
    return ClaimBusy(chat_key, CycleId(cycle_id), busy_until)


class FakeLedgerRepo(FakeRepo):
    """A protocol-complete Repository fake (inherits every seam method from
    ``FakeRepo``) with SCRIPTED ``begin_dispatch`` results: consumed in
    order, then ``default``. The LedgerScheduler only ever calls
    ``begin_dispatch`` — recovery chat keys come from the caller and
    settlement is the handler's job."""

    def __init__(self, results=None, default=None) -> None:
        super().__init__()
        self.requests: list[DispatchRequest] = []
        self.results = list(results or [])
        self.default = default

    async def begin_dispatch(self, request: DispatchRequest):
        self.requests.append(request)
        if self.results:
            return self.results.pop(0)
        return self.default


class FakeDispatchHandler:
    """Records every granted dispatch; returns scripted decisions (then
    ``default``). When ``hold`` is set, the handler waits on it — so a test
    can observe the chat mid-lease."""

    def __init__(self, decisions=None, default=None, hold=None) -> None:
        self.grants: list[DispatchGrant] = []
        self.decisions = list(decisions or [])
        self.default = default if default is not None else Decision(action="delay")
        self.hold = hold

    async def __call__(self, grant: DispatchGrant) -> Decision:
        self.grants.append(grant)
        if self.hold is not None:
            await self.hold.wait()
        if self.decisions:
            return self.decisions.pop(0)
        return self.default


def test_ledger_inbound_notify_begin_dispatch_handler():
    """notify_commit → a typed inbound DispatchRequest → begin_dispatch →
    the injected handler runs with the grant."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    grant = make_grant()
    repo = FakeLedgerRepo(results=[grant])
    handler = FakeDispatchHandler()
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(7))
        await drain_until(lambda: len(handler.grants) == 1)
        assert handler.grants == [grant]
        assert len(repo.requests) == 1
        req = repo.requests[0]
        assert req.chat_key == CK_A
        assert req.cause == DispatchCause.INBOUND
        assert req.wake_kind == DispatchCause.INBOUND
        assert req.scheduled_ts is None
        assert req.started_ts == EPOCH
        assert req.expires_at == EPOCH + 60.0  # finite lease, > started_ts
        assert req.now == EPOCH
        await sched.stop()

    run(scenario())


def test_ledger_writer_order_delegated_to_begin_dispatch():
    """The scheduler never orders or batches commits locally: every
    notify_commit becomes its own begin_dispatch call, and the commit_seq
    is recorded for diagnostics only — attachment is the repository's
    durable writer order."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(
        results=[make_grant(dispatch_id=1), make_grant(dispatch_id=2)]
    )
    handler = FakeDispatchHandler()
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(3))
        await drain_until(lambda: len(repo.requests) == 1)
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(4))
        await drain_until(lambda: len(repo.requests) == 2)
        assert [r.cause for r in repo.requests] == [
            DispatchCause.INBOUND,
            DispatchCause.INBOUND,
        ]
        assert sched.last_commit_seq(CK_A) == CommitSeq(4)
        assert len(handler.grants) == 2
        await sched.stop()

    run(scenario())


def test_ledger_timer_rearm():
    """A timed Decision schedules a typed timer request: the chat re-wakes
    at the exact deadline with cause=TIMER and scheduled_ts set."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(
        results=[make_grant(dispatch_id=1), make_grant(dispatch_id=2)]
    )
    handler = FakeDispatchHandler(
        decisions=[Decision(action="delay", delay_seconds=300.0)]
    )
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(1))
        await drain_until(lambda: len(handler.grants) == 1)
        assert sched.next_wake(CK_A) == EPOCH + 300.0
        clock.advance(300.0)
        await drain_until(lambda: len(repo.requests) == 2)
        req = repo.requests[1]
        assert req.cause == DispatchCause.TIMER
        assert req.wake_kind == DispatchCause.TIMER
        assert req.scheduled_ts == EPOCH + 300.0  # the exact deadline
        assert len(handler.grants) == 2
        await sched.stop()

    run(scenario())


def test_ledger_claim_busy_schedules_precise_horizon_retry():
    """ClaimBusy → a busy_recovery wake at the exact busy_until; when it
    fires the chat re-attempts begin_dispatch and gets a grant."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(
        results=[make_busy(busy_until=EPOCH + 200.0), make_grant(dispatch_id=2)]
    )
    handler = FakeDispatchHandler()
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(1))
        await drain_until(lambda: len(repo.requests) == 1)
        assert len(handler.grants) == 0  # busy: no handler run
        assert sched.next_wake(CK_A) == EPOCH + 200.0  # precise horizon
        clock.advance(200.0)
        await drain_until(lambda: len(repo.requests) == 2)
        req = repo.requests[1]
        assert req.cause == DispatchCause.BUSY_RECOVERY
        assert req.wake_kind == DispatchCause.BUSY_RECOVERY
        assert len(handler.grants) == 1  # recovered grant ran the handler
        await sched.stop()

    run(scenario())


def test_ledger_busy_recovery_repeats_while_still_busy():
    """A busy_recovery that still finds the owner live re-arms at the NEW
    busy horizon — precise retry, never a busy-loop."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(
        results=[
            make_busy(busy_until=EPOCH + 100.0),
            make_busy(busy_until=EPOCH + 250.0),
            make_grant(dispatch_id=3),
        ]
    )
    handler = FakeDispatchHandler()
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(1))
        await drain_until(lambda: len(repo.requests) == 1)
        assert sched.next_wake(CK_A) == EPOCH + 100.0
        clock.advance(100.0)
        await drain_until(lambda: len(repo.requests) == 2)
        assert sched.next_wake(CK_A) == EPOCH + 250.0  # re-armed at new horizon
        clock.advance(150.0)
        await drain_until(lambda: len(repo.requests) == 3)
        assert len(handler.grants) == 1
        await sched.stop()

    run(scenario())


def test_ledger_none_is_safe_noop():
    """begin_dispatch returning None (unknown chat / no eligible commits)
    is a safe no-op: no handler run, no re-arm."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(results=[None])
    handler = FakeDispatchHandler()
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(1))
        await drain_until(lambda: len(repo.requests) == 1)
        await asyncio.sleep(0)
        assert handler.grants == []
        assert sched.next_wake(CK_A) is None
        await sched.stop()

    run(scenario())


def test_ledger_startup_recovery_from_caller_chat_keys():
    """recover() takes durable pending chat keys from the caller and
    schedules a startup dispatch for each."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(
        results=[make_grant(dispatch_id=1), make_grant(dispatch_id=2)]
    )
    handler = FakeDispatchHandler()
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.recover([CK_A, CK_B])
        await drain_until(lambda: len(handler.grants) == 2)
        assert [r.cause for r in repo.requests] == [
            DispatchCause.STARTUP,
            DispatchCause.STARTUP,
        ]
        assert [r.chat_key for r in repo.requests] == [CK_A, CK_B]
        await sched.stop()

    run(scenario())


def test_ledger_notify_startup_single():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(results=[make_grant()])
    handler = FakeDispatchHandler()
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_startup(CK_A)
        await drain_until(lambda: len(handler.grants) == 1)
        assert repo.requests[0].cause == DispatchCause.STARTUP
        await sched.stop()

    run(scenario())


def test_ledger_notify_busy_recovery_explicit():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(results=[make_grant()])
    handler = FakeDispatchHandler()
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_busy_recovery(CK_A)
        await drain_until(lambda: len(handler.grants) == 1)
        assert repo.requests[0].cause == DispatchCause.BUSY_RECOVERY
        await sched.stop()

    run(scenario())


def test_ledger_event_only_decision_no_timed_wake():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(
        results=[make_grant(dispatch_id=1), make_grant(dispatch_id=2)]
    )
    handler = FakeDispatchHandler(default=Decision(action="delay", delay_seconds=None))
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(1))
        await drain_until(lambda: len(handler.grants) == 1)
        assert sched.next_wake(CK_A) is None  # event-only: no timed wake
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(2))
        await drain_until(lambda: len(handler.grants) == 2)
        await sched.stop()

    run(scenario())


def test_ledger_non_positive_delay_treated_as_event_only():
    # Defensive: a 0 delay would busy-loop the scheduler.
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(results=[make_grant()])
    handler = FakeDispatchHandler(decisions=[Decision(action="delay", delay_seconds=0.0)])
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(1))
        await drain_until(lambda: len(handler.grants) == 1)
        assert sched.next_wake(CK_A) is None
        await sched.stop()

    run(scenario())


def test_ledger_commit_during_lease_sets_re_evaluate():
    """A commit arriving while the chat is leased sets the re-evaluate flag:
    the release re-wakes so the new commit is attached by a fresh
    begin_dispatch — data survives, never coalesced locally."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    hold = asyncio.Event()
    repo = FakeLedgerRepo(
        results=[make_grant(dispatch_id=1), make_grant(dispatch_id=2)]
    )
    handler = FakeDispatchHandler(hold=hold)
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(1))
        await drain_until(lambda: sched.is_leased(CK_A))
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(2))  # during the lease
        hold.set()
        await drain_until(lambda: len(repo.requests) == 2)
        assert repo.requests[1].cause == DispatchCause.INBOUND
        assert len(handler.grants) == 2
        await sched.stop()

    run(scenario())


def test_ledger_ordinary_commit_stays_behind_scheduled_timer():
    """An ordinary durable commit stays behind a scheduled delay; only a
    direct/quote/high-pending priority commit may supersede it."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(
        results=[make_grant(dispatch_id=1), make_grant(dispatch_id=2)]
    )
    handler = FakeDispatchHandler(
        decisions=[Decision(action="delay", delay_seconds=300.0)]
    )
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(1))
        await drain_until(lambda: len(handler.grants) == 1)
        assert sched.next_wake(CK_A) == EPOCH + 300.0  # timer scheduled
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(2))  # new commit
        await asyncio.sleep(0)
        assert len(repo.requests) == 1
        clock.advance(300.0)
        await drain_until(lambda: len(repo.requests) == 2)
        assert repo.requests[1].cause == DispatchCause.TIMER
        assert len(handler.grants) == 2
        await sched.stop()

    run(scenario())


def test_ledger_priority_commit_overrides_scheduled_timer():
    """Only the durable priority bit may supersede a delayed timer."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(
        results=[make_grant(dispatch_id=1), make_grant(dispatch_id=2)]
    )
    handler = FakeDispatchHandler(
        decisions=[Decision(action="delay", delay_seconds=300.0)]
    )
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(1))
        await drain_until(lambda: len(handler.grants) == 1)
        assert sched.next_wake(CK_A) == EPOCH + 300.0
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(2), priority=True)
        await drain_until(lambda: len(repo.requests) == 2)
        assert repo.requests[1].cause == DispatchCause.INBOUND
        await sched.stop()

    run(scenario())


def test_ledger_uses_per_chat_dispatch_lease():
    """The initial durable request must use the effective chat lease, not
    the top-level scheduler default."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(results=[make_grant(dispatch_id=1)])
    handler = FakeDispatchHandler()
    sched = LedgerScheduler(
        repo,
        clock,
        handler,
        dispatch_lease_s=60.0,
        lease_for_chat=lambda chat_key: 17.0 if chat_key == CK_A else 31.0,
    )

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, CommitSeq(1))
        await drain_until(lambda: len(repo.requests) == 1)
        request = repo.requests[0]
        await sched.stop()
        return request

    request = run(scenario())
    assert request.expires_at - request.started_ts == 17.0


def test_ledger_notification_racing_begin_is_consumed_by_grant():
    """A notification that arrives while the writer is freezing a dispatch
    must disappear when that grant already attached its CommitSeq."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    entered = asyncio.Event()
    release = asyncio.Event()

    class _SlowRepo(FakeLedgerRepo):
        async def begin_dispatch(self, request: DispatchRequest):
            self.requests.append(request)
            entered.set()
            await release.wait()
            return make_grant(dispatch_id=1, attached=(CommitSeq(1),))

    repo = _SlowRepo()
    handler = FakeDispatchHandler(default=Decision(action="delay"))
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, CommitSeq(1))
        await entered.wait()
        # The writer has not returned its attached boundary yet.
        await sched.notify_commit(CK_A, CommitSeq(1))
        release.set()
        await drain_until(lambda: len(handler.grants) == 1)
        for _ in range(10):
            await asyncio.sleep(0)
        assert len(repo.requests) == 1
        await sched.stop()

    run(scenario())


def test_ledger_multiple_chats_independent():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(
        results=[
            make_grant(dispatch_id=1, chat_key=CK_A),
            make_grant(dispatch_id=2, chat_key=CK_B),
            make_grant(dispatch_id=3, chat_key=CK_B),
            make_grant(dispatch_id=4, chat_key=CK_A),
        ]
    )
    handler = FakeDispatchHandler(
        decisions=[
            Decision(action="delay", delay_seconds=100.0),
            Decision(action="delay", delay_seconds=50.0),
        ]
    )
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(1))
        await sched.notify_commit(CK_B, commit_seq=CommitSeq(1))
        await drain_until(lambda: len(handler.grants) == 2)
        assert sched.next_wake(CK_A) == EPOCH + 100.0
        assert sched.next_wake(CK_B) == EPOCH + 50.0
        clock.advance(50.0)
        await drain_until(lambda: len(handler.grants) == 3)
        assert handler.grants[2].claim.chat_key == CK_B
        assert len(handler.grants) == 3  # A's timer still stands
        clock.advance(50.0)
        await drain_until(lambda: len(handler.grants) == 4)
        assert handler.grants[3].claim.chat_key == CK_A
        await sched.stop()

    run(scenario())


def test_ledger_clean_shutdown_while_leased():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    hold = asyncio.Event()
    repo = FakeLedgerRepo(results=[make_grant()])
    handler = FakeDispatchHandler(hold=hold)
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(1))
        await drain_until(lambda: sched.is_leased(CK_A))
        await sched.stop()  # cancels the mid-dispatch handler

    run(scenario())
    assert not sched.is_leased(CK_A)  # the loop's finally released the lease
    assert len(handler.grants) == 1


def test_ledger_clean_shutdown_while_sleeping():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(results=[make_grant()])
    handler = FakeDispatchHandler(
        decisions=[Decision(action="delay", delay_seconds=300.0)]
    )
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(1))
        await drain_until(lambda: len(handler.grants) == 1)
        assert sched.next_wake(CK_A) == EPOCH + 300.0
        await sched.stop()  # cancels the pending sleep

    run(scenario())
    # the cancelled sleeper must not resurrect anything
    clock.advance(1000.0)
    assert len(handler.grants) == 1
    assert sched.pending_wakes() == 0


def test_ledger_stop_idempotent_and_notify_after_stop_is_noop():
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(results=[make_grant()])
    handler = FakeDispatchHandler()
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.stop()
        await sched.stop()  # second stop is a no-op
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(1))
        await sched.notify_startup(CK_A)
        await sched.notify_busy_recovery(CK_A)
        await sched.recover([CK_A])
        await asyncio.sleep(0)
        assert handler.grants == []
        assert repo.requests == []

    run(scenario())


# ── durable agent barrier: defer re-arm and handler-exception recovery ──────

def make_deferred(
    chat_key=CK_A, resume_at=EPOCH + 200.0, defer_kind="retry"
) -> DispatchDeferred:
    return DispatchDeferred(chat_key, resume_at, defer_kind)


def test_ledger_defer_rearms_at_resume_and_blocks_early_priority():
    """A DispatchDeferred re-arms exactly at resume_at; even priority input
    cannot invoke the agent early while the barrier is active."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(
        results=[make_deferred(resume_at=EPOCH + 200.0), make_grant(dispatch_id=2)]
    )
    handler = FakeDispatchHandler()
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(1))
        await drain_until(lambda: len(repo.requests) == 1)
        assert len(handler.grants) == 0  # deferred: no agent run
        assert sched.next_wake(CK_A) == EPOCH + 200.0  # re-armed at resume_at
        # A priority commit during the defer must NOT invoke the agent early.
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(2), priority=True)
        await asyncio.sleep(0)
        assert len(repo.requests) == 1  # no early begin_dispatch
        assert len(handler.grants) == 0
        # At resume_at the barrier expires and the agent runs.
        clock.advance(200.0)
        await drain_until(lambda: len(handler.grants) == 1)
        assert repo.requests[1].cause == DispatchCause.TIMER
        await sched.stop()

    run(scenario())


def test_ledger_handler_exception_schedules_busy_recovery_without_input():
    """A handler exception schedules busy recovery no later than the grant
    expiry, so the prepared dispatch is recovered without new input."""
    clock = VirtualClock(epoch=EPOCH, auto_advance=False)
    repo = FakeLedgerRepo(
        results=[
            make_grant(dispatch_id=1, expires_at=EPOCH + 60.0),
            make_grant(dispatch_id=2),
        ]
    )

    class _RaisingHandler:
        def __init__(self):
            self.grants: list[DispatchGrant] = []
            self.calls = 0

        async def __call__(self, grant: DispatchGrant) -> Decision:
            self.grants.append(grant)
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return Decision(action="delay")

    handler = _RaisingHandler()
    sched = LedgerScheduler(repo, clock, handler)

    async def scenario():
        sched.start()
        await sched.notify_commit(CK_A, commit_seq=CommitSeq(1))
        await drain_until(lambda: len(handler.grants) == 1)
        # The handler raised; the scheduler re-armed busy recovery at the
        # grant expiry without new input.
        assert sched.next_wake(CK_A) == EPOCH + 60.0
        clock.advance(60.0)
        await drain_until(lambda: len(handler.grants) == 2)
        assert repo.requests[1].cause == DispatchCause.BUSY_RECOVERY
        await sched.stop()

    run(scenario())
