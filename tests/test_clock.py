"""Clock: RealClock epoch/monotonic semantics; VirtualClock determinism.

Async tests run via asyncio.run() so the test extra stays at just pytest.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from pretender.clock import RealClock, VirtualClock


def run(coro):
    return asyncio.run(coro)


# ── RealClock ───────────────────────────────────────────────────────────────

def test_real_clock_epoch_matches_wall_clock():
    clock = RealClock()
    assert abs(clock.now() - time.time()) < 1.0


def test_real_clock_monotonic_matches_time_monotonic():
    clock = RealClock()
    assert abs(clock.monotonic() - time.monotonic()) < 1.0


def test_real_clock_monotonic_non_decreasing():
    clock = RealClock()
    samples = [clock.monotonic() for _ in range(5)]
    assert samples == sorted(samples)


def test_real_clock_epoch_at_roundtrip():
    clock = RealClock()
    mono = clock.monotonic()
    assert abs(clock.epoch_at(mono) - clock.now()) < 0.01
    assert abs(clock.monotonic_at(clock.epoch_at(mono)) - mono) < 0.01


# ── VirtualClock: sleep jumps time ──────────────────────────────────────────

def test_virtual_sleep_advances_time_instantly():
    clock = VirtualClock(epoch=1_700_000_000.0)
    start = clock.now()
    wall_start = time.monotonic()

    async def scenario():
        await clock.sleep(5.0)

    run(scenario())
    assert clock.now() - start == pytest.approx(5.0)
    assert clock.monotonic() == pytest.approx(5.0)
    # no real waiting: the whole 5s sleep must complete in well under a second
    assert time.monotonic() - wall_start < 1.0


def test_virtual_sleep_zero_yields_without_advancing():
    clock = VirtualClock()
    start = clock.now()

    async def scenario():
        await clock.sleep(0)

    run(scenario())
    assert clock.now() == start


def test_virtual_six_hour_scenario_runs_in_milliseconds():
    clock = VirtualClock()
    wall_start = time.monotonic()

    async def scenario():
        await clock.sleep(3600 * 6)

    run(scenario())
    assert clock.now() - clock.epoch_at(0) == pytest.approx(21600.0)
    assert time.monotonic() - wall_start < 1.0


# ── VirtualClock: advance wakes sleepers ────────────────────────────────────

def test_advance_wakes_sleepers_at_deadline():
    clock = VirtualClock(epoch=1_700_000_000.0, auto_advance=False)
    woke_at: list[float] = []

    async def sleeper(seconds: float) -> None:
        await clock.sleep(seconds)
        woke_at.append(clock.now())

    async def scenario():
        task = asyncio.create_task(sleeper(10.0))
        await asyncio.sleep(0)  # let the sleeper register
        clock.advance(5.0)
        assert woke_at == []  # not due yet
        clock.advance(5.0)
        await task

    run(scenario())
    assert woke_at == [1_700_000_010.0]


def test_advance_wakes_sleepers_in_deadline_order():
    clock = VirtualClock(auto_advance=False)
    order: list[str] = []

    async def sleeper(tag: str, seconds: float) -> None:
        await clock.sleep(seconds)
        order.append(tag)

    async def scenario():
        tasks = [
            asyncio.create_task(sleeper("late", 30.0)),
            asyncio.create_task(sleeper("early", 10.0)),
            asyncio.create_task(sleeper("middle", 20.0)),
        ]
        await asyncio.sleep(0)
        clock.advance(30.0)
        await asyncio.gather(*tasks)

    run(scenario())
    assert order == ["early", "middle", "late"]


def test_advance_without_sleepers_just_moves_time():
    clock = VirtualClock()
    clock.advance(42.0)
    assert clock.monotonic() == 42.0
    assert clock.now() == clock.epoch_at(42.0)


def test_negative_advance_rejected():
    clock = VirtualClock()
    with pytest.raises(ValueError):
        clock.advance(-1.0)


# ── VirtualClock: epoch/monotonic consistency ───────────────────────────────

def test_virtual_epoch_at_roundtrip():
    clock = VirtualClock(epoch=1234.5)
    assert clock.epoch_at(10.0) == 1244.5
    assert clock.monotonic_at(1244.5) == 10.0


def test_virtual_sleep_does_not_require_real_time():
    """The whole point: a scheduler test can burn through hours of virtual
    time without any wall-clock waiting or polling."""
    clock = VirtualClock()
    wall_start = time.monotonic()

    async def scenario():
        for _ in range(100):
            await clock.sleep(60.0)

    run(scenario())
    assert clock.monotonic() == pytest.approx(6000.0)
    assert time.monotonic() - wall_start < 1.0


# ── VirtualClock: concurrent sleeps advance to scheduled deadlines ──────────

def test_concurrent_sleeps_advance_to_deadlines_not_summed_delays():
    """Two sleeps scheduled at the same instant must complete at their own
    deadlines (t+10 and t+20), never at t+30 (summed delays)."""
    clock = VirtualClock(epoch=1_700_000_000.0)
    woke_at: list[float] = []

    async def sleeper(seconds: float) -> None:
        await clock.sleep(seconds)
        woke_at.append(clock.now())

    async def scenario():
        await asyncio.gather(
            asyncio.create_task(sleeper(10.0)),
            asyncio.create_task(sleeper(20.0)),
        )

    run(scenario())
    assert sorted(woke_at) == [1_700_000_010.0, 1_700_000_020.0]
    assert clock.monotonic() == pytest.approx(20.0)


def test_concurrent_sleeps_do_not_overshoot_shared_deadline():
    clock = VirtualClock()
    woke_at: list[float] = []

    async def sleeper(seconds: float) -> None:
        await clock.sleep(seconds)
        woke_at.append(clock.now())

    async def scenario():
        await asyncio.gather(
            asyncio.create_task(sleeper(30.0)),
            asyncio.create_task(sleeper(30.0)),
        )

    run(scenario())
    assert woke_at == [clock.epoch_at(30.0), clock.epoch_at(30.0)]
    assert clock.monotonic() == pytest.approx(30.0)


def test_reversed_order_concurrent_sleeps_earliest_deadline_first():
    """A long sleep created FIRST must not advance the clock past a shorter
    one created second: the clock reaches deadlines in order, independent
    of task creation order."""
    clock = VirtualClock(epoch=1_700_000_000.0)
    woke_at: list[float] = []

    async def sleeper(seconds: float) -> None:
        await clock.sleep(seconds)
        woke_at.append(clock.now())

    async def scenario():
        await asyncio.gather(
            asyncio.create_task(sleeper(30.0)),  # long first
            asyncio.create_task(sleeper(10.0)),  # short second
        )

    run(scenario())
    assert sorted(woke_at) == [1_700_000_010.0, 1_700_000_030.0]
    assert clock.monotonic() == pytest.approx(30.0)


def test_reversed_order_three_way_deadlines():
    clock = VirtualClock(epoch=1_700_000_000.0)
    woke_at: list[float] = []

    async def sleeper(seconds: float) -> None:
        await clock.sleep(seconds)
        woke_at.append(clock.now())

    async def scenario():
        await asyncio.gather(
            asyncio.create_task(sleeper(50.0)),
            asyncio.create_task(sleeper(10.0)),
            asyncio.create_task(sleeper(30.0)),
        )

    run(scenario())
    assert sorted(woke_at) == [
        1_700_000_010.0, 1_700_000_030.0, 1_700_000_050.0,
    ]
    assert clock.monotonic() == pytest.approx(50.0)


def test_manual_advance_mode_still_wakes_by_deadline():
    """auto_advance=False keeps the heap semantics: advance() wakes every
    sleeper whose deadline has passed, in deadline order."""
    clock = VirtualClock(epoch=1_700_000_000.0, auto_advance=False)
    woke_at: list[float] = []

    async def sleeper(seconds: float) -> None:
        await clock.sleep(seconds)
        woke_at.append(clock.now())

    async def scenario():
        tasks = [
            asyncio.create_task(sleeper(30.0)),
            asyncio.create_task(sleeper(10.0)),
        ]
        await asyncio.sleep(0)
        clock.advance(10.0)  # wakes the 10s sleeper at its deadline
        await asyncio.sleep(0)
        clock.advance(20.0)  # clock reaches 30: wakes the 30s sleeper
        await asyncio.gather(*tasks)

    run(scenario())
    assert sorted(woke_at) == [1_700_000_010.0, 1_700_000_030.0]