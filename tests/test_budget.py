"""BudgetManager tests: day rollover, cap boundary, rung order, malformed KV,
concurrent-style updates, cost/tokens, capability flags, stable serialization.

The manager depends only on the Repository KV seam (get_kv/set_kv) and an
injected time source, so a protocol-fake repo plus a mutable clock closure
drive every scenario deterministically. Tests run via asyncio.run() so the
test extra stays at just pytest.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from pretender.budget import (
    ALLOWED,
    BLOCKED,
    DEGRADE,
    BudgetManager,
    BudgetUsage,
    LearnerBudget,
)
from pretender.config import BudgetConfig, BudgetRung
from pretender.types import ChatKey
from tests.durable_helpers import open_repo_with_chat

CK = ChatKey("qq:group:123456")
EPOCH = 1_700_000_000.0  # 2023-11-14 UTC


def run(coro):
    return asyncio.run(coro)


def day_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()


class FakeKVRepo:
    """A protocol-fake Repository exposing only the KV seam the manager uses."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.get_calls: list[str] = []
        self.set_calls: list[tuple[str, str]] = []

    async def get_kv(self, k: str) -> str | None:
        self.get_calls.append(k)
        return self.store.get(k)

    async def set_kv(self, k: str, v: str) -> None:
        self.set_calls.append((k, v))
        self.store[k] = v


def make_manager(repo, config=None, epoch: float = EPOCH):
    """Build a manager with a controllable clock closure."""
    t = {"now": float(epoch)}

    def now() -> float:
        return t["now"]

    mgr = BudgetManager(repo, config or BudgetConfig(), now=now)
    return mgr, t


# ── day rollover ─────────────────────────────────────────────────────────────

def test_day_rollover():
    async def scenario():
        repo = FakeKVRepo()
        mgr, t = make_manager(repo)
        day1 = day_of(t["now"])
        await mgr.record(CK, calls=5, tokens=100, cost=1.5)
        snap = await mgr.snapshot(CK)
        assert snap.day == day1 and snap.calls == 5

        # Advance exactly one UTC day: usage rolls over to a fresh day.
        t["now"] += 86400
        day2 = day_of(t["now"])
        assert day2 != day1
        snap2 = await mgr.snapshot(CK)
        assert snap2.day == day2 and snap2.calls == 0 and snap2.tokens == 0

        # The previous day's key is preserved, not overwritten.
        assert repo.store[f"budget:{CK}:{day1}"] is not None
        assert f"budget:{CK}:{day2}" not in repo.store
        return day1, day2

    day1, day2 = run(scenario())
    assert day1 != day2


# ── cap boundary ─────────────────────────────────────────────────────────────

def test_cap_boundary():
    async def scenario():
        config = BudgetConfig(daily_cap=10, rungs=())
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo, config)
        for _ in range(9):
            await mgr.record(CK)
        d = await mgr.decide(CK)
        assert d.kind == ALLOWED and d.remaining == 1

        await mgr.record(CK)  # the 10th call reaches the cap
        d = await mgr.decide(CK)
        assert d.kind == BLOCKED and d.remaining == 0

        # Overflow is clamped: calls never exceed the cap.
        await mgr.record(CK, calls=5)
        snap = await mgr.snapshot(CK)
        assert snap.calls == 10
        return d

    d = run(scenario())
    assert d.kind == BLOCKED


def test_cap_boundary_stop_rung_reported():
    async def scenario():
        config = BudgetConfig(daily_cap=10, rungs=(BudgetRung(at=1.0, action="stop"),))
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo, config)
        for _ in range(10):
            await mgr.record(CK)
        d = await mgr.decide(CK)
        assert d.kind == BLOCKED
        assert d.rung is not None and d.rung.action == "stop"
        return d

    d = run(scenario())
    assert d.rung is not None and d.rung.action == "stop"


# ── rung order (degrade ladder) ──────────────────────────────────────────────

def test_rung_order():
    async def scenario():
        config = BudgetConfig(
            daily_cap=100,
            rungs=(
                BudgetRung(at=0.5, action="degrade", detail="reduce context"),
                BudgetRung(at=0.7, action="degrade", detail="cheaper profile"),
                BudgetRung(at=0.9, action="degrade", detail="drop capabilities"),
            ),
        )
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo, config)

        for _ in range(55):  # fraction 0.55 -> first degrade rung
            await mgr.record(CK)
        d = await mgr.decide(CK)
        assert d.kind == DEGRADE
        assert d.degrade == ("context_reduction",)

        for _ in range(20):  # fraction 0.75 -> first two
            await mgr.record(CK)
        d = await mgr.decide(CK)
        assert d.degrade == ("context_reduction", "profile_fallback")

        for _ in range(20):  # fraction 0.95 -> all three, stable order
            await mgr.record(CK)
        d = await mgr.decide(CK)
        assert d.degrade == (
            "context_reduction",
            "profile_fallback",
            "capability_flags",
        )
        return d

    d = run(scenario())
    assert d.degrade[-1] == "capability_flags"


def test_default_config_degrade_at_0_9():
    async def scenario():
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo)  # default cap=100, degrade rung at 0.9
        for _ in range(90):
            await mgr.record(CK)
        d = await mgr.decide(CK)
        assert d.kind == DEGRADE
        assert d.degrade == ("context_reduction",)
        assert d.rung is not None and d.rung.action == "degrade"
        return d

    d = run(scenario())
    assert d.kind == DEGRADE


def test_warn_rung_is_allowed():
    async def scenario():
        config = BudgetConfig(
            daily_cap=100,
            rungs=(BudgetRung(at=0.8, action="warn", detail="heads up"),),
        )
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo, config)
        for _ in range(80):
            await mgr.record(CK)
        d = await mgr.decide(CK)
        assert d.kind == ALLOWED
        assert d.rung is not None and d.rung.action == "warn"
        return d

    d = run(scenario())
    assert d.kind == ALLOWED


# ── capability flags ─────────────────────────────────────────────────────────

def test_capability_flags_deepest_and_clamped():
    async def scenario():
        # Four degrade rungs: the fourth is beyond the 3-step ladder and
        # clamps to the deepest action (capability_flags).
        config = BudgetConfig(
            daily_cap=10,
            rungs=(
                BudgetRung(at=0.1, action="degrade"),
                BudgetRung(at=0.2, action="degrade"),
                BudgetRung(at=0.3, action="degrade"),
                BudgetRung(at=0.4, action="degrade"),
            ),
        )
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo, config)
        for _ in range(5):
            await mgr.record(CK)
        d = await mgr.decide(CK)
        assert d.kind == DEGRADE
        assert d.degrade == (
            "context_reduction",
            "profile_fallback",
            "capability_flags",
            "capability_flags",
        )
        return d

    d = run(scenario())
    assert d.degrade.count("capability_flags") == 2


# ── malformed / stale KV ─────────────────────────────────────────────────────

def test_malformed_kv_reconciled():
    async def scenario():
        repo = FakeKVRepo()
        mgr, t = make_manager(repo)
        day = day_of(t["now"])
        key = f"budget:{CK}:{day}"
        bad_values = (
            "not json",
            "[]",
            "{}",
            '{"day": "2020-01-01"}',  # missing fields
            '{"day": "2020-01-01", "calls": "a", "tokens": 0, "cost": 0}',  # bad type
            '{"day": "2020-01-01", "calls": 5, "tokens": 0, "cost": "x"}',  # bad cost
        )
        for bad in bad_values:
            repo.store[key] = bad
            snap = await mgr.snapshot(CK)
            assert snap.calls == 0 and snap.tokens == 0 and snap.cost == 0.0

        # A stale value for a different day rolls over to a fresh day.
        repo.store[key] = '{"day": "2020-01-01", "calls": 99, "tokens": 9, "cost": 9.0}'
        snap = await mgr.snapshot(CK)
        assert snap.calls == 0 and snap.tokens == 0 and snap.cost == 0.0
        return True

    assert run(scenario())


def test_malformed_kv_does_not_crash_record():
    async def scenario():
        repo = FakeKVRepo()
        mgr, t = make_manager(repo)
        repo.store[f"budget:{CK}:{day_of(t['now'])}"] = "garbage"
        usage = await mgr.record(CK, calls=3, tokens=10, cost=0.5)
        assert usage.calls == 3 and usage.tokens == 10 and usage.cost == 0.5
        return usage

    usage = run(scenario())
    assert usage.calls == 3


# ── sequential concurrent-style updates ──────────────────────────────────────

def test_concurrent_updates_no_loss():
    async def scenario():
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo)
        await asyncio.gather(*(mgr.record(CK) for _ in range(50)))
        snap = await mgr.snapshot(CK)
        assert snap.calls == 50
        return snap

    snap = run(scenario())
    assert snap.calls == 50


def test_concurrent_updates_with_tokens_and_cost():
    async def scenario():
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo)
        await asyncio.gather(
            *(mgr.record(CK, tokens=10, cost=0.1) for _ in range(20))
        )
        snap = await mgr.snapshot(CK)
        assert snap.calls == 20 and snap.tokens == 200
        assert abs(snap.cost - 2.0) < 1e-9
        return snap

    snap = run(scenario())
    assert snap.calls == 20


# ── cost / tokens ────────────────────────────────────────────────────────────

def test_cost_tokens_accumulate_and_clamp_negative():
    async def scenario():
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo)
        await mgr.record(CK, tokens=100, cost=1.5)
        await mgr.record(CK, tokens=50, cost=0.5)
        snap = await mgr.snapshot(CK)
        assert snap.tokens == 150 and abs(snap.cost - 2.0) < 1e-9

        # Negative inputs are clamped to zero: no underflow.
        await mgr.record(CK, calls=-3, tokens=-10, cost=-5.0)
        snap = await mgr.snapshot(CK)
        assert snap.calls == 2 and snap.tokens == 150 and snap.cost == 2.0
        return snap

    snap = run(scenario())
    assert snap.calls == 2


# ── stable serialization ─────────────────────────────────────────────────────

def test_stable_serialization():
    async def scenario():
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo)
        await mgr.record(CK, tokens=7, cost=1.25)
        key = next(iter(repo.store))
        v1 = repo.store[key]

        # A second manager with the same config/time writes identical bytes.
        repo2 = FakeKVRepo()
        mgr2, _ = make_manager(repo2)
        await mgr2.record(CK, tokens=7, cost=1.25)
        assert repo2.store[key] == v1

        # The blob is deterministic JSON with sorted keys.
        parsed = json.loads(v1)
        assert parsed == {
            "day": day_of(EPOCH),
            "calls": 1,
            "tokens": 7,
            "cost": 1.25,
        }
        assert list(parsed.keys()) == sorted(parsed.keys())
        return v1

    v1 = run(scenario())
    assert isinstance(v1, str)


def test_serialization_round_trips_through_parse():
    async def scenario():
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo)
        await mgr.record(CK, calls=4, tokens=300, cost=3.75)
        snap = await mgr.snapshot(CK)
        assert snap == BudgetUsage(day=day_of(EPOCH), calls=4, tokens=300, cost=3.75)
        return snap

    snap = run(scenario())
    assert snap.calls == 4


# ── immutability ─────────────────────────────────────────────────────────────

def test_decisions_and_usage_are_immutable():
    async def scenario():
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo)
        await mgr.record(CK)
        d = await mgr.decide(CK)
        return d

    d = run(scenario())
    with pytest.raises(FrozenInstanceError):
        d.kind = ALLOWED  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        d.usage.calls = 99  # type: ignore[misc]


# ── atomic reserve (decision + reservation under one lock) ───────────────────

def test_reserve_combines_decision_and_reservation():
    """reserve computes the decision from usage BEFORE the reservation and
    atomically increments the call count under the lock."""
    async def scenario():
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo)
        d = await mgr.reserve(CK, calls=1)
        snap = await mgr.snapshot(CK)
        return d, snap

    d, snap = run(scenario())
    assert d.kind == ALLOWED
    assert d.usage.calls == 0  # decision from usage before the reservation
    assert snap.calls == 1  # the reservation was recorded

def test_reserve_blocked_reserves_nothing():
    """A blocked reserve returns the blocked decision and records nothing."""
    async def scenario():
        config = BudgetConfig(daily_cap=2, rungs=())
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo, config)
        await mgr.record(CK, calls=2)  # at the cap
        d = await mgr.reserve(CK, calls=1)
        snap = await mgr.snapshot(CK)
        return d, snap

    d, snap = run(scenario())
    assert d.kind == BLOCKED
    assert snap.calls == 2  # nothing reserved

def test_reserve_planner_and_embed_never_exceed_cap():
    """Simultaneous planner + embed reservations for the same chat can never
    exceed the cap: the atomic reserve serializes them under one lock."""
    async def scenario():
        config = BudgetConfig(daily_cap=5, rungs=())
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo, config)
        results = await asyncio.gather(*(mgr.reserve(CK, calls=1) for _ in range(10)))
        snap = await mgr.snapshot(CK)
        blocked = sum(1 for r in results if r.kind == BLOCKED)
        return snap, blocked

    snap, blocked = run(scenario())
    assert snap.calls == 5  # never exceeds the cap
    assert blocked == 5  # the remaining 5 were blocked

def test_reserve_retains_reservation_on_failure():
    """A provider failure after a successful reservation retains it (the call
    count stays incremented); the caller records tokens later with calls=0."""
    async def scenario():
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo)
        d = await mgr.reserve(CK, calls=1)  # the provider call is reserved
        assert d.kind == ALLOWED
        # The provider failed: no tokens recorded, reservation retained.
        snap = await mgr.snapshot(CK)
        return snap

    snap = run(scenario())
    assert snap.calls == 1  # the reservation is retained


# ── Gate 5: durable/atomic reservation across DISTINCT manager instances ─────

def test_multi_manager_reservations_never_exceed_cap(tmp_path):
    """DISTINCT BudgetManager instances over the same SQLite DB/chat/day
    reserve atomically: simultaneous planner/embed-style reservations can
    never exceed the cap. The per-instance asyncio lock alone cannot
    serialize them — the BudgetStore writer transaction does."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        config = BudgetConfig(daily_cap=5, rungs=())
        mgr_a = BudgetManager(repo, config, now=lambda: EPOCH)
        mgr_b = BudgetManager(repo, config, now=lambda: EPOCH)
        results = await asyncio.gather(
            *(mgr_a.reserve(CK, calls=1) for _ in range(5)),
            *(mgr_b.reserve(CK, calls=1) for _ in range(5)),
        )
        snap = await mgr_a.snapshot(CK)
        await repo.close()
        blocked = sum(1 for r in results if r.kind == BLOCKED)
        return snap, blocked

    snap, blocked = run(scenario())
    assert snap.calls == 5  # never exceeds the cap across instances
    assert blocked == 5  # the remaining 5 were blocked


def test_multi_manager_record_accumulates_without_loss(tmp_path):
    """DISTINCT manager instances recording tokens/cost over the same DB
    never lose an update (atomic across instances, not just per-instance)."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        config = BudgetConfig(daily_cap=100, rungs=())
        mgr_a = BudgetManager(repo, config, now=lambda: EPOCH)
        mgr_b = BudgetManager(repo, config, now=lambda: EPOCH)
        await asyncio.gather(
            *(mgr_a.record(CK, tokens=10, cost=0.1) for _ in range(10)),
            *(mgr_b.record(CK, tokens=10, cost=0.1) for _ in range(10)),
        )
        snap = await mgr_a.snapshot(CK)
        await repo.close()
        return snap

    snap = run(scenario())
    assert snap.calls == 20
    assert snap.tokens == 200
    assert abs(snap.cost - 2.0) < 1e-9


def test_bulk_reservation_is_all_or_nothing_at_hard_cap(tmp_path):
    """A three-batch embed request at 9/10 must be rejected rather than
    clamping bookkeeping to 10 while allowing three physical calls."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        config = BudgetConfig(daily_cap=10, rungs=())
        planner = BudgetManager(repo, config, now=lambda: EPOCH)
        embed = BudgetManager(repo, config, now=lambda: EPOCH)
        await planner.record(CK, calls=9)
        decision = await embed.reserve(CK, calls=3)
        snap = await planner.snapshot(CK)
        await repo.close()
        return decision, snap

    decision, snap = run(scenario())
    assert decision.kind == BLOCKED
    assert snap.calls == 9


# ── Gate 5 remediation: semantic batch policy admission ──────────────────────
# A multi-call reservation is admitted against the EARLIEST policy rung
# crossed by any individual physical call, not only the pre-batch state and
# hard cap. A batch crossing a degrade rung is DEGRADE (semantic_only: the
# semantic runtime degrades to FTS-only without issuing paid calls, and
# nothing is reserved); a batch crossing a stop rung is BLOCKED all-or-nothing;
# a batch that exactly reaches the cap is admitted.

def test_batch_reserve_crossing_degrade_rung_degrades():
    """A multi-call reservation that crosses a degrade rung mid-batch is
    DEGRADE and semantic_only: the runtime must degrade to FTS-only without
    issuing the paid calls, and nothing is reserved."""

    async def scenario():
        config = BudgetConfig(
            daily_cap=10,
            rungs=(BudgetRung(at=0.9, action="degrade"),),
        )
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo, config)
        await mgr.record(CK, calls=8)  # fraction 0.8: below the degrade rung
        d = await mgr.reserve(CK, calls=2)  # the 2nd call would cross 0.9
        snap = await mgr.snapshot(CK)
        return d, snap

    d, snap = run(scenario())
    assert d.kind == DEGRADE
    assert d.semantic_only is True
    assert d.rung is not None and d.rung.action == "degrade"
    assert d.degrade == ("context_reduction",)
    assert snap.calls == 8  # nothing reserved: FTS-only, no paid calls


def test_batch_reserve_crossing_stop_rung_blocks():
    """A multi-call reservation that crosses a stop rung mid-batch is BLOCKED
    all-or-nothing: nothing is reserved."""

    async def scenario():
        config = BudgetConfig(
            daily_cap=10,
            rungs=(BudgetRung(at=1.0, action="stop"),),
        )
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo, config)
        await mgr.record(CK, calls=8)
        d = await mgr.reserve(CK, calls=3)  # the 3rd call would be at the cap
        snap = await mgr.snapshot(CK)
        return d, snap

    d, snap = run(scenario())
    assert d.kind == BLOCKED
    assert d.semantic_only is True
    assert d.rung is not None and d.rung.action == "stop"
    assert snap.calls == 8  # nothing reserved


def test_batch_reserve_exact_cap_admitted():
    """A multi-call reservation that exactly reaches the hard cap is ALLOWED:
    every call is admitted and the ledger lands exactly on the cap."""

    async def scenario():
        config = BudgetConfig(daily_cap=10, rungs=())
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo, config)
        await mgr.record(CK, calls=8)
        d = await mgr.reserve(CK, calls=2)  # exactly reaches 10
        snap = await mgr.snapshot(CK)
        return d, snap

    d, snap = run(scenario())
    assert d.kind == ALLOWED
    assert d.semantic_only is False
    assert snap.calls == 10  # both calls admitted, exactly at the cap


def test_batch_reserve_earliest_crossed_rung_wins():
    """A batch crossing a degrade rung BEFORE a stop rung degrades: the
    earliest policy rung crossed decides, and no paid calls are issued."""

    async def scenario():
        config = BudgetConfig(
            daily_cap=10,
            rungs=(
                BudgetRung(at=0.9, action="degrade"),
                BudgetRung(at=1.0, action="stop"),
            ),
        )
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo, config)
        await mgr.record(CK, calls=8)
        d = await mgr.reserve(CK, calls=3)  # 2nd call crosses 0.9, 3rd at cap
        snap = await mgr.snapshot(CK)
        return d, snap

    d, snap = run(scenario())
    assert d.kind == DEGRADE  # earliest crossed rung is the degrade rung
    assert d.semantic_only is True
    assert snap.calls == 8  # nothing reserved


def test_batch_reserve_within_cap_stays_allowed():
    """A multi-call reservation that stays below every rung is ALLOWED and
    reserves the full batch."""

    async def scenario():
        config = BudgetConfig(
            daily_cap=10,
            rungs=(BudgetRung(at=0.9, action="degrade"),),
        )
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo, config)
        await mgr.record(CK, calls=5)
        d = await mgr.reserve(CK, calls=3)  # reaches 8, below 0.9
        snap = await mgr.snapshot(CK)
        return d, snap

    d, snap = run(scenario())
    assert d.kind == ALLOWED
    assert d.semantic_only is False
    assert snap.calls == 8  # full batch reserved


def test_single_call_degrade_still_reserves():
    """A single-call DEGRADE keeps the per-call semantics: the call is
    reserved (the per-call client issues the degraded call), so it is NOT
    semantic_only."""

    async def scenario():
        config = BudgetConfig(
            daily_cap=10,
            rungs=(BudgetRung(at=0.9, action="degrade"),),
        )
        repo = FakeKVRepo()
        mgr, _ = make_manager(repo, config)
        await mgr.record(CK, calls=9)  # at the degrade rung
        d = await mgr.reserve(CK, calls=1)
        snap = await mgr.snapshot(CK)
        return d, snap

    d, snap = run(scenario())
    assert d.kind == DEGRADE
    assert d.semantic_only is False
    assert snap.calls == 10  # the degraded call is reserved


def test_batch_reserve_crossing_degrade_rung_atomic_store(tmp_path):
    """The batch degrade decision is atomic over the BudgetStore path too:
    DISTINCT manager instances see the same un-consumed ledger."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        config = BudgetConfig(
            daily_cap=10,
            rungs=(BudgetRung(at=0.9, action="degrade"),),
        )
        mgr = BudgetManager(repo, config, now=lambda: EPOCH)
        await mgr.record(CK, calls=8)
        d = await mgr.reserve(CK, calls=2)
        snap = await mgr.snapshot(CK)
        await repo.close()
        return d, snap

    d, snap = run(scenario())
    assert d.kind == DEGRADE
    assert d.semantic_only is True
    assert snap.calls == 8  # nothing reserved over the atomic store path


# ── LearnerBudget: the narrow background wrapper (Phase 6 P6.4) ──────────────

def test_learner_budget_bounds_concurrency_and_preserves_foreground_reserve():
    """The learner worker's budget wrapper bounds concurrent reservations to
    ``concurrency - foreground_reserve`` while sharing the SAME physical
    per-chat budget state as the foreground manager."""

    async def scenario():
        repo = FakeKVRepo()
        shared = BudgetManager(repo, BudgetConfig(daily_cap=100), now=lambda: EPOCH)
        learner = LearnerBudget(shared, concurrency=3, foreground_reserve=1)
        assert learner.slots == 2
        # Two concurrent reservations hold both background slots.
        d1 = await learner.reserve(CK, calls=1)
        d2 = await learner.reserve(CK, calls=1)
        assert d1.kind == ALLOWED and d2.kind == ALLOWED
        # The shared manager sees the same physical usage.
        snap = await shared.snapshot(CK)
        assert snap.calls == 2
        # Releasing one slot (record) lets a third reservation through.
        await learner.record(CK, calls=0, tokens=5)
        d3 = await learner.reserve(CK, calls=1)
        assert d3.kind == ALLOWED
        await learner.record(CK, calls=0, tokens=1)
        await learner.record(CK, calls=0, tokens=1)
        snap2 = await shared.snapshot(CK)
        return learner.slots, snap2

    slots, snap = run(scenario())
    assert slots == 2
    assert snap.calls == 3


def test_learner_budget_blocked_reserve_releases_slot_zero_call():
    """A blocked/degraded reservation releases its background slot
    immediately (the pipeline skips with zero provider calls)."""

    async def scenario():
        repo = FakeKVRepo()
        shared = BudgetManager(repo, BudgetConfig(daily_cap=2), now=lambda: EPOCH)
        learner = LearnerBudget(shared, concurrency=2, foreground_reserve=0)
        await shared.record(CK, calls=2)  # the cap is reached
        d = await learner.reserve(CK, calls=1)
        # The slot was released: a subsequent allowed reservation works.
        assert d.kind == BLOCKED
        return learner._held

    assert run(scenario()) == 0


def test_learner_budget_validation():
    repo = FakeKVRepo()
    with pytest.raises(ValueError):
        LearnerBudget(BudgetManager(repo, BudgetConfig(), now=lambda: EPOCH),
                      concurrency=1, foreground_reserve=1)
    with pytest.raises(ValueError):
        LearnerBudget(BudgetManager(repo, BudgetConfig(), now=lambda: EPOCH),
                      concurrency=0, foreground_reserve=0)
    with pytest.raises(ValueError):
        LearnerBudget(BudgetManager(repo, BudgetConfig(), now=lambda: EPOCH),
                      concurrency=2, foreground_reserve=-1)
