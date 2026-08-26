"""Daily LLM budget ledger (PLAN.md §4; M6).

A deterministic, async ``BudgetManager`` that stores day-scoped usage in the
Repository KV seam (``get_kv``/``set_kv``) and returns immutable
allowed/degrade/blocked decisions against a ``BudgetConfig``'s ``daily_cap``
and degrade rungs.

Frozen semantics:
- Usage is scoped per chat per UTC day. The KV key is
  ``budget:{chat_key}:{YYYY-MM-DD}`` and the value is a stable JSON blob
  (``{"day": ..., "calls": N, "tokens": N, "cost": F}``).
- A decision is computed from the usage BEFORE the incoming call. Reaching
  the hard cap (``calls >= daily_cap``) is always ``blocked``; otherwise the
  highest engaged rung (``r.at <= calls/daily_cap``) decides the kind —
  ``stop`` -> blocked, ``degrade`` -> degrade, ``warn`` -> allowed.
- A multi-call reservation (``reserve(calls=N)``) is admitted against the
  EARLIEST policy rung crossed by any individual physical call: each call is
  judged on the usage before it, and the first call whose decision is
  non-allowed decides the batch — a stop rung (or the hard cap) rejects the
  whole batch all-or-nothing, a degrade rung degrades it. A rejected or
  degraded batch is ``semantic_only``: the semantic runtime degrades to
  FTS-only without issuing paid calls, and the reservation is not consumed.
- Degrade rungs apply in stable order (ascending ``at``): each configured
  ``degrade`` rung maps to one canonical action from the fixed ladder
  (context reduction -> profile fallback -> capability flags) by its rank
  among the configured degrade rungs.
- Malformed/stale KV is reconciled: a missing key starts a fresh day; a
  value for a different day is rolled over; a malformed value is discarded
  and treated as a fresh day. Counters never underflow (clamped at 0) or
  overflow (clamped at the cap).
- No LLM calls, no dispatch/cursor/outbox changes. This module only reads
  and writes its own KV keys.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from pretender.config import AgentConfig, BudgetConfig, BudgetRung
from pretender.errors import PretenderError
from pretender.seams import BudgetStore, LLMClient, Repository
from pretender.types import ChatKey, TranscriptMessage

# ── Decision kinds ───────────────────────────────────────────────────────────

ALLOWED = "allowed"
DEGRADE = "degrade"
BLOCKED = "blocked"

# ── Canonical degrade ladder, applied in stable order ────────────────────────
# A rung with action "degrade" maps to one of these by its rank among the
# configured degrade rungs: the first degrade rung reduces context, the
# second falls back to a cheaper profile, the third disables capabilities.
# Deeper rungs (beyond the ladder) stay at the deepest action.

DEGRADE_CONTEXT_REDUCTION = "context_reduction"
DEGRADE_PROFILE_FALLBACK = "profile_fallback"
DEGRADE_CAPABILITY_FLAGS = "capability_flags"
DEGRADE_ACTIONS: tuple[str, ...] = (
    DEGRADE_CONTEXT_REDUCTION,
    DEGRADE_PROFILE_FALLBACK,
    DEGRADE_CAPABILITY_FLAGS,
)

_KV_PREFIX = "budget:"


@dataclass(frozen=True)
class BudgetUsage:
    """Immutable day-scoped usage snapshot. ``calls`` is clamped to the cap,
    ``tokens``/``cost`` are nonnegative."""

    day: str
    calls: int
    tokens: int
    cost: float


@dataclass(frozen=True)
class BudgetDecision:
    """Immutable decision for one prospective LLM call.

    ``kind`` is ``allowed`` | ``degrade`` | ``blocked``; ``usage`` is the
    usage the decision was computed from; ``remaining`` is how many calls
    remain before the hard cap (0 when blocked); ``rung`` is the highest
    engaged rung (or None); ``degrade`` lists the canonical degrade actions
    to apply, in stable order, when ``kind == "degrade"``. ``semantic_only``
    is True when the decision directs the semantic runtime to degrade to
    FTS-only without issuing paid calls: every blocked decision, plus a
    multi-call batch reservation that crosses a degrade rung.
    """

    kind: str
    usage: BudgetUsage
    remaining: int
    rung: BudgetRung | None = None
    degrade: tuple[str, ...] = ()
    semantic_only: bool = False


class BudgetManager:
    """Deterministic daily budget ledger for one chat's LLM calls.

    Holds only configuration plus the injected Repository and time source;
    every method is a pure function of its arguments and the persisted KV.
    ``record`` serializes its load-modify-save under an internal lock so
    concurrent updates never lose work against the non-atomic KV seam.

    When the repository also implements the ``BudgetStore`` seam (the real
    ``SqliteRepository`` does), ``reserve``/``record`` run their whole
    load-modify-save through the store's atomic ``budget_update`` — ONE
    writer transaction — so reservations from DISTINCT ``BudgetManager``
    instances over the same database are atomic against each other (the
    per-instance asyncio lock alone cannot serialize them). A repository
    without the store falls back to the legacy lock-protected KV path.
    """

    def __init__(
        self,
        repo: Repository,
        config: BudgetConfig,
        *,
        now: Callable[[], float],
        store: BudgetStore | None = None,
    ) -> None:
        self._repo = repo
        self._config = config
        self._now = now
        self._lock = asyncio.Lock()
        self._store = (
            store if store is not None else (repo if isinstance(repo, BudgetStore) else None)
        )

    @property
    def config(self) -> BudgetConfig:
        return self._config

    # ── public API ───────────────────────────────────────────────────────────

    async def snapshot(self, chat_key: ChatKey) -> BudgetUsage:
        """Current day-scoped usage for ``chat_key`` (no mutation)."""
        day = self._day(self._now())
        usage = self._parse(await self._repo.get_kv(self._key(chat_key, day)), day)
        return usage if usage is not None else self._fresh(day)

    async def decide(self, chat_key: ChatKey) -> BudgetDecision:
        """Immutable decision for the next call, from usage before it."""
        return self._decide(await self.snapshot(chat_key))

    async def reserve(
        self, chat_key: ChatKey, *, calls: int = 1,
        capacity_limit: int | None = None,
    ) -> BudgetDecision:
        """Atomically compute the decision AND reserve the call(s).

        A single call (``calls=1``) is decided from the usage BEFORE it; a
        blocked decision reserves nothing (the caller must not proceed). A
        multi-call reservation is decided against the EARLIEST policy rung
        crossed by any individual physical call (see ``_batch_decision``): a
        batch that crosses a stop rung (or the hard cap) is rejected
        all-or-nothing, and a batch that crosses a degrade rung is degraded.
        A blocked or batch-degraded decision is ``semantic_only`` — the
        semantic runtime degrades to FTS-only without issuing paid calls —
        and reserves nothing. This is the single atomic decision+reservation
        surface the ``BudgetedClient`` and the semantic backfill use, so
        simultaneous planner/embed reservations for the same chat can never
        exceed the cap. A provider failure after a successful reservation
        retains it (the call count stays incremented); the caller records
        tokens later with ``calls=0``.

        With a ``BudgetStore`` the decision+reservation is ONE writer
        transaction, so DISTINCT manager instances over the same database
        serialize here (never exceeding the cap). Without one, the legacy
        per-instance lock path is used.
        """
        day = self._day(self._now())
        key = self._key(chat_key, day)
        requested_calls = max(0, calls)

        if self._store is not None:
            holder: dict[str, BudgetDecision] = {}

            def transform(raw: str | None) -> str | None:
                usage = self._parse(raw, day)
                if usage is None:
                    usage = self._fresh(day)
                decision = self._batch_decision(usage, requested_calls, capacity_limit)
                holder["d"] = decision
                if decision.semantic_only:
                    return None  # blocked / batch-degraded: reserve nothing
                cap = self._config.daily_cap if capacity_limit is None else max(
                    0, min(self._config.daily_cap, capacity_limit)
                )
                if usage.calls + requested_calls > cap:
                    decision = BudgetDecision(
                        kind=BLOCKED, usage=usage, remaining=max(0, cap - usage.calls),
                        semantic_only=True,
                    )
                    holder["d"] = decision
                    return None
                new_calls = min(cap, usage.calls + requested_calls)
                new_usage = BudgetUsage(
                    day=day, calls=new_calls, tokens=usage.tokens, cost=usage.cost
                )
                return self._serialize(new_usage)

            await self._store.budget_update(key, transform=transform)
            return holder["d"]

        async with self._lock:
            usage = self._parse(await self._repo.get_kv(key), day)
            if usage is None:
                usage = self._fresh(day)
            decision = self._batch_decision(usage, requested_calls, capacity_limit)
            if decision.semantic_only:
                return decision
            cap = self._config.daily_cap if capacity_limit is None else max(
                0, min(self._config.daily_cap, capacity_limit)
            )
            if usage.calls + requested_calls > cap:
                return BudgetDecision(
                    kind=BLOCKED, usage=usage, remaining=max(0, cap - usage.calls),
                    semantic_only=True,
                )
            new_calls = min(cap, usage.calls + requested_calls)
            new_usage = BudgetUsage(
                day=day, calls=new_calls, tokens=usage.tokens, cost=usage.cost
            )
            await self._repo.set_kv(key, self._serialize(new_usage))
            return decision

    async def record(
        self,
        chat_key: ChatKey,
        *,
        calls: int = 1,
        tokens: int = 0,
        cost: float = 0.0,
    ) -> BudgetUsage:
        """Persist one call's usage and return the new immutable snapshot.

        Load-modify-save runs under an internal lock so concurrent
        ``record`` calls are serialized and never lose an update (and, with a
        ``BudgetStore``, atomically across DISTINCT manager instances over
        the same database). Negative inputs are clamped to zero (no
        underflow); ``calls`` is clamped to the cap (no overflow).
        """
        day = self._day(self._now())
        key = self._key(chat_key, day)
        if self._store is not None:
            holder: dict[str, BudgetUsage] = {}

            def transform(raw: str | None) -> str | None:
                usage = self._parse(raw, day)
                if usage is None:
                    usage = self._fresh(day)
                cap = self._config.daily_cap
                new_calls = min(cap, usage.calls + max(0, calls))
                new_tokens = usage.tokens + max(0, tokens)
                new_cost = usage.cost + max(0.0, cost)
                new_usage = BudgetUsage(
                    day=day, calls=new_calls, tokens=new_tokens, cost=new_cost
                )
                holder["u"] = new_usage
                return self._serialize(new_usage)

            await self._store.budget_update(key, transform=transform)
            return holder["u"]

        async with self._lock:
            usage = self._parse(await self._repo.get_kv(key), day)
            if usage is None:
                usage = self._fresh(day)
            cap = self._config.daily_cap
            new_calls = min(cap, usage.calls + max(0, calls))
            new_tokens = usage.tokens + max(0, tokens)
            new_cost = usage.cost + max(0.0, cost)
            new_usage = BudgetUsage(
                day=day, calls=new_calls, tokens=new_tokens, cost=new_cost
            )
            await self._repo.set_kv(key, self._serialize(new_usage))
            return new_usage

    # ── pure policy ──────────────────────────────────────────────────────────

    def _decide(self, usage: BudgetUsage) -> BudgetDecision:
        cap = self._config.daily_cap
        calls = usage.calls
        remaining = max(0, cap - calls)
        if calls >= cap:
            # Hard cap: always blocked, regardless of configured rungs.
            engaged = [r for r in self._config.rungs if r.at <= 1.0]
            top = engaged[-1] if engaged else None
            return BudgetDecision(
                kind=BLOCKED,
                usage=usage,
                remaining=0,
                rung=top,
                semantic_only=True,
            )
        fraction = calls / cap
        engaged = [r for r in self._config.rungs if r.at <= fraction]
        top = engaged[-1] if engaged else None
        if top is None:
            return BudgetDecision(kind=ALLOWED, usage=usage, remaining=remaining)
        if top.action == "stop":
            return BudgetDecision(
                kind=BLOCKED,
                usage=usage,
                remaining=remaining,
                rung=top,
                semantic_only=True,
            )
        if top.action == "degrade":
            return BudgetDecision(
                kind=DEGRADE,
                usage=usage,
                remaining=remaining,
                rung=top,
                degrade=self._engaged_degrade_actions(engaged),
            )
        # warn
        return BudgetDecision(kind=ALLOWED, usage=usage, remaining=remaining, rung=top)

    def _batch_decision(
        self, usage: BudgetUsage, requested_calls: int,
        capacity_limit: int | None = None,
    ) -> BudgetDecision:
        """Decision for a reservation of ``requested_calls`` physical calls.

        A single call (``requested_calls <= 1``) keeps the pre-call decision
        unchanged. For a multi-call batch, each individual call is admitted
        against the usage BEFORE it (``usage.calls + i - 1``), and the batch
        follows the EARLIEST policy rung crossed by any individual call: the
        first call whose decision is non-allowed decides the batch kind — a
        stop rung (or the hard cap) rejects the whole batch all-or-nothing, a
        degrade rung degrades it. A batch that stays allowed throughout keeps
        the pre-batch decision. A rejected or degraded batch is
        ``semantic_only``: the semantic runtime degrades to FTS-only without
        issuing paid calls, and the reservation is not consumed.
        """
        if capacity_limit is not None and usage.calls >= capacity_limit:
            return BudgetDecision(
                kind=BLOCKED, usage=usage, remaining=0, semantic_only=True
            )
        d0 = self._decide(usage)
        if requested_calls <= 1 or d0.kind == BLOCKED:
            return d0
        cap = self._config.daily_cap
        for i in range(1, requested_calls + 1):
            if capacity_limit is not None and usage.calls + i > capacity_limit:
                return BudgetDecision(
                    kind=BLOCKED, usage=usage, remaining=0, semantic_only=True
                )
            step_calls = min(cap, usage.calls + i - 1)
            step_usage = BudgetUsage(
                day=usage.day, calls=step_calls, tokens=usage.tokens, cost=usage.cost
            )
            d_i = self._decide(step_usage)
            if d_i.kind == BLOCKED:
                return BudgetDecision(
                    kind=BLOCKED,
                    usage=usage,
                    remaining=0,
                    rung=d_i.rung,
                    semantic_only=True,
                )
            if d_i.kind == DEGRADE:
                return BudgetDecision(
                    kind=DEGRADE,
                    usage=usage,
                    remaining=d0.remaining,
                    rung=d_i.rung,
                    degrade=d_i.degrade,
                    semantic_only=True,
                )
        return d0

    def _engaged_degrade_actions(self, engaged: list[BudgetRung]) -> tuple[str, ...]:
        """Canonical actions of the engaged degrade rungs, in stable order."""
        degrade_rungs = [r for r in self._config.rungs if r.action == "degrade"]
        rank = {id(r): i for i, r in enumerate(degrade_rungs)}
        out: list[str] = []
        for r in engaged:
            if r.action != "degrade":
                continue
            i = rank.get(id(r), 0)
            action = DEGRADE_ACTIONS[i] if i < len(DEGRADE_ACTIONS) else DEGRADE_ACTIONS[-1]
            out.append(action)
        return tuple(out)

    # ── KV helpers ───────────────────────────────────────────────────────────

    def _day(self, ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()

    def _key(self, chat_key: ChatKey, day: str) -> str:
        return f"{_KV_PREFIX}{chat_key}:{day}"

    def _fresh(self, day: str) -> BudgetUsage:
        return BudgetUsage(day=day, calls=0, tokens=0, cost=0.0)

    def _parse(self, raw: str | None, day: str) -> BudgetUsage | None:
        """Reconcile a stored KV value into a usage for ``day``.

        Returns None (caller falls back to a fresh day) for a missing key,
        a malformed blob, or a value belonging to a different day (stale ->
        rollover). Surviving values are sanitized: counts are nonnegative
        ints, cost is a finite nonnegative float, and calls never exceed the
        cap.
        """
        if raw is None:
            return None
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if not isinstance(obj, dict):
            return None
        try:
            stored_day = obj["day"]
            calls = int(obj["calls"])
            tokens = int(obj["tokens"])
            cost = float(obj["cost"])
        except (KeyError, TypeError, ValueError):
            return None
        if stored_day != day:
            return None
        calls = max(0, min(calls, self._config.daily_cap))
        tokens = max(0, tokens)
        if not math.isfinite(cost) or cost < 0:
            cost = 0.0
        return BudgetUsage(day=day, calls=calls, tokens=tokens, cost=cost)

    def _serialize(self, usage: BudgetUsage) -> str:
        """Stable serialization: fixed fields, sorted keys, deterministic."""
        return json.dumps(
            {
                "day": usage.day,
                "calls": usage.calls,
                "tokens": usage.tokens,
                "cost": usage.cost,
            },
            sort_keys=True,
        )


# ── Per-call budgeted client (frozen Oracle runtime design) ──────────────────
# The budget is enforced PER PROVIDER CALL, not per saga: a chat-bound
# decorator wraps the LLMClient that BOTH the Planner and the Replyer use.
# Before every delegate request the call is reserved atomically with the
# decision (``reserve(calls=1)`` — the decision and the reservation share one
# lock, so a concurrent planner/embed reservation for the same chat can never
# exceed the cap); a blocked decision raises ``BudgetBlockedError`` before any
# delegate request; after a successful delegate call the tokens are recorded
# with ``calls=0`` (the reservation already counted the call); a provider
# error retains the reservation (the call count stays incremented, no tokens
# recorded). Degrade actions are applied to the delegated call:
# ``context_reduction`` trims the transcript, ``profile_fallback`` switches to
# the configured fallback profile, and ``capability_flags`` drops the tool
# schema.


class LearnerBudget:
    """A narrow concurrency-bounded wrapper over a shared ``BudgetManager``
    for background learner work (Phase 6 P6.4).

    The learner worker shares the SAME physical per-chat budget state as the
    foreground planner (one atomic cap per chat), while bounding how many
    learner runs may hold a reservation at once to ``concurrency -
    foreground_reserve`` — the foreground reserve is never consumed by
    background learners. ``reserve`` acquires a slot before delegating and
    releases it immediately when the decision is not allowed (the pipeline
    skips with zero provider calls); ``record`` delegates and releases the
    slot. A provider failure that skips ``record`` is released by the
    worker's ``release`` (the wrapper tracks held slots, so a double release
    is a no-op).
    """

    def __init__(
        self,
        manager: BudgetManager,
        *,
        concurrency: int,
        foreground_reserve: int,
        daily_capacity_reserve: int = 0,
        daily_reserve: int | None = None,
        budget_for: Callable[[ChatKey], BudgetManager] | None = None,
    ) -> None:
        if (
            isinstance(concurrency, bool)
            or not isinstance(concurrency, int)
            or concurrency < 1
        ):
            raise ValueError("concurrency must be a positive integer")
        if (
            isinstance(foreground_reserve, bool)
            or not isinstance(foreground_reserve, int)
            or foreground_reserve < 0
        ):
            raise ValueError("foreground_reserve must be a nonnegative integer")
        if foreground_reserve >= concurrency:
            raise ValueError(
                "foreground_reserve must be strictly below concurrency"
            )
        if daily_reserve is not None:
            daily_capacity_reserve = daily_reserve
        if (
            isinstance(daily_capacity_reserve, bool)
            or not isinstance(daily_capacity_reserve, int)
            or daily_capacity_reserve < 0
        ):
            raise ValueError("daily_capacity_reserve must be a nonnegative integer")
        self._manager = manager
        self._budget_for = budget_for
        self._slots = max(1, concurrency - foreground_reserve)
        self._daily_capacity_reserve = daily_capacity_reserve
        self._sem = asyncio.Semaphore(self._slots)
        # A semaphore count alone is not an ownership model: a finally block
        # in task B must never release task A's reservation.  Keep one
        # reservation lease per owning asyncio task and release only that
        # task's lease.  The public API remains compatible with old callers.
        self._reservations: dict[asyncio.Task[object], list[tuple[ChatKey, BudgetManager]]] = {}

    @property
    def slots(self) -> int:
        """The background concurrency bound (``concurrency -
        foreground_reserve``, at least 1)."""
        return self._slots

    @property
    def daily_capacity_reserve(self) -> int:
        """Calls kept available to foreground work for the current day."""
        return self._daily_capacity_reserve

    async def reserve(self, chat_key: ChatKey, *, calls: int = 1) -> BudgetDecision:
        """Acquire a background slot, then delegate the atomic reservation.

        A blocked/degraded decision releases the slot immediately (the
        pipeline skips with zero calls and the slot is never leaked)."""
        await self._sem.acquire()
        task = asyncio.current_task()
        if task is None:
            self._sem.release()
            raise RuntimeError("LearnerBudget.reserve requires an asyncio task")
        manager = self._budget_for(chat_key) if self._budget_for is not None else self._manager
        capacity_limit = max(
            0, manager.config.daily_cap - self._daily_capacity_reserve
        )
        try:
            try:
                decision = await manager.reserve(
                    chat_key, calls=calls, capacity_limit=capacity_limit
                )
            except TypeError:
                # Narrow compatibility for injected legacy budget fakes.  A
                # fake that cannot express a daily reserve is safe only when
                # no reserve was requested.
                if self._daily_capacity_reserve:
                    raise
                decision = await manager.reserve(chat_key, calls=calls)
        except BaseException:
            # The reservation is not owned until the decision is accepted;
            # cancellation while the manager is awaiting must still return
            # the semaphore slot to the pool.
            self._sem.release()
            raise
        if decision.kind != ALLOWED:
            self._sem.release()
        else:
            self._reservations.setdefault(task, []).append((chat_key, manager))
        return decision

    async def record(
        self,
        chat_key: ChatKey,
        *,
        calls: int = 1,
        tokens: int = 0,
        cost: float = 0.0,
    ) -> BudgetUsage:
        """Delegate the usage record and release the held background slot."""
        task = asyncio.current_task()
        lease = self._take(task, chat_key) if task is not None else None
        manager = lease[1] if lease is not None else (
            self._budget_for(chat_key) if self._budget_for is not None else self._manager
        )
        try:
            return await manager.record(
                chat_key, calls=calls, tokens=tokens, cost=cost
            )
        finally:
            if lease is not None:
                self._sem.release()

    def release(self) -> None:
        """Release any held slot (worker shutdown / provider-error path)."""
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is not None:
            self._release(task)

    def _take(
        self, task: asyncio.Task[object] | None, chat_key: ChatKey | None
    ) -> tuple[ChatKey, BudgetManager] | None:
        if task is None:
            return None
        leases = self._reservations.get(task)
        if not leases:
            return None
        for i in range(len(leases) - 1, -1, -1):
            if chat_key is None or leases[i][0] == chat_key:
                lease = leases.pop(i)
                if not leases:
                    self._reservations.pop(task, None)
                return lease
        return None

    def _release(self, task: asyncio.Task[object]) -> None:
        leases = self._reservations.get(task)
        if leases:
            leases.pop()
            if not leases:
                self._reservations.pop(task, None)
            self._sem.release()

    @property
    def _held(self) -> int:
        """Compatibility/debug view of reservations owned by this wrapper."""
        return sum(len(leases) for leases in self._reservations.values())


class BudgetBlockedError(PretenderError):
    """Raised by ``BudgetedClient`` when the chat's daily budget is
    exhausted (the hard cap is reached) before a delegate request."""


class BudgetedClient:
    """A chat-bound per-call budget decorator over an ``LLMClient``.

    Implements the ``LLMClient`` protocol (``complete``) so it can be handed
    to both the Planner and the Replyer. ``chat_key`` is fixed at
    construction — the agent runtime builds one per chat per run.
    """

    def __init__(
        self,
        delegate: LLMClient,
        budget: BudgetManager,
        chat_key: ChatKey,
        *,
        agent_config: AgentConfig | None = None,
    ) -> None:
        self._delegate = delegate
        self._budget = budget
        self._chat_key = chat_key
        self._agent_config = (
            agent_config if agent_config is not None else AgentConfig()
        )

    async def complete(
        self,
        messages: list[TranscriptMessage],
        *,
        profile: str,
        tools: list[dict[str, object]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        deadline: float | None = None,
    ):
        """Admit, reserve, delegate, and record one provider call.

        The decision AND the call reservation are computed atomically under
        the manager's lock (``reserve``), so a concurrent planner/embed
        reservation for the same chat can never push the call count past the
        cap. A blocked decision raises before any delegate request and
        reserves nothing.
        """
        decision = await self._budget.reserve(self._chat_key, calls=1)
        if decision.kind == BLOCKED:
            raise BudgetBlockedError(
                f"daily budget exhausted for {self._chat_key} "
                f"({decision.usage.calls}/{self._budget.config.daily_cap} calls)"
            )
        profile, messages, tools = self._apply_degrade(
            decision, profile, messages, tools
        )
        try:
            resp = await self._delegate.complete(
                messages,
                profile=profile,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                deadline=deadline,
            )
        except Exception:
            # Retain the reservation: the call count stays incremented and
            # no tokens are recorded (the provider error is the caller's to
            # handle — a retry re-reserves).
            raise
        tokens = _usage_tokens(resp.usage)
        await self._budget.record(self._chat_key, calls=0, tokens=tokens)
        return resp

    def _apply_degrade(
        self,
        decision: BudgetDecision,
        profile: str,
        messages: list[TranscriptMessage],
        tools: list[dict[str, object]] | None,
    ) -> tuple[str, list[TranscriptMessage], list[dict[str, object]] | None]:
        """Apply the decision's canonical degrade actions to the delegated
        call: context reduction trims the transcript, profile fallback
        switches to the configured fallback profile, capability flags drop
        the tool schema. An allowed/blocked decision changes nothing."""
        if decision.kind != DEGRADE:
            return profile, messages, tools
        actions = set(decision.degrade)
        if (
            DEGRADE_PROFILE_FALLBACK in actions
            and self._agent_config.fallback_profile is not None
        ):
            profile = self._agent_config.fallback_profile
        if DEGRADE_CONTEXT_REDUCTION in actions:
            messages = _reduce_context(messages)
        if DEGRADE_CAPABILITY_FLAGS in actions:
            tools = None
        return profile, messages, tools


def _reduce_context(
    messages: list[TranscriptMessage], keep: int = 8
) -> list[TranscriptMessage]:
    """Trim a transcript for context reduction: keep the system prompt (the
    first message) plus the ``keep - 1`` most recent messages."""
    if len(messages) <= keep:
        return messages
    return [messages[0], *messages[-(keep - 1) :]]


def _usage_tokens(usage: dict[str, int] | None) -> int:
    """The total token count of one provider response."""
    return int((usage or {}).get("prompt_tokens", 0)) + int(
        (usage or {}).get("completion_tokens", 0)
    )
