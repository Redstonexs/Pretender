"""Idle backoff policy (PLAN.md §1.B; frozen gate spec).

Pure and dependency-free: no Clock, no scheduler, no polling. The
``IdleBackoffController`` decides, for one terminal cycle outcome,
whether the group-only idle backoff applies and what the durable idle
streak becomes afterwards. The gate lane records the returned
``BackoffFacts`` in the DecisionTrace; the session lane persists the
``next_streak`` result.

Frozen semantics:
- Backoff is zero until the idle streak reaches ``start_count``, then
  ``min(cap, base * 2**(streak - start_count))``.
- Only the listed idle end reasons count as an idle cycle:
  ``planner_no_tool_end``, ``planner_wait_rest``, ``tool_pause:wait``.
- Group chats only: a private-chat cycle never applies backoff.
- Focus, a hard trigger, or high pending (``pending >= threshold``)
  bypasses backoff and resets the streak; any non-idle terminal cycle
  also resets it.
"""

from __future__ import annotations

import math

from pretender.types import BackoffFacts

# The only terminal end reasons that count as an idle cycle (PLAN.md §1.B).
IDLE_END_REASONS: frozenset[str] = frozenset(
    {"planner_no_tool_end", "planner_wait_rest", "tool_pause:wait"}
)

# Stable bypass tokens recorded in BackoffFacts.bypass_reason. Consumers
# match on these exact strings, never on free-form text.
BYPASS_NOT_GROUP = "not_group"
BYPASS_FOCUS = "focus"
BYPASS_HARD_TRIGGER = "hard_trigger"
BYPASS_HIGH_PENDING = "high_pending"
BYPASS_NON_IDLE = "non_idle"


def _check_number(name: str, value: float) -> None:
    """A duration parameter: a real number, finite, nonnegative."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")


def _check_count(name: str, value: int) -> None:
    """A count parameter: an integer, nonnegative."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")


class IdleBackoffController:
    """Pure idle-backoff policy for one chat's gate/session lane.

    Holds only configuration; every method is a pure function of its
    arguments, so one controller is safe to share across chats and
    cycles. The durable idle streak itself lives in the session layer
    (``ChatState.idle_streak``); this controller never mutates it.
    """

    def __init__(
        self,
        *,
        base_s: float = 15.0,
        cap_s: float = 300.0,
        start_count: int = 2,
        threshold: int = 8,
    ) -> None:
        """Defaults match ``GateBackoffConfig`` (15 s base, 300 s cap,
        start 2) and the gate's default ``threshold`` of 8."""
        _check_number("base_s", base_s)
        _check_number("cap_s", cap_s)
        _check_count("start_count", start_count)
        _check_count("threshold", threshold)
        self._base_s = float(base_s)
        self._cap_s = float(cap_s)
        self._start_count = start_count
        self._threshold = threshold

    # ── configuration (read-only) ────────────────────────────────────────────

    @property
    def base_s(self) -> float:
        return self._base_s

    @property
    def cap_s(self) -> float:
        return self._cap_s

    @property
    def start_count(self) -> int:
        return self._start_count

    @property
    def threshold(self) -> int:
        return self._threshold

    # ── pure policy ──────────────────────────────────────────────────────────

    def seconds(self, streak: int) -> float:
        """Backoff duration for ``streak`` consecutive idle cycles: zero
        until ``streak >= start_count``, then
        ``min(cap, base * 2**(streak - start_count))``."""
        _check_count("streak", streak)
        if streak < self._start_count:
            return 0.0
        n = streak - self._start_count
        # Overflow guard: once base * 2**n reaches the cap the min() is the
        # cap, so a huge streak returns cap instead of raising OverflowError.
        if self._base_s > 0.0 and n >= math.log2(self._cap_s / self._base_s):
            return self._cap_s
        return min(self._cap_s, self._base_s * 2.0 ** n)

    def is_idle_end(self, end_reason: str | None) -> bool:
        """True when ``end_reason`` is one of the listed idle endings."""
        return end_reason in IDLE_END_REASONS

    def evaluate(
        self,
        streak: int,
        *,
        is_group: bool,
        end_reason: str | None,
        is_focused: bool,
        pending: int,
        hard_trigger: bool = False,
    ) -> BackoffFacts:
        """Decide whether idle backoff applies to this cycle.

        Returns the ``BackoffFacts`` the DecisionTrace records:
        ``applied=True`` with the duration when the group-only backoff
        delays the cycle; otherwise ``applied=False`` with
        ``bypass_reason`` naming why ("not_group", "focus",
        "hard_trigger", "high_pending", "non_idle") — or None when the
        streak is simply below ``start_count``.
        """
        _check_count("streak", streak)
        if not self.is_idle_end(end_reason):
            return BackoffFacts(applied=False, bypass_reason=BYPASS_NON_IDLE)
        reason = self._application_bypass(
            is_group=is_group,
            is_focused=is_focused,
            hard_trigger=hard_trigger,
            pending=pending,
        )
        if reason is not None:
            return BackoffFacts(applied=False, bypass_reason=reason)
        seconds = self.seconds(streak)
        if seconds <= 0.0:
            return BackoffFacts(applied=False)
        return BackoffFacts(applied=True, seconds=seconds)

    def next_streak(
        self,
        streak: int,
        *,
        is_group: bool,
        end_reason: str | None,
        is_focused: bool,
        pending: int,
        hard_trigger: bool = False,
    ) -> int:
        """The durable idle streak after this cycle: ``streak + 1`` for an
        idle end that focus, a hard trigger, or high pending did not
        interrupt, else 0. Group membership never resets the streak — it
        only gates whether backoff applies (``evaluate``)."""
        _check_count("streak", streak)
        if not self.is_idle_end(end_reason):
            return 0
        if (
            self._reset_bypass(
                is_focused=is_focused,
                hard_trigger=hard_trigger,
                pending=pending,
            )
            is not None
        ):
            return 0
        return streak + 1

    # ── internals ────────────────────────────────────────────────────────────

    def _application_bypass(
        self,
        *,
        is_group: bool,
        is_focused: bool,
        hard_trigger: bool,
        pending: int,
    ) -> str | None:
        """Why backoff is NOT applied, in priority order: group-only
        first, then focus, then hard trigger, then high pending."""
        if not is_group:
            return BYPASS_NOT_GROUP
        return self._reset_bypass(
            is_focused=is_focused,
            hard_trigger=hard_trigger,
            pending=pending,
        )

    def _reset_bypass(
        self,
        *,
        is_focused: bool,
        hard_trigger: bool,
        pending: int,
    ) -> str | None:
        """Why the idle streak resets: focus, hard trigger, or high
        pending. Group membership is NOT a reset condition — it only
        gates whether backoff applies."""
        if is_focused:
            return BYPASS_FOCUS
        if hard_trigger:
            return BYPASS_HARD_TRIGGER
        if pending >= self._threshold:
            return BYPASS_HIGH_PENDING
        return None