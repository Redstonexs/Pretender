"""IdleBackoffController: growth sequence/cap, threshold edge, group-only,
listed/rejected end reasons, focus/hard-trigger/high-pending bypasses,
reset behavior, and invalid/nonnegative parameter handling."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from pretender.backoff import (
    BYPASS_FOCUS,
    BYPASS_HARD_TRIGGER,
    BYPASS_HIGH_PENDING,
    BYPASS_NON_IDLE,
    BYPASS_NOT_GROUP,
    IDLE_END_REASONS,
    IdleBackoffController,
)
from pretender.config import GateConfig
from pretender.types import BackoffFacts

IDLE = "planner_no_tool_end"
WAIT = "planner_wait_rest"
PAUSE = "tool_pause:wait"


def ctl(**kw) -> IdleBackoffController:
    return IdleBackoffController(**kw)


def idle_kwargs(**over) -> dict:
    """A cycle that is idle, in a group, unfocused, with no pending."""
    kw = dict(is_group=True, end_reason=IDLE, is_focused=False, pending=0)
    kw.update(over)
    return kw


# ── Growth sequence and cap ─────────────────────────────────────────────────

def test_zero_before_start_count():
    c = ctl()
    assert c.seconds(0) == 0.0
    assert c.seconds(1) == 0.0
    facts = c.evaluate(1, **idle_kwargs())
    assert facts.applied is False
    assert facts.seconds is None
    assert facts.bypass_reason is None  # below start_count is not a bypass


def test_growth_sequence_defaults():
    # base 15, cap 300, start 2: 15, 30, 60, 120, 240, then capped at 300.
    c = ctl()
    assert c.seconds(2) == 15.0
    assert c.seconds(3) == 30.0
    assert c.seconds(4) == 60.0
    assert c.seconds(5) == 120.0
    assert c.seconds(6) == 240.0
    assert c.seconds(7) == 300.0
    assert c.seconds(100) == 300.0


def test_cap_never_exceeded():
    c = ctl(base_s=10.0, cap_s=40.0, start_count=1)
    assert c.seconds(1) == 10.0
    assert c.seconds(2) == 20.0
    assert c.seconds(3) == 40.0
    assert c.seconds(4) == 40.0
    assert c.seconds(10**6) == 40.0  # huge streaks stay capped, no overflow


def test_cap_below_base_clamps_immediately():
    c = ctl(base_s=15.0, cap_s=10.0)
    assert c.seconds(2) == 10.0
    assert c.seconds(9) == 10.0


def test_start_count_zero_backs_off_first_idle_cycle():
    c = ctl(start_count=0)
    assert c.seconds(0) == 15.0
    assert c.seconds(1) == 30.0


def test_zero_base_never_applies():
    c = ctl(base_s=0.0)
    assert c.seconds(5) == 0.0
    facts = c.evaluate(5, **idle_kwargs())
    assert facts.applied is False
    assert facts.bypass_reason is None


def test_evaluate_applied_carries_duration():
    facts = ctl().evaluate(4, **idle_kwargs())
    assert facts.applied is True
    assert facts.seconds == 60.0
    assert facts.bypass_reason is None


# ── Threshold edge ──────────────────────────────────────────────────────────

def test_high_pending_threshold_edge():
    # pending == threshold - 1: backoff still applies; pending == threshold: bypass.
    c = ctl(threshold=8)
    assert c.evaluate(4, **idle_kwargs(pending=7)).applied is True
    facts = c.evaluate(4, **idle_kwargs(pending=8))
    assert facts.applied is False
    assert facts.bypass_reason == BYPASS_HIGH_PENDING
    assert c.evaluate(4, **idle_kwargs(pending=99)).bypass_reason == BYPASS_HIGH_PENDING


def test_high_pending_custom_threshold():
    c = ctl(threshold=3)
    assert c.evaluate(4, **idle_kwargs(pending=2)).applied is True
    assert c.evaluate(4, **idle_kwargs(pending=3)).bypass_reason == BYPASS_HIGH_PENDING


def test_threshold_zero_always_high_pending():
    c = ctl(threshold=0)
    assert c.evaluate(4, **idle_kwargs(pending=0)).bypass_reason == BYPASS_HIGH_PENDING


# ── Group-only behavior ─────────────────────────────────────────────────────

def test_group_only_never_applies_in_private():
    c = ctl()
    facts = c.evaluate(6, **idle_kwargs(is_group=False))
    assert facts.applied is False
    assert facts.bypass_reason == BYPASS_NOT_GROUP
    # even a huge streak never applies in a private chat
    assert c.evaluate(100, **idle_kwargs(is_group=False)).applied is False


def test_group_only_gates_application_not_streak():
    # Group membership is not a reset condition: an idle end still grows
    # the durable streak; only evaluate() refuses to apply backoff.
    c = ctl()
    assert c.next_streak(5, **idle_kwargs(is_group=False)) == 6
    assert c.next_streak(5, **idle_kwargs(is_group=True)) == 6


def test_group_applies():
    assert ctl().evaluate(3, **idle_kwargs(is_group=True)).applied is True


# ── Listed / rejected end reasons ───────────────────────────────────────────

def test_listed_idle_end_reasons_are_exact():
    assert IDLE_END_REASONS == frozenset(
        {"planner_no_tool_end", "planner_wait_rest", "tool_pause:wait"}
    )


def test_each_listed_end_reason_is_idle():
    c = ctl()
    for reason in (IDLE, WAIT, PAUSE):
        assert c.is_idle_end(reason)
        assert c.evaluate(4, **idle_kwargs(end_reason=reason)).applied is True
        assert c.next_streak(4, **idle_kwargs(end_reason=reason)) == 5


def test_rejected_end_reasons_reset_and_bypass():
    c = ctl()
    for reason in ("trigger", "skip", "delay", "reply", "error", "planner_tool_end", None):
        assert not c.is_idle_end(reason)
        facts = c.evaluate(6, **idle_kwargs(end_reason=reason))
        assert facts.applied is False
        assert facts.bypass_reason == BYPASS_NON_IDLE
        assert c.next_streak(6, **idle_kwargs(end_reason=reason)) == 0


def test_non_idle_outranks_other_bypasses():
    c = ctl()
    facts = c.evaluate(
        6,
        **idle_kwargs(
            end_reason="trigger", is_focused=True, hard_trigger=True, pending=99
        ),
    )
    assert facts.bypass_reason == BYPASS_NON_IDLE


# ── Focus / hard-trigger / high-pending bypasses ────────────────────────────

def test_focus_bypasses_and_resets():
    c = ctl()
    facts = c.evaluate(6, **idle_kwargs(is_focused=True))
    assert facts.applied is False
    assert facts.bypass_reason == BYPASS_FOCUS
    assert c.next_streak(6, **idle_kwargs(is_focused=True)) == 0


def test_hard_trigger_bypasses_and_resets():
    c = ctl()
    facts = c.evaluate(6, **idle_kwargs(hard_trigger=True))
    assert facts.applied is False
    assert facts.bypass_reason == BYPASS_HARD_TRIGGER
    assert c.next_streak(6, **idle_kwargs(hard_trigger=True)) == 0


def test_bypass_priority_focus_then_hard_trigger_then_high_pending():
    c = ctl()
    assert (
        c.evaluate(6, **idle_kwargs(is_focused=True, hard_trigger=True, pending=99)).bypass_reason
        == BYPASS_FOCUS
    )
    assert (
        c.evaluate(6, **idle_kwargs(hard_trigger=True, pending=99)).bypass_reason
        == BYPASS_HARD_TRIGGER
    )
    assert c.evaluate(6, **idle_kwargs(pending=99)).bypass_reason == BYPASS_HIGH_PENDING


# ── Reset behavior ──────────────────────────────────────────────────────────

def test_streak_grows_on_consecutive_idle_cycles():
    c = ctl()
    streak = 0
    for expected in (1, 2, 3, 4):
        streak = c.next_streak(streak, **idle_kwargs())
        assert streak == expected


def test_non_idle_terminal_resets_streak():
    c = ctl()
    assert c.next_streak(7, **idle_kwargs(end_reason="skip")) == 0
    assert c.next_streak(7, **idle_kwargs(end_reason="trigger")) == 0


def test_focus_resets_streak_mid_sequence():
    c = ctl()
    s = c.next_streak(0, **idle_kwargs())
    s = c.next_streak(s, **idle_kwargs())
    assert s == 2
    assert c.next_streak(s, **idle_kwargs(is_focused=True)) == 0


def test_high_pending_resets_streak():
    c = ctl()
    assert c.next_streak(5, **idle_kwargs(pending=8)) == 0


def test_hard_trigger_resets_streak():
    c = ctl()
    assert c.next_streak(5, **idle_kwargs(hard_trigger=True)) == 0


def test_reset_then_growth_restarts_from_zero():
    c = ctl()
    s = c.next_streak(0, **idle_kwargs())
    s = c.next_streak(s, **idle_kwargs())
    s = c.next_streak(s, **idle_kwargs(end_reason="skip"))
    assert s == 0
    assert c.next_streak(s, **idle_kwargs()) == 1


# ── Invalid / nonnegative parameter handling ────────────────────────────────

def test_negative_config_rejected():
    with pytest.raises(ValueError):
        ctl(base_s=-1.0)
    with pytest.raises(ValueError):
        ctl(cap_s=-0.5)
    with pytest.raises(ValueError):
        ctl(start_count=-1)
    with pytest.raises(ValueError):
        ctl(threshold=-2)


def test_nonfinite_config_rejected():
    with pytest.raises(ValueError):
        ctl(base_s=math.nan)
    with pytest.raises(ValueError):
        ctl(cap_s=math.inf)
    with pytest.raises(ValueError):
        ctl(base_s=-math.inf)


def test_non_integer_counts_rejected():
    with pytest.raises(TypeError):
        ctl(start_count=2.5)
    with pytest.raises(TypeError):
        ctl(start_count="2")
    with pytest.raises(TypeError):
        ctl(threshold=8.0)
    with pytest.raises(TypeError):
        ctl(threshold=True)


def test_non_number_durations_rejected():
    with pytest.raises(TypeError):
        ctl(base_s="15")
    with pytest.raises(TypeError):
        ctl(cap_s=True)


def test_negative_streak_rejected():
    c = ctl()
    with pytest.raises(ValueError):
        c.seconds(-1)
    with pytest.raises(ValueError):
        c.evaluate(-1, **idle_kwargs())
    with pytest.raises(ValueError):
        c.next_streak(-1, **idle_kwargs())


def test_non_integer_streak_rejected():
    c = ctl()
    with pytest.raises(TypeError):
        c.seconds(2.0)
    with pytest.raises(TypeError):
        c.evaluate(True, **idle_kwargs())
    with pytest.raises(TypeError):
        c.next_streak("3", **idle_kwargs())


# ── Contract compatibility ──────────────────────────────────────────────────

def test_defaults_match_gate_config():
    cfg = GateConfig()
    c = ctl()
    assert c.base_s == cfg.backoff.base_s == 15.0
    assert c.cap_s == cfg.backoff.cap_s == 300.0
    assert c.start_count == cfg.backoff.start_count == 2
    assert c.threshold == cfg.threshold == 8


def test_returns_existing_backoff_facts_type():
    facts = ctl().evaluate(4, **idle_kwargs())
    assert isinstance(facts, BackoffFacts)
    assert facts.applied is True
    assert facts.seconds == 60.0
    # trace-compatible: asdict + dumps round-trips losslessly
    data = json.loads(json.dumps(dataclasses.asdict(facts)))
    assert data == {"applied": True, "seconds": 60.0, "bypass_reason": None}