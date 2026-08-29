"""Gate: the pure evaluator, the five built-in GateFeatures, composition,
both modes, precedence, fail-closed error behavior, and the replay-friendly
DecisionTrace (PLAN.md §1.B; frozen gate spec).

Golden pins from PLAN.md §8: pending 20 → 78 (no trigger), 21 → 80
(trigger); direct @ always ≥ 100; an all-short-reaction batch scoring below
a single long message; presence penalty 0 at ratio 0.25 and −25 at 0.60;
empty pending + infinite idle yields delay, never trigger.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

import pytest

from pretender.backoff import (
    BYPASS_FOCUS,
    BYPASS_HARD_TRIGGER,
    BYPASS_HIGH_PENDING,
    BYPASS_NON_IDLE,
    BYPASS_NOT_GROUP,
)
from pretender.gate import (
    ContentFeature,
    FrequencyScaleFeature,
    Gate,
    PresenceFeature,
    PressureFeature,
    RelevanceFeature,
    compose,
    default_features,
)
from pretender.registry import Registry
from pretender.seams import GateFeature
from pretender.types import (
    BackoffFacts,
    ChatKey,
    Contribution,
    CycleId,
    Decision,
    DecisionTrace,
    GateSnapshot,
    Message,
    MessageRowId,
    Reason,
    SelfId,
    SenderId,
)

CK = ChatKey("qq:group:123456")
SENDER = SenderId("u1")
IDLE_END = "planner_no_tool_end"


def _msg(text: str, **kw: Any) -> Message:
    base: dict[str, Any] = dict(
        chat_key=CK,
        sender_id=SENDER,
        sender_name="alice",
        is_self=False,
        text=text,
    )
    base.update(kw)
    return Message(**base)


def _snapshot(**kw: Any) -> GateSnapshot:
    base: dict[str, Any] = dict(
        chat_key=CK,
        cycle_id=CycleId("c1"),
        start_msg_id=MessageRowId(1),
        through_msg_id=MessageRowId(9),
        evaluated_ts=300.0,
        self_id=SelfId("bot"),
        mode="reply_necessity",
        threshold=8,
        trigger_score=80,
        frequency=1.0,
        pending=1,
        pending_messages=(_msg("hi"),),
        recent=(),
        window_count=1,
        self_count=0,
        last_nonself_ts=250.0,
        idle_seconds=30.0,
        recent_average_interval=60.0,
        self_ratio=0.1,
        is_group=True,
        is_focused=False,
        last_message=None,
    )
    base.update(kw)
    return GateSnapshot(**base)


def _snap(*texts: str, **kw: Any) -> GateSnapshot:
    """A snapshot whose pending batch is one message per text."""
    msgs = tuple(_msg(t) for t in texts)
    kw.setdefault("pending", len(msgs))
    kw.setdefault("pending_messages", msgs)
    return _snapshot(**kw)


def _gate(features: Any = None, **kw: Any) -> Gate:
    return Gate(features, **kw)


def _decision(trace: DecisionTrace) -> Decision:
    """The trace's decision, narrowed (the gate always sets it)."""
    assert trace.decision is not None
    return trace.decision


def _backoff(trace: DecisionTrace) -> BackoffFacts:
    """The trace's backoff facts, narrowed (the gate always sets them)."""
    assert trace.backoff is not None
    return trace.backoff


def _contribution(feature: str, op: str, value: float) -> Contribution:
    return Contribution(feature=feature, op=op, value=value)


# ── Golden pins (PLAN.md §8) ────────────────────────────────────────────────

def test_golden_pending_20_scores_78_no_trigger():
    trace = _gate().evaluate(_snap(*(["x"] * 20)))
    assert _decision(trace).action == "delay"
    assert _decision(trace).score == 78.0
    assert _decision(trace).reason == Reason.DELAY
    # idle 30 < avg 60: timed until the idle bonus activates (60 − 30).
    assert _decision(trace).delay_seconds == 30.0
    assert trace.aggregates == {"max": 0.0, "add": 78.0, "scale": 1.0, "score": 78.0}


def test_golden_pending_21_scores_80_triggers():
    trace = _gate().evaluate(_snap(*(["x"] * 21)))
    assert _decision(trace).action == "trigger"
    assert _decision(trace).score == 80.0
    assert _decision(trace).reason == Reason.TRIGGER


def test_golden_long_text_bonus_triggers_at_15_pending():
    # PLAN.md: with the +10 long-text bonus, ambient chatter crosses 80 at 15.
    trace = _gate().evaluate(_snap(*(["x" * 120] * 15)))
    assert _decision(trace).action == "trigger"
    assert _decision(trace).score == 80.0


def test_golden_all_short_reactions_below_single_long_message():
    short = _gate().evaluate(_snap("哈哈", "嗯嗯", "好的"))
    long = _gate().evaluate(_snap("x" * 120))
    assert _decision(short).score < _decision(long).score
    assert _decision(short).action == "delay"
    assert _decision(long).action == "delay"


def test_golden_empty_pending_infinite_idle_delays_never_triggers():
    for mode in ("reply_necessity", "frequency"):
        trace = _gate().evaluate(
            _snap(mode=mode, pending=0, pending_messages=(), idle_seconds=math.inf)
        )
        assert _decision(trace).action == "delay"
        assert _decision(trace).reason == Reason.DELAY
        assert _decision(trace).delay_seconds is None  # event-only


# ── Hard direct @ / quote triggers ──────────────────────────────────────────

def test_direct_at_hard_trigger():
    trace = _gate().evaluate(_snap("hi", has_direct_at=True))
    assert _decision(trace).action == "trigger"
    assert _decision(trace).score >= 100.0
    assert _decision(trace).reason == Reason.TRIGGER


def test_quote_to_self_hard_trigger():
    trace = _gate().evaluate(_snap("hi", has_quote_to_self=True))
    assert _decision(trace).action == "trigger"
    assert _decision(trace).score >= 100.0
    assert _decision(trace).reason == Reason.TRIGGER


def test_hard_trigger_under_zero_scale():
    class ZeroScale:
        name = "zero_scale"

        def contribute(self, ctx):
            return Contribution(feature="zero_scale", op="scale", value=0.0)

    trace = _gate([ZeroScale()]).evaluate(_snap("hi", has_direct_at=True))
    assert trace.aggregates["scale"] == 0.0
    assert _decision(trace).score == 100.0  # computed 0, floored to 100
    assert _decision(trace).action == "trigger"


def test_hard_trigger_with_presence_at_0_60():
    # computed (100 − 25) · 1 = 75, but the hard trigger floors at 100.
    trace = _gate().evaluate(_snap("hi", has_direct_at=True, self_ratio=0.60))
    assert _decision(trace).action == "trigger"
    assert _decision(trace).score == 100.0


def test_hard_trigger_beats_feature_error():
    class Boom:
        name = "boom"

        def contribute(self, ctx):
            raise RuntimeError("boom")

    trace = _gate([Boom()]).evaluate(_snap("hi", has_direct_at=True))
    assert _decision(trace).action == "trigger"
    assert _decision(trace).reason == Reason.TRIGGER
    errors = [c for c in trace.contributions if c.error is not None]
    assert len(errors) == 1
    assert errors[0].feature == "boom"
    assert errors[0].error is not None and "boom" in errors[0].error


def test_hard_trigger_beats_refusal():
    trace = _gate().evaluate(
        _snap("hi", has_direct_at=True, has_other_assistant=True)
    )
    assert _decision(trace).action == "trigger"
    assert _decision(trace).reason == Reason.TRIGGER


def test_hard_trigger_beats_backoff():
    trace = _gate().evaluate(
        _snap("hi", has_direct_at=True, idle_streak=4, previous_end_reason=IDLE_END),
    )
    assert _decision(trace).action == "trigger"
    assert _backoff(trace).applied is False
    assert _backoff(trace).bypass_reason == BYPASS_HARD_TRIGGER


def test_hard_trigger_traces_everything():
    # All features still run and are traced; nothing short-circuits.
    trace = _gate().evaluate(_snap("在吗", has_direct_at=True))
    assert [c.feature for c in trace.contributions] == [
        "relevance", "content", "pressure", "presence", "frequency",
    ]
    assert _decision(trace).score >= 100.0


# ── Relevance variants ──────────────────────────────────────────────────────

def _relevance(snap: GateSnapshot) -> Contribution:
    trace = _gate().evaluate(snap)
    return next(c for c in trace.contributions if c.feature == "relevance")


def test_relevance_direct_at_100():
    c = _relevance(_snap("hi", has_direct_at=True))
    assert c.op == "max" and c.value == 100.0 and c.reason == "direct_at"


def test_relevance_quote_to_self_100():
    c = _relevance(_snap("hi", has_quote_to_self=True))
    assert c.op == "max" and c.value == 100.0 and c.reason == "quote_to_self"


def test_relevance_name_mention_80():
    c = _relevance(_snap("麦麦 在吗", self_name="麦麦"))
    assert c.value == 80.0 and c.reason == "name_mention"


def test_relevance_private_40():
    c = _relevance(_snap("hi", is_group=False))
    assert c.value == 40.0 and c.reason == "private"


def test_relevance_focus_40():
    c = _relevance(_snap("hi", is_focused=True))
    assert c.value == 40.0 and c.reason == "focus"


def test_relevance_none_0():
    c = _relevance(_snap("hi"))
    assert c.value == 0.0 and c.reason == "none"


def test_relevance_max_semantics_name_beats_private():
    c = _relevance(_snap("麦麦 在吗", self_name="麦麦", is_group=False))
    assert c.value == 80.0  # max(80, 40)


def test_relevance_direct_facts_never_inferred_from_text():
    # Text that LOOKS like a direct @ never overrides a False structured fact.
    c = _relevance(_snap("@bot 在吗", has_direct_at=False, self_name="bot"))
    assert c.value == 0.0


def test_relevance_name_mention_requires_self_name():
    c = _relevance(_snap("麦麦 在吗", self_name=None))
    assert c.value == 0.0


def test_relevance_name_mention_uses_normalized_text():
    # A visible @ is stripped, so it cannot fabricate a name mention.
    c = _relevance(_snap("@麦麦 在吗", self_name="麦麦", has_direct_at=False))
    assert c.value == 0.0


# ── Content: strip-derived cases ────────────────────────────────────────────

def _content(snap: GateSnapshot) -> Contribution:
    trace = _gate().evaluate(snap)
    return next(c for c in trace.contributions if c.feature == "content")


def test_content_question_15():
    c = _content(_snap("在吗"))
    assert c.op == "add" and c.value == 15.0


def test_content_direct_request_20():
    c = _content(_snap("帮我查一下天气"))
    assert c.value == 20.0


def test_content_opinion_solicit_20():
    c = _content(_snap("给点意见"))
    assert c.value == 20.0


def test_content_length_bonus_exclusive_tiers():
    assert _content(_snap("x" * 39)).value == 0.0
    assert _content(_snap("x" * 40)).value == 5.0
    assert _content(_snap("x" * 119)).value == 5.0
    assert _content(_snap("x" * 120)).value == 10.0
    assert _content(_snap("x" * 150)).value == 10.0  # exclusive: never 15


def test_content_all_short_reaction_batch_minus_25():
    c = _content(_snap("哈哈", "嗯嗯", "好的"))
    assert c.value == -25.0


def test_content_strip_derived_request():
    # Quote prefix and media placeholder strip away; the request survives.
    c = _content(_snap("「张三: 在吗」\n帮我查一下[图片]"))
    assert c.value == 20.0


def test_content_normalizes_to_empty_gives_nothing():
    c = _content(_snap("「张三: 在吗」"))
    assert c.value == 0.0


def test_content_combined_signals_and_length():
    # question + request + opinion + 120+ length bonus = 15+20+20+10.
    c = _content(_snap("在吗 帮我查一下 这个怎么样 " + "x" * 120))
    assert c.value == 65.0


def test_content_short_batch_and_length_both_apply():
    # 5 × 8-char laughter = 40 code points: +5 length, −25 short batch.
    c = _content(_snap("哈哈哈哈哈哈哈哈", "哈哈哈哈哈哈哈哈", "哈哈哈哈哈哈哈哈",
                       "哈哈哈哈哈哈哈哈", "哈哈哈哈哈哈哈哈"))
    assert c.value == -20.0


# ── Pressure: exact threshold curve and idle bonus ──────────────────────────

def _pressure(snap: GateSnapshot) -> Contribution:
    trace = _gate().evaluate(snap)
    return next(c for c in trace.contributions if c.feature == "pressure")


def test_pressure_zero_pending_zero():
    c = _pressure(_snap(pending=0, pending_messages=()))
    assert c.value == 0.0


def test_pressure_at_threshold_is_50():
    c = _pressure(_snap(*(["x"] * 8)))
    assert c.value == 50.0


def test_pressure_r2_is_72():
    c = _pressure(_snap(*(["x"] * 16)))
    assert c.value == 72.0


def test_pressure_caps_at_100():
    c = _pressure(_snap(*(["x"] * 40)))
    assert c.value == 100.0
    c = _pressure(_snap(*(["x"] * 100)))
    assert c.value == 100.0


def test_pressure_idle_bonus_requires_pending_avg_and_idle():
    # pending > 0, avg > 0, idle >= avg → +15.
    c = _pressure(_snap(*(["x"] * 20), idle_seconds=60.0, recent_average_interval=60.0))
    assert c.value == 93.0  # 78 + 15
    # idle just below the average: no bonus.
    c = _pressure(_snap(*(["x"] * 20), idle_seconds=59.999, recent_average_interval=60.0))
    assert c.value == 78.0
    # zero pending: no bonus even with infinite idle.
    c = _pressure(_snap(pending=0, pending_messages=(), idle_seconds=math.inf,
                        recent_average_interval=60.0))
    assert c.value == 0.0
    # empty/unavailable average: no bonus.
    c = _pressure(_snap(*(["x"] * 20), idle_seconds=60.0, recent_average_interval=0.0))
    assert c.value == 78.0


def test_pressure_bonus_capped_at_100():
    c = _pressure(_snap(*(["x"] * 40), idle_seconds=60.0, recent_average_interval=60.0))
    assert c.value == 100.0  # 100 + 15 still capped at 100


def test_pressure_nonpositive_threshold_fails_closed():
    trace = _gate().evaluate(_snap("hi", threshold=0))
    assert _decision(trace).action == "delay"
    assert _decision(trace).reason == Reason.FEATURE_FAILURE
    pressure = next(c for c in trace.contributions if c.feature == "pressure")
    assert pressure.error is not None


# ── Presence: signed additive endpoints ─────────────────────────────────────

def _presence(snap: GateSnapshot) -> Contribution:
    trace = _gate().evaluate(snap)
    return next(c for c in trace.contributions if c.feature == "presence")


def test_presence_endpoints_exact():
    assert _presence(_snap("hi", self_ratio=0.25)).value == 0.0
    assert _presence(_snap("hi", self_ratio=0.60)).value == -25.0
    assert _presence(_snap("hi", self_ratio=0.10)).value == 0.0
    assert _presence(_snap("hi", self_ratio=0.80)).value == -25.0


def test_presence_linear_midpoint():
    c = _presence(_snap("hi", self_ratio=0.425))  # midpoint of 0.25..0.60
    assert c.value == pytest.approx(-12.5)


def test_presence_is_signed_additive():
    # The penalty is a NEGATIVE add, never subtracted separately.
    c = _presence(_snap("hello world", self_ratio=0.60))
    assert c.op == "add" and c.value == -25.0
    # composition: content 0 + pressure 1 + presence −25 = −24.
    trace = _gate().evaluate(_snap("hello world", self_ratio=0.60))
    assert trace.aggregates["add"] == pytest.approx(-24.0)


# ── Composition and registration order ──────────────────────────────────────

def test_compose_empty_is_zero_max_zero_add_unit_scale():
    assert compose(()) == (0.0, 0.0, 1.0, 0.0)


def test_compose_negative_max_floored_at_zero():
    assert compose((_contribution("m", "max", -5.0),))[0] == 0.0


def test_compose_sums_adds_and_products_scales():
    contribs = (
        _contribution("a", "add", 10.0),
        _contribution("b", "add", -3.0),
        _contribution("s1", "scale", 0.5),
        _contribution("s2", "scale", 2.0),
    )
    m, a, s, score = compose(contribs)
    assert (m, a, s) == (0.0, 7.0, 1.0)
    assert score == 7.0


def test_compose_rounds_and_floors():
    assert compose((_contribution("a", "add", 78.4),))[3] == 78.0
    assert compose((_contribution("a", "add", -10.0),))[3] == 0.0


def test_compose_skips_error_contributions():
    contribs = (
        _contribution("ok", "add", 10.0),
        Contribution(feature="bad", op="add", value=999.0, error="boom"),
    )
    assert compose(contribs)[1] == 10.0


def test_default_features_registry_shape():
    reg = default_features()
    assert isinstance(reg, Registry)
    assert reg.names() == ("relevance", "content", "pressure", "presence", "frequency")
    assert len(reg) == 5


def test_builtins_register_like_third_party_features():
    reg = default_features()

    class Custom:
        name = "custom"

        def contribute(self, ctx):
            return Contribution(feature="custom", op="add", value=5.0)

    reg.register(Custom())
    assert reg.names() == (
        "relevance", "content", "pressure", "presence", "frequency", "custom",
    )
    # replace=True shadows a builtin in place, keeping its slot.
    reg.register(Custom(), replace=True, name="relevance")
    assert reg.names()[0] == "relevance"
    shadowed = reg.get("relevance")
    assert shadowed is not None and shadowed.name == "custom"


def test_registration_order_does_not_change_score():
    class Plus10:
        name = "plus10"

        def contribute(self, ctx):
            return Contribution(feature="plus10", op="add", value=10.0)

    class Scale2:
        name = "scale2"

        def contribute(self, ctx):
            return Contribution(feature="scale2", op="scale", value=2.0)

    builtins = list(default_features().all())
    forward = [Plus10(), Scale2(), *builtins]
    backward = list(reversed(forward))
    snap = _snap("在吗")  # question +15, relevance 0, pressure 1, presence 0
    ta = _gate(forward).evaluate(snap)
    tb = _gate(backward).evaluate(snap)
    assert _decision(ta).score == _decision(tb).score == 52.0  # (0+26)·2
    assert [c.feature for c in ta.contributions] != [
        c.feature for c in tb.contributions
    ]
    assert [c.feature for c in ta.contributions] == [
        "plus10", "scale2", "relevance", "content", "pressure", "presence", "frequency",
    ]


def test_gate_reads_registry_live():
    reg = default_features()
    gate = _gate(reg)
    before = _decision(gate.evaluate(_snap("在吗"))).score

    class Plus50:
        name = "plus50"

        def contribute(self, ctx):
            return Contribution(feature="plus50", op="add", value=50.0)

    reg.register(Plus50())
    after = _decision(gate.evaluate(_snap("在吗"))).score
    assert after == before + 50.0


def test_trace_order_is_registry_order():
    trace = _gate().evaluate(_snap("hi"))
    assert [c.feature for c in trace.contributions] == [
        "relevance", "content", "pressure", "presence", "frequency",
    ]


# ── Feature errors fail closed ──────────────────────────────────────────────

class Boom:
    name = "boom"

    def contribute(self, ctx):
        raise RuntimeError("boom")


class NanAdd:
    name = "nan_add"

    def contribute(self, ctx):
        return Contribution(feature="nan_add", op="add", value=math.nan)


class NegativeScale:
    name = "neg_scale"

    def contribute(self, ctx):
        return Contribution(feature="neg_scale", op="scale", value=-1.0)


class BadOp:
    name = "bad_op"

    def contribute(self, ctx):
        return Contribution(feature="bad_op", op="bogus", value=1.0)


class NotAContribution:
    name = "not_a_contribution"

    def contribute(self, ctx):
        return 42


class Abstain:
    name = "abstain"

    def contribute(self, ctx):
        return None


def test_feature_exception_traced_and_delays():
    trace = _gate([Boom()]).evaluate(_snap("hi"))
    assert _decision(trace).action == "delay"
    assert _decision(trace).reason == Reason.FEATURE_FAILURE
    assert _decision(trace).delay_seconds is None
    errors = [c for c in trace.contributions if c.error is not None]
    assert len(errors) == 1
    assert errors[0].feature == "boom"
    assert errors[0].op == "error"


def test_invalid_contributions_traced_and_delay():
    for bad in (NanAdd(), NegativeScale(), BadOp()):
        trace = _gate([bad]).evaluate(_snap("hi"))
        assert _decision(trace).action == "delay", bad.name
        assert _decision(trace).reason == Reason.FEATURE_FAILURE, bad.name
        traced = next(c for c in trace.contributions if c.feature == bad.name)
        assert traced.error is not None, bad.name


def test_non_contribution_return_traced_and_delays():
    trace = _gate([NotAContribution()]).evaluate(_snap("hi"))
    assert _decision(trace).reason == Reason.FEATURE_FAILURE
    traced = next(c for c in trace.contributions if c.feature == "not_a_contribution")
    assert traced.error is not None


def test_abstaining_feature_is_not_traced_and_does_not_delay():
    trace = _gate([Abstain()]).evaluate(_snap("hi"))
    assert _decision(trace).reason == Reason.DELAY  # normal score delay
    assert all(c.feature != "abstain" for c in trace.contributions)


def test_error_contributions_excluded_from_aggregation():
    trace = _gate([Boom(), _Plus10()]).evaluate(_snap("hi"))
    assert trace.aggregates["add"] == 10.0  # boom's 0.0 never aggregates


class _Plus10:
    name = "plus10"

    def contribute(self, ctx):
        return Contribution(feature="plus10", op="add", value=10.0)


def test_feature_failure_precedes_backoff():
    trace = _gate([Boom()]).evaluate(
        _snap("hi", idle_streak=4, previous_end_reason=IDLE_END)
    )
    assert _decision(trace).reason == Reason.FEATURE_FAILURE
    assert _backoff(trace).applied is True  # still recorded, but not decisive


# ── Refusal precedence ──────────────────────────────────────────────────────

def test_refusal_skips_even_when_score_would_trigger():
    trace = _gate().evaluate(
        _snap(*(["x"] * 21), has_other_assistant=True)
    )
    assert _decision(trace).score == 80.0  # computed and traced
    assert _decision(trace).action == "skip"
    assert _decision(trace).reason == Reason.REFUSAL
    assert _decision(trace).delay_seconds is None  # skip carries no delay


def test_refusal_precedes_feature_error():
    trace = _gate([Boom()]).evaluate(_snap("hi", has_other_assistant=True))
    assert _decision(trace).reason == Reason.REFUSAL
    assert _decision(trace).action == "skip"
    assert any(c.error is not None for c in trace.contributions)  # still traced


def test_refusal_precedes_backoff():
    trace = _gate().evaluate(
        _snap("hi", has_other_assistant=True, idle_streak=4,
              previous_end_reason=IDLE_END)
    )
    assert _decision(trace).reason == Reason.REFUSAL
    assert _decision(trace).action == "skip"
    # The backoff still applies and is recorded — it is just not decisive.
    assert _backoff(trace).applied is True
    assert _backoff(trace).seconds == 60.0


def test_refusal_precedes_active_hold():
    # The frozen precedence: refusal is next after hard triggers, before
    # backoff — an active durable hold does not outrank it.
    trace = _gate().evaluate(
        _snap("hi", has_other_assistant=True, hold_until=600.0)
    )
    assert _decision(trace).action == "skip"
    assert _decision(trace).reason == Reason.REFUSAL
    # The hold is still recorded as the in-force backoff outcome.
    assert _backoff(trace).applied is True
    assert _backoff(trace).seconds == 300.0


# ── Active durable hold: remaining duration only ────────────────────────────

def test_active_hold_delays_for_remaining_duration():
    trace = _gate().evaluate(_snap("hi", hold_until=600.0))
    assert _decision(trace).action == "delay"
    assert _decision(trace).reason == Reason.BACKOFF
    assert _decision(trace).delay_seconds == 300.0  # 600 − evaluated 300
    assert _backoff(trace).applied is True
    assert _backoff(trace).seconds == 300.0


def test_active_hold_never_regenerates_fresh_hold_from_stale_history():
    # Stale history (idle streak 4 + idle end reason) would compute a fresh
    # 60 s backoff — the active hold must return ONLY its remaining
    # duration, never a fresh hold.
    trace = _gate().evaluate(
        _snap("hi", idle_streak=4, previous_end_reason=IDLE_END, hold_until=600.0)
    )
    assert _decision(trace).action == "delay"
    assert _decision(trace).reason == Reason.BACKOFF
    assert _decision(trace).delay_seconds == 300.0  # not 60.0
    assert _backoff(trace).seconds == 300.0


def test_active_hold_beats_mode_selection():
    # Score 146 >= 80 would trigger, but the durable hold delays first.
    snap = _snap("麦麦 在吗 帮我查一下 这个怎么样 " + "x" * 120,
                 self_name="麦麦", hold_until=600.0)
    trace = _gate().evaluate(snap)
    assert _decision(trace).score >= 80.0
    assert _decision(trace).action == "delay"
    assert _decision(trace).reason == Reason.BACKOFF
    assert _decision(trace).delay_seconds == 300.0


def test_feature_failure_precedes_active_hold():
    # Feature failures delay safely BEFORE backoff (frozen precedence); the
    # hold is the materialized backoff, so the failure's reason wins.
    trace = _gate([Boom()]).evaluate(_snap("hi", hold_until=600.0))
    assert _decision(trace).action == "delay"
    assert _decision(trace).reason == Reason.FEATURE_FAILURE
    assert _decision(trace).delay_seconds is None
    # The hold is still recorded as the in-force backoff outcome.
    assert _backoff(trace).applied is True
    assert _backoff(trace).seconds == 300.0


def test_hard_trigger_beats_active_hold():
    trace = _gate().evaluate(
        _snap("hi", has_direct_at=True, hold_until=600.0)
    )
    assert _decision(trace).action == "trigger"
    assert _decision(trace).reason == Reason.TRIGGER


def test_expired_hold_is_ignored():
    # hold_until == evaluated_ts is NOT active: the expired hold is
    # ignored and MUST NOT regenerate a fresh hold/delay from the previous
    # end reason or idle streak (stale history would compute 60 s).
    trace = _gate().evaluate(
        _snap("hi", idle_streak=4, previous_end_reason=IDLE_END, hold_until=300.0)
    )
    assert _backoff(trace).applied is False
    assert _backoff(trace).bypass_reason == "expired_hold"
    # Mode selection decides instead: score 16 < 80 → plain delay, timed
    # until the idle bonus activates (avg 60 − idle 30).
    assert _decision(trace).action == "delay"
    assert _decision(trace).reason == Reason.DELAY
    assert _decision(trace).delay_seconds == 30.0


def test_expired_hold_ignored_even_without_stale_history():
    # An expired hold with no idle history at all is still ignored: the
    # controller is never consulted, so nothing can be regenerated.
    trace = _gate().evaluate(_snap("hi", hold_until=300.0))
    assert _backoff(trace).applied is False
    assert _backoff(trace).bypass_reason == "expired_hold"
    assert _decision(trace).reason == Reason.DELAY


def test_active_hold_bypassed_by_focus():
    # Focus bypasses an ACTIVE durable hold: the hold's remaining duration
    # is NOT the outcome — mode selection decides.
    trace = _gate().evaluate(
        _snap("hi", is_focused=True, hold_until=600.0, idle_streak=4,
              previous_end_reason=IDLE_END)
    )
    assert _backoff(trace).applied is False
    assert _backoff(trace).bypass_reason == BYPASS_FOCUS
    assert _decision(trace).reason == Reason.DELAY  # mode selection decides


def test_active_hold_bypassed_by_high_pending():
    # pending >= threshold bypasses an ACTIVE durable hold.
    trace = _gate().evaluate(
        _snap(*(["x"] * 8), hold_until=600.0, idle_streak=4,
              previous_end_reason=IDLE_END)
    )
    assert _backoff(trace).applied is False
    assert _backoff(trace).bypass_reason == BYPASS_HIGH_PENDING
    assert _decision(trace).reason == Reason.DELAY  # mode selection decides


# ── reply_necessity mode: trigger / timed / event-only delay ────────────────

def test_reply_necessity_triggers_at_trigger_score():
    trace = _gate().evaluate(_snap(*(["x"] * 21)))
    assert _decision(trace).action == "trigger"
    assert _decision(trace).reason == Reason.TRIGGER


def test_reply_delay_timed_until_idle_bonus_activation():
    # Score 78 < 80; idle 30 < avg 60 → timed until the bonus activates.
    trace = _gate().evaluate(
        _snap(*(["x"] * 20), idle_seconds=30.0, recent_average_interval=60.0)
    )
    assert _decision(trace).action == "delay"
    assert _decision(trace).delay_seconds == 30.0  # avg − idle


def test_reply_delay_event_only_when_bonus_already_active():
    # Bonus active (idle 60 >= avg 60) but score 35 still < 80: nothing to
    # wait for → event-only.
    trace = _gate().evaluate(
        _snap(*(["x"] * 5), idle_seconds=60.0, recent_average_interval=60.0)
    )
    assert _decision(trace).action == "delay"
    assert _decision(trace).delay_seconds is None


def test_reply_delay_event_only_when_pending_zero():
    trace = _gate().evaluate(
        _snap(pending=0, pending_messages=(), idle_seconds=30.0,
              recent_average_interval=60.0)
    )
    assert _decision(trace).action == "delay"
    assert _decision(trace).delay_seconds is None


def test_reply_delay_event_only_when_avg_unavailable():
    for avg in (0.0, None):
        trace = _gate().evaluate(
            _snap(*(["x"] * 5), idle_seconds=30.0, recent_average_interval=avg)
        )
        assert _decision(trace).action == "delay"
        assert _decision(trace).delay_seconds is None


# ── frequency mode: virtual math, no hybrid triggering ──────────────────────

def _freq(*texts: str, **kw: Any) -> GateSnapshot:
    kw.setdefault("mode", "frequency")
    return _snap(*texts, **kw)


def test_frequency_triggers_at_pending_threshold():
    trace = _gate().evaluate(_freq(*(["x"] * 8)))
    assert _decision(trace).action == "trigger"
    assert _decision(trace).reason == Reason.MODE


def test_frequency_virtual_messages_math():
    # idle 180 / avg 60 = 3 virtual; 5 + 3 = 8 >= threshold → trigger.
    trace = _gate().evaluate(
        _freq(*(["x"] * 5), idle_seconds=180.0, recent_average_interval=60.0)
    )
    assert _decision(trace).action == "trigger"
    assert _decision(trace).reason == Reason.MODE


def test_frequency_virtual_capped_at_threshold_minus_one():
    # idle 100000 / avg 60 caps at 7; 1 + 7 = 8 → trigger, never more.
    trace = _gate().evaluate(
        _freq("x", idle_seconds=100000.0, recent_average_interval=60.0)
    )
    assert _decision(trace).action == "trigger"


def test_frequency_empty_pending_never_triggers():
    # virtual caps at threshold − 1, so silence can never speak first.
    trace = _gate().evaluate(
        _freq(pending=0, pending_messages=(), idle_seconds=math.inf,
              recent_average_interval=60.0)
    )
    assert _decision(trace).action == "delay"
    assert _decision(trace).delay_seconds is None  # unreachable → event-only


def test_frequency_no_hybrid_score_triggering():
    # Score 146 >= 80 (name mention + content), but pending + virtual <
    # threshold: contributions are diagnostic only, the mode decides.
    snap = _freq("麦麦 在吗 帮我查一下 这个怎么样 " + "x" * 120,
                 self_name="麦麦", idle_seconds=0.0, recent_average_interval=60.0)
    trace = _gate().evaluate(snap)
    assert _decision(trace).score >= 80.0
    assert _decision(trace).action == "delay"
    assert _decision(trace).reason == Reason.DELAY


def test_frequency_delay_timed_until_trigger_reachable():
    # pending 5, avg 60, idle 120: 60·(8−5) − 120 = 60 s until virtual
    # reaches 3 and the total hits the threshold.
    trace = _gate().evaluate(
        _freq(*(["x"] * 5), idle_seconds=120.0, recent_average_interval=60.0)
    )
    assert _decision(trace).action == "delay"
    assert _decision(trace).delay_seconds == 60.0


def test_frequency_unavailable_avg_event_only():
    for avg in (0.0, None):
        trace = _gate().evaluate(
            _freq(*(["x"] * 5), idle_seconds=180.0, recent_average_interval=avg)
        )
        assert _decision(trace).action == "delay"
        assert _decision(trace).delay_seconds is None
    # pending alone still triggers at the threshold without an average.
    trace = _gate().evaluate(
        _freq(*(["x"] * 8), recent_average_interval=0.0)
    )
    assert _decision(trace).action == "trigger"


def test_frequency_contributions_still_traced():
    trace = _gate().evaluate(_freq("在吗"))
    assert [c.feature for c in trace.contributions] == [
        "relevance", "content", "pressure", "presence", "frequency",
    ]
    assert trace.aggregates["score"] == _decision(trace).score


# ── Idle backoff: bypasses and precedence ───────────────────────────────────

def test_backoff_applies_group_idle():
    trace = _gate().evaluate(
        _snap("hi", idle_streak=4, previous_end_reason=IDLE_END)
    )
    assert _decision(trace).action == "delay"
    assert _decision(trace).reason == Reason.BACKOFF
    assert _decision(trace).delay_seconds == 60.0
    assert _backoff(trace).applied is True
    assert _backoff(trace).seconds == 60.0


def test_backoff_bypassed_by_focus():
    trace = _gate().evaluate(
        _snap("hi", idle_streak=4, is_focused=True, previous_end_reason=IDLE_END)
    )
    assert _backoff(trace).bypass_reason == BYPASS_FOCUS
    assert _decision(trace).reason == Reason.DELAY  # mode selection decides


def test_backoff_bypassed_by_high_pending():
    trace = _gate().evaluate(
        _snap(*(["x"] * 8), idle_streak=4, previous_end_reason=IDLE_END)
    )
    assert _backoff(trace).bypass_reason == BYPASS_HIGH_PENDING
    # Mode selection decides instead: pressure 50 < 80 → plain delay.
    assert _decision(trace).action == "delay"
    assert _decision(trace).reason == Reason.DELAY


def test_backoff_bypassed_in_private():
    trace = _gate().evaluate(
        _snap("hi", idle_streak=4, is_group=False, previous_end_reason=IDLE_END)
    )
    assert _backoff(trace).bypass_reason == BYPASS_NOT_GROUP


def test_backoff_bypassed_by_non_idle_end_reason():
    trace = _gate().evaluate(
        _snap("hi", idle_streak=4, previous_end_reason="trigger")
    )
    assert _backoff(trace).bypass_reason == BYPASS_NON_IDLE


def test_backoff_precedes_mode_choice():
    # Score 146 >= 80 would trigger, but the group idle backoff delays first.
    snap = _snap("麦麦 在吗 帮我查一下 这个怎么样 " + "x" * 120,
                 self_name="麦麦", idle_streak=4, previous_end_reason=IDLE_END)
    trace = _gate().evaluate(snap)
    assert _decision(trace).score >= 80.0
    assert _decision(trace).action == "delay"
    assert _decision(trace).reason == Reason.BACKOFF


def test_backoff_below_start_count_not_applied():
    trace = _gate().evaluate(
        _snap("hi", idle_streak=1, previous_end_reason=IDLE_END)
    )
    assert _backoff(trace).applied is False
    assert _backoff(trace).bypass_reason is None
    assert _decision(trace).reason == Reason.DELAY


def test_backoff_configuration_from_snapshot_is_traced():
    # The gate evaluates with the SNAPSHOT's merged backoff config — never
    # a constructor default — and the trace records exactly those facts.
    trace = _gate().evaluate(
        _snap("hi", backoff_base_s=30.0, backoff_cap_s=600.0, backoff_start_count=3)
    )
    assert trace.config["backoff"] == {
        "base_s": 30.0, "cap_s": 600.0, "start_count": 3, "threshold": 8,
    }
    assert trace.snapshot_facts["backoff_base_s"] == 30.0
    assert trace.snapshot_facts["backoff_cap_s"] == 600.0
    assert trace.snapshot_facts["backoff_start_count"] == 3


def test_backoff_applies_snapshot_config_duration():
    # base 30 / start 3: streak 4 → 30 * 2**(4-3) = 60 s (not the default
    # 15-based 120 s).
    trace = _gate().evaluate(
        _snap("hi", idle_streak=4, previous_end_reason=IDLE_END,
              backoff_base_s=30.0, backoff_start_count=3)
    )
    assert _backoff(trace).applied is True
    assert _backoff(trace).seconds == 60.0
    assert _decision(trace).delay_seconds == 60.0


def test_backoff_snapshot_threshold_gates_high_pending_bypass():
    # The snapshot threshold is the controller's high-pending bypass
    # threshold: pending 6 >= threshold 6 bypasses backoff.
    trace = _gate().evaluate(
        _snap(*(["x"] * 6), threshold=6, idle_streak=4, previous_end_reason=IDLE_END)
    )
    assert _backoff(trace).bypass_reason == BYPASS_HIGH_PENDING
    assert _backoff(trace).applied is False


# ── DecisionTrace: replay-friendly facts ────────────────────────────────────

def test_trace_round_trips_through_json():
    trace = _gate().evaluate(
        _snap("在吗", has_direct_at=True, idle_streak=2)
    )
    data = json.loads(json.dumps(dataclasses.asdict(trace)))
    assert data["chat_key"] == CK
    assert data["mode"] == "reply_necessity"
    assert data["threshold"] == 8
    assert data["trigger_score"] == 80
    assert data["pending"] == 1
    assert data["decision"]["action"] == "trigger"
    assert data["decision"]["score"] >= 100.0
    assert data["decision"]["reason"] == "trigger"
    assert data["contributions"][0]["feature"] == "relevance"
    assert data["contributions"][0]["value"] == 100.0
    assert data["aggregates"]["max"] == 100.0
    assert data["backoff"]["applied"] is False
    assert data["ts"] == 300.0


def test_trace_snapshot_facts_carry_bounds_counts_and_facts():
    trace = _gate().evaluate(
        _snap("hi", has_direct_at=True, has_quote_to_self=False,
              has_other_assistant=True, self_name="麦麦", hold_until=600.0,
              idle_streak=3, previous_end_reason=IDLE_END,
              window_count=5, self_count=2, last_nonself_ts=250.0)
    )
    facts = trace.snapshot_facts
    assert facts["start_msg_id"] == 1
    assert facts["through_msg_id"] == 9
    assert facts["evaluated_ts"] == 300.0
    assert facts["window_count"] == 5
    assert facts["self_count"] == 2
    assert facts["last_nonself_ts"] == 250.0
    assert facts["idle_seconds"] == 30.0
    assert facts["recent_average_interval"] == 60.0
    assert facts["self_ratio"] == 0.1
    assert facts["is_group"] is True
    assert facts["is_focused"] is False
    assert facts["self_name"] == "麦麦"
    assert facts["has_direct_at"] is True
    assert facts["has_quote_to_self"] is False
    assert facts["has_other_assistant"] is True
    assert facts["hold_until"] == 600.0
    assert facts["idle_streak"] == 3
    assert facts["previous_end_reason"] == IDLE_END


def test_trace_config_is_exact_configuration():
    trace = _gate().evaluate(
        _snap("hi", mode="frequency", threshold=12, trigger_score=90, frequency=0.6)
    )
    assert trace.config == {
        "mode": "frequency",
        "threshold": 12,
        "trigger_score": 90,
        "frequency": 0.6,
        "backoff": {"base_s": 15.0, "cap_s": 300.0, "start_count": 2, "threshold": 12},
    }


def test_trace_aggregates_and_backoff_facts():
    trace = _gate().evaluate(_snap("在吗"))
    assert trace.aggregates == {"max": 0.0, "add": 16.0, "scale": 1.0, "score": 16.0}
    assert isinstance(_backoff(trace), BackoffFacts)
    assert _backoff(trace).applied is False
    assert isinstance(trace, DecisionTrace)


def test_trace_records_error_contributions():
    trace = _gate([Boom()]).evaluate(_snap("hi"))
    data = json.loads(json.dumps(dataclasses.asdict(trace)))
    assert data["contributions"][0]["error"] is not None
    assert data["decision"]["reason"] == "feature_failure"


# ── Gate input validation ───────────────────────────────────────────────────

def test_gate_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        _gate().evaluate(_snap("hi", mode="sometimes"))


def test_gate_rejects_non_snapshot():
    with pytest.raises(TypeError):
        _gate().evaluate("not a snapshot")  # type: ignore[arg-type]


def test_gate_rejects_feature_without_name():
    class NoName:
        def contribute(self, ctx):
            return None

    with pytest.raises(TypeError, match="name"):
        _gate([NoName()])


def test_gate_rejects_feature_without_contribute():
    class NoMethod:
        name = "no_method"

    with pytest.raises(TypeError, match="contribute"):
        _gate([NoMethod()])


# ── Built-in feature classes are individually usable ────────────────────────

def test_builtin_features_are_plain_gate_features():
    for feature in (RelevanceFeature(), ContentFeature(), PressureFeature(),
                    PresenceFeature(), FrequencyScaleFeature()):
        assert isinstance(feature.name, str)
        assert callable(feature.contribute)
        assert isinstance(feature, GateFeature)  # runtime_checkable protocol


def test_frequency_scale_formula():
    snap = _snap("hi", frequency=1.0)
    c = FrequencyScaleFeature().contribute(snap)
    assert c.op == "scale" and c.value == 1.0
    assert FrequencyScaleFeature().contribute(_snap("hi", frequency=0.6)).value == 0.8
    assert FrequencyScaleFeature().contribute(_snap("hi", frequency=0.0)).value == 0.5
    # frequency −1 is a legal zero scale; below that the scale is invalid.
    assert FrequencyScaleFeature().contribute(_snap("hi", frequency=-1.0)).value == 0.0
    trace = _gate().evaluate(_snap("hi", frequency=-1.0))
    assert trace.aggregates["scale"] == 0.0
    assert _decision(trace).score == 0.0
    bad = _gate().evaluate(_snap("hi", frequency=-2.0))
    assert _decision(bad).reason == Reason.FEATURE_FAILURE  # negative scale

# ── the clamped-idle runaway ────────────────────────────────────────────────
#
# ``assemble_snapshot`` pins ``idle_seconds`` to the 300 s presence window when
# the window holds no non-self message. That is a LOWER BOUND, not the real
# idle time. Arithmetic that treats it as the truth never converges: production
# wrote 3,660 timer dispatches over 12.8 h, one every 11.6 s, each one
# re-deriving the identical delay from the identical clamped value.

def test_clamped_idle_never_schedules_a_timed_reply_delay():
    """The exact production runaway: pending=7, threshold=8, avg=311.4375 s,
    an empty presence window. ``avg - 300`` is a positive constant every
    evaluation reproduces, so the chat re-armed forever."""
    snap = _snap(
        *["a", "b", "c", "d", "e", "f", "g"],
        threshold=8,
        trigger_score=80,
        last_nonself_ts=None,
        idle_seconds=300.0,
        recent_average_interval=311.4375,
        window_count=0,
        self_count=0,
        self_ratio=0.0,
        is_group=True,
    )
    decision = _gate().evaluate(snap).decision
    assert decision.action == "delay"
    assert decision.delay_seconds is None, "must be event-only, not a re-armed timer"


def test_clamped_idle_never_schedules_a_timed_frequency_delay():
    snap = _snap(
        *["a", "b"],
        mode="frequency",
        threshold=8,
        last_nonself_ts=None,
        idle_seconds=300.0,
        recent_average_interval=311.4375,
        window_count=0,
        self_count=0,
        self_ratio=0.0,
    )
    decision = _gate().evaluate(snap).decision
    assert decision.action == "delay"
    assert decision.delay_seconds is None


def test_measured_idle_still_schedules_a_timed_delay():
    """The guard is narrow: a real measurement keeps the timed delay."""
    snap = _snap(
        "hi",
        threshold=8,
        trigger_score=80,
        last_nonself_ts=250.0,
        idle_seconds=30.0,
        recent_average_interval=60.0,
    )
    decision = _gate().evaluate(snap).decision
    assert decision.action == "delay"
    assert decision.delay_seconds == pytest.approx(30.0)


def test_clamped_idle_grants_the_pressure_idle_bonus():
    """An empty window means the chat HAS gone quiet for at least the window.
    Comparing the floor against a larger average withheld the +15 bonus from
    exactly the lull it exists to detect."""
    snap = _snap(
        *["a", "b", "c", "d", "e", "f", "g"],
        threshold=8,
        last_nonself_ts=None,
        idle_seconds=300.0,
        recent_average_interval=311.4375,
        window_count=0,
        self_count=0,
        self_ratio=0.0,
    )
    (pressure,) = [c for c in _gate().evaluate(snap).contributions if c.feature == "pressure"]
    assert "idle_bonus" in pressure.reason
    assert pressure.value == pytest.approx(53.0)


def test_reserved_profile_speaks_up_after_a_lull():
    """The deployed profile (threshold 8, trigger_score 60): eight unanswered
    messages plus the idle bonus is enough to join in unprompted."""
    snap = _snap(
        *["a", "b", "c", "d", "e", "f", "g", "h"],
        threshold=8,
        trigger_score=60,
        last_nonself_ts=None,
        idle_seconds=300.0,
        recent_average_interval=311.4375,
        window_count=0,
        self_count=0,
        self_ratio=0.0,
    )
    decision = _gate().evaluate(snap).decision
    assert decision.action == "trigger"
    assert decision.score >= 60


def test_alias_names_score_as_a_name_mention():
    """MaiBot's ``bot.alias_names``: a group that has settled on a nickname
    must not be met with silence because the config says something else."""
    from pretender.gate import _name_mentioned

    snap = _snapshot(
        pending_messages=(_msg("bp在吗"),), self_name="bp", self_aliases=("麦麦",)
    )
    assert _name_mentioned(snap)
    snap = _snapshot(
        pending_messages=(_msg("麦麦在吗"),), self_name="bp", self_aliases=("麦麦",)
    )
    assert _name_mentioned(snap)
    # Without the alias configured, the other name is just text.
    snap = _snapshot(pending_messages=(_msg("麦麦在吗"),), self_name="bp")
    assert not _name_mentioned(snap)
