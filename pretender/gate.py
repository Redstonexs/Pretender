"""The pure turn gate (PLAN.md §1.B; frozen gate spec).

Zero LLM cost, zero I/O: one immutable claim-bounded ``GateSnapshot`` in,
one replay-friendly ``DecisionTrace`` out. The score is nothing but the
composition of registered ``GateFeature`` contributions — the five built-ins
register exactly like a third-party one (``default_features()``), and every
contribution lands in the trace.

Frozen semantics implemented here:

- **Composition.** ``max`` contributions aggregate as ``max(all maxes, 0)``,
  ``add`` as their sum, ``scale`` as their product (empty product is 1.0);
  the final score is ``max(0, round((M + A) * S))``. Aggregation is
  commutative, so registration order never changes the score; the trace
  still lists contributions in registry order.
- **The five built-ins.** ``relevance`` (max: direct @ 100, quote-to-self
  100, name mention 80, private 40, focus 40, else 0 — direct-address facts
  come from the snapshot's structured fields, never from visible text);
  ``content`` (add: +15 question, +20 direct request, +20 opinion solicit,
  exclusive length bonus +5/+10, −25 only when the WHOLE batch is short
  reactions); ``pressure`` (add: the exact threshold curve, +15 idle bonus
  only when pending > 0 and avg > 0 and idle ≥ avg, capped at 100);
  ``presence`` (add, signed 0..−25: 0 below self-ratio 0.25, linear to −25
  at 0.60); ``frequency`` (scale: ``0.5 + 0.5 * config.frequency``).
- **Precedence.** A hard direct @ / quote-to-self evaluates and traces
  EVERYTHING, then triggers with score ≥ 100 — beating refusal, scales,
  backoff, and feature errors. Otherwise: other-assistant refusal is a
  terminal SKIP (outranking feature errors and backoff); feature errors
  delay safely; an ACTIVE durable hold (``hold_until > evaluated_ts``)
  delays only for its remaining duration — the backoff controller is never
  consulted while a hold is active, so a fresh hold can never be
  regenerated from stale history — unless focus or high pending
  (``pending >= threshold``) bypasses it; an EXPIRED durable hold is
  ignored entirely and never regenerates a hold/delay from the previous
  end reason or idle streak; then group-only idle backoff; then mode
  selection.
- **Per-chat backoff configuration.** The idle-backoff controller is built
  per evaluation from the snapshot's merged config facts
  (``backoff_base_s``/``backoff_cap_s``/``backoff_start_count`` plus the
  snapshot ``threshold``) — never from a constructor default, so two chats
  with different backoff configs evaluate differently.
- **Modes.** ``reply_necessity`` triggers at score ≥ trigger_score, else
  delays — timed only until the idle bonus activates (``avg − idle`` when
  pending > 0, avg > 0, idle < avg), otherwise event-only. ``frequency``
  uses ONLY pending plus capped idle virtual messages
  (``min(max(idle/avg, 0), threshold − 1)``, so silence can never speak
  first); contributions stay diagnostic (no hybrid score triggering). Its
  delay is timed until the trigger becomes reachable
  (``avg·(threshold − pending) − idle``) when pending > 0 and avg > 0,
  else event-only. An empty/unavailable average always gives an event-only
  delay.
- **Fail closed.** A feature exception or an invalid contribution (bad op,
  non-finite value, negative scale) is traced with ``error`` set, excluded
  from aggregation, and delays the cycle — never a trigger unless a hard
  direct @ / quote applies.
- **Reasons.** Stable ``Reason`` tokens: ``trigger`` (hard or score),
  ``mode`` (frequency-mode trigger), ``refusal`` (terminal skip),
  ``feature_failure``, ``backoff`` (controller or active durable hold),
  ``delay``, ``skip`` (refusal).

No scheduler, cycle, or outbox behavior lives here.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, cast

from pretender.backoff import (
    BYPASS_FOCUS,
    BYPASS_HIGH_PENDING,
    IdleBackoffController,
)
from pretender.registry import Registry
from pretender.seams import GateContext, GateFeature
from pretender.signals import analyze_batch, normalize_text
from pretender.types import (
    BackoffFacts,
    Contribution,
    Decision,
    DecisionTrace,
    GateSnapshot,
    Reason,
)

# A durable hold that expired (``hold_until <= evaluated_ts``) is ignored:
# the controller is never consulted, so a fresh hold can never be
# regenerated from the previous end reason or idle streak. Recorded as the
# BackoffFacts bypass reason (stable token, documented in types.BackoffFacts).
BYPASS_EXPIRED_HOLD = "expired_hold"

# ── Built-in features ───────────────────────────────────────────────────────

# Presence endpoints: 0 below 0.25, linear to −25 at 0.60 (frozen spec).
_PRESENCE_LOW = 0.25
_PRESENCE_HIGH = 0.60
_PRESENCE_MAX_PENALTY = 25.0


class RelevanceFeature:
    """max: direct @ 100, quote-to-self 100, name mention 80, private 40,
    focus 40, else 0.

    The direct-address facts are the snapshot's structured fields
    (``has_direct_at`` / ``has_quote_to_self``), never inferred from visible
    text. Name mention is the structured ``self_name`` appearing in any
    normalized pending text.
    """

    name = "relevance"

    def contribute(self, ctx: GateSnapshot) -> Contribution:
        if ctx.has_direct_at:
            value, reason = 100.0, "direct_at"
        elif ctx.has_quote_to_self:
            value, reason = 100.0, "quote_to_self"
        elif _name_mentioned(ctx):
            value, reason = 80.0, "name_mention"
        elif not ctx.is_group:
            value, reason = 40.0, "private"
        elif ctx.is_focused:
            value, reason = 40.0, "focus"
        else:
            value, reason = 0.0, "none"
        return Contribution(feature=self.name, op="max", value=value, reason=reason)


def _name_mentioned(ctx: GateSnapshot) -> bool:
    """True when the bot's name, or any configured alias, appears in a
    normalized pending text (case-insensitive). No name → no mention.

    Aliases are MaiBot's ``bot.alias_names``: a group that has settled on a
    nickname must not be met with silence merely because the config says
    something else.
    """
    names = [n for n in (ctx.self_name, *ctx.self_aliases) if n]
    if not names:
        return False
    folded = [n.casefold() for n in names]
    return any(
        any(name in normalize_text(m.text).casefold() for name in folded)
        for m in ctx.pending_messages
    )


class ContentFeature:
    """add: +15 question, +20 direct request, +20 opinion solicit, the
    exclusive length bonus (+5 at 40+, +10 at 120+), and −25 only when the
    whole batch is short reactions. All signals run on NORMALIZED text."""

    name = "content"

    def contribute(self, ctx: GateSnapshot) -> Contribution:
        sig = analyze_batch(ctx.pending_messages, ctx)
        value = 0.0
        parts: list[str] = []
        if sig.has_question:
            value += 15.0
            parts.append("question")
        if sig.has_direct_request:
            value += 20.0
            parts.append("request")
        if sig.has_opinion_solicit:
            value += 20.0
            parts.append("opinion")
        if sig.length_bonus:
            value += float(sig.length_bonus)
            parts.append(f"length_bonus_{sig.length_bonus}")
        if sig.is_short_reaction_batch:
            value -= 25.0
            parts.append("short_reaction_batch")
        return Contribution(
            feature=self.name,
            op="add",
            value=value,
            reason="+".join(parts) or "none",
        )


class PressureFeature:
    """add: the exact threshold curve, capped at 100.

    ``r = pending / threshold``; ``r ≤ 1`` → ``min(50, round(50·r²))``;
    ``r > 1`` → ``min(100, 50 + round(50·log1p(r−1)/log1p(4)))``. The +15
    idle bonus applies only when pending > 0 and avg > 0 and idle ≥ avg.
    A non-positive threshold is a feature error (fail closed)."""

    name = "pressure"

    def contribute(self, ctx: GateSnapshot) -> Contribution:
        if ctx.threshold <= 0:
            raise ValueError(
                f"pressure requires threshold > 0, got {ctx.threshold!r}"
            )
        r = ctx.pending / ctx.threshold
        if r <= 1.0:
            base = min(50.0, round(50.0 * r * r))
        else:
            base = min(100.0, 50.0 + round(50.0 * math.log1p(r - 1.0) / math.log1p(4.0)))
        bonus = (
            15.0
            if (
                ctx.pending > 0
                and _positive_avg(ctx.recent_average_interval)
                and (
                    ctx.idle_seconds >= ctx.recent_average_interval
                    # A clamped idle means the whole presence window held no
                    # non-self message, so the real idle is at least the
                    # window and the chat HAS gone quiet. Comparing the floor
                    # against a larger average would withhold the bonus from
                    # exactly the lull it exists to detect.
                    or _idle_is_clamped(ctx)
                )
            )
            else 0.0
        )
        value = min(100.0, base + bonus)
        reason = f"curve={base}" + ("+idle_bonus" if bonus else "")
        return Contribution(feature=self.name, op="add", value=value, reason=reason)


class PresenceFeature:
    """add (signed, 0..−25): self-message ratio over the 300 s window.

    Exactly 0 below 0.25, linear to −25 at 0.60, −25 above. Added once,
    never subtracted."""

    name = "presence"

    def contribute(self, ctx: GateSnapshot) -> Contribution:
        ratio = ctx.self_ratio
        if ratio <= _PRESENCE_LOW:
            value, reason = 0.0, "below_0.25"
        elif ratio >= _PRESENCE_HIGH:
            value, reason = -_PRESENCE_MAX_PENALTY, "at_or_above_0.60"
        else:
            value = -_PRESENCE_MAX_PENALTY * (ratio - _PRESENCE_LOW) / (
                _PRESENCE_HIGH - _PRESENCE_LOW
            )
            reason = "linear"
        return Contribution(feature=self.name, op="add", value=value, reason=reason)


class FrequencyScaleFeature:
    """scale: ``0.5 + 0.5 * config.frequency`` (frozen spec)."""

    name = "frequency"

    def contribute(self, ctx: GateSnapshot) -> Contribution:
        value = 0.5 + 0.5 * ctx.frequency
        return Contribution(
            feature=self.name, op="scale", value=value, reason="frequency_scale"
        )


def default_features() -> Registry:
    """The five built-ins in a Registry, registered exactly like third-party
    features (shape-validated against the ``GateFeature`` protocol)."""
    reg: Registry = Registry("gate_features", protocol=GateFeature)
    reg.register(RelevanceFeature())
    reg.register(ContentFeature())
    reg.register(PressureFeature())
    reg.register(PresenceFeature())
    reg.register(FrequencyScaleFeature())
    return reg


# ── Composition ─────────────────────────────────────────────────────────────

_VALID_OPS = ("max", "add", "scale")


def _validate_contribution(c: Contribution) -> str | None:
    """Why ``c`` is invalid, or None when it is a valid contribution.

    ``max``/``add`` values must be finite (signed); ``scale`` values must be
    finite and nonnegative."""
    if c.op not in _VALID_OPS:
        return f"invalid op {c.op!r}; expected 'max', 'add', or 'scale'"
    if isinstance(c.value, bool) or not isinstance(c.value, (int, float)):
        return f"value must be a number, got {type(c.value).__name__}"
    if not math.isfinite(c.value):
        return f"value must be finite, got {c.value!r}"
    if c.op == "scale" and c.value < 0:
        return f"scale value must be nonnegative, got {c.value!r}"
    return None


def compose(contributions: Sequence[Contribution]) -> tuple[float, float, float, float]:
    """Aggregate traced contributions into ``(max_total, add_total,
    scale_total, score)``.

    Error contributions (``error`` set) are excluded — they were already
    traced and fail the cycle closed. ``max_total = max(all maxes, 0)``,
    ``add_total = sum(adds)``, ``scale_total = product(scales)`` (1.0 when
    no scale contributed), and ``score = max(0, round((M + A) * S))``.
    """
    maxes = [c.value for c in contributions if c.error is None and c.op == "max"]
    adds = [c.value for c in contributions if c.error is None and c.op == "add"]
    scales = [c.value for c in contributions if c.error is None and c.op == "scale"]
    # max(all maxes, 0): the zero is a FLOOR, not an empty-default — a
    # negative max contribution never drags the score below zero.
    max_total = max([0.0, *maxes])
    add_total = sum(adds)
    scale_total = math.prod(scales) if scales else 1.0
    raw = (max_total + add_total) * scale_total
    if not math.isfinite(raw):
        raise ValueError(
            "score composition overflowed: "
            f"max={max_total}, add={add_total}, scale={scale_total}"
        )
    return max_total, add_total, scale_total, max(0.0, round(raw))


# ── The gate evaluator ──────────────────────────────────────────────────────

def _idle_is_clamped(snapshot: GateSnapshot) -> bool:
    """Whether ``idle_seconds`` is the window floor rather than a measurement.

    ``assemble_snapshot`` pins ``idle_seconds`` to the presence-window length
    when the window holds no non-self message (``last_nonself_ts is None``) —
    a conservative LOWER BOUND, not the real idle time, which may be hours.
    Arithmetic that treats the floor as the true value produces a constant
    that never converges, so every such consumer must branch on this first.
    """
    return snapshot.last_nonself_ts is None


def _positive_avg(avg: float | None) -> bool:
    """True when the recent average interval is available and positive."""
    return avg is not None and avg > 0


class Gate:
    """Pure gate evaluator: one immutable snapshot in, one DecisionTrace out.

    Holds only the registered features; the idle-backoff policy is built
    per evaluation from the snapshot's merged config facts, so one gate is
    safe to share across chats and cycles with different backoff configs.
    A ``Registry`` is read live at evaluation time (later registrations are
    picked up); a plain iterable is snapshotted at construction.
    """

    def __init__(
        self,
        features: Registry | Iterable[GateFeature] | None = None,
    ) -> None:
        if features is None:
            features = default_features()
        if isinstance(features, Registry):
            self._registry: Registry | None = features
            self._features: tuple[GateFeature, ...] | None = None
        else:
            self._registry = None
            self._features = tuple(features)
        for f in self._current_features():
            if not isinstance(getattr(f, "name", None), str):
                raise TypeError(
                    f"gate feature must expose a 'name' string, got {f!r}"
                )
            if not callable(getattr(f, "contribute", None)):
                raise TypeError(
                    f"gate feature {getattr(f, 'name', f)!r} must implement contribute(ctx)"
                )

    def _current_features(self) -> tuple[GateFeature, ...]:
        if self._registry is not None:
            return self._registry.all()
        assert self._features is not None
        return self._features

    def evaluate(self, snapshot: GateSnapshot) -> DecisionTrace:
        """Evaluate one claim-bounded snapshot.

        History comes ONLY from the snapshot: ``previous_end_reason`` is
        the per-chat latest terminal end reason the durable layer read —
        the gate receives no separate history argument. Every feature runs
        and is traced (hard triggers never short-circuit); feature
        exceptions and invalid contributions are traced and fail the cycle
        closed as a delay. The idle-backoff controller is built from the
        snapshot's merged backoff config facts. An ACTIVE durable hold
        (``hold_until > evaluated_ts``) delays only for its remaining
        duration and never consults the controller, so a fresh hold can
        never be regenerated from stale history — unless focus or high
        pending bypasses it. An EXPIRED durable hold is ignored and never
        regenerates a hold/delay from the previous end reason or idle
        streak.
        """
        if not isinstance(snapshot, GateSnapshot):
            raise TypeError(
                f"gate evaluates a GateSnapshot, got {type(snapshot).__name__}"
            )
        if snapshot.mode not in ("reply_necessity", "frequency"):
            raise ValueError(f"unknown gate mode: {snapshot.mode!r}")

        contributions = self._evaluate_features(snapshot)
        max_total, add_total, scale_total, score = compose(contributions)
        hard_trigger = snapshot.has_direct_at or snapshot.has_quote_to_self
        backoff_facts = self._backoff_facts(snapshot, hard_trigger)
        decision = self._decide(snapshot, score, hard_trigger, backoff_facts, contributions)
        return DecisionTrace(
            chat_key=snapshot.chat_key,
            mode=snapshot.mode,
            threshold=snapshot.threshold,
            trigger_score=snapshot.trigger_score,
            pending=snapshot.pending,
            contributions=contributions,
            decision=decision,
            ts=snapshot.evaluated_ts,
            snapshot_facts=_snapshot_facts(snapshot),
            config=_config_facts(snapshot),
            aggregates={
                "max": max_total,
                "add": add_total,
                "scale": scale_total,
                "score": score,
            },
            backoff=backoff_facts,
        )

    # ── internals ────────────────────────────────────────────────────────────

    def _backoff_facts(
        self, snapshot: GateSnapshot, hard_trigger: bool
    ) -> BackoffFacts:
        """The idle-backoff outcome for this snapshot.

        - An ACTIVE durable hold (``hold_until > evaluated_ts``) delays
          only for its remaining duration — the controller is never
          consulted, so its stale-history computation (previous end reason
          + idle streak) can never mint a fresh hold. Focus or high
          pending (``pending >= threshold``) bypasses the hold: the bypass
          is recorded and mode selection decides.
        - An EXPIRED durable hold (``hold_until <= evaluated_ts``) is
          ignored: the controller is never consulted, so no fresh
          hold/delay can be regenerated from the previous end reason or
          idle streak.
        - Otherwise the controller decides from the snapshot's merged
          backoff config facts.
        """
        hold_until = snapshot.hold_until
        if hold_until is not None and hold_until > snapshot.evaluated_ts:
            if snapshot.is_focused:
                return BackoffFacts(applied=False, bypass_reason=BYPASS_FOCUS)
            if snapshot.pending >= snapshot.threshold:
                return BackoffFacts(applied=False, bypass_reason=BYPASS_HIGH_PENDING)
            return BackoffFacts(
                applied=True, seconds=hold_until - snapshot.evaluated_ts
            )
        if hold_until is not None:
            return BackoffFacts(applied=False, bypass_reason=BYPASS_EXPIRED_HOLD)
        controller = IdleBackoffController(
            base_s=snapshot.backoff_base_s,
            cap_s=snapshot.backoff_cap_s,
            start_count=snapshot.backoff_start_count,
            threshold=snapshot.threshold,
        )
        return controller.evaluate(
            snapshot.idle_streak,
            is_group=snapshot.is_group,
            end_reason=snapshot.previous_end_reason,
            is_focused=snapshot.is_focused,
            pending=snapshot.pending,
            hard_trigger=hard_trigger,
        )

    def _evaluate_features(self, snapshot: GateSnapshot) -> tuple[Contribution, ...]:
        out: list[Contribution] = []
        for feature in self._current_features():
            try:
                contrib = feature.contribute(cast(GateContext, snapshot))
            except Exception as e:  # noqa: BLE001 — traced, fail closed
                out.append(
                    Contribution(
                        feature=feature.name,
                        op="error",
                        value=0.0,
                        error=f"{type(e).__name__}: {e}",
                    )
                )
                continue
            if contrib is None:
                continue  # abstain: no contribution, no trace entry
            if not isinstance(contrib, Contribution):
                out.append(
                    Contribution(
                        feature=feature.name,
                        op="error",
                        value=0.0,
                        error=(
                            f"returned {type(contrib).__name__}, "
                            "expected Contribution or None"
                        ),
                    )
                )
                continue
            problem = _validate_contribution(contrib)
            if problem is not None:
                out.append(
                    Contribution(
                        feature=contrib.feature,
                        op=contrib.op,
                        value=contrib.value,
                        reason=contrib.reason,
                        error=problem,
                    )
                )
                continue
            out.append(contrib)
        return tuple(out)

    def _decide(
        self,
        snapshot: GateSnapshot,
        score: float,
        hard_trigger: bool,
        backoff_facts: BackoffFacts,
        contributions: tuple[Contribution, ...],
    ) -> Decision:
        pending = snapshot.pending
        # 1. Hard direct @ / quote: beats refusal, scaling, backoff, errors.
        if hard_trigger:
            return Decision(
                action="trigger",
                score=max(100.0, score),
                pending=pending,
                reason=Reason.TRIGGER,
            )
        # 2. Other-assistant refusal is a terminal SKIP (outranks feature
        # errors and backoff; the cycle layer consumes the boundary with an
        # empty outbox).
        if snapshot.has_other_assistant:
            return Decision(
                action="skip", score=score, pending=pending, reason=Reason.REFUSAL
            )
        # 3. Feature failures delay safely.
        if any(c.error is not None for c in contributions):
            return Decision(
                action="delay", score=score, pending=pending, reason=Reason.FEATURE_FAILURE
            )
        # 4. Backoff: an ACTIVE durable hold delays ONLY for its remaining
        # duration (never a fresh hold regenerated from stale history), or
        # the group-only idle backoff applies. Focus / high pending bypass
        # both (recorded in the facts; mode selection decides below).
        if backoff_facts.applied:
            return Decision(
                action="delay",
                score=score,
                delay_seconds=backoff_facts.seconds,
                pending=pending,
                reason=Reason.BACKOFF,
            )
        # 5. Mode selection.
        if snapshot.mode == "reply_necessity":
            if score >= snapshot.trigger_score:
                return Decision(
                    action="trigger", score=score, pending=pending, reason=Reason.TRIGGER
                )
            return Decision(
                action="delay",
                score=score,
                delay_seconds=_reply_delay_seconds(snapshot),
                pending=pending,
                reason=Reason.DELAY,
            )
        # frequency mode: ONLY pending plus capped virtual messages decide.
        _, capped_virtual = _virtual_messages(snapshot)
        if pending + capped_virtual >= snapshot.threshold:
            return Decision(
                action="trigger", score=score, pending=pending, reason=Reason.MODE
            )
        return Decision(
            action="delay",
            score=score,
            delay_seconds=_frequency_delay_seconds(snapshot),
            pending=pending,
            reason=Reason.DELAY,
        )


def _virtual_messages(snapshot: GateSnapshot) -> tuple[float, float]:
    """``(virtual, capped_virtual)`` idle compensation for frequency mode.

    ``virtual = idle_seconds / recent_average_interval`` (0 when the average
    is empty/unavailable), hard-capped at ``threshold − 1`` so silence can
    never speak first. Negative idle never reduces the total."""
    avg = snapshot.recent_average_interval
    if not _positive_avg(avg):
        return 0.0, 0.0
    virtual = snapshot.idle_seconds / avg
    cap = max(snapshot.threshold - 1, 0)
    return virtual, min(max(virtual, 0.0), cap)


def _reply_delay_seconds(snapshot: GateSnapshot) -> float | None:
    """reply_necessity delay: timed ONLY until the idle bonus activates
    (``avg - idle`` when pending > 0, avg > 0, idle < avg); otherwise
    event-only.

    A CLAMPED idle is event-only. With ``last_nonself_ts is None`` the idle is
    the window floor, so for any ``avg > window`` the difference is a positive
    constant the next evaluation reproduces exactly — the chat re-arms the
    same wake forever (observed: 3,660 dispatches in 12.8 h at a fixed
    11.4375 s). The bonus is already active in that state anyway; there is
    nothing left to wait for.
    """
    if _idle_is_clamped(snapshot):
        return None
    avg = snapshot.recent_average_interval
    if (
        snapshot.pending > 0
        and _positive_avg(avg)
        and snapshot.idle_seconds < avg
    ):
        return avg - snapshot.idle_seconds
    return None


def _frequency_delay_seconds(snapshot: GateSnapshot) -> float | None:
    """frequency delay: timed until the trigger becomes reachable
    (``avg·(threshold − pending) − idle`` when pending > 0, avg > 0, and
    pending < threshold); otherwise event-only. With zero pending the
    virtual cap makes the trigger unreachable, so the delay is event-only.
    A clamped idle is event-only for the same reason as
    ``_reply_delay_seconds``."""
    if _idle_is_clamped(snapshot):
        return None
    avg = snapshot.recent_average_interval
    if (
        snapshot.pending > 0
        and _positive_avg(avg)
        and snapshot.pending < snapshot.threshold
    ):
        return max(
            0.0, avg * (snapshot.threshold - snapshot.pending) - snapshot.idle_seconds
        )
    return None


# ── Trace facts (JSON-native, replay-safe) ──────────────────────────────────

def _snapshot_facts(s: GateSnapshot) -> dict[str, object]:
    """The claim-bounded snapshot bounds/counts/facts the trace records."""
    return {
        "chat_key": s.chat_key,
        "cycle_id": s.cycle_id,
        "start_msg_id": s.start_msg_id,
        "through_msg_id": s.through_msg_id,
        "evaluated_ts": s.evaluated_ts,
        "self_id": s.self_id,
        "pending": s.pending,
        "window_count": s.window_count,
        "self_count": s.self_count,
        "last_nonself_ts": s.last_nonself_ts,
        "idle_seconds": s.idle_seconds,
        "recent_average_interval": s.recent_average_interval,
        "self_ratio": s.self_ratio,
        "is_group": s.is_group,
        "is_focused": s.is_focused,
        "self_name": s.self_name,
        "has_direct_at": s.has_direct_at,
        "has_quote_to_self": s.has_quote_to_self,
        "has_other_assistant": s.has_other_assistant,
        "hold_until": s.hold_until,
        "idle_streak": s.idle_streak,
        "previous_end_reason": s.previous_end_reason,
        "backoff_base_s": s.backoff_base_s,
        "backoff_cap_s": s.backoff_cap_s,
        "backoff_start_count": s.backoff_start_count,
    }


def _config_facts(s: GateSnapshot) -> dict[str, object]:
    """The exact configuration the gate evaluated (the backoff facts are
    the snapshot's merged per-chat values the controller was built from)."""
    return {
        "mode": s.mode,
        "threshold": s.threshold,
        "trigger_score": s.trigger_score,
        "frequency": s.frequency,
        "backoff": {
            "base_s": s.backoff_base_s,
            "cap_s": s.backoff_cap_s,
            "start_count": s.backoff_start_count,
            "threshold": s.threshold,
        },
    }