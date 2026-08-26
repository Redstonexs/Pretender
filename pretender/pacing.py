"""Dependency-neutral pacing primitives (PLAN.md §1.B).

This is the LOW-LEVEL home of the pure EWMA message-interval reducer. It
imports nothing from the runtime (session), the storage (repo), or the
types layer: both the runtime ``Session`` and the repository's atomic
ingest path consume the SAME reducer here, so the durable
``chats.avg_interval`` and any in-memory session average can never drift
apart.

The reducer is a pure function over plain floats: the caller owns the
state (a ``ChatState`` in the session layer, a ``chats`` row in the
repository) and decides when to persist. No timestamps are produced here —
callers pass absolute epoch seconds.
"""

from __future__ import annotations

# EWMA smoothing factor for the message-interval average.
EWMA_ALPHA = 0.5


def ewma_interval(
    prev_avg: float | None,
    prev_ts: float | None,
    now: float,
    *,
    alpha: float = EWMA_ALPHA,
) -> float | None:
    """The next EWMA inter-message interval, or None when the sample
    carries no pacing information.

    ``prev_avg`` is the prior average (None before the first sample),
    ``prev_ts`` the timestamp of the prior message (None when there is no
    prior sample), and ``now`` the timestamp of the new message. A missing
    prior timestamp or a non-positive gap (clock skew, same-timestamp
    batches) carries no pacing information and is ignored — it would drag
    the average toward 0. The average is always positive once seeded, so
    idle compensation (``idle_seconds / recent_average_interval``) never
    divides by zero.
    """
    if prev_ts is None:
        return None
    gap = now - prev_ts
    if gap <= 0:
        return None
    if prev_avg is None:
        return gap
    return alpha * gap + (1.0 - alpha) * prev_avg