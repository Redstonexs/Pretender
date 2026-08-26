"""Effect learner: code-owned bounded delta derivation (Phase 6).

The effect learner's model output carries ONLY ``categorization`` and
``confidence`` — the model never sets a weight delta. The bounded delta is
derived HERE, deterministically, from the frozen bands (PLAN.md §D):

- ``adopted``  → +0.5 … +1.0
- ``partial``  → +0.1 … +0.35
- ``rejected`` → −1.0 … −0.4

``confidence`` (a number in [0, 1]) scales linearly within the band. The
result is always within the band, so a downstream ``apply_record_feedback``
call (which clamps to [-1, 1]) can never be out of range.
"""

from __future__ import annotations

__all__ = [
    "EFFECT_CATEGORIZATIONS",
    "EFFECT_BANDS",
    "derive_effect_delta",
]

EFFECT_CATEGORIZATIONS: tuple[str, ...] = ("adopted", "partial", "rejected")

# Frozen code-owned bands: categorization -> (low, high) delta bounds.
EFFECT_BANDS: dict[str, tuple[float, float]] = {
    "adopted": (0.5, 1.0),
    "partial": (0.1, 0.35),
    "rejected": (-1.0, -0.4),
}


def derive_effect_delta(categorization: str, confidence: float) -> float:
    """The deterministic bounded delta for one effect judgment.

    ``categorization`` must be one of ``EFFECT_CATEGORIZATIONS`` and
    ``confidence`` a finite number in [0, 1]; the returned delta is always
    within the categorization's frozen band. The model never supplies the
    delta itself.
    """
    if categorization not in EFFECT_BANDS:
        raise ValueError(
            f"categorization must be one of {EFFECT_CATEGORIZATIONS},"
            f" got {categorization!r}"
        )
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"confidence must be a number, got {confidence!r}")
    confidence = float(confidence)
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence must be in [0, 1], got {confidence!r}")
    low, high = EFFECT_BANDS[categorization]
    return round(low + (high - low) * confidence, 4)