"""Ordered output pipeline: sanitize, split, and Chinese typo.

Public surface for the output lane: the pipeline runner plus the two
built-in stages and their pure functions.
"""

from __future__ import annotations

from pretender.output.pipeline import (
    OutputPipeline,
    detect_protected_spans,
    stable_group_id,
)
from pretender.output.sanitize import SanitizeStage, sanitize_text
from pretender.output.split import SplitStage, split_text
from pretender.output.typo import TypoStage, load_frequency, typo_text

__all__ = [
    "OutputPipeline",
    "SanitizeStage",
    "SplitStage",
    "TypoStage",
    "detect_protected_spans",
    "sanitize_text",
    "split_text",
    "typo_text",
    "load_frequency",
    "stable_group_id",
]
