"""Split stage: punctuation-aware Chinese splitting into at most ``max_split``
parts with stable group/seq/pacing metadata, honouring protected spans.

The stage rewrites ``Outgoing.parts`` (one reply → several messages), sets a
stable content-derived ``group_id`` (matching the outbox's scheme so a retried
finish regroups identically), and records per-part relative pacing in
``platform_ref["part_pacing"]``. ``seq`` is implicit: the outbox conversion
assigns part index as ``seq`` when there is more than one part.

Splitting is punctuation-aware (sentence-ending 。！？!?；;) and never cuts
inside a protected span. It guarantees no empty parts and never exceeds
``max_split``; when a text has no sentence boundaries it hard-splits by
length (also avoiding protected spans) so no single part is pathologically
long.
"""

from __future__ import annotations

import math

from pretender.output.pipeline import (
    _in_span,
    detect_protected_spans,
    get_protected_spans,
    stable_group_id,
)
from pretender.types import Outgoing

_SENTENCE_END = "。！？!?；;"


def split_text(
    text: str,
    max_split: int = 3,
    protected_spans: list[tuple[int, int]] | tuple = (),
) -> list[str]:
    """Split ``text`` into at most ``max_split`` non-empty parts.

    ``protected_spans`` are character offsets on ``text`` that are never cut.
    """
    text = text.strip()
    if not text:
        return []
    if max_split <= 1:
        return [text]
    spans = list(protected_spans) or detect_protected_spans(text)
    boundaries = _sentence_boundaries(text, spans)
    if not boundaries:
        return _hard_split(text, max_split, spans)
    if len(boundaries) <= max_split - 1:
        return _split_at(text, boundaries)
    chosen = _pick_balanced(boundaries, max_split - 1, len(text))
    return _split_at(text, chosen)


def _sentence_boundaries(
    text: str, spans: list[tuple[int, int]]
) -> list[int]:
    """Indices just after sentence-ending punctuation, skipping protected
    spans."""
    boundaries: list[int] = []
    for i, ch in enumerate(text):
        if ch in _SENTENCE_END:
            pos = i + 1
            if pos < len(text) and not _in_span(pos, spans):
                boundaries.append(pos)
    return boundaries


def _split_at(text: str, boundaries: list[int]) -> list[str]:
    parts: list[str] = []
    start = 0
    for b in boundaries:
        part = text[start:b].strip()
        if part:
            parts.append(part)
        start = b
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _pick_balanced(boundaries: list[int], k: int, total_len: int) -> list[int]:
    """Pick ``k`` boundaries that yield roughly equal-length parts: each is
    the boundary closest to its ideal cumulative position."""
    if k <= 0:
        return []
    if k >= len(boundaries):
        return list(boundaries)
    ideal = total_len / (k + 1)
    chosen: list[int] = []
    for i in range(k):
        target = ideal * (i + 1)
        best = min(boundaries, key=lambda b: abs(b - target))
        if best not in chosen:
            chosen.append(best)
    if len(chosen) < k:
        for b in boundaries:
            if b not in chosen:
                chosen.append(b)
                if len(chosen) == k:
                    break
    chosen.sort()
    return chosen


def _hard_split(
    text: str, max_split: int, spans: list[tuple[int, int]]
) -> list[str]:
    """Length-based split for text with no sentence boundaries, avoiding
    cutting inside protected spans."""
    if len(text) <= max_split:
        return [text]
    part_len = math.ceil(len(text) / max_split)
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + part_len, len(text))
        for s, e in spans:
            if s < end < e:
                end = e
                break
        part = text[start:end].strip()
        if part:
            parts.append(part)
        start = end
    return parts


def part_delays(n: int) -> list[float]:
    """Human-ish relative pacing (seconds) per part, cumulative from the base
    ``send_after_ts``: part 0 sends immediately, each later part a bit later."""
    return [round(i * 1.5, 2) for i in range(n)]


class SplitStage:
    """OutputStage that splits one reply into several messages with stable
    group/seq/pacing metadata."""

    name = "split"
    order = 20

    def __init__(self, max_split: int = 3) -> None:
        self.max_split = max_split

    def apply(self, out: Outgoing) -> Outgoing:
        if out.parts is not None:
            return out  # already split
        if out.segments:
            # Text splitting cannot safely construct per-part media/reply
            # payloads. Keep segmented output atomic instead of leaving a
            # downstream converter to silently discard split metadata.
            return out
        spans = get_protected_spans(out)
        parts = split_text(out.text, self.max_split, spans)
        if len(parts) <= 1:
            return out  # nothing to split
        out.parts = parts
        out.group_id = stable_group_id(parts)
        out.platform_ref["part_pacing"] = part_delays(len(parts))
        return out
