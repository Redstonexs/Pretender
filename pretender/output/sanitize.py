"""Sanitize stage: strip leakage / CQ codes / tool-analysis / unsafe markup
without corrupting ordinary text, URLs, or protected spans.

The stage is the mandatory final boundary in the pipeline. It detects protected spans (URLs,
code blocks, inline code, blockquotes) on the incoming text, removes the
unsafe constructs below while skipping anything that overlaps a protected
span, then records the FINAL protected spans on the Outgoing so later stages
(split, and eventually typo) never touch them.

Removal patterns are deliberately narrow so ordinary prose — punctuation,
URLs, math comparisons, quoted content — is never corrupted.
"""

from __future__ import annotations

import re

from pretender.output.pipeline import (
    _overlaps,
    detect_protected_spans,
    get_protected_spans,
    set_protected_spans,
)
from pretender.types import Outgoing

# ── removal patterns ────────────────────────────────────────────────────────
# CQ codes: OneBot v11 markup injection like [CQ:at,qq=123].
_CQ_RE = re.compile(r"\[CQ:[^\]]*\]")
# Reasoning / chain-of-thought leakage blocks. Case-insensitive, tolerant of
# whitespace inside the tag (`< thinking >`), and matches an unterminated
# opening tag (no closing tag) through end-of-text so a leaked block is never
# left half-visible.
_TAG_OPEN = r"<\s*(?:thinking|analysis|reasoning)\s*>"
_TAG_CLOSE = r"</\s*(?:thinking|analysis|reasoning)\s*>"
_THINKING_RE = re.compile(
    rf"{_TAG_OPEN}.*?(?:{_TAG_CLOSE}|$)", re.DOTALL | re.IGNORECASE
)
# Tool-call / tool-result blocks (angle-bracket and bracket forms), likewise
# case-insensitive / whitespace-tolerant / untermination-aware.
_TOOL_OPEN = r"<\s*(?:tool_call|tool_result)\s*>"
_TOOL_CLOSE = r"</\s*(?:tool_call|tool_result)\s*>"
_TOOL_BLOCK_RE = re.compile(
    rf"{_TOOL_OPEN}.*?(?:{_TOOL_CLOSE}|$)", re.DOTALL | re.IGNORECASE
)
_TOOL_BRACKET_RE = re.compile(
    r"\[\s*(?:tool_call|tool_result)\s*\].*?"
    r"(?:\[/\s*(?:tool_call|tool_result)\s*\]|$)",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_MARKER_RE = re.compile(r"\[/?\s*tool_call\s*\]|\[/?\s*tool_result\s*\]", re.IGNORECASE)
# Unsafe markup: script / style / iframe blocks (case-insensitive, and an
# unterminated opening tag is removed through end-of-text).
_UNSAFE_TAG_RE = re.compile(
    r"<\s*(?:script|style|iframe)\b.*?(?:</\s*(?:script|style|iframe)\s*>|$)",
    re.DOTALL | re.IGNORECASE,
)
# Internal-reasoning line leakage (clearly-internal markers only). Consumes
# the trailing newline so a removed first line leaves no blank line behind.
_LEAK_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:我的思考|我的分析|内部思考|推理过程|我的推理|系统提示)[:：][^\n]*(?:\n|$)"
)

REMOVE_PATTERNS = (
    _CQ_RE,
    _THINKING_RE,
    _TOOL_BLOCK_RE,
    _TOOL_BRACKET_RE,
    _TOOL_MARKER_RE,
    _UNSAFE_TAG_RE,
    _LEAK_LINE_RE,
)


def sanitize_text(
    text: str, protected_spans: list[tuple[int, int]] | tuple = ()
) -> tuple[str, list[tuple[int, int]]]:
    """Return ``(cleaned_text, final_protected_spans)``.

    ``protected_spans`` (character offsets on the ORIGINAL text) are never
    removed; the returned spans are recomputed on the cleaned text for
    downstream stages.
    """
    spans = list(protected_spans) or detect_protected_spans(text)
    removals = _collect_removals(text, spans)
    cleaned = _apply_removals(text, removals)
    final_spans = detect_protected_spans(cleaned)
    return cleaned, final_spans


def _collect_removals(
    text: str, spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """All removal intervals (on the original text) that do not overlap a
    protected span, merged so overlapping patterns remove once."""
    intervals: list[tuple[int, int]] = []
    for pat in REMOVE_PATTERNS:
        for m in pat.finditer(text):
            s, e = m.span()
            if _overlaps(s, e, spans):
                continue
            intervals.append((s, e))
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _apply_removals(text: str, removals: list[tuple[int, int]]) -> str:
    if not removals:
        return text
    result = text
    for s, e in reversed(removals):
        result = result[:s] + result[e:]
    return result


class SanitizeStage:
    """OutputStage that strips unsafe constructs and records protected spans."""

    name = "sanitize"
    order = 10

    def apply(self, out: Outgoing) -> Outgoing:
        # Recompute rather than trusting offsets left by an earlier/plugin
        # stage: a plugin may have changed the text since those offsets were
        # recorded, and stale spans must never shield reintroduced leakage.
        spans = detect_protected_spans(out.text)
        cleaned, final_spans = sanitize_text(out.text, spans)
        out.text = cleaned
        # Sanitize already-created parts too (defence in depth: a plugin may
        # pre-populate parts before the pipeline runs). Each part is cleaned
        # against its own protected spans; the recorded spans stay those of
        # the full text for downstream stages.
        if out.parts:
            out.parts = [
                sanitize_text(part, detect_protected_spans(part))[0]
                for part in out.parts
            ]
        set_protected_spans(out, final_spans)
        return out
