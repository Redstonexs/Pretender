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

import random
import re

from pretender.output.kaomoji import detect_kaomoji_spans
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

# Stage directions. LLMs narrate themselves — ``（笑）``, ``(叹气)``,
# ``【思考中】`` — and a group member who writes their own stage directions
# reads as fiction, not as a person. MaiBot strips these in
# ``process_llm_response_segments`` with ``[(\[（](?=.*[一-鿿]).*?[)\]）]``;
# that lookahead scans the WHOLE remaining text, so ``(hello)`` is stripped
# merely because Chinese appears somewhere later. This requires the Chinese to
# be INSIDE the brackets, which is what the rule is actually for. Kaomoji are
# excluded separately (they are bracketed punctuation, not prose).
_STAGE_DIRECTION_RE = re.compile(r"[(\[（【][^)\]）】\n]*[一-鿿][^)\]）】\n]*[)\]）】]")

#: Patterns whose matches are never shielded by a kaomoji span. Leaked tool
#: blocks and CQ codes are bracketed runs containing symbols, so they match
#: the kaomoji pattern too — shielding them would let ``[tool_call]调用[/tool_call]``
#: reach a real chat. Only the stage-direction rule consults kaomoji spans.
REMOVE_PATTERNS = (
    _CQ_RE,
    _THINKING_RE,
    _TOOL_BLOCK_RE,
    _TOOL_BRACKET_RE,
    _TOOL_MARKER_RE,
    _UNSAFE_TAG_RE,
    _LEAK_LINE_RE,
)


#: What to send when sanitizing removed everything. MaiBot's ``"呃呃"``.
EMPTY_FALLBACK = "呃呃"

#: MaiBot's ``_get_random_default_reply``, minus the two entries that
#: interpolate the bot nickname (this stage has no access to it).
DEFAULT_REPLIES = ("不知道哦", "不知道", "不晓得", "懒得说", "()")


def western_ratio(text: str) -> float:
    """Fraction of alphanumeric characters that are ASCII letters.

    MaiBot's ``get_western_ratio``: below 0.1 the text is treated as Chinese
    prose, which is the only case the length guard applies to.
    """
    alnum = [c for c in text if c.isalnum()]
    if not alnum:
        return 0.0
    return sum("a" <= c.lower() <= "z" for c in alnum) / len(alnum)


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
    # A kaomoji is bracketed punctuation, so the stage-direction rule would
    # eat one. It is applied separately, against the kaomoji spans as well —
    # never against the other removals, which must not be shieldable.
    faces = detect_kaomoji_spans(text)
    removals = _merge_intervals(
        removals
        + [
            m.span()
            for m in _STAGE_DIRECTION_RE.finditer(text)
            if not _overlaps(m.start(), m.end(), spans)
            and not _overlaps(m.start(), m.end(), faces)
        ]
    )
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
    return _merge_intervals(intervals)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals = sorted(intervals)
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
    """OutputStage that strips unsafe constructs and records protected spans.

    Also carries MaiBot's two "this reply is unusable" guards from
    ``process_llm_response_segments``: an empty result becomes ``呃呃``, and a
    Chinese wall of text past ``max_length`` becomes a short noncommittal
    reply instead of a wall the bot would never have typed.
    """

    name = "sanitize"
    order = 10

    def __init__(self, max_length: int = 512) -> None:
        #: MaiBot compares against ``max_length * 2``.
        self.max_length = max_length

    def apply(self, out: Outgoing) -> Outgoing:
        # Recompute rather than trusting offsets left by an earlier/plugin
        # stage: a plugin may have changed the text since those offsets were
        # recorded, and stale spans must never shield reintroduced leakage.
        out_text_before = out.text
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
        out.text, final_spans = self._guard(cleaned, final_spans, out, had_text=bool(out_text_before.strip()), has_media=bool(out.segments))
        set_protected_spans(out, final_spans)
        return out

    def _guard(
        self,
        text: str,
        spans: list[tuple[int, int]],
        out: Outgoing,
        *,
        had_text: bool,
        has_media: bool,
    ) -> tuple[str, list[tuple[int, int]]]:
        """MaiBot's post-clean fallbacks.

        The empty fallback only applies when cleaning CONSUMED real text — a
        reply that was ``（这里全是动作描写）`` and nothing else. A message that
        legitimately carries no text (an image or sticker send, where the
        payload is in ``segments``) must stay empty.
        """
        from pretender.output.typo import _seed

        if not text.strip():
            if had_text and not has_media:
                return EMPTY_FALLBACK, []
            return text, spans
        if (
            self.max_length > 0
            and len(text) > self.max_length * 2
            and western_ratio(text) < 0.1
        ):
            rng = random.Random(_seed(out))
            return DEFAULT_REPLIES[rng.randrange(len(DEFAULT_REPLIES))], []
        return text, spans
