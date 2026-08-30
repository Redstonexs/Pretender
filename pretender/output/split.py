"""Split stage: MaiBot's reply splitter, ported.

One reply becomes several QQ messages the way a person types them. The
algorithm is a port of MaiBot's
``src/chat/utils/utils.py::split_into_sentences_w_remove_punctuation``, with
three deliberate additions noted at the bottom.

The shape that matters, and that a length-based splitter cannot imitate:

  * **the separator is consumed.** Cutting at ``，`` drops it. A bubble
    ending in a comma (``我就是麦麦啊，``) is the single most obvious tell
    that a machine did the splitting; no person types that.
  * **``？`` and ``！`` are not separators.** They end a sentence but they
    belong *inside* the bubble, so ``你不知道我吗？`` stays intact. Only
    ``，`` ``,`` ``。`` ``;`` space and newline cut.
  * **the split is probabilistic.** Text is cut at every separator and then
    neighbours are merged back with ``1 - split_strength``: short lines are
    usually one message and only sometimes two. The variability *is* the
    human texture — a deterministic splitter reads as a machine even when
    every individual cut is defensible.

Newlines always cut and never merge across. Cuts are refused inside paired
quotes, next to ``:``/``：`` (so ``12:30`` and ``key: value`` survive),
between alphanumerics separated by a space, and beside a dash.

Pretender adds three things MaiBot does not have:

  * **protected spans** (URLs, code blocks, inline code, blockquotes),
    platform markup (``[CQ:...]``) and kaomoji are no-cut zones. MaiBot can
    and does cut a URL in half, and protects kaomoji only by substituting
    placeholders it then has to restore.
  * **seeded randomness.** MaiBot calls bare ``random``; the outbox here
    needs a retried finish to reproduce the same parts, so the RNG is seeded
    from the durable output identity (``typo._seed``).
  * ``max_split`` is enforced by MaiBot's ``merge_sentences_to_max_count``.

The stage rewrites ``Outgoing.parts``, sets a content-derived ``group_id``
matching the outbox's scheme, and records per-part pacing in
``platform_ref["part_pacing"]`` — derived from how long each bubble would
take to type (MaiBot's ``calculate_typing_time``), so a two-character reply
lands quickly and a long one does not. ``seq`` is implicit: the outbox
conversion assigns part index as ``seq`` when there is more than one part.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

from pretender.output.kaomoji import detect_kaomoji_spans
from pretender.output.pipeline import (
    _in_span,
    detect_protected_spans,
    get_protected_spans,
    stable_group_id,
)
from pretender.types import Outgoing

#: Platform markup that must survive intact until the sanitize stage sees it.
#: Not a protected span (sanitize deliberately rewrites these) — just a
#: no-cut zone for this stage.
_MARKUP_RE = re.compile(r"\[CQ:[^\]]*\]")

#: The cut points. MaiBot's ``separators`` verbatim. Note what is ABSENT:
#: ``？``/``！``/``…``/``～`` are sentence-final but never cut, because the
#: mark belongs to the bubble it terminates.
_SEPARATORS = frozenset({"，", ",", " ", "。", ";", "\n"})

#: Quote pairs. A separator inside an open quote never cuts, so a quoted
#: sentence is delivered whole.
#:
#: Deviation from MaiBot, deliberately: it tracks one ``current_quote_char``
#: and closes only on an EQUAL character, so ``“`` is never closed by ``”``
#: and every separator after the first Chinese open-quote in a reply is
#: shielded for the rest of the text. Asymmetric pairs are mapped properly
#: here; symmetric ASCII quotes keep MaiBot's any-closes-any behaviour.
_QUOTE_PAIRS = {"“": "”", "‘": "’", "「": "」", "『": "』"}
_SYMMETRIC_QUOTES = frozenset({'"', "'"})
_QUOTE_CHARS = (
    frozenset(_QUOTE_PAIRS) | frozenset(_QUOTE_PAIRS.values()) | _SYMMETRIC_QUOTES
)

_COLONS = frozenset({":", "："})
_DASHES = frozenset({"-", "—"})

#: Below this many characters the text is returned as-is (MaiBot's guard).
_MIN_SPLIT_LEN = 3

#: MaiBot's 1-in-100 flourish on a 1–2 character reply: ``在吗`` occasionally
#: arrives as ``在`` then ``吗``. Kept because it is real MaiBot behaviour and
#: is exactly the kind of texture this stage exists to produce.
_TINY_EXPLODE_P = 0.01


def _split_strength(length: int) -> float:
    """MaiBot's length→strength curve. Higher means more, smaller bubbles."""
    if length < 12:
        return 0.2
    if length < 32:
        return 0.6
    return 0.7


def _preprocess(text: str) -> str:
    """Collapse whitespace around newlines so a blank line is one cut, not
    several empty segments."""
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"\n\s*([，,。;\s])", r"\n\1", text)
    text = re.sub(r"([，,。;\s])\s*\n", r"\1\n", text)
    return text


def _quote_mask(text: str) -> list[bool]:
    """``True`` at every index sitting inside an open quote.

    The quote characters themselves are marked ``False`` so a cut may still
    land on the boundary; only the content between them is shielded.
    """
    mask = [False] * len(text)
    opener = ""
    for i, ch in enumerate(text):
        if ch in _QUOTE_CHARS:
            if not opener:
                # A bare closer with nothing open is stray punctuation.
                if ch not in _QUOTE_PAIRS.values():
                    opener = ch
            elif ch == _QUOTE_PAIRS.get(opener) or (
                ch in _SYMMETRIC_QUOTES and opener in _SYMMETRIC_QUOTES
            ):
                opener = ""
        else:
            mask[i] = bool(opener)
    return mask


def _can_split_at(
    text: str, i: int, mask: list[bool], spans: list[tuple[int, int]]
) -> bool:
    """Whether the separator at ``i`` is a legal cut point."""
    if _in_span(i, spans):
        return False  # inside a URL / code block / CQ code
    if mask[i]:
        return False  # inside a quoted passage
    if text[i] == "\n":
        return True  # a newline always cuts
    if i > 0 and text[i - 1] in _COLONS:
        return False
    if i < len(text) - 1 and text[i + 1] in _COLONS:
        return False
    if text[i] == " " and 0 < i < len(text) - 1:
        prev, nxt = text[i - 1], text[i + 1]
        if prev in _DASHES or nxt in _DASHES:
            return False
        # "GPT 4" / "12 34": a space between alphanumerics is spacing, not a
        # sentence break.
        if _is_alnum(prev) and _is_alnum(nxt):
            return False
    return True


def _is_alnum(ch: str) -> bool:
    return ch.isdigit() or ("a" <= ch.lower() <= "z")


def _segment(
    text: str, mask: list[bool], spans: list[tuple[int, int]]
) -> list[tuple[str, str]]:
    """Cut into ``(content, separator)`` pairs at every legal separator."""
    segments: list[tuple[str, str]] = []
    current = ""
    for i, ch in enumerate(text):
        if ch in _SEPARATORS and _can_split_at(text, i, mask, spans):
            if current:
                segments.append((current, ch))
            elif ch in (" ", "\n"):
                # Keep a bare separator so consecutive newlines still act as
                # a boundary for the merge pass.
                segments.append(("", ch))
            current = ""
        else:
            current += ch
    if current:
        segments.append((current, ""))
    return [(c, s) for c, s in segments if c or s]


def _merge(
    segments: list[tuple[str, str]], merge_p: float, rng: random.Random
) -> list[tuple[str, str]]:
    """Probabilistically glue neighbours back together.

    Merging restores the separator that the cut consumed, so a merged pair
    reads as the original sentence. A segment ending in a newline is never
    merged forward — an explicit line break is always a message boundary.
    """
    merged: list[tuple[str, str]] = []
    i = 0
    while i < len(segments):
        content, sep = segments[i]
        if (
            i + 1 < len(segments)
            and content
            and sep != "\n"
            and rng.random() < merge_p
        ):
            next_content, next_sep = segments[i + 1]
            if next_content:
                merged.append((content + sep + next_content, next_sep))
            else:
                merged.append((content, next_sep))
            i += 2
        else:
            merged.append((content, sep))
            i += 1
    return merged


def _finalize(pairs: list[tuple[str, str]]) -> list[str]:
    """Drop the trailing separator from each part and tidy interior newlines.

    A separator shielded from cutting (inside a quote or code block) may
    still sit inside a part; flatten it to a space rather than emitting a
    multi-line bubble.
    """
    out: list[str] = []
    for content, _sep in pairs:
        cleaned = re.sub(r"[^\S\r\n]*[\r\n]+[^\S\r\n]*", " ", content).strip()
        if cleaned:
            out.append(cleaned)
    return out


def merge_to_max_count(
    pairs: list[tuple[str, str]], max_count: int
) -> list[tuple[str, str]]:
    """MaiBot's ``merge_sentences_to_max_count``: fold into at most
    ``max_count`` groups of near-equal size, preserving order.

    Deviation from MaiBot, deliberately: it folds with ``"".join`` on bare
    strings, which silently swallows the separator between two groups and
    yields ``今天天气不错啊我刚出去走了走``. Folding the ``(content, sep)``
    pairs instead restores that separator, exactly as the probabilistic merge
    above already does.
    """
    if max_count <= 0:
        return []
    if len(pairs) <= max_count:
        return pairs
    result: list[tuple[str, str]] = []
    start = 0
    for group in range(max_count):
        remaining_items = len(pairs) - start
        remaining_groups = max_count - group
        size = -(-remaining_items // remaining_groups)  # ceil
        chunk = pairs[start : start + size]
        joined = "".join(content + sep for content, sep in chunk[:-1])
        result.append((joined + chunk[-1][0], chunk[-1][1]))
        start += size
    return result


def split_text(
    text: str,
    max_split: int = 3,
    protected_spans: list[tuple[int, int]] | tuple = (),
    rng: random.Random | None = None,
) -> list[str]:
    """Split ``text`` into at most ``max_split`` non-empty parts.

    ``protected_spans`` are character offsets on ``text`` that are never cut.
    ``rng`` seeds the probabilistic merge; an unseeded ``random.Random`` is
    used when omitted, so callers that need reproducibility must pass one.
    """
    rng = rng if rng is not None else random.Random()
    text = text.strip()
    if not text:
        return []
    if max_split <= 1:
        return [text]

    prepared = _preprocess(text)
    if len(prepared) < _MIN_SPLIT_LEN:
        if rng.random() < _TINY_EXPLODE_P:
            return list(prepared)
        return [prepared]

    # Offsets shift when preprocessing rewrites whitespace, so recompute
    # rather than trusting the caller's spans against a different string.
    if prepared == text and protected_spans:
        spans = list(protected_spans)
    else:
        spans = detect_protected_spans(prepared)
    spans = (
        spans
        + [m.span() for m in _MARKUP_RE.finditer(prepared)]
        + detect_kaomoji_spans(prepared)
    )

    segments = _segment(prepared, _quote_mask(prepared), spans)
    if not segments:
        return [prepared]

    merged = _merge(segments, 1.0 - _split_strength(len(prepared)), rng)
    parts = _finalize(merge_to_max_count(merged, max_split))
    if not parts:
        return [prepared]
    return parts


#: MaiBot's per-character typing costs (``calculate_typing_time``): a Han
#: character takes twice as long to type as anything else.
_CHINESE_CHAR_S = 0.3
_OTHER_CHAR_S = 0.15
#: The keystroke that actually sends the message.
_ENTER_S = 0.3
#: A sticker/emoji message costs a flat second regardless of its length.
_EMOJI_S = 1.0


def typing_time(
    text: str, *, typing_speed: float = 1.0, is_emoji: bool = False
) -> float:
    """Seconds a person would spend typing ``text``.

    A port of MaiBot's ``calculate_typing_time``. ``typing_speed`` is its
    multiplier (0 disables the wait entirely, 1 is human, 2 is slow).

    The lone-Han-character case is MaiBot's: a bubble that is exactly one
    Han character reads as a considered reaction rather than a fast one, so
    it costs triple plus the enter key. Note that MaiBot adds ``_ENTER_S``
    *only* in that branch — its ``return total_time  # 加上回车时间`` comment
    on the normal path does not match its code. The behaviour is ported as
    written, not as commented, so the pacing matches the reference bot.

    One deliberate deviation: MaiBot returns from the lone-character branch
    BEFORE it reads ``typing_speed``, so ``typing_speed = 0`` still waits
    1.2s for a one-character bubble. A knob documented as "send it all at
    once" that quietly does not is a bug, so the zero check comes first here.
    """
    if typing_speed <= 0:
        return 0.0
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    if chinese == 1 and len(text.strip()) == 1:
        return _CHINESE_CHAR_S * 3 + _ENTER_S
    if is_emoji:
        return _EMOJI_S * typing_speed
    total = sum(
        _CHINESE_CHAR_S if "\u4e00" <= ch <= "\u9fff" else _OTHER_CHAR_S
        for ch in text
    )
    return total * typing_speed


def part_delays(parts: Sequence[str], typing_speed: float = 1.0) -> list[float]:
    """Relative pacing (seconds) per part, cumulative from the base
    ``send_after_ts``.

    Part 0 sends immediately — it was already "typed" while the model was
    thinking. Every later part arrives once the person would have finished
    typing it, so a two-character bubble follows quickly and a long one
    takes its time. A flat ladder is the tell this removes: real bubbles do
    not arrive on a metronome.
    """
    delays = [0.0]
    elapsed = 0.0
    for part in parts[1:]:
        elapsed += typing_time(part, typing_speed=typing_speed)
        delays.append(round(elapsed, 2))
    return delays


class SplitStage:
    """OutputStage that splits one reply into several messages with stable
    group/seq/pacing metadata."""

    name = "split"
    order = 20

    def __init__(self, max_split: int = 3, typing_speed: float = 1.0) -> None:
        self.max_split = max_split
        self.typing_speed = typing_speed

    def apply(self, out: Outgoing) -> Outgoing:
        if out.parts is not None:
            return out  # already split
        if out.segments:
            # Text splitting cannot safely construct per-part media/reply
            # payloads. Keep segmented output atomic instead of leaving a
            # downstream converter to silently discard split metadata.
            return out
        # Seeded from the durable output identity: the same reply retried
        # produces the same bubbles, so the outbox regroups identically.
        from pretender.output.typo import _seed

        rng = random.Random(_seed(out))
        spans = get_protected_spans(out)
        parts = split_text(out.text, self.max_split, spans, rng)
        if len(parts) <= 1:
            # Still adopt the single part: the splitter consumes a trailing
            # separator (``好的。`` -> ``好的``), and MaiBot sends that
            # stripped form even when the reply stays one message.
            if parts and parts[0] != out.text:
                out.text = parts[0]
            return out
        out.parts = parts
        out.group_id = stable_group_id(parts)
        out.platform_ref["part_pacing"] = part_delays(parts, self.typing_speed)
        return out
