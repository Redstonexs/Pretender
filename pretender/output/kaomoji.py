"""Kaomoji detection — ``(╯°□°)╯``, ``(・ω・)``, ``╥╯╰╥`` and friends.

A kaomoji is a face built out of punctuation, so every stage that reasons
about punctuation will happily destroy one: split sees the parens and commas
as cut points, typo sees nothing it recognises and may still walk through it.

MaiBot solves this by substituting placeholders before post-processing and
restoring them afterwards
(``utils.py::protect_kaomoji`` / ``recover_kaomoji``). Pretender reports
*spans* instead and feeds them to the existing no-cut / no-mutate machinery
both stages already honour. Same protection, no placeholder round-trip — and
therefore no way for a leaked ``__KAOMOJI_0__`` to reach a real chat, which is
the failure mode the substitution approach carries.

The pattern is MaiBot's, with one addition: platform markup that happens to
look like a bracketed face (``[CQ:at,qq=1]`` matches their regex, because
``:`` is a non-CJK non-alphanumeric character inside brackets) is excluded, so
sanitize can still strip it.
"""

from __future__ import annotations

import re

#: MaiBot's kaomoji pattern: either a bracketed run containing at least one
#: character that is not CJK / ASCII-alphanumeric / whitespace, or a run of
#: 2–15 characters drawn from the face-parts set.
_KAOMOJI_RE = re.compile(
    r"(?:"
    r"[(\[（【]"  # opening bracket
    r"[^()\[\]（）【】]*?"  # lazy, no nested brackets
    r"[^一-龥a-zA-Z0-9\s]"  # at least one symbol character
    r"[^()\[\]（）【】]*?"
    r"[)\]）】]"  # closing bracket
    r")"
    r"|"
    r"(?:[▼▽・ᴥω･﹏^><≧≦￣｀´∀ヮДд︿﹀へ｡ﾟ╥╯╰︶︹•⁄]{2,15})"
)

#: Bracketed constructs that are platform markup, not faces. These must stay
#: visible to the sanitize stage, which exists to remove them.
_NOT_A_FACE_RE = re.compile(r"\[\s*(?:CQ|表情包|图片|回复)\s*[:：]?", re.IGNORECASE)


def detect_kaomoji_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of ``text`` holding a kaomoji, sorted and non-overlapping."""
    spans: list[tuple[int, int]] = []
    for m in _KAOMOJI_RE.finditer(text):
        if _NOT_A_FACE_RE.match(m.group(0)):
            continue
        spans.append(m.span())
    return spans
