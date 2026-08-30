"""Deterministic rendering helpers for the learner lane (Phase 6).

Everything here is PURE: no repository, LLM, or clock access. The helpers
build the prompt surface from a source batch and from selected records:

- ``escape_untrusted`` neutralizes characters that could break the
  untrusted-data wrapper (the closing delimiters and code-fence markers), so
  chat text can never smuggle instructions or fence-break the model output.
- ``render_batch`` renders one ``LearnerBatch`` as an opaque-ref message
  list: each message is ``[N] <escaped text>`` where ``N`` is the opaque
  per-batch ref the model references in its output. The refs are opaque —
  they only make sense within this batch — and the model's ``source_id`` /
  ``source_ids`` fields are validated against exactly this numbering.
- ``render_records`` renders selected records as an escaped reference list
  (the effect learner's ``{{references}}`` surface and future injection).
- ``source_hash`` / ``canonical_content`` are the deterministic identity
  helpers: ``source_hash`` matches the repository's canonical source-batch
  hash byte-for-byte, and ``canonical_content`` is the sorted-key JSON
  rendering that the repository's record content hash is computed from.
  The pipeline never invents hashes (the repository owns them), but these
  helpers let tests and future wiring verify determinism.

The agent is NOT injected here: nothing in this module touches the agent
lane, and ``select_records`` is only a bounded selection helper for future
reference rendering.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any

import orjson

from pretender.seams import AdaptiveRepository
from pretender.types import ChatKey, LearnerBatch, Record

__all__ = [
    "escape_untrusted",
    "render_batch",
    "render_attributed_batch",
    "render_records",
    "select_records",
    "source_hash",
    "canonical_content",
]

# The untrusted-data wrapper delimiters. ``escape_untrusted`` neutralizes
# any occurrence of the closing delimiters inside chat text so a message can
# never close its own wrapper early.
_OPEN_MESSAGE = "<message"
_CLOSE_MESSAGE = "</message>"
_FENCE = "```"
_CLOSE_TAG = re.compile(r"</[A-Za-z][A-Za-z0-9:_-]*>")

# Keys the model is never allowed to set on a record payload (the repository
# owns identity/weight/uses/retirement; the code owns effect deltas).
FORBIDDEN_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "weight",
        "uses",
        "delta",
        "score_delta",
        "content_hash",
        "id",
        "created_ts",
        "retired",
        "source_first_msg_id",
        "source_last_msg_id",
    }
)


def escape_untrusted(text: str) -> str:
    """Neutralize characters that could break the untrusted-data wrapper.

    Backslashes are doubled first (so the later replacements cannot be
    un-escaped), then the closing wrapper delimiters and code-fence markers
    are rewritten to inert forms. The result is safe to embed inside the
    ``<message>`` / ``<record>`` data wrappers.
    """
    if not isinstance(text, str):
        raise TypeError(f"escape_untrusted expects str, got {type(text).__name__}")
    text = text.replace("\\", "\\\\")
    # Escape every XML-ish closing tag, not just the two original wrappers:
    # adaptive context also uses slot wrappers (and records may be rendered
    # in reply prompts).  The slash form remains readable to the model while
    # it cannot terminate a surrounding wrapper.
    text = _CLOSE_TAG.sub(lambda match: "<\\/" + match.group(0)[2:], text)
    text = text.replace(_FENCE, "`\\`\\`")
    return text


def render_batch(batch: LearnerBatch) -> str:
    """Render one source batch as an opaque-ref message list.

    Each message renders as ``[N] <escaped text>`` where ``N`` is the opaque
    per-batch ref (1-based index into ``batch.texts``). The model references
    messages ONLY by these refs; the validators check every ref against this
    exact numbering.
    """
    parts: list[str] = []
    for i, text in enumerate(batch.texts, start=1):
        parts.append(f"[{i}] {escape_untrusted(text)}")
    return "\n".join(parts)


def render_attributed_batch(batch: LearnerBatch) -> str:
    """Render one source batch as an opaque-ref list that names the speaker.

    Each message renders as ``[N] <escaped name>: <escaped text>`` — the same
    opaque refs ``render_batch`` uses, with the display name attached. The
    impression learner needs it: you cannot form an impression OF someone
    from an anonymous wall of text. The platform UID is deliberately NOT
    rendered; the model only ever answers with a ref, and the code resolves
    that ref to the real identity through ``batch.senders``.

    Falls back to the plain rendering when the batch carries no names (a
    hand-built batch), so it is always safe to call.
    """
    names = batch.sender_names or ("",) * len(batch.texts)
    parts: list[str] = []
    for i, (text, name) in enumerate(zip(batch.texts, names, strict=False), start=1):
        who = escape_untrusted(name).strip()
        body = escape_untrusted(text)
        parts.append(f"[{i}] {who}: {body}" if who else f"[{i}] {body}")
    return "\n".join(parts)


def render_records(records: Iterable[Record]) -> str:
    """Render selected records as an escaped reference list for prompts.

    Each record renders as ``<record ref="N"> <escaped body> </record>``
    where the body is the payload's ``text`` field when present (a non-empty
    string), else the canonical JSON rendering. Used by the effect learner's
    ``{{references}}`` surface and by future injection wiring.
    """
    parts: list[str] = []
    for i, rec in enumerate(records, start=1):
        payload = rec.payload if isinstance(rec.payload, dict) else {}
        text = payload.get("text")
        body = text if isinstance(text, str) and text.strip() else canonical_content(payload)
        parts.append(f'<record ref="{i}">\n{escape_untrusted(body)}\n</record>')
    return "\n".join(parts)


async def select_records(
    repo: AdaptiveRepository,
    chat_key: ChatKey,
    learner: str,
    *,
    limit: int = 10,
) -> list[Record]:
    """Bounded selection of a learner's records for reference rendering.

    A thin wrapper over ``repo.select_learner_records`` (highest-weight,
    least-used first, excluding legacy/retired). The agent is NOT injected
    here — this is the selection helper future wiring uses.
    """
    return await repo.select_learner_records(chat_key, learner, limit=limit)


def source_hash(texts: Iterable[str]) -> str:
    """Deterministic hash of a source batch's message texts.

    Matches the repository's canonical ``_source_hash`` byte-for-byte
    (sha256 over each text followed by a NUL separator), so a batch read
    from the repository always verifies against this helper.
    """
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def canonical_content(payload: dict[str, Any]) -> str:
    """Deterministic canonical JSON rendering of a record payload.

    Sorted-key JSON (the same rendering the repository's record content hash
    is computed from), so identical payloads always render identically.
    """
    return orjson.dumps(
        payload, default=str, option=orjson.OPT_SORT_KEYS
    ).decode("utf-8")
