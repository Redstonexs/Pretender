"""Pure context management for the typed provider transcript.

The four deterministic transcript operations the two-stage agent needs
(PLAN.md §F, Phase 2 frozen invariants):

- ``normalize`` — pair-normalization into the canonical role sequence of
  frozen decision #4: an assistant tool-call turn is followed by exactly
  one ``tool`` message per call id, in call order. Orphan ``tool``
  messages and unanswered calls are removed, never left behind.
- ``fold`` — all-or-nothing per assistant turn: a completed tool turn
  (assistant message + ALL of its ``tool`` results) becomes ONE synthetic
  user message headed ``[已折叠的历史工具调用]`` listing id/name/args/
  result. ``reply``/``wait``/``no_action`` calls are dropped from the
  listing; ``tool_search`` is compressed to its matched tool names so
  deferred-tool activation survives the fold.
- ``trim`` — tool-group aware, system message pinned: when the transcript
  exceeds ``2 * max_context_size`` messages it is cut back to
  ``max_context_size`` by dropping the oldest messages, never splitting an
  assistant tool-call turn from its results and never leaving an orphan
  ``tool`` message first.
- ``apply_image_budget`` — images past ``max_image_num`` become the
  literal placeholder ``[图片]``; the newest images are kept.

``build_context`` composes the four into the pipeline the planner and
replyer consume. Everything here is pure, deterministic and stdlib-only
(``json``/``re``/``dataclasses``); no LLM calls, no I/O, no mutation of
inputs.

Image convention: an image inside a transcript message's ``content`` is a
markdown span ``![alt](url)`` (urls must not contain ``)``). The budget
replaces all but the newest ``max_image_num`` spans with ``[图片]``. The
Phase 3 serialization lane renders images into this form before calling
the provider.

``fold`` and ``trim`` normalize their input first, so no operation here
can emit an orphan ``tool`` message or an unanswered assistant tool call.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any, Iterable

from pretender.types import ToolCall, ToolCallId, TranscriptMessage

FOLD_HEADER = "[已折叠的历史工具调用]"
IMAGE_PLACEHOLDER = "[图片]"
MAX_RESULT_CHARS = 200  # per-call result truncation inside a folded listing

# Calls dropped from the folded listing — safe ONLY because the whole turn
# folds (all-or-nothing); a partial drop would orphan the other calls.
_DROPPED_TOOLS = frozenset({"reply", "wait", "no_action"})
_SEARCH_TOOL = "tool_search"
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")

__all__ = [
    "FOLD_HEADER",
    "IMAGE_PLACEHOLDER",
    "MAX_RESULT_CHARS",
    "normalize",
    "fold",
    "trim",
    "apply_image_budget",
    "build_context",
    "serialize",
    "deserialize",
    "render_image_markdown",
]


def normalize(messages: Iterable[TranscriptMessage]) -> list[TranscriptMessage]:
    """Pair-normalize a transcript into the canonical role sequence.

    Canonical form (frozen decision #4): an assistant message with
    ``tool_calls`` is immediately followed by exactly one ``tool`` message
    per call id, in call order. Repair rules:

    - a ``tool`` message that answers no open call (no preceding assistant
      turn, unknown id, or duplicate answer) is dropped;
    - an assistant tool call that is never answered is dropped from its
      message (the message survives when it still has content);
    - tool results are emitted in the order of their calls, not the order
      they arrived;
    - duplicate call ids inside one assistant message are collapsed to the
      first occurrence (defensive: ``TranscriptMessage`` already fails
      closed on them), so a duplicate id can never yield duplicate tool
      results.

    Idempotent: ``normalize(normalize(x)) == normalize(x)``.
    """
    out: list[TranscriptMessage] = []
    open_assistant: TranscriptMessage | None = None
    open_calls: list[ToolCall] = []
    pending_tools: list[TranscriptMessage] = []
    answered: set[str] = set()

    def close_turn() -> None:
        nonlocal open_assistant, open_calls, pending_tools, answered
        if open_assistant is None:
            return
        by_id = {t.tool_call_id: t for t in pending_tools}
        kept = [c for c in open_calls if c.id in by_id]
        if kept:
            out.append(
                TranscriptMessage(
                    role="assistant",
                    content=open_assistant.content,
                    tool_calls=tuple(kept),
                )
            )
            for call in open_calls:
                tool_msg = by_id.get(call.id)
                if tool_msg is not None:
                    out.append(tool_msg)
        elif open_assistant.content:
            out.append(
                TranscriptMessage(role="assistant", content=open_assistant.content)
            )
        # else: nothing left of the turn — drop it entirely
        open_assistant = None
        open_calls = []
        pending_tools = []
        answered = set()

    for m in messages:
        if m.role == "tool":
            if (
                open_assistant is None
                or m.tool_call_id is None
                or m.tool_call_id in answered
            ):
                continue  # orphan or duplicate answer — drop
            answered.add(m.tool_call_id)
            pending_tools.append(m)
            continue
        close_turn()
        if m.role == "assistant" and m.tool_calls:
            open_assistant = m
            # Collapse duplicate call ids to the first occurrence: a
            # duplicate id would otherwise emit the same tool result twice.
            seen: set[str] = set()
            open_calls = []
            for c in m.tool_calls:
                if c.id in seen:
                    continue
                seen.add(c.id)
                open_calls.append(c)
        else:
            out.append(m)
    close_turn()
    return out


def fold(
    messages: Iterable[TranscriptMessage], *, keep_recent: int = 0
) -> list[TranscriptMessage]:
    """Fold completed assistant tool-call turns into one synthetic user
    message — all-or-nothing per turn.

    Each assistant tool-call turn (the assistant message plus ALL of its
    ``tool`` results) becomes a single user message headed
    ``[已折叠的历史工具调用]`` listing id/name/args/result per call.
    ``reply``/``wait``/``no_action`` calls are dropped from the listing;
    ``tool_search`` is compressed to its matched tool names so deferred
    tool activation survives the fold. The last ``keep_recent`` completed
    turns stay unfolded (nothing is dropped from them).

    Input is normalized first: incomplete turns are repaired (unanswered
    calls removed) and orphan ``tool`` messages dropped, so the output
    never contains an orphan tool message or an unanswered call.
    Idempotent for a fixed ``keep_recent``.
    """
    if keep_recent < 0:
        raise ValueError("keep_recent must be >= 0")
    msgs = normalize(messages)
    turns: list[tuple[int, int]] = []
    i = 0
    while i < len(msgs):
        if msgs[i].role == "assistant" and msgs[i].tool_calls:
            j = i + 1
            while j < len(msgs) and msgs[j].role == "tool":
                j += 1
            turns.append((i, j))
            i = j
        else:
            i += 1
    ends = dict(turns)
    fold_starts = {start for start, _ in turns[: max(0, len(turns) - keep_recent)]}
    out: list[TranscriptMessage] = []
    i = 0
    while i < len(msgs):
        if i in fold_starts:
            out.append(_fold_turn(msgs[i : ends[i]]))
            i = ends[i]
        else:
            out.append(msgs[i])
            i += 1
    return out


def trim(
    messages: Iterable[TranscriptMessage], max_context_size: int
) -> list[TranscriptMessage]:
    """Tool-group-aware history trim with the system message pinned.

    When the transcript exceeds ``2 * max_context_size`` messages it is
    cut back to ``max_context_size`` by dropping the OLDEST messages
    (``max_context_size`` is a message count). Rules:

    - the first ``system`` message is pinned and never dropped;
    - an assistant tool-call turn (assistant + its ``tool`` results) is
      dropped or kept as a whole — trimming never splits a turn, so no
      ``tool`` message is orphaned and no call loses its answer; when the
      whole group does not fit the drop budget it is kept whole
      (under-trimming is safe; orphaning is not);
    - an orphan ``tool`` message at the head of the kept window is dropped
      (defensive; normalization already removes these).

    Input is normalized first. Idempotent for a fixed ``max_context_size``.
    """
    if max_context_size < 1:
        raise ValueError("max_context_size must be >= 1")
    msgs = normalize(messages)
    if len(msgs) <= 2 * max_context_size:
        return msgs
    pinned = next((i for i, m in enumerate(msgs) if m.role == "system"), None)
    to_drop = len(msgs) - max_context_size
    dropped = 0
    i = 0
    while dropped < to_drop and i < len(msgs):
        if i == pinned:
            i += 1
            continue
        m = msgs[i]
        if m.role == "assistant" and m.tool_calls:
            j = i + 1
            while j < len(msgs) and msgs[j].role == "tool":
                j += 1
            if j - i <= to_drop - dropped:
                dropped += j - i
                i = j
            else:
                break  # group does not fit the budget — keep it whole
        else:
            dropped += 1
            i += 1
    kept = msgs[i:]
    if pinned is not None and pinned < i:
        # the pinned system message was skipped by the walk, not dropped —
        # put it back at the head of the kept window
        kept = [msgs[pinned]] + kept
    while kept and kept[0].role == "tool":  # defensive: never lead with an orphan
        kept = kept[1:]
    return kept


def apply_image_budget(
    messages: Iterable[TranscriptMessage], max_image_num: int
) -> list[TranscriptMessage]:
    """Reduce images past ``max_image_num`` to the ``[图片]`` placeholder.

    Images are markdown spans ``![alt](url)`` inside ``content``. The
    NEWEST ``max_image_num`` images in transcript order are kept intact;
    every older image becomes the literal placeholder ``[图片]`` (its
    alt/url are dropped). Deterministic and idempotent.
    """
    if max_image_num < 0:
        raise ValueError("max_image_num must be >= 0")
    msgs = list(messages)
    total = sum(len(_IMAGE_RE.findall(m.content or "")) for m in msgs)
    if total <= max_image_num:
        return msgs
    to_replace = total - max_image_num  # oldest images, in transcript order

    def repl(match: re.Match[str]) -> str:
        nonlocal to_replace
        if to_replace > 0:
            to_replace -= 1
            return IMAGE_PLACEHOLDER
        return match.group(0)

    out: list[TranscriptMessage] = []
    for m in msgs:
        if m.content is None or not _IMAGE_RE.search(m.content):
            out.append(m)
            continue
        out.append(replace(m, content=_IMAGE_RE.sub(repl, m.content)))
    return out


def build_context(
    messages: Iterable[TranscriptMessage],
    *,
    max_context_size: int,
    max_image_num: int,
    keep_recent: int = 0,
) -> list[TranscriptMessage]:
    """Compose the context pipeline for the planner/replyer: normalize →
    fold → trim → image budget.

    The output is canonical (no orphan ``tool`` message, no unanswered
    call), deterministic, and at most ``max_context_size`` messages unless
    a single tool group is larger — groups are kept whole. Idempotent for
    fixed parameters.
    """
    msgs = normalize(messages)
    msgs = fold(msgs, keep_recent=keep_recent)
    msgs = trim(msgs, max_context_size)
    return apply_image_budget(msgs, max_image_num)


# ── provider wire serialization (OpenAI-compatible) ─────────────────────────
# The pure transcript → wire lane the LLM layer consumes. It validates the
# canonical role sequence (frozen decision #4) and FAILS CLOSED on any
# malformed sequence rather than emitting a broken provider payload.

def serialize(messages: Iterable[TranscriptMessage]) -> list[dict[str, Any]]:
    """Serialize a canonical transcript into OpenAI-compatible wire dicts.

    The input is normalized first (pair-normalization repairs orphan ``tool``
    messages and unanswered calls), then the canonical form is re-validated
    and serialized. Tool-call ``arguments`` are emitted as JSON strings;
    every assistant tool call is answered by exactly one ``tool`` message in
    call order. A sequence that is still malformed after normalization raises
    ``ValueError`` (fail closed) instead of producing a broken payload.
    """
    msgs = normalize(messages)
    _validate_canonical(msgs)
    return [_to_wire(m) for m in msgs]


def deserialize(wire: Iterable[dict[str, Any]]) -> list[TranscriptMessage]:
    """Parse OpenAI-compatible wire dicts back into ``TranscriptMessage``s.

    The inverse of ``serialize``: tool-call ``arguments`` (JSON strings) are
    parsed back into dicts, and the resulting sequence is validated as
    canonical. Malformed wire input fails closed with ``ValueError``.
    """
    msgs = [_from_wire(d) for d in wire]
    _validate_canonical(msgs)
    return msgs


def render_image_markdown(alt: str, url: str) -> str:
    """Render a safe image markdown span ``![alt](url)``.

    The image convention (see module docstring) forbids ``)`` inside the URL
    so the span never breaks the markdown parser; a URL containing ``)`` is
    rejected rather than silently producing a corrupt span.
    """
    if ")" in url:
        raise ValueError("image url must not contain ')'")
    return f"![{alt}]({url})"


# ── serialization internals ─────────────────────────────────────────────────

def _validate_canonical(msgs: list[TranscriptMessage]) -> None:
    """Fail closed unless ``msgs`` is the canonical role sequence: every
    assistant tool-call turn is immediately followed by exactly one ``tool``
    message per call id, in call order, and no ``tool`` message exists
    outside such a group."""
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m.role == "assistant" and m.tool_calls:
            ids = [c.id for c in m.tool_calls]
            j = i + 1
            got: list[ToolCallId | None] = []
            while j < len(msgs) and msgs[j].role == "tool":
                got.append(msgs[j].tool_call_id)
                j += 1
            if got != ids:
                raise ValueError(
                    "malformed tool turn: every assistant tool call must be "
                    f"answered by exactly one tool result in call order ({ids} != {got})"
                )
            i = j
        else:
            if m.role == "tool":
                raise ValueError("orphan tool message in canonical transcript")
            i += 1


def _to_wire(m: TranscriptMessage) -> dict[str, Any]:
    if m.role == "system":
        return {"role": "system", "content": m.content or ""}
    if m.role == "user":
        content = m.content or ""
        if _IMAGE_RE.search(content):
            # Serialize image markdown into OpenAI-compatible multimodal
            # content parts (text + image_url), preserving ordinary text.
            return {"role": "user", "content": _multimodal_parts(content)}
        return {"role": "user", "content": content}
    if m.role == "assistant":
        d: dict[str, Any] = {"role": "assistant"}
        if m.content is not None:
            d["content"] = m.content
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.name,
                        "arguments": _wire_arguments(c),
                    },
                }
                for c in m.tool_calls
            ]
        return d
    if m.role == "tool":
        d: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": m.tool_call_id,
            "content": m.content or "",
        }
        if m.name is not None:
            d["name"] = m.name
        return d
    raise ValueError(f"cannot serialize unknown role: {m.role!r}")


def _wire_arguments(c: ToolCall) -> str:
    """The OpenAI ``arguments`` JSON string for one tool call.

    A call whose raw arguments could not be parsed re-emits them (as a JSON
    string when they were not already one) so the transcript round-trips
    legally and the tolerant parser can still attempt recovery downstream.
    """
    if c.raw_arguments is not None:
        raw = c.raw_arguments
        if isinstance(raw, str):
            return raw
        return json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(c.arguments, ensure_ascii=False, separators=(",", ":"))


def _multimodal_parts(content: str) -> list[dict[str, Any]]:
    """Split a content string containing image markdown spans into
    OpenAI-compatible multimodal parts: ``{"type": "text", "text": ...}``
    for ordinary text and ``{"type": "image_url", "image_url": {"url": ...}}``
    for each image. Ordinary text is preserved verbatim."""
    parts: list[dict[str, Any]] = []
    pos = 0
    for match in _IMAGE_RE.finditer(content):
        if match.start() > pos:
            text = content[pos : match.start()]
            if text:
                parts.append({"type": "text", "text": text})
        parts.append(
            {"type": "image_url", "image_url": {"url": match.group(2)}}
        )
        pos = match.end()
    if pos < len(content):
        text = content[pos:]
        if text:
            parts.append({"type": "text", "text": text})
    return parts


def _from_wire(d: dict[str, Any]) -> TranscriptMessage:
    role = d.get("role")
    if role == "system":
        return TranscriptMessage(role="system", content=d.get("content"))
    if role == "user":
        content = d.get("content")
        if isinstance(content, list):  # multimodal parts: keep the text
            content = "".join(
                p["text"]
                for p in content
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            ) or None
        return TranscriptMessage(role="user", content=content)
    if role == "assistant":
        calls: list[ToolCall] = []
        for tc in d.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments")
            parsed: Any = {}
            raw_arguments: Any = None
            if isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args)
                except (ValueError, TypeError):
                    # Malformed arguments JSON: preserve the raw string so
                    # the tolerant parser can attempt one-repair/no_action
                    # downstream instead of failing the whole transcript.
                    parsed, raw_arguments = {}, raw_args
            else:
                parsed = raw_args or {}
            if not isinstance(parsed, dict):
                # A non-object arguments value (e.g. a list) is preserved
                # raw for the tolerant parser, never silently dropped.
                parsed, raw_arguments = {}, raw_args
            calls.append(
                ToolCall(
                    id=ToolCallId(tc["id"]),
                    name=fn.get("name", ""),
                    arguments=parsed,
                    raw_arguments=raw_arguments,
                )
            )
        return TranscriptMessage(
            role="assistant", content=d.get("content"), tool_calls=tuple(calls)
        )
    if role == "tool":
        return TranscriptMessage(
            role="tool",
            tool_call_id=ToolCallId(d["tool_call_id"]),
            name=d.get("name"),
            content=d.get("content"),
        )
    raise ValueError(f"cannot deserialize unknown role: {role!r}")


# ── folding internals ───────────────────────────────────────────────────────


def _fold_turn(turn: list[TranscriptMessage]) -> TranscriptMessage:
    """Render one completed assistant tool-call turn as a synthetic user
    message. ``turn[0]`` is the assistant message; the rest are its tool
    results (normalized: one per call, in call order)."""
    assistant = turn[0]
    results = {t.tool_call_id: t for t in turn[1:]}
    lines = [FOLD_HEADER]
    if assistant.content:
        lines.append(assistant.content)
    for call in assistant.tool_calls:
        if call.name in _DROPPED_TOOLS:
            continue
        lines.append(_render_call(call, results.get(call.id)))
    return TranscriptMessage(role="user", content="\n".join(lines))


def _render_call(call: ToolCall, tool_msg: TranscriptMessage | None) -> str:
    if call.name == _SEARCH_TOOL:
        return f"- {call.id}: tool_search → matched: {_matched_names(tool_msg)}"
    args = ""
    if call.arguments:
        args = "(" + json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":")) + ")"
    return f"- {call.id}: {call.name}{args} → {_render_result(tool_msg)}"


def _render_result(tool_msg: TranscriptMessage | None) -> str:
    if tool_msg is None:
        return "no result"
    text = tool_msg.content or "ok"
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + "…"
    return text


def _matched_names(tool_msg: TranscriptMessage | None) -> str:
    """Compress a tool_search result to its matched tool names.

    Convention: the result content is JSON — a bare list of tool names or
    an object with a ``tools``/``matches``/``names`` list. When the
    content does not parse, the raw (truncated) content is kept so no
    match information is lost.
    """
    if tool_msg is None:
        return "?"
    text = tool_msg.content or ""
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        data = None
    names: list[str] | None = None
    if isinstance(data, list):
        names = [str(n) for n in data]
    elif isinstance(data, dict):
        for key in ("tools", "matches", "names"):
            value = data.get(key)
            if isinstance(value, list):
                names = [str(n) for n in value]
                break
    if names:
        return ", ".join(names)
    return _render_result(tool_msg)