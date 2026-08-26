"""Pure context management: pair-normalization, all-or-nothing tool-turn
folding, tool-group-aware trim with the system message pinned, newest-kept
image budget, and randomized property coverage over tool-call sequences.

Every test here is offline and deterministic (stdlib ``random`` with fixed
seeds); no LLM calls, no I/O.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from pretender.context import (
    FOLD_HEADER,
    IMAGE_PLACEHOLDER,
    _validate_canonical,
    apply_image_budget,
    build_context,
    deserialize,
    fold,
    normalize,
    render_image_markdown,
    serialize,
    trim,
)
from pretender.types import ToolCall, ToolCallId, TranscriptMessage

# ── builders ────────────────────────────────────────────────────────────────


def _call(cid: str, name: str = "query_memory", **args: Any) -> ToolCall:
    return ToolCall(id=ToolCallId(cid), name=name, arguments=args or {"q": cid})


def _assistant(calls: list[ToolCall], content: str | None = None) -> TranscriptMessage:
    return TranscriptMessage(role="assistant", content=content, tool_calls=tuple(calls))


def _tool(cid: str, name: str = "query_memory", content: str = "ok") -> TranscriptMessage:
    return TranscriptMessage(
        role="tool", tool_call_id=ToolCallId(cid), name=name, content=content
    )


def _user(text: str) -> TranscriptMessage:
    return TranscriptMessage(role="user", content=text)


def _system(text: str = "persona") -> TranscriptMessage:
    return TranscriptMessage(role="system", content=text)


def _content(m: TranscriptMessage) -> str:
    """Fold outputs are user messages with content; narrow the type."""
    assert m.content is not None
    return m.content


def _assert_canonical(msgs: list[TranscriptMessage]) -> None:
    """The frozen transcript invariant: every assistant tool call is
    answered by exactly one immediately-following ``tool`` message (in
    call order), and no ``tool`` message exists outside such a group."""
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
            assert got == ids, f"turn answers mismatch: {got} != {ids}"
            i = j
        else:
            assert m.role != "tool", f"orphan tool message: {m}"
            i += 1


# ── pair normalization ──────────────────────────────────────────────────────


def test_normalize_drops_orphan_tool():
    msgs = [_tool("c1"), _user("hi")]
    assert normalize(msgs) == [_user("hi")]


def test_normalize_drops_duplicate_answer():
    msgs = [_assistant([_call("c1")]), _tool("c1"), _tool("c1")]
    assert normalize(msgs) == [_assistant([_call("c1")]), _tool("c1")]


def test_normalize_drops_unanswered_calls_keeps_answered():
    msgs = [_assistant([_call("c1"), _call("c2")]), _tool("c1"), _user("hi")]
    assert normalize(msgs) == [_assistant([_call("c1")]), _tool("c1"), _user("hi")]


def test_normalize_orders_tools_by_call_not_arrival():
    msgs = [_assistant([_call("c1"), _call("c2")]), _tool("c2"), _tool("c1")]
    assert normalize(msgs) == [
        _assistant([_call("c1"), _call("c2")]),
        _tool("c1"),
        _tool("c2"),
    ]


def test_normalize_keeps_content_when_all_calls_dropped():
    msgs = [_assistant([_call("c1")], content="thinking..."), _user("hi")]
    assert normalize(msgs) == [
        TranscriptMessage(role="assistant", content="thinking..."),
        _user("hi"),
    ]


def test_normalize_drops_empty_turn_entirely():
    msgs = [_assistant([_call("c1")]), _user("hi")]
    assert normalize(msgs) == [_user("hi")]


def test_normalize_drops_tool_after_turn_closed():
    msgs = [_assistant([_call("c1")]), _tool("c1"), _user("hi"), _tool("c1")]
    assert normalize(msgs) == [_assistant([_call("c1")]), _tool("c1"), _user("hi")]


def test_normalize_two_complete_turns_unchanged():
    msgs = [
        _assistant([_call("c1")]),
        _tool("c1"),
        _user("hi"),
        _assistant([_call("c2")]),
        _tool("c2"),
    ]
    assert normalize(msgs) == msgs


def test_normalize_drops_tool_without_call_id():
    # Fail closed: a tool message without tool_call_id cannot even be
    # constructed (TranscriptMessage rejects it) — the normalize guard for
    # a None tool_call_id is defensive only.
    with pytest.raises(ValueError, match="tool_call_id"):
        TranscriptMessage(role="tool", tool_call_id=None, name="x", content="bad")
    msgs = [_assistant([_call("c1")]), _tool("c1")]
    assert normalize(msgs) == msgs


def test_normalize_duplicate_call_ids_cannot_yield_duplicate_results():
    # Defensive: even if an assistant message somehow carries duplicate
    # call ids (TranscriptMessage fails closed on them at construction),
    # normalize collapses them to the first occurrence — one call, one
    # result, never a duplicated tool result.
    dup = _assistant([_call("c1")])
    object.__setattr__(dup, "tool_calls", (_call("c1"), _call("c1")))
    msgs = [dup, _tool("c1")]
    out = normalize(msgs)
    assert out == [_assistant([_call("c1")]), _tool("c1")]
    _assert_canonical(out)


def test_fold_duplicate_call_ids_cannot_yield_duplicate_results():
    dup = _assistant([_call("c1")])
    object.__setattr__(dup, "tool_calls", (_call("c1"), _call("c1")))
    out = fold([dup, _tool("c1")])
    assert len(out) == 1
    # the call is listed exactly once (the id also appears inside its own
    # arguments, so count the listing prefix, not the bare id)
    assert _content(out[0]).count("- c1:") == 1
    _assert_canonical(out)


def test_normalize_idempotent():
    msgs = [
        _assistant([_call("c1"), _call("c2")]),
        _tool("c2"),
        _tool("c1"),
        _user("hi"),
        _tool("stray"),
    ]
    once = normalize(msgs)
    assert normalize(once) == once


# ── folding ─────────────────────────────────────────────────────────────────


def test_fold_mixed_reply_query_memory_turn():
    """The required exact case: a mixed [reply, query_memory] turn folds
    all-or-nothing into ONE synthetic user message; reply is dropped from
    the listing, query_memory keeps id/name/args/result."""
    msgs = [
        _assistant([_call("c1", "reply"), _call("c2", "query_memory", query="x")]),
        _tool("c1", "reply", content="好的"),
        _tool("c2", "query_memory", content="found: 42"),
    ]
    out = fold(msgs)
    assert len(out) == 1
    assert out[0].role == "user"
    assert out[0].tool_calls == ()
    assert _content(out[0]) == (
        f"{FOLD_HEADER}\n" '- c2: query_memory({"query":"x"}) → found: 42'
    )
    assert "reply" not in _content(out[0])


def test_fold_all_or_nothing_keep_recent():
    """When a turn is NOT folded, nothing is dropped from it — the reply
    tool message stays with its call (no orphan, no half-drop)."""
    msgs = [
        _assistant([_call("c1", "reply"), _call("c2", "query_memory", query="x")]),
        _tool("c1", "reply", content="好的"),
        _tool("c2", "query_memory", content="found: 42"),
    ]
    assert fold(msgs, keep_recent=1) == msgs


def test_fold_drops_reply_wait_no_action_only_with_whole_turn():
    msgs = [
        _assistant(
            [
                _call("c1", "reply"),
                _call("c2", "wait"),
                _call("c3", "query_memory", query="x"),
            ]
        ),
        _tool("c1", "reply", content="好的"),
        _tool("c2", "wait", content="30"),
        _tool("c3", "query_memory", content="found"),
    ]
    out = fold(msgs)
    assert len(out) == 1
    assert "reply" not in _content(out[0]) and "wait" not in _content(out[0])
    assert "c3" in _content(out[0]) and "found" in _content(out[0])
    # the same turn left unfolded keeps every tool message intact
    assert fold(msgs, keep_recent=1) == msgs


def test_fold_no_action_turn_folds_to_header_only():
    msgs = [_assistant([_call("c1", "no_action")]), _tool("c1", "no_action", content="")]
    assert fold(msgs) == [_user(FOLD_HEADER)]


@pytest.mark.parametrize(
    "content,expected",
    [
        (
            '{"matches": ["query_memory", "query_person_profile"]}',
            "matched: query_memory, query_person_profile",
        ),
        ('{"tools": ["query_memory"]}', "matched: query_memory"),
        ('{"names": ["query_jargon"]}', "matched: query_jargon"),
        ('["query_memory", "query_jargon"]', "matched: query_memory, query_jargon"),
        # non-JSON content falls back to the raw result — matches survive
        ("query_memory, query_person_profile", "query_memory, query_person_profile"),
    ],
)
def test_fold_tool_search_compression_preserves_matches(content, expected):
    """tool_search compresses to matched tool names so deferred-tool
    activation survives the fold."""
    msgs = [
        _assistant([_call("c1", "tool_search", query="memory")]),
        _tool("c1", "tool_search", content=content),
    ]
    out = fold(msgs)
    assert len(out) == 1
    assert expected in out[0].content


def test_fold_incomplete_turn_never_orphans():
    """An incomplete turn (one call answered, one not) is repaired before
    folding: the unanswered call disappears, the answered one survives."""
    msgs = [
        _assistant([_call("c1"), _call("c2")]),
        _tool("c1"),
        _user("hi"),
    ]
    out = fold(msgs)
    assert len(out) == 2
    assert out[0].role == "user" and "c1" in _content(out[0])
    assert "c2" not in _content(out[0])
    assert out[1] == _user("hi")


def test_fold_preserves_assistant_content():
    msgs = [
        _assistant([_call("c1")], content="analysis: 42"),
        _tool("c1", content="found"),
    ]
    out = fold(msgs)
    assert _content(out[0]).startswith(f"{FOLD_HEADER}\nanalysis: 42\n")


def test_fold_truncates_long_results():
    long = "x" * 500
    msgs = [_assistant([_call("c1")]), _tool("c1", content=long)]
    out = fold(msgs)
    assert "…" in _content(out[0])
    assert len(_content(out[0])) < len(FOLD_HEADER) + 250


def test_fold_empty_result_renders_ok():
    msgs = [_assistant([_call("c1")]), _tool("c1", content="")]
    assert _content(fold(msgs)[0]).endswith("→ ok")


def test_fold_keep_recent_leaves_last_turns_unfolded():
    msgs = [
        _assistant([_call("c1")]),
        _tool("c1"),
        _user("mid"),
        _assistant([_call("c2")]),
        _tool("c2"),
    ]
    out = fold(msgs, keep_recent=1)
    assert out[0].role == "user" and "c1" in _content(out[0])
    assert out[1:] == msgs[2:]  # the last turn stays fully unfolded


def test_fold_idempotent():
    msgs = [
        _assistant([_call("c1", "reply"), _call("c2")]),
        _tool("c1", "reply"),
        _tool("c2", content="found"),
        _user("hi"),
    ]
    once = fold(msgs)
    assert fold(once) == once
    assert fold(once, keep_recent=1) == once


# ── trim ────────────────────────────────────────────────────────────────────


def test_trim_pins_system_message():
    msgs = [_system()] + [_user(f"m{i}") for i in range(10)]
    out = trim(msgs, max_context_size=3)
    assert out[0] == _system()
    assert len(out) == 3
    assert [m.content for m in out[1:]] == ["m8", "m9"]


def test_trim_to_one_keeps_only_system():
    msgs = [_system()] + [_user(f"m{i}") for i in range(5)]
    assert trim(msgs, max_context_size=1) == [_system()]


def test_trim_pins_mid_transcript_system():
    msgs = [_user("m0"), _user("m1"), _system(), _user("m2"), _user("m3"), _user("m4"), _user("m5")]
    out = trim(msgs, max_context_size=2)
    assert out == [_system(), _user("m5")]


def test_trim_below_threshold_unchanged():
    msgs = [_system()] + [_user(f"m{i}") for i in range(5)]  # len 6 == 2*3
    assert trim(msgs, max_context_size=3) == msgs


def test_trim_above_threshold_cuts_to_max():
    msgs = [_system()] + [_user(f"m{i}") for i in range(6)]  # len 7 > 2*3
    out = trim(msgs, max_context_size=3)
    assert out == [_system(), _user("m4"), _user("m5")]


def test_trim_drops_tool_group_whole():
    msgs = [
        _system(),
        _assistant([_call("c1"), _call("c2")]),
        _tool("c1"),
        _tool("c2"),
        _user("m0"),
        _user("m1"),
        _user("m2"),
    ]
    out = trim(msgs, max_context_size=2)
    assert out == [_system(), _user("m2")]


def test_trim_keeps_tool_group_whole_when_budget_short():
    """A group larger than the drop budget is kept whole — under-trimming
    is safe, orphaning is not."""
    msgs = [
        _system(),
        _assistant([_call(f"c{i}") for i in range(5)]),
        *[_tool(f"c{i}") for i in range(5)],
    ]
    out = trim(msgs, max_context_size=2)
    assert out == msgs
    _assert_canonical(out)


def test_trim_never_splits_turn_across_boundary():
    msgs = [
        _system(),
        _user("m0"),
        _user("m1"),
        _assistant([_call("c1"), _call("c2")]),
        _tool("c1"),
        _tool("c2"),
        _user("m2"),
    ]
    out = trim(msgs, max_context_size=3)
    _assert_canonical(out)
    assert out[0] == _system()
    assert out[-1] == _user("m2")


def test_trim_idempotent():
    msgs = [_system()] + [_user(f"m{i}") for i in range(12)]
    once = trim(msgs, max_context_size=4)
    assert trim(once, max_context_size=4) == once


def test_trim_invalid_max_size():
    with pytest.raises(ValueError):
        trim([], max_context_size=0)


# ── image budget ────────────────────────────────────────────────────────────


def test_image_budget_keeps_newest():
    msgs = [_user("![a](u1) ![b](u2)"), _user("![c](u3)")]
    out = apply_image_budget(msgs, max_image_num=1)
    assert out[0].content == f"{IMAGE_PLACEHOLDER} {IMAGE_PLACEHOLDER}"
    assert out[1].content == "![c](u3)"


def test_image_budget_zero_replaces_all():
    msgs = [_user("![a](u1)"), _user("![b](u2)")]
    out = apply_image_budget(msgs, max_image_num=0)
    assert out[0].content == IMAGE_PLACEHOLDER
    assert out[1].content == IMAGE_PLACEHOLDER


def test_image_budget_within_limit_unchanged():
    msgs = [_user("![a](u1)"), _user("![b](u2)")]
    assert apply_image_budget(msgs, max_image_num=2) == msgs


def test_image_budget_applies_to_all_roles():
    msgs = [_user("![a](u1)"), _tool("c1", content="![b](u2)")]
    out = apply_image_budget(msgs, max_image_num=1)
    assert out[0].content == IMAGE_PLACEHOLDER
    assert out[1].content == "![b](u2)"


def test_image_budget_placeholder_not_touched():
    msgs = [_user(f"{IMAGE_PLACEHOLDER} ![a](u1)")]
    out = apply_image_budget(msgs, max_image_num=0)
    assert out[0].content == f"{IMAGE_PLACEHOLDER} {IMAGE_PLACEHOLDER}"


def test_image_budget_idempotent():
    msgs = [_user("![a](u1) ![b](u2)"), _user("![c](u3)")]
    once = apply_image_budget(msgs, max_image_num=1)
    assert apply_image_budget(once, max_image_num=1) == once


def test_image_budget_invalid_max():
    with pytest.raises(ValueError):
        apply_image_budget([], max_image_num=-1)


# ── build_context pipeline ──────────────────────────────────────────────────


def test_build_context_pipeline():
    msgs = [
        _system(),
        _assistant([_call("c1", "reply"), _call("c2", "query_memory", query="x")]),
        _tool("c1", "reply", content="好的"),
        _tool("c2", "query_memory", content="![img](http://x/1.png) found"),
        _user("![img](http://x/2.png) hi"),
    ]
    out = build_context(msgs, max_context_size=10, max_image_num=1)
    assert out[0] == _system()
    assert out[1].role == "user" and FOLD_HEADER in _content(out[1])
    assert "reply" not in _content(out[1])
    assert out[2].content == "![img](http://x/2.png) hi"
    _assert_canonical(out)


def test_build_context_bounds_and_canonical():
    msgs = [_system()] + [_user(f"m{i}") for i in range(30)]
    out = build_context(msgs, max_context_size=5, max_image_num=2)
    assert len(out) == 5
    assert out[0] == _system()
    _assert_canonical(out)


def test_build_context_deterministic():
    rng = random.Random(7)
    msgs = _random_transcript(rng, 60)
    a = build_context(msgs, max_context_size=8, max_image_num=2)
    b = build_context(msgs, max_context_size=8, max_image_num=2)
    assert a == b


def test_build_context_idempotent():
    rng = random.Random(11)
    msgs = _random_transcript(rng, 40)
    built = build_context(msgs, max_context_size=6, max_image_num=1, keep_recent=1)
    again = build_context(built, max_context_size=6, max_image_num=1, keep_recent=1)
    assert again == built


def test_fold_invalid_keep_recent():
    with pytest.raises(ValueError):
        fold([], keep_recent=-1)


# ── randomized property coverage ────────────────────────────────────────────

_TOOL_NAMES = [
    "query_memory",
    "query_person_profile",
    "query_jargon",
    "reply",
    "wait",
    "no_action",
    "tool_search",
    "send_emoji",
]


def _random_transcript(rng: random.Random, n: int) -> list[TranscriptMessage]:
    """A deliberately messy transcript: complete and incomplete tool turns,
    duplicate and stray answers, orphan tool messages, interleaved plain
    messages, and system messages anywhere."""
    msgs: list[TranscriptMessage] = []
    for _ in range(n):
        r = rng.random()
        if r < 0.20:
            msgs.append(_user(f"user-{rng.randrange(100)}"))
        elif r < 0.28:
            msgs.append(_system())
        elif r < 0.36:
            msgs.append(TranscriptMessage(role="assistant", content=f"plain-{rng.randrange(100)}"))
        elif r < 0.80:
            calls = [
                ToolCall(
                    id=ToolCallId(f"c{rng.randrange(1000)}"),
                    name=rng.choice(_TOOL_NAMES),
                    arguments={"k": rng.randrange(10)},
                )
                for _ in range(rng.randrange(1, 4))
            ]
            # Dedupe call ids: TranscriptMessage fails closed on duplicate
            # ids within one assistant message (they could otherwise yield
            # duplicate tool results after normalization).
            seen: set[str] = set()
            unique: list[ToolCall] = []
            for c in calls:
                if c.id in seen:
                    continue
                seen.add(c.id)
                unique.append(c)
            calls = unique
            msgs.append(
                TranscriptMessage(
                    role="assistant",
                    content=None if rng.random() < 0.8 else "with-content",
                    tool_calls=tuple(calls),
                )
            )
            for _ in range(rng.randrange(0, 4)):
                if calls and rng.random() < 0.75:
                    call = rng.choice(calls)
                    msgs.append(
                        TranscriptMessage(
                            role="tool",
                            tool_call_id=call.id,
                            name=call.name,
                            content=f"result-{call.id}",
                        )
                    )
                else:
                    msgs.append(
                        TranscriptMessage(
                            role="tool",
                            tool_call_id=ToolCallId(f"stray{rng.randrange(1000)}"),
                            name="x",
                            content="stray",
                        )
                    )
        else:
            msgs.append(
                TranscriptMessage(
                    role="tool",
                    tool_call_id=ToolCallId(f"stray{rng.randrange(1000)}"),
                    name="x",
                    content="stray",
                )
            )
    return msgs


def test_property_no_orphans_random_transcripts():
    """Over random tool-call sequences: every surviving tool_call_id stays
    answered, no orphan tool message survives normalize/fold/trim, and
    every operation is idempotent."""
    for seed in range(20):
        rng = random.Random(seed)
        for _ in range(15):
            msgs = _random_transcript(rng, rng.randrange(0, 50))
            max_size = rng.randrange(1, 10)
            max_img = rng.randrange(0, 4)
            keep = rng.randrange(0, 3)

            _assert_canonical(normalize(msgs))
            _assert_canonical(fold(msgs, keep_recent=keep))
            _assert_canonical(trim(msgs, max_size))
            _assert_canonical(
                build_context(
                    msgs,
                    max_context_size=max_size,
                    max_image_num=max_img,
                    keep_recent=keep,
                )
            )

            assert normalize(normalize(msgs)) == normalize(msgs)
            assert (
                fold(fold(msgs, keep_recent=keep), keep_recent=keep)
                == fold(msgs, keep_recent=keep)
            )
            assert trim(trim(msgs, max_size), max_size) == trim(msgs, max_size)
            built = build_context(
                msgs,
                max_context_size=max_size,
                max_image_num=max_img,
                keep_recent=keep,
            )
            assert (
                build_context(
                    built,
                    max_context_size=max_size,
                    max_image_num=max_img,
                    keep_recent=keep,
                )
                == built
            )


# ── provider wire serialization ─────────────────────────────────────────────

def test_serialize_round_trip():
    msgs = [
        _system("persona"),
        _user("hello"),
        _assistant([_call("c1", "query_memory", q="x"), _call("c2", "lookup", id=7)]),
        _tool("c1", "query_memory", content="result-a"),
        _tool("c2", "lookup", content="result-b"),
        _user("thanks"),
    ]
    wire = serialize(msgs)
    assert deserialize(wire) == msgs


def test_serialize_wire_shape():
    msgs = [
        _system("persona"),
        _user("hi"),
        _assistant([_call("c1", "query_memory", q="x")], content="thinking"),
        _tool("c1", "query_memory", content="ok"),
    ]
    wire = serialize(msgs)
    assert wire[0] == {"role": "system", "content": "persona"}
    assert wire[1] == {"role": "user", "content": "hi"}
    # assistant tool-call turn: arguments are JSON strings
    assert wire[2]["role"] == "assistant"
    assert wire[2]["content"] == "thinking"
    tc = wire[2]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["id"] == "c1"
    assert tc["function"]["name"] == "query_memory"
    assert tc["function"]["arguments"] == '{"q":"x"}'
    assert wire[3] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "ok",
        "name": "query_memory",
    }


def test_serialize_omits_content_when_none_on_assistant():
    msgs = [_assistant([_call("c1")]), _tool("c1")]
    wire = serialize(msgs)
    assert "content" not in wire[0]
    assert wire[0]["tool_calls"][0]["function"]["arguments"] == '{"q":"c1"}'


def test_serialize_normalizes_before_emitting():
    # orphan tool + unanswered call are repaired, not emitted broken
    msgs = [_tool("orphan"), _assistant([_call("c1"), _call("c2")]), _tool("c1")]
    wire = serialize(msgs)
    assert deserialize(wire) == [_assistant([_call("c1")]), _tool("c1")]


def test_serialize_fails_closed_on_malformed_after_normalize():
    # A tool message that answers no open call is dropped by normalize, so a
    # canonical transcript never carries an orphan — but a hand-built
    # non-canonical sequence must fail closed rather than emit a broken wire.
    with pytest.raises(ValueError, match="orphan tool"):
        _validate_canonical([_tool("c1")])
    with pytest.raises(ValueError, match="malformed tool turn"):
        _validate_canonical([_assistant([_call("c1"), _call("c2")]), _tool("c1")])


def test_deserialize_preserves_malformed_arguments_raw():
    """Malformed arguments JSON is preserved raw (for the tolerant parser),
    never a transcript failure."""
    msgs = deserialize(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "x", "arguments": "not json"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
    )
    call = msgs[0].tool_calls[0]
    assert call.id == ToolCallId("c1")
    assert call.arguments == {}
    assert call.raw_arguments == "not json"
    # the wire round-trips legally: the raw string is re-emitted verbatim
    assert serialize(msgs)[0]["tool_calls"][0]["function"]["arguments"] == "not json"


def test_deserialize_preserves_non_object_arguments_raw():
    msgs = deserialize(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "x", "arguments": "[1, 2]"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
    )
    call = msgs[0].tool_calls[0]
    assert call.arguments == {}
    assert call.raw_arguments == "[1, 2]"


def test_deserialize_rejects_orphan_tool():
    with pytest.raises(ValueError, match="orphan tool"):
        deserialize([{"role": "tool", "tool_call_id": "c1", "content": "ok"}])


def test_render_image_markdown():
    assert render_image_markdown("alt", "https://x/y.png") == "![alt](https://x/y.png)"
    with pytest.raises(ValueError, match=r"\)"):
        render_image_markdown("alt", "https://x/a)b.png")


def test_serialize_multimodal_wire_parts():
    """Image markdown serializes into OpenAI-compatible multimodal content
    parts (text + image_url), preserving ordinary text; plain text stays a
    plain string."""
    msgs = [
        _system("persona"),
        _user("see ![a](https://x/1.png) and ![b](https://x/2.png) done"),
        _user("no images here"),
    ]
    wire = serialize(msgs)
    assert wire[1]["role"] == "user"
    assert wire[1]["content"] == [
        {"type": "text", "text": "see "},
        {"type": "image_url", "image_url": {"url": "https://x/1.png"}},
        {"type": "text", "text": " and "},
        {"type": "image_url", "image_url": {"url": "https://x/2.png"}},
        {"type": "text", "text": " done"},
    ]
    assert wire[2]["content"] == "no images here"


def test_serialize_multimodal_round_trip_keeps_text():
    msgs = [_user("see ![a](https://x/1.png) now")]
    wire = serialize(msgs)
    assert deserialize(wire) == [_user("see  now")]