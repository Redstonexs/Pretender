"""Tolerant tool-call parsing (Phase 3 capability lane).

Covers: valid ToolCall/LLMResponse shapes, fenced/prose-wrapped JSON, common
JSON mistakes (trailing commas, single quotes, unquoted keys, truncation),
malformed arguments, unknown tools, schema mismatch, duplicate ids, the single
caller repair attempt (success and failure -> no_action), exception
containment, purity (no handler/network I/O), and property-style randomized
malformed inputs that never orphan an id.
"""

from __future__ import annotations

import json
import random
import re
import string

import pytest

from pretender.toolparse import (
    NO_ACTION_NAME,
    _extract_ids_from_text,
    _ids_outside_json_blocks,
    parse_tool_calls,
    tolerant_loads,
    validate_arguments,
)
from pretender.tools import ToolRegistry, tool
from pretender.types import LLMResponse, ToolCall, ToolCallId, ToolResult


@pytest.fixture
def specs():
    @tool("get_weather", description="Weather lookup.")
    def get_weather(city: str, unit: str = "celsius") -> str:
        """Look up weather for a city."""
        return ""

    @tool("add")
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return 0

    @tool("noop")
    def noop(payload: object = None) -> str:
        """Accept anything."""
        return ""

    return [get_weather, add, noop]


def _call(cid, name, args=None, **kw):
    return ToolCall(id=ToolCallId(cid), name=name, arguments=args or {}, **kw)


# ── valid shapes ────────────────────────────────────────────────────────────

def test_valid_toolcall_sequence(specs):
    calls = [
        _call("c1", "get_weather", {"city": "SF"}),
        _call("c2", "add", {"a": 1, "b": 2}),
    ]
    results = parse_tool_calls(calls, specs)
    assert [r.call_id for r in results] == ["c1", "c2"]
    assert [r.name for r in results] == ["get_weather", "add"]
    assert all(r.ok for r in results)
    assert results[0].data == {"city": "SF"}
    assert json.loads(results[0].content) == {"city": "SF"}


def test_valid_llmresponse(specs):
    resp = LLMResponse(
        content="ok",
        tool_calls=(_call("c1", "add", {"a": 1, "b": 2}),),
        finish_reason="tool_calls",
    )
    results = parse_tool_calls(resp, specs)
    assert len(results) == 1
    assert results[0].ok
    assert results[0].data == {"a": 1, "b": 2}


def test_empty_sources(specs):
    assert parse_tool_calls(LLMResponse(content="hi"), specs) == ()
    assert parse_tool_calls((), specs) == ()
    assert parse_tool_calls([], specs) == ()
    assert parse_tool_calls(None, specs) == ()


def test_openai_dict_shape(specs):
    source = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": json.dumps({"city": "SF"}),
        },
    }
    results = parse_tool_calls(source, specs)
    assert len(results) == 1
    assert results[0].call_id == "call_1"
    assert results[0].ok
    assert results[0].data == {"city": "SF"}


def test_tool_calls_wrapper_dict(specs):
    source = {"tool_calls": [{"id": "a1", "name": "add", "arguments": {"a": 1, "b": 2}}]}
    results = parse_tool_calls(source, specs)
    assert [r.call_id for r in results] == ["a1"]
    assert results[0].ok


def test_choices_message_wrapper(specs):
    source = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}}
                    ]
                }
            }
        ]
    }
    results = parse_tool_calls(source, specs)
    assert [r.call_id for r in results] == ["c1"]
    assert results[0].ok


def test_registry_as_tool_specs(specs):
    reg = ToolRegistry()
    for s in specs:
        reg.register(s)
    results = parse_tool_calls(_call("c1", "add", {"a": 1, "b": 2}), reg)
    assert results[0].ok


# ── fenced / prose / JSON mistakes ─────────────────────────────────────────

def test_fenced_json_string(specs):
    text = '```json\n{"tool_calls": [{"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}}]}\n```'
    results = parse_tool_calls(text, specs)
    assert [r.call_id for r in results] == ["c1"]
    assert results[0].ok


def test_prose_around_json(specs):
    text = 'Sure! Here is the JSON you asked for:\n{"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}}\nHope that helps.'
    results = parse_tool_calls(text, specs)
    assert [r.call_id for r in results] == ["c1"]
    assert results[0].ok


def test_trailing_commas(specs):
    text = '{"tool_calls": [{"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2,},},]}'
    results = parse_tool_calls(text, specs)
    assert [r.call_id for r in results] == ["c1"]
    assert results[0].ok


def test_single_quotes_and_unquoted_keys(specs):
    text = "{tool_calls: [{id: c1, name: add, arguments: {a: 1, b: 2}}]}"
    results = parse_tool_calls(text, specs)
    assert [r.call_id for r in results] == ["c1"]
    assert results[0].ok
    assert results[0].data == {"a": 1, "b": 2}


def test_truncated_json_closed_by_repair(specs):
    text = '{"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}'
    results = parse_tool_calls(text, specs)
    assert [r.call_id for r in results] == ["c1"]
    assert results[0].ok


def test_single_quoted_arguments_string(specs):
    text = """{"id": "c1", "name": "add", "arguments": "{'a': 1, 'b': 2}"}"""
    results = parse_tool_calls(text, specs)
    assert results[0].ok
    assert results[0].data == {"a": 1, "b": 2}


# ── malformed arguments / unknown tool / schema ────────────────────────────

def test_malformed_arguments(specs):
    source = {"id": "c1", "name": "add", "arguments": "this is not json at all ###"}
    results = parse_tool_calls(source, specs)
    assert len(results) == 1
    assert results[0].call_id == "c1"
    assert not results[0].ok
    assert results[0].name == "add"
    assert results[0].error.startswith("malformed arguments")


def test_arguments_not_object(specs):
    source = {"id": "c1", "name": "add", "arguments": "[1, 2, 3]"}
    results = parse_tool_calls(source, specs)
    assert not results[0].ok
    assert results[0].error.startswith("malformed arguments")


def test_unknown_tool(specs):
    results = parse_tool_calls(_call("c1", "nope", {"x": 1}), specs)
    assert not results[0].ok
    assert results[0].name == "nope"
    assert results[0].error.startswith("unknown tool")


def test_schema_mismatch_type(specs):
    results = parse_tool_calls(_call("c1", "add", {"a": "one", "b": 2}), specs)
    assert not results[0].ok
    assert results[0].error.startswith("schema mismatch")


def test_schema_missing_required(specs):
    results = parse_tool_calls(_call("c1", "add", {"a": 1}), specs)
    assert not results[0].ok
    assert "missing required" in results[0].error


def test_default_and_optional_params(specs):
    results = parse_tool_calls(_call("c1", "get_weather", {"city": "SF"}), specs)
    assert results[0].ok


def test_missing_tool_name(specs):
    results = parse_tool_calls({"id": "c1", "arguments": {}}, specs)
    assert len(results) == 1
    assert not results[0].ok
    assert results[0].name == NO_ACTION_NAME
    assert "missing tool name" in results[0].error


# ── duplicate ids / ordering ───────────────────────────────────────────────

def test_duplicate_ids_first_wins(specs):
    calls = [
        _call("c1", "add", {"a": 1, "b": 2}),
        _call("c1", "get_weather", {"city": "SF"}),
    ]
    results = parse_tool_calls(calls, specs)
    assert len(results) == 1
    assert results[0].call_id == "c1"
    assert results[0].name == "add"
    assert results[0].ok


def test_duplicate_ids_with_broken_first(specs):
    calls = [
        _call("c1", "nope"),
        _call("c1", "add", {"a": 1, "b": 2}),
    ]
    results = parse_tool_calls(calls, specs)
    assert len(results) == 1
    assert not results[0].ok
    assert results[0].error.startswith("unknown tool")


def test_order_preserved_mixed(specs):
    source = [
        {"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}},
        {"id": "c2", "name": "ghost", "arguments": {}},
        '{"id": "c3", "name": "noop"}',
        _call("c4", "get_weather", {"city": "SF"}),
    ]
    results = parse_tool_calls(source, specs)
    assert [r.call_id for r in results] == ["c1", "c2", "c3", "c4"]
    assert [r.name for r in results] == ["add", "ghost", "noop", "get_weather"]
    assert results[1].error.startswith("unknown tool")
    assert results[2].ok


# ── caller repair: one attempt ─────────────────────────────────────────────

def test_repair_success_strips_comments(specs):
    def strip_comments(s):
        return re.sub(r"/\*.*?\*/", "", s)

    source = '{"id": "c1" /* id */, "name": "add", "arguments": {"a": 1, "b": 2}}'
    results = parse_tool_calls(source, specs, repair=strip_comments)
    assert len(results) == 1
    assert results[0].ok
    assert results[0].data == {"a": 1, "b": 2}


def test_repair_called_exactly_once(specs):
    count = {"n": 0}

    def strip_comments(s):
        count["n"] += 1
        return re.sub(r"/\*.*?\*/", "", s)

    source = '{"id": "c1" /* id */, "name": "add", "arguments": {"a": 1, "b": 2}}'
    results = parse_tool_calls(source, specs, repair=strip_comments)
    assert count["n"] == 1
    assert results[0].ok


def test_repair_failure_degrades_to_no_action(specs):
    def bad_repair(s):
        return "still broken"

    results = parse_tool_calls(
        '{"id": "c1", "name": "add", "arguments": }', specs, repair=bad_repair
    )
    assert len(results) == 1
    assert results[0].call_id == "c1"
    assert not results[0].ok
    assert results[0].name == NO_ACTION_NAME
    assert results[0].error.startswith("unparseable tool call")


def test_repair_raising_is_contained(specs):
    def boom(s):
        raise RuntimeError("kapow")

    results = parse_tool_calls(
        '{"id": "c1", "name": "add", "arguments": }', specs, repair=boom
    )
    assert len(results) == 1
    assert results[0].call_id == "c1"
    assert not results[0].ok
    assert "repair" in results[0].error


# ── exception containment ──────────────────────────────────────────────────

def test_exception_containment_broken_mapping(specs):
    class BrokenArgs(dict):
        def items(self):
            raise RuntimeError("no items")

    results = parse_tool_calls(
        {"id": "c1", "name": "add", "arguments": BrokenArgs({"a": 1, "b": 2})},
        specs,
    )
    assert len(results) == 1
    assert results[0].call_id == "c1"
    assert not results[0].ok
    assert results[0].name == NO_ACTION_NAME
    assert results[0].error.startswith("internal error")


def test_exception_containment_never_raises(specs):
    class Evil:
        def __iter__(self):
            raise RuntimeError("iter boom")

    assert isinstance(parse_tool_calls(Evil(), specs), tuple)

    class EvilMap(dict):
        def __getitem__(self, key):
            raise RuntimeError("get boom")

    results = parse_tool_calls(EvilMap(id="c1", name="add", arguments={}), specs)
    assert isinstance(results, tuple)
    assert all(isinstance(r, ToolResult) for r in results)


# ── id salvage / never orphan ──────────────────────────────────────────────

def test_salvages_ids_from_unparseable_text(specs):
    text = 'response nonsense "id": "call_99" with no real json structure [[['
    results = parse_tool_calls(text, specs)
    assert len(results) == 1
    assert results[0].call_id == "call_99"
    assert not results[0].ok
    assert results[0].name == NO_ACTION_NAME


def test_salvage_does_not_duplicate_parsed_ids(specs):
    # A valid call plus a stray id in trailing prose: both answered, no dupes.
    text = '{"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}} trailing "id": "c2"'
    results = parse_tool_calls(text, specs)
    ids = [r.call_id for r in results]
    assert ids == ["c1", "c2"]
    assert len(ids) == len(set(ids))
    assert results[0].ok
    assert not results[1].ok


def test_nested_argument_id_never_becomes_synthetic_call_id(specs):
    """A nested argument field named ``id`` inside a decoded JSON block must
    never produce a synthetic call id — only the top-level call id counts."""
    text = '{"id": "call_1", "name": "add", "arguments": {"id": "nested", "a": 1, "b": 2}}'
    results = parse_tool_calls(text, specs)
    assert [r.call_id for r in results] == ["call_1"]
    assert results[0].ok
    assert results[0].data == {"id": "nested", "a": 1, "b": 2}

    # same for a tool_calls wrapper with nested ids in every call
    text = (
        '{"tool_calls": ['
        '{"id": "c1", "name": "add", "arguments": {"id": "x", "a": 1, "b": 2}},'
        '{"id": "c2", "name": "add", "arguments": {"id": "y", "a": 3, "b": 4}}'
        "]}"
    )
    results = parse_tool_calls(text, specs)
    assert [r.call_id for r in results] == ["c1", "c2"]
    assert all(r.ok for r in results)


def test_nested_argument_id_in_prose_still_salvaged(specs):
    """An id in PROSE (outside the JSON block) is still a referenced call id
    and is answered — only ids inside a decoded block are ignored."""
    text = '{"id": "c1", "name": "add", "arguments": {"id": "nested", "a": 1, "b": 2}} then "id": "c9"'
    results = parse_tool_calls(text, specs)
    assert [r.call_id for r in results] == ["c1", "c9"]
    assert results[0].ok
    assert not results[1].ok


def test_raw_arguments_toolcall_reaches_tolerant_path(specs):
    """A ToolCall carrying raw (unparseable) arguments runs the tolerant
    one-repair pipeline and degrades to an ok=False result, never raising."""
    call = ToolCall(
        id=ToolCallId("c1"),
        name="add",
        arguments={},
        raw_arguments="not json at all",
    )
    results = parse_tool_calls(call, specs)
    assert len(results) == 1
    assert results[0].call_id == "c1"
    assert not results[0].ok
    assert results[0].name == "add"
    assert results[0].error.startswith("malformed arguments")

    # a raw string that the repair pass CAN fix is recovered
    call2 = ToolCall(
        id=ToolCallId("c2"),
        name="add",
        arguments={},
        raw_arguments='{"a": 1, "b": 2,}',
    )
    results2 = parse_tool_calls(call2, specs)
    assert results2[0].ok
    assert results2[0].data == {"a": 1, "b": 2}

    # a non-object raw value (list) degrades with the id preserved
    call3 = ToolCall(
        id=ToolCallId("c3"),
        name="add",
        arguments={},
        raw_arguments=[1, 2],
    )
    results3 = parse_tool_calls(call3, specs)
    assert results3[0].call_id == "c3"
    assert not results3[0].ok
    assert "malformed arguments" in results3[0].error


# ── purity: no handler / network I/O ───────────────────────────────────────

def test_handlers_never_invoked():
    called = []

    @tool("tracked")
    def tracked(x: int) -> str:
        called.append(x)
        return "ran"

    results = parse_tool_calls(_call("c1", "tracked", {"x": 1}), [tracked])
    assert results[0].ok
    assert called == []


def test_no_network_io(specs, monkeypatch):
    import socket

    def deny(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    results = parse_tool_calls(
        '{"tool_calls": [{"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}}]}',
        specs,
    )
    assert results[0].ok


# ── tolerant_loads / validate_arguments units ──────────────────────────────

def test_tolerant_loads_fixes_common_mistakes():
    assert tolerant_loads('```json\n{"a": 1}\n```') == {"a": 1}
    assert tolerant_loads("{'a': 1,}") == {"a": 1}
    assert tolerant_loads("{a: 1}") == {"a": 1}
    assert tolerant_loads('{"a": [1, 2,]}') == {"a": [1, 2]}
    assert tolerant_loads('prose {"b": true} trailing') == {"b": True}
    assert tolerant_loads('{"a": undefined}') == {"a": None}
    import math

    nan_inf = tolerant_loads('{"a": NaN, "b": Infinity}')
    assert math.isnan(nan_inf["a"]) and math.isinf(nan_inf["b"])
    with pytest.raises(ValueError):
        tolerant_loads("no json here")


def test_validate_arguments_direct():
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "flag": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "mode": {"enum": ["a", "b"]},
            "note": {"type": ["string", "null"]},
        },
        "required": ["city"],
    }
    assert validate_arguments({"city": "SF"}, schema) == []
    assert validate_arguments(
        {"city": "SF", "tags": ["x"], "mode": "a", "note": None}, schema
    ) == []
    assert any("city" in e for e in validate_arguments({}, schema))
    assert any("string" in e for e in validate_arguments({"city": 5}, schema))
    assert validate_arguments({"city": "SF", "flag": 1}, schema) != []
    assert validate_arguments({"city": "SF", "count": True}, schema) != []
    assert validate_arguments({"city": "SF", "mode": "z"}, schema) != []
    assert validate_arguments({"anything": object()}, {}) == []


# ── property-style randomized malformed inputs ─────────────────────────────

def test_property_random_strings_never_orphan(specs):
    rng = random.Random(20260824)
    alphabet = string.ascii_letters + string.digits + '{}[],:"\'` \t\n.-_/*\\'
    for _ in range(500):
        n = rng.randint(0, 120)
        text = "".join(rng.choice(alphabet) for _ in range(n))
        results = parse_tool_calls(text, specs)
        ids = [r.call_id for r in results]
        assert len(ids) == len(set(ids)), (text, results)
        assert all(isinstance(r, ToolResult) for r in results)
        # every id referenced in the PROSE (outside a decoded JSON block) is
        # answered exactly once; ids inside a decoded block (e.g. nested
        # argument fields named "id") never become synthetic call ids
        for cid in _ids_outside_json_blocks(text):
            assert ids.count(cid) == 1, (text, results, cid)


def test_property_random_structured_inputs_never_orphan(specs):
    rng = random.Random(99)
    names = ["add", "get_weather", "ghost", "noop"]
    arg_choices = [
        {"a": 1, "b": 2},
        {"a": "x", "b": 1},
        {},
        None,
        "garbage",
        42,
        {"city": "SF"},
    ]
    for _ in range(300):
        n_calls = rng.randint(0, 6)
        calls = []
        toolcall_ids = []
        all_ids = []
        for j in range(n_calls):
            cid = rng.choice(["call_%d" % j, "c-%d" % j, "id%d" % j, None])
            name = rng.choice(names)
            args = rng.choice(arg_choices)
            if cid is not None:
                all_ids.append(cid)
            kind = rng.random()
            if kind < 0.4:
                calls.append({"id": cid, "name": name, "arguments": args})
            elif kind < 0.7:
                tc = ToolCall(
                    id=ToolCallId(cid or "call_%d" % j),
                    name=name,
                    arguments=args if isinstance(args, dict) else {},
                )
                calls.append(tc)
                if cid is not None:
                    toolcall_ids.append(cid)
            else:
                calls.append(json.dumps({"id": cid, "name": name, "arguments": args}))
        source_kind = rng.choice(["list", "wrapper", "llm"])
        if source_kind == "list":
            source = calls
            expected_ids = all_ids
        elif source_kind == "wrapper":
            source = {"tool_calls": calls}
            expected_ids = all_ids
        else:
            source = LLMResponse(
                content=None,
                tool_calls=tuple(c for c in calls if isinstance(c, ToolCall)),
            )
            expected_ids = toolcall_ids
        results = parse_tool_calls(source, specs)
        result_ids = [r.call_id for r in results]
        assert len(result_ids) == len(set(result_ids)), (source, results)
        for cid in expected_ids:
            assert result_ids.count(cid) == 1, (source, results, cid)
