"""Core tools: dispatch, gating, deferred activation, and the local runtime
callback context (Phase 3, PLAN.md §2 tools/core.py).

Covers: dispatch + JSON-Schema validation; visibility / chat-scope /
adapter-capability gating; deferred-tool activation via ``tool_search``;
``reply``/``wait``/``no_action`` structured results; ``fetch_history`` /
``view_forward_message`` success/failure/limit; handler exception
containment; and no mutation of any input structure.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from pretender.errors import RegistryError, ToolError
from pretender.tools import ToolRegistry, ToolSpec, tool
from pretender.tools.core import (
    CORE_TOOL_NAMES,
    CoreToolRegistry,
    ToolContext,
    dispatch_call,
    register_core_tools,
)
from pretender.types import (
    ChatKey,
    Message,
    MessageId,
    SenderId,
    ToolCall,
    ToolCallId,
    TranscriptMessage,
)

CK = ChatKey("qq:group:123456")


def _msg(
    text: str,
    *,
    sender: str = "user",
    is_self: bool = False,
    msg_id: str = "m1",
) -> Message:
    return Message(
        chat_key=CK,
        sender_id=SenderId(sender),
        sender_name=sender,
        is_self=is_self,
        text=text,
        id=MessageId(msg_id),
    )


def _call(name: str, arguments: dict[str, Any] | None = None, cid: str = "c1") -> ToolCall:
    return ToolCall(
        id=ToolCallId(cid), name=name, arguments=dict(arguments or {})
    )


def _ctx(
    *,
    chat_kind: str = "group",
    capabilities: frozenset[str] = frozenset({"history", "forward"}),
    registry: ToolRegistry | None = None,
    recent: tuple[Message, ...] = (),
    forwards: dict[str, str] | None = None,
    self_name: str | None = None,
) -> ToolContext:
    return ToolContext(
        chat_key=CK,
        chat_kind=chat_kind,
        capabilities=capabilities,
        registry=registry,
        recent=recent,
        forwards=forwards,
        self_name=self_name,
    )


def _run(coro):
    return asyncio.run(coro)


# ── registration / schema ───────────────────────────────────────────────────

def test_core_tools_register_in_deterministic_order():
    reg = register_core_tools()
    assert reg.names() == CORE_TOOL_NAMES
    assert len(reg) == 13  # six core + two knowledge + jargon + two media + two chatctl


def test_core_tool_schemas_are_clean():
    reg = register_core_tools()
    reply = reg.require("reply")
    # the context param (self) is skipped by schema derivation
    assert reply.parameters == {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "reply_to": {"type": ["string", "null"]},
        },
        "required": ["text"],
    }
    wait = reg.require("wait")
    assert wait.parameters["required"] == ["seconds"]
    assert wait.parameters["properties"]["seconds"] == {"type": "number"}
    no_action = reg.require("no_action")
    assert "required" not in no_action.parameters  # reason is optional
    assert reg.require("fetch_history").visibility == "deferred"
    assert reg.require("view_forward_message").visibility == "deferred"
    assert reg.require("view_forward_message").chat_scope == "group"
    # Phase 5 knowledge tools: deferred, capability-gated, clean schemas.
    assert reg.require("query_memory").visibility == "deferred"
    assert reg.require("query_memory").capability == "memory"
    assert reg.require("query_memory").parameters["required"] == ["query"]
    assert reg.require("query_person_profile").visibility == "deferred"
    assert reg.require("query_person_profile").capability == "memory"
    assert reg.require("query_person_profile").parameters["required"] == ["platform_uid"]


def test_register_core_tools_twice_rejects_duplicates():
    reg = register_core_tools()
    with pytest.raises(RegistryError, match="already registered"):
        register_core_tools(reg)


def test_core_tool_descriptions_present():
    reg = register_core_tools()
    for name in CORE_TOOL_NAMES:
        assert reg.require(name).description.strip()


# ── dispatch / schema validation ────────────────────────────────────────────

def test_dispatch_unknown_tool_fails():
    reg = register_core_tools()
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("nope"), ctx, reg))
    assert res.ok is False
    assert res.name == "nope"
    assert "unknown tool" in res.error


def test_dispatch_valid_reply_ok():
    reg = register_core_tools()
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("reply", {"text": "hi"}), ctx, reg))
    assert res.ok is True
    assert res.name == "reply"
    assert res.call_id == ToolCallId("c1")
    assert json.loads(res.content) == {"action": "reply", "text": "hi", "reply_to": None}
    assert ctx.reply_text == "hi"


def test_dispatch_schema_mismatch_missing_required():
    reg = register_core_tools()
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("reply", {}), ctx, reg))
    assert res.ok is False
    assert "schema mismatch" in res.error
    assert "text" in res.error


def test_dispatch_schema_mismatch_wrong_type():
    reg = register_core_tools()
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("wait", {"seconds": "five"}), ctx, reg))
    assert res.ok is False
    assert "schema mismatch" in res.error


def test_dispatch_accepts_mapping_call():
    reg = register_core_tools()
    ctx = _ctx(registry=reg)
    res = _run(
        dispatch_call({"id": "c9", "name": "reply", "arguments": {"text": "yo"}}, ctx, reg)
    )
    assert res.ok is True
    assert res.call_id == ToolCallId("c9")


# ── visibility / chat-scope / capability gating ─────────────────────────────

def test_deferred_tool_rejected_until_activated():
    reg = register_core_tools()
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("fetch_history", {"limit": 5}), ctx, reg))
    assert res.ok is False
    assert "deferred" in res.error
    assert "tool_search" in res.error


def test_hidden_tool_never_dispatches():
    reg = CoreToolRegistry()
    reg.register(tool("secret", visibility="hidden")(lambda self, x: "ok"))
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("secret", {"x": 1}), ctx, reg))
    assert res.ok is False
    assert "hidden" in res.error


def test_chat_scope_gate_private_chat():
    reg = register_core_tools()
    # view_forward_message is chat_scope="group"; force a direct dispatch in
    # a private chat (defense in depth — the schema already excludes it).
    reg.activate("view_forward_message")
    ctx = _ctx(chat_kind="private", registry=reg, forwards={"f1": "content"})
    res = _run(dispatch_call(_call("view_forward_message", {"id": "f1"}), ctx, reg))
    assert res.ok is False
    assert "not available in private chats" in res.error


def test_capability_gate_history_missing():
    reg = register_core_tools()
    reg.activate("fetch_history")
    ctx = _ctx(registry=reg, capabilities=frozenset({"forward"}))
    res = _run(dispatch_call(_call("fetch_history", {"limit": 5}), ctx, reg))
    assert res.ok is False
    assert "history" in res.error


def test_capability_gate_forward_missing():
    reg = register_core_tools()
    reg.activate("view_forward_message")
    ctx = _ctx(registry=reg, capabilities=frozenset({"history"}))
    res = _run(dispatch_call(_call("view_forward_message", {"id": "f1"}), ctx, reg))
    assert res.ok is False
    assert "forward" in res.error


def test_capability_gate_passes_when_present():
    reg = register_core_tools()
    reg.activate("fetch_history")
    ctx = _ctx(registry=reg, capabilities=frozenset({"history"}))
    res = _run(dispatch_call(_call("fetch_history", {"limit": 5}), ctx, reg))
    assert res.ok is True


# ── deferred activation via tool_search ─────────────────────────────────────

def test_tool_search_activates_deferred_by_name():
    reg = register_core_tools()
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("tool_search", {"query": "history"}), ctx, reg))
    assert res.ok is True
    matched = json.loads(res.content)["tools"]
    assert "fetch_history" in matched
    assert reg.is_activated("fetch_history")
    # now dispatchable
    res2 = _run(dispatch_call(_call("fetch_history", {"limit": 5}), ctx, reg))
    assert res2.ok is True


def test_tool_search_activates_by_capability_tag():
    reg = register_core_tools()
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("tool_search", {"capability": "forward"}), ctx, reg))
    matched = json.loads(res.content)["tools"]
    assert matched == ["view_forward_message"]
    assert reg.is_activated("view_forward_message")
    assert not reg.is_activated("fetch_history")


def test_tool_search_no_match_activates_nothing():
    reg = register_core_tools()
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("tool_search", {"query": "zzzznope"}), ctx, reg))
    assert json.loads(res.content)["tools"] == []
    assert reg.activated_names() == ()


def test_tool_search_activation_updates_provider_definitions():
    reg = register_core_tools()
    before = [d["function"]["name"] for d in reg.provider_definitions()]
    assert "fetch_history" not in before
    ctx = _ctx(registry=reg)
    _run(dispatch_call(_call("tool_search", {"query": "history"}), ctx, reg))
    after = [d["function"]["name"] for d in reg.provider_definitions()]
    assert "fetch_history" in after
    # group scope now also exposes the group-scoped forward tool once activated
    _run(dispatch_call(_call("tool_search", {"capability": "forward"}), ctx, reg))
    group = [d["function"]["name"] for d in reg.provider_definitions(scope="group")]
    assert "view_forward_message" in group
    # ...but not in a private scope (chat_scope="group")
    private = [d["function"]["name"] for d in reg.provider_definitions(scope="private")]
    assert "view_forward_message" not in private


def test_activate_ignores_visible_and_unknown():
    reg = register_core_tools()
    assert reg.activate("reply") is False  # visible
    assert reg.activate("nope") is False  # unknown
    assert reg.activate("fetch_history") is True
    assert reg.activate("fetch_history") is False  # idempotent


# ── reply / wait / no_action ────────────────────────────────────────────────

def test_reply_last_wins_and_clears_wait():
    reg = register_core_tools()
    ctx = _ctx(registry=reg)
    _run(dispatch_call(_call("wait", {"seconds": 5}), ctx, reg))
    assert ctx.wait_seconds == 5.0
    _run(dispatch_call(_call("reply", {"text": "first"}), ctx, reg))
    _run(dispatch_call(_call("reply", {"text": "second", "reply_to": "m9"}), ctx, reg))
    assert ctx.reply_text == "second"
    assert ctx.reply_to == "m9"
    assert ctx.wait_seconds is None  # reply cleared the wait
    assert ctx.no_action_verdict is False


def test_reply_empty_text_fails():
    reg = register_core_tools()
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("reply", {"text": "   "}), ctx, reg))
    assert res.ok is False
    assert "non-empty" in res.error
    assert ctx.reply_text is None


def test_wait_structured_and_clears_reply():
    reg = register_core_tools()
    ctx = _ctx(registry=reg)
    _run(dispatch_call(_call("reply", {"text": "hi"}), ctx, reg))
    res = _run(dispatch_call(_call("wait", {"seconds": 3}), ctx, reg))
    assert res.ok is True
    assert json.loads(res.content) == {"action": "wait", "seconds": 3.0}
    assert ctx.wait_seconds == 3.0
    assert ctx.reply_text is None


def test_wait_rejects_non_positive_and_huge():
    reg = register_core_tools()
    ctx = _ctx(registry=reg)
    for bad in (0, -1, 1e9):
        res = _run(dispatch_call(_call("wait", {"seconds": bad}), ctx, reg))
        assert res.ok is False
        assert ctx.wait_seconds is None


def test_no_action_structured_and_clears_reply():
    reg = register_core_tools()
    ctx = _ctx(registry=reg)
    _run(dispatch_call(_call("reply", {"text": "hi"}), ctx, reg))
    res = _run(dispatch_call(_call("no_action", {"reason": "nothing to add"}), ctx, reg))
    assert res.ok is True
    assert json.loads(res.content) == {
        "action": "no_action",
        "reason": "nothing to add",
    }
    assert ctx.no_action_verdict is True
    assert ctx.reply_text is None
    assert ctx.wait_seconds is None


def test_no_action_without_reason():
    reg = register_core_tools()
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("no_action"), ctx, reg))
    assert res.ok is True
    assert json.loads(res.content) == {"action": "no_action", "reason": None}


# ── fetch_history / view_forward_message ────────────────────────────────────

def test_fetch_history_renders_bounded_newest_first():
    reg = register_core_tools()
    reg.activate("fetch_history")
    recent = (
        _msg("newest", sender="c", msg_id="m3"),
        _msg("middle", sender="b", msg_id="m2"),
        _msg("oldest", sender="a", msg_id="m1"),
    )
    ctx = _ctx(registry=reg, recent=recent)
    res = _run(dispatch_call(_call("fetch_history", {"limit": 2}), ctx, reg))
    data = json.loads(res.content)
    assert data["action"] == "fetch_history"
    assert data["count"] == 2
    assert data["messages"] == ["c: newest", "b: middle"]  # newest first


def test_fetch_history_limit_clamped():
    reg = register_core_tools()
    reg.activate("fetch_history")
    recent = tuple(_msg(f"m{i}", sender="u", msg_id=f"m{i}") for i in range(100))
    ctx = _ctx(registry=reg, recent=recent)
    res = _run(dispatch_call(_call("fetch_history", {"limit": 500}), ctx, reg))
    data = json.loads(res.content)
    assert data["count"] == ToolContext.MAX_HISTORY_LIMIT


def test_fetch_history_empty():
    reg = register_core_tools()
    reg.activate("fetch_history")
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("fetch_history"), ctx, reg))
    data = json.loads(res.content)
    assert data["count"] == 0
    assert data["messages"] == []


def test_fetch_history_self_messages_use_bot_name():
    reg = register_core_tools()
    reg.activate("fetch_history")
    recent = (_msg("hello", sender="user", msg_id="m1"), _msg("hi back", sender="bot", is_self=True, msg_id="m2"))
    ctx = _ctx(registry=reg, recent=recent, self_name="麦麦")
    res = _run(dispatch_call(_call("fetch_history", {"limit": 10}), ctx, reg))
    data = json.loads(res.content)
    assert data["messages"] == ["user: hello", "麦麦: hi back"]


def test_view_forward_message_success():
    reg = register_core_tools()
    reg.activate("view_forward_message")
    ctx = _ctx(registry=reg, forwards={"f1": "forwarded content"})
    res = _run(dispatch_call(_call("view_forward_message", {"id": "f1"}), ctx, reg))
    assert res.ok is True
    assert json.loads(res.content) == {
        "action": "view_forward_message",
        "id": "f1",
        "content": "forwarded content",
    }


def test_view_forward_message_unknown_id_fails_safely():
    reg = register_core_tools()
    reg.activate("view_forward_message")
    ctx = _ctx(registry=reg, forwards={"f1": "content"})
    res = _run(dispatch_call(_call("view_forward_message", {"id": "f2"}), ctx, reg))
    assert res.ok is False
    assert "unknown forwarded message" in res.error


# ── handler exception containment ───────────────────────────────────────────

def test_raising_handler_contained():
    reg = CoreToolRegistry()

    def boom(self, x: int) -> str:
        raise RuntimeError("kaboom")

    reg.register(tool("boom")(boom))
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("boom", {"x": 1}), ctx, reg))
    assert res.ok is False
    assert "RuntimeError" in res.error
    assert "kaboom" in res.error


def test_toolerror_handler_keeps_message():
    reg = CoreToolRegistry()

    def bad(self, x: int) -> str:
        raise ToolError("deliberate tool failure")

    reg.register(tool("bad")(bad))
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("bad", {"x": 1}), ctx, reg))
    assert res.ok is False
    assert res.error == "deliberate tool failure"


def test_async_raising_handler_contained():
    reg = CoreToolRegistry()

    async def aboom(self, x: int) -> str:
        raise ValueError("async boom")

    reg.register(tool("aboom")(aboom))
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("aboom", {"x": 1}), ctx, reg))
    assert res.ok is False
    assert "ValueError" in res.error


def test_async_handler_awaited():
    reg = CoreToolRegistry()

    async def aok(self, x: int) -> str:
        return f"got {x}"

    reg.register(tool("aok")(aok))
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("aok", {"x": 7}), ctx, reg))
    assert res.ok is True
    assert res.content == "got 7"


# ── no transcript / input mutation ──────────────────────────────────────────

def test_dispatch_does_not_mutate_inputs():
    reg = register_core_tools()
    recent = (_msg("hello", sender="user", msg_id="m1"),)
    forwards = {"f1": "content"}
    ctx = _ctx(registry=reg, recent=recent, forwards=forwards)
    transcript = [
        TranscriptMessage(role="user", content="hi"),
        TranscriptMessage(role="assistant", content="", tool_calls=(_call("reply", {"text": "yo"}),)),
    ]
    transcript_before = list(transcript)
    recent_before = list(recent)
    forwards_before = dict(forwards)

    _run(dispatch_call(_call("reply", {"text": "yo"}), ctx, reg))
    _run(dispatch_call(_call("fetch_history", {"limit": 5}), ctx, reg))

    assert transcript == transcript_before
    assert list(ctx.recent) == recent_before
    assert dict(ctx.forwards) == forwards_before
    # the ToolCall's arguments dict is not mutated by dispatch
    call = _call("reply", {"text": "yo"})
    _run(dispatch_call(call, ctx, reg))
    assert call.arguments == {"text": "yo"}


# ── timeout / rate-limit enforcement ─────────────────────────────────────────

def test_dispatch_enforces_timeout_on_async_handler():
    reg = CoreToolRegistry()

    async def slow(self, x: int) -> str:
        await asyncio.sleep(5)
        return "too late"

    reg.register(tool("slow", timeout_s=0.05)(slow))
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("slow", {"x": 1}), ctx, reg))
    assert res.ok is False
    assert "timed out" in res.error
    assert "0.05" in res.error


def test_dispatch_timeout_does_not_affect_fast_handlers():
    reg = CoreToolRegistry()

    async def quick(self, x: int) -> str:
        return f"got {x}"

    reg.register(tool("quick", timeout_s=5)(quick))
    ctx = _ctx(registry=reg)
    res = _run(dispatch_call(_call("quick", {"x": 7}), ctx, reg))
    assert res.ok is True
    assert res.content == "got 7"


def test_dispatch_enforces_rate_limit_deterministically():
    reg = CoreToolRegistry()
    calls = {"n": 0}

    def limited(self, x: int) -> str:
        calls["n"] += 1
        return "ran"

    reg.register(tool("limited", rate_limit=2)(limited))
    ctx = _ctx(registry=reg)
    now = [1000.0]

    def clock():
        return now[0]

    # two calls within the window are allowed, the third is rate-limited
    r1 = _run(dispatch_call(_call("limited", {"x": 1}), ctx, reg, clock=clock))
    r2 = _run(dispatch_call(_call("limited", {"x": 2}), ctx, reg, clock=clock))
    r3 = _run(dispatch_call(_call("limited", {"x": 3}), ctx, reg, clock=clock))
    assert r1.ok and r2.ok
    assert r3.ok is False
    assert "rate limit" in r3.error
    assert calls["n"] == 2  # the limited call never executed

    # after the 60s window rolls over, calls are allowed again
    now[0] += 61.0
    r4 = _run(dispatch_call(_call("limited", {"x": 4}), ctx, reg, clock=clock))
    assert r4.ok is True
    assert calls["n"] == 3


def test_dispatch_rate_limit_is_per_tool():
    """Rate accounting is per tool name: one tool's cap never affects
    another's."""
    reg = CoreToolRegistry()
    calls = {"a": 0, "b": 0}

    def make(name):
        def handler(self, x: int) -> str:
            calls[name] += 1
            return "ran"

        return handler

    reg.register(tool("tool_a", rate_limit=1)(make("a")))
    reg.register(tool("tool_b", rate_limit=2)(make("b")))
    ctx = _ctx(registry=reg)
    now = [1000.0]

    def clock():
        return now[0]

    assert _run(dispatch_call(_call("tool_a", {"x": 1}), ctx, reg, clock=clock)).ok
    assert _run(dispatch_call(_call("tool_b", {"x": 1}), ctx, reg, clock=clock)).ok
    # tool_a is exhausted (cap 1); tool_b still has its own budget (cap 2)
    ra = _run(dispatch_call(_call("tool_a", {"x": 2}), ctx, reg, clock=clock))
    rb = _run(dispatch_call(_call("tool_b", {"x": 2}), ctx, reg, clock=clock))
    assert ra.ok is False
    assert rb.ok is True
    assert calls == {"a": 1, "b": 2}


def test_dispatch_unlimited_tools_never_rate_limited():
    reg = CoreToolRegistry()
    calls = {"n": 0}

    def free(self, x: int) -> str:
        calls["n"] += 1
        return "ran"

    reg.register(tool("free")(free))  # no rate_limit
    ctx = _ctx(registry=reg)
    for i in range(50):
        res = _run(dispatch_call(_call("free", {"x": i}), ctx, reg))
        assert res.ok is True
    assert calls["n"] == 50
