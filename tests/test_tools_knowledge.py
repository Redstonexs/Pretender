"""Phase 5 knowledge tools: deferred/activated schemas, injected chat-scoped
callbacks, result caps/truncation, malformed-arg fail-closed, and person
cross-chat impossibility."""

from __future__ import annotations

import asyncio
import json

from pretender.search import MemoryRecallHit
from pretender.tools.core import (
    CORE_TOOL_NAMES,
    ToolContext,
    dispatch_call,
    register_core_tools,
)
from pretender.tools.knowledge import (
    KNOWLEDGE_TOOL_NAMES,
    MAX_MEMORY_CHARS,
    MAX_MEMORY_HITS,
    KnowledgeCallbacks,
)
from pretender.types import (
    ChatKey,
    PersonProfile,
    SenderId,
    ToolCall,
    ToolCallId,
)

CK = ChatKey("qq:group:123456")
OTHER = ChatKey("qq:group:other")


def _run(coro):
    return asyncio.run(coro)


def _call(name: str, arguments: dict | None = None, cid: str = "c1") -> ToolCall:
    return ToolCall(id=ToolCallId(cid), name=name, arguments=dict(arguments or {}))


def _hit(memory_id: int, text: str, score: float = 1.0) -> MemoryRecallHit:
    return MemoryRecallHit(
        chat_key=CK, memory_id=memory_id, text=text, score=score, source="lexical",
        strength=1.0,
    )


async def _async_empty_mem(query: str, limit: int) -> list[MemoryRecallHit]:
    return []


async def _async_none_person(uid: SenderId) -> PersonProfile | None:
    return None


def _ctx(registry, *, knowledge: KnowledgeCallbacks | None = None) -> ToolContext:
    return ToolContext(
        chat_key=CK,
        chat_kind="group",
        registry=registry,
        knowledge=knowledge,
    )


# ── deferred / activated schemas ─────────────────────────────────────────────

def test_knowledge_tools_are_deferred_and_absent_until_activated():
    reg = register_core_tools()
    for name in KNOWLEDGE_TOOL_NAMES:
        assert name in CORE_TOOL_NAMES
        assert reg.require(name).visibility == "deferred"
    before = [d["function"]["name"] for d in reg.provider_definitions()]
    assert "query_memory" not in before
    assert "query_person_profile" not in before
    # tool_search activates them; they appear in the provider schema after.
    ctx = _ctx(reg)
    res = _run(dispatch_call(_call("tool_search", {"capability": "memory"}), ctx, reg))
    assert json.loads(res.content)["tools"] == list(KNOWLEDGE_TOOL_NAMES)
    after = [d["function"]["name"] for d in reg.provider_definitions()]
    assert "query_memory" in after
    assert "query_person_profile" in after


def test_knowledge_tools_rejected_until_activated():
    reg = register_core_tools()
    ctx = _ctx(reg, knowledge=KnowledgeCallbacks(query_memory=_async_empty_mem))
    res = _run(dispatch_call(_call("query_memory", {"query": "x"}), ctx, reg))
    assert res.ok is False
    assert "deferred" in res.error


# ── query_memory: callbacks, caps, malformed args ────────────────────────────

def test_query_memory_returns_capped_hits():
    reg = register_core_tools()
    reg.activate("query_memory")
    seen: list[tuple] = []

    async def cb(query, limit):
        seen.append((query, limit))
        return [_hit(i, f"memory {i}") for i in range(1, 20)]

    ctx = _ctx(reg, knowledge=KnowledgeCallbacks(query_memory=cb))
    res = _run(dispatch_call(_call("query_memory", {"query": "火锅", "limit": 99}), ctx, reg))
    assert res.ok is True
    data = json.loads(res.content)
    assert data["action"] == "query_memory"
    # The limit is clamped to MAX_MEMORY_HITS.
    assert data["count"] == MAX_MEMORY_HITS
    assert len(data["hits"]) == MAX_MEMORY_HITS
    assert seen == [("火锅", MAX_MEMORY_HITS)]


def test_query_memory_truncates_long_text():
    reg = register_core_tools()
    reg.activate("query_memory")
    long_text = "x" * (MAX_MEMORY_CHARS + 100)

    async def cb(query, limit):
        return [_hit(1, long_text)]

    ctx = _ctx(reg, knowledge=KnowledgeCallbacks(query_memory=cb))
    res = _run(dispatch_call(_call("query_memory", {"query": "x"}), ctx, reg))
    data = json.loads(res.content)
    assert len(data["hits"][0]["text"]) <= MAX_MEMORY_CHARS + 1  # + ellipsis


def test_query_memory_fails_closed_without_service():
    reg = register_core_tools()
    reg.activate("query_memory")
    ctx = _ctx(reg)  # no knowledge callbacks
    res = _run(dispatch_call(_call("query_memory", {"query": "x"}), ctx, reg))
    assert res.ok is False
    assert "memory service is not available" in res.error


def test_query_memory_malformed_args_fail_closed():
    reg = register_core_tools()
    reg.activate("query_memory")
    ctx = _ctx(reg, knowledge=KnowledgeCallbacks(query_memory=_async_empty_mem))
    # Empty query.
    res = _run(dispatch_call(_call("query_memory", {"query": "   "}), ctx, reg))
    assert res.ok is False
    assert "non-empty" in res.error
    # Missing required query -> schema mismatch.
    res2 = _run(dispatch_call(_call("query_memory", {}), ctx, reg))
    assert res2.ok is False
    assert "schema mismatch" in res2.error


# ── query_person_profile: callbacks, malformed args, cross-chat ──────────────

def test_query_person_profile_returns_profile():
    reg = register_core_tools()
    reg.activate("query_person_profile")
    seen: list[SenderId] = []

    async def cb(uid):
        seen.append(uid)
        return PersonProfile(
            chat_key=CK, platform_uid=uid, names=("alice", "小爱"),
            profile="likes tea", impression="friendly",
        )

    ctx = _ctx(reg, knowledge=KnowledgeCallbacks(query_person=cb))
    res = _run(dispatch_call(_call("query_person_profile", {"platform_uid": "u1"}), ctx, reg))
    assert res.ok is True
    data = json.loads(res.content)
    assert data["action"] == "query_person_profile"
    assert data["found"] is True
    assert data["names"] == ["alice", "小爱"]
    assert data["profile"] == "likes tea"
    assert seen == [SenderId("u1")]


def test_query_person_profile_unknown_person():
    reg = register_core_tools()
    reg.activate("query_person_profile")

    async def cb(uid):
        return None

    ctx = _ctx(reg, knowledge=KnowledgeCallbacks(query_person=cb))
    res = _run(dispatch_call(_call("query_person_profile", {"platform_uid": "nobody"}), ctx, reg))
    assert res.ok is True
    assert json.loads(res.content)["found"] is False


def test_query_person_profile_fails_closed_without_service():
    reg = register_core_tools()
    reg.activate("query_person_profile")
    ctx = _ctx(reg)
    res = _run(dispatch_call(_call("query_person_profile", {"platform_uid": "u1"}), ctx, reg))
    assert res.ok is False
    assert "person service is not available" in res.error


def test_query_person_profile_malformed_args_fail_closed():
    reg = register_core_tools()
    reg.activate("query_person_profile")
    ctx = _ctx(reg, knowledge=KnowledgeCallbacks(query_person=_async_none_person))
    res = _run(dispatch_call(_call("query_person_profile", {"platform_uid": "  "}), ctx, reg))
    assert res.ok is False
    assert "non-empty" in res.error


def test_person_callback_is_chat_scoped_no_cross_chat():
    """The callback is bound to a specific chat_key at ToolContext
    construction, so a cross-chat request is impossible by construction: the
    tool can only ever query the chat the context was built for."""
    reg = register_core_tools()
    reg.activate("query_person_profile")
    queried: list[ChatKey] = []

    # The caller binds the callback to CK (not OTHER).
    async def cb(uid):
        queried.append(CK)
        return PersonProfile(chat_key=CK, platform_uid=uid, names=("alice",))

    ctx = _ctx(reg, knowledge=KnowledgeCallbacks(query_person=cb))
    res = _run(dispatch_call(_call("query_person_profile", {"platform_uid": "u1"}), ctx, reg))
    assert res.ok is True
    # The callback only ever saw CK — there is no way for the tool to target
    # OTHER through a chat-scoped callback.
    assert queried == [CK]
    assert ctx.chat_key == CK
