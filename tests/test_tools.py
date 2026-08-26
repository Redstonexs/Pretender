"""Tool contracts: @tool schema derivation, docstring description, and
ToolRegistry duplicate/capability/provider behavior."""

from __future__ import annotations

from typing import Any, Literal, Optional

import pytest

from pretender.errors import RegistryError, ToolError
from pretender.tools import ToolRegistry, ToolSpec, tool
from pretender.types import ChatKey


# ── @tool schema derivation ─────────────────────────────────────────────────

def test_schema_string_number_bool():
    @tool("mix")
    def mix(name: str, count: int, ratio: float, enabled: bool) -> str:
        """Do a mix."""
        return ""

    spec = mix
    assert isinstance(spec, ToolSpec)
    assert spec.name == "mix"
    props = spec.parameters["properties"]
    assert props["name"] == {"type": "string"}
    assert props["count"] == {"type": "integer"}
    assert props["ratio"] == {"type": "number"}
    assert props["enabled"] == {"type": "boolean"}
    assert spec.parameters["required"] == ["name", "count", "ratio", "enabled"]


def test_schema_optional_and_defaults():
    @tool()
    def f(name: str, count: int = 5, note: Optional[str] = None) -> str:
        return ""

    spec = f
    assert spec.parameters["required"] == ["name"]
    props = spec.parameters["properties"]
    assert props["count"] == {"type": "integer"}  # default, not required
    assert props["note"]["type"] == ["string", "null"]  # Optional -> nullable


def test_schema_array_and_object():
    @tool()
    def f(tags: list[str], meta: dict[str, int]) -> str:
        return ""

    spec = f
    props = spec.parameters["properties"]
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["meta"] == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }


def test_schema_literal_and_newtype():
    @tool()
    def f(mode: Literal["a", "b"], key: ChatKey) -> str:
        return ""

    spec = f
    props = spec.parameters["properties"]
    assert props["mode"] == {"enum": ["a", "b"]}
    assert props["key"] == {"type": "string"}  # NewType unwraps to str


def test_schema_unknown_type_is_permissive():
    @tool()
    def f(payload: Any, blob: object) -> str:
        return ""

    spec = f
    props = spec.parameters["properties"]
    assert props["payload"] == {}
    assert props["blob"] == {}


def test_docstring_becomes_description():
    @tool("described")
    def described(x: int) -> str:
        """Search recalled memory by keyword."""
        return ""

    assert described.description == "Search recalled memory by keyword."


def test_explicit_description_overrides_docstring():
    @tool("described", description="Explicit.")
    def described(x: int) -> str:
        """Docstring."""
        return ""

    assert described.description == "Explicit."


def test_tool_name_defaults_to_function_name():
    @tool()
    def my_tool_fn(x: int) -> str:
        return ""

    assert my_tool_fn.name == "my_tool_fn"


def test_async_handler_detected():
    @tool()
    async def afn(x: int) -> str:
        return ""

    assert afn.is_async is True


def test_provider_definition_shape():
    @tool("pd", capability="mem")
    def pd(q: str) -> str:
        """Query."""
        return ""

    assert pd.provider_definition() == {
        "type": "function",
        "function": {
            "name": "pd",
            "description": "Query.",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
    }


# ── ToolSpec validation ─────────────────────────────────────────────────────

def test_invalid_tool_name_rejected():
    with pytest.raises(ToolError, match="invalid tool name"):
        ToolSpec(name="1bad", description="", parameters={}, handler=lambda: None)


def test_invalid_visibility_rejected():
    with pytest.raises(ToolError, match="visibility"):
        ToolSpec(
            name="ok", description="", parameters={}, handler=lambda: None,
            visibility="sometimes",
        )


def test_invalid_chat_scope_rejected():
    with pytest.raises(ToolError, match="chat_scope"):
        ToolSpec(
            name="ok", description="", parameters={}, handler=lambda: None,
            chat_scope="everyone",
        )


def test_non_callable_handler_rejected():
    with pytest.raises(ToolError, match="handler"):
        ToolSpec(name="ok", description="", parameters={}, handler="not-callable")  # type: ignore[arg-type]


def test_bad_timeout_and_rate_limit_rejected():
    with pytest.raises(ToolError, match="timeout_s"):
        ToolSpec(name="ok", description="", parameters={}, handler=lambda: None, timeout_s=0)
    with pytest.raises(ToolError, match="rate_limit"):
        ToolSpec(name="ok", description="", parameters={}, handler=lambda: None, rate_limit=-1)


# ── ToolRegistry ────────────────────────────────────────────────────────────

def test_registry_register_and_lookup():
    reg = ToolRegistry()
    spec = tool("a")(lambda x: x)
    reg.register(spec)
    assert reg.require("a") is spec
    assert reg.names() == ("a",)


def test_registry_duplicate_rejected():
    reg = ToolRegistry()
    reg.register(tool("a")(lambda x: x))
    with pytest.raises(RegistryError, match="already registered"):
        reg.register(tool("a")(lambda x: x))


def test_registry_replace_shadows_keeping_order():
    reg = ToolRegistry()
    first = tool("a")(lambda x: x)
    reg.register(first)
    reg.register(tool("b")(lambda x: x))
    second = tool("a")(lambda x: x)
    reg.register(second, replace=True)
    assert reg.require("a") is second
    assert reg.names() == ("a", "b")  # original slot kept


def test_registry_rejects_non_toolspec():
    reg = ToolRegistry()
    with pytest.raises(RegistryError, match="ToolSpec"):
        reg.register("not-a-spec")  # type: ignore[arg-type]


def test_registry_capability_tracking():
    reg = ToolRegistry()
    reg.register(tool("m1", capability="memory")(lambda x: x))
    reg.register(tool("m2", capability="memory")(lambda x: x))
    reg.register(tool("other")(lambda x: x))
    names = [s.name for s in reg.with_capability("memory")]
    assert names == ["m1", "m2"]
    assert reg.with_capability("nope") == ()


def test_registry_capability_updated_on_replace():
    reg = ToolRegistry()
    reg.register(tool("m1", capability="memory")(lambda x: x))
    reg.register(tool("m1", capability="search")(lambda x: x), replace=True)
    assert [s.name for s in reg.with_capability("memory")] == []
    assert [s.name for s in reg.with_capability("search")] == ["m1"]


def test_registry_capability_removed_on_unregister():
    reg = ToolRegistry()
    reg.register(tool("m1", capability="memory")(lambda x: x))
    reg.unregister("m1")
    assert reg.with_capability("memory") == ()


def test_registry_provider_definitions_filters_visibility_and_scope():
    reg = ToolRegistry()
    reg.register(tool("vis", chat_scope="all")(lambda x: x))
    reg.register(tool("def", visibility="deferred")(lambda x: x))
    reg.register(tool("hid", visibility="hidden")(lambda x: x))
    reg.register(tool("grp", chat_scope="group")(lambda x: x))
    names = [d["function"]["name"] for d in reg.provider_definitions()]
    assert names == ["vis", "grp"]  # deferred/hidden excluded
    group_names = [d["function"]["name"] for d in reg.provider_definitions(scope="group")]
    assert group_names == ["vis", "grp"]
    private_names = [d["function"]["name"] for d in reg.provider_definitions(scope="private")]
    assert private_names == ["vis"]


def test_registry_tool_decorator_registers():
    reg = ToolRegistry()

    @reg.tool("registered", capability="mem")
    def registered(q: str) -> str:
        """Query."""
        return ""

    assert reg.require("registered") is registered
    assert [s.name for s in reg.with_capability("mem")] == ["registered"]


# ── query_jargon: the deferred chat-bound jargon tool (Phase 6 P6.4b) ────────

def test_query_jargon_is_deferred_and_activated_via_tool_search():
    from pretender.tools.core import CoreToolRegistry, register_core_tools
    from pretender.tools.knowledge import KNOWLEDGE_TOOL_NAMES

    assert "query_jargon" in KNOWLEDGE_TOOL_NAMES
    reg = register_core_tools()
    spec = reg.require("query_jargon")
    assert spec.visibility == "deferred"
    assert spec.capability == "memory"
    # Not emitted until tool_search activates it.
    names = [d["function"]["name"] for d in reg.provider_definitions()]
    assert "query_jargon" not in names
    assert reg.activate("query_jargon") is True
    names = [d["function"]["name"] for d in reg.provider_definitions()]
    assert "query_jargon" in names


def test_query_jargon_handler_is_chat_bound_and_capped():
    import asyncio
    import json

    from pretender.tools.core import ToolContext, dispatch_call, register_core_tools
    from pretender.tools.knowledge import KnowledgeCallbacks
    from pretender.types import RecordHit, ToolCall, ToolCallId

    async def scenario():
        seen: list[tuple] = []

        async def jargon(query, limit):
            seen.append((query, limit))
            return [
                RecordHit(
                    chat_key=ChatKey("qq:group:123456"),
                    learner="jargon",
                    record_id=1,
                    text="yyds 永远的神",
                    score=1.0,
                )
            ]

        reg = register_core_tools()
        reg.activate("query_jargon")  # the planner's tool_search did this
        ctx = ToolContext(
            chat_key=ChatKey("qq:group:123456"),
            chat_kind="group",
            registry=reg,
            knowledge=KnowledgeCallbacks(query_jargon=jargon),
        )
        result = await dispatch_call(
            ToolCall(id=ToolCallId("c1"), name="query_jargon", arguments={"query": "yyds"}),
            ctx,
        )
        return result, seen

    result, seen = run_async(scenario())
    assert result.ok is True
    assert seen == [("yyds", 3)]  # limit clamped to MAX_JARGON_HITS
    data = json.loads(result.content)
    assert data["action"] == "query_jargon"
    assert data["count"] == 1
    assert data["hits"][0]["text"] == "yyds 永远的神"


def test_query_jargon_fails_closed_without_service():
    from pretender.tools.core import ToolContext, dispatch_call, register_core_tools
    from pretender.types import ToolCall, ToolCallId

    async def scenario():
        reg = register_core_tools()
        reg.activate("query_jargon")
        ctx = ToolContext(
            chat_key=ChatKey("qq:group:123456"),
            chat_kind="group",
            registry=reg,
        )
        result = await dispatch_call(
            ToolCall(id=ToolCallId("c1"), name="query_jargon", arguments={"query": "x"}),
            ctx,
        )
        return result

    result = run_async(scenario())
    assert result.ok is False
    assert "jargon service is not available" in (result.error or "")


def run_async(coro):
    import asyncio

    return asyncio.run(coro)


# ── send_emoji / send_image: deferred, capability-gated media tools (P6.5b) ──

def test_media_tools_are_deferred_and_activated_via_tool_search():
    from pretender.tools.core import CoreToolRegistry, register_core_tools
    from pretender.tools.media import MEDIA_TOOL_NAMES

    assert MEDIA_TOOL_NAMES == ("send_emoji", "send_image")
    reg = register_core_tools()
    for name in MEDIA_TOOL_NAMES:
        spec = reg.require(name)
        assert spec.visibility == "deferred"
        assert spec.capability in ("sticker", "image")
        # Not emitted until tool_search activates it.
        names = [d["function"]["name"] for d in reg.provider_definitions()]
        assert name not in names
        assert reg.activate(name) is True
        names = [d["function"]["name"] for d in reg.provider_definitions()]
        assert name in names


def test_media_tool_schemas_accept_opaque_asset_id_only():
    from pretender.tools.core import register_core_tools

    reg = register_core_tools()
    for name in ("send_emoji", "send_image"):
        spec = reg.require(name)
        assert spec.parameters["required"] == ["asset_id"]
        assert spec.parameters["properties"]["asset_id"] == {"type": "integer"}
        assert spec.description.strip()
