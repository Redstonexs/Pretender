"""Phase 3 planner: tool-loop driver tests.

Covers: legal wire transcript on every exit, every tool id answered
(including malformed/fuzzed calls), the tool-loop cap, prompt rendering and
focus selection, the tolerant tool-JSON fallback with a single injected
repair, usage aggregation, and tool-dispatch error containment.

Async tests run via asyncio.run() so the test extra stays at just pytest.
"""

from __future__ import annotations

import asyncio

import pytest

from pretender.config import ContextConfig
from pretender.context import serialize
from pretender.planner import (
    DEFAULT_MAX_TOOL_ROUNDS,
    PLANNER_FOCUS_PROMPT,
    PLANNER_PROFILE,
    PLANNER_PROMPT,
    PlanIntent,
    Planner,
)
from pretender.prompts import PromptStore
from pretender.tools.base import tool
from pretender.tools.core import ToolContext, register_core_tools
from pretender.types import ChatKey, LLMResponse, ToolCall, ToolCallId, TranscriptMessage


def run(coro):
    return asyncio.run(coro)


# ── fakes / fixtures ─────────────────────────────────────────────────────────


class FakeLLM:
    """Scripted LLMClient: pops one response per complete() call, records
    every call (transcript + kwargs), and validates transcript legality via
    ``serialize`` (raises if the planner ever built a non-canonical wire)."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls: list[tuple[list[TranscriptMessage], dict]] = []

    async def complete(
        self,
        messages,
        *,
        profile,
        tools=None,
        temperature=None,
        max_tokens=None,
        deadline=None,
    ):
        msgs = list(messages)
        serialize(msgs)  # fail the test if the planner built an illegal transcript
        self.calls.append(
            (
                msgs,
                {
                    "profile": profile,
                    "tools": tools,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "deadline": deadline,
                },
            )
        )
        if self.script:
            item = self.script.pop(0)
            if callable(item):
                return item(msgs)
            return item
        return LLMResponse(content=None)


def make_ctx(registry, chat_kind="group"):
    return ToolContext(
        chat_key=ChatKey("console:group:demo"),
        chat_kind=chat_kind,
        capabilities=frozenset({"history", "forward"}),
        registry=registry,
        self_name="麦麦",
    )


def make_planner(script, *, max_tool_rounds=5, repair=None, injected_ctx=None):
    llm = FakeLLM(script)
    prompts = PromptStore()
    registry = register_core_tools()
    cc = ContextConfig(max_context_size=40, max_image_num=3, keep_recent=0)
    if injected_ctx is not None:
        planner = Planner(
            llm,
            prompts,
            registry,
            cc,
            tool_context=injected_ctx,
            max_tool_rounds=max_tool_rounds,
            repair=repair,
        )
    else:
        planner = Planner(
            llm,
            prompts,
            registry,
            cc,
            tool_context_factory=lambda: make_ctx(registry),
            max_tool_rounds=max_tool_rounds,
            repair=repair,
        )
    return planner, llm, registry


def _tool_ids(transcript):
    """The ordered tool_call ids of every assistant tool-call turn."""
    return [
        c.id
        for m in transcript
        if m.role == "assistant"
        for c in m.tool_calls
    ]


def _tool_answer_ids(transcript):
    return [m.tool_call_id for m in transcript if m.role == "tool"]


def _assert_answered(transcript):
    """Every assistant tool call id is answered by exactly one tool message,
    in call order (the canonical frozen-decision #4 shape)."""
    assert _tool_ids(transcript) == _tool_answer_ids(transcript)


# ── terminal exits ───────────────────────────────────────────────────────────


def test_reply_exit_legal_transcript():
    script = [
        LLMResponse(
            content=None,
            tool_calls=(
                ToolCall(
                    id=ToolCallId("call_1"),
                    name="reply",
                    arguments={"text": "你好呀", "reply_to": None},
                ),
            ),
        )
    ]
    planner, llm, _ = make_planner(script)

    result = run(
        planner.plan(
            [TranscriptMessage(role="user", content="在吗")],
            chat_log="小明: 在吗",
            reply_style="自然",
        )
    )

    assert result.intent == PlanIntent.REPLY
    assert result.reply_reference == "你好呀"
    assert result.reply_to is None
    assert result.end_reason == "reply"
    assert not result.degraded
    assert len(result.rounds) == 1
    # legal wire + every id answered
    serialize(result.transcript)
    _assert_answered(result.transcript)
    # the tool message carries the reply tool's JSON result
    tool_msgs = [m for m in result.transcript if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == ToolCallId("call_1")
    assert tool_msgs[0].content
    # profile is the planner profile
    assert llm.calls[0][1]["profile"] == PLANNER_PROFILE


def test_wait_exit():
    script = [
        LLMResponse(
            content=None,
            tool_calls=(ToolCall(id=ToolCallId("call_1"), name="wait", arguments={"seconds": 30}),),
        )
    ]
    planner, _, _ = make_planner(script)
    result = run(
        planner.plan([], chat_log="", reply_style="y")
    )
    assert result.intent == PlanIntent.WAIT
    assert result.wait_seconds == 30
    assert result.end_reason == "wait"
    serialize(result.transcript)
    _assert_answered(result.transcript)


def test_no_action_exit():
    script = [
        LLMResponse(
            content=None,
            tool_calls=(ToolCall(id=ToolCallId("call_1"), name="no_action", arguments={}),),
        )
    ]
    planner, _, _ = make_planner(script)
    result = run(
        planner.plan([], chat_log="", reply_style="y")
    )
    assert result.intent == PlanIntent.NO_ACTION
    assert result.end_reason == "no_action"
    serialize(result.transcript)
    _assert_answered(result.transcript)


def test_no_tool_call_degrades_to_no_action():
    script = [LLMResponse(content="我分析了一下，没什么好说的", tool_calls=())]
    planner, _, _ = make_planner(script)
    result = run(
        planner.plan([], chat_log="", reply_style="y")
    )
    assert result.intent == PlanIntent.NO_ACTION
    assert result.end_reason == "no_tool_call"
    assert result.reply_reference is None  # analysis is never a reply
    assert not result.degraded
    serialize(result.transcript)
    _assert_answered(result.transcript)


def test_empty_response_degrades_to_no_action():
    script = [LLMResponse(content=None, tool_calls=())]
    planner, _, _ = make_planner(script)
    result = run(
        planner.plan([], chat_log="", reply_style="y")
    )
    assert result.intent == PlanIntent.NO_ACTION
    assert result.end_reason == "empty_response"
    serialize(result.transcript)


# ── multi-round / loop cap ───────────────────────────────────────────────────


def test_multi_round_tool_search_then_reply():
    script = [
        LLMResponse(
            content="先搜索",
            tool_calls=(ToolCall(id=ToolCallId("call_1"), name="tool_search", arguments={"query": "history"}),),
        ),
        LLMResponse(
            content=None,
            tool_calls=(ToolCall(id=ToolCallId("call_2"), name="reply", arguments={"text": "好"},),),
        ),
    ]
    planner, llm, registry = make_planner(script)
    result = run(
        planner.plan([], chat_log="", reply_style="y")
    )
    assert result.intent == PlanIntent.REPLY
    assert result.reply_reference == "好"
    assert result.end_reason == "reply"
    assert len(result.rounds) == 2
    serialize(result.transcript)
    _assert_answered(result.transcript)
    # tool_search activated the deferred fetch_history tool, so the second
    # round's provider definitions include it.
    round2_tools = llm.calls[1][1]["tools"]
    names = [t["function"]["name"] for t in round2_tools]
    assert "fetch_history" in names
    assert registry.is_activated("fetch_history")


def test_tool_loop_cap():
    script = [
        LLMResponse(
            content=None,
            tool_calls=(ToolCall(id=ToolCallId(f"call_{i}"), name="tool_search", arguments={"query": "x"}),),
        )
        for i in range(10)
    ]
    planner, llm, _ = make_planner(script, max_tool_rounds=3)
    result = run(
        planner.plan([], chat_log="", reply_style="y")
    )
    assert result.intent == PlanIntent.NO_ACTION
    assert result.end_reason == "tool_round_cap"
    assert len(result.rounds) == 3
    assert len(llm.calls) == 3
    serialize(result.transcript)
    _assert_answered(result.transcript)


# ── malformed / fuzzed calls ─────────────────────────────────────────────────


def test_malformed_tool_json_fallback_reply():
    # The model emitted tool JSON in content instead of structured calls.
    script = [
        LLMResponse(
            content='{"id": "call_9", "name": "reply", "arguments": {"text": "你好"}}',
            tool_calls=(),
        )
    ]
    planner, _, _ = make_planner(script)
    result = run(
        planner.plan([], chat_log="", reply_style="y")
    )
    assert result.intent == PlanIntent.REPLY
    assert result.reply_reference == "你好"
    assert result.end_reason == "reply"
    assert result.degraded  # tolerant fallback was used
    serialize(result.transcript)
    _assert_answered(result.transcript)


def test_fuzzed_calls_all_answered():
    script = [
        LLMResponse(
            content=None,
            tool_calls=(
                ToolCall(id=ToolCallId("call_1"), name="reply", arguments={"text": "ok"}),
                ToolCall(id=ToolCallId("call_2"), name="", arguments=None),  # unknown + bad args
                ToolCall(id=ToolCallId("call_2"), name="no_action", arguments={}),  # duplicate id
                ToolCall(id=ToolCallId("call_3"), name="fetch_history", arguments={"limit": 5}),  # deferred
                ToolCall(id=ToolCallId("call_4"), name="wait", arguments={"seconds": "bogus"}),  # schema
            ),
        )
    ]
    planner, _, _ = make_planner(script)
    result = run(
        planner.plan([], chat_log="", reply_style="y")
    )
    # call_1 staged a reply → terminal REPLY; every id still answered.
    assert result.intent == PlanIntent.REPLY
    assert result.reply_reference == "ok"
    serialize(result.transcript)
    _assert_answered(result.transcript)
    ids = [r.call_id for r in result.tool_results]
    assert ids == [ToolCallId("call_1"), ToolCallId("call_2"), ToolCallId("call_3"), ToolCallId("call_4")]
    by_id = {r.call_id: r for r in result.tool_results}
    assert by_id[ToolCallId("call_1")].ok
    assert not by_id[ToolCallId("call_2")].ok  # unknown tool
    assert not by_id[ToolCallId("call_3")].ok  # deferred, not activated
    assert not by_id[ToolCallId("call_4")].ok  # schema mismatch


def test_repair_callback_invoked_once():
    calls = {"n": 0}

    def repair(text):
        calls["n"] += 1
        return '{"id": "call_1", "name": "no_action", "arguments": {}}'

    script = [LLMResponse(content="no json block here", tool_calls=())]
    planner, _, _ = make_planner(script, repair=repair)
    result = run(
        planner.plan([], chat_log="", reply_style="y")
    )
    assert result.intent == PlanIntent.NO_ACTION
    assert result.end_reason == "no_action"
    assert result.degraded
    assert calls["n"] == 1  # at most one repair attempt per snippet
    serialize(result.transcript)
    _assert_answered(result.transcript)


# ── dispatch error containment ───────────────────────────────────────────────


def test_raising_handler_is_contained():
    def _boom(self, text: str) -> str:
        raise RuntimeError("kaboom")

    spec = tool("boom_tool", description="explodes")(_boom)
    llm = FakeLLM(
        [
            LLMResponse(
                content=None,
                tool_calls=(ToolCall(id=ToolCallId("call_1"), name="boom_tool", arguments={"text": "x"}),),
            ),
            LLMResponse(content=None, tool_calls=()),
        ]
    )
    prompts = PromptStore()
    registry = register_core_tools()
    registry.register(spec)
    cc = ContextConfig()
    planner = Planner(
        llm,
        prompts,
        registry,
        cc,
        tool_context_factory=lambda: make_ctx(registry),
    )
    result = run(planner.plan([], chat_log="", reply_style="y"))
    # dispatch_call itself contains the handler exception → ok=False result.
    assert result.tool_results[0].ok is False
    assert "kaboom" in (result.tool_results[0].error or "")
    serialize(result.transcript)
    _assert_answered(result.transcript)


def test_dispatch_call_raising_is_contained(monkeypatch):
    import pretender.planner as planner_mod

    async def _exploding_dispatch(call, ctx, registry):
        raise ValueError("dispatch exploded")

    monkeypatch.setattr(planner_mod, "dispatch_call", _exploding_dispatch)
    script = [
        LLMResponse(
            content=None,
            tool_calls=(ToolCall(id=ToolCallId("call_1"), name="reply", arguments={"text": "x"}),),
        ),
        LLMResponse(content=None, tool_calls=()),
    ]
    planner, _, _ = make_planner(script)
    result = run(planner.plan([], chat_log="", reply_style="y"))
    # The planner's own containment boundary turns the raising dispatch into
    # an ok=False result; the id is still answered and the loop degrades to
    # no_action rather than raising.
    assert result.tool_results[0].ok is False
    assert "dispatch exploded" in (result.tool_results[0].error or "")
    assert result.degraded
    assert result.intent == PlanIntent.NO_ACTION
    serialize(result.transcript)
    _assert_answered(result.transcript)


# ── usage aggregation ────────────────────────────────────────────────────────


def test_usage_aggregation():
    script = [
        LLMResponse(
            content=None,
            tool_calls=(ToolCall(id=ToolCallId("call_1"), name="tool_search", arguments={"query": "x"}),),
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        ),
        LLMResponse(
            content="分析",
            tool_calls=(),
            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        ),
    ]
    planner, _, _ = make_planner(script)
    result = run(planner.plan([], chat_log="", reply_style="y"))
    assert result.usage == {
        "prompt_tokens": 17,
        "completion_tokens": 8,
        "total_tokens": 25,
    }
    assert result.tokens_in == 17
    assert result.tokens_out == 8


# ── prompt rendering / focus selection ───────────────────────────────────────


def test_prompt_rendering_and_focus_selection():
    script = [
        LLMResponse(
            content=None,
            tool_calls=(ToolCall(id=ToolCallId("call_1"), name="no_action", arguments={}),),
        )
    ]
    planner, llm, _ = make_planner(script)

    run(
        planner.plan(
            [],
            chat_log="小明: 你好",
            reply_style="活泼",
            behavior_style="测试行为风格",
        )
    )
    system = llm.calls[0][0][0]
    assert system.role == "system"
    # The planner gets the persona's ACTION rules, not the replyer identity.
    assert "测试行为风格" in system.content
    assert "小明: 你好" in system.content
    assert "活泼" in system.content
    assert "当前聚焦的聊天" not in system.content

    run(
        planner.plan(
            [],
            chat_log="小明: 你好",
            reply_style="活泼",
            focus_chat="群A",
        )
    )
    system2 = llm.calls[1][0][0]
    assert "当前聚焦的聊天" in system2.content
    assert "群A" in system2.content


def test_prompt_names_are_planner_assets():
    # The planner renders the real package assets (planner / planner_focus).
    assert PLANNER_PROMPT == "planner.txt"
    assert PLANNER_FOCUS_PROMPT == "planner_focus.txt"
    assert DEFAULT_MAX_TOOL_ROUNDS == 5


# ── construction / injected context ──────────────────────────────────────────


def test_construction_requires_exactly_one_context_source():
    llm = FakeLLM()
    prompts = PromptStore()
    registry = register_core_tools()
    cc = ContextConfig()
    with pytest.raises(ValueError):
        Planner(llm, prompts, registry, cc)  # neither
    with pytest.raises(ValueError):
        Planner(
            llm,
            prompts,
            registry,
            cc,
            tool_context_factory=lambda: make_ctx(registry),
            tool_context=make_ctx(registry),
        )  # both
    with pytest.raises(ValueError):
        Planner(
            llm,
            prompts,
            registry,
            cc,
            tool_context_factory=lambda: make_ctx(registry),
            max_tool_rounds=0,
        )


def test_injected_tool_context_is_reused():
    ctx = make_ctx(register_core_tools())
    script = [
        LLMResponse(
            content=None,
            tool_calls=(ToolCall(id=ToolCallId("call_1"), name="reply", arguments={"text": "hi"}),),
        )
    ]
    planner, _, _ = make_planner(script, injected_ctx=ctx)
    result = run(planner.plan([], chat_log="", reply_style="y"))
    assert result.intent == PlanIntent.REPLY
    assert result.reply_reference == "hi"
    assert ctx.reply_text == "hi"  # the injected context carried the verdict


def test_plan_result_is_immutable_typed():
    script = [
        LLMResponse(
            content=None,
            tool_calls=(ToolCall(id=ToolCallId("call_1"), name="no_action", arguments={}),),
        )
    ]
    planner, _, _ = make_planner(script)
    result = run(planner.plan([], chat_log="", reply_style="y"))
    assert isinstance(result.rounds, tuple)
    assert isinstance(result.transcript, tuple)
    assert isinstance(result.tool_results, tuple)
    assert isinstance(result.usage, dict)
