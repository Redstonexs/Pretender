"""Phase 3 replyer: reply-draft rendering tests.

Covers: draft validation/fallback, safe handling of malformed/empty model
content, no planner analysis / tool JSON in the replyer's input or output,
usage aggregation, and the no-output short-circuit.

Async tests run via asyncio.run() so the test extra stays at just pytest.
"""

from __future__ import annotations

import asyncio

import pytest

from pretender.context import serialize
from pretender.prompts import PromptStore
from pretender.replyer import REPLY_PROFILE, REPLYER_PROMPT, ReplyDraft, Replyer
from pretender.types import LLMResponse, TranscriptMessage


def run(coro):
    return asyncio.run(coro)


class FakeLLM:
    """Scripted LLMClient for the replyer: records every call and validates
    transcript legality via ``serialize``."""

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
        serialize(msgs)
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


def make_replyer(script=None):
    llm = FakeLLM(script)
    return Replyer(llm, PromptStore()), llm


# ── happy path ───────────────────────────────────────────────────────────────


def test_reply_draft_basic():
    replyer, llm = make_replyer(
        [
            LLMResponse(
                content="  好的，没问题  ",
                usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            )
        ]
    )
    draft = run(
        replyer.reply(
            reply_reference="你好",
            identity="你是麦麦",
            reply_style="自然",
            reply_to="123",
        )
    )
    assert draft.text == "好的，没问题"
    assert not draft.no_output
    assert draft.reply_to == "123"
    assert draft.usage == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    assert draft.tokens_in == 5
    assert draft.tokens_out == 3

    # transcript is exactly [system(prompt), user(reply_reference)] — legal
    # and free of planner analysis / tool JSON.
    msgs, kwargs = llm.calls[0]
    assert kwargs["profile"] == REPLY_PROFILE
    assert kwargs["tools"] is None  # the replyer never offers tools
    assert [m.role for m in msgs] == ["system", "user"]
    assert "你是麦麦" in msgs[0].content
    assert "自然" in msgs[0].content
    assert msgs[1].content == "你好"


def test_no_planner_analysis_in_input():
    # Even when the planner's internal transcript carried analysis text, the
    # replyer's input carries ONLY the staged reply reference.
    replyer, llm = make_replyer([LLMResponse(content="收到")])
    run(
        replyer.reply(
            reply_reference="好的，我看看",
            identity="你是麦麦",
            reply_style="自然",
        )
    )
    msgs = llm.calls[0][0]
    assert msgs[1].content == "好的，我看看"
    assert "分析" not in msgs[1].content
    assert "tool_calls" not in msgs[1].content
    assert all(m.role in ("system", "user") for m in msgs)


# ── malformed / empty output → safe no-output ────────────────────────────────


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "   ",
        '{"action": "reply", "text": "x"}',
        '{"tool_calls": [{"id": "c1"}]}',
        "[1, 2, 3]",
        "```json\n{\"a\": 1}\n```",
        123,
        ["not", "a", "string"],
    ],
)
def test_structured_or_empty_output_is_no_output(content):
    replyer, llm = make_replyer([LLMResponse(content=content)])
    draft = run(
        replyer.reply(reply_reference="你好", identity="x", reply_style="y")
    )
    assert draft.no_output
    assert draft.text == ""
    # the LLM was still called (content was non-empty reference), but the
    # structured/empty output never reached the user.
    assert len(llm.calls) == 1


def test_fenced_plain_text_is_kept():
    replyer, _ = make_replyer([LLMResponse(content="```\n好的\n```")])
    draft = run(
        replyer.reply(reply_reference="你好", identity="x", reply_style="y")
    )
    assert draft.text == "好的"
    assert not draft.no_output


def test_fenced_json_is_rejected():
    replyer, _ = make_replyer([LLMResponse(content="```\n{\"action\": \"reply\"}\n```")])
    draft = run(
        replyer.reply(reply_reference="你好", identity="x", reply_style="y")
    )
    assert draft.no_output
    assert draft.text == ""


# ── empty reference short-circuit ────────────────────────────────────────────


@pytest.mark.parametrize("reference", ["", "   ", None])
def test_empty_reference_short_circuits_without_llm_call(reference):
    replyer, llm = make_replyer()
    draft = run(
        replyer.reply(reply_reference=reference, identity="x", reply_style="y")
    )
    assert draft.no_output
    assert draft.text == ""
    assert llm.calls == []  # never called the LLM


# ── usage aggregation ────────────────────────────────────────────────────────


def test_usage_aggregation_and_normalization():
    replyer, _ = make_replyer(
        [
            LLMResponse(
                content="好",
                usage={"prompt_tokens": 12, "completion_tokens": 4, "junk": "x"},
            )
        ]
    )
    draft = run(
        replyer.reply(reply_reference="你好", identity="x", reply_style="y")
    )
    assert draft.usage == {"prompt_tokens": 12, "completion_tokens": 4}
    assert draft.tokens_in == 12
    assert draft.tokens_out == 4


# ── draft validation ─────────────────────────────────────────────────────────


def test_reply_draft_validation():
    with pytest.raises(ValueError):
        ReplyDraft(text=123)
    with pytest.raises(ValueError):
        ReplyDraft(text="x", tokens_in=-1)
    with pytest.raises(ValueError):
        ReplyDraft(text="x", tokens_out="nope")
    with pytest.raises(ValueError):
        ReplyDraft(text="x", reply_to=5)
    # non-int usage entries are normalized away
    draft = ReplyDraft(text="x", usage={"prompt_tokens": 1, "bad": "y"})
    assert draft.usage == {"prompt_tokens": 1}


def test_reply_draft_empty_factory():
    draft = ReplyDraft.empty(reply_to="9")
    assert draft.no_output
    assert draft.text == ""
    assert draft.reply_to == "9"
    assert draft.usage == {}


def test_reply_draft_no_output_property():
    assert ReplyDraft(text="").no_output
    assert ReplyDraft(text="   ").no_output
    assert not ReplyDraft(text="hi").no_output


def test_replyer_prompt_name_constant():
    assert REPLYER_PROMPT == "replyer.txt"
    assert REPLY_PROFILE == "reply"
