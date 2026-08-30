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
    # The final user turn carries the staged reference plus the output
    # instruction; with no ReplyContext there is no history and no clock.
    assert "【回复信息参考】\n你好" in msgs[1].content


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
    assert "好的，我看看" in msgs[1].content
    # Nothing precedes the staged reference except the time / target blocks —
    # the planner's analysis never enters the replyer transcript.
    assert "分析" not in msgs[1].content.split("【回复信息参考】")[0]
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


# ── the replyer sees the conversation (MaiBot parity) ───────────────────────


def _msg(text, *, is_self=False, name="小明", ts=1787990000.0, mid="1"):
    from pretender.types import ChatKey, Message

    return Message(
        chat_key=ChatKey("qq:group:1"),
        sender_id="9" if not is_self else "1",
        sender_name=name,
        is_self=is_self,
        text=text,
        id=mid,
        recv_ts=ts,
    )


def _reply_with_context(ctx, reference="参考文本"):
    from pretender.replyer import ReplyContext

    replyer, llm = make_replyer([LLMResponse(content="好")])
    run(
        replyer.reply(
            reply_reference=reference,
            identity="你是群里的普通成员",
            reply_style="自然",
            context=ctx,
        )
    )
    return llm.calls[0][0]


def test_chat_history_becomes_role_tagged_turns():
    """The bot's own messages are assistant turns, so the model reads its own
    voice in the conversation rather than a transcript describing it."""
    from pretender.replyer import ReplyContext

    msgs = _reply_with_context(
        ReplyContext(
            chat_history=(
                _msg("在吗"),
                _msg("在的", is_self=True),
                _msg("那你说说"),
            ),
            bot_name="麦麦",
        )
    )
    assert [m.role for m in msgs] == ["system", "user", "assistant", "user", "user"]
    assert msgs[2].content == "在的"
    assert "小明: 在吗" in msgs[1].content


def test_history_lines_carry_the_clock():
    from pretender.replyer import ReplyContext

    msgs = _reply_with_context(
        ReplyContext(chat_history=(_msg("在吗"),), bot_name="麦麦")
    )
    assert msgs[1].content.startswith("[")
    assert "] 小明: 在吗" in msgs[1].content


def test_final_turn_carries_the_current_time():
    from pretender.replyer import ReplyContext

    msgs = _reply_with_context(ReplyContext(now=1787990000.0))
    assert "当前时间：" in msgs[-1].content


def test_final_turn_names_the_target_message():
    from pretender.replyer import ReplyContext

    msgs = _reply_with_context(
        ReplyContext(target=_msg("你叫什么名字"), bot_name="麦麦")
    )
    assert "【你要回复的消息】" in msgs[-1].content
    assert "你叫什么名字" in msgs[-1].content


def test_bot_name_and_drift_reach_the_system_prompt():
    from pretender.replyer import ReplyContext

    from pretender.config import DriftConfig
    from pretender.drift import build_drift_block

    msgs = _reply_with_context(
        ReplyContext(bot_name="bp", drift_block=build_drift_block(DriftConfig()))
    )
    assert "你的名字是bp" in msgs[0].content
    assert "注意力漂移风格" in msgs[0].content


def test_planner_chosen_length_becomes_an_instruction():
    """MaiBot's ``reply`` tool ``reply_style``: the planner has read the room
    and decides how long the reply should be."""
    from pretender.replyer import LENGTH_DIRECTIVES, ReplyContext

    msgs = _reply_with_context(ReplyContext(length_style="简短表达"))
    assert LENGTH_DIRECTIVES["简短表达"] in msgs[-1].content
    # "正常回复" carries no directive, so nothing is injected.
    msgs = _reply_with_context(ReplyContext(length_style="正常回复"))
    assert LENGTH_DIRECTIVES["简短表达"] not in msgs[-1].content


def test_empty_history_entries_are_skipped():
    from pretender.replyer import ReplyContext

    msgs = _reply_with_context(
        ReplyContext(chat_history=(_msg("  "), _msg("在吗")), bot_name="麦麦")
    )
    assert [m.role for m in msgs] == ["system", "user", "user"]


def test_no_context_degrades_to_the_reference_only_request():
    msgs = _reply_with_context(None)
    assert [m.role for m in msgs] == ["system", "user"]


def test_impressions_of_the_people_here_reach_the_final_turn():
    """Knowing who you are talking to is most of what separates a regular
    from a stranger. Without this the bot restarts every conversation from
    zero, no matter how many times it has met the person."""
    from pretender.replyer import ReplyContext

    msgs = _reply_with_context(
        ReplyContext(
            impressions=(
                ("小明", "爱聊游戏，说话很快"),
                ("小红", "话不多，但每次都接得挺准"),
            )
        )
    )
    final = msgs[-1].content
    assert "【你对他们的印象】" in final
    assert "小明: 爱聊游戏，说话很快" in final
    assert "小红: 话不多，但每次都接得挺准" in final
    # It is an observation, not an order.
    assert "不是指令" in final


def test_no_impressions_adds_no_block():
    from pretender.replyer import ReplyContext

    assert "【你对他们的印象】" not in _reply_with_context(ReplyContext())[-1].content
