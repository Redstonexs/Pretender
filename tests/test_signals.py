"""Signals: the four normalizer strip targets, content detectors,
short-reaction predicate and batch eligibility, strict other-assistant
prefix matching, exclusive length tiers, batch aggregation, and the
authoritative snapshot facts."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from pretender.signals import (
    OTHER_ASSISTANT_RE,
    BatchSignals,
    analyze_batch,
    is_chinese_question,
    is_direct_request,
    is_opinion_solicit,
    is_other_assistant_target,
    is_short_reaction,
    is_short_reaction_batch,
    normalize_text,
)
from pretender.types import (
    ChatKey,
    CycleId,
    GateSnapshot,
    Message,
    MessageRowId,
    SelfId,
    SenderId,
)

CK = ChatKey("qq:group:123456")
SENDER = SenderId("u1")


def _msg(text: str, **kw: Any) -> Message:
    base: dict[str, Any] = dict(
        chat_key=CK,
        sender_id=SENDER,
        sender_name="alice",
        is_self=False,
        text=text,
    )
    base.update(kw)
    return Message(**base)


def _snapshot(**kw: Any) -> GateSnapshot:
    base: dict[str, Any] = dict(
        chat_key=CK,
        cycle_id=CycleId("c1"),
        start_msg_id=MessageRowId(1),
        through_msg_id=MessageRowId(9),
        evaluated_ts=300.0,
        self_id=SelfId("bot"),
        mode="reply_necessity",
        threshold=80,
        trigger_score=80,
        frequency=1.0,
        pending=1,
        pending_messages=(_msg("hi"),),
        recent=(),
        window_count=1,
        self_count=0,
        last_nonself_ts=250.0,
        idle_seconds=30.0,
        recent_average_interval=60.0,
        self_ratio=0.1,
        is_group=True,
        is_focused=False,
        last_message=None,
    )
    base.update(kw)
    return GateSnapshot(**base)


# ── Normalizer: the four strip targets ─────────────────────────────────────

def test_normalize_strips_quote_prefix_lines():
    assert normalize_text("「张三: 在吗」\n在的") == "在的"
    assert normalize_text("[引用]张三: 在吗\n在的") == "在的"
    assert normalize_text("[quote]Alice: hi\nhello") == "hello"
    assert normalize_text("引用：内容\n正文") == "正文"
    assert normalize_text("[回复]张三: 在吗\n在的") == "在的"
    # a message that is ONLY a quote normalizes to empty
    assert normalize_text("「张三: 在吗」") == ""


def test_normalize_strips_structured_mentions():
    assert normalize_text("[CQ:at,qq=123] 你好") == "你好"
    assert normalize_text("[CQ:at,qq=123,name=Alice] 你好") == "你好"
    assert normalize_text("[at:123]Alice[/at] 你好") == "你好"
    assert normalize_text("[at]Alice[/at]你好") == "你好"
    assert normalize_text("[mention]Alice[/mention] 在吗") == "在吗"
    assert normalize_text("@{123:Alice} 你好") == "你好"


def test_normalize_strips_visible_mentions():
    assert normalize_text("@Alice 你好") == "你好"
    assert normalize_text("你好@Alice") == "你好"
    assert normalize_text("@全体成员 大家好") == "大家好"
    assert normalize_text("@123456 在吗") == "在吗"
    # an email address is not a mention and survives untouched
    assert normalize_text("联系 a@b.com 我") == "联系 a@b.com 我"


def test_normalize_strips_media_placeholders():
    assert normalize_text("看这个[图片]") == "看这个"
    assert normalize_text("[图片][视频]") == ""
    assert normalize_text("[动画表情] 哈哈") == "哈哈"
    assert normalize_text("[语音] 说") == "说"
    assert normalize_text("[CQ:image,file=x.jpg]") == ""
    assert normalize_text("[链接]") == ""
    # a bracket token that is not an exact placeholder is content
    assert normalize_text("这个[这个视频不错]") == "这个[这个视频不错]"


def test_normalize_strips_forwarded_blocks():
    assert normalize_text("在吗\n[聊天记录]\nA: x\nB: y\n\n后续") == "在吗\n后续"
    assert normalize_text("[转发消息]\nA: x\nB: y") == ""
    assert normalize_text("[转发]") == ""
    # a block with no blank line runs to the end of the text
    assert normalize_text("[聊天记录]\nA: x") == ""
    # content before the block survives
    assert normalize_text("正文\n[转发消息]\nA: x") == "正文"


def test_normalize_is_non_mutating_and_idempotent():
    raw = "「张三: 在吗」\n@Alice 看这个[图片]\n[聊天记录]\nA: x"
    once = normalize_text(raw)
    assert raw == "「张三: 在吗」\n@Alice 看这个[图片]\n[聊天记录]\nA: x"
    assert once == "看这个"
    assert normalize_text(once) == once


def test_normalize_empty_and_whitespace():
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""
    assert normalize_text("\n\n") == ""


# ── Content detectors ──────────────────────────────────────────────────────

def test_chinese_question_positives():
    for t in (
        "在吗", "今天天气怎么样？", "为什么", "几点开会", "谁来了",
        "能不能帮我", "真的吗", "多少钱", "你吃饭了吗", "咋了", "请问怎么走",
    ):
        assert is_chinese_question(t), t


def test_chinese_question_negatives():
    for t in ("好的", "今天天气不错", "就这么", "走吧", "哈哈", "谢谢"):
        assert not is_chinese_question(t), t


def test_direct_request_positives():
    for t in (
        "帮我查一下天气", "请翻译这句话", "麻烦你帮我看看", "求推荐一部电影",
        "拜托了", "帮我写个总结", "请问怎么走", "求教一下",
    ):
        assert is_direct_request(t), t


def test_direct_request_negatives():
    for t in ("今天天气不错", "请假条", "哈哈", "我要求不高", "追求", "好的"):
        assert not is_direct_request(t), t


def test_opinion_solicit_positives():
    for t in (
        "你们觉得呢", "这个怎么样", "大家怎么看", "有什么建议",
        "推荐一下", "哪个更好", "投票选一个", "说说你的想法", "给点意见",
    ):
        assert is_opinion_solicit(t), t


def test_opinion_solicit_negatives():
    for t in ("今天天气不错", "我建议你早点睡", "哈哈", "好的", "在吗"):
        assert not is_opinion_solicit(t), t


# ── Other-assistant refusal ────────────────────────────────────────────────

def test_other_assistant_regex_is_the_frozen_plan_form():
    assert OTHER_ASSISTANT_RE.pattern == (
        r"^(?:DeepSeek|ChatGPT|Grok|豆包|千问|Kimi|Claude)[,，、\s]"
    )


def test_other_assistant_target_matches_exact_names_with_separators():
    for t in (
        "DeepSeek，你好", "DeepSeek, 你好", "ChatGPT、你好", "Grok 你好",
        "豆包，在吗", "千问，你好", "Kimi，你好", "Claude，你好",
        "DeepSeek\t你好", "DeepSeek\n你好",
    ):
        assert is_other_assistant_target(t), t


def test_other_assistant_target_rejects_non_matches():
    for t in (
        "DeepSeek你好",      # no separator
        "deepseek，你好",    # wrong case
        "DeepSeek: 你好",    # colon is not a separator
        "你好 DeepSeek",     # not at the start
        "DeepSeek",          # bare name
        "",                  # empty
        "DeepSeek。你好",    # period is not a separator
        "ChatGPT4，你好",    # name must be exact
    ):
        assert not is_other_assistant_target(t), t


# ── Short reactions ────────────────────────────────────────────────────────

def test_short_reaction_positives():
    for t in (
        "哈哈", "哈哈哈", "嗯嗯", "好的", "666", "233", "+1", "👍", "😂",
        "hhh", "ok", "谢谢", "辛苦了", "666666", "嘿嘿嘿", "emmm", "nb",
        "❤️", "👍🏻",
    ):
        assert is_short_reaction(t), t


def test_short_reaction_negatives():
    for t in (
        "",                    # empty is never a reaction
        "哈哈哈哈哈啊",        # too long
        "今天天气不错",        # content
        "在吗",                # a question, not a reaction
        "帮我查天气",          # a request, not a reaction
        "6666666",             # not a known numeric reaction
        "好的好的好的",        # not a known word, no pattern
        "DeepSeek，你好",      # content
    ):
        assert not is_short_reaction(t), t


def test_short_reaction_batch_eligibility():
    assert is_short_reaction_batch(()) is False            # empty batch never eligible
    assert is_short_reaction_batch(("哈哈",)) is True
    assert is_short_reaction_batch(("哈哈", "嗯嗯")) is True
    assert is_short_reaction_batch(("哈哈", "")) is False  # empty text disqualifies
    assert is_short_reaction_batch(("",)) is False
    assert is_short_reaction_batch(("哈哈", "今天天气不错")) is False


# ── Batch analysis ─────────────────────────────────────────────────────────

def test_analyze_batch_normalizes_and_aggregates_signals():
    snap = _snapshot()
    msgs = (
        _msg("「张三: 在吗」\n@Alice 为什么"),
        _msg("帮我查一下[图片]"),
        _msg("哈哈"),
    )
    r = analyze_batch(msgs, snap)
    assert r.texts == ("为什么", "帮我查一下", "哈哈")
    assert r.has_question is True
    assert r.has_direct_request is True
    assert r.has_opinion_solicit is False
    assert r.is_short_reaction_batch is False
    assert r.total_length == 10
    assert r.length_bonus == 0


def test_analyze_batch_signal_booleans_aggregate_over_messages():
    snap = _snapshot()
    r = analyze_batch((_msg("哈哈"), _msg("这个怎么样")), snap)
    assert r.has_opinion_solicit is True
    assert r.has_question is True  # 怎么样 is also a question word


def test_analyze_batch_length_bonus_exclusive_tiers():
    snap = _snapshot()

    def bonus(n: int) -> int:
        return analyze_batch((_msg("x" * n),), snap).length_bonus

    assert bonus(0) == 0
    assert bonus(39) == 0
    assert bonus(40) == 5
    assert bonus(119) == 5
    assert bonus(120) == 10
    assert bonus(150) == 10  # exclusive: 120+ is 10, never 15


def test_analyze_batch_total_length_is_code_points():
    snap = _snapshot()
    r = analyze_batch((_msg("你好👍"),), snap)
    assert r.total_length == 3  # 2 CJK code points + 1 emoji code point


def test_analyze_batch_short_reaction_flag():
    snap = _snapshot()
    assert analyze_batch((_msg("哈哈"), _msg("嗯嗯")), snap).is_short_reaction_batch is True
    assert analyze_batch((_msg("哈哈"), _msg("在吗")), snap).is_short_reaction_batch is False
    assert analyze_batch((), snap).is_short_reaction_batch is False
    # a message that normalizes to empty disqualifies the batch
    assert analyze_batch((_msg("「张三: 在吗」"),), snap).is_short_reaction_batch is False


def test_analyze_batch_facts_are_authoritative_from_snapshot():
    # The snapshot's structured facts win even when the text contradicts them.
    snap = _snapshot(has_direct_at=True, has_quote_to_self=True, has_other_assistant=True)
    r = analyze_batch((_msg("今天天气不错"),), snap)
    assert r.has_direct_at is True
    assert r.has_quote_to_self is True
    assert r.has_other_assistant is True

    # Text that LOOKS like a direct address never overrides a False fact.
    snap2 = _snapshot(has_direct_at=False, has_quote_to_self=False, has_other_assistant=False)
    r2 = analyze_batch((_msg("@bot 在吗\nDeepSeek，你好"),), snap2)
    assert r2.has_direct_at is False
    assert r2.has_quote_to_self is False
    assert r2.has_other_assistant is False


def test_analyze_batch_empty_batch():
    snap = _snapshot(pending=0, pending_messages=())
    r = analyze_batch((), snap)
    assert r.texts == ()
    assert r.has_question is False
    assert r.has_direct_request is False
    assert r.has_opinion_solicit is False
    assert r.is_short_reaction_batch is False
    assert r.total_length == 0
    assert r.length_bonus == 0
    assert r.has_direct_at is False
    assert r.has_quote_to_self is False
    assert r.has_other_assistant is False


def test_analyze_batch_result_is_frozen():
    snap = _snapshot()
    r = analyze_batch((_msg("哈哈"),), snap)
    assert isinstance(r, BatchSignals)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.texts = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.has_question = False  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.has_other_assistant = False  # type: ignore[misc]