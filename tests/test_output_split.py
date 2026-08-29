"""Tests for the split stage — MaiBot's splitter, ported.

The invariants that matter are behavioural, not structural: a bubble never
ends with the separator that cut it, ``？``/``！`` stay attached to their
sentence, the split varies between replies but is stable on retry, and no
cut ever lands inside a URL, CQ code, quote or code block.
"""

from __future__ import annotations

import random

import pytest

from pretender.output import SplitStage, split_text
from pretender.output.split import _SEPARATORS
from pretender.types import ChatKey, Outgoing, Segment


def _out(text: str, **kw) -> Outgoing:
    return Outgoing(chat_key=ChatKey("qq:group:1"), text=text, **kw)


def _rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


def _variants(text: str, max_split: int = 3, seeds: int = 40) -> set[tuple[str, ...]]:
    """Every distinct split this text produces across many seeds."""
    return {tuple(split_text(text, max_split, (), _rng(s))) for s in range(seeds)}


def _content(s: str) -> str:
    """The text with every separator removed — what must survive a split."""
    return "".join(ch for ch in s if ch not in _SEPARATORS)


# ── the defect that motivated the rewrite ───────────────────────────────────

def test_no_part_ever_ends_with_the_separator_that_cut_it():
    """``我就是麦麦啊，`` as a standalone QQ message is the machine tell."""
    for text in (
        "我就是麦麦啊，你不知道我吗？",
        "收到，谢谢！",
        "今天天气不错啊，我刚出去走了走，感觉挺舒服的，你呢？",
        "第一句。第二句。第三句。",
    ):
        for parts in _variants(text):
            for part in parts:
                assert part[-1] not in _SEPARATORS, (text, parts)


def test_question_and_exclamation_are_not_cut_points():
    """``？`` and ``！`` end a sentence but never split it: with no other
    separator present the reply stays a single message."""
    assert _variants("你不知道我吗？我就是啊！") == {("你不知道我吗？我就是啊！",)}


def test_question_mark_stays_attached_when_a_comma_cuts():
    for parts in _variants("我就是麦麦啊，你不知道我吗？"):
        assert any(p.endswith("？") for p in parts), parts


def test_trailing_period_is_dropped_like_maibot():
    assert split_text("好的。", 3, (), _rng()) == ["好的"]


# ── content preservation ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text",
    [
        "我就是麦麦啊，你不知道我吗？😄",
        "今天天气不错啊，我刚出去走了走，感觉挺舒服的，你呢？",
        "一。二。三。四。五。六。七。八。九。十。",
        "第一行\n第二行\n第三行",
    ],
)
def test_splitting_loses_nothing_but_separators(text):
    for parts in _variants(text):
        assert _content("".join(parts)) == _content(text), parts


def test_parts_are_never_empty_and_respect_max_split():
    text = "一。二。三。四。五。六。七。八。九。十。"
    for max_split in (2, 3, 4, 5):
        for parts in _variants(text, max_split):
            assert 0 < len(parts) <= max_split
            assert all(p.strip() for p in parts)


def test_empty_text_returns_empty():
    assert split_text("", 3, (), _rng()) == []
    assert split_text("   ", 3, (), _rng()) == []


def test_max_split_one_returns_single_part():
    assert split_text("第一句。第二句。", 1, (), _rng()) == ["第一句。第二句。"]


# ── the probabilistic behaviour ─────────────────────────────────────────────

def test_the_same_text_does_not_always_split_the_same_way():
    """The variability is the point: a deterministic splitter reads as a
    machine even when every individual cut is defensible."""
    assert len(_variants("我就是麦麦啊，你不知道我吗？")) > 1


def test_a_short_reply_is_usually_left_whole():
    """split_strength 0.2 below 12 chars: mostly one bubble, sometimes two."""
    outcomes = [split_text("收到，谢谢！", 3, (), _rng(s)) for s in range(200)]
    whole = sum(len(o) == 1 for o in outcomes)
    assert whole > len(outcomes) * 0.6


def test_a_long_reply_is_usually_split():
    """split_strength 0.7 above 32 chars."""
    text = "今天天气真的很不错啊，我刚刚出去走了走，感觉整个人都清爽了，你要不要也出去转转？"
    outcomes = [split_text(text, 3, (), _rng(s)) for s in range(200)]
    assert sum(len(o) > 1 for o in outcomes) > len(outcomes) * 0.6


def test_same_seed_reproduces_the_same_split():
    text = "今天天气不错啊，我刚出去走了走，感觉挺舒服的，你呢？"
    assert split_text(text, 3, (), _rng(7)) == split_text(text, 3, (), _rng(7))


def test_text_below_three_chars_is_returned_whole():
    # The 1-in-100 explode is MaiBot's; seed 0 does not trigger it.
    assert split_text("在吗", 3, (), _rng(0)) == ["在吗"]


# ── no-cut zones ────────────────────────────────────────────────────────────

def test_never_cuts_inside_a_url():
    text = "看这个 https://example.com/a,b,c 挺好的，你觉得呢？"
    for parts in _variants(text):
        assert any("https://example.com/a,b,c" in p for p in parts), parts


def test_never_cuts_through_platform_markup():
    """Sanitize runs after this stage, so a severed CQ code is never repaired."""
    for parts in _variants("你好[CQ:at,qq=123]世界，再见！"):
        assert any("[CQ:at,qq=123]" in p for p in parts), parts


def test_never_cuts_inside_a_code_block():
    for parts in _variants("试试 ```a, b, c``` 这段，可以吗？"):
        assert any("```a, b, c```" in p for p in parts), parts


def test_never_cuts_inside_a_quoted_passage():
    assert _variants("他说“好的，没问题。”然后就走了。") == {
        ("他说“好的，没问题。”然后就走了",)
    }


def test_never_cuts_a_time_or_a_key_value_pair():
    for parts in _variants("会议在 12:30 开始，别迟到。"):
        assert any("12:30" in p for p in parts), parts


def test_never_cuts_between_alphanumerics():
    for parts in _variants("用 GPT 4 试试，比 Claude 3 好？"):
        assert any("GPT 4" in p for p in parts), parts
        assert any("Claude 3" in p for p in parts), parts


def test_never_cuts_beside_a_dash():
    for parts in _variants("范围是 1 - 100，注意一下。"):
        assert any("1 - 100" in p for p in parts), parts


def test_newline_always_cuts_and_never_merges_across():
    assert _variants("第一行\n第二行\n第三行") == {("第一行", "第二行", "第三行")}


def test_honours_caller_supplied_protected_spans():
    text = "前面，SECRET,TOKEN，后面"
    span = [(text.index("SECRET"), text.index("SECRET") + len("SECRET,TOKEN"))]
    for seed in range(30):
        parts = split_text(text, 3, span, _rng(seed))
        assert any("SECRET,TOKEN" in p for p in parts), parts


# ── the stage ───────────────────────────────────────────────────────────────

def test_stage_sets_parts_group_and_pacing():
    out = _out("今天天气不错啊，我刚出去走了走，感觉挺舒服的，你呢？", idem_key="k1")
    SplitStage(max_split=3).apply(out)
    assert out.parts and len(out.parts) > 1
    assert out.group_id and out.group_id.startswith("g:")
    pacing = out.platform_ref["part_pacing"]
    assert len(pacing) == len(out.parts) and pacing[0] == 0.0


def test_stage_is_stable_for_the_same_idem_key():
    """A retried finish must regroup identically or the outbox forks."""
    text = "今天天气不错啊，我刚出去走了走，感觉挺舒服的，你呢？"
    first, second = _out(text, idem_key="same"), _out(text, idem_key="same")
    SplitStage(max_split=3).apply(first)
    SplitStage(max_split=3).apply(second)
    assert first.parts == second.parts
    assert first.group_id == second.group_id


def test_stage_varies_between_different_replies():
    text = "今天天气不错啊，我刚出去走了走，感觉挺舒服的，你呢？"
    seen = set()
    for i in range(30):
        out = _out(text, idem_key=f"k{i}")
        SplitStage(max_split=3).apply(out)
        seen.add(tuple(out.parts or [out.text]))
    assert len(seen) > 1


def test_stage_adopts_a_single_part_when_it_differs():
    out = _out("好的。", idem_key="k")
    SplitStage(max_split=3).apply(out)
    assert out.parts is None
    assert out.text == "好的"


def test_already_split_is_not_resplit():
    out = _out("第一句。第二句。", parts=["预先", "分好"])
    SplitStage(max_split=3).apply(out)
    assert out.parts == ["预先", "分好"]


def test_segmented_outgoing_is_not_text_split():
    out = _out("第一句。第二句。", segments=[Segment(kind="text", data={"text": "x"})])
    SplitStage(max_split=3).apply(out)
    assert out.parts is None
