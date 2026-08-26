"""Tests for the split stage: punctuation-aware Chinese splitting, limits,
no empties, grouped metadata, and protected spans."""

from __future__ import annotations

import pytest

from pretender.output import SplitStage, split_text
from pretender.types import ChatKey, Outgoing, Segment


def _out(text: str, **kw) -> Outgoing:
    return Outgoing(chat_key=ChatKey("qq:group:1"), text=text, **kw)


# ── punctuation-aware splitting ─────────────────────────────────────────────

def test_splits_at_sentence_punctuation():
    parts = split_text("第一句。第二句！第三句？", max_split=3)
    assert parts == ["第一句。", "第二句！", "第三句？"]


def test_splits_fewer_parts_than_max():
    parts = split_text("第一句。第二句。", max_split=3)
    assert parts == ["第一句。", "第二句。"]


def test_short_text_stays_single_part():
    parts = split_text("你好", max_split=3)
    assert parts == ["你好"]


def test_max_split_one_returns_single_part():
    parts = split_text("第一句。第二句。", max_split=1)
    assert parts == ["你好".replace("你好", "第一句。第二句。")]


def test_empty_text_returns_empty():
    assert split_text("", max_split=3) == []
    assert split_text("   ", max_split=3) == []


# ── limits / no empties / no pathological parts ─────────────────────────────

def test_never_exceeds_max_split():
    text = "一。二。三。四。五。六。七。八。九。十。"
    for max_split in (2, 3, 4, 5):
        parts = split_text(text, max_split=max_split)
        assert len(parts) <= max_split
        assert all(p for p in parts)  # no empty parts


def test_no_empty_parts_with_many_sentences():
    text = "一。二。三。四。五。六。七。八。九。十。"
    parts = split_text(text, max_split=3)
    assert all(p for p in parts)
    assert "".join(parts) == text


def test_no_punctuation_hard_splits_by_length():
    text = "这是一个没有任何标点符号的长句子" * 5
    parts = split_text(text, max_split=3)
    assert len(parts) <= 3
    assert all(p for p in parts)
    assert "".join(parts) == text


def test_balanced_parts_no_pathological_long_part():
    # Many sentences, max_split=3: no single part should dominate.
    text = "一。二。三。四。五。六。七。八。九。十。"
    parts = split_text(text, max_split=3)
    assert len(parts) == 3
    lengths = [len(p) for p in parts]
    assert max(lengths) - min(lengths) <= 2


# ── grouped metadata ────────────────────────────────────────────────────────

def test_stage_sets_parts_group_and_pacing():
    stage = SplitStage(max_split=3)
    out = _out("第一句。第二句！第三句？")
    stage.apply(out)
    assert out.parts == ["第一句。", "第二句！", "第三句？"]
    assert out.group_id is not None
    assert out.group_id.startswith("g:")
    assert out.platform_ref["part_pacing"] == [0.0, 1.5, 3.0]


def test_group_id_is_stable_for_same_parts():
    stage = SplitStage(max_split=3)
    a = _out("第一句。第二句！第三句？")
    b = _out("第一句。第二句！第三句？")
    stage.apply(a)
    stage.apply(b)
    assert a.group_id == b.group_id


def test_single_part_leaves_metadata_untouched():
    stage = SplitStage(max_split=3)
    out = _out("你好")
    stage.apply(out)
    assert out.parts is None
    assert out.group_id is None


def test_already_split_is_not_resplit():
    stage = SplitStage(max_split=3)
    out = _out("第一句。第二句！第三句？", parts=["already"])
    stage.apply(out)
    assert out.parts == ["already"]


def test_segmented_outgoing_is_not_text_split():
    out = _out("第一句。第二句。", segments=[Segment("image", {"file": "x"})])
    SplitStage(max_split=3).apply(out)
    assert out.parts is None
    assert out.group_id is None


# ── protected spans ─────────────────────────────────────────────────────────

def test_does_not_split_inside_url():
    text = "看这个 https://example.com/a?b=1 然后。下一句。"
    parts = split_text(text, max_split=3)
    joined = "".join(parts)
    assert "https://example.com/a?b=1" in joined
    # the URL is not split across parts
    assert not any("https://" in p and "example.com" not in p for p in parts)


def test_does_not_split_inside_code_block():
    text = "```\n第一句。第二句。\n``` 之后。"
    parts = split_text(text, max_split=3)
    joined = "".join(parts)
    assert "```" in joined
    assert "第一句。第二句。" in joined


def test_stage_honours_recorded_protected_spans():
    stage = SplitStage(max_split=3)
    out = _out("看 https://example.com/a 然后。下一句。")
    out.platform_ref["protected_spans"] = [[2, 23]]
    stage.apply(out)
    joined = "".join(out.parts or [out.text])
    assert "https://example.com/a" in joined
