"""Kaomoji spans: faces are protected, platform markup is not."""

from __future__ import annotations

import pytest

from pretender.output.kaomoji import detect_kaomoji_spans


def _found(text: str) -> list[str]:
    return [text[s:e] for s, e in detect_kaomoji_spans(text)]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("我笑死了(╯°□°)╯真的", ["(╯°□°)"]),
        ("开心 (・ω・) 呢", ["(・ω・)"]),
        ("╥╯╰╥ 难过", ["╥╯╰╥"]),
        ("好的【￣▽￣】走了", ["【￣▽￣】"]),
    ],
)
def test_faces_are_detected(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "你好[CQ:at,qq=123]世界",
        "看图[表情包:1]吧",
        "回复[回复:5]内容",
    ],
)
def test_platform_markup_is_not_a_face(text):
    """Sanitize exists to strip these; treating them as faces would shield
    a leaked CQ code from removal."""
    assert _found(text) == []


def test_parenthesised_chinese_is_not_a_face():
    """``（笑）`` is a stage direction to be stripped, not a face to keep."""
    assert _found("好的（笑）我知道了") == []


def test_plain_text_has_no_faces():
    assert _found("今天天气不错，你觉得呢？") == []
