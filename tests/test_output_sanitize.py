"""Tests for the sanitize stage: leakage / CQ / tool-analysis / unsafe markup
removal without corrupting ordinary text, URLs, or protected spans."""

from __future__ import annotations

import pytest

from pretender.output import SanitizeStage, sanitize_text
from pretender.types import ChatKey, Outgoing


def _out(text: str) -> Outgoing:
    return Outgoing(chat_key=ChatKey("qq:group:1"), text=text)


# ── removal patterns ────────────────────────────────────────────────────────

def test_removes_cq_code():
    cleaned, _ = sanitize_text("你好[CQ:at,qq=123]再见")
    assert cleaned == "你好再见"


def test_removes_thinking_block():
    cleaned, _ = sanitize_text("你好<thinking>内部推理</thinking>再见")
    assert cleaned == "你好再见"


def test_removes_analysis_block():
    cleaned, _ = sanitize_text("你好<analysis>分析内容</analysis>再见")
    assert cleaned == "你好再见"


def test_removes_tool_block():
    cleaned, _ = sanitize_text("你好<tool_result>结果</tool_result>再见")
    assert cleaned == "你好再见"


def test_removes_tool_bracket_block():
    cleaned, _ = sanitize_text("你好[tool_call]调用[/tool_call]再见")
    assert cleaned == "你好再见"


def test_removes_unsafe_markup():
    cleaned, _ = sanitize_text("你好<script>alert(1)</script>再见")
    assert cleaned == "你好再见"


def test_removes_leakage_line():
    cleaned, _ = sanitize_text("我的思考：这个问题很难\n你好")
    assert cleaned == "你好"


def test_removes_multiple_patterns_at_once():
    cleaned, _ = sanitize_text(
        "我的分析：先想想\n[CQ:at,qq=1]<thinking>x</thinking>你好"
    )
    assert cleaned == "你好"


# ── ordinary text preservation ──────────────────────────────────────────────

def test_normal_text_unchanged():
    text = "今天天气真好，我们去公园散步吧！"
    cleaned, _ = sanitize_text(text)
    assert cleaned == text


def test_punctuation_and_numbers_preserved():
    text = "价格是 3 < 5，对吗？是的，没错。"
    cleaned, _ = sanitize_text(text)
    assert cleaned == text


def test_math_comparison_not_treated_as_markup():
    # "< 5" must not be stripped as an HTML tag.
    cleaned, _ = sanitize_text("a < b 且 c > d")
    assert cleaned == "a < b 且 c > d"


# ── URLs / protected spans ──────────────────────────────────────────────────

def test_url_preserved():
    text = "看这个 https://example.com/path?q=1 怎么样"
    cleaned, _ = sanitize_text(text)
    assert "https://example.com/path?q=1" in cleaned


def test_url_trailing_punctuation_not_swallowed():
    text = "看 https://example.com/a。然后呢"
    cleaned, _ = sanitize_text(text)
    assert "https://example.com/a" in cleaned
    assert "。然后呢" in cleaned


def test_cq_inside_code_block_preserved():
    text = "```\n[CQ:at,qq=1]\n``` 你好"
    cleaned, _ = sanitize_text(text)
    assert "[CQ:at,qq=1]" in cleaned
    assert "你好" in cleaned


def test_thinking_inside_inline_code_preserved():
    text = "用 `x <thinking>y</thinking>` 表示"
    cleaned, _ = sanitize_text(text)
    assert "<thinking>y</thinking>" in cleaned


def test_protected_spans_recorded_on_outgoing():
    stage = SanitizeStage()
    out = _out("看 https://example.com/a 然后")
    stage.apply(out)
    assert "protected_spans" in out.platform_ref
    spans = out.platform_ref["protected_spans"]
    assert spans  # at least the URL span


def test_sanitize_stage_returns_same_outgoing():
    stage = SanitizeStage()
    out = _out("你好[CQ:at,qq=1]")
    assert stage.apply(out) is out
    assert out.text == "你好"


# ── Gate 4: variants / unterminated blocks / parts ─────────────────────────

def test_removes_uppercase_thinking_block():
    cleaned, _ = sanitize_text("你好<THINKING>内部</THINKING>再见")
    assert cleaned == "你好再见"


def test_removes_whitespace_variant_thinking_block():
    cleaned, _ = sanitize_text("你好< thinking >内部</ thinking >再见")
    assert cleaned == "你好再见"


def test_removes_unterminated_thinking_block():
    # No closing tag: the leaked block is removed through end-of-text.
    cleaned, _ = sanitize_text("你好<thinking>内部推理")
    assert cleaned == "你好"


def test_removes_unterminated_tool_block():
    cleaned, _ = sanitize_text("你好<tool_call>调用参数")
    assert cleaned == "你好"


def test_removes_unterminated_tool_bracket_block():
    cleaned, _ = sanitize_text("你好[tool_result]结果")
    assert cleaned == "你好"


def test_removes_uppercase_unterminated_reasoning():
    cleaned, _ = sanitize_text("开头<REASONING>泄露内容")
    assert cleaned == "开头"


def test_normal_text_with_angle_brackets_untouched():
    # "< b" / "c >" must not be treated as a reasoning/tool/unsafe tag.
    cleaned, _ = sanitize_text("a < b 且 c > d")
    assert cleaned == "a < b 且 c > d"


def test_stage_sanitizes_already_created_parts():
    stage = SanitizeStage()
    out = _out("第一句。第二句。")
    out.parts = ["你好<thinking>内部</thinking>再见", "正常内容"]
    stage.apply(out)
    assert out.parts == ["你好再见", "正常内容"]
    assert out.text == "第一句。第二句。"


def test_stage_sanitizes_parts_preserving_protected_spans():
    stage = SanitizeStage()
    out = _out("全文")
    out.parts = ["看 https://example.com/a <thinking>x</thinking> 好"]
    stage.apply(out)
    assert out.parts == ["看 https://example.com/a  好"]
