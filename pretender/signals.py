"""Stateless text signals for the turn gate (PLAN.md §1.B; frozen gate spec).

Pure and dependency-free: stdlib only, no config, no storage, no clock.
The gate lane consumes these helpers to score pending messages with zero
LLM cost:

- ``normalize_text`` — the four strip targets: quote-prefix lines,
  structured/visible @ mention forms, media placeholders, and forwarded
  blocks. Non-mutating and idempotent.
- ``is_chinese_question`` / ``is_direct_request`` / ``is_opinion_solicit``
  — the three content detectors (+15 / +20 / +20 in the frozen score).
- ``is_short_reaction`` / ``is_short_reaction_batch`` — the −25
  whole-batch-short-reactions penalty. A batch is eligible only when it
  is nonempty and every normalized text is nonempty and a short
  reaction; an empty batch is never eligible.
- ``OTHER_ASSISTANT_RE`` / ``is_other_assistant_target`` — the exact
  PLAN.md refusal form ``^(DeepSeek|ChatGPT|Grok|豆包|千问|Kimi|Claude)
  [,，、\\s]`` (case-sensitive, separator required).
- ``analyze_batch`` — one frozen ``BatchSignals`` per pending batch:
  normalized texts, aggregated signal booleans, total code-point length,
  and the EXCLUSIVE length bonus (10 for 120+, else 5 for 40+, else 0).

Direct-address facts are NEVER inferred from text here: ``BatchSignals``
copies ``has_direct_at`` / ``has_quote_to_self`` / ``has_other_assistant``
from the ``GateSnapshot``'s structured fields, which are authoritative
(frozen gate contract).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from pretender.types import GateSnapshot, Message

# ── Normalizer: the four strip targets ─────────────────────────────────────

# A quote-prefix line: a full-line 「...」 quote block, a bracketed quote
# marker ([引用]/[回复]/[quote]) with anything after it, or 引用/引用：
# followed by content. Only whole lines are dropped.
_QUOTE_LINE_RE = re.compile(
    r"^(?:「[^\n]*」|\[(?:引用|回复|quote)\][^\n]*|引用[:：][^\n]*)$",
    re.IGNORECASE,
)

# Structured platform codes: every [CQ:...] token is a platform artifact
# (at, image, video, ...), never user content.
_CQ_CODE_RE = re.compile(r"\[CQ:[^\]]*\]", re.IGNORECASE)

# Structured mention forms: [at:123]Name[/at], [at]Name[/at],
# [mention]Name[/mention], a bare [at:123] token, a bare [at]/[mention]
# token, and @{123:Name}.
_STRUCTURED_AT_RE = re.compile(
    r"\[(?:at|mention)[^\]]*\][^\n]*?\[/(?:at|mention)\]"
    r"|\[at:[^\]]*\]"
    r"|\[(?:at|mention)\]"
    r"|@\{[^}]*\}",
    re.IGNORECASE,
)

# Visible mention: @ followed by a name run. The lookbehind keeps email
# addresses (a@b.com) intact — an @ after an ASCII letter/digit is not a
# mention.
_VISIBLE_AT_RE = re.compile(r"(?<![A-Za-z0-9])@[^\s@,，。！？!?、；;：:]+")

# Media placeholders: [图片], [动画表情], [语音], [image], ... Only exact
# placeholder tokens are stripped — [这个视频不错] is content and survives.
_MEDIA_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        # QQ/OneBot visible placeholders
        "图片", "表情", "表情包", "动画表情", "语音", "视频", "文件", "音乐",
        "链接", "位置", "红包", "转账", "名片", "骰子", "戳一戳", "礼物",
        "点赞", "收藏", "回复", "语音通话", "视频通话", "位置共享",
        # English renderings
        "image", "picture", "photo", "sticker", "face", "emoji", "voice",
        "audio", "record", "video", "file", "attachment", "music", "song",
        "link", "url", "location", "forward", "chat history",
    }
)
_MEDIA_TOKEN_RE = re.compile(r"\[([^\]\n]+)\]")

# A forward marker line: exactly [X] with X in the marker set.
_FORWARD_LINE_RE = re.compile(r"^\[([^\]]+)\]$", re.IGNORECASE)
_FORWARD_MARKERS: frozenset[str] = frozenset(
    {
        "转发消息", "聊天记录", "群聊记录", "转发",
        "forward", "forwarded", "forwarded messages", "chat history",
    }
)


def _drop_media(match: re.Match[str]) -> str:
    inner = match.group(1).strip().casefold()
    return "" if inner in _MEDIA_PLACEHOLDERS else match.group(0)


def _drop_forward_blocks(lines: list[str]) -> list[str]:
    """Drop forward marker lines and the block that follows each.

    A forward marker is a line that is exactly ``[X]`` with ``X`` in the
    marker set (``[转发消息]``, ``[聊天记录]``, ...). The block runs from
    the marker line through the next blank line, or to the end of the
    text when no blank line follows.
    """
    kept: list[str] = []
    i = 0
    while i < len(lines):
        match = _FORWARD_LINE_RE.match(lines[i].strip())
        if match and match.group(1).strip().casefold() in _FORWARD_MARKERS:
            i += 1  # drop the marker line itself
            while i < len(lines) and lines[i].strip():
                i += 1  # drop the block: everything up to a blank line
            continue
        kept.append(lines[i])
        i += 1
    return kept


def normalize_text(text: str) -> str:
    """Non-mutating normalization: strip the four targets, return a new string.

    Quote-prefix lines and forwarded blocks are dropped whole; mention
    forms and media placeholders are removed token-wise. The input is
    never modified, and the result is idempotent: normalizing twice
    yields the same string.
    """
    kept: list[str] = []
    for line in _drop_forward_blocks(text.split("\n")):
        if _QUOTE_LINE_RE.match(line):
            continue
        line = _CQ_CODE_RE.sub("", line)
        line = _STRUCTURED_AT_RE.sub("", line)
        line = _VISIBLE_AT_RE.sub("", line)
        line = _MEDIA_TOKEN_RE.sub(_drop_media, line).strip()
        if line:
            kept.append(line)
    return "\n".join(kept)


# ── Content detectors ──────────────────────────────────────────────────────

_QUESTION_WORDS: tuple[str, ...] = (
    "什么", "怎么", "怎样", "怎么样", "为什么", "为啥", "为何", "如何",
    "哪里", "哪儿", "哪个", "哪些", "谁", "多少", "啥", "咋", "是否",
    "能不能", "可不可以", "有没有", "是不是", "会不会", "要不要", "该不该",
    "几点", "几个", "多久", "何时", "何地",
)
_QUESTION_ENDERS: tuple[str, ...] = ("吗", "呢", "嘛")


def is_chinese_question(text: str) -> bool:
    """True when the text asks a question in Chinese.

    A question mark (full- or half-width), a trailing question particle
    (吗/呢/嘛), or any question word from the curated list marks a
    question. Heuristic by design: the gate's +15 is a soft bonus, not a
    hard trigger.
    """
    if not text:
        return False
    if "？" in text or "?" in text:
        return True
    if text.endswith(_QUESTION_ENDERS):
        return True
    return any(word in text for word in _QUESTION_WORDS)


_REQUEST_PHRASES: tuple[str, ...] = (
    "帮我", "麻烦", "拜托", "劳驾", "求求", "求你", "帮个忙", "帮我个忙",
    "能不能帮我", "可以帮我", "能帮我", "帮我一下", "帮我看看", "帮我查",
    "帮我写", "帮我找", "帮我算", "帮我做", "帮我改", "帮我弄", "帮我翻译",
    "帮我推荐", "帮我介绍", "帮我解释", "帮我分析", "帮我总结", "帮我整理",
    "帮我查一下",
)
_REQUEST_VERBS: tuple[str, ...] = (
    "帮", "问", "看", "说", "讲", "教", "给", "推荐", "介绍", "解释",
    "分析", "总结", "翻译", "写", "做", "找", "查", "算", "改", "弄",
    "指导", "建议", "告诉", "回答", "回复", "分享", "转", "发",
)


def is_direct_request(text: str) -> bool:
    """True when the text directly asks the bot to do something.

    Request phrases (帮我/麻烦/拜托/...) or 请/求 immediately followed by
    an action verb (请问/求推荐/请翻译/...). Deliberately narrow: 请假,
    追求 and 要求 are NOT requests.
    """
    if not text:
        return False
    if any(phrase in text for phrase in _REQUEST_PHRASES):
        return True
    return any("请" + verb in text or "求" + verb in text for verb in _REQUEST_VERBS)


_OPINION_PHRASES: tuple[str, ...] = (
    "觉得", "认为", "怎么看", "怎么想", "意见", "想法", "看法", "观点",
    "评价", "好不好", "行不行", "值不值得", "喜不喜欢", "哪个好", "哪个更好",
    "选哪个", "选谁", "说说", "聊聊", "讨论一下", "讨论讨论", "大家说",
    "你们说", "给点意见", "给个建议", "有什么推荐", "有什么建议", "推荐吗",
    "推荐一下", "怎么样", "投票", "支持谁", "谁更好", "谁厉害", "谁强",
    "谁牛", "站哪边", "觉得怎么样",
)


def is_opinion_solicit(text: str) -> bool:
    """True when the text asks the group for opinions or a decision.

    Opinion vocabulary (觉得/认为/怎么看/怎么样/推荐一下/投票/...).
    Heuristic by design: statements like 我觉得... also contain 觉得 and
    may over-trigger the soft +20 bonus.
    """
    if not text:
        return False
    return any(phrase in text for phrase in _OPINION_PHRASES)


# ── Short reactions ────────────────────────────────────────────────────────

_REACTION_WORDS: frozenset[str] = frozenset(
    {
        # laughter and interjections (patterns also cover these)
        "哈哈", "嘿嘿", "呵呵", "嘻嘻", "嗯", "嗯嗯", "哦", "哦哦", "啊", "啊啊",
        "额", "呃", "诶", "哎", "嗨", "哈", "嘿", "嘻", "呵",
        # acknowledgments
        "好的", "好滴", "好嘞", "好耶", "好", "行", "中", "可以", "收到",
        "了解", "明白", "知道了", "晓得", "懂了", "ok", "okk", "okay", "get", "got",
        # thanks
        "谢谢", "感谢", "辛苦", "辛苦了", "thx", "thanks", "3q",
        # greetings
        "你好", "hi", "hello", "hey", "早安", "午安", "晚安", "早上好", "晚上好",
        # agreement / disagreement
        "对", "对的", "是", "是的", "没错", "确实", "真实", "同意", "赞成",
        "支持", "反对", "不行", "不要", "好啊", "好呀", "好哦", "好哇",
        # praise
        "不错", "厉害", "牛", "强", "妙", "绝", "棒", "赞", "顶", "牛逼",
        "牛批", "nb", "yyds", "awsl", "xswl", "good", "nice", "great", "cool", "wow",
        # numeric reactions (QQ conventions)
        "666", "233", "+1", "1", "6", "6666", "2333", "66666", "23333",
        "111", "222", "333", "555", "999", "666666",
        # misc
        "泪目", "笑死", "离谱", "蚌埠住了", "绷不住了", "笑不活了",
        "emmm", "emm", "em", "lol", "lmao", "hahaha", "hhh", "hhhh",
    }
)

_LAUGHTER_RE = re.compile(r"^[哈嘿嘻呵呵]+$")
_INTERJECTION_RE = re.compile(r"^[嗯哦啊额呃诶哎噢喔]+$")
_EMOJI_RE = re.compile(
    r"^[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF"
    r"\U0000FE0F\U0000200D]+$"
)
_ASCII_LAUGH_RE = re.compile(r"^h{2,}$")

_MAX_REACTION_LENGTH = 8


def is_short_reaction(text: str) -> bool:
    """True when the text is a short reaction (哈哈/嗯嗯/666/👍/ok/...).

    A reaction is nonempty, at most 8 code points, and either a known
    reaction word (case-insensitive for ASCII), a laughter/interjection
    run, an emoji-only run, or an ASCII laughter run. Questions and
    requests are never reactions (在吗 is a question, not a reaction).
    """
    if not text or len(text) > _MAX_REACTION_LENGTH:
        return False
    if text.casefold() in _REACTION_WORDS:
        return True
    return bool(
        _LAUGHTER_RE.fullmatch(text)
        or _INTERJECTION_RE.fullmatch(text)
        or _EMOJI_RE.fullmatch(text)
        or _ASCII_LAUGH_RE.fullmatch(text)
    )


def is_short_reaction_batch(texts: Sequence[str]) -> bool:
    """True when the WHOLE batch is short reactions.

    Eligibility: the batch must be nonempty and every normalized text
    must be nonempty and a short reaction. An empty batch — or any empty
    normalized text — is never eligible.
    """
    return bool(texts) and all(is_short_reaction(t) for t in texts)


# ── Other-assistant refusal ────────────────────────────────────────────────

# The exact PLAN.md §1.B refusal form: one of the known assistant names,
# case-sensitive, immediately followed by a comma, Chinese comma,
# ideographic comma, or whitespace. A bare name or any other separator
# (colon, period, ...) does NOT match.
OTHER_ASSISTANT_RE = re.compile(
    r"^(?:DeepSeek|ChatGPT|Grok|豆包|千问|Kimi|Claude)[,，、\s]"
)


def is_other_assistant_target(text: str) -> bool:
    """True when the text is addressed to a DIFFERENT assistant.

    Matches the frozen PLAN.md form exactly: the text must START with one
    of DeepSeek|ChatGPT|Grok|豆包|千问|Kimi|Claude (exact case) followed
    by a comma, Chinese comma, ideographic comma, or whitespace.
    """
    return bool(OTHER_ASSISTANT_RE.match(text))


# ── Batch analysis ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BatchSignals:
    """One frozen batch analysis result (PLAN.md §1.B content scoring).

    ``texts`` are the per-message normalized texts. The signal booleans
    aggregate over the batch. ``total_length`` is the total code-point
    length of the normalized texts; ``length_bonus`` is the EXCLUSIVE
    tier: 10 for 120+, else 5 for 40+, else 0.

    The three direct-address facts are copied verbatim from the
    ``GateSnapshot``'s structured fields — authoritative, never inferred
    from text.
    """

    texts: tuple[str, ...]
    has_question: bool
    has_direct_request: bool
    has_opinion_solicit: bool
    is_short_reaction_batch: bool
    total_length: int
    length_bonus: int
    has_direct_at: bool
    has_quote_to_self: bool
    has_other_assistant: bool


def _length_bonus(total: int) -> int:
    """Exclusive length tiers: 10 for 120+, else 5 for 40+, else 0."""
    if total >= 120:
        return 10
    if total >= 40:
        return 5
    return 0


def analyze_batch(messages: tuple[Message, ...], snapshot: GateSnapshot) -> BatchSignals:
    """Analyze one pending batch into a frozen ``BatchSignals`` result.

    Normalizes every message text, aggregates the content signals, sums
    the code-point lengths, applies the exclusive length bonus, and
    copies the authoritative direct-address facts from the snapshot.
    """
    texts = tuple(normalize_text(m.text) for m in messages)
    total = sum(len(t) for t in texts)
    return BatchSignals(
        texts=texts,
        has_question=any(is_chinese_question(t) for t in texts),
        has_direct_request=any(is_direct_request(t) for t in texts),
        has_opinion_solicit=any(is_opinion_solicit(t) for t in texts),
        is_short_reaction_batch=is_short_reaction_batch(texts),
        total_length=total,
        length_bonus=_length_bonus(total),
        has_direct_at=snapshot.has_direct_at,
        has_quote_to_self=snapshot.has_quote_to_self,
        has_other_assistant=snapshot.has_other_assistant,
    )