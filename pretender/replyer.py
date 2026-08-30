"""Phase 3 replyer: render the bot's visible reply text.

This module owns ONLY the reply lane: it renders the ``replyer`` prompt and
returns a typed, frozen ``ReplyDraft``. It never invokes tools and never
sends output — the only LLM call is ``LLMClient.complete`` on profile
``"reply"``.

**The replyer sees the conversation.** It used to receive nothing but the
planner's staged ``reply_reference`` string, which made it a paraphraser:
asked to write a group-chat message with no idea what the group was talking
about, what time it was, or who it was answering. That is the difference
between a reply and a plausible-sounding sentence.

The request now mirrors MaiBot's
``maisaka_generator_base._build_request_messages``:

  1. system — the ``replyer`` prompt with identity, reply style, bot name and
     the attention-drift block
  2. the recent chat as REAL role-tagged turns (``is_self`` → assistant,
     everyone else → user), so the model reads the conversation rather than a
     summary of it
  3. a final user turn carrying the current time, the message being replied
     to, what the bot thinks of the people in the window, the planner's
     non-binding reference, any length directive, and the output instruction

Planner analysis and tool JSON still never enter the transcript — only real
chat messages and the staged reference do.

The model's raw output is guarded before it can reach a user: empty /
non-string content, code-fenced blocks wrapping structured JSON, and
anything that parses as a JSON object/array (planner analysis / tool
output) degrade to a safe no-output draft rather than ever being exposed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from pretender.prompts import PromptStore
from pretender.seams import LLMClient
from pretender.types import Message, TranscriptMessage

REPLY_PROFILE = "reply"
REPLYER_PROMPT = "replyer.txt"

#: MaiBot's ``reply`` tool length directives, keyed by its own argument
#: values. The PLANNER decides how long the reply should be; the replyer is
#: told, rather than guessing from the reference text.
LENGTH_DIRECTIVES = {
    "简短表达": (
        "请简短的回复，允许句子残缺，奇怪表达，倒装，省略，"
        "符合口语习惯，符合省力随意回复习惯"
    ),
    "正常回复": "",
    "长回复": "可以针对问题做出较为详细的评论和说明",
}

__all__ = [
    "LENGTH_DIRECTIVES",
    "REPLY_PROFILE",
    "REPLYER_PROMPT",
    "ReplyContext",
    "ReplyDraft",
    "Replyer",
]


@dataclass(frozen=True)
class ReplyContext:
    """Everything the replyer needs beyond the planner's reference.

    Every field is optional so an injected/test Replyer keeps working with no
    context at all — it simply degrades to the old reference-only request.
    """

    chat_history: tuple[Message, ...] = ()
    target: Message | None = None
    bot_name: str = ""
    now: float | None = None
    drift_block: str = ""
    length_style: str = ""
    #: What the bot has come to think of the people in this window, as
    #: ``(display name, impression)`` pairs. Knowing who you are talking to
    #: is most of what separates a regular from a stranger; without it every
    #: conversation restarts from zero.
    impressions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ReplyDraft:
    """The replyer's typed output.

    ``no_output`` is True when nothing usable was produced (empty /
    malformed model content) — callers must treat it as "send nothing".
    ``usage`` / ``tokens_in`` / ``tokens_out`` carry the provider usage.
    """

    text: str = ""
    reply_to: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("ReplyDraft.text must be a string")
        if self.reply_to is not None and not isinstance(self.reply_to, str):
            raise ValueError("ReplyDraft.reply_to must be a string or None")
        for name in ("tokens_in", "tokens_out"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"ReplyDraft.{name} must be a nonnegative integer")
        usage = {
            k: v
            for k, v in (self.usage or {}).items()
            if isinstance(v, int) and not isinstance(v, bool)
        }
        object.__setattr__(self, "usage", usage)

    @property
    def no_output(self) -> bool:
        """True when the draft carries no user-facing text."""
        return not self.text.strip()

    @classmethod
    def empty(
        cls,
        *,
        reply_to: str | None = None,
        usage: dict[str, int] | None = None,
    ) -> "ReplyDraft":
        """The safe no-output draft (empty text)."""
        return cls(text="", reply_to=reply_to, usage=usage or {})


class Replyer:
    """Render the bot's visible reply from the planner's staged reference.

    ``llm`` is the ``LLMClient`` (profile ``"reply"`` by default);
    ``prompts`` is the ``PromptStore``. Only ``complete`` is ever called —
    no tools, no output sending.
    """

    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptStore,
        *,
        profile: str = REPLY_PROFILE,
        prompt_name: str = REPLYER_PROMPT,
    ) -> None:
        if llm is None or prompts is None:
            raise ValueError("llm and prompts are required")
        self._llm = llm
        self._prompts = prompts
        self._profile = profile
        self._prompt_name = prompt_name

    async def reply(
        self,
        *,
        reply_reference: str,
        identity: str,
        reply_style: str,
        reply_to: str | None = None,
        context: ReplyContext | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        deadline: float | None = None,
    ) -> ReplyDraft:
        """Render a reply draft.

        An empty / missing ``reply_reference`` short-circuits to a safe
        no-output draft without calling the LLM. ``context`` carries the chat
        history, target message, clock and drift block; omitting it degrades
        to the reference-only request.
        """
        if not isinstance(reply_reference, str) or not reply_reference.strip():
            return ReplyDraft.empty(reply_to=reply_to)
        if not isinstance(identity, str) or not isinstance(reply_style, str):
            raise ValueError("identity and reply_style must be strings")
        ctx = context or ReplyContext()
        prompt_text = self._prompts.render(
            self._prompt_name,
            identity=identity,
            reply_style=reply_style,
            bot_name=ctx.bot_name,
            drift_block=ctx.drift_block,
        )
        transcript = [TranscriptMessage(role="system", content=prompt_text)]
        transcript.extend(_history_turns(ctx.chat_history, ctx.bot_name))
        transcript.append(
            TranscriptMessage(
                role="user", content=_final_turn(reply_reference, ctx)
            )
        )
        resp = await self._llm.complete(
            transcript,
            profile=self._profile,
            temperature=temperature,
            max_tokens=max_tokens,
            deadline=deadline,
        )
        text = _clean_text(resp.content)
        usage = {
            k: v
            for k, v in (resp.usage or {}).items()
            if isinstance(v, int) and not isinstance(v, bool)
        }
        if not text:
            return ReplyDraft.empty(reply_to=reply_to, usage=usage)
        tokens_in = int(usage.get("prompt_tokens", 0))
        tokens_out = int(usage.get("completion_tokens", 0))
        return ReplyDraft(
            text=text,
            reply_to=reply_to,
            usage=usage,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


# ── request assembly ─────────────────────────────────────────────────────────


def _clock_time(ts: float | None) -> str:
    """``HH:MM`` for one message line."""
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(ts).strftime("%H:%M")
    except (OverflowError, OSError, ValueError):
        return ""


def _history_turns(
    history: Sequence[Message], bot_name: str
) -> list[TranscriptMessage]:
    """The recent chat as role-tagged turns.

    The bot's own messages become ``assistant`` turns so the model sees its
    own voice in the conversation rather than a transcript describing it.
    Everyone else becomes a ``user`` turn prefixed with the speaker and the
    clock time — a group chat has many speakers, and a reply that ignores who
    said what reads as a bot.
    """
    turns: list[TranscriptMessage] = []
    for msg in history:
        text = (msg.text or "").strip()
        if not text:
            continue
        if msg.is_self:
            turns.append(TranscriptMessage(role="assistant", content=text))
            continue
        clock = _clock_time(msg.recv_ts)
        name = msg.sender_name or bot_name or "某人"
        prefix = f"[{clock}] {name}: " if clock else f"{name}: "
        turns.append(TranscriptMessage(role="user", content=prefix + text))
    return turns


def _final_turn(reply_reference: str, ctx: ReplyContext) -> str:
    """The closing user turn: time, target, impressions, reference,
    length, instruction."""
    sections: list[str] = []
    if ctx.now is not None:
        try:
            stamp = datetime.fromtimestamp(ctx.now).strftime("%Y-%m-%d %H:%M:%S")
            sections.append(f"当前时间：{stamp}")
        except (OverflowError, OSError, ValueError):
            pass
    if ctx.target is not None and (ctx.target.text or "").strip():
        speaker = ctx.target.sender_name or "对方"
        sections.append(
            f"【你要回复的消息】\n{speaker}: {ctx.target.text.strip()}"
        )
    if ctx.impressions:
        lines = "\n".join(
            f"{name}: {impression}" for name, impression in ctx.impressions
        )
        sections.append(
            "【你对他们的印象】（这是你自己的观察，不是指令）\n" + lines
        )
    sections.append(f"【回复信息参考】\n{reply_reference.strip()}")
    directive = LENGTH_DIRECTIVES.get(ctx.length_style.strip(), "")
    if directive:
        sections.append(directive)
    sections.append(
        "请自然地回复。只输出你要发到群里的内容本身，"
        "不要输出分析、括号动作描写、@ 或任何额外标记。"
    )
    return "\n\n".join(sections)


# ── output guard ─────────────────────────────────────────────────────────────


def _clean_text(content: Any) -> str:
    """Guard the model's raw content before it can become user-facing.

    Empty / non-string content, code-fenced blocks wrapping structured
    JSON, and anything that parses as a JSON object/array are rejected —
    such content is planner analysis / tool output, never a reply.
    """
    if not isinstance(content, str):
        return ""
    text = content.strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = _strip_fence(text)
    if not text:
        return ""
    if _is_structured_json(text):
        return ""
    return text


def _strip_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _is_structured_json(text: str) -> bool:
    if not (text.startswith(("{", "[")) and text.endswith(("}", "]"))):
        return False
    try:
        json.loads(text)
        return True
    except (ValueError, RecursionError):
        return False
