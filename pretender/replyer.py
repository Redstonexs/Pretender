"""Phase 3 replyer: render the bot's visible reply text.

This module owns ONLY the reply lane: it renders the ``replyer`` prompt
from the planner's staged ``reply_reference`` and returns a typed, frozen
``ReplyDraft``. It never invokes tools and never sends output — the only
LLM call is ``LLMClient.complete`` on profile ``"reply"`` — and its
transcript never contains planner analysis or tool JSON: the user turn
carries only the staged reply-reference text.

The model's raw output is guarded before it can reach a user: empty /
non-string content, code-fenced blocks wrapping structured JSON, and
anything that parses as a JSON object/array (planner analysis / tool
output) degrade to a safe no-output draft rather than ever being exposed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pretender.prompts import PromptStore
from pretender.seams import LLMClient
from pretender.types import TranscriptMessage

REPLY_PROFILE = "reply"
REPLYER_PROMPT = "replyer.txt"

__all__ = ["REPLY_PROFILE", "REPLYER_PROMPT", "ReplyDraft", "Replyer"]


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
        temperature: float | None = None,
        max_tokens: int | None = None,
        deadline: float | None = None,
    ) -> ReplyDraft:
        """Render a reply draft from the planner's staged reference.

        An empty / missing ``reply_reference`` short-circuits to a safe
        no-output draft without calling the LLM. The transcript is exactly
        ``[system(replyer prompt), user(reply_reference)]`` — planner
        analysis and tool JSON never enter it.
        """
        if not isinstance(reply_reference, str) or not reply_reference.strip():
            return ReplyDraft.empty(reply_to=reply_to)
        if not isinstance(identity, str) or not isinstance(reply_style, str):
            raise ValueError("identity and reply_style must be strings")
        prompt_text = self._prompts.render(
            self._prompt_name,
            identity=identity,
            reply_style=reply_style,
            reply_reference=reply_reference,
        )
        transcript = [
            TranscriptMessage(role="system", content=prompt_text),
            TranscriptMessage(role="user", content=reply_reference),
        ]
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
