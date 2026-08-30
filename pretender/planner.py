"""Phase 3 planner: the deterministic tool-loop driver.

This module owns ONLY the planner lane: it renders the ``planner`` /
``planner_focus`` prompt via ``PromptStore``, drives the model tool loop
against a ``ToolRegistry`` (``dispatch_call``), and returns a typed,
immutable ``PlanResult``. It performs no adapter/outbox/network I/O beyond
the injected ``LLMClient``, and it never frames planner analysis as
user-visible reply text — the only text that can become a reply is what the
``reply`` tool staged in the round's ``ToolContext``.

Invariants (the reason this module exists):

  * every assistant tool-call turn in the internal transcript is answered
    by exactly one ``tool`` message per call id, in call order, on EVERY
    exit — including malformed/fuzzed calls and the round cap;
  * the tolerant ``parse_tool_calls`` lane is used ONLY as a fallback for
    malformed / tool-JSON model output (with one injected ``repair``
    attempt per snippet); a response that yields no recoverable tool call
    degrades to ``no_action`` instead of raising or orphaning an id;
  * the loop stops on a staged terminal verdict (``reply`` / ``wait`` /
    ``no_action``) or after ``max_tool_rounds`` tool rounds, whichever
    comes first;
  * tool dispatch never escapes: a raising handler — or a ``dispatch_call``
    that itself raises — becomes an ``ok=False`` ``ToolResult`` for that
    call's id.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from pretender.config import ContextConfig
from pretender.context import build_context
from pretender.prompts import PromptStore
from pretender.seams import LLMClient
from pretender.tools.base import ToolRegistry
from pretender.tools.core import CoreToolRegistry, ToolContext, dispatch_call
from pretender.toolparse import NO_ACTION_NAME, parse_tool_calls
from pretender.types import LLMResponse, ToolCall, ToolCallId, ToolResult, TranscriptMessage

PLANNER_PROFILE = "planner"
PLANNER_PROMPT = "planner.txt"
PLANNER_FOCUS_PROMPT = "planner_focus.txt"
DEFAULT_MAX_TOOL_ROUNDS = 5

__all__ = [
    "PLANNER_PROFILE",
    "PLANNER_PROMPT",
    "PLANNER_FOCUS_PROMPT",
    "DEFAULT_MAX_TOOL_ROUNDS",
    "PlanIntent",
    "PlanRound",
    "PlanResult",
    "Planner",
]


class PlanIntent(str, Enum):
    """The planner's structured terminal intent.

    Only ``reply`` carries user-facing text (the staged
    ``reply_reference``); ``wait`` carries ``wait_seconds``;
    ``no_action`` is the safe default for every malformed / degenerate
    exit.
    """

    REPLY = "reply"
    WAIT = "wait"
    NO_ACTION = "no_action"


@dataclass(frozen=True)
class PlanRound:
    """One planner tool round: the assistant tool-call turn, its tool
    results (exactly one per call id), the messages appended to the
    transcript this round, and the round's raw provider usage."""

    index: int
    response: LLMResponse
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    messages: tuple[TranscriptMessage, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanResult:
    """The planner's typed terminal output.

    ``intent`` is the terminal verdict. For ``reply``, ``reply_reference``
    is the staged reply text and ``reply_to`` the optional quote target;
    for ``wait``, ``wait_seconds`` is the requested pause. ``tool_results``
    is every tool result across all rounds; ``usage`` / ``tokens_in`` /
    ``tokens_out`` aggregate the provider usage; ``transcript`` is the full
    internal transcript (always canonical); ``rounds`` is the per-round
    breakdown. ``end_reason`` names why the loop stopped; ``degraded`` is
    True when a malformed/tool-JSON fallback or a contained dispatch
    failure produced the outcome.
    """

    intent: PlanIntent
    reply_reference: str | None = None
    reply_to: str | None = None
    #: MaiBot's ``reply`` tool ``reply_style``: how long this reply should be
    #: (``简短表达`` / ``正常回复`` / ``长回复``). "" means no preference.
    length_style: str = ""
    wait_seconds: float | None = None
    tool_results: tuple[ToolResult, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    rounds: tuple[PlanRound, ...] = ()
    transcript: tuple[TranscriptMessage, ...] = ()
    end_reason: str | None = None
    degraded: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.intent, PlanIntent):
            raise ValueError(f"intent must be a PlanIntent, got {self.intent!r}")
        if self.reply_reference is not None and not isinstance(
            self.reply_reference, str
        ):
            raise ValueError("reply_reference must be a string or None")
        if self.reply_to is not None and not isinstance(self.reply_to, str):
            raise ValueError("reply_to must be a string or None")
        if not isinstance(self.length_style, str):
            raise ValueError("length_style must be a string")
        if self.wait_seconds is not None and (
            isinstance(self.wait_seconds, bool)
            or not isinstance(self.wait_seconds, (int, float))
        ):
            raise ValueError("wait_seconds must be a number or None")
        for name in ("tokens_in", "tokens_out"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.degraded, bool):
            raise ValueError("degraded must be a bool")


class Planner:
    """Deterministic tool-loop planner (phase 3 capability lane).

    ``llm`` is the ``LLMClient`` (profile ``"planner"`` by default);
    ``prompts`` is the ``PromptStore``; ``registry`` is the
    ``CoreToolRegistry`` / ``ToolRegistry`` used both for provider tool
    definitions and for ``dispatch_call``; ``context_config`` is the
    ``ContextConfig`` the planner applies to the incoming chat transcript.

    Exactly one of ``tool_context_factory`` (called for a FRESH
    ``ToolContext`` per tool round — the production shape, one context per
    round) or ``tool_context`` (a single injected context reused across
    rounds) must be given. ``max_tool_rounds`` caps the loop;
    ``repair`` is the single injected repair callback handed to the
    tolerant fallback parser.
    """

    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptStore,
        registry: CoreToolRegistry | ToolRegistry,
        context_config: ContextConfig,
        *,
        profile: str = PLANNER_PROFILE,
        tool_context_factory: Callable[[], ToolContext] | None = None,
        tool_context: ToolContext | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        repair: Callable[[str], str] | None = None,
    ) -> None:
        if llm is None or prompts is None or registry is None or context_config is None:
            raise ValueError("llm, prompts, registry and context_config are required")
        if (tool_context_factory is None) == (tool_context is None):
            raise ValueError(
                "provide exactly one of tool_context_factory or tool_context"
            )
        if (
            isinstance(max_tool_rounds, bool)
            or not isinstance(max_tool_rounds, int)
            or max_tool_rounds < 1
        ):
            raise ValueError("max_tool_rounds must be a positive integer")
        self._llm = llm
        self._prompts = prompts
        self._registry = registry
        self._context_config = context_config
        self._profile = profile
        self._tool_context_factory = tool_context_factory
        self._tool_context = tool_context
        self._max_tool_rounds = max_tool_rounds
        self._repair = repair

    # ── the loop ────────────────────────────────────────────────────────────

    async def plan(
        self,
        messages: Iterable[TranscriptMessage],
        *,
        chat_log: str,
        reply_style: str,
        focus_chat: str | None = None,
        bot_name: str = "",
        drift_block: str = "",
        behavior_style: str = "",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        deadline: float | None = None,
        max_tool_rounds: int | None = None,
    ) -> PlanResult:
        """Run the planner loop and return the terminal ``PlanResult``.

        ``messages`` is the chat history in transcript form (the caller
        converts inbound ``Message``s); it is normalized/folded/trimmed via
        ``build_context`` and appended after the rendered system prompt.
        ``chat_log`` is the plain-text chat rendering embedded in the
        system prompt. ``focus_chat`` selects the ``planner_focus`` prompt.
        ``bot_name`` is the configured bot name (the prompts address it by
        name rather than hardcoding one), and ``drift_block`` the rendered
        attention-drift rules — drift governs WHICH pending message gets
        latched onto, which is a planner decision, so it belongs in both
        prompts rather than the replyer alone.

        ``behavior_style`` is MaiBot's ``behavior_style``: the persona's
        rules for WHEN to speak and when to stay out of it. The planner
        decides whether to say anything at all, so this — not the replyer's
        ``identity``, which is about how sentences are phrased — is the half
        of the persona it needs. The replyer keeps ``identity``; the planner
        never sees it, exactly as in MaiBot's ``maisaka_chat.prompt``.
        """
        if not isinstance(reply_style, str):
            raise ValueError("reply_style must be a string")
        if not isinstance(chat_log, str):
            raise ValueError("chat_log must be a string")
        cap = max_tool_rounds if max_tool_rounds is not None else self._max_tool_rounds
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError("max_tool_rounds must be a positive integer")

        prompt_name = PLANNER_FOCUS_PROMPT if focus_chat else PLANNER_PROMPT
        variables: dict[str, object] = {
            "chat_log": chat_log,
            "reply_style": reply_style,
            "bot_name": bot_name,
            "drift_block": drift_block,
            "behavior_style": behavior_style,
        }
        if focus_chat is not None:
            variables["focus_chat"] = focus_chat
        prompt_text = self._prompts.render(prompt_name, **variables)

        cc = self._context_config
        transcript: list[TranscriptMessage] = [
            TranscriptMessage(role="system", content=prompt_text),
            *build_context(
                messages,
                max_context_size=cc.max_context_size,
                max_image_num=cc.max_image_num,
                keep_recent=cc.keep_recent,
            ),
        ]
        tool_defs: list[dict[str, Any]] | None = (
            list(tools) if tools is not None else None
        )

        all_results: list[ToolResult] = []
        usage: dict[str, int] = {}
        rounds: list[PlanRound] = []
        reply_reference: str | None = None
        reply_to: str | None = None
        length_style: str = ""
        wait_seconds: float | None = None
        end_reason: str | None = None
        degraded = False
        intent = PlanIntent.NO_ACTION

        for round_index in range(cap):
            # Refresh the provider schema each round so a tool_search
            # activation in a prior round is emitted for the next one.
            round_tools = (
                tool_defs
                if tool_defs is not None
                else self._registry.provider_definitions(scope="all")
            )
            resp = await self._llm.complete(
                list(transcript),
                profile=self._profile,
                tools=round_tools,
                temperature=temperature,
                max_tokens=max_tokens,
                deadline=deadline,
            )
            usage = _merge_usage(usage, resp.usage)
            calls, used_fallback = self._resolve_calls(resp)
            degraded = degraded or used_fallback

            assistant_msg = TranscriptMessage(
                role="assistant",
                content=resp.content,
                tool_calls=tuple(calls),
            )
            transcript.append(assistant_msg)

            ctx = self._new_context()
            results: list[ToolResult] = []
            for call in calls:
                try:
                    result = await dispatch_call(call, ctx, self._registry)
                except Exception as exc:  # containment boundary — never propagate
                    result = ToolResult(
                        call_id=call.id,
                        name=call.name,
                        ok=False,
                        error=f"dispatch failed: {type(exc).__name__}: {exc}",
                    )
                    degraded = True
                results.append(result)

            for call, result in zip(calls, results):
                transcript.append(_tool_message(call, result))

            round_messages: tuple[TranscriptMessage, ...]
            if calls:
                round_messages = tuple(transcript[-(len(calls) + 1) :])
            else:
                round_messages = (assistant_msg,)
            rounds.append(
                PlanRound(
                    index=round_index,
                    response=resp,
                    tool_calls=tuple(calls),
                    tool_results=tuple(results),
                    messages=round_messages,
                    usage=dict(resp.usage or {}),
                )
            )
            all_results.extend(results)

            # Terminal verdict — the ToolContext's staged outcome is
            # last-wins within the round (reply/wait/no_action clear each
            # other in dispatch order).
            if ctx.reply_text is not None:
                intent = PlanIntent.REPLY
                reply_reference = ctx.reply_text
                reply_to = ctx.reply_to
                length_style = getattr(ctx, "reply_length_style", "") or ""
                end_reason = "reply"
                break
            if ctx.wait_seconds is not None:
                intent = PlanIntent.WAIT
                wait_seconds = ctx.wait_seconds
                end_reason = "wait"
                break
            if ctx.no_action_verdict:
                intent = PlanIntent.NO_ACTION
                end_reason = "no_action"
                break
            if not calls:
                intent = PlanIntent.NO_ACTION
                end_reason = (
                    "empty_response" if resp.content is None else "no_tool_call"
                )
                break
        else:
            intent = PlanIntent.NO_ACTION
            end_reason = "tool_round_cap"

        tokens_in, tokens_out = _usage_tokens(usage)
        return PlanResult(
            intent=intent,
            reply_reference=reply_reference,
            reply_to=reply_to,
            length_style=length_style,
            wait_seconds=wait_seconds,
            tool_results=tuple(all_results),
            usage=usage,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            rounds=tuple(rounds),
            transcript=tuple(transcript),
            end_reason=end_reason,
            degraded=degraded,
        )

    # ── internals ───────────────────────────────────────────────────────────

    def _new_context(self) -> ToolContext:
        if self._tool_context_factory is not None:
            ctx = self._tool_context_factory()
            if not isinstance(ctx, ToolContext):
                raise TypeError(
                    "tool_context_factory must return a ToolContext, got "
                    f"{type(ctx).__name__}"
                )
            return ctx
        assert self._tool_context is not None
        return self._tool_context

    def _repair_structured(
        self, cid: Any, name: str, raw_arguments: Any
    ) -> tuple[str, dict[str, Any], Any]:
        """Give the tolerant parser ONE repair/no_action opportunity for a
        malformed structured call whose raw arguments could not be parsed
        into an object at the LLM layer.

        Returns ``(name, arguments, raw_arguments)`` for the reconstructed
        ``ToolCall``: a successful repair yields the repaired arguments (raw
        cleared); a failed repair / no_action degrades the call name to
        ``NO_ACTION_NAME`` so dispatch answers the id with a ``no_action``
        result — the id is never orphaned.
        """
        repaired = parse_tool_calls(
            ToolCall(
                id=ToolCallId(cid),
                name=name,
                arguments={},
                raw_arguments=raw_arguments,
            ),
            self._registry,
            repair=self._repair,
        )
        if repaired and repaired[0].ok:
            data = repaired[0].data
            return name, (data if isinstance(data, dict) else {}), None
        return NO_ACTION_NAME, {}, None

    def _resolve_calls(self, resp: LLMResponse) -> tuple[list[ToolCall], bool]:
        """Deterministic per-round call resolution.

        Structured ``resp.tool_calls`` are used as-is; duplicate call ids
        are collapsed to the first occurrence (a duplicate id could
        otherwise yield a duplicate tool result). When the model emitted NO
        structured calls but the content carries (possibly malformed) tool
        JSON, the tolerant ``parse_tool_calls`` lane recovers call ids so
        they are answered instead of orphaned. Returns
        ``(calls, used_fallback)``.
        """
        seen: set[str] = set()
        calls: list[ToolCall] = []
        used_fallback = False
        for call in resp.tool_calls:
            if isinstance(call, ToolCall):
                cid = call.id
                name = call.name
                if call.raw_arguments is not None:
                    # Malformed structured call: preserve the raw value and
                    # give the tolerant parser ONE repair/no_action
                    # opportunity (the id is never orphaned). A successful
                    # repair dispatches the repaired arguments; a failed
                    # repair degrades the call to no_action.
                    used_fallback = True
                    name, arguments, raw_arguments = self._repair_structured(
                        cid, name, call.raw_arguments
                    )
                else:
                    arguments = (
                        call.arguments if isinstance(call.arguments, dict) else {}
                    )
                    raw_arguments = None
            elif isinstance(call, Mapping):
                raw_id = call.get("id")
                if raw_id is None:
                    continue  # no id — nothing to answer, nothing orphaned
                cid = str(raw_id)
                name = call.get("name")
                if isinstance(name, bool) or not isinstance(name, str):
                    name = ""  # unknown tool → dispatch fails safely
                arguments = call.get("arguments")
                arguments = arguments if isinstance(arguments, dict) else {}
                raw_arguments = call.get("raw_arguments")
                if raw_arguments is not None:
                    used_fallback = True
                    name, arguments, raw_arguments = self._repair_structured(
                        cid, name, raw_arguments
                    )
            else:
                continue  # not a call unit; no id to answer
            cid = str(cid).strip()
            if not cid or cid in seen:
                continue
            seen.add(cid)
            calls.append(
                ToolCall(
                    id=ToolCallId(cid),
                    name=name,
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                )
            )
        if not calls and resp.content:
            recovered = parse_tool_calls(
                resp.content, self._registry, repair=self._repair
            )
            if recovered:
                used_fallback = True
                for result in recovered:
                    cid = str(result.call_id).strip()
                    if not cid or cid in seen:
                        continue
                    seen.add(cid)
                    calls.append(
                        ToolCall(
                            id=ToolCallId(cid),
                            name=result.name or NO_ACTION_NAME,
                            arguments=(
                                result.data if isinstance(result.data, dict) else {}
                            ),
                        )
                    )
        return calls, used_fallback


# ── pure helpers ─────────────────────────────────────────────────────────────


def _tool_message(call: ToolCall, result: ToolResult) -> TranscriptMessage:
    """The one legal ``tool`` transcript message answering ``call``.

    A failed result still answers the id — its error text (or the raw
    content when the result carries one) rides in the message content so
    the model can correct course.
    """
    if result.ok:
        content = result.content
    else:
        content = result.error or result.content or "tool call failed"
    return TranscriptMessage(
        role="tool",
        tool_call_id=call.id,
        name=result.name or call.name or None,
        content=content,
    )


def _merge_usage(
    accumulated: dict[str, int], fresh: dict[str, int] | None
) -> dict[str, int]:
    out = dict(accumulated)
    for key, value in (fresh or {}).items():
        if isinstance(value, int) and not isinstance(value, bool):
            out[key] = out.get(key, 0) + value
    return out


def _usage_tokens(usage: dict[str, int]) -> tuple[int, int]:
    return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
