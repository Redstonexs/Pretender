"""Phase 6 P6.6b chat-control tools: ``set_focus`` / ``notify_chat``.

These are deferred, STAGED terminal tools. They never touch an adapter, the
outbox, the repository, or any platform send path directly: the live
``ToolContext`` carries chat-bound ``ChatControlCallbacks`` (injected by the
cycle at construction), and a successful call only stages one typed
``ChatControlIntent`` on the context. The CycleRunner applies the staged
intents idempotently in the normal LIVE terminal flow AFTER settle/outbox/
marker, using the dispatch id + intent sequence as the idempotency identity.

Design rules:

  - ``set_focus`` targets a KNOWN chat on the SAME account (platform +
    self_id) as the current chat, with a duration in [30, 3600] seconds.
    Applying it transactionally keeps ONE focus per account (any other
    active focus on the same account is cleared).
  - ``notify_chat`` creates a bounded INTERNAL focus event (TTL <= 1h) on
    the target chat. The event only makes the target chat's gate evaluate
    as focused — delivery still traverses the target chat's normal
    gate/cycle/outbox flow. There is NO bypass platform send.
  - Nothing here writes anywhere: no adapter.send, no outbox rows, no
    repository writes, no platform calls. All of that happens post-terminal
    in the CycleRunner.
  - Dry-run/replay discard controls: the tools are never invoked there (the
    agent never runs in replay, and dry-run never applies staged controls).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pretender.errors import ToolError
from pretender.types import ChatControlIntent, ChatKey

__all__ = [
    "CHATCTL_TOOL_NAMES",
    "ChatControlCallbacks",
    "set_focus",
    "notify_chat",
]

#: The two chat-control tool names, in registration order.
CHATCTL_TOOL_NAMES: tuple[str, ...] = ("set_focus", "notify_chat")

#: Hard bounds on a staged focus duration (seconds).
FOCUS_MIN_S = 30
FOCUS_MAX_S = 3600
#: Hard bounds on a staged notify TTL (seconds).
NOTIFY_MIN_S = 1
NOTIFY_MAX_S = 3600
#: Hard cap on the notify payload length (chars).
NOTIFY_MAX_CHARS = 500


@dataclass(frozen=True)
class ChatControlCallbacks:
    """The chat-bound callbacks the chat-control tools speak to.

    The cycle binds these to the CURRENT chat at ``ToolContext``
    construction, so a cross-account target is impossible by construction.
    ``resolve_chat`` validates that a target chat key is a KNOWN chat on the
    SAME account (platform + self_id) as the current chat. None disables
    the tools (they fail closed).
    """

    resolve_chat: Callable[[str], Awaitable[bool]] | None = None


# ── tool handlers (unbound ToolContext methods; ``self`` is the live ctx) ────

async def set_focus(self: Any, chat_key: str, duration_s: int) -> str:
    """将另一个同账号会话设为专注（30–3600 秒，同账号同时只有一个专注会话）。

    目标会话必须是已知且与当前会话同账号的会话。专注只影响目标会话的门控评估，
    不会直接发送任何消息。与 reply/wait/no_action 不互斥——专注在最终决定后生效。
    """
    return await _stage_control(
        self, "set_focus", chat_key, duration_s=duration_s
    )


async def notify_chat(self: Any, chat_key: str, text: str, ttl_s: int = 3600) -> str:
    """向另一个同账号会话发送内部通知（TTL 不超过 1 小时）。

    通知创建一个有界的内部专注事件：目标会话的门控会按专注评估并正常回应。
    绝不绕过平台直接发送——通知只影响目标会话的门控，实际回复仍走目标会话的
    正常门控/周期/发件箱流程。
    """
    return await _stage_control(
        self, "notify_chat", chat_key, text=text, ttl_s=ttl_s
    )


async def _stage_control(
    ctx: Any,
    tool_name: str,
    chat_key: str,
    *,
    duration_s: int | None = None,
    text: str | None = None,
    ttl_s: int | None = None,
) -> str:
    """Validate and stage ONE typed ``ChatControlIntent``.

    Fail-closed gates, in order: callbacks present → target key shape →
    duration/TTL bounds → notify payload shape → the target resolves to a
    known same-account chat. Nothing is written anywhere — the intent is
    staged on the live context and applied by the CycleRunner after the
    terminal settlement.
    """
    callbacks = getattr(ctx, "_chat_controls_cb", None)
    if callbacks is None:
        raise ToolError("chat controls are not available")
    resolve = callbacks.resolve_chat
    if resolve is None:
        raise ToolError("chat controls are not available")
    if not isinstance(chat_key, str) or not chat_key.strip():
        raise ToolError("chat_key must be a non-empty string")
    if tool_name == "set_focus":
        if isinstance(duration_s, bool) or not isinstance(duration_s, int):
            raise ToolError("duration_s must be an integer")
        if not (FOCUS_MIN_S <= duration_s <= FOCUS_MAX_S):
            raise ToolError(
                f"duration_s must be in [{FOCUS_MIN_S}, {FOCUS_MAX_S}]"
            )
        ttl = duration_s
        payload: str | None = None
    else:
        if isinstance(ttl_s, bool) or not isinstance(ttl_s, int):
            raise ToolError("ttl_s must be an integer")
        if not (NOTIFY_MIN_S <= ttl_s <= NOTIFY_MAX_S):
            raise ToolError(f"ttl_s must be in [{NOTIFY_MIN_S}, {NOTIFY_MAX_S}]")
        if not isinstance(text, str) or not text.strip():
            raise ToolError("text must be a non-empty string")
        if len(text) > NOTIFY_MAX_CHARS:
            raise ToolError(f"text must not exceed {NOTIFY_MAX_CHARS} chars")
        ttl = ttl_s
        payload = text
    if not await resolve(chat_key):
        raise ToolError(
            f"unknown or cross-account target chat: {chat_key!r}"
        )
    intent = ChatControlIntent(
        kind="focus" if tool_name == "set_focus" else "notify",
        target_chat_key=ChatKey(chat_key),
        ttl_s=ttl,
        text=payload,
    )
    ctx._chat_controls.append(intent)
    return json.dumps(
        {
            "action": tool_name,
            "chat_key": chat_key,
            "ttl_s": ttl,
            "text": payload,
        },
        ensure_ascii=False,
    )