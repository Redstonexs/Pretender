"""Core tools and the local runtime callback context (Phase 3, PLAN.md §2).

This module implements the deterministic core tool surface the planner's
tool loop consumes, plus the local (in-process) typed runtime callback
context that tool handlers speak to. Design rules:

  - Handlers are UNBOUND ``ToolContext`` methods: the first parameter is
    ``self`` (the live per-round ``ToolContext`` the cycle injects), so the
    foundation's ``@tool`` schema derivation sees the clean parameter list —
    it skips ``self``/``cls`` exactly as ``tools/base.py`` documents.
  -     ``register_core_tools`` builds one ``CoreToolRegistry`` (a
    ``ToolRegistry`` subclass that tracks deferred-tool activation) with the
    eight tools in deterministic order: ``reply``, ``wait``, ``no_action``,
    ``tool_search`` (all visible) and the deferred, capability-gated
    ``fetch_history`` / ``view_forward_message`` / ``query_memory`` /
    ``query_person_profile``.
  - ``dispatch_call`` resolves one ``ToolCall`` against the registry and
    applies the deterministic gate order — unknown tool → visibility
    (hidden / un-activated deferred) → chat scope → adapter capability →
    JSON-Schema validation → execute. A handler exception NEVER escapes: it
    becomes an ``ok=False`` ``ToolResult`` (``ToolError`` keeps its message;
    any other exception is contained the same way).
  - Nothing here touches adapters, the network, the repository, or the
    transcript: all data handlers read (recent messages, forward map,
    adapter capabilities) is injected by the caller at ``ToolContext``
    construction, and no input structure is ever mutated.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections.abc import Mapping
from typing import Any

from pretender.errors import ToolError
from pretender.tools.base import ToolRegistry, ToolSpec, tool
from pretender.tools.chatctl import (
    CHATCTL_TOOL_NAMES,
    ChatControlCallbacks,
    notify_chat,
    set_focus,
)
from pretender.tools.knowledge import (
    KNOWLEDGE_TOOL_NAMES,
    KnowledgeCallbacks,
    query_jargon,
    query_memory,
    query_person_profile,
)
from pretender.tools.media import (
    MEDIA_TOOL_NAMES,
    MediaCallbacks,
    MediaReplyIntent,
    send_emoji,
    send_image,
)
from pretender.toolparse import validate_arguments
from pretender.types import ChatControlIntent, ChatKey, Message, ToolCall, ToolCallId, ToolResult

__all__ = [
    "CoreToolRegistry",
    "ToolContext",
    "dispatch_call",
    "register_core_tools",
    "core_tool_specs",
    "CORE_TOOL_NAMES",
]

#: The core tool names, in registration order (the six Phase 3 core tools,
#: the two deferred Phase 5 knowledge tools, the Phase 6 P6.4b jargon tool,
#: the two Phase 6 P6.5b media send tools, and the two Phase 6 P6.6b
#: chat-control tools).
#: MaiBot's ``reply`` tool length directives. The planner picks one; the
#: replyer turns it into an instruction (see ``replyer.LENGTH_DIRECTIVES``).
REPLY_LENGTH_STYLES: tuple[str, ...] = ("简短表达", "正常回复", "长回复")

CORE_TOOL_NAMES: tuple[str, ...] = (
    "reply",
    "wait",
    "no_action",
    "tool_search",
    "fetch_history",
    "view_forward_message",
    *KNOWLEDGE_TOOL_NAMES,
    *MEDIA_TOOL_NAMES,
    *CHATCTL_TOOL_NAMES,
)

#: The adapter capability each core tool requires at dispatch time. A tool
#: whose required capability is absent from the context's ``capabilities``
#: fails safely (``ok=False``) rather than attempting any platform call.
_ADAPTER_CAPABILITY: dict[str, str] = {
    "fetch_history": "history",
    "view_forward_message": "forward",
    "send_emoji": "sticker",
    "send_image": "image",
}


class CoreToolRegistry(ToolRegistry):
    """A ``ToolRegistry`` with deterministic deferred-tool activation.

    ``tool_search`` flips a registered deferred tool to "activated":
    activated tools become dispatchable and are emitted by
    ``provider_definitions`` thereafter (deferred tools stay out until then,
    hidden tools never). All base behavior — shape validation, duplicate
    rejection, ``replace=True`` shadowing, capability bookkeeping,
    deterministic order — is inherited unchanged.
    """

    def __init__(self, name: str = "core_tools") -> None:
        super().__init__(name)
        self._activated: set[str] = set()

    def activate(self, name: str) -> bool:
        """Activate a registered deferred tool; returns True when the call
        changed anything. Visible/hidden/unknown tools are ignored."""
        spec = self.get(name)
        if spec is None or spec.visibility != "deferred" or name in self._activated:
            return False
        self._activated.add(name)
        return True

    def is_activated(self, name: str) -> bool:
        return name in self._activated

    def activated_names(self) -> tuple[str, ...]:
        return tuple(n for n in self.names() if n in self._activated)

    def provider_definitions(self, *, scope: str = "all") -> list[dict[str, Any]]:
        """Provider definitions for every visible tool PLUS activated
        deferred tools, in registration order; ``scope`` filters by chat
        scope exactly like the base implementation. Hidden tools and
        un-activated deferred tools are never emitted."""
        out: list[dict[str, Any]] = []
        for spec in self.all():
            if spec.visibility == "hidden":
                continue
            if spec.visibility == "deferred" and not self.is_activated(spec.name):
                continue
            if scope != "all" and spec.chat_scope not in ("all", scope):
                continue
            out.append(spec.provider_definition())
        return out


class ToolContext:
    """Local (in-process) typed runtime callback surface for one tool round.

    The cycle builds one ``ToolContext`` per planner tool round and hands it
    to every dispatched handler. It owns ALL mutable runtime state for the
    round and exposes the six core callbacks:

      - ``reply`` — stages the bot's visible reply text (the replyer's
        input); the LAST call wins.
      - ``wait`` — pauses N seconds; the wait is deliberately not
        interrupted by new messages (PLAN.md §1.B state machine).
      - ``no_action`` — explicit "do nothing" verdict.
      - ``tool_search`` — activates deferred tools in the registry by
        matching name/description/capability keywords; returns the matched
        tool names.
      - ``fetch_history`` — reads the LOCAL recent-message snapshot, gated
        by the adapter ``history`` capability; returns a bounded, rendered
        list.
      - ``view_forward_message`` — reads a forwarded message's rendered
        contents from the LOCAL forward map, gated by the adapter
        ``forward`` capability; unknown ids fail safely.

    ``reply``/``wait``/``no_action`` are mutually exclusive with last-wins
    semantics: staging one clears the others, so the round always ends with
    exactly one explicit verdict. No adapter/network calls and no repository
    access: everything handlers can read is injected at construction.
    """

    #: Hard cap on a single ``wait`` (seconds) — a runaway wait would stall
    #: the scheduler.
    MAX_WAIT_S: float = 3600.0
    #: Hard cap on ``fetch_history``'s ``limit`` (clamped, not rejected).
    MAX_HISTORY_LIMIT: int = 50
    #: Per-message text truncation inside a rendered history line.
    MAX_MESSAGE_CHARS: int = 200

    def __init__(
        self,
        *,
        chat_key: ChatKey,
        chat_kind: str,
        capabilities: frozenset[str] = frozenset(),
        registry: ToolRegistry | None = None,
        recent: tuple[Message, ...] = (),
        forwards: Mapping[str, str] | None = None,
        self_name: str | None = None,
        knowledge: KnowledgeCallbacks | None = None,
        media: MediaCallbacks | None = None,
        chat_controls: ChatControlCallbacks | None = None,
    ) -> None:
        if chat_kind not in ("group", "private"):
            raise ValueError(
                f"chat_kind must be 'group' or 'private', got {chat_kind!r}"
            )
        self.chat_key = chat_key
        self.chat_kind = chat_kind
        self.capabilities = frozenset(capabilities)
        self.registry = registry
        self.recent = tuple(recent)
        self.forwards = dict(forwards or {})
        self.self_name = self_name
        # Injected chat-scoped knowledge callbacks (Phase 5): the deferred
        # query_memory / query_person_profile tools speak to these, never to
        # a repository directly. None disables the knowledge tools (they fail
        # closed with a clear error).
        self._knowledge = knowledge
        # Injected chat-scoped media catalog callbacks (Phase 6 P6.5b): the
        # deferred send_emoji / send_image tools speak to these, never to a
        # repository directly. None disables the media tools (they fail
        # closed with a clear error).
        self._media = media
        # Injected chat-bound chat-control callbacks (Phase 6 P6.6b): the
        # deferred set_focus / notify_chat tools speak to these, never to a
        # repository/adapter directly. None disables the chat-control tools
        # (they fail closed with a clear error).
        self._chat_controls_cb = chat_controls
        # staged outcome (mutually exclusive, last-wins)
        self._reply_text: str | None = None
        self._reply_to: str | None = None
        self._wait_seconds: float | None = None
        self._no_action: bool = False
        # staged media send (mutually exclusive with the text verdicts: the
        # FIRST terminal intent wins and a conflict is a ToolError)
        self._media_intent: MediaReplyIntent | None = None
        # staged chat controls (Phase 6 P6.6b): an ordered list of typed
        # intents the CycleRunner applies idempotently after the terminal
        # settlement. NOT mutually exclusive with the text verdicts — a
        # control rides along with the round's terminal intent.
        self._chat_controls: list[ChatControlIntent] = []
        # The planner's chosen reply length (MaiBot's ``reply_style`` tool
        # argument); "" means no preference.
        self._reply_length_style: str = ""

    # ── read-only outcome accessors ─────────────────────────────────────────

    @property
    def reply_text(self) -> str | None:
        return self._reply_text

    @property
    def reply_to(self) -> str | None:
        return self._reply_to

    @property
    def reply_length_style(self) -> str:
        """The planner's chosen reply length, or ``""`` for the default."""
        return self._reply_length_style

    @property
    def wait_seconds(self) -> float | None:
        return self._wait_seconds

    @property
    def no_action_verdict(self) -> bool:
        return self._no_action

    @property
    def media_intent(self) -> MediaReplyIntent | None:
        """The staged media send (``send_emoji`` / ``send_image``), or None.
        Mutually exclusive with the text verdicts: the first terminal intent
        wins and a conflicting call is a ``ToolError``."""
        return self._media_intent

    @property
    def chat_controls(self) -> tuple[ChatControlIntent, ...]:
        """The staged chat controls (``set_focus`` / ``notify_chat``), in
        staging order. NOT mutually exclusive with the text verdicts — the
        CycleRunner applies them idempotently after the terminal settlement
        (LIVE only)."""
        return tuple(self._chat_controls)

    # ── core tool handlers (the ``self`` param is the live context) ─────────

    def reply(
        self,
        text: str,
        reply_to: str | None = None,
        reply_style: str | None = None,
    ) -> str:
        """Stage the bot's visible reply text (the replyer's input).

        ``reply_style`` is how long the reply should be — ``简短表达``,
        ``正常回复`` or ``长回复``. The planner has read the conversation and
        knows whether this moment wants a two-character reaction or a real
        explanation, so it decides; the replyer is told rather than inferring
        length from the reference text. Anything else is ignored.

        The LAST ``reply`` call wins and clears any staged ``wait`` or
        ``no_action``. A staged media send (``send_emoji``/``send_image``)
        is a conflict — the first terminal intent wins. Returns a structured
        JSON result.
        """
        if not isinstance(text, str) or not text.strip():
            raise ToolError("reply text must be a non-empty string")
        if self._media_intent is not None:
            raise ToolError(
                "reply conflicts with a staged media send; the first"
                " terminal intent wins"
            )
        self._reply_text = text
        self._reply_to = reply_to
        self._reply_length_style = (
            reply_style if reply_style in REPLY_LENGTH_STYLES else ""
        )
        self._wait_seconds = None
        self._no_action = False
        return json.dumps(
            {"action": "reply", "text": text, "reply_to": reply_to},
            ensure_ascii=False,
        )

    def wait(self, seconds: float) -> str:
        """Pause ``seconds`` before the next planner call (deliberately not
        interrupted by new messages). Clears any staged reply or
        ``no_action``. A staged media send is a conflict — the first
        terminal intent wins. Returns a structured JSON result.
        """
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise ToolError("wait seconds must be a number")
        if not math.isfinite(seconds) or seconds <= 0:
            raise ToolError("wait seconds must be a finite positive number")
        if seconds > self.MAX_WAIT_S:
            raise ToolError(f"wait seconds must not exceed {self.MAX_WAIT_S:g}")
        if self._media_intent is not None:
            raise ToolError(
                "wait conflicts with a staged media send; the first"
                " terminal intent wins"
            )
        self._wait_seconds = float(seconds)
        self._reply_text = None
        self._reply_to = None
        self._no_action = False
        return json.dumps(
            {"action": "wait", "seconds": float(seconds)}, ensure_ascii=False
        )

    def no_action(self, reason: str | None = None) -> str:
        """Explicitly take no action this turn. Clears any staged reply or
        wait (the explicit verdict wins). A staged media send is a conflict
        — the first terminal intent wins. Returns a structured JSON result.
        """
        if reason is not None and not isinstance(reason, str):
            raise ToolError("no_action reason must be a string or null")
        if self._media_intent is not None:
            raise ToolError(
                "no_action conflicts with a staged media send; the first"
                " terminal intent wins"
            )
        self._no_action = True
        self._reply_text = None
        self._reply_to = None
        self._wait_seconds = None
        return json.dumps(
            {"action": "no_action", "reason": reason}, ensure_ascii=False
        )

    def tool_search(self, query: str = "", capability: str | None = None) -> str:
        """Search the registry for DEFERRED tools and activate the matches.

        A deferred tool matches when its name or description contains
        ``query`` (case-insensitive) and, when ``capability`` is given, its
        ``capability`` tag equals it. An empty ``query`` matches every
        deferred tool in the (capability-filtered) set. Activated tools
        become dispatchable and are emitted in the provider schema for the
        next planner round. Returns a structured JSON result listing the
        matched tool names.
        """
        if query is not None and not isinstance(query, str):
            raise ToolError("tool_search query must be a string")
        if capability is not None and not isinstance(capability, str):
            raise ToolError("tool_search capability must be a string or null")
        needle = (query or "").strip().lower()
        matched: list[str] = []
        if self.registry is not None:
            for spec in self.registry.all():
                if spec.visibility != "deferred":
                    continue
                if capability is not None and spec.capability != capability:
                    continue
                haystack = f"{spec.name} {spec.description}".lower()
                if needle and needle not in haystack:
                    continue
                matched.append(spec.name)
                activator = getattr(self.registry, "activate", None)
                if activator is not None:
                    activator(spec.name)
        return json.dumps({"tools": matched}, ensure_ascii=False)

    def fetch_history(self, limit: int = 20) -> str:
        """Read the recent chat history from the LOCAL snapshot.

        Gated by the adapter ``history`` capability: without it the call
        fails safely with a clear error. ``limit`` is clamped to
        ``[1, MAX_HISTORY_LIMIT]``; the newest messages are rendered as
        ``sender_name: text`` lines (self messages use the bot's name when
        known). Returns a structured JSON result.
        """
        if "history" not in self.capabilities:
            raise ToolError("adapter does not support the 'history' capability")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ToolError("fetch_history limit must be an integer")
        limit = max(1, min(limit, self.MAX_HISTORY_LIMIT))
        lines: list[str] = []
        for msg in self.recent[:limit]:
            name = (
                self.self_name
                if msg.is_self and self.self_name is not None
                else msg.sender_name
            )
            text = msg.text
            if len(text) > self.MAX_MESSAGE_CHARS:
                text = text[: self.MAX_MESSAGE_CHARS] + "…"
            lines.append(f"{name}: {text}")
        return json.dumps(
            {"action": "fetch_history", "count": len(lines), "messages": lines},
            ensure_ascii=False,
        )

    def view_forward_message(self, id: str) -> str:
        """View a forwarded message's rendered contents from the LOCAL map.

        Gated by the adapter ``forward`` capability. An unknown id fails
        safely with a clear error. Returns a structured JSON result.
        """
        if "forward" not in self.capabilities:
            raise ToolError("adapter does not support the 'forward' capability")
        if not isinstance(id, str) or not id.strip():
            raise ToolError("forward id must be a non-empty string")
        if id not in self.forwards:
            raise ToolError(f"unknown forwarded message: {id}")
        return json.dumps(
            {
                "action": "view_forward_message",
                "id": id,
                "content": self.forwards[id],
            },
            ensure_ascii=False,
        )


# ── core tool specs (built once; frozen and stateless) ──────────────────────

_CORE_SPECS: tuple[ToolSpec, ...] = (
    tool("reply")(ToolContext.reply),
    tool("wait")(ToolContext.wait),
    tool("no_action")(ToolContext.no_action),
    tool("tool_search")(ToolContext.tool_search),
    tool("fetch_history", visibility="deferred", capability="history")(
        ToolContext.fetch_history
    ),
    tool(
        "view_forward_message",
        visibility="deferred",
        capability="forward",
        chat_scope="group",
    )(ToolContext.view_forward_message),
    # Phase 5 knowledge tools: deferred, capability-gated retrieval. They are
    # emitted only after tool_search activates them and are never invoked in
    # the Gate or replay. Their handlers read the injected chat-scoped
    # KnowledgeCallbacks on the live ToolContext.
    tool("query_memory", visibility="deferred", capability="memory")(
        query_memory
    ),
    tool("query_person_profile", visibility="deferred", capability="memory")(
        query_person_profile
    ),
    # Phase 6 P6.4b jargon tool: deferred, chat-bound, activated through
    # tool_search exactly like the knowledge tools. The handler reads the
    # injected chat-scoped jargon callback — no repository/adapter exposure,
    # no cross-chat lookup.
    tool("query_jargon", visibility="deferred", capability="memory")(
        query_jargon
    ),
    # Phase 6 P6.5b media send tools: deferred, capability-gated staged
    # terminal tools. They accept an OPAQUE approved catalog asset id only,
    # stage one typed MediaReplyIntent on the live ToolContext (never an
    # adapter/outbox/usage write), and are mutually exclusive with
    # reply/wait/no_action. The handlers read the injected chat-scoped
    # MediaCallbacks — no repository/adapter exposure, no cross-chat lookup.
    tool("send_emoji", visibility="deferred", capability="sticker")(
        send_emoji
    ),
    tool("send_image", visibility="deferred", capability="image")(
        send_image
    ),
    # Phase 6 P6.6b chat-control tools: deferred, chat-bound staged tools.
    # They validate the target through the injected chat-bound
    # ChatControlCallbacks (never a repository/adapter reference) and stage
    # one typed ChatControlIntent on the live ToolContext — nothing is
    # written at staging time. The CycleRunner applies the intents
    # idempotently in the normal LIVE terminal flow after settle/outbox/
    # marker. No adapter capability is required (internal focus events only).
    tool("set_focus", visibility="deferred", capability="chat_control")(
        set_focus
    ),
    tool("notify_chat", visibility="deferred", capability="chat_control")(
        notify_chat
    ),
)


def core_tool_specs() -> tuple[ToolSpec, ...]:
    """The built-in core tool specs, in deterministic registration order.

    Shared by ``register_core_tools`` (the live registry) and the Phase 6
    P6.6 plugin loader (the staging registry seed), so the built-ins and a
    third-party tool register through the exact same specs.
    """
    return _CORE_SPECS


def register_core_tools(registry: CoreToolRegistry | None = None) -> CoreToolRegistry:
    """Register the six core tools into a (new) ``CoreToolRegistry`` in
    deterministic order and return it. Registering twice into the same
    registry raises ``RegistryError`` (the base's duplicate rejection)."""
    reg = registry if registry is not None else CoreToolRegistry()
    for spec in _CORE_SPECS:
        reg.register(spec)
    return reg


# ── dispatch ────────────────────────────────────────────────────────────────

async def dispatch_call(
    call: ToolCall | Mapping[str, Any],
    ctx: ToolContext,
    registry: ToolRegistry | None = None,
    *,
    rate_limiter: Any = None,
    clock: Any = None,
) -> ToolResult:
    """Resolve and execute one tool call against ``registry`` (defaults to
    ``ctx.registry``), returning exactly one ``ToolResult``.

    Deterministic gate order:

      1. unknown tool → ``ok=False``;
      2. visibility — hidden tools never dispatch; deferred tools dispatch
         only after ``tool_search`` activation;
      3. chat scope — a tool whose ``chat_scope`` excludes the chat's kind
         (group/private) is rejected;
      4. adapter capability — ``fetch_history`` requires ``history``,
         ``view_forward_message`` requires ``forward``;
      5. JSON-Schema validation of ``arguments`` against the spec;
      6. rate limit — a tool whose per-minute ``rate_limit`` is exhausted
         fails safely (``ok=False``) without executing;
      7. execute the handler with the live context; a ``timeout_s`` bounds
         an async handler via ``asyncio.wait_for`` (a timed-out handler
         fails safely, never hangs); exceptions NEVER escape — every one
         becomes an ``ok=False`` result (``ToolError`` keeps its message,
         any other exception is contained the same way).

    ``rate_limiter`` defaults to the registry's shared limiter; ``clock``
    (a ``() -> float`` callable) may be injected for deterministic rate
    tests. No input structure is mutated and nothing is written anywhere.
    """
    call = _normalize_call(call)
    reg = registry if registry is not None else ctx.registry
    spec = reg.get(call.name) if reg is not None else None
    if spec is None:
        return _fail(call, call.name, f"unknown tool: {call.name}")

    # visibility gate
    if spec.visibility == "hidden":
        return _fail(
            call, spec.name, f"tool {spec.name!r} is hidden and cannot be called"
        )
    if spec.visibility == "deferred":
        is_activated = getattr(reg, "is_activated", None)
        if is_activated is None or not is_activated(spec.name):
            return _fail(
                call,
                spec.name,
                f"tool {spec.name!r} is deferred; use tool_search to activate it",
            )

    # chat scope gate
    effective = "group" if ctx.chat_kind == "group" else "private"
    if spec.chat_scope != "all" and spec.chat_scope != effective:
        return _fail(
            call,
            spec.name,
            f"tool {spec.name!r} is not available in {ctx.chat_kind} chats",
        )

    # adapter capability gate
    required = _ADAPTER_CAPABILITY.get(spec.name)
    if required is not None and required not in ctx.capabilities:
        return _fail(
            call,
            spec.name,
            f"tool {spec.name!r} requires adapter capability {required!r}",
        )

    # JSON-Schema validation
    errors = validate_arguments(call.arguments, spec.parameters)
    if errors:
        return _fail(call, spec.name, "schema mismatch: " + "; ".join(errors))

    # rate limit gate (deterministic per-tool accounting)
    limiter = rate_limiter if rate_limiter is not None else getattr(
        reg, "rate_limiter", None
    )
    if limiter is not None and spec.rate_limit is not None:
        now = clock() if clock is not None else None
        if not limiter.allow(spec.name, spec.rate_limit, now=now):
            return _fail(
                call,
                spec.name,
                f"tool {spec.name!r} rate limit exceeded "
                f"({spec.rate_limit}/min)",
            )

    # execute (sync or async handler), bounding async handlers by timeout_s
    # and containing every exception
    try:
        args = dict(call.arguments)
        result = spec.handler(ctx, **args)
        if inspect.isawaitable(result):
            if spec.timeout_s is not None:
                result = await asyncio.wait_for(result, timeout=spec.timeout_s)
            else:
                result = await result
    except asyncio.TimeoutError:
        return _fail(
            call,
            spec.name,
            f"tool {spec.name!r} timed out after {spec.timeout_s:g}s",
        )
    except ToolError as exc:
        return _fail(call, spec.name, str(exc))
    except Exception as exc:  # containment boundary — never propagate
        return _fail(call, spec.name, f"{type(exc).__name__}: {exc}")
    return ToolResult(call_id=call.id, name=spec.name, ok=True, content=str(result))


def _normalize_call(call: ToolCall | Mapping[str, Any]) -> ToolCall:
    if isinstance(call, ToolCall):
        return call
    if not isinstance(call, Mapping):
        raise TypeError("call must be a ToolCall or a mapping")
    return ToolCall(
        id=ToolCallId(str(call.get("id", "")) or "?"),
        name=str(call.get("name", "")),
        arguments=dict(call.get("arguments") or {}),
    )


def _fail(call: ToolCall, name: str, error: str) -> ToolResult:
    return ToolResult(call_id=call.id, name=name, ok=False, error=error)
