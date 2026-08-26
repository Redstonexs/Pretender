"""Shared tool contracts: ``ToolSpec``, the ``@tool`` decorator, and
``ToolRegistry``.

This is the Phase 3 shared write surface (PLAN.md Phase 3 step 1). It
defines how a tool is declared and how its provider-facing definition is
derived — it does NOT implement any core tool. Core tools land in a later
step and register here.

Design rules:
  - ``ToolSpec`` is a frozen dataclass capturing the tool's name,
    description, a CONSERVATIVE JSON Schema derived from the typed handler
    signature, and its metadata (visibility / chat_scope / capability /
    timeout / rate-limit). It fails closed on construction: a bad name,
    visibility, scope, handler shape or numeric metadata is a ``ToolError``.
  - ``@tool`` builds a ``ToolSpec`` from a typed function. The schema is
    conservative: well-understood types (str/int/float/bool, Optional,
    list/tuple, dict, Literal, NewType) map to precise JSON Schema; anything
    unrecognized degrades to a permissive ``{}`` rather than guessing wrong.
  - ``ToolRegistry`` reuses the generic ``Registry`` (deterministic
    insertion order, ``replace=True`` shadowing, duplicate rejection) and
    adds tool-specific behavior: capability tracking and provider tool
    definitions.
"""

from __future__ import annotations

import inspect
import re
import time
import typing
from dataclasses import dataclass
from typing import Any, Callable, get_args, get_origin

from pretender.errors import RegistryError, ToolError
from pretender.registry import Registry

# A tool name is a conservative identifier: letters/digits/underscore/hyphen,
# not starting with a digit. This keeps provider tool names unambiguous.
_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")

_VISIBILITIES = frozenset({"visible", "deferred", "hidden"})
_CHAT_SCOPES = frozenset({"all", "group", "private"})

__all__ = ["ToolSpec", "ToolRegistry", "ToolRateLimiter", "tool"]


class ToolRateLimiter:
    """Deterministic per-tool rate accounting (a rolling per-minute cap).

    ``allow(name, rate_limit, now)`` returns True when a call may proceed
    (and records it) and False when the per-minute cap is exhausted. The
    window is a rolling 60-second window; ``clock`` (a ``() -> float``
    callable) may be injected for deterministic tests.
    """

    _WINDOW_S = 60.0

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock if clock is not None else time.monotonic
        self._calls: dict[str, list[float]] = {}

    def allow(
        self, name: str, rate_limit: int | None, now: float | None = None
    ) -> bool:
        if rate_limit is None:
            return True
        now = now if now is not None else self._clock()
        window_start = now - self._WINDOW_S
        calls = [t for t in self._calls.get(name, []) if t > window_start]
        if len(calls) >= rate_limit:
            self._calls[name] = calls
            return False
        calls.append(now)
        self._calls[name] = calls
        return True


@dataclass(frozen=True)
class ToolSpec:
    """One tool's full declaration.

    ``handler`` is the callable that executes the tool; ``parameters`` is a
    JSON Schema object (the ``parameters`` field of an OpenAI function
    definition). ``visibility`` gates whether the tool is emitted in the
    provider schema at all (``deferred`` tools stay out until tool_search
    activates them); ``chat_scope`` restricts which chats see it.
    ``capability`` is a free-form tag for grouping (e.g. ``"memory"``);
    ``timeout_s`` bounds a single execution; ``rate_limit`` is a per-minute
    call cap (None = unlimited).
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    visibility: str = "visible"
    chat_scope: str = "all"
    capability: str | None = None
    timeout_s: float | None = None
    rate_limit: int | None = None
    is_async: bool = False

    def __post_init__(self) -> None:
        if not _TOOL_NAME_RE.fullmatch(self.name):
            raise ToolError(f"invalid tool name: {self.name!r}")
        if self.visibility not in _VISIBILITIES:
            raise ToolError(
                f"tool {self.name!r}: visibility must be one of "
                f"{sorted(_VISIBILITIES)}, got {self.visibility!r}"
            )
        if self.chat_scope not in _CHAT_SCOPES:
            raise ToolError(
                f"tool {self.name!r}: chat_scope must be one of "
                f"{sorted(_CHAT_SCOPES)}, got {self.chat_scope!r}"
            )
        if not callable(self.handler):
            raise ToolError(f"tool {self.name!r}: handler must be callable")
        if not isinstance(self.parameters, dict):
            raise ToolError(
                f"tool {self.name!r}: parameters must be a JSON Schema dict"
            )
        if self.timeout_s is not None and (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or self.timeout_s <= 0
        ):
            raise ToolError(
                f"tool {self.name!r}: timeout_s must be a positive number"
            )
        if self.rate_limit is not None and (
            isinstance(self.rate_limit, bool)
            or not isinstance(self.rate_limit, int)
            or self.rate_limit <= 0
        ):
            raise ToolError(
                f"tool {self.name!r}: rate_limit must be a positive integer"
            )

    def provider_definition(self) -> dict[str, Any]:
        """The OpenAI-compatible tool definition for this spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry(Registry[ToolSpec]):
    """A ``Registry`` of ``ToolSpec``s with capability tracking and provider
    tool-definition export.

    Reuses the generic ``Registry`` for deterministic ordering, duplicate
    rejection and ``replace=True`` shadowing; adds tool-specific validation
    (only ``ToolSpec`` items, callable handlers) and capability bookkeeping.
    """

    def __init__(self, name: str = "tools") -> None:
        super().__init__(name)
        self._capabilities: dict[str, list[str]] = {}
        # Shared per-tool rate accounting for every dispatch through this
        # registry (rate limits are global per tool, not per chat).
        self.rate_limiter = ToolRateLimiter()

    def register(
        self,
        item: ToolSpec | None = None,
        *,
        replace: bool = False,
        name: str | None = None,
    ) -> ToolSpec | Callable[[ToolSpec], ToolSpec]:
        if item is None:
            def decorator(x: ToolSpec) -> ToolSpec:
                self.register(x, replace=replace, name=name)
                return x

            return decorator
        if not isinstance(item, ToolSpec):
            raise RegistryError(
                f"{self.name}: only ToolSpec items can be registered, got "
                f"{type(item).__name__}"
            )
        key = name or item.name
        if key in self._items and not replace:
            raise RegistryError(
                f"{self.name}: {key!r} is already registered (use replace=True to shadow)"
            )
        if key in self._items:
            old = self._items[key]
            if old.capability and old.capability in self._capabilities:
                bucket = self._capabilities[old.capability]
                if key in bucket:
                    bucket.remove(key)
        if item.capability:
            self._capabilities.setdefault(item.capability, [])
            if key not in self._capabilities[item.capability]:
                self._capabilities[item.capability].append(key)
        return super().register(item, replace=replace, name=name)

    def unregister(self, name: str) -> None:
        item = self._items.get(name)
        if item is not None and item.capability:
            bucket = self._capabilities.get(item.capability)
            if bucket and name in bucket:
                bucket.remove(name)
        super().unregister(name)

    def tool(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        visibility: str = "visible",
        chat_scope: str = "all",
        capability: str | None = None,
        timeout_s: float | None = None,
        rate_limit: int | None = None,
    ) -> Callable[[Callable[..., Any]], ToolSpec]:
        """Decorator that builds a ``ToolSpec`` from a typed function and
        registers it immediately."""
        def decorator(fn: Callable[..., Any]) -> ToolSpec:
            spec = _build_spec(
                fn,
                name=name,
                description=description,
                visibility=visibility,
                chat_scope=chat_scope,
                capability=capability,
                timeout_s=timeout_s,
                rate_limit=rate_limit,
            )
            self.register(spec)
            return spec

        return decorator

    def provider_definitions(self, *, scope: str = "all") -> list[dict[str, Any]]:
        """Provider tool definitions for every VISIBLE tool, in registration
        order. ``scope`` filters by chat scope: ``"all"`` returns every
        visible tool; ``"group"``/``"private"`` return tools whose
        ``chat_scope`` is ``"all"`` or the given scope. Deferred/hidden tools
        are never emitted."""
        out: list[dict[str, Any]] = []
        for spec in self.all():
            if spec.visibility != "visible":
                continue
            if scope != "all" and spec.chat_scope not in ("all", scope):
                continue
            out.append(spec.provider_definition())
        return out

    def with_capability(self, capability: str) -> tuple[ToolSpec, ...]:
        """All tools tagged with ``capability``, in registration order."""
        return tuple(self._items[n] for n in self._capabilities.get(capability, []))


def tool(
    name: str | None = None,
    *,
    description: str | None = None,
    visibility: str = "visible",
    chat_scope: str = "all",
    capability: str | None = None,
    timeout_s: float | None = None,
    rate_limit: int | None = None,
) -> Callable[[Callable[..., Any]], ToolSpec]:
    """Build a ``ToolSpec`` from a typed function (does not register).

    Usage::

        @tool("query_memory", capability="memory")
        def query_memory(q: str, top_k: int = 5) -> str:
            \"\"\"Search recalled memory.\"\"\"
            ...

    The JSON Schema is derived conservatively from the signature: required
    parameters are those without defaults; ``Optional``/``None`` defaults
    make a parameter nullable and optional; unknown types degrade to a
    permissive ``{}``.
    """
    def decorator(fn: Callable[..., Any]) -> ToolSpec:
        return _build_spec(
            fn,
            name=name,
            description=description,
            visibility=visibility,
            chat_scope=chat_scope,
            capability=capability,
            timeout_s=timeout_s,
            rate_limit=rate_limit,
        )

    return decorator


# ── spec construction ───────────────────────────────────────────────────────

def _build_spec(
    fn: Callable[..., Any],
    *,
    name: str | None,
    description: str | None,
    visibility: str,
    chat_scope: str,
    capability: str | None,
    timeout_s: float | None,
    rate_limit: int | None,
) -> ToolSpec:
    return ToolSpec(
        name=name or fn.__name__,
        description=(
            description
            if description is not None
            else (inspect.getdoc(fn) or "").strip()
        ),
        parameters=_schema_from_signature(fn),
        handler=fn,
        visibility=visibility,
        chat_scope=chat_scope,
        capability=capability,
        timeout_s=timeout_s,
        rate_limit=rate_limit,
        is_async=inspect.iscoroutinefunction(fn),
    )


# ── conservative JSON Schema derivation ─────────────────────────────────────

def _schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if pname in ("self", "cls"):
            continue
        hint = hints.get(pname, Any)
        prop = _type_to_schema(hint)
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        elif param.default is None:
            prop = _with_null(prop)
        properties[pname] = prop
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _type_to_schema(t: Any) -> dict[str, Any]:
    if t is Any or t is inspect.Parameter.empty:
        return {}
    origin = get_origin(t)
    if origin is typing.Union:
        args = [a for a in get_args(t) if a is not type(None)]
        if len(args) == 1:
            return _with_null(_type_to_schema(args[0]))
        return {}  # multi-type union: permissive
    if t is str:
        return {"type": "string"}
    if t is int:
        return {"type": "integer"}
    if t is float:
        return {"type": "number"}
    if t is bool:
        return {"type": "boolean"}
    if origin in (list, tuple):
        (item, *_) = get_args(t)
        return {"type": "array", "items": _type_to_schema(item)}
    if origin is dict:
        (_, vtype) = get_args(t)
        return {"type": "object", "additionalProperties": _type_to_schema(vtype)}
    if origin is typing.Literal:
        return {"enum": list(get_args(t))}
    supertype = getattr(t, "__supertype__", None)
    if supertype is not None:
        return _type_to_schema(supertype)
    return {}


def _with_null(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a schema accept ``null`` (for Optional / None-default params)."""
    if not schema:
        return schema
    t = schema.get("type")
    if isinstance(t, str):
        return {**schema, "type": [t, "null"]}
    return schema
