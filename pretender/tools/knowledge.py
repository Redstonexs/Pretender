"""Phase 5/6 knowledge tools: deferred ``query_memory`` /
``query_person_profile`` / ``query_jargon``.

These tools are the ONLY retrieval surface for durable knowledge, and they
are deliberately deferred: they are emitted in the provider schema only
after ``tool_search`` activates them, and they are never invoked in the Gate
or replay. They are registered into the shared ``CoreToolRegistry`` without
changing any core-tool behavior.

The handlers are UNBOUND ``ToolContext`` methods (first parameter ``self``),
exactly like the core tools, so the foundation's ``@tool`` schema derivation
sees a clean parameter list. They receive INJECTED chat-scoped callbacks
(``KnowledgeCallbacks``) rather than any direct repository access: the caller
binds the callbacks to a specific ``chat_key`` at ``ToolContext``
construction, so a cross-chat request is impossible by construction. Results
are capped/truncated before serialization, and every failure mode (no
service, malformed args) fails closed with a clear error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pretender.errors import ToolError
from pretender.search import MemoryRecallHit
from pretender.types import PersonProfile, RecordHit, SenderId

__all__ = [
    "KNOWLEDGE_TOOL_NAMES",
    "MAX_MEMORY_CHARS",
    "MAX_MEMORY_HITS",
    "MAX_PROFILE_CHARS",
    "MAX_JARGON_CHARS",
    "MAX_JARGON_HITS",
    "KnowledgeCallbacks",
    "query_memory",
    "query_person_profile",
    "query_jargon",
]

#: The three knowledge tool names, in registration order.
KNOWLEDGE_TOOL_NAMES: tuple[str, ...] = (
    "query_memory",
    "query_person_profile",
    "query_jargon",
)

#: Hard cap on the number of memory hits returned (clamped, not rejected).
MAX_MEMORY_HITS = 5
#: Per-hit text truncation inside a serialized memory result.
MAX_MEMORY_CHARS = 300
#: Per-field truncation inside a serialized person profile result.
MAX_PROFILE_CHARS = 500
#: Hard cap on the number of jargon hits returned (clamped, not rejected).
MAX_JARGON_HITS = 3
#: Per-hit text truncation inside a serialized jargon result.
MAX_JARGON_CHARS = 300


@dataclass(frozen=True)
class KnowledgeCallbacks:
    """Injected chat-scoped callbacks the knowledge tools speak to.

    The caller binds these to a specific ``chat_key`` at ``ToolContext``
    construction, so the tools never hold a repository reference and a
    cross-chat request is impossible by construction. Either callback may be
    None — the corresponding tool then fails closed with a clear error.
    """

    query_memory: Callable[[str, int], Awaitable[list[MemoryRecallHit]]] | None = None
    query_person: Callable[[SenderId], Awaitable[PersonProfile | None]] | None = None
    query_jargon: Callable[[str, int], Awaitable[list[RecordHit]]] | None = None


async def query_memory(self: Any, query: str, limit: int = 5) -> str:
    """Search recalled memory for this chat (lexical-first)."""
    knowledge = getattr(self, "_knowledge", None)
    cb = knowledge.query_memory if knowledge is not None else None
    if cb is None:
        raise ToolError("memory service is not available")
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query must be a non-empty string")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ToolError("limit must be an integer")
    limit = max(1, min(limit, MAX_MEMORY_HITS))
    hits = await cb(query, limit)
    items: list[dict[str, Any]] = []
    for hit in hits[:MAX_MEMORY_HITS]:  # cap serialized results, never unbounded
        text = hit.text
        if len(text) > MAX_MEMORY_CHARS:
            text = text[:MAX_MEMORY_CHARS] + "…"
        items.append(
            {
                "memory_id": hit.memory_id,
                "text": text,
                "score": hit.score,
                "source": hit.source,
            }
        )
    return json.dumps(
        {"action": "query_memory", "count": len(items), "hits": items},
        ensure_ascii=False,
    )


async def query_person_profile(self: Any, platform_uid: str) -> str:
    """Look up a person's profile in this chat."""
    knowledge = getattr(self, "_knowledge", None)
    cb = knowledge.query_person if knowledge is not None else None
    if cb is None:
        raise ToolError("person service is not available")
    if not isinstance(platform_uid, str) or not platform_uid.strip():
        raise ToolError("platform_uid must be a non-empty string")
    profile = await cb(SenderId(platform_uid))
    if profile is None:
        return json.dumps(
            {"action": "query_person_profile", "found": False},
            ensure_ascii=False,
        )
    profile_text = profile.profile or ""
    impression = profile.impression or ""
    if len(profile_text) > MAX_PROFILE_CHARS:
        profile_text = profile_text[:MAX_PROFILE_CHARS] + "…"
    if len(impression) > MAX_PROFILE_CHARS:
        impression = impression[:MAX_PROFILE_CHARS] + "…"
    return json.dumps(
        {
            "action": "query_person_profile",
            "found": True,
            "platform_uid": platform_uid,
            "names": list(profile.names),
            "profile": profile_text,
            "impression": impression,
        },
        ensure_ascii=False,
    )


async def query_jargon(self: Any, query: str, limit: int = 3) -> str:
    """Look up jargon records for this chat scoped to ``query``.

    The callback is chat-bound (the caller binds it to a specific
    ``chat_key`` at ``ToolContext`` construction), so a cross-chat jargon
    lookup is impossible by construction. Results are capped/truncated
    before serialization; a missing service or malformed args fail closed.
    """
    knowledge = getattr(self, "_knowledge", None)
    cb = knowledge.query_jargon if knowledge is not None else None
    if cb is None:
        raise ToolError("jargon service is not available")
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query must be a non-empty string")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ToolError("limit must be an integer")
    limit = max(1, min(limit, MAX_JARGON_HITS))
    hits = await cb(query, limit)
    items: list[dict[str, Any]] = []
    for hit in hits[:MAX_JARGON_HITS]:  # cap serialized results, never unbounded
        text = hit.text
        if len(text) > MAX_JARGON_CHARS:
            text = text[:MAX_JARGON_CHARS] + "…"
        items.append(
            {
                "record_id": hit.record_id,
                "text": text,
                "score": hit.score,
            }
        )
    return json.dumps(
        {"action": "query_jargon", "count": len(items), "hits": items},
        ensure_ascii=False,
    )
