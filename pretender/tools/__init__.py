"""Tool contracts: the shared Phase 3 write surface plus the Phase 5/6
knowledge retrieval tools, the Phase 6 P6.5b media send tools, and the
Phase 6 P6.6b chat-control tools.

Only the base contracts and the knowledge/media/chat-control tool
handlers/callbacks live here — the core tools are registered by
``pretender.tools.core``.
"""

from pretender.tools.base import ToolRegistry, ToolSpec, tool
from pretender.tools.chatctl import (
    CHATCTL_TOOL_NAMES,
    ChatControlCallbacks,
    notify_chat,
    set_focus,
)
from pretender.tools.knowledge import (
    KNOWLEDGE_TOOL_NAMES,
    MAX_JARGON_CHARS,
    MAX_JARGON_HITS,
    MAX_MEMORY_CHARS,
    MAX_MEMORY_HITS,
    MAX_PROFILE_CHARS,
    KnowledgeCallbacks,
    query_jargon,
    query_memory,
    query_person_profile,
)
from pretender.tools.media import (
    MEDIA_TOOL_NAMES,
    MediaCallbacks,
    MediaReplyIntent,
    catalog_prompt,
    media_segment_for_intent,
    send_emoji,
    send_image,
)

__all__ = [
    "ToolRegistry",
    "ToolSpec",
    "tool",
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
    "MEDIA_TOOL_NAMES",
    "MediaCallbacks",
    "MediaReplyIntent",
    "catalog_prompt",
    "media_segment_for_intent",
    "send_emoji",
    "send_image",
    "CHATCTL_TOOL_NAMES",
    "ChatControlCallbacks",
    "set_focus",
    "notify_chat",
]
