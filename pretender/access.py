"""Where the bot is allowed to speak: the group / private-chat access lists.

This is prior to the gate. ``[gate]`` decides whether THIS moment is worth
replying to; access decides whether the bot is allowed to reply in this chat
at all. A chat the lists exclude is never replied to — not even when the bot
is directly @-ed, because "stop talking in that group" that a direct mention
can override is not a control anyone would trust.

Excluded chats are still read, stored and learned from. The bot keeps
watching the room and keeps picking up how it talks; it just never says
anything there. Muting a chat therefore costs nothing to undo: unlist it and
the bot already knows what has been going on.

Everything here is PURE — no I/O, no clock, no config loading. The predicate
is evaluated per gate cycle and per outbox pump.
"""

from __future__ import annotations

from pretender.config import AccessConfig, AccessListConfig
from pretender.types import ChatKey

__all__ = ["GROUP", "PRIVATE", "chat_id", "chat_kind", "is_muted"]

#: The two chat kinds the lists address, as they appear in a chat key
#: (``qq:group:123456`` / ``qq:private:10001``).
GROUP = "group"
PRIVATE = "private"


def chat_kind(chat_key: ChatKey | str) -> str | None:
    """``"group"``, ``"private"``, or None for a key that is neither.

    None is the console and any future adapter whose keys are not
    group/private shaped. Those are deliberately OUT OF SCOPE for these
    lists: a group whitelist is about QQ groups, and it must not silently
    take the local console down with it.
    """
    parts = str(chat_key).split(":")
    if len(parts) >= 3 and parts[1] in (GROUP, PRIVATE):
        return parts[1]
    return None


def chat_id(chat_key: ChatKey | str) -> str:
    """The bare platform id from a chat key (``qq:group:123456`` -> ``123456``)."""
    parts = str(chat_key).split(":", 2)
    return parts[2] if len(parts) >= 3 else ""


def _listed(access_list: AccessListConfig, chat_key: str, ident: str) -> bool:
    """True when this chat appears in the list, by bare id or by whole key."""
    for entry in access_list.ids:
        candidate = entry.strip()
        if candidate and candidate in (ident, chat_key):
            return True
    return False


def is_muted(config: AccessConfig, chat_key: ChatKey | str) -> bool:
    """True when the bot must not speak in this chat.

    An empty blacklist mutes nothing (the default — adding the section
    changes no behaviour). An empty WHITELIST mutes everything, which is
    what a whitelist means; the doctor reports that state loudly rather
    than leaving an operator wondering why the bot went quiet.
    """
    kind = chat_kind(chat_key)
    if kind is None:
        return False
    access_list = config.groups if kind == GROUP else config.private
    if not access_list.enabled:
        return True
    listed = _listed(access_list, str(chat_key), chat_id(chat_key))
    if access_list.mode == "whitelist":
        return not listed
    return listed
