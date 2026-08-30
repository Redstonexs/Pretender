"""Where the bot is allowed to speak: the ``[access]`` group / private lists.

The invariants that matter are behavioural: the default allows everything
(adding the section changes nothing), an excluded chat is silent even when
directly @-ed, a chat key that is neither group nor private is out of scope
(so a group whitelist never takes the console down with it), and a mute
stops queued outbox rows as well as new replies.
"""

from __future__ import annotations

import pytest

from pretender.access import chat_id, chat_kind, is_muted
from pretender.config import AccessConfig, AccessListConfig, Config
from pretender.errors import ConfigError

GROUP = "qq:group:123456"
OTHER_GROUP = "qq:group:999"
PRIVATE = "qq:private:10001"
OTHER_PRIVATE = "qq:private:20002"
CONSOLE = "console:local"


def _access(**kw) -> AccessConfig:
    return AccessConfig(**kw)


# ── key parsing ─────────────────────────────────────────────────────────────


def test_chat_kind_and_id():
    assert chat_kind(GROUP) == "group"
    assert chat_kind(PRIVATE) == "private"
    assert chat_id(GROUP) == "123456"
    assert chat_id(PRIVATE) == "10001"


def test_a_key_that_is_neither_group_nor_private_has_no_kind():
    for key in (CONSOLE, "weird", "a:b", ""):
        assert chat_kind(key) is None


# ── the default is "no change" ──────────────────────────────────────────────


def test_the_default_allows_everything():
    """An empty blacklist on both lists is exactly today's behaviour, so
    adding the section to an existing config changes nothing."""
    access = _access()
    for key in (GROUP, PRIVATE, CONSOLE):
        assert is_muted(access, key) is False


def test_an_empty_config_carries_the_permissive_default():
    cfg = Config.loads("")
    assert cfg.access.groups.mode == "blacklist"
    assert cfg.access.groups.ids == ()
    assert is_muted(cfg.access, GROUP) is False


# ── blacklist ───────────────────────────────────────────────────────────────


def test_blacklist_silences_only_what_is_listed():
    access = _access(groups=AccessListConfig(ids=("123456",)))
    assert is_muted(access, GROUP) is True
    assert is_muted(access, OTHER_GROUP) is False
    # The group list never touches private chats.
    assert is_muted(access, PRIVATE) is False


def test_blacklist_matches_the_whole_chat_key_too():
    """An operator reading the log sees ``qq:group:123456``; making them
    translate it back to a bare id is a needless way to get it wrong."""
    access = _access(groups=AccessListConfig(ids=("qq:group:123456",)))
    assert is_muted(access, GROUP) is True
    assert is_muted(access, OTHER_GROUP) is False


def test_blacklist_entries_are_whitespace_tolerant():
    access = _access(groups=AccessListConfig(ids=("  123456  ",)))
    assert is_muted(access, GROUP) is True


# ── whitelist ───────────────────────────────────────────────────────────────


def test_whitelist_silences_everything_it_does_not_list():
    access = _access(groups=AccessListConfig(mode="whitelist", ids=("123456",)))
    assert is_muted(access, GROUP) is False
    assert is_muted(access, OTHER_GROUP) is True


def test_an_empty_whitelist_allows_nothing():
    """That is what a whitelist means. It is a footgun, which is why the
    doctor reports it out loud rather than leaving it to be discovered."""
    access = _access(groups=AccessListConfig(mode="whitelist"))
    assert is_muted(access, GROUP) is True
    assert is_muted(access, OTHER_GROUP) is True
    # Still scoped to groups.
    assert is_muted(access, PRIVATE) is False


def test_the_two_lists_are_independent():
    access = _access(
        groups=AccessListConfig(mode="whitelist", ids=("123456",)),
        private=AccessListConfig(ids=("10001",)),
    )
    assert is_muted(access, GROUP) is False
    assert is_muted(access, OTHER_GROUP) is True
    assert is_muted(access, PRIVATE) is True
    assert is_muted(access, OTHER_PRIVATE) is False


# ── the category switch ─────────────────────────────────────────────────────


def test_disabling_private_silences_every_private_chat():
    access = _access(private=AccessListConfig(enabled=False))
    assert is_muted(access, PRIVATE) is True
    assert is_muted(access, OTHER_PRIVATE) is True
    assert is_muted(access, GROUP) is False


def test_disabling_groups_silences_every_group():
    access = _access(groups=AccessListConfig(enabled=False))
    assert is_muted(access, GROUP) is True
    assert is_muted(access, OTHER_GROUP) is True
    assert is_muted(access, PRIVATE) is False


def test_disabled_beats_a_listing():
    """``enabled = false`` is the whole category, so being on the whitelist
    does not buy an exception."""
    access = _access(
        groups=AccessListConfig(enabled=False, mode="whitelist", ids=("123456",))
    )
    assert is_muted(access, GROUP) is True


# ── the console is out of scope ─────────────────────────────────────────────


def test_a_group_whitelist_never_silences_the_console():
    """These lists are about QQ groups and DMs. Silently taking the local
    console down with them would make the dev loop inexplicable."""
    access = _access(
        groups=AccessListConfig(mode="whitelist", ids=("123456",)),
        private=AccessListConfig(enabled=False),
    )
    assert is_muted(access, CONSOLE) is False


# ── config validation ───────────────────────────────────────────────────────


def test_config_rejects_an_unknown_mode():
    with pytest.raises(ConfigError, match="blacklist"):
        AccessListConfig(mode="allowlist")


def test_config_rejects_blank_ids():
    for bad in ("", "   ", 123):
        with pytest.raises(ConfigError, match="ids"):
            AccessListConfig(ids=(bad,))


def test_config_rejects_a_non_boolean_enabled():
    with pytest.raises(ConfigError, match="boolean"):
        AccessListConfig(enabled="yes")


def test_access_loads_from_toml():
    cfg = Config.loads(
        """
        [access.groups]
        mode = "whitelist"
        ids = ["123456", "qq:group:999"]

        [access.private]
        enabled = false
        """
    )
    assert cfg.access.groups.mode == "whitelist"
    assert cfg.access.groups.ids == ("123456", "qq:group:999")
    assert cfg.access.private.enabled is False
    assert is_muted(cfg.access, GROUP) is False
    assert is_muted(cfg.access, OTHER_GROUP) is False  # listed by whole key
    assert is_muted(cfg.access, "qq:group:5") is True
    assert is_muted(cfg.access, PRIVATE) is True
