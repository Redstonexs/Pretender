"""Phase 5 person service: per-chat person identity over KnowledgeRepository.

Pure/durable: observes a chat-scoped platform UID + name idempotently,
normalizes/dedupes/bounds aliases deterministically, retrieves a profile
safely, and applies profile/impression updates via ``cas_person_profile``
with a caller-provided fixed through-message cursor.

Deliberately narrow: no global nickname matching, no LLM/network, no
implicit source scan, no ingestion/cycle wiring. Future integration calls
this after a durable message insert; Ingest is untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pretender.seams import KnowledgeRepository
from pretender.types import (
    ChatKey,
    MessageRowId,
    PersonKey,
    PersonProfile,
    SenderId,
)

__all__ = ["MAX_ALIASES", "ObserveResult", "PersonService", "ProfileResult"]

# The deterministic cap on how many aliases one person may carry. New names
# beyond the cap are dropped (never evicting an existing alias), so the
# alias list is bounded and stable under repeated observation.
MAX_ALIASES = 8


@dataclass(frozen=True)
class ObserveResult:
    """Outcome of one idempotent alias observation.

    ``names`` is the resulting deterministic alias list (first-seen order,
    deduped, bounded). ``created`` is True when a new person row was
    created; ``added`` is True when the observed name was newly merged into
    the alias list (False for a repeat/blank observation).
    """

    chat_key: ChatKey
    platform_uid: SenderId
    person_key: PersonKey | None
    names: tuple[str, ...]
    created: bool
    added: bool


@dataclass(frozen=True)
class ProfileResult:
    """Outcome of one profile/impression CAS application.

    ``applied`` is True only when the CAS commit succeeded. When False,
    ``reason`` is ``"unknown_person"`` or ``"stale_cas"`` and the fields
    reflect the unchanged stored state.
    """

    applied: bool
    reason: str | None = None
    person_key: PersonKey | None = None
    profile: str | None = None
    impression: str | None = None
    through_msg_id: MessageRowId | None = None


class PersonService:
    """Pure/durable per-chat person identity over ``KnowledgeRepository``.

    Depends only on the ``KnowledgeRepository`` seam (``get_person``,
    ``upsert_person``, ``cas_person_profile``) — never on
    ``SqliteRepository``, an LLM, or a source scan.
    """

    def __init__(self, repo: KnowledgeRepository) -> None:
        self._repo = repo

    @staticmethod
    def _normalize(name: str) -> str | None:
        """Strip and collapse internal whitespace; None when blank."""
        if not isinstance(name, str):
            return None
        norm = " ".join(name.split())
        return norm or None

    async def observe(
        self,
        chat_key: ChatKey,
        platform_uid: SenderId,
        name: str,
        *,
        now: float | None = None,
    ) -> ObserveResult:
        """Idempotently observe a chat-scoped platform UID + name.

        Blank names are ignored (no-op). Aliases are merged ATOMICALLY via
        the repository's ``add_person_alias`` (first-seen order, dedupe,
        bounded to ``MAX_ALIASES``) — never a read-modify-upsert that could
        overwrite a concurrent profile write. Existing profile/impression
        are preserved. Returns the resulting alias state.
        """
        norm = self._normalize(name)
        if norm is None:
            return ObserveResult(
                chat_key=chat_key,
                platform_uid=platform_uid,
                person_key=None,
                names=(),
                created=False,
                added=False,
            )
        existing = await self._repo.get_person(chat_key, platform_uid)
        created = existing is None
        names = await self._repo.add_person_alias(
            chat_key, platform_uid, norm, now=now
        )
        if names is None:
            return ObserveResult(
                chat_key=chat_key,
                platform_uid=platform_uid,
                person_key=None,
                names=(),
                created=False,
                added=False,
            )
        added = norm not in (existing.names if existing is not None else ())
        stored = await self._repo.get_person(chat_key, platform_uid)
        person_key = stored.person_key if stored is not None else None
        return ObserveResult(
            chat_key=chat_key,
            platform_uid=platform_uid,
            person_key=person_key,
            names=names,
            created=created,
            added=added,
        )

    async def get_profile(
        self, chat_key: ChatKey, platform_uid: SenderId
    ) -> PersonProfile | None:
        """Retrieve the person's profile safely; None when unknown."""
        return await self._repo.get_person(chat_key, platform_uid)

    async def apply_profile(
        self,
        chat_key: ChatKey,
        platform_uid: SenderId,
        *,
        update: PersonProfile,
        through_msg_id: MessageRowId,
        expected_through_msg_id: MessageRowId | None,
        now: float | None = None,
    ) -> ProfileResult:
        """Apply a profile/impression update via ``cas_person_profile``.

        ``through_msg_id`` is the caller-provided fixed cursor to write;
        ``expected_through_msg_id`` is the CAS fence. Fails closed on
        cross-chat/person mismatch (ValueError), a non-finite ``now``
        (ValueError), an unknown person (``unknown_person``), and a stale
        CAS (``stale_cas`` — nothing changes).
        """
        if update.chat_key != chat_key or update.platform_uid != platform_uid:
            raise ValueError(
                "cross-chat/person mismatch: update targets "
                f"{update.chat_key!r}/{update.platform_uid!r}, not "
                f"{chat_key!r}/{platform_uid!r}"
            )
        if now is not None and not math.isfinite(now):
            raise ValueError("now must be finite")
        existing = await self._repo.get_person(chat_key, platform_uid)
        if existing is None:
            return ProfileResult(applied=False, reason="unknown_person")
        names = update.names if update.names else existing.names
        new_profile = PersonProfile(
            chat_key=chat_key,
            platform_uid=platform_uid,
            names=names,
            profile=update.profile,
            impression=update.impression,
            updated_ts=now,
            profile_through_msg_id=through_msg_id,
        )
        ok = await self._repo.cas_person_profile(
            chat_key, platform_uid, expected_through_msg_id, new_profile
        )
        if not ok:
            # Stale CAS: reread the CURRENT stored state and return the
            # winner's fields — never the loser's, and never erasing the
            # winner's alias/profile fields.
            winner = await self._repo.get_person(chat_key, platform_uid)
            if winner is None:
                return ProfileResult(applied=False, reason="unknown_person")
            return ProfileResult(
                applied=False,
                reason="stale_cas",
                person_key=winner.person_key,
                profile=winner.profile,
                impression=winner.impression,
                through_msg_id=winner.profile_through_msg_id,
            )
        return ProfileResult(
            applied=True,
            person_key=existing.person_key,
            profile=update.profile,
            impression=update.impression,
            through_msg_id=through_msg_id,
        )
