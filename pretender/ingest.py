"""Inbound ingestion: record, commit, THEN wake.

Ordering invariant (PLAN.md §4, "commit before wake"): the normalized event
is written to the JSONL recorder first, then chat identity + message are
committed durably in ONE transaction, and only then is the scheduler woken.
If the wake fired first, the gate would read an empty pending set, return
``skip``, and consume the only wake.

Dispatch-ledger ordering (frozen Oracle advisory): a stable ``EventId`` is
generated BEFORE recording — the corpus event line and the durable commit
metadata share one identity. The recorder writes/flushes the raw event
first; ``Repository.ingest_message`` then atomically creates the message's
``inbound_commits`` metadata (monotonic sequence, event id, chat/message
identity, committed timestamp, wake kind, pending count) and returns the
event/commit/wake data (``event_id``, ``commit_seq``, ``wake_kind``). The
commit marker is exported at-least-once right after the commit (append
marker, then mark exported); a crash between the two is repaired by the
startup export, which re-appends the marker — readers deduplicate by
``(record_type, sequence)``. Duplicates and self echoes use ``wake_kind``
``none`` as appropriate: a duplicate commits no new row, and a committed
self echo is ledger-complete but never attached to a dispatch.

Duplicate input never wakes: the message insert dedupes on
``(platform, self_id, platform_msg_id)`` (ON CONFLICT DO NOTHING), and a
duplicate returns ``inserted=False`` — no wake, no second commit. This is
also what reconciles a real platform echo against the synthetic self echo
the outbox wrote: the echo deduplicates without a second send.

Every ``handle`` returns the typed ``IngestResult`` (durable row id,
inserted flag, echo status, atomic pending count, and the event/commit/wake
data). The result is surfaced verbatim; the wrapper never independently
wakes anything beyond the existing ``inserted`` rule. A verified SELF echo
may atomically reconcile an ambiguous in-flight send ONLY with the trusted
delivery key: the ``delivery_key`` resolver extracts the outbox item's
delivery/idempotency key from the transport metadata the adapter forwarded
(the same key ``OutboxDriver`` put on the outgoing message).
Missing/untrusted keys are ``unproven``; the repository never heuristically
matches.

The chat identity is resolved through an injected callable (the app wires
it to the adapter's identity); a message from an unknown chat is recorded
but not committed — platform/self identity must derive safely from a known
chat, never from the message itself.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any, Awaitable, Callable

from pretender.clock import RealClock
from pretender.log import get_logger
from pretender.record import Recorder, export_marker
from pretender.seams import Repository
from pretender.types import (
    AdapterEvent,
    ChatIdentity,
    ChatKey,
    CorpusMarker,
    EchoStatus,
    EventId,
    IngestResult,
    Message,
    MessageRowId,
    WakeKind,
)

log = get_logger("ingest")

WakeFn = Callable[[ChatKey], Awaitable[Any]]
IdentityFn = Callable[[ChatKey], ChatIdentity | None]
DeliveryKeyFn = Callable[[Message], str | None]
ObservePersonFn = Callable[[Message], Awaitable[Any]]
HarvestMediaFn = Callable[[Message, MessageRowId | None], Awaitable[Any]]
RevokeMediaFn = Callable[[AdapterEvent], Awaitable[Any]]


class Ingest:
    """The inbound pipeline: normalize -> record -> commit -> wake.

    Typed against the Repository seam (protocol), never against a concrete
    implementation.
    """

    def __init__(
        self,
        repo: Repository,
        recorder: Recorder,
        *,
        wake: WakeFn | None = None,
        identity: IdentityFn | None = None,
        clock: Any = None,
        delivery_key: DeliveryKeyFn | None = None,
        observe_person: ObservePersonFn | None = None,
        harvest_media: HarvestMediaFn | None = None,
        revoke_media: RevokeMediaFn | None = None,
    ) -> None:
        self._repo = repo
        self._recorder = recorder
        self._wake = wake
        self._identity = identity
        self._clock = clock if clock is not None else RealClock()
        self._delivery_key = delivery_key
        # Post-ingest person observation (Phase 5): an injected chat-scoped
        # callback that observes a newly inserted non-self message's sender
        # alias. Any observation failure is contained and never undoes the
        # ingest or the scheduler wake.
        self._observe_person = observe_person
        # Post-ingest media harvesting (Phase 6 P6.5b): an injected
        # chat-scoped callback that schedules a bounded background harvest
        # for a newly inserted non-self message (the App wires it to the
        # MediaHarvester). Any harvest failure is contained and never undoes
        # the ingest or the scheduler wake.
        self._harvest_media = harvest_media
        # Source-deletion/recall revocation (Phase 6 P6.5b): an injected
        # callback offered every non-message event (the App wires it to the
        # MediaRevoker). The revoker is contained and self-filtering — a
        # non-recall event is a no-op, and any failure never undoes the
        # recording or raises into the caller.
        self._revoke_media = revoke_media
        if self._revoke_media is None and all(
            hasattr(repo, name)
            for name in (
                "get_message",
                "list_media_candidates",
                "list_media_assets",
                "reject_media_candidate",
                "revoke_media_asset",
            )
        ):
            # Keep the base Repository seam unchanged while making the
            # existing recall-shaped notice useful for the concrete repository
            # the app already injects.  MediaRevoker is fully contained and
            # self-filters non-recall events.
            from pretender.media import MediaRevoker

            self._revoke_media = MediaRevoker(repo).maybe_revoke

    async def handle(
        self,
        event: AdapterEvent,
        *,
        structural_priority: bool = False,
        pending_threshold: int | None = None,
    ) -> IngestResult:
        """Process one adapter event.

        Returns the typed ``IngestResult``: ``inserted`` is True when the
        event was a NEW message that committed and woke the scheduler;
        ``echo_status`` reports the self-echo reconciliation verdict
        (``not_applicable`` for ordinary inbound messages, ``unproven``
        for self messages without a trusted delivery key);
        ``pending_count`` is the atomic current pending non-self count
        for a newly inserted message (None for noninserted/self);
        ``event_id``/``commit_seq``/``wake_kind``/``priority`` are the dispatch-ledger
        event/commit/wake data (the commit marker is exported at-least-
        once right after the commit). The result is surfaced verbatim —
        no additional wake is performed here beyond the ``inserted``
        rule.
        """
        if event.type != "message" or not isinstance(event.payload, Message):
            self._recorder.write_event(event)
            # Source-deletion/recall revocation (Phase 6 P6.5b): every
            # non-message event is offered to the injected revoker (the App
            # wires it to the MediaRevoker). The revoker is contained and
            # self-filtering — a non-recall notice is a no-op, and any
            # failure never undoes the recording or raises into the caller.
            if self._revoke_media is not None:
                try:
                    await self._revoke_media(event)
                except Exception:
                    log.warning(
                        "media recall revocation failed for %s (contained)",
                        event.type,
                        exc_info=True,
                    )
            return IngestResult(echo_status=EchoStatus.NOT_APPLICABLE)

        msg = event.payload
        if msg.recv_ts is None:
            msg = replace(msg, recv_ts=self._clock.now())
        if event.ts is None:
            event = replace(event, ts=msg.recv_ts)

        # 0. Stable event id, generated BEFORE recording: the corpus event
        # line and the durable commit metadata share one identity.
        event_id = EventId(uuid.uuid4().hex)

        # 1. Always-on recording, before anything else.
        self._recorder.write_event(event, event_id=event_id)

        # 2. Durable commit of identity + message + inbound commit
        # metadata, one transaction.
        identity = self._identity(msg.chat_key) if self._identity is not None else None
        if identity is None:
            log.warning(
                "dropping message from unknown chat %s (no identity resolver)",
                msg.chat_key,
            )
            return IngestResult(echo_status=EchoStatus.NOT_APPLICABLE)
        key = (
            self._delivery_key(msg)
            if msg.is_self and self._delivery_key is not None
            else None
        )
        result = await self._repo.ingest_message(
            identity,
            msg,
            self_echo_delivery_key=key,
            event_id=event_id,
            structural_priority=structural_priority,
            pending_threshold=pending_threshold,
        )

        # 3. At-least-once commit marker export: append the marker AFTER
        # the durable commit, then mark it exported. A crash between the
        # two is repaired by the startup export (duplicates are tolerated;
        # readers dedupe by (record_type, sequence)).
        if result.commit_seq is not None:
            marker = CorpusMarker(
                record_type="commit",
                sequence=result.commit_seq,
                chat_key=msg.chat_key,
                event_id=result.event_id,
                wake_kind=result.wake_kind,
                message_row_id=result.row_id,
                priority=result.priority,
            )
            await export_marker(self._recorder, self._repo, marker)

        # 4. Wake only for a genuinely new, scheduler-eligible inbound
        # message (commit-before-wake). Self echoes have a durable commit for
        # presence/replay but wake_kind ``none`` and must not resurrect old
        # pending work through the public callback path.
        if (
            result.inserted
            and result.wake_kind == WakeKind.INBOUND
            and self._wake is not None
        ):
            await self._wake(msg.chat_key)

        # 5. Post-ingest person observation (Phase 5): after a successful
        # durable newly-inserted NON-SELF message, observe its sender alias.
        # Self echoes, duplicates (inserted=False), and untrusted echoes are
        # never observed. Any observation failure is contained — it must not
        # undo the ingest or the scheduler wake above.
        if result.inserted and not msg.is_self and self._observe_person is not None:
            try:
                await self._observe_person(msg)
            except Exception:
                log.warning(
                    "person observation failed for %s/%s (contained)",
                    msg.chat_key,
                    msg.sender_id,
                    exc_info=True,
                )
        # 6. Post-ingest media harvesting (Phase 6 P6.5b): after a successful
        # durable newly-inserted NON-SELF message, offer it to the harvest
        # lane (the App's MediaHarvester schedules a bounded background task
        # when the policy/kind allows). Self echoes, duplicates, and
        # untrusted echoes are never harvested. Any harvest failure is
        # contained — it must not undo the ingest or the scheduler wake.
        if result.inserted and not msg.is_self and self._harvest_media is not None:
            try:
                await self._harvest_media(msg, result.row_id)
            except Exception:
                log.warning(
                    "media harvest scheduling failed for %s (contained)",
                    msg.chat_key,
                    exc_info=True,
                )
        return result
