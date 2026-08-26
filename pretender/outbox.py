"""Durable outbox: at-most-once delivery, one adapter send per row.

Frozen decision #3: an outbox item in an ambiguous in-flight state is never
auto-retried after a crash — preventing duplicate social messages outweighs
rare loss. The driver:

  1. ``to_items`` converts an ``Outgoing`` into an ordered ``OutboxItem``
     batch — a PURE conversion, no database writes. Outbox rows are created
     ONLY by terminal cycle completion (``Repository.finish_cycle``), which
     stamps cycle provenance and rejects cross-chat items. Split output
     (``out.parts``) becomes a batch sharing ``group_id`` and ordered by
     ``seq``; text splitting itself is a later-phase output stage.

     Idempotency keys represent DELIVERY INTENT, not content: they are
     deterministic per ``cycle_id`` plus part index, so the same text may
     send in different cycles while retrying the same completed cycle
     remains idempotent. An explicit caller key is only a namespaced
     component of the derived key.
  2. ``pump`` lists ready items (``send_after_ts`` passed), CASes each
     ``pending -> in_flight`` BEFORE invoking the adapter, sends, and on a
     confirmed send atomically marks the row ``sent`` and writes the
     synthetic self echo (``mark_outbox_sent``). A send that raises leaves
     the item in_flight forever — the outcome is ambiguous, so it is not
     retried; a later real echo may reconcile it. ``drain`` loops ``pump``
     until every due pending row is attempted (the per-call limit only
     bounds one round).
  3. Every outgoing message carries the item's delivery/idempotency key in
     ``Outgoing.delivery_key`` (transport metadata), so a real platform
     echo can carry it back and prove an ambiguous send landed — ingest
     reconciles only with that trusted key.

Typed against the Repository and Adapter seams (protocols), never against
concrete implementations. Timestamps come from the injected clock
(absolute epoch seconds); this module never touches ``time``.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pretender.clock import RealClock
from pretender.log import get_logger
from pretender.errors import AdapterNotReady
from pretender.seams import Adapter, Repository
from pretender.types import ChatKey, CycleId, MessageId, OutboxItem, Outgoing

log = get_logger("outbox")


class OutboxDriver:
    """Turns Outgoings into durable rows (via finish_cycle) and drives
    at-most-once sends."""

    def __init__(self, repo: Repository, adapter: Adapter, clock: Any = None) -> None:
        self._repo = repo
        self._adapter = adapter
        self._clock = clock if clock is not None else RealClock()

    # ── pure conversion (rows are created only by finish_cycle) ─────────────

    def to_items(self, out: Outgoing, cycle_id: CycleId) -> list[OutboxItem]:
        """Convert one Outgoing into its ordered OutboxItem batch.

        Idempotency keys are ``{cycle_id}:{part_index}`` (with an explicit
        caller key as a namespaced component): delivery intent, not
        content — the same text sends in different cycles, while retrying
        the same completed cycle produces the same keys and hydrates the
        same rows.
        """
        # Text splitting must never duplicate a non-text segment payload for
        # every part. Keep segmented Outgoings atomic until a caller provides
        # explicit per-part segments.
        if out.segments:
            parts = [out.text]
        else:
            parts = list(out.parts) if out.parts else ([out.text] if out.text else [])
        if not parts:
            raise ValueError("cannot build an outbox batch from an empty Outgoing")
        if out.group_id is not None:
            group_id = out.group_id
        else:
            # Content-derived: a retried finish produces the same grouping.
            digest = hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()
            group_id = f"g:{digest[:16]}"
        items: list[OutboxItem] = []
        for i, text in enumerate(parts):
            if out.idem_key is not None:
                idem_key = (
                    f"{cycle_id}:{out.idem_key}:{i}"
                    if len(parts) > 1
                    else f"{cycle_id}:{out.idem_key}"
                )
            else:
                idem_key = f"{cycle_id}:{i}"
            items.append(
                OutboxItem(
                    chat_key=out.chat_key,
                    text=text,
                    idem_key=idem_key,
                    segments=tuple(out.segments),
                    payload=dict(out.platform_ref),
                    reply_to=out.reply_to,
                    group_id=group_id,
                    seq=i if len(parts) > 1 else None,
                    send_after_ts=out.send_after_ts,
                )
            )
        return items

    # ── pump / drain ────────────────────────────────────────────────────────

    async def pump(self, chat_key: ChatKey, *, now: float | None = None, limit: int = 10) -> int:
        """Send up to ``limit`` ready items for the chat; returns the send
        count.

        Each item: CAS to in_flight (committed before the adapter runs),
        ``adapter.send``, then mark sent + self echo. A failed send leaves
        the item in_flight and is never retried by this or any later pump.
        """
        ts: float = self._clock.now() if now is None else now
        items = await self._repo.list_ready_outbox(chat_key, now=ts, limit=limit)
        sent = 0
        for item in items:
            assert item.id is not None
            if not self._adapter_ready():
                return sent
            if not await self._repo.attempt_outbox(item.id, ts):
                continue  # no longer pending — someone else took it
            try:
                platform_id = await self._adapter.send(self._to_outgoing(item))
            except AdapterNotReady:
                # This adapter-specific error is raised only before a wire
                # write starts, so retrying the durable row is safe.
                await self._repo.requeue_outbox(item.id)
                return sent
            except Exception:
                # Ambiguous outcome: stay in_flight, never reset/retry.
                log.exception("outbox item %s send failed; left in_flight", item.id)
                continue
            pid = MessageId(platform_id) if platform_id is not None else None
            await self._repo.mark_outbox_sent(item.id, pid, ts)
            sent += 1
        return sent

    def _adapter_ready(self) -> bool:
        # App owns protocol-level readiness (lifecycle/API handshake). The
        # outbox's narrower per-item fence only proves whether a wire write can
        # start right now; requiring OneBot.ready here would deadlock the
        # adapter's own readiness probe and low-level adapter tests.
        connected = getattr(self._adapter, "connected", None)
        return True if connected is None else bool(connected)

    async def drain(self, chat_key: ChatKey, *, now: float | None = None, limit: int = 10) -> int:
        """Send EVERY due pending row for the chat, looping past the
        per-round limit. Returns the total number of sends."""
        total = 0
        while True:
            sent = await self.pump(chat_key, now=now, limit=limit)
            total += sent
            if sent == 0:
                return total

    def _to_outgoing(self, item: OutboxItem) -> Outgoing:
        return Outgoing(
            chat_key=item.chat_key,
            text=item.text,
            segments=list(item.segments),
            reply_to=item.reply_to,
            group_id=item.group_id,
            platform_ref=dict(item.payload),
            # The delivery/idempotency key rides the outgoing transport
            # metadata so a real platform echo can carry it back and prove
            # an ambiguous send landed (trusted-key reconciliation).
            delivery_key=item.idem_key,
        )
