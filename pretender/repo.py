"""The storage seam implementation — the ONLY place SQL text lives.

Implements the async ``Repository`` protocol from seams.py exactly, with no
essential concrete-only surface: every semantic runtime operation the
runtime needs is on the protocol. SQL text lives here (schema DDL lives in
schema.sql); the CLI, ingest, and outbox contain none.

Correctness invariants (PLAN.md §4 and the Gate 1 remediation):
  - ``ON CONFLICT DO NOTHING`` everywhere: one duplicate must never poison
    a 200-op writer batch (each op is additionally isolated in its own
    savepoint, so an expected CAS/fencing failure rolls back only itself).
  - Message uniqueness is ``(platform, self_id, platform_msg_id)``; the
    platform/self identity is derived from the chat row, never from the
    message (a Message carries no platform identity).
  - Forwarded segment payloads never persist: ``_serialize_segments`` drops
    the content of ``forward`` segments (adapters render them as a
    placeholder in ``text``).
  - CJK-bigram FTS updates happen in the same transaction as the message
    insert (external-content FTS5).
  - Cursor movement is reachable ONLY through ``finish_cycle``, which
    derives the new cursor from the stored claim's through boundary after
    checking ownership, start cursor, and unexpired lease. Neither
    ``upsert_chat_state`` nor any caller-provided value can move it, the
    hold window, or the idle streak: the durable hold (``hold_until``,
    None clears it) and idle streak (``idle_streak_after``) are
    materialized in the SAME transaction as the cursor advance, so a later
    ``save_session`` can never reintroduce a crash gap for them.
  - Outbox rows are created ONLY by ``finish_cycle`` (terminal cycle
    completion with a completed durable claim), stamped with cycle
    provenance; cross-chat items are rejected. ``pending -> in_flight`` is
    a durable CAS before the adapter is invoked; only ``in_flight`` can
    become ``sent``; ``in_flight`` is never auto-retried after a restart.
  - A verified self echo reconciles an ambiguous in-flight send ONLY with
    the trusted delivery key (the outbox row's idem_key, forwarded through
    the outgoing transport metadata): sender must match the chat's self
    id, the chat/text/canonical segment/reply payload must match exactly
    one in-flight row, and the transition records the real platform
    id/timestamp. The real echo insertion and the reconciliation share ONE
    transaction and never create a second synthetic echo; missing or
    untrusted keys are ``unproven``, mismatches/wrong state/sender/
    ambiguity are ``conflict``, and duplicate echo events reconcile
    idempotently. This trusted-key ``ingest_message`` flow is the ONLY
    reconciliation path — there is no untrusted item-id surface.
  - ``ingest_message`` atomically folds every newly inserted NON-SELF
    message into the chat's durable ``avg_interval`` using the same
    dependency-neutral ``pacing.ewma_interval`` reducer the runtime
    session layer uses, from prior durable data (the stored average and
    the previous non-self message's timestamp). Self, duplicate, and
    invalid-time messages never change it; only ``avg_interval`` is
    written — the cursor, the hold window, and the idle streak stay
    terminal-owned.
  - ``list_pending_chats`` is the startup-recovery read: every chat with
    a non-self message beyond its durable cursor, in one query.
  - The durable dispatch ledger (frozen Oracle advisory) is the minimal
    serialized dispatch order: ``ingest_message`` atomically creates each
    committed message's ``inbound_commits`` row (stable event id, wake
    kind, pending count); ``begin_dispatch`` atomically claims a prepared
    dispatch, freezes the commit boundary, attaches eligible unassigned
    commits, and records the dispatch; ``settle_dispatch`` owns ALL
    release/delay/terminal movement — the cursor and the outbox move ONLY
    inside a terminal finish. Recovering an expired prepared dispatch
    detaches its attached commits (``dispatch_id = NULL``) in the SAME
    transaction, so they stay eligible for the fresh dispatch instead of
    stranding after a crash. The at-least-once export surface lists
    unexported commit/dispatch markers and marks one exported; the startup
    export appends markers then marks them exported, and readers
    deduplicate by (record_type, sequence). The legacy claim_cycle/
    finish_cycle surface remains for compatibility with the current cycle
    lane; the next integration lane switches all live use to the ledger.
  - No timestamps are produced here — callers pass absolute epoch seconds.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable

import orjson

from pretender.db import Database
from pretender.emoji import validate_candidate
from pretender.errors import ClaimError, RepoError
from pretender.pacing import ewma_interval
from pretender.person import MAX_ALIASES
from pretender.types import (
    ChatControl,
    ChatIdentity,
    ChatKey,
    ChatState,
    ClaimBusy,
    ClaimGrant,
    CommitSeq,
    CorpusMarker,
    CycleClaim,
    CycleFinish,
    CycleId,
    DispatchCause,
    DispatchDeferred,
    DispatchGrant,
    DispatchId,
    DispatchRequest,
    DispatchSettle,
    EchoStatus,
    EmbeddingGeneration,
    EventId,
    IngestResult,
    LearnerBatch,
    LearnerBusy,
    LearnerDraft,
    LearnerGrant,
    LearnerRun,
    LearnerRunRequest,
    LearnerState,
    LexicalHit,
    MediaAsset,
    MediaAssetCandidate,
    MediaKind,
    MemoryRecord,
    MemorySourceBatch,
    MemoryWriteRequest,
    Message,
    MessageId,
    MessageRowId,
    OutboxItem,
    PersonKey,
    PersonProfile,
    PlatformId,
    RecentSnapshot,
    Record,
    RecordHit,
    Segment,
    SelfId,
    SenderId,
    VectorRow,
    WakeKind,
    _MEDIA_KINDS,
)

# ── CJK bigram tokenizer ────────────────────────────────────────────────────

_CJK_RUN = re.compile(r"([\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+)")
_NO_SOURCE_HASH = object()


def bigram_tokenize(text: str) -> list[str]:
    """Tokenize text for the CJK-bigram FTS index.

    Pure CJK runs become overlapping bigrams (``火锅好吃`` ->
    ``火锅 锅好 好吃``); a single CJK char stays as itself; non-CJK runs are
    kept whole. FTS5's unicode61 returns zero rows for a Chinese substring
    query and trigram needs >= 3 chars, so bigrams are what make a
    two-character Chinese search work.
    """
    tokens: list[str] = []
    for run in _CJK_RUN.split(text):
        if not run:
            continue
        if _CJK_RUN.fullmatch(run):
            chars = list(run)
            if len(chars) == 1:
                tokens.append(chars[0])
            else:
                tokens.extend("".join(pair) for pair in zip(chars, chars[1:]))
        else:
            tokens.append(run)
    return tokens


# ── Repository ──────────────────────────────────────────────────────────────

class SqliteRepository:
    """Concrete Repository over a ``Database``. All methods async; all SQL
    text lives here (schema DDL lives in schema.sql)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    @staticmethod
    def read_chat_identity_readonly(
        path: str | Path, chat_key: ChatKey
    ) -> ChatIdentity | None:
        """Read one replay identity through SQLite's read-only mode.

        Replay must not open the writable Database owner because that enables
        WAL/migrations. SQL remains in the repository boundary, while callers
        receive the normal typed identity.
        """
        db_path = Path(path).expanduser().resolve()
        try:
            conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
            try:
                row = conn.execute(
                    "SELECT chat_key, platform, self_id, kind, title FROM chats"
                    " WHERE chat_key = ?",
                    (chat_key,),
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error as e:
            raise RepoError(f"read-only chat lookup failed: {db_path}") from e
        if row is None:
            return None
        return ChatIdentity(
            ChatKey(row[0]), PlatformId(row[1]), SelfId(row[2]), row[3], title=row[4]
        )

    @staticmethod
    def read_settled_dispatch_ids_readonly(
        path: str | Path, chat_key: ChatKey
    ) -> frozenset[DispatchId]:
        """Read the durable settled-dispatch identity set without opening the
        writable Database owner.

        Exact corpus replay compares this witness to exported markers so a
        truncated JSONL file cannot be accepted merely because its surviving
        dispatches are internally consistent.
        """
        db_path = Path(path).expanduser().resolve()
        try:
            conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
            try:
                rows = conn.execute(
                    "SELECT id FROM dispatches WHERE chat_key = ?"
                    " AND state IN ('completed', 'released')"
                    " AND settled_ts IS NOT NULL",
                    (chat_key,),
                ).fetchall()
            finally:
                conn.close()
        except sqlite3.Error as e:
            raise RepoError(f"read-only dispatch lookup failed: {db_path}") from e
        return frozenset(DispatchId(row[0]) for row in rows)

    # ── chats: identity and runtime state ───────────────────────────────────

    async def get_chat(self, chat_key: ChatKey) -> ChatIdentity | None:
        def fn(conn: Any) -> ChatIdentity | None:
            row = conn.execute(
                "SELECT chat_key, platform, self_id, kind, title FROM chats"
                " WHERE chat_key = ?",
                (chat_key,),
            ).fetchone()
            if row is None:
                return None
            return ChatIdentity(
                chat_key=ChatKey(row[0]),
                platform=PlatformId(row[1]),
                self_id=SelfId(row[2]),
                kind=row[3],
                title=row[4],
            )

        return await self._db.read(fn)

    async def upsert_chat(self, chat: ChatIdentity) -> None:
        def fn(conn: Any) -> None:
            self._upsert_chat(conn, chat)

        await self._db.write(fn)

    def _upsert_chat(self, conn: Any, chat: ChatIdentity) -> None:
        conn.execute(
            "INSERT INTO chats(chat_key, platform, self_id, kind, title)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(chat_key) DO UPDATE SET"
            " platform = excluded.platform, self_id = excluded.self_id,"
            " kind = excluded.kind, title = excluded.title",
            (chat.chat_key, chat.platform, chat.self_id, chat.kind, chat.title),
        )

    async def get_chat_state(self, chat_key: ChatKey) -> ChatState | None:
        def fn(conn: Any) -> ChatState | None:
            row = conn.execute(
                "SELECT chat_key, cursor_msg_id, focus_until, hold_until,"
                " avg_interval, idle_streak, cfg_json, agent_resume_at,"
                " wait_streak FROM chats"
                " WHERE chat_key = ?",
                (chat_key,),
            ).fetchone()
            if row is None:
                return None
            return ChatState(
                chat_key=ChatKey(row[0]),
                cursor_msg_id=MessageRowId(row[1]) if row[1] is not None else None,
                focus_until=row[2],
                hold_until=row[3],
                avg_interval=row[4],
                idle_streak=row[5],
                cfg_json=row[6],
                agent_resume_at=row[7],
                wait_streak=row[8],
            )

        return await self._db.read(fn)

    async def upsert_chat_state(self, state: ChatState) -> None:
        def fn(conn: Any) -> None:
            # UPDATE-only, and NEVER the cursor, the hold window, the idle
            # streak, or the agent barrier (agent_resume_at / wait_streak):
            # those are written only by finish_cycle / settle_dispatch
            # (derived from the stored claim boundary and the terminal or
            # defer outcome), so a later save_session can never reintroduce
            # a crash gap for them. Identity arrives via upsert_chat first.
            conn.execute(
                "UPDATE chats SET focus_until = ?, avg_interval = ?,"
                " cfg_json = ? WHERE chat_key = ?",
                (
                    state.focus_until,
                    state.avg_interval,
                    state.cfg_json,
                    state.chat_key,
                ),
            )

        await self._db.write(fn)

    # ── messages: atomic identity+message commit with dedupe status ────────

    async def ingest_message(
        self,
        identity: ChatIdentity | None,
        msg: Message,
        *,
        self_echo_delivery_key: str | None = None,
        event_id: EventId | None = None,
        structural_priority: bool = False,
        pending_threshold: int | None = None,
    ) -> IngestResult:
        """Commit chat identity (when given) + message in ONE transaction.

        Returns the typed ``IngestResult``: the durable row id, the
        ``inserted`` flag (False for a duplicate ``(platform, self_id,
        platform_msg_id)`` row), the self-echo reconciliation status, and
        the atomic CURRENT pending non-self count (``pending_count``) for
        a newly inserted message — non-self messages beyond the durable
        cursor, including this one, computed in this same transaction.
        ``pending_count`` is None for duplicates, self messages, and
        anything that committed nothing. Platform/self identity derives
        from the chat row — the message itself carries none.

        Dispatch-ledger metadata (frozen Oracle advisory): every newly
        inserted message — self included — atomically creates its
        ``inbound_commits`` row in this SAME transaction, stamped with the
        stable ``event_id`` (generated by the caller BEFORE recording; a
        caller that passes none gets a generated one), the message's
        timestamp, the atomic pending count (non-self only), and the
        commit's ``WakeKind``: ``inbound`` for a newly inserted non-self
        message, ``none`` for a newly inserted self echo. Duplicates
        commit no new row (``commit_seq`` None). The result carries the
        event/commit/wake data: ``event_id``, ``commit_seq``, and
        ``wake_kind``.

        A verified SELF message may atomically reconcile an ambiguous
        in-flight send ONLY when ``self_echo_delivery_key`` is the trusted
        delivery key (the outbox row's idem_key, forwarded through the
        outgoing transport metadata): the sender must match the chat's
        self id, the chat/text/canonical segment/reply payload must match
        exactly one in-flight row, and the transition records the real
        platform id/timestamp. The real echo insertion and the
        reconciliation share this one transaction and never create a
        second synthetic echo. Missing/untrusted keys are ``unproven``;
        mismatches, wrong state/sender, or ambiguity are ``conflict`` and
        never move the outbox; duplicate echo events reconcile
        idempotently (``already_reconciled``). This trusted-key flow is
        the ONLY reconciliation path.

        Every newly inserted NON-SELF message additionally folds into the
        chat's durable ``avg_interval`` in this same transaction (the
        dependency-neutral ``pacing.ewma_interval`` reducer over prior
        durable data); self, duplicate, and invalid-time messages never
        change it.
        """

        def fn(conn: Any) -> IngestResult:
            if identity is not None:
                self._upsert_chat(conn, identity)
            if msg.is_self and self_echo_delivery_key is not None:
                # A trusted self echo may reconcile a synthetic local
                # fallback echo (written by ``mark_outbox_sent`` when the
                # platform returned no id) by updating that durable row to
                # the real platform id — instead of inserting a duplicate
                # context message. No new row, no wake, no resend.
                syn_row_id = self._reconcile_synthetic_echo(
                    conn, msg, self_echo_delivery_key
                )
                if syn_row_id is not None:
                    return IngestResult(
                        row_id=syn_row_id, inserted=False,
                        echo_status=EchoStatus.ALREADY_RECONCILED,
                    )
            row_id, inserted = self._insert_message(conn, msg)
            if not msg.is_self:
                if inserted:
                    self._update_avg_interval(conn, msg, row_id)
                    pending = self._pending_count(conn, msg.chat_key)
                    priority = structural_priority or (
                        pending_threshold is not None
                        and pending >= pending_threshold
                    )
                    ev_id, commit_seq = self._insert_commit(
                        conn,
                        msg,
                        row_id,
                        event_id,
                        WakeKind.INBOUND,
                        pending,
                        priority,
                    )
                    return IngestResult(
                        row_id=row_id, inserted=True,
                        echo_status=EchoStatus.NOT_APPLICABLE,
                        pending_count=pending,
                        event_id=ev_id, commit_seq=commit_seq,
                        wake_kind=WakeKind.INBOUND,
                        priority=priority,
                    )
                return IngestResult(
                    row_id=row_id, inserted=False,
                    echo_status=EchoStatus.NOT_APPLICABLE,
                )
            if inserted:
                # A committed self echo is ledger-complete but never wakes:
                # wake_kind none, no pending count.
                ev_id, commit_seq = self._insert_commit(
                    conn, msg, row_id, event_id, WakeKind.NONE, None, False
                )
                if self_echo_delivery_key is None:
                    # Missing/untrusted key: never heuristically matched.
                    return IngestResult(
                        row_id=row_id, inserted=True,
                        echo_status=EchoStatus.UNPROVEN,
                        event_id=ev_id, commit_seq=commit_seq,
                        wake_kind=WakeKind.NONE,
                        priority=False,
                    )
                status = self._reconcile_self_echo(conn, msg, self_echo_delivery_key)
                return IngestResult(
                    row_id=row_id, inserted=True, echo_status=status,
                    event_id=ev_id, commit_seq=commit_seq,
                    wake_kind=WakeKind.NONE,
                    priority=False,
                )
            if self_echo_delivery_key is None:
                return IngestResult(
                    row_id=row_id, inserted=False,
                    echo_status=EchoStatus.UNPROVEN,
                )
            status = self._reconcile_self_echo(conn, msg, self_echo_delivery_key)
            return IngestResult(row_id=row_id, inserted=False, echo_status=status)

        return await self._db.write(fn)

    def _insert_commit(
        self,
        conn: Any,
        msg: Message,
        row_id: MessageRowId,
        event_id: EventId | None,
        wake_kind: str,
        pending_count: int | None,
        priority: bool,
    ) -> tuple[EventId, CommitSeq]:
        """Create the message's ``inbound_commits`` row in the caller's
        transaction. ``event_id`` is the stable id generated before
        recording; a caller that passes none gets a generated one (legacy
        direct-repo callers). ``committed_ts`` is the message's own
        timestamp (the repository never produces timestamps); a message
        without one records 0.0."""
        if event_id is None:
            event_id = EventId(uuid.uuid4().hex)
        cur = conn.execute(
            "INSERT INTO inbound_commits(event_id, chat_key, message_id,"
            " committed_ts, wake_kind, pending_count, priority)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                msg.chat_key,
                row_id,
                msg.recv_ts if msg.recv_ts is not None else 0.0,
                wake_kind,
                pending_count,
                int(priority),
            ),
        )
        return event_id, CommitSeq(cur.lastrowid)

    def _pending_count(self, conn: Any, chat_key: ChatKey) -> int:
        """The chat's CURRENT pending non-self count: non-self messages
        beyond the durable cursor (NULL cursor counts as 0) — the same
        predicate as ``list_pending_chats``, computed in the caller's
        transaction so the count is atomic with the insert that precedes
        it."""
        row = conn.execute(
            "SELECT COUNT(*) FROM messages"
            " WHERE chat_key = ? AND id > COALESCE("
            "  (SELECT cursor_msg_id FROM chats WHERE chat_key = ?), 0)"
            " AND is_self = 0",
            (chat_key, chat_key),
        ).fetchone()
        return int(row[0])

    def _reconcile_self_echo(
        self, conn: Any, msg: Message, delivery_key: str
    ) -> str:
        """Reconcile one verified self echo against the trusted delivery
        key. Returns the ``EchoStatus``; the outbox moves ONLY on a full
        match against exactly one in-flight row."""
        row = conn.execute(
            "SELECT id, chat_key, text, segments_json, reply_to, state"
            " FROM outbox WHERE idem_key = ?",
            (delivery_key,),
        ).fetchone()
        if row is None:
            # A trusted key that matches no row is a mismatch, not a proof.
            return EchoStatus.CONFLICT
        item_id, chat_key, text, segments_json, reply_to, state = row
        if not self._self_echo_matches(conn, msg, chat_key, text, segments_json, reply_to):
            return EchoStatus.CONFLICT
        if state == "sent":
            # Duplicate echo event: the send was already reconciled.
            return EchoStatus.ALREADY_RECONCILED
        if state != "in_flight":
            # pending must go through attempt_outbox first; dropped is
            # terminal. Never transitioned from here.
            return EchoStatus.CONFLICT
        if msg.id is None:
            # No real platform id: the send cannot be proven to have landed.
            return EchoStatus.CONFLICT
        cur = conn.execute(
            "UPDATE outbox SET state = 'sent', sent_ts = ?, platform_msg_id = ?"
            " WHERE id = ? AND state = 'in_flight'",
            (msg.recv_ts, msg.id, item_id),
        )
        if cur.rowcount != 1:
            # Ambiguity: never transition the outbox.
            return EchoStatus.CONFLICT
        return EchoStatus.RECONCILED

    def _reconcile_synthetic_echo(
        self, conn: Any, msg: Message, delivery_key: str
    ) -> MessageRowId | None:
        """Reconcile a trusted self echo against a synthetic local fallback
        echo.

        ``mark_outbox_sent`` writes a synthetic self echo with the local id
        ``local:<item_id>`` when the platform returned no message id, and
        leaves the outbox row ``sent`` with a NULL ``platform_msg_id``. When
        the real platform echo later arrives with its actual id, this
        updates that synthetic durable row (and the outbox row) to the real
        platform id instead of inserting a duplicate context message.

        Returns the updated synthetic row's id when reconciled; None when
        not applicable (no matching sent-with-fallback row, a payload
        mismatch, or no real platform id) — the caller then falls through to
        the normal insert/dedupe path. Fail-closed: never merges a different
        payload, never touches a row that was sent with a real id, and never
        invents a match without the trusted key.
        """
        row = conn.execute(
            "SELECT id, chat_key, text, segments_json, reply_to, state,"
            " platform_msg_id FROM outbox WHERE idem_key = ?",
            (delivery_key,),
        ).fetchone()
        if row is None:
            return None
        item_id, chat_key, text, segments_json, reply_to, state, outbox_pid = row
        # Only a sent row with NO real platform id (the fallback local echo
        # path) is eligible. A row sent with a real id dedupes on the UNIQUE
        # constraint instead; pending/in_flight/dropped go through the
        # normal reconciliation flow.
        if state != "sent" or outbox_pid is not None:
            return None
        if not self._self_echo_matches(conn, msg, chat_key, text, segments_json, reply_to):
            return None
        if msg.id is None:
            # No real platform id: nothing to reconcile to.
            return None
        local_id = f"local:{item_id}"
        syn = conn.execute(
            "SELECT id FROM messages"
            " WHERE platform = (SELECT platform FROM chats WHERE chat_key = ?)"
            "   AND self_id = (SELECT self_id FROM chats WHERE chat_key = ?)"
            "   AND platform_msg_id = ?",
            (chat_key, chat_key, local_id),
        ).fetchone()
        if syn is None:
            # No synthetic row to update: fail closed, fall through.
            return None
        syn_row_id = MessageRowId(syn[0])
        cur = conn.execute(
            "UPDATE messages SET platform_msg_id = ?, recv_ts = ? WHERE id = ?",
            (msg.id, msg.recv_ts, syn_row_id),
        )
        if cur.rowcount != 1:
            return None
        # Record the real platform id on the outbox row for consistency.
        conn.execute(
            "UPDATE outbox SET platform_msg_id = ? WHERE id = ?",
            (msg.id, item_id),
        )
        return syn_row_id

    def _self_echo_matches(
        self,
        conn: Any,
        msg: Message,
        chat_key: ChatKey,
        text: str,
        segments_json: str,
        reply_to: MessageId | None,
    ) -> bool:
        """The full verification: sender is the chat's self id, and the
        chat/text/canonical segment/reply payload match the outbox row
        exactly."""
        chat = conn.execute(
            "SELECT self_id FROM chats WHERE chat_key = ?", (msg.chat_key,)
        ).fetchone()
        if chat is None or chat[0] != msg.sender_id:
            return False
        return (
            msg.chat_key == chat_key
            and msg.text == text
            and self._serialize_segments(msg.segments) == segments_json
            and msg.reply_to == reply_to
        )

    def _insert_message(self, conn: Any, msg: Message) -> tuple[MessageRowId, bool]:
        row = conn.execute(
            "SELECT platform, self_id FROM chats WHERE chat_key = ?",
            (msg.chat_key,),
        ).fetchone()
        if row is None:
            raise RepoError(
                f"cannot store message: chat {msg.chat_key!r} has no identity"
                " (upsert_chat first)"
            )
        platform, self_id = row
        tombstone = (
            conn.execute(
                "SELECT 1 FROM message_deletions WHERE chat_key = ?"
                " AND platform_msg_id = ?",
                (msg.chat_key, msg.id),
            ).fetchone()
            if msg.id is not None
            else None
        )
        cur = conn.execute(
            "INSERT INTO messages(chat_key, platform, self_id, platform_msg_id,"
            " sender_id, sender_name, is_self, text, segments_json, reply_to,"
            " mentions_json, recv_ts, deleted)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(platform, self_id, platform_msg_id) DO NOTHING",
            (
                msg.chat_key,
                platform,
                self_id,
                msg.id,
                msg.sender_id,
                msg.sender_name,
                1 if msg.is_self else 0,
                msg.text,
                self._serialize_segments(msg.segments),
                msg.reply_to,
                orjson.dumps(list(msg.mentions), default=str).decode("utf-8"),
                msg.recv_ts,
                1 if tombstone is not None else 0,
            ),
        )
        if cur.rowcount == 1:
            row_id = MessageRowId(cur.lastrowid)
            self._fts_insert(conn, row_id, msg.text)
            return row_id, True
        # Duplicate: return the existing durable row id, insert nothing.
        row = conn.execute(
            "SELECT id FROM messages WHERE platform = ? AND self_id = ?"
            " AND platform_msg_id = ?",
            (platform, self_id, msg.id),
        ).fetchone()
        return MessageRowId(row[0]), False

    def _update_avg_interval(
        self, conn: Any, msg: Message, row_id: MessageRowId
    ) -> None:
        """Atomically fold one newly inserted non-self message into the
        chat's durable EWMA interval.

        Uses the SAME dependency-neutral ``pacing.ewma_interval`` reducer
        as the runtime session layer, over prior durable data: the chat's
        stored ``avg_interval`` and the previous non-self message's
        ``recv_ts`` (the newest non-self row strictly before this one —
        self messages never participate, so they can never spuriously
        change the average). A missing prior sample or a non-positive gap
        (clock skew, same-timestamp batches) carries no pacing information
        and leaves the average untouched. Only ``avg_interval`` is
        written: the cursor, the hold window, and the idle streak stay
        terminal-owned (written only by ``finish_cycle``).
        """
        if msg.recv_ts is None:
            return  # no timestamp: no pacing sample
        row = conn.execute(
            "SELECT avg_interval,"
            " (SELECT MAX(recv_ts) FROM messages WHERE chat_key = ?"
            "  AND is_self = 0 AND id < ?)"
            " FROM chats WHERE chat_key = ?",
            (msg.chat_key, row_id, msg.chat_key),
        ).fetchone()
        if row is None:
            return  # unknown chat: nothing to update
        prev_avg, prev_ts = row
        avg = ewma_interval(prev_avg, prev_ts, msg.recv_ts)
        if avg is None:
            return  # no prior sample or non-positive gap: no change
        conn.execute(
            "UPDATE chats SET avg_interval = ? WHERE chat_key = ?",
            (avg, msg.chat_key),
        )

    def _serialize_segments(self, segments: tuple[Segment, ...]) -> str:
        payloads = []
        for seg in segments:
            if seg.kind == "forward":
                # Forwarded contents must not persist: keep the kind (so the
                # placeholder renders) but drop the content payload.
                payloads.append({"kind": seg.kind, "data": {}, "raw": None})
            else:
                payloads.append({"kind": seg.kind, "data": seg.data, "raw": seg.raw})
        return orjson.dumps(payloads, default=str).decode("utf-8")

    def _fts_insert(self, conn: Any, row_id: MessageRowId, text: str) -> None:
        tokens = bigram_tokenize(text)
        if tokens:
            conn.execute(
                "INSERT INTO message_fts(rowid, text) VALUES (?, ?)",
                (row_id, " ".join(tokens)),
            )

    async def get_message(self, chat_key: ChatKey, msg_id: MessageId) -> Message | None:
        def fn(conn: Any) -> Message | None:
            row = conn.execute(
                "SELECT id, chat_key, platform, self_id, platform_msg_id,"
                " sender_id, sender_name, is_self, text, segments_json, reply_to,"
                " mentions_json, recv_ts, deleted FROM messages"
                " WHERE chat_key = ? AND platform_msg_id = ?",
                (chat_key, msg_id),
            ).fetchone()
            return self._row_to_message(row) if row is not None else None

        return await self._db.read(fn)

    async def mark_message_deleted(
        self, chat_key: ChatKey, msg_id: MessageId, *, now: float
    ) -> MessageRowId | None:
        """Durably record a platform deletion and mark the local source row.

        The tombstone is retained even when the source message has not been
        ingested yet.  A later insert consults it in its own transaction, so a
        recall-before-message race cannot make the source sendable again.
        """
        if not math.isfinite(now):
            raise ValueError("now must be finite")

        def fn(conn: Any) -> MessageRowId | None:
            if conn.execute(
                "SELECT 1 FROM chats WHERE chat_key = ?", (chat_key,)
            ).fetchone() is None:
                return None
            conn.execute(
                "INSERT INTO message_deletions(chat_key, platform_msg_id, deleted_ts)"
                " VALUES (?, ?, ?) ON CONFLICT(chat_key, platform_msg_id) DO UPDATE"
                " SET deleted_ts = excluded.deleted_ts",
                (chat_key, msg_id, now),
            )
            conn.execute(
                "UPDATE messages SET deleted = 1 WHERE chat_key = ?"
                " AND platform_msg_id = ?",
                (chat_key, msg_id),
            )
            row = conn.execute(
                "SELECT id FROM messages WHERE chat_key = ?"
                " AND platform_msg_id = ?",
                (chat_key, msg_id),
            ).fetchone()
            return MessageRowId(row[0]) if row is not None else None

        return await self._db.write(fn)

    async def is_message_deleted(
        self, chat_key: ChatKey, row_id: MessageRowId
    ) -> bool:
        """Read the durable deletion bit for one local source row."""
        def fn(conn: Any) -> bool:
            row = conn.execute(
                "SELECT deleted FROM messages WHERE chat_key = ? AND id = ?",
                (chat_key, row_id),
            ).fetchone()
            return bool(row and row[0])

        return await self._db.read(fn)

    def _row_to_message(self, row: Any) -> Message:
        (
            id_, chat_key, _platform, _self_id, platform_msg_id, sender_id,
            sender_name, is_self, text, segments_json, reply_to, mentions_json,
            recv_ts, _deleted,
        ) = row
        return Message(
            chat_key=ChatKey(chat_key),
            sender_id=SenderId(sender_id),
            sender_name=sender_name,
            is_self=bool(is_self),
            text=text,
            id=MessageId(platform_msg_id) if platform_msg_id is not None else None,
            segments=tuple(
                Segment(**seg) for seg in orjson.loads(segments_json)
            ),
            reply_to=MessageId(reply_to) if reply_to else None,
            mentions=tuple(SenderId(m) for m in orjson.loads(mentions_json)),
            recv_ts=recv_ts,
            row_id=MessageRowId(id_),
        )

    async def get_recent_snapshot(
        self, chat_key: ChatKey, through_row_id: MessageRowId, since_ts: float, limit: int
    ) -> RecentSnapshot:
        """The single claim-bounded recent-message read the gate consumes.

        Returns the LIMITED rendered list (at most ``limit`` rows, newest
        first, self messages included) plus the FULL-window counts (self
        messages included), the self count, and the last non-self
        timestamp — the limited list never changes the counts. The window
        is bounded by ``recv_ts >= since_ts`` and ``id <= through_row_id``;
        rows outside those bounds never appear, even when inserted after
        the snapshot boundary was fixed.
        """

        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

        def fn(conn: Any) -> RecentSnapshot:
            # One read transaction: the rendered list and the full-window
            # counts come from the SAME snapshot, so a concurrent commit
            # can never skew one against the other.
            conn.execute("BEGIN")
            try:
                rows = conn.execute(
                    "SELECT id, chat_key, platform, self_id, platform_msg_id,"
                    " sender_id, sender_name, is_self, text, segments_json,"
                    " reply_to, mentions_json, recv_ts, deleted FROM messages"
                    " WHERE chat_key = ? AND id <= ? AND recv_ts >= ?"
                    " ORDER BY id DESC LIMIT ?",
                    (chat_key, through_row_id, since_ts, limit),
                ).fetchall()
                counts = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(is_self), 0) FROM messages"
                    " WHERE chat_key = ? AND id <= ? AND recv_ts >= ?",
                    (chat_key, through_row_id, since_ts),
                ).fetchone()
                last = conn.execute(
                    "SELECT MAX(recv_ts) FROM messages"
                    " WHERE chat_key = ? AND id <= ? AND recv_ts >= ?"
                    " AND is_self = 0",
                    (chat_key, through_row_id, since_ts),
                ).fetchone()
            finally:
                conn.execute("ROLLBACK")
            return RecentSnapshot(
                chat_key=chat_key,
                messages=tuple(self._row_to_message(r) for r in rows),
                window_count=counts[0],
                self_count=counts[1],
                last_nonself_ts=last[0],
                since_ts=since_ts,
                through_row_id=through_row_id,
            )

        return await self._db.read(fn)

    async def list_pending_chats(self) -> list[ChatKey]:
        """The startup-recovery read: every chat with pending work.

        A chat is pending when it holds at least one NON-SELF message
        beyond its durable cursor (``cursor_msg_id``; NULL counts as 0 —
        a chat that never finished a cycle has everything pending). Self
        messages never make a chat pending: they are the bot's own
        output, not inbound work. The result is deterministic (ordered by
        chat_key) and includes ALL chats needing an immediate scheduler
        wake after a restart.
        """

        def fn(conn: Any) -> list[ChatKey]:
            rows = conn.execute(
                "SELECT chat_key FROM chats"
                " WHERE EXISTS ("
                "  SELECT 1 FROM messages"
                "  WHERE messages.chat_key = chats.chat_key"
                "   AND messages.id > COALESCE(chats.cursor_msg_id, 0)"
                "   AND messages.is_self = 0)"
                " ORDER BY chat_key",
            ).fetchall()
            return [ChatKey(r[0]) for r in rows]

        return await self._db.read(fn)

    # ── cycles: claim (bounded grant), renew, release, atomic finish ────────

    async def claim_cycle(self, claim: CycleClaim) -> ClaimGrant | ClaimBusy | None:
        """Compare-and-swap returning the bounded pending data.

        ``ClaimGrant`` when the claim succeeded; ``ClaimBusy`` (with the
        active owner's exact ``busy_until``) when the chat is already
        claimed by a live, unexpired cycle; None when the chat is unknown.
        An expired pre-send claim is recovered: marked ``expired`` and
        replaced. The grant's boundary is fixed at claim time:
        ``start_msg_id`` is the chat cursor, ``through_msg_id`` the chat's
        max message row id; arrivals after the claim stay pending for the
        next claim. Pending excludes ``is_self``.
        """

        def fn(conn: Any) -> ClaimGrant | ClaimBusy | None:
            # Defense in depth: the type validates finiteness at
            # construction; never store a non-finite lease.
            if not (math.isfinite(claim.started_ts) and math.isfinite(claim.expires_at)):
                return None
            row = conn.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?",
                (claim.chat_key,),
            ).fetchone()
            if row is None:
                return None  # unknown chat: nothing to claim
            start = row[0] if row[0] is not None else 0
            live = conn.execute(
                "SELECT id, cycle_id, expires_at FROM claims"
                " WHERE chat_key = ? AND state = 'live'",
                (claim.chat_key,),
            ).fetchone()
            if live is not None:
                live_expires = live[2]
                # Recovery uses the same <= expiry semantics as finish and
                # renew; a non-finite stored lease is treated as expired.
                if not math.isfinite(live_expires) or live_expires <= claim.started_ts:
                    conn.execute(
                        "UPDATE claims SET state = 'expired' WHERE id = ?",
                        (live[0],),
                    )
                else:
                    # Live, unexpired owner: report the exact busy_until —
                    # never a raw claim row.
                    return ClaimBusy(
                        chat_key=claim.chat_key,
                        cycle_id=CycleId(live[1]),
                        busy_until=live_expires,
                    )
            through = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM messages WHERE chat_key = ?",
                (claim.chat_key,),
            ).fetchone()[0]
            cur = conn.execute(
                "INSERT INTO claims(chat_key, cycle_id, started_ts, expires_at,"
                " start_msg_id, through_msg_id, state)"
                " VALUES (?, ?, ?, ?, ?, ?, 'live')"
                " ON CONFLICT(chat_key) WHERE state = 'live' DO NOTHING",
                (
                    claim.chat_key,
                    claim.cycle_id,
                    claim.started_ts,
                    claim.expires_at,
                    start,
                    through,
                ),
            )
            if cur.rowcount != 1:
                return None
            pending = self._pending_messages(conn, claim.chat_key, start, through)
            return ClaimGrant(
                claim=claim,
                start_msg_id=MessageRowId(start),
                through_msg_id=MessageRowId(through),
                pending=tuple(pending),
            )

        return await self._db.write(fn)

    def _pending_messages(
        self, conn: Any, chat_key: ChatKey, start: int, through: int
    ) -> list[Message]:
        rows = conn.execute(
            "SELECT id, chat_key, platform, self_id, platform_msg_id,"
            " sender_id, sender_name, is_self, text, segments_json, reply_to,"
            " mentions_json, recv_ts, deleted FROM messages"
            " WHERE chat_key = ? AND id > ? AND id <= ? AND is_self = 0"
            " ORDER BY id",
            (chat_key, start, through),
        ).fetchall()
        return [self._row_to_message(r) for r in rows]

    async def renew_cycle(
        self, chat_key: ChatKey, cycle_id: CycleId, expires_at: float, *, now: float
    ) -> bool:
        """Extend the lease. False when the claim is not live, not ours, or
        already expired — an expired owner cannot renew even before another
        claimant acts."""

        def fn(conn: Any) -> bool:
            # The new lease must be finite and strictly in the future.
            if not math.isfinite(expires_at) or expires_at <= now:
                return False
            row = conn.execute(
                "SELECT expires_at FROM claims"
                " WHERE chat_key = ? AND cycle_id = ? AND state = 'live'",
                (chat_key, cycle_id),
            ).fetchone()
            # Not ours, or the current lease is already expired (or
            # non-finite): an expired owner cannot renew even before
            # another claimant acts.
            if row is None or not math.isfinite(row[0]) or row[0] <= now:
                return False
            cur = conn.execute(
                "UPDATE claims SET expires_at = ?"
                " WHERE chat_key = ? AND cycle_id = ? AND state = 'live'",
                (expires_at, chat_key, cycle_id),
            )
            return cur.rowcount == 1

        return await self._db.write(fn)

    async def release_cycle(self, chat_key: ChatKey, cycle_id: CycleId) -> None:
        """Give the claim back WITHOUT moving the cursor: the pending
        messages stay pending for the next claim."""

        def fn(conn: Any) -> None:
            conn.execute(
                "UPDATE claims SET state = 'released'"
                " WHERE chat_key = ? AND cycle_id = ? AND state = 'live'",
                (chat_key, cycle_id),
            )

        await self._db.write(fn)

    async def finish_cycle(
        self, finish: CycleFinish, outbox: list[OutboxItem], *, now: float
    ) -> None:
        """Atomically: fence ownership, persist the terminal cycle, insert
        the cycle's complete ordered outbox batch (with cycle provenance,
        rejecting cross-chat items), advance the cursor to the claim's
        stored through boundary, materialize the durable hold window and
        idle streak, and release the claim — one transaction.

        The hold window (``finish.hold_until``; None CLEARS it — the
        terminal reset) and the idle streak (``finish.idle_streak_after``;
        0 resets it) are written in the SAME transaction as the cursor
        advance, so a later ``save_session`` can never reintroduce a crash
        gap for them.

        Fences, in order: the claim is live and owned by ``finish.cycle_id``;
        the lease is unexpired (``expires_at > now`` — an expired owner
        cannot finish even before another claimant acts); the chat cursor
        still equals the claim's start boundary. Any fence failure raises
        ClaimError and changes nothing (the writer isolates it in its own
        savepoint, so unrelated batch work still commits).
        """

        def fn(conn: Any) -> None:
            row = conn.execute(
                "SELECT id, cycle_id, started_ts, expires_at, start_msg_id,"
                " through_msg_id FROM claims"
                " WHERE chat_key = ? AND state = 'live'",
                (finish.chat_key,),
            ).fetchone()
            if row is None or row[1] != finish.cycle_id:
                raise ClaimError(
                    f"stale finalization: no live claim owned by cycle"
                    f" {finish.cycle_id!r} on {finish.chat_key}"
                )
            claim_id, _cycle_id, started_ts, expires_at, start_msg_id, through = row
            # Same <= expiry semantics as renew and recovery; a non-finite
            # stored lease is treated as expired.
            if not math.isfinite(expires_at) or expires_at <= now:
                raise ClaimError(
                    f"claim lease expired: cannot finish cycle"
                    f" {finish.cycle_id!r} on {finish.chat_key}"
                )
            cur = conn.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?",
                (finish.chat_key,),
            ).fetchone()
            chat_cursor = cur[0] if cur is not None and cur[0] is not None else 0
            if chat_cursor != start_msg_id:
                raise ClaimError(
                    f"chat cursor moved past the claim's start boundary:"
                    f" cursor {chat_cursor} != claim start {start_msg_id}"
                )
            cur = conn.execute(
                "INSERT INTO cycles(chat_key, started_ts, end_reason, trace_json,"
                " tokens_in, tokens_out) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    finish.chat_key,
                    started_ts,
                    finish.end_reason,
                    finish.trace_json,
                    finish.tokens_in,
                    finish.tokens_out,
                ),
            )
            cycle_row_id = cur.lastrowid
            self._insert_outbox_batch(conn, outbox, finish.chat_key, cycle_row_id)
            # The cursor advances ONLY here, to the claim's through
            # boundary; the durable hold window and idle streak are
            # materialized in the SAME transaction (hold_until=None clears
            # the hold, idle_streak_after=0 resets the streak — the
            # terminal reset), and the agent barrier is cleared
            # (agent_resume_at=NULL) with the wait streak reset
            # (wait_streak=0).
            conn.execute(
                "UPDATE chats SET cursor_msg_id = ?, hold_until = ?,"
                " idle_streak = ?, agent_resume_at = NULL, wait_streak = 0"
                " WHERE chat_key = ?",
                (through, finish.hold_until, finish.idle_streak_after, finish.chat_key),
            )
            conn.execute(
                "UPDATE claims SET state = 'finished' WHERE id = ?", (claim_id,)
            )

        await self._db.write(fn)

    async def get_latest_terminal_end_reason(self, chat_key: ChatKey) -> str | None:
        """The per-chat latest TERMINAL cycle end reason — the gate's only
        history input (``GateSnapshot.previous_end_reason``). Reads the
        ``cycles`` table, which holds ONLY terminal outcomes written by
        ``finish_cycle``; released/expired claims never affect it. None
        when the chat has no terminal cycle yet."""

        def fn(conn: Any) -> str | None:
            row = conn.execute(
                "SELECT end_reason FROM cycles WHERE chat_key = ?"
                " ORDER BY id DESC LIMIT 1",
                (chat_key,),
            ).fetchone()
            return row[0] if row is not None else None

        return await self._db.read(fn)

    # ── durable dispatch ledger (frozen Oracle advisory) ────────────────────
    # The minimal serialized dispatch ledger owned by Repository and driven
    # by Scheduler. ``begin_dispatch`` atomically claims (a prepared
    # dispatch), freezes the commit boundary, attaches eligible unassigned
    # commits, and records the dispatch; ``settle_dispatch`` owns ALL
    # release/delay/terminal movement — the cursor and the outbox move ONLY
    # inside a terminal finish. The at-least-once export surface lists
    # unexported commit/dispatch markers and marks one exported; the
    # startup export appends markers then marks them exported, and readers
    # deduplicate by (record_type, sequence). No SQL or raw rows cross the
    # seam — callers see only the typed boundary types.
    #
    # The legacy claim_cycle/finish_cycle surface remains for compatibility
    # with the current cycle lane; the next integration lane switches all
    # live use to this ledger.

    async def begin_dispatch(
        self, request: DispatchRequest
    ) -> DispatchGrant | ClaimBusy | DispatchDeferred | None:
        """Atomically claim, freeze the boundary, attach, and record.

        One transaction: recover any expired prepared dispatch, freeze the
        commit boundary (the max ``inbound_commits`` sequence at this
        moment — durable writer order resolves timer/inbound ties: a
        commit that wrote first joins this dispatch, a timer dispatch that
        wrote first excludes it), attach every eligible unassigned commit
        (``wake_kind`` != ``none``, no dispatch membership, within the
        boundary), and insert the prepared dispatch row.

        ``DispatchGrant`` when the dispatch was created (``attached`` may
        be empty for a priority wake — ``timer``/``startup``/
        ``busy_recovery`` always create a dispatch, the wake itself is the
        work); ``ClaimBusy`` (with the active owner's exact ``busy_until``)
        when the chat already has a live, unexpired prepared dispatch;
        ``DispatchDeferred`` when the chat's durable agent barrier is still
        active (``agent_resume_at > now`` — no dispatch is created or
        attached, and the scheduler re-arms at ``resume_at``); None when
        the chat is unknown or an ``inbound`` dispatch found no eligible
        commits (no work). An expired prepared dispatch is recovered:
        marked ``expired``, its attached commits detached
        (``dispatch_id = NULL``) in the SAME transaction so they stay
        eligible for the fresh dispatch, and replaced. An EXPIRED agent
        barrier (``agent_resume_at <= now``) is cleared in the same
        transaction and the dispatch is granted normally.
        """

        def fn(conn: Any) -> DispatchGrant | ClaimBusy | DispatchDeferred | None:
            # Defense in depth: the type validates finiteness at
            # construction; never store a non-finite lease.
            if not (
                math.isfinite(request.started_ts)
                and math.isfinite(request.expires_at)
                and math.isfinite(request.now)
            ):
                return None
            row = conn.execute(
                "SELECT cursor_msg_id, agent_resume_at FROM chats"
                " WHERE chat_key = ?",
                (request.chat_key,),
            ).fetchone()
            if row is None:
                return None  # unknown chat: nothing to dispatch
            start = row[0] if row[0] is not None else 0
            resume_at = row[1]
            if resume_at is not None and resume_at > request.now:
                # The durable agent barrier is still active: defer instead
                # of creating/attaching. The scheduler re-arms at resume_at;
                # no agent runs early. The barrier's origin kind is not
                # stored, so a barrier that blocks a fresh wake reads back
                # as a retry barrier (the scheduler only reads resume_at).
                return DispatchDeferred(
                    chat_key=request.chat_key,
                    resume_at=resume_at,
                    defer_kind="retry",
                )
            if resume_at is not None:
                # The barrier expired: clear it and grant normally.
                conn.execute(
                    "UPDATE chats SET agent_resume_at = NULL"
                    " WHERE chat_key = ?",
                    (request.chat_key,),
                )
            live = conn.execute(
                "SELECT id, cycle_id, expires_at FROM dispatches"
                " WHERE chat_key = ? AND state = 'prepared'",
                (request.chat_key,),
            ).fetchone()
            if live is not None:
                live_expires = live[2]
                # Recovery uses the same <= expiry semantics as settlement;
                # a non-finite stored lease is treated as expired.
                if not math.isfinite(live_expires) or live_expires <= request.now:
                    conn.execute(
                        "UPDATE dispatches SET state = 'expired' WHERE id = ?",
                        (live[0],),
                    )
                    # Detach the expired dispatch's attached commits in the
                    # SAME transaction: otherwise the next begin_dispatch
                    # cannot attach them (they still carry the expired
                    # dispatch's id) and they strand after a crash. Mirrors
                    # the release/delay detach in settle_dispatch.
                    conn.execute(
                        "UPDATE inbound_commits SET dispatch_id = NULL"
                        " WHERE dispatch_id = ?",
                        (live[0],),
                    )
                else:
                    # Live, unexpired owner: report the exact busy_until —
                    # never a raw dispatch row.
                    return ClaimBusy(
                        chat_key=request.chat_key,
                        cycle_id=CycleId(live[1]),
                        busy_until=live_expires,
                    )
            # The frozen boundary: the max commit sequence at this moment.
            boundary = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM inbound_commits"
            ).fetchone()[0]
            attached_rows = conn.execute(
                "SELECT ic.id, ic.message_id, ic.priority FROM inbound_commits AS ic"
                " JOIN messages AS m ON m.id = ic.message_id"
                " WHERE ic.chat_key = ? AND ic.dispatch_id IS NULL"
                " AND ic.wake_kind != ? AND ic.id <= ? AND m.id > ?"
                " ORDER BY ic.id",
                (request.chat_key, WakeKind.NONE, boundary, start),
            ).fetchall()
            attached = [CommitSeq(r[0]) for r in attached_rows]
            if request.cause == DispatchCause.INBOUND and not attached:
                return None  # no work: nothing eligible to attach
            # Pending attachment remains non-self/wakeable, but the claim
            # boundary must include every committed message in writer order —
            # especially a trailing self echo used by presence/context.
            through = conn.execute(
                "SELECT COALESCE(MAX(message_id), ?) FROM inbound_commits"
                " WHERE chat_key = ? AND id <= ? AND message_id > ?",
                (start, request.chat_key, boundary, start),
            ).fetchone()[0]
            # The exact attached membership is frozen in the SAME
            # transaction that creates the prepared dispatch: a later
            # released or expired-detached dispatch stays replayable with
            # its exact membership even after the live
            # ``inbound_commits.dispatch_id`` rows are detached.
            attached_json = orjson.dumps(attached).decode("utf-8")
            cur = conn.execute(
                "INSERT INTO dispatches(chat_key, cause, wake_kind, scheduled_ts,"
                " started_ts, expires_at, claimed_ts, cycle_id, start_msg_id,"
                " through_msg_id, commit_boundary, attached_json, state)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared')"
                " ON CONFLICT(chat_key) WHERE state = 'prepared' DO NOTHING",
                (
                    request.chat_key,
                    request.cause,
                    request.wake_kind if request.wake_kind is not None else request.cause,
                    request.scheduled_ts,
                    request.started_ts,
                    request.expires_at,
                    request.now,
                    request.cycle_id,
                    start,
                    through,
                    boundary,
                    attached_json,
                ),
            )
            if cur.rowcount != 1:
                return None  # a prepared dispatch appeared mid-transaction
            dispatch_id = DispatchId(cur.lastrowid)
            if attached:
                conn.execute(
                    "UPDATE inbound_commits SET dispatch_id = ? WHERE id IN ("
                    + self._in_clause(len(attached))
                    + ")",
                    (dispatch_id, *attached),
                )
            pending = self._messages_for_commits(conn, attached)
            return DispatchGrant(
                dispatch_id=dispatch_id,
                claim=CycleClaim(
                    request.chat_key,
                    request.cycle_id,
                    request.started_ts,
                    request.expires_at,
                ),
                start_msg_id=MessageRowId(start),
                through_msg_id=MessageRowId(through),
                attached=tuple(attached),
                pending=tuple(pending),
                commit_boundary=CommitSeq(boundary),
                scheduled_for=request.scheduled_ts,
                cause=request.cause,
                claimed_ts=request.now,
                priority=any(bool(r[2]) for r in attached_rows),
            )

        return await self._db.write(fn)

    async def renew_dispatch(
        self,
        chat_key: ChatKey,
        dispatch_id: DispatchId,
        cycle_id: CycleId,
        expires_at: float,
        *,
        now: float,
    ) -> bool:
        """Extend a prepared dispatch's lease.

        Fenced to the SAME unexpired prepared owner: False when the
        dispatch is not prepared, not owned by ``cycle_id``, or already
        expired — an expired owner cannot renew even before another
        claimant acts. The new lease must be finite and strictly in the
        future (a finite forward extension).
        """

        def fn(conn: Any) -> bool:
            # The new lease must be finite and strictly in the future.
            if not math.isfinite(expires_at) or expires_at <= now:
                return False
            row = conn.execute(
                "SELECT expires_at FROM dispatches"
                " WHERE chat_key = ? AND id = ? AND cycle_id = ?"
                " AND state = 'prepared'",
                (chat_key, dispatch_id, cycle_id),
            ).fetchone()
            # Not ours, or the current lease is already expired (or
            # non-finite): an expired owner cannot renew even before
            # another claimant acts.
            if row is None or not math.isfinite(row[0]) or row[0] <= now:
                return False
            cur = conn.execute(
                "UPDATE dispatches SET expires_at = ?"
                " WHERE chat_key = ? AND id = ? AND cycle_id = ?"
                " AND state = 'prepared'",
                (expires_at, chat_key, dispatch_id, cycle_id),
            )
            return cur.rowcount == 1

        return await self._db.write(fn)

    async def settle_dispatch(
        self, settle: DispatchSettle, outbox: list[OutboxItem], *, now: float
    ) -> None:
        """Atomically settle one prepared dispatch.

        ``release`` and ``delay`` give the claim back WITHOUT moving the
        cursor or touching the outbox (ordinary delay and active hold
        release claims without cursor/session change); the delay trace is
        recorded on the released dispatch row. ``defer`` gives the claim
        back, detaches the attached commits, and records the durable agent
        barrier (``agent_resume_at``) — a wait defer additionally
        increments the wait streak, a retry defer does not. ``finish`` is
        TERMINAL: it fences ownership (prepared dispatch owned by
        ``settle.cycle_id``, unexpired lease, chat cursor still at the
        dispatch's start boundary), persists the terminal cycle, inserts
        the cycle's complete ordered outbox batch (with cycle provenance,
        rejecting cross-chat items), advances the cursor to the dispatch's
        stored through boundary, materializes the durable hold window and
        idle streak, CLEARS the agent barrier and resets the wait streak,
        and marks the dispatch completed with its trace — one transaction.
        Any fence failure raises ClaimError and changes nothing (the writer
        isolates it in its own savepoint, so unrelated batch work still
        commits).
        """

        def fn(conn: Any) -> None:
            row = conn.execute(
                "SELECT id, cycle_id, started_ts, expires_at, start_msg_id,"
                " through_msg_id, state FROM dispatches"
                " WHERE chat_key = ? AND id = ?",
                (settle.chat_key, settle.dispatch_id),
            ).fetchone()
            if row is None or row[1] != settle.cycle_id or row[6] != "prepared":
                raise ClaimError(
                    f"stale settlement: no prepared dispatch"
                    f" {settle.dispatch_id!r} owned by cycle {settle.cycle_id!r}"
                    f" on {settle.chat_key}"
                )
            _id, _cycle_id, started_ts, expires_at, start_msg_id, through, _state = row
            # Same <= expiry semantics as begin_dispatch recovery; a
            # non-finite stored lease is treated as expired.
            if not math.isfinite(expires_at) or expires_at <= now:
                raise ClaimError(
                    f"dispatch lease expired: cannot settle dispatch"
                    f" {settle.dispatch_id!r} on {settle.chat_key}"
                )
            if settle.outcome in ("release", "delay"):
                # No cursor/outbox movement outside terminal settlement.
                # The attached commits are DETACHED so they stay pending
                # for the next dispatch (release/delay give the claim back
                # without consuming the boundary). The frozen
                # ``attached_json`` membership is PRESERVED on the row, so
                # the released dispatch remains replayable with its exact
                # membership; ``settled_ts`` records the settlement time.
                conn.execute(
                    "UPDATE dispatches SET state = 'released', trace_json = ?,"
                    " settled_ts = ?, evaluated_ts = ?, snapshot_json = ?"
                    " WHERE id = ? AND state = 'prepared'",
                    (
                        settle.trace_json,
                        now,
                        settle.evaluated_ts if settle.evaluated_ts is not None else now,
                        settle.snapshot_json,
                        settle.dispatch_id,
                    ),
                )
                conn.execute(
                    "UPDATE inbound_commits SET dispatch_id = NULL"
                    " WHERE dispatch_id = ?",
                    (settle.dispatch_id,),
                )
                return
            if settle.outcome == "defer":
                # Give the claim back WITHOUT cursor/outbox movement: the
                # attached commits are DETACHED (they stay pending for the
                # next dispatch) and the durable agent barrier is recorded.
                # A wait defer additionally increments the wait streak; a
                # retry defer does not. The frozen ``attached_json``
                # membership is preserved on the released row.
                conn.execute(
                    "UPDATE dispatches SET state = 'released', trace_json = ?,"
                    " settled_ts = ?, evaluated_ts = ?, snapshot_json = ?"
                    " WHERE id = ? AND state = 'prepared'",
                    (
                        settle.trace_json,
                        now,
                        settle.evaluated_ts if settle.evaluated_ts is not None else now,
                        settle.snapshot_json,
                        settle.dispatch_id,
                    ),
                )
                conn.execute(
                    "UPDATE inbound_commits SET dispatch_id = NULL"
                    " WHERE dispatch_id = ?",
                    (settle.dispatch_id,),
                )
                if settle.defer_kind == "wait":
                    conn.execute(
                        "UPDATE chats SET agent_resume_at = ?,"
                        " wait_streak = wait_streak + 1 WHERE chat_key = ?",
                        (settle.resume_at, settle.chat_key),
                    )
                else:
                    conn.execute(
                        "UPDATE chats SET agent_resume_at = ? WHERE chat_key = ?",
                        (settle.resume_at, settle.chat_key),
                    )
                return
            cur = conn.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?",
                (settle.chat_key,),
            ).fetchone()
            chat_cursor = cur[0] if cur is not None and cur[0] is not None else 0
            if chat_cursor != start_msg_id:
                raise ClaimError(
                    f"chat cursor moved past the dispatch's start boundary:"
                    f" cursor {chat_cursor} != dispatch start {start_msg_id}"
                )
            cur = conn.execute(
                "INSERT INTO cycles(chat_key, started_ts, end_reason, trace_json,"
                " tokens_in, tokens_out) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    settle.chat_key,
                    started_ts,
                    settle.end_reason,
                    settle.trace_json,
                    settle.tokens_in,
                    settle.tokens_out,
                ),
            )
            cycle_row_id = cur.lastrowid
            self._insert_outbox_batch(conn, outbox, settle.chat_key, cycle_row_id)
            # Chat controls are part of the terminal settlement transaction,
            # not a best-effort follow-up after the cursor has moved.
            for control in settle.chat_controls:
                self._apply_chat_control_conn(conn, control)
            # The cursor advances ONLY here, to the dispatch's through
            # boundary; the durable hold window and idle streak are
            # materialized in the SAME transaction (hold_until=None clears
            # the hold, idle_streak_after=0 resets the streak), and the
            # agent barrier is cleared (agent_resume_at=NULL) with the wait
            # streak reset (wait_streak=0) — the terminal finish.
            conn.execute(
                "UPDATE chats SET cursor_msg_id = ?, hold_until = ?,"
                " idle_streak = ?, agent_resume_at = NULL, wait_streak = 0"
                " WHERE chat_key = ?",
                (through, settle.hold_until, settle.idle_streak_after, settle.chat_key),
            )
            conn.execute(
                "UPDATE dispatches SET state = 'completed', trace_json = ?,"
                " settled_ts = ?, evaluated_ts = ?, snapshot_json = ?"
                " WHERE id = ? AND state = 'prepared'",
                (
                    settle.trace_json,
                    now,
                    settle.evaluated_ts if settle.evaluated_ts is not None else now,
                    settle.snapshot_json,
                    settle.dispatch_id,
                ),
            )

        await self._db.write(fn)

    async def list_unexported_commits(self) -> list[CorpusMarker]:
        """Every ``inbound_commits`` row whose commit marker was not yet
        exported, in sequence order — the startup export's input."""

        def fn(conn: Any) -> list[CorpusMarker]:
            rows = conn.execute(
                "SELECT id, event_id, chat_key, wake_kind, message_id, priority"
                " FROM inbound_commits WHERE exported = 0 ORDER BY id",
            ).fetchall()
            return [
                CorpusMarker(
                    record_type="commit",
                    sequence=CommitSeq(r[0]),
                    chat_key=ChatKey(r[2]),
                    event_id=EventId(r[1]),
                    wake_kind=r[3],
                    message_row_id=MessageRowId(r[4]),
                    priority=bool(r[5]),
                )
                for r in rows
            ]

        return await self._db.read(fn)

    async def list_unexported_dispatches(self) -> list[CorpusMarker]:
        """Every SETTLED ``dispatches`` row whose dispatch marker was not
        yet exported, in dispatch order — the startup export's input.

        Only settled, replayable dispatches (``completed`` or ``released``)
        are ever exported: a prepared/unevaluated dispatch must never be
        emitted on startup, and an expired dispatch is not an evaluation
        (it produces no marker). Each marker carries the FULL frozen
        evaluation metadata from the durable row: the settled ``state``,
        the ``settled_ts`` evaluation timestamp, the fixed
        ``start_msg_id``/``through_msg_id`` message boundaries, the exact
        attached ``CommitSeq`` tuple (``attached_json``, frozen at
        ``begin_dispatch``), the persisted ``trace_json``, the ``cause``,
        the frozen ``commit_boundary``, and ``scheduled_for`` — so replay
        reconstructs the exact live dispatch independent of JSONL marker
        order."""

        def fn(conn: Any) -> list[CorpusMarker]:
            rows = conn.execute(
                "SELECT id, chat_key, cause, commit_boundary, scheduled_ts,"
                " state, settled_ts, start_msg_id, through_msg_id,"
                " attached_json, trace_json, evaluated_ts, snapshot_json"
                " FROM dispatches WHERE exported = 0"
                " AND state IN ('completed', 'released') ORDER BY id",
            ).fetchall()
            return [
                CorpusMarker(
                    record_type="dispatch",
                    sequence=DispatchId(r[0]),
                    chat_key=ChatKey(r[1]),
                    cause=r[2],
                    commit_boundary=CommitSeq(r[3]),
                    scheduled_for=r[4],
                    state=r[5],
                    settled_ts=r[6],
                    start_msg_id=MessageRowId(r[7]),
                    through_msg_id=MessageRowId(r[8]),
                    attached=tuple(
                        CommitSeq(s)
                        for s in (orjson.loads(r[9]) if r[9] else [])
                    ),
                    trace_json=r[10],
                    evaluated_ts=r[11],
                    snapshot_json=r[12],
                )
                for r in rows
            ]

        return await self._db.read(fn)

    async def mark_commit_exported(self, commit_seq: CommitSeq) -> None:
        """Mark one commit marker exported (idempotent)."""

        def fn(conn: Any) -> None:
            conn.execute(
                "UPDATE inbound_commits SET exported = 1 WHERE id = ?",
                (commit_seq,),
            )

        await self._db.write(fn)

    async def mark_dispatch_exported(self, dispatch_id: DispatchId) -> None:
        """Mark one dispatch marker exported (idempotent)."""

        def fn(conn: Any) -> None:
            conn.execute(
                "UPDATE dispatches SET exported = 1 WHERE id = ?", (dispatch_id,)
            )

        await self._db.write(fn)

    async def list_unassigned_commits(self, chat_key: ChatKey) -> list[CommitSeq]:
        """The chat's eligible unassigned commit sequences — committed
        events that never joined a dispatch (crash before ``begin_dispatch``
        or a dispatch that excluded them). The scheduler re-arms from this
        scan after a restart."""

        def fn(conn: Any) -> list[CommitSeq]:
            rows = conn.execute(
                "SELECT id FROM inbound_commits WHERE chat_key = ?"
                " AND dispatch_id IS NULL AND wake_kind != ? ORDER BY id",
                (chat_key, WakeKind.NONE),
            ).fetchall()
            return [CommitSeq(r[0]) for r in rows]

        return await self._db.read(fn)

    async def list_ledger_pending_chats(self) -> list[ChatKey]:
        """The ledger's startup-recovery read: every chat with at least one
        eligible unassigned commit (``wake_kind`` != ``none``, no dispatch
        membership). Deterministic order; the scheduler wakes exactly these
        chats immediately after a restart."""

        def fn(conn: Any) -> list[ChatKey]:
            rows = conn.execute(
                "SELECT DISTINCT chat_key FROM inbound_commits"
                " WHERE dispatch_id IS NULL AND wake_kind != ?"
                " UNION"
                " SELECT DISTINCT chat_key FROM dispatches WHERE state = 'prepared'"
                " ORDER BY chat_key",
                (WakeKind.NONE,),
            ).fetchall()
            return [ChatKey(r[0]) for r in rows]

        return await self._db.read(fn)

    @staticmethod
    def _in_clause(n: int) -> str:
        """``?,?,...`` placeholders for an IN list of n values."""
        return ",".join("?" * n)

    def _messages_for_commits(
        self, conn: Any, attached: list[CommitSeq]
    ) -> list[Message]:
        """The messages of the attached commits, in row order (commit
        sequence order equals message insert order)."""
        if not attached:
            return []
        rows = conn.execute(
            "SELECT m.id, m.chat_key, m.platform, m.self_id, m.platform_msg_id,"
            " m.sender_id, m.sender_name, m.is_self, m.text, m.segments_json,"
            " m.reply_to, m.mentions_json, m.recv_ts, m.deleted"
            " FROM messages m JOIN inbound_commits c ON c.message_id = m.id"
            " WHERE c.id IN (" + self._in_clause(len(attached)) + ")"
            " ORDER BY m.id",
            tuple(attached),
        ).fetchall()
        return [self._row_to_message(r) for r in rows]

    # ── outbox: one row per adapter send, created only by finish_cycle ──────

    def _insert_outbox_batch(
        self, conn: Any, items: list[OutboxItem], chat_key: ChatKey, cycle_id: int
    ) -> None:
        for item in items:
            if item.chat_key != chat_key:
                raise RepoError(
                    f"cross-chat outbox item: {item.chat_key!r} does not match"
                    f" claimed chat {chat_key!r}"
                )
            cur = conn.execute(
                "INSERT INTO outbox(chat_key, cycle_id, group_id, seq, text,"
                " segments_json, payload_json, reply_to, state, send_after_ts,"
                " idem_key) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)"
                " ON CONFLICT(idem_key) DO NOTHING",
                (
                    item.chat_key,
                    cycle_id,
                    item.group_id,
                    item.seq,
                    item.text,
                    self._serialize_segments(item.segments),
                    orjson.dumps(item.payload, default=str).decode("utf-8"),
                    item.reply_to,
                    item.send_after_ts,
                    item.idem_key,
                ),
            )
            if cur.rowcount == 0:
                # Colliding idem_key: hydrate identical durable data or
                # reject the conflicting payload — never silently lose a part.
                row = conn.execute(
                    "SELECT chat_key, group_id, seq, text, segments_json,"
                    " payload_json, reply_to, send_after_ts FROM outbox"
                    " WHERE idem_key = ?",
                    (item.idem_key,),
                ).fetchone()
                if not self._same_outbox_payload(row, item):
                    raise RepoError(
                        f"idem_key collision with conflicting payload:"
                        f" {item.idem_key!r}"
                    )

    def _same_outbox_payload(self, row: Any, item: OutboxItem) -> bool:
        (
            chat_key, group_id, seq, text, segments_json, payload_json,
            reply_to, send_after_ts,
        ) = row
        return (
            chat_key == item.chat_key
            and group_id == item.group_id
            and seq == item.seq
            and text == item.text
            and segments_json == self._serialize_segments(item.segments)
            and payload_json == orjson.dumps(item.payload, default=str).decode("utf-8")
            and reply_to == item.reply_to
            and send_after_ts == item.send_after_ts
        )

    async def list_ready_outbox(
        self, chat_key: ChatKey, *, now: float, limit: int = 10
    ) -> list[OutboxItem]:
        """Pending items whose ``send_after_ts`` has passed. In-flight items
        are never listed — after a crash they stay in_flight forever (the
        send outcome is ambiguous; at-most-once means no auto-retry)."""

        def fn(conn: Any) -> list[OutboxItem]:
            rows = conn.execute(
                "SELECT id, chat_key, cycle_id, group_id, seq, text,"
                " segments_json, payload_json, reply_to, state, send_after_ts,"
                " attempt_started_ts, sent_ts, platform_msg_id, idem_key"
                " FROM outbox WHERE chat_key = ? AND state = 'pending'"
                " AND (send_after_ts IS NULL OR send_after_ts <= ?)"
                " ORDER BY id LIMIT ?",
                (chat_key, now, limit),
            ).fetchall()
            return [self._row_to_outbox(row) for row in rows]

        return await self._db.read(fn)

    async def next_due_outbox(self, chat_key: ChatKey, *, now: float) -> float | None:
        """The earliest ``send_after_ts`` among pending rows for the chat.

        Rows without a ``send_after_ts`` (or already due) count as due now,
        so the result is ``<= now`` whenever anything is ready; None when
        nothing is pending. This is the minimal seam the startup outbox
        worker needs to schedule its next wake without polling.
        """

        def fn(conn: Any) -> float | None:
            row = conn.execute(
                "SELECT MIN(CASE WHEN send_after_ts IS NULL OR send_after_ts <= ?"
                " THEN 0.0 ELSE send_after_ts END) FROM outbox"
                " WHERE chat_key = ? AND state = 'pending'",
                (now, chat_key),
            ).fetchone()
            return row[0] if row[0] is not None else None

        return await self._db.read(fn)

    async def dispatch_delivery_confirmed(
        self, chat_key: ChatKey, dispatch_id: int
    ) -> bool:
        """True only when every durable outbox part for a dispatch is sent."""
        prefix = f"dispatch:{dispatch_id}:"

        def fn(conn: Any) -> bool:
            row = conn.execute(
                "SELECT COUNT(*), SUM(state IN ('pending', 'in_flight')),"
                "SUM(state = 'dropped') FROM outbox"
                " WHERE chat_key = ? AND idem_key LIKE ?",
                (chat_key, prefix + "%"),
            ).fetchone()
            return bool(row[0] and not row[1] and not row[2])

        return await self._db.read(fn)

    async def list_outbox_chats(self) -> list[ChatKey]:
        """Every chat with at least one PENDING outbox row — the startup
        recovery read for pre-existing safe pending/future rows across ALL
        chats. In-flight rows are excluded (their send outcome is ambiguous
        and never auto-retried). Deterministic order (by chat_key)."""

        def fn(conn: Any) -> list[ChatKey]:
            rows = conn.execute(
                "SELECT DISTINCT chat_key FROM outbox"
                " WHERE state = 'pending' ORDER BY chat_key"
            ).fetchall()
            return [ChatKey(r[0]) for r in rows]

        return await self._db.read(fn)

    async def attempt_outbox(self, item_id: int, attempt_started_ts: float) -> bool:
        """Durable CAS ``pending -> in_flight``. Must commit BEFORE the
        adapter is invoked; False when the item is no longer pending."""

        def fn(conn: Any) -> bool:
            cur = conn.execute(
                "UPDATE outbox SET state = 'in_flight', attempt_started_ts = ?"
                " WHERE id = ? AND state = 'pending'",
                (attempt_started_ts, item_id),
            )
            return cur.rowcount == 1

        return await self._db.write(fn)

    async def requeue_outbox(self, item_id: int) -> bool:
        """Return an in-flight row to pending ONLY when an adapter proved it
        never started a write. Generic send failures remain in-flight because
        their outcome is ambiguous and at-most-once delivery forbids retry."""

        def fn(conn: Any) -> bool:
            cur = conn.execute(
                "UPDATE outbox SET state = 'pending', attempt_started_ts = NULL"
                " WHERE id = ? AND state = 'in_flight'"
                " AND platform_msg_id IS NULL",
                (item_id,),
            )
            return cur.rowcount == 1

        return await self._db.write(fn)

    async def mark_outbox_sent(
        self, item_id: int, platform_msg_id: MessageId | None, sent_ts: float
    ) -> bool:
        """Transition ONLY ``in_flight -> sent`` and, in the same
        transaction, write the synthetic self echo: the bot's own message
        appears in its context (self-ratio, presence, recent) with the
        platform id the adapter returned — or a local fallback id when the
        platform returned none. A later real echo with the same platform id
        deduplicates on the UNIQUE constraint. False when the item was not
        in_flight (already sent, still pending, or dropped)."""

        def fn(conn: Any) -> bool:
            row = conn.execute(
                "SELECT chat_key, text, segments_json, reply_to FROM outbox"
                " WHERE id = ?",
                (item_id,),
            ).fetchone()
            if row is None:
                raise RepoError(f"outbox item {item_id} does not exist")
            chat_key, text, segments_json, reply_to = row
            cur = conn.execute(
                "UPDATE outbox SET state = 'sent', sent_ts = ?, platform_msg_id = ?"
                " WHERE id = ? AND state = 'in_flight'",
                (sent_ts, platform_msg_id, item_id),
            )
            if cur.rowcount != 1:
                return False
            self._write_echo(
                conn, chat_key, platform_msg_id, text, segments_json, reply_to,
                sent_ts, item_id,
            )
            return True

        return await self._db.write(fn)

    async def drop_outbox(self, item_id: int) -> bool:
        """Transition ONLY ``pending -> dropped``. A staleness drop must
        never rewrite a sent or in_flight row; False when the item was not
        pending."""

        def fn(conn: Any) -> bool:
            cur = conn.execute(
                "UPDATE outbox SET state = 'dropped'"
                " WHERE id = ? AND state = 'pending'",
                (item_id,),
            )
            return cur.rowcount == 1

        return await self._db.write(fn)

    def _write_echo(
        self,
        conn: Any,
        chat_key: ChatKey,
        platform_msg_id: MessageId | None,
        text: str,
        segments_json: str,
        reply_to: MessageId | None,
        sent_ts: float,
        item_id: int,
    ) -> None:
        chat = conn.execute(
            "SELECT platform, self_id FROM chats WHERE chat_key = ?",
            (chat_key,),
        ).fetchone()
        if chat is None:
            raise RepoError(f"cannot write self echo: chat {chat_key!r} has no identity")
        platform, self_id = chat
        echo_id = platform_msg_id if platform_msg_id is not None else f"local:{item_id}"
        cur = conn.execute(
            "INSERT INTO messages(chat_key, platform, self_id, platform_msg_id,"
            " sender_id, sender_name, is_self, text, segments_json, reply_to,"
            " mentions_json, recv_ts, deleted)"
            " VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, '[]', ?, 0)"
            " ON CONFLICT(platform, self_id, platform_msg_id) DO NOTHING",
            (
                chat_key,
                platform,
                self_id,
                echo_id,
                self_id,
                self_id,
                text,
                segments_json,
                reply_to,
                sent_ts,
            ),
        )
        if cur.rowcount == 1:
            self._fts_insert(conn, MessageRowId(cur.lastrowid), text)

    def _row_to_outbox(self, row: Any) -> OutboxItem:
        (
            id_, chat_key, _cycle_id, group_id, seq, text, segments_json,
            payload_json, reply_to, state, send_after_ts, attempt_started_ts,
            sent_ts, platform_msg_id, idem_key,
        ) = row
        return OutboxItem(
            chat_key=ChatKey(chat_key),
            text=text,
            idem_key=idem_key,
            segments=tuple(Segment(**s) for s in orjson.loads(segments_json)),
            payload=orjson.loads(payload_json),
            reply_to=MessageId(reply_to) if reply_to else None,
            group_id=group_id,
            seq=seq,
            state=state,
            send_after_ts=send_after_ts,
            attempt_started_ts=attempt_started_ts,
            sent_ts=sent_ts,
            platform_msg_id=MessageId(platform_msg_id)
            if platform_msg_id is not None
            else None,
            id=id_,
        )

    # ── Phase 5 knowledge foundation (frozen Oracle advisory) ────────────────
    # Durable source-bounded memory, canonical CJK-bigram FTS documents,
    # per-chat person identity with a CAS profile cursor, embedding
    # generations, and chat-scoped vector rows. Knowledge rows/FTS are
    # authoritative local state; vectors are rebuildable derived state. No
    # provider/network calls in any transaction. Everything is strictly
    # chat-scoped; SQL text lives here.

    @staticmethod
    def _source_hash(texts: tuple[str, ...]) -> str:
        """Deterministic hash of a source batch's message texts."""
        h = hashlib.sha256()
        for text in texts:
            h.update(text.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    @staticmethod
    def _person_key(chat_key: ChatKey, platform_uid: SenderId) -> PersonKey:
        """Deterministic per-chat person key derived from (chat_key,
        platform_uid) — no global nickname matching."""
        digest = hashlib.sha256(
            f"{chat_key}\x00{platform_uid}".encode("utf-8")
        ).hexdigest()
        return PersonKey(digest[:24])

    async def get_memory_watermark(self, chat_key: ChatKey) -> MessageRowId | None:
        """The durable per-chat memory watermark (``chats.memory_through_msg_id``);
        None when the chat is unknown or nothing was summarized yet."""

        def fn(conn: Any) -> MessageRowId | None:
            row = conn.execute(
                "SELECT memory_through_msg_id FROM chats WHERE chat_key = ?",
                (chat_key,),
            ).fetchone()
            if row is None or row[0] is None:
                return None
            return MessageRowId(row[0])

        return await self._db.read(fn)

    async def get_memory_observed_watermark(
        self, chat_key: ChatKey
    ) -> MessageRowId | None:
        """The durable observed memory watermark
        (``chats.memory_observed_through_msg_id``): the watermark snapshot
        observed when the last source batch was read, recorded at commit.
        None when the chat is unknown or nothing was committed yet."""

        def fn(conn: Any) -> MessageRowId | None:
            row = conn.execute(
                "SELECT memory_observed_through_msg_id FROM chats WHERE chat_key = ?",
                (chat_key,),
            ).fetchone()
            if row is None or row[0] is None:
                return None
            return MessageRowId(row[0])

        return await self._db.read(fn)

    async def read_memory_source_batch(
        self, chat_key: ChatKey, *, through_msg_id: MessageRowId, tail: int
    ) -> MemorySourceBatch | None:
        """Read a fixed source batch bounded by the terminal cursor and the
        durable memory watermark, retaining a recent tail.

        The batch covers the local rows in ``(watermark, through_msg_id]``,
        capped to the OLDEST ``tail`` messages (the oldest bounded
        unsummarized chunk) so no source rows are ever skipped — the
        watermark advances to the chunk's last row and the next read picks
        up the next chunk, leaving no gaps. None when nothing is beyond the
        watermark (or the chat is unknown). ``source_hash`` is computed here
        from the batch texts, so the summarizer never invents one, and
        ``observed_watermark`` is the watermark read at batch time — the
        exact snapshot the CAS commit fences on.
        """
        if tail <= 0:
            raise ValueError(f"tail must be positive, got {tail}")

        def fn(conn: Any) -> MemorySourceBatch | None:
            row = conn.execute(
                "SELECT memory_through_msg_id FROM chats WHERE chat_key = ?",
                (chat_key,),
            ).fetchone()
            if row is None:
                return None  # unknown chat
            wm = row[0] if row[0] is not None else 0
            # SQL-bounded I/O: the OLDEST bounded unsummarized chunk is
            # selected with a LIMIT, never fetch-all-then-slice, so a chat
            # with a huge backlog reads at most ``tail`` rows into memory.
            rows = conn.execute(
                "SELECT id, text FROM messages"
                " WHERE chat_key = ? AND id > ? AND id <= ?"
                " ORDER BY id LIMIT ?",
                (chat_key, wm, through_msg_id, tail),
            ).fetchall()
            if not rows:
                return None
            texts = tuple(r[1] for r in rows)
            return MemorySourceBatch(
                chat_key=chat_key,
                first_msg_id=MessageRowId(rows[0][0]),
                last_msg_id=MessageRowId(rows[-1][0]),
                source_hash=self._source_hash(texts),
                texts=texts,
                observed_watermark=MessageRowId(wm) if wm else MessageRowId(0),
            )

        return await self._db.read(fn)

    async def commit_memory_source(self, request: MemoryWriteRequest) -> bool:
        """CAS commit of one memory source range.

        One writer transaction: fence the durable watermark on the exact
        observed snapshot (``request.expected_through_msg_id``, else the
        batch's ``observed_watermark``), verify the source range is beyond
        the watermark and its ``source_hash`` still matches the current
        messages, insert EXACTLY ONE memory record (a batch producing 2+
        records is rejected atomically — a programming error in the
        summarizer), write its canonical FTS document, and advance the
        watermark to the batch's last row. Returns False when the watermark
        moved (stale CAS — nothing changes); raises RepoError for cross-chat,
        hash/range, or multi-record violations.
        """

        def fn(conn: Any) -> bool:
            row = conn.execute(
                "SELECT memory_through_msg_id FROM chats WHERE chat_key = ?",
                (request.chat_key,),
            ).fetchone()
            if row is None:
                raise RepoError(f"unknown chat: {request.chat_key!r}")
            wm = row[0] if row[0] is not None else 0
            expected = request.expected_through_msg_id
            if expected is None:
                expected = request.batch.observed_watermark
            if expected is None:
                expected = 0
            if wm != expected:
                return False  # stale CAS: the watermark moved
            batch = request.batch
            if batch.chat_key != request.chat_key:
                raise RepoError("cross-chat memory batch")
            if batch.first_msg_id <= wm:
                raise RepoError(
                    "source range overlaps the watermark:"
                    f" first {batch.first_msg_id} <= watermark {wm}"
                )
            if len(request.records) != 1:
                raise RepoError(
                    "exactly one memory record per source batch is required,"
                    f" got {len(request.records)}"
                )
            rows = conn.execute(
                "SELECT text FROM messages WHERE chat_key = ?"
                " AND id >= ? AND id <= ? ORDER BY id",
                (request.chat_key, batch.first_msg_id, batch.last_msg_id),
            ).fetchall()
            texts = tuple(r[0] for r in rows)
            if self._source_hash(texts) != batch.source_hash:
                raise RepoError(
                    "source hash mismatch: messages changed since the batch was read"
                )
            rec = request.records[0]
            if rec.chat_key != request.chat_key:
                raise RepoError("cross-chat memory record")
            if rec.source_first_msg_id is not None and (
                rec.source_first_msg_id != batch.first_msg_id
                or rec.source_last_msg_id != batch.last_msg_id
            ):
                raise RepoError(
                    "memory record source range does not match the batch"
                )
            cur = conn.execute(
                "INSERT INTO memories(chat_key, kind, text, cues_json,"
                " strength, created_ts, last_hit_ts, source_first_msg_id,"
                " source_last_msg_id, source_hash)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(chat_key, source_first_msg_id,"
                " source_last_msg_id) DO NOTHING",
                (
                    rec.chat_key,
                    rec.kind,
                    rec.text,
                    orjson.dumps(list(rec.cues), default=str).decode("utf-8"),
                    rec.strength,
                    rec.created_ts,
                    rec.last_hit_ts,
                    batch.first_msg_id,
                    batch.last_msg_id,
                    batch.source_hash,
                ),
            )
            if cur.rowcount == 1:
                self._memory_fts_insert(
                    conn, MessageRowId(cur.lastrowid), rec.text, rec.chat_key
                )
            else:
                # The source range already has a memory row: this batch maps
                # to ZERO new records. Fail closed (never advance the
                # watermark past a range that produced no record) — the CAS
                # fence normally prevents this, so it is a programming error.
                raise RepoError(
                    "memory source range already committed:"
                    f" {batch.first_msg_id}..{batch.last_msg_id}"
                )
            conn.execute(
                "UPDATE chats SET memory_through_msg_id = ?,"
                " memory_observed_through_msg_id = ? WHERE chat_key = ?",
                (batch.last_msg_id, expected, request.chat_key),
            )
            return True

        return await self._db.write(fn)

    def _memory_fts_insert(
        self, conn: Any, memory_id: MessageRowId, text: str, chat_key: ChatKey
    ) -> None:
        """Insert one canonical memory FTS token document and its index row
        in the caller's transaction (external-content FTS5)."""
        tokens = bigram_tokenize(text)
        if not tokens:
            return
        joined = " ".join(tokens)
        cur = conn.execute(
            "INSERT INTO memory_search_docs(chat_key, memory_id, tokens)"
            " VALUES (?, ?, ?)",
            (chat_key, memory_id, joined),
        )
        conn.execute(
            "INSERT INTO memory_search_fts(rowid, tokens) VALUES (?, ?)",
            (cur.lastrowid, joined),
        )

    async def rebuild_memory_fts(self, chat_key: ChatKey) -> None:
        """Local rebuild/backfill of the canonical memory FTS index for one
        chat: tokenizes every existing raw memory text with the repo's
        central ``bigram_tokenize`` and transactionally rebuilds the index
        from the canonical token documents — a rebuild reproduces exactly.
        Idempotent: re-running it reproduces the same index, and it marks
        the chat's FTS bootstrap state as done.
        """

        def fn(conn: Any) -> None:
            conn.execute(
                "DELETE FROM memory_search_docs WHERE chat_key = ?", (chat_key,)
            )
            rows = conn.execute(
                "SELECT id, text FROM memories WHERE chat_key = ?", (chat_key,)
            ).fetchall()
            for memory_id, text in rows:
                tokens = bigram_tokenize(text)
                if tokens:
                    conn.execute(
                        "INSERT INTO memory_search_docs(chat_key, memory_id, tokens)"
                        " VALUES (?, ?, ?)",
                        (chat_key, memory_id, " ".join(tokens)),
                    )
            conn.execute(
                "INSERT INTO memory_search_fts(memory_search_fts) VALUES('rebuild')"
            )
            conn.execute(
                "INSERT INTO memory_fts_state(chat_key, bootstrapped,"
                " backlog_through_msg_id) VALUES (?, 1, NULL)"
                " ON CONFLICT(chat_key) DO UPDATE SET bootstrapped = 1,"
                " backlog_through_msg_id = NULL",
                (chat_key,),
            )

        return await self._db.write(fn)

    async def get_memory_fts_state(
        self, chat_key: ChatKey
    ) -> tuple[bool, MessageRowId | None] | None:
        """The idempotent canonical memory FTS bootstrap/backlog state for
        one chat: ``(bootstrapped, backlog_through_msg_id)``, or None when
        the chat has no recorded state."""

        def fn(conn: Any) -> tuple[bool, MessageRowId | None] | None:
            row = conn.execute(
                "SELECT bootstrapped, backlog_through_msg_id FROM memory_fts_state"
                " WHERE chat_key = ?",
                (chat_key,),
            ).fetchone()
            if row is None:
                return None
            return (
                bool(row[0]),
                MessageRowId(row[1]) if row[1] is not None else None,
            )

        return await self._db.read(fn)

    async def mark_memory_fts_backlog(
        self, chat_key: ChatKey, through_msg_id: MessageRowId
    ) -> None:
        """Record the durable backlog cursor for a chat's canonical memory
        FTS bootstrap (idempotent).

        Marking a backlog re-marks the chat UNBOOTSTRAPPED (``bootstrapped =
        0``), so ``list_memory_fts_unbootstrapped_chats`` re-enumerates it
        and the next ``rebuild_memory_fts`` clears the backlog — the backlog
        state genuinely drives bootstrap enumeration instead of being dead.
        """

        def fn(conn: Any) -> None:
            conn.execute(
                "INSERT INTO memory_fts_state(chat_key, bootstrapped,"
                " backlog_through_msg_id) VALUES (?, 0, ?)"
                " ON CONFLICT(chat_key) DO UPDATE SET bootstrapped = 0,"
                " backlog_through_msg_id = excluded.backlog_through_msg_id",
                (chat_key, through_msg_id),
            )

        return await self._db.write(fn)

    async def list_memory_fts_unbootstrapped_chats(self) -> list[ChatKey]:
        """Chats whose canonical memory FTS index has not been bootstrapped —
        the set the local FTS bootstrap must rebuild at DB start.

        Driven by the durable ``memory_fts_state`` backlog markers: a chat
        is enumerated when it has a state row with ``bootstrapped = 0`` (a
        pending backlog), OR it has memory records but no state row at all
        (a legacy chat never marked — never dropped, so no data loss).
        Deterministic order; idempotent (a rebuild marks the chat
        bootstrapped and clears the backlog)."""
        def fn(conn: Any) -> list[ChatKey]:
            rows = conn.execute(
                "SELECT chat_key FROM memory_fts_state WHERE bootstrapped = 0"
                " UNION"
                " SELECT DISTINCT m.chat_key FROM memories m"
                " LEFT JOIN memory_fts_state s ON s.chat_key = m.chat_key"
                " WHERE s.chat_key IS NULL"
                " ORDER BY chat_key"
            ).fetchall()
            return [ChatKey(r[0]) for r in rows]

        return await self._db.read(fn)

    async def list_memory_pending_chats(self) -> list[tuple[ChatKey, MessageRowId]]:
        """Chats with pending memory work: at least one message beyond the
        durable memory watermark (a crash-after-settlement gap where the
        cursor advanced but the memory watermark did not). Returns
        ``(chat_key, through_msg_id)`` where ``through_msg_id`` is the chat's
        current cursor — the through boundary for the next summarize.
        Deterministic order."""
        def fn(conn: Any) -> list[tuple[ChatKey, MessageRowId]]:
            rows = conn.execute(
                "SELECT chat_key, cursor_msg_id FROM chats"
                " WHERE cursor_msg_id IS NOT NULL"
                "   AND (memory_through_msg_id IS NULL"
                "        OR cursor_msg_id > memory_through_msg_id)"
                " ORDER BY chat_key"
            ).fetchall()
            return [(ChatKey(r[0]), MessageRowId(r[1])) for r in rows]

        return await self._db.read(fn)

    async def query_memory(
        self, chat_key: ChatKey, query: str, *, limit: int = 10
    ) -> list[LexicalHit]:
        """Bounded chat-safe memory FTS query (lexical-first recall).

        Ranking uses a CHAT-LOCAL BM25: the document-frequency and
        average-document-length statistics are computed ONLY from this
        chat's own canonical token documents, so one chat's data never
        changes another chat's scores or order (no global-BM25 influence
        after the chat filter). Ties are broken deterministically by
        memory_id ASC. The FTS MATCH stays parameterized (query-safe) and
        the result is bounded to ``limit``.
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

        def fn(conn: Any) -> list[LexicalHit]:
            tokens = bigram_tokenize(query)
            if not tokens:
                return []
            match = " OR ".join(
                '"' + t.replace('"', '""') + '"' for t in tokens
            )
            rows = conn.execute(
                "SELECT memory_search_fts.rowid, d.memory_id, d.tokens,"
                " m.text, m.source_first_msg_id, m.source_last_msg_id"
                " FROM memory_search_fts"
                " JOIN memory_search_docs d ON d.id = memory_search_fts.rowid"
                " JOIN memories m ON m.id = d.memory_id"
                " WHERE memory_search_fts MATCH ? AND d.chat_key = ?",
                (match, chat_key),
            ).fetchall()
            if not rows:
                return []
            docs = [
                {
                    "memory_id": r[1],
                    "text": r[3],
                    "first": r[4],
                    "last": r[5],
                    "tokens": r[2].split() if r[2] else [],
                }
                for r in rows
            ]
            scored = self._chat_local_bm25(docs, tokens)
            scored.sort(key=lambda s: (-s[1], s[0]))
            out: list[LexicalHit] = []
            for memory_id, score, first, last, text in scored[:limit]:
                out.append(
                    LexicalHit(
                        chat_key=chat_key,
                        memory_id=memory_id,
                        text=text,
                        score=score,
                        source_first_msg_id=MessageRowId(first)
                        if first is not None
                        else None,
                        source_last_msg_id=MessageRowId(last)
                        if last is not None
                        else None,
                    )
                )
            return out

        return await self._db.read(fn)

    @staticmethod
    def _chat_local_bm25(
        docs: list[dict[str, Any]],
        query_tokens: list[str],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> list[tuple[int, float, Any, Any, str]]:
        """Chat-local BM25 over ``docs`` (each with ``memory_id``, ``text``,
        ``first``, ``last``, ``tokens``) for ``query_tokens``.

        All statistics (document count, per-term document frequency, average
        document length) are computed from ``docs`` alone — the chat's own
        canonical token documents — so the score is independent of every
        other chat. Returns ``(memory_id, score, first, last, text)``.
        """
        n = len(docs)
        lengths = [len(d["tokens"]) for d in docs]
        avgdl = sum(lengths) / n if n else 0.0
        df: dict[str, int] = {}
        for t in query_tokens:
            df[t] = sum(1 for d in docs if t in d["tokens"])
        out: list[tuple[int, float, Any, Any, str]] = []
        for d in docs:
            doc_len = len(d["tokens"])
            score = 0.0
            for t in query_tokens:
                tf = d["tokens"].count(t)
                if tf == 0:
                    continue
                idf = math.log(1.0 + (n - df[t] + 0.5) / (df[t] + 0.5))
                denom = (
                    tf + k1 * (1.0 - b + b * doc_len / avgdl)
                    if avgdl > 0
                    else tf + k1
                )
                score += idf * (tf * (k1 + 1.0)) / denom
            out.append((d["memory_id"], score, d["first"], d["last"], d["text"]))
        return out

    async def get_memories(
        self, chat_key: ChatKey, memory_ids: list[int]
    ) -> list[MemoryRecord]:
        """Chat-safe bulk memory lookup: the memory records for the given
        ids, restricted to the chat, in deterministic id order. Unknown ids
        (or ids belonging to another chat) are simply absent."""

        if not memory_ids:
            return []

        def fn(conn: Any) -> list[MemoryRecord]:
            placeholders = ",".join("?" for _ in memory_ids)
            rows = conn.execute(
                "SELECT id, chat_key, kind, text, cues_json, strength, created_ts,"
                " last_hit_ts, source_first_msg_id, source_last_msg_id, source_hash"
                f" FROM memories WHERE chat_key = ? AND id IN ({placeholders})"
                " ORDER BY id",
                (chat_key, *memory_ids),
            ).fetchall()
            return [self._row_to_memory(r) for r in rows]

        return await self._db.read(fn)

    async def list_memory_chats(self) -> list[ChatKey]:
        """Every chat with at least one memory record, in deterministic
        order — the chat-level page the semantic backfill iterates."""
        def fn(conn: Any) -> list[ChatKey]:
            rows = conn.execute(
                "SELECT DISTINCT chat_key FROM memories ORDER BY chat_key"
            ).fetchall()
            return [ChatKey(r[0]) for r in rows]

        return await self._db.read(fn)

    async def list_memory_chats_after(
        self, after: ChatKey, *, limit: int
    ) -> list[ChatKey]:
        """One bounded keyset page of memory chats strictly after ``after``
        (``chat_key > after``, deterministic order).

        The semantic backfill iterates the full chat set in fixed pages
        (``after=""`` yields the first page) instead of loading every chat
        at once, so maintenance stays bounded, cancellable, and fair.
        """
        if isinstance(limit, bool) or limit <= 0:
            raise RepoError("limit must be a positive integer")

        def fn(conn: Any) -> list[ChatKey]:
            rows = conn.execute(
                "SELECT DISTINCT chat_key FROM memories WHERE chat_key > ?"
                " ORDER BY chat_key LIMIT ?",
                (after, limit),
            ).fetchall()
            return [ChatKey(r[0]) for r in rows]

        return await self._db.read(fn)

    async def list_memories(self, chat_key: ChatKey) -> list[MemoryRecord]:
        """One chat's memory records in deterministic id order (chat-scoped
        enumeration for the semantic backfill)."""
        def fn(conn: Any) -> list[MemoryRecord]:
            rows = conn.execute(
                "SELECT id, chat_key, kind, text, cues_json, strength, created_ts,"
                " last_hit_ts, source_first_msg_id, source_last_msg_id, source_hash"
                " FROM memories WHERE chat_key = ? ORDER BY id",
                (chat_key,),
            ).fetchall()
            return [self._row_to_memory(r) for r in rows]

        return await self._db.read(fn)

    async def list_memories_after(
        self, chat_key: ChatKey, after_id: int, *, limit: int
    ) -> list[MemoryRecord]:
        """One bounded, chat-scoped memory page strictly after ``after_id``.

        Incremental semantic work consumes this instead of repeatedly loading
        every historical memory after each terminal cycle.
        """
        if isinstance(after_id, bool) or after_id < 0:
            raise RepoError("after_id must be a nonnegative integer")
        if isinstance(limit, bool) or limit <= 0:
            raise RepoError("limit must be a positive integer")

        def fn(conn: Any) -> list[MemoryRecord]:
            rows = conn.execute(
                "SELECT id, chat_key, kind, text, cues_json, strength, created_ts,"
                " last_hit_ts, source_first_msg_id, source_last_msg_id, source_hash"
                " FROM memories WHERE chat_key = ? AND id > ? ORDER BY id LIMIT ?",
                (chat_key, after_id, limit),
            ).fetchall()
            return [self._row_to_memory(r) for r in rows]

        return await self._db.read(fn)

    @staticmethod
    def _row_to_memory(row: Any) -> MemoryRecord:
        """Hydrate one ``memories`` row into a ``MemoryRecord``."""
        return MemoryRecord(
            id=MessageRowId(row[0]),
            chat_key=ChatKey(row[1]),
            kind=row[2],
            text=row[3],
            cues=tuple(orjson.loads(row[4])),
            strength=row[5],
            created_ts=row[6],
            last_hit_ts=row[7],
            source_first_msg_id=MessageRowId(row[8])
            if row[8] is not None
            else None,
            source_last_msg_id=MessageRowId(row[9])
            if row[9] is not None
            else None,
            source_hash=row[10],
        )

    async def get_person(
        self, chat_key: ChatKey, platform_uid: SenderId
    ) -> PersonProfile | None:
        def fn(conn: Any) -> PersonProfile | None:
            row = conn.execute(
                "SELECT person_key, chat_key, platform_uid, names_json, profile,"
                " impression, updated_ts, profile_through_msg_id FROM persons"
                " WHERE chat_key = ? AND platform_uid = ?",
                (chat_key, platform_uid),
            ).fetchone()
            if row is None:
                return None
            return PersonProfile(
                person_key=PersonKey(row[0]),
                chat_key=ChatKey(row[1]),
                platform_uid=SenderId(row[2]),
                names=tuple(orjson.loads(row[3])),
                profile=row[4],
                impression=row[5],
                updated_ts=row[6],
                profile_through_msg_id=MessageRowId(row[7])
                if row[7] is not None
                else None,
            )

        return await self._db.read(fn)

    async def upsert_person(self, profile: PersonProfile) -> None:
        """Upsert one per-chat person identity. The durable profile cursor
        (``profile_through_msg_id``) is NEVER touched here — it is written
        only by ``cas_person_profile``."""

        def fn(conn: Any) -> None:
            person_key = profile.person_key or self._person_key(
                profile.chat_key, profile.platform_uid
            )
            conn.execute(
                "INSERT INTO persons(person_key, chat_key, platform_uid,"
                " names_json, profile, impression, updated_ts)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(chat_key, platform_uid) DO UPDATE SET"
                " names_json = excluded.names_json, profile = excluded.profile,"
                " impression = excluded.impression, updated_ts = excluded.updated_ts",
                (
                    person_key,
                    profile.chat_key,
                    profile.platform_uid,
                    orjson.dumps(list(profile.names), default=str).decode("utf-8"),
                    profile.profile,
                    profile.impression,
                    profile.updated_ts,
                ),
            )

        return await self._db.write(fn)

    async def add_person_alias(
        self,
        chat_key: ChatKey,
        platform_uid: SenderId,
        name: str,
        *,
        now: float | None = None,
    ) -> tuple[str, ...] | None:
        """ATOMIC alias-only merge for one per-chat person: append ``name``
        to the alias list (first-seen order, deduped, bounded to
        ``MAX_ALIASES``) WITHOUT touching the profile/impression or the
        durable profile cursor. Creates the person when unknown. Returns the
        resulting alias list, or None when the person is unknown and the
        name is blank. This is the single atomic alias operation the
        PersonService uses — never a read-modify-upsert that could overwrite
        a concurrent profile write.
        """
        if not isinstance(name, str) or not name.strip():
            return None

        def fn(conn: Any) -> tuple[str, ...]:
            row = conn.execute(
                "SELECT person_key, names_json FROM persons"
                " WHERE chat_key = ? AND platform_uid = ?",
                (chat_key, platform_uid),
            ).fetchone()
            if row is None:
                person_key = self._person_key(chat_key, platform_uid)
                conn.execute(
                    "INSERT INTO persons(person_key, chat_key, platform_uid,"
                    " names_json, updated_ts) VALUES (?, ?, ?, ?, ?)",
                    (
                        person_key,
                        chat_key,
                        platform_uid,
                        orjson.dumps([name]).decode("utf-8"),
                        now,
                    ),
                )
                return (name,)
            names = tuple(orjson.loads(row[1]))
            if name in names or len(names) >= MAX_ALIASES:
                return names
            new_names = names + (name,)
            conn.execute(
                "UPDATE persons SET names_json = ?, updated_ts = ?"
                " WHERE chat_key = ? AND platform_uid = ?",
                (
                    orjson.dumps(list(new_names)).decode("utf-8"),
                    now,
                    chat_key,
                    platform_uid,
                ),
            )
            return new_names

        return await self._db.write(fn)

    async def cas_person_profile(
        self,
        chat_key: ChatKey,
        platform_uid: SenderId,
        expected_through_msg_id: MessageRowId | None,
        profile: PersonProfile,
    ) -> bool:
        """CAS the durable profile cursor: update the profile content AND
        the cursor in one transaction, fenced on the expected cursor value.

        ``names_json`` is NEVER written here — aliases are authoritative and
        mutated only by ``add_person_alias``, so an alias merged between the
        profile read and this CAS always survives (the CAS can never
        overwrite it). Fail closed: the profile must target the same
        chat+UID (a cross-chat/person profile is rejected), the new cursor
        must not regress the stored cursor, and the update succeeds only
        when the stored cursor still equals the expected value. False when
        the cursor moved (stale CAS) or the person is unknown.
        """
        if profile.chat_key != chat_key or profile.platform_uid != platform_uid:
            raise RepoError(
                "cross-chat/person profile CAS: profile targets "
                f"{profile.chat_key!r}/{profile.platform_uid!r}, not "
                f"{chat_key!r}/{platform_uid!r}"
            )

        def fn(conn: Any) -> bool:
            expected = (
                expected_through_msg_id
                if expected_through_msg_id is not None
                else 0
            )
            new_cursor = (
                profile.profile_through_msg_id
                if profile.profile_through_msg_id is not None
                else 0
            )
            cur = conn.execute(
                "UPDATE persons SET profile = ?, impression = ?,"
                " updated_ts = ?, profile_through_msg_id = ?"
                " WHERE chat_key = ? AND platform_uid = ?"
                " AND COALESCE(profile_through_msg_id, 0) = ?"
                " AND ? >= COALESCE(profile_through_msg_id, 0)",
                (
                    profile.profile,
                    profile.impression,
                    profile.updated_ts,
                    profile.profile_through_msg_id,
                    chat_key,
                    platform_uid,
                    expected,
                    new_cursor,
                ),
            )
            return cur.rowcount == 1

        return await self._db.write(fn)

    @staticmethod
    def _space_id(model: str, revision: str) -> str:
        """The canonical embedding space identity: model + explicit revision."""
        return f"{model}@{revision}"

    async def create_embedding_generation(
        self,
        model: str,
        dim: int,
        *,
        revision: str = "default",
        state: str = "inactive",
        created_ts: float | None = None,
    ) -> EmbeddingGeneration:
        """Create one embedding generation. Idempotent per ``space_id``
        (model + explicit revision): an existing generation with the same
        space AND the same model/revision/dim is returned unchanged. A
        same-space generation with an INCOMPATIBLE requested dim (or
        model/revision) is rejected with ``RepoError`` — never silently
        returned as a mismatched row. No provider side effects.

        ``state`` (default ``inactive``) admits ``inactive`` (manual/
        legacy creation) and ``building`` (the semantic backfill's
        in-progress generation). ``active`` is NOT settable here —
        activation is reached ONLY through
        ``activate_embedding_generation``, which atomically enforces the
        at-most-one-active invariant.
        """

        def fn(conn: Any) -> EmbeddingGeneration:
            if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
                raise RepoError(f"dim must be a positive integer, got {dim!r}")
            if not isinstance(revision, str) or not revision:
                raise RepoError(f"revision must be a non-empty string, got {revision!r}")
            if state not in ("inactive", "building"):
                raise RepoError(
                    f"state must be 'inactive' or 'building', got {state!r}"
                )
            space_id = self._space_id(model, revision)
            conn.execute(
                "INSERT INTO embedding_generations(space_id, model, revision,"
                " dim, state, created_ts) VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(space_id) DO NOTHING",
                (space_id, model, revision, dim, state, created_ts),
            )
            row = conn.execute(
                "SELECT id, space_id, model, revision, dim, state, created_ts,"
                " vector_revision FROM embedding_generations WHERE space_id = ?",
                (space_id,),
            ).fetchone()
            # The space already exists: it must be dimension/model/revision
            # compatible, or the caller asked for a mismatched row. Fail
            # closed rather than silently returning the existing generation.
            if row[2] != model or row[3] != revision or row[4] != dim:
                raise RepoError(
                    "embedding generation for space "
                    f"{space_id!r} already exists with model {row[2]!r}/"
                    f"revision {row[3]!r}/dim {row[4]}, incompatible with"
                    f" requested {model!r}/{revision!r}/dim {dim}"
                )
            return EmbeddingGeneration(
                id=row[0], space_id=row[1], model=row[2], revision=row[3],
                dim=row[4], state=row[5], created_ts=row[6], vector_revision=row[7],
            )

        return await self._db.write(fn)

    async def get_embedding_generation(
        self, generation_id: int
    ) -> EmbeddingGeneration | None:
        def fn(conn: Any) -> EmbeddingGeneration | None:
            row = conn.execute(
                "SELECT id, space_id, model, revision, dim, state, created_ts,"
                " vector_revision FROM embedding_generations WHERE id = ?",
                (generation_id,),
            ).fetchone()
            if row is None:
                return None
            return EmbeddingGeneration(
                id=row[0], space_id=row[1], model=row[2], revision=row[3],
                dim=row[4], state=row[5], created_ts=row[6], vector_revision=row[7],
            )

        return await self._db.read(fn)

    async def activate_embedding_generation(self, generation_id: int) -> bool:
        """Activate one embedding generation: atomically deactivate the
        previous active generation and activate this one. False when the
        generation does not exist (fail closed); the partial unique index
        guarantees at most one active generation."""

        def fn(conn: Any) -> bool:
            row = conn.execute(
                "SELECT id FROM embedding_generations WHERE id = ?",
                (generation_id,),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                "UPDATE embedding_generations SET state = 'inactive'"
                " WHERE state = 'active'"
            )
            conn.execute(
                "UPDATE embedding_generations SET state = 'active' WHERE id = ?",
                (generation_id,),
            )
            return True

        return await self._db.write(fn)

    async def activate_embedding_generation_if_complete(
        self, generation_id: int
    ) -> list[ChatKey] | None:
        """Source-fenced activation in ONE writer transaction.

        The generation is activated ONLY when every current nonempty memory
        source AND every current trusted adaptive record has a valid matching
        vector for it. The model/dim and owner source hash must match the
        generation/owner exactly. The coverage check and activate decision
        share one transaction, so a memory or record committed between a
        scan and this call is visible and activation fails closed.

        Returns None when the generation was activated (the previous active
        generation is deactivated in the SAME transaction — the
        at-most-one-active invariant, so the current active generation is
        preserved until the building generation completes). Returns the
        deterministic list of chats whose memories are missing matching
        vectors when coverage is incomplete: the generation stays
        ``building`` and the caller enqueues those chats for repair. Returns
        [] when the generation does not exist or is not ``building`` (fail
        closed — a manual/legacy inactive generation is never hijacked). No
        provider/network calls in this transaction.
        """

        def fn(conn: Any) -> list[ChatKey] | None:
            row = conn.execute(
                "SELECT id, model, dim, state FROM embedding_generations"
                " WHERE id = ?",
                (generation_id,),
            ).fetchone()
            if row is None or row[3] != "building":
                return []
            gen_id, model, dim, _state = row
            missing = conn.execute(
                "SELECT DISTINCT m.chat_key FROM memories m"
                " WHERE m.text IS NOT NULL AND m.text != ''"
                " AND NOT EXISTS ("
                "  SELECT 1 FROM vectors v"
                "  WHERE v.owner_table = 'memories' AND v.owner_id = m.id"
                "   AND v.generation = ? AND v.model = ? AND v.dim = ?"
                "   AND v.source_hash IS m.source_hash)"
                " UNION SELECT DISTINCT r.chat_key FROM records r"
                " WHERE r.content_hash IS NOT NULL AND r.retired = 0"
                " AND NOT EXISTS ("
                "  SELECT 1 FROM vectors v"
                "  WHERE v.owner_table = 'records' AND v.owner_id = r.id"
                "   AND v.generation = ? AND v.model = ? AND v.dim = ?"
                "   AND v.source_hash = r.content_hash)"
                " ORDER BY chat_key",
                (gen_id, model, dim, gen_id, model, dim),
            ).fetchall()
            if missing:
                return [ChatKey(r[0]) for r in missing]
            conn.execute(
                "UPDATE embedding_generations SET state = 'inactive'"
                " WHERE state = 'active'"
            )
            conn.execute(
                "UPDATE embedding_generations SET state = 'active' WHERE id = ?",
                (generation_id,),
            )
            return None

        return await self._db.write(fn)

    async def list_embedding_generations(self) -> list[EmbeddingGeneration]:
        def fn(conn: Any) -> list[EmbeddingGeneration]:
            rows = conn.execute(
                "SELECT id, space_id, model, revision, dim, state, created_ts,"
                " vector_revision FROM embedding_generations ORDER BY id"
            ).fetchall()
            return [
                EmbeddingGeneration(
                    id=r[0], space_id=r[1], model=r[2], revision=r[3],
                    dim=r[4], state=r[5], created_ts=r[6], vector_revision=r[7],
                )
                for r in rows
            ]

        return await self._db.read(fn)

    def _validate_vector_owner(
        self,
        conn: Any,
        chat_key: ChatKey,
        owner_table: str,
        owner_id: int,
        source_hash: str | None | object = _NO_SOURCE_HASH,
    ) -> None:
        """Fail closed: the vector owner must exist and belong to the chat.

        Ownership is hard-coded to ``memories`` and ``records`` only. A
        record vector additionally enforces the content_hash source
        identity: the record must carry a non-NULL ``content_hash`` and the
        vector's ``source_hash`` must equal it (a record vector is only
        ever derived from the exact record content it claims to represent).
        """
        if owner_table == "memories":
            row = conn.execute(
                "SELECT chat_key FROM memories WHERE id = ?", (owner_id,)
            ).fetchone()
            if row is None or row[0] != chat_key:
                raise RepoError(
                    f"vector owner {owner_table}:{owner_id} does not belong to"
                    f" chat {chat_key!r}"
                )
        elif owner_table == "records":
            row = conn.execute(
                "SELECT chat_key, content_hash FROM records WHERE id = ?",
                (owner_id,),
            ).fetchone()
            if row is None or row[0] != chat_key:
                raise RepoError(
                    f"vector owner {owner_table}:{owner_id} does not belong to"
                    f" chat {chat_key!r}"
                )
            if source_hash is not _NO_SOURCE_HASH and (
                row[1] is None or row[1] != source_hash
            ):
                raise RepoError(
                    "record vector requires a matching content_hash source"
                    f" identity: record {owner_id} content_hash"
                    f" {row[1]!r} != vector source_hash {source_hash!r}"
                )
        else:
            raise RepoError(f"unsupported vector owner table: {owner_table!r}")

    async def upsert_vector(self, chat_key: ChatKey, row: VectorRow) -> None:
        """Upsert one chat-scoped vector row. The owner must exist and
        belong to the chat; the generation must exist AND the row's
        model/dim must match the generation's model/dim (no
        generation/model/dimension mixing). Fail closed otherwise. Bumps the
        generation's durable ``vector_revision`` so a direct repo mutation
        is visible on the next search."""

        def fn(conn: Any) -> None:
            gen = conn.execute(
                "SELECT id, model, dim FROM embedding_generations WHERE id = ?",
                (row.generation,),
            ).fetchone()
            if gen is None:
                raise RepoError(f"unknown embedding generation: {row.generation}")
            if row.model != gen[1] or row.dim != gen[2]:
                raise RepoError(
                    "vector model/dim does not match generation "
                    f"{row.generation}: row {row.model}/{row.dim} != "
                    f"generation {gen[1]}/{gen[2]}"
                )
            self._validate_vector_owner(
                conn, chat_key, row.owner_table, row.owner_id, row.source_hash
            )
            conn.execute(
                "INSERT INTO vectors(owner_table, owner_id, dim, model, generation,"
                " source_hash, blob) VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(owner_table, owner_id, model, generation)"
                " DO UPDATE SET dim = excluded.dim, source_hash = excluded.source_hash,"
                " blob = excluded.blob",
                (
                    row.owner_table,
                    row.owner_id,
                    row.dim,
                    row.model,
                    row.generation,
                    row.source_hash,
                    row.blob,
                ),
            )
            conn.execute(
                "UPDATE embedding_generations SET vector_revision = vector_revision + 1"
                " WHERE id = ?",
                (row.generation,),
            )

        return await self._db.write(fn)

    async def get_vector(
        self,
        chat_key: ChatKey,
        owner_table: str,
        owner_id: int,
        model: str,
        generation: int,
    ) -> VectorRow | None:
        def fn(conn: Any) -> VectorRow | None:
            source_hash = None
            if owner_table == "records":
                owner = conn.execute(
                    "SELECT content_hash FROM records WHERE id = ?", (owner_id,)
                ).fetchone()
                source_hash = owner[0] if owner is not None else None
            self._validate_vector_owner(
                conn, chat_key, owner_table, owner_id, source_hash
            )
            row = conn.execute(
                "SELECT owner_table, owner_id, dim, model, generation, source_hash,"
                " blob FROM vectors WHERE owner_table = ? AND owner_id = ?"
                " AND model = ? AND generation = ?",
                (owner_table, owner_id, model, generation),
            ).fetchone()
            if row is None:
                return None
            if owner_table == "records" and row[5] != source_hash:
                return None
            return VectorRow(
                owner_table=row[0],
                owner_id=row[1],
                dim=row[2],
                model=row[3],
                generation=row[4],
                source_hash=row[5],
                blob=row[6],
            )

        return await self._db.read(fn)

    async def list_vectors(
        self, chat_key: ChatKey, model: str, generation: int,
        owner_table: str | None = None,
    ) -> list[VectorRow]:
        """Every vector row whose owner belongs to the chat, for one model
        and generation — chat-scoped, deterministic order."""

        def fn(conn: Any) -> list[VectorRow]:
            if owner_table is None:
                rows = conn.execute(
                    "SELECT v.owner_table, v.owner_id, v.dim, v.model, v.generation,"
                    " v.source_hash, v.blob FROM vectors v"
                    " JOIN memories m ON m.id = v.owner_id"
                    " WHERE m.chat_key = ? AND v.owner_table = 'memories'"
                    " AND v.model = ? AND v.generation = ?"
                    " ORDER BY v.owner_id",
                    (chat_key, model, generation),
                ).fetchall()
            else:
                if owner_table not in ("memories", "records"):
                    raise ValueError("unsupported vector owner table")
                join = "memories" if owner_table == "memories" else "records"
                rows = conn.execute(
                    (
                        "SELECT v.owner_table, v.owner_id, v.dim, v.model, v.generation,"
                        " v.source_hash, v.blob FROM vectors v JOIN " + join +
                        " o ON o.id = v.owner_id WHERE o.chat_key = ?"
                        " AND v.owner_table = ? AND v.model = ? AND v.generation = ?"
                        + (" AND o.content_hash IS NOT NULL AND v.source_hash = o.content_hash"
                           if owner_table == "records" else "")
                        + " ORDER BY v.owner_id"
                    ),
                    (chat_key, owner_table, model, generation),
                ).fetchall()
            return [self._row_to_vector(r) for r in rows]

        return await self._db.read(fn)

    async def list_vectors_for_memories(
        self,
        chat_key: ChatKey,
        model: str,
        generation: int,
        memory_ids: list[int],
    ) -> list[VectorRow]:
        """Bounded chat-scoped vector lookup for a bounded set of memory ids
        (one memory page).

        The semantic backfill consumes this per memory page instead of
        loading every vector of a chat at once: the IN list is bounded by
        the fixed page size, and unknown or cross-chat ids are simply
        absent (strict chat scoping).
        """
        if not memory_ids:
            return []

        def fn(conn: Any) -> list[VectorRow]:
            placeholders = ",".join("?" for _ in memory_ids)
            rows = conn.execute(
                "SELECT v.owner_table, v.owner_id, v.dim, v.model, v.generation,"
                " v.source_hash, v.blob FROM vectors v"
                " JOIN memories m ON m.id = v.owner_id"
                " WHERE m.chat_key = ? AND v.owner_table = 'memories'"
                " AND v.model = ? AND v.generation = ? AND v.owner_id IN ("
                + placeholders + ")",
                (chat_key, model, generation, *memory_ids),
            ).fetchall()
            return [self._row_to_vector(r) for r in rows]

        return await self._db.read(fn)

    async def list_vectors_for_records(
        self,
        chat_key: ChatKey,
        model: str,
        generation: int,
        record_ids: list[int],
    ) -> list[VectorRow]:
        """Bounded chat-scoped vector lookup for adaptive records."""
        if not record_ids:
            return []

        def fn(conn: Any) -> list[VectorRow]:
            placeholders = ",".join("?" for _ in record_ids)
            rows = conn.execute(
                "SELECT v.owner_table, v.owner_id, v.dim, v.model, v.generation,"
                " v.source_hash, v.blob FROM vectors v JOIN records r"
                " ON r.id = v.owner_id WHERE r.chat_key = ?"
                " AND r.content_hash IS NOT NULL AND r.retired = 0"
                " AND v.owner_table = 'records' AND v.model = ?"
                " AND v.generation = ? AND v.source_hash = r.content_hash"
                " AND v.owner_id IN (" + placeholders + ")",
                (chat_key, model, generation, *record_ids),
            ).fetchall()
            return [self._row_to_vector(r) for r in rows]

        return await self._db.read(fn)

    @staticmethod
    def _row_to_vector(row: Any) -> VectorRow:
        """Hydrate one ``vectors`` row into a ``VectorRow``."""
        return VectorRow(
            owner_table=row[0],
            owner_id=row[1],
            dim=row[2],
            model=row[3],
            generation=row[4],
            source_hash=row[5],
            blob=row[6],
        )

    async def delete_vector(
        self,
        chat_key: ChatKey,
        owner_table: str,
        owner_id: int,
        model: str,
        generation: int,
    ) -> bool:
        def fn(conn: Any) -> bool:
            self._validate_vector_owner(conn, chat_key, owner_table, owner_id)
            cur = conn.execute(
                "DELETE FROM vectors WHERE owner_table = ? AND owner_id = ?"
                " AND model = ? AND generation = ?",
                (owner_table, owner_id, model, generation),
            )
            if cur.rowcount == 1:
                conn.execute(
                    "UPDATE embedding_generations SET vector_revision ="
                    " vector_revision + 1 WHERE id = ?",
                    (generation,),
                )
            return cur.rowcount == 1

        return await self._db.write(fn)

    # ── records / kv / stats ────────────────────────────────────────────────

    async def add_record(self, rec: Record) -> int:
        def fn(conn: Any) -> int:
            cur = conn.execute(
                "INSERT INTO records(chat_key, learner, payload_json, weight,"
                " uses, created_ts) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    rec.chat_key,
                    rec.learner,
                    orjson.dumps(rec.payload, default=str).decode("utf-8"),
                    rec.weight,
                    rec.uses,
                    rec.created_ts,
                ),
            )
            return cur.lastrowid

        return await self._db.write(fn)

    async def get_kv(self, k: str) -> str | None:
        def fn(conn: Any) -> str | None:
            row = conn.execute("SELECT v FROM kv WHERE k = ?", (k,)).fetchone()
            return row[0] if row is not None else None

        return await self._db.read(fn)

    async def set_kv(self, k: str, v: str) -> None:
        def fn(conn: Any) -> None:
            conn.execute(
                "INSERT INTO kv(k, v) VALUES (?, ?)"
                " ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                (k, v),
            )

        await self._db.write(fn)

    async def budget_update(
        self, key: str, *, transform: Callable[[str | None], str | None]
    ) -> str | None:
        """The atomic budget-ledger store operation (``BudgetStore`` seam).

        Loads the ``kv`` row for ``key``, applies the pure ``transform`` to
        the raw value (None when the key is missing), and persists the
        returned string — all inside ONE writer transaction. When
        ``transform`` returns None nothing is written. Returns the
        PRE-transform raw value (None when the key was missing).

        Because every write goes through the single writer connection, this
        load-modify-save is atomic against every other ``budget_update`` from
        DISTINCT ``BudgetManager`` instances over the same database — the
        per-instance asyncio lock alone cannot serialize them, so this is
        what makes simultaneous planner/embed reservations never exceed the
        cap. All budget policy stays in ``BudgetManager``; this only applies
        the transform atomically.
        """

        def fn(conn: Any) -> str | None:
            row = conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
            raw = row[0] if row is not None else None
            new = transform(raw)
            if new is None:
                return raw
            conn.execute(
                "INSERT INTO kv(k, v) VALUES (?, ?)"
                " ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                (key, new),
            )
            return raw

        return await self._db.write(fn)

    async def stats(self) -> dict[str, int]:
        """Basic table statistics for ``pretender db`` (the CLI contains no
        SQL text)."""

        def fn(conn: Any) -> dict[str, int]:
            out: dict[str, int] = {}
            for table in ("messages", "memories", "records", "cycles", "claims",
                          "emoji", "persons", "vectors", "embedding_generations",
                          "memory_search_docs", "media_assets"):
                out[table] = conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            for state in ("pending", "in_flight", "sent", "dropped"):
                out[f"outbox_{state}"] = conn.execute(
                    "SELECT COUNT(*) FROM outbox WHERE state = ?", (state,)
                ).fetchone()[0]
            out["user_version"] = conn.execute(
                "PRAGMA user_version"
            ).fetchone()[0]
            return out

        return await self._db.read(fn)

    # ── Phase 6 adaptive foundation (AdaptiveRepository seam) ────────────────
    # The adaptive learner storage surface: durable per-(chat, learner)
    # state/runs, source-bounded reads with policy-enforced ``is_self = 0``,
    # exact source-batch hash/CAS commits, chat+learner-scoped record reads
    # excluding legacy/retired rows, idempotent exposure/uses, bounded
    # effect feedback with a code-owned reweight, canonical record FTS, and
    # bounded recovery scans. No provider/network calls in any transaction.

    @staticmethod
    def _record_content_hash(payload: dict[str, Any]) -> str:
        """Deterministic content hash of a record payload — the adaptive
        record identity. Computed by the repository, so the learner never
        invents one."""
        h = hashlib.sha256()
        h.update(orjson.dumps(payload, default=str, option=orjson.OPT_SORT_KEYS))
        return h.hexdigest()

    @staticmethod
    def _record_search_text(payload: dict[str, Any]) -> str:
        """The canonical record FTS document text: the payload's ``text``
        field when present (a non-empty string), else the deterministic
        JSON rendering."""
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            return text
        return orjson.dumps(payload, default=str).decode("utf-8")

    @staticmethod
    def _row_to_record(row: Any) -> Record:
        """Hydrate one ``records`` row into a ``Record``."""
        return Record(
            id=row[0],
            chat_key=ChatKey(row[1]) if row[1] is not None else None,
            learner=row[2],
            payload=orjson.loads(row[3]),
            weight=row[4],
            uses=row[5],
            created_ts=row[6],
            content_hash=row[7],
            source_first_msg_id=MessageRowId(row[8])
            if row[8] is not None
            else None,
            source_last_msg_id=MessageRowId(row[9])
            if row[9] is not None
            else None,
            retired=bool(row[10]),
            producing_run_id=row[11] if len(row) > 11 else None,
        )

    async def acquire_learner_run(
        self, request: LearnerRunRequest
    ) -> LearnerGrant | LearnerBusy | None:
        """Compare-and-swap claim of one learner run.

        ``LearnerGrant`` when the claim succeeded; ``LearnerBusy`` (with
        the active owner's exact ``busy_until``) when the chat+learner
        already has a live, unexpired prepared run; None when the chat is
        unknown. An expired prepared run is recovered (marked ``expired``
        and replaced) in the same transaction. The grant's boundary is
        fixed at claim time: ``start_msg_id`` is the chat cursor,
        ``through_msg_id`` the chat's max message row id.
        """

        def fn(conn: Any) -> LearnerGrant | LearnerBusy | None:
            if not (
                math.isfinite(request.started_ts)
                and math.isfinite(request.expires_at)
            ):
                return None
            row = conn.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?",
                (request.chat_key,),
            ).fetchone()
            if row is None:
                return None  # unknown chat: nothing to claim
            start = row[0] if row[0] is not None else 0
            # Recover expired prepared runs for this (chat, learner).
            conn.execute(
                "UPDATE learner_runs SET state = 'expired'"
                " WHERE chat_key = ? AND learner = ? AND state = 'prepared'"
                " AND (expires_at IS NULL OR expires_at <= ?)",
                (request.chat_key, request.learner, request.now),
            )
            live = conn.execute(
                "SELECT id, expires_at FROM learner_runs"
                " WHERE chat_key = ? AND learner = ? AND state = 'prepared'",
                (request.chat_key, request.learner),
            ).fetchone()
            if live is not None:
                live_expires = live[1]
                # Recovery uses the same <= expiry semantics as renew; a
                # non-finite stored lease is treated as expired.
                if not math.isfinite(live_expires) or live_expires <= request.now:
                    conn.execute(
                        "UPDATE learner_runs SET state = 'expired' WHERE id = ?",
                        (live[0],),
                    )
                else:
                    return LearnerBusy(
                        chat_key=request.chat_key,
                        learner=request.learner,
                        run_id=live[0],
                        busy_until=live_expires,
                    )
            through = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM messages WHERE chat_key = ?",
                (request.chat_key,),
            ).fetchone()[0]
            cur = conn.execute(
                "INSERT INTO learner_runs(chat_key, learner, started_ts,"
                " expires_at, start_msg_id, through_msg_id, state)"
                " VALUES (?, ?, ?, ?, ?, ?, 'prepared')",
                (
                    request.chat_key,
                    request.learner,
                    request.started_ts,
                    request.expires_at,
                    start,
                    through,
                ),
            )
            return LearnerGrant(
                chat_key=request.chat_key,
                learner=request.learner,
                run_id=cur.lastrowid,
                started_ts=request.started_ts,
                expires_at=request.expires_at,
                start_msg_id=MessageRowId(start),
                through_msg_id=MessageRowId(through),
            )

        return await self._db.write(fn)

    async def renew_learner_run(
        self,
        chat_key: ChatKey,
        learner: str,
        run_id: int,
        expires_at: float,
        *,
        now: float,
    ) -> bool:
        """Extend the learner run lease. False when the run is not
        prepared, not ours, or already expired — an expired owner cannot
        renew even before another claimant acts."""

        def fn(conn: Any) -> bool:
            if not math.isfinite(expires_at) or expires_at <= now:
                return False
            row = conn.execute(
                "SELECT expires_at FROM learner_runs"
                " WHERE chat_key = ? AND learner = ? AND id = ?"
                " AND state = 'prepared'",
                (chat_key, learner, run_id),
            ).fetchone()
            if row is None or not math.isfinite(row[0]) or row[0] <= now:
                return False
            cur = conn.execute(
                "UPDATE learner_runs SET expires_at = ?"
                " WHERE chat_key = ? AND learner = ? AND id = ?"
                " AND state = 'prepared'",
                (expires_at, chat_key, learner, run_id),
            )
            return cur.rowcount == 1

        return await self._db.write(fn)

    async def release_learner_run(
        self, chat_key: ChatKey, learner: str, run_id: int
    ) -> None:
        """Give the run back WITHOUT moving the watermark: the source rows
        stay pending for the next run."""

        def fn(conn: Any) -> None:
            conn.execute(
                "UPDATE learner_runs SET state = 'released'"
                " WHERE chat_key = ? AND learner = ? AND id = ?"
                " AND state = 'prepared'",
                (chat_key, learner, run_id),
            )

        return await self._db.write(fn)

    async def read_learner_source_batch(
        self,
        chat_key: ChatKey,
        learner: str,
        *,
        through_msg_id: MessageRowId,
        tail: int,
        policy: str = "nonself",
    ) -> LearnerBatch | None:
        """Read a fixed source batch bounded by the run's through boundary
        and the learner's durable watermark, retaining a recent tail.

        The batch covers the local rows in ``(watermark, through_msg_id]``,
        capped to the OLDEST ``tail`` messages (the oldest bounded
        unsummarized chunk) so no source rows are ever skipped. ``policy``
        is ``nonself`` (the default: ``is_self = 0`` is enforced in SQL, so
        the bot's own output never enters a nonself batch) or ``all``. None
        when nothing is beyond the watermark (or the chat is unknown).
        ``source_hash`` is computed here from the batch texts, so the
        learner never invents one, and ``observed_watermark`` is the
        watermark read at batch time — the exact snapshot the CAS commit
        fences on.
        """
        if tail <= 0:
            raise ValueError(f"tail must be positive, got {tail}")
        if policy not in ("nonself", "all"):
            raise ValueError(f"policy must be 'nonself' or 'all', got {policy!r}")

        def fn(conn: Any) -> LearnerBatch | None:
            row = conn.execute(
                "SELECT watermark_msg_id FROM learner_state"
                " WHERE chat_key = ? AND learner = ?",
                (chat_key, learner),
            ).fetchone()
            wm = row[0] if row is not None and row[0] is not None else 0
            self_filter = "AND is_self = 0" if policy == "nonself" else ""
            # SQL-bounded I/O: the OLDEST bounded unsummarized chunk is
            # selected with a LIMIT, never fetch-all-then-slice.
            sql = (
                "SELECT id, text, sender_id, sender_name FROM messages"
                " WHERE chat_key = ? AND id > ? AND id <= ?"
                + self_filter
                + " ORDER BY id LIMIT ?"
            )
            rows = conn.execute(sql, (chat_key, wm, through_msg_id, tail)).fetchall()
            if not rows:
                return None
            texts = tuple(r[1] for r in rows)
            return LearnerBatch(
                chat_key=chat_key,
                learner=learner,
                first_msg_id=MessageRowId(rows[0][0]),
                last_msg_id=MessageRowId(rows[-1][0]),
                # Computed from the TEXTS only: the sender columns below are
                # additional context for the impression learner and must
                # never move an existing watermark's source identity.
                source_hash=self._source_hash(texts),
                texts=texts,
                observed_watermark=MessageRowId(wm) if wm else MessageRowId(0),
                policy=policy,
                source_ids=tuple(MessageRowId(r[0]) for r in rows),
                senders=tuple(SenderId(str(r[2])) for r in rows),
                sender_names=tuple(str(r[3] or "") for r in rows),
            )

        return await self._db.read(fn)

    def _record_fts_insert(
        self, conn: Any, record_id: int, text: str, chat_key: ChatKey
    ) -> None:
        """Insert one canonical record FTS token document and its index row
        in the caller's transaction (external-content FTS5)."""
        tokens = bigram_tokenize(text)
        if not tokens:
            return
        joined = " ".join(tokens)
        cur = conn.execute(
            "INSERT INTO record_search_docs(chat_key, record_id, tokens)"
            " VALUES (?, ?, ?)",
            (chat_key, record_id, joined),
        )
        conn.execute(
            "INSERT INTO record_search_fts(rowid, tokens) VALUES (?, ?)",
            (cur.lastrowid, joined),
        )

    async def commit_learner_source(
        self, request: LearnerDraft, *, now: float
    ) -> bool:
        """CAS commit of one learner source range.

        One writer transaction: fence the durable watermark on the exact
        observed snapshot (``request.expected_through_msg_id``, else the
        batch's ``observed_watermark``), verify the source range is beyond
        the watermark and its ``source_hash`` still matches the current
        messages, and verify the run is prepared and its fixed through
        boundary covers the batch.

        A ``success`` outcome atomically inserts/merges the validated
        records (keyed by the deterministic ``content_hash``), writes their
        opaque ``record_sources`` mapping and canonical FTS documents,
        marks the run ``success``, and advances the watermark to the
        batch's last row — a valid EMPTY result (zero records) still
        advances. A ``malformed``/``cancelled`` outcome settles the run
        WITHOUT advancing the watermark or inserting records. Returns False
        when the watermark moved (stale CAS — nothing changes); raises
        RepoError for cross-chat/hash/range/run violations.
        """

        def fn(conn: Any) -> bool:
            row = conn.execute(
                "SELECT watermark_msg_id FROM learner_state"
                " WHERE chat_key = ? AND learner = ?",
                (request.chat_key, request.learner),
            ).fetchone()
            wm = row[0] if row is not None and row[0] is not None else 0
            expected = request.expected_through_msg_id
            if expected is None:
                expected = request.batch.observed_watermark
            if expected is None:
                expected = 0
            if wm != expected:
                return False  # stale CAS: the watermark moved
            batch = request.batch
            if batch.chat_key != request.chat_key:
                raise RepoError("cross-chat learner source batch")
            if batch.learner != request.learner:
                raise RepoError("cross-learner source batch")
            if batch.first_msg_id <= wm:
                raise RepoError(
                    "source range overlaps the watermark:"
                    f" first {batch.first_msg_id} <= watermark {wm}"
                )
            run_params: tuple[Any, ...]
            if request.run_id is None:
                # Compatibility for the original repository-only API.  New
                # pipeline calls always provide run_id and take the exact CAS
                # path below.
                run = conn.execute(
                    "SELECT id, through_msg_id, expires_at FROM learner_runs"
                    " WHERE chat_key = ? AND learner = ? AND state = 'prepared'",
                    (request.chat_key, request.learner),
                ).fetchone()
            else:
                run = conn.execute(
                    "SELECT id, through_msg_id, expires_at FROM learner_runs"
                    " WHERE chat_key = ? AND learner = ? AND id = ?"
                    " AND state = 'prepared'",
                    (request.chat_key, request.learner, request.run_id),
                ).fetchone()
            if run is None:
                raise RepoError(
                    f"no prepared learner run for {request.learner!r}"
                    f" in {request.chat_key!r}"
                )
            run_id, run_through, run_expires = run
            if not math.isfinite(run_expires) or run_expires <= now:
                raise RepoError("learner run lease expired")
            if request.run_id is not None and run_id != request.run_id:
                raise RepoError("learner run ownership mismatch")
            if batch.last_msg_id > run_through:
                raise RepoError(
                    "source batch extends beyond the run's fixed through"
                    f" boundary: last {batch.last_msg_id} > through {run_through}"
                )
            self_filter = " AND is_self = 0" if batch.policy == "nonself" else ""
            rows = conn.execute(
                "SELECT id, text FROM messages WHERE chat_key = ?"
                " AND id >= ? AND id <= ?" + self_filter + " ORDER BY id",
                (request.chat_key, batch.first_msg_id, batch.last_msg_id),
            ).fetchall()
            if batch.source_ids:
                actual_ids = tuple(MessageRowId(r[0]) for r in rows)
                if actual_ids != batch.source_ids:
                    raise RepoError("learner source identity changed since the batch was read")
            texts = tuple(r[1] for r in rows)
            if self._source_hash(texts) != batch.source_hash:
                raise RepoError(
                    "source hash mismatch: messages changed since the batch was read"
                )
            if request.outcome != "success":
                # malformed/cancelled: settle the run WITHOUT advancing the
                # watermark or inserting records.
                conn.execute(
                    "UPDATE learner_runs SET state = ?, error = ?, settled_ts = ?"
                    " WHERE id = ?",
                    (request.outcome, request.error, now, run_id),
                )
                if request.run_id is not None or request.cadence_s is not None:
                    self._update_learner_schedule(
                        conn, request, run_id, now, success=False
                    )
                return True
            added = 0
            merged = 0
            for rec in request.records:
                if rec.chat_key != request.chat_key:
                    raise RepoError("cross-chat learner record")
                if rec.learner != request.learner:
                    raise RepoError("cross-learner record")
                content_hash = self._record_content_hash(rec.payload)
                payload_json = orjson.dumps(rec.payload, default=str).decode("utf-8")
                # Insert-or-merge keyed by the deterministic content_hash.
                # RETURNING id gives the ACTUAL row id in both branches
                # (lastrowid is unreliable for an upsert that updates).
                cur = conn.execute(
                    "INSERT INTO records(chat_key, learner, payload_json,"
                    " weight, uses, created_ts, content_hash,"
                    " source_first_msg_id, source_last_msg_id, retired, producing_run_id)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)"
                    " ON CONFLICT(chat_key, learner, content_hash)"
                    " WHERE content_hash IS NOT NULL DO NOTHING RETURNING id",
                    (
                        rec.chat_key,
                        rec.learner,
                        payload_json,
                        rec.weight,
                        rec.uses,
                        rec.created_ts,
                        content_hash,
                        batch.first_msg_id,
                        batch.last_msg_id,
                        run_id,
                    ),
                )
                row = cur.fetchone()
                if row is not None:
                    record_id = row[0]
                    added += 1
                else:
                    existing = conn.execute(
                        "SELECT id FROM records WHERE chat_key = ? AND learner = ?"
                        " AND content_hash = ?",
                        (rec.chat_key, rec.learner, content_hash),
                    ).fetchone()
                    if existing is None:
                        raise RepoError("adaptive record identity vanished mid-commit")
                    record_id = existing[0]
                    conn.execute(
                        "UPDATE records SET payload_json = ?, weight = ?, producing_run_id = ?"
                        " WHERE id = ?",
                        (payload_json, rec.weight, run_id, record_id),
                    )
                    merged += 1
                # The opaque source mapping: record_id -> this batch's
                # range. Model source references are opaque refs mapped
                # only in the current batch.
                conn.execute(
                    "INSERT INTO record_sources(record_id, chat_key, learner,"
                    " source_first_msg_id, source_last_msg_id, source_hash, producing_run_id)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(record_id) DO UPDATE SET"
                    " source_first_msg_id = excluded.source_first_msg_id,"
                    " source_last_msg_id = excluded.source_last_msg_id,"
                    " source_hash = excluded.source_hash,"
                    " producing_run_id = excluded.producing_run_id",
                    (
                        record_id,
                        rec.chat_key,
                        rec.learner,
                        batch.first_msg_id,
                        batch.last_msg_id,
                        batch.source_hash,
                        run_id,
                    ),
                )
                self._record_fts_insert(
                    conn, record_id, self._record_search_text(rec.payload),
                    request.chat_key,
                )
            conn.execute(
                "UPDATE learner_runs SET state = 'success', source_hash = ?,"
                " records_added = ?, records_merged = ?, settled_ts = ?"
                " WHERE id = ?",
                (batch.source_hash, added, merged, now, run_id),
            )
            conn.execute(
                "INSERT INTO learner_state(chat_key, learner, watermark_msg_id,"
                " observed_watermark_msg_id, last_run_id, updated_ts)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(chat_key, learner) DO UPDATE SET"
                " watermark_msg_id = excluded.watermark_msg_id,"
                " observed_watermark_msg_id = excluded.observed_watermark_msg_id,"
                " last_run_id = excluded.last_run_id,"
                " updated_ts = excluded.updated_ts",
                (
                    request.chat_key,
                    request.learner,
                    batch.last_msg_id,
                    expected,
                    run_id,
                    now,
                ),
            )
            self._update_learner_schedule(conn, request, run_id, now, success=True)
            return True

        return await self._db.write(fn)

    async def list_learner_records(
        self, chat_key: ChatKey, learner: str, *, limit: int = 100
    ) -> list[Record]:
        """Chat+learner-scoped record enumeration excluding legacy (no
        content_hash) and retired records, in deterministic id order."""

        def fn(conn: Any) -> list[Record]:
            rows = conn.execute(
                "SELECT id, chat_key, learner, payload_json, weight, uses,"
                " created_ts, content_hash, source_first_msg_id,"
                " source_last_msg_id, retired, producing_run_id FROM records"
                " WHERE chat_key = ? AND learner = ?"
                " AND content_hash IS NOT NULL AND retired = 0"
                " ORDER BY id LIMIT ?",
                (chat_key, learner, limit),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

        return await self._db.read(fn)

    async def list_adaptive_record_chats_after(
        self, after: ChatKey, *, limit: int = 64
    ) -> list[ChatKey]:
        """Keyset-paged chats containing trusted adaptive records."""
        if limit <= 0:
            raise ValueError("limit must be positive")

        def fn(conn: Any) -> list[ChatKey]:
            rows = conn.execute(
                "SELECT DISTINCT chat_key FROM records"
                " WHERE chat_key IS NOT NULL AND chat_key > ?"
                " AND content_hash IS NOT NULL AND retired = 0"
                " ORDER BY chat_key LIMIT ?",
                (after, limit),
            ).fetchall()
            return [ChatKey(row[0]) for row in rows]

        return await self._db.read(fn)

    async def list_adaptive_records_after(
        self, chat_key: ChatKey, after_id: int, *, limit: int = 128
    ) -> list[Record]:
        """Keyset-paged trusted records across all learner slots."""
        if limit <= 0:
            raise ValueError("limit must be positive")

        def fn(conn: Any) -> list[Record]:
            rows = conn.execute(
                "SELECT id, chat_key, learner, payload_json, weight, uses,"
                " created_ts, content_hash, source_first_msg_id,"
                " source_last_msg_id, retired, producing_run_id FROM records"
                " WHERE chat_key = ? AND id > ? AND content_hash IS NOT NULL"
                " AND retired = 0 ORDER BY id LIMIT ?",
                (chat_key, after_id, limit),
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

        return await self._db.read(fn)

    async def list_records_for_run(
        self, chat_key: ChatKey, learner: str, run_id: int, *, limit: int = 100
    ) -> list[Record]:
        """Return exactly the trusted records produced by one learner run."""
        def fn(conn: Any) -> list[Record]:
            rows = conn.execute(
                "SELECT id, chat_key, learner, payload_json, weight, uses,"
                " created_ts, content_hash, source_first_msg_id,"
                " source_last_msg_id, retired, producing_run_id FROM records"
                " WHERE chat_key = ? AND learner = ? AND producing_run_id = ?"
                " AND content_hash IS NOT NULL AND retired = 0"
                " ORDER BY id LIMIT ?",
                (chat_key, learner, run_id, limit),
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

        return await self._db.read(fn)

    async def latest_record_for_run(
        self, chat_key: ChatKey, learner: str, run_id: int
    ) -> Record | None:
        """Return the newest trusted record from exactly one learner run."""
        def fn(conn: Any) -> Record | None:
            row = conn.execute(
                "SELECT id, chat_key, learner, payload_json, weight, uses,"
                " created_ts, content_hash, source_first_msg_id,"
                " source_last_msg_id, retired, producing_run_id FROM records"
                " WHERE chat_key = ? AND learner = ? AND producing_run_id = ?"
                " AND content_hash IS NOT NULL AND retired = 0"
                " ORDER BY id DESC LIMIT 1",
                (chat_key, learner, run_id),
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

        return await self._db.read(fn)

    async def select_learner_records(
        self, chat_key: ChatKey, learner: str, *, limit: int = 10
    ) -> list[Record]:
        """Chat+learner-scoped record selection for injection: the highest-
        weight, least-used adaptive records first (deterministic order),
        excluding legacy/retired rows."""

        def fn(conn: Any) -> list[Record]:
            rows = conn.execute(
                "SELECT id, chat_key, learner, payload_json, weight, uses,"
                " created_ts, content_hash, source_first_msg_id,"
                " source_last_msg_id, retired, producing_run_id FROM records"
                " WHERE chat_key = ? AND learner = ?"
                " AND content_hash IS NOT NULL AND retired = 0"
                " ORDER BY weight DESC, uses ASC, id ASC LIMIT ?",
                (chat_key, learner, limit),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

        return await self._db.read(fn)

    async def get_learner_records_by_ids(
        self, chat_key: ChatKey, learner: str, record_ids: list[int]
    ) -> list[Record]:
        """Fetch exactly the trusted records named by a vector result."""
        if not record_ids:
            return []

        def fn(conn: Any) -> list[Record]:
            placeholders = ",".join("?" for _ in record_ids)
            rows = conn.execute(
                "SELECT id, chat_key, learner, payload_json, weight, uses,"
                " created_ts, content_hash, source_first_msg_id,"
                " source_last_msg_id, retired, producing_run_id FROM records"
                " WHERE chat_key = ? AND learner = ?"
                " AND content_hash IS NOT NULL AND retired = 0"
                " AND id IN (" + placeholders + ") ORDER BY id",
                (chat_key, learner, *record_ids),
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

        return await self._db.read(fn)

    async def record_exposure(
        self,
        chat_key: ChatKey,
        learner: str,
        record_id: int,
        run_id: int,
        *,
        now: float,
        dispatch_id: int | None = None,
        slot: str = "",
    ) -> bool:
        """Atomic idempotent exposure: inserts one ``(record_id, run_id)``
        exposure row exactly once. True on the first exposure, False on a
        duplicate (idempotent — nothing changes). Raises RepoError for an
        unknown/cross-chat/legacy/retired record."""

        def fn(conn: Any) -> bool:
            row = conn.execute(
                "SELECT chat_key, learner, content_hash, retired FROM records"
                " WHERE id = ?",
                (record_id,),
            ).fetchone()
            if row is None or row[0] != chat_key or row[1] != learner:
                raise RepoError(
                    f"record {record_id} does not belong to learner"
                    f" {learner!r} in chat {chat_key!r}"
                )
            if row[2] is None or row[3]:
                raise RepoError(
                    f"record {record_id} is legacy or retired — not exposable"
                )
            producing = conn.execute(
                "SELECT chat_key, learner FROM learner_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if producing is None or producing[0] != chat_key or producing[1] != learner:
                raise RepoError("exposure run does not own the selected record")
            cur = conn.execute(
                "INSERT INTO record_exposures(chat_key, learner, record_id,"
                " run_id, exposed_ts, dispatch_id, slot) VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(record_id, run_id) DO NOTHING",
                (chat_key, learner, record_id, run_id, now, dispatch_id, slot),
            )
            return cur.rowcount == 1

        return await self._db.write(fn)

    async def increment_record_uses(
        self, chat_key: ChatKey, learner: str, record_id: int
    ) -> bool:
        """Atomically bump the record's ``uses`` counter. False when the
        record is unknown, cross-chat, legacy, or retired."""

        def fn(conn: Any) -> bool:
            cur = conn.execute(
                "UPDATE records SET uses = uses + 1"
                " WHERE id = ? AND chat_key = ? AND learner = ?"
                " AND content_hash IS NOT NULL AND retired = 0",
                (record_id, chat_key, learner),
            )
            return cur.rowcount == 1

        return await self._db.write(fn)

    async def apply_record_feedback(
        self,
        chat_key: ChatKey,
        learner: str,
        record_id: int,
        effect: float,
        *,
        now: float,
        effect_run_id: int | None = None,
    ) -> float | None:
        """Bounded effect feedback with a code-owned reweight.

        ``effect`` must be finite and within [-1, 1] (a ValueError
        otherwise). The reweight is computed HERE — ``new_weight =
        clamp(weight * (1 + effect), 0.1, 5.0)`` — never by the caller, and
        the record's weight is updated atomically with the feedback row.
        Returns the record's NEW weight, or None when the record is
        unknown/cross-chat/legacy/retired."""

        def fn(conn: Any) -> float | None:
            if isinstance(effect, bool) or not isinstance(effect, (int, float)):
                raise ValueError(f"effect must be a number, got {effect!r}")
            if not math.isfinite(effect) or not (-1.0 <= effect <= 1.0):
                raise ValueError(f"effect must be finite and within [-1, 1], got {effect!r}")
            row = conn.execute(
                "SELECT weight FROM records"
                " WHERE id = ? AND chat_key = ? AND learner = ?"
                " AND content_hash IS NOT NULL AND retired = 0",
                (record_id, chat_key, learner),
            ).fetchone()
            if row is None:
                return None
            weight = row[0]
            if effect_run_id is not None and conn.execute(
                "SELECT 1 FROM record_feedback WHERE record_id = ?"
                " AND effect_run_id = ?",
                (record_id, effect_run_id),
            ).fetchone() is not None:
                return weight
            new_weight = min(5.0, max(0.1, weight * (1.0 + effect)))
            conn.execute(
                "UPDATE records SET weight = ? WHERE id = ?",
                (new_weight, record_id),
            )
            conn.execute(
                "INSERT INTO record_feedback(chat_key, learner, record_id,"
                " effect, reweight, feedback_ts, effect_run_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (chat_key, learner, record_id, effect, new_weight, now, effect_run_id),
            )
            return new_weight

        return await self._db.write(fn)

    async def query_records(
        self, chat_key: ChatKey, learner: str, query: str, *, limit: int = 10
    ) -> list[RecordHit]:
        """Bounded chat+learner-safe record FTS query (lexical-first recall
        over the canonical record token documents).

        Ranking uses a CHAT-LOCAL BM25 (the same statistics discipline as
        ``query_memory``): document-frequency and average-document-length
        statistics are computed ONLY from this chat+learner's own canonical
        token documents, so one chat's data never changes another chat's
        scores or order. Ties are broken deterministically by record_id
        ASC. The FTS MATCH stays parameterized (query-safe) and the result
        is bounded to ``limit``.
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

        def fn(conn: Any) -> list[RecordHit]:
            tokens = bigram_tokenize(query)
            if not tokens:
                return []
            match = " OR ".join(
                '"' + t.replace('"', '""') + '"' for t in tokens
            )
            rows = conn.execute(
                "SELECT record_search_fts.rowid, d.record_id, d.tokens,"
                " r.payload_json"
                " FROM record_search_fts"
                " JOIN record_search_docs d ON d.id = record_search_fts.rowid"
                " JOIN records r ON r.id = d.record_id"
                " WHERE record_search_fts MATCH ? AND d.chat_key = ?"
                " AND r.learner = ? AND r.content_hash IS NOT NULL"
                " AND r.retired = 0",
                (match, chat_key, learner),
            ).fetchall()
            if not rows:
                return []
            docs = [
                {
                    "memory_id": r[1],
                    "text": self._record_search_text(orjson.loads(r[3])),
                    "first": None,
                    "last": None,
                    "tokens": r[2].split() if r[2] else [],
                }
                for r in rows
            ]
            scored = self._chat_local_bm25(docs, tokens)
            scored.sort(key=lambda s: (-s[1], s[0]))
            out: list[RecordHit] = []
            for record_id, score, _first, _last, text in scored[:limit]:
                out.append(
                    RecordHit(
                        chat_key=chat_key,
                        learner=learner,
                        record_id=record_id,
                        text=text,
                        score=score,
                    )
                )
            return out

        return await self._db.read(fn)

    @staticmethod
    def _update_learner_schedule(
        conn: Any, request: LearnerDraft, run_id: int, now: float, *, success: bool
    ) -> None:
        """Materialize cadence/backoff with the learner settlement."""
        old = conn.execute(
            "SELECT watermark_msg_id, observed_watermark_msg_id, cadence_s,"
            " failure_streak FROM learner_state WHERE chat_key = ? AND learner = ?",
            (request.chat_key, request.learner),
        ).fetchone()
        cadence = request.cadence_s
        if cadence is None and old is not None and old[2] is not None:
            cadence = float(old[2])
        if cadence is None:
            cadence = 3600.0
        previous_failures = int(old[3] or 0) if old is not None else 0
        failures = 0 if success else min(previous_failures + 1, 31)
        delay = cadence if success else min(86400.0, cadence * (2 ** (failures - 1)))
        watermark = old[0] if old is not None else None
        observed = old[1] if old is not None else None
        if success:
            watermark = request.batch.last_msg_id
            observed = request.batch.observed_watermark
        conn.execute(
            "INSERT INTO learner_state(chat_key, learner, watermark_msg_id,"
            " observed_watermark_msg_id, last_run_id, updated_ts, cadence_s,"
            " next_due_ts, failure_streak) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(chat_key, learner) DO UPDATE SET"
            " watermark_msg_id = excluded.watermark_msg_id,"
            " observed_watermark_msg_id = excluded.observed_watermark_msg_id,"
            " last_run_id = excluded.last_run_id, updated_ts = excluded.updated_ts,"
            " cadence_s = excluded.cadence_s, next_due_ts = excluded.next_due_ts,"
            " failure_streak = excluded.failure_streak",
            (request.chat_key, request.learner, watermark, observed, run_id,
             now, cadence, now + delay, failures),
        )

    async def get_learner_state(
        self, chat_key: ChatKey, learner: str
    ) -> LearnerState | None:
        """The durable per-(chat, learner) state; None when unrecorded."""

        def fn(conn: Any) -> LearnerState | None:
            row = conn.execute(
                "SELECT chat_key, learner, watermark_msg_id,"
                " observed_watermark_msg_id, last_run_id, updated_ts, cadence_s,"
                " next_due_ts, failure_streak"
                " FROM learner_state WHERE chat_key = ? AND learner = ?",
                (chat_key, learner),
            ).fetchone()
            if row is None:
                return None
            return LearnerState(
                chat_key=ChatKey(row[0]),
                learner=row[1],
                watermark_msg_id=MessageRowId(row[2])
                if row[2] is not None
                else None,
                observed_watermark_msg_id=MessageRowId(row[3])
                if row[3] is not None
                else None,
                last_run_id=row[4],
                updated_ts=row[5],
                cadence_s=row[6],
                next_due_ts=row[7],
                failure_streak=row[8],
            )

        return await self._db.read(fn)

    async def list_learner_pending_chats(
        self, learner: str, *, policy: str = "nonself", now: float | None = None
    ) -> list[ChatKey]:
        """Every chat with pending learner work: at least one message beyond
        the learner's durable watermark (NULL watermark counts as 0).
        Deterministic order; the scheduler wakes exactly these chats after
        a restart."""

        if policy not in ("nonself", "all"):
            raise ValueError("policy must be 'nonself' or 'all'")

        def fn(conn: Any) -> list[ChatKey]:
            self_filter = " AND m.is_self = 0" if policy == "nonself" else ""
            due_filter = ""
            params: list[Any] = [learner]
            if now is not None:
                due_filter = (
                    " AND (s.next_due_ts IS NULL OR s.next_due_ts <= ?"
                    " OR s.updated_ts > ? )"
                )
                params.append(now)
                params.append(now)
            rows = conn.execute(
                "SELECT DISTINCT m.chat_key FROM messages m"
                " LEFT JOIN learner_state s ON s.chat_key = m.chat_key"
                "  AND s.learner = ?"
                " WHERE m.id > COALESCE(s.watermark_msg_id, 0)"
                + self_filter + due_filter +
                " ORDER BY m.chat_key",
                tuple(params),
            ).fetchall()
            return [ChatKey(r[0]) for r in rows]

        return await self._db.read(fn)

    async def list_learner_runs(
        self, chat_key: ChatKey, learner: str, *, limit: int = 20
    ) -> list[LearnerRun]:
        """The bounded recent run ledger for one chat+learner, newest
        first."""

        def fn(conn: Any) -> list[LearnerRun]:
            rows = conn.execute(
                "SELECT id, chat_key, learner, started_ts, expires_at,"
                " start_msg_id, through_msg_id, source_hash, state,"
                " records_added, records_merged, error, settled_ts"
                " FROM learner_runs WHERE chat_key = ? AND learner = ?"
                " ORDER BY id DESC LIMIT ?",
                (chat_key, learner, limit),
            ).fetchall()
            return [
                LearnerRun(
                    id=r[0],
                    chat_key=ChatKey(r[1]),
                    learner=r[2],
                    started_ts=r[3],
                    expires_at=r[4],
                    start_msg_id=MessageRowId(r[5]),
                    through_msg_id=MessageRowId(r[6]),
                    source_hash=r[7],
                    state=r[8],
                    records_added=r[9],
                    records_merged=r[10],
                    error=r[11],
                    settled_ts=r[12],
                )
                for r in rows
            ]

        return await self._db.read(fn)

    # ── Phase 6 P6.5 media catalog (MediaRepository seam) ────────────────────
    # The durable chat-scoped media catalog: pending candidates, capacity-safe
    # transactional approval/eviction, idempotent rejection/revocation,
    # deterministic cooldown-aware selection, atomic uses, and chat-scoped
    # listing. The catalog key is OPAQUE — validation (``emoji.validate_candidate``)
    # rejects local paths, URLs, data/base64 payloads, and raw platform media
    # references as catalog keys. Existing global emoji rows remain
    # legacy/untrusted and are never read here. No provider/network calls, no
    # file fetch, no outbox/send, no plugin load in any transaction.

    async def submit_media_candidate(
        self, candidate: MediaAssetCandidate, *, now: float
    ) -> int:
        """Submit one chat-scoped candidate (idempotent per (chat, kind,
        sha256)).

        Returns the row id — the existing row's id when a row with the same
        (chat, kind, sha256) already exists (its status is never reset: a
        rejected/revoked row stays rejected/revoked). Raises ValueError for
        an invalid catalog key/sha256/mime and RepoError for an unknown
        chat.
        """
        validate_candidate(candidate)
        if not math.isfinite(now):
            raise ValueError("now must be finite")

        def fn(conn: Any) -> int:
            row = conn.execute(
                "SELECT chat_key FROM chats WHERE chat_key = ?",
                (candidate.chat_key,),
            ).fetchone()
            if row is None:
                raise RepoError(f"unknown chat: {candidate.chat_key!r}")
            cur = conn.execute(
                "INSERT INTO media_assets(chat_key, kind, cache_key, sha256,"
                " mime, width, height, description, source_message_id,"
                " source_sender_id, source_sender_name, source_ts, created_ts)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(chat_key, kind, sha256) DO NOTHING",
                (
                    candidate.chat_key,
                    candidate.kind,
                    candidate.cache_key,
                    candidate.sha256,
                    candidate.mime,
                    candidate.width,
                    candidate.height,
                    candidate.description,
                    candidate.source_message_id,
                    candidate.source_sender_id,
                    candidate.source_sender_name,
                    candidate.source_ts,
                    now,
                ),
            )
            if cur.rowcount == 1:
                return cur.lastrowid
            row = conn.execute(
                "SELECT id FROM media_assets WHERE chat_key = ? AND kind = ?"
                " AND sha256 = ?",
                (candidate.chat_key, candidate.kind, candidate.sha256),
            ).fetchone()
            return row[0]

        return await self._db.write(fn)

    async def get_media_candidate(
        self, chat_key: ChatKey, candidate_id: int
    ) -> MediaAssetCandidate | None:
        """One PENDING candidate of the chat, or None when unknown/cross-chat/
        no longer pending."""

        def fn(conn: Any) -> MediaAssetCandidate | None:
            row = conn.execute(
                "SELECT id, chat_key, kind, cache_key, sha256, mime, width,"
                " height, description, source_message_id, source_sender_id,"
                " source_sender_name, source_ts, created_ts FROM media_assets"
                " WHERE chat_key = ? AND id = ? AND safety_status = 'pending'",
                (chat_key, candidate_id),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_media_candidate(row)

        return await self._db.read(fn)

    async def list_media_candidates(
        self, chat_key: ChatKey, *, kind: str | None = None, limit: int = 100
    ) -> list[MediaAssetCandidate]:
        """The chat's PENDING candidates in deterministic id order."""
        if kind is not None and kind not in _MEDIA_KINDS:
            raise ValueError(f"invalid media kind: {kind!r}")
        if isinstance(limit, bool) or limit <= 0:
            raise RepoError("limit must be a positive integer")

        def fn(conn: Any) -> list[MediaAssetCandidate]:
            sql = (
                "SELECT id, chat_key, kind, cache_key, sha256, mime, width,"
                " height, description, source_message_id, source_sender_id,"
                " source_sender_name, source_ts, created_ts FROM media_assets"
                " WHERE chat_key = ? AND safety_status = 'pending'"
            )
            params: list[Any] = [chat_key]
            if kind is not None:
                sql += " AND kind = ?"
                params.append(kind)
            sql += " ORDER BY id LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_media_candidate(r) for r in rows]

        return await self._db.read(fn)

    async def list_media_candidates_after(
        self,
        chat_key: ChatKey,
        after_id: int,
        *,
        kind: str | None = None,
        limit: int = 100,
    ) -> list[MediaAssetCandidate]:
        """Keyset page of pending candidates, ordered by durable id."""
        if kind is not None and kind not in _MEDIA_KINDS:
            raise ValueError(f"invalid media kind: {kind!r}")
        if isinstance(after_id, bool) or not isinstance(after_id, int) or after_id < 0:
            raise RepoError("after_id must be a nonnegative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise RepoError("limit must be a positive integer")

        def fn(conn: Any) -> list[MediaAssetCandidate]:
            sql = (
                "SELECT id, chat_key, kind, cache_key, sha256, mime, width,"
                " height, description, source_message_id, source_sender_id,"
                " source_sender_name, source_ts, created_ts FROM media_assets"
                " WHERE chat_key = ? AND safety_status = 'pending' AND id > ?"
            )
            params: list[Any] = [chat_key, after_id]
            if kind is not None:
                sql += " AND kind = ?"
                params.append(kind)
            sql += " ORDER BY id LIMIT ?"
            params.append(limit)
            return [
                self._row_to_media_candidate(row)
                for row in conn.execute(sql, params).fetchall()
            ]

        return await self._db.read(fn)

    async def approve_media_candidate(
        self, chat_key: ChatKey, candidate_id: int, *, capacity: int, now: float
    ) -> MediaAsset | None:
        """Capacity-safe transactional approval.

        One writer transaction: the candidate must be a PENDING row of the
        chat; when the (chat, kind) approved count is at ``capacity``, the
        least-recently-used approved rows (never-used first, then oldest
        ``last_used_ts``, then id) are evicted in the SAME transaction to
        make room; the candidate is transitioned to ``approved`` with
        ``safety_version + 1`` and ``approved_ts = now``. Returns the
        approved MediaAsset; None when the candidate is unknown, cross-chat,
        or already rejected/revoked. An already-approved row returns the
        existing approved asset (idempotent).
        """
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"capacity must be a positive integer, got {capacity!r}")
        if not math.isfinite(now):
            raise ValueError("now must be finite")

        def fn(conn: Any) -> MediaAsset | None:
            row = conn.execute(
                "SELECT id, chat_key, kind, safety_status FROM media_assets"
                " WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None or row[1] != chat_key:
                return None
            asset_id, _chat, kind, status = row
            if status == "approved":
                return self._get_media_asset_row(conn, chat_key, asset_id)
            if status != "pending":
                return None  # rejected/revoked: fail closed
            # Source deletion is a durable safety fence.  This predicate is
            # evaluated by the same writer transaction as approval, so a
            # recall racing the harvest cannot approve a recalled source.
            source_deleted = conn.execute(
                "SELECT 1 FROM messages WHERE id = ? AND chat_key = ?"
                " AND deleted = 1",
                (
                    conn.execute(
                        "SELECT source_message_id FROM media_assets WHERE id = ?",
                        (asset_id,),
                    ).fetchone()[0],
                    chat_key,
                ),
            ).fetchone()
            if source_deleted is not None:
                conn.execute(
                    "UPDATE media_assets SET safety_status = 'rejected',"
                    " safety_version = safety_version + 1"
                    " WHERE id = ? AND safety_status = 'pending'",
                    (asset_id,),
                )
                return None
            # Capacity-safe eviction: when the approved count is at capacity,
            # evict the least-recently-used approved rows (never-used first,
            # then oldest last_used_ts, then id) in the SAME transaction.
            count = conn.execute(
                "SELECT COUNT(*) FROM media_assets"
                " WHERE chat_key = ? AND kind = ? AND safety_status = 'approved'",
                (chat_key, kind),
            ).fetchone()[0]
            if count >= capacity:
                excess = count - capacity + 1
                conn.execute(
                    "DELETE FROM media_assets WHERE id IN ("
                    " SELECT id FROM media_assets"
                    " WHERE chat_key = ? AND kind = ? AND safety_status = 'approved'"
                    " ORDER BY (last_used_ts IS NULL) DESC, last_used_ts ASC, id ASC"
                    " LIMIT ?)",
                    (chat_key, kind, excess),
                )
            conn.execute(
                "UPDATE media_assets SET safety_status = 'approved',"
                " safety_version = safety_version + 1, approved_ts = ?,"
                " revoked_ts = NULL WHERE id = ? AND safety_status = 'pending'",
                (now, asset_id),
            )
            return self._get_media_asset_row(conn, chat_key, asset_id)

        return await self._db.write(fn)

    async def reject_media_candidate(
        self, chat_key: ChatKey, candidate_id: int
    ) -> bool:
        """Idempotent terminal rejection of a PENDING candidate. False when
        the candidate is unknown, cross-chat, or not pending (already
        approved/rejected/revoked)."""

        def fn(conn: Any) -> bool:
            cur = conn.execute(
                "UPDATE media_assets SET safety_status = 'rejected',"
                " safety_version = safety_version + 1"
                " WHERE id = ? AND chat_key = ? AND safety_status = 'pending'",
                (candidate_id, chat_key),
            )
            return cur.rowcount == 1

        return await self._db.write(fn)

    async def revoke_media_asset(
        self, chat_key: ChatKey, asset_id: int, *, now: float
    ) -> bool:
        """Idempotent terminal revocation of an APPROVED asset. False when
        the asset is unknown, cross-chat, or not approved."""
        if not math.isfinite(now):
            raise ValueError("now must be finite")

        def fn(conn: Any) -> bool:
            row = conn.execute(
                "SELECT cache_key FROM media_assets WHERE id = ? AND chat_key = ?",
                (asset_id, chat_key),
            ).fetchone()
            cur = conn.execute(
                "UPDATE media_assets SET safety_status = 'revoked',"
                " safety_version = safety_version + 1, revoked_ts = ?"
                " WHERE id = ? AND chat_key = ? AND safety_status = 'approved'",
                (now, asset_id, chat_key),
            )
            if row is not None:
                self._drop_media_outbox_for_keys(conn, chat_key, (row[0],))
            return cur.rowcount == 1

        return await self._db.write(fn)

    async def select_media_assets(
        self,
        chat_key: ChatKey,
        kind: str,
        *,
        limit: int = 1,
        cooldown_s: float = 0.0,
        now: float,
    ) -> list[MediaAsset]:
        """Deterministic cooldown-aware selection: APPROVED rows only, in
        deterministic order (least-used first, then least-recently-used,
        then id), excluding rows used within ``cooldown_s`` of ``now``."""
        if kind not in _MEDIA_KINDS:
            raise ValueError(f"invalid media kind: {kind!r}")
        if isinstance(limit, bool) or limit <= 0:
            raise RepoError("limit must be a positive integer")
        if isinstance(cooldown_s, bool) or not isinstance(cooldown_s, (int, float)):
            raise ValueError(f"cooldown_s must be a number, got {cooldown_s!r}")
        if not math.isfinite(cooldown_s) or cooldown_s < 0:
            raise ValueError("cooldown_s must be finite and nonnegative")
        if not math.isfinite(now):
            raise ValueError("now must be finite")

        def fn(conn: Any) -> list[MediaAsset]:
            rows = conn.execute(
                "SELECT id, chat_key, kind, cache_key, sha256, mime, width,"
                " height, description, source_message_id, source_sender_id,"
                " source_sender_name, safety_status, safety_version,"
                " approved_ts, revoked_ts, uses, last_used_ts, created_ts"
                " FROM media_assets"
                " WHERE chat_key = ? AND kind = ? AND safety_status = 'approved'"
                " AND (last_used_ts IS NULL OR last_used_ts + ? <= ?)"
                " ORDER BY uses ASC, last_used_ts ASC, id ASC LIMIT ?",
                (chat_key, kind, cooldown_s, now, limit),
            ).fetchall()
            return [self._row_to_media_asset(r) for r in rows]

        return await self._db.read(fn)

    async def use_media_asset(
        self, chat_key: ChatKey, asset_id: int, *, now: float
    ) -> bool:
        """Atomic idempotent use: bump ``uses`` and set ``last_used_ts`` on
        an APPROVED row. False when the asset is unknown, cross-chat, or not
        approved. Each call is one real use (selection's cooldown reads the
        updated ``last_used_ts``)."""
        if not math.isfinite(now):
            raise ValueError("now must be finite")

        def fn(conn: Any) -> bool:
            cur = conn.execute(
                "UPDATE media_assets SET uses = uses + 1, last_used_ts = ?"
                " WHERE id = ? AND chat_key = ? AND safety_status = 'approved'",
                (now, asset_id, chat_key),
            )
            return cur.rowcount == 1

        return await self._db.write(fn)

    async def list_media_assets(
        self, chat_key: ChatKey, *, kind: str | None = None, limit: int = 100
    ) -> list[MediaAsset]:
        """The chat's asset rows (all statuses) in deterministic id order."""
        if kind is not None and kind not in _MEDIA_KINDS:
            raise ValueError(f"invalid media kind: {kind!r}")
        if isinstance(limit, bool) or limit <= 0:
            raise RepoError("limit must be a positive integer")

        def fn(conn: Any) -> list[MediaAsset]:
            sql = (
                "SELECT id, chat_key, kind, cache_key, sha256, mime, width,"
                " height, description, source_message_id, source_sender_id,"
                " source_sender_name, safety_status, safety_version,"
                " approved_ts, revoked_ts, uses, last_used_ts, created_ts"
                " FROM media_assets WHERE chat_key = ?"
            )
            params: list[Any] = [chat_key]
            if kind is not None:
                sql += " AND kind = ?"
                params.append(kind)
            sql += " ORDER BY id LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_media_asset(r) for r in rows]

        return await self._db.read(fn)

    async def list_media_assets_after(
        self,
        chat_key: ChatKey,
        after_id: int,
        *,
        kind: str | None = None,
        limit: int = 100,
    ) -> list[MediaAsset]:
        """Keyset page of all catalog statuses, ordered by durable id."""
        if kind is not None and kind not in _MEDIA_KINDS:
            raise ValueError(f"invalid media kind: {kind!r}")
        if isinstance(after_id, bool) or not isinstance(after_id, int) or after_id < 0:
            raise RepoError("after_id must be a nonnegative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise RepoError("limit must be a positive integer")

        def fn(conn: Any) -> list[MediaAsset]:
            sql = (
                "SELECT id, chat_key, kind, cache_key, sha256, mime, width,"
                " height, description, source_message_id, source_sender_id,"
                " source_sender_name, safety_status, safety_version,"
                " approved_ts, revoked_ts, uses, last_used_ts, created_ts"
                " FROM media_assets WHERE chat_key = ? AND id > ?"
            )
            params: list[Any] = [chat_key, after_id]
            if kind is not None:
                sql += " AND kind = ?"
                params.append(kind)
            sql += " ORDER BY id LIMIT ?"
            params.append(limit)
            return [
                self._row_to_media_asset(row)
                for row in conn.execute(sql, params).fetchall()
            ]

        return await self._db.read(fn)

    async def authorize_media_send(
        self, chat_key: ChatKey, delivery_key: str, cache_keys: tuple[str, ...]
    ) -> bool:
        """Atomically fence a media outbox send against current catalog state.

        Revocation drops pending and in-flight rows, but this second check is
        required for the race where an attempt CAS happens immediately before
        the send-time resolver runs.  A denied row is terminally dropped, so a
        queued or in-flight recalled asset cannot be retried.
        """
        if not cache_keys or not delivery_key:
            return False

        def fn(conn: Any) -> bool:
            row = conn.execute(
                "SELECT id, state, segments_json FROM outbox"
                " WHERE chat_key = ? AND idem_key = ?",
                (chat_key, delivery_key),
            ).fetchone()
            if row is None or row[1] not in ("pending", "in_flight"):
                return False
            try:
                segments = orjson.loads(row[2])
            except (orjson.JSONDecodeError, TypeError):
                segments = []
            stored_keys = tuple(
                seg.get("data", {}).get("media", {}).get("key")
                for seg in segments
                if isinstance(seg, dict)
                and isinstance(seg.get("data"), dict)
                and isinstance(seg["data"].get("media"), dict)
                and isinstance(seg["data"]["media"].get("key"), str)
            )
            if not stored_keys or set(stored_keys) != set(cache_keys):
                conn.execute(
                    "UPDATE outbox SET state = 'dropped' WHERE id = ?"
                    " AND state IN ('pending', 'in_flight')",
                    (row[0],),
                )
                return False
            placeholders = ",".join("?" for _ in cache_keys)
            approved = conn.execute(
                "SELECT COUNT(*) FROM media_assets WHERE chat_key = ?"
                f" AND cache_key IN ({placeholders}) AND safety_status = 'approved'",
                (chat_key, *cache_keys),
            ).fetchone()[0]
            if approved != len(set(cache_keys)):
                conn.execute(
                    "UPDATE outbox SET state = 'dropped' WHERE id = ?"
                    " AND state IN ('pending', 'in_flight')",
                    (row[0],),
                )
                return False
            return True

        return await self._db.write(fn)

    def _drop_media_outbox_for_keys(
        self, conn: Any, chat_key: ChatKey, cache_keys: tuple[str, ...]
    ) -> None:
        """Drop queued/in-flight media rows matching revoked opaque keys."""
        rows = conn.execute(
            "SELECT id, segments_json FROM outbox WHERE chat_key = ?"
            " AND state IN ('pending', 'in_flight')",
            (chat_key,),
        ).fetchall()
        revoked = set(cache_keys)
        for item_id, raw in rows:
            try:
                segments = orjson.loads(raw)
            except (orjson.JSONDecodeError, TypeError):
                continue
            keys = {
                seg.get("data", {}).get("media", {}).get("key")
                for seg in segments
                if isinstance(seg, dict)
                and isinstance(seg.get("data"), dict)
                and isinstance(seg["data"].get("media"), dict)
            }
            if keys & revoked:
                conn.execute(
                    "UPDATE outbox SET state = 'dropped' WHERE id = ?"
                    " AND state IN ('pending', 'in_flight')",
                    (item_id,),
                )

    @staticmethod
    def _row_to_media_candidate(row: Any) -> MediaAssetCandidate:
        """Hydrate one pending ``media_assets`` row into a candidate."""
        return MediaAssetCandidate(
            id=row[0],
            chat_key=ChatKey(row[1]),
            kind=row[2],
            cache_key=row[3],
            sha256=row[4],
            mime=row[5],
            width=row[6],
            height=row[7],
            description=row[8],
            source_message_id=MessageRowId(row[9]) if row[9] is not None else None,
            source_sender_id=SenderId(row[10]) if row[10] is not None else None,
            source_sender_name=row[11],
            source_ts=row[12],
            created_ts=row[13],
        )

    @staticmethod
    def _row_to_media_asset(row: Any) -> MediaAsset:
        """Hydrate one ``media_assets`` row into a ``MediaAsset``."""
        return MediaAsset(
            id=row[0],
            chat_key=ChatKey(row[1]),
            kind=row[2],
            cache_key=row[3],
            sha256=row[4],
            mime=row[5],
            width=row[6],
            height=row[7],
            description=row[8],
            source_message_id=MessageRowId(row[9]) if row[9] is not None else None,
            source_sender_id=SenderId(row[10]) if row[10] is not None else None,
            source_sender_name=row[11],
            safety_status=row[12],
            safety_version=row[13],
            approved_ts=row[14],
            revoked_ts=row[15],
            uses=row[16],
            last_used_ts=row[17],
            created_ts=row[18],
        )

    def _get_media_asset_row(
        self, conn: Any, chat_key: ChatKey, asset_id: int
    ) -> MediaAsset | None:
        """One ``media_assets`` row of the chat, in the caller's transaction."""
        row = conn.execute(
            "SELECT id, chat_key, kind, cache_key, sha256, mime, width,"
            " height, description, source_message_id, source_sender_id,"
            " source_sender_name, safety_status, safety_version,"
            " approved_ts, revoked_ts, uses, last_used_ts, created_ts"
            " FROM media_assets WHERE chat_key = ? AND id = ?",
            (chat_key, asset_id),
        ).fetchone()
        return self._row_to_media_asset(row) if row is not None else None

    # ── chat controls (Phase 6 P6.6b) ───────────────────────────────────────

    async def apply_chat_control(self, control: ChatControl) -> bool:
        """Idempotently apply one durable chat control.

        - The target chat must be a KNOWN chat on the SAME account
          (platform + self_id) as the source chat; otherwise False
          (rejected — never a cross-account write).
        - A ``focus`` control transactionally advances the account's current
          focus projection. Historical control rows are never deleted, so a
          replacement cannot erase an idempotency key.
        - A duplicate ``(dispatch_id, intent_seq)`` is a no-op (False) — a
          retried settlement of the same dispatch never double-applies.
        """
        if control.kind not in ("focus", "notify"):
            raise ValueError(f"invalid chat control kind: {control.kind!r}")
        if not math.isfinite(control.ttl_until) or not math.isfinite(
            control.created_ts
        ):
            raise ValueError("chat control timestamps must be finite")
        if control.ttl_until <= control.created_ts:
            raise ValueError("ttl_until must be after created_ts")
        ttl = control.ttl_until - control.created_ts
        lower = 30.0 if control.kind == "focus" else 1.0
        if ttl < lower or ttl > 3600.0:
            raise ValueError("chat control TTL is outside its bounded range")
        if control.kind == "focus" and control.text is not None:
            raise ValueError("focus controls cannot carry text")
        if control.kind == "notify" and (
            not isinstance(control.text, str) or not control.text.strip()
            or len(control.text) > 2000
        ):
            raise ValueError("notify controls require bounded text")

        def fn(conn: Any) -> bool:
            return self._apply_chat_control_conn(conn, control)

        return await self._db.write(fn)

    def _apply_chat_control_conn(self, conn: Any, control: ChatControl) -> bool:
        """Apply one already-validated control inside a caller transaction.

        Authorization and the idempotency check intentionally happen before
        the account focus projection changes.  A rejected or duplicate control
        is a no-op and cannot resurrect an old focus.
        """
        target = conn.execute(
            "SELECT platform, self_id FROM chats WHERE chat_key = ?",
            (control.chat_key,),
        ).fetchone()
        if target is None:
            return False
        source = conn.execute(
            "SELECT platform, self_id FROM chats WHERE chat_key = ?",
            (control.source_chat_key,),
        ).fetchone()
        if source is None or (target[0], target[1]) != (source[0], source[1]):
            return False
        existing = conn.execute(
            "SELECT 1 FROM chat_controls WHERE dispatch_id = ? AND intent_seq = ?",
            (control.dispatch_id, control.intent_seq),
        ).fetchone()
        if existing is not None:
            return False
        cur = conn.execute(
            "INSERT INTO chat_controls(chat_key, kind, ttl_until,"
            " created_ts, dispatch_id, intent_seq, source_chat_key, text)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(dispatch_id, intent_seq) DO NOTHING",
            (
                control.chat_key,
                control.kind,
                control.ttl_until,
                control.created_ts,
                control.dispatch_id,
                control.intent_seq,
                control.source_chat_key,
                control.text,
            ),
        )
        if cur.rowcount != 1:
            return False
        if control.kind == "focus":
            conn.execute(
                "INSERT INTO chat_focus_current(platform, self_id, control_id)"
                " VALUES (?, ?, ?) ON CONFLICT(platform, self_id) DO UPDATE"
                " SET control_id = excluded.control_id",
                (target[0], target[1], cur.lastrowid),
            )
        return True

    async def list_active_controls(
        self, chat_key: ChatKey, *, now: float
    ) -> list[ChatControl]:
        """The chat's ACTIVE (``ttl_until > now``) controls, in
        deterministic id order."""
        if not math.isfinite(now):
            raise ValueError("now must be finite")

        def fn(conn: Any) -> list[ChatControl]:
            rows = conn.execute(
                "SELECT id, chat_key, kind, ttl_until, created_ts,"
                " dispatch_id, intent_seq, source_chat_key, text"
                " FROM chat_controls AS cc WHERE cc.chat_key = ? AND cc.ttl_until > ?"
                " AND (cc.kind = 'notify' OR (cc.kind = 'focus' AND cc.id = ("
                "   SELECT control_id FROM chat_focus_current AS fc"
                "   JOIN chats AS target ON target.platform = fc.platform"
                "    AND target.self_id = fc.self_id"
                "   WHERE target.chat_key = cc.chat_key"
                " )))"
                " ORDER BY id",
                (chat_key, now),
            ).fetchall()
            return [self._row_to_chat_control(r) for r in rows]

        return await self._db.read(fn)

    @staticmethod
    def _row_to_chat_control(row: Any) -> ChatControl:
        """Hydrate one ``chat_controls`` row into a ``ChatControl``."""
        return ChatControl(
            chat_key=ChatKey(row[1]),
            kind=row[2],
            ttl_until=row[3],
            created_ts=row[4],
            dispatch_id=row[5],
            intent_seq=row[6],
            source_chat_key=ChatKey(row[7]),
            text=row[8],
        )

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def close(self) -> None:
        await self._db.close()
