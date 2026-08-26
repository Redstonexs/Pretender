"""Shared helpers for the durable I/O lane tests.

Tests run via asyncio.run() so the test extra stays at just pytest.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pretender.db import Database
from pretender.repo import SqliteRepository
from pretender.types import (
    ChatIdentity,
    ChatKey,
    ClaimBusy,
    ClaimGrant,
    CommitSeq,
    CorpusMarker,
    CycleClaim,
    CycleFinish,
    CycleId,
    DispatchGrant,
    DispatchId,
    DispatchRequest,
    DispatchSettle,
    IngestResult,
    Message,
    MessageId,
    MessageRowId,
    OutboxItem,
    PlatformId,
    RecentSnapshot,
    Record,
    SelfId,
    SenderId,
)

CK = ChatKey("qq:group:123456")


def run(coro):
    return asyncio.run(coro)


def make_identity(
    chat_key: str = "qq:group:123456",
    platform: str = "qq",
    self_id: str = "bot-1",
    kind: str = "group",
    title: str | None = None,
) -> ChatIdentity:
    return ChatIdentity(
        ChatKey(chat_key), PlatformId(platform), SelfId(self_id), kind, title
    )


def make_message(
    chat_key: str = "qq:group:123456",
    text: str = "hello",
    msg_id: str | None = "m1",
    sender_id: str = "u1",
    sender_name: str = "user",
    is_self: bool = False,
    recv_ts: float = 1_700_000_000.0,
    reply_to: str | None = None,
    mentions: tuple = (),
    segments: tuple = (),
) -> Message:
    return Message(
        chat_key=ChatKey(chat_key),
        sender_id=SenderId(sender_id),
        sender_name=sender_name,
        is_self=is_self,
        text=text,
        id=MessageId(msg_id) if msg_id is not None else None,
        reply_to=MessageId(reply_to) if reply_to else None,
        mentions=tuple(SenderId(m) for m in mentions),
        recv_ts=recv_ts,
        segments=segments,
    )


def make_claim(
    chat_key: str = "qq:group:123456",
    cycle_id: str = "cy-1",
    started_ts: float = 100.0,
    expires_at: float = 500.0,
) -> CycleClaim:
    return CycleClaim(ChatKey(chat_key), CycleId(cycle_id), started_ts, expires_at)


def make_finish(
    chat_key: str = "qq:group:123456",
    cycle_id: str = "cy-1",
    end_reason: str = "completed",
    hold_until: float | None = None,
    idle_streak_after: int = 0,
    trace_json: str | None = '{"t": 1}',
    tokens_in: int = 10,
    tokens_out: int = 20,
) -> CycleFinish:
    return CycleFinish(
        chat_key=ChatKey(chat_key),
        cycle_id=CycleId(cycle_id),
        end_reason=end_reason,
        hold_until=hold_until,
        idle_streak_after=idle_streak_after,
        trace_json=trace_json,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


async def open_repo(
    path: str | Path, *, batch_window: float = 0.0
) -> tuple[Database, SqliteRepository]:
    """A Database (fast writer: no coalescing wait) + repository on it."""
    db = Database(path, batch_window=batch_window)
    await db.open()
    return db, SqliteRepository(db)


async def open_repo_with_chat(
    path: str | Path, *, batch_window: float = 0.0
) -> tuple[Database, SqliteRepository]:
    """open_repo plus the standard chat identity upserted."""
    db, repo = await open_repo(path, batch_window=batch_window)
    await repo.upsert_chat(make_identity())
    return db, repo


async def finish_batch(
    repo,
    items: list[OutboxItem],
    *,
    chat_key: str = "qq:group:123456",
    cycle_id: str = "cy-1",
    started_ts: float = 100.0,
    expires_at: float = 500.0,
    now: float = 200.0,
    end_reason: str = "completed",
    hold_until: float | None = None,
    idle_streak_after: int = 0,
) -> ClaimGrant:
    """The ONLY supported creation route for outbox rows: claim the chat,
    then finish the cycle with the ordered batch."""
    grant = await repo.claim_cycle(make_claim(chat_key, cycle_id, started_ts, expires_at))
    assert grant is not None, "claim must succeed"
    await repo.finish_cycle(
        make_finish(
            chat_key, cycle_id, end_reason, hold_until, idle_streak_after
        ),
        items,
        now=now,
    )
    return grant


class FakeRepo:
    """A protocol-complete Repository fake: proves Ingest/OutboxDriver
    depend only on the seam, never on SqliteRepository."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.ready_items: list[OutboxItem] = []
        self.attempt_result = True
        self.mark_result = True
        self.ingest_result: IngestResult = IngestResult(row_id=MessageRowId(1), inserted=True)
        self.claim_result: ClaimGrant | ClaimBusy | None = None
        self.dispatch_result: DispatchGrant | ClaimBusy | None = None
        self.unexported_commits: list[CorpusMarker] = []
        self.unexported_dispatches: list[CorpusMarker] = []
        self.unassigned_commits: list[CommitSeq] = []
        self.next_due_result: float | None = None
        self.outbox_chats: list[ChatKey] = []
        self.latest_end_reason: str | None = None
        self.pending_chats: list[ChatKey] = []

    async def get_chat(self, chat_key: ChatKey) -> ChatIdentity | None:
        self.calls.append(("get_chat", chat_key))
        return None

    async def upsert_chat(self, chat: ChatIdentity) -> None:
        self.calls.append(("upsert_chat", chat))

    async def get_chat_state(self, chat_key: ChatKey):
        self.calls.append(("get_chat_state", chat_key))
        return None

    async def upsert_chat_state(self, state) -> None:
        self.calls.append(("upsert_chat_state", state))

    async def ingest_message(
        self,
        identity,
        msg,
        *,
        self_echo_delivery_key=None,
        event_id=None,
        structural_priority=False,
        pending_threshold=None,
    ):
        self.calls.append(
            (
                "ingest_message",
                identity,
                msg,
                self_echo_delivery_key,
                event_id,
                structural_priority,
                pending_threshold,
            )
        )
        return self.ingest_result

    async def get_message(self, chat_key: ChatKey, msg_id: MessageId):
        self.calls.append(("get_message", chat_key, msg_id))
        return None

    async def get_recent_snapshot(
        self, chat_key: ChatKey, through_row_id: MessageRowId, since_ts: float, limit: int
    ) -> RecentSnapshot:
        self.calls.append(
            ("get_recent_snapshot", chat_key, through_row_id, since_ts, limit)
        )
        # Structurally satisfies the seam without inventing concrete DB
        # access: carry the bounds the caller asked for, nothing more.
        return RecentSnapshot(
            chat_key=chat_key, since_ts=since_ts, through_row_id=through_row_id
        )

    async def claim_cycle(self, claim: CycleClaim):
        self.calls.append(("claim_cycle", claim))
        return self.claim_result

    async def renew_cycle(self, chat_key, cycle_id, expires_at, *, now):
        self.calls.append(("renew_cycle", chat_key, cycle_id, expires_at, now))
        return True

    async def release_cycle(self, chat_key, cycle_id) -> None:
        self.calls.append(("release_cycle", chat_key, cycle_id))

    async def finish_cycle(self, finish, outbox, *, now) -> None:
        self.calls.append(("finish_cycle", finish, outbox, now))

    async def get_latest_terminal_end_reason(self, chat_key: ChatKey):
        self.calls.append(("get_latest_terminal_end_reason", chat_key))
        return self.latest_end_reason

    async def begin_dispatch(self, request: DispatchRequest):
        self.calls.append(("begin_dispatch", request))
        return self.dispatch_result

    async def renew_dispatch(self, chat_key, dispatch_id, cycle_id, expires_at, *, now):
        self.calls.append(
            ("renew_dispatch", chat_key, dispatch_id, cycle_id, expires_at, now)
        )
        return True

    async def settle_dispatch(self, settle: DispatchSettle, outbox, *, now):
        self.calls.append(("settle_dispatch", settle, outbox, now))

    async def list_unexported_commits(self):
        self.calls.append(("list_unexported_commits",))
        return list(self.unexported_commits)

    async def list_unexported_dispatches(self):
        self.calls.append(("list_unexported_dispatches",))
        return list(self.unexported_dispatches)

    async def mark_commit_exported(self, commit_seq: CommitSeq) -> None:
        self.calls.append(("mark_commit_exported", commit_seq))

    async def mark_dispatch_exported(self, dispatch_id: DispatchId) -> None:
        self.calls.append(("mark_dispatch_exported", dispatch_id))

    async def list_unassigned_commits(self, chat_key: ChatKey):
        self.calls.append(("list_unassigned_commits", chat_key))
        return list(self.unassigned_commits)

    async def list_ledger_pending_chats(self):
        self.calls.append(("list_ledger_pending_chats",))
        return list(self.pending_chats)

    async def list_ready_outbox(self, chat_key, *, now, limit=10):
        self.calls.append(("list_ready_outbox", chat_key, now, limit))
        return list(self.ready_items)

    async def next_due_outbox(self, chat_key, *, now):
        self.calls.append(("next_due_outbox", chat_key, now))
        return self.next_due_result

    async def list_outbox_chats(self):
        self.calls.append(("list_outbox_chats",))
        return list(self.outbox_chats)

    async def attempt_outbox(self, item_id, attempt_started_ts):
        self.calls.append(("attempt_outbox", item_id, attempt_started_ts))
        return self.attempt_result

    async def requeue_outbox(self, item_id):
        self.calls.append(("requeue_outbox", item_id))
        return True

    async def mark_outbox_sent(self, item_id, platform_msg_id, sent_ts):
        self.calls.append(("mark_outbox_sent", item_id, platform_msg_id, sent_ts))
        return self.mark_result

    async def drop_outbox(self, item_id):
        self.calls.append(("drop_outbox", item_id))
        return True

    async def list_pending_chats(self):
        self.calls.append(("list_pending_chats",))
        return list(self.pending_chats)

    async def add_record(self, rec: Record) -> int:
        self.calls.append(("add_record", rec))
        return 1

    async def get_kv(self, k: str):
        self.calls.append(("get_kv", k))
        return None

    async def set_kv(self, k: str, v: str) -> None:
        self.calls.append(("set_kv", k, v))

    async def stats(self):
        self.calls.append(("stats",))
        return {}

    async def close(self) -> None:
        self.calls.append(("close",))
