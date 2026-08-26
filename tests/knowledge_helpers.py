"""Shared helpers for the Phase 5 knowledge foundation tests.

Tests run via asyncio.run() so the test extra stays at just pytest.
"""

from __future__ import annotations

import hashlib
import struct

from pretender.types import (
    ChatKey,
    MemoryRecord,
    MemorySourceBatch,
    MemoryWriteRequest,
    MessageRowId,
    PersonProfile,
    SenderId,
    VectorRow,
)
from tests.durable_helpers import CK, make_identity, make_message, open_repo_with_chat, run

OTHER = ChatKey("qq:group:other")


def source_hash(texts: tuple[str, ...]) -> str:
    """The deterministic source hash the repository computes for a batch."""
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def f32(*values: float) -> bytes:
    """Pack float32 values into a vector blob."""
    return struct.pack(f"<{len(values)}f", *values)


def make_memory(chat_key: ChatKey = CK, text: str = "remember this", **kw) -> MemoryRecord:
    return MemoryRecord(chat_key=chat_key, text=text, **kw)


def make_person(
    chat_key: ChatKey = CK, uid: str = "u1", **kw
) -> PersonProfile:
    return PersonProfile(chat_key=chat_key, platform_uid=SenderId(uid), **kw)


def make_vector(
    owner_id: int = 1,
    dim: int = 2,
    model: str = "m1",
    generation: int = 1,
    values: tuple[float, ...] = (0.1, 0.2),
    **kw,
) -> VectorRow:
    return VectorRow(
        owner_table="memories",
        owner_id=owner_id,
        dim=dim,
        model=model,
        generation=generation,
        blob=f32(*values),
        **kw,
    )


async def seed_messages(repo, chat_key: ChatKey = CK, n: int = 3, prefix: str = "msg"):
    """Insert ``n`` non-self messages into the chat (row ids 1..n)."""
    for i in range(1, n + 1):
        await repo.ingest_message(
            None,
            make_message(
                chat_key=chat_key,
                msg_id=f"{prefix}-{i}",
                text=f"{prefix} {i}",
                recv_ts=1_700_000_000.0 + i,
            ),
        )


async def read_and_commit(
    repo,
    chat_key: ChatKey = CK,
    *,
    through_msg_id: MessageRowId,
    tail: int = 100,
    expected: MessageRowId | None = None,
    text: str = "summary",
) -> bool:
    """The canonical summarizer flow: read a fixed source batch, produce
    one memory record, and CAS-commit it. ``expected`` defaults to the
    CURRENT durable watermark (the natural sequential summarizer), so a
    caller only passes it explicitly to force a stale CAS. Returns the
    commit result."""
    batch = await repo.read_memory_source_batch(
        chat_key, through_msg_id=through_msg_id, tail=tail
    )
    assert batch is not None, "expected a source batch"
    if expected is None:
        expected = await repo.get_memory_watermark(chat_key)
    rec = make_memory(
        chat_key=chat_key,
        text=text,
        source_first_msg_id=batch.first_msg_id,
        source_last_msg_id=batch.last_msg_id,
        source_hash=batch.source_hash,
    )
    return await repo.commit_memory_source(
        MemoryWriteRequest(
            chat_key=chat_key,
            batch=batch,
            records=(rec,),
            expected_through_msg_id=expected,
        )
    )