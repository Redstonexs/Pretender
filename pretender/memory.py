"""Phase 5 durable memory service: a local orchestration layer.

``MemoryService`` composes the verified knowledge foundation into one
deterministic service surface:

  - ``summarize`` reads a fixed cursor-bounded ``MemorySourceBatch``
    (bounded by the terminal cursor and the durable memory watermark,
    retaining a recent tail), runs an INJECTED async summarizer/compressor
    over it, validates the resulting ``MemoryWriteRequest``, and CAS-commits
    it. The summarizer is never defaulted to a network call: with no
    summarizer injected the service returns an explicit ``unavailable``
    result, never an error. The summarizer runs OUTSIDE any transaction —
    only the CAS commit is transactional.
  - ``recall`` is a thin passthrough to the shared ``MemorySearch``
    (lexical-first, optional semantic). With no embedding service the
    semantic path performs ZERO provider calls and recall is FTS-only.

Everything here is chat-scoped and local-deterministic; no provider/network
call happens inside any transaction. This module performs no background
compression or vector backfill — the M6 learners drive those.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pretender.search import MemoryRecallHit, MemorySearch
from pretender.seams import KnowledgeRepository
from pretender.types import (
    ChatKey,
    MemoryRecord,
    MemorySourceBatch,
    MemoryWriteRequest,
    MessageRowId,
)

__all__ = [
    "DEFAULT_SOURCE_TAIL",
    "MemoryService",
    "SummarizeResult",
    "Summarizer",
    "default_capsule_summarizer",
]

#: The default recent-tail cap when reading a source batch.
DEFAULT_SOURCE_TAIL = 100

#: An injected async summarizer/compressor: turns one source batch into the
#: memory records to CAS-commit. Never defaulted to a network call.
Summarizer = Callable[[MemorySourceBatch], Awaitable[tuple[MemoryRecord, ...]]]


def default_capsule_summarizer(
    *, max_chars: int = 400, separator: str = "\n"
) -> Summarizer:
    """A deterministic LOCAL capsule writer (no network, no LLM).

    Produces exactly ONE memory record per source batch: a deterministic
    capsule of the batch texts (joined, bounded to ``max_chars``), stamped
    with the batch's source range and hash. This is the default
    ``MemoryService`` summarizer surface — runtime wiring to a real
    compressor is a later integration, but the deterministic local capsule
    is always available and never performs a provider call.
    """

    async def summarize(batch: MemorySourceBatch) -> tuple[MemoryRecord, ...]:
        joined = separator.join(batch.texts)
        if len(joined) > max_chars:
            joined = joined[:max_chars]
        return (
            MemoryRecord(
                chat_key=batch.chat_key,
                text=joined,
                source_first_msg_id=batch.first_msg_id,
                source_last_msg_id=batch.last_msg_id,
                source_hash=batch.source_hash,
            ),
        )

    return summarize


@dataclass(frozen=True)
class SummarizeResult:
    """The typed outcome of one ``MemoryService.summarize`` call.

    ``status`` is one of:

      - ``"ok"``: the source batch was summarized and CAS-committed
        (``committed`` True, ``records`` the committed records).
      - ``"unavailable"``: no summarizer is injected — a clear no-op, never
        an error (``reason`` explains).
      - ``"no_work"``: nothing is beyond the durable watermark (or the chat
        is unknown) — nothing to summarize.
      - ``"stale"``: the CAS lost (the watermark moved) — nothing changed.
    """

    status: str
    committed: bool = False
    records: tuple[MemoryRecord, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("ok", "unavailable", "no_work", "stale"):
            raise ValueError(f"invalid status: {self.status!r}")
        if not isinstance(self.committed, bool):
            raise ValueError("committed must be a bool")


class MemoryService:
    """Local orchestration over ``KnowledgeRepository`` + ``MemorySearch``.

    ``repo`` is a ``KnowledgeRepository``; ``search`` is the shared
    ``MemorySearch`` (defaults to one over ``repo`` with the optional
    ``embed``/``vectors``); ``summarizer`` is the injected async
    summarizer/compressor (None disables summarization with an explicit
    ``unavailable`` result); ``tail`` is the recent-tail cap for source
    batches.
    """

    def __init__(
        self,
        repo: KnowledgeRepository,
        *,
        search: MemorySearch | None = None,
        embed: Any = None,
        vectors: Any = None,
        summarizer: Summarizer | None = None,
        tail: int = DEFAULT_SOURCE_TAIL,
    ) -> None:
        if isinstance(tail, bool) or not isinstance(tail, int) or tail <= 0:
            raise ValueError(f"tail must be a positive integer, got {tail!r}")
        self._repo = repo
        self._search = (
            search if search is not None else MemorySearch(repo, embed=embed, vectors=vectors)
        )
        self._summarizer = summarizer
        self._tail = tail

    @property
    def search(self) -> MemorySearch:
        """The shared ``MemorySearch`` this service recalls through."""
        return self._search

    async def summarize(
        self, chat_key: ChatKey, *, through_msg_id: MessageRowId
    ) -> SummarizeResult:
        """Read a fixed source batch, summarize it, validate, and CAS-commit.

        The summarizer runs OUTSIDE any transaction; only the CAS commit is
        transactional. A stale CAS (the watermark moved) changes nothing and
        returns ``"stale"``. With no summarizer injected this returns
        ``"unavailable"`` — never an error.
        """
        if self._summarizer is None:
            return SummarizeResult(
                status="unavailable", reason="no summarizer configured"
            )
        batch = await self._repo.read_memory_source_batch(
            chat_key, through_msg_id=through_msg_id, tail=self._tail
        )
        if batch is None:
            return SummarizeResult(status="no_work")
        records = tuple(await self._summarizer(batch))
        self._validate_records(chat_key, batch, records)
        request = MemoryWriteRequest(
            chat_key=chat_key,
            batch=batch,
            records=records,
            # Forward the exact observed watermark snapshot so the CAS
            # compares against precisely what this batch was read against —
            # a sequential summarizer never races itself.
            expected_through_msg_id=batch.observed_watermark,
        )
        ok = await self._repo.commit_memory_source(request)
        if not ok:
            return SummarizeResult(status="stale", committed=False, records=records)
        return SummarizeResult(status="ok", committed=True, records=records)

    async def recall(
        self, chat_key: ChatKey, query: str, *, limit: int = 10
    ) -> list[MemoryRecallHit]:
        """Lexical-first (optionally semantic) memory recall for one chat."""
        return await self._search.search(chat_key, query, limit=limit)

    @staticmethod
    def _validate_records(
        chat_key: ChatKey, batch: MemorySourceBatch, records: tuple[MemoryRecord, ...]
    ) -> None:
        """Validate the summarizer's records before the CAS commit.

        Fail closed on a cross-chat record, a record whose source range
        does not match the batch, or a batch producing anything other than
        EXACTLY ONE record — a programming error in the summarizer, caught
        before any transaction (the repository enforces the same
        one-record-per-source-range cardinality atomically).
        """
        if len(records) != 1:
            raise ValueError(
                "exactly one memory record per source batch is required,"
                f" got {len(records)}"
            )
        for rec in records:
            if rec.chat_key != chat_key:
                raise ValueError(
                    f"memory record chat_key {rec.chat_key!r} does not match"
                    f" {chat_key!r}"
                )
            if rec.source_first_msg_id is not None and (
                rec.source_first_msg_id != batch.first_msg_id
                or rec.source_last_msg_id != batch.last_msg_id
            ):
                raise ValueError(
                    "memory record source range does not match the batch:"
                    f" {rec.source_first_msg_id}..{rec.source_last_msg_id} != "
                    f"{batch.first_msg_id}..{batch.last_msg_id}"
                )
