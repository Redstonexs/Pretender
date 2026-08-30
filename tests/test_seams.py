"""Repository seam contract: async surface, claim/finish + outbox CAS API,
no independent cursor-advance operation, and preserved runtime_checkable
behavior."""

from __future__ import annotations

import dataclasses
import inspect
import typing
from typing import Any

from pretender.repo import SqliteRepository
from pretender.seams import (
    AdaptiveRepository,
    GateContext,
    KnowledgeRepository,
    MediaRepository,
    Repository,
)
from pretender.types import (
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
    DispatchDeferred,
    DispatchGrant,
    DispatchId,
    DispatchRequest,
    DispatchSettle,
    EmbeddingGeneration,
    EventId,
    GateSnapshot,
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
    MemoryRecord,
    MemorySourceBatch,
    MemoryWriteRequest,
    Message,
    MessageId,
    MessageRowId,
    OutboxItem,
    PersonProfile,
    RecentSnapshot,
    Record,
    RecordHit,
    SelfId,
    SenderId,
    VectorRow,
)

CK = ChatKey("qq:group:123456")


def _snapshot(**kw: Any) -> GateSnapshot:
    base: dict[str, Any] = dict(
        chat_key=CK,
        cycle_id=CycleId("c1"),
        start_msg_id=MessageRowId(1),
        through_msg_id=MessageRowId(9),
        evaluated_ts=300.0,
        self_id=SelfId("bot"),
        mode="normal",
        threshold=60,
        trigger_score=100,
        frequency=0.5,
        pending=2,
        pending_messages=(
            Message(chat_key=CK, sender_id=SenderId("u1"), sender_name="alice",
                    is_self=False, text="hi", row_id=MessageRowId(2)),
            Message(chat_key=CK, sender_id=SenderId("u2"), sender_name="bob",
                    is_self=False, text="yo", row_id=MessageRowId(3)),
        ),
        recent=(),
        window_count=5,
        self_count=2,
        last_nonself_ts=250.0,
        idle_seconds=30.0,
        recent_average_interval=120.0,
        self_ratio=0.1,
        is_group=True,
        is_focused=False,
        last_message=None,
    )
    base.update(kw)
    return GateSnapshot(**base)


def _protocol_members(protocol: type[Any]) -> set[str]:
    get_protocol_members = getattr(typing, "get_protocol_members", None)
    if get_protocol_members is not None:
        return set(get_protocol_members(protocol))

    members: set[str] = set()
    for base in protocol.__mro__:
        members.update(
            name for name in getattr(base, "__annotations__", {})
            if not name.startswith("_")
        )
        members.update(name for name in vars(base) if not name.startswith("_"))
    return members


# The full required surface, in the order the protocol declares it.
REQUIRED_METHODS = (
    "get_chat",
    "upsert_chat",
    "get_chat_state",
    "upsert_chat_state",
    "ingest_message",
    "get_message",
    "get_recent_snapshot",
    "list_pending_chats",
    "claim_cycle",
    "renew_cycle",
    "release_cycle",
    "finish_cycle",
    "get_latest_terminal_end_reason",
    "begin_dispatch",
    "renew_dispatch",
    "settle_dispatch",
    "list_unexported_commits",
    "list_unexported_dispatches",
    "mark_commit_exported",
    "mark_dispatch_exported",
    "list_unassigned_commits",
    "list_ledger_pending_chats",
    "list_ready_outbox",
    "next_due_outbox",
    "list_outbox_chats",
    "attempt_outbox",
    "requeue_outbox",
    "mark_outbox_sent",
    "drop_outbox",
    "add_record",
    "get_kv",
    "set_kv",
    "stats",
    "close",
)


# ── Async surface ───────────────────────────────────────────────────────────

def test_repository_protocol_declares_exactly_the_required_methods():
    protocol_attrs = _protocol_members(Repository)
    assert set(protocol_attrs) == set(REQUIRED_METHODS)


def test_all_repository_methods_are_coroutine_functions():
    for name in REQUIRED_METHODS:
        fn = getattr(Repository, name)
        assert inspect.iscoroutinefunction(fn), f"{name} must be async"


def test_removed_sync_surface_is_gone():
    for name in ("add_outbox", "add_cycle", "list_pending", "add_message",
                 "enqueue_outbox", "get_claim"):
        assert not hasattr(Repository, name), f"{name} must be removed"


# ── No independent cursor movement ──────────────────────────────────────────

def test_no_independent_cursor_advance_method():
    # Cursor movement must not be expressible as a public operation.
    assert not hasattr(Repository, "advance_cursor")
    assert not hasattr(Repository, "set_cursor")
    assert not hasattr(Repository, "move_cursor")
    # ...and no method takes a cursor parameter either.
    for name in _protocol_members(Repository):
        params = inspect.signature(getattr(Repository, name)).parameters
        assert "cursor" not in params, f"{name} takes a cursor parameter"
        assert "after_cursor" not in params, f"{name} takes an after_cursor parameter"


def test_cursor_advance_only_reachable_through_finish_cycle():
    # The finish carries NO cursor value: the cursor derives from the
    # stored claim boundary, never from a caller-provided value.
    assert hasattr(Repository, "finish_cycle")
    assert "cursor_msg_id" not in CycleFinish.__dataclass_fields__
    # ChatState still exposes the cursor as a READ-ONLY view...
    assert "cursor_msg_id" in ChatState.__dataclass_fields__
    # ...but upsert_chat_state must not be able to move it: the state
    # update is the only other writer of the chats row.
    assert "cursor_msg_id" not in ChatIdentity.__dataclass_fields__


def test_claim_carries_boundary_and_pending_in_grant():
    # The claim grant exposes the fixed local-row boundary and the claimed
    # pending messages; the caller cannot choose them.
    assert "start_msg_id" in ClaimGrant.__dataclass_fields__
    assert "through_msg_id" in ClaimGrant.__dataclass_fields__
    assert "pending" in ClaimGrant.__dataclass_fields__
    # The lease is mandatory and finite.
    fields = CycleClaim.__dataclass_fields__
    assert "expires_at" in fields
    assert fields["expires_at"].default is dataclasses.MISSING


# ── Claim / finish and outbox CAS signatures ────────────────────────────────

def test_claim_finish_signatures():
    claim_params = inspect.signature(Repository.claim_cycle).parameters
    assert list(claim_params) == ["self", "claim"]
    assert claim_params["claim"].annotation == "CycleClaim"

    finish_params = inspect.signature(Repository.finish_cycle).parameters
    assert list(finish_params) == ["self", "finish", "outbox", "now"]
    assert finish_params["finish"].annotation == "CycleFinish"
    assert finish_params["outbox"].annotation == "list[OutboxItem]"
    assert finish_params["now"].kind is inspect.Parameter.KEYWORD_ONLY

    renew_params = inspect.signature(Repository.renew_cycle).parameters
    assert list(renew_params) == ["self", "chat_key", "cycle_id", "expires_at", "now"]
    assert renew_params["now"].kind is inspect.Parameter.KEYWORD_ONLY


def test_claim_cycle_returns_typed_grant_busy_or_none():
    # The ambiguous ClaimGrant | None is gone: a live, unexpired owner is
    # reported as a typed ClaimBusy with its exact busy_until; None is
    # reserved for no-work/unknown chats; a grant for a successful claim.
    assert (
        inspect.signature(Repository.claim_cycle).return_annotation
        == "ClaimGrant | ClaimBusy | None"
    )
    # ClaimBusy is a frozen typed result carrying the exact busy_until —
    # never a raw claim row.
    assert dataclasses.is_dataclass(ClaimBusy)
    assert ClaimBusy.__dataclass_fields__["busy_until"].type == "float"
    assert ClaimBusy.__dataclass_fields__["cycle_id"].type == "CycleId"
    assert "id" not in ClaimBusy.__dataclass_fields__
    assert "state" not in ClaimBusy.__dataclass_fields__


def test_outbox_cas_signatures():
    attempt = inspect.signature(Repository.attempt_outbox).parameters
    assert list(attempt) == ["self", "item_id", "attempt_started_ts"]
    assert attempt["item_id"].annotation == "int"
    assert attempt["attempt_started_ts"].annotation == "float"

    sent = inspect.signature(Repository.mark_outbox_sent).parameters
    assert list(sent) == ["self", "item_id", "platform_msg_id", "sent_ts"]
    assert sent["platform_msg_id"].annotation == "MessageId | None"

    drop = inspect.signature(Repository.drop_outbox).parameters
    assert list(drop) == ["self", "item_id"]

    ready = inspect.signature(Repository.list_ready_outbox).parameters
    assert list(ready) == ["self", "chat_key", "now", "limit"]
    assert ready["now"].kind is inspect.Parameter.KEYWORD_ONLY
    assert ready["limit"].default == 10

    next_due = inspect.signature(Repository.next_due_outbox).parameters
    assert list(next_due) == ["self", "chat_key", "now"]
    assert next_due["now"].kind is inspect.Parameter.KEYWORD_ONLY

    ingest = inspect.signature(Repository.ingest_message).parameters
    assert list(ingest) == [
        "self", "identity", "msg", "self_echo_delivery_key", "event_id",
        "structural_priority", "pending_threshold",
    ]
    assert ingest["msg"].annotation == "Message"
    assert ingest["self_echo_delivery_key"].kind is inspect.Parameter.KEYWORD_ONLY
    assert ingest["self_echo_delivery_key"].default is None
    assert ingest["event_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert ingest["event_id"].default is None
    assert ingest["structural_priority"].kind is inspect.Parameter.KEYWORD_ONLY
    assert ingest["structural_priority"].default is False
    assert ingest["pending_threshold"].kind is inspect.Parameter.KEYWORD_ONLY
    assert ingest["pending_threshold"].default is None
    assert (
        inspect.signature(Repository.ingest_message).return_annotation == "IngestResult"
    )


def test_get_latest_terminal_end_reason_signature():
    # The semantic per-chat read of the latest TERMINAL cycle end reason —
    # the gate's only history input. Released/expired claims never affect it.
    params = inspect.signature(Repository.get_latest_terminal_end_reason).parameters
    assert list(params) == ["self", "chat_key"]
    assert params["chat_key"].annotation == "ChatKey"
    assert (
        inspect.signature(Repository.get_latest_terminal_end_reason).return_annotation
        == "str | None"
    )


# ── durable dispatch ledger signatures ──────────────────────────────────────

def test_begin_dispatch_signature():
    # The ledger claim: one typed DispatchRequest, returning the typed
    # grant/busy/deferred/none — never a raw dispatch row.
    params = inspect.signature(Repository.begin_dispatch).parameters
    assert list(params) == ["self", "request"]
    assert params["request"].annotation == "DispatchRequest"
    assert (
        inspect.signature(Repository.begin_dispatch).return_annotation
        == "DispatchGrant | ClaimBusy | DispatchDeferred | None"
    )


def test_renew_dispatch_signature():
    # The ledger lease extension: fenced to the same unexpired prepared
    # owner, with ``now`` keyword-only and a finite forward extension.
    params = inspect.signature(Repository.renew_dispatch).parameters
    assert list(params) == [
        "self", "chat_key", "dispatch_id", "cycle_id", "expires_at", "now",
    ]
    assert params["chat_key"].annotation == "ChatKey"
    assert params["dispatch_id"].annotation == "DispatchId"
    assert params["cycle_id"].annotation == "CycleId"
    assert params["expires_at"].annotation == "float"
    assert params["now"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        inspect.signature(Repository.renew_dispatch).return_annotation == "bool"
    )


def test_settle_dispatch_signature():
    # The ledger settlement: typed DispatchSettle + the ordered outbox
    # batch, with ``now`` keyword-only — the ONLY cursor/outbox path.
    params = inspect.signature(Repository.settle_dispatch).parameters
    assert list(params) == ["self", "settle", "outbox", "now"]
    assert params["settle"].annotation == "DispatchSettle"
    assert params["outbox"].annotation == "list[OutboxItem]"
    assert params["now"].kind is inspect.Parameter.KEYWORD_ONLY


def test_ledger_export_and_recovery_signatures():
    # The at-least-once export surface and the crash-recovery scans: typed
    # markers and sequences, no SQL or raw rows.
    assert (
        inspect.signature(Repository.list_unexported_commits).return_annotation
        == "list[CorpusMarker]"
    )
    assert (
        inspect.signature(Repository.list_unexported_dispatches).return_annotation
        == "list[CorpusMarker]"
    )
    mark = inspect.signature(Repository.mark_commit_exported).parameters
    assert list(mark) == ["self", "commit_seq"]
    assert mark["commit_seq"].annotation == "CommitSeq"
    mark = inspect.signature(Repository.mark_dispatch_exported).parameters
    assert list(mark) == ["self", "dispatch_id"]
    assert mark["dispatch_id"].annotation == "DispatchId"
    scan = inspect.signature(Repository.list_unassigned_commits).parameters
    assert list(scan) == ["self", "chat_key"]
    assert (
        inspect.signature(Repository.list_unassigned_commits).return_annotation
        == "list[CommitSeq]"
    )
    assert (
        inspect.signature(Repository.list_ledger_pending_chats).return_annotation
        == "list[ChatKey]"
    )


def test_get_recent_snapshot_signature():
    # The single claim-bounded recent-message read: chat, claim boundary,
    # window start, and list limit — returning a RecentSnapshot.
    params = inspect.signature(Repository.get_recent_snapshot).parameters
    assert list(params) == ["self", "chat_key", "through_row_id", "since_ts", "limit"]
    assert params["chat_key"].annotation == "ChatKey"
    assert params["through_row_id"].annotation == "MessageRowId"
    assert params["since_ts"].annotation == "float"
    assert params["limit"].annotation == "int"
    assert params["limit"].default is inspect.Parameter.empty
    assert (
        inspect.signature(Repository.get_recent_snapshot).return_annotation
        == "RecentSnapshot"
    )


def test_list_pending_chats_signature():
    # The startup-recovery read: no arguments, returns the pending chat
    # keys (non-self messages beyond the durable cursor).
    params = inspect.signature(Repository.list_pending_chats).parameters
    assert list(params) == ["self"]
    assert (
        inspect.signature(Repository.list_pending_chats).return_annotation
        == "list[ChatKey]"
    )


def test_no_untrusted_reconciliation_surface():
    # The ONLY reconciliation path is the trusted self-echo key flow inside
    # ingest_message: no item-id reconciliation method may exist on the
    # public seam.
    assert not hasattr(Repository, "reconcile_outbox")
    assert not hasattr(Repository, "reconcile")
    assert not hasattr(Repository, "mark_outbox_reconciled")


# ── runtime_checkable behavior ──────────────────────────────────────────────

class _FakeRepo:
    """A minimal concrete implementation: async methods, no cursor ops."""

    async def get_chat(self, chat_key: ChatKey) -> ChatIdentity | None:
        return None

    async def upsert_chat(self, chat: ChatIdentity) -> None:
        pass

    async def get_chat_state(self, chat_key: ChatKey) -> ChatState | None:
        return None

    async def upsert_chat_state(self, state: ChatState) -> None:
        pass

    async def ingest_message(
        self,
        identity: ChatIdentity | None,
        msg: Message,
        *,
        self_echo_delivery_key: str | None = None,
        event_id: EventId | None = None,
    ) -> IngestResult:
        return IngestResult(row_id=MessageRowId(1), inserted=True)

    async def get_message(self, chat_key: ChatKey, msg_id: MessageId) -> Message | None:
        return None

    async def get_recent_snapshot(
        self, chat_key: ChatKey, through_row_id: MessageRowId, since_ts: float, limit: int
    ) -> RecentSnapshot:
        return RecentSnapshot(
            chat_key=chat_key, since_ts=since_ts, through_row_id=through_row_id
        )

    async def list_pending_chats(self) -> list[ChatKey]:
        return []

    async def claim_cycle(self, claim: CycleClaim) -> ClaimGrant | ClaimBusy | None:
        return None

    async def renew_cycle(
        self, chat_key: ChatKey, cycle_id: CycleId, expires_at: float, *, now: float
    ) -> bool:
        return True

    async def release_cycle(self, chat_key: ChatKey, cycle_id: CycleId) -> None:
        pass

    async def finish_cycle(
        self, finish: CycleFinish, outbox: list[OutboxItem], *, now: float
    ) -> None:
        pass

    async def get_latest_terminal_end_reason(self, chat_key: ChatKey) -> str | None:
        return None

    async def begin_dispatch(
        self, request: DispatchRequest
    ) -> DispatchGrant | ClaimBusy | DispatchDeferred | None:
        return None

    async def renew_dispatch(
        self,
        chat_key: ChatKey,
        dispatch_id: DispatchId,
        cycle_id: CycleId,
        expires_at: float,
        *,
        now: float,
    ) -> bool:
        return True

    async def settle_dispatch(
        self, settle: DispatchSettle, outbox: list[OutboxItem], *, now: float
    ) -> None:
        pass

    async def list_unexported_commits(self) -> list[CorpusMarker]:
        return []

    async def list_unexported_dispatches(self) -> list[CorpusMarker]:
        return []

    async def mark_commit_exported(self, commit_seq: CommitSeq) -> None:
        pass

    async def mark_dispatch_exported(self, dispatch_id: DispatchId) -> None:
        pass

    async def list_unassigned_commits(self, chat_key: ChatKey) -> list[CommitSeq]:
        return []

    async def list_ledger_pending_chats(self) -> list[ChatKey]:
        return []

    async def list_ready_outbox(
        self, chat_key: ChatKey, *, now: float, limit: int = 10
    ) -> list[OutboxItem]:
        return []

    async def next_due_outbox(self, chat_key: ChatKey, *, now: float) -> float | None:
        return None

    async def list_outbox_chats(self) -> list[ChatKey]:
        return []

    async def attempt_outbox(self, item_id: int, attempt_started_ts: float) -> bool:
        return True

    async def requeue_outbox(self, item_id: int) -> bool:
        return True

    async def mark_outbox_sent(
        self, item_id: int, platform_msg_id: MessageId | None, sent_ts: float
    ) -> bool:
        return True

    async def drop_outbox(self, item_id: int) -> bool:
        return True

    async def add_record(self, rec: Record) -> int:
        return 1

    async def get_kv(self, k: str) -> str | None:
        return None

    async def set_kv(self, k: str, v: str) -> None:
        pass

    async def stats(self) -> dict[str, int]:
        return {}

    async def close(self) -> None:
        pass


def test_repository_is_runtime_checkable():
    assert isinstance(_FakeRepo(), Repository)


def test_gate_snapshot_satisfies_gate_context_protocol():
    # The concrete claim-bounded snapshot is structurally a GateContext:
    # runtime_checkable accepts it, and the field sets match exactly.
    snap = _snapshot(
        self_name="麦麦",
        has_direct_at=True,
        has_quote_to_self=True,
        has_other_assistant=True,
        hold_until=600.0,
        idle_streak=3,
    )
    assert isinstance(snap, GateContext)
    assert set(GateSnapshot.__dataclass_fields__) == set(
        _protocol_members(GateContext)
    )


def test_gate_context_exposes_all_frozen_evaluation_facts():
    # Every fact a GateFeature may read is a documented protocol attribute —
    # no feature needs undocumented concrete state or direct DB access.
    attrs = _protocol_members(GateContext)
    assert len(attrs) == 34
    for name in (
        "chat_key", "cycle_id", "start_msg_id", "through_msg_id",
        "evaluated_ts", "self_id", "self_name",
        "mode", "threshold", "trigger_score", "frequency",
        "backoff_base_s", "backoff_cap_s", "backoff_start_count",
        "pending", "pending_messages",
        "recent", "window_count", "self_count", "last_nonself_ts",
        "idle_seconds", "recent_average_interval", "self_ratio",
        "is_group", "is_focused", "last_message",
        "has_direct_at", "has_quote_to_self", "has_other_assistant",
        "hold_until", "idle_streak", "previous_end_reason",
        "muted",
    ):
        assert name in attrs, f"GateContext must expose {name}"


def test_gate_context_rejects_old_minimal_shape():
    # The old minimal GateContext shape (13 fields) is no longer a valid
    # GateContext: the claim/window/targeting facts are required protocol
    # attributes, so a stale feature context is rejected at registration.
    class _OldMinimal:
        chat_key = CK
        mode = "normal"
        threshold = 60
        trigger_score = 100
        frequency = 0.5
        pending = 2
        recent = ()
        idle_seconds = 30.0
        recent_average_interval = 120.0
        self_ratio = 0.1
        is_group = True
        is_focused = False
        last_message = None

    assert not isinstance(_OldMinimal(), GateContext)


def test_gate_context_has_no_method_surface():
    # GateContext is a pure data protocol: every protocol attribute is a
    # data attribute, so a frozen dataclass can satisfy it structurally.
    for attr in _protocol_members(GateContext):
        assert not callable(getattr(GateContext, attr, None)), attr


def test_repository_rejects_missing_method_at_runtime():
    class _MissingFinish:
        async def get_chat(self, chat_key: ChatKey) -> ChatIdentity | None:
            return None

    assert not isinstance(_MissingFinish(), Repository)


def test_concrete_repo_methods_are_coroutines():
    repo = _FakeRepo()
    for name in REQUIRED_METHODS:
        assert inspect.iscoroutinefunction(getattr(repo, name)), name


# ── Phase 5 KnowledgeRepository seam (separate protocol) ────────────────────

# The full required surface, in the order the protocol declares it.
REQUIRED_KNOWLEDGE_METHODS = (
    "get_memory_watermark",
    "get_memory_observed_watermark",
    "read_memory_source_batch",
    "commit_memory_source",
    "rebuild_memory_fts",
    "get_memory_fts_state",
    "mark_memory_fts_backlog",
    "list_memory_fts_unbootstrapped_chats",
    "list_memory_pending_chats",
    "query_memory",
    "get_memories",
    "list_memory_chats",
    "list_memories",
    "list_memories_after",
    "list_memory_chats_after",
    "list_vectors_for_memories",
    "get_person",
    "upsert_person",
    "add_person_alias",
    "cas_person_profile",
    "create_embedding_generation",
    "get_embedding_generation",
    "activate_embedding_generation",
    "activate_embedding_generation_if_complete",
    "list_embedding_generations",
    "upsert_vector",
    "get_vector",
    "list_vectors",
    "delete_vector",
)


def test_knowledge_repository_protocol_declares_exactly_the_required_methods():
    protocol_attrs = _protocol_members(KnowledgeRepository)
    assert set(protocol_attrs) == set(REQUIRED_KNOWLEDGE_METHODS)


def test_all_knowledge_repository_methods_are_coroutine_functions():
    for name in REQUIRED_KNOWLEDGE_METHODS:
        fn = getattr(KnowledgeRepository, name)
        assert inspect.iscoroutinefunction(fn), f"{name} must be async"


def test_knowledge_repository_is_separate_from_repository():
    # The Phase 1 Repository seam is NOT enlarged: the knowledge surface is
    # a distinct protocol, so a knowledge-only fake is not a Repository and
    # a Repository-only fake is not a KnowledgeRepository.
    assert not set(REQUIRED_KNOWLEDGE_METHODS) <= set(REQUIRED_METHODS)
    assert not set(REQUIRED_METHODS) <= set(REQUIRED_KNOWLEDGE_METHODS)


def test_sqlite_repository_satisfies_both_protocols():
    # SqliteRepository is the ONE concrete implementation that satisfies
    # both the Phase 1 Repository seam and the Phase 5 KnowledgeRepository
    # seam (the protocol check is structural, so a bare instance suffices).
    repo = SqliteRepository.__new__(SqliteRepository)
    assert isinstance(repo, Repository)
    assert isinstance(repo, KnowledgeRepository)


def test_repository_fake_is_not_a_knowledge_repository():
    # The existing Repository-only fake must NOT satisfy the knowledge
    # seam: knowledge consumers depend on the separate protocol.
    assert isinstance(_FakeRepo(), Repository)
    assert not isinstance(_FakeRepo(), KnowledgeRepository)


class _KnowledgeFake:
    """A minimal knowledge-only fake: satisfies KnowledgeRepository but
    NOT Repository (it implements none of the Phase 1 surface)."""

    async def get_memory_watermark(self, chat_key: ChatKey) -> MessageRowId | None:
        return None

    async def get_memory_observed_watermark(
        self, chat_key: ChatKey
    ) -> MessageRowId | None:
        return None

    async def read_memory_source_batch(
        self, chat_key: ChatKey, *, through_msg_id: MessageRowId, tail: int
    ) -> MemorySourceBatch | None:
        return None

    async def commit_memory_source(self, request: MemoryWriteRequest) -> bool:
        return True

    async def rebuild_memory_fts(self, chat_key: ChatKey) -> None:
        pass

    async def get_memory_fts_state(
        self, chat_key: ChatKey
    ) -> tuple[bool, MessageRowId | None] | None:
        return None

    async def mark_memory_fts_backlog(
        self, chat_key: ChatKey, through_msg_id: MessageRowId
    ) -> None:
        pass

    async def list_memory_fts_unbootstrapped_chats(self) -> list[ChatKey]:
        return []

    async def list_memory_pending_chats(
        self,
    ) -> list[tuple[ChatKey, MessageRowId]]:
        return []

    async def query_memory(
        self, chat_key: ChatKey, query: str, *, limit: int = 10
    ) -> list[LexicalHit]:
        return []

    async def get_memories(
        self, chat_key: ChatKey, memory_ids: list[int]
    ) -> list[MemoryRecord]:
        return []

    async def list_memory_chats(self) -> list[ChatKey]:
        return []

    async def list_memories(self, chat_key: ChatKey) -> list[MemoryRecord]:
        return []

    async def list_memories_after(
        self, chat_key: ChatKey, after_id: int, *, limit: int
    ) -> list[MemoryRecord]:
        return []

    async def list_memory_chats_after(
        self, after: ChatKey, *, limit: int
    ) -> list[ChatKey]:
        return []

    async def list_vectors_for_memories(
        self, chat_key: ChatKey, model: str, generation: int, memory_ids: list[int]
    ) -> list[VectorRow]:
        return []

    async def get_person(
        self, chat_key: ChatKey, platform_uid: SenderId
    ) -> PersonProfile | None:
        return None

    async def upsert_person(self, profile: PersonProfile) -> None:
        pass

    async def add_person_alias(
        self,
        chat_key: ChatKey,
        platform_uid: SenderId,
        name: str,
        *,
        now: float | None = None,
    ) -> tuple[str, ...] | None:
        return (name,)

    async def cas_person_profile(
        self,
        chat_key: ChatKey,
        platform_uid: SenderId,
        expected_through_msg_id: MessageRowId | None,
        profile: PersonProfile,
    ) -> bool:
        return True

    async def create_embedding_generation(
        self,
        model: str,
        dim: int,
        *,
        revision: str = "default",
        state: str = "inactive",
        created_ts: float | None = None,
    ) -> EmbeddingGeneration:
        return EmbeddingGeneration(
            id=1, model=model, dim=dim, revision=revision, state=state
        )

    async def get_embedding_generation(
        self, generation_id: int
    ) -> EmbeddingGeneration | None:
        return None

    async def activate_embedding_generation(self, generation_id: int) -> bool:
        return True

    async def activate_embedding_generation_if_complete(
        self, generation_id: int
    ) -> list[ChatKey] | None:
        return None

    async def list_embedding_generations(self) -> list[EmbeddingGeneration]:
        return []

    async def upsert_vector(self, chat_key: ChatKey, row: VectorRow) -> None:
        pass

    async def get_vector(
        self,
        chat_key: ChatKey,
        owner_table: str,
        owner_id: int,
        model: str,
        generation: int,
    ) -> VectorRow | None:
        return None

    async def list_vectors(
        self, chat_key: ChatKey, model: str, generation: int
    ) -> list[VectorRow]:
        return []

    async def delete_vector(
        self,
        chat_key: ChatKey,
        owner_table: str,
        owner_id: int,
        model: str,
        generation: int,
    ) -> bool:
        return True


def test_knowledge_fake_satisfies_only_knowledge_repository():
    fake = _KnowledgeFake()
    assert isinstance(fake, KnowledgeRepository)
    assert not isinstance(fake, Repository)


def test_knowledge_repository_rejects_missing_method_at_runtime():
    class _MissingCommit:
        async def get_memory_watermark(self, chat_key: ChatKey) -> MessageRowId | None:
            return None

    assert not isinstance(_MissingCommit(), KnowledgeRepository)


def test_knowledge_method_signatures():
    # The typed boundary surface: no SQL or raw rows cross the seam.
    wm = inspect.signature(KnowledgeRepository.get_memory_watermark).parameters
    assert list(wm) == ["self", "chat_key"]
    assert (
        inspect.signature(KnowledgeRepository.get_memory_watermark).return_annotation
        == "MessageRowId | None"
    )

    batch = inspect.signature(KnowledgeRepository.read_memory_source_batch).parameters
    assert list(batch) == ["self", "chat_key", "through_msg_id", "tail"]
    assert batch["through_msg_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert batch["tail"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        inspect.signature(KnowledgeRepository.read_memory_source_batch).return_annotation
        == "MemorySourceBatch | None"
    )

    commit = inspect.signature(KnowledgeRepository.commit_memory_source).parameters
    assert list(commit) == ["self", "request"]
    assert commit["request"].annotation == "MemoryWriteRequest"
    assert (
        inspect.signature(KnowledgeRepository.commit_memory_source).return_annotation
        == "bool"
    )

    query = inspect.signature(KnowledgeRepository.query_memory).parameters
    assert list(query) == ["self", "chat_key", "query", "limit"]
    assert query["limit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert query["limit"].default == 10
    assert (
        inspect.signature(KnowledgeRepository.query_memory).return_annotation
        == "list[LexicalHit]"
    )

    memories = inspect.signature(KnowledgeRepository.get_memories).parameters
    assert list(memories) == ["self", "chat_key", "memory_ids"]
    assert (
        inspect.signature(KnowledgeRepository.get_memories).return_annotation
        == "list[MemoryRecord]"
    )

    cas = inspect.signature(KnowledgeRepository.cas_person_profile).parameters
    assert list(cas) == [
        "self", "chat_key", "platform_uid", "expected_through_msg_id", "profile",
    ]
    assert cas["platform_uid"].annotation == "SenderId"
    assert cas["expected_through_msg_id"].annotation == "MessageRowId | None"
    assert cas["profile"].annotation == "PersonProfile"

    create = inspect.signature(
        KnowledgeRepository.create_embedding_generation
    ).parameters
    assert list(create) == ["self", "model", "dim", "revision", "state", "created_ts"]
    assert create["revision"].kind is inspect.Parameter.KEYWORD_ONLY
    assert create["revision"].default == "default"
    assert create["state"].kind is inspect.Parameter.KEYWORD_ONLY
    assert create["state"].default == "inactive"
    assert create["created_ts"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        inspect.signature(
            KnowledgeRepository.create_embedding_generation
        ).return_annotation
        == "EmbeddingGeneration"
    )

    upsert = inspect.signature(KnowledgeRepository.upsert_vector).parameters
    assert list(upsert) == ["self", "chat_key", "row"]
    assert upsert["row"].annotation == "VectorRow"

    listed = inspect.signature(KnowledgeRepository.list_vectors).parameters
    assert list(listed) == ["self", "chat_key", "model", "generation"]
    assert (
        inspect.signature(KnowledgeRepository.list_vectors).return_annotation
        == "list[VectorRow]"
    )


def test_create_embedding_generation_signature_compatible_across_impls():
    # Protocol/fake compatibility: the concrete SqliteRepository and the
    # knowledge fake both accept the protocol's ``state`` argument with the
    # same backward-compatible default, so any protocol-shaped caller (e.g.
    # the semantic backfill) works identically with either implementation.
    proto = inspect.signature(
        KnowledgeRepository.create_embedding_generation
    ).parameters
    for impl in (
        SqliteRepository.create_embedding_generation,
        _KnowledgeFake.create_embedding_generation,
    ):
        sig = inspect.signature(impl).parameters
        for name in proto:
            assert name in sig, f"{impl.__qualname__} missing param {name}"
            assert sig[name].kind is proto[name].kind, f"{impl.__qualname__}.{name}"
            assert sig[name].default == proto[name].default, (
                f"{impl.__qualname__}.{name} default mismatch"
            )


def test_knowledge_boundary_types_are_frozen_dataclasses():
    for cls in (MemoryRecord, MemorySourceBatch, MemoryWriteRequest,
                PersonProfile, EmbeddingGeneration, VectorRow, LexicalHit):
        assert dataclasses.is_dataclass(cls)
        assert getattr(cls, "__dataclass_params__").frozen


# ── Phase 6 AdaptiveRepository seam (separate protocol) ─────────────────────

# The full required surface, in the order the protocol declares it.
REQUIRED_ADAPTIVE_METHODS = (
    "acquire_learner_run",
    "renew_learner_run",
    "release_learner_run",
    "read_learner_source_batch",
    "commit_learner_source",
    "list_learner_records",
    "select_learner_records",
    "record_exposure",
    "increment_record_uses",
    "apply_record_feedback",
    "query_records",
    "get_learner_state",
    "list_learner_pending_chats",
    "list_learner_runs",
)


def test_adaptive_repository_protocol_declares_exactly_the_required_methods():
    protocol_attrs = _protocol_members(AdaptiveRepository)
    assert set(protocol_attrs) == set(REQUIRED_ADAPTIVE_METHODS)


def test_all_adaptive_repository_methods_are_coroutine_functions():
    for name in REQUIRED_ADAPTIVE_METHODS:
        fn = getattr(AdaptiveRepository, name)
        assert inspect.iscoroutinefunction(fn), f"{name} must be async"


def test_adaptive_repository_is_separate_from_repository_and_knowledge():
    # Neither the Phase 1 Repository seam nor the Phase 5 KnowledgeRepository
    # seam is enlarged: the adaptive surface is a distinct protocol, so an
    # adaptive-only fake is not a Repository/KnowledgeRepository and vice
    # versa.
    assert not set(REQUIRED_ADAPTIVE_METHODS) <= set(REQUIRED_METHODS)
    assert not set(REQUIRED_ADAPTIVE_METHODS) <= set(REQUIRED_KNOWLEDGE_METHODS)
    assert not set(REQUIRED_METHODS) <= set(REQUIRED_ADAPTIVE_METHODS)
    assert not set(REQUIRED_KNOWLEDGE_METHODS) <= set(REQUIRED_ADAPTIVE_METHODS)


def test_sqlite_repository_satisfies_all_three_protocols():
    repo = SqliteRepository.__new__(SqliteRepository)
    assert isinstance(repo, Repository)
    assert isinstance(repo, KnowledgeRepository)
    assert isinstance(repo, AdaptiveRepository)


def test_repository_fake_is_not_an_adaptive_repository():
    # The existing Repository-only fake must NOT satisfy the adaptive seam.
    assert isinstance(_FakeRepo(), Repository)
    assert not isinstance(_FakeRepo(), AdaptiveRepository)


class _AdaptiveFake:
    """A minimal adaptive-only fake: satisfies AdaptiveRepository but NOT
    Repository or KnowledgeRepository (it implements none of their
    surface)."""

    async def acquire_learner_run(
        self, request: LearnerRunRequest
    ) -> LearnerGrant | LearnerBusy | None:
        return None

    async def renew_learner_run(
        self, chat_key: ChatKey, learner: str, run_id: int, expires_at: float, *, now: float
    ) -> bool:
        return True

    async def release_learner_run(
        self, chat_key: ChatKey, learner: str, run_id: int
    ) -> None:
        pass

    async def read_learner_source_batch(
        self,
        chat_key: ChatKey,
        learner: str,
        *,
        through_msg_id: MessageRowId,
        tail: int,
        policy: str = "nonself",
    ) -> LearnerBatch | None:
        return None

    async def commit_learner_source(
        self, request: LearnerDraft, *, now: float
    ) -> bool:
        return True

    async def list_learner_records(
        self, chat_key: ChatKey, learner: str, *, limit: int = 100
    ) -> list[Record]:
        return []

    async def select_learner_records(
        self, chat_key: ChatKey, learner: str, *, limit: int = 10
    ) -> list[Record]:
        return []

    async def record_exposure(
        self,
        chat_key: ChatKey,
        learner: str,
        record_id: int,
        run_id: int,
        *,
        now: float,
    ) -> bool:
        return True

    async def increment_record_uses(
        self, chat_key: ChatKey, learner: str, record_id: int
    ) -> bool:
        return True

    async def apply_record_feedback(
        self,
        chat_key: ChatKey,
        learner: str,
        record_id: int,
        effect: float,
        *,
        now: float,
    ) -> float | None:
        return None

    async def query_records(
        self, chat_key: ChatKey, learner: str, query: str, *, limit: int = 10
    ) -> list[RecordHit]:
        return []

    async def get_learner_state(
        self, chat_key: ChatKey, learner: str
    ) -> LearnerState | None:
        return None

    async def list_learner_pending_chats(self, learner: str) -> list[ChatKey]:
        return []

    async def list_learner_runs(
        self, chat_key: ChatKey, learner: str, *, limit: int = 20
    ) -> list[LearnerRun]:
        return []


def test_adaptive_fake_satisfies_only_adaptive_repository():
    fake = _AdaptiveFake()
    assert isinstance(fake, AdaptiveRepository)
    assert not isinstance(fake, Repository)
    assert not isinstance(fake, KnowledgeRepository)


def test_adaptive_repository_rejects_missing_method_at_runtime():
    class _MissingCommit:
        async def acquire_learner_run(
            self, request: LearnerRunRequest
        ) -> LearnerGrant | LearnerBusy | None:
            return None

    assert not isinstance(_MissingCommit(), AdaptiveRepository)


def test_adaptive_method_signatures():
    # The typed boundary surface: no SQL or raw rows cross the seam.
    acquire = inspect.signature(AdaptiveRepository.acquire_learner_run).parameters
    assert list(acquire) == ["self", "request"]
    assert acquire["request"].annotation == "LearnerRunRequest"
    assert (
        inspect.signature(AdaptiveRepository.acquire_learner_run).return_annotation
        == "LearnerGrant | LearnerBusy | None"
    )

    renew = inspect.signature(AdaptiveRepository.renew_learner_run).parameters
    assert list(renew) == ["self", "chat_key", "learner", "run_id", "expires_at", "now"]
    assert renew["now"].kind is inspect.Parameter.KEYWORD_ONLY

    batch = inspect.signature(AdaptiveRepository.read_learner_source_batch).parameters
    assert list(batch) == ["self", "chat_key", "learner", "through_msg_id", "tail", "policy"]
    assert batch["through_msg_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert batch["tail"].kind is inspect.Parameter.KEYWORD_ONLY
    assert batch["policy"].kind is inspect.Parameter.KEYWORD_ONLY
    assert batch["policy"].default == "nonself"
    assert (
        inspect.signature(AdaptiveRepository.read_learner_source_batch).return_annotation
        == "LearnerBatch | None"
    )

    commit = inspect.signature(AdaptiveRepository.commit_learner_source).parameters
    assert list(commit) == ["self", "request", "now"]
    assert commit["request"].annotation == "LearnerDraft"
    assert commit["now"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        inspect.signature(AdaptiveRepository.commit_learner_source).return_annotation
        == "bool"
    )

    listed = inspect.signature(AdaptiveRepository.list_learner_records).parameters
    assert list(listed) == ["self", "chat_key", "learner", "limit"]
    assert listed["limit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert listed["limit"].default == 100

    selected = inspect.signature(AdaptiveRepository.select_learner_records).parameters
    assert list(selected) == ["self", "chat_key", "learner", "limit"]
    assert selected["limit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert selected["limit"].default == 10

    exposure = inspect.signature(AdaptiveRepository.record_exposure).parameters
    assert list(exposure) == ["self", "chat_key", "learner", "record_id", "run_id", "now"]
    assert exposure["now"].kind is inspect.Parameter.KEYWORD_ONLY

    feedback = inspect.signature(AdaptiveRepository.apply_record_feedback).parameters
    assert list(feedback) == ["self", "chat_key", "learner", "record_id", "effect", "now"]
    assert feedback["effect"].annotation == "float"
    assert feedback["now"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        inspect.signature(AdaptiveRepository.apply_record_feedback).return_annotation
        == "float | None"
    )

    query = inspect.signature(AdaptiveRepository.query_records).parameters
    assert list(query) == ["self", "chat_key", "learner", "query", "limit"]
    assert query["limit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert query["limit"].default == 10
    assert (
        inspect.signature(AdaptiveRepository.query_records).return_annotation
        == "list[RecordHit]"
    )

    state = inspect.signature(AdaptiveRepository.get_learner_state).parameters
    assert list(state) == ["self", "chat_key", "learner"]
    assert (
        inspect.signature(AdaptiveRepository.get_learner_state).return_annotation
        == "LearnerState | None"
    )

    runs = inspect.signature(AdaptiveRepository.list_learner_runs).parameters
    assert list(runs) == ["self", "chat_key", "learner", "limit"]
    assert runs["limit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert runs["limit"].default == 20
    assert (
        inspect.signature(AdaptiveRepository.list_learner_runs).return_annotation
        == "list[LearnerRun]"
    )


def test_adaptive_boundary_types_are_frozen_dataclasses():
    for cls in (LearnerBatch, LearnerDraft, LearnerRunRequest, LearnerGrant,
                LearnerBusy, LearnerState, LearnerRun, RecordHit):
        assert dataclasses.is_dataclass(cls)
        assert getattr(cls, "__dataclass_params__").frozen


# ── Phase 6 P6.5 MediaRepository seam (separate protocol) ───────────────────

# The full required surface, in the order the protocol declares it.
REQUIRED_MEDIA_METHODS = (
    "submit_media_candidate",
    "get_media_candidate",
    "list_media_candidates",
    "approve_media_candidate",
    "reject_media_candidate",
    "revoke_media_asset",
    "select_media_assets",
    "use_media_asset",
    "list_media_assets",
)


def test_media_repository_protocol_declares_exactly_the_required_methods():
    protocol_attrs = _protocol_members(MediaRepository)
    assert set(protocol_attrs) == set(REQUIRED_MEDIA_METHODS)


def test_all_media_repository_methods_are_coroutine_functions():
    for name in REQUIRED_MEDIA_METHODS:
        fn = getattr(MediaRepository, name)
        assert inspect.iscoroutinefunction(fn), f"{name} must be async"


def test_media_repository_is_separate_from_all_other_protocols():
    # No existing seam is enlarged: the media surface is a distinct protocol,
    # so a media-only fake is not a Repository/KnowledgeRepository/
    # AdaptiveRepository and vice versa.
    assert not set(REQUIRED_MEDIA_METHODS) <= set(REQUIRED_METHODS)
    assert not set(REQUIRED_MEDIA_METHODS) <= set(REQUIRED_KNOWLEDGE_METHODS)
    assert not set(REQUIRED_MEDIA_METHODS) <= set(REQUIRED_ADAPTIVE_METHODS)
    assert not set(REQUIRED_METHODS) <= set(REQUIRED_MEDIA_METHODS)
    assert not set(REQUIRED_KNOWLEDGE_METHODS) <= set(REQUIRED_MEDIA_METHODS)
    assert not set(REQUIRED_ADAPTIVE_METHODS) <= set(REQUIRED_MEDIA_METHODS)


def test_sqlite_repository_satisfies_all_four_protocols():
    repo = SqliteRepository.__new__(SqliteRepository)
    assert isinstance(repo, Repository)
    assert isinstance(repo, KnowledgeRepository)
    assert isinstance(repo, AdaptiveRepository)
    assert isinstance(repo, MediaRepository)


def test_repository_fake_is_not_a_media_repository():
    # The existing Repository-only fake must NOT satisfy the media seam.
    assert isinstance(_FakeRepo(), Repository)
    assert not isinstance(_FakeRepo(), MediaRepository)


class _MediaFake:
    """A minimal media-only fake: satisfies MediaRepository but NOT
    Repository/KnowledgeRepository/AdaptiveRepository (it implements none
    of their surface)."""

    async def submit_media_candidate(
        self, candidate: MediaAssetCandidate, *, now: float
    ) -> int:
        return 1

    async def get_media_candidate(
        self, chat_key: ChatKey, candidate_id: int
    ) -> MediaAssetCandidate | None:
        return None

    async def list_media_candidates(
        self, chat_key: ChatKey, *, kind: str | None = None, limit: int = 100
    ) -> list[MediaAssetCandidate]:
        return []

    async def approve_media_candidate(
        self, chat_key: ChatKey, candidate_id: int, *, capacity: int, now: float
    ) -> MediaAsset | None:
        return None

    async def reject_media_candidate(
        self, chat_key: ChatKey, candidate_id: int
    ) -> bool:
        return True

    async def revoke_media_asset(
        self, chat_key: ChatKey, asset_id: int, *, now: float
    ) -> bool:
        return True

    async def select_media_assets(
        self,
        chat_key: ChatKey,
        kind: str,
        *,
        limit: int = 1,
        cooldown_s: float = 0.0,
        now: float,
    ) -> list[MediaAsset]:
        return []

    async def use_media_asset(
        self, chat_key: ChatKey, asset_id: int, *, now: float
    ) -> bool:
        return True

    async def list_media_assets(
        self, chat_key: ChatKey, *, kind: str | None = None, limit: int = 100
    ) -> list[MediaAsset]:
        return []


def test_media_fake_satisfies_only_media_repository():
    fake = _MediaFake()
    assert isinstance(fake, MediaRepository)
    assert not isinstance(fake, Repository)
    assert not isinstance(fake, KnowledgeRepository)
    assert not isinstance(fake, AdaptiveRepository)


def test_media_repository_rejects_missing_method_at_runtime():
    class _MissingApprove:
        async def submit_media_candidate(
            self, candidate: MediaAssetCandidate, *, now: float
        ) -> int:
            return 1

    assert not isinstance(_MissingApprove(), MediaRepository)


def test_media_method_signatures():
    # The typed boundary surface: no SQL or raw rows cross the seam.
    submit = inspect.signature(MediaRepository.submit_media_candidate).parameters
    assert list(submit) == ["self", "candidate", "now"]
    assert submit["candidate"].annotation == "MediaAssetCandidate"
    assert submit["now"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        inspect.signature(MediaRepository.submit_media_candidate).return_annotation
        == "int"
    )

    get = inspect.signature(MediaRepository.get_media_candidate).parameters
    assert list(get) == ["self", "chat_key", "candidate_id"]
    assert (
        inspect.signature(MediaRepository.get_media_candidate).return_annotation
        == "MediaAssetCandidate | None"
    )

    listed = inspect.signature(MediaRepository.list_media_candidates).parameters
    assert list(listed) == ["self", "chat_key", "kind", "limit"]
    assert listed["kind"].kind is inspect.Parameter.KEYWORD_ONLY
    assert listed["kind"].default is None
    assert listed["limit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert listed["limit"].default == 100
    assert (
        inspect.signature(MediaRepository.list_media_candidates).return_annotation
        == "list[MediaAssetCandidate]"
    )

    approve = inspect.signature(MediaRepository.approve_media_candidate).parameters
    assert list(approve) == ["self", "chat_key", "candidate_id", "capacity", "now"]
    assert approve["capacity"].kind is inspect.Parameter.KEYWORD_ONLY
    assert approve["now"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        inspect.signature(MediaRepository.approve_media_candidate).return_annotation
        == "MediaAsset | None"
    )

    reject = inspect.signature(MediaRepository.reject_media_candidate).parameters
    assert list(reject) == ["self", "chat_key", "candidate_id"]

    revoke = inspect.signature(MediaRepository.revoke_media_asset).parameters
    assert list(revoke) == ["self", "chat_key", "asset_id", "now"]
    assert revoke["now"].kind is inspect.Parameter.KEYWORD_ONLY

    select = inspect.signature(MediaRepository.select_media_assets).parameters
    assert list(select) == ["self", "chat_key", "kind", "limit", "cooldown_s", "now"]
    assert select["limit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert select["limit"].default == 1
    assert select["cooldown_s"].kind is inspect.Parameter.KEYWORD_ONLY
    assert select["cooldown_s"].default == 0.0
    assert select["now"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        inspect.signature(MediaRepository.select_media_assets).return_annotation
        == "list[MediaAsset]"
    )

    use = inspect.signature(MediaRepository.use_media_asset).parameters
    assert list(use) == ["self", "chat_key", "asset_id", "now"]
    assert use["now"].kind is inspect.Parameter.KEYWORD_ONLY

    assets = inspect.signature(MediaRepository.list_media_assets).parameters
    assert list(assets) == ["self", "chat_key", "kind", "limit"]
    assert assets["kind"].kind is inspect.Parameter.KEYWORD_ONLY
    assert assets["kind"].default is None
    assert assets["limit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert assets["limit"].default == 100
    assert (
        inspect.signature(MediaRepository.list_media_assets).return_annotation
        == "list[MediaAsset]"
    )


def test_media_boundary_types_are_frozen_dataclasses():
    for cls in (MediaAssetCandidate, MediaAsset):
        assert dataclasses.is_dataclass(cls)
        assert getattr(cls, "__dataclass_params__").frozen
