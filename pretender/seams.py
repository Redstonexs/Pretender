"""The entire extension surface of Pretender, in one file.

A plugin author reads this file to learn every contract the runtime offers.
All Protocols are runtime_checkable so the registry can validate shape at
registration time — structural typing is compile-time only, and the design
requires boot-time validation.

These contracts are deliberately runtime-independent: no imports from
config, clock, or storage implementations. Implementations live in their
own modules; the protocols only name the shape.

Six ways to extend, cheapest first (from PLAN.md §3):
  1. prompt override (no code)     2. config override (no code)
  3. registry decorators           4. plugin discovery
  5. the three hooks               6. escape hatches (Adapter.call, raw payloads)
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Callable, Protocol, runtime_checkable

from pretender.types import (
    AdapterEvent,
    ChatControl,
    ChatIdentity,
    ChatKey,
    ChatState,
    ClaimBusy,
    ClaimGrant,
    CommitSeq,
    Contribution,
    CorpusMarker,
    CycleClaim,
    CycleFinish,
    CycleId,
    DecisionTrace,
    DispatchDeferred,
    DispatchGrant,
    DispatchId,
    DispatchRequest,
    DispatchSettle,
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
    LLMResponse,
    MediaAsset,
    MediaAssetCandidate,
    MemoryRecord,
    MemorySourceBatch,
    MemoryWriteRequest,
    Message,
    MessageId,
    MessageRowId,
    Outgoing,
    OutboxItem,
    PersonProfile,
    RecentSnapshot,
    Record,
    RecordHit,
    SelfId,
    SenderId,
    TranscriptMessage,
    VectorRow,
)


@runtime_checkable
class Adapter(Protocol):
    """A chat platform bridge. OneBot v11 / NapCat is the first implementation.

    ``capabilities`` is a frozenset of platform feature names the adapter
    supports, e.g. {"quote", "at", "image", "sticker", "recall", "history"}.
    ``send`` returns the platform message id, or None when the platform
    returns none (self-echo and effect tracking then fall back to a local id).
    ``call`` is the escape hatch to any platform API Pretender never modelled.
    """

    name: str
    capabilities: frozenset[str]

    async def connect(self) -> None: ...
    def events(self) -> AsyncIterator[AdapterEvent]: ...
    async def send(self, out: Outgoing) -> str | None: ...
    async def call(self, action: str, **params: Any) -> Any: ...


@runtime_checkable
class GateContext(Protocol):
    """Everything a GateFeature may read. The gate (phase 2) provides a
    concrete dataclass implementing this; features must not mutate it.

    The full frozen evaluation facts, all without direct database access:
    the claim/cycle identity and fixed local start/through row ids, the
    evaluation timestamp, the bot's structured self identity, the claimed
    pending message tuple (its count always equals ``pending``), the
    full-window recent facts alongside the limited ``recent`` list, the
    direct-address targeting booleans, the chat's exact merged gate
    configuration (including the idle-backoff base/cap/start facts the
    gate's controller is built from per evaluation), and the durable
    hold/idle state the trace/backoff policy reads.
    """

    chat_key: ChatKey
    cycle_id: CycleId
    start_msg_id: MessageRowId
    through_msg_id: MessageRowId
    evaluated_ts: float
    self_id: SelfId
    mode: str
    threshold: int
    trigger_score: int
    frequency: float
    backoff_base_s: float
    backoff_cap_s: float
    backoff_start_count: int
    pending: int
    pending_messages: tuple[Message, ...]
    recent: tuple[Message, ...]
    window_count: int
    self_count: int
    last_nonself_ts: float | None
    idle_seconds: float
    recent_average_interval: float
    self_ratio: float
    is_group: bool
    is_focused: bool
    last_message: Message | None
    self_name: str | None
    self_aliases: tuple[str, ...]
    has_direct_at: bool
    has_quote_to_self: bool
    has_other_assistant: bool
    hold_until: float | None
    idle_streak: int
    previous_end_reason: str | None


@runtime_checkable
class GateFeature(Protocol):
    """One pure scoring function. The gate score is nothing but the
    composition of registered features; the five built-ins register exactly
    like a third-party one, and every contribution lands in the
    DecisionTrace. ``op`` on the returned Contribution: max | add | scale.
    Return None to abstain."""

    name: str

    def contribute(self, ctx: GateContext) -> Contribution | None: ...


@runtime_checkable
class OutputStage(Protocol):
    """One step of the output pipeline, applied in ``order`` over a MUTABLE
    Outgoing (sanitize marks no-mutate spans, typo honours them, split
    rewrites parts)."""

    name: str
    order: int

    def apply(self, out: Outgoing) -> Outgoing: ...


@runtime_checkable
class LearnerDef(Protocol):
    """One declarative learner definition. All learners share one pipeline:
    prompt → JSON records → store → embed → select → inject → score_delta."""

    name: str
    prompt: str
    cadence_s: int

    def build_batch(self, repo: Repository, chat: ChatIdentity) -> str: ...
    def parse(self, raw: str) -> list[Record]: ...
    def render(self, selected: list[Record]) -> str: ...


@runtime_checkable
class Clock(Protocol):
    """Time source. RealClock for production, VirtualClock for deterministic
    tests (a 6-hour scheduling scenario in milliseconds, no busy polling)."""

    def now(self) -> float: ...
    def monotonic(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


@runtime_checkable
class LLMClient(Protocol):
    """The ONE OpenAI-compatible client abstraction. Providers are swapped
    by profile (base_url); tool-calling and vision ride the same call."""

    async def complete(
        self,
        messages: list[TranscriptMessage],
        *,
        profile: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        deadline: float | None = None,
    ) -> LLMResponse: ...


@runtime_checkable
class Embedder(Protocol):
    """Optional /embeddings access. When unconfigured, memory degrades to
    FTS-only recall — the client must degrade cleanly, never crash."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Repository(Protocol):
    """The storage seam — the only place SQL text lives is the phase-1
    durable lane's repo implementation. All I/O is asynchronous.

    Cursor movement is NOT expressible as an independent operation: the
    per-chat cursor advances only inside ``finish_cycle``, transactionally
    with the durable completed/held outcome (frozen decision #2), and
    always to the claim's stored through boundary — never to a
    caller-provided value. ``upsert_chat_state`` cannot move the cursor,
    the hold window, or the idle streak. Outbox sends are at-most-once:
    ``attempt_outbox`` is the durable compare-and-swap ``pending ->
    in_flight`` performed before the adapter is invoked, only ``in_flight``
    can become ``sent``, and an in-flight item is never auto-retried after
    a restart. Outbox rows are created ONLY by ``finish_cycle`` (terminal
    cycle completion with a completed durable claim); there is no
    standalone enqueue. An ambiguous in-flight send is reconciled ONLY
    through the trusted self-echo key flow inside ``ingest_message`` —
    there is no untrusted item-id reconciliation surface.

    The durable dispatch ledger (frozen Oracle advisory) is the minimal
    serialized dispatch order: ``ingest_message`` atomically creates each
    committed message's ``inbound_commits`` row; ``begin_dispatch``
    atomically claims a prepared dispatch, freezes the commit boundary,
    attaches eligible unassigned commits, and records the dispatch;
    ``settle_dispatch`` owns ALL release/delay/terminal movement — the
    cursor and the outbox move ONLY inside a terminal finish. The
    at-least-once export surface lists unexported commit/dispatch markers
    and marks one exported; the startup export appends markers then marks
    them exported, and readers deduplicate by (record_type, sequence).
    The legacy claim_cycle/finish_cycle surface remains for compatibility
    with the current cycle lane; the next integration lane switches all
    live use to the ledger.
    """

    # chats: identity and runtime state, looked up and upserted separately.
    # upsert_chat_state updates pacing/focus/config only — never the cursor,
    # the hold window, or the idle streak (those are written only by
    # finish_cycle, transactionally with the terminal outcome).
    async def get_chat(self, chat_key: ChatKey) -> ChatIdentity | None: ...
    async def upsert_chat(self, chat: ChatIdentity) -> None: ...
    async def get_chat_state(self, chat_key: ChatKey) -> ChatState | None: ...
    async def upsert_chat_state(self, state: ChatState) -> None: ...

    # messages: atomic identity+message commit returning a typed
    # IngestResult (durable row id, inserted flag, echo status, and the
    # atomic current pending non-self count for a newly inserted message —
    # None for noninserted/self).
    # ingest_message upserts the chat identity and inserts the message in
    # ONE transaction; ``inserted`` is False for a duplicate (platform,
    # self_id, platform_msg_id) row. Every newly inserted message —
    # self included — atomically creates its ``inbound_commits`` row in
    # that SAME transaction, stamped with the stable ``event_id``
    # (generated by the caller BEFORE recording; a caller that passes none
    # gets a generated one), the atomic pending count (non-self only), and
    # the commit's WakeKind (``inbound`` for a newly inserted non-self
    # message, ``none`` for a newly inserted self echo; duplicates commit
    # no new row). The result carries the event/commit/wake data:
    # ``event_id``, ``commit_seq``, ``wake_kind``, and durable ``priority``.
    # Callers provide structured direct/quote priority plus the exact merged
    # pending threshold; the repository computes high-pending priority in the
    # same transaction as the commit, so Scheduler never guesses whether an
    # ordinary wake may override a delayed timer.
    # may atomically reconcile an ambiguous in-flight send ONLY with the
    # trusted ``self_echo_delivery_key`` (the outbox row's
    # delivery/idempotency key, forwarded through the outgoing transport
    # metadata): the sender must match the chat's self id, the
    # chat/text/canonical segment/reply payload must match exactly one
    # in-flight row, and the transition records the real platform
    # id/timestamp — no synthetic echo is ever generated in this path.
    # Missing/untrusted keys are ``unproven``; mismatches, wrong
    # state/sender, or ambiguity are ``conflict`` and never move the
    # outbox; duplicate echo events reconcile idempotently.
    async def ingest_message(
        self,
        identity: ChatIdentity | None,
        msg: Message,
        *,
        self_echo_delivery_key: str | None = None,
        event_id: EventId | None = None,
        structural_priority: bool = False,
        pending_threshold: int | None = None,
    ) -> IngestResult: ...
    async def get_message(self, chat_key: ChatKey, msg_id: MessageId) -> Message | None: ...

    # gate: the single claim-bounded recent-message read the gate consumes.
    # Returns the LIMITED rendered list plus the FULL-window counts (self
    # messages included), the self count, and the last non-self timestamp —
    # the limited list never changes the counts.
    async def get_recent_snapshot(
        self, chat_key: ChatKey, through_row_id: MessageRowId, since_ts: float, limit: int
    ) -> RecentSnapshot: ...

    # startup recovery: every chat with pending work — at least one
    # NON-SELF message beyond its durable cursor (NULL cursor counts as 0).
    # Self messages never make a chat pending. Deterministic order; the
    # scheduler wakes exactly these chats immediately after a restart.
    async def list_pending_chats(self) -> list[ChatKey]: ...

    # cycles: claim_cycle is a compare-and-swap returning the bounded
    # pending data: a ClaimGrant when the claim succeeded, a ClaimBusy
    # (with the active owner's exact busy_until) when the chat is already
    # claimed by a live, unexpired cycle, and None when the chat is
    # unknown. Expired claims may be recovered (a grant). renew_cycle and
    # finish_cycle take ``now`` (absolute epoch) and fail for an expired
    # owner even before another claimant acts. finish_cycle advances the
    # cursor to the claim's stored through boundary after checking
    # ownership, start cursor, and unexpired lease, and inserts the
    # cycle's complete ordered outbox batch in the same transaction. The
    # durable hold window (``finish.hold_until``, None clears it) and the
    # idle streak (``finish.idle_streak_after``) are materialized in that
    # SAME transaction — a later save_session can never reintroduce a
    # crash gap for them. get_latest_terminal_end_reason reads the per-chat
    # latest TERMINAL cycle end reason (the gate's only history input);
    # released/expired claims never affect it.
    async def claim_cycle(self, claim: CycleClaim) -> ClaimGrant | ClaimBusy | None: ...
    async def renew_cycle(
        self, chat_key: ChatKey, cycle_id: CycleId, expires_at: float, *, now: float
    ) -> bool: ...
    async def release_cycle(self, chat_key: ChatKey, cycle_id: CycleId) -> None: ...
    async def finish_cycle(
        self, finish: CycleFinish, outbox: list[OutboxItem], *, now: float
    ) -> None: ...
    async def get_latest_terminal_end_reason(self, chat_key: ChatKey) -> str | None: ...

    # durable dispatch ledger (frozen Oracle advisory): the minimal
    # serialized dispatch order owned by Repository and driven by
    # Scheduler. begin_dispatch atomically claims (a prepared dispatch),
    # freezes the commit boundary, attaches eligible unassigned commits
    # (wake_kind != none, no dispatch membership, within the boundary),
    # and records the dispatch — returning a DispatchGrant, a ClaimBusy
    # (live, unexpired prepared owner with its exact busy_until), or None
    # (unknown chat, or an inbound dispatch with no eligible commits).
    # Priority wakes (timer/startup/busy_recovery) always create a
    # dispatch even with zero attached commits — the wake itself is the
    # work. Durable writer order resolves timer/inbound ties: a commit
    # that writes first joins the dispatch; a timer dispatch that writes
    # first excludes it. begin_dispatch additionally returns a
    # DispatchDeferred when the chat's durable agent barrier is still
    # active (agent_resume_at > now — no dispatch is created or attached,
    # and the scheduler re-arms at resume_at); an expired barrier is
    # cleared and the dispatch is granted normally. renew_dispatch extends
    # a prepared dispatch's lease, fenced to the same unexpired prepared
    # owner with a finite forward extension. settle_dispatch owns ALL
    # release/delay/defer/terminal movement: release and delay give the
    # claim back WITHOUT cursor or outbox movement; defer gives the claim
    # back, detaches the attached commits, and records the durable agent
    # barrier (a wait defer increments the wait streak, a retry defer does
    # not); finish is the ONLY path that advances the cursor (to the
    # dispatch's stored through boundary), creates outbox rows,
    # materializes the durable hold/idle state, clears the agent barrier
    # and resets the wait streak, and records the completed dispatch
    # trace. The at-least-once export surface lists unexported
    # commit/dispatch markers and marks one exported; the startup export
    # appends markers then marks them exported, and readers deduplicate by
    # (record_type, sequence). Dispatch markers carry the frozen
    # ``commit_boundary`` (the max inbound commit sequence at
    # begin_dispatch time, stored in the same transaction) and
    # ``scheduled_for`` (the timer deadline) so replay reconstructs the
    # exact attachment boundary independent of JSONL marker order.
    # ``list_unexported_dispatches`` returns ONLY settled, replayable
    # dispatches (``completed``/``released``) — a prepared/unevaluated
    # dispatch is never exported on startup — and each marker carries the
    # full frozen evaluation metadata from the durable row: the settled
    # ``state``, the ``settled_ts`` evaluation timestamp, the fixed
    # ``start_msg_id``/``through_msg_id`` message boundaries, the exact
    # attached ``CommitSeq`` tuple (frozen at ``begin_dispatch`` in the
    # same transaction, so a released/detached dispatch stays replayable),
    # the persisted ``trace_json``, the ``cause``, the frozen
    # ``commit_boundary``, and ``scheduled_for``.
    # list_unassigned_commits / list_ledger_pending_chats are the ledger's
    # crash-recovery scans.
    # No SQL or raw rows cross the seam. The legacy claim_cycle/
    # finish_cycle surface remains for compatibility with the current
    # cycle lane; the next integration lane switches all live use to this
    # ledger.
    async def begin_dispatch(
        self, request: DispatchRequest
    ) -> DispatchGrant | ClaimBusy | DispatchDeferred | None: ...
    async def renew_dispatch(
        self,
        chat_key: ChatKey,
        dispatch_id: DispatchId,
        cycle_id: CycleId,
        expires_at: float,
        *,
        now: float,
    ) -> bool: ...
    async def settle_dispatch(
        self, settle: DispatchSettle, outbox: list[OutboxItem], *, now: float
    ) -> None: ...
    async def list_unexported_commits(self) -> list[CorpusMarker]: ...
    async def list_unexported_dispatches(self) -> list[CorpusMarker]: ...
    async def mark_commit_exported(self, commit_seq: CommitSeq) -> None: ...
    async def mark_dispatch_exported(self, dispatch_id: DispatchId) -> None: ...
    async def list_unassigned_commits(self, chat_key: ChatKey) -> list[CommitSeq]: ...
    async def list_ledger_pending_chats(self) -> list[ChatKey]: ...

    # outbox: one row per adapter send, created only by finish_cycle.
    # list_ready_outbox returns pending items whose send_after_ts has
    # passed; next_due_outbox is the minimal seam the startup outbox
    # worker needs to schedule its next wake without polling (the earliest
    # send_after_ts among pending rows, or <= now when anything is already
    # due, or None when nothing is pending); attempt_outbox CASes one to
    # in_flight; mark_outbox_sent transitions ONLY in_flight -> sent
    # (writing the synthetic self echo); drop_outbox transitions ONLY
    # pending -> dropped. There is NO untrusted reconciliation surface:
    # an ambiguous in-flight send is reconciled ONLY through the trusted
    # self-echo key flow inside ingest_message.
    async def list_ready_outbox(
        self, chat_key: ChatKey, *, now: float, limit: int = 10
    ) -> list[OutboxItem]: ...
    async def next_due_outbox(self, chat_key: ChatKey, *, now: float) -> float | None: ...
    async def list_outbox_chats(self) -> list[ChatKey]: ...
    async def attempt_outbox(self, item_id: int, attempt_started_ts: float) -> bool: ...
    async def requeue_outbox(self, item_id: int) -> bool: ...
    async def mark_outbox_sent(
        self, item_id: int, platform_msg_id: MessageId | None, sent_ts: float
    ) -> bool: ...
    async def drop_outbox(self, item_id: int) -> bool: ...

    # records / kv / stats
    async def add_record(self, rec: Record) -> int: ...
    async def get_kv(self, k: str) -> str | None: ...
    async def set_kv(self, k: str, v: str) -> None: ...
    async def stats(self) -> dict[str, int]: ...

    async def close(self) -> None: ...


@runtime_checkable
class BudgetStore(Protocol):
    """The smallest durable seam for the daily budget ledger (PLAN.md §4, M6).

    ``SqliteRepository`` implements this over the ``kv`` table WITHOUT
    widening the legacy ``Repository`` protocol (a budget-only fake is not a
    ``Repository`` and vice versa). ``budget_update`` runs the whole
    load -> transform -> save inside ONE writer transaction, so reservations
    from DISTINCT ``BudgetManager`` instances over the same database are
    atomic against each other — the per-instance asyncio lock alone cannot
    serialize them. All budget policy (day rollover, malformed-KV
    reconciliation, rung decisions, serialization) stays in ``BudgetManager``;
    the store only atomically applies a pure ``transform`` to the raw value.
    """

    async def budget_update(
        self, key: str, *, transform: Callable[[str | None], str | None]
    ) -> str | None: ...


@runtime_checkable
class KnowledgeRepository(Protocol):
    """The Phase 5 knowledge storage seam — a SEPARATE protocol from
    ``Repository`` (the Phase 1 seam is deliberately not enlarged).
    ``SqliteRepository`` satisfies BOTH protocols; knowledge-specific test
    fakes implement only this one.

    Everything here is strictly chat-scoped and local-deterministic: no
    provider/network calls in any transaction, no cross-chat identity
    merges, no global nickname matching. Knowledge rows and the canonical
    FTS documents are authoritative local state; vectors are rebuildable
    derived state. SQL text lives only in the repo implementation.
    """

    # memory watermark: the durable per-chat cursor of summarized source.
    # None when the chat is unknown or nothing was summarized yet.
    async def get_memory_watermark(self, chat_key: ChatKey) -> MessageRowId | None: ...

    # the durable observed memory watermark: the watermark snapshot observed
    # when the last source batch was read, recorded at commit. None when the
    # chat is unknown or nothing was committed yet.
    async def get_memory_observed_watermark(
        self, chat_key: ChatKey
    ) -> MessageRowId | None: ...

    # read a fixed source batch bounded by the terminal cursor
    # (``through_msg_id``) and the durable watermark, capped to the OLDEST
    # bounded unsummarized chunk of at most ``tail`` messages (SQL-bounded
    # I/O — never fetch-all-then-slice). None when nothing is beyond the
    # watermark (or the chat is unknown).
    async def read_memory_source_batch(
        self, chat_key: ChatKey, *, through_msg_id: MessageRowId, tail: int
    ) -> MemorySourceBatch | None: ...

    # CAS commit of one memory source range: the memory records, their
    # canonical FTS documents, and the watermark advance happen in ONE
    # writer transaction, fenced on the expected watermark. False when the
    # watermark moved (stale CAS — nothing changes); raises for
    # cross-chat or hash/range violations.
    async def commit_memory_source(self, request: MemoryWriteRequest) -> bool: ...

    # local rebuild/backfill of the canonical memory FTS index for one
    # chat: tokenizes existing raw memory text with the repo's central
    # bigram_tokenize and transactionally rebuilds the index — a rebuild
    # reproduces exactly. Idempotent; marks the chat's bootstrap state.
    async def rebuild_memory_fts(self, chat_key: ChatKey) -> None: ...

    # idempotent canonical memory FTS bootstrap/backlog state for one chat:
    # ``(bootstrapped, backlog_through_msg_id)``, or None when unrecorded.
    async def get_memory_fts_state(
        self, chat_key: ChatKey
    ) -> tuple[bool, MessageRowId | None] | None: ...
    async def mark_memory_fts_backlog(
        self, chat_key: ChatKey, through_msg_id: MessageRowId
    ) -> None: ...

    # local FTS bootstrap/backlog enumeration (DB-start maintenance): chats
    # that have memory records but whose canonical memory FTS index has not
    # been bootstrapped (the set the local bootstrap rebuilds), and chats
    # with pending memory work (a crash-after-settlement gap where the
    # cursor advanced but the memory watermark did not) as
    # ``(chat_key, through_msg_id)`` pairs with the chat's current cursor as
    # the through boundary. Both are deterministic and idempotent.
    async def list_memory_fts_unbootstrapped_chats(self) -> list[ChatKey]: ...
    async def list_memory_pending_chats(
        self,
    ) -> list[tuple[ChatKey, MessageRowId]]: ...

    # bounded chat-safe memory FTS query (lexical-first recall).
    async def query_memory(
        self, chat_key: ChatKey, query: str, *, limit: int = 10
    ) -> list[LexicalHit]: ...

    # chat-safe bulk memory lookup: the memory records (with strength and
    # source range) for a set of memory ids, restricted to the chat, in
    # deterministic id order. Unknown or cross-chat ids are simply absent.
    async def get_memories(
        self, chat_key: ChatKey, memory_ids: list[int]
    ) -> list[MemoryRecord]: ...

    # chat-scoped memory enumeration (the smallest paged seam the semantic
    # backfill needs): every chat with at least one memory record, and one
    # chat's memory records — both deterministic and chat-scoped. The
    # backfill processes chats one at a time (bounded by budget/cancellation),
    # so the enumeration is paged at the chat level.
    async def list_memory_chats(self) -> list[ChatKey]: ...
    async def list_memories(self, chat_key: ChatKey) -> list[MemoryRecord]: ...
    async def list_memories_after(
        self, chat_key: ChatKey, after_id: int, *, limit: int
    ) -> list[MemoryRecord]: ...

    # keyset-paged/bounded enumeration for the semantic backfill: one bounded
    # page of memory chats strictly after ``after`` (``after=""`` yields the
    # first page), and the bounded chat-scoped vector rows for a bounded set
    # of memory ids (one memory page). The backfill never loads all of a
    # chat at once — maintenance stays bounded, cancellable, and fair.
    async def list_memory_chats_after(
        self, after: ChatKey, *, limit: int
    ) -> list[ChatKey]: ...
    async def list_vectors_for_memories(
        self, chat_key: ChatKey, model: str, generation: int, memory_ids: list[int]
    ) -> list[VectorRow]: ...

    # person identity: unique per (chat_key, platform_uid).
    async def get_person(
        self, chat_key: ChatKey, platform_uid: SenderId
    ) -> PersonProfile | None: ...
    async def upsert_person(self, profile: PersonProfile) -> None: ...

    # ATOMIC alias-only merge for one per-chat person: append ``name`` to
    # the alias list (first-seen order, deduped, bounded) WITHOUT touching
    # the profile/impression or the durable profile cursor. Creates the
    # person when unknown. Returns the resulting alias list, or None when
    # the name is blank. The PersonService uses this — never a
    # read-modify-upsert that could overwrite a concurrent profile write.
    async def add_person_alias(
        self,
        chat_key: ChatKey,
        platform_uid: SenderId,
        name: str,
        *,
        now: float | None = None,
    ) -> tuple[str, ...] | None: ...

    # CAS the durable profile cursor: updates the profile content AND the
    # cursor in one transaction, fenced on the expected cursor value. The
    # profile must target the same chat+UID and must not regress the stored
    # cursor. False when the cursor moved (stale CAS) or the person is
    # unknown.
    async def cas_person_profile(
        self,
        chat_key: ChatKey,
        platform_uid: SenderId,
        expected_through_msg_id: MessageRowId | None,
        profile: PersonProfile,
    ) -> bool: ...

    # embedding generations: model/dimension/revision generations with a
    # canonical space identity and an activation state, no provider side
    # effects. At most one active. ``create_embedding_generation`` is
    # idempotent per space_id (model + explicit revision). The optional
    # validated ``state`` (backward-compatible default ``inactive``)
    # admits ``inactive`` (manual/legacy creation) and ``building`` (the
    # semantic backfill's in-progress generation); ``active`` is reached
    # ONLY through ``activate_embedding_generation``, preserving the
    # at-most-one-active invariant.
    async def create_embedding_generation(
        self,
        model: str,
        dim: int,
        *,
        revision: str = "default",
        state: str = "inactive",
        created_ts: float | None = None,
    ) -> EmbeddingGeneration: ...
    async def get_embedding_generation(
        self, generation_id: int
    ) -> EmbeddingGeneration | None: ...
    async def activate_embedding_generation(self, generation_id: int) -> bool: ...
    async def list_embedding_generations(self) -> list[EmbeddingGeneration]: ...

    # source-fenced activation: activates the generation ONLY when every
    # current nonempty memory source has a valid matching vector (model/dim/
    # source_hash) for it, in ONE writer transaction. Returns None when
    # activated (the previous active generation is deactivated in the same
    # transaction — the current active generation is preserved until the
    # building generation completes); the deterministic repair-set of chats
    # whose memories are missing matching vectors when coverage is
    # incomplete (the generation stays ``building``); [] when the generation
    # does not exist or is not ``building`` (fail closed).
    async def activate_embedding_generation_if_complete(
        self, generation_id: int
    ) -> list[ChatKey] | None: ...

    # chat-scoped vector rows (derived state; rebuildable). The owner must
    # exist and belong to the chat; the generation must exist AND the row's
    # model/dim must match the generation's model/dim. Writes bump the
    # generation's durable vector_revision.
    async def upsert_vector(self, chat_key: ChatKey, row: VectorRow) -> None: ...
    async def get_vector(
        self,
        chat_key: ChatKey,
        owner_table: str,
        owner_id: int,
        model: str,
        generation: int,
    ) -> VectorRow | None: ...
    async def list_vectors(
        self, chat_key: ChatKey, model: str, generation: int
    ) -> list[VectorRow]: ...
    async def delete_vector(
        self,
        chat_key: ChatKey,
        owner_table: str,
        owner_id: int,
        model: str,
        generation: int,
    ) -> bool: ...


@runtime_checkable
class AdaptiveRepository(Protocol):
    """The Phase 6 adaptive storage seam — a SEPARATE protocol from
    ``Repository`` and ``KnowledgeRepository`` (neither is enlarged).
    ``SqliteRepository`` satisfies all three; adaptive-specific test fakes
    implement only this one.

    Everything here is strictly chat-scoped and local-deterministic: no
    provider/network calls in any transaction, no cross-chat identity
    merges. Learner rows, sources, and the canonical record FTS documents
    are authoritative local state; vectors are rebuildable derived state.
    SQL text lives only in the repo implementation.
    """

    # learner lease: acquire/recover/renew/release. acquire_learner_run is
    # a compare-and-swap returning a LearnerGrant when the claim succeeded,
    # a LearnerBusy (with the active owner's exact busy_until) when the
    # chat+learner already has a live, unexpired prepared run, and None
    # when the chat is unknown. Expired prepared runs are recovered (marked
    # expired and replaced) in the same transaction. renew_learner_run
    # takes ``now`` (absolute epoch) and fails for an expired owner even
    # before another claimant acts.
    async def acquire_learner_run(
        self, request: LearnerRunRequest
    ) -> LearnerGrant | LearnerBusy | None: ...
    async def renew_learner_run(
        self,
        chat_key: ChatKey,
        learner: str,
        run_id: int,
        expires_at: float,
        *,
        now: float,
    ) -> bool: ...
    async def release_learner_run(
        self, chat_key: ChatKey, learner: str, run_id: int
    ) -> None: ...

    # source-bounded read: the fixed source batch beyond the learner's
    # durable watermark, capped to the OLDEST bounded unsummarized chunk of
    # at most ``tail`` messages (SQL-bounded I/O — never fetch-all-then-
    # slice). ``policy`` is ``nonself`` (the default: ``is_self = 0`` is
    # enforced in SQL, so the bot's own output never enters a nonself
    # batch) or ``all``. None when nothing is beyond the watermark (or the
    # chat is unknown).
    async def read_learner_source_batch(
        self,
        chat_key: ChatKey,
        learner: str,
        *,
        through_msg_id: MessageRowId,
        tail: int,
        policy: str = "nonself",
    ) -> LearnerBatch | None: ...

    # exact source-batch hash/CAS commit: atomically inserts/merges the
    # validated records + their record_sources + the successful run + the
    # watermark advance in ONE writer transaction, fenced on the expected
    # watermark. A valid EMPTY result advances the watermark; a
    # malformed/cancelled outcome settles the run WITHOUT advancing the
    # watermark or inserting records. False when the watermark moved (stale
    # CAS — nothing changes); raises for cross-chat/hash/range violations.
    async def commit_learner_source(
        self, request: LearnerDraft, *, now: float
    ) -> bool: ...

    # chat+learner-scoped record reads excluding legacy (no content_hash)
    # and retired records, in deterministic order.
    async def list_learner_records(
        self, chat_key: ChatKey, learner: str, *, limit: int = 100
    ) -> list[Record]: ...
    async def select_learner_records(
        self, chat_key: ChatKey, learner: str, *, limit: int = 10
    ) -> list[Record]: ...

    # atomic idempotent exposure/uses: record_exposure inserts one
    # (record_id, run_id) exposure row exactly once (False on a duplicate);
    # increment_record_uses atomically bumps the record's uses counter.
    async def record_exposure(
        self,
        chat_key: ChatKey,
        learner: str,
        record_id: int,
        run_id: int,
        *,
        now: float,
    ) -> bool: ...
    async def increment_record_uses(
        self, chat_key: ChatKey, learner: str, record_id: int
    ) -> bool: ...

    # bounded effect feedback with a code-owned reweight clamped to
    # [0.1, 5]: returns the record's NEW weight, or None when the record is
    # unknown/cross-chat/legacy/retired.
    async def apply_record_feedback(
        self,
        chat_key: ChatKey,
        learner: str,
        record_id: int,
        effect: float,
        *,
        now: float,
    ) -> float | None: ...

    # bounded chat+learner-safe record FTS query (lexical-first recall over
    # the canonical record token documents).
    async def query_records(
        self, chat_key: ChatKey, learner: str, query: str, *, limit: int = 10
    ) -> list[RecordHit]: ...

    # state/list methods for bounded recovery.
    async def get_learner_state(
        self, chat_key: ChatKey, learner: str
    ) -> LearnerState | None: ...
    async def list_learner_pending_chats(
        self, learner: str, *, policy: str = "nonself", now: float | None = None
    ) -> list[ChatKey]: ...
    async def list_learner_runs(
        self, chat_key: ChatKey, learner: str, *, limit: int = 20
    ) -> list[LearnerRun]: ...


@runtime_checkable
class MediaRepository(Protocol):
    """The Phase 6 P6.5 media catalog storage seam — a SEPARATE protocol
    from ``Repository``, ``KnowledgeRepository``, and ``AdaptiveRepository``
    (none is enlarged). ``SqliteRepository`` satisfies all four;
    media-specific test fakes implement only this one.

    Everything here is strictly chat-scoped and local-deterministic: no
    provider/network calls in any transaction, no file fetch, no outbox/
    send, no plugin load. The catalog key is OPAQUE — validation rejects
    local paths, URLs, data/base64 payloads, and raw platform media
    references as catalog keys. Existing global ``emoji`` rows remain
    legacy/untrusted and are never read here. SQL text lives only in the
    repo implementation.
    """

    # candidate: submit one chat-scoped candidate (idempotent per
    # (chat, kind, sha256) — an existing row's status is never reset),
    # read one back, and list pending candidates.
    async def submit_media_candidate(
        self, candidate: MediaAssetCandidate, *, now: float
    ) -> int: ...
    async def get_media_candidate(
        self, chat_key: ChatKey, candidate_id: int
    ) -> MediaAssetCandidate | None: ...
    async def list_media_candidates(
        self, chat_key: ChatKey, *, kind: str | None = None, limit: int = 100
    ) -> list[MediaAssetCandidate]: ...

    # approval: capacity-safe transactional approval — the pending row is
    # transitioned to approved and, when the (chat, kind) approved count is
    # at ``capacity``, the least-recently-used approved rows are evicted in
    # the SAME transaction. Returns the approved MediaAsset, or None when
    # the candidate is unknown/cross-chat/rejected/revoked (an
    # already-approved row returns the existing approved asset).
    async def approve_media_candidate(
        self, chat_key: ChatKey, candidate_id: int, *, capacity: int, now: float
    ) -> MediaAsset | None: ...

    # rejection/revocation: idempotent terminal transitions (False when the
    # row is unknown, cross-chat, or not in the expected status).
    async def reject_media_candidate(
        self, chat_key: ChatKey, candidate_id: int
    ) -> bool: ...
    async def revoke_media_asset(
        self, chat_key: ChatKey, asset_id: int, *, now: float
    ) -> bool: ...

    # selection: APPROVED rows only, deterministic order (least-used first,
    # then least-recently-used, then id), cooldown aware (rows used within
    # ``cooldown_s`` of ``now`` are excluded).
    async def select_media_assets(
        self,
        chat_key: ChatKey,
        kind: str,
        *,
        limit: int = 1,
        cooldown_s: float = 0.0,
        now: float,
    ) -> list[MediaAsset]: ...

    # use: atomic idempotent uses bump + cooldown timestamp on an APPROVED
    # row. False when unknown/cross-chat/not approved.
    async def use_media_asset(
        self, chat_key: ChatKey, asset_id: int, *, now: float
    ) -> bool: ...

    # listing: chat-scoped asset rows (all statuses), deterministic id order.
    async def list_media_assets(
        self, chat_key: ChatKey, *, kind: str | None = None, limit: int = 100
    ) -> list[MediaAsset]: ...


@runtime_checkable
class ChatControlRepository(Protocol):
    """The Phase 6 P6.6b chat-control storage seam — a SEPARATE protocol
    from ``Repository``, ``KnowledgeRepository``, ``AdaptiveRepository``,
    and ``MediaRepository`` (none is enlarged). ``SqliteRepository``
    satisfies all five; chat-control test fakes implement only this one.

    Everything here is strictly chat-scoped and local-deterministic: no
    provider/network calls, no platform sends, no outbox writes. Controls
    are durable INTERNAL focus events that only make the TARGET chat's gate
    evaluate as focused — delivery still traverses the target chat's normal
    gate/cycle/outbox flow. SQL text lives only in the repo implementation.
    """

    # apply one control idempotently (UNIQUE dispatch_id + intent_seq). A
    # focus control transactionally clears every other ACTIVE focus control
    # for all chats on the same account (one focus per account). False when
    # the control is a duplicate, the target chat is unknown, or the target
    # is not on the same account (platform + self_id) as the source chat.
    async def apply_chat_control(self, control: ChatControl) -> bool: ...

    # active (ttl_until > now) controls for one chat, deterministic order.
    async def list_active_controls(
        self, chat_key: ChatKey, *, now: float
    ) -> list[ChatControl]: ...


# ── The three hooks ─────────────────────────────────────────────────────────
# Only three points exist (PLAN.md §3.5): on_event, pre_send, on_cycle_end.
# pre_gate/post_gate would duplicate @gate_feature; post_reply/pre_send would
# duplicate @stage. Hooks may be sync or async; the HookBus awaits when needed.

@runtime_checkable
class EventHook(Protocol):
    async def __call__(self, event: AdapterEvent) -> None: ...


@runtime_checkable
class PreSendHook(Protocol):
    """Return a modified Outgoing to replace it, or None to keep it as-is."""

    async def __call__(self, out: Outgoing) -> Outgoing | None: ...


@runtime_checkable
class CycleEndHook(Protocol):
    async def __call__(self, chat_key: ChatKey, trace: DecisionTrace, end_reason: str) -> None: ...


@runtime_checkable
class Plugin(Protocol):
    """A discovered plugin module. ``setup(api)`` is optional; when present
    it receives the disposable ``PluginAPI`` once, after all staging
    registries are populated. The API exposes ONLY the staging registries
    and the frozen Config — never the App, the repository, the adapter, or
    any raw client."""

    name: str

    def setup(self, app: Any) -> None: ...
