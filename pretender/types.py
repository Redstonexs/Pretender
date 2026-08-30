"""Boundary dataclasses for Pretender.

This module is the data contract between every layer: adapters, storage,
the gate, the LLM layer, learners, output stages. Per the design rules it
contains NO operational behavior — only harmless normalization (lowercasing
a segment kind) and validation (a transcript role must be one of the four).
Anything that computes, queries, or mutates state belongs in the module
that owns that behavior.

Frozen unless the design says otherwise. The one deliberate exception is
``Outgoing``: the output pipeline is mutation-shaped (sanitize marks spans,
typo honours them, split rewrites parts), so a frozen Outgoing would make
the OutputStage contract unimplementable.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Any, NewType

# ── Common identifiers ──────────────────────────────────────────────────────
# NewTypes: no runtime cost, but a wrong-typed argument is caught by static
# analysis and reads clearly at call sites.

ChatKey = NewType("ChatKey", str)          # "qq:group:123456" — the PK of chats
PlatformId = NewType("PlatformId", str)    # "qq" | "console" | ...
SelfId = NewType("SelfId", str)            # the bot's own account id on a platform
SenderId = NewType("SenderId", str)        # a human's account id on a platform
MessageId = NewType("MessageId", str)      # platform message id (str: OneBot ids are int32)
MessageRowId = NewType("MessageRowId", int)  # local messages.id — the durable cursor unit
ToolCallId = NewType("ToolCallId", str)    # provider tool_call id
CycleId = NewType("CycleId", str)          # one gate→reply saga; also the log contextvar
PersonKey = NewType("PersonKey", str)      # per-chat person identity
EventId = NewType("EventId", str)          # stable event id, generated BEFORE recording
CommitSeq = NewType("CommitSeq", int)      # inbound_commits.id — the monotonic commit sequence
DispatchId = NewType("DispatchId", int)    # dispatches.id — the monotonic dispatch sequence


# ── Adapter boundary ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Segment:
    """One piece of a message: text, image, face, at, reply, forward, ...

    ``kind`` is an open set (platforms invent segment types); it is only
    lowercased for consistency. ``data`` carries the typed fields for that
    kind (e.g. ``{"url": ...}`` for image); ``raw`` is the untouched
    platform payload for anything Pretender never modelled.
    """

    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", self.kind.lower())


@dataclass(frozen=True)
class Message:
    """A normalized inbound message, as the adapter hands it to ingest.

    ``id`` is the platform message id and may be None where the platform
    provides none (console adapter). ``row_id`` is the LOCAL durable row id
    assigned by the repository on insert — the unit the per-chat cursor
    advances over. The two are deliberately distinct: the platform id is
    what adapters and self-echo reconciliation speak, the row id is what
    storage and the cursor speak. ``text`` is the plain-text rendering the
    gate and context builder read; ``segments`` keep the full fidelity.
    """

    chat_key: ChatKey
    sender_id: SenderId
    sender_name: str
    is_self: bool
    text: str
    id: MessageId | None = None
    segments: tuple[Segment, ...] = ()
    reply_to: MessageId | None = None
    mentions: tuple[SenderId, ...] = ()
    recv_ts: float | None = None
    raw: Any = None
    row_id: MessageRowId | None = None


@dataclass(frozen=True)
class AdapterEvent:
    """Everything an adapter yields from ``events()``.

    ``type`` is one of ``message`` | ``notice`` | ``request`` | ``meta``.
    For ``message`` events ``payload`` is a ``Message``; for everything else
    it is the platform payload (or a later-phase typed notice). ``raw`` is
    always the untouched platform payload.
    """

    type: str
    payload: Any
    raw: Any = None
    ts: float | None = None


@dataclass
class Outgoing:
    """A message Pretender wants to send. MUTABLE by design — see module docstring.

    ``parts`` is written by the splitter stage (one reply → several
    messages); ``send_after_ts`` is the durable pacing timestamp (absolute
    epoch seconds); ``idem_key`` dedupes outbox items; ``platform_ref`` is
    arbitrary plugin-owned JSON that rides along untouched.
    ``delivery_key`` is the outbox item's delivery/idempotency key,
    forwarded through the outgoing transport metadata so a real platform
    echo can carry it back and prove an ambiguous send landed.
    """

    chat_key: ChatKey
    text: str = ""
    segments: list[Segment] = field(default_factory=list)
    reply_to: MessageId | None = None
    group_id: str | None = None
    send_after_ts: float | None = None
    idem_key: str | None = None
    delivery_key: str | None = None
    parts: list[str] | None = None
    skip_post_process: bool = False
    enable_splitter: bool | None = None
    enable_chinese_typo: bool | None = None
    platform_ref: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass(frozen=True)
class ChatIdentity:
    """The durable identity of one chat, mirroring the ``chats`` table."""

    chat_key: ChatKey
    platform: PlatformId
    self_id: SelfId
    kind: str  # "group" | "private"
    title: str | None = None


# ── Ingest boundary ─────────────────────────────────────────────────────────

class WakeKind:
    """Stable machine-readable wake kinds for committed inbound events
    (frozen dispatch-ledger spec). Consumers match on these exact strings,
    never on free-form text.

    - ``inbound``: a newly committed non-self message — the scheduler
      should wake for it.
    - ``timer`` / ``startup`` / ``busy_recovery``: scheduler-originated
      wakes (a timer deadline, a startup re-arm, a busy-horizon retry).
    - ``none``: committed but never wakes the scheduler — a self echo (the
      bot's own output is never pending) or a duplicate. Commits with
      ``none`` are never attached to a dispatch.
    """

    NONE = "none"
    INBOUND = "inbound"
    TIMER = "timer"
    STARTUP = "startup"
    BUSY_RECOVERY = "busy_recovery"


_WAKE_KINDS = frozenset(
    {
        WakeKind.NONE,
        WakeKind.INBOUND,
        WakeKind.TIMER,
        WakeKind.STARTUP,
        WakeKind.BUSY_RECOVERY,
    }
)


class DispatchCause:
    """Stable machine-readable dispatch causes (frozen dispatch-ledger
    spec): why a dispatch was begun. ``inbound`` dispatches attach eligible
    unassigned commits and are skipped when none exist; the priority wakes
    (``timer`` / ``startup`` / ``busy_recovery``) always create a dispatch
    even with zero attached commits — the wake itself is the work."""

    INBOUND = "inbound"
    TIMER = "timer"
    STARTUP = "startup"
    BUSY_RECOVERY = "busy_recovery"


_DISPATCH_CAUSES = frozenset(
    {
        DispatchCause.INBOUND,
        DispatchCause.TIMER,
        DispatchCause.STARTUP,
        DispatchCause.BUSY_RECOVERY,
    }
)


class EchoStatus:
    """Stable machine-readable self-echo reconciliation statuses (frozen
    spec: missing/untrusted keys are ``unproven``, never heuristically
    matched). Consumers match on these exact strings, never on free-form
    text.

    - ``not_applicable``: not a self message — no reconciliation attempted.
    - ``reconciled``: a verified self echo with the trusted delivery key
      atomically transitioned exactly one in-flight outbox row to ``sent``
      with the real platform id/timestamp.
    - ``already_reconciled``: a duplicate echo event whose outbox row was
      already reconciled — idempotent, no second transition.
    - ``unproven``: a self message without a trusted delivery key.
    - ``conflict``: a trusted key with no matching row, a wrong state or
      sender, a payload mismatch, or ambiguity — the outbox never moves.
    """

    NOT_APPLICABLE = "not_applicable"
    RECONCILED = "reconciled"
    ALREADY_RECONCILED = "already_reconciled"
    UNPROVEN = "unproven"
    CONFLICT = "conflict"


_ECHO_STATUSES = frozenset(
    {
        EchoStatus.NOT_APPLICABLE,
        EchoStatus.RECONCILED,
        EchoStatus.ALREADY_RECONCILED,
        EchoStatus.UNPROVEN,
        EchoStatus.CONFLICT,
    }
)


@dataclass(frozen=True)
class IngestResult:
    """The typed outcome of one ``Repository.ingest_message`` call.

    ``row_id`` is the durable local row id (None when nothing was
    committed — non-message events, unknown chats); ``inserted`` is False
    for a duplicate ``(platform, self_id, platform_msg_id)`` row;
    ``echo_status`` is the self-echo reconciliation verdict from
    ``EchoStatus`` (``not_applicable`` for ordinary inbound messages).

    ``pending_count`` is the atomic CURRENT pending non-self count for
    the chat — non-self messages beyond the durable cursor, INCLUDING a
    newly inserted message — computed by the repository in the SAME
    transaction as the insert, relative to the durable cursor. It is None
    whenever the count is not meaningful: nothing was inserted (duplicate,
    non-message event, unknown chat) or the message is self (the bot's own
    output is never pending).

    The dispatch-ledger fields (frozen Oracle advisory) are the event/
    commit/wake data the scheduler consumes: ``event_id`` is the stable
    event id generated BEFORE recording (the corpus event line and the
    durable commit metadata share one identity); ``commit_seq`` is the new
    ``inbound_commits`` monotonic sequence (None when nothing was
    committed — duplicates commit no new row); ``wake_kind`` is the
    commit's ``WakeKind`` — ``inbound`` for a newly inserted non-self
    message, ``none`` for a newly inserted self echo (duplicates carry
    None: nothing was committed). A commit with ``wake_kind`` ``none`` is
    never attached to a dispatch.
    """

    row_id: MessageRowId | None = None
    inserted: bool = False
    echo_status: str = EchoStatus.NOT_APPLICABLE
    pending_count: int | None = None
    event_id: EventId | None = None
    commit_seq: CommitSeq | None = None
    wake_kind: str | None = None
    priority: bool = False

    def __post_init__(self) -> None:
        if self.echo_status not in _ECHO_STATUSES:
            raise ValueError(f"invalid echo status: {self.echo_status!r}")
        if self.pending_count is not None and self.pending_count < 0:
            raise ValueError("pending_count must be nonnegative")
        if self.wake_kind is not None and self.wake_kind not in _WAKE_KINDS:
            raise ValueError(f"invalid wake kind: {self.wake_kind!r}")
        if self.commit_seq is not None and (
            isinstance(self.commit_seq, bool)
            or not isinstance(self.commit_seq, int)
            or self.commit_seq < 0
        ):
            raise ValueError("commit_seq must be a nonnegative integer")
        if not isinstance(self.priority, bool):
            raise ValueError("priority must be a bool")


# ── Gate boundary ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GateSnapshot:
    """The concrete claim-bounded snapshot ONE gate evaluation reads.

    Structurally satisfies the ``GateContext`` protocol in ``seams.py``:
    every field here is exactly a ``GateContext`` attribute, so a snapshot
    can be handed to any ``GateFeature.contribute``. The gate builds one
    snapshot per cycle from the claim-bounded ``ClaimGrant`` +
    ``RecentSnapshot`` plus the chat's exact configuration and durable
    state; features must not mutate it.

    The claim-bounded contract is enforced here, not assumed: ``pending``
    must equal ``len(pending_messages)`` (the claimed tuple is the only
    source of the count), and the full-window counts must stay consistent
    with the limited ``recent`` list. ``pending_messages`` excludes
    ``is_self`` (the bot's own output is never pending); ``recent``
    includes it (the presence penalty reads the full window).
    """

    # claim/cycle identity and the fixed local-row boundary (ClaimGrant)
    chat_key: ChatKey
    cycle_id: CycleId
    start_msg_id: MessageRowId
    through_msg_id: MessageRowId
    evaluated_ts: float
    # structured bot self identity
    self_id: SelfId
    # the exact configuration the gate evaluated
    mode: str
    threshold: int
    trigger_score: int
    frequency: float
    # claimed pending data: the count and the claimed tuple must agree
    pending: int
    pending_messages: tuple[Message, ...]
    # recent window: the LIMITED rendered list plus the FULL-window facts
    recent: tuple[Message, ...]
    window_count: int
    self_count: int
    last_nonself_ts: float | None
    # pacing / presence facts
    idle_seconds: float
    recent_average_interval: float
    self_ratio: float
    is_group: bool
    is_focused: bool
    last_message: Message | None
    # signal-derived and durable-state facts (fail-closed defaults)
    self_name: str | None = None
    #: Other names this bot answers to (MaiBot's ``bot.alias_names``).
    self_aliases: tuple[str, ...] = ()
    #: The chat is excluded by the ``[access]`` lists: the bot watches but
    #: never speaks here. Evaluated BEFORE the hard @-trigger — a mute a
    #: direct mention can override is not a control anyone would trust.
    muted: bool = False
    has_direct_at: bool = False
    has_quote_to_self: bool = False
    has_other_assistant: bool = False
    hold_until: float | None = None
    idle_streak: int = 0
    # the per-chat LATEST TERMINAL end reason (the previous cycle's durable
    # outcome) — the ONLY history input the idle-backoff policy reads. The
    # gate receives no separate history argument: this snapshot field is the
    # sole source, so a fresh hold can never be regenerated from stale
    # history while a durable hold is active.
    previous_end_reason: str | None = None
    # the chat's exact merged idle-backoff configuration (GateBackoffConfig
    # overlaid with the per-chat override). The gate builds its
    # IdleBackoffController from THESE facts per evaluation — never from a
    # constructor default — so two chats with different backoff configs
    # evaluate differently. ``threshold`` (already a snapshot field) is the
    # controller's high-pending bypass threshold.
    backoff_base_s: float = 15.0
    backoff_cap_s: float = 300.0
    backoff_start_count: int = 2

    def __post_init__(self) -> None:
        if len(self.pending_messages) != self.pending:
            raise ValueError(
                "pending count must match the claimed pending messages: "
                f"pending={self.pending}, claimed={len(self.pending_messages)}"
            )
        if not (0 <= self.self_count <= self.window_count):
            raise ValueError("self_count must be within [0, window_count]")
        if len(self.recent) > self.window_count:
            raise ValueError("limited recent list cannot exceed window_count")
        if not math.isfinite(self.evaluated_ts):
            raise ValueError("evaluated_ts must be finite")
        if not math.isfinite(self.backoff_base_s) or self.backoff_base_s < 0:
            raise ValueError("backoff_base_s must be finite and nonnegative")
        if not math.isfinite(self.backoff_cap_s) or self.backoff_cap_s < 0:
            raise ValueError("backoff_cap_s must be finite and nonnegative")
        if self.backoff_cap_s < self.backoff_base_s:
            raise ValueError("backoff_cap_s must be >= backoff_base_s")
        if (
            isinstance(self.backoff_start_count, bool)
            or not isinstance(self.backoff_start_count, int)
            or self.backoff_start_count < 0
        ):
            raise ValueError("backoff_start_count must be a nonnegative integer")


@dataclass(frozen=True)
class RecentSnapshot:
    """The claim-bounded recent-message read ``Repository.get_recent_snapshot``
    returns.

    ``messages`` is the LIMITED rendered list (at most ``limit`` rows);
    ``window_count`` and ``self_count`` are the FULL-window counts across the
    entire 300-second window — the limited list never changes them (frozen
    spec). ``since_ts`` and ``through_row_id`` are the window bounds the
    repository applied, so the gate can record exact snapshot bounds in the
    DecisionTrace.
    """

    chat_key: ChatKey
    messages: tuple[Message, ...] = ()
    window_count: int = 0
    self_count: int = 0
    last_nonself_ts: float | None = None
    since_ts: float | None = None
    through_row_id: MessageRowId | None = None

    def __post_init__(self) -> None:
        if not (0 <= self.self_count <= self.window_count):
            raise ValueError("self_count must be within [0, window_count]")
        if len(self.messages) > self.window_count:
            raise ValueError("limited messages cannot exceed window_count")


@dataclass(frozen=True)
class Contribution:
    """One GateFeature's verdict. ``op`` is how it combines: max | add | scale.

    ``error`` records a feature FAILURE: the gate catches the exception and
    still traces it (feature failures delay safely); a normal verdict leaves
    it None.
    """

    feature: str
    op: str
    value: float
    reason: str | None = None
    error: str | None = None


class Reason:
    """Stable machine-readable gate reasons (frozen spec: the trace ends with
    a machine-readable final reason). Tokens are stable across releases —
    consumers must match on these exact strings, never on free-form text."""

    TRIGGER = "trigger"  # direct @ or quote-to-self hard trigger
    REFUSAL = "refusal"  # other-assistant refusal outranks scaling/backoff
    FEATURE_FAILURE = "feature_failure"  # a feature failed; delay safely
    BACKOFF = "backoff"  # group-only idle backoff applied
    MODE = "mode"  # mode selection (frequency mode)
    DELAY = "delay"  # timed or event-only delay
    SKIP = "skip"  # no action this cycle
    MUTED = "muted"  # the [access] lists exclude this chat


@dataclass(frozen=True)
class Decision:
    """The gate's final verdict for one cycle.

    ``action`` is ``trigger`` | ``delay`` | ``skip``. ``delay`` carries
    ``delay_seconds`` which the scheduler sleeps on — never busy-polling.
    A delay may be EVENT-ONLY: ``delay_seconds=None`` means the cycle still
    releases its claim and the trace records the event, but no sleep is
    scheduled. ``reason`` is a stable machine-readable token from ``Reason``.
    """

    action: str
    score: float = 0.0
    delay_seconds: float | None = None
    pending: int = 0
    reason: str | None = None


@dataclass(frozen=True)
class BackoffFacts:
    """Idle-backoff facts recorded in the DecisionTrace.

    ``applied`` is True when the group-only idle backoff delayed the cycle
    (or an ACTIVE durable hold produced its remaining duration);
    ``seconds`` is the backoff duration. ``bypass_reason`` names why backoff
    was NOT applied: "focus", "high_pending" (backoff is bypassed by focus
    or high pending — including an active durable hold), "not_group",
    "hard_trigger", "non_idle", or "expired_hold" (a durable hold that
    expired is ignored and never regenerates a fresh hold from stale
    history).
    """

    applied: bool = False
    seconds: float | None = None
    bypass_reason: str | None = None


@dataclass(frozen=True)
class DecisionTrace:
    """The full evidence trail for one gate evaluation.

    The score is nothing but the composition of ``contributions``; the trace
    is what ``replay --sweep`` re-scores and what ``on_cycle_end`` hooks see.
    Every field is JSON-native (or a frozen dataclass of JSON-native fields),
    so the trace is replay-safe: ``dataclasses.asdict`` + ``json.dumps``
    round-trips without loss.

    ``snapshot_facts`` carries the claim-bounded snapshot bounds/counts/facts
    (since_ts, through_row_id, window_count, self_count, last_nonself_ts, ...);
    ``config`` carries the exact configuration the gate evaluated;
    ``aggregates`` carries the composed per-op totals (max/add/scale);
    ``backoff`` carries the idle-backoff facts.
    """

    chat_key: ChatKey
    mode: str
    threshold: int
    trigger_score: int
    pending: int
    contributions: tuple[Contribution, ...] = ()
    decision: Decision | None = None
    ts: float | None = None
    snapshot_facts: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    aggregates: dict[str, float] = field(default_factory=dict)
    backoff: BackoffFacts | None = None


# ── Tool / provider transcript boundary ─────────────────────────────────────

@dataclass(frozen=True)
class ToolCall:
    """A provider tool_call. ``visibility`` and ``chat_scope`` gate whether
    the call is emitted in the schema at all (deferred tools stay out until
    tool_search activates them).

    ``raw_arguments`` preserves the provider's ORIGINAL ``arguments`` value
    when it could not be parsed into an object (a malformed JSON string, or
    a non-object value). It is None for a clean call. The tolerant parser
    (``toolparse``) uses it to run its one-repair / ``no_action`` degrade
    path instead of the LLM layer aborting on a malformed payload, and the
    wire serializer re-emits it so the transcript round-trips legally.
    """

    id: ToolCallId
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: Any = None
    visibility: str = "visible"  # "visible" | "deferred" | "hidden"
    chat_scope: str = "all"      # "all" | "group" | "private"


@dataclass(frozen=True)
class ToolResult:
    """The typed answer to one tool call. Exactly one of these must exist
    per ToolCall id before any provider call (frozen decision #4)."""

    call_id: ToolCallId
    name: str
    ok: bool = True
    content: str = ""
    data: Any = None
    error: str | None = None


@dataclass(frozen=True)
class TranscriptMessage:
    """One entry of the canonical provider transcript.

    Frozen decision #4: the transcript is a typed role sequence — an
    assistant tool-call turn is followed by exactly one typed tool result
    per call id. Serialization adapters validate this representation before
    a provider call; this dataclass is what they validate.

    Fail-closed role shape: ``tool_calls`` only on ``assistant`` messages,
    ``tool_call_id``/``name`` only on ``tool`` messages (a ``tool`` message
    REQUIRES ``tool_call_id``), and an assistant message's tool-call ids
    must be unique — a duplicate id could otherwise yield duplicate tool
    results after normalization.
    """

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: ToolCallId | None = None
    name: str | None = None  # tool name, for role == "tool"

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"invalid transcript role: {self.role!r}")
        if self.role == "tool":
            if self.tool_call_id is None:
                raise ValueError("tool message requires tool_call_id")
            if self.tool_calls:
                raise ValueError("tool message cannot carry tool_calls")
        else:
            if self.tool_call_id is not None:
                raise ValueError(
                    f"{self.role!r} message cannot carry tool_call_id"
                )
            if self.name is not None:
                raise ValueError(f"{self.role!r} message cannot carry a tool name")
            if self.role != "assistant" and self.tool_calls:
                raise ValueError(f"{self.role!r} message cannot carry tool_calls")
        if self.role == "assistant":
            ids = [c.id for c in self.tool_calls]
            if len(ids) != len(set(ids)):
                raise ValueError(
                    "assistant tool_calls must have unique ids: "
                    f"{sorted(ids, key=str)}"
                )


@dataclass(frozen=True)
class LLMResponse:
    """The typed result of one provider completion call."""

    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


# ── Storage boundary ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChatState:
    """The durable RUNTIME state of one chat, mirroring the runtime columns
    of the ``chats`` table. Identity lives in ``ChatIdentity`` (which stays
    identity-only); this is the part that changes as the bot lives: where
    the per-chat cursor sits, focus/hold windows, pacing statistics and the
    chat's config overlay. Immutable in memory; the repository swaps whole
    states.

    ``cursor_msg_id``, ``hold_until``, and ``idle_streak`` are READ-ONLY
    views: they are written only by ``finish_cycle`` (derived from the
    stored claim boundary and the terminal outcome), and
    ``upsert_chat_state`` never touches them.

    ``agent_resume_at`` and ``wait_streak`` are the durable agent barrier
    (frozen Oracle advisory): ``agent_resume_at`` is the absolute-epoch time
    before which no agent may run (a wait/retry defer barrier), and
    ``wait_streak`` is the consecutive-wait counter. Both are DEFER/TERMINAL
    OWNED — written only by ``settle_dispatch`` (a defer records the barrier
    and a wait increments the streak) and cleared/reset by a terminal
    finish. ``upsert_chat_state`` never touches them, so the session layer
    cannot claim persistence that bypasses defer/terminal ownership.
    """

    chat_key: ChatKey
    cursor_msg_id: MessageRowId | None = None
    focus_until: float | None = None
    hold_until: float | None = None
    avg_interval: float | None = None
    idle_streak: int = 0
    cfg_json: str | None = None
    agent_resume_at: float | None = None
    wait_streak: int = 0


@dataclass(frozen=True)
class CycleClaim:
    """The durable claim a cycle owns (frozen decision #2). A cycle claims
    its chat before gate evaluation and must release or finish; an expired
    pre-send claim may be recovered by the next cycle.

    The lease is MANDATORY and finite: ``expires_at`` is an absolute epoch
    timestamp strictly after ``started_ts``. The local-row boundary (start
    through fixed-through) is captured by the repository at claim time and
    returned in the ``ClaimGrant`` — the caller cannot choose it.
    """

    chat_key: ChatKey
    cycle_id: CycleId
    started_ts: float
    expires_at: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.started_ts) and math.isfinite(self.expires_at)):
            raise ValueError("claim timestamps must be finite")
        if self.expires_at <= self.started_ts:
            raise ValueError(
                "claim lease must be finite: expires_at must be > started_ts"
            )


@dataclass(frozen=True)
class ClaimGrant:
    """The durable grant of one cycle claim: the claim itself, its fixed
    local-row boundary (``start_msg_id`` exclusive .. ``through_msg_id``
    inclusive), and the claimed pending messages (``is_self`` excluded).
    New arrivals after the claim stay pending for the next claim."""

    claim: CycleClaim
    start_msg_id: MessageRowId
    through_msg_id: MessageRowId
    pending: tuple[Message, ...] = ()


@dataclass(frozen=True)
class ClaimBusy:
    """The typed ``already owned`` outcome of one ``claim_cycle`` call.

    Returned (instead of a bare None) when the chat is already claimed by
    a LIVE, unexpired cycle: ``busy_until`` is the exact absolute-epoch
    expiry of the active owner's lease, so the caller can distinguish
    ``busy`` from ``no work / unknown chat`` (None) and from a grant.
    ``cycle_id`` names the active owner without exposing raw claim rows.
    An expired claim is never reported busy — it is recovered and the
    caller receives a ``ClaimGrant`` instead.
    """

    chat_key: ChatKey
    cycle_id: CycleId
    busy_until: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.busy_until):
            raise ValueError("busy_until must be finite")


@dataclass(frozen=True)
class CycleFinish:
    """The durable completed/held outcome of one cycle. The per-chat cursor
    advances ONLY with this record — never at cycle start (frozen decision
    #2) — and always to the claim's stored through boundary, never to a
    caller-provided value. ``hold_until`` is the durable absolute-epoch
    delay of a held (backoff) outcome; it is distinct from the chat's
    ``focus_until`` (focus mode) in types, schema, and repository.
    ``idle_streak_after`` is the durable idle streak AFTER this cycle: idle
    backoff is materialized transactionally at terminal completion as
    ``idle_streak_after`` + absolute ``hold_until``, never recomputed from
    stale history. A terminal reset (skip / dry-run trigger) passes
    ``hold_until=None`` and ``idle_streak_after=0``, which clears the hold
    and resets the streak in the same transaction as the cursor advance."""

    chat_key: ChatKey
    cycle_id: CycleId
    end_reason: str
    hold_until: float | None = None
    idle_streak_after: int = 0
    trace_json: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0


# ── Durable dispatch ledger boundary (frozen Oracle advisory) ───────────────
# The minimal serialized dispatch ledger owned by Repository and driven by
# Scheduler: inbound_commits records each committed inbound event's
# monotonic sequence, event ID, chat/message identity, wake kind, pending
# count, and eventual dispatch membership; dispatches records the total
# order of prepared and completed inbound/timer/startup/busy-recovery
# dispatches. Durable writer order resolves timer/inbound ties: a commit
# that writes first joins the dispatch; a timer dispatch that writes first
# excludes it. These types are the ONLY surface callers see — no SQL or
# raw rows ever cross the seam.

@dataclass(frozen=True)
class CorpusMarker:
    """One typed commit/dispatch marker line in the JSONL corpus.

    The at-least-once export writes the marker AFTER the durable state it
    describes and marks it exported afterward; a crash between the two is
    repaired by the startup export, which re-appends the marker. Readers
    deduplicate by ``(record_type, sequence)`` — first occurrence wins.

    ``record_type`` is ``"commit"`` (an ``inbound_commits`` row: carries
    ``event_id`` and ``wake_kind``) or ``"dispatch"`` (a ``dispatches``
    row: carries ``cause``). ``sequence`` is the ``CommitSeq`` or
    ``DispatchId`` of the durable row.

    Dispatch markers additionally carry the frozen attachment boundary:
    ``commit_boundary`` is the maximum ``CommitSeq`` at ``begin_dispatch``
    time (the exact boundary replay needs to reconstruct which commits the
    live dispatch included, independent of JSONL marker order) and
    ``scheduled_for`` is the scheduled time (the timer deadline that
    triggered the dispatch; None otherwise). Old v2 markers lack both —
    they read back as None and remain fully readable.

    The v4 replayable settled-dispatch contract adds the full frozen
    evaluation metadata to dispatch markers: ``state`` is the settled
    state (``"completed"`` or ``"released"`` — only settled dispatches are
    ever exported, never a prepared/unevaluated dispatch); ``settled_ts``
    is the dispatch/evaluation timestamp (the absolute-epoch settlement
    time); ``start_msg_id``/``through_msg_id`` are the fixed local-row
    message boundaries captured at claim time; ``attached`` is the EXACT
    attached inbound ``CommitSeq`` tuple frozen at ``begin_dispatch`` (so
    a released dispatch stays replayable even after its live commit rows
    were detached); and ``trace_json`` is the persisted evaluation trace.
    Old v2/v3 markers lack these fields — they read back as None/empty and
    remain fully readable.
    """

    record_type: str  # "commit" | "dispatch"
    sequence: int  # CommitSeq for commit markers, DispatchId for dispatch markers
    chat_key: ChatKey
    event_id: EventId | None = None  # commit markers only
    wake_kind: str | None = None  # commit markers only
    message_row_id: MessageRowId | None = None  # commit markers only
    priority: bool = False  # commit markers only
    cause: str | None = None  # dispatch markers only
    commit_boundary: CommitSeq | None = None  # dispatch markers only
    scheduled_for: float | None = None  # dispatch markers only
    state: str | None = None  # dispatch markers only: "completed" | "released"
    settled_ts: float | None = None  # dispatch markers only: evaluation timestamp
    start_msg_id: MessageRowId | None = None  # dispatch markers only
    through_msg_id: MessageRowId | None = None  # dispatch markers only
    attached: tuple[CommitSeq, ...] = ()  # dispatch markers only: exact membership
    trace_json: str | None = None  # dispatch markers only
    evaluated_ts: float | None = None  # dispatch markers only
    snapshot_json: str | None = None  # dispatch markers only

    def __post_init__(self) -> None:
        if self.record_type not in ("commit", "dispatch"):
            raise ValueError(f"invalid marker record_type: {self.record_type!r}")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("marker sequence must be a nonnegative integer")
        if self.record_type == "commit":
            if self.event_id is None:
                raise ValueError("commit marker requires event_id")
            if self.wake_kind not in _WAKE_KINDS:
                raise ValueError(f"invalid marker wake_kind: {self.wake_kind!r}")
            if self.message_row_id is not None and (
                isinstance(self.message_row_id, bool)
                or not isinstance(self.message_row_id, int)
                or self.message_row_id < 0
            ):
                raise ValueError("commit marker message_row_id must be a nonnegative integer")
            if not isinstance(self.priority, bool):
                raise ValueError("commit marker priority must be a bool")
        else:
            if self.cause not in _DISPATCH_CAUSES:
                raise ValueError(f"invalid marker cause: {self.cause!r}")
            if self.commit_boundary is not None and (
                isinstance(self.commit_boundary, bool)
                or not isinstance(self.commit_boundary, int)
                or self.commit_boundary < 0
            ):
                raise ValueError(
                    "dispatch marker commit_boundary must be a nonnegative integer"
                )
            if self.scheduled_for is not None and not math.isfinite(
                self.scheduled_for
            ):
                raise ValueError("dispatch marker scheduled_for must be finite")
            if self.state is not None and self.state not in ("completed", "released"):
                raise ValueError(
                    "dispatch marker state must be 'completed' or 'released'"
                )
            if self.settled_ts is not None and not math.isfinite(self.settled_ts):
                raise ValueError("dispatch marker settled_ts must be finite")
            if self.evaluated_ts is not None and not math.isfinite(self.evaluated_ts):
                raise ValueError("dispatch marker evaluated_ts must be finite")
            for name, value in (
                ("start_msg_id", self.start_msg_id),
                ("through_msg_id", self.through_msg_id),
            ):
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError(
                        f"dispatch marker {name} must be a nonnegative integer"
                    )
            for seq in self.attached:
                if (
                    isinstance(seq, bool)
                    or not isinstance(seq, int)
                    or seq < 0
                ):
                    raise ValueError(
                        "dispatch marker attached sequences must be nonnegative integers"
                    )


@dataclass(frozen=True)
class DispatchRequest:
    """What the scheduler passes to ``Repository.begin_dispatch``.

    ``cause`` is the ``DispatchCause``: an ``inbound`` dispatch attaches
    eligible unassigned commits and is skipped (None) when none exist; the
    priority wakes (``timer`` / ``startup`` / ``busy_recovery``) always
    create a dispatch even with zero attached commits. ``wake_kind`` is
    the ``WakeKind`` recorded on the dispatch row (defaults to the cause);
    ``scheduled_ts`` is the timer deadline that triggered a timer dispatch
    (None otherwise). ``started_ts``/``expires_at`` are the mandatory
    finite lease, ``now`` the absolute-epoch evaluation time used for
    expiry/recovery decisions.
    """

    chat_key: ChatKey
    cause: str
    cycle_id: CycleId
    started_ts: float
    expires_at: float
    now: float
    wake_kind: str | None = None
    scheduled_ts: float | None = None

    def __post_init__(self) -> None:
        if self.cause not in _DISPATCH_CAUSES:
            raise ValueError(f"invalid dispatch cause: {self.cause!r}")
        if self.wake_kind is not None and self.wake_kind not in _WAKE_KINDS:
            raise ValueError(f"invalid wake kind: {self.wake_kind!r}")
        if not (
            math.isfinite(self.started_ts)
            and math.isfinite(self.expires_at)
            and math.isfinite(self.now)
        ):
            raise ValueError("dispatch timestamps must be finite")
        if self.expires_at <= self.started_ts:
            raise ValueError(
                "dispatch lease must be finite: expires_at must be > started_ts"
            )
        if self.scheduled_ts is not None and not math.isfinite(self.scheduled_ts):
            raise ValueError("scheduled_ts must be finite")


@dataclass(frozen=True)
class DispatchGrant:
    """The durable grant of one ``begin_dispatch`` call.

    ``dispatch_id`` is the new ``dispatches`` row id; ``claim`` is the
    mandatory finite lease (chat/cycle/timestamps); ``start_msg_id`` and
    ``through_msg_id`` are the fixed local-row boundary captured at claim
    time (the cursor advances only to ``through_msg_id``, inside terminal
    settlement); ``attached`` is the tuple of eligible unassigned commit
    sequences the dispatch claimed (empty for a priority wake with no
    commits); ``pending`` is the corresponding non-self messages in row
    order — the claimed pending data the gate reads. ``commit_boundary``
    is the frozen maximum inbound ``CommitSeq`` at ``begin_dispatch`` time
    (stored on the dispatch row in the same transaction — replay's exact
    attachment boundary independent of JSONL marker order);
    ``scheduled_for`` is the scheduled time (the timer deadline that
    triggered the dispatch; None otherwise).

    The v4 replayable settled-dispatch contract adds the remaining frozen
    metadata the grant carries: ``cause`` is the ``DispatchCause`` the
    dispatch was begun with (so the marker export never has to derive it
    from the scheduled time) and ``claimed_ts`` is the absolute-epoch
    ``begin_dispatch`` time (the dispatch timestamp). Both are persisted
    on the durable row in the same transaction.
    """

    dispatch_id: DispatchId
    claim: CycleClaim
    start_msg_id: MessageRowId
    through_msg_id: MessageRowId
    attached: tuple[CommitSeq, ...] = ()
    pending: tuple[Message, ...] = ()
    commit_boundary: CommitSeq = CommitSeq(0)
    scheduled_for: float | None = None
    cause: str | None = None
    claimed_ts: float | None = None
    priority: bool = False

    def __post_init__(self) -> None:
        if self.cause is not None and self.cause not in _DISPATCH_CAUSES:
            raise ValueError(f"invalid dispatch cause: {self.cause!r}")
        if self.claimed_ts is not None and not math.isfinite(self.claimed_ts):
            raise ValueError("claimed_ts must be finite")
        if not isinstance(self.priority, bool):
            raise ValueError("dispatch priority must be a bool")


@dataclass(frozen=True)
class DispatchDeferred:
    """The typed ``begin_dispatch`` outcome when the chat's durable agent
    barrier is still active (``agent_resume_at > now``): no dispatch is
    created or attached. ``resume_at`` is the exact absolute-epoch time the
    barrier expires — the scheduler re-arms there and no agent runs early
    (priority input cannot invoke it). ``defer_kind`` is ``"wait"`` |
    ``"retry"`` (the barrier's origin; the scheduler only reads
    ``resume_at``).
    """

    chat_key: ChatKey
    resume_at: float
    defer_kind: str  # "wait" | "retry"

    def __post_init__(self) -> None:
        if not math.isfinite(self.resume_at):
            raise ValueError("resume_at must be finite")
        if self.defer_kind not in ("wait", "retry"):
            raise ValueError(f"invalid defer kind: {self.defer_kind!r}")


@dataclass(frozen=True)
class DispatchSettle:
    """The durable settlement of one prepared dispatch.

    ``outcome`` is ``"release"`` (give the claim back, no cursor/outbox
    movement), ``"delay"`` (release the claim and record the delay trace —
    ordinary delay and active hold release claims without cursor/session
    change), ``"defer"`` (release the claim, detach the attached commits,
    and record the durable agent barrier — a wait defer additionally
    increments the wait streak, a retry defer does not), or ``"finish"``
    (TERMINAL: cursor advance to the dispatch's stored through boundary,
    outbox batch, durable hold/idle materialization, barrier/streak clear,
    and the completed dispatch trace — the ONLY path that moves the cursor
    or creates outbox rows). ``end_reason`` is required for ``finish``;
    ``resume_at``/``defer_kind`` are required for ``defer``.
    """

    chat_key: ChatKey
    dispatch_id: DispatchId
    cycle_id: CycleId
    outcome: str  # "release" | "delay" | "defer" | "finish"
    end_reason: str | None = None
    hold_until: float | None = None
    idle_streak_after: int = 0
    resume_at: float | None = None
    defer_kind: str | None = None  # "wait" | "retry" (defer only)
    trace_json: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    evaluated_ts: float | None = None
    snapshot_json: str | None = None
    # Controls staged by the agent are committed with the terminal dispatch.
    # Keeping them on the settlement closes the crash gap between cursor
    # settlement and a separate post-settlement control write.
    chat_controls: tuple[ChatControl, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome not in ("release", "delay", "defer", "finish"):
            raise ValueError(f"invalid dispatch outcome: {self.outcome!r}")
        if self.outcome == "finish" and self.end_reason is None:
            raise ValueError("terminal finish requires end_reason")
        if self.outcome == "defer":
            if self.resume_at is None:
                raise ValueError("defer requires resume_at")
            if not math.isfinite(self.resume_at):
                raise ValueError("resume_at must be finite")
            if self.defer_kind not in ("wait", "retry"):
                raise ValueError(f"invalid defer kind: {self.defer_kind!r}")
        if self.evaluated_ts is not None and not math.isfinite(self.evaluated_ts):
            raise ValueError("evaluated_ts must be finite")


@dataclass(frozen=True)
class OutboxItem:
    """One durable outbound delivery — exactly ONE adapter send per row.

    Split output is an ordered batch of rows sharing ``group_id`` and
    ordered by ``seq``; there is no multi-part shape. ``idem_key`` is
    REQUIRED and unique — it is what makes re-enqueue idempotent.

    State model (frozen decision #3, at-most-once): ``pending`` →
    ``in_flight`` is a durable compare-and-swap performed before the adapter
    is invoked; ``in_flight`` is NEVER automatically retried after a crash
    (the send outcome is ambiguous) — a later self echo may reconcile it via
    ``platform_msg_id``. Terminal states: ``sent`` (with ``sent_ts`` and the
    platform message id when the platform returns one) or ``dropped``.
    """

    chat_key: ChatKey
    text: str
    idem_key: str
    segments: tuple[Segment, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)
    reply_to: MessageId | None = None
    group_id: str | None = None
    seq: int | None = None
    state: str = "pending"
    send_after_ts: float | None = None
    attempt_started_ts: float | None = None
    sent_ts: float | None = None
    platform_msg_id: MessageId | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        if self.state not in ("pending", "in_flight", "sent", "dropped"):
            raise ValueError(f"invalid outbox state: {self.state!r}")


@dataclass(frozen=True)
class Record:
    """One learner record, mirroring the ``records`` table. All five
    learners share this shape; only ``payload`` differs.

    ``content_hash`` is the deterministic content hash of ``payload``
    (computed by the repository — the adaptive identity of a record).
    ``source_first_msg_id``/``source_last_msg_id`` are the fixed local-row
    source range the record was produced from (both or neither);
    ``retired`` marks a record explicitly retired from the adaptive
    surface. Legacy rows (written through the legacy ``add_record`` path)
    keep NULL content_hash/source fields and stay untrusted: the adaptive
    surface excludes them.
    """

    learner: str
    payload: dict[str, Any]
    chat_key: ChatKey | None = None
    weight: float = 1.0
    uses: int = 0
    created_ts: float | None = None
    id: int | None = None
    content_hash: str | None = None
    source_first_msg_id: MessageRowId | None = None
    source_last_msg_id: MessageRowId | None = None
    retired: bool = False
    # The durable learner run that last produced this adaptive identity.  It
    # is deliberately carried with the record instead of being reconstructed
    # from an overlapping source range: ranges are not unique after a merge.
    producing_run_id: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("record weight must be finite and nonnegative")
        if (self.source_first_msg_id is None) != (self.source_last_msg_id is None):
            raise ValueError("source range must have both bounds or neither")
        if (
            self.source_first_msg_id is not None
            and self.source_last_msg_id is not None
            and self.source_first_msg_id > self.source_last_msg_id
        ):
            raise ValueError("source_first_msg_id must be <= source_last_msg_id")
        if not isinstance(self.retired, bool):
            raise ValueError("retired must be a bool")


# ── Adaptive foundation (Phase 6) ───────────────────────────────────────────
# Immutable behavior-light boundary types for the adaptive learner lanes
# (frozen Oracle advisory): the runtime mode, settlement notices, the
# per-run adaptive context, and the declarative learner spec/batch/draft/
# grant. Specs are PURE DATA — they carry NO repository, LLM, or clock
# access (no methods that touch them), so a learner can never reach outside
# its declared inputs. Validation is limited to finiteness, identity
# consistency, and the closed outcome/policy/state sets — no operational
# behavior.

class RuntimeMode:
    """Stable machine-readable runtime modes (frozen spec). Consumers match
    on these exact strings, never on free-form text.

    - ``live``: the normal connected runtime.
    - ``dry_run``: the console/dry-run runtime — no live adapter.
    - ``replay``: the replay lane re-scores past traces.
    - ``doctor``: the diagnostic lane.
    """

    LIVE = "live"
    DRY_RUN = "dry_run"
    REPLAY = "replay"
    DOCTOR = "doctor"


_RUNTIME_MODES = frozenset(
    {
        RuntimeMode.LIVE,
        RuntimeMode.DRY_RUN,
        RuntimeMode.REPLAY,
        RuntimeMode.DOCTOR,
    }
)


@dataclass(frozen=True)
class SettlementNotice:
    """One durable learner settlement notice: the outcome of a learner run
    and what it changed.

    ``outcome`` is ``success`` | ``malformed`` | ``cancelled``. A
    successful run may add zero records (a valid EMPTY result still
    advances the watermark); a malformed/cancelled run never advances the
    watermark or inserts records.
    """

    learner: str
    chat_key: ChatKey
    run_id: int
    outcome: str
    records_added: int = 0
    records_merged: int = 0
    watermark: MessageRowId | None = None
    error: str | None = None
    ts: float | None = None

    def __post_init__(self) -> None:
        if self.outcome not in ("success", "malformed", "cancelled"):
            raise ValueError(f"invalid settlement outcome: {self.outcome!r}")
        if (
            isinstance(self.run_id, bool)
            or not isinstance(self.run_id, int)
            or self.run_id < 0
        ):
            raise ValueError("run_id must be a nonnegative integer")
        if self.records_added < 0 or self.records_merged < 0:
            raise ValueError("record counts must be nonnegative")
        if self.ts is not None and not math.isfinite(self.ts):
            raise ValueError("ts must be finite")


@dataclass(frozen=True)
class AgentAdaptiveContext:
    """The immutable per-run adaptive context handed to a learner: the
    runtime mode, the chat/learner identity, the run's fixed source
    boundary, and the durable watermark snapshot. Pure data — no
    repository/LLM/clock access."""

    chat_key: ChatKey
    learner: str
    mode: str
    run_id: int | None = None
    start_msg_id: MessageRowId | None = None
    through_msg_id: MessageRowId | None = None
    watermark: MessageRowId | None = None
    now: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in _RUNTIME_MODES:
            raise ValueError(f"invalid runtime mode: {self.mode!r}")
        if self.run_id is not None and (
            isinstance(self.run_id, bool)
            or not isinstance(self.run_id, int)
            or self.run_id < 0
        ):
            raise ValueError("run_id must be a nonnegative integer")
        if self.now is not None and not math.isfinite(self.now):
            raise ValueError("now must be finite")


@dataclass(frozen=True)
class LearnerSpec:
    """One declarative learner definition (Phase 6). Pure data: a spec
    carries NO repository, LLM, or clock access — the runtime wires those
    in. ``policy`` is ``nonself`` (the default: source reads exclude the
    bot's own messages) or ``all``."""

    name: str
    prompt: str
    cadence_s: int
    policy: str = "nonself"
    batch_size: int = 1
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("learner name must be non-empty")
        if not self.prompt:
            raise ValueError("learner prompt must be non-empty")
        if (
            isinstance(self.cadence_s, bool)
            or not isinstance(self.cadence_s, int)
            or self.cadence_s <= 0
        ):
            raise ValueError("cadence_s must be a positive integer")
        if self.policy not in ("nonself", "all"):
            raise ValueError("policy must be 'nonself' or 'all'")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")


@dataclass(frozen=True)
class AdapterSpec:
    """Declarative adapter metadata exposed to trusted plugins.

    This is intentionally not an adapter factory or client.  Runtime adapter
    instances remain owned by ``App`` and are never reachable through the
    plugin API.
    """

    name: str
    capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("adapter name must be non-empty")
        if not isinstance(self.capabilities, frozenset) or not all(
            isinstance(value, str) and value.strip() for value in self.capabilities
        ):
            raise ValueError("adapter capabilities must be a frozenset of strings")


@dataclass(frozen=True)
class LearnerBatch:
    """A fixed source batch for one learner: the local-row range
    ``(first_msg_id .. last_msg_id]`` beyond the learner's durable
    watermark, capped to a recent tail.

    ``texts`` are the source message texts in row order (``is_self``
    excluded for a ``nonself`` policy), with ``senders``/``sender_names``
    positionally aligned to them; ``source_hash`` is the
    deterministic hash of those texts (computed by the repository, so the
    learner never invents one); ``observed_watermark`` is the durable
    watermark observed when the batch was read — the exact snapshot the CAS
    commit fences on.
    """

    chat_key: ChatKey
    learner: str
    first_msg_id: MessageRowId
    last_msg_id: MessageRowId
    source_hash: str
    texts: tuple[str, ...] = ()
    observed_watermark: MessageRowId | None = None
    # The read policy and exact row identity are part of the source CAS.  The
    # defaults keep hand-built batches from older callers source-compatible.
    policy: str = "nonself"
    source_ids: tuple[MessageRowId, ...] = ()
    # Who wrote each source text, positionally aligned with ``texts``. Only
    # the impression learner reads these; ``source_hash`` is computed from
    # ``texts`` ALONE, so adding them leaves every existing watermark and
    # CAS fence byte-identical. Empty for hand-built batches.
    senders: tuple[SenderId, ...] = ()
    sender_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.first_msg_id > self.last_msg_id:
            raise ValueError("first_msg_id must be <= last_msg_id")
        if not self.source_hash:
            raise ValueError("source_hash must be non-empty")
        if self.policy not in ("nonself", "all"):
            raise ValueError("policy must be 'nonself' or 'all'")
        if self.source_ids and len(self.source_ids) != len(self.texts):
            raise ValueError("source_ids must match texts")
        if self.senders and len(self.senders) != len(self.texts):
            raise ValueError("senders must match texts")
        if self.sender_names and len(self.sender_names) != len(self.texts):
            raise ValueError("sender_names must match texts")


@dataclass(frozen=True)
class LearnerDraft:
    """The CAS commit of one learner source range: the fixed source batch,
    the records the learner produced from it, and the durable watermark the
    caller expects (``expected_through_msg_id``).

    ``outcome`` is ``success`` | ``malformed`` | ``cancelled``: a
    successful commit atomically inserts/merges the validated records +
    their sources + the successful run + the watermark advance (a valid
    EMPTY result still advances); a malformed/cancelled commit settles the
    run WITHOUT advancing the watermark or inserting records.
    """

    chat_key: ChatKey
    learner: str
    batch: LearnerBatch
    records: tuple[Record, ...] = ()
    expected_through_msg_id: MessageRowId | None = None
    outcome: str = "success"
    error: str | None = None
    # New callers fence the exact prepared run.  None is retained for the
    # legacy direct repository surface, which used the sole prepared run.
    run_id: int | None = None
    cadence_s: float | None = None

    def __post_init__(self) -> None:
        if self.batch.chat_key != self.chat_key:
            raise ValueError("batch chat_key must match request chat_key")
        if self.batch.learner != self.learner:
            raise ValueError("batch learner must match request learner")
        for rec in self.records:
            if rec.chat_key != self.chat_key:
                raise ValueError("record chat_key must match request chat_key")
            if rec.learner != self.learner:
                raise ValueError("record learner must match request learner")
        if self.outcome not in ("success", "malformed", "cancelled"):
            raise ValueError("outcome must be 'success', 'malformed' or 'cancelled'")
        if self.run_id is not None and (
            isinstance(self.run_id, bool) or not isinstance(self.run_id, int)
            or self.run_id < 0
        ):
            raise ValueError("run_id must be a nonnegative integer")
        if self.cadence_s is not None and (
            not math.isfinite(self.cadence_s) or self.cadence_s <= 0
        ):
            raise ValueError("cadence_s must be positive and finite")


@dataclass(frozen=True)
class LearnerRunRequest:
    """What the scheduler passes to ``acquire_learner_run``: the chat and
    learner, the mandatory finite lease, and the absolute-epoch evaluation
    time used for expiry/recovery decisions."""

    chat_key: ChatKey
    learner: str
    started_ts: float
    expires_at: float
    now: float

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.started_ts)
            and math.isfinite(self.expires_at)
            and math.isfinite(self.now)
        ):
            raise ValueError("learner run timestamps must be finite")
        if self.expires_at <= self.started_ts:
            raise ValueError(
                "learner run lease must be finite: expires_at must be > started_ts"
            )


@dataclass(frozen=True)
class LearnerGrant:
    """The durable grant of one ``acquire_learner_run`` call: the new
    ``learner_runs`` row id and the fixed local-row boundary captured at
    claim time (``start_msg_id`` is the chat cursor, ``through_msg_id`` the
    chat's max message row id)."""

    chat_key: ChatKey
    learner: str
    run_id: int
    started_ts: float
    expires_at: float
    start_msg_id: MessageRowId
    through_msg_id: MessageRowId

    def __post_init__(self) -> None:
        if (
            isinstance(self.run_id, bool)
            or not isinstance(self.run_id, int)
            or self.run_id < 0
        ):
            raise ValueError("run_id must be a nonnegative integer")
        if not (math.isfinite(self.started_ts) and math.isfinite(self.expires_at)):
            raise ValueError("grant timestamps must be finite")
        if self.expires_at <= self.started_ts:
            raise ValueError("grant lease must be finite: expires_at must be > started_ts")


@dataclass(frozen=True)
class LearnerBusy:
    """The typed ``already owned`` outcome of one ``acquire_learner_run``
    call: the chat+learner already has a LIVE, unexpired prepared run.
    ``busy_until`` is the exact absolute-epoch expiry of the active owner's
    lease; an expired run is never reported busy — it is recovered and the
    caller receives a ``LearnerGrant`` instead."""

    chat_key: ChatKey
    learner: str
    run_id: int
    busy_until: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.busy_until):
            raise ValueError("busy_until must be finite")


@dataclass(frozen=True)
class LearnerState:
    """The durable per-(chat, learner) state, mirroring the
    ``learner_state`` table: the summarized watermark, the observed
    watermark snapshot, and the last settled run id."""

    chat_key: ChatKey
    learner: str
    watermark_msg_id: MessageRowId | None = None
    observed_watermark_msg_id: MessageRowId | None = None
    last_run_id: int | None = None
    updated_ts: float | None = None
    cadence_s: float | None = None
    next_due_ts: float | None = None
    failure_streak: int = 0

    def __post_init__(self) -> None:
        if self.updated_ts is not None and not math.isfinite(self.updated_ts):
            raise ValueError("updated_ts must be finite")
        if self.cadence_s is not None and (
            not math.isfinite(self.cadence_s) or self.cadence_s <= 0
        ):
            raise ValueError("cadence_s must be positive and finite")
        if self.next_due_ts is not None and not math.isfinite(self.next_due_ts):
            raise ValueError("next_due_ts must be finite")
        if (
            isinstance(self.failure_streak, bool)
            or not isinstance(self.failure_streak, int)
            or self.failure_streak < 0
        ):
            raise ValueError("failure_streak must be a nonnegative integer")

    @property
    def failure_count(self) -> int:
        """Compatibility name for callers that call the streak a count."""
        return self.failure_streak

    @property
    def next_due(self) -> float | None:
        return self.next_due_ts


@dataclass(frozen=True)
class LearnerRun:
    """One durable learner run row, mirroring the ``learner_runs`` table."""

    id: int
    chat_key: ChatKey
    learner: str
    started_ts: float
    expires_at: float
    start_msg_id: MessageRowId
    through_msg_id: MessageRowId
    state: str = "prepared"
    source_hash: str | None = None
    records_added: int = 0
    records_merged: int = 0
    error: str | None = None
    settled_ts: float | None = None

    def __post_init__(self) -> None:
        if self.state not in (
            "prepared", "success", "malformed", "cancelled", "expired", "released",
        ):
            raise ValueError(f"invalid learner run state: {self.state!r}")
        if not (math.isfinite(self.started_ts) and math.isfinite(self.expires_at)):
            raise ValueError("run timestamps must be finite")
        if self.settled_ts is not None and not math.isfinite(self.settled_ts):
            raise ValueError("settled_ts must be finite")


@dataclass(frozen=True)
class RecordHit:
    """One bounded record FTS hit for a chat+learner (lexical-first recall
    over the canonical record token documents)."""

    chat_key: ChatKey
    learner: str
    record_id: int
    text: str
    score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")


# ── Knowledge boundary (Phase 5 foundation) ─────────────────────────────────
# Immutable behavior-light boundary types for the durable knowledge lanes
# (frozen Oracle advisory ses_fc73526f6ffeYTfNXN1aQH9sRn): memory records,
# source batches, and write requests; person profiles; vector rows; embedding
# generations; and lexical hits. Validation is limited to finiteness,
# dimension/blob shape, and cross-chat consistency — no operational behavior.
# Knowledge rows/FTS are authoritative local state; vectors are rebuildable
# derived state. Everything is strictly chat-scoped.

@dataclass(frozen=True)
class MemoryRecord:
    """One durable memory row, mirroring the ``memories`` table.

    ``source_first_msg_id``/``source_last_msg_id`` are the fixed local-row
    source range the memory was summarized from (both or neither);
    ``source_hash`` is the deterministic hash of that source content. A
    source range is unique per chat — a later summarizer CAS-commits each
    range exactly once.
    """

    chat_key: ChatKey
    text: str
    kind: str = "memory"
    cues: tuple[str, ...] = ()
    strength: float = 1.0
    created_ts: float | None = None
    last_hit_ts: float | None = None
    source_first_msg_id: MessageRowId | None = None
    source_last_msg_id: MessageRowId | None = None
    source_hash: str | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError("memory strength must be finite and nonnegative")
        for ts in (self.created_ts, self.last_hit_ts):
            if ts is not None and not math.isfinite(ts):
                raise ValueError("memory timestamps must be finite")
        if (self.source_first_msg_id is None) != (self.source_last_msg_id is None):
            raise ValueError("source range must have both bounds or neither")
        if (
            self.source_first_msg_id is not None
            and self.source_last_msg_id is not None
            and self.source_first_msg_id > self.source_last_msg_id
        ):
            raise ValueError("source_first_msg_id must be <= source_last_msg_id")


@dataclass(frozen=True)
class MemorySourceBatch:
    """A fixed source batch: the local-row range ``(first_msg_id ..
    last_msg_id]`` beyond the durable memory watermark, capped to a recent
    tail. ``texts`` are the source message texts in row order;
    ``source_hash`` is the deterministic hash of those texts (computed by
    the repository, so the summarizer never invents one).
    ``observed_watermark`` is the durable memory watermark observed when the
    batch was read — the exact snapshot the CAS commit fences on, so a
    sequential summarizer never races itself.
    """

    chat_key: ChatKey
    first_msg_id: MessageRowId
    last_msg_id: MessageRowId
    source_hash: str
    texts: tuple[str, ...] = ()
    observed_watermark: MessageRowId | None = None

    def __post_init__(self) -> None:
        if self.first_msg_id > self.last_msg_id:
            raise ValueError("first_msg_id must be <= last_msg_id")
        if not self.source_hash:
            raise ValueError("source_hash must be non-empty")


@dataclass(frozen=True)
class MemoryWriteRequest:
    """The CAS commit of one memory source range: the fixed source batch,
    the memory records the summarizer produced from it, and the durable
    watermark the caller expects (``expected_through_msg_id``). The commit
    succeeds only when the stored watermark still equals the expected
    value — a stale CAS loser changes nothing.
    """

    chat_key: ChatKey
    batch: MemorySourceBatch
    records: tuple[MemoryRecord, ...] = ()
    expected_through_msg_id: MessageRowId | None = None

    def __post_init__(self) -> None:
        if self.batch.chat_key != self.chat_key:
            raise ValueError("batch chat_key must match request chat_key")
        for rec in self.records:
            if rec.chat_key != self.chat_key:
                raise ValueError("memory record chat_key must match request chat_key")


@dataclass(frozen=True)
class PersonProfile:
    """One per-chat person identity, mirroring the ``persons`` table.

    Uniqueness is ``(chat_key, platform_uid)`` — there is no global
    nickname matching. ``profile_through_msg_id`` is the durable profile
    cursor, written only by ``cas_person_profile``.
    """

    chat_key: ChatKey
    platform_uid: SenderId
    names: tuple[str, ...] = ()
    profile: str | None = None
    impression: str | None = None
    updated_ts: float | None = None
    profile_through_msg_id: MessageRowId | None = None
    person_key: PersonKey | None = None

    def __post_init__(self) -> None:
        if self.updated_ts is not None and not math.isfinite(self.updated_ts):
            raise ValueError("updated_ts must be finite")


@dataclass(frozen=True)
class EmbeddingGeneration:
    """One embedding generation, mirroring the ``embedding_generations``
    table.

    ``space_id`` is the canonical embedding space identity (derived from the
    model + explicit revision) and is unique per generation; ``revision`` is
    the explicit model revision from the embed profile. At most one
    generation is ``active``; ``building`` marks a generation still being
    populated. ``vector_revision`` is the durable vector mutation revision,
    bumped on every vector write so a direct repo mutation is visible on the
    next search. Vectors are written per generation so old and new
    generations coexist during a model/dimension change.
    """

    id: int | None = None
    space_id: str = ""
    model: str = ""
    revision: str = ""
    dim: int = 0
    state: str = "inactive"
    created_ts: float | None = None
    vector_revision: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.dim, bool)
            or not isinstance(self.dim, int)
            or self.dim <= 0
        ):
            raise ValueError("dim must be a positive integer")
        if self.state not in ("building", "active", "inactive"):
            raise ValueError("state must be 'building', 'active' or 'inactive'")
        if self.created_ts is not None and not math.isfinite(self.created_ts):
            raise ValueError("created_ts must be finite")
        if (
            isinstance(self.vector_revision, bool)
            or not isinstance(self.vector_revision, int)
            or self.vector_revision < 0
        ):
            raise ValueError("vector_revision must be a nonnegative integer")


@dataclass(frozen=True)
class VectorRow:
    """One chat-scoped vector row, mirroring the ``vectors`` table.

    The owner is a local row in ``owner_table`` (Phase 5: ``memories``);
    chat scope is enforced by the repository against the owner. ``dim``
    must be positive and the float32 ``blob`` must be exactly ``dim * 4``
    bytes with no NaN/inf values. ``generation`` is the embedding
    generation the vector belongs to.
    """

    owner_table: str
    owner_id: int
    dim: int
    model: str
    generation: int
    blob: bytes
    source_hash: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.dim, bool)
            or not isinstance(self.dim, int)
            or self.dim <= 0
        ):
            raise ValueError("dim must be a positive integer")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("generation must be a nonnegative integer")
        if not isinstance(self.blob, (bytes, bytearray)):
            raise ValueError("blob must be bytes")
        if len(self.blob) != self.dim * 4:
            raise ValueError("blob length must equal dim * 4 (float32)")
        for i in range(self.dim):
            value = struct.unpack_from("<f", self.blob, i * 4)[0]
            if not math.isfinite(value):
                raise ValueError("vector values must be finite (no NaN/inf)")


@dataclass(frozen=True)
class LexicalHit:
    """One bounded memory FTS hit for a chat (lexical-first recall)."""

    chat_key: ChatKey
    memory_id: int
    text: str
    score: float
    source_first_msg_id: MessageRowId | None = None
    source_last_msg_id: MessageRowId | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")


# ── Media catalog boundary (Phase 6 P6.5 foundation) ─────────────────────────
# Immutable behavior-light boundary types for the durable chat-scoped media
# catalog (frozen Oracle advisory): the pending candidate and the full asset
# row. The catalog key is OPAQUE — validation (in ``emoji.py``, enforced by
# the repository at submit time) rejects local paths, URLs, data/base64
# payloads, and raw platform media references as catalog keys. Existing
# global ``emoji`` rows remain legacy/untrusted and are never read by the
# catalog.

class MediaKind:
    """Stable machine-readable media catalog kinds (frozen spec). Consumers
    match on these exact strings, never on free-form text."""

    STICKER = "sticker"
    IMAGE = "image"


_MEDIA_KINDS = frozenset({MediaKind.STICKER, MediaKind.IMAGE})


class MediaSafetyStatus:
    """Stable machine-readable media asset safety statuses (frozen spec).

    - ``pending``: a submitted candidate awaiting approval.
    - ``approved``: safety-approved and selectable.
    - ``rejected``: a pending candidate rejected at approval time (terminal).
    - ``revoked``: an approved asset revoked after approval (terminal).
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVOKED = "revoked"


_MEDIA_SAFETY_STATUSES = frozenset(
    {
        MediaSafetyStatus.PENDING,
        MediaSafetyStatus.APPROVED,
        MediaSafetyStatus.REJECTED,
        MediaSafetyStatus.REVOKED,
    }
)


@dataclass(frozen=True)
class MediaAssetCandidate:
    """One chat-scoped pending candidate for the durable media catalog.

    ``cache_key`` is the OPAQUE content-addressed cache key (the sha256 hex
    digest of the normalized bytes, as produced by ``media.MediaStore``) —
    never a local path, URL, data/base64 payload, or raw platform media
    reference (the repository rejects those at submit time). ``sha256`` is
    the content sha256 of the original bytes. ``kind`` is ``sticker`` |
    ``image``. ``source_message_id``/``source_sender_id``/
    ``source_sender_name`` carry the source message+sender provenance.
    """

    chat_key: ChatKey
    kind: str
    cache_key: str
    sha256: str
    mime: str
    width: int | None = None
    height: int | None = None
    description: str | None = None
    source_message_id: MessageRowId | None = None
    source_sender_id: SenderId | None = None
    source_sender_name: str | None = None
    source_ts: float | None = None
    created_ts: float | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in _MEDIA_KINDS:
            raise ValueError(f"invalid media kind: {self.kind!r}")
        if not self.cache_key:
            raise ValueError("cache_key must be non-empty")
        if not self.sha256:
            raise ValueError("sha256 must be non-empty")
        if not self.mime:
            raise ValueError("mime must be non-empty")
        for name, value in (("width", self.width), ("height", self.height)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("source_ts", self.source_ts), ("created_ts", self.created_ts)):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.id is not None and (
            isinstance(self.id, bool) or not isinstance(self.id, int) or self.id < 0
        ):
            raise ValueError("id must be a nonnegative integer")


@dataclass(frozen=True)
class MediaAsset:
    """One durable ``media_assets`` row — the catalog's asset view.

    ``safety_status`` is ``pending`` | ``approved`` | ``rejected`` |
    ``revoked``; ``safety_version`` bumps on every approval/revocation
    transition; ``uses``/``last_used_ts`` are the usage/cooldown facts
    selection reads.
    """

    id: int
    chat_key: ChatKey
    kind: str
    cache_key: str
    sha256: str
    mime: str
    width: int | None = None
    height: int | None = None
    description: str | None = None
    source_message_id: MessageRowId | None = None
    source_sender_id: SenderId | None = None
    source_sender_name: str | None = None
    safety_status: str = MediaSafetyStatus.PENDING
    safety_version: int = 0
    approved_ts: float | None = None
    revoked_ts: float | None = None
    uses: int = 0
    last_used_ts: float | None = None
    created_ts: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in _MEDIA_KINDS:
            raise ValueError(f"invalid media kind: {self.kind!r}")
        if self.safety_status not in _MEDIA_SAFETY_STATUSES:
            raise ValueError(f"invalid media safety status: {self.safety_status!r}")
        if (
            isinstance(self.id, bool)
            or not isinstance(self.id, int)
            or self.id < 0
        ):
            raise ValueError("id must be a nonnegative integer")
        if (
            isinstance(self.safety_version, bool)
            or not isinstance(self.safety_version, int)
            or self.safety_version < 0
        ):
            raise ValueError("safety_version must be a nonnegative integer")
        if (
            isinstance(self.uses, bool)
            or not isinstance(self.uses, int)
            or self.uses < 0
        ):
            raise ValueError("uses must be a nonnegative integer")
        for name, value in (("width", self.width), ("height", self.height)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("approved_ts", self.approved_ts),
            ("revoked_ts", self.revoked_ts),
            ("last_used_ts", self.last_used_ts),
            ("created_ts", self.created_ts),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


# ── Chat controls (Phase 6 P6.6b) ───────────────────────────────────────────

class ChatControlKind:
    """Stable machine-readable chat-control kinds (frozen spec). Consumers
    match on these exact strings, never on free-form text.

    - ``focus``: a bounded focus window on the target chat (one focus per
      account).
    - ``notify``: a bounded INTERNAL focus event carrying a payload — the
      target chat's gate evaluates as focused and the payload is shown to
      the agent there. Delivery still traverses the target chat's normal
      gate/cycle/outbox flow; there is NO bypass platform send.
    """

    FOCUS = "focus"
    NOTIFY = "notify"


_CHAT_CONTROL_KINDS = frozenset({ChatControlKind.FOCUS, ChatControlKind.NOTIFY})


@dataclass(frozen=True)
class ChatControlIntent:
    """One STAGED chat control (the typed terminal intent the chat-control
    tools produce).

    ``kind`` is ``focus`` | ``notify``; ``target_chat_key`` is the target
    chat (a KNOWN chat on the SAME account as the issuing chat);
    ``ttl_s`` is the bounded TTL (focus 30..3600, notify 1..3600);
    ``text`` is the bounded notify payload (None for focus). Nothing is
    written at staging time — the CycleRunner applies the intent
    idempotently in the normal LIVE terminal flow after settle/outbox/
    marker, using the dispatch id + intent sequence as the idempotency
    identity.
    """

    kind: str
    target_chat_key: ChatKey
    ttl_s: int = 0
    text: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _CHAT_CONTROL_KINDS:
            raise ValueError(f"invalid chat control kind: {self.kind!r}")
        if isinstance(self.ttl_s, bool) or not isinstance(self.ttl_s, int):
            raise ValueError("ttl_s must be an integer")
        if self.ttl_s <= 0:
            raise ValueError("ttl_s must be a positive integer")
        if self.kind == ChatControlKind.FOCUS and not 30 <= self.ttl_s <= 3600:
            raise ValueError("focus ttl_s must be in [30, 3600]")
        if self.kind == ChatControlKind.NOTIFY and self.ttl_s > 3600:
            raise ValueError("notify ttl_s must be in [1, 3600]")
        if self.kind == ChatControlKind.FOCUS and self.text is not None:
            raise ValueError("focus text must be None")
        if self.kind == ChatControlKind.NOTIFY and (
            not isinstance(self.text, str) or not self.text.strip()
            or len(self.text) > 500
        ):
            raise ValueError("notify text must be a non-empty string of <= 500 chars")


@dataclass(frozen=True)
class ChatControl:
    """One durable internal chat control row (Phase 6 P6.6b).

    ``kind`` is ``focus`` | ``notify``; ``ttl_until`` is the absolute-epoch
    expiry (bounded: focus 30..3600s, notify 1..3600s from creation);
    ``dispatch_id`` + ``intent_seq`` are the idempotency identity (a
    retried settlement of the same dispatch never double-applies);
    ``source_chat_key`` is the chat that issued the control; the target
    ``chat_key`` must be a known chat on the SAME account (platform +
    self_id) as the source. ``text`` is the bounded notify payload (None
    for focus).
    """

    chat_key: ChatKey
    kind: str
    ttl_until: float
    created_ts: float
    dispatch_id: int
    intent_seq: int
    source_chat_key: ChatKey
    text: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _CHAT_CONTROL_KINDS:
            raise ValueError(f"invalid chat control kind: {self.kind!r}")
        if not (math.isfinite(self.ttl_until) and math.isfinite(self.created_ts)):
            raise ValueError("chat control timestamps must be finite")
        if self.ttl_until <= self.created_ts:
            raise ValueError("ttl_until must be after created_ts")
        ttl = self.ttl_until - self.created_ts
        minimum = 30.0 if self.kind == ChatControlKind.FOCUS else 1.0
        if ttl < minimum or ttl > 3600.0:
            raise ValueError("chat control TTL is outside its bounded range")
        if (
            isinstance(self.dispatch_id, bool)
            or not isinstance(self.dispatch_id, int)
            or self.dispatch_id < 0
        ):
            raise ValueError("dispatch_id must be a nonnegative integer")
        if (
            isinstance(self.intent_seq, bool)
            or not isinstance(self.intent_seq, int)
            or self.intent_seq < 0
        ):
            raise ValueError("intent_seq must be a nonnegative integer")
        if self.kind == ChatControlKind.FOCUS and self.text is not None:
            raise ValueError("focus text must be None")
        if self.kind == ChatControlKind.NOTIFY and (
            not isinstance(self.text, str) or not self.text.strip()
            or len(self.text) > 500
        ):
            raise ValueError("notify text must be a non-empty string of <= 500 chars")
