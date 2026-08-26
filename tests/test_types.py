"""Storage-contract types: defaults, immutability, the platform-vs-local
message id distinction, the single-send OutboxItem shape, and the Phase 2
gate boundary types (snapshots, trace facts, event-only delay)."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from pretender.types import (
    AgentAdaptiveContext,
    BackoffFacts,
    ChatIdentity,
    ChatKey,
    ChatState,
    ClaimBusy,
    Contribution,
    CycleClaim,
    CycleFinish,
    CycleId,
    Decision,
    DecisionTrace,
    EchoStatus,
    GateSnapshot,
    IngestResult,
    LearnerBatch,
    LearnerBusy,
    LearnerDraft,
    LearnerGrant,
    LearnerRun,
    LearnerRunRequest,
    LearnerSpec,
    LearnerState,
    MediaAsset,
    MediaAssetCandidate,
    MediaKind,
    MediaSafetyStatus,
    Message,
    MessageId,
    MessageRowId,
    OutboxItem,
    Outgoing,
    PlatformId,
    Reason,
    RecentSnapshot,
    Record,
    RecordHit,
    RuntimeMode,
    Segment,
    SelfId,
    SenderId,
    SettlementNotice,
    ToolCall,
    ToolCallId,
    TranscriptMessage,
)

CK = ChatKey("qq:group:123456")
SENDER = SenderId("u1")


def _msg(**kw: Any) -> Message:
    base: dict[str, Any] = dict(
        chat_key=CK,
        sender_id=SENDER,
        sender_name="alice",
        is_self=False,
        text="hello",
    )
    base.update(kw)
    return Message(**base)


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
            _msg(row_id=MessageRowId(2)),
            _msg(row_id=MessageRowId(3)),
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


# ── Defaults ────────────────────────────────────────────────────────────────

def test_chat_state_defaults():
    st = ChatState(chat_key=CK)
    assert st.cursor_msg_id is None
    assert st.focus_until is None
    assert st.hold_until is None
    assert st.avg_interval is None
    assert st.idle_streak == 0
    assert st.cfg_json is None


def test_cycle_claim_lease_is_mandatory_and_finite():
    claim = CycleClaim(chat_key=CK, cycle_id=CycleId("c1"), started_ts=100.0,
                       expires_at=500.0)
    assert claim.expires_at == 500.0
    with pytest.raises(TypeError):
        CycleClaim(chat_key=CK, cycle_id=CycleId("c1"), started_ts=100.0)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="finite"):
        CycleClaim(chat_key=CK, cycle_id=CycleId("c1"), started_ts=100.0,
                   expires_at=100.0)


def test_cycle_finish_defaults():
    fin = CycleFinish(chat_key=CK, cycle_id=CycleId("c1"), end_reason="no_action")
    # The finish carries NO cursor value: the cursor derives from the
    # stored claim boundary.
    assert not hasattr(fin, "cursor_msg_id")
    assert fin.hold_until is None
    assert fin.idle_streak_after == 0
    assert fin.trace_json is None
    assert fin.tokens_in == 0
    assert fin.tokens_out == 0


def test_cycle_finish_carries_idle_streak_after():
    # Idle backoff is materialized transactionally at terminal completion:
    # the finish carries the durable streak AFTER the cycle.
    fin = CycleFinish(
        chat_key=CK, cycle_id=CycleId("c1"), end_reason="planner_no_tool_end",
        hold_until=1_700_000_600.0, idle_streak_after=3,
    )
    assert fin.hold_until == 1_700_000_600.0
    assert fin.idle_streak_after == 3


def test_outbox_item_defaults():
    item = OutboxItem(chat_key=CK, text="hi", idem_key="k1")
    assert item.state == "pending"
    assert item.segments == ()
    assert item.payload == {}
    assert item.reply_to is None
    assert item.group_id is None
    assert item.seq is None
    assert item.send_after_ts is None
    assert item.attempt_started_ts is None
    assert item.sent_ts is None
    assert item.platform_msg_id is None
    assert item.id is None


def test_message_defaults():
    msg = _msg()
    assert msg.id is None
    assert msg.row_id is None


# ── Immutability ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "make,field",
    [
        (lambda: ChatState(chat_key=CK), "cursor_msg_id"),
        (lambda: CycleClaim(chat_key=CK, cycle_id=CycleId("c1"), started_ts=1.0,
                            expires_at=2.0), "started_ts"),
        (lambda: CycleFinish(chat_key=CK, cycle_id=CycleId("c1"), end_reason="x"), "end_reason"),
        (lambda: OutboxItem(chat_key=CK, text="hi", idem_key="k1"), "text"),
        (lambda: _msg(), "text"),
        (lambda: _snapshot(), "mode"),
        (lambda: RecentSnapshot(chat_key=CK), "window_count"),
        (lambda: BackoffFacts(), "applied"),
    ],
)
def test_storage_types_are_frozen(make, field):
    obj = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, field, "changed")


# ── Message: platform id vs local row id ────────────────────────────────────

def test_message_platform_id_and_row_id_are_distinct_newtypes():
    assert MessageId is not MessageRowId
    # NewType factories return their underlying type unchanged
    assert type(MessageId("123")) is str
    assert type(MessageRowId(7)) is int


def test_message_carries_both_ids():
    msg = _msg(id=MessageId("123"), row_id=MessageRowId(7))
    assert msg.id == MessageId("123")
    assert msg.row_id == MessageRowId(7)
    # platform id is a str-backed NewType, row id an int-backed one
    assert isinstance(msg.id, str)
    assert isinstance(msg.row_id, int)


def test_message_row_id_optional_and_defaults_to_none():
    msg = _msg(id=MessageId("123"))
    assert msg.row_id is None


# ── OutboxItem: one row = one adapter send ──────────────────────────────────

def test_outbox_item_has_no_multi_part_shape():
    assert not hasattr(OutboxItem, "parts")
    assert not hasattr(OutboxItem, "part")


def test_outbox_item_single_send_fields():
    item = OutboxItem(
        chat_key=CK,
        text="one send",
        idem_key="k1",
        segments=(Segment(kind="text", data={"text": "one send"}),),
        payload={"plugin": "ref"},
        reply_to=MessageId("9"),
        group_id="g1",
        seq=2,
    )
    assert item.text == "one send"
    assert item.segments[0].kind == "text"
    assert item.payload == {"plugin": "ref"}
    assert item.reply_to == MessageId("9")
    assert item.group_id == "g1"
    assert item.seq == 2


def test_outbox_item_idem_key_required():
    with pytest.raises(TypeError):
        OutboxItem(chat_key=CK, text="hi")  # type: ignore[call-arg]


def test_outbox_item_state_machine_fields():
    in_flight = OutboxItem(
        chat_key=CK, text="hi", idem_key="k1",
        state="in_flight", attempt_started_ts=50.0,
    )
    assert in_flight.state == "in_flight"
    assert in_flight.attempt_started_ts == 50.0

    sent = OutboxItem(
        chat_key=CK, text="hi", idem_key="k1",
        state="sent", sent_ts=60.0, platform_msg_id=MessageId("42"),
    )
    assert sent.state == "sent"
    assert sent.sent_ts == 60.0
    assert sent.platform_msg_id == MessageId("42")

    dropped = OutboxItem(chat_key=CK, text="hi", idem_key="k1", state="dropped")
    assert dropped.state == "dropped"


def test_outbox_item_rejects_unknown_state():
    with pytest.raises(ValueError, match="invalid outbox state"):
        OutboxItem(chat_key=CK, text="hi", idem_key="k1", state="queued")


# ── ChatIdentity stays identity-only ────────────────────────────────────────

def test_chat_identity_has_no_runtime_state():
    ident = ChatIdentity(
        chat_key=CK, platform=PlatformId("qq"), self_id=SelfId("bot"), kind="group"
    )
    assert ident.title is None
    assert not hasattr(ident, "cursor_msg_id")
    assert not hasattr(ident, "focus_until")
    assert not hasattr(ident, "idle_streak")


# ── Ingest boundary: typed IngestResult ─────────────────────────────────────

def test_ingest_result_defaults():
    result = IngestResult()
    assert result.row_id is None
    assert result.inserted is False
    assert result.echo_status == EchoStatus.NOT_APPLICABLE


def test_ingest_result_carries_row_id_inserted_and_echo_status():
    result = IngestResult(
        row_id=MessageRowId(7), inserted=True, echo_status=EchoStatus.RECONCILED
    )
    assert result.row_id == MessageRowId(7)
    assert result.inserted is True
    assert result.echo_status == "reconciled"


def test_ingest_result_rejects_unknown_echo_status():
    with pytest.raises(ValueError, match="echo status"):
        IngestResult(echo_status="maybe")  # type: ignore[arg-type]


def test_echo_status_tokens_are_stable_machine_readable():
    assert EchoStatus.NOT_APPLICABLE == "not_applicable"
    assert EchoStatus.RECONCILED == "reconciled"
    assert EchoStatus.ALREADY_RECONCILED == "already_reconciled"
    assert EchoStatus.UNPROVEN == "unproven"
    assert EchoStatus.CONFLICT == "conflict"


def test_ingest_result_is_frozen():
    result = IngestResult(row_id=MessageRowId(1), inserted=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.inserted = False  # type: ignore[misc]


def test_ingest_result_pending_count_defaults_to_none():
    # The pending count is meaningful ONLY for a newly inserted non-self
    # message: noninserted (duplicate, non-message event, unknown chat)
    # and self results carry None by default.
    assert IngestResult().pending_count is None
    assert IngestResult(row_id=MessageRowId(1), inserted=False).pending_count is None
    assert (
        IngestResult(
            row_id=MessageRowId(1), inserted=True, echo_status=EchoStatus.UNPROVEN
        ).pending_count
        is None
    )


def test_ingest_result_pending_count_carries_atomic_count():
    result = IngestResult(
        row_id=MessageRowId(3), inserted=True, pending_count=3
    )
    assert result.pending_count == 3


def test_ingest_result_rejects_negative_pending_count():
    with pytest.raises(ValueError, match="pending_count"):
        IngestResult(row_id=MessageRowId(1), inserted=True, pending_count=-1)


def test_claim_busy_carries_owner_and_exact_busy_until():
    busy = ClaimBusy(
        chat_key=CK, cycle_id=CycleId("cy-1"), busy_until=500.0
    )
    assert busy.chat_key == CK
    assert busy.cycle_id == CycleId("cy-1")
    assert busy.busy_until == 500.0


def test_claim_busy_rejects_non_finite_busy_until():
    with pytest.raises(ValueError, match="finite"):
        ClaimBusy(chat_key=CK, cycle_id=CycleId("cy-1"), busy_until=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        ClaimBusy(chat_key=CK, cycle_id=CycleId("cy-1"), busy_until=float("nan"))


def test_claim_busy_is_frozen():
    busy = ClaimBusy(chat_key=CK, cycle_id=CycleId("cy-1"), busy_until=500.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        busy.busy_until = 999.0  # type: ignore[misc]


def test_outgoing_carries_delivery_key_metadata():
    out = Outgoing(chat_key=CK, text="hi")
    assert out.delivery_key is None
    out.delivery_key = "cy-1:0"
    assert out.delivery_key == "cy-1:0"


# ── Phase 2 gate boundary: GateSnapshot ─────────────────────────────────────

def test_gate_snapshot_requires_all_gate_context_fields():
    # The core claim/config/window facts are required constructor arguments:
    # a snapshot is fully populated by the gate, never partially built. Only
    # signal-derived and durable-state facts default (fail-closed).
    with pytest.raises(TypeError):
        GateSnapshot(chat_key=CK, mode="normal")  # type: ignore[call-arg]
    snap = _snapshot()
    assert snap.chat_key == CK
    assert snap.cycle_id == CycleId("c1")
    assert snap.start_msg_id == MessageRowId(1)
    assert snap.through_msg_id == MessageRowId(9)
    assert snap.evaluated_ts == 300.0
    assert snap.self_id == SelfId("bot")
    assert snap.mode == "normal"
    assert snap.threshold == 60
    assert snap.trigger_score == 100
    assert snap.frequency == 0.5
    assert snap.pending == 2
    assert snap.pending_messages == (
        _msg(row_id=MessageRowId(2)),
        _msg(row_id=MessageRowId(3)),
    )
    assert snap.recent == ()
    assert snap.window_count == 5
    assert snap.self_count == 2
    assert snap.last_nonself_ts == 250.0
    assert snap.idle_seconds == 30.0
    assert snap.recent_average_interval == 120.0
    assert snap.self_ratio == 0.1
    assert snap.is_group is True
    assert snap.is_focused is False
    assert snap.last_message is None


def test_gate_snapshot_pending_count_must_match_claimed_messages():
    # The claim-bounded contract: pending is the count of the claimed tuple,
    # never an independent number.
    with pytest.raises(ValueError, match="pending"):
        _snapshot(pending=3)  # the helper claims exactly 2 messages
    with pytest.raises(ValueError, match="pending"):
        _snapshot(pending_messages=(_msg(),))  # helper still claims pending=2
    # zero pending claims zero messages
    snap = _snapshot(pending=0, pending_messages=())
    assert snap.pending == 0
    assert snap.pending_messages == ()


def test_gate_snapshot_validates_window_counts():
    # self_count can never exceed the full-window count...
    with pytest.raises(ValueError, match="self_count"):
        _snapshot(window_count=3, self_count=4)
    # ...and the limited recent list can never exceed it either.
    with pytest.raises(ValueError, match="window_count"):
        _snapshot(recent=(_msg(), _msg()), window_count=1, self_count=1)
    # boundary values are legal
    _snapshot(window_count=0, self_count=0, pending=0, pending_messages=())


def test_gate_snapshot_rejects_non_finite_evaluation_timestamp():
    with pytest.raises(ValueError, match="finite"):
        _snapshot(evaluated_ts=float("inf"))


def test_gate_snapshot_carries_claim_window_and_targeting_facts():
    snap = _snapshot(
        last_message=_msg(),
        self_name="麦麦",
        has_direct_at=True,
        has_quote_to_self=True,
        has_other_assistant=True,
        hold_until=600.0,
        idle_streak=3,
    )
    # claim/cycle identity and the fixed local-row boundary
    assert snap.cycle_id == CycleId("c1")
    assert snap.start_msg_id == MessageRowId(1)
    assert snap.through_msg_id == MessageRowId(9)
    assert snap.evaluated_ts == 300.0
    # structured bot self identity
    assert snap.self_id == SelfId("bot")
    assert snap.self_name == "麦麦"
    # the claimed pending tuple is the source of the count
    assert len(snap.pending_messages) == snap.pending == 2
    # full-window facts alongside the limited recent list
    assert snap.window_count == 5
    assert snap.self_count == 2
    assert snap.last_nonself_ts == 250.0
    assert snap.recent == ()
    # direct-address targeting booleans
    assert snap.has_direct_at is True
    assert snap.has_quote_to_self is True
    assert snap.has_other_assistant is True
    # durable hold/idle state for trace/backoff policy
    assert snap.hold_until == 600.0
    assert snap.idle_streak == 3


def test_gate_snapshot_signal_and_durable_facts_default_fail_closed():
    snap = _snapshot()
    assert snap.self_name is None
    assert snap.has_direct_at is False
    assert snap.has_quote_to_self is False
    assert snap.has_other_assistant is False
    assert snap.hold_until is None
    assert snap.idle_streak == 0
    assert snap.previous_end_reason is None
    # the merged idle-backoff config defaults match GateBackoffConfig
    assert snap.backoff_base_s == 15.0
    assert snap.backoff_cap_s == 300.0
    assert snap.backoff_start_count == 2


def test_gate_snapshot_carries_merged_backoff_config_facts():
    # The exact merged per-chat backoff facts the gate's controller is
    # built from per evaluation.
    snap = _snapshot(backoff_base_s=30.0, backoff_cap_s=600.0, backoff_start_count=3)
    assert snap.backoff_base_s == 30.0
    assert snap.backoff_cap_s == 600.0
    assert snap.backoff_start_count == 3


def test_gate_snapshot_validates_backoff_config_facts():
    with pytest.raises(ValueError, match="backoff_base_s"):
        _snapshot(backoff_base_s=-1.0)
    with pytest.raises(ValueError, match="backoff_base_s"):
        _snapshot(backoff_base_s=float("inf"))
    with pytest.raises(ValueError, match="backoff_cap_s"):
        _snapshot(backoff_cap_s=-1.0)
    with pytest.raises(ValueError, match="backoff_cap_s"):
        _snapshot(backoff_cap_s=float("nan"))
    with pytest.raises(ValueError, match="backoff_cap_s"):
        _snapshot(backoff_base_s=300.0, backoff_cap_s=150.0)  # cap < base
    with pytest.raises(ValueError, match="backoff_start_count"):
        _snapshot(backoff_start_count=-1)
    with pytest.raises(ValueError, match="backoff_start_count"):
        _snapshot(backoff_start_count=True)  # type: ignore[arg-type]
    # boundary values are legal
    _snapshot(backoff_base_s=0.0, backoff_cap_s=0.0, backoff_start_count=0)


def test_gate_snapshot_carries_previous_end_reason():
    # History comes ONLY from the snapshot: the per-chat latest terminal
    # end reason is a snapshot fact, never a separate gate argument.
    snap = _snapshot(previous_end_reason="planner_no_tool_end")
    assert snap.previous_end_reason == "planner_no_tool_end"


def test_gate_snapshot_new_facts_are_frozen():
    snap = _snapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.pending_messages = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.has_direct_at = True  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.idle_streak = 5  # type: ignore[misc]


# ── Phase 2 gate boundary: RecentSnapshot ───────────────────────────────────

def test_recent_snapshot_defaults():
    snap = RecentSnapshot(chat_key=CK)
    assert snap.messages == ()
    assert snap.window_count == 0
    assert snap.self_count == 0
    assert snap.last_nonself_ts is None
    assert snap.since_ts is None
    assert snap.through_row_id is None


def test_recent_snapshot_carries_limited_list_and_full_window_counts():
    snap = RecentSnapshot(
        chat_key=CK,
        messages=(_msg(),),
        window_count=5,
        self_count=2,
        last_nonself_ts=250.0,
        since_ts=0.0,
        through_row_id=MessageRowId(9),
    )
    assert len(snap.messages) == 1  # the LIMITED rendered list
    assert snap.window_count == 5  # the FULL-window count, self included
    assert snap.self_count == 2
    assert snap.last_nonself_ts == 250.0
    assert snap.since_ts == 0.0
    assert snap.through_row_id == MessageRowId(9)


def test_recent_snapshot_validates_counts():
    # self_count can never exceed the full-window count...
    with pytest.raises(ValueError, match="self_count"):
        RecentSnapshot(chat_key=CK, window_count=3, self_count=4)
    # ...and the limited list can never exceed the full-window count either
    # (the rendered list may be limited but cannot change the counts).
    with pytest.raises(ValueError, match="window_count"):
        RecentSnapshot(chat_key=CK, messages=(_msg(), _msg()), window_count=1)
    # boundary values are legal
    RecentSnapshot(chat_key=CK, window_count=0, self_count=0)
    RecentSnapshot(chat_key=CK, messages=(_msg(),), window_count=1, self_count=1)


# ── Phase 2 gate boundary: contributions, reasons, delay, trace ─────────────

def test_contribution_carries_feature_error():
    failed = Contribution(feature="presence", op="add", value=0.0, error="boom")
    assert failed.error == "boom"
    ok = Contribution(feature="at", op="max", value=100.0, reason="direct @")
    assert ok.error is None
    assert ok.reason == "direct @"


def test_reason_tokens_are_stable_machine_readable():
    assert Reason.TRIGGER == "trigger"
    assert Reason.REFUSAL == "refusal"
    assert Reason.FEATURE_FAILURE == "feature_failure"
    assert Reason.BACKOFF == "backoff"
    assert Reason.MODE == "mode"
    assert Reason.DELAY == "delay"
    assert Reason.SKIP == "skip"


def test_decision_event_only_delay():
    # A delay may be event-only: delay_seconds=None still releases the claim
    # and records the event, but schedules no sleep.
    event_only = Decision(action="delay", delay_seconds=None, reason=Reason.DELAY)
    assert event_only.delay_seconds is None
    assert event_only.reason == "delay"
    timed = Decision(action="delay", delay_seconds=300.0, reason=Reason.BACKOFF)
    assert timed.delay_seconds == 300.0


def test_backoff_facts_defaults():
    bf = BackoffFacts()
    assert bf.applied is False
    assert bf.seconds is None
    assert bf.bypass_reason is None
    applied = BackoffFacts(applied=True, seconds=300.0)
    assert applied.seconds == 300.0
    bypassed = BackoffFacts(bypass_reason="focus")
    assert bypassed.bypass_reason == "focus"


def test_decision_trace_is_replay_safe_serializable():
    trace = DecisionTrace(
        chat_key=CK,
        mode="normal",
        threshold=60,
        trigger_score=100,
        pending=2,
        contributions=(
            Contribution(feature="at", op="max", value=100.0, reason="direct @"),
            Contribution(feature="presence", op="add", value=-10.0, error="boom"),
        ),
        decision=Decision(action="trigger", score=100.0, reason=Reason.TRIGGER),
        ts=123.0,
        snapshot_facts={
            "since_ts": 0.0,
            "through_row_id": 9,
            "window_count": 5,
            "self_count": 1,
            "last_nonself_ts": 250.0,
        },
        config={"mode": "normal", "threshold": 60, "trigger_score": 100},
        aggregates={"max": 100.0, "add": -10.0, "scale": 1.0},
        backoff=BackoffFacts(applied=False, bypass_reason="focus"),
    )
    # Every trace field is JSON-native: asdict + dumps round-trips losslessly.
    data = json.loads(json.dumps(dataclasses.asdict(trace)))
    assert data["chat_key"] == CK
    assert data["decision"]["action"] == "trigger"
    assert data["decision"]["reason"] == "trigger"
    assert data["contributions"][0]["feature"] == "at"
    assert data["contributions"][1]["error"] == "boom"
    assert data["snapshot_facts"]["window_count"] == 5
    assert data["snapshot_facts"]["through_row_id"] == 9
    assert data["config"]["threshold"] == 60
    assert data["aggregates"]["max"] == 100.0
    assert data["backoff"]["bypass_reason"] == "focus"
    assert data["ts"] == 123.0


def test_decision_trace_new_fact_fields_default_to_empty():
    trace = DecisionTrace(
        chat_key=CK, mode="normal", threshold=60, trigger_score=100, pending=0
    )
    assert trace.snapshot_facts == {}
    assert trace.config == {}
    assert trace.aggregates == {}
    assert trace.backoff is None


# ── Transcript boundary: fail-closed role shape ─────────────────────────────

def _call(cid: str) -> ToolCall:
    return ToolCall(id=ToolCallId(cid), name="query_memory", arguments={"q": cid})


def test_transcript_valid_role_shapes():
    # The four valid shapes: system/user content-only, assistant with
    # optional tool_calls, tool with a required tool_call_id.
    TranscriptMessage(role="system", content="persona")
    TranscriptMessage(role="user", content="hi")
    TranscriptMessage(role="assistant", content="thinking")
    TranscriptMessage(role="assistant", content=None, tool_calls=(_call("c1"),))
    TranscriptMessage(role="tool", tool_call_id=ToolCallId("c1"), name="query_memory", content="ok")
    # tool content is optional (empty results are legal)
    TranscriptMessage(role="tool", tool_call_id=ToolCallId("c1"), content="")


def test_transcript_rejects_unknown_role():
    with pytest.raises(ValueError, match="role"):
        TranscriptMessage(role="function", content="x")  # type: ignore[arg-type]


def test_transcript_tool_requires_tool_call_id():
    with pytest.raises(ValueError, match="tool_call_id"):
        TranscriptMessage(role="tool", name="query_memory", content="ok")


def test_transcript_tool_cannot_carry_tool_calls():
    with pytest.raises(ValueError, match="tool_calls"):
        TranscriptMessage(
            role="tool", tool_call_id=ToolCallId("c1"), tool_calls=(_call("c1"),)
        )


def test_transcript_non_tool_roles_reject_tool_fields():
    for role in ("system", "user"):
        with pytest.raises(ValueError, match="tool_call_id"):
            TranscriptMessage(
                role=role, content="x", tool_call_id=ToolCallId("c1")  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="tool name"):
            TranscriptMessage(role=role, content="x", name="query_memory")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="tool_calls"):
            TranscriptMessage(role=role, content="x", tool_calls=(_call("c1"),))  # type: ignore[arg-type]
    # assistant may carry tool_calls but never tool_call_id / name
    with pytest.raises(ValueError, match="tool_call_id"):
        TranscriptMessage(
            role="assistant", tool_call_id=ToolCallId("c1")  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="tool name"):
        TranscriptMessage(role="assistant", name="query_memory")  # type: ignore[arg-type]


def test_transcript_assistant_rejects_duplicate_tool_call_ids():
    # A duplicate id could otherwise yield duplicate tool results after
    # normalization — fail closed at construction.
    with pytest.raises(ValueError, match="unique ids"):
        TranscriptMessage(
            role="assistant", tool_calls=(_call("c1"), _call("c1"))
        )
    # distinct ids are fine
    TranscriptMessage(role="assistant", tool_calls=(_call("c1"), _call("c2")))


# ── Phase 6 adaptive foundation types ───────────────────────────────────────

def test_runtime_mode_tokens_are_stable_machine_readable():
    assert RuntimeMode.LIVE == "live"
    assert RuntimeMode.DRY_RUN == "dry_run"
    assert RuntimeMode.REPLAY == "replay"
    assert RuntimeMode.DOCTOR == "doctor"
    assert len({RuntimeMode.LIVE, RuntimeMode.DRY_RUN, RuntimeMode.REPLAY,
                RuntimeMode.DOCTOR}) == 4


def test_settlement_notice_validates_outcome_and_counts():
    ok = SettlementNotice(
        learner="personality", chat_key=CK, run_id=3, outcome="success",
        records_added=2, watermark=MessageRowId(5),
    )
    assert ok.outcome == "success"
    for bad in ("bogus", "", "done"):
        with pytest.raises(ValueError, match="outcome"):
            SettlementNotice(learner="l", chat_key=CK, run_id=1, outcome=bad)
    with pytest.raises(ValueError, match="run_id"):
        SettlementNotice(learner="l", chat_key=CK, run_id=-1, outcome="success")
    with pytest.raises(ValueError, match="nonnegative"):
        SettlementNotice(learner="l", chat_key=CK, run_id=1, outcome="success",
                         records_added=-1)
    with pytest.raises(ValueError, match="finite"):
        SettlementNotice(learner="l", chat_key=CK, run_id=1, outcome="success",
                         ts=float("inf"))


def test_agent_adaptive_context_validates_mode_and_timestamps():
    ok = AgentAdaptiveContext(chat_key=CK, learner="l", mode=RuntimeMode.LIVE)
    assert ok.mode == "live"
    for bad in ("bogus", "livex"):
        with pytest.raises(ValueError, match="runtime mode"):
            AgentAdaptiveContext(chat_key=CK, learner="l", mode=bad)
    with pytest.raises(ValueError, match="run_id"):
        AgentAdaptiveContext(chat_key=CK, learner="l", mode=RuntimeMode.LIVE,
                             run_id=-1)
    with pytest.raises(ValueError, match="finite"):
        AgentAdaptiveContext(chat_key=CK, learner="l", mode=RuntimeMode.LIVE,
                             now=float("nan"))


def test_learner_spec_is_pure_data_and_validates():
    # A spec is declarative: no repo/LLM/clock access anywhere on it.
    spec = LearnerSpec(name="personality", prompt="summarize", cadence_s=3600)
    assert spec.policy == "nonself"
    assert spec.batch_size == 1
    assert spec.enabled is True
    assert not hasattr(spec, "build_batch")
    assert not hasattr(spec, "parse")
    assert not hasattr(spec, "render")
    with pytest.raises(ValueError, match="non-empty"):
        LearnerSpec(name="", prompt="p", cadence_s=60)
    with pytest.raises(ValueError, match="non-empty"):
        LearnerSpec(name="x", prompt="", cadence_s=60)
    with pytest.raises(ValueError, match="cadence_s"):
        LearnerSpec(name="x", prompt="p", cadence_s=0)
    with pytest.raises(ValueError, match="policy"):
        LearnerSpec(name="x", prompt="p", cadence_s=60, policy="bogus")
    with pytest.raises(ValueError, match="batch_size"):
        LearnerSpec(name="x", prompt="p", cadence_s=60, batch_size=0)
    with pytest.raises(ValueError, match="bool"):
        LearnerSpec(name="x", prompt="p", cadence_s=60, enabled="yes")  # type: ignore[arg-type]


def test_learner_batch_validates_range_and_hash():
    with pytest.raises(ValueError, match="first_msg_id"):
        LearnerBatch(chat_key=CK, learner="l", first_msg_id=MessageRowId(5),
                     last_msg_id=MessageRowId(2), source_hash="h")
    with pytest.raises(ValueError, match="non-empty"):
        LearnerBatch(chat_key=CK, learner="l", first_msg_id=MessageRowId(1),
                     last_msg_id=MessageRowId(2), source_hash="")


def test_learner_draft_rejects_cross_chat_and_cross_learner():
    batch = LearnerBatch(chat_key=CK, learner="l", first_msg_id=MessageRowId(1),
                         last_msg_id=MessageRowId(2), source_hash="h")
    with pytest.raises(ValueError, match="batch chat_key"):
        LearnerDraft(chat_key=ChatKey("qq:group:other"), learner="l", batch=batch)
    with pytest.raises(ValueError, match="batch learner"):
        LearnerDraft(chat_key=CK, learner="other", batch=batch)
    rec = Record(learner="l", payload={"text": "x"}, chat_key=ChatKey("qq:group:other"))
    with pytest.raises(ValueError, match="record chat_key"):
        LearnerDraft(chat_key=CK, learner="l", batch=batch, records=(rec,))
    rec2 = Record(learner="other", payload={"text": "x"}, chat_key=CK)
    with pytest.raises(ValueError, match="record learner"):
        LearnerDraft(chat_key=CK, learner="l", batch=batch, records=(rec2,))
    with pytest.raises(ValueError, match="outcome"):
        LearnerDraft(chat_key=CK, learner="l", batch=batch, outcome="bogus")


def test_learner_run_request_lease_is_mandatory_and_finite():
    with pytest.raises(ValueError, match="finite"):
        LearnerRunRequest(chat_key=CK, learner="l", started_ts=1.0,
                          expires_at=float("inf"), now=1.0)
    with pytest.raises(ValueError, match="expires_at"):
        LearnerRunRequest(chat_key=CK, learner="l", started_ts=100.0,
                          expires_at=100.0, now=100.0)


def test_learner_grant_and_busy_validate():
    with pytest.raises(ValueError, match="run_id"):
        LearnerGrant(chat_key=CK, learner="l", run_id=-1, started_ts=1.0,
                     expires_at=2.0, start_msg_id=MessageRowId(0),
                     through_msg_id=MessageRowId(1))
    with pytest.raises(ValueError, match="lease"):
        LearnerGrant(chat_key=CK, learner="l", run_id=1, started_ts=2.0,
                     expires_at=1.0, start_msg_id=MessageRowId(0),
                     through_msg_id=MessageRowId(1))
    with pytest.raises(ValueError, match="finite"):
        LearnerBusy(chat_key=CK, learner="l", run_id=1, busy_until=float("nan"))


def test_learner_state_and_run_validate():
    with pytest.raises(ValueError, match="finite"):
        LearnerState(chat_key=CK, learner="l", updated_ts=float("inf"))
    with pytest.raises(ValueError, match="state"):
        LearnerRun(id=1, chat_key=CK, learner="l", started_ts=1.0, expires_at=2.0,
                   start_msg_id=MessageRowId(0), through_msg_id=MessageRowId(1),
                   state="bogus")
    with pytest.raises(ValueError, match="finite"):
        LearnerRun(id=1, chat_key=CK, learner="l", started_ts=1.0,
                   expires_at=float("nan"), start_msg_id=MessageRowId(0),
                   through_msg_id=MessageRowId(1))


def test_record_hit_validates_score():
    with pytest.raises(ValueError, match="finite"):
        RecordHit(chat_key=CK, learner="l", record_id=1, text="x", score=float("nan"))


def test_record_carries_adaptive_provenance_fields():
    rec = Record(learner="l", payload={"text": "x"}, chat_key=CK,
                 content_hash="h", source_first_msg_id=MessageRowId(1),
                 source_last_msg_id=MessageRowId(2))
    assert rec.content_hash == "h"
    assert rec.retired is False
    # Source range: both bounds or neither.
    with pytest.raises(ValueError, match="source range"):
        Record(learner="l", payload={}, chat_key=CK,
               source_first_msg_id=MessageRowId(1))
    with pytest.raises(ValueError, match="source_first_msg_id"):
        Record(learner="l", payload={}, chat_key=CK,
               source_first_msg_id=MessageRowId(5), source_last_msg_id=MessageRowId(2))
    with pytest.raises(ValueError, match="weight"):
        Record(learner="l", payload={}, chat_key=CK, weight=float("nan"))


def test_adaptive_boundary_types_are_frozen_dataclasses():
    for cls in (SettlementNotice, AgentAdaptiveContext, LearnerSpec, LearnerBatch,
                LearnerDraft, LearnerRunRequest, LearnerGrant, LearnerBusy,
                LearnerState, LearnerRun, RecordHit):
        assert dataclasses.is_dataclass(cls)
        assert getattr(cls, "__dataclass_params__").frozen


# ── Phase 6 P6.5 media catalog boundary types ───────────────────────────────

def _candidate(**kw: Any) -> MediaAssetCandidate:
    base: dict[str, Any] = dict(
        chat_key=CK, kind="sticker", cache_key="c" * 64, sha256="a" * 64,
        mime="image/gif",
    )
    base.update(kw)
    return MediaAssetCandidate(**base)


def _asset(**kw: Any) -> MediaAsset:
    base: dict[str, Any] = dict(
        id=1, chat_key=CK, kind="sticker", cache_key="c" * 64, sha256="a" * 64,
        mime="image/gif",
    )
    base.update(kw)
    return MediaAsset(**base)


def test_media_candidate_validates_kind_dims_and_finiteness():
    with pytest.raises(ValueError, match="kind"):
        _candidate(kind="video")
    with pytest.raises(ValueError, match="width"):
        _candidate(width=0)
    with pytest.raises(ValueError, match="width"):
        _candidate(width=True)
    with pytest.raises(ValueError, match="height"):
        _candidate(height=-1)
    with pytest.raises(ValueError, match="source_ts"):
        _candidate(source_ts=float("nan"))
    with pytest.raises(ValueError, match="created_ts"):
        _candidate(created_ts=float("inf"))
    with pytest.raises(ValueError, match="id"):
        _candidate(id=-1)
    with pytest.raises(ValueError, match="cache_key"):
        _candidate(cache_key="")
    with pytest.raises(ValueError, match="sha256"):
        _candidate(sha256="")
    with pytest.raises(ValueError, match="mime"):
        _candidate(mime="")


def test_media_asset_validates_status_version_uses_and_timestamps():
    with pytest.raises(ValueError, match="kind"):
        _asset(kind="video")
    with pytest.raises(ValueError, match="safety status"):
        _asset(safety_status="bogus")
    with pytest.raises(ValueError, match="safety_version"):
        _asset(safety_version=-1)
    with pytest.raises(ValueError, match="uses"):
        _asset(uses=-1)
    with pytest.raises(ValueError, match="approved_ts"):
        _asset(approved_ts=float("nan"))
    with pytest.raises(ValueError, match="revoked_ts"):
        _asset(revoked_ts=float("inf"))
    with pytest.raises(ValueError, match="last_used_ts"):
        _asset(last_used_ts=float("nan"))
    with pytest.raises(ValueError, match="created_ts"):
        _asset(created_ts=float("nan"))
    with pytest.raises(ValueError, match="id"):
        _asset(id=-1)


def test_media_kind_and_safety_status_constants():
    assert MediaKind.STICKER == "sticker"
    assert MediaKind.IMAGE == "image"
    assert MediaSafetyStatus.PENDING == "pending"
    assert MediaSafetyStatus.APPROVED == "approved"
    assert MediaSafetyStatus.REJECTED == "rejected"
    assert MediaSafetyStatus.REVOKED == "revoked"


def test_media_boundary_types_are_frozen_dataclasses():
    for cls in (MediaAssetCandidate, MediaAsset):
        assert dataclasses.is_dataclass(cls)
        assert getattr(cls, "__dataclass_params__").frozen