"""Phase 6 P6.6b chat controls: staged set_focus/notify_chat tools, the
durable ChatControlRepository (idempotency, one focus per account,
cross-account rejection), and the LIVE-only cycle application (dry/replay
discard controls with zero mutation)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

import pytest

from pretender.clock import VirtualClock
from pretender.config import Config
from pretender.cycle import CycleRunner
from pretender.gate import Gate
from pretender.registry import HookBus
from pretender.tools.chatctl import (
    FOCUS_MAX_S,
    FOCUS_MIN_S,
    NOTIFY_MAX_CHARS,
    NOTIFY_MAX_S,
    ChatControlCallbacks,
    notify_chat,
    set_focus,
)
from pretender.tools.core import (
    CoreToolRegistry,
    ToolContext,
    dispatch_call,
    register_core_tools,
)
from pretender.types import (
    ChatControl,
    ChatControlIntent,
    ChatControlKind,
    ChatKey,
    ChatState,
    CycleId,
    DispatchCause,
    DispatchRequest,
    Message,
    MessageId,
    PlatformId,
    SelfId,
    SenderId,
    ToolCall,
    ToolCallId,
)
from tests.durable_helpers import CK, make_identity, open_repo, run

CK2 = ChatKey("qq:group:222222")
CK3 = ChatKey("qq:group:333333")
OTHER_ACCOUNT = ChatKey("qq:group:999999")


def _identity(chat_key: ChatKey = CK, self_id: str = "bot-1") -> object:
    return make_identity(chat_key=chat_key, self_id=self_id)


def _call(name: str, arguments: dict, cid: str = "c1") -> ToolCall:
    return ToolCall(id=ToolCallId(cid), name=name, arguments=dict(arguments))


def _ctx(
    *,
    chat_key: ChatKey = CK,
    resolve: Callable[[str], Awaitable[bool]] | None = None,
    registry: CoreToolRegistry | None = None,
) -> ToolContext:
    callbacks = ChatControlCallbacks(resolve_chat=resolve) if resolve else None
    return ToolContext(
        chat_key=chat_key,
        chat_kind="group",
        registry=registry,
        chat_controls=callbacks,
    )


async def _resolve_ok(target_key: str) -> bool:
    return target_key in (str(CK2), str(CK3))


# ── tool staging: typed intents, never direct writes ────────────────────────

def test_set_focus_stages_typed_intent():
    reg = register_core_tools()
    reg.activate("set_focus")
    ctx = _ctx(resolve=_resolve_ok, registry=reg)
    result = run(dispatch_call(_call("set_focus", {"chat_key": str(CK2), "duration_s": 300}), ctx, reg))
    assert result.ok
    controls = ctx.chat_controls
    assert len(controls) == 1
    intent = controls[0]
    assert intent.kind == ChatControlKind.FOCUS
    assert intent.target_chat_key == CK2
    assert intent.ttl_s == 300
    assert intent.text is None


def test_notify_chat_stages_typed_intent():
    reg = register_core_tools()
    reg.activate("notify_chat")
    ctx = _ctx(resolve=_resolve_ok, registry=reg)
    result = run(
        dispatch_call(
            _call("notify_chat", {"chat_key": str(CK2), "text": "开会了", "ttl_s": 600}),
            ctx,
            reg,
        )
    )
    assert result.ok
    controls = ctx.chat_controls
    assert len(controls) == 1
    intent = controls[0]
    assert intent.kind == ChatControlKind.NOTIFY
    assert intent.target_chat_key == CK2
    assert intent.ttl_s == 600
    assert intent.text == "开会了"


def test_set_focus_rejects_duration_out_of_bounds():
    reg = register_core_tools()
    reg.activate("set_focus")
    ctx = _ctx(resolve=_resolve_ok, registry=reg)
    for bad in (FOCUS_MIN_S - 1, FOCUS_MAX_S + 1, 0, -5):
        result = run(
            dispatch_call(
                _call("set_focus", {"chat_key": str(CK2), "duration_s": bad}), ctx, reg
            )
        )
        assert not result.ok
        assert "duration_s" in (result.error or "")
    assert ctx.chat_controls == ()


def test_notify_chat_rejects_bad_ttl_and_text():
    reg = register_core_tools()
    reg.activate("notify_chat")
    ctx = _ctx(resolve=_resolve_ok, registry=reg)
    for bad in (0, -1, NOTIFY_MAX_S + 1):
        result = run(
            dispatch_call(
                _call("notify_chat", {"chat_key": str(CK2), "text": "x", "ttl_s": bad}),
                ctx,
                reg,
            )
        )
        assert not result.ok
        assert "ttl_s" in (result.error or "")
    result = run(
        dispatch_call(
            _call("notify_chat", {"chat_key": str(CK2), "text": "   "}), ctx, reg
        )
    )
    assert not result.ok
    assert "text" in (result.error or "")
    result = run(
        dispatch_call(
            _call(
                "notify_chat",
                {"chat_key": str(CK2), "text": "x" * (NOTIFY_MAX_CHARS + 1)},
            ),
            ctx,
            reg,
        )
    )
    assert not result.ok
    assert "text" in (result.error or "")
    assert ctx.chat_controls == ()


def test_controls_reject_unknown_or_cross_account_target():
    reg = register_core_tools()
    reg.activate("set_focus")
    reg.activate("notify_chat")

    async def resolve_unknown(target_key: str) -> bool:
        return False

    ctx = _ctx(resolve=resolve_unknown, registry=reg)
    result = run(
        dispatch_call(_call("set_focus", {"chat_key": str(CK2), "duration_s": 300}), ctx, reg)
    )
    assert not result.ok
    assert "unknown or cross-account" in (result.error or "")
    result = run(
        dispatch_call(
            _call("notify_chat", {"chat_key": str(CK2), "text": "hi"}), ctx, reg
        )
    )
    assert not result.ok
    assert "unknown or cross-account" in (result.error or "")
    assert ctx.chat_controls == ()


def test_controls_fail_closed_without_callbacks():
    reg = register_core_tools()
    reg.activate("set_focus")
    ctx = _ctx(registry=reg)  # no callbacks
    result = run(
        dispatch_call(_call("set_focus", {"chat_key": str(CK2), "duration_s": 300}), ctx, reg)
    )
    assert not result.ok
    assert "not available" in (result.error or "")


def test_controls_are_deferred_until_tool_search():
    reg = register_core_tools()
    ctx = _ctx(resolve=_resolve_ok, registry=reg)
    # Not activated: the dispatch fails safely.
    result = run(
        dispatch_call(_call("set_focus", {"chat_key": str(CK2), "duration_s": 300}), ctx, reg)
    )
    assert not result.ok
    assert "deferred" in (result.error or "")


# ── durable repository: idempotency / one focus / cross-account ─────────────

async def _seed_chats(repo):
    await repo.upsert_chat(make_identity(chat_key=CK, self_id="bot-1"))
    await repo.upsert_chat(make_identity(chat_key=CK2, self_id="bot-1"))
    await repo.upsert_chat(make_identity(chat_key=CK3, self_id="bot-1"))
    await repo.upsert_chat(make_identity(chat_key=OTHER_ACCOUNT, self_id="bot-2"))


def _control(
    *,
    chat_key: ChatKey = CK2,
    kind: str = ChatControlKind.FOCUS,
    ttl_until: float = 500.0,
    created_ts: float = 200.0,
    dispatch_id: int = 1,
    intent_seq: int = 0,
    source: ChatKey = CK,
    text: str | None = None,
) -> ChatControl:
    return ChatControl(
        chat_key=chat_key,
        kind=kind,
        ttl_until=ttl_until,
        created_ts=created_ts,
        dispatch_id=dispatch_id,
        intent_seq=intent_seq,
        source_chat_key=source,
        text=text,
    )


def test_apply_chat_control_is_idempotent(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await _seed_chats(repo)
        control = _control()
        first = await repo.apply_chat_control(control)
        second = await repo.apply_chat_control(control)
        active = await repo.list_active_controls(CK2, now=200.0)
        await repo.close()
        return first, second, active

    first, second, active = run(scenario())
    assert first is True
    assert second is False  # duplicate (dispatch_id, intent_seq): no-op
    assert len(active) == 1


def test_one_focus_per_account(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await _seed_chats(repo)
        # Focus chat A, then focus chat B on the SAME account.
        await repo.apply_chat_control(_control(chat_key=CK2, dispatch_id=1))
        await repo.apply_chat_control(_control(chat_key=CK3, dispatch_id=2))
        a_active = await repo.list_active_controls(CK2, now=200.0)
        b_active = await repo.list_active_controls(CK3, now=200.0)
        # A focus on a DIFFERENT account never clears the same-account focus.
        await repo.apply_chat_control(
            _control(chat_key=OTHER_ACCOUNT, dispatch_id=3, source=OTHER_ACCOUNT)
        )
        b_after = await repo.list_active_controls(CK3, now=200.0)
        await repo.close()
        return a_active, b_active, b_after

    a_active, b_active, b_after = run(scenario())
    assert a_active == []  # cleared: one focus per account
    assert len(b_active) == 1
    assert len(b_after) == 1  # the other-account focus never touched it


def test_apply_chat_control_rejects_unknown_and_cross_account(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await _seed_chats(repo)
        unknown = await repo.apply_chat_control(
            _control(chat_key=ChatKey("qq:group:000000"))
        )
        cross = await repo.apply_chat_control(
            _control(chat_key=OTHER_ACCOUNT, source=CK)
        )
        active = await repo.list_active_controls(OTHER_ACCOUNT, now=200.0)
        await repo.close()
        return unknown, cross, active

    unknown, cross, active = run(scenario())
    assert unknown is False
    assert cross is False  # source and target accounts differ
    assert active == []


def test_list_active_controls_filters_expired(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await _seed_chats(repo)
        await repo.apply_chat_control(_control(ttl_until=500.0, dispatch_id=1))
        await repo.apply_chat_control(
            _control(
                ttl_until=150.0,
                created_ts=100.0,
                dispatch_id=2,
                intent_seq=1,
                kind=ChatControlKind.NOTIFY,
                text="expired",
            )
        )
        active = await repo.list_active_controls(CK2, now=200.0)
        await repo.close()
        return active

    active = run(scenario())
    assert len(active) == 1
    assert active[0].dispatch_id == 1


# ── cycle integration: LIVE-only application, focus merge, dry/replay ───────

async def _begin_dispatch(repo, chat_key: ChatKey = CK):
    await repo.ingest_message(make_identity(chat_key=chat_key), _msg(chat_key))
    return await repo.begin_dispatch(
        DispatchRequest(
            chat_key=chat_key,
            cause=DispatchCause.INBOUND,
            cycle_id=CycleId("cy-1"),
            started_ts=200.0,
            expires_at=500.0,
            now=200.0,
        )
    )


def _msg(chat_key: ChatKey = CK) -> Message:
    return Message(
        chat_key=chat_key,
        sender_id=SenderId("u1"),
        sender_name="user",
        is_self=False,
        text="hi",
        id=MessageId("m1"),
        recv_ts=100.0,
    )


def test_apply_chat_controls_persists_after_terminal_settlement(tmp_path):
    """The CycleRunner applies staged controls idempotently AFTER the
    terminal settlement, using the dispatch id + intent sequence."""

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await _seed_chats(repo)
        runner = CycleRunner(
            repo,
            Gate(),
            Config(),
            clock=VirtualClock(epoch=200.0),
            hooks=HookBus(),
            dry_run=False,
            uuid_fn=lambda: "cy-1",
        )
        grant = await _begin_dispatch(repo)
        intents = (
            ChatControlIntent(
                kind=ChatControlKind.FOCUS, target_chat_key=CK2, ttl_s=300
            ),
            ChatControlIntent(
                kind=ChatControlKind.NOTIFY,
                target_chat_key=CK3,
                ttl_s=600,
                text="开会了",
            ),
        )
        await runner._apply_chat_controls(grant, intents)
        focus = await repo.list_active_controls(CK2, now=200.0)
        notify = await repo.list_active_controls(CK3, now=200.0)
        await repo.close()
        return focus, notify

    focus, notify = run(scenario())
    assert len(focus) == 1
    assert focus[0].kind == ChatControlKind.FOCUS
    assert focus[0].dispatch_id == 1
    assert focus[0].intent_seq == 0
    assert len(notify) == 1
    assert notify[0].kind == ChatControlKind.NOTIFY
    assert notify[0].text == "开会了"
    assert notify[0].intent_seq == 1


def test_focus_merge_makes_target_gate_evaluate_focused(tmp_path):
    """An active focus control extends the target chat's focus_until, so the
    target gate evaluates as focused (the notify traverses the target gate)."""

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await _seed_chats(repo)
        await repo.apply_chat_control(_control(chat_key=CK2, ttl_until=500.0))
        runner = CycleRunner(
            repo,
            Gate(),
            Config(),
            clock=VirtualClock(epoch=200.0),
            hooks=HookBus(),
            dry_run=False,
            uuid_fn=lambda: "cy-1",
        )
        state = ChatState(chat_key=CK2)
        merged = await runner._merge_active_controls(CK2, state)
        await repo.close()
        return merged

    merged = run(scenario())
    assert merged.focus_until == 500.0


def test_dry_run_discards_controls_with_zero_mutation(tmp_path):
    """Dry-run never reads or writes controls: a repo whose control surface
    raises proves the dry-run path never touches it."""

    class _NoControlsRepo:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name in ("apply_chat_control", "list_active_controls"):
                raise AssertionError(f"dry-run must not call {name}")
            return getattr(self._inner, name)

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await _seed_chats(repo)
        guarded = _NoControlsRepo(repo)
        runner = CycleRunner(
            guarded,  # type: ignore[arg-type]
            Gate(),
            Config(),
            clock=VirtualClock(epoch=200.0),
            hooks=HookBus(),
            dry_run=True,
            uuid_fn=lambda: "cy-1",
        )
        grant = await _begin_dispatch(repo)
        decision = await runner.run_dispatch(grant)
        controls = await repo.list_active_controls(CK2, now=200.0)
        await repo.close()
        return decision, controls

    decision, controls = run(scenario())
    assert decision.action in ("trigger", "skip", "delay")
    assert controls == []  # zero mutation


def test_replay_never_reads_controls():
    """Replay is storage-free: the frozen snapshot carries the focus facts
    and no control surface is ever consulted."""
    from pretender.cycle import replay_corpus
    from pretender.types import AdapterEvent

    msg = _msg()
    events = [AdapterEvent(type="message", payload=msg, ts=100.0)]
    result = replay_corpus(
        events, chat_key=CK, identity=make_identity(), cfg=Config()
    )
    assert result.decisions >= 1


# ── protocol conformance / type validation ──────────────────────────────────

def test_sqlite_repo_satisfies_chat_control_repository_protocol():
    from pretender.repo import SqliteRepository as Repo

    # Structural check: the repo exposes the two surface methods.
    assert hasattr(Repo, "apply_chat_control")
    assert hasattr(Repo, "list_active_controls")


def test_chat_control_intent_validation():
    with pytest.raises(ValueError, match="kind"):
        ChatControlIntent(kind="bogus", target_chat_key=CK2, ttl_s=300)
    with pytest.raises(ValueError, match="ttl_s"):
        ChatControlIntent(kind=ChatControlKind.FOCUS, target_chat_key=CK2, ttl_s=0)
    with pytest.raises(ValueError, match="ttl_s"):
        ChatControlIntent(kind=ChatControlKind.FOCUS, target_chat_key=CK2, ttl_s=-1)
    with pytest.raises(ValueError, match="text"):
        ChatControlIntent(
            kind=ChatControlKind.NOTIFY,
            target_chat_key=CK2,
            ttl_s=60,
            text=5,  # type: ignore[arg-type]
        )


def test_chat_control_validation():
    with pytest.raises(ValueError, match="kind"):
        ChatControl(
            chat_key=CK2,
            kind="bogus",
            ttl_until=500.0,
            created_ts=200.0,
            dispatch_id=1,
            intent_seq=0,
            source_chat_key=CK,
        )
    with pytest.raises(ValueError, match="ttl_until"):
        ChatControl(
            chat_key=CK2,
            kind=ChatControlKind.FOCUS,
            ttl_until=100.0,
            created_ts=200.0,
            dispatch_id=1,
            intent_seq=0,
            source_chat_key=CK,
        )
    with pytest.raises(ValueError, match="dispatch_id"):
        ChatControl(
            chat_key=CK2,
            kind=ChatControlKind.FOCUS,
            ttl_until=500.0,
            created_ts=200.0,
            dispatch_id=-1,
            intent_seq=0,
            source_chat_key=CK,
        )