"""Phase 6 P6.6 bounded HookBus: error/timeout containment, pre_send
fail-closed ordering, and no hooks in dry-run/replay."""

from __future__ import annotations

import asyncio
import time

import pytest

from pretender.clock import VirtualClock
from pretender.config import Config
from pretender.cycle import CycleRunner
from pretender.gate import Gate
from pretender.registry import HookBus
from pretender.types import (
    AdapterEvent,
    ChatKey,
    CycleId,
    DecisionTrace,
    DispatchCause,
    DispatchRequest,
    Outgoing,
)
from tests.durable_helpers import (
    CK,
    make_identity,
    make_message,
    open_repo,
    run,
)


def _event() -> AdapterEvent:
    return AdapterEvent(type="message", payload=None)


def _trace() -> DecisionTrace:
    return DecisionTrace(
        chat_key=CK, mode="reply_necessity", threshold=8, trigger_score=80, pending=1
    )


# ── on_event: observational, fail-open ──────────────────────────────────────

def test_on_event_error_is_contained_and_flow_continues():
    bus = HookBus()
    order: list[str] = []

    @bus.on_event
    def boom(event):
        order.append("boom")
        raise RuntimeError("boom")

    @bus.on_event
    def after(event):
        order.append("after")

    run(bus.emit_event(_event()))
    assert order == ["boom", "after"]  # the failing hook never stops the bus


def test_on_event_timeout_is_contained():
    bus = HookBus(timeout_s=0.01)

    @bus.on_event
    async def slow(event):
        await asyncio.sleep(0.1)

    @bus.on_event
    def after(event):
        pass

    run(bus.emit_event(_event()))  # must not raise


def test_sync_hook_timeout_does_not_stall_event_loop():
    bus = HookBus(timeout_s=0.01)

    @bus.on_event
    def slow(event):
        time.sleep(0.15)

    async def scenario():
        started = asyncio.get_running_loop().time()
        task = asyncio.create_task(bus.emit_event(_event()))
        await asyncio.sleep(0)
        await asyncio.sleep(0.02)
        await task
        return asyncio.get_running_loop().time() - started

    assert run(scenario()) < 0.10


# ── on_cycle_end: contained ─────────────────────────────────────────────────

def test_on_cycle_end_error_is_contained():
    bus = HookBus()
    order: list[str] = []

    @bus.on_cycle_end
    def boom(chat_key, trace, end_reason):
        order.append("boom")
        raise RuntimeError("boom")

    @bus.on_cycle_end
    def after(chat_key, trace, end_reason):
        order.append("after")

    run(bus.emit_cycle_end(CK, _trace(), "skip"))
    assert order == ["boom", "after"]


# ── pre_send: fail-closed, ordered, bounded ─────────────────────────────────

def test_pre_send_error_fails_closed_to_no_output():
    bus = HookBus()

    @bus.pre_send
    def boom(out):
        raise RuntimeError("boom")

    result = run(bus.emit_pre_send(Outgoing(chat_key=CK, text="hi")))
    assert result is None  # fail-closed: no output


def test_pre_send_timeout_fails_closed_to_no_output():
    bus = HookBus(timeout_s=0.01)

    @bus.pre_send
    async def slow(out):
        await asyncio.sleep(0.1)
        return out

    result = run(bus.emit_pre_send(Outgoing(chat_key=CK, text="hi")))
    assert result is None  # fail-closed: no output


def test_pre_send_runs_in_registration_order_and_chains():
    bus = HookBus()
    order: list[str] = []

    @bus.pre_send
    def first(out):
        order.append("first")
        out.text += "a"
        return out

    @bus.pre_send
    def keep(out):
        order.append("keep")
        return None  # keep as-is

    @bus.pre_send
    def last(out):
        order.append("last")
        out.text += "b"
        return out

    result = run(bus.emit_pre_send(Outgoing(chat_key=CK, text="")))
    assert order == ["first", "keep", "last"]
    assert result is not None
    assert result.text == "ab"


def test_pre_send_error_stops_chain_and_suppresses():
    bus = HookBus()
    order: list[str] = []

    @bus.pre_send
    def first(out):
        order.append("first")
        return out

    @bus.pre_send
    def boom(out):
        order.append("boom")
        raise RuntimeError("boom")

    @bus.pre_send
    def never(out):
        order.append("never")
        return out

    result = run(bus.emit_pre_send(Outgoing(chat_key=CK, text="hi")))
    assert result is None
    assert order == ["first", "boom"]  # the chain stops at the failure


# ── cycle integration: pre_send before pipeline/outbox conversion ───────────

async def _begin_dispatch(repo):
    await repo.ingest_message(make_identity(), make_message(recv_ts=100.0))
    return await repo.begin_dispatch(
        DispatchRequest(
            chat_key=CK,
            cause=DispatchCause.INBOUND,
            cycle_id=CycleId("cy-1"),
            started_ts=200.0,
            expires_at=500.0,
            now=200.0,
        )
    )


def test_pre_send_runs_before_outbox_conversion(tmp_path):
    """A pre_send hook's modification is visible in the converted outbox
    items — the hook runs BEFORE the pipeline/outbox conversion, never after
    row persistence."""

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        hooks = HookBus()

        @hooks.pre_send
        def append(out):
            out.text += " [hook]"
            return out

        runner = CycleRunner(
            repo,
            Gate(),
            Config(),
            clock=VirtualClock(epoch=200.0),
            hooks=hooks,
            dry_run=False,
            uuid_fn=lambda: "cy-1",
        )
        grant = await _begin_dispatch(repo)
        items = await runner._agent_outbox_items(grant, "hello", None)
        await repo.close()
        return items

    items = run(scenario())
    assert items is not None
    assert len(items) >= 1
    # The hook's modification is visible in the converted items — the hook
    # ran BEFORE the pipeline/outbox conversion (the split stage may split
    # the text, so the joined parts carry the marker).
    assert "[hook]" in "".join(item.text for item in items)


def test_pre_send_failure_suppresses_outbox_rows(tmp_path):
    """A pre_send hook failure fails closed to NO output: the outbox
    conversion never runs and no rows are ever persisted."""

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        hooks = HookBus()

        @hooks.pre_send
        def boom(out):
            raise RuntimeError("boom")

        runner = CycleRunner(
            repo,
            Gate(),
            Config(),
            clock=VirtualClock(epoch=200.0),
            hooks=hooks,
            dry_run=False,
            uuid_fn=lambda: "cy-1",
        )
        grant = await _begin_dispatch(repo)
        items = await runner._agent_outbox_items(grant, "hello", None)
        await repo.close()
        return items

    items = run(scenario())
    assert items is None  # fail-closed: no output


def test_pre_send_never_runs_in_dry_run(tmp_path):
    """pre_send hooks never run in dry-run — the dry-run lane is
    deterministic and plugin-free."""

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        hooks = HookBus()
        seen: list[str] = []

        @hooks.pre_send
        def mark(out):
            seen.append("pre_send")
            return out

        runner = CycleRunner(
            repo,
            Gate(),
            Config(),
            clock=VirtualClock(epoch=200.0),
            hooks=hooks,
            dry_run=True,
            uuid_fn=lambda: "cy-1",
        )
        grant = await _begin_dispatch(repo)
        items = await runner._agent_outbox_items(grant, "hello", None)
        await repo.close()
        return items, seen

    items, seen = run(scenario())
    assert items is not None
    assert seen == []  # the hook never ran


def test_on_cycle_end_never_runs_in_dry_run(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        hooks = HookBus()
        seen: list[str] = []

        @hooks.on_cycle_end
        def mark(chat_key, trace, end_reason):
            seen.append(end_reason)

        runner = CycleRunner(
            repo,
            Gate(),
            Config(),
            clock=VirtualClock(epoch=200.0),
            hooks=hooks,
            dry_run=True,
            uuid_fn=lambda: "cy-1",
        )
        grant = await _begin_dispatch(repo)
        await runner.run_dispatch(grant)
        await repo.close()
        return seen

    seen = run(scenario())
    assert seen == []  # no hooks in dry-run
