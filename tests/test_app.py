"""App: composition, safe start/shutdown, console-only run with startup
re-arm of safe pending outbox work — and NO pump on inbound wake."""

from __future__ import annotations

import asyncio
import io
import threading

import pytest

from pretender.app import App
from pretender.adapters.console import ConsoleAdapter
from pretender.clock import VirtualClock
from pretender.config import Config
from pretender.cycle import CycleRunner, replay_corpus
from pretender.db import Database
from pretender.errors import ConfigError
from pretender.gate import Gate
from pretender.ingest import Ingest
from pretender.record import read_corpus, read_markers
from pretender.repo import SqliteRepository
from pretender.scheduler import LedgerScheduler, Scheduler
from pretender.types import (
    AdapterEvent,
    ChatIdentity,
    ChatKey,
    ChatState,
    CommitSeq,
    CycleId,
    Decision,
    DecisionTrace,
    DispatchCause,
    DispatchGrant,
    DispatchRequest,
    DispatchSettle,
    EchoStatus,
    IngestResult,
    Message,
    MessageId,
    MessageRowId,
    Outgoing,
    PlatformId,
    Reason,
    SelfId,
    SenderId,
)
from tests.durable_helpers import (
    finish_batch,
    make_claim,
    make_identity,
    make_message,
    open_repo,
    run,
)


def make_config(tmp_path) -> Config:
    return Config.from_dict({"storage": {"db_path": str(tmp_path / "data" / "app.db")}})


def make_app(tmp_path, **kw) -> App:
    cfg = make_config(tmp_path)
    clock = kw.pop("clock", VirtualClock())
    input_stream = kw.pop("input_stream", io.StringIO(""))
    adapter = ConsoleAdapter(
        clock=clock, input_stream=input_stream, output_stream=io.StringIO()
    )
    return App.build(cfg, clock=clock, adapter=adapter, **kw)


# ── composition ─────────────────────────────────────────────────────────────

def test_build_wires_the_full_stack(tmp_path):
    app = make_app(tmp_path)
    assert app.cfg is not None
    assert app.db is not None
    assert app.repo is not None
    assert app.recorder is not None
    assert app.adapter is not None
    assert app.ingest is not None
    assert app.outbox is not None
    assert app.clock is not None


def test_build_owns_shared_memory_search_and_person_service(tmp_path):
    """The default stack owns a shared local MemorySearch (FTS-only when no
    optional embed is configured) and a PersonService over the
    SqliteRepository/KnowledgeRepository."""
    app = make_app(tmp_path)
    assert app.memory_search is not None
    assert app.person_service is not None
    # The MemorySearch is FTS-only by default (no optional embed configured).
    assert app.memory_search._embed is None


def test_run_observes_sender_alias_after_ingest(tmp_path):
    """End to end: a fed non-self message is ingested and its sender alias is
    observed into a person row."""
    async def scenario():
        app = make_app(tmp_path)
        await app.start()
        await app.adapter.feed("hello there")
        await app.run()  # run() shuts the app down on EOF (closes the db)
        # Read the observed person from a fresh connection.
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        repo2 = SqliteRepository(db2)
        p = await repo2.get_person(app.adapter.chat_key, SenderId("user"))
        await db2.close()
        return p

    p = run(scenario())
    assert p is not None
    assert "user" in p.names


def test_build_rejects_unsupported_adapter(tmp_path):
    """An adapter that is neither console nor onebot is rejected — no
    message may reach an ungated send path."""

    class TelegramAdapter(ConsoleAdapter):
        name = "telegram"

    cfg = make_config(tmp_path)
    with pytest.raises(ConfigError, match="unsupported adapter"):
        App.build(
            cfg,
            clock=VirtualClock(),
            adapter=TelegramAdapter(clock=VirtualClock()),
        )


def test_build_rejects_onebot_in_dry_run(tmp_path):
    """Dry-run is console-only: a OneBot adapter is rejected so dry-run can
    never accidentally send through a live bridge."""

    class FakeOneBot(ConsoleAdapter):
        name = "onebot"

    cfg = make_config(tmp_path)
    with pytest.raises(ConfigError, match="dry-run"):
        App.build(
            cfg,
            clock=VirtualClock(),
            adapter=FakeOneBot(clock=VirtualClock()),
            dry_run=True,
        )


def test_build_accepts_onebot_in_live_mode(tmp_path):
    """Live mode accepts a OneBot adapter (the Phase 4 live-delivery
    bridge)."""
    from pretender.adapters.onebot import OneBotAdapter

    cfg = make_config(tmp_path)
    app = App.build(
        cfg,
        clock=VirtualClock(),
        adapter=OneBotAdapter(clock=VirtualClock(), self_id="10001"),
        dry_run=False,
    )
    assert app.adapter.name == "onebot"
    assert app.outbox is not None
    assert app.ingest is not None


def test_live_startup_failure_closes_resources(tmp_path):
    """Connect/readiness/recovery are inside the live shutdown guard."""

    class FailingConsole(ConsoleAdapter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.closed = False

        async def connect(self) -> None:
            raise RuntimeError("connect failed")

        async def close(self) -> None:
            self.closed = True

    async def scenario():
        cfg = make_config(tmp_path)
        clock = VirtualClock()
        adapter = FailingConsole(clock=clock, input_stream=io.StringIO(""))
        app = App.build(cfg, clock=clock, adapter=adapter, dry_run=False)
        with pytest.raises(RuntimeError, match="connect failed"):
            await app.run()
        return app, adapter

    app, adapter = run(scenario())
    assert adapter.closed is True
    assert app._started is False


# ── start / shutdown ────────────────────────────────────────────────────────

def test_start_creates_database_with_schema(tmp_path):
    async def scenario():
        app = make_app(tmp_path)
        await app.start()
        version = await app.db.read(
            lambda c: c.execute("PRAGMA user_version").fetchone()[0]
        )
        await app.shutdown()
        return version, (tmp_path / "data" / "app.db").exists()

    version, exists = run(scenario())
    assert version == 15
    assert exists


def test_start_and_shutdown_are_idempotent(tmp_path):
    async def scenario():
        app = make_app(tmp_path)
        await app.start()
        await app.start()  # no-op
        await app.shutdown()
        await app.shutdown()  # no-op
        return True

    assert run(scenario())


def test_shutdown_closes_db(tmp_path):
    async def scenario():
        app = make_app(tmp_path)
        await app.start()
        await app.shutdown()
        with pytest.raises(RuntimeError):
            await app.db.read(lambda c: 1)
        return True

    assert run(scenario())


# ── Gate 5 remediation: shutdown robustness ──────────────────────────────────

def test_shutdown_cleans_up_when_semantic_task_failed(tmp_path):
    """A failed semantic task never poisons shutdown: the scheduler/adapter/
    clients/recorder/DB are still cleaned up."""
    async def scenario():
        app = make_app(tmp_path)

        class ExplodingBackfill:
            def __init__(self):
                self.cancelled = False

            def cancel(self):
                self.cancelled = True

            async def run(self):
                raise RuntimeError("semantic boom")

        app._semantic_backfill = ExplodingBackfill()
        await app.start()
        # The semantic task failed; shutdown must still clean everything.
        await app.shutdown()
        try:
            await app.db.read(lambda c: 1)
            db_closed = False
        except RuntimeError:
            db_closed = True
        return app._semantic_task is None, db_closed

    task_cleared, db_closed = run(scenario())
    assert task_cleared
    assert db_closed


def test_shutdown_cleans_up_when_start_never_called(tmp_path):
    """shutdown cleans owned resources (adapter/clients/recorder/DB) even
    when ``start()`` was never called."""
    async def scenario():
        app = make_app(tmp_path)
        await app.shutdown()
        try:
            await app.db.read(lambda c: 1)
            db_closed = False
        except RuntimeError:
            db_closed = True
        return db_closed

    assert run(scenario())


# ── run loop ────────────────────────────────────────────────────────────────

def test_run_ingests_messages_without_sending(tmp_path):
    """Inbound events are recorded and committed but NEVER pump the outbox
    before Phase 2's gate/cycle."""

    async def scenario():
        app = make_app(tmp_path)
        await app.start()
        await app.adapter.feed("hello there")
        await app.run()  # run() shuts the app down on EOF
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        repo2 = SqliteRepository(db2)
        msg = await repo2.get_message(app.adapter.chat_key, "console:in:1")
        await db2.close()
        return msg, len(app.adapter.sent)

    msg, sent_count = run(scenario())
    assert msg is not None and msg.text == "hello there"
    assert sent_count == 0  # no ungated send


def test_run_rearms_safe_pending_outbox_once_at_startup(tmp_path):
    """Pending rows from a completed durable cycle are re-armed ONCE at
    startup; the inbound message must not trigger a second pump."""

    async def scenario():
        app = make_app(tmp_path)
        await app.start()
        # A completed cycle left one pending row (e.g. crash before send).
        await app.repo.upsert_chat(make_identity(chat_key="console:group:demo",
                                                 platform="console", self_id="bot"))
        await finish_batch(
            app.repo,
            app.outbox.to_items(
                Outgoing(chat_key=app.adapter.chat_key, text="auto reply",
                         idem_key="k1"),
                "cy-1",
            ),
            chat_key="console:group:demo",
        )
        await app.adapter.feed("hello there")
        await app.run()  # the startup drain is awaited before events
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        repo2 = SqliteRepository(db2)
        echo_id = await db2.read(
            lambda c: c.execute(
                "SELECT platform_msg_id FROM messages WHERE is_self = 1"
            ).fetchone()[0]
        )
        echo = await repo2.get_message(app.adapter.chat_key, echo_id)
        await db2.close()
        return echo, len(app.adapter.sent)

    echo, sent_count = run(scenario())
    assert sent_count == 1  # startup re-arm only — no duplicate pump
    assert echo is not None and echo.is_self and echo.text == "auto reply"


def test_run_returns_when_adapter_stream_ends(tmp_path):
    async def scenario():
        app = make_app(tmp_path)
        await app.run()  # empty StringIO -> EOF -> clean return
        return True

    assert run(scenario())


def test_run_is_safe_without_prior_start(tmp_path):
    async def scenario():
        app = make_app(tmp_path)
        await app.run()
        await app.shutdown()
        return True

    assert run(scenario())


class _BlockingStream:
    """A REPL input stream that blocks in readline until ``release`` is
    called, then returns EOF — lets a test observe the worker's future-
    paced send before the run loop ends."""

    def __init__(self) -> None:
        self._release = threading.Event()

    def readline(self) -> str:
        self._release.wait()
        return ""

    def release(self) -> None:
        self._release.set()


# ── outbox worker: startup drain, future pacing, cancellation ───────────────

def test_run_drains_more_than_ten_startup_rows(tmp_path):
    """Startup recovery drains EVERY due pending row, not only ten."""

    async def scenario():
        app = make_app(tmp_path)
        await app.start()
        await app.repo.upsert_chat(make_identity(chat_key="console:group:demo",
                                                 platform="console", self_id="bot"))
        out = Outgoing(
            chat_key=app.adapter.chat_key, text="x",
            parts=[f"m{i}" for i in range(15)], idem_key="batch",
        )
        await finish_batch(
            app.repo, app.outbox.to_items(out, "cy-1"),
            chat_key="console:group:demo",
        )
        await app.run()  # the startup drain is awaited before events
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        states = await db2.read(
            lambda c: [
                r[0] for r in c.execute("SELECT state FROM outbox ORDER BY id")
            ]
        )
        await db2.close()
        return len(app.adapter.sent), states

    sent, states = run(scenario())
    assert sent == 15  # every due row, no duplicates
    assert states == ["sent"] * 15


def test_run_schedules_future_paced_row_without_inbound(tmp_path):
    """A future send_after_ts row is sent when due, with no inbound input."""

    async def scenario():
        stream = _BlockingStream()
        app = make_app(tmp_path, input_stream=stream)
        await app.start()
        await app.repo.upsert_chat(make_identity(chat_key="console:group:demo",
                                                 platform="console", self_id="bot"))
        future = app.clock.now() + 100.0
        out = Outgoing(chat_key=app.adapter.chat_key, text="paced",
                       idem_key="k1", send_after_ts=future)
        await finish_batch(
            app.repo, app.outbox.to_items(out, "cy-1"),
            chat_key="console:group:demo",
        )
        run_task = asyncio.create_task(app.run())  # no inbound at all
        try:
            # Wait for the DURABLE outcome (sent), not the adapter call:
            # the worker may be cancelled between send and mark, which is
            # the at-most-once ambiguity (in_flight, never retried).
            sent_when_due = False
            for _ in range(2000):
                state = await app.repo._db.read(
                    lambda c: c.execute(
                        "SELECT state FROM outbox WHERE id = 1"
                    ).fetchone()[0]
                )
                if state == "sent":
                    sent_when_due = True
                    break
                await asyncio.sleep(0)
        finally:
            stream.release()  # EOF -> run() returns
        await run_task
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        row = await db2.read(
            lambda c: c.execute("SELECT state FROM outbox WHERE id = 1").fetchone()[0]
        )
        await db2.close()
        return row, sent_when_due

    row, sent_when_due = run(scenario())
    assert row == "sent"
    assert sent_when_due is True  # sent when due, with no inbound input


def test_run_cancels_worker_mid_sleep(tmp_path):
    """Shutdown cancels the worker while it sleeps on a future row: clean
    exit, nothing sent, no leaked task."""

    async def scenario():
        clock = VirtualClock(auto_advance=False)
        app = make_app(tmp_path, clock=clock)
        await app.start()
        await app.repo.upsert_chat(make_identity(chat_key="console:group:demo",
                                                 platform="console", self_id="bot"))
        future = clock.now() + 1000.0
        out = Outgoing(chat_key=app.adapter.chat_key, text="paced",
                       idem_key="k1", send_after_ts=future)
        await finish_batch(
            app.repo, app.outbox.to_items(out, "cy-1"),
            chat_key="console:group:demo",
        )
        run_task = asyncio.create_task(app.run())
        # The worker blocks in clock.sleep (auto_advance=False); EOF then
        # shuts the app down, cancelling the worker mid-sleep.
        await asyncio.sleep(0)
        await app.adapter.close()
        await run_task
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        row = await db2.read(
            lambda c: c.execute("SELECT state FROM outbox WHERE id = 1").fetchone()[0]
        )
        await db2.close()
        return row, len(app.adapter.sent)

    row, sent = run(scenario())
    assert row == "pending"  # never sent — the sleep was cancelled
    assert sent == 0


def test_outbox_worker_wakes_from_future_sleep_on_wake(tmp_path):
    """The outbox worker sleeping on a future send_after_ts row is
    INTERRUPTED by a wake event (the CycleRunner's on_outbox after a terminal
    settlement) and promptly drains a newly due row — it does not wait out the
    old future sleep."""

    async def scenario():
        stream = _BlockingStream()
        app = make_app(tmp_path, input_stream=stream)
        await app.start()
        await app.repo.upsert_chat(make_identity(
            chat_key="console:group:demo", platform="console", self_id="bot"))
        future = app.clock.now() + 1000.0
        out = Outgoing(chat_key=app.adapter.chat_key, text="paced",
                       idem_key="k1", send_after_ts=future)
        await finish_batch(
            app.repo, app.outbox.to_items(out, "cy-1"),
            chat_key="console:group:demo",
        )
        run_task = asyncio.create_task(app.run())
        # Let the worker reach its future-row sleep.
        for _ in range(50):
            await asyncio.sleep(0)
        # A terminal settlement creates a NEW due row and wakes the worker.
        out2 = Outgoing(chat_key=app.adapter.chat_key, text="new", idem_key="k2")
        await finish_batch(
            app.repo, app.outbox.to_items(out2, "cy-2"),
            chat_key="console:group:demo",
        )
        app._wake.set()  # the CycleRunner's on_outbox wake
        # The worker drains the new row promptly (before the future sleep).
        sent_new = False
        for _ in range(2000):
            state = await app.repo._db.read(
                lambda c: c.execute(
                    "SELECT state FROM outbox WHERE idem_key = ?", ("cy-2:k2",)
                ).fetchone()
            )
            if state is not None and state[0] == "sent":
                sent_new = True
                break
            await asyncio.sleep(0)
        stream.release()
        await run_task
        await app.shutdown()
        return sent_new

    assert run(scenario())


# ── Phase 2 dry-run: scheduler wiring, zero outbox/send, typed wakes ────────

def test_dry_run_wakes_scheduler_and_runs_cycles(tmp_path):
    """A newly inserted non-self message wakes the scheduler, which runs a
    real claim->snapshot->gate cycle; the trace is emitted per decision
    and shutdown is clean."""

    clock = VirtualClock()
    traces: list[DecisionTrace] = []
    trace_event = asyncio.Event()
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream,
        dry_run=True,
        trace_sink=lambda trace: (traces.append(trace), trace_event.set()),
    )

    async def scenario():
        await app.start()
        await app.adapter.feed("hello there")
        run_task = asyncio.create_task(app.run())
        # The commit wakes the scheduler on the next event-loop turn —
        # BEFORE any EOF.
        stream.release()
        await asyncio.wait_for(trace_event.wait(), timeout=10)
        await run_task
        return True

    assert run(scenario())
    assert len(traces) >= 1
    assert traces[0].chat_key == app.adapter.chat_key
    assert traces[0].decision is not None
    assert traces[0].decision.action == "delay"  # ambient chatter delays


def test_dry_run_zero_outbox_zero_send_with_pending_rows(tmp_path):
    """Dry-run never starts or drains the OutboxDriver and never invokes
    adapter.send — even with pre-existing pending rows."""

    async def scenario():
        app = make_app(tmp_path, dry_run=True)
        await app.start()
        await app.repo.upsert_chat(make_identity(chat_key="console:group:demo",
                                                 platform="console", self_id="bot"))
        await finish_batch(
            app.repo,
            app.outbox.to_items(
                Outgoing(chat_key=app.adapter.chat_key, text="pending",
                         idem_key="k1"),
                "cy-1",
            ),
            chat_key="console:group:demo",
        )
        await app.adapter.feed("hello")
        await app.run()  # dry-run: no drain, no worker, no send
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        states = await db2.read(
            lambda c: [r[0] for r in c.execute("SELECT state FROM outbox ORDER BY id")]
        )
        await db2.close()
        return states, len(app.adapter.sent)

    states, sent = run(scenario())
    assert states == ["pending"]  # untouched
    assert sent == 0  # no send ever


def test_dry_run_typed_echo_wake_rules(tmp_path):
    """Only newly inserted NON-SELF messages wake the scheduler: a
    duplicate never wakes, and a self echo (any echo status) never wakes."""

    clock = VirtualClock()
    calls: list[ChatKey] = []

    async def fake_cycle(chat_key: ChatKey) -> Decision:
        calls.append(chat_key)
        return Decision(action="skip", reason=Reason.SKIP)

    scheduler = Scheduler(clock, fake_cycle)
    app = make_app(tmp_path, clock=clock, dry_run=True, scheduler=scheduler)

    async def scenario():
        await app.start()
        chat = app.adapter.chat_key
        # A newly inserted non-self message wakes (immediate entry).
        await app._maybe_wake(
            AdapterEvent(
                type="message",
                payload=make_message(chat_key="console:group:demo"),
                ts=1.0,
            ),
            IngestResult(row_id=MessageRowId(1), inserted=True),
        )
        assert scheduler.next_wake(chat) is not None
        # A duplicate insert never wakes.
        await app._maybe_wake(
            AdapterEvent(
                type="message",
                payload=make_message(chat_key="console:group:demo"),
                ts=1.0,
            ),
            IngestResult(row_id=MessageRowId(1), inserted=False),
        )
        # A self echo (reconciled) never wakes.
        await app._maybe_wake(
            AdapterEvent(
                type="message",
                payload=make_message(
                    chat_key="console:group:demo", msg_id="echo:1", is_self=True
                ),
                ts=1.0,
            ),
            IngestResult(
                row_id=MessageRowId(2), inserted=True,
                echo_status=EchoStatus.RECONCILED,
            ),
        )
        # An unproven self message never wakes either.
        await app._maybe_wake(
            AdapterEvent(
                type="message",
                payload=make_message(
                    chat_key="console:group:demo", msg_id="echo:2", is_self=True
                ),
                ts=1.0,
            ),
            IngestResult(
                row_id=MessageRowId(3), inserted=True,
                echo_status=EchoStatus.UNPROVEN,
            ),
        )
        await app.shutdown()
        return True

    assert run(scenario())
    assert calls == []  # the scheduler never ran a cycle


def test_dry_run_self_echo_never_runs_a_cycle(tmp_path):
    """End to end: a self message through the run loop never wakes the
    scheduler, so no cycle ever runs for it."""

    clock = VirtualClock()
    calls: list[ChatKey] = []

    async def fake_cycle(chat_key: ChatKey) -> Decision:
        calls.append(chat_key)
        return Decision(action="skip", reason=Reason.SKIP)

    scheduler = Scheduler(clock, fake_cycle)
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream,
        dry_run=True, scheduler=scheduler,
    )

    async def scenario():
        await app.start()
        await app.adapter.feed("self echo", is_self=True)
        run_task = asyncio.create_task(app.run())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        stream.release()
        await run_task
        return True

    assert run(scenario())
    assert calls == []


def test_dry_run_non_self_message_runs_a_cycle(tmp_path):
    """End to end: a non-self message through the run loop wakes the
    scheduler and the injected cycle fn runs exactly once."""

    clock = VirtualClock()
    calls: list[ChatKey] = []
    cycle_event = asyncio.Event()

    async def fake_cycle(chat_key: ChatKey) -> Decision:
        calls.append(chat_key)
        cycle_event.set()
        return Decision(action="skip", reason=Reason.SKIP)

    scheduler = Scheduler(clock, fake_cycle)
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream,
        dry_run=True, scheduler=scheduler,
    )

    async def scenario():
        await app.start()
        await app.adapter.feed("hello")
        run_task = asyncio.create_task(app.run())
        # The commit wakes the scheduler on the next event-loop turn —
        # BEFORE any EOF; the injected cycle fn runs exactly once.
        stream.release()
        await asyncio.wait_for(cycle_event.wait(), timeout=10)
        await run_task
        return True

    assert run(scenario())
    assert calls == [app.adapter.chat_key]

# ── Phase 2 dry-run: startup recovery, priority wake, zero sends ────────────

def test_dry_run_startup_recovery_wakes_pending_chats(tmp_path):
    """A crash/restart with durable pending work re-evaluates it: the
    dry-run App wakes every ``list_pending_chats()`` chat at startup."""
    clock = VirtualClock()
    calls: list[ChatKey] = []
    cycle_event = asyncio.Event()

    async def fake_cycle(chat_key: ChatKey) -> Decision:
        calls.append(chat_key)
        cycle_event.set()
        return Decision(action="skip", reason=Reason.SKIP)

    scheduler = Scheduler(clock, fake_cycle)
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream,
        dry_run=True, scheduler=scheduler,
    )

    async def scenario():
        await app.start()
        # Durable pending work from a previous run (crash before the cycle).
        identity = make_identity(chat_key="console:group:demo",
                                 platform="console", self_id="bot")
        await app.repo.upsert_chat(identity)
        await app.repo.ingest_message(
            identity, make_message(chat_key="console:group:demo", recv_ts=clock.now())
        )
        run_task = asyncio.create_task(app.run())
        await asyncio.wait_for(cycle_event.wait(), timeout=10)
        stream.release()
        await run_task
        return True

    assert run(scenario())
    assert calls == [app.adapter.chat_key]


def test_dry_run_startup_recovery_hold_schedules_remaining_time(tmp_path):
    """A durable active hold at startup schedules only its remaining
    time: the startup wake evaluates, the gate returns the remaining
    backoff delay, and the scheduler re-arms at hold expiry."""
    clock = VirtualClock(auto_advance=False)
    traces: list[DecisionTrace] = []
    trace_event = asyncio.Event()
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream, dry_run=True,
        trace_sink=lambda t: (traces.append(t), trace_event.set()),
    )

    async def scenario():
        await app.start()
        identity = make_identity(chat_key="console:group:demo",
                                 platform="console", self_id="bot")
        await app.repo.upsert_chat(identity)
        now = clock.now()
        await finish_batch(
            app.repo, [], chat_key="console:group:demo",
            started_ts=now, expires_at=now + 60.0,
            hold_until=now + 200.0, now=now,
        )
        await app.repo.ingest_message(
            identity, make_message(chat_key="console:group:demo", recv_ts=clock.now())
        )
        run_task = asyncio.create_task(app.run())
        await asyncio.wait_for(trace_event.wait(), timeout=10)
        assert traces[0].decision is not None
        assert traces[0].decision.action == "delay"
        assert traces[0].decision.reason == Reason.BACKOFF
        assert traces[0].decision.delay_seconds == pytest.approx(200.0)
        # The scheduler re-armed at hold expiry (remaining time only).
        # The re-arm lands after the writer's coalescing window, so poll
        # with real-time sleeps (the app clock is virtual, but the
        # writer's batch window is wall-clock).
        for _ in range(200):
            if app.scheduler.next_wake(app.adapter.chat_key) is not None:
                break
            await asyncio.sleep(0.05)
        assert app.scheduler.next_wake(app.adapter.chat_key) == pytest.approx(
            clock.now() + 200.0
        )
        stream.release()
        await run_task
        return True

    assert run(scenario())


def test_dry_run_priority_wake_for_direct_at(tmp_path):
    """A structurally recognized direct @ takes the priority wake path
    (it may override a scheduled delay); ordinary input never does."""
    clock = VirtualClock(auto_advance=False)
    calls: list[ChatKey] = []
    decisions = iter(
        [
            Decision(action="delay", delay_seconds=300.0),
            Decision(action="skip", reason=Reason.SKIP),
        ]
    )

    async def fake_cycle(chat_key: ChatKey) -> Decision:
        calls.append(chat_key)
        return next(decisions)

    scheduler = Scheduler(clock, fake_cycle)
    app = make_app(tmp_path, clock=clock, dry_run=True, scheduler=scheduler)

    async def scenario():
        await app.start()
        chat = app.adapter.chat_key
        scheduler.start()  # this test drives the scheduler directly
        await scheduler.wake(chat)
        # Wait for the cycle to complete (the re-arm lands synchronously
        # with the lease release).
        for _ in range(1000):
            if len(calls) == 1 and not scheduler.is_leased(chat):
                break
            await asyncio.sleep(0)
        assert scheduler.next_wake(chat) == clock.now() + 300.0
        # Ordinary input during the delay: held (no override).
        await app._maybe_wake(
            AdapterEvent(
                type="message",
                payload=make_message(chat_key="console:group:demo"),
                ts=clock.now(),
            ),
            IngestResult(row_id=MessageRowId(9), inserted=True),
        )
        await asyncio.sleep(0)
        assert scheduler.next_wake(chat) == clock.now() + 300.0  # delay stands
        assert len(calls) == 1
        # Direct @ during the delay: the priority wake overrides it.
        await app._maybe_wake(
            AdapterEvent(
                type="message",
                payload=make_message(
                    chat_key="console:group:demo", mentions=("bot",)
                ),
                ts=clock.now(),
            ),
            IngestResult(row_id=MessageRowId(10), inserted=True),
        )
        for _ in range(1000):
            if len(calls) == 2:
                break
            await asyncio.sleep(0)
        assert len(calls) == 2  # re-evaluated immediately
        await scheduler.stop()
        await app.shutdown()
        return True

    assert run(scenario())


def test_dry_run_trigger_cycle_never_sends(tmp_path):
    """A hard-trigger cycle in dry-run finishes with an EMPTY outbox and
    never invokes the adapter — the zero-send guarantee holds end to end."""
    clock = VirtualClock()
    traces: list[DecisionTrace] = []
    trace_event = asyncio.Event()
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream, dry_run=True,
        trace_sink=lambda t: (traces.append(t), trace_event.set()),
    )

    async def scenario():
        await app.start()
        identity = make_identity(chat_key="console:group:demo",
                                 platform="console", self_id="bot")
        await app.repo.upsert_chat(identity)
        msg = Message(
            chat_key=app.adapter.chat_key,
            sender_id=SenderId("user"),
            sender_name="user",
            is_self=False,
            text="hi",
            id=MessageId("m1"),
            mentions=(SenderId("bot"),),
            recv_ts=clock.now(),
        )
        await app.repo.ingest_message(identity, msg)
        run_task = asyncio.create_task(app.run())  # startup recovery wakes it
        await asyncio.wait_for(trace_event.wait(), timeout=10)
        assert traces[0].decision is not None
        assert traces[0].decision.action == "trigger"
        stream.release()
        await run_task
        return True

    assert run(scenario())
    assert len(app.adapter.sent) == 0  # zero adapter sends


# ── Phase 2 dry-run: immediate commit, next-turn wake dispatcher ─────────────

async def _wait_traces(traces: list, n: int) -> None:
    """Yield until ``n`` traces were produced. The writer's coalescing
    window is wall-clock, so include real-time sleeps (the app clock is
    virtual) — pure event-loop yielding can starve it."""
    for _ in range(2000):
        if len(traces) >= n:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"only {len(traces)} traces after 2000 polls")


async def _drain_until(predicate) -> None:
    """Yield with real-time sleeps until ``predicate`` holds (the writer's
    coalescing window is wall-clock)."""
    for _ in range(2000):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


def _assert_traces_match(live: DecisionTrace, replay: DecisionTrace) -> None:
    """The live cycle and the replay produce the SAME decision, aggregates,
    backoff facts, config, and snapshot facts (cycle ids differ by
    construction)."""
    assert live.decision == replay.decision
    assert live.mode == replay.mode
    assert live.threshold == replay.threshold
    assert live.trigger_score == replay.trigger_score
    assert live.pending == replay.pending
    assert live.aggregates == replay.aggregates
    assert live.backoff == replay.backoff
    assert live.config == replay.config
    lf = {k: v for k, v in live.snapshot_facts.items() if k != "cycle_id"}
    rf = {k: v for k, v in replay.snapshot_facts.items() if k != "cycle_id"}
    assert lf == rf


class _FakeIngest:
    """A non-yielding ingest fake: every ``handle`` returns a committed
    ``IngestResult`` immediately, so several ``_ingest_batched`` calls stay
    in the SAME event-loop turn and coalesce into one next-turn flush."""

    def __init__(self) -> None:
        self.n = 0
        self.events: list[AdapterEvent] = []

    async def handle(self, event: AdapterEvent) -> IngestResult:
        self.events.append(event)
        self.n += 1
        return IngestResult(row_id=MessageRowId(self.n), inserted=True)


def test_dry_run_single_message_commits_and_wakes_without_eof(tmp_path):
    """A quiet single message is recorder+DB committed and scheduler-woken
    on the next event-loop turn — WITHOUT any EOF."""
    clock = VirtualClock()
    calls: list[ChatKey] = []
    cycle_event = asyncio.Event()

    async def fake_cycle(chat_key: ChatKey) -> Decision:
        calls.append(chat_key)
        cycle_event.set()
        return Decision(action="skip", reason=Reason.SKIP)

    scheduler = Scheduler(clock, fake_cycle)
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream,
        dry_run=True, scheduler=scheduler,
    )

    async def scenario():
        await app.start()
        await app.adapter.feed("hello")
        run_task = asyncio.create_task(app.run())
        # The wake fires on the next event-loop turn after the commit —
        # the stream is still open (no EOF).
        await asyncio.wait_for(cycle_event.wait(), timeout=10)
        # Recorder + database durability, before any EOF.
        events = read_corpus(tmp_path / "data" / "app.jsonl")
        assert any(
            e.type == "message" and e.payload.text == "hello" for e in events
        )
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        repo2 = SqliteRepository(db2)
        msg = await repo2.get_message(app.adapter.chat_key, "console:in:1")
        await db2.close()
        assert msg is not None and msg.text == "hello"
        stream.release()
        await run_task
        return True

    assert run(scenario())
    assert calls == [app.adapter.chat_key]


def test_dry_run_non_message_event_records_immediately(tmp_path):
    """A non-message event is recorded immediately through Ingest (no
    early return) and never wakes the scheduler."""
    clock = VirtualClock()
    calls: list[ChatKey] = []

    async def fake_cycle(chat_key: ChatKey) -> Decision:
        calls.append(chat_key)
        return Decision(action="skip", reason=Reason.SKIP)

    scheduler = Scheduler(clock, fake_cycle)
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream,
        dry_run=True, scheduler=scheduler,
    )

    async def scenario():
        await app.start()
        await app.adapter._events.put(
            AdapterEvent(type="notice", payload={"kind": "poke"}, ts=clock.now())
        )
        run_task = asyncio.create_task(app.run())
        # Recorded immediately (recorder), no wake, no cycle.
        await _drain_until(
            lambda: any(
                e.type == "notice"
                for e in read_corpus(tmp_path / "data" / "app.jsonl")
            )
        )
        stream.release()
        await run_task
        return True

    assert run(scenario())
    assert calls == []


def test_dry_run_immediate_commits_coalesce_into_one_wake(tmp_path):
    """Multiple immediate committed events (same event-loop turn) coalesce
    into ONE scheduler wake: the next-turn flush sends one wake per chat,
    then clears the metadata."""
    clock = VirtualClock()
    calls: list[ChatKey] = []

    async def fake_cycle(chat_key: ChatKey) -> Decision:
        calls.append(chat_key)
        return Decision(action="skip", reason=Reason.SKIP)

    scheduler = Scheduler(clock, fake_cycle)
    app = make_app(tmp_path, clock=clock, dry_run=True, scheduler=scheduler)
    app.ingest = _FakeIngest()

    async def scenario():
        await app.start()
        scheduler.start()
        ts = clock.now()
        for i in range(3):
            await app._ingest_batched(
                AdapterEvent(
                    type="message",
                    payload=make_message(
                        chat_key="console:group:demo", msg_id=f"m{i}", recv_ts=ts
                    ),
                    ts=ts,
                )
            )
        # No yield yet: all three commits are in the same turn, coalesced
        # into the single scheduled flush.
        assert app._wake_meta == {app.adapter.chat_key: "ordinary"}
        await asyncio.sleep(0)  # let the one flush run
        assert app._wake_meta == {}  # the flush cleared the metadata
        await _drain_until(lambda: len(calls) == 1)  # the wake is served
        assert len(calls) == 1  # one wake for the burst
        await scheduler.stop()
        await app.shutdown()
        return True

    assert run(scenario())


def test_dry_run_priority_or_coalesces_to_priority(tmp_path):
    """A same-turn flush with an ordinary AND a priority member sends the
    PRIORITY wake (priority OR): it overrides a scheduled hold/delay."""
    clock = VirtualClock(auto_advance=False)
    calls: list[ChatKey] = []
    decisions = iter(
        [
            Decision(action="delay", delay_seconds=300.0),
            Decision(action="skip", reason=Reason.SKIP),
        ]
    )

    async def fake_cycle(chat_key: ChatKey) -> Decision:
        calls.append(chat_key)
        return next(decisions)

    scheduler = Scheduler(clock, fake_cycle)
    app = make_app(tmp_path, clock=clock, dry_run=True, scheduler=scheduler)
    app.ingest = _FakeIngest()

    async def scenario():
        await app.start()
        chat = app.adapter.chat_key
        scheduler.start()
        await scheduler.wake(chat)
        for _ in range(1000):
            if len(calls) == 1 and not scheduler.is_leased(chat):
                break
            await asyncio.sleep(0)
        assert scheduler.next_wake(chat) == clock.now() + 300.0
        # Ordinary + direct @ in the same turn: the flush sends
        # wake_priority, which overrides the scheduled delay.
        ts = clock.now()
        await app._ingest_batched(
            AdapterEvent(
                type="message",
                payload=make_message(
                    chat_key="console:group:demo", msg_id="m1", recv_ts=ts
                ),
                ts=ts,
            )
        )
        await app._ingest_batched(
            AdapterEvent(
                type="message",
                payload=make_message(
                    chat_key="console:group:demo", msg_id="m2",
                    mentions=("bot",), recv_ts=ts,
                ),
                ts=ts,
            )
        )
        assert app._wake_meta == {chat: "priority"}  # priority OR
        await asyncio.sleep(0)
        for _ in range(1000):
            if len(calls) == 2:
                break
            await asyncio.sleep(0)
        assert len(calls) == 2  # re-evaluated immediately
        await scheduler.stop()
        await app.shutdown()
        return True

    assert run(scenario())


def test_dry_run_shutdown_flushes_pending_wake_metadata(tmp_path):
    """Shutdown awaits/cancels-and-flushes any scheduled wake task so
    committed wake metadata is not lost."""
    clock = VirtualClock()
    calls: list[ChatKey] = []

    async def fake_cycle(chat_key: ChatKey) -> Decision:
        calls.append(chat_key)
        return Decision(action="skip", reason=Reason.SKIP)

    scheduler = Scheduler(clock, fake_cycle)
    app = make_app(tmp_path, clock=clock, dry_run=True, scheduler=scheduler)
    app.ingest = _FakeIngest()

    async def scenario():
        await app.start()
        scheduler.start()
        app._scheduler_started = True  # shutdown will drain/stop it
        ts = clock.now()
        await app._ingest_batched(
            AdapterEvent(
                type="message",
                payload=make_message(
                    chat_key="console:group:demo", msg_id="m1", recv_ts=ts
                ),
                ts=ts,
            )
        )
        # The flush is scheduled but has NOT run yet (no yield since the
        # commit): shutdown must flush the pending wake metadata.
        assert app._wake_meta == {app.adapter.chat_key: "ordinary"}
        await app.shutdown()
        assert len(calls) == 1  # the wake was flushed and served
        return True

    assert run(scenario())


def test_dry_run_cross_chat_wakes_are_independent(tmp_path):
    """Per-chat wake metadata is independent: two chats each get their own
    wake and their own cycle."""
    clock = VirtualClock(auto_advance=False)
    calls: list[ChatKey] = []

    async def fake_cycle(chat_key: ChatKey) -> Decision:
        calls.append(chat_key)
        return Decision(action="skip", reason=Reason.SKIP)

    scheduler = Scheduler(clock, fake_cycle)
    app = make_app(tmp_path, clock=clock, dry_run=True, scheduler=scheduler)
    other_key = ChatKey("console:group:other")
    identities = {
        app.adapter.chat_key: app.adapter.identity,
        other_key: ChatIdentity(
            other_key, PlatformId("console"), SelfId("bot"), "group"
        ),
    }
    app.ingest = Ingest(
        app.repo, app.recorder, wake=None,
        identity=lambda ck: identities.get(ck), clock=clock,
    )

    async def scenario():
        await app.start()
        scheduler.start()
        ts = clock.now()
        msg_a = make_message(chat_key=str(app.adapter.chat_key), msg_id="a1",
                             recv_ts=ts)
        msg_b = make_message(chat_key=str(other_key), msg_id="b1", recv_ts=ts)
        await app._ingest_batched(AdapterEvent(type="message", payload=msg_a, ts=ts))
        await app._ingest_batched(AdapterEvent(type="message", payload=msg_b, ts=ts))
        await _drain_until(lambda: len(calls) == 2)
        assert sorted(calls) == sorted([app.adapter.chat_key, other_key])
        await scheduler.stop()
        await app.shutdown()
        return True

    assert run(scenario())


def test_dry_run_batched_self_and_duplicate_never_wake(tmp_path):
    """Through the run loop: a self echo and a duplicate never wake the
    scheduler — no cycle runs for them."""
    clock = VirtualClock(auto_advance=False)
    calls: list[ChatKey] = []

    async def fake_cycle(chat_key: ChatKey) -> Decision:
        calls.append(chat_key)
        return Decision(action="skip", reason=Reason.SKIP)

    scheduler = Scheduler(clock, fake_cycle)
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream, dry_run=True,
        scheduler=scheduler,
    )

    async def scenario():
        await app.start()
        identity = make_identity(chat_key="console:group:demo",
                                 platform="console", self_id="bot")
        await app.repo.upsert_chat(identity)
        # A previously committed message makes the next feed a duplicate;
        # a terminal finish consumes it so startup recovery stays quiet.
        now = clock.now()
        await app.repo.ingest_message(
            identity,
            make_message(chat_key="console:group:demo", msg_id="dup:1",
                         recv_ts=now),
        )
        await finish_batch(
            app.repo, [], chat_key="console:group:demo",
            started_ts=now, expires_at=now + 60.0, now=now,
        )
        await app.adapter.feed("self echo", is_self=True)
        await app.adapter.feed("dup", msg_id="dup:1")
        run_task = asyncio.create_task(app.run())
        stream.release()  # self + duplicate never wake
        await run_task
        return True

    assert run(scenario())
    assert calls == []


def test_dry_run_high_pending_priority_overrides_scheduled_hold(tmp_path):
    """A message that brings the atomic pending count to/above the gate
    threshold takes the PRIORITY wake path: it overrides an already
    scheduled hold/delay and re-evaluates immediately; the gate applies
    the high-pending bypass. (A hold delay releases the claim WITHOUT
    advancing the cursor, so the second message's pending count is 2.)"""
    clock = VirtualClock(auto_advance=False)
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "data" / "app.db")},
         "gate": {"threshold": 2}}
    )
    traces: list[DecisionTrace] = []
    stream = _BlockingStream()
    app = App.build(
        cfg,
        clock=clock,
        adapter=ConsoleAdapter(clock=clock, input_stream=stream,
                               output_stream=io.StringIO()),
        dry_run=True,
        trace_sink=traces.append,
    )

    async def scenario():
        await app.start()
        identity = make_identity(chat_key="console:group:demo",
                                 platform="console", self_id="bot")
        await app.repo.upsert_chat(identity)
        t0 = clock.now()
        await finish_batch(
            app.repo, [], chat_key="console:group:demo",
            started_ts=t0, expires_at=t0 + 60.0, hold_until=t0 + 200.0, now=t0,
        )
        run_task = asyncio.create_task(app.run())
        await app.adapter.feed("m1")  # ts = t0: pending 1 < 2 -> ordinary
        await _wait_traces(traces, 1)  # cycle 1: the hold delay
        # Wait for the re-arm to land BEFORE advancing the clock: the
        # scheduler measures the delay from re-arm time, so a clock
        # advance mid-release would drift the horizon.
        await _drain_until(
            lambda: app.scheduler.next_wake(app.adapter.chat_key) is not None
        )
        assert app.scheduler.next_wake(app.adapter.chat_key) == pytest.approx(
            t0 + 200.0
        )
        clock.advance(10.0)
        await app.adapter.feed("m2")  # ts = t0+10: pending 2 >= 2 -> priority
        await _wait_traces(traces, 2)  # cycle 2 at t0+10: the priority override
        assert traces[1].snapshot_facts["evaluated_ts"] == pytest.approx(t0 + 10.0)
        assert traces[1].backoff is not None
        assert traces[1].backoff.bypass_reason == "high_pending"
        stream.release()
        await run_task
        return t0

    t0 = run(scenario())
    assert len(traces) == 2
    assert traces[0].snapshot_facts["evaluated_ts"] == pytest.approx(t0)
    assert traces[0].decision is not None
    assert traces[0].decision.action == "delay"
    assert traces[0].decision.delay_seconds == pytest.approx(200.0)  # hold
    assert traces[0].backoff is not None and traces[0].backoff.applied


def test_dry_run_ordinary_message_never_overrides_scheduled_hold(tmp_path):
    """An ordinary message (pending below the threshold) during a
    scheduled hold/delay never overrides it: the re-armed wake fires at
    the hold horizon, not at the arrival. (Legacy wake path: an explicitly
    injected generic Scheduler — the default dry-run is ledger-only.)"""
    clock = VirtualClock(auto_advance=False)
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "data" / "app.db")}}
    )
    traces: list[DecisionTrace] = []
    stream = _BlockingStream()
    app = App.build(
        cfg,
        clock=clock,
        adapter=ConsoleAdapter(clock=clock, input_stream=stream,
                               output_stream=io.StringIO()),
        dry_run=True,
        trace_sink=traces.append,
    )
    # Preserve the legacy next-turn-flush/priority arbitration: an
    # explicitly injected generic Scheduler over the same CycleRunner.
    app.scheduler = Scheduler(clock, app._cycle_fn)

    async def scenario():
        await app.start()
        identity = make_identity(chat_key="console:group:demo",
                                 platform="console", self_id="bot")
        await app.repo.upsert_chat(identity)
        t0 = clock.now()
        await finish_batch(
            app.repo, [], chat_key="console:group:demo",
            started_ts=t0, expires_at=t0 + 60.0, hold_until=t0 + 200.0, now=t0,
        )
        run_task = asyncio.create_task(app.run())
        await app.adapter.feed("m1")  # ts = t0: ordinary -> cycle 1
        await _wait_traces(traces, 1)  # cycle 1: the hold delay
        # Wait for the re-arm to land BEFORE advancing the clock (the
        # scheduler measures the delay from re-arm time).
        await _drain_until(
            lambda: app.scheduler.next_wake(app.adapter.chat_key) is not None
        )
        assert app.scheduler.next_wake(app.adapter.chat_key) == pytest.approx(
            t0 + 200.0
        )
        clock.advance(10.0)
        await app.adapter.feed("m2")  # ts = t0+10: ordinary (1 < 8)
        # The ingest path commits AND exports the commit marker (two
        # writer round-trips, each up to the coalescing window), so the
        # wake flush lands a few windows later — give it real time.
        await asyncio.sleep(0.3)
        assert len(traces) == 1  # no evaluation at t0+10: the delay stands
        clock.advance(190.0)  # t0+200: the hold horizon
        await _wait_traces(traces, 2)  # the re-armed wake fires
        assert traces[1].snapshot_facts["evaluated_ts"] == pytest.approx(t0 + 200.0)
        stream.release()
        await run_task
        return t0

    t0 = run(scenario())
    assert len(traces) == 2
    assert traces[0].snapshot_facts["evaluated_ts"] == pytest.approx(t0)
    assert traces[0].decision is not None
    assert traces[0].decision.delay_seconds == pytest.approx(200.0)


def test_app_replay_parity_single_message(tmp_path):
    """The App's immediate-commit ingestion produces the SAME trace as
    replay of the recorded corpus for a single message."""
    clock = VirtualClock(auto_advance=False)
    cfg = make_config(tmp_path)
    traces: list[DecisionTrace] = []
    trace_event = asyncio.Event()
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream, dry_run=True,
        trace_sink=lambda t: (traces.append(t), trace_event.set()),
    )

    async def scenario():
        await app.start()
        await app.adapter.feed("hello")
        run_task = asyncio.create_task(app.run())
        await asyncio.wait_for(trace_event.wait(), timeout=10)
        stream.release()
        await run_task
        return True

    assert run(scenario())
    assert len(traces) == 1
    events = read_corpus(tmp_path / "data" / "app.jsonl")
    replay = replay_corpus(
        events, chat_key=app.adapter.chat_key,
        identity=app.adapter.identity, cfg=cfg,
    )
    assert replay.decisions == 1
    _assert_traces_match(traces[0], replay.traces[0])


def test_dry_run_batched_burst_zero_send_with_pending_rows(tmp_path):
    """A same-ts burst through the run loop never sends and never touches
    the outbox — even with pre-existing pending rows."""
    async def scenario():
        app = make_app(tmp_path, dry_run=True)
        await app.start()
        await app.repo.upsert_chat(make_identity(chat_key="console:group:demo",
                                                 platform="console", self_id="bot"))
        await finish_batch(
            app.repo,
            app.outbox.to_items(
                Outgoing(chat_key=app.adapter.chat_key, text="pending",
                         idem_key="k1"),
                "cy-1",
            ),
            chat_key="console:group:demo",
        )
        for i in range(3):
            await app.adapter.feed(f"m{i}")  # identical recv_ts
        await app.run()  # dry-run: no drain, no worker, no send
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        states = await db2.read(
            lambda c: [r[0] for r in c.execute("SELECT state FROM outbox ORDER BY id")]
        )
        await db2.close()
        return states, len(app.adapter.sent)

    states, sent = run(scenario())
    assert states == ["pending"]  # untouched
    assert sent == 0  # no send ever


def test_dry_run_busy_claim_rearms_and_recovers_after_restart(tmp_path):
    """A crash mid-cycle leaves a live durable claim; on restart the
    startup recovery wake maps it to a timed delay at the busy horizon,
    the scheduler re-arms, and at the horizon the expired claim is
    recovered and evaluated — WITHOUT new input. (Legacy claim_cycle busy
    path: an explicitly injected generic Scheduler — the default dry-run
    is ledger-only.)"""
    clock = VirtualClock(auto_advance=False)

    async def scenario():
        # Run 1: a cycle crashes mid-flight — a live claim + pending msg.
        db, repo = await open_repo(tmp_path / "data" / "app.db")
        identity = make_identity(chat_key="console:group:demo",
                                 platform="console", self_id="bot")
        await repo.upsert_chat(identity)
        await repo.ingest_message(
            identity,
            make_message(chat_key="console:group:demo", recv_ts=clock.now()),
        )
        grant = await repo.claim_cycle(
            make_claim(chat_key="console:group:demo", cycle_id="crash-1",
                       started_ts=clock.now(), expires_at=clock.now() + 60.0)
        )
        assert grant is not None
        await repo.close()  # "crash": the claim stays live
        # Run 2: restart — startup recovery wakes the chat.
        traces: list[DecisionTrace] = []
        trace_event = asyncio.Event()
        stream = _BlockingStream()
        app = make_app(
            tmp_path, clock=clock, input_stream=stream, dry_run=True,
            trace_sink=lambda t: (traces.append(t), trace_event.set()),
        )
        # Preserve the legacy claim_cycle busy path: an explicitly injected
        # generic Scheduler over the same CycleRunner.
        app.scheduler = Scheduler(clock, app._cycle_fn)
        await app.start()
        run_task = asyncio.create_task(app.run())
        # The startup wake -> ClaimBusy -> timed delay at the busy horizon.
        await _drain_until(
            lambda: app.scheduler.next_wake(app.adapter.chat_key) is not None
        )
        assert app.scheduler.next_wake(app.adapter.chat_key) == pytest.approx(
            clock.now() + 60.0
        )
        assert traces == []  # the busy path emits no false trace
        clock.advance(60.0)  # the lease expires: the claim is recovered
        await asyncio.wait_for(trace_event.wait(), timeout=10)
        assert traces[0].decision is not None
        stream.release()
        await run_task
        return True

    assert run(scenario())


# ── Phase 2 dry-run: durable dispatch-ledger integration ─────────────────────

def test_dry_run_default_uses_ledger_scheduler(tmp_path):
    """The default production dry-run App wires the LedgerScheduler over
    CycleRunner.run_dispatch — not the legacy generic Scheduler — and a
    committed event produces a durable dispatch (begin_dispatch, never the
    legacy claim_cycle surface)."""
    clock = VirtualClock()
    traces: list[DecisionTrace] = []
    trace_event = asyncio.Event()
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream, dry_run=True,
        trace_sink=lambda t: (traces.append(t), trace_event.set()),
    )
    assert isinstance(app.scheduler, LedgerScheduler)
    assert app.scheduler._handler.__func__ is CycleRunner.run_dispatch

    async def scenario():
        await app.start()
        await app.adapter.feed("hello")
        run_task = asyncio.create_task(app.run())
        await asyncio.wait_for(trace_event.wait(), timeout=10)
        stream.release()
        await run_task
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        row = await db2.read(
            lambda c: c.execute(
                "SELECT state, commit_boundary FROM dispatches"
            ).fetchone()
        )
        await db2.close()
        return row

    row = run(scenario())
    assert row is not None  # a durable dispatch row was created
    state, boundary = row
    assert state in ("completed", "released")
    assert boundary == 1
    assert len(traces) == 1


def test_dry_run_committed_event_yields_dispatch_marker_and_trace(tmp_path):
    """Each committed inbound event yields a durable dispatch (a dispatches
    row), a commit marker AND a dispatch marker in the corpus, and a
    decision trace."""
    clock = VirtualClock()
    traces: list[DecisionTrace] = []
    trace_event = asyncio.Event()
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream, dry_run=True,
        trace_sink=lambda t: (traces.append(t), trace_event.set()),
    )

    async def scenario():
        await app.start()
        await app.adapter.feed("hello")
        run_task = asyncio.create_task(app.run())
        await asyncio.wait_for(trace_event.wait(), timeout=10)
        stream.release()
        await run_task
        markers = read_markers(tmp_path / "data" / "app.jsonl")
        return markers

    markers = run(scenario())
    assert len(traces) == 1
    assert any(
        m.record_type == "commit" and m.sequence == 1 for m in markers
    )
    dispatch_markers = [m for m in markers if m.record_type == "dispatch"]
    assert len(dispatch_markers) == 1
    assert dispatch_markers[0].sequence == 1
    assert dispatch_markers[0].commit_boundary == CommitSeq(1)


def test_dry_run_immediate_quiet_commit_notification(tmp_path):
    """A quiet single message is committed durably and notified to the
    ledger scheduler immediately — WITHOUT any EOF (the ledger route has no
    next-turn flush)."""
    clock = VirtualClock()
    traces: list[DecisionTrace] = []
    trace_event = asyncio.Event()
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream, dry_run=True,
        trace_sink=lambda t: (traces.append(t), trace_event.set()),
    )

    async def scenario():
        await app.start()
        await app.adapter.feed("hello")
        run_task = asyncio.create_task(app.run())
        await asyncio.wait_for(trace_event.wait(), timeout=10)
        # Recorder + database durability, before any EOF.
        events = read_corpus(tmp_path / "data" / "app.jsonl")
        assert any(
            e.type == "message" and e.payload.text == "hello" for e in events
        )
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        repo2 = SqliteRepository(db2)
        msg = await repo2.get_message(app.adapter.chat_key, "console:in:1")
        await db2.close()
        assert msg is not None and msg.text == "hello"
        stream.release()
        await run_task
        return True

    assert run(scenario())
    assert len(traces) == 1


def test_dry_run_timer_inbound_writer_order_boundaries(tmp_path):
    """A timer re-arm (from a durable hold) followed by an inbound commit:
    the durable writer order resolves the boundary — the later inbound
    dispatch attaches every unassigned commit written before it, and the
    dispatch markers carry the exact frozen boundaries."""
    clock = VirtualClock(auto_advance=False)
    traces: list[DecisionTrace] = []
    trace_event = asyncio.Event()
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream, dry_run=True,
        trace_sink=lambda t: (traces.append(t), trace_event.set()),
    )

    async def scenario():
        await app.start()
        identity = make_identity(chat_key="console:group:demo",
                                 platform="console", self_id="bot")
        await app.repo.upsert_chat(identity)
        t0 = clock.now()
        # A durable active hold makes the first dispatch's delay TIMED, so
        # the scheduler re-arms a timer wake at the hold horizon.
        await finish_batch(
            app.repo, [], chat_key="console:group:demo",
            started_ts=t0, expires_at=t0 + 60.0, hold_until=t0 + 200.0, now=t0,
        )
        await app.adapter.feed("m1")
        run_task = asyncio.create_task(app.run())
        await asyncio.wait_for(trace_event.wait(), timeout=10)
        # The hold delay re-arms a timer wake at the hold horizon.
        await _drain_until(
            lambda: app.scheduler.next_wake(app.adapter.chat_key) is not None
        )
        assert app.scheduler.next_wake(app.adapter.chat_key) == pytest.approx(
            t0 + 200.0
        )
        clock.advance(10.0)
        await app.adapter.feed("m2")
        # Ordinary work remains behind the durable hold/timer. Only a
        # direct/quote/high-pending commit may supersede this deadline.
        await asyncio.sleep(0)
        assert len(traces) == 1
        clock.advance(190.0)
        await _wait_traces(traces, 2)
        stream.release()
        await run_task
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        attached_json = await db2.read(
            lambda c: c.execute(
                "SELECT attached_json FROM dispatches WHERE id = 2"
            ).fetchone()[0]
        )
        await db2.close()
        return attached_json

    attached_json = run(scenario())
    markers = read_markers(tmp_path / "data" / "app.jsonl")
    dispatch_markers = [m for m in markers if m.record_type == "dispatch"]
    assert len(dispatch_markers) == 2, [
        (marker.sequence, marker.state, marker.attached, marker.trace_json)
        for marker in dispatch_markers
    ]
    assert dispatch_markers[0].commit_boundary == CommitSeq(1)
    assert dispatch_markers[1].commit_boundary == CommitSeq(2)
    # The durable writer order resolved the attachment: the later inbound
    # dispatch attached every unassigned commit written before it.
    assert attached_json == "[1,2]"


def test_dry_run_startup_unassigned_commit_recovery_and_marker_repair(tmp_path):
    """A crash after the commit but before any dispatch/export: on restart
    the startup export repairs the missing commit marker and the ledger
    recovery resumes the unassigned commit — a dispatch is created and a
    trace is produced without new input."""
    clock = VirtualClock()

    async def seed():
        db, repo = await open_repo(tmp_path / "data" / "app.db")
        identity = make_identity(chat_key="console:group:demo",
                                 platform="console", self_id="bot")
        await repo.upsert_chat(identity)
        await repo.ingest_message(
            identity,
            make_message(chat_key="console:group:demo", recv_ts=clock.now()),
        )
        await repo.close()

    run(seed())
    traces: list[DecisionTrace] = []
    trace_event = asyncio.Event()
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream, dry_run=True,
        trace_sink=lambda t: (traces.append(t), trace_event.set()),
    )

    async def scenario():
        await app.start()
        run_task = asyncio.create_task(app.run())
        await asyncio.wait_for(trace_event.wait(), timeout=10)
        stream.release()
        await run_task
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        dispatch_count = await db2.read(
            lambda c: c.execute("SELECT COUNT(*) FROM dispatches").fetchone()[0]
        )
        await db2.close()
        markers = read_markers(tmp_path / "data" / "app.jsonl")
        return dispatch_count, markers

    dispatch_count, markers = run(scenario())
    assert len(traces) == 1
    assert dispatch_count == 1  # the recovered commit was dispatched
    # The startup export repaired the missing commit marker.
    commit_markers = [m for m in markers if m.record_type == "commit"]
    assert len(commit_markers) == 1
    assert commit_markers[0].sequence == 1


def test_dry_run_prepared_crash_recovery(tmp_path):
    """A crash mid-dispatch leaves a live prepared dispatch; on restart the
    ledger recovery wakes the chat (a later unassigned commit), begin_dispatch
    reports ClaimBusy, the scheduler re-arms at the busy horizon, and at
    expiry the prepared dispatch is recovered and evaluated — without new
    input."""
    clock = VirtualClock(auto_advance=False)

    async def seed():
        db, repo = await open_repo(tmp_path / "data" / "app.db")
        identity = make_identity(chat_key="console:group:demo",
                                 platform="console", self_id="bot")
        await repo.upsert_chat(identity)
        await repo.ingest_message(
            identity,
            make_message(chat_key="console:group:demo", msg_id="m1",
                         recv_ts=clock.now()),
        )
        grant = await repo.begin_dispatch(
            DispatchRequest(
                chat_key=ChatKey("console:group:demo"),
                cause=DispatchCause.INBOUND,
                cycle_id=CycleId("crash-1"),
                started_ts=clock.now(),
                expires_at=clock.now() + 60.0,
                now=clock.now(),
            )
        )
        assert isinstance(grant, DispatchGrant)
        # A second commit lands after the dispatch froze its boundary: it
        # stays unassigned and makes the chat visible to startup recovery.
        await repo.ingest_message(
            identity,
            make_message(chat_key="console:group:demo", msg_id="m2",
                         recv_ts=clock.now()),
        )
        await repo.close()

    run(seed())
    traces: list[DecisionTrace] = []
    trace_event = asyncio.Event()
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream, dry_run=True,
        trace_sink=lambda t: (traces.append(t), trace_event.set()),
    )

    async def scenario():
        await app.start()
        run_task = asyncio.create_task(app.run())
        # Startup recovery wakes the chat (m2 unassigned); the live prepared
        # dispatch reports ClaimBusy -> busy-horizon re-arm, no false trace.
        await _drain_until(
            lambda: app.scheduler.next_wake(app.adapter.chat_key) is not None
        )
        assert traces == []
        clock.advance(60.0)  # the prepared dispatch's lease expires
        await asyncio.wait_for(trace_event.wait(), timeout=10)
        stream.release()
        await run_task
        return True

    assert run(scenario())
    assert len(traces) == 1


def test_dry_run_duplicates_and_self_no_dispatch(tmp_path):
    """A self echo and a duplicate never create a dispatch: the ledger
    attaches nothing for them (a self echo commits with wake_kind none; a
    duplicate commits no row)."""
    clock = VirtualClock()
    app = make_app(tmp_path, clock=clock, dry_run=True)

    async def scenario():
        await app.start()
        identity = make_identity(chat_key="console:group:demo",
                                 platform="console", self_id="bot")
        await app.repo.upsert_chat(identity)
        now = clock.now()
        # Pre-commit the message the duplicate feed will collide with, and
        # settle it terminally so it is not pending at startup.
        result = await app.repo.ingest_message(
            identity,
            make_message(chat_key="console:group:demo", msg_id="dup:1",
                         recv_ts=now),
        )
        assert result.commit_seq is not None
        grant = await app.repo.begin_dispatch(
            DispatchRequest(
                chat_key=ChatKey("console:group:demo"),
                cause=DispatchCause.INBOUND,
                cycle_id=CycleId("seed-1"),
                started_ts=now,
                expires_at=now + 60.0,
                now=now,
            )
        )
        assert isinstance(grant, DispatchGrant)
        await app.repo.settle_dispatch(
            DispatchSettle(
                chat_key=ChatKey("console:group:demo"),
                dispatch_id=grant.dispatch_id,
                cycle_id=CycleId("seed-1"),
                outcome="finish",
                end_reason="skip",
                trace_json='{"t":1}',
            ),
            [],
            now=now,
        )
        # Start the ledger scheduler so the self echo's notification is
        # actually processed (and finds no eligible work).
        app.scheduler.start()
        app._scheduler_started = True
        # A self echo commits (wake_kind none) but never dispatches.
        await app._ingest_batched(
            AdapterEvent(
                type="message",
                payload=make_message(chat_key="console:group:demo",
                                     msg_id="echo:1", is_self=True, recv_ts=now),
                ts=now,
            )
        )
        # A duplicate commits nothing and never dispatches.
        await app._ingest_batched(
            AdapterEvent(
                type="message",
                payload=make_message(chat_key="console:group:demo",
                                     msg_id="dup:1", recv_ts=now),
                ts=now,
            )
        )
        # Give the scheduler a chance to (not) dispatch.
        await asyncio.sleep(0.05)
        await app.shutdown()
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        dispatch_count = await db2.read(
            lambda c: c.execute("SELECT COUNT(*) FROM dispatches").fetchone()[0]
        )
        await db2.close()
        return dispatch_count

    dispatch_count = run(scenario())
    assert dispatch_count == 1  # only the seed dispatch


def test_dry_run_self_commit_cannot_wake_existing_pending_ledger_work(tmp_path):
    """A self echo has a durable commit marker for presence/replay, but it
    must never notify the ledger or evaluate unrelated unassigned work."""
    clock = VirtualClock()
    app = make_app(tmp_path, clock=clock, dry_run=True)

    async def scenario():
        await app.start()
        identity = make_identity(
            chat_key="console:group:demo", platform="console", self_id="bot"
        )
        await app.repo.upsert_chat(identity)
        await app.repo.ingest_message(
            identity,
            make_message(chat_key="console:group:demo", msg_id="pending", recv_ts=clock.now()),
        )
        app.scheduler.start()
        await app._ingest_batched(
            AdapterEvent(
                type="message",
                payload=make_message(
                    chat_key="console:group:demo",
                    msg_id="self",
                    is_self=True,
                    sender_id="bot",
                    recv_ts=clock.now(),
                ),
                ts=clock.now(),
            )
        )
        for _ in range(10):
            await asyncio.sleep(0)
        count = await app.db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM dispatches").fetchone()[0]
        )
        await app.shutdown()
        return count

    assert run(scenario()) == 0


def test_dry_run_ledger_zero_sends_with_pending_outbox(tmp_path):
    """The ledger dry-run never starts or drains the OutboxDriver and never
    invokes adapter.send — even with pre-existing pending rows."""
    async def scenario():
        app = make_app(tmp_path, dry_run=True)
        assert isinstance(app.scheduler, LedgerScheduler)
        await app.start()
        await app.repo.upsert_chat(make_identity(chat_key="console:group:demo",
                                                 platform="console", self_id="bot"))
        await finish_batch(
            app.repo,
            app.outbox.to_items(
                Outgoing(chat_key=app.adapter.chat_key, text="pending",
                         idem_key="k1"),
                "cy-1",
            ),
            chat_key="console:group:demo",
        )
        await app.adapter.feed("hello")
        await app.run()  # dry-run: no drain, no worker, no send
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        states = await db2.read(
            lambda c: [r[0] for r in c.execute("SELECT state FROM outbox ORDER BY id")]
        )
        await db2.close()
        return states, len(app.adapter.sent)

    states, sent = run(scenario())
    assert states == ["pending"]  # untouched
    assert sent == 0  # no send ever


def test_dry_run_shutdown_stops_ledger_scheduler_safely(tmp_path):
    """Shutdown stops the ledger scheduler safely: the in-flight dispatch
    is drained, the loop task is gone, and shutdown is idempotent."""
    clock = VirtualClock()
    traces: list[DecisionTrace] = []
    trace_event = asyncio.Event()
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream, dry_run=True,
        trace_sink=lambda t: (traces.append(t), trace_event.set()),
    )

    async def scenario():
        await app.start()
        await app.adapter.feed("hello")
        run_task = asyncio.create_task(app.run())
        await asyncio.wait_for(trace_event.wait(), timeout=10)
        stream.release()
        await run_task  # run() shuts the app down on EOF
        await app.shutdown()  # idempotent
        return True

    assert run(scenario())
    assert len(traces) == 1
    assert app.scheduler._task is None  # the ledger loop was stopped
    assert app._started is False


def test_dry_run_generic_injected_scheduler_compatibility(tmp_path):
    """An explicitly injected generic Scheduler keeps the legacy wake path:
    the App preserves the injected scheduler (no forced ledger) and the
    legacy startup recovery wakes pending chats."""
    clock = VirtualClock()
    calls: list[ChatKey] = []
    cycle_event = asyncio.Event()

    async def fake_cycle(chat_key: ChatKey) -> Decision:
        calls.append(chat_key)
        cycle_event.set()
        return Decision(action="skip", reason=Reason.SKIP)

    scheduler = Scheduler(clock, fake_cycle)
    stream = _BlockingStream()
    app = make_app(
        tmp_path, clock=clock, input_stream=stream,
        dry_run=True, scheduler=scheduler,
    )
    assert app.scheduler is scheduler  # the injected scheduler is preserved

    async def scenario():
        await app.start()
        await app.adapter.feed("hello")
        run_task = asyncio.create_task(app.run())
        stream.release()
        await asyncio.wait_for(cycle_event.wait(), timeout=10)
        await run_task
        return True

    assert run(scenario())
    assert calls == [app.adapter.chat_key]


# ── Phase 3 agent: injected planner/replyer/budget, zero network client ──────

class _FakePlanner:
    """Scripted planner fake for the App wiring tests."""

    def __init__(self, result=None):
        self.result = result
        self.calls = 0

    async def plan(self, messages, *, identity, chat_log, reply_style,
                   focus_chat=None, tools=None, temperature=None,
                   max_tokens=None, deadline=None, max_tool_rounds=None):
        self.calls += 1
        from pretender.planner import PlanIntent, PlanResult

        return self.result or PlanResult(intent=PlanIntent.NO_ACTION)


class _FakeReplyer:
    """Scripted replyer fake for the App wiring tests."""

    def __init__(self, draft=None):
        self.draft = draft
        self.calls = 0

    async def reply(self, *, reply_reference, identity, reply_style,
                    reply_to=None, temperature=None, max_tokens=None,
                    deadline=None):
        self.calls += 1
        from pretender.replyer import ReplyDraft

        return self.draft or ReplyDraft.empty()


class _FakeBudget:
    """Scripted budget fake for the App wiring tests."""

    def __init__(self):
        self.decide_calls = 0
        self.record_calls = 0

    async def decide(self, chat_key):
        self.decide_calls += 1
        from pretender.budget import ALLOWED, BudgetDecision, BudgetUsage

        return BudgetDecision(
            kind=ALLOWED,
            usage=BudgetUsage(day="2026-01-01", calls=0, tokens=0, cost=0.0),
            remaining=100,
        )

    async def record(self, chat_key, *, calls=1, tokens=0, cost=0.0):
        self.record_calls += 1


def test_build_accepts_agent_without_network_client(tmp_path):
    """App.build accepts injected planner/replyer/budget and wraps them into
    a PhaseAgent WITHOUT constructing any real network client."""
    from pretender.cycle import PhaseAgent

    planner = _FakePlanner()
    replyer = _FakeReplyer()
    budget = _FakeBudget()
    app = make_app(
        tmp_path, dry_run=True, planner=planner, replyer=replyer, budget=budget
    )
    assert isinstance(app._cycle_fn._agent, PhaseAgent)
    assert app._cycle_fn._agent._planner is planner
    assert app._cycle_fn._agent._replyer is replyer
    assert app._cycle_fn._agent._budget is budget


def test_build_accepts_injected_phase_agent(tmp_path):
    """App.build accepts a ready PhaseAgent object directly."""
    from pretender.cycle import PhaseAgent

    agent = PhaseAgent(_FakePlanner(), _FakeReplyer())
    app = make_app(tmp_path, dry_run=True, agent=agent)
    assert app._cycle_fn._agent is agent


def test_build_requires_both_planner_and_replyer(tmp_path):
    with pytest.raises(ValueError, match="planner and replyer"):
        make_app(tmp_path, dry_run=True, planner=_FakePlanner())


def test_build_default_remains_no_agent(tmp_path):
    """The default build stays no-agent: the current dry-run remains
    zero-send with no LLM surface at all."""
    app = make_app(tmp_path, dry_run=True)
    assert app._cycle_fn._agent is None


def test_dry_run_agent_trigger_zero_outbox_zero_send(tmp_path):
    """End to end: an agent-injected dry-run App evaluates a hard trigger
    but creates ZERO outbox rows and never invokes adapter.send."""
    from pretender.planner import PlanIntent, PlanResult
    from pretender.replyer import ReplyDraft

    clock = VirtualClock()
    traces: list[DecisionTrace] = []
    trace_event = asyncio.Event()
    stream = _BlockingStream()
    planner = _FakePlanner(
        PlanResult(
            intent=PlanIntent.REPLY,
            reply_reference="参考回复",
            tokens_in=10,
            tokens_out=5,
            end_reason="reply",
        )
    )
    replyer = _FakeReplyer(ReplyDraft(text="你好", tokens_in=7, tokens_out=3))
    app = make_app(
        tmp_path, clock=clock, input_stream=stream, dry_run=True,
        trace_sink=lambda t: (traces.append(t), trace_event.set()),
        planner=planner, replyer=replyer,
    )

    async def scenario():
        await app.start()
        identity = make_identity(chat_key="console:group:demo",
                                 platform="console", self_id="bot")
        await app.repo.upsert_chat(identity)
        msg = Message(
            chat_key=app.adapter.chat_key,
            sender_id=SenderId("user"),
            sender_name="user",
            is_self=False,
            text="hi",
            id=MessageId("m1"),
            mentions=(SenderId("bot"),),
            recv_ts=clock.now(),
        )
        await app.repo.ingest_message(identity, msg)
        run_task = asyncio.create_task(app.run())  # startup recovery wakes it
        await asyncio.wait_for(trace_event.wait(), timeout=10)
        assert traces[0].decision is not None
        assert traces[0].decision.action == "trigger"
        stream.release()
        await run_task
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        outbox = await db2.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        cycles = await db2.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        await db2.close()
        return outbox, cycles

    outbox, cycles = run(scenario())
    assert outbox == 0  # zero outbox rows
    assert cycles == [("dry_run_agent_reply",)]  # evaluated, never sent
    assert len(app.adapter.sent) == 0  # zero adapter sends
    assert planner.calls == 1
    assert replyer.calls == 1


# ── Phase 3 live agent lane ──────────────────────────────────────────────────

def test_live_agent_one_console_send_after_terminal_settlement(tmp_path):
    """A LIVE App (dry_run=False) with an agent sends ONE console message
    after a terminal agent output settles: the outbox worker is woken by the
    CycleRunner's on_outbox callback (not drained at startup)."""
    from pretender.planner import PlanIntent, PlanResult
    from pretender.replyer import ReplyDraft

    clock = VirtualClock()
    stream = _BlockingStream()
    planner = _FakePlanner(
        PlanResult(
            intent=PlanIntent.REPLY, reply_reference="参考回复",
            tokens_in=10, tokens_out=5, end_reason="reply",
        )
    )
    replyer = _FakeReplyer(ReplyDraft(text="你好", tokens_in=7, tokens_out=3))
    app = make_app(
        tmp_path, clock=clock, input_stream=stream, planner=planner, replyer=replyer
    )  # dry_run defaults to False → live agent lane

    async def scenario():
        await app.start()
        identity = make_identity(chat_key="console:group:demo",
                                 platform="console", self_id="bot")
        await app.repo.upsert_chat(identity)
        msg = Message(
            chat_key=app.adapter.chat_key, sender_id=SenderId("user"),
            sender_name="user", is_self=False, text="hi", id=MessageId("m1"),
            mentions=(SenderId("bot"),), recv_ts=clock.now(),
        )
        await app.repo.ingest_message(identity, msg)
        run_task = asyncio.create_task(app.run())  # startup recovery wakes it
        sent = False
        for _ in range(3000):
            if app.adapter.sent:
                sent = True
                break
            await asyncio.sleep(0.01)  # real time: let the DB writer flush
        stream.release()
        await run_task
        return sent, list(app.adapter.sent)

    sent, sent_list = run(scenario())
    assert sent
    assert len(sent_list) == 1  # exactly one console send
    assert sent_list[0].text == "你好"


def test_dry_run_agent_zero_send_with_pending_rows(tmp_path):
    """A dry-run App with an agent evaluates the same way but creates ZERO
    outbox rows and never sends — even with a pre-existing pending row."""
    from pretender.planner import PlanIntent, PlanResult
    from pretender.replyer import ReplyDraft

    clock = VirtualClock()
    stream = _BlockingStream()
    planner = _FakePlanner(
        PlanResult(
            intent=PlanIntent.REPLY, reply_reference="参考回复",
            tokens_in=10, tokens_out=5, end_reason="reply",
        )
    )
    replyer = _FakeReplyer(ReplyDraft(text="你好", tokens_in=7, tokens_out=3))
    app = make_app(
        tmp_path, clock=clock, input_stream=stream, dry_run=True,
        planner=planner, replyer=replyer,
    )

    async def scenario():
        await app.start()
        identity = make_identity(chat_key="console:group:demo",
                                 platform="console", self_id="bot")
        await app.repo.upsert_chat(identity)
        # A pre-existing pending outbox row (a completed cycle left it).
        await finish_batch(
            app.repo,
            app.outbox.to_items(
                Outgoing(chat_key=app.adapter.chat_key, text="old", idem_key="k1"),
                "cy-0",
            ),
            chat_key="console:group:demo",
        )
        msg = Message(
            chat_key=app.adapter.chat_key, sender_id=SenderId("user"),
            sender_name="user", is_self=False, text="hi", id=MessageId("m1"),
            mentions=(SenderId("bot"),), recv_ts=clock.now(),
        )
        await app.repo.ingest_message(identity, msg)
        run_task = asyncio.create_task(app.run())
        for _ in range(3000):
            if planner.calls:
                break
            await asyncio.sleep(0.01)  # real time: let the DB writer flush
        stream.release()
        await run_task
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        states = await db2.read(
            lambda c: [r[0] for r in c.execute("SELECT state FROM outbox ORDER BY id")]
        )
        await db2.close()
        return states, len(app.adapter.sent)

    states, sent = run(scenario())
    assert sent == 0  # zero sends
    assert states == ["pending"]  # the pre-existing row stays pending
