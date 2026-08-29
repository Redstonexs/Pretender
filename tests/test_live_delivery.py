"""Phase 4 live-delivery integration: Outgoing→OutputPipeline→durable
OutboxItem, OneBot acceptance/wiring, doctor readiness, and the live worker.

End-to-end with a real SQLite repository + VirtualClock and fake
OneBot/console adapters and fake planner/replyer seams. Async tests run via
asyncio.run() so the test extra stays at just pytest.
"""

from __future__ import annotations

import asyncio
import io
import threading

import pytest

from pretender.app import App
from pretender.adapters.console import ConsoleAdapter
from pretender.clock import VirtualClock
from pretender.config import Config
from pretender.cycle import CycleRunner, PhaseAgent
from pretender.doctor import Doctor
from pretender.gate import Gate
from pretender.media import MediaStore
from pretender.planner import PlanIntent, PlanResult
from pretender.replyer import ReplyDraft
from pretender.types import (
    AdapterEvent,
    ChatKey,
    CycleId,
    DispatchCause,
    DispatchGrant,
    DispatchRequest,
    Message,
    MessageId,
    Outgoing,
    SenderId,
)
from tests.durable_helpers import (
    finish_batch,
    make_identity,
    make_message,
    open_repo,
    run,
)

CK = ChatKey("qq:group:123456")


def run(coro):
    return asyncio.run(coro)


# ── fakes ────────────────────────────────────────────────────────────────────


class FakePlanner:
    """Scripted planner: returns one PlanResult per plan() call."""

    def __init__(self, result=None):
        self.result = result
        self.calls = 0

    async def plan(self, messages, *, identity, chat_log, reply_style,
                   focus_chat=None, bot_name="", drift_block="",
                   tools=None, temperature=None,
                   max_tokens=None, deadline=None, max_tool_rounds=None):
        self.calls += 1
        return self.result or PlanResult(intent=PlanIntent.NO_ACTION)


class FakeReplyer:
    """Scripted replyer: returns one ReplyDraft per reply() call."""

    def __init__(self, draft=None):
        self.draft = draft
        self.calls = 0

    async def reply(self, *, reply_reference, identity, reply_style,
                    reply_to=None, context=None, temperature=None,
                    max_tokens=None, deadline=None):
        self.calls += 1
        return self.draft or ReplyDraft.empty()


class FakeOneBot:
    """A fake OneBot adapter: records sends, resolves trusted delivery keys,
    and derives per-chat identity from the chat key (like the real bridge)."""

    name = "onebot"
    capabilities = frozenset({"quote", "at", "image"})

    def __init__(self, self_id="10001", clock=None, send_delay=0.0):
        self._self_id = self_id
        self._media = MediaStore()
        self._clock = clock if clock is not None else VirtualClock()
        self._send_delay = send_delay
        self.sent: list[Outgoing] = []
        self._delivered: dict[str, str] = {}
        self._seq = 0
        self._closed = False
        self._release = asyncio.Event()
        self._connected = False
        self._protocol_ok = True

    @property
    def connected(self) -> bool:
        """A real readiness/handshake signal (like the OneBot bridge's open
        connection). Tests control it via ``set_connected``."""
        return self._connected

    def set_connected(self, value: bool) -> None:
        self._connected = value

    def set_protocol_ok(self, value: bool) -> None:
        self._protocol_ok = value

    async def connect(self):
        pass

    async def events(self):
        # Block until released so the run loop stays alive while the worker
        # sends (like the console adapter's blocking REPL stream).
        await self._release.wait()
        if False:  # pragma: no cover
            yield

    def release(self):
        self._release.set()

    async def send(self, out):
        if self._send_delay > 0:
            await self._clock.sleep(self._send_delay)
        self._seq += 1
        pid = MessageId(f"ob:{self._seq}")
        self.sent.append(out)
        if out.delivery_key:
            self._delivered[str(pid)] = out.delivery_key
        return pid

    async def call(self, action, **params):
        """The doctor's protocol probe: a benign API round-trip that fails
        when the protocol is not validated."""
        if not self._protocol_ok:
            raise RuntimeError("no protocol response")
        return {"self_id": self._self_id}

    def delivery_key_for(self, msg):
        if not msg.is_self or msg.id is None:
            return None
        return self._delivered.get(str(msg.id))

    async def close(self):
        self._closed = True


class EchoOneBot:
    """A fake OneBot adapter with REAL echo correlation: ``send()`` awaits an
    echo that the ``events()`` receiver resolves — proving the receiver must
    be active before any send succeeds (a startup action's echo succeeds only
    because the receiver is consuming)."""

    name = "onebot"
    capabilities = frozenset({"quote", "at"})

    def __init__(self, self_id="10001"):
        self._self_id = self_id
        self._connected = True
        self._queue: asyncio.Queue = asyncio.Queue()
        self.sent: list[Outgoing] = []
        self._closed = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self):
        pass

    async def events(self):
        # The receiver: consume each send's echo. send() blocks until this
        # consumes, so a send before the receiver is active would hang.
        while True:
            item = await self._queue.get()
            if item is None:
                return  # stream end sentinel
            out, fut = item
            self.sent.append(out)
            if not fut.done():
                fut.set_result(None)
            if False:  # pragma: no cover — makes this an async generator
                yield

    async def send(self, out):
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put((out, fut))
        await fut  # resolves only when the receiver consumes the echo
        return MessageId(f"ob:{len(self.sent)}")

    async def call(self, action, **params):
        return {"self_id": self._self_id}

    def release(self):
        self._queue.put_nowait(None)

    async def close(self):
        self._closed = True
        self._queue.put_nowait(None)


class _BlockingStream:
    """A REPL input stream that blocks in readline until ``release``, then
    EOF — lets a test observe the worker's sends before the run loop ends."""

    def __init__(self):
        self._release = threading.Event()

    def readline(self):
        self._release.wait()
        return ""

    def release(self):
        self._release.set()


# ── helpers ──────────────────────────────────────────────────────────────────


def _trigger_message(recv_ts: float = 100.0, msg_id: str = "m1") -> Message:
    return Message(
        chat_key=CK,
        sender_id=SenderId("u1"),
        sender_name="user",
        is_self=False,
        text="hi",
        id=MessageId(msg_id),
        mentions=(SenderId("bot-1"),),
        recv_ts=recv_ts,
    )


async def _begin_dispatch(repo, *, cycle_id="cy-1", now=200.0) -> DispatchGrant:
    grant = await repo.begin_dispatch(
        DispatchRequest(
            chat_key=CK,
            cause=DispatchCause.INBOUND,
            cycle_id=CycleId(cycle_id),
            started_ts=now,
            expires_at=now + 300.0,
            now=now,
        )
    )
    assert isinstance(grant, DispatchGrant)
    return grant


def _reply_agent(text: str):
    planner = FakePlanner(
        PlanResult(
            intent=PlanIntent.REPLY, reply_reference="参考",
            tokens_in=5, tokens_out=2, end_reason="reply",
        )
    )
    replyer = FakeReplyer(ReplyDraft(text=text, tokens_in=3, tokens_out=1))
    return PhaseAgent(planner, replyer), planner, replyer


def _runner(repo, agent, *, dry_run=False):
    return CycleRunner(
        repo, Gate(), Config(), clock=VirtualClock(epoch=200.0),
        dry_run=dry_run, uuid_fn=lambda: "cy-1", agent=agent,
    )


# ── agent reply passes sanitize/split/typo before the outbox batch ──────────

def test_agent_reply_passes_pipeline_before_outbox(tmp_path):
    """The agent reply is run through the output pipeline (sanitize → split
    → typo) ONCE at terminal settlement: a CQ code is stripped and the text
    is split into ordered parts before the durable outbox batch."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        agent, planner, replyer = _reply_agent("你好[CQ:at,qq=123]世界。再见！")
        decision = await _runner(repo, agent).run_dispatch(grant)
        rows = await db.read(
            lambda c: c.execute(
                "SELECT text, group_id, seq, send_after_ts FROM outbox ORDER BY id"
            ).fetchall()
        )
        await repo.close()
        return decision, rows, planner, replyer

    decision, rows, planner, replyer = run(scenario())
    assert decision.action == "trigger"
    # Sanitize runs FIRST, so the CQ code is gone before split ever sees the
    # text. How many bubbles the split produces is probabilistic (seeded from
    # the durable output identity), so assert the content, not the count.
    texts = [r[0] for r in rows]
    assert "CQ:" not in "".join(texts)
    assert "".join(texts).replace("。", "") == "你好世界再见！"
    assert planner.calls == 1 and replyer.calls == 1


def test_split_parts_share_group_order_pacing(tmp_path):
    """Split parts share a stable content-derived group id, are ordered by
    seq, and the split stage's relative pacing maps to a durable absolute
    send_after_ts (part 0 immediate, later parts at now + delay)."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        agent, _, _ = _reply_agent("第一句\n第二句\n第三句")
        await _runner(repo, agent).run_dispatch(grant)
        rows = await db.read(
            lambda c: c.execute(
                "SELECT text, group_id, seq, send_after_ts FROM outbox ORDER BY id"
            ).fetchall()
        )
        await repo.close()
        return rows

    rows = run(scenario())
    assert [r[0] for r in rows] == ["第一句", "第二句", "第三句"]
    assert len({r[1] for r in rows}) == 1  # one stable group id
    assert [r[2] for r in rows] == [0, 1, 2]  # ordered by seq
    assert rows[0][3] is None  # part 0 sends immediately
    assert rows[1][3] == pytest.approx(200.0 + 1.5)
    assert rows[2][3] == pytest.approx(200.0 + 3.0)


def test_pipeline_honors_per_reply_switches():
    """The per-reply switches on the Outgoing are honored by the pipeline:
    skip_post_process bypasses optional stages, enable_splitter gates split, and
    enable_chinese_typo gates typo."""
    from pretender.config import OutputConfig
    from pretender.output.pipeline import OutputPipeline

    p = OutputPipeline(OutputConfig())
    # skip_post_process: optional stages are bypassed, but core sanitize still
    # runs as the final safety boundary.
    out = Outgoing(chat_key=CK, text="你好[CQ:at,qq=1]世界。再见！",
                   skip_post_process=True)
    p.run(out)
    assert out.text == "你好世界。再见！"
    assert out.parts is None
    # enable_splitter=False: the split stage is skipped.
    out2 = Outgoing(chat_key=CK, text="第一句。第二句！", enable_splitter=False)
    p.run(out2)
    assert out2.parts is None
    # enable_chinese_typo=False: the typo stage is skipped.
    out3 = Outgoing(chat_key=CK, text="你好", enable_chinese_typo=False)
    p.run(out3)
    assert out3.text == "你好"


# ── live worker: terminal output exactly once, delayed part paced ───────────

def test_live_worker_sends_split_parts_with_pacing(tmp_path):
    """A LIVE App (dry_run=False) with an agent sends the split reply parts
    exactly once: part 0 immediately, the delayed part when its
    send_after_ts is due (the worker paces send_after)."""
    clock = VirtualClock()
    stream = _BlockingStream()
    planner = FakePlanner(
        PlanResult(intent=PlanIntent.REPLY, reply_reference="参考",
                   tokens_in=5, tokens_out=2, end_reason="reply")
    )
    replyer = FakeReplyer(ReplyDraft(text="第一句\n第二句", tokens_in=3, tokens_out=1))
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "data" / "app.db")}}
    )
    app = App.build(
        cfg, clock=clock,
        adapter=ConsoleAdapter(clock=clock, input_stream=stream,
                               output_stream=io.StringIO()),
        planner=planner, replyer=replyer,
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
        for _ in range(3000):
            if len(app.adapter.sent) >= 2:
                break
            await asyncio.sleep(0.01)  # real time: let the DB writer flush
        stream.release()
        await run_task
        return [o.text for o in app.adapter.sent]

    sent = run(scenario())
    assert sent == ["第一句", "第二句"]  # both parts, exactly once


# ── dry-run: pipeline evaluates, zero outbox/send ───────────────────────────

def test_dry_run_pipeline_zero_outbox_zero_send(tmp_path):
    """A dry-run App with an agent evaluates the SAME pipeline but creates
    ZERO outbox rows and never sends — even for a split reply."""
    clock = VirtualClock()
    stream = _BlockingStream()
    planner = FakePlanner(
        PlanResult(intent=PlanIntent.REPLY, reply_reference="参考",
                   tokens_in=5, tokens_out=2, end_reason="reply")
    )
    replyer = FakeReplyer(ReplyDraft(text="第一句\n第二句", tokens_in=3, tokens_out=1))
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "data" / "app.db")}}
    )
    app = App.build(
        cfg, clock=clock,
        adapter=ConsoleAdapter(clock=clock, input_stream=stream,
                               output_stream=io.StringIO()),
        planner=planner, replyer=replyer, dry_run=True,
    )

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
        run_task = asyncio.create_task(app.run())
        for _ in range(3000):
            if planner.calls:
                break
            await asyncio.sleep(0.01)
        stream.release()
        await run_task
        from pretender.db import Database
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        outbox = await db2.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        await db2.close()
        return outbox

    outbox = run(scenario())
    assert outbox == 0  # zero outbox rows
    assert len(app.adapter.sent) == 0  # zero sends


# ── OneBot acceptance / config selection ────────────────────────────────────

def test_build_config_selects_onebot(tmp_path):
    """``adapter.name = "onebot"`` selects the OneBot bridge in live mode."""
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "d.db")},
         "adapter": {"name": "onebot"}}
    )
    app = App.build(cfg, clock=VirtualClock(), dry_run=False)
    assert app.adapter.name == "onebot"
    assert app.ingest is not None


def test_build_rejects_unsupported_adapter_config(tmp_path):
    """An unknown ``adapter.name`` is rejected at config load."""
    from pretender.errors import ConfigError

    with pytest.raises(ConfigError, match="adapter.name"):
        Config.from_dict({"adapter": {"name": "telegram"}})


# ── doctor: OneBot handshake + media readiness ──────────────────────────────

def test_doctor_onebot_handshake_and_media_readiness(tmp_path):
    """The doctor validates the selected OneBot adapter's handshake and
    capabilities and reports media readiness safely (no network)."""
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "d.db")}}
    )
    adapter = FakeOneBot(self_id="10001")
    adapter.set_connected(True)  # a real handshake completed
    doctor = Doctor(cfg, adapter=adapter, clock=VirtualClock())
    report = run(doctor.run())
    probe = report.by_name("adapter")
    assert probe is not None
    assert probe.status == "ok"
    assert probe.data["name"] == "onebot"
    assert "quote" in probe.data["capabilities"]
    assert probe.data["ready"] is True
    assert probe.data["media"] is True  # media pipeline ready


# ── real/fallback self echo reconciliation ──────────────────────────────────

def test_onebot_fallback_send_reconciles_without_duplicate(tmp_path):
    """End-to-end: an ambiguous OneBot send (retcode != 0) returns None, the
    outbox writes a synthetic local echo, and a later real self echo binds
    the real id to the delivery key and reconciles the synthetic row — the
    context is never duplicated."""
    import orjson
    from websockets.protocol import State

    from pretender.adapters.onebot import OneBotAdapter
    from pretender.config import OneBotConfig
    from pretender.outbox import OutboxDriver
    from pretender.types import Outgoing

    class FailConn:
        """A fake connection that answers every action with retcode=-1."""

        state = State.OPEN

        def __init__(self, adapter):
            self.adapter = adapter
            self.sent = []

        async def send(self, frame):
            data = orjson.loads(frame)
            self.sent.append(data)
            fut = self.adapter._pending.get(data["echo"])
            if fut is not None and not fut.done():
                fut.set_result({"retcode": -1, "data": None, "echo": data["echo"]})

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        identity = make_identity(chat_key="qq:group:111111", platform="qq", self_id="10001")
        await repo.upsert_chat(identity)
        clock = VirtualClock()
        adapter = OneBotAdapter(
            config=OneBotConfig(host="127.0.0.1", port=0, heartbeat_timeout_s=None),
            clock=clock, normalize_media=False, self_id="10001",
        )
        conn = FailConn(adapter)
        adapter._conn = conn
        adapter._lifecycle_seen = True
        adapter._probe_ok = True
        adapter._generation_self_id = "10001"
        outbox = OutboxDriver(repo, adapter, clock=clock)
        items = outbox.to_items(
            Outgoing(chat_key="qq:group:111111", text="大家好", idem_key="k1"), "cy-1"
        )
        await finish_batch(repo, items, chat_key="qq:group:111111")
        # the worker sends: the ambiguous ack returns None (no real id)
        sent = await outbox.pump(ChatKey("qq:group:111111"), now=clock.now())
        assert sent == 1
        row = await db.read(
            lambda c: c.execute(
                "SELECT state, platform_msg_id FROM outbox WHERE id = 1"
            ).fetchone()
        )
        assert row == ("sent", None)  # synthetic local echo path
        # a real self echo arrives through the adapter's normalization: the
        # real id is bound to the trusted delivery key
        raw_echo = {
            "time": 1700000010,
            "self_id": 10001,
            "post_type": "message_sent",
            "message_type": "group",
            "message_id": 90001,
            "group_id": 111111,
            "user_id": 10001,
            "message": [{"type": "text", "data": {"text": "大家好"}}],
            "sender": {"user_id": 10001, "nickname": "麦麦"},
        }
        event = await adapter._handle_frame(orjson.dumps(raw_echo))
        echo = event.payload
        key = adapter.delivery_key_for(echo)
        assert key == "cy-1:k1"
        result = await repo.ingest_message(identity, echo, self_echo_delivery_key=key)
        rows = await db.read(
            lambda c: c.execute(
                "SELECT platform_msg_id, is_self FROM messages ORDER BY id"
            ).fetchall()
        )
        await repo.close()
        return result, rows

    result, rows = run(scenario())
    assert result.inserted is False  # no duplicate context message
    assert result.echo_status == "already_reconciled"
    assert rows == [("90001", 1)]  # synthetic row updated to the real id, singular


def test_onebot_delivery_key_reconciles_self_echo(tmp_path):
    """The OneBot adapter's delivery_key_for is wired into the ingest/echo
    path: a real self echo carrying the trusted delivery key reconciles the
    in-flight outbox row to sent with the real platform id."""
    from pretender.db import Database

    clock = VirtualClock()
    adapter = FakeOneBot(self_id="10001")
    adapter.set_connected(True)  # a real handshake completed
    planner = FakePlanner(
        PlanResult(intent=PlanIntent.REPLY, reply_reference="参考",
                   tokens_in=5, tokens_out=2, end_reason="reply")
    )
    replyer = FakeReplyer(ReplyDraft(text="你好", tokens_in=3, tokens_out=1))
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "data" / "app.db")}}
    )
    app = App.build(
        cfg, clock=clock, adapter=adapter, planner=planner, replyer=replyer,
    )  # dry_run=False → live agent lane with the OneBot adapter

    async def scenario():
        await app.start()
        identity = make_identity(chat_key="qq:group:123456",
                                 platform="qq", self_id="10001")
        await app.repo.upsert_chat(identity)
        msg = Message(
            chat_key=CK, sender_id=SenderId("u1"), sender_name="user",
            is_self=False, text="hi", id=MessageId("m1"),
            mentions=(SenderId("10001"),), recv_ts=clock.now(),
        )
        await app.repo.ingest_message(identity, msg)
        run_task = asyncio.create_task(app.run())
        for _ in range(3000):
            if adapter.sent:
                break
            await asyncio.sleep(0.01)
        # The adapter sent one message and recorded its delivery key.
        assert len(adapter.sent) == 1
        pid = adapter.sent[0].delivery_key
        assert pid is not None
        # A real self echo carrying the trusted delivery key reconciles the
        # in-flight row to sent with the real platform id.
        echo = Message(
            chat_key=CK, sender_id=SenderId("10001"), sender_name="10001",
            is_self=True, text="你好", id=MessageId("ob:1"),
            recv_ts=clock.now(),
        )
        result = await app.ingest.handle(
            AdapterEvent(type="message", payload=echo, ts=echo.recv_ts)
        )
        db2 = Database(tmp_path / "data" / "app.db")
        await db2.open()
        row = await db2.read(
            lambda c: c.execute(
                "SELECT state, platform_msg_id FROM outbox ORDER BY id LIMIT 1"
            ).fetchone()
        )
        await db2.close()
        adapter.release()
        await run_task
        return result, row

    result, row = run(scenario())
    # The delivery_key_for was wired: the echo resolved to a matching row
    # (reconciled, or already_reconciled because the worker's confirmed send
    # already marked it sent with the real platform id). It is never
    # "unproven" — that would mean the trusted key was not resolved.
    assert result.echo_status in ("reconciled", "already_reconciled")
    assert row == ("sent", "ob:1")  # real platform id recorded


# ── Gate 4: startup recovery across all chats ───────────────────────────────

def test_startup_recovery_drains_all_chats(tmp_path):
    """Pre-existing SAFE pending outbox rows across ALL chats are drained at
    startup (before any wake), not only the console's single chat."""
    clock = VirtualClock()
    adapter = FakeOneBot(self_id="10001")
    adapter.set_connected(True)
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "data" / "app.db")}}
    )
    app = App.build(cfg, clock=clock, adapter=adapter, dry_run=False)

    async def scenario():
        await app.start()
        for ck, text in [("qq:group:111", "a"), ("qq:group:222", "b")]:
            identity = make_identity(chat_key=ck, platform="qq", self_id="10001")
            await app.repo.upsert_chat(identity)
            await finish_batch(
                app.repo,
                app.outbox.to_items(
                    Outgoing(chat_key=ChatKey(ck), text=text, idem_key=f"k-{ck}"),
                    "cy-1",
                ),
                chat_key=ck,
            )
        run_task = asyncio.create_task(app.run())
        for _ in range(3000):
            if len(adapter.sent) >= 2:
                break
            await asyncio.sleep(0.01)
        adapter.release()
        await run_task
        return sorted(o.text for o in adapter.sent)

    sent = run(scenario())
    assert sent == ["a", "b"]  # both chats drained at startup


def test_multi_chat_future_pacing_survives_no_new_wake(tmp_path):
    """Future-paced rows across multiple chats are sent when due WITHOUT a
    new wake: the worker retains the active chats across rounds and rechecks
    them after the future sleep naturally expires."""
    clock = VirtualClock()
    adapter = FakeOneBot(self_id="10001")
    adapter.set_connected(True)
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "data" / "app.db")}}
    )
    app = App.build(cfg, clock=clock, adapter=adapter, dry_run=False)

    async def scenario():
        await app.start()
        future = clock.now() + 100.0
        for ck, text in [("qq:group:111", "a"), ("qq:group:222", "b")]:
            identity = make_identity(chat_key=ck, platform="qq", self_id="10001")
            await app.repo.upsert_chat(identity)
            await finish_batch(
                app.repo,
                app.outbox.to_items(
                    Outgoing(chat_key=ChatKey(ck), text=text, idem_key=f"k-{ck}",
                             send_after_ts=future),
                    "cy-1",
                ),
                chat_key=ck,
            )
        run_task = asyncio.create_task(app.run())
        # No wake is ever issued: the worker's retained active chats pace the
        # future rows and send them when due.
        for _ in range(3000):
            if len(adapter.sent) >= 2:
                break
            await asyncio.sleep(0.01)
        adapter.release()
        await run_task
        return sorted(o.text for o in adapter.sent)

    sent = run(scenario())
    assert sent == ["a", "b"]  # sent when due, no new wake required


# ── Gate 4: no send before readiness / retry after connect ──────────────────

def test_no_send_before_readiness_retry_after_connect(tmp_path):
    """No worker send happens while the adapter is disconnected; once the
    handshake completes, the worker retries and sends the pending row."""
    clock = VirtualClock()
    adapter = FakeOneBot(self_id="10001")
    adapter.set_connected(False)  # NOT ready initially
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "data" / "app.db")}}
    )
    app = App.build(cfg, clock=clock, adapter=adapter, dry_run=False)

    async def scenario():
        await app.start()
        identity = make_identity(chat_key="qq:group:123456",
                                 platform="qq", self_id="10001")
        await app.repo.upsert_chat(identity)
        await finish_batch(
            app.repo,
            app.outbox.to_items(
                Outgoing(chat_key=CK, text="hi", idem_key="k1"), "cy-1"
            ),
            chat_key="qq:group:123456",
        )
        run_task = asyncio.create_task(app.run())
        # The app blocks in readiness (adapter disconnected): no sends.
        await asyncio.sleep(0.2)
        assert len(adapter.sent) == 0  # no send while disconnected
        # The handshake completes: the worker retries and sends.
        adapter.set_connected(True)
        for _ in range(3000):
            if adapter.sent:
                break
            await asyncio.sleep(0.01)
        adapter.release()
        await run_task
        return [o.text for o in adapter.sent]

    sent = run(scenario())
    assert sent == ["hi"]


# ── Gate 4: doctor no-handshake failure ─────────────────────────────────────

def test_doctor_no_handshake_reports_not_ready(tmp_path):
    """The doctor reports NOT-READY (a hard failure) when the OneBot adapter
    never completes a real handshake — not a silent ok from a listening
    background task."""
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "d.db")}}
    )
    adapter = FakeOneBot(self_id="10001")
    adapter.set_connected(False)  # no handshake ever
    doctor = Doctor(cfg, adapter=adapter, clock=VirtualClock())
    report = run(doctor.run())
    probe = report.by_name("adapter")
    assert probe is not None
    assert probe.status == "fail"
    assert "NOT ready" in probe.detail
    assert probe.data["ready"] is False


# ── Gate 4: console regression (cross-chat startup recovery) ────────────────

def test_console_startup_recovery_regression(tmp_path):
    """The console adapter still drains pre-existing pending rows at startup
    (now via the cross-chat recovery) and sends exactly once."""
    clock = VirtualClock()
    stream = _BlockingStream()
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "data" / "app.db")}}
    )
    app = App.build(
        cfg, clock=clock,
        adapter=ConsoleAdapter(clock=clock, input_stream=stream,
                               output_stream=io.StringIO()),
        dry_run=False,
    )

    async def scenario():
        await app.start()
        identity = make_identity(chat_key="console:group:demo",
                                 platform="console", self_id="bot")
        await app.repo.upsert_chat(identity)
        await finish_batch(
            app.repo,
            app.outbox.to_items(
                Outgoing(chat_key=app.adapter.chat_key, text="auto",
                         idem_key="k1"),
                "cy-1",
            ),
            chat_key="console:group:demo",
        )
        run_task = asyncio.create_task(app.run())
        for _ in range(3000):
            if app.adapter.sent:
                break
            await asyncio.sleep(0.01)
        stream.release()
        await run_task
        return [o.text for o in app.adapter.sent]

    sent = run(scenario())
    assert sent == ["auto"]  # exactly one startup send


# ── Gate 4 final: receiver active before any send ───────────────────────────

def test_startup_action_echo_succeeds_receiver_active(tmp_path):
    """A startup outbox send's action echo resolves because the background
    receiver is ACTIVE before any recovery send: send() blocks until the
    receiver consumes the echo, so a send before the receiver would hang."""
    clock = VirtualClock()
    adapter = EchoOneBot(self_id="10001")
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "data" / "app.db")}}
    )
    app = App.build(cfg, clock=clock, adapter=adapter, dry_run=False)

    async def scenario():
        await app.start()
        identity = make_identity(chat_key="qq:group:123456",
                                 platform="qq", self_id="10001")
        await app.repo.upsert_chat(identity)
        await finish_batch(
            app.repo,
            app.outbox.to_items(
                Outgoing(chat_key=CK, text="hi", idem_key="k1"), "cy-1"
            ),
            chat_key="qq:group:123456",
        )
        run_task = asyncio.create_task(app.run())
        for _ in range(3000):
            if adapter.sent:
                break
            await asyncio.sleep(0.01)
        adapter.release()
        await run_task
        return [o.text for o in adapter.sent]

    sent = run(scenario())
    assert sent == ["hi"]  # the echo resolved because the receiver was active


# ── Gate 4 final: reconnect recovery without premature in_flight ────────────

def test_reconnect_recovery_no_premature_in_flight(tmp_path):
    """After a mid-run disconnect, the worker makes NO premature in_flight
    transition; after reconnect it resumes and sends the still-pending row."""
    clock = VirtualClock(auto_advance=False)
    adapter = FakeOneBot(self_id="10001")
    adapter.set_connected(True)
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "data" / "app.db")}}
    )
    app = App.build(cfg, clock=clock, adapter=adapter, dry_run=False)

    async def scenario():
        await app.start()
        identity = make_identity(chat_key="qq:group:123456",
                                 platform="qq", self_id="10001")
        await app.repo.upsert_chat(identity)
        future = clock.now() + 50.0
        await finish_batch(
            app.repo,
            app.outbox.to_items(
                Outgoing(chat_key=CK, text="a", idem_key="k1"), "cy-1"
            ),
            chat_key="qq:group:123456",
        )
        await finish_batch(
            app.repo,
            app.outbox.to_items(
                Outgoing(chat_key=CK, text="b", idem_key="k2",
                         send_after_ts=future),
                "cy-2",
            ),
            chat_key="qq:group:123456",
        )
        run_task = asyncio.create_task(app.run())
        # Wait for the startup drain to complete durably: "a" sent, "b"
        # (future) still pending.
        for _ in range(3000):
            states = await app.repo._db.read(
                lambda c: [r[0] for r in c.execute("SELECT state FROM outbox ORDER BY id")]
            )
            if states == ["sent", "pending"]:
                break
            await asyncio.sleep(0.01)
        # Disconnect before "b" is due, then advance past its due time.
        adapter.set_connected(False)
        clock.advance(60.0)  # "b" is now due, but the adapter is down
        await asyncio.sleep(0.2)
        states = await app.repo._db.read(
            lambda c: [r[0] for r in c.execute("SELECT state FROM outbox ORDER BY id")]
        )
        assert states == ["sent", "pending"]  # no premature in_flight
        # Reconnect: the worker resumes and sends "b".
        adapter.set_connected(True)
        clock.advance(1.0)  # wake the worker's readiness sleeper
        for _ in range(3000):
            if len(adapter.sent) >= 2:
                break
            await asyncio.sleep(0.01)
        adapter.release()
        await run_task
        return [o.text for o in adapter.sent], states

    sent, states = run(scenario())
    assert sent == ["a", "b"]
    assert states == ["sent", "pending"]


# ── Gate 4 final: multi-chat fairness / fresh timing ────────────────────────

def test_multi_chat_fairness_bounded_pump(tmp_path):
    """Each chat gets ONE bounded pump per worker round: a backlogged chat
    cannot starve another chat's due row."""
    clock = VirtualClock()
    adapter = FakeOneBot(self_id="10001")
    adapter.set_connected(True)
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "data" / "app.db")}}
    )
    app = App.build(cfg, clock=clock, adapter=adapter, dry_run=False)

    async def scenario():
        await app.start()
        future = clock.now() + 50.0
        identity_a = make_identity(chat_key="qq:group:111",
                                   platform="qq", self_id="10001")
        await app.repo.upsert_chat(identity_a)
        for i in range(15):
            await finish_batch(
                app.repo,
                app.outbox.to_items(
                    Outgoing(chat_key=ChatKey("qq:group:111"), text=f"a{i}",
                             idem_key=f"a{i}", send_after_ts=future),
                    f"cy-a{i}",
                ),
                chat_key="qq:group:111",
            )
        identity_b = make_identity(chat_key="qq:group:222",
                                   platform="qq", self_id="10001")
        await app.repo.upsert_chat(identity_b)
        await finish_batch(
            app.repo,
            app.outbox.to_items(
                Outgoing(chat_key=ChatKey("qq:group:222"), text="b0",
                         idem_key="b0", send_after_ts=future),
                "cy-b0",
            ),
            chat_key="qq:group:222",
        )
        run_task = asyncio.create_task(app.run())
        for _ in range(3000):
            if len(adapter.sent) >= 16:
                break
            await asyncio.sleep(0.01)
        adapter.release()
        await run_task
        return [o.text for o in adapter.sent]

    sent = run(scenario())
    assert len(sent) == 16
    # Chat B's row is sent in the FIRST due round (bounded pump per chat),
    # not starved behind all 15 of chat A.
    assert sent.index("b0") < 15


def test_slow_send_fresh_clock_next_chat(tmp_path):
    """A slow send advances the clock; the next chat's pump uses a FRESH
    clock, so its sent_ts reflects the time after the slow send."""
    clock = VirtualClock()
    adapter = FakeOneBot(self_id="10001", clock=clock, send_delay=5.0)
    adapter.set_connected(True)
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "data" / "app.db")}}
    )
    app = App.build(cfg, clock=clock, adapter=adapter, dry_run=False)

    async def scenario():
        await app.start()
        future = clock.now() + 50.0
        identity_a = make_identity(chat_key="qq:group:111",
                                   platform="qq", self_id="10001")
        await app.repo.upsert_chat(identity_a)
        await finish_batch(
            app.repo,
            app.outbox.to_items(
                Outgoing(chat_key=ChatKey("qq:group:111"), text="a",
                         idem_key="a", send_after_ts=future),
                "cy-a",
            ),
            chat_key="qq:group:111",
        )
        identity_b = make_identity(chat_key="qq:group:222",
                                   platform="qq", self_id="10001")
        await app.repo.upsert_chat(identity_b)
        await finish_batch(
            app.repo,
            app.outbox.to_items(
                Outgoing(chat_key=ChatKey("qq:group:222"), text="b",
                         idem_key="b", send_after_ts=future),
                "cy-b",
            ),
            chat_key="qq:group:222",
        )
        run_task = asyncio.create_task(app.run())
        for _ in range(3000):
            states = await app.repo._db.read(
                lambda c: [r[0] for r in c.execute("SELECT state FROM outbox ORDER BY id")]
            )
            if states == ["sent", "sent"]:
                break
            await asyncio.sleep(0.01)
        rows = await app.repo._db.read(
            lambda c: c.execute(
                "SELECT text, sent_ts FROM outbox ORDER BY id"
            ).fetchall()
        )
        adapter.release()
        await run_task
        return rows

    rows = run(scenario())
    assert [r[0] for r in rows] == ["a", "b"]
    # Chat A's slow send (5s) advanced the clock; chat B's pump used a FRESH
    # clock, so its sent_ts is at least 5s after chat A's.
    assert rows[1][1] >= rows[0][1] + 5.0


# ── Gate 4 final: strict dry-run adapter restriction ────────────────────────

def test_dry_run_rejects_config_selected_onebot(tmp_path):
    """Dry-run is console-only even for a CONFIG-SELECTED OneBot: it must
    never connect to or consume OneBot traffic."""
    from pretender.errors import ConfigError

    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "d.db")},
         "adapter": {"name": "onebot"}}
    )
    with pytest.raises(ConfigError, match="dry-run"):
        App.build(cfg, clock=VirtualClock(), dry_run=True)


# ── Gate 4 final: doctor protocol-not-ready vs ready ────────────────────────

def test_doctor_protocol_not_ready_reports_fail(tmp_path):
    """The doctor distinguishes adapter LISTENING from validated PROTOCOL
    readiness: an open connection whose protocol probe fails is a hard
    failure, never a false-success report."""
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "d.db")}}
    )
    adapter = FakeOneBot(self_id="10001")
    adapter.set_connected(True)  # the socket is open...
    adapter.set_protocol_ok(False)  # ...but the protocol does not answer
    doctor = Doctor(cfg, adapter=adapter, clock=VirtualClock())
    report = run(doctor.run())
    probe = report.by_name("adapter")
    assert probe is not None
    assert probe.status == "fail"
    assert "protocol probe failed" in probe.detail
    assert probe.data["protocol"] is False


def test_doctor_protocol_ready_reports_ok(tmp_path):
    """An open connection AND a validated protocol round-trip report ok."""
    cfg = Config.from_dict(
        {"storage": {"db_path": str(tmp_path / "d.db")}}
    )
    adapter = FakeOneBot(self_id="10001")
    adapter.set_connected(True)
    adapter.set_protocol_ok(True)
    doctor = Doctor(cfg, adapter=adapter, clock=VirtualClock())
    report = run(doctor.run())
    probe = report.by_name("adapter")
    assert probe is not None
    assert probe.status == "ok"
    assert probe.data["protocol"] is True
    assert probe.data["ready"] is True
