"""Outbox driver: pure to_items conversion (deterministic per-part keys),
at-most-once pump, self-echo reconciliation, pacing, and restart handling.
The driver is typed against the Repository/Adapter seams, so it must work
against protocol-only fakes."""

from __future__ import annotations

import pytest

from pretender.adapters.console import ConsoleAdapter
from pretender.clock import VirtualClock
from pretender.errors import AdapterNotReady
from pretender.ingest import Ingest
from pretender.outbox import OutboxDriver
from pretender.record import Recorder
from pretender.seams import Adapter, Repository
from pretender.types import AdapterEvent, OutboxItem, Outgoing, Segment
from tests.durable_helpers import (
    CK,
    FakeRepo,
    finish_batch,
    make_identity,
    make_message,
    open_repo,
    run,
)


def make_driver(repo, adapter=None, clock=None):
    return OutboxDriver(repo, adapter or ConsoleAdapter(clock=clock), clock=clock)


def outgoing(text="hi", **kw) -> Outgoing:
    return Outgoing(chat_key=CK, text=text, **kw)


async def _noop_wake(chat_key) -> None:
    return None


# ── to_items: pure conversion, deterministic distinct keys ──────────────────

def test_to_items_single_item():
    driver = make_driver(FakeRepo())
    items = driver.to_items(outgoing("hi"), "cy-1")
    assert len(items) == 1
    assert items[0].text == "hi"
    assert items[0].state == "pending"
    assert items[0].seq is None
    assert items[0].idem_key == "cy-1:0"  # delivery intent, not content


def test_segmented_outgoing_is_kept_atomic_when_text_parts_exist():
    """Split text must not repeat the same media/segment payload for every
    part. Until callers construct per-part segments, one durable row is safe."""
    out = outgoing("full text")
    out.parts = ["part one", "part two"]
    out.segments = [Segment("image", {"file": "https://example.test/a.png"})]
    items = make_driver(FakeRepo()).to_items(out, "cy-1")
    assert len(items) == 1
    assert items[0].text == "full text"
    assert items[0].segments == tuple(out.segments)


def test_to_items_split_parts_share_group_and_sequence():
    driver = make_driver(FakeRepo())
    items = driver.to_items(outgoing("one", parts=["one", "two", "three"], group_id="g1"), "cy-1")
    assert [i.text for i in items] == ["one", "two", "three"]
    assert {i.group_id for i in items} == {"g1"}
    assert [i.seq for i in items] == [0, 1, 2]
    assert len({i.idem_key for i in items}) == 3


def test_to_items_split_explicit_key_is_namespaced_and_distinct():
    """A multi-part group with an explicit idem_key derives deterministic,
    distinct per-part keys, namespaced by the cycle."""
    driver = make_driver(FakeRepo())
    a = driver.to_items(outgoing("x", parts=["a", "b"], idem_key="delivery-1"), "cy-1")
    b = driver.to_items(outgoing("x", parts=["a", "b"], idem_key="delivery-1"), "cy-1")
    assert [i.idem_key for i in a] == ["cy-1:delivery-1:0", "cy-1:delivery-1:1"]
    assert [i.idem_key for i in b] == [i.idem_key for i in a]  # deterministic
    assert len({i.idem_key for i in a}) == 2  # distinct per part


def test_to_items_keys_are_cycle_scoped():
    """Same text may send in different cycles (distinct keys); retrying the
    same completed cycle produces the same keys (idempotent)."""
    driver = make_driver(FakeRepo())
    a = driver.to_items(outgoing("hi"), "cy-1")
    b = driver.to_items(outgoing("hi"), "cy-2")
    c = driver.to_items(outgoing("hi"), "cy-1")
    assert a[0].idem_key == "cy-1:0"
    assert b[0].idem_key == "cy-2:0"
    assert c[0].idem_key == a[0].idem_key  # retry of the same cycle
    assert a[0].idem_key != b[0].idem_key  # different cycle, different intent


def test_to_items_empty_outgoing_raises():
    driver = make_driver(FakeRepo())
    with pytest.raises(ValueError):
        driver.to_items(Outgoing(chat_key=CK, text=""), "cy-1")


# ── pump: at-most-once ──────────────────────────────────────────────────────

def test_pump_sends_and_writes_self_echo(tmp_path):
    async def scenario():
        _db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        adapter = ConsoleAdapter(clock=VirtualClock())
        driver = make_driver(repo, adapter)
        await finish_batch(repo, driver.to_items(outgoing("bot reply", idem_key="k1"), "cy-1"))
        sent = await driver.pump(CK, now=100.0)
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT state, platform_msg_id FROM outbox WHERE id = 1"
            ).fetchone()
        )
        echo = await repo.get_message(CK, "console:out:1")
        await repo.close()
        return sent, row, echo, len(adapter.sent)

    sent, row, echo, sent_count = run(scenario())
    assert sent == 1
    assert row == ("sent", "console:out:1")
    assert echo is not None and echo.is_self and echo.text == "bot reply"
    assert sent_count == 1


def test_pump_respects_send_after_pacing(tmp_path):
    async def scenario():
        _db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        adapter = ConsoleAdapter(clock=VirtualClock())
        driver = make_driver(repo, adapter)
        await finish_batch(repo, driver.to_items(outgoing("later", idem_key="k1", send_after_ts=500.0), "cy-1"))
        early = await driver.pump(CK, now=100.0)
        late = await driver.pump(CK, now=600.0)
        await repo.close()
        return early, late, len(adapter.sent)

    early, late, sent_count = run(scenario())
    assert early == 0
    assert late == 1
    assert sent_count == 1


def test_failed_send_is_never_retried(tmp_path):
    async def scenario():
        _db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())

        class FlakyAdapter(ConsoleAdapter):
            async def send(self, out):
                self._seq += 1
                raise RuntimeError("network down")

        adapter = FlakyAdapter(clock=VirtualClock())
        driver = make_driver(repo, adapter)
        await finish_batch(repo, driver.to_items(outgoing("hi", idem_key="k1"), "cy-1"))
        first = await driver.pump(CK, now=100.0)
        second = await driver.pump(CK, now=200.0)  # must NOT retry
        row = await repo._db.read(
            lambda c: c.execute(
                "SELECT state, attempt_started_ts FROM outbox WHERE id = 1"
            ).fetchone()
        )
        await repo.close()
        return first, second, row

    first, second, row = run(scenario())
    assert first == 0 and second == 0
    assert row == ("in_flight", 100.0)  # ambiguous outcome, never reset


def test_in_flight_survives_restart_without_retry(tmp_path):
    path = tmp_path / "t.db"

    async def scenario():
        _db, repo = await open_repo(path)
        await repo.upsert_chat(make_identity())
        adapter = ConsoleAdapter(clock=VirtualClock())
        driver = make_driver(repo, adapter)
        await finish_batch(repo, driver.to_items(outgoing("hi", idem_key="k1"), "cy-1"))
        await repo.attempt_outbox(1, 100.0)  # crash mid-send
        await repo.close()
        _db2, repo2 = await open_repo(path)
        await repo2.upsert_chat(make_identity())
        adapter2 = ConsoleAdapter(clock=VirtualClock())
        driver2 = make_driver(repo2, adapter2)
        sent = await driver2.pump(CK, now=999.0)
        await repo2.close()
        return sent, len(adapter2.sent)

    sent, sent_count = run(scenario())
    assert sent == 0
    assert sent_count == 0  # the adapter was never invoked


def test_split_batch_sends_each_part_once(tmp_path):
    async def scenario():
        _db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        adapter = ConsoleAdapter(clock=VirtualClock())
        driver = make_driver(repo, adapter)
        await finish_batch(
            repo, driver.to_items(outgoing("x", parts=["a", "b"], group_id="g1"), "cy-1")
        )
        sent = await driver.pump(CK, now=100.0)
        states = await repo._db.read(
            lambda c: [
                r[0] for r in c.execute("SELECT state FROM outbox ORDER BY seq")
            ]
        )
        await repo.close()
        return sent, states, len(adapter.sent)

    sent, states, sent_count = run(scenario())
    assert sent == 2
    assert states == ["sent", "sent"]
    assert sent_count == 2


# ── protocol-only dependency ────────────────────────────────────────────────

def test_driver_works_against_protocol_only_fakes():
    """OutboxDriver must depend only on the Repository and Adapter seams."""

    class FakeAdapter:
        name = "fake"
        capabilities = frozenset()

        def __init__(self):
            self.sent: list[Outgoing] = []

        async def connect(self) -> None:
            pass

        async def events(self):
            return
            yield  # pragma: no cover

        async def send(self, out: Outgoing) -> str | None:
            self.sent.append(out)
            return "fake:1"

        async def call(self, action: str, **params):
            return None

    async def scenario():
        fake_repo = FakeRepo()
        assert isinstance(fake_repo, Repository)
        fake_repo.ready_items = [OutboxItem(chat_key=CK, text="hi", idem_key="k1", id=1)]
        adapter = FakeAdapter()
        assert isinstance(adapter, Adapter)
        driver = OutboxDriver(fake_repo, adapter, clock=VirtualClock())
        sent = await driver.pump(CK, now=100.0)
        return sent, adapter.sent, fake_repo.calls

    sent, sent_list, calls = run(scenario())
    assert sent == 1
    assert [c.text for c in sent_list] == ["hi"]
    kinds = [c[0] for c in calls]
    assert kinds == ["list_ready_outbox", "attempt_outbox", "mark_outbox_sent"]


def test_prewrite_adapter_not_ready_requeues_safely():
    """Only an adapter-proven pre-write failure may return an attempted row
    to pending; generic send errors still remain ambiguous/in-flight."""

    class NotReadyAdapter:
        name = "not-ready"
        capabilities = frozenset()

        async def connect(self) -> None:
            pass

        async def events(self):
            return
            yield  # pragma: no cover

        async def send(self, out: Outgoing) -> str | None:
            raise AdapterNotReady("no socket before write")

        async def call(self, action: str, **params):
            return None

    async def scenario():
        repo = FakeRepo()
        repo.ready_items = [OutboxItem(chat_key=CK, text="hi", idem_key="k1", id=1)]
        sent = await OutboxDriver(repo, NotReadyAdapter(), clock=VirtualClock()).pump(
            CK, now=100.0
        )
        return sent, [call[0] for call in repo.calls]

    sent, calls = run(scenario())
    assert sent == 0
    assert calls == ["list_ready_outbox", "attempt_outbox", "requeue_outbox"]


# ── self echo reconciliation ────────────────────────────────────────────────

def test_real_echo_reconciles_without_second_send(tmp_path):
    """The console adapter emits the sent message back as an event; ingest
    must dedupe it against the synthetic echo and never wake."""

    async def scenario():
        _db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        # The adapter serves the same chat the repo knows.
        adapter = ConsoleAdapter(chat_key="qq:group:123456", clock=VirtualClock())
        driver = make_driver(repo, adapter)
        ingest = Ingest(
            repo,
            Recorder(tmp_path / "events.jsonl"),
            wake=_noop_wake,
            identity=lambda ck: make_identity(),
        )
        await finish_batch(repo, driver.to_items(outgoing("bot reply", idem_key="k1"), "cy-1"))
        await driver.pump(CK, now=100.0)
        # The adapter's echo event arrives through the normal event path.
        echo_event = await adapter.events().__anext__()
        result = await ingest.handle(echo_event)
        count = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        )
        await repo.close()
        return result, count

    result, count = run(scenario())
    assert result.inserted is False  # reconciled, no wake
    assert count == 1  # synthetic echo only — no duplicate row


# ── outgoing transport metadata: delivery key propagation ───────────────────

def test_pump_forwards_delivery_key_through_transport_metadata():
    """Every outgoing message carries the outbox item's delivery/
    idempotency key in Outgoing.delivery_key, so a real platform echo can
    carry it back and prove an ambiguous send landed."""

    class KeyAdapter:
        name = "key_adapter"
        capabilities = frozenset()

        def __init__(self):
            self.sent: list[Outgoing] = []

        async def connect(self) -> None:
            pass

        async def events(self):
            return
            yield  # pragma: no cover

        async def send(self, out: Outgoing) -> str | None:
            self.sent.append(out)
            return "fake:1"

        async def call(self, action: str, **params):
            return None

    async def scenario():
        fake_repo = FakeRepo()
        fake_repo.ready_items = [
            OutboxItem(chat_key=CK, text="hi", idem_key="cy-1:0", id=1),
            OutboxItem(chat_key=CK, text="part2", idem_key="cy-1:1", id=2),
        ]
        adapter = KeyAdapter()
        driver = OutboxDriver(fake_repo, adapter, clock=VirtualClock())
        await driver.pump(CK, now=100.0)
        return [(o.text, o.delivery_key) for o in adapter.sent]

    sent = run(scenario())
    assert sent == [("hi", "cy-1:0"), ("part2", "cy-1:1")]
