"""Console adapter: deterministic REPL — feed, events, send + echo, REPL
reader lifecycle."""

from __future__ import annotations

import asyncio
import io

import pytest

from pretender.adapters.console import ConsoleAdapter
from pretender.clock import VirtualClock
from pretender.errors import AdapterError
from pretender.seams import Adapter
from pretender.types import Outgoing
from tests.durable_helpers import run


def make_adapter(**kw) -> ConsoleAdapter:
    return ConsoleAdapter(clock=VirtualClock(), **kw)


# ── shape ───────────────────────────────────────────────────────────────────

def test_adapter_satisfies_protocol():
    assert isinstance(make_adapter(), Adapter)


def test_identity_and_capabilities():
    adapter = make_adapter()
    assert adapter.name == "console"
    assert adapter.capabilities == frozenset()
    assert adapter.identity.platform == "console"
    assert adapter.identity.self_id == "bot"
    assert adapter.identity.kind == "group"


# ── feed / events ───────────────────────────────────────────────────────────

def test_feed_yields_normalized_message_events():
    async def scenario():
        adapter = make_adapter()
        await adapter.feed("你好", sender_id="u9", sender_name="小明")
        event = await adapter.events().__anext__()
        await adapter.close()
        return event

    event = run(scenario())
    assert event.type == "message"
    assert event.payload.chat_key == adapter_chat_key()
    assert event.payload.sender_id == "u9"
    assert event.payload.sender_name == "小明"
    assert event.payload.text == "你好"
    assert event.payload.id == "console:in:1"
    assert event.payload.recv_ts is not None


def adapter_chat_key():
    return "console:group:demo"


def test_feed_ids_are_sequential_and_deterministic():
    async def scenario():
        adapter = make_adapter()
        m1 = await adapter.feed("a")
        m2 = await adapter.feed("b")
        await adapter.close()
        return m1.id, m2.id

    assert run(scenario()) == ("console:in:1", "console:in:2")


def test_events_ends_after_close():
    async def scenario():
        adapter = make_adapter()
        await adapter.feed("a")
        await adapter.close()
        events = []
        async for event in adapter.events():
            events.append(event)
        return events

    events = run(scenario())
    assert len(events) == 1  # the fed message, then the sentinel


# ── send ────────────────────────────────────────────────────────────────────

def test_send_returns_deterministic_id_and_records_outgoing():
    async def scenario():
        out = io.StringIO()
        adapter = make_adapter(output_stream=out)
        pid = await adapter.send(Outgoing(chat_key=adapter.chat_key, text="hi"))
        await adapter.close()
        return pid, adapter.sent, out.getvalue()

    pid, sent, printed = run(scenario())
    assert pid == "console:out:1"
    assert len(sent) == 1 and sent[0].text == "hi"
    assert "hi" in printed


def test_send_emits_self_echo_event():
    async def scenario():
        adapter = make_adapter()
        pid = await adapter.send(Outgoing(chat_key=adapter.chat_key, text="hi"))
        event = await adapter.events().__anext__()
        await adapter.close()
        return pid, event

    pid, event = run(scenario())
    assert event.type == "message"
    assert event.payload.is_self is True
    assert event.payload.id == pid
    assert event.payload.text == "hi"


def test_call_raises_adapter_error():
    async def scenario():
        adapter = make_adapter()
        with pytest.raises(AdapterError):
            await adapter.call("send_group_msg")
        await adapter.close()

    run(scenario())


# ── REPL reader ─────────────────────────────────────────────────────────────

def test_repl_reads_lines_from_input_stream():
    async def scenario():
        adapter = make_adapter(input_stream=io.StringIO("hello\nworld\n"))
        await adapter.connect()
        texts = []
        async for event in adapter.events():
            if event.type == "message":
                texts.append(event.payload.text)
        return texts

    assert run(scenario()) == ["hello", "world"]


def test_repl_ignores_blank_lines_and_ends_on_eof():
    async def scenario():
        adapter = make_adapter(input_stream=io.StringIO("\n\nonly\n\n"))
        await adapter.connect()
        texts = []
        async for event in adapter.events():
            if event.type == "message":
                texts.append(event.payload.text)
        return texts

    assert run(scenario()) == ["only"]


def test_close_is_idempotent_and_cancels_reader():
    async def scenario():
        adapter = make_adapter(input_stream=io.StringIO("a\nb\n"))
        await adapter.connect()
        await asyncio.sleep(0)  # let the reader start
        await adapter.close()
        await adapter.close()
        return adapter._reader

    assert run(scenario()) is None