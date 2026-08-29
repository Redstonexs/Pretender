"""OneBot v11 adapter: reverse/forward WebSocket, message normalization,
group/private IDs, self echo, face/image/sticker placeholders, echo
correlation, retcode fallback + real-echo reconciliation, reconnect/
heartbeat/close, delivery-key resolution, and Adapter Protocol compatibility."""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import orjson
import pytest
from websockets.asyncio.client import connect as ws_connect
from websockets.asyncio.server import serve as ws_serve
from websockets.exceptions import ConnectionClosed
from websockets.frames import CloseCode
from websockets.protocol import State

from pretender.adapters.onebot import OneBotAdapter
from pretender.clock import VirtualClock
from pretender.config import OneBotConfig
from pretender.errors import AdapterError, TransientError
from pretender.media import MediaStore
from pretender.seams import Adapter
from pretender.types import AdapterEvent, ChatKey, Outgoing
from tests.durable_helpers import run
from tests.onebot_fixtures import FIXTURES, api_fail, api_ok


def make_adapter(**kw) -> OneBotAdapter:
    cfg = kw.pop("config", None)
    if cfg is None:
        cfg = OneBotConfig(host="127.0.0.1", port=0, heartbeat_timeout_s=None)
    kw.setdefault("clock", VirtualClock())
    kw.setdefault("normalize_media", False)
    return OneBotAdapter(config=cfg, **kw)


@contextmanager
def capture_onebot_logs(caplog):
    """Capture the adapter logger even after setup_logging disables parent
    propagation (as test_log.py does for the full-suite process)."""
    logger = logging.getLogger("pretender.onebot")
    previous_propagate = logger.propagate
    logger.addHandler(caplog.handler)
    logger.propagate = False
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)
        logger.propagate = previous_propagate


def adapter_port(adapter: OneBotAdapter) -> int:
    return adapter._server.sockets[0].getsockname()[1]


class FakeOneBot:
    """A fake OneBot client that dials the adapter's reverse server."""

    def __init__(
        self,
        port: int,
        path: str = "/onebot/v11/ws",
        token: str | None = None,
        auto_lifecycle: bool = True,
    ):
        separator = "&" if "?" in path else "?"
        self.uri = f"ws://127.0.0.1:{port}{path}{separator}message_format=array"
        self.token = token
        self.auto_lifecycle = auto_lifecycle
        self.ws: Any = None
        self.actions: list[dict] = []

    async def connect(self) -> "FakeOneBot":
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
        self.ws = await ws_connect(self.uri, additional_headers=headers)
        return self

    async def send_event(self, payload: dict) -> None:
        await self.ws.send(orjson.dumps(payload))

    async def next_action(self, timeout: float = 2.0) -> dict:
        """Return the next non-probe action, auto-answering the adapter's
        ``get_login_info`` readiness probe (retcode 0) when it arrives."""
        while True:
            raw = await asyncio.wait_for(self.ws.recv(), timeout)
            data = orjson.loads(raw)
            self.actions.append(data)
            if data.get("action") == "get_login_info":
                await self.respond(
                    data["echo"], retcode=0, data={"user_id": 10001}
                )
                continue
            return data

    async def respond(self, echo: str, retcode: int = 0, data=None, status: str = "ok") -> None:
        await self.ws.send(
            orjson.dumps({"status": status, "retcode": retcode, "data": data, "echo": echo})
        )

    async def close(self) -> None:
        if self.ws is not None:
            await self.ws.close()


async def drain(adapter: OneBotAdapter) -> None:
    async for _ in adapter.events():
        pass


async def wait_events(events: list, n: int, timeout: float = 2.0) -> None:
    for _ in range(int(timeout / 0.02)):
        if len(events) >= n:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"expected {n} events, got {len(events)}")


async def wait_connected(adapter: OneBotAdapter, timeout: float = 3.0) -> None:
    for _ in range(int(timeout / 0.02)):
        if adapter.connected:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("adapter did not connect")


async def wait_disconnected(adapter: OneBotAdapter, timeout: float = 3.0) -> None:
    for _ in range(int(timeout / 0.02)):
        if not adapter.connected:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("adapter did not disconnect")


async def collect_one(adapter: OneBotAdapter, client: FakeOneBot, payload: dict):
    """Send one event and return the first normalized AdapterEvent."""
    events: list = []
    task = asyncio.create_task(drain_into(adapter, events))
    await asyncio.sleep(0.05)
    await client.send_event(payload)
    await wait_events(events, 1)
    await client.close()
    await adapter.close()
    await task
    return events[0]


async def drain_into(adapter: OneBotAdapter, events: list) -> None:
    async for ev in adapter.events():
        events.append(ev)


# ── shape / protocol ────────────────────────────────────────────────────────

def test_adapter_satisfies_protocol():
    assert isinstance(make_adapter(), Adapter)


def test_name_and_capabilities():
    adapter = make_adapter()
    assert adapter.name == "onebot"
    assert {"quote", "at", "image", "face", "sticker"} <= adapter.capabilities


def test_config_mode_validation():
    with pytest.raises(Exception):
        OneBotConfig(mode="bogus")


# ── message normalization ───────────────────────────────────────────────────

def test_group_text_image_normalization():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port, auto_lifecycle=False).connect()
        event = await collect_one(adapter, client, FIXTURES["group_text_image"])
        return event

    event = run(scenario())
    assert event.type == "message"
    msg = event.payload
    assert msg.chat_key == "qq:group:111111"
    assert msg.sender_id == "222222"
    assert msg.sender_name == "小明"
    assert msg.is_self is False
    assert msg.id == "12345"
    assert msg.text == "看看这个 [图片]"
    assert [s.kind for s in msg.segments] == ["text", "image"]
    assert msg.recv_ts == 1700000001.0


def test_private_message_normalization():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port, auto_lifecycle=False).connect()
        event = await collect_one(adapter, client, FIXTURES["private_text"])
        return event

    event = run(scenario())
    msg = event.payload
    assert msg.chat_key == "qq:private:333333"
    assert msg.sender_id == "333333"
    assert msg.text == "在吗"


def test_at_message_mentions():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        event = await collect_one(adapter, client, FIXTURES["group_at"])
        return event

    event = run(scenario())
    msg = event.payload
    assert msg.text == "@麦麦 你好"
    assert msg.mentions == ("10001",)


def test_reply_message():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        event = await collect_one(adapter, client, FIXTURES["group_reply"])
        return event

    event = run(scenario())
    msg = event.payload
    assert msg.reply_to == "12345"
    assert msg.text == "同意"


def test_face_and_sticker_placeholders():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        face = await collect_one(adapter, client, FIXTURES["group_face"])
        adapter2 = make_adapter()
        await adapter2.connect()
        port2 = adapter_port(adapter2)
        client2 = await FakeOneBot(port2).connect()
        sticker = await collect_one(adapter2, client2, FIXTURES["group_sticker"])
        return face.payload, sticker.payload

    face, sticker = run(scenario())
    assert face.text == "[表情]哈哈"
    assert [s.kind for s in face.segments] == ["face", "text"]
    assert sticker.text == "[贴纸]"
    assert [s.kind for s in sticker.segments] == ["sticker"]


def test_self_echo_group():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        event = await collect_one(adapter, client, FIXTURES["self_echo_group"])
        return event

    event = run(scenario())
    msg = event.payload
    assert msg.is_self is True
    assert msg.chat_key == "qq:group:111111"
    assert msg.sender_id == "10001"
    assert msg.id == "90001"


def test_self_echo_private_uses_target_id_for_chat_key():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        event = await collect_one(adapter, client, FIXTURES["self_echo_private"])
        return event

    event = run(scenario())
    msg = event.payload
    assert msg.is_self is True
    assert msg.chat_key == "qq:private:333333"


def test_meta_and_notice_events():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        events: list = []
        task = asyncio.create_task(drain_into(adapter, events))
        await asyncio.sleep(0.05)
        await client.send_event(FIXTURES["heartbeat"])
        await client.send_event(FIXTURES["notice_poke"])
        await wait_events(events, 2)
        await client.close()
        await adapter.close()
        await task
        return events

    events = run(scenario())
    assert events[0].type == "meta"
    assert events[1].type == "notice"


# ── send / echo correlation ─────────────────────────────────────────────────

def test_send_group_msg_echo_correlation():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        task = asyncio.create_task(drain(adapter))
        await asyncio.sleep(0.05)
        out = Outgoing(chat_key="qq:group:111111", text="大家好", delivery_key="cy-1:0")
        send_task = asyncio.create_task(adapter.send(out))
        action = await client.next_action()
        assert action["action"] == "send_group_msg"
        assert action["params"]["group_id"] == 111111
        assert action["params"]["message"] == [{"type": "text", "data": {"text": "大家好"}}]
        echo = action["echo"]
        await client.respond(echo, retcode=0, data={"message_id": 90001})
        pid = await send_task
        await client.close()
        await adapter.close()
        await task
        return pid, adapter

    pid, adapter = run(scenario())
    assert pid == "90001"
    assert adapter._delivered.get("90001") == "cy-1:0"


def test_send_private_msg():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        task = asyncio.create_task(drain(adapter))
        await asyncio.sleep(0.05)
        send_task = asyncio.create_task(
            adapter.send(Outgoing(chat_key="qq:private:333333", text="在的"))
        )
        action = await client.next_action()
        assert action["action"] == "send_private_msg"
        assert action["params"]["user_id"] == 333333
        await client.respond(action["echo"], retcode=0, data={"message_id": 90002})
        pid = await send_task
        await client.close()
        await adapter.close()
        await task
        return pid

    assert run(scenario()) == "90002"


def test_send_target_derived_from_chat_key_ignores_group_id():
    """The send target comes from ``Outgoing.chat_key`` ONLY — a stale
    ``group_id`` on the Outgoing never overrides a private chat key."""
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        task = asyncio.create_task(drain(adapter))
        await asyncio.sleep(0.05)
        send_task = asyncio.create_task(
            adapter.send(Outgoing(chat_key="qq:private:333333", text="在的", group_id="111111"))
        )
        action = await client.next_action()
        assert action["action"] == "send_private_msg"
        assert action["params"]["user_id"] == 333333
        assert "group_id" not in action["params"]
        await client.respond(action["echo"], retcode=0, data={"message_id": 90002})
        await send_task
        await client.close()
        await adapter.close()
        await task
        return True

    assert run(scenario()) is True


def test_forward_supports_singular_root_and_rejects_unproven_flat_group():
    async def scenario():
        adapter = make_adapter()
        valid = await adapter._render_forward(
            ChatKey("qq:group:42"),
            {
                "message_type": "group",
                "group_id": 42,
                "user_id": 7,
                "sender": {"nickname": "Alice"},
                "message": [{"type": "text", "data": {"text": "hello"}}],
            },
        )
        invalid = await adapter._render_forward(
            ChatKey("qq:group:42"),
            {
                "message_type": "private",
                "user_id": 7,
                "message": [{"type": "text", "data": {"text": "wrong"}}],
            },
        )
        node_root = await adapter._render_forward(
            ChatKey("qq:group:42"),
            {
                "message": [
                    {
                        "type": "node",
                        "data": {
                            "user_id": 8,
                            "nickname": "Bob",
                            "message": [
                                {"type": "text", "data": {"text": "node"}}
                            ],
                        },
                    }
                ]
            },
        )
        return valid, invalid, node_root

    assert run(scenario()) == ("Alice: hello", None, "Bob: node")


def test_forward_capability_prefetches_verified_chat_scoped_content():
    async def scenario():
        adapter = make_adapter()

        async def fake_call(action: str, **params: Any):
            assert action == "get_forward_msg"
            assert params == {"id": "f-1"}
            return {
                "messages": [
                    {
                        "message_type": "group",
                        "group_id": 42,
                        "user_id": 7,
                        "sender": {"nickname": "Alice"},
                        "message": [{"type": "text", "data": {"text": "hello"}}],
                    }
                ]
            }

        adapter.call = fake_call  # type: ignore[method-assign]
        await adapter._load_forward(ChatKey("qq:group:42"), "f-1")
        return adapter.capabilities, adapter.forwards_for(ChatKey("qq:group:42"))

    capabilities, forwards = run(scenario())
    assert "forward" in capabilities
    assert forwards == {"f-1": "Alice: hello"}


def test_forward_prefetch_rejects_cross_chat_payload():
    async def scenario():
        adapter = make_adapter()

        async def fake_call(action: str, **params: Any):
            return {
                "messages": [
                    {
                        "message_type": "group",
                        "group_id": 99,
                        "user_id": 7,
                        "message": [{"type": "text", "data": {"text": "wrong"}}],
                    }
                ]
            }

        adapter.call = fake_call  # type: ignore[method-assign]
        await adapter._load_forward(ChatKey("qq:group:42"), "f-1")
        return adapter.forwards_for(ChatKey("qq:group:42"))

    assert run(scenario()) == {}


def test_forward_node_form_renders():
    """Real OneBot/NapCat get_forward_msg node payloads
    ({type:'node', data:{user_id,nickname,message}}) render correctly — the
    protocol provides NO per-node group_id, so none is required."""
    async def scenario():
        adapter = make_adapter()

        async def fake_call(action: str, **params: Any):
            assert action == "get_forward_msg"
            assert params == {"id": "f-2"}
            return {
                "messages": [
                    {
                        "type": "node",
                        "data": {
                            "user_id": 7,
                            "nickname": "Alice",
                            "message": [{"type": "text", "data": {"text": "hello"}}],
                        },
                    },
                    {
                        "type": "node",
                        "data": {
                            "user_id": 8,
                            "nickname": "Bob",
                            "message": [{"type": "text", "data": {"text": "world"}}],
                        },
                    },
                ]
            }

        adapter.call = fake_call  # type: ignore[method-assign]
        await adapter._load_forward(ChatKey("qq:group:42"), "f-2")
        return adapter.forwards_for(ChatKey("qq:group:42"))

    assert run(scenario()) == {"f-2": "Alice: hello\nBob: world"}


def test_forward_node_form_content_key():
    """NapCat node payloads carry the message array under ``content``."""
    async def scenario():
        adapter = make_adapter()

        async def fake_call(action: str, **params: Any):
            return {
                "messages": [
                    {
                        "type": "node",
                        "data": {
                            "user_id": 7,
                            "nickname": "Alice",
                            "content": [{"type": "text", "data": {"text": "hi"}}],
                        },
                    }
                ]
            }

        adapter.call = fake_call  # type: ignore[method-assign]
        await adapter._load_forward(ChatKey("qq:group:42"), "f-3")
        return adapter.forwards_for(ChatKey("qq:group:42"))

    assert run(scenario()) == {"f-3": "Alice: hi"}


def test_forward_malformed_payload_rejected():
    """Malformed get_forward_msg payloads (non-list messages, non-dict
    nodes, node data that is not a dict, string message payloads) are
    rejected — the forward stays absent and fails closed."""
    async def scenario():
        adapter = make_adapter()
        cases = [
            {"messages": "not-a-list"},
            {"messages": [None]},
            {"messages": [{"type": "node", "data": "not-a-dict"}]},
            {
                "messages": [
                    {
                        "type": "node",
                        "data": {
                            "user_id": 7,
                            "nickname": "A",
                            "message": "CQ:string",
                        },
                    }
                ]
            },
        ]
        for i, payload in enumerate(cases):
            async def fake_call(action: str, **params: Any):
                return payload

            adapter.call = fake_call  # type: ignore[method-assign]
            await adapter._load_forward(ChatKey("qq:group:42"), f"bad-{i}")
        return adapter.forwards_for(ChatKey("qq:group:42"))

    assert run(scenario()) == {}


def test_forward_cross_chat_node_rejected():
    """A node that EXPLICITLY claims a different group than the trusted
    inbound chat is rejected (cross-chat untrusted), even in node form."""
    async def scenario():
        adapter = make_adapter()

        async def fake_call(action: str, **params: Any):
            return {
                "messages": [
                    {
                        "type": "node",
                        "data": {
                            "user_id": 7,
                            "nickname": "Alice",
                            "group_id": 99,
                            "message": [{"type": "text", "data": {"text": "wrong"}}],
                        },
                    }
                ]
            }

        adapter.call = fake_call  # type: ignore[method-assign]
        await adapter._load_forward(ChatKey("qq:group:42"), "f-4")
        return adapter.forwards_for(ChatKey("qq:group:42"))

    assert run(scenario()) == {}


def test_forward_private_chat_node_form():
    """Private-chat forwards bind to the trusted inbound chat: a node naming
    the other party or the bot itself is accepted without any group scope."""
    async def scenario():
        adapter = make_adapter(self_id="10001")

        async def fake_call(action: str, **params: Any):
            return {
                "messages": [
                    {
                        "type": "node",
                        "data": {
                            "user_id": 333333,
                            "nickname": "小红",
                            "message": [{"type": "text", "data": {"text": "在吗"}}],
                        },
                    },
                    {
                        "type": "node",
                        "data": {
                            "user_id": 10001,
                            "nickname": "麦麦",
                            "message": [{"type": "text", "data": {"text": "在的"}}],
                        },
                    },
                ]
            }

        adapter.call = fake_call  # type: ignore[method-assign]
        await adapter._load_forward(ChatKey("qq:private:333333"), "f-5")
        return adapter.forwards_for(ChatKey("qq:private:333333"))

    assert run(scenario()) == {"f-5": "小红: 在吗\n麦麦: 在的"}


def test_forward_private_node_author_is_not_chat_provenance():
    """A forwarded node's user_id is its author, not the private chat peer;
    the trusted inbound forward reference provides the chat binding."""
    async def scenario():
        adapter = make_adapter(self_id="10001")

        async def fake_call(action: str, **params: Any):
            return {
                "messages": [
                    {
                        "type": "node",
                        "data": {
                            "user_id": 999999,
                            "nickname": "Stranger",
                            "message": [{"type": "text", "data": {"text": "hi"}}],
                        },
                    }
                ]
            }

        adapter.call = fake_call  # type: ignore[method-assign]
        await adapter._load_forward(ChatKey("qq:private:333333"), "f-6")
        return adapter.forwards_for(ChatKey("qq:private:333333"))

    assert run(scenario()) == {"f-6": "Stranger: hi"}


def test_forward_result_capped():
    """The rendered forward body is bounded: a payload longer than the cap is
    truncated, never stored unbounded."""
    async def scenario():
        adapter = make_adapter()

        async def fake_call(action: str, **params: Any):
            return {
                "messages": [
                    {
                        "type": "node",
                        "data": {
                            "user_id": 7,
                            "nickname": "Alice",
                            "message": [{"type": "text", "data": {"text": "x" * 5000}}],
                        },
                    }
                ]
            }

        adapter.call = fake_call  # type: ignore[method-assign]
        await adapter._load_forward(ChatKey("qq:group:42"), "f-7")
        content = adapter.forwards_for(ChatKey("qq:group:42"))["f-7"]
        return len(content)

    assert run(scenario()) == 4096


def test_forward_segment_schedules_chat_scoped_fetch():
    """An inbound forward segment schedules a chat-scoped get_forward_msg
    fetch; the verified content is exposed via forwards_for for that chat
    only (the synchronous tool-facing interface)."""
    async def scenario():
        adapter = make_adapter()

        async def fake_call(action: str, **params: Any):
            assert action == "get_forward_msg"
            assert params == {"id": "f-8"}
            return {
                "messages": [
                    {
                        "type": "node",
                        "data": {
                            "user_id": 7,
                            "nickname": "Alice",
                            "content": [{"type": "text", "data": {"text": "hello"}}],
                        },
                    }
                ]
            }

        adapter.call = fake_call  # type: ignore[method-assign]
        payload = dict(FIXTURES["group_text_image"])
        payload["message"] = [{"type": "forward", "data": {"id": "f-8"}}]
        msg = await adapter._normalize_message(payload)
        for _ in range(100):
            if adapter.forwards_for(msg.chat_key):
                break
            await asyncio.sleep(0.01)
        return msg.chat_key, adapter.forwards_for(msg.chat_key)

    chat_key, forwards = run(scenario())
    assert chat_key == "qq:group:111111"
    assert forwards == {"f-8": "Alice: hello"}


def test_send_invalid_target_rejected():
    """Malformed chat keys (empty/non-numeric/negative ids, wrong platform,
    wrong kind, extra parts) are rejected safely as AdapterError — never a
    raw ValueError or a misdirected send."""
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        task = asyncio.create_task(drain(adapter))
        await asyncio.sleep(0.05)
        for bad_key in (
            "qq:group:", "qq:group:abc", "qq:group:-1", "qq:group:1:2",
            "tg:group:1", "qq:channel:1", "qq:private:",
        ):
            with pytest.raises(AdapterError):
                await adapter.send(Outgoing(chat_key=bad_key, text="hi"))
        await client.close()
        await adapter.close()
        await task
        return True

    assert run(scenario()) is True


# ── real self id (configured and/or learned from inbound events) ─────────────

def test_self_id_learned_from_inbound_event():
    async def scenario():
        adapter = make_adapter()  # no configured self_id
        assert adapter.self_id is None
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        event = await collect_one(adapter, client, FIXTURES["group_text_image"])
        return event, adapter

    event, adapter = run(scenario())
    assert adapter.self_id == "10001"  # learned from the event
    assert event.payload.sender_id == "222222"


def test_self_id_from_constructor_and_config():
    adapter = make_adapter(self_id="10001")
    assert adapter.self_id == "10001"
    cfg = OneBotConfig(host="127.0.0.1", port=0, heartbeat_timeout_s=None, self_id="10001")
    adapter2 = OneBotAdapter(config=cfg, clock=VirtualClock(), normalize_media=False)
    assert adapter2.self_id == "10001"


def test_learned_self_id_enables_direct_at_detection():
    """A message @-mentioning the LEARNED self id is recognized as a direct
    mention (the gate's direct-@ trigger reads the same identity)."""
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        event = await collect_one(adapter, client, FIXTURES["group_at"])
        return event, adapter

    event, adapter = run(scenario())
    assert adapter.self_id == "10001"
    assert "10001" in event.payload.mentions


def test_send_retcode_fallback_returns_none_and_retains_payload():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        task = asyncio.create_task(drain(adapter))
        await asyncio.sleep(0.05)
        send_task = asyncio.create_task(
            adapter.send(Outgoing(chat_key="qq:group:111111", text="hi", delivery_key="cy-1:7"))
        )
        action = await client.next_action()
        await client.respond(action["echo"], retcode=-1, data=None)
        pid = await send_task
        await client.close()
        await adapter.close()
        await task
        return pid, adapter

    pid, adapter = run(scenario())
    # Ambiguous ack: no real platform id — the outbox writes a synthetic
    # local echo that a later real self echo reconciles (never a duplicate).
    assert pid is None
    # The exact sent payload + delivery key are retained under a stable
    # local id for later real-echo reconciliation.
    assert adapter._delivered.get("onebot:local:cy-1:7") == "cy-1:7"
    payload = adapter._sent_payload.get("onebot:local:cy-1:7")
    assert payload is not None
    assert payload[0] == "qq:group:111111"
    assert payload[1] == "hi"


def test_fallback_echo_reconciles_real_id():
    """A real self echo matching a retained fallback send (same chat/self/
    payload) binds its real id to the delivery key and correlates the exact
    sent payload — the durable side then updates the synthetic row instead
    of inserting a duplicate context message."""
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        events: list = []
        task = asyncio.create_task(drain_into(adapter, events))
        await asyncio.sleep(0.05)
        send_task = asyncio.create_task(
            adapter.send(Outgoing(chat_key="qq:group:111111", text="大家好", delivery_key="cy-1:7"))
        )
        action = await client.next_action()
        await client.respond(action["echo"], retcode=-1, data=None)
        pid = await send_task
        assert pid is None
        # the platform echoes our sent message back with a REAL id
        await client.send_event(FIXTURES["self_echo_group"])
        await wait_events(events, 1)
        echo = events[0].payload
        key = adapter.delivery_key_for(echo)
        await client.close()
        await adapter.close()
        await task
        return pid, echo, key, adapter

    pid, echo, key, adapter = run(scenario())
    assert pid is None
    assert echo.is_self is True
    assert echo.text == "大家好"  # correlated to the exact sent payload
    assert key == "cy-1:7"  # the real id was bound to the delivery key
    assert adapter._delivered.get("90001") == "cy-1:7"


def test_fallback_echo_payload_mismatch_does_not_bind():
    """A real self echo with a DIFFERENT payload never binds: arbitrary echo
    data is not trusted."""
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        events: list = []
        task = asyncio.create_task(drain_into(adapter, events))
        await asyncio.sleep(0.05)
        send_task = asyncio.create_task(
            adapter.send(Outgoing(chat_key="qq:group:111111", text="大家好", delivery_key="cy-1:7"))
        )
        action = await client.next_action()
        await client.respond(action["echo"], retcode=-1, data=None)
        await send_task
        echo_payload = dict(FIXTURES["self_echo_group"])
        echo_payload["message"] = [{"type": "text", "data": {"text": "别的内容"}}]
        await client.send_event(echo_payload)
        await wait_events(events, 1)
        echo = events[0].payload
        key = adapter.delivery_key_for(echo)
        await client.close()
        await adapter.close()
        await task
        return key, adapter

    key, adapter = run(scenario())
    assert key is None  # payload mismatch: never bind
    assert adapter._delivered.get("90001") is None


def test_fallback_echo_ambiguous_does_not_bind():
    """Two retained fallback sends with the SAME payload but different keys
    make the match ambiguous — the echo is never bound."""
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        events: list = []
        task = asyncio.create_task(drain_into(adapter, events))
        await asyncio.sleep(0.05)
        for key in ("cy-1:7", "cy-2:8"):
            send_task = asyncio.create_task(
                adapter.send(Outgoing(chat_key="qq:group:111111", text="大家好", delivery_key=key))
            )
            action = await client.next_action()
            await client.respond(action["echo"], retcode=-1, data=None)
            await send_task
        await client.send_event(FIXTURES["self_echo_group"])
        await wait_events(events, 1)
        echo = events[0].payload
        key = adapter.delivery_key_for(echo)
        await client.close()
        await adapter.close()
        await task
        return key, adapter

    key, adapter = run(scenario())
    assert key is None  # ambiguous: never bind
    assert adapter._delivered.get("90001") is None


def test_delivery_key_for_resolves_correlated_self_echo():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        events: list = []
        task = asyncio.create_task(drain_into(adapter, events))
        await asyncio.sleep(0.05)
        send_task = asyncio.create_task(
            adapter.send(Outgoing(chat_key="qq:group:111111", text="大家好", delivery_key="cy-1:0"))
        )
        action = await client.next_action()
        await client.respond(action["echo"], retcode=0, data={"message_id": 90001})
        pid = await send_task
        # now the platform echoes our sent message back
        await client.send_event(FIXTURES["self_echo_group"])
        await wait_events(events, 1)
        echo_msg = events[0].payload
        key = adapter.delivery_key_for(echo_msg)
        await client.close()
        await adapter.close()
        await task
        return pid, echo_msg, key

    pid, echo_msg, key = run(scenario())
    assert pid == "90001"
    assert echo_msg.is_self is True
    assert echo_msg.text == "大家好"  # correlated to the exact sent payload
    assert key == "cy-1:0"


def test_call_escape_hatch():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        task = asyncio.create_task(drain(adapter))
        await asyncio.sleep(0.05)
        call_task = asyncio.create_task(adapter.call("get_group_info", group_id=111111))
        action = await client.next_action()
        assert action["action"] == "get_group_info"
        await client.respond(action["echo"], retcode=0, data={"group_id": 111111, "name": "群A"})
        data = await call_task
        await client.close()
        await adapter.close()
        await task
        return data

    assert run(scenario()) == {"group_id": 111111, "name": "群A"}


def test_send_without_connection_raises_transient():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        with pytest.raises(TransientError):
            await adapter.send(Outgoing(chat_key="qq:group:111111", text="hi"))
        await adapter.close()

    run(scenario())


# ── reconnect / heartbeat / close ───────────────────────────────────────────

def test_reconnect_after_client_disconnect():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client1 = await FakeOneBot(port).connect()
        events: list = []
        task = asyncio.create_task(drain_into(adapter, events))
        await asyncio.sleep(0.05)
        await client1.send_event(FIXTURES["group_text_image"])
        await wait_events(events, 1)
        await client1.close()
        client2 = await FakeOneBot(port).connect()
        await asyncio.sleep(0.05)
        await client2.send_event(FIXTURES["private_text"])
        await wait_events(events, 2)
        await client2.close()
        await adapter.close()
        await task
        return [e.payload.chat_key for e in events]

    keys = run(scenario())
    assert keys == ["qq:group:111111", "qq:private:333333"]


class FakeConn:
    """A fake websocket connection for watchdog tests: reports OPEN, answers
    or refuses pings, and records close()."""

    def __init__(self, pong: bool = True):
        self.state = State.OPEN
        self._pong = pong
        self.closed = False
        self.close_reason: str | None = None

    async def ping(self):
        fut = asyncio.get_running_loop().create_future()
        if self._pong:
            fut.set_result(None)
        return fut

    async def close(self, code=None, reason=None):
        if not self.closed:
            self.close_reason = reason
        self.closed = True
        self.state = State.CLOSED


def test_watchdog_pings_and_keeps_healthy_quiet_link():
    """A healthy quiet link that ANSWERS the watchdog's ping stays connected
    — no reconnect storm from a merely quiet (not dead) connection."""
    async def scenario():
        clock = VirtualClock(auto_advance=False)
        cfg = OneBotConfig(host="127.0.0.1", port=0, heartbeat_timeout_s=10.0)
        adapter = OneBotAdapter(config=cfg, clock=clock, normalize_media=False)
        await adapter.connect()
        conn = FakeConn(pong=True)
        adapter._conn = conn
        adapter._note_activity()
        # advance past the heartbeat timeout: the watchdog pings, the pong
        # answers, and the healthy quiet link stays connected
        for _ in range(6):
            clock.advance(4.0)
            await asyncio.sleep(0.01)
        await adapter.close()
        return conn.close_reason

    assert run(scenario()) != "heartbeat timeout"  # never watchdog-dropped


def test_watchdog_drops_link_that_does_not_pong():
    """A quiet connection that does NOT answer the watchdog's ping is
    dropped (the pong is awaited before the drop)."""
    async def scenario():
        clock = VirtualClock(auto_advance=False)
        cfg = OneBotConfig(
            host="127.0.0.1",
            port=0,
            heartbeat_timeout_s=10.0,
            ping_timeout_s=0.01,
        )
        adapter = OneBotAdapter(config=cfg, clock=clock, normalize_media=False)
        await adapter.connect()
        conn = FakeConn(pong=False)
        adapter._conn = conn
        adapter._note_activity()
        for _ in range(6):
            clock.advance(4.0)
            await asyncio.sleep(0.02)
        await adapter.close()
        return conn.close_reason

    assert run(scenario()) == "heartbeat timeout"


def test_identity_conflicting_frame_is_closed_and_not_normalized():
    async def scenario():
        adapter = make_adapter(self_id="10001")
        conn = FakeConn(pong=True)
        adapter._conn = conn
        adapter._connection_generation = 1
        raw = {
            "self_id": 20002,
            "post_type": "message",
            "message_type": "group",
            "message_id": 1,
            "group_id": 123,
            "user_id": 9,
            "message": [{"type": "text", "data": {"text": "bad"}}],
        }
        event = await adapter._handle_frame(orjson.dumps(raw), generation=1)
        return event, adapter._identity_conflict, conn.close_reason

    event, conflict, reason = run(scenario())
    assert event is None
    assert conflict is True
    assert reason == "onebot self_id mismatch"


def test_close_is_idempotent_and_cancels_tasks():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        await adapter.close()
        await adapter.close()
        return adapter._closed, adapter._watchdog_task, adapter._reconnect_task

    closed, watchdog, reconnect = run(scenario())
    assert closed is True
    assert watchdog is None  # disabled in this config
    assert reconnect is None


def test_forward_mode_connects_and_receives():
    async def scenario():
        conns: list = []
        async def handler(ws):
            conns.append(ws)
            async for _ in ws:
                pass
        server = await ws_serve(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        cfg = OneBotConfig(
            mode="ws", host="127.0.0.1", port=port,
            path="/onebot/v11/ws", heartbeat_timeout_s=None,
        )
        adapter = OneBotAdapter(config=cfg, clock=VirtualClock(), normalize_media=False)
        await adapter.connect()
        await wait_connected(adapter)
        events: list = []
        task = asyncio.create_task(drain_into(adapter, events))
        await asyncio.sleep(0.05)
        await conns[0].send(orjson.dumps(FIXTURES["group_text_image"]))
        await wait_events(events, 1)
        server.close()
        await server.wait_closed()
        await adapter.close()
        await task
        return events[0].payload.chat_key

    assert run(scenario()) == "qq:group:111111"


def test_forward_mode_reconnects_after_server_restart():
    async def scenario():
        conns: list = []
        async def handler(ws):
            conns.append(ws)
            async for _ in ws:
                pass
        server = await ws_serve(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        cfg = OneBotConfig(
            mode="ws", host="127.0.0.1", port=port,
            path="/onebot/v11/ws", heartbeat_timeout_s=None,
            reconnect_base_s=0.01, reconnect_max_s=0.05,
        )
        adapter = OneBotAdapter(config=cfg, clock=VirtualClock(), normalize_media=False)
        await adapter.connect()
        await wait_connected(adapter)
        server.close()
        await server.wait_closed()
        await wait_disconnected(adapter)
        server2 = await ws_serve(handler, "127.0.0.1", port)
        await wait_connected(adapter)
        server2.close()
        await server2.wait_closed()
        await adapter.close()
        return True

    assert run(scenario()) is True


# ── media normalization in the adapter ──────────────────────────────────────

def test_adapter_normalizes_image_segment_media():
    from PIL import Image
    import io

    def make_png() -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
        return buf.getvalue()

    class FakeFetcher:
        def __init__(self, data):
            self.data = data
            self.calls = []

        async def fetch(self, url):
            self.calls.append(url)
            return self.data

    async def scenario():
        fetcher = FakeFetcher(make_png())
        media = MediaStore(fetcher=fetcher)
        adapter = make_adapter(media=media, normalize_media=True)
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        events: list = []
        task = asyncio.create_task(drain_into(adapter, events))
        await asyncio.sleep(0.05)
        # first frame: the download is SCHEDULED in the background, never
        # awaited inline — the segment has no media yet
        await client.send_event(FIXTURES["group_text_image"])
        await wait_events(events, 1)
        assert events[0].payload.segments[1].data.get("media") is None
        # wait for the background prefetch to finish and cache the asset
        for _ in range(100):
            if media.cached("https://example.com/abc.jpg") is not None:
                break
            await asyncio.sleep(0.01)
        # a later frame with the same URL gets the cached media
        await client.send_event(FIXTURES["group_text_image"])
        await wait_events(events, 2)
        img_seg = events[1].payload.segments[1]
        assert img_seg.kind == "image"
        assert "media" in img_seg.data
        assert img_seg.data["media"]["mime"] == "image/jpeg"
        assert img_seg.data["media"]["width"] == 8
        assert img_seg.data["media"]["height"] == 8
        assert img_seg.data["media"]["data_url"].startswith("data:image/jpeg;base64,")
        assert fetcher.calls == ["https://example.com/abc.jpg"]
        await client.close()
        await adapter.close()
        await task
        return True

    assert run(scenario()) is True


def test_slow_media_does_not_block_following_frame():
    """A slow media download is scheduled in the background: the frame/event
    loop never awaits it, so the following frame is processed immediately."""
    from PIL import Image
    import io

    def make_png() -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
        return buf.getvalue()

    class SlowFetcher:
        def __init__(self, data, release):
            self.data = data
            self.release = release
            self.calls = []

        async def fetch(self, url):
            self.calls.append(url)
            await self.release.wait()
            return self.data

    async def scenario():
        release = asyncio.Event()
        fetcher = SlowFetcher(make_png(), release)
        media = MediaStore(fetcher=fetcher)
        adapter = make_adapter(media=media, normalize_media=True)
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        events: list = []
        task = asyncio.create_task(drain_into(adapter, events))
        await asyncio.sleep(0.05)
        # frame 1: image — the download is scheduled, not awaited
        await client.send_event(FIXTURES["group_text_image"])
        await wait_events(events, 1)
        # frame 2: text — must arrive while the image download is still pending
        await client.send_event(FIXTURES["private_text"])
        await wait_events(events, 2)
        assert not release.is_set()  # the download never blocked the loop
        assert events[0].payload.segments[1].data.get("media") is None
        # release the download; the background task caches the asset
        release.set()
        for _ in range(100):
            if media.cached("https://example.com/abc.jpg") is not None:
                break
            await asyncio.sleep(0.01)
        assert fetcher.calls == ["https://example.com/abc.jpg"]
        await client.close()
        await adapter.close()
        await task
        return True

    assert run(scenario()) is True


# ── Gate 4 final: background receiver / security / readiness / media ─────────

class FakeUpgradeConn:
    """A fake websocket connection for upgrade-path rejection tests: records
    the close code/reason and never opens."""

    def __init__(self, path: str = "/onebot/v11/ws", headers: dict | None = None):
        self.request = SimpleNamespace(path=path, headers=headers or {})
        self.closed: tuple | None = None

    async def close(self, code=None, reason=None):
        self.closed = (code, reason)


def test_send_resolves_via_background_receiver_before_events():
    """A startup send() resolves through the background receiver's internal
    queue even before events() is ever iterated (startup recovery)."""
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        # NO events() iteration here — the persistent background receiver must
        # read the echo response and resolve the pending future on its own.
        send_task = asyncio.create_task(
            adapter.send(
                Outgoing(chat_key="qq:group:111111", text="大家好", delivery_key="cy-1:0")
            )
        )
        action = await client.next_action()  # auto-answers the probe
        assert action["action"] == "send_group_msg"
        await client.respond(action["echo"], retcode=0, data={"message_id": 90001})
        pid = await asyncio.wait_for(send_task, timeout=2.0)
        await client.close()
        await adapter.close()
        return pid

    assert run(scenario()) == "90001"


def test_reverse_origin_upgrade_rejected():
    """A browser-style upgrade carrying an Origin header is rejected at the
    handshake (browsers always send Origin; legitimate OneBot clients do not)."""
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        ws = await ws_connect(
            f"ws://127.0.0.1:{port}/onebot/v11/ws",
            origin="http://evil.example.com",
        )
        # the server rejects the upgrade: the connection is closed
        with pytest.raises(ConnectionClosed):
            await ws.recv()
        await adapter.close()
        return True

    assert run(scenario()) is True


def test_reverse_requires_message_array_negotiation():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        ws = await ws_connect(f"ws://127.0.0.1:{port}/onebot/v11/ws")
        with pytest.raises(ConnectionClosed):
            await ws.recv()
        await adapter.close()

    run(scenario())


def test_forward_uri_requests_message_array_format():
    adapter = make_adapter(
        config=OneBotConfig(mode="ws", host="127.0.0.1", port=3001)
    )
    assert "message_format=array" in adapter._ws_uri()


def test_string_message_payload_is_rejected_not_normalized():
    async def scenario():
        adapter = make_adapter()
        with pytest.raises(AdapterError, match="message_format=array"):
            await adapter._normalize_segments("[CQ:at,qq=10001]")

    run(scenario())


def test_self_id_mismatch_fails_current_generation_closed():
    adapter = make_adapter(self_id="10001")
    adapter._connection_generation = 1
    adapter._generation_self_id = "10001"
    adapter._probe_ok = True
    adapter._learn_self_id({"self_id": 20002}, generation=1)
    assert adapter._identity_conflict is True
    assert adapter.ready is False


def test_reconnect_forces_current_generation_identity_and_probe():
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        first = await FakeOneBot(port).connect()
        raw = await first.ws.recv()
        probe = orjson.loads(raw)
        await first.respond(probe["echo"], retcode=0, data={"user_id": 10001})
        await asyncio.sleep(0.05)
        assert adapter.ready is True
        await first.close()
        await wait_disconnected(adapter)
        second = await FakeOneBot(port).connect()
        assert adapter.ready is False
        raw = await second.ws.recv()
        probe = orjson.loads(raw)
        await second.respond(probe["echo"], retcode=0, data={"user_id": 10001})
        await asyncio.sleep(0.05)
        assert adapter.ready is True
        await second.close()
        await adapter.close()

    run(scenario())


def test_events_drop_stale_generation_queue_entries():
    async def scenario():
        adapter = make_adapter()
        adapter._connection_generation = 2
        stale = AdapterEvent(type="meta", payload={"generation": 1})
        current = AdapterEvent(type="meta", payload={"generation": 2})
        adapter._rx_queue.put_nowait((1, stale))
        adapter._rx_queue.put_nowait((2, current))
        event = await adapter.events().__anext__()
        await adapter.close()
        return event

    assert run(scenario()).payload == {"generation": 2}


def test_zero_retcode_without_message_id_uses_fallback_candidate():
    adapter = make_adapter()
    out = Outgoing(
        chat_key=ChatKey("qq:group:111111"), text="hello", delivery_key="dispatch:2:0"
    )
    local = adapter._fallback_local_id(out)
    assert adapter._send_result({"retcode": 0, "data": {}}, out, provisional=local) is None
    assert adapter._delivered[local] == "dispatch:2:0"


def _unvalidated_cfg(**kw) -> OneBotConfig:
    """Build a OneBotConfig WITHOUT running __post_init__ validation (for
    testing the upgrade-path defense in depth that config load already
    rejects)."""
    cfg = OneBotConfig.__new__(OneBotConfig)
    defaults = dict(
        mode="reverse_ws",
        scheme="ws",
        host="127.0.0.1",
        port=0,
        path="/onebot/v11/ws",
        access_token=None,
        self_id=None,
        action_timeout_s=10.0,
        heartbeat_timeout_s=None,
        ping_timeout_s=10.0,
        reconnect_base_s=3.0,
        reconnect_max_s=60.0,
        media_concurrency=4,
    )
    defaults.update(kw)
    cfg.__dict__.update(defaults)
    return cfg


def test_reverse_non_loopback_bind_requires_auth_at_upgrade():
    """Defense in depth: even if config validation is bypassed, an upgrade on
    a non-loopback bind without a configured token is rejected."""
    async def scenario():
        adapter = OneBotAdapter(
            config=_unvalidated_cfg(host="0.0.0.0", access_token=None),
            clock=VirtualClock(),
            normalize_media=False,
        )
        await adapter.connect()
        conn = FakeUpgradeConn()
        await adapter._on_client(conn)
        assert conn.closed is not None
        assert conn.closed[1] == "auth required for non-loopback bind"
        await adapter.close()
        return True

    assert run(scenario()) is True


def test_unauthenticated_takeover_rejected():
    """An unauthenticated replacement can never take over a connected
    legitimate session: the live session keeps working after the attempt."""
    async def scenario():
        cfg = OneBotConfig(
            host="127.0.0.1", port=0, heartbeat_timeout_s=None, access_token="secret"
        )
        adapter = OneBotAdapter(config=cfg, clock=VirtualClock(), normalize_media=False)
        await adapter.connect()
        port = adapter_port(adapter)
        legit = await FakeOneBot(port, token="secret").connect()
        events: list = []
        task = asyncio.create_task(drain_into(adapter, events))
        await asyncio.sleep(0.05)
        # unauthenticated replacement attempt (no Authorization header)
        bad = FakeUpgradeConn()
        await adapter._on_client(bad)
        assert bad.closed is not None
        assert bad.closed[1] == "unauthorized"
        # the legitimate session is undisturbed and still delivers events
        await legit.send_event(FIXTURES["private_text"])
        await wait_events(events, 1)
        assert events[0].payload.chat_key == "qq:private:333333"
        await legit.close()
        await adapter.close()
        await task
        return True

    assert run(scenario()) is True


def test_ready_requires_login_identity_and_probe():
    """A successful get_login_info response binds identity to this connection
    generation and completes readiness; an open socket alone is not ready."""
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        # socket open, but no lifecycle and no probe yet
        assert adapter.connected is True
        assert adapter.ready is False
        # Answering get_login_info also proves the OneBot identity for this
        # exact connection generation.
        raw = await asyncio.wait_for(client.ws.recv(), 2.0)
        probe = orjson.loads(raw)
        assert probe["action"] == "get_login_info"
        await client.respond(probe["echo"], retcode=0, data={"user_id": 10001})
        await asyncio.sleep(0.05)
        assert adapter._probe_ok is True
        assert adapter.ready is True
        await client.close()
        await adapter.close()
        return True

    assert run(scenario()) is True


def test_readiness_observability_logs_secret_free_generation_and_success(caplog):
    async def scenario():
        cfg = OneBotConfig(
            host="127.0.0.1",
            port=0,
            access_token="super-secret-token",
            heartbeat_timeout_s=None,
        )
        adapter = make_adapter(config=cfg)
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port, token="super-secret-token").connect()
        raw = await asyncio.wait_for(client.ws.recv(), 2.0)
        probe = orjson.loads(raw)
        await client.respond(probe["echo"], retcode=0, data={"user_id": 10001})
        await asyncio.sleep(0.05)
        assert adapter.ready is True
        await client.close()
        await adapter.close()

    caplog.set_level(logging.INFO, logger="pretender.onebot")
    with capture_onebot_logs(caplog):
        run(scenario())
    messages = [record.getMessage() for record in caplog.records]
    assert any("connection generation adopted/reset" in message for message in messages)
    assert sum("readiness established" in message for message in messages) == 1
    assert any("generation=1" in message for message in messages)
    assert "super-secret-token" not in caplog.text
    assert "ws://127.0.0.1" not in caplog.text
    assert "10001" not in caplog.text


def test_readiness_observability_logs_probe_timeout_and_not_ready(caplog):
    async def scenario():
        cfg = OneBotConfig(
            host="127.0.0.1",
            port=0,
            action_timeout_s=0.01,
            heartbeat_timeout_s=None,
        )
        adapter = make_adapter(config=cfg)
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        # Leave get_login_info unanswered so the readiness task times out.
        await asyncio.sleep(0.05)
        assert adapter.ready is False
        await wait_disconnected(adapter)
        assert client.ws.close_code == CloseCode.TRY_AGAIN_LATER
        await client.close()
        await adapter.close()

    caplog.set_level(logging.INFO, logger="pretender.onebot")
    with capture_onebot_logs(caplog):
        run(scenario())
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "readiness probe timed out" in message
        and "generation=1" in message
        and "not-ready" in message
        and "waits for reconnect" in message
        for message in messages
    )
    assert "get_login_info" not in caplog.text


def test_readiness_observability_logs_identity_mismatch_without_ids(caplog):
    adapter = make_adapter(self_id="configured-secret-id")
    adapter._connection_generation = 1
    caplog.set_level(logging.INFO, logger="pretender.onebot")

    with capture_onebot_logs(caplog):
        adapter._learn_self_id({"self_id": "unexpected-id"}, generation=1)

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        record.levelno >= logging.WARNING and "identity mismatch" in record.getMessage()
        for record in caplog.records
    )
    assert "configured-secret-id" not in caplog.text
    assert "unexpected-id" not in caplog.text


def test_failed_probe_closes_generation_and_fresh_probe_can_ready(caplog):
    async def scenario():
        cfg = OneBotConfig(
            host="127.0.0.1",
            port=0,
            access_token="super-secret-token",
            heartbeat_timeout_s=None,
        )
        adapter = make_adapter(config=cfg)
        await adapter.connect()
        port = adapter_port(adapter)

        first = await FakeOneBot(port, token="super-secret-token").connect()
        raw = await asyncio.wait_for(first.ws.recv(), 2.0)
        failed_probe = orjson.loads(raw)
        await first.respond(
            failed_probe["echo"],
            retcode=1,
            data={"user_id": "wrong-account", "payload": "do-not-log"},
        )
        await wait_disconnected(adapter)
        first_close_code = first.ws.close_code

        second = await FakeOneBot(port, token="super-secret-token").connect()
        raw = await asyncio.wait_for(second.ws.recv(), 2.0)
        fresh_probe = orjson.loads(raw)
        await second.respond(fresh_probe["echo"], retcode=0, data={"user_id": 10001})
        await asyncio.sleep(0.05)
        ready = adapter.ready
        generation = adapter.generation
        await second.close()
        await adapter.close()
        return first_close_code, generation, ready

    caplog.set_level(logging.INFO, logger="pretender.onebot")
    with capture_onebot_logs(caplog):
        first_close_code, generation, ready = run(scenario())

    assert first_close_code == CloseCode.TRY_AGAIN_LATER
    assert generation == 2
    assert ready is True
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "readiness probe failed" in message
        and "generation=1" in message
        and "not-ready" in message
        and "waits for reconnect" in message
        for message in messages
    )
    assert any("readiness established" in message and "generation=2" in message for message in messages)
    assert "super-secret-token" not in caplog.text
    assert "ws://127.0.0.1" not in caplog.text
    assert "wrong-account" not in caplog.text
    assert "do-not-log" not in caplog.text
    assert "10001" not in caplog.text


def test_ready_self_id_event_counts_as_lifecycle():
    """A self_id-bearing inbound event (not just a lifecycle meta-event)
    satisfies the lifecycle/self_id half of readiness."""
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        raw = await asyncio.wait_for(client.ws.recv(), 2.0)
        probe = orjson.loads(raw)
        # A probe without user_id is insufficient; a normal event carrying
        # self_id supplies the remaining identity proof.
        await client.respond(probe["echo"], retcode=0, data={})
        await asyncio.sleep(0.05)
        assert adapter.ready is False
        # a normal message event carries self_id -> lifecycle half satisfied
        await client.send_event(FIXTURES["private_text"])
        await asyncio.sleep(0.05)
        assert adapter.ready is True
        await client.close()
        await adapter.close()
        return True

    assert run(scenario()) is True


def test_media_concurrency_capped():
    """Background media downloads are bounded by the concurrency semaphore:
    with cap 2, only 2 fetches run at once; the third waits."""
    from PIL import Image
    import io

    def make_png() -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
        return buf.getvalue()

    class SlowFetcher:
        def __init__(self, data, release):
            self.data = data
            self.release = release
            self.calls: list[str] = []

        async def fetch(self, url):
            self.calls.append(url)
            await self.release.wait()
            return self.data

    async def scenario():
        release = asyncio.Event()
        fetcher = SlowFetcher(make_png(), release)
        media = MediaStore(fetcher=fetcher)
        cfg = OneBotConfig(
            host="127.0.0.1", port=0, heartbeat_timeout_s=None, media_concurrency=2
        )
        adapter = OneBotAdapter(
            config=cfg, clock=VirtualClock(), media=media, normalize_media=True
        )
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        events: list = []
        task = asyncio.create_task(drain_into(adapter, events))
        await asyncio.sleep(0.05)
        for i in range(3):
            payload = dict(FIXTURES["group_text_image"])
            payload["message"] = [
                {"type": "text", "data": {"text": f"m{i}"}},
                {
                    "type": "image",
                    "data": {"file": f"{i}.jpg", "url": f"https://example.com/{i}.jpg"},
                },
            ]
            await client.send_event(payload)
        await wait_events(events, 3)
        await asyncio.sleep(0.05)
        # cap 2: exactly two downloads in flight, the third waits on the semaphore
        assert len(fetcher.calls) == 2
        release.set()
        for _ in range(100):
            if len(fetcher.calls) == 3:
                break
            await asyncio.sleep(0.01)
        assert len(fetcher.calls) == 3
        await client.close()
        await adapter.close()
        await task
        return True

    assert run(scenario()) is True


def test_media_task_cancelled_when_dropped():
    """The in-flight media map is bounded: scheduling beyond max_remember
    CANCELS the oldest live task, so downloads never run unbounded."""
    class BlockingFetcher:
        def __init__(self, release):
            self.release = release
            self.calls: list[str] = []

        async def fetch(self, url):
            self.calls.append(url)
            await self.release.wait()
            return b"\xff\xd8\xff\xe0"

    async def scenario():
        release = asyncio.Event()
        media = MediaStore(fetcher=BlockingFetcher(release))
        adapter = make_adapter(media=media, normalize_media=True, max_remember=2)
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        events: list = []
        task = asyncio.create_task(drain_into(adapter, events))
        await asyncio.sleep(0.05)

        def img_payload(i: int) -> dict:
            payload = dict(FIXTURES["group_text_image"])
            payload["message"] = [
                {
                    "type": "image",
                    "data": {"file": f"{i}.jpg", "url": f"https://example.com/{i}.jpg"},
                },
            ]
            return payload

        await client.send_event(img_payload(0))
        await wait_events(events, 1)
        await asyncio.sleep(0.05)
        first_task = adapter._media_tasks.get("https://example.com/0.jpg")
        assert first_task is not None
        await client.send_event(img_payload(1))
        await client.send_event(img_payload(2))
        await wait_events(events, 3)
        await asyncio.sleep(0.05)
        assert len(adapter._media_tasks) == 2  # bounded
        assert first_task.cancelled() is True  # dropped entry was cancelled
        release.set()
        await client.close()
        await adapter.close()
        await task
        return True

    assert run(scenario()) is True


def test_repeated_identical_fallback_echo_correlation():
    """A consumed local fallback candidate is retired after a successful
    real-echo bind, so a later identical ambiguous send reconciles against
    the remaining candidate instead of staying ambiguous forever."""
    async def scenario():
        adapter = make_adapter()
        await adapter.connect()
        port = adapter_port(adapter)
        client = await FakeOneBot(port).connect()
        events: list = []
        task = asyncio.create_task(drain_into(adapter, events))
        await asyncio.sleep(0.05)
        # first ambiguous send
        send_task = asyncio.create_task(
            adapter.send(
                Outgoing(chat_key="qq:group:111111", text="大家好", delivery_key="cy-1:7")
            )
        )
        action = await client.next_action()
        await client.respond(action["echo"], retcode=-1, data=None)
        assert await send_task is None
        # real echo 1 binds cy-1:7 and RETIRES the local candidate
        await client.send_event(FIXTURES["self_echo_group"])
        await wait_events(events, 1)
        assert adapter.delivery_key_for(events[0].payload) == "cy-1:7"
        assert adapter._delivered.get("onebot:local:cy-1:7") is None  # retired
        # second identical ambiguous send
        send_task2 = asyncio.create_task(
            adapter.send(
                Outgoing(chat_key="qq:group:111111", text="大家好", delivery_key="cy-2:8")
            )
        )
        action2 = await client.next_action()
        await client.respond(action2["echo"], retcode=-1, data=None)
        assert await send_task2 is None
        # real echo 2 (different message id) now reconciles cy-2:8
        echo2 = dict(FIXTURES["self_echo_group"])
        echo2["message_id"] = 90002
        await client.send_event(echo2)
        await wait_events(events, 2)
        assert adapter.delivery_key_for(events[1].payload) == "cy-2:8"
        await client.close()
        await adapter.close()
        await task
        return True

    assert run(scenario()) is True


def test_busy_reverse_port_reports_the_address_and_the_fix():
    """A taken port is the most common first-run failure; it must not surface
    as a bare errno."""

    async def scenario() -> None:
        holder = await ws_serve(lambda _conn: None, "127.0.0.1", 0)
        port = holder.sockets[0].getsockname()[1]
        adapter = make_adapter(
            config=OneBotConfig(host="127.0.0.1", port=port, heartbeat_timeout_s=None)
        )
        try:
            with pytest.raises(AdapterError) as caught:
                await adapter.connect()
            message = str(caught.value)
            assert f"127.0.0.1:{port}" in message
            assert "adapter.onebot.port" in message
        finally:
            holder.close()
            await holder.wait_closed()

    run(scenario())


# ── the platform account's own login state ──────────────────────────────────
#
# ``status.online`` in a heartbeat is the ACCOUNT's session, not the
# transport. They diverge in the worst way: NapCat keeps the socket open,
# keeps heartbeating and keeps answering get_login_info — so the adapter
# reaches ready and everything looks healthy — while the account is logged
# out and no message can arrive. Production sat like that for 20 hours,
# indistinguishable from a bot that simply had nothing to say.

def _heartbeat(online):
    return {
        "post_type": "meta_event",
        "meta_event_type": "heartbeat",
        "self_id": 10001,
        "status": {"online": online, "good": True},
        "interval": 30000,
    }


def test_offline_heartbeat_is_reported(caplog):
    adapter = make_adapter()
    caplog.set_level(logging.INFO, logger="pretender.onebot")
    with capture_onebot_logs(caplog):
        adapter._note_platform_online(_heartbeat(False))
    assert any("OFFLINE" in r.getMessage() for r in caplog.records)
    assert adapter._platform_online is False


def test_only_transitions_are_logged(caplog):
    """A heartbeat every 30 s must not become a log flood."""
    adapter = make_adapter()
    caplog.set_level(logging.INFO, logger="pretender.onebot")
    with capture_onebot_logs(caplog):
        for _ in range(5):
            adapter._note_platform_online(_heartbeat(False))
    assert len([r for r in caplog.records if "OFFLINE" in r.getMessage()]) == 1


def test_coming_back_online_is_reported(caplog):
    adapter = make_adapter()
    adapter._note_platform_online(_heartbeat(False))
    caplog.set_level(logging.INFO, logger="pretender.onebot")
    with capture_onebot_logs(caplog):
        adapter._note_platform_online(_heartbeat(True))
    assert any("is online" in r.getMessage() for r in caplog.records)
    assert adapter._platform_online is True


def test_heartbeat_without_status_is_ignored(caplog):
    adapter = make_adapter()
    caplog.set_level(logging.INFO, logger="pretender.onebot")
    with capture_onebot_logs(caplog):
        adapter._note_platform_online({"post_type": "meta_event",
                                       "meta_event_type": "heartbeat"})
        adapter._note_platform_online({"post_type": "meta_event",
                                       "meta_event_type": "heartbeat",
                                       "status": "not-a-dict"})
    assert adapter._platform_online is None
    assert not caplog.records
