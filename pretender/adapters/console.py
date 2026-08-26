"""Console adapter: a deterministic local REPL — dev and tests without QQ.

This is the phase-1 adapter: no network. It serves one chat
(``console:group:demo`` by default) with a fixed identity, reads lines from
an input stream (``sys.stdin`` by default) and yields them as normalized
``AdapterEvent`` messages. ``send`` prints the outgoing text and returns a
deterministic platform id — and, like a real platform, emits the sent
message back as a self-echo event, which ingest reconciles against the
synthetic echo the outbox wrote (dedupe, no second send).

Deterministic by construction: ids are sequential, timestamps come from the
injected clock, and tests drive it with ``feed`` + a ``StringIO`` input
instead of a terminal.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, AsyncIterator

from pretender.adapters.base import Adapter
from pretender.clock import RealClock
from pretender.errors import AdapterError
from pretender.types import (
    AdapterEvent,
    ChatIdentity,
    ChatKey,
    Message,
    MessageId,
    Outgoing,
    PlatformId,
    SelfId,
    SenderId,
)


class ConsoleAdapter:
    """A local REPL adapter implementing the Adapter protocol."""

    name = "console"
    capabilities: frozenset[str] = frozenset()

    def __init__(
        self,
        *,
        chat_key: str = "console:group:demo",
        self_id: str = "bot",
        clock: Any = None,
        input_stream: Any = None,
        output_stream: Any = None,
    ) -> None:
        self.chat_key = ChatKey(chat_key)
        self.self_id = SelfId(self_id)
        self.identity = ChatIdentity(
            self.chat_key, PlatformId("console"), self.self_id, "group", title="Console"
        )
        self._clock = clock if clock is not None else RealClock()
        self._input = input_stream
        self._output = output_stream
        self._events: asyncio.Queue[AdapterEvent | None] = asyncio.Queue()
        self._seq = 0
        self._sent: list[Outgoing] = []
        self._reader: asyncio.Task[None] | None = None
        self._closed = False

    # ── Adapter protocol ────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Start the REPL reader (explicit stream, or ``sys.stdin``)."""
        if self._reader is None:
            self._reader = asyncio.create_task(self._read_loop())

    async def events(self) -> AsyncIterator[AdapterEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def send(self, out: Outgoing) -> str | None:
        """Print the outgoing text and return a deterministic platform id.

        Also emits the sent message back as a self-echo event, simulating
        what a real platform does — ingest deduplicates it against the
        synthetic echo written by ``mark_outbox_sent``.
        """
        self._seq += 1
        pid = MessageId(f"console:out:{self._seq}")
        self._sent.append(out)
        print(f"[{self.chat_key}] {out.text}", file=self._output)
        echo = Message(
            chat_key=self.chat_key,
            sender_id=SenderId(self.self_id),
            sender_name=self.self_id,
            is_self=True,
            text=out.text,
            id=pid,
            segments=tuple(out.segments),
            reply_to=out.reply_to,
            recv_ts=self._clock.now(),
        )
        await self._events.put(AdapterEvent(type="message", payload=echo, ts=echo.recv_ts))
        return pid

    async def call(self, action: str, **params: Any) -> Any:
        raise AdapterError(f"console adapter has no platform API: {action!r}")

    # ── test / REPL control ─────────────────────────────────────────────────

    async def feed(
        self,
        text: str,
        *,
        sender_id: str = "user",
        sender_name: str = "user",
        is_self: bool = False,
        msg_id: str | None = None,
    ) -> Message:
        """Inject one inbound message (tests) or REPL line."""
        self._seq += 1
        msg = Message(
            chat_key=self.chat_key,
            sender_id=SenderId(sender_id),
            sender_name=sender_name,
            is_self=is_self,
            text=text,
            id=MessageId(msg_id) if msg_id is not None else MessageId(f"console:in:{self._seq}"),
            recv_ts=self._clock.now(),
        )
        await self._events.put(AdapterEvent(type="message", payload=msg, ts=msg.recv_ts))
        return msg

    async def close(self) -> None:
        """Stop the reader and end ``events()``. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except asyncio.CancelledError:
                pass
            self._reader = None
        await self._events.put(None)

    @property
    def sent(self) -> list[Outgoing]:
        return list(self._sent)

    # ── REPL reader ─────────────────────────────────────────────────────────

    async def _read_loop(self) -> None:
        stream = self._input if self._input is not None else sys.stdin
        loop = asyncio.get_running_loop()
        try:
            while not self._closed:
                line = await loop.run_in_executor(None, stream.readline)
                if not line:
                    break
                text = line.rstrip("\n")
                if not text.strip():
                    continue
                await self.feed(text)
        finally:
            # EOF or close: end the events() stream.
            await self._events.put(None)