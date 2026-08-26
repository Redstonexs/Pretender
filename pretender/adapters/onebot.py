"""OneBot v11 adapter: reverse WebSocket (NapCat dials us) or forward WebSocket.

Phase 4 lane: a production-ready OneBot v11 bridge implementing the Adapter
seam. It speaks the OneBot v11 protocol over WebSocket — by default as a
reverse server (NapCat dials us), optionally as an outbound client that
auto-reconnects with exponential backoff.

Responsibilities:
- reverse/forward WebSocket connection + handshake (path + Bearer token auth)
- reverse-server security: browser Origin upgrades rejected, non-loopback
  binds require auth, an unauthenticated connection never takes over a live
  legitimate session
- ``message_format=array`` normalization into AdapterEvent/Message/Segment
- robust group/private chat-key mapping (``qq:group:<id>`` / ``qq:private:<id>``)
- media/face/sticker normalization (via MediaStore; no vision LLM yet)
- echo correlation: ``send_group_msg``/``send_private_msg`` action -> response
- a persistent background receiver owns recv() and normalizes frames into an
  internal async queue consumed by ``events()``; ``send()``/``call()`` echoes
  resolve concurrently even before ``events()`` is iterated (startup recovery)
- protocol readiness (``ready``): an open socket alone is NOT ready — a valid
  OneBot lifecycle/self_id AND a successful API echo/probe are required
- retcode -1 (any non-zero) fallback: the send returns None (the outbox
  writes a synthetic local echo) while the adapter retains the exact sent
  payload + delivery key under a stable local id; a later real self echo is
  matched ONLY on unambiguous same chat/self/payload, its real id is bound to
  the key, the synthetic context is never duplicated, and the consumed local
  fallback candidate is retired so later identical ambiguous sends reconcile
- ``delivery_key_for``: trusted self-echo reconciliation key resolution
- real self id: learned from the configured ``self_id`` and/or every inbound
  OneBot event, exposed consistently via ``self_id``
- reconnect/backoff (forward mode) and re-dial handling (reverse mode)
- ping/pong watchdog (``heartbeat_timeout_s``): a quiet connection is PINGED
  and the pong awaited before it is dropped — a healthy quiet link stays up
- media downloads are scheduled in the background (never awaited inline in
  the frame/event loop), bounded by a concurrency semaphore + a bounded
  in-flight map (dropped entries are cancelled), and served from cache when
  ready
- cancellation-safe close
- capability declarations

App adoption is a later lane; this module is adapter-local.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, AsyncIterator, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import orjson
from websockets.asyncio.client import connect as ws_connect
from websockets.asyncio.server import serve as ws_serve
from websockets.exceptions import ConnectionClosed
from websockets.frames import CloseCode
from websockets.protocol import State

from pretender.adapters.base import Adapter, capability_set
from pretender.clock import RealClock
from pretender.config import OneBotConfig, _is_loopback_host
from pretender.errors import AdapterError, AdapterNotReady, ConfigError, TransientError
from pretender.log import get_logger
from pretender.media import MediaStore, media_segment_data
from pretender.types import (
    AdapterEvent,
    ChatKey,
    Message,
    MessageId,
    Outgoing,
    Segment,
    SenderId,
)

log = get_logger("onebot")

#: Plain-text placeholders for non-text segments (rendered into Message.text).
_TEXT_PLACEHOLDERS = {
    "image": "[图片]",
    "face": "[表情]",
    "sticker": "[贴纸]",
    "record": "[语音]",
    "video": "[视频]",
    "file": "[文件]",
    "location": "[位置]",
    "contact": "[名片]",
    "json": "[卡片]",
    "music": "[音乐]",
    "dice": "[骰子]",
    "rps": "[猜拳]",
    "basketball": "[篮球]",
    "weather": "[天气]",
    "forward": "[转发]",
    "share": "[分享]",
    "gift": "[礼物]",
    "poke": "[戳一戳]",
}
_UNKNOWN_PLACEHOLDER = "[未知]"

#: Media kinds we attempt to download+normalize (image/sticker carry a URL).
_MEDIA_KINDS = frozenset({"image", "sticker"})

#: Prefix of the adapter's stable local fallback ids (retained for later
#: real-echo reconciliation).
_LOCAL_ID_PREFIX = "onebot:local:"

#: Cap on a rendered forward body (chars). Bounds both the stored result and
#: the memory used while rendering a hostile oversized payload.
_MAX_FORWARD_CHARS = 4096


class OneBotAdapter:
    """A OneBot v11 WebSocket adapter implementing the Adapter protocol."""

    name = "onebot"
    capabilities = capability_set(
        "quote", "at", "image", "face", "sticker", "recall", "history", "forward"
    )

    def __init__(
        self,
        *,
        config: OneBotConfig | None = None,
        clock: Any = None,
        media: MediaStore | None = None,
        normalize_media: bool = True,
        self_id: str | None = None,
        max_remember: int = 512,
    ) -> None:
        self._cfg = config if config is not None else OneBotConfig()
        self._clock = clock if clock is not None else RealClock()
        self._media = media if media is not None else MediaStore()
        self._normalize_media = normalize_media
        # The real self id: an explicit constructor value wins, else the
        # configured one; every inbound OneBot event that carries a self_id
        # updates it (the platform is authoritative).
        self._self_id = self_id if self_id is not None else self._cfg.self_id
        self._max_remember = max_remember

        self._conn: Any = None
        self._conn_event: asyncio.Event = asyncio.Event()
        self._ready_event: asyncio.Event = asyncio.Event()
        self._adopt_lock = asyncio.Lock()
        self._server: Any = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._closed = False

        # A persistent background receiver owns recv() on the current
        # connection and normalizes every frame into this internal queue.
        # ``events()`` consumes the queue; ``send()``/``call()`` echoes are
        # resolved by the receiver concurrently — even before ``events()`` is
        # iterated (startup recovery). The receiver is started when a
        # connection is adopted and cancelled when it is replaced/closed.
        # A bounded queue prevents an untrusted peer from buffering arbitrary
        # event frames in memory. Echo responses are handled before enqueueing,
        # so a full application-event queue cannot strand an API action.
        self._rx_queue: asyncio.Queue[tuple[int, AdapterEvent] | None] = asyncio.Queue(
            maxsize=max(32, min(max_remember, 1024))
        )
        self._rx_task: asyncio.Task[None] | None = None
        self._rx_conn: Any = None
        self._probe_task: asyncio.Task[None] | None = None
        self._connection_generation = 0
        self._rx_generation = 0
        self._probe_generation = 0
        self._expected_self_id = self._self_id
        self._generation_self_id: str | None = None
        self._identity_conflict = False

        # echo -> future for in-flight API actions (send/call).
        self._pending: dict[str, asyncio.Future[dict]] = {}
        # platform message id -> delivery key (trusted self-echo reconciliation).
        self._delivered: dict[str, str] = {}
        # platform/local message id -> (chat_key, text, wire_segments,
        # durable_segments, reply_to) of what we sent — retained for fallback
        # reconciliation and payload correlation of self echoes.
        self._sent_payload: dict[
            str,
            tuple[
                str,
                str,
                tuple[Segment, ...],
                tuple[Segment, ...],
                MessageId | None,
            ],
        ] = {}
        # A real self echo may arrive before the action response. Seed a
        # provisional fallback candidate before sending, then remember a bound
        # real id by delivery key so the eventual ambiguous response can return
        # the real id instead of manufacturing a synthetic duplicate.
        self._early_bound: dict[str, str] = {}
        # url -> in-flight background media prefetch task (never awaited in
        # the frame/event loop). Bounded by ``_max_remember``; a dropped entry
        # is cancelled so live downloads never run unbounded.
        self._media_tasks: dict[str, asyncio.Task[None]] = {}
        self._forward_contents: dict[ChatKey, dict[str, str]] = {}
        self._forward_tasks: dict[tuple[ChatKey, str], asyncio.Task[None]] = {}
        # Bounds concurrent background media downloads/decodes.
        self._media_sem = asyncio.Semaphore(self._cfg.media_concurrency)
        self._echo_seq = 0
        self._local_seq = 0
        self._last_activity = 0.0

        # Protocol readiness: an open socket alone is NOT ready. Readiness
        # requires a valid OneBot lifecycle/self_id AND a successful API
        # echo/probe (see ``ready``).
        self._lifecycle_seen = False
        self._probe_ok = False

    # ── Adapter protocol ────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Start the reverse server or the forward reconnection loop."""
        if self._closed:
            raise AdapterError("onebot adapter is closed")
        if self._cfg.mode == "reverse_ws":
            await self._start_reverse_server()
        elif self._cfg.mode == "ws":
            self._reconnect_task = asyncio.create_task(self._forward_loop())
        else:
            raise ConfigError(f"unknown onebot mode {self._cfg.mode!r}")
        if self._cfg.heartbeat_timeout_s is not None:
            self._watchdog_task = asyncio.create_task(self._watchdog())

    async def events(self) -> AsyncIterator[AdapterEvent]:
        """Stream normalized events from the internal receive queue, surviving
        reconnects/re-dials until ``close()``.

        The background receiver owns recv() on the connection and feeds this
        queue, so events flow here regardless of when this iterator is
        consumed — and ``send()``/``call()`` echoes are resolved by the
        receiver concurrently, even before this iterator is started.
        """
        while True:
            if self._closed and self._rx_queue.empty():
                return
            try:
                item = await self._rx_queue.get()
            except asyncio.CancelledError:
                raise
            if item is None:
                return
            generation, event = item
            # Never resolve an event under a replacement account identity.
            if generation != self._connection_generation:
                continue
            yield event

    async def send(self, out: Outgoing) -> str | None:
        """Send one outgoing message; returns the platform message id, or a
        stable local id when the platform ack is ambiguous (retcode != 0)."""
        action, params = self._target_params(out)
        params["message"] = self._outgoing_to_array(out)
        echo = self._new_echo()
        start_generation = self._connection_generation if self.ready else None
        provisional = self._fallback_local_id(out)
        def remember_provisional() -> None:
            if not out.delivery_key:
                return
            self._remember(self._delivered, provisional, out.delivery_key)
            self._remember(
                self._sent_payload, provisional, self._sent_payload_value(out)
            )
        fut: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
        self._pending[echo] = fut
        try:
            await self._write_frame(
                orjson.dumps({"action": action, "params": params, "echo": echo}),
                require_ready=True,
                expected_generation=start_generation,
                wait_for_ready=start_generation is None,
                before_write=remember_provisional,
            )
            resp = await asyncio.wait_for(fut, timeout=self._cfg.action_timeout_s)
        except AdapterNotReady:
            # A pre-write fence failure proves no bytes were handed to the
            # platform. Remove every provisional correlation candidate so a
            # later identical self echo can never bind this unsent attempt.
            self._delivered.pop(provisional, None)
            self._sent_payload.pop(provisional, None)
            if out.delivery_key:
                self._early_bound.pop(out.delivery_key, None)
            raise
        except asyncio.TimeoutError as e:
            if out.delivery_key:
                self._early_bound.pop(out.delivery_key, None)
            raise TransientError(
                f"onebot send timed out waiting for echo {echo}"
            ) from e
        except ConnectionClosed as e:
            if out.delivery_key:
                self._early_bound.pop(out.delivery_key, None)
            raise TransientError("onebot connection closed during send") from e
        except OSError as e:
            if out.delivery_key:
                self._early_bound.pop(out.delivery_key, None)
            raise TransientError(f"onebot send failed: {e}") from e
        except asyncio.CancelledError:
            if out.delivery_key:
                self._early_bound.pop(out.delivery_key, None)
            raise
        finally:
            self._pending.pop(echo, None)
        return self._send_result(resp, out, provisional=provisional)

    async def call(self, action: str, **params: Any) -> Any:
        """Escape hatch: invoke any OneBot API action and return its data."""
        echo = self._new_echo()
        start_generation = self._connection_generation if self.ready else None
        fut: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
        self._pending[echo] = fut
        try:
            await self._write_frame(
                orjson.dumps({"action": action, "params": params, "echo": echo}),
                require_ready=True,
                expected_generation=start_generation,
                wait_for_ready=start_generation is None,
            )
            resp = await asyncio.wait_for(fut, timeout=self._cfg.action_timeout_s)
        except asyncio.TimeoutError as e:
            raise TransientError(f"onebot action {action!r} timed out") from e
        except ConnectionClosed as e:
            raise TransientError(f"onebot connection closed during {action!r}") from e
        finally:
            self._pending.pop(echo, None)
        if resp.get("retcode") != 0:
            raise AdapterError(
                f"onebot action {action!r} failed: retcode={resp.get('retcode')} "
                f"msg={resp.get('message') or resp.get('msg')}"
            )
        return resp.get("data")

    # ── trusted self-echo reconciliation ────────────────────────────────────

    @property
    def self_id(self) -> str | None:
        """The bot's real self id: the configured value, updated by every
        inbound OneBot event that carries one. Exposed consistently for the
        durable App identity, direct-@ detection, and self-echo verification."""
        return self._self_id

    @property
    def generation(self) -> int:
        """Monotonic identifier of the currently adopted connection."""
        return self._connection_generation

    def delivery_key_for(self, msg: Message) -> str | None:
        """Resolve the trusted delivery key for a self echo, if we sent it.

        The real id is bound to the key during normalization (a fallback
        send's real echo is matched on unambiguous same chat/self/payload),
        so this is a plain lookup — never a heuristic match on echo data.
        """
        if not msg.is_self or msg.id is None:
            return None
        return self._delivered.get(msg.id)

    def forwards_for(self, chat_key: ChatKey) -> dict[str, str]:
        """Return only locally verified forward bodies for one chat.

        Tool handlers are synchronous, so inbound forward segments schedule the
        OneBot `get_forward_msg` action ahead of planner execution. Unknown or
        still-loading ids stay absent and fail closed in ToolContext.
        """
        return dict(self._forward_contents.get(chat_key, {}))

    # ── connection management ───────────────────────────────────────────────

    async def _start_reverse_server(self) -> None:
        self._server = await ws_serve(
            self._on_client,
            self._cfg.host,
            self._cfg.port,
            ping_interval=20.0,
            ping_timeout=20.0,
            open_timeout=self._cfg.action_timeout_s,
            max_size=16 * 1024 * 1024,
        )

    async def _on_client(self, conn: Any) -> None:
        """Reverse-server handler: verify the handshake (path, Origin, auth),
        adopt the connection, start the background receiver, and hold it open
        (the receiver owns recv; this coroutine only waits for close)."""
        request = getattr(conn, "request", None)
        path = ""
        query: dict[str, str] = {}
        if request is not None:
            request_target = urlsplit(request.path or "")
            path = request_target.path
            query = dict(parse_qsl(request_target.query, keep_blank_values=True))
        if path != self._cfg.path:
            await conn.close(CloseCode.POLICY_VIOLATION, "unexpected path")
            return
        # Reject browser Origin upgrades: browsers always send an Origin
        # header on WebSocket upgrades; a legitimate OneBot client does not.
        if request is not None and request.headers.get("Origin"):
            await conn.close(CloseCode.POLICY_VIOLATION, "browser origin rejected")
            return
        # A non-loopback bind is reachable by remote clients, so it MUST be
        # authenticated (also enforced at config load).
        if not _is_loopback_host(self._cfg.host) and not self._cfg.access_token:
            await conn.close(
                CloseCode.POLICY_VIOLATION, "auth required for non-loopback bind"
            )
            return
        token = self._cfg.access_token
        if token:
            auth = ""
            if request is not None:
                auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {token}":
                await conn.close(CloseCode.POLICY_VIOLATION, "unauthorized")
                return
        # Authenticate before disclosing/processing format requirements; the
        # event contract itself is fail-closed on array-format negotiation.
        if query.get("message_format") != "array":
            await conn.close(
                CloseCode.POLICY_VIOLATION,
                "message_format=array is required",
            )
            return
        generation = await self._adopt_connection(conn)
        try:
            await conn.wait_closed()
        finally:
            await self._release_connection(conn, generation)

    async def _forward_loop(self) -> None:
        """Forward-mode reconnection loop: dial out, auto-reconnect with
        backoff (websockets' ``async for``), and re-dial on fatal errors."""
        uri = self._ws_uri()
        headers: dict[str, str] = {}
        if self._cfg.access_token:
            headers["Authorization"] = f"Bearer {self._cfg.access_token}"
        attempt = 0
        while not self._closed:
            try:
                async for ws in ws_connect(
                    uri,
                    additional_headers=headers,
                    open_timeout=self._cfg.action_timeout_s,
                    ping_interval=20.0,
                    ping_timeout=20.0,
                    max_size=16 * 1024 * 1024,
                    reconnect_delays=self._reconnect_delays,
                ):
                    attempt = 0
                    generation = await self._adopt_connection(ws)
                    try:
                        await ws.wait_closed()
                    finally:
                        await self._release_connection(ws, generation)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self._closed:
                    return
                log.warning("onebot forward connection failed: %s", e)
                delay = min(
                    self._cfg.reconnect_base_s * (2**attempt),
                    self._cfg.reconnect_max_s,
                )
                attempt += 1
                await self._clock.sleep(delay)

    async def _watchdog(self) -> None:
        """Ping/pong watchdog: probe a quiet connection with a ping and await
        the pong (bounded by ``ping_timeout_s``) BEFORE dropping it. A healthy
        quiet link that answers the ping stays connected — no reconnect storm."""
        timeout = self._cfg.heartbeat_timeout_s
        if timeout is None:
            return
        interval = max(timeout / 3.0, 0.5)
        while not self._closed:
            await self._clock.sleep(interval)
            if self._closed:
                return
            conn = self._conn
            if conn is None or not self._is_open(conn):
                continue
            if self._clock.monotonic() - self._last_activity <= timeout:
                continue
            # Quiet for too long: ping and await the pong before dropping.
            try:
                pong_waiter = await asyncio.wait_for(
                    conn.ping(), timeout=self._cfg.ping_timeout_s
                )
                await asyncio.wait_for(pong_waiter, timeout=self._cfg.ping_timeout_s)
                self._note_activity()
                continue
            except (asyncio.TimeoutError, ConnectionClosed, OSError):
                log.warning(
                    "onebot heartbeat watchdog: no pong within %.1fs; "
                    "dropping connection",
                    self._cfg.ping_timeout_s,
                )
                try:
                    await conn.close(CloseCode.GOING_AWAY, "heartbeat timeout")
                except Exception:
                    pass

    async def _adopt_connection(self, conn: Any) -> int:
        """Serialize connection replacement and publish only fully reset
        generation state. A stale handler cannot replace/cancel a newer
        receiver or probe while this lock is held."""
        async with self._adopt_lock:
            prev = self._conn
            if prev is not None and prev is not conn:
                try:
                    await prev.close(
                        CloseCode.GOING_AWAY, "replaced by new connection"
                    )
                except Exception:
                    pass
            await self._start_receiver(conn)
            generation = self._connection_generation
            self._conn = conn
            self._note_activity()
            self._conn_event.set()
            await self._probe(conn, generation)
            return generation

    async def _release_connection(self, conn: Any, generation: int) -> None:
        """Release only the exact adopted generation; an old handler cannot
        clear a replacement socket, receiver, or readiness probe."""
        async with self._adopt_lock:
            if self._conn is not conn or self._connection_generation != generation:
                return
            self._conn = None
            self._conn_event.set()
            await self._stop_receiver(conn, generation)

    def _ws_uri(self) -> str:
        path = urlsplit(self._cfg.path)
        query = dict(parse_qsl(path.query, keep_blank_values=True))
        query["message_format"] = "array"
        target = urlunsplit(("", "", path.path or "/", urlencode(query), ""))
        return f"{self._cfg.scheme}://{self._cfg.host}:{self._cfg.port}{target}"

    def _reconnect_delays(self):
        """Exponential backoff delays for forward-mode reconnection, bounded
        by ``reconnect_base_s``/``reconnect_max_s``."""
        base = self._cfg.reconnect_base_s
        cap = self._cfg.reconnect_max_s
        delay = base
        while True:
            yield delay
            delay = min(delay * 2, cap)

    async def _wait_for_conn(self) -> Any | None:
        while not self._closed:
            self._conn_event.clear()
            conn = self._conn
            if conn is not None and self._is_open(conn):
                return conn
            await self._conn_event.wait()
        return None

    def _require_conn(self) -> Any:
        conn = self._conn
        if conn is None or not self._is_open(conn):
            raise AdapterNotReady("onebot: no live connection before send")
        return conn

    async def _write_frame(
        self,
        frame: bytes,
        *,
        require_ready: bool,
        expected_conn: Any | None = None,
        expected_generation: int | None = None,
        wait_for_ready: bool = False,
        before_write: Callable[[], None] | None = None,
    ) -> None:
        """Write under the adoption lock through the exact ready generation.

        A reconnect cannot slip between the readiness check and the websocket
        write. If the old generation disappeared before a write started, the
        caller receives ``AdapterNotReady`` and the outbox can safely requeue.
        """
        while True:
            async with self._adopt_lock:
                conn = self._conn
                if conn is None or not self._is_open(conn):
                    raise AdapterNotReady("onebot: no live connection before send")
                if expected_conn is not None and conn is not expected_conn:
                    raise AdapterNotReady("onebot: connection generation replaced")
                if (
                    expected_generation is not None
                    and self._connection_generation != expected_generation
                ):
                    raise AdapterNotReady("onebot: connection generation replaced")
                if not require_ready or self.ready:
                    if before_write is not None:
                        before_write()
                    await conn.send(frame)
                    return
                if not wait_for_ready:
                    raise AdapterNotReady("onebot: connection is not protocol-ready")
            try:
                await asyncio.wait_for(
                    self._ready_event.wait(), timeout=self._cfg.action_timeout_s
                )
            except asyncio.TimeoutError:
                raise AdapterNotReady("onebot: readiness timed out before send") from None

    @staticmethod
    def _is_open(conn: Any) -> bool:
        return getattr(conn, "state", None) is State.OPEN

    def _note_activity(self) -> None:
        self._last_activity = self._clock.monotonic()

    # ── background receiver + protocol readiness ────────────────────────────

    async def _receiver(self, conn: Any, generation: int) -> None:
        """The persistent background receiver: owns recv() on ``conn`` and
        normalizes every frame into the internal queue. Echo responses are
        resolved here, so ``send()``/``call()`` complete even when ``events()``
        is not being iterated (startup recovery)."""
        try:
            async for raw in conn:
                try:
                    event = await self._handle_frame(raw, generation=generation)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A malformed adapter event must not kill echo handling
                    # while leaving a socket advertised as connected.
                    log.exception("onebot: dropping malformed frame")
                    continue
                if event is not None:
                    try:
                        self._rx_queue.put_nowait((generation, event))
                    except asyncio.QueueFull:
                        # Never silently lose an unbounded stream while
                        # continuing to send. Close and reconnect instead.
                        log.warning("onebot event queue full; closing connection")
                        await conn.close(
                            CloseCode.TRY_AGAIN_LATER, "event queue full"
                        )
                        return
        except ConnectionClosed:
            pass
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("onebot receiver error; closing for reconnect")
            try:
                await conn.close(CloseCode.INTERNAL_ERROR, "receiver error")
            except Exception:
                pass
        finally:
            # The reverse/forward owner calls _release_connection after the
            # socket closes. Keep receiver cleanup generation-local; never
            # clear the adapter's current connection from this background task.
            pass

    async def _start_receiver(self, conn: Any) -> None:
        """Adopt ``conn``: cancel any previous receiver and start a new one
        plus a fresh readiness probe for this connection."""
        await self._stop_receiver()
        # Readiness belongs to a connection, never the adapter lifetime. A
        # reconnect must prove lifecycle/self-id and API echo again.
        self._probe_ok = False
        self._lifecycle_seen = False
        self._generation_self_id = None
        self._identity_conflict = False
        self._ready_event.clear()
        self._connection_generation += 1
        generation = self._connection_generation
        self._rx_conn = conn
        self._rx_generation = generation
        self._rx_task = asyncio.create_task(self._receiver(conn, generation))

    async def _stop_receiver(
        self, conn: Any | None = None, generation: int | None = None
    ) -> None:
        """Cancel the receiver for ``conn`` (or the current one when None)
        plus any in-flight readiness probe. Idempotent and cancellation-safe."""
        if conn is not None and self._rx_conn is not conn:
            return
        if generation is not None and self._rx_generation != generation:
            return
        task = self._rx_task
        probe_task = (
            self._probe_task
            if self._probe_generation == self._rx_generation
            else None
        )
        self._rx_task = None
        self._rx_conn = None
        self._probe_task = None
        self._rx_generation = 0
        self._probe_generation = 0
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if probe_task is not None:
            probe_task.cancel()
            try:
                await probe_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _probe(self, conn: Any, generation: int) -> None:
        """Send a ``get_login_info`` API probe on the freshly adopted
        connection and mark ``_probe_ok`` when its echo resolves with
        retcode 0. Runs as a background task — never blocks the receiver."""
        if (
            self._probe_task is not None
            and self._probe_generation == generation
        ):
            self._probe_task.cancel()
        task = asyncio.create_task(self._probe_once(conn, generation))
        self._probe_task = task
        self._probe_generation = generation

    async def _probe_once(self, conn: Any, generation: int) -> None:
        echo = self._new_echo()
        fut: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
        self._pending[echo] = fut
        try:
            await self._write_frame(
                orjson.dumps({"action": "get_login_info", "params": {}, "echo": echo}),
                require_ready=False,
                expected_conn=conn,
                expected_generation=generation,
            )
            resp = await asyncio.wait_for(fut, timeout=self._cfg.action_timeout_s)
            if (
                resp.get("retcode") == 0
                and self._conn is conn
                and self._connection_generation == generation
            ):
                data = resp.get("data") or {}
                sid = str(data.get("user_id") or "").strip()
                if sid:
                    if self._expected_self_id and sid != self._expected_self_id:
                        self._identity_conflict = True
                        self._ready_event.clear()
                        try:
                            await conn.close(
                                CloseCode.POLICY_VIOLATION,
                                "onebot self_id mismatch",
                            )
                        except Exception:
                            pass
                        return
                    if self._expected_self_id is None:
                        self._expected_self_id = sid
                    self._generation_self_id = sid
                    self._self_id = sid
                    self._lifecycle_seen = True
                self._probe_ok = True
                if self.ready:
                    self._ready_event.set()
        except (asyncio.TimeoutError, ConnectionClosed, OSError):
            pass
        finally:
            self._pending.pop(echo, None)

    @property
    def ready(self) -> bool:
        """A reliable protocol-readiness signal for App/Doctor: an open socket
        alone is NOT ready. Ready requires a valid OneBot lifecycle/self_id
        (``_lifecycle_seen``) AND a successful API echo/probe
        (``_probe_ok``)."""
        return (
            self.connected
            and not self._identity_conflict
            and bool(self._generation_self_id)
            and self._probe_ok
        )

    # ── frame handling ──────────────────────────────────────────────────────

    async def _handle_frame(
        self, raw: Any, *, generation: int | None = None
    ) -> AdapterEvent | None:
        self._note_activity()
        try:
            data = orjson.loads(raw)
        except (ValueError, TypeError):
            log.warning("onebot: dropping non-JSON frame")
            return None
        if not isinstance(data, dict):
            return None
        if generation is not None and generation != self._connection_generation:
            return None
        self._learn_self_id(data, generation=generation)
        if self._identity_conflict:
            # Do not normalize, resolve echoes from, or queue a frame from a
            # mismatched account. Close the current socket so a future
            # reconnect must establish a fresh identity/probe generation.
            conn = self._conn
            if conn is not None and self._is_open(conn):
                try:
                    await conn.close(
                        CloseCode.POLICY_VIOLATION, "onebot self_id mismatch"
                    )
                except Exception:
                    pass
            return None
        # A valid OneBot lifecycle meta-event is part of protocol readiness.
        if (
            data.get("post_type") == "meta_event"
            and data.get("meta_event_type") == "lifecycle"
        ):
            self._lifecycle_seen = True
        # API response (echo correlation) — carries echo, no post_type.
        if "echo" in data and "post_type" not in data:
            self._resolve_echo(data)
            return None
        return await self._normalize_event(data)

    def _learn_self_id(self, data: dict, *, generation: int | None = None) -> None:
        """Learn the real self id from any inbound OneBot event that carries
        one (the platform is authoritative). A learned self id is part of
        protocol readiness."""
        sid = data.get("self_id")
        if sid is None:
            return
        sid = str(sid).strip()
        if generation is not None and generation != self._connection_generation:
            return
        if sid:
            if self._expected_self_id and sid != self._expected_self_id:
                self._identity_conflict = True
                self._ready_event.clear()
                return
            if self._expected_self_id is None:
                self._expected_self_id = sid
            if self._generation_self_id and sid != self._generation_self_id:
                self._identity_conflict = True
                self._ready_event.clear()
                return
            self._generation_self_id = sid
            self._self_id = sid
            self._lifecycle_seen = True
            if self._probe_ok and self.connected and not self._identity_conflict:
                self._ready_event.set()

    def _resolve_echo(self, data: dict) -> None:
        echo = data.get("echo")
        if echo is None:
            return
        fut = self._pending.pop(echo, None)
        if fut is not None and not fut.done():
            fut.set_result(data)

    async def _normalize_event(self, data: dict) -> AdapterEvent:
        post_type = data.get("post_type")
        if post_type in ("message", "message_sent"):
            msg = await self._normalize_message(data)
            return AdapterEvent(type="message", payload=msg, raw=data, ts=msg.recv_ts)
        if post_type in ("notice", "request"):
            return AdapterEvent(type=post_type, payload=data, raw=data, ts=self._ts(data))
        # meta_event (lifecycle/heartbeat) and anything else -> "meta".
        return AdapterEvent(type="meta", payload=data, raw=data, ts=self._ts(data))

    def _ts(self, data: dict) -> float | None:
        t = data.get("time")
        if isinstance(t, (int, float)):
            return float(t)
        return self._clock.now()

    async def _normalize_message(self, data: dict) -> Message:
        post_type = data.get("post_type")
        self_id = str(data.get("self_id") or self._self_id or "")
        user_id = data.get("user_id")
        message_id = data.get("message_id")
        msg_type = data.get("message_type")
        group_id = data.get("group_id")
        target_id = data.get("target_id")

        is_self = bool(
            post_type == "message_sent"
            or (user_id is not None and bool(self_id) and str(user_id) == str(self_id))
        )

        if msg_type == "group" and group_id is not None:
            chat_key = ChatKey(f"qq:group:{group_id}")
        else:
            # private: key by the OTHER party (target_id for self echoes).
            other = target_id if target_id is not None else user_id
            chat_key = ChatKey(f"qq:private:{other}")

        sender = data.get("sender") or {}
        sender_name = sender.get("card") or sender.get("nickname") or str(user_id or "")
        sender_id = SenderId(str(user_id)) if user_id is not None else SenderId(str(self_id))

        text, segments, reply_to, mentions = await self._normalize_segments(
            data.get("message")
        )

        msg = Message(
            chat_key=chat_key,
            sender_id=sender_id,
            sender_name=str(sender_name),
            is_self=is_self,
            text=text,
            id=MessageId(str(message_id)) if message_id is not None else None,
            segments=segments,
            reply_to=reply_to,
            mentions=mentions,
            recv_ts=self._ts(data),
            raw=data,
        )
        for segment in segments:
            if segment.kind != "forward":
                continue
            forward_id = segment.data.get("id") or segment.data.get("res_id")
            if forward_id is not None and str(forward_id).strip():
                self._schedule_forward(chat_key, str(forward_id))

        # Correlated self echo: reuse the exact sent payload so the trusted
        # delivery-key reconciliation matches the outbox row byte-for-byte.
        # A fallback send's real echo is matched first (unambiguous same
        # chat/self/payload) and its real id bound to the delivery key.
        if is_self and msg.id is not None:
            if msg.id not in self._sent_payload:
                self._bind_fallback_echo(msg)
            if msg.id in self._sent_payload:
                _chat, sent_text, _wire, sent_segments, sent_reply = self._sent_payload[msg.id]
                msg = replace(
                    msg, text=sent_text, segments=sent_segments, reply_to=sent_reply
                )
        return msg

    def _schedule_forward(self, chat_key: ChatKey, forward_id: str) -> None:
        key = (chat_key, forward_id)
        if key in self._forward_tasks or forward_id in self._forward_contents.get(chat_key, {}):
            return
        if len(self._forward_tasks) >= self._max_remember:
            old_key, old_task = next(iter(self._forward_tasks.items()))
            old_task.cancel()
            self._forward_tasks.pop(old_key, None)
        task = asyncio.create_task(self._load_forward(chat_key, forward_id))
        self._forward_tasks[key] = task
        task.add_done_callback(lambda done: self._forward_tasks.pop(key, None))

    async def _load_forward(self, chat_key: ChatKey, forward_id: str) -> None:
        try:
            data = await self.call("get_forward_msg", id=forward_id)
            content = await self._render_forward(chat_key, data)
            if content is None:
                return
            bucket = self._forward_contents.setdefault(chat_key, {})
            bucket[forward_id] = content
            while len(bucket) > self._max_remember:
                bucket.pop(next(iter(bucket)))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.debug("onebot forward fetch failed for %s", forward_id, exc_info=True)

    async def _render_forward(self, chat_key: ChatKey, data: Any) -> str | None:
        """Render a ``get_forward_msg`` payload into a plain-text body bound
        to the trusted inbound ``chat_key``.

        Accepts both the real OneBot/NapCat node form (``{type: 'node',
        data: {user_id, nickname, message/content}}``) and the normalized
        flat form (``{message_type, group_id, user_id, sender, message}``).
        The content is bound to the chat-scoped forward reference that
        triggered the fetch: a node that EXPLICITLY claims a different chat
        is rejected (cross-chat untrusted), while a node that carries no
        chat scope — the real protocol provides none per node — is accepted.
        Malformed payloads fail closed (None). The result is capped at
        ``_MAX_FORWARD_CHARS``.
        """
        if not isinstance(data, dict):
            return None
        items = data.get("messages")
        # Some OneBot implementations return the singular message shape for a
        # one-node forward. Normalize it to the same bounded loop.
        if items is None and data.get("message") is not None:
            raw_message = data["message"]
            # Standard OneBot's singular response wraps FORWARD NODES in its
            # ``message`` list. A flat message response also uses ``message``
            # for ordinary segments, so discriminate by root provenance.
            if (
                data.get("message_type") is None
                and isinstance(raw_message, list)
                and all(
                    isinstance(item, dict) and item.get("type") == "node"
                    for item in raw_message
                )
            ):
                items = raw_message
            else:
                items = [data]
        if not isinstance(items, list):
            return None
        lines: list[str] = []
        total = 0
        for item in items:
            if not isinstance(item, dict):
                return None
            node_form = item.get("type") == "node"
            node = self._forward_node(item)
            if node is None:
                return None
            if not self._node_in_chat(chat_key, node, node_form=node_form):
                return None
            raw_message = node.get("message")
            if raw_message is None:
                raw_message = node.get("content")
            try:
                text, _segments, _reply, _mentions = await self._normalize_segments(
                    raw_message
                )
            except (AdapterError, TypeError, ValueError):
                return None
            name = (
                node.get("nickname")
                or (node.get("sender") or {}).get("card")
                or (node.get("sender") or {}).get("nickname")
                or str(node.get("user_id") or "")
            )
            line = f"{name}: {text}"
            lines.append(line)
            total += len(line) + 1
            if total >= _MAX_FORWARD_CHARS:
                break
        content = "\n".join(lines)
        return content[:_MAX_FORWARD_CHARS] if content else None

    @staticmethod
    def _forward_node(item: dict) -> dict | None:
        """Extract a forward node payload from either the normalized flat form
        or the real OneBot node form (``{type: 'node', data: {...}}``).
        Anything else is malformed."""
        if item.get("type") == "node":
            data = item.get("data")
            if not isinstance(data, dict):
                return None
            return data
        return item

    def _node_in_chat(
        self, chat_key: ChatKey, node: dict, *, node_form: bool
    ) -> bool:
        """Strict chat authorization for a fetched forward node.

        The fetched content is bound to the trusted inbound chat-scoped
        forward reference (``chat_key``). A node that EXPLICITLY claims a
        different chat is rejected (cross-chat untrusted); a node that
        carries no chat scope — the real OneBot protocol provides none per
        node — is accepted and bound to the trusted chat.
        """
        if chat_key.startswith("qq:group:"):
            gid = node.get("group_id")
            if node_form:
                return gid is None or str(gid) == chat_key.rsplit(":", 1)[-1]
            # Flat records carry provenance and must prove it. Never let a
            # private flat record with an author id masquerade as group data.
            return (
                node.get("message_type") == "group"
                and gid is not None
                and str(gid) == chat_key.rsplit(":", 1)[-1]
            )
        # A node's user_id is its forwarded AUTHOR, not private-chat
        # provenance. The trusted inbound forward reference binds node-form
        # content to its requesting private chat; only an explicit group claim
        # is contradictory. Flat records carry their own message_type instead.
        if node.get("group_id") is not None:
            return False
        return node_form or node.get("message_type") == "private"

    def _bind_fallback_echo(self, msg: Message) -> None:
        """Match a real self echo against retained fallback sends.

        Binds the echo's real id to the delivery key ONLY on an unambiguous
        match of same chat/self/payload against exactly one retained fallback
        entry. Never trusts arbitrary echo data: a mismatch, a missing entry,
        or ambiguity leaves the echo unbound (fail closed). A consumed local
        fallback candidate is RETIRED after a successful bind, so a later
        identical ambiguous send can reconcile against the remaining
        candidates instead of staying ambiguous forever."""
        if msg.id is None:
            return
        matches: list[tuple[str, str]] = []
        for local_id, key in self._delivered.items():
            if not local_id.startswith(_LOCAL_ID_PREFIX):
                continue
            payload = self._sent_payload.get(local_id)
            if payload is None:
                continue
            sent_chat, sent_text, sent_wire, _durable, sent_reply = payload
            if (
                msg.chat_key == sent_chat
                and msg.text == sent_text
                and self._segments_equal(msg.segments, sent_wire)
                and msg.reply_to == sent_reply
            ):
                matches.append((local_id, key))
        if len(matches) != 1:
            return  # none or ambiguous: never bind
        local_id, key = matches[0]
        self._delivered[msg.id] = key
        self._sent_payload[msg.id] = self._sent_payload[local_id]
        self._remember(self._early_bound, key, msg.id)
        # Retire the consumed local fallback candidate: it is now bound to a
        # real platform id, so it must not shadow later identical sends.
        self._delivered.pop(local_id, None)
        self._sent_payload.pop(local_id, None)

    @staticmethod
    def _segments_equal(
        a: tuple[Segment, ...], b: tuple[Segment, ...]
    ) -> bool:
        """Compare segments by kind+data only (the echo's segments carry the
        platform ``raw`` payload; ours carry None)."""
        if len(a) != len(b):
            return False
        return all(x.kind == y.kind and x.data == y.data for x, y in zip(a, b))

    async def _normalize_segments(
        self, message: Any
    ) -> tuple[str, tuple[Segment, ...], MessageId | None, tuple[SenderId, ...]]:
        if not isinstance(message, list):
            raise AdapterError(
                "onebot requires message_format=array; string message payload rejected"
            )
        segments: list[Segment] = []
        texts: list[str] = []
        reply_to: MessageId | None = None
        mentions: list[SenderId] = []
        for item in message:
            if not isinstance(item, dict):
                continue
            seg = await self._normalize_segment(item)
            segments.append(seg)
            texts.append(self._render_segment(seg))
            if seg.kind == "reply" and reply_to is None:
                rid = seg.data.get("id")
                if rid is not None:
                    reply_to = MessageId(str(rid))
            if seg.kind == "at":
                qq = seg.data.get("qq")
                if qq is not None and str(qq) != "all":
                    mentions.append(SenderId(str(qq)))
        return "".join(texts), tuple(segments), reply_to, tuple(mentions)

    async def _normalize_segment(self, item: dict) -> Segment:
        kind = str(item.get("type") or "text").lower()
        data = dict(item.get("data") or {})
        if kind in _MEDIA_KINDS and self._normalize_media:
            data = self._normalize_media_data(data)
        return Segment(kind=kind, data=data, raw=item)

    def _normalize_media_data(self, data: dict) -> dict:
        """Attach cached media to a segment, or schedule the download in the
        background. NEVER awaits a download in the frame/event loop: a slow
        or failing fetch cannot block the following frame."""
        url = data.get("url") or data.get("file")
        if not url or isinstance(url, bytes):
            return data
        url = str(url)
        if url.startswith(("file://", "base64://")):
            return data
        self._schedule_media(url)
        asset = self._media.cached(url)
        if asset is None:
            return data  # not cached yet: keep the original segment
        out = dict(data)
        out["media"] = media_segment_data(asset)
        return out

    def _schedule_media(self, url: str) -> None:
        """Schedule a background download+cache for ``url`` (deduplicated by
        URL, bounded). The task is never awaited inline; when it finishes the
        asset is cached and later frames attach it synchronously. Concurrent
        downloads are bounded by a semaphore; the in-flight map is bounded by
        ``_max_remember`` and a dropped entry is CANCELLED so live downloads
        never run unbounded."""
        if url in self._media_tasks:
            return
        if len(self._media_tasks) >= self._max_remember:
            # Bound the in-flight map: cancel the oldest entry so the number
            # of live download/decode tasks stays bounded.
            old_url, old_task = next(iter(self._media_tasks.items()))
            old_task.cancel()
            self._media_tasks.pop(old_url, None)
        task = asyncio.create_task(self._media_prefetch(url))
        self._media_tasks[url] = task
        task.add_done_callback(self._media_done)

    def _media_done(self, task: asyncio.Task[None]) -> None:
        self._media_tasks = {
            u: t for u, t in self._media_tasks.items() if t is not task
        }

    async def _media_prefetch(self, url: str) -> None:
        async with self._media_sem:
            try:
                await self._media.prefetch(url)
            except Exception:
                log.debug("onebot media prefetch failed for %s", url, exc_info=True)

    def _render_segment(self, seg: Segment) -> str:
        if seg.kind == "text":
            return str(seg.data.get("text") or "")
        if seg.kind == "at":
            name = seg.data.get("name") or seg.data.get("qq")
            return f"@{name}" if name is not None else "@"
        if seg.kind == "reply":
            return ""
        return _TEXT_PLACEHOLDERS.get(seg.kind, _UNKNOWN_PLACEHOLDER)

    # ── outgoing ────────────────────────────────────────────────────────────

    def _target_params(self, out: Outgoing) -> tuple[str, dict]:
        """Derive the OneBot send target from ``out.chat_key`` ONLY
        (``qq:group:<id>`` / ``qq:private:<id>``) — never from the untrusted
        ``out.group_id``. Invalid/malformed targets raise AdapterError."""
        group_id = self._chat_part(out.chat_key, "group")
        if group_id is not None:
            return "send_group_msg", {"group_id": self._target_id(group_id, out)}
        user_id = self._chat_part(out.chat_key, "private")
        if user_id is None:
            raise AdapterError(
                f"cannot resolve onebot send target for {out.chat_key!r}"
            )
        return "send_private_msg", {"user_id": self._target_id(user_id, out)}

    @staticmethod
    def _target_id(raw: str, out: Outgoing) -> int:
        """Validate a chat-key id segment is a plain nonnegative integer;
        anything else (empty, non-numeric, negative) is an AdapterError."""
        try:
            value = int(raw)
        except (ValueError, TypeError):
            raise AdapterError(
                f"invalid onebot send target id {raw!r} for {out.chat_key!r}"
            ) from None
        if value < 0:
            raise AdapterError(
                f"invalid onebot send target id {raw!r} for {out.chat_key!r}"
            )
        return value

    @staticmethod
    def _chat_part(chat_key: str, kind: str) -> str | None:
        """The id segment of a well-formed ``qq:<kind>:<id>`` chat key, or
        None for anything else (wrong platform, wrong kind, extra/missing
        parts, empty id)."""
        parts = chat_key.split(":")
        if len(parts) == 3 and parts[0] == "qq" and parts[1] == kind:
            return parts[2]
        return None

    def _outgoing_to_array(self, out: Outgoing) -> list[dict]:
        segs = self._outgoing_segments(out)
        return [self._segment_to_onebot(s) for s in segs]

    def _outgoing_segments(self, out: Outgoing) -> tuple[Segment, ...]:
        segs = list(out.segments) if out.segments else [Segment("text", {"text": out.text})]
        if out.reply_to is not None:
            segs = [Segment("reply", {"id": str(out.reply_to)})] + segs
        return tuple(segs)

    def _segment_to_onebot(self, seg: Segment) -> dict:
        kind = seg.kind
        data = seg.data
        if kind == "text":
            return {"type": "text", "data": {"text": str(data.get("text") or "")}}
        if kind == "at":
            qq = data.get("qq") or data.get("user_id") or "all"
            return {"type": "at", "data": {"qq": str(qq)}}
        if kind == "reply":
            return {"type": "reply", "data": {"id": str(data.get("id") or "")}}
        if kind == "image":
            file = data.get("file") or data.get("url") or (data.get("media") or {}).get("data_url")
            return {"type": "image", "data": {"file": file} if file is not None else {}}
        if kind == "face":
            return {"type": "face", "data": {"id": str(data.get("id") or "")}}
        if kind == "sticker":
            file = data.get("file") or data.get("url") or (data.get("media") or {}).get("data_url")
            return {"type": "sticker", "data": {"file": file} if file is not None else {}}
        return {"type": kind, "data": dict(data)}

    def _send_result(
        self, resp: dict, out: Outgoing, *, provisional: str | None = None
    ) -> str | None:
        retcode = resp.get("retcode", 0)
        data = resp.get("data") or {}
        if retcode == 0 and data.get("message_id") is not None:
            pid = str(data["message_id"])
            if out.delivery_key:
                self._early_bound.pop(out.delivery_key, None)
            if provisional is not None:
                self._delivered.pop(provisional, None)
                self._sent_payload.pop(provisional, None)
            self._remember_sent(pid, out)
            return pid
        # retcode != 0 (e.g. -1): ambiguous — the platform returned no real
        # message id. Return None (the outbox writes a synthetic local echo
        # that a later real self echo reconciles) and retain the exact sent
        # payload + delivery key under a stable local id so the real echo can
        # be matched unambiguously and bound to the key.
        if out.delivery_key and out.delivery_key in self._early_bound:
            return self._early_bound.pop(out.delivery_key)
        local_id = provisional or self._fallback_local_id(out)
        self._remember(self._sent_payload, local_id, self._sent_payload_value(out))
        if out.delivery_key:
            self._remember(self._delivered, local_id, out.delivery_key)
        return None

    def _fallback_local_id(self, out: Outgoing) -> str:
        """A stable local id for an ambiguous send: deterministic per delivery
        key when one exists, else a sequential id."""
        if out.delivery_key:
            return f"{_LOCAL_ID_PREFIX}{out.delivery_key}"
        local_id = f"{_LOCAL_ID_PREFIX}seq:{self._local_seq}"
        self._local_seq += 1
        return local_id

    def _sent_payload_value(
        self, out: Outgoing
    ) -> tuple[str, str, tuple[Segment, ...], tuple[Segment, ...], MessageId | None]:
        """(chat_key, text, wire_segments, durable_segments, reply_to).

        ``wire_segments`` is what the platform received (used to match a real
        echo unambiguously); ``durable_segments`` is exactly what the outbox
        row stores (``tuple(out.segments)``), so correlating the echo to it
        makes the durable reconciliation match byte-for-byte."""
        return (
            out.chat_key,
            out.text,
            self._outgoing_segments(out),
            tuple(out.segments),
            out.reply_to,
        )

    def _remember_sent(self, pid: str, out: Outgoing) -> None:
        if out.delivery_key:
            self._remember(self._delivered, pid, out.delivery_key)
        self._remember(self._sent_payload, pid, self._sent_payload_value(out))

    def _remember(self, store: dict, key: str, value: Any) -> None:
        store[key] = value
        if len(store) > self._max_remember:
            for k in list(store)[: len(store) - self._max_remember]:
                del store[k]

    def _new_echo(self) -> str:
        self._echo_seq += 1
        return f"ob:{self._echo_seq}"

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Stop background tasks, fail pending sends, close the connection and
        the reverse server. Idempotent and cancellation-safe."""
        if self._closed:
            return
        self._closed = True
        self._ready_event.set()
        self._early_bound.clear()
        for task in (self._watchdog_task, self._reconnect_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        await self._stop_receiver()
        if self._probe_task is not None:
            self._probe_task.cancel()
            try:
                await self._probe_task
            except (asyncio.CancelledError, Exception):
                pass
            self._probe_task = None
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        for task in list(self._media_tasks.values()):
            task.cancel()
        self._media_tasks.clear()
        for task in list(self._forward_tasks.values()):
            task.cancel()
        self._forward_tasks.clear()
        self._forward_contents.clear()
        conn = self._conn
        if conn is not None:
            try:
                await conn.close(CloseCode.GOING_AWAY, "adapter closing")
            except Exception:
                pass
            self._conn = None
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass
            self._server = None
        self._conn_event.set()
        # Wake any blocked events() consumer so it returns.
        try:
            self._rx_queue.put_nowait(None)
        except asyncio.QueueFull:
            # Consumers already have queued events; their next loop observes
            # ``_closed`` and exits without requiring an additional sentinel.
            pass

    @property
    def connected(self) -> bool:
        return self._conn is not None and self._is_open(self._conn)
