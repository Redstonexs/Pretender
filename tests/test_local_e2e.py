"""One fully local production-composition round trip.

This test deliberately composes the real live stack.  The only peers are a
stdlib asyncio HTTP server for the OpenAI-compatible API and the existing
reverse-WebSocket OneBot fixture client.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import signal
import socket
import sys
from pathlib import Path
from typing import Any

import orjson
import pytest
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from pretender.adapters.onebot import OneBotAdapter
from pretender.app import App
from pretender.clock import RealClock
from pretender.config import Config
from pretender.db import Database
from pretender.llm import OpenAIClient
from pretender.outbox import OutboxDriver
from pretender.scheduler import LedgerScheduler
from pretender.types import Message
from tests.onebot_fixtures import FIXTURES
from tests.test_onebot import FakeOneBot


class LocalOpenAI:
    """Small local HTTP/1.1 server for the two OpenAI chat calls."""

    def __init__(self, *, first_planner_delay_s: float = 0.0) -> None:
        self.server: asyncio.AbstractServer | None = None
        self.first_planner_delay_s = first_planner_delay_s
        self._first_planner_delay_used = False
        self.first_planner_delay_started = asyncio.Event()
        self.first_planner_delay_finished = asyncio.Event()
        self.requests: list[dict[str, Any]] = []
        self.auth_present: list[bool] = []
        self.request_events = [asyncio.Event() for _ in range(3)]
        self.errors: list[str] = []

    async def start(self) -> int:
        self.server = await asyncio.start_server(
            self._handle, "127.0.0.1", 0
        )
        socket = self.server.sockets[0]
        return int(socket.getsockname()[1])

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            header_bytes = await reader.readuntil(b"\r\n\r\n")
            lines = header_bytes[:-4].decode("latin-1").split("\r\n")
            headers: dict[str, str] = {}
            for line in lines[1:]:
                name, value = line.split(":", 1)
                headers[name.lower()] = value.strip()
            length = int(headers.get("content-length", "0"))
            body = json.loads((await reader.readexactly(length)).decode("utf-8"))

            # Keep the captured request diagnostic-safe: only presence, never
            # the Authorization value, is retained.
            self.auth_present.append(bool(headers.get("authorization")))
            safe_headers = {
                name: ("<redacted>" if name == "authorization" else value)
                for name, value in headers.items()
            }
            index = len(self.requests)
            self.requests.append({"headers": safe_headers, "body": body})
            if index < len(self.request_events):
                self.request_events[index].set()

            model = body.get("model")
            if model == "planner-local":
                response = {
                    "id": "local-planner-1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-reply-1",
                                        "type": "function",
                                        "function": {
                                            "name": "reply",
                                            "arguments": json.dumps(
                                                {"text": "compose a local reply"},
                                                separators=(",", ":"),
                                            ),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 3},
                }
            elif model == "reply-local":
                response = {
                    "id": "local-reply-1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "local production reply",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 4},
                }
            else:
                self.errors.append(f"unexpected model: {model!r}")
                await self._write_response(writer, 400, {"error": "unexpected model"})
                return
            if (
                model == "planner-local"
                and self.first_planner_delay_s > 0
                and not self._first_planner_delay_used
            ):
                self._first_planner_delay_used = True
                self.first_planner_delay_started.set()
                try:
                    await asyncio.sleep(self.first_planner_delay_s)
                finally:
                    self.first_planner_delay_finished.set()
            await self._write_response(writer, 200, response)
        except (
            asyncio.IncompleteReadError,
            ConnectionClosed,
            ConnectionResetError,
            BrokenPipeError,
        ):
            return
        except Exception as exc:  # surface server failures in the test body
            self.errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    @staticmethod
    async def _write_response(
        writer: asyncio.StreamWriter, status: int, payload: dict[str, Any]
    ) -> None:
        body = orjson.dumps(payload)
        writer.write(
            f"HTTP/1.1 {status} {'OK' if status == 200 else 'Bad Request'}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n".encode("ascii")
            + body
        )
        await writer.drain()


async def _close_task(task: asyncio.Task[Any] | None) -> None:
    if task is None or task.done():
        if task is not None:
            try:
                task.result()
            except BaseException:
                pass
        return
    try:
        await asyncio.wait_for(task, timeout=3.0)
    except BaseException:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except BaseException:
            pass


def _observe_messages(
    app: App, inbound: asyncio.Event, self_echo: asyncio.Event
) -> list[Any]:
    """Observe completion after the real App ingest path returns."""
    assert app.ingest is not None
    results: list[Any] = []
    original_handle = app.ingest.handle

    async def handle(event: Any, **kwargs: Any) -> Any:
        result = await original_handle(event, **kwargs)
        payload = event.payload
        if isinstance(payload, Message):
            if payload.is_self:
                results.append(result)
                self_echo.set()
            elif str(payload.id) == "12347":
                inbound.set()
        return result

    app.ingest.handle = handle  # type: ignore[method-assign]
    return results


async def _handshake(client: FakeOneBot, app: App) -> None:
    raw = await asyncio.wait_for(client.ws.recv(), timeout=3.0)
    probe = orjson.loads(raw)
    client.actions.append(probe)
    assert probe["action"] == "get_login_info"
    await client.respond(probe["echo"], retcode=0, data={"user_id": 10001})
    await client.send_event(copy.deepcopy(FIXTURES["lifecycle"]))
    await asyncio.wait_for(app._receiver_active.wait(), timeout=3.0)


def test_local_production_composition_restart_dedup(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    async def scenario() -> None:
        monkeypatch.setenv("PRETENDER_LOCAL_OPENAI_KEY", "local-test-secret")
        provider = LocalOpenAI()
        port = await provider.start()
        db_path = tmp_path / "runtime" / "pretender.db"
        corpus_path = tmp_path / "runtime" / "corpus.jsonl"
        cfg = Config.from_dict(
            {
                "storage": {"db_path": str(db_path)},
                "output": {"max_split": 1, "typo_rate": 0.0},
                "agent": {
                    "dispatch_lease_s": 5.0,
                    "max_execution_s": 5.0,
                    "retry_delay_s": 0.1,
                },
                "adapter": {
                    "name": "onebot",
                    "onebot": {
                        "mode": "reverse_ws",
                        "host": "127.0.0.1",
                        "port": 0,
                        "heartbeat_timeout_s": None,
                        "action_timeout_s": 1.0,
                    },
                },
                "llm": {
                    "profiles": {
                        "planner": {
                            "base_url": f"http://127.0.0.1:{port}/v1",
                            "api_key": "${PRETENDER_LOCAL_OPENAI_KEY}",
                            "model": "planner-local",
                            "temperature": 0.0,
                            "max_tokens": 64,
                            "timeout_s": 1.0,
                        },
                        "reply": {
                            "base_url": f"http://127.0.0.1:{port}/v1",
                            "api_key": "${PRETENDER_LOCAL_OPENAI_KEY}",
                            "model": "reply-local",
                            "temperature": 0.0,
                            "max_tokens": 64,
                            "timeout_s": 1.0,
                        },
                    }
                },
            }
        )

        app_a: App | None = None
        app_b: App | None = None
        task_a: asyncio.Task[Any] | None = None
        task_b: asyncio.Task[Any] | None = None
        client_a: FakeOneBot | None = None
        client_b: FakeOneBot | None = None
        try:
            clock_a = RealClock()
            adapter_a = OneBotAdapter(
                config=cfg.adapter.onebot, clock=clock_a, normalize_media=False
            )
            app_a = App.build(
                cfg,
                clock=clock_a,
                adapter=adapter_a,
                recorder_path=corpus_path,
                dry_run=False,
            )
            assert isinstance(app_a._llm, OpenAIClient)
            assert isinstance(app_a.outbox, OutboxDriver)
            assert isinstance(app_a.scheduler, LedgerScheduler)
            inbound_a = asyncio.Event()
            self_echo_a = asyncio.Event()
            echo_results_a = _observe_messages(app_a, inbound_a, self_echo_a)
            adapter_ready_a = asyncio.Event()
            original_connect_a = adapter_a.connect

            async def observe_connect_a() -> None:
                await original_connect_a()
                adapter_ready_a.set()

            adapter_a.connect = observe_connect_a  # type: ignore[method-assign]
            send_done = asyncio.Event()
            send_returns: list[str | None] = []
            original_send = adapter_a.send

            async def observe_send(out: Any) -> str | None:
                result = await original_send(out)
                send_returns.append(result)
                send_done.set()
                return result

            adapter_a.send = observe_send  # type: ignore[method-assign]
            pump_done = asyncio.Event()
            original_pump = app_a.outbox.pump

            async def observe_pump(*args: Any, **kwargs: Any) -> int:
                sent = await original_pump(*args, **kwargs)
                if sent:
                    pump_done.set()
                return sent

            app_a.outbox.pump = observe_pump  # type: ignore[method-assign]
            task_a = asyncio.create_task(app_a.run())
            await asyncio.wait_for(adapter_ready_a.wait(), timeout=3.0)
            client_a = await FakeOneBot(adapter_a._server.sockets[0].getsockname()[1]).connect()
            await _handshake(client_a, app_a)

            await client_a.send_event(copy.deepcopy(FIXTURES["group_at"]))
            await asyncio.wait_for(inbound_a.wait(), timeout=3.0)
            action = await client_a.next_action(timeout=3.0)
            assert action["action"] == "send_group_msg"
            assert action["params"]["group_id"] == 111111
            assert action["params"]["message"] == [
                {"type": "text", "data": {"text": "local production reply"}}
            ]
            await client_a.respond(
                action["echo"], retcode=0, data={"message_id": 90001}
            )
            await asyncio.wait_for(send_done.wait(), timeout=3.0)
            assert send_returns == ["90001"]
            await asyncio.wait_for(pump_done.wait(), timeout=3.0)

            self_echo = copy.deepcopy(FIXTURES["self_echo_group"])
            self_echo["message_id"] = 90001
            self_echo["message"] = copy.deepcopy(action["params"]["message"])
            self_echo["raw_message"] = "local production reply"
            await client_a.send_event(self_echo)
            await asyncio.wait_for(self_echo_a.wait(), timeout=3.0)
            await asyncio.wait_for(provider.request_events[1].wait(), timeout=3.0)

            assert provider.errors == []
            assert len(provider.requests) == 2
            assert [request["body"]["model"] for request in provider.requests] == [
                "planner-local",
                "reply-local",
            ]
            assert provider.auth_present == [True, True]
            assert all(
                request["headers"].get("authorization") == "<redacted>"
                for request in provider.requests
            )
            planner_body = provider.requests[0]["body"]
            assert planner_body["stream"] is False
            assert planner_body["messages"][0]["role"] == "system"
            assert any(
                message.get("role") == "user"
                and "你好" in message.get("content", "")
                for message in planner_body["messages"]
            )
            reply_tool = next(
                tool for tool in planner_body["tools"]
                if tool["function"]["name"] == "reply"
            )
            assert reply_tool["type"] == "function"
            assert "text" in reply_tool["function"]["parameters"]["properties"]
            reply_body = provider.requests[1]["body"]
            assert [message["role"] for message in reply_body["messages"]] == [
                "system",
                "user",
            ]
            # The final user turn carries the current time and the staged
            # reference, mirroring MaiBot's replyer request.
            final_turn = reply_body["messages"][1]["content"]
            assert "当前时间：" in final_turn
            assert "【回复信息参考】\ncompose a local reply" in final_turn
            assert "tools" not in reply_body
            diagnostics = app_a._llm.request_dump(
                "planner", {"messages": [{"role": "user", "content": "hidden"}]}
            )
            assert diagnostics["headers"]["Authorization"] == "***"
            assert "local-test-secret" not in repr(diagnostics)
            assert echo_results_a and echo_results_a[0].echo_status in (
                "reconciled",
                "already_reconciled",
            )

            state_db = Database(db_path)
            await state_db.open()
            rows = await state_db.read(
                lambda conn: {
                    "outbox": conn.execute(
                        "SELECT state, platform_msg_id FROM outbox"
                    ).fetchall(),
                    "dispatch": conn.execute(
                        "SELECT state FROM dispatches ORDER BY id"
                    ).fetchall(),
                    "messages": conn.execute(
                        "SELECT platform_msg_id, is_self, text FROM messages ORDER BY id"
                    ).fetchall(),
                }
            )
            await state_db.close()
            assert rows["outbox"] == [("sent", "90001")]
            assert rows["dispatch"] == [("completed",)]
            assert rows["messages"] == [
                ("12347", 0, "@麦麦 你好"),
                ("90001", 1, "local production reply"),
            ]
            assert len([item for item in client_a.actions if item.get("action") == "send_group_msg"]) == 1

            await client_a.close()
            await asyncio.wait_for(adapter_a.close(), timeout=3.0)
            await asyncio.wait_for(task_a, timeout=3.0)
            assert app_a._shutdown is True
            assert app_a._worker is None
            assert app_a._receiver is None
            assert app_a._llm is not None and app_a._llm._closed is True
            assert adapter_a._rx_task is None
            assert adapter_a._server is None

            clock_b = RealClock()
            adapter_b = OneBotAdapter(
                config=cfg.adapter.onebot, clock=clock_b, normalize_media=False
            )
            app_b = App.build(
                cfg,
                clock=clock_b,
                adapter=adapter_b,
                recorder_path=corpus_path,
                dry_run=False,
            )
            inbound_b = asyncio.Event()
            _observe_messages(app_b, inbound_b, asyncio.Event())
            adapter_ready_b = asyncio.Event()
            original_connect_b = adapter_b.connect

            async def observe_connect_b() -> None:
                await original_connect_b()
                adapter_ready_b.set()

            adapter_b.connect = observe_connect_b  # type: ignore[method-assign]
            task_b = asyncio.create_task(app_b.run())
            await asyncio.wait_for(adapter_ready_b.wait(), timeout=3.0)
            client_b = await FakeOneBot(adapter_b._server.sockets[0].getsockname()[1]).connect()
            await _handshake(client_b, app_b)
            await client_b.send_event(copy.deepcopy(FIXTURES["group_at"]))
            await asyncio.wait_for(inbound_b.wait(), timeout=3.0)
            assert len(provider.requests) == 2
            assert [item for item in client_b.actions if item.get("action") == "send_group_msg"] == []
            assert len([item for item in client_a.actions if item.get("action") == "send_group_msg"]) == 1
            await client_b.close()
            await asyncio.wait_for(adapter_b.close(), timeout=3.0)
            await asyncio.wait_for(task_b, timeout=3.0)
            assert app_b._worker is None
            assert app_b._receiver is None
            assert app_b._llm is not None and app_b._llm._closed is True
            assert adapter_b._rx_task is None
            assert adapter_b._server is None
            assert corpus_path.exists()
            corpus_text = corpus_path.read_text(encoding="utf-8")
            assert '"id":"12347"' in corpus_text
            assert '"id":"90001"' in corpus_text
        finally:
            if client_a is not None:
                try:
                    await asyncio.wait_for(client_a.close(), timeout=2.0)
                except BaseException:
                    pass
            if client_b is not None:
                try:
                    await asyncio.wait_for(client_b.close(), timeout=2.0)
                except BaseException:
                    pass
            if app_a is not None and not app_a._shutdown:
                try:
                    await asyncio.wait_for(app_a.shutdown(), timeout=3.0)
                except BaseException:
                    pass
            if app_b is not None and not app_b._shutdown:
                try:
                    await asyncio.wait_for(app_b.shutdown(), timeout=3.0)
                except BaseException:
                    pass
            await _close_task(task_a)
            await _close_task(task_b)
            await asyncio.wait_for(provider.close(), timeout=3.0)

    asyncio.run(scenario())


def test_local_production_composition_provider_timeout_retry(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    async def scenario() -> None:
        monkeypatch.setenv("PRETENDER_LOCAL_OPENAI_KEY", "local-retry-secret")
        provider = LocalOpenAI(first_planner_delay_s=0.4)
        db_path = tmp_path / "runtime" / "pretender.db"
        corpus_path = tmp_path / "runtime" / "corpus.jsonl"
        app: App | None = None
        task: asyncio.Task[Any] | None = None
        client: FakeOneBot | None = None
        adapter: OneBotAdapter | None = None
        try:
            port = await provider.start()
            cfg = Config.from_dict(
                {
                    "storage": {"db_path": str(db_path)},
                    "output": {"max_split": 1, "typo_rate": 0.0},
                    "learn": {"enabled": False},
                    "media": {"enabled": False},
                    "agent": {
                        "dispatch_lease_s": 5.0,
                        "max_execution_s": 5.0,
                        "retry_delay_s": 0.2,
                    },
                    "adapter": {
                        "name": "onebot",
                        "onebot": {
                            "mode": "reverse_ws",
                            "host": "127.0.0.1",
                            "port": 0,
                            "heartbeat_timeout_s": None,
                            "action_timeout_s": 1.0,
                        },
                    },
                    "llm": {
                        "profiles": {
                            "planner": {
                                "base_url": f"http://127.0.0.1:{port}/v1",
                                "api_key": "${PRETENDER_LOCAL_OPENAI_KEY}",
                                "model": "planner-local",
                                "temperature": 0.0,
                                "max_tokens": 64,
                                "timeout_s": 0.05,
                            },
                            "reply": {
                                "base_url": f"http://127.0.0.1:{port}/v1",
                                "api_key": "${PRETENDER_LOCAL_OPENAI_KEY}",
                                "model": "reply-local",
                                "temperature": 0.0,
                                "max_tokens": 64,
                                "timeout_s": 0.05,
                            },
                        }
                    },
                }
            )

            clock = RealClock()
            adapter = OneBotAdapter(
                config=cfg.adapter.onebot, clock=clock, normalize_media=False
            )
            app = App.build(
                cfg,
                clock=clock,
                adapter=adapter,
                recorder_path=corpus_path,
                dry_run=False,
            )
            assert isinstance(app._llm, OpenAIClient)
            assert isinstance(app.outbox, OutboxDriver)
            assert isinstance(app.scheduler, LedgerScheduler)

            inbound = asyncio.Event()
            _observe_messages(app, inbound, asyncio.Event())
            adapter_ready = asyncio.Event()
            original_connect = adapter.connect

            async def observe_connect() -> None:
                await original_connect()
                adapter_ready.set()

            adapter.connect = observe_connect  # type: ignore[method-assign]
            send_done = asyncio.Event()
            send_returns: list[str | None] = []
            original_send = adapter.send

            async def observe_send(out: Any) -> str | None:
                result = await original_send(out)
                send_returns.append(result)
                send_done.set()
                return result

            adapter.send = observe_send  # type: ignore[method-assign]
            pump_done = asyncio.Event()
            original_pump = app.outbox.pump

            async def observe_pump(*args: Any, **kwargs: Any) -> int:
                sent = await original_pump(*args, **kwargs)
                if sent:
                    pump_done.set()
                return sent

            app.outbox.pump = observe_pump  # type: ignore[method-assign]
            task = asyncio.create_task(app.run())
            await asyncio.wait_for(adapter_ready.wait(), timeout=3.0)
            client = await FakeOneBot(
                adapter._server.sockets[0].getsockname()[1]
            ).connect()
            await _handshake(client, app)

            await client.send_event(copy.deepcopy(FIXTURES["group_at"]))
            await asyncio.wait_for(inbound.wait(), timeout=3.0)
            await asyncio.wait_for(
                provider.first_planner_delay_started.wait(), timeout=3.0
            )

            # The delayed first response outlives the configured client
            # timeout.  The bounded wait is deliberately shorter than the
            # scripted delay, proving the first provider call cannot finish.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    provider.first_planner_delay_finished.wait(), timeout=0.12
                )
            assert len(provider.requests) == 1
            assert [
                item for item in client.actions if item.get("action") == "send_group_msg"
            ] == []
            assert send_returns == []

            state_db = Database(db_path)
            await state_db.open()
            try:
                intermediate = await state_db.read(
                    lambda conn: {
                        "outbox": conn.execute(
                            "SELECT state, platform_msg_id FROM outbox"
                        ).fetchall(),
                        "dispatch": conn.execute(
                            "SELECT state FROM dispatches ORDER BY id"
                        ).fetchall(),
                        "barrier": conn.execute(
                            "SELECT agent_resume_at FROM chats"
                        ).fetchall(),
                    }
                )
            finally:
                await state_db.close()
            assert intermediate["outbox"] == []
            assert intermediate["dispatch"] == [("released",)]
            assert intermediate["barrier"] and intermediate["barrier"][0][0] is not None

            # The retry is a fresh planner call; only after it succeeds does
            # the reply call create the single durable output row.
            await asyncio.wait_for(provider.request_events[1].wait(), timeout=3.0)
            await asyncio.wait_for(provider.request_events[2].wait(), timeout=3.0)
            action = await client.next_action(timeout=3.0)
            assert action["action"] == "send_group_msg"
            assert action["params"]["group_id"] == 111111
            assert action["params"]["message"] == [
                {"type": "text", "data": {"text": "local production reply"}}
            ]
            assert isinstance(action.get("echo"), str) and action["echo"]
            await client.respond(
                action["echo"], retcode=0, data={"message_id": 90003}
            )
            await asyncio.wait_for(send_done.wait(), timeout=3.0)
            await asyncio.wait_for(pump_done.wait(), timeout=3.0)
            await asyncio.wait_for(
                provider.first_planner_delay_finished.wait(), timeout=3.0
            )

            assert provider.errors == []
            assert [request["body"]["model"] for request in provider.requests] == [
                "planner-local",
                "planner-local",
                "reply-local",
            ]
            assert provider.auth_present == [True, True, True]
            assert all(
                request["headers"].get("authorization") == "<redacted>"
                for request in provider.requests
            )
            assert "local-retry-secret" not in repr(provider.requests)
            assert send_returns == ["90003"]

            state_db = Database(db_path)
            await state_db.open()
            try:
                final = await state_db.read(
                    lambda conn: {
                        "outbox": conn.execute(
                            "SELECT state, platform_msg_id FROM outbox"
                        ).fetchall(),
                        "dispatch": conn.execute(
                            "SELECT state FROM dispatches ORDER BY id"
                        ).fetchall(),
                    }
                )
            finally:
                await state_db.close()
            assert final["outbox"] == [("sent", "90003")]
            assert final["dispatch"] == [("released",), ("completed",)]
            assert len(
                [item for item in client.actions if item.get("action") == "send_group_msg"]
            ) == 1
        finally:
            if client is not None:
                try:
                    await asyncio.wait_for(client.close(), timeout=2.0)
                except BaseException:
                    pass
            if app is not None and not app._shutdown:
                try:
                    await asyncio.wait_for(app.shutdown(), timeout=3.0)
                except BaseException:
                    pass
            elif adapter is not None:
                try:
                    await asyncio.wait_for(adapter.close(), timeout=3.0)
                except BaseException:
                    pass
            await _close_task(task)
            try:
                await asyncio.wait_for(provider.close(), timeout=3.0)
            except BaseException:
                pass

    asyncio.run(scenario())


def test_cli_sigterm_graceful_shutdown(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        monkeypatch.setenv("PRETENDER_CLI_OPENAI_KEY", "cli-local-secret")
        provider = LocalOpenAI()
        provider_port = await provider.start()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            onebot_port = int(probe.getsockname()[1])

        db_path = tmp_path / "cli-runtime" / "pretender.db"
        corpus_path = tmp_path / "cli-runtime" / "pretender.jsonl"
        config_path = tmp_path / "cli-runtime" / "config.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "\n".join(
                [
                    "[storage]",
                    f'db_path = "{db_path}"',
                    "",
                    '[adapter]',
                    'name = "onebot"',
                    "",
                    "[adapter.onebot]",
                    'mode = "reverse_ws"',
                    'host = "127.0.0.1"',
                    f"port = {onebot_port}",
                    'path = "/onebot/v11/ws"',
                    "heartbeat_timeout_s = 30.0",
                    "action_timeout_s = 1.0",
                    "ping_timeout_s = 1.0",
                    "",
                    "[output]",
                    "max_split = 1",
                    "typo_rate = 0.0",
                    "",
                    "[learn]",
                    "enabled = false",
                    "",
                    "[media]",
                    "enabled = false",
                    "",
                    "[llm.profiles.planner]",
                    f'base_url = "http://127.0.0.1:{provider_port}/v1"',
                    'api_key = "${PRETENDER_CLI_OPENAI_KEY}"',
                    'model = "planner-local"',
                    "temperature = 0.0",
                    "max_tokens = 64",
                    "timeout_s = 1.0",
                    "",
                    "[llm.profiles.reply]",
                    f'base_url = "http://127.0.0.1:{provider_port}/v1"',
                    'api_key = "${PRETENDER_CLI_OPENAI_KEY}"',
                    'model = "reply-local"',
                    "temperature = 0.0",
                    "max_tokens = 64",
                    "timeout_s = 1.0",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["PRETENDER_CLI_OPENAI_KEY"] = "cli-local-secret"
        process: asyncio.subprocess.Process | None = None
        client: FakeOneBot | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pretender",
                "run",
                "--live",
                "--config",
                str(config_path),
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            client = FakeOneBot(onebot_port)
            # Use bounded direct connection retries: breaking out of
            # websockets' reconnect iterator would close the yielded peer.
            async with asyncio.timeout(5.0):
                while client.ws is None:
                    try:
                        await asyncio.wait_for(client.connect(), timeout=0.5)
                    except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
                        await asyncio.sleep(0)
            assert client.ws is not None
            raw = await asyncio.wait_for(client.ws.recv(), timeout=3.0)
            probe = orjson.loads(raw)
            client.actions.append(probe)
            assert probe["action"] == "get_login_info"
            await client.respond(
                probe["echo"], retcode=0, data={"user_id": 10001}
            )
            await client.send_event(copy.deepcopy(FIXTURES["lifecycle"]))

            # A real message proves the app got past the adapter handshake and
            # is consuming events in its live receiver before SIGTERM arrives.
            await client.send_event(copy.deepcopy(FIXTURES["group_at"]))
            await asyncio.wait_for(provider.request_events[0].wait(), timeout=3.0)
            await asyncio.wait_for(provider.request_events[1].wait(), timeout=3.0)
            action = await client.next_action(timeout=3.0)
            assert action["action"] == "send_group_msg"
            await client.respond(action["echo"], retcode=0, data={"message_id": 90002})

            assert process is not None
            process.send_signal(signal.SIGTERM)
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=5.0
            )
            assert process.returncode == 0, (
                f"CLI exited with {process.returncode}; stdout={stdout!r}; "
                f"stderr={stderr!r}"
            )
            assert b"Traceback" not in stderr
            await asyncio.wait_for(client.ws.wait_closed(), timeout=3.0)
            assert client.ws.state is State.CLOSED

            assert db_path.exists()
            assert corpus_path.exists()
            db = Database(db_path)
            await asyncio.wait_for(db.open(), timeout=3.0)
            version = await asyncio.wait_for(
                db.read(
                    lambda conn: conn.execute("PRAGMA user_version").fetchone()[0]
                ),
                timeout=3.0,
            )
            await asyncio.wait_for(db.close(), timeout=3.0)
            assert version == 15
        finally:
            if client is not None:
                try:
                    await asyncio.wait_for(client.close(), timeout=2.0)
                except BaseException:
                    pass
            if process is not None and process.returncode is None:
                try:
                    process.send_signal(signal.SIGTERM)
                    await asyncio.wait_for(process.communicate(), timeout=2.0)
                except BaseException:
                    if process.returncode is None:
                        process.kill()
                    try:
                        await asyncio.wait_for(process.communicate(), timeout=2.0)
                    except BaseException:
                        pass
            await asyncio.wait_for(provider.close(), timeout=3.0)

    asyncio.run(scenario())
