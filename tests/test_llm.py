"""OpenAI-compatible client: request shape, tool calls, malformed payloads,
usage, deadline, status/connection/timeout translation, embeddings, close
idempotence, and secret redaction.

Async tests run via asyncio.run() so the test extra stays at just pytest.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from pretender.clock import VirtualClock
from pretender.config import LLMConfig, LLMProfile
from pretender.errors import (
    LLMPermanentError,
    LLMTransientError,
    TransientError,
)
from pretender.llm import OpenAIClient, is_retryable, redact_request, scrub_credentials
from pretender.types import ToolCall, ToolCallId, TranscriptMessage


def run(coro):
    return asyncio.run(coro)


def _config() -> LLMConfig:
    return LLMConfig(
        profiles={
            "main": LLMProfile(
                base_url="https://api.example.com/v1",
                api_key="sk-test",
                model="test-chat",
                temperature=0.7,
                max_tokens=1200,
                timeout_s=45.0,
            ),
            "embed": LLMProfile(
                base_url="https://api.example.com/v1",
                api_key="sk-test",
                model="test-embed",
                timeout_s=45.0,
            ),
            "alt": LLMProfile(
                base_url="https://alt.example.com/v1",
                model="alt-model",
            ),
        }
    )


def _msgs(*, with_tools: bool = False) -> list[TranscriptMessage]:
    msgs = [
        TranscriptMessage(role="system", content="be nice"),
        TranscriptMessage(role="user", content="hello"),
    ]
    if with_tools:
        msgs += [
            TranscriptMessage(
                role="assistant",
                content=None,
                tool_calls=(
                    ToolCall(id=ToolCallId("c1"), name="reply", arguments={"text": "yo"}),
                ),
            ),
            TranscriptMessage(
                role="tool", tool_call_id=ToolCallId("c1"), name="reply", content="ok"
            ),
        ]
    return msgs


class RecordingClient:
    """Duck-typed httpx.AsyncClient: records every post and returns a
    prepared response."""

    def __init__(
        self, *, status_code: int = 200, json: object = None, text: str | None = None
    ) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.aclose_calls = 0
        self._status_code = status_code
        self._json = json
        self._text = text

    async def post(self, url: str, **kwargs: object) -> httpx.Response:
        self.calls.append((url, kwargs))
        req = httpx.Request("POST", url)
        if self._json is not None:
            return httpx.Response(self._status_code, json=self._json, request=req)
        return httpx.Response(self._status_code, text=self._text or "", request=req)

    async def aclose(self) -> None:
        self.aclose_calls += 1


def _recording_transport(handler):
    """A MockTransport that records every request it serves."""
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return seen, httpx.MockTransport(wrapped)


# ── request shape ────────────────────────────────────────────────────────────

def test_complete_request_shape():
    fake = RecordingClient(
        json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    )
    client = OpenAIClient(_config(), client=fake)
    resp = run(client.complete(_msgs(), profile="main"))

    assert resp.content == "hi"
    assert len(fake.calls) == 1
    url, kwargs = fake.calls[0]
    assert url == "https://api.example.com/v1/chat/completions"
    assert kwargs["headers"] == {"Authorization": "Bearer sk-test"}
    body = kwargs["json"]
    assert body["model"] == "test-chat"
    assert body["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello"},
    ]
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 1200
    assert body["stream"] is False
    assert "tools" not in body
    assert kwargs["timeout"] == 45.0


def test_complete_serializes_tool_turn_wire():
    fake = RecordingClient(
        json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    )
    client = OpenAIClient(_config(), client=fake)
    run(client.complete(_msgs(with_tools=True), profile="main"))

    body = fake.calls[0][1]["json"]
    assert body["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "reply", "arguments": '{"text":"yo"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "reply", "content": "ok"},
    ]


def test_complete_attaches_tools_when_provided():
    fake = RecordingClient(
        json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    )
    client = OpenAIClient(_config(), client=fake)
    tools = [
        {
            "type": "function",
            "function": {"name": "reply", "description": "reply", "parameters": {}},
        }
    ]
    run(client.complete(_msgs(), profile="main", tools=tools))
    assert fake.calls[0][1]["json"]["tools"] == tools

    # an empty list is not attached
    run(client.complete(_msgs(), profile="main", tools=[]))
    assert "tools" not in fake.calls[1][1]["json"]


def test_complete_overrides_profile_defaults():
    fake = RecordingClient(
        json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    )
    client = OpenAIClient(_config(), client=fake)
    run(client.complete(_msgs(), profile="main", temperature=0.1, max_tokens=5))
    body = fake.calls[0][1]["json"]
    assert body["temperature"] == 0.1
    assert body["max_tokens"] == 5


def test_profile_selection_and_unknown_profile():
    fake = RecordingClient(
        json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    )
    client = OpenAIClient(_config(), client=fake)
    run(client.complete(_msgs(), profile="alt"))
    assert fake.calls[0][0] == "https://alt.example.com/v1/chat/completions"
    assert fake.calls[0][1]["json"]["model"] == "alt-model"
    # no api_key on the alt profile -> no Authorization header
    assert fake.calls[0][1]["headers"] == {}

    with pytest.raises(LLMPermanentError) as ei:
        run(client.complete(_msgs(), profile="nope"))
    assert "no LLM profile named 'nope'" in str(ei.value)
    assert not is_retryable(ei.value)


# ── response parsing: tool calls, usage, content ────────────────────────────

def test_complete_parses_tool_calls_and_usage():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "query_memory",
                                "arguments": '{"q":"cats","k":3}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "reply", "arguments": "{}"},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
    }
    client = OpenAIClient(_config(), client=RecordingClient(json=payload))
    resp = run(client.complete(_msgs(), profile="main"))

    assert resp.content is None
    assert resp.finish_reason == "tool_calls"
    assert [c.id for c in resp.tool_calls] == ["call_1", "call_2"]
    assert resp.tool_calls[0].name == "query_memory"
    assert resp.tool_calls[0].arguments == {"q": "cats", "k": 3}
    assert resp.tool_calls[1].arguments == {}
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16}


def test_complete_parses_multimodal_content_and_missing_usage():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                },
                "finish_reason": "stop",
            }
        ]
    }
    client = OpenAIClient(_config(), client=RecordingClient(json=payload))
    resp = run(client.complete(_msgs(), profile="main"))
    assert resp.content == "ab"
    assert resp.usage == {}
    assert resp.tool_calls == ()


def test_complete_multimodal_wire_payload():
    """Image markdown in a user message serializes into OpenAI-compatible
    multimodal content parts (text + image_url), preserving ordinary text."""
    fake = RecordingClient(
        json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    )
    client = OpenAIClient(_config(), client=fake)
    msgs = [
        TranscriptMessage(role="system", content="be nice"),
        TranscriptMessage(
            role="user",
            content="look at ![pic](https://x/y.png) and ![two](https://x/z.png) done",
        ),
    ]
    run(client.complete(msgs, profile="main"))

    body = fake.calls[0][1]["json"]
    user_wire = body["messages"][1]
    assert user_wire["role"] == "user"
    assert user_wire["content"] == [
        {"type": "text", "text": "look at "},
        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        {"type": "text", "text": " and "},
        {"type": "image_url", "image_url": {"url": "https://x/z.png"}},
        {"type": "text", "text": " done"},
    ]
    # ordinary text without images stays a plain string
    run(client.complete(_msgs(), profile="main"))
    assert fake.calls[1][1]["json"]["messages"][1]["content"] == "hello"


def test_complete_multimodal_wire_round_trips_text():
    """deserialize of a multimodal wire keeps the ordinary text."""
    from pretender.context import deserialize

    wire = [
        {"role": "user", "content": [
            {"type": "text", "text": "see "},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
            {"type": "text", "text": " now"},
        ]}
    ]
    msgs = deserialize(wire)
    assert msgs[0].role == "user"
    assert msgs[0].content == "see  now"


# ── malformed provider payload (fail closed) ────────────────────────────────

@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{"message": "oops"}]},
        {"choices": [{"message": {"role": "assistant", "content": "x", "tool_calls": "nope"}}]},
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "x",
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "f", "arguments": "{}"}},
                            {"id": "c1", "function": {"name": "f", "arguments": "{}"}},
                        ],
                    }
                }
            ]
        },
    ],
)
def test_complete_malformed_payload_fails_closed(payload):
    client = OpenAIClient(_config(), client=RecordingClient(json=payload))
    with pytest.raises(LLMPermanentError):
        run(client.complete(_msgs(), profile="main"))


def test_complete_malformed_arguments_reach_tolerant_path():
    """Structured malformed provider tool arguments must NOT abort the
    completion permanently: the call survives with its id and the raw
    arguments preserved for the tolerant one-repair/no_action path."""
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "analysis",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "add", "arguments": "not json at all"},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "add", "arguments": "[1, 2]"},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    client = OpenAIClient(_config(), client=RecordingClient(json=payload))
    resp = run(client.complete(_msgs(), profile="main"))

    assert [c.id for c in resp.tool_calls] == ["call_1", "call_2"]
    assert resp.tool_calls[0].name == "add"
    assert resp.tool_calls[0].arguments == {}
    assert resp.tool_calls[0].raw_arguments == "not json at all"
    assert resp.tool_calls[1].raw_arguments == "[1, 2]"
    assert resp.content == "analysis"  # content coexists with the calls


def test_complete_non_json_body_fails_closed():
    client = OpenAIClient(_config(), client=RecordingClient(text="<html>oops</html>"))
    with pytest.raises(LLMPermanentError):
        run(client.complete(_msgs(), profile="main"))


def test_complete_malformed_transcript_fails_closed():
    # an orphan tool message cannot be serialized into a canonical wire
    bad = [TranscriptMessage(role="tool", tool_call_id=ToolCallId("x"), content="orphan")]
    client = OpenAIClient(_config(), client=RecordingClient(json={}))
    with pytest.raises(LLMPermanentError):
        run(client.complete(bad, profile="main"))


# ── status translation: 429/5xx vs 4xx ───────────────────────────────────────

@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_retryable_statuses_are_transient(status):
    seen, transport = _recording_transport(
        lambda req: httpx.Response(status, json={"error": "boom"}, request=req)
    )
    client = OpenAIClient(_config(), transport=transport)
    with pytest.raises(LLMTransientError) as ei:
        run(client.complete(_msgs(), profile="main"))
    assert str(status) in str(ei.value)
    assert is_retryable(ei.value)
    assert len(seen) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_statuses_are_permanent(status):
    seen, transport = _recording_transport(
        lambda req: httpx.Response(status, json={"error": "nope"}, request=req)
    )
    client = OpenAIClient(_config(), transport=transport)
    with pytest.raises(LLMPermanentError) as ei:
        run(client.complete(_msgs(), profile="main"))
    assert str(status) in str(ei.value)
    assert not is_retryable(ei.value)
    assert len(seen) == 1


# ── connection / timeout translation ────────────────────────────────────────

@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("read timed out"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ConnectError("connection refused"),
        httpx.PoolTimeout("pool timed out"),
        httpx.NetworkError("network down"),
    ],
)
def test_request_errors_translate_to_transient(exc):
    def handler(request):
        raise exc

    seen, transport = _recording_transport(handler)
    client = OpenAIClient(_config(), transport=transport)
    with pytest.raises(LLMTransientError) as ei:
        run(client.complete(_msgs(), profile="main"))
    assert is_retryable(ei.value)
    assert len(seen) == 1


# ── deadline ─────────────────────────────────────────────────────────────────

def test_deadline_already_passed_fails_fast_without_request():
    clock = VirtualClock(epoch=1_700_000_000.0)
    seen, transport = _recording_transport(
        lambda req: httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]}, request=req)
    )
    client = OpenAIClient(_config(), clock=clock, transport=transport)
    with pytest.raises(LLMTransientError) as ei:
        run(client.complete(_msgs(), profile="main", deadline=clock.now() - 1))
    assert "deadline already passed" in str(ei.value)
    assert is_retryable(ei.value)
    assert seen == []  # no provider call was made


def test_deadline_caps_request_timeout_to_remaining():
    clock = VirtualClock(epoch=1_700_000_000.0)
    fake = RecordingClient(
        json={"choices": [{"message": {"role": "assistant", "content": "x"}}]}
    )
    client = OpenAIClient(_config(), clock=clock, client=fake)
    run(client.complete(_msgs(), profile="main", deadline=clock.now() + 5))
    assert fake.calls[0][1]["timeout"] == pytest.approx(5.0)  # min(45, 5)


def test_deadline_beyond_profile_timeout_uses_profile_timeout():
    clock = VirtualClock(epoch=1_700_000_000.0)
    fake = RecordingClient(
        json={"choices": [{"message": {"role": "assistant", "content": "x"}}]}
    )
    client = OpenAIClient(_config(), clock=clock, client=fake)
    run(client.complete(_msgs(), profile="main", deadline=clock.now() + 999))
    assert fake.calls[0][1]["timeout"] == pytest.approx(45.0)


# ── embeddings ───────────────────────────────────────────────────────────────

def test_embed_shape_and_dimension():
    payload = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3], "index": 0},
            {"embedding": [0.4, 0.5, 0.6], "index": 1},
        ]
    }
    fake = RecordingClient(json=payload)
    client = OpenAIClient(_config(), client=fake)
    vectors = run(client.embed(["a", "b"]))

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert len(vectors) == 2
    assert {len(v) for v in vectors} == {3}  # uniform dimension
    url, kwargs = fake.calls[0]
    assert url == "https://api.example.com/v1/embeddings"
    assert kwargs["json"] == {"model": "test-embed", "input": ["a", "b"]}


def test_embed_empty_batch_returns_empty_without_call():
    fake = RecordingClient(json={})
    client = OpenAIClient(_config(), client=fake)
    assert run(client.embed([])) == []
    assert fake.calls == []


def test_embed_inconsistent_dimension_fails_closed():
    payload = {
        "data": [
            {"embedding": [0.1, 0.2], "index": 0},
            {"embedding": [0.1, 0.2, 0.3], "index": 1},
        ]
    }
    client = OpenAIClient(_config(), client=RecordingClient(json=payload))
    with pytest.raises(LLMPermanentError) as ei:
        run(client.embed(["a", "b"]))
    assert "inconsistent dimension" in str(ei.value)


def test_embed_count_mismatch_fails_closed():
    payload = {"data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]}
    client = OpenAIClient(_config(), client=RecordingClient(json=payload))
    with pytest.raises(LLMPermanentError) as ei:
        run(client.embed(["a", "b"]))
    assert "expected 2 vectors" in str(ei.value)


def test_embed_malformed_payload_fails_closed():
    client = OpenAIClient(_config(), client=RecordingClient(json={"data": "nope"}))
    with pytest.raises(LLMPermanentError):
        run(client.embed(["a"]))


# ── close idempotence ────────────────────────────────────────────────────────

def test_aclose_is_idempotent_on_injected_client():
    fake = RecordingClient(json={})
    client = OpenAIClient(_config(), client=fake)
    run(client.aclose())
    run(client.aclose())
    assert fake.aclose_calls == 1


def test_aclose_is_idempotent_on_real_client():
    seen, transport = _recording_transport(
        lambda req: httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]}, request=req)
    )
    client = OpenAIClient(_config(), transport=transport)
    run(client.aclose())
    run(client.aclose())  # must not raise


def test_async_context_manager_closes_once():
    fake = RecordingClient(
        json={"choices": [{"message": {"role": "assistant", "content": "x"}}]}
    )

    async def scenario():
        async with OpenAIClient(_config(), client=fake) as client:
            await client.complete(_msgs(), profile="main")

    run(scenario())
    assert fake.aclose_calls == 1


# ── secret redaction ─────────────────────────────────────────────────────────

def test_redact_request_masks_authorization_and_query():
    dump = redact_request(
        "https://api.example.com/v1/chat/completions?token=sk-querysecret",
        {"Authorization": "Bearer sk-supersecret", "Content-Type": "application/json"},
        {"model": "m"},
    )
    assert dump["headers"]["Authorization"] == "***"
    assert dump["headers"]["Content-Type"] == "application/json"
    assert "sk-querysecret" not in dump["url"]
    assert "token=" not in dump["url"]
    assert "sk-supersecret" not in json.dumps(dump)


def test_no_secret_leakage_in_dump_and_errors():
    cfg = LLMConfig(
        profiles={
            "main": LLMProfile(
                base_url="https://api.example.com/v1?token=sk-querysecret",
                api_key="sk-supersecret",
                model="m",
            )
        }
    )
    seen, transport = _recording_transport(
        lambda req: httpx.Response(400, json={"error": "bad"}, request=req)
    )
    client = OpenAIClient(cfg, transport=transport)
    with pytest.raises(LLMPermanentError) as ei:
        run(client.complete(_msgs(), profile="main"))

    msg = str(ei.value)
    assert "sk-supersecret" not in msg
    assert "sk-querysecret" not in msg

    # the request dump the client would log is also clean
    dump = client.request_dump("main", {"model": "m"})
    assert "sk-supersecret" not in json.dumps(dump)
    assert dump["headers"]["Authorization"] == "***"
    assert "token=" not in dump["url"]

    # the raw wire request still carried the real key (redaction is log-only)
    assert seen[0].headers["authorization"] == "Bearer sk-supersecret"


def test_status_error_scrubs_api_key_from_provider_body():
    """A provider error body that echoes the api_key is scrubbed from the
    raised error message."""
    cfg = LLMConfig(
        profiles={
            "main": LLMProfile(
                base_url="https://api.example.com/v1",
                api_key="sk-echoed-secret",
                model="m",
            )
        }
    )
    seen, transport = _recording_transport(
        lambda req: httpx.Response(
            401, json={"error": "invalid key sk-echoed-secret"}, request=req
        )
    )
    client = OpenAIClient(cfg, transport=transport)
    with pytest.raises(LLMPermanentError) as ei:
        run(client.complete(_msgs(), profile="main"))
    assert "sk-echoed-secret" not in str(ei.value)
    assert "***" in str(ei.value)
    assert len(seen) == 1


def test_request_dump_redacts_transcript_and_tool_arguments():
    """Provider diagnostics never dump the full transcript or tool
    arguments: messages become a count and tools keep only names."""
    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "secret chat text"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "reply",
                            "arguments": '{"text":"secret tool arg"}',
                        },
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "reply",
                    "description": "reply",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }
    dump = redact_request("https://api.example.com/v1/chat/completions", {}, body)
    dumped = json.dumps(dump)
    assert "secret chat text" not in dumped
    assert "secret tool arg" not in dumped
    assert "reply" in dumped  # tool NAME is fine to keep
    assert "<2 message(s) redacted>" in dumped
    assert dump["body"]["tools"][0]["function"] == {"name": "reply"}


def test_scrub_credentials_strips_query_from_nested_urls():
    """Arbitrary query parameter names/tokens and the configured api_key are
    scrubbed from nested exception text while host/path and surrounding
    diagnostics survive."""
    text = (
        "outer: https://api.example.com/v1/chat/completions?api_key=sk-q1 "
        "inner: https://api.example.com/v1?signature=abc&token=xyz "
        "key sk-supersecret"
    )
    out = scrub_credentials(text, api_key="sk-supersecret")
    assert "sk-q1" not in out
    assert "api_key=" not in out
    assert "signature=" not in out
    assert "token=" not in out
    assert "sk-supersecret" not in out
    assert "api.example.com/v1/chat/completions" in out
    assert "api.example.com/v1" in out
    assert "outer:" in out and "inner:" in out


def test_request_error_scrubs_query_secret_from_message():
    """A transport error whose message embeds the request URL with query
    credentials is scrubbed: the secret and query are gone, host/path stay,
    and the classification stays transient."""
    secret_url = "https://api.example.com/v1/chat/completions?token=sk-querysecret"

    def handler(request):
        raise httpx.ConnectError(
            "connect to ('api.example.com', 443) failed: "
            f"request URL {secret_url}",
            request=request,
        )

    seen, transport = _recording_transport(handler)
    client = OpenAIClient(_config(), transport=transport)
    with pytest.raises(LLMTransientError) as ei:
        run(client.complete(_msgs(), profile="main"))
    msg = str(ei.value)
    assert "sk-querysecret" not in msg
    assert "token=" not in msg
    assert "api.example.com/v1/chat/completions" in msg
    assert is_retryable(ei.value)
    assert len(seen) == 1


def test_status_error_scrubs_query_secret_from_provider_body():
    """A provider status body that echoes the request URL with query
    credentials (and the configured api_key) is scrubbed while
    host/path/status survive and the classification stays permanent."""
    cfg = LLMConfig(
        profiles={
            "main": LLMProfile(
                base_url="https://api.example.com/v1",
                api_key="sk-echoed-secret",
                model="m",
            )
        }
    )
    body = (
        '{"error": "invalid token at '
        "https://api.example.com/v1/chat/completions?token=sk-querysecret "
        'key sk-echoed-secret"}'
    )
    seen, transport = _recording_transport(
        lambda req: httpx.Response(401, text=body, request=req)
    )
    client = OpenAIClient(cfg, transport=transport)
    with pytest.raises(LLMPermanentError) as ei:
        run(client.complete(_msgs(), profile="main"))
    msg = str(ei.value)
    assert "sk-querysecret" not in msg
    assert "token=" not in msg
    assert "sk-echoed-secret" not in msg
    assert "api.example.com/v1/chat/completions" in msg
    assert "401" in msg
    assert not is_retryable(ei.value)
    assert len(seen) == 1


def test_status_error_query_secret_scrub_keeps_transient_class():
    """A 5xx body echoing a query secret stays transient (retryable) after
    scrubbing — no regression in typed classification."""
    body = (
        '{"error": "upstream at '
        'https://api.example.com/v1/chat/completions?token=sk-querysecret '
        'is down"}'
    )
    seen, transport = _recording_transport(
        lambda req: httpx.Response(503, text=body, request=req)
    )
    client = OpenAIClient(_config(), transport=transport)
    with pytest.raises(LLMTransientError) as ei:
        run(client.complete(_msgs(), profile="main"))
    msg = str(ei.value)
    assert "sk-querysecret" not in msg
    assert "token=" not in msg
    assert "503" in msg
    assert is_retryable(ei.value)
    assert len(seen) == 1


def test_status_error_keeps_ordinary_diagnostics():
    """A normal provider body (no secrets) keeps its useful text verbatim."""
    seen, transport = _recording_transport(
        lambda req: httpx.Response(
            429, text='{"error": "rate limit exceeded"}', request=req
        )
    )
    client = OpenAIClient(_config(), transport=transport)
    with pytest.raises(LLMTransientError) as ei:
        run(client.complete(_msgs(), profile="main"))
    msg = str(ei.value)
    assert "429" in msg
    assert "rate limit exceeded" in msg
    assert "api.example.com/v1/chat/completions" in msg


# ── retryable classification ─────────────────────────────────────────────────

def test_is_retryable_classification():
    assert is_retryable(LLMTransientError("x"))
    assert is_retryable(TransientError("x"))
    assert not is_retryable(LLMPermanentError("x"))
    assert not is_retryable(ValueError("x"))
