"""OpenAI-compatible provider client (Phase 3 capability lane).

The ONE concrete provider client behind the ``LLMClient``/``Embedder``
protocols in ``seams.py``. It owns every provider HTTP call: a long-lived
``httpx.AsyncClient`` with explicit idempotent ``aclose``, per-call profile
selection from ``LLMConfig``, auth headers, timeout/deadline bounding,
OpenAI-compatible ``/chat/completions`` and ``/embeddings`` JSON requests,
transcript wire serialization via ``context.serialize``, response
  tool-call/usage parsing, and redacted request dumps for logging. Every
  error it raises is scrubbed of URL query credentials and the configured
  api_key (``scrub_credentials``) so provider bodies and transport messages
  cannot leak secrets while host/path/status diagnostics survive.

Error contract — the only thing retry logic may rely on:

- ``LLMTransientError`` — retryable: network blips, timeouts, HTTP 429 and
  5xx, an already-expired deadline.
- ``LLMPermanentError`` — not retryable: unknown profile, malformed
  transcript/request, malformed provider payload, HTTP 4xx.

Both derive from the existing ``TransientError``/``PermanentError`` taxonomy
(and from ``LLMError``), so ``is_retryable`` classifies them unchanged. No
retries are implemented here — the deterministic classification IS the
contract; the caller owns retry policy.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from pretender.clock import RealClock
from pretender.config import LLMConfig, LLMProfile
from pretender.context import serialize
from pretender.errors import (
    ConfigError,
    LLMError,
    LLMPermanentError,
    LLMTransientError,
    TransientError,
)
from pretender.seams import Clock
from pretender.types import LLMResponse, ToolCall, ToolCallId, TranscriptMessage

__all__ = ["OpenAIClient", "is_retryable", "redact_request", "scrub_credentials"]

_CHAT_PATH = "/chat/completions"
_EMBEDDINGS_PATH = "/embeddings"

# Header names whose values are never logged (API keys ride here).
_SENSITIVE_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "api-key", "x-api-key", "x-auth-token"}
)

_MAX_DETAIL_CHARS = 300  # provider error-body excerpt bound for error messages


def is_retryable(exc: BaseException) -> bool:
    """Deterministic retryable classification — the LLM layer performs no
    retries itself; it only guarantees that every failure it raises is either
    ``TransientError`` (retryable) or ``PermanentError`` (not)."""
    return isinstance(exc, TransientError)


def redact_request(
    url: str,
    headers: dict[str, str] | None = None,
    body: Any = None,
    *,
    method: str = "POST",
) -> dict[str, Any]:
    """A log-safe request dump: sensitive header values (``Authorization``,
    ``x-api-key``, ...) are masked and the URL query/fragment are stripped —
    provider keys sometimes ride in query strings. The JSON body is NOT kept
    verbatim: the transcript ``messages`` and tool definitions/arguments are
    redacted so no chat data or tool arguments leak into logs."""
    redacted_headers = {
        name: ("***" if name.lower() in _SENSITIVE_HEADERS else value)
        for name, value in (headers or {}).items()
    }
    parts = urlsplit(url)
    safe_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    return {
        "method": method,
        "url": safe_url,
        "headers": redacted_headers,
        "body": _redact_body(body),
    }


def _redact_body(body: Any) -> Any:
    """Strip chat data and tool arguments from a request body for logging:
    ``messages`` becomes a count, ``tools`` keeps only names."""
    if not isinstance(body, dict):
        return body
    safe: dict[str, Any] = {}
    for key, value in body.items():
        if key == "messages":
            safe[key] = f"<{len(value)} message(s) redacted>"
        elif key == "tools":
            safe[key] = _redact_tools(value)
        else:
            safe[key] = value
    return safe


def _redact_tools(tools: Any) -> Any:
    if not isinstance(tools, list):
        return tools
    out: list[Any] = []
    for t in tools:
        if isinstance(t, dict):
            fn = t.get("function", {})
            out.append(
                {
                    "type": t.get("type"),
                    "function": {"name": fn.get("name") if isinstance(fn, dict) else None},
                }
            )
        else:
            out.append(t)
    return out


class OpenAIClient:
    """Long-lived OpenAI-compatible provider client (implements both the
    ``LLMClient`` and ``Embedder`` protocols).

    One instance serves every profile: the profile (``base_url``, ``model``,
    auth, timeouts) is selected per call. The underlying ``httpx.AsyncClient``
    is owned by this object and MUST be closed explicitly via ``aclose()``
    (idempotent, safe to call more than once); the async context manager is
    provided for convenience.

    Testability: ``transport`` (any ``httpx.AsyncBaseTransport``, e.g.
    ``httpx.MockTransport``) and ``client`` (an ``httpx.AsyncClient`` or a
    duck-typed stand-in) are injectable — the latter wins when both are
    given. ``clock`` may be a ``VirtualClock`` for deterministic deadline
    tests.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        clock: Clock | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
        embed_profile: str = "embed",
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._clock = clock or RealClock()
        self._embed_profile = embed_profile
        self._logger = logger or logging.getLogger("pretender.llm")
        self._closed = False
        if client is not None:
            self._client = client
        else:
            self._client = httpx.AsyncClient(
                transport=transport or httpx.AsyncHTTPTransport(),
                # Every request passes an explicit per-call timeout; the
                # client default only guards against a bug that forgets one.
                timeout=httpx.Timeout(30.0),
            )

    # ── LLMClient protocol ──────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[TranscriptMessage],
        *,
        profile: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        deadline: float | None = None,
    ) -> LLMResponse:
        prof = self._profile(profile)
        try:
            wire = serialize(messages)
        except ValueError as e:
            raise LLMPermanentError(
                f"cannot serialize transcript for {profile!r}: {e}"
            ) from e
        body = self._build_chat_body(
            prof, wire, tools=tools, temperature=temperature, max_tokens=max_tokens
        )
        url = _endpoint_url(prof, _CHAT_PATH)
        headers = _auth_headers(prof)
        timeout = self._request_timeout(prof, deadline)
        self._logger.debug(
            "chat completion request: %s", redact_request(url, headers, body)
        )
        response = await self._post(
            url, json=body, headers=headers, timeout=timeout, api_key=prof.api_key
        )
        try:
            payload = response.json()
        except ValueError as e:
            raise LLMPermanentError(
                f"provider returned a non-JSON body for {profile!r}: {e}"
            ) from e
        return _parse_chat_response(payload, profile=profile)

    # ── Embedder protocol ───────────────────────────────────────────────────

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prof = self._profile(self._embed_profile)
        url = _endpoint_url(prof, _EMBEDDINGS_PATH)
        headers = _auth_headers(prof)
        body: dict[str, Any] = {"model": prof.model, "input": texts}
        self._logger.debug(
            "embeddings request: %s", redact_request(url, headers, body)
        )
        response = await self._post(
            url,
            json=body,
            headers=headers,
            timeout=prof.timeout_s,
            api_key=prof.api_key,
        )
        try:
            payload = response.json()
        except ValueError as e:
            raise LLMPermanentError(
                f"provider returned a non-JSON body for "
                f"{self._embed_profile!r}: {e}"
            ) from e
        return _parse_embed_response(
            payload, n=len(texts), profile=self._embed_profile
        )

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        """Close the owned client. Idempotent: a second call is a no-op even
        when the injected client counts closes."""
        if self._closed:
            return
        self._closed = True
        aclose = getattr(self._client, "aclose", None)
        if aclose is not None:
            await aclose()

    async def __aenter__(self) -> "OpenAIClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ── request dump / diagnostics ──────────────────────────────────────────

    def request_dump(self, profile: str, body: dict[str, Any]) -> dict[str, Any]:
        """A log-safe dump of the request that WOULD be sent for ``profile``
        with ``body`` — the same redaction the client applies at request time
        (Authorization masked, URL query stripped). Never log the raw
        ``_auth_headers`` output."""
        prof = self._profile(profile)
        return redact_request(
            _endpoint_url(prof, _CHAT_PATH), _auth_headers(prof), body
        )

    # ── internals ───────────────────────────────────────────────────────────

    def _profile(self, name: str) -> LLMProfile:
        try:
            return self._config.profile(name)
        except ConfigError as e:
            raise LLMPermanentError(str(e)) from e

    def _request_timeout(self, prof: LLMProfile, deadline: float | None) -> float:
        if deadline is None:
            return prof.timeout_s
        remaining = deadline - self._clock.now()
        if remaining <= 0:
            raise LLMTransientError(
                f"deadline already passed: {deadline:.3f} <= now "
                f"{self._clock.now():.3f}"
            )
        return min(prof.timeout_s, remaining)

    def _build_chat_body(
        self,
        prof: LLMProfile,
        wire: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": prof.model,
            "messages": wire,
            "temperature": (
                temperature if temperature is not None else prof.temperature
            ),
            "max_tokens": max_tokens if max_tokens is not None else prof.max_tokens,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
        return body

    async def _post(
        self,
        url: str,
        *,
        json: Any,
        headers: dict[str, str],
        timeout: float,
        api_key: str | None = None,
    ) -> httpx.Response:
        try:
            response = await self._client.post(
                url, json=json, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            raise _status_error(e, url, api_key) from None
        except httpx.RequestError as e:
            # Transport errors embed the request URL (and sometimes the query
            # credentials) in their message: scrub before surfacing.
            detail = scrub_credentials(str(e), api_key=api_key)
            raise LLMTransientError(f"provider request failed: {detail}") from None


# ── wire helpers ─────────────────────────────────────────────────────────────


def _endpoint_url(prof: LLMProfile, path: str) -> str:
    return prof.base_url.rstrip("/") + path


def _auth_headers(prof: LLMProfile) -> dict[str, str]:
    if prof.api_key:
        return {"Authorization": f"Bearer {prof.api_key}"}
    return {}


def _redact_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


# Any http(s) URL up to a terminator: whitespace or a closing quote/angle/
# paren delimiter. Provider error bodies and transport exceptions often embed
# the request URL verbatim (repr style or plain), and the query may carry the
# credential under an arbitrary parameter name.
_URL_RE = re.compile(r"https?://[^\s'\"<>)]+")


def scrub_credentials(text: str, *, api_key: str | None = None) -> str:
    """Scrub credentials from arbitrary error/exception text.

    Defense-in-depth used by every provider error path (and by the doctor):
    every URL embedded in ``text`` is rewritten with its query and fragment
    stripped — provider keys sometimes ride in query strings, under arbitrary
    parameter names — and a configured ``api_key`` is masked when it appears
    verbatim. Host/path and surrounding diagnostics are preserved so the
    message stays actionable (only the secret-bearing parts are removed).
    """
    text = _scrub_urls(text)
    if api_key and api_key in text:
        text = text.replace(api_key, "***")
    return text


def _scrub_urls(text: str) -> str:
    if not text:
        return text

    def replace(match: "re.Match[str]") -> str:
        try:
            return _redact_url(match.group(0))
        except ValueError:  # malformed URL (bad IPv6/port): keep it intact
            return match.group(0)

    return _URL_RE.sub(replace, text)


def _excerpt(text: str, limit: int = _MAX_DETAIL_CHARS) -> str:
    text = (text or "").strip()
    if len(text) > limit:
        text = text[:limit] + "…"
    return text or "(empty body)"


def _status_error(
    e: httpx.HTTPStatusError, url: str, api_key: str | None = None
) -> LLMError:
    status = e.response.status_code
    # Scrub the FULL body before excerpting so a credential straddling the
    # truncation boundary cannot survive; the body may echo the request URL
    # (with query credentials) or the configured api_key.
    body = scrub_credentials(e.response.text, api_key=api_key)
    message = f"provider {status} from {_redact_url(url)}: {_excerpt(body)}"
    if status == 429 or status >= 500:
        return LLMTransientError(message)
    return LLMPermanentError(message)


# ── response parsing (fail closed) ───────────────────────────────────────────


def _parse_chat_response(payload: Any, *, profile: str) -> LLMResponse:
    if not isinstance(payload, dict):
        raise LLMPermanentError(
            f"malformed chat payload for {profile!r}: not an object"
        )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMPermanentError(
            f"malformed chat payload for {profile!r}: missing choices"
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise LLMPermanentError(
            f"malformed chat payload for {profile!r}: first choice is not an object"
        )
    message = choice.get("message")
    if not isinstance(message, dict):
        raise LLMPermanentError(
            f"malformed chat payload for {profile!r}: choice missing message"
        )
    content = _parse_content(message.get("content"), profile)
    tool_calls = _parse_tool_calls(message.get("tool_calls"), profile)
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str):
        finish_reason = None
    usage = payload.get("usage")
    usage = (
        {k: v for k, v in usage.items() if isinstance(v, int) and not isinstance(v, bool)}
        if isinstance(usage, dict)
        else {}
    )
    return LLMResponse(
        content=content,
        tool_calls=tuple(tool_calls),
        finish_reason=finish_reason,
        usage=usage,
        raw=payload,
    )


def _parse_content(raw: Any, profile: str) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):  # multimodal content parts: keep the text
        parts = [
            item["text"]
            for item in raw
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "".join(parts) if parts else None
    raise LLMPermanentError(
        f"malformed chat payload for {profile!r}: message content is not a string"
    )


def _parse_tool_calls(raw: Any, profile: str) -> list[ToolCall]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise LLMPermanentError(
            f"malformed chat payload for {profile!r}: tool_calls is not a list"
        )
    out: list[ToolCall] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise LLMPermanentError(
                f"malformed chat payload for {profile!r}: tool_call {i} "
                "is not an object"
            )
        call_id = item.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise LLMPermanentError(
                f"malformed chat payload for {profile!r}: tool_call {i} "
                "missing id"
            )
        fn = item.get("function")
        if not isinstance(fn, dict):
            raise LLMPermanentError(
                f"malformed chat payload for {profile!r}: tool_call {i} "
                "missing function"
            )
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            raise LLMPermanentError(
                f"malformed chat payload for {profile!r}: tool_call {i} "
                "missing function.name"
            )
        arguments, raw_arguments = _parse_arguments(fn.get("arguments"), profile, i)
        out.append(
            ToolCall(
                id=ToolCallId(call_id),
                name=name,
                arguments=arguments,
                raw_arguments=raw_arguments,
            )
        )
    ids = [c.id for c in out]
    if len(ids) != len(set(ids)):
        raise LLMPermanentError(
            f"malformed chat payload for {profile!r}: duplicate tool_call ids"
        )
    return out


def _parse_arguments(raw: Any, profile: str, i: int) -> tuple[dict[str, Any], Any]:
    """Parse a tool call's ``arguments`` into ``(dict, raw)``.

    A clean object (or a JSON string that decodes to one) yields
    ``(parsed, None)``. A malformed JSON string or a non-object value is NOT
    a permanent failure: it yields ``({}, raw)`` so the caller can carry the
    raw value into the tolerant one-repair / ``no_action`` path instead of
    aborting the whole completion.
    """
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}, raw  # malformed JSON string → tolerant path
        if not isinstance(parsed, dict):
            return {}, raw  # valid JSON but not an object → tolerant path
        return parsed, None
    if raw is None:
        return {}, None
    return {}, raw  # non-dict non-str (list, number) → tolerant path


def _parse_embed_response(payload: Any, *, n: int, profile: str) -> list[list[float]]:
    if not isinstance(payload, dict):
        raise LLMPermanentError(
            f"malformed embeddings payload for {profile!r}: not an object"
        )
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != n:
        raise LLMPermanentError(
            f"malformed embeddings payload for {profile!r}: expected {n} "
            f"vectors, got {len(data) if isinstance(data, list) else type(data).__name__}"
        )
    by_index: dict[int, list[float]] = {}
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "embedding" not in item:
            raise LLMPermanentError(
                f"malformed embeddings payload for {profile!r}: item {i} "
                "missing embedding"
            )
        vec = item["embedding"]
        if not isinstance(vec, list) or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in vec
        ):
            raise LLMPermanentError(
                f"malformed embeddings payload for {profile!r}: item {i} "
                "embedding is not a number list"
            )
        idx = item.get("index", i)
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
            raise LLMPermanentError(
                f"malformed embeddings payload for {profile!r}: item {i} "
                f"bad index {idx!r}"
            )
        if idx in by_index:
            raise LLMPermanentError(
                f"malformed embeddings payload for {profile!r}: duplicate "
                f"index {idx}"
            )
        by_index[idx] = [float(x) for x in vec]
    dims = {len(v) for v in by_index.values()}
    if len(dims) != 1:
        raise LLMPermanentError(
            f"malformed embeddings payload for {profile!r}: inconsistent "
            f"dimension {sorted(dims)}"
        )
    return [by_index[i] for i in sorted(by_index)]
