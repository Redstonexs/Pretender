"""Media normalization: download -> content-addressed cache -> Pillow
clamp/convert -> base64/data-URL.

Phase 4 lane: this module only fetches, caches, normalizes and encodes
images. It does NOT invoke a vision LLM — a later lane feeds the encoded
payload to the vision profile. Everything here is adapter-agnostic and
testable with an injected fetcher/cache (no network required in tests).

Failure taxonomy (safe size/type behavior):
- ``MediaFetchError`` (Transient): the download itself failed — network
  blip, timeout, HTTP error. Retrying may succeed.
- ``MediaUnsafeError`` (Permanent): the URL or a redirect target is an
  unsafe http(s) host — a localhost/private/link-local literal, a non-http(s)
  scheme, or a DNS hostname that resolves to ANY non-global address (DNS
  rebinding guard). A host that cannot be resolved or validated is also
  rejected (fail closed — there is no safe address to pin). Retrying cannot
  succeed without a change of input.
- ``MediaTooLarge`` (Permanent): the payload exceeds ``max_bytes`` (checked
  both from Content-Length and while streaming).
- ``MediaTypeError`` (Permanent): the bytes are not a supported image type,
  are corrupt/unparseable, or exceed the pre-decode pixel cap. Retrying
  cannot succeed without a change of input.

DNS safety: a DNS hostname is resolved and EVERY resolved address must be a
public IP; the connection is then PINNED to the validated address (via
:class:`_PinnedAsyncTransport`) so it cannot be rebound to a private target
between validation and connect. Redirects obey the same policy on every hop.
Normal HTTPS verification is NOT weakened — the certificate is still checked
against the original hostname.

The content-addressed key is the SHA-256 of the NORMALIZED bytes, so two
URLs serving identical images share one cache entry; the URL->key index
avoids re-downloading a URL we have already normalized.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import ipaddress
import json
import os
import socket
import ssl
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
from PIL import Image

from pretender.clock import RealClock
from pretender.config import MediaConfig
from pretender.context import render_image_markdown
from pretender.emoji import VisionResult, parse_vision_result, validate_catalog_key
from pretender.errors import AdapterNotReady, PermanentError, TransientError
from pretender.log import get_logger
from pretender.seams import MediaRepository
from pretender.types import (
    AdapterEvent,
    ChatKey,
    MediaAssetCandidate,
    MediaSafetyStatus,
    Message,
    MessageId,
    MessageRowId,
    Outgoing,
    Segment,
    TranscriptMessage,
)

#: Supported source image types (by Pillow format name) -> output MIME.
_MIME_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
    "BMP": "image/bmp",
}

#: MIME types we accept as image input.
_MIME_ALLOWLIST = frozenset(_MIME_BY_FORMAT.values())

#: Hostname suffixes that are never safe to fetch from (SSRF guard).
_PRIVATE_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa")

#: Redirect status codes the fetcher follows (with per-hop validation).
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

# The pinned HTTP/1.1 transport reads response framing itself. Bound both a
# single header line and the aggregate header block, and apply one deadline to
# DNS/connect/header/body work so a peer cannot keep a request alive forever by
# dribbling valid-looking header lines.
_MAX_HEADER_BYTES = 64 * 1024
_MAX_HEADER_LINES = 100


class MediaError(PermanentError):
    """Media could not be fetched/normalized (base class)."""


class MediaTooLarge(MediaError):
    """The payload exceeds the configured size limit."""


class MediaTypeError(MediaError):
    """The bytes are not a supported/correct image."""


class MediaUnsafeError(MediaError):
    """The URL or a redirect target is an unsafe http(s) host (SSRF guard)."""


class MediaFetchError(TransientError):
    """The download failed (network/timeout/HTTP) — retrying may succeed."""


def is_safe_media_url(url: str) -> bool:
    """True when ``url`` is an http(s) URL whose host is not a localhost/
    private/link-local literal (SSRF guard).

    Rejects non-http(s) schemes, missing hosts, ``localhost`` and private
    hostname suffixes, and IP literals that are loopback/private/link-local/
    reserved/unspecified. A DNS name that is not an IP literal is allowed
    (its resolution is not checked here — the guard is literal-based).
    """
    try:
        parts = urlparse(url)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    host = parts.hostname
    if not host:
        return False
    host = host.rstrip(".").lower()
    if host == "localhost" or host.endswith(_PRIVATE_HOST_SUFFIXES):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True  # a DNS name, not an IP literal
    return ip.is_global and not ip.is_multicast


async def _resolve_host(host: str) -> list[str]:
    """Resolve ``host`` to its IP addresses (async, non-blocking)."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


async def resolve_safe_host(host: str, resolver=None) -> str:
    """Resolve ``host`` and return a pinned public IP, or raise
    ``MediaUnsafeError``.

    DNS-rebinding guard: EVERY resolved address must be a global (public) IP.
    If any address is loopback/private/link-local/reserved/unspecified the
    host is rejected (fail closed) — a hostname that can resolve to a private
    target is never fetched. A host that cannot be resolved or validated is
    also rejected (fail closed): there is no safe address to pin.

    ``resolver`` is injectable for tests (default: real DNS via
    :func:`_resolve_host`). Returns the first validated public address, which
    the caller pins so the connection cannot be rebound to a private target
    between validation and connect (TOCTOU).
    """
    host = host.strip().lower().rstrip(".")
    if not host:
        raise MediaUnsafeError("empty media host")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not ip.is_global or ip.is_multicast:
            raise MediaUnsafeError(f"unsafe media host {host!r}")
        return str(ip)
    try:
        addrs = await (resolver or _resolve_host)(host)
    except OSError as e:
        raise MediaUnsafeError(
            f"cannot resolve media host {host!r}: {e}"
        ) from e
    if not addrs:
        raise MediaUnsafeError(f"media host {host!r} resolved to no addresses")
    pinned: str | None = None
    for addr in addrs:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            raise MediaUnsafeError(
                f"media host {host!r} resolved to non-IP {addr!r}"
            ) from None
        if not ip.is_global or ip.is_multicast:
            raise MediaUnsafeError(
                f"media host {host!r} resolves to non-global address {addr}"
            )
        if pinned is None:
            pinned = str(ip)
    assert pinned is not None
    return pinned


class _PinnedAsyncTransport(httpx.AsyncBaseTransport):
    """An HTTP/1.1 transport that resolves+validates the hostname and pins the
    connection to the validated public IP, preserving the original ``Host``
    header and TLS verification (SNI + certificate) for the hostname.

    This is the "comparably robust safe transport" for DNS rebinding: the
    connection goes to the validated address, never to whatever the hostname
    happens to resolve to at connect time. HTTPS verification is NOT weakened
    — ``server_hostname`` is the original hostname, so the certificate is
    still checked against it. ``Connection: close`` per request keeps the
    implementation simple and correct.
    """

    def __init__(
        self,
        timeout_s: float,
        max_bytes: int = 15 * 1024 * 1024,
        resolver=None,
    ) -> None:
        self._timeout_s = timeout_s
        self._max_bytes = max_bytes
        self._resolver = resolver or resolve_safe_host

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        host = url.host
        port = url.port or (443 if url.scheme == "https" else 80)
        ssl_ctx = None
        server_hostname = None
        if url.scheme == "https":
            ssl_ctx = ssl.create_default_context()
            server_hostname = host
        writer = None
        try:
            async with asyncio.timeout(self._timeout_s):
                pinned = await self._resolver(host)
                reader, writer = await asyncio.open_connection(
                    pinned,
                    port,
                    ssl=ssl_ctx,
                    server_hostname=server_hostname,
                    limit=_MAX_HEADER_BYTES,
                )
                path = url.raw_path.decode("latin-1") or "/"
                host_header = host if port in (80, 443) else f"{host}:{port}"
                lines = [f"{request.method} {path} HTTP/1.1", f"Host: {host_header}"]
                for name, value in request.headers.items():
                    if name.lower() != "host":
                        lines.append(f"{name}: {value}")
                body = request.content
                if body:
                    lines.append(f"Content-Length: {len(body)}")
                lines.append("Connection: close")
                writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body)
                await writer.drain()
                status_line = await reader.readline()
                header_bytes = len(status_line)
                if not status_line or header_bytes > _MAX_HEADER_BYTES:
                    raise MediaFetchError("media response status line is too large")
                parts = status_line.decode("latin-1").strip().split(" ", 2)
                if len(parts) < 2:
                    raise MediaFetchError("malformed media HTTP status line")
                status_code = int(parts[1])
                headers: list[tuple[str, str]] = []
                for _ in range(_MAX_HEADER_LINES):
                    line = await reader.readline()
                    header_bytes += len(line)
                    if header_bytes > _MAX_HEADER_BYTES:
                        raise MediaFetchError("media response headers are too large")
                    if line in (b"\r\n", b"\n", b""):
                        break
                    name, sep, value = line.decode("latin-1").partition(":")
                    if not sep:
                        raise MediaFetchError("malformed media response header")
                    headers.append((name.strip(), value.strip()))
                else:
                    raise MediaFetchError("media response has too many headers")
                content = await self._read_body(reader, headers)
                return httpx.Response(
                    status_code, headers=headers, content=content, request=request
                )
        except (ValueError, asyncio.LimitOverrunError) as exc:
            raise MediaFetchError(f"malformed media response: {exc}") from None
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

    async def _read_body(self, reader, headers: list[tuple[str, str]]) -> bytes:
        hdrs = {k.lower(): v for k, v in headers}
        declared = None
        if hdrs.get("content-length") is not None:
            try:
                declared = int(hdrs["content-length"])
            except ValueError:
                pass
        if declared is not None and declared > self._max_bytes:
            raise MediaTooLarge(
                f"media too large: {declared} bytes > {self._max_bytes}"
            )
        if hdrs.get("transfer-encoding", "").lower() == "chunked":
            chunks: list[bytes] = []
            total = 0
            while True:
                size_line = await asyncio.wait_for(
                    reader.readline(), timeout=self._timeout_s
                )
                size = int(size_line.split(b";")[0].strip(), 16)
                if size == 0:
                    await asyncio.wait_for(
                        reader.readline(), timeout=self._timeout_s
                    )  # trailing CRLF
                    break
                total += size
                if total > self._max_bytes:
                    raise MediaTooLarge(
                        f"media too large: > {self._max_bytes} bytes"
                    )
                chunks.append(
                    await asyncio.wait_for(
                        reader.readexactly(size), timeout=self._timeout_s
                    )
                )
                await asyncio.wait_for(
                    reader.readline(), timeout=self._timeout_s
                )  # CRLF after chunk
            return b"".join(chunks)
        if declared is not None:
            return await asyncio.wait_for(
                reader.readexactly(declared), timeout=self._timeout_s
            )
        # No length: stream until EOF while enforcing the same hard cap.
        chunks = []
        total = 0
        while True:
            chunk = await asyncio.wait_for(reader.read(64 * 1024), timeout=self._timeout_s)
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if total > self._max_bytes:
                raise MediaTooLarge(f"media too large: > {self._max_bytes} bytes")
            chunks.append(chunk)


@dataclass(frozen=True)
class MediaAsset:
    """A normalized image: clamped, converted, content-addressed.

    ``key`` is the SHA-256 of ``data`` (the normalized bytes) — the cache
    identity. ``source_sha256`` is the SHA-256 of the ORIGINAL fetched bytes
    (before normalization), useful for dedupe/audit. ``mime``/``format``/
    ``width``/``height`` describe the normalized output (always JPEG after
    clamp/convert).
    """

    key: str
    data: bytes
    mime: str
    format: str
    width: int
    height: int
    source_url: str
    source_sha256: str


class MediaCache(Protocol):
    """Content-addressed cache. Implementations may be in-memory or on disk."""

    def get(self, key: str) -> MediaAsset | None: ...
    def put(self, asset: MediaAsset) -> None: ...
    def get_url(self, url: str) -> str | None: ...
    def put_url(self, url: str, key: str) -> None: ...


class InMemoryMediaCache:
    """A process-local cache (tests and default when no cache_dir given)."""

    def __init__(self) -> None:
        self._assets: dict[str, MediaAsset] = {}
        self._urls: dict[str, str] = {}

    def get(self, key: str) -> MediaAsset | None:
        return self._assets.get(key)

    def put(self, asset: MediaAsset) -> None:
        self._assets[asset.key] = asset

    def get_url(self, url: str) -> str | None:
        return self._urls.get(url)

    def put_url(self, url: str, key: str) -> None:
        self._urls[url] = key


class DiskMediaCache:
    """A directory-backed cache: one ``<key>.json`` per asset (data base64)
    plus a ``urls.json`` URL->key index. Writes are atomic (temp + rename)."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._urls_path = self._dir / "urls.json"
        self._urls: dict[str, str] = self._load_urls()

    def _load_urls(self) -> dict[str, str]:
        try:
            raw = self._urls_path.read_text(encoding="utf-8")
        except OSError:
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (ValueError, TypeError):
            return {}

    def _save_urls(self) -> None:
        tmp = self._urls_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._urls), encoding="utf-8")
        os.replace(tmp, self._urls_path)

    def get(self, key: str) -> MediaAsset | None:
        path = self._dir / f"{key}.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError:
            return None
        except ValueError:
            return None
        try:
            return MediaAsset(
                key=raw["key"],
                data=base64.b64decode(raw["data_b64"]),
                mime=raw["mime"],
                format=raw["format"],
                width=int(raw["width"]),
                height=int(raw["height"]),
                source_url=raw["source_url"],
                source_sha256=raw["source_sha256"],
            )
        except (KeyError, TypeError, ValueError):
            return None

    def put(self, asset: MediaAsset) -> None:
        payload = {
            "key": asset.key,
            "data_b64": base64.b64encode(asset.data).decode("ascii"),
            "mime": asset.mime,
            "format": asset.format,
            "width": asset.width,
            "height": asset.height,
            "source_url": asset.source_url,
            "source_sha256": asset.source_sha256,
        }
        path = self._dir / f"{asset.key}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)

    def get_url(self, url: str) -> str | None:
        return self._urls.get(url)

    def put_url(self, url: str, key: str) -> None:
        if self._urls.get(url) == key:
            return
        self._urls[url] = key
        self._save_urls()


class MediaFetcher(Protocol):
    """Fetches a URL's raw bytes. Implementations may be network or injected."""

    async def fetch(self, url: str) -> bytes: ...


class HttpxMediaFetcher:
    """Default fetcher: an httpx GET with validated redirects, a bounded
    timeout, and a hard size cap enforced both from Content-Length and while
    streaming.

    SSRF guard: every hop (the initial URL and each redirect target) is
    validated by :func:`is_safe_media_url` before the request is made — an
    unsafe host (localhost/private/link-local literal) is a
    ``MediaUnsafeError``. Redirects are followed manually so each target is
    checked; a malformed ``Content-Length`` is ignored (the streaming cap
    still applies).
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 10.0,
        max_bytes: int = 15 * 1024 * 1024,
        follow_redirects: bool = True,
        max_redirects: int = 5,
        resolver=None,
    ) -> None:
        self._client = client
        self._timeout_s = timeout_s
        self._max_bytes = max_bytes
        self._follow_redirects = follow_redirects
        self._max_redirects = max_redirects
        self._owns_client = client is None
        self._resolver = resolver or resolve_safe_host
        self._resolver_explicit = resolver is not None

    @staticmethod
    def _safe_content_length(value: str | None) -> int | None:
        """Parse ``Content-Length`` safely: a malformed value is ignored
        (the streaming cap still bounds the payload)."""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    async def _validate_host(self, url: str) -> None:
        """Resolve+validate a DNS hostname (DNS-rebinding guard). IP literals
        are already validated by :func:`is_safe_media_url`; a DNS name is
        resolved and every address must be public, else ``MediaUnsafeError``.

        With an injected client and no explicit resolver the caller owns
        transport safety (tests use mock transports with label hostnames that
        need not resolve); the production path (own client + pinned transport)
        and any explicit resolver always validate."""
        parts = urlparse(url)
        host = parts.hostname
        if not host:
            return
        host = host.rstrip(".").lower()
        try:
            ipaddress.ip_address(host)
            return  # literal: is_safe_media_url already validated it
        except ValueError:
            pass
        if self._client is not None and not self._resolver_explicit:
            return
        await self._resolver(host)  # raises MediaUnsafeError if unsafe

    async def fetch(self, url: str) -> bytes:
        client = self._client
        if client is None:
            client = httpx.AsyncClient(
                timeout=self._timeout_s,
                follow_redirects=False,
                transport=_PinnedAsyncTransport(
                    self._timeout_s, self._max_bytes, self._resolver
                ),
            )
        try:
            current = url
            hops = 0
            while True:
                # The complete redirect hop is bounded, including DNS
                # validation. A stalled resolver must not occupy a background
                # media slot indefinitely before the pinned transport starts.
                async with asyncio.timeout(self._timeout_s):
                    if not is_safe_media_url(current):
                        raise MediaUnsafeError(f"unsafe media URL: {current!r}")
                    # DNS-rebinding guard: resolve+validate DNS hosts on every
                    # hop (the pinned transport also pins the connection to the
                    # validated address; this check covers injected clients).
                    await self._validate_host(current)
                    async with client.stream("GET", current) as resp:
                        if resp.status_code in _REDIRECT_STATUSES:
                            if not self._follow_redirects:
                                raise MediaFetchError(
                                    f"media redirect not followed for {current!r}"
                                )
                            hops += 1
                            if hops > self._max_redirects:
                                raise MediaFetchError(
                                    f"media too many redirects for {url!r}"
                                )
                            location = resp.headers.get("location")
                            if not location:
                                raise MediaFetchError(
                                    f"media redirect without Location for {current!r}"
                                )
                            current = str(resp.url.join(location))
                            continue
                        if resp.status_code >= 400:
                            raise MediaFetchError(
                                f"media download failed: HTTP {resp.status_code} for {current!r}"
                            )
                        length = self._safe_content_length(
                            resp.headers.get("content-length")
                        )
                        if length is not None and length > self._max_bytes:
                            raise MediaTooLarge(
                                f"media too large: {length} bytes > {self._max_bytes}"
                            )
                        chunks: list[bytes] = []
                        total = 0
                        async for chunk in resp.aiter_bytes():
                            total += len(chunk)
                            if total > self._max_bytes:
                                raise MediaTooLarge(
                                    f"media too large: > {self._max_bytes} bytes"
                                )
                            chunks.append(chunk)
                        return b"".join(chunks)
        except MediaError:
            raise
        except (httpx.HTTPError, OSError, TimeoutError) as e:
            raise MediaFetchError(f"media download failed for {url!r}: {e}") from e
        finally:
            if self._owns_client and client is not None:
                await client.aclose()


class MediaStore:
    """The normalization entry point: ``get(url)`` returns a cached or freshly
    downloaded+normalized :class:`MediaAsset`.

    ``fetcher``/``cache`` are injectable for tests; when neither a cache nor a
    ``cache_dir`` is given, an in-memory cache is used. ``max_dim`` clamps the
    longest side; ``quality`` is the JPEG re-encode quality; ``max_pixels`` is
    the explicit pre-decode pixel cap (width × height) enforced from the image
    header BEFORE Pillow decodes the payload, so a decompression bomb is
    rejected without allocating its pixels.
    """

    def __init__(
        self,
        *,
        fetcher: MediaFetcher | None = None,
        cache: MediaCache | None = None,
        cache_dir: str | Path | None = None,
        max_bytes: int = 15 * 1024 * 1024,
        max_dim: int = 1280,
        quality: int = 85,
        max_pixels: int = 25_000_000,
        mime_allowlist: frozenset[str] = _MIME_ALLOWLIST,
    ) -> None:
        self._fetcher = fetcher if fetcher is not None else HttpxMediaFetcher(max_bytes=max_bytes)
        if cache is None:
            cache = DiskMediaCache(cache_dir) if cache_dir is not None else InMemoryMediaCache()
        self._cache = cache
        self._max_bytes = max_bytes
        self._max_dim = max_dim
        self._quality = quality
        self._max_pixels = max_pixels
        self._mime_allowlist = mime_allowlist

    # ── public API ──────────────────────────────────────────────────────────

    async def get(self, url: str) -> MediaAsset:
        """Return the normalized asset for ``url`` (cache hit or download).

        Raises ``MediaFetchError`` (transient), ``MediaUnsafeError`` /
        ``MediaTooLarge`` / ``MediaTypeError`` (permanent) on failure.
        """
        if not is_safe_media_url(url):
            raise MediaUnsafeError(f"unsafe media URL: {url!r}")
        key = self._cache.get_url(url)
        if key is not None:
            asset = self._cache.get(key)
            if asset is not None:
                return asset
        raw = await self._fetch(url)
        asset = self._normalize(raw, url)
        self._cache.put_url(url, asset.key)
        self._cache.put(asset)
        return asset

    def cached(self, url: str) -> MediaAsset | None:
        """Synchronous cache-only lookup — never fetches, never blocks the
        event loop. Returns None for unsafe URLs and cache misses, so the
        frame/event loop can attach media only when it is already cached."""
        if not is_safe_media_url(url):
            return None
        key = self._cache.get_url(url)
        if key is None:
            return None
        return self._cache.get(key)

    def asset_by_key(self, key: str) -> MediaAsset | None:
        """Synchronous cache-only lookup by OPAQUE content-addressed key.

        Never fetches and never blocks the event loop. The key is the sha256
        hex digest of the normalized bytes (the catalog's opaque cache key);
        a key that is not in the cache returns None. This is the ONLY
        cache-key → bytes surface the send-time resolver uses — the durable
        outbox never carries the bytes or a URL.
        """
        if not isinstance(key, str) or not key:
            return None
        return self._cache.get(key)

    async def prefetch(self, url: str) -> MediaAsset | None:
        """Fetch + normalize + cache in the background. Safe failure: returns
        None on any error (the caller keeps the original segment)."""
        try:
            return await self.get(url)
        except Exception:
            return None

    async def _fetch(self, url: str) -> bytes:
        data = await self._fetcher.fetch(url)
        if len(data) > self._max_bytes:
            raise MediaTooLarge(
                f"media too large: {len(data)} bytes > {self._max_bytes}"
            )
        return data

    # ── normalization ───────────────────────────────────────────────────────

    def _normalize(self, data: bytes, url: str) -> MediaAsset:
        source_sha256 = hashlib.sha256(data).hexdigest()
        try:
            img = Image.open(io.BytesIO(data))
            width, height = img.size
            if width * height > self._max_pixels:
                # Pre-decode pixel cap: reject a decompression bomb from the
                # header BEFORE Pillow decodes (and allocates) the pixels.
                raise MediaTypeError(
                    f"image too large: {width}x{height} pixels > {self._max_pixels}"
                )
            img.load()  # force decode so corrupt data raises here
            fmt = (img.format or "JPEG").upper()
            mime = _MIME_BY_FORMAT.get(fmt, "image/jpeg")
            if mime not in self._mime_allowlist:
                raise MediaTypeError(
                    f"unsupported media type {fmt!r} for {url!r}"
                )
            clamped = self._clamp(img)
            out = io.BytesIO()
            clamped.save(out, format="JPEG", quality=self._quality)
            out_bytes = out.getvalue()
        except MediaError:
            raise
        except Exception as e:  # Pillow decode/encode failure
            raise MediaTypeError(f"cannot normalize image {url!r}: {e}") from e
        key = hashlib.sha256(out_bytes).hexdigest()
        return MediaAsset(
            key=key,
            data=out_bytes,
            mime="image/jpeg",
            format="JPEG",
            width=clamped.width,
            height=clamped.height,
            source_url=url,
            source_sha256=source_sha256,
        )

    def _clamp(self, img: Image.Image) -> Image.Image:
        """Clamp the longest side to ``max_dim`` and convert to RGB (flattening
        alpha onto white so the JPEG re-encode is lossless of appearance)."""
        if img.width > self._max_dim or img.height > self._max_dim:
            img = img.copy()
            img.thumbnail((self._max_dim, self._max_dim))
        if img.mode in ("RGBA", "LA", "PA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.getchannel("A"))
            return bg
        return img.convert("RGB")

    # ── encoding helpers ────────────────────────────────────────────────────

    @staticmethod
    def to_data_url(data: bytes, mime: str) -> str:
        """``data:<mime>;base64,...`` — the payload a vision LLM consumes."""
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    def asset_data_url(self, asset: MediaAsset) -> str:
        return self.to_data_url(asset.data, asset.mime)

    def segment_data(self, asset: MediaAsset) -> dict[str, Any]:
        """A plain-dict representation to attach to a ``Segment.data["media"]``
        (JSON-serializable, deterministic per content)."""
        return {
            "key": asset.key,
            "mime": asset.mime,
            "format": asset.format,
            "width": asset.width,
            "height": asset.height,
            "source_url": asset.source_url,
            "data_url": self.asset_data_url(asset),
        }


def media_segment_data(asset: MediaAsset) -> dict[str, Any]:
    """A plain-dict representation to attach to a ``Segment.data["media"]``
    (JSON-serializable, deterministic per content)."""
    return {
        "key": asset.key,
        "mime": asset.mime,
        "format": asset.format,
        "width": asset.width,
        "height": asset.height,
        "source_url": asset.source_url,
        "data_url": MediaStore.to_data_url(asset.data, asset.mime),
    }


# ── Phase 6 P6.5b media runtime ──────────────────────────────────────────────
# The bounded/cancellable/advisory harvest lane and the send-time cache-key
# resolver. Both are LIVE-only, opt-in, and never run inside a terminal
# settlement transaction.

_log = get_logger("media")


def _chat_kind(chat_key: ChatKey) -> str | None:
    """The chat kind of a ``platform:kind:id`` chat key, or None."""
    parts = str(chat_key).split(":")
    if len(parts) >= 3 and parts[1] in ("group", "private"):
        return parts[1]
    return None


def _media_url(msg: Message) -> str | None:
    """The first sticker/image URL of a message's segments, or None."""
    for seg in msg.segments:
        if seg.kind not in ("sticker", "image"):
            continue
        url = seg.data.get("url") or seg.data.get("file")
        if isinstance(url, str) and url:
            return url
    return None


def _media_kind(msg: Message) -> str | None:
    """The catalog kind of a message's first media segment, or None."""
    for seg in msg.segments:
        if seg.kind == "sticker":
            return "sticker"
        if seg.kind == "image":
            return "image"
    return None


class MediaHarvester:
    """Bounded, cancellable, advisory group-sticker harvesting (P6.5b).

    Fired AFTER a durable ingest insertion (the App wires it as Ingest's
    post-insert ``harvest_media`` callback). Every harvest is a background
    task bounded by a semaphore and a per-harvest timeout; failures are
    contained and logged and NEVER change the ingest result or any source
    truth. The catalog key is the OPAQUE content-addressed cache key the
    MediaStore produced — the original URL never enters the catalog.

    Policy (config-owned switches): ``media.enabled`` + ``media.harvest``
    gate harvesting at all; group stickers are harvested for non-self
    messages (``group_nonself_stickers_only``); private stickers/images are
    harvested only when ``private_stickers_enabled`` / ``private_images_enabled``
    are set; group images have no switch and are never harvested.

    Vision (approval) is budget-admitted through the injected background
    learner budget (``LearnerBudget``) when a ``vision_profile`` is
    configured: a missing profile, a blocked budget, a provider failure, or
    a malformed/unsafe/empty structured verdict leaves the candidate PENDING
    (unapproved) — the asset is never approved without an explicit
    ``safe: true`` classification and a valid bounded escaped one-line
    description, and no source truth changes. ``candidate_cap`` is enforced
    BEFORE submit (a chat at/over its pending-candidate bound is skipped),
    and the tracked task set is bounded by ``max_tasks`` so a burst of media
    messages can never grow the background task set without limit.
    """

    def __init__(
        self,
        repo: MediaRepository,
        store: MediaStore,
        *,
        cfg: MediaConfig,
        clock: Any = None,
        llm: Any = None,
        budget: Any = None,
        max_concurrency: int = 2,
        timeout_s: float = 30.0,
        max_tasks: int = 64,
    ) -> None:
        self._repo = repo
        self._store = store
        self._cfg = cfg
        self._clock = clock if clock is not None else RealClock()
        self._llm = llm
        self._budget = budget
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
            raise ValueError("max_concurrency must be a positive integer")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        self._sem = asyncio.Semaphore(max_concurrency)
        self._timeout_s = timeout_s
        if isinstance(max_tasks, bool) or not isinstance(max_tasks, int):
            raise ValueError("max_tasks must be a positive integer")
        if max_tasks < 1:
            raise ValueError("max_tasks must be a positive integer")
        self._max_tasks = max_tasks
        self._tasks: set[asyncio.Task[None]] = set()
        # Admission is a check-then-fetch-then-submit flow.  Reservations
        # close the race where several harvest tasks observe the same pending
        # count before any of them submits.
        self._candidate_lock = asyncio.Lock()
        self._candidate_reservations: dict[ChatKey, int] = {}

    # ── public API ──────────────────────────────────────────────────────────

    def maybe_harvest(self, msg: Message, row_id: MessageRowId | None) -> asyncio.Task | None:
        """Schedule a bounded background harvest for a newly inserted
        message, or return None when the policy/kind excludes it or the
        tracked task set is at its ``max_tasks`` bound. The task is tracked
        for cancellation on shutdown; a finished task is discarded."""
        if not self._policy_allows(msg):
            return None
        if len(self._tasks) >= self._max_tasks:
            _log.warning(
                "media harvest task bound reached (%s); skipping %s",
                self._max_tasks,
                msg.chat_key,
            )
            return None
        task = asyncio.create_task(self._harvest(msg, row_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def cancel(self) -> None:
        """Cancel every in-flight harvest task (shutdown). Idempotent."""
        for task in list(self._tasks):
            task.cancel()

    @property
    def active_count(self) -> int:
        return len(self._tasks)

    # ── policy ──────────────────────────────────────────────────────────────

    def _policy_allows(self, msg: Message) -> bool:
        if not self._cfg.enabled or not self._cfg.harvest:
            return False
        kind = _chat_kind(msg.chat_key)
        seg_kind = _media_kind(msg)
        if kind is None or seg_kind is None:
            return False
        if kind == "group":
            if seg_kind != "sticker":
                return False  # group images have no enable switch
            if self._cfg.group_nonself_stickers_only and msg.is_self:
                return False
            return True
        # private chat
        if seg_kind == "sticker":
            return self._cfg.private_stickers_enabled
        return self._cfg.private_images_enabled

    # ── the bounded harvest task ────────────────────────────────────────────

    async def _harvest(self, msg: Message, row_id: MessageRowId | None) -> None:
        async with self._sem:
            reserved = False
            try:
                async with asyncio.timeout(self._timeout_s):
                    url = _media_url(msg)
                    if url is None:
                        return
                    # Reserve before fetching.  The reservation stays held
                    # through submit/approval, so concurrent tasks cannot all
                    # pass the pending-count check.
                    if not await self._reserve_candidate_slot(msg.chat_key):
                        return
                    reserved = True
                    asset = await self._store.get(url)
                    vision = await self._describe(msg.chat_key, asset)
                    # Recheck immediately before catalog mutation.  The
                    # repository's approval transaction repeats this fence,
                    # closing the recall/approval race between this read and
                    # the eventual approval.
                    if not await self._source_is_live(msg.chat_key, row_id):
                        return
                    candidate = self._candidate(msg, row_id, asset, vision.description)
                    cid = await self._repo.submit_media_candidate(
                        candidate, now=self._clock.now()
                    )
                    # Strict approval: ONLY an explicit safe=true structured
                    # classification with a valid bounded escaped one-line
                    # description approves. Missing/failed/malformed/unsafe
                    # verdicts leave the candidate PENDING (unapproved).
                    if vision.safe and vision.description is not None:
                        if not await self._source_is_live(msg.chat_key, row_id):
                            await self._repo.reject_media_candidate(msg.chat_key, cid)
                            return
                        await self._repo.approve_media_candidate(
                            msg.chat_key,
                            cid,
                            capacity=self._cfg.capacity,
                            now=self._clock.now(),
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                _log.warning(
                    "media harvest failed for %s (contained)",
                    msg.chat_key,
                    exc_info=True,
                )
            finally:
                if reserved:
                    try:
                        await asyncio.shield(
                            self._release_candidate_slot(msg.chat_key)
                        )
                    except asyncio.CancelledError:
                        # The shielded cleanup continues and removes the
                        # reservation even when shutdown cancels this task.
                        pass

    async def _source_is_live(
        self, chat_key: ChatKey, row_id: MessageRowId | None
    ) -> bool:
        """Return false for a durably recalled/deleted source.

        Test seams from the earlier harvest lane may not expose a message
        lookup; those seams retain the advisory behavior while the concrete
        repository supplies the race-safe transactional approval fence.
        """
        if row_id is None:
            return True
        get_by_platform = getattr(self._repo, "get_message_by_row_id", None)
        if get_by_platform is not None:
            try:
                msg = await get_by_platform(chat_key, row_id)
                return msg is not None
            except Exception:
                return False
        # The base repository seam exposes get_message by platform id, not
        # local row id.  Concrete repos additionally expose this tiny check.
        is_deleted = getattr(self._repo, "is_message_deleted", None)
        if is_deleted is None:
            return True
        try:
            return not await is_deleted(chat_key, row_id)
        except Exception:
            return False

    async def _reserve_candidate_slot(self, chat_key: ChatKey) -> bool:
        """Reserve a pending-candidate slot before any network fetch.

        A repository failure is fail closed: an unknown count can never
        authorize a submit.
        """
        try:
            async with self._candidate_lock:
                pending = await self._repo.list_media_candidates(
                    chat_key, limit=self._cfg.candidate_cap
                )
                reserved = self._candidate_reservations.get(chat_key, 0)
                if len(pending) + reserved >= self._cfg.candidate_cap:
                    return False
                self._candidate_reservations[chat_key] = reserved + 1
                return True
        except Exception:
            return False

    async def _release_candidate_slot(self, chat_key: ChatKey) -> None:
        async with self._candidate_lock:
            reserved = self._candidate_reservations.get(chat_key, 0)
            if reserved <= 1:
                self._candidate_reservations.pop(chat_key, None)
            else:
                self._candidate_reservations[chat_key] = reserved - 1

    def _candidate(
        self,
        msg: Message,
        row_id: MessageRowId | None,
        asset: MediaAsset,
        description: str | None,
    ) -> MediaAssetCandidate:
        """One chat-scoped candidate carrying ONLY opaque values: the
        content-addressed cache key, the content sha256, and the scrubbed
        vision description. The original URL never enters the catalog."""
        return MediaAssetCandidate(
            chat_key=msg.chat_key,
            kind=_media_kind(msg) or "sticker",
            cache_key=asset.key,
            sha256=asset.source_sha256,
            mime=asset.mime,
            width=asset.width,
            height=asset.height,
            description=description,
            source_message_id=row_id,
            source_sender_id=msg.sender_id,
            source_sender_name=msg.sender_name,
            source_ts=msg.recv_ts,
        )

    async def _describe(self, chat_key: ChatKey, asset: MediaAsset) -> VisionResult:
        """One budget-admitted vision call describing the asset.

        Returns a STRICT structured verdict: ``safe=True`` ONLY when the
        vision response carried an explicit boolean ``safe: true``
        classification AND a valid bounded escaped one-line description. A
        missing profile, a blocked budget, a provider failure, or a
        malformed/unsafe/empty response yields ``safe=False`` — the
        candidate then stays PENDING (unapproved). The vision prompt carries
        the normalized data URL (the media payload), never the original URL.
        """
        if self._llm is None or not self._cfg.vision_profile:
            return VisionResult(safe=False)
        if self._budget is None:
            return VisionResult(safe=False)
        decision = await self._budget.reserve(chat_key, calls=1)
        if decision.kind != "allowed":
            return VisionResult(safe=False)  # the budget released the slot itself
        try:
            data_url = self._store.asset_data_url(asset)
            msgs = [
                TranscriptMessage(
                    role="system",
                    content=(
                        "你是图片安全审核助手。用一句简短的中文描述图片内容，并给出安全判定。"
                        "必须只输出一个 JSON 对象："
                        '{"safe": true 或 false, "description": "一句简短描述"}。'
                        "描述中不要包含任何 URL、文件路径、平台引用或 base64。"
                    ),
                ),
                TranscriptMessage(
                    role="user",
                    content=render_image_markdown("media", data_url),
                ),
            ]
            resp = await self._llm.complete(
                msgs, profile=self._cfg.vision_profile, max_tokens=64
            )
            tokens = _usage_tokens(resp.usage)
            await self._budget.record(chat_key, calls=0, tokens=tokens)
            return parse_vision_result(resp.content)
        except Exception:
            self._budget.release()
            return VisionResult(safe=False)


class MediaRevoker:
    """Contained source-deletion/recall revocation (P6.5b).

    Wired at the ingest boundary as the ``revoke_media`` callback: when a
    platform recall/delete notice event arrives, the catalog assets whose
    source message was recalled are revoked (approved -> revoked, terminal)
    so a recalled source can never be sent again. This is a LOCAL catalog
    transition only — no platform send/delete API is invented or called.

    The recall payload is parsed conservatively (OneBot v11 ``group_recall``
    / ``friend_recall``): the chat key is derived from the platform ids and
    the recalled platform message id is mapped to the local durable row id
    through the repository's ``get_message`` (when the seam provides it).
    Every failure is contained and logged — a recall that cannot be mapped
    or revoked never raises into the ingest path.
    """

    def __init__(
        self,
        repo: Any,
        *,
        clock: Any = None,
        cfg: MediaConfig | None = None,
    ) -> None:
        self._repo = repo
        self._clock = clock if clock is not None else RealClock()
        self._cfg = cfg

    async def maybe_revoke(self, event: AdapterEvent) -> bool:
        """Revoke catalog assets sourced from a recalled message.

        Returns True when at least one approved asset was revoked; False
        when the event is not a recall, cannot be mapped, or nothing
        matched. Contained: never raises.
        """
        try:
            parsed = _parse_recall(event)
            if parsed is None:
                return False
            chat_key, platform_msg_id = parsed
            mark_deleted = getattr(self._repo, "mark_message_deleted", None)
            if mark_deleted is not None:
                row_id = await mark_deleted(
                    chat_key, MessageId(platform_msg_id), now=self._clock.now()
                )
            else:
                row_id = await self._source_row_id(chat_key, platform_msg_id)
            if row_id is None:
                return False
            revoked = False
            # A recall can arrive while vision approval is still pending.  A
            # pending source must be terminally rejected too, otherwise a
            # later retry could make recalled media sendable.
            list_candidates = getattr(self._repo, "list_media_candidates", None)
            reject_candidate = getattr(
                self._repo, "reject_media_candidate", None
            )
            if list_candidates is not None and reject_candidate is not None:
                candidates = await self._all_media_candidates(chat_key)
                for candidate in candidates:
                    if (
                        candidate.id is None
                        or candidate.source_message_id != row_id
                    ):
                        continue
                    if await reject_candidate(chat_key, candidate.id):
                        revoked = True
            assets = await self._all_media_assets(chat_key)
            for asset in assets:
                if asset.source_message_id != row_id:
                    continue
                if asset.safety_status != MediaSafetyStatus.APPROVED:
                    continue
                if await self._repo.revoke_media_asset(
                    chat_key, asset.id, now=self._clock.now()
                ):
                    revoked = True
            return revoked
        except Exception:
            _log.warning(
                "media recall revocation failed (contained)", exc_info=True
            )
            return False

    async def _all_media_candidates(
        self, chat_key: ChatKey, *, page_size: int = 200
    ) -> list[MediaAssetCandidate]:
        """Complete keyset scan of pending candidates for one chat."""
        page_after = getattr(self._repo, "list_media_candidates_after", None)
        if page_after is None:
            return await self._legacy_complete_scan(
                self._repo.list_media_candidates, chat_key, page_size
            )
        out: list[MediaAssetCandidate] = []
        after = 0
        while True:
            page = await page_after(chat_key, after, limit=page_size)
            if not page:
                return out
            out.extend(page)
            newest = max((candidate.id or after for candidate in page), default=after)
            if newest <= after:
                return out
            after = newest

    async def _all_media_assets(
        self, chat_key: ChatKey, *, page_size: int = 200
    ) -> list[Any]:
        """Complete keyset scan of all catalog statuses for one chat."""
        page_after = getattr(self._repo, "list_media_assets_after", None)
        if page_after is None:
            return await self._legacy_complete_scan(
                self._repo.list_media_assets, chat_key, page_size
            )
        out: list[Any] = []
        after = 0
        while True:
            page = await page_after(chat_key, after, limit=page_size)
            if not page:
                return out
            out.extend(page)
            newest = max((asset.id or after for asset in page), default=after)
            if newest <= after:
                return out
            after = newest

    @staticmethod
    async def _legacy_complete_scan(fetch: Any, chat_key: ChatKey, page_size: int) -> list[Any]:
        """Compatibility path for old fakes; concrete repos use keyset pages."""
        rows = await fetch(chat_key, limit=page_size)
        # Legacy list seams have no cursor parameter.  They are expected to
        # return their complete bounded view; never loop the same page forever.
        return list(rows)

    async def _source_row_id(
        self, chat_key: ChatKey, platform_msg_id: str
    ) -> MessageRowId | None:
        """Map a recalled platform message id to the local durable row id
        via the repository's ``get_message`` (when the seam provides it).
        None when the message is unknown or the repo lacks the lookup."""
        get_message = getattr(self._repo, "get_message", None)
        if get_message is None:
            return None
        try:
            msg = await get_message(chat_key, MessageId(platform_msg_id))
        except Exception:
            return None
        if msg is None:
            return None
        return msg.row_id


def _parse_recall(event: AdapterEvent) -> tuple[ChatKey, str] | None:
    """Parse a OneBot v11 recall notice into ``(chat_key, platform message
    id)``, or None for anything that is not a recall-shaped notice. No
    platform send/delete API is invented — only the local catalog is
    touched."""
    if event.type != "notice":
        return None
    payload = event.payload
    if not isinstance(payload, dict):
        return None
    notice_type = payload.get("notice_type")
    message_id = payload.get("message_id")
    if message_id is None:
        return None
    if notice_type == "group_recall":
        group_id = payload.get("group_id")
        if group_id is None:
            return None
        return ChatKey(f"qq:group:{group_id}"), str(message_id)
    if notice_type == "friend_recall":
        user_id = payload.get("user_id")
        if user_id is None:
            return None
        return ChatKey(f"qq:private:{user_id}"), str(message_id)
    return None


class MediaResolvingAdapter:
    """An Adapter wrapper that resolves opaque media cache keys at send time.

    The durable outbox stores media segments carrying ONLY the opaque
    content-addressed cache key (``data["media"]["key"]``). This wrapper
    maps that key to the normalized bytes via the MediaStore and injects the
    data URL into the outgoing IN MEMORY, immediately before the delegate
    adapter sends — so the durable outbox never carries a URL, file path, or
    base64 payload, while the platform still receives a sendable payload.
    Everything else is delegated unchanged.
    """

    def __init__(self, delegate: Any, store: MediaStore, repo: Any = None) -> None:
        self._delegate = delegate
        self._store = store
        self._repo = repo
        self.name = getattr(delegate, "name", None)
        self.capabilities = getattr(delegate, "capabilities", frozenset())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    async def send(self, out: Outgoing) -> str | None:
        keys = tuple(
            media.get("key")
            for seg in out.segments
            if seg.kind in ("image", "sticker")
            and isinstance((media := seg.data.get("media")), dict)
            and isinstance(media.get("key"), str)
            and media.get("key")
        )
        authorize = getattr(self._repo, "authorize_media_send", None)
        if keys and authorize is not None:
            allowed = (
                bool(out.delivery_key)
                and await authorize(out.chat_key, out.delivery_key, keys)
            )
            if not allowed:
                # AdapterNotReady is the driver's pre-wire, safe-retry
                # exception.  The repository has already durably dropped the
                # revoked row, so requeue_outbox is a no-op.
                raise AdapterNotReady("media asset is no longer sendable")
        return await self._delegate.send(self._resolve(out))

    def _resolve(self, out: Outgoing) -> Outgoing:
        if not out.segments:
            return out
        resolved: list[Segment] = []
        changed = False
        for seg in out.segments:
            if seg.kind in ("image", "sticker"):
                media = seg.data.get("media")
                key = media.get("key") if isinstance(media, dict) else None
                if isinstance(key, str) and key:
                    asset = self._store.asset_by_key(key)
                    if asset is not None:
                        data = dict(seg.data)
                        data["file"] = MediaStore.to_data_url(asset.data, asset.mime)
                        resolved.append(Segment(kind=seg.kind, data=data, raw=seg.raw))
                        changed = True
                        continue
            resolved.append(seg)
        if not changed:
            return out
        out.segments = resolved
        return out


def _usage_tokens(usage: dict[str, int] | None) -> int:
    """The total token count of one provider response."""
    return int((usage or {}).get("prompt_tokens", 0)) + int(
        (usage or {}).get("completion_tokens", 0)
    )
