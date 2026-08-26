"""MediaStore: download, content-addressed cache, Pillow clamp/convert,
base64/data-URL, safe size/type failure behavior, injected fetcher/cache,
SSRF URL/redirect rejection, malformed headers, and the pre-decode pixel cap."""

from __future__ import annotations

import asyncio
import io
import struct
import zlib

import httpx
import pytest
from PIL import Image

from pretender.media import (
    DiskMediaCache,
    HttpxMediaFetcher,
    InMemoryMediaCache,
    MediaAsset,
    MediaFetchError,
    MediaStore,
    MediaTooLarge,
    MediaTypeError,
    MediaUnsafeError,
    _PinnedAsyncTransport,
    is_safe_media_url,
    media_segment_data,
    resolve_safe_host,
)
from tests.durable_helpers import run


def make_png(size=(10, 10), color=(255, 0, 0), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def make_jpeg(size=(10, 10), color=(0, 255, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def make_bomb_png(width=6000, height=6000) -> bytes:
    """A structurally valid PNG whose header claims huge dimensions but whose
    payload is tiny — a decompression bomb. The pre-decode pixel cap must
    reject it from the header BEFORE Pillow decodes (and allocates) pixels."""

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + typ
            + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00" + b"\x00\x00\x00")  # one 1x1 RGB scanline
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", idat)
        + chunk(b"IEND", b"")
    )


class FakeFetcher:
    def __init__(self, responses=None, default=None, error=None):
        self.responses = responses or {}
        self.default = default
        self.error = error
        self.calls: list[str] = []

    async def fetch(self, url: str) -> bytes:
        self.calls.append(url)
        if self.error is not None:
            raise self.error
        if url in self.responses:
            return self.responses[url]
        if self.default is not None:
            return self.default
        raise MediaFetchError(f"no response for {url}")


def make_store(**kw) -> MediaStore:
    return MediaStore(**kw)


def make_httpx_fetcher(handler, **kw) -> HttpxMediaFetcher:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, follow_redirects=False)
    return HttpxMediaFetcher(client=client, **kw)


# ── download + normalize ────────────────────────────────────────────────────

def test_download_and_normalize_to_jpeg():
    async def scenario():
        fetcher = FakeFetcher(default=make_png())
        store = make_store(fetcher=fetcher)
        asset = await store.get("https://example.com/a.png")
        return asset, fetcher

    asset, fetcher = run(scenario())
    assert asset.mime == "image/jpeg"
    assert asset.format == "JPEG"
    assert asset.width == 10
    assert asset.height == 10
    assert asset.source_url == "https://example.com/a.png"
    assert asset.data.startswith(b"\xff\xd8")  # JPEG magic
    assert fetcher.calls == ["https://example.com/a.png"]


def test_content_addressed_cache_same_url_fetched_once():
    async def scenario():
        fetcher = FakeFetcher(default=make_png())
        store = make_store(fetcher=fetcher)
        a1 = await store.get("https://example.com/a.png")
        a2 = await store.get("https://example.com/a.png")
        return a1, a2, fetcher

    a1, a2, fetcher = run(scenario())
    assert a1.key == a2.key
    assert fetcher.calls == ["https://example.com/a.png"]


def test_content_addressed_cache_same_content_different_urls():
    async def scenario():
        data = make_png()
        fetcher = FakeFetcher(
            responses={
                "https://example.com/a.png": data,
                "https://cdn.example.com/b.png": data,
            }
        )
        store = make_store(fetcher=fetcher)
        a1 = await store.get("https://example.com/a.png")
        a2 = await store.get("https://cdn.example.com/b.png")
        return a1, a2, fetcher

    a1, a2, fetcher = run(scenario())
    assert a1.key == a2.key  # identical content -> identical key
    assert fetcher.calls == ["https://example.com/a.png", "https://cdn.example.com/b.png"]


def test_clamp_large_image_to_max_dim():
    async def scenario():
        fetcher = FakeFetcher(default=make_png(size=(3000, 2000)))
        store = make_store(fetcher=fetcher, max_dim=1280)
        asset = await store.get("https://example.com/big.png")
        return asset

    asset = run(scenario())
    assert asset.width <= 1280
    assert asset.height <= 1280
    # aspect preserved: 3000x2000 -> 1280x853 (rounded)
    assert asset.width == 1280
    assert asset.height == 853


def test_convert_rgba_to_rgb_jpeg():
    async def scenario():
        fetcher = FakeFetcher(default=make_png(mode="RGBA"))
        store = make_store(fetcher=fetcher)
        asset = await store.get("https://example.com/alpha.png")
        return asset

    asset = run(scenario())
    assert asset.mime == "image/jpeg"
    assert asset.format == "JPEG"
    # re-open and confirm it is RGB
    img = Image.open(io.BytesIO(asset.data))
    assert img.mode == "RGB"


# ── encoding helpers ────────────────────────────────────────────────────────

def test_to_data_url_and_segment_data():
    data = b"\xff\xd8\xff\xe0"
    url = MediaStore.to_data_url(data, "image/jpeg")
    assert url == "data:image/jpeg;base64,/9j/4A=="

    asset = MediaAsset(
        key="k", data=data, mime="image/jpeg", format="JPEG",
        width=1, height=1, source_url="u", source_sha256="s",
    )
    seg = media_segment_data(asset)
    assert seg["key"] == "k"
    assert seg["mime"] == "image/jpeg"
    assert seg["width"] == 1
    assert seg["data_url"].startswith("data:image/jpeg;base64,")


# ── failure behavior ────────────────────────────────────────────────────────

def test_too_large_raises_media_too_large():
    async def scenario():
        fetcher = FakeFetcher(default=b"x" * 100)
        store = make_store(fetcher=fetcher, max_bytes=50)
        with pytest.raises(MediaTooLarge):
            await store.get("https://example.com/big.bin")
        return fetcher

    run(scenario())


def test_bad_type_raises_media_type_error():
    async def scenario():
        fetcher = FakeFetcher(default=b"not an image at all")
        store = make_store(fetcher=fetcher)
        with pytest.raises(MediaTypeError):
            await store.get("https://example.com/notimg")
        return fetcher

    run(scenario())


def test_fetch_error_is_transient():
    async def scenario():
        fetcher = FakeFetcher(error=MediaFetchError("boom"))
        store = make_store(fetcher=fetcher)
        with pytest.raises(MediaFetchError):
            await store.get("https://example.com/down")
        return fetcher

    run(scenario())


# ── injected cache ──────────────────────────────────────────────────────────

def test_injected_in_memory_cache_avoids_fetch():
    async def scenario():
        fetcher = FakeFetcher(default=make_png())
        cache = InMemoryMediaCache()
        store = make_store(fetcher=fetcher, cache=cache)
        a1 = await store.get("https://example.com/a.png")
        # a second store sharing the same cache hits without fetching
        store2 = make_store(fetcher=fetcher, cache=cache)
        a2 = await store2.get("https://example.com/a.png")
        return a1, a2, fetcher

    a1, a2, fetcher = run(scenario())
    assert a1.key == a2.key
    assert fetcher.calls == ["https://example.com/a.png"]


def test_disk_cache_round_trip(tmp_path):
    async def scenario():
        fetcher = FakeFetcher(default=make_png())
        store = make_store(fetcher=fetcher, cache_dir=tmp_path)
        a1 = await store.get("https://example.com/a.png")
        # a fresh store over the same directory reads from disk, no fetch
        fetcher2 = FakeFetcher(default=make_png())
        store2 = make_store(fetcher=fetcher2, cache_dir=tmp_path)
        a2 = await store2.get("https://example.com/a.png")
        return a1, a2, fetcher2

    a1, a2, fetcher2 = run(scenario())
    assert a1.key == a2.key
    assert a1.data == a2.data
    assert fetcher2.calls == []  # served entirely from disk


def test_disk_cache_class_direct():
    cache = DiskMediaCache("/tmp/pretender-media-test")
    asset = MediaAsset(
        key="abc", data=b"\xff\xd8", mime="image/jpeg", format="JPEG",
        width=2, height=2, source_url="u", source_sha256="s",
    )
    cache.put(asset)
    cache.put_url("https://example.com/x", "abc")
    got = cache.get("abc")
    assert got is not None and got.data == b"\xff\xd8"
    assert cache.get_url("https://example.com/x") == "abc"


# ── SSRF guard: unsafe hosts / redirects ────────────────────────────────────

def test_is_safe_media_url():
    assert is_safe_media_url("https://example.com/a.png")
    assert is_safe_media_url("http://example.com/a.png")
    assert not is_safe_media_url("http://127.0.0.1/a")
    assert not is_safe_media_url("http://localhost/a")
    assert not is_safe_media_url("http://10.1.2.3/a")
    assert not is_safe_media_url("http://192.168.0.1/a")
    assert not is_safe_media_url("http://169.254.169.254/a")
    assert not is_safe_media_url("http://[::1]/a")
    assert not is_safe_media_url("http://[fe80::1]/a")
    assert not is_safe_media_url("http://224.0.0.1/a")
    assert not is_safe_media_url("http://[ff02::1]/a")
    assert not is_safe_media_url("ftp://example.com/a")
    assert not is_safe_media_url("file:///etc/passwd")
    assert not is_safe_media_url("")
    assert not is_safe_media_url("not a url")


def test_unsafe_url_rejected_by_store():
    async def scenario():
        store = make_store()
        for bad in (
            "http://127.0.0.1:8080/x",
            "http://localhost/x",
            "http://10.0.0.1/x",
            "http://192.168.1.1/x",
            "http://169.254.169.254/x",
            "http://[::1]/x",
            "ftp://example.com/x",
        ):
            with pytest.raises(MediaUnsafeError):
                await store.get(bad)
        return True

    assert run(scenario()) is True


def test_public_url_allowed():
    async def scenario():
        fetcher = FakeFetcher(default=make_png())
        store = make_store(fetcher=fetcher)
        asset = await store.get("https://example.com/a.png")
        return asset, fetcher

    asset, fetcher = run(scenario())
    assert asset.width == 10
    assert fetcher.calls == ["https://example.com/a.png"]


def test_redirect_to_private_host_rejected():
    """A redirect to a private/localhost literal is rejected (SSRF guard on
    every hop), even when the initial URL is public."""

    def handler(request):
        if request.url.host == "example.com":
            return httpx.Response(
                302, headers={"location": "http://127.0.0.1:8080/secret"}
            )
        return httpx.Response(200, content=b"ok")

    async def scenario():
        fetcher = make_httpx_fetcher(handler)
        with pytest.raises(MediaUnsafeError):
            await fetcher.fetch("https://example.com/a.png")
        return True

    assert run(scenario()) is True


def test_redirect_chain_to_public_target_allowed():
    def handler(request):
        if request.url.host == "example.com":
            return httpx.Response(302, headers={"location": "https://cdn.example.com/b.png"})
        return httpx.Response(200, content=b"\xff\xd8\xff\xe0")

    async def scenario():
        fetcher = make_httpx_fetcher(handler)
        data = await fetcher.fetch("https://example.com/a.png")
        return data

    assert run(scenario()) == b"\xff\xd8\xff\xe0"


def test_malformed_content_length_parsed_safely():
    """A malformed Content-Length header is ignored (the streaming cap still
    bounds the payload) — never a raw ValueError."""

    def handler(request):
        return httpx.Response(
            200, content=b"\xff\xd8\xff\xe0", headers={"content-length": "garbage"}
        )

    async def scenario():
        fetcher = make_httpx_fetcher(handler)
        data = await fetcher.fetch("https://example.com/a.png")
        return data

    assert run(scenario()) == b"\xff\xd8\xff\xe0"


def test_content_length_cap_still_enforced():
    def handler(request):
        return httpx.Response(
            200, content=b"x" * 100, headers={"content-length": "100"}
        )

    async def scenario():
        fetcher = make_httpx_fetcher(handler, max_bytes=50)
        with pytest.raises(MediaTooLarge):
            await fetcher.fetch("https://example.com/a.png")
        return True

    assert run(scenario()) is True


# ── pre-decode pixel cap (decompression bomb) ───────────────────────────────

def test_decompression_bomb_rejected_before_decode():
    async def scenario():
        fetcher = FakeFetcher(default=make_bomb_png(6000, 6000))
        store = make_store(fetcher=fetcher, max_pixels=25_000_000)
        with pytest.raises(MediaTypeError, match="pixels"):
            await store.get("https://example.com/bomb.png")
        return fetcher

    run(scenario())


def test_pixel_cap_allows_normal_image():
    async def scenario():
        fetcher = FakeFetcher(default=make_png(size=(100, 100)))
        store = make_store(fetcher=fetcher, max_pixels=25_000_000)
        asset = await store.get("https://example.com/a.png")
        return asset

    asset = run(scenario())
    assert asset.width == 100


# ── sync cache lookup + background prefetch ─────────────────────────────────

def test_cached_and_prefetch():
    async def scenario():
        fetcher = FakeFetcher(default=make_png())
        store = make_store(fetcher=fetcher)
        assert store.cached("https://example.com/a.png") is None
        asset = await store.prefetch("https://example.com/a.png")
        assert asset is not None
        assert store.cached("https://example.com/a.png") is not None
        # unsafe URLs never cache and never fetch
        assert store.cached("http://127.0.0.1/x") is None
        assert await store.prefetch("http://127.0.0.1/x") is None
        assert fetcher.calls == ["https://example.com/a.png"]
        return True

    assert run(scenario()) is True


# ── Gate 4 final: DNS rebinding / hostname pinning ───────────────────────────

def test_resolve_safe_host_rejects_private_resolution():
    """A DNS hostname that resolves to ANY non-global address is rejected
    (DNS-rebinding guard) — even when the literal hostname looks public."""
    async def scenario():
        async def addr_resolver(host):
            return ["10.0.0.5"]

        with pytest.raises(MediaUnsafeError, match="non-global"):
            await resolve_safe_host("attacker.example.com", resolver=addr_resolver)
        return True

    assert run(scenario()) is True


def test_resolve_safe_host_rejects_any_private_address_in_mix():
    """A hostname resolving to a MIX of public and private addresses is
    rejected (fail closed): one private address is enough."""
    async def scenario():
        async def addr_resolver(host):
            return ["93.184.216.34", "127.0.0.1"]

        with pytest.raises(MediaUnsafeError, match="non-global"):
            await resolve_safe_host("example.com", resolver=addr_resolver)
        return True

    assert run(scenario()) is True


def test_resolve_safe_host_pins_public_resolution():
    async def scenario():
        async def addr_resolver(host):
            return ["93.184.216.34", "93.184.216.35"]

        pinned = await resolve_safe_host("example.com", resolver=addr_resolver)
        return pinned

    assert run(scenario()) == "93.184.216.34"


def test_resolve_safe_host_unresolvable_rejected():
    """A host that cannot be resolved cannot be safely pinned — fail closed."""
    async def scenario():
        async def addr_resolver(host):
            raise OSError("NXDOMAIN")

        with pytest.raises(MediaUnsafeError, match="cannot resolve"):
            await resolve_safe_host("nope.example.com", resolver=addr_resolver)
        return True

    assert run(scenario()) is True


def test_resolve_safe_host_ip_literals():
    async def scenario():
        with pytest.raises(MediaUnsafeError):
            await resolve_safe_host("127.0.0.1")
        with pytest.raises(MediaUnsafeError):
            await resolve_safe_host("10.0.0.5")
        with pytest.raises(MediaUnsafeError):
            await resolve_safe_host("169.254.169.254")
        pinned = await resolve_safe_host("93.184.216.34")
        return pinned

    assert run(scenario()) == "93.184.216.34"


def test_fetcher_rejects_hostname_resolving_to_private():
    """The fetcher rejects a DNS hostname whose resolution is unsafe, even
    when an injected client would otherwise serve it."""
    async def scenario():
        async def safe_resolver(host):
            raise MediaUnsafeError(
                f"media host {host!r} resolves to non-global address 10.0.0.5"
            )

        fetcher = HttpxMediaFetcher(resolver=safe_resolver)
        with pytest.raises(MediaUnsafeError, match="non-global"):
            await fetcher.fetch("https://attacker.example.com/a.png")
        return True

    assert run(scenario()) is True


def test_fetcher_redirect_to_unsafe_resolution_rejected():
    """Redirect targets obey the same DNS policy: a redirect to a hostname
    that resolves unsafely is rejected on that hop."""
    def handler(request):
        if request.url.host == "example.com":
            return httpx.Response(
                302, headers={"location": "https://cdn.example.com/b.png"}
            )
        return httpx.Response(200, content=b"\xff\xd8\xff\xe0")

    async def scenario():
        async def safe_resolver(host):
            if host == "cdn.example.com":
                raise MediaUnsafeError(
                    f"media host {host!r} resolves to non-global address"
                )
            return "93.184.216.34"

        fetcher = make_httpx_fetcher(handler, resolver=safe_resolver)
        with pytest.raises(MediaUnsafeError, match="non-global"):
            await fetcher.fetch("https://example.com/a.png")
        return True

    assert run(scenario()) is True


def test_pinned_transport_connects_to_pinned_ip():
    """The pinned transport connects to the VALIDATED address (not the
    hostname) and preserves the original Host header — a DNS-rebinding-safe
    transport that does not weaken HTTPS verification."""
    async def scenario():
        received_host: list[str] = []

        async def http_handler(reader, writer):
            await reader.readline()  # request line
            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                name, _, value = line.decode("latin-1").partition(":")
                headers[name.strip().lower()] = value.strip()
            received_host.append(headers.get("host", ""))
            body = b"\xff\xd8\xff\xe0"
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: "
                + str(len(body)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + body
            )
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(http_handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        async def pinned_resolver(host):
            return "127.0.0.1"  # pin to the local server

        transport = _PinnedAsyncTransport(10.0, resolver=pinned_resolver)
        client = httpx.AsyncClient(transport=transport, follow_redirects=False)
        try:
            resp = await client.get(f"http://example.com:{port}/a.png")
            assert resp.status_code == 200
            assert resp.content == b"\xff\xd8\xff\xe0"
            assert received_host == [f"example.com:{port}"]  # Host preserved
        finally:
            await client.aclose()
            server.close()
            await server.wait_closed()
        return True

    assert run(scenario()) is True


def test_fetcher_production_path_pins_to_validated_address():
    """The production fetcher (own client + pinned transport) connects to the
    validated address end to end."""
    async def scenario():
        async def http_handler(reader, writer):
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
            body = b"\xff\xd8\xff\xe0"
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: "
                + str(len(body)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + body
            )
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(http_handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        async def pinned_resolver(host):
            return "127.0.0.1"

        fetcher = HttpxMediaFetcher(resolver=pinned_resolver)
        try:
            data = await fetcher.fetch(f"http://example.com:{port}/a.png")
            assert data == b"\xff\xd8\xff\xe0"
        finally:
            server.close()
            await server.wait_closed()
        return True

    assert run(scenario()) is True


def test_pinned_transport_rejects_oversized_response_headers():
    """The pinned transport has its own bounded header parser; the media body
    cap alone must not be the first resource limit."""

    async def scenario():
        async def handler(reader, writer):
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
            writer.write(b"HTTP/1.1 200 OK\r\nX-Fill: " + b"x" * (70 * 1024))
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        async def resolver(_host):
            return "127.0.0.1"

        client = httpx.AsyncClient(
            transport=_PinnedAsyncTransport(1.0, resolver=resolver)
        )
        try:
            with pytest.raises(MediaFetchError, match="response"):
                await client.get(f"http://example.com:{port}/image")
        finally:
            await client.aclose()
            server.close()
            await server.wait_closed()

    run(scenario())


def test_pinned_transport_rejects_too_many_response_headers():
    async def scenario():
        async def handler(reader, writer):
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
            headers = b"".join(b"X-Test: value\r\n" for _ in range(101))
            writer.write(b"HTTP/1.1 200 OK\r\n" + headers + b"\r\n")
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        async def resolver(_host):
            return "127.0.0.1"

        client = httpx.AsyncClient(
            transport=_PinnedAsyncTransport(1.0, resolver=resolver)
        )
        try:
            with pytest.raises(MediaFetchError, match="too many headers"):
                await client.get(f"http://example.com:{port}/image")
        finally:
            await client.aclose()
            server.close()
            await server.wait_closed()

    run(scenario())


def test_pinned_transport_uses_one_absolute_response_deadline():
    async def scenario():
        async def handler(reader, writer):
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
            writer.write(b"HTTP/1.1 200 OK\r\n")
            await writer.drain()
            await asyncio.sleep(0.15)
            writer.close()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        async def resolver(_host):
            return "127.0.0.1"

        fetcher = HttpxMediaFetcher(timeout_s=0.03, resolver=resolver)
        try:
            with pytest.raises(MediaFetchError, match="download failed"):
                await fetcher.fetch(f"http://example.com:{port}/image")
        finally:
            server.close()
            await server.wait_closed()

    run(scenario())


def test_dns_validation_is_bounded_by_media_request_deadline():
    async def scenario():
        async def slow_resolver(_host):
            await asyncio.sleep(0.1)
            return "203.0.113.10"

        fetcher = HttpxMediaFetcher(timeout_s=0.01, resolver=slow_resolver)
        with pytest.raises(MediaFetchError, match="download failed"):
            await fetcher.fetch("http://example.com/image")

    run(scenario())
