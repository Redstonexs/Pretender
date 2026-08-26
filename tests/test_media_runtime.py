"""Phase 6 P6.5b media runtime: the bounded/cancellable/advisory harvest
lane, the send-time cache-key resolver, and the staged media send tools.

Covers: harvest policy (opt-in group non-self stickers; private/image gated
by config switches); the harvest flow (fetch -> opaque candidate -> vision
approval); vision budget-block/provider-failure/malformed containment; task
cancellation; the MediaResolvingAdapter (durable outbox never carries a
URL/data URL); and the send_emoji / send_image tools (opaque asset id only,
staged intent, mutual exclusion with reply/wait/no_action, capability
gating, zero adapter/outbox/catalog writes during dispatch).

Async tests run via asyncio.run() so the test extra stays at just pytest.
"""

from __future__ import annotations

import asyncio
import io
import json

import pytest
from PIL import Image

from pretender.budget import BudgetDecision, BudgetUsage
from pretender.config import MediaConfig
from pretender.emoji import normalize_description, parse_vision_result
from pretender.media import (
    InMemoryMediaCache,
    MediaHarvester,
    MediaResolvingAdapter,
    MediaStore,
)
from pretender.tools.core import ToolContext, dispatch_call, register_core_tools
from pretender.tools.media import (
    MEDIA_REPLY_MARKER,
    MEDIA_TOOL_NAMES,
    MediaCallbacks,
    MediaReplyIntent,
    catalog_prompt,
    media_segment_for_intent,
)
from pretender.types import (
    ChatKey,
    LLMResponse,
    MediaAsset,
    MediaAssetCandidate,
    MediaKind,
    MediaSafetyStatus,
    Message,
    MessageId,
    MessageRowId,
    Outgoing,
    SenderId,
    Segment,
    ToolCall,
    ToolCallId,
)
from tests.durable_helpers import CK, run


def make_png(size=(10, 10), color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def make_sticker_message(
    *,
    chat_key=CK,
    url="https://example.com/sticker.gif",
    is_self=False,
    sender="u1",
    msg_id="m1",
    row_id=1,
) -> Message:
    return Message(
        chat_key=chat_key,
        sender_id=SenderId(sender),
        sender_name=sender,
        is_self=is_self,
        text="",
        id=MessageId(msg_id),
        segments=(Segment("sticker", {"url": url}),),
        recv_ts=100.0,
        row_id=MessageRowId(row_id),
    )


def make_image_message(*, chat_key=CK, url="https://example.com/a.png", **kw) -> Message:
    return Message(
        chat_key=chat_key,
        sender_id=SenderId("u1"),
        sender_name="u1",
        is_self=False,
        text="",
        id=MessageId("m1"),
        segments=(Segment("image", {"url": url}),),
        recv_ts=100.0,
        row_id=MessageRowId(1),
        **kw,
    )


class FakeFetcher:
    def __init__(self, default=None, error=None):
        self.default = default
        self.error = error
        self.calls: list[str] = []

    async def fetch(self, url: str) -> bytes:
        self.calls.append(url)
        if self.error is not None:
            raise self.error
        if self.default is not None:
            return self.default
        raise RuntimeError(f"no response for {url}")


class FakeMediaRepo:
    """A MediaRepository fake recording every call."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.submitted: list[MediaAssetCandidate] = []
        self.approved: list[int] = []
        self.next_id = 1

    async def submit_media_candidate(self, candidate, *, now):
        self.calls.append(("submit", candidate, now))
        self.submitted.append(candidate)
        cid = self.next_id
        self.next_id += 1
        return cid

    async def approve_media_candidate(self, chat_key, candidate_id, *, capacity, now):
        self.calls.append(("approve", chat_key, candidate_id, capacity, now))
        self.approved.append(candidate_id)
        return MediaAsset(
            id=candidate_id,
            chat_key=chat_key,
            kind="sticker",
            cache_key="c" * 64,
            sha256="a" * 64,
            mime="image/gif",
            safety_status=MediaSafetyStatus.APPROVED,
        )

    async def get_media_candidate(self, chat_key, candidate_id):
        return None

    async def list_media_candidates(self, chat_key, *, kind=None, limit=100):
        return []

    async def reject_media_candidate(self, chat_key, candidate_id):
        return False

    async def revoke_media_asset(self, chat_key, asset_id, *, now):
        return False

    async def select_media_assets(self, chat_key, kind, *, limit=1, cooldown_s=0.0, now):
        return []

    async def use_media_asset(self, chat_key, asset_id, *, now):
        return False

    async def list_media_assets(self, chat_key, *, kind=None, limit=100):
        self.calls.append(("list", chat_key, kind, limit))
        return []


class FakeVisionLLM:
    def __init__(self, content='{"safe": true, "description": "一个微笑的表情"}', error=None):
        self.content = content
        self.error = error
        self.calls: list[tuple] = []

    async def complete(
        self, messages, *, profile, tools=None, temperature=None, max_tokens=None, deadline=None
    ):
        self.calls.append((list(messages), profile))
        if self.error is not None:
            raise self.error
        return LLMResponse(
            content=self.content,
            usage={"prompt_tokens": 5, "completion_tokens": 3},
        )


class FakeBudget:
    def __init__(self, kind="allowed"):
        self.kind = kind
        self.reserve_calls: list[tuple] = []
        self.record_calls: list[tuple] = []
        self.release_calls = 0

    async def reserve(self, chat_key, *, calls=1):
        self.reserve_calls.append((chat_key, calls))
        return BudgetDecision(
            kind=self.kind,
            usage=BudgetUsage(day="2026-01-01", calls=0, tokens=0, cost=0.0),
            remaining=100,
        )

    async def record(self, chat_key, *, calls=0, tokens=0, cost=0.0):
        self.record_calls.append((chat_key, calls, tokens, cost))
        return BudgetUsage(day="2026-01-01", calls=calls, tokens=tokens, cost=cost)

    def release(self) -> None:
        self.release_calls += 1


def make_harvester(
    repo=None,
    *,
    cfg=None,
    llm=None,
    budget=None,
    fetcher=None,
    max_concurrency=2,
    timeout_s=5.0,
    max_tasks=64,
    vision_profile="vision",
) -> MediaHarvester:
    store = MediaStore(
        fetcher=fetcher or FakeFetcher(default=make_png()),
        cache=InMemoryMediaCache(),
    )
    if cfg is None:
        cfg = MediaConfig(enabled=True, harvest=True, vision_profile=vision_profile)
    return MediaHarvester(
        repo or FakeMediaRepo(),
        store,
        cfg=cfg,
        clock=_FakeClock(),
        llm=llm,
        budget=budget,
        max_concurrency=max_concurrency,
        timeout_s=timeout_s,
        max_tasks=max_tasks,
    )


class _FakeClock:
    def now(self) -> float:
        return 200.0


def _approved_asset(
    asset_id=3, kind="sticker", cache_key=None, description="微笑"
) -> MediaAsset:
    return MediaAsset(
        id=asset_id,
        chat_key=CK,
        kind=kind,
        cache_key=cache_key or "c" * 64,
        sha256="a" * 64,
        mime="image/gif",
        description=description,
        safety_status=MediaSafetyStatus.APPROVED,
    )


def _media_callbacks(assets=None, enabled=True):
    assets = list(assets or [])

    async def resolve_asset(asset_id: int):
        for asset in assets:
            if asset.id == asset_id:
                return asset
        return None

    return MediaCallbacks(
        catalog_enabled=lambda: enabled,
        resolve_asset=resolve_asset,
    )


def _ctx(registry, *, capabilities=frozenset({"sticker", "image"}), media=None):
    return ToolContext(
        chat_key=CK,
        chat_kind="group",
        capabilities=capabilities,
        registry=registry,
        media=media,
    )


def _call(name, arguments, cid="c1") -> ToolCall:
    return ToolCall(id=ToolCallId(cid), name=name, arguments=dict(arguments))


# ── MediaReplyIntent / segment / prompt ──────────────────────────────────────

def test_media_reply_intent_validates():
    intent = MediaReplyIntent(kind="emoji", asset_id=3, cache_key="c" * 64)
    assert intent.kind == "emoji"
    assert intent.asset_id == 3
    assert intent.cache_key == "c" * 64
    with pytest.raises(ValueError):
        MediaReplyIntent(kind="video", asset_id=3, cache_key="c" * 64)
    with pytest.raises(ValueError):
        MediaReplyIntent(kind="emoji", asset_id=0, cache_key="c" * 64)
    with pytest.raises(ValueError):
        MediaReplyIntent(kind="emoji", asset_id=3, cache_key="")
    with pytest.raises(ValueError):
        MediaReplyIntent(kind="emoji", asset_id=3, cache_key="https://example.com/x")


def test_media_segment_for_intent_carries_only_opaque_key():
    intent = MediaReplyIntent(kind="emoji", asset_id=3, cache_key="c" * 64)
    seg = media_segment_for_intent(intent)
    assert seg.kind == "sticker"
    assert seg.data == {"media": {"key": "c" * 64}}
    # No URL, path, platform ref, or base64 anywhere in the segment.
    rendered = json.dumps(seg.data)
    assert "http" not in rendered
    assert "base64" not in rendered
    assert "file=" not in rendered
    img = MediaReplyIntent(kind="image", asset_id=4, cache_key="d" * 64)
    assert media_segment_for_intent(img).kind == "image"


def test_catalog_prompt_renders_opaque_ids_only():
    assets = [
        _approved_asset(asset_id=3, description="微笑"),
        _approved_asset(asset_id=7, description="大笑"),
        _approved_asset(asset_id=5, kind="image", description="风景"),
    ]
    text = catalog_prompt(assets)
    assert "3: 微笑" in text
    assert "7: 大笑" in text
    assert "5: 风景" in text
    assert "send_emoji" in text
    assert "send_image" in text
    # No actual URL / data / base64 payload anywhere in the listing.
    assert "http" not in text
    assert "data:" not in text
    assert "base64," not in text
    assert "R0lGOD" not in text
    assert "example.com" not in text


def test_catalog_prompt_empty():
    assert catalog_prompt([]) == ""


def test_catalog_prompt_defensively_removes_source_tokens_and_newlines():
    text = catalog_prompt(
        [
            _approved_asset(
                description="第一行\nhttps://example.com/x /tmp/a base64://payload"
            )
        ]
    )
    assert "example.com" not in text
    assert "/tmp/a" not in text
    assert "base64://" not in text
    assert "\nhttps" not in text


# ── harvest policy (opt-in, config-owned switches) ───────────────────────────

def test_harvest_disabled_or_harvest_off_never_schedules():
    async def scenario():
        h = make_harvester(cfg=MediaConfig(enabled=False, harvest=True))
        a = h.maybe_harvest(make_sticker_message(), MessageRowId(1))
        h2 = make_harvester(cfg=MediaConfig(enabled=True, harvest=False))
        b = h2.maybe_harvest(make_sticker_message(), MessageRowId(1))
        return a, b

    a, b = run(scenario())
    assert a is None
    assert b is None


def test_harvest_group_nonself_sticker_schedules():
    async def scenario():
        h = make_harvester()
        task = h.maybe_harvest(make_sticker_message(), MessageRowId(1))
        if task is not None:
            task.cancel()
        h.cancel()
        return task

    task = run(scenario())
    assert task is not None


def test_harvest_group_self_sticker_excluded_by_default():
    async def scenario():
        h = make_harvester()
        a = h.maybe_harvest(make_sticker_message(is_self=True), MessageRowId(1))
        # group_nonself_stickers_only=False allows self stickers.
        h2 = make_harvester(
            cfg=MediaConfig(enabled=True, harvest=True, group_nonself_stickers_only=False)
        )
        b = h2.maybe_harvest(make_sticker_message(is_self=True), MessageRowId(1))
        if b is not None:
            b.cancel()
        h.cancel()
        h2.cancel()
        return a, b

    a, b = run(scenario())
    assert a is None
    assert b is not None


def test_harvest_group_image_never_scheduled():
    async def scenario():
        h = make_harvester()
        task = h.maybe_harvest(make_image_message(), MessageRowId(1))
        h.cancel()
        return task

    assert run(scenario()) is None


def test_harvest_private_policy_gated_by_config():
    async def scenario():
        priv = ChatKey("qq:private:654321")
        # Private stickers disabled by default.
        h = make_harvester()
        a = h.maybe_harvest(make_sticker_message(chat_key=priv), MessageRowId(1))
        # Private stickers enabled.
        h2 = make_harvester(
            cfg=MediaConfig(enabled=True, harvest=True, private_stickers_enabled=True)
        )
        b = h2.maybe_harvest(make_sticker_message(chat_key=priv), MessageRowId(1))
        if b is not None:
            b.cancel()
        # Private images enabled.
        h3 = make_harvester(
            cfg=MediaConfig(enabled=True, harvest=True, private_images_enabled=True)
        )
        c = h3.maybe_harvest(make_image_message(chat_key=priv), MessageRowId(1))
        if c is not None:
            c.cancel()
        # Private images disabled.
        h4 = make_harvester(cfg=MediaConfig(enabled=True, harvest=True))
        d = h4.maybe_harvest(make_image_message(chat_key=priv), MessageRowId(1))
        for hh in (h, h2, h3, h4):
            hh.cancel()
        return a, b, c, d

    a, b, c, d = run(scenario())
    assert a is None
    assert b is not None
    assert c is not None
    assert d is None


def test_harvest_ignores_non_media_messages():
    async def scenario():
        h = make_harvester()
        msg = Message(
            chat_key=CK,
            sender_id=SenderId("u1"),
            sender_name="u1",
            is_self=False,
            text="hello",
            id=MessageId("m1"),
            segments=(Segment("text", {"text": "hi"}),),
            recv_ts=100.0,
        )
        task = h.maybe_harvest(msg, MessageRowId(1))
        h.cancel()
        return task

    assert run(scenario()) is None


# ── harvest flow: fetch -> opaque candidate -> vision approval ───────────────

def test_harvest_submits_opaque_candidate_and_approves_with_vision():
    async def scenario():
        repo = FakeMediaRepo()
        llm = FakeVisionLLM(
            content='{"safe": true, "description": "一个微笑的表情"}'
        )
        h = make_harvester(repo, llm=llm, budget=FakeBudget())
        task = h.maybe_harvest(make_sticker_message(), MessageRowId(1))
        assert task is not None
        await task
        return repo, llm

    repo, llm = run(scenario())
    assert len(repo.submitted) == 1
    cand = repo.submitted[0]
    # The catalog key is the OPAQUE content-addressed cache key.
    assert len(cand.cache_key) == 64
    assert cand.cache_key == cand.cache_key.lower()
    assert "example.com" not in cand.cache_key
    assert cand.kind == MediaKind.STICKER
    assert cand.source_message_id == MessageRowId(1)
    assert cand.source_sender_id == SenderId("u1")
    assert cand.description == "一个微笑的表情"
    # Vision succeeded -> the candidate was approved.
    assert repo.approved == [1]
    assert len(llm.calls) == 1
    # The vision prompt carries the normalized data URL, never the original.
    prompt_text = llm.calls[0][0][1].content
    assert "example.com" not in prompt_text


def test_harvest_no_vision_profile_leaves_asset_pending():
    async def scenario():
        repo = FakeMediaRepo()
        h = make_harvester(repo, llm=None, budget=None)
        task = h.maybe_harvest(make_sticker_message(), MessageRowId(1))
        assert task is not None
        await task
        return repo

    repo = run(scenario())
    assert len(repo.submitted) == 1
    assert repo.submitted[0].description is None
    assert repo.approved == []  # unapproved: stays pending


def test_harvest_budget_blocked_leaves_asset_pending():
    async def scenario():
        repo = FakeMediaRepo()
        llm = FakeVisionLLM()
        budget = FakeBudget(kind="blocked")
        h = make_harvester(repo, llm=llm, budget=budget)
        task = h.maybe_harvest(make_sticker_message(), MessageRowId(1))
        assert task is not None
        await task
        return repo, llm, budget

    repo, llm, budget = run(scenario())
    assert len(repo.submitted) == 1
    assert repo.approved == []  # blocked budget -> unapproved
    assert llm.calls == []  # zero provider calls
    assert budget.reserve_calls == [(CK, 1)]


def test_harvest_provider_failure_leaves_asset_pending():
    async def scenario():
        repo = FakeMediaRepo()
        llm = FakeVisionLLM(error=RuntimeError("provider 500"))
        budget = FakeBudget()
        h = make_harvester(repo, llm=llm, budget=budget)
        task = h.maybe_harvest(make_sticker_message(), MessageRowId(1))
        assert task is not None
        await task
        return repo, budget

    repo, budget = run(scenario())
    assert len(repo.submitted) == 1
    assert repo.approved == []  # provider failure -> unapproved
    assert budget.release_calls == 1  # the held slot was released


def test_harvest_vision_source_token_is_unapproved():
    async def scenario():
        repo = FakeMediaRepo()
        # A structured description that echoes the original URL is unsafe;
        # stripping the URL must not turn it into an approval.
        llm = FakeVisionLLM(
            content=(
                '{"safe": true, "description": '
                '"好看 https://example.com/sticker.gif 的表情"}'
            )
        )
        h = make_harvester(repo, llm=llm, budget=FakeBudget())
        task = h.maybe_harvest(make_sticker_message(), MessageRowId(1))
        assert task is not None
        await task
        return repo

    repo = run(scenario())
    assert len(repo.submitted) == 1
    desc = repo.submitted[0].description
    assert desc is None
    assert repo.approved == []


def test_harvest_empty_vision_response_unapproved():
    async def scenario():
        repo = FakeMediaRepo()
        llm = FakeVisionLLM(content="   ")
        h = make_harvester(repo, llm=llm, budget=FakeBudget())
        task = h.maybe_harvest(make_sticker_message(), MessageRowId(1))
        assert task is not None
        await task
        return repo

    repo = run(scenario())
    assert repo.submitted[0].description is None
    assert repo.approved == []


@pytest.mark.parametrize(
    "content",
    [
        "一个微笑的表情",  # obsolete freeform output
        '{"description": "一个微笑的表情"}',  # missing explicit approval
        '{"safe": "true", "description": "一个微笑的表情"}',
        '{"safe": false, "description": "一个微笑的表情"}',
        '{"safe": true, "description": ""}',
        '{"safe": true, "description": "x"' + "x" * 120 + "}",
    ],
)
def test_harvest_nonapproved_structured_verdict_stays_pending(content):
    async def scenario():
        repo = FakeMediaRepo()
        h = make_harvester(
            repo, llm=FakeVisionLLM(content=content), budget=FakeBudget()
        )
        task = h.maybe_harvest(make_sticker_message(), MessageRowId(1))
        assert task is not None
        await task
        return repo

    repo = run(scenario())
    assert len(repo.submitted) == 1
    assert repo.submitted[0].description is None
    assert repo.approved == []


def test_vision_description_is_one_line_and_structurally_escaped():
    value = normalize_description("一行\n下一行\t结束")
    assert value == r"一行\u000a下一行\u0009结束"
    assert "\n" not in value and "\t" not in value
    assert parse_vision_result(
        '{"safe": true, "description": "一行\\n下一行"}'
    ).safe


def test_candidate_cap_is_checked_before_fetch():
    class AtCapacity(FakeMediaRepo):
        async def list_media_candidates(self, chat_key, *, kind=None, limit=100):
            return [object()]

    async def scenario():
        repo = AtCapacity()
        fetcher = FakeFetcher(default=make_png())
        h = make_harvester(
            repo,
            cfg=MediaConfig(
                enabled=True, harvest=True, candidate_cap=1, vision_profile="vision"
            ),
            fetcher=fetcher,
            llm=FakeVisionLLM(),
            budget=FakeBudget(),
        )
        task = h.maybe_harvest(make_sticker_message(), MessageRowId(1))
        assert task is not None
        await task
        return repo, fetcher

    repo, fetcher = run(scenario())
    assert fetcher.calls == []
    assert repo.submitted == []


def test_harvester_task_set_has_a_hard_bound():
    async def scenario():
        h = make_harvester(max_tasks=2)
        tasks = [
            h.maybe_harvest(make_sticker_message(msg_id=f"m{i}"), MessageRowId(i))
            for i in range(3)
        ]
        assert tasks[0] is not None and tasks[1] is not None
        assert tasks[2] is None
        h.cancel()
        await asyncio.gather(
            *(task for task in tasks[:2] if task is not None),
            return_exceptions=True,
        )
        return h

    assert run(scenario()).active_count == 0


def test_harvest_fetch_failure_contained():
    async def scenario():
        repo = FakeMediaRepo()
        h = make_harvester(repo, fetcher=FakeFetcher(error=RuntimeError("network down")))
        task = h.maybe_harvest(make_sticker_message(), MessageRowId(1))
        assert task is not None
        await task  # must not raise
        return repo

    repo = run(scenario())
    assert repo.submitted == []
    assert repo.approved == []


def test_harvest_cancellation_cancels_tasks():
    async def scenario():
        repo = FakeMediaRepo()
        h = make_harvester(repo, timeout_s=30.0)
        task = h.maybe_harvest(make_sticker_message(), MessageRowId(1))
        assert task is not None
        assert h.active_count == 1
        h.cancel()
        await asyncio.sleep(0)
        return task, h

    task, h = run(scenario())
    assert task.cancelled() is True
    assert h.active_count == 0


def test_harvest_bounded_by_semaphore():
    async def scenario():
        repo = FakeMediaRepo()
        h = make_harvester(repo, max_concurrency=1, timeout_s=30.0)
        tasks = [
            h.maybe_harvest(make_sticker_message(msg_id=f"m{i}"), MessageRowId(i))
            for i in range(3)
        ]
        assert all(t is not None for t in tasks)
        await asyncio.gather(*[t for t in tasks if t is not None])
        return repo

    repo = run(scenario())
    assert len(repo.submitted) == 3


# ── MediaResolvingAdapter: durable outbox never carries a URL ────────────────

def test_resolving_adapter_injects_data_url_in_memory_only():
    async def scenario():
        store = MediaStore(
            fetcher=FakeFetcher(default=make_png()),
            cache=InMemoryMediaCache(),
        )
        asset = await store.get("https://example.com/a.png")
        sent: list[Outgoing] = []

        class Delegate:
            name = "onebot"
            capabilities = frozenset({"image", "sticker"})

            async def send(self, out: Outgoing) -> str | None:
                sent.append(out)
                return "pid-1"

        adapter = MediaResolvingAdapter(Delegate(), store)
        out = Outgoing(
            chat_key=CK,
            text="",
            segments=[Segment("image", {"media": {"key": asset.key}})],
        )
        pid = await adapter.send(out)
        return pid, sent, asset

    pid, sent, asset = run(scenario())
    assert pid == "pid-1"
    wire = sent[0].segments[0]
    assert wire.kind == "image"
    # The send-time segment carries the data URL (in memory only).
    assert wire.data["file"].startswith("data:image/jpeg;base64,")
    assert wire.data["media"]["key"] == asset.key
    # The durable segment shape (what the outbox stores) never carries it.
    durable = media_segment_for_intent(
        MediaReplyIntent(kind="image", asset_id=1, cache_key=asset.key)
    )
    assert "data:" not in json.dumps(durable.data)
    assert "http" not in json.dumps(durable.data)


def test_resolving_adapter_passthrough_without_media_segments():
    async def scenario():
        store = MediaStore(fetcher=FakeFetcher(default=make_png()), cache=InMemoryMediaCache())
        sent: list[Outgoing] = []

        class Delegate:
            async def send(self, out: Outgoing) -> str | None:
                sent.append(out)
                return None

        adapter = MediaResolvingAdapter(Delegate(), store)
        out = Outgoing(chat_key=CK, text="hello")
        await adapter.send(out)
        return sent

    sent = run(scenario())
    assert sent[0].text == "hello"
    assert sent[0].segments == []


def test_resolving_adapter_unknown_key_passthrough():
    async def scenario():
        store = MediaStore(fetcher=FakeFetcher(default=make_png()), cache=InMemoryMediaCache())
        sent: list[Outgoing] = []

        class Delegate:
            async def send(self, out: Outgoing) -> str | None:
                sent.append(out)
                return None

        adapter = MediaResolvingAdapter(Delegate(), store)
        out = Outgoing(
            chat_key=CK,
            text="",
            segments=[Segment("image", {"media": {"key": "f" * 64}})],
        )
        await adapter.send(out)
        return sent

    sent = run(scenario())
    # Unknown key: the segment is passed through untouched (no crash).
    assert sent[0].segments[0].data == {"media": {"key": "f" * 64}}


# ── send_emoji / send_image tools ────────────────────────────────────────────

def test_media_tools_are_deferred_and_capability_gated():
    reg = register_core_tools()
    assert MEDIA_TOOL_NAMES == ("send_emoji", "send_image")
    for name in MEDIA_TOOL_NAMES:
        spec = reg.require(name)
        assert spec.visibility == "deferred"
        assert spec.parameters["required"] == ["asset_id"]
        assert spec.parameters["properties"]["asset_id"] == {"type": "integer"}
    assert reg.require("send_emoji").capability == "sticker"
    assert reg.require("send_image").capability == "image"
    # Not emitted until tool_search activates them.
    names = [d["function"]["name"] for d in reg.provider_definitions()]
    assert "send_emoji" not in names
    assert "send_image" not in names


def test_send_emoji_stages_intent_without_any_writes():
    async def scenario():
        reg = register_core_tools()
        reg.activate("send_emoji")
        media = _media_callbacks([_approved_asset(asset_id=3)])
        ctx = _ctx(reg, media=media)
        result = await dispatch_call(
            _call("send_emoji", {"asset_id": 3}), ctx, reg
        )
        return result, ctx

    result, ctx = run(scenario())
    assert result.ok is True
    data = json.loads(result.content)
    assert data == {"action": "send_emoji", "asset_id": 3}
    assert ctx.media_intent is not None
    assert ctx.media_intent.kind == "emoji"
    assert ctx.media_intent.asset_id == 3
    assert ctx.media_intent.cache_key == "c" * 64
    # The sentinel reply marker makes the frozen planner loop break; the
    # cycle reads the typed media_intent FIRST and never treats it as text.
    assert ctx.reply_text == MEDIA_REPLY_MARKER
    assert ctx.wait_seconds is None
    assert ctx.no_action_verdict is False


def test_send_image_stages_intent():
    async def scenario():
        reg = register_core_tools()
        reg.activate("send_image")
        media = _media_callbacks([_approved_asset(asset_id=5, kind="image")])
        ctx = _ctx(reg, media=media)
        result = await dispatch_call(
            _call("send_image", {"asset_id": 5}), ctx, reg
        )
        return result, ctx

    result, ctx = run(scenario())
    assert result.ok is True
    assert ctx.media_intent is not None
    assert ctx.media_intent.kind == "image"
    assert ctx.media_intent.cache_key == "c" * 64


def test_send_emoji_rejects_urls_paths_base64_and_raw_cq():
    """The tool accepts an OPAQUE integer asset id ONLY — every URL/path/
    base64/raw-CQ form fails schema validation before the handler runs."""
    async def scenario():
        reg = register_core_tools()
        reg.activate("send_emoji")
        media = _media_callbacks([_approved_asset(asset_id=3)])
        ctx = _ctx(reg, media=media)
        results = []
        for bad in (
            "https://example.com/a.gif",
            "/abs/path/a.gif",
            "./rel/a.gif",
            "data:image/gif;base64,R0lGOD",
            "base64://R0lGOD",
            "{file=a.gif}",
            "3",  # string, not int
            3.5,
            True,
        ):
            res = await dispatch_call(
                _call("send_emoji", {"asset_id": bad}), ctx, reg
            )
            results.append((res.ok, res.error))
        return results, ctx

    results, ctx = run(scenario())
    for ok, error in results:
        assert ok is False, f"asset_id {error!r} must be rejected"
        assert "schema mismatch" in (error or "")
    assert ctx.media_intent is None


def test_send_emoji_requires_capability():
    async def scenario():
        reg = register_core_tools()
        reg.activate("send_emoji")
        media = _media_callbacks([_approved_asset(asset_id=3)])
        ctx = _ctx(reg, capabilities=frozenset({"image"}), media=media)
        result = await dispatch_call(
            _call("send_emoji", {"asset_id": 3}), ctx, reg
        )
        return result

    result = run(scenario())
    assert result.ok is False
    assert "sticker" in (result.error or "")


def test_send_image_requires_capability():
    async def scenario():
        reg = register_core_tools()
        reg.activate("send_image")
        media = _media_callbacks([_approved_asset(asset_id=5, kind="image")])
        ctx = _ctx(reg, capabilities=frozenset({"sticker"}), media=media)
        result = await dispatch_call(
            _call("send_image", {"asset_id": 5}), ctx, reg
        )
        return result

    result = run(scenario())
    assert result.ok is False
    assert "image" in (result.error or "")


def test_send_emoji_fails_closed_without_media_callbacks():
    async def scenario():
        reg = register_core_tools()
        reg.activate("send_emoji")
        ctx = _ctx(reg, media=None)
        result = await dispatch_call(
            _call("send_emoji", {"asset_id": 3}), ctx, reg
        )
        return result

    result = run(scenario())
    assert result.ok is False
    assert "media catalog is not available" in (result.error or "")


def test_send_emoji_fails_closed_when_catalog_disabled():
    async def scenario():
        reg = register_core_tools()
        reg.activate("send_emoji")
        media = _media_callbacks([_approved_asset(asset_id=3)], enabled=False)
        ctx = _ctx(reg, media=media)
        result = await dispatch_call(
            _call("send_emoji", {"asset_id": 3}), ctx, reg
        )
        return result

    result = run(scenario())
    assert result.ok is False
    assert "disabled" in (result.error or "")


def test_send_emoji_unknown_or_unapproved_asset_fails():
    async def scenario():
        reg = register_core_tools()
        reg.activate("send_emoji")
        pending = _approved_asset(asset_id=9)
        pending = MediaAsset(
            id=9,
            chat_key=CK,
            kind="sticker",
            cache_key="c" * 64,
            sha256="a" * 64,
            mime="image/gif",
            safety_status=MediaSafetyStatus.PENDING,
        )
        media = _media_callbacks([pending])
        ctx = _ctx(reg, media=media)
        r1 = await dispatch_call(_call("send_emoji", {"asset_id": 9}), ctx, reg)
        r2 = await dispatch_call(_call("send_emoji", {"asset_id": 999}), ctx, reg)
        return r1, r2

    r1, r2 = run(scenario())
    assert r1.ok is False
    assert "not approved" in (r1.error or "")
    assert r2.ok is False
    assert "unknown" in (r2.error or "")


def test_send_emoji_wrong_kind_fails():
    async def scenario():
        reg = register_core_tools()
        reg.activate("send_emoji")
        media = _media_callbacks([_approved_asset(asset_id=5, kind="image")])
        ctx = _ctx(reg, media=media)
        result = await dispatch_call(
            _call("send_emoji", {"asset_id": 5}), ctx, reg
        )
        return result

    result = run(scenario())
    assert result.ok is False
    assert "not a sticker" in (result.error or "")


# ── mutual exclusion with reply / wait / no_action ───────────────────────────

def test_media_tool_conflicts_with_staged_reply():
    async def scenario():
        reg = register_core_tools()
        reg.activate("send_emoji")
        media = _media_callbacks([_approved_asset(asset_id=3)])
        ctx = _ctx(reg, media=media)
        await dispatch_call(_call("reply", {"text": "hi"}), ctx, reg)
        result = await dispatch_call(
            _call("send_emoji", {"asset_id": 3}), ctx, reg
        )
        return result, ctx

    result, ctx = run(scenario())
    assert result.ok is False
    assert "conflicts" in (result.error or "")
    assert ctx.media_intent is None
    assert ctx.reply_text == "hi"  # the first terminal intent wins


def test_reply_conflicts_with_staged_media_intent():
    async def scenario():
        reg = register_core_tools()
        reg.activate("send_emoji")
        media = _media_callbacks([_approved_asset(asset_id=3)])
        ctx = _ctx(reg, media=media)
        await dispatch_call(_call("send_emoji", {"asset_id": 3}), ctx, reg)
        result = await dispatch_call(_call("reply", {"text": "hi"}), ctx, reg)
        return result, ctx

    result, ctx = run(scenario())
    assert result.ok is False
    assert "conflicts" in (result.error or "")
    assert ctx.media_intent is not None  # the first terminal intent wins


def test_wait_and_no_action_conflict_with_media_intent():
    async def scenario():
        reg = register_core_tools()
        reg.activate("send_emoji")
        media = _media_callbacks([_approved_asset(asset_id=3)])
        ctx = _ctx(reg, media=media)
        await dispatch_call(_call("send_emoji", {"asset_id": 3}), ctx, reg)
        r1 = await dispatch_call(_call("wait", {"seconds": 5}), ctx, reg)
        r2 = await dispatch_call(_call("no_action", {}), ctx, reg)
        return r1, r2

    r1, r2 = run(scenario())
    assert r1.ok is False and "conflicts" in (r1.error or "")
    assert r2.ok is False and "conflicts" in (r2.error or "")


def test_second_media_tool_conflicts():
    async def scenario():
        reg = register_core_tools()
        reg.activate("send_emoji")
        reg.activate("send_image")
        media = _media_callbacks(
            [_approved_asset(asset_id=3), _approved_asset(asset_id=5, kind="image")]
        )
        ctx = _ctx(reg, media=media)
        await dispatch_call(_call("send_emoji", {"asset_id": 3}), ctx, reg)
        result = await dispatch_call(_call("send_image", {"asset_id": 5}), ctx, reg)
        return result, ctx

    result, ctx = run(scenario())
    assert result.ok is False
    assert "already staged" in (result.error or "")
    assert ctx.media_intent is not None
    assert ctx.media_intent.kind == "emoji"  # first terminal intent wins

# ── OneBot echo flow with a durable media segment ────────────────────────────

def test_onebot_echo_flow_with_media_segment():
    """The durable media segment (opaque cache key) sends through the OneBot
    adapter via the send-time resolver, and the existing echo flow binds the
    delivery key — the outbox/self-echo path handles media unchanged."""
    async def scenario():
        from pretender.adapters.onebot import OneBotAdapter
        from pretender.clock import VirtualClock
        from pretender.config import OneBotConfig
        import orjson
        import websockets as ws_lib

        store = MediaStore(
            fetcher=FakeFetcher(default=make_png()),
            cache=InMemoryMediaCache(),
        )
        asset = await store.get("https://example.com/a.png")
        adapter = OneBotAdapter(
            config=OneBotConfig(host="127.0.0.1", port=0, heartbeat_timeout_s=None),
            clock=VirtualClock(),
            normalize_media=False,
            media=store,
        )
        resolver = MediaResolvingAdapter(adapter, store)
        await adapter.connect()
        port = adapter._server.sockets[0].getsockname()[1]
        uri = f"ws://127.0.0.1:{port}/onebot/v11/ws?message_format=array"
        client = await ws_lib.connect(uri)
        drain_task = asyncio.create_task(_drain_events(adapter))
        await asyncio.sleep(0.05)
        out = Outgoing(
            chat_key=CK,
            text="",
            segments=[Segment("image", {"media": {"key": asset.key}})],
            delivery_key="dispatch:1:0",
        )
        send_task = asyncio.create_task(resolver.send(out))
        # The platform receives the RESOLVED payload (data URL in memory).
        action = None
        while True:
            raw = await asyncio.wait_for(client.recv(), timeout=2.0)
            data = orjson.loads(raw)
            if data.get("action") == "get_login_info":
                await client.send(
                    orjson.dumps(
                        {"status": "ok", "retcode": 0, "data": {"user_id": 10001}, "echo": data["echo"]}
                    )
                )
                continue
            action = data
            break
        assert action is not None
        assert action["action"] == "send_group_msg"
        msg = action["params"]["message"]
        assert msg[0]["type"] == "image"
        assert msg[0]["data"]["file"].startswith("data:image/jpeg;base64,")
        await client.send(
            orjson.dumps(
                {"status": "ok", "retcode": 0, "data": {"message_id": 90001}, "echo": action["echo"]}
            )
        )
        pid = await send_task
        await client.close()
        await adapter.close()
        await drain_task
        return pid, adapter

    pid, adapter = run(scenario())
    assert pid == "90001"
    # The trusted delivery key is bound to the real platform id — the echo
    # flow reconciles the media send exactly like a text send.
    assert adapter._delivered.get("90001") == "dispatch:1:0"


async def _drain_events(adapter):
    async for _ in adapter.events():
        pass
