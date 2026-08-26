"""Phase 6 P6.5b media send tools: ``send_emoji`` / ``send_image``.

These are capability-gated, deferred, STAGED terminal tools. They never
touch an adapter, the outbox, usage accounting, or the repository directly:
the live ``ToolContext`` carries chat-bound ``MediaCallbacks`` (injected by
the cycle at construction), and a successful call only stages one typed
``MediaReplyIntent`` on the context. The CycleRunner converts a valid intent
into an ``Outgoing`` media segment ONLY at normal terminal settlement, and
the existing output pipeline / outbox / self-echo path handles delivery.

Design rules:

  - The tools accept an OPAQUE approved catalog asset id only. No URLs, file
    paths, platform references, base64 payloads, or cache-key source values
    ever enter the tool arguments, the staged intent, or the prompt.
  - ``send_emoji`` / ``send_image`` are mutually exclusive with
    ``reply`` / ``wait`` / ``no_action``: the FIRST terminal intent wins and
    a conflicting call is a ``ToolError`` (never a silent override).
  - The staged intent carries the opaque asset id and the opaque
    content-addressed cache key (the sha256 hex digest of the normalized
    bytes) — never a fetchable/executable source.
  - Nothing here writes anywhere: no adapter.send, no outbox rows, no usage
    records, no catalog writes. All of that happens post-terminal in the
    CycleRunner / App.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pretender.emoji import (
    normalize_description,
    validate_asset_id,
    validate_catalog_key,
)
from pretender.errors import ToolError
from pretender.types import MediaAsset, MediaSafetyStatus, Segment

__all__ = [
    "MEDIA_TOOL_NAMES",
    "MediaCallbacks",
    "MediaReplyIntent",
    "catalog_prompt",
    "media_segment_for_intent",
    "send_emoji",
    "send_image",
]

#: The two media send tool names, in registration order.
MEDIA_TOOL_NAMES: tuple[str, ...] = ("send_emoji", "send_image")

#: The catalog kind each tool stages (the tool name -> catalog kind map).
_TOOL_KIND: dict[str, str] = {
    "send_emoji": "emoji",
    "send_image": "image",
}

#: The segment kind each tool stages (the catalog kind -> segment kind map).
_SEGMENT_KIND: dict[str, str] = {
    "emoji": "sticker",
    "image": "image",
}

#: The sentinel reply marker a staged media send sets on the ToolContext so
#: the planner's existing ``reply`` terminal break fires (the planner loop
#: is frozen and cannot be edited). The marker is a NUL-prefixed token that
#: can never be a real reply; the cycle reads the typed ``media_intent``
#: FIRST and never passes the marker to the replyer or the outbox.
MEDIA_REPLY_MARKER = "\u0000media"


@dataclass(frozen=True)
class MediaReplyIntent:
    """One staged media send: the typed terminal intent the media tools
    produce.

    ``kind`` is ``emoji`` | ``image``; ``asset_id`` is the OPAQUE approved
    catalog asset id the planner selected; ``cache_key`` is the OPAQUE
    content-addressed cache key (the sha256 hex digest of the normalized
    bytes) the outbox segment carries. Neither field can name a fetchable/
    executable source — no URLs, paths, platform refs, or base64 payloads.
    """

    kind: str
    asset_id: int
    cache_key: str

    def __post_init__(self) -> None:
        if self.kind not in ("emoji", "image"):
            raise ValueError(f"invalid media intent kind: {self.kind!r}")
        if isinstance(self.asset_id, bool) or not isinstance(self.asset_id, int):
            raise ValueError("asset_id must be an integer")
        if self.asset_id <= 0:
            raise ValueError("asset_id must be a positive integer")
        validate_catalog_key(self.cache_key)


@dataclass(frozen=True)
class MediaCallbacks:
    """The chat-bound catalog callbacks the media tools speak to.

    The cycle binds these to the CURRENT chat at ``ToolContext``
    construction, so a cross-chat lookup is impossible by construction. The
    tools never hold a repository reference — ``resolve_asset`` is the only
    surface they read, and it returns an opaque approved ``MediaAsset`` (or
    None). ``catalog_enabled`` is a sync config gate; None disables the
    tools (they fail closed).
    """

    catalog_enabled: Callable[[], bool] | None = None
    resolve_asset: Callable[[int], Awaitable[MediaAsset | None]] | None = None


def media_segment_for_intent(intent: MediaReplyIntent) -> Segment:
    """The ``Outgoing`` segment carrying an opaque media intent.

    The segment data carries ONLY the opaque content-addressed cache key
    (``data["media"]["key"]``) — never a URL, file path, platform reference,
    or base64 payload. The App's send-time resolver maps the cache key to
    the normalized bytes (in memory) so the durable outbox never carries a
    fetchable source.
    """
    return Segment(
        kind=_SEGMENT_KIND[intent.kind],
        data={"media": {"key": intent.cache_key}},
    )


def catalog_prompt(assets: Sequence[MediaAsset]) -> str:
    """The prompt section listing the chat's approved catalog assets.

    Renders ONLY opaque asset ids and (normalized, escaped, one-line)
    descriptions — never URLs, file paths, platform references, or base64
    payloads. Descriptions are re-normalized defensively so a row written by
    any path can never break the listing structure. The wording allows an
    approved catalog selection via ``send_emoji`` / ``send_image`` and
    forbids every other media source.
    """
    stickers = [a for a in assets if a.kind == "sticker"]
    images = [a for a in assets if a.kind == "image"]
    parts: list[str] = []
    if stickers:
        lines = [
            "【可用表情】(已通过安全审核的收藏表情，只能通过 send_emoji 发送；"
            "禁止使用 URL、文件路径、平台引用或 base64)"
        ]
        for asset in stickers:
            desc = normalize_description(asset.description) or "表情"
            lines.append(f"- {asset.id}: {desc}")
        parts.append("\n".join(lines))
    if images:
        lines = [
            "【可用图片】(已通过安全审核的收藏图片，只能通过 send_image 发送；"
            "禁止使用 URL、文件路径、平台引用或 base64)"
        ]
        for asset in images:
            desc = normalize_description(asset.description) or "图片"
            lines.append(f"- {asset.id}: {desc}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


# ── tool handlers (unbound ToolContext methods; ``self`` is the live ctx) ────

async def send_emoji(self: Any, asset_id: int) -> str:
    """发送一个已通过安全审核的收藏表情（只能使用目录中给出的表情编号）。

    从已审核的收藏表情目录中选择一个表情编号发送。禁止使用 URL、文件路径、
    平台引用或 base64。与 reply/wait/no_action 互斥：第一个最终决定生效。
    """
    return await _stage_media(self, "send_emoji", asset_id)


async def send_image(self: Any, asset_id: int) -> str:
    """发送一张已通过安全审核的收藏图片（只能使用目录中给出的图片编号）。

    从已审核的收藏图片目录中选择一个图片编号发送。禁止使用 URL、文件路径、
    平台引用或 base64。与 reply/wait/no_action 互斥：第一个最终决定生效。
    """
    return await _stage_media(self, "send_image", asset_id)


async def _stage_media(ctx: Any, tool_name: str, asset_id: int) -> str:
    """Resolve the opaque approved asset and stage one ``MediaReplyIntent``.

    Fail-closed gates, in order: media callbacks present → catalog enabled →
    no conflicting terminal intent already staged → the asset resolves and
    is an APPROVED row of the expected kind. A conflict or a bad asset is a
    ``ToolError`` (the planner sees an ``ok=False`` result and can correct
    course). Nothing is written anywhere.
    """
    kind = _TOOL_KIND[tool_name]
    media = getattr(ctx, "_media", None)
    if media is None:
        raise ToolError("media catalog is not available")
    enabled = media.catalog_enabled
    if enabled is not None and not enabled():
        raise ToolError("media catalog is disabled")
    resolve = media.resolve_asset
    if resolve is None:
        raise ToolError("media catalog is not available")
    if (
        ctx.reply_text is not None
        or ctx.wait_seconds is not None
        or ctx.no_action_verdict
    ):
        raise ToolError(
            f"{tool_name} conflicts with an already staged terminal intent"
        )
    if ctx.media_intent is not None:
        raise ToolError(
            "a media send is already staged; the first terminal intent wins"
        )
    try:
        validate_asset_id(asset_id)
    except ValueError as exc:
        raise ToolError(f"invalid media asset: {exc}") from exc
    asset = await resolve(asset_id)
    if asset is None or asset.id is None:
        raise ToolError(f"unknown or unapproved media asset: {asset_id}")
    try:
        validate_asset_id(asset.id)
        validate_catalog_key(asset.cache_key)
    except ValueError as exc:
        # Tool arguments and staged intents carry opaque identifiers only;
        # malformed callback data is a contained tool failure, never a path,
        # URL, or encoded payload that reaches the outbox.
        raise ToolError(f"invalid media asset: {exc}") from exc
    expected = "sticker" if kind == "emoji" else "image"
    if asset.kind != expected:
        raise ToolError(f"asset {asset_id} is not a {expected}")
    if asset.safety_status != MediaSafetyStatus.APPROVED:
        raise ToolError(f"asset {asset_id} is not approved")
    ctx._media_intent = MediaReplyIntent(
        kind=kind, asset_id=asset.id, cache_key=asset.cache_key
    )
    # Sentinel reply marker: makes the frozen planner loop's ``reply``
    # terminal break fire so the round ends here. The cycle reads the typed
    # media_intent FIRST and never treats the marker as reply text.
    ctx._reply_text = MEDIA_REPLY_MARKER
    return json.dumps(
        {"action": tool_name, "asset_id": asset.id}, ensure_ascii=False
    )
