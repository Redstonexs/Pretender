"""Phase 6 P6.5 durable media catalog: chat-scoped candidates, capacity-safe
transactional approval/eviction, idempotent rejection/revocation,
deterministic cooldown-aware selection, atomic uses, opaque-key validation,
and cross-chat isolation — all over the v10 schema via the MediaRepository
surface."""

from __future__ import annotations

import asyncio

import pytest

from pretender.errors import RepoError
from pretender.types import (
    ChatKey,
    MediaAssetCandidate,
    MediaKind,
    MediaSafetyStatus,
    MessageRowId,
    SenderId,
)
from tests.durable_helpers import CK, make_identity, open_repo_with_chat, run

OTHER = ChatKey("qq:group:other")


def make_candidate(
    chat_key=CK,
    kind=MediaKind.STICKER,
    cache_key=None,
    sha256=None,
    mime="image/gif",
    description="smile",
    **kw,
) -> MediaAssetCandidate:
    """A valid candidate: opaque 64-hex cache key + content sha256."""
    if cache_key is None:
        cache_key = "c" * 64
    if sha256 is None:
        sha256 = "a" * 64
    return MediaAssetCandidate(
        chat_key=chat_key,
        kind=kind,
        cache_key=cache_key,
        sha256=sha256,
        mime=mime,
        width=120,
        height=120,
        description=description,
        source_message_id=MessageRowId(1),
        source_sender_id=SenderId("u1"),
        source_sender_name="alice",
        source_ts=100.0,
        **kw,
    )


async def submit(repo, candidate=None, *, now=200.0) -> int:
    return await repo.submit_media_candidate(candidate or make_candidate(), now=now)


async def approve(repo, candidate_id, *, capacity=4, now=300.0):
    return await repo.approve_media_candidate(CK, candidate_id, capacity=capacity, now=now)


# ── Candidate submit / read / list ──────────────────────────────────────────

def test_submit_and_get_candidate_roundtrip(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        cid = await submit(repo)
        got = await repo.get_media_candidate(CK, cid)
        await repo.close()
        return got

    got = run(scenario())
    assert got is not None
    assert got.id == 1
    assert got.chat_key == CK
    assert got.kind == MediaKind.STICKER
    assert got.cache_key == "c" * 64
    assert got.sha256 == "a" * 64
    assert got.mime == "image/gif"
    assert got.width == 120 and got.height == 120
    assert got.description == "smile"
    assert got.source_message_id == MessageRowId(1)
    assert got.source_sender_id == SenderId("u1")
    assert got.source_sender_name == "alice"
    assert got.source_ts == 100.0
    assert got.created_ts == 200.0


def test_submit_is_idempotent_per_chat_kind_sha256(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        cid1 = await submit(repo)
        # Same (chat, kind, sha256): the existing row id returns, no dup.
        cid2 = await submit(repo)
        # A different kind is a distinct row.
        cid3 = await submit(repo, make_candidate(kind=MediaKind.IMAGE))
        # A different chat is a distinct row.
        cid4 = await submit(repo, make_candidate(chat_key=OTHER))
        # A different sha256 is a distinct row.
        cid5 = await submit(repo, make_candidate(sha256="b" * 64))
        listed = await repo.list_media_candidates(CK)
        await repo.close()
        return cid1, cid2, cid3, cid4, cid5, listed

    cid1, cid2, cid3, cid4, cid5, listed = run(scenario())
    assert cid1 == cid2 == 1
    assert cid3 == 2
    assert cid4 == 3
    assert cid5 == 4
    assert len(listed) == 3  # CK only: sticker + image + different sha256
    assert [c.id for c in listed] == [1, 2, 4]


def test_submit_unknown_chat_raises(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        try:
            await submit(repo, make_candidate(chat_key=OTHER))
            raised = False
        except RepoError:
            raised = True
        await repo.close()
        return raised

    assert run(scenario()) is True


def test_list_media_candidates_filters_kind_and_bounds(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await submit(repo, make_candidate(sha256="a" * 64))
        await submit(repo, make_candidate(sha256="b" * 64))
        await submit(repo, make_candidate(sha256="c" * 64, kind=MediaKind.IMAGE))
        stickers = await repo.list_media_candidates(CK, kind=MediaKind.STICKER)
        bounded = await repo.list_media_candidates(CK, limit=2)
        await repo.close()
        return stickers, bounded

    stickers, bounded = run(scenario())
    assert len(stickers) == 2
    assert all(c.kind == MediaKind.STICKER for c in stickers)
    assert len(bounded) == 2


# ── Opaque catalog key validation ───────────────────────────────────────────

@pytest.mark.parametrize(
    "bad_key",
    [
        "/abs/path/sticker.gif",          # absolute local path
        "./rel/sticker.gif",              # relative local path
        "../up/sticker.gif",              # parent-relative path
        "~/sticker.gif",                  # home path
        "C:\\stickers\\a.gif",            # windows path
        "http://example.com/a.gif",       # URL
        "https://example.com/a.gif",      # URL
        "file:///tmp/a.gif",              # file URL
        "data:image/gif;base64,R0lGOD",   # data URL
        "base64://R0lGODlh",              # raw base64 reference
        "{file=sticker.gif}",             # raw platform media reference
        "{url=https://example.com/a.gif}",  # raw platform URL reference
        "not-a-key",                      # arbitrary non-opaque token
        "A" * 64,                         # uppercase hex is not the opaque form
        "c" * 63,                         # wrong length
        "",                               # empty
    ],
)
def test_submit_rejects_invalid_catalog_keys(tmp_path, bad_key):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        try:
            await submit(repo, make_candidate(cache_key=bad_key))
            raised = False
        except ValueError:
            raised = True
        await repo.close()
        return raised

    assert run(scenario()) is True, f"key {bad_key!r} must be rejected"


def test_submit_rejects_invalid_sha256_and_mime(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        results = []
        for bad_sha in ("http://x", ""):
            try:
                await submit(repo, make_candidate(sha256=bad_sha))
                results.append(False)
            except ValueError:
                results.append(True)
        for bad_mime in ("", "  "):
            try:
                await submit(repo, make_candidate(mime=bad_mime))
                results.append(False)
            except ValueError:
                results.append(True)
        await repo.close()
        return results

    assert run(scenario()) == [True, True, True, True]


@pytest.mark.parametrize(
    "description",
    [
        "描述\n注入",
        "查看 https://example.com/a.gif",
        "文件 /tmp/a.gif",
        "data:image/gif;base64,R0lGODlh",
    ],
)
def test_submit_rejects_unescaped_or_source_descriptions(tmp_path, description):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        try:
            await submit(repo, make_candidate(description=description))
        except ValueError:
            return True
        finally:
            await repo.close()
        return False

    assert run(scenario()) is True


# ── Approval: capacity-safe transactional transition ────────────────────────

def test_approve_transitions_pending_to_approved(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        cid = await submit(repo)
        asset = await approve(repo, cid)
        # The candidate is no longer pending.
        assert await repo.get_media_candidate(CK, cid) is None
        # Re-approval of the already-approved row is idempotent.
        again = await approve(repo, cid)
        await repo.close()
        return asset, again

    asset, again = run(scenario())
    assert asset is not None
    assert asset.id == 1
    assert asset.safety_status == MediaSafetyStatus.APPROVED
    assert asset.safety_version == 1
    assert asset.approved_ts == 300.0
    assert asset.uses == 0
    assert again is not None and again.safety_version == 1  # no double bump


def test_approve_evicts_least_recently_used_at_capacity(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        ids = []
        for i in range(3):
            ids.append(await submit(repo, make_candidate(sha256=f"{i:064x}")))
        # Capacity 2: approving the third evicts the least-recently-used
        # approved row (never-used first, then oldest last_used_ts, then id).
        for cid in ids:
            await approve(repo, cid, capacity=2)
        assets = await repo.list_media_assets(CK)
        await repo.close()
        return assets

    assets = run(scenario())
    # Exactly 2 approved rows survive; the first-approved (id 1) was evicted.
    assert [a.id for a in assets] == [2, 3]
    assert all(a.safety_status == MediaSafetyStatus.APPROVED for a in assets)


def test_approve_eviction_prefers_never_used_then_oldest_used(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        ids = []
        for i in range(3):
            ids.append(await submit(repo, make_candidate(sha256=f"{i:064x}")))
        for cid in ids:
            await approve(repo, cid, capacity=3)
        # Use asset 1 (oldest) and asset 3 (newest); asset 2 stays never-used.
        await repo.use_media_asset(CK, ids[0], now=400.0)
        await repo.use_media_asset(CK, ids[2], now=410.0)
        # Capacity 3 with 3 approved: approving a 4th evicts exactly one —
        # the never-used asset 2 (never-used evicts before oldest-used).
        await approve(repo, await submit(repo, make_candidate(sha256="f" * 64)), capacity=3)
        assets = await repo.list_media_assets(CK)
        await repo.close()
        return assets

    assets = run(scenario())
    assert [a.id for a in assets] == [1, 3, 4]
    assert all(a.safety_status == MediaSafetyStatus.APPROVED for a in assets)


def test_approve_eviction_race_is_transactionally_safe(tmp_path):
    """Two concurrent approvals at capacity serialize on the single writer:
    the final approved count never exceeds capacity and every approval
    succeeds (each evicts to make room)."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        ids = []
        for i in range(3):
            ids.append(await submit(repo, make_candidate(sha256=f"{i:064x}")))
        # Fire all three approvals concurrently at capacity 1.
        results = await asyncio.gather(
            *(approve(repo, cid, capacity=1) for cid in ids)
        )
        assets = await repo.list_media_assets(CK)
        await repo.close()
        return results, assets

    results, assets = run(scenario())
    assert all(r is not None for r in results)  # every approval landed
    assert len(assets) == 1  # capacity never exceeded
    assert assets[0].safety_status == MediaSafetyStatus.APPROVED


def test_approve_rejected_or_revoked_returns_none(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        cid = await submit(repo)
        assert await repo.reject_media_candidate(CK, cid) is True
        assert await approve(repo, cid) is None  # rejected: fail closed
        cid2 = await submit(repo, make_candidate(sha256="b" * 64))
        asset = await approve(repo, cid2)
        assert asset is not None
        assert await repo.revoke_media_asset(CK, asset.id, now=400.0) is True
        assert await approve(repo, cid2) is None  # revoked: fail closed
        await repo.close()

    run(scenario())


def test_approve_unknown_or_cross_chat_returns_none(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        cid = await submit(repo)
        assert await approve(repo, 999) is None
        assert await repo.approve_media_candidate(OTHER, cid, capacity=4, now=300.0) is None
        await repo.close()

    run(scenario())


# ── Rejection / revocation: idempotent terminal transitions ─────────────────

def test_reject_and_revoke_are_idempotent(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        cid = await submit(repo)
        assert await repo.reject_media_candidate(CK, cid) is True
        assert await repo.reject_media_candidate(CK, cid) is False  # already rejected
        assert await repo.reject_media_candidate(OTHER, cid) is False  # cross-chat
        cid2 = await submit(repo, make_candidate(sha256="b" * 64))
        asset = await approve(repo, cid2)
        assert asset is not None
        assert await repo.revoke_media_asset(CK, asset.id, now=400.0) is True
        assert await repo.revoke_media_asset(CK, asset.id, now=401.0) is False
        assert await repo.revoke_media_asset(OTHER, asset.id, now=401.0) is False
        # A revoked asset is never selectable.
        assert await repo.select_media_assets(CK, MediaKind.STICKER, limit=5, cooldown_s=0.0, now=500.0) == []
        await repo.close()

    run(scenario())


# ── Selection: approved-only, deterministic, cooldown aware ─────────────────

def test_selection_is_approved_only_and_deterministic(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        pending = await submit(repo, make_candidate(sha256="a" * 64))
        approved = await submit(repo, make_candidate(sha256="b" * 64))
        rejected = await submit(repo, make_candidate(sha256="c" * 64))
        revoked = await submit(repo, make_candidate(sha256="d" * 64))
        await approve(repo, approved)
        await repo.reject_media_candidate(CK, rejected)
        rv = await approve(repo, revoked)
        assert rv is not None
        await repo.revoke_media_asset(CK, rv.id, now=400.0)
        # Only the approved row is selectable.
        selected = await repo.select_media_assets(CK, MediaKind.STICKER, limit=10, cooldown_s=0.0, now=500.0)
        # Deterministic: repeated selection returns the same order.
        selected2 = await repo.select_media_assets(CK, MediaKind.STICKER, limit=10, cooldown_s=0.0, now=500.0)
        await repo.close()
        return pending, approved, selected, selected2

    pending, approved, selected, selected2 = run(scenario())
    assert [a.id for a in selected] == [approved]
    assert [a.id for a in selected2] == [approved]
    assert pending not in [a.id for a in selected]


def test_selection_orders_least_used_first(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        ids = []
        for i in range(3):
            ids.append(await submit(repo, make_candidate(sha256=f"{i:064x}")))
        for cid in ids:
            await approve(repo, cid, capacity=3)
        # Use asset 1 twice, asset 2 once, asset 3 never.
        await repo.use_media_asset(CK, ids[0], now=400.0)
        await repo.use_media_asset(CK, ids[0], now=401.0)
        await repo.use_media_asset(CK, ids[1], now=402.0)
        selected = await repo.select_media_assets(CK, MediaKind.STICKER, limit=3, cooldown_s=0.0, now=500.0)
        await repo.close()
        return [a.id for a in selected]

    assert run(scenario()) == [3, 2, 1]  # least-used first, then id


def test_selection_is_cooldown_aware(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        cid = await submit(repo)
        asset = await approve(repo, cid)
        assert asset is not None
        # Before any use: selectable.
        assert len(await repo.select_media_assets(CK, MediaKind.STICKER, limit=5, cooldown_s=1000.0, now=400.0)) == 1
        await repo.use_media_asset(CK, asset.id, now=400.0)
        # Within cooldown: excluded.
        assert await repo.select_media_assets(CK, MediaKind.STICKER, limit=5, cooldown_s=1000.0, now=500.0) == []
        # Cooldown expired: selectable again.
        again = await repo.select_media_assets(CK, MediaKind.STICKER, limit=5, cooldown_s=1000.0, now=1401.0)
        await repo.close()
        return again

    again = run(scenario())
    assert [a.id for a in again] == [1]


# ── Use: atomic idempotent bump ─────────────────────────────────────────────

def test_use_bumps_uses_and_sets_last_used(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        cid = await submit(repo)
        asset = await approve(repo, cid)
        assert asset is not None
        assert await repo.use_media_asset(CK, asset.id, now=400.0) is True
        assert await repo.use_media_asset(CK, asset.id, now=401.0) is True
        # Unknown / cross-chat / not-approved rows are False, never an error.
        assert await repo.use_media_asset(CK, 999, now=402.0) is False
        assert await repo.use_media_asset(OTHER, asset.id, now=402.0) is False
        pending = await submit(repo, make_candidate(sha256="b" * 64))
        assert await repo.use_media_asset(CK, pending, now=402.0) is False
        assets = await repo.list_media_assets(CK)
        await repo.close()
        return assets

    assets = run(scenario())
    approved = [a for a in assets if a.safety_status == MediaSafetyStatus.APPROVED]
    assert len(approved) == 1
    assert approved[0].uses == 2
    assert approved[0].last_used_ts == 401.0


# ── Cross-chat isolation ────────────────────────────────────────────────────

def test_cross_chat_isolation(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        cid = await submit(repo)
        other_cid = await submit(repo, make_candidate(chat_key=OTHER, sha256="b" * 64))
        await approve(repo, cid)
        await repo.approve_media_candidate(OTHER, other_cid, capacity=4, now=300.0)
        # The same sha256 in two chats is two independent rows.
        dup = await submit(repo, make_candidate(chat_key=OTHER, sha256="a" * 64))
        # Each chat sees only its own rows.
        ck_assets = await repo.list_media_assets(CK)
        other_assets = await repo.list_media_assets(OTHER)
        ck_sel = await repo.select_media_assets(CK, MediaKind.STICKER, limit=5, cooldown_s=0.0, now=400.0)
        other_sel = await repo.select_media_assets(OTHER, MediaKind.STICKER, limit=5, cooldown_s=0.0, now=400.0)
        await repo.close()
        return ck_assets, other_assets, ck_sel, other_sel, dup

    ck_assets, other_assets, ck_sel, other_sel, dup = run(scenario())
    assert [a.id for a in ck_assets] == [1]
    assert [a.id for a in other_assets] == [2, 3]
    assert [a.id for a in ck_sel] == [1]
    assert [a.id for a in other_sel] == [2]
    assert dup == 3  # OTHER's own row, not CK's row 1


# ── Config-driven bounds (protocol separation is covered in test_seams) ─────

def test_approve_and_select_validate_bounds(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        cid = await submit(repo)
        results = []
        for bad in (0, -1, True):
            try:
                await approve(repo, cid, capacity=bad)
                results.append(False)
            except ValueError:
                results.append(True)
        for bad in (-1.0, float("nan"), True):
            try:
                await repo.select_media_assets(
                    CK, MediaKind.STICKER, limit=1, cooldown_s=bad, now=300.0
                )
                results.append(False)
            except ValueError:
                results.append(True)
        await repo.close()
        return results

    assert run(scenario()) == [True, True, True, True, True, True]
