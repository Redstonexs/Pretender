"""Phase 5 person service: per-chat person identity over KnowledgeRepository.

Pure/durable ``PersonService``: idempotent alias observation, deterministic
normalize/dedupe/bound, safe profile retrieval, and CAS profile/impression
application with a caller-provided fixed through-message cursor. No global
nickname matching, no LLM/network, no implicit source scan.

Service-logic tests run against a protocol-only ``FakeKnowledgeRepo`` (proving
the service depends only on the seam); the durable CAS semantics (valid
winner / stale loser / unknown person) are exercised against the real
``SqliteRepository``.
"""

from __future__ import annotations

import dataclasses

import pytest

from pretender.errors import RepoError
from pretender.person import MAX_ALIASES, PersonService
from pretender.types import ChatKey, MessageRowId, PersonProfile, SenderId
from tests.durable_helpers import CK, open_repo_with_chat, run
from tests.knowledge_helpers import make_person

OTHER = ChatKey("qq:group:other")


class FakeKnowledgeRepo:
    """A protocol-only KnowledgeRepository fake: proves PersonService depends
    only on the seam, never on SqliteRepository. No LLM/network anywhere."""

    def __init__(self) -> None:
        self.persons: dict[tuple[ChatKey, SenderId], PersonProfile] = {}
        self.calls: list[tuple] = []
        self.cas_result = True

    async def get_person(self, chat_key, platform_uid):
        self.calls.append(("get_person", chat_key, platform_uid))
        return self.persons.get((chat_key, platform_uid))

    async def upsert_person(self, profile):
        self.calls.append(("upsert_person", profile))
        self.persons[(profile.chat_key, profile.platform_uid)] = profile

    async def add_person_alias(self, chat_key, platform_uid, name, *, now=None):
        self.calls.append(("add_person_alias", chat_key, platform_uid, name, now))
        existing = self.persons.get((chat_key, platform_uid))
        if existing is None:
            profile = PersonProfile(
                chat_key=chat_key, platform_uid=platform_uid,
                names=(name,), updated_ts=now,
            )
            self.persons[(chat_key, platform_uid)] = profile
            return (name,)
        if name in existing.names or len(existing.names) >= MAX_ALIASES:
            return existing.names
        new_names = existing.names + (name,)
        self.persons[(chat_key, platform_uid)] = dataclasses.replace(
            existing, names=new_names, updated_ts=now
        )
        return new_names

    async def cas_person_profile(
        self, chat_key, platform_uid, expected_through_msg_id, profile
    ):
        self.calls.append(
            ("cas_person_profile", chat_key, platform_uid, expected_through_msg_id, profile)
        )
        return self.cas_result


def make_service(repo=None) -> PersonService:
    return PersonService(repo or FakeKnowledgeRepo())


# ── Observe: alias merge, scope, idempotency, cap/order ─────────────────────

def test_observe_merges_aliases_within_chat_only():
    async def scenario():
        repo = FakeKnowledgeRepo()
        svc = PersonService(repo)
        r1 = await svc.observe(CK, SenderId("u1"), "alice")
        r2 = await svc.observe(CK, SenderId("u1"), "小爱")
        # The same platform uid in another chat is a DIFFERENT person — no
        # global nickname matching.
        r3 = await svc.observe(OTHER, SenderId("u1"), "alice")
        return r1, r2, r3, repo.persons

    r1, r2, r3, persons = run(scenario())
    assert r1.names == ("alice",)
    assert r2.names == ("alice", "小爱")
    assert r3.names == ("alice",)
    assert persons[(CK, SenderId("u1"))].names == ("alice", "小爱")
    assert persons[(OTHER, SenderId("u1"))].names == ("alice",)


def test_observe_same_nickname_distinct_uids():
    async def scenario():
        repo = FakeKnowledgeRepo()
        svc = PersonService(repo)
        await svc.observe(CK, SenderId("u1"), "alice")
        await svc.observe(CK, SenderId("u2"), "alice")
        return repo.persons

    persons = run(scenario())
    assert persons[(CK, SenderId("u1"))].names == ("alice",)
    assert persons[(CK, SenderId("u2"))].names == ("alice",)
    assert len(persons) == 2  # same nickname never merges distinct UIDs


def test_observe_idempotent():
    async def scenario():
        repo = FakeKnowledgeRepo()
        svc = PersonService(repo)
        await svc.observe(CK, SenderId("u1"), "alice")
        r2 = await svc.observe(CK, SenderId("u1"), "alice")
        return r2, repo.persons[(CK, SenderId("u1"))].names

    r2, names = run(scenario())
    assert r2.added is False
    assert r2.created is False
    assert names == ("alice",)  # no duplicate alias


def test_observe_alias_cap_and_order():
    async def scenario():
        repo = FakeKnowledgeRepo()
        svc = PersonService(repo)
        for i in range(MAX_ALIASES + 3):
            await svc.observe(CK, SenderId("u1"), f"name{i}")
        return repo.persons[(CK, SenderId("u1"))].names

    names = run(scenario())
    # First-seen order preserved; bounded to MAX_ALIASES (new names dropped).
    assert len(names) == MAX_ALIASES
    assert names == tuple(f"name{i}" for i in range(MAX_ALIASES))


def test_observe_normalizes_whitespace():
    async def scenario():
        repo = FakeKnowledgeRepo()
        svc = PersonService(repo)
        await svc.observe(CK, SenderId("u1"), "  Alice   Smith  ")
        return repo.persons[(CK, SenderId("u1"))].names

    names = run(scenario())
    assert names == ("Alice Smith",)


def test_observe_blank_name_noop():
    async def scenario():
        repo = FakeKnowledgeRepo()
        svc = PersonService(repo)
        r = await svc.observe(CK, SenderId("u1"), "   ")
        return r, repo.persons

    r, persons = run(scenario())
    assert r.added is False
    assert r.created is False
    assert persons == {}  # blank names never create/update a person


def test_observe_preserves_profile_and_impression():
    async def scenario():
        repo = FakeKnowledgeRepo()
        repo.persons[(CK, SenderId("u1"))] = make_person(profile="p", impression="i")
        svc = PersonService(repo)
        await svc.observe(CK, SenderId("u1"), "bob")
        p = repo.persons[(CK, SenderId("u1"))]
        return p.profile, p.impression, p.names

    profile, impression, names = run(scenario())
    assert profile == "p"
    assert impression == "i"
    assert names == ("bob",)


# ── Retrieve profile safely ─────────────────────────────────────────────────

def test_get_profile_unknown_returns_none():
    async def scenario():
        svc = make_service()
        return await svc.get_profile(CK, SenderId("nobody"))

    assert run(scenario()) is None


def test_get_profile_returns_stored_profile():
    async def scenario():
        repo = FakeKnowledgeRepo()
        repo.persons[(CK, SenderId("u1"))] = make_person(profile="likes tea")
        svc = PersonService(repo)
        return await svc.get_profile(CK, SenderId("u1"))

    p = run(scenario())
    assert p is not None
    assert p.profile == "likes tea"


# ── Apply profile/impression via CAS ────────────────────────────────────────

def test_apply_profile_valid_winner(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_person(make_person(names=("alice",)))
        svc = PersonService(repo)
        res = await svc.apply_profile(
            CK, SenderId("u1"),
            update=make_person(profile="likes tea", impression="friendly"),
            through_msg_id=MessageRowId(5),
            expected_through_msg_id=None,
            now=100.0,
        )
        assert res.applied is True
        assert res.reason is None
        assert res.profile == "likes tea"
        assert res.impression == "friendly"
        assert res.through_msg_id == MessageRowId(5)
        p = await repo.get_person(CK, SenderId("u1"))
        assert p.profile == "likes tea"
        assert p.impression == "friendly"
        assert p.profile_through_msg_id == MessageRowId(5)
        assert p.names == ("alice",)  # aliases preserved
        await repo.close()

    run(scenario())


def test_apply_profile_stale_cas_loser(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_person(make_person())
        svc = PersonService(repo)
        # First apply wins and advances the cursor.
        assert (
            await svc.apply_profile(
                CK, SenderId("u1"),
                update=make_person(profile="p1"),
                through_msg_id=MessageRowId(5),
                expected_through_msg_id=None,
            )
        ).applied is True
        # A stale retry (expected 0, actual 5) loses and changes nothing.
        res = await svc.apply_profile(
            CK, SenderId("u1"),
            update=make_person(profile="p2"),
            through_msg_id=MessageRowId(9),
            expected_through_msg_id=None,
        )
        assert res.applied is False
        assert res.reason == "stale_cas"
        assert res.profile == "p1"  # unchanged stored state
        assert res.through_msg_id == MessageRowId(5)
        p = await repo.get_person(CK, SenderId("u1"))
        assert p.profile == "p1"
        assert p.profile_through_msg_id == MessageRowId(5)
        await repo.close()

    run(scenario())


def test_apply_profile_unknown_person(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        svc = PersonService(repo)
        res = await svc.apply_profile(
            CK, SenderId("nobody"),
            update=make_person(uid="nobody", profile="x"),
            through_msg_id=MessageRowId(1),
            expected_through_msg_id=None,
        )
        await repo.close()
        return res

    res = run(scenario())
    assert res.applied is False
    assert res.reason == "unknown_person"


def test_apply_profile_cross_chat_mismatch_raises():
    async def scenario():
        svc = make_service()
        with pytest.raises(ValueError):
            await svc.apply_profile(
                CK, SenderId("u1"),
                update=make_person(chat_key=OTHER, uid="u1", profile="x"),
                through_msg_id=MessageRowId(1),
                expected_through_msg_id=None,
            )

    run(scenario())


def test_apply_profile_nonfinite_now_raises():
    async def scenario():
        svc = make_service()
        with pytest.raises(ValueError):
            await svc.apply_profile(
                CK, SenderId("u1"),
                update=make_person(profile="x"),
                through_msg_id=MessageRowId(1),
                expected_through_msg_id=None,
                now=float("inf"),
            )

    run(scenario())


def test_apply_profile_passes_expected_cursor_to_cas():
    async def scenario():
        repo = FakeKnowledgeRepo()
        repo.persons[(CK, SenderId("u1"))] = make_person()
        svc = PersonService(repo)
        await svc.apply_profile(
            CK, SenderId("u1"),
            update=make_person(profile="p"),
            through_msg_id=MessageRowId(7),
            expected_through_msg_id=MessageRowId(3),
        )
        cas = [c for c in repo.calls if c[0] == "cas_person_profile"]
        assert len(cas) == 1
        _, ck, uid, expected, profile = cas[0]
        assert ck == CK
        assert uid == SenderId("u1")
        assert expected == MessageRowId(3)
        assert profile.profile_through_msg_id == MessageRowId(7)

    run(scenario())


# ── Gate 5: alias/profile race, cursor no-regression, cross-chat CAS ─────────

def test_alias_and_profile_race_preserves_both_fields(tmp_path):
    """A concurrent alias observation and profile CAS never erase each
    other: the atomic alias-only operation preserves profile/impression, and
    the profile CAS preserves aliases."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        svc = PersonService(repo)
        await svc.observe(CK, SenderId("u1"), "alice")
        # Profile CAS wins first.
        assert (
            await svc.apply_profile(
                CK, SenderId("u1"),
                update=make_person(profile="p1", impression="i1"),
                through_msg_id=MessageRowId(5),
                expected_through_msg_id=None,
            )
        ).applied is True
        # A concurrent alias observation merges WITHOUT touching the profile.
        r = await svc.observe(CK, SenderId("u1"), "bob")
        p = await repo.get_person(CK, SenderId("u1"))
        await repo.close()
        return r, p

    r, p = run(scenario())
    assert r.names == ("alice", "bob")
    assert p.profile == "p1"
    assert p.impression == "i1"
    assert p.profile_through_msg_id == MessageRowId(5)


def test_profile_cas_cannot_regress_cursor(tmp_path):
    """The profile CAS never regresses the durable cursor: a write with a
    cursor below the stored one fails even with the right expected fence."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_person(make_person())
        assert await repo.cas_person_profile(
            CK, SenderId("u1"), None,
            make_person(profile="p1", profile_through_msg_id=MessageRowId(5)),
        ) is True
        # A regressed cursor (3 < 5) is rejected.
        ok = await repo.cas_person_profile(
            CK, SenderId("u1"), MessageRowId(5),
            make_person(profile="p2", profile_through_msg_id=MessageRowId(3)),
        )
        p = await repo.get_person(CK, SenderId("u1"))
        await repo.close()
        return ok, p

    ok, p = run(scenario())
    assert ok is False
    assert p.profile == "p1"
    assert p.profile_through_msg_id == MessageRowId(5)


def test_profile_cas_rejects_cross_chat_profile(tmp_path):
    """The profile CAS validates the profile's chat+UID: a cross-chat
    profile is rejected before any write."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_person(make_person())
        try:
            await repo.cas_person_profile(
                CK, SenderId("u1"), None,
                make_person(chat_key=OTHER, uid="u1", profile="x",
                            profile_through_msg_id=MessageRowId(1)),
            )
            raised = False
        except RepoError:
            raised = True
        p = await repo.get_person(CK, SenderId("u1"))
        await repo.close()
        return raised, p

    raised, p = run(scenario())
    assert raised is True
    assert p.profile is None  # nothing written


def test_profile_cas_never_overwrites_names_json(tmp_path):
    """The profile CAS never writes names_json: an alias merged between the
    profile read and the CAS survives (the CAS cannot overwrite it)."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_person(make_person(names=("alice",)))
        # An alias lands between the profile read and the CAS.
        await repo.add_person_alias(CK, SenderId("u1"), "bob")
        # The CAS carries a STALE alias list (only "alice") — it must not
        # overwrite the stored ("alice", "bob").
        ok = await repo.cas_person_profile(
            CK, SenderId("u1"), None,
            make_person(names=("alice",), profile="p1",
                        profile_through_msg_id=MessageRowId(5)),
        )
        p = await repo.get_person(CK, SenderId("u1"))
        await repo.close()
        return ok, p

    ok, p = run(scenario())
    assert ok is True
    assert p.profile == "p1"
    assert p.profile_through_msg_id == MessageRowId(5)
    assert p.names == ("alice", "bob")  # the alias survived the CAS


def test_apply_profile_stale_rereads_winner(tmp_path):
    """A stale profile CAS rereads and returns the WINNER's fields — never
    the loser's, and never erasing the winner's alias/profile fields."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_person(make_person(names=("alice",)))
        svc = PersonService(repo)
        # Winner applies profile p1 at cursor 5.
        assert (
            await svc.apply_profile(
                CK, SenderId("u1"),
                update=make_person(profile="p1"),
                through_msg_id=MessageRowId(5),
                expected_through_msg_id=None,
            )
        ).applied is True
        # A concurrent alias lands between the loser's read and its CAS.
        await svc.observe(CK, SenderId("u1"), "bob")
        # Loser retries with a stale expected cursor.
        res = await svc.apply_profile(
            CK, SenderId("u1"),
            update=make_person(profile="loser"),
            through_msg_id=MessageRowId(9),
            expected_through_msg_id=None,
        )
        p = await repo.get_person(CK, SenderId("u1"))
        await repo.close()
        return res, p

    res, p = run(scenario())
    assert res.applied is False
    assert res.reason == "stale_cas"
    assert res.profile == "p1"  # the winner's profile, not the loser's
    assert p.profile == "p1"
    assert p.names == ("alice", "bob")  # winner's aliases preserved
    assert p.profile_through_msg_id == MessageRowId(5)


# ── Protocol-only, no LLM ───────────────────────────────────────────────────

def test_service_uses_only_knowledge_seam_no_llm():
    async def scenario():
        repo = FakeKnowledgeRepo()
        svc = PersonService(repo)
        await svc.observe(CK, SenderId("u1"), "alice")
        await svc.get_profile(CK, SenderId("u1"))
        await svc.apply_profile(
            CK, SenderId("u1"),
            update=make_person(profile="p"),
            through_msg_id=MessageRowId(1),
            expected_through_msg_id=None,
        )
        return repo.calls

    calls = run(scenario())
    names = {c[0] for c in calls}
    # Only the KnowledgeRepository person methods — never an LLM,
    # never a source scan, never a SqliteRepository-only call.
    assert names <= {"get_person", "add_person_alias", "cas_person_profile"}
