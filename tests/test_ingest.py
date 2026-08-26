"""Ingest: record -> commit -> wake ordering, duplicate suppression,
identity resolution, and trusted-key self-echo reconciliation. Ingest is
typed against the Repository seam, so it must work against a protocol-only
fake."""

from __future__ import annotations

import json

from pretender.ingest import Ingest
from pretender.person import PersonService
from pretender.record import Recorder
from pretender.seams import Repository
from pretender.types import (
    AdapterEvent,
    ChatKey,
    EchoStatus,
    IngestResult,
    MediaAssetCandidate,
    MessageRowId,
    SenderId,
    WakeKind,
)
from tests.durable_helpers import (
    CK,
    FakeRepo,
    make_identity,
    make_message,
    open_repo,
    run,
)


def make_ingest(repo, recorder_path, *, wake=None, identity=None, delivery_key=None, harvest_media=None):
    return Ingest(
        repo,
        Recorder(recorder_path),
        wake=wake,
        identity=identity or (lambda chat_key: make_identity()),
        delivery_key=delivery_key,
        harvest_media=harvest_media,
    )


def test_commit_before_wake(tmp_path):
    """The wake callback must observe the message already durably committed
    on a separate read connection."""

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        seen: list[bool] = []

        async def wake(chat_key: ChatKey) -> None:
            # A fresh read connection sees the committed row.
            msg = await repo.get_message(CK, "m1")
            seen.append(msg is not None and msg.text == "hello")

        ingest = make_ingest(repo, tmp_path / "events.jsonl", wake=wake)
        event = AdapterEvent(type="message", payload=make_message(), ts=1.0)
        result = await ingest.handle(event)
        await repo.close()
        return result, seen

    result, seen = run(scenario())
    assert result.inserted is True
    assert result.row_id == MessageRowId(1)
    assert result.echo_status == EchoStatus.NOT_APPLICABLE
    assert seen == [True]


def test_duplicate_input_never_wakes(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        wakes: list[ChatKey] = []

        async def wake(chat_key: ChatKey) -> None:
            wakes.append(chat_key)

        ingest = make_ingest(repo, tmp_path / "events.jsonl", wake=wake)
        event = AdapterEvent(type="message", payload=make_message(), ts=1.0)
        first = await ingest.handle(event)
        second = await ingest.handle(event)  # same platform id
        count = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        )
        await repo.close()
        return first, second, wakes, count

    first, second, wakes, count = run(scenario())
    assert first.inserted is True and second.inserted is False
    assert wakes == [CK]
    assert count == 1


def test_ingest_commits_identity_and_message(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        ingest = make_ingest(repo, tmp_path / "events.jsonl")
        await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=1.0)
        )
        chat = await repo.get_chat(CK)
        msg = await repo.get_message(CK, "m1")
        await repo.close()
        return chat, msg

    chat, msg = run(scenario())
    assert chat is not None and chat.platform == "qq"
    assert msg is not None and msg.row_id == 1


def test_ingest_records_before_commit(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        ingest = make_ingest(repo, tmp_path / "events.jsonl")
        await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=1.0)
        )
        await repo.close()
        return json.loads(
            (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )

    line = run(scenario())
    assert line["type"] == "message"
    assert line["text"] == "hello"
    assert line["id"] == "m1"


def test_unknown_chat_is_recorded_but_not_committed(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        wakes: list[ChatKey] = []

        async def wake(chat_key: ChatKey) -> None:
            wakes.append(chat_key)

        ingest = make_ingest(
            repo, tmp_path / "events.jsonl", wake=wake, identity=lambda ck: None
        )
        result = await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=1.0)
        )
        chat = await repo.get_chat(CK)
        lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        await repo.close()
        return result, chat, wakes, lines

    result, chat, wakes, lines = run(scenario())
    assert result.inserted is False
    assert result.row_id is None  # nothing committed
    assert chat is None  # nothing committed
    assert wakes == []
    assert len(lines) == 1  # but the corpus recorded it


def test_non_message_event_is_recorded_without_wake(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        wakes: list[ChatKey] = []

        async def wake(chat_key: ChatKey) -> None:
            wakes.append(chat_key)

        ingest = make_ingest(repo, tmp_path / "events.jsonl", wake=wake)
        result = await ingest.handle(
            AdapterEvent(type="notice", payload={"kind": "poke"}, ts=1.0)
        )
        await repo.close()
        return result, wakes

    result, wakes = run(scenario())
    assert result.inserted is False
    assert result.echo_status == EchoStatus.NOT_APPLICABLE
    assert wakes == []


def test_self_message_is_ingested_and_deduped(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        wakes: list[ChatKey] = []

        async def wake(chat_key: ChatKey) -> None:
            wakes.append(chat_key)

        ingest = make_ingest(repo, tmp_path / "events.jsonl", wake=wake)
        event = AdapterEvent(
            type="message",
            payload=make_message(msg_id="console:out:1", is_self=True),
            ts=1.0,
        )
        first = await ingest.handle(event)
        second = await ingest.handle(event)
        await repo.close()
        return first, second, wakes

    first, second, wakes = run(scenario())
    assert first.inserted is True and second.inserted is False
    # Self echoes are durable context/presence records, never scheduler wake
    # events — the public callback must match the ledger/App invariant.
    assert wakes == []


def test_ingest_fills_missing_recv_ts_from_clock(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        ingest = make_ingest(repo, tmp_path / "events.jsonl")
        msg = make_message(recv_ts=None)
        await ingest.handle(AdapterEvent(type="message", payload=msg, ts=None))
        stored = await repo.get_message(CK, "m1")
        await repo.close()
        return stored

    stored = run(scenario())
    assert stored is not None and stored.recv_ts is not None


# ── protocol-only dependency ────────────────────────────────────────────────

def test_ingest_works_against_protocol_only_fake(tmp_path):
    """Ingest must depend only on the Repository seam: a protocol-complete
    fake (no SqliteRepository anywhere) drives the full handle() path."""

    async def scenario():
        fake = FakeRepo()
        assert isinstance(fake, Repository)
        fake.ingest_result = IngestResult(
            row_id=MessageRowId(7), inserted=True, wake_kind=WakeKind.INBOUND
        )
        wakes: list[ChatKey] = []

        async def wake(chat_key: ChatKey) -> None:
            wakes.append(chat_key)

        ingest = Ingest(
            fake,
            Recorder(tmp_path / "events.jsonl"),
            wake=wake,
            identity=lambda ck: make_identity(),
        )
        result = await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=1.0)
        )
        return result, wakes, fake.calls

    result, wakes, calls = run(scenario())
    assert result.inserted is True
    assert result.row_id == MessageRowId(7)
    assert wakes == [CK]
    kinds = [c[0] for c in calls]
    assert kinds == ["ingest_message"]  # identity+message committed atomically
    # No delivery key resolver: a non-self message passes no trusted key.
    assert calls[0][3] is None


def test_ingest_duplicate_against_fake_never_wakes(tmp_path):
    async def scenario():
        fake = FakeRepo()
        fake.ingest_result = IngestResult(row_id=MessageRowId(1), inserted=False)
        wakes: list[ChatKey] = []

        async def wake(chat_key: ChatKey) -> None:
            wakes.append(chat_key)

        ingest = Ingest(
            fake,
            Recorder(tmp_path / "events.jsonl"),
            wake=wake,
            identity=lambda ck: make_identity(),
        )
        result = await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=1.0)
        )
        return result, wakes

    result, wakes = run(scenario())
    assert result.inserted is False
    assert wakes == []


# ── pending count surfacing: verbatim, no extra wake ────────────────────────

def test_ingest_surfaces_pending_count_without_extra_wake(tmp_path):
    """The wrapper surfaces the repository's pending_count verbatim; the
    wake rule is unchanged (exactly one wake for the inserted message)."""

    async def scenario():
        fake = FakeRepo()
        fake.ingest_result = IngestResult(
            row_id=MessageRowId(3),
            inserted=True,
            pending_count=3,
            wake_kind=WakeKind.INBOUND,
        )
        wakes: list[ChatKey] = []

        async def wake(chat_key: ChatKey) -> None:
            wakes.append(chat_key)

        ingest = Ingest(
            fake,
            Recorder(tmp_path / "events.jsonl"),
            wake=wake,
            identity=lambda ck: make_identity(),
        )
        result = await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=1.0)
        )
        return result, wakes

    result, wakes = run(scenario())
    assert result.pending_count == 3
    assert wakes == [CK]  # exactly one wake, unchanged


def test_ingest_duplicate_surfaces_none_pending_count(tmp_path):
    """A duplicate (inserted=False) surfaces pending_count None and never
    wakes — the count is only meaningful for a fresh insert."""

    async def scenario():
        fake = FakeRepo()
        fake.ingest_result = IngestResult(row_id=MessageRowId(1), inserted=False)
        wakes: list[ChatKey] = []

        async def wake(chat_key: ChatKey) -> None:
            wakes.append(chat_key)

        ingest = Ingest(
            fake,
            Recorder(tmp_path / "events.jsonl"),
            wake=wake,
            identity=lambda ck: make_identity(),
        )
        result = await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=1.0)
        )
        return result, wakes

    result, wakes = run(scenario())
    assert result.inserted is False
    assert result.pending_count is None
    assert wakes == []


def test_non_message_event_pending_count_none(tmp_path):
    """A non-message event commits nothing: pending_count stays None."""

    async def scenario():
        fake = FakeRepo()
        ingest = Ingest(
            fake,
            Recorder(tmp_path / "events.jsonl"),
            identity=lambda ck: make_identity(),
        )
        result = await ingest.handle(
            AdapterEvent(type="notice", payload={"kind": "poke"}, ts=1.0)
        )
        return result

    result = run(scenario())
    assert result.inserted is False
    assert result.pending_count is None


# ── trusted delivery key forwarding ─────────────────────────────────────────

def test_self_message_without_delivery_key_is_unproven(tmp_path):
    """A self message without a trusted key is unproven — never
    heuristically matched. Ingest passes NO key; the repository (here the
    fake) reports the unproven verdict."""

    async def scenario():
        fake = FakeRepo()
        fake.ingest_result = IngestResult(
            row_id=MessageRowId(1), inserted=True, echo_status=EchoStatus.UNPROVEN
        )
        ingest = Ingest(
            fake,
            Recorder(tmp_path / "events.jsonl"),
            identity=lambda ck: make_identity(),
        )
        result = await ingest.handle(
            AdapterEvent(
                type="message",
                payload=make_message(msg_id="echo:1", is_self=True),
                ts=1.0,
            )
        )
        return result, fake.calls

    result, calls = run(scenario())
    assert result.echo_status == EchoStatus.UNPROVEN
    assert calls[0][3] is None  # no trusted key passed


def test_delivery_key_resolver_feeds_repo(tmp_path):
    """The resolver extracts the trusted delivery key from the transport
    metadata and passes it through to the repository."""

    async def scenario():
        fake = FakeRepo()
        fake.ingest_result = IngestResult(
            row_id=MessageRowId(9), inserted=True, echo_status=EchoStatus.RECONCILED
        )
        ingest = Ingest(
            fake,
            Recorder(tmp_path / "events.jsonl"),
            identity=lambda ck: make_identity(),
            delivery_key=lambda msg: "cy-1:0",
        )
        result = await ingest.handle(
            AdapterEvent(
                type="message",
                payload=make_message(msg_id="echo:1", is_self=True),
                ts=1.0,
            )
        )
        return result, fake.calls

    result, calls = run(scenario())
    assert result.echo_status == EchoStatus.RECONCILED
    assert calls[0][3] == "cy-1:0"  # the trusted key reached the repository


def test_delivery_key_resolver_not_called_for_non_self(tmp_path):
    """Only self messages may reconcile: the resolver is never consulted
    for ordinary inbound messages."""

    async def scenario():
        fake = FakeRepo()

        def resolver(msg):
            raise AssertionError("resolver must not run for non-self messages")

        ingest = Ingest(
            fake,
            Recorder(tmp_path / "events.jsonl"),
            identity=lambda ck: make_identity(),
            delivery_key=resolver,
        )
        result = await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=1.0)
        )
        return result, fake.calls

    result, calls = run(scenario())
    assert result.echo_status == EchoStatus.NOT_APPLICABLE
    assert calls[0][3] is None


# ── Phase 5: post-ingest person observation ──────────────────────────────────

def _async_observe(observed: list):
    """An async observe_person callback that records the sender id."""

    async def cb(msg):
        observed.append(msg.sender_id)

    return cb


def test_ingest_observes_newly_inserted_non_self_sender(tmp_path):
    """After a successful durable newly-inserted NON-SELF message, the
    sender alias is observed (a person row is created)."""

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        person = PersonService(repo)
        ingest = Ingest(
            repo,
            Recorder(tmp_path / "events.jsonl"),
            identity=lambda ck: make_identity(),
            observe_person=lambda msg: person.observe(
                msg.chat_key, msg.sender_id, msg.sender_name, now=1.0
            ),
        )
        result = await ingest.handle(
            AdapterEvent(
                type="message",
                payload=make_message(sender_id="u1", sender_name="alice"),
                ts=1.0,
            )
        )
        p = await person.get_profile(CK, SenderId("u1"))
        await repo.close()
        return result, p

    result, p = run(scenario())
    assert result.inserted is True
    assert p is not None
    assert p.names == ("alice",)


def test_ingest_does_not_observe_duplicate(tmp_path):
    """A duplicate (inserted=False) is never observed."""

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        observed: list = []
        ingest = Ingest(
            repo,
            Recorder(tmp_path / "events.jsonl"),
            identity=lambda ck: make_identity(),
            observe_person=_async_observe(observed),
        )
        event = AdapterEvent(type="message", payload=make_message(), ts=1.0)
        await ingest.handle(event)
        await ingest.handle(event)  # duplicate
        await repo.close()
        return observed

    observed = run(scenario())
    assert observed == [SenderId("u1")]  # only the first (inserted) message


def test_ingest_does_not_observe_self_message(tmp_path):
    """A self message is never observed."""

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        observed: list = []
        ingest = Ingest(
            repo,
            Recorder(tmp_path / "events.jsonl"),
            identity=lambda ck: make_identity(),
            observe_person=_async_observe(observed),
        )
        await ingest.handle(
            AdapterEvent(
                type="message",
                payload=make_message(msg_id="echo:1", is_self=True),
                ts=1.0,
            )
        )
        await repo.close()
        return observed

    assert run(scenario()) == []


def test_ingest_observation_failure_is_contained(tmp_path):
    """An observation failure is contained: it must not undo the ingest or
    the scheduler wake."""

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        wakes: list[ChatKey] = []

        async def wake(chat_key: ChatKey) -> None:
            wakes.append(chat_key)

        async def boom(msg):
            raise RuntimeError("observation failed")

        ingest = Ingest(
            repo,
            Recorder(tmp_path / "events.jsonl"),
            wake=wake,
            identity=lambda ck: make_identity(),
            observe_person=boom,
        )
        result = await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=1.0)
        )
        msg = await repo.get_message(CK, "m1")
        await repo.close()
        return result, wakes, msg

    result, wakes, msg = run(scenario())
    assert result.inserted is True  # ingest not undone
    assert wakes == [CK]  # scheduler wake not undone
    assert msg is not None and msg.text == "hello"  # durably committed


# ── Phase 6 P6.5b post-ingest media harvest wiring ───────────────────────────

def test_ingest_offers_new_nonself_message_to_harvest_lane(tmp_path):
    """A newly inserted NON-SELF message is offered to the harvest lane
    AFTER the durable commit, with the durable row id; duplicates and self
    messages are never offered. A harvest scheduling failure is contained
    and never undoes the ingest."""

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        offered: list[tuple] = []

        async def harvest(msg, row_id):
            offered.append((msg, row_id))

        ingest = make_ingest(
            repo, tmp_path / "events.jsonl", harvest_media=harvest
        )
        result = await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=1.0)
        )
        # A duplicate is never offered again.
        await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=2.0)
        )
        # A self message is never offered.
        self_msg = make_message(
            msg_id="m2", sender_id="bot-1", is_self=True, text="echo"
        )
        await ingest.handle(
            AdapterEvent(type="message", payload=self_msg, ts=3.0)
        )
        await repo.close()
        return result, offered

    result, offered = run(scenario())
    assert result.inserted is True
    assert len(offered) == 1
    msg, row_id = offered[0]
    assert msg.text == "hello"
    assert row_id == MessageRowId(1)


def test_ingest_harvest_scheduling_failure_contained(tmp_path):
    """A raising harvest callback is contained: the ingest result and the
    scheduler wake are unaffected."""

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        wakes: list[ChatKey] = []

        async def wake(chat_key: ChatKey) -> None:
            wakes.append(chat_key)

        async def harvest(msg, row_id):
            raise RuntimeError("harvest boom")

        ingest = make_ingest(
            repo, tmp_path / "events.jsonl", wake=wake, harvest_media=harvest
        )
        result = await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=1.0)
        )
        await repo.close()
        return result, wakes

    result, wakes = run(scenario())
    assert result.inserted is True
    assert wakes == [CK]  # the wake still fired


def test_ingest_recall_revokes_approved_and_pending_media(tmp_path):
    """A recall-shaped notice is handled locally and contained: approved
    source assets are revoked and pending source candidates cannot later be
    approved.  No platform delete/send operation is involved."""

    async def scenario():
        _db, repo = await open_repo(tmp_path / "t.db")
        ingest = Ingest(
            repo,
            Recorder(tmp_path / "events.jsonl"),
            identity=lambda ck: make_identity(),
        )
        await ingest.handle(
            AdapterEvent(type="message", payload=make_message(), ts=1.0)
        )
        approved_id = await repo.submit_media_candidate(
            MediaAssetCandidate(
                chat_key=CK,
                kind="sticker",
                cache_key="c" * 64,
                sha256="a" * 64,
                mime="image/gif",
                description="微笑",
                source_message_id=MessageRowId(1),
            ),
            now=2.0,
        )
        await repo.approve_media_candidate(CK, approved_id, capacity=4, now=3.0)
        await repo.submit_media_candidate(
            MediaAssetCandidate(
                chat_key=CK,
                kind="sticker",
                cache_key="d" * 64,
                sha256="b" * 64,
                mime="image/gif",
                description="另一张",
                source_message_id=MessageRowId(1),
            ),
            now=4.0,
        )
        result = await ingest.handle(
            AdapterEvent(
                type="notice",
                payload={
                    "notice_type": "group_recall",
                    "group_id": "123456",
                    "message_id": "m1",
                },
                ts=5.0,
            )
        )
        assets = await repo.list_media_assets(CK)
        pending = await repo.list_media_candidates(CK)
        await repo.close()
        return result, assets, pending

    result, assets, pending = run(scenario())
    assert result.inserted is False
    assert pending == []
    assert [asset.safety_status for asset in assets] == ["revoked", "rejected"]
