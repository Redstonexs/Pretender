"""Phase 6 adaptive foundation: the durable per-(chat, learner) state/runs,
source-bounded reads with policy-enforced ``is_self = 0``, exact source-batch
hash/CAS commits (valid empty advances; malformed/cancelled does not), the
chat+learner-scoped record surface excluding legacy/retired rows, idempotent
exposure/uses, bounded effect feedback with a code-owned reweight, canonical
record FTS, and bounded recovery — all over the v9 schema via the
AdaptiveRepository surface."""

from __future__ import annotations

import pytest

from pretender.errors import RepoError
from pretender.types import (
    ChatKey,
    LearnerBatch,
    LearnerBusy,
    LearnerDraft,
    LearnerGrant,
    LearnerRunRequest,
    MessageRowId,
    Record,
)
from tests.durable_helpers import CK, make_identity, make_message, open_repo_with_chat, run

OTHER = ChatKey("qq:group:other")
LEARNER = "personality"


def make_record(chat_key: ChatKey = CK, learner: str = LEARNER, text: str = "fact", **kw) -> Record:
    return Record(learner=learner, payload={"text": text}, chat_key=chat_key, **kw)


async def seed_messages(repo, chat_key: ChatKey = CK, n: int = 3, prefix: str = "msg", is_self: bool = False):
    """Insert ``n`` messages into the chat (row ids 1..n)."""
    for i in range(1, n + 1):
        await repo.ingest_message(
            None,
            make_message(
                chat_key=chat_key,
                msg_id=f"{prefix}-{i}",
                text=f"{prefix} {i}",
                is_self=is_self,
                recv_ts=1_700_000_000.0 + i,
            ),
        )


def make_request(chat_key: ChatKey = CK, learner: str = LEARNER, *, now: float = 100.0) -> LearnerRunRequest:
    return LearnerRunRequest(
        chat_key=chat_key, learner=learner, started_ts=now, expires_at=now + 400.0, now=now,
    )


async def acquire(repo, chat_key: ChatKey = CK, learner: str = LEARNER) -> LearnerGrant:
    grant = await repo.acquire_learner_run(make_request(chat_key, learner))
    assert grant is not None, "acquire must succeed"
    return grant


async def read_batch(repo, grant: LearnerGrant, *, tail: int = 100, policy: str = "nonself"):
    batch = await repo.read_learner_source_batch(
        grant.chat_key, grant.learner,
        through_msg_id=grant.through_msg_id, tail=tail, policy=policy,
    )
    assert batch is not None, "expected a source batch"
    return batch


async def commit_success(repo, grant: LearnerGrant, batch: LearnerBatch, records=(), *, expected=None):
    return await repo.commit_learner_source(
        LearnerDraft(
            chat_key=grant.chat_key,
            learner=grant.learner,
            batch=batch,
            records=tuple(records),
            expected_through_msg_id=expected,
        ),
        now=200.0,
    )


async def first_record_id(repo, chat_key: ChatKey = CK, learner: str = LEARNER) -> int:
    records = await repo.list_learner_records(chat_key, learner)
    assert records, "expected at least one adaptive record"
    rid = records[0].id
    assert rid is not None
    return rid


# ── Learner lease: acquire / busy / recover / renew / release ───────────────

def test_learner_lease_acquire_grant_and_busy(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=3)
        grant = await acquire(repo)
        assert grant.run_id >= 1
        assert grant.start_msg_id == MessageRowId(0)
        assert grant.through_msg_id == MessageRowId(3)
        # A second acquire while the first is live is BUSY with the exact
        # busy_until — never a second grant.
        busy = await repo.acquire_learner_run(make_request())
        assert isinstance(busy, LearnerBusy)
        assert busy.run_id == grant.run_id
        assert busy.busy_until == 500.0
        # A different learner is a separate lease lane.
        other = await repo.acquire_learner_run(make_request(learner="other"))
        assert isinstance(other, LearnerGrant)
        await repo.close()

    run(scenario())


def test_learner_lease_expired_run_is_recovered(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        g1 = await repo.acquire_learner_run(
            LearnerRunRequest(chat_key=CK, learner=LEARNER, started_ts=100.0, expires_at=200.0, now=100.0)
        )
        assert isinstance(g1, LearnerGrant)
        # The lease expired: a new acquire at now=300 recovers it and grants
        # a fresh run (the expired one is marked expired).
        g2 = await repo.acquire_learner_run(
            LearnerRunRequest(chat_key=CK, learner=LEARNER, started_ts=300.0, expires_at=700.0, now=300.0)
        )
        assert isinstance(g2, LearnerGrant)
        assert g2.run_id != g1.run_id
        runs = await repo.list_learner_runs(CK, LEARNER)
        states = {r.state for r in runs}
        await repo.close()
        return states

    assert run(scenario()) == {"prepared", "expired"}


def test_learner_lease_renew_and_release(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=1)
        grant = await acquire(repo)
        # Renew extends the lease; an expired owner cannot renew.
        assert await repo.renew_learner_run(CK, LEARNER, grant.run_id, 900.0, now=100.0) is True
        assert await repo.renew_learner_run(CK, LEARNER, grant.run_id, 950.0, now=901.0) is False
        # Release gives the run back; a fresh acquire succeeds immediately.
        await repo.release_learner_run(CK, LEARNER, grant.run_id)
        g2 = await acquire(repo)
        await repo.close()
        return g2.run_id

    assert run(scenario()) >= 2


def test_learner_lease_unknown_chat_returns_none(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        result = await repo.acquire_learner_run(make_request(chat_key=OTHER))
        await repo.close()
        return result

    assert run(scenario()) is None


# ── Source-bounded reads: policy-enforced is_self = 0 ───────────────────────

def test_nonself_policy_excludes_self_text_from_batch(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2, prefix="human")
        await seed_messages(repo, n=1, prefix="bot", is_self=True)
        grant = await acquire(repo)
        batch = await read_batch(repo, grant)
        # The bot's own message (row 3) never enters a nonself batch.
        assert batch.texts == ("human 1", "human 2")
        assert "bot 1" not in batch.texts
        # The ``all`` policy includes it.
        batch_all = await repo.read_learner_source_batch(
            CK, LEARNER, through_msg_id=grant.through_msg_id, tail=100, policy="all"
        )
        assert batch_all is not None
        assert batch_all.texts == ("human 1", "human 2", "bot 1")
        await repo.close()

    run(scenario())


def test_source_batch_is_sql_bounded_and_retains_oldest_tail(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=5)
        grant = await acquire(repo)
        # tail 3: the OLDEST bounded unsummarized chunk [1, 3].
        batch = await repo.read_learner_source_batch(
            CK, LEARNER, through_msg_id=grant.through_msg_id, tail=3
        )
        assert batch is not None
        assert batch.first_msg_id == MessageRowId(1)
        assert batch.last_msg_id == MessageRowId(3)
        assert batch.observed_watermark == MessageRowId(0)
        await repo.close()

    run(scenario())


def test_source_batch_none_for_unknown_chat_and_no_messages(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        grant = await acquire(repo)
        assert await repo.read_learner_source_batch(
            OTHER, LEARNER, through_msg_id=MessageRowId(5), tail=10
        ) is None
        assert await repo.read_learner_source_batch(
            CK, LEARNER, through_msg_id=grant.through_msg_id, tail=10
        ) is None
        await repo.close()

    run(scenario())


# ── Exact source-batch hash/CAS commit ──────────────────────────────────────

def test_commit_inserts_records_sources_run_and_advances_watermark(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=3)
        grant = await acquire(repo)
        batch = await read_batch(repo, grant)
        assert await commit_success(
            repo, grant, batch, [make_record(text="likes tea")]
        ) is True
        state = await repo.get_learner_state(CK, LEARNER)
        assert state is not None
        assert state.watermark_msg_id == MessageRowId(3)
        assert state.last_run_id == grant.run_id
        # The run is settled success with the exact source hash.
        runs = await repo.list_learner_runs(CK, LEARNER)
        assert runs[0].state == "success"
        assert runs[0].source_hash == batch.source_hash
        assert runs[0].records_added == 1
        # The record reads back with its content_hash and source range.
        records = await repo.list_learner_records(CK, LEARNER)
        assert len(records) == 1
        assert records[0].content_hash is not None
        assert records[0].source_first_msg_id == MessageRowId(1)
        assert records[0].source_last_msg_id == MessageRowId(3)
        # The opaque source mapping exists (record_id -> this batch's range).
        rows = await repo._db.read(
            lambda c: c.execute(
                "SELECT record_id, source_first_msg_id, source_last_msg_id,"
                " source_hash FROM record_sources"
            ).fetchall()
        )
        await repo.close()
        return records[0], rows

    rec, rows = run(scenario())
    assert len(rows) == 1
    assert rows[0][0] == rec.id  # mapped to the actual record id
    assert rows[0][1] == 1 and rows[0][2] == 3  # this batch's source range
    assert len(rows[0][3]) == 64  # the exact source hash is stored


def test_commit_valid_empty_result_advances_watermark(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        grant = await acquire(repo)
        batch = await read_batch(repo, grant)
        # A valid EMPTY result (zero records) still advances the watermark.
        assert await commit_success(repo, grant, batch) is True
        state = await repo.get_learner_state(CK, LEARNER)
        assert state is not None
        assert state.watermark_msg_id == MessageRowId(2)
        runs = await repo.list_learner_runs(CK, LEARNER)
        assert runs[0].state == "success"
        assert runs[0].records_added == 0
        # Nothing is beyond the watermark anymore: no pending work.
        assert await repo.list_learner_pending_chats(LEARNER) == []
        await repo.close()

    run(scenario())


def test_commit_malformed_and_cancelled_do_not_advance(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        grant = await acquire(repo)
        batch = await read_batch(repo, grant)
        # Malformed: the run is settled malformed, nothing advances.
        assert await repo.commit_learner_source(
            LearnerDraft(
                chat_key=CK, learner=LEARNER, batch=batch,
                outcome="malformed", error="bad json",
            ),
            now=200.0,
        ) is True
        state = await repo.get_learner_state(CK, LEARNER)
        assert state is None  # no watermark row was created
        runs = await repo.list_learner_runs(CK, LEARNER)
        assert runs[0].state == "malformed"
        assert runs[0].error == "bad json"
        assert await repo.list_learner_records(CK, LEARNER) == []
        # The source is still pending: a fresh run reads it again.
        g2 = await acquire(repo)
        batch2 = await read_batch(repo, g2)
        assert batch2.first_msg_id == MessageRowId(1)
        # Cancelled: same fail-closed behavior.
        assert await repo.commit_learner_source(
            LearnerDraft(
                chat_key=CK, learner=LEARNER, batch=batch2,
                outcome="cancelled", error="interrupted",
            ),
            now=300.0,
        ) is True
        runs2 = await repo.list_learner_runs(CK, LEARNER)
        await repo.close()
        return runs2[0].state, runs2[0].error

    state, error = run(scenario())
    assert state == "cancelled"
    assert error == "interrupted"


def test_commit_stale_cas_loser_changes_nothing(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=3)
        grant = await acquire(repo)
        batch = await read_batch(repo, grant)
        assert await commit_success(repo, grant, batch, [make_record(text="a")]) is True
        # A retry with the ORIGINAL expected watermark is a stale CAS loser.
        assert await commit_success(
            repo, grant, batch, [make_record(text="b")], expected=MessageRowId(0)
        ) is False
        records = await repo.list_learner_records(CK, LEARNER)
        await repo.close()
        return records

    records = run(scenario())
    assert len(records) == 1  # exactly one record, never duplicated
    assert records[0].payload == {"text": "a"}


def test_commit_merges_identical_content(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        grant = await acquire(repo)
        batch = await read_batch(repo, grant)
        assert await commit_success(repo, grant, batch, [make_record(text="same")]) is True
        # A second run over the next range produces the SAME content: the
        # record is MERGED (same content_hash), not duplicated.
        await seed_messages(repo, n=1, prefix="more")
        g2 = await acquire(repo)
        batch2 = await read_batch(repo, g2)
        assert await commit_success(repo, g2, batch2, [make_record(text="same")]) is True
        records = await repo.list_learner_records(CK, LEARNER)
        runs = await repo.list_learner_runs(CK, LEARNER)
        await repo.close()
        return records, runs

    records, runs = run(scenario())
    assert len(records) == 1  # merged, not duplicated
    assert runs[0].records_merged == 1
    assert runs[1].records_added == 1


def test_commit_rejects_bad_hash_and_cross_chat(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        await seed_messages(repo, n=2)
        grant = await acquire(repo)
        batch = await read_batch(repo, grant)
        # A tampered hash fails closed.
        bad_batch = LearnerBatch(
            chat_key=CK, learner=LEARNER, first_msg_id=batch.first_msg_id,
            last_msg_id=batch.last_msg_id, source_hash="deadbeef", texts=batch.texts,
        )
        with pytest.raises(RepoError):
            await repo.commit_learner_source(
                LearnerDraft(chat_key=CK, learner=LEARNER, batch=bad_batch),
                now=200.0,
            )
        # A cross-chat record is rejected at draft construction (the
        # boundary type validates identity consistency).
        with pytest.raises(ValueError):
            LearnerDraft(
                chat_key=CK, learner=LEARNER, batch=batch,
                records=(make_record(chat_key=OTHER),),
            )
        # A batch beyond the run's fixed through boundary fails closed.
        over = LearnerBatch(
            chat_key=CK, learner=LEARNER, first_msg_id=MessageRowId(1),
            last_msg_id=MessageRowId(99), source_hash=batch.source_hash,
            texts=batch.texts,
        )
        with pytest.raises(RepoError):
            await repo.commit_learner_source(
                LearnerDraft(chat_key=CK, learner=LEARNER, batch=over),
                now=200.0,
            )
        # No prepared run for a never-acquired learner fails closed.
        nobody_batch = LearnerBatch(
            chat_key=CK, learner="nobody", first_msg_id=batch.first_msg_id,
            last_msg_id=batch.last_msg_id, source_hash=batch.source_hash,
            texts=batch.texts,
        )
        with pytest.raises(RepoError):
            await repo.commit_learner_source(
                LearnerDraft(chat_key=CK, learner="nobody", batch=nobody_batch),
                now=200.0,
            )
        await repo.close()

    run(scenario())


# ── list/select records: chat+learner scoped, legacy/retired excluded ───────

def test_list_and_select_exclude_legacy_and_retired(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        await seed_messages(repo, n=2)
        grant = await acquire(repo)
        batch = await read_batch(repo, grant)
        await commit_success(repo, grant, batch, [make_record(text="trusted")])
        # A legacy record (the legacy add_record path — no content_hash).
        await repo.add_record(make_record(text="legacy"))
        # A record for another chat/learner.
        await seed_messages(repo, chat_key=OTHER, n=1, prefix="other")
        g_other = await repo.acquire_learner_run(make_request(chat_key=OTHER, learner=LEARNER))
        assert isinstance(g_other, LearnerGrant)
        b_other = await repo.read_learner_source_batch(
            OTHER, LEARNER, through_msg_id=g_other.through_msg_id, tail=10
        )
        assert b_other is not None
        await commit_success(repo, g_other, b_other, [make_record(chat_key=OTHER, text="other")])
        # Retire the trusted record directly (the retired flag is repo-owned
        # state the adaptive surface honors).
        await repo._db.write(
            lambda c: c.execute("UPDATE records SET retired = 1 WHERE payload_json LIKE '%trusted%'")
        )
        listed = await repo.list_learner_records(CK, LEARNER)
        selected = await repo.select_learner_records(CK, LEARNER)
        other_listed = await repo.list_learner_records(OTHER, LEARNER)
        await repo.close()
        return listed, selected, other_listed

    listed, selected, other_listed = run(scenario())
    # The legacy record (no content_hash) and the retired record are absent;
    # the other chat's record never leaks in.
    assert listed == []
    assert selected == []
    assert [r.payload for r in other_listed] == [{"text": "other"}]


def test_select_orders_by_weight_then_uses(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        grant = await acquire(repo)
        batch = await read_batch(repo, grant)
        await commit_success(
            repo, grant, batch,
            [make_record(text="low", weight=0.5), make_record(text="high", weight=3.0)],
        )
        selected = await repo.select_learner_records(CK, LEARNER, limit=10)
        await repo.close()
        return [r.payload["text"] for r in selected]

    assert run(scenario()) == ["high", "low"]


# ── Atomic idempotent exposure / uses ───────────────────────────────────────

def test_record_exposure_is_idempotent(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        grant = await acquire(repo)
        batch = await read_batch(repo, grant)
        await commit_success(repo, grant, batch, [make_record(text="x")])
        record_id = await first_record_id(repo)
        # First exposure inserts; the duplicate is idempotent (False).
        assert await repo.record_exposure(CK, LEARNER, record_id, grant.run_id, now=300.0) is True
        assert await repo.record_exposure(CK, LEARNER, record_id, grant.run_id, now=301.0) is False
        count = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM record_exposures").fetchone()[0]
        )
        # A different run is a distinct exposure.
        g2 = await acquire(repo)
        assert await repo.record_exposure(CK, LEARNER, record_id, g2.run_id, now=400.0) is True
        count2 = await repo._db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM record_exposures").fetchone()[0]
        )
        # Cross-chat / legacy / retired records fail closed.
        with pytest.raises(RepoError):
            await repo.record_exposure(OTHER, LEARNER, record_id, g2.run_id, now=400.0)
        await repo.close()
        return count, count2

    assert run(scenario()) == (1, 2)


def test_increment_record_uses_is_atomic_and_scoped(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        grant = await acquire(repo)
        batch = await read_batch(repo, grant)
        await commit_success(repo, grant, batch, [make_record(text="x")])
        record_id = await first_record_id(repo)
        assert await repo.increment_record_uses(CK, LEARNER, record_id) is True
        assert await repo.increment_record_uses(CK, LEARNER, record_id) is True
        # Cross-chat / unknown records are False, never an error.
        assert await repo.increment_record_uses(OTHER, LEARNER, record_id) is False
        assert await repo.increment_record_uses(CK, LEARNER, 999) is False
        rec = (await repo.list_learner_records(CK, LEARNER))[0]
        await repo.close()
        return rec.uses

    assert run(scenario()) == 2


# ── Bounded effect feedback + code-owned reweight [0.1, 5] ──────────────────

def test_feedback_reweights_within_bounds(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        grant = await acquire(repo)
        batch = await read_batch(repo, grant)
        await commit_success(repo, grant, batch, [make_record(text="x", weight=1.0)])
        record_id = await first_record_id(repo)
        # Positive effect doubles the weight (bounded by the 5.0 cap).
        w1 = await repo.apply_record_feedback(CK, LEARNER, record_id, 1.0, now=300.0)
        assert w1 == 2.0
        # Repeated positive effects clamp at 5.0.
        w2 = await repo.apply_record_feedback(CK, LEARNER, record_id, 1.0, now=301.0)
        w3 = await repo.apply_record_feedback(CK, LEARNER, record_id, 1.0, now=302.0)
        assert w2 == 4.0
        assert w3 == 5.0
        # Negative effect floors at 0.1.
        w4 = await repo.apply_record_feedback(CK, LEARNER, record_id, -1.0, now=303.0)
        assert w4 == 0.1
        # Unknown/cross-chat records return None.
        assert await repo.apply_record_feedback(OTHER, LEARNER, record_id, 0.5, now=304.0) is None
        assert await repo.apply_record_feedback(CK, LEARNER, 999, 0.5, now=304.0) is None
        # The feedback rows are recorded with the code-owned reweight.
        rows = await repo._db.read(
            lambda c: c.execute(
                "SELECT effect, reweight FROM record_feedback ORDER BY id"
            ).fetchall()
        )
        await repo.close()
        return w1, w2, w3, w4, rows

    w1, w2, w3, w4, rows = run(scenario())
    assert (w1, w2, w3, w4) == (2.0, 4.0, 5.0, 0.1)
    assert rows == [(1.0, 2.0), (1.0, 4.0), (1.0, 5.0), (-1.0, 0.1)]


def test_feedback_rejects_out_of_bounds_effect(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        grant = await acquire(repo)
        batch = await read_batch(repo, grant)
        await commit_success(repo, grant, batch, [make_record(text="x")])
        record_id = await first_record_id(repo)
        for bad in (2.0, -2.0, float("nan"), float("inf")):
            try:
                await repo.apply_record_feedback(CK, LEARNER, record_id, bad, now=300.0)
                raised = False
            except ValueError:
                raised = True
            assert raised, f"effect {bad} must be rejected"
        await repo.close()

    run(scenario())


# ── Record FTS search ───────────────────────────────────────────────────────

def test_record_fts_query_is_chat_and_learner_scoped(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        await seed_messages(repo, n=2)
        grant = await acquire(repo)
        batch = await read_batch(repo, grant)
        await commit_success(repo, grant, batch, [make_record(text="火锅好吃")])
        await seed_messages(repo, n=1, prefix="more")
        g2 = await acquire(repo)
        batch2 = await read_batch(repo, g2)
        await commit_success(repo, g2, batch2, [make_record(text="banana")])
        # A different learner in the same chat.
        g3 = await repo.acquire_learner_run(make_request(learner="other"))
        assert isinstance(g3, LearnerGrant)
        b3 = await repo.read_learner_source_batch(CK, "other", through_msg_id=g3.through_msg_id, tail=10)
        assert b3 is not None
        await commit_success(repo, g3, b3, [make_record(learner="other", text="火锅别的")])
        hits = await repo.query_records(CK, LEARNER, "火锅", limit=10)
        other_hits = await repo.query_records(OTHER, LEARNER, "火锅", limit=10)
        learner_hits = await repo.query_records(CK, "other", "火锅", limit=10)
        await repo.close()
        return hits, other_hits, learner_hits

    hits, other_hits, learner_hits = run(scenario())
    assert len(hits) == 1
    assert hits[0].text == "火锅好吃"
    assert hits[0].learner == LEARNER
    assert other_hits == []  # chat-scoped
    assert [h.text for h in learner_hits] == ["火锅别的"]  # learner-scoped


def test_record_fts_query_is_bounded(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        grant = await acquire(repo)
        batch = await read_batch(repo, grant)
        await commit_success(
            repo, grant, batch,
            [make_record(text=f"火锅{i}") for i in range(1, 4)],
        )
        hits = await repo.query_records(CK, LEARNER, "火锅", limit=2)
        await repo.close()
        return hits

    assert len(run(scenario())) == 2


# ── State / list methods for bounded recovery ───────────────────────────────

def test_learner_state_and_pending_chats(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        assert await repo.get_learner_state(CK, LEARNER) is None
        # No messages: nothing pending.
        assert await repo.list_learner_pending_chats(LEARNER) == []
        await seed_messages(repo, n=2)
        assert await repo.list_learner_pending_chats(LEARNER) == [CK]
        grant = await acquire(repo)
        batch = await read_batch(repo, grant)
        await commit_success(repo, grant, batch, [make_record(text="x")])
        # After the watermark advances, the chat is no longer pending.
        assert await repo.list_learner_pending_chats(LEARNER) == []
        # A new message makes it pending again.
        await seed_messages(repo, n=1, prefix="new")
        assert await repo.list_learner_pending_chats(LEARNER) == [CK]
        state = await repo.get_learner_state(CK, LEARNER)
        await repo.close()
        return state

    state = run(scenario())
    assert state is not None
    assert state.watermark_msg_id == MessageRowId(2)
    assert state.observed_watermark_msg_id == MessageRowId(0)


def test_list_learner_runs_is_bounded_and_newest_first(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        for i in range(3):
            grant = await acquire(repo)
            batch = await read_batch(repo, grant)
            await commit_success(repo, grant, batch, [make_record(text=f"r{i}")])
            await seed_messages(repo, n=1, prefix=f"m{i}")
        runs = await repo.list_learner_runs(CK, LEARNER, limit=2)
        await repo.close()
        return [r.id for r in runs]

    ids = run(scenario())
    assert len(ids) == 2
    assert ids == sorted(ids, reverse=True)  # newest first