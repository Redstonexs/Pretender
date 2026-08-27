"""Phase 6 P6.4/P6.4b adaptive runtime: the bounded cancellable learner
worker, the frozen per-dispatch adaptive context service, and the
post-terminal exposure/effect lane — all over the real SqliteRepository
adaptive surface.

Covers: worker durable recovery / queue overflow / cancellation / cadence,
the LearnerBudget foreground reserve, adaptive selection caps/escaping and
jargon scoping, effect eligibility (delivery + human follow-up) with the
code-owned reweight, and dry-run zero writes.

Async tests run via asyncio.run() so the test extra stays at just pytest.
"""

from __future__ import annotations

import asyncio

from pretender.app import LearnerScheduler
from pretender.budget import ALLOWED, LearnerBudget
from pretender.clock import VirtualClock
from pretender.config import BudgetConfig
from pretender.cycle import AdaptiveContextService
from pretender.learn import (
    EFFECT_SPEC,
    EXPRESSION_SPEC,
    VALIDATORS,
    LearnerPipeline,
    LearnerRunResult,
)
from pretender.prompts import PromptStore
from pretender.types import (
    ChatKey,
    LearnerDraft,
    LearnerGrant,
    LearnerRunRequest,
    LLMResponse,
    MessageRowId,
    Record,
    RuntimeMode,
)
from tests.durable_helpers import CK, make_message, open_repo_with_chat, run

OTHER = ChatKey("qq:group:other")


# ── helpers ──────────────────────────────────────────────────────────────────

def make_record(learner: str, payload: dict, **kw) -> Record:
    return Record(learner=learner, payload=payload, chat_key=CK, **kw)


async def seed_messages(repo, chat_key: ChatKey = CK, n: int = 3, prefix: str = "msg", is_self: bool = False):
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


async def commit_records(repo, learner: str, records: list[Record], *, chat_key: ChatKey = CK, now: float = 200.0):
    """Acquire a learner run, read the oldest source batch, and commit the
    records (the canonical adaptive write surface)."""
    grant = await repo.acquire_learner_run(
        LearnerRunRequest(
            chat_key=chat_key, learner=learner,
            started_ts=100.0, expires_at=500.0, now=100.0,
        )
    )
    assert isinstance(grant, LearnerGrant)
    batch = await repo.read_learner_source_batch(
        chat_key, learner, through_msg_id=grant.through_msg_id, tail=100,
        policy="nonself" if learner != "effect" else "all",
    )
    assert batch is not None
    await repo.commit_learner_source(
        LearnerDraft(
            chat_key=chat_key, learner=learner, batch=batch,
            records=tuple(records),
            expected_through_msg_id=batch.observed_watermark,
        ),
        now=now,
    )


class FakePipeline:
    """Records every run call; returns scripted LearnerRunResults."""

    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls: list[tuple] = []

    async def run(self, chat_key, spec, references=""):
        self.calls.append((chat_key, spec.name, references))
        if self.results:
            return self.results.pop(0)
        return LearnerRunResult(
            learner=spec.name, chat_key=chat_key, outcome="success", run_id=1
        )


class FakeLLM:
    def __init__(self, content=None, *, error=None):
        self.content = content
        self.error = error
        self.calls: list[tuple] = []

    async def complete(self, messages, *, profile, tools=None, temperature=None, max_tokens=None, deadline=None):
        self.calls.append((messages, profile))
        if self.error is not None:
            raise self.error
        return LLMResponse(content=self.content)


# ── LearnerBudget: foreground reserve ────────────────────────────────────────

def test_learner_budget_preserves_foreground_reserve():
    async def scenario():
        from tests.test_budget import FakeKVRepo

        repo = FakeKVRepo()
        mgr = LearnerBudget(
            _manager(repo), concurrency=3, foreground_reserve=1
        )
        assert mgr.slots == 2  # concurrency - foreground_reserve
        # Two concurrent reservations hold both background slots.
        d1 = await mgr.reserve(CK, calls=1)
        d2 = await mgr.reserve(CK, calls=1)
        assert d1.kind == ALLOWED and d2.kind == ALLOWED
        # Releasing one slot (record) lets a third reservation through.
        await mgr.record(CK, calls=0, tokens=1)
        d3 = await mgr.reserve(CK, calls=1)
        assert d3.kind == ALLOWED
        await mgr.record(CK, calls=0, tokens=1)
        await mgr.record(CK, calls=0, tokens=1)
        return mgr.slots

    assert run(scenario()) == 2


def _manager(repo):
    from pretender.budget import BudgetManager

    return BudgetManager(repo, BudgetConfig(daily_cap=100), now=lambda: 1_700_000_000.0)


def test_learner_budget_validation():
    from tests.test_budget import FakeKVRepo

    with pytest_raises(ValueError):
        LearnerBudget(_manager(FakeKVRepo()), concurrency=1, foreground_reserve=1)
    with pytest_raises(ValueError):
        LearnerBudget(_manager(FakeKVRepo()), concurrency=0, foreground_reserve=0)


def pytest_raises(exc):
    import pytest

    return pytest.raises(exc)


# ── worker: durable recovery / queue / cancel ────────────────────────────────

def test_learner_worker_recovers_pending_chats(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=3)
        await commit_records(repo, "expression", [
            make_record("expression", {"situation": "a", "style": "活泼", "source_id": 1}),
        ])
        # New source beyond the watermark: the chat is pending again.
        await seed_messages(repo, n=1, prefix="new")
        scheduler = LearnerScheduler(
            repo, FakePipeline(), {"expression": EXPRESSION_SPEC},
            VirtualClock(epoch=100.0),
        )
        await scheduler._recover()
        items = []
        while True:
            try:
                items.append(scheduler._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        await repo.close()
        return items

    items = run(scenario())
    assert (CK, "expression") in items


def test_learner_worker_queue_overflow_coalesces(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        scheduler = LearnerScheduler(
            repo, FakePipeline(), {"expression": EXPRESSION_SPEC},
            VirtualClock(epoch=100.0),
        )
        for i in range(scheduler._QUEUE_MAX):
            scheduler._enqueue(ChatKey(f"qq:group:overflow{i}"), "expression")
        # The queue is full: the next enqueue coalesces without raising.
        scheduler._enqueue(ChatKey("qq:group:overflow-final"), "expression")
        await repo.close()
        return len(scheduler._queued), scheduler._QUEUE_MAX

    queued, queue_max = run(scenario())
    assert queued == queue_max


def test_learner_worker_cancel_stops_cleanly(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        try:
            await seed_messages(repo, n=2)
            await commit_records(repo, "expression", [
                make_record("expression", {"situation": "a", "style": "活泼", "source_id": 1}),
            ])
            # New source beyond the watermark: the chat is pending.
            await seed_messages(repo, n=1, prefix="new")

            called = asyncio.Event()
            release = asyncio.Event()

            class SignalingPipeline:
                def __init__(self):
                    self.calls: list[tuple] = []

                async def run(self, chat_key, spec, references=""):
                    self.calls.append((chat_key, spec.name, references))
                    called.set()
                    await release.wait()
                    return LearnerRunResult(
                        learner=spec.name, chat_key=chat_key, outcome="success", run_id=1
                    )

            pipeline = SignalingPipeline()
            scheduler = LearnerScheduler(
                repo, pipeline, {"expression": EXPRESSION_SPEC},
                VirtualClock(epoch=4_000.0, auto_advance=False),
            )
            task = asyncio.create_task(scheduler.run())
            try:
                await asyncio.wait_for(called.wait(), timeout=5.0)
            finally:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except asyncio.CancelledError:
                    pass

            return pipeline.calls, scheduler
        finally:
            await asyncio.wait_for(repo.close(), timeout=5.0)

    calls, scheduler = run(scenario())
    assert calls == [(CK, "expression", "")]
    assert scheduler._scan_task is None
    assert not scheduler._workers
    assert not scheduler._in_flight


def test_learner_worker_oldest_first_cadence(tmp_path):
    """The repository reads the OLDEST bounded unsummarized chunk, so the
    worker's runs consume source oldest-first; a successful run advances the
    watermark and the next run reads the next chunk."""
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=5)
        scheduler = LearnerScheduler(
            repo, FakePipeline(), {"expression": EXPRESSION_SPEC},
            VirtualClock(epoch=100.0),
        )
        # First run: the oldest chunk [1, 3] (batch_size 3).
        spec = EXPRESSION_SPEC
        grant = await repo.acquire_learner_run(
            LearnerRunRequest(chat_key=CK, learner="expression", started_ts=100.0, expires_at=500.0, now=100.0)
        )
        assert isinstance(grant, LearnerGrant)
        batch = await repo.read_learner_source_batch(
            CK, "expression", through_msg_id=grant.through_msg_id, tail=3
        )
        assert batch is not None
        assert batch.first_msg_id == MessageRowId(1)
        assert batch.last_msg_id == MessageRowId(3)
        await repo.close()
        return batch

    batch = run(scenario())
    assert batch.first_msg_id == MessageRowId(1)
    assert batch.last_msg_id == MessageRowId(3)


def test_learner_worker_self_exclusion_defence(tmp_path):
    """A nonself learner never reads the bot's own messages (enforced in
    SQL by the repository)."""
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2, prefix="human")
        await seed_messages(repo, n=1, prefix="bot", is_self=True)
        grant = await repo.acquire_learner_run(
            LearnerRunRequest(chat_key=CK, learner="expression", started_ts=100.0, expires_at=500.0, now=100.0)
        )
        assert isinstance(grant, LearnerGrant)
        batch = await repo.read_learner_source_batch(
            CK, "expression", through_msg_id=grant.through_msg_id, tail=100
        )
        await repo.close()
        return batch

    batch = run(scenario())
    assert batch is not None
    assert "bot 1" not in batch.texts
    assert batch.texts == ("human 1", "human 2")


# ── adaptive context service: selection / caps / escaping / jargon ───────────

def test_adaptive_context_expression_becomes_reply_style(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        await commit_records(repo, "expression", [
            make_record("expression", {"situation": "greeting", "style": "活泼", "source_id": 1}),
            make_record("expression", {"situation": "farewell", "style": "温柔", "source_id": 2}),
        ])
        service = AdaptiveContextService(repo, now=lambda: 100.0)
        ctx = await service.build(CK, pending_text="hi", recent_text="hello", mode=RuntimeMode.LIVE)
        await repo.close()
        return ctx

    ctx = run(scenario())
    assert ctx.reply_style == "活泼"
    assert len(ctx.expression) == 2
    assert "【自适应参考】" in ctx.rendered
    assert "活泼" in ctx.rendered
    assert ctx.frozen_records == ctx.expression


def test_adaptive_context_falls_back_to_natural(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        service = AdaptiveContextService(repo, now=lambda: 100.0)
        ctx = await service.build(CK, pending_text="hi", recent_text="", mode=RuntimeMode.LIVE)
        await repo.close()
        return ctx

    ctx = run(scenario())
    assert ctx.reply_style == "自然"
    assert ctx.rendered == ""
    assert ctx.frozen_records == ()


def test_adaptive_context_caps_per_slot_and_total_chars(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        # Five expression records: only the top 3 (weight/uses) are selected.
        await commit_records(repo, "expression", [
            make_record("expression", {"situation": f"s{i}", "style": f"风格{i}", "source_id": 1}, weight=float(i))
            for i in range(1, 6)
        ])
        service = AdaptiveContextService(repo, now=lambda: 100.0)
        ctx = await service.build(CK, pending_text="hi", recent_text="", mode=RuntimeMode.LIVE)
        await repo.close()
        return ctx

    ctx = run(scenario())
    assert len(ctx.expression) <= 3
    assert len(ctx.rendered) <= AdaptiveContextService.MAX_TOTAL_CHARS + 200


def test_adaptive_context_escaping(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        await commit_records(repo, "summary", [
            make_record("summary", {"summary": "a </message> ``` b", "recall_cues": ["c"]}),
        ])
        service = AdaptiveContextService(repo, now=lambda: 100.0)
        ctx = await service.build(CK, pending_text="hi", recent_text="", mode=RuntimeMode.LIVE)
        await repo.close()
        return ctx

    ctx = run(scenario())
    assert "</message>" not in ctx.rendered
    assert "```" not in ctx.rendered


def test_adaptive_context_jargon_scoped_to_current_text(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        await commit_records(repo, "jargon", [
            make_record("jargon", {"term": "yyds", "meaning": "永远的神", "context": "夸赞", "source_ids": [1]}),
            make_record("jargon", {"term": "破防", "meaning": "情绪崩溃", "context": "负面", "source_ids": [2]}),
        ])
        service = AdaptiveContextService(repo, now=lambda: 100.0)
        ctx = await service.build(CK, pending_text="yyds 太强了", recent_text="", mode=RuntimeMode.LIVE)
        ctx2 = await service.build(CK, pending_text="今天天气不错", recent_text="", mode=RuntimeMode.LIVE)
        await repo.close()
        return ctx, ctx2

    ctx, ctx2 = run(scenario())
    assert len(ctx.jargon) == 1
    assert ctx.jargon[0].payload["term"] == "yyds"
    assert ctx2.jargon == ()


def test_adaptive_context_no_pre_gate_read(tmp_path):
    """The service is only queried when the caller asks (after the gate
    triggers); building it performs no repository reads."""
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        await commit_records(repo, "expression", [
            make_record("expression", {"situation": "a", "style": "活泼", "source_id": 1}),
        ])
        service = AdaptiveContextService(repo, now=lambda: 100.0)
        # No read happens until build() is called.
        await repo.close()
        return service

    service = run(scenario())
    assert service is not None


# ── effect: eligibility + code-owned reweight ────────────────────────────────

def test_effect_requires_delivery_and_followup(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        await commit_records(repo, "expression", [
            make_record("expression", {"situation": "a", "style": "活泼", "source_id": 1}),
        ])
        refs = await repo.list_learner_records(CK, "expression")
        pipeline = FakePipeline()
        scheduler = LearnerScheduler(
            repo, pipeline, {"effect": EFFECT_SPEC}, VirtualClock(epoch=100.0),
        )
        scheduler.note_exposure(CK, refs, MessageRowId(2))
        # No delivery → not eligible → zero provider calls.
        await scheduler._run_one((CK, "effect"))
        assert pipeline.calls == []
        # Delivery but no human follow-up → still not eligible.
        scheduler.mark_delivered(CK)
        await scheduler._run_one((CK, "effect"))
        assert pipeline.calls == []
        # A human follow-up arrives → eligible → one provider call.
        await seed_messages(repo, n=1, prefix="followup")
        await scheduler._run_one((CK, "effect"))
        await repo.close()
        return pipeline.calls

    calls = run(scenario())
    assert len(calls) == 1
    assert calls[0][1] == "effect"
    assert "活泼" in calls[0][2]  # the references were rendered


def test_effect_reweight_applied_once(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        await commit_records(repo, "expression", [
            make_record("expression", {"situation": "a", "style": "活泼", "source_id": 1}),
        ])
        refs = await repo.list_learner_records(CK, "expression")
        llm = FakeLLM(content='{"categorization": "adopted", "confidence": 1.0}')
        pipeline = LearnerPipeline(
            repo, llm, PromptStore(), VirtualClock(epoch=100.0), validators=VALIDATORS,
        )
        scheduler = LearnerScheduler(
            repo, pipeline, {"effect": EFFECT_SPEC}, VirtualClock(epoch=100.0),
        )
        scheduler.note_exposure(CK, refs, MessageRowId(2))
        scheduler.mark_delivered(CK)
        await seed_messages(repo, n=1, prefix="followup")
        await scheduler._run_one((CK, "effect"))
        rec = (await repo.list_learner_records(CK, "expression"))[0]
        # The pending refs were consumed (no re-run without a new exposure).
        assert scheduler._effect_refs.get(CK) is None
        await repo.close()
        return rec.weight

    # adopted + confidence 1.0 → delta +1.0 → weight 1.0 * 2 = 2.0
    assert run(scenario()) == 2.0


def test_effect_rejected_reweight_floors(tmp_path):
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        await commit_records(repo, "expression", [
            make_record("expression", {"situation": "a", "style": "活泼", "source_id": 1}),
        ])
        refs = await repo.list_learner_records(CK, "expression")
        llm = FakeLLM(content='{"categorization": "rejected", "confidence": 1.0}')
        pipeline = LearnerPipeline(
            repo, llm, PromptStore(), VirtualClock(epoch=100.0), validators=VALIDATORS,
        )
        scheduler = LearnerScheduler(
            repo, pipeline, {"effect": EFFECT_SPEC}, VirtualClock(epoch=100.0),
        )
        scheduler.note_exposure(CK, refs, MessageRowId(2))
        scheduler.mark_delivered(CK)
        await seed_messages(repo, n=1, prefix="followup")
        await scheduler._run_one((CK, "effect"))
        rec = (await repo.list_learner_records(CK, "expression"))[0]
        await repo.close()
        return rec.weight

    # rejected + confidence 1.0 → delta -0.4 (the rejected band is
    # [-1.0, -0.4]) → weight 1.0 * 0.6 = 0.6
    assert run(scenario()) == 0.6


def test_effect_no_adoption_no_reweight(tmp_path):
    """A malformed effect judgment settles the run WITHOUT advancing the
    watermark and WITHOUT reweighting the references."""
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        await commit_records(repo, "expression", [
            make_record("expression", {"situation": "a", "style": "活泼", "source_id": 1}),
        ])
        refs = await repo.list_learner_records(CK, "expression")
        llm = FakeLLM(content="not json at all")
        pipeline = LearnerPipeline(
            repo, llm, PromptStore(), VirtualClock(epoch=100.0), validators=VALIDATORS,
        )
        scheduler = LearnerScheduler(
            repo, pipeline, {"effect": EFFECT_SPEC}, VirtualClock(epoch=100.0),
        )
        scheduler.note_exposure(CK, refs, MessageRowId(2))
        scheduler.mark_delivered(CK)
        await seed_messages(repo, n=1, prefix="followup")
        await scheduler._run_one((CK, "effect"))
        rec = (await repo.list_learner_records(CK, "expression"))[0]
        # The refs stay pending (the run settled malformed, no reweight).
        assert scheduler._effect_refs.get(CK) is not None
        await repo.close()
        return rec.weight

    assert run(scenario()) == 1.0


# ── exposure: post-terminal, LIVE-only, idempotent ───────────────────────────

def test_exposure_and_uses_are_idempotent(tmp_path):
    """The App's exposure callback records each (record, producing learner
    run) exposure exactly once; the uses bump rides on the FIRST exposure
    only (the exposure's run_id references a real learner_runs row)."""
    from pretender.app import App
    from pretender.config import Config

    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_messages(repo, n=2)
        await commit_records(repo, "expression", [
            make_record("expression", {"situation": "a", "style": "活泼", "source_id": 1}),
        ])
        records = tuple(await repo.list_learner_records(CK, "expression"))
        app = App(Config(), repo=repo, clock=VirtualClock(epoch=100.0), dry_run=False)
        # First exposure: created + uses bumped.
        await app._on_exposure(CK, records, dispatch_id=7, through_msg_id=MessageRowId(2))
        # Duplicate (same record, same producing run): idempotent — no new
        # exposure, no uses bump.
        await app._on_exposure(CK, records, dispatch_id=8, through_msg_id=MessageRowId(2))
        rec = (await repo.list_learner_records(CK, "expression"))[0]
        count = await db.read(
            lambda c: c.execute(
                "SELECT COUNT(*) FROM record_exposures"
            ).fetchone()[0]
        )
        await repo.close()
        return rec.uses, count

    uses, count = run(scenario())
    # Exposure is not effect confirmation: uses advance only after a durable
    # delivered/effect confirmation event.
    assert uses == 0
    assert count == 0  # exposure waits for durable delivery confirmation


# ── dry-run / replay: zero adaptive writes ───────────────────────────────────

def test_learner_worker_never_starts_in_dry_run(tmp_path):
    """The App's learner worker is LIVE-only: dry-run never schedules it."""
    from pretender.app import App
    from pretender.config import Config

    async def scenario():
        cfg = Config.from_dict({
            "storage": {"db_path": str(tmp_path / "app.db")},
            "learn": {"enabled": True, "profiles": {"expression": {}}},
        })
        app = App.build(cfg, dry_run=True)
        await app.start()
        assert app._learner is not None  # built, but never started
        assert app._learner_task is None
        await app.shutdown()
        return app._learner_task

    assert run(scenario()) is None
