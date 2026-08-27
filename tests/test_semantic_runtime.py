"""Phase 5 Gate 5 semantic-runtime: the bounded semantic backfill worker,
the App's optional-embed wiring, budget-reserved backfill, building
interruption/restart, cross-chat isolation, vector direct-mutation
visibility, and clean shutdown.

Focused end-to-end tests over real SQLite + VirtualClock. The semantic lane
is OPTIONAL: absent an embed profile/revision/active generation it performs
ZERO embed calls and recall is FTS-only.
"""

from __future__ import annotations

import asyncio
import io

from pretender.app import App, SemanticBackfill
from pretender.adapters.console import ConsoleAdapter
from pretender.budget import BudgetConfig, BudgetManager
from pretender.clock import VirtualClock
from pretender.config import Config, LLMConfig, LLMProfile
from pretender.embed import OptionalEmbeddingService
from pretender.memory import DEFAULT_SOURCE_TAIL
from pretender.search import MemorySearch
from pretender.types import (
    ChatKey,
    MemoryRecord,
    MemoryWriteRequest,
    MessageRowId,
)
from tests.durable_helpers import CK, make_identity, make_message, open_repo_with_chat, run
from tests.knowledge_helpers import OTHER, make_vector


class FixedEmbedder:
    """Returns a fixed vector for every text; records call count."""

    def __init__(self, vector):
        self.vector = vector
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return [self.vector for _ in texts]


def embed_service(vector, *, space_id="e@r1"):
    return OptionalEmbeddingService(FixedEmbedder(vector), space_id=space_id)


async def seed_memories(repo, chat_key, texts, prefix="m"):
    """Seed one message per text and CAS-commit one memory each; returns the
    memory ids in order."""
    ids = []
    for i, text in enumerate(texts, start=1):
        await repo.ingest_message(
            None,
            make_message(
                chat_key=chat_key, msg_id=f"{prefix}{i}", text=f"src {i}",
                recv_ts=1_700_000_000.0 + i,
            ),
        )
        max_id = await repo._db.read(
            lambda c: c.execute(
                "SELECT COALESCE(MAX(id), 0) FROM messages WHERE chat_key = ?",
                (chat_key,),
            ).fetchone()[0]
        )
        batch = await repo.read_memory_source_batch(
            chat_key, through_msg_id=MessageRowId(max_id), tail=DEFAULT_SOURCE_TAIL
        )
        assert batch is not None
        rec = MemoryRecord(
            chat_key=chat_key, text=text,
            source_first_msg_id=batch.first_msg_id,
            source_last_msg_id=batch.last_msg_id,
            source_hash=batch.source_hash,
        )
        expected = await repo.get_memory_watermark(chat_key)
        assert await repo.commit_memory_source(
            MemoryWriteRequest(
                chat_key=chat_key, batch=batch, records=(rec,),
                expected_through_msg_id=expected,
            )
        ) is True
        mid = await repo._db.read(
            lambda c: c.execute(
                "SELECT id FROM memories WHERE chat_key = ? ORDER BY id DESC LIMIT 1",
                (chat_key,),
            ).fetchone()[0]
        )
        ids.append(mid)
    return ids


def make_worker(repo, embed, *, space_id="e@r1", model="e", revision="r1",
                cap=100, epoch=100.0):
    budget = BudgetManager(repo, BudgetConfig(daily_cap=cap), now=lambda: epoch)
    return SemanticBackfill(
        repo, embed, budget, model=model, revision=revision, space_id=space_id
    )


async def wait_for_worker_built(worker, task, *, timeout=5.0):
    """Wait for readiness with enough state to diagnose a stuck worker."""
    try:
        await asyncio.wait_for(worker._built.wait(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        generations = await worker._repo.list_embedding_generations()
        if task.cancelled():
            worker_exception = "CancelledError"
        elif task.done():
            worker_exception = repr(task.exception())
        else:
            worker_exception = None
        raise AssertionError(
            "semantic worker did not build within "
            f"{timeout:.1f}s: "
            f"queue={list(worker._queue._queue)!r}, "
            f"queued_chats={sorted(worker._queued)!r}, "
            "generation_states="
            f"{[(g.id, g.space_id, g.state) for g in generations]!r}, "
            f"worker_done={task.done()}, "
            f"worker_exception={worker_exception}"
        ) from exc


async def cancel_and_await_worker(task):
    """Always drain a spawned worker, including after a failed wait."""
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


def make_app(tmp_path, cfg, embed_service):
    clock = VirtualClock()
    adapter = ConsoleAdapter(
        clock=clock, input_stream=io.StringIO(""), output_stream=io.StringIO()
    )
    return App.build(cfg, clock=clock, adapter=adapter, embed_service=embed_service)


# ── full backfill builds an active generation; semantic query works ──────────

def test_backfill_builds_active_generation_and_semantic_query_works(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_memories(repo, CK, ["apple pie", "banana split"])
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder, space_id="e@r1")
        await make_worker(repo, embed)._full_backfill()
        gens = await repo.list_embedding_generations()
        active = [g for g in gens if g.state == "active"]
        search = MemorySearch(repo, embed=embed)
        hits = await search.search(CK, "apple", limit=10)
        await repo.close()
        return gens, active, hits

    gens, active, hits = run(scenario())
    assert len(active) == 1
    assert active[0].space_id == "e@r1"
    assert active[0].dim == 3
    # Semantic recall now works (hybrid/semantic hits, not FTS-only).
    assert any(h.source in ("semantic", "hybrid") for h in hits)


def test_active_generation_startup_scan_repairs_missing_committed_memory(tmp_path):
    """An active generation is derived state, not proof that a crash between
    memory commit and advisory enqueue cannot leave coverage stale."""

    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        first = await seed_memories(repo, CK, ["apple pie"])
        embed = embed_service([1.0, 0.0, 0.0])
        worker = make_worker(repo, embed)
        assert await worker._full_backfill() is True
        second = await seed_memories(repo, CK, ["banana split"], prefix="later")
        assert await worker._full_backfill() is True
        active = next(g for g in await repo.list_embedding_generations() if g.state == "active")
        assert active.id is not None
        rows = await repo.list_vectors(CK, "e", active.id)
        await repo.close()
        return first, second, rows

    first, second, rows = run(scenario())
    assert {row.owner_id for row in rows} == set(first + second)


# ── restart resumes a partially-built generation; old generation never mixes ─

def test_backfill_restart_resumes_and_old_generation_never_mixes(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        ids = await seed_memories(repo, CK, ["apple pie", "banana split"])
        # Old ACTIVE generation G1 (space e@r1) with a vector.
        g1 = await repo.create_embedding_generation("e", 3, revision="r1")
        assert g1.id is not None
        await repo.activate_embedding_generation(g1.id)
        await repo.upsert_vector(
            CK, make_vector(owner_id=ids[0], generation=g1.id, model="e", dim=3,
                            values=(1.0, 0.0, 0.0))
        )
        # A partially-built generation G2 (space e@r2): created in the
        # ``building`` state, one vector written, NOT activated (an
        # interrupted build).
        g2 = await repo.create_embedding_generation(
            "e", 3, revision="r2", state="building"
        )
        assert g2.id is not None
        await repo.upsert_vector(
            CK, make_vector(owner_id=ids[0], generation=g2.id, model="e", dim=3,
                            values=(0.0, 1.0, 0.0))
        )
        # Restart the backfill for e@r2: resumes G2, completes, activates.
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder, space_id="e@r2")
        await make_worker(repo, embed, space_id="e@r2", revision="r2")._full_backfill()
        gens = await repo.list_embedding_generations()
        g1_after = next(g for g in gens if g.id == g1.id)
        g2_after = next(g for g in gens if g.id == g2.id)
        # G2 active, G1 inactive (old generation preserved, never mixed).
        assert g2_after.state == "active"
        assert g1_after.state == "inactive"
        # G2 holds vectors for BOTH memories.
        v = await repo.list_vectors(CK, "e", g2.id)
        assert {r.owner_id for r in v} == set(ids)
        await repo.close()

    run(scenario())


# ── the generation is literally ``building`` in the DB before activation ─────

def test_backfill_generation_is_literally_building_before_activation(tmp_path):
    """The worker persists the generation in the ``building`` state and only
    activates it after a complete scan — the DB row is literally building
    before activation, not just a local notion."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_memories(repo, CK, ["apple pie", "banana split"])
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder, space_id="e@r1")
        worker = make_worker(repo, embed)
        # Ensure the building generation but interrupt BEFORE the scan: the
        # durable row must be literally 'building'.
        gen = await worker._ensure_building_generation(
            await repo.list_memory_chats()
        )
        assert gen is not None and gen.id is not None
        db_state = await repo._db.read(
            lambda c: c.execute(
                "SELECT state FROM embedding_generations WHERE id = ?",
                (gen.id,),
            ).fetchone()[0]
        )
        # Complete the scan: the SAME generation is activated only now.
        await worker._full_backfill()
        active = [g for g in await repo.list_embedding_generations()
                  if g.state == "active"]
        await repo.close()
        return gen.id, db_state, active

    gen_id, db_state, active = run(scenario())
    assert db_state == "building"  # literally building in the DB pre-activation
    assert len(active) == 1 and active[0].id == gen_id


# ── restart resumes the SAME building generation (id preserved) ──────────────

def test_backfill_restart_resumes_existing_building_generation(tmp_path):
    """A restart resumes the SAME matching ``building`` generation (id
    preserved) instead of creating a new one."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_memories(repo, CK, ["apple pie"])
        g2 = await repo.create_embedding_generation(
            "e", 3, revision="r2", state="building"
        )
        assert g2.id is not None
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder, space_id="e@r2")
        worker = make_worker(repo, embed, space_id="e@r2", revision="r2")
        resumed = await worker._ensure_building_generation(
            await repo.list_memory_chats()
        )
        before = await repo.list_embedding_generations()
        await worker._full_backfill()
        gens = await repo.list_embedding_generations()
        await repo.close()
        return g2.id, resumed, before, gens

    g2_id, resumed, before, gens = run(scenario())
    assert resumed is not None and resumed.id == g2_id  # same building gen
    assert len(before) == 1  # no new generation created for the resume
    active = [g for g in gens if g.state == "active"]
    assert len(active) == 1 and active[0].id == g2_id  # resumed then activated


# ── a manual/legacy inactive same-space generation is never hijacked ─────────

def test_backfill_never_hijacks_manual_inactive_same_space(tmp_path):
    """A manual/legacy INACTIVE generation in the same space is never
    treated as in-progress: the worker does not write into it, does not
    activate it, and its vectors never mix into the build."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        ids = await seed_memories(repo, CK, ["apple pie", "banana split"])
        # A manual inactive generation in the SAME space (e@r1), with one
        # vector — an unrelated/legacy row the worker must not hijack.
        manual = await repo.create_embedding_generation("e", 3, revision="r1")
        assert manual.state == "inactive"
        assert manual.id is not None
        await repo.upsert_vector(
            CK, make_vector(owner_id=ids[0], generation=manual.id, model="e", dim=3,
                            values=(1.0, 0.0, 0.0))
        )
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder, space_id="e@r1")
        worker = make_worker(repo, embed)
        # No building generation exists, so nothing is ensured: the manual
        # inactive generation is left alone (never treated as in-progress).
        ensured = await worker._ensure_building_generation(
            await repo.list_memory_chats()
        )
        await worker._full_backfill()
        gens = await repo.list_embedding_generations()
        manual_after = next(g for g in gens if g.id == manual.id)
        await repo.close()
        return ensured, manual_after, gens

    ensured, manual_after, gens = run(scenario())
    assert ensured is None  # never resumed an inactive generation
    assert manual_after.state == "inactive"  # never activated
    assert not any(g.state == "active" for g in gens)  # nothing hijacked
    assert len(gens) == 1  # no new generation created over the same space


# ── cache hits use zero extra provider calls ─────────────────────────────────

def test_backfill_cache_hits_zero_extra_calls(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_memories(repo, CK, ["apple pie", "banana split"])
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder, space_id="e@r1")
        # Pre-populate the cache (probe + both memory texts).
        await embed.embed(["pretender semantic probe", "apple pie", "banana split"])
        calls_before = embedder.calls
        await make_worker(repo, embed)._full_backfill()
        # All texts (and the probe) were cache hits: ZERO extra provider calls.
        assert embedder.calls == calls_before
        # But vectors were still written into an active generation.
        gens = await repo.list_embedding_generations()
        active = [g for g in gens if g.state == "active"]
        assert len(active) == 1
        assert active[0].id is not None
        assert len(await repo.list_vectors(CK, "e", active[0].id)) == 2
        await repo.close()

    run(scenario())


# ── blocked budget degrades to FTS-only (no vectors written) ─────────────────

def test_backfill_blocked_budget_degrades_to_fts_only(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_memories(repo, CK, ["apple pie"])
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder, space_id="e@r1")
        budget = BudgetManager(repo, BudgetConfig(daily_cap=1), now=lambda: 100.0)
        # Pre-reserve the chat's budget so the worker's reservation is blocked.
        await budget.record(CK, calls=1)
        worker = SemanticBackfill(
            repo, embed, budget, model="e", revision="r1", space_id="e@r1"
        )
        await worker._full_backfill()
        gens = await repo.list_embedding_generations()
        active = [g for g in gens if g.state == "active"]
        search = MemorySearch(repo, embed=embed)
        hits = await search.search(CK, "apple", limit=10)
        await repo.close()
        return active, hits

    active, hits = run(scenario())
    assert active == []  # never activated
    assert all(h.source == "lexical" for h in hits)  # FTS-only


# ── cross-chat backfill isolation ────────────────────────────────────────────

def test_backfill_cross_chat_isolation(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        ids_ck = await seed_memories(repo, CK, ["apple pie"])
        ids_other = await seed_memories(repo, OTHER, ["apple secret"], prefix="o")
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder, space_id="e@r1")
        await make_worker(repo, embed)._full_backfill()
        gens = await repo.list_embedding_generations()
        active = [g for g in gens if g.state == "active"][0]
        assert active.id is not None
        ck_rows = await repo.list_vectors(CK, "e", active.id)
        other_rows = await repo.list_vectors(OTHER, "e", active.id)
        await repo.close()
        return ids_ck, ids_other, ck_rows, other_rows

    ids_ck, ids_other, ck_rows, other_rows = run(scenario())
    assert {r.owner_id for r in ck_rows} == set(ids_ck)
    assert {r.owner_id for r in other_rows} == set(ids_other)
    assert not (set(ids_ck) & set(ids_other))  # distinct owners, no mixing


# ── vector direct-mutation visibility (v8 vector_revision) ───────────────────

def test_vector_direct_mutation_visible_on_next_search(tmp_path):
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        ids = await seed_memories(repo, CK, ["apple pie", "banana split"])
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder, space_id="e@r1")
        await make_worker(repo, embed)._full_backfill()
        gens = await repo.list_embedding_generations()
        active = [g for g in gens if g.state == "active"][0]
        assert active.id is not None
        search = MemorySearch(repo, embed=embed)
        before = await search.search(CK, "apple", limit=10)
        # Directly mutate a vector (bypassing the index): the durable
        # vector_revision bumps, so the next search reloads and sees it.
        await repo.upsert_vector(
            CK, make_vector(owner_id=ids[1], generation=active.id, model="e", dim=3,
                            values=(0.0, 0.0, 1.0))
        )
        after = await search.search(CK, "apple", limit=10)
        await repo.close()
        return before, after, ids

    before, after, ids = run(scenario())
    # The direct mutation is visible: memory 2's semantic score changed (its
    # vector was rewritten), proving the next search reloaded via the durable
    # vector_revision instead of serving a stale cached index.
    before2 = next(h for h in before if h.memory_id == ids[1])
    after2 = next(h for h in after if h.memory_id == ids[1])
    assert before2.semantic_score != after2.semantic_score


# ── App wiring: configured embed constructs the correct space ────────────────

def test_app_configured_embed_constructs_correct_space(tmp_path):
    """App.build turns a configured embed profile with a revision into a real
    OptionalEmbeddingService bound to the canonical space_id."""
    cfg = Config.from_dict({
        "storage": {"db_path": str(tmp_path / "data" / "app.db")},
        "llm": {"profiles": {"embed": {"model": "e", "revision": "r1"}}},
    })
    embed = embed_service([1.0, 0.0, 0.0], space_id="e@r1")
    app = make_app(tmp_path, cfg, embed)
    assert app.memory_search is not None
    assert app.memory_search._embed is not None
    assert app.memory_search._embed.space_id == "e@r1"
    assert app._semantic_backfill is not None
    assert app._semantic_backfill._space_id == "e@r1"


def test_app_real_config_embed_constructs_disk_cache_without_injection(tmp_path):
    """The production profile path must construct its persistent cache before
    any provider call; an injected embed seam must not hide its constructor."""

    async def scenario():
        cfg = Config.from_dict({
            "storage": {"db_path": str(tmp_path / "data" / "app.db")},
            "llm": {"profiles": {"embed": {"model": "e", "revision": "r1"}}},
        })
        app = App.build(cfg, clock=VirtualClock())
        try:
            assert app._embed_service is not None
            assert app._embed_service.space_id == "e@r1"
            return app._embed_service._cache._path
        finally:
            await app.shutdown()

    cache_path = run(scenario())
    assert cache_path == tmp_path / "data" / "embed_cache"


def test_app_no_embed_profile_fts_only(tmp_path):
    """No embed profile -> FTS-only MemorySearch, ZERO embed calls."""
    cfg = Config.from_dict({"storage": {"db_path": str(tmp_path / "data" / "app.db")}})
    app = make_app(tmp_path, cfg, None)
    assert app.memory_search is not None and app.memory_search._embed is None
    assert app._semantic_backfill is None


def test_app_embed_profile_no_revision_fts_only(tmp_path):
    """An embed profile WITHOUT a revision -> FTS-only (no canonical space)."""
    cfg = Config.from_dict({
        "storage": {"db_path": str(tmp_path / "data" / "app.db")},
        "llm": {"profiles": {"embed": {"model": "e"}}},
    })
    app = make_app(tmp_path, cfg, None)
    assert app.memory_search is not None and app.memory_search._embed is None
    assert app._semantic_backfill is None


# ── App end-to-end: startup backfill -> semantic query works ─────────────────

def test_app_startup_backfill_then_semantic_query_works(tmp_path):
    async def scenario():
        db_path = tmp_path / "data" / "app.db"
        # Seed memories via a separate connection first.
        from pretender.db import Database
        from pretender.repo import SqliteRepository
        db = Database(db_path)
        await db.open()
        repo = SqliteRepository(db)
        await repo.upsert_chat(make_identity())
        await seed_memories(repo, CK, ["apple pie", "banana split"])
        await db.close()

        cfg = Config.from_dict({
            "storage": {"db_path": str(db_path)},
            "llm": {"profiles": {"embed": {"model": "e", "revision": "r1"}}},
        })
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder, space_id="e@r1")
        app = make_app(tmp_path, cfg, embed)
        task = None
        try:
            await app.start()
            # The worker task drains the queue indefinitely; readiness is
            # signaled separately by the built event.
            assert app._semantic_backfill is not None
            task = app._semantic_task
            assert task is not None
            await wait_for_worker_built(app._semantic_backfill, task)
            assert app.memory_search is not None
            hits = await app.memory_search.search(CK, "apple", limit=10)
            return hits
        finally:
            if task is not None:
                await cancel_and_await_worker(task)
            await app.shutdown()

    hits = run(scenario())
    assert any(h.source in ("semantic", "hybrid") for h in hits)


# ── shutdown cancels the semantic worker cleanly ─────────────────────────────

def test_app_shutdown_cancels_semantic_worker_cleanly(tmp_path):
    """Shutdown cancels/drains the semantic worker before the shared LLM/DB
    close — no error, no leaked task."""
    async def scenario():
        db_path = tmp_path / "data" / "app.db"
        from pretender.db import Database
        from pretender.repo import SqliteRepository
        db = Database(db_path)
        await db.open()
        repo = SqliteRepository(db)
        await repo.upsert_chat(make_identity())
        await seed_memories(repo, CK, ["apple pie"])
        await db.close()

        cfg = Config.from_dict({
            "storage": {"db_path": str(db_path)},
            "llm": {"profiles": {"embed": {"model": "e", "revision": "r1"}}},
        })
        embed = embed_service([1.0, 0.0, 0.0], space_id="e@r1")
        app = make_app(tmp_path, cfg, embed)
        await app.start()
        assert app._semantic_task is not None
        await app.shutdown()
        return app._semantic_task

    task = run(scenario())
    assert task is None  # cleared on shutdown


# ── terminal settle path never awaits provider work ──────────────────────────

def test_terminal_settle_path_no_provider_call(tmp_path):
    """The terminal on_memory callback only ENQUEUES backfill work — it never
    awaits provider work, so the settle path performs ZERO embed calls."""
    async def scenario():
        db_path = tmp_path / "data" / "app.db"
        cfg = Config.from_dict({
            "storage": {"db_path": str(db_path)},
            "llm": {"profiles": {"embed": {"model": "e", "revision": "r1"}}},
        })
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder, space_id="e@r1")
        app = make_app(tmp_path, cfg, embed)
        await app.db.open()
        await app.repo.upsert_chat(make_identity())
        await seed_memories(app.repo, CK, ["apple pie"])
        # The terminal settle path (on_memory_default) only enqueues: it
        # never awaits provider work, so ZERO embed calls happen here.
        await app._on_memory_default(CK, MessageRowId(1))
        calls_after = embedder.calls
        assert calls_after == 0  # no provider call in the settle path
        # The chat was enqueued for the background worker (non-blocking).
        assert app._semantic_backfill is not None
        assert not app._semantic_backfill._queue.empty()
        await app.shutdown()
        return calls_after

    run(scenario())


# ── Gate 5 remediation: exact reservation + distinct-manager atomic cap ──────

def test_backfill_reserves_exact_batch_count_and_distinct_managers_share_cap(tmp_path):
    """The backfill reserves exactly ceil(cache-miss texts / batch_size)
    calls against the chat's budget (one per provider batch), and DISTINCT
    BudgetManager instances over the same DB (BudgetStore) share the atomic
    cap — so simultaneous planner/embed reservations can never exceed it."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        # 5 memories -> with batch_size=2, ceil(5/2)=3 provider calls.
        await seed_memories(repo, CK, [f"text {i}" for i in range(5)])
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder, space_id="e@r1", batch_size=2)
        budget = BudgetManager(repo, BudgetConfig(daily_cap=100), now=lambda: 100.0)
        worker = SemanticBackfill(
            repo, embed, budget, model="e", revision="r1", space_id="e@r1"
        )
        await worker._full_backfill()
        usage = await budget.snapshot(CK)
        # A DISTINCT manager over the same DB sees the same atomic usage.
        budget2 = BudgetManager(repo, BudgetConfig(daily_cap=100), now=lambda: 100.0)
        usage2 = await budget2.snapshot(CK)
        await repo.close()
        return usage, usage2

    usage, usage2 = run(scenario())
    assert usage.calls == 3  # exactly ceil(5/2) provider calls reserved
    assert usage2.calls == 3  # distinct manager shares the atomic cap


# ── Gate 5 remediation: blocked build stays building; restart builds ─────────

def test_backfill_blocked_chat_stays_building_and_restart_builds(tmp_path):
    """A blocked chat leaves the generation literally building (never
    activated); a restart with the chat unblocked resumes the SAME building
    generation and activates it."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        await seed_memories(repo, CK, ["apple pie"])
        await seed_memories(repo, OTHER, ["banana split"], prefix="o")
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder, space_id="e@r1")
        # Per-chat budget: CK available, OTHER blocked (cap reached).
        cfg = Config.from_dict({
            "budget": {"daily_cap": 100},
            "chats": [{"key": str(OTHER), "budget": {"daily_cap": 1}}],
        })
        other_budget = BudgetManager(
            repo, cfg.for_chat(OTHER).budget, now=lambda: 100.0
        )
        await other_budget.record(OTHER, calls=1)  # block OTHER
        worker = SemanticBackfill(
            repo, embed, BudgetManager(repo, cfg.budget, now=lambda: 100.0),
            model="e", revision="r1", space_id="e@r1", cfg=cfg, now=lambda: 100.0,
        )
        built = await worker._full_backfill()
        gens = await repo.list_embedding_generations()
        building = [g for g in gens if g.state == "building"]
        active = [g for g in gens if g.state == "active"]
        assert built is False
        assert len(building) == 1  # literally building, never activated
        assert active == []
        # Restart with OTHER unblocked: resumes the SAME building generation.
        cfg2 = Config.from_dict({"budget": {"daily_cap": 100}})
        worker2 = SemanticBackfill(
            repo, embed, BudgetManager(repo, cfg2.budget, now=lambda: 100.0),
            model="e", revision="r1", space_id="e@r1", cfg=cfg2, now=lambda: 100.0,
        )
        built2 = await worker2._full_backfill()
        gens2 = await repo.list_embedding_generations()
        active2 = [g for g in gens2 if g.state == "active"]
        await repo.close()
        return built, building, active, built2, active2

    built, building, active, built2, active2 = run(scenario())
    assert built is False and built2 is True
    assert len(building) == 1 and active == []
    assert len(active2) == 1 and active2[0].id == building[0].id


# ── Gate 5 remediation: exception build stays building ───────────────────────

def test_backfill_exception_stays_building(tmp_path):
    """An exception during a chat's backfill leaves the generation literally
    building (never activated); the embed service contains it and a restart
    resumes."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key=OTHER))
        await seed_memories(repo, CK, ["apple pie"])
        await seed_memories(repo, OTHER, ["banana split"], prefix="o")

        class ExplodingEmbedder:
            def __init__(self):
                self.calls = 0

            async def embed(self, texts):
                self.calls += 1
                if self.calls >= 2:  # OTHER's backfill raises
                    raise RuntimeError("boom")
                return [[1.0, 0.0, 0.0] for _ in texts]

        embedder = ExplodingEmbedder()
        embed = OptionalEmbeddingService(embedder, space_id="e@r1")
        worker = make_worker(repo, embed)
        built = await worker._full_backfill()
        gens = await repo.list_embedding_generations()
        building = [g for g in gens if g.state == "building"]
        active = [g for g in gens if g.state == "active"]
        await repo.close()
        return built, building, active

    built, building, active = run(scenario())
    assert built is False
    assert len(building) == 1  # literally building, never activated
    assert active == []


# ── Gate 5 remediation: new first memory after zero-memory startup builds ────

def test_new_first_memory_after_zero_startup_builds(tmp_path):
    """New memory enqueued after a startup with zero memories kicks off a new
    building/full scan (not a no-op until restart)."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder, space_id="e@r1")
        worker = make_worker(repo, embed)
        # Startup with zero memories: nothing to build.
        built = await worker._full_backfill()
        assert built is False
        assert await repo.list_embedding_generations() == []
        # New memory arrives; the enqueued chat kicks off a full scan.
        await seed_memories(repo, CK, ["apple pie"])
        await worker._backfill_chat(CK)
        gens = await repo.list_embedding_generations()
        active = [g for g in gens if g.state == "active"]
        await repo.close()
        return built, active

    built, active = run(scenario())
    assert built is False
    assert len(active) == 1  # the new first memory built an active generation


# ── Gate 5 remediation: backlog larger than one batch drains fully ───────────

def test_backlog_more_than_one_batch_drains(tmp_path):
    """The terminal on_memory callback drains a backlog larger than one source
    batch fully — the durable watermark catches the cursor (no provider work,
    no ledger/outbox mutation)."""
    async def scenario():
        db_path = tmp_path / "data" / "app.db"
        cfg = Config.from_dict({"storage": {"db_path": str(db_path)}})
        app = make_app(tmp_path, cfg, None)
        await app.db.open()
        await app.repo.upsert_chat(make_identity())
        # Ingest just over one source batch WITHOUT summarizing: the cursor
        # advances but the memory watermark stays behind.
        for i in range(DEFAULT_SOURCE_TAIL + 1):
            await app.repo.ingest_message(
                None,
                make_message(
                    chat_key=CK, msg_id=f"m{i}", text=f"text {i}",
                    recv_ts=1_700_000_000.0 + i,
                ),
            )
        max_id = await app.repo._db.read(
            lambda c: c.execute(
                "SELECT COALESCE(MAX(id),0) FROM messages WHERE chat_key = ?",
                (CK,),
            ).fetchone()[0]
        )
        await app._on_memory_default(CK, MessageRowId(max_id))
        watermark = await app.repo.get_memory_watermark(CK)
        await app.shutdown()
        return watermark, max_id

    watermark, max_id = run(scenario())
    assert watermark == MessageRowId(max_id)  # fully drained to the cursor


# ── Gate 5 remediation: semantic query embeds reserve budget ─────────────────

def test_semantic_query_embed_reserves_budget_and_cache_hits_zero(tmp_path):
    """A semantic QUERY embed reserves exactly one call under the chat's
    budget (a cache hit consumes zero), and DISTINCT managers over the same
    DB share the atomic cap."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_memories(repo, CK, ["apple pie", "banana split"])
        embedder = FixedEmbedder([1.0, 0.0, 0.0])
        embed = OptionalEmbeddingService(embedder, space_id="e@r1")
        await make_worker(repo, embed)._full_backfill()
        budget = BudgetManager(repo, BudgetConfig(daily_cap=100), now=lambda: 100.0)
        search = MemorySearch(repo, embed=embed, budget_for=lambda chat: budget)
        # First query: cache miss -> exactly 1 call reserved (on top of the
        # backfill's dimension-probe reservation of 1).
        await search.search(CK, "apple", limit=10)
        usage = await budget.snapshot(CK)
        assert usage.calls == 2
        # Second query with the SAME text: cache hit -> zero additional calls.
        await search.search(CK, "apple", limit=10)
        usage2 = await budget.snapshot(CK)
        assert usage2.calls == 2
        # A DISTINCT manager over the same DB sees the same atomic usage.
        budget2 = BudgetManager(repo, BudgetConfig(daily_cap=100), now=lambda: 100.0)
        usage3 = await budget2.snapshot(CK)
        await repo.close()
        return usage, usage2, usage3

    usage, usage2, usage3 = run(scenario())
    assert usage.calls == 2
    assert usage2.calls == 2
    assert usage3.calls == 2


# ── Gate 5 remediation: source-fenced activation race ────────────────────────

def test_backfill_activation_race_fails_and_enqueues_repair(tmp_path):
    """A memory committed during the final scan (between the scan and the
    activation transaction) makes activation fail with the repair set; the
    worker enqueues the affected chat and a subsequent backfill completes
    the build — a stale generation is never falsely activated."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await seed_memories(repo, CK, ["apple pie"])
        embed = embed_service([1.0, 0.0, 0.0])
        worker = make_worker(repo, embed)
        real = repo.activate_embedding_generation_if_complete
        raced = False

        async def racing_activation(generation_id):
            nonlocal raced
            if not raced:
                raced = True
                # A memory committed during the final scan: the activation
                # transaction must see it and fail the source fence.
                await seed_memories(repo, CK, ["banana split"], prefix="late")
            return await real(generation_id)

        repo.activate_embedding_generation_if_complete = racing_activation
        built = await worker._full_backfill()
        gens = await repo.list_embedding_generations()
        building = [g for g in gens if g.state == "building"]
        active = [g for g in gens if g.state == "active"]
        queued = list(worker._queue._queue)
        # Resume repair: the enqueued chat's backfill completes the build.
        await worker._backfill_chat(CK)
        gens2 = await repo.list_embedding_generations()
        active2 = [g for g in gens2 if g.state == "active"]
        rows = await repo.list_vectors(CK, "e", active2[0].id)
        await repo.close()
        return built, building, active, queued, active2, len(rows)

    built, building, active, queued, active2, n = run(scenario())
    assert built is False  # activation failed: the fence caught the commit
    assert len(building) == 1 and active == []  # stays building, never active
    assert CK in queued  # the affected chat was enqueued for repair
    assert len(active2) == 1  # the repair completed the build
    assert n == 2  # both memories covered


def test_backfill_preserves_active_generation_until_build_completes(tmp_path):
    """The current ACTIVE generation stays active (searchable) while a new
    building generation is being populated; it is deactivated only when the
    building generation completes (at-most-one-active, same transaction)."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        ids = await seed_memories(repo, CK, ["apple pie", "banana split"])
        # G1 active with vectors (the current generation).
        g1 = await repo.create_embedding_generation("e", 3, revision="r1")
        assert g1.id is not None
        await repo.activate_embedding_generation(g1.id)
        for mid in ids:
            await repo.upsert_vector(
                CK, make_vector(owner_id=mid, generation=g1.id, model="e", dim=3,
                                values=(1.0, 0.0, 0.0))
            )
        # An interrupted build for a NEW space (e@r2): G1 must stay active.
        g2 = await repo.create_embedding_generation(
            "e", 3, revision="r2", state="building"
        )
        assert g2.id is not None
        gens_mid = await repo.list_embedding_generations()
        g1_mid = next(g for g in gens_mid if g.id == g1.id)
        g2_mid = next(g for g in gens_mid if g.id == g2.id)
        # Complete the build: G2 activates, G1 deactivates in the SAME
        # transaction (the source-fenced activation).
        embed = embed_service([1.0, 0.0, 0.0], space_id="e@r2")
        worker = make_worker(repo, embed, space_id="e@r2", revision="r2")
        built = await worker._full_backfill()
        gens_after = await repo.list_embedding_generations()
        g1_after = next(g for g in gens_after if g.id == g1.id)
        g2_after = next(g for g in gens_after if g.id == g2.id)
        await repo.close()
        return g1_mid, g2_mid, built, g1_after, g2_after

    g1_mid, g2_mid, built, g1_after, g2_after = run(scenario())
    assert g1_mid.state == "active" and g2_mid.state == "building"
    assert built is True
    assert g2_after.state == "active" and g1_after.state == "inactive"


# ── Gate 5 remediation: keyset-paged full backfill (fair, cancellable) ───────

def test_backfill_chat_larger_than_one_page_fully_covers(tmp_path):
    """A chat with more memories than one bounded page is fully covered:
    the worker processes one page per round (re-enqueuing for fair
    continuation) and activates only after every page is covered."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        page_size = SemanticBackfill._MEMORY_PAGE
        await seed_memories(repo, CK, [f"text {i}" for i in range(page_size + 1)])
        embed = embed_service([1.0, 0.0, 0.0])
        worker = make_worker(repo, embed)
        # The first round processes ONE page and re-enqueues the chat.
        built = await worker._full_backfill()
        queued = list(worker._queue._queue)
        # Drain the queue to completion (the worker's own loop).
        task = asyncio.create_task(worker.run())
        try:
            await wait_for_worker_built(worker, task)
            gens = await repo.list_embedding_generations()
            active = [g for g in gens if g.state == "active"]
            rows = await repo.list_vectors(CK, "e", active[0].id)
            return built, queued, active, len(rows)
        finally:
            await cancel_and_await_worker(task)
            await repo.close()

    built, queued, active, n = run(scenario())
    assert built is False  # pending pages: not active after one round
    assert CK in queued  # the chat was re-enqueued for continuation
    assert len(active) == 1  # eventually active
    assert n == SemanticBackfill._MEMORY_PAGE + 1  # every page covered


def test_backfill_pages_interleave_across_chats_fairly(tmp_path):
    """One bounded page per chat per round: a multi-page chat never
    monopolizes the scan — both chats advance one page per round."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        page_size = SemanticBackfill._MEMORY_PAGE
        await repo.upsert_chat(make_identity(chat_key=str(OTHER)))
        await seed_memories(repo, CK, [f"ck {i}" for i in range(page_size + 1)])
        await seed_memories(repo, OTHER, [f"other {i}" for i in range(page_size + 1)],
                            prefix="o")
        embed = embed_service([1.0, 0.0, 0.0])
        worker = make_worker(repo, embed)
        built = await worker._full_backfill()
        # Both chats advanced one page and were re-enqueued (fair).
        queued = set(worker._queued)
        # Drain to completion.
        task = asyncio.create_task(worker.run())
        try:
            await wait_for_worker_built(worker, task)
            gens = await repo.list_embedding_generations()
            active = [g for g in gens if g.state == "active"]
            ck_rows = await repo.list_vectors(CK, "e", active[0].id)
            other_rows = await repo.list_vectors(OTHER, "e", active[0].id)
            return built, queued, active, len(ck_rows), len(other_rows)
        finally:
            await cancel_and_await_worker(task)
            await repo.close()

    built, queued, active, n_ck, n_other = run(scenario())
    assert built is False  # pending pages after one round
    assert queued == {CK, OTHER}  # both chats advanced one page (fair)
    assert len(active) == 1
    page_size = SemanticBackfill._MEMORY_PAGE
    assert n_ck == page_size + 1 and n_other == page_size + 1  # every page covered


def test_backfill_cancelled_between_pages_stays_building_and_restart_resumes(tmp_path):
    """Cancellation between pages leaves the generation literally building
    (never activated); a restart resumes the SAME building generation and
    covers every page."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        page_size = SemanticBackfill._MEMORY_PAGE
        await seed_memories(repo, CK, [f"text {i}" for i in range(page_size + 1)])

        class CancellingEmbedder:
            def __init__(self):
                self.worker = None
                self.calls = 0

            async def embed(self, texts):
                self.calls += 1
                if self.calls == 2:  # cancel between the first two pages
                    self.worker.cancel()
                return [[1.0, 0.0, 0.0] for _ in texts]

        embedder = CancellingEmbedder()
        embed = OptionalEmbeddingService(embedder, space_id="e@r1")
        worker = make_worker(repo, embed)
        embedder.worker = worker
        built = await worker._full_backfill()
        gens = await repo.list_embedding_generations()
        building = [g for g in gens if g.state == "building"]
        active = [g for g in gens if g.state == "active"]
        # Restart: a fresh worker resumes the SAME building generation and
        # covers every page (cache hits make the re-scan cheap).
        embed2 = embed_service([1.0, 0.0, 0.0])
        worker2 = make_worker(repo, embed2)
        task2 = asyncio.create_task(worker2.run())
        try:
            await wait_for_worker_built(worker2, task2)
            gens2 = await repo.list_embedding_generations()
            active2 = [g for g in gens2 if g.state == "active"]
            rows = await repo.list_vectors(CK, "e", active2[0].id)
            return built, building, active, active2, len(rows)
        finally:
            await cancel_and_await_worker(task2)
            await repo.close()

    built, building, active, active2, n = run(scenario())
    assert built is False  # cancelled mid-scan
    assert len(building) == 1 and active == []  # literally building
    assert len(active2) == 1 and active2[0].id == building[0].id  # same gen
    assert n == SemanticBackfill._MEMORY_PAGE + 1  # restart covered every page


def test_incremental_backfill_pages_large_chat_into_active_generation(tmp_path):
    """The incremental path (enqueued chat, active generation) processes a
    large backlog in bounded pages, re-enqueuing until every page is
    covered."""
    async def scenario():
        _db, repo = await open_repo_with_chat(tmp_path / "t.db")
        page_size = SemanticBackfill._MEMORY_PAGE
        await seed_memories(repo, CK, ["apple pie"])
        embed = embed_service([1.0, 0.0, 0.0])
        worker = make_worker(repo, embed)
        assert await worker._full_backfill() is True
        # A large backlog arrives after activation.
        await seed_memories(repo, CK, [f"late {i}" for i in range(page_size + 1)],
                            prefix="late")
        # Drain the enqueued chat in bounded pages (the worker re-enqueues
        # until every page is covered).
        rows = []
        for _ in range(2):
            await worker._backfill_chat(CK)
            gen = await worker._active_generation()
            rows = await repo.list_vectors(CK, "e", gen.id)
            if len(rows) == page_size + 2:
                break
        await repo.close()
        return len(rows)

    n = run(scenario())
    assert n == SemanticBackfill._MEMORY_PAGE + 2  # original plus every late page
