"""Phase 5 runtime-local delivery: terminal-dispatch memory maintenance,
DB-start FTS bootstrap + crash repair, and knowledge-tool reachability.

Focused end-to-end tests over real SQLite + VirtualClock. The memory lane is
deterministic and local: the default capsule summarizer and FTS-only recall
perform ZERO provider/LLM/embed calls.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pretender.app import App
from pretender.budget import BudgetConfig, BudgetManager
from pretender.clock import VirtualClock
from pretender.config import AgentConfig, Config, ContextConfig
from pretender.cycle import (
    CycleRunner,
    PhaseAgent,
    _pending_transcript,
    _render_chat_log,
)
from pretender.gate import Gate
from pretender.memory import MemoryService, default_capsule_summarizer
from pretender.person import PersonService
from pretender.planner import PlanIntent, PlanResult, Planner
from pretender.prompts import PromptStore
from pretender.replyer import ReplyDraft, Replyer
from pretender.search import MemorySearch
from pretender.tools.core import register_core_tools
from pretender.types import (
    ChatKey,
    CycleId,
    DispatchCause,
    DispatchGrant,
    DispatchRequest,
    LLMResponse,
    MemoryWriteRequest,
    Message,
    MessageId,
    MessageRowId,
    PersonProfile,
    SenderId,
    ToolCall,
    ToolCallId,
)
from tests.durable_helpers import (
    CK,
    make_identity,
    make_message,
    open_repo,
    open_repo_with_chat,
    run,
)
from tests.knowledge_helpers import make_memory


def _trigger_message(
    recv_ts: float = 100.0, msg_id: str = "m1", text: str = "火锅好吃"
) -> Message:
    return Message(
        chat_key=CK,
        sender_id=SenderId("u1"),
        sender_name="user",
        is_self=False,
        text=text,
        id=MessageId(msg_id),
        mentions=(SenderId("bot-1"),),
        recv_ts=recv_ts,
    )


async def _begin_dispatch(
    repo, *, cycle_id="cy-1", now=200.0, cause=DispatchCause.INBOUND
) -> DispatchGrant:
    grant = await repo.begin_dispatch(
        DispatchRequest(
            chat_key=CK,
            cause=cause,
            cycle_id=CycleId(cycle_id),
            started_ts=now,
            expires_at=now + 300.0,
            now=now,
        )
    )
    assert isinstance(grant, DispatchGrant)
    return grant


def _summarize_on_memory(svc):
    """An ``on_memory`` callback that summarizes one oldest batch via the
    given MemoryService (returns None, as the callback contract requires)."""

    async def on_memory(ck, through):
        await svc.summarize(ck, through_msg_id=through)

    return on_memory


class FakeLLM:
    """Scripted LLMClient: pops one response per complete() call."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls: list[tuple] = []

    async def complete(
        self, messages, *, profile, tools=None, temperature=None,
        max_tokens=None, deadline=None,
    ):
        self.calls.append((list(messages), profile, tools))
        if self.script:
            return self.script.pop(0)
        return LLMResponse(content=None)


class _WaitPlanner:
    async def plan(self, messages, *, chat_log, reply_style,
                   focus_chat=None, bot_name="", drift_block="",
                   behavior_style="",
                   tools=None, temperature=None,
                   max_tokens=None, deadline=None, max_tool_rounds=None):
        return PlanResult(intent=PlanIntent.WAIT, wait_seconds=30.0,
                          tokens_in=2, tokens_out=1, end_reason="wait")


class _EmptyReplyer:
    async def reply(self, *, reply_reference, identity, reply_style,
                    reply_to=None, context=None, temperature=None,
                    max_tokens=None, deadline=None):
        return ReplyDraft.empty()


# ── terminal dispatch / local capsule -> nonempty FTS recall ─────────────────

def test_terminal_dispatch_local_capsule_nonempty_fts_recall(tmp_path):
    """A terminal dispatch fires on_memory (after durable settlement), the
    default local capsule summarizes one batch, and recall is nonempty and
    FTS-only (no embed/LLM)."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        svc = MemoryService(repo, summarizer=default_capsule_summarizer())
        runner = CycleRunner(
            repo, Gate(), Config(), clock=VirtualClock(epoch=200.0),
            dry_run=True, uuid_fn=lambda: "cy-1",
            on_memory=_summarize_on_memory(svc),
        )
        decision = await runner.run_dispatch(grant)
        hits = await svc.recall(CK, "火锅", limit=5)
        wm = await repo.get_memory_watermark(CK)
        await repo.close()
        return decision, hits, wm

    decision, hits, wm = run(scenario())
    assert decision.action == "trigger"
    assert wm == MessageRowId(1)  # durable settlement advanced the watermark
    assert len(hits) == 1
    assert hits[0].text == "火锅好吃"
    assert hits[0].source == "lexical"  # FTS-only, zero embed calls


# ── no callback before settlement / on release / on defer ────────────────────

def test_no_memory_callback_on_release_or_defer(tmp_path):
    """on_memory fires ONLY from terminal settlement: a release (no-agent
    non-dry-run trigger) and a defer (agent wait) never write memory."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        calls: list[tuple] = []

        async def on_memory(ck, through):
            calls.append((ck, through))

        # Release: no-agent non-dry-run trigger -> release, no on_memory.
        grant = await _begin_dispatch(repo)
        runner = CycleRunner(
            repo, Gate(), Config(), clock=VirtualClock(epoch=200.0),
            dry_run=False, uuid_fn=lambda: "cy-1", on_memory=on_memory,
        )
        await runner.run_dispatch(grant)

        # Defer: agent wait -> defer, no on_memory.
        agent = PhaseAgent(_WaitPlanner(), _EmptyReplyer())
        grant2 = await _begin_dispatch(repo, cycle_id="cy-2")
        runner2 = CycleRunner(
            repo, Gate(), Config(), clock=VirtualClock(epoch=200.0),
            dry_run=False, uuid_fn=lambda: "cy-2", agent=agent,
            on_memory=on_memory,
        )
        await runner2.run_dispatch(grant2)

        wm = await repo.get_memory_watermark(CK)
        await repo.close()
        return calls, wm

    calls, wm = run(scenario())
    assert calls == []  # no memory callback on release/defer
    assert wm is None  # nothing summarized


# ── startup FTS bootstrap + crash repair, idempotent ─────────────────────────

def test_startup_fts_bootstrap_idempotent(tmp_path):
    """DB-start maintenance bootstraps the canonical memory FTS index for a
    chat whose docs are missing; re-running reproduces the same index."""
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        # Legacy memory row with no FTS docs / no bootstrap state.
        await repo._db.write(
            lambda c: c.execute(
                "INSERT INTO memories(chat_key, kind, text, cues_json)"
                " VALUES (?, 'capsule', ?, '[]')",
                (CK, "火锅好吃"),
            )
        )
        svc = MemoryService(repo, summarizer=default_capsule_summarizer())
        app = App(repo=repo, memory_service=svc)
        await app._memory_maintenance()
        hits = await repo.query_memory(CK, "火锅", limit=5)
        state = await repo.get_memory_fts_state(CK)
        # Idempotent: a second maintenance run changes nothing.
        await app._memory_maintenance()
        hits2 = await repo.query_memory(CK, "火锅", limit=5)
        await repo.close()
        return hits, state, hits2

    hits, state, hits2 = run(scenario())
    assert len(hits) == 1 and hits[0].text == "火锅好吃"
    assert state == (True, None)  # bootstrapped
    assert hits2 == hits  # idempotent


def test_startup_crash_repair_idempotent(tmp_path):
    """A crash after terminal settlement (cursor advanced, watermark not)
    is repaired at startup: one oldest batch is summarized per pending chat,
    and re-running finds no pending work."""
    async def scenario():
        db, repo = await open_repo_with_chat(tmp_path / "t.db")
        await repo.ingest_message(make_identity(), _trigger_message())
        # Terminal dispatch WITHOUT on_memory: cursor advances, watermark stays.
        grant = await _begin_dispatch(repo)
        runner = CycleRunner(
            repo, Gate(), Config(), clock=VirtualClock(epoch=200.0),
            dry_run=True, uuid_fn=lambda: "cy-1",
        )
        await runner.run_dispatch(grant)
        pending = await repo.list_memory_pending_chats()
        svc = MemoryService(repo, summarizer=default_capsule_summarizer())
        app = App(repo=repo, memory_service=svc)
        await app._memory_maintenance()
        wm = await repo.get_memory_watermark(CK)
        pending2 = await repo.list_memory_pending_chats()
        hits = await svc.recall(CK, "火锅", limit=5)
        await repo.close()
        return pending, wm, pending2, hits

    pending, wm, pending2, hits = run(scenario())
    assert pending == [(CK, MessageRowId(1))]  # the crash-after-settlement gap
    assert wm == MessageRowId(1)  # repaired
    assert pending2 == []  # idempotent
    assert len(hits) == 1


# ── prompt allows a knowledge round; UID is rendered ─────────────────────────

def test_prompt_allows_knowledge_round_and_uid_rendered():
    """Both Chinese planner prompts permit information gathering via
    tool_search/query_memory/query_person_profile before the final verdict,
    and the chat log + pending transcript render the exact escaped sender
    UID so profile lookup has a compliant tool argument."""
    base = Path(__file__).resolve().parent.parent / "pretender" / "prompts"
    for name in ("planner.txt", "planner_focus.txt"):
        text = (base / name).read_text(encoding="utf-8")
        assert "tool_search" in text
        assert "query_memory" in text
        assert "query_person_profile" in text

    msg = make_message(text="hi", sender_id="u1", sender_name="user")
    log = _render_chat_log((msg,), None)
    assert 'user: hi [uid="u1"]' in log
    t = _pending_transcript((msg,))
    assert 'user: hi [uid="u1"]' in t[0].content


# ── deferred search -> knowledge tool -> terminal verdict ────────────────────

def test_deferred_search_knowledge_tool_terminal_verdict(tmp_path):
    """A planner that tool_searches then calls query_person_profile with the
    rendered UID gets a real profile hit, then gives a terminal reply."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        await repo.upsert_person(
            PersonProfile(
                chat_key=CK, platform_uid=SenderId("u1"), names=("user",),
                profile="喜欢火锅", impression="热情",
            )
        )
        grant = await _begin_dispatch(repo)
        llm = FakeLLM(
            [
                LLMResponse(
                    content="搜索",
                    tool_calls=(
                        ToolCall(id=ToolCallId("c1"), name="tool_search",
                                 arguments={"capability": "memory"}),
                    ),
                    usage={"prompt_tokens": 8, "completion_tokens": 2},
                ),
                LLMResponse(
                    content="查人",
                    tool_calls=(
                        ToolCall(id=ToolCallId("c2"), name="query_person_profile",
                                 arguments={"platform_uid": "u1"}),
                    ),
                    usage={"prompt_tokens": 10, "completion_tokens": 3},
                ),
                LLMResponse(
                    content="分析",
                    tool_calls=(
                        ToolCall(id=ToolCallId("c3"), name="reply",
                                 arguments={"text": "参考回复"}),
                    ),
                    usage={"prompt_tokens": 12, "completion_tokens": 4},
                ),
                LLMResponse(
                    content="你好",
                    usage={"prompt_tokens": 6, "completion_tokens": 2},
                ),
            ]
        )
        registry = register_core_tools()
        budget = BudgetManager(repo, BudgetConfig(daily_cap=100), now=lambda: 200.0)
        agent = PhaseAgent.budgeted(
            llm, PromptStore(), registry, ContextConfig(), budget, AgentConfig(),
            memory_search=MemorySearch(repo), person_service=PersonService(repo),
        )
        runner = CycleRunner(
            repo, Gate(), Config(), clock=VirtualClock(epoch=200.0),
            dry_run=False, uuid_fn=lambda: "cy-1", agent=agent,
        )
        decision = await runner.run_dispatch(grant)
        outbox = await db.read(
            lambda c: c.execute("SELECT text FROM outbox").fetchall()
        )
        await repo.close()
        return decision, outbox, llm

    decision, outbox, llm = run(scenario())
    assert decision.action == "trigger"
    assert outbox == [("你好",)]
    # The query_person_profile tool result carried the real profile hit.
    tool_results = llm.calls[2][0]
    tool_contents = [m.content for m in tool_results if m.role == "tool"]
    assert any("喜欢火锅" in c for c in tool_contents)


# ── no LLM/embed calls in the memory lane ────────────────────────────────────

def test_memory_lane_makes_no_llm_or_embed_calls(tmp_path):
    """The terminal-dispatch memory lane (capsule summarize + FTS recall)
    performs ZERO provider/LLM/embed calls: an embed that raises is never
    invoked and no agent/LLM is present."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)

        class _BoomEmbed:
            enabled = True
            space_id = "m@v1"

            async def embed(self, texts):
                raise AssertionError("embed must not be called in this lane")

        search = MemorySearch(repo, embed=_BoomEmbed())
        svc = MemoryService(repo, search=search, summarizer=default_capsule_summarizer())
        runner = CycleRunner(
            repo, Gate(), Config(), clock=VirtualClock(epoch=200.0),
            dry_run=True, uuid_fn=lambda: "cy-1",
            on_memory=_summarize_on_memory(svc),
        )
        decision = await runner.run_dispatch(grant)
        hits = await svc.recall(CK, "火锅", limit=5)
        await repo.close()
        return decision, hits

    decision, hits = run(scenario())
    assert decision.action == "trigger"
    assert len(hits) == 1 and hits[0].source == "lexical"


# ── dry-run / no-agent / legacy regressions ──────────────────────────────────

def test_dry_run_no_agent_terminal_still_fires_memory(tmp_path):
    """A dry-run no-agent terminal trigger still fires on_memory (the memory
    lane is local and independent of the agent/outbox lane)."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        svc = MemoryService(repo, summarizer=default_capsule_summarizer())
        runner = CycleRunner(
            repo, Gate(), Config(), clock=VirtualClock(epoch=200.0),
            dry_run=True, uuid_fn=lambda: "cy-1",
            on_memory=_summarize_on_memory(svc),
        )
        decision = await runner.run_dispatch(grant)
        wm = await repo.get_memory_watermark(CK)
        await repo.close()
        return decision, wm

    decision, wm = run(scenario())
    assert decision.action == "trigger"
    assert wm == MessageRowId(1)
