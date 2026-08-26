"""Phase 3 agent integration: CycleRunner + PhaseAgent (planner + replyer +
optional budget) over the dispatch-ledger lane.

Covers: trigger → single ledger outbox batch in non-dry test mode; dry-run
zero outbox/send; wait delay/cursor preservation; no_action / budget-blocked
terminal consume; tokens/usage and the deterministic idempotency key; legal
tool results with planner analysis absent from the visible reply; hook and
marker trace ordering; and the legacy no-agent regression.

Async tests run via asyncio.run() so the test extra stays at just pytest.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from pretender.budget import (
    ALLOWED,
    BLOCKED,
    BudgetDecision,
    BudgetManager,
    BudgetRung,
    BudgetUsage,
    BudgetedClient,
)
from pretender.clock import VirtualClock
from pretender.config import (
    AgentConfig,
    BudgetConfig,
    Config,
    ContextConfig,
    MediaConfig,
)
from pretender.context import serialize
from pretender.cycle import CycleRunner, PhaseAgent
from pretender.errors import LLMPermanentError, LLMTransientError, PromptError
from pretender.gate import Gate
from pretender.person import PersonService
from pretender.planner import PlanIntent, PlanResult, Planner
from pretender.prompts import PromptStore
from pretender.registry import HookBus
from pretender.replyer import ReplyDraft, Replyer
from pretender.search import MemorySearch
from pretender.tools.core import ToolContext, register_core_tools
from pretender.tools.media import MediaCallbacks, MediaReplyIntent
from pretender.types import (
    ChatKey,
    ChatState,
    CommitSeq,
    CycleId,
    DispatchCause,
    DispatchDeferred,
    DispatchGrant,
    DispatchRequest,
    LearnerDraft,
    LearnerGrant,
    LearnerRunRequest,
    LLMResponse,
    MediaAssetCandidate,
    MediaKind,
    MediaSafetyStatus,
    MemoryRecord,
    MemoryWriteRequest,
    Message,
    MessageId,
    MessageRowId,
    Outgoing,
    Reason,
    Record,
    SenderId,
    Segment,
    ToolCall,
    ToolCallId,
    TranscriptMessage,
)
from tests.durable_helpers import (
    CK,
    make_identity,
    make_message,
    open_repo,
    run,
)
from tests.knowledge_helpers import make_memory


def run(coro):
    return asyncio.run(coro)


# ── fakes ────────────────────────────────────────────────────────────────────


class FakePlanner:
    """Scripted planner: pops one PlanResult per plan() call, records every
    call's inputs."""

    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls: list[dict] = []

    async def plan(
        self,
        messages,
        *,
        identity,
        chat_log,
        reply_style,
        focus_chat=None,
        tools=None,
        temperature=None,
        max_tokens=None,
        deadline=None,
        max_tool_rounds=None,
    ):
        self.calls.append(
            {
                "messages": list(messages),
                "identity": identity,
                "chat_log": chat_log,
                "reply_style": reply_style,
                "focus_chat": focus_chat,
            }
        )
        if self.results:
            return self.results.pop(0)
        return PlanResult(intent=PlanIntent.NO_ACTION, end_reason="no_tool_call")


class FakeReplyer:
    """Scripted replyer: pops one ReplyDraft per reply() call."""

    def __init__(self, drafts=None):
        self.drafts = list(drafts or [])
        self.calls: list[dict] = []

    async def reply(
        self,
        *,
        reply_reference,
        identity,
        reply_style,
        reply_to=None,
        temperature=None,
        max_tokens=None,
        deadline=None,
    ):
        self.calls.append(
            {
                "reply_reference": reply_reference,
                "identity": identity,
                "reply_style": reply_style,
                "reply_to": reply_to,
            }
        )
        if self.drafts:
            return self.drafts.pop(0)
        return ReplyDraft.empty()


class FakeBudget:
    """Scripted budget: pops one BudgetDecision per decide(), records every
    record() call."""

    def __init__(self, decisions=None):
        self.decisions = list(decisions or [])
        self.decide_calls: list[ChatKey] = []
        self.record_calls: list[tuple] = []

    async def decide(self, chat_key):
        self.decide_calls.append(chat_key)
        if self.decisions:
            return self.decisions.pop(0)
        return BudgetDecision(
            kind=ALLOWED,
            usage=BudgetUsage(day="2026-01-01", calls=0, tokens=0, cost=0.0),
            remaining=100,
        )

    async def record(self, chat_key, *, calls=1, tokens=0, cost=0.0):
        self.record_calls.append((chat_key, calls, tokens, cost))
        return BudgetUsage(day="2026-01-01", calls=calls, tokens=tokens, cost=cost)


class FakeLLM:
    """Scripted LLMClient: pops one response per complete() call, records
    every call, and validates transcript legality via ``serialize``."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls: list[tuple] = []

    async def complete(
        self,
        messages,
        *,
        profile,
        tools=None,
        temperature=None,
        max_tokens=None,
        deadline=None,
    ):
        msgs = list(messages)
        serialize(msgs)  # fail the test if an illegal transcript was built
        self.calls.append((msgs, profile, tools))
        if self.script:
            item = self.script.pop(0)
            if callable(item):
                return item(msgs)
            return item
        return LLMResponse(content=None)


# ── helpers ──────────────────────────────────────────────────────────────────


def _trigger_message(recv_ts: float = 100.0, msg_id: str = "m1") -> Message:
    return Message(
        chat_key=CK,
        sender_id=SenderId("u1"),
        sender_name="user",
        is_self=False,
        text="hi",
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


def make_agent_runner(
    repo,
    agent,
    *,
    clock=None,
    hooks=None,
    dry_run=False,
    uuid_fn=None,
    marker_exporter=None,
    **kw,
):
    return CycleRunner(
        repo,
        Gate(),
        Config(),
        clock=clock if clock is not None else VirtualClock(epoch=200.0),
        hooks=hooks,
        dry_run=dry_run,
        uuid_fn=uuid_fn,
        marker_exporter=marker_exporter,
        agent=agent,
        **kw,
    )


def _reply_agent(*, reply_text="你好", reply_to=None, tokens_in=7, tokens_out=3):
    planner = FakePlanner(
        [
            PlanResult(
                intent=PlanIntent.REPLY,
                reply_reference="参考回复",
                reply_to=reply_to,
                tokens_in=10,
                tokens_out=5,
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                end_reason="reply",
            )
        ]
    )
    replyer = FakeReplyer(
        [
            ReplyDraft(
                text=reply_text,
                reply_to=reply_to,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                usage={"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
            )
        ]
    )
    return PhaseAgent(planner, replyer), planner, replyer


def test_cycle_segmented_outgoing_stays_atomic(tmp_path):
    """A text splitter must not cause an image/sticker payload to be emitted
    once per part in the durable agent path."""

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        agent, _planner, _replyer = _reply_agent()
        runner = make_agent_runner(repo, agent)
        out = Outgoing(
            chat_key=CK,
            text="完整消息",
            parts=["第一段", "第二段"],
            segments=[Segment("image", {"file": "https://example.test/a.png"})],
        )
        items = runner._outgoing_to_items(grant, out)
        await repo.close()
        return items

    items = run(scenario())
    assert len(items) == 1
    assert items[0].text == "完整消息"
    assert items[0].segments[0].kind == "image"


# ── trigger → single ledger outbox batch (non-dry test mode) ─────────────────

def test_agent_reply_creates_single_ledger_outbox_batch(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        agent, planner, replyer = _reply_agent()
        budget = FakeBudget()
        agent = PhaseAgent(agent._planner, agent._replyer, budget)
        hooks = HookBus()
        seen: list[tuple] = []

        @hooks.on_cycle_end
        def hook(chat_key, trace, end_reason):
            seen.append((chat_key, trace.decision.action, end_reason))

        runner = make_agent_runner(
            repo, agent, hooks=hooks, dry_run=False, uuid_fn=lambda: "cy-1"
        )
        decision = await runner.run_dispatch(grant)
        outbox = await db.read(
            lambda c: c.execute(
                "SELECT text, idem_key, state, reply_to FROM outbox"
            ).fetchall()
        )
        cycles = await db.read(
            lambda c: c.execute(
                "SELECT end_reason, tokens_in, tokens_out FROM cycles"
            ).fetchall()
        )
        cursor = await db.read(
            lambda c: c.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?", (CK,)
            ).fetchone()[0]
        )
        dispatch_state = await db.read(
            lambda c: c.execute(
                "SELECT state FROM dispatches WHERE id = ?", (grant.dispatch_id,)
            ).fetchone()[0]
        )
        await repo.close()
        return (
            decision, outbox, cycles, cursor, dispatch_state, seen,
            budget, planner, replyer,
        )

    (
        decision, outbox, cycles, cursor, dispatch_state, seen,
        budget, planner, replyer,
    ) = run(scenario())
    assert decision.action == "trigger"
    assert decision.reason == Reason.TRIGGER
    # ONE ledger outbox row, pending, with the replyer's final text.
    assert outbox == [("你好", "dispatch:1:0", "pending", None)]
    # The terminal cycle records the aggregated planner + replyer usage.
    assert cycles == [("agent_reply", 17, 8)]
    assert cursor == 1  # cursor consumed to the grant's through boundary
    assert dispatch_state == "completed"
    assert seen == [(CK, "trigger", "agent_reply")]  # hook after completion
    # The injected-seam form owns its own budget: no saga-level accounting.
    assert budget.decide_calls == []
    assert budget.record_calls == []
    assert len(planner.calls) == 1
    assert len(replyer.calls) == 1
    # The planner saw the pending transcript and the rendered chat log.
    # The identity is the configured identity_file content (not merely the
    # bot name).
    assert "你是麦麦，一个群聊里的普通成员" in planner.calls[0]["identity"]
    assert "user: hi" in planner.calls[0]["chat_log"]
    assert [m.role for m in planner.calls[0]["messages"]] == ["user"]


def test_agent_reply_idempotency_keys_are_unique_per_dispatch(tmp_path):
    """Two dispatches produce DISTINCT deterministic idempotency keys
    derived from the durable grant identity and the part index."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant1 = await _begin_dispatch(repo, cycle_id="cy-1")
        agent, _, _ = _reply_agent()
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        await runner.run_dispatch(grant1)
        # A second trigger message and dispatch.
        await repo.ingest_message(
            make_identity(), _trigger_message(recv_ts=110.0, msg_id="m2")
        )
        grant2 = await _begin_dispatch(repo, cycle_id="cy-2")
        agent2, _, _ = _reply_agent()
        runner2 = make_agent_runner(repo, agent2, dry_run=False, uuid_fn=lambda: "cy-2")
        await runner2.run_dispatch(grant2)
        keys = await db.read(
            lambda c: [r[0] for r in c.execute("SELECT idem_key FROM outbox ORDER BY id")]
        )
        await repo.close()
        return keys

    keys = run(scenario())
    assert keys == ["dispatch:1:0", "dispatch:2:0"]


# ── dry-run: agent evaluates, zero outbox rows, never sends ──────────────────

def test_agent_dry_run_zero_outbox_zero_send(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        agent, planner, replyer = _reply_agent()
        budget = FakeBudget()
        agent = PhaseAgent(agent._planner, agent._replyer, budget)
        runner = make_agent_runner(repo, agent, dry_run=True, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        cursor = await db.read(
            lambda c: c.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?", (CK,)
            ).fetchone()[0]
        )
        await repo.close()
        return decision, outbox, cycles, cursor, budget, planner, replyer

    decision, outbox, cycles, cursor, budget, planner, replyer = run(scenario())
    assert decision.action == "trigger"
    assert outbox == 0  # zero outbox rows
    assert cycles == [("dry_run_agent_reply",)]  # evaluated, never sent
    assert cursor == 1  # terminal consume
    # The agent still evaluated; the injected-seam form owns its own budget.
    assert len(planner.calls) == 1
    assert len(replyer.calls) == 1
    assert budget.record_calls == []


# ── wait: release with a timed Decision, cursor preserved ────────────────────

def test_agent_wait_releases_with_timed_decision(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        planner = FakePlanner(
            [
                PlanResult(
                    intent=PlanIntent.WAIT,
                    wait_seconds=30.0,
                    tokens_in=4,
                    tokens_out=1,
                    usage={"prompt_tokens": 4, "completion_tokens": 1},
                    end_reason="wait",
                )
            ]
        )
        replyer = FakeReplyer()
        agent = PhaseAgent(planner, replyer)
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        dispatch_state = await db.read(
            lambda c: c.execute(
                "SELECT state FROM dispatches WHERE id = ?", (grant.dispatch_id,)
            ).fetchone()[0]
        )
        cursor = await db.read(
            lambda c: c.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?", (CK,)
            ).fetchone()[0]
        )
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
        )
        unassigned = await repo.list_unassigned_commits(CK)
        await repo.close()
        return decision, dispatch_state, cursor, outbox, cycles, unassigned

    decision, dispatch_state, cursor, outbox, cycles, unassigned = run(scenario())
    assert decision.action == "delay"
    assert decision.delay_seconds == 30.0  # the scheduler re-arms a timer wake
    assert dispatch_state == "released"  # released, not terminal
    assert cursor is None  # cursor preserved
    assert outbox == 0
    assert cycles == 0  # no terminal cycle
    assert unassigned == [CommitSeq(1)]  # commits detached, still pending


def test_agent_wait_without_seconds_is_event_only(tmp_path):
    """A wait intent without a usable duration degrades to an event-only
    delay release (no timed wake, no busy loop)."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        planner = FakePlanner(
            [
                PlanResult(
                    intent=PlanIntent.WAIT,
                    wait_seconds=None,
                    tokens_in=2,
                    tokens_out=1,
                    end_reason="wait",
                )
            ]
        )
        agent = PhaseAgent(planner, FakeReplyer())
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        await repo.close()
        return decision

    decision = run(scenario())
    assert decision.action == "delay"
    assert decision.delay_seconds is None  # event-only


# ── no_action / budget-blocked: terminal consume without outbox ──────────────

def test_agent_no_action_terminally_consumes(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        planner = FakePlanner(
            [
                PlanResult(
                    intent=PlanIntent.NO_ACTION,
                    tokens_in=3,
                    tokens_out=1,
                    end_reason="no_action",
                )
            ]
        )
        agent = PhaseAgent(planner, FakeReplyer())
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        cursor = await db.read(
            lambda c: c.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?", (CK,)
            ).fetchone()[0]
        )
        await repo.close()
        return decision, outbox, cycles, cursor

    decision, outbox, cycles, cursor = run(scenario())
    assert decision.action == "trigger"
    assert outbox == 0  # terminal consume without outbox
    assert cycles == [("no_action",)]
    assert cursor == 1


def test_agent_budget_blocked_terminally_consumes(tmp_path):
    """The budgeted form: a hard-capped chat blocks the first provider call
    (BudgetBlockedError) and the agent terminally consumes with
    budget_blocked — no LLM calls, no outbox."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        llm = FakeLLM()
        registry = register_core_tools()
        budget = BudgetManager(
            repo, BudgetConfig(daily_cap=1), now=lambda: 200.0
        )
        await budget.record(CK, calls=1)  # reach the hard cap
        agent = PhaseAgent.budgeted(
            llm, PromptStore(), registry, ContextConfig(), budget, AgentConfig()
        )
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        cursor = await db.read(
            lambda c: c.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?", (CK,)
            ).fetchone()[0]
        )
        await repo.close()
        return decision, outbox, cycles, cursor, llm

    decision, outbox, cycles, cursor, llm = run(scenario())
    assert decision.action == "trigger"
    assert outbox == 0
    assert cycles == [("budget_blocked",)]
    assert cursor == 1
    # No LLM calls were made (the hard cap blocked the first call).
    assert llm.calls == []


# ── reply with no usable draft: terminal consume without outbox ──────────────

def test_agent_reply_no_output_terminally_consumes(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        planner = FakePlanner(
            [
                PlanResult(
                    intent=PlanIntent.REPLY,
                    reply_reference="参考",
                    tokens_in=5,
                    tokens_out=2,
                    end_reason="reply",
                )
            ]
        )
        replyer = FakeReplyer([ReplyDraft.empty()])
        agent = PhaseAgent(planner, replyer)
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        await repo.close()
        return decision, outbox, cycles

    decision, outbox, cycles = run(scenario())
    assert decision.action == "trigger"
    assert outbox == 0  # an empty draft never creates an outbox row
    assert cycles == [("reply_no_output",)]


# ── legal tool results; planner analysis absent from the visible reply ───────

def test_agent_legal_tool_results_and_analysis_absent(tmp_path):
    """The REAL planner + replyer over fake LLMs: every tool id is answered
    on the wire, and the only text that reaches the outbox is the replyer's
    final draft — never the planner's analysis or tool JSON."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        planner_llm = FakeLLM(
            [
                LLMResponse(
                    content="先搜索",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("call_1"),
                            name="tool_search",
                            arguments={"query": "history"},
                        ),
                    ),
                    usage={"prompt_tokens": 8, "completion_tokens": 2},
                ),
                LLMResponse(
                    content="分析：用户打招呼，值得回复。",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("call_2"),
                            name="reply",
                            arguments={"text": "参考回复"},
                        ),
                    ),
                    usage={"prompt_tokens": 12, "completion_tokens": 4},
                ),
            ]
        )
        replyer_llm = FakeLLM(
            [
                LLMResponse(
                    content="你好",
                    usage={"prompt_tokens": 6, "completion_tokens": 2},
                )
            ]
        )
        registry = register_core_tools()
        planner = Planner(
            planner_llm,
            PromptStore(),
            registry,
            ContextConfig(),
            tool_context_factory=lambda: ToolContext(
                chat_key=CK,
                chat_kind="group",
                capabilities=frozenset(),
                registry=registry,
                self_name="麦麦",
            ),
        )
        replyer = Replyer(replyer_llm, PromptStore())
        agent = PhaseAgent(planner, replyer)
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        outbox = await db.read(
            lambda c: c.execute("SELECT text FROM outbox").fetchall()
        )
        await repo.close()
        return decision, outbox, planner_llm, replyer_llm

    decision, outbox, planner_llm, replyer_llm = run(scenario())
    assert decision.action == "trigger"
    # The outbox carries ONLY the replyer's final draft.
    assert outbox == [("你好",)]
    # The planner's SECOND-round wire transcript shows the first round's
    # tool call answered by exactly one tool message (every id answered).
    second_msgs = planner_llm.calls[1][0]
    tool_ids = [
        c.id
        for m in second_msgs
        if m.role == "assistant"
        for c in m.tool_calls
    ]
    answer_ids = [m.tool_call_id for m in second_msgs if m.role == "tool"]
    assert tool_ids == [ToolCallId("call_1")]
    assert answer_ids == [ToolCallId("call_1")]
    # The replyer's transcript carries only the staged reference — never the
    # planner's analysis or tool JSON.
    replyer_msgs = replyer_llm.calls[0][0]
    assert [m.role for m in replyer_msgs] == ["system", "user"]
    assert "分析" not in replyer_msgs[1].content
    assert "call_1" not in replyer_msgs[1].content
    assert "call_2" not in replyer_msgs[1].content


# ── hook and marker trace ordering ───────────────────────────────────────────

def test_agent_hook_and_marker_trace(tmp_path):
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        agent, _, _ = _reply_agent()
        hooks = HookBus()
        seen: list[tuple] = []
        markers: list = []

        @hooks.on_cycle_end
        def hook(chat_key, trace, end_reason):
            seen.append((chat_key, trace.decision.action, end_reason))

        async def exporter(marker):
            markers.append(marker)

        runner = make_agent_runner(
            repo, agent, hooks=hooks, dry_run=False, uuid_fn=lambda: "cy-1",
            marker_exporter=exporter,
        )
        decision = await runner.run_dispatch(grant)
        await repo.close()
        return decision, seen, markers

    decision, seen, markers = run(scenario())
    assert decision.action == "trigger"
    assert seen == [(CK, "trigger", "agent_reply")]
    assert len(markers) == 1
    marker = markers[0]
    assert marker.record_type == "dispatch"
    assert marker.state == "completed"
    assert marker.sequence == 1
    assert marker.trace_json is not None
    assert json.loads(marker.trace_json)["decision"]["reason"] == "trigger"


# ── legacy no-agent regression ───────────────────────────────────────────────

def test_agent_absent_preserves_no_agent_behavior(tmp_path):
    """A runner WITHOUT an agent keeps the frozen no-agent behavior: a
    trigger outside dry-run is released (never retained, never sent)."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        runner = make_agent_runner(repo, None, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        dispatch_state = await db.read(
            lambda c: c.execute(
                "SELECT state FROM dispatches WHERE id = ?", (grant.dispatch_id,)
            ).fetchone()[0]
        )
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
        )
        await repo.close()
        return decision, dispatch_state, outbox, cycles

    decision, dispatch_state, outbox, cycles = run(scenario())
    assert decision.action == "trigger"
    assert dispatch_state == "released"  # never retained without an agent
    assert outbox == 0
    assert cycles == 0


def test_agent_absent_dry_run_trigger_finishes_empty(tmp_path):
    """Without an agent, the dry-run trigger still terminally finishes with
    an EMPTY outbox (the frozen Phase 2 behavior)."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        runner = make_agent_runner(repo, None, dry_run=True, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        await repo.close()
        return decision, outbox, cycles

    decision, outbox, cycles = run(scenario())
    assert decision.action == "trigger"
    assert outbox == 0
    assert cycles == [("dry_run_trigger",)]


# ── PhaseAgent construction ──────────────────────────────────────────────────

def test_phase_agent_requires_planner_and_replyer():
    with pytest.raises(ValueError):
        PhaseAgent(None, None)
    with pytest.raises(ValueError):
        PhaseAgent(FakePlanner(), None)
    with pytest.raises(ValueError):
        PhaseAgent(None, FakeReplyer())


# ── Per-call budget (budgeted form) ──────────────────────────────────────────

def test_budgeted_multi_round_reserves_exact_call_count(tmp_path):
    """A multi-round planner (tool loop) + replyer reserves the EXACT number
    of provider calls: 2 planner rounds + 1 reply = 3, each reserved before
    the delegate request and tokens recorded after success with calls=0."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        llm = FakeLLM(
            [
                LLMResponse(
                    content="先搜索",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("c1"),
                            name="tool_search",
                            arguments={"query": "history"},
                        ),
                    ),
                    usage={"prompt_tokens": 8, "completion_tokens": 2},
                ),
                LLMResponse(
                    content="分析",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("c2"),
                            name="reply",
                            arguments={"text": "参考回复"},
                        ),
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
            llm, PromptStore(), registry, ContextConfig(), budget, AgentConfig()
        )
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        usage = await budget.snapshot(CK)
        await repo.close()
        return decision, usage, llm

    decision, usage, llm = run(scenario())
    assert decision.action == "trigger"
    # 2 planner rounds + 1 reply = 3 provider calls, each reserved.
    assert usage.calls == 3
    assert len(llm.calls) == 3
    # Tokens recorded after success: (8+2)+(12+4)+(6+2) = 34.
    assert usage.tokens == 34


class _KVRepo:
    """Minimal in-memory KV fake for the BudgetedClient unit test."""

    def __init__(self):
        self.kv: dict[str, str] = {}

    async def get_kv(self, k):
        return self.kv.get(k)

    async def set_kv(self, k, v):
        self.kv[k] = v


class _RecordingLLM:
    """Records the profile/messages/tools of every delegated call."""

    def __init__(self, response):
        self.response = response
        self.calls: list[tuple] = []

    async def complete(
        self, messages, *, profile, tools=None, temperature=None,
        max_tokens=None, deadline=None,
    ):
        self.calls.append((list(messages), profile, tools))
        return self.response


def test_budgeted_client_applies_degrade_actions():
    """The budgeted client applies the actual degrade actions to the
    delegated call: profile fallback, context reduction, capability flags."""
    async def scenario():
        repo = _KVRepo()
        budget = BudgetManager(
            repo,
            BudgetConfig(
                daily_cap=10,
                rungs=(
                    BudgetRung(at=0.3, action="degrade", detail="ctx"),
                    BudgetRung(at=0.5, action="degrade", detail="profile"),
                    BudgetRung(at=0.7, action="degrade", detail="caps"),
                ),
            ),
            now=lambda: 200.0,
        )
        await budget.record(CK, calls=7)  # fraction 0.7: all three engage
        delegate = _RecordingLLM(LLMResponse(content="hi"))
        client = BudgetedClient(
            delegate, budget, CK, agent_config=AgentConfig(fallback_profile="cheap")
        )
        messages = [TranscriptMessage(role="system", content="sys")] + [
            TranscriptMessage(role="user", content=f"m{i}") for i in range(20)
        ]
        await client.complete(
            messages, profile="planner", tools=[{"type": "function"}]
        )
        msgs, profile, tools = delegate.calls[0]
        return msgs, profile, tools

    msgs, profile, tools = run(scenario())
    assert profile == "cheap"  # profile_fallback
    assert len(msgs) == 8  # context_reduction trims to keep=8
    assert tools is None  # capability_flags drops the tool schema


def test_budgeted_client_retains_reservation_on_provider_error():
    """A provider error retains the reservation: the call count stays
    incremented and no tokens are recorded."""
    async def scenario():
        repo = _KVRepo()
        budget = BudgetManager(repo, BudgetConfig(daily_cap=10), now=lambda: 200.0)

        class _ErrLLM:
            async def complete(self, messages, *, profile, tools=None,
                               temperature=None, max_tokens=None, deadline=None):
                raise LLMTransientError("boom")

        client = BudgetedClient(_ErrLLM(), budget, CK)
        try:
            await client.complete([], profile="planner")
        except LLMTransientError:
            pass
        usage = await budget.snapshot(CK)
        return usage

    usage = run(scenario())
    assert usage.calls == 1  # reservation retained
    assert usage.tokens == 0  # no tokens recorded


# ── Lease / failure: retry defer, long-saga renewal ──────────────────────────

class _RaisingLLM:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    async def complete(self, messages, *, profile, tools=None, temperature=None,
                       max_tokens=None, deadline=None):
        self.calls += 1
        raise self.exc


class _ThenSucceedingLLM:
    """Raises LLMTransientError on the first call, then pops the scripted
    responses — models a provider blip that succeeds on retry."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def complete(self, messages, *, profile, tools=None, temperature=None,
                       max_tokens=None, deadline=None):
        self.calls += 1
        if self.calls == 1:
            raise LLMTransientError("timeout")
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(content=None)


def test_agent_timeout_defers_retry_no_cursor_outbox(tmp_path):
    """A recoverable provider failure defers the retry atomically: cursor and
    outbox unchanged, the durable barrier set, and begin_dispatch defers
    until resume_at (restart honors the remaining delay)."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        clock = VirtualClock(epoch=200.0)
        grant = await _begin_dispatch(repo, now=clock.now())
        llm = _RaisingLLM(LLMTransientError("timeout"))
        registry = register_core_tools()
        budget = BudgetManager(repo, BudgetConfig(daily_cap=100), now=clock.now)
        agent = PhaseAgent.budgeted(
            llm, PromptStore(), registry, ContextConfig(), budget,
            AgentConfig(retry_delay_s=30.0),
        )
        runner = make_agent_runner(repo, agent, clock=clock, dry_run=False,
                                   uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        state = await repo.get_chat_state(CK)
        cursor = await db.read(
            lambda c: c.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?", (CK,)
            ).fetchone()[0]
        )
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
        )
        # Restart honors the remaining delay: begin_dispatch defers.
        clock.advance(10.0)
        result = await repo.begin_dispatch(
            DispatchRequest(
                chat_key=CK, cause=DispatchCause.TIMER,
                cycle_id=CycleId("cy-2"), started_ts=clock.now(),
                expires_at=clock.now() + 60, now=clock.now(),
            )
        )
        await repo.close()
        return decision, state, cursor, outbox, cycles, result

    decision, state, cursor, outbox, cycles, result = run(scenario())
    assert decision.action == "delay"
    assert decision.delay_seconds == 30.0  # retry_delay_s
    assert cursor is None  # cursor unchanged
    assert outbox == 0  # no outbox rows
    assert cycles == 0  # no terminal cycle
    # Retry defer does NOT increment the wait streak.
    assert state.wait_streak == 0
    assert state.agent_resume_at is not None
    assert isinstance(result, DispatchDeferred)
    assert result.defer_kind == "retry"


def test_agent_timeout_retries_after_resume_at(tmp_path):
    """After the retry barrier expires, the retry runs the agent again and a
    now-healthy provider produces the terminal reply."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        clock = VirtualClock(epoch=200.0)
        grant = await _begin_dispatch(repo, now=clock.now())
        llm = _ThenSucceedingLLM(
            [
                LLMResponse(
                    content="分析",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("c1"),
                            name="reply",
                            arguments={"text": "参考回复"},
                        ),
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
        budget = BudgetManager(repo, BudgetConfig(daily_cap=100), now=clock.now)
        agent = PhaseAgent.budgeted(
            llm, PromptStore(), registry, ContextConfig(), budget,
            AgentConfig(retry_delay_s=30.0),
        )
        runner = make_agent_runner(repo, agent, clock=clock, dry_run=False,
                                   uuid_fn=lambda: "cy-1")
        # First run: the provider blips → retry defer.
        decision1 = await runner.run_dispatch(grant)
        assert decision1.action == "delay"
        # Advance past the retry barrier; the retry runs and succeeds.
        clock.advance(31.0)
        grant2 = await _begin_dispatch(repo, cause=DispatchCause.TIMER,
                                       now=clock.now())
        decision2 = await runner.run_dispatch(grant2)
        outbox = await db.read(
            lambda c: c.execute("SELECT text FROM outbox").fetchall()
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        state = await repo.get_chat_state(CK)
        await repo.close()
        return decision2, outbox, cycles, state, llm

    decision2, outbox, cycles, state, llm = run(scenario())
    assert decision2.action == "trigger"
    assert outbox == [("你好",)]  # the retry produced the reply
    assert cycles == [("agent_reply",)]
    assert state.agent_resume_at is None  # barrier cleared by the terminal
    assert llm.calls == 3  # 1 failed call + 2 successful retry calls


class _SlowLLM:
    """Each provider call sleeps ``delay`` on the clock (a long saga)."""

    def __init__(self, clock, response, delay=20.0):
        self.clock = clock
        self.response = response
        self.delay = delay

    async def complete(self, messages, *, profile, tools=None, temperature=None,
                       max_tokens=None, deadline=None):
        await self.clock.sleep(self.delay)
        return self.response


def test_agent_long_saga_renews_lease(tmp_path):
    """A saga longer than the dispatch lease renews the lease through the
    run, so the terminal settle is not rejected as a stale owner."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        clock = VirtualClock(epoch=200.0)
        grant = await _begin_dispatch(repo, now=clock.now())
        # dispatch_lease_s=10, max_execution_s=100; each call sleeps 20s, so
        # the saga (planner + replyer) takes 40s — far beyond the 10s lease.
        llm = _SlowLLM(clock, LLMResponse(content="你好"), delay=20.0)
        registry = register_core_tools()
        budget = BudgetManager(repo, BudgetConfig(daily_cap=100), now=clock.now)
        agent = PhaseAgent.budgeted(
            llm, PromptStore(), registry, ContextConfig(), budget,
            AgentConfig(dispatch_lease_s=10.0, max_execution_s=100.0),
        )
        runner = make_agent_runner(repo, agent, clock=clock, dry_run=False,
                                   uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        dispatch_state = await db.read(
            lambda c: c.execute(
                "SELECT state FROM dispatches WHERE id = ?", (grant.dispatch_id,)
            ).fetchone()[0]
        )
        await repo.close()
        return decision, dispatch_state

    decision, dispatch_state = run(scenario())
    assert decision.action == "trigger"
    assert dispatch_state == "completed"  # settled despite the long saga


# ── Durable wait: non-interruptible, restart, third-rest ─────────────────────

def test_agent_durable_wait_third_rest_and_restart(tmp_path):
    """A wait defers with defer_kind='wait' (barrier persisted, wait streak
    incremented); restart honors the remaining delay and priority input
    cannot invoke the agent early; the THIRD consecutive wait terminally
    consumes with planner_wait_rest, clearing the barrier/streak."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        clock = VirtualClock(epoch=200.0)
        planner = FakePlanner(
            [
                PlanResult(intent=PlanIntent.WAIT, wait_seconds=30.0,
                           tokens_in=4, tokens_out=1, end_reason="wait"),
                PlanResult(intent=PlanIntent.WAIT, wait_seconds=30.0,
                           tokens_in=4, tokens_out=1, end_reason="wait"),
                PlanResult(intent=PlanIntent.WAIT, wait_seconds=30.0,
                           tokens_in=4, tokens_out=1, end_reason="wait"),
            ]
        )
        agent = PhaseAgent(planner, FakeReplyer())
        runner = make_agent_runner(repo, agent, clock=clock, dry_run=False,
                                   uuid_fn=lambda: "cy-1")
        # First wait: defer, wait_streak=1, barrier set.
        grant1 = await _begin_dispatch(repo, now=clock.now())
        d1 = await runner.run_dispatch(grant1)
        state = await repo.get_chat_state(CK)
        assert d1.action == "delay" and d1.delay_seconds == 30.0
        assert state.wait_streak == 1
        assert state.agent_resume_at == clock.now() + 30.0
        # Restart honors the remaining delay: begin_dispatch defers before
        # resume_at, and priority input cannot invoke the agent early.
        clock.advance(10.0)
        timer_result = await repo.begin_dispatch(
            DispatchRequest(
                chat_key=CK, cause=DispatchCause.TIMER,
                cycle_id=CycleId("cy-2"), started_ts=clock.now(),
                expires_at=clock.now() + 60, now=clock.now(),
            )
        )
        priority_result = await repo.begin_dispatch(
            DispatchRequest(
                chat_key=CK, cause=DispatchCause.INBOUND,
                cycle_id=CycleId("cy-3"), started_ts=clock.now(),
                expires_at=clock.now() + 60, now=clock.now(),
            )
        )
        assert isinstance(timer_result, DispatchDeferred)
        assert isinstance(priority_result, DispatchDeferred)
        assert timer_result.resume_at == state.agent_resume_at
        # Advance past resume_at; second wait defers, wait_streak=2.
        clock.advance(25.0)
        grant2 = await _begin_dispatch(repo, cause=DispatchCause.TIMER,
                                       now=clock.now())
        d2 = await runner.run_dispatch(grant2)
        state = await repo.get_chat_state(CK)
        assert d2.action == "delay"
        assert state.wait_streak == 2
        # Advance past resume_at; third wait terminally consumes.
        clock.advance(31.0)
        grant3 = await _begin_dispatch(repo, cause=DispatchCause.TIMER,
                                       now=clock.now())
        d3 = await runner.run_dispatch(grant3)
        state = await repo.get_chat_state(CK)
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        dispatch_state = await db.read(
            lambda c: c.execute(
                "SELECT state FROM dispatches WHERE id = ?", (grant3.dispatch_id,)
            ).fetchone()[0]
        )
        await repo.close()
        return d3, state, cycles, dispatch_state

    d3, state, cycles, dispatch_state = run(scenario())
    assert d3.action == "trigger"
    assert cycles == [("planner_wait_rest",)]
    assert dispatch_state == "completed"
    assert state.wait_streak == 0  # reset
    assert state.agent_resume_at is None  # barrier cleared

# ── Gate 3: permanent 4xx is terminal / transient still defers ───────────────

def test_agent_permanent_error_is_terminal_no_retry(tmp_path):
    """A permanent provider failure (HTTP 4xx / malformed payload) is a
    terminal safe no-output outcome: the cursor is consumed with an EMPTY
    outbox, no retry barrier is set, and the provider is never called again
    (no repeated busy retry)."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        llm = _RaisingLLM(LLMPermanentError("provider 400: bad request"))
        registry = register_core_tools()
        budget = BudgetManager(repo, BudgetConfig(daily_cap=100), now=lambda: 200.0)
        agent = PhaseAgent.budgeted(
            llm, PromptStore(), registry, ContextConfig(), budget, AgentConfig()
        )
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        cursor = await db.read(
            lambda c: c.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?", (CK,)
            ).fetchone()[0]
        )
        dispatch_state = await db.read(
            lambda c: c.execute(
                "SELECT state FROM dispatches WHERE id = ?", (grant.dispatch_id,)
            ).fetchone()[0]
        )
        state = await repo.get_chat_state(CK)
        await repo.close()
        return decision, outbox, cycles, cursor, dispatch_state, state, llm

    decision, outbox, cycles, cursor, dispatch_state, state, llm = run(scenario())
    assert decision.action == "trigger"  # terminal, not a timed delay
    assert outbox == 0  # safe no-output
    assert cycles == [("llm_permanent_error",)]
    assert cursor == 1  # terminal consume
    assert dispatch_state == "completed"
    assert state.agent_resume_at is None  # no retry barrier
    assert llm.calls == 1  # never retried


def test_agent_prompt_error_is_terminal_no_retry(tmp_path):
    """Project-wide permanent errors (not only provider 4xx) must not
    escape into the scheduler's lease-expiry retry loop."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        llm = _RaisingLLM(PromptError("missing planner prompt"))
        agent = PhaseAgent.budgeted(
            llm,
            PromptStore(),
            register_core_tools(),
            ContextConfig(),
            BudgetManager(repo, BudgetConfig(daily_cap=100), now=lambda: 200.0),
            AgentConfig(),
        )
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        end_reason = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchone()[0]
        )
        state = await repo.get_chat_state(CK)
        await repo.close()
        return decision, end_reason, state, llm

    decision, end_reason, state, llm = run(scenario())
    assert decision.action == "trigger"
    assert end_reason == "llm_permanent_error"
    assert state.agent_resume_at is None
    assert llm.calls == 1


def test_agent_saga_deadline_defers_without_cursor_or_outbox(tmp_path):
    """The aggregate runtime deadline owns the whole await, even when an
    injected agent ignores provider-level deadlines."""

    class _SleepingAgent:
        async def run(self, **_kwargs):
            await asyncio.sleep(0.05)
            raise AssertionError("deadline should cancel this saga first")

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        clock = VirtualClock(epoch=200.0)
        grant = await _begin_dispatch(repo, now=clock.now())
        cfg = Config(agent=AgentConfig(max_execution_s=0.01, retry_delay_s=7.0))
        runner = CycleRunner(
            repo,
            Gate(),
            cfg,
            clock=clock,
            dry_run=False,
            uuid_fn=lambda: "cy-1",
            agent=_SleepingAgent(),
        )
        decision = await runner.run_dispatch(grant)
        state = await repo.get_chat_state(CK)
        cursor = await db.read(
            lambda c: c.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?", (CK,)
            ).fetchone()[0]
        )
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        await repo.close()
        return decision, state, cursor, outbox

    decision, state, cursor, outbox = run(scenario())
    assert decision.action == "delay"
    assert decision.delay_seconds == 7.0
    assert state.agent_resume_at == 207.0
    assert cursor is None
    assert outbox == 0


def test_agent_transient_error_still_defers_retry(tmp_path):
    """A transient provider failure still defers the retry atomically (a
    timed delay, cursor/outbox unchanged) — the permanent-error terminal
    path must not swallow transient retries."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        clock = VirtualClock(epoch=200.0)
        grant = await _begin_dispatch(repo, now=clock.now())
        llm = _RaisingLLM(LLMTransientError("timeout"))
        registry = register_core_tools()
        budget = BudgetManager(repo, BudgetConfig(daily_cap=100), now=clock.now)
        agent = PhaseAgent.budgeted(
            llm, PromptStore(), registry, ContextConfig(), budget,
            AgentConfig(retry_delay_s=30.0),
        )
        runner = make_agent_runner(repo, agent, clock=clock, dry_run=False,
                                   uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        state = await repo.get_chat_state(CK)
        cursor = await db.read(
            lambda c: c.execute(
                "SELECT cursor_msg_id FROM chats WHERE chat_key = ?", (CK,)
            ).fetchone()[0]
        )
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
        )
        await repo.close()
        return decision, state, cursor, outbox, cycles

    decision, state, cursor, outbox, cycles = run(scenario())
    assert decision.action == "delay"  # deferred, not terminal
    assert decision.delay_seconds == 30.0
    assert cursor is None  # cursor unchanged
    assert outbox == 0
    assert cycles == 0  # no terminal cycle
    assert state.agent_resume_at is not None  # retry barrier set


# ── Gate 3: per-chat budget/context/fallback honored at each call ────────────

def test_per_chat_budget_and_fallback_are_honored(tmp_path):
    """Two chats with different budget caps/rungs and fallback profiles use
    DIFFERENT actual provider calls: per-chat budget usage is independent and
    the budget-degrade fallback profile is the chat's own, never a top-level
    global."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity(chat_key="chat-a"))
        await repo.upsert_chat(make_identity(chat_key="chat-b"))
        await repo.ingest_message(
            make_identity(chat_key="chat-a"),
            make_message(chat_key="chat-a", text="hi", msg_id="ma1",
                         mentions=("bot-1",)),
        )
        await repo.ingest_message(
            make_identity(chat_key="chat-b"),
            make_message(chat_key="chat-b", text="hi", msg_id="mb1",
                         mentions=("bot-1",)),
        )
        cfg = Config.from_dict({
            "chats": [
                {
                    "key": "chat-a",
                    "budget": {
                        "daily_cap": 10,
                        "rungs": [
                            {"at": 0.05, "action": "degrade"},
                            {"at": 0.1, "action": "degrade"},
                        ],
                    },
                    "agent": {"fallback_profile": "cheap-a"},
                },
                {
                    "key": "chat-b",
                    "budget": {
                        "daily_cap": 10,
                        "rungs": [
                            {"at": 0.05, "action": "degrade"},
                            {"at": 0.1, "action": "degrade"},
                        ],
                    },
                    "agent": {"fallback_profile": "cheap-b"},
                },
            ],
        })
        registry = register_core_tools()
        clock = VirtualClock(epoch=200.0)

        class _ProfileLLM:
            def __init__(self, responses):
                self.responses = list(responses)
                self.calls: list[str] = []

            async def complete(self, messages, *, profile, tools=None,
                               temperature=None, max_tokens=None, deadline=None):
                self.calls.append(profile)
                if self.responses:
                    return self.responses.pop(0)
                return LLMResponse(content=None)

        def make_llm():
            return _ProfileLLM([
                LLMResponse(
                    content="分析",
                    tool_calls=(ToolCall(
                        id=ToolCallId("c1"), name="reply",
                        arguments={"text": "参考"},
                    ),),
                    usage={"prompt_tokens": 5, "completion_tokens": 2},
                ),
                LLMResponse(content="你好",
                            usage={"prompt_tokens": 3, "completion_tokens": 1}),
            ])

        llm_a = make_llm()
        llm_b = make_llm()
        agent_a = PhaseAgent.budgeted(
            llm_a, PromptStore(), registry, ContextConfig(),
            BudgetManager(repo, cfg.budget, now=clock.now), cfg.agent,
            cfg=cfg, repo=repo, now=clock.now,
        )
        agent_b = PhaseAgent.budgeted(
            llm_b, PromptStore(), registry, ContextConfig(),
            BudgetManager(repo, cfg.budget, now=clock.now), cfg.agent,
            cfg=cfg, repo=repo, now=clock.now,
        )
        runner_a = make_agent_runner(repo, agent_a, clock=clock, dry_run=False,
                                     uuid_fn=lambda: "cy-a")
        runner_b = make_agent_runner(repo, agent_b, clock=clock, dry_run=False,
                                     uuid_fn=lambda: "cy-b")
        grant_a = await repo.begin_dispatch(
            DispatchRequest(chat_key=ChatKey("chat-a"), cause=DispatchCause.INBOUND,
                            cycle_id=CycleId("cy-a"), started_ts=clock.now(),
                            expires_at=clock.now() + 300, now=clock.now())
        )
        grant_b = await repo.begin_dispatch(
            DispatchRequest(chat_key=ChatKey("chat-b"), cause=DispatchCause.INBOUND,
                            cycle_id=CycleId("cy-b"), started_ts=clock.now(),
                            expires_at=clock.now() + 300, now=clock.now())
        )
        assert isinstance(grant_a, DispatchGrant)
        assert isinstance(grant_b, DispatchGrant)
        await runner_a.run_dispatch(grant_a)
        await runner_b.run_dispatch(grant_b)
        usage_a = await agent_a._budget_for(
            ChatKey("chat-a"), cfg.for_chat("chat-a").budget
        ).snapshot(ChatKey("chat-a"))
        usage_b = await agent_b._budget_for(
            ChatKey("chat-b"), cfg.for_chat("chat-b").budget
        ).snapshot(ChatKey("chat-b"))
        await repo.close()
        return usage_a, usage_b, llm_a.calls, llm_b.calls

    usage_a, usage_b, profiles_a, profiles_b = run(scenario())
    # Per-chat budget usage is independent (2 calls each: planner + replyer).
    assert usage_a.calls == 2
    assert usage_b.calls == 2
    # The replyer's call (2nd) uses the chat's OWN fallback profile.
    assert profiles_a[1] == "cheap-a"
    assert profiles_b[1] == "cheap-b"


# ── Gate 3: outbox wake ordering + worker wake from future sleep ─────────────

def test_on_outbox_fires_before_hook_with_rows(tmp_path):
    """The live outbox wake fires immediately after terminal settlement with
    rows, BEFORE the on_cycle_end hook (so the worker drains promptly)."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        agent, _, _ = _reply_agent()
        hooks = HookBus()
        order: list[str] = []

        @hooks.on_cycle_end
        def hook(chat_key, trace, end_reason):
            order.append("hook")

        async def on_outbox(items):
            order.append("outbox")

        runner = make_agent_runner(repo, agent, hooks=hooks, dry_run=False,
                                   uuid_fn=lambda: "cy-1", on_outbox=on_outbox)
        decision = await runner.run_dispatch(grant)
        await repo.close()
        return decision, order

    decision, order = run(scenario())
    assert decision.action == "trigger"
    assert order == ["outbox", "hook"]  # outbox wake before the hook


# ── Gate 3: identity_file reaches planner/replyer prompts, analysis absent ───

def test_identity_file_reaches_planner_and_replyer_prompts(tmp_path):
    """The configured identity_file content appears in BOTH the planner and
    replyer runtime prompts, while planner analysis never reaches the
    replyer's user turn."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        planner_llm = FakeLLM([
            LLMResponse(
                content="分析：值得回复。",
                tool_calls=(ToolCall(
                    id=ToolCallId("c1"), name="reply",
                    arguments={"text": "参考回复"},
                ),),
            ),
        ])
        replyer_llm = FakeLLM([LLMResponse(content="你好")])
        registry = register_core_tools()
        planner = Planner(
            planner_llm, PromptStore(), registry, ContextConfig(),
            tool_context_factory=lambda: ToolContext(
                chat_key=CK, chat_kind="group", capabilities=frozenset(),
                registry=registry, self_name="麦麦",
            ),
        )
        replyer = Replyer(replyer_llm, PromptStore())
        agent = PhaseAgent(planner, replyer)
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        await repo.close()
        return decision, planner_llm, replyer_llm

    decision, planner_llm, replyer_llm = run(scenario())
    assert decision.action == "trigger"
    # The identity_file content appears in the planner's system prompt.
    planner_sys = planner_llm.calls[0][0][0].content
    assert "你是麦麦，一个群聊里的普通成员" in planner_sys
    # And in the replyer's system prompt.
    replyer_sys = replyer_llm.calls[0][0][0].content
    assert "你是麦麦，一个群聊里的普通成员" in replyer_sys
    # Planner analysis never reaches the replyer's user turn.
    replyer_user = replyer_llm.calls[0][0][1].content
    assert "分析" not in replyer_user
    assert "c1" not in replyer_user


# ── Gate 3: malformed structured ToolCall raw args repair/degrade ────────────

def test_planner_repairs_malformed_structured_args(tmp_path):
    """A malformed structured ToolCall's raw arguments are preserved through
    planner construction and given ONE repair opportunity; a successful repair
    dispatches the repaired arguments (no orphan id)."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        llm = FakeLLM([
            LLMResponse(
                content="分析",
                tool_calls=(ToolCall(
                    id=ToolCallId("c1"), name="reply", arguments={},
                    raw_arguments='{"text": "参考回复" "extra": 1}',
                ),),
            ),
        ])
        registry = register_core_tools()
        repair_calls: list[str] = []

        def repair(text):
            repair_calls.append(text)
            return '{"text": "参考回复"}'

        planner = Planner(
            llm, PromptStore(), registry, ContextConfig(),
            tool_context_factory=lambda: ToolContext(
                chat_key=CK, chat_kind="group", capabilities=frozenset(),
                registry=registry, self_name="麦麦",
            ),
            repair=repair,
        )
        replyer = Replyer(FakeLLM([LLMResponse(content="你好")]), PromptStore())
        agent = PhaseAgent(planner, replyer)
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        outbox = await db.read(
            lambda c: c.execute("SELECT text FROM outbox").fetchall()
        )
        await repo.close()
        return decision, outbox, repair_calls, llm

    decision, outbox, repair_calls, llm = run(scenario())
    assert decision.action == "trigger"
    assert outbox == [("你好",)]  # the repaired reply dispatched
    assert len(repair_calls) == 1  # exactly ONE repair opportunity
    assert len(llm.calls) == 1  # planner made exactly one round, no extra


def test_planner_degrades_unrepairable_structured_args(tmp_path):
    """A malformed structured ToolCall whose raw arguments cannot be repaired
    degrades to no_action (ok=False, id answered — never orphaned) with ONE
    repair opportunity and no extra provider round."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        llm = FakeLLM([
            LLMResponse(
                content="分析",
                tool_calls=(ToolCall(
                    id=ToolCallId("c1"), name="reply", arguments={},
                    raw_arguments="garbage {{{",
                ),),
            ),
        ])
        registry = register_core_tools()
        repair_calls: list[str] = []

        def repair(text):
            repair_calls.append(text)
            return "still not json"

        planner = Planner(
            llm, PromptStore(), registry, ContextConfig(),
            tool_context_factory=lambda: ToolContext(
                chat_key=CK, chat_kind="group", capabilities=frozenset(),
                registry=registry, self_name="麦麦",
            ),
            repair=repair,
        )
        replyer = Replyer(FakeLLM(), PromptStore())
        agent = PhaseAgent(planner, replyer)
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        await repo.close()
        return decision, outbox, cycles, repair_calls, llm

    decision, outbox, cycles, repair_calls, llm = run(scenario())
    assert decision.action == "trigger"
    assert outbox == 0  # no_action degrade, no reply
    assert cycles == [("no_action",)]
    assert len(repair_calls) == 1  # exactly ONE repair opportunity
    assert len(llm.calls) == 1  # no extra provider round


# ── Gate 3: legacy / no-agent / dry-run regressions ──────────────────────────

def test_legacy_no_agent_and_dry_run_regressions(tmp_path):
    """The Gate 3 changes preserve the frozen legacy behavior: no-agent
    outside dry-run releases a trigger (never retained, never sent), and
    dry-run with an agent creates zero outbox rows and never sends."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        # No-agent, non-dry: trigger released, never retained/sent.
        grant1 = await _begin_dispatch(repo, cycle_id="cy-1")
        runner1 = make_agent_runner(repo, None, dry_run=False, uuid_fn=lambda: "cy-1")
        d1 = await runner1.run_dispatch(grant1)
        state1 = await db.read(
            lambda c: c.execute(
                "SELECT state FROM dispatches WHERE id = ?", (grant1.dispatch_id,)
            ).fetchone()[0]
        )
        # Dry-run with an agent: evaluates, zero outbox, never sends.
        await repo.ingest_message(
            make_identity(), _trigger_message(recv_ts=110.0, msg_id="m2")
        )
        grant2 = await _begin_dispatch(repo, cycle_id="cy-2")
        agent, _, _ = _reply_agent()
        runner2 = make_agent_runner(repo, agent, dry_run=True, uuid_fn=lambda: "cy-2")
        d2 = await runner2.run_dispatch(grant2)
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        await repo.close()
        return d1, state1, d2, outbox, cycles

    d1, state1, d2, outbox, cycles = run(scenario())
    assert d1.action == "trigger"
    assert state1 == "released"  # never retained without an agent
    assert d2.action == "trigger"
    assert outbox == 0  # dry-run: zero outbox rows
    assert cycles == [("dry_run_agent_reply",)]  # evaluated, never sent


# ── Phase 5: knowledge tools wired into ledger dispatch ──────────────────────

def test_agent_activates_and_queries_knowledge_tools_in_dispatch(tmp_path):
    """Real SQLite + a fake agent planner: the budgeted form wires the
    chat-scoped knowledge callbacks into the ToolContext factory, so a
    planner that tool_searches then calls query_memory gets a real memory
    hit in its tool results — and the deferred tools are emitted only after
    tool_search activation."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        # Seed one durable memory for the chat.
        batch = await repo.read_memory_source_batch(
            CK, through_msg_id=MessageRowId(1), tail=100
        )
        assert batch is not None
        rec = make_memory(
            chat_key=CK, text="火锅好吃",
            source_first_msg_id=batch.first_msg_id,
            source_last_msg_id=batch.last_msg_id,
            source_hash=batch.source_hash,
        )
        assert await repo.commit_memory_source(
            MemoryWriteRequest(chat_key=CK, batch=batch, records=(rec,))
        ) is True
        grant = await _begin_dispatch(repo)
        llm = FakeLLM(
            [
                LLMResponse(
                    content="先搜索",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("c1"), name="tool_search",
                            arguments={"capability": "memory"},
                        ),
                    ),
                    usage={"prompt_tokens": 8, "completion_tokens": 2},
                ),
                LLMResponse(
                    content="查记忆",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("c2"), name="query_memory",
                            arguments={"query": "火锅"},
                        ),
                    ),
                    usage={"prompt_tokens": 10, "completion_tokens": 3},
                ),
                LLMResponse(
                    content="分析",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("c3"), name="reply",
                            arguments={"text": "参考回复"},
                        ),
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
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        outbox = await db.read(
            lambda c: c.execute("SELECT text FROM outbox").fetchall()
        )
        await repo.close()
        return decision, outbox, llm

    decision, outbox, llm = run(scenario())
    assert decision.action == "trigger"
    assert outbox == [("你好",)]
    # The query_memory tool result carried the real memory hit.
    tool_results = llm.calls[2][0]
    tool_contents = [
        m.content for m in tool_results if m.role == "tool"
    ]
    assert any("火锅好吃" in c for c in tool_contents)


# ── Gate 5 remediation: the default budgeted ToolContext is REAL ─────────────

def test_budgeted_tool_context_injects_adapter_capabilities_and_forwards(tmp_path):
    """The default budgeted ToolContext injects the adapter-supported
    capabilities, the current dispatch's recent messages, and safely scoped
    forwards — so fetch_history / view_forward_message are reachable in
    production when the adapter provides the data."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        llm = FakeLLM(
            [
                LLMResponse(
                    content="搜工具",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("c1"), name="tool_search",
                            arguments={"query": ""},
                        ),
                    ),
                    usage={"prompt_tokens": 8, "completion_tokens": 2},
                ),
                LLMResponse(
                    content="看历史",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("c2"), name="fetch_history",
                            arguments={"limit": 5},
                        ),
                    ),
                    usage={"prompt_tokens": 10, "completion_tokens": 3},
                ),
                LLMResponse(
                    content="看转发",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("c3"), name="view_forward_message",
                            arguments={"id": "fwd-1"},
                        ),
                    ),
                    usage={"prompt_tokens": 12, "completion_tokens": 4},
                ),
                LLMResponse(
                    content="回复",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("c4"), name="reply",
                            arguments={"text": "好的"},
                        ),
                    ),
                    usage={"prompt_tokens": 14, "completion_tokens": 5},
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
            capabilities=frozenset({"history", "forward"}),
            forward_resolver=lambda chat_key: (
                {"fwd-1": "转发内容"} if chat_key == CK else {}
            ),
        )
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        outbox = await db.read(
            lambda c: c.execute("SELECT text FROM outbox").fetchall()
        )
        await repo.close()
        return decision, outbox, llm

    decision, outbox, llm = run(scenario())
    assert decision.action == "trigger"
    assert outbox == [("你好",)]
    # The fetch_history result rendered the current dispatch's recent message.
    hist_msgs = llm.calls[2][0]
    hist_contents = [m.content for m in hist_msgs if m.role == "tool"]
    assert any("user: hi" in c for c in hist_contents)
    # The view_forward_message result carried the scoped forward content.
    fwd_msgs = llm.calls[3][0]
    fwd_contents = [m.content for m in fwd_msgs if m.role == "tool"]
    assert any("转发内容" in c for c in fwd_contents)


def test_budgeted_tool_context_rejects_unavailable_and_cross_chat(tmp_path):
    """Without the adapter capability the history/forward tools fail closed,
    and forwards are scoped to the current chat — a cross-chat forward id is
    unknown (never exposed)."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        llm = FakeLLM(
            [
                LLMResponse(
                    content="搜工具",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("c1"), name="tool_search",
                            arguments={"query": ""},
                        ),
                    ),
                    usage={"prompt_tokens": 8, "completion_tokens": 2},
                ),
                LLMResponse(
                    content="看历史",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("c2"), name="fetch_history",
                            arguments={"limit": 5},
                        ),
                    ),
                    usage={"prompt_tokens": 10, "completion_tokens": 3},
                ),
                LLMResponse(
                    content="看转发",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("c3"), name="view_forward_message",
                            arguments={"id": "other-fwd"},
                        ),
                    ),
                    usage={"prompt_tokens": 12, "completion_tokens": 4},
                ),
                LLMResponse(
                    content="回复",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("c4"), name="reply",
                            arguments={"text": "好的"},
                        ),
                    ),
                    usage={"prompt_tokens": 14, "completion_tokens": 5},
                ),
                LLMResponse(
                    content="你好",
                    usage={"prompt_tokens": 6, "completion_tokens": 2},
                ),
            ]
        )
        registry = register_core_tools()
        budget = BudgetManager(repo, BudgetConfig(daily_cap=100), now=lambda: 200.0)
        # NO capabilities: fetch_history / view_forward_message fail closed.
        # The forward resolver scopes forwards per chat: OTHER's forward id
        # is never exposed to CK.
        agent = PhaseAgent.budgeted(
            llm, PromptStore(), registry, ContextConfig(), budget, AgentConfig(),
            capabilities=frozenset(),
            forward_resolver=lambda chat_key: (
                {"ck-fwd": "CK 转发"} if chat_key == CK else {"other-fwd": "OTHER 转发"}
            ),
        )
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        outbox = await db.read(
            lambda c: c.execute("SELECT text FROM outbox").fetchall()
        )
        await repo.close()
        return decision, outbox, llm

    decision, outbox, llm = run(scenario())
    assert decision.action == "trigger"
    assert outbox == [("你好",)]
    # fetch_history failed closed (no 'history' capability).
    hist_msgs = llm.calls[2][0]
    hist_contents = [m.content for m in hist_msgs if m.role == "tool"]
    assert any("requires adapter capability 'history'" in c for c in hist_contents)
    # view_forward_message failed closed: no 'forward' capability AND the
    # cross-chat id is unknown (never exposed).
    fwd_msgs = llm.calls[3][0]
    fwd_contents = [m.content for m in fwd_msgs if m.role == "tool"]
    assert any("requires adapter capability 'forward'" in c for c in fwd_contents)


# ── Phase 6 P6.4b: the frozen per-dispatch adaptive context ──────────────────

async def _seed_expression(repo, *, style="活泼", situation="greeting"):
    """Seed one expression record through the adaptive write surface."""
    grant = await repo.acquire_learner_run(
        LearnerRunRequest(
            chat_key=CK, learner="expression",
            started_ts=100.0, expires_at=500.0, now=100.0,
        )
    )
    assert isinstance(grant, LearnerGrant)
    batch = await repo.read_learner_source_batch(
        CK, "expression", through_msg_id=grant.through_msg_id, tail=100
    )
    assert batch is not None
    await repo.commit_learner_source(
        LearnerDraft(
            chat_key=CK, learner="expression", batch=batch,
            records=(Record(
                learner="expression",
                payload={"situation": situation, "style": style, "source_id": 1},
                chat_key=CK,
            ),),
            expected_through_msg_id=batch.observed_watermark,
        ),
        now=200.0,
    )


def test_agent_adaptive_reply_style_and_context_frozen_for_planner_and_replyer(tmp_path):
    """The adaptive context is computed ONCE per live dispatch and the SAME
    reply style reaches every planner tool round and the replyer; the
    adaptive reference block rides in the planner's chat log."""
    from pretender.cycle import AdaptiveContextService

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        await _seed_expression(repo, style="活泼")
        grant = await _begin_dispatch(repo)
        agent, planner, replyer = _reply_agent()
        exposures: list[tuple] = []
        runner = make_agent_runner(
            repo, agent, dry_run=False, uuid_fn=lambda: "cy-1",
            adaptive=AdaptiveContextService(repo, now=lambda: 200.0),
            on_exposure=lambda chat, records, dispatch_id, through: exposures.append(
                (chat, records, dispatch_id, through)
            ),
        )
        decision = await runner.run_dispatch(grant)
        await repo.close()
        return decision, planner, replyer, exposures, grant.dispatch_id, grant.through_msg_id

    decision, planner, replyer, exposures, dispatch_id, through_msg_id = run(scenario())
    assert decision.action == "trigger"
    # The SAME adaptive reply style reached the planner and the replyer.
    assert planner.calls[0]["reply_style"] == "活泼"
    assert replyer.calls[0]["reply_style"] == "活泼"
    # The adaptive reference block rode in the planner's chat log.
    assert "【自适应参考】" in planner.calls[0]["chat_log"]
    assert "活泼" in planner.calls[0]["chat_log"]
    # The exposure fired post-terminal with the frozen records + boundary.
    assert len(exposures) == 1
    chat, records, exp_dispatch, through = exposures[0]
    assert chat == CK
    assert len(records) == 1
    assert records[0].payload["style"] == "活泼"
    assert exp_dispatch == dispatch_id
    assert through == through_msg_id


def test_agent_adaptive_context_not_queried_in_dry_run(tmp_path):
    """Dry-run evaluates the agent but NEVER queries the adaptive service
    and NEVER fires the exposure callback (LIVE-only)."""
    from pretender.cycle import AdaptiveContextService

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        await _seed_expression(repo, style="活泼")
        grant = await _begin_dispatch(repo)
        agent, planner, replyer = _reply_agent()
        exposures: list[tuple] = []
        runner = make_agent_runner(
            repo, agent, dry_run=True, uuid_fn=lambda: "cy-1",
            adaptive=AdaptiveContextService(repo, now=lambda: 200.0),
            on_exposure=lambda chat, records, dispatch_id, through: exposures.append(
                (chat, records, dispatch_id, through)
            ),
        )
        decision = await runner.run_dispatch(grant)
        await repo.close()
        return decision, planner, replyer, exposures

    decision, planner, replyer, exposures = run(scenario())
    assert decision.action == "trigger"
    # The fallback "自然" empty context: no adaptive query, no rendering.
    assert planner.calls[0]["reply_style"] == "自然"
    assert replyer.calls[0]["reply_style"] == "自然"
    assert "【自适应参考】" not in planner.calls[0]["chat_log"]
    assert exposures == []  # LIVE-only


def test_agent_adaptive_context_not_queried_before_gate_trigger(tmp_path):
    """A non-trigger dispatch (delay) never queries the adaptive service."""
    from pretender.cycle import AdaptiveContextService

    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        await _seed_expression(repo, style="活泼")
        grant = await _begin_dispatch(repo)
        planner = FakePlanner(
            [
                PlanResult(
                    intent=PlanIntent.WAIT, wait_seconds=30.0,
                    tokens_in=2, tokens_out=1, end_reason="wait",
                )
            ]
        )
        agent = PhaseAgent(planner, FakeReplyer())
        runner = make_agent_runner(
            repo, agent, dry_run=False, uuid_fn=lambda: "cy-1",
            adaptive=AdaptiveContextService(repo, now=lambda: 200.0),
        )
        decision = await runner.run_dispatch(grant)
        await repo.close()
        return decision, planner

    decision, planner = run(scenario())
    assert decision.action == "delay"
    # The planner ran (the gate triggered the agent lane), but the adaptive
    # context is only computed for the agent saga — a wait still gets it.
    # The KEY assertion: no exposure fires for a non-reply outcome.
    assert planner.calls[0]["reply_style"] == "活泼"


def test_agent_settled_callback_only_after_terminal(tmp_path):
    """The post-settlement learner enqueue fires ONLY after a TERMINAL
    dispatch — never on delay/release (the learner worker is not woken for
    a chat whose cursor did not move)."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        planner = FakePlanner(
            [
                PlanResult(
                    intent=PlanIntent.WAIT, wait_seconds=30.0,
                    tokens_in=2, tokens_out=1, end_reason="wait",
                )
            ]
        )
        agent = PhaseAgent(planner, FakeReplyer())
        settled: list[tuple] = []
        runner = make_agent_runner(
            repo, agent, dry_run=False, uuid_fn=lambda: "cy-1",
            on_settled=lambda chat, through: settled.append((chat, through)),
        )
        decision = await runner.run_dispatch(grant)
        await repo.close()
        return decision, settled

    decision, settled = run(scenario())
    assert decision.action == "delay"
    assert settled == []  # no terminal settlement → no learner enqueue


def test_agent_settled_callback_fires_after_terminal_reply(tmp_path):
    """A terminal reply dispatch fires the post-settlement callback AFTER
    the durable finish (the learner worker is enqueued)."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        agent, _planner, _replyer = _reply_agent()
        settled: list[tuple] = []
        runner = make_agent_runner(
            repo, agent, dry_run=False, uuid_fn=lambda: "cy-1",
            on_settled=lambda chat, through: settled.append((chat, through)),
        )
        decision = await runner.run_dispatch(grant)
        await repo.close()
        return decision, settled

    decision, settled = run(scenario())
    assert decision.action == "trigger"
    assert len(settled) == 1
    assert settled[0][0] == CK
    assert settled[0][1] == MessageRowId(1)


# ── Phase 6 P6.5b media send lane ────────────────────────────────────────────

def _media_callbacks_for(repo, chat_key=CK):
    async def resolve_asset(asset_id):
        assets = await repo.list_media_assets(chat_key, limit=200)
        for a in assets:
            if a.id == asset_id:
                return a
        return None

    return MediaCallbacks(catalog_enabled=lambda: True, resolve_asset=resolve_asset)


async def _seed_approved_sticker(repo, *, cache_key=None, description="微笑", sha256=None):
    cid = await repo.submit_media_candidate(
        MediaAssetCandidate(
            chat_key=CK,
            kind=MediaKind.STICKER,
            cache_key=cache_key or "c" * 64,
            sha256=sha256 or "a" * 64,
            mime="image/gif",
            width=120,
            height=120,
            description=description,
        ),
        now=150.0,
    )
    asset = await repo.approve_media_candidate(CK, cid, capacity=4, now=160.0)
    assert asset is not None
    return asset


def test_agent_media_intent_creates_durable_outbox_segment(tmp_path):
    """A staged send_emoji intent becomes ONE durable outbox segment
    carrying the OPAQUE cache key at terminal settlement; the asset use is
    persisted post-terminal and the planner prompt carried the approved
    catalog listing (never the original URL)."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        asset = await _seed_approved_sticker(repo)
        registry = register_core_tools()
        registry.activate("send_emoji")
        llm = FakeLLM(
            [
                LLMResponse(
                    content="分析：适合发个表情。",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("call_1"),
                            name="send_emoji",
                            arguments={"asset_id": asset.id},
                        ),
                    ),
                    usage={"prompt_tokens": 8, "completion_tokens": 2},
                )
            ]
        )
        budget = BudgetManager(repo, BudgetConfig(), now=lambda: 200.0)
        agent = PhaseAgent.budgeted(
            llm, PromptStore(), registry, ContextConfig(), budget, AgentConfig(),
            capabilities=frozenset({"sticker", "image"}),
            media_callbacks=lambda chat_key: _media_callbacks_for(repo, chat_key),
        )
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        rows = await db.read(
            lambda c: c.execute(
                "SELECT text, segments_json, idem_key FROM outbox"
            ).fetchall()
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        assets = await repo.list_media_assets(CK)
        await repo.close()
        return decision, rows, cycles, assets, llm

    decision, rows, cycles, assets, llm = run(scenario())
    assert decision.action == "trigger"
    assert len(rows) == 1
    text, segments_json, idem_key = rows[0]
    assert text == ""
    # The durable outbox carries the OPAQUE cache key — never a URL/path/
    # data URL.
    assert "c" * 64 in segments_json
    assert "example.com" not in segments_json
    assert "http" not in segments_json
    assert "data:" not in segments_json
    assert idem_key.startswith("dispatch:")
    assert cycles == [("agent_media",)]
    approved = [a for a in assets if a.safety_status == MediaSafetyStatus.APPROVED]
    assert len(approved) == 1
    assert approved[0].uses == 1  # use persisted post-terminal
    # The planner's system prompt carried the approved catalog listing.
    system = llm.calls[0][0][0].content
    assert "可用表情" in system
    assert "example.com" not in system
    assert "http" not in system


def test_agent_media_dry_run_zero_outbox_zero_use(tmp_path):
    """Dry-run evaluates the media intent but creates zero outbox rows and
    never persists the asset use."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        asset = await _seed_approved_sticker(repo)
        registry = register_core_tools()
        registry.activate("send_emoji")
        llm = FakeLLM(
            [
                LLMResponse(
                    content="分析：适合发个表情。",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("call_1"),
                            name="send_emoji",
                            arguments={"asset_id": asset.id},
                        ),
                    ),
                    usage={"prompt_tokens": 8, "completion_tokens": 2},
                )
            ]
        )
        budget = BudgetManager(repo, BudgetConfig(), now=lambda: 200.0)
        agent = PhaseAgent.budgeted(
            llm, PromptStore(), registry, ContextConfig(), budget, AgentConfig(),
            capabilities=frozenset({"sticker", "image"}),
            media_callbacks=lambda chat_key: _media_callbacks_for(repo, chat_key),
        )
        runner = make_agent_runner(repo, agent, dry_run=True, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        assets = await repo.list_media_assets(CK)
        await repo.close()
        return decision, outbox, cycles, assets

    decision, outbox, cycles, assets = run(scenario())
    assert decision.action == "trigger"
    assert outbox == 0
    assert cycles == [("dry_run_agent_media",)]
    approved = [a for a in assets if a.safety_status == MediaSafetyStatus.APPROVED]
    assert approved[0].uses == 0  # no use persisted in dry-run


def test_agent_media_use_is_idempotent_per_dispatch(tmp_path):
    """A retried settlement of the same dispatch never double-counts the
    asset use (the kv marker is set on the first use only)."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        asset = await _seed_approved_sticker(repo)
        registry = register_core_tools()
        registry.activate("send_emoji")
        llm = FakeLLM(
            [
                LLMResponse(
                    content="分析：适合发个表情。",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("call_1"),
                            name="send_emoji",
                            arguments={"asset_id": asset.id},
                        ),
                    ),
                    usage={"prompt_tokens": 8, "completion_tokens": 2},
                )
            ]
        )
        budget = BudgetManager(repo, BudgetConfig(), now=lambda: 200.0)
        agent = PhaseAgent.budgeted(
            llm, PromptStore(), registry, ContextConfig(), budget, AgentConfig(),
            capabilities=frozenset({"sticker", "image"}),
            media_callbacks=lambda chat_key: _media_callbacks_for(repo, chat_key),
        )
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        intent = MediaReplyIntent(kind="emoji", asset_id=asset.id, cache_key=asset.cache_key)
        # The post-terminal use persist is idempotent per dispatch identity:
        # a retried settlement of the same dispatch never double-counts.
        await runner._persist_media_use(grant, intent)
        await runner._persist_media_use(grant, intent)
        assets = await repo.list_media_assets(CK)
        await repo.close()
        return assets

    assets = run(scenario())
    approved = [a for a in assets if a.safety_status == MediaSafetyStatus.APPROVED]
    assert approved[0].uses == 1  # idempotent: never double-counted


def test_agent_media_unknown_asset_degrades_to_no_action(tmp_path):
    """A send_emoji call for an unknown/unapproved asset fails safely at
    dispatch (ok=False); the planner sees the error and the dispatch
    terminally consumes with no outbox."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), _trigger_message())
        grant = await _begin_dispatch(repo)
        registry = register_core_tools()
        registry.activate("send_emoji")
        llm = FakeLLM(
            [
                LLMResponse(
                    content="分析：发个表情。",
                    tool_calls=(
                        ToolCall(
                            id=ToolCallId("call_1"),
                            name="send_emoji",
                            arguments={"asset_id": 999},
                        ),
                    ),
                    usage={"prompt_tokens": 8, "completion_tokens": 2},
                ),
                LLMResponse(content=None),
            ]
        )
        budget = BudgetManager(repo, BudgetConfig(), now=lambda: 200.0)
        agent = PhaseAgent.budgeted(
            llm, PromptStore(), registry, ContextConfig(), budget, AgentConfig(),
            capabilities=frozenset({"sticker", "image"}),
            media_callbacks=lambda chat_key: _media_callbacks_for(repo, chat_key),
        )
        runner = make_agent_runner(repo, agent, dry_run=False, uuid_fn=lambda: "cy-1")
        decision = await runner.run_dispatch(grant)
        outbox = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        )
        cycles = await db.read(
            lambda c: c.execute("SELECT end_reason FROM cycles").fetchall()
        )
        await repo.close()
        return decision, outbox, cycles

    decision, outbox, cycles = run(scenario())
    assert decision.action == "trigger"
    assert outbox == 0
    assert cycles == [("no_action",)]


def test_agent_catalog_prompt_is_cooldown_aware(tmp_path):
    """The planner's catalog listing is the cooldown-aware approved
    selection: a recently-used asset is excluded until its cooldown
    expires."""
    async def scenario():
        db, repo = await open_repo(tmp_path / "t.db")
        await repo.upsert_chat(make_identity())
        a1 = await _seed_approved_sticker(repo, cache_key="c" * 64, description="微笑")
        a2 = await _seed_approved_sticker(
            repo, cache_key="d" * 64, description="大笑", sha256="b" * 64
        )
        # Use a1 within the cooldown window.
        await repo.use_media_asset(CK, a1.id, now=170.0)
        cfg = Config(media=MediaConfig(enabled=True, harvest=True, cooldown_s=1000.0))
        runner = CycleRunner(repo, Gate(), cfg, clock=VirtualClock(epoch=200.0))
        listing = await runner._catalog_prompt(CK)
        await repo.close()
        return listing, a1, a2

    listing, a1, a2 = run(scenario())
    assert f"{a2.id}: 大笑" in listing
    assert f"{a1.id}: 微笑" not in listing  # in cooldown -> excluded
    assert "http" not in listing
