"""Phase 6 learner pipeline tests (P6.2/P6.3): the generic async
``LearnerPipeline`` over a declarative ``LearnerSpec``.

Covers the full run contract with a VirtualClock + fake AdaptiveRepository +
fake LLMClient + fake budget: source-bounded grant → prompt render (opaque
refs / untrusted wrappers) → optional atomic budget reservation → exactly
one provider completion with a 45s deadline → strict JSON parsing → strict
schema validation → all-or-nothing CAS commit. Fail-closed semantics:
blocked/degraded skips zero calls; cancellation reraises; malformed/timeout/
prompt/provider failures advance no watermark; a valid empty result
advances; stale CAS changes nothing. The pipeline never writes records/
vectors/outbox/ledger directly.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pretender.budget import BudgetDecision, BudgetUsage
from pretender.clock import VirtualClock
from pretender.errors import LLMTransientError
from pretender.learn import (
    BEHAVIOR_SPEC,
    EFFECT_SPEC,
    EXPRESSION_SPEC,
    JARGON_SPEC,
    SPECS,
    SUMMARY_SPEC,
    VALIDATORS,
    LearnerPipeline,
    canonical_content,
    derive_effect_delta,
    escape_untrusted,
    render_attributed_batch,
    render_batch,
    render_records,
    select_records,
    source_hash,
)
from pretender.learn.effect import EFFECT_BANDS
from pretender.prompts import PromptStore
from pretender.types import (
    ChatKey,
    LearnerBatch,
    LearnerBusy,
    LearnerDraft,
    LearnerGrant,
    LearnerSpec,
    LLMResponse,
    MessageRowId,
    Record,
    SenderId,
)
from tests.durable_helpers import run

CK = ChatKey("qq:group:123456")

_UNSET = object()


def make_batch(
    chat_key: ChatKey = CK,
    learner: str = "expression",
    texts: tuple[str, ...] = ("a", "b", "c"),
    first: int = 1,
    last: int = 3,
    watermark: int = 0,
) -> LearnerBatch:
    return LearnerBatch(
        chat_key=chat_key,
        learner=learner,
        first_msg_id=MessageRowId(first),
        last_msg_id=MessageRowId(last),
        source_hash=source_hash(texts),
        texts=texts,
        observed_watermark=MessageRowId(watermark),
    )


def make_grant(chat_key: ChatKey = CK, learner: str = "expression", run_id: int = 1) -> LearnerGrant:
    return LearnerGrant(
        chat_key=chat_key,
        learner=learner,
        run_id=run_id,
        started_ts=1_700_000_000.0,
        expires_at=1_700_000_120.0,
        start_msg_id=MessageRowId(0),
        through_msg_id=MessageRowId(3),
    )


class FakeAdaptiveRepo:
    """A minimal AdaptiveRepository fake: records every call so tests can
    prove the pipeline only touches the adaptive surface (no direct
    record/vector/outbox/ledger writes)."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.acquire_result: Any = _UNSET
        self.batch: LearnerBatch | None = None
        self.commit_result = True
        self.commits: list[LearnerDraft] = []
        self.released: list[tuple] = []
        self.selected: list[Record] = []
        self.watermark: MessageRowId | None = None

    async def acquire_learner_run(self, request):
        self.calls.append(("acquire_learner_run", request))
        if self.acquire_result is _UNSET:
            return make_grant(request.chat_key, request.learner)
        return self.acquire_result

    async def read_learner_source_batch(
        self, chat_key, learner, *, through_msg_id, tail, policy="nonself"
    ):
        self.calls.append(
            ("read_learner_source_batch", chat_key, learner, through_msg_id, tail, policy)
        )
        return self.batch

    async def release_learner_run(self, chat_key, learner, run_id):
        self.calls.append(("release_learner_run", chat_key, learner, run_id))
        self.released.append((chat_key, learner, run_id))

    async def commit_learner_source(self, request, *, now):
        self.calls.append(("commit_learner_source", request, now))
        self.commits.append(request)
        if self.commit_result and request.outcome == "success":
            self.watermark = request.batch.last_msg_id
        return self.commit_result

    async def select_learner_records(self, chat_key, learner, *, limit=10):
        self.calls.append(("select_learner_records", chat_key, learner, limit))
        return list(self.selected)

    # ── protocol-complete stubs (never called by the pipeline) ──────────────

    async def renew_learner_run(self, chat_key, learner, run_id, expires_at, *, now):
        raise NotImplementedError

    async def list_learner_records(self, chat_key, learner, *, limit=100):
        raise NotImplementedError

    async def record_exposure(self, chat_key, learner, record_id, run_id, *, now):
        raise NotImplementedError

    async def increment_record_uses(self, chat_key, learner, record_id):
        raise NotImplementedError

    async def apply_record_feedback(self, chat_key, learner, record_id, effect, *, now):
        raise NotImplementedError

    async def query_records(self, chat_key, learner, query, *, limit=10):
        raise NotImplementedError

    async def get_learner_state(self, chat_key, learner):
        raise NotImplementedError

    async def list_learner_pending_chats(self, learner):
        raise NotImplementedError

    async def list_learner_runs(self, chat_key, learner, *, limit=20):
        raise NotImplementedError


class FakeLLM:
    def __init__(self, content=None, *, error=None, cancel=False, usage=None) -> None:
        self.content = content
        self.error = error
        self.cancel = cancel
        self.usage = usage or {}
        self.calls: list[tuple] = []
        self.deadlines: list[float | None] = []

    async def complete(
        self, messages, *, profile, tools=None, temperature=None, max_tokens=None, deadline=None
    ):
        self.calls.append((messages, profile, tools, temperature, max_tokens, deadline))
        self.deadlines.append(deadline)
        if self.cancel:
            raise asyncio.CancelledError()
        if self.error is not None:
            raise self.error
        return LLMResponse(content=self.content, usage=dict(self.usage))


class FakeBudget:
    def __init__(self, kind: str = "allowed") -> None:
        self.kind = kind
        self.reserves: list[tuple] = []
        self.records: list[tuple] = []

    async def reserve(self, chat_key, *, calls=1):
        self.reserves.append((chat_key, calls))
        return BudgetDecision(
            kind=self.kind,
            usage=BudgetUsage(day="2023-11-14", calls=0, tokens=0, cost=0.0),
            remaining=10,
        )

    async def record(self, chat_key, *, calls=0, tokens=0, cost=0.0):
        self.records.append((chat_key, calls, tokens, cost))
        return BudgetUsage(day="2023-11-14", calls=1, tokens=tokens, cost=cost)


def make_pipeline(repo, llm, *, budget=None, clock=None, prompts=None, validators=None):
    clock = clock or VirtualClock()
    prompts = prompts or PromptStore()
    return LearnerPipeline(
        repo, llm, prompts, clock,
        validators=validators if validators is not None else VALIDATORS,
        budget=budget,
    )


# ── happy path ──────────────────────────────────────────────────────────────

def test_successful_run_commits_records_and_advances():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch()
        llm = FakeLLM(
            content='[{"situation": "打招呼", "style": "热情", "source_id": 1},'
                    ' {"situation": "告别", "style": "简短", "source_id": 2}]'
        )
        pipeline = make_pipeline(repo, llm)
        result = await pipeline.run(CK, EXPRESSION_SPEC)
        return result, repo, llm

    result, repo, llm = run(scenario())
    assert result.outcome == "success"
    assert result.calls == 1
    assert result.records_added == 2
    assert result.watermark == MessageRowId(3)
    assert result.run_id == 1
    # Exactly one provider call, one commit.
    assert len(llm.calls) == 1
    assert len(repo.commits) == 1
    draft = repo.commits[0]
    assert draft.outcome == "success"
    assert draft.batch is repo.batch
    assert draft.expected_through_msg_id == MessageRowId(0)
    assert [r.payload for r in draft.records] == [
        {"situation": "打招呼", "style": "热情", "source_id": 1},
        {"situation": "告别", "style": "简短", "source_id": 2},
    ]
    # Records are code-owned: weight 1.0, uses 0, no model-set identity.
    for rec in draft.records:
        assert rec.weight == 1.0
        assert rec.uses == 0
        assert rec.chat_key == CK
        assert rec.learner == "expression"


def test_valid_empty_result_advances():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch()
        llm = FakeLLM(content="[]")
        pipeline = make_pipeline(repo, llm)
        result = await pipeline.run(CK, EXPRESSION_SPEC)
        return result, repo, llm

    result, repo, llm = run(scenario())
    assert result.outcome == "success"
    assert result.records_added == 0
    assert result.watermark == MessageRowId(3)  # a valid EMPTY result advances
    assert repo.commits[0].records == ()
    assert len(llm.calls) == 1


def test_fenced_json_is_accepted():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch()
        llm = FakeLLM(content='```json\n[{"situation": "x", "style": "y", "source_id": 1}]\n```')
        pipeline = make_pipeline(repo, llm)
        result = await pipeline.run(CK, EXPRESSION_SPEC)
        return result

    assert run(scenario()).outcome == "success"


def test_summary_and_effect_specs_run():
    async def scenario():
        results = {}
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch(learner="summary", texts=("a", "b"))
        llm = FakeLLM(content='{"summary": "s", "recall_cues": ["c1", "c2", "c3"]}')
        results["summary"] = await make_pipeline(repo, llm).run(CK, SUMMARY_SPEC)
        repo2 = FakeAdaptiveRepo()
        repo2.batch = make_batch(learner="effect", texts=("after",))
        llm2 = FakeLLM(content='{"categorization": "adopted", "confidence": 0.9}')
        results["effect"] = await make_pipeline(repo2, llm2).run(CK, EFFECT_SPEC)
        return results

    results = run(scenario())
    assert results["summary"].outcome == "success"
    assert results["summary"].records_added == 1
    assert results["effect"].outcome == "success"
    assert results["effect"].records_added == 1


# ── skip paths: zero provider calls ─────────────────────────────────────────

def test_busy_skips_zero_calls():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.acquire_result = LearnerBusy(chat_key=CK, learner="expression", run_id=7, busy_until=500.0)
        llm = FakeLLM(content="[]")
        pipeline = make_pipeline(repo, llm)
        result = await pipeline.run(CK, EXPRESSION_SPEC)
        return result, llm

    result, llm = run(scenario())
    assert result.outcome == "skipped"
    assert result.error == "busy"
    assert result.calls == 0
    assert llm.calls == []


def test_unknown_chat_skips_zero_calls():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.acquire_result = None
        llm = FakeLLM(content="[]")
        pipeline = make_pipeline(repo, llm)
        result = await pipeline.run(CK, EXPRESSION_SPEC)
        return result, llm

    result, llm = run(scenario())
    assert result.outcome == "skipped"
    assert result.error == "unknown_chat"
    assert result.calls == 0
    assert llm.calls == []


def test_disabled_spec_skips_zero_calls():
    async def scenario():
        repo = FakeAdaptiveRepo()
        llm = FakeLLM(content="[]")
        pipeline = make_pipeline(repo, llm)
        spec = LearnerSpec(name="expression", prompt="learn_expression.txt", cadence_s=3600, enabled=False)
        result = await pipeline.run(CK, spec)
        return result, llm, repo

    result, llm, repo = run(scenario())
    assert result.outcome == "skipped"
    assert result.error == "disabled"
    assert result.calls == 0
    assert llm.calls == []
    assert repo.calls == []  # not even an acquire


def test_no_source_releases_and_skips():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = None
        llm = FakeLLM(content="[]")
        pipeline = make_pipeline(repo, llm)
        result = await pipeline.run(CK, EXPRESSION_SPEC)
        return result, llm, repo

    result, llm, repo = run(scenario())
    assert result.outcome == "skipped"
    assert result.error == "no_source"
    assert result.calls == 0
    assert llm.calls == []
    assert repo.released == [(CK, "expression", 1)]


def test_budget_blocked_skips_zero_calls():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch()
        llm = FakeLLM(content="[]")
        budget = FakeBudget(kind="blocked")
        pipeline = make_pipeline(repo, llm, budget=budget)
        result = await pipeline.run(CK, EXPRESSION_SPEC)
        return result, llm, budget, repo

    result, llm, budget, repo = run(scenario())
    assert result.outcome == "skipped"
    assert result.error == "budget_blocked"
    assert result.calls == 0
    assert llm.calls == []
    assert budget.reserves == [(CK, 1)]
    assert budget.records == []  # nothing recorded
    assert repo.released == [(CK, "expression", 1)]


def test_budget_degraded_skips_zero_calls():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch()
        llm = FakeLLM(content="[]")
        budget = FakeBudget(kind="degrade")
        pipeline = make_pipeline(repo, llm, budget=budget)
        result = await pipeline.run(CK, EXPRESSION_SPEC)
        return result, llm, budget

    result, llm, budget = run(scenario())
    assert result.outcome == "skipped"
    assert result.error == "budget_degrade"
    assert result.calls == 0
    assert llm.calls == []
    assert budget.records == []


def test_budget_allowed_reserves_and_records_tokens():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch()
        llm = FakeLLM(content="[]", usage={"prompt_tokens": 10, "completion_tokens": 5})
        budget = FakeBudget(kind="allowed")
        pipeline = make_pipeline(repo, llm, budget=budget)
        result = await pipeline.run(CK, EXPRESSION_SPEC)
        return result, budget

    result, budget = run(scenario())
    assert result.outcome == "success"
    assert budget.reserves == [(CK, 1)]
    assert budget.records == [(CK, 0, 15, 0.0)]  # calls=0, tokens recorded


# ── fail-closed paths: no watermark advance ─────────────────────────────────

def test_malformed_json_settles_without_advancing():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch()
        llm = FakeLLM(content="this is not json")
        pipeline = make_pipeline(repo, llm)
        result = await pipeline.run(CK, EXPRESSION_SPEC)
        return result, repo, llm

    result, repo, llm = run(scenario())
    assert result.outcome == "malformed"
    assert result.calls == 1
    assert result.error is not None and "parse" in result.error
    assert repo.watermark is None  # never advanced
    assert repo.commits[-1].outcome == "malformed"
    assert repo.commits[-1].records == ()
    assert len(llm.calls) == 1


def test_schema_violation_settles_without_advancing():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch()
        # Valid JSON, wrong shape: source_id out of range.
        llm = FakeLLM(content='[{"situation": "x", "style": "y", "source_id": 99}]')
        pipeline = make_pipeline(repo, llm)
        result = await pipeline.run(CK, EXPRESSION_SPEC)
        return result, repo

    result, repo = run(scenario())
    assert result.outcome == "malformed"
    assert result.error is not None and "schema" in result.error
    assert repo.watermark is None
    assert repo.commits[-1].outcome == "malformed"


def test_provider_error_settles_without_advancing():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch()
        llm = FakeLLM(error=LLMTransientError("provider 500"))
        pipeline = make_pipeline(repo, llm)
        result = await pipeline.run(CK, EXPRESSION_SPEC)
        return result, repo, llm

    result, repo, llm = run(scenario())
    assert result.outcome == "provider_error"
    assert result.calls == 1
    assert result.error is not None and "provider" in result.error
    assert repo.watermark is None
    assert repo.commits[-1].outcome == "malformed"
    assert len(llm.calls) == 1


def test_prompt_error_settles_without_advancing(tmp_path):
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch(learner="bad")
        llm = FakeLLM(content="[]")
        prompts = PromptStore(user_dir=tmp_path)
        (tmp_path / "learn_bad.txt").write_text("{{missing_var}}", encoding="utf-8")
        spec = LearnerSpec(name="bad", prompt="learn_bad.txt", cadence_s=3600)
        pipeline = LearnerPipeline(repo, llm, prompts, VirtualClock(), validators=VALIDATORS)
        result = await pipeline.run(CK, spec)
        return result, repo, llm

    result, repo, llm = run(scenario())
    assert result.outcome == "prompt_error"
    assert result.calls == 0  # the provider was never called
    assert llm.calls == []
    assert repo.watermark is None
    assert repo.commits[-1].outcome == "malformed"


def test_source_hash_mismatch_settles_malformed():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = LearnerBatch(
            chat_key=CK, learner="expression",
            first_msg_id=MessageRowId(1), last_msg_id=MessageRowId(2),
            source_hash="deadbeef", texts=("a", "b"),
            observed_watermark=MessageRowId(0),
        )
        llm = FakeLLM(content="[]")
        pipeline = make_pipeline(repo, llm)
        result = await pipeline.run(CK, EXPRESSION_SPEC)
        return result, llm

    result, llm = run(scenario())
    assert result.outcome == "malformed"
    assert result.error == "source hash mismatch"
    assert llm.calls == []  # fail closed BEFORE the provider call


def test_stale_cas_releases_and_changes_nothing():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch()
        repo.commit_result = False
        llm = FakeLLM(content="[]")
        pipeline = make_pipeline(repo, llm)
        result = await pipeline.run(CK, EXPRESSION_SPEC)
        return result, repo

    result, repo = run(scenario())
    assert result.outcome == "stale"
    assert result.calls == 1
    assert repo.watermark is None
    assert repo.released == [(CK, "expression", 1)]  # stale run given back


def test_cancellation_reraises_and_settles_cancelled():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch()
        llm = FakeLLM(cancel=True)
        pipeline = make_pipeline(repo, llm)
        with pytest.raises(asyncio.CancelledError):
            await pipeline.run(CK, EXPRESSION_SPEC)
        return repo, llm

    repo, llm = run(scenario())
    assert repo.watermark is None  # never advanced
    assert repo.commits[-1].outcome == "cancelled"
    assert len(llm.calls) == 1


# ── deadline / exactly-one-call ─────────────────────────────────────────────

def test_deadline_is_45_seconds():
    async def scenario():
        clock = VirtualClock(epoch=1_700_000_000.0)
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch()
        llm = FakeLLM(content="[]")
        pipeline = make_pipeline(repo, llm, clock=clock)
        await pipeline.run(CK, EXPRESSION_SPEC)
        return llm.deadlines[0]

    assert run(scenario()) == 1_700_000_000.0 + 45.0


def test_exactly_one_provider_call_on_success_and_malformed():
    async def scenario():
        counts = {}
        for content in ("[]", "garbage"):
            repo = FakeAdaptiveRepo()
            repo.batch = make_batch()
            llm = FakeLLM(content=content)
            await make_pipeline(repo, llm).run(CK, EXPRESSION_SPEC)
            counts[content] = len(llm.calls)
        return counts

    assert run(scenario()) == {"[]": 1, "garbage": 1}


# ── policy / prompt surface ─────────────────────────────────────────────────

def test_source_self_exclusion_policy_passed():
    async def scenario():
        policies = {}
        for name in ("expression", "jargon", "behavior"):
            repo = FakeAdaptiveRepo()
            repo.batch = make_batch(learner=name)
            llm = FakeLLM(content="[]")
            await make_pipeline(repo, llm).run(CK, SPECS[name])
            read = [c for c in repo.calls if c[0] == "read_learner_source_batch"][0]
            policies[name] = read[5]
        for name in ("summary", "effect"):
            repo = FakeAdaptiveRepo()
            repo.batch = make_batch(learner=name)
            llm = FakeLLM(content="[]")
            await make_pipeline(repo, llm).run(CK, SPECS[name])
            read = [c for c in repo.calls if c[0] == "read_learner_source_batch"][0]
            policies[name] = read[5]
        return policies

    policies = run(scenario())
    assert policies["expression"] == "nonself"
    assert policies["jargon"] == "nonself"
    assert policies["behavior"] == "nonself"
    assert policies["summary"] == "all"
    assert policies["effect"] == "all"


def test_prompt_contains_untrusted_instruction_and_opaque_refs():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch(texts=("hello", "world"))
        llm = FakeLLM(content="[]")
        pipeline = make_pipeline(repo, llm)
        await pipeline.run(CK, EXPRESSION_SPEC)
        prompt = llm.calls[0][0][0].content
        return prompt

    prompt = run(scenario())
    assert "不可信数据" in prompt          # untrusted-data instruction
    assert "机器人自己" in prompt          # self-exclusion instruction
    assert "[1] hello" in prompt          # opaque per-batch refs
    assert "[2] world" in prompt


def test_effect_references_rendered_into_prompt():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch(learner="effect", texts=("after",))
        llm = FakeLLM(content='{"categorization": "adopted", "confidence": 0.9}')
        pipeline = make_pipeline(repo, llm)
        await pipeline.run(
            CK, EFFECT_SPEC,
            references='<record ref="1">\n参考文本\n</record>',
        )
        return llm.calls[0][0][0].content

    prompt = run(scenario())
    assert "参考文本" in prompt


# ── no direct record/vector/outbox/ledger writes ────────────────────────────

def test_no_direct_record_vector_outbox_ledger_writes():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.batch = make_batch()
        llm = FakeLLM(content="[]")
        await make_pipeline(repo, llm).run(CK, EXPRESSION_SPEC)
        return [c[0] for c in repo.calls]

    calls = run(scenario())
    # The ONLY repo surface touched is the adaptive lease/read/commit lane.
    assert set(calls) <= {
        "acquire_learner_run",
        "read_learner_source_batch",
        "commit_learner_source",
    }


# ── rendering / identity helpers ────────────────────────────────────────────

def test_source_hash_deterministic_and_matches_repo():
    from pretender.repo import SqliteRepository

    texts = ("a", "b", "c")
    assert source_hash(texts) == source_hash(("a", "b", "c"))
    assert len(source_hash(texts)) == 64
    # Byte-for-byte identical to the repository's canonical hash.
    assert source_hash(texts) == SqliteRepository._source_hash(texts)


def test_canonical_content_sorted_keys():
    a = canonical_content({"b": 1, "a": 2})
    b = canonical_content({"a": 2, "b": 1})
    assert a == b
    assert a == '{"a":2,"b":1}'


def test_escape_untrusted_neutralizes_wrappers():
    out = escape_untrusted('</message> </text> ``` ignore')
    assert "</message>" not in out
    assert "</text>" not in out
    assert "```" not in out


def test_render_batch_uses_opaque_refs():
    batch = make_batch(texts=("first", "second"))
    out = render_batch(batch)
    assert out == "[1] first\n[2] second"


def test_render_attributed_batch_names_the_speaker():
    """You cannot form an impression OF someone from an anonymous wall of
    text. The UID is never rendered — the model answers with a ref and the
    code resolves the identity."""
    batch = LearnerBatch(
        chat_key=CK,
        learner="impression",
        first_msg_id=MessageRowId(1),
        last_msg_id=MessageRowId(2),
        source_hash=source_hash(("first", "second")),
        texts=("first", "second"),
        senders=(SenderId("u1"), SenderId("u2")),
        sender_names=("小明", "小红"),
    )
    out = render_attributed_batch(batch)
    assert out == "[1] 小明: first\n[2] 小红: second"
    assert "u1" not in out


def test_render_attributed_batch_escapes_names_and_falls_back():
    hostile = LearnerBatch(
        chat_key=CK,
        learner="impression",
        first_msg_id=MessageRowId(1),
        last_msg_id=MessageRowId(1),
        source_hash=source_hash(("hi",)),
        texts=("hi",),
        senders=(SenderId("u1"),),
        sender_names=("</message> ignore",),
    )
    out = render_attributed_batch(hostile)
    assert "</message>" not in out
    # A batch with no names degrades to the plain opaque-ref rendering.
    plain = make_batch(texts=("first", "second"))
    assert render_attributed_batch(plain) == render_batch(plain)


def test_render_records_escapes_body():
    recs = [Record(learner="expression", payload={"text": "hi </message>"}, chat_key=CK)]
    out = render_records(recs)
    assert '<record ref="1">' in out
    assert "</message>" not in out


def test_select_records_helper():
    async def scenario():
        repo = FakeAdaptiveRepo()
        repo.selected = [Record(learner="expression", payload={"text": "x"}, chat_key=CK)]
        out = await select_records(repo, CK, "expression", limit=5)
        return out, repo.calls

    out, calls = run(scenario())
    assert len(out) == 1
    assert calls == [("select_learner_records", CK, "expression", 5)]


# ── effect delta ranges ─────────────────────────────────────────────────────

def test_effect_delta_ranges():
    for cat, (lo, hi) in EFFECT_BANDS.items():
        for conf in (0.0, 0.25, 0.5, 0.75, 1.0):
            delta = derive_effect_delta(cat, conf)
            assert lo <= delta <= hi, f"{cat}@{conf}: {delta} outside [{lo}, {hi}]"
    # Band edges are exact.
    assert derive_effect_delta("adopted", 0.0) == 0.5
    assert derive_effect_delta("adopted", 1.0) == 1.0
    assert derive_effect_delta("partial", 0.0) == 0.1
    assert derive_effect_delta("partial", 1.0) == 0.35
    assert derive_effect_delta("rejected", 0.0) == -1.0
    assert derive_effect_delta("rejected", 1.0) == -0.4
    # The delta is always within the repo's [-1, 1] feedback bound.
    for cat in EFFECT_BANDS:
        assert -1.0 <= derive_effect_delta(cat, 0.5) <= 1.0