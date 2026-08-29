"""Phase 6 learner definitions tests: the five frozen specs, their strict
validators, the strict JSON parser, and the prompt files.

Covers the frozen policy: nonself SQL policy for expression/jargon/peer
behavior, hard limits, opaque refs must exist, the model can never set
weights/uses/delta, summary includes the full conversation with exactly 3–5
cues, and effect returns categorization/confidence only (the code derives
the bounded delta).
"""

from __future__ import annotations

import json

import pytest

from pretender.learn import (
    BEHAVIOR_MAX,
    BEHAVIOR_SPEC,
    EFFECT_SPEC,
    EXPRESSION_FIELD_MAX,
    EXPRESSION_MAX,
    EXPRESSION_SPEC,
    JARGON_MAX,
    JARGON_SPEC,
    SPECS,
    SUMMARY_CUES_MAX,
    SUMMARY_CUES_MIN,
    SUMMARY_SPEC,
    VALIDATORS,
    LearnerParseError,
    LearnerValidationError,
    parse_json_response,
    source_hash,
)
from pretender.learn.effect import EFFECT_CATEGORIZATIONS
from pretender.prompts import PACKAGE_PROMPT_DIR
from pretender.types import ChatKey, LearnerBatch, MessageRowId

CK = ChatKey("qq:group:123456")


def make_batch(
    texts: tuple[str, ...] = ("a", "b", "c"),
    learner: str = "expression",
) -> LearnerBatch:
    return LearnerBatch(
        chat_key=CK,
        learner=learner,
        first_msg_id=MessageRowId(1),
        last_msg_id=MessageRowId(len(texts)),
        source_hash=source_hash(texts),
        texts=texts,
        observed_watermark=MessageRowId(0),
    )


def validate(name: str, parsed, batch=None):
    return VALIDATORS[name](parsed, batch or make_batch(learner=name), now=100.0)


# ── the five specs ──────────────────────────────────────────────────────────

def test_five_specs_defined_with_frozen_policy():
    assert set(SPECS) == {"expression", "behavior", "jargon", "summary", "effect"}
    # Nonself SQL policy for expression/jargon/peer behavior.
    assert EXPRESSION_SPEC.policy == "nonself"
    assert JARGON_SPEC.policy == "nonself"
    assert BEHAVIOR_SPEC.policy == "nonself"
    # Summary/effect read the FULL conversation (including the bot's own).
    assert SUMMARY_SPEC.policy == "all"
    assert EFFECT_SPEC.policy == "all"
    # Every spec is pure data: only the declared fields, no repo/llm/clock.
    for spec in SPECS.values():
        assert set(spec.__dataclass_fields__) == {
            "name", "prompt", "cadence_s", "policy", "batch_size", "enabled",
        }
        assert spec.enabled is True
        assert spec.cadence_s > 0
        assert spec.batch_size > 0


def test_specs_prompt_files_exist():
    for spec in SPECS.values():
        path = PACKAGE_PROMPT_DIR / spec.prompt
        assert path.is_file(), f"missing prompt file {spec.prompt}"
        text = path.read_text(encoding="utf-8")
        # Self-exclusion + untrusted-data instruction in every prompt.
        assert "不可信数据" in text, f"{spec.prompt} lacks the untrusted-data instruction"
        assert "机器人自己" in text, f"{spec.prompt} lacks the self-exclusion instruction"
        # The model is told never to set weight/uses/delta.
        assert "weight" in text and "delta" in text


# ── expression validator ────────────────────────────────────────────────────

def test_expression_validator_accepts_valid():
    records = validate(
        "expression",
        [
            {"situation": "打招呼", "style": "热情", "source_id": 1},
            {"situation": "告别", "style": "简短", "source_id": 3},
        ],
    )
    assert len(records) == 2
    assert records[0].payload == {"situation": "打招呼", "style": "热情", "source_id": 1}
    assert records[1].payload["source_id"] == 3
    for rec in records:
        assert rec.weight == 1.0 and rec.uses == 0
        assert rec.chat_key == CK and rec.learner == "expression"


def test_expression_validator_rejects_bad_shapes():
    batch = make_batch()
    bad_inputs = [
        "not-a-list",
        [{"situation": "x", "style": "y", "source_id": 1}] * (EXPRESSION_MAX + 1),
        [{"situation": "x", "style": "y"}],  # missing source_id
        [{"situation": "x", "style": "y", "source_id": 0}],  # ref out of range
        [{"situation": "x", "style": "y", "source_id": 4}],  # ref beyond batch
        [{"situation": "x", "style": "y", "source_id": "1"}],  # non-int ref
        [{"situation": "x" * (EXPRESSION_FIELD_MAX + 1), "style": "y", "source_id": 1}],
        [{"situation": "x", "style": "y" * (EXPRESSION_FIELD_MAX + 1), "source_id": 1}],
        [{"situation": "x", "style": "y", "source_id": 1, "weight": 5.0}],
        [{"situation": "x", "style": "y", "source_id": 1, "delta": 0.5}],
        [{"situation": "x", "style": "y", "source_id": 1, "uses": 3}],
        [{"situation": "x", "style": "y", "source_id": 1, "content_hash": "abc"}],
    ]
    for bad in bad_inputs:
        with pytest.raises(LearnerValidationError):
            validate("expression", bad, batch)


def test_expression_valid_empty_list_accepted():
    assert validate("expression", []) == []


# ── behavior validator ──────────────────────────────────────────────────────

def test_behavior_validator_accepts_valid():
    records = validate(
        "behavior",
        [
            {
                "segment_id": "s1", "actor_type": "other_user",
                "learning_type": "observation", "action": "发图", "outcome": "被夸",
                "source_ids": [1, 2],
            },
            {
                "segment_id": "s2", "actor_type": "group_collective",
                "learning_type": "self_reflection", "action": "冷场", "outcome": "沉默",
                "source_ids": [3],
            },
        ],
    )
    assert len(records) == 2
    assert records[0].payload["actor_type"] == "other_user"
    assert records[1].payload["source_ids"] == [3]


def test_behavior_validator_rejects_bad_shapes():
    batch = make_batch()
    base = {
        "segment_id": "s1", "actor_type": "other_user",
        "learning_type": "observation", "action": "a", "outcome": "o",
        "source_ids": [1],
    }
    bad_inputs = [
        "not-a-list",
        [base] * (BEHAVIOR_MAX + 1),
        [{**base, "actor_type": "bot"}],
        [{**base, "learning_type": "guessing"}],
        [{**base, "source_ids": []}],
        [{**base, "source_ids": [9]}],  # ref beyond batch
        [{**base, "source_ids": "1"}],
        [{**base, "segment_id": ""}],
        [{**base, "weight": 2.0}],
        [{**base, "score_delta": 0.3}],
    ]
    for bad in bad_inputs:
        with pytest.raises(LearnerValidationError):
            validate("behavior", bad, batch)


def test_behavior_valid_empty_list_accepted():
    assert validate("behavior", []) == []


# ── jargon validator ────────────────────────────────────────────────────────

def test_jargon_validator_accepts_valid():
    records = validate(
        "jargon",
        [
            {"term": "yyds", "meaning": "永远的神", "context": "夸赞", "source_ids": [1]},
            {"term": "破防", "meaning": "情绪崩溃", "context": "吐槽", "source_ids": [2, 3]},
        ],
    )
    assert len(records) == 2
    assert records[0].payload["term"] == "yyds"
    assert records[1].payload["source_ids"] == [2, 3]


def test_jargon_validator_rejects_bad_shapes():
    batch = make_batch()
    base = {"term": "t", "meaning": "m", "context": "c", "source_ids": [1]}
    bad_inputs = [
        "not-a-list",
        [base] * (JARGON_MAX + 1),
        [{**base, "term": ""}],
        [{**base, "source_ids": []}],
        [{**base, "source_ids": [0]}],
        [{**base, "source_ids": [True]}],
        [{**base, "uses": 1}],
        [{**base, "delta": -0.2}],
    ]
    for bad in bad_inputs:
        with pytest.raises(LearnerValidationError):
            validate("jargon", bad, batch)


def test_jargon_valid_empty_list_accepted():
    assert validate("jargon", []) == []


# ── summary validator ───────────────────────────────────────────────────────

def test_summary_validator_accepts_valid():
    records = validate(
        "summary",
        {"summary": "大家讨论了周末聚餐", "recall_cues": ["谁提议了聚餐", "定了什么时间", "去了哪家店"]},
    )
    assert len(records) == 1
    assert records[0].payload["summary"] == "大家讨论了周末聚餐"
    assert len(records[0].payload["recall_cues"]) == 3


def test_summary_validator_rejects_bad_shapes():
    batch = make_batch(learner="summary")
    bad_inputs = [
        "not-an-object",
        [],
        {"summary": "s", "recall_cues": ["a", "b"]},  # too few cues
        {"summary": "s", "recall_cues": ["a", "b", "c", "d", "e", "f"]},  # too many
        {"summary": "s", "recall_cues": ["a", "", "c"]},  # empty cue
        {"summary": "s", "recall_cues": ["a", 2, "c"]},  # non-string cue
        {"summary": "", "recall_cues": ["a", "b", "c"]},
        {"summary": "s", "recall_cues": ["a", "b", "c"], "weight": 0.5},
        {"summary": "s", "recall_cues": ["a", "b", "c"], "delta": 1.0},
    ]
    for bad in bad_inputs:
        with pytest.raises(LearnerValidationError):
            validate("summary", bad, batch)


def test_summary_cue_bounds_are_exact():
    batch = make_batch(learner="summary")
    for n in range(SUMMARY_CUES_MIN, SUMMARY_CUES_MAX + 1):
        records = validate(
            "summary",
            {"summary": "s", "recall_cues": [f"c{i}" for i in range(n)]},
            batch,
        )
        assert len(records[0].payload["recall_cues"]) == n


# ── effect validator ────────────────────────────────────────────────────────

def test_effect_validator_accepts_valid():
    records = validate(
        "effect", {"categorization": "adopted", "confidence": 0.8},
        make_batch(learner="effect"),
    )
    assert len(records) == 1
    assert records[0].payload == {"categorization": "adopted", "confidence": 0.8}
    # The payload carries NO delta — the code derives it.
    assert "delta" not in records[0].payload
    assert "score_delta" not in records[0].payload


def test_effect_validator_rejects_bad_shapes():
    batch = make_batch(learner="effect")
    bad_inputs = [
        "not-an-object",
        [],
        {"categorization": "maybe", "confidence": 0.5},
        {"categorization": "adopted", "confidence": 1.5},
        {"categorization": "adopted", "confidence": -0.1},
        {"categorization": "adopted", "confidence": "high"},
        {"categorization": "adopted", "confidence": True},
        {"categorization": "adopted", "confidence": 0.5, "delta": 0.7},
        {"categorization": "adopted", "confidence": 0.5, "weight": 3.0},
    ]
    for bad in bad_inputs:
        with pytest.raises(LearnerValidationError):
            validate("effect", bad, batch)


def test_effect_categorizations_match_bands():
    assert set(EFFECT_CATEGORIZATIONS) == {"adopted", "partial", "rejected"}


# ── cross-cutting: refs must exist, model cannot set identity ───────────────

def test_refs_must_exist_across_list_learners():
    batch = make_batch(texts=("only-one",))
    for name in ("expression", "behavior", "jargon"):
        if name == "expression":
            parsed = {"situation": "x", "style": "y", "source_id": 2}
        else:
            parsed = {
                "segment_id": "s", "actor_type": "other_user",
                "learning_type": "observation", "action": "a", "outcome": "o",
                "source_ids": [2],
            } if name == "behavior" else {
                "term": "t", "meaning": "m", "context": "c", "source_ids": [2],
            }
        with pytest.raises(LearnerValidationError):
            validate(name, [parsed], batch)


def test_model_cannot_set_identity_keys_on_any_spec():
    batch = make_batch()
    cases = {
        "expression": {"situation": "x", "style": "y", "source_id": 1},
        "behavior": {
            "segment_id": "s", "actor_type": "other_user",
            "learning_type": "observation", "action": "a", "outcome": "o",
            "source_ids": [1],
        },
        "jargon": {"term": "t", "meaning": "m", "context": "c", "source_ids": [1]},
        "summary": {"summary": "s", "recall_cues": ["a", "b", "c"]},
        "effect": {"categorization": "adopted", "confidence": 0.5},
    }
    for name, base in cases.items():
        for key in ("weight", "uses", "delta", "score_delta", "content_hash", "id", "created_ts", "retired"):
            poisoned = dict(base)
            poisoned[key] = 1
            with pytest.raises(LearnerValidationError):
                validate(name, poisoned, batch)


# ── strict JSON parsing ─────────────────────────────────────────────────────

def test_parse_json_response_accepts_raw_and_fenced():
    assert parse_json_response('[{"a": 1}]') == [{"a": 1}]
    assert parse_json_response('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert parse_json_response('```\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('```json {"a": 1} ```') == {"a": 1}
    assert parse_json_response('  [1, 2]  ') == [1, 2]


def test_parse_json_response_rejects_bad_shapes():
    bad_inputs = [
        None,
        "",
        "   ",
        "not json",
        "```json\n[1, 2]",  # unbalanced fence
        "```json\n[1, 2]\n```\nextra",  # trailing text after the fence
        "prefix ```json\n[1]\n```",  # text before the fence
        "```\n```",  # empty fence body
        "[1, 2",  # truncated JSON
        "```json\n[1, 2]\n```\n```json\n[3]\n```",  # two fences
    ]
    for bad in bad_inputs:
        with pytest.raises(LearnerParseError):
            parse_json_response(bad)


def test_parse_json_response_roundtrips_spec_outputs():
    for spec in SPECS.values():
        # Every spec's validator output must be re-parseable from its JSON.
        assert parse_json_response(json.dumps([])) == []