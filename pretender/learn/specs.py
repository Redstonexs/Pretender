"""The six declarative learner definitions (Phase 6).

Each spec is a pure-data ``LearnerSpec`` (no repository/LLM/clock access)
plus a strict validator that turns a parsed model response into validated
``Record`` objects. The frozen policy:

- ``expression`` / ``jargon`` / ``behavior`` / ``impression`` are
  ``nonself``: the source
  batch excludes the bot's own messages (enforced in SQL by the repository
  AND stated in the prompt — self-learning is a positive feedback loop).
- ``summary`` is ``all``: it must cover the FULL conversation (including the
  bot's own messages) and produce exactly 3–5 recall cues.
- ``effect`` is ``all``: it reads the references shown to the planner plus
  the chat that followed, and returns ONLY ``categorization``/``confidence``
  — the bounded delta is derived by code (``derive_effect_delta``), never
  set by the model.

Hard limits (frozen): expression situation/style ≤ 20 chars and at most 10
records; behavior/jargon at most 5 records; summary exactly 3–5 cues; effect
confidence in [0, 1]. Every ``source_id``/``source_ids`` ref must exist in
the current batch (an opaque per-batch ref in ``[1, len(batch.texts)]``).
The model can never set ``weight``/``uses``/``delta`` (or any identity/
bookkeeping key) on a record payload — the repository owns identity and the
code owns effect deltas.

A valid EMPTY result (``[]`` for the list learners) is accepted and still
advances the watermark; a malformed shape raises ``LearnerValidationError``
and the run settles malformed without advancing.
"""

from __future__ import annotations

from typing import Any, Callable

from pretender.learn.effect import EFFECT_CATEGORIZATIONS
from pretender.learn.render import FORBIDDEN_PAYLOAD_KEYS
from pretender.types import LearnerBatch, LearnerSpec, Record

__all__ = [
    "EXPRESSION_SPEC",
    "BEHAVIOR_SPEC",
    "JARGON_SPEC",
    "SUMMARY_SPEC",
    "EFFECT_SPEC",
    "IMPRESSION_SPEC",
    "SPECS",
    "VALIDATORS",
    "LearnerValidationError",
    "ACTOR_TYPES",
    "LEARNING_TYPES",
    "EXPRESSION_MAX",
    "EXPRESSION_FIELD_MAX",
    "BEHAVIOR_MAX",
    "JARGON_MAX",
    "SUMMARY_CUES_MIN",
    "SUMMARY_CUES_MAX",
    "IMPRESSION_MAX",
    "IMPRESSION_FIELD_MAX",
]

# ── Hard limits (frozen policy) ─────────────────────────────────────────────

EXPRESSION_MAX = 10          # at most 10 expression records per run
EXPRESSION_FIELD_MAX = 20    # situation/style each ≤ 20 chars
BEHAVIOR_MAX = 5             # at most 5 behavior records per run
JARGON_MAX = 5               # at most 5 jargon records per run
SUMMARY_CUES_MIN = 3         # exactly 3–5 recall cues
SUMMARY_CUES_MAX = 5
IMPRESSION_MAX = 5           # at most 5 people per run
IMPRESSION_FIELD_MAX = 80    # one impression is one sentence, not a dossier

ACTOR_TYPES: tuple[str, ...] = ("other_user", "group_collective")
LEARNING_TYPES: tuple[str, ...] = ("observation", "self_reflection")


class LearnerValidationError(ValueError):
    """A parsed model response violates the spec's strict schema. The run
    settles malformed; the watermark never advances."""


# ── The five declarative specs ──────────────────────────────────────────────
# ``prompt`` is a prompt FILE NAME resolved through the PromptStore (a sixth
# learner is a prompt file + a validator, per PLAN.md §D).

EXPRESSION_SPEC = LearnerSpec(
    name="expression",
    prompt="learn_expression.txt",
    cadence_s=3600,
    policy="nonself",
    batch_size=10,
    enabled=True,
)

BEHAVIOR_SPEC = LearnerSpec(
    name="behavior",
    prompt="learn_behavior.txt",
    cadence_s=3600,
    policy="nonself",
    batch_size=10,
    enabled=True,
)

JARGON_SPEC = LearnerSpec(
    name="jargon",
    prompt="learn_jargon.txt",
    cadence_s=3600,
    policy="nonself",
    batch_size=10,
    enabled=True,
)

SUMMARY_SPEC = LearnerSpec(
    name="summary",
    prompt="learn_summary.txt",
    cadence_s=3600,
    policy="all",
    batch_size=200,
    enabled=True,
)

EFFECT_SPEC = LearnerSpec(
    name="effect",
    prompt="learn_effect.txt",
    cadence_s=3600,
    policy="all",
    batch_size=50,
    enabled=True,
)

IMPRESSION_SPEC = LearnerSpec(
    name="impression",
    prompt="learn_impression.txt",
    cadence_s=3600,
    policy="nonself",
    batch_size=50,
    enabled=True,
)

SPECS: dict[str, LearnerSpec] = {
    "expression": EXPRESSION_SPEC,
    "behavior": BEHAVIOR_SPEC,
    "jargon": JARGON_SPEC,
    "summary": SUMMARY_SPEC,
    "effect": EFFECT_SPEC,
    "impression": IMPRESSION_SPEC,
}


# ── shared validation helpers ───────────────────────────────────────────────


def _require_str(item: dict[str, Any], key: str, where: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LearnerValidationError(f"{where}: {key} must be a non-empty string")
    return value


def _as_ref(value: Any) -> int | None:
    """Coerce an opaque ref to an int, or None when it is not one.

    ``"3"`` is accepted as ``3``. Models routinely quote integers — on a
    small model it is the single most common reason a whole learner run is
    thrown away — and the ref is bounds-checked against the batch either
    way, so accepting the quoted spelling cannot widen what the model is
    able to point at. Every other field stays strictly typed. (MaiBot runs
    every learner response through ``json_repair`` for the same reason; this
    is the narrow version of that.)
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return None


def _require_int(item: dict[str, Any], key: str, where: str) -> int:
    ref = _as_ref(item.get(key))
    if ref is None:
        raise LearnerValidationError(f"{where}: {key} must be an integer")
    return ref


def _check_ref(ref: Any, batch: LearnerBatch, where: str) -> int:
    """An opaque per-batch ref must exist: an integer in [1, len(texts)]."""
    value = _as_ref(ref)
    if value is None:
        raise LearnerValidationError(f"{where}: source ref must be an integer")
    if not (1 <= value <= len(batch.texts)):
        raise LearnerValidationError(
            f"{where}: source ref {value} out of range"
            f" (batch has {len(batch.texts)} messages)"
        )
    return value


def _check_refs(refs: Any, batch: LearnerBatch, where: str) -> list[int]:
    if not isinstance(refs, list) or not refs:
        raise LearnerValidationError(f"{where}: source_ids must be a non-empty list")
    return [_check_ref(ref, batch, where) for ref in refs]


def _forbid(item: dict[str, Any], where: str) -> None:
    """The model can never set identity/bookkeeping keys on a payload."""
    extra = FORBIDDEN_PAYLOAD_KEYS & set(item.keys())
    if extra:
        raise LearnerValidationError(
            f"{where}: model cannot set {sorted(extra)}"
        )


def _make_record(batch: LearnerBatch, payload: dict[str, Any], now: float) -> Record:
    """A validated record: weight/uses are code-owned defaults (1.0 / 0),
    identity fields are left for the repository to compute at commit."""
    return Record(
        learner=batch.learner,
        payload=payload,
        chat_key=batch.chat_key,
        weight=1.0,
        uses=0,
        created_ts=now,
    )


# ── the five strict validators ──────────────────────────────────────────────
# Signature: ``(parsed, batch, *, now) -> list[Record]``. ``parsed`` is the
# JSON value parsed from the model response. A valid EMPTY list is accepted
# (it still advances the watermark); every other violation raises.

Validator = Callable[..., list[Record]]


def validate_expression(parsed: Any, batch: LearnerBatch, *, now: float) -> list[Record]:
    """``[{"situation": ≤20字, "style": ≤20字, "source_id": N}]``, ≤10."""
    if not isinstance(parsed, list):
        raise LearnerValidationError("expression: expected a JSON array of records")
    if len(parsed) > EXPRESSION_MAX:
        raise LearnerValidationError(
            f"expression: at most {EXPRESSION_MAX} records, got {len(parsed)}"
        )
    records: list[Record] = []
    for i, item in enumerate(parsed):
        where = f"expression[{i}]"
        if not isinstance(item, dict):
            raise LearnerValidationError(f"{where}: not an object")
        _forbid(item, where)
        situation = _require_str(item, "situation", where)
        style = _require_str(item, "style", where)
        if len(situation) > EXPRESSION_FIELD_MAX:
            raise LearnerValidationError(
                f"{where}: situation exceeds {EXPRESSION_FIELD_MAX} chars"
            )
        if len(style) > EXPRESSION_FIELD_MAX:
            raise LearnerValidationError(
                f"{where}: style exceeds {EXPRESSION_FIELD_MAX} chars"
            )
        source_id = _check_ref(_require_int(item, "source_id", where), batch, where)
        records.append(
            _make_record(
                batch,
                {"situation": situation, "style": style, "source_id": source_id},
                now,
            )
        )
    return records


def validate_behavior(parsed: Any, batch: LearnerBatch, *, now: float) -> list[Record]:
    """``[{segment_id, actor_type, learning_type, action, outcome,
    source_ids}]``, ≤5. Peer behavior: actor_type is other_user or
    group_collective."""
    if not isinstance(parsed, list):
        raise LearnerValidationError("behavior: expected a JSON array of records")
    if len(parsed) > BEHAVIOR_MAX:
        raise LearnerValidationError(
            f"behavior: at most {BEHAVIOR_MAX} records, got {len(parsed)}"
        )
    records: list[Record] = []
    for i, item in enumerate(parsed):
        where = f"behavior[{i}]"
        if not isinstance(item, dict):
            raise LearnerValidationError(f"{where}: not an object")
        _forbid(item, where)
        segment_id = _require_str(item, "segment_id", where)
        actor_type = _require_str(item, "actor_type", where)
        if actor_type not in ACTOR_TYPES:
            raise LearnerValidationError(
                f"{where}: actor_type must be one of {ACTOR_TYPES}, got {actor_type!r}"
            )
        learning_type = _require_str(item, "learning_type", where)
        if learning_type not in LEARNING_TYPES:
            raise LearnerValidationError(
                f"{where}: learning_type must be one of {LEARNING_TYPES},"
                f" got {learning_type!r}"
            )
        action = _require_str(item, "action", where)
        outcome = _require_str(item, "outcome", where)
        source_ids = _check_refs(item.get("source_ids"), batch, where)
        records.append(
            _make_record(
                batch,
                {
                    "segment_id": segment_id,
                    "actor_type": actor_type,
                    "learning_type": learning_type,
                    "action": action,
                    "outcome": outcome,
                    "source_ids": source_ids,
                },
                now,
            )
        )
    return records


def validate_jargon(parsed: Any, batch: LearnerBatch, *, now: float) -> list[Record]:
    """``[{term, meaning, context, source_ids}]``, ≤5."""
    if not isinstance(parsed, list):
        raise LearnerValidationError("jargon: expected a JSON array of records")
    if len(parsed) > JARGON_MAX:
        raise LearnerValidationError(
            f"jargon: at most {JARGON_MAX} records, got {len(parsed)}"
        )
    records: list[Record] = []
    for i, item in enumerate(parsed):
        where = f"jargon[{i}]"
        if not isinstance(item, dict):
            raise LearnerValidationError(f"{where}: not an object")
        _forbid(item, where)
        term = _require_str(item, "term", where)
        meaning = _require_str(item, "meaning", where)
        context = _require_str(item, "context", where)
        source_ids = _check_refs(item.get("source_ids"), batch, where)
        records.append(
            _make_record(
                batch,
                {"term": term, "meaning": meaning, "context": context,
                 "source_ids": source_ids},
                now,
            )
        )
    return records


def validate_summary(parsed: Any, batch: LearnerBatch, *, now: float) -> list[Record]:
    """``{"summary": str, "recall_cues": [3–5 query-shaped sentences]}`` —
    exactly ONE record covering the full conversation."""
    if not isinstance(parsed, dict):
        raise LearnerValidationError("summary: expected a single JSON object")
    _forbid(parsed, "summary")
    summary = _require_str(parsed, "summary", "summary")
    cues = parsed.get("recall_cues")
    if not isinstance(cues, list):
        raise LearnerValidationError("summary: recall_cues must be a list")
    if not (SUMMARY_CUES_MIN <= len(cues) <= SUMMARY_CUES_MAX):
        raise LearnerValidationError(
            f"summary: exactly {SUMMARY_CUES_MIN}..{SUMMARY_CUES_MAX} recall cues,"
            f" got {len(cues)}"
        )
    for j, cue in enumerate(cues):
        if not isinstance(cue, str) or not cue.strip():
            raise LearnerValidationError(
                f"summary: recall_cues[{j}] must be a non-empty string"
            )
    return [
        _make_record(
            batch, {"summary": summary, "recall_cues": list(cues)}, now
        )
    ]


def validate_effect(parsed: Any, batch: LearnerBatch, *, now: float) -> list[Record]:
    """``{"categorization": str, "confidence": 0..1}`` — exactly ONE record.
    The bounded delta is derived by code, never set by the model."""
    if not isinstance(parsed, dict):
        raise LearnerValidationError("effect: expected a single JSON object")
    _forbid(parsed, "effect")
    categorization = _require_str(parsed, "categorization", "effect")
    if categorization not in EFFECT_CATEGORIZATIONS:
        raise LearnerValidationError(
            f"effect: categorization must be one of {EFFECT_CATEGORIZATIONS},"
            f" got {categorization!r}"
        )
    confidence = parsed.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise LearnerValidationError("effect: confidence must be a number")
    confidence = float(confidence)
    if not (0.0 <= confidence <= 1.0):
        raise LearnerValidationError("effect: confidence must be in [0, 1]")
    return [
        _make_record(
            batch, {"categorization": categorization, "confidence": confidence}, now
        )
    ]


def validate_impression(parsed: Any, batch: LearnerBatch, *, now: float) -> list[Record]:
    """``[{"source_id": N, "impression": ≤80字}]``, ≤5, one per person.

    The model names a person only by an opaque ref to something they wrote;
    this resolves that ref to the real ``platform_uid`` through
    ``batch.senders``, so the untrusted response can never invent an
    identity or point at someone who was not in the batch.
    """
    if not isinstance(parsed, list):
        raise LearnerValidationError("impression: expected a JSON array of records")
    if len(parsed) > IMPRESSION_MAX:
        raise LearnerValidationError(
            f"impression: at most {IMPRESSION_MAX} records, got {len(parsed)}"
        )
    records: list[Record] = []
    seen: set[str] = set()
    for i, item in enumerate(parsed):
        where = f"impression[{i}]"
        if not isinstance(item, dict):
            raise LearnerValidationError(f"{where}: not an object")
        _forbid(item, where)
        impression = _require_str(item, "impression", where)
        if len(impression) > IMPRESSION_FIELD_MAX:
            raise LearnerValidationError(
                f"{where}: impression exceeds {IMPRESSION_FIELD_MAX} chars"
            )
        source_id = _check_ref(_require_int(item, "source_id", where), batch, where)
        if not batch.senders:
            raise LearnerValidationError(
                f"{where}: batch carries no sender identity to attribute to"
            )
        platform_uid = str(batch.senders[source_id - 1])
        if platform_uid in seen:
            raise LearnerValidationError(
                f"{where}: duplicate impression for the same person"
            )
        seen.add(platform_uid)
        name = (
            batch.sender_names[source_id - 1] if batch.sender_names else ""
        )
        records.append(
            _make_record(
                batch,
                {
                    "platform_uid": platform_uid,
                    "name": name,
                    "impression": impression,
                    "source_id": source_id,
                },
                now,
            )
        )
    return records


VALIDATORS: dict[str, Validator] = {
    "expression": validate_expression,
    "behavior": validate_behavior,
    "jargon": validate_jargon,
    "summary": validate_summary,
    "effect": validate_effect,
    "impression": validate_impression,
}