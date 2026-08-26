"""Phase 6 learner lane: the generic pipeline + the five declarative specs.

Public surface for the learner lane:

- ``LearnerPipeline`` / ``LearnerRunResult`` — the generic async pipeline
  over a declarative ``LearnerSpec`` (source-bounded grant → prompt render →
  optional budget reservation → exactly one provider completion with a 45s
  deadline → strict JSON parsing → strict schema validation → all-or-nothing
  CAS commit).
- The five frozen specs (``EXPRESSION_SPEC`` … ``EFFECT_SPEC``), their
  ``SPECS`` registry and strict ``VALIDATORS``.
- ``derive_effect_delta`` — the code-owned bounded delta (the model only
  returns categorization/confidence).
- Rendering/identity helpers: ``render_batch`` (opaque per-batch refs),
  ``render_records`` / ``select_records`` (escaped reference surface),
  ``escape_untrusted``, ``source_hash`` / ``canonical_content``.
- ``parse_json_response`` — strict raw-JSON / one-outer-fence parsing.

No runtime worker or app wiring lives here yet (Phase 6 P6.2/P6.3 scope).
"""

from __future__ import annotations

from pretender.learn.effect import (
    EFFECT_BANDS,
    EFFECT_CATEGORIZATIONS,
    derive_effect_delta,
)
from pretender.learn.parse import LearnerParseError, parse_json_response
from pretender.learn.pipeline import (
    DEFAULT_LEARN_DEADLINE_S,
    DEFAULT_LEARN_FAILURE_BACKOFF_CAP_S,
    DEFAULT_LEARN_LEASE_S,
    LEARN_PROFILE,
    LearnerPipeline,
    LearnerRunResult,
)
from pretender.learn.render import (
    canonical_content,
    escape_untrusted,
    render_batch,
    render_records,
    select_records,
    source_hash,
)
from pretender.learn.specs import (
    ACTOR_TYPES,
    BEHAVIOR_MAX,
    BEHAVIOR_SPEC,
    EFFECT_SPEC,
    EXPRESSION_FIELD_MAX,
    EXPRESSION_MAX,
    EXPRESSION_SPEC,
    JARGON_MAX,
    JARGON_SPEC,
    LEARNING_TYPES,
    SPECS,
    SUMMARY_CUES_MAX,
    SUMMARY_CUES_MIN,
    SUMMARY_SPEC,
    VALIDATORS,
    LearnerValidationError,
)

__all__ = [
    # pipeline
    "LearnerPipeline",
    "LearnerRunResult",
    "LEARN_PROFILE",
    "DEFAULT_LEARN_DEADLINE_S",
    "DEFAULT_LEARN_FAILURE_BACKOFF_CAP_S",
    "DEFAULT_LEARN_LEASE_S",
    # the five specs + validators
    "EXPRESSION_SPEC",
    "BEHAVIOR_SPEC",
    "JARGON_SPEC",
    "SUMMARY_SPEC",
    "EFFECT_SPEC",
    "SPECS",
    "VALIDATORS",
    "ACTOR_TYPES",
    "LEARNING_TYPES",
    "EXPRESSION_MAX",
    "EXPRESSION_FIELD_MAX",
    "BEHAVIOR_MAX",
    "JARGON_MAX",
    "SUMMARY_CUES_MIN",
    "SUMMARY_CUES_MAX",
    # effect delta
    "EFFECT_CATEGORIZATIONS",
    "EFFECT_BANDS",
    "derive_effect_delta",
    # parsing
    "LearnerParseError",
    "parse_json_response",
    # validation
    "LearnerValidationError",
    # rendering / identity helpers
    "escape_untrusted",
    "render_batch",
    "render_records",
    "select_records",
    "source_hash",
    "canonical_content",
]
