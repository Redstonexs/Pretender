"""Strict model-output parsing for the learner lane (Phase 6).

The learner pipeline accepts EXACTLY two shapes of model output:

- raw JSON (the whole response content parses as JSON), or
- one outer code fence (`` ```json ... ``` `` or `` ``` ... ``` ``) with
  nothing before or after it.

Anything else — empty content, prose around a fence, an unbalanced fence, a
stray fence inside otherwise-raw JSON, or invalid JSON — raises
``LearnerParseError`` and the run settles malformed WITHOUT advancing the
watermark. There is no tolerant repair lane here: a learner record that is
guessed is worse than no record.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["LearnerParseError", "parse_json_response"]

# One outer code fence: an optional ``json`` tag, then the body, then the
# closing fence — with NOTHING before the opening marker or after the
# closing marker (the ``^``/``$`` anchors). ``\s*`` after the opening marker
# also accepts a single-line `` ```json {...} ``` `` form.
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class LearnerParseError(ValueError):
    """The model response is not raw JSON wrapped in exactly one outer
    fence. The run settles malformed; the watermark never advances."""


def parse_json_response(raw: str | None) -> Any:
    """Strictly parse a model response into a JSON value.

    Accepts raw JSON or exactly one outer code fence. Raises
    ``LearnerParseError`` for every other shape.
    """
    if raw is None:
        raise LearnerParseError("empty model response")
    if not isinstance(raw, str):
        raise LearnerParseError(
            f"model response must be a string, got {type(raw).__name__}"
        )
    text = raw.strip()
    if not text:
        raise LearnerParseError("empty model response")
    if text.startswith("```"):
        match = _FENCE_RE.match(text)
        if match is None:
            raise LearnerParseError(
                "malformed code fence: expected exactly one outer fence"
                " with nothing before or after it"
            )
        body = match.group(1).strip()
    else:
        if "```" in text:
            raise LearnerParseError(
                "stray code fence in response: expected raw JSON or exactly"
                " one outer fence"
            )
        body = text
    if not body:
        raise LearnerParseError("empty JSON body")
    try:
        return json.loads(body)
    except (ValueError, TypeError) as exc:
        raise LearnerParseError(f"invalid JSON: {exc}") from exc