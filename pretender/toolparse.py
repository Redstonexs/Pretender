"""Pure tolerant parsing of model-produced tool calls (Phase 3 capability lane).

This module turns whatever a model actually emitted — clean ``ToolCall``
objects, ``LLMResponse`` wrappers, raw JSON text, fenced/prose-wrapped JSON,
OpenAI-style ``{function: {name, arguments}}`` dicts, or genuine garbage —
into a flat, ordered ``tuple[ToolResult, ...]`` with a hard invariant:

  * exactly ONE ``ToolResult`` is returned for every distinct ``ToolCall`` id
    that is discovered in the input, on EVERY exit (including failures),
  * call ordering is preserved (first occurrence of an id wins; duplicates
    collapse),
  * no exception ever escapes for malformed input — the worst case is a
    ``no_action`` ``ToolResult`` (``ok=False``, ``name="no_action"``),
  * the module performs NO provider / tool / network I/O: tool ``handler``s
    are never invoked, nothing is imported that could touch the network, and
    schema validation is a self-contained, conservative JSON-Schema check.

Recovery pipeline (per JSON text snippet):

  1. ``json.loads`` on the extracted JSON block (prose and `````` fences are
     skipped by ``_extract_json_block``).
  2. A built-in tolerance pass (``_repair_json_text``): trailing commas,
     single-quoted strings, unquoted keys, bare words, unterminated
     strings/objects, stray closers/backticks/control characters.
  3. ONE caller-injected repair attempt (``repair=``) — invoked at most once
     per malformed snippet. If it raises or returns unparseable text, the
     call degrades to ``no_action`` rather than raising.
  4. A TEXT lane (``_units_from_text_lane``) for models that narrate their
     tool call instead of emitting a structured one — the Hermes/MiMo
     ``<tool_call><function=NAME><parameter=KEY>…`` form and the
     ``<tool_call><tool_name>NAME</tool_name>…`` form. It runs ONLY where the
     JSON lanes above recovered nothing, never reinterprets malformed JSON,
     and only inside an explicit ``<tool_call>``/``<function=…>`` wrapper so
     ordinary prose can never become a call.

A recovered call carries no provider id, so a deterministic synthetic id is
assigned (``_finalize_text_units``) — but ONLY when the name is a tool the
caller actually registered, so arbitrary text/JSON never becomes a call.
Text-lane arguments arrive as strings and are coerced to the ``ToolSpec``
parameter schema's declared type; JSON-sourced arguments keep the strict
behaviour.

``arguments`` that arrive as a JSON *string* (the OpenAI convention) are
decoded with the same pipeline; a non-object ``arguments`` (or an object that
fails the ``ToolSpec`` parameter schema) yields an ``ok=False`` result that
still carries the call id, so no id is ever orphaned.

Public API: ``parse_tool_calls``, ``tolerant_loads``, ``validate_arguments``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

from pretender.tools.base import ToolSpec
from pretender.types import LLMResponse, ToolCall, ToolCallId, ToolResult

__all__ = ["NO_ACTION_NAME", "parse_tool_calls", "tolerant_loads", "validate_arguments"]

#: The tool name carried by a degraded ``ToolResult`` (call could not be
#: parsed/recovered at all). Downstream executors treat it as "take no action".
NO_ACTION_NAME = "no_action"

# Matches `"id": "call_1"`, `'call_id': 'x'`, `id = foo-2`, ... — used ONLY to
# salvage call ids from text that failed to decode as JSON at all, so a
# referenced id is never orphaned.
_ID_RE = re.compile(r"""["']?(?:call_id|id)["']?\s*[:=]\s*"?([A-Za-z0-9_.\-]+)"?""")

# The text lane. Models that lack (or ignore) native function calling narrate
# the call in the assistant content instead. These match ONLY inside an
# explicit wrapper — never loose prose tags.
_TOOL_CALL_BLOCK_RE = re.compile(r"<\s*tool_call\s*>(.*?)<\s*/\s*tool_call\s*>", re.S | re.I)
_FUNCTION_BLOCK_RE = re.compile(
    r"<\s*function\s*=\s*([A-Za-z0-9_.\-]+)\s*>(.*?)<\s*/\s*function\s*>", re.S | re.I
)
_PARAMETER_RE = re.compile(
    r"<\s*parameter\s*=\s*([A-Za-z0-9_.\-]+)\s*>(.*?)<\s*/\s*parameter\s*>", re.S | re.I
)
_TAG_PAIR_RE = re.compile(r"<\s*([A-Za-z_][A-Za-z0-9_.\-]*)\s*>(.*?)<\s*/\s*\1\s*>", re.S)
# The key a text-lane unit is flagged with; stripped before the unit is parsed.
_TEXT_LANE_KEY = "__pretender_text_lane__"
_XML_ENTITIES = {
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&apos;": "'",
    "&#39;": "'",
    "&nbsp;": " ",
}
_NAME_TAGS = ("tool_name", "function_name", "name", "tool")
_ARGS_TAGS = ("arguments", "args", "parameters", "params")


# ── public API ──────────────────────────────────────────────────────────────

def parse_tool_calls(
    source: Any,
    tool_specs: Any = (),
    *,
    repair: Callable[[str], str] | None = None,
) -> tuple[ToolResult, ...]:
    """Parse ``source`` into exactly one ``ToolResult`` per discovered id.

    ``source`` may be an ``LLMResponse``, a ``ToolCall``, a sequence of
    ``ToolCall``/dict/str, a dict (``{"tool_calls": [...]}``, a single call,
    or an OpenAI ``{id, function: {name, arguments}}`` shape), or a string of
    (possibly fenced/prose-wrapped/malformed) JSON.

    ``tool_specs`` is an iterable of ``ToolSpec`` (or a mapping of
    name -> ``ToolSpec`` / parameter schema, or a ``ToolRegistry``) whose
    ``parameters`` schemas validate each call's arguments.

    ``repair`` is an optional ``str -> str`` callback given ONE chance to fix
    a JSON snippet the built-in tolerance could not recover. If it raises or
    returns unparseable text, the affected call degrades to a ``no_action``
    result. Never invoked more than once per malformed snippet.

    Never raises for malformed input and never performs I/O.
    """
    specs = _build_spec_map(tool_specs)
    try:
        units = _collect_units(source, repair)
    except Exception as exc:  # containment boundary — never propagate
        units = _salvage_ids(source)
        if not units:
            return ()
        units = [
            _NoParse(cid, f"internal error: {type(exc).__name__}: {exc}")
            for cid in units
        ]
    units = _finalize_text_units(units, specs)
    results: list[ToolResult] = []
    seen: set[str] = set()
    for unit in units:
        res = _parse_unit(unit, specs, repair)
        if res is None:
            continue
        if res.call_id in seen:
            continue  # duplicate id — first occurrence already answered
        seen.add(res.call_id)
        results.append(res)
    return tuple(results)


def tolerant_loads(text: str) -> Any:
    """Best-effort tolerant JSON decode of ``text``.

    Extracts the first JSON block (`````` fences and surrounding prose are
    ignored), then applies ``_repair_json_text`` for common model mistakes
    (trailing commas, single quotes, unquoted keys, bare words, unterminated
    strings/objects). Raises ``json.JSONDecodeError`` (a ``ValueError``) if
    nothing recoverable remains.
    """
    block = _extract_json_block(text)
    if block is None:
        raise json.JSONDecodeError("no JSON block found", text, 0)
    try:
        return json.loads(block)
    except (ValueError, RecursionError):
        fixed = _repair_json_text(block)
        return json.loads(fixed)


def validate_arguments(arguments: Any, parameters: Any) -> list[str]:
    """Validate ``arguments`` against a JSON Schema ``parameters`` object.

    Returns a list of human-readable errors (empty list = valid). The check
    is conservative and tolerant: unknown schema keys and absent schemas
    (``{}`` / non-dict) are permissive, extra properties are allowed unless
    the schema says otherwise, ``bool`` is NOT accepted for ``integer``/
    ``number``, and ``"null"``-typed properties accept ``None``.
    """
    if not isinstance(arguments, Mapping):
        return [f"arguments must be an object, got {_type_of(arguments)}"]
    if not isinstance(parameters, Mapping):
        return []
    errors: list[str] = []
    required = parameters.get("required")
    if isinstance(required, list):
        for key in required:
            if key not in arguments:
                errors.append(f"missing required parameter {key!r}")
    props = parameters.get("properties")
    if isinstance(props, Mapping):
        for key, value in arguments.items():
            if key in props:
                errors.extend(_check_value(value, props[key], f"argument {key!r}"))
            else:
                extra = parameters.get("additionalProperties")
                if isinstance(extra, Mapping) and extra:
                    errors.extend(_check_value(value, extra, f"argument {key!r}"))
    return errors


# ── source collection ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class _NoParse:
    """A call id discovered only via salvage of text that never decoded as
    JSON. Always becomes a ``no_action`` ``ToolResult``."""

    call_id: str
    reason: str = "unparseable tool call"


def _collect_units(source: Any, repair: Callable[[str], str] | None) -> list[Any]:
    """Flatten any accepted source shape into an ordered list of call units
    (``ToolCall``, mapping, or ``_NoParse``)."""
    if isinstance(source, LLMResponse):
        return _collect_units(source.tool_calls, repair)
    if isinstance(source, ToolCall):
        return [source]
    if isinstance(source, str):
        return _units_from_text(source, repair)
    if isinstance(source, bytes):
        try:
            return _units_from_text(source.decode("utf-8", "replace"), repair)
        except Exception:
            return []
    if isinstance(source, (list, tuple)):
        units: list[Any] = []
        for el in source:
            units.extend(_collect_units(el, repair))
        return units
    if isinstance(source, Mapping):
        return _units_from_decoded(source, repair)
    return []  # unknown/garbage source — no ids discoverable, nothing orphaned


def _units_from_text(text: str, repair: Callable[[str], str] | None) -> list[Any]:
    ok, decoded, err = _decode_json(text, repair)
    if ok:
        units = _units_from_decoded(decoded, repair)
        covered = {cid for cid in _ids_from_units(units)}
        # An id mentioned in the PROSE (outside any decoded JSON block) but
        # NOT part of any decoded unit is still a referenced call id —
        # answer it with a no_action rather than orphaning it (e.g.
        # `prose "id": "call_9" [[[` decodes the brackets into an empty
        # structure and would otherwise drop the id). Ids INSIDE a decoded
        # JSON block — including nested argument fields named ``id`` — are
        # never synthetic call ids.
        extras = [i for i in _ids_outside_json_blocks(text) if i not in covered]
        for extra in extras:
            units.append(
                _NoParse(extra, f"unparseable tool call fragment: {err or 'invalid JSON'}")
            )
        if _has_usable_unit(units):
            return units
        # The decode succeeded but yielded nothing that names or answers a
        # call — e.g. the model wrapped its call in tags and the extractor
        # picked up only the inner ``<arguments>`` object. The text lane is
        # the last chance to recover it; failing that, keep the JSON units so
        # the existing drop/answer rules still apply.
        return _units_from_text_lane(text) or units
    units = _units_from_text_lane(text)
    if units:
        return units
    ids = _extract_ids_from_text(text)
    if not ids:
        return []
    return [_NoParse(i, f"unparseable tool call: {err or 'invalid JSON'}") for i in ids]


# ── the text lane ───────────────────────────────────────────────────────────

def _units_from_text_lane(text: str) -> list[Any]:
    """Recover tool calls a model narrated as markup instead of emitting them
    structurally.

    Two shapes are recognised, both only INSIDE an explicit wrapper so that
    ordinary prose containing angle brackets can never become a call:

      ``<tool_call><function=NAME><parameter=KEY>VALUE</parameter>…</function></tool_call>``
      ``<tool_call><tool_name>NAME</tool_name><KEY>VALUE</KEY>…</tool_call>``

    A bare ``<function=NAME>…</function>`` (no ``<tool_call>`` wrapper) is
    accepted too — several models emit it that way. Returns text-lane units
    (flagged mappings without ids); ``_finalize_text_units`` assigns the id
    and coerces the argument types.
    """
    if not isinstance(text, str) or "<" not in text:
        return []
    units: list[Any] = []
    blocks = [m.group(1) for m in _TOOL_CALL_BLOCK_RE.finditer(text)]
    if not blocks:
        # No wrapper: accept bare <function=NAME> blocks, nothing else.
        if not _FUNCTION_BLOCK_RE.search(text):
            return []
        blocks = [text]
    for body in blocks:
        units.extend(_units_from_text_block(body))
    return units


def _units_from_text_block(body: str) -> list[Any]:
    """Every call unit inside one wrapper body."""
    units: list[Any] = []
    matched_any = False
    for match in _FUNCTION_BLOCK_RE.finditer(body):
        matched_any = True
        name = match.group(1).strip()
        args = {
            key.strip(): _unescape_markup(value)
            for key, value in _PARAMETER_RE.findall(match.group(2))
        }
        if name:
            units.append(_text_unit(name, args))
    if matched_any:
        return units
    unit = _unit_from_tag_pairs(body)
    return [unit] if unit is not None else []


def _unit_from_tag_pairs(body: str) -> Any:
    """The ``<tool_name>NAME</tool_name><KEY>VALUE</KEY>`` shape.

    The first name-ish tag names the tool; an ``<arguments>`` tag holding a
    JSON object supplies the arguments wholesale, otherwise every remaining
    tag pair becomes one argument.
    """
    pairs = [(k.strip().lower(), v) for k, v in _TAG_PAIR_RE.findall(body)]
    if not pairs:
        return None
    name = ""
    for key, value in pairs:
        if key in _NAME_TAGS:
            name = _unescape_markup(value).strip()
            break
    if not name:
        return None
    for key, value in pairs:
        if key in _ARGS_TAGS:
            try:
                decoded = json.loads(_unescape_markup(value))
            except (ValueError, RecursionError):
                decoded = None
            if isinstance(decoded, Mapping):
                return _text_unit(name, dict(decoded))
    args = {
        key: _unescape_markup(value)
        for key, value in pairs
        if key not in _NAME_TAGS and key not in _ARGS_TAGS
    }
    return _text_unit(name, args)


def _has_usable_unit(units: list[Any]) -> bool:
    """Whether any unit can actually answer or name a call. A decoded object
    with neither an id nor a tool name (a stray argument object, say) is not a
    call unit, and must not shadow the text lane."""
    for unit in units:
        if isinstance(unit, (_NoParse, ToolCall)):
            return True
        if isinstance(unit, Mapping) and (
            _id_from_unit(unit) is not None or _unit_name(unit)
        ):
            return True
    return False


def _text_unit(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {_TEXT_LANE_KEY: True, "name": name, "arguments": arguments}


def _unescape_markup(value: str) -> str:
    """Undo the handful of entities models actually emit, then trim. Nothing
    else is interpreted — the value is data, never markup."""
    out = value
    for entity, char in _XML_ENTITIES.items():
        if entity in out:
            out = out.replace(entity, char)
    # ``&amp;`` last so ``&amp;lt;`` does not become ``<``.
    if "&amp;" in out:
        out = out.replace("&amp;", "&")
    return out.strip()


# ── finalization: synthetic ids + text-lane type coercion ───────────────────

def _finalize_text_units(units: list[Any], specs: dict[str, dict[str, Any]]) -> list[Any]:
    """Give id-less recovered units a deterministic id and coerce text-lane
    arguments to their declared schema types.

    An id is minted ONLY when the unit names a tool the caller registered —
    arbitrary JSON or markup that happens to carry a ``name`` never becomes a
    dispatchable call. Units that already carry an id are returned untouched,
    so every structured/JSON path keeps its exact behaviour.
    """
    if not units:
        return units
    out: list[Any] = []
    minted = 0
    for unit in units:
        text_lane = isinstance(unit, Mapping) and bool(unit.get(_TEXT_LANE_KEY))
        if not isinstance(unit, Mapping) or _id_from_unit(unit) is not None:
            if text_lane:
                unit = {k: v for k, v in unit.items() if k != _TEXT_LANE_KEY}
            out.append(unit)
            continue
        name = _unit_name(unit)
        if not name or name not in specs:
            # Unknown/absent tool name: leave the unit exactly as it was so it
            # is dropped (or answered) by the existing rules.
            out.append(unit)
            continue
        finalized = {k: v for k, v in unit.items() if k != _TEXT_LANE_KEY}
        minted += 1
        finalized["id"] = f"call_recovered_{minted}"
        if text_lane:
            args = finalized.get("arguments")
            if isinstance(args, Mapping):
                finalized["arguments"] = _coerce_text_arguments(args, specs[name])
        out.append(finalized)
    return out


def _unit_name(unit: Mapping[str, Any]) -> str:
    fn = unit.get("function")
    name = fn.get("name") if isinstance(fn, Mapping) else None
    if name is None:
        name = unit.get("name")
    return name.strip() if isinstance(name, str) else ""


def _coerce_text_arguments(
    arguments: Mapping[str, Any], parameters: Any
) -> dict[str, Any]:
    """Coerce string argument values to the types the schema declares.

    Markup carries only text, so ``<parameter=seconds>5</parameter>`` would
    otherwise fail a ``number`` schema. Applied to text-lane units only; a
    value that will not convert is left alone so ``validate_arguments``
    reports the real mismatch.
    """
    props = parameters.get("properties") if isinstance(parameters, Mapping) else None
    if not isinstance(props, Mapping):
        return dict(arguments)
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        schema = props.get(key)
        out[key] = _coerce_text_value(value, schema) if isinstance(schema, Mapping) else value
    return out


def _coerce_text_value(value: Any, schema: Mapping[str, Any]) -> Any:
    if not isinstance(value, str):
        return value
    declared = schema.get("type")
    types = declared if isinstance(declared, list) else ([declared] if declared else [])
    if not types:
        return value
    text = value.strip()
    if "string" in types:
        # A string already satisfies the schema; only an empty value for a
        # nullable property is worth converting (an omitted <parameter> body).
        return None if (not text and "null" in types) else value
    for name in types:
        if name == "null" and text.lower() in ("", "null", "none"):
            return None
        if name == "boolean" and text.lower() in ("true", "false"):
            return text.lower() == "true"
        if name == "integer":
            try:
                return int(text, 10)
            except ValueError:
                continue
        if name == "number":
            try:
                return float(text)
            except ValueError:
                continue
        if name in ("object", "array"):
            try:
                decoded = json.loads(text)
            except (ValueError, RecursionError):
                continue
            if isinstance(decoded, Mapping if name == "object" else list):
                return decoded
    return value


def _units_from_decoded(decoded: Any, repair: Callable[[str], str] | None) -> list[Any]:
    if isinstance(decoded, Mapping):
        tc = decoded.get("tool_calls")
        if isinstance(tc, (list, tuple)):
            return _normalize_list(tc, repair)
        single = decoded.get("tool_call")
        if isinstance(single, Mapping):
            return [single]
        choices = decoded.get("choices")
        if isinstance(choices, (list, tuple)) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                msg = first.get("message")
                if isinstance(msg, Mapping):
                    nested = msg.get("tool_calls")
                    if isinstance(nested, (list, tuple)):
                        return _normalize_list(nested, repair)
        return [decoded]
    if isinstance(decoded, list):
        return _normalize_list(decoded, repair)
    return []


def _normalize_list(seq: Sequence[Any], repair: Callable[[str], str] | None) -> list[Any]:
    """Normalize a sequence of call units, recursing into string elements so
    a JSON-encoded call inside a ``tool_calls`` list is never orphaned."""
    units: list[Any] = []
    for el in seq:
        if isinstance(el, (Mapping, ToolCall)):
            units.append(el)
        elif isinstance(el, str):
            units.extend(_units_from_text(el, repair))
    return units


def _ids_from_units(units: list[Any]) -> list[Any]:
    ids: list[Any] = []
    for u in units:
        if isinstance(u, ToolCall):
            ids.append(u.id)
        elif isinstance(u, Mapping):
            cid = u.get("id")
            if isinstance(cid, str):
                ids.append(cid)
        elif isinstance(u, _NoParse):
            ids.append(u.call_id)
    return ids


def _salvage_ids(source: Any) -> list[str]:
    """Fallback id extraction used only when collection itself blew up."""
    try:
        text = str(source)
    except Exception:
        return []
    return _extract_ids_from_text(text)


def _extract_ids_from_text(text: str) -> list[str]:
    ids: list[str] = []
    for m in _ID_RE.finditer(text):
        cid = m.group(1).strip()
        if cid and cid not in ids:
            ids.append(cid)
    return ids


def _ids_outside_json_blocks(text: str) -> list[str]:
    """Call ids referenced in ``text`` OUTSIDE any balanced JSON block.

    Nested argument fields named ``id`` live inside a decoded JSON block and
    must never become synthetic call ids — only ids in surrounding prose are
    salvageable references.
    """
    return _extract_ids_from_text(_strip_json_blocks(text))


def _strip_json_blocks(text: str) -> str:
    """Remove every balanced JSON block (string-aware) from ``text``,
    leaving only surrounding prose. An unbalanced opener (e.g. a stray
    ``[[[``) is skipped to its end so its prose is preserved."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "{[":
            depth = 0
            in_str = False
            escaped = False
            j = i
            while j < n:
                c = text[j]
                if in_str:
                    if escaped:
                        escaped = False
                    elif c == "\\":
                        escaped = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c in "{[":
                        depth += 1
                    elif c in "}]":
                        depth -= 1
                        if depth == 0:
                            break
                j += 1
            i = j + 1  # skip the whole block (balanced or not)
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# ── per-unit parsing ────────────────────────────────────────────────────────

def _parse_unit(
    unit: Any,
    specs: dict[str, dict[str, Any]],
    repair: Callable[[str], str] | None,
) -> ToolResult | None:
    try:
        return _parse_unit_impl(unit, specs, repair)
    except Exception as exc:  # containment boundary — never propagate
        cid: Any = None
        try:
            cid = _id_from_unit(unit)
        except Exception:
            cid = None
        if cid is None:
            try:
                cid = next(iter(_extract_ids_from_text(str(unit))), None)
            except Exception:
                cid = None
        if cid is None:
            return None
        return _no_action(cid, f"internal error: {type(exc).__name__}: {exc}")


def _parse_unit_impl(
    unit: Any,
    specs: dict[str, dict[str, Any]],
    repair: Callable[[str], str] | None,
) -> ToolResult | None:
    if isinstance(unit, _NoParse):
        return _no_action(unit.call_id, unit.reason)
    if isinstance(unit, ToolCall):
        unit = {
            "id": unit.id,
            "name": unit.name,
            "arguments": unit.arguments,
            "raw_arguments": unit.raw_arguments,
        }
    if not isinstance(unit, Mapping):
        return None  # not a call unit; no id to answer

    cid = unit.get("id")
    if isinstance(cid, bool):
        cid = None
    elif isinstance(cid, (int, float)):
        cid = str(int(cid)) if float(cid).is_integer() else str(cid)
    if not isinstance(cid, str) or not cid.strip():
        return None  # no id — nothing to answer, nothing orphaned
    cid = cid.strip()

    fn = unit.get("function")
    if isinstance(fn, Mapping):
        name = fn.get("name")
        if name is None:
            name = unit.get("name")
        args = fn.get("arguments", {}) if "arguments" in fn else unit.get("arguments", {})
        raw_args = (
            fn.get("raw_arguments")
            if "raw_arguments" in fn
            else unit.get("raw_arguments")
        )
    else:
        name = unit.get("name")
        args = unit.get("arguments", {})
        raw_args = unit.get("raw_arguments")

    if not isinstance(name, str) or not name.strip():
        return ToolResult(
            call_id=ToolCallId(cid),
            name=NO_ACTION_NAME,
            ok=False,
            error="missing tool name",
        )
    name = name.strip()

    if isinstance(args, str):
        ok, decoded, err = _decode_json(args, repair)
        if not ok:
            return ToolResult(
                call_id=ToolCallId(cid),
                name=name,
                ok=False,
                error=f"malformed arguments: {err or 'invalid JSON'}",
            )
        args = decoded
    elif raw_args is not None:
        # The provider's arguments could not be parsed into an object at the
        # LLM layer; run the SAME tolerant one-repair pipeline on the raw
        # value here so a malformed call degrades instead of aborting.
        if isinstance(raw_args, str):
            ok, decoded, err = _decode_json(raw_args, repair)
            if not ok:
                return ToolResult(
                    call_id=ToolCallId(cid),
                    name=name,
                    ok=False,
                    error=f"malformed arguments: {err or 'invalid JSON'}",
                )
            args = decoded
        else:
            return ToolResult(
                call_id=ToolCallId(cid),
                name=name,
                ok=False,
                error=f"malformed arguments: expected an object, got {_type_of(raw_args)}",
            )
    elif args is None:
        args = {}
    if not isinstance(args, Mapping):
        return ToolResult(
            call_id=ToolCallId(cid),
            name=name,
            ok=False,
            error=f"malformed arguments: expected an object, got {_type_of(args)}",
        )

    parameters = specs.get(name)
    if parameters is None:
        return ToolResult(
            call_id=ToolCallId(cid),
            name=name,
            ok=False,
            error=f"unknown tool: {name}",
        )

    errors = validate_arguments(args, parameters)
    if errors:
        return ToolResult(
            call_id=ToolCallId(cid),
            name=name,
            ok=False,
            error="schema mismatch: " + "; ".join(errors),
        )

    args_dict = dict(args)
    try:
        content = json.dumps(
            args_dict, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        content = repr(args_dict)
    return ToolResult(
        call_id=ToolCallId(cid),
        name=name,
        ok=True,
        content=content,
        data=args_dict,
    )


def _id_from_unit(unit: Any) -> Any:
    if isinstance(unit, ToolCall):
        return unit.id
    if isinstance(unit, _NoParse):
        return unit.call_id
    if isinstance(unit, Mapping):
        cid = unit.get("id")
        if isinstance(cid, bool) or not isinstance(cid, (str, int, float)):
            return None
        return str(cid).strip() or None
    return None


def _no_action(call_id: Any, reason: str) -> ToolResult:
    return ToolResult(
        call_id=ToolCallId(str(call_id)),
        name=NO_ACTION_NAME,
        ok=False,
        error=str(reason),
    )


# ── tool spec index ─────────────────────────────────────────────────────────

def _build_spec_map(tool_specs: Any) -> dict[str, dict[str, Any]]:
    """Normalize any caller-supplied tool collection to ``name -> parameters``.

    Accepts an iterable of ``ToolSpec``, a mapping of name -> ``ToolSpec`` or
    name -> parameter schema (a ``{name, parameters}`` dict is treated as a
    spec; anything else as the schema directly), or a ``ToolRegistry``.
    """
    specs: dict[str, dict[str, Any]] = {}

    def add(name: Any, parameters: Any) -> None:
        if isinstance(name, str) and name:
            specs[name] = dict(parameters) if isinstance(parameters, Mapping) else {}

    if tool_specs is None:
        return specs
    if isinstance(tool_specs, Mapping):
        for key, item in tool_specs.items():
            if isinstance(item, ToolSpec):
                add(item.name, item.parameters)
            elif isinstance(item, Mapping):
                if isinstance(item.get("name"), str) and "parameters" in item:
                    add(item["name"], item.get("parameters"))
                else:
                    add(key, item)
            else:
                add(key, item)
        return specs
    if callable(getattr(tool_specs, "all", None)):
        tool_specs = tool_specs.all()
    for item in tool_specs:
        if isinstance(item, ToolSpec):
            add(item.name, item.parameters)
        elif isinstance(item, Mapping) and isinstance(item.get("name"), str):
            add(item["name"], item.get("parameters"))
    return specs


# ── tolerant JSON decoding ──────────────────────────────────────────────────

def _decode_json(
    text: Any, repair: Callable[[str], str] | None
) -> tuple[bool, Any, str | None]:
    """Decode ``text`` tolerantly, with ONE caller repair attempt on failure.

    Returns ``(ok, value, err)``; ``err`` is None on success. Never raises
    for malformed input (a raising ``repair`` is itself contained).
    """
    if not isinstance(text, str) or not text.strip():
        return False, None, "not text"
    try:
        return True, tolerant_loads(text), None
    except (ValueError, RecursionError):
        pass
    if repair is not None:
        try:
            fixed = repair(text)
        except Exception as exc:
            return False, None, f"repair raised {type(exc).__name__}"
        if isinstance(fixed, str) and fixed.strip():
            try:
                return True, tolerant_loads(fixed), None
            except (ValueError, RecursionError):
                return False, None, "repair produced invalid JSON"
    return False, None, "invalid JSON"


def _extract_json_block(text: str) -> str | None:
    """The first balanced ``{...}``/``[...]`` block in ``text`` (string-aware),
    ignoring fences and surrounding prose. Unbalanced input returns everything
    from the first opener — the repair pass closes the remainder."""
    start = -1
    for idx, ch in enumerate(text):
        if ch in "{[":
            start = idx
            break
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _repair_json_text(text: str) -> str:
    """String-aware scan that fixes common model JSON mistakes:

    * trailing commas before ``}``/``]``,
    * single-quoted strings (``\\'`` escapes, ``"`` inside a single-quoted
      string escaped),
    * unquoted keys and bare-word values (``NaN``/``Infinity``/``undefined``
      -> ``null``, ``true``/``false`` case-insensitively, everything else
      quoted as a string),
    * unknown escapes (backslash dropped), stray backslashes/backticks and
      stray closers dropped, control characters stripped,
    * unterminated strings closed and unterminated brackets balanced.
    """
    out: list[str] = []
    stack: list[str] = []
    in_dq = False
    in_sq = False
    i = 0
    n = len(text)

    def _key_start(j: int) -> bool:
        k = j - 1
        while k >= 0 and text[k] in " \t\r\n":
            k -= 1
        return k >= 0 and text[k] in "{,"

    while i < n:
        ch = text[i]

        if in_dq:
            if ch == "\\":
                if i + 1 >= n:
                    out.append("\\")
                    i += 1
                    continue
                nxt = text[i + 1]
                if nxt == "u":
                    seg = text[i + 1 : i + 6]
                    out.append("\\")
                    out.append(seg)
                    i += 1 + len(seg)
                    continue
                if nxt == "'":
                    out.append("'")  # \\' inside "..." is an invalid escape
                    i += 2
                    continue
                if nxt in '"\\/bfnrt':
                    out.append("\\")
                    out.append(nxt)
                    i += 2
                    continue
                out.append(nxt)  # unknown escape: drop the backslash
                i += 2
                continue
            if ch == '"':
                in_dq = False
            out.append(ch)
            i += 1
            continue

        if in_sq:
            if ch == "\\":
                if i + 1 >= n:
                    out.append("\\")
                    i += 1
                    continue
                nxt = text[i + 1]
                if nxt == "'":
                    out.append("'")
                else:
                    out.append("\\")
                    out.append(nxt)
                i += 2
                continue
            if ch == "'":
                in_sq = False
                out.append('"')  # close as a double quote
                i += 1
                continue
            if ch == '"':
                out.append('\\"')  # " inside '...' must be escaped in "..."
                i += 1
                continue
            out.append(ch)
            i += 1
            continue

        # ── outside any string ──────────────────────────────────────────
        if ch == '"':
            in_dq = True
            out.append(ch)
            i += 1
            continue
        if ch == "'":
            in_sq = True
            out.append('"')
            i += 1
            continue
        if ch in "{[":
            stack.append(ch)
            out.append(ch)
            i += 1
            continue
        if ch in "}]":
            if stack and (
                (ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")
            ):
                stack.pop()
                out.append(ch)
            # else: stray/mismatched closer — dropped
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1  # trailing comma — drop
                continue
            out.append(ch)
            i += 1
            continue
        if ch == "`":
            i += 1  # stray backtick outside a string — drop
            continue
        if ch == "\\":
            i += 1  # stray backslash outside a string — drop
            continue
        if ch == "_" or ch.isalpha():
            j = i
            while j < n and (text[j].isalnum() or text[j] in "_-"):
                j += 1
            ident = text[i:j]
            k = j
            while k < n and text[k] in " \t\r\n":
                k += 1
            if k < n and text[k] == ":" and _key_start(i):
                out.append('"')
                out.append(ident)
                out.append('"')
                i = j
                continue
            low = ident.lower()
            if low in ("null", "nan", "infinity", "undefined"):
                out.append("null")
            elif low == "true":
                out.append("true")
            elif low == "false":
                out.append("false")
            else:
                out.append('"')
                out.append(ident)
                out.append('"')
            i = j
            continue
        if ord(ch) < 0x20 and ch not in "\t\n\r":
            i += 1  # control character — drop
            continue
        out.append(ch)
        i += 1

    if in_dq or in_sq:
        out.append('"')  # unterminated string — close it
    while stack:
        opener = stack.pop()
        out.append("}" if opener == "{" else "]")
    return "".join(out)


def _type_of(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, Mapping):
        return "dict"
    return type(value).__name__


def _matches_type(value: Any, types: list[Any]) -> bool:
    for t in types:
        if t == "null":
            if value is None:
                return True
        elif t == "string":
            if isinstance(value, str):
                return True
        elif t == "integer":
            if isinstance(value, int) and not isinstance(value, bool):
                return True
        elif t == "number":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True
        elif t == "boolean":
            if isinstance(value, bool):
                return True
        elif t == "array":
            if isinstance(value, (list, tuple)):
                return True
        elif t == "object":
            if isinstance(value, Mapping):
                return True
        else:
            return True  # unknown type name — permissive
    return False


def _check_value(value: Any, schema: Any, path: str) -> list[str]:
    if not isinstance(schema, Mapping) or not schema:
        return []
    errors: list[str] = []
    t = schema.get("type")
    types = t if isinstance(t, list) else ([t] if t else [])
    if types and not _matches_type(value, types):
        errors.append(f"{path}: expected type {t!r}, got {_type_of(value)}")
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"{path}: {value!r} not in enum {enum}")
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, Mapping) and items:
            for idx, item in enumerate(value):
                errors.extend(_check_value(item, items, f"{path}[{idx}]"))
    elif isinstance(value, Mapping):
        props = schema.get("properties")
        if isinstance(props, Mapping):
            for k, v in value.items():
                if k in props:
                    errors.extend(_check_value(v, props[k], f"{path}.{k}"))
    return errors
