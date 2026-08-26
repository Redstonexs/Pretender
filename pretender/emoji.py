"""Media catalog key/candidate validation (Phase 6 P6.5 foundation).

Pure, behavior-light validation for the durable chat-scoped media catalog.
The catalog key is OPAQUE: it is a content-addressed cache key (the sha256
hex digest of the normalized bytes, as produced by ``media.MediaStore``),
never a local path, URL, data/base64 payload, or raw platform media
reference. This module validates that opacity — the repository enforces it
at submit time so a catalog row can never name a fetchable/executable
source.

Gate 6: this module also owns the STRICT structured vision verdict. A
candidate is approved ONLY when the vision response carries an explicit
boolean ``safe: true`` classification AND a valid bounded escaped one-line
description (``parse_vision_result``). No arbitrary text ever approves a
candidate; a missing/failed/malformed/unsafe response leaves the candidate
PENDING (unapproved). Catalog descriptions are normalized by
``normalize_description`` into a bounded, escaped, one-line string so a
catalog row can never carry a fetchable source or break the prompt/tool/
outbox structure through its description.

Existing global ``emoji`` rows remain legacy/untrusted: this module and the
catalog never read them. No harvesting, vision, sending, or tool wiring
happens here — this is the foundation the Phase 6 P6.5b harvest/send lanes
build on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pretender.types import MediaAssetCandidate

#: The opaque catalog key / content sha256 shape: a 64-character lowercase
#: hex digest (the content-addressed key ``media.MediaStore`` produces).
_SHA256_HEX = re.compile(r"^[a-f0-9]{64}$")

#: A source-like token a catalog description must never carry.  Keep this
#: deliberately broader than HTTP: a description is not a transport for a
#: file path, platform media reference, data URI, or encoded payload.
_SOURCE_TOKEN = re.compile(
    r"(?ix)"
    r"(?:[a-z][a-z0-9+.-]{1,31}://\S+)"       # http/file/opaque schemes
    r"|(?:data:[^\s]+)"                       # data URI
    r"|(?:base64(?:://|[:,]))\S*"             # explicit base64 reference
    r"|(?:;base64(?:,|$))\S*"                 # data MIME marker
    r"|(?:\{(?:file|url|data|base64)=[^}]*\})"
    r"|(?:\[CQ:[^\]]*(?:file|url|data|base64)=[^\]]*\])"
    r"|(?:[a-z]:[\\/][^\s]+)"                # Windows path
    r"|(?:\\\\[^\s\\]+)"                    # UNC path
    r"|(?:~[\\/][^\s]+)"                      # home-relative path
    r"|(?:/[^\s]+(?:/[^\s]+)+)"              # absolute POSIX path
    r"|(?:(?<!\w)(?:\.\.?[\\/])[^\s]+)"    # relative path
    r"|(?:www\.[^\s]+)"                        # scheme-less web URL
    r"|(?:\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:[/:?#][^\s]*)?)"  # bare domain
    r"|(?:\b(?:file|url|path)=[^\s]+)"         # raw source field
    r"|(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{32,}={0,2}(?![A-Za-z0-9+/=])"
)

#: The maximum length of a catalog description (bounded, one line).
DESCRIPTION_MAX_CHARS = 120

#: Ordinary whitespace runs collapse to a single space so a description is
#: always ONE line; control whitespace is escaped separately below.
_WS_RUN = re.compile(r"\s+")

#: Control characters that must be escaped so a description can never break
#: the prompt/tool/outbox structure (C0 + DEL + C1).
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def validate_catalog_key(key: str) -> None:
    """Validate an opaque catalog cache key.

    The key must be a 64-character lowercase hex string (a sha256 hex
    digest — the content-addressed key ``media.MediaStore`` produces).
    This rejects local paths (``/abs``, ``./rel``, ``~/``, ``C:\\``),
    URLs (``http://``, ``file://``, ...), data/base64 payloads, and raw
    platform media references (``{file=...}``, ``base64://...``) as
    catalog keys. Raises ValueError otherwise.
    """
    if not isinstance(key, str):
        raise ValueError(f"catalog key must be a string, got {type(key).__name__}")
    if not _SHA256_HEX.fullmatch(key):
        raise ValueError(
            "catalog key must be an opaque sha256 hex digest (64 lowercase"
            f" hex chars), got {key!r} — local paths, URLs, data/base64"
            " payloads, and raw platform media references are not valid"
            " catalog keys"
        )


def validate_sha256(value: str) -> None:
    """Validate a content sha256: a 64-character lowercase hex digest."""
    if not isinstance(value, str):
        raise ValueError(f"sha256 must be a string, got {type(value).__name__}")
    if not _SHA256_HEX.fullmatch(value):
        raise ValueError(
            f"sha256 must be a 64-character lowercase hex digest, got {value!r}"
        )


def validate_asset_id(asset_id: int) -> None:
    """Validate an opaque approved catalog asset id (a positive integer).

    The media send tools accept ONLY this opaque id — never a URL, path,
    platform reference, or base64 payload. Raises ValueError otherwise.
    """
    if isinstance(asset_id, bool) or not isinstance(asset_id, int):
        raise ValueError(
            f"asset id must be an integer, got {type(asset_id).__name__}"
        )
    if asset_id <= 0:
        raise ValueError(f"asset id must be a positive integer, got {asset_id!r}")


def scrub_description(text: str | None) -> str | None:
    """Strip URL-ish / raw-media tokens from a vision description before it
    enters the catalog, so a catalog row can never carry a fetchable source
    through its description. Returns None for empty/whitespace input."""
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = _SOURCE_TOKEN.sub("", text).strip()
    return cleaned or None


def _escape_control(text: str) -> str:
    """Escape control characters as ``\\uXXXX`` so a description can never
    inject prompt/tool structure."""
    return _CONTROL_CHARS.sub(lambda m: f"\\u{ord(m.group(0)):04x}", text)


def normalize_description(text: str | None) -> str | None:
    """Normalize a vision description into a bounded, escaped, one-line
    string safe to render inside a prompt listing.

    - None / non-string / empty input -> None.
    - URL-ish and raw-media tokens are stripped (``scrub_description``).
    - Every ordinary whitespace run collapses to one space.
    - Control characters are escaped as literal ``\\uXXXX`` sequences, so
      newlines cannot create a second prompt/tool line.
    - The result is truncated to ``DESCRIPTION_MAX_CHARS`` characters.

    The catalog stores ONLY descriptions produced by this function, so a
    catalog row can never carry a fetchable source or break the prompt/
    tool/outbox structure through its description.
    """
    cleaned = scrub_description(text)
    if cleaned is None:
        return None
    escaped = _escape_control(cleaned)
    collapsed = _WS_RUN.sub(" ", escaped).strip()
    # Do not cut a literal ``\\uXXXX`` escape in half at the catalog bound.
    if len(collapsed) <= DESCRIPTION_MAX_CHARS:
        return collapsed
    pieces: list[str] = []
    used = 0
    i = 0
    while i < len(collapsed):
        piece = collapsed[i : i + 6] if collapsed.startswith("\\u", i) else collapsed[i]
        if used + len(piece) > DESCRIPTION_MAX_CHARS:
            break
        pieces.append(piece)
        used += len(piece)
        i += len(piece)
    return "".join(pieces).rstrip() or None


def _has_source_token(text: str) -> bool:
    """Return whether ``text`` contains a fetchable/raw-media token.

    Scrubbing is suitable for defensive rendering of legacy rows, but it is
    not suitable for an approval verdict: ``"safe https://..."`` must not
    become an approval merely because the URL was removed.
    """
    return _SOURCE_TOKEN.search(text) is not None


@dataclass(frozen=True)
class VisionResult:
    """A strict structured vision verdict.

    ``safe`` is True ONLY when the vision response carried an explicit
    boolean ``safe: true`` classification AND a valid bounded escaped
    one-line ``description``. ``description`` is the normalized catalog
    description (or None when unapprovable).
    """

    safe: bool
    description: str | None = None


def parse_vision_result(text: str | None) -> VisionResult:
    """Parse a STRICT structured vision result.

    The vision response must be a JSON object carrying a boolean ``safe``
    field and a string ``description`` field. Approval requires ``safe`` to
    be exactly ``true`` AND a non-empty bounded escaped one-line
    description. Anything else — malformed JSON, a missing or wrong-typed
    field, an unsafe (``false``) classification, or an empty/overlong
    description — yields an unapprovable result (``safe=False``). No
    arbitrary text ever approves a candidate.
    """
    if not isinstance(text, str) or not text.strip():
        return VisionResult(safe=False)
    data = None
    try:
        data = json.loads(text.strip())
    except (ValueError, RecursionError):
        # Structured vision output is intentionally strict.  Do not recover
        # JSON from prose or markdown fences: arbitrary text must never be
        # able to look like an approval.
        return VisionResult(safe=False)
    if not isinstance(data, dict):
        return VisionResult(safe=False)
    if data.get("safe") is not True:  # strict: exactly the boolean true
        return VisionResult(safe=False)
    raw_description = data.get("description")
    if not isinstance(raw_description, str) or not raw_description.strip():
        return VisionResult(safe=False)
    # Overlong output is malformed for the bounded catalog field; silently
    # truncating it would turn an invalid provider response into approval.
    if len(raw_description) > DESCRIPTION_MAX_CHARS:
        return VisionResult(safe=False)
    # A source-like token is invalid even when useful prose surrounds it.
    if _has_source_token(raw_description):
        return VisionResult(safe=False)
    description = normalize_description(raw_description)
    if description is None:
        return VisionResult(safe=False)
    # Escaping can expand controls.  The stored representation itself must
    # remain bounded, not just the provider's raw string.
    if len(description) > DESCRIPTION_MAX_CHARS:
        return VisionResult(safe=False)
    return VisionResult(safe=True, description=description)


def validate_candidate(candidate: MediaAssetCandidate) -> None:
    """Validate a candidate before it enters the catalog (fail closed).

    The opaque cache key and the content sha256 must be sha256 hex digests;
    the mime must be non-empty. The boundary type already validates
    kind/dimensions/finiteness; this adds the key-opacity rules the
    repository enforces at submit time.
    """
    validate_catalog_key(candidate.cache_key)
    validate_sha256(candidate.sha256)
    if not candidate.mime or not candidate.mime.strip():
        raise ValueError("mime must be non-empty")
    if candidate.description is not None:
        normalized = normalize_description(candidate.description)
        if normalized is None or normalized != candidate.description:
            raise ValueError(
                "description must be a normalized, escaped one-line value"
            )
