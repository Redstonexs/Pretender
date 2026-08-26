"""Always-on JSONL recorder of normalized inbound events — the replay corpus.

Every event the adapters yield is appended here, one JSON object per line,
before anything else happens (ingest records first, then commits, then
wakes). The corpus is what ``replay`` re-scores in later phases, so the
format is normalized: platform payloads become stable fields, and the raw
payload rides along untouched.

The recorder is synchronous by design: a small append + flush per event is
cheap, and the corpus must survive a crash (flush on every write).

The durable dispatch ledger (frozen Oracle advisory) adds typed
commit/dispatch MARKER lines: ``append_marker`` writes one marker after the
durable state it describes, and ``read_markers`` reads them back
deduplicated by ``(record_type, sequence)`` — first occurrence wins, file
order preserved. Export is at-least-once: append the marker, then mark it
exported; a crash between the two is repaired by the startup export
(``export_unexported``), which re-appends the marker — duplicates after a
crash are tolerated because readers dedupe.

``read_corpus`` / ``Recorder.read_events`` read the corpus back
deterministically for replay: file order is preserved, and malformed or
unparseable lines are skipped — a damaged corpus must never crash replay.
Marker lines are skipped by the event reader (they carry no ``type``).

``read_corpus_view`` is the EXACT replay input and fails closed where
parsing losses would silently omit a settled decision: a JSON line that
carries a ``record_type`` but does not parse as a valid commit/dispatch
marker raises ``ValueError`` (a malformed marker record would otherwise be
dropped and its settled dispatch omitted from replay). Non-JSON lines and
malformed event lines are still skipped (a torn write or a damaged event
never crashes the reader); ``read_markers`` stays lenient for general
corpus inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from pretender.seams import Repository
from pretender.types import (
    AdapterEvent,
    ChatKey,
    CommitSeq,
    CorpusMarker,
    DispatchId,
    EventId,
    Message,
    MessageId,
    MessageRowId,
    Segment,
    SenderId,
)


class Recorder:
    """Append-only JSONL file of normalized events."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def write_event(self, event: AdapterEvent, *, event_id: EventId | None = None) -> None:
        """Normalize and append one event.

        ``event_id`` is the stable event id generated before recording; a
        message event carries it so the corpus line links to its commit
        marker even when a crash breaks their adjacency. Non-message
        events never carry one (they commit nothing).
        """
        self.write(self._normalize(event, event_id=event_id))

    def write(self, payload: dict[str, Any]) -> None:
        """Append one raw JSON object (low-level; used by tests)."""
        if self._fh is None:
            raise RuntimeError("recorder is closed")
        line = orjson.dumps(payload, default=str).decode("utf-8")
        self._fh.write(line + "\n")
        self._fh.flush()

    def append_marker(self, marker: CorpusMarker) -> None:
        """Append one typed commit/dispatch marker line (at-least-once
        export: the marker is written AFTER the durable state it describes;
        the caller marks it exported afterward)."""
        payload: dict[str, Any] = {
            "record_type": marker.record_type,
            "sequence": marker.sequence,
            "chat_key": marker.chat_key,
        }
        if marker.record_type == "commit":
            payload["event_id"] = marker.event_id
            payload["wake_kind"] = marker.wake_kind
            payload["message_row_id"] = marker.message_row_id
            payload["priority"] = marker.priority
        else:
            payload["cause"] = marker.cause
            # The frozen attachment boundary and scheduled time ride on the
            # dispatch marker so replay reconstructs the exact boundary
            # independent of JSONL marker order.
            payload["commit_boundary"] = marker.commit_boundary
            payload["scheduled_for"] = marker.scheduled_for
            # The v4 replayable settled-dispatch contract: the full frozen
            # evaluation metadata — settled state, evaluation timestamp,
            # message boundaries, exact attached membership, and trace.
            payload["state"] = marker.state
            payload["settled_ts"] = marker.settled_ts
            payload["start_msg_id"] = marker.start_msg_id
            payload["through_msg_id"] = marker.through_msg_id
            payload["attached"] = list(marker.attached)
            payload["trace_json"] = marker.trace_json
            payload["evaluated_ts"] = marker.evaluated_ts
            payload["snapshot_json"] = marker.snapshot_json
        self.write(payload)

    def close(self) -> None:
        if self._fh is not None:
            # Exact replay needs a durable corpus-completeness witness. Re-scan
            # leniently so an old/torn corpus never prevents live shutdown;
            # the strict reader will reject any malformed marker later.
            dispatches: dict[str, set[int]] = {}
            for marker in read_markers(self._path):
                if (
                    marker.record_type == "dispatch"
                    and marker.state in ("completed", "released")
                    and marker.settled_ts is not None
                ):
                    dispatches.setdefault(str(marker.chat_key), set()).add(
                        int(marker.sequence)
                    )
            self.write(
                {
                    "record_type": "corpus_manifest",
                    "version": 2,
                    "dispatches": {
                        chat_key: sorted(ids)
                        for chat_key, ids in sorted(dispatches.items())
                    },
                }
            )
            self._fh.close()
            self._fh = None

    def read_events(self) -> list[AdapterEvent]:
        """Read the recorded corpus back deterministically (file order).

        Malformed or unparseable lines are skipped — the corpus must never
        crash replay. The append handle is untouched; reading uses a
        separate read handle.
        """
        return read_corpus(self._path)

    def read_markers(self) -> list[CorpusMarker]:
        """Read the typed commit/dispatch markers back, deduplicated by
        ``(record_type, sequence)`` — first occurrence wins, file order
        preserved. Malformed lines are skipped."""
        return read_markers(self._path)

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── normalization ───────────────────────────────────────────────────────

    def _normalize(
        self, event: AdapterEvent, *, event_id: EventId | None = None
    ) -> dict[str, Any]:
        payload = event.payload
        if event.type == "message" and isinstance(payload, Message):
            return {
                "ts": event.ts if event.ts is not None else payload.recv_ts,
                "type": "message",
                "chat_key": payload.chat_key,
                "sender_id": payload.sender_id,
                "sender_name": payload.sender_name,
                "is_self": payload.is_self,
                "text": payload.text,
                "id": payload.id,
                "reply_to": payload.reply_to,
                "mentions": list(payload.mentions),
                "segments": [
                    {"kind": s.kind, "data": s.data} for s in payload.segments
                ],
                "raw": payload.raw,
                "event_id": event_id,
            }
        return {
            "ts": event.ts,
            "type": event.type,
            "payload": payload,
            "raw": event.raw,
        }


# ── The deterministic corpus reader (replay input) ─────────────────────────

def read_corpus(path: str | Path) -> list[AdapterEvent]:
    """Read a recorded corpus back deterministically, in file order.

    Safe by construction: a missing file yields an empty list, and any
    malformed or unparseable line is skipped (never crashes replay). The
    normalized message shape round-trips exactly: ``ts`` becomes the
    message's ``recv_ts``, mentions/segments become tuples, and the raw
    payload rides along untouched.
    """
    p = Path(path)
    if not p.exists():
        return []
    events: list[AdapterEvent] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = orjson.loads(line)
            except (orjson.JSONDecodeError, ValueError):
                continue
            event = _event_from_record(data)
            if event is not None:
                events.append(event)
    return events


def _event_from_record(data: dict[str, Any]) -> AdapterEvent | None:
    """One normalized record back into an ``AdapterEvent``; None when the
    record is malformed (missing fields, wrong types)."""
    try:
        if data.get("type") == "message":
            payload = Message(
                chat_key=ChatKey(data["chat_key"]),
                sender_id=SenderId(data["sender_id"]),
                sender_name=data["sender_name"],
                is_self=bool(data["is_self"]),
                text=data["text"],
                id=MessageId(data["id"]) if data.get("id") is not None else None,
                reply_to=MessageId(data["reply_to"]) if data.get("reply_to") else None,
                mentions=tuple(SenderId(m) for m in data.get("mentions") or ()),
                segments=tuple(Segment(**s) for s in data.get("segments") or ()),
                recv_ts=data.get("ts"),
                raw=data.get("raw"),
            )
            return AdapterEvent(
                type="message", payload=payload, raw=data.get("raw"), ts=data.get("ts")
            )
        return AdapterEvent(
            type=data["type"],
            payload=data.get("payload"),
            raw=data.get("raw"),
            ts=data.get("ts"),
        )
    except (KeyError, TypeError, ValueError):
        return None


# ── The typed marker reader (dispatch-ledger replay input) ──────────────────

def read_markers(path: str | Path) -> list[CorpusMarker]:
    """Read the typed commit/dispatch markers back, deduplicated by
    ``(record_type, sequence)`` — first occurrence wins, file order
    preserved. Safe by construction: a missing file yields an empty list,
    and any malformed or unparseable line is skipped (never crashes
    replay). Duplicate markers after a crash are tolerated: the at-least-
    once export may re-append a marker whose export mark did not commit.
    """
    p = Path(path)
    if not p.exists():
        return []
    seen: set[tuple[str, int]] = set()
    markers: list[CorpusMarker] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = orjson.loads(line)
            except (orjson.JSONDecodeError, ValueError):
                continue
            marker = _marker_from_record(data)
            if marker is None:
                continue
            key = (marker.record_type, marker.sequence)
            if key in seen:
                continue
            seen.add(key)
            markers.append(marker)
    return markers


def _marker_from_record(data: dict[str, Any]) -> CorpusMarker | None:
    """One marker record back into a ``CorpusMarker``; None when the
    record is malformed or not a marker line (event lines carry no
    ``record_type``)."""
    try:
        record_type = data["record_type"]
        if record_type == "commit":
            return CorpusMarker(
                record_type="commit",
                sequence=CommitSeq(data["sequence"]),
                chat_key=ChatKey(data["chat_key"]),
                event_id=EventId(data["event_id"]),
                wake_kind=data["wake_kind"],
                message_row_id=(
                    MessageRowId(data["message_row_id"])
                    if data.get("message_row_id") is not None
                    else None
                ),
                priority=bool(data.get("priority", False)),
            )
        if record_type == "dispatch":
            # Old v2 markers carry no boundary/scheduled metadata and old
            # v3 markers carry no settled-state metadata: they read back as
            # None/empty and remain fully readable.
            return CorpusMarker(
                record_type="dispatch",
                sequence=DispatchId(data["sequence"]),
                chat_key=ChatKey(data["chat_key"]),
                cause=data["cause"],
                commit_boundary=(
                    CommitSeq(data["commit_boundary"])
                    if data.get("commit_boundary") is not None
                    else None
                ),
                scheduled_for=data.get("scheduled_for"),
                state=data.get("state"),
                settled_ts=data.get("settled_ts"),
                start_msg_id=(
                    MessageRowId(data["start_msg_id"])
                    if data.get("start_msg_id") is not None
                    else None
                ),
                through_msg_id=(
                    MessageRowId(data["through_msg_id"])
                    if data.get("through_msg_id") is not None
                    else None
                ),
                attached=tuple(
                    CommitSeq(s) for s in (data.get("attached") or ())
                ),
                trace_json=data.get("trace_json"),
                evaluated_ts=data.get("evaluated_ts"),
                snapshot_json=data.get("snapshot_json"),
            )
        return None
    except (KeyError, TypeError, ValueError):
        return None


# ── The structured marker-driven replay input (v4 dispatch schedule) ────────

@dataclass(frozen=True)
class CorpusView:
    """The structured marker-driven replay input, read from one corpus.

    ``events_by_event_id`` maps every recorded message event's stable
    ``EventId`` to its normalized ``AdapterEvent`` (raw events without a
    durable commit marker are ignored by replay); ``commits`` are the
    commit markers in ``CommitSeq`` order; ``dispatches`` are the dispatch
    markers in ``DispatchId`` order. Markers may be physically exported in
    another order (a crash can reverse the live writer order), so replay
    reconstructs by their durable sequence/boundary/membership, never by
    JSONL order.

    Only SETTLED dispatches (``completed`` or ``released``) are ever
    exported, so every dispatch marker here is replayable; a v2/v3 corpus
    without the v4 settled metadata reads back with ``state``/``attached``
    None/empty and is NOT exactly replayable (the caller reports it).
    """

    events_by_event_id: dict[EventId, AdapterEvent]
    commits: tuple[CorpusMarker, ...]
    dispatches: tuple[CorpusMarker, ...]
    # The last line written by a clean Recorder close. Exact CLI replay
    # requires it, while library-level replay remains usable for old fixtures.
    manifest_dispatches: dict[ChatKey, frozenset[DispatchId]] | None = None


def read_corpus_view(path: str | Path) -> CorpusView:
    """Read the corpus back as the structured marker-driven replay input.

    Fail-closed for the exact replay contract: a missing file yields an
    empty view, and non-JSON lines plus malformed EVENT lines are skipped
    (a torn write or a damaged event never crashes the reader). A JSON
    line that carries a ``record_type`` but does not parse as a valid
    commit/dispatch marker raises ``ValueError`` — the parsing loss would
    silently omit a settled decision from replay, so exact replay fails
    closed instead. Markers are deduplicated by ``(record_type, sequence)``
    — first occurrence wins. Commit markers are sorted by ``CommitSeq`` and
    dispatch markers by ``DispatchId`` (markers may be physically exported
    in another order; replay reconstructs by their durable sequence, not
    JSONL order).
    """
    p = Path(path)
    events_by_event_id: dict[EventId, AdapterEvent] = {}
    commits: list[CorpusMarker] = []
    dispatches: list[CorpusMarker] = []
    if not p.exists():
        return CorpusView(events_by_event_id, (), ())
    seen: set[tuple[str, int]] = set()
    manifest_dispatches: dict[ChatKey, frozenset[DispatchId]] | None = None
    manifest_line: int | None = None
    last_nonempty_line: int | None = None
    with p.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            last_nonempty_line = line_number
            try:
                data = orjson.loads(line)
            except (orjson.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            if data.get("record_type") == "corpus_manifest":
                raw_dispatches = data.get("dispatches")
                if data.get("version") != 2 or not isinstance(raw_dispatches, dict):
                    raise ValueError("malformed corpus manifest")
                parsed: dict[ChatKey, frozenset[DispatchId]] = {}
                for raw_chat, raw_ids in raw_dispatches.items():
                    if (
                        not isinstance(raw_chat, str)
                        or not isinstance(raw_ids, list)
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or value < 0
                            for value in raw_ids
                        )
                        or len(set(raw_ids)) != len(raw_ids)
                    ):
                        raise ValueError("malformed corpus manifest")
                    parsed[ChatKey(raw_chat)] = frozenset(
                        DispatchId(value) for value in raw_ids
                    )
                manifest_dispatches = parsed
                manifest_line = line_number
                continue
            if "record_type" in data:
                marker = _marker_from_record(data)
                if marker is None:
                    raise ValueError(
                        "malformed marker record"
                        f" (record_type={data.get('record_type')!r})"
                    )
                key = (marker.record_type, marker.sequence)
                if key in seen:
                    continue
                seen.add(key)
                if marker.record_type == "commit":
                    commits.append(marker)
                else:
                    dispatches.append(marker)
                continue
            event_id, event = _event_from_record_with_id(data)
            if event_id is not None and event is not None:
                events_by_event_id[event_id] = event
    commits.sort(key=lambda m: m.sequence)
    dispatches.sort(key=lambda m: m.sequence)
    if manifest_line is not None and manifest_line != last_nonempty_line:
        # A prior clean-close manifest cannot attest to later appended/torn
        # data. The next clean close writes a new final manifest.
        manifest_dispatches = None
    return CorpusView(
        events_by_event_id=events_by_event_id,
        commits=tuple(commits),
        dispatches=tuple(dispatches),
        manifest_dispatches=manifest_dispatches,
    )


def _event_from_record_with_id(
    data: dict[str, Any],
) -> tuple[EventId | None, AdapterEvent | None]:
    """One normalized record back into ``(event_id, AdapterEvent)``.

    ``event_id`` is the stable id the corpus event line carries (None for
    non-message events, malformed records, and events recorded without an
    id — those can never be resolved by a commit marker)."""
    event = _event_from_record(data)
    if event is None:
        return None, None
    event_id = data.get("event_id")
    if event_id is None:
        return None, event
    return EventId(event_id), event


# ── At-least-once export (startup) ──────────────────────────────────────────

async def export_unexported(recorder: Recorder, repo: Repository) -> None:
    """The startup export: append every unexported commit/dispatch marker
    to the corpus, then mark it exported.

    At-least-once by construction: a crash between the append and the mark
    re-exports the marker on the next startup; readers deduplicate by
    ``(record_type, sequence)``, so the duplicate line is harmless. Must
    run before accepting new input so the corpus order stays deterministic.
    """
    for marker in await repo.list_unexported_commits():
        recorder.append_marker(marker)
        await repo.mark_commit_exported(CommitSeq(marker.sequence))
    for marker in await repo.list_unexported_dispatches():
        recorder.append_marker(marker)
        await repo.mark_dispatch_exported(DispatchId(marker.sequence))


async def export_marker(recorder: Recorder, repo: Repository, marker: CorpusMarker) -> None:
    """Append ONE marker, then mark it exported (the live ingest path).

    Same at-least-once contract as ``export_unexported``: a crash between
    the append and the mark is repaired by the startup export, which
    re-appends the marker; readers deduplicate.
    """
    recorder.append_marker(marker)
    if marker.record_type == "commit":
        await repo.mark_commit_exported(CommitSeq(marker.sequence))
    else:
        await repo.mark_dispatch_exported(DispatchId(marker.sequence))
