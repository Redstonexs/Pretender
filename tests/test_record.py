"""Recorder: always-on normalized JSONL — format, append, close semantics."""

from __future__ import annotations

import json

import pytest

from pretender.record import Recorder, read_corpus_view, read_markers
from pretender.types import (
    AdapterEvent,
    ChatKey,
    CommitSeq,
    CorpusMarker,
    DispatchCause,
    DispatchId,
    EventId,
    Message,
    MessageRowId,
    Segment,
    WakeKind,
)
from tests.durable_helpers import CK, make_message


def read_lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_recorder_writes_normalized_message_event(tmp_path):
    path = tmp_path / "events.jsonl"
    msg = make_message(
        text="你好",
        segments=(Segment("text", {"text": "你好"}),),
        reply_to="m0",
    )
    event = AdapterEvent(type="message", payload=msg, ts=1_700_000_001.0)
    with Recorder(path) as rec:
        rec.write_event(event)

    lines = read_lines(path)
    assert len(lines) == 2
    line = lines[0]
    assert line["ts"] == 1_700_000_001.0
    assert line["type"] == "message"
    assert line["chat_key"] == "qq:group:123456"
    assert line["sender_id"] == "u1"
    assert line["sender_name"] == "user"
    assert line["is_self"] is False
    assert line["text"] == "你好"
    assert line["id"] == "m1"
    assert line["reply_to"] == "m0"
    assert line["mentions"] == []
    assert line["segments"] == [{"kind": "text", "data": {"text": "你好"}}]
    assert lines[-1] == {
        "record_type": "corpus_manifest",
        "version": 2,
        "dispatches": {},
    }


def test_recorder_uses_event_ts_fallback(tmp_path):
    path = tmp_path / "events.jsonl"
    msg = make_message(recv_ts=1_700_000_002.0)
    event = AdapterEvent(type="message", payload=msg, ts=None)
    with Recorder(path) as rec:
        rec.write_event(event)
    assert read_lines(path)[0]["ts"] == 1_700_000_002.0


def test_recorder_handles_non_message_events(tmp_path):
    path = tmp_path / "events.jsonl"
    event = AdapterEvent(type="notice", payload={"kind": "poke"}, raw={"x": 1})
    with Recorder(path) as rec:
        rec.write_event(event)
    line = read_lines(path)[0]
    assert line["type"] == "notice"
    assert line["payload"] == {"kind": "poke"}


def test_recorder_appends_across_instances(tmp_path):
    path = tmp_path / "events.jsonl"
    with Recorder(path) as rec:
        rec.write_event(AdapterEvent(type="message", payload=make_message(msg_id="a")))
    with Recorder(path) as rec:
        rec.write_event(AdapterEvent(type="message", payload=make_message(msg_id="b")))
    lines = read_lines(path)
    assert [l["id"] for l in lines if l.get("type") == "message"] == ["a", "b"]
    assert lines[-1]["record_type"] == "corpus_manifest"


def test_corpus_view_discards_stale_manifest_after_later_data(tmp_path):
    """An earlier clean-close manifest cannot attest to a later append/torn
    write; exact CLI replay will require the next final manifest."""
    path = tmp_path / "events.jsonl"
    with Recorder(path) as rec:
        rec.write_event(AdapterEvent(type="message", payload=make_message()))
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"type":"notice"}\n')
    assert read_corpus_view(path).manifest_dispatches is None


def test_recorder_serializes_non_json_payloads(tmp_path):
    path = tmp_path / "events.jsonl"

    class Opaque:
        def __str__(self) -> str:
            return "opaque"

    with Recorder(path) as rec:
        rec.write({"ts": 1.0, "raw": Opaque()})
    assert read_lines(path)[0]["raw"] == "opaque"


def test_recorder_write_after_close_raises(tmp_path):
    path = tmp_path / "events.jsonl"
    rec = Recorder(path)
    rec.close()
    rec.close()  # idempotent
    with pytest.raises(RuntimeError):
        rec.write({"ts": 1.0})


def test_recorder_creates_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "events.jsonl"
    with Recorder(path) as rec:
        rec.write({"ts": 1.0})
    assert path.exists()


# ── typed dispatch markers: frozen boundary and scheduled time ──────────────

def test_dispatch_marker_round_trips_boundary_and_scheduled_for(tmp_path):
    """A dispatch marker carries the frozen commit_boundary and
    scheduled_for through append -> read: replay has the exact attachment
    boundary independent of JSONL marker order."""
    path = tmp_path / "events.jsonl"
    marker = CorpusMarker(
        record_type="dispatch",
        sequence=DispatchId(7),
        chat_key=CK,
        cause=DispatchCause.TIMER,
        commit_boundary=CommitSeq(3),
        scheduled_for=200.0,
    )
    with Recorder(path) as rec:
        rec.append_marker(marker)
    (read,) = read_markers(path)
    assert read.record_type == "dispatch"
    assert read.sequence == 7
    assert read.chat_key == CK
    assert read.cause == DispatchCause.TIMER
    assert read.commit_boundary == CommitSeq(3)
    assert read.scheduled_for == 200.0


def test_dispatch_marker_round_trips_nullable_metadata(tmp_path):
    """A non-timer dispatch marker round-trips with boundary 0 and no
    scheduled time."""
    path = tmp_path / "events.jsonl"
    marker = CorpusMarker(
        record_type="dispatch",
        sequence=DispatchId(1),
        chat_key=CK,
        cause=DispatchCause.STARTUP,
        commit_boundary=CommitSeq(0),
        scheduled_for=None,
    )
    with Recorder(path) as rec:
        rec.append_marker(marker)
    (read,) = read_markers(path)
    assert read.commit_boundary == CommitSeq(0)
    assert read.scheduled_for is None


def test_old_dispatch_marker_without_metadata_reads_back(tmp_path):
    """Backward compatibility: a v2 dispatch marker line (no
    commit_boundary / scheduled_for fields) reads back with None metadata —
    old corpora stay fully readable."""
    path = tmp_path / "events.jsonl"
    with Recorder(path) as rec:
        rec.write(
            {
                "record_type": "dispatch",
                "sequence": 5,
                "chat_key": CK,
                "cause": DispatchCause.INBOUND,
            }
        )
    (read,) = read_markers(path)
    assert read.sequence == 5
    assert read.cause == DispatchCause.INBOUND
    assert read.commit_boundary is None
    assert read.scheduled_for is None


def test_commit_marker_round_trip_unchanged(tmp_path):
    """Commit markers are unaffected by the dispatch metadata: they carry
    event_id/wake_kind and no boundary/scheduled fields."""
    path = tmp_path / "events.jsonl"
    marker = CorpusMarker(
        record_type="commit",
        sequence=CommitSeq(1),
        chat_key=CK,
        event_id=EventId("ev-1"),
        wake_kind=WakeKind.INBOUND,
    )
    with Recorder(path) as rec:
        rec.append_marker(marker)
    (read,) = read_markers(path)
    assert read.record_type == "commit"
    assert read.event_id == EventId("ev-1")
    assert read.wake_kind == WakeKind.INBOUND
    assert read.commit_boundary is None
    assert read.scheduled_for is None


def test_v4_dispatch_marker_round_trips_full_settled_metadata(tmp_path):
    """A v4 dispatch marker round-trips the FULL frozen evaluation
    metadata: settled state, evaluation timestamp, message boundaries,
    exact attached membership, and trace — replay reconstructs the exact
    live dispatch from the marker alone."""
    from pretender.types import MessageRowId

    path = tmp_path / "events.jsonl"
    marker = CorpusMarker(
        record_type="dispatch",
        sequence=DispatchId(9),
        chat_key=CK,
        cause=DispatchCause.INBOUND,
        commit_boundary=CommitSeq(3),
        scheduled_for=None,
        state="completed",
        settled_ts=150.0,
        start_msg_id=MessageRowId(0),
        through_msg_id=MessageRowId(2),
        attached=(CommitSeq(1), CommitSeq(2)),
        trace_json='{"t":1}',
    )
    with Recorder(path) as rec:
        rec.append_marker(marker)
    (read,) = read_markers(path)
    assert read.record_type == "dispatch"
    assert read.sequence == 9
    assert read.cause == DispatchCause.INBOUND
    assert read.commit_boundary == CommitSeq(3)
    assert read.scheduled_for is None
    assert read.state == "completed"
    assert read.settled_ts == 150.0
    assert read.start_msg_id == MessageRowId(0)
    assert read.through_msg_id == MessageRowId(2)
    assert read.attached == (CommitSeq(1), CommitSeq(2))
    assert read.trace_json == '{"t":1}'


def test_v3_dispatch_marker_without_settled_fields_reads_back(tmp_path):
    """Backward compatibility: a v3 dispatch marker (boundary/scheduled but
    no settled-state fields) reads back with None/empty settled metadata —
    old corpora stay fully readable."""
    path = tmp_path / "events.jsonl"
    with Recorder(path) as rec:
        rec.write(
            {
                "record_type": "dispatch",
                "sequence": 6,
                "chat_key": CK,
                "cause": DispatchCause.TIMER,
                "commit_boundary": 3,
                "scheduled_for": 200.0,
            }
        )
    (read,) = read_markers(path)
    assert read.sequence == 6
    assert read.cause == DispatchCause.TIMER
    assert read.commit_boundary == CommitSeq(3)
    assert read.scheduled_for == 200.0
    assert read.state is None
    assert read.settled_ts is None
    assert read.start_msg_id is None
    assert read.through_msg_id is None
    assert read.attached == ()
    assert read.trace_json is None


# ── the structured marker-driven replay view (v4 dispatch schedule) ─────────

def test_read_corpus_view_keys_events_by_event_id(tmp_path):
    """The view keys raw message events by their stable EventId and carries
    the commit/dispatch markers sorted by their durable sequence."""
    path = tmp_path / "events.jsonl"
    with Recorder(path) as rec:
        rec.write_event(
            AdapterEvent(
                type="message",
                payload=make_message(msg_id="m1", text="hi"),
                ts=1_700_000_000.0,
            ),
            event_id=EventId("ev-1"),
        )
        rec.append_marker(
            CorpusMarker(
                record_type="commit", sequence=CommitSeq(1), chat_key=CK,
                event_id=EventId("ev-1"), wake_kind=WakeKind.INBOUND,
            )
        )
        rec.append_marker(
            CorpusMarker(
                record_type="dispatch", sequence=DispatchId(1), chat_key=CK,
                cause=DispatchCause.INBOUND, commit_boundary=CommitSeq(1),
                scheduled_for=None, state="completed", settled_ts=150.0,
                start_msg_id=MessageRowId(0), through_msg_id=MessageRowId(1),
                attached=(CommitSeq(1),), trace_json=None,
            )
        )
    view = read_corpus_view(path)
    assert set(view.events_by_event_id) == {EventId("ev-1")}
    assert view.events_by_event_id[EventId("ev-1")].payload.text == "hi"
    assert [(m.record_type, m.sequence) for m in view.commits] == [("commit", 1)]
    assert [(m.record_type, m.sequence) for m in view.dispatches] == [
        ("dispatch", 1)
    ]
    assert view.dispatches[0].attached == (CommitSeq(1),)


def test_read_corpus_view_sorts_markers_by_durable_sequence(tmp_path):
    """Markers may be physically exported in another order (a crash can
    reverse the live writer order): the view sorts commits by CommitSeq and
    dispatches by DispatchId, never by JSONL order."""
    path = tmp_path / "events.jsonl"
    with Recorder(path) as rec:
        # Reversed physical order: dispatch 2, commit 2, dispatch 1, commit 1.
        rec.append_marker(
            CorpusMarker(
                record_type="dispatch", sequence=DispatchId(2), chat_key=CK,
                cause=DispatchCause.INBOUND, commit_boundary=CommitSeq(2),
                scheduled_for=None, state="released", settled_ts=160.0,
                start_msg_id=MessageRowId(0), through_msg_id=MessageRowId(2),
                attached=(CommitSeq(1), CommitSeq(2)), trace_json=None,
            )
        )
        rec.append_marker(
            CorpusMarker(
                record_type="commit", sequence=CommitSeq(2), chat_key=CK,
                event_id=EventId("ev-2"), wake_kind=WakeKind.INBOUND,
            )
        )
        rec.append_marker(
            CorpusMarker(
                record_type="dispatch", sequence=DispatchId(1), chat_key=CK,
                cause=DispatchCause.INBOUND, commit_boundary=CommitSeq(1),
                scheduled_for=None, state="completed", settled_ts=150.0,
                start_msg_id=MessageRowId(0), through_msg_id=MessageRowId(1),
                attached=(CommitSeq(1),), trace_json=None,
            )
        )
        rec.append_marker(
            CorpusMarker(
                record_type="commit", sequence=CommitSeq(1), chat_key=CK,
                event_id=EventId("ev-1"), wake_kind=WakeKind.INBOUND,
            )
        )
    view = read_corpus_view(path)
    assert [m.sequence for m in view.commits] == [1, 2]
    assert [m.sequence for m in view.dispatches] == [1, 2]


def test_read_corpus_view_dedupes_markers(tmp_path):
    """Duplicate markers (a crash re-append) are deduplicated by
    (record_type, sequence) — first occurrence wins."""
    path = tmp_path / "events.jsonl"
    with Recorder(path) as rec:
        rec.append_marker(
            CorpusMarker(
                record_type="commit", sequence=CommitSeq(1), chat_key=CK,
                event_id=EventId("ev-1"), wake_kind=WakeKind.INBOUND,
            )
        )
        rec.append_marker(
            CorpusMarker(
                record_type="commit", sequence=CommitSeq(1), chat_key=CK,
                event_id=EventId("ev-1"), wake_kind=WakeKind.INBOUND,
            )
        )  # crash duplicate
        rec.append_marker(
            CorpusMarker(
                record_type="dispatch", sequence=DispatchId(1), chat_key=CK,
                cause=DispatchCause.INBOUND, commit_boundary=CommitSeq(1),
                scheduled_for=None, state="completed", settled_ts=150.0,
                start_msg_id=MessageRowId(0), through_msg_id=MessageRowId(1),
                attached=(CommitSeq(1),), trace_json=None,
            )
        )
        rec.append_marker(
            CorpusMarker(
                record_type="dispatch", sequence=DispatchId(1), chat_key=CK,
                cause=DispatchCause.INBOUND, commit_boundary=CommitSeq(1),
                scheduled_for=None, state="completed", settled_ts=150.0,
                start_msg_id=MessageRowId(0), through_msg_id=MessageRowId(1),
                attached=(CommitSeq(1),), trace_json=None,
            )
        )  # crash duplicate
    view = read_corpus_view(path)
    assert len(view.commits) == 1
    assert len(view.dispatches) == 1


def test_read_corpus_view_missing_file_empty(tmp_path):
    view = read_corpus_view(tmp_path / "nope.jsonl")
    assert view.events_by_event_id == {}
    assert view.commits == ()
    assert view.dispatches == ()


def test_read_corpus_view_skips_malformed_lines(tmp_path):
    """Malformed or unparseable lines are skipped — a damaged corpus never
    crashes the view reader."""
    path = tmp_path / "events.jsonl"
    path.write_text(
        "not json at all\n"
        '{"record_type": "commit", "sequence": 1, "chat_key": "qq:group:123456",'
        ' "event_id": "ev-1", "wake_kind": "inbound"}\n'
        '{"ts": 1.0, "type": "message", "chat_key": "qq:group:123456",'
        ' "sender_id": "u1", "sender_name": "user", "is_self": false,'
        ' "text": "hi", "id": "m1", "reply_to": null, "mentions": [],'
        ' "segments": [], "raw": null, "event_id": "ev-1"}\n',
        encoding="utf-8",
    )
    view = read_corpus_view(path)
    assert len(view.commits) == 1
    assert set(view.events_by_event_id) == {EventId("ev-1")}


def test_read_corpus_view_fails_closed_on_malformed_dispatch_marker(tmp_path):
    """A JSON line carrying ``record_type: "dispatch"`` that does not parse
    as a valid dispatch marker is a parsing loss of a settled decision:
    the exact replay input fails closed instead of silently dropping it."""
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"record_type": "dispatch", "chat_key": "qq:group:123456",'
        ' "cause": "inbound"}\n',  # missing sequence
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed marker record"):
        read_corpus_view(path)


def test_read_corpus_view_fails_closed_on_invalid_dispatch_marker(tmp_path):
    """A dispatch marker with an invalid settled state is an invalid marker
    record: the exact replay input fails closed."""
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"record_type": "dispatch", "sequence": 1,'
        ' "chat_key": "qq:group:123456", "cause": "inbound",'
        ' "state": "garbage"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed marker record"):
        read_corpus_view(path)


def test_read_corpus_view_fails_closed_on_malformed_commit_marker(tmp_path):
    """A JSON line carrying ``record_type: "commit"`` that does not parse as
    a valid commit marker is equally a marker parsing loss: fail closed."""
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"record_type": "commit", "sequence": 1,'
        ' "chat_key": "qq:group:123456"}\n',  # missing event_id
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed marker record"):
        read_corpus_view(path)


def test_read_corpus_view_ignores_events_without_event_id(tmp_path):
    """A message event recorded without an event_id can never be resolved
    by a commit marker, so it is not keyed in the view."""
    path = tmp_path / "events.jsonl"
    with Recorder(path) as rec:
        rec.write_event(
            AdapterEvent(
                type="message",
                payload=make_message(msg_id="m1", text="hi"),
                ts=1_700_000_000.0,
            )
        )  # no event_id
    view = read_corpus_view(path)
    assert view.events_by_event_id == {}
