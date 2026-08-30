"""CLI: init / run / db / replay — bootstrap smoke tests against a temp
config, plus the SQL-location invariant (the CLI contains no SQL text)."""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import re
import signal
import sqlite3
import sys
from pathlib import Path

import pytest

import pretender.__main__ as module_entry
from pretender.cli import _run_app_with_sigterm, main
from pretender.config import Config
from pretender.record import Recorder
from pretender.gate import Gate
from pretender.cycle import _composition_fingerprint
from pretender.signals import is_other_assistant_target
from pretender.types import (
    AdapterEvent,
    ChatKey,
    CommitSeq,
    CorpusMarker,
    DispatchCause,
    DispatchId,
    EventId,
    MessageRowId,
    WakeKind,
)
from tests.durable_helpers import make_message


def write_config(tmp_path, db_name: str = "cli.db") -> str:
    path = tmp_path / "cfg.toml"
    path.write_text(
        f'[storage]\ndb_path = "{tmp_path / db_name}"\n', encoding="utf-8"
    )
    return str(path)


def write_agent_config(tmp_path, db_name: str = "cli.db") -> str:
    """A config with the planner/reply LLM profiles the live agent build
    requires (no secrets; the offline tests never trigger an LLM call)."""
    path = tmp_path / "cfg.toml"
    path.write_text(
        f'[storage]\ndb_path = "{tmp_path / db_name}"\n'
        "[llm.profiles.planner]\nmodel = \"deepseek-chat\"\n"
        "[llm.profiles.reply]\nmodel = \"deepseek-chat\"\n",
        encoding="utf-8",
    )
    return str(path)


def test_module_entry_propagates_cli_exit_code(monkeypatch):
    """`python -m pretender` must not turn a failing doctor/CLI result into
    a success exit code."""
    import pretender.cli as cli_module

    monkeypatch.setattr(cli_module, "main", lambda argv=None: 7)
    with pytest.raises(SystemExit) as exc:
        module_entry.main()
    assert exc.value.code == 7


def seed_chat(
    tmp_path,
    db_name: str = "cli.db",
    chat_key: str = "qq:group:123456",
    self_id: str = "bot-1",
) -> None:
    """Upsert the durable chat identity replay loads (the CLI replay
    requires it — identity never comes from the corpus)."""
    from tests.durable_helpers import make_identity, open_repo, run

    async def seed():
        _db, repo = await open_repo(tmp_path / db_name)
        await repo.upsert_chat(make_identity(chat_key=chat_key, self_id=self_id))
        await repo.close()

    run(seed())


def seed_settled_dispatch_witness(
    tmp_path, chat_key: str, schedule, settled_ts_base: float
) -> None:
    """Seed only the durable replay witness for synthetic corpus fixtures."""
    conn = sqlite3.connect(tmp_path / "cli.db")
    try:
        for i, (attached, state) in enumerate(schedule, start=1):
            through = max(attached) if attached else 0
            settled_ts = settled_ts_base + i
            conn.execute(
                "INSERT INTO dispatches("
                "id, chat_key, cause, wake_kind, scheduled_ts, started_ts,"
                "expires_at, claimed_ts, cycle_id, start_msg_id, through_msg_id,"
                "state, exported, commit_boundary, attached_json, settled_ts,"
                "evaluated_ts, snapshot_json)"
                " VALUES (?, ?, 'inbound', 'inbound', NULL, ?, ?, ?, ?, ?, ?,"
                " ?, 1, ?, ?, ?, ?, NULL)",
                (
                    i, chat_key, settled_ts, settled_ts + 60.0, settled_ts,
                    f"dispatch:{i}", 0, through, state, through,
                    json.dumps(list(attached)), settled_ts, settled_ts,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def write_ledger(
    tmp_path,
    messages,
    *,
    chat_key: str = "qq:group:123456",
    schedule=None,
    settled_ts_base: float = 1_700_000_000.0,
) -> Path:
    """Write a complete v5 dispatch ledger: event lines + commit markers +
    settled dispatch markers, so the CLI's marker-driven replay has a
    complete ledger to re-score.

    ``schedule`` is a list of ``(attached, state)`` tuples, one per
    dispatch in DispatchId order, where ``attached`` is a tuple of
    CommitSeq ints. Defaults to one completed dispatch per message
    attaching just that message's commit."""
    corpus = Path(tmp_path / "cli.db").with_suffix(".jsonl")
    if schedule is None:
        schedule = [((i,), "completed") for i in range(1, len(messages) + 1)]
    with Recorder(corpus) as rec:
        for i, msg in enumerate(messages, start=1):
            event_id = EventId(f"ev-{i}")
            rec.write_event(
                AdapterEvent(type="message", payload=msg, ts=msg.recv_ts),
                event_id=event_id,
            )
            rec.append_marker(
                CorpusMarker(
                    record_type="commit",
                    sequence=CommitSeq(i),
                    chat_key=ChatKey(chat_key),
                    event_id=event_id,
                    wake_kind=WakeKind.INBOUND,
                    message_row_id=MessageRowId(i),
                    priority=False,
                )
            )
        for i, (attached, state) in enumerate(schedule, start=1):
            through = max(attached) if attached else 0
            pending_messages = [
                dataclasses.asdict(
                    dataclasses.replace(messages[seq - 1], row_id=MessageRowId(seq))
                )
                for seq in attached
            ]
            recent_messages = pending_messages[:]
            evaluated_ts = settled_ts_base + i
            snapshot = {
                "chat_key": chat_key,
                "cycle_id": f"dispatch:{i}",
                "start_msg_id": 0,
                "through_msg_id": through,
                "evaluated_ts": evaluated_ts,
                "self_id": "bot-1",
                "mode": "reply_necessity",
                "threshold": 8,
                "trigger_score": 80,
                "frequency": 1.0,
                "pending": len(pending_messages),
                "pending_messages": pending_messages,
                "recent": recent_messages,
                "window_count": len(recent_messages),
                "self_count": 0,
                "last_nonself_ts": messages[attached[-1] - 1].recv_ts if attached else None,
                "idle_seconds": 0.0,
                "recent_average_interval": 0.0,
                "self_ratio": 0.0,
                "is_group": True,
                "is_focused": False,
                "last_message": pending_messages[-1] if pending_messages else None,
                "self_name": "麦麦",
                "has_direct_at": any(
                    "bot-1" in messages[seq - 1].mentions for seq in attached
                ),
                "has_quote_to_self": False,
                "has_other_assistant": any(
                    is_other_assistant_target(messages[seq - 1].text)
                    for seq in attached
                ),
                "hold_until": None,
                "idle_streak": 0,
                "previous_end_reason": None,
                "backoff_base_s": 15.0,
                "backoff_cap_s": 300.0,
                "backoff_start_count": 2,
            }
            rec.append_marker(
                CorpusMarker(
                    record_type="dispatch",
                    sequence=DispatchId(i),
                    chat_key=ChatKey(chat_key),
                    cause=DispatchCause.INBOUND,
                    commit_boundary=CommitSeq(through),
                    scheduled_for=None,
                    state=state,
                    settled_ts=evaluated_ts,
                    start_msg_id=MessageRowId(0),
                    through_msg_id=MessageRowId(through),
                    attached=tuple(CommitSeq(s) for s in attached),
                    trace_json=json.dumps(
                        {"config": _composition_fingerprint(Gate())}
                    ),
                    evaluated_ts=evaluated_ts,
                    snapshot_json=json.dumps(snapshot, default=str),
                )
            )
    # Exact CLI replay compares the scoped corpus manifest to its durable
    # settled-dispatch witness. Fixtures seed that witness directly; the
    # production path creates it through begin/settle_dispatch.
    seed_settled_dispatch_witness(tmp_path, chat_key, schedule, settled_ts_base)
    return corpus


# ── init ────────────────────────────────────────────────────────────────────

def test_init_creates_database(tmp_path, capsys):
    cfg_path = write_config(tmp_path)
    assert main(["init", "--config", cfg_path]) == 0
    out = capsys.readouterr().out
    assert "initialized" in out
    assert "schema v15" in out
    assert (tmp_path / "cli.db").exists()


def test_init_without_config_uses_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    assert (tmp_path / "data" / "pretender.db").exists()


# ── db ──────────────────────────────────────────────────────────────────────

def test_db_stats_prints_counts(tmp_path, capsys):
    cfg_path = write_config(tmp_path)
    main(["init", "--config", cfg_path])
    assert main(["db", "--config", cfg_path]) == 0
    out = capsys.readouterr().out
    for key in ("messages=0", "outbox_pending=0", "user_version=15", "cycles=0"):
        assert key in out


def test_db_stats_reflects_written_data(tmp_path, capsys):
    cfg_path = write_config(tmp_path)
    main(["init", "--config", cfg_path])
    # Insert a message directly through the storage layer.
    from tests.durable_helpers import make_identity, make_message, open_repo, run

    async def seed():
        _db, repo = await open_repo(tmp_path / "cli.db")
        await repo.upsert_chat(make_identity())
        await repo.ingest_message(make_identity(), make_message())
        await repo.close()

    run(seed())
    main(["db", "--config", cfg_path])
    out = capsys.readouterr().out
    assert "messages=1" in out


# ── run ─────────────────────────────────────────────────────────────────────

def test_run_console_roundtrip(tmp_path, monkeypatch, capsys):
    """A piped line is ingested durably; the run loop exits cleanly on EOF."""
    cfg_path = write_config(tmp_path)
    main(["init", "--config", cfg_path])
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello console\n"))
    assert main(["run", "--dry-run", "--config", cfg_path]) == 0
    # The line was recorded and committed: db stats show one message.
    main(["db", "--config", cfg_path])
    out = capsys.readouterr().out
    assert "messages=1" in out


# ── errors ──────────────────────────────────────────────────────────────────

def test_missing_config_file_exits_1(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["init", "--config", str(tmp_path / "nope.toml")])
    assert exc.value.code == 1
    assert "error" in capsys.readouterr().err


def test_unknown_command_exits_2():
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"])
    assert exc.value.code == 2  # argparse usage error


def test_config_load_roundtrip(tmp_path):
    cfg_path = write_config(tmp_path)
    cfg = Config.load(cfg_path)
    assert cfg.storage.db_path == str(tmp_path / "cli.db")


# ── Phase 2 dry-run / replay ────────────────────────────────────────────────

def test_run_live_requires_profiles(tmp_path, monkeypatch, capsys):
    """``run --live`` is LIVE and requires the planner/reply LLM profiles; a
    config without them is rejected clearly."""
    cfg_path = write_config(tmp_path)
    main(["init", "--config", cfg_path])
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello console\n"))
    with pytest.raises(SystemExit) as exc:
        main(["run", "--live", "--config", cfg_path])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "planner" in err and "reply" in err


def test_run_live_with_profiles(tmp_path, monkeypatch, capsys):
    """``run --live`` with the planner/reply profiles boots a LIVE app
    (dry_run=False) and exits cleanly on EOF with no LLM calls (no inbound
    trigger)."""
    cfg_path = write_agent_config(tmp_path)
    main(["init", "--config", cfg_path])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main(["run", "--live", "--config", cfg_path]) == 0
    out = capsys.readouterr().out
    assert "dry-run" not in out


def test_run_default_is_dry_run(tmp_path, monkeypatch, capsys):
    """Plain ``run`` (no ``--live``) is DRY-RUN by default: no accidental
    live send. It evaluates and prints the trace without requiring the LLM
    profiles."""
    cfg_path = write_config(tmp_path)
    main(["init", "--config", cfg_path])
    monkeypatch.setattr(sys, "stdin", io.StringIO("DeepSeek，你好\n"))
    assert main(["run", "--config", cfg_path]) == 0
    out = capsys.readouterr().out
    assert '"action": "skip"' in out  # a dry-run trace, not a live send


def test_run_live_and_dry_run_are_mutually_exclusive(tmp_path, monkeypatch, capsys):
    """``--live`` and ``--dry-run`` cannot be combined: an ambiguous flag
    combination is rejected by argparse (exit 2), so it can never live-send."""
    cfg_path = write_config(tmp_path)
    main(["init", "--config", cfg_path])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    with pytest.raises(SystemExit) as exc:
        main(["run", "--live", "--dry-run", "--config", cfg_path])
    assert exc.value.code == 2  # argparse usage error
    err = capsys.readouterr().err
    assert "not allowed with" in err


class _CoordinatorProbeApp:
    def __init__(self) -> None:
        self.run_started = asyncio.Event()
        self.run_cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.shutdown_finished = asyncio.Event()
        self.shutdown_calls = 0
        self.order: list[str] = []

    async def run(self) -> None:
        self.run_started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.run_cancelled.set()
            raise
        finally:
            self.order.append("run-finally")

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.order.append("shutdown")
        self.shutdown_finished.set()


def _patch_sigterm(monkeypatch, *, previous=None, invoke_on_install=False):
    """Fake only SIGTERM; leave asyncio.run's SIGINT setup untouched."""
    import pretender.cli as cli_module

    if previous is None:
        previous = object()
    real_signal = cli_module.signal.signal
    calls = []
    handlers = []

    def fake_signal(signum, handler):
        if signum != signal.SIGTERM:
            return real_signal(signum, handler)
        calls.append((signum, handler))
        if not handlers:
            handlers.append(handler)
            if invoke_on_install:
                handler(signum, None)
        return previous

    monkeypatch.setattr(cli_module.signal, "signal", fake_signal)
    return previous, calls, handlers


def test_run_coordinator_natural_completion(monkeypatch):
    app = _CoordinatorProbeApp()
    app.release.set()
    _patch_sigterm(monkeypatch)

    asyncio.run(_run_app_with_sigterm(app))

    assert app.order == ["run-finally", "shutdown"]
    assert app.shutdown_calls == 1


def test_run_coordinator_sigterm_cancels_then_shuts_down(monkeypatch):
    app = _CoordinatorProbeApp()
    _previous, _calls, handlers = _patch_sigterm(monkeypatch)

    async def scenario():
        coordinator = asyncio.create_task(_run_app_with_sigterm(app))
        await asyncio.wait_for(app.run_started.wait(), 1)
        handlers[0](signal.SIGTERM, None)
        await asyncio.wait_for(coordinator, 1)

    asyncio.run(scenario())

    assert app.run_cancelled.is_set()
    assert app.order == ["run-finally", "shutdown"]
    assert app.shutdown_calls == 1


def test_run_coordinator_sigterm_before_run_task_starts(monkeypatch):
    app = _CoordinatorProbeApp()
    _patch_sigterm(monkeypatch, invoke_on_install=True)

    asyncio.run(_run_app_with_sigterm(app))

    assert not app.run_started.is_set()
    assert app.shutdown_calls == 1


def test_run_coordinator_restores_previous_sigterm_handler(monkeypatch):
    app = _CoordinatorProbeApp()
    app.release.set()
    previous, calls, _handlers = _patch_sigterm(monkeypatch)

    asyncio.run(_run_app_with_sigterm(app))

    assert calls[0][0] == signal.SIGTERM
    assert callable(calls[0][1])
    assert calls[-1] == (signal.SIGTERM, previous)


def test_run_coordinator_propagates_external_cancellation(monkeypatch):
    app = _CoordinatorProbeApp()
    _patch_sigterm(monkeypatch)

    async def scenario():
        coordinator = asyncio.create_task(_run_app_with_sigterm(app))
        await asyncio.wait_for(app.run_started.wait(), 1)
        coordinator.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(coordinator, 1)

    asyncio.run(scenario())

    assert app.run_cancelled.is_set()
    assert app.order == ["run-finally", "shutdown"]
    assert app.shutdown_calls == 1


def test_run_dry_run_rejects_config_selected_onebot(tmp_path, monkeypatch, capsys):
    """Dry-run (default or ``--dry-run``) with a CONFIG-SELECTED OneBot is
    rejected: dry-run must remain console-only and never connect to or
    consume OneBot traffic."""
    cfg_path = tmp_path / "onebot.toml"
    cfg_path.write_text(
        f'[storage]\ndb_path = "{tmp_path / "cli.db"}"\n'
        '[adapter]\nname = "onebot"\n',
        encoding="utf-8",
    )
    main(["init", "--config", str(cfg_path)])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    with pytest.raises(SystemExit) as exc:
        main(["run", "--dry-run", "--config", str(cfg_path)])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "console" in err and "onebot" in err


def test_run_dry_run_prints_full_decision_trace(tmp_path, monkeypatch, capsys):
    """``run --dry-run`` prints the full DecisionTrace per decision as one
    JSON line — here a refusal skip."""
    cfg_path = write_config(tmp_path)
    main(["init", "--config", cfg_path])
    monkeypatch.setattr(sys, "stdin", io.StringIO("DeepSeek，你好\n"))
    assert main(["run", "--dry-run", "--config", cfg_path]) == 0
    out = capsys.readouterr().out
    assert '"action": "skip"' in out
    assert '"reason": "refusal"' in out
    assert '"chat_key": "console:group:demo"' in out
    assert '"snapshot_facts"' in out  # full trace, not a summary


def test_replay_prints_traces_and_summary(tmp_path, capsys):
    cfg_path = write_config(tmp_path)
    main(["init", "--config", cfg_path])
    seed_chat(tmp_path)
    write_ledger(
        tmp_path,
        [make_message(chat_key="qq:group:123456", text="DeepSeek，你好")],
    )
    assert main(["replay", "qq:group:123456", "--config", cfg_path]) == 0
    out = capsys.readouterr().out
    assert '"reason": "refusal"' in out
    assert "replay qq:group:123456: 1 decisions, 0 would-have-spoken" in out


def test_replay_sweep_prints_counts(tmp_path, capsys):
    cfg_path = write_config(tmp_path)
    main(["init", "--config", cfg_path])
    seed_chat(tmp_path)
    # A recorded schedule where every dispatch attaches ALL accumulated
    # commits (delay dispatches release the claim, so commits stay
    # pending): the sweep re-scores the FIXED schedule under each
    # threshold x trigger_score combination.
    messages = [
        make_message(
            chat_key="qq:group:123456",
            text="x",
            msg_id=f"m{i}",
            recv_ts=1_700_000_000.0 + i,
        )
        for i in range(10)
    ]
    schedule = [
        (tuple(range(1, i + 1)), "released") for i in range(1, 11)
    ]
    write_ledger(tmp_path, messages, schedule=schedule)
    assert main(["replay", "qq:group:123456", "--sweep", "--config", cfg_path]) == 0
    out = capsys.readouterr().out
    assert "threshold=2 trigger_score=40: would_have_spoken=" in out
    assert "threshold=12 trigger_score=100: would_have_spoken=" in out
    assert "would_have_spoken=5/10" in out  # threshold 2 triggers every 2nd


def test_replay_missing_corpus_reports_no_ledger(tmp_path, capsys):
    """A corpus with no complete dispatch ledger is reported clearly — the
    CLI never silently falls back to the raw event corpus."""
    cfg_path = write_config(tmp_path)
    main(["init", "--config", cfg_path])
    seed_chat(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["replay", "qq:group:123456", "--config", cfg_path])
    assert exc.value.code == 1
    assert "no complete dispatch ledger" in capsys.readouterr().err


def test_replay_v2_corpus_reports_requires_v4(tmp_path, capsys):
    """A v2/v3 corpus (dispatch markers without the v4 settled metadata)
    is reported clearly: exact marker replay requires a v4 corpus."""
    cfg_path = write_config(tmp_path)
    main(["init", "--config", cfg_path])
    seed_chat(tmp_path)
    corpus = Path(tmp_path / "cli.db").with_suffix(".jsonl")
    with Recorder(corpus) as rec:
        rec.write_event(
            AdapterEvent(
                type="message",
                payload=make_message(
                    chat_key="qq:group:123456", text="hi", msg_id="m1"
                ),
                ts=1_700_000_000.0,
            ),
            event_id=EventId("ev-1"),
        )
        rec.append_marker(
            CorpusMarker(
                record_type="commit",
                sequence=CommitSeq(1),
                chat_key=ChatKey("qq:group:123456"),
                event_id=EventId("ev-1"),
                wake_kind=WakeKind.INBOUND,
            )
        )
        # A v2-style dispatch marker: no settled state / attached metadata.
        rec.write(
            {
                "record_type": "dispatch",
                "sequence": 1,
                "chat_key": "qq:group:123456",
                "cause": DispatchCause.INBOUND,
            }
        )
    with pytest.raises(SystemExit) as exc:
        main(["replay", "qq:group:123456", "--config", cfg_path])
    assert exc.value.code == 1
    assert "v4 dispatch markers" in capsys.readouterr().err


def test_replay_mixed_v4_v5_corpus_is_rejected_atomically(tmp_path, capsys):
    """One complete v5 dispatch cannot silently hide an older incomplete
    dispatch marker for the same chat."""
    cfg_path = write_config(tmp_path)
    main(["init", "--config", cfg_path])
    seed_chat(tmp_path)
    corpus = write_ledger(
        tmp_path,
        [make_message(chat_key="qq:group:123456", text="hi", msg_id="m1")],
    )
    with Recorder(corpus) as rec:
        rec.write(
            {
                "record_type": "dispatch",
                "sequence": 99,
                "chat_key": "qq:group:123456",
                "cause": DispatchCause.INBOUND,
            }
        )
    with pytest.raises(SystemExit) as exc:
        main(["replay", "qq:group:123456", "--config", cfg_path])
    assert exc.value.code == 1
    assert "mixes incomplete" in capsys.readouterr().err


def _snapshot_dict(chat_key: str, msg, *, through: int = 1,
                   evaluated_ts: float = 1_700_000_001.0) -> dict:
    """A minimal frozen GateSnapshot dict for one attached message (the v5
    exact-replay shape the CLI's marker-driven replay rehydrates)."""
    pending = [
        dataclasses.asdict(dataclasses.replace(msg, row_id=MessageRowId(through)))
    ]
    return {
        "chat_key": chat_key,
        "cycle_id": "dispatch:1",
        "start_msg_id": 0,
        "through_msg_id": through,
        "evaluated_ts": evaluated_ts,
        "self_id": "bot-1",
        "mode": "reply_necessity",
        "threshold": 8,
        "trigger_score": 80,
        "frequency": 1.0,
        "pending": len(pending),
        "pending_messages": pending,
        "recent": pending,
        "window_count": len(pending),
        "self_count": 0,
        "last_nonself_ts": msg.recv_ts,
        "idle_seconds": 0.0,
        "recent_average_interval": 0.0,
        "self_ratio": 0.0,
        "is_group": True,
        "is_focused": False,
        "last_message": pending[-1],
        "self_name": "麦麦",
        "has_direct_at": False,
        "has_quote_to_self": False,
        "has_other_assistant": False,
        "hold_until": None,
        "idle_streak": 0,
        "previous_end_reason": None,
        "backoff_base_s": 15.0,
        "backoff_cap_s": 300.0,
        "backoff_start_count": 2,
    }


def test_replay_fails_closed_on_malformed_dispatch_marker(tmp_path, capsys):
    """A malformed dispatch marker record is a parsing loss of a settled
    decision: the CLI fails closed instead of silently omitting it."""
    cfg_path = write_config(tmp_path)
    main(["init", "--config", cfg_path])
    seed_chat(tmp_path)
    corpus = Path(tmp_path / "cli.db").with_suffix(".jsonl")
    with Recorder(corpus) as rec:
        rec.write_event(
            AdapterEvent(
                type="message",
                payload=make_message(
                    chat_key="qq:group:123456", text="hi", msg_id="m1"
                ),
                ts=1_700_000_000.0,
            ),
            event_id=EventId("ev-1"),
        )
        rec.append_marker(
            CorpusMarker(
                record_type="commit",
                sequence=CommitSeq(1),
                chat_key=ChatKey("qq:group:123456"),
                event_id=EventId("ev-1"),
                wake_kind=WakeKind.INBOUND,
            )
        )
        # A dispatch marker record missing its required fields.
        rec.write(
            {
                "record_type": "dispatch",
                "chat_key": "qq:group:123456",
                "cause": DispatchCause.INBOUND,
            }
        )
    with pytest.raises(SystemExit) as exc:
        main(["replay", "qq:group:123456", "--config", cfg_path])
    assert exc.value.code == 1
    assert "malformed marker record" in capsys.readouterr().err


def test_replay_fails_closed_on_frozen_snapshot_chat_mismatch(tmp_path, capsys):
    """A frozen snapshot whose chat_key does not match its dispatch marker
    is an inconsistency: the CLI fails closed instead of silently omitting
    the settled decision."""
    cfg_path = write_config(tmp_path)
    main(["init", "--config", cfg_path])
    seed_chat(tmp_path)
    corpus = Path(tmp_path / "cli.db").with_suffix(".jsonl")
    msg = make_message(chat_key="qq:group:123456", text="hi", msg_id="m1")
    snapshot = _snapshot_dict("qq:group:other", msg)  # mismatched chat
    with Recorder(corpus) as rec:
        rec.write_event(
            AdapterEvent(type="message", payload=msg, ts=msg.recv_ts),
            event_id=EventId("ev-1"),
        )
        rec.append_marker(
            CorpusMarker(
                record_type="commit",
                sequence=CommitSeq(1),
                chat_key=ChatKey("qq:group:123456"),
                event_id=EventId("ev-1"),
                wake_kind=WakeKind.INBOUND,
            )
        )
        rec.append_marker(
            CorpusMarker(
                record_type="dispatch",
                sequence=DispatchId(1),
                chat_key=ChatKey("qq:group:123456"),
                cause=DispatchCause.INBOUND,
                commit_boundary=CommitSeq(1),
                scheduled_for=None,
                state="completed",
                settled_ts=1_700_000_001.0,
                start_msg_id=MessageRowId(0),
                through_msg_id=MessageRowId(1),
                attached=(CommitSeq(1),),
                trace_json=json.dumps(
                    {"config": _composition_fingerprint(Gate())}
                ),
                evaluated_ts=1_700_000_001.0,
                snapshot_json=json.dumps(snapshot, default=str),
            )
        )
    seed_settled_dispatch_witness(
        tmp_path, "qq:group:123456", [((1,), "completed")], 1_700_000_000.0
    )
    with pytest.raises(SystemExit) as exc:
        main(["replay", "qq:group:123456", "--config", cfg_path])
    assert exc.value.code == 1
    assert "chat mismatch" in capsys.readouterr().err


def test_replay_identity_from_durable_repo_without_self_corpus(tmp_path, capsys):
    """Replay identity comes from the durable repository, never inferred
    from the corpus: a corpus with NO self messages still replays
    structured @ facts against the real self id."""
    cfg_path = write_config(tmp_path)
    main(["init", "--config", cfg_path])
    seed_chat(tmp_path, self_id="bot-1")
    write_ledger(
        tmp_path,
        [
            make_message(
                chat_key="qq:group:123456",
                text="hi",
                msg_id="m1",
                mentions=("bot-1",),
                recv_ts=1_700_000_000.0,
            )
        ],
    )
    assert main(["replay", "qq:group:123456", "--config", cfg_path]) == 0
    out = capsys.readouterr().out
    assert '"reason": "trigger"' in out
    assert "1 decisions, 1 would-have-spoken" in out


def test_replay_fails_clearly_without_durable_identity(tmp_path, capsys):
    """A chat with no durable identity cannot be replayed: the CLI fails
    clearly instead of guessing identity from the corpus."""
    cfg_path = write_config(tmp_path)
    main(["init", "--config", cfg_path])
    with pytest.raises(SystemExit) as exc:
        main(["replay", "qq:group:123456", "--config", cfg_path])
    assert exc.value.code == 1
    assert "no durable chat identity" in capsys.readouterr().err


# ── doctor ───────────────────────────────────────────────────────────────────

def write_doctor_config(tmp_path, *, profiles=None) -> str:
    """A config for the doctor command: a temp db path plus optional
    ``[llm.profiles.<name>]`` tables (each a dict of string values)."""
    path = tmp_path / "doctor.toml"
    lines = [f'[storage]\ndb_path = "{tmp_path / "doctor.db"}"\n']
    for name, prof in (profiles or {}).items():
        lines.append(f"\n[llm.profiles.{name}]\n")
        for key, value in prof.items():
            lines.append(f'{key} = "{value}"\n')
    path.write_text("".join(lines), encoding="utf-8")
    return str(path)


def test_doctor_command_ok(tmp_path, monkeypatch, capsys):
    """A healthy config (no LLM profiles: the chat/tool probes skip) prints
    the deterministic report and exits 0."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # console probe EOF
    cfg_path = write_doctor_config(tmp_path)
    assert main(["doctor", "--config", cfg_path]) == 0
    out = capsys.readouterr().out
    assert out.startswith("doctor: ok")
    assert "[OK] config" in out
    assert "[OK] database" in out
    assert "[SKIP] llm_chat" in out
    assert "summary: 14 probes: " in out


def test_doctor_command_fail_exits_1(tmp_path, monkeypatch, capsys):
    """A broken config (an LLM profile with an empty base_url) fails the
    config probe: the report prints and the command exits 1."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    cfg_path = write_doctor_config(
        tmp_path, profiles={"planner": {"base_url": "", "model": "m"}}
    )
    assert main(["doctor", "--config", cfg_path]) == 1
    out = capsys.readouterr().out
    assert out.startswith("doctor: fail")
    assert "[FAIL] config" in out
    assert "empty base_url" in out


def test_doctor_command_secret_safe_rendering(tmp_path, monkeypatch, capsys):
    """The printed report never leaks a configured api_key — even when a
    probe fails. (Secrets must resolve from ${ENV}; the literal value is
    never accepted in config.)"""
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setenv("DOCTOR_SECRET_KEY", "sk-supersecret")
    cfg_path = write_doctor_config(
        tmp_path,
        profiles={"planner": {"base_url": "", "model": "m",
                              "api_key": "${DOCTOR_SECRET_KEY}"}},
    )
    assert main(["doctor", "--config", cfg_path]) == 1
    out = capsys.readouterr().out
    assert "sk-supersecret" not in out
    assert "doctor: fail" in out


def test_doctor_command_has_no_sql_and_no_send_path(tmp_path, monkeypatch, capsys):
    """The doctor command never touches the chat send path: it only probes
    and prints. (The SQL-location invariant is enforced below for the whole
    CLI module.)"""
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    cfg_path = write_doctor_config(tmp_path)
    assert main(["doctor", "--config", cfg_path]) == 0
    out = capsys.readouterr().out
    assert "doctor: ok" in out
    assert "send" not in out.lower()


# ── SQL-location invariant ──────────────────────────────────────────────────

def test_cli_contains_no_sql_text():
    """All SQL lives in repo.py (and schema DDL in schema.sql); the CLI
    must contain none."""
    cli_path = Path(main.__module__.replace(".", "/") + ".py")
    text = cli_path.read_text(encoding="utf-8")
    for keyword in ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "PRAGMA",
                    "ALTER", "DROP"):
        assert not re.search(rf"\b{keyword}\b", text), (
            f"SQL keyword {keyword!r} found in cli.py"
        )


# ── logging is actually installed ───────────────────────────────────────────
#
# ``setup_logging`` existed and was tested for two phases without ever being
# called from production code. The ``pretender`` logger therefore had no
# handler and inherited root's WARNING level, so every INFO line — adapter
# readiness, gate decisions, outbox sends — was discarded and ``docker logs``
# stayed empty. A bot that silently never replies and emits nothing at all is
# undiagnosable; this test is what keeps the wiring in place.

def test_run_installs_the_configured_log_handlers(tmp_path, monkeypatch):
    import logging

    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        f'[storage]\ndb_path = "{tmp_path / "log.db"}"\n'
        f'[log]\ndir = "{tmp_path / "logs"}"\nlevel = "INFO"\n',
        encoding="utf-8",
    )
    main(["init", "--config", str(cfg_path)])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main(["run", "--dry-run", "--config", str(cfg_path)]) == 0

    logger = logging.getLogger("pretender")
    assert logger.handlers, "run must install handlers, not leave the logger bare"
    assert logger.level == logging.INFO
    logger.info("probe line")
    written = (tmp_path / "logs" / "pretender.jsonl").read_text(encoding="utf-8")
    assert json.loads(written.splitlines()[-1])["msg"] == "probe line"


def test_run_degrades_to_stderr_when_the_log_dir_is_unusable(tmp_path, monkeypatch):
    """Losing the log file is recoverable; refusing to boot is not."""
    import logging

    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        f'[storage]\ndb_path = "{tmp_path / "log2.db"}"\n'
        f'[log]\ndir = "{blocker}"\nlevel = "INFO"\n',
        encoding="utf-8",
    )
    main(["init", "--config", str(cfg_path)])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main(["run", "--dry-run", "--config", str(cfg_path)]) == 0
    assert logging.getLogger("pretender").handlers
