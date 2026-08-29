"""The command line: ``init`` | ``run`` | ``db`` | ``replay`` | ``doctor``.

Phase 3 is console-only and non-network:

  - ``pretender init`` — create the database file with the full M0 schema.
  - ``pretender run`` — boot the console REPL in DRY-RUN by default: inbound
    lines are recorded and committed, the ledger scheduler drives
    deterministic gate cycles, and a trigger runs the two-stage agent
    (planner + replyer) but creates ZERO outbox rows, never starts the outbox
    worker, and never sends — printing the full ``DecisionTrace`` per decision
    as one JSON line. ``--live`` is the EXPLICIT live mode: it requires the
    ``planner``/``reply`` LLM profiles and sends real messages through the
    configured adapter (console or, when ``adapter.name = "onebot"``, the
    OneBot v11 bridge). Live is never the default, so there is no accidental
    live send.
  - ``pretender db`` — basic table statistics (message/memory/record/cycle
    counts, outbox state histogram, schema version).
  - ``pretender replay <chat>`` — deterministically re-score the recorded
    v4 DISPATCH SCHEDULE through the same snapshot assembler + gate path
    (no outbox or adapter operations; the only storage read is the chat's
    durable identity, which must exist) and print each trace plus a
    would-have-spoken summary. The marker-driven path reconstructs every
    settled dispatch from its frozen attached membership/boundary/settled
    time, so a missing corpus or a corpus without a complete v4 dispatch
    ledger is reported clearly. ``--sweep`` varies the gate constants
    (threshold x trigger_score) through a RuntimeOverlay over the FIXED
    recorded schedule and reports the would-have-spoken count/rate per
    combination.
  - ``pretender doctor`` — run the deterministic preflight probes (config,
    prompts, database, adapter, llm_chat, llm_tools, vision, embed), print
    the secret-free report, and return 0 on a healthy report or 1 when any
    probe failed. No chat output is ever sent.

This module contains NO SQL text: every query goes through the repository
(``repo.stats``). ``main`` returns 0 on success and raises SystemExit(1) on
a handled error (config, database), so ``python -m pretender`` propagates
the exit code.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any

from pretender.app import App, build_replay_gate
from pretender.config import Config
from pretender.cycle import replay_marker_schedule, sweep_marker_schedule
from pretender.db import Database
from pretender.doctor import Doctor, DoctorReport
from pretender.errors import ConfigError, PretenderError, RepoError
from pretender.log import JsonFormatter, setup_logging
from pretender.record import CorpusView, read_corpus_view
from pretender.repo import SqliteRepository
from pretender.types import (
    AdapterEvent,
    ChatIdentity,
    ChatKey,
    DecisionTrace,
    Message,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pretender", description="Pretender — a light MaiBot"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create the database schema")
    p_init.add_argument("--config", default=None, help="path to config TOML")

    p_run = sub.add_parser("run", help="run the bot (console or onebot adapter)")
    p_run.add_argument("--config", default=None, help="path to config TOML")
    mode = p_run.add_mutually_exclusive_group()
    mode.add_argument(
        "--live",
        action="store_true",
        help="run LIVE: send real messages through the configured adapter"
        " (requires the planner/reply LLM profiles; never the default)",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="evaluate the same agent but never create outbox rows or send"
        " (prints the full DecisionTrace per decision; the default)",
    )

    p_db = sub.add_parser("db", help="database utilities")
    p_db.add_argument("--config", default=None, help="path to config TOML")
    p_db.add_argument(
        "--stats", action="store_true", help="print basic table statistics"
    )

    p_replay = sub.add_parser(
        "replay", help="replay a recorded corpus through the gate"
    )
    p_replay.add_argument("chat", help="chat key to replay")
    p_replay.add_argument("--config", default=None, help="path to config TOML")
    p_replay.add_argument(
        "--sweep",
        action="store_true",
        help="sweep gate constants and report would-have-spoken counts",
    )

    p_doctor = sub.add_parser(
        "doctor", help="run the preflight doctor and print its report"
    )
    p_doctor.add_argument("--config", default=None, help="path to config TOML")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return _cmd_init(args)
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "db":
            return _cmd_db(args)
        if args.command == "replay":
            return _cmd_replay(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
    except (PretenderError, OSError) as e:
        print(f"pretender: error: {e}", file=sys.stderr)
        raise SystemExit(1) from None
    return 0


# ── commands ────────────────────────────────────────────────────────────────

def _cmd_init(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)

    async def _init() -> int:
        db = Database(cfg.storage.db_path)
        await db.open()
        try:
            repo = SqliteRepository(db)
            return (await repo.stats())["user_version"]
        finally:
            await db.close()

    version = asyncio.run(_init())
    print(f"initialized {cfg.storage.db_path} (schema v{version})")
    return 0


def _configure_logging(cfg: Config) -> None:
    """Install the ``[log]`` handlers on the ``pretender`` logger.

    Without this call the logger has NO handler and inherits root's WARNING
    level, so every ``log.info`` — adapter readiness, gate decisions, outbox
    sends — is discarded and ``docker logs`` stays empty. That is exactly how
    a bot that silently never replies becomes undiagnosable.

    Degrades rather than refusing to boot: an unusable log directory or an
    unknown level falls back to a stderr-only JSONL stream, because losing the
    log file is recoverable and failing to start is not.
    """
    try:
        setup_logging(
            directory=cfg.log.dir,
            level=cfg.log.level,
            max_bytes=cfg.log.max_bytes,
            backup_count=cfg.log.backup_count,
        )
        return
    except (OSError, ValueError) as exc:
        reason = exc
    logger = logging.getLogger("pretender")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.warning(
        "log setup failed for dir=%r level=%r (%s); using stderr only",
        cfg.log.dir,
        cfg.log.level,
        reason,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    _configure_logging(cfg)
    if args.live:
        # Live is EXPLICIT: it requires the planner/reply LLM profiles and
        # sends real messages through the configured adapter. Never the
        # default — plain ``run`` stays dry-run (zero outbox/send).
        _require_agent_profiles(cfg)
        app = App.build(cfg, dry_run=False)
    else:
        # Dry-run by default: the same agent evaluation (when profiles are
        # configured) but with ZERO outbox rows/worker/send.
        app = App.build(cfg, dry_run=True, trace_sink=_print_trace)
    asyncio.run(_run_app_with_sigterm(app))
    return 0


async def _run_app_with_sigterm(app: Any) -> None:
    """Run ``app`` with CLI-owned SIGTERM coordination.

    ``App.run`` cannot protect the initial ``start()`` call with its own
    ``finally`` block.  Keep shutdown on this task, cancel the run task first,
    and only then perform the defensive cleanup so the two shutdown paths can
    never overlap.
    """
    loop = asyncio.get_running_loop()
    terminated = asyncio.Event()
    sigterm = getattr(signal, "SIGTERM", None)
    signal_api = getattr(signal, "signal", None)
    previous_handler = None
    handler_installed = False

    def _handle_sigterm(_signum: int, _frame: object) -> None:
        # Signal handlers must not do async work.  In particular, shutdown
        # must not be started here because App.run may still be shutting down.
        loop.call_soon_threadsafe(terminated.set)

    if sigterm is not None and callable(signal_api):
        try:
            previous_handler = signal_api(sigterm, _handle_sigterm)
        except (AttributeError, NotImplementedError, OSError, TypeError,
                ValueError, RuntimeError):
            # SIGTERM is unavailable on some platforms and signal.signal()
            # rejects non-main-thread callers.  The CLI remains usable there.
            pass
        else:
            handler_installed = True

    run_task: asyncio.Task[None] | None = None
    termination_task: asyncio.Task[bool] | None = None
    try:
        # Give a handler invoked during registration a chance to set the event
        # before app.run is even scheduled.  This is also the safe path for a
        # SIGTERM arriving during the small setup window.
        await asyncio.sleep(0)
        if not terminated.is_set():
            run_task = asyncio.create_task(app.run())
            termination_task = asyncio.create_task(terminated.wait())
            done, _ = await asyncio.wait(
                (run_task, termination_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if run_task in done:
                # Preserve normal completion and application errors exactly.
                await run_task
            else:
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    # SIGTERM cancellation is the expected graceful exit.
                    pass
    except asyncio.CancelledError:
        # A cancellation from outside this coordinator (including asyncio's
        # SIGINT handling) must remain a cancellation, not become success.
        if run_task is not None and not run_task.done():
            run_task.cancel()
        if run_task is not None:
            try:
                await run_task
            except (asyncio.CancelledError, Exception):
                pass
        raise
    finally:
        if termination_task is not None:
            if not termination_task.done():
                termination_task.cancel()
            try:
                await termination_task
            except asyncio.CancelledError:
                pass
        try:
            # This is serialized after run_task is terminal (or when SIGTERM
            # arrived before it was scheduled), covering the start() gap.
            await app.shutdown()
        finally:
            if handler_installed:
                assert sigterm is not None
                assert callable(signal_api)
                signal_api(sigterm, previous_handler)


def _require_agent_profiles(cfg: Config) -> None:
    """``pretender run --live`` is live and requires the planner and reply
    LLM profiles (the default agent build needs them)."""
    profiles = cfg.llm.profiles
    missing = [p for p in ("planner", "reply") if p not in profiles]
    if missing:
        raise ConfigError(
            "`pretender run --live` is live and requires the planner and"
            f" reply LLM profiles; missing: {', '.join(missing)}"
        )


def _cmd_db(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)

    async def _stats() -> dict[str, int]:
        db = Database(cfg.storage.db_path)
        await db.open()
        try:
            repo = SqliteRepository(db)
            return await repo.stats()
        finally:
            await db.close()

    counts = asyncio.run(_stats())
    for key in sorted(counts):
        print(f"{key}={counts[key]}")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    corpus_path = Path(cfg.storage.db_path).with_suffix(".jsonl")
    chat_key = ChatKey(args.chat)
    identity = _replay_identity(cfg, chat_key)
    try:
        view = read_corpus_view(corpus_path)
    except ValueError as e:
        # A malformed marker record is a parsing loss that would silently
        # omit a settled decision: exact replay fails closed.
        raise PretenderError(f"cannot replay {chat_key!r}: {e}") from e
    _require_complete_ledger(corpus_path, view, chat_key, cfg.storage.db_path)
    # Replay must use the same configured frozen feature composition as live;
    # the replay gate performs no adapter/database/outbox work.
    gate = build_replay_gate(cfg)
    try:
        rows = (
            list(
                sweep_marker_schedule(
                    view,
                    chat_key=chat_key,
                    identity=identity,
                    cfg=cfg,
                    gate=gate,
                )
            )
            if args.sweep
            else None
        )
        result = (
            None
            if rows is not None
            else replay_marker_schedule(
                view, chat_key=chat_key, identity=identity, cfg=cfg, gate=gate
            )
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        raise PretenderError(
            f"cannot replay {chat_key!r}: invalid dispatch ledger: {e}"
        ) from e
    if rows is not None:
        for row in rows:
            print(
                f"threshold={row.threshold} trigger_score={row.trigger_score}:"
                f" would_have_spoken={row.would_have_spoken}/{row.decisions}"
                f" ({row.rate:.3f})"
            )
        return 0
    assert result is not None
    for trace in result.traces:
        _print_trace(trace)
    print(
        f"replay {args.chat}: {result.decisions} decisions,"
        f" {result.would_have_spoken} would-have-spoken ({result.rate:.3f})"
    )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Run the deterministic preflight probes and print the secret-free
    report. Returns 0 on a healthy report, 1 when any probe failed. No chat
    output is ever sent — the doctor only probes and reports."""
    cfg = Config.load(args.config)

    async def _run() -> DoctorReport:
        doctor = Doctor(cfg)
        return await doctor.run()

    report = asyncio.run(_run())
    print(report.render())
    return 0 if not report.failed else 1


# ── helpers ─────────────────────────────────────────────────────────────────

def _require_complete_ledger(
    corpus_path: Path, view: CorpusView, chat_key: ChatKey, db_path: str
) -> None:
    """The marker-driven replay requires a COMPLETE v5 dispatch ledger for
    the chat: at least one settled dispatch marker carrying the v4
    evaluation metadata. A missing corpus, a corpus with no dispatch
    markers, or a v2/v3 corpus (markers without the v4 settled metadata)
    is reported clearly — the CLI never silently falls back to the raw
    event corpus."""
    chat_dispatches = [m for m in view.dispatches if m.chat_key == chat_key]
    if not chat_dispatches:
        raise PretenderError(
            f"cannot replay {chat_key!r}: no complete dispatch ledger in"
            f" {corpus_path} (no settled dispatch markers for this chat;"
            f" record and run the chat first)"
        )
    incomplete = [
        marker
        for marker in chat_dispatches
        if marker.state not in ("completed", "released")
        or marker.settled_ts is None
        or marker.snapshot_json is None
        or marker.evaluated_ts is None
        or marker.start_msg_id is None
        or marker.through_msg_id is None
    ]
    if incomplete:
        raise PretenderError(
            f"cannot replay {chat_key!r}: the corpus in {corpus_path} mixes"
            " incomplete v2-v4 dispatch markers with v5 markers; exact"
            " replay requires one complete v5 ledger"
        )
    settled = [
        marker
        for marker in chat_dispatches
        if marker.state in ("completed", "released")
        and marker.settled_ts is not None
    ]
    if not settled:
        raise PretenderError(
            f"cannot replay {chat_key!r}: the corpus in {corpus_path} lacks"
            f" v4 dispatch markers (settled state/attached metadata);"
            f" exact marker replay requires a v4 corpus"
        )
    if view.manifest_dispatches is None:
        raise PretenderError(
            f"cannot replay {chat_key!r}: corpus has no final completeness"
            " manifest (record a clean corpus close first)"
        )
    exported = frozenset(marker.sequence for marker in settled)
    if exported != view.manifest_dispatches.get(chat_key, frozenset()):
        raise PretenderError(
            f"cannot replay {chat_key!r}: corpus manifest does not match"
            " settled dispatch markers"
        )
    try:
        durable = SqliteRepository.read_settled_dispatch_ids_readonly(db_path, chat_key)
    except RepoError as e:
        raise PretenderError(
            f"cannot replay {chat_key!r}: cannot verify durable dispatch ledger"
        ) from e
    if exported != durable:
        raise PretenderError(
            f"cannot replay {chat_key!r}: corpus dispatch markers do not match"
            " the durable settled ledger (corpus may be truncated or stale)"
        )

def _print_trace(trace: DecisionTrace) -> None:
    """One full DecisionTrace as a single JSON line (replay-safe)."""
    print(json.dumps(dataclasses.asdict(trace), default=str))


def _replay_identity(cfg: Config, chat_key: ChatKey) -> ChatIdentity:
    """The replay identity comes from the DURABLE configured repository
    (the chat's stored identity), never inferred from the corpus — a
    corpus with no self messages must still replay structured @ facts
    against the real self id. Fails clearly when the chat is unknown.
    The replay command delegates its immutable read-only identity lookup to the
    repository rather than opening the normal writable Database owner."""
    try:
        identity = SqliteRepository.read_chat_identity_readonly(
            cfg.storage.db_path, chat_key
        )
    except PretenderError as e:
        raise PretenderError(
            f"cannot replay {chat_key!r}: cannot read durable identity"
        ) from e
    if identity is None:
        raise PretenderError(
            f"cannot replay {chat_key!r}: no durable chat identity in"
            f" {cfg.storage.db_path} (record the chat first)"
        )
    return identity
