"""Database: one-writer serialization + batching, thread-local reads,
lastrowid exposure, WAL, close semantics, and the time.time prohibition."""

from __future__ import annotations

import asyncio
import re
import sqlite3
import threading
from pathlib import Path

import pytest

import pretender
from pretender.db import Database
from tests.durable_helpers import run


# ── open / WAL / schema ─────────────────────────────────────────────────────

def test_open_creates_file_with_schema_and_wal(tmp_path):
    path = tmp_path / "data" / "t.db"

    async def scenario():
        db = Database(path)
        await db.open()
        mode = await db.read(
            lambda c: c.execute("PRAGMA journal_mode").fetchone()[0]
        )
        version = await db.read(
            lambda c: c.execute("PRAGMA user_version").fetchone()[0]
        )
        await db.close()
        return mode, version

    mode, version = run(scenario())
    assert path.exists()
    assert mode == "wal"
    assert version == 15


def test_write_and_read_roundtrip(tmp_path):
    async def scenario():
        db = Database(tmp_path / "t.db")
        await db.open()
        await db.write(
            lambda c: c.execute("INSERT INTO kv(k, v) VALUES ('a', 'b')")
        )
        value = await db.read(
            lambda c: c.execute("SELECT v FROM kv WHERE k = 'a'").fetchone()[0]
        )
        await db.close()
        return value

    assert run(scenario()) == "b"


# ── writer: serialization, batching, lastrowid ──────────────────────────────

def test_write_exposes_lastrowid(tmp_path):
    async def scenario():
        db = Database(tmp_path / "t.db")
        await db.open()
        row_id = await db.write(
            lambda c: c.execute(
                "INSERT INTO kv(k, v) VALUES ('x', 'y')"
            ).lastrowid
        )
        await db.close()
        return row_id

    assert run(scenario()) == 1


def test_writer_serializes_concurrent_writes(tmp_path):
    async def scenario():
        db = Database(tmp_path / "t.db", batch_window=0.0)
        await db.open()

        async def put(i: int) -> None:
            await db.write(
                lambda c, i=i: c.execute(
                    "INSERT INTO kv(k, v) VALUES (?, ?)", (f"k{i}", str(i))
                )
            )

        await asyncio.gather(*[put(i) for i in range(50)])
        count = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM kv").fetchone()[0]
        )
        await db.close()
        return count

    assert run(scenario()) == 50


def test_batching_caps_at_200_ops_per_transaction(tmp_path):
    async def scenario():
        db = Database(tmp_path / "t.db", batch_window=0.0, batch_cap=200)
        await db.open()

        async def put(i: int) -> None:
            await db.write(
                lambda c, i=i: c.execute(
                    "INSERT INTO kv(k, v) VALUES (?, ?)", (f"k{i}", str(i))
                )
            )

        await asyncio.gather(*[put(i) for i in range(250)])
        count = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM kv").fetchone()[0]
        )
        transactions = db.transactions
        await db.close()
        return count, transactions

    count, transactions = run(scenario())
    assert count == 250
    assert transactions == 2  # 200 + 50


def test_coalescing_window_batches_sequential_writes(tmp_path):
    async def scenario():
        db = Database(tmp_path / "t.db", batch_window=0.05)
        await db.open()

        async def put(i: int) -> None:
            await db.write(
                lambda c, i=i: c.execute(
                    "INSERT INTO kv(k, v) VALUES (?, ?)", (f"k{i}", str(i))
                )
            )

        # Fire three writes back-to-back without awaiting: the writer
        # coalesces them into one transaction within the 50 ms window.
        tasks = [asyncio.create_task(put(i)) for i in range(3)]
        await asyncio.gather(*tasks)
        transactions = db.transactions
        await db.close()
        return transactions

    assert run(scenario()) == 1


def test_failed_op_is_isolated_by_savepoint(tmp_path):
    """A failing op in a batch rolls back ONLY its own savepoint: unrelated
    work in the same batch still commits (no writer error contagion)."""

    async def scenario():
        db = Database(tmp_path / "t.db", batch_window=0.0)
        await db.open()

        async def good() -> None:
            await db.write(
                lambda c: c.execute("INSERT INTO kv(k, v) VALUES ('a', '1')")
            )

        async def bad() -> None:
            await db.write(
                lambda c: c.execute("INSERT INTO nonexistent(k) VALUES (1)")
            )

        results = await asyncio.gather(
            asyncio.create_task(good()),
            asyncio.create_task(bad()),
            return_exceptions=True,
        )
        # The bad op fails; the good op in the SAME batch commits.
        assert isinstance(results[0], type(None)) or results[0] is None
        assert isinstance(results[1], sqlite3.OperationalError)
        value = await db.read(
            lambda c: c.execute("SELECT v FROM kv WHERE k = 'a'").fetchone()[0]
        )
        await db.close()
        return value

    assert run(scenario()) == "1"


def test_failed_op_does_not_poison_following_ops(tmp_path):
    """Ops queued AFTER a failing op in the same batch still commit."""

    async def scenario():
        db = Database(tmp_path / "t.db", batch_window=0.0)
        await db.open()

        async def bad() -> None:
            await db.write(
                lambda c: c.execute("INSERT INTO nonexistent(k) VALUES (1)")
            )

        async def good() -> None:
            await db.write(
                lambda c: c.execute("INSERT INTO kv(k, v) VALUES ('b', '2')")
            )

        results = await asyncio.gather(
            asyncio.create_task(bad()),
            asyncio.create_task(good()),
            return_exceptions=True,
        )
        assert isinstance(results[0], sqlite3.OperationalError)
        assert results[1] is None
        value = await db.read(
            lambda c: c.execute("SELECT v FROM kv WHERE k = 'b'").fetchone()[0]
        )
        await db.close()
        return value

    assert run(scenario()) == "2"


def test_commit_failure_fails_every_op_and_recovers(tmp_path):
    """No future is resolved until the outer COMMIT succeeds: an injected
    COMMIT failure (a deferred FK violation) fails EVERY otherwise-
    successful queued operation, rolls the transaction back, and leaves
    the writer connection usable for the next batch."""

    async def scenario():
        db = Database(tmp_path / "t.db", batch_window=0.0)
        await db.open()

        async def good() -> None:
            await db.write(
                lambda c: c.execute("INSERT INTO kv(k, v) VALUES ('a', '1')")
            )

        async def poison() -> None:
            # Defer the FK check to COMMIT: the insert itself succeeds at
            # its savepoint, then the outer COMMIT fails.
            await db.write(
                lambda c: (
                    c.execute("PRAGMA defer_foreign_keys=ON"),
                    c.execute(
                        "INSERT INTO messages(chat_key, platform, self_id,"
                        " sender_id, sender_name, is_self, text)"
                        " VALUES ('ghost', 'qq', 'b', 'u', 'u', 0, 'x')"
                    ),
                )
            )

        results = await asyncio.gather(
            asyncio.create_task(good()),
            asyncio.create_task(poison()),
            return_exceptions=True,
        )
        # BOTH must fail: the savepoint-level success was held, never
        # resolved, and the COMMIT failure fails it too.
        assert all(isinstance(r, Exception) for r in results)
        # The writer connection must be usable again (ROLLBACK restored it).
        await db.write(
            lambda c: c.execute("INSERT INTO kv(k, v) VALUES ('b', '2')")
        )
        value = await db.read(
            lambda c: c.execute("SELECT v FROM kv WHERE k = 'b'").fetchone()[0]
        )
        count = await db.read(
            lambda c: c.execute("SELECT COUNT(*) FROM kv").fetchone()[0]
        )
        await db.close()
        return value, count

    value, count = run(scenario())
    assert value == "2"  # the writer recovered
    assert count == 1  # the rolled-back 'a' row is gone


# ── reads: 2-thread executor, thread-local, query_only ──────────────────────

def test_reads_use_thread_local_connections(tmp_path):
    async def scenario():
        db = Database(tmp_path / "t.db")
        await db.open()
        barrier = threading.Barrier(2, timeout=5)

        def connection_id(conn):
            barrier.wait()
            return id(conn)

        conn_ids = await asyncio.gather(
            *[db.read(connection_id) for _ in range(2)]
        )
        await db.close()
        return conn_ids

    conn_ids = run(scenario())
    assert len(set(conn_ids)) >= 2  # two executor threads, two connections


def test_read_connections_are_query_only(tmp_path):
    async def scenario():
        db = Database(tmp_path / "t.db")
        await db.open()
        try:
            await db.read(
                lambda c: c.execute("INSERT INTO kv(k, v) VALUES ('a', 'b')")
            )
        except sqlite3.OperationalError:
            return "blocked"
        finally:
            await db.close()
        return "allowed"

    assert run(scenario()) == "blocked"


def test_read_sees_committed_writes(tmp_path):
    async def scenario():
        db = Database(tmp_path / "t.db")
        await db.open()
        await db.write(
            lambda c: c.execute("INSERT INTO kv(k, v) VALUES ('a', 'b')")
        )
        # A read on a separate thread-local connection sees the WAL commit.
        value = await db.read(
            lambda c: c.execute("SELECT v FROM kv WHERE k = 'a'").fetchone()[0]
        )
        await db.close()
        return value

    assert run(scenario()) == "b"


# ── close semantics ─────────────────────────────────────────────────────────

def test_close_is_idempotent_and_rejects_writes(tmp_path):
    async def scenario():
        db = Database(tmp_path / "t.db")
        await db.open()
        await db.close()
        await db.close()  # idempotent
        with pytest.raises(RuntimeError):
            await db.write(
                lambda c: c.execute("INSERT INTO kv(k, v) VALUES ('a', 'b')")
            )
        with pytest.raises(RuntimeError):
            await db.read(lambda c: 1)

    run(scenario())


def test_queued_writes_complete_gracefully_on_close(tmp_path):
    """close() drains work queued before it: the writer executes everything
    ahead of the sentinel, so an in-flight write commits, not fails."""

    async def scenario():
        db = Database(tmp_path / "t.db", batch_window=0.05)
        await db.open()
        task = asyncio.create_task(
            db.write(
                lambda c: c.execute("INSERT INTO kv(k, v) VALUES ('a', 'b')")
            )
        )
        await asyncio.sleep(0)  # let the write reach the queue
        await db.close()
        await task  # completes normally — the batch ran before the sentinel
        value = sqlite3.connect(tmp_path / "t.db").execute(
            "SELECT v FROM kv WHERE k = 'a'"
        ).fetchone()[0]
        return value

    assert run(scenario()) == "b"


def test_reopen_persists_data(tmp_path):
    path = tmp_path / "t.db"

    async def scenario():
        db = Database(path)
        await db.open()
        await db.write(
            lambda c: c.execute("INSERT INTO kv(k, v) VALUES ('a', 'b')")
        )
        await db.close()
        db2 = Database(path)
        await db2.open()
        value = await db2.read(
            lambda c: c.execute("SELECT v FROM kv WHERE k = 'a'").fetchone()[0]
        )
        await db2.close()
        return value

    assert run(scenario()) == "b"


# ── the time.time prohibition (PLAN.md §8) ──────────────────────────────────

def test_time_time_forbidden_outside_clock_py():
    root = Path(pretender.__file__).parent
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "clock.py":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"time\.time\s*\(", line):
                offenders.append(f"{path.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, "time.time() outside clock.py:\n" + "\n".join(offenders)
