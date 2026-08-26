"""SQLite access: ONE async writer owner, thread-local reads, WAL.

Design (PLAN.md §5, §7):
  - The database file is opened once, in WAL mode, with the full M0 schema
    applied at open (``schema.apply_schema``).
  - **One writer.** Every mutation goes through ``write(fn)``, which submits
    ``fn`` to a single writer task owning the only write connection. The
    writer serializes transactions and batches compatible work: it drains
    the queue and executes up to ``batch_cap`` (200) ops in ONE transaction,
    coalescing for up to ``batch_window`` (50 ms) after the first op. A
    failure rolls the whole batch back — which is why repo.py uses
    ``ON CONFLICT DO NOTHING`` everywhere: one duplicate must never poison
    a batch.
  - **Two-thread reads.** ``read(fn)`` runs on a 2-thread executor; each
    thread holds its own thread-local read connection (``PRAGMA query_only``
    so a read can never mutate). WAL readers never block the writer.
  - **lastrowid.** ``write`` returns whatever ``fn`` returns, so a caller
    exposes the row id with ``fn = lambda c: c.execute(...).lastrowid``.

Timestamps are NEVER produced here: callers pass absolute epoch seconds
(``time.time`` lives only in clock.py).
"""

from __future__ import annotations

import asyncio
import inspect
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, TypeVar

from pretender.schema import apply_schema

T = TypeVar("T")

WriteFn = Callable[[sqlite3.Connection], T]
ReadFn = Callable[[sqlite3.Connection], T]

# Writer queue item: (fn, future). The sentinel None means "close".
_Item = tuple[WriteFn[Any], asyncio.Future[Any]] | None


class Database:
    """The single SQLite access point for the whole process.

    Not thread-safe by design: use it from the event loop only. All I/O is
    async; the underlying connections live in the writer task and the read
    executor threads.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        batch_cap: int = 200,
        batch_window: float = 0.05,
        read_threads: int = 2,
    ) -> None:
        self._path = str(path)
        self._batch_cap = batch_cap
        self._batch_window = batch_window
        self._executor = ThreadPoolExecutor(
            max_workers=read_threads, thread_name_prefix="pretender-read"
        )
        self._local = threading.local()
        self._writer_conn: sqlite3.Connection | None = None
        self._queue: asyncio.Queue[_Item] = asyncio.Queue()
        self._writer_task: asyncio.Task[None] | None = None
        self._closed = False
        self._lock = asyncio.Lock()
        # Test-visible: number of committed writer transactions.
        self.transactions = 0

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def open(self) -> None:
        """Open the file (creating it and the schema when missing) and start
        the writer task. Idempotent per instance; call once."""
        if self._writer_conn is not None:
            return
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._writer_conn = await asyncio.to_thread(self._open_writer)
        self._writer_task = asyncio.create_task(self._writer_loop())

    def _open_writer(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._path, isolation_level=None, check_same_thread=False
        )
        # check_same_thread=False is safe: the connection is used only from
        # the writer task after this thread finishes.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        apply_schema(conn)
        return conn

    async def close(self) -> None:
        """Stop the writer, close connections, release the read executor.
        Idempotent; pending writes are failed with RuntimeError."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put_nowait(None)  # sentinel wakes the writer
        if self._writer_task is not None:
            await self._writer_task
            self._writer_task = None
        if self._writer_conn is not None:
            self._writer_conn.close()
            self._writer_conn = None
        self._executor.shutdown(wait=True)

    # ── write path: one owner, batched transactions ─────────────────────────

    async def write(self, fn: WriteFn[T]) -> T:
        """Run ``fn(conn)`` inside a writer transaction; return its result.

        ``fn`` is synchronous and receives the writer connection. To expose
        the row id of an insert, return ``conn.lastrowid`` from ``fn``.
        """
        if inspect.iscoroutinefunction(fn):
            raise TypeError("write() fn must be synchronous")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[T] = loop.create_future()
        async with self._lock:
            if self._closed:
                raise RuntimeError("database closed")
            self._queue.put_nowait((fn, fut))
        return await fut

    async def _writer_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            item = await self._queue.get()
            if item is None:
                self._fail_queued()
                return
            batch: list[tuple[WriteFn[Any], asyncio.Future[Any]]] = [item]
            # Drain whatever is already queued, up to the cap.
            while len(batch) < self._batch_cap:
                try:
                    nxt = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if nxt is None:
                    self._execute_batch(batch)
                    self._fail_queued()
                    return
                batch.append(nxt)
            # Coalesce for up to batch_window more.
            deadline = loop.time() + self._batch_window
            while len(batch) < self._batch_cap:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self._queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                if nxt is None:
                    self._execute_batch(batch)
                    self._fail_queued()
                    return
                batch.append(nxt)
            self._execute_batch(batch)

    def _execute_batch(
        self, batch: list[tuple[WriteFn[Any], asyncio.Future[Any]]]
    ) -> None:
        """Run one batch in a single transaction, with each queued logical
        transaction isolated in its own SAVEPOINT.

        A failing op (an expected CAS miss, a fencing ClaimError, a
        constraint violation) rolls back ONLY its own savepoint: the rest
        of the batch — unrelated committed inbound work — still commits.

        NO future is resolved until the outer COMMIT succeeds: successful
        results are held, and a COMMIT failure rolls the whole transaction
        back, fails EVERY otherwise-successful queued operation, and leaves
        the writer connection usable for the next batch."""
        conn = self._writer_conn
        assert conn is not None
        try:
            conn.execute("BEGIN")
        except sqlite3.Error as e:
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(e)
            return
        held: list[tuple[asyncio.Future[Any], Any]] = []
        for fn, fut in batch:
            conn.execute("SAVEPOINT op")
            try:
                result = fn(conn)
            except BaseException as e:
                conn.execute("ROLLBACK TO op")
                conn.execute("RELEASE op")
                if not fut.done():
                    fut.set_exception(e)
                continue
            conn.execute("RELEASE op")
            held.append((fut, result))
        try:
            conn.execute("COMMIT")
            self.transactions += 1
        except sqlite3.Error as e:
            # The outer COMMIT failed: roll the whole transaction back,
            # fail every queued operation (including the savepoint-level
            # successes — their futures were held, never resolved), and
            # restore a usable writer state for the next batch.
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            for fut, _result in held:
                if not fut.done():
                    fut.set_exception(e)
            return
        for fut, result in held:
            if not fut.done():
                fut.set_result(result)

    def _fail_queued(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if item is not None:
                _, fut = item
                if not fut.done():
                    fut.set_exception(RuntimeError("database closed"))

    # ── read path: 2-thread executor, thread-local connections ──────────────

    async def read(self, fn: ReadFn[T]) -> T:
        """Run ``fn(conn)`` on the read executor; return its result.

        Each executor thread lazily opens its own read-only connection
        (thread-local), so concurrent reads never share a connection.
        """
        if self._closed:
            raise RuntimeError("database closed")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._run_read, fn)

    def _run_read(self, fn: ReadFn[T]) -> T:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path)
            conn.execute("PRAGMA query_only = ON")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return fn(conn)