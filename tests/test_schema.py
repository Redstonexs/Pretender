"""Schema: fresh M0 creation, FTS5 external content, the partial unique
claim index, and the MIGRATIONS mechanism (versioned, transactional,
idempotent)."""

from __future__ import annotations

import sqlite3

import pytest

import pretender.schema as schema
from pretender.schema import apply_schema, current_version

ALL_TABLES = {
    "chats", "messages", "message_fts", "persons", "memories", "memory_fts",
    "records", "vectors", "emoji", "outbox", "cycles", "claims", "kv",
    "inbound_commits", "dispatches", "memory_search_docs", "memory_search_fts",
    "embedding_generations", "memory_fts_state", "media_assets",
    "chat_controls", "chat_focus_current",
}


def fresh_conn(path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None)
    return conn


# ── Fresh M0 creation ───────────────────────────────────────────────────────

def test_fresh_schema_creates_every_table(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    # FTS5 external-content tables add shadow tables (message_fts_data, ...);
    # every declared table must exist, extras are the FTS machinery.
    assert ALL_TABLES <= {r[0] for r in rows}
    conn.close()


def test_fresh_schema_sets_user_version_to_15(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    assert current_version(conn) == 15
    conn.close()


def test_fts_tables_are_external_content(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    for table in ("message_fts", "memory_fts"):
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", (table,)
        ).fetchone()[0]
        assert "fts5" in sql
        assert "content=" in sql and "content_rowid=" in sql
    conn.close()


def test_claims_partial_unique_index_enforces_one_live_claim(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    idx = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'claims_one_live'"
    ).fetchone()
    assert idx is not None and "WHERE state = 'live'" in idx[0]
    conn.execute(
        "INSERT INTO claims(chat_key, cycle_id, started_ts, expires_at,"
        " start_msg_id, through_msg_id, state)"
        " VALUES ('c1', 'cy1', 1.0, 100.0, 0, 0, 'live')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO claims(chat_key, cycle_id, started_ts, expires_at,"
            " start_msg_id, through_msg_id, state)"
            " VALUES ('c1', 'cy2', 2.0, 200.0, 0, 0, 'live')"
        )
    # A finished claim does not block a new live one.
    conn.execute(
        "INSERT INTO claims(chat_key, cycle_id, started_ts, expires_at,"
        " start_msg_id, through_msg_id, state)"
        " VALUES ('c2', 'cy3', 3.0, 300.0, 0, 0, 'finished')"
    )
    conn.execute(
        "INSERT INTO claims(chat_key, cycle_id, started_ts, expires_at,"
        " start_msg_id, through_msg_id, state)"
        " VALUES ('c2', 'cy4', 4.0, 400.0, 0, 0, 'live')"
    )
    conn.close()


def test_claims_lease_is_mandatory_in_schema(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO claims(chat_key, cycle_id, started_ts,"
            " start_msg_id, through_msg_id, state)"
            " VALUES ('c1', 'cy1', 1.0, 0, 0, 'live')"
        )
    conn.close()


def test_outbox_has_cycle_provenance_column(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(outbox)").fetchall()}
    assert "cycle_id" in cols
    conn.close()


def test_chats_has_hold_until_distinct_from_focus_until(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chats)").fetchall()}
    assert "hold_until" in cols
    assert "focus_until" in cols
    conn.close()


# ── durable dispatch ledger schema (migration 2) ────────────────────────────

def test_fresh_schema_has_ledger_tables(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    commit_cols = {
        r[1] for r in conn.execute("PRAGMA table_info(inbound_commits)").fetchall()
    }
    assert {
        "id", "event_id", "chat_key", "message_id", "committed_ts",
        "wake_kind", "pending_count", "dispatch_id", "exported", "priority",
    } <= commit_cols
    dispatch_cols = {
        r[1] for r in conn.execute("PRAGMA table_info(dispatches)").fetchall()
    }
    assert {
        "id", "chat_key", "cause", "wake_kind", "scheduled_ts", "started_ts",
        "expires_at", "claimed_ts", "cycle_id", "start_msg_id",
        "through_msg_id", "state", "trace_json", "exported",
        "commit_boundary", "attached_json", "settled_ts", "evaluated_ts",
        "snapshot_json",
    } <= dispatch_cols
    conn.close()


def test_ledger_indexes_make_scans_safe(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    # Unassigned-commit scan: partial index on NULL dispatch_id.
    idx = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index'"
        " AND name = 'inbound_commits_unassigned'"
    ).fetchone()
    assert idx is not None and "dispatch_id IS NULL" in idx[0]
    # Prepared-dispatch recovery: at most ONE prepared dispatch per chat.
    idx = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index'"
        " AND name = 'dispatches_one_prepared'"
    ).fetchone()
    assert idx is not None and "WHERE state = 'prepared'" in idx[0]
    # Unexported-marker scans: partial indexes on exported = 0.
    for name in ("inbound_commits_unexported", "dispatches_unexported"):
        idx = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (name,),
        ).fetchone()
        assert idx is not None and "exported = 0" in idx[0]
    conn.close()


def test_ledger_fks_reject_orphans(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    conn.execute("PRAGMA foreign_keys=ON")  # the writer connection enforces FKs
    apply_schema(conn)
    conn.execute(
        "INSERT INTO chats(chat_key, platform, self_id, kind)"
        " VALUES ('c1', 'qq', 'bot', 'group')"
    )
    conn.execute(
        "INSERT INTO messages(chat_key, platform, self_id, platform_msg_id,"
        " sender_id, sender_name, is_self, text)"
        " VALUES ('c1', 'qq', 'bot', 'm1', 'u1', 'u', 0, 'hi')"
    )
    # A commit row must reference a real message row.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO inbound_commits(event_id, chat_key, message_id,"
            " committed_ts, wake_kind) VALUES ('e1', 'c1', 999, 1.0, 'inbound')"
        )
    conn.close()


def test_migration_from_phase1_state_creates_ledger(tmp_path, monkeypatch):
    """A real Phase 1 database (schema.sql only, version 1) upgrades to the
    ledger schema via migration 2 — the same path every existing install
    takes."""
    conn = fresh_conn(tmp_path / "t.db")
    real_migrations = list(schema.MIGRATIONS)
    monkeypatch.setattr(schema, "MIGRATIONS", [])
    apply_schema(conn)
    assert current_version(conn) == 1
    assert not any(
        r[0] in ("inbound_commits", "dispatches")
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    )
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations)
    apply_schema(conn)
    assert current_version(conn) == 15
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}
    assert {"inbound_commits", "dispatches"} <= names
    conn.close()


def test_migration_backfills_only_legacy_pending_nonself_messages(tmp_path, monkeypatch):
    """A deployed v1 database must not silently lose pending work when the
    ledger becomes the default dispatcher. Consumed/self rows are silent;
    only pending non-self rows wake startup recovery."""
    conn = fresh_conn(tmp_path / "t.db")
    real_migrations = list(schema.MIGRATIONS)
    monkeypatch.setattr(schema, "MIGRATIONS", [])
    apply_schema(conn)
    conn.execute(
        "INSERT INTO chats(chat_key, platform, self_id, kind, cursor_msg_id)"
        " VALUES ('c1', 'qq', 'bot', 'group', 1)"
    )
    for msg_id, is_self in (("old", 0), ("pending", 0), ("self", 1)):
        conn.execute(
            "INSERT INTO messages(chat_key, platform, self_id, platform_msg_id,"
            " sender_id, sender_name, is_self, text, recv_ts)"
            " VALUES ('c1', 'qq', 'bot', ?, 'u', 'u', ?, ?, 1.0)",
            (msg_id, is_self, msg_id),
        )
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations)
    apply_schema(conn)
    rows = conn.execute(
        "SELECT message_id, wake_kind, exported, priority"
        " FROM inbound_commits ORDER BY message_id"
    ).fetchall()
    assert rows == [(1, "none", 1, 0), (2, "inbound", 1, 0), (3, "none", 1, 0)]
    conn.close()


def test_migration_v2_to_v5_adds_dispatch_boundary_and_snapshot(tmp_path, monkeypatch):
    """A real v2 ledger database (migration 2 only) upgrades through v3 to
    v4: the dispatches table gains the frozen commit_boundary column, and
    existing rows default to boundary 0 — old markers stay readable."""
    conn = fresh_conn(tmp_path / "t.db")
    real_migrations = list(schema.MIGRATIONS)
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations[:1])
    apply_schema(conn)
    assert current_version(conn) == 2
    cols = {r[1] for r in conn.execute("PRAGMA table_info(dispatches)").fetchall()}
    assert "commit_boundary" not in cols
    # A v2 dispatch row exists before the upgrade.
    conn.execute(
        "INSERT INTO chats(chat_key, platform, self_id, kind)"
        " VALUES ('c1', 'qq', 'bot', 'group')"
    )
    conn.execute(
        "INSERT INTO dispatches(chat_key, cause, wake_kind, started_ts,"
        " expires_at, claimed_ts, cycle_id, start_msg_id, through_msg_id, state)"
        " VALUES ('c1', 'timer', 'timer', 1.0, 100.0, 1.0, 'cy1', 0, 0,"
        " 'prepared')"
    )
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations)
    apply_schema(conn)
    assert current_version(conn) == 15
    cols = {r[1] for r in conn.execute("PRAGMA table_info(dispatches)").fetchall()}
    assert "commit_boundary" in cols
    assert "attached_json" in cols
    assert "settled_ts" in cols
    assert "evaluated_ts" in cols
    assert "snapshot_json" in cols
    commit_cols = {r[1] for r in conn.execute("PRAGMA table_info(inbound_commits)").fetchall()}
    assert "priority" in commit_cols
    # The pre-existing row read back with the default boundary 0 and NULL
    # v4 metadata.
    row = conn.execute(
        "SELECT commit_boundary, attached_json, settled_ts FROM dispatches WHERE id = 1"
    ).fetchone()
    assert row == (0, None, None)
    conn.close()


def test_migration_v3_to_v5_adds_settled_marker_columns(tmp_path, monkeypatch):
    """A real v3 ledger database (migrations 2+3 only) upgrades to v4: the
    dispatches table gains attached_json and settled_ts, and existing rows
    default to NULL — old v3 markers stay readable."""
    conn = fresh_conn(tmp_path / "t.db")
    real_migrations = list(schema.MIGRATIONS)
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations[:2])
    apply_schema(conn)
    assert current_version(conn) == 3
    cols = {r[1] for r in conn.execute("PRAGMA table_info(dispatches)").fetchall()}
    assert "attached_json" not in cols
    assert "settled_ts" not in cols
    # A v3 dispatch row exists before the upgrade.
    conn.execute(
        "INSERT INTO chats(chat_key, platform, self_id, kind)"
        " VALUES ('c1', 'qq', 'bot', 'group')"
    )
    conn.execute(
        "INSERT INTO dispatches(chat_key, cause, wake_kind, started_ts,"
        " expires_at, claimed_ts, cycle_id, start_msg_id, through_msg_id,"
        " commit_boundary, state)"
        " VALUES ('c1', 'inbound', 'inbound', 1.0, 100.0, 1.0, 'cy1', 0, 0,"
        " 5, 'completed')"
    )
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations)
    apply_schema(conn)
    assert current_version(conn) == 15
    cols = {r[1] for r in conn.execute("PRAGMA table_info(dispatches)").fetchall()}
    assert "attached_json" in cols
    assert "settled_ts" in cols
    assert "evaluated_ts" in cols
    assert "snapshot_json" in cols
    # The pre-existing row reads back with NULL v4 metadata (no frozen
    # membership, no settlement timestamp) and its v3 fields intact.
    row = conn.execute(
        "SELECT commit_boundary, attached_json, settled_ts, state"
        " FROM dispatches WHERE id = 1"
    ).fetchone()
    assert row == (5, None, None, "completed")
    conn.close()


def test_migration_v5_to_v6_adds_agent_barrier_columns(tmp_path, monkeypatch):
    """A real v5 ledger database (migrations 2-5 only) upgrades to v6: the
    chats table gains agent_resume_at and wait_streak, and existing rows
    default to NULL/0 — old rows stay fully readable."""
    conn = fresh_conn(tmp_path / "t.db")
    real_migrations = list(schema.MIGRATIONS)
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations[:4])
    apply_schema(conn)
    assert current_version(conn) == 5
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chats)").fetchall()}
    assert "agent_resume_at" not in cols
    assert "wait_streak" not in cols
    # A v5 chat row exists before the upgrade.
    conn.execute(
        "INSERT INTO chats(chat_key, platform, self_id, kind, cursor_msg_id)"
        " VALUES ('c1', 'qq', 'bot', 'group', 3)"
    )
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations)
    apply_schema(conn)
    assert current_version(conn) == 15
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chats)").fetchall()}
    assert "agent_resume_at" in cols
    assert "wait_streak" in cols
    # The pre-existing row reads back with NULL barrier and 0 streak.
    row = conn.execute(
        "SELECT agent_resume_at, wait_streak FROM chats WHERE chat_key = 'c1'"
    ).fetchone()
    assert row == (None, 0)
    conn.close()


def test_apply_is_idempotent(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    apply_schema(conn)  # second boot: no-op, no error
    assert current_version(conn) == 15
    conn.close()


def test_fts5_is_available():
    # Some distro sqlite3 builds omit FTS5; the schema depends on it.
    conn = sqlite3.connect(":memory:")
    ok = conn.execute(
        "SELECT sqlite_compileoption_used('ENABLE_FTS5')"
    ).fetchone()[0]
    conn.close()
    assert ok == 1


# ── Phase 5 knowledge foundation (migration 7) ──────────────────────────────

def test_fresh_schema_has_knowledge_tables(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    # chats: the durable memory watermark and the per-chat profile cursor.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chats)").fetchall()}
    assert "memory_through_msg_id" in cols
    assert "profile_through_msg_id" in cols
    # memories: the source-bounded fields.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    assert {"source_first_msg_id", "source_last_msg_id", "source_hash"} <= cols
    # persons: the profile cursor and the (chat_key, platform_uid) backstop.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(persons)").fetchall()}
    assert "profile_through_msg_id" in cols
    idx = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index'"
        " AND name = 'persons_chat_uid'"
    ).fetchone()
    assert idx is not None and "chat_key" in idx[0] and "platform_uid" in idx[0]
    # canonical memory FTS documents + external-content index.
    for table in ("memory_search_docs", "memory_search_fts"):
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE name = ?", (table,)
        ).fetchone()
        assert row is not None, table
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'memory_search_fts'"
    ).fetchone()[0]
    assert "fts5" in sql and "content=" in sql and "content_rowid=" in sql
    # embedding generations with the one-active partial unique index.
    cols = {r[1] for r in conn.execute(
        "PRAGMA table_info(embedding_generations)"
    ).fetchall()}
    assert {"id", "model", "dim", "state", "created_ts"} <= cols
    idx = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index'"
        " AND name = 'embedding_generations_one_active'"
    ).fetchone()
    assert idx is not None and "WHERE state = 'active'" in idx[0]
    # the chat-scoped vector table replaced the base vec placeholder.
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}
    assert "vectors" in names
    assert "vec" not in names
    cols = {r[1] for r in conn.execute("PRAGMA table_info(vectors)").fetchall()}
    assert {"owner_table", "owner_id", "dim", "model", "generation",
            "source_hash", "blob"} <= cols
    conn.close()


def test_vectors_schema_enforces_dim_and_blob_length(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    conn.execute(
        "INSERT INTO chats(chat_key, platform, self_id, kind)"
        " VALUES ('c1', 'qq', 'bot', 'group')"
    )
    conn.execute(
        "INSERT INTO memories(chat_key, kind, text) VALUES ('c1', 'memory', 'x')"
    )
    # A zero/negative dim is rejected by the CHECK constraint.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO vectors(owner_table, owner_id, dim, model, generation, blob)"
            " VALUES ('memories', 1, 0, 'm', 1, x'00000000')"
        )
    # A blob whose length is not dim * 4 is rejected.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO vectors(owner_table, owner_id, dim, model, generation, blob)"
            " VALUES ('memories', 1, 2, 'm', 1, x'00000000')"
        )
    # The deterministic owner/generation uniqueness: same owner+model in a
    # different generation coexists; the same generation does not.
    conn.execute(
        "INSERT INTO vectors(owner_table, owner_id, dim, model, generation, blob)"
        " VALUES ('memories', 1, 1, 'm', 1, x'00000000')"
    )
    conn.execute(
        "INSERT INTO vectors(owner_table, owner_id, dim, model, generation, blob)"
        " VALUES ('memories', 1, 1, 'm', 2, x'00000000')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO vectors(owner_table, owner_id, dim, model, generation, blob)"
            " VALUES ('memories', 1, 1, 'm', 1, x'00000000')"
        )
    conn.close()


def test_memories_source_range_unique_index(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    conn.execute(
        "INSERT INTO chats(chat_key, platform, self_id, kind)"
        " VALUES ('c1', 'qq', 'bot', 'group')"
    )
    conn.execute(
        "INSERT INTO memories(chat_key, kind, text, source_first_msg_id,"
        " source_last_msg_id, source_hash)"
        " VALUES ('c1', 'memory', 'a', 1, 3, 'h1')"
    )
    # The same source range never duplicates.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO memories(chat_key, kind, text, source_first_msg_id,"
            " source_last_msg_id, source_hash)"
            " VALUES ('c1', 'memory', 'b', 1, 3, 'h2')"
        )
    # A different range in the same chat is fine; legacy NULL ranges are
    # distinct (SQLite treats NULLs as distinct in a unique index).
    conn.execute(
        "INSERT INTO memories(chat_key, kind, text, source_first_msg_id,"
        " source_last_msg_id, source_hash)"
        " VALUES ('c1', 'memory', 'c', 4, 5, 'h3')"
    )
    conn.execute(
        "INSERT INTO memories(chat_key, kind, text) VALUES ('c1', 'memory', 'legacy')"
    )
    conn.close()


def test_migration_v6_to_v7_adds_knowledge_schema(tmp_path, monkeypatch):
    """A real v6 database (migrations 2-6 only) upgrades to v7: the
    knowledge columns/tables land and legacy rows survive."""
    conn = fresh_conn(tmp_path / "t.db")
    real_migrations = list(schema.MIGRATIONS)
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations[:5])
    apply_schema(conn)
    assert current_version(conn) == 6
    # Legacy data: a chat, a memory, a person, and a vec row.
    conn.execute(
        "INSERT INTO chats(chat_key, platform, self_id, kind)"
        " VALUES ('c1', 'qq', 'bot', 'group')"
    )
    conn.execute(
        "INSERT INTO memories(chat_key, kind, text, strength)"
        " VALUES ('c1', 'memory', 'legacy text', 1.0)"
    )
    conn.execute(
        "INSERT INTO persons(person_key, chat_key, platform_uid, names_json)"
        " VALUES ('p1', 'c1', 'u1', '[]')"
    )
    conn.execute(
        "INSERT INTO vec(owner_table, owner_id, dim, model, blob)"
        " VALUES ('memories', 1, 2, 'm1', x'cdcccc3dcdcc4c3e')"
    )
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations)
    apply_schema(conn)
    assert current_version(conn) == 15
    # chats columns.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chats)").fetchall()}
    assert "memory_through_msg_id" in cols
    assert "profile_through_msg_id" in cols
    # memories columns + legacy row preserved with NULL source fields.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    assert {"source_first_msg_id", "source_last_msg_id", "source_hash"} <= cols
    row = conn.execute(
        "SELECT text, source_first_msg_id, source_last_msg_id, source_hash"
        " FROM memories WHERE id = 1"
    ).fetchone()
    assert row == ("legacy text", None, None, None)
    # persons columns + unique index + legacy row preserved.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(persons)").fetchall()}
    assert "profile_through_msg_id" in cols
    idx = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index'"
        " AND name = 'persons_chat_uid'"
    ).fetchone()
    assert idx is not None
    row = conn.execute(
        "SELECT person_key, chat_key, platform_uid FROM persons"
    ).fetchone()
    assert row == ("p1", "c1", "u1")
    # New tables exist; the vec placeholder was replaced.
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}
    assert {"memory_search_docs", "memory_search_fts",
            "embedding_generations", "vectors"} <= names
    assert "vec" not in names
    # The legacy vec row was preserved into generation 0.
    row = conn.execute(
        "SELECT owner_table, owner_id, dim, model, generation, source_hash"
        " FROM vectors"
    ).fetchone()
    assert row == ("memories", 1, 2, "m1", 0, None)
    conn.close()


def test_migration_v6_to_v7_resolves_person_duplicates(tmp_path, monkeypatch):
    """Legacy duplicate (chat_key, platform_uid) person rows are resolved
    deterministically BEFORE the unique index: the smallest person_key
    survives."""
    conn = fresh_conn(tmp_path / "t.db")
    real_migrations = list(schema.MIGRATIONS)
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations[:5])
    apply_schema(conn)
    conn.execute(
        "INSERT INTO chats(chat_key, platform, self_id, kind)"
        " VALUES ('c1', 'qq', 'bot', 'group')"
    )
    conn.execute(
        "INSERT INTO persons(person_key, chat_key, platform_uid, names_json)"
        " VALUES ('p2', 'c1', 'u1', '[]')"
    )
    conn.execute(
        "INSERT INTO persons(person_key, chat_key, platform_uid, names_json)"
        " VALUES ('p1', 'c1', 'u1', '[]')"
    )
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations)
    apply_schema(conn)
    assert current_version(conn) == 15
    rows = conn.execute(
        "SELECT person_key FROM persons WHERE chat_key = 'c1' AND platform_uid = 'u1'"
    ).fetchall()
    assert rows == [("p1",)]
    conn.close()


def test_migration_v6_to_v7_skips_invalid_legacy_vec_rows(tmp_path, monkeypatch):
    """The data-preserving vec copy keeps only rows that satisfy the new
    constraints (positive dim, exact float32 blob length); invalid legacy
    rows cannot block the migration."""
    conn = fresh_conn(tmp_path / "t.db")
    real_migrations = list(schema.MIGRATIONS)
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations[:5])
    apply_schema(conn)
    conn.execute(
        "INSERT INTO chats(chat_key, platform, self_id, kind)"
        " VALUES ('c1', 'qq', 'bot', 'group')"
    )
    conn.execute(
        "INSERT INTO vec(owner_table, owner_id, dim, model, blob)"
        " VALUES ('memories', 1, 2, 'm1', x'cdcccc3dcdcc4c3e')"
    )
    conn.execute(
        "INSERT INTO vec(owner_table, owner_id, dim, model, blob)"
        " VALUES ('memories', 2, 0, 'bad', x'00000000')"
    )
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations)
    apply_schema(conn)
    assert current_version(conn) == 15
    rows = conn.execute(
        "SELECT owner_id, dim, model, generation FROM vectors"
    ).fetchall()
    assert rows == [(1, 2, "m1", 0)]
    conn.close()


def test_migration_v7_to_v8_preserves_legacy_knowledge_state(tmp_path, monkeypatch):
    """A real v7 database (migrations 2-7 only) upgrades to v8: legacy
    memories/vectors/generations survive; legacy generations become
    inactive/legacy derived state; the observed watermark column and the
    FTS bootstrap/backlog table land."""
    conn = fresh_conn(tmp_path / "t.db")
    real_migrations = list(schema.MIGRATIONS)
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations[:6])
    apply_schema(conn)
    assert current_version(conn) == 7
    # Legacy data: a chat, a memory, an ACTIVE generation, and a vector row.
    conn.execute(
        "INSERT INTO chats(chat_key, platform, self_id, kind)"
        " VALUES ('c1', 'qq', 'bot', 'group')"
    )
    conn.execute(
        "INSERT INTO memories(chat_key, kind, text, strength)"
        " VALUES ('c1', 'memory', 'legacy text', 1.0)"
    )
    conn.execute(
        "INSERT INTO embedding_generations(model, dim, state, created_ts)"
        " VALUES ('m1', 2, 'active', 1.0)"
    )
    conn.execute(
        "INSERT INTO vectors(owner_table, owner_id, dim, model, generation, blob)"
        " VALUES ('memories', 1, 2, 'm1', 1, x'cdcccc3dcdcc4c3e')"
    )
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations)
    apply_schema(conn)
    assert current_version(conn) == 15
    # chats: the observed memory watermark column.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chats)").fetchall()}
    assert "memory_observed_through_msg_id" in cols
    # memory_fts_state: the idempotent bootstrap/backlog table.
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE name = 'memory_fts_state'"
    ).fetchone()
    assert row is not None
    # The legacy generation is preserved ONLY as inactive/legacy derived
    # state: revision 'legacy', forced inactive, unique legacy space_id.
    row = conn.execute(
        "SELECT model, revision, state, space_id, vector_revision"
        " FROM embedding_generations"
    ).fetchone()
    assert row == ("m1", "legacy", "inactive", "m1@legacy:1", 0)
    # The legacy vector row survives untouched.
    row = conn.execute(
        "SELECT owner_id, dim, model, generation FROM vectors"
    ).fetchone()
    assert row == (1, 2, "m1", 1)
    # The rebuilt generations table admits the 'building' state.
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'embedding_generations'"
    ).fetchone()[0]
    assert "building" in sql
    conn.close()


# ── MIGRATIONS ──────────────────────────────────────────────────────────────

def test_migration_applies_and_bumps_version(tmp_path, monkeypatch):
    monkeypatch.setattr(
        schema, "MIGRATIONS", ["CREATE TABLE upgrade_probe (id INTEGER PRIMARY KEY);"]
    )
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    assert current_version(conn) == 2
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE name = 'upgrade_probe'"
    ).fetchone()
    assert row is not None
    conn.close()


def test_migration_applies_to_existing_database(tmp_path, monkeypatch):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    assert current_version(conn) == 15
    # Ship one NEW migration (targeting version 16); all current entries are
    # preserved so the test remains meaningful as schema versions advance.
    monkeypatch.setattr(
        schema,
        "MIGRATIONS",
        list(schema.MIGRATIONS) + ["CREATE TABLE upgrade_probe (id INTEGER PRIMARY KEY);"],
    )
    apply_schema(conn)
    # Only the entries targeting a version ABOVE the current one apply.
    assert current_version(conn) == 16
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE name = 'upgrade_probe'"
    ).fetchone()
    assert row is not None
    conn.close()


def test_multiple_migrations_apply_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(
        schema,
        "MIGRATIONS",
        [
            "CREATE TABLE upgrade_a (id INTEGER PRIMARY KEY);",
            "CREATE TABLE upgrade_b (id INTEGER PRIMARY KEY);",
        ],
    )
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    assert current_version(conn) == 3
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}
    assert {"upgrade_a", "upgrade_b"} <= names
    conn.close()


def test_failed_migration_rolls_back_completely(tmp_path, monkeypatch):
    # Boot a healthy database first, then ship a broken migration that
    # targets a version ABOVE the current one.
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    assert current_version(conn) == 15
    monkeypatch.setattr(
        schema,
        "MIGRATIONS",
        list(schema.MIGRATIONS)
        + [
            "CREATE TABLE upgrade_a (id INTEGER PRIMARY KEY);",
            "CREATE TABLE upgrade_d (id INTEGER",
        ],
    )
    with pytest.raises(sqlite3.OperationalError):
        apply_schema(conn)
    # Version unchanged and no partial table survived the rollback.
    assert current_version(conn) == 15
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE name = 'upgrade_a'"
    ).fetchone()
    assert row is None
    conn.close()


def test_schema_sql_loads_and_contains_core_tables():
    sql = schema.load_schema()
    for table in ("chats", "messages", "outbox", "cycles", "claims", "kv"):
        assert f"CREATE TABLE {table}" in sql
    assert "UNIQUE (platform, self_id, platform_msg_id)" in sql


# ── Phase 6 adaptive foundation (migration 9) ────────────────────────────────

def test_fresh_schema_has_adaptive_tables(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}
    assert {
        "learner_state", "learner_runs", "record_sources",
        "record_search_docs", "record_search_fts",
        "record_exposures", "record_feedback",
    } <= names
    # records: the provenance columns and the retired flag.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(records)").fetchall()}
    assert {"content_hash", "source_first_msg_id", "source_last_msg_id",
            "retired"} <= cols
    # The adaptive record identity index (the merge key).
    idx = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index'"
        " AND name = 'records_adaptive_identity'"
    ).fetchone()
    assert idx is not None and "content_hash IS NOT NULL" in idx[0]
    # learner_runs: the one-prepared-per-(chat, learner) partial unique index.
    idx = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index'"
        " AND name = 'learner_runs_one_prepared'"
    ).fetchone()
    assert idx is not None and "WHERE state = 'prepared'" in idx[0]
    # The canonical record FTS is external-content.
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'record_search_fts'"
    ).fetchone()[0]
    assert "fts5" in sql and "content=" in sql and "content_rowid=" in sql
    # record_exposures: idempotent (record_id, run_id) uniqueness.
    cols = {
        r[2] for r in conn.execute(
            "PRAGMA index_info('sqlite_autoindex_record_exposures_1')"
        ).fetchall()
    }
    assert cols == {"record_id", "run_id"}
    conn.close()


def test_migration_v8_to_v9_preserves_legacy_records(tmp_path, monkeypatch):
    """A real v8 database (migrations 2-8 only) upgrades through v9 to v11:
    legacy records/emoji stay legacy/untrusted (NULL content_hash/source
    range) and remain fully readable."""
    conn = fresh_conn(tmp_path / "t.db")
    real_migrations = list(schema.MIGRATIONS)
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations[:7])
    apply_schema(conn)
    assert current_version(conn) == 8
    # Legacy data: a chat, a legacy record, and an emoji row.
    conn.execute(
        "INSERT INTO chats(chat_key, platform, self_id, kind)"
        " VALUES ('c1', 'qq', 'bot', 'group')"
    )
    conn.execute(
        "INSERT INTO records(chat_key, learner, payload_json, weight, uses)"
        " VALUES ('c1', 'personality', '{\"text\":\"legacy\"}', 1.0, 0)"
    )
    conn.execute(
        "INSERT INTO emoji(sha256, desc) VALUES ('abc', 'smile')"
    )
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations)
    apply_schema(conn)
    assert current_version(conn) == 15
    # The legacy record reads back with NULL provenance and retired = 0 —
    # still legacy/untrusted (the adaptive surface excludes NULL
    # content_hash rows).
    row = conn.execute(
        "SELECT content_hash, source_first_msg_id, source_last_msg_id, retired"
        " FROM records WHERE id = 1"
    ).fetchone()
    assert row == (None, None, None, 0)
    # The emoji row is untouched.
    row = conn.execute("SELECT sha256, desc FROM emoji").fetchone()
    assert row == ("abc", "smile")
    conn.close()


def test_fresh_and_upgraded_schemas_converge(tmp_path, monkeypatch):
    """A fresh database (schema.sql + all migrations) and an upgraded v10
    database (schema.sql + migrations 2-10, then 11/12) converge to the same
    schema: identical table/index sets and identical records/media_assets
    columns."""
    fresh = fresh_conn(tmp_path / "fresh.db")
    apply_schema(fresh)
    upgraded = fresh_conn(tmp_path / "upgraded.db")
    real_migrations = list(schema.MIGRATIONS)
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations[:9])
    apply_schema(upgraded)
    assert current_version(upgraded) == 10
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations)
    apply_schema(upgraded)
    assert current_version(upgraded) == 15

    def schema_snapshot(conn):
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        indexes = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        record_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(records)").fetchall()
        }
        media_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(media_assets)").fetchall()
        }
        return tables, indexes, record_cols, media_cols

    assert schema_snapshot(fresh) == schema_snapshot(upgraded)
    fresh.close()
    upgraded.close()


# ── Phase 6 P6.5 media catalog (migration 10) ───────────────────────────────

def test_fresh_schema_has_media_catalog_table(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}
    assert "media_assets" in names
    cols = {r[1] for r in conn.execute("PRAGMA table_info(media_assets)").fetchall()}
    assert {
        "id", "chat_key", "kind", "cache_key", "sha256", "mime", "width",
        "height", "description", "source_message_id", "source_sender_id",
        "source_sender_name", "source_ts", "safety_status", "safety_version",
        "approved_ts", "revoked_ts", "uses", "last_used_ts", "created_ts",
    } <= cols
    # The unique (chat, kind, sha256) identity (autoindex columns).
    idx_cols = {
        r[2] for r in conn.execute(
            "PRAGMA index_info('sqlite_autoindex_media_assets_1')"
        ).fetchall()
    }
    assert idx_cols == {"chat_key", "kind", "sha256"}
    # The partial indexes make the approved/pending scans cheap.
    idx = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index'"
        " AND name = 'media_assets_approved'"
    ).fetchone()
    assert idx is not None and "WHERE safety_status = 'approved'" in idx[0]
    idx = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index'"
        " AND name = 'media_assets_pending'"
    ).fetchone()
    assert idx is not None and "WHERE safety_status = 'pending'" in idx[0]
    # The kind and safety_status CHECK constraints.
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'media_assets'"
    ).fetchone()[0]
    assert "sticker" in sql and "image" in sql
    assert "pending" in sql and "approved" in sql
    assert "rejected" in sql and "revoked" in sql
    conn.close()


def test_media_assets_unique_chat_kind_sha256(tmp_path):
    conn = fresh_conn(tmp_path / "t.db")
    apply_schema(conn)
    conn.execute(
        "INSERT INTO chats(chat_key, platform, self_id, kind)"
        " VALUES ('c1', 'qq', 'bot', 'group')"
    )
    conn.execute(
        "INSERT INTO media_assets(chat_key, kind, cache_key, sha256, mime)"
        " VALUES ('c1', 'sticker', 'ck1', 'sha1', 'image/gif')"
    )
    # The same (chat, kind, sha256) never duplicates.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO media_assets(chat_key, kind, cache_key, sha256, mime)"
            " VALUES ('c1', 'sticker', 'ck2', 'sha1', 'image/gif')"
        )
    # A different kind or a different chat is a distinct row.
    conn.execute(
        "INSERT INTO media_assets(chat_key, kind, cache_key, sha256, mime)"
        " VALUES ('c1', 'image', 'ck1', 'sha1', 'image/png')"
    )
    conn.execute(
        "INSERT INTO chats(chat_key, platform, self_id, kind)"
        " VALUES ('c2', 'qq', 'bot', 'group')"
    )
    conn.execute(
        "INSERT INTO media_assets(chat_key, kind, cache_key, sha256, mime)"
        " VALUES ('c2', 'sticker', 'ck1', 'sha1', 'image/gif')"
    )
    conn.close()


def test_migration_v9_to_v10_preserves_legacy_emoji(tmp_path, monkeypatch):
    """A real v9 database (migrations 2-9 only) upgrades to v11: the
    media_assets table lands and existing global emoji rows remain
    legacy/untrusted — untouched and fully readable."""
    conn = fresh_conn(tmp_path / "t.db")
    real_migrations = list(schema.MIGRATIONS)
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations[:8])
    apply_schema(conn)
    assert current_version(conn) == 9
    # Legacy data: a chat and a global emoji row.
    conn.execute(
        "INSERT INTO chats(chat_key, platform, self_id, kind)"
        " VALUES ('c1', 'qq', 'bot', 'group')"
    )
    conn.execute(
        "INSERT INTO emoji(sha256, desc, platform_ref_json, uses, last_used_ts)"
        " VALUES ('abc', 'smile', '{\"file\":\"sticker.gif\"}', 3, 1.0)"
    )
    monkeypatch.setattr(schema, "MIGRATIONS", real_migrations)
    apply_schema(conn)
    assert current_version(conn) == 15
    # The legacy emoji row is untouched.
    row = conn.execute(
        "SELECT sha256, desc, platform_ref_json, uses, last_used_ts FROM emoji"
    ).fetchone()
    assert row == ("abc", "smile", '{"file":"sticker.gif"}', 3, 1.0)
    # The media catalog table exists and is empty.
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}
    assert "media_assets" in names
    assert conn.execute("SELECT COUNT(*) FROM media_assets").fetchone()[0] == 0
    conn.close()
