"""Schema loading and versioned migration.

The full M0 schema ships in ``schema.sql`` and is applied once at initial
creation (``PRAGMA user_version`` 0 -> 1). Everything after that is a
compact entry in ``MIGRATIONS`` — a list of SQL strings, each bumping
``user_version`` by one — so the milestone table never accumulates a pile
of dev-time migration files.

``apply_schema`` is idempotent and transactional: DDL is transactional in
SQLite, so a failed migration rolls back completely (no partial tables, no
version bump).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# The M0 schema is version 1. Each MIGRATIONS entry upgrades to the next
# version: MIGRATIONS[0] -> 2, MIGRATIONS[1] -> 3, ...
BASE_VERSION = 1

# Post-initial upgrades. Append-only; never edit an entry once shipped.
#
# Migration 2 — the durable dispatch ledger (frozen Oracle advisory, second
# exceptional Gate 2 authorization): inbound_commits records each committed
# inbound event's monotonic sequence, event ID, chat/message identity, wake
# kind, pending count, and eventual dispatch membership; dispatches records
# the total order of prepared and completed inbound/timer/startup/
# busy-recovery dispatches. Both tables carry an exported flag for the
# at-least-once JSONL marker export (append marker, then mark exported;
# readers dedupe by (record_type, sequence)). The partial unique index
# enforces at most ONE prepared dispatch per chat; the partial indexes make
# unassigned-commit scans and unexported-marker scans cheap.
Migration = str | Callable[[Any], None]


def _migration_13_learner_schedule(conn: Any) -> None:
    """Install learner cadence state as a real, versioned migration.

    SQLite has no portable ``ADD COLUMN IF NOT EXISTS``.  Inspecting the
    table while this *pending migration* is executing is deterministic and
    avoids the old divergent boot-time repair path.  Existing values are
    preserved; newly added fields receive their declared defaults.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(learner_state)")}
    if "cadence_s" not in columns:
        conn.execute("ALTER TABLE learner_state ADD COLUMN cadence_s REAL")
    if "next_due_ts" not in columns:
        conn.execute("ALTER TABLE learner_state ADD COLUMN next_due_ts REAL")
    if "failure_streak" not in columns:
        conn.execute(
            "ALTER TABLE learner_state ADD COLUMN failure_streak INTEGER NOT NULL DEFAULT 0"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS learner_state_due "
        "ON learner_state(learner, next_due_ts, chat_key)"
    )

MIGRATIONS: list[Migration] = [
    """
    CREATE TABLE inbound_commits (
        id            INTEGER PRIMARY KEY,  -- monotonic commit sequence (CommitSeq)
        event_id      TEXT NOT NULL UNIQUE, -- stable event id, generated before recording
        chat_key      TEXT NOT NULL REFERENCES chats(chat_key),
        message_id    INTEGER NOT NULL REFERENCES messages(id),
        committed_ts  REAL NOT NULL,        -- absolute epoch seconds
        wake_kind     TEXT NOT NULL DEFAULT 'inbound'
                      CHECK (wake_kind IN
                             ('inbound','timer','startup','busy_recovery','none')),
        pending_count INTEGER,              -- atomic pending count at commit time
        dispatch_id   INTEGER REFERENCES dispatches(id),  -- eventual membership
        exported      INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX inbound_commits_unassigned
        ON inbound_commits(dispatch_id) WHERE dispatch_id IS NULL;
    CREATE INDEX inbound_commits_chat ON inbound_commits(chat_key, id);
    CREATE INDEX inbound_commits_unexported
        ON inbound_commits(exported) WHERE exported = 0;

    CREATE TABLE dispatches (
        id             INTEGER PRIMARY KEY, -- monotonic dispatch id (DispatchId)
        chat_key       TEXT NOT NULL REFERENCES chats(chat_key),
        cause          TEXT NOT NULL
                       CHECK (cause IN
                              ('inbound','timer','startup','busy_recovery')),
        wake_kind      TEXT NOT NULL DEFAULT 'inbound'
                       CHECK (wake_kind IN
                              ('inbound','timer','startup','busy_recovery','none')),
        scheduled_ts   REAL,                -- timer deadline that triggered the dispatch
        started_ts     REAL NOT NULL,       -- claim start, absolute epoch
        expires_at     REAL NOT NULL,       -- mandatory finite lease
        claimed_ts     REAL NOT NULL,       -- when begin_dispatch ran
        cycle_id       TEXT NOT NULL,
        start_msg_id   INTEGER NOT NULL,    -- chat cursor at dispatch time
        through_msg_id INTEGER NOT NULL,    -- fixed-through boundary
        state          TEXT NOT NULL DEFAULT 'prepared'
                       CHECK (state IN ('prepared','completed','released','expired')),
        trace_json     TEXT,
        exported       INTEGER NOT NULL DEFAULT 0
    );
    CREATE UNIQUE INDEX dispatches_one_prepared
        ON dispatches(chat_key) WHERE state = 'prepared';
    CREATE INDEX dispatches_unexported
        ON dispatches(exported) WHERE exported = 0;
    """,
    # Migration 3 — the frozen dispatch boundary/scheduled metadata (the
    # additive core gap before ledger adoption): dispatches gains
    # ``commit_boundary``, the frozen maximum inbound commit sequence at
    # begin_dispatch time, so replay can reconstruct the exact attachment
    # boundary independent of JSONL marker order (a timer dispatch that
    # wrote first excludes a later commit even when the commit marker is
    # exported first). The scheduled time is the existing ``scheduled_ts``
    # column, exported as ``scheduled_for`` in the dispatch marker. Old
    # rows default to boundary 0 and remain fully readable.
    """
    ALTER TABLE dispatches ADD COLUMN commit_boundary INTEGER NOT NULL DEFAULT 0;
    """,
    # Migration 4 — the replayable settled-dispatch marker contract (the
    # additive gap before App/Replay adoption): dispatches gains
    # ``attached_json``, the exact attached inbound CommitSeq tuple frozen
    # at begin_dispatch in the SAME transaction (so a later released or
    # expired-detached dispatch remains replayable with its exact
    # membership even after the live ``inbound_commits.dispatch_id`` rows
    # are detached), and ``settled_ts``, the absolute-epoch settlement time
    # written by ``settle_dispatch`` (the dispatch/evaluation timestamp the
    # marker exports). Old rows default to NULL and remain fully readable.
    """
    ALTER TABLE dispatches ADD COLUMN attached_json TEXT;
    ALTER TABLE dispatches ADD COLUMN settled_ts REAL;
    """,
    # Migration 5 — direct third-exception ledger repair. Commit rows gain a
    # durable priority bit; settled dispatches retain the exact evaluation
    # timestamp and serialized gate snapshot needed for replay. Existing
    # messages are backfilled exactly once: already-consumed/self rows are
    # ledger-silent, while legacy pending non-self rows become startup work.
    # They are marked exported because their pre-ledger raw event lines lack a
    # stable EventId; exact marker replay reports that incomplete history rather
    # than fabricating it.
    """
    ALTER TABLE inbound_commits ADD COLUMN priority INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE dispatches ADD COLUMN evaluated_ts REAL;
    ALTER TABLE dispatches ADD COLUMN snapshot_json TEXT;
    INSERT OR IGNORE INTO inbound_commits(
        event_id, chat_key, message_id, committed_ts, wake_kind,
        pending_count, dispatch_id, exported, priority
    )
    SELECT
        'legacy:' || m.id,
        m.chat_key,
        m.id,
        COALESCE(m.recv_ts, 0.0),
        CASE
            WHEN m.is_self = 1 OR m.id <= COALESCE(c.cursor_msg_id, 0)
            THEN 'none'
            ELSE 'inbound'
        END,
        NULL,
        NULL,
        1,
        0
    FROM messages AS m
    JOIN chats AS c ON c.chat_key = m.chat_key
    WHERE NOT EXISTS (
        SELECT 1 FROM inbound_commits AS ic WHERE ic.message_id = m.id
    );
    """,
    # Migration 6 — the durable agent barrier (frozen Oracle advisory,
    # Phase 3 Gate 3): chats gains ``agent_resume_at`` (the absolute-epoch
    # time before which no agent may run — a wait/retry defer barrier) and
    # ``wait_streak`` (the consecutive-wait counter, incremented by a wait
    # defer and reset by a terminal finish). Old rows default to NULL/0 and
    # remain fully readable.
    """
    ALTER TABLE chats ADD COLUMN agent_resume_at REAL;
    ALTER TABLE chats ADD COLUMN wait_streak INTEGER NOT NULL DEFAULT 0;
    """,
    # Migration 7 — the Phase 5 knowledge foundation (frozen Oracle
    # advisory ses_fc73526f6ffeYTfNXN1aQH9sRn): durable source-bounded
    # memory fields, canonical CJK-bigram FTS token documents, per-chat
    # person uniqueness and profile cursors, embedding generations, and
    # the chat-scoped vector table. Knowledge rows/FTS are authoritative
    # local state; vectors are rebuildable derived state. No provider or
    # network calls in any migration.
    #
    # 1. chats gains the durable memory watermark (``memory_through_msg_id``)
    #    and the per-chat profile cursor (``profile_through_msg_id``). Both
    #    are cursor-owned: the memory watermark advances only inside
    #    ``commit_memory_source``, the profile cursor only inside
    #    ``cas_person_profile``.
    # 2. memories gains the durable source-bounded fields: the fixed source
    #    range (first/last local message ids), the deterministic
    #    ``source_hash``, and a unique index so source ranges never
    #    duplicate (a later summarizer CAS-commits each range exactly
    #    once). Legacy rows keep NULL source fields and remain fully
    #    readable (SQLite treats NULLs as distinct in a unique index).
    # 3. persons gains the per-chat profile cursor and a (chat_key,
    #    platform_uid) uniqueness backstop. Legacy duplicates are resolved
    #    deterministically BEFORE the unique index: for each (chat_key,
    #    platform_uid) group the row with the smallest person_key survives.
    # 4. The canonical memory FTS documents: the base external-content
    #    ``memory_fts`` design cannot reproduce a CJK-bigram rebuild
    #    exactly (raw text is re-tokenized at index time).
    #    ``memory_search_docs`` persists the tokenized text per memory;
    #    ``memory_search_fts`` is external-content over those canonical
    #    docs, so a rebuild reproduces exactly. Legacy memory rows are
    #    preserved; the runtime repo owns an explicit local rebuild/
    #    backfill method (``rebuild_memory_fts``) that tokenizes existing
    #    raw text with the repo's central ``bigram_tokenize`` and
    #    transactionally rebuilds the index.
    # 5. embedding_generations: model/dimension generations with an
    #    activation state and no provider side effects. At most ONE active
    #    generation (the partial unique index); activating a generation
    #    atomically deactivates the previous active one.
    # 6. The chat-scoped vector table replaces the base ``vec`` placeholder
    #    through a data-preserving migration: valid legacy rows are copied
    #    into generation 0, then the placeholder is dropped. The new table
    #    enforces a positive dim, an exact float32 blob length
    #    (``length(blob) = dim * 4``), and deterministic owner/generation
    #    uniqueness so old and new generations coexist.
    """
    ALTER TABLE chats ADD COLUMN memory_through_msg_id INTEGER;
    ALTER TABLE chats ADD COLUMN profile_through_msg_id INTEGER;

    ALTER TABLE memories ADD COLUMN source_first_msg_id INTEGER;
    ALTER TABLE memories ADD COLUMN source_last_msg_id INTEGER;
    ALTER TABLE memories ADD COLUMN source_hash TEXT;
    CREATE UNIQUE INDEX memories_source_range
        ON memories(chat_key, source_first_msg_id, source_last_msg_id);

    ALTER TABLE persons ADD COLUMN profile_through_msg_id INTEGER;
    DELETE FROM persons WHERE person_key NOT IN (
        SELECT MIN(person_key) FROM persons GROUP BY chat_key, platform_uid
    );
    CREATE UNIQUE INDEX persons_chat_uid ON persons(chat_key, platform_uid);

    CREATE TABLE memory_search_docs (
        id        INTEGER PRIMARY KEY,
        chat_key  TEXT NOT NULL,
        memory_id INTEGER NOT NULL REFERENCES memories(id),
        tokens    TEXT NOT NULL
    );
    CREATE INDEX memory_search_docs_chat ON memory_search_docs(chat_key, memory_id);
    CREATE VIRTUAL TABLE memory_search_fts USING fts5(
        tokens,
        content='memory_search_docs',
        content_rowid='id'
    );

    CREATE TABLE embedding_generations (
        id         INTEGER PRIMARY KEY,
        model      TEXT NOT NULL,
        dim        INTEGER NOT NULL CHECK (dim > 0),
        state      TEXT NOT NULL DEFAULT 'inactive'
                   CHECK (state IN ('active','inactive')),
        created_ts REAL,
        UNIQUE (model, dim)
    );
    CREATE UNIQUE INDEX embedding_generations_one_active
        ON embedding_generations(state) WHERE state = 'active';

    CREATE TABLE vectors (
        id          INTEGER PRIMARY KEY,
        owner_table TEXT NOT NULL,
        owner_id    INTEGER NOT NULL,
        dim         INTEGER NOT NULL CHECK (dim > 0),
        model       TEXT NOT NULL,
        generation  INTEGER NOT NULL,
        source_hash TEXT,
        blob        BLOB NOT NULL CHECK (length(blob) = dim * 4),
        UNIQUE (owner_table, owner_id, model, generation)
    );
    INSERT INTO vectors(owner_table, owner_id, dim, model, generation, source_hash, blob)
    SELECT owner_table, owner_id, dim, model, 0, NULL, blob FROM vec
    WHERE dim > 0 AND length(blob) = dim * 4;
    DROP TABLE vec;
    """,
    # Migration 8 — the Phase 5 Gate 5 retrieval/storage remediation (frozen
    # Oracle advisory): observed memory watermark support, idempotent
    # canonical memory FTS bootstrap/backlog state, and the embedding space
    # identity (space_id + explicit model revision + building state) with a
    # durable vector mutation revision. No provider/network calls.
    #
    # 1. chats gains the observed memory watermark
    #    (``memory_observed_through_msg_id``): the durable cursor of source
    #    rows observed for summarization, distinct from the summarized
    #    watermark (``memory_through_msg_id``). Old rows default to NULL and
    #    remain fully readable.
    # 2. ``memory_fts_state`` is the idempotent canonical memory FTS
    #    bootstrap/backlog state: per chat, whether the canonical token
    #    documents have been bootstrapped and the backlog cursor. A rebuild/
    #    backfill is idempotent — re-running it reproduces the same index.
    # 3. embedding_generations is REBUILT (the state CHECK must admit
    #    ``building``, which SQLite cannot ALTER): it gains ``space_id`` (the
    #    canonical embedding space identity, unique per generation),
    #    ``revision`` (the explicit model revision from the embed profile),
    #    the ``building|active|inactive`` state, and ``vector_revision`` (the
    #    durable vector mutation revision, bumped on every vector write so a
    #    direct repo mutation is visible on the next search). Valid legacy
    #    generations are preserved ONLY as inactive/legacy derived state
    #    (space_id ``model@legacy:<id>``, revision ``legacy``, forced
    #    inactive) — they never become the active generation because they
    #    lack a proper space identity.
    """
    ALTER TABLE chats ADD COLUMN memory_observed_through_msg_id INTEGER;

    CREATE TABLE memory_fts_state (
        chat_key TEXT PRIMARY KEY,
        bootstrapped INTEGER NOT NULL DEFAULT 0,
        backlog_through_msg_id INTEGER
    );

    CREATE TABLE embedding_generations_new (
        id INTEGER PRIMARY KEY,
        space_id TEXT NOT NULL,
        model TEXT NOT NULL,
        revision TEXT NOT NULL,
        dim INTEGER NOT NULL CHECK (dim > 0),
        state TEXT NOT NULL DEFAULT 'inactive'
               CHECK (state IN ('building','active','inactive')),
        created_ts REAL,
        vector_revision INTEGER NOT NULL DEFAULT 0,
        UNIQUE (space_id)
    );
    INSERT INTO embedding_generations_new(
        id, space_id, model, revision, dim, state, created_ts, vector_revision
    )
    SELECT id, model || '@legacy:' || id, model, 'legacy', dim, 'inactive',
           created_ts, 0
    FROM embedding_generations;
    DROP TABLE embedding_generations;
    ALTER TABLE embedding_generations_new RENAME TO embedding_generations;
    CREATE UNIQUE INDEX embedding_generations_one_active
        ON embedding_generations(state) WHERE state = 'active';
    """,
    # Migration 9 — the Phase 6 adaptive foundation (frozen Oracle
    # advisory): durable learner state/runs, record provenance + canonical
    # record FTS documents, and record exposure/feedback. Existing
    # records/emoji stay legacy/untrusted (NULL content_hash/source range —
    # excluded from the adaptive surface); the new migration columns are
    # nullable. No destructive rewrite.
    #
    # 1. records gains the provenance columns (content_hash, the fixed
    #    source range) and the retired flag, plus the chat/learner and
    #    retired-partial indexes. The partial unique index is the adaptive
    #    record identity: (chat_key, learner, content_hash) — the merge key
    #    of the CAS commit. Legacy rows keep NULL content_hash/source
    #    fields and remain fully readable; the adaptive surface excludes
    #    them.
    # 2. learner_state: the durable per-(chat, learner) watermark (the
    #    summarized cursor), the observed watermark snapshot, and the last
    #    settled run id. The watermark advances only inside
    #    ``commit_learner_source``.
    # 3. learner_runs: the durable per-(chat, learner) run ledger with a
    #    mandatory finite lease, the fixed local-row boundary, the exact
    #    source hash, and the settled outcome (success/malformed/cancelled).
    #    The partial unique index enforces at most ONE prepared run per
    #    (chat, learner).
    # 4. record_sources: the opaque source mapping for each adaptive record
    #    (record_id -> source range + hash). Model source references are
    #    opaque refs mapped only in the current batch.
    # 5. The canonical record FTS documents: record_search_docs persists
    #    the tokenized text per record; record_search_fts is
    #    external-content over those canonical docs, so a rebuild
    #    reproduces exactly.
    # 6. record_exposures: idempotent (record_id, run_id) exposure rows.
    # 7. record_feedback: bounded effect feedback with the code-owned
    #    reweight.
    """
    ALTER TABLE records ADD COLUMN content_hash TEXT;
    ALTER TABLE records ADD COLUMN source_first_msg_id INTEGER;
    ALTER TABLE records ADD COLUMN source_last_msg_id INTEGER;
    ALTER TABLE records ADD COLUMN retired INTEGER NOT NULL DEFAULT 0;
    CREATE UNIQUE INDEX records_adaptive_identity
        ON records(chat_key, learner, content_hash) WHERE content_hash IS NOT NULL;
    CREATE INDEX records_learner_chat ON records(learner, chat_key, id);
    CREATE INDEX records_active
        ON records(learner, chat_key, retired) WHERE retired = 0;

    CREATE TABLE learner_state (
        chat_key TEXT NOT NULL REFERENCES chats(chat_key),
        learner TEXT NOT NULL,
        watermark_msg_id INTEGER,
        observed_watermark_msg_id INTEGER,
        last_run_id INTEGER,
        updated_ts REAL,
        cadence_s REAL,
        next_due_ts REAL,
        failure_streak INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (chat_key, learner)
    );

    CREATE TABLE learner_runs (
        id INTEGER PRIMARY KEY,
        chat_key TEXT NOT NULL REFERENCES chats(chat_key),
        learner TEXT NOT NULL,
        started_ts REAL NOT NULL,
        expires_at REAL NOT NULL,
        start_msg_id INTEGER NOT NULL,
        through_msg_id INTEGER NOT NULL,
        source_hash TEXT,
        state TEXT NOT NULL DEFAULT 'prepared'
              CHECK (state IN
                     ('prepared','success','malformed','cancelled','expired','released')),
        records_added INTEGER NOT NULL DEFAULT 0,
        records_merged INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        settled_ts REAL
    );
    CREATE UNIQUE INDEX learner_runs_one_prepared
        ON learner_runs(chat_key, learner) WHERE state = 'prepared';
    CREATE INDEX learner_runs_chat ON learner_runs(chat_key, learner, id);

    CREATE TABLE record_sources (
        id INTEGER PRIMARY KEY,
        record_id INTEGER NOT NULL REFERENCES records(id),
        chat_key TEXT NOT NULL,
        learner TEXT NOT NULL,
        source_first_msg_id INTEGER NOT NULL,
        source_last_msg_id INTEGER NOT NULL,
        source_hash TEXT NOT NULL,
        UNIQUE (record_id)
    );
    CREATE INDEX record_sources_chat
        ON record_sources(chat_key, learner, source_first_msg_id, source_last_msg_id);

    CREATE TABLE record_search_docs (
        id INTEGER PRIMARY KEY,
        chat_key TEXT NOT NULL,
        record_id INTEGER NOT NULL REFERENCES records(id),
        tokens TEXT NOT NULL
    );
    CREATE INDEX record_search_docs_chat ON record_search_docs(chat_key, record_id);
    CREATE VIRTUAL TABLE record_search_fts USING fts5(
        tokens,
        content='record_search_docs',
        content_rowid='id'
    );

    CREATE TABLE record_exposures (
        id INTEGER PRIMARY KEY,
        chat_key TEXT NOT NULL,
        learner TEXT NOT NULL,
        record_id INTEGER NOT NULL REFERENCES records(id),
        run_id INTEGER NOT NULL REFERENCES learner_runs(id),
        exposed_ts REAL NOT NULL,
        UNIQUE (record_id, run_id)
    );
    CREATE INDEX record_exposures_chat
        ON record_exposures(chat_key, learner, record_id);

    CREATE TABLE record_feedback (
        id INTEGER PRIMARY KEY,
        chat_key TEXT NOT NULL,
        learner TEXT NOT NULL,
        record_id INTEGER NOT NULL REFERENCES records(id),
        effect REAL NOT NULL,
        reweight REAL NOT NULL,
        feedback_ts REAL NOT NULL,
        note TEXT
    );
    CREATE INDEX record_feedback_chat
        ON record_feedback(chat_key, learner, record_id);
    """,
    # Migration 10 — the Phase 6 P6.5 durable media catalog (frozen Oracle
    # advisory): chat-scoped media_assets with an opaque cache key, content
    # sha256, kind (sticker/image), mime/dimensions/description, source
    # message+sender provenance, safety status/version, approval/revocation,
    # uses/cooldown, and unique (chat, kind, sha256). Capacity-safe
    # transactional approval/eviction is enforced by the repository (the
    # partial indexes make the approved/pending scans cheap). Existing global
    # emoji rows remain legacy/untrusted — this migration never touches them.
    """
    CREATE TABLE media_assets (
        id                 INTEGER PRIMARY KEY,
        chat_key           TEXT NOT NULL REFERENCES chats(chat_key),
        kind               TEXT NOT NULL
                           CHECK (kind IN ('sticker','image')),
        cache_key          TEXT NOT NULL,   -- OPAQUE content-addressed cache key
        sha256             TEXT NOT NULL,   -- content sha256 of the original bytes
        mime               TEXT NOT NULL,
        width              INTEGER,
        height             INTEGER,
        description        TEXT,
        source_message_id  INTEGER,         -- source message row id (provenance)
        source_sender_id   TEXT,            -- source sender id (provenance)
        source_sender_name TEXT,            -- source sender name (provenance)
        source_ts          REAL,            -- source message timestamp (provenance)
        safety_status      TEXT NOT NULL DEFAULT 'pending'
                           CHECK (safety_status IN
                                  ('pending','approved','rejected','revoked')),
        safety_version     INTEGER NOT NULL DEFAULT 0,
        approved_ts        REAL,
        revoked_ts         REAL,
        uses               INTEGER NOT NULL DEFAULT 0,
        last_used_ts       REAL,
        created_ts         REAL,
        UNIQUE (chat_key, kind, sha256)
    );
    CREATE INDEX media_assets_approved
        ON media_assets(chat_key, kind, safety_status, last_used_ts)
        WHERE safety_status = 'approved';
    CREATE INDEX media_assets_pending
        ON media_assets(chat_key, kind, safety_status, id)
        WHERE safety_status = 'pending';
    """,
    # Migration 11 — the Phase 6 P6.6b durable chat controls (frozen Oracle
    # advisory): chat_controls records bounded INTERNAL focus events that
    # only make the TARGET chat's gate evaluate as focused — delivery still
    # traverses the target chat's normal gate/cycle/outbox flow. ``kind`` is
    # ``focus`` (a bounded focus window, one focus per account) or ``notify``
    # (a bounded internal focus event carrying a payload). ``ttl_until`` is
    # the absolute-epoch expiry (bounded: focus 30..3600s, notify 1..3600s
    # from creation — enforced by the repository). ``dispatch_id`` +
    # ``intent_seq`` are the idempotency identity: a retried settlement of
    # the same dispatch never double-applies (the unique index is the
    # backstop). ``source_chat_key`` is the issuing chat; the target must be
    # a known chat on the SAME account (platform + self_id) — enforced by
    # the repository in the same transaction as the one-focus-per-account
    # projection update. No provider/network calls, no platform sends, no
    # outbox writes.
    """
    CREATE TABLE chat_controls (
        id              INTEGER PRIMARY KEY,
        chat_key        TEXT NOT NULL REFERENCES chats(chat_key),
        kind            TEXT NOT NULL
                        CHECK (kind IN ('focus','notify')),
        ttl_until       REAL NOT NULL,       -- absolute-epoch expiry (bounded)
        created_ts      REAL NOT NULL,
        dispatch_id     INTEGER NOT NULL,    -- provenance (idempotency)
        intent_seq      INTEGER NOT NULL,    -- per-dispatch intent sequence
        source_chat_key TEXT NOT NULL,
        text            TEXT,                -- bounded notify payload
        UNIQUE (dispatch_id, intent_seq)
    );
    CREATE INDEX chat_controls_active
        ON chat_controls(chat_key, ttl_until);
    """,
    # Migration 12 — append-only chat-control application ledger projection.
    # ``chat_controls`` is never deleted or rewritten when focus moves.  This
    # account-scoped projection identifies the one current focus while the
    # ledger retains every applied (dispatch_id, intent_seq) forever.
    """
    CREATE TABLE chat_focus_current (
        platform TEXT NOT NULL,
        self_id TEXT NOT NULL,
        control_id INTEGER NOT NULL REFERENCES chat_controls(id),
        PRIMARY KEY (platform, self_id)
    );
    INSERT OR IGNORE INTO chat_focus_current(platform, self_id, control_id)
    SELECT c.platform, c.self_id, MAX(cc.id)
    FROM chat_controls AS cc
    JOIN chats AS c ON c.chat_key = cc.chat_key
    WHERE cc.kind = 'focus'
    GROUP BY c.platform, c.self_id;
    CREATE INDEX chat_focus_current_control
        ON chat_focus_current(control_id);
    """,
    # Migration 13 — learner scheduling columns.  These columns used to be
    # installed by a version-neutral boot repair, which meant two databases
    # reporting the same user_version could have different schemas.  The
    # migration function is intentionally deterministic and runs only while
    # version 13 is pending.  It is conditional solely because some v12
    # installations already received the old boot repair; both layouts are
    # upgraded to the same result.
    _migration_13_learner_schedule,
    # Migration 14 — exact adaptive provenance.  Record identity and source
    # range are insufficient to identify the producing run after a merge, and
    # a chat-level delivery bit is insufficient to identify an exposure.  The
    # added fields make those identities durable and queryable.
    """
    ALTER TABLE records ADD COLUMN producing_run_id INTEGER;
    ALTER TABLE record_sources ADD COLUMN producing_run_id INTEGER;
    ALTER TABLE record_exposures ADD COLUMN dispatch_id INTEGER;
    ALTER TABLE record_exposures ADD COLUMN slot TEXT NOT NULL DEFAULT '';
    ALTER TABLE record_feedback ADD COLUMN effect_run_id INTEGER;
    CREATE INDEX record_sources_run ON record_sources(producing_run_id);
    CREATE INDEX record_exposures_dispatch
        ON record_exposures(dispatch_id, slot);
    CREATE UNIQUE INDEX record_feedback_run
        ON record_feedback(record_id, effect_run_id)
        WHERE effect_run_id IS NOT NULL;
    """,
    # Migration 15 — durable source-deletion tombstones.  A recall can race
    # the inbound source event, so the deletion identity is retained even when
    # the local message row is not present yet.  Message insertion consults
    # this table and media approval consults the message tombstone in the same
    # writer transaction.
    """
    CREATE TABLE message_deletions (
        chat_key TEXT NOT NULL REFERENCES chats(chat_key),
        platform_msg_id TEXT NOT NULL,
        deleted_ts REAL NOT NULL,
        PRIMARY KEY (chat_key, platform_msg_id)
    );
    CREATE INDEX message_deletions_chat ON message_deletions(chat_key, platform_msg_id);
    """,
]


def load_schema() -> str:
    """The full M0 DDL, as shipped."""
    return SCHEMA_PATH.read_text(encoding="utf-8")


def current_version(conn: Any) -> int:
    """``PRAGMA user_version`` of the connected database."""
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _split_statements(sql: str) -> list[str]:
    """Split a SQL script into individual statements.

    ``--`` comments (full-line and inline) are stripped first so a comment
    containing a semicolon cannot split a statement; the schema and
    migrations are controlled text with no ``--`` inside string literals.
    """
    cleaned: list[str] = []
    for line in sql.splitlines():
        cut = line.find("--")
        cleaned.append(line if cut == -1 else line[:cut])
    return [s.strip() for s in "\n".join(cleaned).split(";") if s.strip()]


def apply_schema(conn: Any) -> int:
    """Create the M0 schema and apply any pending migrations.

    ``conn`` must be in autocommit mode (``isolation_level=None``) so the
    explicit BEGIN/COMMIT here is the only transaction. Returns the new
    ``user_version``. Safe to call on every boot: it is a no-op once the
    database is current.
    """
    version = current_version(conn)
    pending: list[tuple[int, Migration]] = []
    if version == 0:
        pending.append((BASE_VERSION, load_schema()))
    for i, migration in enumerate(MIGRATIONS, start=BASE_VERSION + 1):
        if version < i:
            pending.append((i, migration))
    if not pending:
        return version

    conn.execute("BEGIN")
    try:
        for target, sql in pending:
            if callable(sql):
                sql(conn)
            else:
                for stmt in _split_statements(sql):
                    conn.execute(stmt)
            conn.execute(f"PRAGMA user_version = {target}")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return current_version(conn)
