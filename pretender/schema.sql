-- Pretender M0 schema. Applied once at initial creation (PRAGMA user_version
-- 0 -> 1); every later change ships as an entry in schema.py's MIGRATIONS
-- list, never as an edit to this file. DDL is transactional in SQLite, so
-- schema.py applies the whole file inside one BEGIN/COMMIT.
--
-- Foreign keys are enforced by the writer connection (PRAGMA foreign_keys=ON
-- in db.py): orphan or cross-chat durable records are rejected at the
-- database level, not just in application code.

CREATE TABLE chats (
    chat_key      TEXT PRIMARY KEY,          -- "qq:group:123456"
    platform      TEXT NOT NULL,
    self_id       TEXT NOT NULL,             -- the bot's own account id
    kind          TEXT NOT NULL,             -- 'group' | 'private'
    title         TEXT,
    cursor_msg_id INTEGER,                   -- local messages.id watermark; moved ONLY by finish_cycle
    focus_until   REAL,                      -- focus mode (later phase), absolute epoch
    hold_until    REAL,                      -- durable held outcome; written ONLY by finish_cycle
    avg_interval  REAL,
    idle_streak   INTEGER NOT NULL DEFAULT 0,
    cfg_json      TEXT                       -- arbitrary plugin-owned JSON
);

CREATE TABLE messages (
    id              INTEGER PRIMARY KEY,
    chat_key        TEXT NOT NULL REFERENCES chats(chat_key),
    platform        TEXT NOT NULL,
    self_id         TEXT NOT NULL,
    platform_msg_id TEXT,                    -- NULL where the platform gives none
    sender_id       TEXT NOT NULL,
    sender_name     TEXT NOT NULL DEFAULT '',
    is_self         INTEGER NOT NULL DEFAULT 0,
    text            TEXT NOT NULL DEFAULT '',
    segments_json   TEXT NOT NULL DEFAULT '[]',
    reply_to        TEXT,
    mentions_json   TEXT NOT NULL DEFAULT '[]',
    recv_ts         REAL,                    -- absolute epoch seconds
    deleted         INTEGER NOT NULL DEFAULT 0,
    UNIQUE (platform, self_id, platform_msg_id)
);
CREATE INDEX messages_chat ON messages(chat_key, id);

-- External-content FTS5, CJK-bigram tokenized. repo.py owns the index
-- updates (same transaction as the source write); the raw text is never
-- stored here — only the space-joined bigram tokens.
CREATE VIRTUAL TABLE message_fts USING fts5(
    text,
    content='messages',
    content_rowid='id'
);

CREATE TABLE persons (
    person_key   TEXT PRIMARY KEY,           -- per-chat person identity
    chat_key     TEXT NOT NULL,
    platform_uid TEXT NOT NULL,
    names_json   TEXT NOT NULL DEFAULT '[]',
    profile      TEXT,
    impression   TEXT,
    updated_ts   REAL
);

CREATE TABLE memories (
    id          INTEGER PRIMARY KEY,
    chat_key    TEXT NOT NULL,
    kind        TEXT NOT NULL,
    text        TEXT NOT NULL,
    cues_json   TEXT NOT NULL DEFAULT '[]',
    strength    REAL NOT NULL DEFAULT 1.0,
    created_ts  REAL,
    last_hit_ts REAL
);

CREATE VIRTUAL TABLE memory_fts USING fts5(
    text,
    content='memories',
    content_rowid='id'
);

CREATE TABLE records (
    id           INTEGER PRIMARY KEY,
    chat_key     TEXT,
    learner      TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    weight       REAL NOT NULL DEFAULT 1.0,
    uses         INTEGER NOT NULL DEFAULT 0,
    created_ts   REAL
);

CREATE TABLE vec (
    owner_table TEXT NOT NULL,
    owner_id    INTEGER NOT NULL,
    dim         INTEGER NOT NULL,
    model       TEXT NOT NULL,
    blob        BLOB NOT NULL,
    UNIQUE (owner_table, owner_id, model)
);

CREATE TABLE emoji (
    id                INTEGER PRIMARY KEY,
    sha256            TEXT NOT NULL,
    desc              TEXT,
    platform_ref_json TEXT NOT NULL DEFAULT '{}',
    uses              INTEGER NOT NULL DEFAULT 0,
    last_used_ts      REAL
);

-- One terminal cycle per completed claim; outbox rows carry its id as
-- provenance (every sendable row belongs to a completed durable claim).
CREATE TABLE cycles (
    id          INTEGER PRIMARY KEY,
    chat_key    TEXT NOT NULL REFERENCES chats(chat_key),
    started_ts  REAL NOT NULL,
    end_reason  TEXT NOT NULL,               -- completed | held | ...
    trace_json  TEXT,
    tokens_in   INTEGER NOT NULL DEFAULT 0,
    tokens_out  INTEGER NOT NULL DEFAULT 0
);

-- One row per adapter send (frozen decision #3, at-most-once):
-- pending -> in_flight is a durable CAS before the adapter is invoked;
-- in_flight is NEVER auto-retried after a crash. Terminal: sent | dropped.
-- Rows are created ONLY by finish_cycle, which stamps cycle_id provenance
-- and rejects items whose chat_key differs from the claimed chat.
CREATE TABLE outbox (
    id                 INTEGER PRIMARY KEY,
    chat_key           TEXT NOT NULL REFERENCES chats(chat_key),
    cycle_id           INTEGER NOT NULL REFERENCES cycles(id),
    group_id           TEXT,                 -- split batch: rows share group_id
    seq                INTEGER,              -- and are ordered by seq
    text               TEXT NOT NULL,
    segments_json      TEXT NOT NULL DEFAULT '[]',
    payload_json       TEXT NOT NULL DEFAULT '{}',
    reply_to           TEXT,
    state              TEXT NOT NULL DEFAULT 'pending'
                       CHECK (state IN ('pending','in_flight','sent','dropped')),
    send_after_ts      REAL,                 -- durable pacing, absolute epoch
    attempt_started_ts REAL,
    sent_ts            REAL,
    platform_msg_id    TEXT,
    idem_key           TEXT NOT NULL UNIQUE  -- re-enqueue idempotency
);
CREATE INDEX outbox_ready ON outbox(chat_key, state, send_after_ts);

-- Durable cycle claims (frozen decision #2). start_msg_id is the chat
-- cursor at claim time and through_msg_id the fixed-through boundary: the
-- cursor advances ONLY to through_msg_id, inside finish_cycle, after
-- checking ownership, start cursor, and unexpired lease. The lease is
-- mandatory and finite. The partial unique index is the invariant
-- backstop: at most ONE live claim per chat, enforced by the database.
CREATE TABLE claims (
    id              INTEGER PRIMARY KEY,
    chat_key        TEXT NOT NULL REFERENCES chats(chat_key),
    cycle_id        TEXT NOT NULL,
    started_ts      REAL NOT NULL,
    expires_at      REAL NOT NULL,           -- mandatory finite lease
    start_msg_id    INTEGER NOT NULL,        -- chat cursor at claim time
    through_msg_id  INTEGER NOT NULL,        -- fixed-through boundary
    state           TEXT NOT NULL DEFAULT 'live'
                    CHECK (state IN ('live','finished','released','expired'))
);
CREATE UNIQUE INDEX claims_one_live ON claims(chat_key) WHERE state = 'live';

CREATE TABLE kv (
    k TEXT PRIMARY KEY,
    v TEXT
);