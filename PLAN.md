# Pretender — a light MaiBot

## Context

`/home/redstone/github/Pretender` is empty (one README). The goal is a from-scratch re-implementation
of [MaiBot](https://github.com/Mai-with-u/MaiBot) — an LLM agent that behaves like a human member of a
group chat rather than an assistant that answers questions.

MaiBot works, but it has grown to **723 Python files**. The weight is not in the human-simulation
logic; it is in scaffolding built around it:

| Subsystem | Size | Verdict |
|---|---|---|
| `dashboard/` (React) + `src/webui/` | 25 MB + 1.25 MB | drop → `pretender db` |
| `src/A_memorix/` — vector + graph + PageRank + BM25 + 25 "services" | 2.88 MB, ~130 files | → 3 files |
| `src/plugin_runtime/` — out-of-process plugins over RPC (pipes/TCP/UDS) | 832 KB | → in-process registry |
| `src/common/database/migrations/` — 40 sequential files | — | → one `MIGRATIONS` list |
| `src/config/official_configs.py` — one file | 181 KB | → ~330 LOC + a TOML |
| `src/learners/` — three parallel, structurally identical stacks | 335 KB | → 1 pipeline + 5 definitions |
| `src/chat/replyer/expression_vector_index.py` | 119 KB | → rows in the shared vector table |
| `reasoning_engine.py` + `runtime.py` | 185 KB | → the real logic, ~10× smaller |
| i18n framework, MCP host, telemetry, `statistic.py` | ~400 KB | drop |

Two more structural costs worth naming:

- **MaiBot runs as two processes.** The core exposes FastAPI/uvicorn speaking its own `maim_message`
  protocol; a separate NapCat adapter bridges that to QQ. Pretender speaks **OneBot v11 over WebSocket
  directly** — one process, one less protocol, three fewer dependencies.
- **34 runtime dependencies** (faiss-cpu, pandas, pyarrow, scipy, playwright, fastapi, uvicorn,
  sqlalchemy + sqlmodel, google-genai, mcp, …). Pretender targets **7**: `httpx`, `websockets`,
  `numpy`, `pillow`, `jieba`, `pypinyin`, `orjson`. Everything else is stdlib (`sqlite3`, `asyncio`,
  `tomllib`, `dataclasses`, `logging`).

**Light means code volume and dependency weight — not capability.** Confirmed with the user: the
expression learner, embedding-backed semantic memory, emoji/sticker + image replies, and typo
injection + message splitting all ship. The point is that none of them should need thirty files.

**Target: ~63 files, ~9,500 LOC** (realistically up to 11k — `adapters/onebot.py` and `context.py`
are the two that always exceed their estimate). That is 11× fewer files than MaiBot at capability parity.

### Confirmed decisions

1. **Platform** — platform-agnostic core behind an `Adapter` Protocol; **OneBot v11 / NapCat (QQ)** first.
2. **Capabilities** — expression learner, embedding memory, emoji/image replies, typo + splitting: all in.
3. **LLM layer** — OpenAI-compatible `/chat/completions` only, provider swapped by `base_url`, with
   tool-calling and vision. *(Caveat: `/embeddings` and vision are separate endpoints that many
   providers — DeepSeek among them — do not serve. Profiles may point at different vendors, and the
   bot must degrade cleanly when `embed` is unconfigured.)*
4. **Locale** — Chinese-first persona; every prompt is an external file, no i18n framework.

---

## 1. What actually makes MaiBot feel human

Researched from source. This is the spec.

### A. Two-stage LLM split — never collapse into one call

- **Planner** (`maisaka_chat.prompt`) addresses the model as a *third-party observer*: "You are not
  {bot_name} itself; do not speak on behalf of {bot_name}." It emits free-text analysis **first**,
  then makes tool calls. Nothing to do → no tool call, analysis text only.
- **Replyer** (`maisaka_replyer.prompt`) is a **separate call** that writes the visible message from
  `{identity}`, `{reply_style}` and a `[Reply Reference]` that is explicitly non-binding.

Tools carry `visibility` (`visible` | `deferred` | `hidden`) and `chat_scope` (`all` | `group` |
`private`). Deferred tools stay out of the emitted schema until `tool_search` activates them.
*(MaiBot has a third axis, `stage`; it is dead here because the replyer runs without tools. Recorded
as a deliberate deviation.)*

### B. Turn gating — pure rules, zero LLM cost

The single highest-value component: it costs no tokens and is what stops the bot reading as a
chatbot. Two modes, selected by `reply_trigger_mode`.

**`reply_necessity`** — trigger at score ≥ 80:

```
relevance   @ = 100 | name-mentioned = 80 | quote-reply to bot = 80 | private = 40 | focus = 40 | else 0
content     +15 question   +20 direct request   +20 opinion-solicit
            +5 ≥40 chars   +10 ≥120 chars       −25 whole batch is short reactions
pressure    r = pending / threshold
            r ≤ 1 → min(50, round(50·r²)), +15 if idle ≥ recent average interval
            r > 1 → min(100, 50 + round(50·log1p(r−1)/log1p(4)))
presence    self-message ratio over the last 300 s: 0 below 0.25, linear to −25 at 0.60
final       max(0, round((relevance + content + pressure − presence) · (0.5 + 0.5·frequency)))
```

Text is normalised before scoring — strip **quote prefixes, @mentions, media placeholders and
forwarded blocks** — and a message addressed to a *different* assistant
(`^(DeepSeek|ChatGPT|Grok|豆包|千问|Kimi|Claude)[,，、\s]`) is refused outright.

The pressure span is `MAX(100) − STANDARD(50) = 50`. With `threshold = 8`, pure ambient chatter
(relevance 0, content 0) first crosses 80 at **21 pending messages**; with the `+10` long-text bonus,
at 15; pressure caps at 100 from 40. For a 2000 msg/day group that is **~95 ambient cycles/day**
instead of 2000 — a ~20× cost reduction, and the difference between a presence and a nuisance.

**`frequency`** — trigger at `pending ≥ threshold`; otherwise *idle compensation* counts
`idle_seconds / recent_average_interval` as virtual messages, **hard-capped at `threshold − 1` so
silence can never speak first**. Short of that it returns a `delay` decision carrying `delay_seconds`,
which the scheduler sleeps on — never busy-polling.

**Idle backoff** — consecutive idle cycles (`planner_no_tool_end`, `planner_wait_rest`,
`tool_pause:wait`) back off `min(cap, base·2^(n−start))`; defaults base 15 s, cap 300 s, start 2.
Group chats only, reset by focus, bypassed at high pending.

**State machine** — `RUNNING` / `WAIT` / `STOP`. The `wait` tool pauses N seconds and deliberately
does **not** let new messages interrupt; consecutive waits cap at 3, and hitting the cap forces a rest.
A bot that can be yanked out of a wait by every new message reads as a reflex, not a person.

### C. Attention drift

A 4 × 3 × 3 prompt-injected matrix: `drift_level` (subtle|active|scattered|wild) × `anchor_policy`
(strict|balanced|loose) × `reaction_style` (reserved|natural|lively). **Four** guard clauses, all
required: every drift needs a traceable hook in recent messages; never announce being distracted;
never self-label as ADHD; **never simulate deliberate inefficiency** (the fourth is what stops
"oops, what was I saying" theatre — the most common way drift prompts fail).

Drift renders into **both** prompts. It governs which pending message the observer latches onto,
which is a stage-1 decision; injecting it only into the replyer degrades it to a prose-style knob.

### D. Learners — three identical stacks, one pipeline

Every learner is the same shape: **LLM → JSON records → store → embed → select → inject → score_delta
reweight**. MaiBot builds this three times over 335 KB. One `learn/pipeline.py` plus five declarative
definitions replaces all of it, and adding a sixth learner becomes a prompt file and a record schema.

- **Expression** — `{"situation": "≤20字", "style": "≤20字", "source_id": "N"}`, 3–5 per run, max 10.
  **Must not learn from the bot's own messages** (filtered in SQL *and* stated in the prompt) — self-learning
  is a positive feedback loop that flattens the bot into a caricature of itself within days. Feeds `{reply_style}`.
- **Behavior** — `{segment_id, actor_type, learning_type, action, outcome, source_ids}`, split across
  observation (`other_user`/`group_collective`) and self-reflection. Feeds `{behavior_style}`.
- **Jargon** — mines emerging slang, infers meaning **with and without context**, and can explain a
  term on request (`query_jargon`).
- **Mid-term summary** — on trim, `{"summary": …, "recall_cues": [3–5]}`.
- **Effect** — reads the references shown to the planner plus the chat that followed, judges `adopted`
  and `status`, emits `score_delta` (+0.5…1.0 / +0.1…0.35 / −0.4…−1.0) that reweights the records.
  A bandit loop, not a one-way learner.

### E. Human output post-processing

- **Typo injection** — jieba + pypinyin homophone substitution weighted by character frequency, with
  separate rates for single-char, tone, and whole-word replacement. *pypinyin ships no reverse
  pinyin→chars map; build it at boot from the frequency list (~15 ms).* MaiBot's 358 KB
  `char_frequency.json` becomes a ~10 KB frequency-**ranked** char list — the rank *is* the frequency.
- **Reply splitting** — one reply becomes several messages with human-ish delays.
- Both are per-reply switchable via the `before_post_process` hook (`skip_post_process`,
  `enable_splitter`, `enable_chinese_typo`).

### F. Context management

Trim at 2× `max_context_size` down to 1×. Completed tool turns fold into one synthetic user message
`[已折叠的历史工具调用]` listing id/name/args/result — `reply`/`wait`/`no_action` dropped,
`tool_search` compressed to matched tool names **so deferred-tool activation survives the fold**.
Images past `max_image_num` become `[图片]`, newest kept.

### G. Memory — why no vector DB is needed

- **Write** — messages leaving short-term context are compressed into `{summary, recall_cues}` where
  the 3–5 cues are *query-shaped sentences*, not keywords.
- **Read** — an *impression* of the current chat ("a semantic seed, not a keyword list") is embedded
  and matched against those cues.

Embedding queries against query-shaped cues is what makes recall work without BM25, PageRank, graph
relations, dual-path retrieval or score calibration. The same table serves expression selection.

### H. Focus mode

Only one chat is actively focused at a time — the constraint a human has. Others surface as
`<focus_chat_event>` with `event_type` ∈ `at`|`mention`|`unread_count`|`unviewed_time`.

### I. Emoji / stickers

Sticker choice is **one vision call**, not a search index: a numbered `rows × cols` collage of
candidates goes in, `{"emoji_index": N, "reason": …}` comes out. The library is slot-capped, new
stickers pass a content filter, and an LLM picks the eviction when full.

---

## 2. Module tree

```
pretender/
├─ WIRING & CONTRACTS ───────────────────────────── 9 files, 1,333
│  __init__.py      20   version + public re-exports for plugin authors
│  __main__.py       8   → cli.main()
│  cli.py          180   init | doctor | run | replay | db
│  app.py          200   App: cfg, db, llm, clock, registries; supervise/serve/shutdown
│  config.py       330   frozen dataclasses, tomllib + ${ENV}, per-chat merge, RuntimeOverlay
│  doctor.py       170   preflight: chat, tool-calling, analysis-with-tools, vision, embed dim,
│                        FTS5 present, DB writable, adapter handshake, prompt dir
│  types.py        230   every boundary dataclass. No behaviour.
│  seams.py        150   ALL Protocols in one file — the whole contract surface, one read
│  errors.py        45   Transient vs Permanent (drives every retry decision)
│
├─ INFRA ────────────────────────────────────────── 5 files,   510
│  clock.py         70   Clock protocol; RealClock; VirtualClock (6h of scheduling in ms, in CI)
│  log.py          110   stdlib logging → JSONL, contextvars(chat_key, cycle_id), rotation
│  prompts.py       90   package dir ← user dir overlay, {{var}}, mtime hot-reload
│  registry.py     150   Registry[T] + register-time validation + 3-point HookBus + discovery
│  record.py        90   always-on JSONL of normalised inbound events — the replay corpus
│
├─ MODEL ACCESS ─────────────────────────────────── 4 files,   600
│  llm.py          240   ONE OpenAI-compatible client: named profiles, tools, vision, deadline,
│                        typed retry, usage extraction, request dump
│  toolparse.py    110   tolerant tool-JSON → ONE repair reprompt → degrade to no_action
│  embed.py        130   /embeddings: batching, sha1 disk cache, dim probe, degrade-to-disabled
│  budget.py       120   token/cost ledger, daily cap, degrade rungs
│
├─ STORAGE ──────────────────────────────────────── 5 files,   910   (+ schema.sql, data/)
│  db.py           190   WAL; ONE writer coroutine (50 ms/200 op batches); reads on a 2-thread
│                        executor with thread-local connections; txn(fn) exposing lastrowid
│  schema.py        60   schema.sql loader + MIGRATIONS list via PRAGMA user_version
│  repo.py         340   ~40 typed query functions — the only place SQL text lives
│  vectors.py      170   float32 BLOB ⇄ ndarray, resident matrix per (scope, chat), top-k
│  search.py       150   CJK bigram tokenizer, FTS5 maintenance, bm25, RRF fusion
│
├─ THE GATE ─────────────────────────────────────── 3 files,   530   ← zero LLM cost; the product
│  signals.py      180   the 4 strip targets, short-reaction set, other-assistant refusal,
│                        Chinese question / request / opinion-solicit detectors
│  gate.py         280   GateContext, 5 built-in GateFeatures, composition, both modes, DecisionTrace
│  backoff.py       70   IdleBackoffController
│
├─ RUNTIME ──────────────────────────────────────── 5 files, 1,040
│  ingest.py       160   dedupe, clock-skew EWMA, notices (poke/recall), commit-then-wake
│  scheduler.py    200   heap of (wake_ts, chat_key) w/ lazy invalidation, lease set, re-arm
│  session.py      170   cursor watermark, EWMA interval, self-ratio ring, focus, wait streak
│  cycle.py        320   the saga: planner loop → tool dispatch → reply → emit cap → end_reason
│  outbox.py       190   durable pacing, split atomicity, staleness drop, self-echo write
│
├─ THE TWO-STAGE AGENT ──────────────────────────── 4 files,   730
│  context.py      400   build, trim (tool-group aware), fold, pair-normalise, image budget
│  planner.py      150   stage 1: third-party observer, analysis-then-tools
│  replyer.py      120   stage 2: separate call, non-binding reference
│  drift.py         60   4×3×3 sampler over drift.toml, sticky per-session seed, BOTH prompts
│
├─ KNOWLEDGE ────────────────────────────────────── 5 files,   810
│  memory.py       200   write / hybrid recall / strength decay / prune
│  person.py       140   per-person profile, alias merge, impression
│  expression.py   150   pool + select() → {{reply_style}}
│  emoji.py        160   harvest from group, dedupe, describe, collage-pick, cooldown, evict
│  media.py        160   download, content-addressed cache, clamp, base64, describe-on-ingest
│
├─ tools/ ───────────────────────────────────────── 6 files,   670
│  base.py         140   @tool, JSON schema from signature+docstring, ToolSpec, dispatch,
│                        timeout, capability gating, deferred activation, per-tool rate limit
│  core.py         180   reply, wait, no_action, fetch_history, view_forward_message, tool_search
│  knowledge.py    110   query_memory, query_person_profile, query_jargon
│  media.py        140   send_emoji, send_image
│  chatctl.py       90   notify_chat, set_focus
│
├─ learn/ ───────────────────────────────────────── 8 files,   790
│  runner.py       130   cadence, budget gate, batch selection (is_self excluded in SQL)
│  pipeline.py     170   THE generic learner: prompt → records → store → embed → select → feedback
│  expression.py    70 · behavior.py 80 · jargon.py 90 · summary.py 110   declarative definitions
│  effect.py       130   adoption + outcome judging → score_delta reweight
│
├─ output/ ──────────────────────────────────────── 5 files,   460
│  pipeline.py      90   ordered stages over a MUTABLE Outgoing
│  split.py 90 · typo.py 200 · sanitize.py 70
│
├─ adapters/ ────────────────────────────────────── 4 files,   990
│  base.py          80   Adapter Protocol, capability set, raw call() escape hatch
│  onebot.py       800   forward + reverse WS, echo correlation, segment codec, notices,
│                        reconnect, ping/pong watchdog
│  console.py      100   local REPL adapter — dev and tests without QQ
│
├─ prompts/*.txt         planner, planner_focus, replyer, identity, learn_style, learn_behavior,
│                        learn_jargon, memory_summary, impression, image_desc, emoji_select,
│                        emoji_filter, evaluate_effect, drift.toml
└─ data/char_freq.txt    ~3,500 frequency-ranked chars (10 KB, replaces a 358 KB JSON)
```

---

## 3. The seams — "not limited by the structure"

All Protocols live in **one file** (`seams.py`), so a plugin author reads one file to learn the entire
extension surface. Registration validates shape at boot, because structural typing is compile-time only.

```python
class Adapter(Protocol):
    name: str
    capabilities: frozenset[str]          # {"quote","at","image","sticker","recall","history"}
    async def connect(self) -> None: ...
    def events(self) -> AsyncIterator[AdapterEvent]: ...
    async def send(self, out: Outgoing) -> str | None: ...   # None when the platform returns no id
    async def call(self, action: str, **params) -> Any: ...  # escape hatch to any platform API

class GateFeature(Protocol):
    name: str
    def contribute(self, ctx: GateContext) -> Contribution | None: ...   # op: max | add | scale

class OutputStage(Protocol):
    name: str; order: int
    def apply(self, out: Outgoing) -> Outgoing: ...

class LearnerDef(Protocol):
    name: str; prompt: str; cadence_s: int
    def build_batch(self, repo, chat) -> str: ...
    def parse(self, raw: str) -> list[Record]: ...
    def render(self, selected: list[Record]) -> str: ...   # → a prompt fragment
```

Six ways to extend, cheapest first — none requires touching `pretender/`:

1. **Prompt override** (no code) — drop a same-named file in your prompt dir; hot-reloaded on mtime.
   The whole personality is editable by a non-programmer.
2. **Config override** (no code) — reorder `[output] pipeline`, swap gate mode, retarget `base_url`,
   add per-chat `[[chats]]` overrides.
3. **Registry decorators** — `@tool`, `@gate_feature`, `@stage`, `@learner`, `@adapter`, with
   `replace=True` to shadow a builtin by name.
4. **Discovery** — every `plugins/*.py` under `[plugins] paths` plus every `pretender.plugins` entry
   point; a module may also define `setup(app)`.
5. **Hooks** — three points only: `on_event`, `pre_send`, `on_cycle_end`. *(The synthesis proposed
   seven; `pre_gate`/`post_gate` duplicate `@gate_feature`, which is purer and appears in every
   `DecisionTrace` for free, and `post_reply`/`pre_send` duplicate `@stage`.)*
6. **Escape hatches** — `Adapter.call()` reaches any platform API Pretender never modelled;
   `Message.raw` and `Segment.data` carry untouched payloads; `chats.cfg_json` is arbitrary
   plugin-owned JSON.

The gate deserves special note: the score is *nothing but* a sum of registered `GateFeature`s, and the
five built-ins register exactly like a third-party one. "The boss is in the room, shut up" or "it's
3 a.m., be sleepy" is a pure function that automatically appears in every decision trace and every
replay sweep — and costs zero tokens to evaluate.

---

## 4. Correctness invariants

These are the failure modes the design review found. Each is cheap to honour up front and expensive
to retrofit; write them down as comments and tests, not as folklore.

**Every `tool_call_id` gets a `tool` message before any exit path.** The tool loop has several early
exits (stop-control, wait cap, per-tool timeout, unknown tool, a second `reply` rejected by
`max_replies_per_cycle`). Any of them leaving a call unanswered is a hard OpenAI 400 on the next
request. Structure the loop so the answer is written in a `finally`, not per-branch.

**Fold is all-or-nothing per assistant turn.** "Drop `reply`/`wait`/`no_action`" is only safe applied
to a whole turn; dropping one call's `tool` message from a mixed turn (`[reply, query_memory]`)
orphans the other tool_call.

**Trim must be tool-group aware, and the system message is pinned.** Trimming by message count will
cut an assistant-with-tool_calls away from its `tool` messages, or leave an orphan `tool` first.
Folding handles this; trim is a separate operation and needs the same rule.

**Commit before wake.** `ingest` must await the message's durable commit before waking the scheduler,
or the gate reads an empty pending set, returns `skip`, and consumes the only wake.

**`pending` excludes `is_self`.** The outbox re-injects every send into `messages`, and the cursor
advances at cycle start — so without this the bot's own reply becomes pending and pressure builds
from its own output. (`recent`, used by the presence penalty, *does* include `is_self`.)

**The scheduler heap needs lazy invalidation.** `heapq` has no decrease-key; keep a
`next_wake[chat_key]` map and discard stale entries on pop, or a `delay` for t+300 is silently
overridden by the next immediate wake. Cycle release must push a wake, and a trigger arriving while a
chat is leased must set a re-evaluate flag rather than being dropped.

**Persisted timestamps are absolute epoch seconds.** `RealClock` corrects intra-process drift with
`time.monotonic()`, but `hold_until` / `focus_until` / `send_after_ts` survive restarts, and a
monotonic base resets on reboot.

**Quote-replies score relevance 80.** On OneBot v11 a quote carries **no** `@mention`. Someone
replying directly to the bot with "真的吗" must not be met with silence — this is the single most
visible everyday hole, and MaiBot's own field for it is easy to declare and never read.

**CJK needs a bigram tokenizer.** FTS5's `unicode61` returns **zero rows** for a Chinese substring
query, and `trigram` needs ≥3 characters (so `火锅` never matches). Tokenize to bigrams, quote each
token before OR-joining (bare tokens are FTS5 syntax), and cap term count. `doctor` must probe that
FTS5 exists at all — some distro `sqlite3` builds omit it.

**One writer, `ON CONFLICT DO NOTHING` everywhere.** A single UNIQUE violation would otherwise roll
back all 200 coalesced ops. `UNIQUE(platform, self_id, platform_msg_id)` — message ids are unique per
*account*, not per platform. Forwarded-message contents are never persisted to `messages`.

**`Outgoing` is mutable.** The output pipeline is mutation-shaped (sanitize marks no-mutate spans,
typo honours them, split rewrites parts); a frozen dataclass makes the whole `OutputStage` contract
unimplementable.

**OneBot specifics that bite:** reverse-WS means NapCat dials *us*, so config is a listen host/port
and path, not a dial URL; pin `message_format = array` at handshake or mentions cannot be extracted
reliably; `send_group_msg` may return retcode 1 with **no** `message_id` (self-echo, self-correction
and effect tracking all need a fallback local id); there is no standard history API — NapCat's
`get_group_msg_history` pages by `message_seq`, not id; heartbeats are often disabled, so the watchdog
needs a WS ping/pong fallback or it will reconnect-storm a healthy link.

---

## 5. Storage

One SQLite file, WAL, full `schema.sql` shipped at M0 (so the milestone table does not accumulate
nine dev-time migrations), `PRAGMA user_version` + a `MIGRATIONS` list for changes after that.

```sql
chats(chat_key PK, platform, self_id, kind, title, cursor_msg_id, focus_until,
      avg_interval, idle_streak, cfg_json)
messages(id PK, chat_key, platform, self_id, platform_msg_id, sender_id, sender_name,
         is_self, text, segments_json, reply_to, mentions_json, recv_ts, deleted,
         UNIQUE(platform, self_id, platform_msg_id))
message_fts(rowid, text)                       -- FTS5, bigram-tokenized, external content
persons(person_key PK, chat_key, platform_uid, names_json, profile, impression, updated_ts)
memories(id PK, chat_key, kind, text, cues_json, strength, created_ts, last_hit_ts)
memory_fts(rowid, text)                        -- FTS5, bigram
records(id PK, chat_key, learner, payload_json, weight, uses, created_ts)   -- all 5 learners
vec(owner_table, owner_id, dim, model, blob)   -- float32, L2-normalized at write
emoji(id PK, sha256, desc, platform_ref_json, uses, last_used_ts)
outbox(id PK, chat_key, group_id, seq, parts_json, state, send_after_ts, idem_key UNIQUE)
cycles(id PK, chat_key, started_ts, end_reason, trace_json, tokens_in, tokens_out)
kv(k PK, v)                                    -- embed dim, schema version, budget ledger
```

**Embeddings without a vector DB.** L2-normalise at write, store `float32` in a BLOB, keep one
resident `ndarray` per `(scope, chat)`, and search with a single `matrix @ q` matmul. Under ~50k
vectors that is single-digit milliseconds. Recall is **RRF fusion** over (cosine, `bm25()`) with a
recency × strength weighting — FTS5 ships `bm25()` natively, so MaiBot's BM25 stack is a builtin and
the graph store and PageRank are dropped outright. `expression_vector_index.py` (119 KB) disappears:
expressions are rows in `records` with vectors in the same `vec` table.

Rebuild the matrix incrementally on insert (`np.vstack`); full rebuild only on the hourly decay sweep.

---

## 6. Config

```toml
[bot]
name = "麦麦"
identity_file = "prompts/identity.txt"

[llm.profiles.planner]
base_url = "https://api.deepseek.com/v1"
api_key  = "${DEEPSEEK_API_KEY}"
model    = "deepseek-chat"
temperature = 0.7
max_tokens  = 1200
timeout_s   = 45

[llm.profiles.reply]   # may point at a different vendor
model = "deepseek-chat"

[llm.profiles.vision]  # separate endpoint — not every provider serves one
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model    = "qwen-vl-max"

[llm.profiles.embed]   # optional; memory degrades to FTS-only when absent
base_url = "https://api.siliconflow.cn/v1"
model    = "BAAI/bge-m3"

[gate]
mode = "reply_necessity"     # or "frequency"
threshold = 8
trigger_score = 80
frequency = 1.0

[gate.backoff]
base_s = 15
cap_s = 300
start_count = 2

[drift]
level = "active"
anchor = "balanced"
reaction = "natural"

[output]
pipeline = ["sanitize", "split", "typo"]
max_split = 3
typo_rate = 0.03

[adapter.onebot]
mode = "reverse_ws"          # NapCat dials us
host = "127.0.0.1"
port = 3001
path = "/onebot/v11/ws"

[[chats]]                    # per-chat overrides
key = "qq:group:123456"
gate = { threshold = 12, frequency = 0.6 }
```

An empty TOML must boot a working bot — every field has a default from its dataclass. Secrets are
`${ENV}` references only, never values. **A config key that nothing reads by name at runtime does not
exist**; `doctor` enforces this.

---

## 7. Build order

The ordering constraint that matters: **the gate must exist before the bot ever joins a real group.**
Shipping an always-reply bot into a live QQ group is exactly the behaviour the product exists to
prevent, and a fast route to a group kick or an account ban.

| # | Ships | Why here |
|---|---|---|
| M0 | `types`, `seams`, `config`, `clock`, `log`, `db` + full `schema.sql`, `repo`, `record`, `console` adapter, `ingest`, `outbox` (incl. **self-echo write**) | Messages in and out, durably, with a recording corpus, over a REPL. No LLM. |
| M1 | `llm`, `toolparse`, `doctor`, `context` (**incl. pair-normalisation**), `planner`, `replyer`, `tools/base` + `core` (**incl. `tool_search`**) | The two-stage cycle end to end on the console adapter. `doctor` lands here because M1 is where provider quirks surface. Pair-normalisation is needed the moment there are two tool rounds. |
| M2 | `signals`, `gate`, `backoff`, `scheduler`, `session`, `cycle` | The gate before any real group. Tune against the M0/M1 recording with `replay`. |
| M3 | `adapters/onebot`, `media` | First live group — gated, recorded, replayable. Expect this milestone to be the longest. |
| M4 | `output/*` (split, typo, sanitize), `drift` | It now sounds human, not just times like one. |
| M5 | `embed`, `vectors`, `search`, `memory`, `person` | Recall. Degrades cleanly if no embed profile. |
| M6 | `learn/*` (pipeline first, then the five definitions), `budget` | Budget ships *with* learners, not after — a cadence bug is exactly what a daily cap bounds. |
| M7 | `expression` selection, `emoji`, `tools/media`, `tools/knowledge` | The style and sticker layer. |
| M8 | `registry` plugin discovery, `chatctl`, focus mode, `cli replay --sweep` | Extension surface and cross-chat attention. |

---

## 8. Verification

**Offline, no LLM, no network** — this is the point of keeping gate/context/humanize pure:

- `pytest` golden fixtures pin the scoring function branch by branch. Include: pending 20 → 78 (no
  trigger), 21 → 80 (trigger); `@` always ≥ 100; an all-short-reaction batch scoring below a single
  long message; presence penalty 0 at ratio 0.25 and −25 at 0.60; and the assertion that **empty
  pending + infinite idle yields `delay`, never `trigger`**.
- Property tests over random tool-call sequences assert every `tool_call_id` is answered and no
  orphan `tool` message survives fold or trim.
- `VirtualClock` runs a 6-hour scheduling scenario — backoff growth, idle compensation, wait caps — in
  milliseconds.
- A grep test forbids `time.time()` outside `clock.py`.
- `tomllib.loads()` the shipped sample config in a test. (The synthesis's own sample was not valid
  TOML — multiple key/value pairs on one line.)

**Live, cheap:**

- `pretender doctor` — probes chat, tool-calling, analysis-alongside-tool-calls, vision, embedding
  dimension, FTS5 availability, DB writability, adapter handshake, and that every config key is read.
- `pretender run --dry-run` — prints the full `DecisionTrace` per message and never sends. Watch a
  real group for an hour and read *why* it stayed quiet.
- `pretender replay <chat> --sweep` — re-scores recorded history under varied gate constants and
  reports would-have-spoken rates. Gate tuning is group-specific; expect the first two weeks to be
  iterative, and this is the tool that makes it cost nothing.
- `pretender db --stats` — message/memory/record counts, token spend, cycle end-reason histogram.

**The acceptance test that matters:** in a live group, over a day, the bot should hold well under
~25 % of traffic, answer every `@` and quote-reply, and let ordinary chatter pass without comment.
If it is talking more than that, the presence penalty or the threshold is wrong — not the prompt.
