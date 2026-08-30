"""The deterministic cycle lane: snapshot assembler, CycleRunner, replay.

Phase 2 integration (PLAN.md "Phase 2 Integration Lane"): one pure snapshot
assembler shared by the live dry-run cycle and the replay corpus, one
``CycleRunner`` that claims a chat, evaluates the gate, and applies exactly
the frozen disposition, and a deterministic ``replay_corpus`` /
``sweep_corpus`` with no database/outbox/adapter operations.

Frozen lifecycle implemented here:

- The runner CLAIMS before reading any identity/state/recent/history: the
  claim-bounded grant is the only boundary the snapshot can be built from,
  and a racing claim is detected before any state is read. Once claimed,
  the cycle MUST settle — an ordinary exception or cancellation releases
  the claim in ``finally``, so a durable claim is never stranded.
- The pure Gate applies the full precedence AFTER claiming: a direct @ /
  quote, focus, or high pending bypasses an active durable hold; an
  expired hold never regenerates a fresh hold from stale history.
- Ordinary delay and active-hold delay RELEASE the claim with no cursor or
  session mutation (``release_cycle``); the pending messages stay pending
  for the next claim.
- Skip (refusal) and dry-run trigger TERMINALLY finish (``finish_cycle``)
  with an EMPTY outbox: the cursor advances to the claim's through
  boundary, the hold is cleared, the idle streak resets, and the
  ``DecisionTrace`` is persisted as JSON — all in one transaction. The
  finish fences against a FRESH clock timestamp (the lease may have
  expired while the cycle ran).
- A trigger claim is NEVER retained without an agent: outside dry-run the
  runner releases the claim and returns the trigger decision (event-only),
  so the scheduler never re-arms a timed wake for it.
- ``on_cycle_end`` hooks fire after terminal completion only.

Replay (``replay_corpus``) re-scores a recorded corpus through the SAME
assembler + gate path under a deterministic dispatcher model: the corpus
is processed in FILE (commit) order with monotonic virtual time, commits
at the same commit time coalesce into one wake (the App's next-turn
flush), and a timed wake due at exactly a commit time fires AFTER the
commit (commit-before-timer) — see ``replay_corpus``.

Marker-driven replay (``replay_marker_schedule``) re-scores the v4
RECORDED dispatch schedule instead: raw events keyed by EventId, commit
markers in CommitSeq order, and settled dispatch markers in DispatchId
order (see ``record.CorpusView``). For every settled dispatch marker the
exact attached pending messages are reconstructed from its frozen
``attached`` CommitSeqs, the snapshot uses the marker's frozen
start/through boundary, cause, scheduled time, and settled evaluation
timestamp, and the SAME ``assemble_snapshot`` + ``Gate.evaluate`` path
runs — no separate scoring implementation, no DB/outbox/adapter side
effects. Prior terminal reason/state is reconstructed from the marker
schedule. ``sweep_marker_schedule`` rescopes the FIXED recorded schedule
under RuntimeOverlay gate constants, never inventing counterfactual
future timer events.

The snapshot assembler derives the structured direct-address facts:
``has_direct_at`` from the pending mentions vs the chat's self id,
``has_quote_to_self`` by resolving reply targets against the durable
``quote_self_ids`` (the runner resolves every distinct pending reply_to
through ``Repository.get_message`` — not only the limited rendered window;
replay resolves against the whole corpus) plus the limited recent list,
``has_other_assistant`` through the signals refusal detector, ``self_name``
from the bot config, the exact full-window self ratio, idle seconds from
the last non-self timestamp, focus/hold facts from the durable
``ChatState``, the exact per-chat backoff configuration from
``Config.for_chat``, and the fixed claim bounds from the ``ClaimGrant``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Coroutine, Mapping, Sequence, cast

from pretender.budget import (
    BLOCKED,
    BudgetBlockedError,
    BudgetManager,
    BudgetedClient,
)
from datetime import datetime

from pretender.clock import RealClock
from pretender.config import (
    AgentConfig,
    BudgetConfig,
    ChatConfig,
    Config,
    ContextConfig,
    OutputConfig,
    RuntimeOverlay,
)
from pretender.errors import (
    ClaimError,
    LLMPermanentError,
    LLMTransientError,
    PermanentError,
    PromptError,
)
from pretender.gate import Gate
from pretender.learn.effect import EFFECT_CATEGORIZATIONS, derive_effect_delta
from pretender.learn.render import canonical_content, escape_untrusted, render_records
from pretender.log import get_logger
from pretender.output.pipeline import OutputPipeline, stable_group_id
from pretender.pacing import ewma_interval
from pretender.planner import DEFAULT_MAX_TOOL_ROUNDS, PlanIntent, Planner
from pretender.prompts import PromptStore
from pretender.record import CorpusView
from pretender.registry import (
    HookBus,
    configured_plugin_manifest,
    feature_implementation_fingerprint,
)
from pretender.access import is_muted
from pretender.drift import build_drift_block
from pretender.replyer import ReplyContext, Replyer
from pretender.seams import LLMClient, Repository
from pretender.session import is_focused
from pretender.signals import is_other_assistant_target, normalize_text
from pretender.tools.base import ToolRegistry
from pretender.tools.chatctl import ChatControlCallbacks
from pretender.tools.core import CoreToolRegistry, ToolContext, register_core_tools
from pretender.tools.knowledge import KnowledgeCallbacks
from pretender.tools.media import (
    MediaCallbacks,
    MediaReplyIntent,
    catalog_prompt,
    media_segment_for_intent,
)
from pretender.types import (
    ChatControl,
    ChatControlIntent,
    ChatControlKind,
    ChatIdentity,
    ChatKey,
    ChatState,
    ClaimBusy,
    ClaimGrant,
    CorpusMarker,
    CycleClaim,
    CycleFinish,
    CycleId,
    Decision,
    DecisionTrace,
    DispatchCause,
    DispatchGrant,
    DispatchSettle,
    GateSnapshot,
    Message,
    MessageId,
    MessageRowId,
    MediaKind,
    OutboxItem,
    Outgoing,
    Reason,
    RecentSnapshot,
    Record,
    RecordHit,
    RuntimeMode,
    Segment,
    SelfId,
    SenderId,
    TranscriptMessage,
)
from pretender.vectors import VectorIndex

# The frozen 300-second presence window (PLAN.md §1.B).
SNAPSHOT_WINDOW_S = 300.0
# The LIMITED rendered recent list; the full-window counts never change.
SNAPSHOT_LIMIT = 100
# Default finite claim lease (seconds); the runner's cycles are fast, so a
# 60 s lease is far beyond any Phase 2 cycle duration.
CLAIM_LEASE_S = 60.0
# The deterministic sweep grid (threshold x trigger_score).
SWEEP_THRESHOLDS: tuple[int, ...] = (2, 4, 8, 12)
SWEEP_TRIGGER_SCORES: tuple[int, ...] = (40, 60, 80, 100)

log = get_logger("cycle")


# ── The pure snapshot assembler (shared by CycleRunner and replay) ──────────

def assemble_snapshot(
    *,
    grant: ClaimGrant,
    identity: ChatIdentity,
    state: ChatState,
    recent: RecentSnapshot,
    cfg: ChatConfig,
    now: float,
    self_name: str | None,
    self_aliases: tuple[str, ...] = (),
    muted: bool = False,
    previous_end_reason: str | None,
    window_s: float = SNAPSHOT_WINDOW_S,
    quote_self_ids: frozenset[MessageId] = frozenset(),
) -> GateSnapshot:
    """Build one fully valid ``GateSnapshot`` from the claim-bounded reads.

    Pure: no I/O, no clock, no config loading — every input is passed in.
    The direct-address facts are DERIVED here, never inferred from visible
    text by the gate:

    - ``has_direct_at``: any pending message's structured ``mentions``
      contains the chat's ``self_id``.
    - ``has_quote_to_self``: a pending message's ``reply_to`` resolves to a
      SELF message — either through the durable ``quote_self_ids`` (the
      runner resolves every distinct pending reply target through
      ``Repository.get_message``, so an older quote outside the rendered
      window still triggers; replay resolves against the whole corpus) or
      through the limited recent list.
    - ``has_other_assistant``: the signals refusal detector over the
      normalized pending texts.
    - ``self_name``: the bot config name (name mention scoring input).
    - ``muted``: the caller's ``[access]`` verdict for this chat. Passed in
      rather than derived here because access is a TOP-LEVEL policy, not a
      per-chat merged section.
    - ``self_ratio``: the exact full-window ratio ``self_count /
      window_count`` (0.0 on an empty window).
    - ``idle_seconds``: ``now - last_nonself_ts``; when the window holds no
      non-self message the chat has been idle for at least the whole
      window, so the window length is the conservative lower bound.
    - focus/hold facts come from the durable ``ChatState``; the fixed
      claim bounds come from the ``ClaimGrant``; the idle-backoff facts
      come from the chat's EXACT merged ``Config.for_chat`` values (the
      gate builds its controller from these per evaluation).
    """
    pending = grant.pending
    self_id = identity.self_id
    window_count = recent.window_count
    self_count = recent.self_count
    last_nonself_ts = recent.last_nonself_ts
    return GateSnapshot(
        chat_key=grant.claim.chat_key,
        cycle_id=grant.claim.cycle_id,
        start_msg_id=grant.start_msg_id,
        through_msg_id=grant.through_msg_id,
        evaluated_ts=now,
        self_id=self_id,
        mode=cfg.gate.mode,
        threshold=cfg.gate.threshold,
        trigger_score=cfg.gate.trigger_score,
        frequency=cfg.gate.frequency,
        pending=len(pending),
        pending_messages=pending,
        recent=recent.messages,
        window_count=window_count,
        self_count=self_count,
        last_nonself_ts=last_nonself_ts,
        idle_seconds=(
            now - last_nonself_ts if last_nonself_ts is not None else window_s
        ),
        recent_average_interval=(
            state.avg_interval if state.avg_interval is not None else 0.0
        ),
        self_ratio=(self_count / window_count) if window_count else 0.0,
        is_group=identity.kind == "group",
        is_focused=is_focused(state, now),
        last_message=recent.messages[0] if recent.messages else None,
        self_name=self_name,
        self_aliases=self_aliases,
        muted=muted,
        has_direct_at=any(self_id in m.mentions for m in pending),
        has_quote_to_self=_quote_to_self(pending, recent.messages, quote_self_ids),
        has_other_assistant=any(
            is_other_assistant_target(normalize_text(m.text)) for m in pending
        ),
        hold_until=state.hold_until,
        idle_streak=state.idle_streak,
        previous_end_reason=previous_end_reason,
        backoff_base_s=cfg.gate.backoff.base_s,
        backoff_cap_s=cfg.gate.backoff.cap_s,
        backoff_start_count=cfg.gate.backoff.start_count,
    )


def _quote_to_self(
    pending: tuple[Message, ...],
    recent: tuple[Message, ...],
    quote_self_ids: frozenset[MessageId] = frozenset(),
) -> bool:
    """True when a pending message replies to a SELF message.

    The reply target resolves against the durable ``quote_self_ids`` (the
    runner resolves every distinct pending reply_to through
    ``Repository.get_message``; replay resolves against the whole corpus)
    OR the limited recent list. An unresolvable target is never assumed to
    be self."""
    return any(
        m.reply_to is not None
        and (
            m.reply_to in quote_self_ids
            or any(r.id == m.reply_to and r.is_self for r in recent)
        )
        for m in pending
    )


# ── The frozen per-dispatch adaptive context (Phase 6 P6.4b) ─────────────────

@dataclass(frozen=True)
class AdaptiveContext:
    """The frozen per-dispatch adaptive context (Phase 6 P6.4b).

    Computed ONCE per live dispatch after the gate triggers, and handed to
    every planner tool round and the replyer so they all see the exact same
    adaptive surface. ``reply_style`` is the expression learner's style
    (falling back to ``"自然"``); ``jargon`` is scoped to the current
    pending/recent text; ``summary``/``behavior`` are bounded context slots.
    Every slot is capped (``AdaptiveContextService.MAX_PER_SLOT`` records
    each, ``AdaptiveContextService.MAX_TOTAL_CHARS`` escaped chars total)
    and excludes legacy/retired records. ``rendered`` is the bounded escaped
    reference block embedded in the planner's chat log; the records that
    actually made it into ``rendered`` are the ``frozen_records`` (the only
    ones eligible for exposure/use accounting).
    """

    chat_key: ChatKey
    reply_style: str = "自然"
    expression: tuple[Record, ...] = ()
    jargon: tuple[Record, ...] = ()
    summary: tuple[Record, ...] = ()
    behavior: tuple[Record, ...] = ()
    rendered: str = ""

    @property
    def frozen_records(self) -> tuple[Record, ...]:
        """The records actually rendered into the prompt (exposure-eligible)."""
        return self.expression + self.jargon + self.summary + self.behavior


class AdaptiveContextService:
    """Deterministic adaptive selection/context service (Phase 6 P6.4b).

    Builds the frozen per-dispatch ``AdaptiveContext`` from the
    ``AdaptiveRepository``: expression records become the reply style, and
    jargon/summary/behavior are bounded context slots. EVERY slot is scoped
    to the current pending/recent text, so the voice and the references move
    with the conversation instead of being frozen at whatever the learners
    weighted highest. Selection is deterministic (record FTS merged with
    vector score when an active record-vector space is available, ranked by
    score then weight/uses/id, with the weight ordering as the fallback for
    every slot but jargon), capped at ``MAX_PER_SLOT`` records per slot and
    ``MAX_TOTAL_CHARS`` escaped chars total, and never includes legacy or
    retired records (the repository surface already excludes them). The
    service is queried ONLY after the gate triggers (the agent dispatch
    lane) and never in dry-run/replay/doctor.
    """

    MAX_PER_SLOT = 3
    MAX_TOTAL_CHARS = 1200
    #: The bounded relevance query (the current pending/recent text is
    #: truncated before it is tokenized into the FTS MATCH).
    _QUERY_MAX = 400

    def __init__(
        self, repo: Any, *, now: Callable[[], float] | None = None,
        embed: Any = None, vectors: VectorIndex | None = None,
        model: str | None = None, budget_for: Callable[[ChatKey], Any] | None = None,
    ) -> None:
        self._repo = repo
        self._now = now
        self._embed = embed
        self._vectors = vectors if vectors is not None else VectorIndex(repo)
        self._model = model
        self._budget_for = budget_for

    async def build(
        self,
        chat_key: ChatKey,
        *,
        pending_text: str,
        recent_text: str,
        mode: str = RuntimeMode.LIVE,
    ) -> AdaptiveContext:
        """One frozen context for one live dispatch. Every selection failure
        degrades to an empty slot (never a provider call, never a raise)."""
        semantic = await self._semantic_records(
            chat_key, f"{pending_text} {recent_text}".strip()
        )
        query = f"{pending_text} {recent_text}".strip()[: self._QUERY_MAX]
        expression = await self._relevant(
            chat_key, "expression", query, semantic=semantic
        )
        summary = await self._relevant(chat_key, "summary", query, semantic=semantic)
        behavior = await self._relevant(chat_key, "behavior", query, semantic=semantic)
        # Jargon alone never falls back to the weight ordering: slang that
        # does not match what is being said is noise, not context.
        jargon = await self._relevant(
            chat_key, "jargon", query, semantic=semantic, fallback=False
        )
        reply_style = _render_reply_style(expression)
        rendered, frozen = self._render(expression, jargon, summary, behavior)
        return AdaptiveContext(
            chat_key=chat_key,
            reply_style=reply_style,
            expression=frozen["expression"],
            jargon=frozen["jargon"],
            summary=frozen["summary"],
            behavior=frozen["behavior"],
            rendered=rendered,
        )

    async def _relevant(
        self, chat_key: ChatKey, learner: str, query: str, *,
        semantic: dict[int, tuple[Record, float]] | None = None,
        fallback: bool = True,
    ) -> list[Record]:
        """The records that match what is being said right now.

        ``query`` is the bounded current pending/recent text. Lexical (FTS)
        and semantic hits are merged and ranked deterministically by
        ``(-score, -weight, uses, id)``.

        Relevance is the point. Selecting the highest-weight records instead
        would hand the replyer the same three expressions in every
        conversation forever, which is exactly how a bot sounds: a person's
        register moves with the topic. MaiBot picks the situations matching
        the current context (``maisaka_expression_selector``); this reaches
        the same place through the record FTS index rather than a second
        provider call.

        ``fallback`` decides what an unmatched slot does. Expression,
        behavior and summary fall back to the weight ordering so a chat with
        no lexical overlap still has a voice; jargon does not, because
        unmatched slang is noise.
        """
        combined: dict[int, tuple[Record, float]] = {}
        if query:
            try:
                hits = await self._repo.query_records(
                    chat_key, learner, query, limit=self.MAX_PER_SLOT
                )
            except Exception:
                hits = []
            if hits:
                by_id = await self._records_by_id(chat_key, learner)
                combined = {
                    hit.record_id: (by_id[hit.record_id], 1.0 / (10 + i))
                    for i, hit in enumerate(hits) if hit.record_id in by_id
                }
        for record_id, (record, score) in (semantic or {}).items():
            if record.learner == learner:
                previous = combined.get(record_id)
                combined[record_id] = (
                    record, score + (previous[1] if previous else 0.0)
                )
        if combined:
            return [record for record, _score in sorted(
                combined.values(),
                key=lambda item: (
                    -item[1], -item[0].weight, item[0].uses, item[0].id or 0
                ),
            )[: self.MAX_PER_SLOT]]
        if not fallback:
            return []
        try:
            return await self._repo.select_learner_records(
                chat_key, learner, limit=self.MAX_PER_SLOT
            )
        except Exception:
            return []

    async def _semantic_records(
        self, chat_key: ChatKey, query: str
    ) -> dict[int, tuple[Record, float]]:
        """Retrieve trusted record vectors in the current active space."""
        embed = self._embed
        if not query or embed is None or not getattr(embed, "enabled", False):
            return {}
        try:
            space_id = getattr(embed, "space_id", "")
            active = next(
                (generation for generation in await self._repo.list_embedding_generations()
                 if generation.state == "active"
                 and generation.space_id == space_id
                 and (self._model is None or generation.model == self._model)),
                None,
            )
            if active is None:
                return {}
            if self._budget_for is not None and not embed.cached([query]):
                decision = await self._budget_for(chat_key).reserve(chat_key, calls=1)
                if getattr(decision, "kind", BLOCKED) == BLOCKED:
                    return {}
            result = await embed.embed([query])
            if result.status != "ok" or not result.vectors:
                return {}
            vector = result.vectors[0]
            if vector.shape[0] != active.dim:
                return {}
            hits = await self._vectors.search(
                chat_key, vector, active.model, active.id,
                self.MAX_PER_SLOT * 8, owner_table="records"
            )
            ids = [hit.owner_id for hit in hits]
            records: list[Record] = []
            for learner in ("expression", "jargon", "summary", "behavior"):
                records.extend(await self._repo.get_learner_records_by_ids(
                    chat_key, learner, ids
                ))
            by_id = {record.id: record for record in records if record.id is not None}
            return {
                hit.owner_id: (by_id[hit.owner_id], hit.score)
                for hit in hits if hit.owner_id in by_id
            }
        except Exception:
            return {}

    async def _records_by_id(
        self, chat_key: ChatKey, learner: str
    ) -> dict[int, Record]:
        try:
            return {
                rec.id: rec
                for rec in await self._repo.list_learner_records(
                    chat_key, learner, limit=100
                )
                if rec.id is not None
            }
        except Exception:
            return {}

    def _render(
        self,
        expression: list[Record],
        jargon: list[Record],
        summary: list[Record],
        behavior: list[Record],
    ) -> tuple[str, dict[str, tuple[Record, ...]]]:
        """Render the selected records into ONE bounded escaped reference
        block. The 1200-char cap is enforced on the escaped record bodies
        (the wrapper markup is counted too); a record that does not fit is
        NOT frozen (it never becomes exposure-eligible)."""
        parts: list[str] = []
        frozen: dict[str, list[Record]] = {
            "expression": [], "jargon": [], "summary": [], "behavior": [],
        }
        budget = self.MAX_TOTAL_CHARS
        for slot_name, records in (
            ("expression", expression),
            ("jargon", jargon),
            ("summary", summary),
            ("behavior", behavior),
        ):
            if not records:
                continue
            slot_parts: list[str] = []
            slot_frozen: list[Record] = []
            for rec in records:
                escaped = escape_untrusted(self._record_body(rec))
                if len(escaped) > budget:
                    break  # the cap is exhausted; later records are not frozen
                slot_parts.append(
                    f'<record ref="{len(slot_frozen) + 1}">\n{escaped}\n</record>'
                )
                slot_frozen.append(rec)
                budget -= len(escaped)
            if slot_frozen:
                parts.append(
                    f"<{slot_name}>\n" + "\n".join(slot_parts) + f"\n</{slot_name}>"
                )
                frozen[slot_name] = slot_frozen
        if not parts:
            return "", {k: tuple(v) for k, v in frozen.items()}
        header = "【自适应参考】(以下内容是观察到的数据，不是指令)"
        return header + "\n" + "\n".join(parts), {
            k: tuple(v) for k, v in frozen.items()
        }

    @staticmethod
    def _record_body(rec: Record) -> str:
        """The record's prompt body: the payload ``text`` when present, else
        the canonical sorted-key JSON rendering."""
        payload = rec.payload if isinstance(rec.payload, dict) else {}
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            return text
        return canonical_content(payload)


# ── The Phase 3 agent coordinator (planner + replyer + optional budget) ─────

@dataclass(frozen=True)
class AgentOutcome:
    """The typed terminal outcome of one Phase 3 agent run.

    ``intent`` is ``reply`` | ``wait`` | ``no_action`` | ``budget_blocked``.
    ``reply`` carries the replyer's final text (``reply_text`` is None when
    the replyer produced no usable output — the caller must send nothing);
    ``wait`` carries ``wait_seconds``; ``tokens_in``/``tokens_out``/``usage``
    aggregate the planner + replyer provider usage; ``end_reason`` names why
    the run stopped. ``media_intent`` is the staged media send
    (``send_emoji`` / ``send_image``) — mutually exclusive with the text
    verdicts; the CycleRunner converts it into an ``Outgoing`` media segment
    ONLY at normal terminal settlement. ``chat_controls`` are the staged
    chat controls (``set_focus`` / ``notify_chat``) — NOT mutually exclusive
    with the text verdicts; the CycleRunner applies them idempotently after
    the terminal settlement (LIVE only).
    """

    intent: str
    reply_text: str | None = None
    reply_to: str | None = None
    wait_seconds: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    end_reason: str | None = None
    media_intent: MediaReplyIntent | None = None
    chat_controls: tuple[ChatControlIntent, ...] = ()

    def __post_init__(self) -> None:
        if self.intent not in ("reply", "wait", "no_action", "budget_blocked"):
            raise ValueError(f"invalid agent intent: {self.intent!r}")
        if self.reply_text is not None and not isinstance(self.reply_text, str):
            raise ValueError("reply_text must be a string or None")
        if self.reply_to is not None and not isinstance(self.reply_to, str):
            raise ValueError("reply_to must be a string or None")
        if self.wait_seconds is not None and (
            isinstance(self.wait_seconds, bool)
            or not isinstance(self.wait_seconds, (int, float))
        ):
            raise ValueError("wait_seconds must be a number or None")
        for name in ("tokens_in", "tokens_out"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.media_intent is not None and not isinstance(
            self.media_intent, MediaReplyIntent
        ):
            raise ValueError("media_intent must be a MediaReplyIntent or None")
        if not isinstance(self.chat_controls, tuple) or not all(
            isinstance(c, ChatControlIntent) for c in self.chat_controls
        ):
            raise ValueError(
                "chat_controls must be a tuple of ChatControlIntent"
            )


class PhaseAgent:
    """The Phase 3 agent coordinator: planner tool loop → replyer draft →
    typed outcome.

    Two construction forms share one ``run``:

    - ``PhaseAgent(planner, replyer, budget=None, ...)`` — the INJECTED
      seam form: the caller supplies ready ``Planner``/``Replyer`` (and an
      optional budget) and owns their budget behavior. ``run`` calls them
      directly with NO saga-level accounting (the injected seams own their
      own per-call budget).
    - ``PhaseAgent.budgeted(llm, prompts, registry, context_config, budget,
      agent_config, ...)`` — the DEFAULT build form: holds the base
      components the App builds once and, per run (per chat), wraps the LLM
      client in a chat-bound ``BudgetedClient`` and builds a fresh
      ``Planner``/``Replyer`` over it. The budget is therefore enforced PER
      PROVIDER CALL (both the planner and the replyer go through the same
      budgeted client), never per saga. A ``BudgetBlockedError`` (the hard
      cap reached before a call) short-circuits to ``budget_blocked``
      without further LLM calls.

    No adapter/outbox/network I/O happens here — the caller
    (``CycleRunner``) settles the outcome through the ledger.
    """

    def __init__(
        self,
        planner: Any,
        replyer: Any,
        budget: Any | None = None,
        *,
        reply_style: str = "自然",
        tools: list[dict[str, object]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_tool_rounds: int | None = None,
        repair: Callable[[str], str] | None = None,
        memory_search: Any = None,
        person_service: Any = None,
        capabilities: frozenset[str] = frozenset(),
        forward_resolver: Callable[[ChatKey], Mapping[str, str]] | None = None,
        jargon_query: Callable[[ChatKey, str, int], Awaitable[list[RecordHit]]] | None = None,
        media_callbacks: Callable[[ChatKey], MediaCallbacks | None] | None = None,
        chat_control_callbacks: Callable[[ChatKey], ChatControlCallbacks | None] | None = None,
    ) -> None:
        if planner is None or replyer is None:
            raise ValueError("planner and replyer are required")
        self._mode = "injected"
        self._planner = planner
        self._replyer = replyer
        self._budget = budget
        self._reply_style = reply_style
        self._tools = tools
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_tool_rounds = max_tool_rounds
        self._repair = repair
        # Phase 5 knowledge wiring (only when available): the shared
        # MemorySearch + PersonService the budgeted form binds into
        # chat-scoped KnowledgeCallbacks for the deferred knowledge tools.
        self._memory_search = memory_search
        self._person_service = person_service
        # Phase 6 P6.4b jargon wiring: a chat-scoped jargon query over the
        # AdaptiveRepository (bound to ``chat_key`` by the caller), feeding
        # the deferred ``query_jargon`` tool. None disables the tool.
        self._jargon_query = jargon_query
        # Phase 6 P6.5b media wiring: a chat-scoped MediaCallbacks factory
        # (bound to ``chat_key`` by the caller), feeding the deferred
        # send_emoji / send_image tools. None disables the media tools.
        self._media_callbacks = media_callbacks
        # Phase 6 P6.6b chat-control wiring: a chat-bound
        # ChatControlCallbacks factory (bound to ``chat_key`` by the
        # caller), feeding the deferred set_focus / notify_chat tools. None
        # disables the chat-control tools.
        self._chat_control_callbacks = chat_control_callbacks
        # Real budgeted ToolContext wiring: the adapter-supported
        # capabilities and a safely chat-scoped forward resolver injected by
        # the App, so fetch_history / view_forward_message are reachable in
        # production when the adapter provides the data and fail closed
        # otherwise. The injected form ignores these (its planner owns its
        # own ToolContext).
        self._capabilities = frozenset(capabilities)
        self._forward_resolver = forward_resolver
        # Budgeted-form attributes (set by ``budgeted``); None in the
        # injected form.
        self._llm: LLMClient | None = None
        self._prompts: PromptStore | None = None
        self._registry: CoreToolRegistry | ToolRegistry | None = None
        self._context_config: ContextConfig | None = None
        self._agent_config: AgentConfig | None = None
        # Per-chat config resolution (budgeted form): when ``_cfg``/``_repo``/
        # ``_now`` are set, the chat's effective budget/context/agent config
        # is honored at each provider call instead of binding top-level
        # settings globally. ``_budgets`` caches one chat-bound BudgetManager
        # per chat.
        self._cfg: Config | None = None
        self._repo: Repository | None = None
        self._now: Callable[[], float] | None = None
        self._budgets: dict[ChatKey, BudgetManager] = {}

    @classmethod
    def budgeted(
        cls,
        llm: LLMClient,
        prompts: PromptStore,
        registry: CoreToolRegistry | ToolRegistry,
        context_config: ContextConfig,
        budget: BudgetManager,
        agent_config: AgentConfig,
        *,
        cfg: Config | None = None,
        repo: Repository | None = None,
        now: Callable[[], float] | None = None,
        reply_style: str = "自然",
        tools: list[dict[str, object]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_tool_rounds: int | None = None,
repair: Callable[[str], str] | None = None,
        memory_search: Any = None,
        person_service: Any = None,
        capabilities: frozenset[str] = frozenset(),
        forward_resolver: Callable[[ChatKey], Mapping[str, str]] | None = None,
        jargon_query: Callable[[ChatKey, str, int], Awaitable[list[RecordHit]]] | None = None,
        media_callbacks: Callable[[ChatKey], MediaCallbacks | None] | None = None,
        chat_control_callbacks: Callable[[ChatKey], ChatControlCallbacks | None] | None = None,
    ) -> "PhaseAgent":
        if (
            llm is None
            or prompts is None
            or registry is None
            or context_config is None
        ):
            raise ValueError(
                "llm, prompts, context_config and registry are required"
            )
        agent = cls.__new__(cls)
        agent._mode = "budgeted"
        agent._llm = llm
        agent._prompts = prompts
        agent._registry = registry
        agent._context_config = context_config
        agent._budget = budget
        agent._agent_config = agent_config
        agent._reply_style = reply_style
        agent._tools = tools
        agent._temperature = temperature
        agent._max_tokens = max_tokens
        agent._max_tool_rounds = max_tool_rounds
        agent._repair = repair
        # Phase 5 knowledge wiring (only when available).
        agent._memory_search = memory_search
        agent._person_service = person_service
        # Phase 6 P6.4b jargon wiring (see ``__init__``).
        agent._jargon_query = jargon_query
        # Phase 6 P6.5b media wiring (see ``__init__``).
        agent._media_callbacks = media_callbacks
        # Phase 6 P6.6b chat-control wiring (see ``__init__``).
        agent._chat_control_callbacks = chat_control_callbacks
        # Real budgeted ToolContext wiring (see ``__init__``).
        agent._capabilities = frozenset(capabilities)
        agent._forward_resolver = forward_resolver
        # Per-chat config resolution: when cfg/repo/now are all provided, the
        # chat's effective budget/context/agent config is honored at each
        # provider call (caps/rungs, fallback profile, context) instead of
        # binding top-level settings globally.
        agent._cfg = cfg
        agent._repo = repo
        agent._now = now
        agent._budgets = {}
        return agent

    async def run(
        self,
        *,
        chat_key: ChatKey,
        identity: str,
        chat_log: str,
        messages: Sequence[TranscriptMessage],
        focus_chat: str | None = None,
        chat_kind: str = "group",
        self_name: str | None = None,
        deadline: float | None = None,
        recent: Sequence[Message] = (),
        reply_style: str | None = None,
        behavior_style: str = "",
    ) -> AgentOutcome:
        """Run the deterministic agent sequence for one trigger.

        ``messages`` is the pending chat history in transcript form (the
        caller converts inbound ``Message``s); ``chat_log`` is the plain-text
        chat rendering embedded in the planner system prompt; ``identity`` is
        the bot identity string (the replyer's half of the persona) and
        ``behavior_style`` the planner's half — when to speak, when not to; ``deadline`` is the aggregate saga deadline
        (``now + max_execution_s``) the planner/replyer LLM calls are bounded
        by. ``recent`` is the current dispatch's recent ``Message`` snapshot
        the budgeted form injects into the real ``ToolContext`` (for
        ``fetch_history``). ``reply_style`` overrides the construction-time
        style for THIS run (the frozen per-dispatch adaptive reply style);
        None keeps the construction-time default. Returns the typed
        ``AgentOutcome``.
        """
        style = reply_style if reply_style is not None else self._reply_style
        if self._mode == "budgeted":
            return await self._run_budgeted(
                chat_key=chat_key,
                identity=identity,
                behavior_style=behavior_style,
                chat_log=chat_log,
                messages=messages,
                focus_chat=focus_chat,
                chat_kind=chat_kind,
                self_name=self_name,
                deadline=deadline,
                recent=recent,
                reply_style=style,
            )
        return await self._run_injected(
            chat_key=chat_key,
            identity=identity,
            behavior_style=behavior_style,
            chat_log=chat_log,
            messages=messages,
            focus_chat=focus_chat,
            deadline=deadline,
            recent=recent,
            self_name=self_name,
            reply_style=style,
        )

    async def _reply_context(
        self,
        chat_key: ChatKey,
        recent: Sequence[Message],
        self_name: str | None,
        reply_to: str | None,
        length_style: str = "",
    ) -> ReplyContext:
        """The replyer's view of the conversation.

        Built from the SAME ``recent`` snapshot the planner scored, so the two
        stages never disagree about what was said. The drift block is a pure
        function of config, so it costs nothing to compute here; the
        impressions are a bounded read of people already in the window.
        """
        target = None
        if reply_to:
            for msg in recent:
                if str(msg.id) == str(reply_to):
                    target = msg
                    break
        return ReplyContext(
            chat_history=tuple(recent),
            target=target,
            bot_name=self._bot_name(self_name),
            now=self._now() if self._now is not None else None,
            drift_block=self._drift_block(),
            length_style=length_style or "",
            impressions=await self._impressions(chat_key, recent),
        )

    #: How many people's impressions ride along in one reply request. Bounded
    #: like every other adaptive slot: a wall of dossiers is not context.
    MAX_IMPRESSIONS = 3

    async def _impressions(
        self, chat_key: ChatKey, recent: Sequence[Message]
    ) -> tuple[tuple[str, str], ...]:
        """What the bot thinks of the people who actually spoke here.

        Read from ``persons`` (the impression learner's projection) for the
        distinct non-self senders in the current window, newest speaker
        first. Every failure degrades to no impressions — never a raise, and
        never a provider call.

        Both the name and the impression are escaped: the name came off the
        wire and the impression is model-written text about untrusted chat,
        so neither may close the prompt's own structure.
        """
        if self._person_service is None:
            return ()
        seen: list[SenderId] = []
        for msg in reversed(list(recent)):
            if msg.is_self or msg.sender_id in seen:
                continue
            seen.append(msg.sender_id)
            if len(seen) >= self.MAX_IMPRESSIONS:
                break
        out: list[tuple[str, str]] = []
        for uid in seen:
            try:
                profile = await self._person_service.get_profile(chat_key, uid)
            except Exception:
                continue
            if profile is None or not (profile.impression or "").strip():
                continue
            name = profile.names[0] if profile.names else str(uid)
            out.append((
                escape_untrusted(name),
                escape_untrusted(profile.impression.strip()),
            ))
        return tuple(out)

    def _bot_name(self, self_name: str | None) -> str:
        """The name the prompts address the bot by."""
        if self_name:
            return self_name
        return self._cfg.bot.name if self._cfg is not None else ""

    def _drift_block(self) -> str:
        """The attention-drift prompt block for this chat, or ``""``."""
        if self._cfg is None:
            return ""
        return build_drift_block(self._cfg.drift)

    async def _run_injected(
        self,
        *,
        chat_key: ChatKey,
        identity: str,
        behavior_style: str,
        chat_log: str,
        messages: Sequence[TranscriptMessage],
        focus_chat: str | None,
        deadline: float | None,
        reply_style: str,
        recent: Sequence[Message] = (),
        self_name: str | None = None,
    ) -> AgentOutcome:
        """The injected-seam form: call the injected planner/replyer directly
        with NO saga-level budget accounting (the injected seams own their
        own per-call budget)."""
        result = await self._planner.plan(
            messages,
            chat_log=chat_log,
            reply_style=reply_style,
            focus_chat=focus_chat,
            bot_name=self._bot_name(self_name),
            drift_block=self._drift_block(),
            behavior_style=behavior_style,
            tools=self._tools,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            deadline=deadline,
            max_tool_rounds=self._max_tool_rounds,
        )
        if result.intent == PlanIntent.REPLY:
            draft = await self._replyer.reply(
                reply_reference=result.reply_reference or "",
                identity=identity,
                reply_style=reply_style,
                reply_to=result.reply_to,
                context=await self._reply_context(
                    chat_key, recent, self_name, result.reply_to,
                    result.length_style,
                ),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                deadline=deadline,
            )
            tokens_in = result.tokens_in + draft.tokens_in
            tokens_out = result.tokens_out + draft.tokens_out
            usage = _merge_usage(result.usage, draft.usage)
            if draft.no_output:
                return AgentOutcome(
                    intent="reply",
                    reply_text=None,
                    reply_to=draft.reply_to,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    usage=usage,
                    end_reason="reply_no_output",
                )
            return AgentOutcome(
                intent="reply",
                reply_text=draft.text,
                reply_to=draft.reply_to,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                usage=usage,
                end_reason="reply",
            )
        if result.intent == PlanIntent.WAIT:
            return AgentOutcome(
                intent="wait",
                wait_seconds=result.wait_seconds,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                usage=result.usage,
                end_reason="wait",
            )
        return AgentOutcome(
            intent="no_action",
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            usage=result.usage,
            end_reason=result.end_reason or "no_action",
        )

    async def _run_budgeted(
        self,
        *,
        chat_key: ChatKey,
        identity: str,
        behavior_style: str,
        chat_log: str,
        messages: Sequence[TranscriptMessage],
        focus_chat: str | None,
        chat_kind: str,
        self_name: str | None,
        deadline: float | None,
        recent: Sequence[Message] = (),
        reply_style: str,
    ) -> AgentOutcome:
        """The default-build form: wrap the LLM client in a chat-bound
        ``BudgetedClient`` and build a fresh Planner/Replyer over it, so the
        budget is enforced per provider call for BOTH the planner and the
        replyer."""
        assert self._llm is not None
        assert self._prompts is not None
        assert self._registry is not None
        assert self._context_config is not None
        assert self._budget is not None
        assert self._agent_config is not None
        # Honor per-chat agent/budget/context settings at EACH provider call:
        # resolve the chat's effective config (caps/rungs, fallback profile,
        # context) rather than binding top-level settings globally. When
        # cfg/repo/now are not provided (tests), fall back to the injected
        # top-level budget/context/agent config.
        if self._cfg is not None and self._repo is not None and self._now is not None:
            chat_cfg = self._cfg.for_chat(chat_key)
            budget = self._budget_for(chat_key, chat_cfg.budget)
            context_config = chat_cfg.context
            agent_config = chat_cfg.agent
        else:
            budget = self._budget
            context_config = self._context_config
            agent_config = self._agent_config
        budgeted = BudgetedClient(
            self._llm,
            budget,
            chat_key,
            agent_config=agent_config,
        )
        # The last ToolContext the planner's factory built: the round that
        # staged the terminal verdict owns the staged media intent (the
        # planner's PlanResult does not carry the context).
        last_ctx: list[ToolContext | None] = [None]

        def tool_context_factory() -> ToolContext:
            ctx = ToolContext(
                chat_key=chat_key,
                chat_kind=chat_kind,
                # Real budgeted ToolContext: inject the adapter-supported
                # capabilities, the current dispatch's recent messages, and
                # the safely chat-scoped forwards so fetch_history /
                # view_forward_message are reachable in production when the
                # adapter provides the data, and fail closed otherwise. The
                # recent messages and forwards are scoped to this chat by
                # construction — no cross-chat history/forward access.
                capabilities=self._capabilities,
                recent=tuple(recent),
                forwards=(
                    self._forward_resolver(chat_key)
                    if self._forward_resolver is not None
                    else None
                ),
                registry=self._registry,
                self_name=self_name,
                knowledge=self._knowledge_callbacks(chat_key),
                # Phase 6 P6.5b media wiring: the chat-bound catalog
                # callbacks the deferred send_emoji / send_image tools speak
                # to (None disables them — they fail closed).
                media=(
                    self._media_callbacks(chat_key)
                    if self._media_callbacks is not None
                    else None
                ),
                # Phase 6 P6.6b chat-control wiring: the chat-bound
                # callbacks the deferred set_focus / notify_chat tools speak
                # to (None disables them — they fail closed).
                chat_controls=(
                    self._chat_control_callbacks(chat_key)
                    if self._chat_control_callbacks is not None
                    else None
                ),
            )
            last_ctx[0] = ctx
            return ctx

        planner = Planner(
            budgeted,
            self._prompts,
            self._registry,
            context_config,
            tool_context_factory=tool_context_factory,
            max_tool_rounds=(
                self._max_tool_rounds
                if self._max_tool_rounds is not None
                else DEFAULT_MAX_TOOL_ROUNDS
            ),
            repair=self._repair,
        )
        replyer = Replyer(budgeted, self._prompts)
        try:
            result = await planner.plan(
                messages,
                chat_log=chat_log,
                reply_style=reply_style,
                focus_chat=focus_chat,
                bot_name=self._bot_name(self_name),
                drift_block=self._drift_block(),
                behavior_style=behavior_style,
                tools=self._tools,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                deadline=deadline,
                max_tool_rounds=self._max_tool_rounds,
            )
        except BudgetBlockedError:
            return AgentOutcome(
                intent="budget_blocked", end_reason="budget_blocked"
            )
        # The staged media send (Phase 6 P6.5b): the round that staged it
        # owns the typed intent. A media send is a terminal reply variant —
        # the replyer is never invoked and the marker is never treated as
        # reply text.
        media_intent = last_ctx[0].media_intent if last_ctx[0] is not None else None
        # The staged chat controls (Phase 6 P6.6b): the round that staged
        # them owns the typed intents. They ride along with the terminal
        # verdict (NOT mutually exclusive) and are applied by the
        # CycleRunner after the terminal settlement (LIVE only).
        chat_controls = (
            last_ctx[0].chat_controls if last_ctx[0] is not None else ()
        )
        if result.intent == PlanIntent.REPLY:
            if media_intent is not None:
                return AgentOutcome(
                    intent="reply",
                    media_intent=media_intent,
                    chat_controls=chat_controls,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    usage=result.usage,
                    end_reason="media",
                )
            try:
                draft = await replyer.reply(
                    reply_reference=result.reply_reference or "",
                    identity=identity,
                    reply_style=reply_style,
                    reply_to=result.reply_to,
                    context=await self._reply_context(
                        chat_key, recent, self_name, result.reply_to,
                        result.length_style,
                    ),
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    deadline=deadline,
                )
            except BudgetBlockedError:
                return AgentOutcome(
                    intent="budget_blocked", end_reason="budget_blocked"
                )
            tokens_in = result.tokens_in + draft.tokens_in
            tokens_out = result.tokens_out + draft.tokens_out
            usage = _merge_usage(result.usage, draft.usage)
            if draft.no_output:
                return AgentOutcome(
                    intent="reply",
                    reply_text=None,
                    reply_to=draft.reply_to,
                    chat_controls=chat_controls,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    usage=usage,
                    end_reason="reply_no_output",
                )
            return AgentOutcome(
                intent="reply",
                reply_text=draft.text,
                reply_to=draft.reply_to,
                chat_controls=chat_controls,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                usage=usage,
                end_reason="reply",
            )
        if result.intent == PlanIntent.WAIT:
            return AgentOutcome(
                intent="wait",
                wait_seconds=result.wait_seconds,
                chat_controls=chat_controls,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                usage=result.usage,
                end_reason="wait",
            )
        # no_action — including every degraded planner exit (no_tool_call,
        # empty_response, tool_round_cap, malformed tool JSON).
        return AgentOutcome(
            intent="no_action",
            chat_controls=chat_controls,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            usage=result.usage,
            end_reason=result.end_reason or "no_action",
        )

    def _budget_for(
        self, chat_key: ChatKey, budget_config: BudgetConfig
    ) -> BudgetManager:
        """The chat-bound BudgetManager for ``chat_key``, built from the
        chat's effective ``BudgetConfig`` (caps/rungs) and cached per chat.
        The KV usage lives in the shared Repository, so a per-chat manager
        is stateless apart from its config and lock — rebuilding on a config
        change is safe and never loses usage."""
        mgr = self._budgets.get(chat_key)
        if mgr is None or mgr.config != budget_config:
            assert self._repo is not None and self._now is not None
            mgr = BudgetManager(self._repo, budget_config, now=self._now)
            self._budgets[chat_key] = mgr
        return mgr

    def _knowledge_callbacks(self, chat_key: ChatKey) -> KnowledgeCallbacks | None:
        """The chat-scoped knowledge callbacks for the deferred knowledge
        tools, bound to ``chat_key`` — or None when the knowledge wiring is
        unavailable (no MemorySearch / PersonService). The callbacks are the
        ONLY surface the tools speak to; they never hold a repository
        reference, and being bound to ``chat_key`` makes a cross-chat request
        impossible by construction."""
        if self._memory_search is None or self._person_service is None:
            return None
        jargon = self._jargon_query
        return KnowledgeCallbacks(
            query_memory=lambda query, limit: self._memory_search.search(
                chat_key, query, limit=limit
            ),
            query_person=lambda uid: self._person_service.get_profile(
                chat_key, uid
            ),
            query_jargon=(
                (lambda query, limit: jargon(chat_key, query, limit))
                if jargon is not None
                else None
            ),
        )


def _merge_usage(*usages: dict[str, int]) -> dict[str, int]:
    """Sum the integer usage counters across planner/replyer results."""
    out: dict[str, int] = {}
    for usage in usages:
        for key, value in (usage or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                out[key] = out.get(key, 0) + value
    return out


#: What the replyer is told when the expression learner has produced
#: nothing yet (a fresh deployment, or [learn] disabled).
DEFAULT_REPLY_STYLE = "自然"


def _render_reply_style(expression: Sequence[Record]) -> str:
    """Render the selected expression records into the replyer's style block.

    MaiBot pairs each learned style with the SITUATION it was observed in
    (``expression_learner`` records ``{situation, style}``) and shows the
    replyer several of them, so the model can pick the one that fits the
    current mood. Taking ``expression[0]["style"]`` and calling that the
    entire style — which is what this did — discards the situation, discards
    every other candidate, and collapses a learned repertoire into one word.

    Records missing a usable style are skipped; an empty pool falls back to
    ``DEFAULT_REPLY_STYLE``.
    """
    lines: list[str] = []
    for rec in expression:
        payload = rec.payload or {}
        style = payload.get("style")
        if not isinstance(style, str) or not style.strip():
            continue
        situation = payload.get("situation")
        if isinstance(situation, str) and situation.strip():
            lines.append(f"当{situation.strip()}时，{style.strip()}")
        else:
            lines.append(style.strip())
    if not lines:
        return DEFAULT_REPLY_STYLE
    return "\n".join(lines)


def _render_chat_log(recent: tuple[Message, ...], self_name: str | None) -> str:
    """The plain-text chat rendering the planner prompt embeds.

    Each line renders the clock time, the display name, the message text, and
    the exact JSON-escaped sender UID (``[uid="..."]``) so the planner can
    pass the UID verbatim as the ``platform_uid`` argument to
    ``query_person_profile`` — no nickname lookup is ever performed.

    The timestamp matters: this bot's gate deliberately delays replies by
    minutes, so without it the model cannot tell a five-second-old message
    from a twelve-hour-old one and answers stale conversations as if they
    were live.
    """
    lines = []
    for msg in recent:
        name = self_name if msg.is_self else msg.sender_name
        uid = json.dumps(str(msg.sender_id), ensure_ascii=False)
        clock = _clock_hhmm(msg.recv_ts)
        prefix = f"[{clock}] " if clock else ""
        lines.append(f"{prefix}{name}: {msg.text} [uid={uid}]")
    return "\n".join(lines)


def _clock_hhmm(ts: float | None) -> str:
    """``HH:MM`` for a chat-log line, or ``""`` when the stamp is unusable."""
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(ts).strftime("%H:%M")
    except (OverflowError, OSError, ValueError):
        return ""


def _pending_text(pending: tuple[Message, ...]) -> str:
    """The plain text of the claimed pending messages (the jargon scoping
    input)."""
    return " ".join(m.text for m in pending if m.text)


def _recent_text(recent: tuple[Message, ...]) -> str:
    """The plain text of the recent window (the jargon scoping input)."""
    return " ".join(m.text for m in recent if m.text)


def _pending_transcript(pending: tuple[Message, ...]) -> tuple[TranscriptMessage, ...]:
    """The claimed pending messages as the planner's user-turn transcript.

    Each user turn carries the sender display name and the exact
    JSON-escaped sender UID (``[uid="..."]``) so profile lookup has a
    compliant ``platform_uid`` tool argument. The transcript stays legal
    (role/content only); no nickname lookup is performed.
    """
    return tuple(
        TranscriptMessage(
            role="user",
            content=(
                f"{msg.sender_name}: {msg.text}"
                f" [uid={json.dumps(str(msg.sender_id), ensure_ascii=False)}]"
            ),
        )
        for msg in pending
    )


# ── The CycleRunner (Scheduler CycleFn, no Adapter dependency) ──────────────

def _gate_fingerprint(gate: Any) -> tuple[str, ...]:
    """The ordered gate-feature fingerprint: the feature names in
    registration order — the deterministic identity of the gate's scoring
    composition (Phase 6 P6.6). Exact replay compares the fingerprint
    recorded in a trace's config against the current gate and fails closed
    on a mismatch."""
    current = getattr(gate, "_current_features", None)
    if callable(current):
        features = current()
        return tuple(f.name for f in features)  # type: ignore[union-attr]
    reg = getattr(gate, "_registry", None)
    if reg is not None:
        return tuple(f.name for f in reg.all())
    return ()


def _composition_fingerprint(gate: Any) -> dict[str, Any]:
    """The complete exact-replay identity of the scoring composition."""
    manifests = getattr(gate, "_plugin_manifest", ())
    return {
        "plugin_manifest": [
            m.as_dict() if hasattr(m, "as_dict") else dict(m) for m in manifests
        ],
        "gate_features": list(_gate_fingerprint(gate)),
        "gate_feature_implementations": list(feature_implementation_fingerprint(gate)),
    }


def _trace_with_fingerprint(
    trace: DecisionTrace, gate: Any, *, plugin_manifest: Any = None
) -> DecisionTrace:
    """A copy of ``trace`` whose ``config`` carries the gate-feature
    fingerprint (``config["gate_features"]``), so every persisted trace —
    live and replay — records the exact scoring composition it was produced
    under."""
    if plugin_manifest is not None:
        setattr(gate, "_plugin_manifest", tuple(plugin_manifest))
    composition = _composition_fingerprint(gate)
    return dataclasses.replace(
        trace,
        config={
            **trace.config,
            "gate_features": composition["gate_features"],
            "gate_feature_implementations": composition[
                "gate_feature_implementations"
            ],
            "plugin_manifest": composition["plugin_manifest"],
            # Structured alias for consumers that treat the complete
            # composition as one replay fingerprint.  Keep the component
            # keys above for backwards-compatible trace inspection.
            "composition_fingerprint": composition,
        },
    )


def _set_replay_manifest(gate: Any, cfg: Config) -> None:
    """Attach the import-free current manifest to a replay gate.

    Replay deliberately never calls ``PluginLoader.load``.  A gate supplied
    by a caller may already carry a live manifest; otherwise only the static
    configured manifest is attached.
    """
    if not hasattr(gate, "_plugin_manifest"):
        try:
            setattr(gate, "_plugin_manifest", configured_plugin_manifest(cfg))
        except Exception as exc:
            raise ValueError(f"cannot read plugin manifest: {exc}") from None


def _check_exact_fingerprint(dispatch: CorpusMarker, gate: Any) -> None:
    """Require and compare every component of the exact composition identity."""
    if dispatch.trace_json is None:
        raise ValueError(
            f"dispatch marker {dispatch.sequence} fingerprint missing"
        )
    try:
        recorded = json.loads(dispatch.trace_json)
    except (json.JSONDecodeError, TypeError):
        raise ValueError(
            f"malformed trace_json (dispatch {dispatch.sequence})"
        ) from None
    config = recorded.get("config") if isinstance(recorded, dict) else None
    if not isinstance(config, dict):
        raise ValueError(f"dispatch marker {dispatch.sequence} fingerprint missing")
    required = (
        "plugin_manifest",
        "gate_features",
        "gate_feature_implementations",
    )
    if any(key not in config for key in required):
        raise ValueError(f"dispatch marker {dispatch.sequence} fingerprint missing")
    current = _composition_fingerprint(gate)
    recorded_fp = {
        "plugin_manifest": config["plugin_manifest"],
        "gate_features": config["gate_features"],
        "gate_feature_implementations": config["gate_feature_implementations"],
    }
    if recorded_fp != current:
        raise ValueError(
            f"dispatch marker {dispatch.sequence} fingerprint mismatch"
            f" ({recorded_fp!r} != {current!r})"
        )


class CycleRunner:
    """One gate→disposition saga per chat, typed against the Repository
    seam. Claims a chat, evaluates the gate on the assembled snapshot, and
    applies exactly the frozen disposition (see module docstring). Returns
    the gate's ``Decision`` so the scheduler can re-arm (timed vs
    event-only). Timing and cycle ids are injectable for deterministic
    tests.

    Two entry points share the same assembler + gate + disposition logic:

    - ``__call__(chat_key)`` — the LEGACY cycle lane: claims through
      ``Repository.claim_cycle`` and settles through the legacy
      release/finish surface. Preserved for the current App; the next
      integration lane switches live use to the ledger.
    - ``run_dispatch(grant)`` — the DISPATCH-ledger lane (the
      ``LedgerScheduler`` handler shape): consumes a ``DispatchGrant``
      already claimed by ``Repository.begin_dispatch``, never touches the
      legacy claim surface, and settles every disposition exactly once
      through ``Repository.settle_dispatch``. An ordinary exception or
      cancellation leaves the prepared dispatch recoverable by its lease.
    """

    def __init__(
        self,
        repo: Repository,
        gate: Gate,
        cfg: Config,
        *,
        clock: Any = None,
        hooks: HookBus | None = None,
        dry_run: bool = True,
        uuid_fn: Callable[[], str] | None = None,
        claim_lease_s: float = CLAIM_LEASE_S,
        snapshot_limit: int = SNAPSHOT_LIMIT,
        window_s: float = SNAPSHOT_WINDOW_S,
        trace_sink: Callable[[DecisionTrace], None] | None = None,
        marker_exporter: Callable[[CorpusMarker], Awaitable[None]] | None = None,
        agent: PhaseAgent | None = None,
        on_outbox: Callable[[list[OutboxItem]], Awaitable[None]] | None = None,
        on_memory: Callable[[ChatKey, MessageRowId], Awaitable[None]] | None = None,
        prompts: PromptStore | None = None,
        output_pipeline: Callable[[OutputConfig], OutputPipeline] | None = None,
        adaptive: AdaptiveContextService | None = None,
        on_settled: Callable[[ChatKey, MessageRowId], Awaitable[None]] | None = None,
        on_exposure: Callable[
            [ChatKey, tuple[Record, ...], int, MessageRowId], Awaitable[None]
        ] | None = None,
        on_chat_control: Callable[[ChatKey], Awaitable[None]] | None = None,
    ) -> None:
        self._repo = repo
        self._gate = gate
        self._cfg = cfg
        self._clock = clock if clock is not None else RealClock()
        self._hooks = hooks
        self._dry_run = dry_run
        self._uuid = uuid_fn if uuid_fn is not None else lambda: str(uuid.uuid4())
        self._claim_lease_s = claim_lease_s
        self._snapshot_limit = snapshot_limit
        self._window_s = window_s
        self._trace_sink = trace_sink
        # The at-least-once dispatch marker exporter (wired to
        # ``record.export_marker`` by the caller); None disables the
        # runner-side dispatch marker export.
        self._marker_exporter = marker_exporter
        # The Phase 3 agent coordinator (planner + replyer + optional budget);
        # None preserves the safe no-agent behavior: a trigger outside dry-run
        # is never retained and never sends.
        self._agent = agent
        # The live outbox wake callback: fired after a terminal finish that
        # created outbox rows (the App wakes its outbox worker on it). None
        # in dry-run / no-agent modes.
        self._on_outbox = on_outbox
        # The contained memory-maintenance callback: fired from TERMINAL
        # dispatch settlement ONLY (after the durable finish), passing the
        # chat and the frozen through boundary, so the App can summarize one
        # oldest unprocessed local batch. Never fired on release/delay/defer/
        # replay/gate. Failures are contained by the caller.
        self._on_memory = on_memory
        # The Phase 6 P6.4b adaptive context service: queried ONLY after the
        # gate triggers, in LIVE dispatches only (never dry-run/replay/
        # doctor). None disables adaptive context (the agent falls back to
        # the "自然" empty context).
        self._adaptive = adaptive
        # The contained post-terminal settlement callback: fired from
        # TERMINAL dispatch settlement ONLY, AFTER settle_dispatch, the
        # outbox wake, and the marker export — the App enqueues the chat for
        # its learners here (non-blocking; settlement never awaits LLM/learn/
        # media). Failures are contained by the caller.
        self._on_settled = on_settled
        # The contained post-terminal exposure callback: fired after a
        # terminal reply dispatch that created durable outbox output
        # (LIVE-only), with the records that were actually frozen into the
        # prompt and the dispatch id. Failures are contained by the caller.
        self._on_exposure = on_exposure
        # A durable control settlement wakes the TARGET through the scheduler
        # without creating a synthetic inbound message.
        self._on_chat_control = on_chat_control
        # The PromptStore used to resolve the configured bot identity file
        # into the planner/replyer runtime prompts (built from the config's
        # prompt_dir when not injected).
        self._prompts = prompts if prompts is not None else PromptStore(
            self._cfg.bot.prompt_dir
        )
        # The output-pipeline factory: builds an ``OutputPipeline`` from a
        # chat's effective ``OutputConfig`` (sanitize → split → typo). The
        # agent reply is run through it ONCE at terminal settlement; the
        # durable outbox rows it produces are never re-processed on retry/
        # restart. Injectable for tests.
        self._output_factory = (
            output_pipeline if output_pipeline is not None else OutputPipeline
        )

    async def __call__(self, chat_key: ChatKey) -> Decision:
        """One cycle for ``chat_key``: claim → snapshot → evaluate →
        frozen disposition. Returns the Decision the scheduler re-arms on.

        The claim comes FIRST — before any identity/state/recent/history
        read — so the claim-bounded grant is the only boundary the
        snapshot can be built from, and a racing claim (another live
        cycle) is detected before any state is read. Once claimed, the
        cycle MUST settle: an ordinary exception or cancellation releases
        the claim in ``finally``, so a durable claim is never stranded.
        Terminal completion fences against a FRESH clock timestamp (the
        lease may have expired while the cycle ran; an expired owner
        cannot finish). The pure Gate applies the full precedence after
        claiming: direct @ / quote, focus, and high pending bypass an
        active durable hold; an expired hold never regenerates a fresh
        hold from stale history.
        """
        now = self._clock.now()
        cycle_id = CycleId(self._uuid())
        claim = CycleClaim(
            chat_key, cycle_id, started_ts=now, expires_at=now + self._claim_lease_s
        )
        grant = await self._repo.claim_cycle(claim)
        if grant is None:
            # Unknown chat, or another live cycle owns it (or the claim
            # raced): nothing to evaluate — the owner's release re-wakes.
            return Decision(action="skip", reason=Reason.SKIP)
        if isinstance(grant, ClaimBusy):
            # A LIVE, unexpired cycle owns the chat (e.g. a crash mid-cycle
            # whose lease has not yet expired). Map the busy horizon to a
            # TIMED delay so the scheduler re-arms at/after the exact
            # ``busy_until``: the next wake finds the lease expired,
            # recovers the claim, and gets a grant — WITHOUT new input.
            # Never a terminal skip (the pending messages stay pending for
            # the recovered claim) and never a trace (nothing was
            # evaluated; the owner's cycle owns the evidence).
            return Decision(
                action="delay",
                reason=Reason.DELAY,
                delay_seconds=max(0.0, grant.busy_until - now),
            )
        settled = False
        try:
            identity = await self._repo.get_chat(chat_key)
            if identity is None:
                # The claim succeeded, so the chat row exists; defensive.
                return Decision(action="skip", reason=Reason.SKIP)
            state = await self._repo.get_chat_state(chat_key)
            state = state if state is not None else ChatState(chat_key=chat_key)
            if not self._dry_run:
                state = await self._merge_active_controls(chat_key, state)
            previous_end_reason = await self._repo.get_latest_terminal_end_reason(
                chat_key
            )
            recent = await self._repo.get_recent_snapshot(
                chat_key,
                grant.through_msg_id,
                since_ts=now - self._window_s,
                limit=self._snapshot_limit,
            )
            quote_self_ids = await self._resolve_quote_targets(
                chat_key, grant.pending
            )
            snapshot = assemble_snapshot(
                grant=grant,
                identity=identity,
                state=state,
                recent=recent,
                cfg=self._cfg.for_chat(chat_key),
                now=now,
                self_name=self._cfg.bot.name,
                self_aliases=tuple(self._cfg.bot.alias_names),
                muted=is_muted(self._cfg.access, chat_key),
                previous_end_reason=previous_end_reason,
                window_s=self._window_s,
                quote_self_ids=quote_self_ids,
            )
            trace = self._gate.evaluate(snapshot)
            trace = _trace_with_fingerprint(trace, self._gate)
            if self._trace_sink is not None:
                self._trace_sink(trace)
            decision = trace.decision
            assert decision is not None
            if decision.action == "trigger":
                if self._dry_run:
                    await self._finish(chat_key, cycle_id, "dry_run_trigger", trace)
                else:
                    # Never retain a trigger claim without an agent: release
                    # it and return the trigger decision event-only, so the
                    # scheduler does not re-arm a timed wake for it.
                    await self._repo.release_cycle(chat_key, cycle_id)
            elif decision.action == "skip":
                await self._finish(chat_key, cycle_id, "skip", trace)
            else:
                # Ordinary / active-hold delay: release with no cursor or
                # session mutation; the pending messages stay pending.
                await self._repo.release_cycle(chat_key, cycle_id)
            settled = True
            return decision
        finally:
            if not settled:
                # An ordinary exception or a cancellation: never strand the
                # durable claim. release_cycle is a no-op once the claim is
                # finished or already released/recovered by another cycle.
                await self._repo.release_cycle(chat_key, cycle_id)

    async def run_dispatch(self, grant: DispatchGrant) -> Decision:
        """One dispatch-ledger cycle for a granted dispatch (the
        ``LedgerScheduler`` handler shape).

        The scheduler has already claimed the dispatch through
        ``Repository.begin_dispatch``; this method consumes the grant's
        frozen claim/pending/boundary and reads only the
        identity/state/recent/history needed to build the SAME shared
        ``GateSnapshot``. It NEVER calls the legacy claim/renew/release/
        finish surface: every settled disposition goes through
        ``Repository.settle_dispatch`` exactly once, and an ordinary
        exception or cancellation leaves the prepared dispatch recoverable
        by its lease (no unsafe legacy release). Returns the gate's
        ``Decision`` so the scheduler can re-arm (timed vs event-only).
        """
        now = self._clock.now()
        chat_key = grant.claim.chat_key
        identity = await self._repo.get_chat(chat_key)
        if identity is None:
            # Defensive: a grant implies the chat row exists. Give the
            # claim back without cursor/outbox movement so the prepared
            # dispatch is not stranded.
            await self._settle(grant, outcome="release", now=self._clock.now())
            return Decision(action="skip", reason=Reason.SKIP)
        state = await self._repo.get_chat_state(chat_key)
        state = state if state is not None else ChatState(chat_key=chat_key)
        # Phase 6 P6.6b chat controls (LIVE only): merge the chat's ACTIVE
        # internal focus events into the durable state BEFORE the snapshot
        # is assembled, so the target chat's gate evaluates as focused and
        # the notify traverses the target gate. Dry-run/replay never read
        # controls (the frozen snapshot carries the focus facts there).
        if not self._dry_run:
            state = await self._merge_active_controls(chat_key, state)
        previous_end_reason = await self._repo.get_latest_terminal_end_reason(
            chat_key
        )
        recent = await self._repo.get_recent_snapshot(
            chat_key,
            grant.through_msg_id,
            since_ts=now - self._window_s,
            limit=self._snapshot_limit,
        )
        quote_self_ids = await self._resolve_quote_targets(chat_key, grant.pending)
        # The assembler reads only the frozen claim/pending/boundary; the
        # DispatchGrant carries exactly those fields, so a ClaimGrant view
        # is built from the grant (never re-read from the repository).
        claim_grant = ClaimGrant(
            # DispatchId is the replay-visible cycle identity. The durable
            # claim retains its UUID for ownership/settlement, while every
            # live and replay trace can be joined deterministically by the
            # persisted dispatch marker.
            claim=CycleClaim(
                chat_key,
                CycleId(f"dispatch:{grant.dispatch_id}"),
                grant.claim.started_ts,
                grant.claim.expires_at,
            ),
            start_msg_id=grant.start_msg_id,
            through_msg_id=grant.through_msg_id,
            pending=grant.pending,
        )
        snapshot = assemble_snapshot(
            grant=claim_grant,
            identity=identity,
            state=state,
            recent=recent,
            cfg=self._cfg.for_chat(chat_key),
            now=now,
            self_name=self._cfg.bot.name,
            self_aliases=tuple(self._cfg.bot.alias_names),
            muted=is_muted(self._cfg.access, chat_key),
            previous_end_reason=previous_end_reason,
            window_s=self._window_s,
            quote_self_ids=quote_self_ids,
        )
        snapshot_json = json.dumps(dataclasses.asdict(snapshot), default=str)
        trace = self._gate.evaluate(snapshot)
        trace = _trace_with_fingerprint(trace, self._gate)
        if self._trace_sink is not None:
            self._trace_sink(trace)
        decision = trace.decision
        assert decision is not None
        # The gate verdict is the first fork in "why did/didn't it speak".
        # It exists in the ledger, but only the log answers the question
        # while the operator is watching.
        log.info(
            "gate %s for %s: score=%.0f/%d pending=%d reason=%s delay=%s",
            decision.action,
            grant.claim.chat_key,
            decision.score,
            snapshot.trigger_score,
            decision.pending,
            decision.reason,
            (
                f"{decision.delay_seconds:.1f}s"
                if decision.delay_seconds is not None
                else "event-only"
            ),
        )
        if decision.action == "trigger":
            if self._agent is not None:
                # Phase 3 agent lane: budget decide → planner tool loop →
                # replyer draft → terminal ledger settlement (see
                # ``_run_agent_dispatch``).
                return await self._run_agent_dispatch(
                    grant, trace, snapshot, snapshot_json, now
                )
            if self._dry_run:
                await self._finish_dispatch(
                    grant,
                    "dry_run_trigger",
                    trace,
                    evaluated_ts=now,
                    snapshot_json=snapshot_json,
                )
            else:
                # Never retain a trigger dispatch without an agent: release
                # it and return the trigger decision event-only, so the
                # scheduler does not re-arm a timed wake for it.
                await self._settle(
                    grant,
                    outcome="release",
                    trace=trace,
                    now=self._clock.now(),
                    evaluated_ts=now,
                    snapshot_json=snapshot_json,
                )
        elif decision.action == "skip":
            await self._finish_dispatch(
                grant,
                "skip",
                trace,
                evaluated_ts=now,
                snapshot_json=snapshot_json,
            )
        else:
            # Ordinary / active-hold delay: release with no cursor or
            # session mutation; the pending messages stay pending.
            await self._settle(
                grant,
                outcome="delay",
                trace=trace,
                now=self._clock.now(),
                evaluated_ts=now,
                snapshot_json=snapshot_json,
            )
        return decision

    async def _run_agent_dispatch(
        self,
        grant: DispatchGrant,
        trace: DecisionTrace,
        snapshot: GateSnapshot,
        snapshot_json: str,
        evaluated_ts: float,
    ) -> Decision:
        """One agent cycle for a granted trigger dispatch.

        The agent coordinator runs the deterministic sequence (per-call
        budgeted planner tool loop → replyer draft) and the runner settles
        the outcome through the ledger exactly once:

        - the whole saga is bounded by an aggregate deadline
          (``now + max_execution_s``) and the dispatch lease is renewed
          before/through the run so the foundation's settlement fencing
          never rejects a legitimate terminal settle;
        - a recoverable provider failure/timeout (``LLMTransientError``)
          defers the retry atomically (cursor/outbox unchanged) and returns
          a timed decision at ``retry_delay_s``;
        - ``wait`` defers with ``defer_kind="wait"`` (the foundation
          persists the barrier and increments the wait streak; no early
          priority execution; restart honors the remaining delay) — the
          THIRD consecutive wait terminally consumes with
          ``planner_wait_rest``, clearing the barrier/streak;
        - ``reply`` with a nonempty draft settles a terminal finish with the
          ordered outbox batch OUTSIDE dry-run (dry-run evaluates but
          creates zero outbox rows and never sends);
        - ``reply`` with no usable draft, ``no_action``, and
          ``budget_blocked`` terminally consume the cursor with an EMPTY
          outbox (resetting the wait barrier/streak).

        The terminal settle reuses ``_finish_dispatch``, so the fresh-clock
        settlement fencing, marker export, and ``on_cycle_end`` hook ordering
        are exactly the frozen ones. The returned Decision is what the
        scheduler re-arms on: a timed delay for ``wait``/retry, the gate's
        event-only trigger for every terminal outcome.
        """
        assert self._agent is not None
        chat_key = grant.claim.chat_key
        agent_cfg = self._cfg.for_chat(chat_key).agent
        now = self._clock.now()
        deadline = now + agent_cfg.max_execution_s
        # Phase 6 P6.4b adaptive context: computed ONCE per live dispatch,
        # AFTER the gate triggered, and frozen for every planner tool round
        # and the replyer. Never queried in dry-run/replay/doctor — the
        # agent falls back to the "自然" empty context.
        adaptive: AdaptiveContext | None = None
        chat_log = _render_chat_log(snapshot.recent, snapshot.self_name)
        if not self._dry_run and self._adaptive is not None:
            try:
                adaptive = await self._adaptive.build(
                    chat_key,
                    pending_text=_pending_text(snapshot.pending_messages),
                    recent_text=_recent_text(snapshot.recent),
                    mode=RuntimeMode.LIVE,
                )
            except Exception:
                log.warning(
                    "adaptive context failed for %s (contained)", chat_key,
                    exc_info=True,
                )
            if adaptive is not None and adaptive.rendered:
                chat_log = chat_log + "\n\n" + adaptive.rendered
        # Phase 6 P6.5b approved catalog listing (LIVE only): the planner may
        # select an approved asset by its OPAQUE id via send_emoji /
        # send_image. The listing carries only ids + scrubbed descriptions —
        # never URLs, file paths, platform refs, or base64. Never in
        # dry-run/replay/doctor.
        if not self._dry_run:
            catalog = await self._catalog_prompt(chat_key)
            if catalog:
                chat_log = chat_log + "\n\n" + catalog
        # Phase 6 P6.6b internal notifications (LIVE only): the payloads of
        # the chat's ACTIVE notify controls are appended to the chat log, so
        # the agent knows why it is responding. Never in dry-run/replay.
        if not self._dry_run:
            notify = await self._notify_text(chat_key)
            if notify:
                chat_log = chat_log + "\n\n" + notify
        reply_style = adaptive.reply_style if adaptive is not None else "自然"
        try:
            outcome = await self._run_with_renewal(
                grant,
                self._agent.run(
                    chat_key=chat_key,
                    identity=self._resolve_identity(),
                    behavior_style=self._resolve_behavior_style(),
                    chat_log=chat_log,
                    messages=_pending_transcript(snapshot.pending_messages),
                    chat_kind="group" if snapshot.is_group else "private",
                    self_name=snapshot.self_name,
                    deadline=deadline,
                    recent=snapshot.recent,
                    reply_style=reply_style,
                ),
                deadline=deadline,
            )
        except LLMTransientError as exc:
            # Recoverable provider failure/timeout: defer the retry
            # atomically (cursor/outbox unchanged) and return a timed
            # decision so the scheduler re-arms at resume_at.
            log.warning(
                "agent deferred for %s: transient provider failure (%s); "
                "retrying in %.1fs",
                grant.claim.chat_key,
                exc,
                agent_cfg.retry_delay_s,
            )
            resume_at = self._clock.now() + agent_cfg.retry_delay_s
            await self._settle_defer(
                grant,
                resume_at=resume_at,
                defer_kind="retry",
                trace=trace,
                now=self._clock.now(),
                evaluated_ts=evaluated_ts,
                snapshot_json=snapshot_json,
            )
            return Decision(
                action="delay",
                delay_seconds=agent_cfg.retry_delay_s,
                reason=Reason.DELAY,
            )
        except ClaimError:
            # Renewal/fencing failed before the saga could safely settle.
            # Leave the prepared owner for the scheduler's known-expiry
            # recovery path; never attempt a terminal finish as a stale owner.
            raise
        except PermanentError as exc:
            # Any project-wide permanent failure (bad config/prompt/tool or
            # provider 4xx) will not succeed on retry: terminally consume the
            # cursor with an EMPTY outbox rather than creating a busy-retry
            # loop. The budget reservation is retained by BudgetedClient.
            # This MUST be logged: an unreported 401/404 here is a bot that
            # ingests everything, triggers, calls the provider and then stays
            # silent forever with no other trace than an empty outbox.
            log.warning(
                "agent gave up for %s: permanent provider/config failure (%s); "
                "settling with an empty outbox",
                grant.claim.chat_key,
                exc,
            )
            await self._finish_dispatch(
                grant,
                "llm_permanent_error",
                trace,
                evaluated_ts=evaluated_ts,
                snapshot_json=snapshot_json,
                outbox=[],
                tokens_in=0,
                tokens_out=0,
            )
            assert trace.decision is not None
            return trace.decision
        if outcome.intent == "wait":
            wait_s = outcome.wait_seconds
            if wait_s is None or wait_s <= 0:
                wait_s = None
            # Durable wait: the THIRD consecutive wait terminally consumes
            # with planner_wait_rest (clearing the barrier/streak); otherwise
            # defer with defer_kind="wait" (the foundation increments the
            # wait streak and persists the barrier; no early priority
            # execution; restart honors the remaining delay).
            state = await self._repo.get_chat_state(chat_key)
            wait_streak = state.wait_streak if state is not None else 0
            if wait_streak >= 2:
                await self._finish_dispatch(
                    grant,
                    "planner_wait_rest",
                    trace,
                    evaluated_ts=evaluated_ts,
                    snapshot_json=snapshot_json,
                    outbox=[],
                    tokens_in=outcome.tokens_in,
                    tokens_out=outcome.tokens_out,
                )
                assert trace.decision is not None
                return trace.decision
            resume_at = self._clock.now() + (
                wait_s if wait_s is not None else agent_cfg.retry_delay_s
            )
            await self._settle_defer(
                grant,
                resume_at=resume_at,
                defer_kind="wait",
                trace=trace,
                now=self._clock.now(),
                evaluated_ts=evaluated_ts,
                snapshot_json=snapshot_json,
            )
            if wait_s is None:
                return Decision(action="delay", reason=Reason.DELAY)
            return Decision(
                action="delay", delay_seconds=wait_s, reason=Reason.DELAY
            )
        items: list[OutboxItem] = []
        pre_send_failed = False
        media_intent = outcome.media_intent
        durable_controls = self._chat_control_rows(grant, outcome.chat_controls)
        if media_intent is not None and not self._dry_run:
            # Phase 6 P6.5b media send: convert the valid intent into an
            # Outgoing media segment carrying the OPAQUE cache key ONLY at
            # normal terminal settlement; the existing output pipeline /
            # outbox / self-echo path handles delivery. The pre_send hooks
            # run BEFORE the pipeline/outbox conversion (fail-closed: a
            # timeout/error suppresses the output).
            converted = await self._agent_media_outbox_items(grant, media_intent)
            if converted is None:
                pre_send_failed = True
            else:
                items = converted
        elif outcome.intent == "reply" and outcome.reply_text and not self._dry_run:
            converted = await self._agent_outbox_items(
                grant, outcome.reply_text, outcome.reply_to
            )
            if converted is None:
                pre_send_failed = True
            else:
                items = converted
        if media_intent is not None:
            if not self._dry_run:
                end_reason = "agent_media"
            else:
                end_reason = "dry_run_agent_media"
        elif outcome.intent == "reply":
            if outcome.reply_text and not self._dry_run:
                end_reason = "agent_reply"
            elif self._dry_run and outcome.reply_text:
                end_reason = "dry_run_agent_reply"
            else:
                end_reason = "reply_no_output"
        elif outcome.intent == "no_action":
            # Keep the planner's specific exit (no_tool_call / empty_response
            # / tool_round_cap) — flattening them all to "no_action" makes a
            # model that cannot emit tool calls indistinguishable from one
            # that deliberately stayed quiet, which is the difference between
            # a bug and normal behaviour.
            end_reason = outcome.end_reason or "no_action"
        else:
            end_reason = "budget_blocked"
        if pre_send_failed:
            # A pre_send hook timed out or errored: fail closed to NO output
            # (the terminal finish persists an EMPTY outbox — nothing was
            # ever converted or persisted).
            end_reason = "pre_send_failed"
        log.info(
            "agent %s for %s: end_reason=%s outbox=%d tokens=%d/%d",
            outcome.intent,
            grant.claim.chat_key,
            end_reason,
            len(items),
            outcome.tokens_in,
            outcome.tokens_out,
        )
        await self._finish_dispatch(
            grant,
            end_reason,
            trace,
            evaluated_ts=evaluated_ts,
            snapshot_json=snapshot_json,
            outbox=items,
            tokens_in=outcome.tokens_in,
            tokens_out=outcome.tokens_out,
            chat_controls=durable_controls,
        )
        # The rows were committed atomically with the terminal settlement.
        # Wake each target only after that durable write; this is a scheduler
        # wake, never a fabricated user inbound event.
        if durable_controls and not self._dry_run and self._on_chat_control is not None:
            for target in dict.fromkeys(control.chat_key for control in durable_controls):
                try:
                    await self._on_chat_control(target)
                except Exception:
                    log.warning(
                        "chat control wake failed for %s (contained)",
                        target,
                        exc_info=True,
                    )
        # Phase 6 P6.5b media use/cooldown: persisted ONLY after a terminal
        # settlement that created durable outbox rows, idempotently by
        # dispatch/intent identity (a retried settlement of the same
        # dispatch never double-counts). Contained — never blocks the
        # post-terminal callbacks below.
        if media_intent is not None and items and not self._dry_run:
            await self._persist_media_use(grant, media_intent)
        # Post-terminal exposure (LIVE-only): fired ONLY when a selected
        # record was actually frozen into the prompt AND durable outbox
        # output was created. The adaptive context is None in dry-run, so
        # this never fires there; ``items`` is empty for every non-reply
        # terminal outcome.
        if (
            self._on_exposure is not None
            and adaptive is not None
            and adaptive.frozen_records
            and items
        ):
            try:
                await self._on_exposure(
                    chat_key,
                    adaptive.frozen_records,
                    grant.dispatch_id,
                    grant.through_msg_id,
                )
            except Exception:
                log.warning(
                    "on_exposure failed for %s (contained)", chat_key,
                    exc_info=True,
                )
        assert trace.decision is not None
        return trace.decision

    async def _run_with_renewal(
        self,
        grant: DispatchGrant,
        coro: Coroutine[Any, Any, AgentOutcome],
        *,
        deadline: float,
    ) -> AgentOutcome:
        """Run the agent saga with the dispatch lease renewed to cover the
        whole run (the foundation fences settlement against an expired
        owner, so the lease must not lapse mid-execution). The saga is
        bounded by the aggregate deadline (``max_execution_s``), so a single
        renewal to the deadline plus a lease buffer extends the lease
        through the entire long execution."""
        agent_cfg = self._cfg.for_chat(grant.claim.chat_key).agent
        renewed = await self._repo.renew_dispatch(
            grant.claim.chat_key,
            grant.dispatch_id,
            grant.claim.cycle_id,
            expires_at=(
                self._clock.now() + agent_cfg.max_execution_s + agent_cfg.dispatch_lease_s
            ),
            now=self._clock.now(),
        )
        if not renewed:
            coro.close()
            raise ClaimError(
                f"cannot renew dispatch {grant.dispatch_id!r} for agent saga"
            )
        remaining = deadline - self._clock.now()
        if remaining <= 0:
            coro.close()
            raise LLMTransientError("agent execution deadline already passed")
        try:
            return await asyncio.wait_for(coro, timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise LLMTransientError("agent execution deadline exceeded") from None

    async def _settle_defer(
        self,
        grant: DispatchGrant,
        *,
        resume_at: float,
        defer_kind: str,
        trace: DecisionTrace | None,
        now: float,
        evaluated_ts: float | None,
        snapshot_json: str | None,
    ) -> None:
        """Settle one prepared dispatch as a DEFER through
        ``Repository.settle_dispatch``: the claim is released, the attached
        commits are detached (cursor/outbox unchanged), and the durable agent
        barrier (``agent_resume_at``) is recorded — a wait defer increments
        the wait streak, a retry defer does not. Then arrange the
        at-least-once dispatch marker export (a deferred dispatch is a
        released, non-terminal dispatch)."""
        settle = DispatchSettle(
            chat_key=grant.claim.chat_key,
            dispatch_id=grant.dispatch_id,
            cycle_id=grant.claim.cycle_id,
            outcome="defer",
            resume_at=resume_at,
            defer_kind=defer_kind,
            trace_json=(
                json.dumps(dataclasses.asdict(trace), default=str)
                if trace is not None
                else None
            ),
            evaluated_ts=evaluated_ts,
            snapshot_json=snapshot_json,
        )
        await self._repo.settle_dispatch(settle, [], now=now)
        await self._export_dispatch_marker(
            grant,
            state="released",
            settled_ts=now,
            evaluated_ts=evaluated_ts,
            snapshot_json=snapshot_json,
            trace=trace,
        )

    async def _settle(
        self,
        grant: DispatchGrant,
        *,
        outcome: str,
        trace: DecisionTrace | None = None,
        now: float,
        evaluated_ts: float | None = None,
        snapshot_json: str | None = None,
    ) -> None:
        """Settle one prepared dispatch through ``Repository.settle_dispatch``
        (release/delay: no cursor/outbox movement; the trace rides on the
        released row). Then arrange the at-least-once dispatch marker
        export when an exporter is injected."""
        settle = DispatchSettle(
            chat_key=grant.claim.chat_key,
            dispatch_id=grant.dispatch_id,
            cycle_id=grant.claim.cycle_id,
            outcome=outcome,
            trace_json=(
                json.dumps(dataclasses.asdict(trace), default=str)
                if trace is not None
                else None
            ),
            evaluated_ts=evaluated_ts,
            snapshot_json=snapshot_json,
        )
        await self._repo.settle_dispatch(settle, [], now=now)
        await self._export_dispatch_marker(
            grant,
            state="released",
            settled_ts=now,
            evaluated_ts=evaluated_ts,
            snapshot_json=snapshot_json,
            trace=trace,
        )

    async def _finish_dispatch(
        self,
        grant: DispatchGrant,
        end_reason: str,
        trace: DecisionTrace,
        *,
        evaluated_ts: float,
        snapshot_json: str,
        outbox: list[OutboxItem] | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        chat_controls: tuple[ChatControl, ...] = (),
    ) -> None:
        """Terminal completion: cursor consumed, hold cleared, idle streak
        reset, the ordered outbox batch (EMPTY for every no-agent and
        dry-run disposition), trace persisted — one transaction. The finish
        fences against a FRESH clock timestamp (the lease may have expired
        while the cycle ran; an expired owner cannot finish). The
        ``on_cycle_end`` hook fires only after the durable completion."""
        settled_ts = self._clock.now()
        settle = DispatchSettle(
            chat_key=grant.claim.chat_key,
            dispatch_id=grant.dispatch_id,
            cycle_id=grant.claim.cycle_id,
            outcome="finish",
            end_reason=end_reason,
            hold_until=None,  # terminal reset: clears any durable hold
            idle_streak_after=0,  # terminal reset
            trace_json=json.dumps(dataclasses.asdict(trace), default=str),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            evaluated_ts=evaluated_ts,
            snapshot_json=snapshot_json,
            chat_controls=chat_controls,
        )
        await self._repo.settle_dispatch(settle, outbox or [], now=settled_ts)
        # Contained memory maintenance: after the durable TERMINAL settlement
        # ONLY, fire the on_memory callback with the chat and the frozen
        # through boundary so the App can summarize one oldest unprocessed
        # local batch. Never on release/delay/defer/replay/gate. A failure
        # is contained and logged — it must not block the outbox wake,
        # marker export, or on_cycle_end hooks below.
        if self._on_memory is not None:
            try:
                await self._on_memory(grant.claim.chat_key, grant.through_msg_id)
            except Exception:
                log.warning(
                    "on_memory failed for %s (contained)",
                    grant.claim.chat_key,
                    exc_info=True,
                )
        # Live outbox wake: immediately after the terminal settlement that
        # created outbox rows, BEFORE marker export and hooks, so the outbox
        # worker drains the new rows promptly (the App wakes its worker on
        # this). None in dry-run / no-agent modes.
        if self._on_outbox is not None and outbox:
            await self._on_outbox(outbox)
        await self._export_dispatch_marker(
            grant,
            state="completed",
            settled_ts=settled_ts,
            evaluated_ts=evaluated_ts,
            snapshot_json=snapshot_json,
            trace=trace,
        )
        # Post-terminal settlement callback: fired AFTER settle_dispatch, the
        # outbox wake, and the marker export — the App enqueues the chat for
        # its learners here (non-blocking; settlement never awaits LLM/learn/
        # media). Contained: a failure must not block the hooks below.
        if self._on_settled is not None:
            try:
                await self._on_settled(grant.claim.chat_key, grant.through_msg_id)
            except Exception:
                log.warning(
                    "on_settled failed for %s (contained)",
                    grant.claim.chat_key,
                    exc_info=True,
                )
        # Phase 6 P6.6: hooks run in LIVE mode only — never in dry-run/
        # replay/doctor. The bus is bounded and contained.
        if self._hooks is not None and not self._dry_run:
            await self._hooks.emit_cycle_end(grant.claim.chat_key, trace, end_reason)

    def _chat_control_rows(
        self, grant: DispatchGrant, intents: Sequence[ChatControlIntent]
    ) -> tuple[ChatControl, ...]:
        """Convert staged intents to durable rows before terminal settle."""
        if self._dry_run or not intents:
            return ()
        created = self._clock.now()
        return tuple(
            ChatControl(
                chat_key=intent.target_chat_key,
                kind=intent.kind,
                ttl_until=created + intent.ttl_s,
                created_ts=created,
                dispatch_id=grant.dispatch_id,
                intent_seq=i,
                source_chat_key=grant.claim.chat_key,
                text=intent.text,
            )
            for i, intent in enumerate(intents)
        )

    async def _export_dispatch_marker(
        self,
        grant: DispatchGrant,
        *,
        state: str,
        settled_ts: float,
        evaluated_ts: float | None = None,
        snapshot_json: str | None = None,
        trace: DecisionTrace | None = None,
    ) -> None:
        """Arrange the at-least-once dispatch marker export when an
        exporter is injected: the v4 marker carries the frozen commit
        boundary, scheduled time, settled state, evaluation timestamp,
        message boundaries, exact attached membership, and trace — so
        replay reconstructs the exact live dispatch from the marker alone.
        The exporter is wired to ``record.export_marker`` by the caller; a
        crash between the append and the export mark is repaired by the
        startup export (readers deduplicate by (record_type, sequence)).
        The grant does not always carry the dispatch cause, so it is
        derived from the scheduled time when absent: a timer dispatch
        (``scheduled_for`` set) is marked ``timer``, every other dispatch
        defaults to ``inbound``."""
        if self._marker_exporter is None:
            return
        marker = CorpusMarker(
            record_type="dispatch",
            sequence=grant.dispatch_id,
            chat_key=grant.claim.chat_key,
            cause=(
                grant.cause
                if grant.cause is not None
                else (
                    DispatchCause.TIMER
                    if grant.scheduled_for is not None
                    else DispatchCause.INBOUND
                )
            ),
            commit_boundary=grant.commit_boundary,
            scheduled_for=grant.scheduled_for,
            state=state,
            settled_ts=settled_ts,
            start_msg_id=grant.start_msg_id,
            through_msg_id=grant.through_msg_id,
            attached=grant.attached,
            trace_json=(
                json.dumps(dataclasses.asdict(trace), default=str)
                if trace is not None
                else None
            ),
            evaluated_ts=evaluated_ts,
            snapshot_json=snapshot_json,
        )
        await self._marker_exporter(marker)

    def _resolve_identity(self) -> str:
        """The bot identity text for the planner/replyer runtime prompts.

        The configured ``bot.identity_file`` content is loaded through the
        prompt infrastructure (so the user's ``prompts/identity.txt``
        shadows the shipped default); a missing/unreadable/empty identity
        file falls back safely to the bot name. The identity is prompt
        content only — it never carries planner analysis into the replyer's
        user turn (the replyer's transcript carries only the staged reply
        reference).
        """
        try:
            text = self._prompts.load_identity(self._cfg.bot.identity_file)
        except PromptError:
            text = ""
        if text.strip():
            return text.strip()
        return f"你是{self._cfg.bot.name}"

    def _resolve_behavior_style(self) -> str:
        """The planner's action rules (MaiBot's ``behavior_style``).

        Loaded from ``bot.behavior_file`` through the same prompt
        infrastructure as the identity, so a user ``prompts/behavior.txt``
        shadows the shipped default. A missing or unreadable file degrades
        to ``""`` — the planner keeps its own decision rules and simply
        loses the persona's slant on them.
        """
        try:
            return self._prompts.load_identity(self._cfg.bot.behavior_file).strip()
        except PromptError:
            return ""

    async def _resolve_quote_targets(
        self, chat_key: ChatKey, pending: tuple[Message, ...]
    ) -> frozenset[MessageId]:
        """Resolve every DISTINCT pending reply target against the durable
        repository — not only the limited rendered window — so an older
        quote to a self message outside the window still triggers."""
        targets = {m.reply_to for m in pending if m.reply_to is not None}
        self_ids: set[MessageId] = set()
        for target in targets:
            msg = await self._repo.get_message(chat_key, target)
            if msg is not None and msg.is_self:
                self_ids.add(target)
        return frozenset(self_ids)

    # ── Phase 6 P6.6b chat controls (LIVE only) ─────────────────────────────

    async def _merge_active_controls(
        self, chat_key: ChatKey, state: ChatState
    ) -> ChatState:
        """Merge the chat's ACTIVE internal focus events into the durable
        state: the latest ``ttl_until`` among the active controls extends
        ``focus_until``, so the target chat's gate evaluates as focused and
        the notify traverses the target gate. Contained: a failure leaves
        the state unchanged (the gate evaluates normally)."""
        list_active = getattr(self._repo, "list_active_controls", None)
        if list_active is None:
            return state
        try:
            controls = await list_active(chat_key, now=self._clock.now())
        except Exception:
            return state
        focus_until = state.focus_until
        for control in controls:
            if control.ttl_until > (focus_until or 0):
                focus_until = control.ttl_until
        if focus_until != state.focus_until:
            return dataclasses.replace(state, focus_until=focus_until)
        return state

    async def _notify_text(self, chat_key: ChatKey) -> str:
        """The bounded internal-notification section for the agent's chat
        log (LIVE only): the payloads of the chat's ACTIVE notify controls.
        Empty when there are none or the surface is unavailable."""
        list_active = getattr(self._repo, "list_active_controls", None)
        if list_active is None:
            return ""
        try:
            controls = await list_active(chat_key, now=self._clock.now())
        except Exception:
            return ""
        texts = [c.text for c in controls if c.kind == ChatControlKind.NOTIFY and c.text]
        if not texts:
            return ""
        lines = ["【内部通知】(来自其他会话的内部通知，请据此回应)"]
        for text in texts:
            lines.append(f"- {text}")
        return "\n".join(lines)

    async def _apply_chat_controls(
        self, grant: DispatchGrant, intents: Sequence[ChatControlIntent]
    ) -> None:
        """Apply the staged chat controls idempotently, AFTER the terminal
        settlement (settle/outbox/marker), LIVE only. The idempotency
        identity is the dispatch id + intent sequence (a retried settlement
        of the same dispatch never double-applies). Contained: a failure is
        logged and never blocks the post-terminal callbacks."""
        apply = getattr(self._repo, "apply_chat_control", None)
        if apply is None:
            return
        now = self._clock.now()
        for i, intent in enumerate(intents):
            try:
                await apply(
                    ChatControl(
                        chat_key=intent.target_chat_key,
                        kind=intent.kind,
                        ttl_until=now + intent.ttl_s,
                        created_ts=now,
                        dispatch_id=grant.dispatch_id,
                        intent_seq=i,
                        source_chat_key=grant.claim.chat_key,
                        text=intent.text,
                    )
                )
            except Exception:
                log.warning(
                    "chat control apply failed for %s (contained)",
                    intent.target_chat_key,
                    exc_info=True,
                )

    def _output_pipeline(self, chat_key: ChatKey) -> OutputPipeline:
        """The output pipeline for ``chat_key``, built from the chat's
        EFFECTIVE per-chat ``OutputConfig`` (sanitize → split → typo)."""
        output_cfg = self._cfg.for_chat(chat_key).output
        return self._output_factory(output_cfg)

    async def _agent_outbox_items(
        self, grant: DispatchGrant, text: str, reply_to: str | None
    ) -> list[OutboxItem] | None:
        """One ordered OutboxItem batch for the agent's reply draft.

        The reply is built as an ``Outgoing`` and run through the chat's
        output pipeline (sanitize → split → typo) ONCE at terminal
        settlement; the durable rows it produces are never re-processed on
        outbox retry/restart. Idempotency keys are derived from the DURABLE
        grant identity (``dispatch:{dispatch_id}:{part_index}``) —
        deterministic and unique per dispatch, so a retried settlement of
        the same dispatch hydrates the same rows (``ON CONFLICT(idem_key)
        DO NOTHING``). Split parts share a stable content-derived
        ``group_id`` and are ordered by ``seq``; the split stage's relative
        pacing is mapped to a durable absolute ``send_after_ts`` per part.

        Phase 6 P6.6 pre_send hooks run BEFORE the output pipeline and the
        outbox conversion (LIVE only): a timeout/error fails closed and
        returns None — the caller must treat it as NO output (nothing is
        ever converted or persisted).
        """
        out = Outgoing(
            chat_key=grant.claim.chat_key,
            text=text,
            reply_to=MessageId(reply_to) if reply_to else None,
        )
        if not self._dry_run and self._hooks is not None:
            out = await self._hooks.emit_pre_send(out)
            if out is None:
                return None  # fail-closed: no output
        self._output_pipeline(grant.claim.chat_key).run(out)
        return self._outgoing_to_items(grant, out)

    async def _agent_media_outbox_items(
        self, grant: DispatchGrant, intent: MediaReplyIntent
    ) -> list[OutboxItem] | None:
        """One ordered OutboxItem batch for a staged media send.

        The intent is converted into an ``Outgoing`` carrying a media
        segment with ONLY the OPAQUE content-addressed cache key — never a
        URL, file path, platform reference, or base64 payload. The segment
        is run through the chat's output pipeline (which keeps segmented
        outgoing atomic) and converted with the same dispatch-based
        idempotency keys as a text reply, so a retried settlement hydrates
        the same rows. The send-time resolver maps the cache key to the
        normalized bytes in memory.

        Phase 6 P6.6 pre_send hooks run BEFORE the pipeline/outbox
        conversion (LIVE only): a timeout/error fails closed and returns
        None — the caller must treat it as NO output.
        """
        out = Outgoing(
            chat_key=grant.claim.chat_key,
            text="",
            segments=[media_segment_for_intent(intent)],
        )
        if not self._dry_run and self._hooks is not None:
            out = await self._hooks.emit_pre_send(out)
            if out is None:
                return None  # fail-closed: no output
        self._output_pipeline(grant.claim.chat_key).run(out)
        return self._outgoing_to_items(grant, out)

    async def _persist_media_use(
        self, grant: DispatchGrant, intent: MediaReplyIntent
    ) -> None:
        """Persist the asset use/cooldown AFTER a terminal settlement that
        created durable outbox rows, idempotently by dispatch/intent
        identity (a retried settlement of the same dispatch never
        double-counts — the kv marker is set only on the first use).
        Contained: a failure never blocks the post-terminal callbacks."""
        chat_key = grant.claim.chat_key
        marker = f"media_use:{grant.dispatch_id}:{intent.asset_id}"
        use_asset = getattr(self._repo, "use_media_asset", None)
        if use_asset is None:
            return
        try:
            if await self._repo.get_kv(marker) is not None:
                return
            used = await use_asset(
                chat_key, intent.asset_id, now=self._clock.now()
            )
            if used:
                await self._repo.set_kv(marker, "1")
        except Exception:
            log.warning(
                "media use persist failed for %s (contained)", chat_key,
                exc_info=True,
            )

    async def _catalog_prompt(self, chat_key: ChatKey) -> str:
        """The approved catalog listing for the planner prompt (LIVE only).

        Renders ONLY opaque asset ids and scrubbed descriptions — never
        URLs, file paths, platform refs, or base64. The listing is the
        COOLDOWN-AWARE approved selection (``select_media_assets``), so the
        planner only sees assets eligible to send right now. Empty when the
        media catalog is disabled or the repository lacks the media
        surface. Bounded and contained: a failure degrades to no listing."""
        media_cfg = self._cfg.for_chat(chat_key).media
        if not media_cfg.enabled:
            return ""
        select = getattr(self._repo, "select_media_assets", None)
        if select is None:
            return ""
        try:
            now = self._clock.now()
            stickers = await select(
                chat_key,
                MediaKind.STICKER,
                limit=50,
                cooldown_s=media_cfg.cooldown_s,
                now=now,
            )
            images = await select(
                chat_key,
                MediaKind.IMAGE,
                limit=50,
                cooldown_s=media_cfg.cooldown_s,
                now=now,
            )
        except Exception:
            return ""
        return catalog_prompt([*stickers, *images])

    def _outgoing_to_items(
        self, grant: DispatchGrant, out: Outgoing
    ) -> list[OutboxItem]:
        """Convert a pipeline-processed ``Outgoing`` into its ordered
        ``OutboxItem`` batch with dispatch-based idempotency keys.

        ``parts`` (when the split stage produced them) become a batch
        sharing ``group_id`` and ordered by ``seq``; the split stage's
        ``platform_ref["part_pacing"]`` relative delays are mapped to a
        durable absolute ``send_after_ts`` per part (part 0 sends
        immediately, later parts at ``now + delay``). Protected spans and
        the pacing metadata ride along in ``payload`` untouched.
        """
        # A split stage operates on text. Repeating a full segment payload
        # (image/sticker/reply) for every text part would deliver the original
        # message multiple times on OneBot, so segmented outgoing messages are
        # deliberately kept atomic until part-specific segment construction is
        # implemented.
        if out.segments:
            parts = [out.text]
        else:
            parts = list(out.parts) if out.parts else ([out.text] if out.text else [])
        if not parts:
            return []
        group_id = out.group_id or stable_group_id(parts)
        pacing = out.platform_ref.get("part_pacing") or [0.0] * len(parts)
        base = out.send_after_ts
        now = self._clock.now()
        items: list[OutboxItem] = []
        for i, part in enumerate(parts):
            delay = pacing[i] if i < len(pacing) else 0.0
            if base is not None:
                send_after = base + delay
            elif delay > 0:
                send_after = now + delay
            else:
                send_after = None
            items.append(
                OutboxItem(
                    chat_key=out.chat_key,
                    text=part,
                    idem_key=f"dispatch:{grant.dispatch_id}:{i}",
                    segments=tuple(out.segments),
                    payload=dict(out.platform_ref),
                    reply_to=out.reply_to,
                    group_id=group_id,
                    seq=i if len(parts) > 1 else None,
                    send_after_ts=send_after,
                )
            )
        return items

    async def _finish(
        self,
        chat_key: ChatKey,
        cycle_id: CycleId,
        end_reason: str,
        trace: DecisionTrace,
    ) -> None:
        """Terminal completion: cursor consumed, hold cleared, idle streak
        reset, EMPTY outbox, trace persisted — one transaction. The finish
        fences against a FRESH clock timestamp (the lease may have expired
        while the cycle ran; an expired owner cannot finish). The
        ``on_cycle_end`` hook fires only after the durable completion."""
        finish = CycleFinish(
            chat_key=chat_key,
            cycle_id=cycle_id,
            end_reason=end_reason,
            hold_until=None,  # terminal reset: clears any durable hold
            idle_streak_after=0,  # terminal reset
            trace_json=json.dumps(dataclasses.asdict(trace), default=str),
        )
        await self._repo.finish_cycle(finish, [], now=self._clock.now())
        # Phase 6 P6.6: hooks run in LIVE mode only — never in dry-run/
        # replay/doctor. The bus is bounded and contained.
        if self._hooks is not None and not self._dry_run:
            await self._hooks.emit_cycle_end(chat_key, trace, end_reason)


# ── Replay: the same assembler + Gate.evaluate, no storage at all ───────────

@dataclass(frozen=True)
class ReplayResult:
    """The deterministic outcome of re-scoring one corpus.

    ``traces`` are the per-decision ``DecisionTrace`` objects in corpus
    order; ``would_have_spoken`` counts trigger decisions; ``rate`` is
    ``would_have_spoken / decisions`` (0.0 on an empty corpus).
    """

    chat_key: ChatKey
    traces: tuple[DecisionTrace, ...]
    would_have_spoken: int
    decisions: int
    rate: float


@dataclass(frozen=True)
class SweepRow:
    """One sweep combination's would-have-spoken report."""

    threshold: int
    trigger_score: int
    would_have_spoken: int
    decisions: int
    rate: float


def replay_corpus(
    events: Sequence[Any],
    *,
    chat_key: ChatKey,
    identity: ChatIdentity,
    cfg: Config,
    gate: Gate | None = None,
    overlay: RuntimeOverlay | None = None,
    window_s: float = SNAPSHOT_WINDOW_S,
    limit: int = SNAPSHOT_LIMIT,
) -> ReplayResult:
    """Re-score a recorded corpus through the SAME assembler + gate path,
    modeling the live App's commit-then-wake dispatcher.

    Deterministic and storage-free: no database, outbox, adapter, or clock
    operations. The corpus is processed in FILE order — the durable commit
    order — and synthetic row ids follow the file, so a corpus whose
    timestamps recede or jump never reorders committed rows. The virtual
    clock advances MONOTONICALLY: each event commits at
    ``max(now, recv_ts)`` (a receding timestamp commits at the current
    time; the clock never moves backwards). The simulation mirrors the
    live App + scheduler:

    - Every event is committed immediately at the current virtual time.
      Commits at the SAME commit time coalesce into ONE wake (the App's
      next-turn flush: one evaluation per commit-time group, priority if
      any member qualifies). Self messages never wake and are EXCLUDED
      from every pending batch (they stay in the recent window and the
      full-window counts, exactly like the live repository's claim).
    - Dispatcher tie rule (commit-before-timer): a timed wake due at
      exactly a commit time fires AFTER the commit — the committed
      event's wake metadata coalesces into that evaluation (the live
      claim includes every row committed before the wake), and the
      event's own wake is subsumed. A timed wake due strictly BEFORE the
      commit time fires during the clock advance, BEFORE the commit, and
      does not see the event.
    - A timed delay (``delay_seconds``) re-arms the next evaluation at
      ``now + delay``; ordinary arrivals during the delay stay pending
      and are seen at the scheduled evaluation — they never override it.
      A PRIORITY group DOES override the delay (the App's priority wake
      path): a structurally recognized direct @/quote, or the chat's
      atomic pending non-self count at/above the gate threshold (high
      pending may re-evaluate during a scheduled delay/hold). The gate
      applies the exact precedence. After the last commit the scheduler
      keeps firing due timed wakes (the idle bonus / frequency virtual
      messages can still trigger), so the simulation fires them too.
    - Terminal outcomes (trigger / skip) consume the cursor; delays
      release the claim with the cursor/session unchanged. The durable
      EWMA average evolves from non-self messages in ROW-ID (commit)
      order through the SAME ``pacing.ewma_interval`` reducer the live
      ingest path uses (a non-positive gap — clock skew, receding
      timestamps — carries no pacing information), so idle bonus /
      frequency virtual messages and traces match.
    - Quote targets resolve against the WHOLE corpus seen so far (the
      corpus is the durable store), so an older quote to a self message
      outside the window still triggers.

    ``overlay`` (a RuntimeOverlay) varies gate constants for sweep runs.
    """
    if overlay is not None:
        cfg = overlay.apply(cfg)
    gate = gate if gate is not None else Gate()
    _set_replay_manifest(gate, cfg)
    chat_cfg = cfg.for_chat(chat_key)
    self_name = cfg.bot.name
    # The corpus in FILE order: the durable commit order. Row ids follow
    # this list; timestamps never reorder it.
    commits: list[tuple[float, Message]] = []
    for event in events:
        if event.type != "message" or not isinstance(event.payload, Message):
            continue
        msg = event.payload
        if msg.chat_key != chat_key:
            continue
        ts = msg.recv_ts if msg.recv_ts is not None else event.ts
        if ts is None:
            continue  # no timestamp: cannot place the message in the window
        commits.append((ts, dataclasses.replace(msg, recv_ts=ts)))
    messages: list[Message] = []
    traces: list[DecisionTrace] = []
    would_have_spoken = 0
    cursor = 0
    state = ChatState(chat_key=chat_key)
    previous_end_reason: str | None = None
    next_wake_ts: float | None = None
    eval_seq = 0
    now: float | None = None  # the virtual clock; None until the first commit

    def evaluate(now: float) -> None:
        """One scheduler wake at ``now``: claim every committed row up to
        the through boundary, assemble the snapshot, evaluate the SAME
        gate, and apply the frozen disposition."""
        nonlocal cursor, state, previous_end_reason, next_wake_ts
        nonlocal would_have_spoken, eval_seq
        # The claim's through boundary is the max committed row id: the
        # claim happens after the commit, so every committed row is within
        # it. The monotonic-clock invariant guarantees every committed row
        # has recv_ts <= now (a row with a later recv_ts commits at a
        # later clock time), so no row is excluded by timestamp.
        through = 0
        if messages:
            assert messages[-1].row_id is not None
            through = int(messages[-1].row_id)
        pending = tuple(
            m
            for m in messages
            if m.row_id is not None
            and cursor < int(m.row_id) <= through
            and not m.is_self  # self echoes are never pending (live parity)
        )
        recent = _replay_recent(
            messages, chat_key=chat_key, through=through, since=now - window_s, limit=limit
        )
        quote_self_ids = frozenset(
            m.id
            for m in messages
            if m.is_self
            and m.id is not None
            and m.recv_ts is not None
            and m.recv_ts <= now
        )
        # The durable EWMA reducer, evolved over the non-self messages
        # ARRIVED by ``now`` in ROW-ID (commit) order with the
        # repository's exact prior-sample semantics: the prior sample is
        # the MAX recv_ts among the prior non-self rows (the repository's
        # ``MAX(recv_ts) ... id < ?``), so a receding timestamp is a
        # non-positive gap that carries no pacing information and never
        # drags the average. The live ingest path folds each message at
        # commit time, so an evaluation never sees a later arrival's
        # sample — a timed wake firing before a future arrival must not
        # either.
        avg = _evolve_avg_interval(messages, now)
        if avg is not None:
            state = dataclasses.replace(state, avg_interval=avg)
        eval_seq += 1
        grant = ClaimGrant(
            claim=CycleClaim(
                chat_key,
                CycleId(f"replay:{eval_seq}"),
                started_ts=now,
                expires_at=now + 1.0,
            ),
            start_msg_id=MessageRowId(cursor),
            through_msg_id=MessageRowId(through),
            pending=pending,
        )
        snapshot = assemble_snapshot(
            grant=grant,
            identity=identity,
            state=state,
            recent=recent,
            cfg=chat_cfg,
            now=now,
            self_name=self_name,
            muted=is_muted(cfg.access, chat_key),
            previous_end_reason=previous_end_reason,
            window_s=window_s,
            quote_self_ids=quote_self_ids,
        )
        trace = gate.evaluate(snapshot)
        trace = _trace_with_fingerprint(trace, gate)
        traces.append(trace)
        decision = trace.decision
        assert decision is not None
        if decision.action == "trigger":
            would_have_spoken += 1
            cursor = through
            state = dataclasses.replace(state, hold_until=None, idle_streak=0)
            previous_end_reason = "dry_run_trigger"
            next_wake_ts = None
        elif decision.action == "skip":
            cursor = through
            state = dataclasses.replace(state, hold_until=None, idle_streak=0)
            previous_end_reason = "skip"
            next_wake_ts = None
        elif decision.delay_seconds is not None and decision.delay_seconds > 0:
            # Timed delay: the scheduler re-arms at now + delay; the claim
            # is released with the cursor/session unchanged.
            next_wake_ts = now + decision.delay_seconds
        else:
            next_wake_ts = None  # event-only delay: no timed wake

    # Process the corpus in FILE order (the durable commit order): row ids
    # follow the file, and the virtual clock advances monotonically, so a
    # receding timestamp commits at the current time and never reorders a
    # committed row. Consecutive commits at the same commit time coalesce
    # into ONE wake (the App's next-turn flush).
    i = 0
    while i < len(commits):
        recv_ts, msg = commits[i]
        if now is None:
            now = recv_ts  # the clock starts at the first commit
        commit_ts = max(now, recv_ts)
        # Fire timed wakes due strictly BEFORE this commit (the clock
        # advance to commit_ts passes them): each sees the timeline as of
        # its due time — this commit is NOT yet visible.
        while next_wake_ts is not None and next_wake_ts < commit_ts:
            evaluate(next_wake_ts)
        # Commit every consecutive message whose commit time is exactly
        # commit_ts (the clock never moves backwards, so a receding recv_ts
        # commits at the current time): one coalesced wake per commit time.
        group: list[Message] = []
        while i < len(commits):
            rts, m = commits[i]
            if max(commit_ts, rts) != commit_ts:
                break
            group.append(m)
            i += 1
        for m in group:
            row_id = len(messages) + 1
            messages.append(dataclasses.replace(m, row_id=MessageRowId(row_id)))
        now = commit_ts
        if not any(not m.is_self for m in group):
            continue  # a self-only group never wakes
        if next_wake_ts is not None and next_wake_ts == commit_ts:
            # Tie rule (commit-before-timer): a timed wake due at exactly
            # this commit time fires AFTER the commit and sees the
            # committed group (the live claim includes every row committed
            # before the wake); the group's own wake is subsumed.
            evaluate(commit_ts)
            continue
        if next_wake_ts is not None:
            # A delay is still scheduled beyond this commit: ordinary
            # input never overrides it; a PRIORITY group re-evaluates now
            # (the App's priority wake path) — a structurally recognized
            # direct @/quote, or the chat's atomic pending non-self count
            # at/above the gate threshold (high pending may re-evaluate
            # during a scheduled delay/hold).
            if _is_priority_batch(
                group, identity, messages, cursor, chat_cfg.gate.threshold
            ):
                evaluate(commit_ts)
            continue
        evaluate(commit_ts)
    # The scheduler keeps firing timed wakes after the last commit (the
    # idle bonus / frequency virtual messages can still trigger). The
    # chain terminates when a decision is event-only or terminal; the cap
    # is defensive only (a backoff chain cannot arise in dry-run replay:
    # terminal end reasons are never idle and no hold is materialized).
    for _ in range(1000):
        if next_wake_ts is None:
            break
        evaluate(next_wake_ts)
    decisions = len(traces)
    return ReplayResult(
        chat_key=chat_key,
        traces=tuple(traces),
        would_have_spoken=would_have_spoken,
        decisions=decisions,
        rate=(would_have_spoken / decisions) if decisions else 0.0,
    )


def _is_priority(msg: Message, identity: ChatIdentity) -> bool:
    """Structurally recognized direct @/quote: the message mentions the
    chat's self id or carries a reply target. The gate applies the exact
    precedence (a quote triggers only when the target resolves to a self
    message)."""
    return identity.self_id in msg.mentions or msg.reply_to is not None


def _is_priority_batch(
    batch: list[Message],
    identity: ChatIdentity,
    messages: list[Message],
    cursor: int,
    threshold: int,
) -> bool:
    """The App's priority wake rule for ONE committed group: any member is
    a structurally recognized direct @/quote, or the chat's atomic pending
    non-self count (beyond the cursor, self excluded) is at/above the gate
    threshold. Self echoes never qualify."""
    if any(_is_priority(m, identity) for m in batch if not m.is_self):
        return True
    return _pending_count(messages, cursor) >= threshold


def _pending_count(messages: list[Message], cursor: int) -> int:
    """The atomic pending non-self count the live ingest path reports at
    commit time: non-self messages beyond the durable cursor (self echoes
    never inflate it)."""
    return sum(
        1
        for m in messages
        if not m.is_self
        and m.row_id is not None
        and int(m.row_id) > cursor
    )


def _replay_recent(
    messages: list[Message],
    *,
    chat_key: ChatKey,
    through: int,
    since: float,
    limit: int,
) -> RecentSnapshot:
    """The claim-bounded recent read over the in-memory timeline: the
    LIMITED newest-first list plus the FULL-window counts (self included),
    the self count, and the last non-self timestamp — mirroring
    ``Repository.get_recent_snapshot``."""
    window = [
        m
        for m in messages
        if m.row_id is not None
        and m.row_id <= through
        and m.recv_ts is not None
        and m.recv_ts >= since
    ]
    window.sort(
        key=lambda m: int(m.row_id) if m.row_id is not None else 0, reverse=True
    )  # newest first, stable
    self_count = sum(1 for m in window if m.is_self)
    last_nonself = max(
        (m.recv_ts for m in window if not m.is_self and m.recv_ts is not None),
        default=None,
    )
    return RecentSnapshot(
        chat_key=chat_key,
        messages=tuple(window[:limit]),
        window_count=len(window),
        self_count=self_count,
        last_nonself_ts=last_nonself,
        since_ts=since,
        through_row_id=MessageRowId(through),
    )


def sweep_corpus(
    events: Sequence[Any],
    *,
    chat_key: ChatKey,
    identity: ChatIdentity,
    cfg: Config,
    gate: Gate | None = None,
    thresholds: tuple[int, ...] = SWEEP_THRESHOLDS,
    trigger_scores: tuple[int, ...] = SWEEP_TRIGGER_SCORES,
) -> tuple[SweepRow, ...]:
    """Sweep the gate constants through a RuntimeOverlay (at least
    ``gate.threshold`` and ``gate.trigger_score``) and report the
    would-have-spoken count/rate per combination, in deterministic order."""
    rows: list[SweepRow] = []
    for threshold in thresholds:
        for trigger_score in trigger_scores:
            overlay = RuntimeOverlay()
            overlay.set("gate.threshold", threshold)
            overlay.set("gate.trigger_score", trigger_score)
            result = replay_corpus(
                events,
                chat_key=chat_key,
                identity=identity,
                cfg=cfg,
                gate=gate,
                overlay=overlay,
            )
            rows.append(
                SweepRow(
                    threshold=threshold,
                    trigger_score=trigger_score,
                    would_have_spoken=result.would_have_spoken,
                    decisions=result.decisions,
                    rate=result.rate,
                )
            )
    return tuple(rows)


def _evolve_avg_interval(messages: Sequence[Message], now: float) -> float | None:
    """The durable EWMA average as of ``now``, evolved over the non-self
    messages in ROW-ID (commit) order with the repository's exact
    prior-sample semantics (the prior sample is the MAX recv_ts among the
    prior non-self rows), so a receding timestamp is a non-positive gap
    that carries no pacing information and never drags the average. Self
    messages never participate; a first non-self message carries no prior
    sample."""
    avg: float | None = None
    prev_ts: float | None = None
    for m in messages:
        if m.is_self or m.recv_ts is None or m.recv_ts > now:
            continue
        avg = ewma_interval(avg, prev_ts, m.recv_ts)
        if prev_ts is None or m.recv_ts > prev_ts:
            prev_ts = m.recv_ts
    return avg


def _message_from_snapshot(data: dict[str, Any]) -> Message:
    """Rehydrate one ``dataclasses.asdict(GateSnapshot)`` message.

    The v5 dispatch marker persists the exact evaluated snapshot. Replaying
    that immutable input avoids guessing row identity, focus/hold, or timing
    from a later corpus/database state.
    """
    return Message(
        chat_key=ChatKey(data["chat_key"]),
        sender_id=SenderId(data["sender_id"]),
        sender_name=data["sender_name"],
        is_self=bool(data["is_self"]),
        text=data["text"],
        id=MessageId(data["id"]) if data.get("id") is not None else None,
        row_id=(
            MessageRowId(data["row_id"])
            if data.get("row_id") is not None
            else None
        ),
        segments=tuple(Segment(**segment) for segment in data.get("segments") or ()),
        reply_to=(
            MessageId(data["reply_to"])
            if data.get("reply_to") is not None
            else None
        ),
        mentions=tuple(SenderId(value) for value in data.get("mentions") or ()),
        recv_ts=data.get("recv_ts"),
        raw=data.get("raw"),
    )


def _snapshot_from_marker(snapshot_json: str) -> GateSnapshot:
    """Rehydrate the v5 frozen gate input without consulting live state."""
    data = json.loads(snapshot_json)
    pending = tuple(
        _message_from_snapshot(message)
        for message in data.get("pending_messages") or ()
    )
    recent = tuple(
        _message_from_snapshot(message) for message in data.get("recent") or ()
    )
    last_message = (
        _message_from_snapshot(data["last_message"])
        if data.get("last_message") is not None
        else None
    )
    return GateSnapshot(
        chat_key=ChatKey(data["chat_key"]),
        cycle_id=CycleId(data["cycle_id"]),
        start_msg_id=MessageRowId(data["start_msg_id"]),
        through_msg_id=MessageRowId(data["through_msg_id"]),
        evaluated_ts=data["evaluated_ts"],
        self_id=SelfId(data["self_id"]),
        mode=data["mode"],
        threshold=data["threshold"],
        trigger_score=data["trigger_score"],
        frequency=data["frequency"],
        pending=data["pending"],
        pending_messages=pending,
        recent=recent,
        window_count=data["window_count"],
        self_count=data["self_count"],
        last_nonself_ts=data.get("last_nonself_ts"),
        idle_seconds=data["idle_seconds"],
        recent_average_interval=data["recent_average_interval"],
        self_ratio=data["self_ratio"],
        is_group=bool(data["is_group"]),
        is_focused=bool(data["is_focused"]),
        last_message=last_message,
        self_name=data.get("self_name"),
        has_direct_at=bool(data.get("has_direct_at", False)),
        has_quote_to_self=bool(data.get("has_quote_to_self", False)),
        has_other_assistant=bool(data.get("has_other_assistant", False)),
        hold_until=data.get("hold_until"),
        idle_streak=data.get("idle_streak", 0),
        previous_end_reason=data.get("previous_end_reason"),
        backoff_base_s=data.get("backoff_base_s", 15.0),
        backoff_cap_s=data.get("backoff_cap_s", 300.0),
        backoff_start_count=data.get("backoff_start_count", 2),
    )


def _snapshot_with_config(snapshot: GateSnapshot, cfg: ChatConfig) -> GateSnapshot:
    """Rescore a recorded schedule with only the requested gate constants."""
    gate_cfg = cfg.gate
    return dataclasses.replace(
        snapshot,
        mode=gate_cfg.mode,
        threshold=gate_cfg.threshold,
        trigger_score=gate_cfg.trigger_score,
        frequency=gate_cfg.frequency,
        backoff_base_s=gate_cfg.backoff.base_s,
        backoff_cap_s=gate_cfg.backoff.cap_s,
        backoff_start_count=gate_cfg.backoff.start_count,
    )


# ── Marker-driven replay: the v4 recorded dispatch schedule ─────────────────

def replay_marker_schedule(
    view: CorpusView,
    *,
    chat_key: ChatKey,
    identity: ChatIdentity,
    cfg: Config,
    gate: Gate | None = None,
    overlay: RuntimeOverlay | None = None,
    window_s: float = SNAPSHOT_WINDOW_S,
    limit: int = SNAPSHOT_LIMIT,
) -> ReplayResult:
    """Re-score the RECORDED dispatch schedule through the SAME assembler +
    gate path, dispatch by dispatch, in DispatchId order.

    Deterministic and storage-free: no database, outbox, adapter, or clock
    operations — replay never creates any side effect. The input is the
    structured ``CorpusView`` read from the corpus: raw events keyed by
    EventId, commit markers in CommitSeq order, and dispatch markers in
    DispatchId order. Markers may be physically exported in another order
    (a crash can reverse the live writer order), so replay reconstructs by
    their durable sequence/boundary/membership, never by JSONL order.

    - Raw events WITHOUT a durable commit marker are ignored: only events
      referenced by a commit marker enter the message timeline.
    - The message timeline is built in CommitSeq order (row ids follow the
      commit order, so receding timestamps never reorder committed rows).
    - For every SETTLED dispatch marker (``completed`` or ``released``) in
      DispatchId order, the exact attached pending messages are
      reconstructed from the marker's frozen ``attached`` CommitSeqs (self
      messages are preserved only in the recent/presence history, never
      pending), the snapshot uses the marker's frozen start/through
      boundary, cause, scheduled time, and settled evaluation timestamp,
      and the SAME ``assemble_snapshot`` + ``Gate.evaluate`` path runs.
    - Exact replay FAILS CLOSED instead of omitting settled decisions: an
      invalid dispatch marker (a settled state without ``settled_ts``, or
      an unknown state), an attached commit that does not resolve in the
      corpus, a frozen snapshot whose chat/boundary/evaluated_ts/attached
      membership is inconsistent with its marker, and a malformed frozen
      snapshot all raise ``ValueError``. Only a v2/v3 marker (no settled
      state AND no settled timestamp) is skipped gracefully — it is not a
      settled decision.
    - Prior terminal reason/state is reconstructed from the marker
      schedule: a completed (terminal) dispatch sets the previous end
      reason and resets the hold/idle streak; a released (delay) dispatch
      leaves them unchanged. The durable EWMA average evolves from
      non-self messages in commit order through the SAME
      ``pacing.ewma_interval`` reducer.
    - Traces identify their DispatchId in ``snapshot_facts["cycle_id"]``
      (``dispatch:<id>``) and carry the exact original schedule facts
      (evaluated_ts = settled_ts, the start/through boundary, and the
      attached pending set).

    ``overlay`` (a RuntimeOverlay) varies gate constants for sweep runs.
    """
    if overlay is not None:
        cfg = overlay.apply(cfg)
    gate = gate if gate is not None else Gate()
    _set_replay_manifest(gate, cfg)
    chat_cfg = cfg.for_chat(chat_key)
    self_name = cfg.bot.name
    # The message timeline is reconstructed in COMMIT order, but its local
    # row identity comes from the recorded ``message_row_id``.  CommitSeq is
    # a ledger membership key, not a message row id; they can legitimately
    # differ (and exact replay must preserve that distinction).
    messages: list[Message] = []
    commit_to_message: dict[int, Message] = {}
    for commit in view.commits:
        if commit.chat_key != chat_key or commit.event_id is None:
            continue
        event = view.events_by_event_id.get(commit.event_id)
        if event is None or event.type != "message" or not isinstance(
            event.payload, Message
        ):
            continue
        msg = event.payload
        if msg.chat_key != chat_key:
            continue
        row_id = (
            int(commit.message_row_id)
            if commit.message_row_id is not None
            else len(messages) + 1
        )
        if any(m.row_id is not None and int(m.row_id) == row_id for m in messages):
            raise ValueError(
                f"duplicate recorded message_row_id {row_id} in corpus"
            )
        msg = dataclasses.replace(msg, row_id=MessageRowId(row_id))
        messages.append(msg)
        commit_to_message[commit.sequence] = msg
    traces: list[DecisionTrace] = []
    would_have_spoken = 0
    cursor = 0
    state = ChatState(chat_key=chat_key)
    previous_end_reason: str | None = None
    for dispatch in view.dispatches:
        if dispatch.chat_key != chat_key:
            continue
        # Only SETTLED dispatches are replayable. A v2/v3 marker (no
        # settled state AND no settled timestamp) is not a settled decision
        # and is skipped gracefully — old corpora stay readable. Every
        # OTHER invalid dispatch marker fails closed: a settled decision
        # must never be silently omitted from exact replay.
        if dispatch.state is None and dispatch.settled_ts is None:
            continue
        if dispatch.state not in ("completed", "released"):
            raise ValueError(
                f"invalid dispatch marker {dispatch.sequence}:"
                f" state {dispatch.state!r}"
            )
        if dispatch.settled_ts is None:
            raise ValueError(
                f"invalid dispatch marker {dispatch.sequence}:"
                " settled state without settled_ts"
            )
        # Exact replay is never permitted to silently use a different
        # plugin/feature composition.  This check is deliberately performed
        # even when the current composition is empty.
        _check_exact_fingerprint(dispatch, gate)
        # Every attached commit must resolve in the corpus (both the frozen
        # snapshot path and the reconstruction path): an unresolvable member
        # is a corrupted ledger — fail closed, never silently omit it.
        unresolved = [
            seq for seq in dispatch.attached if seq not in commit_to_message
        ]
        if unresolved:
            raise ValueError(
                f"dispatch marker {dispatch.sequence} attached commit(s) not"
                f" found in corpus: {unresolved}"
            )
        if dispatch.snapshot_json is not None:
            try:
                snapshot = _snapshot_from_marker(dispatch.snapshot_json)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                # A malformed frozen marker is not exact replay evidence.
                # Never silently skip it: a successful replay must represent
                # every settled dispatch in the validated ledger.
                raise ValueError(
                    f"malformed frozen dispatch snapshot (dispatch"
                    f" {dispatch.sequence})"
                ) from None
            if snapshot.chat_key != chat_key:
                raise ValueError(
                    f"dispatch marker {dispatch.sequence} frozen snapshot chat"
                    f" mismatch ({snapshot.chat_key} != {chat_key})"
                )
            expected_cycle = f"dispatch:{dispatch.sequence}"
            if str(snapshot.cycle_id) != expected_cycle:
                raise ValueError(
                    f"dispatch marker {dispatch.sequence} frozen snapshot"
                    f" cycle_id mismatch ({snapshot.cycle_id} != {expected_cycle})"
                )
            # The frozen snapshot must be consistent with the marker's frozen
            # evaluation metadata: the evaluation timestamp, the start/through
            # boundary, and the exact attached membership. An inconsistency is
            # corruption — never silently re-scored.
            if (
                dispatch.evaluated_ts is not None
                and snapshot.evaluated_ts != dispatch.evaluated_ts
            ):
                raise ValueError(
                    f"dispatch marker {dispatch.sequence} frozen snapshot"
                    f" evaluated_ts mismatch"
                    f" ({snapshot.evaluated_ts} != {dispatch.evaluated_ts})"
                )
            if (
                dispatch.start_msg_id is not None
                and snapshot.start_msg_id != dispatch.start_msg_id
            ):
                raise ValueError(
                    f"dispatch marker {dispatch.sequence} frozen snapshot"
                    f" start boundary mismatch"
                    f" ({snapshot.start_msg_id} != {dispatch.start_msg_id})"
                )
            if (
                dispatch.through_msg_id is not None
                and snapshot.through_msg_id != dispatch.through_msg_id
            ):
                raise ValueError(
                    f"dispatch marker {dispatch.sequence} frozen snapshot"
                    f" through boundary mismatch"
                    f" ({snapshot.through_msg_id} != {dispatch.through_msg_id})"
                )
            expected_pending = {
                int(cast(MessageRowId, commit_to_message[seq].row_id))
                for seq in dispatch.attached
                if commit_to_message.get(seq) is not None
                and not commit_to_message[seq].is_self
                and commit_to_message[seq].row_id is not None
            }
            actual_pending = {
                int(m.row_id)
                for m in snapshot.pending_messages
                if m.row_id is not None
            }
            if actual_pending != expected_pending:
                raise ValueError(
                    f"dispatch marker {dispatch.sequence} frozen snapshot"
                    " attached membership mismatch"
                )
            snapshot = _snapshot_with_config(snapshot, chat_cfg)
            trace = gate.evaluate(snapshot)
            trace = _trace_with_fingerprint(trace, gate)
            traces.append(trace)
            decision = trace.decision
            assert decision is not None
            if decision.action == "trigger":
                would_have_spoken += 1
            # Marker schedules are fixed: state transitions/timers already
            # happened live and are captured in the next frozen snapshot.
            continue
        now = dispatch.settled_ts
        # The exact attached pending messages from the marker's frozen
        # membership (self messages stay in the recent/presence history).
        # The cursor filter keeps the sweep honest: a re-scored terminal
        # dispatch advances the cursor, so a later dispatch's attached
        # commits that were already consumed stay out of pending.
        pending = tuple(
            m
            for seq in dispatch.attached
            for m in (commit_to_message[seq],)
            if not m.is_self
            and m.row_id is not None
            and int(m.row_id) > cursor
        )
        through = (
            int(dispatch.through_msg_id)
            if dispatch.through_msg_id is not None
            else (
                int(messages[-1].row_id)
                if messages and messages[-1].row_id is not None
                else 0
            )
        )
        recent = _replay_recent(
            messages, chat_key=chat_key, through=through, since=now - window_s, limit=limit
        )
        quote_self_ids = frozenset(
            m.id
            for m in messages
            if m.is_self
            and m.id is not None
            and m.recv_ts is not None
            and m.recv_ts <= now
        )
        avg = _evolve_avg_interval(messages, now)
        if avg is not None:
            state = dataclasses.replace(state, avg_interval=avg)
        grant = ClaimGrant(
            claim=CycleClaim(
                chat_key,
                CycleId(f"dispatch:{dispatch.sequence}"),
                started_ts=now,
                expires_at=now + 1.0,
            ),
            start_msg_id=(
                dispatch.start_msg_id
                if dispatch.start_msg_id is not None
                else MessageRowId(0)
            ),
            through_msg_id=MessageRowId(through),
            pending=pending,
        )
        snapshot = assemble_snapshot(
            grant=grant,
            identity=identity,
            state=state,
            recent=recent,
            cfg=chat_cfg,
            now=now,
            self_name=self_name,
            muted=is_muted(cfg.access, chat_key),
            previous_end_reason=previous_end_reason,
            window_s=window_s,
            quote_self_ids=quote_self_ids,
        )
        trace = gate.evaluate(snapshot)
        trace = _trace_with_fingerprint(trace, gate)
        traces.append(trace)
        decision = trace.decision
        assert decision is not None
        if decision.action == "trigger":
            would_have_spoken += 1
            cursor = through
            state = dataclasses.replace(state, hold_until=None, idle_streak=0)
            previous_end_reason = "dry_run_trigger"
        elif decision.action == "skip":
            cursor = through
            state = dataclasses.replace(state, hold_until=None, idle_streak=0)
            previous_end_reason = "skip"
        # delay: a released dispatch — no cursor/session change.
    decisions = len(traces)
    return ReplayResult(
        chat_key=chat_key,
        traces=tuple(traces),
        would_have_spoken=would_have_spoken,
        decisions=decisions,
        rate=(would_have_spoken / decisions) if decisions else 0.0,
    )


def sweep_marker_schedule(
    view: CorpusView,
    *,
    chat_key: ChatKey,
    identity: ChatIdentity,
    cfg: Config,
    gate: Gate | None = None,
    thresholds: tuple[int, ...] = SWEEP_THRESHOLDS,
    trigger_scores: tuple[int, ...] = SWEEP_TRIGGER_SCORES,
) -> tuple[SweepRow, ...]:
    """Sweep the gate constants through a RuntimeOverlay over the RECORDED
    dispatch schedule (at least ``gate.threshold`` and
    ``gate.trigger_score``) and report the would-have-spoken count/rate per
    combination, in deterministic order.

    The dispatch schedule is FIXED: every recorded settled dispatch is
    re-scored under the overlay constants — no counterfactual future timer
    events are invented. Traces identify their DispatchId and the exact
    original schedule facts (see ``replay_marker_schedule``)."""
    rows: list[SweepRow] = []
    for threshold in thresholds:
        for trigger_score in trigger_scores:
            overlay = RuntimeOverlay()
            overlay.set("gate.threshold", threshold)
            overlay.set("gate.trigger_score", trigger_score)
            result = replay_marker_schedule(
                view,
                chat_key=chat_key,
                identity=identity,
                cfg=cfg,
                gate=gate,
                overlay=overlay,
            )
            rows.append(
                SweepRow(
                    threshold=threshold,
                    trigger_score=trigger_score,
                    would_have_spoken=result.would_have_spoken,
                    decisions=result.decisions,
                    rate=result.rate,
                )
            )
    return tuple(rows)
