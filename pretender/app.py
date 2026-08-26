"""The App: composes cfg, db, repo, clock, recorder, adapter, ingest, outbox.

Phase 3 is console-only and non-network:

  - ``build`` REJECTS any non-console adapter — no live adapter may be
    configured yet, and no message may reach a real send path ungated.
  - ``dry_run=True`` runs the deterministic DISPATCH-LEDGER lane by default:
    the App wires a ``LedgerScheduler`` over ``Repository`` + ``Clock`` +
    ``CycleRunner.run_dispatch`` (with the runner's at-least-once dispatch
    marker exporter wired to ``record.export_marker``), and the OutboxDriver
    is NEVER started or drained and ``adapter.send`` is NEVER invoked — even
    with pre-existing pending rows. Every AdapterEvent is committed through
    Ingest IMMEDIATELY (recorder + database durability before anything
    else), and every ``IngestResult`` with a durable commit sequence is
    submitted IMMEDIATELY to ``LedgerScheduler.notify_commit(chat_key,
    commit_seq)`` — the App performs no scheduler timer/wake arbitration and
    retains no raw events/metadata. Non-message/self/duplicate results with
    no commit remain no-op (a self echo commits with ``wake_kind`` ``none``
    and is never attached; ``begin_dispatch`` returns None when there is no
    eligible work). At startup the App first runs
    ``record.export_unexported(recorder, repo)`` (repairing marker crash
    gaps) and then ``LedgerScheduler.recover(list_ledger_pending_chats())``
    (resuming unassigned commits), so a crash/restart re-evaluates pending
    messages, a durable active hold schedules only its remaining time, and a
    durable wait/retry barrier re-arms at its remaining resume_at.
    ``shutdown`` drains the ledger scheduler's in-flight dispatch before
    stopping it, so a terminal decision's trace is persisted/printed before
    cancellation.
  - An EXPLICITLY injected generic ``Scheduler`` (tests) keeps the legacy
    wake path: the next-turn flush coalescer, ``wake``/``wake_priority``
    arbitration, and ``list_pending_chats`` startup recovery. The default
    production dry-run is ledger-only.
  - ``dry_run=False`` with an agent is the LIVE agent lane: the adapter is
    connected and its readiness/handshake awaited FIRST (no worker send while
    the adapter is disconnected), then pre-existing SAFE pending/future
    outbox rows are recovered across ALL chats, the LedgerScheduler runs the
    two-stage agent, and the outbox worker is woken after a successful
    terminal agent output creates outbox rows (the CycleRunner's
    ``on_outbox`` callback). Active chats are retained across worker rounds,
    so future-paced rows are rechecked when their sleep expires without a new
    wake. ``dry_run=False`` without an agent keeps the Phase 1 legacy
    behavior: the same cross-chat startup recovery plus a cancellation-safe
    outbox worker that paces future ``send_after_ts`` rows.
  - A Phase 3 agent is built when configured (injected ``agent`` wins; else
    injected ``planner``/``replyer``/``budget`` are wrapped; else a default
    agent is built from the config when the ``planner``/``reply`` LLM
    profiles exist — owning an OpenAIClient, PromptStore, core registry,
    BudgetManager and PhaseAgent). The default build stays no-agent when no
    profiles are configured.
  - ``start``/``shutdown`` are safe and idempotent — the CLI and tests rely
    on that.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Sequence, cast

from pretender.adapters.console import ConsoleAdapter
from pretender.adapters.onebot import OneBotAdapter
from pretender.budget import BLOCKED, BudgetManager, LearnerBudget
from pretender.clock import RealClock
from pretender.config import Config
from pretender.cycle import (
    AdaptiveContextService,
    CycleRunner,
    PhaseAgent,
)
from pretender.db import Database
from pretender.embed import EmbeddingCache, OptionalEmbeddingService
from pretender.errors import ConfigError, PromptError
from pretender.gate import Gate
from pretender.ingest import DeliveryKeyFn, Ingest
from pretender.learn import (
    SPECS,
    VALIDATORS,
    LearnerPipeline,
    derive_effect_delta,
    escape_untrusted,
    render_records,
    canonical_content,
)
from pretender.learn.effect import EFFECT_CATEGORIZATIONS
from pretender.llm import OpenAIClient
from pretender.log import get_logger
from pretender.media import MediaHarvester, MediaResolvingAdapter, MediaStore
from pretender.memory import MemoryService, default_capsule_summarizer
from pretender.outbox import OutboxDriver
from pretender.person import PersonService
from pretender.prompts import PromptStore
from pretender.record import Recorder, export_marker, export_unexported
from pretender.registry import (
    HookBus,
    PluginLoadResult,
    PluginLoader,
    configured_plugin_manifest,
)
from pretender.repo import SqliteRepository
from pretender.scheduler import LedgerScheduler, Scheduler
from pretender.search import MemorySearch
from pretender.seams import Adapter, Repository
from pretender.tools.chatctl import ChatControlCallbacks
from pretender.tools.core import register_core_tools
from pretender.tools.media import MediaCallbacks
from pretender.types import (
    AdapterEvent,
    ChatIdentity,
    ChatKey,
    IngestResult,
    LearnerSpec,
    Message,
    MessageRowId,
    OutboxItem,
    PlatformId,
    Record,
    RecordHit,
    RuntimeMode,
    SelfId,
    SettlementNotice,
    VectorRow,
    WakeKind,
)
from pretender.vectors import ndarray_to_blob

log = get_logger("app")


#: ``_backfill_chat_into`` outcome: the bounded page was processed and more
#: pages remain — the caller continues the chat in a later bounded turn.
_PAGE_MORE = "more"
#: ``_backfill_chat_into`` outcome: the bounded page was processed and no
#: more pages remain.
_PAGE_DONE = "done"


#: The adapters the App may boot. Dry-run is console-only; live mode also
#: accepts the OneBot v11 bridge. Anything else is rejected.
SUPPORTED_ADAPTERS = frozenset({"console", "onebot"})


def _onebot_identity(chat_key: ChatKey, adapter: Any) -> ChatIdentity | None:
    """Derive the durable identity for a OneBot chat key.

    OneBot chat keys are ``qq:group:<id>`` / ``qq:private:<id>``; the bot's
    self id comes from the adapter's public ``self_id`` (configured and/or
    learned from inbound events). Unknown or malformed keys resolve to None
    (the message is recorded but not committed)."""
    parts = chat_key.split(":")
    if len(parts) < 3 or parts[0] != "qq":
        return None
    kind = parts[1]
    if kind not in ("group", "private"):
        return None
    self_id = (
        getattr(adapter, "self_id", None)
        or getattr(adapter, "_self_id", None)
        or ""
    )
    return ChatIdentity(chat_key, PlatformId("qq"), SelfId(self_id), kind)


def _adapter_identity(adapter: Any, chat_key: ChatKey) -> ChatIdentity | None:
    """Resolve the durable identity for ``chat_key`` under the active
    adapter: the console adapter serves one fixed chat; the OneBot adapter
    derives per-chat identities from the chat key."""
    name = getattr(adapter, "name", None)
    if name == "console":
        return adapter.identity if chat_key == adapter.chat_key else None
    if name == "onebot":
        return _onebot_identity(chat_key, adapter)
    return None


def _adapter_forwards(adapter: Any, chat_key: ChatKey) -> dict[str, str]:
    """The safely chat-scoped forward map the adapter exposes for
    ``chat_key`` (via an optional ``forwards_for`` surface), or an empty map
    when the adapter provides none. The map is scoped to the chat by the
    adapter, so a cross-chat forward is impossible by construction."""
    resolver = getattr(adapter, "forwards_for", None)
    if resolver is None:
        return {}
    try:
        result = resolver(chat_key)
        return dict(result or {})
    except Exception:
        return {}


def _budget_resolver(
    repo: Any, cfg: Config, clock: Any
) -> Callable[[ChatKey], BudgetManager]:
    """A per-chat ``BudgetManager`` resolver over the shared Repository KV:
    each chat's effective ``cfg.for_chat(chat).budget`` config, cached per
    chat. Distinct managers over the same database serialize atomically
    through the ``BudgetStore`` seam, so the planner, the semantic backfill,
    and semantic query embeds share one physical budget state per chat."""
    cache: dict[ChatKey, BudgetManager] = {}

    def resolve(chat_key: ChatKey) -> BudgetManager:
        mgr = cache.get(chat_key)
        budget_cfg = cfg.for_chat(chat_key).budget
        if mgr is None or mgr.config != budget_cfg:
            mgr = BudgetManager(repo, budget_cfg, now=clock.now)
            cache[chat_key] = mgr
        return mgr

    return resolve


def _agent_configured(cfg: Config) -> bool:
    """True when the config carries the planner and reply LLM profiles the
    default agent build needs. Plain ``pretender run`` requires these; the
    dry-run route falls back to no-agent gate-only evaluation without them."""
    profiles = cfg.llm.profiles
    return "planner" in profiles and "reply" in profiles


def _jargon_query(
    repo: Any,
) -> Callable[[ChatKey, str, int], Awaitable[list[RecordHit]]]:
    """A chat-scoped jargon query over the AdaptiveRepository — the deferred
    ``query_jargon`` tool's callback source. The returned callable takes the
    chat at call time (the PhaseAgent binds it per chat at ToolContext
    construction), so a cross-chat lookup is impossible by construction."""

    async def query(chat_key: ChatKey, text: str, limit: int) -> list[RecordHit]:
        return await repo.query_records(chat_key, "jargon", text, limit=limit)

    return query


def _media_callbacks(
    repo: Any, cfg: Config
) -> Callable[[ChatKey], MediaCallbacks | None]:
    """A chat-scoped ``MediaCallbacks`` factory for the deferred media send
    tools (Phase 6 P6.5b). The returned callable binds the catalog surface
    to the chat at call time (the PhaseAgent binds it per chat at
    ToolContext construction), so a cross-chat lookup is impossible by
    construction. Returns None when the media catalog is disabled — the
    tools then fail closed."""

    def resolve(chat_key: ChatKey) -> MediaCallbacks | None:
        media_cfg = cfg.for_chat(chat_key).media
        if not media_cfg.enabled:
            return None
        list_assets = getattr(repo, "list_media_assets", None)
        if list_assets is None:
            return None

        async def resolve_asset(asset_id: int) -> Any:
            # The catalog is capacity-bounded per (chat, kind); a generous
            # listing limit covers every id the planner can see (the prompt
            # listing is bounded the same way). The handler validates the
            # approved status/kind — this only resolves the opaque id.
            assets = await list_assets(chat_key, limit=200)
            for asset in assets:
                if asset.id == asset_id:
                    return asset
            return None

        return MediaCallbacks(
            catalog_enabled=lambda: cfg.for_chat(chat_key).media.enabled,
            resolve_asset=resolve_asset,
        )

    return resolve


def _chat_control_callbacks(
    repo: Any,
) -> Callable[[ChatKey], ChatControlCallbacks | None]:
    """A chat-bound ``ChatControlCallbacks`` factory for the deferred
    set_focus / notify_chat tools (Phase 6 P6.6b). The returned callable
    binds the target-chat validation to the CURRENT chat at ToolContext
    construction, so a cross-account target is impossible by construction.
    Returns None when the repository lacks the chat surface — the tools then
    fail closed."""

    def resolve(chat_key: ChatKey) -> ChatControlCallbacks | None:
        get_chat = getattr(repo, "get_chat", None)
        if get_chat is None:
            return None

        async def check(target_key: str) -> bool:
            try:
                target = await get_chat(ChatKey(target_key))
                if target is None:
                    return False
                source = await get_chat(chat_key)
                if source is None:
                    return False
                return (
                    target.platform == source.platform
                    and target.self_id == source.self_id
                )
            except Exception:
                return False

        return ChatControlCallbacks(resolve_chat=check)

    return resolve


# ── Phase 6 P6.6 plugin staging → live conversion ───────────────────────────

def _gate_from_plugins(result: PluginLoadResult) -> Gate:
    """The live Gate built from the frozen plugin gate-feature staging
    registry (the built-ins + every plugin feature, in registration order)."""
    gate = Gate(features=result.gate_features)
    setattr(gate, "_plugin_manifest", result.plugin_manifest)
    return gate


def build_replay_gate(cfg: Config) -> Gate:
    """Build the exact frozen gate composition selected by ``cfg``.

    Replay has no database, adapter, outbox, or worker side effects, but it
    must execute the same trusted feature implementations as live startup so
    the recorded implementation fingerprint is meaningful.  Plugin loading
    here is limited to the declarative staging/setup lane; runtime clients
    are never constructed.
    """
    return _gate_from_plugins(PluginLoader(cfg).load())


def _core_registry_from_plugins(result: PluginLoadResult) -> Any:
    """The live CoreToolRegistry built from the frozen plugin tools staging
    registry (the built-ins + every plugin tool, in registration order)."""
    from pretender.tools.core import CoreToolRegistry

    reg = CoreToolRegistry()
    for spec in result.tools.all():
        reg.register(spec)
    return reg


def _output_factory_from_plugins(
    result: PluginLoadResult,
) -> Callable[[Any], Any]:
    """The output-pipeline factory that seeds every per-chat pipeline with
    the frozen plugin output stages (registered after the built-ins with
    ``replace=True``, so an allowlisted replacement can shadow a built-in)."""
    extra = result.output_stages.all()

    def factory(config: Any) -> Any:
        from pretender.output.pipeline import OutputPipeline

        return OutputPipeline(config, extra_stages=extra)

    return factory


async def _schedule_harvest(
    harvester: MediaHarvester, msg: Message, row_id: MessageRowId | None
) -> None:
    """The Ingest post-insert harvest callback: schedule a bounded
    background harvest (or None when the policy/kind excludes it). The
    scheduling itself never raises — the harvester contains every failure."""
    harvester.maybe_harvest(msg, row_id)


def _build_learner_specs(
    cfg: Config,
    prompts: PromptStore,
    base_specs: dict[str, LearnerSpec] | None = None,
) -> dict[str, LearnerSpec]:
    """The enabled learner specs from the config's ``learn.profiles``.

    Each profile names a known spec (expression/behavior/jargon/summary/
    effect) whose prompt file must load through the PromptStore; a profile
    that names an unknown learner or a missing prompt file is SKIPPED (the
    doctor reports it truthfully). Profile fields override the spec
    defaults. Returns an empty dict when nothing is valid — the worker then
    never starts.
    """
    available = base_specs if base_specs is not None else SPECS
    specs: dict[str, LearnerSpec] = {}
    for name, profile in cfg.learn.profiles.items():
        base = available.get(name)
        if base is None:
            continue
        try:
            prompts.load(base.prompt)
        except PromptError:
            continue
        specs[name] = LearnerSpec(
            name=base.name,
            prompt=base.prompt,
            cadence_s=profile.cadence_s or base.cadence_s,
            policy=profile.policy or base.policy,
            batch_size=profile.batch_size or base.batch_size,
            enabled=(
                profile.enabled if profile.enabled is not None else base.enabled
            ),
        )
    return specs


def _embed_cache_path(cfg: Config) -> Path:
    """A shared on-disk SHA1 embed cache directory beside the database, so a
    restart resumes the semantic backfill from cache hits (zero provider
    calls for already-embedded texts)."""
    return Path(cfg.storage.db_path).with_name("embed_cache")


class _EscapedAdaptiveContext:
    """Keep model-derived reply style inside the untrusted-data boundary.

    The adaptive service owns selection; this small App-side adapter keeps
    that service (and the legacy injected surface) untouched while ensuring a
    style value cannot close a wrapper in the planner/reply prompt.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def build(self, *args: Any, **kwargs: Any) -> Any:
        context = await self._inner.build(*args, **kwargs)
        style = getattr(context, "reply_style", None)
        if not isinstance(style, str) or not is_dataclass(context):
            return context
        return replace(cast(Any, context), reply_style=escape_untrusted(style))


class SemanticBackfill:
    """Bounded semantic vector maintenance worker (Gate 5).

    Runs OUTSIDE any terminal settlement transaction: it enumerates
    chat-scoped memory records, embeds them through the shared
    ``OptionalEmbeddingService`` (cache hits use zero budget/network), and
    writes vectors into a ``building`` generation that is activated ONLY
    after a fixed complete scan — old generations are preserved until then.
    Every embedding cache miss uses the same per-chat admission/reservation
    model as the planner (``BudgetManager.reserve``), so simultaneous
    planner/embed reservations for a chat can never exceed the cap; a
    blocked/failed chat degrades to FTS-only and retains any provider
    reservation. On cancellation/failure the source memory/FTS/outbox/ledger
    stay correct and a restart resumes (the building generation is found and
    re-scanned; cache hits make the re-scan cheap).

    The terminal memory callback may ``enqueue`` a chat (non-blocking) and
    MUST NOT await provider work — the worker drains the queue in the
    background.

    Gate 5 remediation (frozen Oracle advisory):
      - Activation is SOURCE-FENCED transactionally: the coverage check and
        the activate decision share ONE repository write transaction
        (``activate_embedding_generation_if_complete``), so a memory
        committed during the final scan is visible to the check — activation
        fails and the affected chats are enqueued for repair instead of
        falsely activating a stale generation.
      - Startup/full backfill, dimension discovery, and coverage
        verification are keyset-paged/bounded by fixed page sizes (never
        load all of a chat at once); maintenance stays fair (one bounded
        page per chat per round, re-enqueued for continuation), cancellable
        (between pages), and eventually covers every page.
    """

    _MEMORY_PAGE = 128
    _CHAT_PAGE = 64
    _REPAIR_TICK_S = 1.0

    def __init__(
        self,
        repo: Any,
        embed: OptionalEmbeddingService,
        budget: BudgetManager,
        *,
        model: str,
        revision: str,
        space_id: str,
        probe_text: str = "pretender semantic probe",
        cfg: Config | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._repo = repo
        self._embed = embed
        self._budget = budget
        self._model = model
        self._revision = revision
        self._space_id = space_id
        self._probe_text = probe_text
        # Per-chat budget resolution: when ``cfg``/``now`` are provided, each
        # chat's effective ``BudgetConfig`` (``cfg.for_chat(chat).budget``) is
        # honored — the SAME physical KV state and config the planner's
        # chat-bound BudgetManager uses, so simultaneous planner/embed
        # reservations for a chat can never exceed the cap. Without them
        # (tests), the injected single ``budget`` is used.
        self._cfg = cfg
        self._now = now
        self._budgets: dict[ChatKey, BudgetManager] = {}
        self._queue: asyncio.Queue[ChatKey] = asyncio.Queue(maxsize=256)
        self._queued: set[ChatKey] = set()
        # High-water ids are process-local optimization only. Durable startup
        # full scans repair every vector before trusting them after a restart.
        self._last_memory_id: dict[ChatKey, int] = {}
        self._last_record_id: dict[ChatKey, int] = {}
        self._repair_after = ""
        self._cancelled = False
        # Set once the startup full backfill completes with the generation
        # ACTIVE; tests and callers await it instead of the never-completing
        # background task. A blocked/degraded/failed build never sets it.
        self._built = asyncio.Event()

    # ── public surface ──────────────────────────────────────────────────────

    def enqueue(self, chat_key: ChatKey) -> None:
        """Enqueue a chat for incremental backfill (non-blocking). Called by
        the terminal memory callback; never awaits provider work."""
        # A post-settlement memory makes semantic coverage advisory-stale until
        # its bounded repair finishes; never advertise the old ready event.
        self._built.clear()
        if chat_key in self._queued:
            return
        try:
            self._queue.put_nowait(chat_key)
        except asyncio.QueueFull:
            # Startup full scans repair a dropped advisory enqueue. Do not let
            # a high-volume chat turn maintenance into an unbounded queue.
            log.warning("semantic backfill queue full; coalescing %s", chat_key)
            return
        self._queued.add(chat_key)

    def cancel(self) -> None:
        """Request cooperative cancellation between chat pages."""
        self._cancelled = True

    async def run(self) -> None:
        """The background loop: run a full backfill pass at startup, then
        drain enqueued chats until cancelled. Cancellation (``cancel()`` or
        task cancellation) stops cleanly between chats. Every failure is
        CONTAINED and logged — a semantic worker error never escapes to
        poison app shutdown; the worker keeps draining (a failed chat is
        retried on its next enqueue)."""
        try:
            try:
                await self._full_backfill()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning(
                    "semantic full backfill failed (contained)", exc_info=True
                )
            while not self._cancelled:
                try:
                    chat_key = await asyncio.wait_for(
                        self._queue.get(), timeout=self._REPAIR_TICK_S
                    )
                except asyncio.TimeoutError:
                    await self._repair_next_active_chat()
                    continue
                self._queued.discard(chat_key)
                if self._cancelled:
                    return
                try:
                    await self._backfill_chat(chat_key)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.warning(
                        "semantic backfill failed for %s (contained)",
                        chat_key,
                        exc_info=True,
                    )
                try:
                    # Queue overflow is advisory only. A durable rotating
                    # sweep supplies eventual coverage for dropped enqueues.
                    await self._repair_next_active_chat()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.warning("semantic repair sweep failed (contained)", exc_info=True)
        except asyncio.CancelledError:
            raise

    async def _repair_next_active_chat(self) -> None:
        """Repair one bounded page selected by a durable rotating cursor.

        This path is independent of the bounded low-latency queue, so burst
        traffic cannot strand a chat forever after queue overflow.
        """
        if self._cancelled:
            return
        page = await self._repo.list_memory_chats_after(self._repair_after, limit=1)
        if not page:
            self._repair_after = ""
            return
        chat_key = page[0]
        self._repair_after = str(chat_key)
        gen = await self._active_generation()
        if gen is None:
            return
        outcome = await self._backfill_chat_into(chat_key, gen, incremental=True)
        if outcome == _PAGE_DONE and await self._generation_complete(gen):
            self._built.set()

    # ── full backfill ───────────────────────────────────────────────────────

    async def _full_backfill(self) -> bool:
        """One complete scan: ensure the building generation, backfill every
        chat's memories into it (one bounded page per chat per round; chats
        with more pages are re-enqueued for fair continuation), then activate
        it ONLY after EVERY chat completed successfully. Returns True when
        the generation is active (already active, or activated by this scan);
        False when it stays ``building`` (blocked/degraded/exception/cancel/
        pending pages) so a restart resumes it — a partial generation is
        NEVER falsely healthy. The ``_built`` event is set whenever the
        generation is active.

        Activation is SOURCE-FENCED transactionally: the coverage check and
        the activate decision share one repository write transaction, so a
        memory committed during the final scan makes activation fail and the
        affected chats are enqueued for repair (the next enqueue/startup
        completes it)."""
        active = await self._active_generation()
        if active is not None:
            # An active generation is derived state, never a proof of complete
            # coverage. Repair it on every startup/full pass so a crash between
            # memory commit and enqueue cannot leave semantic recall stale.
            pending_pages = False
            async for chat_key in self._iter_memory_chats():
                if self._cancelled:
                    return False
                outcome = await self._backfill_chat_into(
                    chat_key, active, incremental=False
                )
                if outcome is False:
                    return False
                if outcome == _PAGE_MORE:
                    pending_pages = True
                    self.enqueue(chat_key)
            async for chat_key in self._iter_record_chats():
                if self._cancelled:
                    return False
                if chat_key in self._last_memory_id:
                    continue
                outcome = await self._backfill_chat_into(
                    chat_key, active, incremental=False
                )
                if outcome is False:
                    return False
                if outcome == _PAGE_MORE:
                    pending_pages = True
                    self.enqueue(chat_key)
            if self._cancelled:
                return False
            if pending_pages:
                # The active generation stays usable for existing vectors, but
                # it is not *ready* until every queued repair page is checked.
                return False
            if not await self._generation_complete(active):
                return False
            self._built.set()
            return True
        gen = await self._ensure_building_generation()
        if gen is None:
            # Already active, or nothing to build (no chats / blocked at the
            # dimension-determining embed / a manual inactive generation
            # never hijacked).
            active = await self._active_generation() is not None
            if active:
                self._built.set()
            return active
        pending_pages = False
        async for chat_key in self._iter_memory_chats():
            if self._cancelled:
                return False
            outcome = await self._backfill_chat_into(
                chat_key, gen, incremental=False
            )
            if outcome is False:
                # Blocked/degraded/exception for this chat: leave the
                # generation literally building for a restart to resume.
                return False
            if outcome == _PAGE_MORE:
                pending_pages = True
                self.enqueue(chat_key)
        async for chat_key in self._iter_record_chats():
            if self._cancelled:
                return False
            if chat_key in self._last_memory_id:
                continue
            outcome = await self._backfill_chat_into(chat_key, gen, incremental=False)
            if outcome is False:
                return False
            if outcome == _PAGE_MORE:
                pending_pages = True
                self.enqueue(chat_key)
        if self._cancelled:
            return False
        if pending_pages:
            # More pages remain for some chat(s): the queue continues the
            # scan in later bounded turns; the generation stays building
            # until every page is covered (fair, cancellable, eventual).
            return False
        # Source-fenced activation: the coverage check and the activate
        # decision share ONE writer transaction, so a memory committed during
        # the scan is visible to the check — activation fails and the
        # affected chats are enqueued for repair instead of falsely
        # activating a stale generation.
        repair = await self._repo.activate_embedding_generation_if_complete(gen.id)
        if repair is not None:
            for chat_key in repair:
                self.enqueue(chat_key)
            return False
        self._built.set()
        return True

    async def _iter_memory_chats(self) -> AsyncIterator[ChatKey]:
        """Keyset-paged chat enumeration: fixed pages, deterministic order,
        cancellation between pages — never loads every chat at once. A chat
        added mid-scan is caught by the transactional activation fence (its
        memory is missing a vector -> activation fails -> repair enqueue)."""
        after = ""
        while not self._cancelled:
            page = await self._repo.list_memory_chats_after(
                after, limit=self._CHAT_PAGE
            )
            if not page:
                return
            for chat_key in page:
                yield chat_key
            after = page[-1]

    async def _ensure_building_generation(
        self, chats: list[ChatKey] | None = None
    ) -> Any:
        """The building generation to write into, or None when nothing to
        build (an active matching generation already exists, the embed
        degraded, or the dimension-determining embed reservation was
        blocked).

        Creates a fresh ``building`` generation for the configured space
        and, on a restart, RESUMES the matching ``building`` generation —
        the schema ``building`` state is the ONLY in-progress marker. An
        arbitrary inactive/legacy matching generation is NEVER treated as
        in-progress: it is left alone (stays inactive, never written into,
        never activated) so manual/legacy rows are never hijacked.

        The embedding DIMENSION is obtained from a REAL chat-scoped batch
        under that chat's budget — never a non-chat probe charge. Cache hits
        use zero budget/network; a cache miss reserves exactly
        ``ceil(misses / batch_size)`` calls against the first chat's budget.
        The discovery is keyset-paged: it never loads all of a chat's
        memories at once.
        """
        gens = await self._repo.list_embedding_generations()
        for g in gens:
            if g.state == "active" and g.space_id == self._space_id:
                return None  # already built and active
        # Resume ONLY a matching ``building`` generation (the state this
        # worker creates below). Resuming it re-scans and re-writes
        # idempotently (cache hits make the re-scan cheap).
        building = next(
            (
                g
                for g in gens
                if g.state == "building" and g.space_id == self._space_id
            ),
            None,
        )
        if building is not None:
            return building  # resume an interrupted build
        # A matching generation that is NOT building (a manual inactive or
        # legacy-derived row) is never treated as in-progress: the worker
        # does not write into it and does not activate it.
        if any(g.space_id == self._space_id for g in gens):
            return None  # manual/legacy inactive generation: never hijacked
        dim = await self._discover_dimension(chats)
        if dim is None:
            return None
        return await self._repo.create_embedding_generation(
            self._model, dim, revision=self._revision, state="building"
        )

    async def _discover_dimension(
        self, chats: list[ChatKey] | None
    ) -> int | None:
        """The embedding dimension from a REAL chat-scoped batch under that
        chat's budget (no non-chat probe charging). Keyset-paged: memory
        pages are bounded and cancellation is checked between pages — never
        loads all of a chat's memories at once. A cache miss reserves
        exactly ``ceil(misses / batch_size)`` calls; cache hits use zero.
        Returns the dimension, or None when nothing to embed / blocked /
        degraded / cancelled."""
        if chats is None:
            async for chat_key in self._iter_memory_chats():
                dim = await self._discover_dimension_for_chat(chat_key)
                if dim is not None:
                    return dim
            async for chat_key in self._iter_record_chats():
                dim = await self._discover_record_dimension_for_chat(chat_key)
                if dim is not None:
                    return dim
            return None
        for chat_key in chats:
            dim = await self._discover_dimension_for_chat(chat_key)
            if dim is not None:
                return dim
            dim = await self._discover_record_dimension_for_chat(chat_key)
            if dim is not None:
                return dim
        return None

    async def _discover_dimension_for_chat(self, chat_key: ChatKey) -> int | None:
        """One chat's bounded dimension discovery: the first bounded page of
        nonempty memory texts, embedded under the chat's budget."""
        after_id = 0
        while not self._cancelled:
            memories = await self._repo.list_memories_after(
                chat_key, after_id, limit=self._MEMORY_PAGE
            )
            if not memories:
                return None
            nonempty = [m for m in memories if m.text]
            if nonempty:
                texts = [m.text for m in nonempty]
                misses = self._uncached_unique(texts)
                if misses:
                    batches = math.ceil(len(misses) / self._embed.batch_size)
                    decision = await self._budget_for(chat_key).reserve(
                        chat_key, calls=batches
                    )
                    if decision.kind == BLOCKED or decision.semantic_only:
                        return None
                result = await self._embed.embed(texts)
                if result.status != "ok" or not result.vectors:
                    return None
                return int(result.vectors[0].shape[0])
            after_id = max((m.id or 0 for m in memories), default=after_id)
        return None

    async def _discover_record_dimension_for_chat(self, chat_key: ChatKey) -> int | None:
        """Discover a space dimension from a real adaptive-record batch."""
        if not hasattr(self._repo, "list_adaptive_records_after"):
            return None
        records = await self._repo.list_adaptive_records_after(chat_key, 0, limit=self._MEMORY_PAGE)
        texts = [self._record_text(record) for record in records if self._record_text(record)]
        if not texts:
            return None
        misses = self._uncached_unique(texts)
        if misses:
            batches = math.ceil(len(misses) / self._embed.batch_size)
            decision = await self._budget_for(chat_key).reserve(chat_key, calls=batches)
            if decision.kind == BLOCKED or decision.semantic_only:
                return None
        result = await self._embed.embed(texts)
        if result.status != "ok" or not result.vectors:
            return None
        return int(result.vectors[0].shape[0])

    # ── per-chat backfill ───────────────────────────────────────────────────

    async def _backfill_chat(self, chat_key: ChatKey) -> None:
        """Incremental backfill of one enqueued chat into the ACTIVE matching
        generation. When no active generation exists yet (a zero-memory
        startup, or a still-building generation), kick off a full scan so new
        memory builds instead of no-op'ing until restart."""
        gen = await self._active_generation()
        if gen is None:
            await self._full_backfill()
            return
        outcome = await self._backfill_chat_into(chat_key, gen, incremental=True)
        if outcome == _PAGE_MORE:
            # Continue a large chat in another bounded turn; coalescing keeps
            # one hot chat from monopolizing the maintenance worker.
            self.enqueue(chat_key)
        elif outcome == _PAGE_DONE and await self._generation_complete(gen):
            # A paged active-generation repair becomes ready only after the
            # final page and a full durable coverage pass, never after its
            # first bounded page.
            self._built.set()

    async def _active_generation(self) -> Any:
        for g in await self._repo.list_embedding_generations():
            if g.state == "active" and g.space_id == self._space_id:
                return g
        return None

    async def _backfill_chat_into(
        self, chat_key: ChatKey, gen: Any, *, incremental: bool = False
    ) -> str | bool:
        """Embed ONE bounded page of the chat's memory texts and write
        matching vectors into ``gen``. Cache hits use zero budget/network; a
        cache miss reserves exactly ``ceil(misses / batch_size)`` calls for
        the chat (blocked -> FTS-only, retaining nothing). A provider failure
        degrades to FTS-only and retains the reservation.

        Returns ``_PAGE_MORE`` when the page was processed and more pages
        remain (the caller continues the chat in a later bounded turn);
        ``_PAGE_DONE`` when the page was processed and no more pages remain;
        False when it was blocked/degraded/cancelled (the caller must NOT
        activate the generation)."""
        if self._cancelled:
            return False
        after_id = self._last_memory_id.get(chat_key, 0)
        memories = await self._repo.list_memories_after(
            chat_key, after_id, limit=self._MEMORY_PAGE
        )
        if not memories:
            memory_outcome: str | bool = _PAGE_DONE
        else:
            memory_outcome = await self._backfill_memory_page(chat_key, gen, memories, after_id)
        if memory_outcome is False:
            return False
        record_outcome = await self._backfill_record_page(chat_key, gen)
        if record_outcome is False:
            return False
        if memory_outcome == _PAGE_MORE or record_outcome == _PAGE_MORE:
            return _PAGE_MORE
        return _PAGE_DONE

    async def _backfill_memory_page(
        self, chat_key: ChatKey, gen: Any, memories: list[Any], after_id: int
    ) -> str | bool:
        """Process one bounded page of memory owners."""
        nonempty = [m for m in memories if m.text]
        if nonempty:
            existing = {
                row.owner_id: row
                for row in await self._repo.list_vectors_for_memories(
                    chat_key, gen.model, gen.id, [m.id for m in nonempty]
                )
            }
            pending = [
                mem
                for mem in nonempty
                if not self._vector_matches(existing.get(mem.id), mem, gen)
            ]
            if pending:
                texts = [m.text for m in pending]
                misses = self._uncached_unique(texts)
                if misses:
                    batches = math.ceil(len(misses) / self._embed.batch_size)
                    decision = await self._budget_for(chat_key).reserve(
                        chat_key, calls=batches
                    )
                    if decision.kind == BLOCKED or decision.semantic_only:
                        return False  # degrade to FTS-only for this chat
                result = await self._embed.embed(texts)
                if result.status != "ok":
                    return False  # degraded: FTS-only
                for mem, vec in zip(pending, result.vectors):
                    row = VectorRow(
                        owner_table="memories",
                        owner_id=mem.id,
                        dim=gen.dim,
                        model=gen.model,
                        generation=gen.id,
                        blob=ndarray_to_blob(vec),
                        source_hash=mem.source_hash,
                    )
                    await self._repo.upsert_vector(chat_key, row)
        newest = max((m.id or 0 for m in memories), default=after_id)
        self._last_memory_id[chat_key] = max(after_id, newest)
        if len(memories) == self._MEMORY_PAGE:
            return _PAGE_MORE
        return _PAGE_DONE

    async def _backfill_record_page(self, chat_key: ChatKey, gen: Any) -> str | bool:
        """Process one bounded page of trusted records as vector owners."""
        if not hasattr(self._repo, "list_adaptive_records_after"):
            return _PAGE_DONE
        after_id = self._last_record_id.get(chat_key, 0)
        records = await self._repo.list_adaptive_records_after(
            chat_key, after_id, limit=self._MEMORY_PAGE
        )
        if not records:
            return _PAGE_DONE
        pending_records = [r for r in records if self._record_text(r)]
        if pending_records:
            ids = [r.id for r in pending_records if r.id is not None]
            existing = {
                row.owner_id: row
                for row in await self._repo.list_vectors_for_records(
                    chat_key, gen.model, gen.id, ids
                )
            }
            pending = [
                r for r in pending_records
                if not self._record_vector_matches(existing.get(r.id), r, gen)
            ]
            if pending:
                texts = [self._record_text(r) for r in pending]
                misses = self._uncached_unique(texts)
                if misses:
                    batches = math.ceil(len(misses) / self._embed.batch_size)
                    decision = await self._budget_for(chat_key).reserve(
                        chat_key, calls=batches
                    )
                    if decision.kind == BLOCKED or decision.semantic_only:
                        return False
                result = await self._embed.embed(texts)
                if result.status != "ok":
                    return False
                for record, vec in zip(pending, result.vectors):
                    if record.id is None or record.content_hash is None:
                        continue
                    await self._repo.upsert_vector(
                        chat_key,
                        VectorRow(
                            owner_table="records",
                            owner_id=record.id,
                            dim=gen.dim,
                            model=gen.model,
                            generation=gen.id,
                            blob=ndarray_to_blob(vec),
                            source_hash=record.content_hash,
                        ),
                    )
        newest = max((r.id or 0 for r in records), default=after_id)
        self._last_record_id[chat_key] = max(after_id, newest)
        return _PAGE_MORE if len(records) == self._MEMORY_PAGE else _PAGE_DONE

    @staticmethod
    def _record_text(record: Any) -> str:
        payload = record.payload if isinstance(record.payload, dict) else {}
        text = payload.get("text")
        return text if isinstance(text, str) and text.strip() else canonical_content(payload)

    @staticmethod
    def _record_vector_matches(row: Any, record: Any, gen: Any) -> bool:
        return (
            row is not None
            and row.owner_table == "records"
            and row.dim == gen.dim
            and row.model == gen.model
            and row.generation == gen.id
            and row.source_hash == record.content_hash
        )

    async def _iter_record_chats(self) -> AsyncIterator[ChatKey]:
        """Keyset-paged adaptive-record chat enumeration."""
        if not hasattr(self._repo, "list_adaptive_record_chats_after"):
            return
        after = ""
        while not self._cancelled:
            page = await self._repo.list_adaptive_record_chats_after(
                ChatKey(after), limit=self._CHAT_PAGE
            )
            if not page:
                return
            for chat_key in page:
                yield chat_key
            after = str(page[-1])

    @staticmethod
    def _vector_matches(row: Any, memory: Any, gen: Any) -> bool:
        """True when ``row`` is a valid matching vector for ``memory`` in
        ``gen``: same model/dim (the lookup is already generation/model-
        scoped) and the memory's current ``source_hash``."""
        return (
            row is not None
            and row.owner_table == "memories"
            and row.model == gen.model
            and row.generation == gen.id
            and row.dim == gen.dim
            and row.source_hash == memory.source_hash
        )

    def _uncached_unique(self, texts: list[str]) -> list[str]:
        unique = list(dict.fromkeys(texts))
        cached = set(self._embed.cached(unique))
        return [text for text in unique if text not in cached]

    async def _generation_complete(self, gen: Any) -> bool:
        """Verify durable coverage before relying on a generation.
        Keyset-paged: memory pages are bounded and cancellation is checked
        between pages — never loads all of a chat's memories/vectors at
        once."""
        async for chat_key in self._iter_memory_chats():
            after_id = 0
            while not self._cancelled:
                memories = await self._repo.list_memories_after(
                    chat_key, after_id, limit=self._MEMORY_PAGE
                )
                if not memories:
                    break
                nonempty = [m for m in memories if m.text]
                if nonempty:
                    existing = {
                        row.owner_id: row
                        for row in await self._repo.list_vectors_for_memories(
                            chat_key, gen.model, gen.id, [m.id for m in nonempty]
                        )
                    }
                    for memory in nonempty:
                        if not self._vector_matches(
                            existing.get(memory.id), memory, gen
                        ):
                            return False
                after_id = max((m.id or 0 for m in memories), default=after_id)
        async for chat_key in self._iter_record_chats():
            after_id = 0
            while not self._cancelled:
                records = await self._repo.list_adaptive_records_after(
                    chat_key, after_id, limit=self._MEMORY_PAGE
                )
                if not records:
                    break
                trusted = [r for r in records if self._record_text(r)]
                if trusted:
                    existing = {
                        row.owner_id: row
                        for row in await self._repo.list_vectors_for_records(
                            chat_key, gen.model, gen.id,
                            [r.id for r in trusted if r.id is not None],
                        )
                    }
                    for record in trusted:
                        if not self._record_vector_matches(
                            existing.get(record.id), record, gen
                        ):
                            return False
                after_id = max((r.id or 0 for r in records), default=after_id)
        return True

    def _budget_for(self, chat_key: ChatKey) -> BudgetManager:
        """The chat-bound BudgetManager for ``chat_key``, built from the
        chat's effective ``BudgetConfig`` (``cfg.for_chat(chat).budget``) and
        cached per chat — the SAME physical KV state/config the planner's
        chat-bound manager uses. Without per-chat config (tests), the
        injected single ``budget`` is used."""
        if self._cfg is not None and self._now is not None:
            mgr = self._budgets.get(chat_key)
            budget_cfg = self._cfg.for_chat(chat_key).budget
            if mgr is None or mgr.config != budget_cfg:
                mgr = BudgetManager(self._repo, budget_cfg, now=self._now)
                self._budgets[chat_key] = mgr
            return mgr
        return self._budget


class LearnerScheduler:
    """The bounded cancellable learner worker (Phase 6 P6.4).

    Owned by the App; starts ONLY in ``RuntimeMode.LIVE`` with ``learn``
    enabled and at least one valid profile. It feeds the generic
    ``LearnerPipeline`` (one provider completion per run, NO retry) with the
    shared per-chat ``LearnerBudget`` (the same physical budget state the
    foreground planner uses, bounded to ``concurrency - foreground_reserve``
    concurrent runs).

    Scheduling is a bounded queue plus durable state:

    - **Durable state recovery** — at startup every chat with pending source
      beyond a learner's durable watermark (``list_learner_pending_chats``)
      is enqueued; the watermark is the only in-progress marker.
    - **Periodic scans** — a background scan re-enqueues pending chats, so a
      dropped enqueue (queue overflow) or new source is eventually covered.
    - **Source oldest-first cadence** — the repository's
      ``read_learner_source_batch`` reads the OLDEST bounded unsummarized
      chunk, so each run consumes the oldest pending source first.
    - **Post-settlement enqueue only** — the App enqueues a chat ONLY after
      a terminal dispatch settlement (after settle_dispatch, the outbox
      wake, and the marker export); the enqueue is non-blocking and
      settlement never awaits LLM/learn/media work.

    Failures are CONTAINED (logged, the worker keeps draining); cancellation
    is re-raised after a best-effort settle (the pipeline settles the run
    ``cancelled``). The worker never runs in dry_run/replay/doctor and never
    writes records/vectors/outbox/ledger directly — the only write surface
    is the pipeline's ``commit_learner_source``.

    Effect handling (P6.4b): the worker holds the chat's pending effect
    references (the records frozen into a terminal reply prompt, handed over
    by the App's exposure callback — in-memory only, never reconstructed
    across a crash gap). The effect learner runs ONLY when the references
    exist, delivery was confirmed (the outbox worker's ``mark_delivered``),
    and a human follow-up is pending beyond the effect watermark; the
    judgment's categorization/confidence is derived by code
    (``derive_effect_delta``) and applied once per referenced record.
    """

    _QUEUE_MAX = 256
    _TICK_S = 1.0
    _NOTICE_LIMIT = 64
    #: The bounded eligibility scan for the effect learner: reads at most
    #: this many source rows beyond the effect watermark to prove a human
    #: follow-up exists after the reference boundary.
    _EFFECT_SCAN_TAIL = 10_000
    _MAX_ROW_ID = 2**63 - 1

    def __init__(
        self,
        repo: Any,
        pipeline: Any,
        specs: dict[str, LearnerSpec],
        clock: Any,
        *,
        scan_interval_s: float = 60.0,
        budget: LearnerBudget | None = None,
    ) -> None:
        self._repo = repo
        self._pipeline = pipeline
        self._specs = dict(specs)
        self._clock = clock
        self._scan_interval_s = float(scan_interval_s)
        self._budget = budget
        self._slots = budget.slots if budget is not None else 1
        self._queue: asyncio.Queue[tuple[ChatKey, str]] = asyncio.Queue(
            maxsize=self._QUEUE_MAX
        )
        self._queued: set[tuple[ChatKey, str]] = set()
        self._in_flight: set[tuple[ChatKey, str]] = set()
        self._cancelled = False
        self._scan_task: asyncio.Task[None] | None = None
        self._workers: list[asyncio.Task[None]] = []
        # Pending effect references per chat (in-memory only — a crash gap
        # is never reconstructed): the records frozen into the last terminal
        # reply prompt.
        self._effect_refs: dict[ChatKey, list[Record]] = {}
        # The effect reference boundary per chat: the dispatch's through
        # boundary when the references were shown (a human follow-up is a
        # message AFTER this boundary).
        self._effect_boundary: dict[ChatKey, MessageRowId] = {}
        self._effect_dispatch: dict[ChatKey, int] = {}
        # Chats whose pending outbox rows were confirmed sent (the outbox
        # worker's ``mark_delivered``).
        self._delivered: set[ChatKey] = set()
        self._delivered_dispatches: set[tuple[ChatKey, int]] = set()
        # Bounded settlement-notice log (tests/diagnostics).
        self._notices: list[SettlementNotice] = []

    # ── public surface ──────────────────────────────────────────────────────

    def enqueue(self, chat_key: ChatKey) -> None:
        """Enqueue every enabled learner for ``chat_key`` (non-blocking).

        Called by the App AFTER a terminal dispatch settlement (post
        settle_dispatch/outbox wake/marker export). Never awaits LLM/learn/
        media work; a full queue coalesces (the periodic scan re-finds the
        chat)."""
        for learner in self._specs:
            self._enqueue(chat_key, learner)

    def note_exposure(
        self,
        chat_key: ChatKey,
        records: Sequence[Record],
        through_msg_id: MessageRowId,
        dispatch_id: int | None = None,
    ) -> None:
        """Record the frozen records of a terminal reply dispatch as pending
        effect references (in-memory only — never reconstructed across a
        crash gap), with the dispatch's through boundary as the effect
        reference boundary (a human follow-up is a message AFTER it)."""
        if records:
            self._effect_refs[chat_key] = list(records)
            self._effect_boundary[chat_key] = through_msg_id
            if dispatch_id is not None:
                self._effect_dispatch[chat_key] = dispatch_id
                self._delivered_dispatches.discard((chat_key, dispatch_id))

    def mark_delivered(self, chat_key: ChatKey) -> None:
        """Confirm delivery for ``chat_key``'s pending outbox rows (the
        outbox worker calls this after a successful send)."""
        self._delivered.add(chat_key)
        dispatch_id = self._effect_dispatch.get(chat_key)
        if dispatch_id is not None:
            self._delivered_dispatches.add((chat_key, dispatch_id))

    def cancel(self) -> None:
        """Request cooperative cancellation between runs."""
        self._cancelled = True

    @property
    def notices(self) -> tuple[SettlementNotice, ...]:
        return tuple(self._notices)

    @property
    def specs(self) -> dict[str, LearnerSpec]:
        return dict(self._specs)

    # ── the background loop ─────────────────────────────────────────────────

    async def run(self) -> None:
        """Recover durable pending work, start the periodic scan, and drain
        the queue with bounded concurrency until cancelled. Cancellation
        cancels the scan and every worker and re-raises."""
        try:
            await self._recover()
            self._scan_task = asyncio.create_task(self._scan_loop())
            self._workers = [
                asyncio.create_task(self._worker()) for _ in range(self._slots)
            ]
            try:
                await asyncio.gather(*self._workers)
            except asyncio.CancelledError:
                for worker in self._workers:
                    worker.cancel()
                raise
        except asyncio.CancelledError:
            raise
        finally:
            if self._scan_task is not None:
                self._scan_task.cancel()
                try:
                    await self._scan_task
                except asyncio.CancelledError:
                    pass
                self._scan_task = None
            self._workers = []

    async def _worker(self) -> None:
        """One bounded worker: pull queue items and run them (the pipeline
        itself is the single-flight guard — a busy/leased chat+learner
        skips with zero calls)."""
        while not self._cancelled:
            try:
                item = await asyncio.wait_for(
                    self._queue.get(), timeout=self._TICK_S
                )
            except asyncio.TimeoutError:
                continue
            self._queued.discard(item)
            if self._cancelled:
                return
            await self._run_one(item)

    async def _recover(self) -> None:
        """Durable state recovery: enqueue every chat with pending source
        beyond each learner's durable watermark."""
        for learner in self._specs:
            try:
                for chat_key in await self._pending(learner):
                    self._enqueue(chat_key, learner)
            except Exception:
                log.warning(
                    "learner recovery failed for %s (contained)",
                    learner,
                    exc_info=True,
                )

    async def _scan_loop(self) -> None:
        """Periodic scan: re-enqueue pending chats so a dropped enqueue
        (queue overflow) or new source is eventually covered."""
        while not self._cancelled:
            await self._clock.sleep(self._scan_interval_s)
            if self._cancelled:
                return
            try:
                await self._scan()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("learner scan failed (contained)", exc_info=True)

    async def _scan(self) -> None:
        for learner in self._specs:
            for chat_key in await self._pending(learner):
                self._enqueue(chat_key, learner)

    async def _pending(self, learner: str) -> list[ChatKey]:
        """Read only source that is due, with compatibility for old fakes."""
        spec = self._specs[learner]
        try:
            return await self._repo.list_learner_pending_chats(
                learner, policy=spec.policy, now=self._clock.now()
            )
        except TypeError:
            return await self._repo.list_learner_pending_chats(learner)

    # ── per-item run ────────────────────────────────────────────────────────

    async def _run_one(self, item: tuple[ChatKey, str]) -> None:
        chat_key, learner = item
        if (chat_key, learner) in self._in_flight:
            return
        self._in_flight.add((chat_key, learner))
        try:
            try:
                state = await self._repo.get_learner_state(chat_key, learner)
                if (
                    state is not None
                    and state.next_due_ts is not None
                    and state.next_due_ts > self._clock.now()
                ):
                    return
            except (AttributeError, NotImplementedError):
                pass
            spec = self._specs[learner]
            if learner == "effect":
                await self._run_effect(chat_key, spec)
            else:
                await self._run_pipeline(chat_key, spec)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning(
                "learner run failed for %s/%s (contained)",
                chat_key,
                learner,
                exc_info=True,
            )
        finally:
            self._in_flight.discard((chat_key, learner))
            if self._budget is not None:
                release = getattr(self._budget, "release", None)
                if release is not None:
                    release()

    async def _run_pipeline(self, chat_key: ChatKey, spec: LearnerSpec) -> None:
        """One generic pipeline run (no provider retry — a failure settles
        the run and the next scheduled run picks the source up again)."""
        result = await self._pipeline.run(chat_key, spec)
        self._record_notice(result)

    async def _run_effect(self, chat_key: ChatKey, spec: LearnerSpec) -> None:
        """The effect learner: runs ONLY when pending references exist,
        delivery was confirmed, and a human follow-up (a message AFTER the
        reference boundary) is pending beyond the effect watermark. The
        judgment's categorization/confidence is derived by code and applied
        once per referenced record."""
        refs = self._effect_refs.get(chat_key, [])
        dispatch_id = self._effect_dispatch.get(chat_key)
        delivered = (
            (dispatch_id is not None and (chat_key, dispatch_id) in self._delivered_dispatches)
            or (dispatch_id is None and chat_key in self._delivered)
        )
        if not refs or not delivered:
            return  # not eligible: zero provider calls
        boundary = self._effect_boundary.get(chat_key)
        if boundary is None:
            return
        if not await self._has_followup(chat_key, boundary):
            return  # no human follow-up yet
        references = render_records(refs)
        result = await self._pipeline.run(chat_key, spec, references=references)
        self._record_notice(result)
        if result.outcome == "success" and result.records_added >= 1:
            await self._apply_effect(chat_key, refs, result.run_id)
            self._effect_refs.pop(chat_key, None)
            self._effect_boundary.pop(chat_key, None)
            if dispatch_id is not None:
                self._delivered_dispatches.discard((chat_key, dispatch_id))
            self._effect_dispatch.pop(chat_key, None)
            self._delivered.discard(chat_key)

    async def _has_followup(
        self, chat_key: ChatKey, boundary: MessageRowId
    ) -> bool:
        """True when the effect learner has source beyond its durable
        watermark that includes a message AFTER the reference boundary (a
        human follow-up). The read is SQL-bounded (``tail`` caps the rows)."""
        try:
            batch = await self._repo.read_learner_source_batch(
                chat_key,
                "effect",
                through_msg_id=MessageRowId(self._MAX_ROW_ID),
                tail=self._EFFECT_SCAN_TAIL,
                # The delivery's self echo is not a human follow-up.  Only a
                # non-self source row strictly after the selected dispatch
                # boundary can authorize the effect run.
                policy="nonself",
            )
        except Exception:
            return False
        if batch is None:
            return False
        return batch.last_msg_id > boundary

    async def _apply_effect(
        self, chat_key: ChatKey, refs: Sequence[Record], effect_run_id: int | None
    ) -> None:
        """Derive the code-owned delta from the newest effect record and
        apply it ONCE per referenced record (bounded by the repository's
        [0.1, 5] clamp)."""
        try:
            if effect_run_id is None:
                return
            latest = getattr(self._repo, "latest_record_for_run", None)
            if latest is not None:
                record = await latest(chat_key, "effect", effect_run_id)
                records = [record] if record is not None else []
            else:
                records = await self._repo.list_records_for_run(
                    chat_key, "effect", effect_run_id, limit=100
                )
        except Exception:
            return
        if not records:
            return
        payload = records[-1].payload or {}  # the newest effect record
        categorization = payload.get("categorization")
        confidence = payload.get("confidence")
        if categorization not in EFFECT_CATEGORIZATIONS:
            return
        if isinstance(confidence, bool) or not isinstance(
            confidence, (int, float)
        ):
            return
        try:
            delta = derive_effect_delta(categorization, float(confidence))
        except (ValueError, TypeError):
            return
        for rec in refs:
            if rec.id is None:
                continue
            try:
                await self._repo.apply_record_feedback(
                    chat_key, rec.learner, rec.id, delta,
                    now=self._clock.now(), effect_run_id=effect_run_id,
                )
            except Exception:
                log.warning(
                    "effect feedback failed for %s/%s (contained)",
                    chat_key,
                    rec.learner,
                    exc_info=True,
                )

    # ── helpers ─────────────────────────────────────────────────────────────

    def _enqueue(self, chat_key: ChatKey, learner: str) -> None:
        if (chat_key, learner) in self._queued or (
            chat_key,
            learner,
        ) in self._in_flight:
            return
        try:
            self._queue.put_nowait((chat_key, learner))
        except asyncio.QueueFull:
            # Queue overflow is advisory only: the periodic scan supplies
            # eventual coverage for dropped enqueues.
            log.warning(
                "learner queue full; coalescing %s/%s", chat_key, learner
            )
            return
        self._queued.add((chat_key, learner))

    def _record_notice(self, result: Any) -> None:
        """Convert one run result into a bounded ``SettlementNotice``."""
        outcome = result.outcome
        if outcome in ("success", "skipped", "stale"):
            notice_outcome = "success"
        elif outcome in ("malformed", "provider_error", "prompt_error"):
            notice_outcome = "malformed"
        else:
            notice_outcome = "cancelled"
        self._notices.append(
            SettlementNotice(
                learner=result.learner,
                chat_key=result.chat_key,
                run_id=result.run_id or 0,
                outcome=notice_outcome,
                records_added=result.records_added,
                records_merged=result.records_merged,
                watermark=result.watermark,
                error=result.error,
                ts=self._clock.now(),
            )
        )
        if len(self._notices) > self._NOTICE_LIMIT:
            del self._notices[: len(self._notices) - self._NOTICE_LIMIT]


class App:
    """The composed runtime. ``build`` wires the default console-only
    stack; tests may inject any component."""

    def __init__(
        self,
        cfg: Config | None = None,
        *,
        clock: Any = None,
        db: Database | None = None,
        repo: SqliteRepository | None = None,
        recorder: Recorder | None = None,
        adapter: Adapter | None = None,
        ingest: Ingest | None = None,
        outbox: OutboxDriver | None = None,
        scheduler: Scheduler | None = None,
        cycle: CycleRunner | None = None,
        hooks: HookBus | None = None,
        dry_run: bool = False,
        trace_sink: Any = None,
        gate: Gate | None = None,
        agent: PhaseAgent | None = None,
        planner: Any = None,
        replyer: Any = None,
        budget: Any = None,
        llm: Any = None,
        memory_search: MemorySearch | None = None,
        person_service: PersonService | None = None,
        memory_service: MemoryService | None = None,
        embed_service: OptionalEmbeddingService | None = None,
        semantic_backfill: Any = None,
        backfill_budget: Any = None,
        embed_llm: Any = None,
        learner: Any = None,
        adaptive: Any = None,
        learn_llm: Any = None,
        media_harvester: MediaHarvester | None = None,
        media_llm: Any = None,
        output_pipeline: Callable[[Any], Any] | None = None,
        plugin_manifest: tuple[Any, ...] = (),
    ) -> None:
        self.cfg = cfg if cfg is not None else Config()
        self.clock = clock if clock is not None else RealClock()
        self.db = db
        self.repo = repo
        self.recorder = recorder
        self.adapter = adapter
        self.ingest = ingest
        self.outbox = outbox
        self.dry_run = dry_run
        self.plugin_manifest = tuple(plugin_manifest)
        self.hooks = hooks if hooks is not None else HookBus()
        # Phase 5 knowledge stack (owned by the default build): a shared
        # local MemorySearch (FTS-only when no optional embed is configured),
        # a PersonService over the SqliteRepository/KnowledgeRepository, and
        # a MemoryService with the deterministic local capsule summarizer
        # (no provider/network/LLM).
        self.memory_search = memory_search
        self.person_service = person_service
        self.memory_service = memory_service
        # The optional embedding service (Gate 5): a real
        # ``OptionalEmbeddingService`` when ``llm.profiles.embed`` carries an
        # explicit revision, else None (FTS-only, ZERO embed calls). The
        # owned embed LLM client (built by ``build``) is closed on shutdown.
        self._embed_service = embed_service
        self._embed_llm = embed_llm
        # The bounded semantic backfill worker and its per-chat budget
        # manager (built by ``build`` when an embed service exists); the
        # background task is scheduled at startup and cancelled/drained in
        # shutdown before the shared LLM/DB close.
        self._semantic_backfill = semantic_backfill
        self._backfill_budget = backfill_budget
        self._semantic_task: asyncio.Task[None] | None = None
        # Phase 6 P6.4 learner wiring: the bounded cancellable learner
        # worker (built by ``build`` when learn is enabled with a valid
        # profile), the frozen per-dispatch adaptive context service, and the
        # owned learn LLM client (closed on shutdown). The worker task is
        # scheduled at startup ONLY in LIVE mode and cancelled/drained in
        # shutdown before the shared LLM/DB close.
        self._learner = learner
        self._adaptive = (
            _EscapedAdaptiveContext(adaptive) if adaptive is not None else None
        )
        self._learn_llm = learn_llm
        self._learner_task: asyncio.Task[None] | None = None
        # Phase 6 P6.5b media harvest lane: the bounded/cancellable/advisory
        # MediaHarvester (built by ``build`` when the media catalog is
        # enabled). Its in-flight tasks are cancelled in shutdown before the
        # shared LLM/DB close.
        self._media_harvester = media_harvester
        # The owned media vision LLM client (built by ``build`` when the
        # media catalog has a vision_profile); closed on shutdown.
        self._media_llm = media_llm
        # The owned LLM client (built by ``build`` when the agent is
        # configured); closed on shutdown. Injected seams retain precedence.
        self._llm = llm
        # The Phase 3 agent coordinator: an injected PhaseAgent, or one built
        # from injected planner/replyer/budget. Never constructed from a real
        # network client — the injected seams are the only LLM surface.
        self._agent = agent
        if self._agent is None and (planner is not None or replyer is not None):
            if planner is None or replyer is None:
                raise ValueError("planner and replyer must both be injected")
            self._agent = PhaseAgent(planner, replyer, budget)
        # The scheduler's cycle fn: an injected CycleRunner, or one built
        # from the repository seam (the scheduler itself is rebuilt on a
        # restart, since Scheduler.stop() is terminal). The runner's
        # at-least-once dispatch marker exporter is wired to
        # ``record.export_marker`` for the ledger dry-run lane.
        self._cycle_fn: Any = cycle
        if self._cycle_fn is None and repo is not None:
            self._cycle_fn = CycleRunner(
                repo,
                gate if gate is not None else Gate(),
                self.cfg,
                clock=self.clock,
                hooks=self.hooks,
                dry_run=dry_run,
                trace_sink=trace_sink,
                marker_exporter=(
                    (lambda marker: export_marker(recorder, repo, marker))
                    if recorder is not None
                    else None
                ),
                agent=self._agent,
                on_outbox=(self._wake_outbox if not dry_run else None),
                on_memory=self._on_memory_default,
                adaptive=cast(Any, self._adaptive),
                on_settled=(self._on_settled if not dry_run else None),
                on_exposure=(self._on_exposure if not dry_run else None),
                on_chat_control=(self._wake_control_target if not dry_run else None),
                output_pipeline=output_pipeline,
            )
        self.scheduler = scheduler
        if self.scheduler is None and self._cycle_fn is not None:
            if (dry_run or self._agent is not None) and cycle is None:
                # Default production dry-run AND the live agent lane: the
                # durable dispatch-ledger scheduler over Repository + Clock +
                # CycleRunner.run_dispatch. An explicitly injected generic
                # Scheduler (or cycle) keeps the legacy wake path.
                self.scheduler = LedgerScheduler(
                    cast(Repository, self.repo),
                    self.clock,
                    self._cycle_fn.run_dispatch,
                    dispatch_lease_s=self.cfg.agent.dispatch_lease_s,
                    lease_for_chat=lambda chat_key: self.cfg.for_chat(
                        chat_key
                    ).agent.dispatch_lease_s,
                )
            else:
                self.scheduler = Scheduler(self.clock, self._cycle_fn)
        self._scheduler_started = False
        self._scheduler_ever_started = False
        self._started = False
        self._shutdown = False
        self._worker: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        # The background adapter-event receiver (live mode): consumes
        # ``adapter.events()`` and ingests every event. It is started BEFORE
        # the outbox worker so echo responses resolve before any send, and
        # ``_receiver_active`` signals that it is actively consuming.
        self._receiver: asyncio.Task[None] | None = None
        self._receiver_active = asyncio.Event()
        # Chats with newly settled outbox rows, collected by the live
        # ``on_outbox`` wake callback so the worker drains exactly the chats
        # that produced output (the OneBot adapter serves many chats; the
        # console adapter always drains its single chat).
        self._drain_chats: set[ChatKey] = set()
        # Chats with durable pending outbox rows that the worker retains
        # ACROSS rounds (startup recovery + future-paced rows): after a
        # future ``send_after_ts`` sleep naturally expires, the worker
        # rechecks/drains these chats without requiring a new wake.
        self._active_chats: set[ChatKey] = set()
        # Exposure intents remain pending until the exact dispatch's outbox
        # delivery is confirmed.
        self._pending_exposures: dict[
            tuple[ChatKey, int], tuple[tuple[Record, ...], MessageRowId]
        ] = {}
        self._outbox_dispatches: dict[ChatKey, set[int]] = {}
        self._confirmed_dispatches: set[tuple[ChatKey, int]] = set()
        # The LEGACY ingestion wake boundary (explicitly injected generic
        # Scheduler only): chat_key -> wake kind ("priority" or "ordinary")
        # for commits that have NOT yet been flushed to the scheduler. Raw
        # AdapterEvents are never retained here — every event is committed
        # through Ingest immediately in _ingest_batched; only the post-commit
        # wake metadata accumulates, and exactly one flush task is scheduled
        # for the next event-loop turn. The ledger route never uses this.
        self._wake_meta: dict[ChatKey, str] = {}
        self._flush_task: asyncio.Task[None] | None = None

    @classmethod
    def build(
        cls,
        cfg: Config | None = None,
        *,
        clock: Any = None,
        adapter: Adapter | None = None,
        recorder_path: str | Path | None = None,
        dry_run: bool = False,
        trace_sink: Any = None,
        gate: Gate | None = None,
        scheduler: Scheduler | None = None,
        cycle: CycleRunner | None = None,
        agent: PhaseAgent | None = None,
        planner: Any = None,
        replyer: Any = None,
        budget: Any = None,
        llm: Any = None,
        embed_service: OptionalEmbeddingService | None = None,
    ) -> App:
        """Wire the default stack: Database -> SqliteRepository -> Recorder
        -> ConsoleAdapter -> Ingest -> OutboxDriver -> Scheduler/CycleRunner.

        Phase 2 rejects non-console adapters: the gate/cycle that would
        gate real sends exists only in dry-run, so no other adapter may be
        configured. A Phase 3 agent is built when configured (injected
        ``agent`` wins; else injected ``planner``/``replyer``/``budget`` are
        wrapped; else a default agent is built from the config when the
        ``planner``/``reply`` LLM profiles exist — owning an OpenAIClient,
        PromptStore, core registry, BudgetManager and PhaseAgent). The
        default build stays no-agent when no profiles are configured.
        """
        cfg = cfg if cfg is not None else Config()
        clock = clock if clock is not None else RealClock()
        if adapter is not None:
            name = getattr(adapter, "name", None)
            if name not in SUPPORTED_ADAPTERS:
                raise ConfigError(
                    f"unsupported adapter {name!r}; supported:"
                    f" {', '.join(sorted(SUPPORTED_ADAPTERS))}"
                )
        # Phase 6 P6.6 explicit-trust plugins: build the core staging
        # registries seeded with the built-ins, load the configured plugins
        # deterministically (paths then entry points — no auto-discovery, no
        # hot reload), and freeze. Any resolve/import/setup/validation
        # failure raises HERE — before the adapter/network/DB worker start —
        # and leaves no usable partial registry.
        plugin_result: PluginLoadResult | None = None
        configured_manifest: tuple[Any, ...] = ()
        if cfg.plugins.paths or cfg.plugins.entry_points:
            if dry_run:
                # Dry-run (and replay, which uses the same declarative helper)
                # records/checks identity without importing trusted code.
                configured_manifest = configured_plugin_manifest(cfg)
            else:
                plugin_result = PluginLoader(cfg).load()
                configured_manifest = plugin_result.plugin_manifest
        if plugin_result is not None:
            gate = gate if gate is not None else _gate_from_plugins(plugin_result)
            hooks = plugin_result.hooks
            core_registry = _core_registry_from_plugins(plugin_result)
            output_factory = _output_factory_from_plugins(plugin_result)
        else:
            hooks = HookBus(timeout_s=cfg.plugins.hook_timeout_s)
            core_registry = None
            output_factory = None
        if gate is None:
            gate = Gate()
        if configured_manifest and not hasattr(gate, "_plugin_manifest"):
            setattr(gate, "_plugin_manifest", configured_manifest)
        db = Database(cfg.storage.db_path)
        repo = SqliteRepository(db)
        recorder = Recorder(
            recorder_path
            if recorder_path is not None
            else Path(cfg.storage.db_path).with_suffix(".jsonl")
        )
        if adapter is None:
            if cfg.adapter.name == "onebot":
                adapter = OneBotAdapter(config=cfg.adapter.onebot, clock=clock)
            else:
                adapter = ConsoleAdapter(clock=clock)
        # Strict dry-run restriction: dry-run is console-only — even a
        # CONFIG-SELECTED OneBot is rejected, so dry-run never connects to or
        # consumes OneBot traffic.
        if dry_run and getattr(adapter, "name", None) != "console":
            raise ConfigError(
                "dry-run supports only the console adapter;"
                f" got {getattr(adapter, 'name', None)!r}"
            )
        # Phase 6 P6.5b media runtime: a shared MediaStore (the adapter's own
        # normalization store when it has one, so the harvest cache and the
        # send-time resolver share ONE content-addressed cache) and, when the
        # media catalog is enabled, a send-time MediaResolvingAdapter wrapper
        # that maps opaque cache keys to bytes IN MEMORY at send time — the
        # durable outbox never carries a URL, file path, or base64 payload.
        media_store = getattr(adapter, "_media", None) or MediaStore()
        if cfg.media.enabled:
            adapter = MediaResolvingAdapter(adapter, media_store, repo=repo)
        outbox = OutboxDriver(repo, adapter, clock=clock)
        # Trusted delivery-key resolver: wired only when the adapter
        # exposes it (the console adapter does not; OneBot does). Missing/
        # untrusted keys stay ``unproven``.
        delivery_key = cast(
            DeliveryKeyFn | None, getattr(adapter, "delivery_key_for", None)
        )
        # Phase 5 knowledge stack: a shared local MemorySearch (FTS-only when
        # no optional embed is configured), a PersonService over the
        # SqliteRepository/KnowledgeRepository, and a MemoryService with the
        # deterministic local capsule summarizer (no provider/network/LLM).
        # The PersonService also drives post-ingest person observation.
        #
        # Optional embedding service (Gate 5): a configured
        # ``llm.profiles.embed`` with an explicit revision becomes a real
        # ``OptionalEmbeddingService`` over an OpenAIClient embedder, bound to
        # the canonical ``space_id`` (model@revision) with a shared SHA1
        # cache. Absent profile/revision -> FTS-only (embed=None, ZERO embed
        # calls). An injected ``embed_service`` retains precedence.
        embed_profile = cfg.llm.profiles.get("embed")
        embed_llm: Any = None
        if embed_service is None and embed_profile is not None and embed_profile.revision:
            embed_llm = OpenAIClient(cfg.llm, clock=clock)
            embed_space = embed_profile.space_id()
            assert embed_space is not None  # revision is set above
            embed_service = OptionalEmbeddingService(
                embed_llm,
                cache=EmbeddingCache(path=_embed_cache_path(cfg)),
                space_id=embed_space,
            )
        # The bounded semantic backfill worker + its per-chat budget manager
        # (built when an embed service exists). The worker runs OUTSIDE any
        # terminal settlement transaction and is scheduled at startup.
        backfill_budget = BudgetManager(repo, cfg.budget, now=clock.now)
        semantic_backfill: Any = None
        if (
            embed_service is not None
            and embed_profile is not None
            and embed_profile.revision
        ):
            semantic_backfill = SemanticBackfill(
                repo,
                embed_service,
                backfill_budget,
                model=embed_profile.model,
                revision=embed_profile.revision,
                space_id=embed_service.space_id,
                # Per-chat budget resolution: the worker honors each chat's
                # effective ``cfg.for_chat(chat).budget`` — the SAME physical
                # KV state/config the planner's chat-bound BudgetManager uses,
                # so planner and embed share atomic budget state.
                cfg=cfg,
                now=clock.now,
            )
        memory_search = MemorySearch(
            repo,
            embed=embed_service,
            # Semantic QUERY embeds reserve exactly one call under the chat's
            # effective budget (``cfg.for_chat(chat).budget``) — the SAME
            # physical KV state/config the planner's chat-bound BudgetManager
            # uses, so planner and embed share atomic budget state.
            budget_for=_budget_resolver(repo, cfg, clock),
        )
        person_service = PersonService(repo)
        memory_service = MemoryService(
            repo, search=memory_search, summarizer=default_capsule_summarizer()
        )
        # Phase 6 P6.4/6 P6.5b shared background budget: the learner worker
        # and the media harvest vision lane share ONE LearnerBudget over the
        # chat budget (the SAME physical KV state the foreground planner
        # uses, bounded to ``concurrency - foreground_reserve`` concurrent
        # runs). Built unconditionally — it is cheap and the learner worker
        # reuses it below.
        learner_budget = LearnerBudget(
            BudgetManager(repo, cfg.budget, now=clock.now),
            concurrency=cfg.learn.concurrency,
            foreground_reserve=cfg.learn.foreground_reserve,
            # Keep the same foreground reserve at the daily-cap boundary as
            # at the semaphore boundary.  This is intentionally derived from
            # the existing config knob; no new config surface is required.
            daily_capacity_reserve=cfg.learn.foreground_reserve,
            budget_for=_budget_resolver(repo, cfg, clock),
        )
        # Phase 6 P6.5b media harvest lane: the bounded/cancellable/advisory
        # MediaHarvester, built when the media catalog is enabled. Its vision
        # approval is budget-admitted through the shared learner budget when
        # a vision_profile is configured; a missing profile/blocked budget/
        # provider failure/malformed description leaves the candidate
        # PENDING (unapproved). LIVE-only: the App schedules harvests only
        # in live mode (never dry-run/replay/doctor).
        media_harvester: MediaHarvester | None = None
        media_llm: Any = None
        if cfg.media.enabled:
            if cfg.media.vision_profile is not None:
                media_llm = OpenAIClient(cfg.llm, clock=clock)
            media_harvester = MediaHarvester(
                repo,
                media_store,
                cfg=cfg.media,
                clock=clock,
                llm=media_llm,
                budget=learner_budget,
            )
        # The wake is issued by the run loop AFTER handle() returns
        # (post-commit), filtered to newly inserted non-self messages —
        # see _maybe_wake. The identity resolver handles both the console
        # adapter (one fixed chat) and the OneBot adapter (per-chat).
        ingest = Ingest(
            repo,
            recorder,
            wake=None,
            identity=lambda chat_key: _adapter_identity(adapter, chat_key),
            clock=clock,
            delivery_key=delivery_key,
            observe_person=(
                lambda msg: person_service.observe(
                    msg.chat_key, msg.sender_id, msg.sender_name, now=clock.now()
                )
            ),
            harvest_media=(
                (
                    lambda msg, row_id: _schedule_harvest(
                        media_harvester, msg, row_id
                    )
                )
                if media_harvester is not None
                else None
            ),
        )
        # Build the Phase 3 agent when configured (injected seams retain
        # precedence): an injected agent wins; else injected
        # planner/replyer/budget are wrapped; else a default agent is built
        # from the config when the planner/reply profiles exist.
        prompts = PromptStore(cfg.bot.prompt_dir)
        if agent is None and (planner is not None or replyer is not None):
            if planner is None or replyer is None:
                raise ValueError("planner and replyer must both be injected")
            agent = PhaseAgent(planner, replyer, budget)
        elif agent is None and _agent_configured(cfg):
            llm = OpenAIClient(cfg.llm, clock=clock)
            registry = (
                core_registry if core_registry is not None else register_core_tools()
            )
            budget_mgr = BudgetManager(repo, cfg.budget, now=clock.now)
            agent = PhaseAgent.budgeted(
                llm, prompts, registry, cfg.context, budget_mgr, cfg.agent,
                cfg=cfg, repo=repo, now=clock.now,
                memory_search=memory_search, person_service=person_service,
                # Real budgeted ToolContext: inject the adapter-supported
                # capabilities and a safely chat-scoped forward resolver so
                # fetch_history / view_forward_message are reachable in
                # production when the adapter provides the data, and fail
                # closed otherwise.
                capabilities=getattr(adapter, "capabilities", frozenset()),
                forward_resolver=lambda chat_key: _adapter_forwards(
                    adapter, chat_key
                ),
                jargon_query=_jargon_query(repo),
                # Phase 6 P6.5b media wiring: the chat-bound catalog
                # callbacks the deferred send_emoji / send_image tools speak
                # to (None when the catalog is disabled — they fail closed).
                media_callbacks=_media_callbacks(repo, cfg),
                # Phase 6 P6.6b chat-control wiring: the chat-bound
                # callbacks the deferred set_focus / notify_chat tools speak
                # to (None when the repo lacks the surface — they fail
                # closed).
                chat_control_callbacks=_chat_control_callbacks(repo),
            )
        # Phase 6 P6.4 learner wiring: build the bounded cancellable learner
        # worker when learn is enabled with at least one valid profile (a
        # known spec whose prompt file loads). The worker feeds the generic
        # LearnerPipeline with a shared per-chat LearnerBudget (the SAME
        # physical budget state the foreground planner uses, bounded to
        # ``concurrency - foreground_reserve`` concurrent runs) and is
        # started ONLY in LIVE mode. The adaptive context service is built
        # alongside it (queried only after the gate triggers, in live
        # dispatches only).
        learner: Any = None
        adaptive: Any = None
        learn_llm: Any = None
        if cfg.learn.enabled:
            base_specs = (
                {spec.name: spec for spec in plugin_result.learners.all()}
                if plugin_result is not None
                else SPECS
            )
            # A plugin may declaratively replace/retune a known learner. A
            # brand-new name still needs a validator, which is intentionally
            # not a runtime surface on PluginAPI and is therefore skipped.
            specs = {
                name: spec
                for name, spec in _build_learner_specs(
                    cfg, prompts, base_specs
                ).items()
                if name in VALIDATORS
            }
            if specs:
                learn_llm = OpenAIClient(cfg.llm, clock=clock)
                # The shared LearnerBudget built above (the media harvest
                # vision lane uses the same physical budget state).
                pipeline = LearnerPipeline(
                    repo,
                    learn_llm,
                    prompts,
                    clock,
                    validators=VALIDATORS,
                    budget=cast(Any, learner_budget),
                )
                learner = LearnerScheduler(
                    repo,
                    pipeline,
                    specs,
                    clock,
                    scan_interval_s=float(cfg.learn.cadence_s),
                    budget=learner_budget,
                )
                adaptive = AdaptiveContextService(
                    repo,
                    now=clock.now,
                    embed=embed_service,
                    model=embed_profile.model if embed_profile is not None else None,
                    budget_for=_budget_resolver(repo, cfg, clock),
                )
        return cls(
            cfg,
            clock=clock,
            db=db,
            repo=repo,
            recorder=recorder,
            adapter=adapter,
            ingest=ingest,
            outbox=outbox,
            scheduler=scheduler,
            cycle=cycle,
            hooks=hooks,
            dry_run=dry_run,
            trace_sink=trace_sink,
            gate=gate,
            agent=agent,
            planner=planner,
            replyer=replyer,
            budget=budget,
            llm=llm,
            memory_search=memory_search,
            person_service=person_service,
            memory_service=memory_service,
            embed_service=embed_service,
            semantic_backfill=semantic_backfill,
            backfill_budget=backfill_budget,
            embed_llm=embed_llm,
            learner=learner,
            adaptive=adaptive,
            learn_llm=learn_llm,
            media_harvester=media_harvester,
            media_llm=media_llm,
            output_pipeline=output_factory,
            plugin_manifest=configured_manifest,
        )

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Open the database (schema applied on first boot). Idempotent."""
        if self._started:
            return
        assert self.db is not None
        await self.db.open()
        self._started = True
        # Persist the startup identity in the same durable KV owner as the
        # runtime.  This is metadata only: no plugin module is imported here,
        # which keeps dry-run/replay startup safe.
        manifest_payload = [
            m.as_dict() if hasattr(m, "as_dict") else dict(m)
            for m in self.plugin_manifest
        ]
        if self.repo is not None:
            await self.repo.set_kv(
                "plugin_manifest",
                json.dumps(
                    manifest_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        if self.recorder is not None:
            # This is an ordinary metadata line, not a dispatch marker, so
            # old marker readers remain compatible.  It is intentionally
            # written before any input is accepted.
            self.recorder.write(
                {"type": "plugin_manifest", "manifest": manifest_payload}
            )
        # Local, idempotent memory maintenance at DB start: bootstrap the
        # canonical memory FTS index for chats whose docs are missing, and
        # repair crash-after-settlement gaps. Contained — never blocks.
        await self._memory_maintenance()
        # Schedule/resume the bounded semantic backfill AFTER the local
        # FTS/capsule bootstrap, as a background task (never blocks adapter
        # startup or dry-run sends); cancelled/drained in shutdown.
        await self._start_semantic_backfill()
        # Schedule/resume the bounded learner worker (LIVE-only; never in
        # dry_run/replay/doctor); cancelled/drained in shutdown before the
        # shared LLM/DB close.
        await self._start_learner()

    async def _start_learner(self) -> None:
        """Schedule/resume the bounded learner worker as a background task.
        LIVE-only: the worker never runs in dry_run/replay/doctor. Idempotent:
        a running task is never duplicated."""
        if self._learner is None:
            return
        if self.dry_run:
            return  # LIVE-only
        if self._learner_task is not None and not self._learner_task.done():
            return  # already running
        self._learner_task = asyncio.create_task(self._learner.run())

    async def _start_semantic_backfill(self) -> None:
        """Schedule/resume the semantic backfill worker as a background task.
        Idempotent: a running task is never duplicated."""
        if self._semantic_backfill is None:
            return
        if self._semantic_task is not None and not self._semantic_task.done():
            return  # already running
        self._semantic_task = asyncio.create_task(self._semantic_backfill.run())

    async def run(self) -> None:
        """Start the scheduler, serve the console adapter, and feed every
        committed event to the scheduler.

        Dry-run: the outbox driver is never started or drained and
        ``adapter.send`` is never invoked — even with pre-existing pending
        rows. The default ledger route submits every durable commit to
        ``LedgerScheduler.notify_commit`` and recovers unassigned commits at
        startup; an explicitly injected generic Scheduler keeps the legacy
        next-turn-flush wake path.

        Live (non-dry-run): the adapter is connected and its readiness
        awaited first (no worker send while disconnected), then pre-existing
        safe pending/future outbox rows are recovered across ALL chats. With
        an agent, the ledger scheduler runs the agent lane and the outbox
        worker is additionally woken after a successful terminal agent output
        creates outbox rows (the CycleRunner's ``on_outbox`` callback).
        Without an agent (legacy Phase 1), the same cross-chat startup
        recovery runs and the worker paces future rows.
        """
        await self.start()
        assert self.adapter is not None and self.ingest is not None
        if self.dry_run:
            try:
                await self._start_scheduler()
                await self.adapter.connect()
                async for event in self.adapter.events():
                    await self._ingest_batched(event)
                if not isinstance(self.scheduler, LedgerScheduler):
                    # Stream end: flush any scheduled wake task so committed
                    # wake metadata is not lost (the ledger route has no
                    # flush; the finally's shutdown flush is then a no-op).
                    await self._flush_pending_wakes()
            finally:
                await self.shutdown()
        else:
            assert self.repo is not None and self.outbox is not None
            try:
                # Live: connect/handshake readiness FIRST — no worker may send
                # while the adapter is disconnected. Then start the background
                # receiver so echo responses resolve before any send (a startup
                # action's echo succeeds only because the receiver is active).
                await self.adapter.connect()
                await self._wait_adapter_ready()
                self._receiver = asyncio.create_task(self._receive_loop())
                await self._receiver_active.wait()
                # Startup recovery: drain pre-existing SAFE pending/future outbox
                # rows across ALL chats (in-flight rows are never touched), and
                # retain those chats as active so the worker paces the future
                # rows without a new wake. Each chat is drained with BOUNDED,
                # readiness-checked pumps so a mid-drain disconnect never causes
                # a premature in_flight transition.
                outbox_chats = await self.repo.list_outbox_chats()
                startup_rounds = (
                    None if getattr(self.adapter, "chat_key", None) else 1
                )
                for chat in outbox_chats:
                    await self._drain_chat_safely(chat, rounds=startup_rounds)
                self._active_chats = set(outbox_chats)
                if self._agent is not None:
                    # Live agent lane: start the ledger scheduler and the outbox
                    # worker. Pre-existing rows were drained above; the worker is
                    # additionally woken after a successful terminal agent output
                    # creates outbox rows (the CycleRunner's on_outbox callback).
                    await self._start_scheduler()
                    self._worker = asyncio.create_task(self._outbox_worker())
                else:
                    # Legacy no-agent lane: outbox worker, no scheduler.
                    self._worker = asyncio.create_task(self._outbox_worker())
                await self._receiver
            finally:
                await self.shutdown()

    async def _receive_loop(self) -> None:
        """Consume every adapter event and ingest it (the live run loop's
        event feed). Signals ``_receiver_active`` as soon as it starts, so
        the outbox worker never sends before echo responses can resolve."""
        assert self.adapter is not None
        self._receiver_active.set()
        async for event in self.adapter.events():
            await self._ingest_batched(event)

    async def _drain_chat_safely(
        self, chat: ChatKey, *, rounds: int | None = 1
    ) -> None:
        """Drain every due row for one chat with BOUNDED pumps, checking
        adapter readiness before each pump so no premature in_flight
        transition happens while the adapter is disconnected (recover safely
        after reconnect). Each pump uses a fresh clock."""
        assert self.outbox is not None
        remaining = rounds
        while remaining is None or remaining > 0:
            if not self._adapter_ready():
                await self._wait_adapter_ready()
            sent = await self.outbox.pump(chat, limit=10)
            if sent < 10:
                return
            if remaining is not None:
                remaining -= 1

    async def _start_scheduler(self) -> None:
        """Start the scheduler (rebuilding it for a restart) and run the
        startup recovery: ledger marker export + unassigned-commit recovery
        for the LedgerScheduler, or pending-chat wakes for the legacy
        Scheduler."""
        if self.scheduler is None:
            return
        if self._scheduler_ever_started:
            # Scheduler.stop() is terminal: rebuild for a restart.
            self.scheduler = self._rebuild_scheduler()
        self.scheduler.start()
        self._scheduler_started = True
        self._scheduler_ever_started = True
        assert self.repo is not None
        if isinstance(self.scheduler, LedgerScheduler):
            # Ledger startup: repair marker crash gaps (at-least-once
            # export), then resume every chat with an eligible unassigned
            # commit. A crash/restart during an ordinary delay re-evaluates
            # the pending messages; a durable active hold schedules only its
            # remaining time (the gate's backoff delay re-arms the scheduler
            # at hold expiry); a durable wait/retry barrier re-arms at its
            # remaining resume_at.
            assert self.recorder is not None
            await export_unexported(self.recorder, self.repo)
            await self.scheduler.recover(
                await self.repo.list_ledger_pending_chats()
            )
        else:
            # Legacy startup: re-evaluate every chat with durable pending
            # work.
            for chat_key in await self.repo.list_pending_chats():
                await self.scheduler.wake(chat_key)

    def _rebuild_scheduler(self) -> Any:
        """Scheduler.stop() is terminal: rebuild the same kind for a
        restart. The ledger lane rebuilds a ``LedgerScheduler`` over the
        same repository/clock/``run_dispatch`` handler; the legacy lane
        rebuilds a generic ``Scheduler`` over the same cycle fn."""
        if isinstance(self.scheduler, LedgerScheduler):
            assert self.repo is not None
            return LedgerScheduler(
                cast(Repository, self.repo),
                self.clock,
                self._cycle_fn.run_dispatch,
                dispatch_lease_s=self.cfg.agent.dispatch_lease_s,
                lease_for_chat=lambda chat_key: self.cfg.for_chat(
                    chat_key
                ).agent.dispatch_lease_s,
            )
        return Scheduler(self.clock, self._cycle_fn)

    async def _maybe_wake(self, event: AdapterEvent, result: IngestResult) -> None:
        """The typed wake rule for ONE committed member (the single-event
        immediate-wake path used by tests with an injected generic
        Scheduler; the ledger route notifies through ``notify_commit``
        instead). Post-commit by construction: ``ingest.handle`` returned
        after the durable commit."""
        if isinstance(self.scheduler, LedgerScheduler):
            return  # the ledger route owns its own notification
        kind = self._wake_kind(event, result)
        if kind is None or self.scheduler is None:
            return
        if kind == "priority":
            await self.scheduler.wake_priority(event.payload.chat_key)
        else:
            await self.scheduler.wake(event.payload.chat_key)

    def _wake_kind(self, event: AdapterEvent, result: IngestResult) -> str | None:
        """The wake kind for ONE committed member: None (no wake),
        ``"priority"``, or ``"ordinary"``.

        Only a newly inserted NON-SELF message wakes: duplicates
        (``inserted=False``) and self echoes (any ``echo_status``) never
        do. Priority: a structurally recognized direct @/quote (the
        message mentions the chat's self id or carries a reply target),
        OR the atomic pending count at/above the chat's gate threshold
        (high pending may re-evaluate during a scheduled delay/hold).
        Ordinary members stay ordinary — they can never override an
        existing scheduled delay."""
        if self.scheduler is None or not result.inserted:
            return None
        payload = event.payload
        if isinstance(payload, Message) and not payload.is_self:
            if self._is_priority_wake(payload):
                return "priority"
            if (
                result.pending_count is not None
                and result.pending_count
                >= self.cfg.for_chat(payload.chat_key).gate.threshold
            ):
                return "priority"
            return "ordinary"
        return None

    async def _ingest_batched(self, event: AdapterEvent) -> None:
        """Commit EVERY adapter event through Ingest immediately (recorder
        + database durability before anything else).

        Ledger route (default dry-run): every ``IngestResult`` with a
        durable commit sequence is submitted IMMEDIATELY to
        ``LedgerScheduler.notify_commit(chat_key, commit_seq)`` — the App
        performs no scheduler timer/wake arbitration and retains no raw
        events/metadata. Non-message/self/duplicate results with no commit
        remain no-op (a self echo commits with ``wake_kind`` ``none`` and
        is never attached; ``begin_dispatch`` returns None when there is
        no eligible work).

        Legacy route (explicitly injected generic Scheduler): accumulate
        only the post-commit wake metadata (``chat_key ->
        ordinary/priority``) and schedule exactly ONE flush task for the
        next event-loop turn. The flush task runs on the next event-loop
        turn, so every commit completed BEFORE that turn is coalesced into
        one wake per chat (priority if ANY member qualifies, else
        ordinary); commits that land AFTER the flush already ran go to a
        later flush. The flush never sleeps on timestamps and never waits
        for another input or EOF — it is purely next-turn. No raw
        AdapterEvent is retained after this call.
        """
        if self.ingest is None:
            return
        payload = event.payload
        structural_priority = (
            self._is_priority_wake(payload)
            if isinstance(payload, Message)
            else False
        )
        pending_threshold = (
            self.cfg.for_chat(payload.chat_key).gate.threshold
            if isinstance(payload, Message)
            else None
        )
        if isinstance(self.scheduler, LedgerScheduler):
            result = await self.ingest.handle(
                event,
                structural_priority=structural_priority,
                pending_threshold=pending_threshold,
            )
            if (
                result.commit_seq is not None
                and result.wake_kind == WakeKind.INBOUND
                and isinstance(payload, Message)
            ):
                await self.scheduler.notify_commit(
                    payload.chat_key, result.commit_seq, priority=result.priority
                )
        else:
            result = await self.ingest.handle(event)
            kind = self._wake_kind(event, result)
            if kind is not None:
                chat_key = event.payload.chat_key
                if kind == "priority":
                    self._wake_meta[chat_key] = "priority"
                else:
                    self._wake_meta.setdefault(chat_key, "ordinary")
                self._schedule_flush()
        # Phase 6 P6.6 on_event hook: observational, fail-open, LIVE only
        # (never in dry-run/replay/doctor). The bus contains every failure.
        if not self.dry_run and self.hooks is not None:
            try:
                await self.hooks.emit_event(event)
            except Exception:
                log.warning(
                    "on_event hook emission failed (contained)", exc_info=True
                )

    def _schedule_flush(self) -> None:
        """Queue exactly ONE flush task for the next event-loop turn. If
        one is already scheduled, the pending metadata coalesces into it."""
        if self._flush_task is not None and not self._flush_task.done():
            return
        self._flush_task = asyncio.create_task(self._flush_wakes())

    async def _flush_wakes(self) -> None:
        """Send ONE scheduler wake per chat with pending metadata: priority
        if any member qualifies, else ordinary. Runs on the next event-loop
        turn after the last commit it coalesces; never sleeps on timestamps
        and never waits for another input or EOF."""
        try:
            await self._flush_wakes_now()
        finally:
            self._flush_task = None
            if self._wake_meta:
                # Commits landed while the flush was running: they coalesce
                # into a later flush.
                self._schedule_flush()

    async def _flush_wakes_now(self) -> None:
        """Send the currently accumulated wake metadata (one wake per chat,
        priority if any member qualifies, else ordinary) and clear it.
        Legacy route only: the ledger route never accumulates wake
        metadata."""
        meta = self._wake_meta
        self._wake_meta = {}
        if self.scheduler is None or isinstance(self.scheduler, LedgerScheduler):
            return
        for chat_key, kind in meta.items():
            if kind == "priority":
                await self.scheduler.wake_priority(chat_key)
            else:
                await self.scheduler.wake(chat_key)

    async def _flush_pending_wakes(self) -> None:
        """Await/cancel-and-flush any scheduled wake task so committed wake
        metadata is not lost (stream end / shutdown)."""
        while self._flush_task is not None or self._wake_meta:
            task = self._flush_task
            if task is not None:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                continue
            await self._flush_wakes_now()

    def _is_priority_wake(self, payload: Message) -> bool:
        """Structural recognition only (no storage read): the message
        mentions the chat's self id or carries a reply target. The gate
        applies the exact precedence (quote-to-self requires the target
        to resolve to a self message)."""
        identity = _adapter_identity(self.adapter, payload.chat_key)
        self_id = getattr(identity, "self_id", None)
        if self_id is not None and self_id in payload.mentions:
            return True
        return payload.reply_to is not None

    def _adapter_ready(self) -> bool:
        """True when the adapter reports a real readiness/handshake. A
        console-style adapter (no ``connected`` property) is ready once
        connected; a OneBot-style adapter is ready only when its connection
        is open."""
        ready = getattr(self.adapter, "ready", None)
        if ready is not None:
            return bool(ready)
        connected = getattr(self.adapter, "connected", None)
        if connected is None:
            return True
        return bool(connected)

    async def _wait_adapter_ready(self) -> None:
        """Block until the adapter reports ready (a real handshake), so no
        worker send happens while the adapter is disconnected. Console-style
        adapters return immediately. Interruptible by cancellation
        (shutdown)."""
        if getattr(self.adapter, "ready", None) is not None:
            while not bool(getattr(self.adapter, "ready", None)):
                await self.clock.sleep(0.5)
            return
        if getattr(self.adapter, "connected", None) is None:
            return
        while not bool(getattr(self.adapter, "connected", None)):
            await self.clock.sleep(0.5)

    async def _wake_outbox(self, items: list[OutboxItem]) -> None:
        """Live outbox wake callback: fired by the CycleRunner after a
        terminal finish that created outbox rows. Records the settled
        chats and wakes the outbox worker so it drains exactly those rows —
        the worker is never started with a startup drain in the live agent
        lane."""
        for item in items:
            self._drain_chats.add(item.chat_key)
            try:
                if str(item.idem_key).startswith("dispatch:"):
                    dispatch_id = int(str(item.idem_key).split(":", 2)[1])
                    self._outbox_dispatches.setdefault(item.chat_key, set()).add(
                        dispatch_id
                    )
            except (ValueError, IndexError):
                pass
        self._wake.set()

    async def _wake_control_target(self, chat_key: ChatKey) -> None:
        """Wake a durably controlled target without inventing an inbound row."""
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, LedgerScheduler):
            await self.scheduler.notify_startup(chat_key)
        else:
            await self.scheduler.wake_priority(chat_key)

    async def _on_settled(
        self, chat_key: ChatKey, through_msg_id: MessageRowId
    ) -> None:
        """Default ``on_settled`` callback: fired by the CycleRunner from
        TERMINAL dispatch settlement ONLY (after settle_dispatch, the outbox
        wake, and the marker export). Enqueues the chat for every enabled
        learner — NON-BLOCKING, never awaits LLM/learn/media work (the
        settlement path must never stall on the learner)."""
        if self._learner is None:
            return
        try:
            self._learner.enqueue(chat_key)
        except Exception:
            log.warning(
                "learner enqueue failed for %s (contained)", chat_key,
                exc_info=True,
            )
        if self._semantic_backfill is not None:
            try:
                self._semantic_backfill.enqueue(chat_key)
            except Exception:
                log.warning(
                    "semantic backfill enqueue failed for %s (contained)",
                    chat_key, exc_info=True,
                )

    async def _on_exposure(
        self,
        chat_key: ChatKey,
        records: tuple[Record, ...],
        dispatch_id: int,
        through_msg_id: MessageRowId,
    ) -> None:
        """Default ``on_exposure`` callback: fired by the CycleRunner after a
        TERMINAL reply dispatch that created durable outbox output
        (LIVE-only — the runner never fires it in dry-run). Records
        idempotent exposures/uses for the records that were actually frozen
        into the prompt (an exposure is created once per (record, producing
        learner run); the uses bump rides on the FIRST exposure only), then
        hands the frozen records to the learner worker as pending effect
        references (with the dispatch's through boundary as the effect
        reference boundary). Every failure is contained and logged."""
        self._pending_exposures[(chat_key, dispatch_id)] = (records, through_msg_id)
        if (chat_key, dispatch_id) in self._confirmed_dispatches:
            await self._confirm_exposure_delivery(chat_key, dispatch_id)

    async def _confirm_exposure_delivery(
        self, chat_key: ChatKey, dispatch_id: int
    ) -> None:
        """Persist one exact selection only after its delivery succeeds."""
        pending = self._pending_exposures.pop((chat_key, dispatch_id), None)
        self._confirmed_dispatches.discard((chat_key, dispatch_id))
        if pending is None or self.repo is None:
            return
        records, through_msg_id = pending
        now = self.clock.now()
        for rec in records:
            if rec.id is None or rec.producing_run_id is None:
                continue
            try:
                created = await self.repo.record_exposure(
                    chat_key, rec.learner, rec.id, rec.producing_run_id,
                    now=now, dispatch_id=dispatch_id, slot=rec.learner,
                )
                if created:
                    await self.repo.increment_record_uses(
                        chat_key, rec.learner, rec.id
                    )
            except Exception:
                log.warning(
                    "exposure failed for %s/%s (contained)",
                    chat_key, rec.learner, exc_info=True,
                )
        if self._learner is not None:
            try:
                self._learner.note_exposure(
                    chat_key, records, through_msg_id, dispatch_id=dispatch_id
                )
                self._learner.mark_delivered(chat_key)
                self._learner.enqueue(chat_key)
            except Exception:
                log.warning(
                    "effect note failed for %s (contained)", chat_key,
                    exc_info=True,
                )

    async def _producing_run_id(
        self, chat_key: ChatKey, learner: str, record: Record
    ) -> int | None:
        """Return the exact persisted run that produced ``record``, or None.

        The ``record_exposures.run_id`` column references ``learner_runs``
        (a schema FK), so the exposure must be tied to a REAL learner run —
        never a synthetic dispatch id.  Range reconstruction is ambiguous
               after merges and is intentionally not used."""
        return record.producing_run_id

    async def _on_memory_default(
        self, chat_key: ChatKey, through_msg_id: MessageRowId
    ) -> None:
        """Default ``on_memory`` callback: fired by the CycleRunner from
        TERMINAL dispatch settlement ONLY (after the durable finish), with
        the chat and the frozen through boundary. Drains the local memory
        backlog FULLY but BOUNDEDLY — repeating local batches until the
        durable watermark catches the cursor — via the owned MemoryService
        (deterministic local capsule — no provider/network/LLM), then
        ENQUEUES the chat for semantic backfill (non-blocking; the terminal
        settle path never awaits provider work). Any failure is contained
        and logged; it never blocks outbox/marker/hook behavior. No
        pre-settlement/defer/release/replay write happens here.
        """
        if self.memory_service is None:
            return
        try:
            # Drain the local memory backlog FULLY but BOUNDEDLY: repeat
            # local batches until the durable watermark catches the cursor.
            # Each "ok" advances the watermark by one bounded batch, so the
            # loop terminates exactly when the watermark catches the cursor
            # ("no_work") — the bound is the backlog size, never an infinite
            # loop. The local capsule summarizer performs NO provider work.
            while True:
                result = await self.memory_service.summarize(
                    chat_key, through_msg_id=through_msg_id
                )
                if result.status in ("no_work", "unavailable", "stale"):
                    break
                if result.status != "ok":
                    break
        except Exception:
            log.warning(
                "memory summarize failed for %s (contained)", chat_key, exc_info=True
            )
        if self._semantic_backfill is not None:
            self._semantic_backfill.enqueue(chat_key)

    async def _memory_maintenance(self) -> None:
        """Local, idempotent memory maintenance at DB start.

        1. Bootstrap the canonical memory FTS index for every chat whose
           docs are missing (no provider/network/LLM).
        2. Repair crash-after-settlement gaps: drain the local memory backlog
           fully but boundedly per chat with pending memory work (the same
           bounded maintenance the terminal ``on_memory`` callback runs).

        Every failure is contained and logged; this never blocks startup.
        """
        if self.repo is None or self.memory_service is None:
            return
        try:
            for chat_key in await self.repo.list_memory_fts_unbootstrapped_chats():
                try:
                    await self.repo.rebuild_memory_fts(chat_key)
                except Exception:
                    log.warning(
                        "memory FTS bootstrap failed for %s (contained)",
                        chat_key,
                        exc_info=True,
                    )
        except Exception:
            log.warning(
                "memory FTS bootstrap enumeration failed (contained)",
                exc_info=True,
            )
        try:
            for chat_key, through_msg_id in await self.repo.list_memory_pending_chats():
                await self._on_memory_default(chat_key, through_msg_id)
        except Exception:
            log.warning(
                "memory pending-chat maintenance failed (contained)",
                exc_info=True,
            )

    def _worker_chats(self) -> list[ChatKey]:
        """The chats the outbox worker should drain this round: the chats
        with newly settled rows (from the live wake callback), the chats
        retained as active across rounds (startup recovery + future-paced
        rows), plus the console adapter's single chat. Deterministic order
        (sorted) so no chat is starved by iteration order."""
        chats = set(self._drain_chats)
        self._drain_chats.clear()
        chats |= self._active_chats
        chat_key = getattr(self.adapter, "chat_key", None)
        if chat_key is not None:
            chats.add(chat_key)
        return sorted(chats)

    async def _outbox_worker(self) -> None:
        """Drain every due pending row, then sleep until the next due
        ``send_after_ts`` and drain again — non-busy, no polling, no
        inbound input required. Idles on a wake event when nothing is
        pending. A future-row sleep is INTERRUPTED by a wake event (the
        CycleRunner's ``on_outbox`` after a terminal settlement), so a newly
        due row is reconsidered promptly instead of waiting out the old
        sleep. Active chats are retained across rounds, so a future-paced
        row is rechecked/drained when its sleep naturally expires without a
        new wake. Each chat gets ONE BOUNDED pump per round with a FRESH
        clock, so a slow or backlogged chat cannot starve others or use a
        stale timestamp, and the sleep targets the ACTUAL next due after the
        sends. No send happens while the adapter is disconnected (no
        premature in_flight transition). Stopped by cancellation in
        ``shutdown``."""
        assert self.repo is not None and self.adapter is not None
        assert self.outbox is not None
        while True:
            if not self._adapter_ready():
                # No worker send while the adapter is disconnected: wait for
                # a real handshake (interruptible by shutdown cancellation).
                await self._wait_adapter_ready()
                continue
            chats = self._worker_chats()
            if not chats:
                await self._wake.wait()
                self._wake.clear()
                continue
            # One BOUNDED pump per chat per round with a FRESH clock, then
            # retain the chats that still hold pending rows (future-paced
            # rows) and find the actual next due across them.
            still_active: list[ChatKey] = []
            next_due: float | None = None
            for chat in chats:
                now = self.clock.now()
                sent = await self.outbox.pump(chat, now=now, limit=10)
                if sent > 0:
                    # A terminal exposure callback records intent only.  The
                    # worker now has a successful adapter result, so commit
                    # each pending dispatch for this chat.  This keeps the
                    # effect lane delivery-gated rather than chat-boolean or
                    # creation-time based.
                    for pending_dispatch in self._outbox_dispatches.get(chat, ()):
                        confirmed = True
                        check_delivery = getattr(
                            self.repo, "dispatch_delivery_confirmed", None
                        )
                        if check_delivery is not None:
                            confirmed = await check_delivery(chat, pending_dispatch)
                        if confirmed:
                            self._confirmed_dispatches.add((chat, pending_dispatch))
                            await self._confirm_exposure_delivery(chat, pending_dispatch)
                due = await self.repo.next_due_outbox(chat, now=self.clock.now())
                if due is not None:
                    still_active.append(chat)
                    if next_due is None or due < next_due:
                        next_due = due
            self._active_chats = set(still_active)
            if next_due is None:
                await self._wake.wait()
                self._wake.clear()
                continue
            now = self.clock.now()
            if next_due > now:
                # Sleep until the ACTUAL next due row (fresh clock after the
                # sends), but wake early when a new outbox row arrives (the
                # CycleRunner's on_outbox wake). The clock sleep and the wake
                # event race; the loser is cancelled.
                sleep_task = asyncio.create_task(self.clock.sleep(next_due - now))
                wake_task = asyncio.create_task(self._wake.wait())
                try:
                    done, pending = await asyncio.wait(
                        {sleep_task, wake_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                except asyncio.CancelledError:
                    for t in (sleep_task, wake_task):
                        t.cancel()
                    raise
                for t in pending:
                    t.cancel()
                for t in pending:
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
                if wake_task in done:
                    self._wake.clear()
                    continue  # a new row arrived: reconsider now

    async def _drain_scheduler(self) -> None:
        """Let the scheduler finish its in-flight cycle before ``stop()``:
        a terminal decision's trace must be persisted/printed before
        shutdown cancels the loop. Bounded by a short WALL-CLOCK budget
        (not an iteration count — a cycle spans several writer batches,
        each up to the 50 ms coalescing window, so a fixed yield count
        can cancel a healthy cycle mid-flight): a timed re-arm (delay) is
        not waited out — the loop gives up after the budget and stops the
        scheduler anyway."""
        if self.scheduler is None or self.adapter is None:
            return
        chat_key = getattr(self.adapter, "chat_key", None)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 2.0
        while loop.time() < deadline:
            if (
                (chat_key is None or not self.scheduler.is_leased(chat_key))
                and self.scheduler.pending_wakes() == 0
            ):
                return
            await asyncio.sleep(0)

    async def shutdown(self) -> None:
        """Stop the scheduler (drained, then cancelled), stop the outbox
        worker (cancellation), then close adapter, recorder, and database.
        Idempotent and safe to call from a ``finally``.

        Robust to every failure mode: a failed/cancelled semantic task never
        poisons the rest of shutdown, and owned resources (adapter, LLM
        clients, recorder, DB) are ALWAYS closed — even when ``start()`` was
        never called or a cleanup step raises/cancels (the resource close
        runs in a ``finally``)."""
        if self._shutdown:
            return
        self._shutdown = True
        self._started = False
        try:
            # Shutdown mid-stream: flush any scheduled wake task so committed
            # wake metadata is not lost BEFORE the scheduler is drained/
            # stopped, so the wake is served and the terminal decision's
            # trace is persisted (commit-before-wake).
            await self._flush_pending_wakes()
            # Cancel/drain the semantic backfill worker BEFORE the shared
            # LLM/DB close: the worker must not touch a closed client or
            # database. Catch ALL exceptions — a failed semantic task must
            # not poison the rest of shutdown.
            if self._semantic_task is not None:
                if self._semantic_backfill is not None:
                    self._semantic_backfill.cancel()
                self._semantic_task.cancel()
                try:
                    await self._semantic_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._semantic_task = None
            # Cancel/drain the learner worker BEFORE the shared LLM/DB close:
            # the worker must not touch a closed client or database. The
            # pipeline settles any in-flight run ``cancelled`` (best-effort,
            # shielded) and re-raises; the worker's cancellation is
            # suppressed here so it never poisons the rest of shutdown.
            if self._learner_task is not None:
                if self._learner is not None:
                    self._learner.cancel()
                self._learner_task.cancel()
                try:
                    await self._learner_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._learner_task = None
            # Cancel the bounded media harvest tasks BEFORE the shared
            # LLM/DB close: an in-flight harvest must not touch a closed
            # client or database. The harvester's tasks contain their own
            # failures; cancellation is best-effort.
            if self._media_harvester is not None:
                self._media_harvester.cancel()
            if self.scheduler is not None and self._scheduler_started:
                await self._drain_scheduler()
                await self.scheduler.stop()
                self._scheduler_started = False
            if self._worker is not None:
                self._worker.cancel()
                try:
                    await self._worker
                except asyncio.CancelledError:
                    pass
                self._worker = None
            if self._receiver is not None:
                self._receiver.cancel()
                try:
                    await self._receiver
                except asyncio.CancelledError:
                    pass
                self._receiver = None
        finally:
            # Always close owned resources, even if a step above raised or
            # was cancelled, and even when start() was never called.
            close = getattr(self.adapter, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    pass
            if self._llm is not None:
                aclose = getattr(self._llm, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass
            if self._learn_llm is not None:
                aclose = getattr(self._learn_llm, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass
            if self._media_llm is not None:
                aclose = getattr(self._media_llm, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass
            if self._embed_llm is not None:
                aclose = getattr(self._embed_llm, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass
            if self.recorder is not None:
                self.recorder.close()
            if self.repo is not None:
                await self.repo.close()
