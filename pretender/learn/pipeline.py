"""The generic async learner pipeline (Phase 6).

One pipeline serves every learner: a declarative ``LearnerSpec`` drives the
whole run, and a per-spec validator turns the parsed model output into
validated records. The run is a strict sequence:

1. **source-bounded grant** — ``acquire_learner_run`` (a durable CAS lease).
   Busy / unknown chat skips with zero provider calls.
2. **source-bounded read** — ``read_learner_source_batch`` with the spec's
   ``policy`` (``nonself`` excludes the bot's own messages in SQL) and
   ``batch_size`` tail. Nothing beyond the watermark releases the run and
   skips with zero calls.
3. **optional per-chat atomic budget reservation** — ``budget.reserve``
   (one call). A blocked OR degraded decision skips with zero calls (the
   reservation is not consumed).
4. **exactly one provider completion** with a 45s deadline (absolute epoch).
   Tokens are recorded after the call; there is NO retry — a failure settles
   the run and the next scheduled run picks the source up again.
5. **strict parsing** — raw JSON or exactly one outer code fence, via the
   spec's parser (``parse_json_response``). No tolerant repair lane.
6. **strict validation** — the spec's validator enforces the hard limits,
   opaque refs must exist, and the model can never set
   ``weight``/``uses``/``delta``.
7. **all-or-nothing commit** — ``commit_learner_source`` (one writer
   transaction, CAS-fenced on the observed watermark). A valid EMPTY result
   advances the watermark; a stale CAS changes nothing.

Fail-closed semantics: malformed / timeout / prompt / provider failures
settle the run WITHOUT advancing the watermark (the source stays pending for
the next run); cancellation settles the run ``cancelled`` (best-effort,
shielded) and RERAISES. No provider call ever happens inside a transaction,
and the pipeline never writes records/vectors/outbox/ledger directly — the
only write surface is ``commit_learner_source``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from pretender.budget import BudgetDecision, BudgetManager, BudgetUsage
from pretender.clock import Clock
from pretender.errors import PromptError
from pretender.learn.parse import LearnerParseError, parse_json_response
from pretender.learn.render import (
    render_attributed_batch,
    render_batch,
    source_hash,
)
from pretender.learn.specs import LearnerValidationError, Validator
from pretender.prompts import PromptStore
from pretender.seams import AdaptiveRepository, LLMClient
from pretender.types import (
    ChatKey,
    LearnerBatch,
    LearnerBusy,
    LearnerDraft,
    LearnerRunRequest,
    LearnerSpec,
    MessageRowId,
    TranscriptMessage,
)

__all__ = [
    "DEFAULT_LEARN_DEADLINE_S",
    "DEFAULT_LEARN_LEASE_S",
    "LEARN_PROFILE",
    "LearnerRunResult",
    "LearnerPipeline",
]

LEARN_PROFILE = "learn"
DEFAULT_LEARN_DEADLINE_S = 45.0
DEFAULT_LEARN_LEASE_S = 120.0
DEFAULT_LEARN_FAILURE_BACKOFF_CAP_S = 86400.0

_ALLOWED = "allowed"

# Outcome strings of a learner run (see ``LearnerRunResult``).
OUTCOME_SUCCESS = "success"
OUTCOME_SKIPPED = "skipped"
OUTCOME_STALE = "stale"
OUTCOME_MALFORMED = "malformed"
OUTCOME_PROVIDER_ERROR = "provider_error"
OUTCOME_PROMPT_ERROR = "prompt_error"


@dataclass(frozen=True)
class LearnerRunResult:
    """The typed outcome of one learner run.

    ``outcome`` is ``success`` (committed — possibly zero records, the
    watermark advanced), ``skipped`` (no provider call: busy/unknown chat,
    no source, budget blocked/degraded, disabled spec), ``stale`` (the CAS
    commit lost — the watermark moved, nothing changed), ``malformed``
    (parse/schema failure — settled, watermark NOT advanced),
    ``provider_error`` (the provider call failed — settled, watermark NOT
    advanced), or ``prompt_error`` (prompt rendering failed — settled,
    watermark NOT advanced). ``calls`` is the number of provider calls made
    (always 0 or 1). ``usage`` carries the provider usage of the call.
    """

    learner: str
    chat_key: ChatKey
    outcome: str
    run_id: int | None = None
    records_added: int = 0
    records_merged: int = 0
    watermark: MessageRowId | None = None
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    calls: int = 0

    def __post_init__(self) -> None:
        if self.outcome not in (
            OUTCOME_SUCCESS,
            OUTCOME_SKIPPED,
            OUTCOME_STALE,
            OUTCOME_MALFORMED,
            OUTCOME_PROVIDER_ERROR,
            OUTCOME_PROMPT_ERROR,
        ):
            raise ValueError(f"invalid learner run outcome: {self.outcome!r}")
        if self.run_id is not None and (
            isinstance(self.run_id, bool)
            or not isinstance(self.run_id, int)
            or self.run_id < 0
        ):
            raise ValueError("run_id must be a nonnegative integer")
        if self.records_added < 0 or self.records_merged < 0:
            raise ValueError("record counts must be nonnegative")
        if self.calls not in (0, 1):
            raise ValueError("calls must be 0 or 1 (exactly one provider call max)")
        usage = {
            k: v
            for k, v in (self.usage or {}).items()
            if isinstance(v, int) and not isinstance(v, bool)
        }
        object.__setattr__(self, "usage", usage)


class LearnerPipeline:
    """The generic async learner pipeline over a declarative ``LearnerSpec``.

    ``repo`` is the ``AdaptiveRepository``; ``llm`` the ``LLMClient``
    (profile ``"learn"`` by default); ``prompts`` the ``PromptStore`` that
    resolves each spec's prompt file; ``clock`` the time source.
    ``validators`` maps spec name -> validator (the five frozen validators
    from ``pretender.learn.specs`` by default). ``budget`` is the optional
    per-chat ``BudgetManager`` (or any object with the same
    ``reserve``/``record`` surface); ``deadline_s`` is the provider deadline
    (45s by default); ``lease_s`` the run lease.
    """

    def __init__(
        self,
        repo: AdaptiveRepository,
        llm: LLMClient,
        prompts: PromptStore,
        clock: Clock,
        *,
        validators: dict[str, Validator] | None = None,
        budget: BudgetManager | None = None,
        profile: str = LEARN_PROFILE,
        deadline_s: float = DEFAULT_LEARN_DEADLINE_S,
        lease_s: float = DEFAULT_LEARN_LEASE_S,
        logger: logging.Logger | None = None,
    ) -> None:
        if repo is None or llm is None or prompts is None or clock is None:
            raise ValueError("repo, llm, prompts and clock are required")
        if deadline_s <= 0 or lease_s <= 0:
            raise ValueError("deadline_s and lease_s must be positive")
        self._repo = repo
        self._llm = llm
        self._prompts = prompts
        self._clock = clock
        self._validators = dict(validators or {})
        self._budget = budget
        self._profile = profile
        self._deadline_s = float(deadline_s)
        self._lease_s = float(lease_s)
        self._logger = logger or logging.getLogger("pretender.learn")

    # ── the run ─────────────────────────────────────────────────────────────

    async def run(
        self,
        chat_key: ChatKey,
        spec: LearnerSpec,
        *,
        references: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LearnerRunResult:
        """Run one learner over the chat's pending source range.

        ``references`` is the rendered reference surface for the effect
        learner (empty by default — the runner wires real references later).
        """
        if not spec.enabled:
            return LearnerRunResult(
                learner=spec.name, chat_key=chat_key, outcome=OUTCOME_SKIPPED,
                error="disabled",
            )
        now = self._clock.now()
        grant = await self._repo.acquire_learner_run(
            LearnerRunRequest(
                chat_key=chat_key,
                learner=spec.name,
                started_ts=now,
                expires_at=now + self._lease_s,
                now=now,
            )
        )
        if grant is None:
            return LearnerRunResult(
                learner=spec.name, chat_key=chat_key, outcome=OUTCOME_SKIPPED,
                error="unknown_chat",
            )
        if isinstance(grant, LearnerBusy):
            return LearnerRunResult(
                learner=spec.name, chat_key=chat_key, outcome=OUTCOME_SKIPPED,
                error="busy",
            )
        run_id = grant.run_id
        batch: LearnerBatch | None = None
        budget_reserved = False
        cancellation_settled = False
        try:
            batch = await self._repo.read_learner_source_batch(
                chat_key,
                spec.name,
                through_msg_id=grant.through_msg_id,
                tail=spec.batch_size,
                policy=spec.policy,
            )
            if batch is None:
                # Nothing beyond the watermark: give the run back, no call.
                await self._repo.release_learner_run(chat_key, spec.name, run_id)
                return LearnerRunResult(
                    learner=spec.name, chat_key=chat_key, outcome=OUTCOME_SKIPPED,
                    run_id=run_id, error="no_source",
                )
            # Defensive determinism check: the batch's hash must match its
            # own texts (the repository computes it; this is a fail-closed
            # guard against an inconsistent batch).
            if source_hash(batch.texts) != batch.source_hash:
                await self._settle(
                    chat_key, spec.name, batch, run_id, "malformed",
                    "source hash mismatch", self._clock.now(), cadence_s=spec.cadence_s,
                )
                return LearnerRunResult(
                    learner=spec.name, chat_key=chat_key, outcome=OUTCOME_MALFORMED,
                    run_id=run_id, error="source hash mismatch",
                )

            # Optional per-chat atomic budget reservation: blocked OR
            # degraded skips with zero provider calls.
            if self._budget is not None:
                decision = await self._budget.reserve(chat_key, calls=1)
                if decision.kind != _ALLOWED:
                    await self._repo.release_learner_run(chat_key, spec.name, run_id)
                    return LearnerRunResult(
                        learner=spec.name, chat_key=chat_key,
                        outcome=OUTCOME_SKIPPED, run_id=run_id,
                        error=f"budget_{decision.kind}",
                    )
                budget_reserved = True

            # Prompt rendering (the spec's prompt file + the opaque-ref
            # batch rendering). A render failure settles malformed.
            try:
                prompt_text = self.render_prompt(spec, batch, references=references)
            except PromptError as exc:
                await self._settle(
                    chat_key, spec.name, batch, run_id, "malformed",
                    f"prompt: {exc}", self._clock.now(), cadence_s=spec.cadence_s,
                )
                return LearnerRunResult(
                    learner=spec.name, chat_key=chat_key,
                    outcome=OUTCOME_PROMPT_ERROR, run_id=run_id,
                    error=str(exc),
                )

            # Exactly one provider completion with the absolute deadline.
            deadline = now + self._deadline_s
            try:
                resp = await self._llm.complete(
                    [TranscriptMessage(role="system", content=prompt_text)],
                    profile=self._profile,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    deadline=deadline,
                )
            except asyncio.CancelledError:
                cancellation_settled = True
                await self._settle_cancelled(chat_key, spec.name, batch, run_id,
                                             self._clock.now(), cadence_s=spec.cadence_s)
                raise
            except Exception as exc:
                error = f"provider: {type(exc).__name__}: {exc}"
                await self._settle(
                    chat_key, spec.name, batch, run_id, "malformed", error,
                    self._clock.now(),
                    cadence_s=spec.cadence_s,
                )
                return LearnerRunResult(
                    learner=spec.name, chat_key=chat_key,
                    outcome=OUTCOME_PROVIDER_ERROR, run_id=run_id,
                    error=error, calls=1,
                )
            # Provider time is deliberately not reused as the durable CAS /
            # settlement timestamp.  A slow provider may have crossed the
            # lease boundary; all post-provider durable writes use this fresh
            # sample (and settlement helpers sample again on their callers).
            now = self._clock.now()
            if self._budget is not None:
                await self._budget.record(
                    chat_key, calls=0, tokens=_usage_tokens(resp.usage)
                )
                budget_reserved = False

            # Strict parsing: raw JSON or exactly one outer fence.
            try:
                parsed = parse_json_response(resp.content)
            except LearnerParseError as exc:
                error = f"parse: {exc}"
                await self._settle(
                    chat_key, spec.name, batch, run_id, "malformed", error, now,
                    cadence_s=spec.cadence_s,
                )
                return LearnerRunResult(
                    learner=spec.name, chat_key=chat_key,
                    outcome=OUTCOME_MALFORMED, run_id=run_id,
                    error=error, calls=1, usage=resp.usage,
                )

            # Strict schema validation via the spec's validator.
            validator = self._validators.get(spec.name)
            if validator is None:
                await self._settle(
                    chat_key, spec.name, batch, run_id, "malformed",
                    f"no validator registered for {spec.name!r}", now,
                    cadence_s=spec.cadence_s,
                )
                return LearnerRunResult(
                    learner=spec.name, chat_key=chat_key,
                    outcome=OUTCOME_MALFORMED, run_id=run_id,
                    error=f"no validator registered for {spec.name!r}",
                    calls=1, usage=resp.usage,
                )
            try:
                records = validator(parsed, batch, now=now)
            except LearnerValidationError as exc:
                error = f"schema: {exc}"
                await self._settle(
                    chat_key, spec.name, batch, run_id, "malformed", error, now,
                    cadence_s=spec.cadence_s,
                )
                return LearnerRunResult(
                    learner=spec.name, chat_key=chat_key,
                    outcome=OUTCOME_MALFORMED, run_id=run_id,
                    error=error, calls=1, usage=resp.usage,
                )

            # All-or-nothing CAS commit (one writer transaction). A valid
            # EMPTY result still advances the watermark.
            committed = await self._repo.commit_learner_source(
                LearnerDraft(
                    chat_key=chat_key,
                    learner=spec.name,
                    batch=batch,
                    records=tuple(records),
                    expected_through_msg_id=batch.observed_watermark,
                    run_id=run_id,
                    cadence_s=spec.cadence_s,
                ),
                now=now,
            )
            if not committed:
                # Stale CAS: the watermark moved; nothing changed. Give the
                # stale run back so the next acquire starts fresh.
                await self._repo.release_learner_run(chat_key, spec.name, run_id)
                return LearnerRunResult(
                    learner=spec.name, chat_key=chat_key, outcome=OUTCOME_STALE,
                    run_id=run_id, calls=1, usage=resp.usage,
                )
            return LearnerRunResult(
                learner=spec.name, chat_key=chat_key, outcome=OUTCOME_SUCCESS,
                run_id=run_id, records_added=len(records),
                watermark=batch.last_msg_id, calls=1, usage=resp.usage,
            )
        except asyncio.CancelledError:
            # Cancellation reraises after a best-effort cancelled settle.
            if batch is not None and not cancellation_settled:
                cancellation_settled = True
                # Never reuse the pre-provider/pre-cancellation timestamp.
                await self._settle_cancelled(
                    chat_key, spec.name, batch, run_id, self._clock.now(),
                    cadence_s=spec.cadence_s,
                )
            else:
                await self._repo.release_learner_run(chat_key, spec.name, run_id)
            raise
        except Exception:
            if batch is not None:
                await self._settle(
                    chat_key, spec.name, batch, run_id, "malformed",
                    "learner internal error", self._clock.now(),
                    cadence_s=spec.cadence_s,
                )
            else:
                await self._repo.release_learner_run(chat_key, spec.name, run_id)
            raise
        finally:
            if budget_reserved and self._budget is not None:
                release = getattr(self._budget, "release", None)
                if release is not None:
                    release()

    # ── prompt rendering ────────────────────────────────────────────────────

    def render_prompt(
        self,
        spec: LearnerSpec,
        batch: LearnerBatch,
        *,
        references: str = "",
    ) -> str:
        """Render the spec's prompt file with the opaque-ref batch surface.

        The prompt template receives ``{{messages}}`` (the escaped opaque-ref
        message list), ``{{attributed_messages}}`` (the same list with each
        speaker's display name, which only the impression learner needs),
        ``{{learner}}``, ``{{chat_key}}``, ``{{count}}`` and ``{{references}}``
        (the effect learner's reference surface). A missing variable raises
        ``PromptError`` (fail closed).
        """
        return self._prompts.render(
            spec.prompt,
            messages=render_batch(batch),
            attributed_messages=render_attributed_batch(batch),
            learner=spec.name,
            chat_key=str(batch.chat_key),
            count=len(batch.texts),
            references=references,
        )

    # ── settlement helpers ──────────────────────────────────────────────────

    async def _settle(
        self,
        chat_key: ChatKey,
        learner: str,
        batch: LearnerBatch,
        run_id: int,
        outcome: str,
        error: str,
        now: float,
        *,
        cadence_s: float | None = None,
    ) -> None:
        """Best-effort fail-closed settle: the run is settled WITHOUT
        advancing the watermark or inserting records. A stale settle (the
        watermark moved) releases the run instead."""
        try:
            ok = await asyncio.shield(
                self._repo.commit_learner_source(
                    LearnerDraft(
                        chat_key=chat_key,
                        learner=learner,
                        batch=batch,
                        run_id=run_id,
                        cadence_s=cadence_s,
                        outcome=outcome,
                        error=error,
                    ),
                    now=now,
                )
            )
            if not ok:
                await asyncio.shield(
                    self._repo.release_learner_run(chat_key, learner, run_id)
                )
        except Exception as exc:  # containment: the outcome already stands
            self._logger.warning(
                "learner settle failed for %s/%s run %s: %s",
                chat_key, learner, run_id, exc,
            )

    async def _settle_cancelled(
        self,
        chat_key: ChatKey,
        learner: str,
        batch: LearnerBatch,
        run_id: int,
        now: float,
        *,
        cadence_s: float | None = None,
    ) -> None:
        # The caller's pre-cancellation sample is not authoritative.  Always
        # fence cancellation settlement with a clock read made here, after
        # cancellation was observed.
        await self._settle(
            chat_key, learner, batch, run_id, "cancelled", "cancelled",
            self._clock.now(), cadence_s=cadence_s,
        )


def _usage_tokens(usage: dict[str, int] | None) -> int:
    """The total token count of one provider response."""
    return int((usage or {}).get("prompt_tokens", 0)) + int(
        (usage or {}).get("completion_tokens", 0)
    )
