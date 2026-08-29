"""Doctor: deterministic preflight probes for the whole runtime.

The doctor is the ``pretender doctor`` preflight (PLAN.md §7, M1): it probes
every seam the bot depends on and collects ALL results instead of failing
fast, so one run reports every broken piece at once. It never sends chat
output and never wires the CLI — it only produces a structured report a CLI
command can render later.

Probes (fixed order, all async, all injectable):

  - ``config``     — the Config is structurally valid; every LLM profile has
                     a base_url and a model; the profile list is reported
                     (api_key PRESENCE is reported, never the key).
  - ``prompts``    — every required prompt asset loads and renders with the
                     standard variable set.
  - ``database``   — the Database is writable (a probe kv row round-trips
                     and is removed in the same transaction), FTS5 is
                     present and queryable, and the Repository seam answers
                     ``stats()``.
  - ``adapter``    — the adapter handshake (``connect``) succeeds and its
                     name/capabilities are reported.
  - ``llm_chat``   — a minimal chat completion round-trips through the
                     LLMClient (skipped when no LLM profile is configured).
  - ``llm_tools``  — a completion WITH tools returns a well-formed tool call
                     (skipped when no LLM profile is configured).
  - ``vision``     — OPTIONAL capability: skipped unless a ``vision`` profile
                     is configured; a configured-but-broken vision profile is
                     a hard failure.
  - ``embed``      — OPTIONAL capability: skipped unless an ``embed`` profile
                     is configured; a configured-but-broken embed profile is
                     a hard failure.

Status model: ``ok`` | ``fail`` | ``skip``. ``skip`` marks an unavailable
OPTIONAL capability (not configured) — never a failure; ``fail`` marks a
required probe that broke or an optional capability that IS configured but
does not work. The report status is ``fail`` iff any probe failed.

Secrets: every probe detail/error is scrubbed of every configured
``api_key`` before it is recorded, and probe ``data`` never carries secret
values, so ``DoctorReport.render()`` is safe to print.

Seams are consumed without modification: ``Config``, ``PromptStore``,
``OpenAIClient``/``LLMClient``/``Embedder``, ``Database``/``Repository`` and
the ``Adapter`` protocol. Anything not injected is built from the config and
owned (and closed) by the doctor; injected seams are never closed.
"""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, cast

from pretender.adapters.console import ConsoleAdapter
from pretender.adapters.onebot import OneBotAdapter
from pretender.clock import RealClock
from pretender.config import Config
from pretender.context import render_image_markdown, serialize
from pretender.emoji import parse_vision_result
from pretender.db import Database
from pretender.errors import ConfigError, PromptError
from pretender.llm import OpenAIClient, scrub_credentials
from pretender.prompts import PromptStore
from pretender.repo import SqliteRepository
from pretender.seams import Adapter, Clock, Embedder, LLMClient, Repository
from pretender.types import TranscriptMessage

__all__ = ["Doctor", "DoctorReport", "ProbeResult"]

# Fixed probe order — the report is deterministic by construction.
_PROBE_ORDER = (
    "config",
    "prompts",
    "database",
    "adapter",
    "llm_chat",
    "llm_tools",
    "vision",
    "embed",
    "semantic",
    "learner",
    "media_catalog",
    "plugins",
    "chat_controls",
)

# The prompt assets the bot's personality/planner/replyer lanes require.
_REQUIRED_PROMPTS = ("identity.txt", "planner.txt", "replyer.txt", "planner_focus.txt")

# The standard variable set each templated prompt must render with. A prompt
# that references anything else is a broken asset, not a doctor bug.
_PROMPT_VARIABLES: dict[str, dict[str, str]] = {
    "planner.txt": {
        "identity": "probe",
        "chat_log": "",
        "reply_style": "probe",
        "bot_name": "probe",
        "drift_block": "",
    },
    "planner_focus.txt": {
        "identity": "probe",
        "chat_log": "",
        "reply_style": "probe",
        "focus_chat": "probe",
        "bot_name": "probe",
        "drift_block": "",
    },
    # The reply reference is no longer a system-prompt slot: it rides in the
    # final user turn, alongside the time and the target message.
    "replyer.txt": {
        "identity": "probe",
        "reply_style": "probe",
        "bot_name": "probe",
        "drift_block": "",
    },
}

# The minimal tool the tool-calling probe offers the provider.
_PROBE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "doctor_probe",
        "description": "A doctor probe tool.",
        "parameters": {"type": "object", "properties": {}},
    },
}

# Profile preference for the chat/tool probes: the bot's planner lane first.
_CHAT_PROFILE_PREFERENCE = ("planner", "reply", "main", "chat")


def _build_adapter(cfg: Config, clock: Clock) -> Adapter:
    """Build the adapter selected by the config (``adapter.name``): the
    console REPL by default, or the OneBot v11 bridge. An unknown selection
    is a ConfigError — the doctor never boots an unsupported adapter."""
    name = cfg.adapter.name
    if name == "console":
        return ConsoleAdapter(clock=clock)
    if name == "onebot":
        return OneBotAdapter(config=cfg.adapter.onebot, clock=clock)
    raise ConfigError(f"unsupported adapter {name!r}")


@dataclass(frozen=True)
class ProbeResult:
    """One probe's verdict.

    ``status`` is ``ok`` | ``fail`` | ``skip``. ``detail`` is a short
    human-readable summary; ``error`` is the (scrubbed) failure message when
    the probe failed; ``data`` carries structured, secret-free facts (profile
    names, capabilities, embed dimension, ...) for later CLI rendering.
    """

    name: str
    status: str  # "ok" | "fail" | "skip"
    detail: str = ""
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ("ok", "fail", "skip"):
            raise ValueError(f"invalid probe status: {self.status!r}")

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class DoctorReport:
    """The full doctor result: every probe, the aggregate status, and a
    deterministic text rendering for the CLI."""

    probes: tuple[ProbeResult, ...]
    status: str  # "ok" | "fail"
    summary: str

    def by_name(self, name: str) -> ProbeResult | None:
        for probe in self.probes:
            if probe.name == name:
                return probe
        return None

    @property
    def failed(self) -> bool:
        return self.status == "fail"

    def render(self) -> str:
        """A deterministic, secret-free text report (CLI-ready)."""
        lines = [f"doctor: {self.status}"]
        for probe in self.probes:
            line = f"  [{probe.status.upper()}] {probe.name}"
            if probe.detail:
                line += f" — {probe.detail}"
            if probe.error:
                line += f" — error: {probe.error}"
            lines.append(line)
        lines.append(f"summary: {self.summary}")
        return "\n".join(lines)


class Doctor:
    """Runs every probe against the configured (or injected) seams.

    ``run()`` is single-use: owned seams are closed when it finishes. Injected
    seams are never closed — the caller owns their lifecycle.
    """

    PROBES = _PROBE_ORDER

    def __init__(
        self,
        cfg: Config,
        *,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        db: Database | None = None,
        repo: Repository | None = None,
        adapter: Adapter | None = None,
        prompts: PromptStore | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._cfg = cfg
        self._clock = clock if clock is not None else RealClock()
        self._owned: list[Any] = []

        # LLM client: injected, or one OpenAIClient built from the config
        # (it implements both the LLMClient and Embedder protocols).
        self._llm: LLMClient = llm if llm is not None else OpenAIClient(cfg.llm, clock=self._clock)
        if llm is None:
            self._owned.append(self._llm)
        self._embedder: Embedder | None = embedder
        if self._embedder is None and hasattr(self._llm, "embed"):
            self._embedder = cast(Embedder, self._llm)
        # The LIVE provider vector dimension measured by the embed probe
        # (None when the probe was skipped or failed): the semantic probe
        # compares the active generation's dim against it, so a generation
        # built under a different embedding dimension is never reported
        # healthy.
        self._embed_dim: int | None = None

        # Storage: injected, or a Database + SqliteRepository from the config.
        self._db: Database = db if db is not None else Database(cfg.storage.db_path)
        if db is None:
            self._owned.append(self._db)
        self._repo: Repository = repo if repo is not None else SqliteRepository(self._db)

        # Adapter: injected, or the adapter selected by the config (the
        # console adapter by default; the OneBot v11 bridge when
        # ``adapter.name == "onebot"``).
        self._adapter: Adapter = (
            adapter if adapter is not None else _build_adapter(cfg, self._clock)
        )
        if adapter is None:
            self._owned.append(self._adapter)

        # Prompts: injected, or the configured user dir over package defaults.
        self._prompts = prompts if prompts is not None else PromptStore(cfg.bot.prompt_dir)

    # ── entry point ─────────────────────────────────────────────────────────

    async def run(self) -> DoctorReport:
        """Run every probe, collecting ALL results (never fail fast), then
        close the seams this doctor built. Owned seams are ALWAYS closed —
        even when a probe raises or the run is cancelled (the cleanup runs
        in a ``finally``)."""
        results: list[ProbeResult] = []
        try:
            for name in _PROBE_ORDER:
                probe = getattr(self, f"_probe_{name}")
                try:
                    result = await probe()
                except Exception as e:  # belt-and-suspenders: probes self-contain
                    result = ProbeResult(name, "fail", error=self._scrub(str(e)))
                results.append(result)
        finally:
            await self._cleanup()
        status = "fail" if any(r.status == "fail" for r in results) else "ok"
        counts = Counter(r.status for r in results)
        parts = [f"{counts[s]} {s}" for s in ("ok", "fail", "skip") if counts[s]]
        summary = f"{len(results)} probes: " + ", ".join(parts)
        return DoctorReport(probes=tuple(results), status=status, summary=summary)

    # ── probes ──────────────────────────────────────────────────────────────

    async def _probe_config(self) -> ProbeResult:
        cfg = self._cfg
        profiles = sorted(cfg.llm.profiles)
        problems: list[str] = []
        for name, prof in cfg.llm.profiles.items():
            if not prof.base_url:
                problems.append(f"profile {name!r}: empty base_url")
            if not prof.model:
                problems.append(f"profile {name!r}: empty model")
        # A configured embed profile must carry an explicit revision to form
        # a canonical embedding space; without one semantic recall degrades
        # to FTS-only. Report it (never a secret) as invalid semantic config.
        embed = cfg.llm.profiles.get("embed")
        if embed is not None and not embed.revision:
            problems.append(
                "profile 'embed': missing revision (semantic recall degrades"
                " to FTS-only; set llm.profiles.embed.revision)"
            )
        # Validate the output pipeline for the global config AND every merged
        # chat override now. Unknown/unsafe stage ordering must never wait for
        # an agent reply to surface at runtime.
        from pretender.output.pipeline import OutputPipeline

        output_configs = [("default", cfg.output)] + [
            (str(chat.key), cfg.for_chat(chat.key).output) for chat in cfg.chats
        ]
        for label, output in output_configs:
            try:
                OutputPipeline(output).validate()
            except ConfigError as exc:
                problems.append(f"output {label!r}: {exc}")
        if problems:
            return ProbeResult(
                "config", "fail", detail="; ".join(problems), data={"profiles": profiles}
            )
        no_key = sorted(n for n, p in cfg.llm.profiles.items() if not p.api_key)
        detail = f"{len(profiles)} profile(s) configured"
        if no_key:
            detail += f"; no api_key on: {', '.join(no_key)}"
        # Phase 6 P6.6 plugin config facts (redacted: plugin-owned per-chat
        # cfg_json values are never reported).
        plugin_facts: dict[str, Any] = {
            "paths": list(cfg.plugins.paths),
            "entry_points": list(cfg.plugins.entry_points),
            "allow_replace": list(cfg.plugins.allow_replace),
            "hook_timeout_s": cfg.plugins.hook_timeout_s,
        }
        return ProbeResult(
            "config",
            "ok",
            detail=detail,
            data={
                "profiles": profiles,
                "chats": len(cfg.chats),
                "plugins": plugin_facts,
            },
        )

    async def _probe_prompts(self) -> ProbeResult:
        store = self._prompts
        problems: list[str] = []
        for name in _REQUIRED_PROMPTS:
            try:
                text = store.load(name)
            except PromptError as e:
                problems.append(f"{name}: {e}")
                continue
            if not text.strip():
                problems.append(f"{name}: empty")
        if not problems:
            for name, variables in _PROMPT_VARIABLES.items():
                try:
                    store.render(name, **variables)
                except PromptError as e:
                    problems.append(f"{name}: render failed: {e}")
        # The configured bot identity file must load through the prompt
        # infrastructure: a missing, unreadable, or empty (dead) identity
        # file is a broken asset, not a doctor bug.
        load_identity = getattr(store, "load_identity", None)
        if load_identity is not None:
            try:
                load_identity(self._cfg.bot.identity_file)
            except PromptError as e:
                problems.append(f"identity: {e}")
        if problems:
            return ProbeResult("prompts", "fail", detail="; ".join(problems))
        return ProbeResult(
            "prompts",
            "ok",
            detail=f"{len(_REQUIRED_PROMPTS)} assets load and render",
            data={"assets": list(_REQUIRED_PROMPTS)},
        )

    async def _probe_database(self) -> ProbeResult:
        db = self._db
        try:
            open_ = getattr(db, "open", None)
            if open_ is not None:
                await open_()

            # Writable: a probe kv row round-trips and is removed in the SAME
            # transaction, so a doctor run leaves no residue.
            def write_probe(conn: Any) -> str | None:
                conn.execute(
                    "INSERT INTO kv(k, v) VALUES (?, ?)"
                    " ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                    ("doctor.probe", "1"),
                )
                row = conn.execute(
                    "SELECT v FROM kv WHERE k = ?", ("doctor.probe",)
                ).fetchone()
                conn.execute("DELETE FROM kv WHERE k = ?", ("doctor.probe",))
                return row[0] if row is not None else None

            value = await db.write(write_probe)
            if value != "1":
                return ProbeResult("database", "fail", detail="write probe did not round-trip")

            # FTS5: the virtual table exists AND a MATCH query executes.
            def fts_probe(conn: Any) -> int | None:
                row = conn.execute(
                    "SELECT name FROM sqlite_master"
                    " WHERE type = 'table' AND name = 'message_fts'"
                ).fetchone()
                if row is None:
                    return None
                conn.execute(
                    "SELECT count(*) FROM message_fts WHERE message_fts MATCH ?",
                    ("probe",),
                ).fetchone()
                return int(conn.execute("PRAGMA user_version").fetchone()[0])

            version = await db.read(fts_probe)
            if version is None:
                return ProbeResult("database", "fail", detail="FTS5 message_fts table missing")

            # Repository seam: one typed read proves the repo works end to end.
            stats = await self._repo.stats()
            return ProbeResult(
                "database",
                "ok",
                detail="writable; FTS5 present",
                data={"fts5": True, "user_version": version, "stats": stats},
            )
        except Exception as e:
            return ProbeResult("database", "fail", error=self._scrub(str(e)))

    async def _probe_adapter(self) -> ProbeResult:
        adapter = self._adapter
        name = getattr(adapter, "name", None)
        caps = getattr(adapter, "capabilities", None)
        if name is None or caps is None:
            return ProbeResult("adapter", "fail", detail="adapter missing name/capabilities")
        try:
            connect = getattr(adapter, "connect", None)
            if connect is not None:
                await connect()
            data: dict[str, Any] = {"name": name, "capabilities": sorted(caps)}
            detail = f"{name} connected"
            # Wait for a REAL readiness/identity/API handshake — not merely a
            # listening/background task. A console-style adapter (no
            # ``connected`` property) is ready immediately after connect; a
            # OneBot-style adapter must report an open connection within the
            # probe budget, else the doctor reports NOT-READY (a hard
            # failure, never a silent ok).
            connected = getattr(adapter, "connected", None)
            if connected is not None or getattr(adapter, "ready", None) is not None:
                ready = await self._wait_adapter_ready(adapter, timeout_s=5.0)
                data["ready"] = ready
                if not ready:
                    return ProbeResult(
                        "adapter", "fail",
                        detail=f"{name} connected but NOT ready (no handshake"
                        " within 5s)",
                        data=data,
                    )
                detail = f"{name} connected and ready"
            generation = getattr(adapter, "generation", None)
            # Validate the PROTOCOL with a benign API round-trip (OneBot
            # only): an open socket is not validated readiness — the platform
            # must actually answer. A failed probe is a hard failure, never a
            # false-success report.
            if name == "onebot":
                protocol_ok, protocol_detail = await self._probe_protocol(adapter)
                data["protocol"] = protocol_ok
                data["protocol_detail"] = protocol_detail
                if not protocol_ok:
                    return ProbeResult(
                        "adapter", "fail",
                        detail=f"{name} connected but protocol probe failed:"
                        f" {protocol_detail}",
                        data=data,
                    )
                if generation is not None and (
                    getattr(adapter, "generation", None) != generation
                    or not bool(getattr(adapter, "ready", True))
                ):
                    return ProbeResult(
                        "adapter", "fail",
                        detail=f"{name} connection changed during protocol probe",
                        data=data,
                    )
                detail = f"{name} connected, ready, protocol ok"
                # OneBot/media readiness: verify the media pipeline can
                # normalize a tiny in-memory image (no network, no side
                # effects) and report the media store's presence. A broken
                # media pipeline is a hard failure for the OneBot adapter (it
                # normalizes inbound media).
                media_ok, media_detail = self._probe_media(adapter)
                data["media"] = media_ok
                data["media_detail"] = media_detail
                if not media_ok:
                    return ProbeResult(
                        "adapter", "fail", detail=f"{name} connected; {media_detail}",
                        data=data,
                    )
                detail = f"{name} connected, ready, protocol ok; {media_detail}"
            result = ProbeResult("adapter", "ok", detail=detail, data=data)
        except Exception as e:
            result = ProbeResult("adapter", "fail", error=self._scrub(str(e)))
        return result

    async def _probe_protocol(self, adapter: Adapter) -> tuple[bool, str]:
        """A benign OneBot API round-trip proving the protocol works end to
        end — an open socket (``connected``) is not validated readiness.
        Returns ``(ok, detail)``; the detail is scrubbed of secrets."""
        call = getattr(adapter, "call", None)
        if call is None:
            return False, "adapter has no call surface"
        try:
            await asyncio.wait_for(call("get_login_info"), timeout=5.0)
            return True, "protocol ok"
        except Exception as e:
            return False, self._scrub(str(e))

    async def _wait_adapter_ready(
        self, adapter: Adapter, *, timeout_s: float = 5.0
    ) -> bool:
        """Wait up to ``timeout_s`` for the adapter to report a real
        readiness/handshake (its ``connected`` property). Returns False when
        the handshake does not complete in time — the doctor then reports
        not-ready instead of a silent ok."""
        if getattr(adapter, "ready", None) is not None:
            deadline = self._clock.now() + timeout_s
            while self._clock.now() < deadline:
                if bool(getattr(adapter, "ready", None)):
                    return True
                await self._clock.sleep(0.1)
            return bool(getattr(adapter, "ready", None))
        if getattr(adapter, "connected", None) is None:
            return True
        deadline = self._clock.now() + timeout_s
        while self._clock.now() < deadline:
            if bool(getattr(adapter, "connected", None)):
                return True
            await self._clock.sleep(0.1)
        return bool(getattr(adapter, "connected", None))

    def _probe_media(self, adapter: Adapter) -> tuple[bool, str]:
        """A safe, network-free media-readiness check for the OneBot
        adapter: normalize a tiny in-memory image through its MediaStore to
        prove the Pillow pipeline works. Returns ``(ok, detail)``."""
        media = getattr(adapter, "_media", None)
        if media is None:
            return False, "media: no media store"
        try:
            import io

            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (2, 2), (255, 0, 0)).save(buf, format="PNG")
            asset = media._normalize(buf.getvalue(), "probe://tiny.png")
            return True, f"media ok ({asset.width}x{asset.height})"
        except Exception as e:
            return False, f"media failed: {self._scrub(str(e))}"

    async def _probe_llm_chat(self) -> ProbeResult:
        profile = self._chat_profile()
        if profile is None:
            return ProbeResult("llm_chat", "skip", detail="no LLM profile configured")
        try:
            resp = await self._llm.complete(
                [
                    TranscriptMessage(role="system", content="You are a connectivity probe."),
                    TranscriptMessage(role="user", content="Reply with exactly: pong"),
                ],
                profile=profile,
                max_tokens=16,
            )
            if resp.content is None and not resp.tool_calls:
                return ProbeResult(
                    "llm_chat", "fail", detail=f"profile {profile!r} returned an empty response"
                )
            return ProbeResult(
                "llm_chat",
                "ok",
                detail=f"profile {profile!r} completed",
                data={"profile": profile, "finish_reason": resp.finish_reason},
            )
        except Exception as e:
            return ProbeResult(
                "llm_chat", "fail", detail=f"profile {profile!r}", error=self._scrub(str(e))
            )

    async def _probe_llm_tools(self) -> ProbeResult:
        profile = self._chat_profile()
        if profile is None:
            return ProbeResult("llm_tools", "skip", detail="no LLM profile configured")
        try:
            resp = await self._llm.complete(
                [
                    TranscriptMessage(
                        role="system",
                        content="You are a tool-calling probe. Always call the provided tool.",
                    ),
                    TranscriptMessage(role="user", content="Call the doctor_probe tool now."),
                ],
                profile=profile,
                tools=[_PROBE_TOOL],
                max_tokens=64,
            )
            if not resp.tool_calls:
                return ProbeResult(
                    "llm_tools", "fail", detail=f"profile {profile!r} returned no tool call"
                )
            bad = [c for c in resp.tool_calls if not c.name or not isinstance(c.arguments, dict)]
            if bad:
                return ProbeResult(
                    "llm_tools", "fail", detail=f"profile {profile!r} returned malformed tool call(s)"
                )
            # Analysis/content and tool calls must be able to coexist in one
            # response (the planner's analysis + tool decision turn): a
            # response carrying BOTH is valid, never a failure.
            detail = f"profile {profile!r} returned {len(resp.tool_calls)} tool call(s)"
            if resp.content:
                detail += " with analysis content"
            return ProbeResult(
                "llm_tools",
                "ok",
                detail=detail,
                data={"profile": profile, "tool_calls": [c.name for c in resp.tool_calls]},
            )
        except Exception as e:
            return ProbeResult(
                "llm_tools", "fail", detail=f"profile {profile!r}", error=self._scrub(str(e))
            )

    async def _probe_vision(self) -> ProbeResult:
        if "vision" not in self._cfg.llm.profiles:
            return ProbeResult("vision", "skip", detail="no vision profile configured")
        try:
            msgs = [
                TranscriptMessage(
                    role="system",
                    content=(
                        "You are a vision safety probe. Return exactly one JSON object "
                        'with boolean "safe" and a short one-line "description"; '
                        "do not return prose, URLs, paths, or base64."
                    ),
                ),
                TranscriptMessage(
                    role="user",
                    content=render_image_markdown("probe", "https://example.com/probe.png"),
                ),
            ]
            # Verify the REAL wire shape: the image markdown must serialize
            # into OpenAI-compatible multimodal content parts (text +
            # image_url), not a bare markdown string.
            wire = serialize(msgs)
            parts = wire[1].get("content")
            if not (
                isinstance(parts, list)
                and any(
                    isinstance(p, dict)
                    and p.get("type") == "image_url"
                    and isinstance(p.get("image_url"), dict)
                    and "url" in p["image_url"]
                    for p in parts
                )
            ):
                return ProbeResult(
                    "vision", "fail", detail="vision wire is not multimodal"
                )
            resp = await self._llm.complete(msgs, profile="vision", max_tokens=32)
            if resp.content is None and not resp.tool_calls:
                return ProbeResult("vision", "fail", detail="vision profile returned an empty response")
            verdict = parse_vision_result(resp.content)
            if not verdict.safe:
                return ProbeResult(
                    "vision",
                    "fail",
                    detail="vision profile returned malformed or unapproved structured output",
                )
            return ProbeResult(
                "vision", "ok", detail="vision profile completed", data={"profile": "vision"}
            )
        except Exception as e:
            return ProbeResult("vision", "fail", detail="vision profile", error=self._scrub(str(e)))

    async def _probe_embed(self) -> ProbeResult:
        if "embed" not in self._cfg.llm.profiles:
            return ProbeResult("embed", "skip", detail="no embed profile configured")
        if self._embedder is None:
            # An injected LLMClient without embed support: build an embedder.
            self._embedder = OpenAIClient(self._cfg.llm, clock=self._clock)
            self._owned.append(self._embedder)
        try:
            vectors = await self._embedder.embed(["doctor probe"])
            if not vectors:
                return ProbeResult("embed", "fail", detail="embed profile returned no vectors")
            dims = {len(v) for v in vectors}
            if len(dims) != 1 or not dims or next(iter(dims)) <= 0:
                return ProbeResult(
                    "embed", "fail", detail=f"inconsistent dimension {sorted(dims)}"
                )
            if len(vectors) != 1:
                return ProbeResult(
                    "embed", "fail",
                    detail=f"embed profile returned {len(vectors)} vectors for one input",
                )
            if any(
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for vector in vectors for value in vector
            ):
                return ProbeResult("embed", "fail", detail="embed profile returned non-finite vector")
            if any(
                math.fsum(float(value) * float(value) for value in vector) <= 0.0
                for vector in vectors
            ):
                return ProbeResult("embed", "fail", detail="embed profile returned zero vector")
            dim = next(iter(dims))
            self._embed_dim = dim  # the live provider dimension (semantic probe input)
            prof = self._cfg.llm.profiles.get("embed")
            space = prof.space_id() if prof is not None else None
            data: dict[str, Any] = {"dimension": dim}
            if space is not None:
                data["space_id"] = space
            return ProbeResult("embed", "ok", detail=f"dimension {dim}", data=data)
        except Exception as e:
            return ProbeResult("embed", "fail", detail="embed profile", error=self._scrub(str(e)))

    async def _probe_semantic(self) -> ProbeResult:
        """Semantic readiness: FTS-only vs a real active matching generation.

        - No embed profile -> ``skip`` (FTS-only recall).
        - A configured embed profile without an explicit revision -> ``fail``
          (no canonical space; semantic recall degrades to FTS-only).
        - A configured embed profile whose live provider dimension cannot be
          measured (a blocked/unavailable embedder) -> ``skip`` (FTS-only
          recall — the runtime degrades to FTS-only on a blocked/degraded
          embed service).
        - A configured revision but NO active generation matching the space ->
          ``fail`` (degraded/not-ready — never a false healthy claim).
        - An active matching generation whose model/revision do NOT match the
          configured embed profile -> ``fail`` (a stale/foreign generation).
        - An active matching generation whose dim does NOT match the LIVE
          provider vector dimension -> ``fail`` (a stale generation built
          under a different embedding dimension).
        - An active matching generation with COMPLETE, USABLE vector coverage
          (every memory record across all chats has a vector in it whose dim
          matches the generation, whose ``source_hash`` matches the memory's
          current source hash, and whose float32 values are finite with a
          nonzero norm) -> ``ok``.
        - Anything less (partial coverage, dimension mismatch, stale
          source_hash, zero/non-finite vectors) -> ``fail`` (never a false
          healthy partial/bad generation).

        The repository must expose the knowledge surface; a repo that does not
        (e.g. a minimal fake) yields ``skip`` — the doctor cannot assess
        readiness without it. Secret-safe: only the space_id/dim/count are
        reported.
        """
        embed = self._cfg.llm.profiles.get("embed")
        if embed is None:
            return ProbeResult("semantic", "skip", detail="no embed profile (FTS-only)")
        space = embed.space_id()
        if space is None:
            return ProbeResult(
                "semantic", "fail",
                detail="embed profile missing revision (semantic recall degrades"
                " to FTS-only; set llm.profiles.embed.revision)",
            )
        if not hasattr(self._repo, "list_embedding_generations"):
            return ProbeResult(
                "semantic", "skip",
                detail="repository lacks the knowledge surface (cannot assess)",
            )
        repo: Any = self._repo
        try:
            # The LIVE provider vector dimension, measured by the embed
            # probe. A blocked/unavailable embed profile (no measurement)
            # degrades semantic recall to FTS-only: skip, never a false
            # claim.
            live_dim = self._embed_dim
            if live_dim is None:
                return ProbeResult(
                    "semantic", "skip",
                    detail="embed profile blocked/unavailable (FTS-only)",
                )
            gens = await repo.list_embedding_generations()
            active = [g for g in gens if g.state == "active" and g.space_id == space]
            if not active:
                return ProbeResult(
                    "semantic", "fail",
                    detail=f"no active matching generation for space {space!r}"
                    " (semantic not ready)",
                    data={"space_id": space, "active": False},
                )
            gen = active[0]
            # The active generation must match the configured model/revision
            # (space already matched). A mismatch is a stale/foreign
            # generation — never a false healthy claim.
            if gen.model != embed.model or gen.revision != embed.revision:
                return ProbeResult(
                    "semantic", "fail",
                    detail=f"active generation {gen.id} model/revision mismatch"
                    f" ({gen.model}@{gen.revision} != {embed.model}@{embed.revision})",
                    data={"space_id": space, "active": True, "dim": gen.dim},
                )
            # The active generation must match the LIVE provider dimension:
            # a generation built under a different embedding dimension is
            # stale — never a false healthy claim.
            if gen.dim != live_dim:
                return ProbeResult(
                    "semantic", "fail",
                    detail=f"active generation {gen.id} dimension {gen.dim} != live"
                    f" provider dimension {live_dim}",
                    data={"space_id": space, "active": True, "dim": gen.dim},
                )
            # Complete USABLE vector coverage: EVERY memory record across
            # all chats must have a vector in the active generation whose
            # dim matches the generation, whose source_hash matches the
            # memory's CURRENT source hash, and whose float32 values are
            # finite with a nonzero norm. A partial/stale/bad generation is
            # never healthy.
            chats = await repo.list_memory_chats()
            total_vectors = 0
            missing: list[str] = []
            dim_mismatch: list[str] = []
            stale: list[str] = []
            bad: list[str] = []
            for chat in chats:
                memories = [m for m in await repo.list_memories(chat) if m.text]
                vectors = await repo.list_vectors(chat, gen.model, gen.id)
                total_vectors += len(vectors)
                mem_ids = {m.id for m in memories}
                vec_ids = {v.owner_id for v in vectors}
                if not mem_ids <= vec_ids:
                    missing.append(f"{chat}:{len(mem_ids - vec_ids)}")
                by_id = {v.owner_id: v for v in vectors}
                for mem in memories:
                    row = by_id.get(mem.id)
                    if row is None:
                        continue  # already counted as missing
                    if row.dim != gen.dim:
                        dim_mismatch.append(f"{chat}:{mem.id}")
                    elif row.source_hash != mem.source_hash:
                        stale.append(f"{chat}:{mem.id}")
                    elif not self._vector_usable(row):
                        bad.append(f"{chat}:{mem.id}")
            if missing:
                return ProbeResult(
                    "semantic", "fail",
                    detail=f"active generation {gen.id} has incomplete vector"
                    f" coverage (missing {', '.join(missing)})",
                    data={
                        "space_id": space, "active": True, "dim": gen.dim,
                        "vectors": total_vectors,
                    },
                )
            if dim_mismatch:
                return ProbeResult(
                    "semantic", "fail",
                    detail=f"active generation {gen.id} has dimension mismatch"
                    f" (dim {gen.dim}; {', '.join(dim_mismatch)})",
                    data={
                        "space_id": space, "active": True, "dim": gen.dim,
                        "vectors": total_vectors,
                    },
                )
            if stale:
                return ProbeResult(
                    "semantic", "fail",
                    detail=f"active generation {gen.id} has stale vectors"
                    f" (source_hash mismatch: {', '.join(stale)})",
                    data={
                        "space_id": space, "active": True, "dim": gen.dim,
                        "vectors": total_vectors,
                    },
                )
            if bad:
                return ProbeResult(
                    "semantic", "fail",
                    detail=f"active generation {gen.id} has unusable vectors"
                    f" (zero/non-finite: {', '.join(bad)})",
                    data={
                        "space_id": space, "active": True, "dim": gen.dim,
                        "vectors": total_vectors,
                    },
                )
            return ProbeResult(
                "semantic", "ok",
                detail=f"active generation {gen.id} (dim {gen.dim}) complete",
                data={
                    "space_id": space, "active": True, "dim": gen.dim,
                    "vectors": total_vectors,
                },
            )
        except Exception as e:
            return ProbeResult("semantic", "fail", error=self._scrub(str(e)))

    async def _probe_learner(self) -> ProbeResult:
        """Learner worker readiness (Phase 6 P6.4): truthfully reports
        disabled / blocked / missing-prompt / ready WITHOUT any provider
        call — only the config, the prompt store, and the repository's
        adaptive surface are consulted.

        - ``learn.enabled = False`` -> ``skip`` (an optional capability).
        - Enabled with NO configured profiles -> ``fail`` (blocked).
        - A profile naming an unknown learner -> ``fail`` (blocked).
        - A profile whose spec's prompt file does not load -> ``fail``
          (missing-prompt).
        - Enabled with at least one valid profile -> ``ok`` (the bounded
          worker would start in LIVE mode).
        """
        cfg = self._cfg
        if not cfg.learn.enabled:
            return ProbeResult("learner", "skip", detail="learn disabled")
        profiles = cfg.learn.profiles
        if not profiles:
            return ProbeResult(
                "learner", "fail",
                detail="learn enabled but no profiles configured (blocked)",
            )
        from pretender.learn import SPECS

        problems: list[str] = []
        valid = 0
        for name in sorted(profiles):
            spec = SPECS.get(name)
            if spec is None:
                problems.append(f"profile {name!r}: unknown learner")
                continue
            try:
                self._prompts.load(spec.prompt)
            except PromptError as e:
                problems.append(f"profile {name!r}: missing prompt: {e}")
                continue
            valid += 1
        data: dict[str, Any] = {
            "profiles": sorted(profiles),
            "valid": valid,
        }
        if valid == 0:
            detail = "blocked: no valid learner profiles"
            if problems:
                detail += " (" + "; ".join(problems) + ")"
            return ProbeResult(
                "learner", "fail", detail=detail, data=data
            )
        if problems:
            return ProbeResult(
                "learner", "fail", detail="; ".join(problems), data=data
            )
        return ProbeResult(
            "learner", "ok",
            detail=f"worker ready ({valid} profile(s))", data=data,
        )

    async def _probe_media_catalog(self) -> ProbeResult:
        """Media catalog readiness (Phase 6 P6.5b): truthfully reports
        disabled / empty / ready and the live vision prerequisites WITHOUT
        any provider call — only the config and the local media_assets table
        are consulted.

        - ``media.enabled = False`` -> ``skip`` (an optional capability).
        - Enabled but the media_assets table cannot be read -> ``fail``
          (cannot assess).
        - Enabled with zero approved assets -> ``ok`` with detail "catalog
          enabled but empty".
        - Enabled with approved assets -> ``ok`` with detail "catalog
          ready (N approved asset(s))".
        - Vision prerequisites ride in the detail/data: the configured
          ``vision_profile`` (must exist under llm.profiles — the config
          already validates this) and whether the background learner budget
          is enabled (the harvest vision lane is budget-admitted through
          it). A missing vision profile means harvested assets stay PENDING
          (unapproved). Approval is STRICT: a candidate is approved only on
          an explicit structured ``safe: true`` classification with a valid
          bounded escaped one-line description — never from arbitrary text.
        """
        cfg = self._cfg
        if not cfg.media.enabled:
            return ProbeResult("media_catalog", "skip", detail="media catalog disabled")
        db = self._db

        def count_media(conn: Any) -> dict[str, int]:
            rows = conn.execute(
                "SELECT safety_status, COUNT(*) FROM media_assets"
                " GROUP BY safety_status"
            ).fetchall()
            return {str(r[0]): int(r[1]) for r in rows}

        try:
            raw = await db.read(count_media)
        except Exception as e:
            return ProbeResult(
                "media_catalog", "fail", error=self._scrub(str(e))
            )
        counts = raw if isinstance(raw, dict) else {}
        approved = int(counts.get("approved", 0))
        pending = int(counts.get("pending", 0))
        vision = cfg.media.vision_profile
        vision_ready = vision is not None and vision in cfg.llm.profiles
        budget_ready = cfg.learn.enabled
        data: dict[str, Any] = {
            "approved": approved,
            "pending": pending,
            "vision_profile": vision,
            "vision_ready": vision_ready,
            "learner_budget": budget_ready,
            "strict_vision": True,
        }
        detail = (
            f"catalog ready ({approved} approved asset(s))"
            if approved > 0
            else "catalog enabled but empty (no approved assets)"
        )
        prereqs: list[str] = []
        if not vision_ready:
            prereqs.append("no vision profile (harvested assets stay pending)")
        if not budget_ready:
            prereqs.append("learn disabled (no background budget for vision)")
        if vision_ready:
            prereqs.append(
                "vision approval requires a structured safe=true classification"
            )
        if prereqs:
            detail += "; " + "; ".join(prereqs)
        return ProbeResult("media_catalog", "ok", detail=detail, data=data)

    async def _probe_plugins(self) -> ProbeResult:
        """Plugin preflight (Phase 6 P6.6): truthfully reports the
        configured explicit-trust plugin surface WITHOUT importing any
        unconfigured code — only the configured ``plugins.paths`` module
        files and the explicit ``plugins.entry_points`` names are resolved.
        ``setup`` is NEVER called (the doctor is a preflight; setup has
        runtime side effects).

        - No plugins configured -> ``skip`` (an optional capability).
        - A configured path/entry point that cannot be resolved/imported ->
          ``fail`` (blocked).
        - A resolved plugin without a valid ``name`` -> ``fail``.
        - All configured sources resolve -> ``ok`` with the loaded
          fingerprint (the plugin names in deterministic order).
        """
        cfg = self._cfg
        if not cfg.plugins.paths and not cfg.plugins.entry_points:
            return ProbeResult("plugins", "skip", detail="no plugins configured")
        from pretender.registry import load_plugin_entry_point, load_plugin_module

        problems: list[str] = []
        names: list[str] = []
        for path in cfg.plugins.paths:
            try:
                module = load_plugin_module(path)
            except Exception as e:
                problems.append(f"{path}: {self._scrub(str(e))}")
                continue
            name = getattr(module, "name", None)
            if not isinstance(name, str) or not name.strip():
                problems.append(f"{path}: plugin must expose a 'name' string")
            else:
                names.append(name)
        for ep in cfg.plugins.entry_points:
            try:
                plugin = load_plugin_entry_point(ep)
            except Exception as e:
                problems.append(f"entry point {ep!r}: {self._scrub(str(e))}")
                continue
            name = getattr(plugin, "name", None)
            if not isinstance(name, str) or not name.strip():
                problems.append(
                    f"entry point {ep!r}: plugin must expose a 'name' string"
                )
            else:
                names.append(name)
        data: dict[str, Any] = {
            "paths": list(cfg.plugins.paths),
            "entry_points": list(cfg.plugins.entry_points),
            "allow_replace": list(cfg.plugins.allow_replace),
            "hook_timeout_s": cfg.plugins.hook_timeout_s,
            "names": names,
        }
        if problems:
            return ProbeResult(
                "plugins", "fail", detail="; ".join(problems), data=data
            )
        return ProbeResult(
            "plugins",
            "ok",
            detail=f"{len(names)} plugin(s) load ({', '.join(names)})",
            data=data,
        )

    async def _probe_chat_controls(self) -> ProbeResult:
        """Chat-control readiness (Phase 6 P6.6b): truthfully reports the
        durable surface WITHOUT any write — only the local ``chat_controls``
        table and the repository's chat-control surface are consulted.

        - The repository lacks the chat-control surface -> ``skip`` (cannot
          assess — the same pattern as the semantic probe).
        - The ``chat_controls`` table is missing -> ``fail``.
        - The surface is present -> ``ok`` with the active control count.
        """
        if not hasattr(self._repo, "apply_chat_control") or not hasattr(
            self._repo, "list_active_controls"
        ):
            return ProbeResult(
                "chat_controls", "skip",
                detail="repository lacks the chat-control surface (cannot assess)",
            )
        db = self._db

        def count_controls(conn: Any) -> dict[str, Any] | None:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
                " AND name = 'chat_controls'"
            ).fetchone()
            if row is None:
                return None
            total = int(
                conn.execute("SELECT COUNT(*) FROM chat_controls").fetchone()[0]
            )
            return {"total": total}

        try:
            raw = await db.read(count_controls)
        except Exception as e:
            return ProbeResult("chat_controls", "fail", error=self._scrub(str(e)))
        if raw is None:
            return ProbeResult(
                "chat_controls", "fail", detail="chat_controls table missing"
            )
        total = int(raw.get("total", 0)) if isinstance(raw, dict) else 0
        return ProbeResult(
            "chat_controls",
            "ok",
            detail=f"surface ready ({total} control row(s))",
            data={"controls": total},
        )

    # ── helpers ─────────────────────────────────────────────────────────────

    def _chat_profile(self) -> str | None:
        """The profile the chat/tool probes use: the bot's planner lane first,
        then any configured profile; None when nothing is configured."""
        profiles = self._cfg.llm.profiles
        for name in _CHAT_PROFILE_PREFERENCE:
            if name in profiles:
                return name
        if profiles:
            return sorted(profiles)[0]
        return None

    def _vector_usable(self, row: Any) -> bool:
        """A stored vector is usable when its float32 values are all finite
        and its norm is nonzero (a zero vector matches nothing)."""
        import struct

        try:
            values = struct.unpack_from(f"<{row.dim}f", row.blob)
        except (struct.error, TypeError):
            return False
        if not all(math.isfinite(v) for v in values):
            return False
        return math.fsum(v * v for v in values) > 0.0

    def _scrub(self, text: str) -> str:
        """Scrub credentials from a probe error before it is recorded.

        Strips query/fragment credentials from any embedded URL (provider
        errors and transport messages often echo the request URL) and masks
        every configured api_key — defense in depth, since the LLM layer
        already redacts its own errors."""
        text = scrub_credentials(text)
        for prof in self._cfg.llm.profiles.values():
            if prof.api_key:
                text = text.replace(prof.api_key, "***")
        return text

    async def _cleanup(self) -> None:
        """Close every seam this doctor built (never injected ones)."""
        for obj in self._owned:
            close = getattr(obj, "aclose", None) or getattr(obj, "close", None)
            if close is None:
                continue
            try:
                await close()
            except Exception:
                pass
