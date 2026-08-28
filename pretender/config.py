"""Configuration: frozen dataclasses, TOML loading, ${ENV} expansion.

Rules from PLAN.md §6:
  - An empty TOML must boot a working bot — every field has a dataclass default.
  - Secrets are ${ENV} references only, never literal values. A reference to
    an unset variable is a ConfigError at load time.
  - A config key that nothing reads by name at runtime does not exist:
    unknown keys are rejected at load, so `doctor` can later enforce that
    every schema key is actually read.
  - Per-chat overrides ([[chats]]) merge over the top-level sections.
  - RuntimeOverlay applies dotted-path overrides on top of a frozen Config
    without rewriting any file.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import math
import os
import re
import tomllib
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin

from pretender.errors import ConfigError
from pretender.types import ChatKey

_ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


# ── Numeric validation (fail closed at load) ────────────────────────────────

def _check_positive_int(name: str, value: Any) -> None:
    """A strictly positive integer (bools are not ints here)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")
    if value <= 0:
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")


def _check_nonneg_int(name: str, value: Any) -> None:
    """A nonnegative integer (bools are not ints here)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be a nonnegative integer, got {value!r}")
    if value < 0:
        raise ConfigError(f"{name} must be a nonnegative integer, got {value!r}")


def _check_finite_nonneg(name: str, value: Any) -> None:
    """A finite nonnegative number (bools are not numbers here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a finite nonnegative number, got {value!r}")
    if not math.isfinite(value) or value < 0:
        raise ConfigError(f"{name} must be a finite nonnegative number, got {value!r}")


def _check_positive_number(name: str, value: Any) -> None:
    """A finite strictly positive number (bools are not numbers here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a positive number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{name} must be a positive number, got {value!r}")


def _is_local_host(host: str) -> bool:
    """True for an EXPLICITLY local-only host: a loopback/private/link-local
    IP literal or a ``localhost`` hostname. A DNS name that is not an IP
    literal is never local-only (it may resolve anywhere), so a plaintext
    forward endpoint with a bearer token must not point at one."""
    host = host.strip().lower().rstrip(".")
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _is_loopback_host(host: str) -> bool:
    """True for an EXPLICITLY loopback-only host: a loopback IP literal or a
    ``localhost`` hostname. A reverse-server bind on a NON-loopback host is
    reachable by remote clients, so it must require authentication."""
    host = host.strip().lower().rstrip(".")
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback


# ── Sections ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BotConfig:
    name: str = "麦麦"
    identity_file: str = "prompts/identity.txt"
    prompt_dir: str = "prompts"  # user prompt dir; overlays package defaults


@dataclass(frozen=True)
class LLMProfile:
    """One named provider profile. Profiles may point at different vendors;
    a profile with only ``model`` set inherits the other defaults.

    ``revision`` is the explicit model revision used to form the embedding
    space identity (``model@revision``) for the ``embed`` profile. An embed
    profile without a revision has no canonical space and degrades to
    FTS-only recall.
    """

    base_url: str = "https://api.deepseek.com/v1"
    api_key: str | None = None
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 1200
    timeout_s: float = 45.0
    revision: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.revision, str):
            raise ConfigError(
                f"llm profile revision must be a string, got {self.revision!r}"
            )
        if self.revision and not self.revision.strip():
            raise ConfigError("llm profile revision must not be blank")

    def space_id(self) -> str | None:
        """The canonical embedding space identity (``model@revision``), or
        None when no revision is configured (no canonical space)."""
        if not self.revision:
            return None
        return f"{self.model}@{self.revision}"


@dataclass(frozen=True)
class LLMConfig:
    profiles: dict[str, LLMProfile] = field(default_factory=dict)

    def profile(self, name: str) -> LLMProfile:
        try:
            return self.profiles[name]
        except KeyError:
            raise ConfigError(
                f"no LLM profile named {name!r} (configured: {sorted(self.profiles)})"
            ) from None


@dataclass(frozen=True)
class GateBackoffConfig:
    base_s: float = 15.0
    cap_s: float = 300.0
    start_count: int = 2

    def __post_init__(self) -> None:
        _check_finite_nonneg("gate.backoff.base_s", self.base_s)
        _check_finite_nonneg("gate.backoff.cap_s", self.cap_s)
        if self.cap_s < self.base_s:
            raise ConfigError(
                f"gate.backoff.cap_s ({self.cap_s!r}) must be >= "
                f"gate.backoff.base_s ({self.base_s!r})"
            )
        _check_nonneg_int("gate.backoff.start_count", self.start_count)


@dataclass(frozen=True)
class GateConfig:
    mode: str = "reply_necessity"  # "reply_necessity" | "frequency"
    threshold: int = 8
    trigger_score: int = 80
    frequency: float = 1.0
    backoff: GateBackoffConfig = field(default_factory=GateBackoffConfig)

    def __post_init__(self) -> None:
        if self.mode not in ("reply_necessity", "frequency"):
            raise ConfigError(
                f"gate.mode must be 'reply_necessity' or 'frequency', got {self.mode!r}"
            )
        _check_positive_int("gate.threshold", self.threshold)
        _check_positive_int("gate.trigger_score", self.trigger_score)
        if isinstance(self.frequency, bool) or not isinstance(
            self.frequency, (int, float)
        ):
            raise ConfigError(
                f"gate.frequency must be a number in [0, 1], got {self.frequency!r}"
            )
        if not math.isfinite(self.frequency) or not (0.0 <= self.frequency <= 1.0):
            raise ConfigError(
                f"gate.frequency must be in [0, 1], got {self.frequency!r}"
            )
        if not isinstance(self.backoff, GateBackoffConfig):
            raise ConfigError(
                "gate.backoff must be a GateBackoffConfig table, got "
                f"{type(self.backoff).__name__}"
            )


@dataclass(frozen=True)
class DriftConfig:
    level: str = "active"      # subtle | active | scattered | wild
    anchor: str = "balanced"   # strict | balanced | loose
    reaction: str = "natural"  # reserved | natural | lively


@dataclass(frozen=True)
class OutputConfig:
    pipeline: tuple[str, ...] = ("sanitize", "split", "typo")
    max_split: int = 3
    typo_rate: float = 0.03


@dataclass(frozen=True)
class ContextConfig:
    """The context-management knobs the pure context lane reads.

    ``max_context_size`` is a message count (trim cuts back to it at 2×);
    ``max_image_num`` is how many newest images survive the image budget
    (older ones become ``[图片]``); ``keep_recent`` is how many completed
    tool turns stay unfolded by ``fold``.
    """

    max_context_size: int = 40
    max_image_num: int = 3
    keep_recent: int = 0

    def __post_init__(self) -> None:
        _check_positive_int("context.max_context_size", self.max_context_size)
        _check_nonneg_int("context.max_image_num", self.max_image_num)
        _check_nonneg_int("context.keep_recent", self.keep_recent)


@dataclass(frozen=True)
class BudgetRung:
    """One degrade rung on the daily budget ladder.

    ``at`` is the fraction of ``daily_cap`` at which the rung engages
    (0..1); ``action`` is ``warn`` | ``degrade`` | ``stop``; ``detail`` is a
    human description. Rungs must be sorted ascending by ``at``.
    """

    at: float = 1.0
    action: str = "stop"
    detail: str = ""

    def __post_init__(self) -> None:
        _check_finite_nonneg("budget.rungs.at", self.at)
        if self.at > 1.0:
            raise ConfigError(f"budget.rungs.at must be in [0, 1], got {self.at!r}")
        if self.action not in ("warn", "degrade", "stop"):
            raise ConfigError(
                f"budget.rungs.action must be 'warn', 'degrade' or 'stop', "
                f"got {self.action!r}"
            )


@dataclass(frozen=True)
class BudgetConfig:
    """Daily LLM budget: a hard per-day call cap plus a ladder of degrade
    rungs that engage as the cap is approached."""

    daily_cap: int = 100
    rungs: tuple[BudgetRung, ...] = (
        BudgetRung(at=0.8, action="warn", detail="warn the operator"),
        BudgetRung(at=0.9, action="degrade", detail="degrade to a cheaper model"),
        BudgetRung(at=1.0, action="stop", detail="hard stop for the day"),
    )

    def __post_init__(self) -> None:
        _check_positive_int("budget.daily_cap", self.daily_cap)
        for r in self.rungs:
            if not isinstance(r, BudgetRung):
                raise ConfigError(
                    "budget.rungs must be a BudgetRung table, got "
                    f"{type(r).__name__}"
                )
        ats = [r.at for r in self.rungs]
        if ats != sorted(ats):
            raise ConfigError(
                "budget.rungs must be sorted ascending by 'at'"
            )


@dataclass(frozen=True)
class AgentConfig:
    """The Phase 3 agent runtime knobs (frozen Oracle advisory).

    ``dispatch_lease_s`` is the finite lease granted to each prepared
    dispatch (the scheduler's default re-arm horizon); ``max_execution_s``
    bounds one agent run; ``retry_delay_s`` is the delay before a retry
    after a transient failure. ``fallback_profile`` names the LLM profile
    the budget-degrade ``profile_fallback`` action falls back to (None
    disables the fallback). This section carries NO secrets — API keys stay
    env-only in ``llm.profiles`` — and per-chat overrides deep-merge exactly
    like every other section.
    """

    dispatch_lease_s: float = 60.0
    max_execution_s: float = 300.0
    retry_delay_s: float = 30.0
    fallback_profile: str | None = None

    def __post_init__(self) -> None:
        _check_positive_number("agent.dispatch_lease_s", self.dispatch_lease_s)
        _check_positive_number("agent.max_execution_s", self.max_execution_s)
        _check_positive_number("agent.retry_delay_s", self.retry_delay_s)
        if self.fallback_profile is not None and not isinstance(
            self.fallback_profile, str
        ):
            raise ConfigError(
                f"agent.fallback_profile must be a string or None, got "
                f"{self.fallback_profile!r}"
            )


@dataclass(frozen=True)
class LearnProfile:
    """One named learner profile under ``[learn.profiles.<name>]``.

    Every field is optional: a profile with only ``cadence_s`` set inherits
    the top-level ``[learn]`` defaults for the rest. ``policy`` is
    ``nonself`` (source reads exclude the bot's own messages) or ``all``.
    """

    cadence_s: int | None = None
    batch_size: int | None = None
    policy: str | None = None
    enabled: bool | None = None

    def __post_init__(self) -> None:
        if self.cadence_s is not None:
            _check_positive_int("learn.profiles.cadence_s", self.cadence_s)
        if self.batch_size is not None:
            _check_positive_int("learn.profiles.batch_size", self.batch_size)
        if self.policy is not None and self.policy not in ("nonself", "all"):
            raise ConfigError(
                f"learn.profiles.policy must be 'nonself' or 'all', got {self.policy!r}"
            )
        if self.enabled is not None and not isinstance(self.enabled, bool):
            raise ConfigError(
                f"learn.profiles.enabled must be a boolean, got {self.enabled!r}"
            )


@dataclass(frozen=True)
class LearnConfig:
    """The Phase 6 adaptive learner knobs (frozen Oracle advisory).

    Disabled by default: ``enabled = False`` boots a working bot with no
    learner wiring. ``cadence_s``/``batch_size`` are the top-level defaults
    named profiles override; ``concurrency`` bounds how many learner runs
    may execute at once and ``foreground_reserve`` the slots reserved for
    foreground (interactive) work — it must be strictly below
    ``concurrency``. This section carries NO secrets and NO plugin loading:
    ``[plugins]`` stays the explicit, path-safe plugin surface.
    """

    enabled: bool = False
    cadence_s: int = 3600
    batch_size: int = 1
    concurrency: int = 1
    foreground_reserve: int = 0
    profiles: dict[str, LearnProfile] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigError(f"learn.enabled must be a boolean, got {self.enabled!r}")
        _check_positive_int("learn.cadence_s", self.cadence_s)
        _check_positive_int("learn.batch_size", self.batch_size)
        _check_positive_int("learn.concurrency", self.concurrency)
        _check_nonneg_int("learn.foreground_reserve", self.foreground_reserve)
        if self.foreground_reserve >= self.concurrency:
            raise ConfigError(
                "learn.foreground_reserve must be strictly below"
                f" learn.concurrency ({self.foreground_reserve} >= {self.concurrency})"
            )
        for name, profile in self.profiles.items():
            if not isinstance(profile, LearnProfile):
                raise ConfigError(
                    "learn.profiles must be a LearnProfile table, got "
                    f"{type(profile).__name__}"
                )


@dataclass(frozen=True)
class MediaConfig:
    """The Phase 6 P6.5 media catalog knobs (frozen Oracle advisory).

    Disabled by default: ``enabled = False`` boots a working bot with no
    media catalog wiring. ``harvest`` gates whether group stickers are
    harvested at all (default False). ``group_nonself_stickers_only``
    restricts harvesting to non-self stickers in group chats (default
    True); ``private_stickers_enabled``/``private_images_enabled`` gate
    private-chat media (default False — private chats and images are
    disabled). ``candidate_cap`` bounds pending candidates per chat
    (1..16); ``capacity`` is the approved-asset cap per (chat, kind);
    ``cooldown_s`` is the per-asset cooldown after use; ``vision_profile``
    names the explicit LLM profile used to describe candidates (None
    disables description). This section carries NO secrets and NO plugin
    loading.
    """

    enabled: bool = False
    harvest: bool = False
    group_nonself_stickers_only: bool = True
    private_stickers_enabled: bool = False
    private_images_enabled: bool = False
    candidate_cap: int = 16
    capacity: int = 32
    cooldown_s: float = 3600.0
    vision_profile: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "harvest",
            "group_nonself_stickers_only",
            "private_stickers_enabled",
            "private_images_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(
                    f"media.{name} must be a boolean, got {getattr(self, name)!r}"
                )
        _check_positive_int("media.candidate_cap", self.candidate_cap)
        if self.candidate_cap > 16:
            raise ConfigError(
                f"media.candidate_cap must be <= 16, got {self.candidate_cap!r}"
            )
        _check_positive_int("media.capacity", self.capacity)
        _check_finite_nonneg("media.cooldown_s", self.cooldown_s)
        if self.vision_profile is not None and (
            not isinstance(self.vision_profile, str)
            or not self.vision_profile.strip()
        ):
            raise ConfigError(
                "media.vision_profile must be a non-empty string or None, got "
                f"{self.vision_profile!r}"
            )


@dataclass(frozen=True)
class OneBotConfig:
    """Connection + runtime knobs for the OneBot v11 adapter.

    ``mode`` is ``reverse_ws`` (default: NapCat dials us — we run a WebSocket
    server) or ``ws`` (we dial a OneBot WebSocket endpoint and auto-reconnect
    with exponential backoff). ``scheme`` is ``ws`` or ``wss`` and only
    matters in forward (``ws``) mode. ``access_token`` is the OneBot v11
    ``Authorization: Bearer <token>`` secret — sent on outbound handshakes and
    verified on inbound ones (None disables auth). It is a ${ENV}-only secret:
    a literal value is a ConfigError at load time. Forward-mode transport is
    fail-closed: a plaintext ``ws://`` endpoint carrying an access token must
    be explicitly local-only (a loopback/private/link-local literal host) —
    a remote plaintext endpoint with a bearer token is a ConfigError, so the
    token never travels in the clear to a remote host.

    ``self_id`` is the bot's own account id (optional): the adapter learns it
    from every inbound OneBot event anyway, and the configured value is the
    pre-connection identity. ``action_timeout_s`` bounds how long
    ``send``/``call`` wait for the API response echo. ``heartbeat_timeout_s``
    is the ping/pong watchdog: when no inbound frame (OneBot meta heartbeat
    or any traffic) arrives within this window the watchdog PINGS the
    connection and awaits the pong (bounded by ``ping_timeout_s``) before
    dropping it — a healthy quiet link that answers the ping stays connected.
    None disables the watchdog. ``reconnect_base_s``/``reconnect_max_s``
    bound the exponential backoff for outbound (``ws``) reconnection.
    ``media_concurrency`` bounds how many background media downloads/decode
    tasks run at once (a semaphore); excess URLs queue and are bounded by the
    adapter's in-flight map, so image downloads never run unbounded.

    Reverse-server security: reverse_ws is loopback-only until this adapter
    supports a configured TLS context. A bearer token on plaintext remote
    transport is not sufficient because it can be intercepted and replayed.
    Browser Origin upgrades are always rejected.
    """

    mode: str = "reverse_ws"  # "reverse_ws" (NapCat dials us) | "ws" (we dial out)
    scheme: str = "ws"  # "ws" | "wss" (forward mode only)
    host: str = "127.0.0.1"
    port: int = 3001
    path: str = "/onebot/v11/ws"
    access_token: str | None = None
    self_id: str | None = None
    action_timeout_s: float = 10.0
    heartbeat_timeout_s: float | None = 30.0
    ping_timeout_s: float = 10.0
    reconnect_base_s: float = 3.0
    reconnect_max_s: float = 60.0
    media_concurrency: int = 4

    def __post_init__(self) -> None:
        if self.mode not in ("reverse_ws", "ws"):
            raise ConfigError(
                f"adapter.onebot.mode must be 'reverse_ws' or 'ws', got {self.mode!r}"
            )
        if self.scheme not in ("ws", "wss"):
            raise ConfigError(
                f"adapter.onebot.scheme must be 'ws' or 'wss', got {self.scheme!r}"
            )
        if self.self_id is not None:
            if not isinstance(self.self_id, str) or not self.self_id.strip():
                raise ConfigError(
                    "adapter.onebot.self_id must be a non-empty string"
                )
        _check_positive_number("adapter.onebot.action_timeout_s", self.action_timeout_s)
        if self.heartbeat_timeout_s is not None:
            _check_positive_number(
                "adapter.onebot.heartbeat_timeout_s", self.heartbeat_timeout_s
            )
        _check_positive_number("adapter.onebot.ping_timeout_s", self.ping_timeout_s)
        _check_positive_number("adapter.onebot.reconnect_base_s", self.reconnect_base_s)
        _check_positive_number("adapter.onebot.reconnect_max_s", self.reconnect_max_s)
        _check_positive_int("adapter.onebot.media_concurrency", self.media_concurrency)
        if (
            self.mode == "ws"
            and self.access_token
            and self.scheme == "ws"
            and not _is_local_host(self.host)
        ):
            raise ConfigError(
                "adapter.onebot: plaintext ws:// with an access token requires "
                "a local-only host (loopback/private/link-local literal) or "
                'scheme = "wss"'
            )
        if self.mode == "reverse_ws" and not _is_loopback_host(self.host):
            raise ConfigError(
                "adapter.onebot: reverse_ws is loopback-only until TLS "
                "support is configured; use a local reverse proxy for remote access"
            )


@dataclass(frozen=True)
class AdapterConfig:
    """Adapter selection + connection knobs.

    ``name`` selects the live adapter: ``"console"`` (the local REPL, the
    default and the only adapter allowed in dry-run) or ``"onebot"`` (the
    OneBot v11 WebSocket bridge, live only). ``onebot`` carries the OneBot
    connection/runtime knobs.
    """

    name: str = "console"  # "console" | "onebot"
    onebot: OneBotConfig = field(default_factory=OneBotConfig)

    def __post_init__(self) -> None:
        if self.name not in ("console", "onebot"):
            raise ConfigError(
                f"adapter.name must be 'console' or 'onebot', got {self.name!r}"
            )


@dataclass(frozen=True)
class StorageConfig:
    db_path: str = "data/pretender.db"


@dataclass(frozen=True)
class LogConfig:
    dir: str = "logs"
    level: str = "INFO"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 3


@dataclass(frozen=True)
class PluginsConfig:
    """The Phase 6 P6.6 explicit-trust plugin surface.

    ``paths`` is the ordered list of plugin module files. Relative paths
    resolve against the config file's directory at load; every resolved
    path must stay STRICTLY INSIDE the config root and be a ``.py`` file,
    and duplicates are rejected (a ConfigError at load time).
    ``entry_points`` is the ordered list of EXPLICIT entry-point names in
    the ``pretender.plugins`` group — there is NO auto-discovery.
    ``allow_replace`` is the operator allowlist of protected core names a
    plugin may replace (a protected name is replaceable ONLY when it is on
    this list). ``hook_timeout_s`` bounds every hook invocation. No
    auto-loading and no hot reload: only these explicit sources are
    resolved, once, at startup.
    """

    paths: tuple[str, ...] = ()
    entry_points: tuple[str, ...] = ()
    allow_replace: tuple[str, ...] = ()
    hook_timeout_s: float = 5.0
    #: The config root the paths were resolved against (set by the loader;
    #: preserved across to_dict/from_dict round-trips so re-resolution is
    #: stable). Internal — not part of the user-facing schema.
    base_dir: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("paths", "entry_points", "allow_replace"):
            for value in getattr(self, name):
                if not isinstance(value, str) or not value.strip():
                    raise ConfigError(
                        f"plugins.{name} entries must be non-empty strings"
                    )
        _check_positive_number("plugins.hook_timeout_s", self.hook_timeout_s)


# ── Per-chat overrides ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChatOverride:
    """One [[chats]] entry. Any section left out falls back to the top-level
    config; ``cfg_json`` is arbitrary plugin-owned JSON.

    ``*_raw`` holds the raw TOML section dicts (explicit keys only),
    populated by the loader, so the per-chat merge preserves EXPLICIT FIELD
    PRESENCE: a chat can reset a non-default global value to its default
    (e.g. threshold 12 -> 8) while omitted fields inherit the parent.
    """

    key: ChatKey
    gate: GateConfig | None = None
    drift: DriftConfig | None = None
    output: OutputConfig | None = None
    context: ContextConfig | None = None
    budget: BudgetConfig | None = None
    agent: AgentConfig | None = None
    learn: LearnConfig | None = None
    media: MediaConfig | None = None
    cfg_json: dict[str, Any] = field(default_factory=dict)
    gate_raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    drift_raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    output_raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    context_raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    budget_raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    agent_raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    learn_raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    media_raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # section field -> raw-presence field, used by the loader (_build).
    _raw_fields = {
        "gate": "gate_raw",
        "drift": "drift_raw",
        "output": "output_raw",
        "context": "context_raw",
        "budget": "budget_raw",
        "agent": "agent_raw",
        "learn": "learn_raw",
        "media": "media_raw",
    }


@dataclass(frozen=True)
class ChatConfig:
    """The effective, fully-merged config for one chat."""

    key: ChatKey
    gate: GateConfig
    drift: DriftConfig
    output: OutputConfig
    context: ContextConfig
    budget: BudgetConfig
    agent: AgentConfig
    learn: LearnConfig
    media: MediaConfig
    cfg_json: dict[str, Any]


# ── Root ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    bot: BotConfig = field(default_factory=BotConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    learn: LearnConfig = field(default_factory=LearnConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    log: LogConfig = field(default_factory=LogConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)
    chats: tuple[ChatOverride, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for c in self.chats:
            if c.key in seen:
                raise ConfigError(f"duplicate chat override key: {c.key!r}")
            seen.add(c.key)
        if self.media.vision_profile is not None:
            if self.media.vision_profile not in self.llm.profiles:
                raise ConfigError(
                    "media.vision_profile names unknown LLM profile"
                    f" {self.media.vision_profile!r} (configured:"
                    f" {sorted(self.llm.profiles)})"
                )

    def chat_override(self, key: str) -> ChatOverride | None:
        for c in self.chats:
            if c.key == key:
                return c
        return None

    def for_chat(self, key: str) -> ChatConfig:
        """The effective config for one chat: top-level defaults overlaid
        with that chat's [[chats]] override, if any.

        The override DEEP-MERGES with EXPLICIT FIELD PRESENCE: only fields
        the override actually sets (keys present in the raw config data)
        replace the parent's, and nested sections merge recursively — so a
        partial override never clobbers non-default parent values, while a
        chat CAN reset a non-default global to its default (e.g. threshold
        12 -> 8) by setting it explicitly.
        """
        o = self.chat_override(key)
        if o is None:
            return ChatConfig(
                key=ChatKey(key),
                gate=self.gate,
                drift=self.drift,
                output=self.output,
                context=self.context,
                budget=self.budget,
                agent=self.agent,
                learn=self.learn,
                media=self.media,
                cfg_json={},
            )
        return ChatConfig(
            key=o.key,
            gate=_merge_section(self.gate, o.gate, o.gate_raw),
            drift=_merge_section(self.drift, o.drift, o.drift_raw),
            output=_merge_section(self.output, o.output, o.output_raw),
            context=_merge_section(self.context, o.context, o.context_raw),
            budget=_merge_section(self.budget, o.budget, o.budget_raw),
            agent=_merge_section(self.agent, o.agent, o.agent_raw),
            learn=_merge_section(self.learn, o.learn, o.learn_raw),
            media=_merge_section(self.media, o.media, o.media_raw),
            cfg_json=o.cfg_json,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def redacted_dict(self) -> dict[str, Any]:
        """``to_dict()`` with every plugin-owned diagnostic value redacted.

        Per-chat ``cfg_json`` values (arbitrary plugin-owned JSON) are
        replaced with a placeholder, so config diagnostics never leak
        plugin-owned data. Everything else (profiles, gate constants, ...)
        is unchanged.
        """
        data = self.to_dict()
        for chat in data.get("chats", []):
            if isinstance(chat, dict) and "cfg_json" in chat:
                chat["cfg_json"] = {"<redacted>": True}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: str | Path | None = None) -> Config:
        cfg = _build(cls, _expand_env(dict(data)))
        return _resolve_plugins(cfg, base_dir)

    @classmethod
    def loads(cls, text: str, *, base_dir: str | Path | None = None) -> Config:
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"invalid TOML: {e}") from e
        return cls.from_dict(data, base_dir=base_dir)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load from a TOML file. ``None`` (or an empty file) yields the
        default config — an empty TOML must boot a working bot. Relative
        ``plugins.paths`` entries resolve against the config file's
        directory (the config root)."""
        if path is None:
            return cls()
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as e:
            raise ConfigError(f"cannot read config file {p}: {e}") from e
        return cls.loads(text, base_dir=p.parent)


# ── Per-chat deep merge (presence-preserving) ───────────────────────────────

def _merge_section(parent: Any, override: Any, raw: dict[str, Any]) -> Any:
    """Merge a partial override over a parent value, preserving EXPLICIT
    FIELD PRESENCE: only keys present in the raw config data are taken
    from the override, so a chat can reset a non-default global value to
    its default (e.g. threshold 12 -> 8) while omitted fields inherit the
    parent. Nested dataclass sections merge recursively against their own
    raw dicts.
    """
    if override is None:
        return parent
    if dataclasses.is_dataclass(parent) and dataclasses.is_dataclass(override):
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(parent):
            if f.name not in raw:
                continue  # not explicitly set: inherit the parent
            ov = getattr(override, f.name)
            pv = getattr(parent, f.name)
            raw_v = raw[f.name]
            if (
                dataclasses.is_dataclass(ov)
                and dataclasses.is_dataclass(pv)
                and isinstance(raw_v, dict)
            ):
                kwargs[f.name] = _merge_section(pv, ov, raw_v)
            else:
                kwargs[f.name] = ov
        return dataclasses.replace(typing.cast(Any, parent), **kwargs)
    return override


# ── Plugin path resolution (Phase 6 P6.6 explicit-trust semantics) ──────────

def _resolve_plugins(cfg: Config, base_dir: str | Path | None) -> Config:
    """Resolve and validate ``plugins.paths`` against the config root.

    Relative paths resolve against the config file's directory
    (``base_dir``; the current working directory when the config was built
    from text/dict). After resolution every path must be a ``.py`` file
    STRICTLY INSIDE the config root (an out-of-root path is a ConfigError),
    and duplicates are rejected. The resolved absolute paths replace the
    configured ones, and the resolution base is recorded on the config so a
    ``to_dict``/``from_dict`` round-trip (e.g. ``RuntimeOverlay.apply``)
    re-resolves against the SAME root — never the caller's CWD. Existence
    is NOT checked here — the plugin loader (App startup / doctor preflight)
    reports a missing file truthfully.
    """
    if not cfg.plugins.paths:
        return cfg
    if base_dir is not None:
        base = Path(base_dir).resolve()
    elif cfg.plugins.base_dir:
        base = Path(cfg.plugins.base_dir).resolve()
    else:
        base = Path.cwd().resolve()
    resolved: list[str] = []
    seen: set[Path] = set()
    for raw in cfg.plugins.paths:
        p = Path(raw)
        if not p.is_absolute():
            p = base / p
        p = p.resolve()
        try:
            p.relative_to(base)
        except ValueError:
            raise ConfigError(
                f"plugins.paths entry {raw!r} resolves outside the config"
                f" root {base}"
            ) from None
        if p.suffix != ".py":
            raise ConfigError(
                f"plugins.paths entry {raw!r} must be a .py module file"
            )
        if p in seen:
            raise ConfigError(f"duplicate plugins.paths entry {raw!r}")
        seen.add(p)
        resolved.append(str(p))
    return dataclasses.replace(
        cfg,
        plugins=dataclasses.replace(
            cfg.plugins, paths=tuple(resolved), base_dir=str(base)
        ),
    )


# ── ${ENV} expansion ────────────────────────────────────────────────────────

#: Config fields that must resolve from ${ENV} — a literal secret value is a
#: ConfigError at load time (PLAN.md §6: secrets are references, never values).
#: Only LLM profile api_keys and the OneBot access token are secrets; a
#: plugin-owned ``cfg_json`` field named ``api_key`` is arbitrary data and is
#: never treated as one.
_SECRET_FIELDS = frozenset({"api_key"})


def _is_secret_path(path: str) -> bool:
    parts = path.split(".")
    if (
        len(parts) >= 3
        and parts[0] == "llm"
        and parts[1] == "profiles"
        and parts[-1] in _SECRET_FIELDS
    ):
        return True
    return (
        len(parts) == 3
        and parts[0] == "adapter"
        and parts[1] == "onebot"
        and parts[-1] == "access_token"
    )


def _expand_env(value: Any, path: str = "") -> Any:
    if isinstance(value, str):
        m = _ENV_REF.fullmatch(value)
        if m:
            name = m.group(1)
            if name not in os.environ:
                raise ConfigError(
                    f"environment variable {name!r} referenced in config is not set"
                )
            resolved = os.environ[name]
            if _is_secret_path(path) and not resolved.strip():
                raise ConfigError(
                    f"environment variable {name!r} referenced for secret "
                    f"{path!r} must not be empty or whitespace-only"
                )
            return resolved
        if _is_secret_path(path):
            raise ConfigError(
                f"secret {path!r} must be a ${{ENV}} reference, "
                "not a literal value"
            )
        return value
    if isinstance(value, dict):
        return {
            k: _expand_env(v, f"{path}.{k}" if path else k) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_expand_env(v, path) for v in value]
    return value


# ── Schema-driven building with unknown-key rejection ───────────────────────

def _build(cls: type[Any], data: dict[str, Any]) -> Any:
    if not dataclasses.is_dataclass(cls):
        return data
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"unknown config key(s) for {cls.__name__}: {unknown}")
    kwargs: dict[str, Any] = {}
    for name, f in fields.items():
        if name not in data:
            continue
        kwargs[name] = _coerce(hints[name], data[name], f"{cls.__name__}.{name}")
    result = cls(**kwargs)
    # Preserve explicit field presence: stash the raw section dicts (only
    # the keys the user actually wrote) so per-chat merges can distinguish
    # "explicitly set to the default" from "omitted".
    raw_fields = getattr(cls, "_raw_fields", None)
    if raw_fields:
        for section, attr in raw_fields.items():
            if attr in data and isinstance(data[attr], dict):
                raw = data[attr]  # preserved across a to_dict/from_dict round-trip
            else:
                raw = data.get(section)
            object.__setattr__(result, attr, raw if isinstance(raw, dict) else {})
    return result


def _coerce(expected: Any, value: Any, path: str) -> Any:
    origin = get_origin(expected)

    # NewType (ChatKey, ...)
    supertype = getattr(expected, "__supertype__", None)
    if supertype is not None:
        return expected(_coerce(supertype, value, path))

    # Optional[X] / X | None
    if origin in (typing.Union, types.UnionType):
        args = get_args(expected)
        if type(None) in args and value is None:
            return None
        args = [a for a in args if a is not type(None)]
        if len(args) == 1:
            return _coerce(args[0], value, path)
        return value

    if origin is dict:
        (_, vtype) = get_args(expected)
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a table, got {type(value).__name__}")
        return {k: _coerce(vtype, v, f"{path}.{k}") for k, v in value.items()}

    if origin in (list, tuple):
        (etype, *_) = get_args(expected)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected an array, got {type(value).__name__}")
        coerced = [_coerce(etype, v, path) for v in value]
        return tuple(coerced) if origin is tuple else coerced

    if dataclasses.is_dataclass(expected):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a table, got {type(value).__name__}")
        return _build(typing.cast(type[Any], expected), value)

    # Plain scalars: validate the TOML type matches the schema.
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if expected is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    if expected is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a boolean, got {value!r}")
        return value
    return value


# ── RuntimeOverlay ──────────────────────────────────────────────────────────

class RuntimeOverlay:
    """Mutable dotted-path overrides applied on top of a frozen Config.

    ``overlay.set("gate.threshold", 12)`` then ``overlay.apply(cfg)`` returns
    a NEW Config with the override in effect — no file is touched, and the
    original stays frozen. Paths are validated against the config schema at
    apply time, so a typo'd path is a ConfigError, not silent inaction.
    """

    def __init__(self) -> None:
        self._overrides: dict[str, Any] = {}

    def set(self, path: str, value: Any) -> None:
        if not path or any(not part for part in path.split(".")):
            raise ConfigError(f"invalid overlay path: {path!r}")
        self._overrides[path] = value

    def get(self, path: str, default: Any = None) -> Any:
        return self._overrides.get(path, default)

    def clear(self) -> None:
        self._overrides.clear()

    def items(self) -> tuple[tuple[str, Any], ...]:
        return tuple(sorted(self._overrides.items()))

    def apply(self, cfg: Config) -> Config:
        data = cfg.to_dict()
        for path, value in self._overrides.items():
            parts = path.split(".")
            node: Any = data
            for part in parts[:-1]:
                if not isinstance(node, dict) or part not in node:
                    raise ConfigError(
                        f"overlay path {path!r} does not exist in config"
                    )
                node = node[part]
            last = parts[-1]
            if not isinstance(node, dict) or last not in node:
                raise ConfigError(f"overlay path {path!r} does not exist in config")
            node[last] = value
        return Config.from_dict(data)


# Convenience alias matching the module's public surface.
load_config = Config.load
