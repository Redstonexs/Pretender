"""Ordered output pipeline: a registry of ``OutputStage``s run in config
order over a MUTABLE ``Outgoing``.

The pipeline owns the shared "protected span" helpers both built-in stages
use: sanitize marks no-mutate spans (URLs, code blocks, inline code,
blockquotes) and split refuses to cut them. Protected spans ride on
``Outgoing.platform_ref["protected_spans"]`` as ``[start, end]`` character
offsets into the current ``out.text`` — the one mutable, plugin-owned field
that survives untouched through the outbox conversion.

Ordering comes from ``OutputConfig.pipeline`` (a config override per
PLAN.md §3.2). The core sanitizer is an invariant final stage: it is not a
replaceable registry entry and is run after every configured/plugin stage.
Per-reply switches are honoured: ``skip_post_process`` bypasses optional stages,
``enable_splitter`` (None → config default) gates the split stage.
"""

from __future__ import annotations

import hashlib
import re

from pretender.config import OutputConfig
from pretender.errors import ConfigError
from pretender.registry import Registry
from pretender.seams import OutputStage
from pretender.types import Outgoing

# ── protected spans ─────────────────────────────────────────────────────────
# "No-mutate" spans shared by the stages. Sanitize records them so split
# (and, later, typo) never touch them.

_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_QUOTE_RE = re.compile(r"(?:^|\n)\s*>+[^\n]*")

# Trailing punctuation a URL match should not swallow (ASCII + CJK).
_URL_TRAIL = ".,;:!?)]}，。！？；：、…"

PROTECT_PATTERNS = (_URL_RE, _FENCE_RE, _INLINE_CODE_RE, _QUOTE_RE)


def detect_protected_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of ``text`` that must not be mutated, merged and
    sorted. URLs have trailing punctuation trimmed so a sentence-final
    period stays outside the protected span."""
    spans: list[tuple[int, int]] = []
    for pat in PROTECT_PATTERNS:
        for m in pat.finditer(text):
            s, e = m.span()
            if pat is _URL_RE:
                while e > s and text[e - 1] in _URL_TRAIL:
                    e -= 1
            if e > s:
                spans.append((s, e))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _in_span(pos: int, spans: list[tuple[int, int]]) -> bool:
    for s, e in spans:
        if s <= pos < e:
            return True
    return False


def _overlaps(s: int, e: int, spans: list[tuple[int, int]]) -> bool:
    for ps, pe in spans:
        if s < pe and ps < e:
            return True
    return False


def get_protected_spans(out: Outgoing) -> list[tuple[int, int]]:
    """The spans recorded on the Outgoing, or freshly detected when none are
    recorded (e.g. split running without a preceding sanitize stage)."""
    raw = out.platform_ref.get("protected_spans")
    if raw:
        return [tuple(x) for x in raw]
    return detect_protected_spans(out.text)


def set_protected_spans(out: Outgoing, spans: list[tuple[int, int]]) -> None:
    out.platform_ref["protected_spans"] = [[s, e] for s, e in spans]


def stable_group_id(parts: list[str]) -> str:
    """Content-derived group id matching the outbox's scheme, so a retried
    finish produces the same grouping."""
    digest = hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()
    return f"g:{digest[:16]}"


class OutputPipeline:
    """An ordered, shape-validated registry of output stages plus the runner
    that applies them over a mutable Outgoing.

    ``extra_stages`` (Phase 6 P6.6) are the frozen plugin output stages from
    the staging registry: they are registered AFTER the built-ins with
    ``replace=True``. The mandatory core ``sanitize`` stage is never shadowed
    and always runs last.
    """

    def __init__(
        self,
        config: OutputConfig | None = None,
        *,
        extra_stages: tuple[OutputStage, ...] = (),
    ) -> None:
        self.config = config or OutputConfig()
        self._registry: Registry[OutputStage] = Registry("output", OutputStage)
        self._register_builtins()
        for stage in extra_stages:
            # The frozen staging registry contains the built-in entries too.
            # Do not attempt to replace the mandatory sanitizer when seeding a
            # per-chat pipeline.
            if stage.name != "sanitize":
                self.register(stage, replace=True)

    def _register_builtins(self) -> None:
        # Lazy imports avoid a circular import: sanitize/split import the
        # shared helpers from this module.
        from pretender.output.sanitize import SanitizeStage
        from pretender.output.split import SplitStage
        from pretender.output.typo import TypoStage

        self.register(SanitizeStage())
        self.register(SplitStage(max_split=self.config.max_split))
        self.register(TypoStage(typo_rate=self.config.typo_rate))

    # ── registration (delegates to the typed Registry) ──────────────────────

    def register(self, stage=None, *, replace: bool = False, name: str | None = None):
        if replace and (name or getattr(stage, "name", None)) == "sanitize":
            raise ConfigError("the core sanitize stage is never replaceable")
        return self._registry.register(stage, replace=replace, name=name)

    def unregister(self, name: str) -> None:
        self._registry.unregister(name)

    def get(self, name: str) -> OutputStage | None:
        return self._registry.get(name)

    def names(self) -> tuple[str, ...]:
        return self._registry.names()

    def stages(self) -> tuple[OutputStage, ...]:
        """Registered stages in ``config.pipeline`` order.

        Fail-closed: an unknown stage name raises ``ConfigError`` (never
        silently skipped). The configured sanitizer entry is ignored for
        ordering and the mandatory core instance is appended last. An empty
        pipeline uses all registered stages sorted by ``order``, followed by
        the core sanitizer.
        """
        core = self._registry.get("sanitize")
        if core is None:
            raise ConfigError("output pipeline has no core sanitize stage")
        if not self.config.pipeline:
            stages = tuple(
                sorted(
                    (s for s in self._registry.all() if s.name != "sanitize"),
                    key=lambda s: s.order,
                )
            )
            return stages + (core,)
        result: list[OutputStage] = []
        for name in self.config.pipeline:
            stage = self._registry.get(name)
            if stage is None:
                raise ConfigError(
                    f"unknown output stage in pipeline: {name!r} "
                    f"(registered: {', '.join(self._registry.names())})"
                )
            if stage.name != "sanitize":
                result.append(stage)
        return tuple(result) + (core,)

    def validate(self) -> None:
        """Validate configured stage names/order after optional plugin stages
        have been registered. Doctor calls this for every effective chat
        config; runtime calls it through ``stages()`` before output is sent."""
        self.stages()

    # ── runner ──────────────────────────────────────────────────────────────

    def run(self, out: Outgoing) -> Outgoing:
        """Apply the ordered stages to ``out`` in place, honouring the
        per-reply switches."""
        stages = self.stages()
        if out.skip_post_process:
            # skip_post_process can bypass optional transforms, never the
            # final safety boundary.
            return stages[-1].apply(out)
        for stage in stages:
            if stage.name == "split" and not self._split_enabled(out):
                continue
            if stage.name == "typo" and not self._typo_enabled(out):
                continue
            out = stage.apply(out)
        return out

    def _split_enabled(self, out: Outgoing) -> bool:
        flag = out.enable_splitter
        if flag is None:
            return (
                "split" in self.config.pipeline
                if self.config.pipeline
                else True
            )
        return bool(flag)

    def _typo_enabled(self, out: Outgoing) -> bool:
        flag = out.enable_chinese_typo
        if flag is None:
            return self.config.typo_rate > 0
        return bool(flag)
