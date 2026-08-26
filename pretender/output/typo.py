"""Deterministic, protected-span-safe Chinese typo stage.

The stage intentionally makes at most a few same-pinyin substitutions. Its
default RNG is seeded from the durable output identity, so a replay/retry
produces identical text rather than creating a new outbox payload.
"""

from __future__ import annotations

import hashlib
import random
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Iterable

import jieba
from pypinyin import Style, pinyin

from pretender.output.pipeline import _in_span, detect_protected_spans
from pretender.types import Outgoing

# The frequency asset ships inside the installed package (see
# pyproject.toml package-data), so typo behaviour survives outside the source
# tree. ``importlib.resources`` resolves it whether the package is a plain
# directory or a zipped wheel.
_PACKAGE = "pretender.output"
_ASSET_NAME = "data/char_freq.txt"


def _asset_source(path: str | None):
    """Return a ``read_text``-able source for the frequency asset: the
    packaged file when ``path`` is None, else the given filesystem path."""
    if path is None:
        return resources.files(_PACKAGE).joinpath(_ASSET_NAME)
    return Path(path)


@lru_cache(maxsize=8)
def load_frequency(path: str | None = None) -> dict[str, int]:
    """Load a small ``char frequency`` asset, ignoring malformed rows."""
    result: dict[str, int] = {}
    try:
        lines = _asset_source(path).read_text(encoding="utf-8").splitlines()
    except (OSError, FileNotFoundError):
        return result
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2 or len(fields[0]) != 1:
            continue
        try:
            frequency = int(fields[1])
        except ValueError:
            continue
        if frequency > 0:
            result[fields[0]] = frequency
    return result


@lru_cache(maxsize=8)
def _pinyin_candidates(path: str | None = None) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[tuple[str, int]]] = {}
    for char, frequency in load_frequency(path).items():
        value = pinyin(char, style=Style.NORMAL, heteronym=False)
        if not value or not value[0]:
            continue
        groups.setdefault(value[0][0], []).append((char, frequency))
    return {
        key: tuple(char for char, _ in sorted(values, key=lambda item: (-item[1], item[0])))
        for key, values in groups.items()
    }


def _is_chinese(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _seed(out: Outgoing) -> int:
    token = out.idem_key or out.group_id or out.text
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)


def typo_text(
    text: str,
    *,
    rate: float,
    rng: random.Random,
    asset_path: str | None = None,
    protected_spans: Iterable[tuple[int, int]] = (),
    max_mutations: int = 2,
) -> str:
    """Return a conservatively typo-mutated string.

    URLs/code/quotes/mentions are protected by character offsets. Jieba is
    used to walk natural-language token boundaries; only Han characters with a
    same-pinyin candidate are eligible. Adjacent substitutions are forbidden.
    """
    if rate <= 0 or not text or max_mutations <= 0:
        return text
    spans = list(protected_spans) or detect_protected_spans(text)
    candidates = _pinyin_candidates(asset_path)
    result = list(text)
    offset = 0
    mutations = 0
    previous_mutation = -2
    # jieba tokenization gives a stable traversal without touching ASCII tokens.
    for token in jieba.lcut(text, HMM=False):
        token_start = text.find(token, offset)
        if token_start < 0:
            continue
        offset = token_start + len(token)
        for local, char in enumerate(token):
            pos = token_start + local
            if mutations >= max_mutations or pos == previous_mutation + 1:
                continue
            if not _is_chinese(char) or _in_span(pos, spans) or rng.random() >= rate:
                continue
            values = pinyin(char, style=Style.NORMAL, heteronym=False)
            if not values or not values[0]:
                continue
            choices = [item for item in candidates.get(values[0][0], ()) if item != char]
            if not choices:
                continue
            result[pos] = choices[rng.randrange(len(choices))]
            mutations += 1
            previous_mutation = pos
    return "".join(result)


class TypoStage:
    """OutputStage applying deterministic same-pinyin substitutions."""

    name = "typo"
    order = 30

    def __init__(
        self,
        *,
        typo_rate: float = 0.0,
        asset_path: str | None = None,
        rng: random.Random | None = None,
        max_mutations: int = 2,
    ) -> None:
        self.typo_rate = max(0.0, min(1.0, typo_rate))
        self.asset_path = asset_path
        self._rng = rng
        self.max_mutations = max_mutations

    def apply(self, out: Outgoing) -> Outgoing:
        if out.skip_post_process or out.enable_chinese_typo is False:
            return out
        if out.enable_chinese_typo is None and self.typo_rate <= 0:
            return out
        rng = self._rng or random.Random(_seed(out))
        spans = detect_protected_spans(out.text)
        if out.parts:
            out.parts = [
                typo_text(
                    part,
                    rate=self.typo_rate,
                    rng=rng,
                    asset_path=self.asset_path,
                    protected_spans=detect_protected_spans(part),
                    max_mutations=self.max_mutations,
                )
                for part in out.parts
            ]
        out.text = typo_text(
            out.text,
            rate=self.typo_rate,
            rng=rng,
            asset_path=self.asset_path,
            protected_spans=spans,
            max_mutations=self.max_mutations,
        )
        return out
