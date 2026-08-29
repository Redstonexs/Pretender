"""Attention drift: the 4x3x3 matrix and its four guard clauses.

Drift used to be dead config — ``[drift]`` parsed into ``DriftConfig`` and
nothing read it. These tests pin that it reaches both prompts.
"""

from __future__ import annotations

import pytest

from pretender.config import DriftConfig
from pretender.drift import (
    ANCHOR_POLICY_RULES,
    DRIFT_LEVEL_RULES,
    REACTION_STYLE_RULES,
    build_drift_block,
)


def test_matrix_is_four_by_three_by_three():
    assert set(DRIFT_LEVEL_RULES) == {"subtle", "active", "scattered", "wild"}
    assert set(ANCHOR_POLICY_RULES) == {"strict", "balanced", "loose"}
    assert set(REACTION_STYLE_RULES) == {"reserved", "natural", "lively"}


@pytest.mark.parametrize("level", sorted(DRIFT_LEVEL_RULES))
@pytest.mark.parametrize("anchor", sorted(ANCHOR_POLICY_RULES))
@pytest.mark.parametrize("reaction", sorted(REACTION_STYLE_RULES))
def test_every_cell_renders_its_three_rules(level, anchor, reaction):
    block = build_drift_block(
        DriftConfig(level=level, anchor=anchor, reaction=reaction)
    )
    assert DRIFT_LEVEL_RULES[level] in block
    assert ANCHOR_POLICY_RULES[anchor] in block
    assert REACTION_STYLE_RULES[reaction] in block


def test_all_four_guards_are_always_present():
    """The fourth guard is load-bearing: without "do not simulate real
    inefficiency" a drift prompt degrades into distraction theatre."""
    block = build_drift_block(DriftConfig())
    assert "必须能从最近消息里找到明确触发点" in block  # traceable hook
    assert "漂移不是单纯换话题" in block  # not topic-switching
    assert "不要真的降效" in block  # not real inefficiency
    assert "不要自称 ADHD" in block  # never self-labelled


def test_unknown_values_degrade_to_empty_rather_than_raising():
    """A typo in one config key must not silence the bot."""
    assert build_drift_block(DriftConfig(level="nope")) == ""
    assert build_drift_block(DriftConfig(anchor="nope")) == ""
    assert build_drift_block(DriftConfig(reaction="nope")) == ""
    assert build_drift_block(None) == ""
