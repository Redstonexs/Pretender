from __future__ import annotations

import random

from pretender.output.pipeline import OutputPipeline
from pretender.output.typo import TypoStage, load_frequency, typo_text
from pretender.types import ChatKey, Outgoing


def _out(text: str) -> Outgoing:
    return Outgoing(chat_key=ChatKey("qq:group:1"), text=text, idem_key="dispatch:1:0")


def test_typo_text_is_deterministic_same_pinyin():
    value = typo_text("我在这里", rate=1.0, rng=random.Random(1), max_mutations=1)
    assert value != "我在这里"
    assert len(value) == len("我在这里")


def test_typo_preserves_url_code_and_mentions():
    text = "看看 https://example.test/a 在 `code` @bot"
    value = typo_text(text, rate=1.0, rng=random.Random(2), max_mutations=4)
    assert "https://example.test/a" in value
    assert "`code`" in value
    assert "@bot" in value


def test_typo_stage_honors_switches_and_parts():
    stage = TypoStage(typo_rate=1.0, rng=random.Random(3), max_mutations=1)
    out = _out("我在这里")
    out.parts = ["我在这里", "你在那边"]
    stage.apply(out)
    # Two input parts, plus at most one correction bubble (MaiBot sends the
    # correct word as its own message after a typo).
    assert out.parts and 2 <= len(out.parts) <= 3
    disabled = _out("我在这里")
    disabled.enable_chinese_typo = False
    stage.apply(disabled)
    assert disabled.text == "我在这里"


def test_frequency_asset_and_pipeline_registration():
    assert load_frequency()["在"] > 0
    pipeline = OutputPipeline()
    assert pipeline.get("typo") is not None
