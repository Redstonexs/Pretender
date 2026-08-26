"""Tests for the ordered output pipeline runner and OutputStage protocol."""

from __future__ import annotations

import pytest

from pretender.config import OutputConfig
from pretender.errors import ConfigError, RegistryError
from pretender.output import OutputPipeline, SanitizeStage, SplitStage
from pretender.seams import OutputStage
from pretender.types import ChatKey, Outgoing


def _out(text: str, **kw) -> Outgoing:
    return Outgoing(chat_key=ChatKey("qq:group:1"), text=text, **kw)


def test_builtins_registered_in_order():
    pipe = OutputPipeline()
    assert pipe.names() == ("sanitize", "split", "typo")
    assert [s.name for s in pipe.stages()] == ["split", "typo", "sanitize"]


def test_stages_satisfy_outputstage_protocol():
    assert isinstance(SanitizeStage(), OutputStage)
    assert isinstance(SplitStage(), OutputStage)
    assert isinstance(OutputPipeline().get("sanitize"), OutputStage)


def test_config_pipeline_reorders_stages():
    # Optional stages retain configured order; core sanitize is final.
    cfg = OutputConfig(pipeline=("sanitize", "typo", "split"))
    pipe = OutputPipeline(cfg)
    assert [s.name for s in pipe.stages()] == ["typo", "split", "sanitize"]


def test_unknown_stage_in_pipeline_raises():
    # Fail closed: an unknown stage name is a ConfigError, never silently
    # skipped.
    cfg = OutputConfig(pipeline=("sanitize", "split", "nope"))
    pipe = OutputPipeline(cfg)
    with pytest.raises(ConfigError):
        pipe.stages()


def test_split_before_configured_sanitize_is_repaired_to_core_final():
    cfg = OutputConfig(pipeline=("split", "sanitize"))
    pipe = OutputPipeline(cfg)
    assert [s.name for s in pipe.stages()] == ["split", "sanitize"]


def test_typo_before_configured_sanitize_is_repaired_to_core_final():
    cfg = OutputConfig(pipeline=("typo", "sanitize"))
    pipe = OutputPipeline(cfg)
    assert [s.name for s in pipe.stages()] == ["typo", "sanitize"]


def test_split_without_configured_sanitize_gets_core_final():
    cfg = OutputConfig(pipeline=("split",))
    pipe = OutputPipeline(cfg)
    assert [s.name for s in pipe.stages()] == ["split", "sanitize"]


def test_sanitize_only_pipeline_is_safe():
    cfg = OutputConfig(pipeline=("sanitize",))
    pipe = OutputPipeline(cfg)
    assert [s.name for s in pipe.stages()] == ["sanitize"]


def test_run_repairs_unsafe_pipeline_with_core_final():
    cfg = OutputConfig(pipeline=("split", "sanitize"))
    pipe = OutputPipeline(cfg)
    out = _out("第一句。第二句！")
    pipe.run(out)
    assert out.parts == ["第一句。", "第二句！"]


def test_empty_pipeline_falls_back_to_order():
    cfg = OutputConfig(pipeline=())
    pipe = OutputPipeline(cfg)
    assert [s.name for s in pipe.stages()] == ["split", "typo", "sanitize"]


def test_register_and_replace():
    pipe = OutputPipeline()

    class Custom:
        name = "custom"
        order = 5

        def apply(self, out):
            out.text = out.text + "!"
            return out

    pipe.register(Custom())
    assert "custom" in pipe.names()
    custom = pipe.get("custom")
    assert custom is not None and custom.order == 5

    # replace keeps the original slot
    class Custom2:
        name = "custom"
        order = 5

        def apply(self, out):
            return out

    pipe.register(Custom2(), replace=True)
    assert pipe.get("custom") is not None

    pipe.unregister("custom")
    assert "custom" not in pipe.names()


def test_register_duplicate_raises():
    pipe = OutputPipeline()
    with pytest.raises(RegistryError):
        pipe.register(SanitizeStage())


def test_register_shape_violation_raises():
    pipe = OutputPipeline()

    class Bad:
        name = "bad"  # missing order and apply

    with pytest.raises(RegistryError):
        pipe.register(Bad())


def test_run_applies_sanitize_then_split():
    pipe = OutputPipeline()
    out = _out("第一句。第二句！第三句？")
    pipe.run(out)
    assert out.parts == ["第一句。", "第二句！", "第三句？"]


def test_skip_post_process_bypasses_everything():
    pipe = OutputPipeline()
    out = _out("第一句。第二句！第三句？", skip_post_process=True)
    pipe.run(out)
    # untouched: no sanitize, no split
    assert out.text == "第一句。第二句！第三句？"
    assert out.parts is None


def test_enable_splitter_false_skips_split_but_runs_sanitize():
    pipe = OutputPipeline()
    out = _out("第一句。第二句！第三句？", enable_splitter=False)
    pipe.run(out)
    assert out.parts is None  # split skipped
    assert out.text == "第一句。第二句！第三句？"  # sanitize left it intact


def test_enable_splitter_true_runs_split():
    pipe = OutputPipeline()
    out = _out("第一句。第二句！第三句？", enable_splitter=True)
    pipe.run(out)
    assert out.parts == ["第一句。", "第二句！", "第三句？"]


def test_run_returns_same_mutable_outgoing():
    pipe = OutputPipeline()
    out = _out("第一句。第二句！第三句？")
    assert pipe.run(out) is out


def test_sanitize_records_protected_spans_for_split():
    # A URL is a protected span: sanitize records it, split must not cut it.
    pipe = OutputPipeline()
    text = "看这个 https://example.com/a?b=1 然后。下一句。"
    out = _out(text)
    pipe.run(out)
    assert "protected_spans" in out.platform_ref
    # the URL survives intact inside a single part
    joined = "".join(out.parts or [out.text])
    assert "https://example.com/a?b=1" in joined
