"""Phase 6 P6.6 explicit-trust plugins: deterministic loading, staging
registries, protected/allowlisted replacement, failure rollback, freeze
(no runtime mutation / hot reload), and the exact-replay gate-feature
fingerprint."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pretender.config import Config
from pretender.errors import ConfigError, RegistryError
from pretender.gate import Gate
from pretender.registry import (
    HookBus,
    PluginAPI,
    PluginLoadResult,
    PluginLoader,
    load_plugin_module,
)
from pretender.types import (
    AdapterEvent,
    ChatIdentity,
    ChatKey,
    CommitSeq,
    CorpusMarker,
    DispatchCause,
    DispatchId,
    EventId,
    Message,
    MessageId,
    MessageRowId,
    PlatformId,
    SelfId,
    SenderId,
    WakeKind,
)
from pretender.record import CorpusView
from pretender.cycle import replay_marker_schedule
from pretender.cycle import _composition_fingerprint

CK = ChatKey("qq:group:123456")
SELF = SenderId("bot-1")


def _write_plugin(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / f"{name}.py"
    # Production manifests require an explicit version.  Keep the historical
    # plugin snippets concise while allowing tests that exercise a deliberate
    # version mismatch to provide their own declaration.
    if "version" not in body and "__version__" not in body:
        body = 'version = "1"\n' + body
    p.write_text(body, encoding="utf-8")
    return p


def _cfg_with_paths(tmp_path: Path, *paths: Path) -> Config:
    """A config FILE in ``tmp_path`` whose relative plugin paths resolve
    inside the config root (the realistic explicit-trust layout)."""
    rel = ", ".join(f'"{p.name}"' for p in paths)
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text(f"[plugins]\npaths = [{rel}]\n", encoding="utf-8")
    return Config.load(cfg_file)


# ── deterministic order / duplicates / rollback ─────────────────────────────

def test_plugin_loader_resolves_paths_in_deterministic_order(tmp_path):
    first = _write_plugin(
        tmp_path,
        "first",
        'name = "first"\n\ndef setup(api):\n    api.hooks.on_event(lambda e: None)\n',
    )
    second = _write_plugin(
        tmp_path,
        "second",
        'name = "second"\n\ndef setup(api):\n    api.hooks.on_event(lambda e: None)\n',
    )
    cfg = _cfg_with_paths(tmp_path, first, second)
    result = PluginLoader(cfg).load()
    assert result.plugin_names == ("first", "second")
    # The staging registries are frozen after load.
    assert result.gate_features.frozen
    assert result.output_stages.frozen
    assert result.tools.frozen
    assert result.hooks.frozen


def test_plugin_loader_rejects_duplicate_plugin_names(tmp_path):
    a = _write_plugin(tmp_path, "a", 'name = "dup"\n\ndef setup(api):\n    pass\n')
    b = _write_plugin(tmp_path, "b", 'name = "dup"\n\ndef setup(api):\n    pass\n')
    cfg = _cfg_with_paths(tmp_path, a, b)
    with pytest.raises(RegistryError, match="duplicate plugin name"):
        PluginLoader(cfg).load()


def test_plugin_loader_rejects_missing_name(tmp_path):
    p = _write_plugin(tmp_path, "noname", "def setup(api):\n    pass\n")
    cfg = _cfg_with_paths(tmp_path, p)
    with pytest.raises(RegistryError, match="'name' string"):
        PluginLoader(cfg).load()


def test_plugin_loader_rejects_import_error(tmp_path):
    p = _write_plugin(tmp_path, "broken", "raise RuntimeError('boom')\n")
    cfg = _cfg_with_paths(tmp_path, p)
    with pytest.raises(RegistryError, match="cannot import plugin module"):
        PluginLoader(cfg).load()


def test_plugin_loader_rejects_missing_entry_point():
    cfg = Config.loads('[plugins]\nentry_points = ["pretender.nope"]\n')
    with pytest.raises(RegistryError, match="no entry point named"):
        PluginLoader(cfg).load()


def test_plugin_setup_failure_aborts_with_no_partial_registry(tmp_path):
    """A plugin whose setup raises aborts the whole load: the staging
    registries are never frozen and no usable partial registry exists."""
    p = _write_plugin(
        tmp_path,
        "badsetup",
        'name = "bad"\n\ndef setup(api):\n    raise RuntimeError("setup boom")\n',
    )
    cfg = _cfg_with_paths(tmp_path, p)
    with pytest.raises(RegistryError, match="setup failed"):
        PluginLoader(cfg).load()


def test_plugin_setup_called_once(tmp_path):
    marker = tmp_path / "calls.txt"
    p = _write_plugin(
        tmp_path,
        "once",
        'name = "once"\n\n'
        "def setup(api):\n"
        f"    with open({str(marker)!r}, 'a') as f:\n"
        "        f.write('once\\n')\n",
    )
    cfg = _cfg_with_paths(tmp_path, p)
    PluginLoader(cfg).load()
    assert marker.read_text(encoding="utf-8") == "once\n"


# ── protected names / replacement allowlist ─────────────────────────────────

def test_protected_sanitize_replacement_requires_allowlist(tmp_path):
    body = (
        'name = "repl"\n\n'
        "class MySanitize:\n"
        '    name = "sanitize"\n'
        "    order = 10\n"
        "    def apply(self, out):\n"
        "        return out\n\n"
        "def setup(api):\n"
        "    api.output_stages.register(MySanitize(), replace=True)\n"
    )
    p = _write_plugin(tmp_path, "repl", body)
    cfg = _cfg_with_paths(tmp_path, p)
    with pytest.raises(RegistryError, match="sanitize"):
        PluginLoader(cfg).load()
    # The operator allowlist cannot replace the mandatory core sanitizer.
    cfg_file = tmp_path / "cfg2.toml"
    cfg_file.write_text(
        f'[plugins]\npaths = ["{p.name}"]\nallow_replace = ["sanitize"]\n',
        encoding="utf-8",
    )
    cfg2 = Config.load(cfg_file)
    with pytest.raises(RegistryError, match="sanitize"):
        PluginLoader(cfg2).load()


def test_protected_terminal_intent_replacement_requires_allowlist(tmp_path):
    body = (
        'name = "replreply"\n\n'
        "from pretender.tools import tool\n\n"
        '@tool("reply")\n'
        "def reply(text: str) -> str:\n"
        '    """A replacement reply tool."""\n'
        "    return text\n\n"
        "def setup(api):\n"
        "    api.tools.register(reply, replace=True)\n"
    )
    p = _write_plugin(tmp_path, "replreply", body)
    cfg = _cfg_with_paths(tmp_path, p)
    with pytest.raises(RegistryError, match="protected core name"):
        PluginLoader(cfg).load()
    cfg_file = tmp_path / "cfg2.toml"
    cfg_file.write_text(
        f'[plugins]\npaths = ["{p.name}"]\nallow_replace = ["reply"]\n',
        encoding="utf-8",
    )
    cfg2 = Config.load(cfg_file)
    result = PluginLoader(cfg2).load()
    assert result.tools.require("reply").name == "reply"


def test_plugin_registers_gate_feature_and_tool(tmp_path):
    body = (
        'name = "ext"\n\n'
        "from pretender.tools import tool\n\n"
        "class MyFeature:\n"
        '    name = "my_feature"\n'
        "    def contribute(self, ctx):\n"
        "        return None\n\n"
        '@tool("my_tool")\n'
        "def my_tool(text: str) -> str:\n"
        '    """A plugin tool."""\n'
        "    return text\n\n"
        "def setup(api):\n"
        "    api.gate_features.register(MyFeature())\n"
        "    api.tools.register(my_tool)\n"
    )
    p = _write_plugin(tmp_path, "ext", body)
    cfg = _cfg_with_paths(tmp_path, p)
    result = PluginLoader(cfg).load()
    assert result.gate_features.names() == (
        "relevance",
        "content",
        "pressure",
        "presence",
        "frequency",
        "my_feature",
    )
    assert result.tools.names()[-1] == "my_tool"
    # The live Gate built from the frozen staging registry includes the
    # plugin feature in its fingerprint.
    from pretender.cycle import _gate_fingerprint

    gate = Gate(features=result.gate_features)
    assert _gate_fingerprint(gate)[-1] == "my_feature"


def test_plugin_api_does_not_expose_app_repo_adapter(tmp_path):
    """The disposable PluginAPI exposes ONLY the staging registries and the
    frozen Config — never the App/repo/adapter/raw clients."""
    marker = tmp_path / "attrs.txt"
    p = _write_plugin(
        tmp_path,
        "probe",
        'name = "probe"\n\n'
        "def setup(api):\n"
        f"    with open({str(marker)!r}, 'w') as f:\n"
        "        f.write(','.join(sorted(a for a in vars(api) if not a.startswith('_'))))\n",
    )
    cfg = _cfg_with_paths(tmp_path, p)
    PluginLoader(cfg).load()
    attrs = marker.read_text(encoding="utf-8").split(",")
    assert set(attrs) == {
        "gate_features",
        "output_stages",
        "tools",
        "learners",
        "adapters",
        "hooks",
        "config",
    }


# ── no runtime mutation / hot reload ────────────────────────────────────────

def test_frozen_registries_reject_runtime_mutation(tmp_path):
    p = _write_plugin(tmp_path, "m", 'name = "m"\n\ndef setup(api):\n    pass\n')
    cfg = _cfg_with_paths(tmp_path, p)
    result = PluginLoader(cfg).load()
    with pytest.raises(RegistryError, match="frozen"):
        result.gate_features.register(type("X", (), {"name": "x"})())
    with pytest.raises(RegistryError, match="frozen"):
        result.output_stages.clear()
    with pytest.raises(RegistryError, match="frozen"):
        result.tools.unregister("reply")
    with pytest.raises(RegistryError, match="frozen"):
        result.hooks.on_event(lambda e: None)


def test_hook_bus_freeze_prevents_registration():
    bus = HookBus()
    bus.on_event(lambda e: None)
    bus.freeze()
    with pytest.raises(RegistryError, match="frozen"):
        bus.on_event(lambda e: None)
    with pytest.raises(RegistryError, match="frozen"):
        bus.pre_send(lambda o: None)
    with pytest.raises(RegistryError, match="frozen"):
        bus.on_cycle_end(lambda c, t, r: None)
    with pytest.raises(RegistryError, match="frozen"):
        bus.clear()


# ── exact-replay gate-feature fingerprint ───────────────────────────────────

def _identity() -> ChatIdentity:
    return ChatIdentity(CK, PlatformId("qq"), SelfId("bot-1"), "group")


def _msg(text: str, *, recv_ts: float = 100.0) -> Message:
    return Message(
        chat_key=CK,
        sender_id=SenderId("u1"),
        sender_name="user",
        is_self=False,
        text=text,
        id=MessageId("m1"),
        recv_ts=recv_ts,
    )


def _view_with_trace(trace_json: str | None) -> CorpusView:
    msg = _msg("hi")
    ev = EventId("ev-1")
    commit = CorpusMarker(
        record_type="commit",
        sequence=CommitSeq(1),
        chat_key=CK,
        event_id=ev,
        wake_kind=WakeKind.INBOUND,
    )
    dispatch = CorpusMarker(
        record_type="dispatch",
        sequence=DispatchId(1),
        chat_key=CK,
        cause=DispatchCause.INBOUND,
        commit_boundary=CommitSeq(1),
        state="completed",
        settled_ts=101.0,
        start_msg_id=MessageRowId(0),
        through_msg_id=MessageRowId(1),
        attached=(CommitSeq(1),),
        trace_json=trace_json,
    )
    return CorpusView(
        events_by_event_id={ev: AdapterEvent(type="message", payload=msg, ts=100.0)},
        commits=(commit,),
        dispatches=(dispatch,),
    )


def test_replay_fails_closed_on_gate_feature_fingerprint_mismatch():
    """Exact replay fails closed when the recorded trace's gate-feature
    fingerprint differs from the current gate's scoring composition."""
    fingerprint = _composition_fingerprint(Gate())
    fingerprint["gate_features"].append("extra_feature")
    recorded = {
        "chat_key": str(CK),
        "mode": "reply_necessity",
        "threshold": 8,
        "trigger_score": 80,
        "pending": 1,
        "decision": {"action": "trigger", "score": 100.0},
        "config": fingerprint,
    }
    view = _view_with_trace(json.dumps(recorded))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        replay_marker_schedule(
            view, chat_key=CK, identity=_identity(), cfg=Config()
        )


def test_replay_matches_when_fingerprint_is_identical():
    recorded = {
        "chat_key": str(CK),
        "mode": "reply_necessity",
        "threshold": 8,
        "trigger_score": 80,
        "pending": 1,
        "decision": {"action": "trigger", "score": 100.0},
        "config": _composition_fingerprint(Gate()),
    }
    view = _view_with_trace(json.dumps(recorded))
    result = replay_marker_schedule(
        view, chat_key=CK, identity=_identity(), cfg=Config()
    )
    assert result.decisions == 1
    # The re-scored trace carries the current fingerprint.
    assert result.traces[0].config["gate_features"] == [
        "relevance",
        "content",
        "pressure",
        "presence",
        "frequency",
    ]


def test_replay_rejects_missing_fingerprint_for_legacy_corpora():
    """Legacy traces without the complete composition proof fail closed."""
    recorded = {
        "chat_key": str(CK),
        "mode": "reply_necessity",
        "threshold": 8,
        "trigger_score": 80,
        "pending": 1,
        "decision": {"action": "trigger", "score": 100.0},
        "config": {"mode": "reply_necessity"},
    }
    view = _view_with_trace(json.dumps(recorded))
    with pytest.raises(ValueError, match="fingerprint missing"):
        replay_marker_schedule(view, chat_key=CK, identity=_identity(), cfg=Config())


def test_live_trace_carries_gate_feature_fingerprint():
    """Every live DecisionTrace config carries the ordered gate-feature
    fingerprint (the deterministic scoring identity)."""
    gate = Gate()
    from pretender.cycle import _trace_with_fingerprint
    from pretender.types import DecisionTrace

    trace = DecisionTrace(
        chat_key=CK, mode="reply_necessity", threshold=8, trigger_score=80, pending=1
    )
    traced = _trace_with_fingerprint(trace, gate)
    assert traced.config["gate_features"] == [
        "relevance",
        "content",
        "pressure",
        "presence",
        "frequency",
    ]


# ── App.build integration: abort before startup, wire after success ─────────

def test_app_build_aborts_on_plugin_failure(tmp_path):
    """A configured plugin that fails setup aborts App.build BEFORE any
    adapter/network/DB worker start — no usable partial App exists."""
    from pretender.app import App

    p = _write_plugin(
        tmp_path,
        "bad",
        'name = "bad"\n\ndef setup(api):\n    raise RuntimeError("setup boom")\n',
    )
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text(
        f'[plugins]\npaths = ["{p.name}"]\n[storage]\ndb_path = "{tmp_path / "t.db"}"\n',
        encoding="utf-8",
    )
    cfg = Config.load(cfg_file)
    with pytest.raises(RegistryError, match="setup failed"):
        # Live startup loads trusted plugin code and therefore surfaces setup
        # failure before any runtime component starts.
        App.build(cfg, dry_run=False)


def test_app_build_wires_plugin_gate_and_hooks(tmp_path):
    """A healthy plugin's gate feature, tool, and hook are wired into the
    built App (the staging registries are frozen and used)."""
    from pretender.app import App

    body = (
        'name = "ext"\n\n'
        "from pretender.tools import tool\n\n"
        "class MyFeature:\n"
        '    name = "my_feature"\n'
        "    def contribute(self, ctx):\n"
        "        return None\n\n"
        '@tool("my_tool")\n'
        "def my_tool(text: str) -> str:\n"
        '    """A plugin tool."""\n'
        "    return text\n\n"
        "def setup(api):\n"
        "    api.gate_features.register(MyFeature())\n"
        "    api.tools.register(my_tool)\n"
        "    api.hooks.on_event(lambda e: None)\n"
    )
    p = _write_plugin(tmp_path, "ext", body)
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text(
        f'[plugins]\npaths = ["{p.name}"]\n[storage]\ndb_path = "{tmp_path / "t.db"}"\n',
        encoding="utf-8",
    )
    cfg = Config.load(cfg_file)
    app = App.build(cfg, dry_run=False)
    try:
        # The plugin gate feature is in the runner's gate fingerprint.
        from pretender.cycle import _gate_fingerprint

        assert "my_feature" in _gate_fingerprint(app._cycle_fn._gate)
        # The plugin hook is registered on the frozen bus.
        assert len(app.hooks) == 1
        assert app.hooks.frozen
    finally:
        asyncio.run(app.shutdown())
