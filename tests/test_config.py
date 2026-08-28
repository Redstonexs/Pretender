"""Config: defaults, TOML loading, ${ENV} expansion, per-chat merge,
RuntimeOverlay, unknown-key rejection."""

from __future__ import annotations

import tomllib
import typing
from dataclasses import FrozenInstanceError, fields

import pytest

from pretender.config import (
    AgentConfig,
    BudgetConfig,
    ChatConfig,
    ContextConfig,
    Config,
    DriftConfig,
    GateConfig,
    LearnConfig,
    MediaConfig,
    OutputConfig,
    LLMProfile,
    OneBotConfig,
    RuntimeOverlay,
    load_config,
)
from pretender.errors import ConfigError


# ── defaults / empty TOML ───────────────────────────────────────────────────

def test_empty_toml_produces_working_config():
    cfg = Config.loads("")
    assert cfg.bot.name == "麦麦"
    assert cfg.gate.mode == "reply_necessity"
    assert cfg.gate.threshold == 8
    assert cfg.gate.trigger_score == 80
    assert cfg.gate.backoff.base_s == 15.0
    assert cfg.gate.backoff.cap_s == 300.0
    assert cfg.output.pipeline == ("sanitize", "split", "typo")
    assert cfg.adapter.onebot.mode == "reverse_ws"
    assert cfg.adapter.onebot.port == 3001
    assert cfg.storage.db_path == "data/pretender.db"
    assert cfg.log.max_bytes == 10 * 1024 * 1024
    assert cfg.chats == ()


def test_load_config_none_returns_defaults():
    cfg = load_config(None)
    assert cfg.bot.name == "麦麦"


def test_config_is_frozen():
    cfg = Config()
    with pytest.raises(FrozenInstanceError):
        cfg.bot.name = "other"  # type: ignore[misc]


# ── sample TOML ─────────────────────────────────────────────────────────────

def test_sample_toml_is_syntactically_valid(sample_config_path):
    tomllib.loads(sample_config_path.read_text(encoding="utf-8"))


def test_sample_toml_loads_with_env(monkeypatch, sample_config_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dash")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-sf")
    monkeypatch.setenv("ONEBOT_ACCESS_TOKEN", "ob-token")
    cfg = load_config(sample_config_path)
    assert cfg.llm.profile("planner").api_key == "sk-test"
    assert cfg.llm.profile("reply").model == "deepseek-chat"
    assert cfg.llm.profile("vision").base_url.startswith("https://dashscope")
    assert cfg.gate.threshold == 8
    assert cfg.chats[0].key == "qq:group:123456"
    assert cfg.chats[0].gate.threshold == 12  # type: ignore[union-attr]
    assert cfg.chats[0].gate.frequency == 0.6  # type: ignore[union-attr]
    assert cfg.adapter.onebot.self_id == "10001"
    assert cfg.adapter.onebot.access_token == "ob-token"


def test_docker_example_is_minimal_live_config(monkeypatch, sample_config_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-docker-test")
    monkeypatch.setenv("ONEBOT_ACCESS_TOKEN", "ob-docker-test")

    cfg = load_config(sample_config_path.parent / "config.docker.example.toml")

    assert set(cfg.llm.profiles) == {"planner", "reply"}
    assert cfg.llm.profile("planner").base_url == "https://api.deepseek.com/v1"
    assert cfg.llm.profile("planner").model == "deepseek-chat"
    assert cfg.llm.profile("planner").api_key == "sk-docker-test"
    assert cfg.llm.profile("reply").base_url == "https://api.deepseek.com/v1"
    assert cfg.llm.profile("reply").model == "deepseek-chat"
    assert cfg.llm.profile("reply").api_key == "sk-docker-test"
    assert cfg.adapter.name == "onebot"
    assert cfg.adapter.onebot.mode == "reverse_ws"
    assert cfg.adapter.onebot.host == "127.0.0.1"
    assert cfg.adapter.onebot.port == 3001
    assert cfg.adapter.onebot.path == "/onebot/v11/ws"
    assert cfg.adapter.onebot.access_token == "ob-docker-test"
    assert cfg.adapter.onebot.self_id is None
    assert cfg.storage.db_path == "data/pretender.db"


# ── ${ENV} expansion ────────────────────────────────────────────────────────

def test_env_expansion(monkeypatch):
    monkeypatch.setenv("PRETENDER_TEST_KEY", "secret-value")
    cfg = Config.loads('[llm.profiles.planner]\napi_key = "${PRETENDER_TEST_KEY}"')
    assert cfg.llm.profile("planner").api_key == "secret-value"


def test_env_expansion_missing_variable_raises(monkeypatch):
    monkeypatch.delenv("PRETENDER_NEVER_SET", raising=False)
    with pytest.raises(ConfigError, match="PRETENDER_NEVER_SET"):
        Config.loads('[llm.profiles.planner]\napi_key = "${PRETENDER_NEVER_SET}"')


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_llm_api_key_env_value_rejected(monkeypatch, value):
    monkeypatch.setenv("PRETENDER_BLANK_API_KEY", value)
    with pytest.raises(ConfigError, match="PRETENDER_BLANK_API_KEY"):
        Config.loads(
            '[llm.profiles.planner]\napi_key = "${PRETENDER_BLANK_API_KEY}"'
        )


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_onebot_access_token_env_value_rejected(monkeypatch, value):
    monkeypatch.setenv("PRETENDER_BLANK_ACCESS_TOKEN", value)
    with pytest.raises(ConfigError, match="PRETENDER_BLANK_ACCESS_TOKEN"):
        Config.loads(
            '[adapter.onebot]\naccess_token = "${PRETENDER_BLANK_ACCESS_TOKEN}"'
        )


def test_non_secret_empty_env_value_remains_allowed(monkeypatch):
    monkeypatch.setenv("PRETENDER_EMPTY_BOT_NAME", "")
    cfg = Config.loads('[bot]\nname = "${PRETENDER_EMPTY_BOT_NAME}"')
    assert cfg.bot.name == ""


def test_literal_api_key_rejected_at_load():
    """Secrets must resolve from ${ENV}: a literal api_key value is a
    ConfigError at load time, never silently accepted."""
    with pytest.raises(ConfigError, match=r"api_key.*\$\{ENV\}"):
        Config.loads('[llm.profiles.planner]\napi_key = "sk-literal-secret"')
    with pytest.raises(ConfigError, match=r"api_key.*\$\{ENV\}"):
        Config.from_dict(
            {"llm": {"profiles": {"planner": {"api_key": "sk-literal-secret"}}}}
        )


def test_literal_api_key_rejected_in_any_profile(monkeypatch):
    monkeypatch.setenv("PRETENDER_OK_KEY", "sk-ok")
    # one env-resolved profile is fine...
    cfg = Config.loads('[llm.profiles.planner]\napi_key = "${PRETENDER_OK_KEY}"')
    assert cfg.llm.profile("planner").api_key == "sk-ok"
    # ...but a literal in ANY profile is rejected
    with pytest.raises(ConfigError, match=r"api_key.*\$\{ENV\}"):
        Config.loads(
            '[llm.profiles.planner]\napi_key = "${PRETENDER_OK_KEY}"\n'
            '[llm.profiles.reply]\napi_key = "sk-literal"'
        )


def test_literal_access_token_rejected_at_load():
    """The OneBot access token is a ${ENV}-only secret: a literal value is a
    ConfigError at load time, never silently accepted."""
    with pytest.raises(ConfigError, match=r"access_token.*\$\{ENV\}"):
        Config.loads('[adapter.onebot]\naccess_token = "sk-literal"')
    with pytest.raises(ConfigError, match=r"access_token.*\$\{ENV\}"):
        Config.from_dict({"adapter": {"onebot": {"access_token": "sk-literal"}}})


def test_access_token_env_expansion(monkeypatch):
    monkeypatch.setenv("ONEBOT_TOKEN", "secret")
    cfg = Config.loads('[adapter.onebot]\naccess_token = "${ONEBOT_TOKEN}"')
    assert cfg.adapter.onebot.access_token == "secret"
    with pytest.raises(ConfigError, match="ONEBOT_NEVER_SET"):
        Config.loads('[adapter.onebot]\naccess_token = "${ONEBOT_NEVER_SET}"')


def test_onebot_self_id_validation():
    with pytest.raises(ConfigError, match="self_id"):
        OneBotConfig(self_id="")
    with pytest.raises(ConfigError, match="self_id"):
        OneBotConfig(self_id="   ")
    assert OneBotConfig(self_id="10001").self_id == "10001"


def test_onebot_scheme_validation():
    with pytest.raises(ConfigError, match="scheme"):
        OneBotConfig(scheme="http")
    assert OneBotConfig(scheme="wss").scheme == "wss"


def test_onebot_ping_timeout_validation():
    with pytest.raises(ConfigError, match="ping_timeout_s"):
        OneBotConfig(ping_timeout_s=0)
    with pytest.raises(ConfigError, match="ping_timeout_s"):
        OneBotConfig(ping_timeout_s=-1)


def test_forward_ws_with_token_remote_host_rejected():
    """No plaintext remote bearer traffic: a forward ws:// endpoint with an
    access token must be explicitly local-only or wss://."""
    with pytest.raises(ConfigError, match="local-only"):
        OneBotConfig(mode="ws", host="example.com", scheme="ws", access_token="tok")
    with pytest.raises(ConfigError, match="local-only"):
        OneBotConfig(mode="ws", host="onebot.example.com", scheme="ws", access_token="tok")


def test_forward_wss_with_token_remote_host_allowed():
    cfg = OneBotConfig(mode="ws", host="example.com", scheme="wss", access_token="tok")
    assert cfg.scheme == "wss"


def test_forward_ws_with_token_local_host_allowed():
    for host in ("127.0.0.1", "localhost", "10.0.0.5", "192.168.1.10"):
        cfg = OneBotConfig(mode="ws", host=host, scheme="ws", access_token="tok")
        assert cfg.host == host


def test_reverse_ws_with_token_remote_host_rejected_without_tls():
    """A bearer token is insufficient for plaintext remote reverse WS."""
    with pytest.raises(ConfigError, match="loopback-only"):
        OneBotConfig(
            mode="reverse_ws", host="0.0.0.0", scheme="ws", access_token="tok"
        )


def test_reverse_ws_non_loopback_rejected_without_tls():
    """Reverse_ws remains loopback-only until TLS support is available."""
    with pytest.raises(ConfigError, match="loopback-only"):
        OneBotConfig(mode="reverse_ws", host="0.0.0.0", access_token=None)
    with pytest.raises(ConfigError, match="loopback-only"):
        OneBotConfig(mode="reverse_ws", host="192.168.1.10", access_token=None)
    # loopback binds remain supported.
    assert OneBotConfig(mode="reverse_ws", host="127.0.0.1", access_token=None)
    assert OneBotConfig(mode="reverse_ws", host="localhost", access_token=None)


def test_onebot_media_concurrency_validation():
    with pytest.raises(ConfigError, match="media_concurrency"):
        OneBotConfig(media_concurrency=0)
    with pytest.raises(ConfigError, match="media_concurrency"):
        OneBotConfig(media_concurrency=-1)
    with pytest.raises(ConfigError, match="media_concurrency"):
        OneBotConfig(media_concurrency=True)
    assert OneBotConfig(media_concurrency=4).media_concurrency == 4


def test_env_expansion_success_for_api_key(monkeypatch):
    monkeypatch.setenv("PRETENDER_TEST_KEY", "secret-value")
    cfg = Config.loads('[llm.profiles.planner]\napi_key = "${PRETENDER_TEST_KEY}"')
    assert cfg.llm.profile("planner").api_key == "secret-value"


def test_non_secret_literal_strings_still_allowed(monkeypatch):
    """Only secret fields require ${ENV}; ordinary strings stay literal."""
    cfg = Config.loads('[bot]\nname = "麦麦"\n[llm.profiles.planner]\nmodel = "m"')
    assert cfg.bot.name == "麦麦"
    assert cfg.llm.profile("planner").model == "m"


# ── Gate 5: embed profile revision / space identity ─────────────────────────

def test_embed_profile_revision_and_space_id():
    prof = LLMProfile(model="m", revision="r1")
    assert prof.revision == "r1"
    assert prof.space_id() == "m@r1"
    # No revision -> no canonical space.
    assert LLMProfile(model="m").space_id() is None
    # A blank revision is rejected.
    with pytest.raises(ConfigError, match="revision"):
        LLMProfile(revision="   ")
    # A non-string revision is rejected.
    with pytest.raises(ConfigError, match="revision"):
        LLMProfile(revision=typing.cast(str, 123))


def test_config_loads_embed_revision():
    cfg = Config.loads('[llm.profiles.embed]\nmodel = "m"\nrevision = "r1"')
    assert cfg.llm.profile("embed").revision == "r1"
    assert cfg.llm.profile("embed").space_id() == "m@r1"
    # The revision round-trips through from_dict/from_dict.
    cfg2 = Config.from_dict(cfg.to_dict())
    assert cfg2.llm.profile("embed").space_id() == "m@r1"


def test_plugin_cfg_json_api_key_field_is_not_a_secret():
    """A plugin-owned ``cfg_json`` field named ``api_key`` is arbitrary data,
    never treated as a config secret."""
    cfg = Config.loads(
        '[[chats]]\nkey = "qq:group:1"\ncfg_json = { api_key = "plugin-data" }'
    )
    assert cfg.for_chat("qq:group:1").cfg_json == {"api_key": "plugin-data"}


def test_env_expansion_applies_to_any_string(monkeypatch):
    monkeypatch.setenv("PRETENDER_BOT_NAME", "测试")
    cfg = Config.loads('[bot]\nname = "${PRETENDER_BOT_NAME}"')
    assert cfg.bot.name == "测试"


def test_partial_env_reference_is_left_alone(monkeypatch):
    monkeypatch.setenv("PRETENDER_PARTIAL", "x")
    cfg = Config.loads('[llm.profiles.planner]\nmodel = "pre-${PRETENDER_PARTIAL}"')
    assert cfg.llm.profile("planner").model == "pre-${PRETENDER_PARTIAL}"


# ── unknown-key rejection ───────────────────────────────────────────────────

def test_unknown_top_level_key_rejected():
    with pytest.raises(ConfigError, match="unknown config key"):
        Config.loads("[bogus]\nx = 1")


def test_unknown_nested_key_rejected():
    with pytest.raises(ConfigError, match="unknown config key"):
        Config.loads("[gate]\nbogus = 1")


def test_unknown_profile_key_rejected():
    with pytest.raises(ConfigError, match="unknown config key"):
        Config.loads("[llm.profiles.planner]\nbogus = 1")


def test_unknown_chat_override_key_rejected():
    with pytest.raises(ConfigError, match="unknown config key"):
        Config.loads('[[chats]]\nkey = "qq:group:1"\nbogus = 2')


def test_invalid_toml_raises_config_error():
    with pytest.raises(ConfigError, match="invalid TOML"):
        Config.loads("this is not toml = =")


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "nope.toml")


def test_wrong_scalar_type_rejected():
    with pytest.raises(ConfigError, match="expected an integer"):
        Config.loads("[gate]\nthreshold = \"eight\"")


# ── named LLM profiles ──────────────────────────────────────────────────────

def test_named_profiles_and_defaults():
    cfg = Config.loads("[llm.profiles.reply]\nmodel = \"deepseek-chat\"")
    profile = cfg.llm.profile("reply")
    assert profile.model == "deepseek-chat"
    assert profile.base_url == "https://api.deepseek.com/v1"  # inherited default
    assert profile.temperature == 0.7
    assert profile.max_tokens == 1200


def test_missing_profile_raises():
    cfg = Config()
    with pytest.raises(ConfigError, match="no LLM profile"):
        cfg.llm.profile("nope")


# ── per-chat override merge ─────────────────────────────────────────────────

def test_for_chat_merges_override_over_defaults():
    cfg = Config.loads(
        '[[chats]]\nkey = "qq:group:123456"\ngate = { threshold = 12, frequency = 0.6 }'
    )
    chat = cfg.for_chat("qq:group:123456")
    assert chat.gate.threshold == 12
    assert chat.gate.frequency == 0.6
    assert chat.gate.mode == "reply_necessity"  # inherited from top level
    assert chat.gate.trigger_score == 80
    assert chat.gate.backoff.base_s == 15.0


def test_for_chat_coerces_all_optional_sections_to_frozen_dataclasses():
    cfg = Config.loads(
        '[[chats]]\n'
        'key = "qq:group:typed"\n'
        '[chats.gate]\nthreshold = 12\n'
        '[chats.drift]\nlevel = "wild"\n'
        '[chats.output]\nmax_split = 5\n'
        '[chats.context]\nmax_context_size = 60\n'
        '[chats.budget]\ndaily_cap = 50\n'
        '[chats.agent]\nmax_execution_s = 600\n'
        '[chats.learn]\nenabled = true\n'
        '[chats.media]\nenabled = true\n'
    )
    override = cfg.chats[0]
    section_types = (
        (override.gate, GateConfig),
        (override.drift, DriftConfig),
        (override.output, OutputConfig),
        (override.context, ContextConfig),
        (override.budget, BudgetConfig),
        (override.agent, AgentConfig),
        (override.learn, LearnConfig),
        (override.media, MediaConfig),
    )
    for section, section_type in section_types:
        assert isinstance(section, section_type)
        field_name = fields(section_type)[0].name
        with pytest.raises(FrozenInstanceError):
            setattr(section, field_name, getattr(section, field_name))

    chat = cfg.for_chat("qq:group:typed")
    assert isinstance(chat, ChatConfig)
    for section, section_type in zip(
        (
            chat.gate,
            chat.drift,
            chat.output,
            chat.context,
            chat.budget,
            chat.agent,
            chat.learn,
            chat.media,
        ),
        (
            GateConfig,
            DriftConfig,
            OutputConfig,
            ContextConfig,
            BudgetConfig,
            AgentConfig,
            LearnConfig,
            MediaConfig,
        ),
    ):
        assert isinstance(section, section_type)
        field_name = fields(section_type)[0].name
        with pytest.raises(FrozenInstanceError):
            setattr(section, field_name, getattr(section, field_name))
    assert chat.gate.threshold == 12
    assert chat.drift.level == "wild"
    assert chat.output.max_split == 5
    assert chat.context.max_context_size == 60
    assert chat.budget.daily_cap == 50
    assert chat.agent.max_execution_s == 600.0
    assert chat.learn.enabled is True
    assert chat.media.enabled is True


def test_for_chat_without_override_returns_defaults():
    cfg = Config()
    chat = cfg.for_chat("qq:group:999")
    assert chat.gate.threshold == 8
    assert chat.cfg_json == {}


def test_for_chat_deep_merges_partial_override_over_non_default_parent():
    """A partial per-chat override must deep-merge: parent non-default
    values (including nested sections) survive the override."""
    cfg = Config.loads(
        '[gate]\nthreshold = 12\n[gate.backoff]\ncap_s = 500\n'
        '[[chats]]\nkey = "qq:group:123456"\ngate = { frequency = 0.6 }'
    )
    chat = cfg.for_chat("qq:group:123456")
    assert chat.gate.frequency == 0.6       # override applied
    assert chat.gate.threshold == 12        # parent non-default preserved
    assert chat.gate.mode == "reply_necessity"  # parent default preserved
    assert chat.gate.backoff.cap_s == 500   # nested parent non-default preserved
    assert chat.gate.backoff.base_s == 15.0


def test_for_chat_deep_merges_nested_override():
    cfg = Config.loads(
        '[gate.backoff]\nbase_s = 30\n'
        '[[chats]]\nkey = "qq:group:1"\ngate = { backoff = { cap_s = 900 } }'
    )
    chat = cfg.for_chat("qq:group:1")
    assert chat.gate.backoff.cap_s == 900   # nested override applied
    assert chat.gate.backoff.base_s == 30   # nested parent preserved
    assert chat.gate.threshold == 8         # untouched parent default


def test_for_chat_full_override_still_replaces():
    cfg = Config.loads(
        '[gate]\nthreshold = 12\n'
        '[[chats]]\nkey = "qq:group:1"\n'
        'gate = { threshold = 20, frequency = 0.9, mode = "frequency" }'
    )
    chat = cfg.for_chat("qq:group:1")
    assert chat.gate.threshold == 20
    assert chat.gate.frequency == 0.9
    assert chat.gate.mode == "frequency"


def test_for_chat_explicit_default_resets_non_default_parent():
    """Explicit field presence: a chat can reset a non-default global to
    its default (mode frequency -> reply_necessity, threshold 12 -> 8)."""
    cfg = Config.loads(
        '[gate]\nthreshold = 12\nmode = "frequency"\n'
        '[[chats]]\nkey = "qq:group:1"\n'
        'gate = { threshold = 8, mode = "reply_necessity" }'
    )
    chat = cfg.for_chat("qq:group:1")
    assert chat.gate.threshold == 8  # explicit default resets the parent's 12
    assert chat.gate.mode == "reply_necessity"  # explicit default
    assert chat.gate.frequency == 1.0  # omitted: inherits the parent default


def test_for_chat_nested_explicit_default_reset():
    cfg = Config.loads(
        '[gate.backoff]\ncap_s = 500\n'
        '[[chats]]\nkey = "qq:group:1"\ngate = { backoff = { cap_s = 300 } }'
    )
    chat = cfg.for_chat("qq:group:1")
    assert chat.gate.backoff.cap_s == 300  # explicit default resets 500
    assert chat.gate.backoff.base_s == 15.0  # omitted: parent default


def test_for_chat_presence_survives_round_trip():
    """to_dict/from_dict must preserve explicit field presence: a partial
    override still deep-merges after a round-trip."""
    cfg = Config.loads(
        '[gate]\nthreshold = 12\n'
        '[[chats]]\nkey = "qq:group:1"\ngate = { frequency = 0.6 }'
    )
    rebuilt = Config.from_dict(cfg.to_dict())
    chat = rebuilt.for_chat("qq:group:1")
    assert chat.gate.frequency == 0.6
    assert chat.gate.threshold == 12  # parent non-default preserved
    assert chat.gate.mode == "reply_necessity"  # parent default preserved


def test_chat_override_cfg_json():
    cfg = Config.loads(
        '[[chats]]\nkey = "qq:group:1"\ncfg_json = { note = "hi" }'
    )
    assert cfg.for_chat("qq:group:1").cfg_json == {"note": "hi"}


def test_duplicate_chat_keys_rejected():
    with pytest.raises(ConfigError, match="duplicate chat override"):
        Config.loads(
            '[[chats]]\nkey = "qq:group:1"\n[[chats]]\nkey = "qq:group:1"'
        )


def test_gate_mode_validation():
    with pytest.raises(ConfigError, match="gate.mode"):
        Config.loads("[gate]\nmode = \"sometimes\"")


# ── numeric gate validation (fail closed at load) ───────────────────────────

def test_gate_threshold_must_be_positive_int():
    for bad in (0, -1):
        with pytest.raises(ConfigError, match="gate.threshold"):
            Config.loads(f"[gate]\nthreshold = {bad}")
    with pytest.raises(ConfigError, match="threshold"):
        Config.loads("[gate]\nthreshold = 8.5")
    with pytest.raises(ConfigError, match="threshold"):
        Config.loads("[gate]\nthreshold = true")
    # boundary: 1 is legal
    assert Config.loads("[gate]\nthreshold = 1").gate.threshold == 1


def test_gate_trigger_score_must_be_positive_int():
    for bad in (0, -5):
        with pytest.raises(ConfigError, match="gate.trigger_score"):
            Config.loads(f"[gate]\ntrigger_score = {bad}")
    with pytest.raises(ConfigError, match="trigger_score"):
        Config.loads("[gate]\ntrigger_score = 80.5")
    assert Config.loads("[gate]\ntrigger_score = 1").gate.trigger_score == 1


def test_gate_frequency_must_be_in_unit_interval():
    for bad in (-0.1, 1.1, 2.0):
        with pytest.raises(ConfigError, match="gate.frequency"):
            Config.loads(f"[gate]\nfrequency = {bad}")
    with pytest.raises(ConfigError, match="frequency"):
        Config.loads("[gate]\nfrequency = \"high\"")
    # boundaries are legal
    assert Config.loads("[gate]\nfrequency = 0").gate.frequency == 0.0
    assert Config.loads("[gate]\nfrequency = 1").gate.frequency == 1.0


def test_gate_backoff_base_and_cap_must_be_finite_nonnegative():
    with pytest.raises(ConfigError, match="gate.backoff.base_s"):
        Config.loads("[gate.backoff]\nbase_s = -1")
    with pytest.raises(ConfigError, match="gate.backoff.cap_s"):
        Config.loads("[gate.backoff]\ncap_s = -1")
    # non-finite values (TOML cannot express them; from_dict can)
    with pytest.raises(ConfigError, match="gate.backoff.base_s"):
        Config.from_dict({"gate": {"backoff": {"base_s": float("inf")}}})
    with pytest.raises(ConfigError, match="gate.backoff.base_s"):
        Config.from_dict({"gate": {"backoff": {"base_s": float("nan")}}})
    with pytest.raises(ConfigError, match="gate.backoff.cap_s"):
        Config.from_dict({"gate": {"backoff": {"cap_s": float("inf")}}})
    # zero base is legal (backoff stays zero until the cap)
    assert Config.loads("[gate.backoff]\nbase_s = 0").gate.backoff.base_s == 0.0


def test_gate_backoff_cap_must_be_at_least_base():
    with pytest.raises(ConfigError, match="cap_s"):
        Config.loads("[gate.backoff]\nbase_s = 300\ncap_s = 150")
    # equal is legal
    cfg = Config.loads("[gate.backoff]\nbase_s = 60\ncap_s = 60")
    assert cfg.gate.backoff.cap_s == cfg.gate.backoff.base_s == 60.0


def test_gate_backoff_start_count_must_be_nonnegative_int():
    with pytest.raises(ConfigError, match="start_count"):
        Config.loads("[gate.backoff]\nstart_count = -1")
    with pytest.raises(ConfigError, match="start_count"):
        Config.loads("[gate.backoff]\nstart_count = 1.5")
    with pytest.raises(ConfigError, match="start_count"):
        Config.loads("[gate.backoff]\nstart_count = true")
    assert Config.loads("[gate.backoff]\nstart_count = 0").gate.backoff.start_count == 0


def test_unsafe_per_chat_override_rejected_at_load():
    # Per-chat overrides are validated at load, exactly like top-level —
    # an override that is invalid on its own (base 600 > default cap 300)
    # never loads.
    with pytest.raises(ConfigError, match="gate.threshold"):
        Config.loads('[[chats]]\nkey = "qq:group:1"\ngate = { threshold = 0 }')
    with pytest.raises(ConfigError, match="gate.frequency"):
        Config.loads('[[chats]]\nkey = "qq:group:1"\ngate = { frequency = 2.0 }')
    with pytest.raises(ConfigError, match="cap_s"):
        Config.loads(
            '[[chats]]\nkey = "qq:group:1"\n'
            'gate = { backoff = { base_s = 900, cap_s = 60 } }'
        )
    with pytest.raises(ConfigError, match="cap_s"):
        Config.loads(
            '[[chats]]\nkey = "qq:group:1"\ngate = { backoff = { base_s = 600 } }'
        )


def test_unsafe_merged_override_rejected_at_for_chat():
    # A partial override that is individually valid (cap 50 >= base 15) and
    # a parent that is individually valid (cap 500 >= base 100) can still
    # produce an invalid MERGED config (cap 50 < base 100) — fail closed at
    # merge time.
    cfg = Config.loads(
        '[gate.backoff]\nbase_s = 100\ncap_s = 500\n'
        '[[chats]]\nkey = "qq:group:1"\ngate = { backoff = { cap_s = 50 } }'
    )
    with pytest.raises(ConfigError, match="cap_s"):
        cfg.for_chat("qq:group:1")


def test_unsafe_overlay_value_rejected():
    overlay = RuntimeOverlay()
    overlay.set("gate.threshold", 0)
    with pytest.raises(ConfigError, match="gate.threshold"):
        overlay.apply(Config())
    overlay2 = RuntimeOverlay()
    overlay2.set("gate.frequency", 1.5)
    with pytest.raises(ConfigError, match="gate.frequency"):
        overlay2.apply(Config())


# ── RuntimeOverlay ──────────────────────────────────────────────────────────

def test_overlay_applies_dotted_path():
    cfg = Config()
    overlay = RuntimeOverlay()
    overlay.set("gate.threshold", 12)
    overlay.set("bot.name", "测试")
    merged = overlay.apply(cfg)
    assert merged.gate.threshold == 12
    assert merged.bot.name == "测试"
    # original untouched, overlay reusable
    assert cfg.gate.threshold == 8
    assert overlay.apply(cfg).gate.threshold == 12


def test_overlay_unknown_path_raises():
    overlay = RuntimeOverlay()
    overlay.set("gate.bogus", 1)
    with pytest.raises(ConfigError, match="does not exist"):
        overlay.apply(Config())


def test_overlay_get_clear_items():
    overlay = RuntimeOverlay()
    assert overlay.get("gate.threshold") is None
    overlay.set("gate.threshold", 9)
    assert overlay.get("gate.threshold") == 9
    assert overlay.items() == (("gate.threshold", 9),)
    overlay.clear()
    assert overlay.items() == ()


def test_overlay_invalid_path_rejected():
    overlay = RuntimeOverlay()
    with pytest.raises(ConfigError, match="invalid overlay path"):
        overlay.set("", 1)
    with pytest.raises(ConfigError, match="invalid overlay path"):
        overlay.set("gate..threshold", 1)


def test_overlay_merges_into_nested_profile():
    cfg = Config.loads('[llm.profiles.planner]\nmodel = "deepseek-chat"')
    overlay = RuntimeOverlay()
    overlay.set("llm.profiles.planner.model", "deepseek-reasoner")
    merged = overlay.apply(cfg)
    assert merged.llm.profile("planner").model == "deepseek-reasoner"
    assert cfg.llm.profile("planner").model == "deepseek-chat"  # original frozen


# ── round-trip ──────────────────────────────────────────────────────────────

def test_to_dict_from_dict_round_trip():
    cfg = Config.loads(
        '[[chats]]\nkey = "qq:group:1"\ngate = { threshold = 12 }'
    )
    rebuilt = Config.from_dict(cfg.to_dict())
    assert rebuilt == cfg
    assert rebuilt.for_chat("qq:group:1").gate.threshold == 12


# ── [context] config ────────────────────────────────────────────────────────

def test_context_defaults():
    cfg = Config()
    assert cfg.context.max_context_size == 40
    assert cfg.context.max_image_num == 3
    assert cfg.context.keep_recent == 0


def test_context_loads_from_toml():
    cfg = Config.loads(
        "[context]\nmax_context_size = 60\nmax_image_num = 5\nkeep_recent = 2"
    )
    assert cfg.context.max_context_size == 60
    assert cfg.context.max_image_num == 5
    assert cfg.context.keep_recent == 2


def test_context_validation():
    with pytest.raises(ConfigError, match="context.max_context_size"):
        Config.loads("[context]\nmax_context_size = 0")
    with pytest.raises(ConfigError, match="context.max_context_size"):
        Config.loads("[context]\nmax_context_size = -1")
    with pytest.raises(ConfigError, match="context.max_image_num"):
        Config.loads("[context]\nmax_image_num = -1")
    with pytest.raises(ConfigError, match="context.keep_recent"):
        Config.loads("[context]\nkeep_recent = -1")
    with pytest.raises(ConfigError, match="max_context_size"):
        Config.loads("[context]\nmax_context_size = 1.5")
    # boundaries are legal
    assert Config.loads("[context]\nmax_context_size = 1").context.max_context_size == 1
    assert Config.loads("[context]\nmax_image_num = 0").context.max_image_num == 0
    assert Config.loads("[context]\nkeep_recent = 0").context.keep_recent == 0


def test_context_per_chat_partial_override():
    cfg = Config.loads(
        "[context]\nmax_context_size = 60\n"
        '[[chats]]\nkey = "qq:group:1"\ncontext = { max_image_num = 7 }'
    )
    chat = cfg.for_chat("qq:group:1")
    assert chat.context.max_image_num == 7   # override applied
    assert chat.context.max_context_size == 60  # parent non-default preserved
    assert chat.context.keep_recent == 0     # parent default preserved


def test_context_overlay():
    overlay = RuntimeOverlay()
    overlay.set("context.max_context_size", 80)
    merged = overlay.apply(Config())
    assert merged.context.max_context_size == 80
    with pytest.raises(ConfigError, match="context.max_context_size"):
        overlay2 = RuntimeOverlay()
        overlay2.set("context.max_context_size", 0)
        overlay2.apply(Config())


# ── [budget] config ─────────────────────────────────────────────────────────

def test_budget_defaults():
    cfg = Config()
    assert cfg.budget.daily_cap == 100
    assert [r.action for r in cfg.budget.rungs] == ["warn", "degrade", "stop"]
    assert [r.at for r in cfg.budget.rungs] == [0.8, 0.9, 1.0]


def test_budget_loads_from_toml():
    cfg = Config.loads(
        "[budget]\ndaily_cap = 50\n"
        '[[budget.rungs]]\nat = 0.5\naction = "warn"\n'
        '[[budget.rungs]]\nat = 1.0\naction = "stop"\n'
    )
    assert cfg.budget.daily_cap == 50
    assert [(r.at, r.action) for r in cfg.budget.rungs] == [(0.5, "warn"), (1.0, "stop")]


def test_budget_validation():
    with pytest.raises(ConfigError, match="budget.daily_cap"):
        Config.loads("[budget]\ndaily_cap = 0")
    with pytest.raises(ConfigError, match="budget.daily_cap"):
        Config.loads("[budget]\ndaily_cap = -5")
    with pytest.raises(ConfigError, match="budget.rungs.at"):
        Config.loads("[budget]\nrungs = [{ at = 1.5, action = \"warn\" }]")
    with pytest.raises(ConfigError, match="budget.rungs.action"):
        Config.loads("[budget]\nrungs = [{ at = 0.5, action = \"pause\" }]")
    with pytest.raises(ConfigError, match="sorted ascending"):
        Config.loads(
            "[budget]\nrungs = [{ at = 1.0, action = \"stop\" }, { at = 0.5, action = \"warn\" }]"
        )


def test_budget_per_chat_partial_override():
    cfg = Config.loads(
        "[budget]\ndaily_cap = 200\n"
        '[[chats]]\nkey = "qq:group:1"\nbudget = { daily_cap = 50 }'
    )
    chat = cfg.for_chat("qq:group:1")
    assert chat.budget.daily_cap == 50   # override applied
    assert [r.action for r in chat.budget.rungs] == ["warn", "degrade", "stop"]  # inherited


def test_budget_overlay():
    overlay = RuntimeOverlay()
    overlay.set("budget.daily_cap", 300)
    merged = overlay.apply(Config())
    assert merged.budget.daily_cap == 300
    with pytest.raises(ConfigError, match="budget.daily_cap"):
        overlay2 = RuntimeOverlay()
        overlay2.set("budget.daily_cap", 0)
        overlay2.apply(Config())


def test_context_budget_round_trip():
    cfg = Config.loads(
        "[context]\nmax_context_size = 60\n[budget]\ndaily_cap = 50\n"
        '[[chats]]\nkey = "qq:group:1"\ncontext = { max_image_num = 7 }'
    )
    rebuilt = Config.from_dict(cfg.to_dict())
    assert rebuilt == cfg
    assert rebuilt.for_chat("qq:group:1").context.max_image_num == 7


# ── [agent] config ──────────────────────────────────────────────────────────

def test_agent_defaults():
    cfg = Config()
    assert cfg.agent.dispatch_lease_s == 60.0
    assert cfg.agent.max_execution_s == 300.0
    assert cfg.agent.retry_delay_s == 30.0
    assert cfg.agent.fallback_profile is None


def test_agent_loads_from_toml():
    cfg = Config.loads(
        "[agent]\ndispatch_lease_s = 90\nmax_execution_s = 600\n"
        "retry_delay_s = 45\nfallback_profile = \"reply\""
    )
    assert cfg.agent.dispatch_lease_s == 90.0
    assert cfg.agent.max_execution_s == 600.0
    assert cfg.agent.retry_delay_s == 45.0
    assert cfg.agent.fallback_profile == "reply"


def test_agent_validation():
    for key in ("dispatch_lease_s", "max_execution_s", "retry_delay_s"):
        for bad in (0, -1):
            with pytest.raises(ConfigError, match=f"agent.{key}"):
                Config.loads(f"[agent]\n{key} = {bad}")
        with pytest.raises(ConfigError, match=f"agent.{key}"):
            Config.from_dict({"agent": {key: float("inf")}})
        with pytest.raises(ConfigError, match=f"agent.{key}"):
            Config.from_dict({"agent": {key: float("nan")}})
    with pytest.raises(ConfigError, match="fallback_profile"):
        Config.loads("[agent]\nfallback_profile = 7")
    # boundaries are legal
    cfg = Config.loads("[agent]\ndispatch_lease_s = 1")
    assert cfg.agent.dispatch_lease_s == 1.0


def test_agent_per_chat_partial_override():
    cfg = Config.loads(
        "[agent]\nmax_execution_s = 600\n"
        '[[chats]]\nkey = "qq:group:1"\nagent = { retry_delay_s = 90 }'
    )
    chat = cfg.for_chat("qq:group:1")
    assert chat.agent.retry_delay_s == 90.0   # override applied
    assert chat.agent.max_execution_s == 600.0  # parent non-default preserved
    assert chat.agent.dispatch_lease_s == 60.0  # parent default preserved
    assert chat.agent.fallback_profile is None


def test_agent_overlay():
    overlay = RuntimeOverlay()
    overlay.set("agent.dispatch_lease_s", 120)
    merged = overlay.apply(Config())
    assert merged.agent.dispatch_lease_s == 120.0
    with pytest.raises(ConfigError, match="agent.dispatch_lease_s"):
        overlay2 = RuntimeOverlay()
        overlay2.set("agent.dispatch_lease_s", 0)
        overlay2.apply(Config())


# ── Phase 6 [learn] section ─────────────────────────────────────────────────

def test_learn_defaults_disabled():
    cfg = Config()
    assert cfg.learn.enabled is False
    assert cfg.learn.cadence_s == 3600
    assert cfg.learn.batch_size == 1
    assert cfg.learn.concurrency == 1
    assert cfg.learn.foreground_reserve == 0
    assert cfg.learn.profiles == {}


def test_learn_loads_from_toml():
    cfg = Config.loads(
        "[learn]\n"
        "enabled = true\n"
        "cadence_s = 7200\n"
        "batch_size = 2\n"
        "concurrency = 4\n"
        "foreground_reserve = 1\n"
        "[learn.profiles.personality]\n"
        "cadence_s = 3600\n"
        "policy = \"all\"\n"
    )
    assert cfg.learn.enabled is True
    assert cfg.learn.cadence_s == 7200
    assert cfg.learn.concurrency == 4
    assert cfg.learn.foreground_reserve == 1
    profile = cfg.learn.profiles["personality"]
    assert profile.cadence_s == 3600
    assert profile.policy == "all"
    assert profile.batch_size is None  # inherits the top-level default


def test_learn_validation():
    with pytest.raises(ConfigError, match="learn.cadence_s"):
        Config.loads("[learn]\ncadence_s = 0")
    with pytest.raises(ConfigError, match="learn.batch_size"):
        Config.loads("[learn]\nbatch_size = -1")
    with pytest.raises(ConfigError, match="learn.concurrency"):
        Config.loads("[learn]\nconcurrency = 0")
    with pytest.raises(ConfigError, match="learn.foreground_reserve"):
        Config.loads("[learn]\nforeground_reserve = -1")
    # The foreground reserve must be strictly below concurrency.
    with pytest.raises(ConfigError, match="strictly below"):
        Config.loads("[learn]\nconcurrency = 2\nforeground_reserve = 2")
    with pytest.raises(ConfigError, match="strictly below"):
        Config.loads("[learn]\nconcurrency = 1\nforeground_reserve = 1")
    # Profile validation.
    with pytest.raises(ConfigError, match="learn.profiles.cadence_s"):
        Config.loads("[learn.profiles.p]\ncadence_s = 0")
    with pytest.raises(ConfigError, match="learn.profiles.policy"):
        Config.loads("[learn.profiles.p]\npolicy = \"bogus\"")
    with pytest.raises(ConfigError, match="expected a boolean"):
        Config.loads("[learn.profiles.p]\nenabled = \"yes\"")
    # Unknown keys are rejected like every other section.
    with pytest.raises(ConfigError, match="unknown config key"):
        Config.loads("[learn]\nbogus = 1")


def test_learn_per_chat_partial_override():
    cfg = Config.loads(
        "[learn]\n"
        "enabled = true\n"
        "concurrency = 4\n"
        "[[chats]]\n"
        "key = \"qq:group:1\"\n"
        "[chats.learn]\n"
        "enabled = false\n"
    )
    # The override only replaces the explicitly-set field; the rest inherit.
    chat = cfg.for_chat("qq:group:1")
    assert chat.learn.enabled is False
    assert chat.learn.concurrency == 4
    # A chat without an override inherits the top-level section.
    other = cfg.for_chat("qq:group:2")
    assert other.learn.enabled is True
    assert other.learn.concurrency == 4


def test_learn_overlay():
    overlay = RuntimeOverlay()
    overlay.set("learn.enabled", True)
    merged = overlay.apply(Config())
    assert merged.learn.enabled is True
    with pytest.raises(ConfigError, match="learn.concurrency"):
        overlay2 = RuntimeOverlay()
        overlay2.set("learn.concurrency", 0)
        overlay2.apply(Config())


def test_learn_round_trip():
    cfg = Config.loads("[learn]\nenabled = true\nconcurrency = 3")
    data = cfg.to_dict()
    assert data["learn"]["enabled"] is True
    assert data["learn"]["concurrency"] == 3
    back = Config.from_dict(data)
    assert back.learn == cfg.learn


# ── Phase 6 P6.5 media catalog config ───────────────────────────────────────

def test_media_config_defaults_are_disabled_and_strict():
    cfg = Config()
    media = cfg.media
    # Disabled by default: an empty TOML boots with no media wiring.
    assert media.enabled is False
    assert media.harvest is False
    # Group nonself stickers only; private chats and images disabled.
    assert media.group_nonself_stickers_only is True
    assert media.private_stickers_enabled is False
    assert media.private_images_enabled is False
    # Strict caps.
    assert media.candidate_cap == 16
    assert media.capacity == 32
    assert media.cooldown_s == 3600.0
    assert media.vision_profile is None


def test_media_config_bounds():
    with pytest.raises(ConfigError, match="candidate_cap"):
        Config.loads("[media]\ncandidate_cap = 0")
    with pytest.raises(ConfigError, match="candidate_cap"):
        Config.loads("[media]\ncandidate_cap = 17")
    with pytest.raises(ConfigError, match="candidate_cap"):
        Config.loads("[media]\ncandidate_cap = true")
    with pytest.raises(ConfigError, match="capacity"):
        Config.loads("[media]\ncapacity = 0")
    with pytest.raises(ConfigError, match="cooldown_s"):
        Config.loads("[media]\ncooldown_s = -1")
    with pytest.raises(ConfigError, match="vision_profile"):
        Config.loads("[media]\nvision_profile = \"\"")
    with pytest.raises(ConfigError, match="expected a boolean"):
        Config.loads("[media]\nenabled = \"yes\"")
    with pytest.raises(ConfigError, match="unknown config key"):
        Config.loads("[media]\nbogus = 1")


def test_media_config_loads_from_toml():
    cfg = Config.loads(
        "[media]\n"
        "enabled = true\n"
        "harvest = true\n"
        "candidate_cap = 8\n"
        "capacity = 16\n"
        "cooldown_s = 120\n"
    )
    assert cfg.media.enabled is True
    assert cfg.media.harvest is True
    assert cfg.media.candidate_cap == 8
    assert cfg.media.capacity == 16
    assert cfg.media.cooldown_s == 120.0


def test_media_vision_profile_must_name_an_existing_llm_profile():
    # An explicit vision profile must exist under [llm.profiles].
    with pytest.raises(ConfigError, match="vision_profile"):
        Config.loads('[media]\nvision_profile = "nope"')
    cfg = Config.loads(
        '[llm.profiles.vision]\nmodel = "qwen-vl-max"\n'
        '[media]\nvision_profile = "vision"'
    )
    assert cfg.media.vision_profile == "vision"


def test_media_per_chat_partial_override():
    cfg = Config.loads(
        "[media]\n"
        "enabled = true\n"
        "capacity = 64\n"
        "[[chats]]\n"
        "key = \"qq:group:1\"\n"
        "[chats.media]\n"
        "enabled = false\n"
    )
    # The override only replaces the explicitly-set field; the rest inherit.
    chat = cfg.for_chat("qq:group:1")
    assert chat.media.enabled is False
    assert chat.media.capacity == 64
    # A chat without an override inherits the top-level section.
    other = cfg.for_chat("qq:group:2")
    assert other.media.enabled is True
    assert other.media.capacity == 64


# ── Phase 6 P6.6 plugins config: path resolution / redaction ────────────────

def test_plugins_defaults():
    cfg = Config()
    assert cfg.plugins.paths == ()
    assert cfg.plugins.entry_points == ()
    assert cfg.plugins.allow_replace == ()
    assert cfg.plugins.hook_timeout_s == 5.0


def test_plugins_relative_paths_resolve_against_config_root(tmp_path):
    plugin = tmp_path / "p.py"
    plugin.write_text("name = 'p'\n", encoding="utf-8")
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text('[plugins]\npaths = ["p.py"]\n', encoding="utf-8")
    cfg = Config.load(cfg_file)
    assert cfg.plugins.paths == (str(plugin.resolve()),)


def test_plugins_paths_survive_overlay_round_trip(tmp_path):
    """A to_dict/from_dict round-trip (RuntimeOverlay.apply) re-resolves
    against the RECORDED config root, never the caller's CWD."""
    plugin = tmp_path / "p.py"
    plugin.write_text("name = 'p'\n", encoding="utf-8")
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text('[plugins]\npaths = ["p.py"]\n', encoding="utf-8")
    cfg = Config.load(cfg_file)
    overlay = RuntimeOverlay()
    overlay.set("gate.threshold", 12)
    applied = overlay.apply(cfg)
    assert applied.plugins.paths == (str(plugin.resolve()),)
    assert applied.gate.threshold == 12


def test_plugins_out_of_root_path_rejected(tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("name = 'x'\n", encoding="utf-8")
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text(
        f'[plugins]\npaths = ["{outside}"]\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="outside the config root"):
        Config.load(cfg_file)


def test_plugins_duplicate_path_rejected(tmp_path):
    plugin = tmp_path / "p.py"
    plugin.write_text("name = 'p'\n", encoding="utf-8")
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text(
        '[plugins]\npaths = ["p.py", "p.py"]\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="duplicate plugins.paths"):
        Config.load(cfg_file)


def test_plugins_non_py_path_rejected(tmp_path):
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text('[plugins]\npaths = ["plugin.txt"]\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=r"\.py module file"):
        Config.load(cfg_file)


def test_plugins_entry_points_and_allowlist_validation():
    with pytest.raises(ConfigError, match="non-empty strings"):
        Config.loads('[plugins]\nentry_points = [""]')
    with pytest.raises(ConfigError, match="non-empty strings"):
        Config.loads('[plugins]\nallow_replace = ["  "]')
    with pytest.raises(ConfigError, match="hook_timeout_s"):
        Config.loads("[plugins]\nhook_timeout_s = 0")
    cfg = Config.loads(
        '[plugins]\nentry_points = ["a", "b"]\nallow_replace = ["sanitize"]\n'
        "hook_timeout_s = 2.5\n"
    )
    assert cfg.plugins.entry_points == ("a", "b")
    assert cfg.plugins.allow_replace == ("sanitize",)
    assert cfg.plugins.hook_timeout_s == 2.5


def test_redacted_dict_masks_plugin_owned_cfg_json():
    cfg = Config.loads(
        '[[chats]]\nkey = "qq:group:1"\n'
        '[chats.cfg_json]\nsecret = "plugin-owned-value"\n'
    )
    data = cfg.redacted_dict()
    chat = data["chats"][0]
    assert chat["cfg_json"] == {"<redacted>": True}
    # The raw dict still carries the value (only diagnostics are redacted).
    assert cfg.chats[0].cfg_json == {"secret": "plugin-owned-value"}


def test_media_round_trip():
    cfg = Config.loads("[media]\nenabled = true\ncandidate_cap = 8")
    data = cfg.to_dict()
    assert data["media"]["enabled"] is True
    assert data["media"]["candidate_cap"] == 8
    back = Config.from_dict(data)
    assert back.media == cfg.media
