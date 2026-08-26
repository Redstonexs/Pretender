"""Prompt store: user-dir overlay, {{var}} rendering, mtime hot reload."""

from __future__ import annotations

import os

import pytest

from pretender.errors import PromptError
from pretender.prompts import PACKAGE_PROMPT_DIR, PromptStore, render_text


def _write(path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ── package defaults ────────────────────────────────────────────────────────

def test_package_default_identity_exists():
    store = PromptStore()
    text = store.load("identity.txt")
    assert "麦麦" in text


# ── load_identity (bot.identity_file) ────────────────────────────────────────

def test_load_identity_default_path_resolves_package():
    """The default ``bot.identity_file = "prompts/identity.txt"`` resolves
    through the prompt infrastructure to the shipped package identity."""
    store = PromptStore()
    text = store.load_identity("prompts/identity.txt")
    assert "麦麦" in text


def test_load_identity_user_dir_shadows_package(tmp_path):
    _write(tmp_path / "identity.txt", "用户自定义身份")
    store = PromptStore(user_dir=tmp_path)
    assert store.load_identity("prompts/identity.txt") == "用户自定义身份"


def test_load_identity_absolute_path(tmp_path):
    _write(tmp_path / "custom_identity.txt", "绝对路径身份")
    store = PromptStore()
    assert store.load_identity(str(tmp_path / "custom_identity.txt")) == "绝对路径身份"


def test_load_identity_missing_raises(tmp_path):
    store = PromptStore()
    with pytest.raises(PromptError, match="cannot read identity file"):
        store.load_identity(str(tmp_path / "nope.txt"))


def test_load_identity_empty_raises(tmp_path):
    _write(tmp_path / "empty.txt", "   \n")
    store = PromptStore()
    with pytest.raises(PromptError, match="empty"):
        store.load_identity(str(tmp_path / "empty.txt"))


def test_missing_prompt_raises():
    store = PromptStore()
    with pytest.raises(PromptError, match="not found"):
        store.load("no_such_prompt.txt")


# ── user dir overlay ────────────────────────────────────────────────────────

def test_user_dir_overrides_package_default(tmp_path):
    _write(tmp_path / "identity.txt", "用户自定义身份")
    store = PromptStore(user_dir=tmp_path)
    assert store.load("identity.txt") == "用户自定义身份"


def test_user_file_only(tmp_path):
    _write(tmp_path / "custom.txt", "hello")
    store = PromptStore(user_dir=tmp_path)
    assert store.load("custom.txt") == "hello"


def test_user_file_appearing_later_shadows_package(tmp_path):
    store = PromptStore(user_dir=tmp_path)
    assert "麦麦" in store.load("identity.txt")  # package default first
    _write(tmp_path / "identity.txt", "后来出现的用户文件")
    assert store.load("identity.txt") == "后来出现的用户文件"


def test_traversal_outside_roots_rejected(tmp_path):
    store = PromptStore(user_dir=tmp_path)
    with pytest.raises(PromptError, match="invalid prompt name"):
        store.load("../outside.txt")
    with pytest.raises(PromptError, match="invalid prompt name"):
        store.load("/etc/passwd")


# ── {{var}} rendering ───────────────────────────────────────────────────────

def test_render_substitutes_variables():
    assert render_text("你好，{{name}}！", name="麦麦") == "你好，麦麦！"


def test_render_ignores_whitespace_inside_braces():
    assert render_text("{{ name }}", name="x") == "x"


def test_render_missing_variable_raises():
    with pytest.raises(PromptError, match="missing prompt variable.*name"):
        render_text("你好，{{name}}！")


def test_render_reports_all_missing_variables():
    with pytest.raises(PromptError, match="a, b"):
        render_text("{{a}} {{b}} {{a}}")


def test_render_extra_variables_are_fine():
    assert render_text("{{a}}", a="1", unused="2") == "1"


def test_store_render_combines_load_and_render(tmp_path):
    _write(tmp_path / "greet.txt", "你好，{{name}}")
    store = PromptStore(user_dir=tmp_path)
    assert store.render("greet.txt", name="麦麦") == "你好，麦麦"
    with pytest.raises(PromptError, match="missing prompt variable"):
        store.render("greet.txt")


def test_render_variable_named_name_works(tmp_path):
    # "name" is a common prompt variable; it must not collide with the
    # loader's own parameter.
    _write(tmp_path / "who.txt", "我是{{name}}")
    store = PromptStore(user_dir=tmp_path)
    assert store.render("who.txt", name="麦麦") == "我是麦麦"


# ── mtime hot reload ────────────────────────────────────────────────────────

def test_reload_when_mtime_changes(tmp_path):
    path = tmp_path / "live.txt"
    _write(path, "version 1")
    store = PromptStore(user_dir=tmp_path)
    assert store.load("live.txt") == "version 1"

    _write(path, "version 2")
    # force a distinct mtime (filesystems may have coarse granularity)
    st = path.stat()
    os.utime(path, (st.st_atime + 2, st.st_mtime + 2))
    assert store.load("live.txt") == "version 2"


def test_unchanged_file_is_cached(tmp_path):
    path = tmp_path / "cached.txt"
    _write(path, "same")
    store = PromptStore(user_dir=tmp_path)
    assert store.load("cached.txt") == "same"
    assert store.load("cached.txt") == "same"  # served from cache


def test_invalidate_forces_reread(tmp_path):
    path = tmp_path / "inv.txt"
    _write(path, "v1")
    store = PromptStore(user_dir=tmp_path)
    assert store.load("inv.txt") == "v1"
    _write(path, "v2")
    store.invalidate("inv.txt")
    assert store.load("inv.txt") == "v2"


def test_package_dir_is_real_directory():
    assert PACKAGE_PROMPT_DIR.is_dir()
    assert (PACKAGE_PROMPT_DIR / "identity.txt").is_file()


# ── Phase 3 prompt assets ────────────────────────────────────────────────────

PHASE3_ASSETS = ("planner.txt", "planner_focus.txt", "replyer.txt")

# The {{var}} set each Phase 3 asset declares (must match the file exactly).
PHASE3_VARS = {
    "planner.txt": ("identity", "chat_log", "reply_style"),
    "planner_focus.txt": ("identity", "chat_log", "reply_style", "focus_chat"),
    "replyer.txt": ("identity", "reply_style", "reply_reference"),
}


def test_phase3_assets_load():
    store = PromptStore()
    for name in PHASE3_ASSETS:
        assert store.load(name).strip()


def test_phase3_assets_render_with_variables():
    store = PromptStore()
    for name, vars_ in PHASE3_VARS.items():
        store.render(name, **{v: "x" for v in vars_})


def test_phase3_missing_variable_fails_closed():
    store = PromptStore()
    for name, vars_ in PHASE3_VARS.items():
        # Omit one required variable at a time; rendering must raise.
        for omitted in vars_:
            supplied = {v: "x" for v in vars_ if v != omitted}
            with pytest.raises(PromptError, match="missing prompt variable"):
                store.render(name, **supplied)