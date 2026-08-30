"""Focused safety tests for the standard-library deployment wizard."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import socket
import subprocess
from types import SimpleNamespace

import pytest

from pretender.config import Config
from scripts import deploy


def provider(key: str = "generic-api-key") -> deploy.Provider:
    return deploy.Provider(
        "https://llm.example.test/v1", "provider-planner", "provider-reply", key
    )


@pytest.fixture(autouse=True)
def english_wizard():
    """Assertions below match message text, so pin the language."""

    previous = deploy._LANGUAGE
    deploy.set_language("en")
    yield
    deploy.set_language(previous)


#: Passed explicitly so wizard tests never probe or bind a real port.
TEST_PORT = 34101


def plan(tmp_path: Path, target: str = "compose", service: str = "foreground", home_path: Path | None = None):
    return deploy.build_plan(
        target,
        provider(),
        deploy.Features(),
        "onebot-token",
        project_root=tmp_path / "project",
        home=home_path or tmp_path / "home",
        native_service=service,
        port=TEST_PORT,
    )


class FakeRunner:
    def __init__(self, *, running_volume: str = "", existing_container: bool = False):
        self.calls: list[tuple[list[str], dict]] = []
        self.running_volume = running_volume
        self.existing_container = existing_container

    def run(self, argv, **kwargs):
        command = list(argv)
        self.calls.append((command, kwargs))
        if command[:3] == ["docker", "ps", "--filter"]:
            return SimpleNamespace(returncode=0, stdout=self.running_volume, stderr="")
        if command[:4] == ["docker", "container", "inspect", "pretender"]:
            return SimpleNamespace(
                returncode=0 if self.existing_container else 1, stdout="", stderr=""
            )
        if command[:5] == ["systemctl", "--user", "is-active", "--quiet", "pretender.service"]:
            return SimpleNamespace(returncode=3, stdout="", stderr="")
        if command[:5] == ["systemctl", "--user", "is-enabled", "--quiet", "pretender.service"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_rendered_config_loads_with_generic_key_and_has_no_literal_secrets(monkeypatch):
    text = deploy.render_config(provider("api-secret-not-for-config"), deploy.Features(), "/state/pretender.db")
    monkeypatch.setenv(deploy.LLM_ENV, "generic-key")
    monkeypatch.setenv(deploy.ONEBOT_ENV, "generated-or-configured-token")

    cfg = Config.loads(text)

    assert "api-secret-not-for-config" not in text
    assert cfg.llm.profile("planner").api_key == "generic-key"
    assert cfg.llm.profile("reply").api_key == "generic-key"
    assert cfg.adapter.onebot.access_token == "generated-or-configured-token"
    assert cfg.storage.db_path == "/state/pretender.db"
    assert "harvest" not in text


@pytest.mark.parametrize("bad", ["", "   ", "bad key", 'bad"key', "bad#key", "bad`key", "bad$key", r"bad\key", "bad\nkey"])
def test_secret_values_are_restricted_to_safe_raw_env_characters(bad):
    with pytest.raises(deploy.DeployError):
        deploy.validate_secret(bad, "API key")

    with pytest.raises(deploy.DeployError):
        deploy.build_plan("native", provider(bad), deploy.Features(), "safe-token", project_root="/tmp/project", home="/tmp/home")


def test_safe_secret_characters_are_accepted():
    assert deploy.validate_secret("sk.live_1-2/3+=:@%", "API key")


def test_url_and_model_validation_including_remote_http_rejection():
    assert deploy.validate_base_url("https://api.example.test/v1")
    assert deploy.validate_model_id("vendor/model-name") == "vendor/model-name"
    with pytest.raises(deploy.DeployError, match="plain http"):
        deploy.validate_base_url("http://llm.example.test/v1")
    assert deploy.validate_base_url("http://127.0.0.1:8080/v1", allow_http_loopback=True)
    with pytest.raises(deploy.DeployError, match="credentials"):
        deploy.validate_base_url("https://user:pass@example.test/v1")
    with pytest.raises(deploy.DeployError, match="query"):
        deploy.validate_base_url("https://example.test/v1?secret=bad")


def test_media_catalog_is_the_only_media_feature():
    text = deploy.render_config(
        provider(), deploy.Features(splitting=False, typo=False, media_enabled=True), "data/pretender.db"
    )
    assert 'pipeline = ["sanitize"]' in text
    assert "max_split = 1" in text
    assert "typo_rate = 0.0" in text
    assert "enabled = true" in text
    assert "harvest" not in text


def test_trusted_root_is_checkout_root_not_cwd(monkeypatch):
    monkeypatch.chdir("/")
    root = deploy.trusted_project_root()
    assert root == Path(deploy.__file__).resolve().parents[1]
    assert (root / "pyproject.toml").exists()


def test_compose_commands_use_trusted_explicit_file_and_directory(tmp_path):
    p = plan(tmp_path, "compose")
    commands = deploy.docker_compose_commands(p)
    for command in commands[2:]:
        assert command[:6] == [
            "docker", "compose", "-f", str(p.project_root / "docker-compose.yml"),
            "--project-directory", str(p.project_root),
        ]
    assert commands[-1][-2:] == ["up", "-d"]


def test_controlled_compose_environment_overrides_inherited_values(monkeypatch, tmp_path):
    monkeypatch.setenv(deploy.LLM_ENV, "inherited-key")
    monkeypatch.setenv(deploy.ONEBOT_ENV, "inherited-token")
    monkeypatch.setenv("PRETENDER_IMAGE", "inherited-image")
    env = deploy.controlled_compose_env(plan(tmp_path))
    assert env[deploy.LLM_ENV] == "generic-api-key"
    assert env[deploy.ONEBOT_ENV] == "onebot-token"
    assert env["PRETENDER_IMAGE"] == deploy.DEFAULT_IMAGE
    assert "DEEPSEEK_API_KEY" not in deploy.render_docker_environment(plan(tmp_path))


def test_compose_setup_uses_controlled_env_for_each_compose_command(tmp_path, monkeypatch):
    monkeypatch.setenv(deploy.LLM_ENV, "inherited")
    p = plan(tmp_path, "compose")
    runner = FakeRunner()
    deploy.setup_plan(p, runner=runner, listener_checker=lambda: None)
    compose_calls = [
        (command, kwargs)
        for command, kwargs in runner.calls
        if command[:2] == ["docker", "compose"]
    ]
    assert compose_calls[0][0][-1] == "pull"
    assert compose_calls[1][0][-2:] == ["config", "--quiet"]
    assert all(call[1]["env"][deploy.LLM_ENV] == "generic-api-key" for call in compose_calls)
    assert all(call[1]["env"][deploy.ONEBOT_ENV] == "onebot-token" for call in compose_calls)
    assert all(call[1]["env"]["PRETENDER_IMAGE"] == deploy.DEFAULT_IMAGE for call in compose_calls)
    init_calls = [call for call in compose_calls if "init" in call[0]]
    assert init_calls
    assert init_calls[0][0][-7:] == ["run", "--rm", "--no-deps", "pretender", "init", "--config", "/config/config.toml"]


def test_docker_run_is_detached_and_uses_argument_arrays(tmp_path):
    commands = deploy.docker_run_commands(plan(tmp_path, "docker"))
    run = commands[-1]
    assert run.index("--detach") < run.index(deploy.DEFAULT_IMAGE)
    assert run[run.index("--label") + 1] == "io.pretender.wizard-managed=true"
    assert "--network" in run and "host" in run
    assert "--restart" in run and "unless-stopped" in run
    init = commands[-2]
    assert "--rm" in init and "--detach" not in init and "--restart" not in init
    assert all(isinstance(item, list) for item in commands)


def test_docker_mounts_are_unambiguous_for_colon_paths(tmp_path):
    p = plan(tmp_path / "project:with-colon", "docker")
    commands = deploy.docker_run_commands(p)
    bind = f"type=bind,src={p.config_path.resolve()},dst=/config/config.toml,readonly"
    named = "type=volume,src=pretender-data,dst=/config/data"

    for command in (commands[-2], commands[-1]):
        assert "--volume" not in command
        assert command.count("--mount") == 2
        assert command.count(bind) == 1
        assert named in command


def test_running_volume_is_rejected_before_any_generated_file_changes(tmp_path):
    p = plan(tmp_path, "compose")
    runner = FakeRunner(running_volume="already-running\n")
    with pytest.raises(deploy.DeployError, match="pretender-data"):
        deploy.setup_plan(p, runner=runner, listener_checker=lambda: None)
    assert not p.config_path.exists()
    assert not p.environment_path.exists()


def test_rejected_second_destination_leaves_first_destination_unchanged(tmp_path):
    p = plan(tmp_path, "compose")
    p.config_path.parent.mkdir(parents=True)
    p.config_path.write_text(f"# {deploy.WIZARD_MARKER}\nold config\n")
    p.environment_path.write_text("operator environment\n")
    with pytest.raises(deploy.DeployError, match="non-wizard"):
        deploy.setup_plan(p, runner=FakeRunner(), force=True, listener_checker=lambda: None)
    assert p.config_path.read_text().endswith("old config\n")
    assert p.environment_path.read_text() == "operator environment\n"


def test_existing_container_is_rejected_before_generated_files(tmp_path):
    p = plan(tmp_path, "docker")
    with pytest.raises(deploy.DeployError, match="already exists"):
        deploy.setup_plan(
            p, runner=FakeRunner(existing_container=True), listener_checker=lambda: None
        )
    assert not p.config_path.exists()
    assert not p.environment_path.exists()


def test_setup_failure_recovers_generated_files(tmp_path):
    p = plan(tmp_path, "docker")

    class PullFailure(FakeRunner):
        def run(self, argv, **kwargs):
            if list(argv)[:2] == ["docker", "pull"]:
                raise RuntimeError("pull failed")
            return super().run(argv, **kwargs)

    with pytest.raises(RuntimeError, match="pull failed"):
        deploy.setup_plan(p, runner=PullFailure(), listener_checker=lambda: None)
    assert not p.config_path.exists()
    assert not p.environment_path.exists()


def test_init_failure_recovers_generated_files(tmp_path):
    p = plan(tmp_path, "docker")

    class InitFailure(FakeRunner):
        def run(self, argv, **kwargs):
            if "init" in list(argv):
                raise RuntimeError("init failed")
            return super().run(argv, **kwargs)

    with pytest.raises(RuntimeError, match="init failed"):
        deploy.setup_plan(p, runner=InitFailure(), listener_checker=lambda: None)
    assert not p.config_path.exists()
    assert not p.environment_path.exists()


def test_docker_run_init_precedes_detached_labeled_start(tmp_path):
    p = plan(tmp_path, "docker")
    runner = FakeRunner()
    deploy.setup_plan(p, runner=runner, listener_checker=lambda: None)
    deploy.start_plan(p, runner=runner, listener_checker=lambda: None)
    init_index = next(i for i, (cmd, _) in enumerate(runner.calls) if "init" in cmd)
    start_index = next(i for i, (cmd, _) in enumerate(runner.calls) if "--detach" in cmd)
    assert init_index < start_index
    init = runner.calls[init_index][0]
    assert "--rm" in init and "--detach" not in init and "--restart" not in init
    start = runner.calls[start_index][0]
    assert start.index("--detach") < start.index(p.image)
    assert "io.pretender.wizard-managed=true" in start


def test_atomic_commit_rolls_back_when_second_replace_fails(monkeypatch, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    deploy.atomic_write(first, f"# {deploy.WIZARD_MARKER}\nold first\n")
    deploy.atomic_write(second, f"# {deploy.WIZARD_MARKER}\nold second\n")
    real_replace = deploy.os.replace
    count = 0

    def fail_second(source, destination):
        nonlocal count
        count += 1
        if count == 2:  # fail the second destination replacement
            raise OSError("simulated commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr(deploy.os, "replace", fail_second)
    with pytest.raises(OSError):
        deploy._commit_artifacts(
            [(first, "# Generated by the Pretender deployment wizard.\nnew first\n", 0o644),
             (second, "# Generated by the Pretender deployment wizard.\nnew second\n", 0o644)],
            force=True,
        )
    assert first.read_text().endswith("old first\n")
    assert second.read_text().endswith("old second\n")


def test_systemd_environment_helper_escapes_and_unit_has_no_secret(tmp_path):
    assert deploy.escape_systemd_value('a"\\$`b') == r'a\"\\\$\`b'
    with pytest.raises(deploy.DeployError, match="newline or NUL"):
        deploy.escape_systemd_value("line\nsecret")
    p = deploy.build_plan("native", provider(), deploy.Features(), "onebot-token", project_root=tmp_path / "project", home=tmp_path / "home", native_service="systemd")
    unit = deploy.render_systemd_unit(p)
    assert str(p.config_path.resolve()) in unit
    assert str(p.environment_path.resolve()) in unit
    assert "onebot-token" not in unit and "generic-api-key" not in unit
    for required in ("RestartSec=10", "StartLimitIntervalSec=60", "StartLimitBurst=3", "EnvironmentFile=", "Restart=on-failure", "TimeoutStopSec=30", "UMask=0077", "NoNewPrivileges=true", "PrivateTmp=true"):
        assert required in unit


@pytest.mark.parametrize("unsafe", ["home with space", "home'quote", 'home"quote', r"home\path", "home$path", "home`path"])
def test_systemd_unit_rejects_unsafe_path_tokens(tmp_path, unsafe):
    p = deploy.build_plan(
        "native", provider(), deploy.Features(), "onebot-token",
        project_root=tmp_path / "project", home=tmp_path / unsafe,
        native_service="systemd",
    )
    with pytest.raises(deploy.DeployError, match="unsupported"):
        deploy.render_systemd_unit(p)


def test_native_vulnerability_guards_and_foreground_setup_without_start(tmp_path, monkeypatch):
    p = plan(tmp_path, "native")
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000)
    runner = FakeRunner()
    deploy.setup_plan(p, runner=runner, listener_checker=lambda: None)
    venv_commands = [command for command, _ in runner.calls if command[:3] == [deploy.sys.executable, "-m", "venv"]]
    assert len(venv_commands) == 1
    assert ".venv-staging-" in venv_commands[0][-1]
    assert p.venv_path.is_dir()
    assert p.config_path.exists() and p.environment_path.stat().st_mode & 0o777 == 0o600
    assert not any("pretender" in command and "run" in command for command, _ in runner.calls)
    deploy.start_plan(p, runner=runner, listener_checker=lambda: None)
    init_index = next(i for i, (command, _) in enumerate(runner.calls) if "init" in command)
    start_index = next(i for i, (command, _) in enumerate(runner.calls) if command[-4:] == ["run", "--live", "--config", str(p.config_path.resolve())])
    assert init_index < start_index
    assert start_index > init_index
    assert p.environment_path.parent.stat().st_mode & 0o777 == 0o700


def test_unowned_existing_venv_is_rejected_before_writes(tmp_path, monkeypatch):
    p = plan(tmp_path, "native")
    p.venv_path.mkdir(parents=True)
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000)
    with pytest.raises(deploy.DeployError, match="wizard-owned"):
        deploy.setup_plan(p, runner=FakeRunner(), listener_checker=lambda: None)
    assert not p.config_path.exists()


def test_new_venv_and_marker_are_removed_when_init_fails(tmp_path, monkeypatch):
    p = plan(tmp_path, "native")
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000)

    class InitFailure(FakeRunner):
        def run(self, argv, **kwargs):
            command = list(argv)
            if command and command[0] == deploy.sys.executable and command[1:3] == ["-m", "venv"]:
                Path(command[-1]).mkdir(parents=True, exist_ok=True)
            if "init" in command:
                raise RuntimeError("init failed")
            return super().run(argv, **kwargs)

    with pytest.raises(RuntimeError, match="init failed"):
        deploy.setup_plan(p, runner=InitFailure(), listener_checker=lambda: None)
    assert not p.venv_path.exists()
    assert not p.venv_marker_path.exists()


def test_owned_existing_venv_survives_failed_setup(tmp_path, monkeypatch):
    p = plan(tmp_path, "native")
    p.venv_path.mkdir(parents=True)
    p.venv_marker_path.write_text(f"# {deploy.WIZARD_MARKER}\n")
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000)

    class InstallFailure(FakeRunner):
        def run(self, argv, **kwargs):
            if "pip" in list(argv):
                raise RuntimeError("install failed")
            return super().run(argv, **kwargs)

    with pytest.raises(RuntimeError, match="install failed"):
        deploy.setup_plan(p, runner=InstallFailure(), listener_checker=lambda: None)
    assert p.venv_path.is_dir()
    assert p.venv_marker_path.exists()


def test_docker_parent_symlink_is_rejected_before_writing(tmp_path):
    p = plan(tmp_path, "docker")
    p.project_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (p.project_root / "config").symlink_to(outside, target_is_directory=True)
    with pytest.raises(deploy.DeployError, match="symlinked parent"):
        deploy.setup_plan(p, runner=FakeRunner(), listener_checker=lambda: None)
    assert not (outside / "config.toml").exists()


def test_native_parent_symlink_is_rejected_before_writing(tmp_path):
    p = plan(tmp_path, "native")
    outside = tmp_path / "outside"
    outside.mkdir()
    (p.home / ".config").mkdir(parents=True)
    (p.home / ".config" / "pretender").symlink_to(outside, target_is_directory=True)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000)
    try:
        with pytest.raises(deploy.DeployError, match="symlinked parent"):
            deploy.setup_plan(p, runner=FakeRunner(), listener_checker=lambda: None)
    finally:
        monkeypatch.undo()
    assert not (outside / "config.toml").exists()


def test_native_state_parent_symlink_is_rejected_before_writing(tmp_path, monkeypatch):
    p = plan(tmp_path, "native")
    outside = tmp_path / "outside-state"
    outside.mkdir()
    (p.home / ".local").mkdir(parents=True)
    (p.home / ".local" / "state").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000)
    with pytest.raises(deploy.DeployError, match="symlinked parent"):
        deploy.setup_plan(p, runner=FakeRunner(), listener_checker=lambda: None)
    assert not (outside / "pretender").exists()


def test_active_systemd_service_is_rejected_before_writes(tmp_path, monkeypatch):
    p = plan(tmp_path, "native", "systemd")
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000)

    class Active(FakeRunner):
        def run(self, argv, **kwargs):
            result = super().run(argv, **kwargs)
            if list(argv)[:5] == ["systemctl", "--user", "is-active", "--quiet", "pretender.service"]:
                result.returncode = 0
            return result

    with pytest.raises(deploy.DeployError, match="already active"):
        deploy.setup_plan(p, runner=Active(), listener_checker=lambda: None)
    assert not p.config_path.exists()


def test_dry_run_and_second_start_gate(tmp_path):
    # Express flow: target, provider, model, then the setup and start gates.
    # The OneBot token is generated without a prompt.
    answers = iter(["1", "1", "", "y", "n"])
    hidden = iter(["secret-api"])
    output: list[str] = []
    runner = FakeRunner()
    result = deploy.run_wizard(
        input_fn=lambda _: next(answers),
        secret_fn=lambda _: next(hidden),
        output=output.append,
        runner=runner,
        listener_checker=lambda: None,
        project_root=tmp_path,
        home=tmp_path / "home",
        port=TEST_PORT,
    )
    assert result is not None
    assert not any(command[-2:] == ["up", "-d"] for command, _ in runner.calls)
    assert (tmp_path / "config" / "config.toml").exists()
    assert "secret-api" not in "\n".join(output)


def test_dry_run_has_no_writes_or_runner_calls(tmp_path):
    answers = iter(["1", "1", "", "n"])
    hidden = iter(["secret-api"])
    output: list[str] = []
    runner = FakeRunner()
    result = deploy.run_wizard(
        dry_run=True,
        input_fn=lambda _: next(answers),
        secret_fn=lambda _: next(hidden),
        output=output.append,
        runner=runner,
        project_root=tmp_path,
        home=tmp_path / "home",
        port=TEST_PORT,
    )
    assert result is None
    assert runner.calls == []
    assert not (tmp_path / "config").exists()
    assert not (tmp_path / ".env").exists()
    assert "secret-api" not in "\n".join(output)


def test_native_foreground_is_default_and_generated_token_is_not_printed(monkeypatch, tmp_path):
    answers = iter(["3", "", "1", ""])
    hidden = iter(["safe-api"])
    output: list[str] = []
    monkeypatch.setattr(deploy.secrets, "token_hex", lambda count: "a" * (count * 2))

    result = deploy.gather_plan(
        input_fn=lambda _: next(answers),
        secret_fn=lambda _: next(hidden),
        output=output.append,
        project_root=tmp_path / "project",
        home=tmp_path / "home",
        port=TEST_PORT,
    )
    assert result.native_service == "foreground"
    assert result.onebot_token == "a" * 64
    assert "a" * 64 not in "\n".join(output)


def test_native_init_and_foreground_use_private_umask_and_database_modes(monkeypatch, tmp_path):
    p = plan(tmp_path, "native")
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000)
    umasks: list[int] = []
    real_umask = deploy.os.umask

    def record_umask(mask):
        umasks.append(mask)
        return real_umask(mask)

    monkeypatch.setattr(deploy.os, "umask", record_umask)

    class DatabaseInit(FakeRunner):
        def run(self, argv, **kwargs):
            command = list(argv)
            if "init" in command:
                for suffix in ("", "-wal", "-shm"):
                    path = Path(f"{p.storage_path}{suffix}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("database")
                    path.chmod(0o644)
            return super().run(argv, **kwargs)

    runner = DatabaseInit()
    deploy.setup_plan(p, runner=runner, listener_checker=lambda: None)
    deploy.start_plan(p, runner=runner, listener_checker=lambda: None)
    assert p.environment_path.parent.stat().st_mode & 0o777 == 0o700
    for suffix in ("", "-wal", "-shm"):
        assert Path(f"{p.storage_path}{suffix}").stat().st_mode & 0o777 == 0o600
    assert umasks.count(0o077) >= 2


def test_docker_lock_is_held_through_declined_start_gate(tmp_path):
    answers = iter(["1", "1", "", "y", "n"])
    hidden = iter(["secret-api"])
    runner = FakeRunner()
    events: list[str] = []

    def ask(prompt):
        if "Start the live process" in prompt:
            events.append("gate")
            assert any(command[:2] == ["docker", "create"] for command, _ in runner.calls)
            assert not any(command[:2] == ["docker", "rm"] for command, _ in runner.calls)
        return next(answers)

    deploy.run_wizard(
        input_fn=ask,
        secret_fn=lambda _: next(hidden),
        output=lambda _: None,
        runner=runner,
        listener_checker=lambda: None,
        project_root=tmp_path,
        home=tmp_path / "home",
        port=TEST_PORT,
    )
    create = next(i for i, (command, _) in enumerate(runner.calls) if command[:2] == ["docker", "create"])
    remove = next(i for i, (command, _) in enumerate(runner.calls) if command[:2] == ["docker", "rm"])
    assert create < remove
    assert runner.calls[create][0][runner.calls[create][0].index("--label") + 1] == deploy.WIZARD_LOCK_LABEL
    assert events == ["gate"]


def test_docker_lock_creation_failure_writes_nothing(tmp_path):
    p = plan(tmp_path, "docker")

    class LockFailure(FakeRunner):
        def run(self, argv, **kwargs):
            if list(argv)[:2] == ["docker", "create"]:
                raise RuntimeError("lock exists")
            return super().run(argv, **kwargs)

    with pytest.raises(RuntimeError, match="lock exists"):
        deploy.setup_plan(p, runner=LockFailure(), listener_checker=lambda: None)
    assert not p.config_path.exists()
    assert not p.environment_path.exists()


def test_native_session_lock_spans_second_gate_and_releases(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000)
    answers = iter(["3", "", "1", "", "y", "n"])
    hidden = iter(["secret-api"])
    events: list[str] = []

    class Lock:
        held = False

        def acquire(self):
            events.append("acquire")
            self.held = True

        def release(self):
            events.append("release")
            self.held = False

    lock = Lock()
    deploy.run_wizard(
        input_fn=lambda _: next(answers),
        secret_fn=lambda _: next(hidden),
        output=lambda _: None,
        runner=FakeRunner(),
        session_lock=lock,
        listener_checker=lambda: None,
        project_root=tmp_path,
        home=tmp_path / "home",
        port=TEST_PORT,
    )
    assert events == ["acquire", "release"]


def test_all_targets_share_one_user_lock_path(tmp_path):
    home = tmp_path / "home"
    paths = {
        plan(tmp_path / "compose", "compose", home_path=home).shared_lock_path,
        plan(tmp_path / "docker", "docker", home_path=home).shared_lock_path,
        plan(tmp_path / "native", "native", home_path=home).shared_lock_path,
    }
    assert paths == {home / ".local" / "state" / "pretender" / ".wizard.lock"}


def test_held_shared_lock_blocks_a_cross_target_wizard_before_writes(tmp_path):
    shared_home = tmp_path / "shared-home"
    compose_plan = plan(tmp_path / "first", "compose", home_path=shared_home)
    native_plan = plan(tmp_path / "second", "native", home_path=shared_home)
    first = deploy.UserWizardLock(compose_plan)
    second = deploy.UserWizardLock(native_plan)
    first.acquire()
    try:
        with pytest.raises(deploy.DeployError, match="another native deployment wizard"):
            second.acquire()
        assert not native_plan.config_path.exists()
        assert not native_plan.environment_path.exists()
    finally:
        second.release()
        first.release()


def test_force_environment_update_uses_private_backup_not_checkout_backup(tmp_path):
    p = plan(tmp_path, "docker")
    p.config_path.parent.mkdir(parents=True)
    p.config_path.write_text(f"# {deploy.WIZARD_MARKER}\nold config\n")
    old_environment = f"# {deploy.WIZARD_MARKER}\n{deploy.LLM_ENV}=old-key\n{deploy.ONEBOT_ENV}=old-token\n"
    p.environment_path.write_text(old_environment)
    deploy.setup_plan(p, runner=FakeRunner(), force=True, listener_checker=lambda: None)
    assert "generic-api-key" in p.environment_path.read_text()
    assert not list(p.project_root.glob(".env.bak*"))
    assert list((p.home / ".local" / "state" / "pretender" / "deploy-backups").glob("*"))


def test_force_environment_failure_restores_old_value(tmp_path):
    p = plan(tmp_path, "docker")
    p.config_path.parent.mkdir(parents=True)
    p.config_path.write_text(f"# {deploy.WIZARD_MARKER}\nold config\n")
    old_environment = f"# {deploy.WIZARD_MARKER}\n{deploy.LLM_ENV}=old-key\n{deploy.ONEBOT_ENV}=old-token\n"
    p.environment_path.write_text(old_environment)

    class InitFailure(FakeRunner):
        def run(self, argv, **kwargs):
            if "init" in list(argv):
                raise RuntimeError("init failed")
            return super().run(argv, **kwargs)

    with pytest.raises(RuntimeError, match="init failed"):
        deploy.setup_plan(p, runner=InitFailure(), force=True, listener_checker=lambda: None)
    assert p.environment_path.read_text() == old_environment


def test_failed_secret_restore_preserves_private_backup(tmp_path, monkeypatch):
    p = plan(tmp_path, "docker")
    p.environment_path.parent.mkdir(parents=True)
    p.environment_path.write_text(f"# {deploy.WIZARD_MARKER}\nold-secret\n")
    receipt = deploy._commit_artifacts(
        [(p.environment_path, "# Generated by the Pretender deployment wizard.\nnew-secret\n", 0o600)],
        force=True,
        root=p.project_root,
        secret_paths={p.environment_path},
        secret_backup_root=p.home,
    )
    real_replace = deploy.os.replace

    def fail_restore(source, destination):
        if destination == p.environment_path and Path(source).parent.name == "deploy-backups":
            raise OSError("restore failed")
        return real_replace(source, destination)

    monkeypatch.setattr(deploy.os, "replace", fail_restore)
    with pytest.raises(deploy.DeployError, match="backups were preserved"):
        receipt.rollback()
    assert list((p.home / ".local" / "state" / "pretender" / "deploy-backups").glob("*"))


def test_enabled_inactive_systemd_service_is_rejected(tmp_path, monkeypatch):
    p = plan(tmp_path, "native", "systemd")
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000)

    class Enabled(FakeRunner):
        def run(self, argv, **kwargs):
            result = super().run(argv, **kwargs)
            if list(argv)[:5] == ["systemctl", "--user", "is-enabled", "--quiet", "pretender.service"]:
                result.returncode = 0
            return result

    with pytest.raises(deploy.DeployError, match="already enabled"):
        deploy.setup_plan(p, runner=Enabled(), listener_checker=lambda: None)
    assert not p.config_path.exists()


def test_systemd_start_failure_disables_service(tmp_path, monkeypatch):
    p = plan(tmp_path, "native", "systemd")
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000)

    class StartFailure(FakeRunner):
        def run(self, argv, **kwargs):
            command = list(argv)
            if command[:4] == ["systemctl", "--user", "start", "pretender.service"]:
                raise RuntimeError("start failed")
            return super().run(argv, **kwargs)

    runner = StartFailure()
    with pytest.raises(RuntimeError, match="start failed"):
        deploy.start_plan(p, runner=runner, listener_checker=lambda: None)
    assert any(command[:4] == ["systemctl", "--user", "disable", "pretender.service"] for command, _ in runner.calls)


def test_systemd_success_requires_post_start_liveness(tmp_path, monkeypatch):
    p = plan(tmp_path, "native", "systemd")
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000)

    class BecomesActive(FakeRunner):
        def __init__(self):
            super().__init__()
            self.started = False

        def run(self, argv, **kwargs):
            command = list(argv)
            result = super().run(argv, **kwargs)
            if command[:4] == ["systemctl", "--user", "start", "pretender.service"]:
                self.started = True
            if command[:5] == ["systemctl", "--user", "is-active", "--quiet", "pretender.service"]:
                result.returncode = 0 if self.started else 3
            return result

    runner = BecomesActive()
    deploy.start_plan(p, runner=runner, listener_checker=lambda: None)
    assert not any(command[:4] == ["systemctl", "--user", "disable", "pretender.service"] for command, _ in runner.calls)
    assert sum(command[:5] == ["systemctl", "--user", "is-active", "--quiet", "pretender.service"] for command, _ in runner.calls) >= 2


def test_systemd_failed_post_start_liveness_disables_service(tmp_path, monkeypatch):
    p = plan(tmp_path, "native", "systemd")
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(deploy.time, "sleep", lambda _: None)

    class NeverActive(FakeRunner):
        def run(self, argv, **kwargs):
            result = super().run(argv, **kwargs)
            if list(argv)[:5] == ["systemctl", "--user", "is-active", "--quiet", "pretender.service"]:
                result.returncode = 3
            return result

    runner = NeverActive()
    with pytest.raises(deploy.DeployError, match="did not become active"):
        deploy.start_plan(p, runner=runner, listener_checker=lambda: None)
    assert any(command[:4] == ["systemctl", "--user", "disable", "pretender.service"] for command, _ in runner.calls)


def test_systemd_disable_failure_is_reported(tmp_path, monkeypatch):
    p = plan(tmp_path, "native", "systemd")
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000)

    class CleanupFailure(FakeRunner):
        def run(self, argv, **kwargs):
            command = list(argv)
            if command[:4] == ["systemctl", "--user", "start", "pretender.service"]:
                raise RuntimeError("start failed")
            if command[:4] == ["systemctl", "--user", "disable", "pretender.service"]:
                raise RuntimeError("disable failed")
            return super().run(argv, **kwargs)

    with pytest.raises(deploy.DeployError, match="disabling the service also failed"):
        deploy.start_plan(p, runner=CleanupFailure(), listener_checker=lambda: None)


def test_declining_second_gate_never_enables_systemd(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000)
    answers = iter(["3", "s", "1", "", "y", "n"])
    hidden = iter(["secret-api"])
    runner = FakeRunner()
    deploy.run_wizard(
        input_fn=lambda _: next(answers),
        secret_fn=lambda _: next(hidden),
        output=lambda _: None,
        runner=runner,
        listener_checker=lambda: None,
        project_root=tmp_path,
        home=tmp_path / "home",
        port=TEST_PORT,
    )
    assert not any(command[:3] == ["systemctl", "--user", "enable"] for command, _ in runner.calls)


def test_existing_venv_upgrade_uses_staging_and_keeps_old_on_failure(tmp_path, monkeypatch):
    p = plan(tmp_path, "native")
    p.venv_path.mkdir(parents=True)
    old = p.venv_path / "old-install"
    old.write_text("old")
    p.venv_marker_path.write_text(f"# {deploy.WIZARD_MARKER}\n")
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000)

    class InstallFailure(FakeRunner):
        def run(self, argv, **kwargs):
            if "pip" in list(argv):
                raise RuntimeError("upgrade failed")
            return super().run(argv, **kwargs)

    with pytest.raises(RuntimeError, match="upgrade failed"):
        deploy.setup_plan(p, runner=InstallFailure(), force=True, listener_checker=lambda: None)
    assert old.read_text() == "old"
    assert not list(p.venv_path.parent.glob(".venv-staging-*"))


def test_venv_swap_failure_restores_owned_old_venv(tmp_path, monkeypatch):
    p = plan(tmp_path, "native")
    p.venv_path.mkdir(parents=True)
    (p.venv_path / "old-install").write_text("old")
    p.venv_marker_path.write_text(f"# {deploy.WIZARD_MARKER}\n")
    staging = p.venv_path.parent / ".venv-staging-test"
    staging.mkdir(parents=True)
    (staging / "new-install").write_text("new")
    real_replace = deploy.os.replace

    def fail_promote(source, destination):
        if source == staging and destination == p.venv_path:
            raise OSError("swap failed")
        return real_replace(source, destination)

    monkeypatch.setattr(deploy.os, "replace", fail_promote)
    with pytest.raises(OSError, match="swap failed"):
        deploy.promote_venv(p, staging)
    assert (p.venv_path / "old-install").read_text() == "old"
    shutil.rmtree(staging)


# ── port selection ──────────────────────────────────────────────────────────


def bound_port() -> tuple[socket.socket, int]:
    """Hold a loopback port open so the collision paths have something real."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock, sock.getsockname()[1]


def test_selected_port_reaches_the_rendered_config(monkeypatch):
    text = deploy.render_config(provider(), deploy.Features(), "data/pretender.db", 3457)
    monkeypatch.setenv(deploy.LLM_ENV, "generic-key")
    monkeypatch.setenv(deploy.ONEBOT_ENV, "generated-token")

    cfg = Config.loads(text)

    assert cfg.adapter.onebot.port == 3457
    assert cfg.adapter.onebot.path == deploy.DEFAULT_ONEBOT_PATH


@pytest.mark.parametrize("bad", ["", "80", "0", "65536", "http", "3001x", True, None])
def test_port_validation_rejects_privileged_and_malformed_values(bad):
    with pytest.raises(deploy.DeployError):
        deploy.validate_port(bad)


def test_find_free_port_skips_the_busy_one():
    sock, busy = bound_port()
    try:
        assert deploy.port_available(busy) is False
        assert deploy.find_free_port(busy) > busy
    finally:
        sock.close()


def test_check_listener_reports_the_busy_address():
    sock, busy = bound_port()
    try:
        with pytest.raises(deploy.DeployError, match=str(busy)):
            deploy.check_listener(port=busy)
    finally:
        sock.close()
    deploy.check_listener(port=busy)


def test_wizard_offers_a_free_port_and_refuses_a_busy_one():
    sock, busy = bound_port()
    try:
        messages: list[str] = []
        prompts: list[str] = []
        # First answer insists on the busy port; the wizard re-asks instead of
        # writing a config that cannot bind.
        answers = iter([str(busy), ""])

        def ask(prompt):
            prompts.append(prompt)
            return next(answers)

        chosen = deploy._prompt_port(ask, messages.append, preferred=busy)
    finally:
        sock.close()
    assert chosen != busy
    assert deploy.port_available(chosen)
    assert len(prompts) == 2
    assert any(str(busy) in message for message in messages)


def test_setup_uses_the_plan_port_for_the_listener_check(tmp_path, monkeypatch):
    checked: list[int] = []
    monkeypatch.setattr(deploy, "check_listener", lambda port=deploy.DEFAULT_PORT: checked.append(port))
    p = replace(plan(tmp_path, "compose"), port=3456)

    deploy.setup_plan(p, runner=FakeRunner())

    assert checked == [3456]


def test_plan_exposes_the_napcat_url_without_the_token(tmp_path):
    p = replace(plan(tmp_path, "compose"), port=3456, onebot_token="super-secret-token")

    text = deploy.napcat_instructions(p)

    assert p.onebot_url == "ws://127.0.0.1:3456/onebot/v11/ws?message_format=array"
    assert p.onebot_url in text
    assert "super-secret-token" not in text
    assert deploy.ONEBOT_ENV in text


def test_deployment_default_port_is_3002_and_url_remains_centralized(tmp_path):
    p = deploy.build_plan(
        "compose", provider(), deploy.Features(), "onebot-token",
        project_root=tmp_path / "project", home=tmp_path / "home",
    )

    assert deploy.DEFAULT_PORT == 3002
    assert p.port == 3002
    assert p.onebot_url == "ws://127.0.0.1:3002/onebot/v11/ws?message_format=array"
    assert deploy.DEFAULT_ONEBOT_PATH in deploy.napcat_instructions(p)


@pytest.mark.parametrize(
    "image",
    [
        deploy.DEFAULT_IMAGE,
        "registry.example.test/pretender:v2.4.0",
        "registry.example.test:5443/pretender:stable-1",
        "registry.example.test/pretender@sha256:" + "a" * 64,
    ],
)
def test_image_reference_accepts_explicit_tag_or_digest(image):
    assert deploy.validate_image_reference(image) == image


@pytest.mark.parametrize(
    "image",
    [
        "",
        "registry.example.test/pretender",
        "registry.example.test/pretender:latest",
        "registry.example.test/pretender:LaTeSt",
        "registry.example.test/pretender@sha256:abc",
        " registry.example.test/pretender:v1",
        "registry.example.test/pretender:v1 ",
    ],
)
def test_image_reference_rejects_unpinned_or_malformed_images(image):
    with pytest.raises(deploy.DeployError):
        deploy.validate_image_reference(image)


def test_build_plan_enforces_image_reference_contract(tmp_path):
    with pytest.raises(deploy.DeployError, match="latest"):
        deploy.build_plan(
            "compose", provider(), deploy.Features(), "onebot-token",
            project_root=tmp_path / "project", home=tmp_path / "home",
            image="example.test/pretender:latest",
        )


def test_napcat_instructions_cover_docker_network_and_login_safety(tmp_path):
    p = plan(tmp_path, "compose")
    for language in ("zh", "en"):
        deploy.set_language(language)
        text = deploy.napcat_instructions(p)
        assert "host" in text.lower()
        assert "ports:" in text
        assert "/app/.config/QQ" in text
        assert "/app/napcat/config" in text
        assert "message_format=array" in text
        assert "ACCOUNT" in text
        assert "NAPCAT_QUICK_PASSWORD" not in text
        assert "onebot-token" not in text


def test_shell_management_commands_are_reconcile_and_backup_safe():
    source = (Path(deploy.__file__).resolve().parents[1] / "deploy.sh").read_text()

    assert "./deploy.sh validate" in source
    assert "./deploy.sh backup" in source
    assert "compose restart" not in source
    assert source.count("compose up -d --force-recreate") >= 2
    assert "backup_database" in source
    assert "compose pull" in source
    assert "compose config --quiet" in source
    assert "wait_for_core_liveness" in source
    assert "pretender doctor" not in source
    assert "PRAGMA integrity_check" in source
    assert "docker exec" in source
    assert "docker cp" in source
    assert "mode=ro" in source
    assert "os.O_EXCL" in source
    assert "chown \"$backup_owner" not in source
    assert "stat -c '%u:%g:%a:%F'" in source
    assert "compose run" not in source
    assert '"$backup_dir:/backup:rw"' not in source
    assert "XDG_STATE_HOME" in source
    assert "not a full volume backup" in source
    assert "return 2" in source
    assert 'chmod 700 "$backup_dir"' in source
    assert 'chmod 600 "$backup_temp_path"' in source
    executable_lines = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "down -v" not in executable_lines


def test_shell_validate_reports_transport_only_and_distinct_missing_peer(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "config").mkdir()
    shutil.copy2(Path(deploy.__file__).resolve().parents[1] / "deploy.sh", checkout / "deploy.sh")
    (checkout / "scripts" / "deploy.py").write_text("# test marker\n")
    (checkout / "docker-compose.yml").write_text("services: {}\n")
    (checkout / ".env").write_text("PRETENDER_IMAGE=example.test/pretender:v1\n")
    (checkout / "config" / "config.toml").write_text(
        '[adapter.onebot]\nhost = "127.0.0.1"\nport = 3002\n'
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "docker").write_text(
        """#!/usr/bin/env python3
import os, sys

args = sys.argv[1:]
if args[:2] == ["compose", "version"] or args[:1] == ["info"]:
    raise SystemExit(0)
if args[:1] == ["compose"]:
    if "ps" in args and "-q" in args:
        print("live-container")
    raise SystemExit(0)
if args[:1] == ["ps"]:
    print("live-container")
    raise SystemExit(0)
if args[:1] == ["inspect"]:
    value = args[args.index("--format") + 1]
    if "State.Status" in value:
        print("running:true:false")
    elif "Config.Image" in value:
        print("example.test/pretender:v1")
    elif "NetworkMode" in value:
        print("host")
    elif "config.toml" in value:
        print("bind:false:" + os.environ["FAKE_ROOT"] + "/config/config.toml")
    elif "/config/data" in value:
        print("volume:pretender-data")
    elif "RestartCount" in value:
        print("0")
    raise SystemExit(0)
if args[:1] == ["exec"]:
    script = args[args.index("-c") + 1]
    if "tomllib" in script:
        if os.environ.get("FAKE_TOPOLOGY", "1") != "1":
            raise SystemExit(2)
        print("3002")
    elif "os.listdir" in script and "tcp6" in script:
        print("%s %s" % (os.environ.get("FAKE_LISTENER", "1"), os.environ.get("FAKE_PEER", "0")))
    raise SystemExit(0)
raise SystemExit(0)
"""
    )
    (fake_bin / "docker").chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_ROOT": str(checkout),
        "PRETENDER_LANG": "en",
    }

    with_peer = subprocess.run(
        ["sh", str(checkout / "deploy.sh"), "validate"],
        cwd=checkout, env={**env, "FAKE_PEER": "1"},
        capture_output=True, text=True, check=False,
    )
    assert with_peer.returncode == 0
    assert "transport peer established" in with_peer.stdout
    assert "protocol readiness" in with_peer.stdout

    without_peer = subprocess.run(
        ["sh", str(checkout / "deploy.sh"), "validate"],
        cwd=checkout, env={**env, "FAKE_PEER": "0"},
        capture_output=True, text=True, check=False,
    )
    assert without_peer.returncode == 2
    assert "not protocol-ready" in without_peer.stdout

    no_listener = subprocess.run(
        ["sh", str(checkout / "deploy.sh"), "validate"],
        cwd=checkout, env={**env, "FAKE_LISTENER": "0", "FAKE_PEER": "0"},
        capture_output=True, text=True, check=False,
    )
    assert no_listener.returncode == 1
    assert "could not prove" in no_listener.stdout

    topology_mismatch = subprocess.run(
        ["sh", str(checkout / "deploy.sh"), "validate"],
        cwd=checkout, env={**env, "FAKE_TOPOLOGY": "0"},
        capture_output=True, text=True, check=False,
    )
    assert topology_mismatch.returncode == 1
    assert "canonical path" in topology_mismatch.stdout


def test_service_socket_probe_parses_tcp_and_tcp6_and_requires_fd_ownership(tmp_path):
    source = (Path(deploy.__file__).resolve().parents[1] / "deploy.sh").read_text()
    start = source.index("validation_socket_probe='") + len("validation_socket_probe='")
    end = source.index("'\n        validation_socket_probe_result=", start)
    probe = source[start:end]
    port = 3003
    port_hex = f"{port:04X}"

    def tcp_row(address, state, inode):
        return (
            f"0: {address}:{port_hex} 00000000:0000 {state} "
            "00000000:00000000 00:00000000 00000000 1000 0 "
            f"{inode} 1"
        )

    def make_proc(root, fd_inodes, tcp_rows, tcp6_rows):
        (root / "1" / "fd").mkdir(parents=True)
        (root / "net").mkdir()
        for descriptor, inode in enumerate(fd_inodes, 3):
            (root / "1" / "fd" / str(descriptor)).symlink_to(f"socket:[{inode}]")
        (root / "net" / "tcp").write_text("header\n" + "\n".join(tcp_rows) + "\n")
        (root / "net" / "tcp6").write_text("header\n" + "\n".join(tcp6_rows) + "\n")

    owned = tmp_path / "proc-owned"
    make_proc(
        owned,
        [111, 222],
        [tcp_row("0100007F", "0A", 111)],
        [tcp_row("00000000000000000000000001000000", "01", 222)],
    )
    result = subprocess.run(
        ["python3", "-c", probe, str(port), str(owned)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "1 1"

    mismatched = tmp_path / "proc-mismatched"
    make_proc(
        mismatched,
        [111],
        [tcp_row("0100007F", "0A", 999)],
        [tcp_row("00000000000000000000000001000000", "01", 999)],
    )
    result = subprocess.run(
        ["python3", "-c", probe, str(port), str(mismatched)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "0 0"


# ── language ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "environ,expected",
    [
        ({"LANG": "zh_CN.UTF-8"}, "zh"),
        ({"LANG": "en_US.UTF-8"}, "en"),
        ({"LC_ALL": "zh_TW.UTF-8", "LANG": "en_US.UTF-8"}, "zh"),
        ({"LANG": "C"}, "zh"),
        ({}, "zh"),
        ({"PRETENDER_LANG": "en", "LANG": "zh_CN.UTF-8"}, "en"),
    ],
)
def test_language_detection(environ, expected):
    assert deploy.detect_language(environ) == expected


def test_every_message_key_exists_in_both_languages():
    assert set(deploy.MESSAGES["zh"]) == set(deploy.MESSAGES["en"])


def test_unknown_message_key_falls_back_to_itself():
    deploy.set_language("zh")
    assert deploy.t("no.such.key") == "no.such.key"
    assert "Docker Compose" in deploy.t("prompt.target")


def test_unsupported_language_is_rejected():
    with pytest.raises(deploy.DeployError):
        deploy.set_language("fr")


# ── non-interactive plans ───────────────────────────────────────────────────


def test_non_interactive_plan_reads_secrets_from_the_environment(tmp_path):
    p = deploy.plan_from_options(
        environ={deploy.LLM_ENV: "sk-from-env", deploy.ONEBOT_ENV: "token-from-env"},
        project_root=tmp_path / "project",
        home=tmp_path / "home",
        port=TEST_PORT,
    )

    assert p.provider.api_key == "sk-from-env"
    assert p.onebot_token == "token-from-env"
    assert p.provider.planner_model == "deepseek-chat"
    assert p.provider.reply_model == "deepseek-chat"
    assert p.port == TEST_PORT
    assert deploy.plan_summary(p).count(deploy.t("summary.redacted")) == 2


def test_non_interactive_plan_generates_a_token_when_unset(tmp_path):
    p = deploy.plan_from_options(
        environ={deploy.LLM_ENV: "sk-from-env"},
        project_root=tmp_path / "project",
        home=tmp_path / "home",
        port=TEST_PORT,
    )

    assert len(p.onebot_token) == 64


def test_non_interactive_plan_requires_the_api_key(tmp_path):
    with pytest.raises(deploy.DeployError, match=deploy.LLM_ENV):
        deploy.plan_from_options(
            environ={},
            project_root=tmp_path / "project",
            home=tmp_path / "home",
            port=TEST_PORT,
        )


def test_non_interactive_custom_provider_requires_a_base_url(tmp_path):
    with pytest.raises(deploy.DeployError, match="base-url"):
        deploy.plan_from_options(
            provider_kind="custom",
            environ={deploy.LLM_ENV: "sk-from-env"},
            project_root=tmp_path / "project",
            home=tmp_path / "home",
            port=TEST_PORT,
        )


def test_prebuilt_plan_skips_every_question(tmp_path):
    p = deploy.plan_from_options(
        environ={deploy.LLM_ENV: "sk-from-env"},
        project_root=tmp_path,
        home=tmp_path / "home",
        port=TEST_PORT,
    )

    def refuse(prompt):
        raise AssertionError(f"unexpected prompt: {prompt}")

    result = deploy.run_wizard(
        dry_run=True, plan=p, assume_yes=True, input_fn=refuse,
        secret_fn=refuse, output=lambda _: None, runner=FakeRunner(),
    )

    assert result is p


# ── older interpreters ──────────────────────────────────────────────────────


def test_project_name_is_readable_without_tomllib(monkeypatch):
    monkeypatch.setattr(deploy, "tomllib", None)
    manifest = Path(deploy.__file__).resolve().parents[1] / "pyproject.toml"

    assert deploy._manifest_project_name(manifest) == "pretender"
    assert deploy.trusted_project_root() == manifest.parent


def test_generated_env_and_config_agree_on_the_port_and_secrets(tmp_path, monkeypatch):
    """The two generated files have to work together: the config references
    env names, the .env supplies them, and both name the chosen port."""

    answers = iter(["1", "1", "", "y", "n"])
    hidden = iter(["sk-live-key"])
    monkeypatch.setattr(deploy.secrets, "token_hex", lambda count: "b" * (count * 2))

    p = deploy.run_wizard(
        input_fn=lambda _: next(answers),
        secret_fn=lambda _: next(hidden),
        output=lambda _: None,
        runner=FakeRunner(),
        listener_checker=lambda: None,
        project_root=tmp_path,
        home=tmp_path / "home",
        port=3457,
    )

    assert p is not None
    environment = dict(
        line.split("=", 1)
        for line in p.environment_path.read_text().splitlines()
        if "=" in line and not line.startswith("#")
    )
    assert environment[deploy.LLM_ENV] == "sk-live-key"
    assert environment[deploy.ONEBOT_ENV] == "b" * 64
    assert p.environment_path.stat().st_mode & 0o777 == 0o600
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    cfg = Config.loads(p.config_path.read_text())

    assert cfg.adapter.onebot.port == 3457
    assert cfg.adapter.onebot.access_token == "b" * 64
    assert cfg.llm.profile("planner").api_key == "sk-live-key"
    assert "sk-live-key" not in p.config_path.read_text()


# ── the generated config must expose the reply gate ─────────────────────────
#
# The wizard used to emit only llm/output/media/adapter/storage. With no
# ``[gate]`` section there is nothing in the file for an operator to turn, and
# at the dataclass default (trigger_score 80) an unaddressed group message
# tops out around 66 — so the bot was structurally @-only with no visible
# switch. "I am not finding any switch to let it talk" is that gap.

def test_render_config_exposes_the_gate(tmp_path, monkeypatch):
    text = deploy.render_config(provider(), deploy.Features(), "data/pretender.db")
    assert "[gate]" in text
    assert "[bot]" in text
    assert "[log]" in text

    monkeypatch.setenv(deploy.LLM_ENV, "secret")
    monkeypatch.setenv(deploy.ONEBOT_ENV, "token")
    cfg = Config.loads(text)
    assert cfg.gate.threshold == deploy.DEFAULT_GATE_THRESHOLD
    assert cfg.gate.trigger_score == deploy.DEFAULT_GATE_TRIGGER_SCORE
    assert cfg.bot.name == deploy.DEFAULT_BOT_NAME
    assert cfg.log.dir == "logs"


def test_generated_trigger_score_lets_the_bot_join_a_conversation():
    """Below the dataclass default of 80, which no unaddressed group message
    can reach."""
    assert deploy.DEFAULT_GATE_TRIGGER_SCORE < 80


def test_render_config_documents_how_to_retune_the_gate():
    """The switch is only a switch if the file says what it does."""
    text = deploy.render_config(provider(), deploy.Features(), "data/pretender.db")
    gate_comments = [
        line for line in text.splitlines()
        if line.startswith("#") and ("trigger_score" in line or "threshold" in line)
    ]
    assert gate_comments


# ── the adaptive stack reaches the generated config ─────────────────────────


def _zen(**kw):
    from scripts.deploy import Provider

    return Provider(
        "https://opencode.ai/zen/v1",
        "ling-3.0-flash-fin-free",
        "ling-3.0-flash-fin-free",
        "sk-x",
        **kw,
    )


def test_generated_config_enables_the_learners():
    """Without [learn] the reply style is frozen at 自然 forever, which is
    what the first deployment shipped."""
    from scripts.deploy import Features, render_config

    text = render_config(_zen(), Features(), "data/p.db")
    assert "[learn]\nenabled = true" in text
    for learner in (
        "expression", "behavior", "jargon", "summary", "effect", "impression",
    ):
        assert f"[learn.profiles.{learner}]" in text


def test_generated_config_gives_the_learners_a_provider_profile():
    """[learn] without [llm.profiles.learn] is learning switched off: every
    run fails with "no LLM profile named 'learn'" and no record is ever
    written. The first deployment shipped exactly that."""
    from scripts.deploy import Features, render_config

    text = render_config(_zen(), Features(), "data/p.db")
    assert "[llm.profiles.learn]" in text
    # Nothing to call when learning is off, so nothing is written.
    off = render_config(_zen(), Features(learning=False), "data/p.db")
    assert "[llm.profiles.learn]" not in off


def test_generated_config_ships_the_access_lists_permissive():
    """The wizard writes the lists so an operator can find them, with the
    empty-blacklist default that changes nothing until ids are added."""
    from scripts.deploy import Features, render_config

    text = render_config(_zen(), Features(), "data/p.db")
    assert "[access.groups]" in text
    assert "[access.private]" in text
    assert text.count('mode = "blacklist"') == 2
    assert text.count("ids = []") == 2


def test_generated_config_paces_the_bubbles():
    """A constant delay ladder is a tell: real bubbles arrive when the person
    finished typing them, so the wait scales with the bubble."""
    from scripts.deploy import Features, render_config

    text = render_config(_zen(), Features(), "data/p.db")
    assert "typing_speed = 1.0" in text


def test_generated_config_carries_drift():
    """[drift] used to be parsed and read by nothing."""
    from scripts.deploy import Features, render_config

    text = render_config(_zen(), Features(), "data/p.db")
    assert '[drift]\nlevel = "active"' in text


def test_vision_profile_is_written_only_when_the_provider_has_one():
    """"Supports vision" means the model READS the image. opencode.ai/zen's
    hy3-free returns HTTP 200 for an image part and then answers "you did not
    upload an image" — accepting the request shape is not the same thing."""
    from scripts.deploy import Features, Provider, render_config

    openai = Provider(
        "https://api.openai.com/v1", "gpt-4o-mini", "gpt-4o-mini", "sk-x"
    )
    text = render_config(openai, Features(media_enabled=True), "data/p.db")
    assert "[llm.profiles.vision]" in text
    assert 'model = "gpt-4o-mini"' in text
    assert "[media]\nenabled = true" in text

    for base, planner in (
        ("https://api.deepseek.com/v1", "deepseek-chat"),
        ("https://opencode.ai/zen/v1", "ling-3.0-flash-fin-free"),
    ):
        text = render_config(
            Provider(base, planner, planner, "sk-x"),
            Features(media_enabled=True),
            "data/p.db",
        )
        # Present only as a commented-out stub explaining why there is none.
        assert "[llm.profiles.vision]" not in text.replace(
            "# [llm.profiles.vision]", ""
        )
        assert "# No vision profile" in text
        # Media needs vision to describe an incoming image, so it stays off.
        assert "[media]\nenabled = false" in text


def test_missing_embeddings_are_explained_not_silently_omitted():
    from scripts.deploy import Features, render_config

    text = render_config(_zen(), Features(), "data/p.db")
    assert "[llm.profiles.embed]" not in text.replace("# [llm.profiles.embed]", "")
    assert "# No embed profile" in text


def test_typo_rate_matches_maibot():
    from scripts.deploy import Features, render_config

    text = render_config(_zen(), Features(), "data/p.db")
    assert "typo_rate = 0.01" in text


def test_generated_config_loads(tmp_path, monkeypatch):
    from scripts.deploy import Features, render_config

    from pretender.config import load_config

    monkeypatch.setenv("PRETENDER_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("ONEBOT_ACCESS_TOKEN", "tok")
    path = tmp_path / "config.toml"
    path.write_text(
        render_config(_zen(), Features(media_enabled=True), "data/p.db"),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert sorted(cfg.llm.profiles) == ["learn", "planner", "reply"]
    assert cfg.learn.enabled and len(cfg.learn.profiles) == 6
    assert cfg.drift.level == "active"
    # No vision on this provider, so media degrades off rather than harvesting
    # stickers it can never describe.
    assert cfg.media.enabled is False
