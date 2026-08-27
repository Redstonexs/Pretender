"""Distribution smoke: the built wheel must contain schema.sql, register
the ``pretender`` console script, and boot ``pretender init`` from an
isolated install."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    """Build the wheel once per module (pip wheel; the build backend is
    fetched by pip's build isolation)."""
    dist = tmp_path_factory.mktemp("dist")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps",
             "-w", str(dist)],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        # setuptools leaves a build/ artifact in the project root.
        build_dir = ROOT / "build"
        if build_dir.exists():
            import shutil

            shutil.rmtree(build_dir)
    wheels = list(dist.glob("pretender-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_wheel_contains_schema_sql(wheel):
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
    assert "pretender/schema.sql" in names, "schema.sql missing from wheel"
    assert any(n.startswith("pretender/prompts/") for n in names)


def test_wheel_contains_typo_frequency_asset(wheel):
    """The typo frequency asset must ship inside the wheel so typo behaviour
    does not silently disappear outside the source tree."""
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
    assert "pretender/output/data/char_freq.txt" in names, (
        "typo frequency asset missing from wheel"
    )


def test_wheel_registers_console_script(wheel):
    with zipfile.ZipFile(wheel) as z:
        entry = z.read("pretender-1.0.2.dist-info/entry_points.txt").decode()
    assert "pretender = pretender.cli:main" in entry


def test_isolated_install_runs_pretender_init(wheel, tmp_path):
    """Install the wheel into a fresh venv (dependencies resolved from the
    index; the pretender package itself comes from the wheel) and run
    ``pretender init`` through the console script."""
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    pip = venv / "bin" / "pip"
    subprocess.run(
        [str(pip), "install", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    )
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(f'[storage]\ndb_path = "{tmp_path / "iso.db"}"\n', encoding="utf-8")
    result = subprocess.run(
        [str(venv / "bin" / "pretender"), "init", "--config", str(cfg)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "initialized" in result.stdout
    assert "schema v15" in result.stdout
    assert (tmp_path / "iso.db").exists()


def test_installed_wheel_typo_asset_loads_and_mutates(wheel, tmp_path):
    """From an isolated wheel install, the packaged typo asset must load and
    produce a deterministic same-pinyin mutation."""
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    pip = venv / "bin" / "pip"
    subprocess.run(
        [str(pip), "install", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    )
    code = (
        "import random\n"
        "from pretender.output.typo import load_frequency, typo_text\n"
        "assert load_frequency().get('在', 0) > 0, 'asset did not load'\n"
        "out = typo_text('我在这里', rate=1.0, rng=random.Random(1), max_mutations=1)\n"
        "assert out != '我在这里' and len(out) == len('我在这里'), out\n"
        "print('TYPO_OK')\n"
    )
    result = subprocess.run(
        [str(venv / "bin" / "python"), "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "TYPO_OK" in result.stdout
