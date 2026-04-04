"""TDD reproduce-first loop (isolated temp project)."""
from __future__ import annotations

from pathlib import Path

import pytest

from sandbox.tdd_debug_loop import TDDDebugLoop


@pytest.fixture
def tiny_project(tmp_path: Path) -> Path:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_repro.py").write_text(
        "def test_repro_failure():\n    assert False, 'bug'\n",
        encoding="utf-8",
    )
    return tmp_path


def test_ensure_red_requires_failing_test(tiny_project: Path) -> None:
    loop = TDDDebugLoop(tiny_project)
    red = loop.ensure_red("tests/test_repro.py")
    assert red.ok is True
    assert red.outcome.exit_code == 1


def test_ensure_red_fails_when_test_passes(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    loop = TDDDebugLoop(tmp_path)
    red = loop.ensure_red("tests/test_ok.py")
    assert red.ok is False


def test_green_and_regression_after_fix(tiny_project: Path) -> None:
    loop = TDDDebugLoop(tiny_project)
    assert loop.ensure_red("tests/test_repro.py").ok
    (tiny_project / "tests" / "test_repro.py").write_text(
        "def test_repro_failure():\n    assert True\n", encoding="utf-8"
    )
    assert loop.ensure_green("tests/test_repro.py").ok
    reg = loop.run_full_regression()
    assert reg.ok is True


def test_run_cycle_skip_red(tiny_project: Path) -> None:
    (tiny_project / "tests" / "test_repro.py").write_text(
        "def test_repro_failure():\n    assert True\n", encoding="utf-8"
    )
    loop = TDDDebugLoop(tiny_project)
    res = loop.run_cycle_after_fix("tests/test_repro.py", skip_red_check=True)
    assert res.success is True
