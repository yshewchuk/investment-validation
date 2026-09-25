"""``changed_tests`` in tools/oc_check.py must also pick up tests that scan
source files as text and so never name the module they check (e.g. every
``fail("CODE")`` literal in a package). Those tests carry a
``# land: always-run`` marker line; ``changed_tests`` appends every such test
file whenever any engine/checks/tools module changed, even though the marker
test itself names no changed module.
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


def _load_oc_check(root: Path):
    spec = importlib.util.spec_from_file_location(
        "oc_check", str(root / "tools" / "oc_check.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


THIS_ROOT = Path(__file__).resolve().parents[1]
OC_CHECK = _load_oc_check(THIS_ROOT)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.com",
         *args],
        capture_output=True, text=True, check=True,
    )


def _init_repo(tmp_path: Path) -> Path:
    root = tmp_path
    _git(root, "init", "-q")
    (root / "engine" / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "engine" / "pkg" / "mod.py").write_text("VALUE = 1\n")
    (root / "tests" / "test_scan.py").write_text(
        "from __future__ import annotations\n"
        "# land: always-run\n"
        "\n"
        "def test_nothing():\n"
        "    assert True\n"
    )
    (root / "tests" / "test_other.py").write_text(
        "def test_nothing_else():\n"
        "    assert True\n"
    )
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    base_sha = _git(root, "rev-parse", "HEAD").stdout.strip()
    _git(root, "update-ref", "refs/remotes/origin/main", base_sha)
    return root


def test_always_run_marker_picked_when_module_changed(tmp_path):
    root = _init_repo(tmp_path)
    (root / "engine" / "pkg" / "mod.py").write_text("VALUE = 2\n")
    picked = OC_CHECK.changed_tests(root)
    assert "tests/test_scan.py" in picked
    assert "tests/test_other.py" not in picked


def test_always_run_marker_not_picked_without_a_module_change(tmp_path):
    root = _init_repo(tmp_path)
    picked = OC_CHECK.changed_tests(root)
    assert "tests/test_scan.py" not in picked
    assert "tests/test_other.py" not in picked
