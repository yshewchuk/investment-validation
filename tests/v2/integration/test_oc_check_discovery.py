"""Real Git discovery for the wrapper's root and v2 test locations.

# packages: ops
"""
import subprocess
import tempfile
from pathlib import Path

import pytest

from tools import oc_check


def _write(root, path, text):
    """Write one synthetic tracked or untracked file."""
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


@pytest.fixture
def repo(tmp_path):
    """Create a real diff base; clean disposable repository bytes afterward."""
    with tempfile.TemporaryDirectory(dir=tmp_path) as directory:
        root = Path(directory)

        def seed(files):
            """Commit the synthetic base and its local origin/main reference."""
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            for path, text in files.items():
                _write(root, path, text)
            git = ["git", "-C", str(root)]
            subprocess.run(git + ["add", "."], check=True)
            subprocess.run(git + ["-c", "user.name=synthetic", "-c",
                                  "user.email=synthetic@example.com", "commit", "-qm", "base"], check=True)
            subprocess.run(git + ["update-ref", "refs/remotes/origin/main", "HEAD"], check=True)
            return root

        yield seed


@pytest.mark.parametrize("location", ["tests", "tests/v2/ops", "tests/v2/integration",
                                     "tests/v2/models/training"])
@pytest.mark.parametrize("tracked", [True, False])
def test_changed_root_and_nested_tests_are_selected(repo, location, tracked):
    """Tracked edits and new tests are found without an engine-module change."""
    path = f"{location}/test_changed.py"
    files = {"README.md": "base\n"}
    if tracked:
        files[path] = "# before\n"
    root = repo(files)
    _write(root, path, "# after\n")
    _write(root, f"{location}/helper.py", "# not a test\n")
    assert oc_check.changed_tests(root) == [path]


@pytest.mark.parametrize("module", ["engine/example.py", "checks/example.py", "tools/example.py"])
@pytest.mark.parametrize("nested", [False, True])
def test_importers_and_markers_preserve_root_order_and_include_v2(repo, module, nested):
    """Real module diffs retain old root ordering and add distinct v2 paths once."""
    locations = ["tests"] + (["tests/v2/ops/deep"] if nested else [])
    imported = module[:-3].replace("/", ".")
    files = {module: "VALUE = 1\n"}
    for location in locations:
        files.update({f"{location}/test_a_changed.py": "# base\n",
                      f"{location}/test_b_importer.py": f"from {imported} import VALUE\n",
                      f"{location}/test_c_both.py": f"import {imported}\n# land: always-run\n",
                      f"{location}/test_d_always.py": "  # land: always-run  \n",
                      f"{location}/test_e_unrelated.py": "# no dependency\n",
                      f"{location}/test_f_near_marker.py": "# land: always-run extra\n"})
    root = repo(files)
    _write(root, module, "VALUE = 2\n")
    for location in locations:
        _write(root, f"{location}/test_a_changed.py", "# changed\n")
    expected = [f"{p}/test_a_changed.py" for p in locations]
    expected += [f"{p}/test_{name}.py" for p in locations for name in ("b_importer", "c_both")]
    expected += [f"{p}/test_d_always.py" for p in locations]
    assert oc_check.changed_tests(root) == expected


def test_test_only_diff_does_not_activate_unchanged_markers_or_importers(repo):
    """The existing engine/checks/tools-change condition remains necessary."""
    root = repo({"tests/test_root.py": "# land: always-run\n",
                 "tests/v2/ops/test_nested.py": "# land: always-run\nimport engine.example\n",
                 "tests/v2/ops/test_changed.py": "# base\n"})
    _write(root, "tests/v2/ops/test_changed.py", "# changed\n")
    assert oc_check.changed_tests(root) == ["tests/v2/ops/test_changed.py"]


def test_deleted_tests_and_non_test_helpers_are_not_selected(repo):
    """Deletion filtering and test filename boundaries remain intact."""
    root = repo({"tests/test_gone.py": "# base\n", "tests/v2/ops/test_gone.py": "# base\n"})
    (root / "tests/test_gone.py").unlink()
    (root / "tests/v2/ops/test_gone.py").unlink()
    _write(root, "tests/v2/ops/helper.py", "# not a test\n")
    _write(root, "tests/fixtures/test_fixture.py", "# outside the test layout\n")
    assert oc_check.changed_tests(root) == []
