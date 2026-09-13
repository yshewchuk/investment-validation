"""O29 staged-byte lint, precise coverage ratchets, and hook drift controls."""
from __future__ import annotations

import copy
import subprocess

from checks import install_hooks
from checks.rearchitecture_phase1_coverage import SUITE_VERSION, compare
from checks.rearchitecture_phase1_lint import check, run

CONFIG = b'[lint]\nselect = ["E4", "E7", "E9", "F", "I"]\n'
LOCK = "ruff==0.16.7\n"


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)


def test_linter_names_actual_rule_and_missing_pin():
    result = check({"engine/v2/ops/planted.py": b"import json\n"}, LOCK, CONFIG)
    assert not result["ok"]
    assert result["findings"][0]["code"] == "F401"
    assert check({}, "ruff>=0.1\n", CONFIG)["code"] == "LINTER_UNAVAILABLE"
    assert check({}, "ruff==0.0.1\n", CONFIG)["code"] == "LINTER_VERSION_DRIFT"


def test_staged_bad_worktree_good_and_the_reverse(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "requirements.txt").write_text(LOCK)
    (tmp_path / "ruff.toml").write_bytes(CONFIG)
    path = tmp_path / "engine/v2/ops/planted.py"
    path.parent.mkdir(parents=True)
    path.write_text("import json\n")
    git(tmp_path, "add", ".")
    path.write_text("VALUE = 1\n")
    assert not run(tmp_path)["ok"]
    assert run(tmp_path, worktree=True)["ok"]
    git(tmp_path, "add", "engine/v2/ops/planted.py")
    path.write_text("import json\n")
    assert run(tmp_path)["ok"]
    assert not run(tmp_path, worktree=True)["ok"]


def coverage_document(executed=50001, executable=100000):
    return {"suite_version": SUITE_VERSION, "packages": {
        "engine.v2.ops": {"executed": executed, "executable": executable,
                          "empty": False, "missing_files": []}}}


def test_ratchet_does_not_round_away_a_small_regression():
    old = coverage_document()
    new = coverage_document(executed=50000)
    assert compare(new, old)[0]["code"] == "COVERAGE_REGRESSION"
    assert compare(old, new) == []


def test_unmeasured_missing_and_new_packages_do_not_get_free_zero_baseline():
    old = coverage_document()
    missing = copy.deepcopy(old)
    missing["packages"]["engine.v2.ops"]["missing_files"] = ["engine/v2/ops/unmeasured.py"]
    assert compare(missing, old)[0]["code"] == "COVERAGE_FILES_MISSING"
    assert compare(coverage_document(executed=0), old)[0]["code"] == "COVERAGE_UNMEASURED_PACKAGE"
    old["packages"]["engine.v2.ops"].update(empty=True, executed=0, executable=0)
    assert compare(coverage_document(), old)[0]["code"] == "COVERAGE_NEW_PACKAGE_BASELINE_REQUIRED"


def test_empty_inventory_or_impossible_counts_cannot_pass_coverage():
    old = coverage_document()
    assert compare({"suite_version": SUITE_VERSION, "packages": {}}, old) == [
        {"code": "COVERAGE_PACKAGE_INVENTORY_DRIFT"}]
    for executed, executable in ((101, 100), (-1, 100), (True, 100)):
        result = compare(coverage_document(executed, executable), old)
        assert result[0]["code"] == "COVERAGE_COUNTS_INVALID"
    changed = coverage_document()
    changed["suite_version"] = "unreviewed"
    assert compare(changed, old)[0]["code"] == "COVERAGE_SUITE_DRIFT"


def test_clean_clone_missing_outdated_and_nonexecutable_hook(tmp_path):
    git(tmp_path, "init", "-q")
    source = tmp_path / "checks/hooks/pre-commit"
    source.parent.mkdir(parents=True)
    source.write_text("#!/bin/sh\ntrue\n")
    assert not install_hooks.state(tmp_path)["installed"]
    assert install_hooks.install(tmp_path)["current"]
    source.write_text("#!/bin/sh\nfalse\n")
    assert not install_hooks.state(tmp_path)["current"]
    assert install_hooks.install(tmp_path, force=True)["current"]
    target = install_hooks.hooks_dir(tmp_path) / "pre-commit"
    target.chmod(0o644)
    assert not install_hooks.state(tmp_path)["executable"]
