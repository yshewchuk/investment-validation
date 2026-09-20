"""O29 staged-byte lint, precise coverage ratchets, and hook drift controls."""
from __future__ import annotations

import copy
import subprocess

from checks import install_hooks
from checks.v2_coverage_ratchet import PHASE1_SUITE_VERSION as SUITE_VERSION, phase1_compare as compare
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


def coverage_document(executed=50001, executable=100000, mode="serial"):
    doc = {"suite_version": SUITE_VERSION, "packages": {
        "engine.v2.ops": {"executed": executed, "executable": executable,
                          "empty": False, "missing_files": []}}}
    if mode is not None:
        doc["mode"] = mode
    return doc


def test_parallel_mode_baseline_is_refused():
    """A baseline captured under --parallel can bake in executor_watchdog.py's
    real /proc-race noise (execution only ever ADDS lines under contention)
    as if it were the truth; refuse it outright rather than let a later
    serial measurement read as a false regression against it."""
    baseline = coverage_document(mode="parallel")
    measured = coverage_document()
    assert compare(measured, baseline)[0]["code"] == "BASELINE_NOT_SERIAL"


def test_parallel_measurement_against_serial_baseline_has_no_mode_finding():
    """The refusal targets the baseline's mode, not the measurement's: a
    --parallel measurement compared against a serial baseline stays fine."""
    baseline = coverage_document()
    measured = coverage_document(mode="parallel")
    assert compare(measured, baseline) == []


def test_baseline_missing_mode_for_todays_suite_version_is_refused():
    """A baseline that predates this field, for the CURRENT suite version,
    cannot be trusted as serial either -- it is exactly today's committed
    baselines before they were migrated to carry "mode"."""
    baseline = coverage_document(mode=None)
    measured = coverage_document()
    assert compare(measured, baseline)[0]["code"] == "BASELINE_NOT_SERIAL"


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
