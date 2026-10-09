"""The package test selector and the test-layout ratchet.

# land: always-run
# packages: engine.v2.contracts, engine.v2.foundation, engine.v2.data, engine.v2.features, engine.v2.models, engine.v2.registry, engine.v2.domain.generation, engine.v2.domain.scenarios, engine.v2.domain.valuation, engine.v2.domain.simulation, engine.v2.scoring, engine.v2.evaluation, engine.v2.research, engine.v2.ledger, engine.v2.models.training, engine.v2.serving, engine.v2.ops, engine.v2.dashboard, engine.v2.diagnosis, engine.v2.parity

Path mapping, the reverse-import closure, integration declarations,
unmapped-path and unsafe-path failure, and a cache regression. Ratchet: R1-R6.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from checks import test_layout_budget as tb  # noqa: E402
from checks import test_selection as ts  # noqa: E402


def test_fixed_dependency_spellings_have_no_unresolved_loads():
    """Classify five source files in a bounded graph, not their full closure.

    The adjacent regression checks fixed training targets and package inits.
    """
    from tools import mutation_pilot as mp

    paths = [
        "tests/v2/integration/test_computed_moves.py",
        "tests/v2/foundation/test_v2_ops_foundation.py",
        "tests/v2/models/test_v2_models_no_fit.py",
        "tests/v2/models/training/test_v2_models_training_recipe_fields.py",
        "tools/baseline_export.py",
    ]
    assert mp.unresolved_import_files(paths) == set()
    graph = mp.build_import_graph(paths)
    assert mp.dynamic_files(graph) == set()


def test_lazy_training_imports_keep_every_fixed_target_and_package_init():
    from tools import mutation_pilot as mp

    tests = [
        "tests/v2/models/test_v2_models_no_fit.py",
        "tests/v2/models/training/test_v2_models_training_recipe_fields.py",
    ]
    targets = {f"engine/models/training/{name}.py" for name in (
        "gate", "gate_forecast_analog", "implied_t1", "runup_move", "size_model", "iv_crush",
    )}
    targets |= {"engine/__init__.py", "engine/models/__init__.py",
                "engine/models/training/__init__.py"}
    graph = mp.build_import_graph(tests + sorted(targets))
    for test in tests:
        assert targets <= graph.precise[test]


def test_phase1_coverage_keeps_relocated_foundation_file(tmp_path):
    """A missing required test must not silently disappear from the suite."""
    from checks.v2_coverage_ratchet import phase1_suite

    relative = "tests/v2/foundation/test_v2_ops_foundation.py"
    assert relative in phase1_suite(tmp_path)
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text("")
    assert relative in phase1_suite(tmp_path)


def _integration(root, name, body):
    directory = root / "tests/v2/integration"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(body)
    return ts.INTEGRATION + name


# --------------------------------------------------------------------------
# selector
# --------------------------------------------------------------------------


def test_package_selection_is_the_reverse_closure_plus_meta(tmp_path):
    selection = ts.select(["engine/v2/data/store.py", "tests/v2/data/test_store.py"],
                          root=tmp_path)
    assert not selection.full_suite
    assert "tests/v2/data" in selection.targets
    assert "tests/v2/scoring" in selection.targets       # a higher layer may import data
    assert "tests/v2/contracts" not in selection.targets  # a lower layer may not
    assert set(ts.META_TESTS) <= set(selection.targets)


def test_longest_directory_selects_models_training_not_models(tmp_path):
    selection = ts.select(["engine/v2/models/training/fit.py"], root=tmp_path)
    assert not selection.full_suite
    assert "tests/v2/models/training" in selection.targets
    assert "tests/v2/models" not in selection.targets


@pytest.mark.parametrize("path", [
    "engine/v2/models/training.py", "tests/v2/models/training.py",
])
def test_conflicting_names_map_to_the_containing_package(tmp_path, path):
    selection = ts.select([path], root=tmp_path)
    assert not selection.full_suite
    assert "tests/v2/models" in selection.targets


def test_only_imports_and_sink_edges(tmp_path):
    foundation = ts.select(["engine/v2/foundation/env.py"], root=tmp_path)
    assert "tests/v2/parity" in foundation.targets          # parity only_imports 0.5
    assert "tests/v2/contracts" not in foundation.targets    # a lower layer may not
    dashboard = ts.select(["engine/v2/dashboard/app.py"], root=tmp_path)
    assert "tests/v2/diagnosis" in dashboard.targets         # a sink may import any package


@pytest.mark.parametrize("path", [
    "tools/mutation_pilot.py", "README.md", "tests/test_legacy.py", "checks/x.py",
    "engine/v2/__init__.py", ".github/workflows/x.yml",
])
def test_unmapped_paths_are_full_suite(tmp_path, path):
    selection = ts.select([path], root=tmp_path)
    assert selection.full_suite and path in selection.reason
    if path.startswith(".github/"):
        assert selection.reason == f"{path} is outside the layer map"


@pytest.mark.parametrize("path", [
    "/etc/passwd", "engine/v2/./ops/x.py", "engine/v2/ops/../../checks/x.py",
])
def test_traversal_paths_are_full_suite(tmp_path, path):
    selection = ts.select([path], root=tmp_path)
    assert selection.full_suite
    assert "unsafe" in selection.reason and path in selection.reason


def test_markdown_selects_meta_not_the_full_suite(tmp_path):
    selection = ts.select(["guides/anything.md", "engine/v2/models/ARCHITECTURE.md"],
                          root=tmp_path)
    assert not selection.full_suite
    assert set(ts.META_TESTS) <= set(selection.targets)


def test_integration_test_selects_itself_and_declared_packages(tmp_path):
    path = _integration(tmp_path, "test_x.py", "# packages: ops\n")
    selection = ts.select([path], root=tmp_path)
    assert path in selection.targets and "tests/v2/ops" in selection.targets
    assert path in ts.select(["engine/v2/ops/run.py"], root=tmp_path).targets


@pytest.mark.parametrize("body", [
    "", "# packages:\n", "# packages: ops, ops\n", "# packages: nope\n",
    "# packages: ops\n# packages: models\n",
])
def test_invalid_integration_declaration_is_full_suite(tmp_path, body):
    selection = ts.select([_integration(tmp_path, "test_bad.py", body)], root=tmp_path)
    assert selection.full_suite and selection.errors


def test_misplaced_declaration_on_a_package_test_is_full_suite(tmp_path):
    directory = tmp_path / "tests/v2/ops"
    directory.mkdir(parents=True)
    (directory / "test_x.py").write_text("# packages: ops\n")
    assert ts.select(["tests/v2/ops/test_x.py"], root=tmp_path).full_suite


def test_a_stale_cache_cannot_widen_or_replace_the_selection(tmp_path):
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache/test_selection.json").write_text(
        json.dumps({"targets": ["tests/"], "full_suite": True}))
    selection = ts.select(["engine/v2/dashboard/app.py"], root=tmp_path)
    assert not selection.full_suite and "tests/" not in selection.targets
    assert "tests/v2/dashboard" in selection.targets


# --------------------------------------------------------------------------
# ratchet (R1-R6)
# --------------------------------------------------------------------------


def test_r1_a_new_test_outside_the_layout_fails(tmp_path):
    report = tb.check_layout({"tests/test_a.py"}, {"tests/test_a.py", "tests/test_b.py"},
                             1, 1, tmp_path)
    assert any("tests/test_b.py" in finding for finding in report.findings)


def test_r1_pytest_suffix_pattern_is_rejected_without_changing_root_budget(tmp_path):
    assert tb.is_test("tests/foo_test.py")
    assert tb._rooted("tests/test_a.py")
    assert not tb._rooted("tests/foo_test.py")
    report = tb.check_layout(
        {"tests/test_a.py"},
        {"tests/test_a.py", "tests/foo_test.py"},
        1,
        1,
        tmp_path,
    )
    assert any("tests/foo_test.py" in finding for finding in report.findings)


@pytest.mark.parametrize("path", ["checks/test_selection.py", "checks/test_layout_budget.py"])
def test_r1_ignores_new_check_implementation_modules(tmp_path, path):
    report = tb.check_layout({"tests/test_a.py"}, {"tests/test_a.py", path}, 1, 1, tmp_path)
    assert report.ok


def test_r2_root_growth_fails(tmp_path):
    report = tb.check_layout({"tests/test_a.py"}, {"tests/test_a.py", "tests/test_b.py"},
                             1, 2, tmp_path)
    assert any(finding.startswith("root test set grew") for finding in report.findings)


def test_r3_stale_or_increased_budget_fails(tmp_path):
    base = {"tests/test_a.py", "tests/test_b.py"}
    head = {"tests/test_a.py", "tests/v2/ops/test_b.py"}
    assert any("budget 2 != root test count 1" in f
               for f in tb.check_layout(base, head, 2, 2, tmp_path).findings)
    assert tb.check_layout(base, head, 2, 1, tmp_path).ok
    assert any("grew" in f for f in tb.check_layout({"tests/test_a.py"}, {"tests/test_a.py"},
                                                    1, 2, tmp_path).findings)


def test_r4_unknown_package_directory_fails(tmp_path):
    report = tb.check_layout(set(), {"tests/v2/nope/test_x.py"}, 0, 0, tmp_path)
    assert any("names no package" in finding for finding in report.findings)


def test_r4_unknown_package_helper_module_fails(tmp_path):
    report = tb.check_layout(set(), {"tests/v2/nope/helper.py"}, 0, 0, tmp_path)
    assert any("names no package" in finding for finding in report.findings)


def test_r4_deleted_unknown_package_path_does_not_fail(tmp_path):
    report = tb.check_layout({"tests/v2/retired/test_x.py"}, set(), 0, 0, tmp_path)
    assert report.ok


def test_r5_rename_as_delete_plus_add_lowers_the_budget(tmp_path):
    base = {"tests/test_a.py", "tests/test_b.py"}
    head = {"tests/v2/ops/test_a.py", "tests/test_b.py"}
    assert tb.check_layout(base, head, 2, 1, tmp_path).ok
    assert tb.check_layout(base, head, 2, 2, tmp_path).findings


def test_r6_a_revert_restoring_a_root_test_fails(tmp_path):
    report = tb.check_layout({"tests/v2/ops/test_a.py", "tests/test_b.py"},
                             {"tests/test_a.py", "tests/test_b.py"}, 1, 2, tmp_path)
    assert any(finding.startswith("root test set grew") for finding in report.findings)
    assert any(finding.startswith("budget grew") for finding in report.findings)


def test_ratchet_validates_integration_declarations(tmp_path):
    path = _integration(tmp_path, "test_x.py", "# packages: ops\n")
    assert tb.check_layout(set(), {path}, 0, 0, tmp_path).ok
    (tmp_path / path).write_text("# packages: nope\n")
    report = tb.check_layout(set(), {path}, 0, 0, tmp_path)
    assert any("unknown package" in finding for finding in report.findings)


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                   env=tb._git_env())


def test_cli_reads_real_git_and_accepts_a_staged_move(tmp_path, monkeypatch):
    redirected = str(tmp_path / "redirected.git")
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.setenv(name, redirected)
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "checks").mkdir()
    _git(root, "init")
    _git(root, "symbolic-ref", "HEAD", "refs/heads/main")
    (root / "tests/test_a.py").write_text("")
    (root / "tests/test_b.py").write_text("")
    (root / "checks/test_layout_budget.txt").write_text("2\n")
    _git(root, "add", "-A")
    _git(root, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-m", "base")
    (root / "tests/v2/ops").mkdir(parents=True)
    _git(root, "mv", "tests/test_a.py", "tests/v2/ops/test_a.py")
    (root / "checks/test_layout_budget.txt").write_text("1\n")
    _git(root, "add", "-A")
    assert tb.main(["--repo-root", str(root), "--base-ref", "main", "--quiet"]) == 0


def _require_origin_main_for_cli_test():
    args = ["git", "-C", str(ROOT), "rev-parse", "--verify", "--quiet",
            "refs/remotes/origin/main"]
    probe = subprocess.run(args, capture_output=True, env=tb._git_env())
    if probe.returncode == 1:
        pytest.skip("origin/main not fetched (shallow manual run); the ratchet is enforced on pull_request runs")
    assert probe.returncode == 0, probe.stderr.decode().strip()


def test_submitted_checkout_guard_skips_when_origin_main_is_missing(monkeypatch):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(pytest.skip.Exception, match="origin/main not fetched"):
        _require_origin_main_for_cli_test()


def test_submitted_checkout_guard_allows_origin_main_when_present(monkeypatch):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, b"abc123\n", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    _require_origin_main_for_cli_test()


def test_cli_checks_the_submitted_checkout():
    _require_origin_main_for_cli_test()
    assert tb.main(["--quiet"]) == 0


def test_cli_fallback_counts_only_root_budget_pattern(tmp_path):
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "checks").mkdir()
    _git(root, "init")
    _git(root, "symbolic-ref", "HEAD", "refs/heads/main")
    (root / "tests/test_a.py").write_text("")
    (root / "tests/foo_test.py").write_text("")
    _git(root, "add", "-A")
    _git(root, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-m", "base")
    (root / "checks/test_layout_budget.txt").write_text("1\n")
    assert tb.main(["--repo-root", str(root), "--base-ref", "main", "--quiet"]) == 0
