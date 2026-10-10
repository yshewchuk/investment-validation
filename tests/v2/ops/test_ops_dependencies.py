"""Negative controls for planned ops boundaries and their one-way ratchet."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from checks import import_layers as layers
from checks import ops_dependencies as ops

ROOT = Path(__file__).resolve().parents[3]
EMPTY = {"forbidden": [], "cycles": []}


def policy(packages, exceptions=None):
    return {
        "levels": {"entry": 5, "runtime": 4, "workflows": 3, "stores": 2,
                   "legacy": 1, "native": 1, "providers": 1, "core": 0},
        "packages": packages,
        "exceptions": copy.deepcopy(EMPTY if exceptions is None else exceptions),
    }


def sources(**modules):
    return {ops.SOURCE + name.replace(".", "/") + ".py": text.encode()
            for name, text in modules.items()}


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True,
                          env={key: value for key, value in os.environ.items()
                               if not key.startswith("GIT_")}).stdout.strip()


def repository(root, files, rules, *, bootstrap=False):
    git(root, "init", "-q")
    git(root, "config", "user.name", "Synthetic Test")
    git(root, "config", "user.email", "test@example.invalid")
    for path, blob in files.items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob)
    destination = root / ops.POLICY
    destination.parent.mkdir(parents=True)
    if not bootstrap:
        destination.write_text(json.dumps(rules))
    git(root, "add", ".")
    git(root, "commit", "-qm", "synthetic base")
    base = git(root, "rev-parse", "HEAD")
    destination.write_text(json.dumps(rules))
    return base


@pytest.mark.parametrize("statement", [
    "import engine.v2.ops.high",
    "from engine.v2.ops.high import value",
    "from engine.v2.ops import high as alias",
    "from . import high",
    "from .high import value",
    "def lazy():\n    from .high import value",
    "if False:\n    from . import high",
])
def test_lazy_and_static_import_forms_are_checked(statement):
    actual = ops.findings(sources(low=statement, high=""), policy({"core": ["low"], "runtime": ["high"]}))
    assert actual == {"forbidden": {("low", "high")}, "cycles": set()}


def test_package_initializer_and_multilevel_relative_imports():
    files = sources(**{"providers.__init__": "from .. import low",
                       "providers.client": "from . import helper\nfrom ..low import value",
                       "providers.helper": "", "low": ""})
    rules = policy({"providers": ["providers", "providers.client", "providers.helper"], "core": ["low"]})
    assert ops.findings(files, rules) == {"forbidden": set(), "cycles": set()}


def test_directions_include_transitive_downward_and_forbid_peers():
    names = ("entry", "runtime", "workflows", "stores", "legacy", "native", "providers", "core")
    rules = policy({name: [name + "_module"] for name in names})
    for source in names:
        for target in names:
            files = sources(**{name + "_module": "" for name in names})
            files[ops.SOURCE + source + "_module.py"] = f"from . import {target}_module".encode()
            actual = ops.findings(files, rules)
            forbidden = source != target and rules["levels"][source] <= rules["levels"][target]
            assert bool(actual["forbidden"]) == forbidden, (source, target)


def test_cycles_are_exact_edges_not_member_sets():
    rules = policy({"core": ["a", "b", "c"]})
    files = sources(a="from . import b", b="from . import c", c="from . import a")
    before = ops.findings(files, rules)
    assert before["cycles"] == {("a", "b"), ("b", "c"), ("c", "a")}
    files[ops.SOURCE + "a.py"] += b"\nfrom . import c"
    after = ops.findings(files, rules)
    assert ops.check_findings(after, before, before) == ["ops cycles: new: a -> c"]


def test_explicit_namespace_import_is_not_lost_beside_submodule():
    files = sources(__init__="from . import a", a="import engine.v2.ops, engine.v2.ops.b", b="")
    actual = ops.findings(files, policy({"entry": [""], "core": ["a", "b"]}))
    assert actual["forbidden"] == {("a", "")}
    assert actual["cycles"] == {("", "a"), ("a", "")}


def test_stale_exceptions_must_be_removed_and_cannot_return():
    known = {"forbidden": {("a", "b")}, "cycles": set()}
    clean = {"forbidden": set(), "cycles": set()}
    assert ops.check_findings(known, known, known) == []
    assert ops.check_findings(clean, known, known) == ["ops forbidden: stale exception: a -> b"]
    assert ops.check_findings(clean, clean, known) == []
    assert ops.check_findings(known, known, clean) == ["ops forbidden: expanded exception: a -> b"]


@pytest.mark.parametrize("packages", [{"core": []}, {"core": ["a", "b"]},
                                     {"core": ["a", "a"]}, {"unknown": ["a"]}])
def test_incomplete_duplicate_or_unknown_ownership_refuses(packages):
    with pytest.raises(ValueError):
        ops.findings(sources(a=""), policy(packages))


def test_invalid_source_refuses():
    with pytest.raises(SyntaxError):
        ops.findings(sources(a="def invalid"), policy({"core": ["a"]}))


def test_duplicate_source_module_cannot_hide_a_forbidden_import():
    files = sources(low="from . import high", high="", **{"low.__init__": ""})
    with pytest.raises(ValueError, match="duplicate ops source"):
        ops.findings(files, policy({"core": ["low"], "runtime": ["high"]}))


@pytest.mark.parametrize("level", [float("nan"), float("inf"), "4", True])
def test_invalid_levels_cannot_disable_direction_checks(level):
    rules = policy({"core": ["low"], "runtime": ["high"]})
    rules["levels"]["core"] = level
    with pytest.raises(ValueError, match="levels must be integers"):
        ops.findings(sources(low="from . import high", high=""), rules)


@pytest.mark.parametrize("bootstrap", [False, True])
def test_git_environment_cannot_redirect_repository(tmp_path, monkeypatch, bootstrap):
    intended, other = tmp_path / "intended", tmp_path / "other"
    intended.mkdir()
    other.mkdir()
    rules = policy({"core": ["a"]})
    base = repository(intended, sources(a=""), rules, bootstrap=bootstrap)
    repository(other, sources(other=""), policy({"core": ["other"]}))
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    assert ops.check_repository(intended, base) == []


def test_git_fixture_setup_ignores_inherited_environment(tmp_path, monkeypatch):
    intended, other = tmp_path / "intended", tmp_path / "other"
    intended.mkdir()
    other.mkdir()
    original = repository(other, sources(other=""), policy({"core": ["other"]}))
    for name, value in {"GIT_DIR": other / ".git", "GIT_WORK_TREE": other,
                        "GIT_INDEX_FILE": other / ".git/index",
                        "GIT_OBJECT_DIRECTORY": other / ".git/objects"}.items():
        monkeypatch.setenv(name, str(value))
    repository(intended, sources(a=""), policy({"core": ["a"]}))
    assert git(other, "rev-parse", "HEAD") == original
    assert ops.SOURCE + "a.py" in git(intended, "ls-files").splitlines()
    assert ops.SOURCE + "a.py" not in git(other, "ls-files").splitlines()


@pytest.mark.parametrize("bootstrap", [False, True])
def test_real_git_path_rejects_new_violation_even_if_added_to_policy(tmp_path, bootstrap):
    rules = policy({"core": ["low"], "runtime": ["high"]})
    base = repository(tmp_path, sources(low="", high=""), rules, bootstrap=bootstrap)
    assert ops.check_repository(tmp_path, base) == []
    (tmp_path / ops.SOURCE / "low.py").write_text("def lazy():\n    from . import high\n")
    assert ops.check_repository(tmp_path, base) == ["ops forbidden: new: low -> high"]
    rules["exceptions"]["forbidden"].append(["low", "high"])
    (tmp_path / ops.POLICY).write_text(json.dumps(rules))
    assert ops.check_repository(tmp_path, base) == ["ops forbidden: expanded exception: low -> high"]


def test_real_git_path_removal_and_reintroduction(tmp_path):
    exceptions = {"forbidden": [["low", "high"]], "cycles": []}
    rules = policy({"core": ["low"], "runtime": ["high"]}, exceptions)
    base = repository(tmp_path, sources(low="from . import high", high=""), rules)
    assert ops.check_repository(tmp_path, base) == []
    (tmp_path / ops.SOURCE / "low.py").write_text("")
    assert ops.check_repository(tmp_path, base) == ["ops forbidden: stale exception: low -> high"]
    rules["exceptions"] = copy.deepcopy(EMPTY)
    (tmp_path / ops.POLICY).write_text(json.dumps(rules))
    assert ops.check_repository(tmp_path, base) == []
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "remove exception")
    base = git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / ops.SOURCE / "low.py").write_text("from . import high")
    (tmp_path / ops.POLICY).write_text(json.dumps({**rules, "exceptions": exceptions}))
    assert ops.check_repository(tmp_path, base) == ["ops forbidden: expanded exception: low -> high"]


def test_real_git_path_fails_closed_for_missing_source_policy_or_base(tmp_path):
    rules = policy({"core": ["a"]})
    base = repository(tmp_path, sources(a=""), rules)
    with pytest.raises(subprocess.CalledProcessError):
        ops.check_repository(tmp_path, "missing-ref")
    (tmp_path / ops.POLICY).write_text("not json")
    with pytest.raises(json.JSONDecodeError):
        ops.check_repository(tmp_path, base)
    (tmp_path / ops.POLICY).write_text(json.dumps(rules))
    (tmp_path / ops.SOURCE / "a.py").unlink()
    with pytest.raises(FileNotFoundError):
        ops.check_repository(tmp_path, base)


def test_real_tree_matches_enumerated_findings_and_layer_ci_entrypoint():
    rules = json.loads((ROOT / ops.POLICY).read_bytes())
    files = {path: (ROOT / path).read_bytes() for path in ops.tracked_paths(ROOT)
             if path.startswith(ops.SOURCE) and path.endswith(".py")}
    actual = ops.findings(files, rules)
    assert actual == {kind: {tuple(edge) for edge in edges} for kind, edges in rules["exceptions"].items()}
    result = subprocess.run([sys.executable, "checks/import_layers.py", "--all"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_worker_does_not_import_the_cli_entrypoint_even_lazily():
    rules = json.loads((ROOT / ops.POLICY).read_bytes())
    files = {path: (ROOT / path).read_bytes() for path in ops.tracked_paths(ROOT)
             if path.startswith(ops.SOURCE) and path.endswith(".py")}
    actual = ops.findings(files, rules)
    assert ("worker", "cli") not in actual["forbidden"]
    assert ("worker", "cli") not in actual["cycles"]
    files[ops.SOURCE + "worker.py"] += (
        b"\n\ndef _planted():\n    from engine.v2.ops import cli\n")
    assert ("worker", "cli") in ops.findings(files, rules)["forbidden"]


def test_snapshot_stages_does_not_import_the_stage_registry_even_lazily():
    rules = json.loads((ROOT / ops.POLICY).read_bytes())
    files = {path: (ROOT / path).read_bytes() for path in ops.tracked_paths(ROOT)
             if path.startswith(ops.SOURCE) and path.endswith(".py")}
    actual = ops.findings(files, rules)
    assert ("snapshot_stages", "stages") not in actual["forbidden"]
    assert ("stages", "snapshot_stages") not in actual["forbidden"]
    assert ("snapshot_stages", "core.snapshot_contracts") not in actual["forbidden"]
    assert ("stages", "core.snapshot_contracts") not in actual["forbidden"]
    files[ops.SOURCE + "snapshot_stages.py"] += (
        b"\n\ndef _planted():\n    from engine.v2.ops import stages\n")
    assert ("snapshot_stages", "stages") in ops.findings(files, rules)["forbidden"]


def test_finality_does_not_import_the_legacy_adapter_even_lazily():
    rules = json.loads((ROOT / ops.POLICY).read_bytes())
    files = {path: (ROOT / path).read_bytes() for path in ops.tracked_paths(ROOT)
             if path.startswith(ops.SOURCE) and path.endswith(".py")}
    actual = ops.findings(files, rules)
    assert ("finality", "legacy_adapter") not in actual["cycles"]
    assert ("legacy_adapter", "finality") not in actual["cycles"]
    files[ops.SOURCE + "finality.py"] += (
        b"\n\ndef _planted():\n    from engine.v2.ops import legacy_adapter\n")
    assert ("finality", "legacy_adapter") in ops.findings(files, rules)["cycles"]


def test_nightly_trigger_does_not_import_cli_even_lazily():
    rules = json.loads((ROOT / ops.POLICY).read_bytes())
    files = {path: (ROOT / path).read_bytes() for path in ops.tracked_paths(ROOT)
             if path.startswith(ops.SOURCE) and path.endswith(".py")}
    actual = ops.findings(files, rules)
    assert ("nightly_trigger", "cli") not in actual["forbidden"]
    assert not [edge for edge in actual["forbidden"] if edge[0] == "workflows.commands"]
    assert rules["exceptions"]["forbidden"] == []
    files[ops.SOURCE + "nightly_trigger.py"] += (
        b"\n\ndef _planted():\n    from engine.v2.ops import cli\n")
    assert ("nightly_trigger", "cli") in ops.findings(files, rules)["forbidden"]
    files[ops.SOURCE + "workflows/commands.py"] += (
        b"\n\ndef _planted():\n    from engine.v2.ops import supervisor\n")
    assert ("workflows.commands", "supervisor") in ops.findings(files, rules)["forbidden"]


def test_unit_receipts_does_not_import_incremental_data_even_lazily():
    rules = json.loads((ROOT / ops.POLICY).read_bytes())
    files = {path: (ROOT / path).read_bytes() for path in ops.tracked_paths(ROOT)
             if path.startswith(ops.SOURCE) and path.endswith(".py")}
    assert ["unit_receipts", "incremental_data"] not in rules["exceptions"]["cycles"]
    assert ("unit_receipts", "incremental_data") not in ops.findings(files, rules)["cycles"]
    files[ops.SOURCE + "unit_receipts.py"] += (
        b"\n\ndef _planted():\n    from engine.v2.ops import incremental_data\n")
    assert ("unit_receipts", "incremental_data") in ops.findings(files, rules)["cycles"]


def test_layer_cli_rejects_planted_ops_violation(tmp_path):
    files = {name.replace(".", "/") + "/__init__.py": b""
             for name in [*layers.CONTAINERS, *(p.dotted for p in layers.PACKAGES)]}
    files.update(sources(low="", high=""))
    rules = policy({"entry": [""], "core": ["low"], "runtime": ["high"]})
    base = repository(tmp_path, files, rules)
    command = [sys.executable, str(ROOT / "checks/import_layers.py"),
               "--repo-root", str(tmp_path), "--all", "--base-ref", base]
    valid = subprocess.run(command, capture_output=True, text=True)
    assert valid.returncode == 0, valid.stdout + valid.stderr
    (tmp_path / ops.SOURCE / "low.py").write_text("def lazy():\n    from . import high\n")
    invalid = subprocess.run(command, capture_output=True, text=True)
    assert invalid.returncode == 1, invalid.stdout + invalid.stderr
    assert "[ops-dependency]" in invalid.stderr
    assert "ops forbidden: new: low -> high" in invalid.stderr


def test_external_layers_and_dynamic_import_restriction_remain():
    assert not layers.check_files({"engine/v2/serving/example.py": b"from engine.v2.ops import cli"}).ok
    assert not layers.check_files(sources(a="import importlib\nimportlib.import_module('engine.v2.ops.cli')")).ok
    assert layers.check_files({"engine/v2/dashboard/example.py": b"from engine.v2.ops import cli"}).ok


def test_checker_registration_does_not_expand_runtime_mutation_pool():
    config = tomllib.loads((ROOT / "tools/mutation_pilot.toml").read_text())
    registered = config["modules"]["ops_dependency_checks"]
    assert registered["excluded"]
    assert registered["mutate"] == []
    assert registered["tests"] == ["tests/v2/ops/test_ops_dependencies.py"]
    assert registered["tests"][0] not in config["modules"]["ops_runtime"]["tests"]
