"""Add-only registration proof and the production selector/CLI boundary.

# packages: foundation, ops
"""
from __future__ import annotations

import copy
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import mutation_pilot as pilot

CONFIG = "tools/mutation_pilot.toml"
READERS = {
    "tests/v2/integration/test_mutation_ci.py", "tests/v2/integration/test_gremlin_ci.py", "tests/test_gremlins_ci.py",
    "tests/test_mutation_results.py", "tests/test_checks_mutation_ratchet.py",
    "tests/v2/integration/test_mutation_registration_selection.py",
    "tests/v2/ops/test_final_holdout_authority.py",
}
BASE = '''
[defaults]
timeout_constant = 2.0
timeout_multiplier = 5.0
max_children = 2
pytest_args = []
deselect = []
copy = ["."]
[pr_selection]
inert = []
inert_skip = []
full_suite = ["tools/*", ".github/*", "requirements*.txt", "tests/conftest.py"]
always_run = ["tests/test_always.py"]
[modules.alpha]
why = "synthetic"
mutate = ["engine/old.py"]
tests = ["tests/test_old*.py"]
[modules.beta]
mutate = ["engine/beta.py"]
tests = ["tests/test_beta.py"]
'''


@pytest.fixture
def case(monkeypatch):
    base = tomllib.loads(BASE)
    cfg = copy.deepcopy(base)
    cfg["modules"]["alpha"]["mutate"].append("engine/new.py")
    cfg["modules"]["alpha"]["tests"].append("tests/test_new.py")
    graph = {p: set() for p in READERS | {
        "engine/old.py", "engine/new.py", "engine/beta.py", "tests/test_old.py",
        "tests/test_old_other.py", "tests/test_new.py", "tests/test_importer.py",
        "tests/test_always.py", "tests/test_unrelated.py", "tests/test_beta.py",
    }}
    graph["tests/test_new.py"] = {"engine/new.py"}
    graph["tests/test_importer.py"] = {"engine/new.py"}
    monkeypatch.setattr(pilot, "unresolved_import_files", lambda paths: set())
    monkeypatch.setattr(pilot, "_path_reader_tests", lambda *a, **k: set())  # synthetic graph files are not on disk; an unreadable closure file would count as a literal reader
    return base, cfg, [CONFIG, "engine/new.py", "tests/test_new.py"], graph


def test_reader_registry_mirrors_all_mutation_config_readers():
    assert READERS == pilot._REGISTRATION_READERS
    assert "tests/v2/ops/test_final_holdout_authority.py" in READERS


def selected(case):
    base, cfg, changed, graph = case
    return pilot.select_pr_tests(cfg, changed, base_cfg=base, graph=graph)


@pytest.mark.parametrize("fields", [("mutate",), ("tests",), ("mutate", "tests")])
def test_additions_union_ordinary_selection_module_tests_and_readers(case, fields):
    base, cfg, changed, graph = case
    for field, path in (("mutate", "engine/new.py"), ("tests", "tests/test_new.py")):
        if field not in fields:
            cfg["modules"]["alpha"][field] = base["modules"]["alpha"][field]
            changed.remove(path)
    expected = READERS | {"tests/test_old.py", "tests/test_old_other.py", "tests/test_always.py"}
    expected |= set(pilot.select_pr_tests(cfg, changed[1:], graph=graph))
    if "tests" in fields:
        expected.add("tests/test_new.py")
    assert selected(case) == sorted(expected)
    assert "tests/test_unrelated.py" not in selected(case)


def test_multiple_modules_and_insertions_preserve_all_old_tests(case):
    base, cfg, changed, graph = case
    cfg["modules"]["alpha"]["tests"].reverse()  # new literal may precede preserved old entries
    cfg["modules"]["beta"]["tests"].append("tests/test_unrelated.py")
    changed.append("tests/test_unrelated.py")
    assert {"tests/test_beta.py", "tests/test_unrelated.py"} <= set(selected(case))


def test_noncanonical_test_only_registration_retains_selector_regressions(case):
    base, cfg, changed, graph = case
    cfg["modules"]["alpha"] = copy.deepcopy(base["modules"]["alpha"])
    cfg["modules"]["beta"]["tests"].append("tests/test_new.py")
    changed.remove("engine/new.py")
    assert "tests/v2/integration/test_mutation_registration_selection.py" in selected(case)


def test_source_fail_safe_and_test_only_leaf_behavior_survive(case):
    base, cfg, changed, graph = case
    dynamic = pilot._ImportGraph(graph)
    dynamic.precise = copy.deepcopy(graph)
    dynamic["tests/test_unrelated.py"] = set(graph) - {"tests/test_unrelated.py"}
    assert "tests/test_unrelated.py" in selected((base, cfg, changed, dynamic))
    cfg["modules"]["alpha"]["mutate"] = base["modules"]["alpha"]["mutate"]
    changed.remove("engine/new.py")
    assert "tests/test_unrelated.py" not in selected((base, cfg, changed, dynamic))


@pytest.mark.parametrize("problem", [
    "removal", "reorder", "duplicate", "new_module", "removed_module", "excluded",
    "skip", "why", "defaults", "numeric_type", "selection", "unknown_top",
    "unknown_module", "unknown_defaults", "wrong_list_type", "wrong_member_type",
    "glob", "overlap", "untracked", "absent_diff", "traversal", "backslash",
    "missing_reader", "unmatched_old_test", "not_a_test", "unchanged",
])
def test_unsupported_registration_changes_fail_closed(case, problem):
    base, cfg, changed, graph = case
    mod = cfg["modules"]["alpha"]
    if problem == "removal":
        mod["mutate"].remove("engine/old.py")
    elif problem == "reorder":
        base["modules"]["alpha"]["mutate"].append("engine/beta.py")
        mod["mutate"].insert(0, "engine/beta.py")
    elif problem == "duplicate":
        mod["tests"].append("tests/test_new.py")
    elif problem == "new_module":
        cfg["modules"]["new"] = copy.deepcopy(mod)
    elif problem == "removed_module":
        del cfg["modules"]["beta"]
    elif problem in {"excluded", "why"}:
        mod[problem] = "changed"
    elif problem == "skip":
        mod["skip"] = ["engine/new.py"]
    elif problem == "defaults":
        cfg["defaults"]["max_children"] = 3
    elif problem == "numeric_type":
        cfg["defaults"]["timeout_constant"] = 2  # 2 == 2.0 is not exact preservation
    elif problem == "selection":
        cfg["pr_selection"]["always_run"] = []
    elif problem == "unknown_top":
        base["future"] = cfg["future"] = {}
    elif problem == "unknown_module":
        base["modules"]["beta"]["future"] = cfg["modules"]["beta"]["future"] = "x"
    elif problem == "unknown_defaults":
        base["defaults"]["future"] = cfg["defaults"]["future"] = "x"
    elif problem == "wrong_list_type":
        mod["tests"] = "tests/test_new.py"
    elif problem == "wrong_member_type":
        mod["tests"].append(1)
    elif problem in {"glob", "untracked", "traversal", "backslash", "not_a_test"}:
        path = {"glob": "tests/test_*.py", "untracked": "tests/test_missing.py",
                "traversal": "tests/../test_new.py", "backslash": "tests/test_\\x.py",
                "not_a_test": "tests/helper.py"}[problem]
        mod["tests"][-1] = path
        changed.append(path)
        if problem != "untracked":
            graph[path] = set()
    elif problem == "overlap":
        base["modules"]["alpha"]["mutate"] = ["engine/*.py"]
        mod["mutate"][0] = "engine/*.py"
    elif problem == "absent_diff":
        changed.remove("tests/test_new.py")
    elif problem == "missing_reader":
        del graph["tests/test_mutation_results.py"]
    elif problem == "unmatched_old_test":
        del graph["tests/test_old.py"], graph["tests/test_old_other.py"]
    elif problem == "unchanged":
        cfg["modules"] = copy.deepcopy(base["modules"])
    assert selected(case) is None


@pytest.mark.parametrize("coverage", ["absent", "empty", "new_only"])
def test_affected_module_needs_existing_nonempty_test_coverage(case, coverage):
    base, cfg, changed, graph = case
    for config in (base, cfg):
        if coverage == "absent":
            config["modules"]["alpha"].pop("tests")
        else:
            config["modules"]["alpha"]["tests"] = []
    if coverage == "new_only":
        cfg["modules"]["alpha"]["tests"].append("tests/test_new.py")
    else:
        changed.remove("tests/test_new.py")
    assert selected(case) is None


@pytest.mark.parametrize("base", [None, {}, [], {"modules": []}])
def test_missing_or_malformed_parsed_base_is_full_suite(case, base):
    _, cfg, changed, graph = case
    assert selected((base, cfg, changed, graph)) is None


@pytest.mark.parametrize("path", [
    "tools/other.py", ".github/workflows/tests.yml", "requirements.txt",
    "tests/conftest.py", "tests/test_deleted.py", "engine/unreached.py",
])
def test_other_full_suite_reasons_are_unchanged(case, path):
    case[2].append(path)
    assert selected(case) is None


@pytest.mark.parametrize("failure", ["graph", "scan"])
def test_analysis_failure_is_still_full_suite(case, monkeypatch, failure):
    def fail(*args):
        raise OSError("synthetic")

    base, cfg, changed, graph = case
    monkeypatch.setattr(pilot, "build_import_graph", fail)
    if failure == "scan":
        monkeypatch.setattr(pilot, "unresolved_import_files", fail)
    assert pilot.select_pr_tests(cfg, changed, base_cfg=base,
                                 graph=graph if failure == "scan" else None) is None


def test_mutation_matrix_semantics_are_unchanged(case):
    _, cfg, changed, graph = case
    assert pilot.changed_modules(cfg, ["alpha", "beta"], changed, graph=graph) == ["alpha", "beta"]


@pytest.fixture
def config_repo(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Selector test")
    git("config", "user.email", "selector@example.invalid")
    path = tmp_path / CONFIG
    path.parent.mkdir()
    path.write_text(BASE)
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    path.write_text(BASE.replace('"engine/old.py"', '"engine/old.py", "engine/new.py"'))
    git("add", ".")
    git("commit", "-qm", "head")
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    return git, path, base, tomllib.loads(path.read_text())


def test_base_blob_matches_exact_merge_base_and_head(config_repo):
    git, path, base, cfg = config_repo
    head = git("rev-parse", "HEAD")
    git("checkout", "-qb", "advanced-base", base)
    path.write_text(BASE.replace("max_children = 2", "max_children = 3"))
    git("commit", "-qam", "base advanced independently")
    advanced = git("rev-parse", "HEAD")
    git("checkout", "-q", head)
    assert pilot.registration_base_config(advanced, cfg, [CONFIG]) == tomllib.loads(BASE)
    assert pilot.registration_base_config(base, cfg, []) is None
    cfg["defaults"]["max_children"] = 4
    assert pilot.registration_base_config(base, cfg, [CONFIG]) is None


@pytest.mark.parametrize("base", ["", "main", "--help", "0" * 40])
def test_unavailable_or_non_sha_base_fails_closed(config_repo, base):
    _, _, _, cfg = config_repo
    assert pilot.registration_base_config(base, cfg, [CONFIG]) is None


@pytest.mark.parametrize("content", ["invalid [ toml", "#" * (1024 * 1024 + 1)],
                         ids=["malformed", "oversized"])
def test_bad_or_oversized_base_blob_fails_closed(config_repo, content):
    git, path, _, cfg = config_repo
    path.write_text(content)
    git("commit", "-qam", "bad base")
    base = git("rev-parse", "HEAD")
    path.write_text(BASE)
    git("commit", "-qam", "valid head")
    assert pilot.registration_base_config(base, tomllib.loads(BASE), [CONFIG]) is None


def test_git_timeout_fails_closed(config_repo, monkeypatch):
    _, _, base, cfg = config_repo

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 10
        raise subprocess.TimeoutExpired("git", 10)

    monkeypatch.setattr(pilot.subprocess, "run", timeout)
    assert pilot.registration_base_config(base, cfg, [CONFIG]) is None


def test_cli_threads_verified_base_and_keeps_missing_base_full(case, tmp_path, monkeypatch, capsys):
    base, cfg, changed, graph = case
    path = tmp_path / "changed"
    path.write_bytes("\0".join(changed).encode())
    monkeypatch.setattr(pilot, "build_import_graph", lambda: graph)
    args = SimpleNamespace(changed_files=str(path))
    assert pilot.cmd_select_tests(cfg, args) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "__ALL__"
    assert out.splitlines()[1] == "# tools/mutation_pilot.toml change is not an add-only registration"
    monkeypatch.setattr(pilot, "registration_base_config", lambda sha, config, paths: base)
    assert pilot.cmd_select_tests(cfg, args) == 0
    assert [ln for ln in capsys.readouterr().out.splitlines() if not ln.startswith("#")] == selected(case)


def test_workflow_passes_its_base_sha_and_new_test_is_registered():
    root = Path(__file__).resolve().parents[3]
    workflow = (root / ".github/workflows/tests.yml").read_text()
    assert '--base-sha "$BASE_SHA"' in workflow
    cfg = tomllib.loads((root / CONFIG).read_text())
    path = Path(__file__).relative_to(root).as_posix()
    assert any(path in mod.get("tests", []) for mod in cfg["modules"].values())


def test_architecture_documents_resource_and_coverage_fallbacks():
    root = Path(__file__).resolve().parents[3]
    contract = " ".join((root / "ARCHITECTURE.md").read_text().split())
    assert "size/time limits; exceeding either limit selects the full suite" in contract
    assert "without nonempty base and head test expansions selects the full suite" in contract


def _with_scanner(base, cfg, graph):
    # A module whose tests scan a whole tree (see issue #672): the import graph never links it.
    for config in (base, cfg):
        config["modules"]["scan"] = {"mutate": ["engine/scan.py"], "tests": ["tests/test_scan.py"],
                                     "scans_tree": "engine/"}
    graph.update({"engine/scan.py": set(), "tests/test_scan.py": set(),
                  "engine/newb.py": set(), "tests/test_newb.py": {"engine/newb.py"}})


@pytest.mark.parametrize("which, expected", [
    ("alpha", ["alpha", "scan"]),
    ("beta", ["beta", "scan"]),
])
def test_registration_selects_exactly_the_registered_module_plus_tree_scanners(
        case, which, expected):
    base, cfg, changed, graph = case
    _with_scanner(base, cfg, graph)
    if which == "beta":
        cfg["modules"]["alpha"] = copy.deepcopy(base["modules"]["alpha"])
        cfg["modules"]["beta"]["mutate"].append("engine/newb.py")
        cfg["modules"]["beta"]["tests"].append("tests/test_newb.py")
        changed = [CONFIG, "engine/newb.py", "tests/test_newb.py"]
    got = pilot.changed_modules(cfg, ["alpha", "beta", "scan"], changed, graph=graph, base_cfg=base)
    assert got == expected


@pytest.mark.parametrize("path, expected", [
    ("engine/beta.py", ["beta"]),
    ("tests/test_beta.py", ["beta"]),
])
def test_ordinary_change_does_not_pull_in_tree_scanners(case, path, expected):
    base, cfg, _, graph = case
    _with_scanner(base, cfg, graph)
    got = pilot.changed_modules(cfg, ["alpha", "beta", "scan"], [path], graph=graph)
    assert got == expected


def test_registration_plus_other_changed_path_unions_their_selections(case):
    base, cfg, changed, graph = case
    for config in (base, cfg):
        config["modules"]["gamma"] = {"mutate": ["engine/gamma.py"], "tests": ["tests/test_gamma.py"]}
    graph.update({"engine/gamma.py": set(), "tests/test_gamma.py": set()})
    got = pilot.changed_modules(cfg, ["alpha", "beta", "gamma"], [*changed, "engine/beta.py"],
                                graph=graph, base_cfg=base)
    assert got == ["alpha", "beta"]


@pytest.mark.parametrize("edit", ["defaults", "no_base"])
def test_unverified_config_change_still_selects_every_module(case, edit):
    base, cfg, changed, graph = case
    if edit == "defaults":
        cfg["defaults"]["max_children"] = 3
    else:
        base = None
    got = pilot.changed_modules(cfg, ["alpha", "beta"], changed, graph=graph, base_cfg=base)
    assert got == ["alpha", "beta"]


@pytest.mark.parametrize("value, ok", [("engine/", True), (5, False), ("", False)])
def test_scans_tree_validator_itself_refuses_bad_values(case, value, ok):
    # The same value sits in base AND cfg, so the identity check cannot reject first.
    base, cfg, changed, graph = case
    for config in (base, cfg):
        config["modules"]["beta"]["scans_tree"] = value
    got = pilot.registration_tests(cfg, base, set(changed), set(graph))
    assert (got is not None) is ok


def test_gremlin_matrix_reads_base_sha_from_environment(case, tmp_path, monkeypatch):
    import gremlin_pilot

    base, cfg, changed, graph = case
    path = tmp_path / "changed"
    path.write_bytes("\0".join(changed).encode())
    monkeypatch.setattr(gremlin_pilot.pilot, "build_import_graph", lambda: graph)
    seen = []

    def fake_base(sha, config, paths):
        seen.append(sha)
        return base

    monkeypatch.setattr(gremlin_pilot.pilot, "registration_base_config", fake_base)
    monkeypatch.setenv("BASE_SHA", "a" * 40)
    assert gremlin_pilot.select_modules(cfg, "", str(path)) == ["alpha"]
    assert seen == ["a" * 40]


def test_mutation_matrix_reads_base_sha_from_environment(case, tmp_path, monkeypatch, capsys):
    base, cfg, changed, graph = case
    path = tmp_path / "changed"
    path.write_bytes("\0".join(changed).encode())
    monkeypatch.setattr(pilot, "build_import_graph", lambda: graph)
    seen = []

    def fake_base(sha, config, paths):
        seen.append(sha)
        return base

    monkeypatch.setattr(pilot, "registration_base_config", fake_base)
    monkeypatch.setenv("BASE_SHA", "b" * 40)
    assert pilot.cmd_matrix(cfg, SimpleNamespace(only="", changed_files=str(path))) == 0
    assert capsys.readouterr().out.strip() == '["alpha"]'
    assert seen == ["b" * 40]


@pytest.mark.parametrize("extra", [[], ["engine/NOTES.md"]])
def test_scanner_needs_a_python_change_under_its_prefix(case, extra):
    base, cfg, _, graph = case
    _with_scanner(base, cfg, graph)
    for config in (base, cfg):
        config["pr_selection"]["inert"] = ["engine/NOTES.md"]
    cfg["modules"]["alpha"]["mutate"] = list(base["modules"]["alpha"]["mutate"])
    changed = [CONFIG, "tests/test_new.py", *extra]
    got = pilot.changed_modules(cfg, ["alpha", "beta", "scan"], changed, graph=graph, base_cfg=base)
    assert got == ["alpha"]


def _with_path_tests(case, base_value, cfg_value):
    base, cfg, changed, graph = case
    if base_value is not None:
        base["pr_selection"]["path_tests"] = base_value
    if cfg_value is not None:
        cfg["pr_selection"]["path_tests"] = cfg_value
    return base, cfg, changed, graph


def test_path_tests_change_is_not_an_add_only_registration(case):
    base, cfg, changed, graph = _with_path_tests(
        case, None, {"engine/*": ["tests/test_always.py"]})
    assert pilot.registration_tests(cfg, base, set(changed), set(graph)) is None


def test_identical_path_tests_keep_the_registration_valid(case):
    same = {"engine/*": ["tests/test_unrelated.py"]}
    result = selected(_with_path_tests(case, dict(same), dict(same)))
    assert isinstance(result, list) and result
    assert "tests/test_unrelated.py" in result and "tests/test_new.py" in result


def test_empty_path_tests_list_is_rejected(case):
    empty = {"engine/*": []}
    base, cfg, changed, graph = _with_path_tests(case, dict(empty), dict(empty))
    assert pilot.registration_tests(cfg, base, set(changed), set(graph)) is None
