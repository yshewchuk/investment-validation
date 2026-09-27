"""Mutation CI: row/summary building, query filters, history, config and workflow shape.

Data-free and mutmut-free: every mutmut state file here is synthetic, written
in the layout mutmut 3.8 uses (``mutants/<file>.meta``, ``mutmut-stats.json``).
"""
from __future__ import annotations
# land: always-run

import csv
import io
import json
import re
import signal
import subprocess
import sys
import time
import tomllib
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import mutation_pilot as pilot  # noqa: E402
import mutation_report as rep  # noqa: E402
import mutation_results as mr  # noqa: E402

SEP = "ǁ"
SOURCE = '''\
def add(a, b):
    return a + b


class Box:
    def size(self, n):
        if n > 1:
            return n
        return 0
'''
ADD = "engine.v2.toy.x_add__mutmut_"
SIZE = f"engine.v2.toy.x{SEP}Box{SEP}size__mutmut_"


def make_work(tmp_path: Path, codes: dict[str, int | None], hashes=None, fingerprint=None) -> Path:
    work = tmp_path / "work"
    (work / "engine" / "v2").mkdir(parents=True)
    (work / "engine" / "v2" / "toy.py").write_text(SOURCE)
    meta = work / "mutants" / "engine" / "v2" / "toy.py.meta"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({
        "exit_code_by_key": codes,
        "hash_by_function_name": hashes or {"x_add": "h1", f"x{SEP}Box{SEP}size": "h2"},
        "type_check_error_by_key": {}, "durations_by_key": {k: 0.0 for k in codes},
        "estimated_durations_by_key": {k: 0.0 for k in codes}}))
    (work / "mutants" / "mutmut-stats.json").write_text(
        json.dumps({"config_fingerprint": fingerprint or {"a": 1}}))
    return work


INFO = {"run_id": "7", "sha": "abc", "ref": "refs/heads/main", "trigger": "push",
        "mode": "incremental"}
FILES = ["engine/v2/toy.py"]


# -- row building ------------------------------------------------------------------

def test_status_mapping_covers_every_mutmut_outcome():
    assert mr.report_status(1) == "killed"
    assert mr.report_status(37) == "killed"  # type-check rejection
    assert mr.report_status(0) == "survived"
    assert mr.report_status(33) == mr.report_status(5) == "no_tests"
    assert mr.report_status(36) == "timeout"
    assert mr.report_status(-9) == mr.report_status(2) == mr.report_status(999) == "suspicious"
    assert mr.report_status(None) == "skipped"
    assert set(mr._REPORT_STATUS.values()) == set(mr.STATUSES)


def test_function_names_from_mutant_keys():
    assert mr.function_of(ADD + "3") == "add"
    assert mr.function_of(SIZE + "12") == "Box.size"
    assert mr.function_of("engine.v2.a.x__private__mutmut_1") == "_private"


def test_rows_carry_the_schema_and_locate_each_mutant(tmp_path):
    work = make_work(tmp_path, {ADD + "1": 1, SIZE + "1": 0, SIZE + "2": None})
    rows = mr.build_rows(work, "toy", FILES, INFO, diffs=False)
    assert [tuple(r) for r in rows] == [mr.ROW_FIELDS] * 3
    by = {r["mutant_name"]: r for r in rows}
    assert by[ADD + "1"]["status"] == "killed" and by[ADD + "1"]["function"] == "add"
    assert by[ADD + "1"]["line"] == 1
    assert by[SIZE + "1"]["status"] == "survived" and by[SIZE + "1"]["line"] == 6
    assert by[SIZE + "2"]["status"] == "skipped"
    assert all(r["schema_version"] == mr.SCHEMA_VERSION and r["sha"] == "abc" for r in rows)
    assert all(r["retested_this_run"] is None for r in rows)  # no snapshot: unknown


def test_changed_line_points_at_the_removed_line():
    lines = SOURCE.splitlines()
    diff = "--- a\n+++ b\n@@ -1,4 +1,4 @@\n-    if n > 1:\n+    if n >= 1:\n"
    assert mr.changed_line(diff, lines, (6, 9)) == 7
    assert mr.changed_line("", lines, (6, 9)) == 6
    assert mr.changed_line(diff, lines, None) is None


def test_retested_marks_only_what_this_run_decided(tmp_path):
    before_codes = {ADD + "1": 1, ADD + "2": 0, SIZE + "1": None}
    work = make_work(tmp_path, before_codes)
    snap = mr.snapshot(work, FILES)
    # The run: add's verdicts stand, size's pending mutant is decided, add #3 is
    # new, and add #2 flips (a test got stronger).
    meta = work / "mutants" / "engine" / "v2" / "toy.py.meta"
    data = json.loads(meta.read_text())
    data["exit_code_by_key"] = {ADD + "1": 1, ADD + "2": 1, ADD + "3": 0, SIZE + "1": 0}
    meta.write_text(json.dumps(data))
    rows = {r["mutant_name"]: r for r in mr.build_rows(work, "toy", FILES, INFO, before=snap,
                                                         diffs=False)}
    assert rows[ADD + "1"]["retested_this_run"] is False
    assert rows[ADD + "2"]["retested_this_run"] is True
    assert rows[ADD + "3"]["retested_this_run"] is True
    assert rows[SIZE + "1"]["retested_this_run"] is True


def test_a_changed_function_hash_or_config_counts_as_retested(tmp_path):
    work = make_work(tmp_path, {ADD + "1": 1, SIZE + "1": 1})
    snap = mr.snapshot(work, FILES)
    snap["files"]["engine/v2/toy.py"]["hash_by_function_name"]["x_add"] = "old"
    rows = {r["mutant_name"]: r["retested_this_run"]
            for r in mr.build_rows(work, "toy", FILES, INFO, before=snap, diffs=False)}
    assert rows == {ADD + "1": True, SIZE + "1": False}
    snap["config_fingerprint"] = {"a": 2}
    rows = mr.build_rows(work, "toy", FILES, INFO, before=snap, diffs=False)
    assert all(r["retested_this_run"] for r in rows)


def test_triage_file_format_and_staleness(tmp_path):
    path = tmp_path / "triage.toml"
    path.write_text(f'''
[[triage]]
mutant = "{ADD}1"
verdict = "EQUIVALENT"
note = "a + b is commutative here"
diff_contains = "b + a"

[[triage]]
mutant = "{SIZE}1"
verdict = "LOW-VALUE"
note = "debug-only branch"
''')
    tri = mr.load_triage(path)
    fresh = mr.triage_for(ADD + "1", "-    return a + b\n+    return b + a\n", tri)
    assert fresh == {"verdict": "EQUIVALENT", "note": "a + b is commutative here", "stale": False}
    stale = mr.triage_for(ADD + "1", "+    return a - b\n", tri)
    assert stale["stale"] is True
    assert not mr.is_triaged({"triage": stale}) and mr.is_triaged({"triage": fresh})
    assert mr.triage_for(SIZE + "1", None, tri)["stale"] is False  # no pin: never stale
    assert mr.triage_for("other", None, tri) is None
    path.write_text('[[triage]]\nmutant = "m"\nverdict = "MAYBE"\nnote = "x"\n')
    with pytest.raises(ValueError):
        mr.load_triage(path)


def test_the_checked_in_triage_file_parses():
    mr.load_triage()  # raises on a malformed entry


# -- summaries -------------------------------------------------------------------------

def _row(module, file, function, status, triage=None, retested=True, name="m"):
    return {"schema_version": 1, "run_id": "7", "sha": "abc", "ref": "r", "trigger": "push",
            "mode": "incremental", "module": module, "file": file, "function": function,
            "line": 3, "mutant_name": name, "status": status, "mutmut_status": status,
            "retested_this_run": retested, "diff": "-a\n+b\n", "triage": triage}


def test_summary_counts_and_scores_per_module_and_file():
    rows = [_row("m", "a.py", "f", "killed"), _row("m", "a.py", "f", "timeout"),
            _row("m", "a.py", "g", "survived"), _row("m", "b.py", "h", "skipped"),
            _row("m", "b.py", "h", "no_tests",
                 triage={"verdict": "LOW-VALUE", "note": "n", "stale": False})]
    s = mr.summarize(rows, "m", INFO, ["a.py", "b.py", "c.py"], run_exit_code=0)
    assert (s["total"], s["checked"], s["skipped"]) == (5, 4, 1)
    assert s["score"] == 0.5  # (killed + timeout) / checked; no_tests counts against
    assert s["survived_untriaged"] == 1
    assert s["files"]["a.py"]["score"] == round(2 / 3, 4)
    assert s["files"]["b.py"]["score"] == 0.0
    assert s["files"]["c.py"]["score"] is None and s["files"]["c.py"]["total"] == 0
    assert s["run_exit_code"] == 0 and s["retested_this_run"] == 4  # b.py/h is skipped: excluded


def test_merge_combines_modules_and_totals(tmp_path):
    dirs = []
    for name, statuses in (("a", ["killed", "survived"]), ("b", ["killed", "killed"])):
        rows = [_row(name, f"{name}.py", "f", s, name=f"{name}{i}") for i, s in enumerate(statuses)]
        d = tmp_path / name
        d.mkdir()
        mr.write_jsonl(d / "results.jsonl", rows)
        (d / "summary.json").write_text(json.dumps(mr.summarize(rows, name, INFO, [f"{name}.py"])))
        dirs.append(d)
    merged = mr.merge_dirs(dirs, tmp_path / "merged")
    assert set(merged["modules"]) == {"a", "b"}
    assert (merged["total"], merged["killed"], merged["score"]) == (4, 3, 0.75)
    assert len(mr.read_jsonl(tmp_path / "merged" / "results.jsonl")) == 4
    assert "| **all** | 4 |" in (tmp_path / "merged" / "summary.md").read_text()


def test_job_summary_lists_untriaged_survivors_in_changed_functions():
    rows = [_row("m", "a.py", "f", "survived", name="n1"),
            _row("m", "a.py", "g", "survived", name="n2"),
            _row("m", "a.py", "f", "survived", name="n3",
                 triage={"verdict": "EQUIVALENT", "note": "x", "stale": False})]
    s = mr.summarize(rows, "m", INFO, ["a.py"])
    md = mr.markdown(s, rows, {("a.py", "f")})
    assert "functions this push changed: 1" in md and "<code>f</code>" in md
    assert "<code>g</code>" not in md
    md = mr.markdown(s, rows, None)
    assert "re-tested this run: 2" in md


def test_git_diff_hunks_map_to_new_side_lines():
    diff = ("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -3 +3 @@\n-a\n+b\n"
            "@@ -10,2 +9,0 @@\n-c\n-d\n@@ -20,0 +21,3 @@\n+e\n+f\n+g\n"
            "--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n")
    assert mr.changed_new_lines(diff) == {"x.py": {3, 9, 21, 22, 23}}


def test_changed_functions_from_a_real_git_history(tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    (tmp_path / "toy.py").write_text(SOURCE)
    git("add", ".")
    git("commit", "-qm", "one")
    (tmp_path / "toy.py").write_text(SOURCE.replace("n > 1", "n > 2"))
    git("commit", "-qam", "two")
    assert mr.changed_functions("HEAD~1", "HEAD", ["toy.py"], repo=tmp_path) == {("toy.py", "Box.size")}


# -- query CLI ------------------------------------------------------------------------

ROWS = [_row("canonical", "engine/v2/foundation/canonical.py", "_number", "survived", name="c1"),
        _row("canonical", "engine/v2/foundation/canonical.py", "_number", "killed", name="c2"),
        _row("pnl_sim", "engine/pnl_sim.py", "simulate", "no_tests", name="p1"),
        _row("pnl_sim", "engine/pnl_sim.py", "Leg.sign", "survived", name="p2",
             triage={"verdict": "EQUIVALENT", "note": "n", "stale": False}),
        _row("pnl_sim", "engine/pnl_sim.py", "Leg.size", "survived", name="p3",
             triage={"verdict": "EQUIVALENT", "note": "n", "stale": True})]


def names(rows):
    return [r["mutant_name"] for r in rows]


def test_query_filters():
    assert names(rep.filter_rows(ROWS, modules=["pnl_sim"])) == ["p1", "p2", "p3"]
    assert names(rep.filter_rows(ROWS, files=["engine/v2/*"])) == ["c1", "c2"]
    assert names(rep.filter_rows(ROWS, files=["engine/pnl_sim.py"], functions=["Leg.*"])) == ["p2", "p3"]
    assert names(rep.filter_rows(ROWS, statuses=["survived", "no_tests"])) == ["c1", "p1", "p2", "p3"]
    # untriaged: survived/no_tests without a current (non-stale) triage entry
    assert names(rep.filter_rows(ROWS, untriaged=True)) == ["c1", "p1", "p3"]
    changed = {("engine/pnl_sim.py", "simulate")}
    assert names(rep.filter_rows(ROWS, changed=changed)) == ["p1"]
    assert rep.filter_rows(ROWS, changed=set()) == []
    with pytest.raises(ValueError):
        rep.filter_rows(ROWS, statuses=["dead"])


def test_query_output_formats():
    buf = io.StringIO()
    rep.emit(ROWS[:2], "jsonl", show_diff=False, history_rows=False, out=buf)
    assert [json.loads(line)["mutant_name"] for line in buf.getvalue().splitlines()] == ["c1", "c2"]
    buf = io.StringIO()
    rep.emit(ROWS[3:], "csv", show_diff=False, history_rows=False, out=buf)
    parsed = list(csv.DictReader(io.StringIO(buf.getvalue())))
    assert parsed[0]["triage_verdict"] == "EQUIVALENT" and parsed[1]["triage_stale"] == "True"
    assert list(parsed[0]) == rep.CSV_FIELDS
    buf = io.StringIO()
    rep.emit(ROWS[3:], "table", show_diff=True, history_rows=False, out=buf)
    text = buf.getvalue()
    assert "[EQUIVALENT STALE]" in text and "    -a" in text and "-- 2 mutants" in text


def test_query_cli_reads_a_results_directory(tmp_path, capsys):
    mr.write_jsonl(tmp_path / "results.jsonl", ROWS)
    assert rep.main(["--dir", str(tmp_path), "--untriaged", "--format", "jsonl"]) == 0
    out = [json.loads(x)["mutant_name"] for x in capsys.readouterr().out.splitlines()]
    assert out == ["c1", "p1", "p3"]


def test_history_merges_runs_oldest_first_with_deltas():
    def summary(score_a, score_all, sha):
        return {"sha": sha, "mode": "full", "total": 10, "killed": 5, "survived": 5,
                "no_tests": 0, "skipped": 0, "survived_untriaged": 5, "score": score_all,
                "modules": {"a": {"total": 4, "killed": 2, "survived": 2, "no_tests": 0,
                                  "skipped": 0, "survived_untriaged": 2, "score": score_a}}}
    runs = [({"databaseId": 2, "createdAt": "2026-09-20T00:00:00Z"}, summary(0.75, 0.6, "s2")),
            ({"databaseId": 1, "createdAt": "2026-09-13T00:00:00Z"}, summary(0.5, 0.5, "s1"))]
    rows = rep.merge_history(runs)
    assert [(r["run_id"], r["module"]) for r in rows] == [(1, "a"), (1, "ALL"), (2, "a"), (2, "ALL")]
    assert [r["delta"] for r in rows] == [None, None, 0.25, 0.1]
    assert [r["module"] for r in rep.merge_history(runs, ["a"])] == ["a", "a"]


def test_report_cache_refuses_the_repo(monkeypatch):
    monkeypatch.setenv("MUTATION_REPORT_CACHE", str(ROOT / "scratch"))
    with pytest.raises(SystemExit):
        rep.cache_dir()


# -- driver --------------------------------------------------------------------------

def test_survivors_reset_when_the_tests_change(tmp_path):
    work = make_work(tmp_path, {ADD + "1": 1, ADD + "2": 0, SIZE + "1": 33})
    assert pilot.reset_on_test_change(work, FILES, "d1") == 0  # first run: record only
    assert pilot.reset_on_test_change(work, FILES, "d1") == 0  # unchanged
    assert pilot.reset_on_test_change(work, FILES, "d2") == 2
    meta = json.loads((work / "mutants" / "engine" / "v2" / "toy.py.meta").read_text())
    assert meta["exit_code_by_key"] == {ADD + "1": 1, ADD + "2": None, SIZE + "1": None}


def test_tests_digest_covers_selected_tests_and_shared_helpers(tmp_path):
    (tmp_path / "tests").mkdir()
    for name in ("test_a.py", "test_b.py", "conftest.py"):
        (tmp_path / "tests" / name).write_text("x = 1\n")
    base = pilot.tests_digest(tmp_path, ["tests/test_a.py"])
    (tmp_path / "tests" / "test_b.py").write_text("x = 2\n")  # not selected
    assert pilot.tests_digest(tmp_path, ["tests/test_a.py"]) == base
    (tmp_path / "tests" / "conftest.py").write_text("x = 2\n")  # shared helper
    assert pilot.tests_digest(tmp_path, ["tests/test_a.py"]) != base


def test_expand_is_ordered_and_refuses_dead_patterns():
    tracked = ["engine/v2/a/x.py", "engine/v2/a/y.py", "engine/v2/b/z.py"]
    assert pilot.expand(["engine/v2/a/*.py"], tracked, skip=["*/y.py"]) == ["engine/v2/a/x.py"]
    with pytest.raises(SystemExit):
        pilot.expand(["engine/v2/c/*.py"], tracked)


def test_config_hash_is_stable_across_calls():
    assert pilot.config_hash(CFG, "no_fit") == pilot.config_hash(CFG, "no_fit")


def test_config_hash_changes_when_the_modules_own_section_changes():
    base = pilot.config_hash(CFG, "no_fit")
    import copy
    mutated = copy.deepcopy(CFG)
    mutated["modules"]["no_fit"]["why"] = mutated["modules"]["no_fit"]["why"] + " (edited)"
    assert pilot.config_hash(mutated, "no_fit") != base


def test_config_hash_changes_when_defaults_changes():
    base = pilot.config_hash(CFG, "no_fit")
    import copy
    mutated = copy.deepcopy(CFG)
    mutated["defaults"]["timeout_multiplier"] = mutated["defaults"]["timeout_multiplier"] + 1
    assert pilot.config_hash(mutated, "no_fit") != base


def test_config_hash_ignores_a_different_modules_section():
    base = pilot.config_hash(CFG, "no_fit")
    import copy
    mutated = copy.deepcopy(CFG)
    mutated["modules"]["canonical"]["why"] = mutated["modules"]["canonical"]["why"] + " (edited)"
    assert pilot.config_hash(mutated, "no_fit") == base


def test_config_hash_ignores_a_newly_added_module():
    base = pilot.config_hash(CFG, "no_fit")
    import copy
    mutated = copy.deepcopy(CFG)
    mutated["modules"]["brand_new_module"] = dict(mutated["modules"]["no_fit"])
    assert pilot.config_hash(mutated, "no_fit") == base


def test_config_hash_refuses_an_unknown_module():
    with pytest.raises(SystemExit):
        pilot.config_hash(CFG, "not_a_real_module")


# -- mutmut diagnostics on the ops_legacy CI shard (run 36001042208) ---------
#
# mutmut only logs "failed to collect stats. runner returned 1" and swallows the
# child pytest output that explains it. The driver's remedy is mutmut's own
# supported ``debug = true`` config (``mutmut_config_text``): it is full-run
# verbosity, so it is on only when the environment opts in through
# ``MUTATION_PILOT_DEBUG`` or on the ops_legacy CI shard by default. These are
# mock tests: they never run mutmut or pytest; they check the config we hand
# mutmut, that the environment and module name reach the generated ``setup.cfg``,
# and that ``cmd_run`` keeps the real exit code without rerunning anything.
# ops_catalog_state now shares that CI-only default; it is covered by separate
# tests below so the ops_legacy ones above stay untouched.

DIAG_DEFAULTS = {"pytest_args": ["-p", "no:xdist", "-p", "no:cacheprovider"],
                 "deselect": ["tests/test_a.py::gate_needs_git"],
                 "timeout_constant": 2.0, "timeout_multiplier": 5.0, "max_children": 2}


def test_stats_debug_honors_an_explicit_value_over_the_ci_default(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    for on in ("1", "true", "TRUE", " yes ", "on"):
        monkeypatch.setenv("MUTATION_PILOT_DEBUG", on)
        assert pilot.stats_debug_enabled("ops_legacy")
        assert pilot.stats_debug_enabled("pnl_sim")
    for off in ("", "0", "false", "nope", "off"):
        monkeypatch.setenv("MUTATION_PILOT_DEBUG", off)
        assert not pilot.stats_debug_enabled("ops_legacy")
        assert not pilot.stats_debug_enabled("pnl_sim")
    # an explicit off wins even on the CI shard that would otherwise default on
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    for off in ("0", "false", "off"):
        monkeypatch.setenv("MUTATION_PILOT_DEBUG", off)
        assert not pilot.stats_debug_enabled("ops_legacy")


def test_stats_debug_defaults_on_only_for_ops_legacy_in_ci(monkeypatch):
    monkeypatch.delenv("MUTATION_PILOT_DEBUG", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert pilot.stats_debug_enabled("ops_legacy")
    assert not pilot.stats_debug_enabled("pnl_sim")
    assert not pilot.stats_debug_enabled("toy")
    # every other shard, and every local run, stays quiet
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert not pilot.stats_debug_enabled("ops_legacy")
    monkeypatch.setenv("GITHUB_ACTIONS", "false")
    assert not pilot.stats_debug_enabled("ops_legacy")


def test_stats_debug_defaults_on_for_ops_catalog_state_in_ci(monkeypatch):
    monkeypatch.delenv("MUTATION_PILOT_DEBUG", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert pilot.stats_debug_enabled("ops_catalog_state")
    # an unrelated shard stays quiet even under CI
    assert not pilot.stats_debug_enabled("pnl_sim")
    # and ops_catalog_state itself is quiet in every local run
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert not pilot.stats_debug_enabled("ops_catalog_state")
    monkeypatch.setenv("GITHUB_ACTIONS", "false")
    assert not pilot.stats_debug_enabled("ops_catalog_state")


def test_stats_debug_explicit_off_beats_the_ops_catalog_state_ci_default(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    for off in ("", "0", "false", "off"):
        monkeypatch.setenv("MUTATION_PILOT_DEBUG", off)
        assert not pilot.stats_debug_enabled("ops_catalog_state")
    for on in ("1", "true", "yes", "on"):
        monkeypatch.setenv("MUTATION_PILOT_DEBUG", on)
        assert pilot.stats_debug_enabled("ops_catalog_state")


def test_stats_debug_defaults_on_for_ops_runtime_in_ci(monkeypatch):
    monkeypatch.delenv("MUTATION_PILOT_DEBUG", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert pilot.stats_debug_enabled("ops_runtime")
    # an explicit off wins even on the CI shard that would otherwise default on
    monkeypatch.setenv("MUTATION_PILOT_DEBUG", "0")
    assert not pilot.stats_debug_enabled("ops_runtime")
    # and ops_runtime itself is quiet in every local run
    monkeypatch.delenv("MUTATION_PILOT_DEBUG", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert not pilot.stats_debug_enabled("ops_runtime")


def test_mutmut_config_text_adds_debug_only_when_requested():
    off = pilot.mutmut_config_text(DIAG_DEFAULTS, ["engine/x.py"], ["tests/test_a.py"],
                                   ["tests"], debug=False)
    assert "debug" not in off
    assert "process_isolation = forkserver" in off  # the load-bearing keys survive
    assert "--deselect=tests/test_a.py::gate_needs_git" in off
    on = pilot.mutmut_config_text(DIAG_DEFAULTS, ["engine/x.py"], ["tests/test_a.py"],
                                  ["tests"], debug=True)
    assert on.startswith(off) and on.endswith("debug = true\n")


def test_the_env_opt_in_reaches_the_generated_setup_cfg(tmp_path, monkeypatch):
    repo, home = tmp_path / "repo", tmp_path / "home"
    (repo / "engine").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "engine" / "x.py").write_text("x = 1\n")
    (repo / "tests" / "test_a.py").write_text("def t(): pass\n")
    cfg = {"defaults": dict(DIAG_DEFAULTS, copy=["engine", "tests"]),
           "modules": {"toy": {"mutate": ["engine/x.py"], "tests": ["tests/test_a.py"]}}}
    monkeypatch.setattr(pilot, "REPO", repo)
    monkeypatch.setenv("MUTATION_PILOT_HOME", str(home))
    monkeypatch.setattr(pilot, "_tracked", lambda paths: ["engine/x.py", "tests/test_a.py"])
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("MUTATION_PILOT_DEBUG", raising=False)
    work = pilot.sync_workdir("toy", cfg, fresh=True)
    assert "debug = true" not in (work / "setup.cfg").read_text()
    monkeypatch.setenv("MUTATION_PILOT_DEBUG", "1")
    work = pilot.sync_workdir("toy", cfg, fresh=True)
    assert "debug = true" in (work / "setup.cfg").read_text()


def test_ci_default_enables_debug_only_for_ops_legacy_setup_cfg(tmp_path, monkeypatch):
    repo, home = tmp_path / "repo", tmp_path / "home"
    (repo / "engine").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "engine" / "x.py").write_text("x = 1\n")
    (repo / "tests" / "test_a.py").write_text("def t(): pass\n")
    mod = {"mutate": ["engine/x.py"], "tests": ["tests/test_a.py"]}
    cfg = {"defaults": dict(DIAG_DEFAULTS, copy=["engine", "tests"]),
           "modules": {"ops_legacy": dict(mod), "pnl_sim": dict(mod)}}
    monkeypatch.setattr(pilot, "REPO", repo)
    monkeypatch.setenv("MUTATION_PILOT_HOME", str(home))
    monkeypatch.setattr(pilot, "_tracked", lambda paths: ["engine/x.py", "tests/test_a.py"])
    monkeypatch.delenv("MUTATION_PILOT_DEBUG", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    ops = pilot.sync_workdir("ops_legacy", cfg, fresh=True)
    assert "debug = true" in (ops / "setup.cfg").read_text()
    other = pilot.sync_workdir("pnl_sim", cfg, fresh=True)
    assert "debug = true" not in (other / "setup.cfg").read_text()


def test_ci_default_enables_debug_for_ops_catalog_state_setup_cfg(tmp_path, monkeypatch):
    repo, home = tmp_path / "repo", tmp_path / "home"
    (repo / "engine").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "engine" / "x.py").write_text("x = 1\n")
    (repo / "tests" / "test_a.py").write_text("def t(): pass\n")
    mod = {"mutate": ["engine/x.py"], "tests": ["tests/test_a.py"]}
    cfg = {"defaults": dict(DIAG_DEFAULTS, copy=["engine", "tests"]),
           "modules": {"ops_catalog_state": dict(mod), "pnl_sim": dict(mod)}}
    monkeypatch.setattr(pilot, "REPO", repo)
    monkeypatch.setenv("MUTATION_PILOT_HOME", str(home))
    monkeypatch.setattr(pilot, "_tracked", lambda paths: ["engine/x.py", "tests/test_a.py"])
    monkeypatch.delenv("MUTATION_PILOT_DEBUG", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    cat = pilot.sync_workdir("ops_catalog_state", cfg, fresh=True)
    assert "debug = true" in (cat / "setup.cfg").read_text()
    other = pilot.sync_workdir("pnl_sim", cfg, fresh=True)
    assert "debug = true" not in (other / "setup.cfg").read_text()


def _cmd_run(monkeypatch, tmp_path, rc, budget=None):
    """Run cmd_run with mutmut's subprocess stubbed; return (rc_out, calls)."""
    work = tmp_path / "work"
    (work / "mutants").mkdir(parents=True)
    monkeypatch.setattr(pilot, "sync_workdir", lambda *a, **k: work)
    monkeypatch.setattr(pilot, "mutate_files", lambda cfg, name, tracked=None: ["engine/x.py"])
    monkeypatch.setattr(pilot, "test_files", lambda cfg, name, tracked=None: ["tests/test_a.py"])
    monkeypatch.setattr(pilot, "tests_digest", lambda *a: "digest")
    monkeypatch.setattr(pilot, "reset_on_test_change", lambda *a, **k: 0)
    monkeypatch.setattr(mr, "snapshot", lambda *a, **k: {})
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return types.SimpleNamespace(returncode=rc)

    monkeypatch.setattr(pilot.subprocess, "run", fake_run)
    args = types.SimpleNamespace(module="toy", fresh=False, max_children=None, globs=[],
                                 time_budget_seconds=budget)
    return pilot.cmd_run({"defaults": DIAG_DEFAULTS, "modules": {"toy": {}}}, args), calls


def test_cmd_run_preserves_the_exit_code_without_rerunning(monkeypatch, tmp_path):
    rc, calls = _cmd_run(monkeypatch, tmp_path, rc=1)
    assert rc == 1  # the tool error still fails the job; scores are never silenced
    assert len(calls) == 1  # exactly one subprocess: no automatic pytest/mutmut rerun
    cmd, kw = calls[0]
    assert cmd[:5] == [sys.executable, "-u", "-m", "mutmut", "run"]
    assert "pytest" not in " ".join(cmd)
    assert kw["cwd"] == tmp_path / "work"


def test_cmd_run_no_longer_strips_the_harness_env(monkeypatch, tmp_path):
    # mutmut sets MUTANT_UNDER_TEST itself after launch and pytest sets
    # PYTEST_CURRENT_TEST, so filtering the parent environment cannot prevent a
    # child from inheriting them -- and had no supported basis. The only pinned
    # vars are the BLAS/OpenMP thread caps.
    monkeypatch.setenv("MUTANT_UNDER_TEST", "stats")
    monkeypatch.setenv("COVERAGE_PROCESS_START", "/rc")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "t::u (call)")
    _, calls = _cmd_run(monkeypatch, tmp_path, rc=0)
    env = calls[0][1]["env"]
    assert env["MUTANT_UNDER_TEST"] == "stats"
    assert env["COVERAGE_PROCESS_START"] == "/rc"
    assert env["PYTEST_CURRENT_TEST"] == "t::u (call)"
    assert env["OMP_NUM_THREADS"] == "1"  # the thread pin is still applied


def test_cmd_run_passes_the_time_budget_to_the_bounded_runner(monkeypatch, tmp_path):
    seen = {}

    def fake_bounded(cmd, cwd, env, budget):
        seen.update(cmd=cmd, cwd=cwd, budget=budget)
        return pilot.TIME_BUDGET_STOP_RC

    monkeypatch.setattr(pilot, "_run_with_time_budget", fake_bounded)
    rc, calls = _cmd_run(monkeypatch, tmp_path, rc=0, budget=3300)
    assert rc == pilot.TIME_BUDGET_STOP_RC == 124
    assert calls == []  # the unbounded subprocess.run path was NOT taken
    assert seen["budget"] == 3300
    assert seen["cmd"][:5] == [sys.executable, "-u", "-m", "mutmut", "run"]


def test_a_clean_time_box_stop_exits_the_process_with_status_124():
    """sys.exit(-1) exits 255; the workflow gate compares the process status, so the
    sentinel must be a valid status that survives the process boundary unchanged."""
    code = ("import sys; sys.path.insert(0, 'tools'); import mutation_pilot as p; "
            "p.cmd_run = lambda cfg, args: p.TIME_BUDGET_STOP_RC; "
            "sys.exit(p.main(['run', 'ops_runtime']))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 124, proc.stderr
    src = (ROOT / "tools" / "mutation_pilot.py").read_text()
    assert 'if __name__ == "__main__":\n    sys.exit(main())' in src


def test_run_with_time_budget_sigints_at_the_budget_and_reports_the_stop(monkeypatch, tmp_path):
    waits = []

    class FakeProc:
        pid = 4242

        def wait(self, timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired(cmd="mutmut", timeout=timeout)
            return 0

    signals = []
    monkeypatch.setattr(pilot.subprocess, "Popen", lambda cmd, **kw: FakeProc())
    monkeypatch.setattr(pilot.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(pilot, "SIGINT_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(pilot, "time", types.SimpleNamespace(monotonic=lambda: 1000.0))
    rc = pilot._run_with_time_budget([sys.executable, "-m", "mutmut", "run"], tmp_path, {}, 30.0)
    assert rc == pilot.TIME_BUDGET_STOP_RC == 124
    assert signals == [(4242, signal.SIGINT)]  # clean stop: no SIGKILL
    assert waits == [30.0, 0.01]  # budget, then the (patched) grace period


def test_run_with_time_budget_escalates_to_sigkill_when_sigint_is_ignored(monkeypatch, tmp_path):
    waits = []

    class FakeProc:
        pid = 777

        def wait(self, timeout=None):
            waits.append(timeout)
            if len(waits) <= 2:
                raise subprocess.TimeoutExpired(cmd="mutmut", timeout=timeout)
            return -9

    signals = []
    monkeypatch.setattr(pilot.subprocess, "Popen", lambda cmd, **kw: FakeProc())
    monkeypatch.setattr(pilot.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(pilot, "SIGINT_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(pilot, "time", types.SimpleNamespace(monotonic=lambda: 0.0))
    rc = pilot._run_with_time_budget(["mutmut"], tmp_path, {}, 5.0)
    assert rc == pilot.TIME_BUDGET_STOP_RC == 124
    assert signals == [(777, signal.SIGINT), (777, signal.SIGKILL)]
    assert waits == [5.0, 0.01, None]  # budget, grace, then the final reap


# -- config and workflow shape ----------------------------------------------------------

CFG = pilot.load_config()
WORKFLOW = yaml.safe_load((ROOT / ".github" / "workflows" / "mutation.yml").read_text())
JOBS = WORKFLOW["jobs"]


def _tracked(*paths):
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--", *paths],
                         check=True, capture_output=True, text=True).stdout
    return out.split()


def test_every_v2_file_is_in_exactly_one_module():
    tracked = _tracked("engine")
    owner: dict[str, list[str]] = {}
    for name in CFG["modules"]:
        for f in pilot.mutate_files(CFG, name, tracked):
            owner.setdefault(f, []).append(name)
    v2 = [f for f in tracked if f.startswith("engine/v2/") and f.endswith(".py")]
    assert [f for f in v2 if f not in owner] == []
    assert {f: o for f, o in owner.items() if len(o) > 1} == {}
    legacy = sorted(f for f in owner if not f.startswith("engine/v2/"))
    assert legacy == ["engine/models/no_fit.py", "engine/pnl_sim.py"]  # the pilot's only


def test_ops_catalog_is_split_into_disjoint_role_shards():
    """ops_catalog was one 12-file job and exceeded the 330-minute mutmut step
    cap (CI run 35950291319). Its shards must stay split (small, disjoint, and
    collectively exactly its original files) or the timeout comes back."""
    tracked = _tracked("engine")
    assert "ops_catalog" not in CFG["modules"]
    shards = [n for n in pilot.enabled_modules(CFG) if n.startswith("ops_catalog")]
    assert len(shards) >= 3
    owned = [pilot.mutate_files(CFG, s, tracked) for s in shards]
    flat = [f for files in owned for f in files]
    assert len(flat) == len(set(flat))  # mutate sets are disjoint
    assert max(len(files) for files in owned) <= 6
    assert set(flat) == {f"engine/v2/ops/{n}" for n in [
        "__init__.py", "bootstrap.py", "catalog.py", "checkpoints.py", "errors.py",
        "lifecycle.py", "migrations.py", "scheduler.py", "schema.py", "schema_runtime.py",
        "store_barrier.py", "submission.py"]}
    for s in shards:  # each shard keeps the job-stack pool it was timed out with
        assert pilot.test_files(CFG, s)


def test_modules_are_well_formed():
    tracked_tests = _tracked("tests")
    for name, mod in CFG["modules"].items():
        assert re.fullmatch(r"[a-z0-9_]+", name), name  # cache restore-key prefixes stay unique
        if mod.get("excluded"):
            assert len(mod["excluded"]) > 20, name  # a real reason
            continue
        assert mod["why"] and pilot.test_files(CFG, name, tracked_tests), name
        assert not any("test_v2_dashboard_browser" in t for t in mod["tests"])  # needs npm


def test_matrix_is_built_from_the_toml(capsys):
    enabled = pilot.enabled_modules(CFG)
    assert enabled and all(not CFG["modules"][n].get("excluded") for n in enabled)

    class Args:
        only = ""
    pilot.cmd_matrix(CFG, Args)
    assert json.loads(capsys.readouterr().out) == enabled
    Args.only = f"{enabled[-1]}, {enabled[0]}"
    pilot.cmd_matrix(CFG, Args)
    assert json.loads(capsys.readouterr().out) == [enabled[0], enabled[-1]]
    excluded = [n for n, m in CFG["modules"].items() if m.get("excluded")]
    for bad in ["nope"] + excluded[:1]:
        Args.only = bad
        with pytest.raises(SystemExit):
            pilot.cmd_matrix(CFG, Args)

    plan = " ".join(s.get("run", "") for s in JOBS["plan"]["steps"])
    assert "tools/gremlin_pilot.py matrix" in plan
    assert "tools/mutation_pilot.py" not in plan  # gremlins runner, not the old mutmut one
    assert JOBS["mutate"]["strategy"]["matrix"]["module"] == \
        "${{ fromJSON(needs.plan.outputs.modules) }}"
    assert JOBS["mutate"]["strategy"]["fail-fast"] is False


def test_workflow_triggers_and_concurrency():
    on = WORKFLOW.get("on", WORKFLOW.get(True))  # PyYAML reads a bare `on` as True
    assert on["push"]["branches"] == ["main"]
    assert "pull_request" not in on  # PR runs are gated behind Tests via workflow_run instead
    assert on["workflow_run"] == {"workflows": ["Tests"], "types": ["completed"]}
    assert on["schedule"] and "cron" in on["schedule"][0]
    assert set(on["workflow_dispatch"]["inputs"]) == {"fresh", "modules"}
    # A PR's gated workflow_run run gets its own per-branch group (the PR
    # number lives inside a possibly-empty array and is unsafe to index at
    # this top level -- see the workflow's "PR context: workflow_run"
    # comment) and cancels its predecessor; push also cancels an older
    # still-running push. schedule/dispatch keep the exact old group value
    # and stay non-cancelling.
    assert WORKFLOW["concurrency"]["group"] == (
        "mutation-gremlins-${{ github.event_name == 'workflow_run' "
        "&& github.event.workflow_run.head_branch || github.ref }}"
    )
    assert WORKFLOW["concurrency"]["cancel-in-progress"] == \
        "${{ github.event_name == 'workflow_run' || github.event_name == 'push' }}"
    assert WORKFLOW["permissions"] == {"contents": "read"}
    plan = JOBS["plan"]["steps"][-1]["run"]
    assert '"$EVENT" = "push"' in plan and '"$FRESH" = "false"' in plan and "mode=full" in plan


def test_gremlins_plan_checks_out_full_history_only_for_pull_request():
    # fetch-depth 0 is needed to diff against the PR base sha; push/schedule/
    # dispatch keep the default shallow depth (1) -- byte-identical to before,
    # since the ternary's false branch is a literal 1, not an omitted default.
    checkout = JOBS["plan"]["steps"][0]
    assert checkout["uses"] == "actions/checkout@v4"
    assert checkout["with"]["fetch-depth"] == \
        "${{ github.event_name == 'workflow_run' && '0' || 1 }}"


def test_gremlins_plan_narrows_the_matrix_on_pull_request_via_changed_files():
    plan_step = JOBS["plan"]["steps"][-1]
    assert plan_step["env"]["BASE_SHA"] == (
        "${{ github.event_name == 'workflow_run' && "
        "join(github.event.workflow_run.pull_requests.*.base.sha, ',') || "
        "github.event.pull_request.base.sha }}"
    )
    plan = plan_step["run"]
    # push/schedule/dispatch: CHANGED_ARGS stays empty, so the matrix command is
    # byte-identical to the pre-selection command (no --changed-files at all,
    # since an empty unquoted expansion contributes zero argv words).
    assert 'CHANGED_ARGS=""' in plan
    assert 'if [ "$EVENT" = "pull_request" ]; then' in plan
    assert 'git diff -z --no-renames --name-only "$BASE_SHA"...HEAD' in plan
    assert 'CHANGED_ARGS="--changed-files' in plan
    assert 'modules=$(python3 tools/gremlin_pilot.py matrix --only "$ONLY" $CHANGED_ARGS)' in plan


def test_workflow_pins_python_and_mutmut():
    header = (ROOT / "requirements.txt").read_text()
    assert f"# python {WORKFLOW['env']['PYTHON_VERSION']} " in header
    dev = (ROOT / "requirements-dev.txt").read_text()
    assert re.search(r"^mutmut==\d+\.\d+\.\d+$", dev, re.M)
    for job in JOBS.values():
        for step in job["steps"]:
            if str(step.get("uses", "")).startswith("actions/setup-python"):
                assert step["with"]["python-version"] == "${{ env.PYTHON_VERSION }}"


def _step(job, predicate, jobs=None):
    jobs = JOBS if jobs is None else jobs
    return next(s for s in jobs[job]["steps"] if predicate(s))


def test_workflow_cache_key_and_restore_policy():
    assert WORKFLOW["env"]["GREMLINS_CACHE"] == ".gremlins_cache"
    key = _step("mutate", lambda s: s.get("id") == "key")["run"]
    for part in ("needs.plan.outputs.gremlins", "steps.py.outputs.python-version",
                 'tools/mutation_pilot.py config-hash "$MODULE"', "matrix.module"):
        assert part in key, part
    assert 'ch=$(python3 tools/mutation_pilot.py config-hash "$MODULE")' in key
    assert "gremlin-base-fp/tools/mutation_pilot.py" not in key  # config-hash must run against the PR's own checkout, never the base-commit worktree
    assert "hashFiles('tools/mutation_pilot.toml')" not in key
    # the FULL tracked-input fingerprint joins the namespace, assigned under
    # set -e rather than interpolated into the echo (where echo's own exit
    # status would mask a failed digest and a truncated key go live)
    assert "set -euo pipefail" in key
    assert 'fp=$(python3 tools/gremlin_pilot.py fingerprint "$MODULE")' in key
    prefix_line = next(ln.strip() for ln in key.splitlines() if "prefix=" in ln)
    assert "${fp}" in prefix_line and "${ch}" in prefix_line and "$(" not in prefix_line
    restore = _step("mutate", lambda s: s.get("uses", "").startswith("actions/cache/restore"))
    save = _step("mutate", lambda s: s.get("uses", "").startswith("actions/cache/save"))
    prefix = "${{ steps.key.outputs.prefix }}"
    exact = prefix + "${{ github.sha }}-${{ github.run_id }}-${{ github.run_attempt }}"
    assert restore["with"]["key"] == exact  # version+python+TOML+module+fingerprint...
    assert save["with"]["key"] == exact  # ...the SAME full namespace on both sides
    assert restore["with"]["restore-keys"] == prefix  # the same key without the sha
    assert restore["if"] == "needs.plan.outputs.mode == 'incremental'"  # full: no restore
    assert "always()" in save["if"]
    paths = restore["with"]["path"].splitlines()
    assert paths == save["with"]["path"].splitlines()
    assert paths == ["${{ env.GREMLINS_CACHE }}"]  # the gremlins cache directory only
    # a full run starts from scratch instead: --fresh is what clears .gremlins_cache
    run = _step("mutate", lambda s: s.get("id") == "run")["run"]
    assert 'if [ "$MODE" = "full" ]; then FRESH="--fresh"; fi' in run
    assert 'gremlin_pilot.py run "$MODULE" $FRESH' in run


def test_workflow_is_report_only():
    run = _step("mutate", lambda s: s.get("id") == "run")
    assert "tools/gremlin_pilot.py run" in run["run"]
    assert "set +e" in run["run"] and "gremlin-rc" in run["run"]
    gate = _step("mutate", lambda s: s.get("name", "").startswith("Fail only on a tool error"))
    assert 'rc" != 0' in gate["run"] and "score" not in gate["run"]
    # scores/survivors never fail a job; a TOOL failure always does: a nonzero
    # pytest/gremlins rc, a missing/malformed current raw JSON, or an ERROR
    # result -- all gate through the export step, whose rc read defaults to -1
    # so even a timeout-killed run (no rc file) fails rather than passing empty.
    export = _step("mutate", lambda s: s.get("name") == "Export report")["run"]
    assert "tools/gremlin_results.py export" in export
    assert '--run-exit-code "$rc"' in export
    assert 'cat "$RUNNER_TEMP/gremlin-rc" 2>/dev/null || echo -1' in export
    uploads = [s for j in JOBS.values() for s in j["steps"]
               if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    assert {u["with"]["retention-days"] for u in uploads} == {90}
    assert {u["with"]["name"] for u in uploads} == {"gremlin-raw-${{ matrix.module }}",
                                                    "mutation-module-${{ matrix.module }}",
                                                    "mutation-report"}
    assert all("always()" in u["if"] for u in uploads)  # raw + module uploads survive failures
    assert "always()" in JOBS["report"]["if"]
    download = _step("report", lambda s: str(s.get("uses", "")).startswith("actions/download"))
    assert download["with"]["pattern"] == "mutation-module-*"  # never the merged artifact
    assert tomllib.loads((ROOT / "tools" / "mutation_pilot.toml").read_text())  # parses


def test_workflow_run_gate_requires_success_and_a_same_repo_pr():
    """The plan job's own `if:` is what stops a PR's mutation matrix from
    starting before its Tests run has succeeded -- and skips it outright for
    a fork PR (github.event.workflow_run.pull_requests is empty there) or a
    workflow_run from a non-PR Tests run (push/schedule/dispatch), rather
    than erroring on an unsafe pull_requests[0] index."""
    gate = JOBS["plan"]["if"]
    assert "github.event_name != 'workflow_run'" in gate
    assert "github.event.workflow_run.conclusion == 'success'" in gate
    assert "github.event.workflow_run.event == 'pull_request'" in gate
    assert "join(github.event.workflow_run.pull_requests.*.number, ',') != ''" in gate
    assert "pull_requests[0]" not in gate
    plan_checkout = JOBS["plan"]["steps"][0]
    assert plan_checkout["with"]["ref"] == \
        "${{ github.event_name == 'workflow_run' && github.event.workflow_run.head_sha || '' }}"
    # a skipped plan (fork PR / unsuccessful or non-PR workflow_run) must not
    # let mutate start anyway: outputs.modules is unset, not '[]', on a
    # skipped job, so the guard checks needs.plan.result first.
    assert JOBS["mutate"]["if"] == "needs.plan.result == 'success' && needs.plan.outputs.modules != '[]'"
    mutate_checkout = JOBS["mutate"]["steps"][0]
    assert mutate_checkout["with"]["ref"] == \
        "${{ github.event_name == 'workflow_run' && github.event.workflow_run.head_sha || '' }}"


def test_workflow_run_recovers_pr_base_sha_without_indexing_pull_requests_zero():
    key_step = _step("mutate", lambda s: s.get("id") == "key")
    key_env = key_step["env"]
    assert key_env["EVENT"] == \
        "${{ github.event_name == 'workflow_run' && 'pull_request' || github.event_name }}"
    assert key_env["BASE_SHA"] == (
        "${{ github.event_name == 'workflow_run' && "
        "join(github.event.workflow_run.pull_requests.*.base.sha, ',') || "
        "github.event.pull_request.base.sha }}"
    )
    assert "pull_requests[0]" not in key_env["BASE_SHA"]
    # a PR run must never overwrite main's incremental cache: the save step
    # is skipped whenever the event is workflow_run (which, by the plan
    # job's own gate, is only ever a PR run for this workflow).
    save = _step("mutate", lambda s: s.get("name") == "Save gremlins cache")
    assert save["if"] == \
        "always() && steps.key.outcome == 'success' && github.event_name != 'workflow_run'"
    run_step = _step("mutate", lambda s: s.get("id") == "run")
    assert run_step["timeout-minutes"] == "${{ github.event_name == 'workflow_run' && 60 || 330 }}"
    export_env = _step("mutate", lambda s: s.get("name") == "Export report")["env"]
    assert export_env["BEFORE"] == (
        "${{ github.event_name == 'push' && github.event.before || "
        "(github.event_name == 'workflow_run' && "
        "join(github.event.workflow_run.pull_requests.*.base.sha, ',') || '') }}"
    )


# -- dual CI: the independent mutmut workflow (mutation-mutmut.yml) --------------------
#
# Both backends must keep running with their DISTINCT results: separate
# concurrency groups, cache namespaces, module artifacts and aggregate
# artifacts, so neither workflow can block, overwrite or silently merge into
# the other. Same contract as the gremlins workflow: push -> incremental,
# weekly/dispatch -> full, scores report-only, tool errors fail, and the
# aggregate refuses to publish anything but the planned module set as complete.

MUT_YML = ROOT / ".github" / "workflows" / "mutation-mutmut.yml"
MUTMUT = yaml.safe_load(MUT_YML.read_text())
MUT_JOBS = MUTMUT["jobs"]


def test_mutmut_workflow_triggers_modes_and_a_separate_concurrency_group():
    on = MUTMUT.get("on", MUTMUT.get(True))  # PyYAML reads a bare `on` as True
    assert on["push"]["branches"] == ["main"]
    assert "pull_request" not in on  # PR runs are gated behind Tests via workflow_run instead
    assert on["workflow_run"] == {"workflows": ["Tests"], "types": ["completed"]}
    assert on["schedule"] and "cron" in on["schedule"][0]
    assert set(on["workflow_dispatch"]["inputs"]) == {"fresh", "modules"}
    assert MUTMUT["concurrency"]["group"] == (
        "mutation-mutmut-${{ github.event_name == 'workflow_run' "
        "&& github.event.workflow_run.head_branch || github.ref }}"
    )
    assert MUTMUT["concurrency"]["cancel-in-progress"] == \
        "${{ github.event_name == 'workflow_run' || github.event_name == 'push' }}"
    gremlin_on = WORKFLOW.get("on", WORKFLOW.get(True))
    assert MUTMUT["concurrency"]["group"] != WORKFLOW["concurrency"]["group"]
    assert MUTMUT["permissions"] == {"contents": "read"}
    # weekly FULL on both, staggered so the two full runs do not queue at once
    mut_cron = on["schedule"][0]["cron"].split()
    gre_cron = gremlin_on["schedule"][0]["cron"].split()
    assert mut_cron != gre_cron and mut_cron[2:] == gre_cron[2:]  # same day, other time
    plan = MUT_JOBS["plan"]["steps"][-1]["run"]
    assert '"$EVENT" = "push"' in plan and '"$FRESH" = "false"' in plan and "mode=full" in plan


def test_mutmut_plan_checks_out_full_history_only_for_pull_request():
    checkout = MUT_JOBS["plan"]["steps"][0]
    assert checkout["uses"] == "actions/checkout@v4"
    assert checkout["with"]["fetch-depth"] == \
        "${{ github.event_name == 'workflow_run' && '0' || 1 }}"


def test_mutmut_plan_narrows_the_matrix_on_pull_request_via_changed_files():
    plan_step = MUT_JOBS["plan"]["steps"][-1]
    assert plan_step["env"]["BASE_SHA"] == (
        "${{ github.event_name == 'workflow_run' && "
        "join(github.event.workflow_run.pull_requests.*.base.sha, ',') || "
        "github.event.pull_request.base.sha }}"
    )
    plan = plan_step["run"]
    assert 'CHANGED_ARGS=""' in plan
    assert 'if [ "$EVENT" = "pull_request" ]; then' in plan
    assert 'git diff -z --no-renames --name-only "$BASE_SHA"...HEAD' in plan
    assert 'CHANGED_ARGS="--changed-files' in plan
    assert 'modules=$(python3 tools/mutation_pilot.py matrix --only "$ONLY" $CHANGED_ARGS)' in plan


def test_mutmut_plan_step_uses_the_mutmut_driver_and_gates_its_own_failure():
    plan = MUT_JOBS["plan"]["steps"][-1]["run"]
    assert "set -euo pipefail" in plan
    assert 'modules=$(python3 tools/mutation_pilot.py matrix --only "$ONLY" $CHANGED_ARGS)' in plan
    assert "gremlin_pilot" not in plan  # the mutmut runner, not the gremlins one
    # the list is assigned, then echoed by name: an interpolated command
    # substitution inside the echo would hide its failure behind echo's rc
    assert 'echo "modules=$modules"' in plan
    assert 'mutmut=$(sed -n' in plan and 'echo "mutmut=$(' not in plan
    # matrix --only refuses a names-nothing value (below), so a mistyped subset
    # aborts the plan instead of publishing the empty matrix.
    assert MUT_JOBS["mutate"]["if"] == \
        "needs.plan.result == 'success' && needs.plan.outputs.modules != '[]'"


def test_mutmut_matrix_refuses_a_names_nothing_subset_not_an_empty_matrix(capsys):
    enabled = pilot.enabled_modules(CFG)

    class Args:
        only = ","
    with pytest.raises(SystemExit):  # GitHub renders a dispatched empty input as ""
        pilot.cmd_matrix(CFG, Args)
    Args.only = " , , "
    with pytest.raises(SystemExit):
        pilot.cmd_matrix(CFG, Args)
    Args.only = "   "  # blank -- and only blank -- still means every enabled module
    pilot.cmd_matrix(CFG, Args)
    assert json.loads(capsys.readouterr().out) == enabled


def test_mutmut_workflow_caches_only_mutmut_state_in_its_own_namespace():
    key = _step("mutate", lambda s: s.get("id") == "key", MUT_JOBS)["run"]
    for part in ("needs.plan.outputs.mutmut", "steps.py.outputs.python-version",
                 'tools/mutation_pilot.py config-hash "$MODULE"', "matrix.module"):
        assert part in key, part
    assert 'ch=$(python3 tools/mutation_pilot.py config-hash "$MODULE")' in key
    assert "hashFiles('tools/mutation_pilot.toml')" not in key
    assert "mutation-mutmut" in key
    restore = _step("mutate", lambda s: str(s.get("uses", "")).startswith("actions/cache/restore"),
                    MUT_JOBS)
    save = _step("mutate", lambda s: str(s.get("uses", "")).startswith("actions/cache/save"),
                 MUT_JOBS)
    prefix = "${{ steps.key.outputs.prefix }}"
    exact = prefix + "${{ github.sha }}-${{ github.run_id }}-${{ github.run_attempt }}"
    assert restore["with"]["key"] == exact == save["with"]["key"]
    assert restore["with"]["restore-keys"] == prefix
    assert restore["if"] == "needs.plan.outputs.mode == 'incremental'"  # full: no restore
    assert "always()" in save["if"]
    paths = restore["with"]["path"].splitlines()
    assert paths == save["with"]["path"].splitlines()
    assert all("STATE" in p for p in paths)  # mutmut's work-copy state only...
    assert not any("gremlins" in p or "GREMLINS" in p for p in paths)  # ...never the .gremlins_cache
    # the gremlins key prefix cannot collide: different version source AND name
    gre_key = _step("mutate", lambda s: s.get("id") == "key")["run"]
    assert "mutation-gremlins" in gre_key and "mutation-mutmut${{ needs.plan.outputs.mutmut }}" in key


def test_mutmut_workflow_is_report_only():
    run = _step("mutate", lambda s: s.get("id") == "run", MUT_JOBS)
    assert "tools/mutation_pilot.py run" in run["run"] and "--max-children" in run["run"]
    assert "set +e" in run["run"] and "mutation-rc" in run["run"]
    # per-push runs are time-boxed to 60 minutes (52-minute driver budget) and
    # the weekly full run keeps its 330-minute step limit (322-minute budget)
    assert run["timeout-minutes"] == \
        "${{ needs.plan.outputs.mode == 'incremental' && 60 || 330 }}"
    assert ('if [ "$MODE" = "incremental" ]; then BUDGET=$((52 * 60)); '
            'else BUDGET=$((322 * 60)); fi') in run["run"]
    assert '--time-budget-seconds "$BUDGET"' in run["run"]
    gate = _step("mutate", lambda s: s.get("name", "").startswith("Fail only on a tool error"),
                 MUT_JOBS)
    assert 'rc" != 0' in gate["run"] and "score" not in gate["run"]
    # the incremental time-box stop (124) is INCOMPLETE, not a tool error: it is
    # exempted from the gate; every other nonzero rc -- and every 124 in full
    # mode, where there is no time box to exempt it -- still fails
    assert '[ "$rc" = "124" ] && [ "$MODE" = "incremental" ]' in gate["run"]
    assert "INCOMPLETE" in gate["run"]
    assert "exit 0" in gate["run"]
    # a timeout-killed run leaves no rc file: the export gate reads -1, never a
    # silent pass, while survivors/scores never exit nonzero.
    export = _step("mutate", lambda s: s.get("name") == "Export report", MUT_JOBS)["run"]
    assert "tools/mutation_results.py export" in export
    assert '--run-exit-code "$rc"' in export
    assert 'cat "$RUNNER_TEMP/mutation-rc" 2>/dev/null || echo -1' in export
    uploads = [s for j in MUT_JOBS.values() for s in j["steps"]
               if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    assert {u["with"]["retention-days"] for u in uploads} == {90}
    assert {u["with"]["name"] for u in uploads} == {"mutation-mutmut-module-${{ matrix.module }}",
                                                    "mutation-mutmut-report"}
    assert all("always()" in u["if"] for u in uploads)


def test_mutmut_workflow_run_gate_and_pr_context_mirror_the_gremlins_workflow():
    """Same rationale as the gremlins workflow's own
    test_workflow_run_gate_requires_success_and_a_same_repo_pr -- see there
    for why. This backend's plan job carries an identical gate."""
    gate = MUT_JOBS["plan"]["if"]
    assert "github.event_name != 'workflow_run'" in gate
    assert "github.event.workflow_run.conclusion == 'success'" in gate
    assert "github.event.workflow_run.event == 'pull_request'" in gate
    assert "join(github.event.workflow_run.pull_requests.*.number, ',') != ''" in gate
    assert "pull_requests[0]" not in gate
    plan_checkout = MUT_JOBS["plan"]["steps"][0]
    assert plan_checkout["with"]["ref"] == \
        "${{ github.event_name == 'workflow_run' && github.event.workflow_run.head_sha || '' }}"
    assert MUT_JOBS["mutate"]["if"] == \
        "needs.plan.result == 'success' && needs.plan.outputs.modules != '[]'"
    mutate_checkout = MUT_JOBS["mutate"]["steps"][0]
    assert mutate_checkout["with"]["ref"] == \
        "${{ github.event_name == 'workflow_run' && github.event.workflow_run.head_sha || '' }}"


def test_mutmut_workflow_run_never_saves_state_or_reads_pull_requests_zero():
    save = _step("mutate", lambda s: s.get("name") == "Save mutmut state", MUT_JOBS)
    assert save["if"] == \
        "always() && steps.key.outcome == 'success' && github.event_name != 'workflow_run'"
    export_env = _step("mutate", lambda s: s.get("name") == "Export report", MUT_JOBS)["env"]
    assert export_env["BEFORE"] == (
        "${{ github.event_name == 'push' && github.event.before || "
        "(github.event_name == 'workflow_run' && "
        "join(github.event.workflow_run.pull_requests.*.base.sha, ',') || '') }}"
    )
    assert "pull_requests[0]" not in export_env["BEFORE"]


def test_mutmut_report_job_gates_the_merge_on_the_planned_module_set():
    assert MUT_JOBS["report"]["env"]["EXPECTED_MODULES"] == "${{ needs.plan.outputs.modules }}"
    download = _step("report", lambda s: str(s.get("uses", "")).startswith("actions/download"),
                     MUT_JOBS)
    assert download["with"]["pattern"] == "mutation-mutmut-module-*"  # never the merged one
    merge = _step("report", lambda s: s.get("name") == "Merge module reports", MUT_JOBS)["run"]
    assert "tools/mutation_results.py merge" in merge
    assert '--expected-modules "$EXPECTED_MODULES"' in merge
    assert 'echo "$?" > "$RUNNER_TEMP/mutmut-merge-rc"' in merge  # rc captured, not short-circuited
    assert "set +e" in merge and "set -e" in merge
    # the diagnostic artifact must survive the nonzero merge rc: uploaded with
    # always(), and only the LAST step fails the job on that rc.
    steps = MUT_JOBS["report"]["steps"]
    upload = next(s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact"))
    gate = steps[-1]
    assert "always()" in upload["if"] and upload["with"]["name"] == "mutation-mutmut-report"
    assert gate["name"].startswith("Fail only on a merge error") and 'rc" != 0' in gate["run"]
    assert "always()" in gate["if"]


def test_the_two_workflows_own_disjoint_artifact_names():
    """One workflow can never download, merge or overwrite the other's output."""
    def upload_names(jobs):
        return {s["with"]["name"] for j in jobs.values() for s in j["steps"]
                if str(s.get("uses", "")).startswith("actions/upload-artifact")}
    gre_uploads, mut_uploads = upload_names(JOBS), upload_names(MUT_JOBS)
    assert gre_uploads & mut_uploads == set()  # mutation-report != mutation-mutmut-report
    gre_pattern = _step("report", lambda s: str(s.get("uses", "")).startswith("actions/download"))
    mut_pattern = _step("report", lambda s: str(s.get("uses", "")).startswith("actions/download"),
                        MUT_JOBS)
    assert gre_pattern["with"]["pattern"] != mut_pattern["with"]["pattern"]
    assert WORKFLOW["name"] != MUTMUT["name"]


def test_both_mutation_matrices_cap_parallelism_so_tests_never_starve():
    """A public-repo free-plan account gets 20 concurrent Actions runners
    total. Each backend's matrix has 30+ per-module jobs with no
    max-parallel, so one cold run of either workflow can occupy every
    runner and every PR's `test` job queues behind it. Both matrices cap at
    3: with two PRs open at once (each capable of running both workflows),
    that is at most 2 x 2 x 3 = 12 mutation runners account-wide, leaving
    >= 8 free for Tests/plan/report."""
    assert JOBS["mutate"]["strategy"]["max-parallel"] == 3
    assert MUT_JOBS["mutate"]["strategy"]["max-parallel"] == 3
    assert JOBS["mutate"]["strategy"]["fail-fast"] is False
    assert MUT_JOBS["mutate"]["strategy"]["fail-fast"] is False
    total = JOBS["mutate"]["strategy"]["max-parallel"] + \
        MUT_JOBS["mutate"]["strategy"]["max-parallel"]
    assert total <= 6


# -- cancel-stale-runs.yml: no leftover Tests/mutation/mutation-mutmut runs ------------

CANCEL_YML = ROOT / ".github" / "workflows" / "cancel-stale-runs.yml"
CANCEL = yaml.safe_load(CANCEL_YML.read_text())
CANCEL_JOBS = CANCEL["jobs"]


def test_cancel_stale_runs_triggers_on_pr_close_with_actions_write():
    on = CANCEL.get("on", CANCEL.get(True))  # PyYAML reads a bare `on` as True
    assert on["pull_request"]["types"] == ["closed"]
    assert CANCEL["permissions"] == {"actions": "write"}


def test_cancel_stale_runs_is_guarded_to_same_repo_prs():
    # A fork PR's GITHUB_TOKEN is forced read-only for pull_request-triggered
    # workflows no matter what `permissions:` asks for, so the cancel calls
    # could never succeed there; this guard skips the job outright instead
    # of leaving a noisy failed run.
    assert CANCEL_JOBS["cancel"]["if"] == \
        "github.event.pull_request.head.repo.full_name == github.repository"


def test_cancel_stale_runs_covers_all_three_workflows_by_branch_or_pr_number():
    step = CANCEL_JOBS["cancel"]["steps"][0]
    run = step["run"]
    for wf in ("Tests", "mutation", "mutation-mutmut"):
        assert wf in run
    assert "in_progress" in run and "queued" in run
    assert "actions/runs/$id/cancel" in run
    assert "PR_BRANCH" in step["env"] and "PR_NUMBER" in step["env"]
    assert step["env"]["PR_BRANCH"] == "${{ github.event.pull_request.head.ref }}"
    assert step["env"]["PR_NUMBER"] == "${{ github.event.pull_request.number }}"
    # matches by head_branch (covers Tests, a direct pull_request trigger) OR
    # by the run's own linked pull_requests[].number (covers mutation/
    # mutation-mutmut, triggered via workflow_run) -- `// []` so an empty
    # array (a fork PR) is a plain no-match, not a jq error.
    assert ".head_branch ==" in run
    assert "pull_requests // []" in run
    assert "any(.number ==" in run


# -- merge expected-modules contract: the aggregate can never lie about scope ----------

def _module_artifact(tmp_path: Path, name: str, statuses, info=None, tag=None, run_exit_code=None):
    """A directory in the shape the mutate job uploads (summary.json + results)."""
    info = info or INFO
    rows = [_row(name, "a.py", "f", s, name=f"{tag or name}{i}") for i, s in enumerate(statuses)]
    d = tmp_path / (tag or name)
    d.mkdir()
    mr.write_jsonl(d / "results.jsonl", rows)
    (d / "summary.json").write_text(json.dumps(
        mr.summarize(rows, name, info, ["a.py"], run_exit_code=run_exit_code)))
    return d


def test_merge_with_the_exact_planned_set_is_a_valid_run(tmp_path):
    a = _module_artifact(tmp_path, "a", ["killed", "survived"])
    b = _module_artifact(tmp_path, "b", ["killed", "timeout"])
    merged = mr.merge_dirs([a, b], tmp_path / "m", ["a", "b"])
    assert merged["complete"] and not merged["tool_error"] and merged["failure_reasons"] == []
    assert merged["score"] == 0.75 and merged["total"] == 4
    assert merged["expected_modules"] == ["a", "b"]
    assert merged["module_contract"] == {"expected": ["a", "b"], "present": ["a", "b"],
                                         "missing": [], "unexpected": [], "duplicate": {},
                                         "complete_set": True}


def test_merge_missing_a_planned_module_is_never_a_clean_report(tmp_path):
    """The whole point: a module whose job died before uploading must not let a
    subset publish as the latest completed run."""
    a = _module_artifact(tmp_path, "a", ["killed", "killed"])
    merged = mr.merge_dirs([a], tmp_path / "m", ["a", "b", "c"])
    assert not merged["complete"] and merged["tool_error"]
    assert merged["score"] is None  # withheld: no input measurement made this number
    assert merged["total"] == 2  # counts stay auditable for what really arrived
    assert any(r.startswith("MISSING_MODULES") and "b, c" in r for r in merged["failure_reasons"])
    assert merged["module_contract"]["missing"] == ["b", "c"]
    md = (tmp_path / "m" / "summary.md").read_text()
    assert "NOT the set the plan expected" in md and "MISSING_MODULES" in md


def test_merge_reports_a_module_the_plan_never_scheduled(tmp_path):
    a = _module_artifact(tmp_path, "a", ["killed"])
    x = _module_artifact(tmp_path, "stale", ["survived"])  # e.g. an old artifact name
    merged = mr.merge_dirs([a, x], tmp_path / "m", ["a"])
    assert merged["tool_error"] and merged["score"] is None
    assert any(r.startswith("UNEXPECTED_MODULES") and "stale" in r for r in merged["failure_reasons"])
    assert merged["module_contract"]["unexpected"] == ["stale"]


def test_merge_with_no_module_directories_writes_a_null_diagnostic(tmp_path):
    merged = mr.merge_dirs([], tmp_path / "m", ["a", "b"])
    assert merged["tool_error"] and not merged["complete"]
    assert merged["total"] is None and merged["score"] is None  # null, never a fabricated zero
    assert any(r.startswith("NO_MODULE_REPORTS") for r in merged["failure_reasons"])
    assert any(r.startswith("MISSING_MODULES") for r in merged["failure_reasons"])


def test_merge_duplicate_module_report_against_the_contract_is_a_diagnostic(tmp_path):
    a1 = _module_artifact(tmp_path, "a", ["killed"], tag="a1")
    a2 = _module_artifact(tmp_path, "a", ["killed", "killed"], tag="a2")
    merged = mr.merge_dirs([a1, a2], tmp_path / "m", ["a"])
    assert merged["tool_error"] and merged["total"] is None  # no honest aggregate exists
    assert "a" not in merged["modules"]  # not resolved to whichever dir read first
    assert any(r.startswith("DUPLICATE_MODULES") for r in merged["failure_reasons"])
    assert set(merged["module_contract"]["duplicate"]["a"]) == {str(a1), str(a2)}


def test_merge_mixed_provenance_is_flagged_under_the_contract(tmp_path):
    a = _module_artifact(tmp_path, "a", ["killed"])
    b = _module_artifact(tmp_path, "b", ["killed"], info=dict(INFO, run_id="8"))
    merged = mr.merge_dirs([a, b], tmp_path / "m", ["a", "b"])
    assert merged["tool_error"] and merged["score"] is None
    assert any(r.startswith("RUN_PROVENANCE_MISMATCH") and "run_ids" in r
               for r in merged["failure_reasons"])


# A module whose mutmut run exited nonzero checked only part of its mutants: its
# report is complete-looking, so the merge must catch the recorded exit code or
# the aggregate would publish a truncated run as the latest completed full score
# with a clean exit. The per-push time box is the one exception: an incremental
# run stopped by its own budget exits 124 (TIME_BUDGET_STOP_RC), which is
# INCOMPLETE, not a tool error; the same 124 in a full run still fails; and -1
# (no exit code recorded, or a step-timeout kill) is a tool error in every mode.

def test_merge_propagates_a_failed_module_run_as_an_incomplete_tool_error(tmp_path):
    a = _module_artifact(tmp_path, "a", ["killed", "survived"])
    b = _module_artifact(tmp_path, "b", ["killed"], run_exit_code=2)
    merged = mr.merge_dirs([a, b], tmp_path / "m", ["a", "b"])
    assert not merged["complete"] and merged["tool_error"]
    assert merged["score"] is None  # a partial run never carries a clean aggregate score
    assert merged["total"] == 3  # counts stay auditable for what really arrived
    assert merged["failed_run_modules"] == {"b": [2]}  # the affected module, by name
    assert any(r.startswith("RUN_INCOMPLETE") and r.split(":")[1].strip().startswith("b's")
               for r in merged["failure_reasons"]), merged["failure_reasons"]
    md = (tmp_path / "m" / "summary.md").read_text()
    assert "RUN_INCOMPLETE" in md and "b" in md and "INCOMPLETE" in md
    # and the CLI -- the workflow's gate on its rc -- must not read it as success
    assert mr.main(["merge", "--out", str(tmp_path / "m2"), "--expected-modules", '["a", "b"]',
                    str(a), str(b)]) == 1


def test_merge_an_incremental_time_box_stop_is_incomplete_but_not_a_tool_error(tmp_path):
    a = _module_artifact(tmp_path, "a", ["killed", "survived"])
    b = _module_artifact(tmp_path, "b", ["killed"], run_exit_code=124)  # INFO: incremental mode
    merged = mr.merge_dirs([a, b], tmp_path / "m", ["a", "b"])
    assert not merged["complete"] and not merged["tool_error"]
    assert merged["score"] is None  # still withheld: not a complete measurement
    assert merged["failed_run_modules"] == {"b": [124]}
    md = (tmp_path / "m" / "summary.md").read_text()
    assert "TIME_BUDGET_STOP" in md and "b" in md and "INCOMPLETE" in md
    assert "per-push time box" in md and "not a tool failure" in md
    # the merge CLI is the report job's gate: a time-box stop passes it
    assert mr.main(["merge", "--out", str(tmp_path / "m2"), "--expected-modules", '["a", "b"]',
                    str(a), str(b)]) == 0


def test_merge_a_time_box_stop_in_a_full_run_is_still_a_tool_error(tmp_path):
    full = dict(INFO, mode="full")
    a = _module_artifact(tmp_path, "a", ["killed", "survived"], info=full)
    b = _module_artifact(tmp_path, "b", ["killed"], info=full, run_exit_code=124)
    merged = mr.merge_dirs([a, b], tmp_path / "m", ["a", "b"])
    assert not merged["complete"] and merged["tool_error"]
    assert merged["score"] is None
    assert merged["failed_run_modules"] == {"b": [124]}
    assert any(r.startswith("TIME_BUDGET_STOP") for r in merged["failure_reasons"])
    assert mr.main(["merge", "--out", str(tmp_path / "m2"), "--expected-modules", '["a", "b"]',
                    str(a), str(b)]) == 1


def test_merge_a_missing_exit_code_in_an_incremental_run_is_a_tool_error(tmp_path):
    """-1 means the runner recorded no exit code (or the step was killed): never exempt."""
    a = _module_artifact(tmp_path, "a", ["killed", "survived"])
    b = _module_artifact(tmp_path, "b", ["killed"], run_exit_code=-1)  # INFO: incremental mode
    merged = mr.merge_dirs([a, b], tmp_path / "m", ["a", "b"])
    assert not merged["complete"] and merged["tool_error"]
    assert merged["score"] is None
    assert merged["failed_run_modules"] == {"b": [-1]}
    assert any(r.startswith("TIMEOUT_KILL") for r in merged["failure_reasons"]), merged["failure_reasons"]
    assert mr.main(["merge", "--out", str(tmp_path / "m2"), "--expected-modules", '["a", "b"]',
                    str(a), str(b)]) == 1


def test_merge_zero_and_absent_run_exit_codes_stay_a_valid_run(tmp_path):
    a = _module_artifact(tmp_path, "a", ["killed"], run_exit_code=0)
    b = _module_artifact(tmp_path, "b", ["killed"])  # export recorded no rc: local merge
    merged = mr.merge_dirs([a, b], tmp_path / "m", ["a", "b"])
    assert merged["complete"] and not merged["tool_error"]
    assert merged["failed_run_modules"] == {} and merged["failure_reasons"] == []
    assert merged["score"] == 1.0


def test_merge_timeout_and_a_broken_set_flag_both(tmp_path):
    """A missing module AND a timed-out one: every violation is named, not the
    first one found."""
    a = _module_artifact(tmp_path, "a", ["killed"], run_exit_code=-1)
    merged = mr.merge_dirs([a], tmp_path / "m", ["a", "b"])
    assert not merged["complete"] and merged["tool_error"] and merged["score"] is None
    assert merged["failed_run_modules"] == {"a": [-1]}
    prefixes = {r.split(":")[0] for r in merged["failure_reasons"]}
    assert {"MISSING_MODULES", "TIMEOUT_KILL"} <= prefixes


def test_merge_without_a_contract_keeps_the_historical_shape(tmp_path):
    a = _module_artifact(tmp_path, "a", ["killed", "survived"])
    b = _module_artifact(tmp_path, "b", ["killed"])
    merged = mr.merge_dirs([a, b], tmp_path / "m")
    assert {"complete", "tool_error", "failure_reasons", "module_contract",
            "expected_modules"} & set(merged) == set()
    assert merged["score"] == round(2 / 3, 4)
    with pytest.raises(mr.MergeError):  # no contract to attribute the ambiguity to
        mr.merge_dirs([a, _module_artifact(tmp_path, "a", ["killed"], tag="dup")], tmp_path / "m2")


@pytest.mark.parametrize("raw", ["", "not json", '{"a": 1}', '["a", "a"]', '"a"'])
def test_a_broken_contract_is_refused_not_treated_as_no_contract(raw):
    with pytest.raises(mr.MergeError):
        mr.parse_expected_modules(raw)
    assert mr.parse_expected_modules(None) is None


def test_merge_cli_exit_codes(tmp_path):
    a = _module_artifact(tmp_path, "a", ["killed"])
    b = _module_artifact(tmp_path, "b", ["killed"])
    assert mr.main(["merge", "--out", str(tmp_path / "x0"), "--expected-modules", '["a", "b"]',
                    str(a), str(b)]) == 0
    assert mr.main(["merge", "--out", str(tmp_path / "x1"), "--expected-modules", '["a", "b"]',
                    str(a)]) == 1  # incomplete diagnostic -> the report job fails
    assert mr.main(["merge", "--out", str(tmp_path / "x2"), "--expected-modules", "[]"]) == 1
    assert mr.main(["merge", "--out", str(tmp_path / "x3"), "--expected-modules", "broken"]) == 2


# -- mutation_report.py: select the mutmut workflow without moving the default ---------

def test_report_default_is_the_gremlins_workflow_and_artifact():
    assert rep.DEFAULT_BACKEND == "gremlins"
    assert rep.backend_sources("gremlins") == ("mutation.yml", "mutation-report")
    assert (rep.WORKFLOW, rep.ARTIFACT) == ("mutation.yml", "mutation-report")
    assert rep.backend_sources("mutmut") == ("mutation-mutmut.yml", "mutation-mutmut-report")
    with pytest.raises(SystemExit):
        rep.backend_sources("nonsense")


def test_report_cli_backend_selects_the_workflow_and_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("MUTATION_REPORT_CACHE", str(tmp_path / "cache"))
    calls = []

    def fake_gh(args, repo):
        calls.append(["gh", *args])
        workflow = args[args.index("--workflow") + 1]
        # distinct run ids, so the gremlins fetch never answers from the mutmut
        # run's cache directory (and vice versa)
        return [{"databaseId": 42 if workflow == "mutation.yml" else 43}]

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        dest = Path(cmd[cmd.index("-D") + 1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "summary.json").write_text("{}")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rep, "gh_json", fake_gh)
    monkeypatch.setattr(rep.subprocess, "run", fake_run)
    monkeypatch.setattr(mr, "read_jsonl", lambda p: [])

    assert rep.main([]) == 0  # default: unchanged gremlins behavior
    assert calls[0][calls[0].index("--workflow") + 1] == "mutation.yml"
    assert calls[1][calls[1].index("-n") + 1] == "mutation-report"
    calls.clear()
    assert rep.main(["--backend", "mutmut"]) == 0
    assert calls[0][calls[0].index("--workflow") + 1] == "mutation-mutmut.yml"
    assert calls[1][calls[1].index("-n") + 1] == "mutation-mutmut-report"
    calls.clear()
    assert rep.main(["--backend", "mutmut", "--history", "1"]) == 0  # trend follows the backend
    assert calls and calls[0][calls[0].index("--workflow") + 1] == "mutation-mutmut.yml"


# -- fetch_run cache: keyed by run ID AND artifact, never by run ID alone --------------

def _fake_download(present):
    """A fake ``gh run download`` that serves only the artifacts named in
    ``present`` (writing the artifact's own name as the summary), recording
    every invocation."""
    calls = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        artifact = cmd[cmd.index("-n") + 1]
        dest = Path(cmd[cmd.index("-D") + 1])
        if artifact not in present:
            return types.SimpleNamespace(returncode=1, stdout="",
                                         stderr=f"{artifact} not found")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "summary.json").write_text(artifact)
        (dest / "results.jsonl").write_text("")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    return calls, fake_run


def test_fetch_run_caches_per_artifact_so_two_backends_cannot_share_a_run_id(monkeypatch,
                                                                              tmp_path):
    """GitHub run ids are shared between the two workflows: asking the same run
    for the other backend must fetch that backend's own artifact, not read the
    first backend's cached copy."""
    monkeypatch.setenv("MUTATION_REPORT_CACHE", str(tmp_path / "cache"))
    calls, fake_run = _fake_download({"mutation-report", "mutation-mutmut-report"})
    monkeypatch.setattr(rep.subprocess, "run", fake_run)
    gre = rep.fetch_run("123", None, "mutation-report")
    mut = rep.fetch_run("123", None, "mutation-mutmut-report")
    assert gre is not None and mut is not None and gre != mut
    assert gre.name == "mutation-report" and mut.name == "mutation-mutmut-report"
    assert (gre / "summary.json").read_text() == "mutation-report"  # its own artifact
    assert (mut / "summary.json").read_text() == "mutation-mutmut-report"
    assert rep.fetch_run("123", None, "mutation-report") == gre
    assert rep.fetch_run("123", None, "mutation-mutmut-report") == mut
    assert len(calls) == 2  # both cached hits, neither served the other's download


def test_fetch_run_absent_backend_artifact_is_never_answered_from_the_other(monkeypatch,
                                                                            tmp_path):
    """A run of one workflow has no artifact for the other: the request stays
    a miss (None) instead of being shadowed by the cached sibling."""
    monkeypatch.setenv("MUTATION_REPORT_CACHE", str(tmp_path / "cache"))
    calls, fake_run = _fake_download({"mutation-report"})
    monkeypatch.setattr(rep.subprocess, "run", fake_run)
    assert rep.fetch_run("123", None, "mutation-report") is not None
    assert rep.fetch_run("123", None, "mutation-mutmut-report") is None
    assert len(calls) == 2  # a real download attempt, failed -- not a stale cache hit


def test_fetch_run_cache_hit_requires_a_matching_artifact_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("MUTATION_REPORT_CACHE", str(tmp_path / "cache"))
    dest = tmp_path / "cache" / "123" / "mutation-mutmut-report"
    dest.mkdir(parents=True)
    (dest / "summary.json").write_text("{}")  # a markerless entry is never trusted
    calls, fake_run = _fake_download({"mutation-mutmut-report"})
    monkeypatch.setattr(rep.subprocess, "run", fake_run)
    found = rep.fetch_run("123", None, "mutation-mutmut-report")
    assert len(calls) == 1 and found == dest
    assert (found / ".artifact").read_text().strip() == "mutation-mutmut-report"
    assert rep.fetch_run("123", None, "mutation-mutmut-report") == dest
    assert len(calls) == 1  # now a genuine cache hit


def test_report_cli_two_backends_same_run_id_read_their_own_downloads(monkeypatch, tmp_path):
    monkeypatch.setenv("MUTATION_REPORT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(rep, "gh_json", lambda args, repo: [{"databaseId": 123}])
    calls, fake_run = _fake_download({"mutation-report"})  # run 123 is a gremlins run
    seen = []
    monkeypatch.setattr(rep.subprocess, "run", fake_run)
    monkeypatch.setattr(mr, "read_jsonl", lambda p: seen.append(Path(p).parent.name) or [])
    assert rep.main([]) == 0
    assert seen == ["mutation-report"]
    with pytest.raises(SystemExit):  # the cached gremlins copy must not answer here
        rep.main(["--backend", "mutmut", "--run", "123"])
    assert seen == ["mutation-report"] and calls[-1][calls[-1].index("-n") + 1] == \
        "mutation-mutmut-report"


# -- PR module selection: module ownership, else the inert allowlist, else --
# -- every enabled module (never zero for an unrecognized path) --------------
#
# The rule `changed_modules` implements (see its docstring in
# tools/mutation_pilot.py for the full ordering): a changed path selects the
# ENABLED module that owns it (`module_owns_changed_path`); failing that, a
# path on the small docs-only `[pr_selection] inert` allowlist selects
# nothing; any other path selects EVERY requested module, immediately, for
# the whole changed-file list. An EXCLUDED module's ownership does not count
# in the first step -- it never runs, so a path only it claims is exactly as
# unrecognized as one no module claims.

def _sel_cfg():
    return {
        "pr_selection": {"inert": ["*.md", "docs/*", "guides/*"]},
        "defaults": {},
        "modules": {
            "alpha": {"why": "x", "mutate": ["engine/a.py"], "tests": ["tests/test_a.py"]},
            "beta": {"why": "x", "mutate": ["engine/b.py"], "tests": ["tests/test_b.py"]},
        },
    }


@pytest.fixture(autouse=True)
def _empty_import_graph_for_synthetic_cfg(request, monkeypatch):
    """Every test marked `synthetic_cfg` builds a throwaway cfg via
    `_sel_cfg()`/`_defect_list_cfg()` whose `mutate`/`tests` paths
    (engine/a.py, tests/test_b.py, ...) never exist as real tracked files --
    so a call to `changed_modules` without an explicit `graph=` kwarg was
    ALREADY getting an empty dependency closure for every one of those
    modules (`module_dependency_closure`'s `_closure_roots` never matches a
    fictional path against the real tracked set), just after silently
    building the whole real ~884-file graph first to get there. This
    fixture skips straight to that same empty closure by patching
    `build_import_graph` to return `{}` -- no assertion's outcome changes,
    only the wasted real-repo parse does. Tests against the real `CFG`
    (test_real_toml_*, the artifacts/frozen_inputs/research-replay closure
    tests, ...) are never marked `synthetic_cfg` and always build the real
    graph."""
    if request.node.get_closest_marker("synthetic_cfg"):
        monkeypatch.setattr(pilot, "build_import_graph", lambda *a, **k: {})


def test_read_changed_files_strips_blanks_and_refuses_a_bad_path(tmp_path):
    assert pilot.read_changed_files("") == []
    with pytest.raises(SystemExit):  # missing: an operator/workflow bug, not "no changes"
        pilot.read_changed_files(str(tmp_path / "nope.txt"))
    with pytest.raises(SystemExit):  # a directory is not a valid --changed-files path either
        pilot.read_changed_files(str(tmp_path))
    f = tmp_path / "changed.txt"
    # NUL-delimited, as `git diff -z --name-only` writes it; a doubled NUL
    # (as a trailing separator would produce) must not yield a blank entry.
    f.write_bytes(b"engine/a.py\0\0tests/test_b.py\0")
    assert pilot.read_changed_files(str(f)) == ["engine/a.py", "tests/test_b.py"]


@pytest.mark.synthetic_cfg
def test_changed_modules_selects_only_the_owning_module():
    cfg2 = _sel_cfg()
    assert pilot.changed_modules(cfg2, ["alpha", "beta"], ["engine/a.py"]) == ["alpha"]
    assert pilot.changed_modules(cfg2, ["alpha", "beta"], ["tests/test_b.py"]) == ["beta"]


@pytest.mark.synthetic_cfg
def test_changed_modules_a_deleted_own_test_file_still_selects_its_module():
    # alpha's test file, tests/test_a.py, was deleted by this PR: git diff
    # --name-only reports it in `changed`, but module_owns_changed_path
    # matches the bare path against alpha's OWN configured patterns, never
    # against a tracked-file list, so the deletion changes nothing.
    cfg2 = _sel_cfg()
    assert pilot.changed_modules(cfg2, ["alpha", "beta"], ["tests/test_a.py"]) == ["alpha"]


@pytest.mark.synthetic_cfg
def test_changed_modules_a_deleted_own_source_file_still_selects_its_module():
    # The source-side mirror of the test above: engine/a.py, alpha's mutate
    # file, deleted by this PR.
    cfg2 = _sel_cfg()
    assert pilot.changed_modules(cfg2, ["alpha", "beta"], ["engine/a.py"]) == ["alpha"]


def test_changed_modules_a_deleted_own_source_file_selects_via_glob_ownership():
    # gamma owns its sources through a glob pattern (engine/pkg/*.py), not an
    # explicit path list, unlike alpha/beta above. A deleted file under that
    # glob must still select gamma -- module_owns_changed_path's fnmatch
    # check treats a glob and a literal path the same way.
    cfg2 = {
        "pr_selection": {"inert": ["*.md", "docs/*", "guides/*"]},
        "defaults": {},
        "modules": {
            "gamma": {"why": "x", "mutate": ["engine/pkg/*.py"], "tests": ["tests/test_g.py"]},
            "beta": {"why": "x", "mutate": ["engine/b.py"], "tests": ["tests/test_b.py"]},
        },
    }
    assert pilot.changed_modules(cfg2, ["gamma", "beta"], ["engine/pkg/gone.py"]) == ["gamma"]


@pytest.mark.synthetic_cfg
def test_module_owns_changed_path_matches_tests_and_mutate_minus_skip():
    cfg2 = _sel_cfg()
    assert pilot.module_owns_changed_path(cfg2, "alpha", "tests/test_a.py") is True
    assert pilot.module_owns_changed_path(cfg2, "alpha", "engine/a.py") is True
    assert pilot.module_owns_changed_path(cfg2, "alpha", "engine/b.py") is False
    assert pilot.module_owns_changed_path(cfg2, "alpha", "tests/test_b.py") is False


@pytest.mark.synthetic_cfg
def test_changed_modules_respects_the_incoming_names_subset():
    # beta's own file changed, but beta was already excluded (e.g. by
    # --only). beta still OWNS engine/b.py (it is an enabled module in the
    # full cfg), so the path is "explained" and must not fall through to
    # "select every name in names" -- it just contributes nothing to this
    # narrower `names` list.
    cfg2 = _sel_cfg()
    assert pilot.changed_modules(cfg2, ["alpha"], ["engine/b.py"]) == []


@pytest.mark.synthetic_cfg
def test_changed_modules_empty_change_selects_nothing():
    cfg2 = _sel_cfg()
    assert pilot.changed_modules(cfg2, ["alpha", "beta"], []) == []


@pytest.mark.synthetic_cfg
def test_changed_modules_an_inert_path_selects_nothing():
    cfg2 = _sel_cfg()
    assert pilot.changed_modules(cfg2, ["alpha", "beta"], ["docs/readme.md"]) == []


@pytest.mark.synthetic_cfg
def test_changed_modules_all_docs_change_selects_nothing():
    cfg2 = _sel_cfg()
    changed = ["README.md", "docs/design/notes.md", "guides/how_to.md"]
    assert pilot.changed_modules(cfg2, ["alpha", "beta"], changed) == []


@pytest.mark.synthetic_cfg
def test_changed_modules_mixed_docs_and_owned_change_selects_just_that_module():
    cfg2 = _sel_cfg()
    changed = ["docs/design/notes.md", "engine/a.py"]
    assert pilot.changed_modules(cfg2, ["alpha", "beta"], changed) == ["alpha"]


@pytest.mark.synthetic_cfg
def test_is_inert_changed_path_matches_only_the_configured_patterns():
    cfg2 = _sel_cfg()
    assert pilot.is_inert_changed_path(cfg2, "README.md") is True
    assert pilot.is_inert_changed_path(cfg2, "docs/anything/nested.txt") is True
    assert pilot.is_inert_changed_path(cfg2, "guides/exp133_structure_search.md") is True
    assert pilot.is_inert_changed_path(cfg2, "engine/a.py") is False
    assert pilot.is_inert_changed_path(cfg2, "tools/mutation_pilot.py") is False


# A cfg shaped like the real tools/mutation_pilot.toml's ownership boundaries:
# alpha/beta are ordinary enabled modules under engine/v2/, contracts is
# EXCLUDED (mirrors the real `contracts` module, which owns
# engine/v2/__init__.py but never runs). `names` below is what
# `enabled_modules` returns for this cfg -- contracts is never in it.
def _defect_list_cfg():
    return {
        "pr_selection": {"inert": ["*.md", "docs/*", "guides/*"]},
        "defaults": {},
        "modules": {
            "alpha": {"why": "x", "mutate": ["engine/v2/alpha/*.py"],
                      "tests": ["tests/test_v2_alpha.py"]},
            "beta": {"why": "x", "mutate": ["engine/v2/beta/*.py"],
                     "tests": ["tests/test_v2_beta.py"]},
            "contracts": {"why": "x", "excluded": "no mutants",
                          "mutate": ["engine/v2/__init__.py", "engine/v2/contracts/*.py"]},
        },
    }


_DEFECT_LIST_PATHS = [
    "tests/fixtures/v2_ui_mock_api.py",
    "engine/v2/__init__.py",  # owned only by the EXCLUDED contracts module
    "engine/analogs.py",  # a legacy top-level engine/*.py file no module owns
    "experiments/EXP-169_menu_search/run.py",
    "tools/mutation_pilot.py",
    "tools/gremlin_pilot.py",
    "tools/mutation_results.py",
    "tools/mutation_pilot.toml",
    ".github/workflows/mutation.yml",
    ".github/workflows/mutation-mutmut.yml",
    "requirements.txt",
    "requirements-dev.txt",
    "tests/conftest.py",
]


@pytest.mark.synthetic_cfg
@pytest.mark.parametrize("changed_path", _DEFECT_LIST_PATHS)
def test_changed_modules_selects_every_enabled_module_for_an_unrecognized_path(changed_path):
    # Each of these is a real path that the pre-2026-09-26 rule selected ZERO
    # modules for (it owns none of them and none was on any shared-input
    # list), silently skipping the whole mutation matrix on a PR that
    # touched it. None is inert either (none is *.md/docs/*/guides/*), so
    # the rule must select every enabled module -- `names` here IS every
    # enabled module, i.e. `pilot.enabled_modules(cfg)` (contracts excluded).
    cfg2 = _defect_list_cfg()
    names = pilot.enabled_modules(cfg2)
    assert names == ["alpha", "beta"]
    assert pilot.changed_modules(cfg2, names, [changed_path]) == names


@pytest.mark.synthetic_cfg
def test_changed_modules_respects_only_even_when_selecting_everything():
    # The "select everything" branch selects every name in the INCOMING
    # `names` (already --only-filtered), not every module in the cfg.
    cfg2 = _defect_list_cfg()
    assert pilot.changed_modules(cfg2, ["alpha"], ["tools/mutation_pilot.py"]) == ["alpha"]


@pytest.mark.synthetic_cfg
def test_cmd_matrix_changed_files_narrows_the_matrix(tmp_path, monkeypatch, capsys):
    cfg2 = _sel_cfg()
    monkeypatch.setattr(pilot, "enabled_modules", lambda c: ["alpha", "beta"])
    changed = tmp_path / "changed.txt"
    changed.write_bytes(b"engine/a.py\0")
    args = types.SimpleNamespace(only="", changed_files=str(changed))
    assert pilot.cmd_matrix(cfg2, args) == 0
    assert json.loads(capsys.readouterr().out) == ["alpha"]


@pytest.mark.synthetic_cfg
def test_cmd_matrix_without_changed_files_is_unaffected(monkeypatch, capsys):
    cfg2 = _sel_cfg()
    monkeypatch.setattr(pilot, "enabled_modules", lambda c: ["alpha", "beta"])
    args = types.SimpleNamespace(only="", changed_files="")
    assert pilot.cmd_matrix(cfg2, args) == 0
    assert json.loads(capsys.readouterr().out) == ["alpha", "beta"]


@pytest.mark.synthetic_cfg
def test_cmd_matrix_accepts_missing_changed_files_attr_for_backward_compatibility(monkeypatch, capsys):
    # older call sites (and the pre-existing tests above) build an
    # `args` namespace with no `changed_files` at all; that must keep working.
    cfg2 = _sel_cfg()
    monkeypatch.setattr(pilot, "enabled_modules", lambda c: ["alpha", "beta"])
    args = types.SimpleNamespace(only="")
    assert pilot.cmd_matrix(cfg2, args) == 0
    assert json.loads(capsys.readouterr().out) == ["alpha", "beta"]


def test_real_toml_inert_allowlist_is_the_small_docs_only_list():
    # Locks in the intended small allowlist: widening it (even by one
    # pattern) is a real design decision, not something that should drift
    # silently through an unrelated toml edit. The allowlist names root-level
    # Markdown files explicitly (2026-09-26) rather than a broad "*.md",
    # which used to make every package-local README.md (e.g.
    # engine/v2/foundation/README.md) inert too.
    cfg2 = pilot.load_config()
    assert cfg2["pr_selection"]["inert"] == [
        "README.md",
        "EARNINGS_VOL_PROGRAM_PLAN.md",
        "PROJECT_ASSESSMENT_2026-09-05.md",
        "RECOVERY.md",
        "TECH_DEBT.md",
        "docs/*",
        "guides/*",
    ]


@pytest.mark.synthetic_cfg
def test_is_inert_changed_path_respects_inert_skip():
    cfg2 = _sel_cfg()
    cfg2["pr_selection"]["inert_skip"] = ["engine/dashboard/static/*.md"]
    assert pilot.is_inert_changed_path(cfg2, "docs/readme.md") is True
    assert pilot.is_inert_changed_path(cfg2, "engine/dashboard/static/notes.md") is False


def test_real_toml_inert_skip_carves_dashboard_static_out_of_the_md_pattern():
    assert CFG["pr_selection"]["inert_skip"] == ["engine/dashboard/static/*.md"]
    assert pilot.is_inert_changed_path(CFG, "engine/dashboard/static/notes.md") is False
    # The root-level Markdown files named explicitly in `inert` are still
    # inert -- but a package-local README.md (e.g. under engine/v2/) is NOT:
    # `inert` no longer has a broad "*.md" pattern for `inert_skip` to carve
    # anything out of, so an engine/**/README.md change now correctly
    # selects every enabled module instead of being silently skipped.
    assert pilot.is_inert_changed_path(CFG, "README.md") is True
    assert pilot.is_inert_changed_path(CFG, "engine/v2/foundation/README.md") is False


def test_a_md_file_under_dashboard_static_selects_every_enabled_module():
    names = pilot.enabled_modules(CFG)
    assert pilot.changed_modules(CFG, names, ["engine/dashboard/static/CHANGELOG.md"]) == names


def test_a_package_local_readme_is_no_longer_inert_and_selects_every_module():
    # Regression for the pre-2026-09-26 broad "*.md" inert pattern, which
    # made a real, non-dashboard package README (e.g.
    # engine/v2/foundation/README.md) inert and silently skipped the whole
    # PR matrix. The allowlist now names root-level Markdown files
    # explicitly, so this unrecognized path selects every enabled module.
    names = pilot.enabled_modules(CFG)
    assert pilot.changed_modules(CFG, names, ["engine/v2/foundation/README.md"]) == names


# -- reverse import closure ---------------------------------------------------

def test_build_import_graph_resolves_a_package_import_through_its_init(tmp_path, monkeypatch):
    (tmp_path / "engine" / "pkg").mkdir(parents=True)
    (tmp_path / "engine" / "pkg" / "__init__.py").write_text(
        "from engine.pkg.inner import thing\n")
    (tmp_path / "engine" / "pkg" / "inner.py").write_text("thing = 1\n")
    (tmp_path / "engine" / "user.py").write_text("import engine.pkg\n")
    tracked = ["engine/pkg/__init__.py", "engine/pkg/inner.py", "engine/user.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    # user.py imports the PACKAGE (engine.pkg) -> resolves to its __init__.py
    # -> whose own `from engine.pkg.inner import thing` reaches inner.py, so
    # the closure from user.py would reach inner.py transitively.
    assert "engine/pkg/__init__.py" in graph["engine/user.py"]
    assert "engine/pkg/inner.py" in graph["engine/pkg/__init__.py"]
    assert graph["engine/pkg/inner.py"] == set()


def test_build_import_graph_a_submodule_import_still_depends_on_the_parent_inits(tmp_path, monkeypatch):
    # `import engine.pkg.inner` runs engine/pkg/__init__.py before inner.py,
    # even though this statement never names engine.pkg itself and
    # engine/pkg/__init__.py is empty (no `from .inner import ...`). The
    # graph must still record that dependency, or a change to
    # engine/pkg/__init__.py looks unrelated to code that only ever imports
    # the deeper submodule.
    (tmp_path / "engine" / "pkg").mkdir(parents=True)
    (tmp_path / "engine" / "pkg" / "__init__.py").write_text("")
    (tmp_path / "engine" / "pkg" / "inner.py").write_text("thing = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_importer.py").write_text("import engine.pkg.inner\n")
    tracked = ["engine/pkg/__init__.py", "engine/pkg/inner.py", "tests/test_importer.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert "engine/pkg/__init__.py" in graph["tests/test_importer.py"]
    assert "engine/pkg/inner.py" in graph["tests/test_importer.py"]

    # End-to-end: a module ("owner") owns engine/pkg/__init__.py; a
    # different module ("importer") owns only the test that imports the
    # submodule. A change to engine/pkg/__init__.py must still select
    # "importer", even though "importer" never owns that path and its own
    # source/test never names engine.pkg directly.
    cfg2 = {
        "pr_selection": {"inert": []},
        "defaults": {},
        "modules": {
            "owner": {"why": "x", "mutate": ["engine/pkg/__init__.py"], "tests": []},
            "importer": {"why": "x", "mutate": ["engine/pkg/inner.py"],
                         "tests": ["tests/test_importer.py"]},
        },
    }
    selected = pilot.changed_modules(cfg2, ["owner", "importer"],
                                      ["engine/pkg/__init__.py"], graph=graph)
    assert selected == ["owner", "importer"]


def test_build_import_graph_resolves_relative_imports(tmp_path, monkeypatch):
    (tmp_path / "engine" / "a" / "b").mkdir(parents=True)
    (tmp_path / "engine" / "a" / "__init__.py").write_text("")
    (tmp_path / "engine" / "a" / "sibling.py").write_text("X = 1\n")
    (tmp_path / "engine" / "a" / "b" / "__init__.py").write_text("")
    (tmp_path / "engine" / "a" / "b" / "mod.py").write_text(
        "from .. import sibling\nfrom . import other\n")
    (tmp_path / "engine" / "a" / "b" / "other.py").write_text("Y = 1\n")
    tracked = ["engine/a/__init__.py", "engine/a/sibling.py", "engine/a/b/__init__.py",
               "engine/a/b/mod.py", "engine/a/b/other.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    # `from ..` (level 2) reaches the sibling module one package up;
    # `from .` (level 1) reaches the module in mod.py's own package.
    assert {"engine/a/sibling.py", "engine/a/b/other.py"} <= graph["engine/a/b/mod.py"]


def test_build_import_graph_ignores_imports_outside_engine_and_tests(tmp_path, monkeypatch):
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "user.py").write_text(
        "import os\nimport tools.mutation_pilot\nfrom checks import layer_map\n")
    tracked = ["engine/user.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["engine/user.py"] == set()


def test_build_import_graph_raises_on_a_syntax_error(tmp_path, monkeypatch):
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "broken.py").write_text("def broken(:\n    pass\n")
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    with pytest.raises(SyntaxError):
        pilot.build_import_graph(["engine/broken.py"])


@pytest.mark.synthetic_cfg
def test_changed_modules_falls_back_to_selecting_all_when_the_graph_build_raises(monkeypatch):
    cfg2 = _sel_cfg()

    def boom(*_a, **_k):
        raise SyntaxError("engine/a.py: invalid syntax")

    monkeypatch.setattr(pilot, "build_import_graph", boom)
    assert pilot.changed_modules(cfg2, ["alpha", "beta"], ["engine/a.py"]) == ["alpha", "beta"]


@pytest.mark.synthetic_cfg
def test_module_dependency_closure_follows_edges_transitively():
    graph = {
        "engine/a.py": {"engine/b.py"},
        "engine/b.py": {"engine/c.py"},
        "engine/c.py": set(),
        "tests/test_a.py": {"engine/a.py"},
    }
    cfg2 = _sel_cfg()  # alpha: mutate=[engine/a.py], tests=[tests/test_a.py]
    closure = pilot.module_dependency_closure(cfg2, "alpha", graph)
    assert closure == {"engine/a.py", "engine/b.py", "engine/c.py", "tests/test_a.py"}


@pytest.mark.synthetic_cfg
def test_changed_modules_selects_a_module_that_only_transitively_depends_on_the_changed_path():
    # alpha does not OWN engine/c.py, but alpha's own test file imports
    # alpha's source file, which imports engine/c.py transitively -- alpha
    # must still be selected.
    graph = {
        "engine/a.py": {"engine/c.py"},
        "engine/b.py": set(),
        "engine/c.py": set(),
        "tests/test_a.py": {"engine/a.py"},
        "tests/test_b.py": set(),
    }
    cfg2 = _sel_cfg()
    assert pilot.changed_modules(cfg2, ["alpha", "beta"], ["engine/c.py"], graph=graph) == ["alpha"]


def test_artifacts_change_selects_every_module_whose_tests_transitively_import_foundation():
    # The Opus-blocking counterexample this round fixes: the pre-fix rule
    # selected ONLY ['foundation'] for engine/v2/foundation/artifacts.py,
    # even though 27 other enabled modules' tests import
    # engine.v2.foundation (whose own __init__.py re-exports names from
    # artifacts.py, so the import graph reaches artifacts.py from any test
    # that imports the package).
    names = pilot.enabled_modules(CFG)
    selected = pilot.changed_modules(CFG, names, ["engine/v2/foundation/artifacts.py"])
    assert "foundation" in selected

    tracked_tests = _tracked("tests")
    real_dependents = set()
    for name in names:
        for rel in pilot.test_files(CFG, name, tracked_tests):
            if "engine.v2.foundation" in (ROOT / rel).read_text():
                real_dependents.add(name)
                break
    assert len(real_dependents) >= 10, "expected many real dependents for this assertion to mean anything"
    assert real_dependents <= set(selected)
    assert len(selected) > 1  # NOT just ['foundation'] -- the defect this round fixes


def test_frozen_inputs_change_selects_chooser_via_the_checks_bridge():
    # The Opus-blocking counterexample THIS round fixes: the pre-fix rule
    # selected ONLY ['scoring_application'] for
    # engine/v2/scoring/frozen_inputs.py, because chooser's own test
    # (tests/test_phase4_capture_strict.py) reaches frozen_inputs only
    # through checks/phase4_frozen_bridge.py -- a file the old
    # engine/tests-only graph never parsed at all. checks/tools/experiments
    # are now graph nodes too, so the bridge closes the chain.
    names = pilot.enabled_modules(CFG)
    selected = pilot.changed_modules(CFG, names, ["engine/v2/scoring/frozen_inputs.py"])
    assert "chooser" in selected
    assert "scoring_application" in selected  # already selected before this fix
    assert len(selected) > 1


def test_build_import_graph_covers_checks_tools_and_experiments_too():
    # Not just engine/tests: any tracked top-level package is a graph node,
    # with no hand-kept root list.
    graph = pilot.build_import_graph()
    assert any(rel.startswith("checks/") for rel in graph)
    assert any(rel.startswith("tools/") for rel in graph)
    assert any(rel.startswith("experiments/") for rel in graph)


def test_build_import_graph_a_synthetic_test_reaches_engine_through_checks(tmp_path, monkeypatch):
    # A test file imports a `checks.*` bridge module, which imports the real
    # `engine.*` module -- the synthetic version of the frozen_inputs defect
    # above, isolated from the real repo's file layout.
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "y.py").write_text("Y = 1\n")
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "x.py").write_text("from engine import y\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_bridge.py").write_text("from checks import x\n")
    tracked = ["engine/y.py", "checks/x.py", "tests/test_bridge.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert "checks/x.py" in graph["tests/test_bridge.py"]
    assert "engine/y.py" in graph["checks/x.py"]

    cfg2 = {
        "pr_selection": {"inert": []},
        "defaults": {},
        "modules": {
            "target": {"why": "x", "mutate": ["engine/y.py"],
                       "tests": ["tests/test_bridge.py"]},
        },
    }
    selected = pilot.changed_modules(cfg2, ["target"], ["engine/y.py"], graph=graph)
    assert selected == ["target"]


def test_build_import_graph_resolves_an_importlib_import_module_string_literal(tmp_path, monkeypatch):
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "y.py").write_text("Y = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_dynamic.py").write_text(
        "import importlib\n"
        "importlib.import_module('engine.y')\n")
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["engine/y.py", "tests/test_dynamic.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    # the ONE allowed shape resolves a specific edge -- unrelated.py stays
    # out, which a fail-safe select-all would not distinguish.
    assert graph["tests/test_dynamic.py"] == {"engine/y.py"}


def test_build_import_graph_a_bare_dunder_import_always_fails_safe(tmp_path, monkeypatch):
    # `__import__(...)` is unconditionally dynamic now, even with a literal
    # argument -- the allowlist has no shape for it at all, unlike the one
    # narrow `importlib.import_module(...)` call it used to share this
    # resolution with.
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "y.py").write_text("Y = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_dynamic.py").write_text("__import__('engine.y')\n")
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["engine/y.py", "tests/test_dynamic.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tests/test_dynamic.py"] == {"engine/y.py", "engine/unrelated.py"}


def test_build_import_graph_a_non_literal_import_module_argument_fails_safe(tmp_path, monkeypatch):
    # A computed/variable argument is dynamic and this static graph cannot
    # see it -- sound by construction means the WHOLE FILE fails safe,
    # never a silent no-op.
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "y.py").write_text("Y = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_dynamic.py").write_text(
        "import importlib\n"
        "name = 'engine.' + 'y'\n"
        "importlib.import_module(name)\n")
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["engine/y.py", "tests/test_dynamic.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tests/test_dynamic.py"] == {"engine/y.py", "engine/unrelated.py"}


def test_build_import_graph_import_module_with_package_kwarg_fails_safe(tmp_path, monkeypatch):
    # A `package=` keyword makes the name relative to the CALLING file's own
    # package -- this graph does not attempt that resolution, even when the
    # name itself is a literal, so the whole file fails safe.
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "y.py").write_text("Y = 1\n")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "loader.py").write_text(
        "import importlib\n"
        "importlib.import_module('.y', package='engine')\n")
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["engine/y.py", "tools/loader.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tools/loader.py"] == {"engine/y.py", "engine/unrelated.py"}


def test_build_import_graph_a_subprocess_with_non_literal_argv_fails_safe(tmp_path, monkeypatch):
    # The argv itself is a variable -- this graph cannot see whether it
    # launches Python or not, so it must fail safe rather than silently
    # ignore the call.
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "runner.py").write_text(
        "import subprocess\n"
        "def go(argv):\n"
        "    subprocess.run(argv)\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["tools/runner.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tools/runner.py"] == {"engine/unrelated.py"}


def test_build_import_graph_a_subprocess_with_a_variable_interpreter_fails_safe(tmp_path, monkeypatch):
    # The argv IS a literal list, but its first element is a variable this
    # graph cannot prove is or isn't the Python interpreter.
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "runner.py").write_text(
        "import subprocess\n"
        "def go(interp):\n"
        "    subprocess.run([interp, 'engine/unrelated.py'])\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["tools/runner.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tools/runner.py"] == {"engine/unrelated.py"}


def test_build_import_graph_a_subprocess_dash_c_fails_safe(tmp_path, monkeypatch):
    # `-c` runs arbitrary inline code -- never resolvable to a tracked file,
    # always fail-safe when the interpreter is Python.
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "runner.py").write_text(
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-c', 'print(1)'])\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["tools/runner.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tools/runner.py"] == {"engine/unrelated.py"}


def test_build_import_graph_any_sys_path_mutation_fails_safe_including_the_file_relative_form(tmp_path, monkeypatch):
    # The repo-root idiom, and any other sys.path mutation, is ALWAYS
    # fail-safe again -- static __file__-relative evaluation for sys.path
    # was removed in this round.
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "script.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["checks/script.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["checks/script.py"] == {"engine/unrelated.py"}


def test_build_import_graph_a_conftest_pytest_plugins_literal_is_an_edge(tmp_path, monkeypatch):
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "myplugin.py").write_text("P = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "conftest.py").write_text(
        "pytest_plugins = ['checks.myplugin']\n")
    (tmp_path / "tests" / "test_a.py").write_text("X = 1\n")
    tracked = ["checks/myplugin.py", "tests/conftest.py", "tests/test_a.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert "checks/myplugin.py" in graph["tests/conftest.py"]


def test_build_import_graph_a_conftest_pytest_plugins_non_literal_fails_safe(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "conftest.py").write_text(
        "import os\n"
        "pytest_plugins = [os.environ.get('PLUGIN', 'x')]\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["tests/conftest.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tests/conftest.py"] == {"engine/unrelated.py"}


def test_the_real_tests_conftest_fails_safe_via_its_own_sys_path_insert():
    # tests/conftest.py itself does
    # `REPO_ROOT = Path(__file__).resolve().parents[1]` then
    # `sys.path.insert(0, str(REPO_ROOT))` -- a two-statement form this
    # graph never resolves (no cross-statement variable tracking), so it
    # fails safe. Every test file's dependency closure includes
    # tests/conftest.py via `_conftest_ancestors`, so this is why a
    # non-inert change today still selects all 33 enabled modules.
    graph = pilot.build_import_graph()
    tracked_set = set(graph)
    assert graph["tests/conftest.py"] == tracked_set - {"tests/conftest.py"}


def test_conftest_dynamic_classification_guards_static_analysis_holes():
    # See https://github.com/yshewchuk/investment-validation/issues/42:
    # `_is_dynamic_file`'s allowlist has several known holes (string-target
    # monkeypatch.setattr/mock.patch, pytest.importorskip, getattr-based
    # imports of importlib/sys, __import__ via globals()/builtins, asyncio
    # subprocess-exec calls, __path__/sys.meta_path edits, pytest_plugins
    # outside a conftest.py or under an `if`, and `from pkg import *`
    # re-exports). None of them matter today, because tests/conftest.py
    # itself always classifies DYNAMIC (its own sys.path.insert), so every
    # test file's dependency closure already includes tests/conftest.py and
    # mutation selection can never narrow past those holes. This is a
    # dedicated guard, separate from
    # test_the_real_tests_conftest_fails_safe_via_its_own_sys_path_insert
    # above, so a failure here points straight at issue #42 instead of only
    # restating the fail-safe fact.
    graph = pilot.build_import_graph()
    tracked_set = set(graph)
    assert graph["tests/conftest.py"] == tracked_set - {"tests/conftest.py"}, (
        "tests/conftest.py is no longer fail-safe; mutation selection "
        "would narrow and expose the static-analysis holes in issue #42. "
        "Close them first."
    )


def test_build_import_graph_a_spec_from_file_location_call_always_fails_safe(tmp_path, monkeypatch):
    # Loader constructs get no static path evaluation any more, literal or
    # not -- ANY reference to `spec_from_file_location` fails the whole file
    # safe, dropped from an earlier round's literal-resolution behavior.
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "y.py").write_text("Y = 1\n")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "loader.py").write_text(
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('y', 'engine/y.py')\n")
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["engine/y.py", "tools/loader.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tools/loader.py"] == {"engine/y.py", "engine/unrelated.py"}


def test_build_import_graph_a_runpy_run_path_call_always_fails_safe(tmp_path, monkeypatch):
    # `runpy` is one of the always-dynamic modules now -- importing it at
    # all fails the whole file safe, literal run_path target or not.
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "y.py").write_text("Y = 1\n")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "runner.py").write_text(
        "import runpy\n"
        "runpy.run_path('engine/y.py', run_name='__main__')\n")
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["engine/y.py", "tools/runner.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tools/runner.py"] == {"engine/y.py", "engine/unrelated.py"}


def test_build_import_graph_a_dynamic_loader_path_selects_every_enabled_module(tmp_path, monkeypatch):
    # A non-literal path argument to spec_from_file_location cannot be
    # resolved statically -- the whole file is marked as depending on
    # everything, so a change to any tracked file makes changed_modules
    # select every module that reaches this file (here, "target" itself,
    # since the dynamic-loader file IS target's own mutate file).
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "dynamic.py").write_text(
        "import importlib.util\n"
        "def load(path):\n"
        "    return importlib.util.spec_from_file_location('m', path)\n")
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["engine/dynamic.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["engine/dynamic.py"] == {"engine/unrelated.py"}
    cfg2 = {
        "pr_selection": {"inert": []},
        "defaults": {},
        "modules": {
            "target": {"why": "x", "mutate": ["engine/dynamic.py"], "tests": []},
        },
    }
    selected = pilot.changed_modules(cfg2, ["target"], ["engine/unrelated.py"], graph=graph)
    assert selected == ["target"]


def test_build_import_graph_a_sys_path_insert_selects_every_enabled_module(tmp_path, monkeypatch):
    # sys.path.insert(...) can make a later plain `import x` resolve to a
    # file this static graph cannot predict -- same fail-safe as a dynamic
    # loader-call path.
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "pathhack.py").write_text(
        "import sys\n"
        "sys.path.insert(0, 'somewhere')\n")
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["engine/pathhack.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["engine/pathhack.py"] == {"engine/unrelated.py"}
    cfg2 = {
        "pr_selection": {"inert": []},
        "defaults": {},
        "modules": {
            "target": {"why": "x", "mutate": ["engine/pathhack.py"], "tests": []},
        },
    }
    selected = pilot.changed_modules(cfg2, ["target"], ["engine/unrelated.py"], graph=graph)
    assert selected == ["target"]


def test_build_import_graph_an_env_var_sys_path_insert_still_fails_safe(tmp_path, monkeypatch):
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "script.py").write_text(
        "import sys, os\n"
        "sys.path.insert(0, os.environ['SOME_PATH'])\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["checks/script.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["checks/script.py"] == {"engine/unrelated.py"}


def test_build_import_graph_a_python_subprocess_literal_script_always_fails_safe(tmp_path, monkeypatch):
    # Subprocess edge inference is dropped entirely as of this round: EVERY
    # subprocess call fails the whole file safe now, including one with a
    # fully literal, `-u`-flagged Python script argv that an earlier round
    # would have resolved to a specific edge.
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "runner.py").write_text(
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-u', 'tools/worker.py'])\n")
    (tmp_path / "tools" / "worker.py").write_text("W = 1\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["tools/runner.py", "tools/worker.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tools/runner.py"] == {"tools/worker.py", "engine/unrelated.py"}


def test_build_import_graph_a_python_subprocess_dash_m_always_fails_safe(tmp_path, monkeypatch):
    # Same as above for `-m`, an absolute interpreter path, and `-X`/`-W`
    # flags -- none of these ever earn a specific edge any more.
    (tmp_path / "tools").mkdir()
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "worker.py").write_text("W = 1\n")
    (tmp_path / "tools" / "runner.py").write_text(
        "import subprocess\n"
        "subprocess.run(['/usr/bin/python3', '-X', 'utf8', '-W', 'ignore',"
        " '-m', 'checks.worker'])\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["tools/runner.py", "checks/worker.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tools/runner.py"] == {"checks/worker.py", "engine/unrelated.py"}


def test_build_import_graph_a_subprocess_dash_m_pytest_always_fails_safe(tmp_path, monkeypatch):
    # `-m pytest` (a THIRD-PARTY module, not a tracked one): the old
    # resolution would have tried `_resolve_dotted("pytest", ...)`, found no
    # tracked file, and added no edge, silently missing that the module
    # under test collects and runs the CURRENT tracked test tree -- now it
    # fails the whole file safe instead of silently doing nothing.
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "runner.py").write_text(
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-m', 'pytest'])\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["tools/runner.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tools/runner.py"] == {"engine/unrelated.py"}


def test_build_import_graph_a_dynamic_python_subprocess_target_fails_safe(tmp_path, monkeypatch):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "runner.py").write_text(
        "import subprocess, sys\n"
        "def go(script):\n"
        "    subprocess.run([sys.executable, script])\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["tools/runner.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tools/runner.py"] == {"engine/unrelated.py"}


def test_build_import_graph_any_subprocess_use_fails_safe_even_a_non_python_command(tmp_path, monkeypatch):
    # Reverses the pre-this-round behavior on purpose: subprocess edge
    # inference is dropped entirely, so even a `git`/`gh`/non-Python
    # command -- previously treated as provably inert -- now fails the
    # whole file safe. The PR-owned test this replaces asserted the
    # OPPOSITE (`graph["tools/runner.py"] == set()`) and could not be kept
    # once every subprocess reference became unconditionally dynamic.
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "runner.py").write_text(
        "import subprocess\n"
        "def go(args):\n"
        "    subprocess.run(['git', *args])\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["tools/runner.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tools/runner.py"] == {"engine/unrelated.py"}


def test_build_import_graph_from_sys_import_path_always_fails_safe(tmp_path, monkeypatch):
    # `from sys import path` binds the list directly, with no `sys.`
    # attribute access at the use site at all -- the allowlist bans the
    # IMPORT STATEMENT itself, not just a later attribute reference.
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "script.py").write_text(
        "from sys import path\n"
        "path.insert(0, 'somewhere')\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["checks/script.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["checks/script.py"] == {"engine/unrelated.py"}


def test_build_import_graph_an_aliased_sys_path_reference_always_fails_safe(tmp_path, monkeypatch):
    # `import sys as s` then `s.path` (or a bare `p = sys.path`, the same
    # attribute-access node either way) -- aliasing `sys` does not escape
    # the check, which tracks every name a file binds to the `sys` module.
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "script.py").write_text(
        "import sys as s\n"
        "p = s.path\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["checks/script.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["checks/script.py"] == {"engine/unrelated.py"}


def test_build_import_graph_site_addsitedir_always_fails_safe(tmp_path, monkeypatch):
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "script.py").write_text(
        "import site\n"
        "site.addsitedir('somewhere')\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["checks/script.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["checks/script.py"] == {"engine/unrelated.py"}


def test_build_import_graph_a_monkeypatch_syspath_prepend_always_fails_safe(tmp_path, monkeypatch):
    # A pytest fixture method, not an import at all -- `syspath_prepend` is
    # checked as a standalone name (an attribute access on WHATEVER object),
    # since this graph never knows a bare `monkeypatch` parameter is really
    # pytest's own fixture.
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_it(monkeypatch):\n"
        "    monkeypatch.syspath_prepend('somewhere')\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["tests/test_x.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tests/test_x.py"] == {"engine/unrelated.py"}


def test_build_import_graph_a_bare_pythonpath_string_always_fails_safe(tmp_path, monkeypatch):
    # Not inside a subprocess call at all (subprocess already fails safe on
    # its own) -- a bare "PYTHONPATH" string, e.g. an `os.environ` key, is
    # checked as its own standalone trigger.
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "script.py").write_text(
        "import os\n"
        "os.environ['PYTHONPATH'] = 'somewhere'\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["checks/script.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["checks/script.py"] == {"engine/unrelated.py"}


def test_build_import_graph_import_module_with_a_positional_package_argument_fails_safe(tmp_path, monkeypatch):
    # `package` passed POSITIONALLY (not as `package=`) is the same relative
    # -name risk the keyword form is -- the allowed shape requires EXACTLY
    # one positional argument and no others, however they're spelled.
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "y.py").write_text("Y = 1\n")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "loader.py").write_text(
        "import importlib\n"
        "importlib.import_module('.y', 'engine')\n")
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["engine/y.py", "tools/loader.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tools/loader.py"] == {"engine/y.py", "engine/unrelated.py"}


def test_build_import_graph_an_aliased_import_module_always_fails_safe(tmp_path, monkeypatch):
    # `from importlib import import_module as im` is a `from importlib
    # import ...`, which fails safe unconditionally regardless of the
    # aliasing -- the ONE allowed shape requires the unaliased attribute
    # form `importlib.import_module(...)` and nothing else.
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "y.py").write_text("Y = 1\n")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "loader.py").write_text(
        "from importlib import import_module as im\n"
        "im('engine.y')\n")
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["engine/y.py", "tools/loader.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tools/loader.py"] == {"engine/y.py", "engine/unrelated.py"}


def test_build_import_graph_importlib_dunder_import_always_fails_safe(tmp_path, monkeypatch):
    # `importlib.__import__(...)` -- an attribute access on `importlib`
    # OTHER than `import_module` -- is dynamic even though the argument is a
    # literal, exactly like `importlib.util`/`importlib.reload` would be.
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "y.py").write_text("Y = 1\n")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "loader.py").write_text(
        "import importlib\n"
        "importlib.__import__('engine.y')\n")
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["engine/y.py", "tools/loader.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tools/loader.py"] == {"engine/y.py", "engine/unrelated.py"}


def test_build_import_graph_an_annotated_conftest_pytest_plugins_always_fails_safe(tmp_path, monkeypatch):
    # An ANNOTATED assignment (`ast.AnnAssign`, not `ast.Assign`) is outside
    # the allowed shape regardless of the value being a literal list --
    # `_pytest_plugins_targets` only ever sees `ast.Assign`/`ast.AugAssign`,
    # so this is invisible to it and must be caught separately.
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "myplugin.py").write_text("P = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "conftest.py").write_text(
        "pytest_plugins: list = ['checks.myplugin']\n")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "unrelated.py").write_text("Z = 1\n")
    tracked = ["checks/myplugin.py", "tests/conftest.py", "engine/unrelated.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)
    assert graph["tests/conftest.py"] == {"checks/myplugin.py", "engine/unrelated.py"}


def test_the_real_executor_reaches_the_worker_it_launches_as_a_subprocess(tmp_path, monkeypatch):
    # The real miss an earlier round's Opus review found: engine/v2/ops/
    # executor.py launches engine/v2/ops/worker.py via
    # `subprocess.Popen([sys.executable, "-u", "-m", "engine.v2.ops.worker"],
    # ...)`, which the pre-allowlist code neither resolved as a specific
    # edge nor marked dynamic -- a silent, unexplained gap. Under the
    # allowlist, `executor.py`'s own `import subprocess` fails the whole
    # file safe, so it now depends on every other tracked file, including
    # worker.py, with no special-casing of this one call needed.
    graph = pilot.build_import_graph()
    assert "engine/v2/ops/worker.py" in graph["engine/v2/ops/executor.py"]


def test_conftest_own_imports_are_a_closure_root_for_tests_under_its_directory(tmp_path, monkeypatch):
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "y.py").write_text("Y = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "conftest.py").write_text("from engine import y\n")
    (tmp_path / "tests" / "test_a.py").write_text("X = 1\n")  # never imports engine.y itself
    tracked = ["engine/y.py", "tests/conftest.py", "tests/test_a.py"]
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    graph = pilot.build_import_graph(tracked)

    cfg2 = {
        "pr_selection": {"inert": []},
        "defaults": {},
        "modules": {
            "alpha": {"why": "x", "mutate": ["engine/other.py"], "tests": ["tests/test_a.py"]},
        },
    }
    # alpha's own test (test_a.py) never imports engine/y.py, and alpha
    # never mutates it -- but tests/conftest.py, which applies to every test
    # under tests/, does. alpha must still be selected. Add alpha's mutate
    # target as its own graph key (no imports of its own) so it is a valid
    # tracked_set member for `changed_modules`.
    graph["engine/other.py"] = set()
    selected = pilot.changed_modules(cfg2, ["alpha"], ["engine/y.py"], graph=graph)
    assert selected == ["alpha"]


def test_domain_valuation_init_change_selects_its_dependents():
    names = pilot.enabled_modules(CFG)
    selected = set(pilot.changed_modules(CFG, names, ["engine/v2/domain/valuation/__init__.py"]))
    assert "domain_features" in selected  # owns it directly (glob ownership)
    # engine/v2/scoring/stages.py and engine/v2/scoring/financial.py both
    # `from engine.v2.domain.valuation import ...` -- their owning modules
    # must be selected too, even though neither owns the changed path.
    assert "scoring_stages" in selected
    assert "scoring_application" in selected


def test_a_leaf_module_change_now_selects_every_module_via_the_sys_path_failsafe():
    # Before 2026-09-26, engine/v2/research/*.py was imported by nothing
    # outside the research module itself, so this changed path selected
    # exactly ["research"]. The path-based-loader fail-safe added that day
    # changed this: a large share of tracked checks/*.py files use the
    # common `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`
    # idiom to make themselves runnable as standalone scripts, which the
    # fail-safe (deliberately, per spec) cannot distinguish from a genuinely
    # unpredictable sys.path mutation -- each such file is marked as
    # depending on EVERY tracked file. Because nearly every enabled module's
    # test suite transitively reaches at least one checks/*.py file (the
    # same "chooser via checks bridge" path proven elsewhere in this file),
    # this is no longer a leaf change: it now selects all 33 enabled
    # modules, same as an unrecognized path would. This is a real, reported
    # breadth effect of the fail-safe (see PR discussion), not a bug in this
    # test.
    names = pilot.enabled_modules(CFG)
    assert pilot.changed_modules(CFG, names, ["engine/v2/research/replay.py"]) == names


def test_import_graph_build_is_fast():
    start = time.monotonic()
    graph = pilot.build_import_graph()
    elapsed = time.monotonic() - start
    assert elapsed < 15.0, f"import graph build took {elapsed:.2f}s (budget: 15s)"
    assert len(graph) > 800  # whole repo now, not just engine/+tests/


# -- mutate job summary: mutant-level cache reuse vs re-tested this run ------

def test_markdown_reports_cache_reuse_counts():
    rows = [_row("m", "a.py", "f", "killed", retested=True, name="n1"),
            _row("m", "a.py", "f", "survived", retested=False, name="n2"),
            _row("m", "a.py", "g", "survived", retested=False, name="n3")]
    s = mr.summarize(rows, "m", INFO, ["a.py"])
    assert s["retested_this_run"] == 1
    md = mr.markdown(s, rows, None)
    assert "**cache reuse**: 2 of 3 mutant(s) reused from cache, 1 re-tested this run." in md


def test_markdown_excludes_skipped_mutants_from_cache_reuse_denominator():
    # a skipped mutant (never run, this run or any prior one) has no cached
    # verdict to "reuse" -- it must not inflate the reused count.
    rows = [_row("m", "a.py", "f", "killed", retested=True, name="n1"),
            _row("m", "a.py", "f", "survived", retested=False, name="n2"),
            _row("m", "a.py", "g", "skipped", retested=False, name="n3")]
    s = mr.summarize(rows, "m", INFO, ["a.py"])
    assert (s["total"], s["checked"], s["retested_this_run"]) == (3, 2, 1)
    md = mr.markdown(s, rows, None)
    assert "**cache reuse**: 1 of 2 mutant(s) reused from cache, 1 re-tested this run." in md


def test_summarize_excludes_a_retested_skipped_mutant_from_the_retested_count():
    # Unlike the test above, this skipped mutant itself has
    # retested_this_run=True -- build_rows() sets that whenever the mutmut
    # config fingerprint changed, even though the mutant was never actually
    # run. It must still not count toward retested_this_run, or the count
    # can exceed `checked` and markdown's cache-reuse line understates reuse.
    rows = [_row("m", "a.py", "f", "killed", retested=True, name="n1"),
            _row("m", "a.py", "f", "survived", retested=False, name="n2"),
            _row("m", "a.py", "g", "skipped", retested=True, name="n3")]
    s = mr.summarize(rows, "m", INFO, ["a.py"])
    assert (s["total"], s["checked"], s["retested_this_run"]) == (3, 2, 1)
    md = mr.markdown(s, rows, None)
    assert "**cache reuse**: 1 of 2 mutant(s) reused from cache, 1 re-tested this run." in md


def test_summarize_retested_this_run_is_none_when_no_snapshot_row_has_it():
    # no before-snapshot at all (e.g. the very first run): every row's own
    # retested_this_run is None (build_rows sets this when before is None).
    # summarize() must report None too, not coerce it to a false "0".
    rows = [_row("m", "a.py", "f", "killed", retested=None),
            _row("m", "a.py", "f", "survived", retested=None)]
    s = mr.summarize(rows, "m", INFO, ["a.py"])
    assert s["retested_this_run"] is None
    md = mr.markdown(s, rows, None)
    assert "cache reuse" not in md  # unknown provenance: say nothing, not a false count
