"""Mutation CI: row/summary building, query filters, history, config and workflow shape.

Data-free and mutmut-free: every mutmut state file here is synthetic, written
in the layout mutmut 3.8 uses (``mutants/<file>.meta``, ``mutmut-stats.json``).
"""
from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import sys
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
    assert s["run_exit_code"] == 0 and s["retested_this_run"] == 5


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


def _cmd_run(monkeypatch, tmp_path, rc):
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
    args = types.SimpleNamespace(module="toy", fresh=False, max_children=None, globs=[])
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
    assert "tools/mutation_pilot.py matrix" in plan
    assert JOBS["mutate"]["strategy"]["matrix"]["module"] == \
        "${{ fromJSON(needs.plan.outputs.modules) }}"
    assert JOBS["mutate"]["strategy"]["fail-fast"] is False


def test_workflow_triggers_and_concurrency():
    on = WORKFLOW.get("on", WORKFLOW.get(True))  # PyYAML reads a bare `on` as True
    assert on["push"]["branches"] == ["main"]
    assert on["schedule"] and "cron" in on["schedule"][0]
    assert set(on["workflow_dispatch"]["inputs"]) == {"fresh", "modules"}
    assert WORKFLOW["concurrency"]["group"] == "mutation-${{ github.ref }}"
    assert WORKFLOW["permissions"] == {"contents": "read"}
    plan = JOBS["plan"]["steps"][-1]["run"]
    assert '"$EVENT" = "push"' in plan and '"$FRESH" = "false"' in plan and "mode=full" in plan


def test_workflow_pins_python_and_mutmut():
    header = (ROOT / "requirements.txt").read_text()
    assert f"# python {WORKFLOW['env']['PYTHON_VERSION']} " in header
    dev = (ROOT / "requirements-dev.txt").read_text()
    assert re.search(r"^mutmut==\d+\.\d+\.\d+$", dev, re.M)
    for job in JOBS.values():
        for step in job["steps"]:
            if str(step.get("uses", "")).startswith("actions/setup-python"):
                assert step["with"]["python-version"] == "${{ env.PYTHON_VERSION }}"


def _step(job, predicate):
    return next(s for s in JOBS[job]["steps"] if predicate(s))


def test_workflow_cache_key_and_restore_policy():
    key = _step("mutate", lambda s: s.get("id") == "key")["run"]
    for part in ("needs.plan.outputs.mutmut", "steps.py.outputs.python-version",
                 "hashFiles('tools/mutation_pilot.toml')", "matrix.module"):
        assert part in key, part
    restore = _step("mutate", lambda s: s.get("uses", "").startswith("actions/cache/restore"))
    save = _step("mutate", lambda s: s.get("uses", "").startswith("actions/cache/save"))
    prefix = "${{ steps.key.outputs.prefix }}"
    assert restore["with"]["key"] == save["with"]["key"]
    assert restore["with"]["key"].startswith(prefix + "${{ github.sha }}")
    assert restore["with"]["restore-keys"] == prefix  # the same key without the sha
    assert restore["if"] == "needs.plan.outputs.mode == 'incremental'"  # full: no restore
    assert "always()" in save["if"]
    paths = restore["with"]["path"].splitlines()
    assert paths == save["with"]["path"].splitlines()
    assert all(p.startswith("${{ env.STATE }}/") for p in paths)  # mutmut state only
    assert "investing-plan-mutation-pilot" in JOBS["mutate"]["env"]["STATE"]


def test_workflow_is_report_only():
    run = _step("mutate", lambda s: s.get("id") == "run")
    assert "set +e" in run["run"] and "mutation-rc" in run["run"]
    gate = _step("mutate", lambda s: s.get("name", "").startswith("Fail only on a tool error"))
    assert 'rc" != 0' in gate["run"] and "score" not in gate["run"]
    uploads = [s for j in JOBS.values() for s in j["steps"]
               if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    assert {u["with"]["retention-days"] for u in uploads} == {90}
    assert {u["with"]["name"] for u in uploads} == {"mutation-module-${{ matrix.module }}",
                                                    "mutation-report"}
    assert "always()" in JOBS["report"]["if"]
    download = _step("report", lambda s: str(s.get("uses", "")).startswith("actions/download"))
    assert download["with"]["pattern"] == "mutation-module-*"  # never the merged artifact
    assert tomllib.loads((ROOT / "tools" / "mutation_pilot.toml").read_text())  # parses
