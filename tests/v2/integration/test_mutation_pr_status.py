"""PR mutation status: PR-owned (red) versus infra (warning) classification.

The unit tests feed ``classify`` fixture summaries; the CLI tests run the real
``main`` over real files in ``tmp_path`` (summary.json, results.jsonl, diff,
baseline, ``$GITHUB_OUTPUT``), so file reading and output writing are not stubbed.

# packages: foundation, ops
"""
from __future__ import annotations

import copy
import json

import pytest

from tools import gremlin_results
from tools import mutation_pr_status as ps

DIFF = """\
diff --git a/engine/v2/a.py b/engine/v2/a.py
--- a/engine/v2/a.py
+++ b/engine/v2/a.py
@@ -3,0 +4,2 @@ def f():
+x = 1
+y = 2
@@ -10 +12 @@ def g():
-old
+new
@@ -20,2 +21,0 @@ def h():
-gone
-gone2
diff --git a/engine/v2/dead.py b/engine/v2/dead.py
--- a/engine/v2/dead.py
+++ /dev/null
@@ -1,3 +0,0 @@
-a
-b
-c
"""


def mod(**kw):
    """A module summary whose complete/tool_error/failure_reasons follow gremlin_results rules."""
    base = {"run_exit_code": 0, "skipped_reason": None, "counts_backend": {"error": 0},
            "failure_reasons": [], "complete": True, "tool_error": False}
    base.update(kw)
    rc = base["run_exit_code"]
    skipped = base["skipped_reason"] is not None
    counts = base["counts_backend"]
    errored = bool(counts.get("error")) if isinstance(counts, dict) else False
    if "complete" not in kw:
        base["complete"] = rc in (None, 0) and not skipped
    if "tool_error" not in kw:
        base["tool_error"] = (rc not in (None, 0) and not skipped) or errored
    if "failure_reasons" not in kw:
        reasons = []
        if rc not in (None, 0):
            reasons.append("TIMEOUT_KILL" if rc == -1 else "RUN_INCOMPLETE")
        if errored:
            reasons.append("ERROR_RESULTS")
        base["failure_reasons"] = reasons
    return base


def row(module, file, line, status="survived"):
    return {"module": module, "file": file, "line": line, "status": status}


def merged(**modules):
    """The merged summary gremlin_results.merge writes for these modules (no merge codes)."""
    return {"modules": modules, "expected_modules": sorted(modules),
            "complete": all(m["complete"] for m in modules.values()),
            "tool_error": any(m["tool_error"] for m in modules.values()),
            "failure_reasons": [f"{n}: {r}" for n, m in modules.items()
                                for r in m["failure_reasons"]]}


CHANGED = ps.parse_changed_lines(DIFF)


def test_parse_changed_lines_added_changed_deleted():
    assert CHANGED == {"engine/v2/a.py": {4, 5, 12}}


def verdict(m, rows=(), baseline=None):
    return ps.classify(m, list(rows), CHANGED, baseline)


def test_survivor_on_changed_line_is_pr_owned_and_red():
    v = verdict(merged(a=mod()), [row("a", "engine/v2/a.py", 5)])
    assert v["state"] == "failure" and v["pr_owned"] == ["a"]
    assert "engine/v2/a.py:5" in v["modules"]["a"]["reason"]


def test_survivor_on_untouched_line_or_killed_on_changed_line_is_ok():
    rows = [row("a", "engine/v2/a.py", 99), row("a", "engine/v2/a.py", 4, "killed"),
            row("a", "engine/v2/other.py", 4)]
    v = verdict(merged(a=mod()), rows)
    assert v["state"] == "success" and v["modules"]["a"]["class"] == "ok"


def test_module_test_failure_is_pr_owned_unless_main_fails_too():
    m = merged(a=mod(run_exit_code=1, failure_reasons=["RUN_INCOMPLETE"]))
    assert verdict(m)["state"] == "failure"
    base = {"modules": {"a": mod(run_exit_code=1)}}
    v = verdict(m, baseline=base)
    assert v["state"] == "neutral" and v["modules"]["a"]["class"] == "infra"
    assert "main" in v["modules"]["a"]["reason"]


def test_pr_owned_failure_wins_over_merge_level_tool_error():
    m = merged(a=mod(run_exit_code=1), b=mod())
    assert m["tool_error"] is True and m["complete"] is False
    v = verdict(m)
    assert v["state"] == "failure"
    assert v["modules"]["a"]["class"] == "pr_owned"
    m = merged(a=mod(run_exit_code=-1), b=mod())
    assert m["tool_error"] is True and m["complete"] is False
    v = verdict(m, [row("a", "engine/v2/a.py", 5)])
    assert v["state"] == "failure"
    assert v["modules"]["a"]["class"] == "pr_owned"


def test_each_infra_class_is_a_warning_never_red():
    cases = {
        "timeout": mod(run_exit_code=-1, failure_reasons=["TIMEOUT_KILL"]),
        "signal": mod(run_exit_code=137),
        "errors": mod(counts_backend={"error": 2}, failure_reasons=["ERROR_RESULTS"]),
        "skipped": mod(skipped_reason="pr_closed_mid_run"),
        "rawbad": mod(failure_reasons=["RAW_INVALID"]),
        "rawbad_rc1": mod(run_exit_code=1, failure_reasons=["RAW_INVALID", "RUN_INCOMPLETE"]),
    }
    m = merged(**cases)
    m["expected_modules"].append("missing")
    v = verdict(m)
    assert v["state"] == "neutral"
    assert sorted(v["infra"]) == sorted([*cases, "missing"])
    assert v["pr_owned"] == []
    assert "infra warning" in v["description"]


def test_mixed_result_is_red_and_description_fits_status_limit():
    modules = {f"module_{i}_with_a_long_name": mod(run_exit_code=1) for i in range(30)}
    v = verdict(merged(**modules, infra_one=mod(run_exit_code=-1)))
    assert v["state"] == "failure" and len(v["pr_owned"]) == 30
    assert len(v["description"]) <= 140  # GitHub's commit-status description limit


def test_clean_run_description():
    v = verdict(merged(a=mod(), b=mod()))
    assert v["state"] == "success" and "2 module(s)" in v["description"]


def _write_report(tmp_path, m, rows):
    d = tmp_path / "merged"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(m))
    (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    diff = tmp_path / "pr.diff"
    diff.write_text(DIFF)
    return d, diff


def test_main_writes_state_and_annotations(tmp_path, capsys):
    d, diff = _write_report(tmp_path, merged(a=mod(), b=mod(run_exit_code=-1)),
                            [row("a", "engine/v2/a.py", 12)])
    out = tmp_path / "gh_output"
    rc = ps.main(["--merged-dir", str(d), "--diff-file", str(diff),
                  "--github-output", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "state=failure\n" in text and "description=gremlins: PR-owned problem in a" in text
    stdout = capsys.readouterr().out
    assert "::error::a:" in stdout and "::warning::b: timeout or cancelled" in stdout


def test_main_defaults_github_output_to_the_env_variable(tmp_path, monkeypatch):
    d, diff = _write_report(tmp_path, merged(a=mod(), b=mod(run_exit_code=-1)),
                            [row("a", "engine/v2/a.py", 12)])
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    rc = ps.main(["--merged-dir", str(d), "--diff-file", str(diff)])
    assert rc == 0 and "state=failure\n" in out.read_text()


def test_main_without_any_github_output_still_prints_the_verdict(tmp_path, monkeypatch, capsys):
    d, diff = _write_report(tmp_path, merged(a=mod(), b=mod(run_exit_code=-1)),
                            [row("a", "engine/v2/a.py", 12)])
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    rc = ps.main(["--merged-dir", str(d), "--diff-file", str(diff)])
    assert rc == 0 and '"state": "failure"' in capsys.readouterr().out


def test_main_reads_baseline_file(tmp_path):
    d, diff = _write_report(tmp_path, merged(a=mod(run_exit_code=1)), [])
    base = tmp_path / "base.json"
    base.write_text(json.dumps(merged(a=mod(run_exit_code=1))))
    out = tmp_path / "gh_output"
    ps.main(["--merged-dir", str(d), "--diff-file", str(diff),
             "--baseline-summary", str(base), "--github-output", str(out)])
    assert "state=neutral\n" in out.read_text()


def test_main_unreadable_inputs_never_red(tmp_path, capsys):
    out = tmp_path / "gh_output"
    rc = ps.main(["--merged-dir", str(tmp_path / "nowhere"), "--github-output", str(out)])
    assert rc == 0 and "state=neutral\n" in out.read_text()
    assert "::warning::merged mutation report unavailable" in capsys.readouterr().out
    d, _ = _write_report(tmp_path, merged(a=mod(run_exit_code=1)), [])
    out2 = tmp_path / "gh_output2"
    ps.main(["--merged-dir", str(d), "--diff-file", str(tmp_path / "no.diff"),
             "--baseline-summary", str(tmp_path / "no.json"), "--github-output", str(out2)])
    assert "state=failure\n" in out2.read_text()
    assert "baseline summary unreadable" in capsys.readouterr().out


def test_missing_or_undecodable_diff_attributes_no_survivor(tmp_path, capsys):
    d = tmp_path / "m"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(merged(a=mod())))
    (d / "results.jsonl").write_text(json.dumps(row("a", "engine/v2/a.py", 5)) + "\n")
    bad = tmp_path / "bad.diff"
    bad.write_bytes(b"+++ b/engine/v2/a.py\n@@ -1 +5 @@\n\xff\xfe\n")
    for i, diff_args in enumerate([[], ["--diff-file", str(tmp_path / "no.diff")],
                                   ["--diff-file", str(bad)]]):
        out = tmp_path / f"gh_{i}"
        assert ps.main(["--merged-dir", str(d), "--github-output", str(out), *diff_args]) == 0
        text = out.read_text()
        assert "state=neutral\n" in text  # a survivor may sit on a changed line: never a pass
        assert "description=gremlins: PR diff unavailable -- survivors not attributed; not a pass\n" in text
        assert "PR diff unavailable" in capsys.readouterr().out


def test_corrupt_results_jsonl_warns(tmp_path, capsys):
    d, diff = _write_report(tmp_path, merged(a=mod()), [])
    (d / "results.jsonl").write_text("{not json\n")
    out = tmp_path / "gh_output"
    assert ps.main(["--merged-dir", str(d), "--diff-file", str(diff),
                    "--github-output", str(out)]) == 0
    assert "1 malformed results.jsonl row(s) ignored" in capsys.readouterr().out


def test_exit_band_boundaries():
    v = verdict(merged(lo=mod(run_exit_code=0), top=mod(run_exit_code=127),
                       sig=mod(run_exit_code=128), neg=mod(run_exit_code=-9),
                       timeout=mod(run_exit_code=-1)))
    classes = {n: m["class"] for n, m in v["modules"].items()}
    assert classes == {"lo": "ok", "top": "pr_owned", "sig": "infra", "neg": "infra",
                       "timeout": "infra"}


def test_changed_line_survivor_wins_over_infra_and_baseline_excuses():
    base = {"modules": {"main_fails": mod(run_exit_code=1)}}
    m = merged(timed_out=mod(run_exit_code=-1), main_fails=mod(run_exit_code=1),
               killed=mod(run_exit_code=137), rawbad=mod(failure_reasons=["RAW_INVALID"]),
               errors=mod(counts_backend={"error": 1}))
    rows = [row(n, "engine/v2/a.py", 4) for n in m["modules"]]
    v = verdict(m, rows, baseline=base)
    assert v["state"] == "failure"
    assert sorted(v["pr_owned"]) == sorted(m["modules"])
    # The same modules without a changed-line survivor stay infra / excused.
    v2 = verdict(m, [row(n, "engine/v2/a.py", 99) for n in m["modules"]], baseline=base)
    assert v2["state"] == "neutral" and v2["pr_owned"] == []


def test_survivor_from_rows_without_summary_entry_is_pr_owned():
    # A module reported twice is dropped from ``modules`` by the merge; its rows remain.
    m = {"modules": {"ok": mod()}, "expected_modules": ["ok", "dup"]}
    v = verdict(m, [row("dup", "engine/v2/a.py", 5)])
    assert v["state"] == "failure" and v["pr_owned"] == ["dup"]
    v = verdict(m, [row("dup", "engine/v2/a.py", 99)])
    assert v["state"] == "success" and v["modules"]["dup"]["class"] == "infra"  # "ok" was measured


def test_truncation_keeps_the_infra_count():
    modules = {f"module_{i}_with_a_long_name": mod(run_exit_code=1) for i in range(30)}
    v = verdict(merged(**modules, infra_one=mod(run_exit_code=-1)))
    assert len(v["description"]) <= 140  # GitHub's commit-status description limit
    assert v["description"].endswith("; 1 infra warning(s)")


def test_baseline_only_excuses_the_same_module_failing_with_a_test_exit():
    pr = merged(a=mod(run_exit_code=1))
    # (a) main green for the module: the PR's failure is PR-owned.
    assert verdict(pr, baseline={"modules": {"a": mod(run_exit_code=0)}})["state"] == "failure"
    # (b) main's module was skipped (placeholder rc): it must not excuse the PR.
    skipped = {"modules": {"a": mod(run_exit_code=1, skipped_reason="pr_closed_mid_run")}}
    assert verdict(pr, baseline=skipped)["state"] == "failure"
    # (c) main fails a different module only.
    other = {"modules": {"b": mod(run_exit_code=1)}}
    v = verdict(pr, baseline=other)
    assert v["state"] == "failure" and v["pr_owned"] == ["a"]


def test_baseline_that_is_not_a_test_exit_failure_excuses_nothing():
    pr = merged(a=mod(run_exit_code=1))
    for base_rc in (-1, 128, 137, -9, 0, None):
        base = {"modules": {"a": mod(run_exit_code=base_rc)}}
        assert verdict(pr, baseline=base)["state"] == "failure", base_rc
    for junk in ([], "x", 3, {"modules": []}, {"modules": {"a": "bad"}}, {"modules": None}):
        assert verdict(pr, baseline=junk)["state"] == "failure", junk
    # Positive control: a test-exit failure on main for the same module excuses it.
    assert verdict(pr, baseline={"modules": {"a": mod(run_exit_code=127)}})["state"] == "neutral"


def test_survivor_rows_of_another_module_are_not_attributed():
    v = verdict(merged(a=mod(), b=mod()), [row("b", "engine/v2/a.py", 5)])
    assert v["pr_owned"] == ["b"] and v["modules"]["a"]["class"] == "ok"


def test_main_matches_non_ascii_diff_path(tmp_path):
    path = "engine/v2/café.py"
    d, diff = _write_report(tmp_path, merged(a=mod()), [row("a", path, 12)])
    diff.write_text(DIFF.replace("engine/v2/a.py", path), encoding="utf-8")
    out = tmp_path / "gh_output"
    assert ps.main(["--merged-dir", str(d), "--diff-file", str(diff),
                    "--github-output", str(out)]) == 0
    assert "state=failure\n" in out.read_text()


def test_malformed_result_rows_are_ignored_with_a_warning(tmp_path, capsys):
    d, diff = _write_report(tmp_path, merged(a=mod()), [row("a", "engine/v2/a.py", 99)])
    with (d / "results.jsonl").open("a") as fh:
        fh.write("5\nnull\n[]\n" + json.dumps({"status": "survived"}) + "\n"
                 + json.dumps({"module": "a", "file": None, "line": 4, "status": "survived"}) + "\n"
                 + json.dumps({"module": "a", "file": "engine/v2/a.py", "line": "4",
                               "status": "survived"}) + "\n")
    out = tmp_path / "gh_output"
    assert ps.main(["--merged-dir", str(d), "--diff-file", str(diff),
                    "--github-output", str(out)]) == 0
    text = out.read_text()
    assert "state=neutral\n" in text  # dropped rows may have hidden a survivor
    assert ("description=gremlins: result rows unreadable or malformed -- survivors not "
            "attributed; not a pass\n") in text
    assert "6 malformed results.jsonl row(s) ignored" in capsys.readouterr().out


def test_parse_changed_lines_handles_quoted_and_spaced_paths():
    diff = (
        'diff --git "a/engine/v2/caf\\303\\251.py" "b/engine/v2/caf\\303\\251.py"\n'
        '--- "a/engine/v2/caf\\303\\251.py"\n'
        '+++ "b/engine/v2/caf\\303\\251.py"\n'
        "@@ -1 +3 @@\n-x\n+y\n"
        "diff --git a/engine/v2/with space.py b/engine/v2/with space.py\n"
        "--- a/engine/v2/with space.py\t\n"
        "+++ b/engine/v2/with space.py\t\n"
        "@@ -1,0 +7,2 @@\n+a\n+b\n"
        'diff --git "a/engine/v2/q\\"t\\t.py" "b/engine/v2/q\\"t\\t.py"\n'
        '+++ "b/engine/v2/q\\"t\\t.py"\n'
        "@@ -1 +2 @@\n-x\n+y\n"
        "diff --git a/engine/v2/plain.py b/engine/v2/plain.py\n"
        "+++ b/engine/v2/plain.py\n"
        "@@ -1 +9 @@\n-x\n+y\n"
    )
    assert ps.parse_changed_lines(diff) == {
        "engine/v2/café.py": {3},
        "engine/v2/with space.py": {7, 8},
        'engine/v2/q"t\t.py': {2},
        "engine/v2/plain.py": {9},
    }


def test_one_bad_results_line_does_not_discard_the_good_ones(tmp_path, capsys):
    d, diff = _write_report(tmp_path, merged(a=mod()), [row("a", "engine/v2/a.py", 5)])
    with (d / "results.jsonl").open("a") as fh:
        fh.write("{not json\n")
    out = tmp_path / "gh_output"
    assert ps.main(["--merged-dir", str(d), "--diff-file", str(diff),
                    "--github-output", str(out)]) == 0
    assert "state=failure\n" in out.read_text()  # the good row's survivor is still found
    assert "1 malformed results.jsonl row(s) ignored" in capsys.readouterr().out


def test_missing_or_undecodable_results_file_warns_unreadable(tmp_path, capsys):
    d, diff = _write_report(tmp_path, merged(a=mod()), [])
    (d / "results.jsonl").write_bytes(b"\xff\xfe\n")
    out = tmp_path / "gh_output"
    assert ps.main(["--merged-dir", str(d), "--diff-file", str(diff),
                    "--github-output", str(out)]) == 0
    assert "results.jsonl unreadable" in capsys.readouterr().out
    (d / "results.jsonl").unlink()
    assert ps.main(["--merged-dir", str(d), "--diff-file", str(diff),
                    "--github-output", str(out)]) == 0
    assert "results.jsonl unreadable" in capsys.readouterr().out


def test_list_valued_modules_in_summary_or_baseline_never_crash(tmp_path, capsys):
    d, diff = _write_report(tmp_path, {"modules": [], "expected_modules": ["a"]}, [])
    out = tmp_path / "gh_output"
    assert ps.main(["--merged-dir", str(d), "--diff-file", str(diff),
                    "--github-output", str(out)]) == 0
    assert "state=neutral\n" in out.read_text()
    assert "merged mutation report unavailable" in capsys.readouterr().out
    d2 = tmp_path / "two"
    d2.mkdir()
    (d2 / "summary.json").write_text(json.dumps(merged(a=mod(run_exit_code=1))))
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"modules": ["a"]}))
    out2 = tmp_path / "gh_output2"
    assert ps.main(["--merged-dir", str(d2), "--diff-file", str(diff),
                    "--baseline-summary", str(base), "--github-output", str(out2)]) == 0
    assert "state=failure\n" in out2.read_text()  # nothing is excused by a malformed baseline
    assert "baseline summary unreadable" in capsys.readouterr().out


def test_classify_tolerates_odd_shapes_in_parsed_json():
    m = {"modules": {"a": mod(counts_backend=[1], failure_reasons=7), 5: mod()},
         "expected_modules": ["b", 3, None]}
    v = ps.classify(m, [row("a", "x.py", 1), 5, None, {"module": 9}], {}, baseline={"modules": 3})
    assert v["state"] == "neutral"  # the malformed entry is infra: nothing was measured
    assert v["modules"] == {
        "a": {"class": "infra", "reason": "malformed module entry (counts_backend)"},
        "b": {"class": "infra", "reason": "no report: job cancelled, runner or setup error"}}
    v = ps.classify({"modules": {"a": mod()}, "expected_modules": "a"}, [], {}, [1])
    assert v["modules"] == {"a": {"class": "ok", "reason": ""}}


def test_unparsable_summary_or_baseline_json_degrades(tmp_path, capsys):
    d, diff = _write_report(tmp_path, merged(a=mod(run_exit_code=1)), [])
    (d / "summary.json").write_text("{not json")
    out = tmp_path / "gh_output"
    assert ps.main(["--merged-dir", str(d), "--diff-file", str(diff),
                    "--github-output", str(out)]) == 0
    assert "state=neutral\n" in out.read_text()
    assert "merged mutation report unavailable" in capsys.readouterr().out
    (d / "summary.json").write_text(json.dumps(merged(a=mod(run_exit_code=1))))
    base = tmp_path / "base.json"
    base.write_text("{not json")
    out2 = tmp_path / "gh_output2"
    assert ps.main(["--merged-dir", str(d), "--diff-file", str(diff),
                    "--baseline-summary", str(base), "--github-output", str(out2)]) == 0
    assert "state=failure\n" in out2.read_text()
    assert "baseline summary unreadable" in capsys.readouterr().out


def _run_main(tmp_path, name, m, rows=(), baseline=None, diff=True):
    """Run main() over real files; return (annotation lines, verdict JSON, github output)."""
    root = tmp_path / name
    root.mkdir()
    d, dfile = _write_report(root, m, list(rows))
    out = root / "gh_output"
    args = ["--merged-dir", str(d), "--github-output", str(out)]
    if diff:
        args += ["--diff-file", str(dfile)]
    if baseline is not None:
        bfile = root / "base.json"
        bfile.write_text(json.dumps(baseline))
        args += ["--baseline-summary", str(bfile)]
    assert ps.main(args) == 0
    return d, out


def _split(stdout):
    lines = stdout.splitlines()
    start = lines.index("{")
    return [x for x in lines[:start]], json.loads("\n".join(lines[start:]))


REASON_CASES = [
    # (module summary, rows, baseline, level, exact reason)
    (None, [], None, "warning", "no report: job cancelled, runner or setup error"),
    (mod(skipped_reason="pr_closed_mid_run"), [], None, "warning",
     "skipped (pr_closed_mid_run)"),
    (mod(run_exit_code=-1), [], None, "warning", "timeout or cancelled"),
    (mod(run_exit_code=137), [], None, "warning", "killed by signal (exit 137)"),
    (mod(run_exit_code=-9), [], None, "warning", "killed by signal (exit -9)"),
    (mod(counts_backend={"error": 2}), [], None, "warning",
     "gremlin ERROR results (tool errors)"),
    (mod(failure_reasons=["RAW_INVALID"]), [], None, "warning", "invalid raw report"),
    (mod(run_exit_code=1), [], merged(a=mod(run_exit_code=1)), "warning",
     "module run fails on main too (exit 1)"),
    (mod(run_exit_code=2), [], None, "error", "module run failed (exit 2)"),
    (mod(), [row("a", "engine/v2/a.py", 5)], None, "error",
     "1 survivor(s) on changed lines: engine/v2/a.py:5"),
]


@pytest.mark.parametrize("summary_mod,rows,baseline,level,reason", REASON_CASES)
def test_published_reason_text_in_annotation_and_json(tmp_path, capsys, summary_mod, rows,
                                                      baseline, level, reason):
    m = {"modules": {} if summary_mod is None else {"a": summary_mod},
         "expected_modules": ["a"], "complete": True, "tool_error": False,
         "failure_reasons": []}
    _run_main(tmp_path, "case", m, rows, baseline)
    notes, verdict = _split(capsys.readouterr().out)
    cls = "pr_owned" if level == "error" else "infra"
    assert verdict["modules"]["a"] == {"class": cls, "reason": reason}
    suffix = "" if level == "error" else " (infra, not PR-owned)"
    assert notes == [f"::{level}::a: {reason}{suffix}"]
    assert verdict["state"] == ("failure" if level == "error" else "neutral")


def test_published_descriptions_and_degradation_messages_are_exact(tmp_path, capsys):
    # clean, infra-only and PR-owned descriptions
    _, out = _run_main(tmp_path, "clean", merged(a=mod(), b=mod()))
    assert "description=gremlins: no PR-owned problems in 2 module(s)\n" in out.read_text()
    _, out = _run_main(tmp_path, "infra", merged(a=mod(run_exit_code=-1), b=mod()))
    assert ("description=gremlins: report incomplete or tool error (1 infra warning(s)); "
            "not a pass\n") in out.read_text()
    _, out = _run_main(tmp_path, "owned", merged(a=mod(run_exit_code=1), b=mod(run_exit_code=-1)))
    assert ("description=gremlins: PR-owned problem in a; 1 infra warning(s)\n"
            in out.read_text())
    capsys.readouterr()
    # unavailable report
    out = tmp_path / "gh_unavailable"
    assert ps.main(["--merged-dir", str(tmp_path / "nowhere"), "--github-output", str(out)]) == 0
    assert out.read_text() == ("state=neutral\n"
                               "description=gremlins: merged report unavailable (infra); "
                               "no PR-owned result\n")
    notes, verdict = _split(capsys.readouterr().out)
    assert notes == ["::warning::merged mutation report unavailable; "
                     "not reporting red for infra"]
    assert verdict == {"state": "neutral", "pr_owned": [], "infra": ["report"], "modules": {},
                       "merge_issues": [], "attribution_problems": [],
                       "description": "gremlins: merged report unavailable (infra); "
                                      "no PR-owned result"}


def test_exact_degradation_warnings(tmp_path, capsys):
    d, diff = _write_report(tmp_path, merged(a=mod()), [])
    out = tmp_path / "gh_output"
    base = tmp_path / "missing_base.json"
    ps.main(["--merged-dir", str(d), "--baseline-summary", str(base),
             "--github-output", str(out)])
    notes, verdict = _split(capsys.readouterr().out)
    assert notes == [
        "::warning::PR diff unavailable; survivors cannot be attributed to the PR",
        "::warning::main baseline summary unreadable; no failure is excused as pre-existing"]
    assert set(verdict) == {"state", "description", "pr_owned", "infra", "modules",
                            "merge_issues", "attribution_problems"}
    (d / "results.jsonl").write_text("{bad\n")
    ps.main(["--merged-dir", str(d), "--diff-file", str(diff), "--github-output", str(out)])
    notes, _ = _split(capsys.readouterr().out)
    assert notes == ["::warning::1 malformed results.jsonl row(s) ignored"]
    (d / "results.jsonl").unlink()
    ps.main(["--merged-dir", str(d), "--diff-file", str(diff), "--github-output", str(out)])
    notes, _ = _split(capsys.readouterr().out)
    assert notes == ["::warning::results.jsonl unreadable; "
                     "survivors cannot be attributed to the PR"]


def test_survivor_reason_lists_first_five_sorted_hits_with_full_count():
    changed = {"engine/v2/a.py": set(range(1, 9))}
    rows = [row("a", "engine/v2/a.py", n) for n in (8, 3, 1, 7, 2, 6, 5)]
    v = ps.classify(merged(a=mod()), rows, changed)
    assert v["modules"]["a"]["reason"] == (
        "7 survivor(s) on changed lines: engine/v2/a.py:1, engine/v2/a.py:2, "
        "engine/v2/a.py:3, engine/v2/a.py:5, engine/v2/a.py:6")


def test_all_infra_run_is_not_red_but_never_worded_as_a_pass(tmp_path):
    # Every scheduled job died: no summary entry for any module.
    m = {"modules": {}, "expected_modules": ["a", "b"]}
    v = verdict(m)
    assert v["state"] == "neutral" and v["pr_owned"] == []
    assert v["description"] == "gremlins: no module measured (2 infra warning(s)); not a pass"
    # One measured module changes the wording back.
    v = verdict({"modules": {"a": mod()}, "expected_modules": ["a", "b"]})
    assert v["state"] == "success"
    assert v["description"] == "gremlins: no PR-owned problems; 1 infra warning(s)"


def test_neutral_state_only_when_nothing_was_measured():
    assert verdict(merged(a=mod(run_exit_code=-1)))["state"] == "neutral"
    assert verdict(merged(a=mod(run_exit_code=-1), b=mod()))["state"] == "neutral"
    assert verdict(merged(a=mod(run_exit_code=-1), b=mod(run_exit_code=1)))["state"] == "failure"
    assert verdict({"modules": {}, "expected_modules": []})["state"] == "success"


def _merged_with(**extra):
    m = merged(a=mod(), b=mod())
    m.update(extra)
    return m


def test_every_merged_report_shape_maps_to_a_state(capsys):
    """Self-review table: each input the merged report can carry -> resulting state."""
    clean = ps.classify(_merged_with(), [], {})
    assert clean["state"] == "success" and clean["merge_issues"] == []
    # merge-level tool error with every module looking clean: never success
    for extra in ({"tool_error": True}, {"complete": False},
                  {"tool_error": True, "complete": False,
                   "failure_reasons": ["RUN_PROVENANCE_MISMATCH: x"]}):
        v = ps.classify(_merged_with(**extra), [], {})
        assert v["state"] == "neutral", extra
        assert v["description"] == ("gremlins: report incomplete or tool error "
                                    "(0 infra warning(s)); not a pass")
    # complete=True / tool_error=False explicitly: still success
    assert ps.classify(_merged_with(tool_error=False, complete=True), [], {})["state"] == "success"
    # merge-level reasons are listed, per-module reasons are not
    v = ps.classify(_merged_with(
        tool_error=True,
        failure_reasons=["a: ERROR_RESULTS", "RUN_PROVENANCE_MISMATCH: x", "MISSING_MODULES: y",
                         "DUPLICATE_MODULES: z", "UNEXPECTED_MODULES: w", "NO_MODULE_REPORTS: v"]),
        [], {})
    assert v["merge_issues"] == ["RUN_PROVENANCE_MISMATCH: x", "MISSING_MODULES: y",
                                 "DUPLICATE_MODULES: z", "UNEXPECTED_MODULES: w",
                                 "NO_MODULE_REPORTS: v"]
    # PR-owned still wins over a merge tool error
    v = ps.classify(_merged_with(tool_error=True), [row("a", "engine/v2/a.py", 5)], CHANGED)
    assert v["state"] == "failure" and v["pr_owned"] == ["a"]
    assert v["description"].startswith("gremlins: PR-owned problem in a")
    # all modules infra: the nothing-measured wording wins over the tool-error wording
    v = ps.classify({"modules": {}, "expected_modules": ["a"], "tool_error": True}, [], {})
    assert v["state"] == "neutral" and "no module measured" in v["description"]
    # a module that reports tool_error is infra even with no other reason
    v = ps.classify(merged(a=mod(tool_error=True), b=mod()), [], {})
    assert v["modules"]["a"] == {"class": "infra", "reason": "module reported a tool error"}
    assert v["state"] == "neutral"  # a's tool error is also the merged summary's tool error
    # skipped placeholder: merge reports complete=False -> never success from the tool
    v = ps.classify(_merged_with(complete=False, skipped_modules={"a": "pr_closed_mid_run"}),
                    [], {})
    assert v["state"] == "neutral"
    # empty plan with a tool error is still neutral, not success
    assert ps.classify({"modules": {}, "tool_error": True}, [], {})["state"] == "neutral"


def test_merge_issues_are_warned_by_main(tmp_path, capsys):
    m = _merged_with(tool_error=True, failure_reasons=["RUN_PROVENANCE_MISMATCH: x"])
    d, diff = _write_report(tmp_path, m, [])
    out = tmp_path / "gh_output"
    assert ps.main(["--merged-dir", str(d), "--diff-file", str(diff),
                    "--github-output", str(out)]) == 0
    assert "state=neutral\n" in out.read_text()
    notes, _ = _split(capsys.readouterr().out)
    assert notes == ["::warning::merge: RUN_PROVENANCE_MISMATCH: x"]


def _case(tmp_path, name, survivor, diff_mode, rows_mode):
    root = tmp_path / name
    root.mkdir()
    d = root / "m"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(merged(a=mod())))
    good = json.dumps(row("a", "engine/v2/a.py", 5 if survivor else 99)) + "\n"
    if rows_mode == "ok":
        (d / "results.jsonl").write_text(good)
    elif rows_mode == "malformed":
        (d / "results.jsonl").write_text(good + "{bad\n")
    elif rows_mode == "undecodable":
        (d / "results.jsonl").write_bytes(b"\xff\xfe\n")
    # rows_mode == "missing": no file
    args = ["--merged-dir", str(d), "--github-output", str(root / "out")]
    if diff_mode == "ok":
        (root / "pr.diff").write_text(DIFF)
        args += ["--diff-file", str(root / "pr.diff")]
    elif diff_mode == "undecodable":
        (root / "pr.diff").write_bytes(b"+++ b/engine/v2/a.py\n@@ -1 +5 @@\n\xff\n")
        args += ["--diff-file", str(root / "pr.diff")]
    elif diff_mode == "hunks_without_header":
        (root / "pr.diff").write_text("@@ -1 +5 @@\n+x\n")
        args += ["--diff-file", str(root / "pr.diff")]
    elif diff_mode == "missing_file":
        args += ["--diff-file", str(root / "absent.diff")]
    # diff_mode == "no_flag": no --diff-file
    assert ps.main(args) == 0
    return (root / "out").read_text()


def test_unattributable_survivors_are_never_success(tmp_path):
    """Audit: every combination of diff/result-row degradation vs the oracle."""
    n = 0
    for survivor in (True, False):
        for diff_mode in ("ok", "no_flag", "missing_file", "undecodable", "hunks_without_header"):
            for rows_mode in ("ok", "malformed", "undecodable", "missing"):
                n += 1
                text = _case(tmp_path, f"c{n}", survivor, diff_mode, rows_mode)
                attributable = diff_mode == "ok" and rows_mode == "ok"
                rows_have_survivor = survivor and rows_mode in ("ok", "malformed")
                if rows_have_survivor and diff_mode == "ok":
                    want = "failure"  # the good row is a survivor on a changed line
                elif attributable:
                    want = "success"
                else:
                    want = "neutral"
                assert f"state={want}\n" in text, (survivor, diff_mode, rows_mode, text)


def test_degraded_inputs_other_than_attribution_never_produce_success(tmp_path):
    # unreadable summary, malformed summary, merge tool error, incomplete run: neutral
    d = tmp_path / "s1"
    d.mkdir()
    (d / "summary.json").write_text("{not json")
    cases = (("s2", {"modules": []}), ("s3", _merged_with(tool_error=True)),
             ("s4", _merged_with(complete=False)), ("s5", {}),
             ("s6", {"modules": None}), ("s7", {"modules": {}}),
             ("s9", {"expected_modules": ["a"]}),
             ("s10", {"modules": {"a": {}}, "expected_modules": "a"}))
    for name, summary in cases:
        dd = tmp_path / name
        dd.mkdir()
        (dd / "summary.json").write_text(json.dumps(summary))
        (dd / "results.jsonl").write_text("")
        (dd / "pr.diff").write_text(DIFF)
    for name in ("s1", *(name for name, _ in cases)):
        out = tmp_path / f"{name}.out"
        assert ps.main(["--merged-dir", str(tmp_path / name), "--diff-file",
                        str(tmp_path / name / "pr.diff"), "--github-output", str(out)]) == 0
        assert "state=neutral\n" in out.read_text(), name
    # A missing/unreadable baseline only stops excusing failures: it can add red, never green.
    (tmp_path / "b").mkdir()
    d5, diff5 = _write_report(tmp_path / "b", merged(a=mod(run_exit_code=1)), [])
    out = tmp_path / "b.out"
    ps.main(["--merged-dir", str(d5), "--diff-file", str(diff5), "--baseline-summary",
             str(tmp_path / "nope.json"), "--github-output", str(out)])
    assert "state=failure\n" in out.read_text()


def test_every_degraded_shape_is_never_success(tmp_path):
    """One degradation per case: a valid clean report vs every invalid shape."""
    base = merged(a=mod())
    extra = row("a", "engine/v2/a.py", 99, "killed")
    failures = []
    counter = {"n": 0}

    def run(label, want, summary, rows=(), raw_lines=None, diff_mode="ok"):
        counter["n"] += 1
        root = tmp_path / f"c{counter['n']}"
        d = root / "m"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(summary))
        if raw_lines is None:
            (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        else:
            (d / "results.jsonl").write_text("".join(x + "\n" for x in raw_lines))
        out = root / "out"
        args = ["--merged-dir", str(d), "--github-output", str(out)]
        diff = root / "pr.diff"
        if diff_mode == "ok":
            diff.write_text(DIFF)
            args += ["--diff-file", str(diff)]
        elif diff_mode == "missing":
            args += ["--diff-file", str(root / "no.diff")]
        elif diff_mode == "undecodable":
            diff.write_bytes(b"\xff\xfe\n")
            args += ["--diff-file", str(diff)]
        elif diff_mode == "hunks_without_header":
            diff.write_text("@@ -1 +5 @@\n+x\n")
            args += ["--diff-file", str(diff)]
        elif diff_mode == "empty":
            diff.write_text("")
            args += ["--diff-file", str(diff)]
        elif diff_mode == "whitespace":
            diff.write_text(" \n\t\n")
            args += ["--diff-file", str(diff)]
        assert ps.main(args) == 0, label
        state = None
        for line in out.read_text().splitlines():
            if line.startswith("state="):
                state = line.split("=", 1)[1]
        if state != want:
            failures.append(f"{label}: state={state}, want {want}")

    # Control: the un-degraded report is a success.
    run("control", "success", copy.deepcopy(base))
    # Summary level: each key missing/ill-typed, or the whole summary replaced.
    for key in ("complete", "tool_error"):
        s = copy.deepcopy(base)
        del s[key]
        run(f"summary: delete {key}", "neutral", s)
        for bad in (None, "true", "false", 0, 1, [], {}):
            s = copy.deepcopy(base)
            s[key] = bad
            run(f"summary: {key}={bad!r}", "neutral", s)
    s = copy.deepcopy(base)
    del s["failure_reasons"]
    run("summary: delete failure_reasons", "neutral", s)
    for bad in (None, "x", {}, 0):
        s = copy.deepcopy(base)
        s["failure_reasons"] = bad
        run(f"summary: failure_reasons={bad!r}", "neutral", s)
    s = copy.deepcopy(base)
    del s["modules"]
    run("summary: delete modules", "neutral", s)
    for bad in (None, [], "x", 5):
        s = copy.deepcopy(base)
        s["modules"] = bad
        run(f"summary: modules={bad!r}", "neutral", s)
    for bad in ("a", 5, {}):
        s = copy.deepcopy(base)
        s["expected_modules"] = bad
        run(f"summary: expected_modules={bad!r}", "neutral", s)
    for whole in ([], "x", None, 5, {}):
        run(f"summary: whole={whole!r}", "neutral", whole)
    # Module entry level (single module a, so a malformed entry is the only infra).
    for key in ("run_exit_code", "counts_backend", "failure_reasons", "tool_error"):
        s = copy.deepcopy(base)
        del s["modules"]["a"][key]
        run(f"module: delete {key}", "neutral", s)
    for key, bads in (("run_exit_code", ("0", 0.0, True, [], {})),
                      ("counts_backend", (None, [], "x", 0)),
                      ("failure_reasons", (None, "x", {})),
                      ("tool_error", (None, "no", 0, 1))):
        for bad in bads:
            s = copy.deepcopy(base)
            s["modules"]["a"][key] = bad
            run(f"module: {key}={bad!r}", "neutral", s)
    # Row level: one extra row with a single bad field, or one raw junk line.
    for field in ("module", "file", "status"):
        r = copy.deepcopy(extra)
        del r[field]
        run(f"row: delete {field}", "neutral", copy.deepcopy(base), rows=[r])
        for bad in (None, 1, "", []):
            r = copy.deepcopy(extra)
            r[field] = bad
            run(f"row: {field}={bad!r}", "neutral", copy.deepcopy(base), rows=[r])
    r = copy.deepcopy(extra)
    del r["line"]
    run("row: delete line", "neutral", copy.deepcopy(base), rows=[r])
    for bad in (None, 0, -1, 1.5, True, "5"):
        r = copy.deepcopy(extra)
        r["line"] = bad
        run(f"row: line={bad!r}", "neutral", copy.deepcopy(base), rows=[r])
    for bad in ("KILLED", "zapped", "error"):
        r = copy.deepcopy(extra)
        r["status"] = bad
        run(f"row: status={bad!r}", "neutral", copy.deepcopy(base), rows=[r])
    for raw in ('"x"', "5", "null", "[]", "{", "{}"):
        run(f"row: raw {raw!r}", "neutral", copy.deepcopy(base), raw_lines=[raw])
    # Diff level: unavailable in every way.
    for mode in ("missing", "undecodable", "hunks_without_header", "empty", "whitespace"):
        run(f"diff: {mode}", "neutral", copy.deepcopy(base), diff_mode=mode)
    # Control: one valid extra row on an untouched line is still a success.
    run("control: valid extra row", "success", copy.deepcopy(base), rows=[extra])
    assert failures == []


def test_valid_shapes_are_accepted(tmp_path):
    """Positive control: every shape the validators accept classifies without degradation."""
    for status in ("killed", "survived", "timeout", "suspicious", "excluded"):
        assert ps._valid_row(row("a", "engine/v2/a.py", 99, status))
    assert ps._valid_summary(merged(a=mod(run_exit_code=None)))
    assert ps.classify_module("a", mod(run_exit_code=None), [], {}, set()) == ("ok", "")
    expected_none = merged(a=mod())
    expected_none["expected_modules"] = None
    assert ps._valid_summary(expected_none)
    expected_omitted = merged(a=mod())
    del expected_omitted["expected_modules"]
    assert ps._valid_summary(expected_omitted)
    planned = {"modules": {}, "expected_modules": ["a"], "complete": True,
               "tool_error": False, "failure_reasons": []}
    assert ps._valid_summary(planned)

    def state_of(name, summary, rows=()):
        _, out = _run_main(tmp_path, name, summary, rows)
        return next(line.split("=", 1)[1] for line in out.read_text().splitlines()
                    if line.startswith("state="))

    for status in ("killed", "survived", "timeout", "suspicious", "excluded"):
        assert state_of(f"row_{status}", merged(a=mod()),
                        [row("a", "engine/v2/a.py", 99, status)]) == "success"
    assert state_of("expected_none", expected_none,
                    [row("a", "engine/v2/a.py", 99)]) == "success"
    assert state_of("expected_omitted", expected_omitted,
                    [row("a", "engine/v2/a.py", 99)]) == "success"
    assert state_of("null_run_exit_code", merged(a=mod(run_exit_code=None)),
                    [row("a", "engine/v2/a.py", 99)]) == "success"
    assert state_of("planned_but_nothing_measured", planned) == "neutral"


def test_row_statuses_match_gremlin_results():
    assert ps._ROW_STATUSES == tuple(gremlin_results.STATUSES)


def test_description_always_carries_the_infra_count():
    long_problem = "p" * 200
    for owned in ([], ["m"]):
        for n in (0, 1, 3):
            for unattributed in (False, True):
                for problem in (long_problem, "short problem"):
                    for unsafe in (False, True):
                        infra = [f"i{k}" for k in range(n)]
                        names = sorted(set(owned + infra + ["ok1"]))
                        desc = ps._description(owned, infra, names, unattributed,
                                               [problem], unsafe)
                        assert len(desc) <= ps.DESC_MAX
                        if n > 0:
                            assert f"{n} infra warning(s)" in desc
    # The concrete bug: an unattributed problem must not drop the infra count.
    m = merged(a=mod(), b=mod())
    del m["modules"]["b"]
    v = ps.classify(m, [], {}, None,
                    ("PR diff unavailable -- survivors not attributed; not a pass",))
    assert v["state"] == "neutral"
    assert "1 infra warning(s)" in v["description"]