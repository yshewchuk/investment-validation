"""PR mutation status: PR-owned (red) versus infra (warning) classification.

The unit tests feed ``classify`` fixture summaries; the CLI tests run the real
``main`` over real files in ``tmp_path`` (summary.json, results.jsonl, diff,
baseline, ``$GITHUB_OUTPUT``), so file reading and output writing are not stubbed.

# packages: foundation, ops
"""
from __future__ import annotations

import json

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
    base = {"run_exit_code": 0, "skipped_reason": None, "counts_backend": {"error": 0},
            "failure_reasons": [], "complete": True, "tool_error": False}
    base.update(kw)
    return base


def row(module, file, line, status="survived"):
    return {"module": module, "file": file, "line": line, "status": status}


def merged(**modules):
    return {"modules": modules, "expected_modules": sorted(modules)}


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
    assert v["state"] == "success" and v["modules"]["a"]["class"] == "infra"
    assert "main" in v["modules"]["a"]["reason"]


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
    assert v["state"] == "success"
    assert sorted(v["infra"]) == sorted([*cases, "missing"])
    assert v["pr_owned"] == []
    assert "infra warning" in v["description"]


def test_mixed_result_is_red_and_description_fits_status_limit():
    modules = {f"module_{i}_with_a_long_name": mod(run_exit_code=1) for i in range(30)}
    v = verdict(merged(**modules, infra_one=mod(run_exit_code=-1)))
    assert v["state"] == "failure" and len(v["pr_owned"]) == 30
    assert len(v["description"]) <= ps.DESC_MAX


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


def test_main_reads_baseline_file(tmp_path):
    d, diff = _write_report(tmp_path, merged(a=mod(run_exit_code=1)), [])
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"modules": {"a": mod(run_exit_code=1)}}))
    out = tmp_path / "gh_output"
    ps.main(["--merged-dir", str(d), "--diff-file", str(diff),
             "--baseline-summary", str(base), "--github-output", str(out)])
    assert "state=success\n" in out.read_text()


def test_main_unreadable_inputs_never_red(tmp_path, capsys):
    out = tmp_path / "gh_output"
    rc = ps.main(["--merged-dir", str(tmp_path / "nowhere"), "--github-output", str(out)])
    assert rc == 0 and "state=success\n" in out.read_text()
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
        assert "state=success\n" in out.read_text()
    assert "PR diff unavailable" in capsys.readouterr().out


def test_corrupt_results_jsonl_warns(tmp_path, capsys):
    d, diff = _write_report(tmp_path, merged(a=mod()), [])
    (d / "results.jsonl").write_text("{not json\n")
    out = tmp_path / "gh_output"
    assert ps.main(["--merged-dir", str(d), "--diff-file", str(diff),
                    "--github-output", str(out)]) == 0
    assert "results.jsonl unreadable" in capsys.readouterr().out