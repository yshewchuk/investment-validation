"""PR mutation status: failure, success and neutral for a PR-owned verdict.

# packages: foundation, ops
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import mutation_pr_status as ps

ROOT = Path(__file__).resolve().parents[3]


def mod(**kw):
    base = {"run_exit_code": 0, "skipped_reason": None, "counts_backend": {"error": 0},
            "failure_reasons": [], "complete": True, "tool_error": False}
    base.update(kw)
    return base


def clean(*names):
    modules = {n: mod() for n in names}
    return {"modules": modules, "expected_modules": sorted(names), "complete": True,
            "tool_error": False, "failure_reasons": []}


def row(module, file, status="survived"):
    return {"module": module, "file": file, "status": status}


def test_failure_cases():
    summary = clean("a")
    changed = {"engine/v2/a.py"}
    verdict = ps.classify(summary, [row("a", "engine/v2/a.py")], changed)
    assert verdict["state"] == "failure" and "engine/v2/a.py" in verdict["description"]
    for rc in (1, 127):
        verdict = ps.classify({"modules": {"a": mod(run_exit_code=1 if rc == 1 else rc)}},
                              [], set())
        assert verdict["state"] == "failure" and "a" in verdict["description"], rc
    degraded = {"modules": {"a": mod(run_exit_code=1, complete=False, tool_error=True)}}
    verdict = ps.classify(degraded, None, None)
    assert verdict["state"] == "failure" and "a" in verdict["description"]


def test_success_cases():
    verdict = ps.classify(clean("a"), [], {"engine/v2/z.py"})
    assert verdict["state"] == "success" and verdict["description"].endswith("1 module(s)")
    verdict = ps.classify(clean("a", "b", "c"), [], {"engine/v2/z.py"})
    assert verdict["state"] == "success" and verdict["description"].endswith("3 module(s)")
    verdict = ps.classify(clean("a"), [row("a", "engine/v2/other.py")], {"engine/v2/a.py"})
    assert verdict["state"] == "success"
    for status in ("killed", "timeout", "suspicious", "excluded"):
        verdict = ps.classify(clean("a"), [row("a", "engine/v2/a.py", status)],
                              {"engine/v2/a.py"})
        assert verdict["state"] == "success", status
    planned = {"modules": {"a": mod()}, "expected_modules": None, "complete": True,
               "tool_error": False, "failure_reasons": []}
    assert ps.classify(planned, [], {"engine/v2/z.py"})["state"] == "success"


NEUTRAL_CASES = [
    (lambda: {"modules": {}, "expected_modules": []}, "no module measured"),
    (lambda: {"modules": {"b": mod()}, "expected_modules": ["a"], "complete": True,
              "tool_error": False, "failure_reasons": []}, "module a not clean"),
    (lambda: {**clean("a"), "modules": {"a": mod(run_exit_code=-1)}}, "module a not clean"),
    (lambda: {**clean("a"), "modules": {"a": mod(run_exit_code=128)}}, "module a not clean"),
    (lambda: {**clean("a"), "modules": {"a": mod(run_exit_code=137)}}, "module a not clean"),
    (lambda: {**clean("a"), "modules": {"a": mod(skipped_reason="x")}}, "module a not clean"),
    (lambda: {**clean("a"), "modules": {"a": mod(counts_backend={"error": 2})}},
             "module a not clean"),
    (lambda: {**clean("a"), "tool_error": True}, "tool error"),
    (lambda: {**clean("a"), "complete": False}, "run incomplete"),
    (lambda: {**clean("a"), "failure_reasons": ["RAW_INVALID"]}, "failure reasons"),
    (lambda: {**clean("a"), "failure_reasons": ["RAW_INCONSISTENT"]}, "failure reasons"),
    (lambda: {**clean("a"), "modules": {"a": mod(run_exit_code="0")}}, "module a not clean"),
    (lambda: {**clean("a"), "modules": {"a": mod(run_exit_code=True)}}, "module a not clean"),
    (lambda: {**clean("a"), "modules": {"a": {"skipped_reason": None,
                                              "counts_backend": {"error": 0},
                                              "failure_reasons": [], "complete": True,
                                              "tool_error": False}}}, "module a not clean"),
    (lambda: {**clean("a"), "modules": {"a": mod(counts_backend=[1])}}, "module a not clean"),
    (lambda: None, "merged report unavailable"),
    (lambda: [], "merged report unavailable"),
    (lambda: "x", "merged report unavailable"),
    (lambda: {}, "no module measured"),
    (lambda: {"modules": None}, "no module measured"),
    (lambda: {"modules": ["a"]}, "no module measured"),
    (lambda: {**clean("a"), "complete": None}, "run incomplete"),
    (lambda: {**clean("a"), "complete": "true"}, "run incomplete"),
]


@pytest.mark.parametrize("make,reason", NEUTRAL_CASES)
def test_everything_else_is_neutral(make, reason):
    verdict = ps.classify(make(), [], {"engine/v2/z.py"})
    assert verdict["state"] == "neutral"
    assert verdict["reasons"] and reason in verdict["reasons"][0]
    assert verdict["description"].startswith("gremlins: not a clean measured run")


def test_neutral_when_rows_or_changed_files_are_missing():
    verdict = ps.classify(clean("a"), None, {"engine/v2/z.py"})
    assert verdict["state"] == "neutral" and "rows" in verdict["reasons"][0]
    verdict = ps.classify(clean("a"), [], None)
    assert verdict["state"] == "neutral" and "changed files" in verdict["reasons"][0]
    verdict = ps.classify(clean("a"), [], set())
    assert verdict["state"] == "neutral" and "changed files" in verdict["reasons"][0]


def test_description_fits_the_limit():
    modules = {f"module_{i}_with_a_long_name": mod(run_exit_code=1) for i in range(30)}
    verdict = ps.classify({"modules": modules, "expected_modules": sorted(modules),
                           "complete": True, "tool_error": True,
                           "failure_reasons": ["x"]}, [], set())
    assert verdict["state"] == "failure" and len(verdict["description"]) <= 140
    bad = {f"module_{i}_with_a_long_name": mod(skipped_reason="x") for i in range(30)}
    verdict = ps.classify({"modules": bad, "expected_modules": sorted(bad), "complete": True,
                           "tool_error": False, "failure_reasons": []}, [], {"engine/v2/z.py"})
    assert verdict["state"] == "neutral" and len(verdict["description"]) <= 140


def _report(tmp_path, summary, rows_lines, changed_lines, subdir="report"):
    root = tmp_path / subdir
    root.mkdir()
    (root / "summary.json").write_text(json.dumps(summary))
    if rows_lines is not None:
        (root / "results.jsonl").write_text("".join(x + "\n" for x in rows_lines))
    changed = root / "changed.txt"
    changed.write_text("".join(x + "\n" for x in changed_lines))
    out = root / "gh_output"
    return ["--merged-dir", str(root), "--changed-files", str(changed),
            "--github-output", str(out)], out


def test_cli_clean_run_success(tmp_path, capsys):
    argv, out = _report(tmp_path, clean("a"), [], ["engine/v2/a.py"])
    assert ps.main(argv) == 0
    assert "state=success\n" in out.read_text()
    assert '"state": "success"' in capsys.readouterr().out


def test_cli_survivor_on_changed_file_fails(tmp_path, capsys):
    argv, out = _report(tmp_path, clean("a"),
                        [json.dumps(row("a", "engine/v2/a.py"))], ["engine/v2/a.py"])
    assert ps.main(argv) == 0
    assert "state=failure\n" in out.read_text()
    assert "::error::" in capsys.readouterr().out


def test_cli_invalid_summary_is_neutral(tmp_path, capsys):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "summary.json").write_text("{not json")
    (merged / "results.jsonl").write_text("")
    out = tmp_path / "gh_output"
    assert ps.main(["--merged-dir", str(merged), "--github-output", str(out)]) == 0
    assert "state=neutral\n" in out.read_text()
    assert "::warning::" in capsys.readouterr().out


def test_cli_malformed_rows_are_neutral(tmp_path):
    bad_rows = ("{not json",
                json.dumps({"module": "a", "file": "f", "status": "zapped"}),
                json.dumps({"module": "a", "status": "survived"}))
    for i, bad in enumerate(bad_rows):
        argv, out = _report(tmp_path, clean("a"), [bad], ["engine/v2/a.py"],
                            subdir=f"bad{i}")
        assert ps.main(argv) == 0
        assert "state=neutral\n" in out.read_text()
    argv, out = _report(tmp_path, clean("a"),
                        [json.dumps(row("a", "engine/v2/a.py", "killed"))],
                        ["engine/v2/a.py"], subdir="control")
    assert ps.main(argv) == 0
    assert "state=success\n" in out.read_text()


def test_cli_absent_or_empty_changed_files_is_neutral(tmp_path, capsys):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "summary.json").write_text(json.dumps(clean("a")))
    (merged / "results.jsonl").write_text("")
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    for i, changed in enumerate((None, tmp_path / "nope.txt", empty)):
        out = tmp_path / f"out{i}"
        argv = ["--merged-dir", str(merged), "--github-output", str(out)]
        if changed is not None:
            argv += ["--changed-files", str(changed)]
        assert ps.main(argv) == 0
        assert "state=neutral\n" in out.read_text()
        assert "changed files unavailable" in capsys.readouterr().out


def test_cli_missing_results_is_neutral(tmp_path):
    argv, out = _report(tmp_path, clean("a"), None, ["engine/v2/a.py"])
    assert ps.main(argv) == 0
    assert "state=neutral\n" in out.read_text()


def test_output_defaults_to_the_env_variable(tmp_path, monkeypatch):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "summary.json").write_text(json.dumps(clean("a")))
    (merged / "results.jsonl").write_text("")
    changed = tmp_path / "changed.txt"
    changed.write_text("engine/v2/a.py\n")
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert ps.main(["--merged-dir", str(merged), "--changed-files", str(changed)]) == 0
    assert "state=success\n" in out.read_text()


def test_unwritable_output_fails_loudly_after_classification(tmp_path, capsys):
    argv, _ = _report(tmp_path, clean("a"), [], ["engine/v2/a.py"], subdir="merged")
    argv[argv.index("--github-output") + 1] = str(tmp_path)
    with pytest.raises(OSError):
        ps.main(argv)
    assert '"state": "success"' in capsys.readouterr().out
    res = subprocess.run([sys.executable, str(ROOT / "tools" / "mutation_pr_status.py"),
                          *argv], capture_output=True, text=True)
    assert res.returncode != 0 and '"state"' in res.stdout and res.stderr
    res = subprocess.run([sys.executable, str(ROOT / "tools" / "mutation_pr_status.py")],
                         capture_output=True, text=True)
    assert res.returncode == 2
