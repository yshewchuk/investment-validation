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
    assert verdict["description"] == "gremlins: PR-owned problem: 1 surviving mutant(s) in engine/v2/a.py"
    assert verdict["reasons"] == ["1 surviving mutant(s) in engine/v2/a.py"]
    assert verdict["state"] == "failure"
    for rc in (1, 127):
        summary = clean("a")
        summary["modules"]["a"] = mod(run_exit_code=rc)
        verdict = ps.classify(summary, [], {"engine/v2/z.py"})
        assert verdict["state"] == "neutral", rc
        assert verdict["description"] == (
            f"gremlins: not a clean measured run (module a exited {rc}); not a pass"), rc
        assert verdict["reasons"][0] == f"module a exited {rc}", rc
        verdict = ps.classify(summary, [row("a", "engine/v2/a.py")],
                              {"engine/v2/a.py"})
        assert verdict["state"] == "neutral", rc


def test_two_survivors_in_one_file_are_counted(tmp_path):
    verdict = ps.classify(clean("a"), [row("a", "engine/v2/a.py"),
                                       row("a", "engine/v2/a.py")],
                          {"engine/v2/a.py"})
    assert verdict["state"] == "failure"
    assert verdict["description"] == (
        "gremlins: PR-owned problem: 2 surviving mutant(s) in engine/v2/a.py")
    assert verdict["reasons"] == ["2 surviving mutant(s) in engine/v2/a.py"]
    argv, out = _report(tmp_path, clean("a"),
                        [json.dumps(row("a", "engine/v2/a.py"))] * 2,
                        ["engine/v2/a.py"])
    assert ps.main(argv) == 0
    assert ("description=gremlins: PR-owned problem: 2 surviving mutant(s) in "
            "engine/v2/a.py\n") in out.read_text()


def test_cli_attribution_is_by_changed_file(tmp_path):
    argv, out = _report(tmp_path, clean("a"), [json.dumps(row("a", "engine/v2/a.py"))],
                        ["engine/v2/b.py"], subdir="other")
    assert ps.main(argv) == 0
    assert "state=success\n" in out.read_text()
    argv, out = _report(tmp_path, clean("a"), [json.dumps(row("a", "engine/v2/a.py"))],
                        ["engine/v2/a.py", "engine/v2/b.py"], subdir="both")
    assert ps.main(argv) == 0
    assert "state=failure\n" in out.read_text()


def test_integrity_comes_before_attribution():
    for summary in (None, {"modules": None},
                    {"modules": {"a": mod()}, "expected_modules": ["a", "b"],
                     "complete": True, "tool_error": False, "failure_reasons": []}):
        verdict = ps.classify(summary, [row("a", "engine/v2/a.py")],
                              {"engine/v2/a.py"})
        assert verdict["state"] == "neutral", summary
    verdict = ps.classify(clean("a"), None, {"engine/v2/a.py"})
    assert verdict["state"] == "neutral"
    verdict = ps.classify({**clean("a"), "complete": False},
                          [row("a", "engine/v2/a.py")], {"engine/v2/a.py"})
    assert verdict["state"] == "neutral"
    verdict = ps.classify({**clean("a"), "modules": {"a": mod(run_exit_code=1)}},
                          [row("a", "engine/v2/a.py")], None)
    assert verdict["state"] == "neutral"
    for summary in ({**clean("a"), "tool_error": True},
                    {**clean("a"), "failure_reasons": ["X"]},
                    {**clean("a"), "modules": {"a": mod(counts_backend={"error": 2})}},
                    {**clean("a"), "modules": {"a": mod(skipped_reason="x")}}):
        verdict = ps.classify(summary, [row("a", "engine/v2/a.py")],
                              {"engine/v2/a.py"})
        assert verdict["state"] == "neutral"


def test_success_cases():
    verdict = ps.classify(clean("a"), [], {"engine/v2/z.py"})
    assert verdict["state"] == "success" and verdict["description"].endswith("1 module(s)")
    verdict = ps.classify(clean("a", "b", "c"), [], {"engine/v2/z.py"})
    assert verdict["state"] == "success" and verdict["description"].endswith("3 module(s)")
    assert (ps.classify(clean("a", "b", "c"), [], {"engine/v2/z.py"})["description"]
            == "gremlins: no PR-owned problems in 3 module(s)")
    verdict = ps.classify(clean("a"), [row("a", "engine/v2/other.py")], {"engine/v2/a.py"})
    assert verdict["state"] == "success"
    for status in ("killed", "timeout", "suspicious", "excluded"):
        verdict = ps.classify(clean("a"), [row("a", "engine/v2/a.py", status)],
                              {"engine/v2/a.py"})
        assert verdict["state"] == "success", status
    planned_unknown = {"modules": {"a": mod()}, "expected_modules": None,
                       "complete": True, "tool_error": False, "failure_reasons": []}
    planned_unknown_verdict = ps.classify(planned_unknown, [], {"engine/v2/z.py"})
    assert planned_unknown_verdict["state"] == "neutral"
    assert planned_unknown_verdict["reasons"][0] == "planned modules unknown"


def test_non_survivor_statuses_load_and_stay_success(tmp_path):
    argv, out = _report(tmp_path, clean("a"),
                        [json.dumps(row("a", "engine/v2/a.py", s))
                         for s in ("killed", "timeout", "suspicious", "excluded")],
                        ["engine/v2/a.py"])
    assert ps.main(argv) == 0
    assert "state=success" in out.read_text()


def _malformed_cases():
    cases = []

    def add_exp(id_, fn):
        cases.append((id_, fn))

    for id_, setup in (
            ("expected_modules_deleted", lambda s: s.pop("expected_modules")),
            ("expected_modules_none", lambda s: s.__setitem__("expected_modules", None)),
            ("expected_modules_str", lambda s: s.__setitem__("expected_modules", "a")),
            ("expected_modules_int_items", lambda s: s.__setitem__("expected_modules", [1]))):
        add_exp(id_, setup)

    def field_case(field, value, id_):
        def fn(s):
            m = s["modules"]["a"]
            if value is _DELETED:
                m.pop(field)
            else:
                m[field] = value
        return id_, fn

    for field in ("complete", "tool_error", "failure_reasons"):
        cases.append(field_case(field, _DELETED, f"{field}_missing"))
        cases.append(field_case(field, None, f"{field}_none"))
        cases.append(field_case(field, "x", f"{field}_wrong_type"))
    cases.append(field_case("complete", False, "complete_false"))
    cases.append(field_case("tool_error", True, "tool_error_true"))
    cases.append(field_case("failure_reasons", ["X"], "failure_reasons_nonempty"))

    def cb_case(cb, id_):
        def fn(s):
            s["modules"]["a"]["counts_backend"] = cb
        return id_, fn

    cases.append(cb_case({}, "counts_backend_error_missing"))
    cases.append(cb_case({"error": None}, "counts_backend_error_none"))
    cases.append(cb_case({"error": "0"}, "counts_backend_error_str"))
    cases.append(cb_case({"error": True}, "counts_backend_error_bool"))
    return cases


class _Deleted:
    pass


_DELETED = _Deleted()


@pytest.mark.parametrize("id_,mutate", _malformed_cases(),
                         ids=[c[0] for c in _malformed_cases()])
def test_malformed_planned_or_module_fields_are_neutral(id_, mutate):
    summary = clean("a")
    mutate(summary)
    verdict = ps.classify(summary, [], {"engine/v2/z.py"})
    assert verdict["state"] == "neutral", id_


def _strict_type_cases():
    def classify_case(mutate):
        def build(tmp_path):
            summary = clean("a")
            mutate(summary)
            return ps.classify(summary, [], {"engine/v2/z.py"})
        return build

    def set_module(field, value):
        return lambda s: s["modules"].__setitem__("a", mod(**{field: value}))

    cases = [
        ("expected_modules_empty",
         classify_case(lambda s: s.__setitem__("expected_modules", []))),
        ("counts_backend_error_float_zero",
         classify_case(set_module("counts_backend", {"error": 0.0}))),
        ("counts_backend_error_bool_false",
         classify_case(set_module("counts_backend", {"error": False}))),
        ("module_complete_int", classify_case(set_module("complete", 1))),
        ("module_tool_error_int", classify_case(set_module("tool_error", 0))),
        ("row_module_unexpected_str",
         lambda tmp_path: ps.classify(clean("a"),
                                      [row("zzz", "engine/v2/a.py")],
                                      {"engine/v2/a.py"})),
    ]

    def row_module_int(tmp_path):
        argv, out = _report(tmp_path, clean("a"),
                            [json.dumps({"module": 5, "file": "engine/v2/a.py",
                                         "status": "survived"})],
                            ["engine/v2/a.py"])
        assert ps.main(argv) == 0
        text = out.read_text()
        assert "state=neutral" in text
        return text

    cases.append(("row_module_int", row_module_int))
    return cases


@pytest.mark.parametrize("id_,build", _strict_type_cases(),
                         ids=[c[0] for c in _strict_type_cases()])
def test_strict_types_and_row_modules_are_neutral(id_, build, tmp_path):
    result = build(tmp_path)
    if isinstance(result, dict):
        assert result["state"] == "neutral", id_
        if id_ == "row_module_unexpected_str":
            assert "row module zzz not planned" in result["reasons"], id_
    else:
        assert "state=neutral" in result, id_
        if id_ == "row_module_int":
            assert "result rows unreadable" in result, id_


def _top_level_cases():
    cases = []
    for key in ("complete", "tool_error", "failure_reasons"):
        cases.append((f"{key}-missing", key, _DELETED))
        cases.append((f"{key}-none", key, None))
    cases += [("complete-wrong-str", "complete", "true"),
              ("complete-wrong-int", "complete", 1),
              ("tool_error-wrong-str", "tool_error", "false"),
              ("tool_error-wrong-int", "tool_error", 0),
              ("failure_reasons-wrong-empty-str", "failure_reasons", ""),
              ("failure_reasons-wrong-empty-dict", "failure_reasons", {}),
              ("failure_reasons-wrong-int", "failure_reasons", 0)]
    return cases


@pytest.mark.parametrize("key,value", [c[1:] for c in _top_level_cases()],
                         ids=[c[0] for c in _top_level_cases()])
def test_top_level_summary_fields_must_be_strict(key, value):
    summary = clean("a")
    if value is _DELETED:
        summary.pop(key)
    else:
        summary[key] = value
    verdict = ps.classify(summary, [], {"engine/v2/z.py"})
    assert verdict["state"] == "neutral", key


def test_top_level_tool_error_missing_is_neutral_not_failure():
    summary = clean("a")
    summary.pop("tool_error")
    verdict = ps.classify(summary, [row("a", "engine/v2/a.py")],
                          {"engine/v2/a.py"})
    assert verdict["state"] == "neutral"


def test_four_survivor_files_are_abbreviated():
    files = [f"engine/v2/{c}.py" for c in "abcd"]
    verdict = ps.classify(clean("a"), [row("a", f) for f in files], set(files))
    assert verdict["state"] == "failure"
    assert len(verdict["description"]) <= 140
    assert verdict["description"] == (
        "gremlins: PR-owned problem: 4 surviving mutant(s) in "
        "engine/v2/a.py, engine/v2/b.py, engine/v2/c.py, ...")
    assert verdict["reasons"][0].startswith("4 surviving mutant(s)")


NEUTRAL_CASES = [
    (lambda: {"modules": {}, "expected_modules": []}, "no module measured"),
    (lambda: {"modules": {"b": mod()}, "expected_modules": ["a"], "complete": True,
              "tool_error": False, "failure_reasons": []}, "module a missing"),
    (lambda: {**clean("a"), "modules": {"a": mod(run_exit_code=1)}}, "module a exited 1"),
    (lambda: {**clean("a"), "modules": {"a": mod(run_exit_code=-1)}}, "module a exited -1"),
    (lambda: {**clean("a"), "modules": {"a": mod(run_exit_code=128)}}, "module a exited 128"),
    (lambda: {**clean("a"), "modules": {"a": mod(run_exit_code=137)}}, "module a exited 137"),
    (lambda: {**clean("a"), "modules": {"a": mod(skipped_reason="x")}}, "module a not clean"),
    (lambda: {**clean("a"), "modules": {"a": mod(counts_backend={"error": 2})}},
             "module a not clean"),
    (lambda: {**clean("a"), "tool_error": True}, "tool error"),
    (lambda: {**clean("a"), "complete": False}, "run incomplete"),
    (lambda: {**clean("a"), "failure_reasons": ["RAW_INVALID"]}, "failure reasons"),
    (lambda: {**clean("a"), "failure_reasons": ["RAW_INCONSISTENT"]}, "failure reasons"),
    (lambda: {**clean("a"), "modules": {"a": mod(run_exit_code="0")}}, "module a not clean"),
    (lambda: {**clean("a"), "modules": {"a": mod(run_exit_code=True)}}, "module a not clean"),
    (lambda: {**clean("a"), "modules": {"a": mod(run_exit_code=None)}}, "module a not clean"),
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


def test_only_the_last_module_with_a_null_exit_code_is_neutral():
    summary = clean("a", "b", "c")
    summary["modules"]["c"] = mod(run_exit_code=None)
    verdict = ps.classify(summary, [], {"engine/v2/z.py"})
    assert verdict["state"] == "neutral"
    assert verdict["reasons"] == ["module c not clean"]


def test_neutral_when_rows_or_changed_files_are_missing():
    verdict = ps.classify(clean("a"), None, {"engine/v2/z.py"})
    assert verdict["state"] == "neutral" and "rows" in verdict["reasons"][0]
    verdict = ps.classify(clean("a"), [], None)
    assert verdict["state"] == "neutral" and "changed files" in verdict["reasons"][0]
    verdict = ps.classify(clean("a"), [], set())
    assert verdict["state"] == "neutral" and "changed files" in verdict["reasons"][0]


def test_missing_module_description_is_exact():
    summary = {"modules": {"b": mod()}, "expected_modules": ["a"], "complete": True,
               "tool_error": False, "failure_reasons": []}
    verdict = ps.classify(summary, [], {"engine/v2/z.py"})
    assert verdict["description"] == "gremlins: not a clean measured run (module a missing); not a pass"


def test_description_fits_the_limit():
    modules = {f"module_{i}_with_a_long_name": mod(run_exit_code=1) for i in range(30)}
    verdict = ps.classify({"modules": modules, "expected_modules": sorted(modules),
                           "complete": True, "tool_error": True,
                           "failure_reasons": ["x"]}, [], {"engine/v2/z.py"})
    assert verdict["state"] == "neutral" and len(verdict["description"]) <= 140
    assert verdict["description"]
    assert verdict["description"].endswith("; not a pass")
    assert verdict["description"].startswith("gremlins: not a clean measured run (")
    bad = {f"module_{i}_with_a_long_name": mod(skipped_reason="x") for i in range(30)}
    verdict = ps.classify({"modules": bad, "expected_modules": sorted(bad), "complete": True,
                           "tool_error": False, "failure_reasons": []}, [], {"engine/v2/z.py"})
    assert verdict["state"] == "neutral" and len(verdict["description"]) <= 140
    assert verdict["description"].endswith("; not a pass")


def test_cut_truncates_at_exactly_the_limit():
    files = [f"engine/v2/{c}_" + "x" * 60 + ".py" for c in "abc"]
    rows = [row("a", f) for f in files]
    verdict = ps.classify(clean("a"), rows, set(files))
    assert verdict["state"] == "failure"
    assert len(verdict["description"]) == 140
    assert verdict["description"].endswith("...")
    full = "gremlins: PR-owned problem: 3 surviving mutant(s) in " + ", ".join(sorted(files))
    assert verdict["description"] == full[:137] + "..."


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


def test_neutral_description_is_exact_and_exported(tmp_path):
    argv, out = _report(tmp_path, {**clean("a"), "complete": False}, [],
                        ["engine/v2/a.py"])
    assert ps.main(argv) == 0
    expected = "gremlins: not a clean measured run (run incomplete); not a pass"
    assert f"description={expected}\n" in out.read_text()
    assert ps.classify({**clean("a"), "complete": False}, [],
                       {"engine/v2/a.py"})["description"] == expected


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
    changed = tmp_path / "changed.txt"
    changed.write_text("engine/v2/a.py\n")
    out = tmp_path / "gh_output"
    assert ps.main(["--merged-dir", str(merged), "--changed-files", str(changed),
                    "--github-output", str(out)]) == 0
    assert "state=neutral\n" in out.read_text()
    out_text = out.read_text()
    cap_text = capsys.readouterr().out
    assert "merged report unavailable" in out_text or "merged report unavailable" in cap_text
    assert "::warning::" in cap_text


def test_cli_missing_summary_is_neutral(tmp_path, capsys):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "results.jsonl").write_text("")
    changed = tmp_path / "changed.txt"
    changed.write_text("engine/v2/a.py\n")
    out = tmp_path / "gh_output"
    assert ps.main(["--merged-dir", str(merged), "--changed-files", str(changed),
                    "--github-output", str(out)]) == 0
    assert "state=neutral\n" in out.read_text()
    assert "merged report unavailable" in capsys.readouterr().out


def test_cli_malformed_rows_are_neutral(tmp_path):
    bad_rows = ("{not json",
                json.dumps({"module": "a", "file": "f", "status": "zapped"}),
                json.dumps({"module": "a", "status": "survived"}),
                json.dumps({"file": "f", "status": "survived"}),
                json.dumps({"module": "a", "file": 7, "status": "survived"}),
                "[1, 2]",
                '"x"')
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


def test_one_malformed_row_makes_the_whole_report_neutral(tmp_path):
    survivor = json.dumps(row("a", "engine/v2/a.py"))
    argv, out = _report(tmp_path, clean("a"), [survivor, "{not json"], ["engine/v2/a.py"],
                        subdir="with_junk")
    assert ps.main(argv) == 0
    assert "state=neutral\n" in out.read_text()
    argv, out = _report(tmp_path, clean("a"), [survivor], ["engine/v2/a.py"], subdir="control")
    assert ps.main(argv) == 0
    assert "state=failure\n" in out.read_text()


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
