"""tools/mutation_results.py (mutmut backend): merge_dirs's expected-module
contract, and issue #64's skipped_reason marker for a module whose ``mutate``
job never ran mutmut at all because the triggering PR closed while it was
queued.

Every module report here is built with mutation_results.py's own
summarize()/write_jsonl(), never hand-typed JSON: a "real report" is
whatever the real function produces for real rows, so a change to that
function's own shape breaks these tests before it breaks the merge.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import mutation_results as mr  # noqa: E402

INFO = {"run_id": "77", "sha": "deadbeef", "ref": "refs/heads/main", "trigger": "workflow_run"}


def real_row(module, file, name, status, **over):
    row = {"schema_version": mr.SCHEMA_VERSION, **INFO, "mode": "full", "module": module,
           "file": file, "function": "f", "line": 3, "mutant_name": name, "status": status,
           "mutmut_status": mr.mutmut_status(0 if status == "survived" else 1),
           "retested_this_run": True, "diff": "-a\n+b\n", "triage": None}
    row.update(over)
    return row


def write_report(out: Path, rows: list[dict], module: str, files: list[str], *,
                 run_exit_code=0, elapsed=5.0, skipped_reason=None) -> dict:
    """A module report directory in exactly the shape export_module() writes:
    real rows plus a real summarize() call (never a hand-built summary.json)."""
    summary = mr.summarize(rows, module, {**INFO, "mode": "full"}, files,
                           run_exit_code=run_exit_code, elapsed_seconds=elapsed,
                           skipped_reason=skipped_reason)
    out.mkdir(parents=True, exist_ok=True)
    mr.write_jsonl(out / "results.jsonl", rows)
    (out / "summary.json").write_text(json.dumps(summary, sort_keys=True))
    return summary


def merge(tmp_path, dirs, *, expected, name="merged"):
    out = tmp_path / name
    code = mr.main(["merge", "--out", str(out), "--expected-modules", expected,
                    *(str(d) for d in dirs)])
    return code, out


def summary_of(out: Path) -> dict:
    return json.loads((out / "summary.json").read_text())


def test_export_summarize_marks_a_skipped_module_incomplete_but_never_a_tool_error(tmp_path):
    """Opus gate on 733fd56: rc=0 and 0 rows are what a real 0-mutant module
    AND a PR-closed skip both look like; skipped_reason is the only signal
    that tells them apart, and merge_dirs (not export/summarize itself) is
    what must act on it -- summarize() just needs to carry the field through
    honestly, the same way it already carries run_exit_code."""
    real = write_report(tmp_path / "real", [], "alpha", ["a.py"])
    assert real["skipped_reason"] is None
    stub = write_report(tmp_path / "stub", [], "beta", ["b.py"],
                        skipped_reason="pr_closed_mid_run")
    assert stub["skipped_reason"] == "pr_closed_mid_run"
    # summarize() alone does not decide complete/tool_error for a module (only
    # merge_dirs does, for the mutmut backend) -- both reports look identical
    # apart from the marker.
    assert real["total"] == stub["total"] == 0 and real["score"] == stub["score"] is None


def test_merge_treats_a_skipped_module_as_incomplete_never_a_tool_error(tmp_path):
    """The blocker: one real module (alpha, one killed mutant) plus one
    PR-closed stub (beta) must merge to complete=false, a withheld score,
    skipped_modules naming beta, and -- critically -- tool_error=false, so
    report_status never posts a false green status on a merge race, but also
    never a false red one that looks like a broken workflow."""
    a_rows = [real_row("alpha", "a.py", "m1", "killed")]
    write_report(tmp_path / "alpha", a_rows, "alpha", ["a.py"])
    write_report(tmp_path / "beta", [], "beta", ["b.py"], skipped_reason="pr_closed_mid_run")
    code, out = merge(tmp_path, [tmp_path / "alpha", tmp_path / "beta"],
                      expected=json.dumps(["alpha", "beta"]))
    assert code == 0  # NOT a tool error: report must not go red on a merge race
    m = summary_of(out)
    assert m["complete"] is False  # NOT a valid full run either: report must not go green
    assert m["tool_error"] is False
    assert m["score"] is None  # counts stay auditable, but no number no real run made
    assert m["killed"] == 1 and m["checked"] == 1  # alpha's real counts
    assert m["skipped_modules"] == {"beta": "pr_closed_mid_run"}
    assert set(m["modules"]) == {"alpha", "beta"}  # present: MISSING_MODULES must not fire
    assert m["module_contract"]["missing"] == [] and m["module_contract"]["complete_set"] is True
    assert any(r.startswith("PR_CLOSED_SKIPPED") and "beta" in r for r in m["failure_reasons"])
    md = (out / "summary.md").read_text()
    assert "skipped" in md.lower() and "INCOMPLETE" in md


def test_merge_a_fully_skipped_run_still_withholds_its_score(tmp_path):
    """Every scheduled module got stubbed (an extreme merge race): still not
    a tool error, still not a complete measurement."""
    write_report(tmp_path / "alpha", [], "alpha", ["a.py"], skipped_reason="pr_closed_mid_run")
    write_report(tmp_path / "beta", [], "beta", ["b.py"], skipped_reason="pr_closed_mid_run")
    code, out = merge(tmp_path, [tmp_path / "alpha", tmp_path / "beta"],
                      expected=json.dumps(["alpha", "beta"]))
    assert code == 0
    m = summary_of(out)
    assert m["complete"] is False and m["tool_error"] is False and m["score"] is None
    assert set(m["skipped_modules"]) == {"alpha", "beta"}


def test_a_real_full_run_with_no_skips_is_unaffected(tmp_path):
    """skipped_modules must be empty (not merely falsy-omitted) and complete
    stays true for the ordinary, no-skip case -- the existing MISSING_MODULES
    contract this sits alongside must keep working exactly as before."""
    a_rows = [real_row("alpha", "a.py", "m1", "killed")]
    b_rows = [real_row("beta", "b.py", "m2", "survived")]
    write_report(tmp_path / "alpha", a_rows, "alpha", ["a.py"])
    write_report(tmp_path / "beta", b_rows, "beta", ["b.py"])
    code, out = merge(tmp_path, [tmp_path / "alpha", tmp_path / "beta"],
                      expected=json.dumps(["alpha", "beta"]))
    assert code == 0
    m = summary_of(out)
    assert m["complete"] is True and m["tool_error"] is False
    assert m["skipped_modules"] == {}
    assert m["score"] == 0.5
