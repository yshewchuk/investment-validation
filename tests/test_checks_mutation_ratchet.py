"""checks/mutation_ratchet.py: per-module mutation-score ratchet.

Synthetic rows/fixtures throughout -- no real mutmut run, no data/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from checks import mutation_ratchet as ratchet
import mutation_results as mr  # noqa: E402


def row(module, status, name, diff=""):
    return {"module": module, "status": status, "mutant_name": name, "diff": diff}


def write_triage(tmp_path: Path, *entries: tuple[str, str, str]) -> Path:
    path = tmp_path / "triage.toml"
    body = "\n".join(
        f'[[triage]]\nmutant = "{name}"\nverdict = "{verdict}"\nnote = "n"\n'
        + (f'diff_contains = "{pin}"\n' if pin else "")
        for name, verdict, pin in entries)
    path.write_text(body)
    return path


# -- module_counts -----------------------------------------------------------------

def test_untriaged_survivor_counts_against_the_ratio():
    rows = [row("m", "killed", "m1"), row("m", "survived", "m2")]
    counts = ratchet.module_counts(rows, {})
    assert counts["m"] == {"total": 2, "checked_effective": 2, "killed_effective": 1,
                           "survived_untriaged": 1, "triaged": 0}


def test_live_triage_removes_the_mutant_from_both_sides_of_the_ratio(tmp_path):
    triage_path = write_triage(tmp_path, ("m2", "EQUIVALENT", "same"))
    rows = [row("m", "killed", "m1"), row("m", "survived", "m2", diff="same text")]
    triage = mr.load_triage(triage_path)
    counts = ratchet.module_counts(rows, triage)
    assert counts["m"] == {"total": 2, "checked_effective": 1, "killed_effective": 1,
                           "survived_untriaged": 0, "triaged": 1}


def test_stale_triage_entry_counts_as_untriaged_again(tmp_path):
    triage_path = write_triage(tmp_path, ("m2", "EQUIVALENT", "text that moved"))
    rows = [row("m", "survived", "m2", diff="the function changed since")]
    triage = mr.load_triage(triage_path)
    counts = ratchet.module_counts(rows, triage)
    assert counts["m"]["triaged"] == 0
    assert counts["m"]["survived_untriaged"] == 1
    assert counts["m"]["checked_effective"] == 1


def test_a_rows_own_baked_in_triage_field_is_ignored(tmp_path):
    """module_counts recomputes triage from the live file, never from a row's
    own `triage` field (which may reflect the triage file as it stood when
    some earlier CI run exported results.jsonl)."""
    rows = [{"module": "m", "status": "survived", "mutant_name": "m2", "diff": "x",
            "triage": {"verdict": "EQUIVALENT", "note": "old", "stale": False}}]
    counts = ratchet.module_counts(rows, {})  # nothing in the CURRENT triage file
    assert counts["m"]["triaged"] == 0
    assert counts["m"]["survived_untriaged"] == 1


def test_skipped_mutants_are_excluded_but_counted_in_total():
    rows = [row("m", "skipped", "m1"), row("m", "killed", "m2")]
    counts = ratchet.module_counts(rows, {})
    assert counts["m"] == {"total": 2, "checked_effective": 1, "killed_effective": 1,
                           "survived_untriaged": 0, "triaged": 0}


def test_no_tests_counts_against_the_ratio_like_survived():
    rows = [row("m", "no_tests", "m1"), row("m", "killed", "m2")]
    counts = ratchet.module_counts(rows, {})
    assert counts["m"]["killed_effective"] == 1 and counts["m"]["survived_untriaged"] == 1


def test_build_measurement_reads_the_current_triage_file_not_the_summary(tmp_path):
    triage_path = write_triage(tmp_path, ("m2", "LOW-VALUE", "sur"))
    rows = [row("m", "killed", "m1"), row("m", "survived", "m2", diff="survived text")]
    summary = {"mode": "full", "run_id": "7", "sha": "abc"}
    measured = ratchet.build_measurement(rows, summary, triage_path=triage_path)
    assert measured["schema_version"] == ratchet.SCHEMA_VERSION
    assert measured["mode"] == "full" and measured["run_id"] == "7" and measured["sha"] == "abc"
    assert measured["modules"]["m"]["triaged"] == 1


# -- compare -------------------------------------------------------------------

def module_doc(killed=8, checked=10):
    return {"killed_effective": killed, "checked_effective": checked}


def measurement(mode="full", **modules):
    return {"schema_version": ratchet.SCHEMA_VERSION, "mode": mode,
            "modules": {name: doc for name, doc in modules.items()}}


def test_regression_is_flagged_by_cross_multiplication_not_float_rounding():
    baseline = measurement(m=module_doc(8, 10))       # 0.80
    measured = measurement(m=module_doc(79, 100))      # 0.79 -- a real, small drop
    findings = ratchet.compare(measured, baseline)
    assert findings == [{"code": "MUTATION_REGRESSION", "module": "m",
                         "previous": [8, 10], "current": [79, 100]}]
    assert ratchet.compare(baseline, measured) == []  # the reverse (improvement) is clean


def test_equal_ratio_is_not_a_regression_even_with_different_totals():
    baseline = measurement(m=module_doc(8, 10))
    measured = measurement(m=module_doc(80, 100))
    assert ratchet.compare(measured, baseline) == []


def test_new_module_requires_an_explicit_baseline_not_a_free_pass():
    baseline = measurement()
    measured = measurement(m=module_doc())
    assert ratchet.compare(measured, baseline) == [
        {"code": "MUTATION_NEW_MODULE_BASELINE_REQUIRED", "module": "m"}]


def test_a_module_removed_from_the_matrix_is_not_an_error():
    """Unlike coverage's fixed PACKAGES, mutation modules are a CI-scope
    choice that changes over time (e.g. an `excluded` module); a baseline
    entry for a module no longer measured is simply unused, not a failure."""
    baseline = measurement(m=module_doc(), gone=module_doc())
    measured = measurement(m=module_doc())
    assert ratchet.compare(measured, baseline) == []


def test_incremental_measurement_or_baseline_short_circuits_without_module_noise():
    full = measurement(mode="full", m=module_doc(1, 100))       # would look terrible
    incremental = measurement(mode="incremental", m=module_doc(100, 100))
    assert ratchet.compare(incremental, full) == [{"code": "MUTATION_MEASUREMENT_NOT_FULL"}]
    assert ratchet.compare(full, incremental) == [{"code": "MUTATION_BASELINE_NOT_FULL"}]
    both = ratchet.compare(incremental, incremental)
    assert {f["code"] for f in both} == {"MUTATION_MEASUREMENT_NOT_FULL", "MUTATION_BASELINE_NOT_FULL"}
    assert all("module" not in f for f in both)  # no per-module findings alongside the mode ones


def test_missing_mode_is_treated_as_not_full():
    baseline = {"schema_version": ratchet.SCHEMA_VERSION, "modules": {"m": module_doc()}}
    measured = measurement(m=module_doc())
    assert ratchet.compare(measured, baseline)[0]["code"] == "MUTATION_BASELINE_NOT_FULL"


def test_impossible_or_wrong_typed_counts_cannot_pass():
    baseline = measurement(m=module_doc())
    for killed, checked in ((11, 10), (-1, 10), (True, 10)):
        measured = measurement(m=module_doc(killed, checked))
        assert ratchet.compare(measured, baseline)[0]["code"] == "MUTATION_COUNTS_INVALID"


def test_empty_measured_modules_is_not_a_vacuous_pass():
    """A real full run always covers every enabled module; an empty
    measurement means a broken artifact, not a legitimately narrow one --
    it must never read as ok=true just because there is nothing to loop over."""
    baseline = measurement(m=module_doc())
    empty = measurement()
    assert ratchet.compare(empty, baseline) == [{"code": "MUTATION_MEASUREMENT_EMPTY"}]
    assert ratchet.compare(empty, measurement()) == [{"code": "MUTATION_MEASUREMENT_EMPTY"}]


def test_invalid_baseline_counts_are_refused_not_silently_trusted():
    """A hand-edited or corrupted baseline entry (missing fields, or
    killed > checked) must not KeyError or be cross-multiplied as-is."""
    baseline = measurement(m={"checked_effective": 10})  # no killed_effective
    measured = measurement(m=module_doc())
    assert ratchet.compare(measured, baseline) == [
        {"code": "MUTATION_BASELINE_COUNTS_INVALID", "module": "m"}]
    impossible_baseline = measurement(m=module_doc(11, 10))  # killed > checked
    assert ratchet.compare(measured, impossible_baseline) == [
        {"code": "MUTATION_BASELINE_COUNTS_INVALID", "module": "m"}]


def test_zero_checked_on_either_side_skips_the_module_without_failing():
    baseline = measurement(m=module_doc(0, 0))
    measured = measurement(m=module_doc(0, 0))
    assert ratchet.compare(measured, baseline) == []
    baseline2 = measurement(m=module_doc(5, 5))
    measured2 = measurement(m=module_doc(0, 0))  # every mutant now triaged away
    assert ratchet.compare(measured2, baseline2) == []


# -- CLI / files -----------------------------------------------------------------

def test_read_report_dir_reads_results_and_summary(tmp_path):
    (tmp_path / "results.jsonl").write_text(
        json.dumps({"module": "m", "status": "killed", "mutant_name": "m1", "diff": ""}) + "\n")
    (tmp_path / "summary.json").write_text(json.dumps({"mode": "full", "run_id": "7", "sha": "abc"}))
    rows, summary = ratchet.read_report_dir(tmp_path)
    assert rows == [{"module": "m", "status": "killed", "mutant_name": "m1", "diff": ""}]
    assert summary["mode"] == "full"


def test_cli_requires_exactly_one_of_dir_or_input(tmp_path, capsys):
    with pytest.raises(SystemExit):
        ratchet.main([])
    with pytest.raises(SystemExit):
        ratchet.main(["--dir", str(tmp_path), "--input", str(tmp_path / "x.json")])


def test_cli_reports_missing_baseline_without_a_measurement_source_error(tmp_path):
    input_path = tmp_path / "measured.json"
    input_path.write_text(json.dumps(measurement(m=module_doc())))
    missing_baseline = tmp_path / "no_such_baseline.json"
    rc = ratchet.main(["--input", str(input_path), "--baseline", str(missing_baseline)])
    assert rc == 1


def test_cli_output_writes_the_measurement_never_the_baseline(tmp_path):
    (tmp_path / "results.jsonl").write_text(
        json.dumps({"module": "m", "status": "killed", "mutant_name": "m1", "diff": ""}) + "\n")
    (tmp_path / "summary.json").write_text(json.dumps({"mode": "full", "run_id": "7", "sha": "abc"}))
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(measurement(m=module_doc(1, 1))))
    output_path = tmp_path / "out" / "measured.json"
    empty_triage = tmp_path / "empty_triage.toml"
    empty_triage.write_text("")
    rc = ratchet.main(["--dir", str(tmp_path), "--output", str(output_path),
                       "--baseline", str(baseline_path), "--triage", str(empty_triage)])
    assert rc == 0
    written = json.loads(output_path.read_text())
    assert written["modules"]["m"]["killed_effective"] == 1
    # the baseline file on disk is byte-identical to what we wrote before the run
    assert json.loads(baseline_path.read_text()) == measurement(m=module_doc(1, 1))
