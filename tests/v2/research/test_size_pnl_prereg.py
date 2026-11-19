from __future__ import annotations

import ast
import csv
import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml

from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import source_closure
from experiments import lib, size_pnl_prereg

ROOT = Path(__file__).resolve().parents[3]
PREREG_PATH = ROOT / "experiments" / "size_pnl_prereg.py"
RUN_PATH = ROOT / "experiments" / "EXP-188_size_decision_pnl_rescore" / "run.py"
STAMP = "2026-11-17T20:00:00+00:00"
BASE_SPEC = {
    "id": "EXP-188",
    "title": "t",
    "hypothesis": "h",
    "strategy": "*",
    "grid": {},
    "preregistered_at": STAMP,
}


def _write_spec(case, *, drop=(), **changes):
    spec = {**BASE_SPEC, **changes}
    for key in drop:
        spec.pop(key, None)
    path = case / "spec.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def _digest(spec_path):
    return lib.spec_hash(lib.load_spec(spec_path))


def _planned_row(digest, **overrides):
    row = {
        "id": "EXP-188",
        "spec_hash": digest,
        "date": "2026-11-17",
        "stage": "planned",
        "oos_mean_mid": "",
        "sharpe_trade": "",
        "promoted": "False",
    }
    row.update(overrides)
    return row


def _write_ledger(path, rows, columns=tuple(lib.LEDGER_COLUMNS)):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _run_log_path(case):
    return case / "results" / "run_log.jsonl"


def _write_run_log(case, text):
    path = _run_log_path(case)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _kw(case, spec_path, ledger_path):
    return {"spec_path": spec_path, "ledger_path": ledger_path,
            "run_log_path": _run_log_path(case)}


def _check(case, spec_path, ledger_path):
    return size_pnl_prereg.require_preregistration(**_kw(case, spec_path, ledger_path))


def refusal(callable_):
    with pytest.raises(OpsError) as exc:
        callable_()
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    return exc.value.problem.details["reason"]


# --------------------------------------------------------------------------
# require_preregistration: refusals, one reason code each
# --------------------------------------------------------------------------


def test_missing_ledger_file_is_planned_row_missing(tmp_path):
    spec = _write_spec(tmp_path)
    reason = refusal(lambda: _check(tmp_path, spec, tmp_path / "LEDGER.csv"))
    assert reason == "planned_row_missing"


def test_other_id_or_ran_only_is_planned_row_missing(tmp_path):
    spec = _write_spec(tmp_path)
    digest = _digest(spec)
    other_id = _write_ledger(tmp_path / "other.csv",
                             [_planned_row(digest, id="EXP-999")])
    ran_only = _write_ledger(tmp_path / "ran.csv",
                             [_planned_row(digest, stage="ran")])
    assert refusal(lambda: _check(tmp_path, spec, other_id)) == "planned_row_missing"
    assert refusal(lambda: _check(tmp_path, spec, ran_only)) == "planned_row_missing"


def test_two_planned_rows_are_ambiguous(tmp_path):
    spec = _write_spec(tmp_path)
    digest = _digest(spec)
    ledger = _write_ledger(tmp_path / "LEDGER.csv",
                           [_planned_row(digest), _planned_row(digest)])
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "planned_row_ambiguous"


def test_spec_edited_after_registration_is_hash_mismatch(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    _write_spec(tmp_path, hypothesis="edited after registration")
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "spec_hash_mismatch"


def test_editing_only_preregistered_at_keeps_the_hash(tmp_path):
    spec = _write_spec(tmp_path)
    digest = _digest(spec)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(digest)])
    _write_spec(tmp_path, preregistered_at="2026-11-17T21:00:00+00:00")
    receipt = _check(tmp_path, spec, ledger)
    assert receipt["spec_hash"] == digest
    assert receipt["spec_hash_checked"] is True


def test_happy_path_returns_checked_receipt(tmp_path):
    spec = _write_spec(tmp_path)
    digest = _digest(spec)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(digest)])
    receipt = _check(tmp_path, spec, ledger)
    assert receipt["spec_hash_checked"] is True
    assert receipt["spec_hash"] == digest
    assert receipt["first_run_ts"] is None


def test_missing_spec_is_spec_unreadable(tmp_path):
    spec = tmp_path / "missing.yaml"
    assert refusal(lambda: _check(tmp_path, spec, tmp_path / "LEDGER.csv")) == "spec_unreadable"


def test_non_mapping_spec_is_spec_unreadable(tmp_path):
    spec = tmp_path / "spec.yaml"
    spec.write_text("- one\n- two\n")
    assert refusal(lambda: _check(tmp_path, spec, tmp_path / "LEDGER.csv")) == "spec_unreadable"


def test_wrong_id_is_spec_id_mismatch(tmp_path):
    spec = _write_spec(tmp_path, id="EXP-999")
    assert refusal(lambda: _check(tmp_path, spec, tmp_path / "LEDGER.csv")) == "spec_id_mismatch"


def test_missing_preregistered_at_is_refused(tmp_path):
    spec = _write_spec(tmp_path, drop=("preregistered_at",))
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "preregistered_at_missing"


def test_run_log_earlier_than_stamp_is_preregistered_after_run(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    _write_run_log(tmp_path, json.dumps({"ts": "2026-11-17T19:00:00+00:00"}) + "\n")
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "preregistered_after_run"


def test_run_log_later_than_stamp_passes(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    _write_run_log(tmp_path, json.dumps({"ts": "2026-11-17T21:00:00+00:00"}) + "\n")
    receipt = _check(tmp_path, spec, ledger)
    assert receipt["spec_hash_checked"] is True
    assert receipt["first_run_ts"].startswith("2026-11-17 21:00:00")


def test_malformed_run_log_is_refused(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    _write_run_log(tmp_path, "not json\n")
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "run_log_malformed"


def test_ledger_missing_a_column_is_ledger_malformed(tmp_path):
    spec = _write_spec(tmp_path)
    columns = [c for c in lib.LEDGER_COLUMNS if c != "promoted"]
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))],
                           columns=columns)
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "ledger_malformed"


def test_fifo_spec_is_spec_unreadable_and_returns_promptly(tmp_path):
    spec = tmp_path / "spec.yaml"
    os.mkfifo(spec)
    assert refusal(lambda: _check(tmp_path, spec, tmp_path / "LEDGER.csv")) == "spec_unreadable"


def test_oversized_spec_is_spec_unreadable(tmp_path):
    spec = tmp_path / "spec.yaml"
    spec.write_text("id: EXP-188\npad: '" + "a" * (1 << 20) + "'\n")
    assert refusal(lambda: _check(tmp_path, spec, tmp_path / "LEDGER.csv")) == "spec_unreadable"


def test_directory_as_ledger_is_ledger_malformed(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = tmp_path / "ledger_dir"
    ledger.mkdir()
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "ledger_malformed"


def test_huge_second_run_log_line_does_not_block_a_valid_first_line(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    first = json.dumps({"ts": "2026-11-17T21:00:00+00:00"})
    _write_run_log(tmp_path, first + "\n" + "x" * (1 << 20) + "\n")
    receipt = _check(tmp_path, spec, ledger)
    assert receipt["spec_hash_checked"] is True
    assert receipt["first_run_ts"].startswith("2026-11-17 21:00:00")


def test_leading_blank_lines_do_not_hide_an_earlier_first_run(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    record = json.dumps({"ts": "2026-11-17T19:00:00+00:00"})
    _write_run_log(tmp_path, "\n\n" + record + "\n")
    assert refusal(lambda: _check(tmp_path, spec, ledger)) == "preregistered_after_run"


def test_leading_blank_lines_then_later_run_passes_with_exact_first_run_ts(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    record = json.dumps({"ts": "2026-11-17T21:00:00+00:00"})
    _write_run_log(tmp_path, "\n\n" + record + "\n")
    receipt = _check(tmp_path, spec, ledger)
    assert receipt["spec_hash_checked"] is True
    assert receipt["first_run_ts"] == "2026-11-17 21:00:00+00:00"


def test_run_log_of_only_blank_lines_has_no_first_run_ts(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row(_digest(spec))])
    _write_run_log(tmp_path, "\n\n")
    receipt = _check(tmp_path, spec, ledger)
    assert receipt["spec_hash_checked"] is True
    assert receipt["first_run_ts"] is None


def test_register_planned_without_stamp_refuses_and_writes_no_row(tmp_path):
    spec = _write_spec(tmp_path, drop=("preregistered_at",))
    ledger = tmp_path / "LEDGER.csv"
    reason = refusal(lambda: size_pnl_prereg.register_planned(
        spec_path=spec, ledger_path=ledger, today="2026-11-17"))
    assert reason == "preregistered_at_missing"
    assert not ledger.exists()


# --------------------------------------------------------------------------
# register_planned
# --------------------------------------------------------------------------


def test_register_planned_appends_once_then_is_idempotent(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = tmp_path / "LEDGER.csv"
    assert size_pnl_prereg.register_planned(spec_path=spec, ledger_path=ledger,
                                            today="2026-11-17") is True
    assert size_pnl_prereg.register_planned(spec_path=spec, ledger_path=ledger,
                                            today="2026-11-17") is False
    receipt = _check(tmp_path, spec, ledger)
    assert receipt["spec_hash_checked"] is True


def test_register_planned_refuses_a_different_planned_hash(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = _write_ledger(tmp_path / "LEDGER.csv", [_planned_row("0" * 64)])
    reason = refusal(lambda: size_pnl_prereg.register_planned(
        spec_path=spec, ledger_path=ledger, today="2026-11-17"))
    assert reason == "planned_row_conflict"


def test_register_planned_refuses_run_log_earlier_than_stamp_and_writes_no_row(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = tmp_path / "LEDGER.csv"
    _write_run_log(tmp_path, json.dumps({"ts": "2026-11-17T19:00:00+00:00"}) + "\n")
    reason = refusal(lambda: size_pnl_prereg.register_planned(
        spec_path=spec, ledger_path=ledger, today="2026-11-17",
        run_log_path=_run_log_path(tmp_path)))
    assert reason == "preregistered_after_run"
    assert not ledger.exists()


def test_register_planned_with_later_run_log_registers(tmp_path):
    spec = _write_spec(tmp_path)
    ledger = tmp_path / "LEDGER.csv"
    _write_run_log(tmp_path, json.dumps({"ts": "2026-11-17T21:00:00+00:00"}) + "\n")
    assert size_pnl_prereg.register_planned(
        spec_path=spec, ledger_path=ledger, today="2026-11-17",
        run_log_path=_run_log_path(tmp_path)) is True
    assert size_pnl_prereg._planned_hashes(ledger) == [_digest(spec)]


# --------------------------------------------------------------------------
# no-op regression: no failure setup may ever return a receipt
# --------------------------------------------------------------------------


def _setup_missing_ledger(case):
    return _kw(case, _write_spec(case), case / "LEDGER.csv")


def _setup_ambiguous(case):
    spec = _write_spec(case)
    digest = _digest(spec)
    ledger = _write_ledger(case / "LEDGER.csv", [_planned_row(digest), _planned_row(digest)])
    return _kw(case, spec, ledger)


def _setup_hash_mismatch(case):
    spec = _write_spec(case)
    ledger = _write_ledger(case / "LEDGER.csv", [_planned_row("0" * 64)])
    return _kw(case, spec, ledger)


def _setup_spec_missing(case):
    return _kw(case, case / "missing.yaml", case / "LEDGER.csv")


def _setup_id_mismatch(case):
    return _kw(case, _write_spec(case, id="EXP-999"), case / "LEDGER.csv")


def _setup_stamp_missing(case):
    spec = _write_spec(case, drop=("preregistered_at",))
    ledger = _write_ledger(case / "LEDGER.csv", [_planned_row(_digest(spec))])
    return _kw(case, spec, ledger)


def _setup_run_before_stamp(case):
    spec = _write_spec(case)
    ledger = _write_ledger(case / "LEDGER.csv", [_planned_row(_digest(spec))])
    _write_run_log(case, json.dumps({"ts": "2026-11-17T19:00:00+00:00"}) + "\n")
    return _kw(case, spec, ledger)


def _setup_run_log_malformed(case):
    spec = _write_spec(case)
    ledger = _write_ledger(case / "LEDGER.csv", [_planned_row(_digest(spec))])
    _write_run_log(case, "not json\n")
    return _kw(case, spec, ledger)


def _setup_ledger_malformed(case):
    spec = _write_spec(case)
    columns = [c for c in lib.LEDGER_COLUMNS if c != "promoted"]
    ledger = _write_ledger(case / "LEDGER.csv", [_planned_row(_digest(spec))], columns=columns)
    return _kw(case, spec, ledger)


_FAILURE_SETUPS = (
    ("planned_row_missing", _setup_missing_ledger),
    ("planned_row_ambiguous", _setup_ambiguous),
    ("spec_hash_mismatch", _setup_hash_mismatch),
    ("spec_unreadable", _setup_spec_missing),
    ("spec_id_mismatch", _setup_id_mismatch),
    ("preregistered_at_missing", _setup_stamp_missing),
    ("preregistered_after_run", _setup_run_before_stamp),
    ("run_log_malformed", _setup_run_log_malformed),
    ("ledger_malformed", _setup_ledger_malformed),
)


def test_no_failure_setup_ever_returns_a_receipt(tmp_path):
    for reason, build in _FAILURE_SETUPS:
        case = tmp_path / reason
        case.mkdir()
        kwargs = build(case)
        # Each setup must refuse with its own reason; a normal return fails here.
        assert refusal(lambda: size_pnl_prereg.require_preregistration(**kwargs)) == reason


# --------------------------------------------------------------------------
# the real spec
# --------------------------------------------------------------------------


def test_real_spec_is_well_formed():
    spec = lib.load_spec(size_pnl_prereg.SPEC_PATH)
    assert spec["id"] == "EXP-188"
    assert spec.get("preregistered_at")
    assert len(spec["roster"]) == 12
    assert len(spec["pairs"]) == 3
    retuned = spec["short_vol"]["arms"]["retuned"]
    assert retuned["c_grid"] == [0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
    assert "ONE multiplier" in retuned["breadth"]
    assert spec["judgement"]["bootstrap"]["resamples"] == 10000
    assert set(spec["decisions"]) == {"d1", "d2", "d3", "d4", "d5"}
    assert spec["conditional_followup"]["in_scope_here"] is False


# --------------------------------------------------------------------------
# run.py
# --------------------------------------------------------------------------


def _load_run():
    loader = importlib.util.spec_from_file_location("exp188_run", RUN_PATH)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def test_run_refuses_without_planned_row_before_any_stage(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lib, "LEDGER_PATH", tmp_path / "LEDGER.csv")
    run = _load_run()
    code = run.main(["--stage", "inner", "--out-dir", str(tmp_path / "o"), "--expected", str(tmp_path / "e.json"),
                     "--sources", "x:y", "--no-ledger"])
    assert code == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == "EXPERIMENT_IDENTITY_CONFLICT"
    assert "planned" in refused["message"].lower()


def test_run_requires_the_no_ledger_flag():
    run = _load_run()
    with pytest.raises(SystemExit):
        run.main([])


# --------------------------------------------------------------------------
# closure isolation
# --------------------------------------------------------------------------


def _closure_modules(closure):
    modules = set()
    for key in closure:
        if not key.endswith(".py"):
            continue
        rel = key[:-3]
        if rel.endswith("/__init__"):
            rel = rel[:-len("/__init__")]
        modules.add(rel.replace("/", "."))
    return modules


def test_size_panel_closure_is_deterministic_and_excludes_prereg():
    closure = source_closure(ROOT, ["experiments/size_panel.py"])
    assert source_closure(ROOT, ["experiments/size_panel.py"]) == closure
    assert "experiments/size_panel.py" in closure
    assert not [key for key in closure if "size_pnl" in key or "EXP-188" in key]


def test_prereg_module_imports_no_module_from_the_closure():
    closure_modules = _closure_modules(source_closure(ROOT, ["experiments/size_panel.py"]))
    imported = set()
    for node in ast.walk(ast.parse(PREREG_PATH.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    assert "experiments.size_panel" not in imported
    # The spec requires these two imports. The closure reaches engine.v2.ops.errors through
    # engine.v2.ops.fingerprints, so those names and their parent packages are exempt.
    allowed = set()
    for name in ("engine.v2.ops.errors", "experiments.lib"):
        parts = name.split(".")
        allowed.update(".".join(parts[:index]) for index in range(1, len(parts) + 1))
    assert not (imported & (closure_modules - allowed))
