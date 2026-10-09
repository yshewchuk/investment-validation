"""Promotion is bound to a recording receipt for the exact primary run.

A promotion must be authorized by the primary metrics artifact whose bytes a
``ran`` ledger row recorded. These tests drive the real evaluator and a
per-test temporary ledger, then prove the four refusal cases: no receipt, a
receipt from another run of the same spec, a receipt for another spec, and
metrics edited after recording. A clean recorded run is accepted: the recorder
finalized its checklist against the ledger, so a good run is not blocked by
the evaluator's pre-recording snapshot.

# packages: engine.v2.evaluation
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments import lib
from experiments import promote


@pytest.fixture(autouse=True)
def _temp_ledger(tmp_root, monkeypatch):
    """Every ordinary test records into this test's temporary ledger.

    Patching ``lib.LEDGER_PATH`` is what keeps a run that omits an explicit
    path from appending to the checkout's ``experiments/LEDGER.csv``.
    """
    monkeypatch.setattr(lib, "LEDGER_PATH", tmp_root / "experiments" / "LEDGER.csv")


def _spec(grid: bool = False, exp_id: str = "EXP-901") -> dict:
    return {
        "id": exp_id,
        "title": "synthetic receipt probe",
        "price_source": "orats",
        "primary_spec": {"probe": "primary"},
        "grid": {"probe": ["second", "third"]} if grid else {},
        "walk_forward": {"min_train_years": 1},
        "preregistered_at": "2019-01-01T00:00:00+00:00",
    }


def _trades() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=12, freq="90D")
    base = [-0.1, 0.2] * 6
    rows = []
    for i, (date, ret) in enumerate(zip(dates, base)):
        # Three fill alphas per event so the report's fill-sensitivity item has
        # a real sweep (only 0.5 is the headline/mid set).
        for alpha in (0.0, 0.5, 1.0):
            rows.append({
                "event_id": f"probe-{i}", "ticker": "TEST",
                "event_date": date, "entry_date": date - pd.Timedelta(days=1),
                "exit_date": date + pd.Timedelta(days=1), "fill_alpha": alpha,
                "entry_cost": 1.0, "exit_value": 1 + ret, "ret": ret,
            })
    return pd.DataFrame(rows)


def _run(spec, run_dir, **kwargs):
    return lib.evaluate_with_grid(
        spec, _trades(), run_dir, alphas=(0.0, 0.5, 1.0), fractions=(0.05,),
        mc_paths=2, stress=False, write_report=False, **kwargs,
    )


def _experiment(tmp_root, spec, exp_id: str) -> Path:
    folder = tmp_root / "experiments" / f"{exp_id}_synthetic"
    folder.mkdir(parents=True)
    lib.save_spec(spec, folder / "spec.yaml")
    return folder


def _weak_champion() -> dict:
    return {"cagr": -1.0, "sharpe_trade": -5.0, "mean": -1.0,
            "years_positive": 0, "years_evaluated": 1}


def test_recorded_run_is_accepted_and_eligible(tmp_root):
    spec = _spec(exp_id="EXP-901")
    folder = _experiment(tmp_root, spec, "EXP-901")
    result = _run(spec, folder)

    loaded_spec, results = promote.load_experiment_metrics(
        "EXP-901", root=tmp_root / "experiments")
    assert loaded_spec["id"] == "EXP-901"
    assert results["run_id"] == result.results["run_id"]
    # ``record_evaluation`` now finalizes its checklist after appending the
    # ran row; promotion validates the stored artifact's receipt and returns
    # that stored checklist without recomputing it.
    failed = [i for i in results["checklist"] if i["status"] == "FAIL"]
    assert not failed, failed
    assert results["checklist_fails"] == 0
    promoted, reasons = promote.decide(results, _weak_champion())
    assert promoted, reasons


def test_planned_only_run_refuses(tmp_root):
    spec = _spec(exp_id="EXP-902")
    folder = _experiment(tmp_root, spec, "EXP-902")
    lib.ledger_append([{
        "id": spec["id"], "spec_hash": lib.spec_hash(spec), "date": "2019-01-01",
        "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "",
        "promoted": "False",
    }])
    _run(spec, folder, record=False)

    with pytest.raises(promote.PromotionRefused,
                       match=promote.PROMOTION_LEDGER_RECEIPT_MISSING):
        promote.load_experiment_metrics("EXP-902", root=tmp_root / "experiments")


def test_smoke_overwrite_invalidates_an_earlier_receipt(tmp_root):
    spec = _spec(exp_id="EXP-903")
    folder = _experiment(tmp_root, spec, "EXP-903")
    _run(spec, folder)
    receipt = lib.receipt_path(folder, spec)
    assert receipt.is_file()

    _run(spec, folder, record=False)
    assert not receipt.exists(), "the overwrite must drop the stale receipt"

    with pytest.raises(promote.PromotionRefused,
                       match=promote.PROMOTION_LEDGER_RECEIPT_MISSING):
        promote.load_experiment_metrics("EXP-903", root=tmp_root / "experiments")


def test_receipt_for_another_spec_refuses(tmp_root):
    spec = _spec(exp_id="EXP-905")
    folder = _experiment(tmp_root, spec, "EXP-905")
    _run(spec, folder)

    receipt = lib.receipt_path(folder, spec)
    doc = json.loads(receipt.read_text())
    doc["spec_hash"] = "0" * 64
    receipt.write_text(json.dumps(doc))

    with pytest.raises(promote.PromotionRefused,
                       match=promote.PROMOTION_LEDGER_RECEIPT_MISSING):
        promote.load_experiment_metrics("EXP-905", root=tmp_root / "experiments")


def test_metrics_edited_after_recording_refuses(tmp_root):
    spec = _spec(exp_id="EXP-906")
    folder = _experiment(tmp_root, spec, "EXP-906")
    _run(spec, folder)

    metrics = lib.metrics_path(folder, spec)
    doc = json.loads(metrics.read_text())
    doc["headline"]["mean"] = 999.0
    metrics.write_text(json.dumps(doc))

    with pytest.raises(promote.PromotionRefused,
                       match=promote.PROMOTION_LEDGER_RECEIPT_MISSING):
        promote.load_experiment_metrics("EXP-906", root=tmp_root / "experiments")


def test_secondary_arm_metrics_cannot_authorize_the_primary(tmp_root):
    spec = _spec(grid=True, exp_id="EXP-907")
    folder = _experiment(tmp_root, spec, "EXP-907")
    _run(spec, folder)

    primary_metrics = lib.metrics_path(folder, spec)
    promote.validate_recording_receipt(spec, primary_metrics)

    cell = {**spec, "primary_spec": {"probe": "second"}, "grid_cell": True}
    cell_metrics = lib.metrics_path(folder, cell)
    assert cell_metrics.is_file() and cell_metrics != primary_metrics
    with pytest.raises(promote.PromotionRefused,
                       match=promote.PROMOTION_LEDGER_RECEIPT_MISSING):
        promote.validate_recording_receipt(spec, cell_metrics)


def test_second_run_metrics_with_first_receipt_refuses(tmp_root):
    spec = _spec(exp_id="EXP-908")
    folder = _experiment(tmp_root, spec, "EXP-908")

    _run(spec, folder)
    first_receipt = lib.receipt_path(folder, spec).read_text()
    _run(spec, folder)

    metrics = lib.metrics_path(folder, spec)
    assert json.loads(metrics.read_text())["recording_mode"] == "recorded"
    # Pair the SECOND run's metrics with the FIRST run's receipt: both are
    # "recorded" and share a spec hash, so only the run ID distinguishes them.
    lib.receipt_path(folder, spec).write_text(first_receipt)
    with pytest.raises(promote.PromotionRefused,
                       match=promote.PROMOTION_LEDGER_RECEIPT_MISSING):
        promote.validate_recording_receipt(spec, metrics)


def test_legacy_ledger_header_without_run_id_still_authorizes(tmp_root):
    spec = _spec(exp_id="EXP-909")
    folder = _experiment(tmp_root, spec, "EXP-909")
    legacy = tmp_root / "legacy_ledger.csv"
    # A pre-run_id header: the recorder must append a ``ran`` row to it without
    # rewriting the header, and the receipt/artifact binding must still pass.
    legacy.write_text("id,spec_hash,date,stage,oos_mean_mid,sharpe_trade,promoted\n")

    _run(spec, folder, ledger_path=legacy)
    rows = lib.ledger_read(legacy)
    assert "run_id" not in rows.columns
    assert rows["stage"].tolist() == ["ran"]

    promote.validate_recording_receipt(
        spec, lib.metrics_path(folder, spec), ledger_path=legacy)


def test_run_id_ledger_header_records_and_verifies_the_row(tmp_root):
    spec = _spec(exp_id="EXP-912")
    folder = _experiment(tmp_root, spec, "EXP-912")
    runid_ledger = tmp_root / "runid_ledger.csv"
    # A ledger whose header already carries run_id: the recorder must append
    # the run's own ID into the ``ran`` row, and promotion must match that row.
    runid_ledger.write_text(
        "id,spec_hash,date,stage,oos_mean_mid,sharpe_trade,promoted,run_id\n")

    _run(spec, folder, ledger_path=runid_ledger)
    rows = lib.ledger_read(runid_ledger)
    assert "run_id" in rows.columns
    assert rows["stage"].tolist() == ["ran"]
    metrics_run_id = json.loads(lib.metrics_path(folder, spec).read_text())["run_id"]
    assert metrics_run_id
    assert rows["run_id"].tolist() == [metrics_run_id]

    promote.validate_recording_receipt(
        spec, lib.metrics_path(folder, spec), ledger_path=runid_ledger)


def test_promote_cli_dry_run_accepts_recorded_primary(tmp_root, monkeypatch):
    spec = _spec(exp_id="EXP-910")
    folder = _experiment(tmp_root, spec, "EXP-910")
    result = _run(spec, folder)
    assert result.results["run_id"]

    # Only experiment discovery is stubbed to the temporary tree: the evaluator,
    # ledger recorder, receipt producer and promotion validator all stay real.
    monkeypatch.setattr(lib, "EXPERIMENTS_DIR", tmp_root / "experiments")
    champion = tmp_root / "champion.json"
    champion.write_text(json.dumps(_weak_champion()))

    rc = promote.main(["EXP-910", "--champion-metrics", str(champion), "--dry-run"])
    assert rc == 0


def test_metrics_replaced_after_digest_read_returns_validated_object(tmp_root, monkeypatch):
    spec = _spec(exp_id="EXP-911")
    folder = _experiment(tmp_root, spec, "EXP-911")
    _run(spec, folder)

    metrics = lib.metrics_path(folder, spec)
    original = json.loads(metrics.read_bytes())
    tampered = {**original, "headline": {**original["headline"], "mean": 999.0}}
    tampered_bytes = json.dumps(tampered).encode()

    real_read_bytes = Path.read_bytes

    def poisoned(self):
        data = real_read_bytes(self)
        if self == metrics:
            # Swap the on-disk file the moment the validator's (single) read
            # lands: everything after that — digest check included — must act
            # on the snapshot, never the later bytes.
            metrics.write_bytes(tampered_bytes)
        return data

    monkeypatch.setattr(Path, "read_bytes", poisoned)
    artifact = promote.validate_recording_receipt(spec, metrics)

    assert artifact["headline"]["mean"] == original["headline"]["mean"]
    assert artifact["run_id"] == original["run_id"]
    monkeypatch.undo()
    assert metrics.read_bytes() == tampered_bytes, \
        "the regression only counts if the swap really happened"
