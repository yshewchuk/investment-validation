import csv
from types import SimpleNamespace

import pytest

from engine.v2.data import errors
from engine.v2.research.experiment_trades import load_trades
from engine.v2.research.holdout_refusal import record_holdout_refusal

SNAPSHOT = SimpleNamespace(snapshot_id="snap-1")


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_refused_holdout_read_records_one_plain_row_and_still_raises(tmp_path):
    ledger = tmp_path / "LEDGER.csv"
    with pytest.raises(errors.DataError) as caught:
        with record_holdout_refusal("EXP-900", ledger):
            load_trades(None, SNAPSHOT, "any", as_of_month=None)
    assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
    rows = _rows(ledger)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "EXP-900"
    assert row["stage"] == "refused:HOLDOUT_ACCESS_DENIED"
    assert row["date"]
    assert row["oos_mean_mid"] == row["sharpe_trade"] == row["spec_hash"] == ""


def test_other_errors_write_no_row(tmp_path):
    ledger = tmp_path / "LEDGER.csv"
    with pytest.raises(errors.DataError):
        with record_holdout_refusal("EXP-900", ledger):
            raise errors.fail("POPULATION_COLLAPSED", "empty")
    assert not ledger.exists()


def test_ledger_failure_does_not_mask_the_refusal(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(errors.DataError) as caught:
        with record_holdout_refusal("EXP-900", blocker / "LEDGER.csv"):
            raise errors.fail("HOLDOUT_ACCESS_DENIED", "denied")
    assert caught.value.code == "HOLDOUT_ACCESS_DENIED"


def test_refused_rows_do_not_count_as_specs_tried(tmp_path):
    from experiments.lib import LEDGER_COLUMNS, ledger_append, ledger_context

    ledger = tmp_path / "LEDGER.csv"
    base = {"date": "2026-10-10", "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False"}
    rows = [{"id": "EXP-900", "spec_hash": "aaa", "stage": "ran", **base},
            {"id": "EXP-901", "spec_hash": "", "stage": "refused:HOLDOUT_ACCESS_DENIED", **base},
            {"id": "EXP-902", "spec_hash": "bbb", "stage": "refused", **base}]
    ledger_append([{name: row[name] for name in LEDGER_COLUMNS} for row in rows], path=ledger)
    assert ledger_context("aaa", path=ledger) == {"specs_tried": 1, "this_spec_rows": 1, "promotions": 0}