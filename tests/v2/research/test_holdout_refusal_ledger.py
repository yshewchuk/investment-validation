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