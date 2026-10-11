from experiments.lib import LEDGER_COLUMNS, ledger_append, ledger_context


def test_refused_rows_do_not_count_as_specs_tried(tmp_path):
    ledger = tmp_path / "LEDGER.csv"
    base = {"date": "2026-10-10", "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False"}
    rows = [{"id": "EXP-900", "spec_hash": "aaa", "stage": "ran", **base},
            {"id": "EXP-901", "spec_hash": "", "stage": "refused", **base},
            {"id": "EXP-902", "spec_hash": "bbb", "stage": "refused:HOLDOUT_ACCESS_DENIED", **base}]
    ledger_append([{name: row[name] for name in LEDGER_COLUMNS} for row in rows], path=ledger)
    assert ledger_context("aaa", path=ledger) == {"specs_tried": 1, "this_spec_rows": 1, "promotions": 0}