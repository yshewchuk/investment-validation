"""EXP-189 arm E per-event choices: the persisted rows carry the sweep's own marks, the month files are immutable and bound by receipts."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from experiments import size_pnl_receipts as receipts
from experiments import size_pnl_shortvol as sv
from experiments import size_width_choices as choices
from experiments import size_width_driver as driver
from experiments import size_width_report as report
from experiments.size_pnl_pool import build_pool
from experiments.size_pnl_shortvol import Refusal
from experiments.size_width_sizing import GRID
from experiments.size_width_success import LADDER
from experiments.size_width_sweep import E_ARM, Sweep
from tests.v2.research.test_size_pnl_driver import OK, chunk as event_chunk, sources, use_chunks
from tests.v2.research.test_size_pnl_shortvol import _tables
from tests.v2.research.test_size_width_driver import bounded, m0, taking  # noqa: F401  (bounded is an autouse fixture)
from tests.v2.research.test_size_width_fallback import _fb, _run as fallback_run, _scene
from tests.v2.research.test_size_width_report import _refused, _text, _write
from tests.v2.research.test_size_width_sweep import CUR, DRAW, NAME, WINDOW, _small_window, _two_months

PERIOD = "2024-06"
MARKS = ["exit_value", "exit_value_mid", "pnl_dollars", "pnl_dollars_mid", "lots", "debit_dollars",
         "secured_per_contract", *choices.ALPHA_COLS]


def _chain(monkeypatch, **kwargs):
    """The two-month synthetic chain on a ladder sweep over every structure: (sweep, the burn-in month's fresh choices, the output rows)."""
    _small_window(monkeypatch)
    pool = build_pool(*_tables(), floor=None)
    chunk, prior, cur, sessions = _two_months()
    sweep = Sweep(ladder=True, **kwargs)
    sweep.add_month(chunk, prior, pool, output=False, sessions=sessions, draws=DRAW)
    burn = list(sweep.fresh_choices)
    fresh = sweep.add_month(chunk, cur, pool, output=True, sessions=sessions, draws=DRAW)
    return sweep, burn, fresh


def _same(a, b):
    return bool(a == b) or bool(np.isnan(a) and np.isnan(b))


def test_burn_in_records_nothing_and_the_output_month_holds_one_row_per_event_and_structure(monkeypatch):
    sweep, burn, _fresh = _chain(monkeypatch, choices=True)
    assert burn == []
    frame = pd.concat(sweep.fresh_choices, ignore_index=True)
    assert list(frame.columns) == choices.COLS
    assert len(frame) == len(CUR) * len(sv.NAMES)
    assert set(zip(frame["structure"], frame["event_id"])) == {(n, e) for n in sv.NAMES for e in CUR}


def test_k_and_the_scores_are_the_chosen_k_and_the_k1_history(monkeypatch):
    sweep, _burn, _fresh = _chain(monkeypatch, choices=True)
    frame = pd.concat(sweep.fresh_choices, ignore_index=True)
    for name in sv.NAMES:
        mine = frame[frame["structure"] == name].set_index("event_id")
        by_k = {k: sweep.scored[(name, k)][-1].set_index("event_id")["exp_pnl_sim"] for k in GRID}
        for eid, row in mine.iterrows():
            assert row["k"] in by_k
            assert _same(row["exp_pnl_sim"], by_k[row["k"]][eid])
            assert _same(row["exp_pnl_sim_k1"], by_k[1.0][eid])


def test_taken_rows_carry_the_marks_of_their_arm_e_trade_and_the_ladder_rets(monkeypatch):
    sweep, _burn, _fresh = _chain(monkeypatch, choices=True)
    frame = pd.concat(sweep.fresh_choices, ignore_index=True)
    taken = frame[frame["status"] == "taken"]
    trades = pd.concat([sweep.rows([(n, E_ARM)], WINDOW) for n in sv.NAMES], ignore_index=True)
    assert not taken.empty
    got = taken.merge(trades, on=["event_id", "structure"], suffixes=("", "_t"))
    assert len(got) == len(taken) == len(trades)
    for col in ("entry_cost", "exit_value", "exit_value_mid", "pnl_dollars", "pnl_dollars_mid", "secured_per_contract"):
        np.testing.assert_allclose(got[col].to_numpy(), got[f"{col}_t"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(got["debit_dollars"].to_numpy(), (got["entry_cost"] * 100 * got["lots"]).to_numpy())
    for a in LADDER:
        tag = f"{a:.2f}"
        assert np.isfinite(got[f"pnl_a{tag}"].to_numpy()).all()
        ret = ((got[f"exit_a{tag}"] - got[f"entry_a{tag}"]) / got[f"entry_a{tag}"]).to_numpy()
        want = got[f"ret_a{tag}"].to_numpy()
        ok = np.isfinite(want)
        np.testing.assert_allclose(ret[ok], want[ok], rtol=1e-9)


def test_non_taken_rows_have_no_marks_and_keep_their_status_and_reason(monkeypatch):
    sweep, _burn, _fresh = _chain(monkeypatch, choices=True)
    frame = pd.concat(sweep.fresh_choices, ignore_index=True)
    rest = frame[frame["status"] != "taken"]
    assert rest[MARKS].isna().all().all()
    dec = sweep.declined_rows()
    dec = dec[dec["arm"] == E_ARM].set_index(["structure", "event_id"])
    rest = rest.set_index(["structure", "event_id"])
    assert set(rest.index) == set(dec.index)
    np.testing.assert_array_equal(rest["status"].to_numpy(), dec.loc[rest.index, "status"].to_numpy())
    np.testing.assert_array_equal(rest["reason"].fillna("none").to_numpy(),
                                  dec.loc[rest.index, "reason"].fillna("none").to_numpy())


def test_a_sweep_without_choices_and_the_default_fallback_record_nothing(tmp_path, monkeypatch):
    sweep, burn, _fresh = _chain(monkeypatch)
    assert sweep.record_choices is False and sweep.fresh_choices == [] and burn == []
    fallback = _fb(tmp_path)
    _small_window(monkeypatch)
    fallback_run(fallback, _scene())
    assert fallback.record_choices is False and fallback.fresh_choices == []


# ---- write_month and write_receipt ----------------------------------------------

def _frame(value=0.5):
    """One month's choice rows of one structure: event B is taken with marks, event A is declined; ``value`` is B's exp_pnl_sim."""
    decided = pd.DataFrame({"event_id": ["B", "A"], "event_date": pd.to_datetime(["2024-06-10", "2024-06-10"]),
                            "structure": [NAME, NAME], "k": [1.0, 1.0], "exp_pnl_sim": [value, 0.1],
                            "status": ["taken", "not_taken"], "reason": [None, "below_cutoff"], "entry_cost": [2.0, 2.0]})
    marks = pd.DataFrame({"event_id": ["B"], "lots": [10.0], "exit_value": [2.5], "pnl_dollars": [500.0]})
    secured = pd.DataFrame({"event_id": ["B"], "secured_per_contract": [1000.0]})
    return choices.choice_rows(decided, None, marks, marks, secured, {})


def test_write_month_keeps_the_file_for_the_same_rows_and_refuses_other_bytes(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    first = choices.write_month(out, PERIOD, [_frame()])
    path = out / "e_choices" / f"{PERIOD}.parquet"
    before = path.read_bytes()
    assert first == {"period": PERIOD, "sha256": first["sha256"], "rows": 2, "taken": 1}
    assert choices.write_month(out, PERIOD, [_frame()]) == first
    assert path.read_bytes() == before
    with pytest.raises(Refusal) as caught:
        choices.write_month(out, PERIOD, [_frame(value=0.9)])
    assert caught.value.details["reason"] == "month_freeze_changed"
    assert path.read_bytes() == before
    assert list((out / "e_choices").iterdir()) == [path]


def test_an_empty_month_writes_a_schemed_empty_file(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    got = choices.write_month(out, PERIOD, [])
    assert got == {"period": PERIOD, "sha256": got["sha256"], "rows": 0, "taken": 0}
    frame = pd.read_parquet(out / "e_choices" / f"{PERIOD}.parquet")
    assert list(frame.columns) == choices.COLS and len(frame) == 0
    for col in choices.COLS:
        if col not in ("event_id", "event_date", "structure", "status", "reason"):
            assert frame[col].dtype == np.float64, col


# ---- the driver writes the months and binds them ----------------------------------

def _assert_bound(out, receipt):
    """Each month that ``receipt`` names is a file under e_choices/ with its sha256, row count and taken count; the totals add up."""
    assert receipt["months"]
    for period, month in receipt["months"].items():
        path = out / "e_choices" / f"{period}.parquet"
        frame = pd.read_parquet(path)
        assert month["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert month["rows"] == len(frame) and month["taken"] == int((frame["status"] == "taken").sum())
    assert receipt["rows"] == sum(m["rows"] for m in receipt["months"].values())
    assert receipt["taken"] == sum(m["taken"] for m in receipt["months"].values())


def test_the_inner_stage_writes_one_month_file_per_output_month_bound_by_its_receipt(tmp_path, m0, taking, monkeypatch):
    out = tmp_path / "out"
    use_chunks(monkeypatch, [("2023Q3", event_chunk("E1")), ("2024Q3", event_chunk("E2"))])
    driver.run(sources(), "inner", out, event_ids=["E1", "E2"], prereg=OK)
    receipt = receipts.load_receipt(out / "receipts" / "e_choices_inner.json")
    assert sorted(receipt["months"]) == ["2024-08"]  # E1 (June 2023) is burn-in history; E2 (August 2024) is the output month
    _assert_bound(out, receipt)


def test_the_outer_stage_writes_each_frozen_month_beside_its_freeze_and_binds_it(tmp_path, m0, taking, monkeypatch):
    out, ids = tmp_path / "out", ["E1", "E2", "E3", "E4"]
    use_chunks(monkeypatch, [("2023Q3", event_chunk("E1")), ("2024Q3", event_chunk("E2"))])
    driver.run(sources(), "inner", out, event_ids=ids, prereg=OK)
    use_chunks(monkeypatch, [("2024Q3", event_chunk("E2")), ("2025Q1", event_chunk("E3", "E4"))])
    driver.run(sources(), "outer", out, event_ids=ids, prereg=OK)
    receipt = receipts.load_receipt(out / "receipts" / "e_choices_outer.json")
    frozen = sorted(p.stem for p in (out / "months").glob("*.parquet"))
    assert sorted(receipt["months"]) == frozen == ["2025-01", "2025-02"]
    _assert_bound(out, receipt)


# ---- the report line -------------------------------------------------------------------

def test_the_report_line_appears_with_a_receipt_and_a_receipt_for_other_inputs_refuses(tmp_path):
    _write(tmp_path, passing=("A",))
    report.assemble(tmp_path)
    assert "T5b arm E per-event choices" not in _text(tmp_path)
    receipts.write_receipt(tmp_path / "receipts" / "e_choices_inner.json",
                           {"stage": "inner", "inputs_hash": "H", "scope": None, "months": {}, "rows": 4, "taken": 1})
    report.assemble(tmp_path)
    text = _text(tmp_path)
    assert "### T5b arm E per-event choices (persisted, not analysed here)" in text
    assert "inner: 4 rows (1 taken); outer: not persisted." in text
    other = tmp_path / "other"
    _write(other, passing=("A",))
    receipts.write_receipt(other / "receipts" / "e_choices_inner.json",
                           {"stage": "inner", "inputs_hash": "OTHER", "scope": None, "months": {}, "rows": 0, "taken": 0})
    _refused(other, "tuning_receipt_changed")
