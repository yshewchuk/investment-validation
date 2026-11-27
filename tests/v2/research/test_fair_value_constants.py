"""Tests for fill model F slice F6b: tick rule, price-space solve, run_fit outputs and isolation from F.
Synthetic fills only (ticker XYZ). Expected values are hand-written from the spec, never from the functions under test.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from experiments import fair_value_constants as fvk, fair_value_fills as fvf, fair_value_fit as ff
from experiments import fair_value_run as run, fair_value_solve as fvs

ORDER = "2030-03-04"
FILL = "2030-03-04T15:30:00-05:00"
SNAP = "2030-03-04T15:46:00-05:00"
RECOVERY = [(0.02, 0.5), (0.03, 0.348), (0.05, 1.0), (0.07, 0.412), (0.10, 0.5), (0.15, 0.54), (0.20, 0.62), (0.25, 0.70), (0.30, 0.70)]


def make_row(h, c, side, *, hit=50.0, oi=1000.0, **over):
    """Quote (1-h, 1+h) on a call, strike 100, spot 101, expiry = order date: V = mid = 1. The price-space x is c."""
    price = 1.0 + c * h if side == "buy" else 1.0 - c * h
    row = {"fill_time": FILL, "snapshot_time": SNAP, "ticker": "XYZ", "expiry": ORDER, "strike": 100.0, "right": "C",
           "side": side, "qty": 1.0, "price": price, "bid": 1.0 - h, "ask": 1.0 + h, "bid_size": 50.0, "ask_size": 50.0,
           "open_interest": oi, "quote_repaired": "false", "spot": 101.0, "vol_date": None, "smvVol": None,
           "rates_date": None, "iRate5wk": None, "iRateLt": None, "borrow30": None, "borrow2yr": None, "divAmt": None,
           "divDate": None}
    row["ask_size" if side == "buy" else "bid_size"] = hit
    row.update(over)
    return row


def fills_file(tmp_path, rows, name="fills.csv") -> Path:
    p = tmp_path / name
    p.write_bytes(pd.DataFrame(rows).to_csv(index=False, na_rep="").encode())
    return p


def solve_rows(tmp_path, rows):
    d, _ = ff.price_fills(fills_file(tmp_path, rows))
    return fvs.solve(d)


def recovery_rows():
    """Base 0.30 and spread 0.40: x = min(1, max(0.30 + 0.40 * min(4h, 1), tick / h)), tick by hand: penny 0.01 off the 0.05 grid."""
    return [make_row(h, x, side) for _ in range(6) for h, x in RECOVERY for side in ("buy", "sell")]


def size_row():
    return make_row(0.10, 0.75, "buy", hit=5.0)


def oi_row():
    return make_row(0.10, 0.67, "buy", oi=50.0)


# --- 1. tick table ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bid,ask,tick", [(1.00, 1.10, 0.05), (3.00, 3.10, 0.10), (2.95, 3.05, 0.10), (2.90, 3.05, 0.05),
                                          (1.01, 1.10, 0.01), (1.00, 1.11, 0.01), (0.0, 0.05, 0.05), (0.15, 0.35, 0.05)])
def test_tick_table(bid, ask, tick):
    assert float(fvk.tick_size(bid, ask)) == pytest.approx(tick, abs=1e-12)


def test_grid_test_uses_tick_floor_multiple(monkeypatch):
    monkeypatch.setitem(fvk.TICK_FLOOR, "multiple", 0.10)
    assert float(fvk.tick_size(1.05, 1.15)) == pytest.approx(0.01, abs=1e-12)  # off the 0.10 grid: penny
    assert float(fvk.tick_size(1.10, 1.20)) == pytest.approx(0.05, abs=1e-12)  # on the 0.10 grid: nickel


# --- 2 and 3. exact recovery through run_fit; boundaries of the 100 and 30 thresholds ----------------------------------

def test_recovery_through_run_fit(tmp_path):
    out = tmp_path / "out"
    receipt = ff.run_fit(fills_file(tmp_path, recovery_rows()), out)
    fit = receipt["fit"]
    assert fit["unflagged"]["n"] == 108 and fit["unflagged"]["fitted"] is True
    assert fit["unflagged"]["tie_count"] == 1
    assert fit["unflagged"]["loss"] == pytest.approx(0.0, abs=1e-9)
    assert fit["unflagged"]["floor_bound"] == 36  # h = 0.02, 0.05, 0.10 are floor-bound: 3 x 12 fills
    assert fit["size_only"]["fitted"] is False and fit["oi_only"]["fitted"] is False
    doc = json.loads((out / "constants.json").read_text())
    c = doc["constants"]
    expected = {"C_BASE": pytest.approx(0.30, abs=1e-9), "C_SPREAD": pytest.approx(0.40, abs=1e-9), "C_THIN_SIZE": 0.15, "C_THIN_OI": 0.10,
                "SPREAD_SCALE": 0.50, "THIN_SIZE_BELOW": 10, "THIN_OI_BELOW": 100, "UNKNOWN_THIN": 0.5, "GUARD_FILL_FRACTION": 0.25,
                "TICK_FLOOR": {"penny": 0.01, "nickel": 0.05, "dime": 0.10, "multiple": 0.05, "mid_break": 3.00}}
    assert c == expected
    assert doc["sensitivity_variants"] == {"spread025": {"C_SPREAD": 0.25}, "spread050": {"C_SPREAD": 0.5}}
    assert receipt["constants_sha256"] == hashlib.sha256((out / "constants.json").read_bytes()).hexdigest()
    canon = json.dumps(c, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert receipt["constants_id"] == doc["constants_id"] == hashlib.sha256(canon).hexdigest()
    assert sorted(p.name for p in out.iterdir()) == ["constants.json", "fit_receipt.json"]


def test_source_hashes_are_file_bytes(tmp_path):
    receipt = ff.run_fit(fills_file(tmp_path, recovery_rows()), tmp_path / "out")
    assert receipt["constants_source_sha256"] == hashlib.sha256(Path(fvk.__file__).read_bytes()).hexdigest()
    assert receipt["solve_source_sha256"] == hashlib.sha256(Path(fvs.__file__).read_bytes()).hexdigest()


def test_base_and_spread_need_100_unflagged(tmp_path):
    rows = [make_row(h, x, "buy") for h, x in RECOVERY] * 11  # 99 unflagged fills
    constants, fit, _ = solve_rows(tmp_path, rows)
    assert (constants["C_BASE"], constants["C_SPREAD"]) == (0.25, 0.5)
    assert fit["unflagged"]["fitted"] is False and fit["unflagged"]["tie_count"] is None
    constants, fit, _ = solve_rows(tmp_path, rows + [make_row(0.10, 0.46, "buy")])  # 100 fills
    assert fit["unflagged"]["fitted"] is True and fit["unflagged"]["tie_count"] is not None
    assert constants["C_BASE"] == pytest.approx(0.30, abs=1e-9) and constants["C_SPREAD"] == pytest.approx(0.40, abs=1e-9)


def test_v_term_enters_the_solve(tmp_path):
    # 15 buys quoted 0.93/1.27 and 15 sells quoted 0.73/1.07, all at expiry: V = intrinsic = 1 (spot 101, strike 100), inside both quotes.
    # v = -0.1 / 0.17 on every fill (buy: (V - mid) / half; sell: (mid - V) / half). Prices come from the test's own x formula.
    rows = []
    for _ in range(15):
        for side, bid, ask in (("buy", 0.93, 1.27), ("sell", 0.73, 1.07)):
            mid, half = (bid + ask) / 2.0, (ask - bid) / 2.0
            term = min((ask - bid) / mid / 0.5, 1.0)
            v = (1.0 - mid) / half if side == "buy" else (mid - 1.0) / half
            x = v + (0.25 + 0.5 * term + 0.20) * (1.0 - v)  # base 0.25, spread 0.50, thin size 0.20, open interest not thin
            assert 0.01 / half < x < 1.0
            price = mid + x * half if side == "buy" else mid - x * half
            rows.append(make_row(half, 0.0, side, hit=5.0, bid=bid, ask=ask, price=price))
    d, _ = ff.price_fills(fills_file(tmp_path, rows))
    assert d["v"].to_numpy() == pytest.approx([-0.1 / 0.17] * 30, abs=1e-9)
    constants, fit, _ = fvs.solve(d)
    assert (constants["C_BASE"], constants["C_SPREAD"]) == (0.25, 0.5)
    assert constants["C_THIN_SIZE"] == pytest.approx(0.20, abs=1e-9)
    assert fit["size_only"]["fitted"] is True and fit["size_only"]["tie_count"] == 1
    assert fit["size_only"]["loss"] == pytest.approx(0.0, abs=1e-9)


def test_size_penalty_fitted_at_30_not_29(tmp_path):
    constants, fit, _ = solve_rows(tmp_path, [size_row()] * 29)
    assert constants["C_THIN_SIZE"] == 0.15 and fit["size_only"]["fitted"] is False and fit["size_only"]["tie_count"] is None
    constants, fit, _ = solve_rows(tmp_path, [size_row()] * 30)
    # h = 0.10, base 0.25, spread 0.50 (term 0.4): pred = max(0.45 + P, 0.50) = 0.75 at P = 0.30
    assert constants["C_THIN_SIZE"] == pytest.approx(0.30, abs=1e-9)
    assert fit["size_only"]["fitted"] is True and fit["size_only"]["tie_count"] == 1
    assert constants["C_THIN_OI"] == 0.10


def test_oi_penalty_fitted_at_30_not_29(tmp_path):
    constants, fit, _ = solve_rows(tmp_path, [oi_row()] * 29)
    assert constants["C_THIN_OI"] == 0.10 and fit["oi_only"]["fitted"] is False
    constants, fit, _ = solve_rows(tmp_path, [oi_row()] * 30)  # pred = max(0.45 + 0.22, 0.50) = 0.67
    assert constants["C_THIN_OI"] == pytest.approx(0.22, abs=1e-9) and fit["oi_only"]["fitted"] is True


def test_penalties_are_independent(tmp_path):
    constants, fit, _ = solve_rows(tmp_path, [size_row()] * 30 + [oi_row()] * 29)
    assert constants["C_THIN_SIZE"] == pytest.approx(0.30, abs=1e-9) and constants["C_THIN_OI"] == 0.10
    assert fit["oi_only"]["fitted"] is False
    constants, _, _ = solve_rows(tmp_path, [size_row()] * 29 + [oi_row()] * 30)
    assert constants["C_THIN_SIZE"] == 0.15 and constants["C_THIN_OI"] == pytest.approx(0.22, abs=1e-9)


def test_penalty_fitted_with_few_unflagged(tmp_path):
    constants, fit, _ = solve_rows(tmp_path, [size_row()] * 30 + [make_row(0.10, 0.45, "buy")] * 10)
    assert fit["unflagged"]["fitted"] is False and (constants["C_BASE"], constants["C_SPREAD"]) == (0.25, 0.5)
    assert constants["C_THIN_SIZE"] == pytest.approx(0.30, abs=1e-9) and fit["size_only"]["fitted"] is True


def test_both_and_unknown_do_not_change_fits(tmp_path):
    base = recovery_rows() + [size_row()] * 30 + [oi_row()] * 30
    extra = [make_row(0.10, 1.0, "buy", hit=5.0, oi=50.0)] * 20 + [make_row(0.10, 0.0, "buy", hit=None)] * 20
    without, _, _ = solve_rows(tmp_path, base)
    with_both, _, _ = solve_rows(tmp_path, base + extra)
    assert without == with_both


# --- 4. tie rules -----------------------------------------------------------------------------------------------------

def test_tie_rules(tmp_path):
    # h = 0.30: term 1, nickel tick, floor 0.1667. Predicted x = min(1, base + spread) so loss is zero on base + spread = 0.6: 61 pairs.
    rows = [make_row(0.30, 0.6, "buy" if i % 2 else "sell") for i in range(100)]
    constants, fit, _ = solve_rows(tmp_path, rows)
    assert constants["C_BASE"] == pytest.approx(0.60, abs=1e-9) and constants["C_SPREAD"] == pytest.approx(0.0, abs=1e-9)
    assert fit["unflagged"]["tie_count"] == 61
    # base 0.60, spread 0: pred = min(1, 0.60 + P) = 1 for P in 0.40..1.00: 61 ties, largest wins
    constants, fit, _ = solve_rows(tmp_path, rows + [make_row(0.30, 1.0, "buy", hit=5.0)] * 30)
    assert constants["C_THIN_SIZE"] == pytest.approx(1.00, abs=1e-9) and fit["size_only"]["tie_count"] == 61


def test_base_spread_tie_with_different_sums_prefers_largest(tmp_path):
    # h = 0.10: quote 0.90/1.10, half 0.10, nickel tick, floor 0.05 / 0.10 = 0.5; spread term 0.4. Every fill has v = 0 and x = 0.5,
    # so pred = max(base + 0.4 * spread, 0.5). Loss is zero whenever base + 0.4 * spread <= 0.5. Optimal pairs include
    # (0.00, 1.00), (0.50, 0.00) and (0.10, 1.00), with sums 1.00, 0.50 and 1.10. The largest sum wins: base 0.10, spread 1.00.
    rows = [make_row(0.10, 0.5, "buy")] * 100
    constants, fit, _ = solve_rows(tmp_path, rows)
    assert constants["C_BASE"] == pytest.approx(0.10, abs=1e-9) and constants["C_SPREAD"] == pytest.approx(1.00, abs=1e-9)
    assert fit["unflagged"]["loss"] == pytest.approx(0.0, abs=1e-9)


# --- 5. residual statistics -------------------------------------------------------------------------------------------

def test_residual_stats_hand_values(tmp_path):
    rows = [make_row(0.02, 0.39, "buy"), make_row(0.10, 0.35, "buy"),
            make_row(0.10, 0.9, "buy", hit=5.0, oi=50.0), make_row(0.10, 0.5, "buy", hit=5.0, oi=50.0)]
    _, _, res = solve_rows(tmp_path, rows)
    # constants stay current (no block reaches its threshold): both fills have c = 0.25 + 0.2 + 0.15 + 0.10 = 0.70, floor 0.5
    both = res["fitted"]["both"]
    assert both["n"] == 2
    assert both["mean"] == pytest.approx(0.0, abs=1e-9) and both["median"] == pytest.approx(0.0, abs=1e-9)
    assert both["mean_abs"] == pytest.approx(0.2, abs=1e-9)
    assert both["p10"] == pytest.approx(-0.16, abs=1e-9) and both["p90"] == pytest.approx(0.16, abs=1e-9)
    # current F without the floor: tight quote c = 0.25 + 0.5 * 0.08 = 0.29 (residual +0.10); wide quote c = 0.45 (residual -0.10)
    cur = res["current_unflagged"]
    assert cur["n"] == 2 and cur["mean"] == pytest.approx(0.0, abs=1e-9)
    assert cur["mean_abs"] == pytest.approx(0.10, abs=1e-9)
    assert cur["p10"] == pytest.approx(-0.08, abs=1e-9) and cur["p90"] == pytest.approx(0.08, abs=1e-9)
    assert res["fitted"]["size_only"] == {"n": 0, "mean": None, "median": None, "mean_abs": None, "p10": None, "p90": None}


# --- 6. isolation from F ----------------------------------------------------------------------------------------------

def test_new_fills_leave_f_unchanged(tmp_path):
    fvf_id = run.fvf_id()
    out = tmp_path / "out"
    ff.run_fit(fills_file(tmp_path, recovery_rows()), out)  # fits 0.30 / 0.40, which differ from the current constants
    assert json.loads((out / "constants.json").read_text())["constants"]["C_BASE"] == pytest.approx(0.30, abs=1e-9)
    assert (fvf.C_BASE, fvf.C_SPREAD, fvf.C_THIN_SIZE, fvf.C_THIN_OI) == (0.25, 0.50, 0.15, 0.10)
    assert run.fvf_id() == fvf_id


# --- 7. failure inside solve leaves no output; progress lines carry no ticker ------------------------------------------

def test_solve_failure_leaves_nothing_and_progress_has_no_ticker(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    p = fills_file(tmp_path, recovery_rows())
    ff.run_fit(p, tmp_path / "ok")
    assert "[fv_fit]" in caplog.text and "XYZ" not in caplog.text

    def boom(d):
        raise RuntimeError("boom")

    monkeypatch.setattr(fvs, "solve", boom)
    with pytest.raises(RuntimeError, match="boom"):
        ff.run_fit(p, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert sorted(x.name for x in tmp_path.iterdir()) == ["fills.csv", "ok"]  # no .fit-* staging directory left behind
