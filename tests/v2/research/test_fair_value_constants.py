"""Tests for fill model F slice F6b: mid-anchored (model A) price-space solve, run_fit outputs and isolation from F.
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
# (h, x) with x = 0.30 + 0.40 * min(4h, 1), hand-computed: the spread term is min(s / 0.5, 1) with s = 2h.
# h = 0.005 and 0.02 are the small quotes where the retired one-tick floor used to bind; model A has no floor.
RECOVERY = [(0.005, 0.308), (0.02, 0.332), (0.03, 0.348), (0.05, 0.38), (0.07, 0.412), (0.10, 0.46), (0.15, 0.54), (0.20, 0.62),
            (0.25, 0.70), (0.30, 0.70)]


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


def recovery_rows(**over):
    """Base 0.30 and spread 0.40: x = 0.30 + 0.40 * min(4h, 1) per the RECOVERY table. 6 copies, buys and sells: 120 fills."""
    return [make_row(h, x, side, **over) for _ in range(6) for h, x in RECOVERY for side in ("buy", "sell")]


def size_row():
    return make_row(0.10, 0.75, "buy", hit=5.0)


def oi_row():
    return make_row(0.10, 0.67, "buy", oi=50.0)


# --- 1. recovery through run_fit; document contents and ids --------------------------------------------------------------

def test_recovery_through_run_fit(tmp_path):
    out = tmp_path / "out"
    receipt = ff.run_fit(fills_file(tmp_path, recovery_rows()), out)
    fit = receipt["fit"]
    assert set(fit["unflagged"]) == {"n", "fitted", "loss", "tie_count"}
    assert fit["unflagged"]["n"] == 120 and fit["unflagged"]["fitted"] is True
    assert fit["unflagged"]["tie_count"] == 1
    assert fit["unflagged"]["loss"] == pytest.approx(0.0, abs=1e-9)  # no floor: the h = 0.005 and 0.02 rows fit exactly too
    assert fit["size_only"]["fitted"] is False and fit["oi_only"]["fitted"] is False
    doc = json.loads((out / "constants.json").read_text())
    assert set(doc) == {"constants_id", "anchor", "constants", "sensitivity_variants"}
    assert doc["anchor"] == "mid"
    expected = {"C_BASE": pytest.approx(0.30, abs=1e-9), "C_SPREAD": pytest.approx(0.40, abs=1e-9), "C_THIN_SIZE": 0.15, "C_THIN_OI": 0.10,
                "SPREAD_SCALE": 0.50, "THIN_SIZE_BELOW": 10, "THIN_OI_BELOW": 100, "UNKNOWN_THIN": 0.5, "GUARD_FILL_FRACTION": 0.25}
    assert doc["constants"] == expected
    assert doc["sensitivity_variants"] == {"spread050": {"C_SPREAD": 0.5}}
    assert receipt["constants_sha256"] == hashlib.sha256((out / "constants.json").read_bytes()).hexdigest()
    body = {k: v for k, v in doc.items() if k != "constants_id"}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert receipt["constants_id"] == doc["constants_id"] == hashlib.sha256(canon).hexdigest()
    assert sorted(p.name for p in out.iterdir()) == ["constants.json", "fit_receipt.json"]
    written = json.loads((out / "fit_receipt.json").read_text())
    assert written["fit"] == receipt["fit"]
    assert set(written["fit"]) == {"unflagged", "size_only", "oi_only"}


def test_source_hashes_are_file_bytes(tmp_path):
    receipt = ff.run_fit(fills_file(tmp_path, recovery_rows()), tmp_path / "out")
    assert receipt["constants_source_sha256"] == hashlib.sha256(Path(fvk.__file__).read_bytes()).hexdigest()
    assert receipt["solve_source_sha256"] == hashlib.sha256(Path(fvs.__file__).read_bytes()).hexdigest()


# --- 2. thresholds 100 and 30 -----------------------------------------------------------------------------------------

def test_base_and_spread_need_100_unflagged(tmp_path):
    rows = [make_row(h, x, "buy") for h, x in RECOVERY[:9]] * 11  # 99 unflagged fills
    constants, fit, _ = solve_rows(tmp_path, rows)
    assert (constants["C_BASE"], constants["C_SPREAD"]) == (0.25, 0.5)
    assert fit["unflagged"]["fitted"] is False and fit["unflagged"]["tie_count"] is None
    # KEPT with 99 rows: base 0.25, spread 0.50. |x - c| per RECOVERY row: 0.048 + 0.042 + 0.038 + 0.030 + 0.022 + 0.010 + 0.010 + 0.030 + 0.050
    # = 0.280 over 9 rows, so loss 0.28 / 9.
    assert fit["unflagged"]["n"] == 99 and fit["unflagged"]["loss"] == pytest.approx(0.28 / 9, abs=1e-9)
    assert fit["size_only"] == {"n": 0, "fitted": False, "loss": None, "tie_count": None}
    assert fit["oi_only"] == {"n": 0, "fitted": False, "loss": None, "tie_count": None}
    constants, fit, _ = solve_rows(tmp_path, rows + [make_row(0.10, 0.46, "buy")])  # 100 fills: 0.30 + 0.40 * 0.4 = 0.46
    assert fit["unflagged"]["fitted"] is True and fit["unflagged"]["tie_count"] == 1  # only (0.30, 0.40) has zero loss
    assert fit["unflagged"]["n"] == 100 and fit["unflagged"]["loss"] == pytest.approx(0.0, abs=1e-9)
    assert constants["C_BASE"] == pytest.approx(0.30, abs=1e-9) and constants["C_SPREAD"] == pytest.approx(0.40, abs=1e-9)


def test_v_is_not_used(tmp_path):
    # Same quotes and same prices, so the same price-space x. Near set: spot 101, V = 1 = mid, so v = 0 on every fill.
    # Far set: spot 110, intrinsic 10 clamps to the upper quote edge 1 + h, so v = +1 on buys and -1 on sells.
    near = recovery_rows()
    far = recovery_rows(spot=110.0)
    d_near, _ = ff.price_fills(fills_file(tmp_path, near, "near.csv"))
    d_far, _ = ff.price_fills(fills_file(tmp_path, far, "far.csv"))
    assert d_near["v"].to_numpy() == pytest.approx([0.0] * 120, abs=1e-9)
    assert d_far["v"].to_numpy() == pytest.approx([1.0, -1.0] * 60, abs=1e-9)
    constants_near, _, res_near = fvs.solve(d_near)
    constants_far, _, res_far = fvs.solve(d_far)
    assert constants_far == constants_near
    assert constants_far["C_BASE"] == pytest.approx(0.30, abs=1e-9) and constants_far["C_SPREAD"] == pytest.approx(0.40, abs=1e-9)
    assert res_far["fitted"] == res_near["fitted"]
    fitted = res_far["fitted"]["unflagged"]
    assert fitted["n"] == 120 and fitted["mean"] == pytest.approx(0.0, abs=1e-9) and fitted["mean_abs"] == pytest.approx(0.0, abs=1e-9)


def test_current_residual_uses_v(tmp_path):
    # Quote 0.90 / 1.10: half 0.10, spread term 0.4. Spot 101.05 puts V = 1.05 = mid + 0.5 * half, so v = +0.5 on the buy, -0.5 on the sell.
    # Current F (no thin penalties): c = 0.25 + 0.5 * 0.4 = 0.45. Current residual = x - (v + c * (1 - v)); fitted residual = x - c.
    # Buy x = 0.6: current 0.6 - (0.5 + 0.45 * 0.5) = -0.125; fitted 0.6 - 0.45 = 0.15.
    # Sell x = 0.3: current 0.3 - (-0.5 + 0.45 * 1.5) = 0.125; fitted 0.3 - 0.45 = -0.15.
    rows = [make_row(0.10, 0.6, "buy", spot=101.05), make_row(0.10, 0.3, "sell", spot=101.05)]
    d, _ = ff.price_fills(fills_file(tmp_path, rows))
    assert d["v"].to_numpy() == pytest.approx([0.5, -0.5], abs=1e-9)
    constants, fit, res = fvs.solve(d)
    assert fit["unflagged"]["fitted"] is False
    assert (constants["C_BASE"], constants["C_SPREAD"]) == (0.25, 0.5)
    fitted = res["fitted"]["unflagged"]
    assert fitted["n"] == 2 and fitted["mean"] == pytest.approx(0.0, abs=1e-9)
    assert fitted["mean_abs"] == pytest.approx(0.15, abs=1e-9)
    cur = res["current_unflagged"]
    assert cur["n"] == 2 and cur["mean"] == pytest.approx(0.0, abs=1e-9)
    assert cur["mean_abs"] == pytest.approx(0.125, abs=1e-9)
    assert cur["p10"] == pytest.approx(-0.10, abs=1e-9) and cur["p90"] == pytest.approx(0.10, abs=1e-9)


def test_size_penalty_fitted_at_30_not_29(tmp_path):
    constants, fit, _ = solve_rows(tmp_path, [size_row()] * 29)
    assert constants["C_THIN_SIZE"] == 0.15 and fit["size_only"]["fitted"] is False and fit["size_only"]["tie_count"] is None
    # KEPT with 29 rows: pred = 0.25 + 0.5 * 0.4 + 0.15 = 0.60 against x 0.75, so loss 0.15
    assert fit["size_only"]["n"] == 29 and fit["size_only"]["loss"] == pytest.approx(0.15, abs=1e-9)
    assert fit["unflagged"] == {"n": 0, "fitted": False, "loss": None, "tie_count": None}
    assert fit["oi_only"] == {"n": 0, "fitted": False, "loss": None, "tie_count": None}
    constants, fit, _ = solve_rows(tmp_path, [size_row()] * 30)
    # h = 0.10, base 0.25, spread 0.50 (term 0.4): pred = min(1, 0.45 + P) = 0.75 at P = 0.30, the only zero-loss grid point
    assert constants["C_THIN_SIZE"] == pytest.approx(0.30, abs=1e-9)
    assert fit["size_only"]["fitted"] is True and fit["size_only"]["tie_count"] == 1
    assert fit["size_only"]["n"] == 30 and fit["size_only"]["loss"] == pytest.approx(0.0, abs=1e-9)  # pred 0.45 + 0.30 = 0.75 = x
    assert constants["C_THIN_OI"] == 0.10


def test_oi_penalty_fitted_at_30_not_29(tmp_path):
    constants, fit, _ = solve_rows(tmp_path, [oi_row()] * 29)
    assert constants["C_THIN_OI"] == 0.10 and fit["oi_only"]["fitted"] is False
    # KEPT with 29 rows: pred = 0.25 + 0.5 * 0.4 + 0.10 = 0.55 against x 0.67, so loss 0.12
    assert fit["oi_only"]["n"] == 29 and fit["oi_only"]["tie_count"] is None
    assert fit["oi_only"]["loss"] == pytest.approx(0.12, abs=1e-9)
    assert fit["unflagged"] == {"n": 0, "fitted": False, "loss": None, "tie_count": None}
    assert fit["size_only"] == {"n": 0, "fitted": False, "loss": None, "tie_count": None}
    constants, fit, _ = solve_rows(tmp_path, [oi_row()] * 30)  # pred = min(1, 0.45 + P) = 0.67 at P = 0.22
    assert constants["C_THIN_OI"] == pytest.approx(0.22, abs=1e-9) and fit["oi_only"]["fitted"] is True
    assert fit["oi_only"]["tie_count"] == 1 and fit["oi_only"]["n"] == 30
    assert fit["oi_only"]["loss"] == pytest.approx(0.0, abs=1e-9)  # pred 0.45 + 0.22 = 0.67 = x


def test_penalties_are_independent(tmp_path):
    constants, fit, _ = solve_rows(tmp_path, [size_row()] * 30 + [oi_row()] * 29)
    assert constants["C_THIN_SIZE"] == pytest.approx(0.30, abs=1e-9) and constants["C_THIN_OI"] == 0.10
    assert fit["oi_only"]["fitted"] is False
    constants, _, _ = solve_rows(tmp_path, [size_row()] * 29 + [oi_row()] * 30)
    assert constants["C_THIN_SIZE"] == 0.15 and constants["C_THIN_OI"] == pytest.approx(0.22, abs=1e-9)


def test_both_penalties_fitted_with_nonzero_v(tmp_path):
    # spot 101.05, h = 0.10: V = 1.05, so v = +0.5 on buys and -0.5 on sells. Sides alternate, buy first.
    # Fewer than 100 unflagged, so base 0.25 and spread 0.50 are kept (term 0.4, base part 0.45).
    # Size rows x = 0.75: pred = 0.45 + P, so P = 0.30. OI rows x = 0.67: P = 0.22.
    sides = ("buy", "sell")
    size = [make_row(0.10, 0.75, sides[i % 2], hit=5.0, spot=101.05) for i in range(30)]
    oi = [make_row(0.10, 0.67, sides[i % 2], oi=50.0, spot=101.05) for i in range(30)]
    d, _ = ff.price_fills(fills_file(tmp_path, size + oi))
    assert d["v"].to_numpy() == pytest.approx([0.5, -0.5] * 30, abs=1e-9)
    constants, fit, _ = fvs.solve(d)
    assert (constants["C_BASE"], constants["C_SPREAD"]) == (0.25, 0.5)
    assert constants["C_THIN_SIZE"] == pytest.approx(0.30, abs=1e-9) and constants["C_THIN_OI"] == pytest.approx(0.22, abs=1e-9)
    assert fit["size_only"]["fitted"] is True and fit["size_only"]["tie_count"] == 1
    assert fit["oi_only"]["fitted"] is True and fit["oi_only"]["tie_count"] == 1


def test_penalty_fitted_with_few_unflagged(tmp_path):
    constants, fit, _ = solve_rows(tmp_path, [size_row()] * 30 + [make_row(0.10, 0.45, "buy")] * 10)
    assert fit["unflagged"]["fitted"] is False and (constants["C_BASE"], constants["C_SPREAD"]) == (0.25, 0.5)
    assert constants["C_THIN_SIZE"] == pytest.approx(0.30, abs=1e-9) and fit["size_only"]["fitted"] is True


def test_both_and_unknown_do_not_change_fits(tmp_path):
    base = recovery_rows() + [size_row()] * 30 + [oi_row()] * 30
    extra = [make_row(0.10, 1.0, "buy", hit=5.0, oi=50.0)] * 20 + [make_row(0.10, 0.0, "buy", hit=None)] * 20
    without, fit_without, res_without = solve_rows(tmp_path, base)
    with_both, fit_with, res_with = solve_rows(tmp_path, base + extra)
    assert without == with_both
    for name in ("unflagged", "size_only", "oi_only"):
        assert fit_with[name] == fit_without[name]
        assert res_with["fitted"][name] == res_without["fitted"][name]
    assert res_with["current_unflagged"] == res_without["current_unflagged"]
    assert res_with["fitted"]["both"]["n"] == 20 and res_with["fitted"]["unknown"]["n"] == 20


# --- 3. tie rules -----------------------------------------------------------------------------------------------------

def test_tie_rules(tmp_path):
    # h = 0.30: term 1, so pred = min(1, base + spread). Loss is zero on base + spread = 0.6: 61 pairs, all with equal sum.
    # The largest base wins: base 0.60, spread 0.00.
    rows = [make_row(0.30, 0.6, "buy" if i % 2 else "sell") for i in range(100)]
    constants, fit, _ = solve_rows(tmp_path, rows)
    assert constants["C_BASE"] == pytest.approx(0.60, abs=1e-9) and constants["C_SPREAD"] == pytest.approx(0.0, abs=1e-9)
    assert fit["unflagged"]["tie_count"] == 61
    assert fit["unflagged"]["n"] == 100 and fit["unflagged"]["fitted"] is True
    assert fit["unflagged"]["loss"] == pytest.approx(0.0, abs=1e-9)  # pred = min(1, 0.60 + 0) = 0.60 = x
    assert fit["size_only"] == {"n": 0, "fitted": False, "loss": None, "tie_count": None}
    # Size penalty with base 0.60, spread 0: pred = min(1, 0.60 + P) = 1 for P in 0.40..1.00, so 61 ties; the largest P wins.
    constants, fit, _ = solve_rows(tmp_path, rows + [make_row(0.30, 1.0, "buy", hit=5.0)] * 30)
    assert constants["C_THIN_SIZE"] == pytest.approx(1.00, abs=1e-9) and fit["size_only"]["tie_count"] == 61
    assert fit["size_only"]["n"] == 30 and fit["size_only"]["loss"] == pytest.approx(0.0, abs=1e-9)  # pred = min(1, 0.60 + 1.00) = 1 = x
    assert fit["unflagged"]["tie_count"] == 61  # the size rows do not move the unflagged fit


def test_base_spread_tie_prefers_largest_sum(tmp_path):
    # h = 0.10: term 0.4, x = 0.5 on every fill. pred = base + 0.4 * spread, so loss is zero on base + 0.4 * spread = 0.5.
    # On the 0.01 grid that needs spread a multiple of 0.05 (0 to 1): 21 pairs, base = 0.50 - 0.4 * spread.
    # The largest sum is base 0.10, spread 1.00 (sum 1.10).
    rows = [make_row(0.10, 0.5, "buy")] * 100
    constants, fit, _ = solve_rows(tmp_path, rows)
    assert constants["C_BASE"] == pytest.approx(0.10, abs=1e-9) and constants["C_SPREAD"] == pytest.approx(1.00, abs=1e-9)
    assert fit["unflagged"]["loss"] == pytest.approx(0.0, abs=1e-9)
    assert fit["unflagged"]["tie_count"] == 21
    assert fit["unflagged"]["n"] == 100 and fit["unflagged"]["fitted"] is True


# --- 4. residual statistics -------------------------------------------------------------------------------------------

def test_residual_stats_hand_values(tmp_path):
    rows = [make_row(0.02, 0.39, "buy"), make_row(0.10, 0.35, "buy"),
            make_row(0.10, 0.9, "buy", hit=5.0, oi=50.0), make_row(0.10, 0.5, "buy", hit=5.0, oi=50.0)]
    _, _, res = solve_rows(tmp_path, rows)
    # constants stay current (no block reaches its threshold): both fills have c = 0.25 + 0.5 * 0.4 + 0.15 + 0.10 = 0.70
    both = res["fitted"]["both"]
    assert both["n"] == 2
    assert both["mean"] == pytest.approx(0.0, abs=1e-9) and both["median"] == pytest.approx(0.0, abs=1e-9)
    assert both["mean_abs"] == pytest.approx(0.2, abs=1e-9)
    assert both["p10"] == pytest.approx(-0.16, abs=1e-9) and both["p90"] == pytest.approx(0.16, abs=1e-9)
    # current F: tight quote c = 0.25 + 0.5 * 0.08 = 0.29 (residual +0.10); wide quote c = 0.45 (residual -0.10)
    cur = res["current_unflagged"]
    assert cur["n"] == 2 and cur["mean"] == pytest.approx(0.0, abs=1e-9)
    assert cur["mean_abs"] == pytest.approx(0.10, abs=1e-9)
    assert cur["p10"] == pytest.approx(-0.08, abs=1e-9) and cur["p90"] == pytest.approx(0.08, abs=1e-9)
    assert res["fitted"]["size_only"] == {"n": 0, "mean": None, "median": None, "mean_abs": None, "p10": None, "p90": None}


def test_residual_stats_asymmetric_per_group(tmp_path):
    # h = 0.10: term 0.4, base part 0.25 + 0.5 * 0.4 = 0.45. Thin size adds 0.15, thin oi adds 0.10, unknown size (thin_size 0.5) adds 0.075.
    # Centres c: unflagged 0.45, size_only 0.60, oi_only 0.55, both 0.70, unknown 0.525. Residual = x - c. 4 rows per group, all below thresholds.
    rows = [
        make_row(0.10, 0.85, "buy"), make_row(0.10, 0.55, "buy"), make_row(0.10, 0.45, "buy"), make_row(0.10, 0.35, "buy"),  # 0.40, 0.10, 0.00, -0.10
        make_row(0.10, 0.80, "buy", hit=5.0), make_row(0.10, 0.40, "buy", hit=5.0),
        make_row(0.10, 0.90, "buy", hit=5.0), make_row(0.10, 0.60, "buy", hit=5.0),  # 0.20, -0.20, 0.30, 0.00
        make_row(0.10, 0.25, "buy", oi=50.0), make_row(0.10, 0.65, "buy", oi=50.0),
        make_row(0.10, 0.75, "buy", oi=50.0), make_row(0.10, 0.85, "buy", oi=50.0),  # -0.30, 0.10, 0.20, 0.30
        make_row(0.10, 0.90, "buy", hit=5.0, oi=50.0), make_row(0.10, 0.80, "buy", hit=5.0, oi=50.0),
        make_row(0.10, 0.50, "buy", hit=5.0, oi=50.0), make_row(0.10, 0.20, "buy", hit=5.0, oi=50.0),  # 0.20, 0.10, -0.20, -0.50
        make_row(0.10, 0.625, "buy", hit=None), make_row(0.10, 0.825, "buy", hit=None),
        make_row(0.10, 0.425, "buy", hit=None), make_row(0.10, 0.725, "buy", hit=None),  # 0.10, 0.30, -0.10, 0.20
    ]
    constants, _, res = solve_rows(tmp_path, rows)
    assert (constants["C_BASE"], constants["C_SPREAD"], constants["C_THIN_SIZE"], constants["C_THIN_OI"]) == (0.25, 0.5, 0.15, 0.10)
    # (n, mean, median, mean_abs) from the residual lists above:
    # unflagged: mean 0.40 / 4; median (0.00 + 0.10) / 2; mean_abs 0.60 / 4
    # size_only: mean 0.30 / 4; median (0.00 + 0.20) / 2; mean_abs 0.70 / 4
    # oi_only: mean 0.30 / 4; median (0.10 + 0.20) / 2; mean_abs 0.90 / 4
    # both: mean -0.40 / 4; median (-0.20 + 0.10) / 2; mean_abs 1.00 / 4
    # unknown: mean 0.50 / 4; median (0.10 + 0.20) / 2; mean_abs 0.70 / 4
    expected = {"unflagged": (4, 0.10, 0.05, 0.15), "size_only": (4, 0.075, 0.10, 0.175), "oi_only": (4, 0.075, 0.15, 0.225),
                "both": (4, -0.10, -0.05, 0.25), "unknown": (4, 0.125, 0.15, 0.175)}
    for name, (n, mean, median, mean_abs) in expected.items():
        stats = res["fitted"][name]
        assert stats["n"] == n, name
        assert stats["mean"] == pytest.approx(mean, abs=1e-9), name
        assert stats["median"] == pytest.approx(median, abs=1e-9), name
        assert stats["mean_abs"] == pytest.approx(mean_abs, abs=1e-9), name


# --- 5. isolation from F ----------------------------------------------------------------------------------------------

def test_new_fills_leave_f_unchanged(tmp_path):
    out = tmp_path / "out"
    ff.run_fit(fills_file(tmp_path, recovery_rows()), out)  # fits 0.30 / 0.40, which differ from the current constants
    assert json.loads((out / "constants.json").read_text())["constants"]["C_BASE"] == pytest.approx(0.30, abs=1e-9)
    assert (fvf.C_BASE, fvf.C_SPREAD, fvf.C_THIN_SIZE, fvf.C_THIN_OI) == (0.25, 0.50, 0.15, 0.10)
    assert run.fvf_id() == "735f3e6cb2fa65c303599152cfa6fead2d7950e9bcb367117d50ed25faffc0b7"


# --- 6. failure inside solve leaves no output; progress lines carry no ticker ------------------------------------------

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
