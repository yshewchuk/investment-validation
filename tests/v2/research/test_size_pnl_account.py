"""EXP-188 Amendment 2: the cash-secured $100,000 account sensitivity. Synthetic fixtures only; no store."""
import math

import numpy as np
import pandas as pd
import pytest

from engine.structures import BUY, SELL
from experiments import size_pnl_account as account
from experiments import size_pnl_driver as driver

margin = account.margin
POLICY = account.POLICY
COLS = ["entry_date", "exit_date", "entry_cost", "exit_value", "secured_per_contract"]


def _leg(side, strike, qty):
    return {"side": side, "strike": float(strike), "qty": float(qty)}


def _trades(rows):
    return pd.DataFrame(rows, columns=COLS)


def _fixture():
    """Four trades on one entry date: the 80,000 trade cannot be funded (66% of 100,000 leaves 0.825 contracts)."""
    return _trades([
        ["2025-03-03", "2025-03-31", 0.4, 0.45, 20000.0],
        ["2025-03-03", "2025-03-31", 0.2, 0.1, 9500.0],
        ["2025-03-03", "2025-03-31", 0.5, 0.55, 30000.0],
        ["2025-03-03", "2025-03-31", 1.0, 0.9, 80000.0],
    ])


def test_secured_per_contract_bfly_like_and_empty():
    legs = [_leg(BUY, 100, 1), _leg(SELL, 95, 2), _leg(BUY, 90, 1), _leg(SELL, 85, 0)]  # the last is a reference leg
    assert account.secured_per_contract(legs) == 2 * 95 * 100 == 19000.0
    blob = {"entry": [{"side": "buy", "qty": 1, "strike": 100}, {"side": "sell", "qty": 2, "strike": 95}, {"side": "buy", "qty": 1, "strike": 90}]}
    assert account.secured_per_contract(legs) == margin.secured_per_contract(blob)
    assert account.secured_per_contract([_leg(BUY, 90, 1)]) == 0.0  # long puts secure nothing
    assert account.secured_per_contract([]) == 0.0


def test_straddle_rows_secured_zero_on_taken_and_nan_otherwise():
    dec = pd.DataFrame({"event_id": ["e1", "e2"], "status": ["taken", "not_taken"]})
    taken = pd.DataFrame({"event_id": ["e1"], "exit_value": [0.1], "rung": ["rung1"], "lots": [1], "pnl_dollars": [8.5],
                          "entry_cost": [0.015], "pnl_per_share": [0.085]})
    events = pd.DataFrame({"event_id": ["e1", "e2"], "pred_move": [3.0, 4.0], "pred_crush": [1.0, 2.0]})
    out = driver._rows("M0", "straddle", "R_L", dec, taken, events).set_index("event_id")
    assert out.loc["e1", "secured_per_contract"] == 0.0
    assert math.isnan(out.loc["e2", "secured_per_contract"])


def test_penny_debit_is_capped_by_margin_not_by_debit():
    rows = _trades([["2025-01-02", "2025-01-20", 0.015, 0.1, 9500.0]])
    t = margin.simulate(rows, **POLICY)
    contracts = float(t.loc[0, "contracts"])
    assert contracts == math.floor(0.25 * 0.66 * 100_000 / 9500) == 1
    assert contracts <= math.floor(0.66 * 100_000 / 9500) and contracts < 10_000 / (0.015 * 100)
    metrics = account.simulate_account(rows, "primary")
    assert metrics["funded"] == 1 and metrics["wanted"] == 1
    assert metrics["final_equity"] == pytest.approx(100_000 + 1 * (0.1 - 0.015) * 100)


def test_funded_bounded_by_wanted_and_secured_never_exceeds_cap():
    rows = _fixture()
    metrics = account.simulate_account(rows, "primary")
    assert metrics["wanted"] == 4 and metrics["funded"] == 3
    assert metrics["funded"] <= metrics["wanted"]
    t = margin.simulate(rows, **POLICY)
    assert t.loc[0, "secured_per_contract"] == 80000.0 and not t.loc[0, "funded"]  # expensive first, then refused
    lhs = t["secured_before"] + t["contracts"] * t["secured_per_contract"]
    assert np.all(lhs <= 0.66 * t["equity_at_entry"] + 1e-9)


def test_simulate_account_matches_margin_simulate_final_equity():
    rows = _fixture()
    t = margin.simulate(rows, **POLICY)
    assert account.simulate_account(rows, "primary")["final_equity"] == pytest.approx(t.attrs["final_equity"])


def test_straddle_secured_from_debit_is_never_infinite():
    rows = account.straddle_rows(_trades([["2025-02-03", "2025-02-28", 0.015, 0.02, 0.0]]))
    assert rows.loc[0, "secured_per_contract"] == pytest.approx(1.5)
    t = margin.simulate(rows[COLS], **POLICY)
    assert np.isfinite(t["contracts"].to_numpy(dtype=float)).all()
    metrics = account.simulate_account(rows, "primary")
    assert all(math.isfinite(metrics[k]) for k in ("final_equity", "total_return", "max_drawdown"))


def test_empty_rows_give_the_empty_metrics():
    assert account.simulate_account(_trades([]), "primary") == {
        "wanted": 0, "funded": 0, "final_equity": 100_000.0, "total_return": 0.0, "max_drawdown": 0.0, "peak_open": 0}


def _taken():
    return pd.DataFrame({
        "model_id": ["M0", "M0", "M1"], "book": ["short_vol"] * 3, "arm_or_rule": ["as_is"] * 3,
        "structure": ["TWIN-P", "BFLY-P", "TWIN-P"], "event_id": ["a", "b", "c"],
        "entry_date": ["2025-01-06", "2025-02-03", "2025-03-03"], "exit_date": ["2025-01-31", "2025-02-28", "2025-03-31"],
        "entry_cost": [0.5, 0.4, 0.5], "exit_value": [0.6, 0.3, 0.7], "secured_per_contract": [9500.0, 9500.0, 9500.0]})


def test_accounts_skips_mid_mark_when_exit_value_mid_absent_or_all_nan():
    assert {r["mark"] for r in account.accounts(_taken())} == {"primary"}
    assert {r["mark"] for r in account.accounts(_taken().assign(exit_value_mid=np.nan))} == {"primary"}
    marks = {r["mark"] for r in account.accounts(_taken().assign(exit_value_mid=[0.6, 0.3, 0.7]))}
    assert marks == {"primary", "mid"}


def test_differences_signs_median_minus_mean_and_other_minus_m0():
    metrics = {"wanted": 1, "funded": 1, "total_return": 0.0, "max_drawdown": 0.0, "peak_open": 1}
    results = [{"mark": "primary", "kind": "book", "key": "as_is", "model": model, "final_equity": equity, **metrics}
               for model, equity in (("H-sq", 100.0), ("H-abs", 120.0), ("M0", 100.0), ("M1", 90.0))]
    diffs = {d["contrast"]: d for d in account.differences(results, {"H": ("H-sq", "H-abs"), "E": ("E-raw", "E-log")}, ["M0", "M1", "H-sq", "H-abs"])}
    assert diffs["H"]["diff"] == 20.0  # median (H-abs) minus mean (H-sq)
    assert diffs["M0-vs-M1"]["diff"] == -10.0  # other minus M0: the other model did worse
    assert diffs["M0-vs-H-sq"]["diff"] == 0.0
    assert "E" not in diffs  # a pair with an absent member is skipped
    text = account.tables(results, list(diffs.values()))[0]
    assert "not the pre-registered verdict" in text and "primary marks" in text
