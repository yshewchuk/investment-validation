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
    legs = [_leg(BUY, 100, 1), _leg(SELL, 95, 2), _leg(BUY, 90, 1), _leg(SELL, 200, 0)]  # the last is a reference leg
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


def test_simulate_account_refuses_non_finite_inputs():
    mid_nan = _fixture().assign(exit_value_mid=[0.45, np.nan, 0.55, 0.9])
    with pytest.raises(ValueError, match="account inputs must be finite: exit_value_mid"):
        account.simulate_account(mid_nan, "mid")
    assert math.isfinite(account.simulate_account(mid_nan, "primary")["final_equity"])  # primary marks do not read exit_value_mid
    secured_nan = _fixture().assign(secured_per_contract=[20000.0, np.nan, 30000.0, 80000.0])
    with pytest.raises(ValueError, match="account inputs must be finite: secured_per_contract"):
        account.simulate_account(secured_nan, "primary")


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


@pytest.mark.parametrize("order", ["gain_first", "loss_first"])
def test_same_date_offsetting_exits_give_no_intra_date_drawdown(order):
    """Two funded trades (one contract each) exit on one date with +10 and -10: the date-level curve never dips, in either row order."""
    gain = ["2025-03-03", "2025-03-31", 0.4, 0.5, 9500.0]  # +10 per contract
    loss = ["2025-03-03", "2025-03-31", 0.5, 0.4, 9500.0]  # -10 per contract
    rows = _trades([gain, loss] if order == "gain_first" else [loss, gain])
    metrics = account.simulate_account(rows, "primary")
    assert metrics["wanted"] == 2 and metrics["funded"] == 2
    assert metrics["final_equity"] == pytest.approx(100_000.0)
    assert metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-12)  # the per-date sum is 0, so the curve is flat


# Exact oracle, hand-derived (not computed from the code). Policy: start 100,000, cap .66, target .25.
# Each trade is one contract: 0.25 * headroom / 10,000 = 1.65 (A, flat book), 1.4 (B, A still open), 1.65 (C), floored to 1.
# Primary pnl per contract: A +100, B -50, C +50. Mid pnl per contract: A +200, B 0, C 0.
ORACLE_COLS = ["entry_date", "exit_date", "entry_cost", "exit_value", "exit_value_mid", "secured_per_contract"]


def _oracle_frame():
    return pd.DataFrame([
        ["2024-01-02", "2024-01-10", 1.0, 2.0, 3.0, 10000.0],  # A
        ["2024-01-05", "2024-01-12", 1.0, 0.5, 1.0, 10000.0],  # B
        ["2024-02-01", "2024-02-05", 1.0, 1.5, 1.0, 10000.0],  # C
    ], columns=ORACLE_COLS)


def test_oracle_primary_marks():
    # Realised curve at exit dates: 100000 -> 100100 (01-10) -> 100050 (01-12) -> 100100 (02-05).
    metrics = account.simulate_account(_oracle_frame(), "primary")
    assert metrics["wanted"] == 3 and metrics["funded"] == 3
    assert metrics["final_equity"] == pytest.approx(100100.0)
    assert metrics["total_return"] == pytest.approx(0.001)
    assert metrics["peak_open"] == 2
    assert metrics["max_drawdown"] == pytest.approx(100050 / 100100 - 1)


def test_oracle_mid_marks_use_exit_value_mid():
    # Mid pnl: +200, 0, 0 -> the curve only rises, so no drawdown.
    metrics = account.simulate_account(_oracle_frame(), "mid")
    assert metrics["wanted"] == 3 and metrics["funded"] == 3
    assert metrics["final_equity"] == pytest.approx(100200.0)
    assert metrics["total_return"] == pytest.approx(0.002)
    assert metrics["max_drawdown"] == 0.0
    assert metrics["peak_open"] == 2
    assert account.simulate_account(_oracle_frame(), "primary")["final_equity"] != pytest.approx(metrics["final_equity"])


def test_oracle_accounts_book_rows_for_both_marks():
    frame = _oracle_frame().assign(model_id="M1", book="short_vol", arm_or_rule="as_is", structure="TWIN-P",
                                   event_id=["a", "b", "c"])
    book = {r["mark"]: r for r in account.accounts(frame) if r["kind"] == "book"}
    assert set(book) == {"primary", "mid"}
    assert book["primary"]["model"] == "M1" and book["primary"]["key"] == "as_is"
    assert book["primary"]["final_equity"] == pytest.approx(100100.0)
    assert book["mid"]["final_equity"] == pytest.approx(100200.0)
