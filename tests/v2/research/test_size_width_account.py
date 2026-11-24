from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments import size_pnl_account, size_width_account

COLUMNS = ["entry_date", "exit_date", "entry_cost", "exit_value", "exit_value_mid", "secured_per_contract"]


def _one_trade(**overrides):
    row = {
        "entry_date": "2024-01-02",
        "exit_date": "2024-02-02",
        "entry_cost": 1.0,
        "exit_value": 1.5,
        "exit_value_mid": 2.0,
        "secured_per_contract": 5000.0,
    }
    row.update(overrides)
    return pd.DataFrame([row], columns=COLUMNS)


def test_policy_is_the_base_policy_with_a_200k_start():
    assert size_width_account.POLICY == {**size_pnl_account.POLICY, "start_equity": 200000.0}
    assert size_pnl_account.POLICY["start_equity"] == 100000.0


def test_empty_rows_return_the_untouched_account():
    result = size_width_account.simulate_account(pd.DataFrame(columns=COLUMNS), "mid")
    assert result["funded"] == 0
    assert result["final_equity"] == 200000.0
    assert result["total_return"] == 0.0


def test_one_winning_trade_mid_beats_primary_on_the_200k_base():
    primary = size_width_account.simulate_account(_one_trade(), "primary")
    mid = size_width_account.simulate_account(_one_trade(), "mid")
    assert primary["funded"] == 1
    assert mid["funded"] == 1
    assert mid["final_equity"] > 200000.0
    assert primary["final_equity"] > 200000.0
    assert mid["total_return"] == pytest.approx(mid["final_equity"] / 200000.0 - 1.0)
    assert mid["total_return"] > primary["total_return"]


def test_non_finite_exit_value_mid_raises_value_error():
    rows = _one_trade(exit_value_mid=np.inf)
    with pytest.raises(ValueError, match="finite"):
        size_width_account.simulate_account(rows, "mid")


def test_one_trade_final_equity_matches_the_hand_computed_literal():
    # By hand, under POLICY (start 200,000, cap 0.66, target share 0.25):
    #   headroom = 0.66 * 200,000 = 132,000; want = 0.25 * 132,000 / 5,000 = 6.6; floor -> 6 contracts.
    #   entry: 6 * 1.0 * 100 = 600 paid, so cash = 199,400.
    #   primary exit 1.5: 6 * 150 = 900 back, final = 200,300 (pnl 300 = 6 * (150 - 100)).
    #   mid exit 2.0:     6 * 200 = 1,200 back, final = 200,600 (pnl 600 = 6 * (200 - 100)).
    primary = size_width_account.simulate_account(_one_trade(), "primary")
    mid = size_width_account.simulate_account(_one_trade(), "mid")
    assert primary["final_equity"] == pytest.approx(200300.0)
    assert mid["final_equity"] == pytest.approx(200600.0)


def test_nat_exit_date_raises_value_error():
    rows = _one_trade(exit_date=None)
    with pytest.raises(ValueError, match="valid entry_date"):
        size_width_account.simulate_account(rows, "primary")


def test_bad_mark_raises_value_error():
    with pytest.raises(ValueError, match="mark must be"):
        size_width_account.simulate_account(_one_trade(), "bogus")


def test_bad_date_raises_value_error():
    rows = _one_trade(entry_date="not-a-date")
    with pytest.raises(ValueError):
        size_width_account.simulate_account(rows, "primary")


def test_secured_per_contract_counts_short_puts_at_strike_times_100():
    legs = [
        {"side": "sell", "qty": 1, "strike": 50.0},
        {"side": "buy", "qty": 1, "strike": 40.0},
    ]
    assert size_width_account.secured_per_contract(legs) == 5000.0


def test_quarter_returns_one_trade_pays_in_its_exit_quarter():
    quarters = pd.period_range("2024-01-01", periods=2, freq="Q")
    rows = _one_trade()  # exits 2024-02-02, so 2024Q1
    final = size_width_account.simulate_account(rows, "mid")["final_equity"]
    pnl = final - 200000.0
    assert pnl == pytest.approx(600.0)  # 6 contracts x (2.0 - 1.0) x 100
    got = size_width_account.quarter_returns(rows, quarters)
    assert got == pytest.approx([pnl / 200000.0, 0.0])


def test_quarter_returns_of_no_rows_are_all_zero():
    quarters = pd.period_range("2024-01-01", periods=2, freq="Q")
    assert size_width_account.quarter_returns(pd.DataFrame(columns=COLUMNS), quarters) == [0.0, 0.0]


def test_quarter_returns_include_an_exit_after_the_window():
    quarters = pd.period_range("2024-01-01", periods=2, freq="Q")
    rows = _one_trade(entry_date="2024-07-02", exit_date="2024-08-02")  # exits in 2024Q3, after the two given quarters
    gain = size_width_account.simulate_account(rows, "mid")["final_equity"] - 200000.0
    assert size_width_account.quarter_returns(rows, quarters) == pytest.approx([0.0, 0.0, gain / 200000.0])


def test_quarter_returns_compound_on_the_equity_at_each_quarter_start():
    rows = pd.DataFrame([
        {"entry_date": "2024-01-02", "exit_date": "2024-02-02", "entry_cost": 1.0, "exit_value": 1.5,
         "exit_value_mid": 2.0, "secured_per_contract": 5000.0},
        {"entry_date": "2024-03-04", "exit_date": "2024-05-03", "entry_cost": 1.0, "exit_value": 1.5,
         "exit_value_mid": 2.0, "secured_per_contract": 5000.0},
    ], columns=COLUMNS)
    quarters = pd.period_range("2024-01-01", periods=3, freq="Q")
    # trade 1 exits in Q1 (2024-02-02) before trade 2 is entered, so trade 2 is sized on the equity left after trade 1:
    # its gain is the account's final equity with both trades less the account's final equity with trade 1 alone.
    gain_1 = size_width_account.simulate_account(rows.iloc[[0]], "mid")["final_equity"] - 200000.0
    gain_2 = size_width_account.simulate_account(rows, "mid")["final_equity"] - (200000.0 + gain_1)
    got = size_width_account.quarter_returns(rows, quarters)  # trade 1 exits in Q1, trade 2 in Q2
    assert got == pytest.approx([gain_1 / 200000.0, gain_2 / (200000.0 + gain_1), 0.0])


def test_quarter_returns_fill_the_contiguous_range_after_the_window():
    quarters = pd.period_range("2024-07-01", periods=2, freq="Q")  # 2024Q3, 2024Q4
    rows = pd.DataFrame([
        {"entry_date": "2024-12-02", "exit_date": "2025-01-15", "entry_cost": 1.0, "exit_value": 1.5,
         "exit_value_mid": 2.0, "secured_per_contract": 5000.0},  # exits 2025Q1
        {"entry_date": "2025-06-02", "exit_date": "2025-07-15", "entry_cost": 1.0, "exit_value": 1.5,
         "exit_value_mid": 2.0, "secured_per_contract": 5000.0},  # exits 2025Q3
    ], columns=COLUMNS)
    got = size_width_account.quarter_returns(rows, quarters)
    assert len(got) == 5  # 2024Q3, 2024Q4, 2025Q1, 2025Q2, 2025Q3
    assert got[3] == 0.0  # 2025Q2 carries no funded pnl but stays in the contiguous range
    assert got[4] > 0.0
