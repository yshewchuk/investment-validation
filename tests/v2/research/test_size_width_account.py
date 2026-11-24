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
    with pytest.raises(ValueError):
        size_width_account.simulate_account(rows, "mid")


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
