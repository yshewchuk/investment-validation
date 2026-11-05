# packages: models

import pandas as pd
import pytest

from engine.v2.models.training.conditional_exit import conditional_exit


def old_conditional_exit(raw: pd.DataFrame) -> pd.DataFrame:
    """The EXP-161 implementation before it moved into engine."""
    out = raw.copy()
    hold = (~out["exit_arb_ok"].fillna(True).astype(bool)) | (out["exit_value"] < 0)
    out.loc[hold, "exit_value"] = out.loc[hold, "exit_value_expiry"]
    out = out.dropna(subset=["entry_cost", "exit_value"]).copy()
    out["pnl"] = out["exit_value"] - out["entry_cost"]
    out["ret"] = out["pnl"] / out["entry_cost"]
    if (out["ret"] < -1.0 - 1e-8).any():
        raise RuntimeError("conditional exits still contain a loss below the debit")
    return out


@pytest.fixture
def candidate_fills():
    return pd.DataFrame({
        "event_id": ["repair_arb", "repair_negative", "keep_null_arb",
                     "drop_missing_cost", "drop_missing_expiry"],
        "exit_arb_ok": [False, True, None, False, True],
        "exit_value": [3.0, -1.0, 2.0, 1.0, -1.0],
        "exit_value_expiry": [4.0, 2.0, 9.0, 5.0, None],
        "entry_cost": [5.0, 1.0, 1.0, None, 2.0],
    })


def test_shared_conditional_exit_matches_exp161(candidate_fills):
    source = candidate_fills.copy(deep=True)
    expected = old_conditional_exit(candidate_fills)
    actual = conditional_exit(candidate_fills)

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(candidate_fills, source)
    assert actual["event_id"].tolist() == [
        "repair_arb", "repair_negative", "keep_null_arb"]
    assert actual["exit_value"].tolist() == [4.0, 2.0, 2.0]
    assert actual["pnl"].tolist() == [-1.0, 1.0, 1.0]
    assert actual["ret"].tolist() == [-0.2, 1.0, 1.0]


def test_shared_conditional_exit_accepts_exact_full_debit_loss():
    frame = pd.DataFrame({
        "exit_arb_ok": [True],
        "exit_value": [0.0],
        "exit_value_expiry": [0.0],
        "entry_cost": [1.0],
    })

    result = conditional_exit(frame)

    assert result["exit_value"].tolist() == [0.0]
    assert result["entry_cost"].tolist() == [1.0]
    assert result["pnl"].tolist() == [-1.0]
    assert result["ret"].tolist() == [-1.0]


def test_shared_conditional_exit_refuses_loss_below_full_debit():
    frame = pd.DataFrame({
        "exit_arb_ok": [True],
        "exit_value": [-1.0],
        "exit_value_expiry": [-1.2e-8],
        "entry_cost": [1.0],
    })

    with pytest.raises(RuntimeError, match="loss below the debit"):
        conditional_exit(frame)


def test_shared_conditional_exit_accepts_loss_within_tolerance():
    frame = pd.DataFrame({
        "exit_arb_ok": [False],
        "exit_value": [1.0],
        "exit_value_expiry": [-1.0e-8],
        "entry_cost": [1.0],
    })

    result = conditional_exit(frame)

    assert result["ret"].iloc[0] == pytest.approx(-1.0 - 1.0e-8)


def test_shared_conditional_exit_missing_column_raises_without_mutation():
    frame = pd.DataFrame({
        "exit_arb_ok": [False],
        "exit_value": [1.0],
        "entry_cost": [1.0],
    })
    source = frame.copy(deep=True)

    with pytest.raises(KeyError):
        conditional_exit(frame)
    pd.testing.assert_frame_equal(frame, source)
