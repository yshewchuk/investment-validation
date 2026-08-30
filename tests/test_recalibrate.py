"""Win-rate recalibration — the layer that makes a shipped probability honest.

The property under test: a forecast that is systematically over-confident in its
raw form becomes calibrated after passing through a map fitted on held-out
(realized) pairs, while its ranking is preserved. Fitting is causal — only pairs
whose event had closed by the cutoff may inform the map.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.recalibrate import MIN_PAIRS, fit_recalibration, load_pairs


def overconfident_pairs(
    n: int = 800,
    *,
    bias: float = 0.15,
    strategy: str = "STR-THRU",
    alpha: float = 0.5,
    exit_date: str = "2022-06-01",
    seed: int = 0,
) -> pd.DataFrame:
    """Pairs whose raw win is ``bias`` too high relative to the true outcome prob."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.05, 0.95, n)
    p_true = np.clip(raw - bias, 0.0, 1.0)
    outcome = (rng.uniform(0.0, 1.0, n) < p_true).astype(float)
    return pd.DataFrame(
        {
            "strategy": strategy,
            "fill_alpha": alpha,
            "event_id": [f"e{i}" for i in range(n)],
            "ticker": "X",
            "event_date": pd.Timestamp("2022-01-01"),
            "exit_date": pd.Timestamp(exit_date),
            "raw_win": raw,
            "outcome": outcome,
        }
    )


class TestFit:
    def test_removes_a_known_level_bias(self):
        m = fit_recalibration(
            "STR-THRU", 0.5, before=pd.Timestamp("2023-01-01"),
            pairs=overconfident_pairs(bias=0.15),
        )
        cal = m.transform(np.array([0.30, 0.50, 0.70]))
        truth = np.clip(np.array([0.30, 0.50, 0.70]) - 0.15, 0, 1)
        assert np.all(np.abs(cal - truth) < 0.06), (cal, truth)

    def test_is_monotone_and_bounded(self):
        m = fit_recalibration(
            "STR-THRU", 0.5, before=pd.Timestamp("2023-01-01"),
            pairs=overconfident_pairs(),
        )
        grid = np.linspace(0.0, 1.0, 50)
        cal = m.transform(grid)
        assert np.all(np.diff(cal) >= -1e-9)
        assert cal.min() >= 0.0 and cal.max() <= 1.0

    def test_selects_strategy_and_alpha(self):
        pool = pd.concat(
            [
                overconfident_pairs(strategy="STR-THRU", alpha=0.5, seed=1),
                overconfident_pairs(strategy="STR-RUNUP", alpha=0.0, seed=2),
            ],
            ignore_index=True,
        )
        m = fit_recalibration("STR-RUNUP", 0.0, before=pd.Timestamp("2023-01-01"), pairs=pool)
        assert m is not None and m.strategy == "STR-RUNUP" and m.alpha == 0.0

    def test_too_few_pairs_refuses_rather_than_fitting_noise(self):
        m = fit_recalibration(
            "STR-THRU", 0.5, before=pd.Timestamp("2023-01-01"),
            pairs=overconfident_pairs(n=MIN_PAIRS - 1),
        )
        assert m is None


class TestCausality:
    def test_only_pairs_closed_before_the_cutoff_are_used(self):
        past = overconfident_pairs(300, exit_date="2021-06-01", seed=3)
        future = overconfident_pairs(300, exit_date="2024-06-01", seed=4)
        pool = pd.concat([past, future], ignore_index=True)
        m = fit_recalibration(
            "STR-THRU", 0.5, before=pd.Timestamp("2022-01-01"), pairs=pool
        )
        assert m is not None and m.n == 300
        assert m.fitted_through == pd.Timestamp("2022-01-01")

    def test_a_pair_closing_on_the_cutoff_is_excluded(self):
        pool = overconfident_pairs(300, exit_date="2022-06-01")
        m_on = fit_recalibration(
            "STR-THRU", 0.5, before=pd.Timestamp("2022-06-01"), pairs=pool
        )
        m_after = fit_recalibration(
            "STR-THRU", 0.5, before=pd.Timestamp("2022-06-02"), pairs=pool
        )
        assert m_on is None
        assert m_after is not None and m_after.n == 300


class TestLoad:
    def test_missing_pairs_file_loads_empty(self, tmp_path):
        assert load_pairs(tmp_path / "absent.parquet").empty

    def test_fit_on_empty_pairs_returns_none(self, tmp_path):
        empty = load_pairs(tmp_path / "absent.parquet")
        assert fit_recalibration("STR-THRU", 0.5, before=None, pairs=empty) is None
