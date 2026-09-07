"""Predict the absolute stock move from T-14 to the last pre-print close.

STR-RUNUP holds an entry-ATM straddle while the stock itself can move several
percent before the position is closed. EXP-149 showed that omitting this path
leaves forecast PnL badly biased: holding spot fixed understated return by 29
percentage points, while a causal absolute-move distribution reduced forecast
MAE from 33.5% to 25.2%.

The useful target is magnitude, not direction. The live-domain stock drift was
positive 60.4% of the time, but a signed walk-forward model had no ranking
skill and was worse than a zero-move forecast in most years. This model therefore
predicts ``100 * abs(log(spot_T1 / spot_T14))``. Direction remains a separate
draw in the PnL simulator.

Features are observed at T-14. The panel row is anchored near the print and
cannot supply those features without leakage, so the dataset uses the same
decision-date construction as :mod:`engine.models.training.implied_t1`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from engine.models.training import implied_t1
from engine.models.training.common import SEED, fit_final, log, walk_forward

__all__ = [
    "FEATURES",
    "TARGET",
    "HORIZON",
    "LogTargetRegressor",
    "fit",
    "build_dataset",
    "train",
]

TARGET = "runup_abs_move_d14"
HORIZON = 14
FEATURES: tuple[str, ...] = implied_t1.FEATURES


@dataclass
class LogTargetRegressor:
    """A log-target estimator whose public predictions stay in percentage points."""

    estimator: HistGradientBoostingRegressor

    def predict(self, X) -> np.ndarray:
        transformed = np.asarray(self.estimator.predict(X), dtype=float)
        return np.maximum(np.expm1(transformed), 0.0)


def fit(X, y, seed: int = SEED):
    target = np.log1p(np.maximum(np.asarray(y, dtype=float), 0.0))
    estimator = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=seed,
    ).fit(X, target)
    return LogTargetRegressor(estimator)


def _daily_for(events: pd.DataFrame) -> pd.DataFrame:
    from engine.data import store
    from engine.features import DAILY_STATE_FIELDS

    dates = pd.to_datetime(events["event_date"])
    first = int(dates.dt.year.min()) - 1
    last = int(dates.dt.year.max()) + 1
    return store.read_table(
        "daily_market",
        years=range(first, last + 1),
        columns=["ticker", "date", "src_iv", *DAILY_STATE_FIELDS.keys()],
    )


def build_dataset(
    events: pd.DataFrame,
    *,
    panel: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """One row per event, with T-14 features and realized T-14-to-T-1 move.

    ``last_pre_print`` is session-aware: an AMC event exits on the event-date
    close, while a BMO event exits on the preceding close. Missing or nonpositive
    spot observations produce a missing target rather than a zero move.
    """
    if daily is None:
        daily = _daily_for(events)
    frame = implied_t1.build_dataset(
        events,
        panel=panel,
        daily=daily,
        decision_days=(int(horizon),),
    )
    if frame.empty:
        frame[TARGET] = np.nan
        return frame

    from engine.features import daily_state_frame

    exit_rows = frame[["ticker", "last_pre_print"]].copy()
    exit_state = daily_state_frame(
        exit_rows,
        daily=daily,
        as_of_column="last_pre_print",
    )
    entry_spot = pd.to_numeric(frame["spot"], errors="coerce").to_numpy(float)
    exit_spot = pd.to_numeric(exit_state["spot"], errors="coerce").to_numpy(float)
    valid = (
        np.isfinite(entry_spot)
        & np.isfinite(exit_spot)
        & (entry_spot > 0)
        & (exit_spot > 0)
    )
    signed = np.full(len(frame), np.nan)
    signed[valid] = 100.0 * np.log(exit_spot[valid] / entry_spot[valid])
    frame["runup_signed_move_d14"] = signed
    frame[TARGET] = np.abs(signed)
    return frame


def train(dataset: pd.DataFrame, *, seed: int = SEED, first_test_year: int = 2018):
    result = walk_forward(
        dataset,
        FEATURES,
        TARGET,
        fit,
        first_test_year=first_test_year,
        seed=seed,
    )
    log(
        f"runup_move: OOS n={result.metrics['n']:,} "
        f"r={result.metrics['r']:.4f} mae={result.metrics['mae']:.4f}pp"
    )
    model = fit_final(dataset, FEATURES, TARGET, fit, seed=seed)
    return model, result
