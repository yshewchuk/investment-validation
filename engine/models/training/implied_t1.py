"""Champion `implied_t1` model — the quoted implied move at the last pre-print close.

STR-RUNUP buys volatility early and sells it immediately before the print, so
its entire P&L is the run-up in quoted implied move between those two dates. The
edge is a timing question — *how much is this name's implied move still going to
climb?* — and EXP-043 established that the answer is predictable: MAE 3.3–4.0pp,
r 0.60–0.72, top-vs-bottom decile realized spread 15–22pp, positive in every
year 2015–2026.

**Leak discipline is the whole difficulty here.** A model deciding at fourteen
trading days out may only see what was observable fourteen days out. That rules
out the panel's market-state block, every value of which is read at the last
pre-print close — the very quantity being predicted. So the features are the
event-history block (which depends only on prior quarters) plus
:func:`~engine.features.daily_state_frame` at the decision date, and nothing
else. Getting this wrong would produce a spectacular model and a worthless one.

Rather than one model per decision day, the decision days are pooled with
``days_to_print`` as a feature. That makes a single champion usable at any entry
day, which is what the scoring engine needs to compare entry days for the same
event — and the per-decile entry-day optimization is Phase 2's backlog item 3,
which this model is the input to rather than a substitute for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from engine.calendar import trading_calendar
from engine.features import (
    DAILY_STATE_COLUMNS,
    EVENT_HISTORY_FEATURES,
    daily_state_frame,
    entry_feature_frame,
)
from engine.models.training.common import (
    SEED,
    decile_spread,
    fit_final,
    log,
    regression_metrics,
    walk_forward,
)

__all__ = ["FEATURES", "TARGET", "DECISION_DAYS", "fit", "build_dataset", "train"]

TARGET = "im_t1"

#: Trading days before the last pre-print close at which a decision might be
#: taken. Mirrors the EXP-043 grid; ``-14`` is STR-RUNUP's default entry.
DECISION_DAYS = (25, 20, 15, 14, 10, 7, 5, 3, 2)

FEATURES: tuple[str, ...] = (
    *EVENT_HISTORY_FEATURES,
    *DAILY_STATE_COLUMNS,
    "days_to_print",
    "days_before_print",
)


def fit(X, y, seed: int = SEED):
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=seed,
    ).fit(X, y)


def build_dataset(
    events: pd.DataFrame,
    *,
    panel: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
    decision_days=DECISION_DAYS,
) -> pd.DataFrame:
    """One row per (event, decision day), with the T−1 implied move as target.

    ``events`` needs ``ticker``, ``event_date`` and ``session``. The target is
    read at the last pre-print close for that session — not at "the day before
    the event date", which is a different date for AMC names and would put the
    target one session away from where the trade actually exits.
    """
    cal = trading_calendar()
    rows: list[dict] = []
    for event in events.itertuples(index=False):
        session = getattr(event, "session", None)
        if session is None or pd.isna(session):
            continue
        event_date = pd.Timestamp(event.event_date).normalize()
        try:
            pre = cal.last_pre_print(event_date, str(session))
        except KeyError:
            continue
        for j in decision_days:
            try:
                as_of = cal.shift(pre, -int(j))
            except KeyError:
                continue
            rows.append(
                {
                    "ticker": str(event.ticker),
                    "event_date": event_date,
                    "session": str(session),
                    "last_pre_print": pre,
                    "entry_date": as_of,
                    "days_before_print": float(j),
                    "year": int(event_date.year),
                }
            )
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    log(f"implied_t1: {len(frame):,} (event, decision day) rows")
    frame = entry_feature_frame(frame, panel=panel, daily=daily, as_of_column="entry_date")

    # Target: the quoted implied move at the exit close. Read through the same
    # as-of machinery as the features so a coverage gap produces a missing
    # target rather than a value borrowed from a neighbouring date.
    target_rows = frame[["ticker", "last_pre_print"]].copy()
    target_rows = daily_state_frame(target_rows, daily=daily, as_of_column="last_pre_print")
    frame[TARGET] = target_rows["im"].to_numpy()

    # A run-up model must not be graded on rows where the implied move did not
    # move because the quote is simply the same stale row.
    frame["runup_pp"] = frame[TARGET] - frame["im"]
    return frame


def train(dataset: pd.DataFrame, *, seed: int = SEED, first_test_year: int = 2015):
    result = walk_forward(
        dataset, FEATURES, TARGET, fit, first_test_year=first_test_year, seed=seed
    )
    _add_runup_metrics(result)
    model = fit_final(dataset, FEATURES, TARGET, fit, seed=seed)
    return model, result


def _add_runup_metrics(result) -> None:
    """Score the model on the *run-up* as well as the level.

    This exists to stop a false comparison. EXP-043 predicted ``runup_pp`` — the
    *change* in implied move from the decision day to T−1 — and reported r
    0.60–0.72. This model predicts the T−1 **level**, which is a much more
    autocorrelated target, so its r is not comparable to that number and quoting
    the two side by side would flatter this one.

    MAE *is* comparable: predicted and realized run-up differ from the predicted
    and realized level by the same known ``im`` at the decision date, so the
    residual — and therefore the error — is identical.

    Recording both makes the honest comparison available rather than leaving the
    reader to notice the mismatch.
    """
    frame = result.frame
    if "im" not in frame.columns:
        return
    known = frame["im"].to_numpy(dtype=float)
    predicted_runup = frame["pred"].to_numpy(dtype=float) - known
    realized_runup = frame[TARGET].to_numpy(dtype=float) - known
    ok = np.isfinite(predicted_runup) & np.isfinite(realized_runup)
    if ok.sum() < 2:
        return
    stats = regression_metrics(realized_runup[ok], predicted_runup[ok])
    result.metrics["runup_r"] = stats["r"]
    result.metrics["runup_mae"] = stats["mae"]
    result.metrics["runup_decile_spread"] = decile_spread(
        realized_runup[ok], predicted_runup[ok]
    )
    log(
        f"run-up (the EXP-043-comparable target): r={stats['r']:.4f} "
        f"mae={stats['mae']:.4f}pp; the level r={result.metrics['r']:.4f} is NOT "
        "comparable to EXP-043's 0.60-0.72"
    )
