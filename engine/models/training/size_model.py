"""Champion `size` model — predicted |earnings move|, as a percent of spot.

The v1.3 architecture: an equal-weight blend of OLS and an MLP(64, 32), which
beat either component across the EXP-030 / 031 / 036 bake-offs and which five
separate attempts to extend the feature space failed to improve on.

**One deliberate change from the legacy feature list, and the reason for it.**
The research model's ten features include ``implied_move`` — the oquants quoted
implied move for the event being scored. It is genuinely pre-print information,
so it leaks nothing and it backtests beautifully. It is also unavailable for any
event that has not happened yet: it is read from the oquants moves file, which
only lists realized events. A model built on it would have scored every
historical claim in this repo and had nothing to read on the first morning it
was asked to earn its keep.

The live equivalent is ``or_implied`` — ORATS' quoted implied move at the last
pre-print close, sourced from ``daily_market``, present for upcoming events. This
module trains on that instead, and :func:`compare_feature_sets` measures what
the substitution costs so the decision is recorded as a number rather than an
assertion. ``engine.features.assert_live_available`` stops the original list
from being re-adopted by accident.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from engine.models.training.common import (
    SEED,
    BlendModel,
    fit_final,
    log,
    walk_forward,
)

__all__ = ["FEATURES", "LEGACY_FEATURES", "TARGET", "fit", "train", "compare_feature_sets"]

TARGET = "abs_move"

#: The servable feature list. ``or_implied`` replaces the legacy
#: ``implied_move``; ``mcap_log`` is the era-normalized market cap Phase 0
#: corrected (EXP-037 — small caps move ~2.3× mega caps, the effect the legacy
#: panel's understated pre-2017 caps partly buried).
#:
#: The three absolute-valued inputs came from EXP-109. This blend is half
#: linear, and a signed input against a magnitude target is often V-shaped —
#: ``mean_prior_move`` against ``abs_move`` runs 8.35 → 4.60 → 7.82 across its
#: deciles on a Spearman of +0.013, a shape the OLS half cannot represent at
#: all. Adding the magnitudes improved walk-forward OOS MAE in 11 of 14 years
#: (Wilcoxon p=0.0134), and the gain survived dropping the best year. It is a
#: 0.36% relative improvement: real, consistent, and small.
#:
#: ``has_implied_quote`` came from EXP-111 and is adopted as a KNOWN NULL on
#: accuracy (+0.0052pp, 9/14 years, p=0.27). ``or_implied`` encodes "no quote"
#: as 0, and that zero is a liquidity fact — 31.5% of the smallest market-cap
#: decile against 1.6% of the largest — so the column carries a price and an
#: availability flag on one axis. The indicator separates them, and it is what
#: preserves the liquidity signal when the value is later nulled for being
#: wrong. Two separate models for the quoted and unquoted populations was tested
#: in the same run and is reliably WORSE (3/14 years, p=0.03).
FEATURES = (
    "has_implied_quote",
    "mean_prior_abs_move",
    "abs_dist_ema",
    "abs_dist_high",
    "ema12r_abs",
    "mean_prior_move",
    "signed_streak",
    "dist_high",
    "dist_ema",
    "spy_vol20",
    "spy_dd252",
    "mean_prior_implied_move",
    "or_implied",
    "or_rvol30",
    "mcap_log",
)

#: The research list, kept only so the substitution can be measured. Never
#: registered — it contains a feature no live event can supply.
LEGACY_FEATURES = (
    "ema12r_abs",
    "mean_prior_move",
    "signed_streak",
    "dist_high",
    "dist_ema",
    "spy_vol20",
    "spy_dd252",
    "mean_prior_implied_move",
    "implied_move",
    "or_rvol30",
)

#: Sanity bounds from the legacy candidate builder. Values outside them are
#: sentinel damage or unit errors, not observations.
BOUNDS = {"or_implied": (0.0, 60.0), "or_rvol30": (0.0, 700.0), "abs_move": (0.0, 200.0)}

#: The panel admits events at 4 prior; the size model's headline feature is a
#: 12-span EMA, so it wants more history than that to be worth trusting.
MIN_PRIOR = 4


def fit(X, y, seed: int = SEED):
    """OLS + MLP(64, 32), equal weight.

    The MLP is scaled — an unscaled network over features spanning market cap in
    logs and streak counts in single digits trains on whichever feature happens
    to be largest.
    """
    ols = LinearRegression().fit(X, y)
    nn = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=seed,
        ),
    ).fit(X, y)
    return BlendModel(models=(ols, nn))


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    """Clean the panel down to the rows a size model may learn from."""
    data = panel.copy()
    for column, (lo, hi) in BOUNDS.items():
        if column in data.columns:
            values = pd.to_numeric(data[column], errors="coerce")
            data[column] = values.where((values >= lo) & (values <= hi))
    return data[data["n_prior"] >= MIN_PRIOR].reset_index(drop=True)


def train(panel: pd.DataFrame, *, seed: int = SEED, first_test_year: int = 2013):
    """Walk-forward evaluation plus the refit model that serves live events."""
    data = prepare(panel)
    result = walk_forward(
        data, FEATURES, TARGET, fit, first_test_year=first_test_year, seed=seed
    )
    model = fit_final(data, FEATURES, TARGET, fit, seed=seed)
    return model, result


def compare_feature_sets(panel: pd.DataFrame, *, seed: int = SEED, first_test_year: int = 2013):
    """What the live-servable feature list costs against the research one.

    Both are evaluated on the same rows — the intersection where every feature
    of *either* list is present — because a model scored on a larger or easier
    subset would win on coverage rather than on merit.
    """
    data = prepare(panel)
    both = list(dict.fromkeys([*FEATURES, *LEGACY_FEATURES]))
    complete = np.isfinite(data[both].to_numpy(dtype=float)).all(axis=1)
    complete &= np.isfinite(data[TARGET].to_numpy(dtype=float))
    shared = data[complete].reset_index(drop=True)
    log(f"feature-set comparison on {len(shared):,} rows complete on both lists")

    out = {}
    for label, features in (("servable", FEATURES), ("legacy", LEGACY_FEATURES)):
        result = walk_forward(
            shared, features, TARGET, fit, first_test_year=first_test_year, seed=seed
        )
        out[label] = result.metrics
    return out
