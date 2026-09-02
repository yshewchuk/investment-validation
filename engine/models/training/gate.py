"""Champion `gate` model — predicted per-trade return at mid fills.

The plan's second verdict-defining lesson is that the exposure is the asset and
the gate is the optimization: base exposures are thin (+2–4%/trade), and the
compounding comes from selecting within them. EXP-049 established the approach —
train directly on realized *mid-fill* returns rather than on a proxy — and got
+4.6%/trade in the top quintile.

Three things make this module's version different from the experiment's, all of
them consequences of Phase 1 owning the replay:

**It trains on the engine's own trades.** EXP-049 learned from a 3,622-row
legacy trade set priced at one fill convention. The gate here learns from
:mod:`engine.replay` output — the same code path that will price the trade it is
gating — over an unselected event universe, at the alpha grid, so the target is
literally ``ret`` at ``alpha = 0.5``.

**It is fitted per strategy.** A signal that concentrates a long straddle held
through a print is not the same signal that concentrates one sold before it; the
registry supports a champion per (strategy, role) precisely so these do not have
to share one.

**Its features are as-of the entry**, not as-of the print — see
:data:`~engine.features.EVENT_HISTORY_FEATURES` for why that distinction is
load-bearing for STR-RUNUP.

The threshold is chosen on the training years only, never on the year being
gated, and stored in the registry entry: it is part of the decision rule, not a
display preference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from engine.features import DAILY_STATE_COLUMNS, EVENT_HISTORY_FEATURES, entry_feature_frame
from engine.replay import legs_spot_dte
from engine.models.training.common import SEED, fit_final, log, walk_forward

__all__ = ["FEATURES", "TARGET", "GATE_ALPHA", "TOP_FRACTION", "fit", "build_dataset", "train"]

TARGET = "ret"

#: The gate is trained on mid fills because mid is the assumption the whole
#: program's positive verdict rests on, and Phase 5 exists to measure whether it
#: holds. Training on worst-case fills would optimize selection for a world we
#: do not intend to trade in.
GATE_ALPHA = 0.5

#: Share of candidates the gate passes. EXP-049's headline used the top quintile.
TOP_FRACTION = 0.20

FEATURES: tuple[str, ...] = (
    *EVENT_HISTORY_FEATURES,
    *DAILY_STATE_COLUMNS,
    "days_to_print",
    "entry_cost_pct",
    "dte_entry",
)


def fit(X, y, seed: int = SEED):
    return HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=seed,
    ).fit(X, y)


def build_dataset(
    trades: pd.DataFrame,
    *,
    panel: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
    alpha: float = GATE_ALPHA,
) -> pd.DataFrame:
    """Feature rows for the mid-fill slice of a strategy's replayed trades."""
    rows = trades[np.isclose(trades["fill_alpha"].astype(float), alpha)].copy()
    if rows.empty:
        return rows
    rows["event_date"] = pd.to_datetime(rows["event_date"])
    rows["entry_date"] = pd.to_datetime(rows["entry_date"])
    rows["year"] = rows["event_date"].dt.year

    # A book decided before it is entered is scored on what it EARNED at the
    # entry (TARGET stays `ret`, the realized return on the D0 fill) but may
    # only see what was knowable at the decision. That asymmetry is the whole
    # point: the gate learns to select a session early and is graded on what
    # the trade actually made. See guides/str_thru_t2_decision.md §5.5.
    decided_early = (
        "decision_date" in rows.columns
        and rows["decision_date"].notna().any()
        and (pd.to_datetime(rows["decision_date"]) < rows["entry_date"]).any()
    )
    if decided_early:
        rows["decision_date"] = pd.to_datetime(rows["decision_date"])
    # Spot and DTE live in the `legs` blob, not in the Tier-2 columns. Read them
    # back through the replay's own helper so the gate's premium feature is
    # quoted against the spot the trade was actually priced at.
    if "spot_entry" not in rows.columns or "dte_entry" not in rows.columns:
        rows["spot_entry"], rows["dte_entry"] = legs_spot_dte(rows)

    as_of_column = "decision_date" if decided_early else "entry_date"
    frame = entry_feature_frame(rows, panel=panel, daily=daily, as_of_column=as_of_column)

    # The premium as a share of spot: a straddle costing 12% of spot is a
    # different proposition from one costing 3%. For a book decided early it is
    # the QUOTED premium against the DECISION close's spot — the entry cost is
    # not knowable when the call is made, and using it would leak the fill.
    cost = frame["quoted_cost"] if decided_early else frame["entry_cost"]
    spot = frame["spot_decision"] if decided_early else frame["spot_entry"]
    frame["entry_cost_pct"] = (
        pd.to_numeric(cost, errors="coerce")
        / pd.to_numeric(spot, errors="coerce") * 100.0
    )
    if decided_early:
        frame["dte_entry"] = pd.to_numeric(frame["dte_decision"], errors="coerce")
    else:
        frame["dte_entry"] = pd.to_numeric(frame["dte_entry"], errors="coerce")
    log(f"gate dataset: {len(frame):,} trades at alpha={alpha}, as_of={as_of_column}")
    return frame


def choose_threshold(scores, top_fraction: float = TOP_FRACTION) -> float:
    """Score at or above which a candidate passes.

    Computed on training-year predictions only. A threshold picked on the year
    being gated is a rank the model could not have known, and it would make
    every walk-forward year look like a selection the gate actually made.
    """
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, 1.0 - top_fraction))


def train(
    dataset: pd.DataFrame,
    *,
    seed: int = SEED,
    first_test_year: int = 2020,
    top_fraction: float = TOP_FRACTION,
):
    """Walk-forward the gate, then report what gating would have bought.

    The extra statistics are the point of a gate and are not visible in r or
    MAE: the mean return of the passed set against the ungated base, and the
    top-minus-bottom decile spread.
    """
    result = walk_forward(
        dataset, FEATURES, TARGET, fit, first_test_year=first_test_year, seed=seed
    )
    model = fit_final(dataset, FEATURES, TARGET, fit, seed=seed)

    scored = result.frame
    threshold = choose_threshold(scored["pred"], top_fraction)
    passed = scored[scored["pred"] >= threshold]
    base_mean = float(scored[TARGET].mean())
    gated_mean = float(passed[TARGET].mean()) if len(passed) else float("nan")

    result.metrics.update(
        {
            "threshold": threshold,
            "top_fraction": top_fraction,
            "base_mean_ret": base_mean,
            "gated_mean_ret": gated_mean,
            "gate_lift": gated_mean - base_mean,
            "base_win_rate": float((scored[TARGET] > 0).mean()),
            "gated_win_rate": float((passed[TARGET] > 0).mean()) if len(passed) else None,
            "n_passed": int(len(passed)),
        }
    )
    log(
        f"gate: base {base_mean:+.4f}/trade → gated {gated_mean:+.4f} "
        f"(lift {gated_mean - base_mean:+.4f}) on {len(passed):,} of {len(scored):,}"
    )
    return model, result, threshold


def by_year_gate_table(result, threshold: float) -> pd.DataFrame:
    """Ungated vs gated mean return per out-of-sample year.

    A gate that only works in two of nine years is a gate that works in two of
    nine years, and the yearly table is where that shows up.
    """
    scored = result.frame.copy()
    scored["passed"] = scored["pred"] >= threshold
    rows = []
    for year, group in scored.groupby("year", sort=True):
        passed = group[group["passed"]]
        rows.append(
            {
                "year": int(year),
                "n": int(len(group)),
                "base_mean": float(group[TARGET].mean()),
                "n_passed": int(len(passed)),
                "gated_mean": float(passed[TARGET].mean()) if len(passed) else None,
                "gated_win": float((passed[TARGET] > 0).mean()) if len(passed) else None,
            }
        )
    return pd.DataFrame(rows)
