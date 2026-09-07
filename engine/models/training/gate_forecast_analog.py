"""STR-THRU gate: the 41 registered features plus the forecast and analog
evidence EXP-145 found had ranking value the incumbent's feature set alone
does not capture ("arm 7" in that comparison).

Two extra signal groups, both already computed elsewhere in the engine and
reused here rather than re-derived:

* **Forecast** — the size model's own predicted absolute move
  (``engine.data.features.tier4``'s ``pred_abs_move``, the same number
  ``engine.payoff.PAYOFF_DRIVER['STR-THRU'] == 'abs_move'`` already prices
  STR-THRU from), plus its p10/p90/sd and ``forecast_edge`` — the predicted
  move minus the quoted implied move at entry, both percent of spot. Positive
  means the model expects a bigger move than the market is pricing, which is
  STR-THRU's stated thesis (a mispriced expected move).
* **Analog** — the matched historical-trade statistics from
  :mod:`engine.analogs`, the same empirical layer the board already shows
  next to the model layer for every event.

Both are servable for an event that has not happened yet: the forecast
through :func:`engine.data.features.tier4.serving_model` (the same leak-safe
path :meth:`engine.score.Scorer._size_from_forecast` already uses for
forecast-sized structures), and the analog stats through the exact
:class:`engine.analogs.AnalogMatcher` call
:meth:`engine.score.Scorer._score_analogs` already runs for every event on
the board. Neither is a second implementation with a second set of answers —
see ``engine/score.py``'s ``_forecast_for_gate``/``_gate_feature_frame`` for
the live-serving side of this module's training-time counterpart.

Registered as ``gate_midfill_str_thru_forecast_analog`` — a distinct id from
the incumbent ``gate_midfill_str_thru``, sharing its (strategy, role) key so
promoting it demotes the incumbent atomically (:func:`engine.models.registry.register`).
"""
from __future__ import annotations

import pandas as pd

from engine.analogs import bucket_frame, match_frame
from engine.data.features import tier4
from engine.models.training import gate as gate_mod
from engine.models.training.common import SEED

__all__ = [
    "STRATEGY",
    "FORECAST_COLS",
    "ANALOG_COLS",
    "EXTRA_FEATURES",
    "FEATURES",
    "TARGET",
    "GATE_ALPHA",
    "TOP_FRACTION",
    "fit",
    "build_dataset",
    "train",
    "choose_threshold",
    "by_year_gate_table",
]

#: This module is STR-THRU specific — the extra features were validated for
#: this strategy (EXP-145) and nothing else. A future STR-RUNUP variant is a
#: new experiment, not a parameter here.
STRATEGY = "STR-THRU"

FORECAST_COLS: tuple[str, ...] = (
    "pred_abs_move", "pred_abs_move_p10", "pred_abs_move_p90", "pred_abs_move_sd",
    "forecast_edge",
)
ANALOG_COLS: tuple[str, ...] = ("analog_mean", "analog_win_rate", "analog_n")
EXTRA_FEATURES: tuple[str, ...] = FORECAST_COLS + ANALOG_COLS

#: The incumbent's 41 features plus the six above. Order matters only for
#: readability — both training and live serving index by name, never by
#: position.
FEATURES: tuple[str, ...] = tuple(gate_mod.FEATURES) + EXTRA_FEATURES

TARGET = gate_mod.TARGET
GATE_ALPHA = gate_mod.GATE_ALPHA
TOP_FRACTION = gate_mod.TOP_FRACTION


def fit(X, y, seed: int = SEED):
    return gate_mod.fit(X, y, seed=seed)


def _attach_forecast(frame: pd.DataFrame) -> pd.DataFrame:
    """Join the stored Tier-4 forecast and derive ``forecast_edge``.

    ``tier4.load_forecasts()`` is the walk-forward-safe, monthly-fold table
    already on disk — no fresh fitting here, same as every other historical
    consumer of Tier 4.
    """
    forecasts = tier4.load_forecasts()
    if forecasts is None or forecasts.empty:
        raise RuntimeError(
            "data/features/tier4_forecasts.parquet is missing or empty — "
            "run engine.data.features.tier4.build_table first"
        )
    keep = ["ticker", "event_date",
            *[c for c in FORECAST_COLS if c != "forecast_edge"]]
    fc = forecasts[keep].drop_duplicates(["ticker", "event_date"]).copy()
    fc["event_date"] = pd.to_datetime(fc["event_date"])

    out = frame.copy()
    out["event_date"] = pd.to_datetime(out["event_date"])
    out = out.merge(fc, on=["ticker", "event_date"], how="left")
    out["forecast_edge"] = out["pred_abs_move"] - out["im"]
    return out


def _attach_analogs(frame: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Join per-event matched analog statistics, computed causally.

    Reuses :class:`engine.score.Scorer` for the same enrichment (mcap,
    implied move, prior moves, spot/DTE from the legs blob) the live board's
    analog layer runs on — imported here rather than at module load, since
    ``engine.score`` does not import this module and importing it eagerly
    would pull the whole scoring stack (registry, replay, structures) into
    every training-module import.
    """
    from engine.score import Scorer

    scorer = Scorer(trades=trades)
    bucketed = bucket_frame(scorer.trades)
    matched = match_frame(
        bucketed, scorer.matcher, strategy=STRATEGY, alpha=GATE_ALPHA,
        progress_every=2000,
    )
    return frame.merge(
        matched[["event_id", "analog_mean", "analog_win_rate", "analog_n"]],
        on="event_id", how="left",
    )


def build_dataset(
    trades: pd.DataFrame,
    *,
    panel: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
    alpha: float = GATE_ALPHA,
) -> pd.DataFrame:
    """The incumbent's feature frame (:func:`gate.build_dataset`) plus forecast
    and analog columns joined on."""
    base = gate_mod.build_dataset(trades, panel=panel, daily=daily, alpha=alpha)
    if base.empty:
        return base
    base = _attach_forecast(base)
    base = _attach_analogs(base, trades)
    return base


def train(
    dataset: pd.DataFrame,
    *,
    seed: int = SEED,
    first_test_year: int = 2020,
    top_fraction: float = TOP_FRACTION,
):
    return gate_mod.train(
        dataset, seed=seed, first_test_year=first_test_year,
        top_fraction=top_fraction, features=FEATURES,
    )


def choose_threshold(scores, top_fraction: float = TOP_FRACTION) -> float:
    return gate_mod.choose_threshold(scores, top_fraction)


def by_year_gate_table(result, threshold: float) -> pd.DataFrame:
    return gate_mod.by_year_gate_table(result, threshold)
