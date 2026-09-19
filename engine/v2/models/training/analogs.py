"""Build the frozen board analog-matcher population from the full universe (P5-4).

The training-side builder for ``engine.v2.models.analog_artifact``. Scoring
reads those artifacts and never calls this module (layer 6 is above scoring),
and the builder opens with the v2 no-fit guard, so a request-time rebuild
under ``no_fit_guard()`` raises instead of quietly succeeding.

The legacy construction is ``engine.analogs.AnalogMatcher`` over
``Scorer.trades`` (``Scorer._enrich`` -> ``bucket_frame``), statement for
statement:

* population edges -- ``bucket_frame``'s implied-ratio tercile edges over the
  WHOLE enriched frame (every strategy and alpha): the finite ratios' 1/3 and
  2/3 quantiles, or ``(0.9, 1.1)`` below 30 finite ratios;
* the ``(strategy, alpha)`` pool -- ``strategy ==`` and
  ``np.isclose(fill_alpha, alpha)``;
* the causal slice at ``cutoff`` -- ``exit_date < cutoff`` (normalized), its
  own edges from the slice's finite ratios (same quantile rule), and the
  slice re-bucketed on them; with no cutoff, the pool as bucketed on the
  population edges.

What this builder takes is the enriched, bucketed frame itself -- the rows
legacy's ``bucket_frame`` produced -- and nothing about any scorer context.
``tools/phase5_datasets.board_analog_trades`` builds that frame from the full
panel; a Scorer built on a narrower context produced a different frame, and
that difference is the defect this artifact removes.

Lineage. Every artifact declares ``tier2.trades``, ``tier3.panel`` and
``tier2.daily_market`` with NO time bound: the population edges are read
from every trade, future ones included (legacy behaviour, kept), so a
correction anywhere can move them. That makes the analog states conservative
under correction propagation -- any hit on those tables invalidates them all
-- which is the honest reading of what they depend on.
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from engine.v2.models.analog_artifact import (
    BoardAnalogPoolArtifact,
    make_board_analog_pool_artifact,
)
from engine.v2.models.lineage import DataDependency, Lineage
from engine.v2.models.no_fit import forbid_fitting
from engine.v2.scoring.native_analog import TERCILE_LABELS

__all__ = [
    "ANALOG_TRADE_COLUMNS",
    "BOARD_ANALOG_LINEAGE",
    "build_board_analog_pool_artifact",
    "iter_board_analog_pool_artifacts",
    "population_implied_edges",
]

#: Columns the builder reads from the enriched, bucketed trades frame.
ANALOG_TRADE_COLUMNS = (
    "trade_id", "strategy", "fill_alpha", "event_date", "exit_date",
    "mcap_bucket", "dte_band", "moneyness_band", "implied_ratio", "ret",
)
#: The source tables behind every board analog artifact (see module doc).
BOARD_ANALOG_LINEAGE = Lineage(data=tuple(
    DataDependency(table=table) for table in ("tier2.trades", "tier3.panel",
                                              "tier2.daily_market")))
#: Legacy's quantile floor and fallback edges (``bucket_frame``/``match``).
_MIN_FINITE = 30
_FALLBACK_EDGES = (0.9, 1.1)


def _edges_from(ratios: pd.Series) -> tuple[float, float]:
    """Legacy's tercile edges over a ratio column (order-independent)."""
    values = pd.to_numeric(ratios, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) >= _MIN_FINITE:
        return tuple(float(edge) for edge in np.quantile(finite, [1 / 3, 2 / 3]))
    return _FALLBACK_EDGES


def population_implied_edges(trades: pd.DataFrame) -> tuple[float, float]:
    """``bucket_frame``'s population edges: over EVERY row of the frame."""
    return _edges_from(trades["implied_ratio"])


def _terciles(ratios: pd.Series, edges: tuple[float, float]) -> np.ndarray:
    """Legacy ``engine.analogs._bucket(ratio, edges, ("low", "mid", "high"))``."""
    values = pd.to_numeric(ratios, errors="coerce").to_numpy(dtype=float)
    index = np.searchsorted(np.asarray(edges, dtype=float), values, side="right")
    out = np.array(TERCILE_LABELS, dtype=object)[np.clip(index, 0, len(TERCILE_LABELS) - 1)]
    out[~np.isfinite(values)] = None
    return out


def _label(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    return str(value)


def _days(values: pd.Series) -> list[str | None]:
    stamps = pd.to_datetime(values)
    return [None if pd.isna(stamp) else str(stamp.date()) for stamp in stamps]


def _returns(values: pd.Series) -> list[float | None]:
    """``_summarize``'s ``to_numeric`` + finite filter: a dropped return is ``None``."""
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return [float(value) if np.isfinite(value) else None for value in numbers]


def _check_columns(trades: pd.DataFrame) -> None:
    missing = [column for column in ANALOG_TRADE_COLUMNS if column not in trades.columns]
    if missing:
        raise ValueError(f"analog trades frame is missing columns: {missing}")


def _require_lineage(lineage: Lineage) -> None:
    if not isinstance(lineage, Lineage) or not lineage.declared:
        raise ValueError("a frozen analog pool must declare its lineage")


def _base_pool(trades: pd.DataFrame, strategy: str, alpha: float) -> pd.DataFrame:
    return trades[(trades["strategy"] == strategy)
                  & np.isclose(trades["fill_alpha"].astype(float), float(alpha))]


def _freeze(base: pd.DataFrame, *, strategy: str, alpha: float, cutoff: Any,
            population_edges: tuple[float, float], lineage: Lineage) -> BoardAnalogPoolArtifact:
    pool = base
    causal_edges = None
    edges = population_edges
    if cutoff is not None:
        stamp = pd.Timestamp(cutoff).normalize()
        pool = base[pd.to_datetime(base["exit_date"]) < stamp]
        if len(pool):
            causal_edges = _edges_from(pool["implied_ratio"])
            edges = causal_edges
    terciles = _terciles(pool["implied_ratio"], edges)
    rows = zip(
        pool["trade_id"].astype(str).tolist(), _days(pool["event_date"]),
        _days(pool["exit_date"]), map(_label, pool["mcap_bucket"].tolist()),
        map(_label, pool["dte_band"].tolist()), map(_label, pool["moneyness_band"].tolist()),
        terciles.tolist(), _returns(pool["ret"]),
    )
    return make_board_analog_pool_artifact(
        strategy=strategy, alpha=alpha,
        cutoff=None if cutoff is None else str(pd.Timestamp(cutoff).date()),
        population_edges=population_edges, causal_edges=causal_edges,
        rows=list(rows), lineage=lineage,
    )


def build_board_analog_pool_artifact(
    trades: pd.DataFrame,
    *,
    strategy: str,
    alpha: float,
    cutoff: Any,
    lineage: Lineage = BOARD_ANALOG_LINEAGE,
    population_edges: tuple[float, float] | None = None,
) -> BoardAnalogPoolArtifact:
    """Freeze one ``(strategy, alpha, cutoff)`` causal slice.

    ``trades`` is the FULL enriched, bucketed analog population (every
    strategy and alpha: the population edges are computed over all of it
    unless ``population_edges`` passes them in precomputed).
    """
    forbid_fitting("engine.v2.models.training.analogs.build_board_analog_pool_artifact")
    _require_lineage(lineage)
    _check_columns(trades)
    edges = population_implied_edges(trades) if population_edges is None else tuple(
        float(edge) for edge in population_edges)
    return _freeze(_base_pool(trades, strategy, alpha), strategy=strategy, alpha=alpha,
                   cutoff=cutoff, population_edges=edges, lineage=lineage)


def iter_board_analog_pool_artifacts(
    trades: pd.DataFrame,
    keys: Iterable[tuple[str, float, Any]],
    *,
    lineage: Lineage = BOARD_ANALOG_LINEAGE,
) -> Iterator[tuple[tuple[str, float, Any], BoardAnalogPoolArtifact]]:
    """``(key, artifact)`` for every requested causal key, one at a time.

    Each ``(strategy, alpha)`` pool is sliced once; artifacts are yielded,
    not collected, so a caller writing each to disk holds one at a time.
    """
    forbid_fitting("engine.v2.models.training.analogs.iter_board_analog_pool_artifacts")
    _require_lineage(lineage)
    _check_columns(trades)
    edges = population_implied_edges(trades)
    bases: dict[tuple[str, float], pd.DataFrame] = {}
    for strategy, alpha, cutoff in keys:
        pair = (str(strategy), round(float(alpha), 4))
        if pair not in bases:
            bases[pair] = _base_pool(trades, *pair)
        yield (strategy, alpha, cutoff), _freeze(
            bases[pair], strategy=pair[0], alpha=pair[1], cutoff=cutoff,
            population_edges=edges, lineage=lineage)
