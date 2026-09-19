"""Build frozen residual-pool artifacts from explicit source rows (P5-4).

The training-side builders for ``engine.v2.models.residual_artifact``.
Scoring reads these artifacts and never calls this module (layer 6 is above
scoring; ``checks/layer_map.py`` forbids the reverse import), and both
builders open with the v2 no-fit guard, so a ``no_fit_guard()`` block -- the
score-request discipline -- makes a request-time rebuild raise rather than
quietly succeed.

Paired pool: the legacy construction is ``Scorer._residual_pool``
(``engine/score.py``): Tier-4 forecasts inner-joined on ``(ticker,
event_date)`` with the panel's realized ``abs_move`` and the crush table's
realized ``crush_pct_iv30``; ``err_move = abs_move - pred_abs_move`` and
``err_crush = crush_pct_iv30 - pred_iv_crush_30``; rows missing either error
dropped (missing means NaN: legacy's ``dropna`` keeps an infinite error).
:func:`build_paired_residual_pool_artifact` is that arithmetic over rows
the CALLER supplies in full. It takes no scorer context of any kind: legacy
scoped the crush table to the tickers a Scorer happened to load, and
that scoping is exactly what made the pool move with context. The causal
cutoff is explicit too -- events dated on/after ``cutoff`` never enter.

Driver pool: ``native_payoff.driver_residual_pool``'s filter and bucketing
(legacy ``registry.bucket_residuals``), run once here and frozen, so scoring
performs only the per-prediction bucket lookup.
"""
from __future__ import annotations

from datetime import date
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

from engine.v2.models.lineage import Lineage
from engine.v2.models.no_fit import forbid_fitting
from engine.v2.models.residual_artifact import (
    DriverResidualPoolArtifact,
    PairedResidualPoolArtifact,
    make_driver_residual_pool_artifact,
    make_paired_residual_pool_artifact,
)
from engine.v2.scoring import native_payoff

from .legacy_adapter import forbid_fitting as forbid_legacy_fitting

__all__ = ["build_driver_residual_pool_artifact", "build_paired_residual_pool_artifact",
           "freeze_stored_driver_residual_pool"]


def _require_lineage(lineage: Lineage) -> None:
    if not isinstance(lineage, Lineage) or not lineage.declared:
        raise ValueError("a frozen residual pool must declare its lineage "
                         "(data dependencies and/or upstream states)")


def _finite_field(row: Mapping[str, Any], name: str) -> float | None:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def build_driver_residual_pool_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
    model_id: str,
    fold: Any,
    lineage: Lineage,
    deciles: int = native_payoff.DECILES,
    min_pool: int = native_payoff.MIN_POOL,
) -> DriverResidualPoolArtifact | None:
    """Freeze one driver model's held-out ``(prediction, residual)`` pool.

    ``rows`` in the model's own stored order (order is part of the state:
    draws index into the pool). ``None`` when no valid row exists -- the
    same condition ``driver_residual_pool`` returns ``None`` on.
    """
    forbid_fitting("engine.v2.models.training.residuals.build_driver_residual_pool_artifact")
    _require_lineage(lineage)
    predictions: list[float] = []
    residuals: list[float] = []
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        prediction, residual = _finite_field(row, "prediction"), _finite_field(row, "residual")
        if prediction is not None and residual is not None:
            predictions.append(prediction)
            residuals.append(residual)
    if not residuals:
        return None
    buckets = native_payoff.bucket_residual_pool(
        predictions, residuals, deciles=deciles, min_pool=min_pool,
    )
    return make_driver_residual_pool_artifact(
        role=role, model_id=model_id, fold=fold, flat_residuals=residuals,
        buckets=buckets, deciles=deciles, min_pool=min_pool, lineage=lineage,
    )


def freeze_stored_driver_residual_pool(
    flat_residuals: Sequence[float],
    buckets: Mapping[str, Any] | None,
    *,
    role: str,
    model_id: str,
    lineage: Lineage,
    fold: Any = None,
    deciles: int = native_payoff.DECILES,
) -> DriverResidualPoolArtifact | None:
    """Freeze a pool that was ALREADY bucketed at training time, as stored.

    A full-refit champion (``engine.models.registry.ModelArtifact``) keeps
    only its flat held-out residuals and the decile buckets
    ``registry.bucket_residuals`` built from them -- the predictions are gone
    once the artifact is saved, so :func:`build_driver_residual_pool_artifact`
    cannot rebuild the buckets. This wraps the stored arrays unchanged
    (legacy ``ModelArtifact.residual_pool`` serves exactly these), keeping
    only finite flat residuals as ``ModelArtifact.__post_init__`` does.
    ``min_pool`` is the stored bucket floor, or ``native_payoff.MIN_POOL``
    when the champion carries no buckets (it is then unused). ``None`` when
    no finite residual exists (legacy raises ``RegistryError`` there).
    """
    forbid_legacy_fitting("engine.v2.models.training.residuals.freeze_stored_driver_residual_pool")
    forbid_fitting("engine.v2.models.training.residuals.freeze_stored_driver_residual_pool")
    _require_lineage(lineage)
    flat = [float(value) for value in flat_residuals if isfinite(float(value))]
    if not flat:
        return None
    frozen = None
    min_pool = native_payoff.MIN_POOL
    if buckets:
        min_pool = int(buckets.get("min_pool", 0))
        frozen = {"edges": [float(edge) for edge in buckets["edges"]],
                  "pools": [[float(value) for value in pool] for pool in buckets["pools"]]}
    return make_driver_residual_pool_artifact(
        role=role, model_id=model_id, fold=fold, flat_residuals=flat, buckets=frozen,
        deciles=deciles, min_pool=min_pool, lineage=lineage,
    )


def _day(value: Any) -> str:
    return str(value)[:10]


def _parse_day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _cutoff_bound(cutoff: Any) -> date | None:
    """The build's parsed cutoff: ``None`` only when there is none. A given
    cutoff that does not parse (malformed text, ``""``, NaT) is a caller
    error and raises -- it must never read as "no cutoff" and admit every
    row."""
    if cutoff is None:
        return None
    bound = _parse_day(cutoff)
    if bound is None:
        raise ValueError(f"paired residual pool cutoff {cutoff!r} is not a date")
    return bound


def _causal(day: str, bound: date | None) -> bool:
    """Dated strictly before ``bound`` (always, without one); an undated
    row cannot be shown causal, so it is dropped given a cutoff."""
    if bound is None:
        return True
    parsed = _parse_day(day)
    return parsed is not None and parsed < bound


def _present_field(row: Mapping[str, Any], name: str) -> float | None:
    """A row's float, or ``None`` where it is MISSING (absent, unparsable,
    None or NaN). +/-inf is present: the legacy paired pool drops only
    missing values (``dropna``), so an infinite error is kept (R4-20 gap 1).
    """
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        return None
    return None if value != value else value


def _index(rows: Iterable[Mapping[str, Any]], column: str) -> dict[tuple[str, str], list[float]]:
    """``(ticker, day) -> [values]``, keeping duplicates (an inner merge's
    cartesian semantics, exactly as ``pandas.merge`` would pair them)."""
    index: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = _present_field(row, column)
        if value is None:
            continue
        index.setdefault((str(row["ticker"]), _day(row["event_date"])), []).append(value)
    return index


def _paired_rows(forecasts, outcomes, crush, cutoff) -> list[tuple]:
    realized_move = _index(outcomes, "abs_move")
    realized_crush = _index(crush, "crush_pct_iv30")
    bound = _cutoff_bound(cutoff)
    rows: list[tuple] = []
    for forecast in forecasts:
        key = (str(forecast["ticker"]), _day(forecast["event_date"]))
        if not _causal(key[1], bound):
            continue
        pred_move = _present_field(forecast, "pred_abs_move")
        pred_crush = _present_field(forecast, "pred_iv_crush_30")
        if pred_move is None or pred_crush is None:
            continue
        for move in realized_move.get(key, ()):
            for crush_value in realized_crush.get(key, ()):
                err_move, err_crush = move - pred_move, crush_value - pred_crush
                # Legacy ``dropna``: inf - inf is NaN and drops; a lone inf stays.
                if err_move == err_move and err_crush == err_crush:
                    rows.append((key[1], key[0], pred_move, err_move, err_crush))
    return rows


def build_paired_residual_pool_artifact(
    forecasts: Iterable[Mapping[str, Any]],
    outcomes: Iterable[Mapping[str, Any]],
    crush: Iterable[Mapping[str, Any]],
    *,
    move_model_id: str,
    crush_model_id: str,
    cutoff: Any,
    lineage: Lineage,
) -> PairedResidualPoolArtifact:
    """Freeze the paired move/crush error pool from the full universe.

    ``forecasts``: ``{ticker, event_date, pred_abs_move, pred_iv_crush_30}``
    (the stored Tier-4 forecasts); ``outcomes``: ``{ticker, event_date,
    abs_move}``; ``crush``: ``{ticker, event_date, crush_pct_iv30}``. Every
    row the caller passes is eligible -- the builder never narrows by any
    context -- and the result depends only on the rows' content, not their
    order.
    """
    forbid_fitting("engine.v2.models.training.residuals.build_paired_residual_pool_artifact")
    _require_lineage(lineage)
    rows = _paired_rows(list(forecasts), list(outcomes), list(crush), cutoff)
    return make_paired_residual_pool_artifact(
        move_model_id=move_model_id, crush_model_id=crush_model_id,
        cutoff=cutoff, rows=rows, lineage=lineage,
    )
