"""The DYN-SV chooser's derived columns, natively (R4-20 remaining gap (a)).

Legacy builds the champion's 67 inputs in ``engine/score.py``
``Scorer._chooser_frame``. Seventeen are primitive facts it reads off the
feature frame (the event-history recursions, the market/regime block and
``dte_entry``); every other one is COMPUTED there from the scoring pass. This
module is that arithmetic, a line-for-line port over native values:

* :func:`entry_cost_pct` -- ``Scorer._entry_features``' ``entry_cost_pct``;
* :func:`size_band_columns` -- the sizing forecast and its Tier-4 band
  (``pred_abs_move``/``_sd``/``_p10``/``_p90``/``_resid_n`` and
  ``tier4_pred_abs_move_sd``) from the served size fold's pool;
* :func:`geometry_columns` -- ``half_width_pct_spot``,
  ``width_over_forecast``, ``anchor_over_spot``, ``n_legs``;
* :func:`rel_spread` -- ``engine/score.py`` ``_mean_relative_spread``;
* :func:`chain_depth` -- ``engine/score.py`` ``_chain_depth``, the
  conditioning observable of ``n_admissible`` (the lookup itself is
  ``admissible_table.n_admissible_for`` over the frozen table);
* :func:`schematic_columns` -- ``Scorer._chooser_schematics`` (EXP-165);
* :func:`producer_columns` -- the ``pred_im_t1_d14``/``pred_runup_abs_move_d14``
  predictions and bands;
* :func:`knn_analog_columns` -- ``Scorer._chooser_analogs``, the k-NN block
  over the frozen ``ChooserAnalogPoolArtifact``;
* :func:`forecast_edge` -- ``tier4_forecast_edge``.

Every function returns NaN exactly where legacy does. The tests import the
legacy methods and compare bit for bit.
"""
from __future__ import annotations

from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from engine.v2.models.chooser_analog_pool import (
    CHOOSER_ANALOG_DIMS,
    CHOOSER_ANALOG_K,
    ChooserAnalogPoolArtifact,
)
from engine.v2.scoring.native_gate_features import forecast_interval

__all__ = [
    "ANALOG_COLUMNS",
    "BREAKEVEN_COLUMNS",
    "PRODUCER_COLUMNS",
    "SHAPE_COLUMNS",
    "SIZE_BAND_COLUMNS",
    "analog_arrays",
    "chain_depth",
    "entry_cost_pct",
    "forecast_edge",
    "geometry_columns",
    "knn_analog_columns",
    "producer_columns",
    "rel_spread",
    "schematic_columns",
    "size_band_columns",
]

_NAN = float("nan")
#: engine/score.py ``Scorer._CHOOSER_ANALOG_COLUMNS``.
ANALOG_COLUMNS = ("analog_mean", "analog_win_rate", "analog_p10", "analog_p90",
                  "analog_n")
#: engine/score.py ``_CHOOSER_BREAKEVEN`` / ``_CHOOSER_SHAPE``.
BREAKEVEN_COLUMNS = (
    "breakeven_down_room_forecast", "breakeven_up_room_forecast",
    "max_profit_pct_spot", "max_profit_over_cost", "max_profit_over_secured",
)
SHAPE_COLUMNS = tuple(
    f"shape_pnl_{side}_m{i}" for side in ("down", "up") for i in (1, 2, 3, 4))
SIZE_BAND_COLUMNS = ("pred_abs_move", "pred_abs_move_sd", "pred_abs_move_p10",
                     "pred_abs_move_p90", "pred_abs_move_resid_n",
                     "tier4_pred_abs_move_sd")
#: engine/score.py ``_chooser_frame``'s other Tier-4 producers, in order.
PRODUCER_COLUMNS: dict[str, tuple[str, ...]] = {
    "pred_im_t1_d14": ("pred_im_t1_d14", "pred_im_t1_d14_p10", "pred_im_t1_d14_p90"),
    "pred_runup_abs_move_d14": (
        "pred_runup_abs_move_d14", "pred_runup_abs_move_d14_p10",
        "pred_runup_abs_move_d14_p90", "pred_runup_abs_move_d14_sd"),
}
_ARRAY_CACHE: dict[str, dict[str, tuple[np.ndarray, ...]]] = {}
_ARRAY_CACHE_LIMIT = 4


def _number(value: Any) -> float:
    """legacy ``float(x) if x is not None else nan`` (NaN stays NaN)."""
    if value is None:
        return _NAN
    try:
        return float(value)
    except (TypeError, ValueError):
        return _NAN


def entry_cost_pct(cost: Any, spot: Any) -> float:
    """``entry_cost / spot_entry * 100`` in float64, as legacy's pandas does
    (a zero spot gives +/-inf or NaN, never an exception)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(_number(cost)) / np.float64(_number(spot)) * 100.0)


def size_band_columns(m: float, pool: Mapping[str, Any] | None) -> dict[str, float]:
    """The sizing forecast ``m`` and its served-fold band.

    Legacy sets the four band values only when ``p10`` is finite; a fold with
    no usable pool (``pool`` None) leaves them NaN.
    """
    s = p10 = p90 = resid_n = _NAN
    if isfinite(m):
        try:
            band = forecast_interval(
                [m], () if pool is None else pool["predictions"],
                () if pool is None else pool["residuals"],
                floor=0.0 if pool is None else pool.get("interval_floor", 0.0))
            if np.isfinite(band[0][0]):
                p10, p90, s, resid_n = (float(band[0][0]), float(band[1][0]),
                                        float(band[2][0]), float(band[3][0]))
        except (KeyError, TypeError, ValueError):
            pass
    return {"pred_abs_move": m, "pred_abs_move_sd": s, "pred_abs_move_p10": p10,
            "pred_abs_move_p90": p90, "pred_abs_move_resid_n": resid_n,
            "tier4_pred_abs_move_sd": s}


def producer_columns(output: str, prediction: float | None,
                     pool: Mapping[str, Any] | None) -> dict[str, float]:
    """One other Tier-4 producer: its prediction and band, NaN when the fold
    serves no finite prediction (legacy leaves the columns NaN)."""
    columns = PRODUCER_COLUMNS[output]
    out = {name: _NAN for name in columns}
    if prediction is None or not isfinite(prediction):
        return out
    band = forecast_interval(
        [prediction], () if pool is None else pool["predictions"],
        () if pool is None else pool["residuals"],
        floor=0.0 if pool is None else pool.get("interval_floor", 0.0))
    out[columns[0]] = float(prediction)
    out[columns[1]] = float(band[0][0])
    out[columns[2]] = float(band[1][0])
    if len(columns) > 3:
        out[columns[3]] = float(band[2][0])
    return out


def _strike(leg: Mapping[str, Any]) -> float:
    return float(leg["strike"])


def geometry_columns(legs: Sequence[Mapping[str, Any]], spot_value: Any,
                     m: float) -> dict[str, float]:
    """``_chooser_frame``'s geometry block; the anchor is the first leg's
    strike (legacy ``result.strike = priced.legs[0].strike``)."""
    legs = list(legs or ())
    anchor = _strike(legs[0]) if legs and legs[0].get("strike") else _NAN
    if legs:
        half = max(_strike(leg) for leg in legs) - anchor if np.isfinite(anchor) else _NAN
    else:
        half = _NAN
    spot = float(spot_value) if spot_value is not None else _NAN
    half_pct = 100.0 * half / spot if np.isfinite(half) and spot else _NAN
    return {
        "half_width_pct_spot": half_pct,
        "width_over_forecast": (half_pct / m if np.isfinite(half_pct) and np.isfinite(m)
                                and m > 0 else _NAN),
        "anchor_over_spot": anchor / spot if np.isfinite(anchor) and spot else _NAN,
        "n_legs": float(len(legs)),
    }


def rel_spread(legs: Sequence[Mapping[str, Any]]) -> float:
    """engine/score.py ``_mean_relative_spread`` over the priced legs."""
    values = []
    for leg in legs or ():
        if leg.get("bid") is None or leg.get("ask") is None:
            return _NAN
        mid = 0.5 * (float(leg["bid"]) + float(leg["ask"]))
        if mid > 0:
            values.append((float(leg["ask"]) - float(leg["bid"])) / mid)
    return float(np.mean(values)) if values else _NAN


def _put_quotes(quotes: Mapping[Any, Mapping[str, Any]], expiry: str):
    """``(strike, bid, ask)`` of the puts listed at ``expiry``, in key order."""
    rows = []
    for key, quote in quotes.items():
        parts = key if isinstance(key, tuple) else str(key).split(":")
        if len(parts) != 3 or str(parts[0]).upper() != "P":
            continue
        if str(parts[2])[:10] != expiry:
            continue
        try:
            strike = float(parts[1])
        except (TypeError, ValueError):
            continue
        rows.append((strike, _number(quote.get("bid")), _number(quote.get("ask"))))
    return rows


def chain_depth(quotes: Mapping[Any, Mapping[str, Any]] | None, expiry: Any,
                spot: Any, m: float, s: float) -> float:
    """Two-sided-quoted put strikes at ``expiry`` inside ``spot*(1 +/- (m+3s)/100)``.

    ``Scorer._live_chain_depth`` + ``_chain_depth`` over the quote domain the
    structure was priced on. NaN when there is no chain, no expiry, no
    finite ``m``/``s`` or no spot -- which the table maps to its fallback.
    """
    if (not quotes or expiry is None or not np.isfinite(m) or not np.isfinite(s)
            or not spot):
        return _NAN
    puts = _put_quotes(quotes, str(expiry)[:10])
    if not puts:
        return _NAN
    seen: set[float] = set()
    unique = []
    for strike, bid, ask in sorted(puts, key=lambda row: row[0]):
        if strike not in seen:
            seen.add(strike)
            unique.append((strike, bid, ask))
    strikes = np.asarray([row[0] for row in unique], dtype=float)
    bid = np.nan_to_num(np.asarray([row[1] for row in unique], dtype=float), nan=0.0)
    ask = np.nan_to_num(np.asarray([row[2] for row in unique], dtype=float), nan=0.0)
    quoted = (bid > 0) & (ask > 0) & np.isfinite(strikes)
    reach = (m + 3.0 * s) / 100.0
    lo, hi = float(spot) * (1.0 - reach), float(spot) * (1.0 + reach)
    return float(((strikes >= lo) & (strikes <= hi) & quoted).sum())


def _entry_legs(legs: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, float, float]]:
    return [(str(leg["right"]), str(leg["side"]),
             float(leg["qty"] if "qty" in leg else leg["quantity"]),
             float(leg["strike"])) for leg in legs]


def _exit_cash(entry, S: float) -> float:
    total = 0.0
    for right, side, qty, strike in entry:
        sign = -1.0 if side == "sell" else 1.0
        intr = max(strike - S, 0.0) if right == "P" else max(S - strike, 0.0)
        total += sign * qty * intr
    return total


def _crossings(pts: np.ndarray, vals: np.ndarray) -> list[float]:
    crossings = []
    for i in range(len(vals) - 1):
        if (vals[i] <= 0.0 < vals[i + 1]) or (vals[i + 1] <= 0.0 < vals[i]):
            if np.isfinite(pts[i + 1]) or np.isfinite(pts[i]):
                if vals[i + 1] != vals[i]:
                    t = -vals[i] / (vals[i + 1] - vals[i])
                    crossings.append(pts[i] + t * (pts[i + 1] - pts[i]))
    return crossings


def _rooms(crossings, spot: float, m: float) -> tuple[float, float]:
    down_room = up_room = _NAN
    if np.isfinite(m) and m > 0:
        move = spot * float(m) / 100.0
        below = [c for c in crossings if c < spot]
        above = [c for c in crossings if c > spot]
        if below:
            down_room = (spot - max(below)) / move
        if above:
            up_room = (min(above) - spot) / move
    return down_room, up_room


def _schematics(entry, spot: float, cost: float, m: float, s: float,
                out: dict[str, float]) -> None:
    """Writes into ``out`` in legacy's order, so a step that raises (a zero
    spot) leaves exactly the columns legacy had already written."""
    strikes = sorted({strike for _r, _s, _q, strike in entry})
    knots = np.unique(np.concatenate([np.array([0.0, spot]), np.array(strikes)]))
    pnl = np.array([_exit_cash(entry, k) for k in knots]) - cost
    pnl_inf = -cost
    vals = np.concatenate([pnl, [pnl_inf]])
    pts = np.concatenate([knots, [np.inf]])
    crossings = _crossings(pts, vals)
    max_profit = float(max(vals.max(), pnl_inf))
    down_room, up_room = _rooms(crossings, spot, m)
    sd = float(s) if np.isfinite(s) and s > 0 else 0.1 * (m or 0.0)
    p = float(m) if np.isfinite(m) else 0.0
    grid = [max(p - sd, 0.0), p, p + sd, p + 2 * sd]
    secured = float(sum(qty * strike * 100.0
                        for _right, side, qty, strike in entry if side == "sell"))
    out["breakeven_down_room_forecast"] = down_room
    out["breakeven_up_room_forecast"] = up_room
    out["max_profit_pct_spot"] = 100.0 * max_profit / spot
    out["max_profit_over_cost"] = max_profit / cost if cost and cost > 0.05 else _NAN
    out["max_profit_over_secured"] = 100.0 * max_profit / secured if secured > 0 else _NAN
    for side_name, sgn in (("down", -1.0), ("up", 1.0)):
        for i, move_pct in enumerate(grid, start=1):
            frac = min(move_pct / 100.0, 0.9)
            S = spot * (1.0 + sgn * frac)
            out[f"shape_pnl_{side_name}_m{i}"] = 100.0 * (_exit_cash(entry, S) - cost) / spot


def schematic_columns(legs: Sequence[Mapping[str, Any]], spot_value: Any,
                      cost_value: Any, m: float, s: float) -> dict[str, float]:
    """``Scorer._chooser_schematics``: breakeven rooms against the forecast
    move, max-profit ratios and the P&L at four down/four up grid moves,
    from the priced entry legs. All NaN without legs, spot or cost; a step
    that raises keeps what was written before it (legacy's ``except:
    pass``)."""
    out = {name: _NAN for name in (*BREAKEVEN_COLUMNS, *SHAPE_COLUMNS)}
    spot = float(spot_value) if spot_value is not None else _NAN
    cost = float(cost_value) if cost_value is not None else _NAN
    if not legs or not np.isfinite(spot) or not np.isfinite(cost):
        return out
    entry = _entry_legs(legs)
    try:
        _schematics(entry, spot, cost, m, s, out)
    except Exception:  # noqa: BLE001 -- legacy's own blanket except
        pass
    return out


def forecast_edge(m: float, or_implied: float) -> float:
    return m - or_implied if np.isfinite(m) and np.isfinite(or_implied) else _NAN


def analog_arrays(pool: ChooserAnalogPoolArtifact,
                  strategy: str) -> tuple[np.ndarray, ...] | None:
    """``(X, y, closed)`` for one structure, in the pool's stored order --
    legacy ``Scorer._chooser_analog_pool``'s per-structure arrays. Cached by
    content hash and read-only."""
    cached = _ARRAY_CACHE.get(pool.content_hash)
    if cached is None:
        cached = {}
        for name, rows in pool.strategies:
            arrays = (np.asarray([row[2:] for row in rows], dtype=float),
                      np.asarray([row[1] for row in rows], dtype=float),
                      np.asarray([row[0] for row in rows], dtype="datetime64[D]"))
            for array in arrays:
                array.setflags(write=False)
            cached[name] = arrays
        if len(_ARRAY_CACHE) >= _ARRAY_CACHE_LIMIT:
            _ARRAY_CACHE.pop(next(iter(_ARRAY_CACHE)))
        _ARRAY_CACHE[pool.content_hash] = cached
    return cached.get(str(strategy))


def knn_analog_columns(pool: ChooserAnalogPoolArtifact | None, strategy: str,
                       entry_date: Any, frame: Mapping[str, float]) -> dict[str, float]:
    """``Scorer._chooser_analogs``: the 25 nearest prior candidates of the
    same structure that CLOSED strictly before this entry, rescaled on the
    eligible pool, summarized as dollar P&L. NaN when there is no pool, no
    entry date, a non-finite key or fewer than K eligible rows."""
    blank = {name: _NAN for name in ANALOG_COLUMNS}
    if pool is None or entry_date is None:
        return blank
    rows = analog_arrays(pool, strategy)
    if rows is None:
        return blank
    vals = np.array([frame.get(d, _NAN) for d in CHOOSER_ANALOG_DIMS], dtype=float)
    if not np.isfinite(vals).all():
        return blank
    X, y, closed = rows
    eligible = closed < np.datetime64(str(entry_date)[:10], "D")
    if eligible.sum() < CHOOSER_ANALOG_K:
        return blank
    X, y = X[eligible], y[eligible]
    scale = X.std(0, ddof=1)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    distance = (((X - vals) / scale) ** 2).mean(1)
    take = np.argpartition(distance, CHOOSER_ANALOG_K - 1)[:CHOOSER_ANALOG_K]
    near = y[take]
    return {
        "analog_mean": float(near.mean()),
        "analog_win_rate": float((near > 0).mean()),
        "analog_p10": float(np.quantile(near, 0.10)),
        "analog_p90": float(np.quantile(near, 0.90)),
        "analog_n": float(CHOOSER_ANALOG_K),
    }
