"""Panel-row staging: one call composes four pinned-snapshot reads into a raw row.

Slice 1 of the cutover PR-6 design: ``scan_panel_row`` is the native raw-row
producer's single call for one ``BoardRequest`` key's ``panel_row``/
``panel_anchor`` pair. It never assigns ``tier4_row`` or ``quote_rows`` (the
raw-row producer's own job), and ``key.strategy`` never changes which reads it
makes or which keys the result carries -- every call builds the full superset.

``key.event_date`` is validated and normalized to its naive midnight calendar
day once, here, before any repository read: board enumeration legitimately
preserves an intraday event clock, and an intraday timestamp means the same
calendar event day as its midnight form -- both must resolve to the same
row/bar decisions, so neither may observe the event day's own close. The key is
never mutated; every helper downstream receives the validated event day.

The four always-made pinned-snapshot dependencies:

* ``scan_daily_state_inputs`` (``key.ticker``);
* a bounded ``computed_moves`` read (``key.ticker``, rows strictly before
  ``min(event_day, decision_session)`` -- the decision-session cutoff, not just
  the event-date one -- and whose outcome source was available on or before
  ``decision_session``), feeding ``panel_math.history_features`` with the
  non-skipped rows only (``skipped=true`` rows are excluded before that call,
  never treated as a zero move); it also supplies the stored
  ``mean_prior_or_implied`` (all rows, regardless of skipped/availability);
* ``price_history_query.get_price_series`` of the fixed ``REGIME_TICKER``
  (``"^GSPC"``) as of ``decision_session``, whose ``close_raw`` feeds
  ``regime.add_regime_features`` -- the single source for the nine ``spy_*``
  regime features (the names are historical; a ``daily_market`` spot is never
  a close, so it is never read here). A caller may supply a build-scoped
  cache to share this raw series across rows;
* ``price_history_query.get_price_series`` as of ``decision_session``, feeding
  ``runup_math.add_runup_features``.

``panel_anchor`` is the latest (freshest) of whichever
contributing reads' own source dates resolve, or -- for the computed-move
history -- the outcome-availability dates of the rows actually used; never a
caller-asserted value.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from engine.v2.contracts.data import (
    DataQuery,
    KeyPredicate,
    PriceQuery,
    PriceSeriesRow,
    SnapshotRef,
    TimeInterval,
)
from engine.v2.data import errors, price_history_query, price_history_table, repository
from engine.v2.features import daily_state_inputs, panel_math, regime, runup_math

__all__ = ["CONTEXT_TICKERS", "PanelRowInputs", "scan_panel_row"]

_COMPUTED_MOVES_TABLE = "computed_moves"
#: Fixed market-context tickers the nightly's per-ticker history backfill
#: includes (they have no earnings event, so nothing else would backfill
#: them). ``"SPY"`` here only keeps its ``daily_market`` history populated for
#: the legacy market-wide read; the ``spy_*`` regime features never read it --
#: their source is the ``REGIME_TICKER`` ``price_history`` close.
CONTEXT_TICKERS: tuple[str, ...] = ("SPY",)
_BATCH_LIMIT = 1000
_RESULT_LIMIT = 10000

#: Generous margin over add_runup_features's 300-session minimum.
_RUNUP_LOOKBACK_SESSIONS = 400

_IMPLIED_COLUMNS = ("event_ordinal", "admitted", "implied_move_pct",
                    "mean_prior_or_implied", "n_prior_implied")
_COMPUTED_COLUMNS = ("ticker", "event_date", "realized_move_pct", "skipped",
                     "available_as_of_date", *_IMPLIED_COLUMNS)
_REGIME_COLUMNS = (
    "spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol5", "spy_vol20",
    "spy_vol60", "spy_vol252", "spy_vol20_rel252",
)
_RUNUP_COLUMNS = (
    "signed_streak", "ema12r_abs", "dist_high", "dist_ema", "ret5", "ret10", "ret20",
)


@dataclass(frozen=True, slots=True)
class PanelRowInputs:
    """The raw panel row (one flat mapping) and the latest contributing source date."""

    panel_row: Mapping[str, Any]
    panel_anchor: Any


def _is_nonblank(value: object) -> bool:
    """True only for a string that is not empty or whitespace alone."""
    return isinstance(value, str) and bool(value.strip())


def _session_day(value: object, field: str, *,
                 normalize_intraday: bool = False) -> pd.Timestamp:
    """One explicit naive calendar day, or a typed refusal.

    ``normalize_intraday`` is only ever set for ``key.event_date``, whose
    producer legitimately carries an intraday event clock: a naive intraday
    timestamp is then accepted and folded to its midnight calendar day, since
    both spellings name the same event day. Every other field (and every other
    caller of this validator) keeps the strict explicit-midnight rule and its
    exact message; numbers, invalid/missing values and timezone-aware values
    refuse either way.
    """
    if normalize_intraday:
        message = f"{field} must be a naive calendar date or timestamp"
    else:
        message = f"{field} must be a naive midnight calendar day"
    if isinstance(value, numbers.Number):
        raise errors.fail("CONTRACT_MISMATCH", message, details={"field": field})
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        raise errors.fail("CONTRACT_MISMATCH", message, details={"field": field}) from None
    if pd.isna(day) or day.tzinfo is not None:
        raise errors.fail("CONTRACT_MISMATCH", message, details={"field": field})
    if not normalize_intraday and day != day.normalize():
        raise errors.fail("CONTRACT_MISMATCH", message, details={"field": field})
    return day.normalize() if normalize_intraday else day


def _window(history_start: object, decision_session: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    """``(start, decision)`` as naive midnight days, or a typed refusal."""
    start = _session_day(history_start, "history_start")
    decision = _session_day(decision_session, "decision_session")
    if start > decision:
        raise errors.fail("CONTRACT_MISMATCH",
                          "history_start must be on or before decision_session")
    return start, decision


def _pinned_version(snapshot: SnapshotRef, table_name: str):
    """The pinned dataset version handle for ``table_name`` (present + nonblank)."""
    version = snapshot.table_versions.get(table_name)
    if version is None or not _is_nonblank(version.dataset_version_id):
        raise errors.fail("CONTRACT_MISMATCH",
                          f"{table_name} dataset version is missing or blank",
                          details={"table_name": table_name})
    return version


def _consume(data_repository: repository.Repository, query: DataQuery,
             table_name: str) -> list[dict[str, object]]:
    """Consume the whole bounded scan into copied rows before returning any."""
    rows: list[dict[str, object]] = []
    for batch in data_repository.scan(query, table_name=table_name):
        for row in batch.to_pylist():
            if len(rows) >= query.max_result_rows:
                raise errors.fail("RESULT_LIMIT_EXCEEDED",
                                  "scan exceeded the bounded result cap",
                                  details={"table_name": table_name})
            rows.append(dict(row))
    return rows


def _read_computed_moves(data_repository: repository.Repository, snapshot: SnapshotRef,
                         ticker: str, history_start: pd.Timestamp, decision: pd.Timestamp,
                         event_day: pd.Timestamp) -> list[dict[str, object]]:
    """``ticker``'s computed_moves rows strictly before the scored event day and
    the decision session, whichever of the two is earlier.

    Every row in the window is returned; ``_available_rows`` applies the
    outcome-availability filter separately. ``event_day`` is the caller's
    already-validated, already-midnight-normalized event day.
    On a v5 table, rows whose ``not_computed_reason`` is set (other than ``no_price_window``) are dropped, so a later refusal row cannot clear the stored implied mean.
    """
    contract = data_repository.table_contract(snapshot, _COMPUTED_MOVES_TABLE)
    missing = [c for c in _IMPLIED_COLUMNS
               if c not in {col.name for col in contract.columns}]
    if missing:
        raise errors.fail("CONTRACT_MISMATCH",
                          "computed_moves lacks the stored implied-history columns (v3 table)",
                          details={"table_name": _COMPUTED_MOVES_TABLE,
                                   "missing_columns": missing})
    has_reason = "not_computed_reason" in {col.name for col in contract.columns}
    columns = (*_COMPUTED_COLUMNS, "not_computed_reason") if has_reason else _COMPUTED_COLUMNS
    version = _pinned_version(snapshot, _COMPUTED_MOVES_TABLE)
    start = history_start.date().isoformat()
    end = min(event_day, decision).date().isoformat()
    if start >= end:
        return []
    key_filter = (KeyPredicate(column="ticker", operator="eq", values=(ticker,)),)
    time_interval = TimeInterval(
        column="event_date",
        start_inclusive=start,
        end_exclusive=end,
    )
    bound = data_repository.scan_population_bound(
        snapshot.snapshot_id, table_name=_COMPUTED_MOVES_TABLE,
        table_contract_ref=version.table_contract_ref,
        key_filter=key_filter, time_interval=time_interval)
    max_result_rows = min(_RESULT_LIMIT, bound)
    max_batch_rows = (min(contract.maximum_batch_rows, _BATCH_LIMIT, max_result_rows)
                      if max_result_rows > 0
                      else min(contract.maximum_batch_rows, _BATCH_LIMIT))
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id,
        table_contract_ref=version.table_contract_ref,
        columns=columns,
        key_filter=key_filter,
        time_interval=time_interval,
        order_by=tuple(contract.primary_key),
        max_batch_rows=max_batch_rows,
        max_result_rows=max_result_rows,
    )
    rows = _consume(data_repository, query, _COMPUTED_MOVES_TABLE)
    if has_reason:  # v5: a refusal row (not_orats, no_session, ticker-level) never shapes confirmed history
        rows = [row for row in rows if row["not_computed_reason"] in (None, "no_price_window")]
    return rows


def _available_rows(rows: list[dict[str, object]],
                    decision: pd.Timestamp) -> list[dict[str, object]]:
    """Rows whose outcome source was available on or before the decision session."""
    eligible: list[dict[str, object]] = []
    for row in rows:
        available = row["available_as_of_date"]
        if available is None:
            continue
        if _session_day(available, "available_as_of_date") <= decision:
            eligible.append(row)
    return eligible


def _require_implied_keys(rows: list[dict[str, object]]) -> None:
    """Refuse any row that lacks a stored implied-history column."""
    missing = sorted({c for row in rows for c in _IMPLIED_COLUMNS if c not in row})
    if missing:
        raise errors.fail("CONTRACT_MISMATCH",
                          "computed_moves row lacks the stored implied-history columns",
                          details={"table_name": _COMPUTED_MOVES_TABLE,
                                   "missing_columns": missing})


def _validated_implied_fields(
    last: dict[str, object],
) -> tuple[bool, float | None, int, float | None]:
    """The latest row's admitted/mean/n_prior/implied, each type-checked."""
    admitted = last["admitted"]
    if not isinstance(admitted, bool):
        raise errors.fail("CONTRACT_MISMATCH",
                          "computed_moves admitted must be a bool",
                          details={"table_name": _COMPUTED_MOVES_TABLE, "field": "admitted"})
    n_prior = last["n_prior_implied"]
    if isinstance(n_prior, bool) or not isinstance(n_prior, int) or n_prior < 0:
        raise errors.fail("CONTRACT_MISMATCH",
                          "computed_moves n_prior_implied must be a non-negative int",
                          details={"table_name": _COMPUTED_MOVES_TABLE,
                                   "field": "n_prior_implied"})
    mean = last["mean_prior_or_implied"]
    implied = last["implied_move_pct"]
    for name, value in (("mean_prior_or_implied", mean), ("implied_move_pct", implied)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, numbers.Real) \
                or not math.isfinite(value):
            raise errors.fail("CONTRACT_MISMATCH",
                              f"computed_moves {name} must be None or a finite real number",
                              details={"table_name": _COMPUTED_MOVES_TABLE, "field": name})
    if (mean is None) != (n_prior == 0):
        raise errors.fail("CONTRACT_MISMATCH",
                          "computed_moves mean_prior_or_implied is inconsistent with "
                          "n_prior_implied",
                          details={"table_name": _COMPUTED_MOVES_TABLE,
                                   "field": "mean_prior_or_implied"})
    return admitted, mean, n_prior, implied


def _stored_implied_mean(rows: list[dict[str, object]]) -> float | None:
    """The scored event's mean prior implied move from the latest stored row.

    The scored event has no row of its own; this is one carry step from the
    latest row, reading stored aggregates, not a recomputation of the history.
    """
    if not rows:
        return None
    _require_implied_keys(rows)
    last = max(rows, key=lambda row: pd.Timestamp(row["event_date"]))
    admitted, mean, n_prior, implied = _validated_implied_fields(last)
    if not admitted:
        return None
    if implied is None:
        return mean
    if n_prior == 0:
        return float(implied)
    return float((mean * n_prior + implied) / (n_prior + 1))


def _read_regime_prices(data_repository: repository.Repository, snapshot: SnapshotRef,
                        history_start: pd.Timestamp, decision: pd.Timestamp
                        ) -> tuple[PriceSeriesRow, ...]:
    """The ``REGIME_TICKER`` (^GSPC) ``price_history`` series through the
    decision, in date order, dropping any row before ``history_start``.

    Read exactly like :func:`_runup_from_prices` -- one pinned
    ``get_price_series`` as of ``decision_session`` (whole pinned retrieval
    history, ``lookback_sessions=0`` like the decision-calendar reader). The
    series carries no dividend adjustment, so ``close_raw`` is the regime
    ``close``; ``get_price_series`` refuses ``CONTRACT_MISMATCH`` when the
    pinned snapshot has no rows for the ticker at all -- by decision there is
    no empty-source fallback here (the nightly requires the ticker at capture).
    """
    query = PriceQuery(
        ticker=price_history_table.REGIME_TICKER,
        session_date=decision.date().isoformat(),
        observation_ceiling=decision.date().isoformat(),
        lookback_sessions=0,
    )
    rows = price_history_query.get_price_series(data_repository, query, snapshot)
    start = history_start.date().isoformat()
    return tuple(row for row in rows if row.date >= start)


def _kept_prior_moves(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Non-skipped computed_moves rows, ascending event_date; raises on a
    null move (a repository-integrity violation outside skipped=true)."""
    kept = sorted((row for row in rows if not row["skipped"]),
                  key=lambda row: pd.Timestamp(row["event_date"]))
    for row in kept:
        if row["realized_move_pct"] is None:
            raise errors.fail("CONTRACT_MISMATCH",
                              "non-skipped computed_moves row has a null realized_move_pct",
                              details={"table_name": _COMPUTED_MOVES_TABLE})
    return kept


def _history_from_kept_moves(kept: list[dict[str, object]]) -> dict[str, float | None]:
    """Non-skipped prior moves in ascending ``event_date`` order -> ``history_features``."""
    prior_moves = [row["realized_move_pct"] for row in kept]
    prior_abs = [abs(value) for value in prior_moves]
    return panel_math.history_features(prior_moves, prior_abs)


def _regime_from_prices(regime_rows: tuple[PriceSeriesRow, ...], event_day: pd.Timestamp,
                        decision: pd.Timestamp) -> tuple[dict[str, float], object]:
    """The 9 legacy ``spy_*`` floats and the separate ``regime_asof`` anchor,
    over the ^GSPC ``price_history`` closes.

    The event frame is built from the validated ``event_day`` (no other key
    field is needed): an unnormalized event timestamp would let ``_anchor_index``
    select the event day's own close.
    """
    if regime_rows:
        ordered = sorted(regime_rows, key=lambda row: row.date)
        try:
            market = pd.DataFrame(
                {"date": pd.to_datetime([row.date for row in ordered]),
                 "close": [float(row.close_raw) for row in ordered]},
            )
        except (TypeError, ValueError, OverflowError):
            raise errors.fail("CONTRACT_MISMATCH", "price_history value conversion failed",
                              details={"table_name":
                                       price_history_table.PRICE_HISTORY_TABLE_NAME}) from None
    else:
        market = pd.DataFrame({"date": [], "close": []})
    events = pd.DataFrame({"date": [event_day], "decision": [decision]})
    row = regime.add_regime_features(events, market, as_of_column="decision").iloc[0]
    features = {column: float(row[column]) for column in _REGIME_COLUMNS}
    return features, row["regime_asof"]


def _runup_from_prices(data_repository: repository.Repository, snapshot: SnapshotRef,
                       ticker: str, decision: pd.Timestamp, event_day: pd.Timestamp,
                       history: dict[str, float | None],
                       kept_prior_moves: list[dict[str, object]]
                       ) -> tuple[dict[str, float], object]:
    """The 7 legacy runup floats and the separate ``runup_asof`` anchor, with the
    current event frame row anchored on the validated ``event_day``."""
    query = PriceQuery(
        ticker=ticker,
        session_date=decision.date().isoformat(),
        observation_ceiling=decision.date().isoformat(),
        lookback_sessions=_RUNUP_LOOKBACK_SESSIONS,
    )
    rows = price_history_query.get_price_series(data_repository, query, snapshot)
    prices = pd.DataFrame(
        {"date": pd.to_datetime([row.date for row in rows]),
         "close_adj": [row.close_adj for row in rows]},
    )
    # add_runup_features's own convention is DataFrame-NaN for an absent numeric;
    # history_features uses None, so feed NaN into the frame rather than let a
    # None survive into float(...) (which would raise). The merged panel_row keeps
    # the untouched history None values from _history_from_kept_moves.
    # One row per kept prior event plus the current event last: _signed_streak
    # resets row 0 of every group, so a one-row frame would pin signed_streak to
    # 0 regardless of prior history. add_runup_features sorts by
    # ["ticker", "date"], every kept prior event_date is strictly before
    # event_day (the bounded computed_moves read caps at
    # min(event_day, decision), exclusive), and there is one ticker -- so
    # iloc[-1] is always the current event regardless of input order.
    frame_rows = [
        {"ticker": ticker, "date": pd.Timestamp(row["event_date"]),
         "decision": decision, "move": float(row["realized_move_pct"]),
         "n_prior": float("nan"), "ema12_prior_abs_move": float("nan"),
         "mean_prior_abs_move": float("nan")}
        for row in kept_prior_moves
    ]
    frame_rows.append({
        "ticker": ticker, "date": event_day,
        "decision": decision, "move": float("nan"),
        "n_prior": history["n_prior"],
        "ema12_prior_abs_move": _as_float_or_nan(history["ema12_prior_abs_move"]),
        "mean_prior_abs_move": _as_float_or_nan(history["mean_prior_abs_move"]),
    })
    frame = pd.DataFrame(frame_rows)
    result = runup_math.add_runup_features(frame, {ticker: prices}, "decision")
    row = result.iloc[-1]
    features = {column: float(row[column]) for column in _RUNUP_COLUMNS}
    return features, row["runup_asof"]


def _as_float_or_nan(value: object) -> float:
    """A float, mapping a missing ``None``/NA to ``NaN`` (the frame's convention)."""
    if value is None or (isinstance(value, float) and value != value):
        return float("nan")
    return float(value)


def _anchor(*sources: object) -> pd.Timestamp | None:
    """The latest present contributing source date, or ``None`` when all absent."""
    present: list[pd.Timestamp] = []
    for source in sources:
        if source is None:
            continue
        try:
            stamp = pd.Timestamp(source)
        except (TypeError, ValueError, OverflowError):
            continue
        if pd.isna(stamp):
            continue
        present.append(stamp)
    return max(present) if present else None


def _legacy_named_inputs(daily_values: Mapping[str, float],
                         runup_features: Mapping[str, float]) -> dict[str, float]:
    """Legacy-named model inputs the panel math does not produce.

    ``or_implied``/``or_rvol30`` are the daily-state ``im``/``rvol30`` (absent
    when those are absent; panel.py ``ORATS_FEATURES``). ``abs_dist_*`` are
    absolute values, NaN stays NaN (features.py ``add_absolute_features``).
    ``has_implied_quote`` is 1.0 only for a present value above 0
    (features.py ``add_quote_indicators``).
    """
    implied = daily_values.get("im")
    out: dict[str, float] = {
        "abs_dist_high": abs(runup_features["dist_high"]),
        "abs_dist_ema": abs(runup_features["dist_ema"]),
        "has_implied_quote": 1.0 if implied is not None and implied > 0 else 0.0,
    }
    if implied is not None:
        out["or_implied"] = implied
    rvol = daily_values.get("rvol30")
    if rvol is not None:
        out["or_rvol30"] = rvol
    return out


def scan_panel_row(
    repository: repository.Repository,
    snapshot: SnapshotRef,
    key: Any,
    *,
    decision_session: object,
    history_start: object,
    spy_market_cache: dict[tuple[str, str, str, str], tuple[PriceSeriesRow, ...]] | None = None,
) -> PanelRowInputs:
    """One ``BoardRequest`` key's full-superset raw panel row and latest anchor.

    ``key.event_date`` is validated (and, for this field only, intraday-normalized)
    before the first repository read, so a directly constructed invalid key
    refuses with ``CONTRACT_MISMATCH``/``field=event_date`` without touching the
    data layer; every later use is the validated ``event_day``, never a reread.
    """
    start, decision = _window(history_start, decision_session)

    # Refuse/normalize once, here, before any repository access.
    event_day = _session_day(key.event_date, "event_date", normalize_intraday=True)

    daily_state = daily_state_inputs.scan_daily_state_inputs(
        repository, snapshot, ticker=key.ticker,
        history_start=start, decision_session=decision,
    )

    computed_rows = _read_computed_moves(repository, snapshot, key.ticker, start, decision, event_day)
    kept_moves = _kept_prior_moves(_available_rows(computed_rows, decision))
    history = _history_from_kept_moves(kept_moves)
    history_asof = _anchor(*(row["available_as_of_date"] for row in kept_moves))

    spy_rows: tuple[PriceSeriesRow, ...]
    if spy_market_cache is None:
        spy_rows = _read_regime_prices(repository, snapshot, start, decision)
    else:
        spy_key = (
            snapshot.snapshot_id,
            _pinned_version(snapshot, price_history_table.PRICE_HISTORY_TABLE_NAME).dataset_version_id,
            start.isoformat(),
            decision.isoformat(),
        )
        cached_spy_rows = spy_market_cache.get(spy_key)
        if cached_spy_rows is None:
            spy_rows = _read_regime_prices(repository, snapshot, start, decision)
            spy_market_cache[spy_key] = spy_rows
        else:
            spy_rows = tuple(cached_spy_rows)
    regime_features, regime_asof = _regime_from_prices(spy_rows, event_day, decision)

    runup_features, runup_asof = _runup_from_prices(
        repository, snapshot, key.ticker, decision, event_day, history, kept_moves,
    )

    panel_anchor = _anchor(daily_state.source_session, regime_asof, runup_asof,
                           history_asof)
    panel_row: dict[str, Any] = {
        **daily_state.values,
        **history,
        "mean_prior_or_implied": _stored_implied_mean(computed_rows),
        **regime_features,
        **runup_features,
        **_legacy_named_inputs(daily_state.values, runup_features),
        "date": event_day.date().isoformat(),
    }
    return PanelRowInputs(panel_row=panel_row, panel_anchor=panel_anchor)
