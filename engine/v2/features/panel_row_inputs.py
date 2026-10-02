"""Panel-row staging: one call composes four pinned-snapshot reads into a raw row.

Slice 1 of the cutover PR-6 design: ``scan_panel_row`` is the native raw-row
producer's single call for one ``BoardRequest`` key's ``panel_row``/
``panel_anchor`` pair. It never assigns ``tier4_row`` or ``quote_rows`` (the
raw-row producer's own job), and ``key.strategy`` never changes which reads it
makes or which keys the result carries -- every call builds the full superset.

The four always-made pinned-snapshot dependencies:

* ``scan_daily_state_inputs`` (``key.ticker``);
* a bounded ``computed_moves`` read (``key.ticker``, rows strictly before
  ``key.event_date``), feeding ``panel_math.history_features``;
* a new bounded ``daily_market`` read for the fixed ticker ``"SPY"``, feeding
  ``regime.add_regime_features`` (a different, raw chronological shape than the
  derived/lagged ``scan_daily_state_inputs`` mapping, so not reused from it);
* ``price_history_query.get_price_series`` as of ``decision_session``, feeding
  ``runup_math.add_runup_features``.

``panel_anchor`` is the loosest (earliest, most conservative) of whichever
contributing reads' own source dates resolve -- never a caller-asserted value.
"""
from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from engine.v2.contracts.data import (
    DataQuery,
    KeyPredicate,
    PriceQuery,
    SnapshotRef,
    TimeInterval,
)
from engine.v2.data import errors, price_history_query, repository
from engine.v2.features import daily_state_inputs, panel_math, regime, runup_math

__all__ = ["PanelRowInputs", "scan_panel_row"]

_COMPUTED_MOVES_TABLE = "computed_moves"
_DAILY_MARKET_TABLE = "daily_market"
_SPY = "SPY"
_BATCH_LIMIT = 1000
_RESULT_LIMIT = 10000

#: Generous margin over add_runup_features's 300-session minimum.
_RUNUP_LOOKBACK_SESSIONS = 400

_COMPUTED_COLUMNS = ("ticker", "event_date", "realized_move_pct", "skipped")
_SPY_COLUMNS = ("ticker", "date", "spot")
_REGIME_COLUMNS = (
    "spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol5", "spy_vol20",
    "spy_vol60", "spy_vol252", "spy_vol20_rel252",
)
_RUNUP_COLUMNS = (
    "signed_streak", "ema12r_abs", "dist_high", "dist_ema", "ret5", "ret10", "ret20",
)


@dataclass(frozen=True, slots=True)
class PanelRowInputs:
    """The raw panel row (one flat mapping) and the loosest contributing source date."""

    panel_row: Mapping[str, Any]
    panel_anchor: Any


def _is_nonblank(value: object) -> bool:
    """True only for a string that is not empty or whitespace alone."""
    return isinstance(value, str) and bool(value.strip())


def _session_day(value: object, field: str) -> pd.Timestamp:
    """One explicit naive midnight calendar day, or a typed refusal."""
    message = f"{field} must be a naive midnight calendar day"
    if isinstance(value, numbers.Number):
        raise errors.fail("CONTRACT_MISMATCH", message, details={"field": field})
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        raise errors.fail("CONTRACT_MISMATCH", message, details={"field": field}) from None
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise errors.fail("CONTRACT_MISMATCH", message, details={"field": field})
    return day


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
                         key: Any, history_start: pd.Timestamp, decision: pd.Timestamp
                         ) -> list[dict[str, object]]:
    """``key.ticker``'s computed_moves rows strictly before the scored event and
    the decision session, whichever of the two is earlier."""
    contract = data_repository.table_contract(snapshot, _COMPUTED_MOVES_TABLE)
    version = _pinned_version(snapshot, _COMPUTED_MOVES_TABLE)
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id,
        table_contract_ref=version.table_contract_ref,
        columns=_COMPUTED_COLUMNS,
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=(key.ticker,)),),
        time_interval=TimeInterval(
            column="event_date",
            start_inclusive=history_start.date().isoformat(),
            end_exclusive=min(key.event_date, decision).date().isoformat(),
        ),
        order_by=tuple(contract.primary_key),
        max_batch_rows=min(contract.maximum_batch_rows, _BATCH_LIMIT),
        max_result_rows=min(contract.maximum_result_rows, _RESULT_LIMIT),
    )
    return _consume(data_repository, query, _COMPUTED_MOVES_TABLE)


def _read_spy_market(data_repository: repository.Repository, snapshot: SnapshotRef,
                     history_start: pd.Timestamp, decision: pd.Timestamp
                     ) -> list[dict[str, object]]:
    """``"SPY"``'s raw chronological ``daily_market`` series through the decision."""
    contract = data_repository.table_contract(snapshot, _DAILY_MARKET_TABLE)
    version = _pinned_version(snapshot, _DAILY_MARKET_TABLE)
    end = decision + pd.Timedelta(days=1)
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id,
        table_contract_ref=version.table_contract_ref,
        columns=_SPY_COLUMNS,
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=(_SPY,)),),
        time_interval=TimeInterval(
            column="date",
            start_inclusive=history_start.date().isoformat(),
            end_exclusive=end.date().isoformat(),
        ),
        order_by=tuple(contract.primary_key),
        max_batch_rows=min(contract.maximum_batch_rows, _BATCH_LIMIT),
        max_result_rows=min(contract.maximum_result_rows, _RESULT_LIMIT),
    )
    return _consume(data_repository, query, _DAILY_MARKET_TABLE)


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


def _regime_from_spy(spy_rows: list[dict[str, object]], key: Any,
                     decision: pd.Timestamp) -> tuple[dict[str, float], object]:
    """The 9 legacy ``spy_*`` floats and the separate ``regime_asof`` anchor."""
    if spy_rows:
        ordered = sorted(spy_rows, key=lambda row: pd.Timestamp(row["date"]))
        try:
            market = pd.DataFrame(
                {"date": pd.to_datetime([row["date"] for row in ordered]),
                 "close": [float(row["spot"]) for row in ordered]},
            )
        except (TypeError, ValueError, OverflowError):
            raise errors.fail("CONTRACT_MISMATCH", "daily_market value conversion failed",
                              details={"table_name": _DAILY_MARKET_TABLE}) from None
    else:
        market = pd.DataFrame({"date": [], "close": []})
    events = pd.DataFrame({"date": [key.event_date], "decision": [decision]})
    row = regime.add_regime_features(events, market, as_of_column="decision").iloc[0]
    features = {column: float(row[column]) for column in _REGIME_COLUMNS}
    return features, row["regime_asof"]


def _runup_from_prices(data_repository: repository.Repository, snapshot: SnapshotRef,
                       key: Any, decision: pd.Timestamp,
                       history: dict[str, float | None],
                       kept_prior_moves: list[dict[str, object]]
                       ) -> tuple[dict[str, float], object]:
    """The 7 legacy runup floats and the separate ``runup_asof`` anchor."""
    query = PriceQuery(
        ticker=key.ticker,
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
    # key.event_date (the bounded computed_moves read caps at
    # min(event_date, decision), exclusive), and there is one ticker -- so
    # iloc[-1] is always the current event regardless of input order.
    frame_rows = [
        {"ticker": key.ticker, "date": pd.Timestamp(row["event_date"]),
         "decision": decision, "move": float(row["realized_move_pct"]),
         "n_prior": float("nan"), "ema12_prior_abs_move": float("nan"),
         "mean_prior_abs_move": float("nan")}
        for row in kept_prior_moves
    ]
    frame_rows.append({
        "ticker": key.ticker, "date": pd.Timestamp(key.event_date),
        "decision": decision, "move": float("nan"),
        "n_prior": history["n_prior"],
        "ema12_prior_abs_move": _as_float_or_nan(history["ema12_prior_abs_move"]),
        "mean_prior_abs_move": _as_float_or_nan(history["mean_prior_abs_move"]),
    })
    frame = pd.DataFrame(frame_rows)
    result = runup_math.add_runup_features(frame, {key.ticker: prices}, "decision")
    row = result.iloc[-1]
    features = {column: float(row[column]) for column in _RUNUP_COLUMNS}
    return features, row["runup_asof"]


def _as_float_or_nan(value: object) -> float:
    """A float, mapping a missing ``None``/NA to ``NaN`` (the frame's convention)."""
    if value is None or (isinstance(value, float) and value != value):
        return float("nan")
    return float(value)


def _anchor(*sources: object) -> pd.Timestamp | None:
    """The earliest present contributing source date, or ``None`` when all absent."""
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
    return min(present) if present else None


def scan_panel_row(
    repository: repository.Repository,
    snapshot: SnapshotRef,
    key: Any,
    *,
    decision_session: object,
    history_start: object,
) -> PanelRowInputs:
    """One ``BoardRequest`` key's full-superset raw panel row and loosest anchor."""
    start, decision = _window(history_start, decision_session)

    daily_state = daily_state_inputs.scan_daily_state_inputs(
        repository, snapshot, ticker=key.ticker,
        history_start=start, decision_session=decision,
    )

    computed_rows = _read_computed_moves(repository, snapshot, key, start, decision)
    kept_moves = _kept_prior_moves(computed_rows)
    history = _history_from_kept_moves(kept_moves)

    spy_rows = _read_spy_market(repository, snapshot, start, decision)
    regime_features, regime_asof = _regime_from_spy(spy_rows, key, decision)

    runup_features, runup_asof = _runup_from_prices(
        repository, snapshot, key, decision, history, kept_moves,
    )

    panel_anchor = _anchor(daily_state.source_session, regime_asof, runup_asof)
    panel_row: dict[str, Any] = {
        **daily_state.values,
        **history,
        **regime_features,
        **runup_features,
    }
    return PanelRowInputs(panel_row=panel_row, panel_anchor=panel_anchor)
