"""Snapshot-native monthly D14 calculations, retaining the pinned panel history.

This is deliberately not the live ``scan_panel_row`` population: legacy's
entry frame left-joins the stored event-history block, including its warm-up
and source merge. Replacing that producer is a separate migration.
"""
from __future__ import annotations

from itertools import groupby

import numpy as np
import pandas as pd

from engine.v2.contracts.data import DataQuery, KeyPredicate, TimeInterval
from engine.v2.data import errors
from engine.v2.features import panel_math

HISTORY_COLUMNS = (
    "n_prior", "mean_prior_move", "mean_prior_abs_move", "mean_prior_or_implied",
    *[f"ema{s}_prior_{kind}" for kind in ("move", "abs_move") for s in (2, 4, 8, 12)],
    "ema12r_abs", "signed_streak",
)
DAILY_COLUMNS = (
    *panel_math.DAILY_STATE_FIELDS.values(),
    *[f"{panel_math.DAILY_STATE_FIELDS[f]}_d{lag}"
      for f in panel_math.LAGGED_FIELDS for lag in panel_math.DAILY_STATE_LAGS],
)
_TABLES = ("earnings_events", "feature_panel", "daily_market")
# Fixed population shared by the current monthly recipe contracts. Advancing
# this window is a recipe/population change, not snapshot discovery.
_EVENT_START, _WINDOW_END = "2017-01-01", "2027-01-01"


def input_rows(repository, snapshot, table, columns, *, start, end, cap=500000):
    """One explicitly bounded read; Repository owns contract and object checks."""
    contract = repository.table_contract(snapshot, table)
    version = snapshot.table_versions[table]
    keys = (() if table == "feature_panel" else (KeyPredicate(
        column="year", operator="in", values=tuple(range(int(start[:4]), int(end[:4])))),))
    interval = (TimeInterval(column=contract.observation_time_column,
                             start_inclusive=start, end_exclusive=end)
                if table == "feature_panel" else None)
    population = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=table, table_contract_ref=version.table_contract_ref,
        key_filter=keys, time_interval=interval)
    limit = min(population, cap)
    query = DataQuery(snapshot_id=snapshot.snapshot_id,
                      table_contract_ref=version.table_contract_ref, columns=tuple(columns),
                      key_filter=keys, time_interval=interval, order_by=contract.primary_key,
                      max_batch_rows=min(contract.maximum_batch_rows, 1000, max(1, limit)),
                      max_result_rows=limit)
    count = 0
    for batch in repository.scan(query, table_name=table):
        for row in batch.to_pylist():
            count += 1
            if count > limit:
                raise errors.fail("RESULT_LIMIT_EXCEEDED", "D14 input exceeds its row bound")
            yield row


def input_frame(repository, snapshot, table, columns, **kwargs):
    """Materialize input_rows as a frame with the requested column order."""
    return pd.DataFrame(input_rows(repository, snapshot, table, columns, **kwargs), columns=columns)


def decision_rows(events, calendar):
    """Same supplied-session index arithmetic as legacy's pre-print/shift pair."""
    try:
        days = pd.DatetimeIndex(getattr(calendar, "days", ()))
    except (TypeError, ValueError) as exc:
        raise errors.fail("CONTRACT_MISMATCH", "D14 calendar has invalid days") from exc
    if days.empty or days.hasnans or days.tz is not None or not days.is_unique \
            or not days.is_monotonic_increasing or not days.equals(days.normalize()):
        raise errors.fail("CONTRACT_MISMATCH", "D14 calendar must be ordered unique naive days")
    rows = []
    for event in events.itertuples(index=False):
        if pd.isna(event.session):
            continue
        if event.session not in ("AMC", "BMO"):
            raise errors.fail("CONTRACT_MISMATCH", "D14 event has an unknown session")
        day = pd.Timestamp(event.event_date).normalize()
        side = "right" if event.session == "AMC" else "left"
        pre = int(days.searchsorted(day, side=side)) - 1
        if pre < 14:
            continue
        rows.append(dict(ticker=event.ticker, event_date=day, session=event.session,
                         last_pre_print=days[pre], entry_date=days[pre - 14],
                         days_before_print=14.0, year=day.year))
    return pd.DataFrame(rows, columns=("ticker", "event_date", "session", "last_pre_print",
                                       "entry_date", "days_before_print", "year"))


def daily_features(frame, repository, snapshot):
    """Attach entry/exit state in place from one bounded native daily scan."""
    for column in (*DAILY_COLUMNS, "im_t1", "exit_spot"):
        frame[column] = np.nan
    groups = {ticker: group for ticker, group in frame.groupby("ticker", sort=True)}
    stream = input_rows(repository, snapshot, "daily_market",
                   ("ticker", "date", "src_iv", *panel_math.DAILY_STATE_FIELDS),
                   start="2016-01-01", end=_WINDOW_END, cap=20000000)
    for ticker, stream_rows in groupby(stream, key=lambda row: row["ticker"]):
        group = groups.get(ticker)
        if group is None:
            continue
        rows = []
        for row in stream_rows:
            if len(rows) >= 10000:
                raise errors.fail("RESULT_LIMIT_EXCEEDED", "D14 ticker exceeds its daily row bound")
            rows.append({**row, "date": pd.Timestamp(row["date"])})
        for index, event in group.iterrows():
            values = panel_math.daily_state_lookup(rows, event.entry_date)
            for column, value in values.items():
                frame.at[index, column] = value
            exit_state = panel_math.daily_state_lookup(rows, event.last_pre_print)
            frame.at[index, "im_t1"] = exit_state.get("im", np.nan)
            frame.at[index, "exit_spot"] = exit_state.get("spot", np.nan)
    return frame


def validate_role(role):
    """Refuse an unsupported monthly role before any input read."""
    if role not in ("implied_t1", "runup_move"):
        raise errors.fail("CONTRACT_MISMATCH", "D14 datasets support only implied_t1 and runup_move")


def calculate_frame(repository, snapshot, *, calendar, role):
    """Calculate a frame from caller-admitted inputs, without a source receipt.

    This helper checks the supplied snapshot against its repository, for direct
    callers too. It does not certify calendar provenance or attach training_inputs;
    use build_d14_dataset for the receipt-verified training boundary.
    Snapshot provenance and session cutoffs do not certify original intraday
    availability or independently validate the stored historical aggregates.
    """
    validate_role(role)
    if repository.resolve(snapshot.snapshot_id) != snapshot:
        raise errors.fail("IDENTITY_CONFLICT", "D14 snapshot differs from its catalog identity")
    if any(table not in snapshot.table_versions for table in _TABLES):
        raise errors.fail("CONTRACT_MISMATCH", "D14 snapshot lacks a required input table")
    events = input_frame(repository, snapshot, "earnings_events",
                   ("event_id", "ticker", "event_date", "session", "year"),
                   start=_EVENT_START, end=_WINDOW_END)
    # Legacy reads year partitions in order, with event_id order inside each.
    events = events.sort_values(["year", "event_id"], kind="stable")
    frame = decision_rows(events, calendar)
    if frame.empty:
        raise errors.fail("CONTRACT_MISMATCH", "D14 inputs contain no eligible decision rows")
    history = input_frame(repository, snapshot, "feature_panel", ("ticker", "date", *HISTORY_COLUMNS),
                    start=_EVENT_START, end=_WINDOW_END)
    # Arrow's all-null/empty projection otherwise becomes object dtype, changing
    # the legacy dataset fingerprint even though the missing mask is identical.
    for column in HISTORY_COLUMNS:
        dtype = "int64" if column in ("n_prior", "signed_streak") else "float64"
        if dtype == "int64" and history[column].isna().any():
            raise errors.fail("CONTRACT_MISMATCH", "D14 integer history contains nulls")
        history[column] = pd.to_numeric(history[column]).astype(dtype)
    history["date"] = pd.to_datetime(history["date"])
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    frame = frame.merge(history.rename(columns={"date": "event_date"}),
                        on=["ticker", "event_date"], how="left", validate="many_to_one")
    frame = daily_features(frame, repository, snapshot)
    span = frame["event_date"] - pd.to_datetime(frame["entry_date"])
    frame["days_to_print"] = span.dt.days.astype(float)
    for column in ("iv30", "iv10", "exern_iv30", "exern_iv10"):
        frame[f"pre_{column}"] = frame[column]
    frame["runup_pp"] = frame["im_t1"] - frame["im"]
    if role == "runup_move":
        entry, exit_ = frame["spot"].to_numpy(float), frame["exit_spot"].to_numpy(float)
        valid = np.isfinite(entry) & np.isfinite(exit_) & (entry > 0) & (exit_ > 0)
        signed = np.full(len(frame), np.nan)
        signed[valid] = 100.0 * np.log(exit_[valid] / entry[valid])
        frame["runup_signed_move_d14"] = signed
        frame["runup_abs_move_d14"] = np.abs(signed)
    return frame.drop(columns="exit_spot")
