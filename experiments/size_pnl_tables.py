"""EXP-188 real-store Sources adapter, slice 2: table-backed providers for `size_pnl_driver.Sources` (design section 13).

`tables` returns `fetch` (labels and lead production forecasts), `facts` (sizing inputs), `plan_frame` (print windows) and `table_hash`. Labels (`computed_moves`, `iv_crush_targets`) are read event-isolated through `label_rows`, one proven fragment per requested event. Each label carries the later of the two tables' `available_as_of_date` (text compare), or None when either is None. Every non-null `available_as_of_date` is validated as ISO text that round-trips: when both of a label's dates are non-null, a malformed one refuses `source_value_invalid`, even on a skipped label; when either is null the label's availability is None and the other date is not checked here. A null date is allowed only where the label itself is absent or skipped, since `read_labels` refuses a kept label without a valid one. `feature_panel` and `tier4_forecasts` are single logical partitions read for the requested keys' tickers. `earnings_events` is read in full (only admitted ids are kept). A session other than AMC/BMO, or an event with no calendar window, is dropped from `plan_frame` and counted on stderr. A requested event with no production forecast (or a NaN one), or with no label row, is excluded and counted on stderr; `empty_population` refuses when no requested event has a forecast. Refusals are typed OpsErrors with a `reason`: `source_table_missing` (a table or column is absent), `source_value_invalid` (more than one row, or a bad value), `empty_population`, `plan_frame_empty`, `production_hash_mismatch` or `max_object_bytes_invalid`. A DataError other than CONTRACT_MISMATCH, including a planner HOLDOUT_ACCESS_DENIED, propagates unchanged (the `_guarded` behaviour). Nothing is imputed here.
"""
import datetime as dt
import math
import resource
import sys
import time
import types

import numpy as np
import pandas as pd

from engine.v2.contracts import KeyPredicate
from engine.v2.data.errors import DataError
from engine.v2.ops.errors import fail
from engine.v2.research._pricing import AMC, BMO, trading_calendar_from_snapshot
from engine.v2.research._scan import read_table
from engine.v2.research.event_history import plan_event_targets
from experiments import size_pnl_inputs
from experiments.size_pnl_readers import AVAILABLE, LABEL_FETCH, PRODUCTION_COLUMNS

_EVENT_COLUMNS = ["event_id", "ticker", "event_date", "session"]
_PLAN_COLUMNS = ["event_id", "ticker", "event_date", "session", "entry_date", "exit_date"]
_KEY = ["ticker", "event_date"]
_CRUSH_EVENT_COLUMNS = [*_EVENT_COLUMNS, "date_conflict", "event_cluster_id"]
# FLT_MAX-style placeholder: any |value| at or above this is a sentinel, not a measurement.
_FLT_MAX_BOUND = 3.0e38
# One stderr progress line per this many planned events scanned by label_rows.
SCAN_PROGRESS_EVERY = 2000


def _missing(msg):
    return fail("EXPERIMENT_IDENTITY_CONFLICT", msg, details={"reason": "source_table_missing"})


def _invalid(msg):
    return fail("EXPERIMENT_IDENTITY_CONFLICT", msg, details={"reason": "source_value_invalid"})


def _guarded(name, call, *args, **kwargs):
    """Map a CONTRACT_MISMATCH read failure to `source_table_missing`; other errors propagate."""
    try:
        return call(*args, **kwargs)
    except DataError as exc:
        if exc.code != "CONTRACT_MISMATCH":
            raise
        raise _missing(f"{name} or a requested column is absent from the snapshot") from None


def _values_ok(values, *, allow_nan):
    """True when every value is finite (NaN too, if allowed) and below the FLT_MAX sentinel."""
    arr = np.asarray(values, dtype=float)
    ok = np.isfinite(arr) & (np.abs(arr) < _FLT_MAX_BOUND)
    return bool((ok | np.isnan(arr)).all() if allow_nan else ok.all())


def _floats(frame, column):
    try:
        return pd.to_numeric(frame[column]).astype(float)
    except (TypeError, ValueError):
        raise _invalid(f"{column} must be numeric") from None


def _norm(frame, date_col):
    """Copy of `frame` with ticker as str and the date column normalised to midnight Timestamps."""
    out = frame.copy()
    out["ticker"] = out["ticker"].astype(str)
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    return out


def _keys(frame):
    return set(zip(frame["ticker"], frame["event_date"]))


def _session(value):
    return str(value).strip().upper()


def _flag(value):
    """A date_conflict flag as a bool; a null flag counts as not conflicting."""
    return False if pd.isna(value) else bool(value)


def _iso_day(value):
    """A label's availability as ISO date text, or None when null. Dates are cut to their day; any other value passes through for the reader to refuse."""
    if value is None or value is pd.NaT or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    return value


def _iso_text(value):
    """True when value is ISO date text that round-trips (`date.fromisoformat(v).isoformat() == v`)."""
    try:
        return isinstance(value, str) and dt.date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _later(first, second):
    """The later of two ISO dates (text compare); None when either is None. Each non-None date must be ISO text that round-trips."""
    if any(v is not None and not _iso_text(v) for v in (first, second)):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "a label table holds a malformed available_as_of_date", details={"reason": "source_value_invalid"})
    return None if first is None or second is None else max(first, second)


def _log_read(table, keys, rows, started):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[pnl_tables] read {table} keys={keys} rows={rows} elapsed={time.monotonic() - started:.1f}s rss={rss:.0f}MiB",
          file=sys.stderr)


def _cap(value):
    """The read byte cap: a positive int. A missing cap would reach the store as None, so it is refused before any read."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise fail("INVALID_EXPERIMENT_SPEC", "max_object_bytes must be a positive integer", details={"reason": "max_object_bytes_invalid"})
    return value


def label_rows(repository, snapshot, table, columns, want, *, max_object_bytes):
    """Rows of an outcome table for the requested events only, one event-isolated fragment per event.

    `want` holds event_id, ticker and event_date (Timestamp) for the requested events; the plan refuses unless each has one fragment.
    """
    _cap(max_object_bytes)
    started = time.monotonic()
    contract = repository.table_contract(snapshot, table)
    plan = list(plan_event_targets(repository, snapshot, want, contract, columns=tuple(columns)))
    total, rows = len(plan), []
    print(f"[pnl_tables] scan {table} start events={total} rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}MiB",
          file=sys.stderr)
    for done, (_, request, _) in enumerate(plan, start=1):
        for batch in repository.scan(request, table_name=table, max_object_bytes=max_object_bytes):
            rows.extend(batch.to_pylist())
        if done % SCAN_PROGRESS_EVERY == 0:
            rate = done / max(time.monotonic() - started, 1e-9)
            print(f"[pnl_tables] scan {table} {done}/{total} events, {rate:.1f}/s, ETA {(total - done) / rate:.0f}s",
                  file=sys.stderr)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[pnl_tables] read {table} events={len(want)} rows={len(rows)} elapsed={time.monotonic() - started:.1f}s rss={rss:.0f}MiB",
          file=sys.stderr)
    return rows


def tables(repository, snapshot, admitted, *, crush_snapshot, max_object_bytes, expected_hash,
           read=read_table, calendar=trading_calendar_from_snapshot, label_rows=label_rows):
    _cap(max_object_bytes)
    if "tier4_forecasts" not in snapshot.table_versions:
        raise _missing("the forecast table is absent from the snapshot")
    table_hash = snapshot.table_versions["tier4_forecasts"].manifest_hash
    if table_hash != expected_hash:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "the production table does not match the expected hash",
                   details={"reason": "production_hash_mismatch"})
    if "iv_crush_targets" not in crush_snapshot.table_versions:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "the crush snapshot has no iv_crush_targets",
                   details={"reason": "crush_snapshot_invalid"})
    crush_table_hash = crush_snapshot.table_versions["iv_crush_targets"].manifest_hash

    def _call(table, columns, batch_filter, *, key_filter=(), snap=snapshot):
        return _guarded(table, read, repository, snap, table, list(columns), batch_filter=batch_filter,
                        key_filter=key_filter, max_object_bytes=max_object_bytes)

    def _read(table, columns, keys, date_col="event_date"):
        started = time.monotonic()
        columns = list(columns)
        if not keys:
            out = _norm(pd.DataFrame(columns=columns), date_col)
        else:
            def keep(batch):
                norm = _norm(batch, date_col)
                hit = [key in keys for key in zip(norm["ticker"], norm[date_col])]
                return batch.loc[np.asarray(hit, dtype=bool)]
            # Logical single partitions: the requested keys' tickers bound the read; the batch filter keeps only the keys.
            key_filter = (KeyPredicate(column="ticker", operator="in", values=tuple(sorted({t for t, _ in keys}))),)
            out = _norm(_call(table, columns, keep, key_filter=key_filter), date_col)
        _log_read(table, len(keys), len(out), started)
        return out

    admitted_ids = {str(eid) for eid in admitted["events"]["event_id"]}
    started = time.monotonic()
    raw = _call("earnings_events", _EVENT_COLUMNS,
                lambda batch: batch.loc[batch["event_id"].astype(str).isin(admitted_ids).to_numpy()])
    events = _norm(raw, "event_date").assign(event_id=lambda f: f["event_id"].astype(str))
    _log_read("earnings_events", len(admitted_ids), len(events), started)
    dupes = int(events["event_id"].duplicated().sum())
    absent = len(admitted_ids - set(events["event_id"]))
    if dupes or absent:
        print(f"[pnl_tables] earnings_events duplicated={dupes} absent={absent}", file=sys.stderr)
        if dupes:
            raise _invalid("earnings_events repeats an admitted event id")
        raise _missing("admitted events are absent from earnings_events")
    events = events[_EVENT_COLUMNS]

    # Crush usability: the crush snapshot's earnings_events (read in full) must carry the event id with the same ticker, date and session,
    # unflagged for date conflicts (a (ticker, date) under two ids is flagged on both rows), and an iv_crush_targets fragment for its key.
    # The main snapshot is never used for crush labels.
    crush_events = _norm(_call("earnings_events", _CRUSH_EVENT_COLUMNS, lambda batch: batch, snap=crush_snapshot), "event_date")
    crush_events = crush_events.assign(event_id=lambda f: f["event_id"].astype(str), session=lambda f: f["session"].map(_session))
    flagged = crush_events["date_conflict"].map(_flag).astype(bool).to_numpy() | crush_events.duplicated(_KEY, keep=False).to_numpy()
    clean = crush_events.loc[~flagged]
    crush_set = set(zip(clean["event_id"], clean["ticker"], clean["event_date"], clean["session"]))
    have = {(r.partition_key, r.time_min) for r in _guarded("iv_crush_targets", repository.fragment_records, crush_snapshot, "iv_crush_targets")}
    usable_crush = {eid for eid, ticker, day, session in zip(events["event_id"], events["ticker"], events["event_date"], events["session"])
                    if pd.notna(day) and (eid, ticker, day, _session(session)) in crush_set and (ticker, day.date().isoformat()) in have}
    print(f"[pnl_tables] crush_label_missing={len(events) - len(usable_crush)} of {len(events)}", file=sys.stderr)
    # Move usability: the main snapshot's computed_moves fragment must hold the event's key; the planner never sees the others.
    have_moves = {(r.partition_key, r.time_min) for r in _guarded("computed_moves", repository.fragment_records, snapshot, "computed_moves")}
    usable_moves = {eid for eid, ticker, day in zip(events["event_id"], events["ticker"], events["event_date"])
                    if pd.notna(day) and (ticker, day.date().isoformat()) in have_moves}

    def _labels(table, value, want, snap):
        """`value` and the availability (column `available`, ISO text or None) for each requested event from one outcome table:
        at most one row per event. An event with no row (or a skipped one) gets NaN and None availability: excluded, not refused."""
        columns = [*_KEY, value, "skipped", AVAILABLE]
        if want.empty:
            rows = _norm(pd.DataFrame(columns=columns), "event_date")
        else:
            fetched = _guarded(table, label_rows, repository, snap, table, columns, want, max_object_bytes=max_object_bytes)
            rows = _norm(pd.DataFrame(fetched, columns=columns), "event_date")
            if rows.duplicated(_KEY).any() or not _keys(rows) <= _keys(want):
                raise _invalid(f"{table} must hold at most one row per requested event")
        vals = _floats(rows, value)
        if not _values_ok(vals, allow_nan=True):
            raise _invalid(f"{table}.{value} must be finite or NaN and below the FLT_MAX sentinel")
        return rows[_KEY].assign(value=vals.where(~rows["skipped"].astype(bool)).to_numpy(),
                                 available=rows[AVAILABLE].map(_iso_day).to_numpy())

    def fetch(columns, ids):
        columns = tuple(columns)
        want = events[events["event_id"].isin(ids)]
        keys = _keys(want)
        if set(columns) == set(LABEL_FETCH):
            kind = "labels"
            moves = _labels("computed_moves", "realized_move_pct", want[want["event_id"].isin(usable_moves)], snapshot)
            moves = (moves.assign(value=moves["value"].abs())
                     .rename(columns={"value": "abs_move", "available": "available_moves"}))
            # Requested events without a usable crush label get no crush row: the left merge leaves them NaN with no availability.
            crush = (_labels("iv_crush_targets", "crush_pct_iv30", want[want["event_id"].isin(usable_crush)], crush_snapshot)
                     .rename(columns={"value": "crush_pct_iv30", "available": "available_crush"}))
            out = (want[["event_id", *_KEY]].merge(moves, on=_KEY, how="left")
                   .merge(crush, on=_KEY, how="left"))
            both = out["available_moves"].notna() & out["available_crush"].notna()
            out[AVAILABLE] = [_later(a, b) if ok else None
                              for a, b, ok in zip(out["available_moves"], out["available_crush"], both)]
            out = out[["event_id", "abs_move", "crush_pct_iv30", AVAILABLE]]
            print(f"[pnl_tables] excluded move_label_missing={int(out['abs_move'].isna().sum())} "
                  f"crush_label_missing={int(out['crush_pct_iv30'].isna().sum())} of {len(want)}", file=sys.stderr)
        elif set(columns) == set(PRODUCTION_COLUMNS):
            kind = "production"
            fc = _read("tier4_forecasts", (*_KEY, "pred_abs_move", "pred_iv_crush_30"), keys)
            if fc.duplicated(_KEY).any():
                raise _invalid("the forecast table repeats a (ticker, event_date) key")
            out = want[["event_id", *_KEY]].merge(fc, on=_KEY, how="inner")
            out = out.assign(pred_abs_move=_floats(out, "pred_abs_move"), pred_iv_crush_30=_floats(out, "pred_iv_crush_30"))
            if not all(_values_ok(out[col], allow_nan=True) for col in ("pred_abs_move", "pred_iv_crush_30")):
                raise _invalid("forecast values must be finite and below the FLT_MAX sentinel")
            # A requested event with no forecast row, or a NaN/None prediction, is excluded (never imputed); the rest are kept.
            out = out[out[["pred_abs_move", "pred_iv_crush_30"]].notna().all(axis=1)]
            print(f"[pnl_tables] excluded production_forecast_missing={len(want) - len(out)} of {len(want)}", file=sys.stderr)
            if len(want) and out.empty:
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "no requested event has a production forecast",
                           details={"reason": "empty_population"})
            out = out.assign(horizon_sessions=size_pnl_inputs.LEAD)[["event_id", "pred_abs_move", "pred_iv_crush_30", "horizon_sessions"]]
        else:
            raise _invalid("unsupported column set for a source fetch")
        print(f"[pnl_tables] fetch {kind} ids={len(ids)} rows={len(out)}", file=sys.stderr)
        return out

    def facts(ids):
        want = events[events["event_id"].isin(ids)]
        keys = _keys(want)
        # `or_implied` is the pre-print ORATS quote the panel carries as its own implied move.
        panel = _read("feature_panel", ("ticker", "date", "pre_iv30", "mcap_usd", "or_implied"), keys, "date").rename(columns={"date": "event_date"})
        if not keys <= _keys(panel):
            raise _missing("a requested event has no feature_panel row")
        if panel.duplicated(_KEY).any():
            raise _invalid("facts repeat a (ticker, event_date) key")
        out = want[["event_id", *_KEY]].merge(panel, on=_KEY, how="left").set_index("event_id")
        vals = pd.DataFrame({"pre_iv30": pd.to_numeric(out["pre_iv30"], errors="coerce"),
                             "mcap_usd": pd.to_numeric(out["mcap_usd"], errors="coerce"),
                             "implied_move": pd.to_numeric(out["or_implied"], errors="coerce")}).astype(float)
        if not _values_ok(vals, allow_nan=False):
            raise _invalid("facts values must be finite and below the FLT_MAX sentinel")
        if not ((vals["mcap_usd"] >= 1e6) & (vals["pre_iv30"] > 0) & (vals["implied_move"] > 0)).all():
            raise _invalid("mcap_usd must be USD, not thousands or millions; pre_iv30 and implied_move must be positive")
        print(f"[pnl_tables] facts ids={len(ids)}", file=sys.stderr)
        return vals

    cal = _guarded("the trading calendar", calendar, repository, snapshot, max_object_bytes=max_object_bytes)
    kept, dropped = [], 0
    for eid, ticker, day, session in zip(events["event_id"], events["ticker"], events["event_date"], events["session"]):
        sess = str(session).upper()
        if sess not in (AMC, BMO):
            dropped += 1
            continue
        try:
            kept.append((eid, ticker, day, sess, cal.last_pre_print(day, sess), cal.first_post_print(day, sess)))
        except KeyError:
            dropped += 1
    print(f"[pnl_tables] plan_frame events={len(events)} dropped={dropped}", file=sys.stderr)
    if not kept:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "no admitted event has an AMC/BMO session with a calendar print window",
                   details={"reason": "plan_frame_empty"})
    plan_frame = pd.DataFrame(kept, columns=_PLAN_COLUMNS).sort_values("event_id", kind="mergesort").reset_index(drop=True)
    return types.SimpleNamespace(fetch=fetch, facts=facts, plan_frame=plan_frame, table_hash=table_hash,
                                 crush_table_hash=crush_table_hash)
