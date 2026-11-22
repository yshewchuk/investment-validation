"""EXP-188 real-store Sources adapter, slice 2: table-backed providers for `size_pnl_driver.Sources` (design section 13).

`tables` returns `fetch` (labels and lead production forecasts), `facts` (sizing inputs), `plan_frame` (print windows) and `table_hash`. Labels (`computed_moves`, `iv_crush_targets`) are read event-isolated through `label_rows`, one proven fragment per requested event. `feature_panel` and `tier4_forecasts` are single logical partitions read for the requested keys' tickers. `earnings_events` is read in full (only admitted ids are kept). A session other than AMC/BMO, or an event with no calendar window, is dropped from `plan_frame` and counted on stderr. Refusals are typed OpsErrors with a `reason`: `source_table_missing`, `source_value_invalid`, `plan_frame_empty` or `production_hash_mismatch`. Nothing is imputed here.
"""
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
from experiments.size_pnl_readers import LABEL_COLUMNS, PRODUCTION_COLUMNS

_EVENT_COLUMNS = ["event_id", "ticker", "event_date", "session"]
_PLAN_COLUMNS = ["event_id", "ticker", "event_date", "session", "entry_date", "exit_date"]
_KEY = ["ticker", "event_date"]
# FLT_MAX-style placeholder: any |value| at or above this is a sentinel, not a measurement.
_FLT_MAX_BOUND = 3.0e38


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


def _log_read(table, keys, rows, started):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[pnl_tables] read {table} keys={keys} rows={rows} elapsed={time.monotonic() - started:.1f}s rss={rss:.0f}MiB",
          file=sys.stderr)


def label_rows(repository, snapshot, table, columns, want, *, max_object_bytes):
    """Rows of an outcome table for the requested events only, one event-isolated fragment per event.

    `want` holds event_id, ticker and event_date (Timestamp) for the requested events; the plan refuses unless each has one fragment.
    """
    started = time.monotonic()
    contract = repository.table_contract(snapshot, table)
    plan = plan_event_targets(repository, snapshot, want, contract, columns=tuple(columns))
    rows = [row for _, request, _ in plan
            for batch in repository.scan(request, table_name=table, max_object_bytes=max_object_bytes)
            for row in batch.to_pylist()]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[pnl_tables] read {table} events={len(want)} rows={len(rows)} elapsed={time.monotonic() - started:.1f}s rss={rss:.0f}MiB",
          file=sys.stderr)
    return rows


def tables(repository, snapshot, admitted, *, max_object_bytes, expected_hash,
           read=read_table, calendar=trading_calendar_from_snapshot, label_rows=label_rows):
    if "tier4_forecasts" not in snapshot.table_versions:
        raise _missing("the forecast table is absent from the snapshot")
    table_hash = snapshot.table_versions["tier4_forecasts"].manifest_hash
    if table_hash != expected_hash:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "the production table does not match the expected hash",
                   details={"reason": "production_hash_mismatch"})

    def _call(table, columns, batch_filter, *, key_filter=()):
        return _guarded(table, read, repository, snapshot, table, list(columns), batch_filter=batch_filter,
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

    def _labels(table, value, want):
        """`value` for each requested event from one outcome table: exactly one row per event, NaN where skipped."""
        columns = [*_KEY, value, "skipped"]
        if want.empty:
            rows = _norm(pd.DataFrame(columns=columns), "event_date")
        else:
            fetched = _guarded(table, label_rows, repository, snapshot, table, columns, want, max_object_bytes=max_object_bytes)
            rows = _norm(pd.DataFrame(fetched, columns=columns), "event_date")
            if len(rows) != len(want) or rows.duplicated(_KEY).any() or _keys(rows) != _keys(want):
                raise _invalid(f"{table} must hold exactly one row per requested event")
        vals = _floats(rows, value)
        if not _values_ok(vals, allow_nan=True):
            raise _invalid(f"{table}.{value} must be finite or NaN and below the FLT_MAX sentinel")
        return rows[_KEY].assign(value=vals.where(~rows["skipped"].astype(bool)).to_numpy())

    def fetch(columns, ids):
        columns = tuple(columns)
        want = events[events["event_id"].isin(ids)]
        keys = _keys(want)
        if set(columns) == set(LABEL_COLUMNS):
            kind = "labels"
            moves = _labels("computed_moves", "realized_move_pct", want)
            moves = moves.assign(value=moves["value"].abs()).rename(columns={"value": "abs_move"})
            crush = _labels("iv_crush_targets", "crush_pct_iv30", want).rename(columns={"value": "crush_pct_iv30"})
            out = (want[["event_id", *_KEY]].merge(moves, on=_KEY, how="left")
                   .merge(crush, on=_KEY, how="left"))
            out = out[["event_id", "abs_move", "crush_pct_iv30"]]
        elif set(columns) == set(PRODUCTION_COLUMNS):
            kind = "production"
            fc = _read("tier4_forecasts", (*_KEY, "pred_abs_move", "pred_iv_crush_30"), keys)
            if fc.duplicated(_KEY).any():
                raise _invalid("the forecast table repeats a (ticker, event_date) key")
            if not keys <= _keys(fc):
                raise _missing("an admitted event has no forecast")
            out = want[["event_id", *_KEY]].merge(fc, on=_KEY, how="inner").assign(horizon_sessions=size_pnl_inputs.LEAD)
            if not all(_values_ok(_floats(out, col), allow_nan=False) for col in ("pred_abs_move", "pred_iv_crush_30")):
                raise _invalid("forecast values must be finite and below the FLT_MAX sentinel")
            out = out[["event_id", "pred_abs_move", "pred_iv_crush_30", "horizon_sessions"]]
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
    return types.SimpleNamespace(fetch=fetch, facts=facts, plan_frame=plan_frame, table_hash=table_hash)
