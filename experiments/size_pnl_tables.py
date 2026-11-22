"""EXP-188 real-store Sources adapter, slice 2: table-backed providers for `size_pnl_driver.Sources` (design section 13).

`tables` returns `fetch` (labels and lead production forecasts), `facts` (sizing inputs), `plan_frame` (print windows) and `table_hash`. `earnings_events` is read in full (only admitted ids are kept); every other table is read only for the requested keys' tickers and event years. A session other than AMC/BMO, or an event with no calendar window, is dropped from `plan_frame` and counted on stderr. Refusals are typed OpsErrors with a `reason`: `source_table_missing`, `source_value_invalid`, `plan_frame_empty` or `production_hash_mismatch`. Nothing is imputed here.
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


def tables(repository, snapshot, admitted, *, max_object_bytes, expected_hash,
           read=read_table, calendar=trading_calendar_from_snapshot):
    if "tier4_forecasts" not in snapshot.table_versions:
        raise _missing("the forecast table is absent from the snapshot")
    table_hash = snapshot.table_versions["tier4_forecasts"].manifest_hash
    if table_hash != expected_hash:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "the production table does not match the expected hash",
                   details={"reason": "production_hash_mismatch"})

    def _call(table, columns, batch_filter, *, partition_keys=None, key_filter=()):
        try:
            return read(repository, snapshot, table, list(columns), batch_filter=batch_filter,
                        partition_keys=partition_keys, key_filter=key_filter, max_object_bytes=max_object_bytes)
        except DataError as exc:
            if exc.code != "CONTRACT_MISMATCH":
                raise
            raise _missing(f"{table} or a requested column is absent from the snapshot") from None

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
            # tables are partitioned by the year of their event date, so reading the keys' years loses no key
            partition_keys = sorted({str(day.year) for _, day in keys})
            key_filter = (KeyPredicate(column="ticker", operator="in", values=tuple(sorted({t for t, _ in keys}))),)
            out = _norm(_call(table, columns, keep, partition_keys=partition_keys, key_filter=key_filter), date_col)
        _log_read(table, len(keys), len(out), started)
        return out

    admitted_ids = {str(eid) for eid in admitted["events"]["event_id"]}
    started = time.monotonic()
    raw = _call("earnings_events", _EVENT_COLUMNS,
                lambda batch: batch.loc[batch["event_id"].astype(str).isin(admitted_ids).to_numpy()],
                partition_keys=None, key_filter=())
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

    def fetch(columns, ids):
        columns = tuple(columns)
        want = events[events["event_id"].isin(ids)]
        keys = _keys(want)
        if set(columns) == set(LABEL_COLUMNS):
            kind = "labels"
            moves = _read("computed_moves", (*_KEY, "realized_move_pct"), keys)
            moves = moves.assign(abs_move=moves["realized_move_pct"].abs())
            crush = _read("iv_crush_targets", (*_KEY, "crush_pct_iv30", "skipped"), keys)
            crush = crush.assign(crush_pct_iv30=crush["crush_pct_iv30"].where(~crush["skipped"].astype(bool)))
            out = (want[["event_id", *_KEY]].merge(moves[[*_KEY, "abs_move"]], on=_KEY, how="left")
                   .merge(crush[[*_KEY, "crush_pct_iv30"]], on=_KEY, how="left"))
            out = out[["event_id", "abs_move", "crush_pct_iv30"]]
        elif set(columns) == set(PRODUCTION_COLUMNS):
            kind = "production"
            fc = _read("tier4_forecasts", (*_KEY, "pred_abs_move", "pred_iv_crush_30"), keys)
            if fc.duplicated(_KEY).any():
                raise _invalid("the forecast table repeats a (ticker, event_date) key")
            out = want[["event_id", *_KEY]].merge(fc, on=_KEY, how="inner").assign(horizon_sessions=size_pnl_inputs.LEAD)
            out = out[["event_id", "pred_abs_move", "pred_iv_crush_30", "horizon_sessions"]]
        else:
            raise _invalid("unsupported column set for a source fetch")
        print(f"[pnl_tables] fetch {kind} ids={len(ids)} rows={len(out)}", file=sys.stderr)
        return out

    def facts(ids):
        want = events[events["event_id"].isin(ids)]
        keys = _keys(want)
        panel = _read("feature_panel", ("ticker", "date", "pre_iv30", "mcap_usd"), keys, "date").rename(columns={"date": "event_date"})
        moves = _read("computed_moves", (*_KEY, "implied_move_pct"), keys)
        if not (keys <= _keys(panel) and keys <= _keys(moves)):
            raise _missing("a requested event has no feature_panel or computed_moves row")
        if panel.duplicated(_KEY).any() or moves.duplicated(_KEY).any():
            raise _invalid("facts repeat a (ticker, event_date) key")
        out = (want[["event_id", *_KEY]].merge(panel, on=_KEY, how="left")
               .merge(moves, on=_KEY, how="left").set_index("event_id"))
        vals = pd.DataFrame({"pre_iv30": pd.to_numeric(out["pre_iv30"], errors="coerce"),
                             "mcap_usd": pd.to_numeric(out["mcap_usd"], errors="coerce"),
                             "implied_move": pd.to_numeric(out["implied_move_pct"], errors="coerce")}).astype(float)
        if not (np.isfinite(vals) & (vals.abs() < _FLT_MAX_BOUND)).all().all():
            raise _invalid("facts values must be finite and below the FLT_MAX sentinel")
        if not ((vals["mcap_usd"] >= 1e6) & (vals["pre_iv30"] > 0) & (vals["implied_move"] > 0)).all():
            raise _invalid("mcap_usd must be USD, not thousands or millions; pre_iv30 and implied_move must be positive")
        print(f"[pnl_tables] facts ids={len(ids)}", file=sys.stderr)
        return vals

    cal = calendar(repository, snapshot, max_object_bytes=max_object_bytes)
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
