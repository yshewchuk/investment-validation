"""EXP-188 S2b: admitted-key production and label readers.

Labels and production rows are read ONLY through an injected ``fetch(columns, event_ids)``
and ONLY for event ids the shared population admission returned (holdout and
conflicting-sibling events are dropped before any read, counted by reason). No raw table
reader is reachable from here. The window arguments let the synthetic-calibration addendum
read inner years only. No production caller yet (S3/S4/S9). Progress: one stderr line per
call with counts (never per item).
"""
import sys

import numpy as np
import pandas as pd

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from experiments.size_pnl_inputs import LEAD

_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
_DENIED = "HOLDOUT_ACCESS_DENIED"
LABEL_COLUMNS = ("abs_move", "crush_pct_iv30")
PRODUCTION_COLUMNS = ("pred_abs_move", "pred_iv_crush_30", "horizon_sessions")


class Admitted(dict):
    """Admission result; built by admit_events only. Readers refuse any other mapping."""


def _conflict(reason, message):
    return fail(_CONFLICT, message, details={"reason": reason})


def _denied(reason, message):
    return fail(_DENIED, message, details={"reason": reason})


def _dates(admitted):
    if not isinstance(admitted, Admitted):
        raise _denied("admission_not_issued", "readers accept only an admission issued by admit_events")
    return admitted["events"].set_index("event_id")["event_date"]


def _in_window(dates, start, end):
    keep = pd.Series(True, index=dates.index, dtype=bool)
    if start is not None:
        keep &= dates >= pd.Timestamp(start)
    if end is not None:
        keep &= dates < pd.Timestamp(end)
    return keep


def admit_events(repository, snapshot, as_of_month, *, max_object_bytes, load=None):
    """Full-universe admission through the shared reader; dropped events never reach a read."""
    if isinstance(max_object_bytes, bool) or not isinstance(max_object_bytes, int) or max_object_bytes <= 0:
        raise fail("INVALID_EXPERIMENT_SPEC", "max_object_bytes must be a positive integer",
                   details={"reason": "max_object_bytes_invalid"})
    if load is None:
        from engine.v2.research.experiment_population import load_population
        load = load_population
    frame = load(repository, snapshot, as_of_month=as_of_month, purpose="selection",
                 event_ids=None, max_object_bytes=max_object_bytes)
    if not {"event_id", "event_date"} <= set(frame.columns):
        raise _conflict("admission_malformed", "the admitted population lacks event_id or event_date")
    events = pd.DataFrame({"event_id": frame["event_id"].astype(str),
                           "event_date": pd.to_datetime(frame["event_date"])})
    if events["event_id"].duplicated().any():
        raise _conflict("duplicate_event_id", "the admitted population repeats an event id")
    events = events.sort_values("event_id", kind="mergesort").reset_index(drop=True)
    counts = {}
    for row in frame.attrs.get("holdout_exclusions", []):
        for label in row["memberships"]:
            counts[label] = counts.get(label, 0) + 1
    excluded = dict(sorted(counts.items()))
    print(f"[pnl_readers] admitted={len(events)} excluded={excluded}", file=sys.stderr)
    return Admitted({"events": events, "excluded": excluded})


def _fetch(fetch, admitted, columns, ids):
    """The single isolation gate: every label and production read goes through here."""
    ids = sorted(ids)
    if not set(ids) <= set(_dates(admitted).index):
        raise _denied("key_not_admitted", "requested event ids are not admitted")
    if not ids:
        return pd.DataFrame(columns=list(columns), index=pd.Index([], name="event_id"))
    got = fetch(tuple(columns), ids)
    if not isinstance(got, pd.DataFrame) or not {"event_id", *columns} <= set(got.columns):
        raise _conflict("fetch_malformed", "the fetch result lacks event_id or a requested column")
    if not set(got["event_id"]) <= set(ids):
        raise _denied("unrequested_key_returned", "the fetch returned an event id that was not requested")
    if got["event_id"].duplicated().any():
        raise _conflict("duplicate_event_id", "the fetch returned a duplicate event id")
    return got.loc[:, ["event_id", *columns]].set_index("event_id")


def _checked_production(frame, admitted):
    """Shared production-frame admission for read_production and model_tables.

    event_date is not trusted: it is replaced by the admitted date. Returns a new frame with
    columns event_date, pred_move, pred_crush (float), indexed like the input.
    """
    dates = _dates(admitted)
    if not {"pred_move", "pred_crush"} <= set(frame.columns):
        raise _conflict("production_series_failed", "the production frame lacks pred_move or pred_crush")
    if not len(frame):
        raise _conflict("production_series_failed", "production predictions are empty")
    try:
        values = frame[["pred_move", "pred_crush"]].to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise _conflict("production_series_failed", "a production prediction is not numeric") from None
    if not np.isfinite(values).all():
        raise _conflict("production_series_failed", "production predictions are not finite")
    if frame.index.duplicated().any():
        raise _conflict("duplicate_event_id", "the production frame repeats an event id")
    if not set(frame.index) <= set(dates.index):
        raise _conflict("event_ids_outside_population", "a production event id is outside the admitted population")
    return pd.DataFrame({"event_date": dates.reindex(frame.index),
                         "pred_move": values[:, 0], "pred_crush": values[:, 1]}, index=frame.index)


def _window_ids(admitted, start, end, event_ids=None):
    dates = _dates(admitted)
    ids = set(dates.index[_in_window(dates, start, end).to_numpy()])
    if event_ids is not None:
        requested = set(event_ids)
        if not requested <= set(dates.index):
            raise _denied("key_not_admitted", "requested event ids are not admitted")
        ids &= requested
    return sorted(ids)


def read_labels(fetch, admitted, *, start=None, end=None, event_ids=None):
    """Labels for admitted events in the window; NaN labels are dropped, never imputed.

    Month-by-month calls (start = month start, end = next month start) are how a caller keeps
    R6; this interface is what the synthetic arm reuses for inner years.
    """
    ids = _window_ids(admitted, start, end, event_ids)
    frame = _fetch(fetch, admitted, LABEL_COLUMNS, ids)
    raw = frame[list(LABEL_COLUMNS)]
    coerced = pd.DataFrame({col: pd.to_numeric(raw[col], errors="coerce") for col in LABEL_COLUMNS},
                           index=raw.index).astype(float)
    bad = (raw.notna() & (coerced.isna() | np.isinf(coerced))).any(axis=1)
    if bad.any():
        raise _conflict("label_not_finite", "a label is non-numeric or infinite")
    out = coerced.copy()
    out["event_date"] = _dates(admitted).reindex(out.index)
    keep = out[list(LABEL_COLUMNS)].notna().all(axis=1)
    print(f"[pnl_readers] labels requested={len(ids)} label_missing={int((~keep).sum())}", file=sys.stderr)
    return out[keep]


def _is_hash(value):
    return isinstance(value, str) and value.strip() != ""


def _production_binding(table_hash, frame):
    """Binding of a production table: its hash and the sorted rows of the validated frame."""
    srt = frame.sort_index()
    rows = [[eid, day.isoformat(), move, crush] for eid, day, move, crush in
            zip(srt.index.tolist(), srt["event_date"].tolist(), srt["pred_move"].tolist(),
                srt["pred_crush"].tolist())]
    return content_hash({"table": table_hash, "rows": rows})


def read_production(fetch, admitted, *, expected_hash, table_hash, start=None, end=None):
    """Lead-horizon production forecasts for admitted events in the window (R1 whole-run refusal)."""
    if not (_is_hash(expected_hash) and _is_hash(table_hash)):
        raise _conflict("production_hash_invalid", "the production hashes must be non-blank strings")
    if table_hash != expected_hash:
        raise _conflict("production_hash_mismatch", "the production table does not match the expected hash")
    ids = _window_ids(admitted, start, end)
    frame = _fetch(fetch, admitted, PRODUCTION_COLUMNS, ids)
    if (frame["horizon_sessions"] != LEAD).any():
        raise _conflict("lead_unavailable", "a production row is not at the lead horizon")
    built = pd.DataFrame({"pred_move": frame["pred_abs_move"].to_numpy(),
                          "pred_crush": frame["pred_iv_crush_30"].to_numpy()}, index=frame.index)
    out = _checked_production(built, admitted)
    print(f"[pnl_readers] production rows={len(out)} production_missing={len(ids) - len(out)}",
          file=sys.stderr)
    binding = _production_binding(table_hash, out)
    return {"frame": out, "binding": binding, "table_hash": table_hash}


def _candidate_series(bound):
    ids, preds = [], []
    for year in sorted(bound["years"]):
        block = bound["years"][year]
        ids.extend(str(eid) for eid in block["event_ids"])
        preds.extend(np.asarray(block["pred"], dtype=float).tolist())
    if len(set(ids)) != len(ids):
        raise _conflict("duplicate_event_id", "an event id repeats across inner years")
    values = np.asarray(preds, dtype=float)
    if not np.isfinite(values).all():
        raise _conflict("nonfinite_prediction", "a candidate prediction is not finite")
    return pd.Series(values, index=pd.Index(ids, name="event_id"), dtype=float)


def _blend_series(bound):
    parts = {cid: _candidate_series(comp) for cid, comp in sorted(bound["components"].items())}
    cids = list(parts)
    base = parts[cids[0]].sort_index()
    for cid in cids[1:]:
        if set(parts[cid].index) != set(base.index):
            raise _conflict("component_ids_differ", "blend components cover different event ids")
    stacked = [parts[cid].reindex(base.index).to_numpy() for cid in cids]
    return pd.Series(np.mean(stacked, axis=0), index=base.index)


def _consumed_binding(table, source, crush_binding):
    """Hash of the final consumed table: source binding, crush binding, and the sorted rows."""
    srt = table.sort_index()
    rows = [[eid, day.isoformat(), float(move), float(cr)] for eid, day, move, cr in
            zip(srt.index.tolist(), srt["event_date"].tolist(), srt["pred_move"].tolist(),
                srt["pred_crush"].tolist())]
    return content_hash({"source": source, "crush": crush_binding, "rows": rows})


def model_tables(bound, roster, admitted, production, *, start=None, end=None):
    """Per-model (event_date, pred_move, pred_crush) tables; crush is the production forecast, fixed for every model."""
    dates = _dates(admitted)
    prod = _checked_production(production["frame"], admitted)
    table_hash = production.get("table_hash")
    if not _is_hash(table_hash) or _production_binding(table_hash, prod) != production["binding"]:
        raise _conflict("production_binding_mismatch", "the production binding does not match its frame and table hash")
    crush = prod["pred_crush"]
    out = {}
    for model in roster["models"]:
        mid = model["model_id"]
        if model["kind"] == "production":
            frame = prod
            if start is not None or end is not None:
                frame = frame[_in_window(frame["event_date"], start, end).to_numpy()]
            out[mid] = {"status": "ready", "table": frame,
                        "binding": _consumed_binding(frame, production["binding"], production["binding"]),
                        "dropped": {"no_crush_forecast": 0}}
            continue
        entry = bound[mid]
        if entry["status"] != "bound":
            out[mid] = {"status": "excluded", "reason": entry["reason"]}
            continue
        series = _blend_series(entry) if entry.get("kind") == "blend" else _candidate_series(entry)
        if not set(series.index) <= set(dates.index):
            raise _conflict("event_ids_outside_population", "a model event id is outside the admitted population")
        table = pd.DataFrame({"event_date": dates.reindex(series.index), "pred_move": series.to_numpy()},
                             index=series.index)
        table = table[_in_window(table["event_date"], start, end).to_numpy()]
        table = table.assign(pred_crush=crush.reindex(table.index))
        has = table["pred_crush"].notna()
        table = table[has.to_numpy()]
        out[mid] = {"status": "ready", "table": table,
                    "binding": _consumed_binding(table, entry["binding"], production["binding"]),
                    "dropped": {"no_crush_forecast": int((~has).sum())}}
    ready = [entry for entry in out.values() if entry["status"] == "ready"]
    print(f"[pnl_readers] models ready={len(ready)} excluded={len(out) - len(ready)} "
          f"dropped_no_crush={sum(e['dropped']['no_crush_forecast'] for e in ready)}", file=sys.stderr)
    return out
