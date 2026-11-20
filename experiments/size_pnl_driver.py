"""EXP-188 S9: driver of the SIZE P&L re-score, inner (S9a, 2023-2024) and outer (S9b, 2025, #2660) stages; the synthetic arm gets a gated helper in its own slice."""
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from experiments import label_availability, size_pnl_decisions as decisions, size_pnl_inputs as inputs, size_pnl_ladder as ladder, size_pnl_prereg
from experiments import size_pnl_readers as readers, size_pnl_shortvol as shortvol, size_pnl_straddle as straddle, size_pnl_tuning as tuning
from experiments.size_pnl_pool import build_pool
from experiments.size_pnl_shortvol import Refusal

INNER = ("2023-01-01", "2025-01-01")
OUTER = ("2025-01-01", "2026-01-01")
WARMUP_START = "2024-07-01"  # six months of inner history so the trailing cutoff is defined in 2025-01
OUTER_YEAR = 2025  # the roster is resolved for this year
STAGES = ("inner", "outer")
ALPHA = tuning.ALPHA
_DENIED, _CONFLICT = "HOLDOUT_ACCESS_DENIED", "EXPERIMENT_IDENTITY_CONFLICT"
_FACTS = ("pre_iv30", "mcap_usd", "implied_move")
_MARKS = ["exit_value", "rung", "lots", "pnl_dollars"]
# The inputs receipt binds only what is the same in both stages. Production and tables are excluded because their windows
# differ by stage (the production binding hashes the windowed frame), so each stage's receipts bind them instead.
_BOUND = ("spec_hash", "identity", "roster", "models", "snapshot", "labels")  # the pre-2025 labels read is identical in both


@dataclass(frozen=True)
class Sources:
    """Every store-facing input of the run; ``snapshot_id`` is the pinned v2 snapshot, receipted here."""
    plan: dict
    read: Callable
    selection: dict
    pair_candidates: dict
    blend_ids: dict
    expected: dict
    panel_id: str
    admit: Callable
    admitted: readers.Admitted
    fetch: Callable
    production_hash: str
    production_table_hash: str
    plan_frame: pd.DataFrame
    load_index: Callable
    facts: Callable
    snapshot_id: str


def check_sources(src):
    if not (isinstance(src, Sources) and isinstance(src.expected, dict) and {"plan_id", "registration_id", "seal_id", "panel_id"} <= set(src.expected) and isinstance(src.plan_frame, pd.DataFrame)
            and isinstance(src.plan, dict) and {"plan_id", "horizons"} <= set(src.plan) and isinstance(src.plan.get("anchor"), dict) and "run_id" in src.plan["anchor"]
            and all(isinstance(getattr(src, f), dict) for f in ("selection", "pair_candidates", "blend_ids")) and isinstance(src.snapshot_id, str) and src.snapshot_id.strip() and all(callable(getattr(src, f)) for f in ("read", "admit", "fetch", "load_index", "facts"))):
        raise fail("INVALID_EXPERIMENT_SPEC", "sources must be a Sources with an object `expected`", details={"reason": "sources_invalid"})
    return src


def _log(message):
    print(f"[pnl_driver] {message}", file=sys.stderr)


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _content(value):
    if isinstance(value, (np.ndarray, pd.Series, pd.Index)): return pd.util.hash_array(np.asarray(value)).tobytes()
    return repr(sorted(value.items()) if isinstance(value, dict) else value).encode()


def _window(stage):
    return (WARMUP_START, OUTER[1]) if stage == "outer" else INNER


def _prepare(src, stage):
    """Steps 2-4: bind run, roster, models; read production and tables. Returns (roster, bound_tables, production)."""
    check_sources(src)
    if stage not in STAGES: raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    start, end = (INNER[0], OUTER[1]) if stage == "outer" else INNER  # the outer pool needs all history from INNER[0]
    identity = inputs.bind_run(src.plan, src.read, expected=src.expected, panel_id=src.panel_id)
    roster = inputs.resolve_roster(src.plan, src.selection, outer_year=OUTER_YEAR, pair_candidates=src.pair_candidates, blend_ids=src.blend_ids)
    bound = inputs.bind_models(src.plan, src.read, roster, src.admit)
    production = readers.read_production(src.fetch, src.admitted, expected_hash=src.production_hash, table_hash=src.production_table_hash, start=start, end=end)
    return roster, {"identity": identity, "bound": bound, "tables": readers.model_tables(bound, roster, src.admitted, production, start=start, end=end)}, production


def model_events(table, facts, chunk, exits):
    """One model's events in one ladder chunk. A -1 dte_exit (no exit quotes) becomes expiry minus exit_date."""
    ids = [str(e) for e in chunk["event_id"]]
    dte = {e: int((pd.Timestamp(chunk["expiry"][i]) - pd.Timestamp(exits[e])).days) if d < 0 else d for i, (e, d) in enumerate(zip(ids, chunk["dte_exit"].tolist()))}
    part = table[table.index.astype(str).isin(ids)]
    eids = part.index.astype(str).to_numpy()
    fx = facts.set_axis(facts.index.astype(str)).reindex(eids)
    move = part["pred_move"].to_numpy(dtype=float)
    out = pd.DataFrame({"event_id": eids, "event_date": pd.to_datetime(part["event_date"]).to_numpy(), "pred_move": move, "pred_abs_move": move, "pred_crush": part["pred_crush"].to_numpy(dtype=float), "dte_exit": [dte[e] for e in eids]})
    for col in _FACTS: out[col] = fx[col].to_numpy(dtype=float)
    return out


def _smoke(src, event_ids):
    """The smoke id set, or None. Ids outside the admitted population refuse; a subset never widens anything."""
    if event_ids is None: return None
    if not (isinstance(event_ids, (set, frozenset, list, tuple)) and all(isinstance(e, str) for e in event_ids)):
        raise fail("INVALID_EXPERIMENT_SPEC", "smoke event ids must be a collection of strings", details={"reason": "event_ids_invalid"})
    ids = {str(e) for e in event_ids}
    if not ids <= set(src.admitted["events"]["event_id"].astype(str)): raise fail(_DENIED, "smoke event ids include a key that is not admitted", details={"reason": "key_not_admitted"})
    if not ids: raise Refusal(_CONFLICT, "the smoke event id set is empty", {"reason": "empty_population"})
    return ids


def _restrict(table, ids):
    return table if ids is None else table[table.index.astype(str).isin(list(ids))]


def _plan_window(plan, window, ids):
    dates = pd.to_datetime(plan["event_date"])
    keep = ((dates >= pd.Timestamp(window[0])) & (dates < pd.Timestamp(window[1]))).to_numpy()
    if ids is not None: keep = keep & plan["event_id"].astype(str).isin(list(ids)).to_numpy()
    return plan[keep]


def _scope(ids):
    return None if ids is None else content_hash(sorted(ids))


def _plan_rows(plan):
    """The windowed plan as rows sorted by event_id, so a changed window or date changes the receipt."""
    return sorted([[str(e), str(d), str(en), str(x), s, t] for e, d, en, x, s, t in zip(plan["event_id"], plan["event_date"], plan["entry_date"], plan["exit_date"], plan["session"], plan["ticker"])], key=lambda r: r[0])


def _receipt(path, body, reason):
    """``tuning.write_receipt``, with a changed replay refused under ``reason`` so the refusal names the receipt."""
    try:
        return tuning.write_receipt(path, body)
    except shortvol.Refusal as refusal:
        if refusal.details.get("reason") != "tuning_receipt_changed": raise
        raise shortvol.Refusal(_CONFLICT, str(refusal), {**refusal.details, "reason": reason}) from refusal


def _exits(src):
    """Event id -> the event's exit_date as an ISO date, where known."""
    return {str(e): pd.Timestamp(x).date().isoformat() for e, x in zip(src.plan_frame["event_id"], src.plan_frame["exit_date"]) if pd.notna(x)}


def _pool_tables(ready):
    """Each ready model's table on the ids every ready model has: one population rule for both stages (design section 4)."""
    common = set.intersection(*(set(e["table"].index.astype(str)) for e in ready.values()))
    return {mid: e["table"][e["table"].index.astype(str).isin(common)] for mid, e in ready.items()}


def _label_rows(frames):
    """Label rows [event_id, abs_move, crush_pct_iv30] of the label frames, sorted by event_id (the receipts' shape)."""
    return sorted([[str(e), float(a), float(c)] for f in frames for e, a, c in zip(f.index, f["abs_move"], f["crush_pct_iv30"])], key=lambda r: r[0])


def _common_dropped(ready):
    """Ids in some ready model's table but not in every one's: the common-support drop (design section 4)."""
    sets = [set(e["table"].index.astype(str)) for e in ready.values()]
    return len(set.union(*sets) - set.intersection(*sets))


def _printed(labels, exits, cutoff):
    """The labels whose event's exit_date is printed before ``cutoff`` (an ISO date)."""
    ids = labels.index.astype(str)
    return labels[ids.isin([e for e in ids if e in exits and label_availability.available_before(exits[e], cutoff)])]


def _tally(total, chunk):
    for reason, n in chunk.get("skipped", {}).items(): total[str(reason)] = total.get(str(reason), 0) + int(n)
    return total


def _sims(chunk, events, table, labels, exits):
    """One model's simulations per calendar month; each month's pool uses only rows and labels known before it."""
    sims, months = {}, events["event_date"].dt.to_period("M")
    for period in sorted(set(months)):
        pool = build_pool(table[table["event_date"] < period.start_time], _printed(labels, exits, period.start_time.date().isoformat()))
        sims.update(shortvol.simulate(chunk, events[months == period], pool))
    return sims


def _inner(src, out, ready, plan, facts_all, labels):
    exits = _exits(src)
    items, quarters = {mid: [] for mid in ready}, pd.to_datetime(plan["event_date"]).dt.to_period("Q").nunique()
    kept, started, skipped, common, digest = 0, time.monotonic(), {}, {}, hashlib.sha256()
    pool_tables = _pool_tables(ready)
    for number, (label, chunk) in enumerate(ladder.ladder_chunks(src.admitted, plan, src.load_index), start=1):
        digest.update(str(label).encode() + b"".join(f"{key}\0".encode() + _content(chunk[key]) for key in sorted(chunk)))  # the consumed quotes, bound
        kept += len(chunk["event_id"])
        skipped = _tally(skipped, chunk)
        built = {}
        for mid, entry in ready.items():
            events = model_events(entry["table"], facts_all, chunk, exits)  # facts_all: the run's only facts call
            built[mid] = (events, _sims(chunk, events, pool_tables[mid], labels, exits))
        # A no-sim event, or an id missing from ANY model's table, leaves for ALL, cutoff history included (design section 4).
        dropped = {e for events, sims in built.values() for e in events["event_id"] if sims.get(e) is None}
        dropped |= {e for events, _s in built.values() for e in set(map(str, chunk["event_id"])) - set(events["event_id"])}
        common[label] = len(dropped)
        for mid, (events, sims) in built.items():
            items[mid].append((chunk, events[~events["event_id"].isin(dropped)], {e: s for e, s in sims.items() if e not in dropped}))
        elapsed = max(time.monotonic() - started, 1e-9)
        _log(f"inner quarter {number}/{quarters} {label} events={len(chunk['event_id'])} " f"rate={kept / elapsed:.2f}/s eta={(quarters - number) * elapsed / number:.0f}s rss={_rss_mb():.0f}MB")
    bad = set()  # structure-level refusal at ANY c, name or model: the event leaves every model's items, history included
    for mid in ready:
        for c in tuning.C_GRID:
            for name in shortvol.NAMES:
                rows = pd.concat([shortvol.score_events(chunk, events, sims, name, alpha=tuning.ALPHA, c=c) for chunk, events, sims in items[mid]], ignore_index=True)
                bad |= set(rows.loc[rows["status"] != "scored", "event_id"].astype(str))
    for mid in ready:  # the common population is one event set for history and trades alike (design section 4)
        items[mid] = [(chunk, events[~events["event_id"].isin(bad)], {e: s for e, s in sims.items() if e not in bad}) for chunk, events, sims in items[mid]]
    by_model = {mid: sum(len(events) for _c, events, _s in items[mid]) for mid in ready}
    hashes = {}
    for i, mid in enumerate(ready, start=1):
        started = time.monotonic()
        body = tuning.tune(items[mid], mid)
        hashes[mid] = tuning.write_receipt(out / "receipts" / f"tuning_{mid}.json", body)["content_hash"]
        _log(f"tuning model {i}/{len(ready)} {mid} elapsed={time.monotonic() - started:.1f}s " f"c={body['c']} skipped={body['skipped']}")
    return hashes, max(by_model.values(), default=0), skipped, by_model, common, len(bad), "sha256:" + digest.hexdigest()


def _outer_gate(out, ready, scope, body):
    """Refuses before any chunk is built: every ready model needs its tuning receipt, and inner.json must bind those
    receipts, have been tuned on this run's event scope, and bind this run's inputs."""
    receipts = {mid: tuning.load_receipt(out / "receipts" / f"tuning_{mid}.json") for mid in ready}
    inner = tuning.load_receipt(out / "receipts" / "inner.json")
    if any(inner["models"].get(mid) != receipts[mid]["content_hash"] for mid in ready):
        raise Refusal(_CONFLICT, "inner.json does not bind every ready model's tuning receipt", {"reason": "inner_receipt_mismatch"})
    if "scope" not in inner or inner["scope"] != scope: raise Refusal(_CONFLICT, "inner.json was tuned on a different event scope", {"reason": "scope_mismatch"})
    inputs_inner, now = tuning.load_receipt(out / "receipts" / "inputs_inner.json"), json.loads(json.dumps(body))
    if any(inputs_inner.get(k) != now[k] for k in _BOUND): raise Refusal(_CONFLICT, "inputs_inner.json was bound to different inputs", {"reason": "inner_inputs_mismatch"})
    if inner["inputs"] != inputs_inner["content_hash"]: raise Refusal(_CONFLICT, "inner.json was tuned on different inner tables", {"reason": "inner_tables_mismatch"})
    return receipts


def _no_ready():
    return shortvol.Refusal(_CONFLICT, "no model is ready to be tuned or scored", {"reason": "no_ready_models"})


def _unbound_outer(tables, ready):
    """Refuses a ready model with no table rows in the outer year (its 2025 predictions are not bound yet). Partial coverage is not refused: it is a common-support drop."""
    for mid in ready:
        dates = pd.to_datetime(tables[mid]["table"]["event_date"])
        if not ((dates >= pd.Timestamp(OUTER[0])) & (dates < pd.Timestamp(OUTER[1]))).any():
            raise Refusal(_CONFLICT, "2025 predictions are not bound for a ready model with no 2025 rows", {"reason": "outer_predictions_unbound"})


def _model_state(c):
    edges = pd.DataFrame({"event_date": pd.Series([], dtype="datetime64[ns]"), "edge": pd.Series([], dtype=float)})
    return {"c": float(c), "scored": {"as_is": [], "retuned": []}, "edges": edges}


def _rows(mid, book, arm, dec, taken, events):
    """Decision rows of one (model, book, arm), with the marks of the taken rows and the model's forecast (pred_move, pred_crush) merged on event_id."""
    take = [c for c in [*_MARKS, "entry_cost", "pnl_per_share"] if c not in dec.columns]
    fc = events[["event_id", "pred_move", "pred_crush"]].assign(event_id=events["event_id"].astype(str))
    frame = dec.merge(taken[["event_id", *take]], on="event_id", how="left").merge(fc, on="event_id", how="left")
    if "structure" not in frame.columns: frame = frame.assign(structure=None)
    return frame.assign(model_id=mid, book=book, arm_or_rule=arm)


def _month(mid, table, events, chunk, labels, exits, start_m, st):
    """Steps 7a-7d for one model and one calendar month, with the pool of rows and labels known before the month (R6).
    Warm-up months (before OUTER[0]) only add history and emit no rows."""
    output = start_m >= pd.Timestamp(OUTER[0])
    pool = build_pool(table[table["event_date"] < start_m], _printed(labels, exits, start_m.date().isoformat()))
    sims = shortvol.simulate(chunk, events, pool)
    ids, frames = set(events["event_id"].astype(str)), []
    for arm, c in (("as_is", 1.0), ("retuned", st["c"])):
        for name in shortvol.NAMES:
            st["scored"][arm].append(shortvol.score_events(chunk, events, sims, name, alpha=ALPHA, c=c))
            if not output: continue
            decided = shortvol.decide(pd.concat(st["scored"][arm], ignore_index=True))
            mine = decided[decided["event_id"].astype(str).isin(ids) & (decided["structure"] == name)]
            frames.append(_rows(mid, "short_vol", arm, mine, shortvol.score(chunk, events, mine, name, ALPHA, c=c), events))
    first = None
    for rule in straddle.RULES:
        dec = straddle.decide(events, st["edges"], rule, chunk)
        first = dec if first is None else first
        if output: frames.append(_rows(mid, "straddle", rule, dec, straddle.score(chunk, events, dec, ALPHA), events))
    st["edges"] = pd.concat([st["edges"], first[["event_date", "edge"]]], ignore_index=True)
    return frames


def _outer(src, out, ready, receipts, plan, ids, facts_all, labels):
    """The outer stage: per ladder chunk and calendar month, decide, then read that month's labels (R6). Publishes decisions."""
    exits, pools = _exits(src), _pool_tables(ready)
    dates = pd.to_datetime(plan["event_date"])
    quarter, month = dates.dt.to_period("Q").astype(str).to_numpy(), dates.dt.to_period("M").to_numpy()
    total, state = len(set(month)), {mid: _model_state(receipts[mid]["c"]) for mid in ready}
    frames, done, started, skipped, fresh = [], 0, time.monotonic(), {}, []
    for label, chunk in ladder.ladder_chunks(src.admitted, plan, src.load_index):
        skipped = _tally(skipped, chunk)
        events = {mid: model_events(pools[mid], facts_all, chunk, exits) for mid in ready}
        for period in sorted({m for m, q in zip(month, quarter) if q == label}):
            start_m, month_frames = period.start_time, []
            for mid in ready:
                at = events[mid][events[mid]["event_date"].dt.to_period("M") == period]
                month_frames += _month(mid, pools[mid], at, chunk, labels, exits, start_m, state[mid])
            frames += month_frames
            done += 1
            rate = done / max(time.monotonic() - started, 1e-9)
            skips = pd.concat(month_frames)["reason"].value_counts().to_dict() if month_frames else {}
            _log(f"outer month {period} {done}/{total} rate={rate:.2f}/s eta={(total - done) / rate:.0f}s skips={skips} rss={_rss_mb():.0f}MB")
            if start_m >= pd.Timestamp(OUTER[0]):  # R6: only after this month's rows are complete
                got = readers.read_labels(src.fetch, src.admitted, start=start_m, end=(period + 1).start_time, event_ids=ids)
                fresh.append(got)
                labels = pd.concat([labels, got])
    inputs_hash = tuning.load_receipt(out / "receipts" / "inputs_outer.json")["content_hash"]
    body = {"stage": "outer", "models": list(ready), "ladder_skipped": skipped, "scope": _scope(ids), "inputs": inputs_hash,
            "tuning": {mid: receipts[mid]["content_hash"] for mid in ready}, "c": {mid: receipts[mid]["c"] for mid in ready},
            "labels": content_hash(_label_rows(fresh)), "common_dropped": _common_dropped(ready)}
    rows = decisions.publish(out, frames, body)
    return rows, skipped


def run(src, stage, out_dir, *, event_ids=None, prereg=size_pnl_prereg.require_preregistration):
    """Run one stage, write its receipts under ``out_dir``, and return a JSON-safe summary.

    ``event_ids`` is an optional smoke set: every model table, the plan and the labels are restricted to it, and its
    scope is receipted, so a smoke receipt cannot pass for a full-population one.
    """
    started = time.monotonic()
    receipt = prereg()
    check_sources(src)
    # The supervisor owns heavy jobs: this run must sit under tools/bounded_run.py --heavy.
    if os.environ.get("BOUNDED_RUN_NESTED") != "1":
        raise fail("RESOURCE_UNAVAILABLE", "run under tools/bounded_run.py --heavy", details={"reason": "not_bounded"})
    out, ids = Path(out_dir), _smoke(src, event_ids)
    roster, bound_tables, production = _prepare(src, stage)  # refuses any stage but "inner" or "outer" before reading
    tables = bound_tables["tables"]
    ready = {mid: {**e, "table": _restrict(e["table"], ids)} for mid, e in tables.items() if e["status"] == "ready"}
    excluded = {mid: e["reason"] for mid, e in tables.items() if e["status"] != "ready"}
    full = src.plan_frame
    if full[[c for c in ("event_date", "entry_date", "exit_date") if c in full.columns]].isna().to_numpy().any(): raise fail("INVALID_EXPERIMENT_SPEC", "plan dates must not be null", details={"reason": "plan_dates_null"})
    ladder.check_plan(src.admitted, full)  # the full plan, before the window can drop any row
    plan = _plan_window(full, _window(stage), ids)
    admitted_dates = dict(zip(src.admitted["events"]["event_id"].astype(str), pd.to_datetime(src.admitted["events"]["event_date"])))
    if any(str(e) not in admitted_dates for e in plan["event_id"]): raise fail(_DENIED, "plan event ids include a key that is not admitted", details={"reason": "key_not_admitted"})
    facts_all = src.facts(sorted(str(e) for e in plan["event_id"]))  # the only facts call of the run
    if not isinstance(facts_all, pd.DataFrame) or not facts_all.columns.is_unique: raise fail("INVALID_EXPERIMENT_SPEC", "facts source is incomplete", details={"reason": "facts_incomplete"})
    raw = facts_all.reindex(columns=list(_FACTS)).where(lambda c: c.notna(), np.nan)  # pd.NA and None are NaN
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    if (not set(_FACTS) <= set(facts_all.columns) or set(facts_all.index.astype(str)) != {str(e) for e in plan["event_id"]}
            or not facts_all.index.astype(str).is_unique or (numeric.isna() & raw.notna()).to_numpy().any() or np.isinf(numeric.to_numpy(dtype=float)).any()): raise fail("INVALID_EXPERIMENT_SPEC", "facts source is incomplete", details={"reason": "facts_incomplete"})
    facts_all = pd.concat([facts_all.drop(columns=list(_FACTS)), numeric], axis=1)
    _log(f"facts_nan_rows={int(facts_all[list(_FACTS)].isna().any(axis=1).sum())}")
    facts_rows = sorted([str(e), *[None if pd.isna(v) else float(v) for v in vals]] for e, vals in zip(facts_all.index, facts_all[list(_FACTS)].to_numpy(dtype=float)))
    labels = readers.read_labels(src.fetch, src.admitted, start=INNER[0], end=OUTER[0], event_ids=ids)  # the pre-2025 labels; outer adds each month's
    scope = _scope(ids)
    body = {"stage": "inputs", "spec_hash": receipt["spec_hash"], "identity": bound_tables["identity"],
            "roster": roster,  # inputs-level status per model: arrays are not receipted
            "models": {mid: ({"status": "ready"} if mid in ready and m["status"] == "deferred"
                             else {k: m[k] for k in ("status", "reason", "binding") if k in m})
                       for mid, m in bound_tables["bound"].items()},
            "tables": {mid: e["binding"] for mid, e in ready.items()},
            "production": production["binding"], "scope": scope, "plan": content_hash(_plan_rows(plan)),
            "facts": content_hash(facts_rows), "labels": content_hash(_label_rows([labels])), "snapshot": src.snapshot_id}
    receipts = {}
    if stage == "outer":  # preflight: a refusal here leaves no outer input receipt, and no chunk is built
        if not ready: raise _no_ready()
        receipts = _outer_gate(out, ready, scope, body)
        _unbound_outer(tables, ready)
    inputs_receipt = _receipt(out / "receipts" / f"inputs_{stage}.json", body, "inputs_receipt_changed")
    if not ready: raise _no_ready()
    if plan.empty: raise shortvol.Refusal(_CONFLICT, "the plan window holds no events", {"reason": "empty_population"})
    if stage == "inner":
        hashes, rows, skipped, by_model, common, structure, ladder_hash = _inner(src, out, ready, plan, facts_all, labels)
        _receipt(out / "receipts" / "inner.json",
                 {"stage": "inner", "models": hashes, "events": rows, "events_by_model": by_model, "ladder": ladder_hash,
                  "excluded": excluded, "scope": scope, "inputs": inputs_receipt["content_hash"],
                  "ladder_skipped": skipped, "common_dropped": common, "common_dropped_structure": structure}, "inner_receipt_changed")
        _log(f"stage={stage} end elapsed={time.monotonic() - started:.1f}s rss={_rss_mb():.0f}MB "
             f"ladder_skipped={skipped} common_dropped={common} common_dropped_structure={structure}")
    else:
        rows, skipped = _outer(src, out, ready, receipts, plan, ids, facts_all, labels)
        _log(f"stage={stage} end elapsed={time.monotonic() - started:.1f}s rss={_rss_mb():.0f}MB ladder_skipped={skipped}")
    return {"stage": stage, "models": sorted(ready), "excluded": excluded, "rows": rows, "ladder_skipped": skipped}
