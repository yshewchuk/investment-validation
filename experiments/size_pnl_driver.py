"""EXP-188 S9a: inner-stage (2023-2024) driver of the SIZE P&L re-score; outer stage S9b (#2660); the synthetic arm gets a gated helper in its own slice."""
import hashlib
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
from experiments import label_availability, size_pnl_inputs as inputs, size_pnl_ladder as ladder, size_pnl_prereg
from experiments import size_pnl_readers as readers, size_pnl_shortvol as shortvol, size_pnl_tuning as tuning
from experiments.size_pnl_pool import build_pool

INNER = ("2023-01-01", "2025-01-01")
OUTER_YEAR = 2025  # the roster is resolved for this year; the outer stage itself is S9b
STAGES = ("inner",)
_DENIED, _CONFLICT = "HOLDOUT_ACCESS_DENIED", "EXPERIMENT_IDENTITY_CONFLICT"
_FACTS = ("pre_iv30", "mcap_usd", "implied_move")


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
    if not (isinstance(src, Sources) and isinstance(src.expected, dict) and isinstance(src.plan_frame, pd.DataFrame)
            and isinstance(src.plan, dict) and isinstance(src.snapshot_id, str) and src.snapshot_id.strip() and all(callable(getattr(src, f)) for f in ("read", "admit", "fetch", "load_index", "facts"))):
        raise fail("INVALID_EXPERIMENT_SPEC", "sources must be a Sources with an object `expected`", details={"reason": "sources_invalid"})
    return src


def _log(message):
    print(f"[pnl_driver] {message}", file=sys.stderr)


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _content(value):
    if isinstance(value, (np.ndarray, pd.Series, pd.Index)): return pd.util.hash_array(np.asarray(value)).tobytes()
    return repr(sorted(value.items()) if isinstance(value, dict) else value).encode()


def _prepare(src, stage):
    """Steps 2-4: bind run, roster, models; read production and tables. Returns (roster, bound_tables, production)."""
    check_sources(src)
    if stage not in STAGES: raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    start, end = INNER
    identity = inputs.bind_run(src.plan, src.read, expected=src.expected, panel_id=src.panel_id)
    roster = inputs.resolve_roster(src.plan, src.selection, outer_year=OUTER_YEAR,
                                   pair_candidates=src.pair_candidates, blend_ids=src.blend_ids)
    bound = inputs.bind_models(src.plan, src.read, roster, src.admit)
    production = readers.read_production(src.fetch, src.admitted, expected_hash=src.production_hash,
                                         table_hash=src.production_table_hash, start=start, end=end)
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


def _printed(labels, exits, cutoff):
    """The labels whose event's exit_date is printed before ``cutoff`` (an ISO date)."""
    ids = labels.index.astype(str)
    return labels[ids.isin([e for e in ids if e in exits and label_availability.available_before(exits[e], cutoff)])]


def _sims(chunk, events, table, labels, exits):
    """One model's simulations per calendar month; each month's pool uses only rows and labels known before it."""
    sims, months = {}, events["event_date"].dt.to_period("M")
    for period in sorted(set(months)):
        pool = build_pool(table[table["event_date"] < period.start_time], _printed(labels, exits, period.start_time.date().isoformat()))
        sims.update(shortvol.simulate(chunk, events[months == period], pool))
    return sims


def _inner(src, out, ready, plan, facts_all, labels):
    exits = {str(e): pd.Timestamp(x).date().isoformat()  # event id -> exit_date (ISO), where known
             for e, x in zip(src.plan_frame["event_id"], src.plan_frame["exit_date"]) if pd.notna(x)}
    items, quarters = {mid: [] for mid in ready}, pd.to_datetime(plan["event_date"]).dt.to_period("Q").nunique()
    kept, started, skipped, common, digest = 0, time.monotonic(), {}, {}, hashlib.sha256()
    common_ids = set.intersection(*(set(e["table"].index.astype(str)) for e in ready.values()))  # forecast coverage shared by every ready model
    pool_tables = {mid: e["table"][e["table"].index.astype(str).isin(common_ids)] for mid, e in ready.items()}  # residual pools use only common ids
    for number, (label, chunk) in enumerate(ladder.ladder_chunks(src.admitted, plan, src.load_index), start=1):
        digest.update(str(label).encode() + b"".join(f"{key}\0".encode() + _content(chunk[key]) for key in sorted(chunk)))  # the consumed quotes, bound
        kept += len(chunk["event_id"])
        for reason, n in chunk.get("skipped", {}).items(): skipped[str(reason)] = skipped.get(str(reason), 0) + int(n)
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
                rows = pd.concat([shortvol.score_events(chunk, events, sims, name, alpha=tuning.ALPHA, c=c)
                                  for chunk, events, sims in items[mid]], ignore_index=True)
                bad |= set(rows.loc[rows["status"] != "scored", "event_id"].astype(str))
    for mid in ready:  # the common population is one event set for history and trades alike (design section 4)
        items[mid] = [(chunk, events[~events["event_id"].isin(bad)], {e: s for e, s in sims.items() if e not in bad})
                      for chunk, events, sims in items[mid]]
    by_model = {mid: sum(len(events) for _c, events, _s in items[mid]) for mid in ready}
    hashes = {}
    for i, mid in enumerate(ready, start=1):
        started = time.monotonic()
        body = tuning.tune(items[mid], mid)
        hashes[mid] = tuning.write_receipt(out / "receipts" / f"tuning_{mid}.json", body)["content_hash"]
        _log(f"tuning model {i}/{len(ready)} {mid} elapsed={time.monotonic() - started:.1f}s "
             f"c={body['c']} skipped={body['skipped']}")
    return hashes, max(by_model.values(), default=0), skipped, by_model, common, len(bad), "sha256:" + digest.hexdigest()


def run(src, stage, out_dir, *, prereg=size_pnl_prereg.require_preregistration):
    """Run the inner stage, write its receipts under ``out_dir``, and return a JSON-safe summary."""
    started = time.monotonic()
    check_sources(src)
    receipt = prereg()
    # The supervisor owns heavy jobs: this run must sit under tools/bounded_run.py --heavy.
    if os.environ.get("BOUNDED_RUN_NESTED") != "1":
        raise fail("RESOURCE_UNAVAILABLE", "run under tools/bounded_run.py --heavy", details={"reason": "not_bounded"})
    out = Path(out_dir)
    roster, bound_tables, production = _prepare(src, stage)  # refuses any stage but "inner" before reading
    ready = {mid: e for mid, e in bound_tables["tables"].items() if e["status"] == "ready"}
    excluded = {mid: e["reason"] for mid, e in bound_tables["tables"].items() if e["status"] != "ready"}
    full = src.plan_frame
    if full[[c for c in ("event_date", "entry_date", "exit_date") if c in full.columns]].isna().to_numpy().any(): raise fail("INVALID_EXPERIMENT_SPEC", "plan dates must not be null", details={"reason": "plan_dates_null"})
    ladder.check_plan(src.admitted, full)  # the full plan, before the window can drop any row
    dates = pd.to_datetime(full["event_date"])  # the inner window, as a row mask
    plan = full[((dates >= pd.Timestamp(INNER[0])) & (dates < pd.Timestamp(INNER[1]))).to_numpy()]
    admitted_dates = dict(zip(src.admitted["events"]["event_id"].astype(str), pd.to_datetime(src.admitted["events"]["event_date"])))
    if any(str(e) not in admitted_dates for e in plan["event_id"]): raise fail(_DENIED, "plan event ids include a key that is not admitted", details={"reason": "key_not_admitted"})
    facts_all = src.facts(sorted(str(e) for e in plan["event_id"]))  # the only facts call of the run
    if not isinstance(facts_all, pd.DataFrame): raise fail("INVALID_EXPERIMENT_SPEC", "facts source is incomplete", details={"reason": "facts_incomplete"})
    raw = facts_all.reindex(columns=list(_FACTS)).where(lambda c: c.notna(), np.nan)  # pd.NA and None are NaN
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    if (not set(_FACTS) <= set(facts_all.columns) or set(facts_all.index.astype(str)) != {str(e) for e in plan["event_id"]}
            or not facts_all.index.astype(str).is_unique or (numeric.isna() & raw.notna()).to_numpy().any() or np.isinf(numeric.to_numpy(dtype=float)).any()): raise fail("INVALID_EXPERIMENT_SPEC", "facts source is incomplete", details={"reason": "facts_incomplete"})
    facts_all = pd.concat([facts_all.drop(columns=list(_FACTS)), numeric], axis=1)
    _log(f"facts_nan_rows={int(facts_all[list(_FACTS)].isna().any(axis=1).sum())}")
    facts_rows = sorted([str(e), *[None if pd.isna(v) else float(v) for v in vals]] for e, vals in zip(facts_all.index, facts_all[list(_FACTS)].to_numpy(dtype=float)))
    labels = readers.read_labels(src.fetch, src.admitted, start=INNER[0], end=INNER[1], event_ids=None)  # inner only
    label_rows = sorted([[str(e), float(a), float(c)] for e, a, c in zip(labels.index, labels["abs_move"], labels["crush_pct_iv30"])], key=lambda r: r[0])
    body = {"stage": "inputs", "spec_hash": receipt["spec_hash"], "identity": bound_tables["identity"],
            "roster": roster,  # inputs-level status per model: arrays are not receipted
            "models": {mid: ({"status": "ready"} if mid in ready and m["status"] == "deferred"
                             else {k: m[k] for k in ("status", "reason", "binding") if k in m})
                       for mid, m in bound_tables["bound"].items()},
            "tables": {mid: e["binding"] for mid, e in ready.items()},
            "production": production["binding"], "scope": None, "plan": content_hash(_plan_rows(plan)),
            "facts": content_hash(facts_rows), "labels": content_hash(label_rows), "snapshot": src.snapshot_id}
    inputs_receipt = _receipt(out / "receipts" / "inputs_inner.json", body, "inputs_receipt_changed")
    if not ready: raise shortvol.Refusal(_CONFLICT, "no model is ready to be tuned or scored", {"reason": "no_ready_models"})
    if plan.empty: raise shortvol.Refusal(_CONFLICT, "the plan window holds no events", {"reason": "empty_population"})
    hashes, rows, skipped, by_model, common, structure, ladder_hash = _inner(src, out, ready, plan, facts_all, labels)
    _receipt(out / "receipts" / "inner.json",
             {"stage": "inner", "models": hashes, "events": rows, "events_by_model": by_model, "ladder": ladder_hash,
              "excluded": excluded, "scope": None, "inputs": inputs_receipt["content_hash"],
              "ladder_skipped": skipped, "common_dropped": common, "common_dropped_structure": structure}, "inner_receipt_changed")
    _log(f"stage={stage} end elapsed={time.monotonic() - started:.1f}s rss={_rss_mb():.0f}MB "
         f"ladder_skipped={skipped} common_dropped={common} common_dropped_structure={structure}")
    return {"stage": stage, "models": sorted(ready), "excluded": excluded, "rows": rows, "ladder_skipped": skipped}
