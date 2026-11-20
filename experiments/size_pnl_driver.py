"""EXP-188 S9a: driver for the inner stage (2023-2024) of the SIZE decision-level P&L re-score.

Orchestrates S2a-S8 through the injected ``Sources``: binds inputs, runs the S3 ladder per quarter, tunes each
model's divisor and writes the receipts. The outer stage, ``--event-ids`` smoke and decisions output are S9b
(issue #2660). ``prepare`` and ``model_events`` serve the synthetic arm.
"""
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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


def _log(message):
    print(f"[pnl_driver] {message}", file=sys.stderr)


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def prepare(src, stage):
    """Steps 2-4: bind run, roster and models, then read production and the stage's tables.

    Returns ``(roster, bound_tables, production)``; ``bound_tables`` holds ``identity``, ``bound`` and ``tables``.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    start, end = INNER
    identity = inputs.bind_run(src.plan, src.read, expected=src.expected, panel_id=src.panel_id)
    roster = inputs.resolve_roster(src.plan, src.selection, outer_year=OUTER_YEAR,
                                   pair_candidates=src.pair_candidates, blend_ids=src.blend_ids)
    bound = inputs.bind_models(src.plan, src.read, roster, src.admit)
    production = readers.read_production(src.fetch, src.admitted, expected_hash=src.production_hash,
                                         table_hash=src.production_table_hash, start=start, end=end)
    tables = readers.model_tables(bound, roster, src.admitted, production, start=start, end=end)
    return roster, {"identity": identity, "bound": bound, "tables": tables}, production


def model_events(table, facts, chunk):
    """One model's events in one ladder chunk: its table rows joined with facts and the chunk's dte_exit."""
    ids = [str(e) for e in chunk["event_id"]]
    dte = dict(zip(ids, chunk["dte_exit"].tolist()))
    part = table[table.index.astype(str).isin(ids)]
    eids = part.index.astype(str).to_numpy()
    fx = facts.copy()
    fx.index = fx.index.astype(str)
    fx = fx.reindex(eids)
    move = part["pred_move"].to_numpy(dtype=float)
    out = pd.DataFrame({"event_id": eids, "event_date": pd.to_datetime(part["event_date"]).to_numpy(),
                        "pred_move": move, "pred_abs_move": move,
                        "pred_crush": part["pred_crush"].to_numpy(dtype=float), "dte_exit": [dte[e] for e in eids]})
    for col in _FACTS:
        out[col] = fx[col].to_numpy(dtype=float)
    return out


def _plan_window(plan):
    dates = pd.to_datetime(plan["event_date"])
    return plan[((dates >= pd.Timestamp(INNER[0])) & (dates < pd.Timestamp(INNER[1]))).to_numpy()]


def _plan_rows(plan):
    """The windowed plan as rows sorted by event_id, so a changed window or date changes the receipt."""
    rows = [[str(e), str(d), str(en), str(x), s, t] for e, d, en, x, s, t in zip(
        plan["event_id"], plan["event_date"], plan["entry_date"], plan["exit_date"], plan["session"], plan["ticker"])]
    return sorted(rows, key=lambda r: r[0])


def _exit_map(plan_frame):
    """Event id -> exit_date as an ISO date, for the events whose exit is known."""
    return {str(e): pd.Timestamp(x).date().isoformat()
            for e, x in zip(plan_frame["event_id"], plan_frame["exit_date"]) if pd.notna(x)}


def _printed(labels, exits, cutoff):
    """The labels whose event's exit_date is printed before ``cutoff`` (an ISO date)."""
    ids = labels.index.astype(str)
    return labels[ids.isin([e for e in ids if e in exits and label_availability.available_before(exits[e], cutoff)])]


def _sims(chunk, events, table, labels, exits):
    """One model's simulations per calendar month; each month's pool uses only rows and labels known before it."""
    sims, months = {}, events["event_date"].dt.to_period("M")
    for period in sorted(set(months)):
        start_m = period.start_time
        pool = build_pool(table[table["event_date"] < start_m], _printed(labels, exits, start_m.date().isoformat()))
        sims.update(shortvol.simulate(chunk, events[months == period], pool))
    return sims


def _inner(src, out, ready, plan, facts_all):
    labels = readers.read_labels(src.fetch, src.admitted, start=INNER[0], end=INNER[1], event_ids=None)
    exits = _exit_map(src.plan_frame)
    items = {mid: [] for mid in ready}
    quarters = pd.to_datetime(plan["event_date"]).dt.to_period("Q").nunique()
    kept, started, skipped = 0, time.monotonic(), {}
    for number, (label, chunk) in enumerate(ladder.ladder_chunks(src.admitted, plan, src.load_index), start=1):
        kept += len(chunk["event_id"])
        for reason, n in chunk.get("skipped", {}).items():
            skipped[str(reason)] = skipped.get(str(reason), 0) + int(n)
        for mid, entry in ready.items():
            events = model_events(entry["table"], facts_all, chunk)  # facts_all: the run's only facts call
            items[mid].append((chunk, events, _sims(chunk, events, entry["table"], labels, exits)))
        elapsed = max(time.monotonic() - started, 1e-9)
        _log(f"inner quarter {number}/{quarters} {label} events={len(chunk['event_id'])} "
             f"rate={kept / elapsed:.2f}/s eta={(quarters - number) * elapsed / number:.0f}s rss={_rss_mb():.0f}MB")
    by_model = {mid: sum(len(events) for _c, events, _s in items[mid]) for mid in ready}
    hashes = {}
    for i, mid in enumerate(ready, start=1):
        started = time.monotonic()
        body = tuning.tune(items[mid], mid)
        hashes[mid] = tuning.write_receipt(out / "receipts" / f"tuning_{mid}.json", body)["content_hash"]
        _log(f"tuning model {i}/{len(ready)} {mid} elapsed={time.monotonic() - started:.1f}s "
             f"c={body['c']} skipped={body['skipped']}")
    return hashes, kept, skipped, by_model


def run(src, stage, out_dir, *, prereg=size_pnl_prereg.require_preregistration):
    """Run the inner stage, write its receipts under ``out_dir``, and return a JSON-safe summary."""
    started = time.monotonic()
    receipt = prereg()
    _log(f"stage={stage} start rss={_rss_mb():.0f}MB")
    out = Path(out_dir)
    roster, bound_tables, production = prepare(src, stage)  # refuses any stage but "inner" before reading
    tables = bound_tables["tables"]
    ready = {mid: e for mid, e in tables.items() if e["status"] == "ready"}
    excluded = {mid: e["reason"] for mid, e in tables.items() if e["status"] != "ready"}
    plan = _plan_window(src.plan_frame)
    admitted_dates = dict(zip(src.admitted["events"]["event_id"].astype(str),
                              pd.to_datetime(src.admitted["events"]["event_date"])))
    if any(str(e) not in admitted_dates for e in plan["event_id"]):
        raise fail(_DENIED, "plan event ids include a key that is not admitted", details={"reason": "key_not_admitted"})
    if any(pd.Timestamp(d) != admitted_dates[str(e)] for e, d in zip(plan["event_id"], plan["event_date"])):
        raise fail(_DENIED, "plan event dates differ from the admitted dates",
                   details={"reason": "plan_date_mismatch"})
    facts_all = src.facts(sorted(str(e) for e in plan["event_id"]))  # the only facts call of the run
    facts_rows = sorted([str(e), *[None if pd.isna(v) else float(v) for v in vals]] for e, vals in
                        zip(facts_all.index, facts_all[list(_FACTS)].to_numpy(dtype=float)))
    body = {"stage": "inputs", "spec_hash": receipt["spec_hash"], "identity": bound_tables["identity"],
            "roster": roster,  # inputs-level status per model: arrays are not receipted
            "models": {mid: {k: m[k] for k in ("status", "reason", "binding") if k in m}
                       for mid, m in bound_tables["bound"].items()},
            "tables": {mid: e["binding"] for mid, e in ready.items()},
            "production": production["binding"], "scope": None, "plan": content_hash(_plan_rows(plan)),
            "facts": content_hash(facts_rows), "snapshot": src.snapshot_id}
    inputs_receipt = tuning.write_receipt(out / "receipts" / "inputs_inner.json", body)
    if not ready:
        raise shortvol.Refusal(_CONFLICT, "no model is ready to be tuned or scored", {"reason": "no_ready_models"})
    if plan.empty:
        raise shortvol.Refusal(_CONFLICT, "the plan window holds no events", {"reason": "empty_population"})
    hashes, rows, skipped, by_model = _inner(src, out, ready, plan, facts_all)
    tuning.write_receipt(out / "receipts" / "inner.json",
                         {"stage": "inner", "models": hashes, "events": rows, "events_by_model": by_model,
                          "excluded": excluded, "scope": None, "inputs": inputs_receipt["content_hash"],
                          "ladder_skipped": skipped})
    _log(f"stage={stage} end elapsed={time.monotonic() - started:.1f}s rss={_rss_mb():.0f}MB "
         f"ladder_skipped={skipped}")
    return {"stage": stage, "models": sorted(ready), "excluded": excluded, "rows": rows, "ladder_skipped": skipped}
