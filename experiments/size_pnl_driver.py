"""EXP-188 S9: driver for the SIZE decision-level P&L re-score (inner and outer stages).

S9 of guides/size_decision_pnl_design.md (sections 2, 6-9). A pure orchestrator over the slice modules
S2a-S8: it binds the inputs, reads production and labels only through the injected ``Sources``, runs
the S3 ladder one quarter at a time and writes the receipts. The synthetic arm
(guides/size_pnl_synthetic_calibration_design.md) will call ``prepare`` and ``model_events``.

- inner (2023-2024): per-model pools and simulations, the divisor re-tune (``tuning``), then the
  ``tuning_<model>`` receipts and the ``inner`` receipt.
- outer (2025): refuses before any chunk is built unless every ready model has its tuning receipt and
  ``inner.json`` exists. Per quarter and month: decisions use only labels printed before the month
  (R6). A month's rows are complete before that month's labels are read, and only then are the labels
  added to the pool. Warm-up months (2024 H2) add history only. Then ``decisions.parquet`` and the
  ``outer`` receipt are written.

The v14 2025 refit-prediction binder is a separate slice; until it exists a full outer run refuses
``outer_predictions_unbound``.

No production caller beyond ``run.py``.
"""
import hashlib
import json
import os
import resource
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from experiments import label_availability
from experiments import size_pnl_inputs as inputs
from experiments import size_pnl_ladder as ladder
from experiments import size_pnl_prereg
from experiments import size_pnl_readers as readers
from experiments import size_pnl_shortvol as shortvol
from experiments import size_pnl_straddle as straddle
from experiments import size_pnl_tuning as tuning
from experiments.size_pnl_pool import build_pool
from experiments.size_pnl_shortvol import Refusal

INNER = ("2023-01-01", "2025-01-01")
OUTER = ("2025-01-01", "2026-01-01")
WARMUP_START = "2024-07-01"  # six months of inner history so the trailing cutoff is defined in 2025-01
ALPHA = tuning.ALPHA
STAGES = ("inner", "outer")
_DENIED = "HOLDOUT_ACCESS_DENIED"
_FACTS = ("pre_iv30", "mcap_usd", "implied_move")
_MARKS = ["exit_value", "rung", "lots", "pnl_dollars"]
_FRONT = ["model_id", "book", "arm_or_rule", "structure", "event_id", "event_date", "status", "reason"]
_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
_HASH_BLOCK = 1 << 20
MAX_DECISIONS_BYTES = 4 << 30


@dataclass(frozen=True)
class Sources:
    """Every store-facing input of the run; tests pass fakes, the real store adapter is a later slice.

    ``snapshot_id`` is the pinned v2 snapshot id the chains come from; changed chain bytes refuse INPUT_CHANGED
    upstream; the id is receipted here.
    """
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


def _window(stage):
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    return INNER if stage == "inner" else (WARMUP_START, OUTER[1])


def _table_window(stage):
    """The model tables' window: the outer pool needs all history from INNER[0] (design section 4)."""
    _window(stage)  # validates the stage
    return (INNER[0], OUTER[1]) if stage == "outer" else INNER


def _strip(models):
    """Inputs-level status per model: status, reason and binding only (arrays are not receipted)."""
    return {mid: {k: m[k] for k in ("status", "reason", "binding") if k in m} for mid, m in models.items()}


def prepare(src, stage):
    """Steps 2-4: bind the run and roster, bind the models, then read production and the stage's tables.

    Returns ``(roster, bound_tables, production)``. ``bound_tables`` is ``{"identity", "bound", "tables"}``:
    the bound run identity, each model's binding status, and every model's table (ready or excluded).
    """
    start, end = _table_window(stage)
    identity = inputs.bind_run(src.plan, src.read, expected=src.expected, panel_id=src.panel_id)
    roster = inputs.resolve_roster(src.plan, src.selection, outer_year=int(OUTER[0][:4]),
                                   pair_candidates=src.pair_candidates, blend_ids=src.blend_ids)
    bound = inputs.bind_models(src.plan, src.read, roster, src.admit)
    production = readers.read_production(src.fetch, src.admitted, expected_hash=src.production_hash,
                                         table_hash=src.production_table_hash)
    tables = readers.model_tables(bound, roster, src.admitted, production, start=start, end=end)
    return roster, {"identity": identity, "bound": bound, "tables": tables}, production


def model_events(table, facts, chunk):
    """One model's events in one ladder chunk: its table rows, joined with facts and the chunk's dte_exit."""
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
                        "pred_crush": part["pred_crush"].to_numpy(dtype=float),
                        "dte_exit": [dte[e] for e in eids]})
    for col in _FACTS:
        out[col] = fx[col].to_numpy(dtype=float)
    return out


def _smoke(src, event_ids):
    """The smoke id set, or None. Ids outside the admitted population refuse; a subset never widens anything."""
    if event_ids is None:
        return None
    if not (isinstance(event_ids, (set, frozenset, list, tuple)) and all(isinstance(e, str) for e in event_ids)):
        raise fail("INVALID_EXPERIMENT_SPEC", "smoke event ids must be a collection of strings",
                   details={"reason": "event_ids_invalid"})
    ids = {str(e) for e in event_ids}
    if not ids <= set(src.admitted["events"]["event_id"].astype(str)):
        raise fail(_DENIED, "smoke event ids include a key that is not admitted",
                   details={"reason": "key_not_admitted"})
    return ids


def _restrict(table, ids):
    return table if ids is None else table[table.index.astype(str).isin(list(ids))]


def _plan_window(plan, window, ids):
    dates = pd.to_datetime(plan["event_date"])
    keep = ((dates >= pd.Timestamp(window[0])) & (dates < pd.Timestamp(window[1]))).to_numpy()
    if ids is not None:
        keep = keep & plan["event_id"].astype(str).isin(list(ids)).to_numpy()
    return plan[keep]


def _plan_rows(plan):
    """The windowed plan as plain rows sorted by event_id, so a changed window or date changes the receipt."""
    rows = [[str(e), str(d), str(en), str(x), s, t] for e, d, en, x, s, t in zip(
        plan["event_id"], plan["event_date"], plan["entry_date"], plan["exit_date"], plan["session"], plan["ticker"])]
    return sorted(rows, key=lambda r: r[0])


_BOUND = ("spec_hash", "identity", "roster", "models", "production", "snapshot")  # not "tables": windows differ by stage


def _outer_gate(out, ready, scope, body):
    """Refuses before any chunk is built: every ready model needs its tuning receipt, and inner.json must exist,
    bind those receipts, have been tuned on this run's event scope, and bind this run's inputs."""
    receipts = {mid: tuning.load_receipt(out / "receipts" / f"tuning_{mid}.json") for mid in ready}
    inner = tuning.load_receipt(out / "receipts" / "inner.json")
    if any(inner["models"].get(mid) != receipts[mid]["content_hash"] for mid in ready):
        raise Refusal(_CONFLICT, "inner.json does not bind every ready model's tuning receipt",
                      {"reason": "inner_receipt_mismatch"})
    if "scope" not in inner or inner["scope"] != scope:
        raise Refusal(_CONFLICT, "inner.json was tuned on a different event scope", {"reason": "scope_mismatch"})
    inputs_inner, now = tuning.load_receipt(out / "receipts" / "inputs_inner.json"), json.loads(json.dumps(body))
    if any(inputs_inner.get(k) != now[k] for k in _BOUND):
        raise Refusal(_CONFLICT, "inputs_inner.json was bound to different inputs", {"reason": "inner_inputs_mismatch"})
    if inner["inputs"] != inputs_inner["content_hash"]:
        raise Refusal(_CONFLICT, "inner.json was tuned on different inner tables", {"reason": "inner_tables_mismatch"})
    return receipts


def _exit_map(plan_frame):
    """Event id -> the event's exit_date as an ISO date, for the events whose exit is known."""
    return {str(e): pd.Timestamp(x).date().isoformat() for e, x in zip(plan_frame["event_id"], plan_frame["exit_date"])
            if pd.notna(x)}


def _printed(labels, exits, cutoff):
    """The labels whose event's exit_date is printed before ``cutoff`` (an ISO date); later exits are not yet known."""
    ids = labels.index.astype(str)
    return labels[ids.isin([e for e in ids if e in exits and label_availability.available_before(exits[e], cutoff)])]


def _chunk_facts(facts, chunk):
    """The shared facts frame restricted to one chunk's event ids, so no second source call is made."""
    fx = facts.copy()
    fx.index = fx.index.astype(str)
    return fx.reindex([str(e) for e in chunk["event_id"]])


def _tally(total, chunk):
    """Adds one ladder chunk's skip counts (reason -> count) into the running per-reason total."""
    for reason, n in chunk.get("skipped", {}).items():
        total[str(reason)] = total.get(str(reason), 0) + int(n)
    return total


def _sims(chunk, events, table, labels, exits):
    """One model's simulations for one chunk, one calendar month at a time: a month's pool holds only the rows dated
    before it, with labels printed before it (the outer stage's rule)."""
    sims, months = {}, events["event_date"].dt.to_period("M")
    for period in sorted(set(months)):
        start_m = period.start_time
        pool = build_pool(table[table["event_date"] < start_m], _printed(labels, exits, start_m.date().isoformat()))
        sims.update(shortvol.simulate(chunk, events[months == period], pool))
    return sims


def _inner(src, out, ready, plan, ids, facts_all):
    labels = readers.read_labels(src.fetch, src.admitted, start=INNER[0], end=INNER[1], event_ids=ids)
    exits = _exit_map(src.plan_frame)
    items = {mid: [] for mid in ready}
    quarters = pd.to_datetime(plan["event_date"]).dt.to_period("Q").nunique()
    kept, started, skipped = 0, time.monotonic(), {}
    for number, (label, chunk) in enumerate(ladder.ladder_chunks(src.admitted, plan, src.load_index), start=1):
        facts = _chunk_facts(facts_all, chunk)
        kept += len(chunk["event_id"])
        skipped = _tally(skipped, chunk)
        for mid, entry in ready.items():
            events = model_events(entry["table"], facts, chunk)
            items[mid].append((chunk, events, _sims(chunk, events, entry["table"], labels, exits)))
        elapsed = max(time.monotonic() - started, 1e-9)
        _log(f"inner quarter {number}/{quarters} {label} events={len(chunk['event_id'])} "
             f"rate={kept / elapsed:.2f}/s eta={(quarters - number) * elapsed / number:.0f}s rss={_rss_mb():.0f}MB")
    hashes = {}
    for i, mid in enumerate(ready, start=1):
        started = time.monotonic()
        body = tuning.tune(items[mid], mid)
        hashes[mid] = tuning.write_receipt(out / "receipts" / f"tuning_{mid}.json", body)["content_hash"]
        _log(f"tuning model {i}/{len(ready)} {mid} elapsed={time.monotonic() - started:.1f}s "
             f"c={body['c']} skipped={body['skipped']}")
    return hashes, kept, skipped


def _model_state(c):
    edges = pd.DataFrame({"event_date": pd.Series([], dtype="datetime64[ns]"), "edge": pd.Series([], dtype=float)})
    return {"c": float(c), "scored": {"as_is": [], "retuned": []}, "edges": edges}


def _rows(mid, book, arm, dec, taken, events):
    """Decision rows of one (model, book, arm), with the marks of the taken rows and the model's forecast
    (pred_move, pred_crush) of the month's events merged on event_id."""
    take = [c for c in [*_MARKS, "entry_cost", "pnl_per_share"] if c not in dec.columns]
    fc = events[["event_id", "pred_move", "pred_crush"]].assign(event_id=events["event_id"].astype(str))
    frame = dec.merge(taken[["event_id", *take]], on="event_id", how="left").merge(fc, on="event_id", how="left")
    if "structure" not in frame.columns:
        frame = frame.assign(structure=None)
    return frame.assign(model_id=mid, book=book, arm_or_rule=arm)


def _month(mid, table, events, chunk, labels, exits, start_m, st):
    """Steps 7a-7d for one model and one calendar month. Warm-up months only add history."""
    output = start_m >= pd.Timestamp(OUTER[0])
    cutoff = start_m.date().isoformat()
    pool = build_pool(table[table["event_date"] < start_m], _printed(labels, exits, cutoff))
    sims = shortvol.simulate(chunk, events, pool)
    ids = set(events["event_id"].astype(str))
    frames = []
    for arm, c in (("as_is", 1.0), ("retuned", st["c"])):
        for name in shortvol.NAMES:
            st["scored"][arm].append(shortvol.score_events(chunk, events, sims, name, alpha=ALPHA, c=c))
            if not output:
                continue
            decided = shortvol.decide(pd.concat(st["scored"][arm], ignore_index=True))
            mine = decided[decided["event_id"].astype(str).isin(ids) & (decided["structure"] == name)]
            taken = shortvol.score(chunk, events, mine, name, ALPHA, c=c)
            frames.append(_rows(mid, "short_vol", arm, mine, taken, events))
    first = None
    for rule in straddle.RULES:
        dec = straddle.decide(events, st["edges"], rule, chunk)
        first = dec if first is None else first
        if output:
            frames.append(_rows(mid, "straddle", rule, dec, straddle.score(chunk, events, dec, ALPHA), events))
    st["edges"] = pd.concat([st["edges"], first[["event_date", "edge"]]], ignore_index=True)
    return frames


def _unique_tmp(out):
    """A fresh, uniquely named ``.tmp`` file in ``out``; the caller unlinks it unless it was promoted."""
    with tempfile.NamedTemporaryFile(dir=out, delete=False, suffix=".tmp") as handle:
        return Path(handle.name)


def _write_decisions(tmp, frames):
    """Write the decision rows to ``tmp`` and fsync it. Returns (sha256 of the tmp bytes, rows)."""
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_FRONT)
    frame = frame[_FRONT + [c for c in frame.columns if c not in _FRONT]]
    frame.to_parquet(tmp, index=False)
    digest = hashlib.sha256()
    with open(tmp, "r+b") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""):
            digest.update(block)
        os.fsync(handle.fileno())
    return digest.hexdigest(), len(frame)


def _file_sha256(path):
    """Streamed sha256 of the regular file at ``path``; refuses a symlink, a non-regular file or an oversized one."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise Refusal(_CONFLICT, f"decisions artifact cannot be opened safely: {exc}",
                      {"reason": "decisions_artifact_invalid"}) from exc
    digest = hashlib.sha256()
    with os.fdopen(fd, "rb") as handle:
        st = os.fstat(handle.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_DECISIONS_BYTES:
            raise Refusal(_CONFLICT, "decisions artifact must be a regular file of at most MAX_DECISIONS_BYTES",
                          {"reason": "decisions_artifact_invalid"})
        total = 0
        for block in iter(lambda: handle.read(min(_HASH_BLOCK, MAX_DECISIONS_BYTES + 1 - total)), b""):
            total += len(block)
            if total > MAX_DECISIONS_BYTES:
                raise Refusal(_CONFLICT, "decisions artifact must be a regular file of at most MAX_DECISIONS_BYTES",
                              {"reason": "decisions_artifact_invalid"})
            digest.update(block)
    return digest.hexdigest()


def _unbound_outer(roster, tables, ready):
    """Refuses a non-production model with no table rows in the outer year (its 2025 predictions are unbound)."""
    sources = {m["model_id"]: m["source"] for m in roster["models"]}
    for mid in ready:
        dates = pd.to_datetime(tables[mid]["table"]["event_date"])
        in_year = (dates >= pd.Timestamp(OUTER[0])) & (dates < pd.Timestamp(OUTER[1]))
        if sources[mid] != "production" and not in_year.any():
            raise Refusal(_CONFLICT, "v14 2025 refit predictions are not bound for a non-production model",
                          {"reason": "outer_predictions_unbound"})


def _outer(src, out, ready, receipts, plan, ids, facts_all):
    labels = readers.read_labels(src.fetch, src.admitted, start=INNER[0], end=OUTER[0], event_ids=ids)
    exits = _exit_map(src.plan_frame)
    dates = pd.to_datetime(plan["event_date"])
    quarter, month = dates.dt.to_period("Q").astype(str).to_numpy(), dates.dt.to_period("M").to_numpy()
    total = len(set(month))
    state = {mid: _model_state(receipts[mid]["c"]) for mid in ready}
    frames, done, started, skipped = [], 0, time.monotonic(), {}
    for label, chunk in ladder.ladder_chunks(src.admitted, plan, src.load_index):
        facts = _chunk_facts(facts_all, chunk)
        skipped = _tally(skipped, chunk)
        events = {mid: model_events(entry["table"], facts, chunk) for mid, entry in ready.items()}
        for period in sorted({m for m, q in zip(month, quarter) if q == label}):
            start_m = period.start_time
            month_frames = []
            for mid, entry in ready.items():
                ev = events[mid]
                at = ev[ev["event_date"].dt.to_period("M") == period]
                month_frames += _month(mid, entry["table"], at, chunk, labels, exits, start_m, state[mid])
            frames += month_frames
            done += 1
            rate = done / max(time.monotonic() - started, 1e-9)
            skips = pd.concat(month_frames)["reason"].value_counts().to_dict() if month_frames else {}
            _log(f"outer month {period} {done}/{total} rate={rate:.2f}/s eta={(total - done) / rate:.0f}s "
                 f"skips={skips} rss={_rss_mb():.0f}MB")
            if start_m >= pd.Timestamp(OUTER[0]):  # R6: only after this month's rows are complete
                fresh = readers.read_labels(src.fetch, src.admitted, start=start_m,
                                            end=(period + 1).start_time, event_ids=ids)
                labels = pd.concat([labels, fresh])
    final, tmp = out / "decisions.parquet", _unique_tmp(out)
    try:
        sha, rows = _write_decisions(tmp, frames)
        # The receipt refuses a changed replay before the final file is touched.
        tuning.write_receipt(out / "receipts" / "outer.json",
                             {"stage": "outer", "decisions_sha256": sha, "rows": rows, "models": list(ready),
                              "ladder_skipped": skipped})
        if final.exists() and _file_sha256(final) == sha:  # exact replay: keep the existing file untouched
            tmp.unlink()
        else:
            os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)  # no-op after a successful replace
    return rows, skipped


def run(src, stage, out_dir, *, event_ids=None, prereg=size_pnl_prereg.require_preregistration, tuned=None):
    """Run one stage, write its receipts under ``out_dir``, and return a JSON-safe summary.

    ``tuned`` is reserved: the outer stage reads the tuning receipts from disk, so it must be None.
    """
    started = time.monotonic()
    receipt = prereg()
    if tuned is not None:
        raise ValueError("tuned is reserved; the outer stage reads the tuning receipts from disk")
    _log(f"stage={stage} start rss={_rss_mb():.0f}MB")
    out = Path(out_dir)
    ids = _smoke(src, event_ids)
    if ids is not None and not ids:
        raise Refusal(_CONFLICT, "the smoke event id set is empty", {"reason": "empty_population"})
    roster, bound_tables, production = prepare(src, stage)
    tables = bound_tables["tables"]
    ready = {mid: {**e, "table": _restrict(e["table"], ids)} for mid, e in tables.items() if e["status"] == "ready"}
    excluded = {mid: e["reason"] for mid, e in tables.items() if e["status"] != "ready"}
    scope = None if ids is None else content_hash(sorted(ids))
    plan = _plan_window(src.plan_frame, _window(stage), ids)
    admitted_ids = set(src.admitted["events"]["event_id"].astype(str))
    if any(str(e) not in admitted_ids for e in plan["event_id"]):
        raise fail(_DENIED, "plan event ids include a key that is not admitted", details={"reason": "key_not_admitted"})
    facts_all =src.facts(sorted(str(e) for e in plan["event_id"]))  # the only facts call of the run
    facts_rows = sorted(([str(e), *[None if pd.isna(v) else float(v) for v in vals]]
                         for e, vals in zip(facts_all.index, facts_all[list(_FACTS)].to_numpy(dtype=float))),
                        key=lambda r: r[0])
    body = {"stage": "inputs", "spec_hash": receipt["spec_hash"], "identity": bound_tables["identity"],
            "roster": roster, "models": _strip(bound_tables["bound"]),
            "tables": {mid: e["binding"] for mid, e in tables.items() if e["status"] == "ready"},
            "production": production["binding"], "scope": scope, "plan": content_hash(_plan_rows(plan)),
            "facts": content_hash(facts_rows), "snapshot": src.snapshot_id}
    inputs_receipt = tuning.write_receipt(out / "receipts" / f"inputs_{stage}.json", body)
    if not ready:
        raise Refusal(_CONFLICT, "no model is ready to be tuned or scored", {"reason": "no_ready_models"})
    receipts = _outer_gate(out, ready, scope, body) if stage == "outer" else {}
    if stage == "outer":
        _unbound_outer(roster, tables, ready)
    if plan.empty:
        raise Refusal(_CONFLICT, "the plan window holds no events", {"reason": "empty_population"})
    if stage == "inner":
        hashes, rows, skipped = _inner(src, out, ready, plan, ids, facts_all)
        tuning.write_receipt(out / "receipts" / "inner.json",
                             {"stage": "inner", "models": hashes, "events": rows, "excluded": excluded,
                              "scope": scope, "inputs": inputs_receipt["content_hash"], "ladder_skipped": skipped})
    else:
        rows, skipped = _outer(src, out, ready, receipts, plan, ids, facts_all)
    _log(f"stage={stage} end elapsed={time.monotonic() - started:.1f}s rss={_rss_mb():.0f}MB "
         f"ladder_skipped={skipped}")
    return {"stage": stage, "models": sorted(ready), "excluded": excluded, "rows": rows, "ladder_skipped": skipped}
