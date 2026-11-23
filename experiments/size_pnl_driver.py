"""EXP-188 S9: driver of the SIZE P&L re-score, inner (S9a, 2023-2024) and outer (S9b, 2025, #2660) stages; the synthetic arm runs through `run_synthetic` (SY2)."""
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from experiments import label_availability, size_pnl_decisions as decisions, size_pnl_inputs as inputs, size_pnl_ladder as ladder, size_pnl_prereg
from experiments import size_pnl_readers as readers, size_pnl_shortvol as shortvol, size_pnl_straddle as straddle, size_pnl_tuning as tuning, size_pnl_equivalence as equivalence, size_pnl_outer_inputs as outer_inputs, size_pnl_gprod as gprod
from experiments.size_pnl_pool import TOLERANCE, build_pool
from experiments.size_pnl_shortvol import Refusal
from experiments import size_pnl_report as report, size_pnl_synthetic as synthetic, size_pnl_synthetic_eval as synthetic_eval

INNER = ("2023-01-01", "2025-01-01")
OUTER = ("2025-01-01", "2026-01-01")
WARMUP_START = "2024-07-01"  # six months of inner history so the trailing cutoff is defined in 2025-01
INNER_EMIT_FROM = pd.Timestamp("2023-07-01")  # six months of 2023 history (burn-in) feed the trailing cutoffs; rows are emitted from 2023H2 through 2024
OUTER_YEAR = 2025  # the roster is resolved for this year
STAGES = ("inner", "outer")
ALPHA = tuning.ALPHA
_DENIED, _CONFLICT = "HOLDOUT_ACCESS_DENIED", "EXPERIMENT_IDENTITY_CONFLICT"
_FACTS = ("pre_iv30", "mcap_usd", "implied_move")
_MARKS = ["exit_value", "rung", "lots", "pnl_dollars"]
# The inputs receipt binds only what is the same in both stages. Production and tables are excluded because their windows
# differ by stage (the production binding hashes the windowed frame), so each stage's receipts bind them instead.
_BOUND = ("spec_hash", "identity", "roster", "models", "snapshot", "crush_table_hash", "labels")  # the pre-2025 labels read is identical in both


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
    crush_table_hash: str = ""
    exclusions: dict = field(default_factory=dict)  # reason -> set of event ids, filled by the table providers
    gate: object = None  # Amendment 3: a size_pnl_gprod.Gate, or None (the pre-registered run)


def check_sources(src):
    if not (isinstance(src, Sources) and isinstance(src.expected, dict) and isinstance(src.exclusions, dict) and {"plan_id", "registration_id", "seal_id", "panel_id"} <= set(src.expected) and isinstance(src.plan_frame, pd.DataFrame)
            and isinstance(src.plan, dict) and {"plan_id", "horizons"} <= set(src.plan) and isinstance(src.plan.get("anchor"), dict) and "run_id" in src.plan["anchor"]
            and all(isinstance(getattr(src, f), dict) for f in ("selection", "pair_candidates", "blend_ids")) and isinstance(src.snapshot_id, str) and src.snapshot_id.strip() and isinstance(src.crush_table_hash, str) and all(callable(getattr(src, f)) for f in ("read", "admit", "fetch", "load_index", "facts"))):
        raise fail("INVALID_EXPERIMENT_SPEC", "sources must be a Sources with an object `expected`", details={"reason": "sources_invalid"})
    return src


def _log(message):
    print(f"[pnl_driver] {message}", file=sys.stderr)


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _content(value):
    if isinstance(value, (np.ndarray, pd.Series, pd.Index)): return pd.util.hash_array(np.asarray(value)).tobytes()
    return repr(sorted(value.items()) if isinstance(value, dict) else value).encode()


def _bind_chunk(digest, label, chunk):
    digest.update(str(label).encode() + b"".join(f"{key}\0".encode() + _content(chunk[key]) for key in sorted(chunk)))  # the consumed quotes, bound


def _window(stage):
    return (WARMUP_START, OUTER[1]) if stage == "outer" else INNER


def _prepare(src, stage, ids=None):
    """Steps 2-4: bind run, roster, models; read production and tables. Returns (roster, bound_tables, production)."""
    check_sources(src)
    if stage not in STAGES: raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    start, end = (INNER[0], OUTER[1]) if stage == "outer" else INNER  # the outer pool needs all history from INNER[0]
    identity = inputs.bind_run(src.plan, src.read, expected=src.expected, panel_id=src.panel_id)
    roster = inputs.resolve_roster(src.plan, src.selection, outer_year=OUTER_YEAR, pair_candidates=src.pair_candidates, blend_ids=src.blend_ids)
    bound = inputs.bind_models(src.plan, src.read, roster, src.admit)
    if stage == "outer": bound = outer_inputs.bind_outer(src.plan, src.read, bound, roster, src.admit, outer_year=OUTER_YEAR)
    production = readers.read_production(src.fetch, src.admitted, expected_hash=src.production_hash, table_hash=src.production_table_hash, start=start, end=end, event_ids=ids)
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


def _ready(tables, ids):
    """The ready entries of a stage's bound tables, each restricted to the smoke ids."""
    return {mid: {**e, "table": _restrict(e["table"], ids)} for mid, e in tables.items() if e["status"] == "ready"}


def _models_body(bound, ready):
    """The inputs receipt's ``models``: a ready deferred model is ``{"status": "ready"}``, else its status, reason and binding."""
    return {mid: ({"status": "ready"} if mid in ready and m["status"] == "deferred"
                  else {k: m[k] for k in ("status", "reason", "binding") if k in m})
            for mid, m in bound.items()}


def _plan_window(plan, window, ids):
    dates = pd.to_datetime(plan["event_date"])
    keep = ((dates >= pd.Timestamp(window[0])) & (dates < pd.Timestamp(window[1]))).to_numpy()
    if ids is not None: keep = keep & plan["event_id"].astype(str).isin(list(ids)).to_numpy()
    return plan[keep]


def _facts_rows(facts, ids):
    """The receipted facts of ``ids``: each event id with its numeric facts, sorted. The stage and inner facts hashes share it."""
    sub = facts[facts.index.astype(str).isin(sorted(ids))]
    return sorted([str(e), *[None if pd.isna(v) else float(v) for v in vals]] for e, vals in zip(sub.index, sub[list(_FACTS)].to_numpy(dtype=float)))


def _scope(ids):
    return None if ids is None else content_hash(sorted(ids))


def _table_binding(entry, ids):
    """A ready model's binding. A smoke run (``ids``), or a table restricted by the facts exclusion, binds the rows it consumed, so its receipt cannot pass for a full one."""
    if ids is None and not entry.get("restricted"): return entry["binding"]
    rows = sorted([str(i), str(d), float(m), float(c)] for i, d, m, c in zip(entry["table"].index, entry["table"]["event_date"], entry["table"]["pred_move"], entry["table"]["pred_crush"]))
    return content_hash({"binding": entry["binding"], "scope": _scope(ids), "rows": rows})


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


def _sessions(src):
    """Event id -> (entry_date, exit_date) as ISO dates, where both are known: the Amendment 2 account dates. Empty without the columns."""
    plan = src.plan_frame
    if not {"entry_date", "exit_date"} <= set(plan.columns): return {}
    return {str(e): (pd.Timestamp(en).date().isoformat(), pd.Timestamp(x).date().isoformat())
            for e, en, x in zip(plan["event_id"], plan["entry_date"], plan["exit_date"]) if pd.notna(en) and pd.notna(x)}


def _common_ids(ready):
    """The ids every ready model's table holds: the paired population E (design section 4). Callers guard an empty ``ready``."""
    return set.intersection(*(set(e["table"].index.astype(str)) for e in ready.values()))


def _facts_missing(facts):
    """Ids (str) with any NaN fact: the events the driver drops from every model table (never imputed)."""
    return set(facts.index[facts[list(_FACTS)].isna().any(axis=1).to_numpy()].astype(str))


def _drop_ids(ready, missing):
    """Drops the ``missing`` ids from every ready model's table. A table that loses no row is passed through as the same object
    (its binding stays the model_tables binding); a table that loses a row is marked ``restricted``."""
    out = {}
    for mid, e in ready.items():
        keep = ~e["table"].index.astype(str).isin(missing)
        out[mid] = e if keep.all() else {**e, "table": e["table"][keep], "restricted": True}
    return out


def _drop_facts_missing(src, ready, facts_all):
    """Drops each event with a NaN fact from every ready model's table, before any binding or receipt; records the ids under
    ``facts_missing`` in ``src.exclusions`` and returns the ready entries."""
    missing = _facts_missing(facts_all)
    src.exclusions.setdefault("facts_missing", set()).update(missing)
    _log(f"facts_missing={len(missing)}")
    return _drop_ids(ready, missing)


def _unlabelled(inner_window, labels):
    """Pre-2025 window ids (str) with no label row. Their ids are already recorded by the fetch's exclusions, so this only drops.
    2025 events cannot be pre-filtered: decisions are frozen before that month's labels are read (R6), so the outer stage's no-simulation drop governs them."""
    return {str(e) for e in inner_window["event_id"]} - set(labels.index.astype(str))


def _pool_tables(ready):
    """Each ready model's table on the ids every ready model has: one population rule for both stages (design section 4)."""
    common = _common_ids(ready)
    return {mid: e["table"][e["table"].index.astype(str).isin(common)] for mid, e in ready.items()}


def _label_rows(frames):
    """Label rows [event_id, abs_move, crush_pct_iv30, available] of the label frames, sorted by event_id (the receipts' shape).
    ``available`` is the frame's ISO availability text: pool membership depends on it, so a changed date changes the receipt."""
    return sorted([[str(e), float(a), float(c), str(v)] for f in frames for e, a, c, v in zip(f.index, f["abs_move"], f["crush_pct_iv30"], f["available"])], key=lambda r: r[0])


def _common_dropped(ready):
    """Ids in some ready model's table but not in every one's: the common-support drop (design section 4)."""
    sets = [set(e["table"].index.astype(str)) for e in ready.values()]
    return len(set.union(*sets) - set.intersection(*sets))


def _printed(labels, cutoff):
    """The labels in the pool for a month starting at ``cutoff`` (an ISO date): a label enters a month's pool only when its
    native ``available_as_of_date`` (the label frame's ``available`` column) is strictly before the month start. The plan's
    exit_date is not used, since it can precede the label's availability."""
    return labels[label_availability.printed_before(labels["available"], cutoff)]


def _tally(total, chunk):
    for reason, n in chunk.get("skipped", {}).items(): total[str(reason)] = total.get(str(reason), 0) + int(n)
    return total


def _sims(chunk, events, table, labels):
    """One model's simulations per calendar month; each month's pool uses only rows and labels known before it."""
    sims, months = {}, events["event_date"].dt.to_period("M")
    for period in sorted(set(months)):
        pool = build_pool(table[table["event_date"] < period.start_time], _printed(labels, period.start_time.date().isoformat()))
        sims.update(shortvol.simulate(chunk, events[months == period], pool))
    return sims


def _equivalence(table, events, sims, chunk, labels):
    """The fast-path check of one model on one chunk, run on the first simulable event's calendar month; None when nothing is simulable."""
    simable = sorted(str(e) for e in events["event_id"] if sims.get(str(e)) is not None)
    if not simable: return None
    first = events[events["event_id"].astype(str) == simable[0]]
    period = pd.Timestamp(first["event_date"].iloc[0]).to_period("M")
    month = events[events["event_date"].dt.to_period("M") == period]
    pool = build_pool(table[table["event_date"] < period.start_time], _printed(labels, period.start_time.date().isoformat()))
    return equivalence.check_chunk(chunk, month, sims, pool, alpha=ALPHA)


def _reason_counts(reasons):
    """{reason: n} of a reason Series, sorted by reason."""
    counts = reasons.value_counts()
    return {str(key): int(counts[key]) for key in sorted(counts.index)}


def _inner(src, out, ready, plan, facts_all, labels):
    exits = _exits(src)
    items, quarters = {mid: [] for mid in ready}, pd.to_datetime(plan["event_date"]).dt.to_period("Q").nunique()
    kept, started, skipped, common, digest = 0, time.monotonic(), {}, {}, hashlib.sha256()
    pool_tables = _pool_tables(ready)
    pending, checked = set(ready), {}
    for number, (label, chunk) in enumerate(ladder.ladder_chunks(src.admitted, plan, src.load_index), start=1):
        _bind_chunk(digest, label, chunk)
        kept +=len(chunk["event_id"])
        skipped = _tally(skipped, chunk)
        built = {}
        for mid, entry in ready.items():
            events = model_events(entry["table"], facts_all, chunk, exits)  # facts_all: the run's only facts call
            built[mid] = (events, _sims(chunk, events, pool_tables[mid], labels))
        for mid in sorted(pending):
            got = _equivalence(pool_tables[mid], *built[mid], chunk, labels)
            if got is None: continue
            checked[mid] = got
            pending.discard(mid)
            _log(f"equivalence model {mid} checked={got['checked']} skipped={got['skipped']} max_abs_diff={got['max_abs_diff']:.3e}")
        # A no-sim event, or an id missing from ANY model's table, leaves for ALL, cutoff history included (design section 4).
        dropped = {e for events, sims in built.values() for e in events["event_id"] if sims.get(e) is None}
        dropped |= {e for events, _s in built.values() for e in set(map(str, chunk["event_id"])) - set(events["event_id"])}
        common[label] = len(dropped)
        for mid, (events, sims) in built.items():
            items[mid].append((chunk, events[~events["event_id"].isin(dropped)], {e: s for e, s in sims.items() if e not in dropped}))
        elapsed = max(time.monotonic() - started, 1e-9)
        _log(f"inner quarter {number}/{quarters} {label} events={len(chunk['event_id'])} " f"rate={kept / elapsed:.2f}/s eta={(quarters - number) * elapsed / number:.0f}s rss={_rss_mb():.0f}MB")
    if pending: raise fail(_CONFLICT, "no simulable sample for the fast-path equivalence check", details={"reason": "equivalence_empty", "models": sorted(pending)})
    nontrades, missing_facts = {}, {}  # per model, counts at c=1.0 only: non-"scored" rows by reason. A non-trade reason (short-vol, P&L 0) stays in every model's and c's items; any other reason is a missing fact, counted apart
    for i, mid in enumerate(ready, start=1):
        started = time.monotonic()
        rows = pd.concat([shortvol.score_events(chunk, events, sims, name, alpha=tuning.ALPHA, c=1.0) for chunk, events, sims in items[mid] for name in shortvol.NAMES], ignore_index=True)
        missed = rows.loc[rows["status"] != "scored"]
        reasons = missed["reason"].fillna(missed["status"]).astype(str)
        is_non_trade = reasons.isin(synthetic_eval.NON_TRADE_REASONS)
        nontrades[mid] = _reason_counts(reasons[is_non_trade])
        missing_facts[mid] = _reason_counts(reasons[~is_non_trade])
        _log(f"non-trade recount model {i}/{len(ready)} {mid} elapsed={time.monotonic() - started:.1f}s rss={_rss_mb():.0f}MB")
    by_model = {mid: sum(len(events) for _c, events, _s in items[mid]) for mid in ready}
    hashes = {}
    for i, mid in enumerate(ready, start=1):
        started = time.monotonic()
        body = tuning.tune(items[mid], mid)
        hashes[mid] = tuning.write_receipt(out / "receipts" / f"tuning_{mid}.json", body)["content_hash"]
        _log(f"tuning model {i}/{len(ready)} {mid} elapsed={time.monotonic() - started:.1f}s " f"c={body['c']} skipped={body['skipped']}")
    return hashes, max(by_model.values(), default=0), skipped, by_model, common, nontrades, missing_facts, "sha256:" + digest.hexdigest(), checked


def _equivalence_ok(entry):
    """A model's fast-path evidence: a dict with an int ``checked`` >= 1 and a numeric ``max_abs_diff`` within TOLERANCE."""
    return (isinstance(entry, dict) and isinstance(entry.get("checked"), int) and not isinstance(entry.get("checked"), bool) and entry["checked"] >= 1
            and isinstance(entry.get("max_abs_diff"), (int, float)) and not isinstance(entry.get("max_abs_diff"), bool) and entry["max_abs_diff"] <= TOLERANCE)


def _outer_gate(out, ready, scope, body, inner_tables, inner_production, inner_plan, inner_facts, inner_models):
    """Refuses before any chunk is built: every ready model needs its tuning receipt, and inner.json must bind those
    receipts, have been tuned on this run's event scope, carry every ready model's fast-path equivalence evidence, and
    bind the inner window's inputs. The inner window's inputs, plan and facts are re-bound here from this run's own
    sources, not trusted from the receipts."""
    receipts = {mid: tuning.load_receipt(out / "receipts" / f"tuning_{mid}.json") for mid in ready}
    inner = tuning.load_receipt(out / "receipts" / "inner.json")
    if any(inner["models"].get(mid) != receipts[mid]["content_hash"] for mid in ready):
        raise Refusal(_CONFLICT, "inner.json does not bind every ready model's tuning receipt", {"reason": "inner_receipt_mismatch"})
    if not (out / "receipts" / "inner_decisions.json").exists() or not (out / "decisions_inner.parquet").exists():
        raise Refusal(_CONFLICT, "inner decisions are missing or not bound to inner.json", {"reason": "inner_decisions_missing"})
    idec = tuning.load_receipt(out / "receipts" / "inner_decisions.json")
    if idec.get("stage") != "inner_decisions" or idec.get("inputs") != inner["inputs"] or idec.get("scope") != inner["scope"]:
        raise Refusal(_CONFLICT, "inner decisions are missing or not bound to inner.json", {"reason": "inner_decisions_mismatch"})
    # A hash-valid receipt can still carry the wrong shapes: refuse them here rather than raise a builtin exception below.
    if not (isinstance(idec.get("models"), list) and isinstance(idec.get("tuning"), dict) and isinstance(idec.get("c"), dict)):
        raise Refusal(_CONFLICT, "inner decisions are missing or not bound to inner.json", {"reason": "inner_decisions_mismatch"})
    # Subset semantics: the inner stage may have tuned models the outer excluded (a missing or partial 2025 refit), so the
    # receipt may cover a superset of the ready models; every ready model must match on tuning and c.
    if not set(ready) <= set(idec.get("models", [])) \
            or any(idec.get("tuning", {}).get(mid) != inner["models"][mid] or idec.get("c", {}).get(mid) != receipts[mid]["c"] for mid in ready) \
            or decisions._file_sha256(out / "decisions_inner.parquet") != idec.get("decisions_sha256"):
        raise Refusal(_CONFLICT, "inner decisions are missing or not bound to inner.json", {"reason": "inner_decisions_mismatch"})
    if "scope" not in inner or inner["scope"] != scope: raise Refusal(_CONFLICT, "inner.json was tuned on a different event scope", {"reason": "scope_mismatch"})
    evidence = inner.get("equivalence")
    if not isinstance(evidence, dict) or not all(_equivalence_ok(evidence.get(mid)) for mid in ready):
        raise Refusal(_CONFLICT, "inner.json lacks the fast-path equivalence evidence for a ready model", {"reason": "inner_equivalence_missing"})
    inputs_inner, now = tuning.load_receipt(out / "receipts" / "inputs_inner.json"), json.loads(json.dumps({**body, "models": inner_models}))
    if any(inputs_inner.get(k) != now[k] for k in _BOUND): raise Refusal(_CONFLICT, "inputs_inner.json was bound to different inputs", {"reason": "inner_inputs_mismatch"})
    if inputs_inner.get("plan") != inner_plan: raise Refusal(_CONFLICT, "inputs_inner.json was bound to a different inner plan", {"reason": "inner_plan_mismatch"})
    if inputs_inner.get("facts") != inner_facts: raise Refusal(_CONFLICT, "inputs_inner.json was bound to different inner facts", {"reason": "inner_facts_mismatch"})
    inner_now = json.loads(json.dumps({"tables": inner_tables, "production": inner_production}))
    if inner["inputs"] != inputs_inner["content_hash"] or any(inputs_inner.get(k) != inner_now[k] for k in ("tables", "production")):
        raise Refusal(_CONFLICT, "inner.json was tuned on different inner tables", {"reason": "inner_tables_mismatch"})
    return receipts


def _no_ready():
    return shortvol.Refusal(_CONFLICT, "no model is ready to be tuned or scored", {"reason": "no_ready_models"})


def _unbound_outer(ready):
    """Refuses a ready model with no table rows in the outer year (its 2025 predictions are not bound yet). Partial coverage is not refused: it is a common-support drop. ``ready`` holds the consumed, smoke-restricted tables."""
    for mid in ready:
        dates = pd.to_datetime(ready[mid]["table"]["event_date"])
        if not ((dates >= pd.Timestamp(OUTER[0])) & (dates < pd.Timestamp(OUTER[1]))).any():
            raise Refusal(_CONFLICT, "2025 predictions are not bound for a ready model with no 2025 rows", {"reason": "outer_predictions_unbound"})


def _model_state(c):
    edges = pd.DataFrame({"event_date": pd.Series([], dtype="datetime64[ns]"), "edge": pd.Series([], dtype=float)})
    return {"c": float(c), "scored": {"as_is": [], "retuned": []}, "edges": edges}


def _rows(mid, book, arm, dec, taken, events, taken_mid=None, secured=None, sessions=None):
    """Decision rows of one (model, book, arm), with the marks of the taken rows, the Amendment 1 sensitivity marks (``exit_value_mid``,
    ``pnl_dollars_mid``: NaN when ``taken_mid`` is None) and the model's forecast (pred_move, pred_crush) merged on event_id.
    Amendment 2 account columns (entry_date, exit_date, secured_per_contract) are added only when ``sessions`` is not None: entry_date
    and exit_date (ISO, from ``sessions``) and secured_per_contract (from ``secured``; 0.0 for the straddle, which
    size_pnl_account.straddle_rows later restates as debit x 100, and NaN for short vol without ``secured``), filled on taken rows only.
    With ``sessions`` None the frame has exactly the pre-Amendment 2 columns, so the synthetic run's decisions carry no account columns."""
    take = [c for c in [*_MARKS, "entry_cost", "pnl_per_share"] if c not in dec.columns]
    fc = events[["event_id", "pred_move", "pred_crush"]].assign(event_id=events["event_id"].astype(str))
    frame = dec.merge(taken[["event_id", *take]], on="event_id", how="left").merge(fc, on="event_id", how="left")
    if taken_mid is None: frame = frame.assign(exit_value_mid=np.nan, pnl_dollars_mid=np.nan)
    else: frame = frame.merge(taken_mid[["event_id", "exit_value", "pnl_dollars"]].rename(columns={"exit_value": "exit_value_mid", "pnl_dollars": "pnl_dollars_mid"}), on="event_id", how="left")
    if "structure" not in frame.columns: frame = frame.assign(structure=None)
    if sessions is None: return frame.assign(model_id=mid, book=book, arm_or_rule=arm)
    if secured is None: frame = frame.assign(secured_per_contract=np.nan)
    else: frame = frame.merge(secured[["event_id", "secured_per_contract"]], on="event_id", how="left")
    sess, ids = sessions, frame["event_id"].astype(str)
    taken_row = (frame["status"] == "taken").to_numpy()
    sec = pd.Series(0.0, index=frame.index) if book == "straddle" else frame["secured_per_contract"]
    frame = frame.assign(entry_date=ids.map(lambda e: sess.get(e, (np.nan, np.nan))[0]).where(taken_row),
                         exit_date=ids.map(lambda e: sess.get(e, (np.nan, np.nan))[1]).where(taken_row),
                         secured_per_contract=sec.where(taken_row))
    return frame.assign(model_id=mid, book=book, arm_or_rule=arm)


def _month_scores(mid, table, events, chunk, labels, start_m, st, *, arms=None):
    """Pass A for one model and calendar month: simulate once and score every (arm, name), with the pool of rows and
    labels known before the month (R6). Returns the scored frames by (arm, name) and the ids with no simulation."""
    pool = build_pool(table[table["event_date"] < start_m], _printed(labels, start_m.date().isoformat()))
    sims = shortvol.simulate(chunk, events, pool)
    scored = {(arm, name): shortvol.score_events(chunk, events, sims, name, alpha=ALPHA, c=c)
              for arm, c in (arms or (("as_is", 1.0), ("retuned", st["c"]))) for name in shortvol.NAMES}
    return scored, {str(e) for e in events["event_id"] if sims.get(e) is None}


def _kept(frame, bad):
    return frame[~frame["event_id"].astype(str).isin(bad)]


def _month(mid, events, scored, bad, chunk, start_m, st, *, arms=None, emit_from=None, sessions=None, gate=None):
    """Pass B for one model and calendar month: drop the ``bad`` ids from the events, the scored frames and the straddle
    inputs, then steps 7a-7d on Pass A's frames. Warm-up months (before OUTER[0]) apply the same mask (they feed history)
    but only add history and emit no rows. ``sessions`` (the Amendment 2 account dates) is None in the synthetic run: then the
    secured-cash lookup is not called and the rows carry no account columns; a dict (possibly empty) keeps them.
    ``gate`` (Amendment 3) is a per-model month decider from size_pnl_gprod.month_decider; it adds the G_PROD straddle rows on output months only; it never touches ``st['edges']`` or the short-vol frames."""
    output = start_m >= (pd.Timestamp(OUTER[0]) if emit_from is None else emit_from)
    events = _kept(events, bad)
    ids, frames = set(events["event_id"].astype(str)), []
    for arm, c in (arms or (("as_is", 1.0), ("retuned", st["c"]))):
        for name in shortvol.NAMES:
            st["scored"][arm].append(_kept(scored[(arm, name)], bad))
            if not output: continue
            decided = shortvol.decide(pd.concat(st["scored"][arm], ignore_index=True))
            mine = decided[decided["event_id"].astype(str).isin(ids) & (decided["structure"] == name)]
            taken_mid = shortvol.score(chunk, events, mine, name, ALPHA, c=c, mid_marks=True)  # Amendment 1 sensitivity marks
            secured = None if sessions is None else shortvol.secured(chunk, events, mine, name, c)
            frames.append(_rows(mid, "short_vol", arm, mine, shortvol.score(chunk, events, mine, name, ALPHA, c=c), events, taken_mid=taken_mid,
                                secured=secured, sessions=sessions))
    first = None
    for rule in straddle.RULES:
        dec = straddle.decide(events, st["edges"], rule, chunk)
        first = dec if first is None else first
        if output: frames.append(_rows(mid, "straddle", rule, dec, straddle.score(chunk, events, dec, ALPHA), events,
                                       taken_mid=straddle.score(chunk, events, dec, ALPHA, mid_marks=True), sessions=sessions))
    if gate is not None and output:
        dec = gate(events, chunk)
        frames.append(_rows(mid, "straddle", straddle.GATE_RULE, dec, straddle.score(chunk, events, dec, ALPHA),
                            events, taken_mid=straddle.score(chunk, events, dec, ALPHA, mid_marks=True), sessions=sessions))
    st["edges"] = pd.concat([st["edges"], first[["event_date", "edge"]]], ignore_index=True)
    return frames


def _outer(src, out, ready, receipts, plan, ids, facts_all, labels):
    """The outer stage: per ladder chunk and calendar month, decide, then read that month's labels (R6). Publishes decisions."""
    gate = src.gate
    exits, pools, sessions = _exits(src), _pool_tables(ready), _sessions(src)
    dates = pd.to_datetime(plan["event_date"])
    quarter, month = dates.dt.to_period("Q").astype(str).to_numpy(), dates.dt.to_period("M").to_numpy()
    total, state = len(set(month)), {mid: _model_state(receipts[mid]["c"]) for mid in ready}
    frames, done, started, skipped, fresh, digest, dropped = [], 0, time.monotonic(), {}, [], hashlib.sha256(), set()
    _log(f"outer start months={total} models={len(ready)} gate={'G_PROD' if gate else 'none'}")
    for label, chunk in ladder.ladder_chunks(src.admitted, plan, src.load_index):
        _bind_chunk(digest, label, chunk)
        skipped = _tally(skipped, chunk)
        events = {mid: model_events(pools[mid], facts_all, chunk, exits) for mid in ready}
        for period in sorted({m for m, q in zip(month, quarter) if q == label}):
            start_m, month_frames = period.start_time, []
            at = {mid: events[mid][events[mid]["event_date"].dt.to_period("M") == period] for mid in ready}
            scored = {mid: _month_scores(mid, pools[mid], at[mid], chunk, labels, start_m, state[mid]) for mid in ready}
            bad = set().union(*(m for _s, m in scored.values()))  # one shared population for all models (design section 4): ids with no simulation
            dropped |= bad
            for mid in ready:
                month_frames += _month(mid, at[mid], scored[mid][0], bad, chunk, start_m, state[mid], sessions=sessions,
                                       gate=None if gate is None else gprod.month_decider(gate, pools[mid], _printed(labels, start_m.date().isoformat()), start_m))
            frames += month_frames
            done += 1
            rate = done / max(time.monotonic() - started, 1e-9)
            skips = pd.concat(month_frames)["reason"].value_counts().to_dict() if month_frames else {}
            _log(f"outer month {period} {done}/{total} rate={rate:.2f}/s eta={(total - done) / rate:.0f}s scoreability_dropped={len(dropped)} skips={skips} rss={_rss_mb():.0f}MB")
            if start_m >= pd.Timestamp(OUTER[0]):  # R6: only after this month's rows are complete and frozen
                decisions.freeze_month(out, period, month_frames)
                got =readers.read_labels(src.fetch, src.admitted, start=start_m, end=(period + 1).start_time, event_ids=ids)
                fresh.append(got)
                labels = pd.concat([labels, got])
    inputs_hash = tuning.load_receipt(out / "receipts" / "inputs_outer.json")["content_hash"]
    body = {"stage": "outer", "models": list(ready), "ladder_skipped": skipped, "scope": _scope(ids), "inputs": inputs_hash,
            "tuning": {mid: receipts[mid]["content_hash"] for mid in ready}, "c": {mid: receipts[mid]["c"] for mid in ready},
            "labels": content_hash(_label_rows(fresh)), "common_dropped": _common_dropped(ready),
            "scoreability_dropped": len(dropped), "ladder": "sha256:" + digest.hexdigest()}
    if gate is not None: body["gate"] = gprod.receipt_body(gate)
    rows = decisions.publish(out, frames, body)
    return rows, skipped


def _inner_decisions(src, out, ready, receipts, plan, facts_all, labels, inputs_hash, scope):
    """Publishes the inner stage's decisions (``decisions_inner.parquet``, receipt ``inner_decisions``) for the pre-registered
    verdict, which reads 2024. A second pass over the inner plan, the same per-chunk, per-month loop as ``_outer``: it costs a
    second simulation of the inner window. Retuned rows use each model's frozen c from its tuning receipt. ``labels`` is read-only."""
    exits, pools = _exits(src), _pool_tables(ready)
    dates = pd.to_datetime(plan["event_date"])
    quarter, month = dates.dt.to_period("Q").astype(str).to_numpy(), dates.dt.to_period("M").to_numpy()
    total, state = len(set(month)), {mid: _model_state(receipts[mid]["c"]) for mid in ready}
    frames, done, started, skipped, digest, dropped = [], 0, time.monotonic(), {}, hashlib.sha256(), set()
    _log(f"inner-decisions start months={total} models={len(ready)}")
    for label, chunk in ladder.ladder_chunks(src.admitted, plan, src.load_index):
        _bind_chunk(digest, label, chunk)
        skipped = _tally(skipped, chunk)
        events = {mid: model_events(pools[mid], facts_all, chunk, exits) for mid in ready}
        for period in sorted({m for m, q in zip(month, quarter) if q == label}):
            start_m, month_frames = period.start_time, []
            at = {mid: events[mid][events[mid]["event_date"].dt.to_period("M") == period] for mid in ready}
            scored = {mid: _month_scores(mid, pools[mid], at[mid], chunk, labels, start_m, state[mid]) for mid in ready}
            bad = set().union(*(m for _s, m in scored.values()))  # one shared population for all models, as in _outer
            dropped |= bad
            for mid in ready:
                month_frames += _month(mid, at[mid], scored[mid][0], bad, chunk, start_m, state[mid], emit_from=INNER_EMIT_FROM)
            frames += [f.assign(in_sample=f["arm_or_rule"] == "retuned") for f in month_frames]  # only the retuned arm uses the chosen c
            done += 1
            rate = done / max(time.monotonic() - started, 1e-9)
            _log(f"inner-decisions month {period} {done}/{total} rate={rate:.2f}/s eta={(total - done) / rate:.0f}s scoreability_dropped={len(dropped)} rss={_rss_mb():.0f}MB")
    body = {"stage": "inner_decisions", "models": list(ready), "scope": scope, "inputs": inputs_hash,
            "tuning": {mid: receipts[mid]["content_hash"] for mid in ready}, "c": {mid: receipts[mid]["c"] for mid in ready},
            "emit_from": INNER_EMIT_FROM.date().isoformat(), "scoreability_dropped": len(dropped),
            "ladder_skipped": skipped, "ladder": "sha256:" + digest.hexdigest()}
    return decisions.publish(out, frames, body, stem="decisions_inner", receipt="inner_decisions")


def run(src, stage, out_dir, *, event_ids=None, prereg=size_pnl_prereg.require_preregistration):
    """Run one stage, write its receipts under ``out_dir``, and return a JSON-safe summary.

    ``event_ids`` is an optional smoke set: every model table, the plan and the labels are restricted to it, and its
    scope is receipted, so a smoke receipt cannot pass for a full-population one.
    """
    started = time.monotonic()
    _log(f"stage={stage} start rss={_rss_mb():.0f}MB")
    receipt = prereg()
    check_sources(src)
    # The supervisor owns heavy jobs: this run must sit under tools/bounded_run.py --heavy.
    if os.environ.get("BOUNDED_RUN_NESTED") != "1":
        raise fail("RESOURCE_UNAVAILABLE", "run under tools/bounded_run.py --heavy", details={"reason": "not_bounded"})
    out, ids = Path(out_dir), _smoke(src, event_ids)
    roster, bound_tables, production = _prepare(src, stage, ids)  # refuses any stage but "inner" or "outer" before reading
    tables = bound_tables["tables"]
    ready = _ready(tables, ids)
    excluded = {mid: e["reason"] for mid, e in tables.items() if e["status"] != "ready"}
    full = src.plan_frame
    if full[[c for c in ("event_date", "entry_date", "exit_date") if c in full.columns]].isna().to_numpy().any(): raise fail("INVALID_EXPERIMENT_SPEC", "plan dates must not be null", details={"reason": "plan_dates_null"})
    ladder.check_plan(src.admitted, full)  # the full plan, before the window can drop any row
    plan = _plan_window(full, _window(stage), ids)
    admitted_dates = dict(zip(src.admitted["events"]["event_id"].astype(str), pd.to_datetime(src.admitted["events"]["event_date"])))
    if any(str(e) not in admitted_dates for e in plan["event_id"]): raise fail(_DENIED, "plan event ids include a key that is not admitted", details={"reason": "key_not_admitted"})
    if plan.empty: raise shortvol.Refusal(_CONFLICT, "the plan window holds no events", {"reason": "empty_population"})  # before any receipt
    inner_window = _plan_window(full, INNER, ids)  # the inner window's plan: the outer gate re-binds its facts and plan
    stage_ids = {str(e) for e in plan["event_id"]}
    wanted = stage_ids | {str(e) for e in inner_window["event_id"]}
    if ready: wanted &= _common_ids(ready)  # facts only for events that survived the earlier exclusions
    facts_all = _load_facts(src, wanted)  # the only facts call of the run: the stage plan's ids and the inner window's
    _log(f"facts_nan_rows={int(facts_all[list(_FACTS)].isna().any(axis=1).sum())}")
    ready = _drop_facts_missing(src, ready, facts_all)  # before any table binding or receipt
    if ready and not _common_ids(ready):
        raise shortvol.Refusal(_CONFLICT, "no event survives the exclusions", {"reason": "empty_population"})
    survivors = facts_all[~facts_all.index.astype(str).isin(_facts_missing(facts_all))]  # the facts receipts cover facts-complete events only
    facts_rows = _facts_rows(survivors, stage_ids)
    labels = readers.read_labels(src.fetch, src.admitted, start=INNER[0], end=OUTER[0], event_ids=ids)  # the pre-2025 labels; outer adds each month's
    ready = _drop_ids(ready, _unlabelled(inner_window, labels))  # before the body: a pre-2025 event with no label leaves every table
    if ready and not _common_ids(ready):
        raise shortvol.Refusal(_CONFLICT, "no event survives the exclusions", {"reason": "empty_population"})
    scope = _scope(ids)
    body = {"stage": "inputs", "spec_hash": receipt["spec_hash"], "identity": bound_tables["identity"],
            "roster": roster,  # inputs-level status per model: arrays are not receipted
            "models": _models_body(bound_tables["bound"], ready),
            "tables": {mid: _table_binding(e, ids) for mid, e in ready.items()},
            "production": production["binding"], "scope": scope, "plan": content_hash(_plan_rows(plan)),
            "facts": content_hash(facts_rows), "labels": content_hash(_label_rows([labels])), "snapshot": src.snapshot_id,
            "crush_table_hash": src.crush_table_hash}
    receipts = {}
    if stage == "outer":  # preflight: a refusal here leaves no outer input receipt, and no chunk is built
        if not ready: raise _no_ready()
        _, inner_bound, inner_production = _prepare(src, "inner", ids)  # the inner window's bindings, to match inputs_inner.json
        inner_ready = _drop_facts_missing(src, _ready(inner_bound["tables"], ids), facts_all)
        inner_ready = _drop_ids(inner_ready, _unlabelled(inner_window, labels))  # the same label drop as the inner run
        inner_tables = {mid: _table_binding(e, ids) for mid, e in inner_ready.items()}
        inner_models = _models_body(inner_bound["bound"], inner_ready)  # inner bindings only: the outer year is not in inputs_inner.json
        inner_plan = content_hash(_plan_rows(inner_window))
        inner_facts = content_hash(_facts_rows(survivors, {str(e) for e in inner_window["event_id"]}))
        receipts = _outer_gate(out, ready, scope, body, inner_tables, inner_production["binding"], inner_plan, inner_facts, inner_models)
        _unbound_outer(ready)
    inputs_receipt = _receipt(out / "receipts" / f"inputs_{stage}.json", body, "inputs_receipt_changed")
    if not ready: raise _no_ready()
    if stage == "inner":
        hashes, rows, skipped, by_model, common, non_trades, missing_facts, ladder_hash, checked = _inner(src, out, ready, plan, facts_all, labels)
        _receipt(out / "receipts" / "inner.json",
                 {"stage": "inner", "models": hashes, "events": rows, "events_by_model": by_model, "ladder": ladder_hash,
                  "excluded": excluded, "scope": scope, "inputs": inputs_receipt["content_hash"],
                  "ladder_skipped": skipped, "common_dropped": common, "non_trades": non_trades, "missing_facts": missing_facts,
                  "equivalence": checked}, "inner_receipt_changed")
        tuned = {mid: tuning.load_receipt(out / "receipts" / f"tuning_{mid}.json") for mid in ready}
        inner_decisions_rows = _inner_decisions(src, out, ready, tuned, plan, facts_all, labels, inputs_receipt["content_hash"], scope)
        _log(f"stage={stage} end elapsed={time.monotonic() - started:.1f}s rss={_rss_mb():.0f}MB "
             f"ladder_skipped={skipped} common_dropped={common} non_trades={non_trades} missing_facts={missing_facts} inner_decisions_rows={inner_decisions_rows}")
    else:
        rows, skipped = _outer(src, out, ready, receipts, plan, ids, facts_all, labels)
        _log(f"stage={stage} end elapsed={time.monotonic() - started:.1f}s rss={_rss_mb():.0f}MB ladder_skipped={skipped}")
    _log(f"funnel admitted={len(src.admitted['events'])} paired={len(_common_ids(ready) & stage_ids)} excluded: "
         + " ".join(f"{reason}={len(ids_)}" for reason, ids_ in sorted(src.exclusions.items())) + f" common_dropped={_common_dropped(ready)}")
    summary = {"stage": stage, "models": sorted(ready), "excluded": excluded, "rows": rows, "ladder_skipped": skipped}
    if stage == "inner": summary["inner_decisions_rows"] = inner_decisions_rows
    return summary


def _load_facts(src, wanted):
    """The one facts call for ``wanted`` ids, refused unless complete, unique, finite and numeric; returns the normalised frame."""
    wanted = set(wanted)
    facts_all = src.facts(sorted(wanted))
    if not isinstance(facts_all, pd.DataFrame) or not facts_all.columns.is_unique: raise fail("INVALID_EXPERIMENT_SPEC", "facts source is incomplete", details={"reason": "facts_incomplete"})
    raw = facts_all.reindex(columns=list(_FACTS)).where(lambda c: c.notna(), np.nan)  # pd.NA and None are NaN
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    if (not set(_FACTS) <= set(facts_all.columns) or set(facts_all.index.astype(str)) != wanted
            or not facts_all.index.astype(str).is_unique or (numeric.isna() & raw.notna()).to_numpy().any() or np.isinf(numeric.to_numpy(dtype=float)).any()): raise fail("INVALID_EXPERIMENT_SPEC", "facts source is incomplete", details={"reason": "facts_incomplete"})
    return pd.concat([facts_all.drop(columns=list(_FACTS)), numeric], axis=1)


_AS_IS = (("as_is", 1.0),)


def _synthetic_scores(src, ready, plan, facts_all, labels):
    """As-is scoring of the synthetic tables over the inner plan, with the static inner labels only (no outer read)."""
    exits, pools = _exits(src), _pool_tables(ready)
    dates = pd.to_datetime(plan["event_date"])
    quarter, month = dates.dt.to_period("Q").astype(str).to_numpy(), dates.dt.to_period("M").to_numpy()
    total, state = len(set(month)), {mid: _model_state(1.0) for mid in ready}
    frames, done, started, skipped, digest, dropped = [], 0, time.monotonic(), {}, hashlib.sha256(), set()
    declined = {mid: {name: [0, 0] for name in shortvol.NAMES} for mid in ready}  # width declines: [declined, candidates] per short-vol unit
    _log(f"synthetic start months={total} models={len(ready)}")
    for label, chunk in ladder.ladder_chunks(src.admitted, plan, src.load_index):
        _bind_chunk(digest, label, chunk)
        skipped = _tally(skipped, chunk)
        events = {mid: model_events(pools[mid], facts_all, chunk, exits) for mid in ready}
        for period in sorted({m for m, q in zip(month, quarter) if q == label}):
            start_m, month_frames = period.start_time, []
            at = {mid: events[mid][events[mid]["event_date"].dt.to_period("M") == period] for mid in ready}
            scored = {mid: _month_scores(mid, pools[mid], at[mid], chunk, labels, start_m, state[mid], arms=_AS_IS) for mid in ready}
            for mid in ready:  # Pass A rows, before the shared mask: only a width_declined row is counted as a width decline (other non-scored reasons are other non-trades or missing facts)
                for name in shortvol.NAMES:
                    f = scored[mid][0][("as_is", name)]
                    declined[mid][name][0] += int(((f["status"] != "scored") & (f["reason"] == "width_declined")).sum())
                    declined[mid][name][1] += len(f)
            bad = set().union(*(m for _s, m in scored.values()))  # one shared population for all models: ids with no simulation
            ids = [set(at[mid]["event_id"].astype(str)) for mid in ready]  # an id missing from any model's month frame leaves for all
            bad |= set().union(*ids) - set.intersection(*ids)
            dropped |= bad
            for mid in ready:
                month_frames += _month(mid, at[mid], scored[mid][0], bad, chunk, start_m, state[mid], arms=_AS_IS, emit_from=pd.Timestamp(INNER[0]))
            frames += month_frames
            done += 1
            rate = done / max(time.monotonic() - started, 1e-9)
            _log(f"synthetic month {period} {done}/{total} rate={rate:.2f}/s eta={(total - done) / rate:.0f}s scoreability_dropped={len(dropped)} rss={_rss_mb():.0f}MB")
    return frames, skipped, "sha256:" + digest.hexdigest(), len(dropped), declined


def _frames_digest(frames):
    frame = pd.concat(frames, ignore_index=True)
    digest = hashlib.sha256(repr(list(zip(frame.columns, frame.dtypes.astype(str)))).encode())  # the schema: names and dtypes
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _fit(table, labels, kept):
    """n, rmse and mae (percent points) of one model's pred_move on the kept ids; the SY1 error is pred minus abs_move."""
    sub = table[table.index.astype(str).isin(kept)]
    truth = pd.Series(labels["abs_move"].to_numpy(dtype=float), index=labels.index.astype(str)).reindex(sub.index.astype(str)).to_numpy(dtype=float)
    err = sub["pred_move"].to_numpy(dtype=float) - truth
    return {"n": len(sub), "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))}


def _replay_equal(out, inputs_hash, ladder_hash, frames_hash, declined_hash):
    """The recorded replay verdict if it was written for this run's inputs, ladder, scored frames and decline counts, else None (X7 not run)."""
    path = out / "receipts" / "synthetic_replay.json"
    if not path.exists(): return None
    loaded = tuning.load_receipt(path)
    ok = loaded["inputs"] == inputs_hash and loaded["ladder"] == ladder_hash and loaded.get("frames") == frames_hash and loaded.get("declined") == declined_hash
    return loaded["equal"] if ok and isinstance(loaded["equal"], bool) else None


def _write_t11(out, text):
    """Publish the T11 report: a per-process tmp file in ``out`` written and fsynced, ``os.replace``, then ``out`` fsynced; the tmp is removed on failure."""
    tmp = out / f".t11_synthetic.md.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, out / "t11_synthetic.md")
        tuning._fsync_dir(out)
    finally:
        tmp.unlink(missing_ok=True)


def run_synthetic(src, out_dir, *, event_ids=None, replay=False, prereg=size_pnl_prereg.require_preregistration):
    """The SY2 synthetic stage on the inner window: score, evaluate X1-X7, publish decisions_synthetic and T11. Never tunes or reads 2025."""
    started = time.monotonic()
    _log(f"stage=synthetic start rss={_rss_mb():.0f}MB")
    if not isinstance(replay, bool): raise fail("INVALID_EXPERIMENT_SPEC", "replay must be a boolean", details={"reason": "replay_invalid"})
    receipt = prereg()
    spec, digest = size_pnl_prereg.load_spec()
    if digest != receipt["spec_hash"]: raise Refusal(_CONFLICT, "the spec differs from the preregistered one", {"reason": "spec_not_preregistered"})
    check_sources(src)
    if os.environ.get("BOUNDED_RUN_NESTED") != "1":
        raise fail("RESOURCE_UNAVAILABLE", "run under tools/bounded_run.py --heavy", details={"reason": "not_bounded"})
    identity = inputs.bind_run(src.plan, src.read, expected=src.expected, panel_id=src.panel_id)  # as _prepare: before any read
    arm = synthetic.arm_from_spec(spec)
    out, ids = Path(out_dir), _smoke(src, event_ids)
    full = src.plan_frame
    if full[[c for c in ("event_date", "entry_date", "exit_date") if c in full.columns]].isna().to_numpy().any(): raise fail("INVALID_EXPERIMENT_SPEC", "plan dates must not be null", details={"reason": "plan_dates_null"})
    ladder.check_plan(src.admitted, full)
    plan = _plan_window(full, INNER, ids)
    if not set(plan["event_id"].astype(str)) <= set(src.admitted["events"]["event_id"].astype(str)): raise fail(_DENIED, "plan event ids include a key that is not admitted", details={"reason": "key_not_admitted"})
    if plan.empty: raise shortvol.Refusal(_CONFLICT, "the plan window holds no events", {"reason": "empty_population"})
    planned = {str(e) for e in plan["event_id"]}  # the reads take only planned events, never an admitted key outside the plan
    stage_ids = {str(e) for e in plan["event_id"]}
    production =readers.read_production(src.fetch, src.admitted, expected_hash=src.production_hash, table_hash=src.production_table_hash, start=INNER[0], end=INNER[1], event_ids=planned)
    labels = readers.read_labels(src.fetch, src.admitted, start=INNER[0], end=INNER[1], event_ids=planned)
    ready = _ready(synthetic.generate(src.admitted, labels, production, arm), ids)
    if not ready: raise _no_ready()
    facts_all = _load_facts(src, stage_ids & _common_ids(ready))  # facts only for events that have a forecast, as in the real run
    ready = _drop_facts_missing(src, ready, facts_all)  # before any table binding, scoring or receipt, as in the real run
    survivors = facts_all[~facts_all.index.astype(str).isin(_facts_missing(facts_all))]  # the facts receipt covers surviving events only
    scope = _scope(ids)
    frames, skipped, ladder_hash, dropped, declined = _synthetic_scores(src, ready, plan, facts_all, labels)
    full = pd.concat(frames, ignore_index=True)
    pop = synthetic_eval.population(full)  # E, the evaluated population; undetermined rows stay published as evidence
    kept, tickers = set(pop["event_id"].astype(str)), dict(zip(plan["event_id"].astype(str), plan["ticker"]))
    if not kept: raise Refusal(_CONFLICT, "the scored population is empty", {"reason": "empty_population"})
    if pd.to_datetime(pop["event_date"]).dt.to_period("Q").nunique() < 2 or len({tickers[e] for e in kept}) < 2:
        raise Refusal(_CONFLICT, "too few quarters or tickers to bootstrap", {"reason": "underclustered_population"})
    body = {"stage": "inputs_synthetic", "spec_hash": receipt["spec_hash"], "identity": identity, "arm": content_hash(arm), "snapshot": src.snapshot_id, "scope": scope,
            "plan": content_hash(_plan_rows(plan)),
            "facts": content_hash(_facts_rows(survivors, stage_ids)), "labels": content_hash(_label_rows([labels])),
            "production": production["binding"], "tables": {mid: _table_binding(e, ids) for mid, e in ready.items()}, "crush_table_hash": src.crush_table_hash}
    inputs_receipt = _receipt(out / "receipts" / "inputs_synthetic.json", body, "inputs_receipt_changed")  # after both population refusals: a refused run pins nothing
    if replay:  # X7 evidence: the replay verdict is receipted, so a later plain run can read it
        again = _synthetic_scores(src, ready, plan, facts_all, labels)
        equal = (_frames_digest(frames) == _frames_digest(again[0]) and declined == again[4] and ladder_hash == again[2])
        _receipt(out / "receipts" / "synthetic_replay.json", {"stage": "synthetic_replay", "scope": scope, "inputs": inputs_receipt["content_hash"],
                                                             "ladder": ladder_hash, "frames": _frames_digest(frames), "declined": content_hash(declined), "equal": equal}, "synthetic_replay_changed")
    replay_equal = _replay_equal(out, inputs_receipt["content_hash"], ladder_hash, _frames_digest(frames), content_hash(declined))
    common, pools = _common_dropped(ready), _pool_tables(ready)
    fits = {mid: _fit(pools[mid], labels, kept) for mid in ready}  # rmse and mae on E
    undetermined = len(set(full["event_id"].astype(str)) - kept)
    no_prediction = {str(e) for e in plan["event_id"]} - set().union(*(set(e["table"].index.astype(str)) for e in ready.values()))  # in no model table
    _log(f"synthetic common_dropped={common} events={len(kept)} undetermined={undetermined} scoreability_dropped={dropped}")
    meta = {mid: {"family": e["family"], "params": e["params"], "rmse": fits[mid]["rmse"], "mae": fits[mid]["mae"]} for mid, e in ready.items()}
    result = synthetic_eval.evaluate(full, meta, tickers, arm, replay_equal=replay_equal, declined=declined)
    body = {"stage": "synthetic", "models": sorted(ready), "scope": scope, "inputs": inputs_receipt["content_hash"], "arm": content_hash(arm),
            "ladder_skipped": skipped, "ladder": ladder_hash, "scoreability_dropped": dropped, "common_dropped": common, "events": len(kept), "declined": declined,
            "events_undetermined": undetermined, "planned": len(plan), "events_no_prediction": len(no_prediction),
            "events_by_model": {mid: len(set(e["table"].index.astype(str)) & kept) for mid, e in ready.items()},
            "evaluation": result["hash"],
            "stop": result["stop"], "metrics": fits}
    rows = decisions.publish(out, frames, body, stem="decisions_synthetic", receipt="synthetic")
    _write_t11(out, report.build_report(report.t11_synthetic(result)))
    _log(f"stage=synthetic end elapsed={time.monotonic() - started:.1f}s rss={_rss_mb():.0f}MB ladder_skipped={skipped} stop={result['stop']} "
         f"planned={len(plan)} events_no_prediction={len(no_prediction)}")
    return {"stage": "synthetic", "models": sorted(ready), "rows": rows, "stop": result["stop"]}


def plan_binding(plan_frame):
    """The outer input receipt's ``plan`` hash for the full (unsmoked) outer window: what ``run`` receipts, so the report can bind its ticker map to it."""
    return content_hash(_plan_rows(_plan_window(plan_frame, _window("outer"), None)))


def inner_plan_binding(plan_frame):
    """The inner input receipt's ``plan`` hash for the full inner window: what the report binds the pooled inner tickers to."""
    return content_hash(_plan_rows(_plan_window(plan_frame, INNER, None)))
