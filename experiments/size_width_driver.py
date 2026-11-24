"""EXP-189 driver: the width sweep's inner (2023H2-2024) and outer (2025) stages over EXP-188's inputs (design sections 3-6).

Every sweep is built on ``PRICING``, the one pricing seam. The inner stage holds every quarter chunk in memory (W3's contract:
the fake and the anchor re-use the months). The outer stage freezes each 2025 month before its labels are read (R6)."""
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments import size_pnl_decisions as decisions
from experiments import size_pnl_driver as base
from experiments import size_pnl_readers as readers
from experiments import size_pnl_receipts as receipts
from experiments import size_pnl_shortvol
from experiments import size_width_errmodel as errmodel
from experiments import size_width_fake as fake
from experiments import size_width_prereg
from experiments import size_width_tuning as tuning
from experiments.size_pnl_pool import build_pool
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_sweep import RET_COLS, WINDOWS, Sweep
from experiments.size_width_success import LADDER, breakeven_alpha

PRICING = size_pnl_shortvol
_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"


def _log(message):
    print(f"[width_driver] {message}", file=sys.stderr)


def _progress(tag, done, total, started):
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = done / elapsed * 60
    eta = (total - done) * elapsed / done
    _log(f"{tag} month {done}/{total} rate={rate:.2f}/min eta={eta:.0f}s rss={base._rss_mb():.0f}MB")


def _total(plan):
    return int(pd.to_datetime(plan["event_date"]).dt.to_period("M").nunique())


def _period(events):
    return events["event_date"].dt.to_period("M").iloc[0]


def _pool(table, labels, start):
    return build_pool(table[table["event_date"] < start], base._printed(labels, start.date().isoformat()))


def _months(src, plan, table, facts_all, exits, emit_from):
    """(chunk, events, output) per ladder chunk and calendar month of ``plan``; ``output`` is False before ``emit_from``, where a month only adds history."""
    for _label, chunk in base.ladder.ladder_chunks(src.admitted, plan, src.load_index):
        events = base.model_events(table, facts_all, chunk, exits)
        months = events["event_date"].dt.to_period("M")
        for period in sorted(months.unique()):
            yield chunk, events[months == period], bool(period.start_time >= emit_from)


def _mean(rows, col):
    if rows.empty or col not in rows.columns:
        return None
    value = float(pd.to_numeric(rows[col], errors="coerce").mean())
    return value if math.isfinite(value) else None


def _stats(sweep, by_structure, window):
    rows = sweep.rows([(n, by_structure[n]) for n in NAMES], window)
    means = {col: _mean(rows, col) for col in RET_COLS}
    x = None
    if all(means[col] is not None for col in RET_COLS):
        found = breakeven_alpha({a: {"mean": means[col]} for a, col in zip(LADDER, RET_COLS)})
        x = None if found is None else float(found)
    return {"trades": int(len(rows)), "x": x, "mid": means.get("ret_a0.50"), "alpha0": means.get("ret_a0.00")}


def _declined(sweep):
    out = {}
    for (name, arm), reasons in sweep.declined.items():
        declined = sum(reasons.values())
        taken = sum(len(f) for f in sweep.taken.get((name, arm), []))
        out.setdefault(name, {})[str(arm)] = float(declined / (declined + taken)) if declined + taken else 0.0
    return out


def _load(src, stage, ids, receipt):
    """The loaded run: the M0 table on the stage's plan with its facts and labels, and the inputs body (stage-independent but for the table and plan bindings)."""
    roster, bound, _production = base._prepare(src, stage, ids)
    m0 = next((m["model_id"] for m in roster["models"] if "M0" in m["roles"]), None)
    entry = bound["tables"].get(m0)
    if entry is None or entry["status"] != "ready":
        raise Refusal(_CONFLICT, "M0 inputs are missing or failed", {"reason": "m0_inputs_missing"})
    ready = {"M0": {**entry, "table": base._restrict(entry["table"], ids)}}
    full = src.plan_frame
    if full[[c for c in ("event_date", "entry_date", "exit_date") if c in full.columns]].isna().to_numpy().any():
        raise fail("INVALID_EXPERIMENT_SPEC", "plan dates must not be null", details={"reason": "plan_dates_null"})
    base.ladder.check_plan(src.admitted, full)
    plan = base._plan_window(full, base._window(stage), ids)
    if plan.empty:
        raise Refusal(_CONFLICT, "the plan window holds no events", {"reason": "empty_population"})
    inner_window = base._plan_window(full, base.INNER, ids)
    stage_ids = {str(e) for e in plan["event_id"]}
    wanted = (stage_ids | {str(e) for e in inner_window["event_id"]}) & set(ready["M0"]["table"].index.astype(str))
    facts_all = base._load_facts(src, wanted)
    ready = base._drop_facts_missing(src, ready, facts_all)
    labels = readers.read_labels(src.fetch, src.admitted, start=base.INNER[0], end=base.OUTER[0], event_ids=ids)
    unlabelled = base._unlabelled(inner_window, labels)
    _log(f"facts_missing={len(base._facts_missing(facts_all))} unlabelled={len(unlabelled)}")
    ready = base._drop_ids(ready, unlabelled)
    table = ready["M0"]["table"]
    if table.empty:
        raise Refusal(_CONFLICT, "no event survives the exclusions", {"reason": "empty_population"})
    dates = pd.to_datetime(table["event_date"])
    if stage == "outer" and not ((dates >= pd.Timestamp(base.OUTER[0])) & (dates < pd.Timestamp(base.OUTER[1]))).any():
        raise Refusal(_CONFLICT, "2025 predictions are not bound for M0", {"reason": "outer_predictions_unbound"})
    survivors = facts_all[~facts_all.index.astype(str).isin(base._facts_missing(facts_all))]
    inner_ids = {str(e) for e in inner_window["event_id"]}
    inner_table = table[pd.to_datetime(table["event_date"]) < pd.Timestamp(base.OUTER[0])]
    body = {"stage": "inputs", "spec_hash": receipt["spec_hash"], "identity": bound["identity"], "snapshot": src.snapshot_id,
            "crush_table_hash": src.crush_table_hash, "production_hash": src.production_hash,
            "labels": content_hash(base._label_rows([labels])), "scope": base._scope(ids),
            "inner_table": content_hash(sorted([str(i), str(d), float(m), float(c)] for i, d, m, c in zip(inner_table.index, inner_table["event_date"], inner_table["pred_move"], inner_table["pred_crush"]))),
            "inner_plan": content_hash(base._plan_rows(inner_window)), "inner_facts": content_hash(base._facts_rows(survivors, inner_ids))}
    inputs_hash = content_hash(body)
    inputs = {**body, "inputs_hash": inputs_hash, "table": base._table_binding(ready["M0"], ids),
              "plan": content_hash(base._plan_rows(plan)), "facts": content_hash(base._facts_rows(survivors, stage_ids))}
    return {"ids": ids, "plan": plan, "table": table, "labels": labels, "facts_all": facts_all, "sessions": base._sessions(src),
            "exits": base._exits(src), "inputs_hash": inputs_hash, "inputs": inputs}


def _inner(src, out, loaded):
    """The inner stage. It materialises every inner quarter chunk (``months``), which the fake and the anchor re-use (W3's contract)."""
    ids, plan, table, labels = loaded["ids"], loaded["plan"], loaded["table"], loaded["labels"]
    sessions, inputs_hash = loaded["sessions"], loaded["inputs_hash"]
    months = list(_months(src, plan, table, loaded["facts_all"], loaded["exits"], base.INNER_EMIT_FROM))
    _log(f"materialised months={len(months)}")
    sweep, started, total = Sweep(ladder=True, pricing=PRICING), time.monotonic(), _total(plan)
    for done, (chunk, events, output) in enumerate(months, start=1):
        sweep.add_month(chunk, events, _pool(table, labels, _period(events).start_time), output=output, sessions=sessions)
        _progress("inner", done, total, started)
    fit_rows = table[pd.to_datetime(table["event_date"]) < pd.Timestamp(base.OUTER[0])]
    printed = base._printed(labels, base.OUTER[0])
    try:
        model = errmodel.fit(fit_rows["pred_move"].to_numpy(dtype=float), printed["abs_move"].reindex(fit_rows.index).to_numpy(dtype=float))
    except OpsError as exc:
        if ids is None or exc.problem.details.get("reason") != "m0_inputs_insufficient":
            raise
        model, c = None, {"status": "smoke_no_fake", "calibration": None}
    else:
        c = fake.run_fake(months, table, printed, model, sessions, out, inputs_hash, pricing=PRICING)
    _log(f"c_status={c['status']}")
    anchor = {name: {str(k): v for k, v in row.items()} for name, row in fake.run_anchor(months, table, printed, sessions, pricing=PRICING).items()}
    res = tuning.run_inner(sweep, out, inputs_hash)
    results = res["results"]
    by = {"production": {n: 1.0 for n in NAMES}, "A": tuning.factors("A", res["tuned"]["A"]), "B": tuning.factors("B", res["tuned"]["B"]),
          "D": tuning.factors("D", res["tuned"]["D"]), "E": tuning.factors("E", res["tuned"]["E"])}
    if c["status"] == "ok":
        by["C"] = tuning.factors("C", c["body"])
        results["C"] = {w: tuning.evaluate(sweep, by["C"], WINDOWS[w]) for w in ("2023H2", "2024")}
    stats = {label: {w: _stats(sweep, by[label], WINDOWS[w]) for w in ("2023H2", "2024")} for label in by}
    body = {"stage": "inner", "inputs_hash": inputs_hash, "scope": base._scope(ids), "smoke": ids is not None, "c_status": c["status"],
            "model": model, "calibration": c.get("calibration"), "anchor": anchor, "results": results, "stats": stats,
            "declined": _declined(sweep)}
    receipts.write_receipt(out / "receipts" / "inner_summary.json", body)
    return {"stage": "inner", "c_status": c["status"], "arms": sorted(by), "months": len(months)}


def _preflight(out, loaded):
    """The outer stage's preflight: the inner summary, its binding to these inputs and scope, and every arm's frozen factors. Returns (arms, by)."""
    ids, inputs_hash = loaded["ids"], loaded["inputs_hash"]
    inner = receipts.load_receipt(out / "receipts" / "inner_summary.json")
    if inner["inputs_hash"] != inputs_hash or inner["scope"] != base._scope(ids):
        raise Refusal(_CONFLICT, "inner_summary.json is bound to other inputs or another scope", {"reason": "tuning_receipt_changed"})
    arms = [a for a in ("A", "B", "C", "D", "E") if a != "C" or inner["c_status"] == "ok"]
    by = {arm: tuning.frozen(out, arm, inputs_hash) for arm in arms}
    return arms, by


def _outer(src, out, loaded, pre=None):
    """The outer stage. Preflight first: the inner summary and every arm's frozen factors, before any chunk is built. Then each
    2025 month is frozen before its labels are read (R6)."""
    ids, inputs_hash, plan = loaded["ids"], loaded["inputs_hash"], loaded["plan"]
    arms, by = _preflight(out, loaded) if pre is None else pre
    sweep, labels, table, sessions = Sweep(ladder=True, pricing=PRICING), loaded["labels"], loaded["table"], loaded["sessions"]
    total, started, count, fresh_labels = _total(plan), time.monotonic(), 0, []
    for count, (chunk, events, output) in enumerate(_months(src, plan, table, loaded["facts_all"], loaded["exits"], pd.Timestamp(base.OUTER[0])), start=1):
        period = _period(events)
        fresh = sweep.add_month(chunk, events, _pool(table, labels, period.start_time), output=output, sessions=sessions)
        _progress("outer", count, total, started)
        if output:
            taken = fresh.assign(model_id="M0", book="short_vol", arm_or_rule=fresh["arm"], status="taken", reason=None)
            dec = sweep.declined_rows().assign(model_id="M0", book="short_vol", arm_or_rule=lambda f: f["arm"])
            decisions.freeze_month(out, period, [taken, dec])
            got = readers.read_labels(src.fetch, src.admitted, start=period.start_time, end=(period + 1).start_time, event_ids=ids)
            fresh_labels.append(got)
            labels = pd.concat([labels, got])
    results = tuning.run_outer(sweep, out, arms, inputs_hash)
    by = {"production": {n: 1.0 for n in NAMES}, **by}
    stats = {label: {"2025": _stats(sweep, by[label], WINDOWS["2025"])} for label in results}
    body = {"stage": "outer", "inputs_hash": inputs_hash, "scope": base._scope(ids), "smoke": ids is not None, "arms": arms,
            "results": {label: {"2025": results[label]} for label in results}, "stats": stats, "declined": _declined(sweep),
            "inputs_outer": content_hash(loaded["inputs"]), "labels_2025": content_hash(base._label_rows(fresh_labels))}
    receipts.write_receipt(out / "receipts" / "outer_summary.json", body)
    return {"stage": "outer", "arms": arms, "months": count}


def run(src, stage, out_dir, *, event_ids=None, prereg=size_width_prereg.require_preregistration):
    """Run one stage and write its receipts under ``out_dir``; returns a JSON-safe summary. ``event_ids`` is an optional smoke set."""
    started = time.monotonic()
    _log(f"stage={stage} start rss={base._rss_mb():.0f}MB")
    receipt = prereg()
    base.check_sources(src)
    if os.environ.get("BOUNDED_RUN_NESTED") != "1":
        raise fail("RESOURCE_UNAVAILABLE", "run under tools/bounded_run.py --heavy", details={"reason": "not_bounded"})
    if stage not in base.STAGES:
        raise ValueError(f"stage must be one of {base.STAGES}, got {stage!r}")
    out = Path(out_dir)
    ids = base._smoke(src, event_ids)
    _log(f"scope={'smoke' if ids is not None else 'full'} events={'all' if ids is None else len(ids)}")
    loaded = _load(src, stage, ids, receipt)
    pre = _preflight(out, loaded) if stage == "outer" else None
    out.mkdir(parents=True, exist_ok=True)
    base._receipt(out / "receipts" / f"inputs_{stage}.json", loaded["inputs"], "inputs_receipt_changed")
    summary = _inner(src, out, loaded) if stage == "inner" else _outer(src, out, loaded, pre)
    _log(f"stage={stage} end elapsed={time.monotonic() - started:.1f}s rss={base._rss_mb():.0f}MB")
    return summary
