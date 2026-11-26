"""EXP-190 Q6a: Stage A driver, the first end-to-end slice (design sections 4, 5, 7, 11). No CLI, loader or caller yet;
not included: arm S, B1, B2, B3, their tables and CRPS, headline loss flags, smoke. `years` is taken from the caller
and is NOT checked against the design's Y0 rule yet (issue #2947); no B0 conditional value or unconditional width
terciles (the raw rows are stored so they can be computed at report time, issue #3042). Writes out_dir/fits and
out_dir/stage_a_report.json, schema size_quantile_stage_a.v1: inputs (bindings, hashes, years, bootstrap), status,
reasons, arm_verdicts, fit_status, selection_period_disclosure, not_included, floor_report, arms (judge_arm results,
no paired_ids), content_hash. Published atomically and immutably; identical existing bytes are a replay.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from engine.v2.foundation import content_hash, ensure_directory, fsync_directory
from engine.v2.ops.errors import fail
from engine.v2.research.progress import Progress, decision, stage
from experiments.size_pnl_contrast import RESAMPLES, SEED
from experiments.size_pnl_prereg import _read_bounded
from experiments.size_quantile_arm_verdict import judge_arm, stage_a_verdict
from experiments.size_quantile_baselines import pool_quantiles
from experiments.size_quantile_calibration import QCOLS, calibrate, predictions_from_receipts, quantile_points
from experiments.size_quantile_metrics import crps_quantile
from experiments.size_quantile_prereg import SPEC_PATH, load_spec, require_preregistration
from experiments.size_quantile_sources import training_labels
from experiments.size_quantile_walk import fit_walk_forward

SCHEMA, REPORT, MAX_REPORT_BYTES = "size_quantile_stage_a.v1", "stage_a_report.json", 512 << 20
PERIOD_START, PERIOD_END = pd.Timestamp("2023-07-01"), pd.Timestamp("2025-12-31")

def _refuse(reason, message):
    return fail("EXPERIMENT_IDENTITY_CONFLICT", f"{reason}: {message}", details={"reason": reason})

def _plain(value):
    return json.loads(json.dumps(value, default=lambda item: item.to_dict("records") if isinstance(item, pd.DataFrame)
                                 else item.tolist()), parse_constant=lambda _: None)

def _score(sealed, walk, production, labels):
    """Step 5, after an all-ok walk: admitted labels, calibration, CRPS and the verdicts."""
    predictions = predictions_from_receipts(walk)
    targets = pd.DataFrame(sealed["context"]["target_events"]).set_index("event_id")
    in_period = [e["event_id"] for e in sealed["events"]
                 if PERIOD_START <= pd.Timestamp(targets.at[e["event_id"], "event_date"]) <= PERIOD_END]
    ids = sorted(set(predictions.index) | set(in_period))
    with stage("stage A inputs"):
        frame = training_labels(sealed, ids, sealed["evaluation_labels_before"])
        y = pd.Series(frame["absolute_move_pct"].to_numpy(float), index=frame["event_id"].astype(str))
        labels_hash = content_hash(sorted(_plain(frame[["event_id", "absolute_move_pct"]].values.tolist())))
        decision("quantile stage A", f"labels requested={len(ids)} qualified={len(frame)} in_period={len(in_period)}")
    # Refuse, never drop: the scored-share denominator is every admitted in-period event, and the metrics need y.
    if missing := [e for e in in_period if not np.isfinite(y.get(e, np.nan))]:
        raise _refuse("admitted_label_missing", f"{len(missing)} in-period admitted events have no qualified label")
    events = targets.loc[ids, ["event_date", "ticker", "label_end"]].assign(
        y=y.reindex(ids).to_numpy(float), pred_move=production["frame"]["pred_move"].reindex(ids).to_numpy(float),
        m0=lambda d: d["pred_move"], m3=np.nan)
    arms, pools, floors = calibrate(predictions, events)
    with stage("stage A CRPS and verdicts"):
        b0 = pool_quantiles(production["frame"], labels, events)
        raw = pd.concat([pd.DataFrame(np.asarray(r["raw"]["{}x{}".format(*r["configuration"])], float),
                                      index=r["event_ids"], columns=QCOLS) for r in walk.values()])
        results = {}
        for arm in ("Q", "Q_raw"):
            af = arms[arm]
            y_arm, q = events["y"].reindex(af.index).to_numpy(float), af[QCOLS].to_numpy(float)
            ok, months = af["reason"].isna().to_numpy() & np.isfinite(y_arm), list(af["month"])
            crps, progress = np.full(len(af), np.nan), Progress(f"stage A CRPS {arm}", len(af))
            for k in np.flatnonzero(ok):
                crps[k] = crps_quantile(y_arm[k], *quantile_points(q[k], *pools[(arm, months[k])]))
                progress.tick()
            history = pd.DataFrame({"q50": q[ok, 3], "y": y_arm[ok], "label_available": (pd.to_datetime(
                events["label_end"].reindex(af.index)[ok]) + pd.Timedelta(days=1)).to_numpy()})
            results[arm] = judge_arm(arm, af.assign(crps=crps), b0, events, history, raw=raw)
        verdict = stage_a_verdict(results["Q"], results["Q_raw"])
    return verdict, results, floors, labels_hash

def _publish(out_dir, body):
    """Atomic and immutable (os.link never overwrites): identical existing bytes are a replay; returns the document."""
    data = json.dumps({**(plain := _plain(body)), "content_hash": content_hash(plain)}, sort_keys=True,
                      allow_nan=False).encode()
    ensure_directory(out_dir)
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, out_dir / REPORT)
        except FileExistsError:
            try:
                if _read_bounded(out_dir / REPORT, MAX_REPORT_BYTES).encode("utf-8") != data:
                    raise ValueError("report bytes differ")
            except (OSError, ValueError):
                raise _refuse("stage_a_report_changed", f"{REPORT} exists and does not match this run") from None
            decision("quantile stage A", f"replay: {REPORT} holds identical bytes; file untouched")
        else:
            fsync_directory(out_dir)
            decision("quantile stage A", f"publish: wrote {REPORT} ({len(data)} bytes)")
    finally:
        os.unlink(tmp)
    return json.loads(data)

def run_stage_a(sealed, years, production, labels, out_dir, *, spec_path=SPEC_PATH, ledger_path=None,
                run_log_path=None, experiments_dir=None):
    """Run Stage A and publish its report; returns the parsed document (schema in the module docstring)."""
    try:
        with stage("quantile stage A"):
            guard = require_preregistration(spec_path=spec_path, ledger_path=ledger_path,
                                            run_log_path=run_log_path, experiments_dir=experiments_dir)
            spec, digest = load_spec(spec_path)
            if digest != guard["spec_hash"]:
                raise _refuse("spec_changed", "the spec changed after its pre-registration check")
            if not (isinstance(years, (list, tuple)) and all(isinstance(y, str) for y in years)
                    and {"2023", "2024", "2025"} <= set(years) and years[-1] == "2025"):
                raise fail("INVALID_EXPERIMENT_SPEC", "years invalid", details={"reason": "stage_a_years_invalid"})
            walk = fit_walk_forward(sealed, spec, years, Path(out_dir) / "fits")
            scoring = all(receipt["status"] == "ok" for receipt in walk.values())
            decision("quantile stage A", "branch=scoring" if scoring else "branch=fit_failed: no arm scored")
            verdict, results, floors, eval_hash = (_score(sealed, walk, production, labels) if scoring
                                                   else (stage_a_verdict(None, None), {}, [], None))
            b0_hash = content_hash(sorted(_plain(labels[["abs_move", "crush_pct_iv30"]].reset_index().values.tolist())))
            inputs = {"sealed_binding": sealed["binding"], "spec_hash": guard["spec_hash"],
                      "preregistered_at": guard["preregistered_at"], "years": list(years),
                      "sklearn_version": sklearn.__version__, "production_binding": production["binding"],
                      "production_table_hash": production["table_hash"], "b0_labels_hash": b0_hash,
                      "evaluation_labels_hash": eval_hash,
                      "evaluation_labels_before": sealed["evaluation_labels_before"],
                      "fit_receipt_hashes": {year: r["content_hash"] for year, r in walk.items()},
                      "bootstrap": {"resamples": RESAMPLES, "seed": SEED}}
            # stage_b_arms is deliberately not stored: which arms enter Stage B is an open decision (issue #3042).
            body = {"schema_version": SCHEMA, "inputs": inputs, "status": verdict["status"],
                    "reasons": verdict["reasons"], "arm_verdicts": verdict["arms"],
                    "fit_status": {year: r["status"] for year, r in walk.items()},
                    "selection_period_disclosure": spec["stage_a"]["selection_period_disclosure"],
                    "not_included": ["y0_check", "S", "B1", "B2", "B3", "baseline_crps", "headline_loss_flags",
                                     "b0_conditional_value", "unconditional_width_terciles"],
                    "floor_report": floors,
                    "arms": {arm: {**{k: v for k, v in res.items() if k != "paired_ids"}, "conditional": {
                        **res["conditional"], "rows": res["conditional"]["rows"].rename_axis("event_id").reset_index()}}
                        for arm, res in results.items()}}
            document = _publish(Path(out_dir), body)
            decision("quantile stage A", f"final status={document['status']} reasons={document['reasons']}")
            return document
    except MemoryError:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "stage A ran out of memory") from None
