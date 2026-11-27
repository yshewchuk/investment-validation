"""EXP-190 Q6b Stage A driver (caller size_quantile_run checks Y0). Q and Q_raw are judged against B0, with CRPS: Q, Q_raw
and B0, and Q4 option B (an in-period event with no label stays in the denominator, unpaired). Not included: S, B1, B2,
B3, headline loss flags, y0 check, B0 conditional value, unconditional width terciles. Years unchecked vs Y0 (#2947).
out_dir/fits, out_dir/stage_a_report.json; size_quantile_stage_a.v3: inputs, status, reasons, arm_verdicts, fit_status,
selection_period_disclosure, not_included, floor_report, arms, crps_exclusions, unlabelled_in_period, content_hash."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from engine.v2.foundation import content_hash, ensure_directory, fsync_directory
from engine.v2.ops.errors import fail
from engine.v2.research.progress import decision, stage
from experiments.size_pnl_contrast import RESAMPLES, SEED
from experiments.size_pnl_prereg import _read_bounded
from experiments.size_pnl_readers import read_labels, read_production
from experiments.size_quantile_arm_verdict import judge_arm, stage_a_verdict
from experiments.size_quantile_baselines import pool_quantiles
from experiments.size_quantile_calibration import MIN_TAIL, PERIOD, QCOLS, calibrate, predictions_from_receipts, score_crps
from experiments.size_quantile_prereg import SPEC_PATH, load_spec, require_preregistration
from experiments.size_quantile_sources import training_labels
from experiments.size_quantile_walk import fit_walk_forward

SCHEMA, REPORT, MAX_REPORT_BYTES = "size_quantile_stage_a.v3", "stage_a_report.json", 512 << 20
NOT_INCLUDED = ["y0_check", "S", "B1", "B2", "B3", "headline_loss_flags", "b0_conditional_value",
                "unconditional_width_terciles"]
FIT_FAILED = {"status": "not_testable", "reasons": ["Q:fit_failed", "Q_raw:fit_failed"],
              "arms": {"Q": "not_testable", "Q_raw": "not_testable"}}

def _refuse(reason, message):
    return fail("EXPERIMENT_IDENTITY_CONFLICT", f"{reason}: {message}", details={"reason": reason})

def _plain(value):
    return json.loads(json.dumps(value, default=lambda item: item.to_dict("records") if isinstance(item, pd.DataFrame)
                                 else item.tolist()), parse_constant=lambda _: None)

def hash_labels(labels):
    return content_hash(sorted(_plain(labels[["abs_move", "crush_pct_iv30"]].reset_index().values.tolist())))

def _score(sealed, walk, production, fetch, admitted):
    """Step 5, after an all-ok walk: admitted labels, calibration, CRPS and the verdicts. Returns (verdict, arms, floor
    rows, CRPS exclusions, labels hash, B0 hash, unlabelled outside the periods, unlabelled in period per year)."""
    predictions = predictions_from_receipts(walk)
    targets = pd.DataFrame(sealed["context"]["target_events"]).set_index("event_id")
    in_period = [e["event_id"] for e in sealed["events"]
                 if PERIOD[0] <= pd.Timestamp(targets.at[e["event_id"], "event_date"]) <= PERIOD[1]]
    ids = sorted(set(predictions.index) | set(in_period))
    with stage("stage A inputs"):
        frame = training_labels(sealed, ids, sealed["evaluation_labels_before"])
        y = pd.Series(frame["absolute_move_pct"].to_numpy(float), index=frame["event_id"].astype(str))
        labels_hash = content_hash(sorted(_plain(frame[["event_id", "absolute_move_pct"]].values.tolist())))
    in_set = set(in_period)  # Q4 option B: an in-period event with no label stays in the denominator, never paired
    unlabelled = [e for e in ids if e not in in_set and not np.isfinite(y.get(e, np.nan))]  # outside the periods
    per_year = {yr: sum(pd.Timestamp(targets.at[e, "event_date"]).year == int(yr) for e in in_period
                        if not np.isfinite(y.get(e, np.nan))) for yr in ("2023", "2024", "2025")}
    decision("quantile stage A", f"labels requested={len(ids)} qualified={len(frame)} in_period={len(in_period)} "
             f"unlabelled={len(unlabelled)} unlabelled_in_period={per_year}")
    events = targets.loc[ids, ["event_date", "ticker", "label_end"]].assign(y=y.reindex(ids).to_numpy(float),
        pred_move=production["frame"]["pred_move"].reindex(ids).to_numpy(float), m0=lambda d: d["pred_move"],
        m3=np.nan)
    arms, pools, floors = calibrate(predictions, events)
    with stage("stage A CRPS and verdicts"):
        labels = read_labels(fetch, admitted)  # after the walk: predictions are written before outcomes are read
        b0_hash = hash_labels(labels)
        raw = pd.concat([pd.DataFrame(np.asarray(r["raw"]["{}x{}".format(*r["configuration"])], float),
                                      index=r["event_ids"], columns=QCOLS) for r in walk.values()])
        frames = {"Q": arms["Q"], "Q_raw": arms["Q_raw"], "B0": pool_quantiles(production["frame"], labels, events)}
        scored, exclusions, extra = score_crps(frames, events, pools)
        results = {}
        for arm in ("Q", "Q_raw"):
            arm_frame = scored[arm]
            ev = events.loc[arm_frame.index]
            ok = arm_frame["reason"].isna().to_numpy() & np.isfinite(ev["y"].to_numpy(float))
            history = pd.DataFrame({"q50": arm_frame["q50"].to_numpy(float)[ok], "y": ev["y"].to_numpy(float)[ok],
                                    "label_available": (pd.to_datetime(ev["label_end"].to_numpy()[ok])
                                                        + pd.Timedelta(days=1)).to_numpy()})
            results[arm] = judge_arm(arm, arm_frame, scored["B0"], events, history, raw=raw)
        verdict = stage_a_verdict(results["Q"], results["Q_raw"])
    return (verdict, results, [*floors.to_dict("records"), *extra], exclusions, labels_hash, b0_hash, unlabelled,
            per_year)

def _publish(out_dir, body, name=REPORT, reason="stage_a_report_changed"):
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
            os.link(tmp, out_dir / name)
        except FileExistsError:
            try:
                if _read_bounded(out_dir / name, MAX_REPORT_BYTES).encode("utf-8") != data:
                    raise ValueError("report bytes differ")
            except (OSError, ValueError):
                raise _refuse(reason, f"{name} exists and does not match this run") from None
            decision("quantile stage A", f"replay: {name} holds identical bytes; file untouched")
        else:
            fsync_directory(out_dir)
            decision("quantile stage A", f"publish: wrote {name} ({len(data)} bytes)")
    finally:
        os.unlink(tmp)
    return json.loads(data)

def run_stage_a(sealed, years, admitted, fetch, out_dir, *, production_hash, table_hash, spec_path=SPEC_PATH,
                ledger_path=None, run_log_path=None, experiments_dir=None, run_inputs_hash=None):
    """Run Stage A and publish its report; returns the parsed document (schema in the module docstring)."""
    try:
        with stage("quantile stage A"):
            guard = require_preregistration(spec_path=spec_path, ledger_path=ledger_path,
                                            run_log_path=run_log_path, experiments_dir=experiments_dir)
            spec, digest = load_spec(spec_path)
            if digest != guard["spec_hash"]:
                raise _refuse("spec_changed", "the spec changed after its pre-registration check")
            if spec["distribution_tails"]["min_donors"] != MIN_TAIL:
                raise _refuse("stage_a_tails_mismatch", f"spec distribution_tails.min_donors is "
                              f"{spec['distribution_tails']['min_donors']} but the code's MIN_TAIL is {MIN_TAIL}")
            if not (isinstance(years, (list, tuple)) and all(isinstance(y, str) for y in years)
                    and {"2023", "2024", "2025"} <= set(years) and years[-1] == "2025"):
                raise fail("INVALID_EXPERIMENT_SPEC", "years invalid", details={"reason": "stage_a_years_invalid"})
            production = read_production(fetch, admitted, expected_hash=production_hash, table_hash=table_hash)
            walk = fit_walk_forward(sealed, spec, years, Path(out_dir) / "fits")
            scoring = all(receipt["status"] == "ok" for receipt in walk.values())
            decision("quantile stage A", "branch=scoring" if scoring else "branch=fit_failed: no arm scored")
            if scoring:
                out = _score(sealed, walk, production, fetch, admitted)
            else:
                out = (FIT_FAILED, {}, [], [], None, None, None, {})
            verdict, results, floors, exclusions, labels_hash, b0_hash, unlabelled, per_year = out
            inputs = {"sealed_binding": sealed["binding"], "spec_hash": guard["spec_hash"],
                      "preregistered_at": guard["preregistered_at"], "years": list(years),
                      "sklearn_version": sklearn.__version__, "production_binding": production["binding"],
                      "production_table_hash": production["table_hash"], "b0_labels_hash": b0_hash,
                      "evaluation_labels_hash": labels_hash, "unlabelled_outside_periods": unlabelled,
                      "evaluation_labels_before": sealed["evaluation_labels_before"],
                      "fit_receipt_hashes": {year: r["content_hash"] for year, r in walk.items()},
                      "bootstrap": {"resamples": RESAMPLES, "seed": SEED}, "run_inputs_hash": run_inputs_hash}
            # stage_b_arms is deliberately not stored: which arms enter Stage B is an open decision (issue #3042).
            body = {"schema_version": SCHEMA, "inputs": inputs, "status": verdict["status"],
                    "reasons": verdict["reasons"], "arm_verdicts": verdict["arms"],
                    "fit_status": {year: r["status"] for year, r in walk.items()},
                    "selection_period_disclosure": spec["stage_a"]["selection_period_disclosure"],
                    "not_included": NOT_INCLUDED, "floor_report": floors,
                    "arms": {arm: {**{k: v for k, v in res.items() if k != "paired_ids"}, "conditional": {
                        **res["conditional"], "rows": res["conditional"]["rows"].rename_axis("event_id").reset_index()}}
                        for arm, res in results.items()},
                    "crps_exclusions": exclusions, "unlabelled_in_period": per_year}
            document = _publish(Path(out_dir), body)
            decision("quantile stage A", f"final status={document['status']} reasons={document['reasons']}")
            return document
    except MemoryError:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "stage A ran out of memory") from None
