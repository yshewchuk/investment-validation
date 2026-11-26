"""EXP-190 Q3c, arm S (reported only, cannot pass). No production caller (Q6 driver is first); a caller fitting real
data must first pass `size_quantile_prereg.require_preregistration`. Rules: user decisions of 2026-11-26 on issue #2879
(separate design PR); the 0.5 floor is provisional. Monthly outputs are returned, not persisted."""
from pathlib import Path

import numpy as np
import sklearn

from engine.v2.foundation import content_hash
from engine.v2.models.regression_ridge_hgb_recipe import regression_recipe
from engine.v2.models.training.magnitude_ridge_hgb import fit_regression_predictions
from engine.v2.ops.errors import fail
from engine.v2.research.progress import Progress, decision, stage
from experiments import lib
from experiments.label_availability import label_cutoff_reason
from experiments.size_quantile_fits import _label_array, _matrix, arm_id, q_raw
from experiments.size_quantile_folds import annual_folds
from experiments.size_quantile_sources import training_labels
from experiments.size_quantile_walk import _labels, _load, _publish

SCHEMA, SOURCE, SCALE_FLOOR, BURN_IN = "size_quantile_arm_s_fit.v1", "S", 0.5, "scale_burn_in"
COLUMNS = ["median", "scale"]


def _admit(sealed, spec, raw_receipts):
    years, digest = list(raw_receipts), lib.spec_hash(spec)
    consecutive = bool(years) and all(int(b) - int(a) == 1 for a, b in zip(years, years[1:]))
    folds = annual_folds(sealed, years) if consecutive else {}
    if not consecutive or any(
            r["inputs"]["year"] != y or r["inputs"]["sealed_binding"] != sealed["binding"]
            or r["inputs"]["spec_hash"] != digest or list(r["event_ids"]) != folds[y]["evaluation_event_ids"]
            for y, r in raw_receipts.items()):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "raw_receipt_mismatch", details={"reason": "raw_receipt_mismatch"})
    return folds


def _fit(spec, sealed, config, ids, x, y, eval_x, targets):
    fixed, cols = spec["grid"]["fixed"], sealed["model_columns"]
    recipe = regression_recipe(target_domain="signed", model="history_hgb", columns=cols,
                               feature_dictionary_hash=sealed["dictionary_hash"], max_leaf_nodes=config[0],
                               min_samples_leaf=config[1], learning_rate=fixed["learning_rate"],
                               l2_regularization=fixed["l2_regularization"], max_iter=fixed["max_iter"],
                               loss="absolute_error")

    def predict(tx, ty, rows):
        return np.asarray(fit_regression_predictions(recipe, tx, ty, rows, columns=cols)["predictions"], float)

    years = sorted({targets[e]["event_date"][:4] for e in ids})
    progress, held_rows, oof = Progress("arm S inner folds", len(years)), [], []
    for year in years:
        held = [i for i, e in enumerate(ids) if targets[e]["event_date"][:4] == year]
        inner = [i for i, e in enumerate(ids) if targets[e]["event_date"][:4] < year
                 and label_cutoff_reason(targets[e]["label_end"], f"{year}-01-01") is None]
        progress.tick()
        if inner:
            held_rows += held
            oof += predict(x[inner], y[inner], x[held]).tolist()
        else:
            decision("arm S", f"skipped inner year {year}: no label available before {year}-01-01")
    if not held_rows:
        raise ValueError("no out-of-fold residual")
    rows = np.asarray(held_rows)
    decision("arm S", f"out-of-fold rows={len(rows)}")
    scale = predict(x[rows], np.abs(y[rows] - np.asarray(oof)), eval_x)
    return predict(x, y, eval_x), scale


def _months(sealed, spec, evaluation, first, state):
    targets, predicted, by_month = state["targets"], state["predicted"], {}
    minimum, months = spec["calibration"]["min_events"], spec["calibration"]["window_months"]
    for e in evaluation:
        by_month.setdefault(targets[e]["event_date"][:7], []).append(e)
    scored, unscored, used, burned = {}, {}, set(), 0
    for month, events in sorted(by_month.items()):
        start, index = f"{month}-01", int(month[:4]) * 12 + int(month[5:7]) - 1 - months
        lower = f"{index // 12:04d}-{index % 12 + 1:02d}-01"
        donors = [] if first else sorted((targets[e]["event_date"], e) for e in predicted
                                         if lower <= targets[e]["event_date"] < start)
        labelled = _labels(training_labels(sealed, [e for _, e in donors], start)) if len(donors) >= minimum else []
        if len(labelled) < minimum:
            unscored.update(dict.fromkeys(events, BURN_IN))
            burned += 1
            continue
        ids = [e for e, _ in labelled]
        used.update(ids)
        y = _label_array([v for _, v in labelled], "donor_label_invalid")
        scale = np.maximum([predicted[e][1] for e in ids], SCALE_FLOOR)
        z = np.quantile((y - np.asarray([predicted[e][0] for e in ids], float)) / scale, spec["levels"])
        raw = np.asarray([predicted[e][0] + max(predicted[e][1], SCALE_FLOOR) * z for e in events], float)
        q = q_raw(raw)
        scored.update({e: {"raw": raw[i].tolist(), "q": q[i].tolist()} for i, e in enumerate(events)})
    return scored, unscored, used, burned


def _year(sealed, spec, raw_receipts, fold, year, first, state):
    config, cols = tuple(raw_receipts[year]["configuration"]), sealed["model_columns"]
    train = _labels(training_labels(sealed, fold["training_event_ids"], fold["training_labels_before"]))
    evaluation, ids = fold["evaluation_event_ids"], [e for e, _ in train]
    x, eval_x = _matrix(state["by_id"], ids, cols), _matrix(state["by_id"], evaluation, cols)
    y = _label_array([v for _, v in train], "training_label_invalid")
    decision("arm S", f"year={year} cfg={config[0]}x{config[1]} train={len(train)} eval={len(evaluation)}")
    inputs = {"sealed_binding": sealed["binding"], "spec_hash": lib.spec_hash(spec), "year": year,
              "sklearn_version": sklearn.__version__, "training_labels_before": fold["training_labels_before"],
              "training_labels_hash": content_hash([list(p) for p in train]),
              "evaluation_event_ids_hash": content_hash(list(evaluation)),
              "raw_receipt_hash": raw_receipts[year]["content_hash"]}
    fixed = {"schema_version": SCHEMA, "inputs": inputs, "levels": COLUMNS, "configuration": list(config),
             "selection_losses": {}, "arm_id": arm_id(spec, sealed, SOURCE, config), "event_ids": list(evaluation)}
    path = state["directory"] / f"arm_s_fit_{year}.json"
    if path.exists():
        receipt = _load(path, fixed, {SOURCE})
        decision("arm S", f"year={year} replay, no refit")
    else:
        try:
            median, scale = _fit(spec, sealed, config, ids, x, y, eval_x, state["targets"])
            status, failure, raw = "ok", None, {SOURCE: np.column_stack([median, scale]).tolist()}
        except ValueError as exc:
            status, failure, raw = "fit_failed", str(exc)[:200], {}
            decision("arm S", f"year={year} fit_failed: {failure}")
        receipt = _publish(path, {**fixed, "status": status, "failure": failure, "raw": raw}, {SOURCE})
    if receipt["status"] == "ok":
        state["predicted"].update(zip(evaluation, (tuple(row) for row in receipt["raw"][SOURCE])))
    scored, unscored, used, burned = (_months(sealed, spec, evaluation, first, state)
                                      if receipt["status"] == "ok" else ({}, {}, set(), 0))
    predicted = state["predicted"]
    bound = [e for e in scored if predicted[e][1] < SCALE_FLOOR]
    donor_bound = [d for d in used if predicted[d][1] < SCALE_FLOOR]
    floor = {"events": len(scored), "events_bound": len(bound), "donors": len(used), "donors_bound": len(donor_bound)}
    decision("arm S", f"year={year} months burned in={burned} floor_bound events={len(bound)}/{len(scored)} "
             f"donors={len(donor_bound)}/{len(used)} scored={len(scored)} unscored={len(unscored)}")
    return {"status": receipt["status"], "receipt": receipt, "arm_id": receipt["arm_id"], "scored": scored,
            "unscored": unscored, "floor": floor}


def fit_arm_s(sealed, spec, raw_receipts, out_dir):
    """Return {year: outcome} for arm S; each year's fit receipt is replayed or written as arm_s_fit_<year>.json."""
    try:
        folds, years = _admit(sealed, spec, raw_receipts), list(raw_receipts)
        state = {"directory": Path(out_dir), "predicted": {},
                 "targets": {e["event_id"]: e for e in sealed["context"]["target_events"]},
                 "by_id": {r["event_id"]: r for r in sealed["rows"]}}
        results, progress = {}, Progress("arm S years", len(years))
        for year in years:
            with stage(f"arm S {year}"):
                results[year] = _year(sealed, spec, raw_receipts, folds[year], year, year == years[0], state)
            progress.tick()
        return results
    except MemoryError:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "arm S ran out of memory") from None
