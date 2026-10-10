"""Read-only, fold-local positive-move prior baseline and deterministic reporting."""
import json

import numpy as np
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from engine.v2.contracts.data import SnapshotRef
from engine.v2.foundation import from_document
from engine.v2.ops.errors import fail
from engine.v2.ops.experiment_folds import TrainFoldRule, fit_walk_forward_fold
from engine.v2.ops.experiments import ExperimentSpec, resolve_experiment_plan
from engine.v2.research.prediction_inputs import load_prediction_targets

LOG_LOSS_CLIP = 1e-15
RUNNER = "experiments/v2_prediction.py"


def _years(plan):
    folds = plan.get("folds")
    if (plan.get("runner") != RUNNER or plan.get("price_source") != "computed_moves"
            or type(plan.get("seed")) is not int or not 0 <= plan["seed"] < 2**32
            or plan.get("economic_params") != {} or not isinstance(folds, list) or not folds
            or any(not isinstance(y, str) or len(y) != 4 or not y.isascii()
                   or not y.isdecimal() or int(y) < 1 for y in folds)
            or folds != sorted(set(folds))):
        raise fail("INVALID_EXPERIMENT_SPEC", "unsupported prediction runner, seed, economics, source or year folds")
    return folds


def validate_prediction_spec(spec):
    """Validate consumed declarations before registration or any target read."""
    if not isinstance(spec, ExperimentSpec) or spec.input_files:
        raise fail("INVALID_EXPERIMENT_SPEC", "prediction requires a specification without file inputs")
    plan = resolve_experiment_plan(spec)
    _years(plan.as_document())
    return plan


def _metrics(rows):
    labels = rows["positive_move"].to_numpy(dtype=float)
    scores = rows["score"].to_numpy(dtype=float)
    clipped = np.clip(scores, LOG_LOSS_CLIP, 1 - LOG_LOSS_CLIP)
    reliability, deciles = [], []
    for index in range(10):
        selected = rows[np.minimum((scores * 10).astype(int), 9) == index]
        reliability.append({"bin": index + 1, "lower": index / 10, "upper": (index + 1) / 10,
                            **_summary(selected)})
    ranked = rows.sort_values(["score", "event_id"], kind="stable")
    for index, positions in enumerate(np.array_split(np.arange(len(ranked)), 10), 1):
        selected = ranked.iloc[positions]
        deciles.append({"decile": index, **_summary(selected), "event_ids": selected.event_id.tolist()})
    ece = sum(r["count"] * abs(r["mean_probability"] - r["positive_rate"])
              for r in reliability if r["count"]) / len(rows)
    return {"count": len(rows), "brier": float(np.mean((scores - labels) ** 2)),
            "log_loss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
            "log_loss_clip": LOG_LOSS_CLIP, "ece": ece, "reliability": reliability,
            "rank_deciles": deciles}


def _summary(rows):
    return {"count": len(rows), "mean_probability": float(rows.score.mean()) if len(rows) else None,
            "positive_rate": float(rows.positive_move.mean()) if len(rows) else None}


def prediction_result(repository, registration):
    """Evaluate an admitted registration; no writes, current-head lookup or final-read option."""
    binding = registration.document
    plan = binding["execution_plan"]
    years = _years(plan)
    rows = load_prediction_targets(repository, from_document(SnapshotRef, binding["snapshot"]),
        as_of_month=binding["holdouts"]["holdout_as_of_month"], purpose="selection",
        event_ids=binding["event_ids"]).sort_values("event_id", kind="stable")
    folds, predictions = [], []
    for year in years:
        train = rows[(rows.event_date.dt.year < int(year)) & (rows.target_available_on < year + "-01-01")]
        test = rows[rows.event_date.dt.year == int(year)].copy()
        if train.empty or test.empty or train.positive_move.nunique() != 2:
            raise fail("EXPERIMENT_VARIANT_FAILED", "prediction fold needs test rows and both training classes")
        with threadpool_limits(limits=1):
            fitted = fit_walk_forward_fold(DummyClassifier(strategy="prior", random_state=plan["seed"]),
                np.ones((len(train), 1)), train.positive_move, np.ones((len(test), 1)), TrainFoldRule())
        test["score"] = fitted.test_scores
        test["fold"] = year
        predictions.append(test)
        folds.append({"year": year, "train_count": len(train), "test_count": len(test),
                      "train_event_ids": train.event_id.tolist(), "test_event_ids": test.event_id.tolist(),
                      "target_available_before": year + "-01-01", "threshold": fitted.threshold,
                      "metrics": _metrics(test)})
    import pandas as pd

    scored = pd.concat(predictions, ignore_index=True)
    return {"schema_version": "native_prediction_result.v1.0", "target": "positive_move",
            "model": "DummyClassifier(strategy=prior)", "features": ["constant_intercept"],
            "attempted_variants": 1, "variant_id": registration.variant_id, "run_id": registration.run_id,
            "provenance": binding, "registered_population_count": len(rows),
            "population_labels": sorted(set(rows.population_use)),
            "holdout_results": {"random": "excluded; not evaluated", "rolling": "excluded; not evaluated"},
            "metrics": _metrics(scored), "folds": folds,
            "predictions": scored[["event_id", "fold", "positive_move", "score", "population_use"]]
                .to_dict(orient="records")}


def render_prediction_report(registration, result, *, spec, no_ledger):
    """Render REPORT.md without publishing artifacts or claiming trading returns."""
    binding = registration.document
    if (not isinstance(spec, ExperimentSpec) or spec.spec_hash != binding["spec_hash"]
            or result["variant_id"] != registration.variant_id or result["provenance"] != binding
            or result["run_id"] != registration.run_id):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "report specification differs from registration")
    metrics = result["metrics"]
    lines = ["# Native prediction report", "", spec.hypothesis, "",
             f"Recording mode: {'no-ledger smoke; not recorded' if no_ledger else 'recorded evaluation'}.",
             f"Run: {registration.run_id}; variant: {registration.variant_id}.",
             f"Snapshot: {binding['snapshot']['snapshot_id']}; spec: {binding['spec_hash']}.",
             "Target: positive_move (realized move > 0). Attempted variants: 1.",
             "Baseline: fold-local DummyClassifier prior with a constant intercept; no market features.",
             "Population: " + ", ".join(result["population_labels"]) + ".",
             "Random holdout: excluded; not evaluated. Rolling holdout: excluded; not evaluated.", "",
             f"OOF count: {metrics['count']}; Brier: {metrics['brier']:.8g}; log-loss: {metrics['log_loss']:.8g}; weighted ECE: {metrics['ece']:.8g}.",
             "", "## Folds", "| Year | Train | Test | Brier |", "| --- | ---: | ---: | ---: |"]
    lines.extend(f"| {f['year']} | {f['train_count']} | {f['test_count']} | {f['metrics']['brier']:.8g} |"
                 for f in result["folds"])
    for key, index, title in (("reliability", "bin", "OOF reliability bins"), ("rank_deciles", "decile", "OOF rank deciles")):
        lines += ["", f"## {title}", "| Group | Count | Mean probability | Positive rate |", "| --- | ---: | ---: | ---: |"]
        lines.extend(f"| {r[index]} | {r['count']} | {r['mean_probability']} | {r['positive_rate']} |"
                     for r in metrics[key])
    lines += ["", "## Limitations", "Expanding training uses strictly earlier years and labels available before each test year.",
              "No holdout, market-feature, return, P&L or promotion claims; this is a fixed prior baseline.",
              "Reliability uses [lower, upper) bins, with 1 in the last bin; ECE is count-weighted.",
              "Log-loss clips probabilities to [1e-15, 1-1e-15]. Deciles rank score ascending then canonical event_id; ties are stable.",
              "Ten rank groups split counts evenly, with remainder rows assigned to earlier groups; empty bins have null rates.",
              "", "## Complete metrics and provenance", "```json", json.dumps(result, sort_keys=True, indent=2, allow_nan=False), "```", ""]
    return "\n".join(lines)
