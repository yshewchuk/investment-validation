"""Exploratory reconstructed-history gate evaluation on independently priced arms."""
import io
import json
from collections.abc import Mapping
from contextlib import redirect_stdout
from dataclasses import asdict
from math import fsum

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from engine.v2.contracts.data import SnapshotRef
from engine.v2.foundation import canonical_json, from_document
from engine.v2.ops.errors import fail
from engine.v2.ops.experiment_folds import TrainFoldRule, fit_walk_forward_fold
from engine.v2.ops.experiments import ExperimentSpec
from engine.v2.research._plan import plan_events
from engine.v2.research._pricing import STRUCTURES, trading_calendar_from_snapshot
from engine.v2.research.event_history import load_event_history
from engine.v2.research.experiment_population import load_population
from experiments.v2_gate_variant import _plan, price_variant

RUNNER = "experiments/v2_gate.py"
FEATURES = ("n_prior", "mean_prior_abs_move")


def validate_gate_spec(spec):
    """Only declarations consumed by this fixed recipe reach registration."""
    if not isinstance(spec, ExperimentSpec) or not isinstance(spec.economic_params, Mapping):
        raise fail("INVALID_EXPERIMENT_SPEC", "gate evaluation requires a specification")
    plan = _plan(spec, spec.economic_params.get("strategy"))
    years = list(plan.folds)
    if (plan.runner != RUNNER or plan.seed != 0 or not years
            or any(len(y) != 4 or not y.isascii() or not y.isdecimal() or int(y) < 1 for y in years)
            or years != sorted(set(years))):
        raise fail("INVALID_EXPERIMENT_SPEC", "gate evaluation requires its fixed runner, seed and year folds")
    return plan


def _metadata(repository, snapshot, binding, strategy):
    events = load_population(repository, snapshot, as_of_month=binding["holdouts"]["holdout_as_of_month"],
                             purpose="sweep", event_ids=binding["event_ids"])
    calendar = trading_calendar_from_snapshot(repository, snapshot, extend_days=0)
    rows = plan_events(STRUCTURES[strategy](), events, calendar).frame
    if len(rows) != len(events) or set(rows.event_id) != set(events.event_id):
        raise fail("EXPERIMENT_VARIANT_FAILED", "gate population needs complete calendar plans")
    rows["decision_date"] = pd.to_datetime(rows.decision_date)
    rows["gate_session"] = rows[["decision_date", "entry_date"]].min(axis=1)
    try:
        rows["feature_ceiling"] = [calendar.shift(day, -1, side="exact") for day in rows.gate_session]
    except KeyError:
        raise fail("EXPERIMENT_VARIANT_FAILED", "gate history requires a prior completed session") from None
    rows["population_use"] = rows.event_id.map(events.set_index("event_id").population_use)
    return rows.sort_values("event_id", kind="stable"), calendar


def _features(repository, snapshot, rows, binding, calendar):
    values, inputs = [], []
    for row in rows.itertuples(index=False):
        history = load_event_history(repository, snapshot, event_id=row.event_id,
            as_of_month=binding["holdouts"]["holdout_as_of_month"], purpose="sweep",
            decision_session=row.feature_ceiling.date().isoformat(),
            history_start=calendar.first.date().isoformat())
        features = history["features"]
        count, mean = features["n_prior"], features["mean_prior_abs_move"]
        if count == 0 and mean is None:
            mean = 0.0
        try:
            pair = np.asarray([count, mean], dtype=float)
            if not np.isfinite(pair).all() or (pair < 0).any():
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            raise fail("EXPERIMENT_VARIANT_FAILED", "gate historical features are invalid") from None
        values.append(pair)
        inputs.append({"event_id": row.event_id, "gate_session": row.gate_session.date().isoformat(),
            "feature_ceiling": row.feature_ceiling.date().isoformat(), "model_values": pair.tolist(),
            "history": {key: value for key, value in history.items() if key != "holdout_exclusions"}})
    return pd.DataFrame(values, columns=FEATURES), inputs


def _prices(repository, snapshot, spec, binding, rows):
    with redirect_stdout(io.StringIO()):
        priced = price_variant(repository, snapshot, spec, strategy=spec.economic_params["strategy"],
            as_of_month=binding["holdouts"]["holdout_as_of_month"], event_ids=list(rows.event_id))
    trades = priced.replay.trades.set_index("event_id").loc[list(rows.event_id)].reset_index()
    columns = ["fill_alpha", "entry_cost", "exit_value", "pnl", "ret"]
    if (canonical_json(priced.execution_plan.as_document()) != canonical_json(binding["execution_plan"])
            or not np.isfinite(trades[columns].to_numpy(dtype=float)).all()):
        raise fail("EXPERIMENT_VARIANT_FAILED", "gate replay must be complete and finite")
    return trades, json.loads(canonical_json(asdict(priced.stage_inputs)))


def _summary(rows):
    return {"count": len(rows), "mean_mark_pnl":
            fsum(row["pnl"] / len(rows) for row in rows) if rows else None}


def gate_result(repository, registration, spec):
    """Fit one admitted arm; test quotes are opened only after each fold freezes."""
    plan = validate_gate_spec(spec)
    binding = registration.document
    if "final_read" in binding:
        raise fail("HOLDOUT_ACCESS_DENIED", "gate evaluation has no final-read authority")
    if (spec.spec_hash != binding["spec_hash"] or
            canonical_json(plan.as_document()) != canonical_json(binding["execution_plan"])):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "gate specification differs from registration")
    snapshot = from_document(SnapshotRef, binding["snapshot"])
    rows, calendar = _metadata(repository, snapshot, binding, spec.economic_params["strategy"])
    folds, predictions = [], []
    for year in plan.folds:
        test = rows[rows.gate_session.dt.year == int(year)]
        if test.empty:
            raise fail("EXPERIMENT_VARIANT_FAILED", "gate fold has no evaluation events")
        ceiling = min(pd.Timestamp(year + "-01-01"), test.feature_ceiling.min())
        train = rows[(rows.gate_session.dt.year < int(year)) & (rows.exit_date < ceiling)]
        if train.empty:
            raise fail("EXPERIMENT_VARIANT_FAILED", "gate fold has no closed training events")
        train_x, train_history = _features(repository, snapshot, train, binding, calendar)
        test_x, test_history = _features(repository, snapshot, test, binding, calendar)
        train_prices, train_inputs = _prices(repository, snapshot, spec, binding, train)
        labels = (train_prices.pnl > 0).astype(int)
        if labels.nunique() != 2:
            raise fail("EXPERIMENT_VARIANT_FAILED", "gate fold needs both training outcome classes")
        with threadpool_limits(limits=1):
            fitted = fit_walk_forward_fold(DecisionTreeClassifier(max_depth=1, random_state=0),
                train_x, labels, test_x, TrainFoldRule())
        decisions = tuple((float(score), bool(score >= fitted.threshold)) for score in fitted.test_scores)
        test_prices, test_inputs = _prices(repository, snapshot, spec, binding, test)
        predicted = [{"event_id": row.event_id, "fold": year, "score": score,
            "selected": selected, "pnl": float(row.pnl),
            "entry_cost": float(row.entry_cost), "exit_value": float(row.exit_value),
            "population_use": population} for row, (score, selected), population in
            zip(test_prices.itertuples(index=False), decisions, test.population_use, strict=True)]
        predictions.extend(predicted)
        folds.append({"year": year, "training_exits_before": ceiling.date().isoformat(),
            "threshold": fitted.threshold, "train_inputs": train_inputs, "test_inputs": test_inputs,
            "train_history": train_history, "test_history": test_history,
            "training_cashflows": train_prices[["event_id", "entry_cost", "exit_value", "pnl"]].to_dict("records"),
            "training_exits": [stamp.date().isoformat() for stamp in train.exit_date],
            "training_labels": labels.tolist(), "all_candidates": _summary(predicted),
            "selected": _summary([row for row in predicted if row["selected"]])})
    return {"schema_version": "native_gate_result.v1.0", "variant_id": registration.variant_id,
        "run_id": registration.run_id, "attempted_variants": 1, "provenance": binding,
        "model": "fold-local DecisionTreeClassifier(max_depth=1, random_state=0); training-score median gate",
        "features": list(FEATURES), "empty_history_mean": 0.0,
        "availability": {"basis": "reconstructed_prior_completed_session", "source_publication": "unverified",
                         "event_schedule_as_known": "unverified"},
        "holdout_results": {"random": "excluded; not evaluated", "rolling": "excluded; not evaluated"},
        "all_candidates": _summary(predictions),
        "selected": _summary([row for row in predictions if row["selected"]]),
        "folds": folds, "predictions": predictions}


def render_gate_report(registration, result, *, spec, no_ledger):
    """A deterministic arm report; no winner selection or promotion authority."""
    if (spec.spec_hash != registration.document["spec_hash"] or result["variant_id"] != registration.variant_id
            or result["run_id"] != registration.run_id
            or canonical_json(result["provenance"]) != canonical_json(registration.document)):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "gate report identity differs from registration")
    return "\n".join(["# Exploratory reconstructed-history gate report", "", spec.hypothesis, "",
        f"Recording mode: {'no-ledger smoke; not recorded' if no_ledger else 'recorded evaluation'}.",
        "Attempted variants: 1. Each arm is independently priced from its registered snapshot.",
        "Random holdout: excluded; not evaluated. Rolling holdout: excluded; not evaluated.",
        "Features use the session strictly before decision/entry; training exits precede the fold cutoff.",
        "The model and gate threshold use training rows only; ties at the threshold are selected.",
        "Empty history retains zero count and maps only its missing mean to zero.",
        "P&L uses daily marks; it is not fill-validated. Historical schedule and source availability are unverified.",
        "The fixed replay includes no commissions, financing or portfolio sizing; means are per candidate in each group.",
        "This is exploratory reconstruction, not observed point-in-time, holdout, winner or promotion evidence.",
        "", "## Complete metrics and consumed inputs", "```json",
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False), "```", ""])
