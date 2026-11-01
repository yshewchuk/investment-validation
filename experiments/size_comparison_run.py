"""Zero-retry comparison journal on the existing search-owned catalog and worker."""
import math
from statistics import mean

from engine.models.no_fit import forbid_fitting as forbid_legacy_fitting
from engine.v2.foundation import canonical_json, content_hash
from engine.v2.models.no_fit import forbid_fitting
from engine.v2.ops.errors import fail
from engine.v2.ops.migrations import require_current
from experiments.magnitude_search_execution import run_fitted_trial
from experiments.magnitude_search_inputs import load_inputs, publish_inputs
from experiments.magnitude_search_records import MIGRATIONS, OWNER, append_record, read_record
from experiments.size_comparator import COLUMNS, _values, fit_evidence, recipe
from experiments.size_job_budget import resolved_limits
from experiments.size_search_binding import fold_inputs, prepare_plan


def _equal(a, b):
    if canonical_json(a) != canonical_json(b):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "comparison journal differs from admitted inputs")


def _validate(result, inputs, fold, plan):
    """Reconstruct coverage, abstentions and exact admitted component-fit accounting."""
    indexed = {(r["event_id"], r["horizon_sessions"]): r for r in inputs["features"]}
    expected, details = [], []
    for h in plan["horizons"]:
        training = [r["event_id"] for r in inputs["training"] if 0 <= r["absolute_move_pct"] <= 200
                    and _values(indexed[r["event_id"], h])[0] is not None]
        training_abstention_counts = {}
        for r in inputs["training"]:
            if 0 <= r["absolute_move_pct"] <= 200:
                reason = _values(indexed[r["event_id"], h])[1]
                if reason is not None:
                    training_abstention_counts[reason] = training_abstention_counts.get(reason, 0) + 1
        fitted = len(training) >= 500
        abstention_counts = {}
        fallback = mean(r["absolute_move_pct"] for r in inputs["training"])
        for event in fold["evaluation_event_ids"]:
            row = indexed[event, h]
            reason = _values(row)[1] or ("insufficient_training" if not fitted else None)
            if reason is not None:
                abstention_counts[reason] = abstention_counts.get(reason, 0) + 1
            expected.append({"event_id": event, "horizon_sessions": h, "abstention": reason,
                "history_mean": row["mean_prior_abs_move"] if row["n_prior"] else fallback,
                "quote_status": row["current_quote_status"]})
        details.append({"horizon_sessions": h, "training_event_ids": training,
                        "logical_fits": int(fitted), "component_fits": 2 * int(fitted),
                        "abstention_counts": dict(sorted(abstention_counts.items())),
                        "training_abstention_counts": dict(sorted(training_abstention_counts.items()))})
    _equal(sorted(result) if isinstance(result, dict) else None, sorted(["event", "forecasts", "fit_details"]))
    _equal(result["event"], "succeeded")
    _equal(result["fit_details"], details)
    expected.sort(key=lambda r: (r["event_id"], r["horizon_sessions"]))
    _equal(type(result["forecasts"]) is list, True)
    _equal(len(result["forecasts"]), len(expected))
    for row, wanted in zip(result["forecasts"], expected):
        _equal(sorted(row) if isinstance(row, dict) else None, sorted([*wanted, "predicted_absolute_move_pct"]))
        _equal({k: v for k, v in row.items() if k != "predicted_absolute_move_pct"}, wanted)
        value = row["predicted_absolute_move_pct"]
        if (wanted["abstention"] is not None and value is not None) or (wanted["abstention"] is None
                and (type(value) not in (int, float) or not math.isfinite(value))):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "invalid comparison forecast")


def execute(conn, store, repository, registration, plan, fold, *, limits, replay_only=False):
    """Freeze comparator/sanity forecasts before the selected outer target opens."""
    _equal(limits, plan["request"]["limits"]["child"])
    regression = plan.get("schema_version") == "regression_search_plan.v1"
    validate, evidence, admit = _validate, fit_evidence, fold_inputs
    if regression:
        from experiments import iv_crush_comparison_run as iv
        validate, evidence, admit = iv.validate_result, iv.fit_evidence, iv.fold_inputs
        _equal(plan, iv.prepare_plan(conn, store, registration, plan["request"]))
        execution = iv.execution(plan)
    else:
        _equal(plan, prepare_plan(conn, store, registration, plan["request"],
                                  plan_schema=plan.get("schema_version")))
        execution = {"schema_version": "size_comparison_execution.v1", "features": COLUMNS,
                     "fit_recipe": registration.document["size_comparison"]}
        _equal(execution["fit_recipe"], recipe())
    require_current(conn, OWNER, MIGRATIONS)
    seal = read_record(conn, store, registration.run_id, "seal")
    _equal(seal.get("plan_id") if isinstance(seal, dict) else None, plan["plan_id"])
    execution_key = "regression_execution" if regression else "magnitude_execution"
    candidates = [{**group["outer"], "evaluation_labels_before":
        registration.document[execution_key]["evaluation_labels_before"]} for group in plan["folds"]]
    if fold not in candidates:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "comparison requires an exact sealed outer fold")
    key = content_hash([plan["plan_id"], fold["year"], "comparison"])
    def read(slot):
        return read_record(conn, store, registration.run_id, "comparison-" + slot + ":" + key)
    def append(slot, body, **kw):
        return append_record(conn, store, registration.run_id, "comparison-" + slot + ":" + key, body, **kw)
    started = {"plan_id": plan["plan_id"], "fold": fold, "execution": execution, "limits": limits}
    terminal = read("terminal")
    if terminal is None:
        if replay_only or (regression and read("start") is not None):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "comparison is not complete")
        forbid_fitting("size_comparison.execute")
        forbid_legacy_fitting("size_comparison.execute")
        resolved_limits(conn, store, registration, plan, limits)
    if terminal is not None:
        _equal(sorted(terminal) if isinstance(terminal, dict) else None, ["inputs_id", "result"])
        _equal(read("start"), started)
        stored = load_inputs(store, read("inputs"), terminal["inputs_id"])
    inputs = stored if regression and terminal is not None else admit(
        repository, registration.document, fold, **({"plan": plan} if regression else {}))
    if regression:
        _equal(inputs.get("target"), plan["target"])
    expected = evidence(inputs, fold, plan)
    validation_options = {"fits": expected} if regression else {}
    if terminal is not None:
        if not regression:
            _equal(stored, inputs)
        validate(terminal["result"], inputs, fold, plan, **validation_options)
        for fit in expected:
            for state in ("started", "completed"):
                _equal(read(state + ":" + str(fit["horizon_sessions"])), fit)
        return terminal["result"]
    append("start", started, replay=False)
    manifest, inputs_id = publish_inputs(store, inputs)
    append("inputs", manifest)
    completed, pending = [], []
    def progress(state, fit):
        index = len(completed)
        if state not in ("started", "completed") or index >= len(expected) or (state == "started") == bool(pending):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "comparison fit sequence differs")
        _equal(fit, expected[index])
        append(state + ":" + str(fit["horizon_sessions"]), fit, replay=False)
        if state == "started":
            pending.append(fit)
        else:
            completed.append(pending.pop())
    resolved = resolved_limits(conn, store, registration, plan, limits)
    result = run_fitted_trial({"manifest": manifest, "inputs_id": inputs_id, "store_root": str(store.root)},
                              fold, plan, execution, resolved, progress)
    _equal(completed, expected)
    _equal(pending, [])
    validate(result, inputs, fold, plan, **validation_options)
    append("terminal", {"inputs_id": inputs_id, "result": result}, replay=False)
    return result
