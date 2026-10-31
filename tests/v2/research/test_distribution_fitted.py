"""Synthetic admission doubles, real bounded CAS, and tiny authorized paired fits."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from engine.models import no_fit as legacy_guard
from engine.v2.data.errors import DataError
from engine.v2.foundation import ArtifactStore, canonical_json, content_hash
from engine.v2.models import no_fit as native_guard
from engine.v2.ops.errors import OpsError, fail
from engine.v2.research.magnitude_distribution import histogram
from experiments import distribution_fitted as fitted
from experiments.distribution_candidate_fit import fit_candidate as numeric_fit
from experiments.distribution_candidate_recipe import candidate_recipe, trial_identity
from experiments.distribution_class_support import training_support


@pytest.fixture
def case(tmp_path, monkeypatch):
    recipe = candidate_recipe("H")
    columns = recipe["point_recipe"]["features"]
    train, events = ["Tz", "Ta", "Tm", "Tb"], ["Z", "A"]
    labels = [0.0, 0.25, 20.0, 22.0]
    features = [{"event_id": event, "context_id": content_hash("context"), "horizon_sessions": horizon,
                 **{name: (None if index == 1 and event in train else 0.0) for index, name in enumerate(columns)}}
                for event in train + events for horizon in (5, 3, 1)]
    inputs = {"features": features, "training": [{"event_id": event, "absolute_move_pct": y} for event, y in zip(train, labels)]}
    calls, records = [], {("source-run", "seal"): {"untouched": True}}
    state = {"plan_id": content_hash("plan"), "inputs": inputs}

    def context(role, outer, inner):
        calls.append(("context", role, outer, inner))
        fold = {"training_candidate_event_ids": train, "evaluation_event_ids": events,
                "year": inner if role == "inner" else outer, "training_labels_before": "2023-01-01"}
        plan = {"plan_id": state["plan_id"], "recipe": recipe, "context_id": content_hash("context")}
        trials = [trial_identity(recipe, context_id=plan["context_id"], fold_id=content_hash([role, outer, inner, fold]),
            inputs_id=content_hash("inputs"), budget_id=content_hash("budget"), horizon=horizon, role=role) for horizon in (5, 3, 1)]
        admission = {"trials": trials, "plan_id": plan["plan_id"], "manifest": {"actual": "manifest"}}
        return deepcopy({"plan": plan, "native_plan": {"folds": []}, "fold": fold, "admission": admission, "inputs": state["inputs"]})

    trials = SimpleNamespace(conn=object(), store=ArtifactStore(tmp_path / "store"), own=SimpleNamespace(run_id="own-run"),
                             read_fit_context=context)

    def append(conn, store, run, slot, body, *, replay=True):
        calls.append(("append", slot, replay))
        assert run == "own-run"
        if (run, slot) in records:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "already claimed")
        records[run, slot] = deepcopy(body)
        return deepcopy(body)

    monkeypatch.setattr(fitted, "append_record", append)
    monkeypatch.setattr(fitted, "read_record", lambda conn, store, run, slot: deepcopy(records.get((run, slot))))

    def fit(recipe, x, y, prediction, *, columns):
        calls.append(("fit", deepcopy((x, y, prediction, columns))))
        counts = [0] * 81
        for value in y:
            counts[min(int(value * 4), 80)] += 1
        support = training_support(counts)
        masses = [count / len(y) for count in counts]
        trees = support["trees_per_iteration"]
        return {"schema_version": "distribution_candidate_fit.v1", "recipe_id": recipe["recipe_id"],
            "probabilities": [masses.copy() for _ in prediction], "point_predictions": [sum(y) / len(y)] * len(prediction),
            "tail": histogram(masses, tail_excesses=[value - 20 for value in y if value >= 20] or None)["tail"],
            "diagnostics": {"training_support": support, "classifier": {"class_labels": support["observed_class_labels"],
                "iterations": 100, "trees_per_iteration": trees, "total_trees": trees * 100},
                "point": {"iterations_declared": 100, "trees_per_iteration_declared": 1},
                "all_null_training_columns": [columns[1]], "training_rows": len(y), "prediction_rows": len(prediction)}}

    monkeypatch.setattr(fitted, "fit_candidate", fit)
    return SimpleNamespace(fits=fitted.DistributionFits(trials), trials=trials, recipe=recipe, columns=columns,
        state=state, records=records, calls=calls, context=context, events=events, inputs=inputs)


def _error(code, action):
    with pytest.raises(OpsError) as caught:
        action()
    assert caught.value.code == code
    return caught.value


def _receipt(case):
    return next(value for (run, slot), value in case.records.items() if slot.startswith("distribution:fit-result:"))


def _rehash(record):
    record["result_id"] = content_hash({key: value for key, value in record.items() if key != "result_id"})


@pytest.mark.parametrize("guard", [legacy_guard, native_guard])
def test_fit_guards_precede_context_and_all_publication(case, guard):
    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden, match="DistributionFits.fit"):
        case.fits.fit(object(), object(), object(), object())
    assert case.calls == []


@pytest.mark.parametrize("horizon", [5, 3, 1])
@pytest.mark.parametrize("role,inner", [("inner", "2023"), ("outer", None)])
def test_fit_claim_matrix_order_tail_shards_and_exact_replay(case, horizon, role, inner):
    result = case.fits.fit(role, "2024", horizon, inner)
    assert set(result) == {"receipt", "rows", "tail"}
    claim = result["receipt"]["claim"]
    assert claim["trial"]["horizon"] == horizon and claim["trial"]["role"] == role
    assert claim["outer_year"] == "2024" and claim["inner_year"] == inner
    assert set(claim["evidence"]) == {"horizon_sessions", "training_matrix_id", "prediction_matrix_id"}
    assert [row["event_id"] for row in result["rows"]] == ["Z", "A"]
    assert result["rows"][0]["probabilities"] == [0.25, 0.25] + [0.0] * 78 + [0.5]
    assert result["rows"][0]["point_prediction"] == 10.5625
    assert result["tail"] == {"policy": "empirical_excess", "sample_count": 2, "excesses": [0.0, 2.0]}
    assert result["receipt"]["tail"]["excesses"]["row_count"] == 2
    assert result["receipt"]["forecasts"]["row_count"] == 2
    assert "probabilities" not in canonical_json(result["receipt"])
    assert result["receipt"]["status"] == "completed" and result["receipt"]["final_read_authority"] == "none"
    assert [call[0] for call in case.calls] == ["context", "append", "fit", "append"]
    assert case.calls[1][2] is False and case.calls[3][2] is False
    assert case.calls[2][1][1] == [0.0, 0.25, 20.0, 22.0]
    assert case.records["source-run", "seal"] == {"untouched": True}
    assert case.fits.read(role, "2024", horizon, inner) == result
    assert case.fits.fit(role, "2024", horizon, inner) == result
    assert sum(call[0] == "fit" for call in case.calls) == 1
    assert sum(call[0] == "context" for call in case.calls) == 3


def test_caller_mutation_cannot_change_stored_results(case):
    result = case.fits.fit("inner", "2024", 1, "2023")
    expected = deepcopy(result)
    result["rows"][0]["probabilities"].clear()
    result["tail"]["excesses"].append(123)
    result["receipt"]["diagnostics"].clear()
    assert case.fits.read("inner", "2024", 1, "2023") == expected


@pytest.mark.parametrize("horizon", [True, 1.0, "1", None, 2])
def test_invalid_horizon_never_claims(case, horizon):
    _error("INVALID_EXPERIMENT_SPEC", lambda: case.fits.fit("inner", "2024", horizon, "2023"))
    assert [call[0] for call in case.calls] == ["context"]


@pytest.mark.parametrize("stage", ["fit_candidate", "write_section"])
def test_interrupted_fit_cannot_retry_or_read_completion(case, monkeypatch, stage):
    error = fail("RESOURCE_UNAVAILABLE", "native failure")
    invoked = []

    def explode(*args, **kwargs):
        invoked.append(1)
        raise error

    monkeypatch.setattr(fitted, stage, explode)
    assert _error(error.code, lambda: case.fits.fit("inner", "2024", 1, "2023")) is error
    _error("EXPERIMENT_IDENTITY_CONFLICT", lambda: case.fits.fit("inner", "2024", 1, "2023"))
    _error("EXPERIMENT_IDENTITY_CONFLICT", lambda: case.fits.read("inner", "2024", 1, "2023"))
    assert invoked == [1] and not any("fit-result:" in slot for run, slot in case.records)


def test_completion_commit_failure_is_interrupted_not_retried(case, monkeypatch):
    original = fitted.append_record

    def append(*args, **kwargs):
        if args[3].startswith("distribution:fit-result:"):
            raise fail("RESOURCE_UNAVAILABLE", "commit failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(fitted, "append_record", append)
    _error("RESOURCE_UNAVAILABLE", lambda: case.fits.fit("inner", "2024", 1, "2023"))
    _error("EXPERIMENT_IDENTITY_CONFLICT", lambda: case.fits.fit("inner", "2024", 1, "2023"))
    assert sum(call[0] == "fit" for call in case.calls) == 1


@pytest.mark.parametrize("change", ["features", "labels", "order", "plan"])
def test_changed_matrix_order_or_source_plan_cannot_replay(case, change):
    case.fits.fit("inner", "2024", 1, "2023")
    if change == "features":
        next(row for row in case.inputs["features"] if row["horizon_sessions"] == 1)[case.columns[0]] = 100
    elif change == "labels":
        case.inputs["training"][0]["absolute_move_pct"] = 0.5
    elif change == "order":
        case.inputs["training"].reverse()
    else:
        case.state["plan_id"] = content_hash("different-plan")
    _error("EXPERIMENT_IDENTITY_CONFLICT", lambda: case.fits.read("inner", "2024", 1, "2023"))
    assert sum(call[0] == "fit" for call in case.calls) == 1


@pytest.mark.parametrize("member", ["schema_version", "claim", "result_id", "status", "final_read_authority"])
def test_replay_binds_closed_completion_envelope(case, member):
    case.fits.fit("inner", "2024", 1, "2023")
    _receipt(case)[member] = "tampered"
    _error("EXPERIMENT_IDENTITY_CONFLICT", lambda: case.fits.read("inner", "2024", 1, "2023"))


@pytest.mark.parametrize("member", ["training_support", "classifier", "point", "all_null_training_columns", "training_rows"])
def test_diagnostics_checked_against_admitted_training_not_only_hash(case, member):
    case.fits.fit("inner", "2024", 1, "2023")
    record = _receipt(case)
    record["diagnostics"][member] = None
    _rehash(record)
    _error("EXPERIMENT_IDENTITY_CONFLICT", lambda: case.fits.read("inner", "2024", 1, "2023"))


def test_tail_sample_values_and_order_are_training_only(case):
    case.fits.fit("inner", "2024", 1, "2023")
    record = _receipt(case)
    record["tail"]["excesses"] = fitted.write_section(case.trials.store, "distribution_fit_tail",
        [{"position": 0, "excess": 2.0}, {"position": 1, "excess": 0.0}], lambda row: (row["position"],))
    _rehash(record)
    _error("EXPERIMENT_IDENTITY_CONFLICT", lambda: case.fits.read("inner", "2024", 1, "2023"))


def test_forecast_cohort_must_exactly_match_native_events(case):
    result = case.fits.fit("inner", "2024", 1, "2023")
    record = _receipt(case)
    rows = deepcopy(result["rows"])
    rows[0]["event_id"] = "other"
    record["forecasts"] = fitted.write_section(case.trials.store, "distribution_fit_forecasts", sorted(rows, key=fitted._event), fitted._event)
    _rehash(record)
    _error("EXPERIMENT_IDENTITY_CONFLICT", lambda: case.fits.read("inner", "2024", 1, "2023"))


@pytest.mark.parametrize("error", [MemoryError(), OSError("I/O"), fail("RESOURCE_LIMIT_EXCEEDED", "cap")])
def test_native_failures_escape_without_completion(case, monkeypatch, error):
    def explode(*args, **kwargs):
        raise error

    monkeypatch.setattr(fitted, "fit_candidate", explode)
    with pytest.raises(type(error)) as caught:
        case.fits.fit("inner", "2024", 1, "2023")
    assert caught.value is error
    assert not any("fit-result:" in slot for run, slot in case.records)


def test_read_remains_allowed_under_both_guards_and_never_fits(case, monkeypatch):
    result = case.fits.fit("inner", "2024", 1, "2023")
    monkeypatch.setattr(fitted, "fit_candidate", lambda *args, **kwargs: pytest.fail("replay fitted"))
    with legacy_guard.no_fit_guard(), native_guard.no_fit_guard():
        assert case.fits.read("inner", "2024", 1, "2023") == result


@pytest.mark.parametrize("labels,tail,trees", [([0.0, 0.25, 0.0, 0.25], [], 1), ([0.0, 0.25, 20.0, 22.0], [0.0, 2.0], 3)])
def test_tiny_real_binary_and_multiclass_pairs(case, monkeypatch, labels, tail, trees):
    for row, label in zip(case.inputs["training"], labels):
        row["absolute_move_pct"] = label
    monkeypatch.setattr(fitted, "fit_candidate", numeric_fit)
    result = case.fits.fit("inner", "2024", 1, "2023")
    assert result["tail"]["excesses"] == tail
    assert result["receipt"]["diagnostics"]["classifier"]["trees_per_iteration"] == trees
    assert result["rows"][0]["point_prediction"] == pytest.approx(sum(labels) / 4)
    assert case.fits.read("inner", "2024", 1, "2023") == result


def test_public_surface_is_wrapper_validation_and_publication():
    assert fitted.__all__ == ["DistributionFits", "validate_fit_output", "publish_fit_output"]


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), 10 ** 400])
def test_replayed_points_require_finite_nonnegative_plain_numbers(case, value):
    result = case.fits.fit("inner", "2024", 1, "2023")
    result["rows"][0]["point_prediction"] = value
    record = _receipt(case)
    record["forecasts"] = fitted.write_section(case.trials.store, "distribution_fit_forecasts",
        sorted(result["rows"], key=fitted._event), fitted._event)
    _rehash(record)
    _error("EXPERIMENT_IDENTITY_CONFLICT", lambda: case.fits.read("inner", "2024", 1, "2023"))


@pytest.mark.parametrize("mode", ["bad_sum", "absent_class"])
def test_replayed_probabilities_enforce_numeric_and_absent_class_policy(case, mode):
    result = case.fits.fit("inner", "2024", 1, "2023")
    masses = result["rows"][0]["probabilities"]
    masses[2] = 0.25
    if mode == "absent_class":
        masses[0] = 0.0
    record = _receipt(case)
    record["forecasts"] = fitted.write_section(case.trials.store, "distribution_fit_forecasts",
        sorted(result["rows"], key=fitted._event), fitted._event)
    _rehash(record)
    if mode == "bad_sum":
        with pytest.raises(DataError) as caught:
            case.fits.read("inner", "2024", 1, "2023")
        assert caught.value.code == "CONTRACT_MISMATCH"
    else:
        _error("EXPERIMENT_IDENTITY_CONFLICT", lambda: case.fits.read("inner", "2024", 1, "2023"))


def test_missing_start_cannot_be_replaced_by_valid_completion(case):
    result = case.fits.fit("inner", "2024", 1, "2023")
    del case.records["own-run", "distribution:fit-start:" + result["receipt"]["claim"]["trial"]["trial_id"]]
    _error("EXPERIMENT_IDENTITY_CONFLICT", lambda: case.fits.read("inner", "2024", 1, "2023"))


def test_fresh_fit_guards_also_precede_completed_replay(case):
    case.fits.fit("inner", "2024", 1, "2023")
    case.calls.clear()
    with native_guard.no_fit_guard(), pytest.raises(native_guard.RuntimeFitForbidden):
        case.fits.fit("inner", "2024", 1, "2023")
    assert case.calls == []
