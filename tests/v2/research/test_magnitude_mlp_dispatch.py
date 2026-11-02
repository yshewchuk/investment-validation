"""Standalone neural dispatch and contained synthetic fits, not search admission."""
import copy
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from engine.models import no_fit as legacy_no_fit
from engine.v2.foundation import canonical_json, content_hash
from engine.v2.models import no_fit
from engine.v2.models.magnitude_candidate import magnitude_candidate_family
from engine.v2.models.magnitude_mlp_recipe import mlp_recipe
from engine.v2.models.regression_candidate import regression_candidate_recipe
from engine.v2.models.training import magnitude_mlp
from engine.v2.ops.errors import OpsError
from experiments import magnitude_candidate_recipe as assembly
from experiments import magnitude_search_execution as execution
from experiments.magnitude_candidate_fit import fit_candidate
from experiments.magnitude_fitted import fitted_forecasts
from experiments.magnitude_search_request_versions import compile_request
from experiments.size_fit_diagnostics import evidence
from tests.v2.research.magnitude_search_fit_source import fit_source
from tests.v2.research.test_magnitude_search_execution import LIMITS
from tests.v2.research.test_magnitude_search_request import request as v2_request
from tests.v2.research.test_magnitude_search_request_v3 import v3_request
from tests.v2.research.test_magnitude_search_request_v4 import v4_request
from tests.v2.research.test_magnitude_search_request_v5 import v5_request

COLUMNS = ["signal", "constant", "empty"]
DICTIONARY = "sha256:" + "1" * 64
PROFILES = ["relu32_l2_0p01", "relu64x32_l2_0p01", "relu64x32_l2_0p1"]


def recipe(profile=PROFILES[0]):
    return mlp_recipe(columns=COLUMNS, feature_dictionary_hash=DICTIONARY, profile=profile)


def cohort():
    """No evaluation outcomes: the child receives training targets and features only."""
    training = [{"event_id": f"train-{i:03}", "absolute_move_pct": 6 + 8 * i / 79}
                for i in range(80)]
    values = [(r["event_id"], -2 + 4 * i / 79) for i, r in enumerate(training)]
    values += [("eval-z", -1.), ("eval-a", 1.)]
    rows = [{"event_id": event, "context_id": DICTIONARY, "horizon_sessions": horizon,
             "signal": value + horizon / 10, "constant": 7.,
             "empty": 100. if event.startswith("eval") else None,
             "unused_future_field": 99999.}
            for event, value in values for horizon in (5, 1)]
    inputs = {"features": sorted(rows, key=lambda row: (row["event_id"], row["horizon_sessions"])),
              "training": training}
    fold = {"training_candidate_event_ids": [r["event_id"] for r in training],
            "evaluation_event_ids": ["eval-z", "eval-a"]}
    policy = {"schema_version": "size_candidate_execution.v1", "model": "mlp",
              "features": COLUMNS, "fit_recipe": recipe()}
    return inputs, fold, {"horizons": [5, 1]}, policy


@pytest.mark.parametrize("profile", PROFILES)
def test_recipe_dispatch_preserves_complete_metadata_without_caller_mutation(profile):
    columns, parameters = COLUMNS.copy(), {"profile": profile}
    result = assembly.candidate_recipe("mlp", parameters, columns=columns,
                                      feature_dictionary_hash=DICTIONARY)
    assert result == recipe(profile)
    assert magnitude_candidate_family(result) == "mlp"
    result["features"].clear()
    result["estimator"]["params"]["hidden_layer_sizes"].clear()
    assert columns == COLUMNS and parameters == {"profile": profile}
    assert recipe(profile)["features"] == COLUMNS


@pytest.mark.parametrize("parameters", [{}, {"profile": PROFILES[0], "alpha": .1},
                                        {"hidden_layer_sizes": [32]}, None])
def test_wrong_mlp_parameter_keys_refuse_before_construction(monkeypatch, parameters):
    monkeypatch.setattr(assembly, "mlp_recipe", lambda **kwargs: pytest.fail("constructed"))
    with pytest.raises(ValueError, match="exact parameter keys"):
        assembly.candidate_recipe("mlp", parameters, columns=None, feature_dictionary_hash=None)


def test_recipe_dispatch_is_source_free_and_invalid_schema_stays_lazy():
    probe = '''
import importlib.abc
import sys
class RejectTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert not fullname.startswith(('engine.v2.models.training', 'sklearn',
            'engine.v2.research', 'engine.v2.ops', 'engine.v2.data', 'engine.data')), fullname
sys.meta_path.insert(0, RejectTraining())
from experiments.magnitude_candidate_recipe import candidate_recipe
from experiments.magnitude_candidate_fit import fit_candidate
value = candidate_recipe('mlp', {'profile':'relu32_l2_0p01'}, columns=['x'],
                         feature_dictionary_hash='sha256:' + '1' * 64)
value['schema_version'] = 'magnitude_mlp_recipe.v2'
try:
    fit_candidate(value, None, None, None, columns=None)
except ValueError:
    pass
else:
    raise AssertionError('invalid schema admitted')
'''
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr


class Poison:
    def __array__(self, *args, **kwargs):
        pytest.fail("array inspected before guard or policy refusal")


@pytest.mark.parametrize("guard", [legacy_no_fit, no_fit])
def test_both_guards_precede_neural_dispatch(guard, monkeypatch):
    monkeypatch.setattr(magnitude_mlp, "fit_mlp_predictions", lambda *a, **k: pytest.fail("fit"))
    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden):
        fit_candidate(recipe(), Poison(), Poison(), Poison(), columns=COLUMNS)


def test_dispatch_delegates_once_and_preserves_the_entire_result(monkeypatch):
    calls, arrays = [], [object(), object(), object()]
    result = {"predictions": [7.], "opaque_evidence": object()}
    policy = recipe()
    def fitted(*args, **kwargs):
        calls.append((args, kwargs))
        return result
    monkeypatch.setattr(magnitude_mlp, "fit_mlp_predictions", fitted)
    assert fit_candidate(policy, *arrays, columns=COLUMNS) is result
    assert len(calls) == 1
    assert all(actual is expected for actual, expected in zip(calls[0][0], [policy, *arrays]))
    assert calls[0][1] == {"columns": COLUMNS}


@pytest.mark.parametrize("error", [ValueError("fit"), MemoryError("memory"), OSError("io"),
                                  RuntimeError("unexpected")])
def test_horizon_path_preserves_failure_taxonomy_and_never_retries(monkeypatch, error):
    inputs, fold, plan, policy = cohort()
    indexed = {(row["event_id"], row["horizon_sessions"]): row for row in inputs["features"]}
    calls, progress = [], []
    def refusing(*args, **kwargs):
        calls.append(1)
        raise error
    monkeypatch.setattr(magnitude_mlp, "fit_mlp_predictions", refusing)
    expected = OpsError if isinstance(error, ValueError) else type(error)
    with pytest.raises(expected) as caught:
        fitted_forecasts(indexed, pd.DataFrame(inputs["training"]), fold, plan, policy,
            strict_failures=True, progress=lambda state, fit: progress.append(state))
    assert calls == [1] and progress == ["started"]
    if isinstance(error, ValueError):
        assert caught.value.code == "EXPERIMENT_VARIANT_FAILED"
    else:
        assert caught.value is error


@pytest.mark.parametrize("factory", [v2_request, v3_request, v4_request, v5_request])
def test_existing_request_versions_do_not_admit_neural_search(factory):
    request = factory()
    unchanged = canonical_json(compile_request(request))
    request["space"]["mlp"] = {"profile": PROFILES}
    with pytest.raises(OpsError) as caught:
        compile_request(request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert canonical_json(compile_request(factory())) == unchanged


@pytest.mark.parametrize("factory,digest", [
    (v2_request, "31b7f493b3f2742e4a948767a352f3fa2bbe408b01898fcb9e0371c4b598fac0"),
    (v3_request, "08b06276c5099e12a1238eafd2f4341bc0d430c7ef3b96a7151c92677a7e03de"),
    (v4_request, "da6a620e6808ff204ef072d2befa4080306b1d143bbae25aa010406d1d178b72"),
    (v5_request, "a88310ec02b72184245707dec084da4d8b1d5f47e3714cb5311625ec66d0e6ab"),
])
def test_existing_request_schedule_identities_match_pre_neural_dispatch(factory, digest):
    # Captured from published #1943 tree (0fd94938), not recomputed by this assertion.
    assert content_hash(compile_request(factory())) == "sha256:" + digest


def test_signed_recipe_dispatch_does_not_gain_neural_admission():
    with pytest.raises(ValueError):
        regression_candidate_recipe("mlp", {"profile": PROFILES[0]}, target_domain="signed",
                                    columns=COLUMNS, feature_dictionary_hash=DICTIONARY)


def test_real_contained_child_preserves_forecasts_diagnostics_and_component_count(tmp_path):
    inputs, fold, plan, policy = cohort()
    before = copy.deepcopy(inputs)
    source = fit_source(inputs, tmp_path)
    progress = []
    result = execution.run_fitted_trial(source, fold, plan, policy,
        execution.execution_limits(LIMITS), lambda state, fit: progress.append((state, fit)))
    assert result["event"] == "succeeded"
    checked = evidence(result, policy, plan["horizons"])
    assert [state for state, _ in progress] == ["started", "completed"] * 2
    assert sum(item["component_fits"] for item in checked["diagnostics"]) == 2
    assert [item["horizon_sessions"] for item in checked["diagnostics"]] == [5, 1]
    expected = []
    for horizon, item in zip(plan["horizons"], checked["diagnostics"]):
        indexed = {r["event_id"]: r for r in inputs["features"] if r["horizon_sessions"] == horizon}
        matrices = [[[indexed[e][column] for column in COLUMNS] for e in events]
                    for events in (fold["training_candidate_event_ids"], fold["evaluation_event_ids"])]
        direct = magnitude_mlp.fit_mlp_predictions(policy["fit_recipe"], matrices[0],
            [r["absolute_move_pct"] for r in inputs["training"]], matrices[1], columns=COLUMNS)
        assert item == {"horizon_sessions": horizon, **{k: v for k, v in direct.items() if k != "predictions"}}
        expected += [{"event_id": event, "horizon_sessions": horizon,
                      "predicted_absolute_move_pct": value}
                     for event, value in zip(fold["evaluation_event_ids"], direct["predictions"])]
    assert result["forecasts"] == sorted(expected, key=lambda row: (row["event_id"], row["horizon_sessions"]))
    assert np.isfinite([row["predicted_absolute_move_pct"] for row in result["forecasts"]]).all()
    assert all(fit["training_event_ids"] == fold["training_candidate_event_ids"] for _, fit in progress)
    assert all(fit["evaluation_event_ids"] == fold["evaluation_event_ids"] for _, fit in progress)
    assert inputs == before
