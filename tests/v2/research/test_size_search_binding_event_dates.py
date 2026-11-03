import pytest

from engine.v2.foundation import content_hash
from engine.v2.models.training import fit_routes, magnitude_elasticnet, magnitude_ridge_hgb
from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_evidence, size_fit_diagnostics, size_search_binding
from experiments.magnitude_candidate_recipe import candidate_recipe


@pytest.mark.parametrize("value", [1, "yes", None])
def test_fold_inputs_rejects_non_bool_date_requirement_before_admission(monkeypatch, value):
    verified = []

    def require(condition):
        if not condition:
            raise ValueError("invalid event date requirement")

    monkeypatch.setattr(size_search_binding, "_require", require)
    monkeypatch.setattr(size_search_binding, "verify_installation", lambda binding: verified.append(binding))

    with pytest.raises(ValueError, match="invalid event date requirement"):
        size_search_binding.fold_inputs(None, None, None, requires_event_dates=value)

    assert verified == []


def test_mlp_replay_requires_digest_of_admitted_training_dates():
    recipe = candidate_recipe("mlp", {"profile": "relu32_l2_0p01"}, columns=["x"],
        feature_dictionary_hash="sha256:" + "1" * 64, requires_event_dates=True)
    dates_id = content_hash(["2024-01-02", "2024-01-01"])
    diagnostic = {"schema_version": "magnitude_mlp_fit.v1", "recipe_id": content_hash(recipe),
        "horizon_sessions": 5, "component_fits": 1, "converged": True, "n_iter": 1,
        "input_features": 1, "loss": 0.1, "loss_scale": "standardized_target",
        "training_event_dates_id": dates_id}
    execution = {"schema_version": "size_candidate_execution.v1", "fit_recipe": recipe}
    inputs = {"training": [{"event_id": "b"}, {"event_id": "a"}], "features": [
        {"event_id": "a", "horizon_sessions": 5, "event_date": "2024-01-01"},
        {"event_id": "b", "horizon_sessions": 5, "event_date": "2024-01-02"}]}
    result = {"event": "succeeded", "diagnostics": [diagnostic]}
    assert size_fit_diagnostics.evidence(result, execution, [5], inputs, {}) == {
        "diagnostics": [diagnostic]}

    tampered = {**result, "diagnostics": [{**diagnostic,
        "training_event_dates_id": content_hash(["wrong-date"])}]}
    with pytest.raises(OpsError) as caught:
        size_fit_diagnostics.evidence(tampered, execution, [5], inputs, {})
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_mlp_replay_refuses_malformed_dates_when_digest_matches():
    recipe = candidate_recipe("mlp", {"profile": "relu32_l2_0p01"}, columns=["x"],
        feature_dictionary_hash="sha256:" + "1" * 64, requires_event_dates=True)
    dates_id = content_hash(["2024-01-02", "wrong-date"])
    diagnostic = {"schema_version": "magnitude_mlp_fit.v1", "recipe_id": content_hash(recipe),
        "horizon_sessions": 5, "component_fits": 1, "converged": True, "n_iter": 1,
        "input_features": 1, "loss": 0.1, "loss_scale": "standardized_target",
        "training_event_dates_id": dates_id}
    execution = {"schema_version": "size_candidate_execution.v1", "fit_recipe": recipe}
    inputs = {"training": [{"event_id": "b"}, {"event_id": "a"}], "features": [
        {"event_id": "b", "horizon_sessions": 5, "event_date": "2024-01-02"},
        {"event_id": "a", "horizon_sessions": 5, "event_date": "wrong-date"}]}
    result = {"event": "succeeded", "diagnostics": [diagnostic]}
    with pytest.raises(OpsError) as caught:
        size_fit_diagnostics.evidence(result, execution, [5], inputs, {})
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


@pytest.mark.parametrize("value, expected", [
    (20240101, False),
    ("2024-02-30", False),
    ("20240101", False),
    ("2024-01-01T00:00:00", False),
    ("2024-01-01", True),
])
def test_is_canonical_date(value, expected):
    assert size_search_binding._is_canonical_date(value) is expected


@pytest.mark.parametrize("route, recipe", [
    ("magnitude", {"schema_version": "magnitude_fitted_recipe.v1"}),
    ("regression_candidate", {"model": "ridge"}),
    ("magnitude_candidate", {"model": "ridge"}),
    ("magnitude_candidate", {"model": "mlp", "requires_event_dates": 1}),
])
def test_fit_execution_refuses_unsupported_event_dates(monkeypatch, route, recipe):
    monkeypatch.setattr(fit_routes, "check_fitting_allowed", lambda path: None)
    with pytest.raises(ValueError, match="event.date"):
        fit_routes.fit_execution(route, recipe, [], [], [], columns=[], event_dates=["2024-01-01"])


@pytest.mark.parametrize("module, func_name", [
    (magnitude_elasticnet, "fit_elasticnet_predictions"),
    (magnitude_ridge_hgb, "fit_ridge_hgb_predictions"),
])
def test_direct_fitters_refuse_event_dates(monkeypatch, module, func_name):
    monkeypatch.setattr(module, "forbid_fitting", lambda *a, **k: None)
    monkeypatch.setattr(module, "forbid_v2_fitting", lambda *a, **k: None)
    func = getattr(module, func_name)
    with pytest.raises(ValueError, match="event dates"):
        func(None, None, None, None, columns=None, event_dates=["2024-01-01"])


@pytest.mark.parametrize("requires_event_dates", [True, 1, "yes"])
def test_admit_fold_inputs_refuses_unsupported_event_dates(requires_event_dates):
    binding = {"schema_version": "native_experiment_registration.v1.8"}
    with pytest.raises(ValueError):
        magnitude_search_evidence.admit_fold_inputs(
            None, binding, None, history_failure=None, requires_event_dates=requires_event_dates)


@pytest.mark.parametrize("eval_date", [None, "wrong-date", "20240101"])
def test_mlp_replay_refuses_malformed_evaluation_dates(eval_date):
    recipe = candidate_recipe("mlp", {"profile": "relu32_l2_0p01"}, columns=["x"],
        feature_dictionary_hash="sha256:" + "1" * 64, requires_event_dates=True)
    dates_id = content_hash(["2024-01-02", "2024-01-01"])
    diagnostic = {"schema_version": "magnitude_mlp_fit.v1", "recipe_id": content_hash(recipe),
        "horizon_sessions": 5, "component_fits": 1, "converged": True, "n_iter": 1,
        "input_features": 1, "loss": 0.1, "loss_scale": "standardized_target",
        "training_event_dates_id": dates_id}
    execution = {"schema_version": "size_candidate_execution.v1", "fit_recipe": recipe}
    eval_row = {"event_id": "c", "horizon_sessions": 5}
    if eval_date is not None:
        eval_row["event_date"] = eval_date
    inputs = {"training": [{"event_id": "b"}, {"event_id": "a"}], "features": [
        {"event_id": "b", "horizon_sessions": 5, "event_date": "2024-01-02"},
        {"event_id": "a", "horizon_sessions": 5, "event_date": "2024-01-01"},
        eval_row]}
    result = {"event": "succeeded", "diagnostics": [diagnostic]}
    with pytest.raises(OpsError) as caught:
        size_fit_diagnostics.evidence(result, execution, [5], inputs, {})
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"