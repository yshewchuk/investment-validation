import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import size_fit_diagnostics, size_search_binding
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