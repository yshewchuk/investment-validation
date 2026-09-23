import hashlib
import json
from dataclasses import replace

import pytest

from engine.v2.models import (
    FrozenInference,
    ModelBinding,
    ModelRelease,
)
from engine.v2.models.contracts import ArtifactMember
from engine.v2.scoring.frozen_executor import (
    FrozenStageExecutor,
    FrozenStageRefusal,
)


def _release(root) -> tuple[ModelRelease, str]:
    payload = {
        "schema_version": "linear_estimator.v1.0",
        "feature_order": ["beta", "alpha"],
        "outputs": [
            {
                "name": "prediction",
                "intercept": 1.0,
                "coefficients": [10.0, 2.0],
            },
            {
                "name": "confidence",
                "intercept": -1.0,
                "coefficients": [1.0, -1.0],
            },
        ],
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    (root / "estimator.json").write_bytes(raw)
    artifact_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
    binding = ModelBinding(
        binding_id="str-thru-size",
        model_id="size-v1",
        role="size",
        strategy_id="STR-THRU",
        decision_clock_id="entry-close",
        adapter="json-linear.v1",
        feature_order=("beta", "alpha"),
        output_names=("prediction", "confidence"),
        members=(
            ArtifactMember(
                name="estimator",
                path="estimator.json",
                content_hash=artifact_hash,
            ),
        ),
    )
    return (
        ModelRelease(
            release_id="release-v1",
            deployment_id="deployment-v1",
            bindings=(binding,),
        ),
        artifact_hash,
    )


def _executor(tmp_path):
    release, artifact_hash = _release(tmp_path)
    return (
        FrozenStageExecutor(
            inference=FrozenInference(tmp_path),
            release=release,
            binding_id="str-thru-size",
        ),
        artifact_hash,
    )


def test_runtime_features_map_deterministically_to_one_named_output_row(tmp_path):
    executor, _ = _executor(tmp_path)

    forward = executor.execute({"alpha": 3.0, "beta": 2.0})
    reversed_input = executor.execute({"beta": 2.0, "alpha": 3.0})

    assert forward == reversed_input
    assert forward.request.feature_order == ("beta", "alpha")
    assert forward.request.rows == ((2.0, 3.0),)
    assert dict(forward.outputs) == {
        "prediction": pytest.approx(27.0),
        "confidence": pytest.approx(-2.0),
    }
    with pytest.raises(TypeError):
        forward.outputs["prediction"] = 0.0


def test_missing_feature_refuses_before_inference(tmp_path):
    executor, _ = _executor(tmp_path)

    with pytest.raises(FrozenStageRefusal) as error:
        executor.execute({"beta": 2.0})

    assert error.value.code == "MISSING_FEATURES"
    assert error.value.reason_codes == ("MISSING_FEATURES",)
    assert error.value.missing_features == ("alpha",)


def test_per_role_vector_does_not_fall_back_to_cross_role_merged_features(tmp_path):
    executor, _ = _executor(tmp_path)
    facts = {
        "alpha": 3.0, "beta": 2.0,
        "role_model_inputs": {"size": {"alpha": 3.0}},
    }
    with pytest.raises(FrozenStageRefusal) as error:
        executor.execute(facts)
    assert error.value.reason_codes == ("MISSING_FEATURES",)
    assert error.value.missing_features == ("beta",)


def test_stage_owned_derived_features_override_role_primitives(tmp_path):
    executor, _ = _executor(tmp_path)
    result = executor.execute({
        "alpha": 3.0, "beta": 2.0,
        "role_model_inputs": {"size": {"alpha": 3.0, "beta": 2.0}},
        "_native_stage_facts": {"beta": 8.0},
    })
    assert result.request.rows == ((8.0, 3.0),)


def test_model_not_ready_preserves_inference_refusal(tmp_path):
    executor, _ = _executor(tmp_path)
    (tmp_path / "estimator.json").unlink()

    with pytest.raises(FrozenStageRefusal) as error:
        executor.execute({"alpha": 3.0, "beta": 2.0})

    assert error.value.code == "MODEL_NOT_READY"
    assert error.value.reason_codes == ("ARTIFACT_INVALID",)


def test_result_carries_verified_artifact_provenance(tmp_path):
    executor, artifact_hash = _executor(tmp_path)

    result = executor.execute({"alpha": 3.0, "beta": 2.0})

    assert result.model_id == "size-v1"
    assert result.request.release_id == "release-v1"
    assert result.request.binding_id == "str-thru-size"
    assert result.artifact_hashes == (artifact_hash,)


def test_predict_exposes_named_outputs_for_native_stages(tmp_path):
    executor, _ = _executor(tmp_path)

    assert dict(executor.predict({"alpha": 3.0, "beta": 2.0})) == {
        "prediction": pytest.approx(27.0),
        "confidence": pytest.approx(-2.0),
    }


def test_executor_rejects_inference_lineage_that_differs_from_binding(tmp_path):
    release, _ = _release(tmp_path)

    class WrongLineageInference(FrozenInference):
        def infer(self, model_release, request):
            result = super().infer(model_release, request)
            return replace(result, artifact_hashes=("sha256:not-the-artifact",))

    executor = FrozenStageExecutor(
        inference=WrongLineageInference(tmp_path),
        release=release,
        binding_id="str-thru-size",
    )

    with pytest.raises(FrozenStageRefusal) as error:
        executor.execute({"alpha": 3.0, "beta": 2.0})

    assert error.value.code == "ARTIFACT_PROVENANCE_MISMATCH"
    assert error.value.reason_codes == ("ARTIFACT_PROVENANCE_MISMATCH",)
