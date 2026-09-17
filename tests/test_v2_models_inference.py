import hashlib
import json

import pytest

from engine.v2.foundation import from_document, to_document
from engine.v2.models import (
    MODEL_NOT_READY,
    FrozenInference,
    InferenceRequest,
    ModelBinding,
    ModelRelease,
    PredictionFrame,
    RuntimeFitForbidden,
)
from engine.v2.models.contracts import ArtifactMember


def _release(root):
    payload = {
        "schema_version": "linear_estimator.v1.0",
        "feature_order": ["x"],
        "outputs": [{"name": "prediction", "intercept": 1.0, "coefficients": [2.0]}],
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    path = root / "estimator.json"
    path.write_bytes(raw)
    member = ArtifactMember(
        name="estimator", path="estimator.json",
        content_hash="sha256:" + hashlib.sha256(raw).hexdigest(),
    )
    binding = ModelBinding(
        binding_id="b1", model_id="m1", role="size", strategy_id="*",
        decision_clock_id="entry-close", adapter="json-linear.v1",
        feature_order=("x",), output_names=("prediction",), members=(member,),
    )
    return ModelRelease(release_id="r1", deployment_id="d1", bindings=(binding,))


def test_cold_and_warm_prediction_frames_are_equal(tmp_path):
    release = _release(tmp_path)
    request = InferenceRequest(release_id="r1", binding_id="b1", feature_order=("x",), rows=((3.0,),))
    inference = FrozenInference(tmp_path)
    cold = inference.infer(release, request)
    warm = inference.infer(release, request)
    assert cold == warm
    assert cold.predictions == ((7.0,),)
    frame = PredictionFrame(model_id="m1", release_id="r1", binding_id="b1",
                             ordered_outputs=cold.output_names, rows=cold.predictions,
                             artifact_hashes=cold.artifact_hashes)
    assert from_document(PredictionFrame, to_document(frame)) == frame


def test_missing_or_incompatible_artifact_is_model_not_ready(tmp_path):
    release = _release(tmp_path)
    (tmp_path / "estimator.json").unlink()
    request = InferenceRequest(release_id="r1", binding_id="b1", feature_order=("x",), rows=((3.0,),))
    result = FrozenInference(tmp_path).infer(release, request)
    assert result.status == MODEL_NOT_READY
    assert result.reason_codes == ("ARTIFACT_INVALID",)


def test_loaded_artifact_rejects_runtime_fit(tmp_path):
    release = _release(tmp_path)
    request = InferenceRequest(release_id="r1", binding_id="b1", feature_order=("x",), rows=((3.0,),))
    inference = FrozenInference(tmp_path)
    inference.infer(release, request)
    with pytest.raises(RuntimeFitForbidden):
        next(iter(inference._cache.values())).fit()
