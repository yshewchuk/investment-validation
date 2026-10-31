"""Closed explicit cohort layouts refuse before I/O; historical untagged rows replay."""
from copy import deepcopy

import pytest

from engine.v2.foundation import ArtifactStore, content_hash
from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_inputs as inputs
from experiments import magnitude_search_shards as shards
from tests.v2.research.test_magnitude_search_inputs import COHORT


def _refuses(call, message="search cohort preflight schema is unsupported"):
    with pytest.raises(OpsError) as caught:
        call()
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.message == message
    assert caught.value.problem.retryable is False


@pytest.mark.parametrize("schema", ["research_unknown_preflight.v1", "", 17, True, [], {}])
@pytest.mark.parametrize("operation", ["load", "fit", "publish"])
def test_unsupported_explicit_layout_refuses_before_shard_io(tmp_path, monkeypatch, schema, operation):
    store = ArtifactStore(tmp_path / "store")
    cohort = deepcopy(COHORT)
    manifest, _ = inputs.publish_inputs(store, cohort)
    manifest["inline"]["preflight.schema_version"] = schema
    identity = shards.manifest_id(manifest)  # Self-consistent corruption, not merely a stale hash.
    cohort["preflight"]["schema_version"] = schema
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    monkeypatch.setattr(store, "read_verified", lambda *a, **k: pytest.fail("shard read"))
    monkeypatch.setattr(store, "publish_bytes", lambda *a, **k: pytest.fail("shard publication"))
    message = ("search cohort manifest is invalid" if isinstance(schema, (list, dict))
               else "search cohort preflight schema is unsupported")
    if operation == "load":
        _refuses(lambda: inputs.load_inputs(store, manifest, identity), message)
    elif operation == "fit":
        _refuses(lambda: inputs.load_fit_inputs(store, manifest, identity, ["E2"], [1], ["x"]), message)
    else:
        _refuses(lambda: inputs.publish_inputs(store, cohort))
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("schema", ["absent", None, "research_magnitude_preflight.v1"])
@pytest.mark.parametrize("explicit_target", [False, True])
def test_legacy_magnitude_layout_and_whole_cohort_adapter_remain_unchanged(tmp_path, schema, explicit_target):
    store = ArtifactStore(tmp_path / "store")
    cohort = deepcopy(COHORT)
    if schema != "absent":
        cohort["preflight"]["schema_version"] = schema
    if explicit_target:
        cohort["target"] = {"column": "absolute_move_pct"}
    manifest, identity = inputs.publish_inputs(store, cohort)
    assert set(manifest["sections"]) == {member["section"] for member in shards.LIST_MEMBERS}
    assert inputs.load_inputs(store, manifest, identity) == cohort
    assert inputs.load_fit_inputs(store, manifest, identity, ["E2"], [1], ["x"]) == {
        "features": cohort["features"], "training": cohort["training"]}
    cohort["preflight"]["schema_version"] = "opaque_historical_whole_cohort"
    assert inputs.load_inputs(store, cohort, content_hash(cohort)) == cohort
