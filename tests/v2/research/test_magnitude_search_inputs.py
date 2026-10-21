"""The one cohort loader: v2 manifests equal v1 records, and every disagreement refuses."""
import copy

import pytest

from engine.v2.foundation import ArtifactStore, content_hash
from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_shards as shards
from experiments.magnitude_search_inputs import load_inputs, stored_id


def _ids(*names):
    return list(names)


COHORT = {
    "preflight": {
        "event_ids": _ids("E1", "E2"),
        "planned_inputs": [{"event_id": name, "fragment_id": "f", "object_id": "o", "content_hash": "h"}
                           for name in ("E1", "E2")],
        "count": 2,
    },
    "stored_inputs_id": "stored",
    "training_inputs": {
        "candidate_event_ids": _ids("E1", "E2"),
        "qualified_event_ids": _ids("E1"),
        "read_inputs": [{"event_id": "E1", "fragment_id": "f", "object_id": "o", "content_hash": "h"}],
        "omitted_labels": [{"event_id": "E2", "reason": "late"}],
        "rows": 1,
    },
    "training": [{"event_id": "E1", "absolute_move_pct": 1.5}],
    "features": [{"event_id": name, "context_id": "c", "horizon_sessions": 1, "x": 0.5}
                 for name in ("E1", "E2")],
}


def _path_value(inputs, path):
    value = inputs
    for name in path.split("."):
        value = value[name]
    return value


def _manifest(store, cohort=COHORT):
    inline, sections = copy.deepcopy(cohort), {}
    for member in shards.LIST_MEMBERS:
        path = member["section"]
        rows = _path_value(cohort, path)
        key = member["key"]
        if key is None:
            rows = [{"position": index, "row": item} for index, item in enumerate(rows)]
            key = lambda item: (item["position"],)
        sections[path] = shards.write_section(store, path, rows, key)
        *parents, leaf = path.split(".")
        _path_value(inline, ".".join(parents)).pop(leaf) if parents else inline.pop(leaf)
    return shards.build_manifest(inline, sections)


def _conflict(call):
    with pytest.raises(OpsError) as caught:
        call()
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.retryable is False


def test_v2_manifest_loads_to_the_same_inputs_as_the_v1_record(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    manifest = _manifest(store)
    inputs = load_inputs(store, manifest, stored_id(manifest))
    assert inputs == COHORT
    assert inputs is not manifest["inline"]
    assert "features" not in manifest["inline"] and manifest["sections"]["features"]["row_count"] == 2
    assert stored_id(manifest) == shards.manifest_id(manifest)
    assert load_inputs(store, COHORT, content_hash(COHORT)) == COHORT
    assert stored_id(COHORT) == content_hash(COHORT)


def test_expected_id_must_match_for_both_formats(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    manifest = _manifest(store)
    _conflict(lambda: load_inputs(store, manifest, "0" * 64))
    _conflict(lambda: load_inputs(store, COHORT, "0" * 64))
    _conflict(lambda: load_inputs(store, None, "0" * 64))
    _conflict(lambda: load_inputs(store, manifest, content_hash({"other": 1})))
    _conflict(lambda: load_inputs(store, COHORT, content_hash({"other": 1})))


def test_manifest_must_carry_exactly_the_inventory_sections(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    manifest = _manifest(store)
    missing = copy.deepcopy(manifest)
    missing["sections"].pop("training")
    _conflict(lambda: load_inputs(store, missing, shards.manifest_id(missing)))


def test_inline_member_colliding_with_a_section_is_a_conflict(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    manifest = _manifest(store)
    clash = copy.deepcopy(manifest)
    clash["inline"]["training"] = []
    _conflict(lambda: load_inputs(store, clash, shards.manifest_id(clash)))


def test_keyless_section_positions_must_be_consecutive(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    manifest = _manifest(store)
    gapped = copy.deepcopy(manifest)
    gapped["sections"]["preflight.event_ids"] = shards.write_section(
        store, "preflight.event_ids",
        [{"position": 0, "row": "E1"}, {"position": 2, "row": "E2"}], lambda item: (item["position"],))
    _conflict(lambda: load_inputs(store, gapped, shards.manifest_id(gapped)))


def test_corrupt_shard_fails_closed(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    manifest = _manifest(store)
    ref = manifest["sections"]["features"]["shards"][0]["ref"]
    path = tmp_path / "store" / ref["storage_key"]
    path.chmod(0o644)
    path.write_bytes(b"{}")
    _conflict(lambda: load_inputs(store, manifest, shards.manifest_id(manifest)))


def test_extra_manifest_section_is_a_conflict(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    manifest = _manifest(store)
    extra = copy.deepcopy(manifest)
    extra["sections"]["unlisted"] = extra["sections"]["features"]
    _conflict(lambda: load_inputs(store, extra, shards.manifest_id(extra)))


def test_malformed_keyless_wrapper_is_a_conflict(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    manifest = _manifest(store)
    bad = copy.deepcopy(manifest)
    bad["sections"]["preflight.event_ids"] = shards.write_section(
        store, "preflight.event_ids",
        [{"position": 0, "row": "E1", "extra": 1}], lambda item: (item["position"],))
    _conflict(lambda: load_inputs(store, bad, shards.manifest_id(bad)))
