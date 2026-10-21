"""The one cohort loader: v2 manifests equal v1 records, and every disagreement refuses."""
import copy

import pytest

from engine.v2.foundation import ArtifactStore, content_hash
from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_inputs as inputs_module
from experiments import magnitude_search_shards as shards
from experiments.magnitude_search_inputs import load_fit_inputs, load_inputs, publish_inputs


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
        "target_recipe": {"secondary_diagnostics": _ids("mae", "quantile_calibration")},
        "holdout_exclusions": [],
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


def _scalars(value, prefix, skip, out):
    for name, item in value.items():
        path = prefix + name
        if path in skip:
            continue
        if isinstance(item, dict):
            _scalars(item, path + ".", skip, out)
        else:
            out[path] = item


def _position(item):
    return (item["position"],)


def _manifest(store, cohort=COHORT):
    sections = {}
    for member in shards.LIST_MEMBERS:
        path = member["section"]
        rows = _path_value(cohort, path)
        key = member["key"]
        if key is None:
            rows = [{"position": index, "row": item} for index, item in enumerate(rows)]
            key = _position
        sections[path] = shards.write_section(store, path, rows, key)
    inline = {}
    _scalars(cohort, "", set(sections), inline)
    return shards.build_manifest(inline, sections)


def _conflict(call):
    with pytest.raises(OpsError) as caught:
        call()
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.retryable is False


def test_v2_manifest_loads_to_the_same_inputs_as_the_v1_record(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    manifest = _manifest(store)
    inputs = load_inputs(store, manifest, shards.manifest_id(manifest))
    assert inputs == COHORT
    assert inputs is not manifest["inline"]
    assert "features" not in manifest["inline"] and manifest["sections"]["features"]["row_count"] == 2
    assert load_inputs(store, COHORT, content_hash(COHORT)) == COHORT


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
    clash["inline"]["training"] = 1
    _conflict(lambda: load_inputs(store, clash, shards.manifest_id(clash)))


def test_keyless_section_positions_must_be_consecutive(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    manifest = _manifest(store)
    gapped = copy.deepcopy(manifest)
    gapped["sections"]["preflight.event_ids"] = shards.write_section(
        store, "preflight.event_ids",
        [{"position": 0, "row": "E1"}, {"position": 2, "row": "E2"}], _position)
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
        [{"position": 0, "row": "E1", "extra": 1}], _position)
    _conflict(lambda: load_inputs(store, bad, shards.manifest_id(bad)))


def _fit_cohort(count):
    events = [f"E{index:03d}" for index in range(count)]
    cohort = copy.deepcopy(COHORT)
    cohort["training"] = [{"event_id": event, "absolute_move_pct": 1.0 + index, "extra": "dropped"}
                          for index, event in enumerate(events[:3])]
    cohort["features"] = [{"event_id": event, "context_id": "c", "horizon_sessions": horizon, "x": 0.5, "y": 2.0,
                           "unused": "dropped"} for event in events for horizon in (1, 3, 5)]
    return cohort, events


def test_load_fit_inputs_streams_many_shards_and_projects_only_fit_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(shards, "TARGET_BYTES", 300)
    monkeypatch.setattr(shards, "MAX_SHARD_BYTES", 600)
    cohort, events = _fit_cohort(12)
    store = ArtifactStore(tmp_path)
    manifest, inputs_id = publish_inputs(store, cohort)
    assert len(manifest["sections"]["features"]["shards"]) > 3
    evaluation = [events[7], events[9]]
    loaded = load_fit_inputs(store, manifest, inputs_id, evaluation, [1, 5], ("x",))
    wanted = set(events[:3]) | set(evaluation)
    assert loaded["training"] == [{"event_id": row["event_id"], "absolute_move_pct": row["absolute_move_pct"]}
                                  for row in cohort["training"]]
    assert loaded["features"] == [{"event_id": event, "context_id": "c", "horizon_sessions": horizon, "x": 0.5}
                                  for event in events if event in wanted for horizon in (1, 5)]


def test_load_fit_inputs_refuses_a_wrong_id_a_v1_record_and_a_corrupt_shard(tmp_path):
    cohort, events = _fit_cohort(4)
    store = ArtifactStore(tmp_path)
    manifest, inputs_id = publish_inputs(store, cohort)
    for record, expected in ((manifest, "sha256:" + "0" * 64), (cohort, content_hash(cohort))):
        with pytest.raises(OpsError) as caught:
            load_fit_inputs(store, record, expected, [], [1], ("x",))
        assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    ref = manifest["sections"]["features"]["shards"][0]["ref"]
    path = store.root / ref["storage_key"]
    size = path.stat().st_size
    path.chmod(0o644)
    path.write_bytes(b"x" * size)
    with pytest.raises(OpsError) as caught:
        load_fit_inputs(store, manifest, inputs_id, [], [1], ("x",))
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_load_fit_inputs_refuses_inline_paths_that_collide_with_sections(tmp_path):
    store = ArtifactStore(tmp_path)
    cohort, _ = _fit_cohort(2)
    manifest, _ = publish_inputs(store, cohort)
    for bad_inline in ({**manifest["inline"], "training": 1}, {**manifest["inline"], "preflight": 1}):
        record = {**manifest, "inline": bad_inline}
        record_id = shards.manifest_id(record)
        with pytest.raises(OpsError) as caught:
            load_fit_inputs(store, record, record_id, [], [1], ("x",))
        assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_load_fit_inputs_reads_one_shard_at_a_time(tmp_path, monkeypatch):
    monkeypatch.setattr(shards, "TARGET_BYTES", 300)
    monkeypatch.setattr(shards, "MAX_SHARD_BYTES", 600)
    cohort, _ = _fit_cohort(12)
    store = ArtifactStore(tmp_path)
    manifest, inputs_id = publish_inputs(store, cohort)
    reads = []
    real_read = store.read_verified

    def counting_read(ref):
        reads.append(ref)
        return real_read(ref)

    store.read_verified = counting_read
    at_row = []
    original = inputs_module.iter_section

    def spy(store_, section, entry, key):
        for row in original(store_, section, entry, key):
            if section == "features":
                at_row.append(len(reads))
            yield row

    monkeypatch.setattr(inputs_module, "iter_section", spy)
    load_fit_inputs(store, manifest, inputs_id, [], [1], ("x",))
    assert len(reads) > 3
    assert at_row == sorted(at_row)
    assert at_row[0] == len(manifest["sections"]["training"]["shards"]) + 1
    assert at_row[0] < at_row[-1]
