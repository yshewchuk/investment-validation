"""In-process stored-panel memo: read once per key, fingerprint-checked reuse, private copies."""
import pandas as pd
import pytest

from engine.v2.contracts import SnapshotRef
from engine.v2.foundation import content_hash, to_document
from engine.v2.ops.errors import OpsError
from engine.v2.research.eligibility_policy import eligibility_policy
from experiments import size_search_binding as binding_mod
from experiments.size_search_binding import _stored_panel


@pytest.fixture(autouse=True)
def _fresh_memo():
    binding_mod._PANEL_MEMO.clear()
    yield
    binding_mod._PANEL_MEMO.clear()


class _Repo:
    def resolve(self, value):
        return value


def _snapshot(tag):
    return SnapshotRef(snapshot_id=tag, manifest_hash=content_hash(tag + ":manifest"),
                       table_versions={}, calendar_version="synthetic-calendar",
                       source_priority_version="synthetic-priority", finality_receipt_refs=(),
                       knowledge_mode_by_table={})


def _panel():
    return {"context": {"target_events": [{"event_id": "t1", "label_end": "2023-01-01"},
                                          {"event_id": "e1", "label_end": "2023-06-01"}]},
            "feature_snapshot": {"feature_snapshot_id": "snap"},
            "rows": [{"event_id": "t1", "v": 1}, {"event_id": "e1", "v": 2}]}


def _panel_request():
    return {"feature_snapshot_id": "snap", "history_snapshot": "hist",
            "price_history_snapshot": "price", "as_of_month": "2025-01",
            "history_start": "2010-01-01", "source_start": "2015-01-01",
            "policy": eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30),
            "event_ids": ["e1", "t1"], "horizons": [1, 3, 5],
            "limits": dict.fromkeys(["max_source_rows", "max_fragments", "max_object_bytes",
                                     "max_total_bytes", "max_output_rows"], 1000),
            "metadata_limits": dict.fromkeys(["max_catalog_tables", "max_catalog_fragments",
                                              "max_catalog_bytes", "max_fragments", "max_partitions",
                                              "max_tickers", "max_scan_rows", "max_object_bytes",
                                              "max_total_object_bytes", "max_batch_rows",
                                              "max_retained_rows", "max_retained_metadata_bytes",
                                              "max_proof_bytes"], 100)}


def _binding():
    panel = _panel()
    return {"size_panel_request": _panel_request(),
            "size_panel": {"context": panel["context"], "feature_snapshot": panel["feature_snapshot"]}}


@pytest.fixture
def counting_read(monkeypatch):
    calls = {"n": 0}

    def fake_read(repository, source, feature, *, columns=None, lineage=True, **request):
        calls["n"] += 1
        return _panel()

    monkeypatch.setattr("experiments.size_panel.read", fake_read)
    return calls


def test_five_calls_with_same_key_read_once(counting_read):
    repo, source, binding = _Repo(), _snapshot("snap-1"), _binding()
    for _ in range(5):
        _stored_panel(repo, source, binding, ["a", "b"])
    assert counting_read["n"] == 1


def test_tampered_binding_size_panel_refuses(counting_read):
    repo, source, binding = _Repo(), _snapshot("snap-2"), _binding()
    _stored_panel(repo, source, binding, ["a", "b"])
    binding["size_panel"]["feature_snapshot"] = {"feature_snapshot_id": "forged"}
    with pytest.raises(OpsError) as error:
        _stored_panel(repo, source, binding, ["a", "b"])
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert counting_read["n"] == 1


def test_tampered_memoised_panel_refuses_without_reread(counting_read):
    repo, source, binding = _Repo(), _snapshot("snap-3"), _binding()
    _stored_panel(repo, source, binding, ["a", "b"])
    next(iter(binding_mod._PANEL_MEMO.values()))["panel"]["context"]["target_events"][0]["event_id"] = "poisoned"
    with pytest.raises(OpsError) as error:
        _stored_panel(repo, source, binding, ["a", "b"])
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert counting_read["n"] == 1


def test_different_key_triggers_second_read(counting_read):
    repo, source, binding = _Repo(), _snapshot("snap-4"), _binding()
    _stored_panel(repo, source, binding, ["a", "b"])
    _stored_panel(repo, source, binding, ["a", "b", "c"])
    assert counting_read["n"] == 2
    assert len(binding_mod._PANEL_MEMO) == 1


def test_warm_memo_still_type_validates_horizons(counting_read):
    repo, source, binding = _Repo(), _snapshot("snap-4b"), _binding()
    _stored_panel(repo, source, binding, ["a", "b"])
    binding["size_panel_request"]["horizons"] = [1.0, 3, 5]
    with pytest.raises(OpsError) as error:
        _stored_panel(repo, source, binding, ["a", "b"])
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert counting_read["n"] == 1


def test_returned_frame_is_a_private_copy(counting_read):
    repo, source, binding = _Repo(), _snapshot("snap-5"), _binding()
    first = _stored_panel(repo, source, binding, ["a", "b"])
    pristine = _panel()
    first["rows"].append({"event_id": "evil"})
    first["context"]["target_events"] = []
    assert _stored_panel(repo, source, binding, ["a", "b"]) == pristine
    assert counting_read["n"] == 1


def test_fold_inputs_reuses_panel_once_per_process(counting_read, monkeypatch):
    monkeypatch.setattr(binding_mod, "verify_installation", lambda binding: None)
    fold = {"training_candidate_event_ids": ["t1"], "evaluation_event_ids": ["e1"],
            "training_labels_before": "2024-01-01"}
    monkeypatch.setattr(binding_mod, "_folds", lambda binding, request: [{"outer": {}, "inner": [fold]}])
    monkeypatch.setattr(binding_mod, "preflight_magnitude_targets", lambda *a, **k: {"preflight_id": "pf"})
    train = pd.DataFrame([{"event_id": "t1", "absolute_move_pct": 2.0}])
    train.attrs.update({"inputs_id": "train"})
    monkeypatch.setattr(binding_mod, "qualify_magnitude_training_targets", lambda *a, **k: train)
    panel = _panel()
    binding = {"size_request": {"schema_version": "magnitude_search_request.v4", "inner_years": {}}, "snapshot": to_document(_snapshot("snap-6")),
               "size_panel_request": _panel_request(),
               "size_panel": {"context": panel["context"], "feature_snapshot": panel["feature_snapshot"]},
               "magnitude_execution": {"evaluation_labels_before": "2024-01-01"},
               "holdouts": {"holdout_as_of_month": "2024-02"}}
    repo = _Repo()
    for _ in range(5):
        inputs = binding_mod.fold_inputs(repo, binding, fold)
    assert counting_read["n"] == 1
    assert inputs["stored_inputs_id"] == content_hash(binding["size_panel"])
