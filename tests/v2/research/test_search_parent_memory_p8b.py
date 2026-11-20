"""Search parent memory P8b: compact worker registration and the parent-sealed plan."""
import json
from unittest.mock import Mock

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import ArtifactError, canonical_json, content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_trials as trials_mod
from experiments import v2_magnitude_search_run as run_mod
from experiments.native_registration import NativeRegistration

V17 = "native_experiment_registration.v1.7"
COMPACT = {"schema_version": V17, "magnitude_execution": {"small": 1},
           "regression_plan": {"target_recipe": {"recipe": "r"}, "events": {"$size_artifact": "h1"}}}
BINDING = json.dumps(COMPACT, sort_keys=True).encode()
FULL = {"schema_version": V17, "magnitude_execution": {"small": 1},
        "regression_plan": {"target_recipe": {"recipe": "r"}, "events": [{"e": 1}]}}
PLAN = {"plan_id": "p1", "folds": []}
REF = ArtifactRef(artifact_id="a", content_hash="sha256:" + "0" * 64, schema_ref="s",
                  byte_size=1, storage_key="k")


class FakeStore:
    def __init__(self, plan_bytes=None, error=None):
        self.root, self.plan_bytes, self.error = "/unused", plan_bytes, error

    def read_verified(self, ref):
        if self.error is not None:
            raise self.error
        return self.plan_bytes


def test_lean_view_skips_hydration_and_keeps_small_sections():
    hydrate = Mock(return_value=FULL)
    registration = NativeRegistration("run", BINDING, hydrate=hydrate)
    assert registration.lean == {"schema_version": V17, "magnitude_execution": {"small": 1},
                                 "regression_plan": {"target_recipe": {"recipe": "r"}}}
    hydrate.assert_not_called()


def test_lean_view_keeps_snapshot_and_holdouts_without_hydration():
    full = {"schema_version": V17, "magnitude_execution": {"small": 1},
            "snapshot": {"snapshot_id": "s1"}, "target_snapshot": {"snapshot_id": "t1"},
            "holdouts": {"holdout_as_of_month": "2020-01"}}
    binding = json.dumps(full, sort_keys=True).encode()
    hydrate = Mock(return_value=FULL)
    registration = NativeRegistration("run", binding, hydrate=hydrate)
    assert registration.lean["snapshot"] == {"snapshot_id": "s1"}
    assert registration.lean["target_snapshot"] == {"snapshot_id": "t1"}
    assert registration.lean["holdouts"] == {"holdout_as_of_month": "2020-01"}
    hydrate.assert_not_called()


def test_variant_id_matches_old_v17_document_rule():
    new = NativeRegistration("run", BINDING, hydrate=Mock(return_value=FULL))
    old = NativeRegistration("run", BINDING, FULL)
    assert new.variant_id == old.variant_id == content_hash(json.loads(BINDING))


def test_document_hydrates_on_each_access():
    hydrate = Mock(return_value=FULL)
    registration = NativeRegistration("run", BINDING, hydrate=hydrate)
    assert registration.document == FULL
    assert registration.document == FULL
    assert hydrate.call_count == 2


def test_placeholder_in_lean_path_falls_back_to_hydration():
    compact = json.dumps({"schema_version": V17, "magnitude_execution": {"$size_artifact": "h2"}}).encode()
    hydrate = Mock(return_value=FULL)
    registration = NativeRegistration("run", compact, hydrate=hydrate)
    assert registration.lean == FULL
    hydrate.assert_called_once()


def _patch_worker(monkeypatch, *, binding, store):
    monkeypatch.setattr(run_mod, "_PARALLEL", None)
    monkeypatch.setattr(run_mod, "_WORKER_WATCHDOG_STOP", None)
    monkeypatch.setattr(run_mod, "_WORKER_TERMINATING", False)
    monkeypatch.setattr(run_mod, "connect", lambda *a, **k: Mock())
    monkeypatch.setattr(run_mod, "ArtifactStore", lambda root: store)
    monkeypatch.setattr(run_mod, "verified_registration_bytes", lambda *a, **k: binding)
    monkeypatch.setattr(run_mod, "Repository", Mock())
    monkeypatch.setattr(run_mod, "SearchTrials", Mock())
    monkeypatch.setattr(run_mod, "start_rss_watchdog", lambda memory_bytes: (lambda: None))
    monkeypatch.setattr(run_mod.signal, "signal", Mock())
    monkeypatch.setattr(run_mod, "load_registration_document", Mock(side_effect=AssertionError("hydrated")))
    monkeypatch.setattr(run_mod, "read_search_plan", Mock(side_effect=AssertionError("rebuilt")))


def test_worker_uses_stored_plan_and_compact_registration(monkeypatch):
    _patch_worker(monkeypatch, binding=BINDING, store=FakeStore(canonical_json(PLAN).encode()))
    run_mod._init_trial_worker("c", "s", "native_x", {}, REF, 1)
    assert isinstance(run_mod._PARALLEL, tuple)
    _, _, trials, registration, plan = run_mod._PARALLEL
    assert isinstance(registration, NativeRegistration)
    assert plan == PLAN
    assert trials.trusted_plan == PLAN
    assert run_mod.load_registration_document.call_count == 0
    assert run_mod.read_search_plan.call_count == 0


def test_worker_clears_shard_rows_and_hydrates_without_them(monkeypatch):
    captured = {}

    def verified(conn, store, run_id, *, max_evidence_bytes, verified_size_rows):
        captured["rows"] = verified_size_rows
        verified_size_rows["h1"] = {"row": 1}
        return BINDING

    _patch_worker(monkeypatch, binding=BINDING, store=FakeStore(canonical_json(PLAN).encode()))
    monkeypatch.setattr(run_mod, "verified_registration_bytes", verified)
    loader = Mock(return_value=FULL)
    monkeypatch.setattr(run_mod, "load_registration_document", loader)
    run_mod._init_trial_worker("c", "s", "native_x", {}, REF, 1)
    assert captured["rows"] == {}
    _, _, _, registration, _ = run_mod._PARALLEL
    assert registration.document == FULL
    loader.assert_called_once()
    assert loader.call_args.args[1] == BINDING
    assert "verified_size_rows" not in loader.call_args.kwargs


def test_worker_refuses_missing_registration(monkeypatch):
    _patch_worker(monkeypatch, binding=None, store=FakeStore(canonical_json(PLAN).encode()))
    run_mod._init_trial_worker("c", "s", "native_x", {}, REF, 1)
    assert set(run_mod._PARALLEL) == {"worker_problem", "worker_error_type"}
    assert run_mod._PARALLEL["worker_problem"]["code"] == "INVALID_EXPERIMENT_SPEC"


def test_worker_refuses_tampered_stored_plan(monkeypatch):
    store = FakeStore(error=ArtifactError("INTEGRITY_FAILED", "x"))
    _patch_worker(monkeypatch, binding=BINDING, store=store)
    run_mod._init_trial_worker("c", "s", "native_x", {}, REF, 1)
    assert set(run_mod._PARALLEL) == {"worker_problem", "worker_error_type"}
    assert run_mod._PARALLEL["worker_problem"]["code"] == "EXPERIMENT_IDENTITY_CONFLICT"


@pytest.mark.parametrize("ref", ["not-a-ref", None, {"artifact_id": "a"}])
def test_stored_plan_refuses_non_artifact_reference(ref):
    with pytest.raises(OpsError) as raised:
        run_mod._stored_plan(FakeStore(canonical_json(PLAN).encode()), ref)
    assert raised.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


@pytest.mark.parametrize(("code", "expected"), [
    ("INTEGRITY_FAILED", "EXPERIMENT_IDENTITY_CONFLICT"),
    ("NOT_FOUND", "RESOURCE_UNAVAILABLE"),
])
def test_stored_plan_maps_artifact_error_from_read_verified(code, expected):
    with pytest.raises(OpsError) as raised:
        run_mod._stored_plan(FakeStore(error=ArtifactError(code, "x")), REF)
    assert raised.value.code == expected


def test_stored_plan_refuses_oversize_reference_before_read():
    oversize = ArtifactRef(artifact_id="a", content_hash="sha256:" + "0" * 64, schema_ref="s",
                           byte_size=run_mod.MAX_REPORT_BYTES + 1, storage_key="k")
    store = Mock()
    store.read_verified = Mock(side_effect=AssertionError("read"))
    with pytest.raises(OpsError) as raised:
        run_mod._stored_plan(store, oversize)
    assert raised.value.code == "RESOURCE_LIMIT_EXCEEDED"
    store.read_verified.assert_not_called()


def test_stored_plan_refuses_non_json_bytes():
    with pytest.raises(OpsError) as raised:
        run_mod._stored_plan(FakeStore(b"{not json"), REF)
    assert raised.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


@pytest.mark.parametrize("payload", [b'{"plan_id": 1}', b'{"folds": []}', b"[]"])
def test_stored_plan_refuses_json_without_string_plan_id(payload):
    with pytest.raises(OpsError) as raised:
        run_mod._stored_plan(FakeStore(payload), REF)
    assert raised.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def _trials():
    return trials_mod.SearchTrials(None, None, None, None, {}, REF)


def test_binding_uses_trusted_plan_without_rebuilding(monkeypatch):
    monkeypatch.setattr(trials_mod, "source_closure", lambda *a, **k: [])
    monkeypatch.setattr(trials_mod, "read_search_plan", Mock(side_effect=AssertionError("rebuilt")))
    trials = _trials()
    trials.trusted_plan = PLAN
    plan, binding = trials._binding()
    assert plan is PLAN
    assert binding["plan_id"] == "p1"


def test_binding_rebuilds_plan_without_trusted_plan(monkeypatch):
    reader = Mock(return_value={"plan_id": "p2"})
    monkeypatch.setattr(trials_mod, "source_closure", lambda *a, **k: [])
    monkeypatch.setattr(trials_mod, "read_search_plan", reader)
    plan, _ = _trials()._binding()
    reader.assert_called_once()
    assert plan == {"plan_id": "p2"}


def test_admit_refuses_seal_that_differs_from_trusted_plan(monkeypatch):
    monkeypatch.setattr(trials_mod, "source_closure", lambda *a, **k: [])
    monkeypatch.setattr(trials_mod, "require_current", lambda *a, **k: None)
    trials = _trials()
    trials.trusted_plan = PLAN
    trials._read = lambda slot: {"plan_id": "other"}
    with pytest.raises(OpsError) as raised:
        trials._admit()
    assert raised.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_admit_returns_plan_when_seal_matches_trusted_plan(monkeypatch):
    monkeypatch.setattr(trials_mod, "source_closure", lambda *a, **k: [])
    monkeypatch.setattr(trials_mod, "require_current", lambda *a, **k: None)
    trials = _trials()
    trials.trusted_plan = PLAN
    _, binding = trials._binding()
    trials._read = lambda slot: binding
    assert trials._admit() is PLAN


def test_worker_setup_maps_registration_tamper_to_identity_conflict(monkeypatch):
    def tampered(*a, **k):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "x")

    _patch_worker(monkeypatch, binding=BINDING, store=FakeStore(canonical_json(PLAN).encode()))
    monkeypatch.setattr(run_mod, "verified_registration_bytes", tampered)
    run_mod._init_trial_worker("c", "s", "native_x", {}, REF, 1)
    assert set(run_mod._PARALLEL) == {"worker_problem", "worker_error_type"}
    assert run_mod._PARALLEL["worker_problem"]["code"] == "EXPERIMENT_IDENTITY_CONFLICT"
