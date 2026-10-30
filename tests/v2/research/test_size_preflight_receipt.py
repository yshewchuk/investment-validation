"""SIZE preflight receipt: deterministic issue, cheap admit refusals, CLI receipt wiring."""
import dataclasses
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import canonical_json, from_document, to_document
from engine.v2.ops.errors import OpsError, fail
from experiments import eligible_history, size_panel, size_preflight_receipt
from experiments import size_search_binding as binding
from experiments.native_registration import MAX_NATIVE_EVIDENCE_BYTES
from experiments.size_preflight_receipt import SCHEMA, admit, input_bindings, issue
from tests.v2.research.test_magnitude_search_request import request as request_fixture
from tests.v2.research.test_size_cli_admission import setup_cli
from tests.v2.research.test_size_review_contract import panel_request
from tests.v2.research.test_size_search_integration import case  # noqa: F401


def bindings_for(case, **overrides):
    return input_bindings(case.repository, case.spec, case.request, case.panel_request,
                          **{**case.args, **overrides})


def test_register_with_matching_receipt_does_no_heavy_reads(case, monkeypatch):
    registration, spec = binding.prepare(case.repository, case.spec, case.request,
                                         case.panel_request, **case.args)
    bindings = input_bindings(case.repository, case.spec, case.request, case.panel_request,
                              **case.args)
    ref = issue(case.store, bindings, registration)

    def forbidden(*args, **kwargs):
        pytest.fail("admit reached a heavy or repository read")

    for module, name in ((binding, "prepare"), (binding, "_read_panel"), (binding, "fold_inputs"),
                         (binding, "preflight_magnitude_targets"),
                         (binding, "qualify_magnitude_training_targets"),
                         (size_panel, "read"), (size_panel, "materialize"),
                         (eligible_history, "materialize")):
        monkeypatch.setattr(module, name, forbidden)
    for name in ("resolve", "resolve_pinned", "fragment_records", "table_contract"):
        monkeypatch.setattr(type(case.repository), name, forbidden)
    admitted, prepared = admit(case.store, ref, bindings, spec)
    assert admitted.binding_json == registration.binding_json
    assert admitted.run_id == registration.run_id
    assert prepared.spec_hash == spec.spec_hash


def test_issue_is_deterministic(case):
    bindings = bindings_for(case)
    first = issue(case.store, bindings, case.registration)
    second = issue(case.store, bindings, case.registration)
    assert first == second


def test_issue_refuses_when_scope_snapshot_moved(case, monkeypatch):
    bindings = deepcopy(bindings_for(case))
    bindings["scope_snapshot"]["snapshot_id"] = "moved"
    monkeypatch.setattr(case.store, "publish_bytes",
                        lambda *args, **kwargs: pytest.fail("issue published after moved scope"))
    with pytest.raises(OpsError) as caught:
        issue(case.store, bindings, case.registration)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details == {"reason": "receipt_input_mismatch"}


def test_missing_receipt_refuses(case):
    bindings = bindings_for(case)
    ref = issue(case.store, bindings, case.registration)
    digest = "a" * 64
    missing = dataclasses.replace(ref, content_hash="sha256:" + digest,
                                  storage_key=f"objects/{digest[:2]}/{digest}")
    with pytest.raises(OpsError) as caught:
        admit(case.store, missing, bindings, case.spec)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.details == {"reason": "receipt_missing"}


@pytest.mark.parametrize("change", ["training_start", "labels_before", "scope",
                                    "request_limit", "panel_limit", "spec"])
def test_changed_input_refuses(case, change):
    bindings = bindings_for(case)
    ref = issue(case.store, bindings, case.registration)
    if change == "training_start":
        changed = bindings_for(case, training_start="2023-01-01")
    elif change == "labels_before":
        changed = bindings_for(case, labels_before="2024-06-30")
    elif change == "scope":
        changed = bindings_for(case, scope="panel-prices")
    elif change == "request_limit":
        request = deepcopy(case.request)
        request["limits"]["child"]["memory_bytes"] += 1
        changed = input_bindings(case.repository, case.spec, request, case.panel_request,
                                 **case.args)
    elif change == "panel_limit":
        panel = deepcopy(case.panel_request)
        panel["limits"]["max_source_rows"] += 1
        changed = input_bindings(case.repository, case.spec, case.request, panel, **case.args)
    else:
        changed = input_bindings(case.repository,
                                 dataclasses.replace(case.spec, hypothesis="changed receipt"),
                                 case.request, case.panel_request, **case.args)
    assert changed != bindings
    with pytest.raises(OpsError) as caught:
        admit(case.store, ref, changed, case.spec)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details == {"reason": "receipt_input_mismatch"}


def test_changed_experiment_id_is_input_mismatch(case):
    bindings = bindings_for(case)
    ref = issue(case.store, bindings, case.registration)
    changed = dataclasses.replace(case.spec, experiment_id=case.spec.experiment_id + "-x")
    changed_bindings = input_bindings(case.repository, changed, case.request,
                                      case.panel_request, **case.args)
    with pytest.raises(OpsError) as caught:
        admit(case.store, ref, changed_bindings, changed)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details == {"reason": "receipt_input_mismatch"}


def test_input_files_refused(case):
    bindings = bindings_for(case)
    ref = issue(case.store, bindings, case.registration)
    with pytest.raises(OpsError) as caught:
        admit(case.store, ref, bindings,
              dataclasses.replace(case.spec, input_files=("x.csv",)))
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_changed_validator_refuses(case, monkeypatch):
    bindings = bindings_for(case)
    ref = issue(case.store, bindings, case.registration)
    monkeypatch.setattr(size_preflight_receipt, "validator_identity",
                        lambda: {"source_closure": {}, "environment": {}})
    with pytest.raises(OpsError) as caught:
        admit(case.store, ref, bindings, case.spec)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details == {"reason": "receipt_validator_stale"}


@pytest.mark.parametrize("kind", ["not_json", "wrong_keys", "mutated_payload"])
def test_corrupt_receipt_refuses(case, kind):
    bindings = bindings_for(case)
    if kind == "not_json":
        ref = case.store.publish_bytes(b"not json", schema_ref=SCHEMA)
    elif kind == "wrong_keys":
        ref = case.store.publish_bytes(
            canonical_json({"schema_version": SCHEMA, "bindings": bindings}).encode("utf-8"),
            schema_ref=SCHEMA)
    else:
        compact = json.loads(case.registration.binding_json)
        compact["spec_hash"] = "sha256:" + "0" * 64
        payload = case.store.publish_bytes(canonical_json(compact).encode("utf-8"),
                                           schema_ref=binding.SCHEMA)
        receipt = {"schema_version": SCHEMA, "bindings": bindings,
                   "validator": binding.validator_identity(),
                   "run_id": case.registration.run_id, "registration": to_document(payload)}
        ref = case.store.publish_bytes(canonical_json(receipt).encode("utf-8"), schema_ref=SCHEMA)
    with pytest.raises(OpsError) as caught:
        admit(case.store, ref, bindings, case.spec)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details == {"reason": "receipt_corrupt"}


@pytest.mark.parametrize("code", ["RESOURCE_LIMIT_EXCEEDED", "RESOURCE_UNAVAILABLE",
                                  "EXPERIMENT_IDENTITY_CONFLICT"])
def test_hydration_refusals_keep_their_codes(case, monkeypatch, code):
    bindings = bindings_for(case)
    ref = issue(case.store, bindings, case.registration)

    def refuse(*args, **kwargs):
        raise fail(code, "x")

    monkeypatch.setattr(size_preflight_receipt, "load_registration_document", refuse)
    with pytest.raises(OpsError) as caught:
        admit(case.store, ref, bindings, case.spec)
    assert caught.value.code == code
    assert caught.value.problem.details.get("reason") != "receipt_corrupt"


@pytest.mark.parametrize("change", ["panel_request", "feature_snapshot_id"])
def test_payload_panel_not_bound_to_receipt_refuses(case, change):
    bindings = bindings_for(case)
    changed = deepcopy(bindings)
    if change == "panel_request":
        changed["panel_request"] = "sha256:" + "1" * 64
    else:
        changed["snapshots"]["feature_snapshot_id"]["snapshot_id"] = "sha256:" + "1" * 64
    ref = issue(case.store, changed, case.registration)
    with pytest.raises(OpsError) as caught:
        admit(case.store, ref, changed, case.spec)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details == {"reason": "receipt_corrupt"}


def test_payload_request_not_bound_to_receipt_refuses(case):
    bindings = bindings_for(case)
    changed = deepcopy(bindings)
    changed["size_request"] = "sha256:" + "2" * 64
    ref = issue(case.store, changed, case.registration)
    with pytest.raises(OpsError) as caught:
        admit(case.store, ref, changed, case.spec)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details == {"reason": "receipt_corrupt"}


@pytest.mark.parametrize("run_id", ["native_forged", 5])
def test_forged_run_id_refuses(case, run_id):
    bindings = bindings_for(case)
    ref = issue(case.store, bindings, case.registration)
    receipt = json.loads(case.store.read_verified(ref))
    receipt["run_id"] = run_id
    forged = case.store.publish_bytes(canonical_json(receipt).encode("utf-8"), schema_ref=SCHEMA)
    with pytest.raises(OpsError) as caught:
        admit(case.store, forged, bindings, case.spec)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details == {"reason": "receipt_corrupt"}


def test_oversized_payload_ref_refused_before_read(case, monkeypatch):
    bindings = bindings_for(case)
    ref = issue(case.store, bindings, case.registration)
    receipt = json.loads(case.store.read_verified(ref))
    payload = from_document(ArtifactRef, receipt["registration"])
    oversized_payload = dataclasses.replace(payload, byte_size=MAX_NATIVE_EVIDENCE_BYTES + 1)
    receipt["registration"] = to_document(oversized_payload)
    oversized = case.store.publish_bytes(canonical_json(receipt).encode("utf-8"), schema_ref=SCHEMA)
    read_verified = case.store.read_verified

    def guarded(read_ref):
        if read_ref == oversized_payload:
            pytest.fail("oversized payload ref was read before its size check")
        return read_verified(read_ref)

    monkeypatch.setattr(case.store, "read_verified", guarded)
    with pytest.raises(OpsError) as caught:
        admit(case.store, oversized, bindings, case.spec)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details == {"reason": "receipt_corrupt"}


def test_register_without_receipt_ref_exits_two():
    from experiments import size_search as cli

    with pytest.raises(SystemExit) as caught:
        cli.main(["register", "--spec", "spec", "--request", "request",
                  "--catalog", "catalog", "--store-root", "/tmp", "--no-ledger",
                  "--panel-request", "panel", "--scope", "development",
                  "--training-start", "2010-01-01", "--labels-before", "2025-01-01"])
    assert caught.value.code == 2


def test_preflight_result_carries_receipt_ref(monkeypatch, tmp_path, capsys):
    from experiments import size_search as cli

    store_root = tmp_path / "store"
    store_root.mkdir()
    setup_cli(monkeypatch)
    monkeypatch.setattr(cli, "_read_document",
                        lambda path: request_fixture() if path == "request"
                        else panel_request() if path == "panel" else {})
    monkeypatch.setattr(cli, "input_bindings",
                        lambda *args, **kwargs: {"scope_snapshot": {"snapshot_id": "dev"}})
    monkeypatch.setattr(cli, "prepare", lambda *args, **kwargs: (
        SimpleNamespace(binding_json=b"{}", run_id="synthetic-run",
                        document={"snapshot": {"snapshot_id": "dev"}}), object()))
    assert cli.main(["preflight", "--spec", "spec", "--request", "request",
                     "--catalog", "catalog", "--store-root", str(store_root), "--no-ledger",
                     "--panel-request", "panel", "--scope", "development",
                     "--training-start", "2010-01-01", "--labels-before", "2025-01-01"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["receipt_ref"]["schema_ref"] == SCHEMA


def test_register_missing_receipt_file_is_receipt_missing(monkeypatch, tmp_path, capsys):
    from experiments import size_search as cli

    store_root = tmp_path / "store"
    store_root.mkdir()
    setup_cli(monkeypatch)

    def read_document(path):
        if path == "receipt":
            raise FileNotFoundError(path)
        if path == "request":
            return request_fixture()
        if path == "panel":
            return panel_request()
        return {}

    monkeypatch.setattr(cli, "_read_document", read_document)
    monkeypatch.setattr(cli, "input_bindings", lambda *args, **kwargs: {})
    assert cli.main(["register", "--spec", "spec", "--request", "request",
                     "--catalog", "catalog", "--store-root", str(store_root), "--no-ledger",
                     "--panel-request", "panel", "--scope", "development",
                     "--training-start", "2010-01-01", "--labels-before", "2025-01-01",
                     "--receipt-ref", "receipt"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["refused"] == "INVALID_EXPERIMENT_SPEC"
    assert "receipt is missing" in result["message"]


def test_validator_identity_covers_receipt_module():
    sources = binding.validator_identity()["source_closure"]
    assert "experiments/size_preflight_receipt.py" in sources
    assert "experiments/size_search.py" in sources