"""Seal execution declarations over synthetic metadata without reading any labels."""

from dataclasses import replace

import pytest

from engine.v2.foundation import artifact_reference, canonical_json, content_hash, to_document
from engine.v2.foundation.final_holdout import request_from_document
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import resolve_experiment_plan
from experiments import magnitude_inputs as inputs
from experiments import native_outcomes as outcomes
from experiments import native_registration as native
from tests.data_scan_support import commit_tables
from tests.ops_support import FakeClock
from tests.v2.research.test_magnitude_folds import _plan
from tests.v2.research.test_magnitude_registration import (
    _call,
    _declaration,
    _forbid,
    _refuses,
    _rewrite,
)
from tests.v2.research.test_magnitude_registration import source as source
from tests.v2.research.test_native_outcomes import _clock, _rows
from tests.v2.research.test_native_registration import _catalog, _evidence, _objects

LABELS_BEFORE = "2025-02-01"
EXECUTION = {
    "schema_version": "magnitude_execution.v1",
    "evaluation_labels_before": LABELS_BEFORE,
    "training_labels": "omit_skipped_unavailable_or_late_before_numeric_parse",
    "evaluation_labels": "complete_finite_strictly_before_cutoff",
    "model": "fold_local_arithmetic_mean",
    "features": ["constant_intercept"],
    "primary_metric": "event_pooled_rmse",
    "secondary_metrics": ["mae"],
    "quantile_calibration": "not_implemented",
}
BAD_CUTOFFS = (
    False, 20250201,  # Non-string wire values.
    "20250201", "2025-2-01",  # Parseable but noncanonical days.
    "2025-02-29", "2025-13-01",  # Impossible calendar dates.
    "2025-02-01T00:00:00", "2025-02-01T00:00:00Z",  # Timestamps.
    " 2025-02-01", "2025-02-01 ",  # Padded values.
    "２０２５-０２-０１", "2025-٠٢-01",  # Non-ASCII digits.
)


def _execution_call(source, **changes):
    """Exercise the public registrar with an explicit evaluation-label clock."""
    return _call(source, **{"magnitude_labels_before": LABELS_BEFORE, **changes})


def _unchanged(source, before, objects):
    assert _catalog(source.conn) == before
    assert _objects(source.store) == objects


class _NoLedgerAccess:
    def __fspath__(self):
        raise AssertionError("no-ledger calls must never inspect the ledger destination")


def test_execution_policy_is_literal_sealed_and_metadata_only(source, monkeypatch):
    """The shared fixture poisons physical outcome verification and outcome planning."""
    expected_plan = _plan(source.repository, source.snapshot)
    source.repository.scans.clear()
    before, objects = _catalog(source.conn), _objects(source.store)
    registered = _execution_call(source)
    document = registered.document
    assert document["schema_version"] == "native_experiment_registration.v1.3"
    assert document["magnitude_execution"] == EXECUTION
    assert document["magnitude_plan"] == expected_plan
    assert document["magnitude_request"] == {
        "training_start": "2023-12-10", "evaluation_years": ["2024"]}
    assert document["event_ids"] == sorted(event["event_id"] for event in source.events[:2])
    assert registered.binding_json == canonical_json(document).encode()
    assert registered.variant_id == content_hash(document)
    assert source.repository.pins == ["shadow"]
    assert {name for name, _ in source.repository.scans} == {"earnings_events", "daily_market"}
    fold = document["magnitude_plan"]["folds"][0]
    assert fold["training_labels_before"] == "2023-12-22"
    assert document["magnitude_execution"]["evaluation_labels_before"] != fold["training_labels_before"]
    after, published = _catalog(source.conn), _objects(source.store)
    changed = {table for table in before if before[table] != after[table]}
    assert changed == {"hypotheses", "experiment_runs"}
    assert all(len(after[table]) == len(before[table]) + 1 for table in changed)
    assert {path: published[path] for path in objects} == objects
    assert [published[path] for path in published.keys() - objects.keys()] == [registered.binding_json]
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    source.conn.execute("PRAGMA query_only=ON")
    assert _execution_call(source) == registered
    assert _execution_call(source, admit=True) == registered
    assert _execution_call(source, admit=True, expected=registered) == registered
    assert native.read_native_registration(source.conn, source.store, source.spec) == registered
    _unchanged(source, after, published)
    document["magnitude_execution"]["features"].append("unsealed")
    assert registered.document["magnitude_execution"] == EXECUTION


@pytest.mark.parametrize("cutoff", BAD_CUTOFFS)
@pytest.mark.parametrize("admit", [False, True])
def test_bad_execution_dates_refuse_before_repository_access(source, monkeypatch, cutoff, admit):
    before, objects = _catalog(source.conn), _objects(source.store)
    monkeypatch.setattr(source.repository, "resolve_pinned", _forbid)
    with _refuses(OpsError, "INVALID_EXPERIMENT_SPEC"):
        _execution_call(source, admit=admit, magnitude_labels_before=cutoff)
    _unchanged(source, before, objects)


@pytest.mark.parametrize("changes", [
    {"magnitude_training_start": None},
    {"final_read": _declaration()},
    {"spec_changes": {"seed": False}},
    {"spec_changes": {"seed": 1}},
    {"spec_changes": {"economic_params": {"fill": 0.5}}},
    {"spec_changes": {"price_source": "option_chains"}},
])
@pytest.mark.parametrize("admit", [False, True])
def test_invalid_execution_declarations_never_pin_or_publish(source, monkeypatch, changes, admit):
    changes = dict(changes)
    spec = replace(source.spec, **changes.pop("spec_changes", {}))
    before, objects = _catalog(source.conn), _objects(source.store)
    monkeypatch.setattr(source.repository, "resolve_pinned", _forbid)
    with _refuses(OpsError, "INVALID_EXPERIMENT_SPEC"):
        _execution_call(source, spec=spec, admit=admit, **changes)
    _unchanged(source, before, objects)


@pytest.mark.parametrize("cutoff", ["2024-01-02", "2024-01-03", "2025-01-01", "2025-01-02"])
def test_cutoff_must_follow_every_evaluation_event_after_fresh_planning(source, monkeypatch, cutoff):
    """Later-fold events count too; all dates are canonical and no reservation exists."""
    before, objects = _catalog(source.conn), _objects(source.store)
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    with _refuses(OpsError, "INVALID_EXPERIMENT_SPEC"):
        _execution_call(source, spec=replace(source.spec, folds=("2024", "2025")),
                        magnitude_labels_before=cutoff)
    assert {name for name, _ in source.repository.scans} == {"earnings_events", "daily_market"}
    _unchanged(source, before, objects)


@pytest.mark.parametrize("change", ["add", "omit-cutoff", "omit-both", "change-cutoff"])
def test_original_reservation_owns_execution_mode_before_metadata_scans(source, monkeypatch, change):
    if change == "add":
        _call(source)
        changes = {}
    else:
        _execution_call(source)
        changes = {
            "omit-cutoff": {"magnitude_labels_before": None},
            "omit-both": {"magnitude_labels_before": None, "magnitude_training_start": None},
            "change-cutoff": {"magnitude_labels_before": "2025-02-02"},
        }[change]
    before, objects = _catalog(source.conn), _objects(source.store)
    monkeypatch.setattr(source.repository, "scan", _forbid)
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    with _refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        if change == "omit-both":
            native.require_native_registration(source.conn, source.store, source.repository, source.spec,
                code_root=source.code_root, as_of_month="2026-01")
        else:
            _execution_call(source, admit=True, **changes)
    _unchanged(source, before, objects)


def _forged_refuses(source, registered, document, monkeypatch):
    """Repair artifact/catalog identities before testing the independent wire guard."""
    forged = _rewrite(source, registered, document)
    before, objects = _catalog(source.conn), _objects(source.store)
    monkeypatch.setattr(source.repository, "scan", _forbid)
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    with _refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        native.verify_native_registration(source.conn, source.store, forged)
    with _refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        _execution_call(source, admit=True)
    _unchanged(source, before, objects)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "magnitude_execution.v2"),
    ("training_labels", "parse_before_availability_filter"),
    ("evaluation_labels", "finite_available_on_or_before_cutoff"),
    ("model", "global_arithmetic_mean"),
    ("features", ["market_feature"]),
    ("primary_metric", "fold_mean_rmse"),
    ("secondary_metrics", ["mse"]),
    ("quantile_calibration", "implemented"),
    ("unknown", True),
    *(("evaluation_labels_before", value) for value in (*BAD_CUTOFFS, None)),
])
def test_rehashed_policy_and_cutoff_changes_have_independent_wire_oracles(
        source, monkeypatch, field, value):
    registered = _execution_call(source)
    document = registered.document
    document["magnitude_execution"][field] = value
    assert document["magnitude_plan"] == registered.document["magnitude_plan"]
    _forged_refuses(source, registered, document, monkeypatch)


@pytest.mark.parametrize("kind", ["missing", "null", "list", "empty", "missing-field", "final"])
def test_rehashed_v13_requires_complete_execution_and_no_final_hybrid(source, monkeypatch, kind):
    registered = _execution_call(source)
    document = registered.document
    if kind == "missing":
        document.pop("magnitude_execution")
    elif kind == "missing-field":
        document["magnitude_execution"].pop("model")
    elif kind == "final":
        document["final_read"] = to_document(_declaration())
    else:
        document["magnitude_execution"] = {"null": None, "list": [], "empty": {}}[kind]
    _forged_refuses(source, registered, document, monkeypatch)


@pytest.mark.parametrize("field,value", [("seed", False), ("seed", 1), ("economic_params", {"fill": 0.5})])
def test_rehashed_execution_spec_guard_repairs_detached_spec_and_catalog_links(source, monkeypatch, field, value):
    registered = _execution_call(source)
    document = registered.document
    document["execution_plan"][field] = value
    spec_hash = replace(source.spec, **{field: value}).spec_hash
    document["spec_hash"] = spec_hash
    for table in ("hypotheses", "experiment_runs"):
        source.conn.execute(f"UPDATE {table} SET spec_hash=? WHERE run_id=?", (spec_hash, registered.run_id))
    _forged_refuses(source, registered, document, monkeypatch)


@pytest.mark.parametrize("cutoff", ["2025-01-01", "2025-01-02"])
def test_rehashed_valid_date_cannot_precede_or_equal_later_fold_event(source, monkeypatch, cutoff):
    registered = _execution_call(source, spec=replace(source.spec, folds=("2024", "2025")))
    document = registered.document
    document["magnitude_execution"]["evaluation_labels_before"] = cutoff
    assert document["magnitude_plan"] == registered.document["magnitude_plan"]
    _forged_refuses(source, registered, document, monkeypatch)


@pytest.mark.parametrize("version", [0, 1, 2])
def test_older_schemas_reject_rehashed_execution_field_hybrids(source, monkeypatch, version):
    registered = _call(source, magnitude_training_start="2023-12-10" if version == 2 else None,
                       final_read=_declaration() if version == 1 else None)
    document = registered.document
    assert document["schema_version"] == f"native_experiment_registration.v1.{version}"
    document["magnitude_execution"] = dict(EXECUTION)
    _forged_refuses(source, registered, document, monkeypatch)


@pytest.mark.parametrize("version", [0, 1, 2])
def test_omitting_execution_preserves_actual_older_schema_bytes(source, monkeypatch, version):
    """A source-only fixed runner and pinned environment isolate the optional field."""
    environment = {"synthetic_numeric_environment": "fixed"}
    monkeypatch.setattr(native, "environment_identity", lambda _: dict(environment))
    plan = _plan(source.repository, source.snapshot) if version == 2 else None
    declaration = _declaration() if version == 1 else None
    options = {"magnitude_training_start": "2023-12-10" if version == 2 else None,
               "final_read": declaration}
    expected = {
        "schema_version": f"native_experiment_registration.v1.{version}",
        "spec_hash": source.spec.spec_hash,
        "execution_plan": resolve_experiment_plan(source.spec).as_document(),
        "snapshot": to_document(source.snapshot), "scope": "shadow",
        "source_closure": native.source_closure(source.code_root, ["run.py"]),
        "environment": environment,
        "event_ids": sorted(event["event_id"] for event in source.events[:2 if version == 2 else 4]),
        "holdouts": {"holdout_as_of_month": "2026-01", "random_membership_version": "canonical-event-sha256.v1",
                     "rolling_membership_version": "calendar-months.v1"},
    }
    if declaration:
        expected["final_read"] = to_document(declaration)
    if plan:
        expected.update(magnitude_request={"training_start": "2023-12-10", "evaluation_years": ["2024"]},
                        magnitude_plan=plan)
    registered = _call(source, **options)
    assert registered.binding_json == canonical_json(expected).encode()
    assert registered.variant_id == content_hash(expected)
    before, objects = _catalog(source.conn), _objects(source.store)
    assert _call(source, magnitude_labels_before=None, **options) == registered
    assert _call(source, admit=True, **options) == registered
    _unchanged(source, before, objects)


def test_execution_registration_needs_no_computed_moves_table(source):
    tables = ("earnings_events", "daily_market")
    snapshot = commit_tables(source.conn, FakeClock(),
        {name: source.repository.fragment_records(source.snapshot, name) for name in tables},
        {name: source.repository.table_contract(source.snapshot, name) for name in tables},
        scope="execution-metadata-only", receipt_id="execution-metadata-only",
        attempt_id="execution-metadata-only", store=source.store)
    registered = _execution_call(source, scope="execution-metadata-only")
    assert registered.document["snapshot"] == to_document(snapshot)
    assert "computed_moves" not in registered.document["snapshot"]["table_versions"]
    before, objects = _catalog(source.conn), _objects(source.store)
    assert _execution_call(source, admit=True, scope="execution-metadata-only") == registered
    _unchanged(source, before, objects)


@pytest.mark.parametrize("no_ledger", [False, True])
@pytest.mark.parametrize("version", [0, 1, 3])
def test_valid_v13_and_older_outcome_contracts_publish_replay_and_export(source, monkeypatch, tmp_path, no_ledger, version):
    """Real catalog/store/CSV publication remains isolated from synthetic input labels."""
    registered = (_execution_call(source) if version == 3 else _call(source,
        magnitude_training_start=None, final_read=_declaration() if version == 1 else None))
    assert registered.document["schema_version"] == f"native_experiment_registration.v1.{version}"
    native.verify_native_registration(source.conn, source.store, registered)
    monkeypatch.setattr(source.repository, "scan", _forbid)
    monkeypatch.setattr(source.repository, "resolve_pinned", _forbid)
    _clock(monkeypatch, "2026-01-02")
    ledger = tmp_path / "synthetic-ledger" / "ledger.csv"
    options = {"no_ledger": no_ledger, "ledger_path": _NoLedgerAccess() if no_ledger else ledger}
    before, objects = _catalog(source.conn), _objects(source.store)
    assert outcomes.replay_native_outcome(source.conn, source.store, registered, **options) is None
    _unchanged(source, before, objects)
    assert not ledger.parent.exists()
    report = "Synthetic magnitude baseline report; no market-data computation.\n"
    result = outcomes.publish_native_outcome(source.conn, source.store, registered,
        report=report, attempted=True, **options)
    outcome = result["outcome"]
    assert outcome["run_id"] == registered.run_id
    assert outcome["variant_id"] == registered.variant_id
    assert outcome["attempted_variants"] == 1 and outcome["failure_code"] is None
    assert outcome["report_ref"] == to_document(artifact_reference(report.encode(), "native_experiment_report.v1.0"))
    assert outcome["ledger_row"] == {
        "id": source.spec.experiment_id, "spec_hash": registered.variant_id, "date": "2026-01-02",
        "stage": "ran", "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False"}
    slot = "native_smoke" if no_ledger else "native_outcome"
    assert _evidence(source, registered.run_id)[slot + "_receipt"] == result["receipt_ref"]
    assert result["receipt"]["recording_mode"] == ("smoke" if no_ledger else "recorded")
    if no_ledger:
        assert outcome["ledger_destination"] is None
        assert not ledger.parent.exists()
    else:
        assert outcome["ledger_destination"] == content_hash(str(ledger.resolve()))
        assert _rows(ledger) == [outcome["ledger_row"]]
    after, published = _catalog(source.conn), _objects(source.store)
    assert {key for key in before if before[key] != after[key]} == {"experiment_runs"}
    assert {path: published[path] for path in objects} == objects
    ledger_bytes = None if no_ledger else ledger.read_bytes()
    _clock(monkeypatch, "2026-02-10")
    assert outcomes.replay_native_outcome(source.conn, source.store, registered, **options) == result
    assert outcomes.publish_native_outcome(source.conn, source.store, registered,
        report=report, attempted=True, **options) == result
    path = outcomes.export_native_report(source.conn, source.store, registered, **options)
    assert path.read_text() == report
    assert not path.samefile(source.store.root / outcome["report_ref"]["storage_key"])
    assert outcomes.export_native_report(source.conn, source.store, registered, **options) == path
    _unchanged(source, after, published)
    assert (not ledger.parent.exists()) if no_ledger else ledger.read_bytes() == ledger_bytes


@pytest.mark.parametrize("operation", ["publish", "refusal", "replay", "export"])
@pytest.mark.parametrize("no_ledger", [False, True])
@pytest.mark.parametrize("kind", ["v1.2", "rehashed-v1.3-policy"])
def test_unexecutable_or_tampered_registration_refuses_before_result_access(
        source, monkeypatch, tmp_path, operation, no_ledger, kind):
    if kind == "v1.2":
        registered = _call(source)
    else:
        registered = _execution_call(source)
        document = registered.document
        document["magnitude_execution"]["model"] = "global_arithmetic_mean"
        registered = _rewrite(source, registered, document)
    before, objects = _catalog(source.conn), _objects(source.store)
    ledger = tmp_path / "never-created" / "ledger.csv"
    refusal = canonical_json({"schema_version": "native_experiment_refusal.v1.0",
        "failure_code": "EXPERIMENT_IDENTITY_CONFLICT", "request": {"run_id": registered.run_id},
        "variant_id": None}).encode()
    ref = artifact_reference(refusal, "native_experiment_refusal.v1.0")
    publish, read = source.store.publish_bytes, source.store.read_verified

    def private_only(payload, *, schema_ref):
        assert (payload, schema_ref) == (refusal, ref.schema_ref)
        return publish(payload, schema_ref=schema_ref)

    def registration_only(reference):
        assert reference.schema_ref == registered.document["schema_version"]
        return read(reference)

    monkeypatch.setattr(source.store, "publish_bytes", private_only)
    monkeypatch.setattr(source.store, "read_verified", registration_only)
    monkeypatch.setattr(source.repository, "scan", _forbid)
    for name in ("_evidence", "_reconcile", "_report_path", "_read", "ledger_append"):
        monkeypatch.setattr(outcomes, name, _forbid)
    method = {"publish": outcomes.publish_native_outcome, "refusal": outcomes.publish_native_outcome,
              "replay": outcomes.replay_native_outcome, "export": outcomes.export_native_report}[operation]
    options = ({"report": "Synthetic result must not publish", "attempted": True} if operation == "publish"
               else {"problem": fail("FEATURE_LOOKAHEAD", "synthetic").problem} if operation == "refusal" else {})
    with pytest.raises(OpsError) as caught:
        method(source.conn, source.store, registered, no_ledger=no_ledger,
               ledger_path=_NoLedgerAccess() if no_ledger else ledger, **options)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.retryable is False
    assert caught.value.problem.details == {"refusal_ref": to_document(ref)}
    assert read(ref) == refusal
    assert _catalog(source.conn) == before
    assert _objects(source.store) == {**objects, ref.storage_key: refusal}
    assert not ledger.parent.exists() and not (source.store.root / "native_reports").exists()


def test_v13_cannot_prepare_final_read(source, monkeypatch):
    registered = _execution_call(source)
    before, objects = _catalog(source.conn), _objects(source.store)
    monkeypatch.setattr(source.repository, "scan", _forbid)
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    with _refuses(OpsError, "INVALID_EXPERIMENT_SPEC"):
        native.prepare_final_holdout_request(source.conn, source.store, registered)
    _unchanged(source, before, objects)


def test_final_request_reference_remains_exclusive_to_v11(source, monkeypatch):
    registered = _call(source, magnitude_training_start=None, final_read=_declaration())
    request = native.prepare_final_holdout_request(source.conn, source.store, registered)
    ref = artifact_reference(registered.binding_json, "native_experiment_registration.v1.3")
    forged = replace(request, registration_ref=ref, variant_id=ref.content_hash)
    before, objects = _catalog(source.conn), _objects(source.store)
    monkeypatch.setattr(source.store, "read_verified", _forbid)
    with pytest.raises(ValueError, match="^invalid final-read artifact reference$"):
        request_from_document(to_document(forged))
    _unchanged(source, before, objects)
    assert source.conn.execute("SELECT COUNT(*) FROM final_holdout_authorizations").fetchone()[0] == 0


@pytest.mark.parametrize("kind,code", [
    ("unregistered", "INVALID_EXPERIMENT_SPEC"),
    ("v1.0", "EXPERIMENT_IDENTITY_CONFLICT"),
    ("v1.1", "EXPERIMENT_IDENTITY_CONFLICT"),
    ("v1.2", "EXPERIMENT_IDENTITY_CONFLICT"),
    ("changed-cutoff", "EXPERIMENT_IDENTITY_CONFLICT"),
    ("omitted-clock", "INVALID_EXPERIMENT_SPEC"),
])
def test_input_caller_refuses_unbound_or_changed_execution_before_reads(source, monkeypatch, kind, code):
    if kind.startswith("v1."):
        _call(source, magnitude_training_start="2023-12-10" if kind == "v1.2" else None,
              final_read=_declaration() if kind == "v1.1" else None)
    elif kind != "unregistered":
        _execution_call(source)
    before, objects = _catalog(source.conn), _objects(source.store)
    monkeypatch.setattr(source.repository, "scan", _forbid)
    monkeypatch.setattr(inputs, "preflight_magnitude_targets", _forbid)
    cutoff = None if kind == "omitted-clock" else "2025-02-02" if kind == "changed-cutoff" else LABELS_BEFORE
    with _refuses(OpsError, code):
        inputs.admit_magnitude_inputs(source.conn, source.store, source.repository, source.spec,
            code_root=source.code_root, as_of_month="2026-01",
            magnitude_training_start="2023-12-10", magnitude_labels_before=cutoff)
    _unchanged(source, before, objects)
