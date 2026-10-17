"""Native magnitude preregistration over real synthetic metadata, never outcomes."""
import inspect
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.contracts import FinalHoldoutDeclaration
from engine.v2.data.errors import DataError
from engine.v2.data.errors import fail as data_fail
from engine.v2.foundation import artifact_reference, canonical_json, content_hash, to_document
from engine.v2.foundation.final_holdout import request_from_document
from engine.v2.ops.errors import OpsError
from engine.v2.ops.errors import fail as ops_fail
from engine.v2.ops.experiments import resolve_experiment_plan
from engine.v2.research.magnitude_folds import magnitude_window
from experiments import native_outcomes as outcomes
from experiments import native_registration as native
from tests.data_scan_support import commit_tables
from tests.ops_support import FakeClock
from tests.v2.research import magnitude_trial_support as support
from tests.v2.research.magnitude_trial_support import (
    PlanRepository,
    catalog_rows,
    history_event,
)
from tests.v2.research.test_magnitude_folds import (
    forbid_fold_outcomes,
    magnitude_fold_fixture,
    plan_test_magnitude_folds,
)
from tests.v2.research.test_native_registration import artifact_object_bytes, native_spec


@pytest.fixture
def source(tmp_path, monkeypatch, request):
    """Commit protected, ambiguous and out-of-window siblings with poisoned labels."""
    train, test = history_event("2023-12-12"), {**history_event("2024-01-03"), "session": "BMO"}
    before, future = history_event("2023-11-01"), history_event("2025-01-02")
    random, rolling = history_event("2023-12-13", protected=True), history_event("2025-12-03")
    alias = history_event("2024-01-05")
    events = [train, test, before, future, random, rolling, alias,
              {**alias, "event_id": "ambiguous-sibling", "session": "BMO"}]
    events.extend(getattr(request, "param", ()))
    conn, original, snapshot = magnitude_fold_fixture(tmp_path, events)
    repository = PlanRepository(conn, original._store)
    forbid_fold_outcomes(monkeypatch, repository, snapshot)
    root = tmp_path / "code"
    root.mkdir()
    (root / "run.py").write_text("raise AssertionError('registration must not execute a runner')\n")
    value = SimpleNamespace(conn=conn, store=original._store, repository=repository,
        snapshot=snapshot, code_root=root, events=events,
        spec=native_spec(price_source="computed_moves", economic_params={}, seed=0, runner="run.py"))
    yield value
    conn.close()


def call_magnitude_registration(source, *, admit=False, spec=None, **changes):
    """Use the actual native registration and read-only admission entrypoints."""
    method = native.require_native_registration if admit else native.register_native
    return method(source.conn, source.store, source.repository, spec or source.spec,
        **{"code_root": source.code_root, "as_of_month": "2026-01",
           "magnitude_training_start": "2023-12-10", **changes})


@pytest.mark.parametrize("factory,code,other", [
    (ops_fail, "INVALID_EXPERIMENT_SPEC", "EXPERIMENT_IDENTITY_CONFLICT"),
    (data_fail, "CONTRACT_MISMATCH", "OBJECT_CORRUPT"),
])
@pytest.mark.parametrize("wrong_field", ["code", "retryability"])
def test_refusal_oracle_rejects_wrong_code_or_retryability(factory, code, other, wrong_field):
    """A misleading message or incorrectly retryable problem must fail the oracle."""
    error = factory(other if wrong_field == "code" else code, f"Misleading message names {code}; inspect the typed envelope")
    assert code in str(error)  # The old unanchored exception-text match would accept this.
    if wrong_field == "code":
        assert error.code != code
    if wrong_field == "retryability":
        error = type(error)(replace(error.problem, retryable=True))
    with pytest.raises(AssertionError):
        with support.refuses(type(error), code):
            raise error


def test_full_plan_is_sealed_and_idempotently_admitted(source, monkeypatch):
    """Bind the independently planned metadata and only its selected population."""
    expected_plan = plan_test_magnitude_folds(source.repository, source.snapshot)
    source.repository.scans.clear()
    registered = call_magnitude_registration(source)
    document = registered.document
    assert document == {"schema_version": "native_experiment_registration.v1.2",
        "spec_hash": source.spec.spec_hash, "execution_plan": resolve_experiment_plan(source.spec).as_document(),
        "snapshot": to_document(source.snapshot), "scope": "shadow",
        "source_closure": native.source_closure(source.code_root, ["run.py"]),
        "environment": native.environment_identity(1),
        "event_ids": sorted(event["event_id"] for event in source.events[:2]),
        "holdouts": {"holdout_as_of_month": "2026-01", "random_membership_version": "canonical-event-sha256.v1",
                     "rolling_membership_version": "calendar-months.v1"},
        "magnitude_request": {"training_start": "2023-12-10", "evaluation_years": ["2024"]},
        "magnitude_plan": expected_plan}
    assert registered.binding_json == canonical_json(document).encode()
    assert registered.variant_id == content_hash(document)
    assert document["magnitude_plan"]["folds"] == [{"year": "2024",
        "training_labels_before": "2023-12-22",
        "training_candidate_event_ids": [source.events[0]["event_id"]],
        "evaluation_event_ids": [source.events[1]["event_id"]]}]
    clocks = document["magnitude_plan"]["events"][1]["forecasts"]
    assert [(clock["horizon_sessions"], clock["pre_announcement_close"], clock["source_session_ceiling"])
            for clock in clocks] == [(5, "2024-01-02", "2023-12-22"),
                                     (3, "2024-01-02", "2023-12-27"), (1, "2024-01-02", "2023-12-29")]
    plan = document["magnitude_plan"]
    assert plan["plan_id"] == content_hash({key: value for key, value in plan.items() if key != "plan_id"})
    for table, field in (("earnings_events", "population_input"), ("daily_market", "calendar_input")):
        expected = {"table": table, "dataset_version": source.snapshot.table_versions[table].dataset_version_id,
            "columns": ["date"] if table == "daily_market" else [
                "event_id", "ticker", "event_date", "session", "date_conflict", "event_cluster_id"],
            "objects": [{"fragment_id": record.fragment_id, "object_id": record.object_ref.object_id,
                         "content_hash": record.object_ref.content_hash}
                        for record in source.repository.fragment_records(source.snapshot, table)]}
        if table == "daily_market":
            expected.update(snapshot_id=source.snapshot.snapshot_id, calendar_version=source.snapshot.calendar_version)
        assert plan[field] == expected
    assert source.repository.pins == ["shadow"]
    assert {name for name, _ in source.repository.scans} == {"earnings_events", "daily_market"}
    before, objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(source.store, "publish_bytes", support.forbid_access)
    source.conn.execute("PRAGMA query_only=ON")
    assert call_magnitude_registration(source, event_ids=list(reversed(document["event_ids"]))) == registered
    assert call_magnitude_registration(source, admit=True) == registered
    assert call_magnitude_registration(source, admit=True, expected=registered) == registered
    assert native.read_native_registration(source.conn, source.store, source.spec) == registered
    assert catalog_rows(source.conn) == before and artifact_object_bytes(source.store) == objects
    document["magnitude_request"]["evaluation_years"].clear()
    assert registered.document["magnitude_request"]["evaluation_years"] == ["2024"]
    with support.refuses(OpsError, "INVALID_EXPERIMENT_SPEC"):
        native.prepare_final_holdout_request(source.conn, source.store, registered)


def test_absent_outcomes_table_is_not_a_registration_dependency(source):
    """Use another real committed snapshot that carries only the metadata tables."""
    tables = ("earnings_events", "daily_market")
    snapshot = commit_tables(source.conn, FakeClock(),
        {name: source.repository.fragment_records(source.snapshot, name) for name in tables},
        {name: source.repository.table_contract(source.snapshot, name) for name in tables},
        scope="metadata-only", receipt_id="metadata-only", attempt_id="metadata-only", store=source.store)
    registered = call_magnitude_registration(source, scope="metadata-only")
    assert registered.document["snapshot"] == to_document(snapshot)
    assert "computed_moves" not in registered.document["snapshot"]["table_versions"]
    assert call_magnitude_registration(source, admit=True, scope="metadata-only") == registered


@pytest.mark.parametrize("changes", [
    {"magnitude_training_start": "2024-01-01"}, {"magnitude_training_start": "2023-02-30"},
    {"magnitude_training_start": True}, {"magnitude_training_start": []},
    {"valid_final_read": True}, {"spec_changes": {"price_source": "option_chains"}},
    {"spec_changes": {"folds": ("fold",)}}, {"spec_changes": {"folds": ("2025", "2024")}},
])
@pytest.mark.parametrize("admit", [False, True])
def test_bad_declarations_refuse_before_any_repository_access(source, monkeypatch, changes, admit):
    """Window and source validation precede pins, scans and registration effects."""
    changes = dict(changes)
    if changes.pop("valid_final_read", False):
        changes["final_read"] = final_holdout_declaration()
    spec = replace(source.spec, **changes.pop("spec_changes", {}))
    before, objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(source.repository, "resolve_pinned", support.forbid_access)
    with support.refuses(OpsError, "INVALID_EXPERIMENT_SPEC"):
        call_magnitude_registration(source, spec=spec, admit=admit, **changes)
    assert catalog_rows(source.conn) == before and artifact_object_bytes(source.store) == objects


@pytest.mark.parametrize("change", ["start", "years", "month", "scope", "snapshot", "source", "environment", "omitted", "expected"])
def test_identity_drift_refuses_before_metadata_scan(source, monkeypatch, change):
    """Stored identity is automatically checked even when expected is omitted."""
    registered = call_magnitude_registration(source)
    kwargs = {}
    if change == "start":
        kwargs["magnitude_training_start"] = "2023-12-11"
    elif change == "years":
        kwargs["spec"] = replace(source.spec, folds=("2024", "2025"))
    elif change == "month":
        kwargs["as_of_month"] = "2026-02"
    elif change == "scope":
        source.conn.execute("INSERT INTO data_snapshot_heads SELECT 'alternate', snapshot_id, generation, updated_at, update_receipt_ref "
                            "FROM data_snapshot_heads WHERE scope='shadow'")
        kwargs["scope"] = "alternate"
    elif change == "snapshot":
        names = ("earnings_events", "daily_market")
        snapshot = commit_tables(source.conn, FakeClock(),
            {name: source.repository.fragment_records(source.snapshot, name) for name in names},
            {name: source.repository.table_contract(source.snapshot, name) for name in names},
            scope="alternate", receipt_id="alternate", attempt_id="alternate", store=source.store)
        source.conn.execute("UPDATE data_snapshot_heads SET snapshot_id=?, generation=generation+1 WHERE scope='shadow'",
                            (snapshot.snapshot_id,))
    elif change == "source":
        (source.code_root / "run.py").write_text("raise AssertionError('changed source')\n")
    elif change == "environment":
        monkeypatch.setattr(native, "environment_identity", lambda _: {"changed": True})
    elif change == "omitted":
        kwargs["magnitude_training_start"] = None
    else:
        kwargs["expected"] = replace(registered, binding_json=b"{}")
    before, objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(source.repository, "scan", support.forbid_access)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        call_magnitude_registration(source, admit=True, **kwargs)
    assert catalog_rows(source.conn) == before and artifact_object_bytes(source.store) == objects


def test_missing_reservation_refuses_before_metadata_scan(source, monkeypatch):
    """Admission never substitutes fresh registration for a missing reservation."""
    monkeypatch.setattr(source.repository, "scan", support.forbid_access)
    with support.refuses(OpsError, "INVALID_EXPERIMENT_SPEC"):
        call_magnitude_registration(source, admit=True)


@pytest.mark.parametrize("declared", [False, True])
def test_adding_magnitude_mode_to_an_old_registration_refuses_before_scan(source, monkeypatch, declared):
    """An old immutable reservation cannot acquire a new research plan implicitly."""
    call_magnitude_registration(source, magnitude_training_start=None, final_read=final_holdout_declaration() if declared else None)
    monkeypatch.setattr(source.repository, "scan", support.forbid_access)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        call_magnitude_registration(source, admit=True)


def test_late_calendar_failure_never_publishes_or_retries(source, monkeypatch):
    """Real population admission followed by a failed calendar yields no binding."""
    from engine.v2.data.errors import fail
    original, calls = source.repository.scan, []
    def scan(query, *, table_name):
        calls.append(table_name)
        if table_name == "daily_market":
            raise fail("OBJECT_CORRUPT", "synthetic calendar failure")
        yield from original(query, table_name=table_name)
    monkeypatch.setattr(source.repository, "scan", scan)
    before, objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with support.refuses(DataError, "OBJECT_CORRUPT"):
        call_magnitude_registration(source)
    assert calls.count("daily_market") == 1 and "earnings_events" in calls
    assert catalog_rows(source.conn) == before and artifact_object_bytes(source.store) == objects


@pytest.mark.parametrize("index,code", [(2, "CONTRACT_MISMATCH"), (4, "HOLDOUT_ACCESS_DENIED"),
    (5, "HOLDOUT_ACCESS_DENIED"), (6, "HOLDOUT_ACCESS_DENIED")])
def test_protected_or_outside_explicit_population_refuses_before_calendar(source, monkeypatch, index, code):
    """Native admission remains upstream of any calendar or outcome payload read."""
    original = source.repository.scan
    def scan(query, *, table_name):
        assert table_name == "earnings_events"
        yield from original(query, table_name=table_name)
    monkeypatch.setattr(source.repository, "scan", scan)
    before, objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with support.refuses(DataError, code):
        call_magnitude_registration(source, event_ids=[source.events[index]["event_id"]])
    assert catalog_rows(source.conn) == before and artifact_object_bytes(source.store) == objects


def rewrite_magnitude_registration(source, registered, document):
    """Rebind synthetic catalog evidence so malformed content reaches the payload guard."""
    encoded = canonical_json(document).encode()
    ref = source.store.publish_bytes(encoded, schema_ref=document["schema_version"])
    variant = content_hash(document)
    source.conn.execute("UPDATE hypotheses SET input_hash=?, payload_hash=? WHERE run_id=?",
        (variant, content_hash({"spec": document["spec_hash"], "input": variant}), registered.run_id))
    source.conn.execute("UPDATE experiment_runs SET input_hash=?, evidence_json=? WHERE run_id=?",
        (variant, json.dumps({"native_registration": to_document(ref), "variant_id": variant,
                             "planned_variants": 1}), registered.run_id))
    return replace(registered, binding_json=encoded)


@pytest.mark.parametrize("kind", ["v1.0", "v1.1", "final", "unknown", "plan-none", "request-none",
    "plan-extra", "plan-missing", "request-extra", "request-start", "snapshot", "month", "version",
    "years", "source", "plan-hash", "plan-schema", "events-none", "event-id", "empty", "duplicate",
    "event-missing", "event-extra", "clock-none", "clock-extra", "clock-hash", "clock-foreign", "clock-bool",
    "folds-none", "folds-empty", "fold-extra", "fold-missing", "fold-year", "fold-foreign", "fold-overlap",
    "fold-wrong-year", "fold-duplicate", "fold-boundary-type", "horizons", "horizons-bool"])
def test_rehashed_malformed_or_cross_context_plans_refuse_without_reads(source, monkeypatch, kind):
    """Valid outer hashes cannot bless malformed nested metadata or a final-read hybrid."""
    registered = call_magnitude_registration(source)
    doc = registered.document
    plan = doc["magnitude_plan"]
    if kind.startswith("v1."):
        doc["schema_version"] = "native_experiment_registration." + kind
    elif kind in ("final", "unknown"):
        doc["final_read" if kind == "final" else "unknown"] = {}
    elif kind in ("plan-none", "request-none"):
        doc["magnitude_plan" if kind == "plan-none" else "magnitude_request"] = None
    elif kind == "plan-extra":
        plan["unknown"] = True
    elif kind == "plan-missing":
        plan.pop("fold_policy")
    elif kind.startswith("request-"):
        doc["magnitude_request"]["unknown" if kind == "request-extra" else "training_start"] = "changed"
    elif kind in ("snapshot", "month", "version", "years", "plan-schema"):
        field = {"snapshot": "snapshot_id", "month": "holdout_as_of_month", "version": "rolling_membership_version",
                 "years": "evaluation_years", "plan-schema": "schema_version"}[kind]
        plan[field] = ["2025"] if kind == "years" else "changed"
    elif kind == "source":
        doc["execution_plan"]["price_source"] = "option_chains"
    elif kind.startswith("horizons"):
        plan["horizons"] = [5, 3, True] if kind == "horizons-bool" else [1, 3, 5]
    elif kind == "event-id":
        plan["events"][0]["event_id"] = True
    elif kind == "event-missing":
        plan["events"][0].pop("ticker")
    elif kind == "event-extra":
        plan["events"][0]["outcome"] = 5.0
    elif kind.startswith("clock-"):
        clock = plan["events"][0]["forecasts"][0]
        field = {"clock-extra": "unknown", "clock-hash": "cutoff_id", "clock-foreign": "event_date",
                 "clock-bool": "horizon_sessions", "clock-none": "cutoff_id"}[kind]
        clock[field] = True if kind == "clock-bool" else "changed"
        if kind != "clock-hash":
            clock["cutoff_id"] = content_hash({k: v for k, v in clock.items() if k != "cutoff_id"})
        if kind == "clock-none":
            plan["events"][0]["forecasts"] = None
    elif kind.startswith("fold"):
        fold = plan["folds"][0]
        if kind in ("folds-none", "folds-empty"):
            plan["folds"] = None if kind == "folds-none" else []
        elif kind == "fold-missing":
            fold.pop("training_labels_before")
        elif kind == "fold-overlap":
            fold["training_candidate_event_ids"] = list(fold["evaluation_event_ids"])
        elif kind == "fold-wrong-year":
            fold["training_candidate_event_ids"], fold["evaluation_event_ids"] = (
                fold["evaluation_event_ids"], fold["training_candidate_event_ids"])
        elif kind == "fold-duplicate":
            fold["training_candidate_event_ids"] *= 2
        else:
            field = {"fold-extra": "unknown", "fold-year": "year", "fold-foreign": "training_candidate_event_ids",
                     "fold-boundary-type": "training_labels_before"}[kind]
            fold[field] = ["foreign"] if kind == "fold-foreign" else None if kind == "fold-boundary-type" else "changed"
    elif kind == "duplicate":
        plan["events"].append(plan["events"][0])
    elif kind in ("events-none", "empty"):
        plan["events"] = None if kind == "events-none" else []
    plan["plan_id"] = "changed" if kind == "plan-hash" else content_hash({k: v for k, v in plan.items() if k != "plan_id"})
    forged = rewrite_magnitude_registration(source, registered, doc)
    before, objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(source.repository, "scan", support.forbid_access)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        native.verify_native_registration(source.conn, source.store, forged)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        call_magnitude_registration(source, admit=True)
    assert catalog_rows(source.conn) == before and artifact_object_bytes(source.store) == objects


def test_stored_integrity_does_not_replace_fresh_semantic_admission(source):
    """A self-consistent stored digest still cannot override freshly planned fold bounds."""
    registered = call_magnitude_registration(source)
    doc = registered.document
    doc["magnitude_plan"]["folds"][0]["training_labels_before"] = "2024-01-01"
    plan = doc["magnitude_plan"]
    plan["plan_id"] = content_hash({k: v for k, v in plan.items() if k != "plan_id"})
    forged = rewrite_magnitude_registration(source, registered, doc)
    native.verify_native_registration(source.conn, source.store, forged)
    source.repository.scans.clear()
    before, objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        call_magnitude_registration(source, admit=True)
    assert {name for name, _ in source.repository.scans} == {"earnings_events", "daily_market"}
    assert catalog_rows(source.conn) == before and artifact_object_bytes(source.store) == objects


_DESCRIPTORS = ("target_recipe", "fold_policy", "population_input", "calendar_input", "availability")
_BAD_DATES = (None, True, "20231212", "2023-2-12", "2023-02-30", "2023-12-12T00:00:00", " 2023-12-12")
_NESTED_MALFORMED = [
    *((name, value) for name in _DESCRIPTORS for value in (None, [], {}, {"unknown": True})),
    *((name + ".unknown", True) for name in _DESCRIPTORS),
    ("target_recipe.secondary_diagnostics", None), ("target_recipe.estimand", "conditional_median"),
    ("fold_policy.inner_search", "adaptive"), ("availability.source_publication", "verified"),
    *((name + "." + key, value) for name in ("population_input", "calendar_input")
      for key, value in (("table", None), ("columns", None), ("columns", ["outcome"]),
                         ("dataset_version", None), ("dataset_version", "foreign"),
                         ("objects", None), ("objects", []), ("objects", [None]),
                         ("objects", "$duplicate"), ("objects", [{}]),
                         ("objects.0.unknown", True), ("objects.0.fragment_id", None),
                         ("objects.0.fragment_id", " padded "), ("objects.0.object_id", None),
                         ("objects.0.object_id", " padded "), ("objects.0.content_hash", "sha256:bad"))),
    ("calendar_input.snapshot_id", None), ("calendar_input.snapshot_id", "foreign"),
    ("calendar_input.calendar_version", None), ("calendar_input.calendar_version", "foreign"),
    *((name, value) for name in ("training_start", "events.0.event_date", "folds.0.training_labels_before",
                                "events.0.forecasts.0.pre_announcement_close",
                                "events.0.forecasts.0.source_session_ceiling") for value in _BAD_DATES),
    *(("evaluation_years", years) for years in (None, [], {}, [True], [2024], ["24"], ["２０２４"],
          ["0000"], ["0001"], ["9999"], ["2024", "2024"], ["2025", "2024"])),
    ("training_start", "2024-01-01"), ("holdout_as_of_month", None), ("holdout_as_of_month", "2026-13"),
    ("holdout_as_of_month", "202601"), ("random_membership_version", None), ("rolling_membership_version", []),
    ("random_membership_version", "unknown"), ("rolling_membership_version", "unknown"), ("snapshot_id", None),
    ("events.0.session", "UNKNOWN"), ("events.0.population_use", "training"), ("events.0.event_id", " padded "),
    *(("events.0.forecasts.0.calendar_id", value) for value in (None, True, "bad", content_hash("foreign"))),
    ("events.0.forecasts.0.pre_announcement_close", "2023-12-11"),
]


@pytest.mark.parametrize("path,value", _NESTED_MALFORMED)
def test_rehashed_nested_wire_schema_refuses_before_metadata_scan(source, monkeypatch, path, value):
    """Rehash and synchronize declarations so nested schema checks, not stale digests, refuse."""
    registered = call_magnitude_registration(source)
    doc = registered.document
    plan = doc["magnitude_plan"]
    target = plan
    keys = [int(key) if key.isdecimal() else key for key in path.split(".")]
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = target[keys[-1]] * 2 if value == "$duplicate" else value
    doc["magnitude_request"] = {key: plan[key] for key in ("training_start", "evaluation_years")}
    doc["execution_plan"]["folds"] = plan["evaluation_years"]
    if path == "evaluation_years" and value == []:
        plan["folds"] = []
    doc["holdouts"] = {key: plan[key] for key in doc["holdouts"]}
    if path == "snapshot_id":
        doc["snapshot"]["snapshot_id"] = plan["calendar_input"]["snapshot_id"] = value
    if path == "calendar_input.calendar_version" and value is None:
        doc["snapshot"]["calendar_version"] = None
    if path.endswith("dataset_version") and value is None:
        table = plan[keys[0]]["table"]
        doc["snapshot"]["table_versions"][table]["dataset_version_id"] = None
    doc["event_ids"] = sorted(event["event_id"] for event in plan["events"])
    for event in plan["events"]:
        for clock in event["forecasts"]:
            clock.update(event_date=event["event_date"], session=event["session"])
            clock["cutoff_id"] = content_hash({key: item for key, item in clock.items() if key != "cutoff_id"})
    plan["plan_id"] = content_hash({key: item for key, item in plan.items() if key != "plan_id"})
    forged = rewrite_magnitude_registration(source, registered, doc)
    before, objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(source.repository, "scan", support.forbid_access)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        native.verify_native_registration(source.conn, source.store, forged)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        call_magnitude_registration(source, admit=True)
    assert catalog_rows(source.conn) == before and artifact_object_bytes(source.store) == objects


def final_holdout_declaration():
    """Synthetic final-request metadata only; none of these referenced objects are read."""
    return FinalHoldoutDeclaration(
        evaluation_recipe_ref=artifact_reference(b"recipe", "final_holdout_recipe.v1.0"),
        completeness_bar_ref=artifact_reference(b"bar", "final_holdout_completeness.v1.0"),
        frozen_model_refs=(artifact_reference(b"model metadata", "synthetic_model_metadata.v1.0"),))


@pytest.mark.parametrize("declared", [False, True])
def test_older_registration_bytes_remain_exact(source, declared):
    """The optional extension adds no fields or byte changes to either old schema."""
    declaration = final_holdout_declaration()
    registered = call_magnitude_registration(source, magnitude_training_start=None, final_read=declaration if declared else None)
    context = {"holdout_as_of_month": "2026-01", "random_membership_version": "canonical-event-sha256.v1",
               "rolling_membership_version": "calendar-months.v1"}
    expected = {"schema_version": "native_experiment_registration.v1.0", "spec_hash": source.spec.spec_hash,
        "execution_plan": resolve_experiment_plan(source.spec).as_document(), "snapshot": to_document(source.snapshot),
        "scope": "shadow", "source_closure": native.source_closure(source.code_root, ["run.py"]),
        "environment": native.environment_identity(1), "holdouts": context,
        "event_ids": sorted(e["event_id"] for e in source.events[:4])}
    if declared:
        expected.update(schema_version="native_experiment_registration.v1.1", final_read=to_document(declaration))
    assert registered.binding_json == canonical_json(expected).encode()
    assert call_magnitude_registration(source, admit=True, magnitude_training_start=None,
                 final_read=declaration if declared else None) == registered


@pytest.mark.parametrize("invalid_years,invalid_start", [
    ([], "2023-12-10"), (None, "2023-12-10"), ([2024], "2023-12-10"),
    (["2024", "2024"], "2023-12-10"), (["2025", "2024"], "2023-12-10"),
    (["２０２４"], "2023-12-10"), (["0001"], "0001-01-01"), (["9999"], "2023-12-10"),
    (["2024"], None), (["2024"], "2023-02-30"), (["2024"], "2024-01-01"),
])
def test_public_window_is_detached_and_rejects_invalid_requests(invalid_years, invalid_start):
    """The registration declaration shares the planner's pure validation rules."""
    years = ["2024"]
    result = magnitude_window(years, "2023-12-10")
    years.append("2025")
    assert result == {"training_start": "2023-12-10", "evaluation_years": ["2024"]}
    with support.refuses(DataError, "CONTRACT_MISMATCH"):
        magnitude_window(invalid_years, invalid_start)


def test_multiple_spec_years_define_the_complete_registered_fold_list(source):
    """No second evaluation-year declaration can disagree with the execution plan."""
    spec = replace(source.spec, folds=("2024", "2025"))
    expected = plan_test_magnitude_folds(source.repository, source.snapshot, evaluation_years=["2024", "2025"])
    registered = call_magnitude_registration(source, spec=spec)
    assert registered.document["magnitude_plan"] == expected
    assert [fold["year"] for fold in expected["folds"]] == ["2024", "2025"]
    assert call_magnitude_registration(source, spec=spec, admit=True) == registered


@pytest.mark.parametrize("source", [[history_event(day, ticker="AAA") for day in
    ("2023-12-12", "2023-12-14", "2024-01-03", "2024-01-04")]], indirect=True)
@pytest.mark.parametrize("kind", ["events", "event-tie", "train", "train-tie", "test", "test-tie"])
def test_rehashed_membership_permutations_refuse_before_scans(source, monkeypatch, kind):
    """Preserve members and hashes, then reject noncanonical date or same-day ID order."""
    registered = call_magnitude_registration(source)
    native.verify_native_registration(source.conn, source.store, registered)
    assert call_magnitude_registration(source, admit=True) == registered
    doc = registered.document
    plan = doc["magnitude_plan"]
    events, fold = plan["events"], plan["folds"][0]
    ids = [event["event_id"] for event in events]
    assert ids == [event["event_id"] for event in sorted(events, key=lambda e: (e["event_date"], e["event_id"]))]
    train, test = fold["training_candidate_event_ids"], fold["evaluation_event_ids"]
    assert len(train) == len(test) == 3
    assert train == ids[:3] and test == ids[3:]
    if kind.startswith("event"):
        changed = events
    else:
        changed = train if kind.startswith("train") else test
    if kind.endswith("tie"):
        members = {event["event_id"]: event for event in events}
        first, second = changed[:2] if changed is events else [members[key] for key in changed[:2]]
        assert first["event_date"] == second["event_date"] and first["event_id"] < second["event_id"]
        changed[0], changed[1] = changed[1], changed[0]
    else:
        changed.reverse()
    plan["plan_id"] = content_hash({key: value for key, value in plan.items() if key != "plan_id"})
    forged = rewrite_magnitude_registration(source, registered, doc)
    before, objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(source.repository, "scan", support.forbid_access)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        native.verify_native_registration(source.conn, source.store, forged)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        call_magnitude_registration(source, admit=True)
    assert catalog_rows(source.conn) == before and artifact_object_bytes(source.store) == objects


def _rehash_clocks_and_plan(doc):
    """Repair only digests; intentional cross-link changes remain independently testable."""
    plan = doc["magnitude_plan"]
    for event in plan["events"]:
        for clock in event["forecasts"]:
            clock["cutoff_id"] = content_hash({key: value for key, value in clock.items() if key != "cutoff_id"})
    plan["plan_id"] = content_hash({key: value for key, value in plan.items() if key != "plan_id"})


def _assert_forged_refusal(source, registered, doc, monkeypatch, *, spec=None):
    """Exercise public stored verification first, before fresh admission can mask a guard."""
    _rehash_clocks_and_plan(doc)
    forged = rewrite_magnitude_registration(source, registered, doc)
    before, objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(source.repository, "scan", support.forbid_access)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        native.verify_native_registration(source.conn, source.store, forged)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        call_magnitude_registration(source, admit=True, spec=spec)
    assert catalog_rows(source.conn) == before and artifact_object_bytes(source.store) == objects


@pytest.mark.parametrize("link", ["events", "years", "snapshot", "holdout_as_of_month",
    "random_membership_version", "rolling_membership_version"])
def test_outer_links_refuse_with_an_otherwise_unchanged_valid_plan(source, monkeypatch, link):
    """Alter only the enclosing link so nested plan-shape checks cannot mask it."""
    registered = call_magnitude_registration(source)
    doc = registered.document
    if link == "events":
        doc["event_ids"] = doc["event_ids"][1:]
    elif link == "years":
        doc["execution_plan"]["folds"] = ["2025"]
    elif link == "snapshot":
        tables = ("earnings_events", "daily_market")
        other = commit_tables(source.conn, FakeClock(),
            {name: source.repository.fragment_records(source.snapshot, name) for name in tables},
            {name: source.repository.table_contract(source.snapshot, name) for name in tables},
            scope="other", receipt_id="other", attempt_id="other", store=source.store)
        doc["snapshot"] = to_document(other)
        assert other.snapshot_id != source.snapshot.snapshot_id
    else:
        doc["holdouts"][link] = "2026-02" if link == "holdout_as_of_month" else "different-membership.v1"
    assert doc["magnitude_plan"] == registered.document["magnitude_plan"]
    _assert_forged_refusal(source, registered, doc, monkeypatch)


@pytest.mark.parametrize("kind", ["reversed", "duplicate"])
def test_year_order_refuses_after_repairing_fold_and_outer_declarations(source, monkeypatch, kind):
    """Corresponding folds and all year declarations agree except for the ordering rule."""
    spec = replace(source.spec, folds=("2024", "2025"))
    registered = call_magnitude_registration(source, spec=spec)
    doc = registered.document
    plan = doc["magnitude_plan"]
    if kind == "reversed":
        plan["evaluation_years"].reverse()
        plan["folds"].reverse()
    else:
        plan["evaluation_years"].insert(0, plan["evaluation_years"][0])
        plan["folds"].insert(0, plan["folds"][0])
    doc["magnitude_request"]["evaluation_years"] = plan["evaluation_years"]
    doc["execution_plan"]["folds"] = plan["evaluation_years"]
    assert [fold["year"] for fold in plan["folds"]] == plan["evaluation_years"]
    _assert_forged_refusal(source, registered, doc, monkeypatch, spec=spec)


@pytest.mark.parametrize("kind", ["duplicate-event", "blank-event-id", "null-ticker", "blank-ticker",
    "clock-schema", "clock-session", "clock-event-date", "clock-short", "clock-long", "clock-bool",
    "clock-anchor-null", "clock-anchor-compact", "empty-train"])
def test_independent_event_clock_and_training_guards(source, monkeypatch, kind):
    """Repair unrelated identities/cardinalities instead of relying on secondary refusals."""
    registered = call_magnitude_registration(source)
    doc = registered.document
    plan = doc["magnitude_plan"]
    event = plan["events"][0]
    clocks = event["forecasts"]
    if kind == "duplicate-event":
        plan["events"].insert(0, event)
        doc["event_ids"] = sorted(item["event_id"] for item in plan["events"])
    elif kind == "blank-event-id":
        old, event["event_id"] = event["event_id"], ""
        doc["event_ids"] = sorted(item["event_id"] for item in plan["events"])
        plan["folds"][0]["training_candidate_event_ids"] = ["" if key == old else key
            for key in plan["folds"][0]["training_candidate_event_ids"]]
    elif kind.endswith("ticker"):
        event["ticker"] = None if kind == "null-ticker" else ""
    elif kind in ("clock-schema", "clock-session", "clock-event-date"):
        key, value = {"clock-schema": ("schema_version", "earnings_forecast_cutoff.v2"),
            "clock-session": ("session", "BMO"), "clock-event-date": ("event_date", "2023-12-11")}[kind]
        clocks[0][key] = value
    elif kind == "clock-short":
        clocks.pop()
    elif kind == "clock-long":
        clocks.append(dict(clocks[-1]))
    elif kind == "clock-bool":
        assert clocks[-1]["horizon_sessions"] == 1
        clocks[-1]["horizon_sessions"] = True
    elif kind.startswith("clock-anchor"):
        for clock in clocks:
            clock["pre_announcement_close"] = None if kind.endswith("null") else "20231212"
    else:
        plan["folds"][0]["training_candidate_event_ids"] = []
    _assert_forged_refusal(source, registered, doc, monkeypatch)


@pytest.mark.parametrize("admit", [False, True])
def test_no_caller_supplied_plan_parameter(source, monkeypatch, admit):
    """A valid prebuilt plan cannot bypass the public metadata reconstruction boundary."""
    registered = call_magnitude_registration(source)
    method = native.require_native_registration if admit else native.register_native
    parameters = inspect.signature(method).parameters
    assert "magnitude_plan" not in parameters
    assert all(parameter.kind != inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    before, objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(source.repository, "resolve_pinned", support.forbid_access)
    with pytest.raises(TypeError):
        call_magnitude_registration(source, admit=admit, magnitude_plan=registered.document["magnitude_plan"])
    assert catalog_rows(source.conn) == before and artifact_object_bytes(source.store) == objects


def test_prediction_cli_rejects_magnitude_clock_for_default_target(monkeypatch, capsys):
    """Magnitude clocks cannot silently change the default target before store access."""
    from experiments import v2_prediction as cli
    monkeypatch.setattr(cli.foundation, "ArtifactStore", support.forbid_access)
    with pytest.raises(SystemExit) as caught:
        cli.main(["register", "--spec", "unused", "--catalog", "unused", "--store-root", "unused",
                  "--as-of-month", "2026-01", "--magnitude-training-start", "2023-12-10"])
    assert caught.value.code == 2
    assert "positive_move accepts neither" in capsys.readouterr().err


def test_registration_effects_are_only_its_binding_and_reservation(source, monkeypatch):
    """Poison shared fit/history execution and compare complete pre-registration state."""
    from engine.v2.features import panel_math
    from engine.v2.ops import experiment_folds
    from engine.v2.research import event_history, magnitude_features
    from experiments import prediction_report
    for module, name in ((experiment_folds, "fit_walk_forward_fold"), (prediction_report, "fit_walk_forward_fold"),
            (panel_math, "history_features"), (event_history, "history_features"),
            (event_history, "load_event_history"), (magnitude_features, "load_event_history")):
        monkeypatch.setattr(module, name, support.forbid_access)
    before, objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
    registered = call_magnitude_registration(source)
    after, published = catalog_rows(source.conn), artifact_object_bytes(source.store)
    changed = {table for table in before.keys() | after.keys() if before.get(table) != after.get(table)}
    assert changed == {"hypotheses", "experiment_runs"}
    assert all(len(after[table]) == len(before[table]) + 1 for table in changed)
    assert all(published[path] == data for path, data in objects.items())
    assert [published[path] for path in published.keys() - objects.keys()] == [registered.binding_json]
    assert call_magnitude_registration(source, admit=True) == registered
    assert catalog_rows(source.conn) == after and artifact_object_bytes(source.store) == published


def test_cross_year_event_permutation_is_not_masked_by_fold_membership(source, monkeypatch):
    """Swap evaluation years while every fold's internal member order remains intact."""
    spec = replace(source.spec, folds=("2024", "2025"))
    registered = call_magnitude_registration(source, spec=spec)
    doc = registered.document
    plan = doc["magnitude_plan"]
    assert [event["event_date"][:4] for event in plan["events"]] == ["2023", "2024", "2025"]
    plan["events"][1], plan["events"][2] = plan["events"][2], plan["events"][1]
    for fold in plan["folds"]:
        for name in ("training_candidate_event_ids", "evaluation_event_ids"):
            assert fold[name] == [event["event_id"] for event in plan["events"] if event["event_id"] in fold[name]]
    _assert_forged_refusal(source, registered, doc, monkeypatch, spec=spec)


@pytest.mark.parametrize("source", [[history_event("2023-12-14")]], indirect=True)
@pytest.mark.parametrize("change", ["start", "years", "month", "events"])
def test_changed_register_request_cannot_overwrite_its_reservation(source, monkeypatch, change):
    """Otherwise-valid altered requests refuse before publication on the register path."""
    call_magnitude_registration(source)
    source.repository.scans.clear()
    changes = {"start": {"magnitude_training_start": "2023-12-11"},
        "years": {"spec": replace(source.spec, folds=("2024", "2025"))},
        "month": {"as_of_month": "2026-02"},
        "events": {"event_ids": [event["event_id"] for event in source.events[:2]]}}[change]
    before, objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(source.store, "publish_bytes", support.forbid_access)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        call_magnitude_registration(source, **changes)
    assert {name for name, _ in source.repository.scans} == {"earnings_events", "daily_market"}
    assert catalog_rows(source.conn) == before and artifact_object_bytes(source.store) == objects


@pytest.mark.parametrize("years,start", [(("2024", "2026"), "2023-12-10"), (("2024",), "2023-12-30")])
def test_empty_evaluation_or_training_fold_never_publishes(source, monkeypatch, years, start):
    """Propagate the planner's exact nonretryable refusal without reserving a partial run."""
    before, objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(source.store, "publish_bytes", support.forbid_access)
    with support.refuses(DataError, "POPULATION_COLLAPSED"):
        call_magnitude_registration(source, spec=replace(source.spec, folds=years), magnitude_training_start=start)
    assert {name for name, _ in source.repository.scans} == {"earnings_events", "daily_market"}
    assert catalog_rows(source.conn) == before and artifact_object_bytes(source.store) == objects


@pytest.mark.parametrize("kind", ["horizon-value", "calendar-digest", "schedule-availability"])
def test_clock_values_and_schedule_status_have_independent_refusal_oracles(source, monkeypatch, kind):
    """Keep calendar agreement, wire types and other descriptors valid for each guard."""
    registered = call_magnitude_registration(source)
    doc = registered.document
    plan = doc["magnitude_plan"]
    if kind == "horizon-value":
        clock = plan["events"][0]["forecasts"][0]
        assert clock["horizon_sessions"] == 5
        clock["horizon_sessions"] = 4
        assert type(clock["horizon_sessions"]) is int
    elif kind == "calendar-digest":
        for event in plan["events"]:
            for clock in event["forecasts"]:
                clock["calendar_id"] = "not-a-digest"
        assert {clock["calendar_id"] for event in plan["events"]
                for clock in event["forecasts"]} == {"not-a-digest"}
    else:
        assert plan["availability"]["event_schedule_as_known"] == "unverified"
        plan["availability"]["event_schedule_as_known"] = "observed"
    _assert_forged_refusal(source, registered, doc, monkeypatch)


@pytest.mark.parametrize("operation", ["publish", "refusal", "replay", "export"])
@pytest.mark.parametrize("no_ledger", [False, True])
def test_magnitude_outcome_paths_refuse_before_result_access(source, monkeypatch, tmp_path, operation, no_ledger):
    """A valid preregistration permits read/admission, but only private conflict evidence."""
    registered = call_magnitude_registration(source)
    assert native.read_native_registration(source.conn, source.store, source.spec) == registered
    assert call_magnitude_registration(source, admit=True) == registered
    before, objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
    ledger = tmp_path / "unused" / "ledger.csv"
    refusal = canonical_json({"schema_version": "native_experiment_refusal.v1.0",
        "failure_code": "EXPERIMENT_IDENTITY_CONFLICT", "request": {"run_id": registered.run_id},
        "variant_id": None}).encode()
    ref = artifact_reference(refusal, "native_experiment_refusal.v1.0")
    publish = source.store.publish_bytes
    def private_only(payload, *, schema_ref):
        assert payload == refusal and schema_ref == ref.schema_ref
        return publish(payload, schema_ref=schema_ref)
    monkeypatch.setattr(source.store, "publish_bytes", private_only)
    monkeypatch.setattr(source.repository, "scan", support.forbid_access)
    for name in ("_evidence", "_reconcile", "_report_path", "ledger_append"):
        monkeypatch.setattr(outcomes, name, support.forbid_access)
    method = {"publish": outcomes.publish_native_outcome, "refusal": outcomes.publish_native_outcome,
              "replay": outcomes.replay_native_outcome, "export": outcomes.export_native_report}[operation]
    options = ({"report": "Synthetic result must not publish", "attempted": True} if operation == "publish"
               else {"problem": ops_fail("FEATURE_LOOKAHEAD", "synthetic").problem} if operation == "refusal" else {})
    with pytest.raises(OpsError) as caught:
        method(source.conn, source.store, registered, no_ledger=no_ledger, ledger_path=ledger, **options)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.retryable is False
    assert caught.value.problem.details == {"refusal_ref": to_document(ref)}
    assert source.store.read_verified(ref) == refusal
    assert catalog_rows(source.conn) == before
    assert artifact_object_bytes(source.store) == {**objects, ref.storage_key: refusal}
    assert not ledger.parent.exists() and not (source.store.root / "native_reports").exists()


@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize("no_ledger", [False, True])
def test_older_registration_schemas_keep_complete_outcome_cycle(source, tmp_path, declared, no_ledger):
    """The outcome schema fence preserves both existing schemas and recording modes."""
    registered = call_magnitude_registration(source, magnitude_training_start=None, final_read=final_holdout_declaration() if declared else None)
    assert registered.document["schema_version"] == "native_experiment_registration.v1." + str(int(declared))
    options = {"no_ledger": no_ledger, "ledger_path": tmp_path / "ledger.csv"}
    result = outcomes.publish_native_outcome(source.conn, source.store, registered,
        report="Synthetic compatibility report", attempted=True, **options)
    assert outcomes.replay_native_outcome(source.conn, source.store, registered, **options) == result
    path = outcomes.export_native_report(source.conn, source.store, registered, **options)
    assert path.read_text() == "Synthetic compatibility report"
    assert options["ledger_path"].exists() is not no_ledger


@pytest.mark.parametrize("source", [[history_event("2024-01-04")]], indirect=True)
@pytest.mark.parametrize("kind", ["one-anchor", "one-calendar", "missing-evaluation"])
def test_cross_clock_and_fold_consistency_guards_are_independent(source, monkeypatch, kind):
    """Valid dates/digests and nonempty subsequences cannot mask agreement or coverage."""
    registered = call_magnitude_registration(source)
    doc = registered.document
    plan = doc["magnitude_plan"]
    if kind == "one-anchor":
        plan["events"][0]["forecasts"][0]["pre_announcement_close"] = "2023-12-11"
    elif kind == "one-calendar":
        for clock in plan["events"][0]["forecasts"]:
            clock["calendar_id"] = content_hash("different valid calendar identity")
    else:
        assert len(plan["folds"][0]["evaluation_event_ids"]) == 2
        plan["folds"][0]["evaluation_event_ids"].pop()
    _assert_forged_refusal(source, registered, doc, monkeypatch)


def test_v12_reference_is_invalid_final_request_metadata_before_catalog_access(source, monkeypatch):
    """Only pure request decoding: no authority call, terminal prompt or authorization."""
    registered = call_magnitude_registration(source, magnitude_training_start=None, final_read=final_holdout_declaration())
    request = native.prepare_final_holdout_request(source.conn, source.store, registered)
    ref = artifact_reference(registered.binding_json, "native_experiment_registration.v1.2")
    forged = replace(request, registration_ref=ref, variant_id=ref.content_hash)
    monkeypatch.setattr(source.store, "read_verified", support.forbid_access)
    with pytest.raises(ValueError, match="^invalid final-read artifact reference$"):
        request_from_document(to_document(forged))
    assert source.conn.execute("SELECT COUNT(*) FROM final_holdout_authorizations").fetchone()[0] == 0
