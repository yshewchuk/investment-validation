"""Native magnitude preregistration over real synthetic metadata, never outcomes."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.contracts import FinalHoldoutDeclaration
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import artifact_reference, canonical_json, content_hash, to_document
from engine.v2.ops.errors import OpsError
from engine.v2.ops.experiments import resolve_experiment_plan
from engine.v2.research.magnitude_folds import magnitude_window
from experiments import native_registration as native
from tests.data_scan_support import commit_tables
from tests.ops_support import FakeClock
from tests.v2.research.test_event_history import _event
from tests.v2.research.test_magnitude_folds import _fixture, _metadata_only, _plan
from tests.v2.research.test_native_registration import _catalog, _objects, _spec
from tests.v2.research.test_prediction_inputs import TrackingRepository


class PlanRepository(TrackingRepository):
    """Allow the registrar's one head pin while retaining real tracked scans."""
    def __init__(self, conn, store):
        super().__init__(conn, store)
        self.pins = []

    def resolve_pinned(self, scope):
        self.pins.append(scope)
        return Repository.resolve_pinned(self, scope)


@pytest.fixture
def source(tmp_path, monkeypatch):
    """Commit protected, ambiguous and out-of-window siblings with poisoned labels."""
    train, test = _event("2023-12-12"), {**_event("2024-01-03"), "session": "BMO"}
    before, future = _event("2023-11-01"), _event("2025-01-02")
    random, rolling = _event("2023-12-13", protected=True), _event("2025-12-03")
    alias = _event("2024-01-05")
    events = [train, test, before, future, random, rolling, alias,
              {**alias, "event_id": "ambiguous-sibling", "session": "BMO"}]
    conn, original, snapshot = _fixture(tmp_path, events)
    repository = PlanRepository(conn, original._store)
    _metadata_only(monkeypatch, repository, snapshot)
    root = tmp_path / "code"
    root.mkdir()
    (root / "run.py").write_text("raise AssertionError('registration must not execute a runner')\n")
    value = SimpleNamespace(conn=conn, store=original._store, repository=repository,
        snapshot=snapshot, code_root=root, events=events,
        spec=_spec(price_source="computed_moves", economic_params={}, seed=0, runner="run.py"))
    yield value
    conn.close()


def _call(source, *, admit=False, spec=None, **changes):
    """Use the actual native registration and read-only admission entrypoints."""
    method = native.require_native_registration if admit else native.register_native
    return method(source.conn, source.store, source.repository, spec or source.spec,
        **{"code_root": source.code_root, "as_of_month": "2026-01",
           "magnitude_training_start": "2023-12-10", **changes})


def _forbid(*args, **kwargs):
    """Fail if a pre-admission refusal touches metadata or an effect boundary."""
    raise AssertionError("forbidden access")


def test_full_plan_is_sealed_and_idempotently_admitted(source, monkeypatch):
    """Bind the independently planned metadata and only its selected population."""
    expected_plan = _plan(source.repository, source.snapshot)
    source.repository.scans.clear()
    registered = _call(source)
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
    assert source.repository.pins == ["shadow"]
    assert {name for name, _ in source.repository.scans} == {"earnings_events", "daily_market"}
    before, objects = _catalog(source.conn), _objects(source.store)
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    source.conn.execute("PRAGMA query_only=ON")
    assert _call(source, event_ids=list(reversed(document["event_ids"]))) == registered
    assert _call(source, admit=True) == registered
    assert _call(source, admit=True, expected=registered) == registered
    assert native.read_native_registration(source.conn, source.store, source.spec) == registered
    assert _catalog(source.conn) == before and _objects(source.store) == objects
    document["magnitude_request"]["evaluation_years"].clear()
    assert registered.document["magnitude_request"]["evaluation_years"] == ["2024"]
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        native.prepare_final_holdout_request(source.conn, source.store, registered)


def test_absent_outcomes_table_is_not_a_registration_dependency(source):
    """Use another real committed snapshot that carries only the metadata tables."""
    tables = ("earnings_events", "daily_market")
    snapshot = commit_tables(source.conn, FakeClock(),
        {name: source.repository.fragment_records(source.snapshot, name) for name in tables},
        {name: source.repository.table_contract(source.snapshot, name) for name in tables},
        scope="metadata-only", receipt_id="metadata-only", attempt_id="metadata-only", store=source.store)
    registered = _call(source, scope="metadata-only")
    assert registered.document["snapshot"] == to_document(snapshot)
    assert "computed_moves" not in registered.document["snapshot"]["table_versions"]
    assert _call(source, admit=True, scope="metadata-only") == registered


@pytest.mark.parametrize("changes", [
    {"magnitude_training_start": "2024-01-01"}, {"magnitude_training_start": "2023-02-30"},
    {"magnitude_training_start": True}, {"magnitude_training_start": []},
    {"final_read": {}}, {"spec_changes": {"price_source": "option_chains"}},
    {"spec_changes": {"folds": ("fold",)}}, {"spec_changes": {"folds": ("2025", "2024")}},
])
@pytest.mark.parametrize("admit", [False, True])
def test_bad_declarations_refuse_before_any_repository_access(source, monkeypatch, changes, admit):
    """Window and source validation precede pins, scans and registration effects."""
    changes = dict(changes)
    spec = replace(source.spec, **changes.pop("spec_changes", {}))
    before, objects = _catalog(source.conn), _objects(source.store)
    monkeypatch.setattr(source.repository, "resolve_pinned", _forbid)
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        _call(source, spec=spec, admit=admit, **changes)
    assert _catalog(source.conn) == before and _objects(source.store) == objects


@pytest.mark.parametrize("change", ["start", "years", "month", "scope", "snapshot", "source", "environment", "omitted", "expected"])
def test_identity_drift_refuses_before_metadata_scan(source, monkeypatch, change):
    """Stored identity is automatically checked even when expected is omitted."""
    registered = _call(source)
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
    before, objects = _catalog(source.conn), _objects(source.store)
    monkeypatch.setattr(source.repository, "scan", _forbid)
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _call(source, admit=True, **kwargs)
    assert _catalog(source.conn) == before and _objects(source.store) == objects


def test_missing_reservation_refuses_before_metadata_scan(source, monkeypatch):
    """Admission never substitutes fresh registration for a missing reservation."""
    monkeypatch.setattr(source.repository, "scan", _forbid)
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        _call(source, admit=True)


@pytest.mark.parametrize("declared", [False, True])
def test_adding_magnitude_mode_to_an_old_registration_refuses_before_scan(source, monkeypatch, declared):
    """An old immutable reservation cannot acquire a new research plan implicitly."""
    _call(source, magnitude_training_start=None, final_read=_declaration() if declared else None)
    monkeypatch.setattr(source.repository, "scan", _forbid)
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _call(source, admit=True)


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
    before, objects = _catalog(source.conn), _objects(source.store)
    with pytest.raises(DataError, match="OBJECT_CORRUPT"):
        _call(source)
    assert calls.count("daily_market") == 1 and "earnings_events" in calls
    assert _catalog(source.conn) == before and _objects(source.store) == objects


@pytest.mark.parametrize("index,code", [(2, "CONTRACT_MISMATCH"), (4, "HOLDOUT_ACCESS_DENIED"),
    (5, "HOLDOUT_ACCESS_DENIED"), (6, "HOLDOUT_ACCESS_DENIED")])
def test_protected_or_outside_explicit_population_refuses_before_calendar(source, monkeypatch, index, code):
    """Native admission remains upstream of any calendar or outcome payload read."""
    original = source.repository.scan
    def scan(query, *, table_name):
        assert table_name == "earnings_events"
        yield from original(query, table_name=table_name)
    monkeypatch.setattr(source.repository, "scan", scan)
    before, objects = _catalog(source.conn), _objects(source.store)
    with pytest.raises(DataError, match=code):
        _call(source, event_ids=[source.events[index]["event_id"]])
    assert _catalog(source.conn) == before and _objects(source.store) == objects


def _rewrite(source, registered, document):
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
    registered = _call(source)
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
    forged = _rewrite(source, registered, doc)
    before, objects = _catalog(source.conn), _objects(source.store)
    monkeypatch.setattr(source.repository, "scan", _forbid)
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        native.verify_native_registration(source.conn, source.store, forged)
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _call(source, admit=True)
    assert _catalog(source.conn) == before and _objects(source.store) == objects


def test_stored_integrity_does_not_replace_fresh_semantic_admission(source):
    """A self-consistent stored digest still cannot override freshly planned fold bounds."""
    registered = _call(source)
    doc = registered.document
    doc["magnitude_plan"]["folds"][0]["training_labels_before"] = "2024-01-01"
    plan = doc["magnitude_plan"]
    plan["plan_id"] = content_hash({k: v for k, v in plan.items() if k != "plan_id"})
    forged = _rewrite(source, registered, doc)
    native.verify_native_registration(source.conn, source.store, forged)
    source.repository.scans.clear()
    before, objects = _catalog(source.conn), _objects(source.store)
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _call(source, admit=True)
    assert {name for name, _ in source.repository.scans} == {"earnings_events", "daily_market"}
    assert _catalog(source.conn) == before and _objects(source.store) == objects


def _declaration():
    """Synthetic final-request metadata only; none of these referenced objects are read."""
    return FinalHoldoutDeclaration(
        evaluation_recipe_ref=artifact_reference(b"recipe", "final_holdout_recipe.v1.0"),
        completeness_bar_ref=artifact_reference(b"bar", "final_holdout_completeness.v1.0"),
        frozen_model_refs=(artifact_reference(b"model metadata", "synthetic_model_metadata.v1.0"),))


@pytest.mark.parametrize("declared", [False, True])
def test_older_registration_bytes_remain_exact(source, declared):
    """The optional extension adds no fields or byte changes to either old schema."""
    declaration = _declaration()
    registered = _call(source, magnitude_training_start=None, final_read=declaration if declared else None)
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
    assert _call(source, admit=True, magnitude_training_start=None,
                 final_read=declaration if declared else None) == registered


def test_public_window_is_detached_and_rejects_invalid_requests():
    """The registration declaration shares the planner's pure validation rules."""
    years = ["2024"]
    result = magnitude_window(years, "2023-12-10")
    years.append("2025")
    assert result == {"training_start": "2023-12-10", "evaluation_years": ["2024"]}
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        magnitude_window(["2024", "2024"], "2023-12-10")


def test_multiple_spec_years_define_the_complete_registered_fold_list(source):
    """No second evaluation-year declaration can disagree with the execution plan."""
    spec = replace(source.spec, folds=("2024", "2025"))
    expected = _plan(source.repository, source.snapshot, evaluation_years=["2024", "2025"])
    registered = _call(source, spec=spec)
    assert registered.document["magnitude_plan"] == expected
    assert [fold["year"] for fold in expected["folds"]] == ["2024", "2025"]
    assert _call(source, spec=spec, admit=True) == registered
