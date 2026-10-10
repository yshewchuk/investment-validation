"""Real synthetic native outcomes, immutable conflicts, and interrupted replay."""
import builtins
import csv
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.data.errors import make_problem as make_data_problem
from engine.v2.foundation import (
    ArtifactStore,
    canonical_json,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, make_problem
from experiments import native_outcomes as outcomes
from tests.v2.research.test_native_registration import (
    _call,
    _catalog,
    _evidence,
    _forbid,
    _objects,
    _ref,
    _spec,
)
from tests.v2.research.test_native_registration import (
    source as source,
)

REPORT = "# Synthetic native result\nFresh fitted result: 0.25.\n"


def _publish(source, registration, ledger, **kwargs):
    """Publish a genuine synthetic successful result unless explicitly overridden."""
    options = {"report": REPORT, "attempted": True, "ledger_path": ledger, **kwargs}
    return outcomes.publish_native_outcome(source.conn, source.store, registration, **options)


def _replay(source, registration, ledger, **kwargs):
    """Resume through the public replay entrypoint using the saved catalog intent."""
    return outcomes.replay_native_outcome(source.conn, source.store, registration,
                                          ledger_path=ledger, **kwargs)


def _document(source, document):
    """Read and verify one persisted typed reference."""
    return json.loads(source.store.read_verified(from_document(ArtifactRef, document)))


def _rows(path):
    """Inspect CSV using the same string representation frozen in an outcome."""
    return list(csv.DictReader(io.StringIO(path.read_text(), newline="")))


def _clock(monkeypatch, day):
    """Choose the publication date without changing filesystem or catalog clocks."""
    class Clock:
        @staticmethod
        def now(zone):
            assert zone is timezone.utc
            return datetime.fromisoformat(day + "T12:30:00+00:00")
    monkeypatch.setattr(outcomes, "datetime", Clock)


def _assert_conflict(source, captured, registration):
    """Every identity conflict is nonretryable and carries readable private evidence."""
    error = captured.value
    assert error.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert error.problem.retryable is False
    receipt = _document(source, error.problem.details["refusal_ref"])
    assert receipt == {
        "schema_version": "native_experiment_refusal.v1.0",
        "failure_code": "EXPERIMENT_IDENTITY_CONFLICT",
        "request": {"run_id": registration.run_id}, "variant_id": None,
    }


def _assert_objects_preserved(source, before):
    """Conflicts may publish private evidence but never rewrite earlier objects."""
    after = _objects(source.store)
    assert {key: after[key] for key in before} == before


def test_registration_is_planned_until_an_actual_outcome(source, tmp_path):
    """Preregistration and an empty replay cannot create an attempted-variant claim."""
    registration = _call(source)
    before, objects = _catalog(source.conn), _objects(source.store)
    evidence = _evidence(source, registration.run_id)
    assert evidence["planned_variants"] == 1
    assert not {"attempted_variants", "variants_tried", "native_outcome", "native_smoke"} & evidence.keys()
    ledger = tmp_path / "never-created" / "ledger.csv"
    assert _replay(source, registration, ledger) is None
    assert _replay(source, registration, ledger, no_ledger=True) is None
    assert not ledger.parent.exists()
    assert _catalog(source.conn) == before
    assert _objects(source.store) == objects


def test_success_binds_catalog_intent_exact_ledger_and_verified_receipt(source, tmp_path, monkeypatch):
    """Use the existing global run slot and keep all durable file effects outside SQL."""
    registration = _call(source)
    ledger = tmp_path / "recorded" / "ledger.csv"
    before = _catalog(source.conn)
    real_publish, real_append = source.store.publish_bytes, outcomes.ledger_append
    calls = []

    def publish(data, *, schema_ref):
        assert not source.conn.in_transaction
        calls.append(schema_ref)
        return real_publish(data, schema_ref=schema_ref)

    def append(rows, **kwargs):
        assert not source.conn.in_transaction
        calls.append("durable-ledger")
        return real_append(rows, **kwargs)

    monkeypatch.setattr(source.store, "publish_bytes", publish)
    monkeypatch.setattr(outcomes, "ledger_append", append)
    _clock(monkeypatch, "2026-01-02")
    result = _publish(source, registration, ledger)
    outcome, receipt = result["outcome"], result["receipt"]
    evidence = _evidence(source, registration.run_id)
    assert calls == ["native_experiment_report.v1.0", outcomes.SCHEMA, "durable-ledger",
                     "native_experiment_completion.v1.0"]
    assert _document(source, evidence["native_outcome"]) == outcome
    assert evidence["native_outcome_receipt"] == result["receipt_ref"]
    assert _document(source, result["receipt_ref"]) == receipt
    assert _rows(ledger) == [outcome["ledger_row"]]
    assert outcome["ledger_row"] == {
        "id": "synthetic-native", "spec_hash": registration.variant_id,
        "date": "2026-01-02", "stage": "ran", "oos_mean_mid": "",
        "sharpe_trade": "", "promoted": "False",
    }
    assert outcome["attempted_variants"] == 1
    assert outcome["failure_code"] is None
    assert not {"status", "holdouts", "snapshot_id", "no_ledger"} & outcome.keys()
    assert outcome["variant_id"] == registration.variant_id
    assert outcome["run_id"] == registration.run_id
    assert outcome["ledger_destination"] == content_hash(str(ledger.resolve()))
    assert source.store.read_verified(from_document(ArtifactRef, outcome["report_ref"])) == REPORT.encode()
    assert receipt == {
        "schema_version": "native_experiment_completion.v1.0",
        "outcome_ref": evidence["native_outcome"], "recording_mode": "recorded",
        "report_ref": outcome["report_ref"], "ledger_destination": outcome["ledger_destination"],
        "ledger_row_hash": content_hash(outcome["ledger_row"]),
    }
    after = _catalog(source.conn)
    assert {key: value for key, value in after.items() if key != "experiment_runs"} == {
        key: value for key, value in before.items() if key != "experiment_runs"}
    assert not list(tmp_path.rglob("REPORT.md"))
    assert not list(tmp_path.rglob("recording_receipt.json"))


def test_replay_and_repeat_publish_reuse_first_date_without_refitting(source, tmp_path, monkeypatch):
    """Replaying a frozen intent never fits, scans inputs, or replaces the first row date."""
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    _clock(monkeypatch, "2026-01-02")
    first = _publish(source, registration, ledger)
    before, objects, ledger_bytes = _catalog(source.conn), _objects(source.store), ledger.read_bytes()
    _clock(monkeypatch, "2026-02-10")
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression, "fit", _forbid)
    monkeypatch.setattr(source.repository, "scan", _forbid)
    monkeypatch.setattr(source.repository, "resolve_pinned", _forbid)
    real_publish = source.store.publish_bytes

    def only_completion(data, *, schema_ref):
        assert schema_ref == "native_experiment_completion.v1.0"
        return real_publish(data, schema_ref=schema_ref)

    with monkeypatch.context() as patch:
        patch.setattr(source.store, "publish_bytes", only_completion)
        assert _replay(source, registration, ledger) == first
    assert _publish(source, registration, ledger) == first
    assert _catalog(source.conn) == before
    assert ledger.read_bytes() == ledger_bytes
    _assert_objects_preserved(source, objects)
    assert _rows(ledger)[0]["date"] == "2026-01-02"


def test_concurrent_connections_share_one_outcome_and_exact_row(source, tmp_path):
    """Independent SQLite handles and artifact stores cannot reserve duplicate outcomes."""
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    barrier = Barrier(4)

    def worker(_):
        conn = connect(source.catalog_path, must_exist=True)
        try:
            store = ArtifactStore(source.store.root)
            barrier.wait(timeout=15)
            return outcomes.publish_native_outcome(conn, store, registration, report=REPORT,
                                                    attempted=True, ledger_path=ledger)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(4)))
    assert all(result == results[0] for result in results)
    assert _rows(ledger) == [results[0]["outcome"]["ledger_row"]]
    evidence = _evidence(source, registration.run_id)
    assert evidence["native_outcome_receipt"] == results[0]["receipt_ref"]
    assert _document(source, evidence["native_outcome"]) == results[0]["outcome"]


@pytest.mark.parametrize("change", ["report", "status", "destination", "variant"])
def test_changed_outcome_preserves_original_and_publishes_conflict(source, tmp_path, change):
    """Global identity conflicts cannot change the prior report, intent, CSV, or receipt."""
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    first = _publish(source, registration, ledger)
    before, objects, ledger_bytes = _catalog(source.conn), _objects(source.store), ledger.read_bytes()
    options, current, target = {}, registration, ledger
    if change == "report":
        options["report"] = REPORT + "Changed scientific result.\n"
    elif change == "status":
        options = {"report": None, "problem": make_problem("EXPERIMENT_VARIANT_FAILED", "failed")}
    elif change == "destination":
        target = tmp_path / "elsewhere" / "ledger.csv"
    else:
        document = registration.document
        document["environment"]["synthetic_change"] = True
        current = replace(registration, binding_json=canonical_json(document).encode())
    with pytest.raises(OpsError) as captured:
        _publish(source, current, target, **options)
    _assert_conflict(source, captured, current)
    assert _catalog(source.conn) == before
    assert ledger.read_bytes() == ledger_bytes
    assert not (tmp_path / "elsewhere").exists()
    _assert_objects_preserved(source, objects)
    assert _replay(source, registration, ledger) == first


def test_changing_attempted_count_is_an_identity_conflict(source, tmp_path):
    """A nonattempted refusal cannot be reclassified as an attempted variant."""
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    kwargs = {"report": None, "problem": make_problem("FEATURE_LOOKAHEAD", "synthetic")}
    _publish(source, registration, ledger, attempted=False, **kwargs)
    before, ledger_bytes, objects = _catalog(source.conn), ledger.read_bytes(), _objects(source.store)
    with pytest.raises(OpsError) as captured:
        _publish(source, registration, ledger, attempted=True, **kwargs)
    _assert_conflict(source, captured, registration)
    assert _catalog(source.conn) == before
    assert ledger.read_bytes() == ledger_bytes
    _assert_objects_preserved(source, objects)


@pytest.mark.parametrize("code,stage", [("FEATURE_LOOKAHEAD", "refused"),
    ("EXPERIMENT_VARIANT_FAILED", "failed"), ("HOLDOUT_ACCESS_DENIED", "refused")])
@pytest.mark.parametrize("attempted", [False, True])
@pytest.mark.parametrize("no_ledger", [False, True])
def test_admitted_refusal_has_no_report_and_correct_attempt_accounting(
        source, tmp_path, code, stage, attempted, no_ledger):
    """R3/R4/R5 record only typed refused/failed evidence, including before-fit failures."""
    registration = _call(source)
    ledger = tmp_path / "refusal" / "ledger.csv"
    before = _objects(source.store)
    result = _publish(source, registration, ledger, report=None,
                      problem=(make_data_problem if code == "HOLDOUT_ACCESS_DENIED" else make_problem)(
                          code, "secret exception text must not be persisted"),
                      attempted=attempted, no_ledger=no_ledger)
    outcome = result["outcome"]
    assert outcome["failure_code"] == code
    assert outcome["ledger_row"]["stage"] == stage
    assert outcome["attempted_variants"] == int(attempted)
    assert outcome["report_ref"] is result["receipt"]["report_ref"] is None
    added = set(_objects(source.store)) - before.keys()
    assert len(added) == 2
    assert all(b"secret exception" not in _objects(source.store)[key] for key in added)
    if no_ledger:
        assert not ledger.parent.exists()
    else:
        assert _rows(ledger) == [outcome["ledger_row"]]
    assert _replay(source, registration, ledger, no_ledger=no_ledger) == result


@pytest.mark.parametrize("code", ["INVALID_EXPERIMENT_SPEC", "SNAPSHOT_UNRESOLVED",
                                  "EXPERIMENT_IDENTITY_CONFLICT"])
@pytest.mark.parametrize("admitted", [False, True])
def test_pre_admission_refusals_are_private_receipt_only(source, tmp_path, code, admitted):
    """R1/R2/R6 receipts never invent registration, reports, or ledger identities."""
    registration = _call(source) if admitted else None
    before, objects = _catalog(source.conn), _objects(source.store)
    request = {"experiment_id": "synthetic-rejected", "scope": "shadow"}
    ref = outcomes.publish_native_refusal(source.store, problem=make_problem(code, "hidden raw error"),
                                          request=request, registration=registration)
    assert _document(source, to_document(ref)) == {
        "schema_version": "native_experiment_refusal.v1.0", "failure_code": code,
        "request": request, "variant_id": registration.variant_id if registration else None,
    }
    assert _catalog(source.conn) == before
    assert len(set(_objects(source.store)) - objects.keys()) == 1
    assert not list(tmp_path.rglob("*.csv"))
    assert not list(tmp_path.rglob("REPORT.md"))


class PoisonLedger:
    """Smoke mode must not even coerce, inspect, or truth-test its ledger argument."""
    def __fspath__(self):
        raise AssertionError("smoke resolved ledger path")

    def __str__(self):
        raise AssertionError("smoke stringified ledger path")

    def __bool__(self):
        raise AssertionError("smoke truth-tested ledger path")


def test_smoke_ignores_pathlike_sentinel_and_does_not_consume_recorded_slot(source, tmp_path, monkeypatch):
    """The separate smoke identity may finish without accessing any ledger destination."""
    registration = _call(source)
    sentinel = PoisonLedger()
    with monkeypatch.context() as patch:
        patch.setattr(outcomes, "ledger_append", _forbid)
        smoke = _publish(source, registration, sentinel, no_ledger=True)
        assert _replay(source, registration, sentinel, no_ledger=True) == smoke
    evidence = _evidence(source, registration.run_id)
    assert "native_outcome" not in evidence and "native_outcome_receipt" not in evidence
    assert _document(source, evidence["native_smoke"]) == smoke["outcome"]
    assert evidence["native_smoke_receipt"] == smoke["receipt_ref"]
    assert smoke["outcome"]["ledger_destination"] is None
    assert smoke["receipt"]["ledger_row_hash"] is None
    assert smoke["receipt"]["recording_mode"] == "smoke"
    ledger = tmp_path / "later" / "ledger.csv"
    recorded = _publish(source, registration, ledger, report=REPORT + "Later recorded result.\n")
    assert _rows(ledger) == [recorded["outcome"]["ledger_row"]]
    assert _evidence(source, registration.run_id)["native_smoke"] == evidence["native_smoke"]
    assert _replay(source, registration, sentinel, no_ledger=True) == smoke


def test_smoke_never_resolves_reads_creates_or_locks_nonexistent_ledger(source, tmp_path, monkeypatch):
    """Instrument actual path and open primitives rather than only mocking CSV append."""
    registration = _call(source)
    parent = tmp_path / "untouched-ledger-tree"
    ledger = parent / "ledger.csv"
    real_resolve = Path.resolve

    def guard_path(path):
        if isinstance(path, (str, bytes, os.PathLike)):
            decoded = Path(os.fsdecode(path))
            assert decoded != parent and parent not in decoded.parents, "smoke touched ledger tree"

    def resolve(path, *args, **kwargs):
        guard_path(path)
        return real_resolve(path, *args, **kwargs)

    def guarded(original):
        def call(path, *args, **kwargs):
            guard_path(path)
            return original(path, *args, **kwargs)
        return call

    with monkeypatch.context() as patch:
        patch.setattr(Path, "resolve", resolve)
        for owner in (builtins, io, os):
            patch.setattr(owner, "open", guarded(owner.open))
        first = _publish(source, registration, ledger, no_ledger=True)
        assert _replay(source, registration, ledger, no_ledger=True) == first
    assert not parent.exists()


@pytest.mark.parametrize("operation", ["publish", "replay"])
@pytest.mark.parametrize("field", ["registration", "intent", "report", "receipt"])
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_damaged_referenced_bytes_refuse_before_ledger(
        source, tmp_path, monkeypatch, field, damage, operation):
    """Replay verifies every stored dependency before any ledger access or mutation."""
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    result = _publish(source, registration, ledger)
    evidence = _evidence(source, registration.run_id)
    refs = {"registration": to_document(_ref(source, registration)),
            "intent": evidence["native_outcome"], "report": result["outcome"]["report_ref"],
            "receipt": evidence["native_outcome_receipt"]}
    path = source.store.verify(from_document(ArtifactRef, refs[field]))
    if damage == "missing":
        path.unlink()
    else:
        path.chmod(0o600)
        path.write_bytes(b"broken immutable evidence")
    before, objects, ledger_bytes = _catalog(source.conn), _objects(source.store), ledger.read_bytes()
    monkeypatch.setattr(outcomes, "ledger_append", _forbid)
    with pytest.raises(OpsError) as captured:
        (_replay if operation == "replay" else _publish)(source, registration, ledger)
    _assert_conflict(source, captured, registration)
    assert _catalog(source.conn) == before
    assert ledger.read_bytes() == ledger_bytes
    _assert_objects_preserved(source, objects)


@pytest.mark.parametrize("change", ["status", "date", "variant", "destination", "schema",
                                    "run_id", "attempted", "row", "failure_code", "report"])
def test_semantically_invalid_saved_intent_refuses_before_ledger(source, tmp_path, monkeypatch, change):
    """A well-hashed replacement cannot evade saved-intent binding and shape checks."""
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    result = _publish(source, registration, ledger)
    altered = json.loads(canonical_json(result["outcome"]))
    if change == "date":
        altered["ledger_row"]["date"] = "not-a-date"
    elif change == "row":
        altered["ledger_row"]["promoted"] = "True"
    elif change == "variant":
        altered["variant_id"] = "sha256:wrong-variant"
    elif change == "destination":
        altered["ledger_destination"] = content_hash("another destination")
    elif change == "schema":
        altered["schema_version"] = "invalid"
    elif change == "run_id":
        altered["run_id"] = "another-run"
    elif change == "attempted":
        altered["attempted_variants"] = False
    elif change == "report":
        altered["report_ref"] = None
    elif change == "failure_code":
        altered["failure_code"] = "INVALID_EXPERIMENT_SPEC"
    else:
        altered["ledger_row"]["stage"] = "failed"
    ref = source.store.publish_bytes(canonical_json(altered).encode(), schema_ref=outcomes.SCHEMA)
    evidence = _evidence(source, registration.run_id)
    evidence["native_outcome"] = to_document(ref)
    source.conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                        (canonical_json(evidence), registration.run_id))
    before, objects, ledger_bytes = _catalog(source.conn), _objects(source.store), ledger.read_bytes()
    monkeypatch.setattr(outcomes, "ledger_append", _forbid)
    with pytest.raises(OpsError) as captured:
        _replay(source, registration, ledger)
    _assert_conflict(source, captured, registration)
    assert _catalog(source.conn) == before
    assert ledger.read_bytes() == ledger_bytes
    _assert_objects_preserved(source, objects)


@pytest.mark.parametrize("window", ["report_object", "intent_object", "intent_catalog",
                                     "durable_ledger", "completion_object", "completion_catalog"])
def test_real_publication_failure_windows_reconcile_saved_intent(source, tmp_path, monkeypatch, window):
    """Crash after each actual durable effect, then recover only from persisted state."""
    registration = _call(source)
    ledger = tmp_path / "interrupted" / "ledger.csv"
    _clock(monkeypatch, "2026-01-02")
    original_catalog, original_objects = _catalog(source.conn), _objects(source.store)
    published = []
    real_publish, real_reserve, real_append = source.store.publish_bytes, outcomes._reserve, outcomes.ledger_append

    def publish(data, *, schema_ref):
        assert not source.conn.in_transaction
        ref = real_publish(data, schema_ref=schema_ref)
        published.append((schema_ref, ref))
        mapping = {"native_experiment_report.v1.0": "report_object",
                   outcomes.SCHEMA: "intent_object",
                   "native_experiment_completion.v1.0": "completion_object"}
        if mapping.get(schema_ref) == window:
            raise OSError("injected after real durable artifact publication")
        return ref

    def reserve(*args, **kwargs):
        result = real_reserve(*args, **kwargs)
        assert not source.conn.in_transaction
        key = args[2]
        if ((key == "native_outcome" and window == "intent_catalog")
                or (key == "native_outcome_receipt" and window == "completion_catalog")):
            raise OSError("injected after real catalog commit")
        return result

    def append(*args, **kwargs):
        assert not source.conn.in_transaction
        count = real_append(*args, **kwargs)
        if window == "durable_ledger":
            assert count == 1 and len(_rows(ledger)) == 1
            raise OSError("injected after actual durable CSV append")
        return count

    with monkeypatch.context() as patch:
        patch.setattr(source.store, "publish_bytes", publish)
        patch.setattr(outcomes, "_reserve", reserve)
        patch.setattr(outcomes, "ledger_append", append)
        with pytest.raises(OpsError, match="RESOURCE_UNAVAILABLE"):
            _publish(source, registration, ledger)
    assert published
    for _, ref in published:
        source.store.read_verified(ref)
    _assert_objects_preserved(source, original_objects)
    evidence = _evidence(source, registration.run_id)
    has_intent = window not in {"report_object", "intent_object"}
    has_row = window in {"durable_ledger", "completion_object", "completion_catalog"}
    assert ("native_outcome" in evidence) == has_intent
    assert ("native_outcome_receipt" in evidence) == (window == "completion_catalog")
    assert ledger.exists() == has_row
    if has_row:
        before_ledger = ledger.read_bytes()
        assert len(_rows(ledger)) == 1
    _clock(monkeypatch, "2026-03-04")
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression, "fit", _forbid)
    monkeypatch.setattr(source.repository, "scan", _forbid)
    if has_intent:
        saved = _document(source, evidence["native_outcome"])
        recovered = _replay(source, registration, ledger)
        assert recovered["outcome"] == saved
        assert recovered["outcome"]["ledger_row"]["date"] == "2026-01-02"
    else:
        assert _catalog(source.conn) == original_catalog
        assert _replay(source, registration, ledger) is None
        # No committed intent exists yet; repeat publication of the already-produced result.
        recovered = _publish(source, registration, ledger)
        assert recovered["outcome"]["ledger_row"]["date"] == "2026-03-04"
    assert _rows(ledger) == [recovered["outcome"]["ledger_row"]]
    if has_row:
        assert ledger.read_bytes() == before_ledger
    assert _replay(source, registration, ledger) == recovered
    assert _document(source, _evidence(source, registration.run_id)["native_outcome_receipt"]) == recovered["receipt"]


@pytest.mark.parametrize("options", [
    {"attempted": False}, {"attempted": 1}, {"report": ""}, {"report": "  "},
    {"report": b"bytes"}, {"no_ledger": 1}, {"ledger_path": None},
    {"problem": make_problem("FEATURE_LOOKAHEAD", "refused")},
    {"report": None, "problem": make_problem("SNAPSHOT_UNRESOLVED", "unresolved")},
])
def test_invalid_outcome_requests_have_no_publication_effect(source, tmp_path, options):
    """Only an explicit actual success or supported typed refusal can be published."""
    registration = _call(source)
    ledger = tmp_path / "invalid" / "ledger.csv"
    before, objects = _catalog(source.conn), _objects(source.store)
    kwargs = {"report": REPORT, "attempted": True, "ledger_path": ledger, **options}
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        outcomes.publish_native_outcome(source.conn, source.store, registration, **kwargs)
    assert _catalog(source.conn) == before
    assert _objects(source.store) == objects
    assert not ledger.parent.exists()


@pytest.mark.parametrize("no_ledger", [False, True])
def test_report_export_is_fixed_create_only_and_exactly_replayable(source, tmp_path, no_ledger):
    """Completed immutable reports may be materialized only at their fixed store address."""
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    result = _publish(source, registration, ledger, no_ledger=no_ledger)
    before, objects = _catalog(source.conn), _objects(source.store)
    path = outcomes.export_native_report(source.conn, source.store, registration,
                                          no_ledger=no_ledger, ledger_path=ledger)
    assert path == source.store.root / "native_reports" / registration.run_id / (
        "smoke" if no_ledger else "recorded") / "REPORT.md"
    assert path.read_text() == REPORT
    assert not path.is_symlink()
    assert path.stat().st_mode & 0o222 == 0
    assert outcomes.export_native_report(source.conn, source.store, registration,
                                          no_ledger=no_ledger, ledger_path=ledger) == path
    assert _catalog(source.conn) == before
    assert _objects(source.store) == objects
    assert _replay(source, registration, ledger, no_ledger=no_ledger) == result


@pytest.mark.parametrize("kind", ["foreign_bytes", "file_symlink", "parent_symlink"])
def test_report_export_rejects_existing_foreign_files_and_symlinks(source, tmp_path, kind):
    """Do not overwrite or follow a conflicting export file or ancestor symlink."""
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    _publish(source, registration, ledger)
    path = source.store.root / "native_reports" / registration.run_id / "recorded" / "REPORT.md"
    foreign = tmp_path / "foreign-report.txt"
    foreign.write_text("preexisting private unrelated bytes")
    if kind == "parent_symlink":
        path.parent.parent.mkdir(parents=True)
        directory = tmp_path / "foreign-directory"
        directory.mkdir()
        path.parent.symlink_to(directory, target_is_directory=True)
    else:
        path.parent.mkdir(parents=True)
        if kind == "foreign_bytes":
            path.write_bytes(foreign.read_bytes())
        else:
            path.symlink_to(foreign)
    before, objects, ledger_bytes = _catalog(source.conn), _objects(source.store), ledger.read_bytes()
    with pytest.raises(OpsError) as captured:
        outcomes.export_native_report(source.conn, source.store, registration, ledger_path=ledger)
    _assert_conflict(source, captured, registration)
    assert foreign.read_text() == "preexisting private unrelated bytes"
    if kind == "parent_symlink":
        assert path.parent.is_symlink() and not path.exists()
    else:
        assert path.read_bytes() == foreign.read_bytes()
        assert path.is_symlink() == (kind == "file_symlink")
    assert ledger.read_bytes() == ledger_bytes
    assert _catalog(source.conn) == before
    _assert_objects_preserved(source, objects)


@pytest.mark.parametrize("code", [None, "FEATURE_LOOKAHEAD", "HOLDOUT_ACCESS_DENIED",
                                  "EXPERIMENT_VARIANT_FAILED"])
def test_no_report_export_for_absent_or_refused_outcome(source, tmp_path, code):
    """A registration or typed refusal never gains an exportable current report."""
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    if code:
        problem = (make_data_problem if code == "HOLDOUT_ACCESS_DENIED" else make_problem)(code, "synthetic")
        _publish(source, registration, ledger, report=None, problem=problem, attempted=False)
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        outcomes.export_native_report(source.conn, source.store, registration, ledger_path=ledger)
    assert not list(source.store.root.rglob("REPORT.md"))


@pytest.mark.parametrize("slot", ["native_outcome", "native_outcome_receipt"])
def test_null_saved_reference_is_corruption_not_missing_intent(source, tmp_path, monkeypatch, slot):
    """A present but invalid slot cannot silently be forgotten or repaired."""
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    _publish(source, registration, ledger)
    evidence = _evidence(source, registration.run_id)
    evidence[slot] = None
    source.conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                        (canonical_json(evidence), registration.run_id))
    before, objects, ledger_bytes = _catalog(source.conn), _objects(source.store), ledger.read_bytes()
    monkeypatch.setattr(outcomes, "ledger_append", _forbid)
    with pytest.raises(OpsError) as captured:
        _replay(source, registration, ledger)
    _assert_conflict(source, captured, registration)
    assert _catalog(source.conn) == before
    assert ledger.read_bytes() == ledger_bytes
    _assert_objects_preserved(source, objects)


def test_competing_results_on_separate_connections_keep_one_winner(source, tmp_path):
    """A concurrent contradictory result loses without rewriting the winning intent."""
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    barrier = Barrier(2)

    def worker(report):
        conn = connect(source.catalog_path, must_exist=True)
        try:
            store = ArtifactStore(source.store.root)
            barrier.wait(timeout=15)
            try:
                result = outcomes.publish_native_outcome(conn, store, registration,
                                                          report=report, attempted=True, ledger_path=ledger)
                return result, None
            except OpsError as error:
                return None, error
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, [REPORT, REPORT + "Competing result.\n"]))
    successes = [result for result, error in results if error is None]
    failures = [error for result, error in results if error is not None]
    assert len(successes) == len(failures) == 1
    assert failures[0].code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert _document(source, failures[0].problem.details["refusal_ref"])["failure_code"] == failures[0].code
    assert _rows(ledger) == [successes[0]["outcome"]["ledger_row"]]
    assert _replay(source, registration, ledger) == successes[0]


def test_refusal_details_are_frozen_and_part_of_global_result_identity(source, tmp_path):
    """Event-specific typed failure details remain recoverable and cannot drift on retry."""
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    details = {"event_ids": [source.events[0]["event_id"]], "reason": "synthetic-lookahead"}
    problem = make_problem("FEATURE_LOOKAHEAD", "private raw exception", details=details)
    result = _publish(source, registration, ledger, report=None, problem=problem, attempted=False)
    assert result["outcome"]["failure_details"] == details
    assert b"private raw exception" not in canonical_json(result["outcome"]).encode()
    assert _replay(source, registration, ledger) == result
    before, objects, ledger_bytes = _catalog(source.conn), _objects(source.store), ledger.read_bytes()
    changed = make_problem("FEATURE_LOOKAHEAD", "private raw exception", details={**details, "reason": "changed"})
    with pytest.raises(OpsError) as captured:
        _publish(source, registration, ledger, report=None, problem=changed, attempted=False)
    _assert_conflict(source, captured, registration)
    assert _catalog(source.conn) == before
    assert ledger.read_bytes() == ledger_bytes
    _assert_objects_preserved(source, objects)


@pytest.mark.parametrize("field", ["intent", "report", "receipt"])
def test_wrong_typed_artifact_schema_fails_closed_even_with_valid_bytes(source, tmp_path, monkeypatch, field):
    """Schema substitution is corrupt evidence even when content addressing verifies."""
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    result = _publish(source, registration, ledger)
    evidence = _evidence(source, registration.run_id)
    if field == "report":
        outcome = result["outcome"]
        reference = from_document(ArtifactRef, outcome["report_ref"])
        replacement = source.store.publish_bytes(source.store.read_verified(reference), schema_ref="foreign.v1.0")
        outcome["report_ref"] = to_document(replacement)
        replacement = source.store.publish_bytes(canonical_json(outcome).encode(), schema_ref=outcomes.SCHEMA)
        evidence["native_outcome"] = to_document(replacement)
    else:
        key = "native_outcome" if field == "intent" else "native_outcome_receipt"
        reference = from_document(ArtifactRef, evidence[key])
        replacement = source.store.publish_bytes(source.store.read_verified(reference), schema_ref="foreign.v1.0")
        evidence[key] = to_document(replacement)
    # Remove completion for the report case so the report check stands independently.
    if field == "report":
        del evidence["native_outcome_receipt"]
    source.conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                        (canonical_json(evidence), registration.run_id))
    before, objects, ledger_bytes = _catalog(source.conn), _objects(source.store), ledger.read_bytes()
    monkeypatch.setattr(outcomes, "ledger_append", _forbid)
    with pytest.raises(OpsError) as captured:
        _replay(source, registration, ledger)
    _assert_conflict(source, captured, registration)
    assert _catalog(source.conn) == before
    assert ledger.read_bytes() == ledger_bytes
    _assert_objects_preserved(source, objects)


@pytest.mark.parametrize("operation", ["publish", "replay", "export"])
@pytest.mark.parametrize("kind", [OSError, RuntimeError])
def test_destination_resolution_errors_are_typed_and_redacted(source, tmp_path, monkeypatch, operation, kind):
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    before = _catalog(source.conn)
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == ledger:
            raise kind("secret-local-path")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    function = {"publish": _publish, "replay": _replay,
                "export": lambda s, r, p: outcomes.export_native_report(s.conn, s.store, r, ledger_path=p)}[operation]
    with pytest.raises(OpsError) as captured:
        function(source, registration, ledger)
    assert captured.value.code == "RESOURCE_UNAVAILABLE"
    assert "secret-local-path" not in str(captured.value)
    assert _catalog(source.conn) == before
    assert not ledger.exists()


@pytest.mark.parametrize("fault", ["mkdir", "link", "directory_sync"])
def test_report_export_storage_errors_are_typed_and_replayable(source, tmp_path, monkeypatch, fault):
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    result = _publish(source, registration, ledger)
    before = ledger.read_bytes()
    report_dir = source.store.root / "native_reports" / registration.run_id / "recorded"
    target, name = ((outcomes.foundation, "ensure_directory") if fault == "mkdir" else
                    (outcomes.os, "link") if fault == "link" else
                    (outcomes.foundation, "fsync_directory"))
    original = getattr(target, name)

    def fail(*args, **kwargs):
        if any(Path(arg) == report_dir or Path(arg) == report_dir / "REPORT.md"
               for arg in args if isinstance(arg, (str, Path))):
            raise OSError("private-filesystem-diagnostic")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(target, name, fail)
        with pytest.raises(OpsError) as captured:
            outcomes.export_native_report(source.conn, source.store, registration, ledger_path=ledger)
        assert captured.value.code == "RESOURCE_UNAVAILABLE"
        assert "private-filesystem" not in str(captured.value)
    assert ledger.read_bytes() == before
    exported = outcomes.export_native_report(source.conn, source.store, registration, ledger_path=ledger)
    assert exported.read_text() == REPORT
    assert _replay(source, registration, ledger) == result


@pytest.mark.parametrize("no_ledger", [False, True])
@pytest.mark.parametrize("operation", ["publish", "replay"])
def test_dangling_completion_never_replaces_missing_intent(source, tmp_path, no_ledger, operation):
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    _publish(source, registration, ledger, no_ledger=no_ledger)
    evidence = _evidence(source, registration.run_id)
    del evidence["native_smoke" if no_ledger else "native_outcome"]
    source.conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                        (canonical_json(evidence), registration.run_id))
    before, objects = _catalog(source.conn), _objects(source.store)
    ledger_bytes = ledger.read_bytes() if ledger.exists() else None
    with pytest.raises(OpsError) as captured:
        if operation == "publish":
            _publish(source, registration, ledger, report=REPORT + "Changed", no_ledger=no_ledger)
        else:
            _replay(source, registration, ledger, no_ledger=no_ledger)
    _assert_conflict(source, captured, registration)
    assert _catalog(source.conn) == before
    assert (ledger.read_bytes() if ledger.exists() else None) == ledger_bytes
    _assert_objects_preserved(source, objects)


def test_distinct_registered_arms_and_legacy_rows_do_not_share_replay_identity(source, tmp_path):
    registration = _call(source)
    other = _call(source, spec=_spec(primary_arm_id="other", arms=["other"]))
    ledger = tmp_path / "ledger.csv"
    legacy = {"id": _spec().experiment_id, "spec_hash": "legacy-spec", "date": "2026-01-01",
              "stage": "ran", "oos_mean_mid": "0.1", "sharpe_trade": "0.2", "promoted": "False"}
    outcomes.ledger_append([legacy], path=ledger)
    prefix = ledger.read_bytes()
    first, second = _publish(source, registration, ledger), _publish(source, other, ledger)
    assert ledger.read_bytes().startswith(prefix)
    assert _rows(ledger) == [legacy, first["outcome"]["ledger_row"], second["outcome"]["ledger_row"]]
    before = ledger.read_bytes()
    assert _replay(source, registration, ledger) == first
    assert _replay(source, other, ledger) == second
    assert ledger.read_bytes() == before


def test_same_full_legacy_key_with_different_bytes_refuses_without_rewriting(source, tmp_path):
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    row = {"id": _spec().experiment_id, "spec_hash": registration.variant_id, "date": "2020-01-01",
           "stage": "ran", "oos_mean_mid": "legacy-value", "sharpe_trade": "", "promoted": "False"}
    outcomes.ledger_append([row], path=ledger)
    before = ledger.read_bytes()
    with pytest.raises(OpsError) as captured:
        _publish(source, registration, ledger)
    _assert_conflict(source, captured, registration)
    assert ledger.read_bytes() == before
    evidence = _evidence(source, registration.run_id)
    assert "native_outcome_receipt" not in evidence


@pytest.mark.parametrize("field", ["snapshot", "holdouts"])
@pytest.mark.parametrize("operation", ["publish", "replay"])
def test_derived_provenance_requires_verified_registration(source, tmp_path, monkeypatch, field, operation):
    """Omitted duplicate pins cannot substitute for the immutable registration's pins."""
    registration = _call(source)
    ledger = tmp_path / "ledger.csv"
    _publish(source, registration, ledger)
    document = registration.document
    if field == "snapshot":
        document[field]["snapshot_id"] = "changed-snapshot"
    else:
        document[field]["holdout_as_of_month"] = "1900-01"
    changed = replace(registration, binding_json=canonical_json(document).encode())
    before, objects, ledger_bytes = _catalog(source.conn), _objects(source.store), ledger.read_bytes()
    monkeypatch.setattr(outcomes, "ledger_append", _forbid)
    with pytest.raises(OpsError) as captured:
        (_publish if operation == "publish" else _replay)(source, changed, ledger)
    _assert_conflict(source, captured, changed)
    assert _catalog(source.conn) == before
    assert ledger.read_bytes() == ledger_bytes
    _assert_objects_preserved(source, objects)
