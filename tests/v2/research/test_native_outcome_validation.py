"""Validate pending native intents before any ledger or completion side effect."""
import json
from copy import deepcopy

import pytest

from engine.v2.data.errors import make_problem as make_data_problem
from engine.v2.foundation import canonical_json, content_hash, to_document
from engine.v2.ops.errors import OpsError, make_problem
from experiments import lib
from experiments import native_outcomes as outcomes
from tests.v2.research.test_native_outcomes import (
    REPORT,
    PoisonLedger,
    _assert_conflict,
    _assert_objects_preserved,
    _document,
    _publish,
    _replay,
    _rows,
)
from tests.v2.research.test_native_registration import (
    _call,
    _catalog,
    _evidence,
    _forbid,
    _objects,
)
from tests.v2.research.test_native_registration import (
    source as source,
)

FIELDS = ("schema_version", "run_id", "variant_id", "attempted_variants", "failure_code",
          "failure_details", "report_ref", "ledger_row", "ledger_destination")
UNRELATED_ROW = {"id": "synthetic-unrelated", "spec_hash": "synthetic-unrelated-hash",
                 "date": "2026-01-01", "stage": "ran", "oos_mean_mid": "",
                 "sharpe_trade": "", "promoted": "False"}


@pytest.fixture(autouse=True)
def safe_default_ledger(tmp_path, monkeypatch):
    """Even an empty-path regression may only touch this test's synthetic fallback."""
    path = tmp_path / "safe-default" / "ledger.csv"
    monkeypatch.setattr(lib, "LEDGER_PATH", path)
    return path


def _csv_state(path):
    return path.read_bytes() if path.exists() else None


def _seed_csv(path, existing):
    if existing:
        lib.ledger_append([UNRELATED_ROW], path=path)
    return _csv_state(path)


def _invoke(operation, source, registration, ledger, *, no_ledger=False):
    if operation == "publish":
        return _publish(source, registration, ledger, no_ledger=no_ledger)
    if operation == "replay":
        return _replay(source, registration, ledger, no_ledger=no_ledger)
    return outcomes.export_native_report(source.conn, source.store, registration,
                                         ledger_path=ledger, no_ledger=no_ledger)


def _pending(source, registration, ledger, monkeypatch, *, no_ledger=False, **kwargs):
    """Interrupt actual publication after the committed intent, before completion."""
    key = "native_smoke" if no_ledger else "native_outcome"
    original = outcomes._reserve

    def interrupt(*args, **options):
        result = original(*args, **options)
        if args[2] == key:
            raise OSError("synthetic interruption after intent commit")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(outcomes, "_reserve", interrupt)
        with pytest.raises(OpsError, match="RESOURCE_UNAVAILABLE"):
            _publish(source, registration, ledger, no_ledger=no_ledger, **kwargs)
    evidence = _evidence(source, registration.run_id)
    assert key in evidence and key + "_receipt" not in evidence
    outcome = _document(source, evidence[key])
    assert set(outcome) == set(FIELDS)
    return key, outcome


class EmptyPath:
    def __fspath__(self):
        return ""


@pytest.mark.parametrize("operation", ["publish", "replay", "export"])
@pytest.mark.parametrize("pathlike", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_empty_destination_is_r1_before_publication(
        source, monkeypatch, safe_default_ledger, operation, pathlike, existing):
    """An empty explicit destination cannot hash cwd and fall back to another CSV."""
    registration = _call(source)
    ledger_before = _seed_csv(safe_default_ledger, existing)
    before, objects = _catalog(source.conn), _objects(source.store)
    with pytest.raises(OpsError) as captured:
        _invoke(operation, source, registration, EmptyPath() if pathlike else "")
    assert captured.value.code == "INVALID_EXPERIMENT_SPEC"
    assert _catalog(source.conn) == before
    assert _objects(source.store) == objects
    assert _csv_state(safe_default_ledger) == ledger_before
    assert not list(source.store.root.rglob("REPORT.md"))


@pytest.mark.parametrize("operation", ["publish", "replay", "export"])
@pytest.mark.parametrize("kind", ["file_symlink", "dangling_symlink", "directory"])
def test_invalid_destination_is_r1_without_conflict_evidence(
        source, tmp_path, safe_default_ledger, operation, kind):
    """Invalid caller paths refuse before ledger access or conflict publication."""
    registration = _call(source)
    destination, target = tmp_path / "destination", tmp_path / "target.csv"
    fallback_before = _seed_csv(safe_default_ledger, True)
    target_before = _seed_csv(target, kind == "file_symlink")
    if kind == "directory":
        destination.mkdir()
    else:
        destination.symlink_to(target)
    before, objects = _catalog(source.conn), _objects(source.store)
    with pytest.raises(OpsError) as captured:
        _invoke(operation, source, registration, destination)
    assert captured.value.code == "INVALID_EXPERIMENT_SPEC"
    assert _catalog(source.conn) == before
    assert _objects(source.store) == objects
    assert _csv_state(target) == target_before
    assert _csv_state(safe_default_ledger) == fallback_before
    assert not destination.with_name(destination.name + ".append.lock").exists()
    assert not list(source.store.root.rglob("REPORT.md"))


@pytest.mark.parametrize("operation", ["publish", "replay", "export"])
@pytest.mark.parametrize("change", ["pathlike", "falsey_pathlike", "cwd"])
def test_one_frozen_destination_is_hashed_and_written(
        source, tmp_path, monkeypatch, safe_default_ledger, operation, change):
    """A mutable PathLike or a later cwd change cannot redirect the actual append."""
    registration = _call(source)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    ledger, other = first / "ledger.csv", second / "ledger.csv"
    fallback_before = _seed_csv(safe_default_ledger, True)
    if operation != "publish":
        _pending(source, registration, ledger, monkeypatch)
    calls = []

    class ChangingPath:
        def __fspath__(self):
            calls.append("fspath")
            return str(ledger if len(calls) == 1 else other)

        def __bool__(self):
            return change != "falsey_pathlike"

    monkeypatch.chdir(first)
    destination = "ledger.csv" if change == "cwd" else ChangingPath()
    real_append = outcomes.ledger_append

    def append(*args, **kwargs):
        monkeypatch.chdir(second)
        return real_append(*args, **kwargs)

    monkeypatch.setattr(outcomes, "ledger_append", append)
    _invoke(operation, source, registration, destination)
    evidence = _evidence(source, registration.run_id)
    saved = _document(source, evidence["native_outcome"])
    assert saved["ledger_destination"] == content_hash(str(ledger))
    assert _rows(ledger) == [saved["ledger_row"]]
    assert not other.exists()
    assert calls == ([] if change == "cwd" else ["fspath"])
    assert _csv_state(safe_default_ledger) == fallback_before


@pytest.mark.parametrize("operation", ["publish", "replay", "export"])
def test_smoke_keeps_poison_destination_untouched(
        source, monkeypatch, safe_default_ledger, operation):
    """All public smoke paths, including pending export, ignore ledger coercion."""
    registration = _call(source)
    poison = PoisonLedger()
    fallback_before = _seed_csv(safe_default_ledger, True)
    if operation != "publish":
        _pending(source, registration, poison, monkeypatch, no_ledger=True)
    monkeypatch.setattr(outcomes, "ledger_append", _forbid)
    _invoke(operation, source, registration, poison, no_ledger=True)
    evidence = _evidence(source, registration.run_id)
    assert "native_smoke_receipt" in evidence and "native_outcome" not in evidence
    assert _csv_state(safe_default_ledger) == fallback_before


def _mutations(outcome, group):
    """Independent well-hashed documents exercise every required field and relation."""
    if group == "missing":
        for field in FIELDS:
            altered = deepcopy(outcome)
            del altered[field]
            yield "missing-" + field, altered
    elif group == "types":
        for field in FIELDS:
            altered = deepcopy(outcome)
            altered[field] = []
            yield "wrong-type-" + field, altered
        for value in (None, {}, [], "outcome", 1):
            yield "non-object-" + repr(value), value
    elif group == "details":
        for value in (None, [], "details", {"unexpected": "success detail"}):
            yield "success-details-" + repr(value), {**outcome, "failure_details": value}
        refused = {**outcome, "failure_code": "FEATURE_LOOKAHEAD", "report_ref": None,
                   "attempted_variants": 0, "ledger_row": {**outcome["ledger_row"], "stage": "refused"}}
        for value in (None, [], "details"):
            yield "refusal-details-" + repr(value), {**refused, "failure_details": value}
        del refused["failure_details"]
        yield "refusal-details-missing", refused
    elif group == "values":
        yield "extra-field", {**outcome, "unexpected": True}
        for value in (False, True, -1, 0, 2, 1.0, "1"):
            yield "attempt-count-" + repr(value), {**outcome, "attempted_variants": value}
        for value in ("", "INVALID_EXPERIMENT_SPEC", "SNAPSHOT_UNRESOLVED",
                      "EXPERIMENT_IDENTITY_CONFLICT", "UNKNOWN", 1, {}):
            yield "failure-code-" + repr(value), {**outcome, "failure_code": value}
        for field in ("schema_version", "run_id", "variant_id", "ledger_destination"):
            yield "wrong-" + field, {**outcome, field: "different-identity"}
    elif group == "rows":
        for field in outcome["ledger_row"]:
            row = {**outcome["ledger_row"]}
            del row[field]
            yield "row-missing-" + field, {**outcome, "ledger_row": row}
            yield "row-type-" + field, {**outcome, "ledger_row": {**outcome["ledger_row"], field: []}}
        for field, value in (("date", "2026-02-30"), ("date", "20260101"),
                             ("stage", "failed"), ("stage", "refused"), ("promoted", "True"),
                             ("id", "another"), ("spec_hash", "another"), ("unexpected", "")):
            yield "row-value-" + field + "-" + value, {
                **outcome, "ledger_row": {**outcome["ledger_row"], field: value}}
    elif group == "references":
        for value in (None, {}, "report", {**outcome["report_ref"], "unexpected": True},
                      {**outcome["report_ref"], "schema_ref": "foreign.v1.0"}):
            yield "report-reference-" + repr(value), {**outcome, "report_ref": value}
        for field in outcome["report_ref"]:
            reference = {**outcome["report_ref"]}
            del reference[field]
            yield "report-missing-" + field, {**outcome, "report_ref": reference}
            yield "report-type-" + field, {
                **outcome, "report_ref": {**outcome["report_ref"], field: []}}


def _save_intent(source, registration, key, document):
    # Preserve malformed JSON scalar types (canonical_json normalizes 1.0 to 1).
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ref = source.store.publish_bytes(payload, schema_ref=outcomes.SCHEMA)
    assert json.loads(source.store.read_verified(ref)) == document
    evidence = _evidence(source, registration.run_id)
    assert key + "_receipt" not in evidence
    evidence[key] = to_document(ref)
    source.conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                        (canonical_json(evidence), registration.run_id))


def _assert_saved_refusal(source, registration, ledger, monkeypatch, operation, no_ledger, label):
    """Only private refusal evidence may be added; no CSV helper or completion runs."""
    before, objects, ledger_before = _catalog(source.conn), _objects(source.store), _csv_state(ledger)
    parent_existed = ledger.parent.exists()
    original = source.store.publish_bytes

    def only_refusal(data, *, schema_ref):
        assert schema_ref == "native_experiment_refusal.v1.0", label
        return original(data, schema_ref=schema_ref)

    def forbid(*_args, **_kwargs):
        raise AssertionError("invalid saved intent reached ledger/completion: " + label)

    with monkeypatch.context() as patch:
        patch.setattr(source.store, "publish_bytes", only_refusal)
        patch.setattr(outcomes, "ledger_append", forbid)
        patch.setattr(outcomes, "_reserve", forbid)
        with pytest.raises(OpsError) as captured:
            _invoke(operation, source, registration, PoisonLedger() if no_ledger else ledger,
                    no_ledger=no_ledger)
    _assert_conflict(source, captured, registration)
    assert _catalog(source.conn) == before, label
    assert _csv_state(ledger) == ledger_before, label
    assert ledger.parent.exists() == parent_existed, label
    _assert_objects_preserved(source, objects)
    assert not list(source.store.root.rglob("REPORT.md")), label


@pytest.mark.parametrize("no_ledger", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("attempted", [False, True])
@pytest.mark.parametrize("code", ["FEATURE_LOOKAHEAD", "HOLDOUT_ACCESS_DENIED", "EXPERIMENT_VARIANT_FAILED"])
def test_complete_pending_refusal_preserves_typed_details(
        source, tmp_path, monkeypatch, no_ledger, existing, attempted, code):
    """Legitimate structured failure details and both attempt counts still recover."""
    registration = _call(source)
    ledger = tmp_path / "target" / "ledger.csv"
    ledger_before = _seed_csv(ledger, existing)
    details = {"reason": "synthetic-rejection", "event_ids": [source.events[0]["event_id"]]}
    factory = make_data_problem if code == "HOLDOUT_ACCESS_DENIED" else make_problem
    key, saved = _pending(source, registration, ledger, monkeypatch, no_ledger=no_ledger,
                          report=None, attempted=attempted,
                          problem=factory(code, "synthetic failure", details=details))
    result = _replay(source, registration, PoisonLedger() if no_ledger else ledger, no_ledger=no_ledger)
    assert result["outcome"] == saved
    assert saved["failure_details"] == details and saved["attempted_variants"] == int(attempted)
    assert saved["report_ref"] is result["receipt"]["report_ref"] is None
    assert key + "_receipt" in _evidence(source, registration.run_id)
    if no_ledger:
        assert _csv_state(ledger) == ledger_before
    else:
        assert _rows(ledger) == ([UNRELATED_ROW] if existing else []) + [saved["ledger_row"]]
    assert not list(source.store.root.rglob("REPORT.md"))


@pytest.mark.parametrize("operation", ["replay", "export"])
@pytest.mark.parametrize("no_ledger", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("group", ["missing", "types", "details", "values", "rows", "references"])
def test_incomplete_or_invalid_pending_intent_is_r6_before_effects(
        source, tmp_path, monkeypatch, operation, no_ledger, existing, group):
    """A valid artifact hash and committed intent cannot authorize malformed results."""
    registration = _call(source)
    ledger = tmp_path / "target" / "ledger.csv"
    _seed_csv(ledger, existing)
    key, outcome = _pending(source, registration, ledger, monkeypatch, no_ledger=no_ledger)
    for label, document in _mutations(outcome, group):
        _save_intent(source, registration, key, document)
        _assert_saved_refusal(source, registration, ledger, monkeypatch, operation, no_ledger, label)


@pytest.mark.parametrize("operation", ["replay", "export"])
@pytest.mark.parametrize("no_ledger", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("report_bytes", [b"", b" \t\r\n", "\u2003\u00a0".encode(),
                                         b"\xff\xfe", b"\xed\xa0\x80", b"\xed\xbf\xbf"])
def test_invalid_pending_report_bytes_are_r6_before_effects(
        source, tmp_path, monkeypatch, operation, no_ledger, existing, report_bytes):
    """Empty, blank, non-UTF-8 and encoded surrogates remain invalid when well-hashed."""
    registration = _call(source)
    ledger = tmp_path / "target" / "ledger.csv"
    _seed_csv(ledger, existing)
    key, outcome = _pending(source, registration, ledger, monkeypatch, no_ledger=no_ledger)
    ref = source.store.publish_bytes(report_bytes, schema_ref="native_experiment_report.v1.0")
    assert source.store.read_verified(ref) == report_bytes
    _save_intent(source, registration, key, {**outcome, "report_ref": to_document(ref)})
    _assert_saved_refusal(source, registration, ledger, monkeypatch, operation, no_ledger, repr(report_bytes))


@pytest.mark.parametrize("operation", ["replay", "export"])
@pytest.mark.parametrize("no_ledger", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_complete_pending_success_still_reconciles_normally(
        source, tmp_path, monkeypatch, operation, no_ledger, existing):
    """Fail-closed validation preserves legitimate interrupted completion and export."""
    registration = _call(source)
    ledger = tmp_path / "target" / "ledger.csv"
    ledger_before = _seed_csv(ledger, existing)
    key, saved = _pending(source, registration, ledger, monkeypatch, no_ledger=no_ledger)
    objects = _objects(source.store)
    result = _invoke(operation, source, registration, PoisonLedger() if no_ledger else ledger,
                     no_ledger=no_ledger)
    if operation == "export":
        assert result.read_bytes() == REPORT.encode()
    evidence = _evidence(source, registration.run_id)
    assert _document(source, evidence[key]) == saved
    assert _document(source, evidence[key + "_receipt"])["outcome_ref"] == evidence[key]
    if no_ledger:
        assert _csv_state(ledger) == ledger_before
    else:
        assert _rows(ledger) == ([UNRELATED_ROW] if existing else []) + [saved["ledger_row"]]
    _assert_objects_preserved(source, objects)


def _transaction_effects(source, root, ledger, fallback):
    """Capture durable bytes, lock files, exports, and even newly created directories."""
    return {
        "objects": _objects(source.store),
        "store_files": {str(path.relative_to(source.store.root)): path.read_bytes()
                        for path in source.store.root.rglob("*") if path.is_file()},
        "target_csv": _csv_state(ledger),
        "default_csv": _csv_state(fallback),
        "locks": {str(path.relative_to(root)): path.read_bytes()
                  for path in root.rglob("*.lock") if path.is_file()},
        "reports": {str(path.relative_to(root)): path.read_bytes()
                    for path in root.rglob("REPORT.md")},
        "directories": sorted(str(path.relative_to(root))
                              for path in root.rglob("*") if path.is_dir()),
    }


@pytest.mark.parametrize("operation", ["publish", "replay", "export"])
@pytest.mark.parametrize("no_ledger", [False, True], ids=["recorded", "smoke"])
@pytest.mark.parametrize("state,existing", [("fresh", False), ("pending", False),
                                            ("pending", True), ("completed", True)])
@pytest.mark.parametrize("finish", ["rollback", "commit"])
def test_active_caller_transaction_is_r1_before_native_effects(
        source, tmp_path, monkeypatch, safe_default_ledger, operation, no_ledger,
        state, existing, finish):
    """Public entrypoints cannot publish, record, or hijack a caller's transaction."""
    registration = _call(source)
    ledger = tmp_path / "target" / "ledger.csv"
    _seed_csv(ledger, existing)
    _seed_csv(safe_default_ledger, True)
    if state == "pending":
        key, saved = _pending(source, registration, ledger, monkeypatch, no_ledger=no_ledger)
    elif state == "completed":
        _publish(source, registration, ledger, no_ledger=no_ledger)
        report = outcomes.export_native_report(source.conn, source.store, registration,
                                                ledger_path=ledger, no_ledger=no_ledger)
        assert report.read_bytes() == REPORT.encode()

    source.conn.execute("CREATE TABLE caller_owned (value TEXT NOT NULL)")
    committed = _catalog(source.conn)
    source.conn.execute("BEGIN")
    source.conn.execute("INSERT INTO caller_owned VALUES ('caller-owned-uncommitted')")
    assert source.conn.in_transaction
    before = _catalog(source.conn)
    effects = _transaction_effects(source, tmp_path, ledger, safe_default_ledger)

    with pytest.raises(OpsError) as captured:
        _invoke(operation, source, registration, ledger, no_ledger=no_ledger)

    assert captured.value.code == "INVALID_EXPERIMENT_SPEC"
    assert captured.value.problem.retryable is False
    assert source.conn.in_transaction
    assert _catalog(source.conn) == before
    assert _transaction_effects(source, tmp_path, ledger, safe_default_ledger) == effects
    getattr(source.conn, finish)()
    assert not source.conn.in_transaction
    assert _catalog(source.conn) == (committed if finish == "rollback" else before)
    assert _transaction_effects(source, tmp_path, ledger, safe_default_ledger) == effects

    if state == "pending" and finish == "rollback":
        result = _replay(source, registration, ledger, no_ledger=no_ledger)
        assert result["outcome"] == saved
        assert key + "_receipt" in _evidence(source, registration.run_id)
        assert source.conn.execute("SELECT COUNT(*) FROM caller_owned").fetchone()[0] == 0
        assert not source.conn.in_transaction
        if no_ledger:
            assert _csv_state(ledger) == effects["target_csv"]
        else:
            assert _rows(ledger) == ([UNRELATED_ROW] if existing else []) + [saved["ledger_row"]]
        assert _csv_state(safe_default_ledger) == effects["default_csv"]
        assert not list(source.store.root.rglob("REPORT.md"))


@pytest.mark.parametrize("operation", ["publish", "replay", "export"])
def test_active_caller_transaction_refuses_before_dependency_access(
        source, tmp_path, monkeypatch, safe_default_ledger, operation):
    """Recorded calls refuse before path coercion, catalog SQL, or artifact access."""
    registration = _call(source)
    ledger = tmp_path / "untouched" / "ledger.csv"
    _seed_csv(safe_default_ledger, True)
    source.conn.execute("CREATE TABLE caller_owned (value TEXT NOT NULL)")
    committed = _catalog(source.conn)
    source.conn.execute("BEGIN")
    source.conn.execute("INSERT INTO caller_owned VALUES ('caller-owned-uncommitted')")
    before = _catalog(source.conn)
    effects = _transaction_effects(source, tmp_path, ledger, safe_default_ledger)
    statements = []
    source.conn.set_trace_callback(statements.append)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(source.store, "read_verified", _forbid)
            patch.setattr(source.store, "publish_bytes", _forbid)
            with pytest.raises(OpsError) as captured:
                _invoke(operation, source, registration, PoisonLedger())
    finally:
        source.conn.set_trace_callback(None)
    assert captured.value.code == "INVALID_EXPERIMENT_SPEC"
    assert statements == []
    assert source.conn.in_transaction
    assert _catalog(source.conn) == before
    assert _transaction_effects(source, tmp_path, ledger, safe_default_ledger) == effects
    source.conn.rollback()
    assert not source.conn.in_transaction
    assert _catalog(source.conn) == committed


NESTED_REFUSAL_DETAILS = {
    "reason": "synthetic refusal",
    "path": ["features", ["candidate", {"seen": ["early", "late"]}]],
    "checks": [{"window": [1, 2]}, ["first", "second"]],
}


def _nested_refusal_problem(code, tuples):
    """Use real typed problems with independently specified JSON-list expectations."""
    details = {
        "reason": "synthetic refusal",
        "path": ("features", ("candidate", {"seen": ("early", "late")})),
        "checks": [{"window": (1, 2)}, ("first", "second")],
    } if tuples else deepcopy(NESTED_REFUSAL_DETAILS)
    factory = make_data_problem if code == "HOLDOUT_ACCESS_DENIED" else make_problem
    return factory(code, "synthetic nested refusal", details=details)


@pytest.mark.parametrize("code,stage", [("FEATURE_LOOKAHEAD", "refused"),
    ("HOLDOUT_ACCESS_DENIED", "refused"), ("EXPERIMENT_VARIANT_FAILED", "failed")])
@pytest.mark.parametrize("no_ledger", [False, True], ids=["recorded", "smoke"])
@pytest.mark.parametrize("attempted", [False, True])
@pytest.mark.parametrize("tuples_first", [True, False], ids=["tuple-first", "list-first"])
def test_canonical_refusal_publish_repeat_and_replay_keep_original_bytes(
        source, tmp_path, monkeypatch, code, stage, no_ledger, attempted, tuples_first):
    """Nested tuple/list equivalents complete once and retain their first date."""
    from tests.v2.research.test_native_outcomes import _clock

    registration = _call(source)
    ledger = tmp_path / "canonical-refusal" / "ledger.csv"
    ledger_before = _seed_csv(ledger, True)
    destination = PoisonLedger() if no_ledger else ledger
    objects_before = _objects(source.store)
    problem = _nested_refusal_problem(code, tuples_first)
    _clock(monkeypatch, "2026-01-02")
    first = _publish(source, registration, destination, no_ledger=no_ledger,
                     report=None, attempted=attempted, problem=problem)
    outcome = first["outcome"]
    assert outcome["failure_code"] == code
    assert outcome["failure_details"] == NESTED_REFUSAL_DETAILS
    assert outcome["attempted_variants"] == int(attempted)
    assert outcome["ledger_row"]["stage"] == stage
    assert outcome["ledger_row"]["date"] == "2026-01-02"
    assert outcome["report_ref"] is first["receipt"]["report_ref"] is None
    key = "native_smoke" if no_ledger else "native_outcome"
    evidence = _evidence(source, registration.run_id)
    assert _document(source, evidence[key])["failure_details"] == NESTED_REFUSAL_DETAILS
    assert evidence[key + "_receipt"] == first["receipt_ref"]
    before, objects, ledger_bytes = _catalog(source.conn), _objects(source.store), _csv_state(ledger)
    assert len(objects.keys() - objects_before.keys()) == 2  # Outcome and completion only.
    if no_ledger:
        assert ledger_bytes == ledger_before
    else:
        assert _rows(ledger) == [UNRELATED_ROW, outcome["ledger_row"]]

    _clock(monkeypatch, "2026-02-10")
    for repeated in (problem, _nested_refusal_problem(code, not tuples_first)):
        assert _publish(source, registration, destination, no_ledger=no_ledger,
                        report=None, attempted=attempted, problem=repeated) == first
        assert _replay(source, registration, destination, no_ledger=no_ledger) == first
        assert _catalog(source.conn) == before
        assert _objects(source.store) == objects
        assert _csv_state(ledger) == ledger_bytes
    with pytest.raises(OpsError) as captured:
        outcomes.export_native_report(source.conn, source.store, registration,
                                      ledger_path=destination, no_ledger=no_ledger)
    assert captured.value.code == "INVALID_EXPERIMENT_SPEC"
    assert _catalog(source.conn) == before
    assert _objects(source.store) == objects
    assert _csv_state(ledger) == ledger_bytes
    assert not list(tmp_path.rglob("REPORT.md"))


@pytest.mark.parametrize("code", ["FEATURE_LOOKAHEAD", "HOLDOUT_ACCESS_DENIED",
                                  "EXPERIMENT_VARIANT_FAILED"])
@pytest.mark.parametrize("no_ledger", [False, True], ids=["recorded", "smoke"])
@pytest.mark.parametrize("change", ["nested-value", "nested-order", "boolean-for-integer"])
def test_canonical_refusal_changed_nested_details_are_r6_without_replacement(
        source, tmp_path, monkeypatch, code, no_ledger, change):
    """Canonical equality preserves nested values and order as immutable identity."""
    registration = _call(source)
    ledger = tmp_path / "canonical-conflict" / "ledger.csv"
    _seed_csv(ledger, True)
    destination = PoisonLedger() if no_ledger else ledger
    first = _publish(source, registration, destination, no_ledger=no_ledger,
                     report=None, problem=_nested_refusal_problem(code, True))
    before, objects, ledger_bytes = _catalog(source.conn), _objects(source.store), _csv_state(ledger)
    altered = deepcopy(NESTED_REFUSAL_DETAILS)
    if change == "nested-value":
        altered["path"][1][1]["seen"][1] = "different"
    elif change == "nested-order":
        altered["checks"][1].reverse()
    else:
        altered["checks"][0]["window"][0] = True
    factory = make_data_problem if code == "HOLDOUT_ACCESS_DENIED" else make_problem
    with monkeypatch.context() as patch:
        patch.setattr(outcomes, "ledger_append", _forbid)
        with pytest.raises(OpsError) as captured:
            _publish(source, registration, destination, no_ledger=no_ledger, report=None,
                     problem=factory(code, "synthetic changed refusal", details=altered))
    _assert_conflict(source, captured, registration)
    assert _catalog(source.conn) == before
    assert _csv_state(ledger) == ledger_bytes
    _assert_objects_preserved(source, objects)
    assert len(_objects(source.store).keys() - objects.keys()) == 1  # Private R6 only.
    assert _replay(source, registration, destination, no_ledger=no_ledger) == first
    assert _catalog(source.conn) == before
    assert _csv_state(ledger) == ledger_bytes
    assert not list(tmp_path.rglob("REPORT.md"))


@pytest.mark.parametrize("code", ["FEATURE_LOOKAHEAD", "HOLDOUT_ACCESS_DENIED",
                                  "EXPERIMENT_VARIANT_FAILED"])
@pytest.mark.parametrize("no_ledger", [False, True], ids=["recorded", "smoke"])
@pytest.mark.parametrize("operation", ["publish", "replay"])
def test_canonical_refusal_pending_tuple_intent_recovers_without_rewriting(
        source, tmp_path, monkeypatch, code, no_ledger, operation):
    """A committed pre-completion tuple intent remains recoverable across dates."""
    from tests.v2.research.test_native_outcomes import _clock

    registration = _call(source)
    ledger = tmp_path / "canonical-pending" / "ledger.csv"
    ledger_before = _seed_csv(ledger, True)
    destination = PoisonLedger() if no_ledger else ledger
    problem = _nested_refusal_problem(code, True)
    _clock(monkeypatch, "2026-01-02")
    key, saved = _pending(source, registration, destination, monkeypatch,
                          no_ledger=no_ledger, report=None, problem=problem, attempted=False)
    assert saved["failure_details"] == NESTED_REFUSAL_DETAILS
    assert _csv_state(ledger) == ledger_before
    intent_ref = _evidence(source, registration.run_id)[key]
    objects = _objects(source.store)
    _clock(monkeypatch, "2026-02-10")
    if operation == "publish":
        result = _publish(source, registration, destination, no_ledger=no_ledger,
                          report=None, problem=problem, attempted=False)
    else:
        result = _replay(source, registration, destination, no_ledger=no_ledger)
    assert result["outcome"] == saved
    assert result["outcome"]["failure_details"] == NESTED_REFUSAL_DETAILS
    assert result["outcome"]["ledger_row"]["date"] == "2026-01-02"
    assert result["outcome"]["attempted_variants"] == 0
    assert result["outcome"]["report_ref"] is result["receipt"]["report_ref"] is None
    evidence = _evidence(source, registration.run_id)
    assert evidence[key] == intent_ref
    assert evidence[key + "_receipt"] == result["receipt_ref"]
    assert result["receipt"]["outcome_ref"] == intent_ref
    _assert_objects_preserved(source, objects)
    assert len(_objects(source.store).keys() - objects.keys()) == 1  # Completion only.
    if no_ledger:
        assert _csv_state(ledger) == ledger_before
    else:
        assert _rows(ledger) == [UNRELATED_ROW, saved["ledger_row"]]
    before, objects, ledger_bytes = _catalog(source.conn), _objects(source.store), _csv_state(ledger)
    assert _publish(source, registration, destination, no_ledger=no_ledger,
                    report=None, problem=problem, attempted=False) == result
    assert _replay(source, registration, destination, no_ledger=no_ledger) == result
    assert _catalog(source.conn) == before
    assert _objects(source.store) == objects
    assert _csv_state(ledger) == ledger_bytes
    assert not list(tmp_path.rglob("REPORT.md"))


@pytest.mark.parametrize("no_ledger", [False, True], ids=["recorded", "smoke"])
def test_canonical_refusal_republish_still_validates_original_pending_date(
        source, tmp_path, monkeypatch, no_ledger):
    """Date masking for identity cannot bless an invalid saved publication date."""
    registration = _call(source)
    ledger = tmp_path / "canonical-invalid-date" / "ledger.csv"
    ledger_bytes = _seed_csv(ledger, True)
    destination = PoisonLedger() if no_ledger else ledger
    problem = _nested_refusal_problem("FEATURE_LOOKAHEAD", True)
    key, saved = _pending(source, registration, destination, monkeypatch,
                          no_ledger=no_ledger, report=None, problem=problem, attempted=False)
    saved["ledger_row"]["date"] = "2026-02-30"
    _save_intent(source, registration, key, saved)
    before, objects = _catalog(source.conn), _objects(source.store)
    monkeypatch.setattr(outcomes, "ledger_append", _forbid)
    with pytest.raises(OpsError) as captured:
        _publish(source, registration, destination, no_ledger=no_ledger,
                 report=None, problem=problem, attempted=False)
    _assert_conflict(source, captured, registration)
    assert _catalog(source.conn) == before
    assert _csv_state(ledger) == ledger_bytes
    assert key + "_receipt" not in _evidence(source, registration.run_id)
    _assert_objects_preserved(source, objects)
    assert len(_objects(source.store).keys() - objects.keys()) == 1
    assert not list(tmp_path.rglob("REPORT.md"))
