"""Native parity helper wiring for the supervisor sidecar.

These tests cover only ``engine.v2.ops.nightly``'s new identity, keying,
schema-version, and submit helpers. Real catalog fixtures are used where job
IDs and succeeded parent rows need to be meaningful; direct rows are used when
a branch needs exact timestamp/hash control.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from engine.v2.foundation import ArtifactStore, format_timestamp
from engine.v2.foundation.artifacts import ArtifactError
from engine.v2.ops import nightly
from engine.v2.ops.checkpoints import artifact, register_artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.native_parity_report import _REFUSALS_SCHEMA_VERSION, _RECORDS_SCHEMA_VERSION
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, job_id_for, submit_graph
from tests.ops_support import catalog

ROOT = Path(__file__).resolve().parents[1]
_POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})
_OMIT = object()
_RECORDS_OK = json.dumps({
    "schema_version": _RECORDS_SCHEMA_VERSION,
    "records": {},
}).encode()
_REFUSALS_OK = json.dumps({
    "schema_version": _REFUSALS_SCHEMA_VERSION,
    "refusals": {},
    "unkeyable_refusals": [],
}).encode()


def _stamp(clock):
    return format_timestamp(clock.now())


def _insert_job(conn, clock, *, idempotency_key, kind, resource_class,
                checkpoint_contract_ref, state="succeeded", stamp=None):
    job_id = job_id_for("shadow", idempotency_key)
    ts = stamp or _stamp(clock)
    conn.execute(
        "INSERT INTO jobs (job_id, namespace, idempotency_key, request_digest, principal, "
        "kind, spec_json, resource_class, checkpoint_contract_ref, retry_json, state, "
        "priority, max_attempts, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?)",
        (job_id, "shadow", idempotency_key, "digest", "operator", kind, "{}", resource_class,
         checkpoint_contract_ref, "{}", state, 0, 1, ts, ts))
    conn.commit()
    return job_id


def _seed_score_job(conn, clock, *, as_of, scope_hash, state="succeeded", stamp=None):
    key = "nightly:" + as_of + ":" + scope_hash + ":score"
    return _insert_job(conn, clock, idempotency_key=key, kind=nightly._LEGACY_SCORE_KIND,
                       resource_class=nightly._LEGACY_SCORE_KIND,
                       checkpoint_contract_ref="legacy_action.v1.0", state=state, stamp=stamp)


def _seed_batch_job(conn, clock, *, as_of, scope_hash, state="succeeded", stamp=None):
    key = nightly._native_score_batch_key(as_of, scope_hash)
    return _insert_job(conn, clock, idempotency_key=key, kind="native_score_batch",
                       resource_class="io_fetch",
                       checkpoint_contract_ref="native_score_batch_records.v2.0",
                       state=state, stamp=stamp)


def _seed_attempt(conn, supervisor, clock, job_id, *, attempt_number=1, state="succeeded"):
    attempt_id = "attempt-" + job_id + "-" + str(attempt_number)
    ts = _stamp(clock)
    lease = format_timestamp(clock.now() + timedelta(seconds=1))
    conn.execute(
        "INSERT INTO attempts (attempt_id, job_id, attempt_number, fence, supervisor_epoch, "
        "host_boot_id, state, process_state, resources_json, created_at, lease_expires_at, "
        "ended_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'verified_dead', ?, ?, ?, ?)",
        (attempt_id, job_id, attempt_number, attempt_number, supervisor.epoch_id,
         supervisor.boot_id, state, "{}", ts, lease, ts))
    conn.commit()
    return attempt_id


def _publish_output(conn, store, clock, attempt_id, name, payload, schema_ref):
    ref = store.publish_bytes(payload, schema_ref=schema_ref)
    register_artifact(conn, ref, attempt_id, clock)
    conn.execute("INSERT INTO attempt_outputs VALUES (?, ?, ?)",
                 (attempt_id, name, ref.artifact_id))
    conn.commit()


def _seed_batch_documents(conn, store, supervisor, clock, job_id, *, records_document=_RECORDS_OK,
                          refusals_document=_REFUSALS_OK):
    attempt_id = _seed_attempt(conn, supervisor, clock, job_id)
    if records_document is not _OMIT:
        _publish_output(conn, store, clock, attempt_id, "records", records_document,
                        _RECORDS_SCHEMA_VERSION)
    if refusals_document is not _OMIT:
        _publish_output(conn, store, clock, attempt_id, "refusals", refusals_document,
                        _REFUSALS_SCHEMA_VERSION)
    return attempt_id


def _mark_score_succeeded(conn, clock, *, session, tickers=("FAKE",),
                          year_start=2025, year_end=2026):
    plan = nightly.build_nightly_plan(ROOT, session)
    requests = nightly.build_legacy_job_requests(
        plan, tickers=tickers, year_start=year_start, year_end=year_end)
    submit_graph(conn, registry(), _POLICY, requests, clock=clock)
    score = next(request for request in requests if request.idempotency_key.endswith(":score"))
    job_id = job_id_for("shadow", score.idempotency_key)
    conn.execute("UPDATE jobs SET state = 'succeeded', updated_at = ? WHERE job_id = ?",
                 (clock.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"), job_id))
    conn.commit()
    return score.idempotency_key


def _submit(conn, tmp_path, clock, store=None):
    return nightly.submit_native_parity_if_ready(
        conn, registry(), _POLICY, store or ArtifactStore(tmp_path),
        catalog_path=str(tmp_path / "ops.sqlite"), objects_root=str(tmp_path),
        code_source=ROOT, clock=clock)


def _native_parity_job_count(conn):
    return conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE kind = ?", ("native_parity",)
    ).fetchone()["n"]


def _native_parity_row(conn):
    return conn.execute("SELECT * FROM jobs WHERE kind = ?", ("native_parity",)).fetchone()


def test_scope_from_native_score_batch_key_parses_and_rejects():
    parse = nightly._scope_from_native_score_batch_key
    assert parse("nightly:S1:HASH1:native_score_batch") == ("S1", "HASH1")
    assert parse("nightly:S1:sha256:814516c45d0dea73b:native_score_batch") == (
        "S1", "sha256:814516c45d0dea73b")
    assert parse("shadow:S1:HASH1:native_score_batch") is None
    assert parse("nightly:S1:HASH1:score") is None
    assert parse("nightly:HASH1:native_score_batch") is None
    assert parse("nightly::HASH1:native_score_batch") is None
    assert parse("nightly:S1::native_score_batch") is None


def test_native_parity_key_shape():
    assert nightly._native_parity_key("S1", "sha256:abcd") == (
        "nightly:S1:sha256:abcd:native_parity")


def test_native_parity_identity_requires_succeeded_batch_job(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    assert nightly._native_parity_identity(conn) is None
    _seed_score_job(conn, clock, as_of="S1", scope_hash="H1")
    assert nightly._native_parity_identity(conn) is None


def test_native_parity_identity_skips_malformed_batch_key(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    _insert_job(conn, clock, idempotency_key="nightly:S1:H1:other", kind="native_score_batch",
                resource_class="io_fetch", checkpoint_contract_ref="native_score_batch_records.v2.0")
    assert nightly._native_parity_identity(conn) is None


def test_native_parity_identity_skips_blank_created_at(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    batch_job_id = _seed_batch_job(conn, clock, as_of="S1", scope_hash="H1")
    _seed_score_job(conn, clock, as_of="S1", scope_hash="H1")
    conn.execute("UPDATE jobs SET created_at = '' WHERE job_id = ?", (batch_job_id,))
    conn.commit()
    assert nightly._native_parity_identity(conn) is None


def test_native_parity_identity_picks_latest_as_of(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    first_stamp = _stamp(clock)
    clock.advance(60)
    second_stamp = _stamp(clock)
    _seed_batch_job(conn, clock, as_of="2026-01-01", scope_hash="A", stamp=first_stamp)
    _seed_score_job(conn, clock, as_of="2026-01-01", scope_hash="A", stamp=first_stamp)
    batch_job_id = _seed_batch_job(conn, clock, as_of="2026-01-02", scope_hash="B",
                                   stamp=second_stamp)
    score_job_id = _seed_score_job(conn, clock, as_of="2026-01-02", scope_hash="B",
                                   stamp=second_stamp)

    assert nightly._native_parity_identity(conn) == (
        "2026-01-02", "B", score_job_id, batch_job_id)


def test_native_parity_identity_picks_latest_created_at_inside_same_as_of(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    first_stamp = _stamp(clock)
    clock.advance(60)
    second_stamp = _stamp(clock)
    _seed_batch_job(conn, clock, as_of="2026-01-01", scope_hash="B", stamp=first_stamp)
    _seed_score_job(conn, clock, as_of="2026-01-01", scope_hash="B", stamp=first_stamp)
    batch_job_id = _seed_batch_job(conn, clock, as_of="2026-01-01", scope_hash="A",
                                   stamp=second_stamp)
    score_job_id = _seed_score_job(conn, clock, as_of="2026-01-01", scope_hash="A",
                                   stamp=second_stamp)

    assert nightly._native_parity_identity(conn) == (
        "2026-01-01", "A", score_job_id, batch_job_id)


def test_native_parity_identity_ties_break_on_scope_hash(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    stamp = _stamp(clock)
    _seed_batch_job(conn, clock, as_of="S1", scope_hash="sha256:aaa", stamp=stamp)
    _seed_score_job(conn, clock, as_of="S1", scope_hash="sha256:aaa", stamp=stamp)
    batch_job_id = _seed_batch_job(conn, clock, as_of="S1", scope_hash="sha256:bbb", stamp=stamp)
    score_job_id = _seed_score_job(conn, clock, as_of="S1", scope_hash="sha256:bbb", stamp=stamp)

    assert nightly._native_parity_identity(conn) == (
        "S1", "sha256:bbb", score_job_id, batch_job_id)


def test_native_parity_identity_requires_paired_succeeded_score_job(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    _seed_batch_job(conn, clock, as_of="S1", scope_hash="H1")
    assert nightly._native_parity_identity(conn) is None

    score_job_id = _seed_score_job(conn, clock, as_of="S1", scope_hash="H1", state="queued")
    assert nightly._native_parity_identity(conn) is None

    conn.execute("UPDATE jobs SET state = 'succeeded' WHERE job_id = ?", (score_job_id,))
    conn.commit()

    batch_job_id = job_id_for("shadow", nightly._native_score_batch_key("S1", "H1"))
    assert nightly._native_parity_identity(conn) == (
        "S1", "H1", score_job_id, batch_job_id)


def test_native_parity_identity_falls_back_to_an_older_fully_paired_batch_when_the_newest_is_unpaired(
        tmp_path):
    conn, clock, _ = catalog(tmp_path)
    first_stamp = _stamp(clock)
    older_batch_job_id = _seed_batch_job(conn, clock, as_of="S1", scope_hash="H1",
                                         stamp=first_stamp)
    older_score_job_id = _seed_score_job(conn, clock, as_of="S1", scope_hash="H1",
                                         stamp=first_stamp)
    clock.advance(60)
    second_stamp = _stamp(clock)
    _seed_batch_job(conn, clock, as_of="S2", scope_hash="H2", stamp=second_stamp)

    assert nightly._native_parity_identity(conn) == (
        "S1", "H1", older_score_job_id, older_batch_job_id)


def test_native_parity_identity_prefers_the_shadow_namespace_over_a_newer_smoke_one(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    first_stamp = _stamp(clock)
    batch_job_id = _seed_batch_job(conn, clock, as_of="S1", scope_hash="H1", stamp=first_stamp)
    score_job_id = _seed_score_job(conn, clock, as_of="S1", scope_hash="H1", stamp=first_stamp)
    clock.advance(60)
    smoke_stamp = _stamp(clock)
    smoke_key = nightly._native_score_batch_key("S1", "H1")
    smoke_job_id = job_id_for("smoke", smoke_key)
    conn.execute(
        "INSERT INTO jobs (job_id, namespace, idempotency_key, request_digest, principal, "
        "kind, spec_json, resource_class, checkpoint_contract_ref, retry_json, state, "
        "priority, max_attempts, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?)",
        (smoke_job_id, "smoke", smoke_key, "digest", "operator", "native_score_batch", "{}",
         "io_fetch", "native_score_batch_records.v2.0", "{}", "succeeded", 0, 1,
         smoke_stamp, smoke_stamp))
    conn.commit()

    identity = nightly._native_parity_identity(conn)

    assert identity == ("S1", "H1", score_job_id, batch_job_id)
    assert identity[3] != smoke_job_id


def test_native_score_batch_document_schema_ok_accepts_current_documents(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    batch_job_id = _seed_batch_job(conn, clock, as_of="S1", scope_hash="H1")
    _seed_batch_documents(conn, store, supervisor, clock, batch_job_id)

    assert nightly._native_score_batch_document_schema_ok(conn, store, batch_job_id) is True


@pytest.mark.parametrize("records_document,refusals_document", [
    (b'{"schema_version": "old.records", "records": {}}', _REFUSALS_OK),
    (_RECORDS_OK, b'{"schema_version": "old.refusals", "refusals": {}, '
                  b'"unkeyable_refusals": []}'),
])
def test_native_score_batch_document_schema_ok_reports_confirmed_mismatch(tmp_path,
                                                                          records_document,
                                                                          refusals_document):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    batch_job_id = _seed_batch_job(conn, clock, as_of="S1", scope_hash="H1")
    _seed_batch_documents(conn, store, supervisor, clock, batch_job_id,
                          records_document=records_document,
                          refusals_document=refusals_document)

    assert nightly._native_score_batch_document_schema_ok(
        conn, store, batch_job_id) is False


def test_native_score_batch_document_schema_ok_raises_without_succeeded_attempt(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    batch_job_id = _seed_batch_job(conn, clock, as_of="S1", scope_hash="H1")

    with pytest.raises(OpsError) as exc:
        nightly._native_score_batch_document_schema_ok(conn, store, batch_job_id)

    assert exc.value.code == "VALIDATION_FAILED"
    assert exc.value.problem.details["native_score_batch_job_id"] == batch_job_id


def test_native_score_batch_document_schema_ok_raises_when_output_is_missing(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    batch_job_id = _seed_batch_job(conn, clock, as_of="S1", scope_hash="H1")
    _seed_batch_documents(conn, store, supervisor, clock, batch_job_id,
                          refusals_document=_OMIT)

    with pytest.raises(OpsError) as exc:
        nightly._native_score_batch_document_schema_ok(conn, store, batch_job_id)

    assert exc.value.code == "VALIDATION_FAILED"
    assert "refusals" in exc.value.problem.message


def test_native_score_batch_document_schema_ok_raises_on_invalid_json(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    batch_job_id = _seed_batch_job(conn, clock, as_of="S1", scope_hash="H1")
    _seed_batch_documents(conn, store, supervisor, clock, batch_job_id,
                          records_document=b"not json")

    with pytest.raises(OpsError) as exc:
        nightly._native_score_batch_document_schema_ok(conn, store, batch_job_id)

    assert exc.value.code == "VALIDATION_FAILED"
    assert exc.value.problem.retryable is False
    assert "records.json is not valid JSON" in exc.value.problem.message
    assert exc.value.problem.details["native_score_batch_job_id"] == batch_job_id


@pytest.mark.parametrize("name", ["records", "refusals"])
@pytest.mark.parametrize("corruption", ["hash", "size"])
def test_native_score_batch_corrupt_committed_bytes_refuse_submission(tmp_path, name, corruption):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    batch_job_id = _seed_batch_job(conn, clock, as_of="S1", scope_hash="H1")
    _seed_score_job(conn, clock, as_of="S1", scope_hash="H1")
    attempt_id = _seed_batch_documents(conn, store, supervisor, clock, batch_job_id)
    row = conn.execute("SELECT artifact_id FROM attempt_outputs WHERE attempt_id=? AND name=?",
                       (attempt_id, name)).fetchone()
    ref = artifact(conn, store, row[0])
    path = store.root / ref.storage_key
    payload = path.read_bytes()
    # Corrupt bytes produced by the real ArtifactStore, retaining its original reference.
    path.chmod(0o600)
    path.write_bytes(b"!" + payload[1:] if corruption == "hash" else payload + b" ")

    with pytest.raises(ArtifactError) as raw:
        store.read_verified(ref)
    assert raw.value.code == "INTEGRITY_FAILED"
    for check in (
        lambda: nightly._native_score_batch_document_schema_ok(conn, store, batch_job_id),
        lambda: _submit(conn, tmp_path, clock, store),
    ):
        with pytest.raises(OpsError) as exc:
            check()
        assert exc.value.code == "VALIDATION_FAILED"
        assert exc.value.problem.retryable is False
        assert exc.value.problem.message == (
            "native_score_batch " + name + ".json failed artifact verification")
        assert exc.value.problem.details == {"native_score_batch_job_id": batch_job_id}
        assert exc.value.__suppress_context__ is True
    assert _native_parity_job_count(conn) == 0


@pytest.mark.parametrize("name", ["records", "refusals"])
def test_native_score_batch_missing_object_preserves_artifact_error(tmp_path, name):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    batch_job_id = _seed_batch_job(conn, clock, as_of="S1", scope_hash="H1")
    attempt_id = _seed_batch_documents(conn, store, supervisor, clock, batch_job_id)
    row = conn.execute("SELECT artifact_id FROM attempt_outputs WHERE attempt_id=? AND name=?",
                       (attempt_id, name)).fetchone()
    ref = artifact(conn, store, row[0])
    (store.root / ref.storage_key).unlink()

    with pytest.raises(ArtifactError) as exc:
        nightly._native_score_batch_document_schema_ok(conn, store, batch_job_id)
    assert exc.value.code == "MISSING"


@pytest.mark.parametrize("error", [ArtifactError("UNSAFE_PATH", "private path"),
                                  OSError("private read failure")])
@pytest.mark.parametrize("method", ["verify", "read_verified"])
def test_native_score_batch_preserves_other_storage_errors(tmp_path, monkeypatch, error, method):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    batch_job_id = _seed_batch_job(conn, clock, as_of="S1", scope_hash="H1")
    _seed_batch_documents(conn, store, supervisor, clock, batch_job_id)

    def fail_read(ref):
        raise error

    monkeypatch.setattr(store, method, fail_read)
    with pytest.raises(type(error)) as exc:
        nightly._native_score_batch_document_schema_ok(conn, store, batch_job_id)
    assert exc.value is error


def test_native_score_batch_document_schema_ok_raises_on_non_mapping_document(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    batch_job_id = _seed_batch_job(conn, clock, as_of="S1", scope_hash="H1")
    _seed_batch_documents(conn, store, supervisor, clock, batch_job_id,
                          records_document=b"[]")

    with pytest.raises(OpsError) as exc:
        nightly._native_score_batch_document_schema_ok(conn, store, batch_job_id)

    assert exc.value.code == "VALIDATION_FAILED"
    assert "records.json" in exc.value.problem.message


def test_native_score_batch_document_schema_ok_raises_on_missing_schema_version_key(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    batch_job_id = _seed_batch_job(conn, clock, as_of="S1", scope_hash="H1")
    _seed_batch_documents(conn, store, supervisor, clock, batch_job_id,
                          records_document=b'{"other_field": 1}')

    with pytest.raises(OpsError) as exc:
        nightly._native_score_batch_document_schema_ok(conn, store, batch_job_id)

    assert exc.value.code == "VALIDATION_FAILED"
    assert exc.value.problem.retryable is False
    assert "records.json is missing its schema_version tag" in exc.value.problem.message
    assert exc.value.problem.details["native_score_batch_job_id"] == batch_job_id


def test_submit_native_parity_if_ready_returns_none_without_catalog(tmp_path):
    _, clock, _ = catalog(tmp_path)

    assert nightly.submit_native_parity_if_ready(
        None, registry(), _POLICY, ArtifactStore(tmp_path),
        catalog_path=str(tmp_path / "ops.sqlite"), objects_root=str(tmp_path),
        code_source=ROOT, clock=clock) is None


def test_submit_native_parity_if_ready_returns_none_without_identity(tmp_path):
    conn, clock, _ = catalog(tmp_path)

    assert _submit(conn, tmp_path, clock) is None
    assert _native_parity_job_count(conn) == 0


def test_submit_native_parity_if_ready_dedupes_existing_job(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    batch_job_id = _seed_batch_job(conn, clock, as_of="S1", scope_hash="H1")
    _seed_score_job(conn, clock, as_of="S1", scope_hash="H1")
    _seed_batch_documents(conn, store, supervisor, clock, batch_job_id)

    receipt = _submit(conn, tmp_path, clock, store=store)
    assert receipt is not None
    before = _native_parity_job_count(conn)

    assert _submit(conn, tmp_path, clock, store=store) is None
    assert _native_parity_job_count(conn) == before


def test_submit_native_parity_if_ready_refuses_stale_schema(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    batch_job_id = _seed_batch_job(conn, clock, as_of="S1", scope_hash="H1")
    _seed_score_job(conn, clock, as_of="S1", scope_hash="H1")
    _seed_batch_documents(conn, store, supervisor, clock, batch_job_id,
                          records_document=b'{"schema_version": "old.records", "records": {}}')

    with pytest.raises(OpsError) as exc:
        _submit(conn, tmp_path, clock, store=store)

    assert exc.value.code == "VALIDATION_FAILED"
    assert exc.value.problem.details == {
        "reason": "schema_mismatch",
        "native_score_batch_job_id": batch_job_id,
    }
    assert _native_parity_job_count(conn) == 0


def test_submit_native_parity_if_ready_submits_real_paired_score_job(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    score_key = _mark_score_succeeded(conn, clock, session="2026-01-01")
    session, scope_hash = nightly._session_scope_from_score_key(score_key)
    batch_job_id = _seed_batch_job(conn, clock, as_of=session, scope_hash=scope_hash)
    _seed_batch_documents(conn, store, supervisor, clock, batch_job_id)

    receipt = _submit(conn, tmp_path, clock, store=store)

    assert receipt.kind == "native_parity"
    assert receipt.state == "queued"
    assert receipt.idempotency_key == nightly._native_parity_key(session, scope_hash)

    row = _native_parity_row(conn)
    spec = json.loads(row["spec_json"])
    score_job_id = job_id_for("shadow", score_key)
    assert spec["parameters"]["input_bindings"] == {
        "score.json": score_job_id + "#legacy_score",
        "records.json": batch_job_id + "#records",
        "refusals.json": batch_job_id + "#refusals",
    }
    assert spec["dependency_job_ids"] == [score_job_id, batch_job_id]
    assert row["kind"] == "native_parity"
    assert row["checkpoint_contract_ref"] == "native_parity_report.v1.1"
    assert row["resource_class"] == "validation"
