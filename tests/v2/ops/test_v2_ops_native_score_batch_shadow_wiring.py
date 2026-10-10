"""Cutover PR-7a: native_score_batch shadow submission wiring.

``native_score_batch`` is a real GRAPH node and an OPTIONAL stage, but no
submission path in ``nightly.py`` ever builds a kind for it -- exactly like
``computed_moves_refresh``, the only submitter is ``supervisor.Service``'s own
tick sidecar (``Service._reconcile_native_score_batch_shadow``), gated first on
a verified production model release root and then on
``nightly.submit_native_score_batch_shadow_if_ready``. These tests prove only
nightly.py's/supervisor.py's own new code: identity/key helpers, the
release-root gate, and the dedup branch of the still-stubbed submit path (the
raw-row producer is cutover PR-6 and is explicitly not built here).

Fixtures use REAL payload shapes: the legacy "score" job rows below are built
by ``build_legacy_job_requests`` from a real ``build_nightly_plan`` plan,
submitted through the real ``submit_graph``, and only then UPDATEd to
``succeeded`` -- never a hand-written ``spec_json``.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from engine.paths import ROOT
from engine.v2.contracts import JobReceipt
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, canonical_json, content_hash
from engine.v2.models import deployment
from engine.v2.ops import nightly
from engine.v2.ops.checkpoints import registered_artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import environment_identity
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.stages import LegacyParameters, registry
from engine.v2.ops.submission import (
    NamespacePolicy,
    job_id_for,
    request_digest,
    submit_graph,
)
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY, catalog

_POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


def _mark_score_succeeded(conn, clock, *, session, tickers=("FAKE",),
                          year_start=2025, year_end=2026):
    """A minimal, REAL legacy "score" job row -- built with the same
    ``build_nightly_plan``/``build_legacy_job_requests`` pair production uses
    (``input_mode="legacy"`` default, so its parameters carry no
    ``snapshot_generation_id``), submitted through the real ``submit_graph``,
    then its ``state`` set directly to ``succeeded`` -- the only way to put a
    row at a specific logical time, mirroring
    ``test_v2_ops_computed_moves_nightly_wiring.py::_mark_refresh_succeeded``.
    Returns the score job's own idempotency key.
    """
    plan = nightly.build_nightly_plan(ROOT, session)
    requests = nightly.build_legacy_job_requests(
        plan, tickers=tickers, year_start=year_start, year_end=year_end)
    submit_graph(conn, registry(), _POLICY, requests, clock=clock)
    score = next(request for request in requests
                 if request.idempotency_key.endswith(":score"))
    job_id = job_id_for("shadow", score.idempotency_key)
    conn.execute("UPDATE jobs SET state = 'succeeded', updated_at = ? WHERE job_id = ?",
                 (clock.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"), job_id))
    conn.commit()
    return score.idempotency_key


def _native_score_batch_job_count(conn):
    return conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE kind = ?", ("native_score_batch",)
    ).fetchone()["n"]


def _assert_sidecar_runtime_identity(resource_class, environment_ref,
                                     expected_resource_class):
    assert resource_class == expected_resource_class
    profile = profile_named(DEFAULT_POLICY, expected_resource_class)
    thread_count = profile.thread_count or profile.cpu_count
    assert environment_ref == content_hash(environment_identity(thread_count))


# --------------------------------------------------------------------------
# key helpers / graph wiring
# --------------------------------------------------------------------------


def test_session_scope_from_score_key_parses_and_rejects():
    assert nightly._session_scope_from_score_key("nightly:S1:HASH1:score") == ("S1", "HASH1")
    assert nightly._session_scope_from_score_key("nightly:S1:score") is None
    assert nightly._session_scope_from_score_key("nightly:S1:HASH1:refresh") is None
    assert nightly._session_scope_from_score_key("nightly::HASH1:score") is None
    assert nightly._session_scope_from_score_key("shadow:S1:HASH1:score") is None
    assert nightly._session_scope_from_score_key(
        "nightly:2026-01-01:sha256:814516c45d0dea73b:score") == (
            "2026-01-01", "sha256:814516c45d0dea73b")


def test_native_score_batch_key_shape():
    assert nightly._native_score_batch_key("S1", "HASH1") == "nightly:S1:HASH1:native_score_batch"


def test_graph_and_optional_include_native_score_batch():
    from engine.v2.ops.submission import RetryPolicy

    assert nightly.GRAPH["native_score_batch"] == ("score",)
    assert "native_score_batch" in nightly.OPTIONAL
    assert registry().get("native_score_batch").retry == RetryPolicy(
        "bounded", 5, (30, 120, 600, 1800))


def test_stage_sequence_excludes_native_score_batch():
    plan = nightly.build_nightly_plan(ROOT, "2026-01-01")
    stages = nightly._stage_sequence(plan, include_prerequisites=True, snapshot=None,
                                     refresh_mode="legacy")
    assert "native_score_batch" not in stages
    assert "score" in stages


# --------------------------------------------------------------------------
# _native_score_batch_identity / submit_native_score_batch_shadow_if_ready
# --------------------------------------------------------------------------


def test_identity_returns_none_with_no_succeeded_score_job(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    assert nightly._native_score_batch_identity(conn) is None
    assert nightly.submit_native_score_batch_shadow_if_ready(
        conn, registry(), _POLICY, store, "irrelevant-root",
        catalog_path=str(tmp_path / "ops.sqlite"), objects_root=str(tmp_path),
        code_source=ROOT, clock=clock) is None


def test_identity_finds_latest_session_and_reports_no_snapshot_pinned(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    score_key = _mark_score_succeeded(conn, clock, session="2026-01-01")
    session, scope_hash = nightly._session_scope_from_score_key(score_key)

    assert nightly._native_score_batch_identity(conn) == (session, scope_hash, None)

    before = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
    assert nightly.submit_native_score_batch_shadow_if_ready(
        conn, registry(), _POLICY, store, "irrelevant-root",
        catalog_path=str(tmp_path / "ops.sqlite"), objects_root=str(tmp_path),
        code_source=ROOT, clock=clock) is None
    assert conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == before


def test_identity_picks_latest_of_two_sessions(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    _mark_score_succeeded(conn, clock, session="2026-01-01")
    later_key = _mark_score_succeeded(conn, clock, session="2026-01-02")

    identity = nightly._native_score_batch_identity(conn)

    assert identity is not None
    assert identity[:2] == nightly._session_scope_from_score_key(later_key)


def test_identity_orders_by_created_at_not_scope_hash(tmp_path):
    """CodeRabbit/Opus gate finding: the old code picked the "latest" score
    job in a session by ``max(scope_hash)`` -- a content hash with no time
    meaning -- so whichever of two same-session jobs (e.g. a legacy-mode
    run followed by a snapshot-mode rerun) happened to hash higher won, not
    whichever ran later. This builds two real succeeded "score" jobs in the
    SAME session with different ``scope_hash`` (different ticker sets), each
    from its real ``build_legacy_job_requests`` request graph (never a
    hand-written hash), and submits the HIGHER-hash job first and the
    LOWER-hash job second -- so hash order and creation order disagree by
    construction -- then asserts the later-created job wins.

    Two candidates are enough: ordering them by their verified hashes makes
    the disagreement certain, so no search over many candidates is needed
    (each request-graph build recomputes the full worker source manifest, so
    a 200-candidate search cost minutes). Each graph is built once and the
    same requests are both hashed and submitted. Both are built from the
    SAME ``plan`` object: ``build_nightly_plan`` stamps ``decision_clock``
    from a real ``SystemClock`` when no ``clock=`` is passed, so hashes
    from separate plans fold in different instants and are incomparable.
    """
    conn, clock, _ = catalog(tmp_path)
    session = "2026-01-01"
    plan = nightly.build_nightly_plan(ROOT, session)

    candidates = []
    for tickers in (("FIRST",), ("LATER",)):
        requests = nightly.build_legacy_job_requests(plan, tickers=tickers,
                                                       year_start=2025, year_end=2026)
        score = next(r for r in requests if r.idempotency_key.endswith(":score"))
        _, scope_hash = nightly._session_scope_from_score_key(score.idempotency_key)
        candidates.append((scope_hash, requests, score.idempotency_key))

    (first_hash, first_requests, first_key), (later_hash, later_requests, later_key) = sorted(
        candidates, key=lambda item: item[0], reverse=True)
    assert later_hash < first_hash, "candidate hashes must be distinct"

    def _submit_succeeded(requests, score_key):
        submit_graph(conn, registry(), _POLICY, requests, clock=clock)
        conn.execute("UPDATE jobs SET state = 'succeeded', updated_at = ? WHERE job_id = ?",
                     (clock.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                      job_id_for("shadow", score_key)))
        conn.commit()

    _submit_succeeded(first_requests, first_key)
    clock.advance(60)
    _submit_succeeded(later_requests, later_key)

    identity = nightly._native_score_batch_identity(conn)

    assert identity[:2] == (session, later_hash)  # time order (the fix) picks `later`
    assert identity[1] != first_hash  # the old max(scope_hash) selector would pick `first`


def test_identity_skips_row_with_missing_created_at(tmp_path):
    """A row whose ``created_at`` is blank (defensive: the column is
    ``NOT NULL`` in the schema, so this should never happen, but the
    function must never crash on it or let it silently win a time
    comparison) is excluded entirely -- the function falls back to the
    next valid candidate, never raises."""
    conn, clock, _ = catalog(tmp_path)
    valid_key = _mark_score_succeeded(conn, clock, session="2026-01-01")
    clock.advance(3600)
    blank_key = _mark_score_succeeded(conn, clock, session="2026-01-02")
    conn.execute("UPDATE jobs SET created_at = '' WHERE job_id = ?",
                 (job_id_for("shadow", blank_key),))
    conn.commit()

    identity = nightly._native_score_batch_identity(conn)

    assert identity is not None
    assert identity[:2] == nightly._session_scope_from_score_key(valid_key)


def test_identity_tie_breaks_deterministically_when_created_at_matches(tmp_path):
    """Two succeeded "score" jobs in the same session landing on the exact
    same ``created_at`` (no clock advance between them) must still produce
    one consistent, deterministic answer -- never a crash, and never a
    different answer across repeated calls."""
    conn, clock, _ = catalog(tmp_path)
    session = "2026-01-01"
    key_a = _mark_score_succeeded(conn, clock, session=session, tickers=("AAAA",))
    key_b = _mark_score_succeeded(conn, clock, session=session, tickers=("BBBB",))

    first_call = nightly._native_score_batch_identity(conn)
    second_call = nightly._native_score_batch_identity(conn)

    assert first_call is not None
    assert first_call == second_call
    assert first_call[:2] in (nightly._session_scope_from_score_key(key_a),
                              nightly._session_scope_from_score_key(key_b))


def test_submit_dedupes_an_existing_job_under_the_key(tmp_path, monkeypatch):
    """The dedup branch in isolation from the currently-unbuilt PR-6 producer:
    ``_native_score_batch_identity`` is monkeypatched for THIS ONE test to
    pretend the succeeded "score" job pinned a snapshot, so the exists-check
    is reached even though today's production default never pins one."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    score_key = _mark_score_succeeded(conn, clock, session="2026-01-01")
    session, scope_hash = nightly._session_scope_from_score_key(score_key)
    key = nightly._native_score_batch_key(session, scope_hash)
    batch_job_id = job_id_for("shadow", key)
    stamp = clock.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    conn.execute(
        "INSERT INTO jobs (job_id, namespace, idempotency_key, request_digest, principal, kind, "
        "spec_json, resource_class, checkpoint_contract_ref, retry_json, state, priority, "
        "max_attempts, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (batch_job_id, "shadow", key, "digest", "operator", "native_score_batch",
         "{}", "legacy_score", "legacy_action.v1.0", "{}", "queued", 0, 1, stamp, stamp))
    conn.commit()
    monkeypatch.setattr(nightly, "_native_score_batch_identity",
                        lambda conn: (session, scope_hash, True))

    before = _native_score_batch_job_count(conn)
    assert nightly.submit_native_score_batch_shadow_if_ready(
        conn, registry(), _POLICY, store, "irrelevant-root",
        catalog_path=str(tmp_path / "ops.sqlite"), objects_root=str(tmp_path),
        code_source=ROOT, clock=clock) is None
    assert _native_score_batch_job_count(conn) == before


def test_submit_raises_when_native_identity_lacks_snapshot_id(tmp_path, monkeypatch):
    """A ready identity without its pin fails before it can be submitted."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    score_key = _mark_score_succeeded(conn, clock, session="2026-01-01")
    session, scope_hash = nightly._session_scope_from_score_key(score_key)
    monkeypatch.setattr(nightly, "_native_score_batch_identity",
                        lambda conn: (session, scope_hash, True))

    with pytest.raises(OpsError):
        nightly.submit_native_score_batch_shadow_if_ready(
            conn, registry(), _POLICY, store, "irrelevant-root",
            catalog_path=str(tmp_path / "ops.sqlite"), objects_root=str(tmp_path),
            code_source=ROOT, clock=clock)


def test_submit_builds_one_shadow_batch_request_from_snapshot_parameters(tmp_path, monkeypatch):
    """A snapshot-pinned identity submits one projection worker request.

    The request keeps the score job's identity and carries producer scope and
    pinned catalog/object roots, without doing source reads or staging event
    artifacts in the submit builder.
    """
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    score_key = _mark_score_succeeded(conn, clock, session="2026-01-01")
    session, scope_hash = nightly._session_scope_from_score_key(score_key)
    monkeypatch.setattr(nightly, "_native_score_batch_identity",
                        lambda conn: (session, scope_hash, True))

    snapshot_id = "snapshot-2026-01-01-abc123"
    horizon_days = 21
    tickers = ("FILTER-A", "FILTER-B")
    assert snapshot_id != scope_hash

    captured = []

    def _capture_submit(conn, registry, policy, request, *, clock):
        captured.append(request)
        return JobReceipt(
            job_id=job_id_for(request.namespace, request.idempotency_key),
            namespace=request.namespace, idempotency_key=request.idempotency_key,
            request_digest=request_digest(request), kind=request.job.kind,
            spec_hash=None, state="queued", priority=request.job.priority,
            created_at=clock.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            fence=1, attempt_count=0)

    monkeypatch.setattr("engine.v2.ops.submission.submit", _capture_submit)
    release_root = str(tmp_path / "verified-release-root")
    receipt = nightly.submit_native_score_batch_shadow_if_ready(
        conn, registry(), _POLICY, store, release_root,
        catalog_path=str(tmp_path / "ops.sqlite"), objects_root=str(tmp_path),
        code_source=ROOT, clock=clock, snapshot_id=snapshot_id,
        horizon_days=horizon_days, tickers=tickers)

    assert len(captured) == 1
    request = captured[0]
    _assert_sidecar_runtime_identity(
        request.job.resource_class, request.job.environment_ref, "projection")
    assert request.namespace == "shadow"
    assert request.principal == "operator"
    assert request.job.kind == "native_score_batch"
    assert request.job.retry_policy_ref == "bounded"
    assert request.job.output_namespace == "shadow"
    key = nightly._native_score_batch_key(session, scope_hash)
    assert request.idempotency_key == key
    assert nightly._scope_from_native_score_batch_key(key) == (session, scope_hash)

    parameters = request.job.parameters
    assert parameters["input_bindings"] is None
    assert request.job.input_refs == ()
    assert request.job.dependency_job_ids == ()
    assert parameters["calendar_revision"] == ""
    assert parameters["catalog_path"] == str(tmp_path / "ops.sqlite")
    assert parameters["objects_root"] == str(tmp_path)
    assert parameters["horizon_days"] == horizon_days
    assert parameters["tickers"] == list(tickers)
    assert parameters["producer_mode"] == "snapshot"
    assert parameters["as_of"] == session
    assert parameters["snapshot_id"] == snapshot_id
    assert parameters["release_root"] == release_root
    assert parameters["expected_ids"] == [session + "|" + scope_hash]

    assert receipt is not None and receipt.idempotency_key == key
    assert _native_score_batch_job_count(conn) == 0


def test_service_tick_submits_without_running_pinned_producer(tmp_path, monkeypatch):
    """A slow/failing producer seam is not called by the supervisor tick.

    The tick submits the snapshot-pinned native worker while retaining its
    independent lookup/backoff bookkeeping; the producer runs after claim.
    """
    conn, clock, _ = catalog(tmp_path)
    session = "2026-01-01"
    scope_hash = "sha256:0123456789abcdef01234567"
    snapshot_id = "snap-2026-01-01-producer-fails-99"
    earnings_revision = "earnings-dataset-revision-1313"
    tickers = ("FILTER-A", "FILTER-B")
    horizon_days = 21
    snapshot = SimpleNamespace(
        snapshot_id=snapshot_id,
        table_versions={"earnings_events": SimpleNamespace(
            dataset_version_id=earnings_revision)})
    producer_parameters = LegacyParameters(
        expected_ids=(session + "|" + scope_hash,), session=session,
        tickers=tickers, horizon_days=horizon_days,
        snapshot_generation_id=snapshot_id)

    resolved = []

    def _resolve(repository, snapshot_generation_id):
        resolved.append(snapshot_generation_id)
        return snapshot

    monkeypatch.setattr(Repository, "resolve", _resolve)

    producer_calls = []

    def _producer(repository, resolved_snapshot, *, as_of, horizon_days, tickers):
        producer_calls.append({"as_of": as_of, "horizon_days": horizon_days,
                               "tickers": tickers})
        return [], {"schema_version": "native_score_batch_producer_refusals.v1.0",
                    "refusals": []}

    monkeypatch.setattr(
        "engine.v2.ops.nightly_raw_row_producer.build_native_score_batch_events",
        _producer)
    monkeypatch.setattr(nightly, "_native_score_batch_identity",
                        lambda conn: (session, scope_hash, producer_parameters))

    service = Service(conn, tmp_path, registry(), _POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)
    monkeypatch.setattr(service, "_native_release_root_or_none",
                        lambda: str(tmp_path / "verified-release-root"))

    import time
    started = time.perf_counter()
    service.tick()
    elapsed = time.perf_counter() - started

    assert elapsed < 3.0
    assert resolved == []
    assert producer_calls == []
    assert _native_score_batch_job_count(conn) == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM artifacts").fetchone()["n"] == 0
    assert service._native_score_batch_memo is None
    assert service._native_score_batch_lookup_memo is None


def test_service_tick_completes_while_a_claimed_producer_is_blocked(tmp_path, monkeypatch):
    """A blocked native worker producer must not delay the tick or sibling lease.

    The real native worker resolves and calls the snapshot producer on its
    worker thread. While that call is blocked, the tick still submits its
    shadow job and renews every OTHER running attempt's lease.
    """
    import threading
    import time

    from engine.v2.ops import native_score_batch

    conn, clock, supervisor = catalog(tmp_path)
    session = "2026-01-01"
    snapshot_id = "snap-2026-01-01-producer-blocks-99"
    earnings_revision = "earnings-dataset-revision-1313"
    tickers = ("FILTER-A", "FILTER-B")
    horizon_days = 21
    snapshot = SimpleNamespace(
        snapshot_id=snapshot_id,
        table_versions={"earnings_events": SimpleNamespace(
            dataset_version_id=earnings_revision)})
    score_key = _mark_score_succeeded(conn, clock, session=session, tickers=tickers)
    score_session, score_scope = nightly._session_scope_from_score_key(score_key)
    other_job_id = job_id_for("shadow", score_key)
    producer_parameters = LegacyParameters(
        expected_ids=(score_session + "|" + score_scope,), session=score_session,
        tickers=tickers, horizon_days=horizon_days,
        snapshot_generation_id=snapshot_id)

    resolved = []

    def _resolve(repository, snapshot_generation_id):
        resolved.append(snapshot_generation_id)
        return snapshot

    monkeypatch.setattr(Repository, "resolve", _resolve)
    catalog_path = conn.execute(
        "PRAGMA database_list").fetchone()["file"]

    producer_calls = []
    producer_started = threading.Event()
    release_producer = threading.Event()

    def _producer(repository, resolved_snapshot, *, as_of, horizon_days, tickers):
        producer_calls.append({
            "snapshot": resolved_snapshot, "as_of": as_of,
            "horizon_days": horizon_days, "tickers": tickers})
        producer_started.set()
        release_producer.wait(timeout=5)
        return [], {"schema_version": "native_score_batch_producer_refusals.v1.0",
                    "refusals": []}

    monkeypatch.setattr(
        "engine.v2.ops.nightly_raw_row_producer.build_native_score_batch_events",
        _producer)
    monkeypatch.setattr(
        nightly, "_native_score_batch_identity",
        lambda conn: (score_session, score_scope, producer_parameters))
    from engine.v2.scoring import release_bindings
    monkeypatch.setattr(
        release_bindings, "resolve_release_binding", lambda root: object())
    monkeypatch.setattr(
        release_bindings, "resolve_gate_policy", lambda binding, root: {})
    monkeypatch.setattr(
        native_score_batch, "assemble_score_batch_inputs",
        lambda **kwargs: ({}, ()))
    monkeypatch.setattr(
        native_score_batch, "score_batch", lambda batch, fields: object())
    monkeypatch.setattr(
        native_score_batch, "_native_score_batch_documents",
        lambda *args: (
            {"schema_version": "native_score_batch_records.v2.0", "records": {}},
            {"schema_version": "native_score_batch_refusals.v2.0",
             "refusals": {}, "unkeyable_refusals": []}))

    worker_root = tmp_path / "blocked-native-worker"
    worker_root.mkdir()
    worker_parameters = {
        "expected_ids": [score_session + "|" + score_scope],
        "release_root": str(tmp_path / "verified-release-root"),
        "as_of": score_session,
        "snapshot_id": snapshot_id,
        "feature_names": [],
        "producer_mode": "snapshot",
        "catalog_path": catalog_path,
        "objects_root": str(tmp_path / "objects"),
        "horizon_days": horizon_days,
        "tickers": list(tickers),
    }
    worker_results = []
    worker = threading.Thread(
        target=lambda: worker_results.append(
            native_score_batch.run_native_score_batch_worker(
                worker_parameters, worker_root)))
    worker.start()

    service = Service(conn, tmp_path, registry(), TEST_POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)
    service.identity = supervisor
    monkeypatch.setattr(
        service, "_native_release_root_or_none",
        lambda: str(tmp_path / "verified-release-root"))
    claim = SimpleNamespace(
        job_id=other_job_id, attempt_id="other-attempt", fence=1)
    service.running[claim.attempt_id] = SimpleNamespace(
        claim=claim, failure=None, peak=0)
    monkeypatch.setattr(
        "engine.v2.ops.executor.poll",
        lambda conn, running, **kwargs: {"done": False, "memory": 0})
    monkeypatch.setattr(service, "_track_steps", lambda running, status: None)
    monkeypatch.setattr(service, "_observation_due", lambda running, status: False)
    heartbeat_calls = []

    def _heartbeat(conn, attempt_id, fence, *, clock, lease_seconds):
        heartbeat_calls.append((attempt_id, lease_seconds))
        return True

    monkeypatch.setattr("engine.v2.ops.supervisor.heartbeat", _heartbeat)
    monkeypatch.setattr(service, "_launch", lambda claim: None)

    try:
        assert producer_started.wait(timeout=1)
        clock.advance(1)
        started = time.perf_counter()
        service.tick()
        elapsed = time.perf_counter() - started

        assert elapsed < 3.0
        assert worker.is_alive()
        assert ("other-attempt", 120) in heartbeat_calls
        assert resolved == [snapshot_id]
        assert producer_calls == [{
            "snapshot": snapshot, "as_of": score_session,
            "horizon_days": horizon_days, "tickers": list(tickers)}]
        assert _native_score_batch_job_count(conn) == 1
        row = conn.execute(
            "SELECT idempotency_key FROM jobs WHERE kind = ?",
            ("native_score_batch",)).fetchone()
        assert row["idempotency_key"] == nightly._native_score_batch_key(
            score_session, score_scope)
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM artifacts").fetchone()["n"] == 0
    finally:
        release_producer.set()
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert worker_results
    assert worker_results[0]["completed_ids"] == [score_session + "|" + score_scope]
    assert service._native_score_batch_memo is None
    assert service._native_score_batch_lookup_memo is None


def test_service_submits_worker_producer_parameters_without_staging(tmp_path, monkeypatch):
    """The active tick sidecar submits producer scope without source reads.

    Snapshot resolution and the complete event/refusal production happen in
    the claimed native worker, so the sidecar leaves no staged artifacts.
    """
    conn, clock, _ = catalog(tmp_path)
    session = "2026-01-01"
    scope_hash = "sha256:0123456789abcdef01234567"
    snapshot_id = "snap-2026-01-01-staged-77"
    earnings_revision = "earnings-dataset-revision-4242"
    tickers = ("FILTER-A", "FILTER-B")
    horizon_days = 21
    release_root = str(tmp_path / "verified-release-root")
    snapshot = SimpleNamespace(
        snapshot_id=snapshot_id,
        table_versions={"earnings_events": SimpleNamespace(
            dataset_version_id=earnings_revision)})
    producer_parameters = LegacyParameters(
        expected_ids=(session + "|" + scope_hash,), session=session,
        tickers=tickers, horizon_days=horizon_days,
        snapshot_generation_id=snapshot_id)

    events_document = [
        {"event_id": session + "|" + scope_hash + "|FILTER-A",
         "ticker": "FILTER-A", "as_of": session, "horizon_days": horizon_days},
    ]
    refusals_document = {
        "schema_version": "native_score_batch_producer_refusals.v1.0",
        "refusals": [{"event_id": session + "|" + scope_hash + "|FILTER-B",
                      "code": "INTRADAY_EVENT_NOT_ADMITTED"}]}

    resolved = []

    def _resolve(repository, snapshot_generation_id):
        resolved.append(snapshot_generation_id)
        return snapshot

    monkeypatch.setattr(Repository, "resolve", _resolve)

    producer_calls = []

    def _producer(repository, resolved_snapshot, *, as_of, horizon_days, tickers):
        producer_calls.append({
            "repository": repository, "snapshot": resolved_snapshot,
            "as_of": as_of, "horizon_days": horizon_days, "tickers": tickers})
        return events_document, refusals_document

    monkeypatch.setattr(
        "engine.v2.ops.nightly_raw_row_producer.build_native_score_batch_events",
        _producer)
    monkeypatch.setattr(nightly, "_native_score_batch_identity",
                        lambda conn: (session, scope_hash, producer_parameters))

    service = Service(conn, tmp_path, registry(), _POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)
    monkeypatch.setattr(service, "_native_release_root_or_none", lambda: release_root)

    service._reconcile_native_score_batch_shadow()  # must not raise

    assert resolved == []
    assert producer_calls == []
    assert _native_score_batch_job_count(conn) == 1
    row = conn.execute(
        "SELECT namespace, state, resource_class, spec_json FROM jobs WHERE kind = ?",
        ("native_score_batch",)).fetchone()
    assert row["namespace"] == "shadow"
    assert row["state"] == "queued"
    spec = json.loads(row["spec_json"])
    _assert_sidecar_runtime_identity(
        row["resource_class"], spec["environment_ref"], "projection")
    assert spec["input_refs"] == []
    parameters = spec["parameters"]
    assert parameters["input_bindings"] is None
    assert parameters["snapshot_id"] == snapshot_id
    assert parameters["calendar_revision"] == ""
    assert parameters["as_of"] == session
    assert parameters["release_root"] == release_root
    assert parameters["catalog_path"] == str(tmp_path / "ops.sqlite")
    assert parameters["objects_root"] == str(tmp_path)
    assert parameters["horizon_days"] == horizon_days
    assert parameters["tickers"] == list(tickers)
    assert parameters["expected_ids"] == [session + "|" + scope_hash]
    assert conn.execute("SELECT COUNT(*) AS n FROM artifacts").fetchone()["n"] == 0

    service._reconcile_native_score_batch_shadow()  # existing job: pure no-op
    assert producer_calls == []
    assert _native_score_batch_job_count(conn) == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM artifacts").fetchone()["n"] == 0


def test_native_worker_produces_events_from_its_pinned_snapshot(tmp_path, monkeypatch):
    from engine.v2.ops import native_score_batch

    class Connection:
        closed = False

        def close(self):
            self.closed = True

    conn = Connection()
    snapshot_id = "snapshot-worker-test"
    snapshot = SimpleNamespace(
        snapshot_id=snapshot_id,
        table_versions={"earnings_events": SimpleNamespace(
            dataset_version_id="earnings-revision-worker-test")})

    class Repository:
        def __init__(self, connection, store):
            assert connection is conn
            assert isinstance(store, ArtifactStore)

        def resolve(self, requested_snapshot_id):
            assert requested_snapshot_id == snapshot_id
            return snapshot

    catalog_paths = []
    monkeypatch.setattr("engine.v2.ops.catalog.connect",
                        lambda path, must_exist=True: catalog_paths.append(path) or conn)
    monkeypatch.setattr("engine.v2.data.repository.Repository", Repository)
    produced = []
    producer_refusals = {
        "schema_version": "native_score_batch_producer_refusals.v1.0",
        "refusals": [],
    }

    def producer(repository, resolved_snapshot, *, as_of, horizon_days, tickers):
        produced.append((resolved_snapshot, as_of, horizon_days, tickers))
        return [], producer_refusals

    monkeypatch.setattr(
        "engine.v2.ops.nightly_raw_row_producer.build_native_score_batch_events",
        producer)
    from engine.v2.scoring import release_bindings
    monkeypatch.setattr(release_bindings, "resolve_release_binding", lambda root: object())
    monkeypatch.setattr(release_bindings, "resolve_gate_policy", lambda binding, root: {})
    assembled_calls = []
    monkeypatch.setattr(
        native_score_batch, "assemble_score_batch_inputs",
        lambda **kwargs: (assembled_calls.append(kwargs) or ({}, ())))
    monkeypatch.setattr(native_score_batch, "score_batch", lambda batch, fields: object())
    records_doc = {"schema_version": "native_score_batch_records.v2.0", "records": {}}
    refusals_doc = {"schema_version": "native_score_batch_refusals.v2.0",
                    "refusals": {}, "unkeyable_refusals": []}
    monkeypatch.setattr(native_score_batch, "_native_score_batch_documents",
                        lambda *args: (records_doc, refusals_doc))
    root = tmp_path / "worker-staging"
    root.mkdir()
    # A retry may reuse its staging directory; production inputs must be rebuilt.
    (root / "events.json").write_text("[]", encoding="utf-8")
    (root / "producer_refusals.json").write_text(
        json.dumps({"schema_version": "native_score_batch_producer_refusals.v1.0",
                    "refusals": [{"stale": True}]}), encoding="utf-8")

    result = native_score_batch.run_native_score_batch_worker({
        "expected_ids": ["session|scope"], "release_root": str(tmp_path / "release"),
        "as_of": "2026-01-01", "snapshot_id": snapshot_id,
        "calendar_revision": "", "feature_names": [],
        "producer_mode": "snapshot",
        "catalog_path": str(tmp_path / "catalog.sqlite"),
        "objects_root": str(tmp_path / "objects"),
        "horizon_days": 21, "tickers": ["AAA", "BBB"],
    }, root)

    assert catalog_paths == [str(tmp_path / "catalog.sqlite")]
    assert conn.closed
    assert produced == [(snapshot, "2026-01-01", 21, ["AAA", "BBB"])]
    assert assembled_calls[0]["snapshot_id"] == snapshot_id
    assert assembled_calls[0]["calendar_revision"] == "earnings-revision-worker-test"
    assert (root / "events.json").read_text(encoding="utf-8") == "[]"
    assert json.loads((root / "producer_refusals.json").read_text()) == producer_refusals
    assert result["outputs"][0]["path"] == "records.json"

    def failed_producer(repository, resolved_snapshot, *, as_of, horizon_days, tickers):
        raise RuntimeError("producer failed")

    monkeypatch.setattr(
        "engine.v2.ops.nightly_raw_row_producer.build_native_score_batch_events",
        failed_producer)
    failed_root = tmp_path / "worker-failure"
    failed_root.mkdir()
    with pytest.raises(RuntimeError, match="producer failed"):
        native_score_batch.run_native_score_batch_worker({
            "expected_ids": ["session|scope"], "release_root": str(tmp_path / "release"),
            "as_of": "2026-01-01", "snapshot_id": snapshot_id,
            "calendar_revision": "", "feature_names": [],
            "producer_mode": "snapshot",
            "catalog_path": str(tmp_path / "catalog.sqlite"),
            "objects_root": str(tmp_path / "objects"),
            "horizon_days": 21, "tickers": ["AAA", "BBB"],
        }, failed_root)
    assert not (failed_root / "events.json").exists()
    assert not (failed_root / "producer_refusals.json").exists()
    assert not (failed_root / "records.json").exists()
    assert not (failed_root / "refusals.json").exists()


def test_worker_without_producer_parameters_requires_staged_events(tmp_path):
    from engine.v2.ops.native_score_batch import _worker_event_documents

    with pytest.raises(FileNotFoundError):
        _worker_event_documents({"calendar_revision": "earnings-rev"}, tmp_path)
    (tmp_path / "events.json").write_text("[]", encoding="utf-8")
    assert _worker_event_documents({
        "calendar_revision": "earnings-rev",
        "catalog_path": str(tmp_path / "catalog.sqlite"),
    }, tmp_path) == ([], None, "earnings-rev")
    with pytest.raises(OpsError, match="VALIDATION_FAILED"):
        _worker_event_documents({"producer_mode": "snapshot"}, tmp_path)
    with pytest.raises(OpsError, match="VALIDATION_FAILED"):
        _worker_event_documents({
            "producer_mode": "snapshot",
            "catalog_path": str(tmp_path / "catalog.sqlite"),
        }, tmp_path)


# --------------------------------------------------------------------------
# Service's release-root gate
# --------------------------------------------------------------------------


def test_service_tick_never_submits_without_model_release_root(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    _mark_score_succeeded(conn, clock, session="2026-01-01")
    monkeypatch.delenv(deployment.MODEL_RELEASE_ROOT_ENV, raising=False)
    service = Service(conn, tmp_path, registry(), _POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)

    service._reconcile_native_score_batch_shadow()  # must not raise

    assert _native_score_batch_job_count(conn) == 0


def test_service_sidecar_keeps_legacy_unpinned_identity_as_noop(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    _mark_score_succeeded(conn, clock, session="2026-01-01")
    service = Service(conn, tmp_path, registry(), _POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)
    monkeypatch.setattr(service, "_native_release_root_or_none", lambda: "fake-release")

    service._reconcile_native_score_batch_shadow()

    assert _native_score_batch_job_count(conn) == 0
    assert service._native_score_batch_lookup_memo is None


def test_service_tick_never_submits_with_release_root_but_no_promoted_release(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    _mark_score_succeeded(conn, clock, session="2026-01-01")
    monkeypatch.setenv(deployment.MODEL_RELEASE_ROOT_ENV, str(tmp_path))
    service = Service(conn, tmp_path, registry(), _POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)

    service._reconcile_native_score_batch_shadow()  # must not raise

    assert _native_score_batch_job_count(conn) == 0
    # current_pointer returned None on the cheap check alone -- nothing to memoize.
    assert service._native_release_memo is None


def test_identity_lookup_failure_is_throttled_not_retried_every_tick(tmp_path, monkeypatch):
    """CodeRabbit round 6, real finding: before the fix, a persistently
    raising _native_score_batch_identity ran again on EVERY tick -- the
    not_before it wrote into the memo was never read back. This bypasses
    the release-root gate (irrelevant to the bug) by monkeypatching
    _native_release_root_or_none directly."""
    conn, clock, _ = catalog(tmp_path)
    service = Service(conn, tmp_path, registry(), _POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)
    monkeypatch.setattr(service, "_native_release_root_or_none", lambda: "fake-root")
    calls = []

    def _boom(conn):
        calls.append(1)
        raise ValueError("planted identity lookup failure")

    monkeypatch.setattr(nightly, "_native_score_batch_identity", _boom)

    service._reconcile_native_score_batch_shadow()  # 1st failure: attempts -> 1, backed off
    assert len(calls) == 1
    assert service._native_score_batch_lookup_memo["attempts"] == 1
    assert service._native_score_batch_lookup_memo["not_before"] > 0

    service._reconcile_native_score_batch_shadow()  # still inside backoff: must NOT call again
    assert len(calls) == 1

    clock.advance(10_000)  # well past the backoff window
    service._reconcile_native_score_batch_shadow()
    assert len(calls) == 2
    assert service._native_score_batch_lookup_memo["attempts"] == 2


def test_identity_lookup_failure_never_spends_a_real_identitys_attempt_budget(tmp_path, monkeypatch):
    """CodeRabbit round 7, real finding: a lookup failure must not
    overwrite/replace a real identity's own build-attempt memo -- the two
    now live in separate slots entirely."""
    conn, clock, _ = catalog(tmp_path)
    service = Service(conn, tmp_path, registry(), _POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)
    monkeypatch.setattr(service, "_native_release_root_or_none", lambda: "fake-root")
    real_identity = ("2026-01-01", "sha256:realhash", False)
    service._native_score_batch_memo = {"identity": real_identity, "attempts": 3,
                                        "not_before": 0.0}

    def _boom(conn):
        raise ValueError("planted identity lookup failure")

    monkeypatch.setattr(nightly, "_native_score_batch_identity", _boom)

    service._reconcile_native_score_batch_shadow()

    # The lookup failure landed in its OWN slot...
    assert service._native_score_batch_lookup_memo["attempts"] == 1
    # ...and the real identity's own memo (attempts=3) is untouched.
    assert service._native_score_batch_memo == {"identity": real_identity, "attempts": 3,
                                                 "not_before": 0.0}
