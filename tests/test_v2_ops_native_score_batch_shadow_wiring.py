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
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.contracts import JobReceipt
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, canonical_json
from engine.v2.models import deployment
from engine.v2.ops import nightly
from engine.v2.ops.checkpoints import registered_artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.stages import LegacyParameters, registry
from engine.v2.ops.submission import (
    NamespacePolicy,
    job_id_for,
    request_digest,
    submit_graph,
)
from engine.v2.ops.supervisor import Service
from tests.ops_support import catalog

ROOT = Path(__file__).resolve().parents[1]
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
    assert nightly.GRAPH["native_score_batch"] == ("score",)
    assert "native_score_batch" in nightly.OPTIONAL


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


def test_submit_raises_when_snapshot_pinned_and_producer_missing(tmp_path, monkeypatch):
    """B2: with a snapshot-pinned "score" job and NO existing job under the
    batch key, the unbuilt PR-6 raw-row producer must raise so
    ``Service._reconcile_native_score_batch_shadow``'s problem-reporting and
    backoff machinery surfaces it -- never silently return ``None``."""
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


def test_submit_builds_one_shadow_batch_request_from_the_slice5_refs(tmp_path, monkeypatch):
    """Slice 5 regression: the valid snapshot-pinned path now BUILDS and
    submits instead of raising -- and it builds only from the three
    caller-supplied refs. ``submission.submit`` is intercepted, so this pins
    the ONE request the builder hands that boundary (ARCHITECTURE.md "The
    builder only builds"), never what admission does with it: the staged
    ``events.json``/``producer_refusals.json`` artifact ids bind both worker
    files AND ride ``JobSpec.input_refs`` (a direct artifact binding is only
    admitted through ``spec.input_refs``, ``input_bindings.py:66``), and the
    supplied ``calendar_revision`` is serialized verbatim into
    ``NativeScoreBatchParameters``. The PR-6 producer is patched to a
    tripwire -- the builder must never call it (importing the producer here
    to patch it is the test's own import, never nightly.py's)."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    score_key = _mark_score_succeeded(conn, clock, session="2026-01-01")
    session, scope_hash = nightly._session_scope_from_score_key(score_key)
    monkeypatch.setattr(nightly, "_native_score_batch_identity",
                        lambda conn: (session, scope_hash, True))

    events_ref = store.publish_bytes(
        b"[]", schema_ref="native_score_batch_events.v1.0").artifact_id
    producer_refusals_ref = store.publish_bytes(
        b'{"schema_version": "native_score_batch_producer_refusals.v1.0", "refusals": []}',
        schema_ref="native_score_batch_producer_refusals.v1.0").artifact_id
    calendar_revision = "earnings-events-dataset-rev-42"
    snapshot_id = "snapshot-2026-01-01-abc123"
    assert len({events_ref, producer_refusals_ref, calendar_revision}) == 3
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
    producer_calls = []

    def _producer_tripwire(*args, **kwargs):
        producer_calls.append((args, kwargs))
        raise AssertionError("the builder must not call the PR-6 producer")

    monkeypatch.setattr(
        "engine.v2.ops.nightly_raw_row_producer.build_native_score_batch_events",
        _producer_tripwire)

    release_root = str(tmp_path / "verified-release-root")
    receipt = nightly.submit_native_score_batch_shadow_if_ready(
        conn, registry(), _POLICY, store, release_root,
        catalog_path=str(tmp_path / "ops.sqlite"), objects_root=str(tmp_path),
        code_source=ROOT, clock=clock, events_ref=events_ref,
        producer_refusals_ref=producer_refusals_ref, calendar_revision=calendar_revision,
        snapshot_id=snapshot_id)

    assert len(captured) == 1
    request = captured[0]
    assert request.namespace == "shadow"
    assert request.principal == "operator"
    assert request.job.kind == "native_score_batch"
    assert request.job.output_namespace == "shadow"
    # key/identity behavior is unchanged: the batch key is still THIS score
    # job's own session+scope_hash, and the existing parser round-trips it.
    key = nightly._native_score_batch_key(session, scope_hash)
    assert request.idempotency_key == key
    assert nightly._scope_from_native_score_batch_key(key) == (session, scope_hash)

    parameters = request.job.parameters
    assert parameters["input_bindings"] == {
        "events.json": events_ref, "producer_refusals.json": producer_refusals_ref}
    assert request.job.input_refs == (events_ref, producer_refusals_ref)
    assert parameters["calendar_revision"] == calendar_revision
    assert parameters["as_of"] == session
    assert parameters["snapshot_id"] == snapshot_id
    assert parameters["release_root"] == release_root
    assert parameters["expected_ids"] == [session + "|" + scope_hash]

    assert receipt is not None and receipt.idempotency_key == key
    assert producer_calls == []
    # intercepted at the boundary: the builder wrote nothing to the catalog.
    assert _native_score_batch_job_count(conn) == 0


def test_service_backs_off_one_attempt_when_pinned_producer_fails(tmp_path, monkeypatch):
    """Slice 5's whole-producer-failure path: a snapshot-pinned identity whose
    ``Repository.resolve`` succeeds but whose PR-6 ``build_native_score_batch_events``
    raise must surface through ``Service._reconcile_native_score_batch_shadow``'s
    catch/backoff -- the exception lands BEFORE any publish/register/submit, so
    nothing is submitted or registered -- with exactly one spent sidecar build
    attempt and ``not_before`` pushed into the future, while the separate
    identity-LOOKUP memo (whose own lookup succeeded this tick) stays untouched."""
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
        raise RuntimeError("planted whole-producer failure")

    monkeypatch.setattr(
        "engine.v2.ops.nightly_raw_row_producer.build_native_score_batch_events",
        _producer)
    monkeypatch.setattr(nightly, "_native_score_batch_identity",
                        lambda conn: (session, scope_hash, producer_parameters))

    service = Service(conn, tmp_path, registry(), _POLICY, clock=clock,
                      code_source=ROOT, store_root=tmp_path)
    monkeypatch.setattr(service, "_native_release_root_or_none",
                        lambda: str(tmp_path / "verified-release-root"))

    service._reconcile_native_score_batch_shadow()  # must not raise

    assert resolved == [snapshot_id]  # resolve ran, on the pinned generation id
    assert len(producer_calls) == 1
    call = producer_calls[0]
    assert call["as_of"] == session
    assert call["horizon_days"] == horizon_days
    assert call["tickers"] == tickers
    assert _native_score_batch_job_count(conn) == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM artifacts").fetchone()["n"] == 0
    memo = service._native_score_batch_memo
    assert memo is not None
    assert memo["identity"] == (session, scope_hash, producer_parameters)
    assert memo["attempts"] == 1
    assert memo["not_before"] > clock.monotonic()
    # the identity lookup itself succeeded this tick -- its own failure
    # memo must be untouched (None), never conflated with this backoff.
    assert service._native_score_batch_lookup_memo is None


def test_service_stages_both_producer_documents_before_shadow_submission(tmp_path, monkeypatch):
    """Slice 5's core claim: a snapshot-pinned identity reaches the REAL
    shadow submission only after the sidecar's full staging path ran -- the
    raw-row producer is called exactly once for this session with the
    identity's own nondefault ``horizon_days`` and ticker filter and the
    exact snapshot ``Repository.resolve`` returned, both JSON-ready
    documents are then published into the real artifact store AND
    registered in the catalog, and the submitted job binds the SAME two
    artifact ids that its ``input_bindings`` name (in the producer's own
    order) -- so a job can never reference a one-sided stage. Everything
    except the four documented seams stays real: the catalog connection,
    the ``Service`` and its ``ArtifactStore``, ``register_artifact``,
    ``Repository`` itself, and ``submit_native_score_batch_shadow_if_ready``
    (the test never intercepts the submission boundary)."""
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

    assert resolved == [snapshot_id]
    assert len(producer_calls) == 1
    call = producer_calls[0]
    assert isinstance(call["repository"], Repository)
    assert call["snapshot"] is snapshot
    assert call["as_of"] == session
    assert call["horizon_days"] == horizon_days
    assert call["tickers"] == tickers

    assert _native_score_batch_job_count(conn) == 1
    row = conn.execute(
        "SELECT namespace, state, spec_json FROM jobs WHERE kind = ?",
        ("native_score_batch",)).fetchone()
    assert row["namespace"] == "shadow"
    assert row["state"] == "queued"
    spec = json.loads(row["spec_json"])

    events_ref, producer_refusals_ref = spec["input_refs"]
    assert len({events_ref, producer_refusals_ref, snapshot_id, earnings_revision,
                scope_hash}) == 5
    parameters = spec["parameters"]
    assert parameters["input_bindings"] == {
        "events.json": events_ref, "producer_refusals.json": producer_refusals_ref}
    assert parameters["snapshot_id"] == snapshot_id
    assert parameters["calendar_revision"] == earnings_revision
    assert parameters["as_of"] == session
    assert parameters["release_root"] == release_root
    assert parameters["expected_ids"] == [session + "|" + scope_hash]

    registered = {ref: registered_artifact(conn, ref)
                  for ref in (events_ref, producer_refusals_ref)}
    assert set(registered) == {events_ref, producer_refusals_ref}
    assert registered[events_ref].schema_ref == "native_score_batch_events.v1.0"
    assert registered[producer_refusals_ref].schema_ref == \
        "native_score_batch_producer_refusals.v1.0"
    assert service.store.read_verified(registered[events_ref]) == \
        canonical_json(events_document).encode()
    assert service.store.read_verified(registered[producer_refusals_ref]) == \
        canonical_json(refusals_document).encode()

    artifact_rows_before = conn.execute("SELECT COUNT(*) AS n FROM artifacts").fetchone()["n"]

    service._reconcile_native_score_batch_shadow()  # existing job: must be a pure no-op

    assert len(producer_calls) == 1
    assert _native_score_batch_job_count(conn) == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM artifacts").fetchone()["n"] == \
        artifact_rows_before


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
