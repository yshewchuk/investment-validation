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

from pathlib import Path

from engine.v2.foundation import ArtifactStore
from engine.v2.models import deployment
from engine.v2.ops import nightly
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, job_id_for, submit_graph
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

    assert nightly._native_score_batch_identity(conn) == (session, scope_hash, False)

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
