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

import pytest

from engine.v2.foundation import ArtifactStore
from engine.v2.models import deployment
from engine.v2.ops import nightly
from engine.v2.ops.errors import OpsError
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


def test_identity_orders_by_created_at_not_scope_hash(tmp_path):
    """CodeRabbit/Opus gate finding: the old code picked the "latest" score
    job in a session by ``max(scope_hash)`` -- a content hash with no time
    meaning -- so whichever of two same-session jobs (e.g. a legacy-mode
    run followed by a snapshot-mode rerun) happened to hash higher won, not
    whichever ran later. This builds two real succeeded "score" jobs in the
    SAME session with different ``scope_hash`` (different ticker sets),
    searching ticker candidates (computing each candidate's real
    idempotency key through the same ``build_legacy_job_requests`` path,
    never a hand-written hash) until the SECOND, later-created job has the
    LOWER scope_hash -- guaranteeing hash order and creation order
    disagree -- then asserts the later-created job wins.

    Every candidate (including the first job) is built from the SAME
    ``plan`` object, never a fresh ``build_nightly_plan`` call per job:
    ``build_nightly_plan`` stamps ``decision_clock`` from a real
    ``SystemClock`` when no ``clock=`` is passed, so two separate calls
    fold two different wall-clock instants into ``scope_hash`` via
    ``plan_identity`` -- making hashes from different calls incomparable
    and the ``<`` comparison below flaky by chance (~50%). Reusing one
    ``plan`` keeps ``decision_clock`` fixed, so only ``tickers`` varies
    each candidate's hash.
    """
    conn, clock, _ = catalog(tmp_path)
    session = "2026-01-01"
    plan = nightly.build_nightly_plan(ROOT, session)

    def _mark(tickers):
        requests = nightly.build_legacy_job_requests(plan, tickers=tickers,
                                                       year_start=2025, year_end=2026)
        submit_graph(conn, registry(), _POLICY, requests, clock=clock)
        score = next(r for r in requests if r.idempotency_key.endswith(":score"))
        job_id = job_id_for("shadow", score.idempotency_key)
        conn.execute("UPDATE jobs SET state = 'succeeded', updated_at = ? WHERE job_id = ?",
                     (clock.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"), job_id))
        conn.commit()
        return score.idempotency_key

    candidate_tickers = [("FIRST",)] + [(f"CAND{candidate}",) for candidate in range(200)]
    candidate_hashes = []
    for tickers in candidate_tickers:
        requests = nightly.build_legacy_job_requests(plan, tickers=tickers,
                                                       year_start=2025, year_end=2026)
        score_request = next(r for r in requests if r.idempotency_key.endswith(":score"))
        _, candidate_hash = nightly._session_scope_from_score_key(score_request.idempotency_key)
        candidate_hashes.append((candidate_hash, tickers))

    first_hash, first_tickers = max(candidate_hashes, key=lambda item: item[0])
    later_hash, later_tickers = min(candidate_hashes, key=lambda item: item[0])
    assert later_hash < first_hash, "candidate hashes must be distinct"

    first_key = _mark(first_tickers)
    clock.advance(60)
    later_key = _mark(later_tickers)
    assert later_hash < first_hash  # hash order alone would pick `first`

    identity = nightly._native_score_batch_identity(conn)

    assert identity[:2] == (session, later_hash)  # time order (the fix) picks `later`


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
