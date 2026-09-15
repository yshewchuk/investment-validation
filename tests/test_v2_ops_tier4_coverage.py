"""P2-C02 (Phase 2 review closeout): launch refusal for missing Tier-4
serving-cache coverage, and same-generation binding for barrier-only stages.

The coverage half reuses ``tests/test_v2_ops_snapshot_stages.py``'s ``Case``
fixture (real SQLite catalog + ``ArtifactStore`` + a real committed synthetic
snapshot) and runs the REAL ``legacy_materialize`` worker subprocess, exactly
as that file does, so the pinned Tier-4 cache files this task's check reads
are genuinely materialized on disk -- not mocked. Only the ``legacy_score``
worker's argv is swapped for the recording stub the sibling file already
installs (the real legacy scorer cannot run on this synthetic tree).

The generation-binding half exercises ``engine.v2.ops.generation_binding``
directly against the same real catalog/store, plus one direct call into
``Service._pin_read_set`` proving the barrier-kind hook actually reaches it.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import joblib
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.data import legacy_materialization as lm  # noqa: E402
from engine.v2.data.reference_inputs import TIER4_SERVING_DIR  # noqa: E402
from engine.v2.foundation import to_document  # noqa: E402
from engine.v2.ops.catalog import transaction  # noqa: E402
from engine.v2.ops.checkpoints import register_artifact  # noqa: E402
from engine.v2.ops.errors import OpsError  # noqa: E402
from engine.v2.ops.generation_binding import accepted_generation_refs, refuse_generation_mismatch  # noqa: E402
from engine.v2.ops.scheduler import Claim  # noqa: E402
from engine.v2.ops.stages import BARRIER_ONLY_REASONS  # noqa: E402
from engine.v2.ops.supervisor import Service  # noqa: E402
from tests.test_v2_data_legacy_materialization import CALENDAR_PATH, DIRECT_SCOPE, EVIDENCE_SCOPE  # noqa: E402
from tests.test_v2_ops_snapshot_stages import MANIFEST, SESSION, Case, case  # noqa: E402,F401

EXPECTED_POPULATION = {"earnings_events": 2, "daily_market": 3, "trades": 2,
                       "option_chains": 2, "feature_panel": 2, "tier4_forecasts": 2}
#: "AAA|S1|2020-01-15" (case.submit_score's own default population) folds to
#: 2020-01 against SESSION="2026-09-12" -- Jan 2020 is far earlier than the
#: decision date, so the served fold is the EVENT's own fold.
FOLD = "202001"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _panel_sha(case) -> str:
    (record,) = case.repository.fragment_records(case.snap, "feature_panel")
    return record.object_ref.content_hash.removeprefix("sha256:")


def _cache_bytes(*, model_id="size", fold_start="2020-01-01", tier3_snapshot, features=("a",)) -> bytes:
    buf = io.BytesIO()
    joblib.dump({"estimator": object(), "model_id": model_id, "fold_start": fold_start,
                "tier3_snapshot": tier3_snapshot, "features": list(features)}, buf)
    return buf.getvalue()


def _registry_bytes(*, features=("a",)) -> bytes:
    return json.dumps({"models": [{"id": "size", "champion": True, "produces": "pred_abs_move",
                                   "features": list(features), "target": "y"}]}).encode()


def _custom_request(case, *, registry_bytes, tier4_refs=()):
    registry_hash = case.store.publish_bytes(registry_bytes, schema_ref="legacy_pinned_ref.v1").content_hash
    calendar_hash = case.store.publish_bytes(
        b"date\n2020-01-02\n2021-01-04\n", schema_ref="legacy_pinned_ref.v1").content_hash
    registry_refs = (lm.format_pinned_ref("engine/models/registry.json", registry_hash), *tier4_refs)
    calendar_refs = (lm.format_pinned_ref(CALENDAR_PATH, calendar_hash),)
    return lm.build_materialization_request(
        case.repository, case.store, case.snap, case.snapshot_object,
        direct_scope=DIRECT_SCOPE, evidence_scope=EVIDENCE_SCOPE,
        registry_and_model_refs=registry_refs, calendar_refs=calendar_refs,
        expected_population=EXPECTED_POPULATION)


def _materialize_and_score(case, monkeypatch, *, registry_bytes, tier4_refs, key):
    request = _custom_request(case, registry_bytes=registry_bytes, tier4_refs=tier4_refs)
    request_ref = case.publish(to_document(request), "legacy_materialization_request.v1.0")
    mat = case.submit_materialize(f"mat-{key}", request_ref=request_ref)
    assert case.run(mat) == "succeeded", case.failure(mat)
    case.install_stub(monkeypatch)
    stage = case.submit_score(f"score-{key}", mat + "#" + MANIFEST, deps=(mat,), request_ref=request_ref)
    return stage


# --------------------------------------------------------------------------
# launch refusal: TIER4_CACHE_MISSING
# --------------------------------------------------------------------------


def test_covered_population_launches(case, monkeypatch):
    """A champion registered + its exact serving cache pinned and
    materialized for the required fold -> the launch proceeds."""
    panel_sha = _panel_sha(case)
    cache = case.store.publish_bytes(
        _cache_bytes(tier3_snapshot=panel_sha), schema_ref="legacy_pinned_ref.v1")
    ref = lm.format_pinned_ref(f"{TIER4_SERVING_DIR}/size_{FOLD}_{panel_sha[:12]}.joblib", cache.content_hash)
    stage = _materialize_and_score(case, monkeypatch, registry_bytes=_registry_bytes(),
                                   tier4_refs=(ref,), key="cov")
    assert case.run(stage) == "succeeded", case.failure(stage)
    (record,) = case.recorded()
    assert record["envelope"]["parameters"]["expected_ids"] == ["legacy_score"]


def test_missing_cache_refuses_before_any_attempt(case, monkeypatch):
    """A champion registered but NO serving cache pinned at all -> refused
    with TIER4_CACHE_MISSING, no attempt output, before the worker runs."""
    stage = _materialize_and_score(case, monkeypatch, registry_bytes=_registry_bytes(),
                                   tier4_refs=(), key="miss")
    assert case.run(stage) == "failed"
    failure = case.failure(stage)
    assert "TIER4_CACHE_MISSING" in failure
    # No attempt output exists: the recording stub (which stands in for the
    # real legacy_score worker) never ran, because prepare_launch refused
    # before the worker was ever spawned.
    assert case.recorded() == []
    (attempt,) = case.attempts(stage)
    assert case.conn.execute("SELECT COUNT(*) FROM attempt_outputs WHERE attempt_id=?",
                             (attempt,)).fetchone()[0] == 0


def test_missing_cache_lists_every_reason(case, monkeypatch):
    """A registered champion whose pinned cache is for the WRONG panel hash
    -> still refused, and the failure names the triple."""
    panel_sha = _panel_sha(case)
    cache = case.store.publish_bytes(
        _cache_bytes(tier3_snapshot="b" * 64), schema_ref="legacy_pinned_ref.v1")
    ref = lm.format_pinned_ref(f"{TIER4_SERVING_DIR}/size_{FOLD}_{panel_sha[:12]}.joblib", cache.content_hash)
    stage = _materialize_and_score(case, monkeypatch, registry_bytes=_registry_bytes(),
                                   tier4_refs=(ref,), key="badpanel")
    assert case.run(stage) == "failed"
    failure = case.failure(stage)
    assert "TIER4_CACHE_MISSING" in failure
    assert '"reason": "panel_mismatch"' in failure or "panel_mismatch" in failure
    assert case.recorded() == []


def test_no_champions_is_a_noop(case, monkeypatch):
    """No champion has a Tier-4 ``produces`` (the default synthetic registry)
    -> nothing is required, the launch proceeds exactly as before this task."""
    stage = _materialize_and_score(case, monkeypatch, registry_bytes=b'{"models": []}',
                                   tier4_refs=(), key="empty")
    assert case.run(stage) == "succeeded", case.failure(stage)


def test_walked_back_finality_session_moves_the_required_fold(case, monkeypatch):
    """P2-C03 landed after this task started: ``legacy_score`` now resolves
    its real ``as_of`` from a bound ``finality.json`` (which may walk the
    requested session BACK), not from the requested ``session`` directly.
    Event 2026-10-05, requested session 2026-10-01 (fold 202610), finality
    walked back to 2026-09-15 (fold 202609) -- only the RESOLVED fold's cache
    is pinned. The old (pre-fix) behaviour of folding on the requested date
    would refuse this as TIER4_CACHE_MISSING for fold 202610; the fix must
    launch clean instead, because 202609 is what the worker will actually
    ask for."""
    panel_sha = _panel_sha(case)
    cache = case.store.publish_bytes(
        _cache_bytes(fold_start="2026-09-01", tier3_snapshot=panel_sha), schema_ref="legacy_pinned_ref.v1")
    tier4_ref = lm.format_pinned_ref(f"{TIER4_SERVING_DIR}/size_202609_{panel_sha[:12]}.joblib",
                                     cache.content_hash)
    request = _custom_request(case, registry_bytes=_registry_bytes(), tier4_refs=(tier4_ref,))
    request_ref = case.publish(to_document(request), "legacy_materialization_request.v1.0")
    mat = case.submit_materialize("mat-walkback", request_ref=request_ref)
    assert case.run(mat) == "succeeded", case.failure(mat)
    case.install_stub(monkeypatch)

    finality_doc = {"date": "2026-09-15", "is_final": True, "detail": "walked back for test"}
    finality_ref = case.publish(finality_doc, "legacy_action.v1.0")
    bindings = {"snapshot_ref.json": case.snapshot_ref.artifact_id,
               "materialization_request.json": request_ref.artifact_id,
               "materialization_manifest.json": mat + "#" + MANIFEST,
               "finality.json": finality_ref.artifact_id}
    parameters = {"expected_ids": ["legacy_score"], "session": "2026-10-01",
                 "tickers": ["AAA", "BBB"], "year_start": 2020, "year_end": 2026,
                 "expected_population": ["AAA|S1|2026-10-05"], "input_mode": "snapshot",
                 "input_bindings": bindings}
    refs = (case.snapshot_ref.artifact_id, request_ref.artifact_id, finality_ref.artifact_id)
    stage = case.submit("legacy_score", parameters, refs, "score-walkback", "legacy_score",
                        "legacy_action.v1.0", deps=(mat,))
    assert case.run(stage) == "succeeded", case.failure(stage)


# --------------------------------------------------------------------------
# generation binding
# --------------------------------------------------------------------------


def _publish_manifest(case, file_refs):
    document = {"schema_version": "legacy_input_manifest.v1.0", "manifest_id": "m1",
               "file_refs": [{"path": p, "content_hash": h, "byte_size": 1} for p, h in file_refs],
               "table_contract_refs": [], "registry_and_model_refs": [], "calendar_ref": None,
               "selected_session": SESSION, "finality_receipt_refs": [], "knowledge_mode_by_table": {},
               "availability_evidence_refs": [], "read_set_complete": True,
               "capture_implementation_ref": "test.v1"}
    ref = case.store.publish_bytes(json.dumps(document, sort_keys=True).encode(),
                                   schema_ref="legacy_input_manifest.v1.0")
    with transaction(case.conn):
        register_artifact(case.conn, ref, None, case.clock)
    return ref, document


def _accept_generation(case, file_refs, *, receipt_id):
    """A committed import receipt for ``case.snap`` pinning ``file_refs`` as
    its own ``LegacyInputManifest`` -- reuses the materialize job's own
    attempt (a REAL row) as the anchor ``attempt_input_bindings`` needs."""
    manifest_ref, _ = _publish_manifest(case, file_refs)
    anchor = case.submit_materialize(f"anchor-{receipt_id}")
    assert case.run(anchor) == "succeeded", case.failure(anchor)
    attempt_id = case.attempts(anchor)[-1]
    with transaction(case.conn):
        case.conn.execute(
            "INSERT INTO attempt_input_bindings VALUES (?,?,?,?,?)",
            (attempt_id, "legacy_manifest.json", manifest_ref.artifact_id, manifest_ref.artifact_id,
             manifest_ref.content_hash))
        case.conn.execute(
            "INSERT INTO data_import_receipts (receipt_id, attempt_id, fence, source_manifest_hash, "
            "result_snapshot_id, status, registered_at, scope) VALUES (?, ?, 1, ?, ?, 'committed', ?, "
            "'shadow')",
            (receipt_id, attempt_id, "sha256:" + "0" * 64, case.snap.snapshot_id,
             case.clock.now().isoformat().replace("+00:00", "Z")))
    return attempt_id


def test_no_accepted_generation_yet_is_a_noop(case):
    assert accepted_generation_refs(case.conn, case.store, receipt_id="") is None
    assert accepted_generation_refs(case.conn, case.store, receipt_id="never-committed") is None
    refuse_generation_mismatch(case.conn, case.store, receipt_id="",
                               barrier_manifest={"file_refs": []})  # does not raise


def test_matching_manifest_proceeds(case):
    _accept_generation(case, [("data/x.csv", "sha256:" + "1" * 64)], receipt_id="acc-match")
    barrier = {"file_refs": [{"path": "data/x.csv", "content_hash": "sha256:" + "1" * 64}]}
    refuse_generation_mismatch(case.conn, case.store, receipt_id="acc-match",
                               barrier_manifest=barrier)  # does not raise


def test_overlapping_path_with_a_different_hash_refuses(case):
    _accept_generation(case, [("data/x.csv", "sha256:" + "1" * 64)], receipt_id="acc-mismatch")
    barrier = {"file_refs": [{"path": "data/x.csv", "content_hash": "sha256:" + "2" * 64}]}
    with pytest.raises(OpsError) as err:
        refuse_generation_mismatch(case.conn, case.store, receipt_id="acc-mismatch",
                                   barrier_manifest=barrier)
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details["reason"] == "generation_mismatch"
    assert err.value.problem.details["paths"] == ["data/x.csv"]


def test_non_overlapping_extra_paths_are_allowed(case):
    """A barrier stage's own finality-coverage frames -- paths the accepted
    generation never pinned -- are not a disagreement."""
    _accept_generation(case, [("data/x.csv", "sha256:" + "1" * 64)], receipt_id="acc-extra")
    barrier = {"file_refs": [
        {"path": "data/x.csv", "content_hash": "sha256:" + "1" * 64},
        {"path": "data/finality_only.csv", "content_hash": "sha256:" + "9" * 64}]}
    refuse_generation_mismatch(case.conn, case.store, receipt_id="acc-extra",
                               barrier_manifest=barrier)  # does not raise


def test_tampered_or_missing_pinned_receipt_is_refused(case):
    """External review #5, step 4: a receipt_id that does not resolve to a
    committed receipt with a pinned manifest -- deleted, tampered, or simply
    never committed -- refuses INPUT_CHANGED rather than silently proceeding
    as a no-op. Only an EMPTY receipt_id (see
    ``test_no_accepted_generation_yet_is_a_noop``) is a no-op."""
    with pytest.raises(OpsError) as err:
        refuse_generation_mismatch(case.conn, case.store, receipt_id="does-not-exist",
                                   barrier_manifest={"file_refs": []})
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details["reason"] == "generation_receipt_missing"
    assert err.value.problem.details["receipt_id"] == "does-not-exist"


def _claim(kind, *, namespace="shadow", parameters=None, attempt_id="att_test"):
    from engine.v2.contracts import JobSpec
    from engine.v2.contracts.operations import ResolvedResources

    spec = JobSpec(kind=kind, implementation_ref="x", spec_hash=None, environment_ref="x",
                   parameters=parameters or {}, output_namespace=namespace,
                   resource_class="validation", retry_policy_ref="bounded",
                   checkpoint_contract_ref="legacy_action.v1.0")
    resources = ResolvedResources(effective_host_budget_bytes=1, reserved_memory_bytes=1,
                                  assigned_cpu_ids=(), thread_count=1, scratch_limit_bytes=1,
                                  executor_mode="fake", containment="none", provider_leases=(),
                                  resource_profile_version="v1")
    return Claim(job_id="job_test", attempt_id=attempt_id, attempt_number=1, fence=1, spec=spec,
                resources=resources, lease_expires_at="2026-09-13T00:00:00.000000Z")


def _service(case):
    return Service(case.conn, case.root, __import__(
        "engine.v2.ops.stages", fromlist=["registry"]).registry(), None, clock=case.clock,
        code_source=ROOT, store_root=case.live_store)


def _snapshot_mode_params(case, *, bindings, receipt_id):
    """A barrier job's own parameters as ``nightly._stage_parameters`` stamps
    them inside a snapshot-mode plan graph: the plan's pinned
    ``snapshot_id``/``scope``/``snapshot_generation_receipt_id`` (external
    review #5), alongside the ordinary ``legacy_manifest.json`` binding every
    barrier kind has always carried."""
    return {"input_bindings": bindings, "snapshot_generation_id": case.snap.snapshot_id,
           "snapshot_generation_scope": "shadow", "snapshot_generation_receipt_id": receipt_id}


def test_pin_read_set_refuses_a_snapshot_mode_barrier_kind_on_mismatch(case):
    """Wiring proof: ``Service._pin_read_set`` -- the real pre-launch hook,
    not just the helper module in isolation -- refuses a barrier-only kind
    that carries the plan's snapshot marker and whose manifest disagrees
    with the accepted generation, BEFORE ``store_barrier.pin_read_set`` ever
    runs (which would need real files under ``store_root``)."""
    assert "legacy_finality" in BARRIER_ONLY_REASONS
    _accept_generation(case, [("data/x.csv", "sha256:" + "1" * 64)], receipt_id="acc-wired")
    manifest_ref, _ = _publish_manifest(case, [("data/x.csv", "sha256:" + "2" * 64)])
    claim = _claim("legacy_finality", parameters=_snapshot_mode_params(
        case, bindings={"legacy_manifest.json": manifest_ref.artifact_id}, receipt_id="acc-wired"))
    with pytest.raises(OpsError) as err:
        _service(case)._pin_read_set(claim)
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details["reason"] == "generation_mismatch"


def test_pin_read_set_snapshot_mode_barrier_kind_proceeds_on_a_match(case, monkeypatch):
    """The companion positive case: a snapshot-mode barrier job whose
    manifest AGREES with the accepted generation passes the check cleanly.
    ``store_barrier.pin_read_set`` itself is stubbed out -- it would need
    real files under ``store_root`` matching real hashes, which is a
    concern of a different module entirely; this isolates the generation
    check specifically."""
    import engine.v2.ops.supervisor as supervisor_mod

    _accept_generation(case, [("data/x.csv", "sha256:" + "1" * 64)], receipt_id="acc-match-wired")
    manifest_ref, document = _publish_manifest(case, [("data/x.csv", "sha256:" + "1" * 64)])
    monkeypatch.setattr(supervisor_mod, "pin_read_set", lambda *a, **k: None)
    claim = _claim("legacy_finality", parameters=_snapshot_mode_params(
        case, bindings={"legacy_manifest.json": manifest_ref.artifact_id},
        receipt_id="acc-match-wired"))
    result = _service(case)._pin_read_set(claim)  # does not raise
    assert result == document


def test_pin_read_set_snapshot_mode_barrier_job_without_a_pinned_receipt_is_refused(case, monkeypatch):
    """Compatibility (external review #5, step 3): a job planned before this
    fix carries ``snapshot_generation_id`` (it IS a snapshot-mode plan) but
    has no ``snapshot_generation_receipt_id`` at all -- no existing field
    identifies which receipt it was pinned to, since the snapshot id alone
    is exactly the ambiguous value the review flagged. Refused typed with a
    clear re-plan message; never falls back to "latest"."""
    import engine.v2.ops.supervisor as supervisor_mod

    _accept_generation(case, [("data/x.csv", "sha256:" + "1" * 64)], receipt_id="acc-legacy-plan")
    manifest_ref, _ = _publish_manifest(case, [("data/x.csv", "sha256:" + "1" * 64)])
    monkeypatch.setattr(supervisor_mod, "pin_read_set", lambda *a, **k: None)
    claim = _claim("legacy_finality", parameters={
        "input_bindings": {"legacy_manifest.json": manifest_ref.artifact_id},
        "snapshot_generation_id": case.snap.snapshot_id, "snapshot_generation_scope": "shadow"})
        # no snapshot_generation_receipt_id -- the pre-fix shape
    with pytest.raises(OpsError) as err:
        _service(case)._pin_read_set(claim)
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details["reason"] == "generation_not_pinned"


def test_reviewer_finding5_repro_old_job_survives_newer_reference_only_import(case, monkeypatch):
    """External review finding #5, faithfully reproduced: import A commits;
    a job is planned pinned to A's generation (``pin_snapshot_inputs``
    stamps the EXACT receipt id it resolved, per the fix); a later
    reference-only reimport (B) commits a NEW receipt against the SAME
    snapshot id with a DIFFERENT model file. The old job's barrier stage --
    whose own read of the legacy tree has not changed since capture -- must
    still validate cleanly against A, never against B.

    Before the fix (``_snapshot_mode_params`` stamping only
    ``snapshot_generation_id``/``scope``, the pre-fix shape), this same
    sequence raises INPUT_CHANGED/generation_mismatch: the launch-time check
    re-resolved "latest committed receipt for this snapshot id", which is B
    by the time this barrier stage launches -- an unrelated, later import
    retroactively invalidating an unchanged, already-planned job."""
    import engine.v2.ops.supervisor as supervisor_mod

    _accept_generation(case, [("engine/models/registry.json", "sha256:" + "1" * 64)],
                       receipt_id="gen-A")
    # The old job's own barrier read, captured before B ever existed -- and
    # unchanged since (this is the point: the barrier's read set is fine).
    manifest_ref, document = _publish_manifest(
        case, [("engine/models/registry.json", "sha256:" + "1" * 64)])
    old_job_params = _snapshot_mode_params(
        case, bindings={"legacy_manifest.json": manifest_ref.artifact_id}, receipt_id="gen-A")
    # A later reference-only reimport of the SAME snapshot pins a DIFFERENT
    # model file. It must not retroactively invalidate the already-planned job.
    _accept_generation(case, [("engine/models/registry.json", "sha256:" + "2" * 64)],
                       receipt_id="gen-B")
    monkeypatch.setattr(supervisor_mod, "pin_read_set", lambda *a, **k: None)
    claim = _claim("legacy_finality", parameters=old_job_params)
    result = _service(case)._pin_read_set(claim)  # must not raise
    assert result == document


def test_a_new_plan_after_a_reference_only_reimport_pins_the_newer_receipt(case):
    """The one place "latest" is still correct: PLANNING a NEW job after B
    commits must pick up B, not A -- ``committed_receipt_for_snapshot`` is
    plan-time-only resolution, and ``pin_snapshot_inputs`` calls it exactly
    once per plan."""
    from engine.v2.data.reference_catalog import committed_receipt_for_snapshot

    _accept_generation(case, [("data/x.csv", "sha256:" + "1" * 64)], receipt_id="new-plan-A")
    _accept_generation(case, [("data/x.csv", "sha256:" + "2" * 64)], receipt_id="new-plan-B")
    resolved = committed_receipt_for_snapshot(case.conn, scope="shadow",
                                              snapshot_id=case.snap.snapshot_id)
    assert resolved == "new-plan-B"


def test_pin_read_set_legacy_mode_barrier_job_ignores_a_mismatching_shadow_snapshot(case, monkeypatch):
    """The review fix itself: a LEGACY-mode barrier job (no
    ``snapshot_generation_id`` -- the default plan shape) must proceed even
    though a committed shadow import exists whose manifest disagrees with
    this job's own read set. Before the fix, comparing against "newest
    committed in scope" would have refused this on every ordinary nightly
    the moment any shadow snapshot existed, regardless of this run's own
    input mode. Covers both directions named in review: the committed
    snapshot can be NEWER or OLDER than what the barrier job itself reads --
    the fix does not look at the committed generation at all when
    ungated, so direction cannot matter; this asserts it for a newer one."""
    import engine.v2.ops.supervisor as supervisor_mod

    _accept_generation(case, [("data/x.csv", "sha256:" + "1" * 64)], receipt_id="acc-legacy-newer")
    manifest_ref, document = _publish_manifest(case, [("data/x.csv", "sha256:" + "2" * 64)])
    monkeypatch.setattr(supervisor_mod, "pin_read_set", lambda *a, **k: None)
    claim = _claim("legacy_finality", parameters={
        "input_bindings": {"legacy_manifest.json": manifest_ref.artifact_id}})  # no snapshot marker
    result = _service(case)._pin_read_set(claim)  # does not raise despite the mismatch above
    assert result == document


def test_pin_read_set_leaves_non_barrier_kinds_unchecked(case, monkeypatch):
    """A non-barrier ``legacy_*`` kind (``legacy_decisions``: has a declared
    store read domain but is not in ``BARRIER_ONLY_REASONS``) never reaches
    the generation-binding check at all -- it is bound through the
    snapshot-backed materialization request instead (task 6b) when it runs
    snapshot-backed, and through the unchanged Phase-1 read set otherwise."""
    import engine.v2.ops.supervisor as supervisor_mod

    calls = []
    monkeypatch.setattr(supervisor_mod, "refuse_generation_mismatch",
                        lambda *a, **k: calls.append(a))
    manifest_ref, _ = _publish_manifest(case, [("data/x.csv", "sha256:" + "2" * 64)])
    claim = _claim("legacy_decisions", parameters=_snapshot_mode_params(
        case, bindings={"legacy_manifest.json": manifest_ref.artifact_id}, receipt_id="irrelevant"))
    try:
        _service(case)._pin_read_set(claim)
    except OpsError:
        pass  # pin_read_set's own file check has nothing real to verify here
    assert calls == []
