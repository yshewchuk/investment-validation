"""Focused regression tests for A2/A3/A5/A6/A7 (see the phase-1 defect brief).

A1 and A4 have their own end-to-end coverage in
``tests/test_v2_ops_supervised_legacy.py``; this file targets the pieces that
do not need a real subprocess.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from engine.v2.foundation import ArtifactStore, SystemClock, content_hash
from engine.v2.ops import cli
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import environment_identity
from engine.v2.ops.legacy_adapter import _action_score, _load_finality
from engine.v2.ops.nightly import build_legacy_job_requests, build_nightly_plan
from engine.v2.ops.plans import nightly_plan
from engine.v2.ops.profiles import DEFAULT_POLICY, GIB, POLICY_VERSION, profile_named
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, submit

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# A2: the planned population is keyed before strike/expiry exist
# --------------------------------------------------------------------------


class _FakeScorer:
    def __init__(self, context):
        self.analog_entry_coverage = 1.0


def _stub_scoring(monkeypatch, rows):
    import engine.dashboard.nightly as nightly_module
    import engine.features as features_module
    import engine.score as score_module

    monkeypatch.setattr(score_module, "Scorer", _FakeScorer)
    monkeypatch.setattr(features_module.FeatureContext, "load",
                        staticmethod(lambda tickers, years: object()))
    monkeypatch.setattr(score_module, "score_calendar",
                        lambda *a, **k: pd.DataFrame(rows))
    monkeypatch.setattr(nightly_module, "strike_ladder", lambda *a, **k: [])


def _score_parameters(**overrides):
    parameters = {"tickers": ["FAKE"], "year_start": 2024, "year_end": 2026,
                 "session": "2026-09-12", "horizon_days": 35, "alt_strikes": 1}
    parameters.update(overrides)
    return parameters


def _write_finality(root, date="2026-09-12"):
    """P2-C03: every legacy stage that resolves the finality session now
    reads its bound ``finality.json`` — no walk-back, matching ``session``."""
    (root / "finality.json").write_text(json.dumps({
        "date": date, "is_final": True, "market_wide": True,
        "daily_share": 1.0, "chain_share": 1.0, "covered": 1, "detail": "final"}))


def test_action_score_accepts_a_population_planned_before_strikes_exist(monkeypatch, tmp_path):
    rows = [{"ticker": "FAKE", "strategy": "TWIN-P", "event_date": "2026-09-12",
            "strike": 100.0, "expiry": "2026-10-16"},
           {"ticker": "FAKE", "strategy": "TWIN-P", "event_date": "2026-09-12",
            "strike": 105.0, "expiry": "2026-10-16"}]
    _stub_scoring(monkeypatch, rows)
    _write_finality(tmp_path)
    result = _action_score(_score_parameters(
        expected_population=("FAKE|TWIN-P|2026-09-12",)), tmp_path)
    assert result["hash"]
    document = json.loads((tmp_path / "score.json").read_text())
    assert document["observed_population"] == ["FAKE|TWIN-P|2026-09-12"]
    assert len(document["rows"]) == 2  # both alt-strike rows share one population key


def test_action_score_rejects_missing_duplicate_and_unplanned_keys(monkeypatch, tmp_path):
    rows = [{"ticker": "FAKE", "strategy": "TWIN-P", "event_date": "2026-09-12"}]
    _stub_scoring(monkeypatch, rows)
    _write_finality(tmp_path)
    with pytest.raises(OpsError, match="must be planned"):
        _action_score(_score_parameters(expected_population=()), tmp_path)
    with pytest.raises(OpsError, match="duplicate"):
        _action_score(_score_parameters(
            expected_population=("FAKE|TWIN-P|2026-09-12", "FAKE|TWIN-P|2026-09-12")), tmp_path)
    with pytest.raises(OpsError, match="differs from planned"):
        _action_score(_score_parameters(
            expected_population=("OTHER|TWIN-P|2026-09-12",)), tmp_path)


def test_load_finality_requires_the_upstream_artifact(tmp_path):
    with pytest.raises(OpsError, match="INPUT_CHANGED"):
        _load_finality(tmp_path)
    (tmp_path / "finality.json").write_text('{"is_final": true}')
    assert _load_finality(tmp_path) == {"is_final": True}


def test_decision_commit_depends_on_both_score_and_finality():
    plan = build_nightly_plan(str(REPO), "2026-09-12")
    requests = build_legacy_job_requests(plan, tickers=("FAKE",),
                                         year_start=2025, year_end=2026)
    by_kind = {r.job.kind: r for r in requests}
    decisions = by_kind["legacy_decisions"]
    finality_key = by_kind["legacy_finality"].idempotency_key
    score_key = by_kind["legacy_score"].idempotency_key
    from engine.v2.ops.submission import job_id_for
    assert job_id_for("shadow", finality_key) in decisions.job.dependency_job_ids
    assert job_id_for("shadow", score_key) in decisions.job.dependency_job_ids
    assert "finality.json" in decisions.job.parameters["input_bindings"]


def test_nightly_plan_population_is_required_before_submit_and_hashed():
    unplanned = nightly_plan(str(REPO), "2026-09-12", manifest_ref="art_x")
    assert "planned_population" in unplanned["blocked_prerequisites"]
    planned = nightly_plan(str(REPO), "2026-09-12", manifest_ref="art_x",
                           expected_population=("FAKE|TWIN-P|2026-09-12",))
    assert planned["blocked_prerequisites"] == []
    assert planned["expected_population"] == ["FAKE|TWIN-P|2026-09-12"]
    other = nightly_plan(str(REPO), "2026-09-12", manifest_ref="art_x",
                         expected_population=("OTHER|TWIN-P|2026-09-12",))
    assert planned["plan_hash"] != other["plan_hash"]


def test_cli_submit_refuses_a_nightly_plan_with_blocked_prerequisites(tmp_path, capsys):
    root = tmp_path / "ops"
    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(root), "plan", "nightly", "--as-of", "2026-09-12"]) == 0
    plan_ref = json.loads(capsys.readouterr().out)["plan_ref"]
    code = cli.main(["--root", str(root), "submit", "--plan", plan_ref,
                     "--idempotency-key", "k1"])
    out = json.loads(capsys.readouterr().out)
    assert code == 2
    assert out["code"] == "INVALID_REQUEST"


def test_cli_plan_nightly_accepts_expected_population_file(tmp_path, capsys):
    root = tmp_path / "ops"
    population_file = tmp_path / "population.json"
    population_file.write_text(json.dumps(["FAKE|TWIN-P|2026-09-12"]))
    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(root), "plan", "nightly", "--as-of", "2026-09-12",
                     "--expected-population", str(population_file)]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["plan"]["expected_population"] == ["FAKE|TWIN-P|2026-09-12"]
    assert "planned_population" not in document["plan"]["blocked_prerequisites"]


# --------------------------------------------------------------------------
# A3: the environment ref every emitted request carries must equal what
# ``_launch`` computes from the same resource class.
# --------------------------------------------------------------------------


def test_every_emitted_request_environment_ref_matches_launch_formula():
    from engine.v2.ops.nightly import _legacy_resource

    plan = build_nightly_plan(str(REPO), "2026-09-12")
    requests = build_legacy_job_requests(plan, tickers=("FAKE",),
                                         year_start=2025, year_end=2026,
                                         include_prerequisites=False)
    assert len(requests) == 13
    for request in requests:
        profile = profile_named(DEFAULT_POLICY, _legacy_resource(request.job.kind))
        thread_count = profile.thread_count or profile.cpu_count
        expected_ref = content_hash(environment_identity(thread_count))
        assert request.job.environment_ref == expected_ref, request.job.kind


def test_legacy_render_previously_carried_the_wrong_thread_count():
    """The projection profile has 2 CPUs; the old hard-coded map used 4."""
    profile = profile_named(DEFAULT_POLICY, "projection")
    assert profile.cpu_count == 2
    plan = build_nightly_plan(str(REPO), "2026-09-12")
    requests = build_legacy_job_requests(plan, tickers=("FAKE",),
                                         year_start=2025, year_end=2026,
                                         include_prerequisites=False)
    render = next(r for r in requests if r.job.kind == "legacy_render")
    assert render.job.environment_ref == content_hash(environment_identity(profile.cpu_count))


# --------------------------------------------------------------------------
# A5: the measured legacy-scoring peak needs a bigger reservation.
# --------------------------------------------------------------------------


def test_legacy_score_and_validation_reservations_cover_the_measured_peak():
    measured_peak_bytes = int(4.15 * (1 << 30))
    for name in ("legacy_score", "validation"):
        profile = profile_named(DEFAULT_POLICY, name)
        assert profile.memory_bytes >= measured_peak_bytes
        assert profile.measured is False
    assert POLICY_VERSION == DEFAULT_POLICY.version
    assert "2026-09-14" in POLICY_VERSION


# --------------------------------------------------------------------------
# 2026-09-14 (v4): every profile a read-set-staging kind can carry admits the
# real 1.67 GiB read set a 4-ticker snapshot nightly measured
# (``needed_bytes=1796916876`` against the old 1 GiB ``validation`` scratch
# limit -- see ``profiles.py``'s module docstring).
# --------------------------------------------------------------------------

#: The exact real failure's needed_bytes (2026-09-14 4-ticker snapshot nightly).
MEASURED_READ_SET_BYTES = 1796916876


def test_policy_version_is_v4():
    assert POLICY_VERSION == "ops_resources.2026-09-14.v4"
    assert POLICY_VERSION == DEFAULT_POLICY.version


def test_scratch_admits_the_measured_read_set_for_every_staging_profile():
    """Deliverable 1: finality/decisions/selfcheck (``validation``),
    legacy_score/legacy_decision_replay (``legacy_score``),
    legacy_model_evidence (``model_evidence``) and legacy_render
    (``projection``) all now admit the measured 1.67 GiB read set with
    margin, and memory is untouched from v3."""
    for name, old_memory_gib in (("validation", 5), ("legacy_score", 5),
                                 ("model_evidence", 4), ("projection", 2)):
        profile = profile_named(DEFAULT_POLICY, name)
        assert profile.scratch_bytes >= MEASURED_READ_SET_BYTES
        assert profile.scratch_bytes == 4 * GIB
        assert profile.memory_bytes == old_memory_gib * GIB  # unchanged by v4


# --------------------------------------------------------------------------
# 2026-09-14 right-sizing: legacy_materialize gets its own small profile,
# and DEFAULT_POLICY stays structurally valid.
# --------------------------------------------------------------------------


_MATERIALIZE_SNAPSHOT_INPUTS = {"snapshot_ref_artifact_id": "art_snapshot",
                                "materialization_request_ref": "art_request",
                                "scratch_estimate_bytes": 1024}


def test_planned_legacy_materialize_job_gets_the_materialize_resource_class():
    plan = build_nightly_plan(str(REPO), "2026-09-14")
    requests = build_legacy_job_requests(plan, tickers=("AAA",), year_start=2025, year_end=2026,
                                         input_refs=("art_m",), input_mode="snapshot",
                                         snapshot_inputs=_MATERIALIZE_SNAPSHOT_INPUTS)
    by_kind = {r.job.kind: r for r in requests}
    assert by_kind["legacy_materialize"].job.resource_class == "materialize"
    # And the profile itself resolves and is small relative to legacy_rebuild.
    profile = profile_named(DEFAULT_POLICY, "materialize")
    assert profile.memory_bytes == 2 * (1 << 30)
    assert profile.memory_bytes < profile_named(DEFAULT_POLICY, "legacy_rebuild").memory_bytes


def test_default_policy_has_no_structural_problems():
    from engine.v2.ops.profiles import policy_problems
    assert policy_problems(DEFAULT_POLICY) == []


# --------------------------------------------------------------------------
# A6: the canary adapter runner submits a real job instead of calling the
# legacy adapter in-process. Only request construction is tested here — the
# real run would load the real scorer.
# --------------------------------------------------------------------------


def _prepared_root(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    payload = (root / "unused.txt")
    payload.write_bytes(b"prepared canary input")
    manifest = {"schema_version": "phase1_canary_inputs.v1.0", "root": str(root),
               "files": [{"artifact_id": "sha256:" + hashlib.sha256(payload.read_bytes()).hexdigest(),
                          "path": "unused.txt", "bytes": payload.stat().st_size}],
               "file_count": 1}
    (root / "INPUT_MANIFEST.json").write_text(json.dumps(manifest))
    (root / "score_requests.json").write_text(json.dumps(
        [{"canary_id": "c1", "request": {"ticker": "FAKE"}}]))
    return root


def test_canary_adapter_builds_a_kind_registered_request_accepted_by_submit(tmp_path):
    from checks.rearchitecture_phase1_adapter import build_manifest, build_request

    root = _prepared_root(tmp_path)
    manifest_document = build_manifest(root)
    assert manifest_document["read_set_complete"] is True
    paths = {ref["path"] for ref in manifest_document["file_refs"]}
    assert paths == {"unused.txt", "score_requests.json"}

    store = ArtifactStore(tmp_path / "catalog")
    manifest_ref = store.publish_bytes(json.dumps(manifest_document, sort_keys=True).encode(),
                                       schema_ref="legacy_input_manifest.v1.0")
    request = build_request(manifest_ref.artifact_id)
    assert request.job.kind == "legacy_score_requests"
    assert request.job.parameters["requests_path"] == "legacy/score_requests.json"
    assert "legacy_score_requests" in registry().names()

    clock = SystemClock()
    conn = open_catalog(tmp_path / "catalog" / "ops.sqlite", clock=clock)
    try:
        from engine.v2.ops.catalog import transaction
        from engine.v2.ops.checkpoints import register_artifact
        with transaction(conn):
            register_artifact(conn, manifest_ref, None, clock)
        policy = NamespacePolicy({"operator": frozenset({"shadow", "smoke"})})
        receipt = submit(conn, registry(), policy, request, clock=clock)
        assert receipt.state == "queued"
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 2026-09-14 (A1 follow-up): a plan-time scratch check, so a job whose legacy
# read set exceeds its resource profile's scratch budget is refused before
# submission, never only discovered later at claim time.
# --------------------------------------------------------------------------


def _manifest_ref(store, conn, clock, byte_sizes):
    """Publish and register a ``legacy_input_manifest.v1.0`` whose
    ``file_refs`` sum to exactly ``sum(byte_sizes)`` -- the same total
    ``_populate_legacy_staging``/``plan_scratch_problems`` compute."""
    from engine.v2.ops.catalog import transaction
    from engine.v2.ops.checkpoints import register_artifact

    document = {"schema_version": "legacy_input_manifest.v1.0", "manifest_id": "m1",
               "file_refs": [{"path": f"f{i}.bin", "content_hash": "sha256:" + str(i) * 64,
                              "byte_size": size} for i, size in enumerate(byte_sizes)],
               "table_contract_refs": [], "registry_and_model_refs": [], "calendar_ref": None,
               "selected_session": "2026-09-14", "finality_receipt_refs": [],
               "knowledge_mode_by_table": {}, "availability_evidence_refs": [],
               "read_set_complete": True, "capture_implementation_ref": "test.v1"}
    ref = store.publish_bytes(json.dumps(document, sort_keys=True).encode(),
                              schema_ref="legacy_input_manifest.v1.0")
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return ref


def test_plan_scratch_problems_admits_the_measured_1_7gib_read_set(tmp_path):
    """Deliverable 3 (part 2): the current v4 profiles admit exactly the real
    1.67 GiB read set for every kind that stages it -- finality, decisions,
    selfcheck, score, decision_replay, model_evidence and render alike."""
    from engine.v2.ops.nightly import plan_scratch_problems

    (tmp_path / "catalog").mkdir()
    store = ArtifactStore(tmp_path / "catalog")
    clock = SystemClock()
    conn = open_catalog(tmp_path / "catalog" / "ops.sqlite", clock=clock)
    try:
        manifest_ref = _manifest_ref(store, conn, clock, [MEASURED_READ_SET_BYTES])
        plan = build_nightly_plan(str(REPO), "2026-09-14")
        requests = build_legacy_job_requests(
            plan, tickers=("AAA", "BBB", "CCC", "DDD"), year_start=2025, year_end=2026,
            input_refs=(manifest_ref.artifact_id,))
        by_kind = {r.job.kind for r in requests}
        assert {"legacy_finality", "legacy_decisions", "legacy_selfcheck", "legacy_score",
               "legacy_decision_replay", "legacy_model_evidence", "legacy_render"} <= by_kind
        assert plan_scratch_problems(conn, store, requests) == []
    finally:
        conn.close()


def test_refuse_oversize_plan_blocks_submission_with_details(tmp_path):
    """Deliverable 2: a plan whose declared read set exceeds a profile's
    scratch is refused at plan time, with the offending kind/profile/needed/
    limit -- never only discovered later at claim time."""
    from engine.v2.ops.nightly import refuse_oversize_plan

    (tmp_path / "catalog").mkdir()
    store = ArtifactStore(tmp_path / "catalog")
    clock = SystemClock()
    conn = open_catalog(tmp_path / "catalog" / "ops.sqlite", clock=clock)
    try:
        oversize = profile_named(DEFAULT_POLICY, "validation").scratch_bytes + 1
        manifest_ref = _manifest_ref(store, conn, clock, [oversize])
        plan = build_nightly_plan(str(REPO), "2026-09-14")
        requests = build_legacy_job_requests(plan, tickers=("FAKE",), year_start=2025, year_end=2026,
                                             input_refs=(manifest_ref.artifact_id,))
        with pytest.raises(OpsError) as err:
            refuse_oversize_plan(conn, store, requests)
        assert err.value.code == "RESOURCE_LIMIT_EXCEEDED"
        jobs = {job["kind"]: job for job in err.value.problem.details["jobs"]}
        assert jobs["legacy_finality"] == {
            "kind": "legacy_finality", "profile": "validation",
            "needed_bytes": oversize,
            "scratch_bytes": profile_named(DEFAULT_POLICY, "validation").scratch_bytes}
        # A kind whose profile (legacy_rebuild, 20 GiB) already clears this
        # oversize manifest is never reported.
        assert "legacy_settlement" not in jobs
    finally:
        conn.close()


def test_populate_legacy_staging_still_refuses_an_oversize_read_set_at_claim_time():
    """Deliverable 3 (part 3, defence in depth): even with the plan-time
    check above, the original claim-time guard in
    ``supervisor.Service._populate_legacy_staging`` still refuses on its
    own -- a plan-time bypass or a stale plan built before a profile shrank
    must not let an oversize read set reach staging."""
    from engine.v2.contracts import JobSpec
    from engine.v2.contracts.operations import ResolvedResources
    from engine.v2.ops.scheduler import Claim
    from engine.v2.ops.supervisor import Service

    spec = JobSpec(kind="legacy_finality", implementation_ref="x", spec_hash=None,
                   environment_ref="x", parameters={}, output_namespace="shadow",
                   resource_class="validation", retry_policy_ref="bounded",
                   checkpoint_contract_ref="legacy_action.v1.0")
    resources = ResolvedResources(effective_host_budget_bytes=1, reserved_memory_bytes=1,
                                  assigned_cpu_ids=(), thread_count=1, scratch_limit_bytes=1,
                                  executor_mode="fake", containment="none", provider_leases=(),
                                  resource_profile_version="v1")
    claim = Claim(job_id="job_test", attempt_id="att_test", attempt_number=1, fence=1, spec=spec,
                 resources=resources, lease_expires_at="2026-09-14T00:00:00.000000Z")
    manifest = {"file_refs": [{"path": "f.bin", "byte_size": 2}]}  # 2 bytes > scratch_limit_bytes=1
    service = Service.__new__(Service)  # no real conn/store setup needed: this raises first
    with pytest.raises(OpsError) as err:
        Service._populate_legacy_staging(service, claim, manifest)
    assert err.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert err.value.problem.details == {"needed_bytes": 2, "scratch_limit_bytes": 1}
