"""Phase 3 guide §5.5 items 2-3: live engineering health and semantic status.

Item 2 -- ``engineering_gate_effect`` now records one durable engineering
observation per scheduled occurrence via ``health.record_check``: a retry or
a later same-night generation updates that occurrence's own row (bumping its
retry count) without ever adding a second night, and a night nobody ever
observed reads back as unknown, never green
(``tests/test_v2_ops_generation_effects.py``'s own
``test_health_streak_counts_one_occurrence_per_night_not_per_generation``
already proves the underlying table collapses correctly; this file proves
the effect actually WRITES into it, and the engineering-history window built
on top).

Item 3 -- ``publication_effect`` now writes a versioned
``operations_status.json`` sidecar (``engine.v2.contracts.OperationsStatus``)
into the fenced publisher's own scope root on every attempt, success or
failure, carrying the pinned requested/resolved session, conflicts and
degraded model evidence read back out of the render bundle's own
``data/flags.json`` (P2-C08, never reconstructed), the bound selfcheck, and
-- on a failed update -- the old release's continued availability plus the
failure's own reason.

Real SQLite/ArtifactStore/submission machinery throughout, the same
technique ``tests/test_v2_ops_effects_graph.py`` and
``tests/test_v2_ops_generation_effects.py`` use; ``publication_effect`` is
called directly against a real claim, exactly as
``supervisor.Service._coordinator_effect`` calls it.
"""
from __future__ import annotations

import io
import json
import tarfile

import pytest

from engine.v2.contracts import EngineeringNight, JobSpec, SubmitRequest
from engine.v2.ops import effects_graph
from engine.v2.ops.effects_graph import engineering_gate_effect, publication_effect
from engine.v2.ops.errors import OpsError, make_problem
from engine.v2.ops.health import (
    budget_streak,
    engineering_history,
    engineering_streak_from_history,
    record_check,
    trailing_occurrences,
)
from engine.v2.ops.input_bindings import resolve_and_record
from engine.v2.ops.lifecycle import Outcome, commit_attempt, request_cancel
from engine.v2.ops.publication import current as release_current
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.catalog import dumps as _dumps
from engine.v2.ops.submission import NamespacePolicy, job_id_for, submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY, catalog, sample
from tests.test_v2_ops_effects_graph import (
    FAKE_STORE_ROOT,
    POLICY,
    REPO,
    DEFAULT_POLICY,
    _params,
    _publish,
    _publish_raw,
    _seed_decisions,
    _submit_and_claim,
    _succeed_parent,
)
from tests.test_v2_ops_effects_graph import _commit as _ops_commit
from tests.test_v2_ops_effects_graph import _open as _ops_open
from tests.test_v2_ops_effects_graph import _row as _decision_row

SCOPE = "shadow"
SESSION = "2026-09-12"

#: What checks/rearchitecture_phase1_gate.py prints, reduced to one row per
#: section. The real gate needs a git checkout (``git ls-files``) and
#: /usr/bin/python3 with the repo's dependencies. Mutation CI runs these tests
#: in a work copy that has neither. These tests cover how the effect records
#: its receipts and history, not what the gate checks.
#: test_v2_ops_effects_graph runs the real gate.
GATE_OUTPUT = {"structural": {"layers": {"ok": True}},
               "engineering": {"budgets": {"ok": True}, "coverage": {"ok": False}},
               "code_hash": "sha256:" + "c" * 64}


@pytest.fixture
def stub_gate(monkeypatch):
    monkeypatch.setattr(effects_graph, "_run_engineering_gate_subprocess",
                        lambda repo_root: json.loads(json.dumps(GATE_OUTPUT)))


# --------------------------------------------------------------------------
# item 2: engineering_history / record_check -- pure table behaviour
# --------------------------------------------------------------------------


def test_three_nights_pass_retry_unobserved(tmp_path):
    """Three nights: pass, retry-then-pass, unobserved -- history shows 2
    observed, 1 unknown, with the retry-then-pass night's own retry count
    at 1; ``budget_streak`` still reports the current streak (unaffected by
    the unrelated unobserved night)."""
    conn, clock, _ = catalog(tmp_path)
    night_pass, night_retry, night_unobserved = "2026-09-10", "2026-09-11", "2026-09-12"
    record_check(conn, night_pass, "engineering", True, {"attempt": 1})
    record_check(conn, night_retry, "engineering", False, {"attempt": 1})
    record_check(conn, night_retry, "engineering", True, {"attempt": 2})

    history = engineering_history(conn, [night_pass, night_retry, night_unobserved])
    by_night = {row["occurrence"]: row for row in history}
    assert by_night[night_pass]["status"] == "pass"
    assert by_night[night_pass]["retry_count"] == 0
    assert by_night[night_retry]["status"] == "pass"
    assert by_night[night_retry]["retry_count"] == 1
    assert by_night[night_retry]["detail"] == {"attempt": 2}
    assert by_night[night_unobserved]["status"] == "unknown"
    assert by_night[night_unobserved]["retry_count"] == 0
    assert by_night[night_unobserved]["detail"] is None

    observed = [row for row in history if row["status"] != "unknown"]
    unknown = [row for row in history if row["status"] == "unknown"]
    assert len(observed) == 2
    assert len(unknown) == 1
    conn.close()


def test_trailing_occurrences_window_is_trading_sessions_not_calendar_days():
    """2026-09-14 review fix: SESSION (2026-09-12) is a Saturday, so the
    window -- SCHEDULED TRADING sessions, never calendar days -- ends at the
    last real trading day before it (Friday 2026-09-11); a wider window
    also skips the Labor Day holiday (Monday 2026-09-07), never counting a
    night nobody was ever scheduled to observe."""
    window = trailing_occurrences(SESSION, nights=3)
    assert window == ("2026-09-09", "2026-09-10", "2026-09-11")
    wider = trailing_occurrences(SESSION, nights=5)
    assert wider == ("2026-09-04", "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11")


def test_trailing_occurrences_refuses_rather_than_falls_back_silently(monkeypatch):
    """A broken trading calendar must REFUSE, never silently degrade to
    calendar days -- guide §5.5 item 2's own instruction, 2026-09-14 review
    fix."""
    import engine.v2.ops.legacy_adapter as legacy_adapter

    def _broken(start, end):
        raise RuntimeError("calendar source unavailable")

    monkeypatch.setattr(legacy_adapter, "projected_trading_sessions", _broken)
    with pytest.raises(OpsError, match="VALIDATION_FAILED"):
        trailing_occurrences(SESSION, nights=3)


def test_engineering_gate_effect_records_one_observation_per_night(tmp_path, stub_gate):
    """The wiring guide §5.5 item 2 actually adds: a real
    ``engineering_gate_effect`` call now populates ``health_observations``,
    keyed by the job's own (requested) session -- never only the
    per-generation watermark."""
    conn, clock, supervisor, store, root = _ops_open(tmp_path)
    try:
        claim = _submit_and_claim(conn, clock, supervisor, kind="engineering_gate", key="gate-1",
                                  parameters=_params("engineering_gate", SESSION, SCOPE))
        result = engineering_gate_effect(conn, store, claim, REPO, clock=clock)
        _ops_commit(conn, clock, claim, result)

        history = engineering_history(conn, [SESSION])
        assert history[0]["occurrence"] == SESSION
        assert history[0]["status"] in ("pass", "fail")
        assert history[0]["retry_count"] == 0
        streak = budget_streak(conn)
        assert streak["latest"]["scope"] == SCOPE
    finally:
        conn.close()


def test_engineering_gate_effect_two_generations_in_one_night_count_one_night(tmp_path, stub_gate):
    """Two DISTINCT generations of the SAME scheduled night (different
    ``deployment``/``decision_clock`` -- guide §5.5 item 1's own pinning)
    both record an observation, but ``health_observations`` still holds
    exactly one row for that occurrence -- one retry, never two nights."""
    conn, clock, supervisor, store, root = _ops_open(tmp_path)
    try:
        gen1 = _submit_and_claim(
            conn, clock, supervisor, kind="engineering_gate", key="gate-gen1",
            parameters=_params("engineering_gate", SESSION, SCOPE, deployment="shadow:impl-1",
                               decision_clock="2026-09-12T01:00:00.000000Z"))
        result1 = engineering_gate_effect(conn, store, gen1, REPO, clock=clock)
        _ops_commit(conn, clock, gen1, result1)

        gen2 = _submit_and_claim(
            conn, clock, supervisor, kind="engineering_gate", key="gate-gen2",
            parameters=_params("engineering_gate", SESSION, SCOPE, deployment="shadow:impl-2",
                               decision_clock="2026-09-12T05:00:00.000000Z"))
        result2 = engineering_gate_effect(conn, store, gen2, REPO, clock=clock)
        _ops_commit(conn, clock, gen2, result2)

        assert conn.execute("SELECT COUNT(*) FROM health_observations WHERE kind='engineering'"
                            ).fetchone()[0] == 1
        row = conn.execute("SELECT attempts, receipt_json FROM health_observations "
                           "WHERE occurrence=? AND kind='engineering'", (SESSION,)).fetchone()
        assert row["attempts"] == 1
        assert json.loads(row["receipt_json"])["generation"] != ""
        history = engineering_history(conn, [SESSION])
        assert len(history) == 1
        assert history[0]["retry_count"] == 1
    finally:
        conn.close()


# --------------------------------------------------------------------------
# item 3: the operations_status.json sidecar publication_effect writes
# --------------------------------------------------------------------------


def _bundle_with_flags(flags):
    """A minimal render bundle tar carrying a real ``data/flags.json`` (the
    same relative layout ``dashboard/render.py``'s ``render_bundle`` writes)
    -- so ``_bundle_flags`` reads a REAL P2-C08 flag list rather than an
    empty one."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        board = b"hello board\n"
        info = tarfile.TarInfo("bundle/board.html")
        info.size = len(board)
        archive.addfile(info, io.BytesIO(board))
        payload = json.dumps({"as_of": SESSION, "flags": list(flags)}).encode()
        info = tarfile.TarInfo("bundle/data/flags.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _setup_publication(conn, clock, supervisor, store, *, session, deployment, decision_clock,
                       bundle_bytes, selfcheck_ok=True, engineering_ok=True,
                       finality_session=None):
    """A publication job with real, resolvable bindings -- ``_publication_
    setup`` (``tests/test_v2_ops_effects_graph.py``) with two additions this
    file needs: a caller-supplied bundle (so its ``data/flags.json`` is
    real) and an explicit generation pin (``deployment``/``decision_clock``)
    so two calls for the SAME session mint two DISTINCT releases.

    Decisions are seeded at the EFFECTIVE (finality-resolved) session, same
    as ``_publication_setup``'s own ``decisions_session`` -- a genuine
    walk-back night's decisions watermark lives at the resolved date, never
    the merely-requested one, so ``_decision_gate`` still passes.
    """
    tag = SCOPE + ":" + session + ":" + deployment
    _seed_decisions(conn, clock, SCOPE, finality_session or session,
                    predictions=[_decision_row("evt-1", "pred")])
    finality_doc = {"date": finality_session or session, "is_final": True, "market_wide": True,
                    "daily_share": 1.0, "chain_share": 1.0, "covered": 1, "detail": "final"}
    finality_ref = _publish(store, conn, clock, finality_doc, "legacy_action.v1.0")
    finality_job = _succeed_parent(conn, clock, supervisor, key="fin-" + tag,
                                   output_name="legacy_finality", ref=finality_ref)
    bundle_ref = _publish_raw(store, conn, clock, bundle_bytes, "legacy_action.v1.0")
    projection_job = _succeed_parent(conn, clock, supervisor, key="proj-" + tag,
                                     output_name="legacy_render", ref=bundle_ref)
    selfcheck_ref = _publish(store, conn, clock, {"ok": selfcheck_ok}, "legacy_action.v1.0")
    selfcheck_job = _succeed_parent(conn, clock, supervisor, key="self-" + tag,
                                    output_name="legacy_selfcheck", ref=selfcheck_ref)
    engineering_ref = _publish(store, conn, clock, {"ok": engineering_ok}, "engineering_gate.v1.0")
    engineering_job = _succeed_parent(conn, clock, supervisor, key="eng-" + tag,
                                      output_name="engineering_gate", ref=engineering_ref)
    input_bindings = {"bundle.tar": projection_job + "#legacy_render",
                      "finality.json": finality_job + "#legacy_finality",
                      "selfcheck.json": selfcheck_job + "#legacy_selfcheck",
                      "engineering_gate.json": engineering_job + "#engineering_gate"}
    claim = _submit_and_claim(
        conn, clock, supervisor, kind="publication", key="pub-" + tag,
        parameters=_params("publication", session, SCOPE, input_bindings=input_bindings,
                           deployment=deployment, decision_clock=decision_clock),
        dependency_job_ids=(projection_job, finality_job, selfcheck_job, engineering_job))
    resolve_and_record(conn, store, claim)
    return claim


def test_status_carries_conflicts_degraded_evidence_and_selfcheck(tmp_path):
    """Conflicts and degraded model evidence are read back out of the
    render bundle's own flags, never recomputed; the bound selfcheck
    carries through verbatim -- guide §5.5 item 3."""
    conn, clock, supervisor, store, root = _ops_open(tmp_path)
    try:
        bundle = _bundle_with_flags([
            {"kind": "calendar_date_conflict", "detail": "AAA has two sessions"},
            {"kind": "model_evidence_stale", "detail": "rebuild failed: boom"},
            {"kind": "quota_unknown", "detail": "ignored -- not a conflict or degraded flag"}])
        claim = _setup_publication(conn, clock, supervisor, store, session=SESSION,
                                   deployment="shadow:impl-1", decision_clock="2026-09-12T01:00:00Z",
                                   bundle_bytes=bundle, selfcheck_ok=True)
        result = publication_effect(conn, store, claim, root, REPO, clock=clock, store_root=FAKE_STORE_ROOT)
        _ops_commit(conn, clock, claim, result)

        target = root / "releases" / SCOPE
        document = json.loads((target / "operations_status.json").read_text())
        assert document["schema_version"] == "operations_status.v1.0"
        assert document["conflicts"] == [{"kind": "calendar_date_conflict", "detail": "AAA has two sessions"}]
        assert document["degraded_model_evidence"] == [
            {"kind": "model_evidence_stale", "detail": "rebuild failed: boom"}]
        assert document["selfcheck"] == {"ok": True}
        assert document["failed_update"] is False
        assert document["release_id"] == release_current(target)
    finally:
        conn.close()


def test_status_surfaces_requested_and_resolved_session_verbatim(tmp_path):
    """A genuine walk-back (finality resolves earlier than requested) is
    carried into the status document exactly as ``resolve_effective_
    session`` produced it -- never re-derived."""
    conn, clock, supervisor, store, root = _ops_open(tmp_path)
    try:
        claim = _setup_publication(conn, clock, supervisor, store, session="2026-09-12",
                                   deployment="shadow:impl-1", decision_clock="2026-09-12T01:00:00Z",
                                   bundle_bytes=_bundle_with_flags([]), finality_session="2026-09-10")
        result = publication_effect(conn, store, claim, root, REPO, clock=clock, store_root=FAKE_STORE_ROOT)
        _ops_commit(conn, clock, claim, result)

        target = root / "releases" / SCOPE
        document = json.loads((target / "operations_status.json").read_text())
        assert document["requested_session"] == "2026-09-12"
        assert document["resolved_session"] == "2026-09-10"
    finally:
        conn.close()


def test_prior_selfcheck_observation_carries_into_status(tmp_path):
    """A previously bound selfcheck observation (``ok`` False, a real
    detail) reaches the status document unchanged -- never silently dropped
    or reset to unknown."""
    conn, clock, supervisor, store, root = _ops_open(tmp_path)
    try:
        claim = _setup_publication(conn, clock, supervisor, store, session=SESSION,
                                   deployment="shadow:impl-1", decision_clock="2026-09-12T01:00:00Z",
                                   bundle_bytes=_bundle_with_flags([]), selfcheck_ok=False,
                                   engineering_ok=True)
        with pytest.raises(OpsError, match="PUBLICATION_REFUSED"):
            publication_effect(conn, store, claim, root, REPO, clock=clock, store_root=FAKE_STORE_ROOT)

        target = root / "releases" / SCOPE
        document = json.loads((target / "operations_status.json").read_text())
        assert document["selfcheck"] == {"ok": False}
        assert document["failed_update"] is True
    finally:
        conn.close()


def test_failed_update_keeps_old_release_and_shows_reason(tmp_path):
    """A NEW generation that fails before publication (its own engineering
    gate is false, so ``stage_release`` never becomes eligible and
    ``publish_local`` refuses) leaves the prior release current and usable,
    and the status sidecar shows the failed-update reason for THIS attempt
    -- guide §5.5 item 2's "bind status to the current candidate/published
    release" plus item 3's failed-update test."""
    conn, clock, supervisor, store, root = _ops_open(tmp_path)
    try:
        good = _setup_publication(conn, clock, supervisor, store, session=SESSION,
                                  deployment="shadow:impl-1", decision_clock="2026-09-12T01:00:00Z",
                                  bundle_bytes=_bundle_with_flags([]))
        good_result = publication_effect(conn, store, good, root, REPO, clock=clock, store_root=FAKE_STORE_ROOT)
        _ops_commit(conn, clock, good, good_result)
        target = root / "releases" / SCOPE
        first_release_id = release_current(target)
        assert first_release_id is not None

        bad = _setup_publication(conn, clock, supervisor, store, session=SESSION,
                                 deployment="shadow:impl-2", decision_clock="2026-09-12T05:00:00Z",
                                 bundle_bytes=_bundle_with_flags([]), engineering_ok=False)
        with pytest.raises(OpsError, match="PUBLICATION_REFUSED"):
            publication_effect(conn, store, bad, root, REPO, clock=clock, store_root=FAKE_STORE_ROOT)

        # the old release is still current and readable
        assert release_current(target) == first_release_id
        assert (target / "releases" / first_release_id / "bundle.tar").is_file()

        document = json.loads((target / "operations_status.json").read_text())
        assert document["release_id"] == first_release_id
        assert document["attempted_release_id"] != first_release_id
        assert document["failed_update"] is True
        assert document["failed_update_reason"].startswith("PUBLICATION_REFUSED")
        assert document["stale"] is True
    finally:
        conn.close()


# --------------------------------------------------------------------------
# no-history release: the API-level "unknown, not green" case is
# ``tests/test_v2_serving_api.py``'s own ``test_operations_route_unknown_
# history_is_not_green``; this proves the ops side that feeds it never
# manufactures a "pass" for a night nobody observed.
# --------------------------------------------------------------------------


def test_unobserved_history_window_is_never_rendered_as_a_pass(tmp_path):
    conn, clock, supervisor, store, root = _ops_open(tmp_path)
    try:
        claim = _setup_publication(conn, clock, supervisor, store, session=SESSION,
                                   deployment="shadow:impl-1", decision_clock="2026-09-12T01:00:00Z",
                                   bundle_bytes=_bundle_with_flags([]))
        result = publication_effect(conn, store, claim, root, REPO, clock=clock, store_root=FAKE_STORE_ROOT)
        _ops_commit(conn, clock, claim, result)

        target = root / "releases" / SCOPE
        document = json.loads((target / "operations_status.json").read_text())
        statuses = {row["status"] for row in document["engineering_history"]}
        # no engineering_gate job ever ran in this test -- every night in the
        # window is unknown, never a fabricated pass.
        assert statuses == {"unknown"}
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 2026-09-14 review fix, item 1: the REAL observed failure shape --
# publication is BLOCKED (or cancelled) by an upstream dependency BEFORE
# publication_effect ever runs, so publication_effect's own sidecar write
# (everything above this section) never happens. Driven through a real
# ``supervisor.Service.tick()`` -- the actual hook site
# (``Service._reconcile_publication_status`` ->
# ``effects_graph.reconcile_publication_status``) -- with the failing/
# cancelled upstream attempt committed directly (``commit_attempt``/
# ``request_cancel``, no live subprocess), the same technique
# ``tests/test_v2_ops_effects_graph.py``'s own ``_succeed_parent`` uses for
# its SUCCEEDING parent jobs.
# --------------------------------------------------------------------------


def _blocked_publication_setup(conn, clock, supervisor, store, root, *, tag, gate_key):
    """A published, current release (so there is an old one to keep
    serving), plus a queued ``engineering_gate`` job and a ``publication``
    job depending on it -- neither claimed yet."""
    good = _setup_publication(conn, clock, supervisor, store, session=SESSION,
                              deployment="shadow:impl-1", decision_clock="2026-09-12T01:00:00Z",
                              bundle_bytes=_bundle_with_flags([]))
    good_result = publication_effect(conn, store, good, root, REPO, clock=clock, store_root=FAKE_STORE_ROOT)
    _ops_commit(conn, clock, good, good_result)
    target = root / "releases" / SCOPE
    first_release_id = release_current(target)
    assert first_release_id is not None

    gate_job = JobSpec(kind="engineering_gate", implementation_ref="test-impl", spec_hash=None,
                       environment_ref="test-env",
                       parameters=_params("engineering_gate", SESSION, SCOPE,
                                          deployment="shadow:impl-2",
                                          decision_clock="2026-09-12T05:00:00Z"),
                       output_namespace=SCOPE, resource_class="delivery",
                       retry_policy_ref="bounded", checkpoint_contract_ref="effect_receipt.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(namespace=SCOPE, idempotency_key=gate_key,
          principal="operator", job=gate_job), clock=clock)
    gate_job_id = job_id_for(SCOPE, gate_key)

    pub_job = JobSpec(kind="publication", implementation_ref="test-impl", spec_hash=None,
                      environment_ref="test-env",
                      parameters=_params("publication", SESSION, SCOPE, input_bindings={},
                                         deployment="shadow:impl-2",
                                         decision_clock="2026-09-12T05:00:00Z"),
                      dependency_job_ids=(gate_job_id,),
                      output_namespace=SCOPE, resource_class="delivery",
                      retry_policy_ref="bounded", checkpoint_contract_ref="effect_receipt.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(namespace=SCOPE, idempotency_key="pub-" + tag,
          principal="operator", job=pub_job), clock=clock)
    pub_job_id = job_id_for(SCOPE, "pub-" + tag)
    return target, first_release_id, gate_job_id, pub_job_id


def _tick_service(conn, root, clock):
    service = Service(conn, root, registry(), TEST_POLICY, clock=clock, code_source=REPO)
    service.start()
    try:
        service.tick()
    finally:
        service.close()


def test_upstream_engineering_gate_failure_blocks_publication_and_status_shows_it(tmp_path):
    """The real 2026-09-14 failure shape: ``engineering_gate`` fails,
    ``block_descendants`` blocks the dependent ``publication`` job BEFORE
    it is ever claimed -- ``publication_effect`` never runs. Without the
    review fix, the sidecar would keep describing the OLD release as fine
    (``failed_update=False``). With it, the old release stays current and
    the sidecar names the real cause."""
    conn, clock, supervisor, store, root = _ops_open(tmp_path)
    try:
        target, first_release_id, gate_job_id, pub_job_id = _blocked_publication_setup(
            conn, clock, supervisor, store, root, tag="blocked", gate_key="gate-fail")

        gate_claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                                supervisor=supervisor, clock=clock, registry=registry())
        assert gate_claim is not None and gate_claim.job_id == gate_job_id
        failure = make_problem("VALIDATION_FAILED", "engineering gate refused: boom")
        commit_attempt(conn, gate_claim.attempt_id, gate_claim.fence,
                       Outcome(False, "verified_dead", 1, failure), clock=clock)

        assert conn.execute("SELECT state FROM jobs WHERE job_id = ?", (pub_job_id,)
                            ).fetchone()["state"] == "blocked"
        # the sidecar still describes the OLD release, since publication_effect
        # never ran, until the service tick's reconciliation catches it.
        before = json.loads((target / "operations_status.json").read_text())
        assert before["failed_update"] is False

        _tick_service(conn, root, clock)

        document = json.loads((target / "operations_status.json").read_text())
        assert document["release_id"] == first_release_id
        assert document["attempted_release_id"] is None
        assert document["failed_update"] is True
        assert "engineering_gate" in document["failed_update_reason"]
        assert "VALIDATION_FAILED" in document["failed_update_reason"]
        assert release_current(target) == first_release_id
        assert (target / "releases" / first_release_id / "bundle.tar").is_file()
    finally:
        conn.close()


def test_cancelled_upstream_job_blocks_publication_and_status_shows_it(tmp_path):
    """The other real observed shape: the run is cancelled. ``request_cancel``
    on the still-queued (never claimed) ``engineering_gate`` job takes the
    immediate-cancel branch, blocking the dependent ``publication`` job the
    same way a failure does; the sidecar must show the cancellation, not
    stay silently green."""
    conn, clock, supervisor, store, root = _ops_open(tmp_path)
    try:
        target, first_release_id, gate_job_id, pub_job_id = _blocked_publication_setup(
            conn, clock, supervisor, store, root, tag="cancelled", gate_key="gate-cancel")

        request_cancel(conn, gate_job_id, None, clock=clock)
        assert conn.execute("SELECT state FROM jobs WHERE job_id = ?", (gate_job_id,)
                            ).fetchone()["state"] == "cancelled"
        assert conn.execute("SELECT state FROM jobs WHERE job_id = ?", (pub_job_id,)
                            ).fetchone()["state"] == "blocked"

        _tick_service(conn, root, clock)

        document = json.loads((target / "operations_status.json").read_text())
        assert document["release_id"] == first_release_id
        assert document["failed_update"] is True
        assert "engineering_gate" in document["failed_update_reason"]
        assert "cancelled" in document["failed_update_reason"].lower()
        assert release_current(target) == first_release_id
    finally:
        conn.close()


def test_reconcile_does_not_clobber_a_later_successful_publish(tmp_path):
    """A LATER, successful publication attempt's own sidecar (a fresher
    ``generated_at``) must survive a subsequent tick even though the
    earlier failed/blocked publication job row is still sitting in the
    table forever -- ``reconcile_publication_status``'s own ``updated_at``
    vs. ``generated_at`` comparison, not a one-shot flag."""
    conn, clock, supervisor, store, root = _ops_open(tmp_path)
    try:
        target, first_release_id, gate_job_id, pub_job_id = _blocked_publication_setup(
            conn, clock, supervisor, store, root, tag="recovers", gate_key="gate-fail-2")
        gate_claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                                supervisor=supervisor, clock=clock, registry=registry())
        failure = make_problem("VALIDATION_FAILED", "engineering gate refused: boom")
        commit_attempt(conn, gate_claim.attempt_id, gate_claim.fence,
                       Outcome(False, "verified_dead", 1, failure), clock=clock)
        _tick_service(conn, root, clock)
        blocked_document = json.loads((target / "operations_status.json").read_text())
        assert blocked_document["failed_update"] is True

        clock.advance(60)
        good_again = _setup_publication(conn, clock, supervisor, store, session=SESSION,
                                        deployment="shadow:impl-3",
                                        decision_clock="2026-09-12T09:00:00Z",
                                        bundle_bytes=_bundle_with_flags([]))
        good_result = publication_effect(conn, store, good_again, root, REPO, clock=clock, store_root=FAKE_STORE_ROOT)
        _ops_commit(conn, clock, good_again, good_result)
        second_release_id = release_current(target)
        assert second_release_id not in (None, first_release_id)

        _tick_service(conn, root, clock)

        document = json.loads((target / "operations_status.json").read_text())
        assert document["release_id"] == second_release_id
        assert document["failed_update"] is False
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 2026-09-14 review fix, item 3 (streak consistency): engineering_streak
# must be derived from the SAME windowed history the document shows, never
# budget_streak's own separate unbounded scan.
# --------------------------------------------------------------------------


def test_engineering_streak_is_windowed_not_budget_streaks_unbounded_scan(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    # a failure recorded well BEFORE the trailing window even starts
    record_check(conn, "2026-08-01", "engineering", False, {"attempt": 0})
    window = trailing_occurrences(SESSION, nights=3)
    for occurrence in window:
        record_check(conn, occurrence, "engineering", False, {"occurrence": occurrence})

    raw = engineering_history(conn, window)
    history = tuple(EngineeringNight(**row) for row in raw)
    windowed = engineering_streak_from_history(history)
    unbounded = budget_streak(conn)

    assert windowed["consecutive_nights"] == len(window)
    # budget_streak's own unbounded, all-time scan counts the 2026-08-01
    # failure too, so it disagrees with what the document's own history
    # actually shows -- confirming OperationsStatus must use the windowed
    # derivation, never budget_streak directly.
    assert unbounded["consecutive_nights"] == len(window) + 1
    assert windowed["consecutive_nights"] != unbounded["consecutive_nights"]
    conn.close()


# --------------------------------------------------------------------------
# 2026-09-14 SECOND review fix: reconcile_publication_status must look only
# at each scope's OWN LATEST publication job (any state), never an
# unordered scan of every terminal-non-success row -- reviewed against
# /root/phase2-shadow-ops, where several historical blocked publications
# with no sidecar yet meant the first pass wrote whichever row SQLite
# happened to return first, not the actual latest failure.
# --------------------------------------------------------------------------

TWO_SCOPE_POLICY = NamespacePolicy({"operator": frozenset({"shadow", "smoke"})})


def _insert_publication_job(conn, clock, *, scope, key, state, updated_at, reason=None):
    """A minimal, real ``publication`` job row (submitted through
    ``submit()``, so ``spec_json``/``output_namespace``/policy checks are
    all genuine) whose ``state``/``updated_at``/``failure_json`` are then
    set directly -- the only way to put a row at a SPECIFIC logical time
    independent of when it was actually inserted, which is exactly the
    "SQLite rowid order != time order" scenario this fix must handle."""
    job = JobSpec(kind="publication", implementation_ref="test-impl", spec_hash=None,
                 environment_ref="test-env",
                 parameters=_params("publication", SESSION, scope, input_bindings={}),
                 output_namespace=scope, resource_class="delivery",
                 retry_policy_ref="bounded", checkpoint_contract_ref="effect_receipt.v1.0")
    submit(conn, registry(), TWO_SCOPE_POLICY, SubmitRequest(
        namespace=scope, idempotency_key=key, principal="operator", job=job), clock=clock)
    job_id = job_id_for(scope, key)
    failure = _dumps(make_problem("VALIDATION_FAILED", reason)) if reason else None
    conn.execute("UPDATE jobs SET state = ?, updated_at = ?, failure_json = ?, "
                "next_eligible_at = NULL WHERE job_id = ?", (state, updated_at, failure, job_id))
    conn.commit()
    return job_id


def test_reconcile_names_the_newest_of_several_blocked_jobs_in_one_scope(tmp_path):
    """(a) Three historical blocked publications in ONE scope, inserted so
    SQLite rowid order != their logical time order -- the sidecar must name
    the NEWEST one's own reason, never whichever row SQLite returns first."""
    conn, clock, supervisor, store, root = _ops_open(tmp_path)
    try:
        target = root / "releases" / SCOPE
        # insertion order: middle, oldest, newest -- rowid order agrees with
        # NEITHER the time order nor its reverse.
        _insert_publication_job(conn, clock, scope=SCOPE, key="pub-mid", state="blocked",
                                updated_at="2026-09-11T00:00:00.000000Z", reason="middle failure")
        _insert_publication_job(conn, clock, scope=SCOPE, key="pub-old", state="blocked",
                                updated_at="2026-09-10T00:00:00.000000Z", reason="oldest failure")
        newest_id = _insert_publication_job(conn, clock, scope=SCOPE, key="pub-new", state="blocked",
                                            updated_at="2026-09-12T00:00:00.000000Z",
                                            reason="newest failure")

        _tick_service(conn, root, clock)

        document = json.loads((target / "operations_status.json").read_text())
        assert "newest failure" in document["failed_update_reason"]
        assert "middle failure" not in document["failed_update_reason"]
        assert "oldest failure" not in document["failed_update_reason"]
        assert document["attempted_release_id"] is None
        assert document["failed_update"] is True
        assert newest_id  # sanity: the id we asserted against was really built
    finally:
        conn.close()


@pytest.mark.parametrize("older_first", [True, False])
def test_reconcile_skips_an_older_blocked_job_behind_a_newer_success(tmp_path, older_first):
    """(b) An older blocked job plus a NEWER succeeded publication in the
    same scope: no failure sidecar is written, regardless of which one was
    inserted first."""
    conn, clock, supervisor, store, root = _ops_open(tmp_path)
    try:
        target = root / "releases" / SCOPE
        if older_first:
            _insert_publication_job(conn, clock, scope=SCOPE, key="pub-old", state="blocked",
                                    updated_at="2026-09-10T00:00:00.000000Z", reason="stale failure")
            _insert_publication_job(conn, clock, scope=SCOPE, key="pub-good", state="succeeded",
                                    updated_at="2026-09-12T00:00:00.000000Z")
        else:
            _insert_publication_job(conn, clock, scope=SCOPE, key="pub-good", state="succeeded",
                                    updated_at="2026-09-12T00:00:00.000000Z")
            _insert_publication_job(conn, clock, scope=SCOPE, key="pub-old", state="blocked",
                                    updated_at="2026-09-10T00:00:00.000000Z", reason="stale failure")

        _tick_service(conn, root, clock)

        assert not (target / "operations_status.json").exists(), (
            f"older_first={older_first}: a failure sidecar must not be written when "
            "the scope's LATEST publication job already succeeded")
    finally:
        conn.close()


def test_reconcile_handles_two_scopes_independently(tmp_path):
    """(c) One scope's latest publication is blocked, the other's already
    succeeded -- each scope's own sidecar reflects only its own latest job."""
    conn, clock, supervisor, store, root = _ops_open(tmp_path)
    try:
        _insert_publication_job(conn, clock, scope=SCOPE, key="pub-shadow-blocked", state="blocked",
                                updated_at="2026-09-12T00:00:00.000000Z", reason="shadow failure")
        _insert_publication_job(conn, clock, scope="smoke", key="pub-smoke-good", state="succeeded",
                                updated_at="2026-09-12T00:00:00.000000Z")

        _tick_service(conn, root, clock)

        shadow_target = root / "releases" / SCOPE
        smoke_target = root / "releases" / "smoke"
        shadow_document = json.loads((shadow_target / "operations_status.json").read_text())
        assert shadow_document["failed_update"] is True
        assert "shadow failure" in shadow_document["failed_update_reason"]
        assert not (smoke_target / "operations_status.json").exists()
    finally:
        conn.close()


def test_publication_status_reconcile_failure_prints_once_while_persisting(tmp_path, monkeypatch, capsys):
    """Minor, 2026-09-14 second review fix: ``tick()`` runs roughly every
    second, so a PERSISTING reconciliation problem must not print one line
    per tick -- only when the (code, message) changes from the last one
    this ``Service`` instance actually printed; a genuinely different
    problem (or the same one recurring after a clean pass) prints again."""
    conn, clock, supervisor, store, root = _ops_open(tmp_path)
    try:
        service = Service(conn, root, registry(), TEST_POLICY, clock=clock, code_source=REPO)
        import engine.v2.ops.supervisor as supervisor_module

        def _broken(*args, **kwargs):
            raise OpsError(make_problem("VALIDATION_FAILED", "boom"))

        monkeypatch.setattr(supervisor_module, "reconcile_publication_status", _broken)
        service._reconcile_publication_status()
        service._reconcile_publication_status()
        service._reconcile_publication_status()
        printed = [line for line in capsys.readouterr().out.splitlines() if line]
        assert len(printed) == 1

        def _different(*args, **kwargs):
            raise OpsError(make_problem("VALIDATION_FAILED", "a different boom"))

        monkeypatch.setattr(supervisor_module, "reconcile_publication_status", _different)
        service._reconcile_publication_status()
        printed = [line for line in capsys.readouterr().out.splitlines() if line]
        assert len(printed) == 1
    finally:
        conn.close()
