"""Cutover PR-4 redo, slice 2B(b): the ``native_parity`` tick-loop sidecar.

``native_parity`` has no submission path of its own -- its only submitter is
``supervisor.Service._reconcile_native_parity``, mirroring
``_reconcile_computed_moves_refresh``/``_reconcile_native_score_batch_shadow``
in shape, with ONE addition those two do not have: a CONFIRMED
``records.json``/``refusals.json`` ``schema_version`` mismatch (an ``OpsError``
whose ``problem.details["reason"] == "schema_mismatch"``) is a permanent wait
state for that specific ``native_score_batch`` job id -- those artifacts can
never change schema underneath a finished job -- so it is remembered in its
own single-slot memo and costs literally nothing on every later tick (no
artifact read, no attempt spent, ``self._native_parity_memo`` not even read).

These tests prove only supervisor.py's own new code: the identity gate, the
memo arithmetic (always ``self._native_parity_memo``, never the
``self._computed_moves_memo`` slot ``_computed_moves_backoff`` hardcodes), and
that short-circuit. ``nightly._native_parity_identity`` and
``nightly.submit_native_parity_if_ready`` are both monkeypatched at the module
level, exactly how
``tests/test_v2_ops_native_score_batch_shadow_wiring.py`` stubs
``nightly._native_score_batch_identity`` -- nightly.py's own behaviour is
nightly.py's tests to prove.
"""
from __future__ import annotations

from pathlib import Path

from engine.v2.ops import nightly
from engine.v2.ops.errors import OpsError, fail, make_problem
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, job_id_for
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY, catalog

ROOT = Path(__file__).resolve().parents[1]
_POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})
#: (as_of, scope_hash, score_job_id, native_score_batch_job_id) -- the shape
#: nightly._native_parity_identity returns; the first two are what the sidecar
#: keys its idempotency check on, the last is the schema-mismatch memo's key.
_IDENTITY = ("2026-01-01", "sha256:0123456789abcdef0123", "score-job", "batch-job")


def _service(tmp_path, conn, clock, policy=_POLICY):
    return Service(conn, tmp_path, registry(), policy, clock=clock,
                   code_source=ROOT, store_root=tmp_path)


def _identity_stub(monkeypatch, identity):
    monkeypatch.setattr(nightly, "_native_parity_identity", lambda conn: identity)


def _submit_stub(monkeypatch, outcome):
    """Records every call; returns (or raises) ``outcome``."""
    calls = []

    def _stub(conn, registry_, policy, store, **kwargs):
        calls.append({"policy": policy, "kwargs": kwargs})
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(nightly, "submit_native_parity_if_ready", _stub)
    return calls


def _report_spy(monkeypatch, service):
    seen = []
    monkeypatch.setattr(service, "_report_native_parity_problem", seen.append)
    return seen


def _schema_mismatch(native_score_batch_job_id):
    return OpsError(make_problem(
        "VALIDATION_FAILED",
        "native_score_batch records/refusals schema_version is stale for this identity",
        details={"reason": "schema_mismatch",
                 "native_score_batch_job_id": native_score_batch_job_id}))


def _parity_job_id(identity):
    return job_id_for("shadow", nightly._native_parity_key(identity[0], identity[1]))


def _insert_parity_job(conn, clock, identity):
    """A real ``native_parity`` row already under this identity's key -- the
    "already submitted" state the sidecar must dedupe against."""
    stamp = clock.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    key = nightly._native_parity_key(identity[0], identity[1])
    conn.execute(
        "INSERT INTO jobs (job_id, namespace, idempotency_key, request_digest, principal, kind, "
        "spec_json, resource_class, checkpoint_contract_ref, retry_json, state, priority, "
        "max_attempts, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (_parity_job_id(identity), "shadow", key, "digest", "operator", "native_parity",
         "{}", "validation", "native_parity_report.v1.1", "{}", "queued", 0, 1, stamp, stamp))
    conn.commit()


# --------------------------------------------------------------------------
# _native_parity_identity_or_none
# --------------------------------------------------------------------------


def test_identity_gate_returns_none_and_leaves_the_memo_alone_when_there_is_no_identity(
        tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    service = _service(tmp_path, conn, clock)
    _identity_stub(monkeypatch, None)
    service._native_parity_memo = None

    assert service._native_parity_identity_or_none(0.0) is None
    assert service._native_parity_memo is None
    assert service._last_native_parity_problem is None


def test_identity_gate_clears_the_memo_when_a_job_already_exists(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    service = _service(tmp_path, conn, clock)
    _identity_stub(monkeypatch, _IDENTITY)
    _insert_parity_job(conn, clock, _IDENTITY)
    service._native_parity_memo = {"identity": _IDENTITY, "attempts": 2, "not_before": 10_000.0}

    assert service._native_parity_identity_or_none(0.0) is None
    # nothing left to back off from once the job exists -- the memo is dropped,
    # not kept as a capped entry.
    assert service._native_parity_memo is None


def test_identity_gate_lookup_failure_lands_in_the_identity_none_bucket(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    service = _service(tmp_path, conn, clock)
    reported = _report_spy(monkeypatch, service)
    calls = []

    def _boom(conn):
        calls.append(1)
        raise ValueError("planted identity lookup failure")

    monkeypatch.setattr(nightly, "_native_parity_identity", _boom)

    assert service._native_parity_identity_or_none(0.0) is None
    memo = service._native_parity_lookup_memo  # the SEPARATE lookup slot...
    assert service._native_parity_memo is None  # ...never the real-identity memo
    assert memo["attempts"] == 1
    assert memo["not_before"] == Service._COMPUTED_MOVES_BACKOFF_SECONDS[0]
    assert len(reported) == 1
    # the wrong-slot defect class: this sidecar's own arithmetic must never
    # touch the computed_moves memo _computed_moves_backoff hardcodes.
    assert service._computed_moves_memo is None

    # still inside the window -- the lookup is not re-run at all.
    assert service._native_parity_identity_or_none(1.0) is None
    assert len(calls) == 1


def test_identity_gate_lookup_failure_is_uncapped_and_clamps_to_the_slowest_window(
        tmp_path, monkeypatch):
    """Unlike a real identity's capped attempts, a lookup failure retries
    forever -- there is no "new identity" signal of its own to reset on, so
    giving up after 5 would silently disable this stage for the rest of the
    process's life. Past the schedule's end it clamps to its LAST entry
    (3600s) rather than raising or stopping."""
    conn, clock, _ = catalog(tmp_path)
    service = _service(tmp_path, conn, clock)
    reported = _report_spy(monkeypatch, service)
    calls = []

    def _boom(conn):
        calls.append(1)
        raise ValueError("planted identity lookup failure")

    monkeypatch.setattr(nightly, "_native_parity_identity", _boom)

    schedule = Service._COMPUTED_MOVES_BACKOFF_SECONDS
    now = 0.0
    for attempt in range(1, 8):  # 7 consecutive failures, past the 5-attempt cap
        assert service._native_parity_identity_or_none(now) is None
        memo = service._native_parity_lookup_memo
        assert memo["attempts"] == attempt
        assert memo["not_before"] == now + schedule[min(attempt - 1, len(schedule) - 1)]
        now = memo["not_before"]
    assert len(calls) == 7
    assert len(reported) == 7
    assert service._native_parity_lookup_memo["attempts"] == 7  # never capped, unlike a real identity


def test_identity_lookup_failure_never_spends_a_real_identitys_attempt_budget(
        tmp_path, monkeypatch):
    """CodeRabbit round 2, real finding: a lookup failure must not
    overwrite/replace a real identity's own submission-attempt memo -- the
    two now live in separate slots entirely."""
    conn, clock, _ = catalog(tmp_path)
    service = _service(tmp_path, conn, clock)
    reported = _report_spy(monkeypatch, service)
    real_identity = _IDENTITY
    service._native_parity_memo = {"identity": real_identity, "attempts": 3,
                                   "not_before": 0.0}

    def _boom(conn):
        raise ValueError("planted identity lookup failure")

    monkeypatch.setattr(nightly, "_native_parity_identity", _boom)

    assert service._native_parity_identity_or_none(0.0) is None

    # The lookup failure landed in its OWN slot...
    assert service._native_parity_lookup_memo["attempts"] == 1
    # ...and the real identity's own memo (attempts=3) is untouched.
    assert service._native_parity_memo == {"identity": real_identity, "attempts": 3,
                                           "not_before": 0.0}
    assert len(reported) == 1


# --------------------------------------------------------------------------
# _reconcile_native_parity
# --------------------------------------------------------------------------


def test_reconcile_is_a_noop_without_an_identity(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    service = _service(tmp_path, conn, clock)
    _identity_stub(monkeypatch, None)
    calls = _submit_stub(monkeypatch, object())

    service._reconcile_native_parity()

    assert calls == []


def test_reconcile_submits_once_for_a_fresh_identity_under_its_own_shadow_policy(
        tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    service = _service(tmp_path, conn, clock)
    _identity_stub(monkeypatch, _IDENTITY)
    calls = _submit_stub(monkeypatch, object())

    service._reconcile_native_parity()

    assert len(calls) == 1
    assert calls[0]["policy"].allows("operator", "shadow")
    assert calls[0]["kwargs"]["objects_root"] == str(tmp_path)
    # a real submission clears the memo -- nothing left to retry.
    assert service._native_parity_memo is None


def test_reconcile_never_attempts_past_the_cap_for_the_same_identity(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    service = _service(tmp_path, conn, clock)
    _identity_stub(monkeypatch, _IDENTITY)
    calls = _submit_stub(monkeypatch, object())
    memo = {"identity": _IDENTITY, "attempts": Service._COMPUTED_MOVES_MAX_ATTEMPTS,
            "not_before": 0.0}
    service._native_parity_memo = memo

    service._reconcile_native_parity()

    assert calls == []
    assert service._native_parity_memo is memo
    assert service._native_parity_memo["attempts"] == Service._COMPUTED_MOVES_MAX_ATTEMPTS


def test_reconcile_schema_mismatch_parks_the_job_id_and_leaves_the_memo_untouched(
        tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    service = _service(tmp_path, conn, clock)
    _identity_stub(monkeypatch, _IDENTITY)
    reported_identity = (_IDENTITY[0], _IDENTITY[1], _IDENTITY[2], "reported-batch-job")
    memo = {"identity": _IDENTITY, "attempts": 2, "not_before": 0.0}
    service._native_parity_memo = memo
    reported = _report_spy(monkeypatch, service)
    calls = _submit_stub(monkeypatch, _schema_mismatch(reported_identity[3]))

    service._reconcile_native_parity()

    assert len(calls) == 1
    assert service._native_parity_schema_mismatch_job_id == reported_identity[3]
    assert service._native_parity_schema_mismatch_job_id != _IDENTITY[3]
    # ZERO cost: no attempt spent, the very same memo object, untouched.
    assert service._native_parity_memo is memo
    assert memo["attempts"] == 2
    assert memo["not_before"] == 0.0
    # a permanent wait state is not a reportable problem.
    assert reported == []
    assert service._last_native_parity_problem is None

    # a LATER tick carrying the SAME batch job id short-circuits ahead of all
    # memo/backoff machinery -- even though this memo still has 3 attempts left
    # and no backoff window, submit is never reached again for that job id.
    _identity_stub(monkeypatch, reported_identity)
    service._reconcile_native_parity()
    service._reconcile_native_parity()
    assert len(calls) == 1
    assert service._native_parity_memo is memo
    assert memo["attempts"] == 2


def test_schema_mismatch_never_short_circuits_a_new_batch_job_id(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    service = _service(tmp_path, conn, clock)
    _identity_stub(monkeypatch, _IDENTITY)
    _submit_stub(monkeypatch, _schema_mismatch(_IDENTITY[3]))
    service._reconcile_native_parity()
    assert service._native_parity_schema_mismatch_job_id == _IDENTITY[3]

    new_identity = (_IDENTITY[0], _IDENTITY[1], _IDENTITY[2], "brand-new-batch-job")
    _identity_stub(monkeypatch, new_identity)
    calls = _submit_stub(monkeypatch, object())

    service._reconcile_native_parity()

    assert len(calls) == 1  # the stale parked id does not block a new job
    assert service._native_parity_schema_mismatch_job_id == _IDENTITY[3]
    assert service._native_parity_memo is None


def test_reconcile_spends_one_attempt_on_a_plain_exception(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    service = _service(tmp_path, conn, clock)
    _identity_stub(monkeypatch, _IDENTITY)
    reported = _report_spy(monkeypatch, service)
    failure = ValueError("planted parity build failure")
    calls = _submit_stub(monkeypatch, failure)

    service._reconcile_native_parity()

    memo = service._native_parity_memo
    assert memo["identity"] == _IDENTITY
    assert memo["attempts"] == 1
    assert memo["not_before"] == Service._COMPUTED_MOVES_BACKOFF_SECONDS[0]
    assert reported == [failure]
    # an ordinary transient failure is NOT a confirmed mismatch.
    assert service._native_parity_schema_mismatch_job_id is None

    service._reconcile_native_parity()  # still inside the window -- no second attempt
    assert len(calls) == 1
    assert service._native_parity_memo["attempts"] == 1

    clock.advance(Service._COMPUTED_MOVES_BACKOFF_SECONDS[0])
    service._reconcile_native_parity()
    assert len(calls) == 2
    assert service._native_parity_memo["attempts"] == 2
    assert service._native_parity_memo["not_before"] == (
        Service._COMPUTED_MOVES_BACKOFF_SECONDS[0]
        + Service._COMPUTED_MOVES_BACKOFF_SECONDS[1])


def test_reconcile_treats_a_retryable_ops_error_without_the_mismatch_reason_as_transient(
        tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    service = _service(tmp_path, conn, clock)
    _identity_stub(monkeypatch, _IDENTITY)
    reported = _report_spy(monkeypatch, service)
    failure = OpsError(make_problem("RESOURCE_UNAVAILABLE", "a transient dependency was down",
                                    details={"reason": "unavailable"}))
    _submit_stub(monkeypatch, failure)

    service._reconcile_native_parity()

    memo = service._native_parity_memo
    assert memo["attempts"] == 1
    assert memo["not_before"] == Service._COMPUTED_MOVES_BACKOFF_SECONDS[0]
    assert len(reported) == 1
    assert service._native_parity_schema_mismatch_job_id is None


def test_reconcile_spends_the_full_attempt_budget_on_a_non_retryable_ops_error(
        tmp_path, monkeypatch):
    """CodeRabbit round 6: a malformed committed artifact raises a
    non-retryable VALIDATION_FAILED whose underlying bytes can never change --
    the budget is spent in full on the FIRST failure (like a capped-out
    identity settling into the slowest cadence), never walked backoff-by-backoff."""
    conn, clock, _ = catalog(tmp_path)
    service = _service(tmp_path, conn, clock)
    _identity_stub(monkeypatch, _IDENTITY)
    reported = _report_spy(monkeypatch, service)
    failure = fail("VALIDATION_FAILED", "boom")
    calls = _submit_stub(monkeypatch, failure)

    service._reconcile_native_parity()

    assert failure.problem.retryable is False
    memo = service._native_parity_memo
    assert memo["identity"] == _IDENTITY
    assert memo["attempts"] == Service._COMPUTED_MOVES_MAX_ATTEMPTS
    assert len(reported) == 1
    # reported once, but it is NOT a confirmed mismatch -- no parking.
    assert service._native_parity_schema_mismatch_job_id is None

    # the cap short-circuits the SAME identity immediately -- the expensive
    # path is never reached again, with no backoff window to wait out.
    service._reconcile_native_parity()
    assert len(calls) == 1
    assert service._native_parity_memo is memo
    assert memo["attempts"] == Service._COMPUTED_MOVES_MAX_ATTEMPTS


def test_reconcile_clears_the_memo_on_a_receipt(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    service = _service(tmp_path, conn, clock)
    _identity_stub(monkeypatch, _IDENTITY)
    service._native_parity_memo = {"identity": _IDENTITY, "attempts": 3, "not_before": 0.0}
    _submit_stub(monkeypatch, object())

    service._reconcile_native_parity()

    assert service._native_parity_memo is None
    assert service._last_native_parity_problem is None


def test_reconcile_spends_an_attempt_when_submit_returns_none(tmp_path, monkeypatch):
    """Defensive: ``_native_parity_identity_or_none`` already confirmed no job
    exists, so a ``None`` back means something else won the race -- still one
    spent attempt with backoff, never a silent free retry every tick."""
    conn, clock, _ = catalog(tmp_path)
    service = _service(tmp_path, conn, clock)
    _identity_stub(monkeypatch, _IDENTITY)
    calls = _submit_stub(monkeypatch, None)

    service._reconcile_native_parity()

    assert len(calls) == 1
    memo = service._native_parity_memo
    assert memo["identity"] == _IDENTITY
    assert memo["attempts"] == 1
    assert memo["not_before"] == Service._COMPUTED_MOVES_BACKOFF_SECONDS[0]
    assert service._native_parity_schema_mismatch_job_id is None


# --------------------------------------------------------------------------
# tick wiring
# --------------------------------------------------------------------------


def test_tick_calls_the_native_parity_sidecar_once_per_tick(tmp_path, monkeypatch):
    """Mirrors how tests/test_v2_ops_computed_moves_nightly_wiring.py drives a
    real ``Service.tick()`` -- a ``ResourcePolicy`` (``TEST_POLICY``), never the
    ``NamespacePolicy`` stand-in the submit-only tests use, because
    ``claim_next`` needs ``.profiles``. The other three sidecars are stubbed to
    isolate this one's own call."""
    conn, clock, _ = catalog(tmp_path)
    service = _service(tmp_path, conn, clock, policy=TEST_POLICY)
    calls = []
    monkeypatch.setattr(service, "_reconcile_native_parity", lambda: calls.append(1))
    for name in ("_reconcile_publication_status", "_reconcile_computed_moves_refresh",
                 "_reconcile_native_score_batch_shadow"):
        monkeypatch.setattr(service, name, lambda: None)

    service.start()
    try:
        service.tick()
        assert len(calls) == 1
        service.tick()
    finally:
        service.close()

    assert len(calls) == 2
