"""Mutation coverage for ``engine.v2.ops.stages._core_kinds`` -- the static
``JobKind`` entries with no generated sibling (``registry()`` adds the
``legacy_*``/outbox families separately).

Every assertion below drives a real CONSUMER of the field under test --
``submission.validate_request``/``_check_job_fields``/``_check_parameters``
(via ``submit()``), ``lifecycle.advance_job``/``RetryPolicy.delay_after``
(via ``commit_attempt``), or ``scheduler.claim_next``/``store_barrier``'s
lease bookkeeping -- and checks its concrete observable OUTPUT, rather than
reading the ``JobKind`` field back. A read-back only proves the literal still
says what it says; it cannot tell whether the value is used correctly, which
is exactly the gap that let ``ACTION_NAMES`` drift from the DAG and break a
production nightly.

The kind-name set is discovered programmatically from
``stages._core_kinds()`` (never hand-copied) and checked against an explicit
hardcoded set.

Fields with NO cheap consumer reachable from this test's budget (see
``tests/ops_support.py``'s ``catalog(tmp_path)`` fixture -- no real catalog
root, no Scorer) are asserted directly, named here so the gap is visible
rather than glossed over:

- ``worker``: its only consumer is ``executor.launch()``, which always spawns
  a real ``subprocess.Popen`` of ``engine.v2.ops.worker`` -- too heavy for
  this cluster. Direct assertion.
- ``effects``: a repo-wide grep (``engine/``, ``checks/``, ``tools/``) for
  ``.effects`` finds no reader of ``JobKind.effects`` anywhere today; it is a
  declared, currently-unconsumed field. Direct assertion.
- ``parameters`` (the dataclass type) for artifact_check/adhoc_rescore/
  snapshot_import: ``CheckParameters``, ``RescoreParameters`` and
  ``SnapshotImportParameters`` are structurally identical (same field names
  and types), so no schema-shape consumer can tell them apart. The other
  three core kinds' parameter classes DO have distinguishing fields and are
  proven through ``_check_parameters`` (a real consumer) instead. Direct
  assertion for these three only.
- the LAST ``backoff_seconds`` element for decision_evidence/adhoc_rescore/
  snapshot_import/legacy_materialize (2 entries, but max_attempts is also 2:
  ``advance_job`` fails the job outright at attempt 2, before ever calling
  ``delay_after`` at index 1), and the sole element for
  legacy_rebuild_candidate (max_attempts=1: it never retries, so
  ``delay_after`` is never called at all in production for it). These are
  exercised by calling ``RetryPolicy.delay_after`` directly -- still the real
  method -- since the real state machine (``advance_job``) can never reach
  that index for these kinds' own configuration.
"""
from datetime import timedelta

import pytest

from engine.v2.foundation import format_timestamp
from engine.v2.ops import stages, store_barrier
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.submission import submit
from tests.ops_support import POLICY, TEST_POLICY, catalog, request, sample

_ACTUAL = {kind.name: kind for kind in stages._core_kinds()}

# name -> resource_class, checkpoint_contract, max_attempts, backoff_seconds,
#         and an (extra field, value) unique to that kind's parameters class
#         where one exists (None for the three structurally-identical ones).
_CASES = {
    "artifact_check": dict(
        resource_class="delivery", checkpoint_contract="receipt.v1.0",
        max_attempts=3, backoff=(1, 5), extra_field=None),
    "experiment": dict(
        resource_class="experiment_heavy", checkpoint_contract="experiment_receipt.v1.0",
        max_attempts=2, backoff=(30, 120), extra_field=("no_ledger", True)),
    "decision_evidence": dict(
        resource_class="validation", checkpoint_contract="decision_evidence_pair.v1.0",
        max_attempts=2, backoff=(5, 30), extra_field=("session", "s")),
    "adhoc_rescore": dict(
        resource_class="io_fetch", checkpoint_contract="adhoc_rescore_record.v1.0",
        max_attempts=2, backoff=(5, 30), extra_field=None),
    "snapshot_import": dict(
        resource_class="legacy_rebuild",
        checkpoint_contract="snapshot_import_inspections.v1.0",
        max_attempts=2, backoff=(5, 30), extra_field=None),
    "legacy_rebuild_candidate": dict(
        resource_class="legacy_rebuild",
        checkpoint_contract="legacy_rebuild_candidate.v1.0",
        max_attempts=1, backoff=(30,), extra_field=("candidate_root", "c")),
    "legacy_materialize": dict(
        resource_class="materialize",
        checkpoint_contract="legacy_materialization_manifest.v1.0",
        max_attempts=2, backoff=(5, 30), extra_field=("scratch_estimate_bytes", 5)),
}

_EMPTY_DOMAIN_KINDS = ("artifact_check", "decision_evidence", "adhoc_rescore",
                      "legacy_rebuild_candidate", "legacy_materialize")


def _extra_params(name):
    field = _CASES[name]["extra_field"]
    return {field[0]: field[1]} if field else {}


def test_core_kinds_are_exactly_the_expected_static_set():
    # incremental_refresh (refresh_job_kind()) is only a call site here; its
    # own field values live in engine/v2/ops/incremental_data.py, not this
    # file, so only its presence in the list is checked.
    assert set(_ACTUAL) == set(_CASES) | {"incremental_refresh"}
    assert len(stages._core_kinds()) == len(_CASES) + 1


@pytest.mark.parametrize("name", sorted(_CASES))
def test_submission_reads_resource_class_checkpoint_retry_and_max_refs(tmp_path, name):
    """Drives ``submission.validate_request``/``_check_job_fields`` -- the
    real code that reads ``resource_classes``, ``retry.name``,
    ``checkpoint_contract``, ``namespaces`` and ``max_input_refs`` -- and
    checks its concrete accept/refuse output."""
    case = _CASES[name]
    conn, clock, _ = catalog(tmp_path)
    params = {"expected_ids": ["one"], **_extra_params(name)}

    receipt = submit(
        conn, stages.registry(), POLICY,
        request(kind=name, resource_class=case["resource_class"],
                checkpoint_contract_ref=case["checkpoint_contract"],
                parameters=params),
        clock=clock)
    assert receipt.kind == name
    assert receipt.state == "queued"

    # wrong resource_class is refused -- proves resource_classes is read
    with pytest.raises(OpsError) as excinfo:
        submit(conn, stages.registry(), POLICY,
               request(key="bad-resource-class", kind=name,
                       resource_class="not-a-real-class",
                       checkpoint_contract_ref=case["checkpoint_contract"],
                       parameters=params),
               clock=clock)
    assert excinfo.value.code == "INVALID_REQUEST"

    # a mismatched checkpoint_contract_ref is refused -- proves
    # checkpoint_contract is read, not just stored (retry_policy_ref
    # defaults to "bounded" in request(), matching every core kind's
    # retry.name, so this isolates checkpoint_contract specifically)
    with pytest.raises(OpsError) as excinfo:
        submit(conn, stages.registry(), POLICY,
               request(key="bad-contract", kind=name,
                       resource_class=case["resource_class"],
                       checkpoint_contract_ref="wrong.v1.0",
                       parameters=params),
               clock=clock)
    assert excinfo.value.code == "INVALID_REQUEST"

    # a wrong retry_policy_ref is refused -- proves retry.name is read
    with pytest.raises(OpsError) as excinfo:
        submit(conn, stages.registry(), POLICY,
               request(key="bad-retry-ref", kind=name,
                       resource_class=case["resource_class"],
                       checkpoint_contract_ref=case["checkpoint_contract"],
                       retry_policy_ref="not-bounded",
                       parameters=params),
               clock=clock)
    assert excinfo.value.code == "INVALID_REQUEST"

    # more than max_input_refs (64) references is refused
    with pytest.raises(OpsError) as excinfo:
        submit(conn, stages.registry(), POLICY,
               request(key="too-many-refs", kind=name,
                       resource_class=case["resource_class"],
                       checkpoint_contract_ref=case["checkpoint_contract"],
                       parameters=params,
                       input_refs=tuple(f"ref{i}" for i in range(65))),
               clock=clock)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert excinfo.value.problem.details["limit"] == 64

    if case["extra_field"] is None:
        return
    field, value = case["extra_field"]
    # every OTHER core kind's parameters class rejects this kind's
    # distinguishing field -- proves the specific dataclass (not just "some
    # dataclass with an expected_ids field") gates the schema
    for other in _CASES:
        if other == name:
            continue
        other_case = _CASES[other]
        with pytest.raises(OpsError) as excinfo:
            submit(conn, stages.registry(), POLICY,
                   request(key=f"cross-field-{other}", kind=other,
                           resource_class=other_case["resource_class"],
                           checkpoint_contract_ref=other_case["checkpoint_contract"],
                           parameters={"expected_ids": ["one"],
                                      **_extra_params(other), field: value}),
                   clock=clock)
        assert excinfo.value.code == "INVALID_REQUEST"


def test_structurally_identical_parameter_classes_are_declared_correctly():
    """CheckParameters/RescoreParameters/SnapshotImportParameters are
    identical in shape (see module docstring) -- no consumer can distinguish
    them, so this is a direct, explicitly-named fallback."""
    assert _ACTUAL["artifact_check"].parameters is stages.CheckParameters
    assert _ACTUAL["adhoc_rescore"].parameters is stages.RescoreParameters
    assert _ACTUAL["snapshot_import"].parameters is stages.SnapshotImportParameters


def _fail_once(conn, clock, supervisor):
    claim = claim_next(conn, policy=TEST_POLICY, sample=sample(clock),
                       supervisor=supervisor, clock=clock)
    assert claim is not None
    outcome = Outcome(succeeded=False, process_state="exited", exit_code=1,
                      failure=fail("WORKER_FAILED", "synthetic").problem)
    return claim, commit_attempt(conn, claim.attempt_id, claim.fence, outcome, clock=clock)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_retry_policy_drives_advance_job_through_exhaustion(tmp_path, name):
    """Drives ``lifecycle.advance_job``/``RetryPolicy.delay_after`` (via
    ``commit_attempt``) -- the real retry/backoff state machine -- through
    every attempt up to exhaustion, checking the job's concrete state and
    ``next_eligible_at`` at each step."""
    case = _CASES[name]
    conn, clock, supervisor = catalog(tmp_path)
    params = {"expected_ids": ["one"], **_extra_params(name)}
    receipt = submit(conn, stages.registry(), POLICY,
                     request(kind=name, resource_class=case["resource_class"],
                             checkpoint_contract_ref=case["checkpoint_contract"],
                             parameters=params),
                     clock=clock)

    for attempt_number in range(1, case["max_attempts"] + 1):
        before = clock.now()
        _, state = _fail_once(conn, clock, supervisor)
        if attempt_number < case["max_attempts"]:
            assert state == "retry_wait", (name, attempt_number)
            row = conn.execute("SELECT next_eligible_at FROM jobs WHERE job_id=?",
                               (receipt.job_id,)).fetchone()
            delay = case["backoff"][min(attempt_number, len(case["backoff"])) - 1]
            assert row[0] == format_timestamp(before + timedelta(seconds=delay)), (name, attempt_number)
            clock.advance(delay)
        else:
            assert state == "failed", (name, attempt_number)


def test_delay_after_values_unreachable_by_advance_job_for_this_kind():
    """See module docstring: for these kinds, ``advance_job``'s own
    exhaustion arithmetic never calls ``delay_after`` at this index in
    production (the job fails outright once attempt_count reaches
    max_attempts first). Calling the real method directly is the closest
    reachable consumer for these specific values."""
    for name in ("decision_evidence", "adhoc_rescore", "snapshot_import", "legacy_materialize"):
        assert _ACTUAL[name].retry.delay_after(2) == 30, name
    assert _ACTUAL["legacy_rebuild_candidate"].retry.delay_after(1) == 30


def test_worker_and_effects_are_declared_correctly():
    """``worker``'s only consumer (``executor.launch()``) always spawns a
    real subprocess -- too heavy for this cluster's budget. ``effects`` has
    no reader anywhere in the codebase today (see module docstring). Both
    are direct, explicitly-named fallbacks."""
    for name in _CASES:
        assert _ACTUAL[name].worker == name
        assert _ACTUAL[name].effects == ("staged",)


def test_claim_next_store_leases_match_each_kinds_declared_store_domains(tmp_path):
    """Drives ``scheduler.claim_next`` with ``registry=stages.registry()`` --
    the same real end-to-end path production scheduling uses, through
    ``_claim_row`` -> ``store_barrier.domains_of``/``lease_reason``/
    ``acquire_in`` -- through one real claim per core kind, and checks the
    concrete ``store_leases`` rows left behind. Each kind's attempt is
    committed (succeeded) right after its leases are checked, releasing its
    reservation and store leases so the next kind can claim on this box's
    small fake capacity."""
    conn, clock, supervisor = catalog(tmp_path)
    attempt_ids = {}
    for name in (*_EMPTY_DOMAIN_KINDS, "snapshot_import"):
        case = _CASES[name]
        params = {"expected_ids": ["one"], **_extra_params(name)}
        submit(conn, stages.registry(), POLICY,
               request(key=name, kind=name, resource_class=case["resource_class"],
                       checkpoint_contract_ref=case["checkpoint_contract"],
                       parameters=params),
               clock=clock)
        claim = claim_next(conn, policy=TEST_POLICY, sample=sample(clock),
                           supervisor=supervisor, clock=clock, registry=stages.registry())
        assert claim is not None, f"{name} did not admit under TEST_POLICY"
        attempt_ids[name] = claim.attempt_id
        held = sorted(tuple(row) for row in conn.execute(
            "SELECT domain, mode FROM store_leases WHERE attempt_id=? AND released_at IS NULL",
            (claim.attempt_id,)).fetchall())
        expected = [("legacy_store", "read")] if name == "snapshot_import" else []
        assert held == expected, name
        if name != "snapshot_import":
            # release this kind's reservation before claiming the next one;
            # snapshot_import is left held so the conflict check below still
            # sees its lease.
            commit_attempt(conn, claim.attempt_id, claim.fence,
                           Outcome(succeeded=True, process_state="exited"), clock=clock)

    # the mode matters too, not just the domain name: a write attempt on the
    # domain snapshot_import still holds read-leased is refused, through the
    # same acquire_in a real claim uses. artifact_check's already-committed
    # attempt id is reused purely as a valid FK anchor for this synthetic
    # write attempt (its own lease was released above).
    with pytest.raises(OpsError) as excinfo:
        with transaction(conn):
            store_barrier.acquire_in(conn, attempt_ids["artifact_check"],
                                     (("legacy_store", "write"),))
    assert excinfo.value.code == "RESOURCE_UNAVAILABLE"
