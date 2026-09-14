"""Real 2026-09-14 defect: checkpointed attempts recorded their outputs under
positional names ('0', '1', ...) instead of the worker's declared names, so a
downstream ``job_<id>#<name>`` input binding could not find its parent's
output (``INPUT_CHANGED``: "input binding names an output its parent did not
produce"). Root causes, both in ``engine/v2/ops/supervisor.py``:

1. ``Service._checkpoint_refs`` discarded the ``OutputCandidate`` names it was
   given and enumerated the committed ``artifact_refs`` instead
   (``[(str(index), ref) for index, ref in enumerate(...)]``).
2. ``Service._reuse_staged_checkpoint`` paired a ``SELECT name ... ORDER BY
   name`` from the producer attempt against ``receipt.artifact_refs`` BY
   INDEX -- silently swapping names onto the wrong artifacts whenever a
   kind's output order was not alphabetical, and happily reusing (and thus
   propagating) a pre-fix catalog's positional names.

This file is the real-catalog/real-``Service`` proof for the fix, following
the patterns in ``tests/test_v2_ops_nightly_completion.py`` (``artifact_check``
as a cheap, real, checkpoint-contract kind) and
``tests/test_v2_ops_recovery_ownership.py`` (real ``Service``/``claim_next``
without a live subprocess).
"""
from __future__ import annotations

from pathlib import Path

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import SystemClock
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.checkpoints import artifact, register_artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.input_bindings import resolve_and_record
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, job_id_for, submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY, sample

REPO = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


# --------------------------------------------------------------------------
# shared fixtures
# --------------------------------------------------------------------------


def _service(tmp_path, name):
    """A real ``Service`` over a fresh catalog + artifact store -- no
    subprocess is ever launched in this file; every checkpoint/reuse call
    below invokes the real supervisor method directly, against real staged
    files and a real SQLite catalog."""
    root = tmp_path / name
    root.mkdir()
    store_root = root / "prod"
    store_root.mkdir()
    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    service = Service(conn, root, registry(), TEST_POLICY, clock=clock,
                      code_source=REPO, store_root=store_root)
    service.start()
    return service, conn, clock


def _submit_and_claim(conn, clock, service, *, key, kind="artifact_check",
                      implementation_ref="test-impl", environment_ref="test-env",
                      parameters=None, dependency_job_ids=(), input_refs=(),
                      checkpoint_contract_ref="receipt.v1.0"):
    job = JobSpec(kind=kind, implementation_ref=implementation_ref, spec_hash=None,
                  environment_ref=environment_ref, parameters=parameters or {"expected_ids": ()},
                  input_refs=input_refs, dependency_job_ids=dependency_job_ids,
                  output_namespace="shadow", resource_class="delivery",
                  retry_policy_ref="bounded", checkpoint_contract_ref=checkpoint_contract_ref)
    submit(conn, registry(), POLICY, SubmitRequest(
        namespace="shadow", idempotency_key=key, principal="operator", job=job), clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                       supervisor=service.identity, clock=clock, registry=registry())
    assert claim is not None, f"job {key!r} did not admit"
    return job_id_for("shadow", key), claim


def _stage(service, attempt_id, files):
    """Write ``{filename: bytes}`` into the attempt's real staging dir and
    return the ``outputs`` list ``_checkpoint_refs`` expects, in the SAME
    (possibly non-alphabetical) order ``files`` was given."""
    staging = service.store.staging_dir(attempt_id)
    outputs = []
    for name, filename, schema, data in files:
        (staging / filename).write_bytes(data)
        outputs.append({"name": name, "path": filename, "schema": schema})
    return outputs


def _commit_named_outputs(conn, clock, claim, refs):
    """Exactly what ``Service._commit_success``'s own ``effects`` closure does
    for the non-effect-kind branch: register each artifact and record its
    ``attempt_outputs`` row under the given name."""
    def effects(inner_conn):
        for name, ref in refs:
            register_artifact(inner_conn, ref, claim.attempt_id, clock)
            inner_conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                               (claim.attempt_id, name, ref.artifact_id))
    commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                   clock=clock, effects=effects)


def _commit_positional_outputs(conn, clock, claim, refs):
    """Simulate a PRE-FIX catalog: the producer's attempt_outputs rows are
    named by enumeration index ('0', '1', ...), bypassing the (now-fixed)
    ``_checkpoint_refs`` naming entirely via a direct SQL insert -- exactly
    what the old ``[(str(index), ref) for index, ref in enumerate(...)]``
    line used to record."""
    def effects(inner_conn):
        for index, (_, ref) in enumerate(refs):
            register_artifact(inner_conn, ref, claim.attempt_id, clock)
            inner_conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                               (claim.attempt_id, str(index), ref.artifact_id))
    commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                   clock=clock, effects=effects)


# --------------------------------------------------------------------------
# 1. fresh checkpoint: worker-declared names, non-alphabetical order
# --------------------------------------------------------------------------


def test_checkpoint_records_worker_declared_names_not_enumeration_index(tmp_path):
    service, conn, clock = _service(tmp_path, "case1")
    _, claim = _submit_and_claim(conn, clock, service, key="producer")
    outputs = _stage(service, claim.attempt_id, [
        ("zeta", "zeta.json", "receipt.v1.0", b'{"which": "zeta"}'),
        ("alpha", "alpha.json", "receipt.v1.0", b'{"which": "alpha"}'),
    ])

    refs = service._checkpoint_refs(claim, outputs, None)

    names = [name for name, _ in refs]
    assert names == ["zeta", "alpha"], "names must be the worker's, in the worker's order"
    by_name = dict(refs)
    assert set(by_name) == {"zeta", "alpha"}
    # By content, not only by name: each name's artifact holds ITS bytes.
    assert service.store.read_verified(by_name["zeta"]) == b'{"which": "zeta"}'
    assert service.store.read_verified(by_name["alpha"]) == b'{"which": "alpha"}'


# --------------------------------------------------------------------------
# 2. a child job's job#name binding resolves and reads the right bytes
# --------------------------------------------------------------------------


def test_child_binding_reads_named_outputs_bytes_not_the_other_ones(tmp_path):
    service, conn, clock = _service(tmp_path, "case2")
    producer_job_id, producer_claim = _submit_and_claim(conn, clock, service, key="producer")
    outputs = _stage(service, producer_claim.attempt_id, [
        ("zeta", "zeta.json", "receipt.v1.0", b'{"which": "zeta"}'),
        ("alpha", "alpha.json", "receipt.v1.0", b'{"which": "alpha"}'),
    ])
    refs = service._checkpoint_refs(producer_claim, outputs, None)
    _commit_named_outputs(conn, clock, producer_claim, refs)

    _, child_claim = _submit_and_claim(
        conn, clock, service, key="child",
        parameters={"expected_ids": (),
                    "input_bindings": {"in.json": producer_job_id + "#zeta"}},
        dependency_job_ids=(producer_job_id,))

    resolved = resolve_and_record(conn, service.store, child_claim)
    binding = resolved["in.json"]
    ref = artifact(conn, service.store, binding.artifact_id)
    assert service.store.read_verified(ref) == b'{"which": "zeta"}'
    assert service.store.read_verified(ref) != b'{"which": "alpha"}'


# --------------------------------------------------------------------------
# 3. reuse: same cache key, same (non-alphabetical) name -> artifact mapping
# --------------------------------------------------------------------------


def test_reuse_maps_names_by_artifact_identity_non_alphabetical(tmp_path):
    service, conn, clock = _service(tmp_path, "case3")
    shared = dict(implementation_ref="impl-reuse", environment_ref="env-reuse",
                  parameters={"expected_ids": ()})

    _, producer_claim = _submit_and_claim(conn, clock, service, key="reuse-producer", **shared)
    outputs = _stage(service, producer_claim.attempt_id, [
        ("zeta", "zeta.json", "receipt.v1.0", b'{"which": "zeta"}'),
        ("alpha", "alpha.json", "receipt.v1.0", b'{"which": "alpha"}'),
    ])
    refs = service._checkpoint_refs(producer_claim, outputs, None)
    _commit_named_outputs(conn, clock, producer_claim, refs)

    _, consumer_claim = _submit_and_claim(conn, clock, service, key="reuse-consumer", **shared)
    assert service._reuse_staged_checkpoint(consumer_claim, None) is True

    rows = conn.execute("SELECT name, artifact_id FROM attempt_outputs WHERE attempt_id = ?",
                        (consumer_claim.attempt_id,)).fetchall()
    got = {name: artifact_id for name, artifact_id in rows}
    producer_by_name = {name: ref.artifact_id for name, ref in refs}
    assert got == producer_by_name
    assert got["zeta"] != got["alpha"]
    # By content again: the reused mapping is the right way round.
    zeta_ref = artifact(conn, service.store, got["zeta"])
    assert service.store.read_verified(zeta_ref) == b'{"which": "zeta"}'


# --------------------------------------------------------------------------
# 4. reuse refusal: a positionally-named producer (pre-fix catalog) is never
#    reused, and never propagates '0'/'1' onto the new attempt
# --------------------------------------------------------------------------


def test_reuse_refuses_a_positionally_named_producer(tmp_path):
    service, conn, clock = _service(tmp_path, "case4")
    shared = dict(implementation_ref="impl-positional", environment_ref="env-positional",
                  parameters={"expected_ids": ()})

    _, producer_claim = _submit_and_claim(conn, clock, service, key="pos-producer", **shared)
    outputs = _stage(service, producer_claim.attempt_id, [
        ("zeta", "zeta.json", "receipt.v1.0", b'{"which": "zeta"}'),
        ("alpha", "alpha.json", "receipt.v1.0", b'{"which": "alpha"}'),
    ])
    refs = service._checkpoint_refs(producer_claim, outputs, None)
    _commit_positional_outputs(conn, clock, producer_claim, refs)
    # Sanity: the simulated old catalog really does hold '0'/'1', not names.
    stored_names = {row[0] for row in conn.execute(
        "SELECT name FROM attempt_outputs WHERE attempt_id = ?", (producer_claim.attempt_id,))}
    assert stored_names == {"0", "1"}

    _, consumer_claim = _submit_and_claim(conn, clock, service, key="pos-consumer", **shared)
    assert service._reuse_staged_checkpoint(consumer_claim, None) is False

    rows = conn.execute("SELECT name FROM attempt_outputs WHERE attempt_id = ?",
                        (consumer_claim.attempt_id,)).fetchall()
    assert rows == [], "a refused reuse must record nothing -- the caller recomputes"


# --------------------------------------------------------------------------
# 5. nightly regression: a legacy_finality-shaped barrier action, followed
#    by a child job binding #legacy_finality, succeeds end to end
# --------------------------------------------------------------------------


def test_legacy_finality_shaped_checkpoint_then_child_binding_succeeds(tmp_path):
    """The real 2026-09-14 shape: ``legacy_finality``'s worker emits its
    primary output named after the action itself plus an ``extra`` output
    (``legacy_finality_coverage``) -- see ``worker.py`` dispatch and
    ``legacy_adapter._action_finality``. A downstream job (``legacy_score``/
    ``legacy_settlement`` in production) binds ``job_<id>#legacy_finality``.
    Before the fix this failed ``INPUT_CHANGED`` because the parent's
    ``attempt_outputs`` held '0'/'1', not 'legacy_finality'/
    'legacy_finality_coverage'.
    """
    service, conn, clock = _service(tmp_path, "case5")
    finality_job_id, finality_claim = _submit_and_claim(
        conn, clock, service, key="legacy-finality")
    outputs = _stage(service, finality_claim.attempt_id, [
        ("legacy_finality", "finality.json", "legacy_action.v1.0",
         b'{"date": "2026-09-14", "is_final": true}'),
        ("legacy_finality_coverage", "finality_coverage.json", "finality_coverage.v1.0",
         b'{"schema_version": "finality_coverage.v1.0", "covered_tickers": ["FAKE"]}'),
    ])
    refs = service._checkpoint_refs(finality_claim, outputs, None)
    names = [name for name, _ in refs]
    assert names == ["legacy_finality", "legacy_finality_coverage"]
    _commit_named_outputs(conn, clock, finality_claim, refs)

    _, child_claim = _submit_and_claim(
        conn, clock, service, key="legacy-score",
        parameters={"expected_ids": (),
                    "input_bindings": {
                        "finality.json": finality_job_id + "#legacy_finality",
                        "finality_coverage.json": finality_job_id + "#legacy_finality_coverage"}},
        dependency_job_ids=(finality_job_id,))

    # Before the fix, this raised OpsError(INPUT_CHANGED, "input binding
    # names an output its parent did not produce").
    resolved = resolve_and_record(conn, service.store, child_claim)
    assert service.store.read_verified(
        artifact(conn, service.store, resolved["finality.json"].artifact_id)
    ) == b'{"date": "2026-09-14", "is_final": true}'
    assert service.store.read_verified(
        artifact(conn, service.store, resolved["finality_coverage.json"].artifact_id)
    ) == b'{"schema_version": "finality_coverage.v1.0", "covered_tickers": ["FAKE"]}'


# --------------------------------------------------------------------------
# 6. supervisor safety net: a checkpoint output-count mismatch is refused
#    typed rather than silently mispairing names to artifacts
# --------------------------------------------------------------------------


def test_checkpoint_output_count_mismatch_is_refused_typed(tmp_path, monkeypatch):
    """Deliverable 1's "refuse typed if the counts differ": force
    ``commit_checkpoint`` to report fewer artifact_refs than declared
    outputs (never true along the real code path -- see the comment in
    ``_checkpoint_refs`` -- but the safety net must still fire if it ever
    stops holding)."""
    import engine.v2.ops.supervisor as supervisor_module

    service, conn, clock = _service(tmp_path, "case6")
    _, claim = _submit_and_claim(conn, clock, service, key="mismatch")
    outputs = _stage(service, claim.attempt_id, [
        ("zeta", "zeta.json", "receipt.v1.0", b'{"which": "zeta"}'),
        ("alpha", "alpha.json", "receipt.v1.0", b'{"which": "alpha"}'),
    ])

    real_commit_checkpoint = supervisor_module.commit_checkpoint

    class _Truncated:
        def __init__(self, receipt):
            self._receipt = receipt
            self.artifact_refs = receipt.artifact_refs[:1]

        def __getattr__(self, item):
            return getattr(self._receipt, item)

    def _fake_commit_checkpoint(*args, **kwargs):
        return _Truncated(real_commit_checkpoint(*args, **kwargs))

    monkeypatch.setattr(supervisor_module, "commit_checkpoint", _fake_commit_checkpoint)

    try:
        service._checkpoint_refs(claim, outputs, None)
        raise AssertionError("expected a CHECKPOINT_INCOMPATIBLE refusal")
    except OpsError as exc:
        assert exc.code == "CHECKPOINT_INCOMPATIBLE"
