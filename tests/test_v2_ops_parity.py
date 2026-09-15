"""D15 loading: ``engine.v2.ops.parity.load_score_parity_inputs`` — two real
committed ``legacy_score`` jobs (no subprocess, no legacy tree, no provider
calls: every job is driven straight through the real submit/claim/
``commit_attempt`` machinery with a synthetic ``score.json`` artifact
registered as its output, mirroring
``tests/test_v2_ops_nightly_completion.py``'s ``_succeed_parent`` pattern),
read back as plain rows plus the snapshot binding.

The comparison itself (``compare_records``, tolerances, ``ComparisonReceipt``)
lives at ``checks/rearchitecture_phase2_parity.py`` and is tested there —
this module never re-derives or exercises field comparison logic.
"""
from __future__ import annotations

import json

import pytest

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import ArtifactStore, to_document
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.input_bindings import ResolvedBinding, record_resolved_bindings
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from engine.v2.ops.parity import load_score_parity_inputs
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, job_id_for, submit
from tests.ops_support import catalog as ops_catalog
from tests.ops_support import sample
from tests.test_checks_phase2_gate import _snapshot_ref

POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


def _row(row_id, **fields):
    base = {"row_id": row_id, "ticker": "FAKE", "strategy": "TWIN-P",
           "event_date": "2026-09-10", "strike": 100.0, "expiry": "2026-10-16",
           "iv30": 0.42}
    base.update(fields)
    return base


def _score_doc(rows, *, session="2026-09-10", tickers=("FAKE",)):
    keys = [f'{r["ticker"]}|{r["strategy"]}|{r["event_date"]}' for r in rows]
    return {"rows": rows, "expected_population": keys, "observed_population": sorted(keys),
           "ladder": {}, "tickers": list(tickers), "analog_entry_coverage": {},
           "session": session, "requested_session": session}


def _submit_score_job(conn, clock, supervisor, store, *, key, rows, session="2026-09-10",
                      tickers=("FAKE",), horizon_days=35, snapshot_ref=None, input_mode=None,
                      effect_scope=""):
    """Submit, claim and synthetically succeed one ``legacy_score`` job: a
    real row in ``jobs``/``attempts``, a real ``score.json`` artifact
    registered under ``attempt_outputs`` name ``legacy_score``, and, when
    ``snapshot_ref`` is given, a real ``attempt_input_bindings`` row for
    ``snapshot_ref.json``."""
    doc = _score_doc(rows, session=session, tickers=tickers)
    parameters = {"expected_ids": ("legacy_score",), "session": session,
                 "tickers": tuple(tickers), "year_start": 2020, "year_end": 2026,
                 "horizon_days": horizon_days, "expected_population": tuple(doc["expected_population"]),
                 "effect_scope": effect_scope}
    if input_mode is not None:
        parameters["input_mode"] = input_mode
    if input_mode == "snapshot":
        # input_mode_problems (stages.py) requires all three snapshot
        # bindings present (and no mutable legacy_manifest.json) before it
        # will admit a snapshot-mode legacy_score job at submission at all;
        # the placeholder values are never resolved by this test.
        parameters["input_bindings"] = {"snapshot_ref.json": "snap#ref",
                                        "materialization_request.json": "snap#request",
                                        "materialization_manifest.json": "snap#manifest"}
    job = JobSpec(kind="legacy_score", implementation_ref="x", spec_hash=None, environment_ref="x",
                 parameters=parameters, output_namespace="shadow", resource_class="legacy_score",
                 retry_policy_ref="bounded", checkpoint_contract_ref="legacy_action.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(
        namespace="shadow", idempotency_key=key, principal="operator", job=job), clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=supervisor,
                       clock=clock, registry=registry())
    score_ref = store.publish_bytes(json.dumps(doc, sort_keys=True).encode(),
                                    schema_ref="legacy_action.v1.0")
    snap_ref_artifact = None
    if snapshot_ref is not None:
        snap_ref_artifact = store.publish_bytes(
            json.dumps(to_document(snapshot_ref), sort_keys=True).encode(),
            schema_ref="snapshot_ref.v1.0")

    def effects(inner_conn):
        register_artifact(inner_conn, score_ref, claim.attempt_id, clock)
        inner_conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                           (claim.attempt_id, "legacy_score", score_ref.artifact_id))
        if snap_ref_artifact is not None:
            register_artifact(inner_conn, snap_ref_artifact, claim.attempt_id, clock)
            record_resolved_bindings(inner_conn, claim.attempt_id, {
                "snapshot_ref.json": ResolvedBinding(
                    name="snapshot_ref.json", binding=snap_ref_artifact.artifact_id,
                    artifact_id=snap_ref_artifact.artifact_id,
                    content_hash=snap_ref_artifact.content_hash)})

    commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                   clock=clock, effects=effects)
    return job_id_for("shadow", key)


def _world(tmp_path, *, legacy_rows, snapshot_rows, legacy_session="2026-09-10",
          snapshot_session="2026-09-10", bind_snapshot=False, legacy_input_mode="legacy",
          snapshot_input_mode="snapshot", legacy_effect_scope="", snapshot_effect_scope=""):
    conn, clock, supervisor = ops_catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    snap_ref = _snapshot_ref("snap_current") if bind_snapshot else None
    legacy_job = _submit_score_job(conn, clock, supervisor, store, key="legacy",
                                   rows=legacy_rows, session=legacy_session,
                                   input_mode=legacy_input_mode, effect_scope=legacy_effect_scope)
    snapshot_job = _submit_score_job(conn, clock, supervisor, store, key="snapshot",
                                     rows=snapshot_rows, session=snapshot_session,
                                     snapshot_ref=snap_ref, input_mode=snapshot_input_mode,
                                     effect_scope=snapshot_effect_scope)
    return conn, store, legacy_job, snapshot_job, snap_ref


def _load(conn, store, legacy_job, snapshot_job):
    return load_score_parity_inputs(conn, store, legacy_job_id=legacy_job,
                                    snapshot_job_id=snapshot_job)


def test_loads_both_committed_score_documents_keyed_by_row_id(tmp_path):
    id_a = "FAKE|TWIN-P|2026-09-10|100.0|2026-10-16"
    id_b = "FAKE|TWIN-C|2026-09-10|100.0|2026-10-16"
    legacy = [_row(id_a), _row(id_b)]
    snapshot = [_row(id_a, iv30=0.5)]
    conn, store, legacy_job, snapshot_job, _ = _world(tmp_path, legacy_rows=legacy, snapshot_rows=snapshot)
    inputs = _load(conn, store, legacy_job, snapshot_job)
    assert inputs.legacy_job_id == legacy_job
    assert inputs.snapshot_job_id == snapshot_job
    assert set(inputs.legacy_rows) == {id_a, id_b}
    assert set(inputs.snapshot_rows) == {id_a}
    assert inputs.legacy_rows[id_a]["row_id"] == id_a
    assert inputs.snapshot_rows[id_a]["iv30"] == 0.5
    assert inputs.snapshot_ref is None


def test_session_mismatch_between_jobs_is_refused(tmp_path):
    rows = [_row("FAKE|TWIN-P|2026-09-10|100.0|2026-10-16")]
    conn, store, legacy_job, snapshot_job, _ = _world(
        tmp_path, legacy_rows=rows, snapshot_rows=rows,
        legacy_session="2026-09-10", snapshot_session="2026-09-11")
    with pytest.raises(OpsError) as err:
        _load(conn, store, legacy_job, snapshot_job)
    assert err.value.code == "INPUT_CHANGED"


def test_resolves_snapshot_binding_when_recorded(tmp_path):
    rows = [_row("FAKE|TWIN-P|2026-09-10|100.0|2026-10-16")]
    conn, store, legacy_job, snapshot_job, snap_ref = _world(
        tmp_path, legacy_rows=rows, snapshot_rows=rows, bind_snapshot=True)
    inputs = _load(conn, store, legacy_job, snapshot_job)
    assert inputs.snapshot_ref is not None
    assert inputs.snapshot_ref.snapshot_id == snap_ref.snapshot_id
    assert inputs.snapshot_ref.manifest_hash == snap_ref.manifest_hash


def test_missing_committed_score_is_refused(tmp_path):
    conn, clock, supervisor = ops_catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    rows = [_row("FAKE|TWIN-P|2026-09-10|100.0|2026-10-16")]
    legacy_job = _submit_score_job(conn, clock, supervisor, store, key="legacy", rows=rows,
                                   input_mode="legacy")
    with pytest.raises(OpsError) as err:
        _load(conn, store, legacy_job, "job_never_submitted")
    assert err.value.code == "INPUT_CHANGED"


# -- Finding #3: D15 self-parity (the reviewer's repro certified a snapshot
# job compared against itself). --------------------------------------------


def test_same_job_passed_as_both_sides_is_refused(tmp_path):
    """The reviewer's exact repro: pass one committed ``legacy_score`` job
    (run in snapshot mode) as BOTH ``legacy_job_id`` and ``snapshot_job_id``.
    Before the fix this produced ``agree`` over a nonzero compared
    population; it must now be refused before either job is even loaded."""
    conn, clock, supervisor = ops_catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    rows = [_row("FAKE|TWIN-P|2026-09-10|100.0|2026-10-16")]
    job = _submit_score_job(conn, clock, supervisor, store, key="only", rows=rows,
                            input_mode="snapshot")
    with pytest.raises(OpsError) as err:
        _load(conn, store, job, job)
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details.get("reason") == "SAME_JOB"


def test_legacy_side_not_actually_legacy_mode_is_refused(tmp_path):
    rows = [_row("FAKE|TWIN-P|2026-09-10|100.0|2026-10-16")]
    conn, store, legacy_job, snapshot_job, _ = _world(
        tmp_path, legacy_rows=rows, snapshot_rows=rows, legacy_input_mode="snapshot")
    with pytest.raises(OpsError) as err:
        _load(conn, store, legacy_job, snapshot_job)
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details.get("reason") == "LEGACY_MODE_MISMATCH"


def test_snapshot_side_not_actually_snapshot_mode_is_refused(tmp_path):
    rows = [_row("FAKE|TWIN-P|2026-09-10|100.0|2026-10-16")]
    conn, store, legacy_job, snapshot_job, _ = _world(
        tmp_path, legacy_rows=rows, snapshot_rows=rows, snapshot_input_mode="legacy")
    with pytest.raises(OpsError) as err:
        _load(conn, store, legacy_job, snapshot_job)
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details.get("reason") == "SNAPSHOT_MODE_MISMATCH"


def test_scope_mismatch_between_jobs_is_refused(tmp_path):
    rows = [_row("FAKE|TWIN-P|2026-09-10|100.0|2026-10-16")]
    conn, store, legacy_job, snapshot_job, _ = _world(
        tmp_path, legacy_rows=rows, snapshot_rows=rows,
        legacy_effect_scope="shadow", snapshot_effect_scope="FAKE")
    with pytest.raises(OpsError) as err:
        _load(conn, store, legacy_job, snapshot_job)
    assert err.value.code == "INPUT_CHANGED"
