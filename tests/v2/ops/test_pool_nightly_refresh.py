"""#192 slice 4: the pool-nightly refresh sidecar (``pool_refresh.py``).

Synthetic only: an in-memory ``sqlite3`` fake plus monkeypatched planners and
submission, so nothing here reads or writes a real model release store or
``data/``. Each test proves one step of the phase machine: finality/features
identity, the three training submitters, the ``phase5_state_stage`` stage, and the
``models_promote`` swap -- plus ``Service._reconcile_pool_nightly_refresh``'s
graceful degradation.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from checks.phase5_release_assembly import PRIOR_USABLE, PriorClass
from engine.v2.foundation import content_hash, to_document
from engine.v2.models import deployment
from engine.v2.ops import pool_refresh, supervisor, training
from engine.v2.ops.errors import OpsError, fail, make_problem
from engine.v2.ops.fingerprints import (
    CODE_ASSET_DIRS,
    CODE_ASSET_FILES,
    worker_source_manifest,
)
from engine.v2.ops.plans import request_from_plan as real_request_from_plan
from engine.v2.ops.stages import registry
from engine.v2.ops.state_stage import state_stage_plan
from engine.v2.ops.submission import NamespacePolicy, job_id_for, validate_request
from engine.v2.ops.supervisor import Service

REPO_ROOT = Path(__file__).resolve().parents[3]
SESSION = "2026-01-02"
PRIOR = "prior1"
MANIFEST = "manifest-1"
TAG = pool_refresh.manifest_tag(MANIFEST)


def _db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE jobs (job_id TEXT PRIMARY KEY, "
        "namespace TEXT NOT NULL DEFAULT 'shadow', kind TEXT, state TEXT, "
        "idempotency_key TEXT, spec_json TEXT, created_at TEXT, "
        "UNIQUE (namespace, idempotency_key));"
        "CREATE TABLE attempts (attempt_id TEXT, job_id TEXT, state TEXT, "
        "attempt_number INTEGER);"
        "CREATE TABLE attempt_outputs (attempt_id TEXT, name TEXT, artifact_id TEXT);")
    return conn


def _add_score(conn, *, session=SESSION, manifest=MANIFEST, scope="sha256:abc",
               created_at="2026-01-01T00:00:00Z", state="succeeded", stage="finality"):
    key = f"nightly:{session}:{scope}:{stage}"
    bindings = {} if manifest is None else {"legacy_manifest.json": manifest}
    spec = json.dumps({"parameters": {"input_bindings": bindings}})
    conn.execute("INSERT INTO jobs (job_id, kind, state, idempotency_key, spec_json, created_at) "
                 "VALUES (?, ?, ?, ?, ?, ?)",
                 (job_id_for("shadow", key), f"legacy_{stage}", state, key, spec, created_at))
    return key


def _add_score_spec(conn, *, key, created_at, spec_json, state="succeeded",
                    kind="legacy_finality"):
    conn.execute("INSERT INTO jobs (job_id, kind, state, idempotency_key, spec_json, created_at) "
                 "VALUES (?, ?, ?, ?, ?, ?)",
                 (job_id_for("shadow", key), kind, state, key, spec_json, created_at))


def _add_training(conn, phase, *, session=SESSION, prior=PRIOR, tag=TAG, state="queued"):
    key = pool_refresh.phase_key(session, phase, prior, tag)
    job_id = job_id_for("shadow", key)
    conn.execute("INSERT INTO jobs (job_id, kind, state, idempotency_key, spec_json, created_at) "
                 "VALUES (?, 'training', ?, ?, NULL, NULL)", (job_id, state, key))
    return job_id


def _add_attempt(conn, job_id, attempt_id, attempt_number, outputs, *, state="succeeded"):
    conn.execute("INSERT INTO attempts (attempt_id, job_id, state, attempt_number) "
                 "VALUES (?, ?, ?, ?)", (attempt_id, job_id, state, attempt_number))
    for name in outputs:
        conn.execute("INSERT INTO attempt_outputs (attempt_id, name, artifact_id) "
                     "VALUES (?, ?, 'a')", (attempt_id, name))
    return attempt_id


def _succeeded_training(conn, phase, outputs, *, session=SESSION, prior=PRIOR):
    job_id = _add_training(conn, phase, session=session, prior=prior, state="succeeded")
    _add_attempt(conn, job_id, f"att-{phase}", 1, outputs)
    return job_id


def _pointer_bytes(release_id):
    """The exact ``DEPLOYED`` encoding ``deployment._encode`` writes, via public APIs."""
    pointer = deployment.PointerState(
        sequence=1, release_id=release_id, previous_release_id=None,
        action="promote", at="2026-01-01T00:00:00Z")
    return json.dumps(to_document(pointer), indent=2, sort_keys=True).encode("utf-8")


def _write_pointer(release_root, release_id):
    deploy = Path(release_root) / "deployment"
    deploy.mkdir(parents=True, exist_ok=True)
    (deploy / "DEPLOYED").write_bytes(_pointer_bytes(release_id))


def _wire(monkeypatch, tmp_path, *, pointer=PRIOR, prior_class=None):
    """Patch every collaborator ``pool_refresh`` calls; records each call. The fake
    pointers carry no catalog, so the real classifier would report
    ``NO_POOL_CATALOG``; ``classify_prior`` is stubbed to ``USABLE`` to keep the
    carry-cycle assertions, and ``prior_class`` overrides that stub."""
    rec = SimpleNamespace(training=[], stage=[], promote=[], requests=[], submits=[])

    def training_plan(**kwargs):
        rec.training.append(kwargs)
        return {"kind": "training", "mode": "shadow", "parameters": dict(kwargs)}

    def promote_plan(**kwargs):
        rec.promote.append(kwargs)
        return {"kind": "promote", **kwargs}

    def stage_plan(**kwargs):
        rec.stage.append(kwargs)
        return {"kind": "phase5_state_stage", **kwargs}

    def request_from_plan(plan, key):
        rec.requests.append((plan, key))
        return ("request", plan, key)

    def submit(conn, registry, policy, request, *, clock):
        rec.submits.append(request)
        return "receipt"

    def classify_prior(_release_root, _prior_id):
        return (prior_class if prior_class is not None else
                PriorClass(PRIOR_USABLE, "the fake prior catalog is usable"))

    deploy = Path(tmp_path) / "deployment"
    pointer_file = deploy / "DEPLOYED"
    if pointer is None:
        pointer_file.unlink(missing_ok=True)
    else:
        _write_pointer(tmp_path, pointer)
    monkeypatch.setattr(pool_refresh, "classify_prior", classify_prior)
    monkeypatch.setattr(pool_refresh.training, "training_plan", training_plan)
    monkeypatch.setattr(pool_refresh.training, "promote_plan", promote_plan)
    monkeypatch.setattr(pool_refresh.state_stage, "state_stage_plan", stage_plan)
    monkeypatch.setattr(pool_refresh, "request_from_plan", request_from_plan)
    monkeypatch.setattr(pool_refresh, "submit", submit)
    monkeypatch.setattr(pool_refresh, "_implementation_ref", lambda code_source: "impl")
    return rec


def _submit(conn, tmp_path, *, code_source=REPO_ROOT):
    return pool_refresh.submit_pool_nightly_refresh_if_ready(
        conn, None, None, None, tmp_path, code_source=code_source, clock=None)


# --------------------------------------------------------------------------
# 1. pool_refresh_identity
# --------------------------------------------------------------------------


def test_identity_picks_the_latest_sessions_manifest():
    conn = _db()
    _add_score(conn, session="2026-01-01", manifest="old")
    _add_score(conn, session="2026-01-02", manifest="new")
    assert pool_refresh.pool_refresh_identity(conn) == ("2026-01-02", "new")


def test_identity_is_none_without_a_succeeded_score_job():
    conn = _db()
    _add_score(conn, state="queued")
    assert pool_refresh.pool_refresh_identity(conn) is None


def test_identity_is_none_when_the_binding_is_missing():
    conn = _db()
    _add_score(conn, manifest=None)
    assert pool_refresh.pool_refresh_identity(conn) is None


def test_manifest_tag_matches_a_literal_hashlib_oracle():
    expected = hashlib.sha256(b"sha256:abc").hexdigest()[:8]
    assert pool_refresh.manifest_tag("sha256:abc") == expected


@pytest.mark.parametrize("binding", [5, ""])
def test_identity_is_none_for_a_non_string_or_empty_binding(binding):
    conn = _db()
    _add_score(conn, manifest=binding)
    assert pool_refresh.pool_refresh_identity(conn) is None


def test_identity_from_a_features_only_session():
    conn = _db()
    _add_score(conn, stage="features", manifest="features-manifest")
    assert pool_refresh.pool_refresh_identity(conn) == (SESSION, "features-manifest")


def test_identity_is_none_for_a_legacy_score_row_alone():
    conn = _db()
    _add_score(conn, stage="score", manifest="score-manifest")
    assert pool_refresh.pool_refresh_identity(conn) is None


def test_identity_prefers_the_later_session_across_finality_and_features():
    conn = _db()
    _add_score(conn, session="2026-01-01", stage="finality", manifest="old")
    _add_score(conn, session="2026-01-02", stage="features", manifest="new")
    assert pool_refresh.pool_refresh_identity(conn) == ("2026-01-02", "new")


# --------------------------------------------------------------------------
# 2. nothing to do: no pointer, no conn, already-advanced pointer
# --------------------------------------------------------------------------


def test_no_pointer_no_conn_or_already_advanced_submits_nothing(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path, pointer=None)
    conn = _db()
    _add_score(conn)
    assert _submit(conn, tmp_path) == []
    assert _submit(None, tmp_path) == []

    rec2 = _wire(monkeypatch, tmp_path, pointer=f"pools-{SESSION}-{TAG}")
    assert _submit(conn, tmp_path) == []
    assert rec.submits == [] and rec2.submits == []


# --------------------------------------------------------------------------
# 2b. pool_cycle_pending: durable-state-only "is this cycle still running?"
# --------------------------------------------------------------------------


def test_pool_cycle_pending_true_with_identity_and_pointer(tmp_path):
    conn = _db()
    _add_score(conn)
    _write_pointer(tmp_path, PRIOR)
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is True


def test_pool_cycle_pending_true_while_a_training_job_runs(tmp_path):
    conn = _db()
    _add_score(conn)
    _write_pointer(tmp_path, PRIOR)
    _add_training(conn, "train-paired", state="running")
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is True


def test_pool_cycle_pending_false_when_the_release_is_deployed(tmp_path):
    conn = _db()
    _add_score(conn)
    _write_pointer(tmp_path, f"pools-{SESSION}-{TAG}")
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is False


@pytest.mark.parametrize("state", ["failed", "cancelled", "blocked"])
def test_pool_cycle_pending_false_when_a_phase_stalled(tmp_path, state):
    conn = _db()
    _add_score(conn)
    _write_pointer(tmp_path, PRIOR)
    _add_training(conn, "stage", state=state)
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is False


def test_pool_cycle_pending_true_while_the_promote_runs_after_the_pointer_moved(tmp_path):
    conn = _db()
    _add_score(conn)
    _write_pointer(tmp_path, f"pools-{SESSION}-{TAG}")
    _add_training(conn, "promote", state="running")
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is True


def test_pool_cycle_pending_false_when_the_promote_succeeded_and_pointer_moved(tmp_path):
    conn = _db()
    _add_score(conn)
    _write_pointer(tmp_path, f"pools-{SESSION}-{TAG}")
    _add_training(conn, "promote", state="succeeded")
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is False


def test_pool_cycle_pending_false_when_the_promote_failed(tmp_path):
    conn = _db()
    _add_score(conn)
    _write_pointer(tmp_path, f"pools-{SESSION}-{TAG}")
    _add_training(conn, "promote", state="failed")
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is False


def test_pool_cycle_pending_false_after_training_when_stage_and_promote_are_missing(
        tmp_path):
    conn = _db()
    _add_score(conn)
    _write_pointer(tmp_path, PRIOR)
    for phase in pool_refresh.TRAINING_PHASES:
        _add_training(conn, phase, state="succeeded")

    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is False


def test_pool_cycle_pending_false_without_a_pointer(tmp_path):
    conn = _db()
    _add_score(conn)
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is False


def test_pool_cycle_pending_false_without_identity_or_conn(tmp_path):
    conn = _db()
    _write_pointer(tmp_path, PRIOR)
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is False
    _add_score(conn)
    assert pool_refresh.pool_cycle_pending(None, tmp_path) is False


def test_pool_cycle_pending_false_when_the_pointer_is_malformed(tmp_path):
    conn = _db()
    _add_score(conn)
    deploy = tmp_path / "deployment"
    deploy.mkdir()
    (deploy / "DEPLOYED").write_bytes(b"{not json")
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is False


# --------------------------------------------------------------------------
# 3. all three training keys missing -> exactly three submissions
# --------------------------------------------------------------------------


def test_all_training_phases_missing_submit_three(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    assert _submit(conn, tmp_path) == ["receipt"] * 3
    assert [key for _plan, key in rec.requests] == [
        pool_refresh.phase_key(SESSION, phase, PRIOR, TAG)
        for phase in pool_refresh.TRAINING_PHASES]
    assert [key for _plan, key in rec.requests] == [
        f"nightly:{SESSION}:pool:{phase}:{TAG}:{PRIOR}"
        for phase in pool_refresh.TRAINING_PHASES]
    for call in rec.training:
        assert call["cutoffs"] == (SESSION,)
        assert call["manifest_ref"] == MANIFEST
    modes = {call["mode"]: call for call in rec.training}
    assert modes["state"]["state"] == "paired_residual_pool"
    assert "board_analog" in modes and "trailing_cutoff" in modes
    board = modes["board_analog"]
    assert board["alpha"] == pool_refresh.FILL_ALPHA == 0.5
    for plan, _key in rec.requests:
        assert plan["mode"] == "shadow"
        assert plan["implementation_ref"] == "impl"


def test_two_manifests_same_session_and_prior_use_distinct_tags(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn, manifest="old-manifest", created_at="2026-01-01T00:00:00Z")
    old_tag = hashlib.sha256(b"old-manifest").hexdigest()[:8]
    for phase in pool_refresh.TRAINING_PHASES:
        _add_training(conn, phase, tag=old_tag, state="succeeded")
    _add_score(conn, manifest="new-manifest", scope="sha256:zzz",
               created_at="2026-01-01T01:00:00Z")
    new_tag = hashlib.sha256(b"new-manifest").hexdigest()[:8]
    assert old_tag != new_tag
    assert (pool_refresh.phase_key(SESSION, "train-paired", PRIOR, new_tag)
            != pool_refresh.phase_key(SESSION, "train-paired", PRIOR, old_tag))

    # the old tag's succeeded jobs do not suppress submission under the new tag
    assert _submit(conn, tmp_path) == ["receipt"] * 3
    assert [key for _plan, key in rec.requests] == [
        pool_refresh.phase_key(SESSION, phase, PRIOR, new_tag)
        for phase in pool_refresh.TRAINING_PHASES]

    # the release id carries the tag: a pointer at the new tag's release is a no-op
    _wire(monkeypatch, tmp_path, pointer=f"pools-{SESSION}-{new_tag}")
    assert _submit(conn, tmp_path) == []
    # ... while the old tag's release id does not suppress the new tag's work
    _wire(monkeypatch, tmp_path, pointer=f"pools-{SESSION}-{old_tag}")
    assert _submit(conn, tmp_path) == ["receipt"] * 3


def test_phase_keys_differ_across_incumbents(monkeypatch, tmp_path):
    for phase in pool_refresh.PHASES:
        assert (pool_refresh.phase_key(SESSION, phase, "prior-A", TAG)
                != pool_refresh.phase_key(SESSION, phase, "prior-B", TAG))
    rec = _wire(monkeypatch, tmp_path, pointer="prior-B")
    conn = _db()
    _add_score(conn)
    for phase in pool_refresh.TRAINING_PHASES:
        _add_training(conn, phase, prior="prior-A", state="succeeded")
    assert _submit(conn, tmp_path) == ["receipt"] * 3
    assert [key for _plan, key in rec.requests] == [
        pool_refresh.phase_key(SESSION, phase, "prior-B", TAG)
        for phase in pool_refresh.TRAINING_PHASES]


# --------------------------------------------------------------------------
# 4. an existing key (even failed) is never resubmitted
# --------------------------------------------------------------------------


def test_existing_failed_training_job_is_not_resubmitted(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    _add_training(conn, "train-paired", state="failed")
    assert _submit(conn, tmp_path) == ["receipt"] * 2
    assert [key for _plan, key in rec.requests] == [
        pool_refresh.phase_key(SESSION, "train-board", PRIOR, TAG),
        pool_refresh.phase_key(SESSION, "train-trailing", PRIOR, TAG)]


# --------------------------------------------------------------------------
# 5. all keys exist but one is still running -> wait
# --------------------------------------------------------------------------


def test_waiting_while_a_training_job_runs(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    _add_training(conn, "train-paired", state="succeeded")
    _add_training(conn, "train-board", state="succeeded")
    _add_training(conn, "train-trailing", state="running")
    assert _submit(conn, tmp_path) == []
    assert rec.submits == []


def test_plans_are_built_lazily_only_for_missing_phases(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    for phase in pool_refresh.TRAINING_PHASES:
        _add_training(conn, phase, state="running")

    def _boom(**kwargs):
        raise AssertionError("training_plan must not be built when nothing is missing")

    monkeypatch.setattr(pool_refresh.training, "training_plan", _boom)
    assert _submit(conn, tmp_path) == []


# --------------------------------------------------------------------------
# 6. all training succeeded -> one stage carrying only plain state refs
# --------------------------------------------------------------------------


def test_all_training_succeeded_submits_the_stage_with_plain_refs(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    board_file = "board_analog_matcher__STR-THRU__0.5000__2026-10-01.json"
    paired = _succeeded_training(
        conn, "train-paired",
        ["training/paired_residual_pool.json", "training/paired_residual_pool.summary.json"])
    board = _succeeded_training(
        conn, "train-board",
        [f"training/{board_file}", "training/board_analog_matcher.summary.json"])
    trailing = _succeeded_training(
        conn, "train-trailing", ["training/trailing_pnl_cutoff__2026-10.json"])

    assert _submit(conn, tmp_path) == ["receipt"]
    assert len(rec.stage) == 1
    stage = rec.stage[0]
    assert stage["prior_release_id"] == PRIOR
    assert stage["release_id"] == f"pools-{SESSION}-{TAG}"
    assert stage["cutoff"] == SESSION
    assert stage["release_root"] == str(tmp_path)
    assert stage["manifest_ref"] == MANIFEST
    assert stage["source_root"] == REPO_ROOT
    assert stage["state_refs"] == {
        "paired_residual_pool.json": f"{paired}#training/paired_residual_pool.json",
        board_file: f"{board}#training/{board_file}",
        "trailing_pnl_cutoff__2026-10.json":
            f"{trailing}#training/trailing_pnl_cutoff__2026-10.json"}
    assert "board_analog_matcher.summary.json" not in stage["state_refs"]
    assert rec.requests[-1][1] == pool_refresh.phase_key(SESSION, "stage", PRIOR, TAG)


def test_state_refs_use_only_the_latest_succeeded_attempt(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    job_id = _add_training(conn, "train-paired", state="succeeded")
    _add_attempt(conn, job_id, "att-1", 1, ["training/paired_residual_pool__old.json"])
    _add_attempt(conn, job_id, "att-2", 2, ["training/paired_residual_pool.json"])
    _succeeded_training(conn, "train-board", ["training/board.json"])
    _succeeded_training(conn, "train-trailing", ["training/trailing.json"])

    assert _submit(conn, tmp_path) == ["receipt"]
    refs = rec.stage[0]["state_refs"]
    assert refs["paired_residual_pool.json"] == (
        f"{job_id}#training/paired_residual_pool.json")
    assert "paired_residual_pool__old.json" not in refs


def test_state_refs_exclude_wrong_prefix_suffix_and_failed_attempts(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    paired = _add_training(conn, "train-paired", state="succeeded")
    _add_attempt(conn, paired, "att-paired", 1,
                 ["training/paired.json", "other/x.json", "training/x.txt"])
    _succeeded_training(conn, "train-board", ["training/board.json"])
    trailing = _add_training(conn, "train-trailing", state="succeeded")
    _add_attempt(conn, trailing, "att-trailing", 1, ["training/trailing.json"])
    _add_attempt(conn, trailing, "att-failed", 2, ["training/y.json"], state="failed")

    assert _submit(conn, tmp_path) == ["receipt"]
    refs = rec.stage[0]["state_refs"]
    assert set(refs) == {"paired.json", "board.json", "trailing.json"}
    assert "x.json" not in refs and "y.json" not in refs


# --------------------------------------------------------------------------
# 7. a duplicate state name or a missing succeeded attempt refuses
# --------------------------------------------------------------------------


def test_duplicate_state_file_name_raises(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    _succeeded_training(conn, "train-paired", ["training/paired.json"])
    _succeeded_training(conn, "train-board", ["training/paired.json"])
    _succeeded_training(conn, "train-trailing", ["training/trailing.json"])
    with pytest.raises(OpsError) as err:
        _submit(conn, tmp_path)
    assert err.value.code == "VALIDATION_FAILED"
    assert err.value.problem.message == "a state file name appears twice"
    assert err.value.problem.details == {"name": "paired.json"}


def test_training_job_without_a_succeeded_attempt_raises(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    job_id = _add_training(conn, "train-paired", state="succeeded")  # no attempt row
    _succeeded_training(conn, "train-board", ["training/board.json"])
    _succeeded_training(conn, "train-trailing", ["training/trailing.json"])
    with pytest.raises(OpsError) as err:
        _submit(conn, tmp_path)
    assert err.value.code == "VALIDATION_FAILED"
    assert err.value.problem.message == "a training job has no succeeded attempt"
    assert err.value.problem.details == {"job_id": job_id}


# --------------------------------------------------------------------------
# 8. stage / promote sequencing
# --------------------------------------------------------------------------


def _all_training_succeeded(conn):
    for phase in pool_refresh.TRAINING_PHASES:
        _add_training(conn, phase, state="succeeded")


def test_stage_that_is_not_yet_succeeded_waits(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    _all_training_succeeded(conn)
    _add_training(conn, "stage", state="running")
    assert _submit(conn, tmp_path) == []
    assert rec.submits == [] and rec.promote == []


def test_failed_stage_job_blocks_promote(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    _all_training_succeeded(conn)
    _add_training(conn, "stage", state="failed")
    assert _submit(conn, tmp_path) == []
    assert rec.promote == [] and rec.submits == []


def test_stage_succeeded_submits_a_guarded_promote(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    _all_training_succeeded(conn)
    _add_training(conn, "stage", state="succeeded")

    assert _submit(conn, tmp_path) == ["receipt"]
    assert len(rec.promote) == 1
    assert rec.promote[0]["expected_previous_release_id"] == PRIOR
    assert rec.promote[0]["release_id"] == f"pools-{SESSION}-{TAG}"
    assert rec.promote[0]["release_root"] == str(Path(tmp_path) / "deployment")
    assert rec.requests[-1][1] == pool_refresh.phase_key(SESSION, "promote", PRIOR, TAG)


def test_existing_promote_is_idempotent(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    _all_training_succeeded(conn)
    _add_training(conn, "stage", state="succeeded")
    _add_training(conn, "promote", state="queued")
    assert _submit(conn, tmp_path) == []
    assert rec.submits == []


def test_promote_request_pins_the_code_source_implementation(monkeypatch, tmp_path):
    recorded = []
    monkeypatch.setattr(pool_refresh, "classify_prior",
                        lambda _root, _pid: PriorClass(PRIOR_USABLE,
                                                     "fake prior catalog is usable"))
    monkeypatch.setattr(pool_refresh.deployment, "current_pointer",
                        lambda root: SimpleNamespace(release_id=PRIOR))
    monkeypatch.setattr(
        pool_refresh, "submit",
        lambda conn, registry, policy, request, *, clock: recorded.append(request) or "receipt")
    conn = _db()
    _add_score(conn)
    _all_training_succeeded(conn)
    _add_training(conn, "stage", state="succeeded")
    _write_pointer(tmp_path, PRIOR)

    assert _submit(conn, tmp_path) == ["receipt"]
    expected = content_hash(worker_source_manifest(REPO_ROOT))
    assert recorded[0].job.implementation_ref == expected


# --------------------------------------------------------------------------
# 9. request_from_plan accepts a real state_stage_plan document
# --------------------------------------------------------------------------


def test_real_request_from_plan_accepts_a_state_stage_plan(tmp_path):
    plan = state_stage_plan(release_root=str(tmp_path), prior_release_id="a", release_id="b",
                            cutoff="2026-10-01",
                            state_refs={"x.json": "job_abc#training/x.json"},
                            manifest_ref=MANIFEST, source_root=REPO_ROOT)
    request = real_request_from_plan(plan, "nightly:2026-10-01:pool:stage:a")
    assert request.job.kind == "phase5_state_stage"
    validate_request(registry(), NamespacePolicy({"operator": frozenset({"shadow"})}), request)
    assert request.job.dependency_job_ids == ("job_abc",)
    assert request.job.input_refs == (MANIFEST,)
    assert request.job.parameters["input_bindings"]["legacy_manifest.json"] == MANIFEST
    assert request.job.retry_policy_ref == "none"


def test_real_request_from_plan_separates_job_refs_from_direct_artifact_refs(tmp_path):
    plan = state_stage_plan(release_root=str(tmp_path), prior_release_id="a", release_id="b",
                            cutoff="2026-10-01",
                            state_refs={"x.json": "job_a#training/x.json",
                                        "y.json": "artifact_1"},
                            manifest_ref=MANIFEST, source_root=REPO_ROOT)
    request = real_request_from_plan(plan, "nightly:2026-10-01:pool:stage:a")
    assert request.job.dependency_job_ids == ("job_a",)
    assert request.job.input_refs == ("artifact_1", MANIFEST)


def test_real_request_from_plan_maps_every_job_ref_to_a_dependency(tmp_path):
    refs = {"p.json": "job_a#training/p.json", "b.json": "job_b#training/b.json",
            "t.json": "job_c#training/t.json"}
    plan = state_stage_plan(release_root=str(tmp_path), prior_release_id="a", release_id="b",
                            cutoff="2026-10-01", state_refs=refs,
                            manifest_ref=MANIFEST, source_root=REPO_ROOT)
    request = real_request_from_plan(plan, "nightly:2026-10-01:pool:stage:a")
    assert request.job.dependency_job_ids == ("job_a", "job_b", "job_c")
    assert request.job.input_refs == (MANIFEST,)


# --------------------------------------------------------------------------
# 10. Service._reconcile_pool_nightly_refresh degrades gracefully
# --------------------------------------------------------------------------


def test_reconcile_swallows_missing_root_and_reports_once(monkeypatch, capsys):
    service = SimpleNamespace(conn=None, registry=None, store=None, code_source=None,
                              clock=None, _last_pool_refresh_problem=None)

    def _missing():
        raise deployment.MissingReleaseRoot()

    monkeypatch.setattr(deployment, "production_release_root", _missing)
    assert Service._reconcile_pool_nightly_refresh(service) is False
    assert capsys.readouterr().out == ""
    assert service._last_pool_refresh_problem is None

    monkeypatch.setattr(deployment, "production_release_root", lambda: "/somewhere")

    planted = fail("INPUT_CHANGED", "planted pool refresh failure")

    def _boom(*args, **kwargs):
        raise planted

    monkeypatch.setattr(pool_refresh, "submit_pool_nightly_refresh_if_ready", _boom)
    Service._reconcile_pool_nightly_refresh(service)
    service._pool_refresh_not_before = 0.0
    Service._reconcile_pool_nightly_refresh(service)
    assert service._last_pool_refresh_problem == ("INPUT_CHANGED", "planted pool refresh failure")
    out = capsys.readouterr().out
    assert out.count("pool_nightly_refresh_reconcile_failed") == 1
    assert "INPUT_CHANGED" in out and "planted pool refresh failure" in out
    document = to_document(planted.problem)
    payload = next(json.loads(line) for line in out.splitlines()
                   if line.startswith("{") and "pool_nightly_refresh_reconcile_failed" in line)
    assert payload["event"] == "pool_nightly_refresh_reconcile_failed"
    for field in ("code", "category", "retryable"):
        assert payload["problem"][field] == document[field]

    monkeypatch.setattr(pool_refresh, "submit_pool_nightly_refresh_if_ready",
                        lambda *args, **kwargs: [])
    service._pool_refresh_not_before = 0.0
    Service._reconcile_pool_nightly_refresh(service)
    assert service._last_pool_refresh_problem is None


def test_reconcile_pauses_after_a_refusal(monkeypatch):
    service = SimpleNamespace(conn=None, registry=None, store=None, code_source=None,
                              clock=None, _last_pool_refresh_problem=None)
    monkeypatch.setattr(deployment, "production_release_root", lambda: "/somewhere")
    calls = []

    def _boom(*args, **kwargs):
        calls.append(1)
        raise fail("INPUT_CHANGED", "planted pool refresh failure")

    monkeypatch.setattr(pool_refresh, "submit_pool_nightly_refresh_if_ready", _boom)

    assert Service._reconcile_pool_nightly_refresh(service) is False
    assert service._pool_refresh_not_before > time.monotonic()
    assert len(calls) == 1

    assert Service._reconcile_pool_nightly_refresh(service) is False
    assert len(calls) == 1


def test_reconcile_runs_again_after_the_pause_expires(monkeypatch):
    service = SimpleNamespace(conn=None, registry=None, store=None, code_source=None,
                              clock=None, _last_pool_refresh_problem=None)
    monkeypatch.setattr(deployment, "production_release_root", lambda: "/somewhere")
    calls = []

    def _boom(*args, **kwargs):
        calls.append(1)
        raise fail("INPUT_CHANGED", "planted pool refresh failure")

    monkeypatch.setattr(pool_refresh, "submit_pool_nightly_refresh_if_ready", _boom)
    Service._reconcile_pool_nightly_refresh(service)
    assert len(calls) == 1

    service._pool_refresh_not_before = 0.0
    Service._reconcile_pool_nightly_refresh(service)
    assert len(calls) == 2


def test_reconcile_clean_pass_clears_the_pause(monkeypatch):
    service = SimpleNamespace(conn=None, registry=None, store=None, code_source=None,
                              clock=None, _last_pool_refresh_problem=None,
                              _pool_refresh_not_before=-1.0)
    monkeypatch.setattr(deployment, "production_release_root", lambda: "/somewhere")
    monkeypatch.setattr(pool_refresh, "submit_pool_nightly_refresh_if_ready",
                        lambda *args, **kwargs: [])

    assert Service._reconcile_pool_nightly_refresh(service) is False
    assert service._pool_refresh_not_before == 0.0


def test_reconcile_missing_root_does_not_pause(monkeypatch):
    service = SimpleNamespace(conn=None, registry=None, store=None, code_source=None,
                              clock=None, _last_pool_refresh_problem=None)

    def _missing():
        raise deployment.MissingReleaseRoot()

    monkeypatch.setattr(deployment, "production_release_root", _missing)
    assert Service._reconcile_pool_nightly_refresh(service) is False
    assert getattr(service, "_pool_refresh_not_before", 0.0) == 0.0


def test_tick_calls_reconcile_pool_nightly_refresh(monkeypatch):
    monkeypatch.setattr(supervisor, "claim_next", lambda *args, **kwargs: None)
    monkeypatch.setattr(supervisor, "expire_leases", lambda *args, **kwargs: None)
    monkeypatch.setattr(supervisor, "sample_capacity", lambda *args, **kwargs: None)
    fake_self = SimpleNamespace(
        _clock_check=lambda: None,
        _reclaim_terminal_staging=lambda: None,
        reconcile=lambda: None,
        running={},
        conn=None,
        clock=None,
        root=None,
        identity=None,
        policy=None,
        registry=None,
        _reconcile_publication_status=lambda: None,
        _reconcile_computed_moves_refresh=lambda: None,
        _reconcile_native_score_batch_shadow=lambda: None,
        _reconcile_native_parity=lambda: None,
    )
    fake_self._reconcile_pool_nightly_refresh = lambda: True
    assert Service.tick(fake_self) is True
    fake_self._reconcile_pool_nightly_refresh = lambda: False
    assert Service.tick(fake_self) is False


# --------------------------------------------------------------------------
# 11. identity: unparseable key and blank created_at rows are ignored
# --------------------------------------------------------------------------


def _raw_score(conn, job_id, key, created_at, manifest, *, kind="legacy_finality"):
    spec = json.dumps({"parameters": {"input_bindings": {"legacy_manifest.json": manifest}}})
    conn.execute("INSERT INTO jobs (job_id, kind, state, idempotency_key, spec_json, created_at) "
                 "VALUES (?, ?, 'succeeded', ?, ?, ?)",
                 (job_id, kind, key, spec, created_at))


def test_identity_ignores_the_unparseable_key_and_blank_created_at():
    conn = _db()
    _raw_score(conn, "j-bad-key", "garbage", "2026-01-05T00:00:00Z", "bad")
    _raw_score(conn, "j-null-time", "nightly:2026-01-06:sha256:x:finality", None, "null")
    _raw_score(conn, "j-blank-time", "nightly:2026-01-07:sha256:y:finality", "", "blank")
    _add_score(conn, session="2026-01-02", manifest="good")
    assert pool_refresh.pool_refresh_identity(conn) == ("2026-01-02", "good")


def test_identity_ignores_a_malformed_score_key_inside_the_scan_range():
    conn = _db()
    _raw_score(conn, "j-nightly-bad", "nightly:bad", "2026-01-09T00:00:00Z", "bad")
    _add_score(conn, session="2026-01-02", manifest="good",
               created_at="2026-01-01T00:00:00Z")
    assert pool_refresh.pool_refresh_identity(conn) == ("2026-01-02", "good")


def test_identity_ignores_an_unparseable_stage_suffix():
    conn = _db()
    _raw_score(conn, "j-other-stage", "nightly:2026-01-09:sha256:x:refresh",
               "2026-01-09T00:00:00Z", "other")
    _add_score(conn, session="2026-01-02", manifest="good",
               created_at="2026-01-01T00:00:00Z")
    assert pool_refresh.pool_refresh_identity(conn) == ("2026-01-02", "good")


# --------------------------------------------------------------------------
# 12. identity tie-breaks within a session, and session before created_at
# --------------------------------------------------------------------------


def test_identity_prefers_the_later_created_at_within_a_session():
    conn = _db()
    _add_score(conn, session="2026-01-02", manifest="early", scope="sha256:aaa",
               created_at="2026-01-01T00:00:00Z")
    _add_score(conn, session="2026-01-02", manifest="late", scope="sha256:bbb",
               created_at="2026-01-01T01:00:00Z")
    assert pool_refresh.pool_refresh_identity(conn) == ("2026-01-02", "late")


def test_identity_breaks_equal_created_at_on_the_larger_scope_hash():
    conn = _db()
    _add_score(conn, session="2026-01-02", manifest="low", scope="sha256:aaa",
               created_at="2026-01-01T00:00:00Z")
    _add_score(conn, session="2026-01-02", manifest="high", scope="sha256:zzz",
               created_at="2026-01-01T00:00:00Z")
    assert pool_refresh.pool_refresh_identity(conn) == ("2026-01-02", "high")


def test_identity_prefers_the_later_created_at_over_a_larger_scope_hash():
    conn = _db()
    _add_score(conn, session="2026-01-02", manifest="earlier-bigger-scope",
               scope="sha256:zzz", created_at="2026-01-01T00:00:00Z")
    _add_score(conn, session="2026-01-02", manifest="later-smaller-scope",
               scope="sha256:aaa", created_at="2026-01-01T01:00:00Z")
    assert pool_refresh.pool_refresh_identity(conn) == ("2026-01-02", "later-smaller-scope")


def test_identity_ties_created_at_to_the_larger_scope_hash_in_reverse_key_order():
    conn = _db()
    _add_score(conn, session="2026-01-02", manifest="smaller-scope",
               scope="sha256:aaa", created_at="2026-01-01T00:00:00Z")
    _add_score(conn, session="2026-01-02", manifest="larger-scope",
               scope="sha256:zzz", created_at="2026-01-01T00:00:00Z")
    assert pool_refresh.pool_refresh_identity(conn) == ("2026-01-02", "larger-scope")


def test_identity_prefers_the_later_session_over_an_earlier_created_at():
    conn = _db()
    _add_score(conn, session="2026-01-02", manifest="older_session",
               created_at="2026-01-01T00:00:00Z")
    _add_score(conn, session="2026-01-05", manifest="newer_session",
               created_at="2020-01-01T00:00:00Z")
    assert pool_refresh.pool_refresh_identity(conn) == ("2026-01-05", "newer_session")


# --------------------------------------------------------------------------
# 13. an all-summary output set is an empty stage set -> refuse
# --------------------------------------------------------------------------


def test_state_refs_with_only_summaries_is_a_typed_refusal(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    for phase in pool_refresh.TRAINING_PHASES:
        _succeeded_training(conn, phase,
                            ["training/x.summary.json", "training_result"])
    with pytest.raises(OpsError) as err:
        _submit(conn, tmp_path)
    assert err.value.problem.code == "VALIDATION_FAILED"
    assert err.value.problem.message == "no training state files to stage"
    assert err.value.problem.details == {}


# --------------------------------------------------------------------------
# 14. the real training planners build the three phase plans
# --------------------------------------------------------------------------


def test_real_training_planners_build_the_three_phase_plans(monkeypatch, tmp_path):
    recorded = []
    monkeypatch.setattr(pool_refresh, "classify_prior",
                        lambda _root, _pid: PriorClass(PRIOR_USABLE,
                                                     "fake prior catalog is usable"))
    monkeypatch.setattr(pool_refresh.deployment, "current_pointer",
                        lambda root: SimpleNamespace(release_id=PRIOR))

    def submit(conn, registry, policy, request, *, clock):
        recorded.append(request)
        return "receipt"

    monkeypatch.setattr(pool_refresh, "submit", submit)
    conn = _db()
    _add_score(conn)
    _write_pointer(tmp_path, PRIOR)
    assert _submit(conn, tmp_path) == ["receipt"] * 3

    plans = {request.job.parameters["mode"]: request.job.parameters for request in recorded}
    assert set(plans) == {"state", "board_analog", "trailing_cutoff"}
    paired, board, trailing = plans["state"], plans["board_analog"], plans["trailing_cutoff"]
    assert paired["state"] == "paired_residual_pool" and paired["cutoffs"] == (SESSION,)
    assert board["alpha"] == pool_refresh.FILL_ALPHA and board["cutoffs"] == (SESSION,)
    assert board["strategies"] == ()
    assert trailing["cutoffs"] == (SESSION,)
    for parameters in plans.values():
        assert parameters["input_bindings"] == {"legacy_manifest.json": MANIFEST}
    impl = content_hash(worker_source_manifest(REPO_ROOT))
    assert impl
    assert all(request.job.implementation_ref == impl for request in recorded)


# --------------------------------------------------------------------------
# 15. a generic reconcile failure is typed and sanitized
# --------------------------------------------------------------------------


def test_reconcile_generic_failure_is_typed_and_sanitized(monkeypatch, capsys):
    service = SimpleNamespace(conn=None, registry=None, store=None, code_source=None,
                              clock=None, _last_pool_refresh_problem=None)
    monkeypatch.setattr(deployment, "production_release_root", lambda: "/somewhere")

    def _boom(*args, **kwargs):
        raise RuntimeError("secret /tmp/path")

    monkeypatch.setattr(pool_refresh, "submit_pool_nightly_refresh_if_ready", _boom)
    Service._reconcile_pool_nightly_refresh(service)
    assert service._last_pool_refresh_problem == (
        "VALIDATION_FAILED", "pool nightly refresh reconciliation failed")
    out = capsys.readouterr().out
    assert "VALIDATION_FAILED" in out
    assert "pool nightly refresh reconciliation failed" in out
    assert "secret" not in out
    assert "/tmp/path" not in out
    document = to_document(make_problem(
        "VALIDATION_FAILED", "pool nightly refresh reconciliation failed"))
    payload = next(json.loads(line) for line in out.splitlines()
                   if line.startswith("{") and "pool_nightly_refresh_reconcile_failed" in line)
    assert payload["event"] == "pool_nightly_refresh_reconcile_failed"
    for field in ("code", "category", "retryable"):
        assert payload["problem"][field] == document[field]


# --------------------------------------------------------------------------
# 16. identity bounds the per-tick spec read
# --------------------------------------------------------------------------


def _oversize_score_spec():
    filler = "x" * (pool_refresh.MAX_SPEC_BYTES + 1)
    return json.dumps({"parameters": {"input_bindings": {"legacy_manifest.json": filler}}})


def test_identity_ignores_an_oversize_spec_and_keeps_the_older_job():
    conn = _db()
    _add_score(conn, session="2026-01-02", manifest="good")
    _add_score_spec(conn, key="nightly:2026-01-05:sha256:z:finality",
                    created_at="2026-01-09T00:00:00Z", spec_json=_oversize_score_spec())
    assert pool_refresh.pool_refresh_identity(conn) == ("2026-01-02", "good")


def test_identity_is_none_when_the_only_score_spec_is_oversize():
    conn = _db()
    _add_score_spec(conn, key="nightly:2026-01-05:sha256:z:finality",
                    created_at="2026-01-09T00:00:00Z", spec_json=_oversize_score_spec())
    assert pool_refresh.pool_refresh_identity(conn) is None


def test_identity_is_none_for_an_unparseable_spec_without_raising():
    conn = _db()
    _add_score_spec(conn, key="nightly:2026-01-05:sha256:z:finality",
                    created_at="2026-01-05T00:00:00Z", spec_json="{not json")
    assert pool_refresh.pool_refresh_identity(conn) is None


def test_identity_is_none_for_a_non_mapping_spec():
    conn = _db()
    _add_score_spec(conn, key="nightly:2026-01-05:sha256:z:finality",
                    created_at="2026-01-05T00:00:00Z", spec_json="[]")
    assert pool_refresh.pool_refresh_identity(conn) is None


@pytest.mark.parametrize("spec", [
    {"parameters": []},
    {"parameters": {"input_bindings": [1]}},
    {"parameters": "x"},
])
def test_identity_is_none_for_a_non_mapping_nested_layer(spec):
    conn = _db()
    _add_score_spec(conn, key="nightly:2026-01-05:sha256:z:finality",
                    created_at="2026-01-05T00:00:00Z", spec_json=json.dumps(spec))
    assert pool_refresh.pool_refresh_identity(conn) is None


def _add_nonqualifying(conn, index, *, session="2021-01-01"):
    """A newer ``nightly:`` key that the residual kind/succeeded filter rejects."""
    key = f"nightly:{session}:sha256:{index:04d}:train"
    conn.execute("INSERT INTO jobs (job_id, kind, state, idempotency_key, spec_json, created_at) "
                 "VALUES (?, 'training', 'succeeded', ?, '{}', '2021-01-01T00:00:00Z')",
                 (job_id_for("shadow", key), key))
    return key


def test_identity_scan_window_is_bounded_by_max_scan_rows():
    conn = _db()
    _add_score(conn, session="2020-01-01", manifest="oldest")
    for index in range(pool_refresh.MAX_SCAN_ROWS + 20):
        _add_nonqualifying(conn, index)
    assert pool_refresh.pool_refresh_identity(conn) is None


def test_identity_finds_the_score_inside_the_scan_window():
    conn = _db()
    _add_score(conn, session="2020-01-01", manifest="oldest")
    for index in range(10):
        _add_nonqualifying(conn, index)
    assert pool_refresh.pool_refresh_identity(conn) == ("2020-01-01", "oldest")


def test_identity_is_unaffected_by_older_rows_beyond_the_limit():
    conn = _db()
    _add_score(conn, session="2026-06-01", manifest="newest")
    for index in range(100):
        _add_score(conn, session="2020-01-01", manifest=f"old-{index}",
                   scope=f"sha256:{index:03d}", created_at="2026-12-31T00:00:00Z")
    assert pool_refresh.pool_refresh_identity(conn) == ("2026-06-01", "newest")


# --------------------------------------------------------------------------
# 17. the pointer read is bounded to a bounded regular file
# --------------------------------------------------------------------------


def _assert_pointer_refused(deployment_dir):
    with pytest.raises(OpsError) as err:
        pool_refresh._read_pointer(deployment_dir)
    assert err.value.code == "VALIDATION_FAILED"
    assert err.value.problem.message == "deployment pointer is not a bounded regular file"


def test_bounded_pointer_allows_a_missing_file(tmp_path):
    assert pool_refresh._read_pointer(tmp_path) is None


def test_bounded_pointer_allows_a_small_regular_file(tmp_path):
    (tmp_path / "DEPLOYED").write_bytes(_pointer_bytes(PRIOR))
    assert pool_refresh._read_pointer(tmp_path).release_id == PRIOR


def test_bounded_pointer_rejects_a_symlink(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}")
    (tmp_path / "DEPLOYED").symlink_to(target)
    _assert_pointer_refused(tmp_path)


def test_bounded_pointer_rejects_a_directory(tmp_path):
    (tmp_path / "DEPLOYED").mkdir()
    _assert_pointer_refused(tmp_path)


def test_bounded_pointer_rejects_an_oversize_regular_file(tmp_path):
    (tmp_path / "DEPLOYED").write_bytes(b"x" * (pool_refresh.MAX_POINTER_BYTES + 1))
    _assert_pointer_refused(tmp_path)


def _deployment_dir(tmp_path):
    deploy = tmp_path / "deployment"
    deploy.mkdir()
    return deploy


def _assert_submission_pointer_refused(rec):
    assert rec.submits == []


def test_production_submission_refuses_an_oversize_pointer(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path, pointer=None)
    conn = _db()
    _add_score(conn)
    (tmp_path / "deployment").mkdir()
    (tmp_path / "deployment" / "DEPLOYED").write_bytes(
        b"x" * (pool_refresh.MAX_POINTER_BYTES + 1))
    with pytest.raises(OpsError) as err:
        _submit(conn, tmp_path)
    assert err.value.code == "VALIDATION_FAILED"
    assert err.value.problem.message == "deployment pointer is not a bounded regular file"
    _assert_submission_pointer_refused(rec)


def test_production_submission_refuses_a_symlink_pointer(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path, pointer=None)
    conn = _db()
    _add_score(conn)
    deploy = _deployment_dir(tmp_path)
    target = tmp_path / "target.json"
    target.write_text("{}")
    (deploy / "DEPLOYED").symlink_to(target)
    with pytest.raises(OpsError) as err:
        _submit(conn, tmp_path)
    assert err.value.code == "VALIDATION_FAILED"
    assert err.value.problem.message == "deployment pointer is not a bounded regular file"
    _assert_submission_pointer_refused(rec)


def test_production_submission_refuses_a_fifo_pointer(monkeypatch, tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("os.mkfifo is unsupported on this platform")
    rec = _wire(monkeypatch, tmp_path, pointer=None)
    conn = _db()
    _add_score(conn)
    deploy = _deployment_dir(tmp_path)
    os.mkfifo(deploy / "DEPLOYED")
    with pytest.raises(OpsError) as err:
        _submit(conn, tmp_path)
    assert err.value.code == "VALIDATION_FAILED"
    assert err.value.problem.message == "deployment pointer is not a bounded regular file"
    _assert_submission_pointer_refused(rec)


def test_production_submission_refuses_an_undecodable_pointer(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path, pointer=None)
    conn = _db()
    _add_score(conn)
    deploy = _deployment_dir(tmp_path)
    (deploy / "DEPLOYED").write_bytes(b"{not json")
    with pytest.raises(OpsError) as err:
        _submit(conn, tmp_path)
    assert err.value.code == "VALIDATION_FAILED"
    assert err.value.problem.message == "deployment pointer could not be decoded"
    _assert_submission_pointer_refused(rec)


# --------------------------------------------------------------------------
# 18. the real board plan defaults to no strategies (all replay strategies)
# --------------------------------------------------------------------------


def test_real_board_training_plan_defaults_to_no_strategies():
    plan = training.training_plan(
        mode="board_analog", alpha=pool_refresh.FILL_ALPHA, cutoffs=(SESSION,),
        manifest_ref=MANIFEST)
    assert plan["parameters"]["strategies"] == ()


# --------------------------------------------------------------------------
# 19. one non-succeeded training phase blocks the stage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["running", "failed"])
@pytest.mark.parametrize("phase", pool_refresh.TRAINING_PHASES)
def test_a_non_succeeded_training_phase_blocks_the_stage(monkeypatch, tmp_path, phase, state):
    rec = _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    for other in pool_refresh.TRAINING_PHASES:
        if other != phase:
            _add_training(conn, other, state="succeeded")
    _add_training(conn, phase, state=state)

    assert _submit(conn, tmp_path) == []
    assert rec.stage == [] and rec.submits == []


# --------------------------------------------------------------------------
# 20. the promote request pins the caller's code source, not the repo root
# --------------------------------------------------------------------------


_CODE_BASE_FILES = {
    "engine/__init__.py": "",
    "engine/v2/__init__.py": "",
    "engine/v2/ops/__init__.py": "",
    "engine/v2/ops/worker.py": "",
    "engine/models/__init__.py": "",
    "engine/models/ensemble.py": "",
    "engine/models/registry.py": "",
    "engine/models/training/__init__.py": "",
    "engine/models/training/common.py": "",
    "engine/models/training/runup_move.py": "",
}


def _seed_code_source(root):
    for relative, content in _CODE_BASE_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for relative in CODE_ASSET_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    for relative in CODE_ASSET_DIRS:
        path = root / relative / "index.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html></html>")


def test_promote_request_pins_the_override_code_source(monkeypatch, tmp_path):
    other_source = tmp_path / "code"
    _seed_code_source(other_source)
    monkeypatch.setattr(pool_refresh, "classify_prior",
                        lambda _root, _pid: PriorClass(PRIOR_USABLE,
                                                     "fake prior catalog is usable"))
    monkeypatch.setattr(pool_refresh.deployment, "current_pointer",
                        lambda root: SimpleNamespace(release_id=PRIOR))
    recorded = []
    monkeypatch.setattr(
        pool_refresh, "submit",
        lambda conn, registry, policy, request, *, clock: recorded.append(request) or "receipt")
    conn = _db()
    _add_score(conn)
    _all_training_succeeded(conn)
    _add_training(conn, "stage", state="succeeded")
    _write_pointer(tmp_path, PRIOR)

    assert _submit(conn, tmp_path, code_source=other_source) == ["receipt"]
    override = content_hash(worker_source_manifest(other_source))
    repo_root = content_hash(worker_source_manifest(REPO_ROOT))
    assert override != repo_root
    assert recorded[0].job.implementation_ref == override
    assert recorded[0].job.implementation_ref != repo_root


# --------------------------------------------------------------------------
# 21. the real training submits pin the caller's code source, not the repo root
# --------------------------------------------------------------------------


def test_training_requests_pin_the_override_code_source(monkeypatch, tmp_path):
    other_source = tmp_path / "code"
    _seed_code_source(other_source)
    monkeypatch.setattr(pool_refresh, "classify_prior",
                        lambda _root, _pid: PriorClass(PRIOR_USABLE,
                                                     "fake prior catalog is usable"))
    monkeypatch.setattr(pool_refresh.deployment, "current_pointer",
                        lambda root: SimpleNamespace(release_id=PRIOR))
    recorded = []
    monkeypatch.setattr(
        pool_refresh, "submit",
        lambda conn, registry, policy, request, *, clock: recorded.append(request) or "receipt")
    conn = _db()
    _add_score(conn)
    _write_pointer(tmp_path, PRIOR)

    assert _submit(conn, tmp_path, code_source=other_source) == ["receipt"] * 3
    override = content_hash(worker_source_manifest(other_source))
    repo_root = content_hash(worker_source_manifest(REPO_ROOT))
    assert override != repo_root
    assert len(recorded) == len(pool_refresh.TRAINING_PHASES)
    assert all(request.job.implementation_ref == override for request in recorded)
    assert all(request.job.implementation_ref != repo_root for request in recorded)


# --------------------------------------------------------------------------
# 22. the DEPLOYED pointer is read exactly once per submission
# --------------------------------------------------------------------------


def test_pointer_is_read_exactly_once_per_submission(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    calls = []
    real_read = pool_refresh.read_bounded_state

    def counting_read(*args, **kwargs):
        calls.append(args)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(pool_refresh, "read_bounded_state", counting_read)
    assert _submit(conn, tmp_path) == ["receipt"] * 3
    assert len(calls) == 1
    assert _submit(conn, tmp_path) == ["receipt"] * 3
    assert len(calls) == 2


# --------------------------------------------------------------------------
# 23. the identity query plan needs no temp b-tree for its ORDER BY
# --------------------------------------------------------------------------


def test_identity_query_plan_searches_the_key_range_without_sorting():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE jobs (namespace TEXT NOT NULL DEFAULT 'shadow', "
        "idempotency_key TEXT NOT NULL, kind TEXT, state TEXT, spec_json TEXT, "
        "created_at TEXT, PRIMARY KEY (namespace, idempotency_key))")
    rows = conn.execute(
        "EXPLAIN QUERY PLAN " + pool_refresh.IDENTITY_SQL,
        (pool_refresh.MAX_SCAN_ROWS, pool_refresh._FINALITY_KIND,
         pool_refresh._FEATURES_KIND, pool_refresh.MAX_SPEC_BYTES,
         pool_refresh.MAX_SCORE_ROWS)).fetchall()
    inner = [row for row in rows if "jobs" in row["detail"]]
    inner_plan = " ".join(row["detail"] for row in inner)
    assert inner_plan, " ".join(row["detail"] for row in rows)
    assert "SEARCH" in inner_plan.upper()
    assert "SCAN" not in inner_plan.upper()
    assert "idempotency_key>" in inner_plan and "idempotency_key<" in inner_plan
    inner_parents = {row["parent"] for row in inner}
    sorted_inner = [row["detail"] for row in rows
                    if "USE TEMP B-TREE FOR ORDER BY" in row["detail"].upper()
                    and row["parent"] in inner_parents]
    assert not sorted_inner


def test_identity_finds_the_score_job_among_later_keys():
    conn = _db()
    _add_score(conn, session=SESSION, manifest=MANIFEST)
    for index in range(40):
        key = f"retrain:2026-01-{index + 1:02d}:job"
        conn.execute(
            "INSERT INTO jobs (job_id, kind, state, idempotency_key, spec_json, created_at) "
            "VALUES (?, 'training', 'succeeded', ?, '{}', '2027-01-01T00:00:00Z')",
            (job_id_for("shadow", key), key))
    assert pool_refresh.pool_refresh_identity(conn) == (SESSION, MANIFEST)


# --------------------------------------------------------------------------
# 24. stalled cycles are not resubmitted and are not pending
# --------------------------------------------------------------------------


def test_failed_training_phase_is_a_stalled_cycle(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    _add_training(conn, "train-paired", state="succeeded")
    _add_training(conn, "train-board", state="failed")
    _add_training(conn, "train-trailing", state="succeeded")

    assert _submit(conn, tmp_path) == []
    assert rec.submits == []
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is False


def test_failed_promote_phase_is_a_stalled_cycle(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    _all_training_succeeded(conn)
    _add_training(conn, "stage", state="succeeded")
    _add_training(conn, "promote", state="failed")

    assert _submit(conn, tmp_path) == []
    assert rec.submits == []
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is False


def test_running_stage_is_pending_and_not_resubmitted(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    _all_training_succeeded(conn)
    _add_training(conn, "stage", state="running")

    assert _submit(conn, tmp_path) == []
    assert rec.submits == []
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is True


def test_running_promote_is_pending_and_not_resubmitted(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, tmp_path)
    conn = _db()
    _add_score(conn)
    _all_training_succeeded(conn)
    _add_training(conn, "stage", state="succeeded")
    _add_training(conn, "promote", state="running")

    assert _submit(conn, tmp_path) == []
    assert rec.submits == []
    assert pool_refresh.pool_cycle_pending(conn, tmp_path) is True


# --------------------------------------------------------------------------
# 25. the promote lookup is a binary key range, not a LIKE prefix
# --------------------------------------------------------------------------


def _add_promote(conn, *, session=SESSION, tag=TAG, prior=PRIOR, state="failed"):
    key = f"nightly:{session}:pool:promote:{tag}:{prior}"
    conn.execute("INSERT INTO jobs (job_id, kind, state, idempotency_key, spec_json, created_at) "
                 "VALUES (?, 'training', ?, ?, NULL, NULL)",
                 (job_id_for("shadow", key), state, key))
    return key


def test_promote_lookup_excludes_other_sessions_and_tags():
    conn = _db()
    _add_promote(conn, prior=PRIOR, state="failed")
    _add_promote(conn, session="2026-01-03", state="running")
    _add_promote(conn, session=f"{SESSION}x", state="running")
    _add_promote(conn, tag="othertag", state="running")
    _add_promote(conn, tag=TAG[:-1] + ("0" if TAG[-1] != "0" else "1"), state="running")
    assert pool_refresh._promote_states_with_unknown_prior(conn, SESSION, TAG) == ["failed"]


def test_promote_lookup_returns_a_prior_with_uppercase_and_odd_chars():
    conn = _db()
    _add_promote(conn, prior="Prior_ABC-1!?", state="running")
    assert pool_refresh._promote_states_with_unknown_prior(conn, SESSION, TAG) == ["running"]


def test_promote_lookup_honours_both_key_bounds():
    conn = _db()
    _add_promote(conn, prior="", state="failed")
    _add_promote(conn, prior="9", state="running")
    for suffix, state in ((";x", "cancelled"), ("9", "blocked")):
        key = f"nightly:{SESSION}:pool:promote:{TAG}{suffix}"
        conn.execute(
            "INSERT INTO jobs (job_id, kind, state, idempotency_key, spec_json, created_at) "
            "VALUES (?, 'training', ?, ?, NULL, NULL)",
            (job_id_for("shadow", key), state, key))
    assert pool_refresh._promote_states_with_unknown_prior(conn, SESSION, TAG) == [
        "running", "failed"]
