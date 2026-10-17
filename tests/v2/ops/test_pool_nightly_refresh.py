"""#192 slice 4: the pool-nightly refresh sidecar (``pool_refresh.py``).

Synthetic only: an in-memory ``sqlite3`` fake plus monkeypatched planners and
submission, so nothing here reads or writes a real model release store or
``data/``. Each test proves one step of the phase machine: score identity, the
three training submitters, the ``phase5_state_stage`` stage, and the
``models_promote`` swap -- plus ``Service._reconcile_pool_nightly_refresh``'s
graceful degradation.
"""
from __future__ import annotations

import ast
import inspect
import json
import sqlite3
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.models import deployment
from engine.v2.ops import pool_refresh
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.plans import request_from_plan as real_request_from_plan
from engine.v2.ops.stages import registry
from engine.v2.ops.state_stage import state_stage_plan
from engine.v2.ops.submission import NamespacePolicy, job_id_for, validate_request
from engine.v2.ops.supervisor import Service

REPO_ROOT = Path(__file__).resolve().parents[3]
SESSION = "2026-01-02"
PRIOR = "prior1"
MANIFEST = "manifest-1"


def _db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE jobs (job_id TEXT, kind TEXT, state TEXT, idempotency_key TEXT, "
        "spec_json TEXT, created_at TEXT);"
        "CREATE TABLE attempts (attempt_id TEXT, job_id TEXT, state TEXT, "
        "attempt_number INTEGER);"
        "CREATE TABLE attempt_outputs (attempt_id TEXT, name TEXT, artifact_id TEXT);")
    return conn


def _add_score(conn, *, session=SESSION, manifest=MANIFEST, scope="sha256:abc",
               created_at="2026-01-01T00:00:00Z", state="succeeded"):
    key = f"nightly:{session}:{scope}:score"
    bindings = {} if manifest is None else {"legacy_manifest.json": manifest}
    spec = json.dumps({"parameters": {"input_bindings": bindings}})
    conn.execute("INSERT INTO jobs (job_id, kind, state, idempotency_key, spec_json, created_at) "
                 "VALUES (?, 'legacy_score', ?, ?, ?, ?)",
                 (job_id_for("shadow", key), state, key, spec, created_at))
    return key


def _add_training(conn, phase, *, session=SESSION, prior=PRIOR, state="queued"):
    key = pool_refresh.phase_key(session, phase, prior)
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


def _wire(monkeypatch, *, pointer=PRIOR):
    """Patch every collaborator ``pool_refresh`` calls; records each call."""
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

    if pointer is None:
        monkeypatch.setattr(pool_refresh.deployment, "current_pointer", lambda root: None)
    else:
        monkeypatch.setattr(pool_refresh.deployment, "current_pointer",
                            lambda root: SimpleNamespace(release_id=pointer))
    monkeypatch.setattr(pool_refresh.training, "training_plan", training_plan)
    monkeypatch.setattr(pool_refresh.training, "promote_plan", promote_plan)
    monkeypatch.setattr(pool_refresh.state_stage, "state_stage_plan", stage_plan)
    monkeypatch.setattr(pool_refresh, "request_from_plan", request_from_plan)
    monkeypatch.setattr(pool_refresh, "submit", submit)
    monkeypatch.setattr(pool_refresh, "_implementation_ref", lambda code_source: "impl")
    return rec


def _submit(conn, tmp_path):
    return pool_refresh.submit_pool_nightly_refresh_if_ready(
        conn, None, None, None, tmp_path, code_source=REPO_ROOT, clock=None)


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


# --------------------------------------------------------------------------
# 2. nothing to do: no pointer, no conn, already-advanced pointer
# --------------------------------------------------------------------------


def test_no_pointer_no_conn_or_already_advanced_submits_nothing(monkeypatch, tmp_path):
    rec = _wire(monkeypatch, pointer=None)
    conn = _db()
    _add_score(conn)
    assert _submit(conn, tmp_path) == []
    assert _submit(None, tmp_path) == []

    rec2 = _wire(monkeypatch, pointer=f"pools-{SESSION}")
    assert _submit(conn, tmp_path) == []
    assert rec.submits == [] and rec2.submits == []


# --------------------------------------------------------------------------
# 3. all three training keys missing -> exactly three submissions
# --------------------------------------------------------------------------


def test_all_training_phases_missing_submit_three(monkeypatch, tmp_path):
    rec = _wire(monkeypatch)
    conn = _db()
    _add_score(conn)
    assert _submit(conn, tmp_path) == ["receipt"] * 3
    assert [key for _plan, key in rec.requests] == [
        pool_refresh.phase_key(SESSION, phase, PRIOR) for phase in pool_refresh.TRAINING_PHASES]
    assert [key for _plan, key in rec.requests] == [
        f"nightly:{SESSION}:pool:{phase}:{PRIOR}" for phase in pool_refresh.TRAINING_PHASES]
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


# --------------------------------------------------------------------------
# 4. an existing key (even failed) is never resubmitted
# --------------------------------------------------------------------------


def test_existing_failed_training_job_is_not_resubmitted(monkeypatch, tmp_path):
    rec = _wire(monkeypatch)
    conn = _db()
    _add_score(conn)
    _add_training(conn, "train-paired", state="failed")
    assert _submit(conn, tmp_path) == ["receipt"] * 2
    assert [key for _plan, key in rec.requests] == [
        pool_refresh.phase_key(SESSION, "train-board", PRIOR),
        pool_refresh.phase_key(SESSION, "train-trailing", PRIOR)]


# --------------------------------------------------------------------------
# 5. all keys exist but one is still running -> wait
# --------------------------------------------------------------------------


def test_waiting_while_a_training_job_runs(monkeypatch, tmp_path):
    rec = _wire(monkeypatch)
    conn = _db()
    _add_score(conn)
    _add_training(conn, "train-paired", state="succeeded")
    _add_training(conn, "train-board", state="succeeded")
    _add_training(conn, "train-trailing", state="running")
    assert _submit(conn, tmp_path) == []
    assert rec.submits == []


def test_plans_are_built_lazily_only_for_missing_phases(monkeypatch, tmp_path):
    _wire(monkeypatch)
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
    rec = _wire(monkeypatch)
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
    assert stage["release_id"] == f"pools-{SESSION}"
    assert stage["cutoff"] == SESSION
    assert stage["release_root"] == str(tmp_path)
    assert stage["state_refs"] == {
        "paired_residual_pool.json": f"{paired}#training/paired_residual_pool.json",
        board_file: f"{board}#training/{board_file}",
        "trailing_pnl_cutoff__2026-10.json":
            f"{trailing}#training/trailing_pnl_cutoff__2026-10.json"}
    assert "board_analog_matcher.summary.json" not in stage["state_refs"]
    assert rec.requests[-1][1] == pool_refresh.phase_key(SESSION, "stage", PRIOR)


def test_state_refs_use_only_the_latest_succeeded_attempt(monkeypatch, tmp_path):
    rec = _wire(monkeypatch)
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


# --------------------------------------------------------------------------
# 7. a duplicate state name or a missing succeeded attempt refuses
# --------------------------------------------------------------------------


def test_duplicate_state_file_name_raises(monkeypatch, tmp_path):
    _wire(monkeypatch)
    conn = _db()
    _add_score(conn)
    _succeeded_training(conn, "train-paired", ["training/paired.json"])
    _succeeded_training(conn, "train-board", ["training/paired.json"])
    _succeeded_training(conn, "train-trailing", ["training/trailing.json"])
    with pytest.raises(OpsError) as err:
        _submit(conn, tmp_path)
    assert err.value.code == "VALIDATION_FAILED"


def test_training_job_without_a_succeeded_attempt_raises(monkeypatch, tmp_path):
    _wire(monkeypatch)
    conn = _db()
    _add_score(conn)
    _add_training(conn, "train-paired", state="succeeded")  # no attempt row
    _succeeded_training(conn, "train-board", ["training/board.json"])
    _succeeded_training(conn, "train-trailing", ["training/trailing.json"])
    with pytest.raises(OpsError) as err:
        _submit(conn, tmp_path)
    assert err.value.code == "VALIDATION_FAILED"


# --------------------------------------------------------------------------
# 8. stage / promote sequencing
# --------------------------------------------------------------------------


def _all_training_succeeded(conn):
    for phase in pool_refresh.TRAINING_PHASES:
        _add_training(conn, phase, state="succeeded")


def test_stage_that_is_not_yet_succeeded_waits(monkeypatch, tmp_path):
    rec = _wire(monkeypatch)
    conn = _db()
    _add_score(conn)
    _all_training_succeeded(conn)
    _add_training(conn, "stage", state="running")
    assert _submit(conn, tmp_path) == []
    assert rec.submits == [] and rec.promote == []


def test_failed_stage_job_blocks_promote(monkeypatch, tmp_path):
    rec = _wire(monkeypatch)
    conn = _db()
    _add_score(conn)
    _all_training_succeeded(conn)
    _add_training(conn, "stage", state="failed")
    assert _submit(conn, tmp_path) == []
    assert rec.promote == [] and rec.submits == []


def test_stage_succeeded_submits_a_guarded_promote(monkeypatch, tmp_path):
    rec = _wire(monkeypatch)
    conn = _db()
    _add_score(conn)
    _all_training_succeeded(conn)
    _add_training(conn, "stage", state="succeeded")

    assert _submit(conn, tmp_path) == ["receipt"]
    assert len(rec.promote) == 1
    assert rec.promote[0]["expected_previous_release_id"] == PRIOR
    assert rec.promote[0]["release_id"] == f"pools-{SESSION}"
    assert rec.promote[0]["release_root"] == str(Path(tmp_path) / "deployment")
    assert rec.requests[-1][1] == pool_refresh.phase_key(SESSION, "promote", PRIOR)


def test_existing_promote_is_idempotent(monkeypatch, tmp_path):
    rec = _wire(monkeypatch)
    conn = _db()
    _add_score(conn)
    _all_training_succeeded(conn)
    _add_training(conn, "stage", state="succeeded")
    _add_training(conn, "promote", state="queued")
    assert _submit(conn, tmp_path) == []
    assert rec.submits == []


# --------------------------------------------------------------------------
# 9. request_from_plan accepts a real state_stage_plan document
# --------------------------------------------------------------------------


def test_real_request_from_plan_accepts_a_state_stage_plan(tmp_path):
    plan = state_stage_plan(release_root=str(tmp_path), prior_release_id="a", release_id="b",
                            cutoff="2026-10-01",
                            state_refs={"x.json": "job_abc#training/x.json"},
                            source_root=REPO_ROOT)
    request = real_request_from_plan(plan, "nightly:2026-10-01:pool:stage:a")
    assert request.job.kind == "phase5_state_stage"
    validate_request(registry(), NamespacePolicy({"operator": frozenset({"shadow"})}), request)
    assert request.job.dependency_job_ids == ("job_abc",)
    assert request.job.input_refs == ()
    assert request.job.retry_policy_ref == "none"


# --------------------------------------------------------------------------
# 10. Service._reconcile_pool_nightly_refresh degrades gracefully
# --------------------------------------------------------------------------


def test_reconcile_swallows_missing_root_and_reports_once(monkeypatch, capsys):
    service = SimpleNamespace(conn=None, registry=None, store=None, code_source=None,
                              clock=None, _last_pool_refresh_problem=None)

    def _missing():
        raise deployment.MissingReleaseRoot()

    monkeypatch.setattr(deployment, "production_release_root", _missing)
    assert Service._reconcile_pool_nightly_refresh(service) is None
    assert capsys.readouterr().out == ""
    assert service._last_pool_refresh_problem is None

    monkeypatch.setattr(deployment, "production_release_root", lambda: "/somewhere")

    def _boom(*args, **kwargs):
        raise fail("INPUT_CHANGED", "planted pool refresh failure")

    monkeypatch.setattr(pool_refresh, "submit_pool_nightly_refresh_if_ready", _boom)
    Service._reconcile_pool_nightly_refresh(service)
    Service._reconcile_pool_nightly_refresh(service)
    assert service._last_pool_refresh_problem == ("INPUT_CHANGED", "planted pool refresh failure")
    out = capsys.readouterr().out
    assert out.count("pool_nightly_refresh_reconcile_failed") == 1
    assert "INPUT_CHANGED" in out and "planted pool refresh failure" in out

    monkeypatch.setattr(pool_refresh, "submit_pool_nightly_refresh_if_ready",
                        lambda *args, **kwargs: [])
    Service._reconcile_pool_nightly_refresh(service)
    assert service._last_pool_refresh_problem is None


def test_tick_calls_reconcile_pool_nightly_refresh():
    tree = ast.parse(textwrap.dedent(inspect.getsource(Service.tick)))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "_reconcile_pool_nightly_refresh"]
    assert calls


# --------------------------------------------------------------------------
# 11. identity: unparseable key and blank created_at rows are ignored
# --------------------------------------------------------------------------


def _raw_score(conn, job_id, key, created_at, manifest):
    spec = json.dumps({"parameters": {"input_bindings": {"legacy_manifest.json": manifest}}})
    conn.execute("INSERT INTO jobs (job_id, kind, state, idempotency_key, spec_json, created_at) "
                 "VALUES (?, 'legacy_score', 'succeeded', ?, ?, ?)",
                 (job_id, key, spec, created_at))


def test_identity_ignores_the_unparseable_key_and_blank_created_at():
    conn = _db()
    _raw_score(conn, "j-bad-key", "garbage", "2026-01-05T00:00:00Z", "bad")
    _raw_score(conn, "j-null-time", "nightly:2026-01-06:sha256:x:score", None, "null")
    _raw_score(conn, "j-blank-time", "nightly:2026-01-07:sha256:y:score", "", "blank")
    _add_score(conn, session="2026-01-02", manifest="good")
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
    _wire(monkeypatch)
    conn = _db()
    _add_score(conn)
    for phase in pool_refresh.TRAINING_PHASES:
        _succeeded_training(conn, phase,
                            ["training/x.summary.json", "training_result"])
    with pytest.raises(OpsError) as err:
        _submit(conn, tmp_path)
    assert err.value.problem.code == "VALIDATION_FAILED"
    assert err.value.problem.message == "no training state files to stage"


# --------------------------------------------------------------------------
# 14. the real training planners build the three phase plans
# --------------------------------------------------------------------------


def test_real_training_planners_build_the_three_phase_plans(monkeypatch, tmp_path):
    recorded = []
    monkeypatch.setattr(pool_refresh.deployment, "current_pointer",
                        lambda root: SimpleNamespace(release_id=PRIOR))

    def submit(conn, registry, policy, request, *, clock):
        recorded.append(request)
        return "receipt"

    monkeypatch.setattr(pool_refresh, "submit", submit)
    conn = _db()
    _add_score(conn)
    assert _submit(conn, tmp_path) == ["receipt"] * 3

    plans = {request.job.parameters["mode"]: request.job.parameters for request in recorded}
    assert set(plans) == {"state", "board_analog", "trailing_cutoff"}
    paired, board, trailing = plans["state"], plans["board_analog"], plans["trailing_cutoff"]
    assert paired["state"] == "paired_residual_pool" and paired["cutoffs"] == (SESSION,)
    assert board["alpha"] == pool_refresh.FILL_ALPHA and board["cutoffs"] == (SESSION,)
    assert trailing["cutoffs"] == (SESSION,)
    for parameters in plans.values():
        assert parameters["input_bindings"] == {"legacy_manifest.json": MANIFEST}
    impl = pool_refresh._implementation_ref(REPO_ROOT)
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
