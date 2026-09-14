"""P2-5/Task5 (D20): the export, engineering-gate, publication and backup
outbox effects wired into the nightly job DAG.

Real SQLite (``open_catalog``), a real ``ArtifactStore`` and real submission/
claim/commit_attempt machinery throughout. Parent jobs whose only role is to
give a coordinator effect a durable, resolvable ``job_<id>#<output>`` binding
are seeded the same way ``tests/test_v2_ops_nightly_completion.py`` seeds
them (``_succeed_parent``): a real ``artifact_check`` submission driven all
the way through ``commit_attempt``, never a mock. A committed decision itself
is seeded directly through ``engine.v2.ledger.decisions.insert`` plus the
outbox/watermark calls ``commit_decisions_in_transaction`` makes — this
file's focus is the downstream effects graph, and decision_commit's own
validation path already has its own test file.

The coordinator effect functions (``ledger_export_effect``,
``engineering_gate_effect``, ``publication_effect``, ``backup_effect``) are
called directly against a real claim, exactly the way
``supervisor.Service._coordinator_effect`` calls them — no subprocess is
spun up for the trivial worker itself, since that dispatch is a two-line
receipt writer covered by the DAG/binding tests at the bottom of this file.
"""
from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

import pytest

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import ArtifactStore, content_hash, format_timestamp
from engine.v2.ledger.decisions import insert, set_authority
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.effects_graph import (
    EXPORT_PURPOSES,
    backup_effect,
    engineering_gate_effect,
    ledger_export_effect,
    publication_effect,
)
from engine.v2.ops.errors import OpsError
from engine.v2.ops.input_bindings import resolve_and_record, resolve_bindings
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from engine.v2.ops.nightly import build_legacy_job_requests, build_nightly_plan
from engine.v2.ops.outbox import enqueue, watermark
from engine.v2.ops.publication import current as release_current
from engine.v2.ops.publication import publish_local
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor, claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, job_id_for, submit, submit_graph
from tests.ops_support import DEFAULT_POLICY, FakeClock, sample

REPO = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})
SESSION = "2026-09-12"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _open(tmp_path):
    root = tmp_path / "ops"
    root.mkdir()
    clock = FakeClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    epoch = begin_epoch(conn, clock=clock, boot_id="boot", pid=1)
    supervisor = Supervisor(epoch, "boot")
    store = ArtifactStore(root)
    return conn, clock, supervisor, store, root


def _publish(store, conn, clock, value, schema_ref):
    ref = store.publish_bytes(json.dumps(value, sort_keys=True).encode(), schema_ref=schema_ref)
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return ref


def _row(event, kind):
    return {"row_id": event + "-" + kind, "event_id": event, "ticker": "FAKE",
            "strategy": "TWIN-P", "event_date": SESSION, "status": "resolved",
            "resolved_at": SESSION + "T21:00:00+00:00"}


def _ensure_authority(conn, clock):
    if conn.execute("SELECT 1 FROM decision_authority WHERE singleton=1").fetchone() is None:
        with transaction(conn):
            set_authority(conn, None, "catalog", format_timestamp(clock.now()))


def _seed_decisions(conn, clock, scope, session, *, predictions=(), outcomes=(), extra=()):
    """Committed rows plus the 'decisions' watermark and export/release_intent
    outbox effects, in the exact shape ``commit_decisions_in_transaction``
    leaves them in — seeded directly since this file's subject is what runs
    downstream of that commit, not the commit's own guarded path."""
    _ensure_authority(conn, clock)
    ts = format_timestamp(clock.now())
    release_key = content_hash([scope, session, "decisions-seed", predictions, outcomes])
    with transaction(conn):
        for row in predictions:
            insert(conn, logical_key="pred:" + row["row_id"], decision_id="prediction:" + row["row_id"],
                  payload=row, purpose="shadow", kind="prediction", validations={}, created_at=ts)
        for row in outcomes:
            insert(conn, logical_key="out:" + row["row_id"], decision_id="outcome:" + row["row_id"],
                  payload=row, purpose="legacy_import", kind="outcome", validations={}, created_at=ts)
        for row in extra:
            insert(conn, logical_key="extra:" + row["row_id"],
                  decision_id="prediction:" + row["row_id"] + ":extra", payload=row,
                  purpose="research_reconstruction", kind="prediction", validations={}, created_at=ts)
        enqueue(conn, "export", release_key, {"validation": "seed"})
        enqueue(conn, "release_intent", release_key, {"validation": "seed"})
        watermark(conn, "nightly", scope, "decisions", session, release_key, clock=clock)
    return release_key


def _seed_settlement(conn, clock, scope, session, *, outcomes):
    ts = format_timestamp(clock.now())
    settlement_key = content_hash(["settlement", scope, session, outcomes])
    with transaction(conn):
        for row in outcomes:
            insert(conn, logical_key="settle:" + row["row_id"],
                  decision_id="outcome:" + row["row_id"] + ":settle", payload=row,
                  purpose="legacy_import", kind="outcome", validations={}, created_at=ts)
        enqueue(conn, "export", settlement_key, {"settlement_candidate": "seed"})
        watermark(conn, "nightly", scope, "settlement", session, settlement_key, clock=clock)
    return settlement_key


def _params(kind, session, scope, **extra):
    base = {"expected_ids": (kind,), "session": session, "effect_scope": scope}
    base.update(extra)
    return base


def _submit_and_claim(conn, clock, supervisor, *, kind, key, parameters, dependency_job_ids=()):
    job = JobSpec(kind=kind, implementation_ref="test-impl", spec_hash=None,
                 environment_ref="test-env", parameters=parameters,
                 dependency_job_ids=dependency_job_ids, output_namespace="shadow",
                 resource_class="delivery", retry_policy_ref="bounded",
                 checkpoint_contract_ref="effect_receipt.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(namespace="shadow", idempotency_key=key,
          principal="operator", job=job), clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=supervisor,
                       clock=clock, registry=registry())
    assert claim is not None, "job did not claim"
    return claim


def _succeed_parent(conn, clock, supervisor, *, key, output_name, ref):
    """A genuinely succeeded parent job/attempt, without a live subprocess —
    the same helper ``test_v2_ops_nightly_completion.py`` uses."""
    job = JobSpec(kind="artifact_check", implementation_ref="parent-setup", spec_hash=None,
                 environment_ref="parent-setup", parameters={"expected_ids": ()},
                 output_namespace="shadow", resource_class="delivery",
                 retry_policy_ref="bounded", checkpoint_contract_ref="receipt.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(namespace="shadow", idempotency_key=key,
          principal="operator", job=job), clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=supervisor,
                       clock=clock, registry=registry())

    def effects(inner_conn):
        register_artifact(inner_conn, ref, claim.attempt_id, clock)
        inner_conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                           (claim.attempt_id, output_name, ref.artifact_id))

    commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                   clock=clock, effects=effects)
    return job_id_for("shadow", key)


def _commit(conn, clock, claim, result):
    """Mirror ``supervisor.Service._finish``'s own effects closure exactly."""
    effect, extra_refs = result
    def effects(inner_conn):
        for name, ref in extra_refs:
            register_artifact(inner_conn, ref, claim.attempt_id, clock)
            inner_conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                               (claim.attempt_id, name, ref.artifact_id))
        if effect is not None:
            effect(inner_conn)
    commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                   clock=clock, effects=effects)


# --------------------------------------------------------------------------
# ledger_export
# --------------------------------------------------------------------------


def test_ledger_export_filters_purposes_and_binds_render(tmp_path):
    assert EXPORT_PURPOSES == ("legacy_import", "shadow")
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        prediction, outcome, extra = _row("evt-1", "pred"), _row("evt-1", "out"), _row("evt-2", "pred")
        _seed_decisions(conn, clock, scope, SESSION, predictions=[prediction], outcomes=[outcome],
                        extra=[extra])

        claim = _submit_and_claim(conn, clock, supervisor, kind="ledger_export", key="export-1",
                                  parameters=_params("ledger_export", SESSION, scope))
        result = ledger_export_effect(conn, store, claim, root, REPO, clock=clock)
        effect, extra_refs = result
        assert [name for name, _ in extra_refs] == ["ledger_export"]
        tar_ref = extra_refs[0][1]
        _commit(conn, clock, claim, result)

        with tarfile.open(store.verify(tar_ref)) as archive:
            members = {m.name for m in archive.getmembers() if m.isfile()}
            predicted = json.loads(archive.extractfile("predictions/" + SESSION + ".jsonl").read())
            observed = json.loads(archive.extractfile("outcomes/" + SESSION + ".jsonl").read())
        assert members == {"predictions/" + SESSION + ".jsonl", "outcomes/" + SESSION + ".jsonl"}
        assert predicted == prediction
        assert observed == outcome
        assert extra["row_id"] not in json.dumps(predicted)

        export_job = job_id_for("shadow", "export-1")
        render_spec = JobSpec(kind="legacy_render", implementation_ref="x", spec_hash=None,
                              environment_ref="x",
                              parameters={"input_bindings": {
                                  "ledger_generation.tar": export_job + "#ledger_export"}},
                              dependency_job_ids=(export_job,), output_namespace="shadow",
                              resource_class="projection", retry_policy_ref="bounded",
                              checkpoint_contract_ref="legacy_action.v1.0")
        resolved = resolve_bindings(conn, store, render_spec)
        assert resolved["ledger_generation.tar"].artifact_id == tar_ref.artifact_id

        outbox_row = conn.execute("SELECT state FROM outbox WHERE kind='export'").fetchone()
        assert outbox_row["state"] == "delivered"
        wm = conn.execute("SELECT occurrence, receipt_ref FROM watermarks WHERE pipeline='nightly' "
                          "AND scope=? AND stage='export'", (scope,)).fetchone()
        assert wm["occurrence"] == SESSION
        assert wm["receipt_ref"] == tar_ref.content_hash
    finally:
        conn.close()


def test_ledger_export_rerun_is_idempotent(tmp_path):
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        prediction = _row("evt-1", "pred")
        _seed_decisions(conn, clock, scope, SESSION, predictions=[prediction])

        first_claim = _submit_and_claim(conn, clock, supervisor, kind="ledger_export", key="run-1",
                                        parameters=_params("ledger_export", SESSION, scope))
        first = ledger_export_effect(conn, store, first_claim, root, REPO, clock=clock)
        first_ref = first[1][0][1]
        _commit(conn, clock, first_claim, first)

        second_claim = _submit_and_claim(conn, clock, supervisor, kind="ledger_export", key="run-2",
                                         parameters=_params("ledger_export", SESSION, scope))
        second = ledger_export_effect(conn, store, second_claim, root, REPO, clock=clock)
        second_ref = second[1][0][1]
        _commit(conn, clock, second_claim, second)

        assert first_ref.content_hash == second_ref.content_hash
        rows = conn.execute("SELECT state FROM outbox WHERE kind='export'").fetchall()
        assert [r["state"] for r in rows] == ["delivered"]
        watermarks = conn.execute(
            "SELECT occurrence FROM watermarks WHERE pipeline='nightly' AND scope=? "
            "AND stage='export'", (scope,)).fetchall()
        assert len(watermarks) == 1
    finally:
        conn.close()


def test_ledger_export_settlement_absent_when_not_committed(tmp_path):
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        _seed_decisions(conn, clock, scope, SESSION, predictions=[_row("evt-1", "pred")])
        claim = _submit_and_claim(conn, clock, supervisor, kind="ledger_export", key="export-1",
                                  parameters=_params("ledger_export", SESSION, scope))
        result = ledger_export_effect(conn, store, claim, root, REPO, clock=clock)
        _commit(conn, clock, claim, result)
        row = conn.execute("SELECT receipt_json FROM outbox WHERE kind='export'").fetchone()
        receipt = json.loads(row["receipt_json"])
        assert receipt["settlement"] == {"present": False, "release_key": None}
    finally:
        conn.close()


def test_ledger_export_settlement_present_when_committed(tmp_path):
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        _seed_decisions(conn, clock, scope, SESSION, predictions=[_row("evt-1", "pred")])
        settlement_key = _seed_settlement(conn, clock, scope, SESSION, outcomes=[_row("evt-1", "out")])
        claim = _submit_and_claim(conn, clock, supervisor, kind="ledger_export", key="export-1",
                                  parameters=_params("ledger_export", SESSION, scope))
        result = ledger_export_effect(conn, store, claim, root, REPO, clock=clock)
        _commit(conn, clock, claim, result)

        rows = {r["logical_key"]: r for r in conn.execute(
            "SELECT logical_key, state, receipt_json FROM outbox WHERE kind='export'")}
        assert rows[settlement_key]["state"] == "delivered"
        receipt = json.loads(rows[settlement_key]["receipt_json"])
        assert receipt["settlement"] == {"present": True, "release_key": settlement_key}
        # settlement's own watermark is untouched by export beyond what was seeded
        wm = conn.execute("SELECT occurrence FROM watermarks WHERE pipeline='nightly' AND scope=? "
                          "AND stage='settlement'", (scope,)).fetchone()
        assert wm["occurrence"] == SESSION
    finally:
        conn.close()


def test_ledger_export_scope_subset_leaves_full_scope_untouched(tmp_path):
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        subset_scope = "shadow:abc123"
        _seed_decisions(conn, clock, subset_scope, SESSION, predictions=[_row("evt-1", "pred")])
        claim = _submit_and_claim(conn, clock, supervisor, kind="ledger_export", key="export-1",
                                  parameters=_params("ledger_export", SESSION, subset_scope))
        result = ledger_export_effect(conn, store, claim, root, REPO, clock=clock)
        _commit(conn, clock, claim, result)

        subset_wm = conn.execute("SELECT 1 FROM watermarks WHERE pipeline='nightly' AND scope=? "
                                 "AND stage='export'", (subset_scope,)).fetchone()
        full_wm = conn.execute("SELECT 1 FROM watermarks WHERE pipeline='nightly' AND scope='shadow' "
                               "AND stage='export'").fetchone()
        assert subset_wm is not None
        assert full_wm is None
    finally:
        conn.close()


# --------------------------------------------------------------------------
# engineering_gate
# --------------------------------------------------------------------------


def test_engineering_gate_effect_runs_the_real_gate_excluding_coverage(tmp_path):
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        claim = _submit_and_claim(conn, clock, supervisor, kind="engineering_gate", key="gate-1",
                                  parameters=_params("engineering_gate", SESSION, scope))
        result = engineering_gate_effect(conn, store, claim, REPO, clock=clock)
        effect, extra_refs = result
        assert [name for name, _ in extra_refs] == ["engineering_gate"]
        document = json.loads(store.read_verified(extra_refs[0][1]))
        assert document["schema_version"] == "engineering_gate.v1.0"
        assert "coverage" not in document["rows"]
        assert set(document["rows"]) >= {"imports", "readmes", "hygiene", "budgets", "lint", "hook"}
        assert document["ok"] is True, document["rows"]
        _commit(conn, clock, claim, result)
        wm = conn.execute("SELECT occurrence, receipt_ref FROM watermarks WHERE pipeline='nightly' "
                          "AND scope=? AND stage='engineering_gate'", (scope,)).fetchone()
        assert wm["occurrence"] == SESSION
        assert wm["receipt_ref"] == extra_refs[0][1].content_hash
    finally:
        conn.close()


# --------------------------------------------------------------------------
# publication
# --------------------------------------------------------------------------


def _publish_raw(store, conn, clock, data, schema_ref):
    ref = store.publish_bytes(data, schema_ref=schema_ref)
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return ref


#: A synthetic, base64-hidden AWS-access-key-shaped string — hidden so the
#: repo's own hygiene scan (this file is itself scanned) never flags this
#: file, while the *decoded* bytes at test runtime still match
#: ``checks.repo_hygiene.CREDENTIAL_PATTERNS`` for the planted-secret scenario.
_FAKE_CREDENTIAL = base64.b64decode(b"QUtJQUFCQ0RFRkdISUpLTE1OT1A=").decode()


def _bundle_tar(*, secret=False):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        data = b"hello board\n" + ((_FAKE_CREDENTIAL + "\n").encode() if secret else b"")
        info = tarfile.TarInfo("board.html")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _publication_setup(conn, clock, supervisor, store, *, scope, session,
                       selfcheck="ok", engineering="ok", secret=False,
                       decisions_session=None, no_entry=False, finality_session=None):
    """A publication job with real, resolvable job-id bindings to a fake
    render bundle plus fake selfcheck/engineering-gate parent outputs.

    ``decisions_session`` lets the decisions watermark be seeded at a
    session other than the release's own (for the earlier-session refusal
    case); ``no_entry`` seeds no rows at all but still advances the
    watermark, mirroring a genuine no-entry night's commit. ``finality_session``
    lets the bound finality document resolve somewhere other than ``session``
    (P2-C03 walk-back); it defaults to ``session`` (no walk-back).
    """
    _seed_decisions(conn, clock, scope, decisions_session or session,
                    predictions=() if no_entry else [_row("evt-1", "pred")])
    tag = scope + ":" + session
    finality_doc = {"date": finality_session or session, "is_final": True, "market_wide": True,
                    "daily_share": 1.0, "chain_share": 1.0, "covered": 1, "detail": "final"}
    finality_ref = _publish(store, conn, clock, finality_doc, "legacy_action.v1.0")
    finality_job = _succeed_parent(conn, clock, supervisor, key="fin-" + tag,
                                   output_name="legacy_finality", ref=finality_ref)
    bundle_ref = _publish_raw(store, conn, clock, _bundle_tar(secret=secret), "legacy_action.v1.0")
    projection_job = _succeed_parent(conn, clock, supervisor, key="proj-" + tag,
                                     output_name="legacy_render", ref=bundle_ref)
    input_bindings = {"bundle.tar": projection_job + "#legacy_render",
                      "finality.json": finality_job + "#legacy_finality"}
    dependency_job_ids = [projection_job, finality_job]
    if selfcheck != "missing":
        selfcheck_ref = _publish(store, conn, clock, {"ok": selfcheck == "ok"}, "legacy_action.v1.0")
        selfcheck_job = _succeed_parent(conn, clock, supervisor, key="self-" + tag,
                                        output_name="legacy_selfcheck", ref=selfcheck_ref)
        input_bindings["selfcheck.json"] = selfcheck_job + "#legacy_selfcheck"
        dependency_job_ids.append(selfcheck_job)
    engineering_ref = _publish(store, conn, clock, {"ok": engineering == "ok"}, "engineering_gate.v1.0")
    engineering_job = _succeed_parent(conn, clock, supervisor, key="eng-" + tag,
                                      output_name="engineering_gate", ref=engineering_ref)
    input_bindings["engineering_gate.json"] = engineering_job + "#engineering_gate"
    dependency_job_ids.append(engineering_job)

    claim = _submit_and_claim(
        conn, clock, supervisor, kind="publication", key="pub-" + tag,
        parameters=_params("publication", session, scope, input_bindings=input_bindings),
        dependency_job_ids=tuple(dependency_job_ids))
    resolve_and_record(conn, store, claim)
    return claim


def test_publication_all_gates_valid_advances_current_and_watermarks(tmp_path):
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        claim = _publication_setup(conn, clock, supervisor, store, scope=scope, session=SESSION)
        result = publication_effect(conn, store, claim, root, REPO, clock=clock)
        assert result == (None, ())
        commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                       clock=clock)

        target = root / "releases" / scope
        assert release_current(target) is not None
        publication_wm = conn.execute(
            "SELECT occurrence FROM watermarks WHERE pipeline='nightly' AND scope=? "
            "AND stage='publication'", (scope,)).fetchone()
        delivery_wm = conn.execute(
            "SELECT occurrence FROM watermarks WHERE pipeline='nightly' AND scope=? "
            "AND stage='delivery'", (scope,)).fetchone()
        assert publication_wm["occurrence"] == SESSION
        assert delivery_wm["occurrence"] == SESSION
    finally:
        conn.close()


@pytest.mark.parametrize("kwargs,match", [
    ({"selfcheck": "missing"}, "PUBLICATION_REFUSED"),
    ({"selfcheck": "failed"}, "PUBLICATION_REFUSED"),
    ({"engineering": "failed"}, "PUBLICATION_REFUSED"),
    ({"secret": True}, "PUBLICATION_REFUSED"),
])
def test_publication_refused_when_a_gate_is_invalid(tmp_path, kwargs, match):
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        claim = _publication_setup(conn, clock, supervisor, store, scope=scope, session=SESSION,
                                   **kwargs)
        target = root / "releases" / scope
        with pytest.raises(OpsError, match=match):
            publication_effect(conn, store, claim, root, REPO, clock=clock)
        assert release_current(target) is None
    finally:
        conn.close()


def test_publication_refused_on_stale_fence(tmp_path):
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        claim = _publication_setup(conn, clock, supervisor, store, scope=scope, session=SESSION)
        with transaction(conn):
            conn.execute("UPDATE attempts SET lease_expires_at=? WHERE attempt_id=?",
                        ("2000-01-01T00:00:00.000000Z", claim.attempt_id))
        target = root / "releases" / scope
        with pytest.raises(OpsError, match="LEASE_LOST"):
            publication_effect(conn, store, claim, root, REPO, clock=clock)
        assert release_current(target) is None
    finally:
        conn.close()


def test_publication_older_occurrence_cannot_replace_newer(tmp_path):
    """Two different scopes, each staged and published while its own
    decisions watermark matched its own session — the strict per-session
    decision gate means one release can never legitimately see a stale
    decisions watermark, so this is the only way to independently publish
    two releases at different occurrences without either being refused for
    the wrong reason.

    ``releases`` carries no scope column (publication.py, as authored): its
    occurrence staleness check is global. Re-publishing the older release —
    still genuinely CURRENT for its own scope's pointer, so the per-scope
    "current changed" check passes — is still refused once a globally newer
    occurrence exists elsewhere, at the ``newest > row["occurrence"]`` line.
    """
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        older_scope = "shadow"
        older = _publication_setup(conn, clock, supervisor, store, scope=older_scope,
                                   session="2026-09-10")
        publication_effect(conn, store, older, root, REPO, clock=clock)
        older_target = root / "releases" / older_scope
        older_release_id = "rel" + content_hash([older_scope, "2026-09-10"]).split(":")[1][:24]
        pointer_after_older = release_current(older_target)

        newer_scope = "shadow:newer"
        newer = _publication_setup(conn, clock, supervisor, store, scope=newer_scope,
                                   session="2026-09-12")
        publication_effect(conn, store, newer, root, REPO, clock=clock)

        with pytest.raises(OpsError, match="STALE_EXPECTATION"):
            publish_local(conn, older, store, older_target, older_release_id,
                          scope=older_scope, clock=clock)
        assert release_current(older_target) == pointer_after_older
    finally:
        conn.close()


def test_publication_refused_when_decisions_watermark_is_an_earlier_session(tmp_path):
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        claim = _publication_setup(conn, clock, supervisor, store, scope=scope, session="2026-09-12",
                                   decisions_session="2026-09-11")
        target = root / "releases" / scope
        with pytest.raises(OpsError, match="PUBLICATION_REFUSED"):
            publication_effect(conn, store, claim, root, REPO, clock=clock)
        assert release_current(target) is None
    finally:
        conn.close()


def test_publication_passes_at_the_correct_session(tmp_path):
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        claim = _publication_setup(conn, clock, supervisor, store, scope=scope, session=SESSION,
                                   decisions_session=SESSION)
        publication_effect(conn, store, claim, root, REPO, clock=clock)
        assert release_current(root / "releases" / scope) is not None
    finally:
        conn.close()


def test_publication_passes_on_a_no_entry_night(tmp_path):
    """Zero decision rows still commits (export/release_intent still
    enqueued) and still advances the decisions watermark for its session —
    the decision gate cares about the watermark, never row count."""
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        claim = _publication_setup(conn, clock, supervisor, store, scope=scope, session=SESSION,
                                   no_entry=True)
        publication_effect(conn, store, claim, root, REPO, clock=clock)
        assert release_current(root / "releases" / scope) is not None
    finally:
        conn.close()


# --------------------------------------------------------------------------
# backup
# --------------------------------------------------------------------------


def test_backup_failure_retries_alone_other_watermarks_unchanged(tmp_path):
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        _seed_decisions(conn, clock, scope, SESSION, predictions=[_row("evt-1", "pred")])
        decisions_before = conn.execute(
            "SELECT receipt_ref FROM watermarks WHERE pipeline='nightly' AND scope=? "
            "AND stage='decisions'", (scope,)).fetchone()["receipt_ref"]

        def fault(point):
            if point == "after_manifest_before_ack":
                raise RuntimeError("synthetic backup failure")

        first = _submit_and_claim(conn, clock, supervisor, kind="backup", key="backup-1",
                                  parameters=_params("backup", SESSION, scope))
        with pytest.raises(RuntimeError, match="synthetic backup failure"):
            backup_effect(conn, store, first, root, clock=clock, fault=fault)

        # the effect is releasable immediately (no lease wait) and nothing
        # else moved
        effect_row = conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchone()
        assert effect_row["state"] == "pending"
        assert conn.execute("SELECT 1 FROM watermarks WHERE pipeline='nightly' AND scope=? "
                            "AND stage='backup'", (scope,)).fetchone() is None
        decisions_after = conn.execute(
            "SELECT receipt_ref FROM watermarks WHERE pipeline='nightly' AND scope=? "
            "AND stage='decisions'", (scope,)).fetchone()["receipt_ref"]
        assert decisions_after == decisions_before

        # retrying (a fresh claim, the same deterministic backup key) succeeds
        second = _submit_and_claim(conn, clock, supervisor, kind="backup", key="backup-2",
                                   parameters=_params("backup", SESSION, scope))
        result = backup_effect(conn, store, second, root, clock=clock)
        _commit(conn, clock, second, result)

        rows = conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchall()
        assert [r["state"] for r in rows] == ["delivered"]
        backup_wm = conn.execute("SELECT occurrence FROM watermarks WHERE pipeline='nightly' "
                                 "AND scope=? AND stage='backup'", (scope,)).fetchone()
        assert backup_wm["occurrence"] == SESSION
        decisions_final = conn.execute(
            "SELECT receipt_ref FROM watermarks WHERE pipeline='nightly' AND scope=? "
            "AND stage='decisions'", (scope,)).fetchone()["receipt_ref"]
        assert decisions_final == decisions_before
    finally:
        conn.close()


# --------------------------------------------------------------------------
# DAG: new stages, parents, bindings, submit_graph acceptance
# --------------------------------------------------------------------------


def test_new_stages_are_wired_with_parents_bindings_and_submit(tmp_path):
    plan = build_nightly_plan(str(REPO), SESSION)
    requests = build_legacy_job_requests(plan, tickers=("FAKE",), year_start=2025, year_end=2026)
    by_kind = {r.job.kind: r for r in requests}
    for kind in ("ledger_export", "engineering_gate", "publication", "backup"):
        assert kind in by_kind

    def job_id(kind):
        return job_id_for("shadow", by_kind[kind].idempotency_key)

    # stage -> emitted job kind: decision_commit/settlement/projection/selfcheck
    # run as their ``legacy_*`` action; ledger_export/engineering_gate/
    # publication/backup are pure stages whose kind IS the stage name.
    decision_commit_deps = by_kind["ledger_export"].job.dependency_job_ids
    assert job_id("legacy_decisions") in decision_commit_deps
    assert job_id("legacy_settlement") not in decision_commit_deps
    assert by_kind["engineering_gate"].job.dependency_job_ids == ()
    render_deps = by_kind["legacy_render"].job.dependency_job_ids
    assert job_id("ledger_export") in render_deps
    publication_deps = set(by_kind["publication"].job.dependency_job_ids)
    assert publication_deps == {job_id("legacy_selfcheck"), job_id("engineering_gate"),
                                job_id("legacy_render"), job_id("legacy_decisions"),
                                job_id("legacy_finality")}
    assert job_id("legacy_decisions") in by_kind["backup"].job.dependency_job_ids
    assert job_id("legacy_settlement") not in by_kind["backup"].job.dependency_job_ids

    for kind, effect_scope_expected in (("ledger_export", "shadow"), ("publication", "shadow")):
        assert by_kind[kind].job.parameters["effect_scope"] == effect_scope_expected

    conn = open_catalog(tmp_path / "ops.sqlite", clock=FakeClock())
    try:
        receipts = submit_graph(conn, registry(), POLICY, requests, clock=FakeClock())
        assert len(receipts) == len(requests)
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == len(requests)
    finally:
        conn.close()


def test_subset_ticker_run_uses_hashed_effect_scope():
    plan = build_nightly_plan(str(REPO), SESSION)
    full = build_legacy_job_requests(plan, tickers=("A", "B"), year_start=2025, year_end=2026,
                                     full_universe=("A", "B"))
    subset = build_legacy_job_requests(plan, tickers=("A",), year_start=2025, year_end=2026,
                                       full_universe=("A", "B"))
    full_scope = {r.job.kind: r.job.parameters["effect_scope"] for r in full}["ledger_export"]
    subset_scope = {r.job.kind: r.job.parameters["effect_scope"] for r in subset}["ledger_export"]
    assert full_scope == "shadow"
    assert subset_scope.startswith("shadow:") and subset_scope != "shadow"
