"""Guide §5.5 item 1 ("Separate a retry from a new same-session plan").

The Sep-14 2026 operator trace: a 4-ticker nightly for session 2026-09-10 was
planned and submitted, all 14 jobs were cancelled, then a resource-profile
merge (``ef7020a``) landed and the operator re-planned the identical
session/tickers/population/snapshot. The new plan's own ``submit`` call
returned ``IDEMPOTENCY_CONFLICT`` on the first stage -- ``nightly.py``'s
``_scope_hash`` derived every stage's idempotency key from tickers/years/
population/snapshot only, never the plan's own pinned implementation, legacy
manifest, or decision clock, so the new plan's stages collided with the old
(cancelled) plan's stage keys under a changed payload.

This file tests the fix (``nightly._plan_identity``, folded into
``_scope_hash``, plus the mirrored release identity in
``effects_graph._generation_ref``):

1. An identical resubmission (same plan, same or different CLI key) is a
   true retry: same jobs, no duplicates.
2. The operator's own scenario: plan -> submit -> cancel all -> change the
   legacy manifest -> re-plan -> submit succeeds with fresh job ids, and the
   old cancelled rows are untouched.
3. The general (non-nightly) idempotency contract this fix must not weaken:
   same key + different payload is still ``IDEMPOTENCY_CONFLICT``.
4. Release/publication identity distinguishes a permitted new generation;
   the first release stays readable; rollback (restaging the prior content
   under a fresh id, the only release-pointer path that exists) works.
5. Decision divergence: two generations that try to commit DIFFERENT content
   for the SAME scheduled decision occurrence never silently duplicate --
   the first committed prediction stays authoritative and the second is
   recorded as a durable divergence, without failing the job.

2026-09-14 follow-up (this file's second pass, still guide §5.5 item 1):
the "stop and report" finding above item 4 used to name -- publishing or
committing decisions for a SECOND generation of an already-run session was
refused outright -- is now resolved. ``outbox.watermark`` (``engine/v2/ops/
outbox.py``) is generation-scoped (a new ``generation`` column/key, default
``""`` for every pre-existing caller), so engineering_gate/ledger_export/
backup/publication/delivery each record one receipt PER GENERATION instead
of one per ``(scope, session)`` -- a second generation no longer collides
with an earlier one's already-completed receipt. ``publication.publish_local``
also now checks (``outbox.watermark_would_conflict``) whether its own
acknowledgement would be refused BEFORE it moves ``CURRENT``, closing the
window where a foreseeable refusal could leave the pointer naming an
unacknowledged release. Decisions/predictions stay generation-INDEPENDENT
by design (item 5): the first committed record is never superseded by a
later generation automatically; see ``engine.v2.ops.decision_commit`` and
``engine.v2.ledger.decisions.record_divergence``.
"""
from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash, to_document
from engine.v2.ops import cli
from engine.v2.ops import effects_graph
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.decision_commit import (
    commit_decisions_in_transaction,
    import_settlement_candidates_in_transaction,
    validated_decision_candidate,
)
from engine.v2.ops.errors import OpsError
from engine.v2.ops.input_bindings import record_resolved_bindings, resolve_bindings
from engine.v2.ops.lifecycle import Outcome, commit_attempt, request_cancel
from engine.v2.ops.nightly import _plan_identity, build_legacy_job_requests, build_nightly_plan
from engine.v2.ops.publication import current as release_current
from engine.v2.ops.publication import publish_local, stage_release
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor, claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, submit, submit_graph
from tests.ops_support import DEFAULT_POLICY, sample

REPO = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})
SESSION = "2026-09-10"


def _nightly_manifest_dict(note: str = "") -> dict:
    """A minimal manifest dict that satisfies the capture-inputs plan-time
    guard (``engine.v2.data.legacy_nightly_read_plan.manifest_problems``) --
    this file's jobs are cancelled or never claimed, so nothing actually
    reads these paths; they only need to be SHAPED like a real nightly
    capture so ``cli.py``'s guard (added alongside ``capture_inputs.py``)
    admits the plan. ``note`` varies the manifest's own content hash, which
    is exactly what this file's ``manifest_v1``/``manifest_v2`` scenario
    needs -- a different legacy manifest identity between generations.
    """
    from engine.v2.data.legacy_nightly_read_plan import NIGHTLY_CAPTURE_IMPLEMENTATION_REF

    tables = ("daily_market", "option_chains", "earnings_events", "trades")
    file_refs = [{"path": f"data/curated/{table}/year=2024/part-0000.parquet",
                 "content_hash": "sha256:" + "0" * 64, "byte_size": 1} for table in tables]
    file_refs.append({"path": "data/raw/fetch/orats/ab/placeholder.meta.json",
                      "content_hash": "sha256:" + "0" * 64, "byte_size": 1})
    return {"manifest_id": "m1", "note": note, "file_refs": file_refs, "table_contract_refs": [],
            "registry_and_model_refs": ["placeholder::sha256:" + "0" * 64],
            "calendar_ref": "placeholder::sha256:" + "0" * 64, "selected_session": SESSION,
            "finality_receipt_refs": [], "knowledge_mode_by_table": {},
            "availability_evidence_refs": [], "read_set_complete": True,
            "capture_implementation_ref": NIGHTLY_CAPTURE_IMPLEMENTATION_REF}


def _publish(store, conn, clock, value, schema_ref):
    ref = store.publish_bytes(json.dumps(value, sort_keys=True).encode(), schema_ref=schema_ref)
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return ref


# --------------------------------------------------------------------------
# 1. identical resubmission stays idempotent
# --------------------------------------------------------------------------


def test_identical_resubmission_of_the_same_plan_returns_the_same_jobs(tmp_path):
    plan = build_nightly_plan(str(REPO), SESSION)
    requests = build_legacy_job_requests(plan, tickers=("FAKE",), year_start=2025, year_end=2026)
    conn = open_catalog(tmp_path / "ops.sqlite", clock=SystemClock())
    try:
        first = submit_graph(conn, registry(), POLICY, requests, clock=SystemClock())
        second = submit_graph(conn, registry(), POLICY, requests, clock=SystemClock())
        assert [r.job_id for r in first] == [r.job_id for r in second]
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == len(requests)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 2. the operator's own Sep-14 scenario
# --------------------------------------------------------------------------


def test_operator_scenario_replan_after_manifest_change_gets_fresh_jobs_and_old_rows_survive(
        tmp_path, capsys):
    root = tmp_path / "ops"
    population_file = tmp_path / "population.json"
    population_file.write_text(json.dumps(["FAKE|TWIN-P|" + SESSION]))
    manifest_v1 = tmp_path / "manifest_v1.json"
    manifest_v1.write_text(json.dumps(_nightly_manifest_dict()))
    manifest_v2 = tmp_path / "manifest_v2.json"
    manifest_v2.write_text(json.dumps(_nightly_manifest_dict("post-ef7020a profile change")))

    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()

    # Generation 1: planned at "3d9b1b9", submitted, then every job cancelled
    # before anything ran -- exactly the operator's own sequence.
    assert cli.main(["--root", str(root), "plan", "nightly", "--as-of", SESSION,
                     "--tickers", "FAKE", "--input-manifest", str(manifest_v1),
                     "--expected-population", str(population_file)]) == 0
    plan_ref_1 = json.loads(capsys.readouterr().out)["plan_ref"]
    assert cli.main(["--root", str(root), "submit", "--plan", plan_ref_1,
                     "--idempotency-key", "gen1"]) == 0
    submission_1 = json.loads(capsys.readouterr().out)
    gen1_job_ids = [job["job_id"] for job in submission_1["jobs"]]
    assert len(gen1_job_ids) == 13  # the legacy-mode DAG (no snapshot materialize stage)

    conn = open_catalog(root / "catalog.sqlite", clock=SystemClock())
    try:
        for job_id in gen1_job_ids:
            request_cancel(conn, job_id, None, clock=SystemClock())
        states = {row[0] for row in conn.execute(
            "SELECT state FROM jobs WHERE job_id IN (" + ",".join("?" * len(gen1_job_ids)) + ")",
            gen1_job_ids)}
        assert states <= {"cancelled", "blocked"}
    finally:
        conn.close()

    # Generation 2: re-planned after a manifest change (standing in for the
    # operator's ``ef7020a`` code change -- both change the plan's own
    # ``implementation_ref``/manifest identity the same way).
    assert cli.main(["--root", str(root), "plan", "nightly", "--as-of", SESSION,
                     "--tickers", "FAKE", "--input-manifest", str(manifest_v2),
                     "--expected-population", str(population_file)]) == 0
    plan_ref_2 = json.loads(capsys.readouterr().out)["plan_ref"]
    assert plan_ref_2 != plan_ref_1

    # This is the operator's actual failure: before the fix, this raised
    # IDEMPOTENCY_CONFLICT on the first stage even though NOTHING with a
    # conflicting digest should still be "in the way" -- the old rows are
    # cancelled, and this is a different plan.
    assert cli.main(["--root", str(root), "submit", "--plan", plan_ref_2,
                     "--idempotency-key", "gen2"]) == 0
    submission_2 = json.loads(capsys.readouterr().out)
    gen2_job_ids = [job["job_id"] for job in submission_2["jobs"]]
    assert len(gen2_job_ids) == 13
    assert set(gen2_job_ids).isdisjoint(gen1_job_ids)

    conn = open_catalog(root / "catalog.sqlite", clock=SystemClock())
    try:
        rows = {row["job_id"]: row["state"] for row in conn.execute(
            "SELECT job_id, state FROM jobs WHERE job_id IN (" +
            ",".join("?" * len(gen1_job_ids)) + ")", gen1_job_ids)}
        assert all(state in ("cancelled", "blocked") for state in rows.values())
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 26
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 3. documented CLI key semantics for nightly submission
# --------------------------------------------------------------------------


def test_same_plan_different_cli_idempotency_key_is_still_a_retry(tmp_path, capsys):
    root = tmp_path / "ops"
    population_file = tmp_path / "population.json"
    population_file.write_text(json.dumps(["FAKE|TWIN-P|" + SESSION]))
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(_nightly_manifest_dict()))
    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(root), "plan", "nightly", "--as-of", SESSION,
                     "--tickers", "FAKE", "--input-manifest", str(manifest_file),
                     "--expected-population", str(population_file)]) == 0
    plan_ref = json.loads(capsys.readouterr().out)["plan_ref"]

    assert cli.main(["--root", str(root), "submit", "--plan", plan_ref,
                     "--idempotency-key", "s1"]) == 0
    jobs_1 = [j["job_id"] for j in json.loads(capsys.readouterr().out)["jobs"]]
    assert cli.main(["--root", str(root), "submit", "--plan", plan_ref,
                     "--idempotency-key", "s2"]) == 0
    jobs_2 = [j["job_id"] for j in json.loads(capsys.readouterr().out)["jobs"]]
    assert jobs_1 == jobs_2

    conn = sqlite3.connect(root / "catalog.sqlite")
    try:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == len(jobs_1)
    finally:
        conn.close()


def test_same_key_different_payload_is_still_idempotency_conflict(tmp_path):
    """The GENERAL contract ``submission.py`` documents: for a kind where the
    supplied key IS the identity (every non-nightly plan), the same key with
    a different payload must still refuse. Nightly's own fix (job identity
    bound to the plan, not the CLI key) must not weaken this."""
    conn = open_catalog(tmp_path / "ops.sqlite", clock=SystemClock())
    try:
        def job(value):
            return JobSpec(kind="artifact_check", implementation_ref="x", spec_hash=None,
                           environment_ref="x", parameters={"expected_ids": (), "input_bindings": None},
                           output_namespace="shadow", resource_class="delivery",
                           retry_policy_ref="bounded", checkpoint_contract_ref="receipt.v1.0")

        first = SubmitRequest(namespace="shadow", idempotency_key="dup", principal="operator",
                              job=job(1))
        submit(conn, registry(), POLICY, first, clock=SystemClock())
        second = SubmitRequest(namespace="shadow", idempotency_key="dup", principal="operator",
                               job=JobSpec(kind="artifact_check", implementation_ref="y",
                                          spec_hash=None, environment_ref="x",
                                          parameters={"expected_ids": (), "input_bindings": None},
                                          output_namespace="shadow", resource_class="delivery",
                                          retry_policy_ref="bounded",
                                          checkpoint_contract_ref="receipt.v1.0"))
        with pytest.raises(OpsError) as excinfo:
            submit(conn, registry(), POLICY, second, clock=SystemClock())
        assert excinfo.value.code == "IDEMPOTENCY_CONFLICT"
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 4a. plan identity: pure unit coverage
# --------------------------------------------------------------------------


def test_plan_identity_changes_with_implementation_manifest_or_decision_clock():
    base = {"implementation_ref": "impl-1", "decision_clock": "2026-09-10T00:00:00.000000Z"}
    identity = _plan_identity(base, ("art_m1",))
    assert identity == {"implementation_ref": "impl-1",
                        "decision_clock": "2026-09-10T00:00:00.000000Z",
                        "legacy_manifest_ref": "art_m1"}
    assert _plan_identity(dict(base, implementation_ref="impl-2"), ("art_m1",)) != identity
    assert _plan_identity(dict(base, decision_clock="2026-09-11T00:00:00.000000Z"),
                          ("art_m1",)) != identity
    assert _plan_identity(base, ("art_m2",)) != identity
    assert _plan_identity(base, ("art_m1",)) == identity  # same inputs, same identity


def test_generation_ref_distinguishes_plans_and_falls_back_when_absent():
    def claim(**params):
        return SimpleNamespace(spec=SimpleNamespace(parameters=params))

    # Absent (a pre-existing lower-level test claim, or a barrier-only kind
    # that never sets deployment/decision_clock): empty, the old formula.
    assert effects_graph._generation_ref(claim(session=SESSION)) == ""

    gen1 = claim(deployment="shadow:impl-1", decision_clock="2026-09-10T00:00:00.000000Z",
                input_bindings={"legacy_manifest.json": "art_m1"})
    gen2 = claim(deployment="shadow:impl-2", decision_clock="2026-09-10T00:00:00.000000Z",
                input_bindings={"legacy_manifest.json": "art_m1"})
    gen1_retry = claim(deployment="shadow:impl-1", decision_clock="2026-09-10T00:00:00.000000Z",
                       input_bindings={"legacy_manifest.json": "art_m1"})
    assert effects_graph._generation_ref(gen1) == effects_graph._generation_ref(gen1_retry)
    assert effects_graph._generation_ref(gen1) != effects_graph._generation_ref(gen2)
    with_projection_a = claim(deployment="shadow:impl-1", decision_clock="2026-09-10T00:00:00.000000Z",
                              input_bindings={"legacy_manifest.json": "art_m1",
                                              "projection_binding.json": "art_projection_a"})
    with_projection_b = claim(deployment="shadow:impl-1", decision_clock="2026-09-10T00:00:00.000000Z",
                              input_bindings={"legacy_manifest.json": "art_m1",
                                              "projection_binding.json": "art_projection_b"})
    assert effects_graph._generation_ref(with_projection_a) != effects_graph._generation_ref(with_projection_b)


# --------------------------------------------------------------------------
# 4b. release identity: a second generation gets a distinct release, the
# first stays readable, and rollback (the only existing pointer-moving
# path) works.
# --------------------------------------------------------------------------


def _files(store, tag):
    return {"board.html": store.publish_bytes(("<html>" + tag + "</html>").encode(),
                                              schema_ref="release_file.v1.0")}


def _binding(release_id, occurrence, files):
    return content_hash({"release_id": release_id, "occurrence": occurrence, "files": files})


def _gate_receipt(store, kind, binding):
    return store.publish_bytes(json.dumps({"kind": kind, "status": "passed",
                                           "input_hash": binding}).encode(),
                               schema_ref="gate_receipt.v1.0")


def _gate(ref, binding):
    return {"ok": True, "receipt_ref": ref.content_hash, "input_hash": binding,
            "receipt_artifact": to_document(ref)}


def _all_gates(store, release_id, occurrence, files):
    binding = _binding(release_id, occurrence, files)
    return {kind: _gate(_gate_receipt(store, kind, binding), binding)
            for kind in ("decision", "projection", "security", "engineering")}


def _publish_and_stage(conn, store, claim, release_id, occurrence, tag, *, expected_current, clock):
    files = _files(store, tag)
    gates = _all_gates(store, release_id, occurrence, files)
    staged = stage_release(conn, store, release_id, occurrence, files,
                           expected_current=expected_current, gates=gates, clock=clock, claim=claim)
    assert staged["eligible"] is True
    return files


def test_second_generation_stages_a_distinct_release_without_manifest_conflict(tmp_path):
    """The part of guide §5.5 item 1 this task's fix actually delivers:
    ``stage_release``'s own "release manifest changed" ``IDEMPOTENCY_CONFLICT``
    (content-hash-keyed by ``release_id`` alone) no longer fires for a
    genuinely new same-session generation, because ``publication_effect`` now
    gives it a DISTINCT release id (``effects_graph._generation_ref``). The
    first release's own row/manifest is untouched -- never overwritten,
    never re-hashed."""
    from tests.ops_support import catalog, enqueue_claim

    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    scope, occurrence = "shadow", SESSION

    gen1_ref = effects_graph._generation_ref(SimpleNamespace(spec=SimpleNamespace(parameters={
        "deployment": "shadow:impl-1", "decision_clock": "2026-09-10T00:00:00.000000Z",
        "input_bindings": {"legacy_manifest.json": "art_m1"}})))
    gen2_ref = effects_graph._generation_ref(SimpleNamespace(spec=SimpleNamespace(parameters={
        "deployment": "shadow:impl-2", "decision_clock": "2026-09-10T05:00:00.000000Z",
        "input_bindings": {"legacy_manifest.json": "art_m2"}})))
    assert gen1_ref != gen2_ref
    release_id_1 = "rel" + content_hash([scope, occurrence, gen1_ref]).split(":")[1][:24]
    release_id_2 = "rel" + content_hash([scope, occurrence, gen2_ref]).split(":")[1][:24]
    assert release_id_1 != release_id_2

    fenced_1 = enqueue_claim(conn, clock, supervisor, key="pub-gen1")
    _publish_and_stage(conn, store, fenced_1, release_id_1, occurrence, "gen1",
                       expected_current=None, clock=clock)
    manifest_1_before = conn.execute("SELECT manifest_hash FROM releases WHERE release_id=?",
                                     (release_id_1,)).fetchone()[0]

    fenced_2 = enqueue_claim(conn, clock, supervisor, key="pub-gen2")
    _publish_and_stage(conn, store, fenced_2, release_id_2, occurrence, "gen2",
                       expected_current=release_id_1, clock=clock)  # no IDEMPOTENCY_CONFLICT

    assert conn.execute("SELECT manifest_hash FROM releases WHERE release_id=?",
                        (release_id_1,)).fetchone()[0] == manifest_1_before
    assert conn.execute("SELECT COUNT(*) FROM releases").fetchone()[0] == 2


def test_second_generation_publishes_current_moves_both_releases_readable_rollback_works(
        tmp_path):
    """2026-09-14 follow-up, now resolved: a second generation for the SAME
    (scope, session) occurrence CAN become the published release. With a
    DISTINCT release id (previous test) AND its own generation-scoped
    "publication"/"delivery" watermark row (``publish_local``'s new
    ``generation=`` parameter, threaded from ``effects_graph._generation_ref``
    the same way the release id already is), generation 2's ``_acknowledge``
    no longer collides with generation 1's already-completed receipt.

    The prior release stays on disk and readable, and rollback to it works
    through the only release-pointer path that exists: staging its same
    content again under a fresh release id (a third "generation" -- the
    rollback decision itself) and publishing that.
    """
    from tests.ops_support import catalog, enqueue_claim

    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    target = tmp_path / "public"
    scope, occurrence = "shadow", SESSION

    def _ref(deployment, decision_clock, manifest):
        return effects_graph._generation_ref(SimpleNamespace(spec=SimpleNamespace(parameters={
            "deployment": deployment, "decision_clock": decision_clock,
            "input_bindings": {"legacy_manifest.json": manifest}})))

    gen1_ref = _ref("shadow:impl-1", "2026-09-10T00:00:00.000000Z", "art_m1")
    gen2_ref = _ref("shadow:impl-2", "2026-09-10T05:00:00.000000Z", "art_m2")
    gen3_ref = _ref("shadow:impl-1", "2026-09-10T09:00:00.000000Z", "art_m1")  # the rollback
    release_id_1 = "rel" + content_hash([scope, occurrence, gen1_ref]).split(":")[1][:24]
    release_id_2 = "rel" + content_hash([scope, occurrence, gen2_ref]).split(":")[1][:24]
    rollback_id = "rel" + content_hash([scope, occurrence, gen3_ref]).split(":")[1][:24]

    fenced_1 = enqueue_claim(conn, clock, supervisor, key="pub-gen1")
    _publish_and_stage(conn, store, fenced_1, release_id_1, occurrence, "gen1",
                       expected_current=None, clock=clock)
    assert publish_local(conn, fenced_1, store, target, release_id_1, scope=scope, clock=clock,
                         generation=gen1_ref)["delivered"] is True
    assert release_current(target) == release_id_1

    fenced_2 = enqueue_claim(conn, clock, supervisor, key="pub-gen2")
    _publish_and_stage(conn, store, fenced_2, release_id_2, occurrence, "gen2",
                       expected_current=release_id_1, clock=clock)
    assert publish_local(conn, fenced_2, store, target, release_id_2, scope=scope, clock=clock,
                         generation=gen2_ref)["delivered"] is True
    assert release_current(target) == release_id_2

    # Both releases are on disk, byte-for-byte what each generation shipped.
    assert (target / "releases" / release_id_1 / "board.html").read_text() == "<html>gen1</html>"
    assert (target / "releases" / release_id_2 / "board.html").read_text() == "<html>gen2</html>"

    # Each generation owns its own "publication" receipt; neither rewrote
    # the other's.
    gen1_wm = conn.execute(
        "SELECT receipt_ref FROM watermarks WHERE pipeline='nightly' AND scope=? "
        "AND stage='publication' AND generation=?", (scope, gen1_ref)).fetchone()
    gen2_wm = conn.execute(
        "SELECT receipt_ref FROM watermarks WHERE pipeline='nightly' AND scope=? "
        "AND stage='publication' AND generation=?", (scope, gen2_ref)).fetchone()
    assert gen1_wm["receipt_ref"] == release_id_1
    assert gen2_wm["receipt_ref"] == release_id_2

    # Rollback: restage generation 1's own content under a fresh release id
    # (the operator's rollback decision, its own generation) and publish it.
    _publish_and_stage(conn, store, fenced_2, rollback_id, occurrence, "gen1",
                       expected_current=release_id_2, clock=clock)
    assert publish_local(conn, fenced_2, store, target, rollback_id, scope=scope, clock=clock,
                         generation=gen3_ref)["delivered"] is True
    assert release_current(target) == rollback_id
    assert (target / "releases" / rollback_id / "board.html").read_text() == "<html>gen1</html>"
    # All three releases -- both real generations and the rollback -- remain
    # on disk and readable.
    assert conn.execute("SELECT COUNT(*) FROM releases WHERE published_at IS NOT NULL"
                        ).fetchone()[0] == 3


def test_publish_refusal_from_a_watermark_conflict_leaves_current_untouched(tmp_path):
    """Guide §5.4/§5.5 item 1 ordering fix, regression-tested directly: the
    old bug swapped ``CURRENT`` BEFORE ``_acknowledge``'s own ``watermark()``
    call, so a foreseeable ``IDEMPOTENCY_CONFLICT`` there still left
    ``CURRENT`` naming an unacknowledged release (``published_at IS NULL``).

    This forces that exact conflict -- two DIFFERENT release ids for the
    SAME generation at the SAME occurrence (what a bug elsewhere, or a
    caller reusing a generation identity in error, could produce) -- and
    proves the refusal is now known, via ``outbox.watermark_would_conflict``,
    and raised BEFORE ``CURRENT`` ever moves.
    """
    from tests.ops_support import catalog, enqueue_claim

    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    target = tmp_path / "public"
    scope, occurrence = "shadow", SESSION
    generation = "same-generation"

    fenced_1 = enqueue_claim(conn, clock, supervisor, key="pub-a")
    _publish_and_stage(conn, store, fenced_1, "rel-a", occurrence, "a",
                       expected_current=None, clock=clock)
    assert publish_local(conn, fenced_1, store, target, "rel-a", scope=scope, clock=clock,
                         generation=generation)["delivered"] is True
    assert release_current(target) == "rel-a"

    fenced_2 = enqueue_claim(conn, clock, supervisor, key="pub-b")
    _publish_and_stage(conn, store, fenced_2, "rel-b", occurrence, "b",
                       expected_current="rel-a", clock=clock)
    with pytest.raises(OpsError) as excinfo:
        publish_local(conn, fenced_2, store, target, "rel-b", scope=scope, clock=clock,
                      generation=generation)
    assert excinfo.value.code == "IDEMPOTENCY_CONFLICT"

    # The ordering fix: CURRENT never moved -- the refusal was known before
    # the swap, not the old bug of swapping then rolling only the SQL back
    # while the filesystem pointer stayed changed.
    assert release_current(target) == "rel-a"
    assert (target / "releases" / "rel-a" / "board.html").read_text() == "<html>a</html>"
    row = conn.execute("SELECT published_at FROM releases WHERE release_id='rel-b'").fetchone()
    assert row["published_at"] is None

    # A retry of the identical (refused) publish conflicts again, every
    # time -- this is a genuine identity error, not a transient crash.
    with pytest.raises(OpsError) as excinfo:
        publish_local(conn, fenced_2, store, target, "rel-b", scope=scope, clock=clock,
                      generation=generation)
    assert excinfo.value.code == "IDEMPOTENCY_CONFLICT"
    assert release_current(target) == "rel-a"


# --------------------------------------------------------------------------
# 5. conflicting same-occurrence decision content across generations is a
# durable DIVERGENCE, never a silent duplicate and never a job failure. The
# first committed prediction stays authoritative (guide §5.5 item 1,
# semantic #3); true superseding decisions remain ledger-phase scope.
# --------------------------------------------------------------------------


def _score_and_finality(strike=100.0):
    score = {"ticker": "FAKE", "event_id": "event-1", "event_date": SESSION,
             "as_of": SESSION, "entry_date": SESSION, "evidence_cutoff": SESSION,
             "strategy": "TWIN-P", "strike": strike, "expiry": "2026-10-16",
             "session": "AMC", "snapshot_hash": "sha256:" + "a" * 64}
    finality = {"date": SESSION, "is_final": True, "market_wide": True,
                "daily_share": 1.0, "chain_share": 1.0, "covered": 1}
    return score, finality


def _decision_evidence(score_ref, finality_ref, plan_ref, score, finality, plan):
    expected = plan["expected_population"]
    common = {"schema_version": "decision_receipt.v1.0", "score_artifact_id": score_ref.artifact_id,
              "score_content_hash": score_ref.content_hash, "finality_artifact_id": finality_ref.artifact_id,
              "finality_content_hash": finality_ref.content_hash, "plan_artifact_id": plan_ref.artifact_id,
              "plan_content_hash": plan_ref.content_hash, "session": SESSION,
              "deployment": plan["deployment"], "decision_clock": plan["decision_clock"],
              "expected_population": expected}
    receipts = {kind: dict(common, kind=kind) for kind in
               ("causality", "coverage", "finality", "selection", "replay")}
    receipts["causality"]["observed_cutoffs"] = {expected[0]: score["evidence_cutoff"]}
    receipts["coverage"]["observed_population"] = expected
    receipts["finality"].update(observed_finality_hash=content_hash(finality),
                                covered_tickers=[score["ticker"]])
    receipts["selection"]["eligible_candidate_keys"] = expected
    receipts["replay"].update(source_rows=[score], replayed_rows=[score],
                              source_rows_hash=content_hash([score]),
                              replayed_rows_hash=content_hash([score]), findings=[])
    return {"schema_version": "decision_evidence.v1.0", "receipts": receipts}


def _ensure_authority(conn, clock):
    from engine.v2.ledger.decisions import set_authority

    if conn.execute("SELECT 1 FROM decision_authority WHERE singleton=1").fetchone() is None:
        with transaction(conn):
            set_authority(conn, None, "catalog", "2026-09-10T00:00:00.000000Z")


def _commit_one_prediction(conn, store, clock, supervisor, *, key, deployment, decision_clock,
                           row_id):
    _ensure_authority(conn, clock)
    score, finality = _score_and_finality()
    # ``expected_population``/``candidates.population`` key off
    # ticker|strategy|event_date (``decision_validation.population_key``) --
    # narrower than ``row_id`` (``decision_replay.score_row_id``, which also
    # carries strike/expiry and is the candidate's own generation-independent
    # decision identity).
    population_key = "|".join((score["ticker"], score["strategy"], score["event_date"]))
    plan = {"schema_version": "decision_plan.v1.0", "session": SESSION,
            "deployment": deployment, "decision_clock": decision_clock,
            "expected_population": [population_key]}
    score_ref = _publish(store, conn, clock, {"rows": [score]}, "legacy_action.v1.0")
    finality_ref = _publish(store, conn, clock, finality, "legacy_action.v1.0")
    plan_ref = _publish(store, conn, clock, plan, "decision_plan.v1.0")
    evidence = _decision_evidence(score_ref, finality_ref, plan_ref, score, finality, plan)
    evidence_ref = _publish(store, conn, clock, evidence, "decision_evidence.v1.0")
    bindings = {"score.json": score_ref.artifact_id, "finality.json": finality_ref.artifact_id,
               "decision_plan.json": plan_ref.artifact_id, "decision_evidence.json": evidence_ref.artifact_id}
    refs = (score_ref.artifact_id, finality_ref.artifact_id, plan_ref.artifact_id, evidence_ref.artifact_id)
    job = JobSpec(kind="legacy_decisions", implementation_ref="x", spec_hash=None, environment_ref="x",
                 parameters={"expected_ids": ("legacy_decisions",), "session": SESSION,
                            "input_bindings": bindings},
                 input_refs=refs, output_namespace="shadow", resource_class="validation",
                 retry_policy_ref="bounded", checkpoint_contract_ref="legacy_action.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(namespace="shadow", idempotency_key=key,
          principal="operator", job=job), clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=supervisor,
                       clock=clock, registry=registry())
    assert claim is not None, "job did not claim"
    resolved = resolve_bindings(conn, store, claim.spec)
    with transaction(conn):
        record_resolved_bindings(conn, claim.attempt_id, resolved)
    decision = {"row_id": row_id, "event_id": "event-1", "ticker": "FAKE", "strategy": "TWIN-P",
               "event_date": SESSION, "as_of": SESSION, "written_at": decision_clock,
               "decision_ts": decision_clock, "snapshot_hash": score["snapshot_hash"],
               "score": score, "finality": finality}
    candidate_ref = _publish(store, conn, clock, {"rows": [decision]}, "legacy_action.v1.0")
    candidates, context = validated_decision_candidate(conn, store, claim, candidate_ref)
    with transaction(conn):
        receipts = commit_decisions_in_transaction(conn, claim, candidates, context, clock=clock)
    # Free this job's "validation" resource reservation so a SECOND
    # generation's own job can be admitted -- mirrors the real coordinator
    # finishing the attempt right after its commit succeeds.
    commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                   clock=clock)
    return receipts


def test_second_generation_conflicting_decision_content_is_recorded_as_a_divergence(tmp_path):
    """2026-09-14 follow-up, now resolved: two generations of the SAME
    session/scope that both try to commit a prediction for the SAME row
    (ticker/strategy/event_date/strike/expiry -- ``decision_replay.
    score_row_id``, generation-independent) under DIFFERENT ``deployment``/
    ``decision_clock`` (exactly what a genuinely new plan -- new code or a
    new decision_clock -- pins) no longer fails the second generation's job.
    ``engine.v2.ledger.decisions.insert`` still refuses the second INSERT
    (its stable ``decision_id`` vs. generation-scoped ``logical_key``
    contract, unchanged), but ``decision_commit.commit_decisions_in_transaction``
    now catches that refusal and records a durable, append-only divergence
    (``decision_divergences``: both payload hashes, both generation refs, a
    reason) instead of raising. The first committed prediction stays
    authoritative; true superseding decisions (an explicit ``supersedes``/
    ``supersede_reason``) remain ledger-phase scope, out of this task.
    """
    from tests.ops_support import catalog

    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    row_id = "FAKE|TWIN-P|" + SESSION + "|100.0|2026-10-16"

    receipts_1 = _commit_one_prediction(
        conn, store, clock, supervisor, key="gen1-decisions",
        deployment="shadow:impl-1", decision_clock="2026-09-10T21:00:00.000000Z", row_id=row_id)
    assert len(receipts_1) == 1
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    watermark_after_gen1 = conn.execute(
        "SELECT occurrence, receipt_ref FROM watermarks WHERE pipeline='nightly' AND scope='shadow' "
        "AND stage='decisions'").fetchone()
    assert watermark_after_gen1 is not None

    # The job succeeds -- no OpsError -- and commits nothing new for this row.
    receipts_2 = _commit_one_prediction(
        conn, store, clock, supervisor, key="gen2-decisions",
        deployment="shadow:impl-2", decision_clock="2026-09-11T03:00:00.000000Z", row_id=row_id)
    assert receipts_2 == []

    # No double-record: still exactly the one committed prediction, and it
    # is still generation 1's own content (downstream settlement reads this
    # row by decision_id, so it necessarily keeps using generation 1's
    # prediction).
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    committed = conn.execute("SELECT decision_id, payload_json FROM decisions").fetchone()
    assert committed["decision_id"] == "prediction:" + row_id
    assert json.loads(committed["payload_json"])["decision_ts"] == "2026-09-10T21:00:00.000000Z"

    # The decisions watermark for this (scope, session) still names
    # generation 1's own receipt -- a second, differently-keyed generation
    # never silently advanced it.
    watermark_after_gen2 = conn.execute(
        "SELECT occurrence, receipt_ref FROM watermarks WHERE pipeline='nightly' AND scope='shadow' "
        "AND stage='decisions'").fetchone()
    assert tuple(watermark_after_gen2) == tuple(watermark_after_gen1)

    # A durable divergence was recorded: both payload hashes, both
    # generation refs and a reason.
    divergences = conn.execute("SELECT * FROM decision_divergences").fetchall()
    assert len(divergences) == 1
    divergence = dict(divergences[0])
    assert divergence["decision_id"] == "prediction:" + row_id
    assert divergence["scope"] == "shadow" and divergence["occurrence"] == SESSION
    assert divergence["existing_payload_hash"] != divergence["attempted_payload_hash"]
    assert divergence["existing_generation_ref"] != divergence["attempted_generation_ref"]
    assert divergence["reason"]

    # An identical retry of generation 2's own (still-divergent) content is
    # a no-op: no second divergence row.
    _commit_one_prediction(
        conn, store, clock, supervisor, key="gen2-decisions-retry",
        deployment="shadow:impl-2", decision_clock="2026-09-11T03:00:00.000000Z", row_id=row_id)
    assert conn.execute("SELECT COUNT(*) FROM decision_divergences").fetchone()[0] == 1


def test_settlement_after_divergence_uses_generation_1s_committed_prediction(tmp_path):
    """The literal claim in guide §5.5 item 1 semantic #3: downstream
    settlement keeps using the first committed prediction.
    ``import_settlement_candidates_in_transaction`` validates an incoming
    settlement against whatever ``decisions`` holds for that decision_id
    (``_validate_settlement``) -- which is, and after a diverging generation
    2 stays, generation 1's row. If settlement had somehow resolved against
    generation 2's (never-committed) content instead, there would be no
    committed prediction to validate against and this would raise
    ``VALIDATION_FAILED``."""
    from tests.ops_support import catalog

    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    row_id = "FAKE|TWIN-P|" + SESSION + "|100.0|2026-10-16"

    _commit_one_prediction(
        conn, store, clock, supervisor, key="gen1-decisions",
        deployment="shadow:impl-1", decision_clock="2026-09-10T21:00:00.000000Z", row_id=row_id)
    _commit_one_prediction(
        conn, store, clock, supervisor, key="gen2-decisions",
        deployment="shadow:impl-2", decision_clock="2026-09-11T03:00:00.000000Z", row_id=row_id)
    assert conn.execute("SELECT COUNT(*) FROM decision_divergences").fetchone()[0] == 1

    settlement_job = JobSpec(kind="legacy_settlement", implementation_ref="x", spec_hash=None,
                             environment_ref="x",
                             parameters={"expected_ids": ("legacy_settlement",), "session": SESSION},
                             output_namespace="shadow", resource_class="legacy_rebuild",
                             retry_policy_ref="bounded", checkpoint_contract_ref="legacy_action.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(namespace="shadow", idempotency_key="settle",
          principal="operator", job=settlement_job), clock=clock)
    settlement_claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                                  supervisor=supervisor, clock=clock, registry=registry())
    assert settlement_claim is not None

    row = {"row_id": row_id, "ticker": "FAKE", "strategy": "TWIN-P", "event_date": SESSION,
           "status": "unresolvable", "resolved_at": SESSION + "T22:00:00.000000Z"}
    captured = [{"row": row,
                "original_b64": base64.b64encode(json.dumps(row).encode()).decode("ascii")}]
    candidate_ref = _publish(store, conn, clock, {"rows": captured}, "legacy_action.v1.0")
    with transaction(conn):
        receipts = import_settlement_candidates_in_transaction(
            conn, settlement_claim, candidate_ref, captured, clock=clock)
    assert len(receipts) == 1
    assert receipts[0]["decision_id"].startswith("outcome:" + row_id + ":")
    assert json.loads(receipts[0]["payload_json"])["row_id"] == row_id


def test_second_generation_identical_retry_of_the_same_decision_is_idempotent(tmp_path):
    """The companion positive case: retrying with the IDENTICAL deployment/
    decision_clock (a true retry of the same generation, e.g. after a
    worker crash) commits nothing new -- ``insert`` returns the existing
    row, matching the guide's "a retry of an identical decision is
    idempotent"."""
    from tests.ops_support import catalog

    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    row_id = "FAKE|TWIN-P|" + SESSION + "|100.0|2026-10-16"
    kwargs = dict(deployment="shadow:impl-1", decision_clock="2026-09-10T21:00:00.000000Z",
                 row_id=row_id)

    first = _commit_one_prediction(conn, store, clock, supervisor, key="retry-1", **kwargs)
    second = _commit_one_prediction(conn, store, clock, supervisor, key="retry-2", **kwargs)
    assert first[0]["decision_id"] == second[0]["decision_id"]
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
