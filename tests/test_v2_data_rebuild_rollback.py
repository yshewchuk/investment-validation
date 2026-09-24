"""P2-7/Task7b, §10, D16 (synthetic): candidate rebuild write confinement,
comparison, promotion and rollback.

Real-data runs of ``engine.data.rebuild.rebuild`` are a later, approved heavy
session (task brief "out of scope"). This module's "tiny monkeypatchable
builder path" (task brief's own phrase) is
``engine.v2.ops.snapshot_import.run_legacy_rebuild``, monkeypatched to write
the SAME synthetic store ``tests.test_v2_data_import.build_legacy_store``
already builds, instead of shelling out to the real legacy rebuild against
real raw data. Every other piece of the pipeline — inspection, publish,
commit, comparison, promotion, rollback — is the real production code.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts import RollbackReceipt  # noqa: E402
from engine.v2.data import catalog, manifests  # noqa: E402
from engine.v2.data.documents import decode_document  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.errors import fail as fail_data  # noqa: E402
from engine.v2.data.import_snapshot import plan_import  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactStore, SystemClock  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.ops.checkpoints import artifact as artifact_ref  # noqa: E402
from engine.v2.ops.errors import OpsError  # noqa: E402
from engine.v2.ops.snapshot_import import (  # noqa: E402
    protected_paths_hash,
    save_import_plan,
    submit_import,
    worker_legacy_rebuild_candidate,
)
from engine.v2.ops.snapshot_promotion import (  # noqa: E402
    build_comparison_receipt,
    legacy_rebuild_candidate_effect,
    promote,
    rollback,
)
from engine.v2.ops.stages import registry  # noqa: E402
from engine.v2.ops.submission import NamespacePolicy  # noqa: E402
from engine.v2.ops.supervisor import Service  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    RECEIPT,
    catalog_and_store,
    contract_for,
    contract_ref_for,
    fake_hash,
    publish_and_inspect,
)
from tests.ops_support import TEST_POLICY  # noqa: E402
from tests.test_v2_data_import import _run_until_terminal, _submit_and_run, build_legacy_store  # noqa: E402
from tests.test_v2_data_legacy_materialization import PANEL_ROWS, TIER4_ROWS  # noqa: E402

POLICY = NamespacePolicy({"operator": frozenset({"shadow", "smoke"})})


def _claim(candidate_root, protected_paths, protected_before_hash):
    return SimpleNamespace(spec=SimpleNamespace(parameters={
        "candidate_root": str(candidate_root),
        "protected_paths": tuple(str(p) for p in protected_paths),
        "protected_before_hash": protected_before_hash}))


def _import_scope(root, source_root, conn, clock, *, scope, idempotency_key,
                  expected_head=None, expected_generation=0):
    receipt, state, plan = _submit_and_run(
        root, source_root, conn, clock, idempotency_key=idempotency_key,
        expected_head=expected_head, expected_generation=expected_generation, scope=scope)
    return receipt, state, plan


def _head(conn, scope):
    row = conn.execute("SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope=?",
                       (scope,)).fetchone()
    return (row["snapshot_id"], row["generation"]) if row is not None else (None, 0)


# --------------------------------------------------------------------------
# candidate worker: write confinement
# --------------------------------------------------------------------------


def test_candidate_writes_stay_under_its_own_root(tmp_path, monkeypatch):
    candidate_root = tmp_path / "candidate"
    protected = [tmp_path / "live_store", tmp_path / "dashboard"]
    for path in protected:
        path.mkdir(parents=True)
        (path / "untouched.txt").write_text("do not touch")

    import engine.v2.ops.snapshot_import as si_mod
    monkeypatch.setattr(si_mod, "run_legacy_rebuild",
                        lambda root, repo_root, **kw: (build_legacy_store(Path(root)),
                                                       {"schema_version": "rebuild_report.v1.0"})[1])

    before_hash = protected_paths_hash(protected)
    worker_root = tmp_path / "staging"
    worker_root.mkdir()
    result = worker_legacy_rebuild_candidate(
        {"candidate_root": str(candidate_root), "protected_paths": [str(p) for p in protected],
         "tables": (), "sample": None},
        worker_root)
    assert result["completed_ids"] == ["legacy_rebuild_candidate"]
    assert (candidate_root / "data" / "features" / "SNAPSHOT").is_file()

    claim = _claim(candidate_root, protected, before_hash)
    effect, extra = legacy_rebuild_candidate_effect(None, None, claim, (), clock=SystemClock())
    assert effect is None and extra == ()
    for path in protected:
        assert (path / "untouched.txt").read_text() == "do not touch"


def test_candidate_write_outside_root_is_refused(tmp_path, monkeypatch):
    candidate_root = tmp_path / "candidate"
    protected = [tmp_path / "live_store"]
    protected[0].mkdir(parents=True)
    (protected[0] / "untouched.txt").write_text("do not touch")

    import engine.v2.ops.snapshot_import as si_mod

    def _leaky_rebuild(root, repo_root, **kw):
        build_legacy_store(Path(root))
        # A planted bug: the rebuild also writes outside its own root.
        (protected[0] / "untouched.txt").write_text("mutated by a buggy rebuild")
        return {"schema_version": "rebuild_report.v1.0"}

    monkeypatch.setattr(si_mod, "run_legacy_rebuild", _leaky_rebuild)
    before_hash = protected_paths_hash(protected)
    worker_root = tmp_path / "staging"
    worker_root.mkdir()
    worker_legacy_rebuild_candidate(
        {"candidate_root": str(candidate_root), "protected_paths": [str(p) for p in protected],
         "tables": (), "sample": None},
        worker_root)

    claim = _claim(candidate_root, protected, before_hash)
    with pytest.raises(OpsError) as excinfo:
        legacy_rebuild_candidate_effect(None, None, claim, (), clock=SystemClock())
    assert excinfo.value.code == "VALIDATION_FAILED"


# --------------------------------------------------------------------------
# end to end: rebuild candidate -> import -> compare -> promote -> rollback
# --------------------------------------------------------------------------


def test_candidate_promotion_and_rollback(tmp_path, monkeypatch):
    ops_root = tmp_path / "ops"
    ops_root.mkdir()
    target_source = tmp_path / "legacy_store"
    target_source.mkdir()
    build_legacy_store(target_source)

    clock = SystemClock()
    conn = open_catalog(ops_root / "ops.sqlite", clock=clock)
    try:
        # 1. Import the original ("target") store under its production scope.
        receipt1, state1, _ = _import_scope(ops_root, target_source, conn, clock,
                                            scope="legacy_primary", idempotency_key="target-1")
        assert state1 == "succeeded", conn.execute(
            "SELECT failure_json FROM jobs WHERE job_id=?", (receipt1.job_id,)).fetchone()[0]
        original_snapshot_id, original_generation = _head(conn, "legacy_primary")

        objects_before = conn.execute("SELECT COUNT(*) FROM data_objects").fetchone()[0]
        fragments_before = conn.execute("SELECT COUNT(*) FROM data_fragments").fetchone()[0]
        versions_before = conn.execute("SELECT COUNT(*) FROM data_dataset_versions").fetchone()[0]
        snapshots_before = conn.execute("SELECT COUNT(*) FROM data_snapshots").fetchone()[0]

        # 2. Run the candidate rebuild worker (monkeypatched builder), then
        #    import ITS root under a private candidate scope, through the
        #    same §7 pipeline.
        candidate_root = tmp_path / "candidate"
        import engine.v2.ops.snapshot_import as si_mod
        monkeypatch.setattr(si_mod, "run_legacy_rebuild",
                            lambda root, repo_root, **kw: build_legacy_store(
                                Path(root), rows_per_part=3) or {"schema_version": "rebuild_report.v1.0"})
        worker_root = tmp_path / "candidate_worker"
        worker_root.mkdir()
        worker_legacy_rebuild_candidate(
            {"candidate_root": str(candidate_root), "protected_paths": (), "tables": (), "sample": None},
            worker_root)

        candidate_scope = "candidate:job-1"
        receipt2, state2, _ = _import_scope(ops_root, candidate_root, conn, clock,
                                            scope=candidate_scope, idempotency_key="candidate-1")
        assert state2 == "succeeded", conn.execute(
            "SELECT failure_json FROM jobs WHERE job_id=?", (receipt2.job_id,)).fetchone()[0]
        candidate_snapshot_id, _ = _head(conn, candidate_scope)

        store = ArtifactStore(ops_root)

        # 3. Promote without a comparison receipt is refused.
        with pytest.raises(OpsError):
            promote(conn, store, candidate_scope=candidate_scope, target_scope="legacy_primary",
                   expected_snapshot_id=original_snapshot_id, expected_generation=original_generation,
                   comparison_receipt_id="art_does_not_exist", clock=clock)

        comparison_ref = build_comparison_receipt(
            conn, store, candidate_scope=candidate_scope, target_scope="legacy_primary", clock=clock)

        # 4. Promote with a stale expected generation is a conflict.
        with pytest.raises(DataError) as stale:
            promote(conn, store, candidate_scope=candidate_scope, target_scope="legacy_primary",
                   expected_snapshot_id=original_snapshot_id, expected_generation=original_generation + 5,
                   comparison_receipt_id=comparison_ref.artifact_id, clock=clock)
        assert stale.value.code == "SNAPSHOT_CONFLICT"
        assert _head(conn, "legacy_primary") == (original_snapshot_id, original_generation)

        # 5. A valid promotion moves the target head to the candidate snapshot.
        promote(conn, store, candidate_scope=candidate_scope, target_scope="legacy_primary",
               expected_snapshot_id=original_snapshot_id, expected_generation=original_generation,
               comparison_receipt_id=comparison_ref.artifact_id, clock=clock)
        promoted_snapshot_id, promoted_generation = _head(conn, "legacy_primary")
        assert promoted_snapshot_id == candidate_snapshot_id
        assert promoted_generation == original_generation + 1

        # 6. A failed candidate import (a different candidate scope) leaves
        #    the just-promoted target head, and every protected path outside
        #    its own root, unchanged.
        protected = [target_source]
        protected_before = protected_paths_hash(protected)
        failing_candidate_root = tmp_path / "candidate_failed"
        failing_candidate_root.mkdir()
        build_legacy_store(failing_candidate_root)
        import engine.v2.ops.snapshot_promotion as sp_mod
        real_commit = sp_mod.commit_snapshot_for_attempt
        monkeypatch.setattr(
            sp_mod, "commit_snapshot_for_attempt",
            lambda *a, **k: (_ for _ in ()).throw(fail_data("MANIFEST_CORRUPT", "injected fault")))
        receipt3, state3, _ = _import_scope(ops_root, failing_candidate_root, conn, clock,
                                            scope="candidate:job-2", idempotency_key="candidate-2")
        monkeypatch.setattr(sp_mod, "commit_snapshot_for_attempt", real_commit)
        assert state3 == "failed"
        assert _head(conn, "legacy_primary") == (promoted_snapshot_id, promoted_generation)
        assert protected_paths_hash(protected) == protected_before
        assert _head(conn, "candidate:job-2") == (None, 0)

        # Immutable rows are at least what the two successful imports (target,
        # candidate) produced; the failed candidate-2 import (step 6) added
        # nothing to them (its own transaction rolled back in full).
        assert conn.execute("SELECT COUNT(*) FROM data_objects").fetchone()[0] >= objects_before
        assert conn.execute("SELECT COUNT(*) FROM data_fragments").fetchone()[0] >= fragments_before
        assert conn.execute("SELECT COUNT(*) FROM data_dataset_versions").fetchone()[0] >= versions_before
        assert conn.execute("SELECT COUNT(*) FROM data_snapshots").fetchone()[0] >= snapshots_before
        pre_rollback_counts = _immutable_row_counts(conn)

        # 7. Rollback restores the prior head; both snapshots still resolve;
        #    no immutable row is deleted, rewritten, or newly minted for it.
        rollback(conn, store, scope="legacy_primary", to_snapshot_id=original_snapshot_id,
                expected_snapshot_id=promoted_snapshot_id, expected_generation=promoted_generation,
                clock=clock)
        rolled_back_id, rolled_back_generation = _head(conn, "legacy_primary")
        assert rolled_back_id == original_snapshot_id
        assert rolled_back_generation == promoted_generation + 1
        Repository(conn).resolve(original_snapshot_id)
        Repository(conn).resolve(candidate_snapshot_id)

        rollback_receipt = conn.execute(
            "SELECT update_receipt_ref FROM data_snapshot_heads WHERE scope='legacy_primary'").fetchone()[0]
        assert rollback_receipt is not None
        assert _immutable_row_counts(conn) == pre_rollback_counts

        # 8. rollback() now produces a strictly decodable RollbackReceipt
        #    (task P2-C01/P2-C07) naming the real prior/resulting snapshots
        #    and generations, with the generation strictly increasing.
        receipt_bytes = store.read_verified(artifact_ref(conn, store, rollback_receipt))
        receipt = decode_document(RollbackReceipt, json.loads(receipt_bytes))
        assert receipt.prior_snapshot_id == promoted_snapshot_id
        assert receipt.resulting_snapshot_id == original_snapshot_id
        assert receipt.prior_generation == promoted_generation
        assert receipt.resulting_generation == rolled_back_generation
        assert receipt.resulting_generation == receipt.prior_generation + 1
    finally:
        conn.close()


def test_comparison_receipt_on_unready_candidate_raises_typed_snapshot_not_ready(tmp_path):
    """``build_comparison_receipt`` refuses a candidate scope with no
    committed head via ``fail("SNAPSHOT_NOT_READY", ...)``. That code must be
    registered in ``engine.v2.contracts.operations.FAILURE_CODES`` -- before
    the fix it was not, so ``make_problem`` raised a bare ``ValueError``
    instead of the typed ``OpsError`` this test expects.
    """
    ops_root = tmp_path / "ops"
    ops_root.mkdir()
    clock = SystemClock()
    conn = open_catalog(ops_root / "ops.sqlite", clock=clock)
    try:
        store = ArtifactStore(ops_root)
        with pytest.raises(OpsError) as excinfo:
            build_comparison_receipt(conn, store, candidate_scope="candidate:never-imported",
                                     target_scope="legacy_primary", clock=clock)
        assert excinfo.value.code == "SNAPSHOT_NOT_READY"
        assert excinfo.value.problem.category == "dependency"
        assert excinfo.value.problem.retryable is True
    finally:
        conn.close()


def _immutable_row_counts(conn) -> tuple[int, int, int, int]:
    return (
        conn.execute("SELECT COUNT(*) FROM data_objects").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM data_fragments").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM data_dataset_versions").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM data_snapshots").fetchone()[0])


# --------------------------------------------------------------------------
# promote()'s refusal branches (§10 point 4), each proven to leave every
# scope head exactly where it was. Real committed snapshots throughout —
# ``data_scan_support``'s publish/inspect/commit pipeline, so
# ``build_comparison_receipt`` decides from catalog state, not from a
# hand-written document.
# --------------------------------------------------------------------------


def _reversioned(contract):
    """A second, genuinely registered contract for the same table: new
    contract_id, definition_hash recomputed by the real registration recipe
    (``legacy_mapping``'s own two-pass placeholder), so ``commit_snapshot``'s
    ``table_contract_hash`` re-verification accepts it."""
    fields = dataclasses.replace(contract, contract_id=contract.contract_id.replace(".v1", ".v2"),
                                 semantic_version=contract.semantic_version + ".candidate")
    placeholder = dataclasses.replace(fields, definition_hash="sha256:" + "0" * 64)
    return dataclasses.replace(placeholder, definition_hash=manifests.table_contract_hash(placeholder))


def _commit(conn, clock, store, *, scope, rows_by_table, contracts, receipt_id, expected_head=None,
            expected_generation=0):
    """One real ``catalog.commit_snapshot`` of single-fragment tables, as
    ``data_scan_support.commit_tables`` does but with an explicit head
    expectation — so a scope's head can be ADVANCED, not just created."""
    table_manifests, records, objects = {}, [], []
    for name, rows in rows_by_table.items():
        contract = contracts[name]
        ref = contract_ref_for(contract)
        record = publish_and_inspect(store, contract, ref, rows, partition_key="all")
        table_manifests[name] = manifests.dataset_manifest(
            ref, [record], knowledge_mode="reconstructed", coverage_receipt_refs=(RECEIPT,),
            availability_evidence_refs=())
        records.append(record)
        objects.append(record.object_ref)
    snap = manifests.snapshot_ref(table_manifests, calendar_version="cal.v1",
                                  source_priority_version="prio.v1",
                                  finality_receipt_refs=(RECEIPT,))
    catalog.commit_snapshot(conn, scope=scope, request_hash=fake_hash(receipt_id),
                            contracts=list(contracts.values()), objects=objects, records=records,
                            manifests=list(table_manifests.values()), snapshot=snap,
                            expected_head_snapshot_id=expected_head,
                            expected_head_generation=expected_generation, receipt_id=receipt_id,
                            attempt_id="att-" + receipt_id, fence=1, fence_check=lambda _c: None,
                            clock=clock, store=store)
    return snap.snapshot_id


def test_promote_refuses_when_candidate_and_target_disagree_on_a_table_contract(tmp_path):
    """The data behind promote()'s contract_mismatches refusal: identical
    bytes in both scopes, but the candidate's rebuild re-registered
    feature_panel under a second contract version. The comparison receipt
    must flag EXACTLY that one table (tier4_forecasts, identical in data and
    contract, must not be swept in) and promote() must refuse with the table
    list preserved — the live target head staying put through it."""
    conn, clock, store = catalog_and_store(tmp_path)
    try:
        panel, tier4 = contract_for("feature_panel"), contract_for("tier4_forecasts")
        rows = {"feature_panel": PANEL_ROWS[:1], "tier4_forecasts": TIER4_ROWS}
        target_id = _commit(conn, clock, store, scope="legacy_primary", rows_by_table=rows,
                            contracts={"feature_panel": panel, "tier4_forecasts": tier4},
                            receipt_id="r-mismatch-target")
        candidate_id = _commit(conn, clock, store, scope="candidate:mismatch", rows_by_table=rows,
                               contracts={"feature_panel": _reversioned(panel),
                                          "tier4_forecasts": tier4},
                               receipt_id="r-mismatch-candidate")
        assert candidate_id != target_id

        receipt = build_comparison_receipt(conn, store, candidate_scope="candidate:mismatch",
                                           target_scope="legacy_primary", clock=clock)
        document = json.loads(store.read_verified(artifact_ref(conn, store, receipt.artifact_id)))
        assert document["contract_mismatches"] == ["feature_panel"]

        with pytest.raises(OpsError) as refusal:
            promote(conn, store, candidate_scope="candidate:mismatch", target_scope="legacy_primary",
                    expected_snapshot_id=target_id, expected_generation=1,
                    comparison_receipt_id=receipt.artifact_id, clock=clock)
        assert refusal.value.code == "VALIDATION_FAILED"
        assert refusal.value.problem.message == "candidate and target disagree on a table contract"
        assert refusal.value.problem.details == {"tables": ["feature_panel"]}
        assert _head(conn, "legacy_primary") == (target_id, 1)
        assert _head(conn, "candidate:mismatch") == (candidate_id, 1)
    finally:
        conn.close()


def test_promote_refuses_a_comparison_receipt_presented_to_the_wrong_promotion(tmp_path):
    """The receipt pins BOTH scopes it was built for; presenting it to a
    different pairing — with either position differing while the other
    matches — must refuse before the catalog heads are even consulted,
    leaving every head untouched (and never minting an update receipt)."""
    conn, clock, store = catalog_and_store(tmp_path)
    try:
        panel = contract_for("feature_panel")
        target_id = _commit(conn, clock, store, scope="legacy_primary",
                            rows_by_table={"feature_panel": PANEL_ROWS[:1]},
                            contracts={"feature_panel": panel}, receipt_id="r-scope-target")
        candidate_id = _commit(conn, clock, store, scope="candidate:scope",
                               rows_by_table={"feature_panel": PANEL_ROWS},
                               contracts={"feature_panel": panel}, receipt_id="r-scope-candidate")
        receipt = build_comparison_receipt(conn, store, candidate_scope="candidate:scope",
                                           target_scope="legacy_primary", clock=clock)
        for swap in ({"candidate_scope": "legacy_primary", "target_scope": "legacy_primary"},
                     {"candidate_scope": "candidate:scope", "target_scope": "legacy_replica"}):
            with pytest.raises(OpsError) as refusal:
                promote(conn, store, expected_snapshot_id=target_id, expected_generation=1,
                        comparison_receipt_id=receipt.artifact_id, clock=clock, **swap)
            assert refusal.value.code == "VALIDATION_FAILED"
            assert refusal.value.problem.message == "comparison receipt does not match this promotion"
            assert refusal.value.problem.details == {"comparison_receipt_id": receipt.artifact_id}
        assert _head(conn, "legacy_primary") == (target_id, 1)
        assert _head(conn, "candidate:scope") == (candidate_id, 1)
        assert _head(conn, "legacy_replica") == (None, 0)
    finally:
        conn.close()


def test_promote_refuses_a_stale_comparison_receipt_and_leaves_the_promoted_head_intact(tmp_path):
    """The receipt pins the candidate snapshot IT compared. One real
    promotion (target head -> C1, generation bumped) then a second candidate
    commit moves the candidate head to C2; re-presenting the C1-pinned
    receipt must refuse STALE_EXPECTATION — with the just-promoted pointer
    and generation exactly intact, the generation NOT bumped a third time."""
    conn, clock, store = catalog_and_store(tmp_path)
    try:
        panel = contract_for("feature_panel")
        target_id = _commit(conn, clock, store, scope="legacy_primary",
                            rows_by_table={"feature_panel": PANEL_ROWS[:1]},
                            contracts={"feature_panel": panel}, receipt_id="r-stale-target")
        c1 = _commit(conn, clock, store, scope="candidate:stale",
                     rows_by_table={"feature_panel": PANEL_ROWS},
                     contracts={"feature_panel": panel}, receipt_id="r-stale-c1")
        receipt = build_comparison_receipt(conn, store, candidate_scope="candidate:stale",
                                           target_scope="legacy_primary", clock=clock)
        promoted_ref = promote(conn, store, candidate_scope="candidate:stale",
                               target_scope="legacy_primary", expected_snapshot_id=target_id,
                               expected_generation=1, comparison_receipt_id=receipt.artifact_id,
                               clock=clock)
        promoted = json.loads(store.read_verified(artifact_ref(conn, store, promoted_ref.artifact_id)))
        assert promoted["schema_version"] == "snapshot_update_receipt.v1.0"
        assert promoted["action"] == "promote"
        assert promoted["scope"] == "legacy_primary"
        assert promoted["from_snapshot_id"] == target_id
        assert promoted["to_snapshot_id"] == c1
        assert promoted["comparison_receipt_ref"] == receipt.artifact_id
        assert _head(conn, "legacy_primary") == (c1, 2)

        c2 = _commit(conn, clock, store, scope="candidate:stale",
                     rows_by_table={"feature_panel": PANEL_ROWS[1:]},
                     contracts={"feature_panel": panel}, receipt_id="r-stale-c2",
                     expected_head=c1, expected_generation=1)
        assert _head(conn, "candidate:stale") == (c2, 2)

        with pytest.raises(OpsError) as refusal:
            promote(conn, store, candidate_scope="candidate:stale", target_scope="legacy_primary",
                    expected_snapshot_id=c1, expected_generation=2,
                    comparison_receipt_id=receipt.artifact_id, clock=clock)
        assert refusal.value.code == "STALE_EXPECTATION"
        assert refusal.value.problem.message == \
            "comparison receipt is stale; the candidate head has moved"
        assert refusal.value.problem.details == {"candidate_scope": "candidate:stale"}
        assert _head(conn, "legacy_primary") == (c1, 2)
    finally:
        conn.close()
