"""D04 (Python half) + D12: idempotent, conflict-safe, atomic snapshot commit,
plus the rollback primitive — phase-2 guide §7.3, §10, §12.

Every commit here runs against a real SQLite catalog (``open_catalog``) and
real, content-derived ``FragmentRecord``/``DatasetManifest``/``SnapshotRef``
values built through ``engine.v2.data.manifests`` — reusing
``tests/test_v2_data_manifests.py``'s synthetic-fixture helpers exactly as the
task brief directs, rather than re-deriving fake identities here.

Two constructions make a genuine "same ID, different content" collision
reachable without hand-forging an inconsistent record (which
``verify_fragment_record``/``verify_dataset_manifest``/``verify_snapshot_ref``
would themselves catch as ``MANIFEST_CORRUPT`` before the catalog ever sees
it):

* a fragment's ``fragment_id`` excludes ``input_receipt_refs``, so
  ``_record_for(year, input_receipt_refs=...)`` with a different value is a
  self-consistent record sharing the same ``fragment_id``
  (``test_v2_data_manifests.py::test_fragment_id_stable_but_manifest_hash_moves_with_provenance``);
* a dataset version's / snapshot's id sorts its evidence refs while its
  ``manifest_hash`` covers the full, caller-ordered document, so the same
  evidence cited in a different order is a self-consistent manifest/snapshot
  sharing the same id but a different ``manifest_hash``.

An object conflict is built directly: the same ``object_id`` (content-derived
from ``ArtifactStore``, forced here by reusing the year-derived id a
synthetic ``FragmentInspection`` already carries) with a different
``content_hash`` — this changes the *fragment*'s id too (object content is
part of a fragment's identity payload), so the mutated fragment is a brand
new one, isolating the object-level conflict from a fragment-level one. A
contract conflict is built the same way one level up: the same
``contract_id`` with a different ``table_name``, its own ``definition_hash``
correctly recomputed so the pre-transaction verification step does not itself
reject it before the catalog gets a chance to.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.data import manifests  # noqa: E402
from engine.v2.data.catalog import commit_snapshot, move_head, record_failed_import  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import content_hash  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from tests.ops_support import FakeClock  # noqa: E402
from tests.test_v2_data_manifests import (  # noqa: E402
    _SEC_CONTRACT,
    _SEC_REF,
    RECEIPT_A,
    RECEIPT_B,
    _inspection_for,
    _record_for,
)

_TABLES = ("data_contracts", "data_objects", "data_fragments", "data_dataset_versions",
          "data_version_fragments", "data_snapshots", "data_snapshot_tables",
          "data_snapshot_heads", "data_import_receipts")


def _hash(label: str) -> str:
    return content_hash({"label": label})


def _noop_fence(conn) -> None:
    return None


def _catalog(tmp_path):
    clock = FakeClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    return conn, clock


def _row_counts(conn) -> dict:
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in _TABLES}


def _manifest_and_snapshot(records, *, coverage=(RECEIPT_A,), finality=(RECEIPT_A,)):
    manifest = manifests.dataset_manifest(
        _SEC_REF, records, knowledge_mode="reconstructed", coverage_receipt_refs=coverage,
        availability_evidence_refs=())
    snap = manifests.snapshot_ref(
        {"securities": manifest}, calendar_version="cal.v1", source_priority_version="prio.v1",
        finality_receipt_refs=finality)
    return manifest, snap


def _commit(conn, clock, records, *, scope="shadow", receipt_id, attempt_id="att-1", fence=1,
           expected_head_snapshot_id=None, expected_head_generation=0, contract=_SEC_CONTRACT,
           coverage=(RECEIPT_A,), finality=(RECEIPT_A,), fault=None):
    manifest, snap = _manifest_and_snapshot(records, coverage=coverage, finality=finality)
    receipt = commit_snapshot(
        conn, scope=scope, request_hash=_hash(f"{receipt_id}-request"), contracts=[contract],
        objects=[r.object_ref for r in records], records=list(records), manifests=[manifest],
        snapshot=snap, expected_head_snapshot_id=expected_head_snapshot_id,
        expected_head_generation=expected_head_generation, receipt_id=receipt_id,
        attempt_id=attempt_id, fence=fence, fence_check=_noop_fence, clock=clock, fault=fault)
    return receipt, snap


# --------------------------------------------------------------------------
# D04 (python half): same-ID identical payload is a no-op
# --------------------------------------------------------------------------


def test_same_id_identical_payload_is_a_noop_across_a_second_snapshot(tmp_path):
    conn, clock = _catalog(tmp_path)
    record_2024 = _record_for("2024")
    _commit(conn, clock, [record_2024], receipt_id="r1", scope="shadow")
    before = _row_counts(conn)

    record_2025 = _record_for("2025")
    _commit(conn, clock, [record_2024, record_2025], receipt_id="r2", scope="other")
    after = _row_counts(conn)

    assert after["data_contracts"] == before["data_contracts"]  # same contract_id reused, no-op
    assert after["data_objects"] == before["data_objects"] + 1  # only 2025's object is new
    assert after["data_fragments"] == before["data_fragments"] + 1  # only 2025's fragment is new
    assert after["data_dataset_versions"] == before["data_dataset_versions"] + 1  # new membership set
    assert after["data_snapshots"] == before["data_snapshots"] + 1


# --------------------------------------------------------------------------
# D04 (python half): same-ID different payload raises IDENTITY_CONFLICT,
# zero rows changed, for each of contract/object/fragment/dataset-version/snapshot
# --------------------------------------------------------------------------


def test_identity_conflict_contract_different_payload(tmp_path):
    conn, clock = _catalog(tmp_path)
    _commit(conn, clock, [_record_for("2024")], receipt_id="r1")
    before = _row_counts(conn)

    mutated = dataclasses.replace(_SEC_CONTRACT, table_name="not_securities")
    mutated = dataclasses.replace(mutated, definition_hash=manifests.table_contract_hash(mutated))
    assert mutated.contract_id == _SEC_CONTRACT.contract_id
    assert mutated.definition_hash != _SEC_CONTRACT.definition_hash

    with pytest.raises(DataError) as err:
        _commit(conn, clock, [_record_for("2025")], receipt_id="r2", scope="other", contract=mutated)
    assert err.value.code == "IDENTITY_CONFLICT"
    assert _row_counts(conn) == before


def test_identity_conflict_object_different_payload(tmp_path):
    conn, clock = _catalog(tmp_path)
    base = _record_for("2024")
    _commit(conn, clock, [base], receipt_id="r1")
    before = _row_counts(conn)

    base_insp = _inspection_for("2024")
    mutated_insp = dataclasses.replace(
        base_insp, object_ref=dataclasses.replace(base_insp.object_ref,
                                                   content_hash=_hash("mutated-object")))
    mutated = manifests.fragment_record(mutated_insp, _SEC_REF, input_receipt_refs=(RECEIPT_A,),
                                        import_request_hash=base.import_request_hash)
    assert mutated.object_ref.object_id == base.object_ref.object_id
    assert mutated.fragment_id != base.fragment_id  # object content is part of fragment identity

    with pytest.raises(DataError) as err:
        _commit(conn, clock, [mutated], receipt_id="r2", scope="other")
    assert err.value.code == "IDENTITY_CONFLICT"
    assert _row_counts(conn) == before


def test_identity_conflict_fragment_different_payload(tmp_path):
    conn, clock = _catalog(tmp_path)
    base = _record_for("2024")
    _commit(conn, clock, [base], receipt_id="r1")
    before = _row_counts(conn)

    mutated = _record_for("2024", input_receipt_refs=(RECEIPT_B,))
    assert mutated.fragment_id == base.fragment_id
    assert mutated.manifest_hash != base.manifest_hash

    with pytest.raises(DataError) as err:
        _commit(conn, clock, [mutated], receipt_id="r2", scope="other")
    assert err.value.code == "IDENTITY_CONFLICT"
    assert _row_counts(conn) == before


def test_identity_conflict_dataset_version_different_payload(tmp_path):
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _commit(conn, clock, [record], receipt_id="r1", coverage=(RECEIPT_A, RECEIPT_B))
    before = _row_counts(conn)

    manifest_b, snap_b = _manifest_and_snapshot([record], coverage=(RECEIPT_B, RECEIPT_A))
    manifest_a, _ = _manifest_and_snapshot([record], coverage=(RECEIPT_A, RECEIPT_B))
    assert manifest_a.dataset_version_ref.dataset_version_id == \
        manifest_b.dataset_version_ref.dataset_version_id
    assert manifest_a.dataset_version_ref.manifest_hash != manifest_b.dataset_version_ref.manifest_hash

    with pytest.raises(DataError) as err:
        commit_snapshot(
            conn, scope="other", request_hash=_hash("r2-request"), contracts=[_SEC_CONTRACT],
            objects=[record.object_ref], records=[record], manifests=[manifest_b], snapshot=snap_b,
            expected_head_snapshot_id=None, expected_head_generation=0, receipt_id="r2",
            attempt_id="att-1", fence=1, fence_check=_noop_fence, clock=clock)
    assert err.value.code == "IDENTITY_CONFLICT"
    assert _row_counts(conn) == before


def test_identity_conflict_snapshot_different_payload(tmp_path):
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _commit(conn, clock, [record], receipt_id="r1", finality=(RECEIPT_A, RECEIPT_B))
    before = _row_counts(conn)

    manifest, snap_a = _manifest_and_snapshot([record], finality=(RECEIPT_A, RECEIPT_B))
    _, snap_b = _manifest_and_snapshot([record], finality=(RECEIPT_B, RECEIPT_A))
    assert snap_a.snapshot_id == snap_b.snapshot_id
    assert snap_a.manifest_hash != snap_b.manifest_hash

    with pytest.raises(DataError) as err:
        commit_snapshot(
            conn, scope="other", request_hash=_hash("r2-request"), contracts=[_SEC_CONTRACT],
            objects=[record.object_ref], records=[record], manifests=[manifest], snapshot=snap_b,
            expected_head_snapshot_id=None, expected_head_generation=0, receipt_id="r2",
            attempt_id="att-1", fence=1, fence_check=_noop_fence, clock=clock)
    assert err.value.code == "IDENTITY_CONFLICT"
    assert _row_counts(conn) == before


# --------------------------------------------------------------------------
# manifests are re-verified before the transaction opens
# --------------------------------------------------------------------------


def test_row_sum_mismatch_is_refused_before_any_row_is_written(tmp_path):
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    manifest, snap = _manifest_and_snapshot([record])
    tampered = dataclasses.replace(manifest, row_count=manifest.row_count + 1)

    with pytest.raises(DataError) as err:
        commit_snapshot(
            conn, scope="shadow", request_hash=_hash("bad-request"), contracts=[_SEC_CONTRACT],
            objects=[record.object_ref], records=[record], manifests=[tampered], snapshot=snap,
            expected_head_snapshot_id=None, expected_head_generation=0, receipt_id="r-bad",
            attempt_id="att-1", fence=1, fence_check=_noop_fence, clock=clock)
    assert err.value.code == "MANIFEST_CORRUPT"
    assert _row_counts(conn) == {t: 0 for t in _TABLES}


# --------------------------------------------------------------------------
# D12: repeating a full import is idempotent (receipt-level short-circuit)
# --------------------------------------------------------------------------


def test_d12_repeating_the_same_receipt_is_idempotent(tmp_path):
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    receipt1, snap = _commit(conn, clock, [record], receipt_id="r1", scope="shadow")
    head1 = conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = 'shadow'").fetchone()

    # Same receipt_id, same content, the stale original expected_head_* — the
    # replay must not re-evaluate that expectation against the now-advanced
    # head at all.
    receipt2, _ = _commit(conn, clock, [record], receipt_id="r1", scope="shadow")
    head2 = conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = 'shadow'").fetchone()

    assert tuple(head1) == tuple(head2)
    assert receipt1.status == receipt2.status == "committed"
    assert receipt1.resulting_head_snapshot_id == receipt2.resulting_head_snapshot_id == snap.snapshot_id
    assert receipt1.resulting_head_generation == receipt2.resulting_head_generation


def test_d12_same_receipt_id_different_outcome_is_identity_conflict(tmp_path):
    conn, clock = _catalog(tmp_path)
    _commit(conn, clock, [_record_for("2024")], receipt_id="r1", scope="shadow")

    with pytest.raises(DataError) as err:
        _commit(conn, clock, [_record_for("2025")], receipt_id="r1", scope="other")
    assert err.value.code == "IDENTITY_CONFLICT"


# --------------------------------------------------------------------------
# D12: two connections race the same expected parent/generation
# --------------------------------------------------------------------------


def test_d12_concurrent_expected_parent_commits_one_winner_one_conflict(tmp_path):
    conn, clock = _catalog(tmp_path)
    base_record = _record_for("2024")
    _, snap_a = _commit(conn, clock, [base_record], receipt_id="r0", scope="shadow")

    conn2 = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    record_x = _record_for("2025")
    record_y = _record_for("2026")
    manifest_x, snap_x = _manifest_and_snapshot([base_record, record_x])
    manifest_y, snap_y = _manifest_and_snapshot([base_record, record_y])

    receipt_x = commit_snapshot(
        conn, scope="shadow", request_hash=_hash("x-request"), contracts=[_SEC_CONTRACT],
        objects=[base_record.object_ref, record_x.object_ref], records=[base_record, record_x],
        manifests=[manifest_x], snapshot=snap_x, expected_head_snapshot_id=snap_a.snapshot_id,
        expected_head_generation=1, receipt_id="rx", attempt_id="att-x", fence=1,
        fence_check=_noop_fence, clock=clock)

    with pytest.raises(DataError) as err:
        commit_snapshot(
            conn2, scope="shadow", request_hash=_hash("y-request"), contracts=[_SEC_CONTRACT],
            objects=[base_record.object_ref, record_y.object_ref], records=[base_record, record_y],
            manifests=[manifest_y], snapshot=snap_y, expected_head_snapshot_id=snap_a.snapshot_id,
            expected_head_generation=1, receipt_id="ry", attempt_id="att-y", fence=1,
            fence_check=_noop_fence, clock=clock)
    assert err.value.code == "SNAPSHOT_CONFLICT"

    # the catalog stays consistent: the winner still resolves fully
    resolved = Repository(conn2).resolve(receipt_x.resulting_head_snapshot_id)
    assert resolved.snapshot_id == snap_x.snapshot_id

    # a failed receipt can be recorded, without moving the head
    failed = record_failed_import(conn2, receipt_id="ry-failed", request_hash=_hash("y-request"),
                                  attempt_id="att-y", fence=1, problem=err.value.problem, clock=clock)
    assert failed.status == "conflict"
    head = conn2.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = 'shadow'").fetchone()
    assert head["snapshot_id"] == snap_x.snapshot_id
    assert head["generation"] == 2
    conn2.close()


# --------------------------------------------------------------------------
# rollback: move_head bumps the generation, keeps both snapshots resolvable
# --------------------------------------------------------------------------


def test_rollback_move_head_keeps_both_snapshots_resolvable(tmp_path):
    conn, clock = _catalog(tmp_path)
    record_a = _record_for("2024")
    _, snap_a = _commit(conn, clock, [record_a], receipt_id="ra", scope="shadow")

    record_b = _record_for("2025")
    manifest_ab, snap_b = _manifest_and_snapshot([record_a, record_b])
    commit_snapshot(
        conn, scope="shadow", request_hash=_hash("b-request"), contracts=[_SEC_CONTRACT],
        objects=[record_a.object_ref, record_b.object_ref], records=[record_a, record_b],
        manifests=[manifest_ab], snapshot=snap_b, expected_head_snapshot_id=snap_a.snapshot_id,
        expected_head_generation=1, receipt_id="rb", attempt_id="att-1", fence=1,
        fence_check=_noop_fence, clock=clock)

    before = _row_counts(conn)
    move_head(conn, scope="shadow", to_snapshot_id=snap_a.snapshot_id,
             expected_snapshot_id=snap_b.snapshot_id, expected_generation=2,
             receipt_ref="rollback-1", clock=clock)
    after = _row_counts(conn)

    head = conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = 'shadow'").fetchone()
    assert head["snapshot_id"] == snap_a.snapshot_id
    assert head["generation"] == 3
    # no immutable row changed — only data_snapshot_heads moved
    assert {k: v for k, v in after.items() if k != "data_snapshot_heads"} == \
        {k: v for k, v in before.items() if k != "data_snapshot_heads"}

    assert Repository(conn).resolve(snap_a.snapshot_id).snapshot_id == snap_a.snapshot_id
    assert Repository(conn).resolve(snap_b.snapshot_id).snapshot_id == snap_b.snapshot_id


def test_move_head_rejects_stale_expectation(tmp_path):
    conn, clock = _catalog(tmp_path)
    _, snap_a = _commit(conn, clock, [_record_for("2024")], receipt_id="ra", scope="shadow")

    with pytest.raises(DataError) as err:
        move_head(conn, scope="shadow", to_snapshot_id=snap_a.snapshot_id,
                 expected_snapshot_id="snap_" + "0" * 32, expected_generation=99,
                 receipt_ref="rollback-bad", clock=clock)
    assert err.value.code == "SNAPSHOT_CONFLICT"
