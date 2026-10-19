"""``generic_streaming.StreamingTableWriter`` (#876 slice 2a).

A whole-table contract only: rows arrive in strictly increasing primary-key
order, are staged as fixed-size parquet row groups in the attempt's staging
directory, and ``finish()`` returns a ``GenericTableCandidate`` the unchanged
``commit_generic_table_candidate`` commits.  These tests cover the commit, the
batch-size independence of the published bytes, the refusals (leaving neither a
staged file nor a catalog commit), the terminal state machine and bounded
memory.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta

import pyarrow.parquet as pq
import pytest

from engine.v2.contracts import (
    ColumnContract,
    CoverageKey,
    CoverageOutcome,
    ObjectRef,
    TableContract,
    TableContractRef,
    TimeInterval,
)
from engine.v2.data import (
    generic_incremental,
    generic_streaming,
    manifests,
    objects,
)
from engine.v2.data import incremental as daily_incremental
from engine.v2.data.catalog import commit_snapshot
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, content_hash, to_document
from tests.ops_support import catalog

TABLE = "whole_panel"
_EPOCH = date(2025, 1, 1)

_WHOLE_COLUMNS = (
    ColumnContract(name="ticker", physical_type="string", nullable=False),
    ColumnContract(name="date", physical_type="string", nullable=False),
    ColumnContract(name="value", physical_type="float64", nullable=True),
)
_PARTITIONED_COLUMNS = _WHOLE_COLUMNS + (
    ColumnContract(name="year", physical_type="int64", nullable=False),
)


def _make_contract(contract_id, columns, *, primary_key, partition_columns):
    fields = dict(
        contract_id=contract_id,
        table_name=TABLE,
        semantic_version="v1",
        columns=columns,
        primary_key=primary_key,
        duplicate_policy="reject",
        foreign_keys=(),
        partition_columns=partition_columns,
        filterable_columns=(),
        orderable_columns=(),
        finality_semantics="final",
        provenance_semantics="source",
        coverage_semantics="full",
        schema_evolution_policy="reject",
        maximum_batch_rows=1_000_000,
    )
    placeholder = TableContract(definition_hash="sha256:" + "0" * 64, **fields)
    return TableContract(definition_hash=manifests.table_contract_hash(placeholder), **fields)


def _whole_contract():
    return _make_contract("ctr_whole", _WHOLE_COLUMNS,
                          primary_key=("ticker", "date"), partition_columns=())


def _partitioned_contract():
    return _make_contract("ctr_part", _PARTITIONED_COLUMNS,
                          primary_key=("ticker", "date"), partition_columns=("year",))


def _ref(contract):
    return TableContractRef(contract_id=contract.contract_id,
                            definition_hash=contract.definition_hash)


def _coverage(contract, *, state="complete", column="date"):
    key = CoverageKey(item_key="k", session_date="2025-01-02", ticker="AAA")
    coverage = daily_incremental.build_completed_coverage(
        _ref(contract), source="fixture", endpoint="fixture",
        interval=TimeInterval(column=column, start_inclusive="2025-01-02",
                              end_exclusive="2025-01-02T23:59:59"),
        expected=(key,),
        outcomes=(CoverageOutcome(key=key, status="present", receipt_id="r",
                                  revision_id="rev", finality="final"),),
        acquisition_receipt_refs=("r",), completed_at="2026-09-17T00:00:00Z")
    return replace(coverage, state=state)


def _row(i, *, ticker="AAA", value=None):
    return {"ticker": ticker, "date": (_EPOCH + timedelta(days=i)).isoformat(),
            "value": float(i) if value is None else value}


def _rows(count):
    return [_row(i) for i in range(count)]


def _parent_with_prior(conn, store, clock, contract, prior_rows, partition):
    ref = _ref(contract)
    published = store.publish_bytes(generic_incremental._parquet_bytes(contract, prior_rows),
                                    schema_ref="parquet_fragment.v1.0")
    obj = ObjectRef(kind="parquet_fragment", object_id=published.artifact_id,
                    content_hash=published.content_hash, byte_size=published.byte_size)
    inspection = objects.inspect_fragment(store, obj, contract, ref, partition)
    record = manifests.fragment_record(
        inspection, ref, input_receipt_refs=("r",),
        import_request_hash=content_hash({"seed": contract.table_name}))
    manifest = manifests.dataset_manifest(
        ref, (record,), knowledge_mode="reconstructed",
        coverage_receipt_refs=("r",), availability_evidence_refs=())
    snapshot = manifests.snapshot_ref(
        {contract.table_name: manifest}, calendar_version="cal.v1",
        source_priority_version="frozen", finality_receipt_refs=("r",))
    commit_snapshot(
        conn, scope="streaming", request_hash=content_hash({"base": contract.table_name}),
        contracts=(contract,), objects=(obj,), records=(record,), manifests=(manifest,),
        snapshot=snapshot, expected_head_snapshot_id=None, expected_head_generation=0,
        receipt_id="base", attempt_id="base", fence=1,
        fence_check=lambda _conn: None, clock=clock, store=store)
    return Repository(conn).resolve_full(snapshot.snapshot_id)


def _parent(conn, store, clock, contract, seed_row):
    return _parent_with_prior(
        conn, store, clock, contract, (seed_row,),
        generic_incremental._partition(contract, seed_row))


def _table_rows(resolved, store, contract):
    manifest = resolved.table_manifests[contract.table_name]
    ids = {item.fragment_id for item in manifest.fragment_refs}
    records = tuple(item for item in resolved.records if item.fragment_id in ids)
    return generic_incremental._load_rows(store, records, contract)


def _fragment_hash(candidate):
    fragment_id = candidate.table_manifest.fragment_refs[0].fragment_id
    record = next(item for item in candidate.records if item.fragment_id == fragment_id)
    return record.object_ref.content_hash


def _published_object(candidate):
    fragment_id = candidate.table_manifest.fragment_refs[0].fragment_id
    record = next(item for item in candidate.records if item.fragment_id == fragment_id)
    return record.object_ref


def _dataset_version(candidate):
    return candidate.table_manifest.dataset_version_ref.dataset_version_id


def _staging_entries(store, attempt_id):
    return sorted(item.name for item in store.staging_dir(attempt_id).iterdir())


def _head_snapshot(conn, scope="streaming"):
    row = conn.execute(
        "SELECT snapshot_id FROM data_snapshot_heads WHERE scope = ?", (scope,)).fetchone()
    return row["snapshot_id"]


def _setup(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    return conn, clock, store


def test_commit_scans_back_equal(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))
    rows = _rows(5)

    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-1", _coverage(contract))
    writer.append_rows(rows)
    candidate = writer.finish()

    assert candidate.rewritten_partitions == 1
    assert candidate.changeset.outcome == "changed"
    assert candidate.changeset.normalized_payloads == len(rows)
    assert candidate.changeset.changed_partitions == ("__whole__",)
    assert _staging_entries(store, "attempt-1") == []

    committed = generic_incremental.commit_generic_table_candidate(
        conn, store, candidate, scope="streaming",
        expected_head_snapshot_id=parent.snapshot.snapshot_id, expected_head_generation=1,
        clock=clock)
    resolved = Repository(conn).resolve_full(committed.resulting_head_snapshot_id)
    assert _table_rows(resolved, store, contract) == tuple(rows)


def test_batch_size_does_not_change_identity(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))
    rows = _rows(20)

    identities = {}
    for label, size in (("one", 1), ("seven", 7), ("all", len(rows))):
        writer = generic_streaming.StreamingTableWriter(
            store, parent, TABLE, "attempt-" + label, _coverage(contract))
        for start in range(0, len(rows), size):
            writer.append_rows(rows[start:start + size])
        candidate = writer.finish()
        identities[label] = (_fragment_hash(candidate), _dataset_version(candidate))

    assert len(set(identities.values())) == 1

    changed = [dict(row, value=row["value"] + 0.5) for row in rows]
    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-changed", _coverage(contract))
    writer.append_rows(changed)
    other = writer.finish()
    assert _fragment_hash(other) != identities["all"][0]
    assert _dataset_version(other) != identities["all"][1]


@pytest.mark.parametrize(
    ("label", "bad_rows", "message"),
    (
        ("out_of_order", [_row(2), _row(1)],
         "streamed rows are not in strictly increasing primary-key order"),
        ("duplicate_key", [_row(3), _row(3)],
         "streamed rows are not in strictly increasing primary-key order"),
        ("unknown_column", [dict(_row(0), bogus=1)],
         "table row does not satisfy contract columns"),
        ("null_non_nullable", [dict(_row(0), date=None)],
         "a streamed row omits a required column value"),
        ("wrong_type_float", [dict(_row(0), value="not-a-float")],
         "a streamed row has a value incompatible with its column type"),
        ("mixed_key_type", [_row(0), dict(_row(1), ticker=123)],
         "a streamed row has a value incompatible with its column type"),
        ("none_row", [None], "a streamed row must be a mapping"),
        ("list_row", [[1, 2]], "a streamed row must be a mapping"),
    ),
)
def test_bad_rows_refused_leave_nothing(tmp_path, label, bad_rows, message):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))

    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-" + label, _coverage(contract))
    with pytest.raises(DataError, match=r"^CONTRACT_MISMATCH: " + message) as exc:
        writer.append_rows(bad_rows)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert _staging_entries(store, "attempt-" + label) == []
    assert _head_snapshot(conn) == parent.snapshot.snapshot_id
    with pytest.raises(
            DataError,
            match=r"^INPUT_CHANGED: the streaming writer is no longer open") as terminal:
        writer.append_rows([_row(9)])
    assert terminal.value.code == "INPUT_CHANGED"


def test_non_sequence_batch_refused(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))

    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-nonseq", _coverage(contract))
    with pytest.raises(
            DataError,
            match=r"^CONTRACT_MISMATCH: streamed rows must be a sequence") as exc:
        writer.append_rows(None)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert _staging_entries(store, "attempt-nonseq") == []
    with pytest.raises(
            DataError,
            match=r"^INPUT_CHANGED: the streaming writer is no longer open") as terminal:
        writer.append_rows([_row(9)])
    assert terminal.value.code == "INPUT_CHANGED"


def test_staging_oserror_propagates(tmp_path, monkeypatch):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))

    def boom(self):
        raise OSError("disk full")

    monkeypatch.setattr(generic_streaming.StreamingTableWriter, "_write_row_group", boom)

    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-oserror", _coverage(contract))
    writer.append_rows(_rows(1))
    with pytest.raises(OSError, match="disk full"):
        writer.finish()
    assert _staging_entries(store, "attempt-oserror") == []
    assert _head_snapshot(conn) == parent.snapshot.snapshot_id


def test_temporal_value_mismatch_refused(tmp_path):
    conn, clock, store = _setup(tmp_path)
    columns = (
        ColumnContract(name="ticker", physical_type="string", nullable=False),
        ColumnContract(name="date", physical_type="string", nullable=False),
        ColumnContract(name="ts", physical_type="timestamp[us]", nullable=False),
    )
    contract = _make_contract("ctr_ts", columns,
                              primary_key=("ticker", "date"), partition_columns=())
    parent = _parent_with_prior(
        conn, store, clock, contract,
        [{"ticker": "AAA", "date": "2025-01-01", "ts": datetime(2025, 1, 1)}], "__whole__")

    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-ts", _coverage(contract))
    with pytest.raises(
            DataError,
            match=r"^CONTRACT_MISMATCH: a streamed row has a value incompatible with its column type"
    ) as exc:
        writer.append_rows([{"ticker": "AAA", "date": "2025-01-02", "ts": "2025-01-01"}])
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert _staging_entries(store, "attempt-ts") == []
    assert _head_snapshot(conn) == parent.snapshot.snapshot_id


def test_zero_rows_refused(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))

    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-zero", _coverage(contract))
    with pytest.raises(
            DataError,
            match=r"^CONTRACT_MISMATCH: a streaming table writer cannot finish with no rows"
    ) as exc:
        writer.finish()
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert _staging_entries(store, "attempt-zero") == []
    assert _head_snapshot(conn) == parent.snapshot.snapshot_id
    with pytest.raises(
            DataError,
            match=r"^INPUT_CHANGED: the streaming writer is no longer open") as terminal:
        writer.finish()
    assert terminal.value.code == "INPUT_CHANGED"


def test_non_whole_table_contract_refused(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _partitioned_contract()
    parent = _parent(conn, store, clock, contract,
                     dict(_row(0), year=2025))

    with pytest.raises(
            DataError,
            match=r"^UNSUPPORTED_CONTRACT: the streaming writer supports whole-table contracts only"
    ) as exc:
        generic_streaming.StreamingTableWriter(
            store, parent, TABLE, "attempt-part", _coverage(contract))
    assert exc.value.code == "UNSUPPORTED_CONTRACT"
    assert _staging_entries(store, "attempt-part") == []
    assert _head_snapshot(conn) == parent.snapshot.snapshot_id


def test_incomplete_coverage_refused(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))

    with pytest.raises(
            DataError,
            match=r"^INPUT_CHANGED: incomplete coverage cannot build a candidate") as exc:
        generic_streaming.StreamingTableWriter(
            store, parent, TABLE, "attempt-partial", _coverage(contract, state="partial"))
    assert exc.value.code == "INPUT_CHANGED"
    assert _staging_entries(store, "attempt-partial") == []
    assert _head_snapshot(conn) == parent.snapshot.snapshot_id


def test_append_after_finish_refused(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))

    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-finished", _coverage(contract))
    writer.append_rows(_rows(2))
    writer.finish()
    for action in (lambda: writer.append_rows([_row(5)]), writer.finish):
        with pytest.raises(
                DataError,
                match=r"^INPUT_CHANGED: the streaming writer is no longer open") as exc:
            action()
        assert exc.value.code == "INPUT_CHANGED"

    assert _staging_entries(store, "attempt-finished") == []
    assert _head_snapshot(conn) == parent.snapshot.snapshot_id


def test_abort_removes_staging_and_refuses_append(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))

    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-abort", _coverage(contract))
    writer.append_rows(_rows(2))
    writer.abort()
    assert _staging_entries(store, "attempt-abort") == []
    for action in (lambda: writer.append_rows([_row(5)]), writer.finish):
        with pytest.raises(
                DataError,
                match=r"^INPUT_CHANGED: the streaming writer is no longer open") as exc:
            action()
        assert exc.value.code == "INPUT_CHANGED"

    assert _head_snapshot(conn) == parent.snapshot.snapshot_id


def test_bounded_memory_writes_row_groups(tmp_path, monkeypatch):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))
    monkeypatch.setattr(generic_streaming, "ROW_GROUP_ROWS", 100)

    calls = []
    real_write = pq.ParquetWriter.write_table

    def spy(self, table, *args, **kwargs):
        calls.append(table.num_rows)
        return real_write(self, table, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetWriter, "write_table", spy)

    rows = _rows(250)
    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-memory", _coverage(contract))
    for start in range(0, len(rows), 3):
        writer.append_rows(rows[start:start + 3])
        assert len(writer._buffer) <= generic_streaming.ROW_GROUP_ROWS
    writer.finish()

    assert calls == [100, 100, 50]
    assert max(calls) <= 100


def test_cross_batch_order_refused(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))

    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-cross-batch", _coverage(contract))
    writer.append_rows(_rows(3))
    with pytest.raises(
            DataError,
            match=r"^CONTRACT_MISMATCH: streamed rows are not in strictly increasing "
                  r"primary-key order") as exc:
        writer.append_rows([_row(2)])
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert _staging_entries(store, "attempt-cross-batch") == []
    assert _head_snapshot(conn) == parent.snapshot.snapshot_id


def test_caller_mutation_does_not_change_published_rows(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))
    rows = _rows(5)
    originals = [dict(row) for row in rows]

    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-mutation", _coverage(contract))
    writer.append_rows(rows)
    for row in rows:
        row["value"] = -1.0
        row["ticker"] = "ZZZ"
    candidate = writer.finish()

    committed = generic_incremental.commit_generic_table_candidate(
        conn, store, candidate, scope="streaming",
        expected_head_snapshot_id=parent.snapshot.snapshot_id, expected_head_generation=1,
        clock=clock)
    resolved = Repository(conn).resolve_full(committed.resulting_head_snapshot_id)
    assert _table_rows(resolved, store, contract) == tuple(originals)


def test_row_groups_fixed_size_and_batch_independent(tmp_path, monkeypatch):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))
    group_rows = 100
    monkeypatch.setattr(generic_streaming, "ROW_GROUP_ROWS", group_rows)

    row_sets = {
        "below": _rows(group_rows - 1),
        "equal": _rows(group_rows),
        "above": _rows(group_rows + 1),
    }
    identities = {}
    for label, rows in row_sets.items():
        identities[label] = set()
        for batch_size in (1, 7, len(rows)):
            writer = generic_streaming.StreamingTableWriter(
                store, parent, TABLE, "attempt-rg-%s-%d" % (label, batch_size),
                _coverage(contract))
            for start in range(0, len(rows), batch_size):
                writer.append_rows(rows[start:start + batch_size])
            candidate = writer.finish()
            identities[label].add(
                (_fragment_hash(candidate), _dataset_version(candidate)))
            path = objects.verify_object_path(store, _published_object(candidate))
            metadata = pq.ParquetFile(path).metadata
            sizes = [metadata.row_group(i).num_rows
                     for i in range(metadata.num_row_groups)]
            expected = [group_rows] * (len(rows) // group_rows)
            if len(rows) % group_rows:
                expected.append(len(rows) % group_rows)
            assert sizes == expected
            assert pq.read_table(path).to_pylist() == [dict(row) for row in rows]
        assert len(identities[label]) == 1


def test_changeset_identity(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))
    coverage = _coverage(contract)
    rows = _rows(5)

    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-changeset", coverage)
    writer.append_rows(rows)
    candidate = writer.finish()
    changeset = candidate.changeset

    assert changeset.outcome == "changed"
    assert changeset.changed_partitions == ("__whole__",)
    assert changeset.normalized_payloads == len(rows)
    assert candidate.rewritten_partitions == 1
    assert changeset.changes == ()
    assert changeset.coverage_receipt_refs == (coverage.coverage_id,)
    assert (changeset.base_dataset_version_ref.dataset_version_id
            != changeset.result_dataset_version_ref.dataset_version_id)
    assert candidate.changeset_hash == content_hash(to_document(candidate.changeset))

    committed = generic_incremental.commit_generic_table_candidate(
        conn, store, candidate, scope="streaming",
        expected_head_snapshot_id=parent.snapshot.snapshot_id, expected_head_generation=1,
        clock=clock)
    resolved = Repository(conn).resolve_full(committed.resulting_head_snapshot_id)
    assert (resolved.table_manifests[TABLE].dataset_version_ref
            == changeset.result_dataset_version_ref)


def _date_row(i, *, asof):
    return {"ticker": "AAA", "date": (_EPOCH + timedelta(days=i)).isoformat(),
            "asof": asof}

def test_changeset_identity_varies_with_rows(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))

    seen = {}
    hashes = {}
    for count in (3, 7):
        writer = generic_streaming.StreamingTableWriter(
            store, parent, TABLE, "attempt-count-%d" % count, _coverage(contract))
        writer.append_rows(_rows(count))
        candidate = writer.finish()
        assert candidate.changeset.normalized_payloads == count
        seen[count] = candidate.changeset.changeset_id
        hashes[count] = candidate.changeset_hash

    assert seen[3] != seen[7]

    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-same-size", _coverage(contract))
    writer.append_rows([dict(r, value=r["value"] + 0.5) for r in _rows(3)])
    same_size = writer.finish()
    assert same_size.changeset.changeset_id != seen[3]
    assert same_size.changeset_hash != hashes[3]


def test_replaces_prior_all_partition(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    prior = [dict(_row(i), ticker="ZZZ") for i in range(4)]
    parent = _parent_with_prior(conn, store, clock, contract, prior, "all")
    rows = _rows(6)

    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-all", _coverage(contract))
    writer.append_rows(rows)
    candidate = writer.finish()

    assert candidate.changeset.changed_partitions == ("all",)
    assert candidate.changeset.rewritten_partitions == 1
    assert candidate.rewritten_partitions == 1

    committed = generic_incremental.commit_generic_table_candidate(
        conn, store, candidate, scope="streaming",
        expected_head_snapshot_id=parent.snapshot.snapshot_id, expected_head_generation=1,
        clock=clock)
    resolved = Repository(conn).resolve_full(committed.resulting_head_snapshot_id)
    manifest = resolved.table_manifests[TABLE]
    assert len(manifest.fragment_refs) == 1
    ids = {item.fragment_id for item in manifest.fragment_refs}
    assert [item.partition_key for item in resolved.records if item.fragment_id in ids] == ["all"]
    assert _table_rows(resolved, store, contract) == tuple(rows)


def test_replaces_prior_whole_partition(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    prior = [dict(_row(i), ticker="ZZZ") for i in range(4)]
    parent = _parent_with_prior(conn, store, clock, contract, prior, "__whole__")
    rows = _rows(6)

    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-whole", _coverage(contract))
    writer.append_rows(rows)
    candidate = writer.finish()

    assert candidate.changeset.changed_partitions == ("__whole__",)
    committed = generic_incremental.commit_generic_table_candidate(
        conn, store, candidate, scope="streaming",
        expected_head_snapshot_id=parent.snapshot.snapshot_id, expected_head_generation=1,
        clock=clock)
    resolved = Repository(conn).resolve_full(committed.resulting_head_snapshot_id)
    manifest = resolved.table_manifests[TABLE]
    assert len(manifest.fragment_refs) == 1
    ids = {item.fragment_id for item in manifest.fragment_refs}
    assert [item.partition_key for item in resolved.records if item.fragment_id in ids] == ["__whole__"]
    assert _table_rows(resolved, store, contract) == tuple(rows)


def test_coverage_for_another_table_refused(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))
    other = _make_contract("ctr_other", _WHOLE_COLUMNS,
                           primary_key=("ticker", "date"), partition_columns=())

    with pytest.raises(
            DataError,
            match=r"^CONTRACT_MISMATCH: the streaming coverage is for a different table"
    ) as exc:
        generic_streaming.StreamingTableWriter(
            store, parent, TABLE, "attempt-cov", _coverage(other))
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert _staging_entries(store, "attempt-cov") == []
    assert _head_snapshot(conn) == parent.snapshot.snapshot_id


def test_context_manager_exception_aborts(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))

    with pytest.raises(RuntimeError):
        with generic_streaming.StreamingTableWriter(
                store, parent, TABLE, "attempt-ctx-exc", _coverage(contract)) as writer:
            writer.append_rows(_rows(2))
            raise RuntimeError("boom")
    assert _staging_entries(store, "attempt-ctx-exc") == []
    assert _head_snapshot(conn) == parent.snapshot.snapshot_id
    with pytest.raises(
            DataError,
            match=r"^INPUT_CHANGED: the streaming writer is no longer open"):
        writer.append_rows([_row(5)])


def test_context_manager_clean_exit_aborts(tmp_path):
    conn, clock, store = _setup(tmp_path)
    contract = _whole_contract()
    parent = _parent(conn, store, clock, contract, _row(-1))

    with generic_streaming.StreamingTableWriter(
            store, parent, TABLE, "attempt-ctx-clean", _coverage(contract)) as writer:
        writer.append_rows(_rows(2))
    assert _staging_entries(store, "attempt-ctx-clean") == []
    assert _head_snapshot(conn) == parent.snapshot.snapshot_id
    with pytest.raises(
            DataError,
            match=r"^INPUT_CHANGED: the streaming writer is no longer open"):
        writer.append_rows([_row(5)])


def test_conversion_failure_refused(tmp_path):
    conn, clock, store = _setup(tmp_path)
    columns = (
        ColumnContract(name="ticker", physical_type="string", nullable=False),
        ColumnContract(name="date", physical_type="string", nullable=False),
        ColumnContract(name="asof", physical_type="int64", nullable=False),
    )
    contract = _make_contract("ctr_date", columns,
                              primary_key=("ticker", "date"), partition_columns=())
    parent = _parent_with_prior(
        conn, store, clock, contract,
        [_date_row(0, asof=1)], "__whole__")

    writer = generic_streaming.StreamingTableWriter(
        store, parent, TABLE, "attempt-conv", _coverage(contract))
    writer.append_rows([_date_row(1, asof=2**70)])
    with pytest.raises(
            DataError,
            match=r"^CONTRACT_MISMATCH: a staged row group does not satisfy the table contract"
    ) as exc:
        writer.finish()
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert _staging_entries(store, "attempt-conv") == []
    assert _head_snapshot(conn) == parent.snapshot.snapshot_id
