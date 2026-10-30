"""Standalone, synthetic two-target publication with native protected readers."""
from dataclasses import replace
from datetime import timedelta

import pytest

from engine.v2.contracts.data import ColumnContract, DataQuery, KeyPredicate
from engine.v2.data import manifests, objects
from engine.v2.data.errors import DataError
from engine.v2.data.iv_crush import build_rows
from engine.v2.data.iv_crush_table import IV_CRUSH_CONTRACT
from engine.v2.data.repository import Repository
from engine.v2.research.event_target_store import publish_event_targets
from engine.v2.research.iv_crush_inputs import load_iv_crush_targets
from tests.data_scan_support import catalog_and_store, contract_for
from tests.v2.research.magnitude_trial_support import history_event

EVENTS = contract_for("earnings_events")
SOURCE = manifests.snapshot_ref({}, calendar_version="synthetic-calendar",
    source_priority_version="synthetic-priority", finality_receipt_refs=())


def events():
    return [{c.name: row.get(c.name) for c in EVENTS.columns}
            for row in (history_event("2024-01-04"), history_event("2024-02-05"))]


def targets():
    return build_rows("SYNTH", [{"event_date": r["event_date"].date().isoformat(), "session": r["session"]}
        for r in events()], [{"date": day, "iv30": value} for day, value in (
            ("2024-01-04", 50), ("2024-01-05", 30), ("2024-02-05", 40), ("2024-02-06", 60))],
        source_id="synthetic-input")


def publish(repository, **overrides):
    return publish_event_targets(repository, **{**dict(source_snapshot=SOURCE,
        provenance={"producer": "synthetic-test", "source_publication": "unverified"},
        events_contract=EVENTS, events=events(), target_contract=IV_CRUSH_CONTRACT, targets=targets(),
        max_rows=20, max_object_bytes=100000, max_total_bytes=1000000), **overrides})


def scan(repository, snapshot, table, contract):
    column = contract.partition_columns[0]
    value = 2024 if column == "year" else "SYNTH"
    count = sum(r.row_count for r in repository.fragment_records(snapshot, table)
                if r.partition_key == str(value))
    return [row for batch in repository.scan(DataQuery(snapshot_id=snapshot.snapshot_id,
        table_contract_ref=snapshot.table_versions[table].table_contract_ref,
        columns=tuple(c.name for c in contract.columns), key_filter=(KeyPredicate(column=column,
            operator="eq", values=(value,)),), order_by=contract.primary_key,
        max_batch_rows=count, max_result_rows=count), table_name=table) for row in batch.to_pylist()]


def test_iv_roundtrip_isolated_objects_replay_and_no_source_reads(tmp_path, monkeypatch):
    conn, _, store = catalog_and_store(tmp_path)
    repository = Repository(conn, store)
    # No source repository is supplied: this source identity need not be committed locally.
    assert conn.execute("SELECT COUNT(*) FROM data_snapshots").fetchone()[0] == 0
    snapshot = publish(repository)
    assert set(snapshot.table_versions) == {"earnings_events", "iv_crush_targets"}
    assert snapshot.knowledge_mode_by_table == {name: "reconstructed" for name in snapshot.table_versions}
    records = repository.fragment_records(snapshot, "iv_crush_targets")
    assert len(records) == 2 and all(r.row_count == 1 and r.time_min == r.time_max for r in records)
    metadata = repository.fragment_records(snapshot, "earnings_events")
    assert len(metadata) == 1 and metadata[0].row_count == 2
    frame = load_iv_crush_targets(repository, snapshot, as_of_month="2025-01", labels_before="2024-07-01")
    assert frame.crush_pct_iv30.tolist() == [-40., 50.]
    assert scan(repository, snapshot, "earnings_events", EVENTS) == events()
    changes = conn.total_changes
    assert publish(repository, events=events()[::-1], targets=targets()[::-1]) == snapshot
    assert conn.total_changes == changes
    partition = objects.partition_logical_hash(store, [r.object_ref for r in records],
        IV_CRUSH_CONTRACT, snapshot.table_versions["iv_crush_targets"].table_contract_ref, "SYNTH")
    assert partition.startswith("sha256:")
    conn.close()


def test_metadata_batches_are_bounded_and_partition_local(tmp_path):
    conn, _, store = catalog_and_store(tmp_path)
    repository = Repository(conn, store)
    supplied = [{**events()[0], "event_id": f"synthetic-{i:04d}"} for i in range(258)]
    later = {**events()[1], "event_id": "synthetic-later", "year": 2023,
             "event_date": events()[1]["event_date"].replace(year=2023)}
    snapshot = publish(repository, events=[*events(), *supplied, later], max_rows=300)
    metadata = repository.fragment_records(snapshot, "earnings_events")
    assert sorted((r.partition_key, r.row_count) for r in metadata) == [("2023", 1), ("2024", 4), ("2024", 256)]
    assert all(r.row_count == 1 for r in repository.fragment_records(snapshot, "iv_crush_targets"))
    assert len(scan(repository, snapshot, "earnings_events", EVENTS)) == 260
    conn.close()


def test_aliases_and_conflicts_are_preserved_not_silently_selected(tmp_path):
    conn, _, store = catalog_and_store(tmp_path)
    repository = Repository(conn, store)
    rows = events()
    rows.append({**rows[0], "event_id": "ambiguous-alias", "session": "BMO", "date_conflict": True,
                 "event_cluster_id": "synthetic-cluster", "claim_count": 2})
    snapshot = publish(repository, events=rows)
    saved = scan(repository, snapshot, "earnings_events", EVENTS)
    assert sorted(saved, key=lambda r: r["event_id"]) == sorted(rows, key=lambda r: r["event_id"])
    result = load_iv_crush_targets(repository, snapshot, as_of_month="2025-01", labels_before="2024-07-01")
    assert result.event_id.tolist() == [rows[1]["event_id"]]
    assert len(result.attrs["holdout_exclusions"]) == 2
    conn.close()


def test_second_event_keyed_target_contract(tmp_path):
    contract = replace(IV_CRUSH_CONTRACT, contract_id="synthetic_outcomes.v1", table_name="synthetic_outcomes",
        columns=(ColumnContract(name="ticker", physical_type="string", nullable=False),
                 ColumnContract(name="event_date", physical_type="string", nullable=False),
                 ColumnContract(name="outcome", physical_type="float64", nullable=True)))
    contract = replace(contract, definition_hash=manifests.table_contract_hash(contract))
    rows = [{"ticker": "SYNTH", "event_date": r["event_date"], "outcome": value}
            for r, value in zip(targets(), [None, -10.])]
    conn, _, store = catalog_and_store(tmp_path)
    repository = Repository(conn, store)
    snapshot = publish(repository, target_contract=contract, targets=rows)
    assert scan(repository, snapshot, contract.table_name, contract) == rows
    assert all(r.row_count == 1 for r in repository.fragment_records(snapshot, contract.table_name))
    conn.close()


@pytest.mark.parametrize("overrides,code", [
    ({"max_rows": 1}, "RESULT_LIMIT_EXCEEDED"),
    ({"max_rows": True}, "CONTRACT_MISMATCH"),
    ({"max_object_bytes": 1}, "RESULT_LIMIT_EXCEEDED"),
    ({"max_total_bytes": 1}, "RESULT_LIMIT_EXCEEDED"),
    ({"max_object_bytes": 2000}, "RESULT_LIMIT_EXCEEDED"),
    ({"max_total_bytes": 10000}, "RESULT_LIMIT_EXCEEDED"),
    ({"events": []}, "CONTRACT_MISMATCH"),
    ({"targets": []}, "CONTRACT_MISMATCH"),
    ({"targets": [{**r, "ticker": "MISSING"} for r in targets()]}, "CONTRACT_MISMATCH"),
    ({"targets": targets() + targets()[:1]}, "CONTRACT_MISMATCH"),
    ({"events": events() + events()[:1]}, "CONTRACT_MISMATCH"),
    ({"targets": [{**r, "ticker": None} for r in targets()]}, "CONTRACT_MISMATCH"),
    ({"events": [{**r, "date_conflict": None} for r in events()]}, "CONTRACT_MISMATCH"),
    ({"events": [{**r, "event_date": r["event_date"] + timedelta(hours=1)} for r in events()]}, "CONTRACT_MISMATCH"),
    ({"events": [{**r, "event_id": []} for r in events()]}, "CONTRACT_MISMATCH"),
    ({"targets": [{k: v for k, v in r.items() if k != "session"} for r in targets()]}, "CONTRACT_MISMATCH"),
    ({"targets": [{**r, "unrecognized": 1} for r in targets()]}, "CONTRACT_MISMATCH"),
    ({"provenance": {}}, "CONTRACT_MISMATCH"),
    ({"target_contract": replace(IV_CRUSH_CONTRACT, definition_hash="sha256:" + "a" * 64)}, "CONTRACT_MISMATCH"),
])
def test_refusals_never_commit_partial_snapshot(tmp_path, overrides, code):
    conn, _, store = catalog_and_store(tmp_path)
    with pytest.raises(DataError) as caught:
        publish(Repository(conn, store), **overrides)
    assert caught.value.code == code
    assert conn.execute("SELECT COUNT(*) FROM data_snapshots").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM data_snapshot_heads").fetchone()[0] == 0
    conn.close()


def test_explicit_target_subset_keeps_all_metadata(tmp_path):
    conn, _, store = catalog_and_store(tmp_path)
    repository = Repository(conn, store)
    snapshot = publish(repository, targets=targets()[:1])
    assert len(scan(repository, snapshot, "earnings_events", EVENTS)) == 2
    frame = load_iv_crush_targets(repository, snapshot, as_of_month="2025-01", labels_before="2024-07-01",
                                 event_ids=[events()[0]["event_id"]])
    assert frame.crush_pct_iv30.tolist() == [-40.]
    conn.close()


def test_request_identity_binds_source_provenance_caps_and_explicit_output(tmp_path):
    (tmp_path / "output").mkdir()
    (tmp_path / "source").mkdir()
    conn, _, store = catalog_and_store(tmp_path / "output")
    source_conn, _, source_store = catalog_and_store(tmp_path / "source")
    repository = Repository(conn, store)
    source_before = (source_conn.total_changes, list(source_store.root.rglob("*")))
    first = publish(repository)
    publish(repository, provenance={"producer": "second-assertion"})
    publish(repository, max_rows=21)
    publish(repository, source_snapshot=replace(SOURCE, snapshot_id="snap_" + "a" * 32))
    receipts = conn.execute("SELECT source_manifest_hash FROM data_import_receipts").fetchall()
    assert len(receipts) == len({r[0] for r in receipts}) == 4
    assert source_before == (source_conn.total_changes, list(source_store.root.rglob("*")))
    assert repository.resolve(first.snapshot_id) == first
    source_conn.close()
    conn.close()


def test_selected_reader_never_opens_other_target_object(tmp_path, monkeypatch):
    conn, _, store = catalog_and_store(tmp_path)
    repository = Repository(conn, store)
    snapshot = publish(repository)
    forbidden = next(r.object_ref.object_id for r in repository.fragment_records(snapshot, "iv_crush_targets")
                     if r.time_min == "2024-02-05")
    original = objects.read_verified_bytes

    def guarded(store, ref, **kwargs):
        assert ref.object_id != forbidden
        return original(store, ref, **kwargs)

    monkeypatch.setattr(objects, "read_verified_bytes", guarded)
    result = load_iv_crush_targets(repository, snapshot, as_of_month="2025-01", labels_before="2024-07-01",
                                  event_ids=[events()[0]["event_id"]])
    assert result.crush_pct_iv30.tolist() == [-40.]
    conn.close()


@pytest.mark.parametrize("changes", [dict(provenance={"producer": "second-assertion"}),
    dict(max_rows=21), dict(source_snapshot=replace(SOURCE, snapshot_id="snap_" + "a" * 32))])
def test_reconciled_publication_replay_returns_committed_reader_reference(tmp_path, changes):
    conn, _, store = catalog_and_store(tmp_path)
    repository = Repository(conn, store)
    first = publish(repository)
    second = publish(repository, **changes)
    assert second == repository.resolve(second.snapshot_id) == first
    before = conn.total_changes
    replay = publish(repository, **changes)
    assert conn.total_changes == before
    assert replay == second == repository.resolve(replay.snapshot_id)
    frame = load_iv_crush_targets(repository, replay, as_of_month="2025-01", labels_before="2024-07-01")
    assert frame.crush_pct_iv30.tolist() == [-40., 50.]
    conn.close()


@pytest.mark.parametrize("provenance", [{1: "discarded", "1": "retained"},
    {"nested": [{1: "discarded", "1": "retained"}]}, {"nested": {2: "value"}}])
def test_nonstring_provenance_keys_refuse_before_hashing_or_publication(tmp_path, provenance):
    conn, _, store = catalog_and_store(tmp_path)
    repository = Repository(conn, store)
    publish(repository, provenance={"1": "retained"})
    before = conn.total_changes
    with pytest.raises(DataError) as caught:
        publish(repository, provenance=provenance)
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert not caught.value.problem.retryable
    assert conn.total_changes == before
    assert conn.execute("SELECT COUNT(*) FROM data_import_receipts").fetchone()[0] == 1
    conn.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1, 2}, b"bytes", object()])
def test_non_json_provenance_refuses_without_writes(tmp_path, value):
    conn, _, store = catalog_and_store(tmp_path)
    with pytest.raises(DataError) as caught:
        publish(Repository(conn, store), provenance={"nested": [value]})
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert conn.execute("SELECT COUNT(*) FROM data_snapshots").fetchone()[0] == 0
    conn.close()


def test_valid_nested_provenance_preserves_fields_and_distinguishes_requests(tmp_path):
    conn, _, store = catalog_and_store(tmp_path)
    repository = Repository(conn, store)
    provenance = {"1": "retained", "nested": [{"2": [None, True, 2, 0.25, "text"]}]}
    first = publish(repository, provenance=provenance)
    assert provenance == {"1": "retained", "nested": [{"2": [None, True, 2, 0.25, "text"]}]}
    changed = {**provenance, "additional": "must-remain-bound"}
    assert publish(repository, provenance=changed) == first
    receipts = conn.execute("SELECT source_manifest_hash FROM data_import_receipts").fetchall()
    assert len(receipts) == len({r[0] for r in receipts}) == 2
    before = conn.total_changes
    assert publish(repository, provenance=provenance) == first
    assert conn.total_changes == before
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(DataError) as caught:
        publish(repository, provenance=cyclic)
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert conn.total_changes == before
    conn.close()


@pytest.mark.parametrize("argument,contract", [("target_contract", IV_CRUSH_CONTRACT),
                                               ("events_contract", EVENTS)])
@pytest.mark.parametrize("field", ["filterable_columns", "orderable_columns"])
def test_rehashed_structurally_invalid_contract_refuses_before_any_write(tmp_path, argument, contract, field):
    conn, _, store = catalog_and_store(tmp_path)
    malformed = replace(contract, **{field: ("missing",)})
    malformed = replace(malformed, definition_hash=manifests.table_contract_hash(malformed))
    before = conn.total_changes
    files = {p.relative_to(store.root): p.read_bytes() for p in store.root.rglob("*") if p.is_file()}
    with pytest.raises(DataError) as caught:
        publish(Repository(conn, store), **{argument: malformed})
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert not caught.value.problem.retryable
    assert conn.total_changes == before
    assert {p.relative_to(store.root): p.read_bytes() for p in store.root.rglob("*") if p.is_file()} == files
    conn.close()
