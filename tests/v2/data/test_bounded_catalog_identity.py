"""Synthetic, metadata-only native identity admission oracles."""
from copy import deepcopy
from dataclasses import replace

import pytest

from engine.v2.contracts.data import FragmentRecord
from engine.v2.data import _bounded_catalog_identity as identity
from engine.v2.data import documents, errors, manifests, objects
from engine.v2.data._bounded_catalog_transfer import _Times
from engine.v2.data.repository import Repository
from engine.v2.foundation import content_hash, to_document
from tests.v2.data.bounded_catalog_identity_support import manifest, record, valid_catalog


@pytest.fixture
def fixture():
    conn, expected, oracle, payload, records = valid_catalog()
    conn.close()  # B2 must work with detached metadata; SQL is no longer available.
    return expected, oracle, payload, records


def test_reconstruct_exact_native_identity_without_io(fixture, monkeypatch, capsys, caplog):
    expected, oracle, payload, by_table = fixture
    def forbidden(*_, **__):
        pytest.fail("identity reconstruction attempted I/O")
    for name in ("read_verified_bytes", "verify_object_path", "inspect_fragment"):
        monkeypatch.setattr(objects, name, forbidden)
    for name in ("resolve", "resolve_full", "_manifest", "_records"):
        monkeypatch.setattr(Repository, name, forbidden)
    before = deepcopy(payload)
    actual = identity.reconstruct_catalog(payload, expected)
    assert actual.snapshot == oracle.snapshot == expected
    assert actual.table_manifests == oracle.table_manifests
    assert actual.contracts == tuple(sorted(oracle.contracts, key=lambda x: x.table_name))
    assert actual.records == tuple(r for name in sorted(by_table) for r in by_table[name])
    assert {o.object_id: o for o in actual.objects} == {o.object_id: o for o in oracle.objects}
    assert len(actual.objects) == 2 and len(actual.records) == 4
    assert actual.table_manifests["empty"].fragment_refs == ()
    assert payload == before
    assert capsys.readouterr() == ("", "") and not caplog.records


@pytest.mark.parametrize("section,key,value", [
    ("snapshot", "s.snapshot_id", "snap_wrong"),
    ("snapshot", "s.manifest_hash", content_hash({"wrong": 1})),
    ("snapshot", "s.parent_snapshot_id", "snap_wrong"),
    ("snapshot", "s.calendar_version", "wrong"),
    ("snapshot", "s.source_priority_version", "wrong"),
    ("snapshot", "s.finality_receipt_refs_json", ("wrong",)),
    ("snapshot", "s.knowledge_mode_by_table_json", {}),
    ("tables", "t.snapshot_id", "snap_wrong"),
    ("tables", "t.table_name", "wrong"),
    ("tables", "t.dataset_version_id", "dsv_wrong"),
    ("tables", "d.dataset_version_id", "dsv_wrong"),
    ("tables", "d.contract_id", "wrong"),
    ("tables", "d.parent_dataset_version_id", "dsv_wrong"),
    ("tables", "d.manifest_hash", content_hash({"wrong": 1})),
    ("tables", "d.logical_content_hash", content_hash({"wrong": 1})),
    ("tables", "d.row_count", 9),
    ("tables", "d.knowledge_mode", "observed"),
    ("tables", "c.contract_id", "wrong"),
    ("tables", "c.schema_version", "table_contract.v99.0"),
    ("tables", "c.table_name", "wrong"),
    ("tables", "c.definition_hash", content_hash({"wrong": 1})),
    ("fragments", "m.dataset_version_id", "dsv_wrong"),
    ("fragments", "m.ordinal", 1),
    ("fragments", "m.fragment_id", "frag_wrong"),
    ("fragments", "f.fragment_id", "frag_wrong"),
    ("fragments", "f.object_id", "wrong"),
    ("fragments", "f.contract_id", "wrong"),
    ("fragments", "f.partition_key", "wrong"),
    ("fragments", "f.row_count", 3),
    ("fragments", "f.byte_hash", content_hash({"wrong": 1})),
    ("fragments", "f.logical_content_hash", content_hash({"wrong": 1})),
    ("fragments", "f.import_request_hash", content_hash({"wrong": 1})),
    ("fragments", "f.input_receipt_refs_json", ("wrong",)),
    ("fragments", "o.object_id", "wrong"),
    ("fragments", "o.kind", "wrong"),
    ("fragments", "o.content_hash", content_hash({"wrong": 1})),
    ("fragments", "o.byte_size", 129),
])
def test_each_stored_identity_and_binding_is_checked(fixture, section, key, value):
    expected, _, payload, _ = fixture
    payload = deepcopy(payload)
    target = payload.snapshot if section == "snapshot" else getattr(payload, section)[0]
    target[key] = value
    with pytest.raises(errors.DataError) as caught:
        identity.reconstruct_catalog(payload, expected)
    assert caught.value.code == "MANIFEST_CORRUPT"


@pytest.mark.parametrize("mutation", ["missing_table", "duplicate_table", "table_order", "missing_fragment",
    "duplicate_fragment", "fragment_order", "bad_contract", "bad_time", "bad_hash"])
def test_malformed_or_incomplete_metadata_has_no_result(fixture, mutation):
    expected, _, payload, _ = fixture
    payload = deepcopy(payload)
    if mutation == "missing_table":
        payload = replace(payload, tables=payload.tables[1:])
    elif mutation == "duplicate_table":
        payload = replace(payload, tables=payload.tables + (payload.tables[0],))
    elif mutation == "table_order":
        payload = replace(payload, tables=tuple(reversed(payload.tables)))
    elif mutation == "missing_fragment":
        payload = replace(payload, fragments=payload.fragments[1:])
    elif mutation == "duplicate_fragment":
        payload = replace(payload, fragments=payload.fragments + (payload.fragments[0],))
    elif mutation == "fragment_order":
        payload = replace(payload, fragments=tuple(reversed(payload.fragments)))
    elif mutation == "bad_contract":
        definition = payload.tables[0]["c.definition_json"]
        payload.tables[0]["c.definition_json"] = replace(definition, primary_key=("absent",))
    elif mutation == "bad_time":
        payload.fragments[0]["f.time_bounds_json"] = _Times("2021-01-02", "2020-01-02")
    else:
        payload.tables[0]["d.partition_logical_hashes_json"] = {"2020": "not-a-hash"}
    with pytest.raises(errors.DataError) as caught:
        identity.reconstruct_catalog(payload, expected)
    assert caught.value.code == "MANIFEST_CORRUPT"


@pytest.mark.parametrize("mutation", ["shape", "hash", "id", "tables", "modes", "parent", "calendar", "priority", "evidence"])
def test_full_expected_snapshot_is_required(fixture, mutation):
    expected, _, payload, _ = fixture
    changes = {"shape": None, "hash": {"manifest_hash": "invalid"},
        "id": {"snapshot_id": "different"}, "tables": {"table_versions": {}},
        "modes": {"knowledge_mode_by_table": {}}, "parent": {"parent_snapshot_id": "different"},
        "calendar": {"calendar_version": "different"}, "priority": {"source_priority_version": "different"},
        "evidence": {"finality_receipt_refs": ("different",)}}[mutation]
    expected = None if changes is None else replace(expected, **changes)
    with pytest.raises(errors.DataError) as caught:
        identity.reconstruct_catalog(payload, expected)
    assert caught.value.code == "SNAPSHOT_UNRESOLVED"
    assert caught.value.problem.details == {}


@pytest.mark.parametrize("builder", ["table_contract_hash", "fragment_record", "dataset_manifest", "snapshot_ref"])
def test_native_errors_are_the_same_exception_and_complete_envelope(
        fixture, monkeypatch, builder, capsys, caplog):
    expected, _, payload, _ = fixture
    original = errors.fail("CONTRACT_MISMATCH", "synthetic native diagnostic", stage="synthetic-stage",
                           details={"synthetic-field": [1, "native detail"]})
    def fail(*_, **__):
        raise original
    monkeypatch.setattr(manifests, builder, fail)
    with pytest.raises(errors.DataError) as caught:
        identity.reconstruct_catalog(payload, expected)
    assert caught.value is original
    assert caught.value.problem == original.problem
    assert capsys.readouterr() == ("", "") and not caplog.records


@pytest.mark.parametrize("case", ["overlap", "order", "missing_hash", "extra_hash", "observed"])
def test_reachable_native_diagnostics_cross_boundary_unchanged(fixture, case):
    expected, _, payload, by_table = fixture
    payload = deepcopy(payload)
    table = payload.tables[0]
    definition = table["c.definition_json"]
    records = by_table["alpha"]
    marker = "synthetic-partition-marker"
    first = record(definition, marker)
    changes = {}
    if case == "overlap":
        records = (first, first)
    elif case == "order":
        records = (record(definition, "z"), first)
    elif case == "missing_hash":
        records = (first, record(definition, marker, first="2021-01-01", last="2021-01-02"))
    elif case == "extra_hash":
        changes = {"partition_logical_hashes": {marker: content_hash({"synthetic": 1})}}
        table["d.partition_logical_hashes_json"] = changes["partition_logical_hashes"]
    else:
        changes = {"knowledge_mode": "observed", "availability_evidence_refs": ()}
        table["d.knowledge_mode"] = "observed"
        table["d.evidence_json"] = replace(table["d.evidence_json"], availability_evidence_refs=())
    if case in {"overlap", "order", "missing_hash"}:
        for row, rec in zip(payload.fragments[:2], records, strict=True):
            row.update({"m.fragment_id": rec.fragment_id, "f.fragment_id": rec.fragment_id,
                "f.partition_key": rec.partition_key, "f.logical_content_hash": rec.logical_content_hash,
                "f.key_bounds_json": replace(row["f.key_bounds_json"],
                    primary_key_min=rec.primary_key_min, primary_key_max=rec.primary_key_max)})
    with pytest.raises(errors.DataError) as native:
        manifest(definition, records, **changes)
    with pytest.raises(errors.DataError) as actual:
        identity.reconstruct_catalog(payload, expected)
    assert actual.value.problem == native.value.problem


def test_valid_but_nonexact_nested_expected_version_is_unresolved(fixture):
    expected, _, payload, _ = fixture
    versions = dict(expected.table_versions)
    versions["alpha"] = replace(versions["alpha"], manifest_hash=content_hash({"different": 1}))
    expected = replace(expected, table_versions=versions)
    with pytest.raises(errors.DataError) as caught:
        identity.reconstruct_catalog(payload, expected)
    assert caught.value.code == "SNAPSHOT_UNRESOLVED"


def test_new_decode_failure_is_fixed_and_suppresses_raw_context(fixture):
    expected, _, payload, _ = fixture
    payload = deepcopy(payload)
    payload.fragments[0]["f.time_bounds_json"] = _Times("synthetic-invalid-date", None)
    with pytest.raises(errors.DataError) as caught:
        identity.reconstruct_catalog(payload, expected)
    problem = caught.value.problem
    assert (problem.code, problem.category, problem.retryable, problem.stage, problem.details) == (
        "MANIFEST_CORRUPT", "integrity", False, None, {})
    assert problem.message == "catalog identity malformed"
    assert caught.value.__suppress_context__ is True


@pytest.mark.parametrize("name", ["zeta", "empty"])
@pytest.mark.parametrize("field,value", [("c.definition_hash", content_hash({"wrong": 1})),
    ("d.row_count", 99), ("d.knowledge_mode", "observed")])
def test_unselected_and_empty_tables_are_authenticated(fixture, name, field, value):
    expected, _, payload, _ = fixture
    payload = deepcopy(payload)
    next(row for row in payload.tables if row["t.table_name"] == name)[field] = value
    with pytest.raises(errors.DataError) as caught:
        identity.reconstruct_catalog(payload, expected)
    assert caught.value.code == "MANIFEST_CORRUPT"


def test_results_are_detached_from_payload_and_each_other(fixture):
    expected, _, payload, _ = fixture
    before = deepcopy(payload)
    first = identity.reconstruct_catalog(payload, expected)
    second = identity.reconstruct_catalog(payload, expected)
    first.snapshot.knowledge_mode_by_table.clear()
    first.snapshot.table_versions.clear()
    first.table_manifests["alpha"].partition_logical_hashes["synthetic"] = "mutation"
    first.table_manifests.clear()
    assert payload == before
    assert second.snapshot == expected
    assert second.table_manifests["alpha"].partition_logical_hashes == {}
    assert identity.reconstruct_catalog(payload, expected) == second


@pytest.mark.parametrize("kind", ["receipt_list", "dataset_dict", "contract_dict",
                                  "nested_subclass", "coerced_key"])
def test_expected_reference_must_not_change_during_native_roundtrip(fixture, kind):
    from engine.v2.data import documents
    from engine.v2.foundation import to_document
    from tests.v2.data.bounded_catalog_identity_support import normalized_snapshot_impostor
    expected, _, payload, _ = fixture
    supplied = normalized_snapshot_impostor(expected, kind)
    assert supplied != expected
    assert documents.decode_document(type(expected), to_document(supplied)) == expected
    with pytest.raises(errors.DataError) as caught:
        identity.reconstruct_catalog(payload, supplied)
    problem = caught.value.problem
    assert (problem.code, problem.category, problem.retryable, problem.stage,
            problem.message, problem.details) == (
        "SNAPSHOT_UNRESOLVED", "dependency", False, None, "invalid expected snapshot", {})


def test_fragment_record_with_float_key_bound_decodes():
    from tests.v2.data.bounded_catalog_identity_support import contract, record
    from dataclasses import replace
    rec = record(contract("alpha"), "2020")
    rec = replace(rec, primary_key_min=("FIX", 22.5), primary_key_max=("FIX", 23.0))
    assert documents.decode_document(FragmentRecord, to_document(rec)) == rec


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_fragment_record_nonfinite_key_bound_refuses(bad):
    from tests.v2.data.bounded_catalog_identity_support import contract, record
    from dataclasses import replace
    rec = replace(record(contract("alpha"), "2020"), primary_key_min=("FIX", bad))
    with pytest.raises(documents.DocumentError):
        documents.decode_document(FragmentRecord, to_document(rec))
