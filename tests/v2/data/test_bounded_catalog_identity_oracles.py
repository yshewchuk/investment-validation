"""Independent native oracle and diagnostic evidence for B2 admission."""
from dataclasses import replace

import pytest

from engine.v2.data import manifests, objects
from engine.v2.data.errors import DataError
from engine.v2.foundation import content_hash
from tests.v2.data.bounded_catalog_identity_support import (
    contract,
    manifest,
    record,
    valid_catalog,
)


def test_native_valid_fixture_and_b1b_transfer_agree_without_objects(monkeypatch):
    def forbidden(*_, **__):
        pytest.fail("synthetic metadata fixture opened an object")
    for name in ("read_verified_bytes", "verify_object_path", "inspect_fragment"):
        monkeypatch.setattr(objects, name, forbidden)
    conn, expected, oracle, payload, by_table = valid_catalog()
    try:
        assert oracle.snapshot == expected
        assert set(oracle.table_manifests) == {"alpha", "empty", "zeta"}
        assert len(oracle.contracts) == 3 and len(oracle.objects) == 2 and len(oracle.records) == 4
        assert [row["t.table_name"] for row in payload.tables] == ["alpha", "empty", "zeta"]
        assert [row["m.ordinal"] for row in payload.fragments] == [0, 1, 0, 1]
        assert (payload.probe.tables, payload.probe.catalog_fragments) == (3, 4)
        for name, expected_records in by_table.items():
            actual = oracle.table_manifests[name]
            assert actual.fragment_refs == tuple(manifests.fragment_ref(r) for r in expected_records)
            assert actual.row_count == sum(r.row_count for r in expected_records)
        assert oracle.table_manifests["empty"].fragment_refs == ()
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


@pytest.mark.parametrize("case,code,category,message", [
    ("overlap", "MANIFEST_CORRUPT", "integrity",
     "fragment key ranges overlap or are out of order within partition 'synthetic-sensitive-partition'"),
    ("order", "MANIFEST_CORRUPT", "integrity",
     "fragment membership is out of partition_key order at 'synthetic-sensitive-partition'"),
    ("missing_hash", "MANIFEST_CORRUPT", "integrity",
     "partition 'synthetic-sensitive-partition' has 2 fragments but no partition_logical_hashes entry"),
    ("extra_hash", "MANIFEST_CORRUPT", "integrity",
     "partition_logical_hashes has key(s) not in fragment membership: ['synthetic-sensitive-partition']"),
    ("observed", "CONTRACT_MISMATCH", "validation",
     "observed dataset version requires non-empty availability_evidence_refs"),
])
def test_reachable_native_diagnostics_are_not_fixed_redacted(case, code, category, message):
    definition = contract("alpha")
    first = record(definition, "synthetic-sensitive-partition")
    changes = {}
    if case == "overlap":
        records = (first, first)
    elif case == "order":
        records = (replace(first, partition_key="z"), first)
    elif case == "missing_hash":
        records = (first, record(definition, first.partition_key, first="2021-01-01", last="2021-01-02"))
    elif case == "extra_hash":
        records = ()
        changes = {"partition_logical_hashes": {first.partition_key: content_hash({"synthetic": 1})}}
    else:
        records = (first,)
        changes = {"knowledge_mode": "observed", "availability_evidence_refs": ()}
    with pytest.raises(DataError) as caught:
        manifest(definition, records, **changes)
    problem = caught.value.problem
    assert (problem.code, problem.category, problem.message) == (code, category, message)
    assert problem.retryable is False and problem.stage is None and problem.details == {}
