"""Metadata-only planning for a whole-table copy beyond the public scan cap."""
from dataclasses import replace

import pytest

from engine.v2.contracts import KeyPredicate, TimeInterval
from engine.v2.data import legacy_materialization as lm
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import content_hash, to_document
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    hand_built_record,
    table_from_rows,
)
from tests.test_v2_data_legacy_materialization import CHAIN_ROWS, DM_ROWS

ROW_COUNT = 9_123_661


def oversized_case(tmp_path, table_name="daily_market"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    conn, clock, store = catalog_and_store(tmp_path)
    contract = contract_for(table_name)
    rows = DM_ROWS[("2020", "AAA")] if table_name == "daily_market" else CHAIN_ROWS["2020"]
    table = table_from_rows(contract, rows)
    # Only manifest metadata is large: planning must never open these bytes.
    record = hand_built_record(
        store, contract, contract_ref_for(contract), table, partition_key="2020",
        row_count=ROW_COUNT, primary_key_min=("AAA",), primary_key_max=("AAA",),
        time_min="2020-01-01", time_max="2020-12-31")
    snapshot = commit_tables(conn, clock, {table_name: [record]}, {table_name: contract})
    repo = Repository(conn, store)
    query = lm._build_table_query(repo, snapshot, table_name, {"tickers": ["AAA"], "years": [2020]})
    return conn, clock, store, repo, snapshot, query


def test_whole_copy_keeps_exact_query_provenance_and_public_cap(tmp_path):
    conn, _, _, repo, snapshot, query = oversized_case(tmp_path)
    assert query.max_result_rows == ROW_COUNT
    plan = lm.explain_materialization_dependencies(repo, snapshot, "daily_market", query)
    assert plan.request_hash == content_hash(to_document(query))
    assert plan.snapshot_ref == snapshot
    assert sum(item.estimated_rows for item in plan.dependencies) == ROW_COUNT
    assert all(item.maximum_rows == ROW_COUNT and item.columns == query.columns
               and item.predicates == query.key_filter for item in plan.dependencies)
    with pytest.raises(DataError, match="QUERY_NOT_BOUNDED"):
        repo.explain_dependencies(query, table_name="daily_market")
    with pytest.raises(DataError, match="QUERY_NOT_BOUNDED"):
        next(repo.scan(query, table_name="daily_market"))
    conn.close()


@pytest.mark.parametrize("change", [
    {"time_interval": None},
    {"max_batch_rows": ROW_COUNT},
    {"key_filter": (KeyPredicate(column="ticker", operator="in", values=("AAA",)),)},
    {"columns": ("ticker",)},
    {"columns": ("not_a_column",)},
    {"order_by": ("year",)},
    {"time_interval": TimeInterval(column="date", start_inclusive="2020-01-02",
                                   end_exclusive="2021-01-01")},
    {"time_interval": TimeInterval(column="date", start_inclusive=None, end_exclusive=None)},
])
def test_copy_exception_cannot_hide_invalid_or_scoped_queries(tmp_path, change):
    conn, _, _, repo, snapshot, query = oversized_case(tmp_path)
    with pytest.raises(DataError):
        lm.explain_materialization_dependencies(repo, snapshot, "daily_market", replace(query, **change))
    conn.close()


def test_unsupported_query_and_snapshot_binding_are_refused(tmp_path):
    conn, _, _, repo, snapshot, query = oversized_case(tmp_path)
    with pytest.raises(DataError, match="UNSUPPORTED_CONTRACT"):
        lm.explain_materialization_dependencies(repo, snapshot, "securities", query)
    with pytest.raises(DataError, match="UNSUPPORTED_CONTRACT"):
        lm.explain_materialization_dependencies(repo, snapshot, "daily_market", object())
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        lm.explain_materialization_dependencies(repo, snapshot, "daily_market", replace(query, snapshot_id="other"))
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        lm.explain_materialization_dependencies(repo, snapshot, "daily_market", replace(
            query, table_contract_ref=contract_ref_for(contract_for("option_chains"))))
    conn.close()


def test_evidence_scoped_materialization_retains_scan_cap(tmp_path):
    conn, _, _, repo, snapshot, query = oversized_case(tmp_path, "option_chains")
    with pytest.raises(DataError, match="QUERY_NOT_BOUNDED"):
        lm.explain_materialization_dependencies(repo, snapshot, "option_chains",
                                               replace(query, max_result_rows=ROW_COUNT))
    conn.close()
