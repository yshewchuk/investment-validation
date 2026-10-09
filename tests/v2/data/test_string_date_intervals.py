"""The real computed_moves date representation must admit exact event queries."""
from dataclasses import replace

import pytest

from engine.v2.contracts.data import DataQuery, TimeInterval
from engine.v2.data import query
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT as CONTRACT
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_ref_for,
    publish_and_inspect,
)


def _move(ticker, day):
    return dict(ticker=ticker, event_date=day, realized_move_pct=1.0,
                available_as_of_date="2024-05-05", implied_move_pct=2.0, quarter_ordinal=1,
                skipped=False, computed_at="2024-05-05T00:00:00Z", source_hash="synthetic",
                capture_id="synthetic")


def test_string_dates_share_pruning_and_row_interval_semantics(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    ref = contract_ref_for(CONTRACT)
    records = [publish_and_inspect(store, CONTRACT, ref, [_move(ticker, day)], ticker)
               for ticker, day in [("A", "2024-05-01"), ("B", "2024-05-02"), ("C", "2024-05-03")]]
    snapshot = commit_tables(conn, clock, {"computed_moves": records},
                              {"computed_moves": CONTRACT}, store=store)
    repository = Repository(conn, store)
    try:
        for start, end in [("2024-05-02", "2024-05-03"),
                           ("2024-05-02T00:00:00Z", "2024-05-03T00:00:00Z"),
                           ("2024-05-01T20:00:00-04:00", "2024-05-02T20:00:00-04:00")]:
            interval = TimeInterval(column="event_date", start_inclusive=start, end_exclusive=end)
            bound = repository.scan_population_bound(snapshot.snapshot_id, table_name="computed_moves",
                table_contract_ref=ref, time_interval=interval)
            assert bound == 1
            request = DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=ref,
                columns=("ticker", "event_date"), key_filter=(), time_interval=interval,
                order_by=CONTRACT.primary_key, max_batch_rows=1, max_result_rows=bound)
            assert [row for batch in repository.scan(request, table_name="computed_moves")
                    for row in batch.to_pylist()] == [{"ticker": "B", "event_date": "2024-05-02"}]
            assert not query.row_matches({"ticker": "C", "event_date": "2024-05-03"}, CONTRACT, request)
            with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
                query.row_matches({"ticker": "B", "event_date": "invalid"}, CONTRACT, request)
            with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
                query.fragment_may_match(replace(records[1], time_min="invalid"), CONTRACT, request)
    finally:
        conn.close()
