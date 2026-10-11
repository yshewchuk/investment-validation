from types import SimpleNamespace

from engine.v2.contracts.data import PriceQuery
from engine.v2.data.price_history_query import get_price_series
import pandas as pd
import pytest

from engine.v2.data.price_history import _chosen_retrieved_at


class _Batch:
    def __init__(self, rows):
        self._rows = rows

    def to_pylist(self):
        return self._rows


class _PinnedPriceHistoryRepository:
    def __init__(self, rows):
        self._rows = rows

    def scan_population_bound(self, *args, **kwargs):
        return len(self._rows)

    def scan(self, query, *, table_name):
        assert table_name == "price_history"
        yield _Batch(self._rows)


def test_price_history_uses_session_date_ceiling_not_retrieval_time():
    earlier_retrieval = "2026-10-08T20:00:00Z"
    retrieved_after_ceiling = "2026-10-09T00:15:00Z"
    rows = [
        {
            "date": "2026-10-07",
            "close_adj": 99.0,
            "close_raw": 99.0,
            "high_raw": 100.0,
            "retrieved_at": earlier_retrieval,
            "deleted": False,
            "source_kind": "tier1",
            "source_hash": "earlier-2026-10-07",
        },
        {
            "date": "2026-10-08",
            "close_adj": 100.0,
            "close_raw": 100.0,
            "high_raw": 101.0,
            "retrieved_at": earlier_retrieval,
            "deleted": False,
            "source_kind": "tier1",
            "source_hash": "earlier-2026-10-08",
        },
        {
            "date": "2026-10-08",
            "close_adj": 101.0,
            "close_raw": 101.0,
            "high_raw": 102.0,
            "retrieved_at": retrieved_after_ceiling,
            "deleted": False,
            "source_kind": "tier1",
            "source_hash": "revision-2026-10-08",
        },
        {
            "date": "2026-10-06",
            "close_adj": 98.0,
            "close_raw": 98.0,
            "high_raw": 99.0,
            "retrieved_at": retrieved_after_ceiling,
            "deleted": False,
            "source_kind": "tier1",
            "source_hash": "backfill-2026-10-06",
        },
        {
            "date": "2026-10-09",
            "close_adj": 102.0,
            "close_raw": 102.0,
            "high_raw": 103.0,
            "retrieved_at": retrieved_after_ceiling,
            "deleted": False,
            "source_kind": "tier1",
            "source_hash": "2026-10-09",
        },
    ]
    snapshot = SimpleNamespace(
        snapshot_id="pinned-snapshot",
        table_versions={"price_history": SimpleNamespace(table_contract_ref="contract")},
    )
    query = PriceQuery(
        ticker="XYZ",
        session_date="2026-10-08",
        observation_ceiling="2026-10-08T23:59:59Z",
        lookback_sessions=0,
    )

    result = get_price_series(_PinnedPriceHistoryRepository(rows), query, snapshot)

    assert [row.date for row in result] == ["2026-10-06", "2026-10-07", "2026-10-08"]
    assert result[0].close_adj == 98.0
    assert result[0].retrieved_at == retrieved_after_ceiling
    assert result[0].source_hash == "backfill-2026-10-06"
    assert result[1].close_adj == 99.0
    assert result[1].retrieved_at == earlier_retrieval
    assert result[2].close_adj == 101.0
    assert result[2].retrieved_at == retrieved_after_ceiling
    assert result[2].source_hash == "revision-2026-10-08"


def test_chosen_retrieval_requires_cutoff_for_replay():
    stored_rows = pd.DataFrame({"retrieved_at": ["2026-10-08T20:00:00Z"]})

    with pytest.raises(ValueError, match="cutoff is required"):
        _chosen_retrieved_at(stored_rows, None)


def test_chosen_retrieval_rejects_cutoff_in_latest_snapshot_mode():
    stored_rows = pd.DataFrame({"retrieved_at": ["2026-10-08T20:00:00Z"]})

    with pytest.raises(ValueError, match="cutoff must be omitted"):
        _chosen_retrieved_at(stored_rows, "2026-10-08T23:59:59Z", latest_snapshot=True)
