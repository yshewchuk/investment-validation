"""``_fetch_rows`` sizes the scan's result limit to EXACTLY the pinned
membership bound -- there is no fixed caller cap lowering it anymore."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.v2.contracts import KeyPredicate, TableContractRef
from engine.v2.data.price_history_query import _COLUMNS, _fetch_rows
from engine.v2.data.price_history_table import PRICE_HISTORY_TABLE_NAME

_TICKER = "MRK"


class _Batch:
    def __init__(self, rows):
        self._rows = rows

    def to_pylist(self):
        return self._rows


class _FakeRepository:
    def __init__(self, bound, ticker):
        self._bound = bound
        self._key_filter = (KeyPredicate(column="ticker", operator="eq", values=(ticker,)),)

    def scan_population_bound(self, snapshot_id, *, table_name, table_contract_ref, key_filter):
        assert snapshot_id == "s"
        assert table_name == PRICE_HISTORY_TABLE_NAME
        assert key_filter == self._key_filter
        return self._bound

    def scan(self, query, *, table_name):
        assert table_name == PRICE_HISTORY_TABLE_NAME
        assert query.key_filter == self._key_filter
        assert query.max_result_rows == self._bound
        expected_max_batch_rows = min(50_000, self._bound) if self._bound > 0 else 50_000
        assert query.max_batch_rows == expected_max_batch_rows
        if self._bound == 0:
            return []
        row = {column: (False if column == "deleted" else f"v-{column}") for column in _COLUMNS}
        half = self._bound // 2
        return [_Batch([row] * half), _Batch([row] * (self._bound - half))]


@pytest.mark.parametrize("bound", (0, 7, 250_000))
def test_result_limit_is_the_exact_pinned_membership_bound(bound):
    contract_ref = TableContractRef(contract_id="price_history@1",
                                    definition_hash="sha256:" + "0" * 64)
    snapshot_ref = SimpleNamespace(
        snapshot_id="s",
        table_versions={PRICE_HISTORY_TABLE_NAME: SimpleNamespace(
            table_contract_ref=contract_ref, dataset_version_id="dv-1")})
    rows = _fetch_rows(_FakeRepository(bound, _TICKER), snapshot_ref, _TICKER)
    assert len(rows) == bound
