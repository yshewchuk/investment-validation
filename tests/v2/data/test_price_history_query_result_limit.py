"""``_fetch_rows`` sizes the scan's result limit to EXACTLY the pinned
membership bound -- there is no fixed caller cap lowering it anymore."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.v2.contracts import KeyPredicate, TableContractRef
from engine.v2.data.price_history_query import _fetch_rows
from engine.v2.data.price_history_table import PRICE_HISTORY_TABLE_NAME

_TICKER = "MRK"

_EXPECTED_COLUMNS = ("date", "close_adj", "close_raw", "high_raw", "retrieved_at",
                     "deleted", "source_kind", "source_hash")


def _fake_row(index):
    marker = index % 3
    return {column: (False if column == "deleted" else f"v{marker}-{column}")
            for column in _EXPECTED_COLUMNS}


class _Batch:
    def __init__(self, rows):
        self._rows = rows

    def to_pylist(self):
        return self._rows


class _FakeRepository:
    def __init__(self, bound, ticker):
        self._bound = bound
        self._key_filter = (KeyPredicate(column="ticker", operator="eq", values=(ticker,)),)
        self.byte_caps: list = []

    def scan_population_bound(self, snapshot_id, *, table_name, table_contract_ref, key_filter):
        assert snapshot_id == "s"
        assert table_name == PRICE_HISTORY_TABLE_NAME
        assert key_filter == self._key_filter
        return self._bound

    def scan(self, query, *, table_name, max_object_bytes=None):
        assert table_name == PRICE_HISTORY_TABLE_NAME
        assert query.key_filter == self._key_filter
        assert query.columns == _EXPECTED_COLUMNS
        assert query.max_result_rows == self._bound
        expected_max_batch_rows = min(50_000, self._bound) if self._bound > 0 else 50_000
        assert query.max_batch_rows == expected_max_batch_rows
        self.byte_caps.append(max_object_bytes)
        if self._bound == 0:
            return []
        rows = [_fake_row(i) for i in range(self._bound)]
        half = self._bound // 2
        return [_Batch(rows[:half]), _Batch(rows[half:])]


def _snapshot_ref():
    contract_ref = TableContractRef(contract_id="price_history@1",
                                    definition_hash="sha256:" + "0" * 64)
    return SimpleNamespace(
        snapshot_id="s",
        table_versions={PRICE_HISTORY_TABLE_NAME: SimpleNamespace(
            table_contract_ref=contract_ref, dataset_version_id="dv-1")})


@pytest.mark.parametrize("bound", (0, 7, 250_000))
def test_result_limit_is_the_exact_pinned_membership_bound(bound):
    rows = _fetch_rows(_FakeRepository(bound, _TICKER), _snapshot_ref(), _TICKER)
    assert len(rows) == bound
    assert list(rows.columns) == list(_EXPECTED_COLUMNS)
    if bound > 0:
        assert rows.iloc[0].to_dict() == _fake_row(0)
        assert rows.iloc[1].to_dict() == _fake_row(1)
        assert rows.iloc[-1].to_dict() == _fake_row(bound - 1)


def test_max_object_bytes_is_forwarded_to_the_scan_and_none_means_no_cap():
    capped = _FakeRepository(3, _TICKER)
    _fetch_rows(capped, _snapshot_ref(), _TICKER, max_object_bytes=4096)
    assert capped.byte_caps == [4096]
    uncapped = _FakeRepository(3, _TICKER)
    _fetch_rows(uncapped, _snapshot_ref(), _TICKER)
    assert uncapped.byte_caps == [None]
