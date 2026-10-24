"""``_fetch_rows`` sizes the scan's result limit to EXACTLY the pinned
membership bound -- there is no fixed caller cap lowering it anymore."""
from __future__ import annotations

from types import SimpleNamespace

from engine.v2.contracts import TableContractRef
from engine.v2.data.price_history_query import _COLUMNS, _fetch_rows
from engine.v2.data.price_history_table import PRICE_HISTORY_TABLE_NAME

_BOUND = 250_000


class _Batch:
    def __init__(self, rows):
        self._rows = rows

    def to_pylist(self):
        return self._rows


class _FakeRepository:
    def scan_population_bound(self, snapshot_id, *, table_name, table_contract_ref, key_filter):
        assert snapshot_id == "s"
        assert table_name == PRICE_HISTORY_TABLE_NAME
        return _BOUND

    def scan(self, query, *, table_name):
        assert table_name == PRICE_HISTORY_TABLE_NAME
        assert query.max_result_rows == _BOUND
        row = {column: (False if column == "deleted" else f"v-{column}") for column in _COLUMNS}
        return [_Batch([row] * (_BOUND // 2)), _Batch([row] * (_BOUND - _BOUND // 2))]


def test_result_limit_is_the_exact_pinned_membership_bound():
    contract_ref = TableContractRef(contract_id="price_history@1",
                                    definition_hash="sha256:" + "0" * 64)
    snapshot_ref = SimpleNamespace(
        snapshot_id="s",
        table_versions={PRICE_HISTORY_TABLE_NAME: SimpleNamespace(
            table_contract_ref=contract_ref, dataset_version_id="dv-1")})
    rows = _fetch_rows(_FakeRepository(), snapshot_ref, "MRK")
    assert len(rows) == _BOUND
