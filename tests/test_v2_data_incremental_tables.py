from pathlib import Path

import pyarrow.parquet as pq
import pytest

from engine.v2.contracts import RevisionCandidate, TableContract
from engine.v2.data.errors import DataError
from engine.v2.data.incremental_tables import (
    GenericRevision,
    logical_key_for_row,
    merge_table_rows,
    revision_hash,
)
from engine.v2.data.legacy_mapping import build_legacy_mapping
from engine.v2.foundation import from_document
from tests.test_v2_data_objects import _contract, _daily_market_rows


def _revision(contract, row, revision_id, *, ordinal=1, deleted=False, partition="2024"):
    payload = None if deleted else row
    key = logical_key_for_row(contract, row)
    candidate = RevisionCandidate(
        revision_id=revision_id, logical_key=key, source="fixture",
        source_priority=0, finality="final", revision_ordinal=ordinal,
        received_at="2026-09-17T00:00:00Z",
        content_hash=revision_hash(logical_key=key, row=payload, deleted=deleted))
    return GenericRevision(candidate=candidate, row=payload, deleted=deleted,
                           partition_key=partition)


def test_generic_merge_handles_append_correction_tombstone_and_retry():
    contract = _contract("daily_market")
    prior = _daily_market_rows()
    append = dict(prior[0], ticker="ZZZ", date=prior[0]["date"], year=2024)
    correction = dict(prior[0], spot=prior[0]["spot"] + 1)
    tombstone = _revision(contract, prior[1], "delete", deleted=True)
    merged = merge_table_rows(contract, prior, (), (
        _revision(contract, append, "append"),
        _revision(contract, correction, "correct", ordinal=2),
        tombstone,
    ))
    assert [item.revision_kind for item in merged.changes] == ["correction", "tombstone", "append"]
    assert merged.changed_partitions == ("2024",)
    replay = merge_table_rows(contract, merged.rows, merged.winners, (tombstone,))
    assert replay.changes == ()
    assert replay.changed_partitions == ()


def test_generic_merge_refuses_equal_rank_conflict():
    contract = _contract("daily_market")
    row = _daily_market_rows()[0]
    left = _revision(contract, dict(row, spot=1), "left")
    right = _revision(contract, dict(row, spot=2), "right")
    with pytest.raises(DataError) as exc:
        merge_table_rows(contract, (row,), (), (left, right))
    assert exc.value.code == "IDENTITY_CONFLICT"


def test_frozen_curated_tables_use_the_same_merge_rules():
    mapping = build_legacy_mapping()
    sources = {
        name: sorted(Path("data/curated").glob(name + "/year=*/part-*.parquet"))
        for name in ("securities", "earnings_events", "daily_market", "option_chains",
                     "option_daily", "trades")}
    sources["feature_panel"] = [Path("data/features/panel.parquet")]
    sources["tier4_forecasts"] = [Path("data/features/tier4_forecasts.parquet")]
    for table_name, paths in sources.items():
        assert paths and paths[0].is_file(), table_name
        contract = from_document(TableContract, mapping["tables"][table_name])
        rows = tuple(pq.read_table(paths[0]).slice(0, 2).to_pylist())
        assert rows, table_name
        base = tuple(_revision(contract, row, "base-" + str(index))
                     for index, row in enumerate(rows))
        corrected = dict(rows[0])
        mutable = next(column.name for column in contract.columns
                        if column.name not in contract.primary_key
                        and column.name not in contract.partition_columns)
        corrected[mutable] = _bump(corrected[mutable])
        incoming = [_revision(contract, corrected, "correction", ordinal=2)]
        if len(rows) > 1:
            partition = (str(rows[1][contract.partition_columns[0]])
                         if contract.partition_columns else "__whole__")
            incoming.append(_revision(
                contract, rows[1], "tombstone", ordinal=2, deleted=True,
                partition=partition))
        incoming = tuple(incoming)
        incremental = merge_table_rows(contract, rows, base, incoming)
        clean = merge_table_rows(contract, (), (), (*base, *incoming))
        replay = merge_table_rows(contract, incremental.rows, incremental.winners, ())
        assert incremental.rows == clean.rows, table_name
        assert replay.changes == (), table_name


def _bump(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if value is None:
        return "fixture"
    return str(value) + "-corrected"
