"""Stored qualified panel read streams one fragment at a time and equals the all-rows read."""
# ruff: noqa: F811
import pytest

from engine.v2.data.errors import DataError
from engine.v2.research import _magnitude_panel_store as store
from engine.v2.research import magnitude_panel_qualified_table as table
from experiments import size_panel
from tests.v2.research.test_magnitude_panel_qualified import panel_data, read  # noqa: F401


def _spy_scan(monkeypatch, repository):
    seen = {"batch_rows": [], "max_batch_rows": None}
    original = repository.scan

    def scan(request, **kwargs):
        if kwargs.get("table_name") != table.TABLE:
            yield from original(request, **kwargs)
            return
        seen["max_batch_rows"] = request.max_batch_rows
        for batch in original(request, **kwargs):
            seen["batch_rows"].append(batch.num_rows)
            yield batch

    monkeypatch.setattr(repository, "scan", scan)
    return seen


def test_stored_read_batches_one_fragment_and_equals_all_rows_read(panel_data, monkeypatch):
    data = panel_data
    monkeypatch.setattr(store, "_FRAGMENT_ROWS", 1)  # three single-row fragments
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    columns = [*table.FEATURES, "has_implied_quote"]
    context = artifact["context"]
    resolved = store.existing(data.repository, context)
    records = data.repository.fragment_records(resolved, table.TABLE)
    total = sum(r.row_count for r in records)
    assert len(records) > 1 and total > max(r.row_count for r in records)
    # Reference: the former all-rows read (one batch cap equal to the total).
    reference = data.repository.scan
    seen = _spy_scan(monkeypatch, data.repository)
    streamed = read(data, artifact)["rows"]
    assert seen["max_batch_rows"] == max(r.row_count for r in records) < total
    assert seen["batch_rows"] and max(seen["batch_rows"]) <= seen["max_batch_rows"]
    assert sum(seen["batch_rows"]) == total
    monkeypatch.setattr(data.repository, "scan", reference)
    from engine.v2.contracts.data import DataQuery, KeyPredicate
    request = DataQuery(snapshot_id=resolved.snapshot_id,
        table_contract_ref=resolved.table_versions[table.TABLE].table_contract_ref,
        columns=tuple(c.name for c in table.CONTRACT.columns), order_by=table.CONTRACT.primary_key,
        key_filter=(KeyPredicate(column="context_id", operator="eq", values=(context["context_id"],)),),
        max_batch_rows=min(total, table.CONTRACT.maximum_batch_rows), max_result_rows=total)
    everything = [r for b in reference(request, table_name=table.TABLE) for r in b.to_pylist()]
    expected = [{k: row[k] for k in (*table.METADATA, *columns)} for row in everything]
    assert streamed == expected
    assert all(row["lineage"] for row in streamed)
    slim = read(data, artifact, lineage=False)["rows"]
    assert slim == [{k: v for k, v in row.items() if k != "lineage"} for row in streamed]


def test_stored_read_still_refuses_a_corrupt_row_without_returning_rows(panel_data, monkeypatch):
    data = panel_data
    monkeypatch.setattr(store, "_FRAGMENT_ROWS", 1)  # three single-row fragments
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    original = data.repository.scan

    def scan(request, **kwargs):
        if kwargs.get("table_name") != table.TABLE:
            yield from original(request, **kwargs)
            return
        for index, batch in enumerate(original(request, **kwargs)):
            rows = batch.to_pylist()
            if index == 1:
                rows[0]["signed_streak"] = 123.
            yield type("B", (), {"to_pylist": lambda self, rows=rows: rows, "num_rows": len(rows)})()

    monkeypatch.setattr(data.repository, "scan", scan)
    with pytest.raises(DataError) as caught:
        read(data, artifact)
    assert caught.value.code == "CONTRACT_MISMATCH"
