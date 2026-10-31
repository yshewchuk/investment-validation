"""Decoded-byte cap threading: ``read_table``/``load_population`` forward
``max_object_bytes`` into ``Repository.scan``; omitted (``None``) keeps the
scan call identical to today's, and a real fixture read refuses over the cap
while the default read succeeds.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.contracts.data import DatasetVersionRef, SnapshotRef, TableContractRef
from engine.v2.data import errors
from engine.v2.data.repository import Repository
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.research._scan import read_table
from engine.v2.research.experiment_population import load_population
from tests.v2.research.test_eligible_history import source

_COLUMNS = ("event_id", "ticker", "event_date", "session", "date_conflict", "event_cluster_id")
_CONTRACT_REF = TableContractRef(contract_id="earnings_events", definition_hash="sha256:" + "0" * 64)
_CONTRACT = SimpleNamespace(maximum_batch_rows=8, primary_key=("event_id",), partition_columns=())


def _snapshot():
    version = DatasetVersionRef(dataset_version_id="dv-cap", table_contract_ref=_CONTRACT_REF,
                                manifest_hash="sha256:" + "1" * 64)
    return SnapshotRef(snapshot_id="snap_cap_test", manifest_hash="sha256:" + "2" * 64,
                       table_versions={"earnings_events": version},
                       calendar_version="calendar_v1", source_priority_version="priority_v1",
                       finality_receipt_refs=(),
                       knowledge_mode_by_table={"earnings_events": "reconstructed"})


def _non_holdout_ids(count):
    ids, index = [], 0
    while len(ids) < count:
        candidate = f"ev-cap-{index}"
        if not ExperimentHoldouts.random_membership(candidate):
            ids.append(candidate)
        index += 1
    return ids


def _events_frame():
    ids = _non_holdout_ids(2)
    return pd.DataFrame([{"event_id": key, "ticker": "CAPS",
                          "event_date": pd.Timestamp(f"2020-01-0{rank + 5}"), "session": "AMC",
                          "date_conflict": False, "event_cluster_id": None}
                         for rank, key in enumerate(ids)])


class _Batch:
    def __init__(self, frame):
        self._frame = frame
        self.num_rows = len(frame)

    def to_pandas(self):
        return self._frame.copy()


class _RecordingRepository:
    """The scan surface ``_scan.read_table``/``load_population`` drive, recording each cap."""

    def __init__(self, snapshot, frame):
        self.snapshot, self.frame = snapshot, frame
        self.caps = []

    def resolve(self, snapshot_id):
        return self.snapshot if snapshot_id == self.snapshot.snapshot_id else None

    def table_contract(self, snapshot, table_name):
        return _CONTRACT

    def fragment_records(self, snapshot, table_name):
        return [SimpleNamespace(partition_key="2020")]

    def scan_population_bound(self, snapshot_id, **kwargs):
        return len(self.frame)

    def scan(self, query, *, table_name, max_object_bytes=None):
        self.caps.append(max_object_bytes)
        yield _Batch(self.frame)


def _repository():
    snapshot = _snapshot()
    return snapshot, _RecordingRepository(snapshot, _events_frame())


def test_read_table_forwards_max_object_bytes_to_scan():
    snapshot, repository = _repository()
    frame = read_table(repository, snapshot, "earnings_events", _COLUMNS, max_object_bytes=4096)
    assert repository.caps == [4096]
    assert len(frame) == 2


def test_read_table_without_cap_keeps_scan_kwarg_none():
    snapshot, repository = _repository()
    read_table(repository, snapshot, "earnings_events", _COLUMNS)
    assert repository.caps == [None]


def test_load_population_forwards_max_object_bytes_to_scan():
    snapshot, repository = _repository()
    frame = load_population(repository, snapshot, as_of_month="2024-01", max_object_bytes=4096)
    assert repository.caps == [4096]
    assert len(frame) == 2


def test_load_population_without_cap_keeps_scan_kwarg_none():
    snapshot, repository = _repository()
    load_population(repository, snapshot, as_of_month="2024-01")
    assert repository.caps == [None]


def test_fixture_load_population_decoded_cap_refuses_and_default_reads(tmp_path):
    data = source(tmp_path)
    try:
        with pytest.raises(errors.DataError) as raised:
            load_population(data.repository, data.snapshot, as_of_month="2026-01",
                            max_object_bytes=1)
        assert raised.value.code == "RESULT_LIMIT_EXCEEDED"
        frame = load_population(data.repository, data.snapshot, as_of_month="2026-01")
        assert not frame.empty
    finally:
        data.conn.close()


def test_fixture_read_table_decoded_cap_bounds_the_same_scan(tmp_path):
    data = source(tmp_path)
    try:
        assert isinstance(data.repository, Repository)
        with pytest.raises(errors.DataError) as raised:
            read_table(data.repository, data.snapshot, "earnings_events", _COLUMNS,
                       max_object_bytes=1)
        assert raised.value.code == "RESULT_LIMIT_EXCEEDED"
        assert len(read_table(data.repository, data.snapshot, "earnings_events", _COLUMNS)) > 0
    finally:
        data.conn.close()
