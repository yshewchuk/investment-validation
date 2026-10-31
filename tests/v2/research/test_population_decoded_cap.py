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
from engine.v2.research import magnitude_features
from engine.v2.research._scan import read_table
from engine.v2.research.eligibility_policy import plan_eligibility
from engine.v2.research.experiment_population import canonical_events, load_population
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
        return [SimpleNamespace(partition_key="2020", fragment_id="frag-2020",
                                object_ref=SimpleNamespace(object_id="obj-2020",
                                                           content_hash="sha256:" + "3" * 64))]

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


class _NoWorkRepository:
    """A snapshot surface that must stay untouched: no fragments, no scans."""

    def __init__(self, snapshot):
        self.snapshot = snapshot

    def resolve(self, snapshot_id):
        raise AssertionError("an invalid cap must refuse before snapshot resolution")

    def table_contract(self, snapshot, table_name):
        return _CONTRACT

    def fragment_records(self, snapshot, table_name):
        return []

    def scan_population_bound(self, snapshot_id, **kwargs):
        raise AssertionError("no bound may be planned")

    def scan(self, query, *, table_name, max_object_bytes=None):
        raise AssertionError("no scan may start")


@pytest.mark.parametrize("cap", [0, True, -1, "x"])
def test_read_table_refuses_invalid_cap_even_without_partitions(cap):
    snapshot = _snapshot()
    with pytest.raises(errors.DataError) as raised:
        read_table(_NoWorkRepository(snapshot), snapshot, "earnings_events", _COLUMNS,
                   max_object_bytes=cap)
    assert raised.value.code == "QUERY_NOT_BOUNDED"


@pytest.mark.parametrize("cap", [0, True, -1, "x"])
def test_load_population_refuses_invalid_cap_before_any_work(cap):
    snapshot = _snapshot()
    with pytest.raises(errors.DataError) as raised:
        load_population(_NoWorkRepository(snapshot), snapshot, as_of_month="2024-01",
                        max_object_bytes=cap)
    assert raised.value.code == "QUERY_NOT_BOUNDED"


def test_valid_cap_passes_admission_on_a_partitionless_snapshot():
    snapshot = _snapshot()
    frame = read_table(_NoWorkRepository(snapshot), snapshot, "earnings_events", _COLUMNS,
                       max_object_bytes=4096)
    assert frame.empty and list(frame.columns) == list(_COLUMNS)


def test_valid_cap_forwards_unchanged_to_every_partition_scan():
    snapshot = _snapshot()
    repository = _RecordingRepository(snapshot, _events_frame())
    load_population(repository, snapshot, as_of_month="2024-01",
                    max_object_bytes=2 * 10**9 + 7)
    assert repository.caps == [2 * 10**9 + 7]


def _calendar_snapshot():
    version = DatasetVersionRef(dataset_version_id="dv-cal", table_contract_ref=_CONTRACT_REF,
                                manifest_hash="sha256:" + "4" * 64)
    return SnapshotRef(snapshot_id="snap_cal_test", manifest_hash="sha256:" + "5" * 64,
                       table_versions={"daily_market": version},
                       calendar_version="calendar_v1", source_priority_version="priority_v1",
                       finality_receipt_refs=(),
                       knowledge_mode_by_table={"daily_market": "reconstructed"})


def _calendar_repository():
    snapshot = _calendar_snapshot()
    frame = pd.DataFrame({"date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]})
    return snapshot, _RecordingRepository(snapshot, frame)


def test_calendar_input_forwards_max_object_bytes_to_daily_market_scan():
    snapshot, repository = _calendar_repository()
    calendar, _ = magnitude_features.calendar_input(repository, snapshot, max_object_bytes=4096)
    assert repository.caps == [4096]
    assert calendar.days == ("2020-01-02", "2020-01-03")


def test_calendar_input_without_cap_keeps_scan_kwarg_none():
    snapshot, repository = _calendar_repository()
    magnitude_features.calendar_input(repository, snapshot)
    assert repository.caps == [None]


def test_fixture_calendar_input_decoded_cap_refuses_and_default_reads(tmp_path):
    data = source(tmp_path)
    try:
        with pytest.raises(errors.DataError) as raised:
            magnitude_features.calendar_input(data.repository, data.snapshot, max_object_bytes=1)
        assert raised.value.code == "RESULT_LIMIT_EXCEEDED"
        calendar, _ = magnitude_features.calendar_input(data.repository, data.snapshot)
        assert calendar.days
    finally:
        data.conn.close()


class _PinnedCappedCalendarRepository(_RecordingRepository):
    """Pinned so the memo applies; a capped scan refuses like the bounded decoder."""

    def __init__(self, snapshot, frame):
        super().__init__(snapshot, frame)
        self.pins_snapshots = True

    def scan(self, query, *, table_name, max_object_bytes=None):
        self.caps.append(max_object_bytes)
        if max_object_bytes is not None:
            raise errors.fail("RESULT_LIMIT_EXCEEDED", "capped scan refuses")
        yield _Batch(self.frame)


def test_calendar_cap_is_never_served_from_a_warm_uncapped_memo():
    snapshot = _calendar_snapshot()
    frame = pd.DataFrame({"date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]})
    repository = _PinnedCappedCalendarRepository(snapshot, frame)
    warm, _ = magnitude_features.calendar_input(repository, snapshot)
    assert repository.caps == [None]
    with pytest.raises(errors.DataError) as raised:
        magnitude_features.calendar_input(repository, snapshot, max_object_bytes=1)
    assert raised.value.code == "RESULT_LIMIT_EXCEEDED"
    assert repository.caps == [None, 1]
    again, _ = magnitude_features.calendar_input(repository, snapshot)
    assert repository.caps == [None, 1]
    assert again == warm


class _UntouchableRepository:
    """Any repository use means the invalid cap was not refused at entry."""

    def __getattr__(self, name):
        raise AssertionError("an invalid cap must refuse before any repository use")


@pytest.mark.parametrize("cap", [0, True, -1])
def test_canonical_events_refuses_invalid_cap_before_repository_use(cap):
    snapshot = _snapshot()
    with pytest.raises(errors.DataError) as raised:
        canonical_events(_UntouchableRepository(), snapshot, max_object_bytes=cap)
    assert raised.value.code == "QUERY_NOT_BOUNDED"


@pytest.mark.parametrize("cap", [0, True, -1])
def test_plan_eligibility_refuses_invalid_cap_before_repository_use(cap):
    snapshot = _snapshot()
    with pytest.raises(errors.DataError) as raised:
        plan_eligibility(_UntouchableRepository(), snapshot, as_of_month="2026-01",
                         policy=None, metadata=None, evaluation_event_ids=[],
                         max_object_bytes=cap)
    assert raised.value.code == "QUERY_NOT_BOUNDED"


@pytest.mark.parametrize("cap", [0, True, -1])
def test_calendar_input_refuses_invalid_cap_before_repository_use(cap):
    snapshot = _calendar_snapshot()
    with pytest.raises(errors.DataError) as raised:
        magnitude_features.calendar_input(_UntouchableRepository(), snapshot,
                                          max_object_bytes=cap)
    assert raised.value.code == "QUERY_NOT_BOUNDED"
