"""Plain-fake coverage for outcome_binding anchors, bound and fragment_index."""
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.data.errors import DataError
from engine.v2.research import outcome_binding

TICKER, DAY = "SYNTH", "2024-01-05"
KEY = (TICKER, DAY)
HASH = "sha256:" + "1" * 64
LIMITS = {"max_object_bytes": 1 << 20}
SNAPSHOT = SimpleNamespace(snapshot_id="snap_test", table_versions={})


def record(row_count=1, key=KEY, partition_key=TICKER, object_id="obj-1", content_hash=HASH):
    return SimpleNamespace(partition_key=partition_key, row_count=row_count,
        primary_key_min=key, primary_key_max=key, time_min=key[1], time_max=key[1],
        object_ref=SimpleNamespace(object_id=object_id, content_hash=content_hash, byte_size=8))


def row(**changes):
    base = {"ticker": TICKER, "event_date": DAY, "session": "AMC", "pre_close_date": "2024-01-04",
        "post_close_date": "2024-01-05", "outcome_object_id": "obj-1", "outcome_content_hash": HASH}
    return {**base, **changes}


def event():
    return SimpleNamespace(ticker=TICKER, event_date=pd.Timestamp(DAY), session="AMC")


def repository(records):
    return SimpleNamespace(fragment_records=lambda snapshot, table: records)


def test_fragment_index_returns_unique_rows_keyed_by_primary_min():
    one, two = record(), record(key=("OTHER", DAY), partition_key="OTHER")
    assert outcome_binding.fragment_index(repository([one, two]), SNAPSHOT) == {KEY: one,
        ("OTHER", DAY): two}


def test_fragment_index_refuses_duplicate_identity():
    with pytest.raises(DataError) as caught:
        outcome_binding.fragment_index(repository([record(), record()]), SNAPSHOT)
    assert caught.value.code == "HOLDOUT_ACCESS_DENIED"


def test_bound_refuses_missing_anchor_row():
    with pytest.raises(DataError) as caught:
        outcome_binding.bound(event(), {}, {KEY: record()})
    assert caught.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("records", [
    pytest.param({}, id="missing-record"),
    pytest.param({KEY: record(row_count=2)}, id="row-count-not-one"),
    pytest.param({KEY: record(object_id="other")}, id="object-id-mismatch"),
    pytest.param({KEY: record(content_hash="sha256:" + "0" * 64)}, id="content-hash-mismatch"),
])
def test_bound_refuses_unauthenticated_outcome(records):
    with pytest.raises(DataError) as caught:
        outcome_binding.bound(event(), {KEY: row()}, records)
    assert caught.value.code == "HOLDOUT_ACCESS_DENIED"


def test_bound_null_bounds_return_none_and_valid_bracket_returns_row():
    assert outcome_binding.bound(event(), {KEY: row(pre_close_date=None, post_close_date=None)},
        {KEY: record()}) is None
    expected = row()
    assert outcome_binding.bound(event(), {KEY: expected}, {KEY: record()}) is expected


@pytest.mark.parametrize("changes", [
    pytest.param({"pre_close_date": "bad-date"}, id="non-iso"),
    pytest.param({"pre_close_date": None}, id="one-null-pre"),
    pytest.param({"post_close_date": None}, id="one-null-post"),
    pytest.param({"pre_close_date": "2024-01-08", "post_close_date": "2024-01-04"}, id="reversed"),
    pytest.param({"pre_close_date": "2023-12-29"}, id="span-over-five-days"),
    pytest.param({"pre_close_date": "2024-01-08", "post_close_date": "2024-01-09"}, id="event-outside"),
    pytest.param({"session": "BMO"}, id="session-mismatch"),
])
def test_bound_refuses_invalid_bracket(changes):
    with pytest.raises(DataError) as caught:
        outcome_binding.bound(event(), {KEY: row(**changes)}, {KEY: record()})
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_anchors_refuses_missing_required_ticker():
    with pytest.raises(DataError) as caught:
        outcome_binding.anchors(repository([record()]), SNAPSHOT, LIMITS, required_tickers=("OTHER",))
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_anchors_returns_empty_for_zero_rows():
    empty = SimpleNamespace(partition_key=TICKER, row_count=0)
    assert outcome_binding.anchors(repository([empty]), SNAPSHOT, LIMITS) == {}
