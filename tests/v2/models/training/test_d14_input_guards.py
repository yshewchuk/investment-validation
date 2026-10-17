"""Pinned input bounds and stable refusals at the additive native D14 boundary."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest
from pandas.testing import assert_frame_equal

from engine.v2.data import errors
from engine.v2.data.reference_catalog import insert_reference_inputs
from engine.v2.models.training import d14_dataset
from tests.data_scan_support import commit_tables
from tests.v2.models.training.test_d14_dataset import _forbidden, _legacy
from tests.v2.models.training.test_d14_dataset import pinned as pinned
from tools import phase5_d14_inputs


@pytest.mark.parametrize("pinned", ("boundary",), indirect=True)
def test_fixed_monthly_population_matches_legacy_across_year_boundary(pinned, monkeypatch):
    expected = _legacy(pinned, "implied_t1", monkeypatch)
    actual = phase5_d14_inputs.build_dataset("implied_t1", **pinned["inputs"])
    assert_frame_equal(actual[expected.columns], expected, check_exact=True)
    assert set(actual.year) == {2024, 2025}
    assert {2016, 2027} <= set(pinned["events"].year)


@pytest.mark.parametrize("pinned", ("blank_session",), indirect=True)
def test_blank_session_refuses_like_legacy_instead_of_changing_population(pinned, monkeypatch):
    with pytest.raises(ValueError, match="session"):
        _legacy(pinned, "implied_t1", monkeypatch)
    with pytest.raises(errors.DataError) as err:
        phase5_d14_inputs.build_dataset("implied_t1", **pinned["inputs"])
    assert err.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("days", ([], ["2024-01-02", "2024-01-01"],
                                  ["2024-01-02", "2024-01-02"], [None],
                                  ["2024-01-02T01:00:00"], ["2024-01-02T00:00:00Z"], ["invalid"]))
def test_calendar_shape_has_stable_refusal(days):
    with pytest.raises(errors.DataError) as err:
        d14_dataset._decisions(pd.DataFrame(), SimpleNamespace(days=days))
    assert err.value.code == "CONTRACT_MISMATCH"


def test_unknown_role_refuses_before_inputs():
    with pytest.raises(errors.DataError) as err:
        d14_dataset.build_d14_dataset(None, None, calendar=None, role="unsupported")
    assert err.value.code == "CONTRACT_MISMATCH"


def test_missing_input_table_refuses_before_reads(pinned, monkeypatch):
    repo, snapshot = pinned["repository"], pinned["snapshot"]
    snapshot = dataclasses.replace(snapshot, table_versions={
        key: value for key, value in snapshot.table_versions.items() if key != "feature_panel"})
    monkeypatch.setattr(repo, "resolve", lambda _: snapshot)
    monkeypatch.setattr(repo, "scan", _forbidden)
    with pytest.raises(errors.DataError) as err:
        d14_dataset.build_d14_dataset(repo, snapshot, calendar=None, role="implied_t1")
    assert err.value.code == "CONTRACT_MISMATCH"


def test_table_default_bound_and_scan_overflow_are_enforced(pinned, monkeypatch):
    repo, snapshot = pinned["repository"], pinned["snapshot"]
    monkeypatch.setattr(repo, "scan_population_bound", lambda *args, **kw: 500001)
    queries = []
    monkeypatch.setattr(repo, "scan", lambda query, **kw: queries.append(query) or iter(()))
    assert list(d14_dataset._rows(repo, snapshot, "earnings_events", ("event_id",),
                                 start="2017-01-01", end="2027-01-01")) == []
    assert queries[0].max_result_rows == 500000
    monkeypatch.setattr(repo, "scan", lambda *args, **kw: iter([pa.table({"event_id": ["a", "b"]})]))
    with pytest.raises(errors.DataError) as err:
        list(d14_dataset._rows(repo, snapshot, "earnings_events", ("event_id",),
                               start="2017-01-01", end="2027-01-01", cap=1))
    assert err.value.code == "RESULT_LIMIT_EXCEEDED"


def test_ticker_row_bound_refuses_before_daily_calculation(monkeypatch):
    monkeypatch.setattr(d14_dataset, "_rows", lambda *args, **kw: iter(
        [{"ticker": "A", "date": "2024-01-01"}] * 10001))
    monkeypatch.setattr(d14_dataset.panel_math, "daily_state_lookup", _forbidden)
    with pytest.raises(errors.DataError) as err:
        d14_dataset._daily_features(pd.DataFrame({"ticker": ["A"]}), None, None)
    assert err.value.code == "RESULT_LIMIT_EXCEEDED"


@pytest.mark.parametrize("exit_spot", (0.0, -1.0))
def test_positive_entry_nonpositive_exit_has_missing_target(pinned, monkeypatch, exit_spot):
    original = d14_dataset._rows

    def rows(*args, **kwargs):
        for row in original(*args, **kwargs):
            if args[2] == "daily_market" and row["ticker"] == "AMC" \
                    and pd.Timestamp(row["date"]) >= pd.Timestamp("2024-06-25"):
                row = {**row, "spot": exit_spot}
            yield row

    monkeypatch.setattr(d14_dataset, "_rows", rows)
    frame = phase5_d14_inputs.build_dataset("runup_move", **pinned["inputs"])
    event = frame[(frame.ticker == "AMC") & (frame.event_date == "2024-07-05")].iloc[0]
    assert event.spot > 0
    assert pd.isna(event.runup_abs_move_d14)


def test_committed_receipt_for_other_snapshot_refuses(pinned):
    other = commit_tables(pinned["conn"], pinned["clock"],
                          {"earnings_events": pinned["records"]["earnings_events"]},
                          {"earnings_events": pinned["contracts"]["earnings_events"]},
                          store=pinned["store"], scope="other", receipt_id="r2", attempt_id="att-2")
    assert other.snapshot_id != pinned["snapshot"].snapshot_id
    insert_reference_inputs(pinned["conn"], "r2", (pinned["calendar_ref"],))
    pinned["conn"].commit()
    with pytest.raises(errors.DataError) as err:
        phase5_d14_inputs.build_dataset("implied_t1", **{**pinned["inputs"], "receipt_id": "r2"})
    assert err.value.code == "IDENTITY_CONFLICT"


def test_uncommitted_receipt_refuses(pinned):
    pinned["conn"].execute(
        "INSERT INTO data_import_receipts (receipt_id, attempt_id, fence, source_manifest_hash, "
        "status, registered_at, scope) SELECT 'failed', attempt_id, fence, source_manifest_hash, "
        "'failed', registered_at, scope FROM data_import_receipts WHERE receipt_id='r1'")
    pinned["conn"].commit()
    with pytest.raises(errors.DataError) as err:
        phase5_d14_inputs.build_dataset("implied_t1", **{**pinned["inputs"], "receipt_id": "failed"})
    assert err.value.code == "IDENTITY_CONFLICT"


@pytest.mark.parametrize("pinned,count", (("no_calendar", 0), ("normal", 2)), indirect=["pinned"])
def test_receipt_requires_exactly_one_calendar(pinned, count):
    if count == 2:
        insert_reference_inputs(pinned["conn"], "r1", (dataclasses.replace(
            pinned["calendar_ref"], legacy_path="data/raw/yf/second.csv"),))
    pinned["conn"].commit()
    with pytest.raises(errors.DataError) as err:
        phase5_d14_inputs.build_dataset("implied_t1", **pinned["inputs"])
    assert err.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("pinned", ("no_refs",), indirect=True)
def test_receipt_without_reference_inputs_refuses(pinned):
    with pytest.raises(errors.DataError) as err:
        phase5_d14_inputs.build_dataset("implied_t1", **pinned["inputs"])
    assert err.value.code == "SNAPSHOT_NOT_READY"


@pytest.mark.parametrize("case", ("missing", "invalid"))
def test_catalog_errors_are_typed(pinned, tmp_path, case):
    path = tmp_path / "bad.sqlite"
    if case == "invalid":
        path.write_bytes(b"not a sqlite catalog")
    with pytest.raises(errors.DataError) as err:
        phase5_d14_inputs.build_dataset("implied_t1", **{**pinned["inputs"], "catalog": path})
    assert err.value.code == "CONTRACT_MISMATCH"


def test_calendar_is_bounded_before_read(pinned, monkeypatch):
    ref = dataclasses.replace(pinned["calendar_ref"], byte_size=phase5_d14_inputs.MAX_CALENDAR_BYTES + 1)
    monkeypatch.setattr(pinned["store"], "read_verified", _forbidden)
    with pytest.raises(errors.DataError) as err:
        phase5_d14_inputs._calendar(pinned["store"], ref)
    assert err.value.code == "RESULT_LIMIT_EXCEEDED"


def test_calendar_consumes_exact_verified_bytes(pinned, monkeypatch):
    store = pinned["store"]
    original = store.read_verified

    def read(ref):
        data = original(ref)
        replacement = store.root / "replacement"
        replacement.write_bytes(b"changed after verification")
        replacement.replace(store.root / ref.storage_key)
        return data

    monkeypatch.setattr(store, "read_verified", read)
    actual = phase5_d14_inputs._calendar(store, pinned["calendar_ref"])
    assert list(actual.days[:len(pinned["days"])]) == list(pinned["days"])
    with pytest.raises(errors.DataError) as err:
        phase5_d14_inputs._calendar(store, pinned["calendar_ref"])
    assert err.value.code == "OBJECT_CORRUPT"


@pytest.mark.parametrize("data", (b"", b"Price,Close\nTicker,^GSPC\nDate,\n",
                                  b"Price,Close\nTicker,^GSPC\nDate,\nbad-date,100\n"))
def test_malformed_calendar_date_refuses(pinned, data):
    obj = pinned["store"].publish_bytes(data, schema_ref="parquet_fragment.v1")
    ref = dataclasses.replace(pinned["calendar_ref"], object_id=obj.artifact_id,
                              content_hash=obj.content_hash, byte_size=obj.byte_size)
    with pytest.raises(errors.DataError) as err:
        phase5_d14_inputs._calendar(pinned["store"], ref)
    assert err.value.code == "CONTRACT_MISMATCH"
