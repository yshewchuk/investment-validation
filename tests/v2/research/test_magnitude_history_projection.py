"""Full-source admission and identity oracles for bounded projected history reads."""
import hashlib
import json

import pyarrow.parquet as pq
import pytest

from engine.v2.data import objects
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import canonical_json, to_document
from engine.v2.research import magnitude_history_store as stored
from engine.v2.research.magnitude_history_table import FEATURES, METADATA, TABLE
from tests.v2.research.magnitude_trial_support import build_history, poison_outcomes
from tests.v2.research.test_magnitude_history_store import source  # noqa: F401


def _projection(value, snapshot, context, *, columns=("n_prior", "mean_prior_abs_move")):
    return stored.read_stored_for_events(value.repository, snapshot, context,
        event_ids=[context["temporal_plan"]["events"][0]["event_id"]], columns=columns)


def test_projected_rows_keep_full_snapshot_identity_and_raw_numeric_values(source, monkeypatch):  # noqa: F811
    build, snapshot = build_history(source)
    context = build["context"]
    # Independent physical row decode and literal envelope hash, without the reader.
    rows = [row for record in source.repository.fragment_records(snapshot, TABLE)
        for row in pq.read_table(objects.verify_object_path(source.store, record.object_ref)).to_pylist()]
    rows.sort(key=lambda row: (row["event_id"], row["context_id"], row["horizon_sessions"]))
    identity = {"context": context, "feature_snapshot": to_document(snapshot), "rows": rows}
    expected = "sha256:" + hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    full = stored.read_stored(source.repository, snapshot, context)
    poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
    columns = ("mean_prior_move", "n_prior", "ema12_prior_abs_move", "mean_prior_abs_move")
    projected = _projection(source, snapshot, context, columns=columns)
    event = context["temporal_plan"]["events"][0]["event_id"]
    keys = ("event_id", "context_id", "horizon_sessions", *columns)
    assert projected == {"context": context, "feature_snapshot": to_document(snapshot),
        "rows": [{key: row[key] for key in keys} for row in rows if row["event_id"] == event],
        "inputs_id": expected}
    assert full["inputs_id"] == expected
    assert _projection(source, snapshot, context, columns=columns) == projected
    assert all("history_json" not in row for row in projected["rows"])


def test_projection_globally_admits_metadata_before_numbers_in_bounded_batches(source, monkeypatch):  # noqa: F811
    build, snapshot = build_history(source)
    original, requests = Repository.scan, []
    metadata_rows, checked_features = [], []
    metadata, features = stored._metadata, stored._validate_features

    def scan(repository, request, *, table_name):
        if table_name == TABLE:
            requests.append(request)
            assert request.max_batch_rows == 3
            assert request.max_result_rows == 12
            assert len(request.key_filter) == 1 and request.key_filter[0].column == "context_id"
        yield from original(repository, request, table_name=table_name)

    def admit(row, *args):
        metadata_rows.append(row["event_id"])
        return metadata(row, *args)

    def validate(row, count):
        assert len(metadata_rows) == 12
        checked_features.append(row["event_id"])
        return features(row, count)

    monkeypatch.setattr(Repository, "scan", scan)
    monkeypatch.setattr(stored, "_metadata", admit)
    monkeypatch.setattr(stored, "_validate_features", validate)
    result = _projection(source, snapshot, build["context"])
    assert [request.columns for request in requests] == [METADATA, (*METADATA, *FEATURES)]
    assert len(result["rows"]) == 3 and metadata_rows == checked_features


@pytest.mark.parametrize("field,value", [
    ("cutoff_id", "wrong"), ("source_session_ceiling", "2099-01-01"),
    ("history_status", "empty"), ("history_json", "{}"), ("event_date", "1900-01-01"),
])
def test_unselected_metadata_corruption_refuses_before_any_feature_scan(source, monkeypatch, field, value):  # noqa: F811
    build, snapshot = build_history(source)
    context = build["context"]
    selected = context["temporal_plan"]["events"][0]["event_id"]
    original, changed = stored._stream_rows, []

    def corrupt(repository, snapshot, context, columns, count):
        assert tuple(columns) == METADATA, "numeric columns opened before full metadata admission"
        for row in original(repository, snapshot, context, columns, count):
            if row["event_id"] != selected and not changed:
                row[field] = value
                changed.append(1)
            yield row

    monkeypatch.setattr(stored, "_stream_rows", corrupt)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        _projection(source, snapshot, context)
    assert changed == [1]


@pytest.mark.parametrize("field,value", [
    ("n_prior", 999), ("mean_prior_move", float("nan")),
    ("mean_prior_abs_move", -1), ("ema12_prior_abs_move", 0),
])
def test_unselected_unprojected_feature_corruption_still_refuses(source, monkeypatch, field, value):  # noqa: F811
    build, snapshot = build_history(source)
    context = build["context"]
    selected = context["temporal_plan"]["events"][0]["event_id"]
    original, changed = stored._stream_rows, []

    def corrupt(repository, snapshot, context, columns, count):
        for row in original(repository, snapshot, context, columns, count):
            if tuple(columns) != METADATA and row["event_id"] != selected and not changed:
                row[field] = value
                changed.append(1)
            yield row

    monkeypatch.setattr(stored, "_stream_rows", corrupt)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        _projection(source, snapshot, context, columns=("n_prior",))
    assert changed == [1]


@pytest.mark.parametrize("pass_number", [1, 2])
@pytest.mark.parametrize("change", ["missing", "extra", "metadata"])
def test_each_pass_requires_exact_coverage_and_unchanged_metadata(source, monkeypatch, pass_number, change):  # noqa: F811
    build, snapshot = build_history(source)
    original = stored._stream_rows

    def corrupt(repository, snapshot, context, columns, count):
        number = 1 if tuple(columns) == METADATA else 2
        for index, row in enumerate(original(repository, snapshot, context, columns, count)):
            if number == pass_number and index == count - 1:
                if change == "missing":
                    continue
                if change == "metadata":
                    row["history_json"] = row["history_json"] + " "
                if change == "extra":
                    yield row
            yield row

    monkeypatch.setattr(stored, "_stream_rows", corrupt)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        _projection(source, snapshot, build["context"])


@pytest.mark.parametrize("events,columns", [(["absent"], ["n_prior"]), ([], ["history_json"]),
                                          ([], ["n_prior", "n_prior"])])
def test_invalid_projection_refuses_before_payload(source, monkeypatch, events, columns):  # noqa: F811
    build, snapshot = build_history(source)
    monkeypatch.setattr(stored, "_stream_rows", lambda *args: pytest.fail("invalid projection scanned"))
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        stored.read_stored_for_events(source.repository, snapshot, build["context"],
                                      event_ids=events, columns=columns)


def test_empty_projection_still_hashes_and_validates_every_row(source):  # noqa: F811
    build, snapshot = build_history(source)
    full = stored.read_stored(source.repository, snapshot, build["context"])
    result = stored.read_stored_for_events(source.repository, snapshot, build["context"],
                                          event_ids=[], columns=[])
    assert result["rows"] == [] and result["inputs_id"] == full["inputs_id"]


@pytest.mark.parametrize("boundary", ["_metadata", "_validate_features", "_stream_identity"])
def test_refused_projection_closes_fragment_streams_even_with_retained_traceback(source, monkeypatch, boundary):  # noqa: F811
    build, snapshot = build_history(source)
    original = Repository._fragment_rows
    active = []

    def tracked(*args, **kwargs):
        active.append(1)
        try:
            yield from original(*args, **kwargs)
        finally:
            active.pop()

    def refuse(*args):
        if boundary == "_stream_identity":
            next(args[-1])
        assert active
        raise ValueError("synthetic refusal")

    monkeypatch.setattr(Repository, "_fragment_rows", tracked)
    monkeypatch.setattr(stored, boundary, refuse)
    with pytest.raises(ValueError, match="synthetic refusal") as caught:
        _projection(source, snapshot, build["context"])
    assert caught.value.__traceback__ is not None
    assert active == []


def test_streamed_identity_matches_canonical_numeric_and_unicode_envelope():
    # Independent canonical identity covers null, -0, exponent edges and escaped lineage.
    class Snapshot:
        pass

    rows = [{"n": value, "history_json": json.dumps({"text": '雪😀"\\\n'})}
            for value in (None, -0.0, 1e-7, 1e-6, 1e20, 1e21, 7)]
    # An ordinary real snapshot document is unnecessary for the internal framing oracle.
    from unittest.mock import patch
    with patch.object(stored, "to_document", return_value={"snapshot_id": "雪"}):
        expected = {"context": {"a": "😀"}, "feature_snapshot": {"snapshot_id": "雪"}, "rows": rows}
        digest = "sha256:" + hashlib.sha256(canonical_json(expected).encode()).hexdigest()
        assert stored._stream_identity(expected["context"], Snapshot(), iter(rows)) == digest
