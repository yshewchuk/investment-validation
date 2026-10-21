"""Full-source admission and identity oracles for bounded projected history reads."""
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from engine.v2.data import errors, objects
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import canonical_json, to_document
from engine.v2.research import magnitude_history_store as stored
from engine.v2.research.magnitude_history_inputs import admit_stored
from engine.v2.research.magnitude_history_table import FEATURES, METADATA, TABLE
from tests.v2.research.magnitude_trial_support import build_history, poison_outcomes
from tests.v2.research.test_magnitude_history_store import source  # noqa: F401


def _projection(value, snapshot, context, *, columns=("n_prior", "mean_prior_abs_move")):
    return stored.read_stored_for_events(value.repository, snapshot, context,
        event_ids=[context["temporal_plan"]["events"][0]["event_id"]], columns=columns)


def _assert_fresh_bounded_scans(value, columns):
    assert [(table, request.columns, request.max_batch_rows, request.max_result_rows)
            for table, request in value.repository.scans] == [
        (TABLE, projection, 3, 12) for projection in columns]


@pytest.mark.parametrize("reader", ["full", "projection", "admission"])
@pytest.mark.parametrize("change,code,message", [
    ("receipt", "IDENTITY_CONFLICT", "history build receipt differs from its context"),
    ("snapshot", "IDENTITY_CONFLICT", "feature snapshot is not the committed context output"),
    ("contract", "CONTRACT_MISMATCH", "stored history contract differs"),
    ("coverage", "HOLDOUT_ACCESS_DENIED", "stored history object coverage differs"),
    ("isolation", "HOLDOUT_ACCESS_DENIED", "stored history event isolation cannot be proven"),
])
def test_all_read_paths_share_exact_admission_refusals_before_payload(source, monkeypatch, reader, change, code, message):  # noqa: F811
    build, snapshot = build_history(source)
    context = build["context"]
    records = source.repository.fragment_records(snapshot, TABLE)
    if change == "receipt":
        class CorruptReceipt:
            def execute(self, statement, parameters=()):
                cursor = source.conn.execute(statement, parameters)
                if "FROM data_import_receipts WHERE receipt_id=" in statement:
                    row = {**dict(cursor.fetchone()), "scope": "wrong"}
                    return SimpleNamespace(fetchone=lambda: row)
                return cursor
        monkeypatch.setattr(source.repository, "_conn", CorruptReceipt())
    elif change == "snapshot":
        monkeypatch.setattr(source.repository, "resolve", lambda _: source.snapshot)
    elif change == "contract":
        contract = source.repository.table_contract(snapshot, TABLE)
        monkeypatch.setattr(source.repository, "table_contract",
                            lambda *_: replace(contract, semantic_version="0.0.0"))
    else:
        corrupted = records[:-1] if change == "coverage" else (
            *records[:-1], replace(records[-1], primary_key_max=(*records[-1].primary_key_max[:-1], 6)))
        monkeypatch.setattr(source.repository, "fragment_records", lambda *_: corrupted)
    monkeypatch.setattr(source.repository, "scan",
                        lambda *args, **kwargs: pytest.fail("refused admission opened payloads"))
    with pytest.raises(DataError) as caught:
        if reader == "projection":
            _projection(source, snapshot, context)
        elif reader == "full":
            stored.read_stored(source.repository, snapshot, context)
        else:
            admit_stored(source.repository, snapshot, context)
    assert caught.value.problem == errors.fail(code, message).problem


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


@pytest.mark.parametrize("prior_success", [False, True])
@pytest.mark.parametrize("field,value", [
    ("cutoff_id", "wrong"), ("source_session_ceiling", "2099-01-01"),
    ("history_status", "empty"), ("history_json", "{}"), ("event_date", "1900-01-01"),
])
def test_each_read_revalidates_unselected_metadata_before_feature_scan(source, monkeypatch, field, value, prior_success):  # noqa: F811
    build, snapshot = build_history(source)
    context = build["context"]
    selected = context["temporal_plan"]["events"][0]["event_id"]
    if prior_success:
        assert len(_projection(source, snapshot, context)["rows"]) == 3
    source.repository.scans.clear()
    original, changed = stored._stream_rows, []

    def corrupt(repository, snapshot, context, columns, count):
        assert tuple(columns) == METADATA, "numeric columns opened before full metadata admission"
        for row in original(repository, snapshot, context, columns, count):
            if row["event_id"] != selected and not changed:
                row[field] = value
                changed.append(1)
            yield row

    monkeypatch.setattr(stored, "_stream_rows", corrupt)
    with pytest.raises(DataError) as caught:
        _projection(source, snapshot, context)
    assert caught.value.problem == errors.fail(
        "CONTRACT_MISMATCH", "stored history metadata differs from its context").problem
    assert changed == [1]
    _assert_fresh_bounded_scans(source, [METADATA])


@pytest.mark.parametrize("prior_success", [False, True])
@pytest.mark.parametrize("field,value,message", [
    ("n_prior", 999, "stored history count differs"),
    ("mean_prior_move", float("nan"), "stored history feature or missingness is invalid"),
    ("mean_prior_abs_move", -1, "stored history feature or missingness is invalid"),
    ("ema12_prior_abs_move", 0, "stored history feature or missingness is invalid"),
])
def test_each_read_revalidates_unselected_unprojected_features(source, monkeypatch, field, value, message, prior_success):  # noqa: F811
    build, snapshot = build_history(source)
    context = build["context"]
    selected = context["temporal_plan"]["events"][0]["event_id"]
    if prior_success:
        assert len(_projection(source, snapshot, context, columns=("n_prior",))["rows"]) == 3
    source.repository.scans.clear()
    original, changed = stored._stream_rows, []

    def corrupt(repository, snapshot, context, columns, count):
        for row in original(repository, snapshot, context, columns, count):
            if tuple(columns) != METADATA and row["event_id"] != selected and not changed:
                row[field] = value
                changed.append(1)
            yield row

    monkeypatch.setattr(stored, "_stream_rows", corrupt)
    with pytest.raises(DataError) as caught:
        _projection(source, snapshot, context, columns=("n_prior",))
    assert caught.value.problem == errors.fail("CONTRACT_MISMATCH", message).problem
    assert changed == [1]
    _assert_fresh_bounded_scans(source, [METADATA, (*METADATA, *FEATURES)])


def test_second_read_reverifies_unselected_stored_object_bytes(source):  # noqa: F811
    build, snapshot = build_history(source)
    context = build["context"]
    selected = context["temporal_plan"]["events"][0]["event_id"]
    record = next(record for record in source.repository.fragment_records(snapshot, TABLE)
                  if record.primary_key_min[0] != selected)
    path = objects.verify_object_path(source.store, record.object_ref)
    assert len(_projection(source, snapshot, context)["rows"]) == 3
    source.repository.scans.clear()
    # Corrupt only this synthetic fixture's unselected object after a successful
    # read. Keep the same Repository, snapshot, context and projection request.
    payload = path.read_bytes()
    path.chmod(0o600)
    path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    with pytest.raises(DataError) as caught:
        _projection(source, snapshot, context)
    assert caught.value.problem == errors.fail(
        "OBJECT_CORRUPT", "published object bytes do not match its recorded hash").problem
    _assert_fresh_bounded_scans(source, [METADATA])


@pytest.mark.parametrize("pass_number,change,message", [
    (1, "missing", "stored history row coverage differs"),
    (1, "extra", "stored history row coverage differs"),
    (1, "metadata", "stored history metadata differs from its context"),
    (2, "missing", "stored history row count changed"),
    (2, "extra", "stored history row count changed"),
    (2, "metadata", "stored history count or metadata changed"),
])
def test_each_pass_requires_exact_coverage_and_unchanged_metadata(source, monkeypatch, pass_number, change, message):  # noqa: F811
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
    with pytest.raises(DataError) as caught:
        _projection(source, snapshot, build["context"])
    assert caught.value.problem == errors.fail("CONTRACT_MISMATCH", message).problem


@pytest.mark.parametrize("events,columns", [
    (["absent"], ["n_prior"]), ([], ["history_json"]), ([], ["n_prior", "n_prior"]),
    (None, ["n_prior"]), ([["absent"]], ["n_prior"]), ("absent", ["n_prior"]),
    ([None], ["n_prior"]), ([1], ["n_prior"]),
    ([], None), ([], [["n_prior"]]), ([], "n_prior"), ([], [None]), ([], [1]),
])
def test_invalid_projection_refuses_before_payload(source, monkeypatch, events, columns):  # noqa: F811
    build, snapshot = build_history(source)
    monkeypatch.setattr(stored, "_stream_rows", lambda *args: pytest.fail("invalid projection scanned"))
    with pytest.raises(DataError) as caught:
        stored.read_stored_for_events(source.repository, snapshot, build["context"],
                                      event_ids=events, columns=columns)
    assert caught.value.problem == errors.fail(
        "CONTRACT_MISMATCH", "stored history projection differs from its context").problem


def test_empty_projection_still_hashes_and_validates_every_row(source):  # noqa: F811
    build, snapshot = build_history(source)
    full = stored.read_stored(source.repository, snapshot, build["context"])
    result = stored.read_stored_for_events(source.repository, snapshot, build["context"],
                                          event_ids=[], columns=[])
    assert result["rows"] == [] and result["inputs_id"] == full["inputs_id"]


@pytest.mark.parametrize("prior_success", [False, True])
@pytest.mark.parametrize("boundary", ["_metadata", "_validate_features", "_stream_identity"])
def test_refused_projection_closes_fragment_streams_even_with_retained_traceback(source, monkeypatch, boundary, prior_success):  # noqa: F811
    build, snapshot = build_history(source)
    if prior_success:
        assert len(_projection(source, snapshot, build["context"])["rows"]) == 3
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
        raise errors.fail("CONTRACT_MISMATCH", "synthetic refusal")

    monkeypatch.setattr(Repository, "_fragment_rows", tracked)
    monkeypatch.setattr(stored, boundary, refuse)
    with pytest.raises(DataError) as caught:
        _projection(source, snapshot, build["context"])
    assert caught.value.problem == errors.fail("CONTRACT_MISMATCH", "synthetic refusal").problem
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
