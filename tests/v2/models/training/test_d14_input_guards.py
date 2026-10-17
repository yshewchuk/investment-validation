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
from engine.v2.foundation import ArtifactStore, artifact_reference
from engine.v2.models.training import build_d14_dataset, d14_dataset, d14_inputs
from tests.data_scan_support import commit_tables
from tests.v2.models.training.d14_support import forbidden, legacy_frame
from tests.v2.models.training.d14_support import pinned as pinned
from tools import phase5_d14_inputs


@pytest.mark.parametrize("pinned", ("boundary",), indirect=True)
def test_fixed_monthly_population_matches_legacy_across_year_boundary(pinned, monkeypatch):
    outside = pinned["events"].loc[pinned["events"].event_id == "boundary-2016"].iloc[0]
    assert outside.session == "AMC"
    assert sum(pinned["days"] <= outside.event_date) >= 15
    expected = legacy_frame(pinned, "implied_t1", monkeypatch)
    actual = phase5_d14_inputs.build_dataset("implied_t1", **pinned["inputs"])
    assert_frame_equal(actual[expected.columns], expected, check_exact=True)
    assert set(actual.year) == {2017, 2024, 2025, 2026}
    assert outside.event_date not in set(actual.event_date)
    assert pd.Timestamp("2017-01-02") in set(actual.event_date)
    assert pd.Timestamp("2026-01-02") in set(actual.event_date)
    history = actual.loc[actual.event_date == pd.Timestamp("2026-01-02"), list(d14_dataset.HISTORY_COLUMNS)]
    assert len(history) == 1 and history.notna().all().all()
    assert history.iloc[0].n_prior == 26
    assert history.iloc[0].mean_prior_move == 26.5


@pytest.mark.parametrize("pinned", ("blank_session",), indirect=True)
def test_blank_session_refuses_like_legacy_instead_of_changing_population(pinned, monkeypatch):
    with pytest.raises(ValueError, match="session"):
        legacy_frame(pinned, "implied_t1", monkeypatch)
    with pytest.raises(errors.DataError) as err:
        phase5_d14_inputs.build_dataset("implied_t1", **pinned["inputs"])
    assert err.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("pinned", ("warmup",), indirect=True)
def test_calendar_warmup_boundary_matches_legacy(pinned, monkeypatch):
    expected = legacy_frame(pinned, "implied_t1", monkeypatch)
    actual = build_d14_dataset("implied_t1", **pinned["inputs"])
    assert_frame_equal(actual[expected.columns], expected, check_exact=True)
    assert pinned["days"][13] not in set(actual.event_date)
    row = actual[actual.event_date == pinned["days"][14]].iloc[0]
    assert row.entry_date == pinned["days"][0]


@pytest.mark.parametrize("pinned", ("early_daily",), indirect=True)
def test_first_recipe_year_retains_prior_year_daily_history(pinned, monkeypatch):
    expected = legacy_frame(pinned, "implied_t1", monkeypatch)
    actual = build_d14_dataset("implied_t1", **pinned["inputs"])
    assert_frame_equal(actual[expected.columns], expected, check_exact=True)
    assert len(actual) == 1 and actual.iloc[0].entry_date.year == 2016
    assert actual.iloc[0].spot > 0


@pytest.mark.parametrize("days", ([], ["2024-01-02", "2024-01-01"],
                                  ["2024-01-02", "2024-01-02"], [None],
                                  ["2024-01-02T01:00:00"], ["2024-01-02T00:00:00Z"], ["invalid"]))
def test_calendar_shape_has_stable_refusal(days):
    with pytest.raises(errors.DataError) as err:
        d14_dataset.decision_rows(pd.DataFrame(), SimpleNamespace(days=days))
    assert err.value.code == "CONTRACT_MISMATCH"


def test_unknown_role_refuses_before_inputs():
    with pytest.raises(errors.DataError) as err:
        d14_dataset.calculate_frame(None, None, calendar=None, role="unsupported")
    assert err.value.code == "CONTRACT_MISMATCH"


def test_verified_builder_refuses_unknown_role_before_io(monkeypatch):
    monkeypatch.setattr(d14_inputs.sqlite3, "connect", forbidden)
    with pytest.raises(errors.DataError) as err:
        build_d14_dataset("unsupported", catalog=None, artifact_root=None,
                          snapshot_id=None, receipt_id=None)
    assert err.value.code == "CONTRACT_MISMATCH"


def test_cli_refuses_unknown_role_before_inputs(pinned, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(phase5_d14_inputs, "build_dataset", forbidden)
    out = tmp_path / "out"
    args = ["--role", "unsupported", "--out", str(out)]
    for name, value in pinned["inputs"].items():
        args += ["--" + name.replace("_", "-"), str(value)]
    with pytest.raises(SystemExit) as caught:
        phase5_d14_inputs.main(args)
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert "argument --role: invalid choice" in captured.err
    assert captured.out == ""
    assert not out.exists()


@pytest.mark.parametrize("column", ("n_prior", "signed_streak"))
def test_null_integer_history_has_typed_refusal(pinned, monkeypatch, column):
    original = d14_dataset.input_frame

    def frame(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[2] == "feature_panel":
            result[column] = result[column].astype(object)
            result.loc[result.index[0], column] = None
        return result

    monkeypatch.setattr(d14_dataset, "input_frame", frame)
    with pytest.raises(errors.DataError) as err:
        build_d14_dataset("implied_t1", **pinned["inputs"])
    assert err.value.code == "CONTRACT_MISMATCH"
    assert str(err.value) == "CONTRACT_MISMATCH: D14 integer history contains nulls"


def test_missing_input_table_refuses_before_reads(pinned, monkeypatch):
    repo, snapshot = pinned["repository"], pinned["snapshot"]
    snapshot = dataclasses.replace(snapshot, table_versions={
        key: value for key, value in snapshot.table_versions.items() if key != "feature_panel"})
    monkeypatch.setattr(repo, "resolve", lambda _: snapshot)
    monkeypatch.setattr(repo, "scan", forbidden)
    with pytest.raises(errors.DataError) as err:
        d14_dataset.calculate_frame(repo, snapshot, calendar=None, role="implied_t1")
    assert err.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("count", (1, 2))
def test_table_default_bound_and_scan_overflow_are_enforced(pinned, monkeypatch, count):
    repo, snapshot = pinned["repository"], pinned["snapshot"]
    monkeypatch.setattr(repo, "scan_population_bound", lambda *args, **kw: 500001)
    queries = []
    monkeypatch.setattr(repo, "scan", lambda query, **kw: queries.append(query) or iter(()))
    assert list(d14_dataset.input_rows(repo, snapshot, "earnings_events", ("event_id",),
                                 start="2017-01-01", end="2027-01-01")) == []
    assert queries[0].max_result_rows == 500000
    monkeypatch.setattr(repo, "scan", lambda *args, **kw: iter([pa.table({"event_id": ["a", "b"][:count]})]))
    rows = d14_dataset.input_rows(repo, snapshot, "earnings_events", ("event_id",),
                            start="2017-01-01", end="2027-01-01", cap=1)
    if count == 1:
        assert list(rows) == [{"event_id": "a"}]
    else:
        with pytest.raises(errors.DataError) as err:
            list(rows)
        assert err.value.code == "RESULT_LIMIT_EXCEEDED"


def test_daily_scan_uses_twenty_million_row_cap(pinned, monkeypatch):
    repo = pinned["repository"]
    queries = []
    monkeypatch.setattr(repo, "scan_population_bound", lambda *args, **kw: 20000001)
    monkeypatch.setattr(repo, "scan", lambda query, **kw: queries.append(query) or iter(()))
    frame = pd.DataFrame({"ticker": ["A"], "entry_date": ["2024-01-01"],
                          "last_pre_print": ["2024-02-01"]})
    d14_dataset.daily_features(frame, repo, pinned["snapshot"])
    assert len(queries) == 1
    assert queries[0].max_result_rows == 20000000


@pytest.mark.parametrize("count", (10000, 10001))
def test_ticker_row_bound_is_inclusive(monkeypatch, count):
    monkeypatch.setattr(d14_dataset, "input_rows", lambda *args, **kw: iter(
        [{"ticker": "A", "date": "2024-01-01"}] * count))
    calls = []
    monkeypatch.setattr(d14_dataset.panel_math, "daily_state_lookup",
                        lambda rows, day: calls.append(len(rows)) or {})
    frame = pd.DataFrame({"ticker": ["A"], "entry_date": ["2024-01-01"],
                          "last_pre_print": ["2024-02-01"]})
    if count == 10000:
        assert len(d14_dataset.daily_features(frame, None, None)) == 1
        assert calls == [10000, 10000]
    else:
        with pytest.raises(errors.DataError) as err:
            d14_dataset.daily_features(frame, None, None)
        assert err.value.code == "RESULT_LIMIT_EXCEEDED"
        assert calls == []


@pytest.mark.parametrize("exit_spot", (0.0, -1.0))
def test_positive_entry_nonpositive_exit_has_missing_target(pinned, monkeypatch, exit_spot):
    original = d14_dataset.input_rows

    def rows(*args, **kwargs):
        for row in original(*args, **kwargs):
            if args[2] == "daily_market" and row["ticker"] == "AMC" \
                    and pd.Timestamp(row["date"]) >= pd.Timestamp("2024-06-25"):
                row = {**row, "spot": exit_spot}
            yield row

    monkeypatch.setattr(d14_dataset, "input_rows", rows)
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


def test_absent_snapshot_has_typed_refusal(pinned):
    with pytest.raises(errors.DataError) as err:
        build_d14_dataset("implied_t1", **{**pinned["inputs"], "snapshot_id": "missing"})
    assert err.value.code == "SNAPSHOT_NOT_FOUND"


@pytest.mark.parametrize("failure", ("catalog", "training"))
def test_cli_reports_typed_refusals_without_traceback(pinned, monkeypatch, tmp_path, capsys, failure):
    from engine.v2.models import training

    inputs = dict(pinned["inputs"])
    if failure == "catalog":
        inputs["catalog"] = tmp_path / "missing.sqlite"
        expected = "CONTRACT_MISMATCH: D14 catalog is unavailable"
    else:
        def refuse(*args, **kwargs):
            raise training.TrainingRefused((training.ReceiptIssue("fixture", "FUTURE_LABEL", "late"),))

        monkeypatch.setattr(training, "run_training_job", refuse)
        expected = "FUTURE_LABEL at fixture: late"
    args = ["--role", "implied_t1", "--out", str(tmp_path / "out"), "--plan-only"]
    for name, value in inputs.items():
        args += ["--" + name.replace("_", "-"), str(value)]
    assert phase5_d14_inputs.main(args) == 2
    captured = capsys.readouterr()
    assert captured.err == expected + "\n"
    assert captured.out == ""
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("past_limit", (False, True))
def test_calendar_byte_bound_is_inclusive(pinned, monkeypatch, past_limit):
    monkeypatch.setattr(d14_inputs, "MAX_CALENDAR_BYTES", pinned["calendar_ref"].byte_size - past_limit)
    if past_limit:
        monkeypatch.setattr(ArtifactStore, "read_verified", forbidden)
        with pytest.raises(errors.DataError) as err:
            build_d14_dataset("implied_t1", **pinned["inputs"])
        assert err.value.code == "RESULT_LIMIT_EXCEEDED"
    else:
        expected = legacy_frame(pinned, "implied_t1", monkeypatch)
        actual = build_d14_dataset("implied_t1", **pinned["inputs"])
        assert_frame_equal(actual[expected.columns], expected, check_exact=True)


@pytest.mark.parametrize("pinned,past_limit", (
    ("calendar_at_limit", False), ("calendar_past_limit", True)), indirect=["pinned"])
def test_unpatched_calendar_default_byte_bound(pinned, monkeypatch, past_limit):
    assert pinned["calendar_ref"].byte_size == 16 * 1024 * 1024 + past_limit
    if past_limit:
        monkeypatch.setattr(ArtifactStore, "read_verified", forbidden)
        with pytest.raises(errors.DataError) as err:
            build_d14_dataset("implied_t1", **pinned["inputs"])
        assert err.value.code == "RESULT_LIMIT_EXCEEDED"
    else:
        expected = legacy_frame(pinned, "implied_t1", monkeypatch)
        actual = build_d14_dataset("implied_t1", **pinned["inputs"])
        assert_frame_equal(actual[expected.columns], expected, check_exact=True)


def test_calendar_consumes_exact_verified_bytes(pinned, monkeypatch):
    original = ArtifactStore.read_verified

    def read(store, ref):
        data = original(store, ref)
        assert ref == artifact_reference(data, "parquet_fragment.v1")
        replacement = store.root / "replacement"
        replacement.write_bytes(b"changed after verification")
        replacement.replace(store.root / ref.storage_key)
        return data

    expected = legacy_frame(pinned, "implied_t1", monkeypatch)
    monkeypatch.setattr(ArtifactStore, "read_verified", read)
    actual = build_d14_dataset("implied_t1", **pinned["inputs"])
    assert_frame_equal(actual[expected.columns], expected, check_exact=True)
    with pytest.raises(errors.DataError) as err:
        build_d14_dataset("implied_t1", **pinned["inputs"])
    assert err.value.code == "OBJECT_CORRUPT"


@pytest.mark.parametrize("pinned", tuple({"calendar_bytes": data} for data in (
    b"", b"Price,Close\nTicker,^GSPC\nDate,\n", b"Price,Close\nTicker,^GSPC\nDate,\nbad-date,100\n",
    b"Price,Close\nTicker,^GSPC\nDate,\n2024-01-02,100\n,100\n")), indirect=True)
def test_malformed_calendar_date_refuses(pinned):
    with pytest.raises(errors.DataError) as err:
        build_d14_dataset("implied_t1", **pinned["inputs"])
    assert err.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("native", (False, True))
def test_cli_checks_both_no_fit_guards_before_reading(pinned, monkeypatch, tmp_path, native):
    from engine.models import no_fit as legacy_guard
    from engine.v2.models import no_fit as native_guard

    guard = native_guard if native else legacy_guard
    monkeypatch.setattr(phase5_d14_inputs, "build_dataset", forbidden)
    args = ["--role", "implied_t1", "--out", str(tmp_path / "out")]
    for name, value in pinned["inputs"].items():
        args += ["--" + name.replace("_", "-"), str(value)]
    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden):
        phase5_d14_inputs.main(args)
