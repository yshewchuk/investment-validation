"""Native snapshot readers against untouched legacy D14 builders, using synthetic data."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from engine.calendar import TradingCalendar, projected_trading_days
from engine.data.features import tier4
from engine.v2.data import errors
from engine.v2.data.computed_moves import native_trading_calendar
from engine.v2.models.training import (
    RecipeKey,
    TrainingRefused,
    build_d14_dataset,
    current_recipes,
    d14_inputs,
    dataset_fingerprint,
    fold_receipts,
    plan_folds,
    prepare_dataset,
    receipt_issues,
    run_training_job,
)
from engine.v2.models.training.d14_dataset import calculate_frame
from tests.v2.models.training.d14_support import forbidden, legacy_frame
from tests.v2.models.training.d14_support import pinned as pinned
from tools import phase5_d14_inputs


def _recipe(role):
    return current_recipes()[RecipeKey(role, "*", "tier4_monthly")]


@pytest.mark.parametrize("role", ("implied_t1", "runup_move"))
def test_exact_frame_masks_folds_receipts_and_single_scan(pinned, role, monkeypatch):
    expected = legacy_frame(pinned, role, monkeypatch)
    monkeypatch.setattr(tier4.store, "read_table", forbidden)
    monkeypatch.setattr("engine.features.load_panel", forbidden)
    actual = phase5_d14_inputs.build_dataset(role, **pinned["inputs"])
    assert_frame_equal(actual[expected.columns], expected, check_exact=True)
    left, right = [prepare_dataset(_recipe(role), frame) for frame in (expected, actual)]
    np.testing.assert_array_equal(left.features_complete, right.features_complete)
    np.testing.assert_array_equal(left.target_finite, right.target_finite)
    assert dataset_fingerprint(_recipe(role), left) == dataset_fingerprint(_recipe(role), right)
    for lf, rf in zip(plan_folds(_recipe(role), left), plan_folds(_recipe(role), right), strict=True):
        np.testing.assert_array_equal(lf.train, rf.train)
        np.testing.assert_array_equal(lf.test, rf.test)
        a = fold_receipts(_recipe(role), left, lf, dataset_fp="same")
        b = fold_receipts(_recipe(role), right, rf, dataset_fp="same")
        assert a == b
        assert receipt_issues(_recipe(role), *b) == ()
    repo = pinned["repository"]
    calendar = native_trading_calendar({"index": pd.DataFrame({"date": pinned["days"]})})
    calculated = calculate_frame(repo, pinned["snapshot"], calendar=calendar, role=role)
    assert "training_inputs" not in calculated.attrs
    counters = repo.read_counters.snapshot()
    assert counters["daily_market"]["scans"] == 1


@pytest.mark.parametrize("role", ("implied_t1", "runup_move"))
def test_predictions_receipts_and_resume_identity(pinned, role, monkeypatch, tmp_path):
    legacy = legacy_frame(pinned, role, monkeypatch)
    native = phase5_d14_inputs.build_dataset(role, **pinned["inputs"])
    recipe = _recipe(role)
    recipe = dataclasses.replace(recipe, folds=dataclasses.replace(recipe.folds, min_train_rows=2),
                                 estimator=dataclasses.replace(recipe.estimator, params={
                                     **recipe.estimator.params, "max_iter": 5, "min_samples_leaf": 2,
                                     "early_stopping": False}))
    out1, out2 = tmp_path / "legacy", tmp_path / "native"
    run_training_job(recipe, legacy, out1)
    run_training_job(recipe, native, out2, input_identity=native.attrs["training_inputs"])
    predictions = list(out1.glob("folds/*/predictions.parquet"))
    assert predictions
    assert {p.relative_to(out1) for p in predictions} == {
        p.relative_to(out2) for p in out2.glob("folds/*/predictions.parquet")}
    for path in predictions:
        assert_frame_equal(pd.read_parquet(path), pd.read_parquet(out2 / path.relative_to(out1)),
                           check_exact=True)
    resumed = run_training_job(recipe, native, out2, input_identity=native.attrs["training_inputs"])
    assert resumed.count("resumed") == len(resumed.outcomes)
    for field in native.attrs["training_inputs"]:
        with pytest.raises(TrainingRefused, match="RESUME_MISMATCH"):
            run_training_job(recipe, native, out2,
                             input_identity={**native.attrs["training_inputs"], field: "changed"})
    assert json.loads((out2 / "job.json").read_text())["input_identity"] == native.attrs["training_inputs"]


@pytest.mark.parametrize("missing", ("catalog", "artifact_root", "snapshot_id", "receipt_id"))
def test_cli_requires_each_pin_before_reads(pinned, monkeypatch, tmp_path, missing):
    monkeypatch.setattr("engine.features.load_panel", forbidden)
    monkeypatch.setattr(phase5_d14_inputs, "build_dataset", forbidden)
    args = ["--role", "runup_move", "--out", str(tmp_path / "out")]
    for name, value in pinned["inputs"].items():
        if name != missing:
            args += ["--" + name.replace("_", "-"), str(value)]
    with pytest.raises(SystemExit) as err:
        phase5_d14_inputs.main(args)
    assert err.value.code == 2


def test_wrong_receipt_refuses(pinned):
    with pytest.raises(errors.DataError, match="receipt") as err:
        phase5_d14_inputs.build_dataset("implied_t1", **{**pinned["inputs"], "receipt_id": "missing"})
    assert err.value.code == "IDENTITY_CONFLICT"


@pytest.mark.parametrize("pinned", ("empty", "all_null"), indirect=True)
def test_missing_history_preserves_numeric_dtypes_and_identity(pinned, monkeypatch):
    oracle = legacy_frame(pinned, "implied_t1", monkeypatch)
    actual = phase5_d14_inputs.build_dataset("implied_t1", **pinned["inputs"])
    assert_frame_equal(actual[oracle.columns], oracle, check_exact=True)
    recipe = _recipe("implied_t1")
    assert dataset_fingerprint(recipe, prepare_dataset(recipe, actual)) == \
        dataset_fingerprint(recipe, prepare_dataset(recipe, oracle))


@pytest.mark.parametrize("cutoff", ("entry_date", "last_pre_print"))
def test_future_quotes_do_not_leak_into_features_or_labels(pinned, cutoff, monkeypatch):
    from engine.v2.models.training import d14_dataset

    baseline = phase5_d14_inputs.build_dataset("runup_move", **pinned["inputs"])
    event = baseline[baseline.ticker == "AMC"].iloc[0]
    original = d14_dataset.input_rows

    def rows(*args, **kwargs):
        for row in original(*args, **kwargs):
            if args[2] == "daily_market" and row["ticker"] == "AMC" \
                    and pd.Timestamp(row["date"]) > event[cutoff]:
                row = {**row, "implied_move": 900.0, "spot": 900.0, "iv30": 900.0}
            yield row

    monkeypatch.setattr(d14_dataset, "input_rows", rows)
    changed = phase5_d14_inputs.build_dataset("runup_move", **pinned["inputs"])
    columns = list(_recipe("runup_move").features)
    if cutoff == "last_pre_print":
        columns += ["im_t1", "runup_abs_move_d14"]
    assert_frame_equal(changed.loc[[event.name], columns], baseline.loc[[event.name], columns],
                       check_exact=True)
    if cutoff == "entry_date":
        assert changed.loc[event.name, "runup_abs_move_d14"] != event.runup_abs_move_d14


def test_late_training_labels_are_still_refused(pinned, tmp_path):
    frame = phase5_d14_inputs.build_dataset("implied_t1", **pinned["inputs"])
    frame.loc[frame.ticker == "AMC", "last_pre_print"] = pd.Timestamp("2030-01-01")
    with pytest.raises(TrainingRefused, match="FUTURE_LABEL"):
        run_training_job(_recipe("implied_t1"), frame, tmp_path / "late", plan_only=True)


def test_dates_missing_masks_and_nonpositive_prices(pinned):
    frame = phase5_d14_inputs.build_dataset("runup_move", **pinned["inputs"])
    assert set(frame.ticker) == set(pinned["events"].ticker) - {"NOSESSION"}
    july = frame[frame.event_date == pd.Timestamp("2024-07-05")].set_index("ticker")
    assert july.loc["AMC", "last_pre_print"] == pd.Timestamp("2024-07-05")
    assert july.loc["BMO", "last_pre_print"] == pd.Timestamp("2024-07-03")
    assert (frame.days_before_print == 14).all()
    assert (frame.days_to_print > 14).all()
    assert frame[frame.ticker.isin(["ZERO", "NEG", "NODAILY"])].runup_abs_move_d14.isna().all()
    assert frame[frame.ticker == "NOPANEL"].n_prior.isna().all()
    assert frame[frame.ticker == "NULL"].im_t1.isna().all()
    assert "NOSESSION" not in set(frame.ticker)


@pytest.mark.parametrize("last", ("2018-11-30", "2024-12-31"))
def test_pinned_calendar_retains_exact_legacy_projection(last):
    observed = pd.DatetimeIndex([pd.Timestamp(last)])
    actual = native_trading_calendar({"index": pd.DataFrame({"date": observed})})
    expected = [*observed, *projected_trading_days(last, observed[-1] + pd.Timedelta(days=400))]
    assert list(actual.days) == expected


def build_with_projection_oracle(pinned, monkeypatch):
    """Observe the public calculation boundary without bypassing verified loading."""
    observed = pinned["days"]
    expected = TradingCalendar([*observed, *projected_trading_days(
        observed[-1], observed[-1] + pd.Timedelta(days=400))])
    original = d14_inputs.calculate_frame
    calls = []

    def calculate(repository, snapshot, *, calendar, role):
        assert list(calendar.days) == list(expected.days), "calendar projection differs"
        assert calendar.observed_through == observed[-1]
        frame = original(repository, snapshot, calendar=calendar, role=role)
        calls.append(repository.read_counters.snapshot())
        return frame

    monkeypatch.setattr(d14_inputs, "calculate_frame", calculate)
    frame = build_d14_dataset("implied_t1", **pinned["inputs"])
    assert len(calls) == 1
    assert calls[0]["daily_market"]["scans"] == 1
    return frame


def test_verified_builder_retains_exact_legacy_projection(pinned, monkeypatch):
    expected = legacy_frame(pinned, "implied_t1", monkeypatch)
    # This fixture's 401st day is a trading session, making the +1 error observable.
    last = pinned["days"][-1]
    assert projected_trading_days(last, last + pd.Timedelta(days=401))[-1] == last + pd.Timedelta(days=401)
    actual = build_with_projection_oracle(pinned, monkeypatch)
    assert_frame_equal(actual[expected.columns], expected, check_exact=True)
    corrupted = actual[expected.columns].copy()
    corrupted.loc[0, "spot"] += 1
    with pytest.raises(AssertionError, match="spot"):
        assert_frame_equal(corrupted, expected, check_exact=True)


@pytest.mark.parametrize("extension", (0, 401))
def test_public_projection_oracle_rejects_planted_extension_defects(pinned, monkeypatch, extension):
    original = d14_inputs.computed_moves.native_trading_calendar

    def calendar(frames, **kwargs):
        return original(frames, **{**kwargs, "extend_days": extension})

    monkeypatch.setattr(d14_inputs.computed_moves, "native_trading_calendar", calendar)
    with pytest.raises(AssertionError, match="calendar projection differs"):
        build_with_projection_oracle(pinned, monkeypatch)


def test_native_cli_plan_is_idempotent_with_legacy_readers_disabled(pinned, monkeypatch, tmp_path):
    from engine.v2.models import training

    monkeypatch.setattr("engine.features.load_panel", forbidden)
    monkeypatch.setattr(tier4.store, "read_table", forbidden)
    original = training.run_training_job
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs["plan_only"])
        return original(*args, **kwargs, fit=forbidden)

    monkeypatch.setattr(training, "run_training_job", run)
    args = ["--role", "implied_t1", "--out", str(tmp_path / "planned"), "--plan-only"]
    for name, value in pinned["inputs"].items():
        args += ["--" + name.replace("_", "-"), str(value)]
    assert phase5_d14_inputs.main(args) == 0
    identity = (tmp_path / "planned" / "job.json").read_text()
    assert phase5_d14_inputs.main(args) == 0
    assert (tmp_path / "planned" / "job.json").read_text() == identity
    job = json.loads((tmp_path / "planned" / "job.json").read_text())
    assert calls == [True, True]
    assert job["plan_only"] is True
    assert job["folds"]
    assert not list((tmp_path / "planned").glob("folds/*/predictions.parquet"))
    assert job["input_identity"] == {
        "schema_version": "d14.snapshot_dataset.v1", "receipt_id": "r1",
        "snapshot_id": pinned["snapshot"].snapshot_id,
        "calendar_hash": pinned["calendar_ref"].content_hash,
        "table_versions": {table: pinned["snapshot"].table_versions[table].dataset_version_id
                           for table in ("earnings_events", "feature_panel", "daily_market")}}


@pytest.mark.parametrize("root", ("data", "artifact"))
@pytest.mark.parametrize("child", (False, True))
def test_cli_respects_configured_data_root(pinned, monkeypatch, tmp_path, root, child):
    from engine import paths

    alternate = tmp_path / "alternate-root" / "data"
    monkeypatch.setattr(paths, "DATA", alternate)
    unsafe = alternate if root == "data" else pinned["inputs"]["artifact_root"]
    if child:
        unsafe /= "unsafe"
    monkeypatch.setattr(phase5_d14_inputs, "build_dataset", forbidden)
    args = ["--role", "implied_t1", "--out", str(unsafe), "--plan-only"]
    for name, value in pinned["inputs"].items():
        args += ["--" + name.replace("_", "-"), str(value)]
    with pytest.raises(SystemExit) as err:
        phase5_d14_inputs.main(args)
    assert err.value.code == 2
    assert not alternate.exists()


@pytest.mark.parametrize("case", ("no_events", "no_session", "no_calendar_history"))
def test_no_decisions_refuse_before_history_or_daily_reads(pinned, case, monkeypatch):
    from engine.v2.models.training import d14_dataset

    original = d14_dataset.input_frame
    original_rows = d14_dataset.input_rows
    calls = []

    def rows(*args, **kwargs):
        assert args[2] == "earnings_events"
        return original_rows(*args, **kwargs)

    def scan(*args, **kwargs):
        calls.append(args[2])
        assert args[2] == "earnings_events"
        events = original(*args, **kwargs)
        if case == "no_events":
            events = events.iloc[:0]
        elif case == "no_session":
            events["session"] = None
        else:
            events["event_date"] = pd.Timestamp("2000-01-01")
        return events

    monkeypatch.setattr(d14_dataset, "input_frame", scan)
    monkeypatch.setattr(d14_dataset, "input_rows", rows)
    with pytest.raises(errors.DataError, match="no eligible decision") as err:
        phase5_d14_inputs.build_dataset("implied_t1", **pinned["inputs"])
    assert err.value.code == "CONTRACT_MISMATCH"
    assert calls == ["earnings_events"]


def test_wrong_snapshot_ref_refuses_before_reads(pinned, monkeypatch):
    calendar = native_trading_calendar({"index": pd.DataFrame({"date": pinned["days"]})})
    forged = dataclasses.replace(pinned["snapshot"], calendar_version="different")
    monkeypatch.setattr(pinned["repository"], "scan", forbidden)
    with pytest.raises(errors.DataError, match="catalog identity") as err:
        calculate_frame(pinned["repository"], forged, calendar=calendar, role="implied_t1")
    assert err.value.code == "IDENTITY_CONFLICT"
