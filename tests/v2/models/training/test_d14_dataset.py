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
from engine.data.schemas import coerce
from engine.models.training import implied_t1
from engine.v2.data import errors
from engine.v2.data.computed_moves import native_trading_calendar
from engine.v2.data.reference_catalog import ReferenceInput, insert_reference_inputs
from engine.v2.data.repository import Repository
from engine.v2.models.training import (
    RecipeKey,
    TrainingRefused,
    current_recipes,
    dataset_fingerprint,
    fold_receipts,
    plan_folds,
    prepare_dataset,
    receipt_issues,
    run_training_job,
)
from engine.v2.models.training.d14_dataset import HISTORY_COLUMNS, build_d14_dataset
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
    publish_bytes,
)
from tools import phase5_d14_inputs


def _inputs():
    days = projected_trading_days("2023-10-01", "2025-03-31")
    dates = ("2024-01-02", "2024-02-01", "2024-07-05", "2024-12-02", "2025-01-02")
    events, panel, daily = [], [], []
    tickers = ("AMC", "BMO", "GAP", "NULL", "ZERO", "NEG", "NOPANEL", "NODAILY", "NOSESSION")
    for ti, ticker in enumerate(tickers):
        for ei, text in enumerate(dates):
            day = pd.Timestamp(text)
            events.append(dict(event_id=f"{ti:02}-{ei}", ticker=ticker, event_date=day,
                               year=day.year, session=None if ticker == "NOSESSION" else
                               "BMO" if ticker == "BMO" else "AMC", src_orats=True,
                               src_oquants=False, src_nasdaq=False, src_yfinance=False,
                               date_agree=True, date_conflict=False))
            if ticker != "NOPANEL":
                panel.append(dict(ticker=ticker, date=day, k=12, year=day.year,
                                  **{c: 1.5 + ti / 10 + ei / 50 for c in HISTORY_COLUMNS
                                     if c not in ("n_prior", "signed_streak")},
                                  n_prior=12, signed_streak=-2, or_iv30=9999.0))
        if ticker == "NODAILY":
            continue
        for di, day in enumerate(days):
            if ticker == "GAP" and di % 4 == 0:
                continue
            row = dict(ticker=ticker, date=day, year=day.year,
                       src_iv=None if di % 11 == 0 else "orats")
            row.update({field: 3 + di / 100 + ti / 50 for field in implied_t1.DAILY_STATE_COLUMNS
                        if "_d" not in field})
            row["implied_move"] = row.pop("im")
            row["spot"] = (0.0 if ticker == "ZERO" else -1.0 if ticker == "NEG" else 100 + di / 4)
            if ticker == "NULL":
                row["implied_move"] = None
            daily.append(row)
    return pd.DataFrame(events).sort_values(["year", "event_id"]).reset_index(drop=True), \
        pd.DataFrame(panel), pd.DataFrame(daily).sort_values(["ticker", "date"]), days


@pytest.fixture
def pinned(tmp_path, request):
    events, panel, daily, days = _inputs()
    case = getattr(request, "param", "normal")
    if case == "empty":
        panel = panel.iloc[:0]
    elif case == "all_null":
        panel["ema12_prior_move"] = np.nan
    elif case == "boundary":
        for year in (2016, 2027):
            event = events.iloc[0].to_dict()
            event.update(event_id=f"boundary-{year}", event_date=pd.Timestamp(f"{year}-07-01"), year=year)
            events = pd.concat([events, pd.DataFrame([event])], ignore_index=True)
    elif case == "blank_session":
        events.loc[0, "session"] = ""
    conn, clock, store = catalog_and_store(tmp_path)
    frames = {"earnings_events": events, "feature_panel": panel, "daily_market": daily}
    contracts = {table: contract_for(table) for table in frames}
    records = {}
    for table, frame in frames.items():
        contract = contracts[table]
        records[table] = []
        parts = ([] if frame.empty else [(None, frame)]) if table == "feature_panel" \
            else frame.groupby("year", sort=True)
        for year, part in parts:
            part = part.sort_values(list(contract.primary_key))
            records[table].append(publish_and_inspect(
                store, contract, contract_ref_for(contract), part.astype(object).where(pd.notna(part), None).to_dict("records"),
                "all" if year is None else str(year)))
    snapshot = commit_tables(conn, clock, records, contracts, store=store)
    # Original yfinance CSV's three header rows, then dates. Exact immutable object.
    obj = publish_bytes(store, ("Price,Close\nTicker,^GSPC\nDate,\n" +
                               "\n".join(f"{d.date()},100" for d in days) + "\n").encode())
    ref = ReferenceInput(kind="model_registry" if case == "no_calendar" else "calendar", legacy_path="data/raw/yf/index.csv",
                         object_id=obj.object_id, content_hash=obj.content_hash, byte_size=obj.byte_size)
    if case != "no_refs":
        insert_reference_inputs(conn, "r1", (ref,))
    conn.commit()
    yield dict(repository=Repository(conn, store), snapshot=snapshot, events=events,
               panel=panel, daily=daily, days=days, conn=conn, calendar_ref=ref,
               records=records, contracts=contracts, clock=clock, store=store,
               inputs=dict(catalog=tmp_path / "catalog.sqlite", artifact_root=store.root,
                           snapshot_id=snapshot.snapshot_id, receipt_id="r1"))
    conn.close()


def _legacy(pinned, role, monkeypatch):
    calendar = TradingCalendar(list(pinned["days"]) + list(projected_trading_days(
        pinned["days"][-1], pinned["days"][-1] + pd.Timedelta(days=400))))
    monkeypatch.setattr(implied_t1, "trading_calendar", lambda: calendar)
    def read(name, **kwargs):
        frame = pinned["events"] if name == "earnings_events" else pinned["daily"]
        return coerce(frame[frame.year.isin(kwargs["years"])], name, only=kwargs["columns"])

    monkeypatch.setattr(tier4.store, "read_table", read)
    factory = tier4.im_t1_feature_model if role == "implied_t1" else tier4.runup_move_feature_model
    return factory().prepare(pinned["panel"]).rename(columns={"date": "event_date"})


def _recipe(role):
    return current_recipes()[RecipeKey(role, "*", "tier4_monthly")]


def _forbidden(*args, **kwargs):
    raise AssertionError("legacy reader reached")


@pytest.mark.parametrize("role", ("implied_t1", "runup_move"))
def test_exact_frame_masks_folds_receipts_and_single_scan(pinned, role, monkeypatch):
    expected = _legacy(pinned, role, monkeypatch)
    monkeypatch.setattr(tier4.store, "read_table", _forbidden)
    monkeypatch.setattr("engine.features.load_panel", _forbidden)
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
    build_d14_dataset(repo, pinned["snapshot"], calendar=calendar, role=role)
    counters = repo.read_counters.snapshot()
    assert counters["daily_market"]["scans"] == 1


@pytest.mark.parametrize("role", ("implied_t1", "runup_move"))
def test_predictions_receipts_and_resume_identity(pinned, role, monkeypatch, tmp_path):
    legacy = _legacy(pinned, role, monkeypatch)
    native = phase5_d14_inputs.build_dataset(role, **pinned["inputs"])
    recipe = _recipe(role)
    recipe = dataclasses.replace(recipe, folds=dataclasses.replace(recipe.folds, min_train_rows=2),
                                 estimator=dataclasses.replace(recipe.estimator, params={
                                     **recipe.estimator.params, "max_iter": 5, "min_samples_leaf": 2,
                                     "early_stopping": False}))
    out1, out2 = tmp_path / "legacy", tmp_path / "native"
    run_training_job(recipe, legacy, out1)
    run_training_job(recipe, native, out2, input_identity=native.attrs["training_inputs"])
    for path in out1.glob("folds/*/predictions.parquet"):
        assert_frame_equal(pd.read_parquet(path), pd.read_parquet(out2 / path.relative_to(out1)),
                           check_exact=True)
    resumed = run_training_job(recipe, native, out2, input_identity=native.attrs["training_inputs"])
    assert resumed.count("resumed") == len(resumed.outcomes)
    with pytest.raises(TrainingRefused, match="RESUME_MISMATCH"):
        run_training_job(recipe, native, out2,
                         input_identity={**native.attrs["training_inputs"], "receipt_id": "changed"})
    assert json.loads((out2 / "job.json").read_text())["input_identity"] == native.attrs["training_inputs"]


def test_cli_requires_pins_before_legacy_reads(monkeypatch, tmp_path):
    monkeypatch.setattr("engine.features.load_panel", _forbidden)
    with pytest.raises(SystemExit) as err:
        phase5_d14_inputs.main(["--role", "runup_move", "--out", str(tmp_path / "out")])
    assert err.value.code == 2


def test_wrong_receipt_refuses(pinned):
    with pytest.raises(errors.DataError, match="receipt") as err:
        phase5_d14_inputs.build_dataset("implied_t1", **{**pinned["inputs"], "receipt_id": "missing"})
    assert err.value.code == "IDENTITY_CONFLICT"


@pytest.mark.parametrize("pinned", ("empty", "all_null"), indirect=True)
def test_missing_history_preserves_numeric_dtypes_and_identity(pinned, monkeypatch):
    oracle = _legacy(pinned, "implied_t1", monkeypatch)
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
    original = d14_dataset._rows

    def rows(*args, **kwargs):
        for row in original(*args, **kwargs):
            if args[2] == "daily_market" and row["ticker"] == "AMC" \
                    and pd.Timestamp(row["date"]) > event[cutoff]:
                row = {**row, "implied_move": 900.0, "spot": 900.0, "iv30": 900.0}
            yield row

    monkeypatch.setattr(d14_dataset, "_rows", rows)
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


def test_native_cli_plan_and_resume_with_legacy_readers_disabled(pinned, monkeypatch, tmp_path):
    monkeypatch.setattr("engine.features.load_panel", _forbidden)
    monkeypatch.setattr(tier4.store, "read_table", _forbidden)
    args = ["--role", "implied_t1", "--out", str(tmp_path / "planned"), "--plan-only"]
    for name, value in pinned["inputs"].items():
        args += ["--" + name.replace("_", "-"), str(value)]
    assert phase5_d14_inputs.main(args) == 0
    assert phase5_d14_inputs.main(args) == 0
    job = json.loads((tmp_path / "planned" / "job.json").read_text())
    assert job["input_identity"] == {
        "schema_version": "d14.snapshot_dataset.v1", "receipt_id": "r1",
        "snapshot_id": pinned["snapshot"].snapshot_id,
        "calendar_hash": pinned["calendar_ref"].content_hash,
        "table_versions": {table: pinned["snapshot"].table_versions[table].dataset_version_id
                           for table in ("earnings_events", "feature_panel", "daily_market")}}


@pytest.mark.parametrize("root", ("data", "artifact"))
def test_cli_respects_configured_data_root(pinned, monkeypatch, tmp_path, root):
    from engine import paths

    alternate = tmp_path / "alternate-root" / "data"
    monkeypatch.setattr(paths, "DATA", alternate)
    unsafe = (alternate if root == "data" else pinned["inputs"]["artifact_root"]) / "unsafe"
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

    original = d14_dataset._scan
    original_rows = d14_dataset._rows
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

    monkeypatch.setattr(d14_dataset, "_scan", scan)
    monkeypatch.setattr(d14_dataset, "_rows", rows)
    with pytest.raises(errors.DataError, match="no eligible decision") as err:
        phase5_d14_inputs.build_dataset("implied_t1", **pinned["inputs"])
    assert err.value.code == "CONTRACT_MISMATCH"
    assert calls == ["earnings_events"]


def test_wrong_snapshot_ref_refuses_before_reads(pinned):
    calendar = native_trading_calendar({"index": pd.DataFrame({"date": pinned["days"]})})
    forged = dataclasses.replace(pinned["snapshot"], calendar_version="different")
    with pytest.raises(errors.DataError, match="catalog identity") as err:
        build_d14_dataset(pinned["repository"], forged, calendar=calendar, role="implied_t1")
    assert err.value.code == "IDENTITY_CONFLICT"
