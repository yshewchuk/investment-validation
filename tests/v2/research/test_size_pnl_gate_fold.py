"""EXP-188 Amendment 3: size_pnl_gate_fold on synthetic fixtures only (no real data, no real wiring, no network)."""
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments import size_pnl_gate_fold as fold

FEATS = ("f1", "f2")
SEED = fold.SEED


def _trades():
    # 692 events on consecutive days in 2023-2024, one Dec-30-2024 event whose exit is 2025-01-02,
    # and 7 events in 2025.
    base = pd.Timestamp("2023-01-01")
    events = [base + pd.Timedelta(days=i) for i in range(692)]
    events += [pd.Timestamp("2024-12-30")]
    events += [pd.Timestamp("2025-01-10") + pd.Timedelta(days=i) for i in range(7)]
    exits = [e + pd.Timedelta(days=5) for e in events]
    exits[692] = pd.Timestamp("2025-01-02")
    return pd.DataFrame({
        "event_id": np.arange(len(events)),
        "ticker": "T",
        "event_date": pd.to_datetime(events),
        "entry_date": pd.to_datetime(events),
        "exit_date": pd.to_datetime(exits),
    })


def _build(t):
    rng = np.random.default_rng(7)
    n = len(t)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    ret = 0.5 * f1 + 0.01 * rng.normal(size=n)
    return pd.DataFrame({"event_id": t["event_id"].to_numpy(), "event_date": t["event_date"].to_numpy(),
                         "f1": f1, "f2": f2, "ret": ret})


def _fit(X, y, seed):
    return HistGradientBoostingRegressor(max_iter=20, random_state=seed).fit(X, y)


def _train(trades, through=2024, seed=SEED, fit=_fit):
    return fold.train_fold(trades, through=through, build=_build, fit=fit, features=FEATS, target="ret",
                           seed=seed, threshold=fold.REGISTERED_THRESHOLD)


def _wiring(trades=None):
    t = _trades() if trades is None else trades
    return (lambda: t, _build, _fit, list(FEATS), "ret")


def test_cut_trades_drops_2025_events_and_dec30_with_2025_exit():
    kept, stats = fold.cut_trades(_trades(), 2024)
    assert stats == {"rows_in": 700, "rows_kept": 692,
                     "dropped_event_after_cutoff": 7, "dropped_exit_after_cutoff": 1}
    assert len(kept) == 692
    assert kept["event_date"].max() <= pd.Timestamp("2024-12-31")


def test_train_fold_happy_path(monkeypatch):
    trades = _trades()
    calls = []

    def counting_fit(X, y, seed):
        calls.append(seed)
        return _fit(X, y, seed)

    model, rec = _train(trades, fit=counting_fit)
    assert hasattr(model, "predict")
    assert calls == [SEED]
    assert rec["fit_count"] == 1
    assert rec["training_rows"] == 692
    assert pd.Timestamp(rec["event_date_max"]) <= pd.Timestamp("2024-12-31")
    assert pd.Timestamp(rec["exit_date_max"]) <= pd.Timestamp("2024-12-31")
    _, again = _train(trades)
    assert again["recipe_hash"] == rec["recipe_hash"]
    _, other_seed = _train(trades, seed=SEED + 1)
    assert other_seed["recipe_hash"] != rec["recipe_hash"]
    monkeypatch.setattr(fold, "MIN_TRAIN_ROWS", 100)  # 2023-only fold has fewer rows than the production minimum
    _, other_through = _train(trades, through=2023)
    assert other_through["recipe_hash"] != rec["recipe_hash"]


def test_train_fold_receipt_dates_come_from_complete_rows():
    trades = _trades()
    kept, _ = fold.cut_trades(trades, 2024)
    first, last = kept["event_date"].min(), kept["event_date"].max()

    def nan_at_edges(t):
        ds = _build(t)
        ds.loc[ds["event_date"].isin([first, last]), "f1"] = np.nan  # earliest and latest kept events are incomplete
        return ds

    _, rec = fold.train_fold(trades, through=2024, build=nan_at_edges, fit=_fit, features=FEATS, target="ret",
                             seed=SEED, threshold=fold.REGISTERED_THRESHOLD)
    complete_dates = kept.loc[~kept["event_date"].isin([first, last]), "event_date"]
    assert rec["training_rows"] == len(kept) - 2
    assert rec["event_date_min"] == complete_dates.min().date().isoformat()
    assert rec["event_date_max"] == complete_dates.max().date().isoformat()
    # exit_date_max is taken over the fitted rows only: the dropped latest event's exit must not count.
    assert rec["exit_date_max"] == (complete_dates.max() + pd.Timedelta(days=5)).date().isoformat()


def test_train_fold_refuses_dataset_dated_after_cutoff():
    def leaky_build(t):
        ds = _build(t)
        extra = ds.iloc[[0]].copy()
        extra["event_date"] = pd.Timestamp("2025-01-03")
        extra["event_id"] = 99999
        return pd.concat([ds, extra], ignore_index=True)

    with pytest.raises(ValueError):
        fold.train_fold(_trades(), through=2024, build=leaky_build, fit=_fit, features=FEATS, target="ret",
                        seed=SEED, threshold=fold.REGISTERED_THRESHOLD)


def test_train_fold_refuses_below_min_train_rows():
    with pytest.raises(ValueError):
        _train(_trades().iloc[:100])


def test_refuse_protected(tmp_path):
    protected = tmp_path / "data" / "models"
    protected.mkdir(parents=True)
    with pytest.raises(SystemExit, match="refused"):
        fold.refuse_protected(protected / "x.joblib", tmp_path)
    registry = tmp_path / "engine" / "models" / "registry.json"
    registry.parent.mkdir(parents=True)
    with pytest.raises(SystemExit, match="refused"):
        fold.refuse_protected(registry, tmp_path)
    existing = tmp_path / "exists.txt"
    existing.write_text("x")
    with pytest.raises(SystemExit, match="refused"):
        fold.refuse_protected(existing, tmp_path)
    fresh = tmp_path / "fresh.joblib"
    assert fold.refuse_protected(fresh, tmp_path) == fresh.resolve()


def test_refuse_protected_checks_every_root(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    (second / "data" / "models").mkdir(parents=True)
    with pytest.raises(SystemExit, match="refused"):
        fold.refuse_protected(second / "data" / "models" / "x.joblib", [first, second])
    fresh = tmp_path / "fresh.joblib"
    assert fold.refuse_protected(fresh, [first, second]) == fresh.resolve()


def test_file_sha256(tmp_path):
    f = tmp_path / "blob.bin"
    data = b"gate fold bytes" * 1000
    f.write_bytes(data)
    assert fold.file_sha256(f) == hashlib.sha256(data).hexdigest()
    link = tmp_path / "link.bin"
    link.symlink_to(f)
    with pytest.raises((ValueError, OSError)):  # O_NOFOLLOW refuses to open a symlink
        fold.file_sha256(link)
    with pytest.raises(ValueError):
        fold.file_sha256(tmp_path)  # a directory is not a regular file
    with pytest.raises(ValueError):
        fold.file_sha256(f, max_bytes=10)


def test_build_inputs_2025_only_no_labels_and_nan_kept():
    def nan_build(t):
        ds = _build(t)
        ds.loc[(ds["event_date"] >= pd.Timestamp("2025-01-01")).to_numpy().argmax(), "f1"] = np.nan
        return ds

    out, receipt = fold.build_inputs(_trades(), year=2025, build=nan_build, features=FEATS)
    assert list(out.columns) == ["event_id", "f1", "f2"]
    assert not set(out.columns) & set(fold.LABEL_COLUMNS)
    assert receipt["rows"] == 7 and len(out) == 7
    assert receipt["incomplete_rows"] == 1
    assert int(out["f1"].isna().sum()) == 1
    assert set(out["event_id"]) == set(range(693, 700))
    with pytest.raises(ValueError):
        fold.build_inputs(_trades(), year=2025, build=_build, features=("f1", "ret"))


def test_main_fit_end_to_end(tmp_path):
    out = tmp_path / "gate_fold_2024.joblib"
    stub = {}

    def recording_fit(X, y, seed):
        model = _fit(X, y, seed)
        stub["model"], stub["X"] = model, X
        return model

    wiring = (lambda: _trades(), _build, recording_fit, list(FEATS), "ret")
    rc = fold.main(["fit", "--train-through", "2024", "--out", str(out)], wiring=wiring, root=tmp_path)
    assert rc == 0
    receipt = json.loads((tmp_path / "gate_fold_2024.joblib.receipt.json").read_text())
    assert receipt["artifact_sha256"] == fold.file_sha256(out)
    art = joblib.load(out)
    assert {"model", "features", "threshold"} <= set(art)
    assert art["threshold"] == fold.REGISTERED_THRESHOLD
    assert art["train_through"] == 2024
    np.testing.assert_allclose(art["model"].predict(stub["X"]), stub["model"].predict(stub["X"]))


def test_main_fit_refuses_train_through_after_2024(tmp_path):
    out = tmp_path / "gate_fold_2025.joblib"
    with pytest.raises(SystemExit, match="refused: --train-through must be <= 2024"):
        fold.main(["fit", "--train-through", "2025", "--out", str(out)], wiring=_wiring(), root=tmp_path)
    assert list(tmp_path.iterdir()) == []


REGISTRY = Path(__file__).resolve().parents[3] / "engine" / "models" / "registry.json"


def _registry_entry(entry_id):
    data = json.loads(REGISTRY.read_text())
    if isinstance(data, dict):
        data = next(data[k] for k in ("entries", "models") if k in data)
    return next(e for e in data if e.get("id") == entry_id)


def test_real_wiring_matches_registered_entry():
    _, _, _, features, target = fold._real_wiring(2024)
    entry = _registry_entry(fold.REGISTERED_ID)
    assert target == "ret"
    assert len(features) == 50
    assert list(features) == list(entry["features"])
    assert fold.REGISTERED_THRESHOLD == entry["threshold"]
    assert fold.SEED == entry["seed"]


def test_main_fit_stdout_is_only_receipt_json(tmp_path, capsys):
    out = tmp_path / "gate.joblib"

    def noisy_build(t):
        print("engine analog matching 1/2")  # builder progress must not reach stdout
        return _build(t)

    wiring = (lambda: _trades(), noisy_build, _fit, list(FEATS), "ret")
    rc = fold.main(["fit", "--train-through", "2024", "--out", str(out)], wiring=wiring, root=tmp_path)
    assert rc == 0
    captured = capsys.readouterr()
    receipt = json.loads((tmp_path / "gate.joblib.receipt.json").read_text())
    assert json.loads(captured.out) == receipt
    assert "engine analog matching" in captured.err


def test_train_fold_exit_date_uses_fill_alpha_half_rows_only():
    base = _trades()
    half = base.assign(fill_alpha=0.5)
    late = base.assign(fill_alpha=0.9, exit_date=base["event_date"] + pd.Timedelta(days=9))
    trades = pd.concat([half, late], ignore_index=True)
    kept, _ = fold.cut_trades(trades, 2024)
    is_half = np.isclose(kept["fill_alpha"].to_numpy(dtype=float), 0.5)
    half_max = kept.loc[is_half, "exit_date"].max()
    assert kept.loc[~is_half, "exit_date"].max() > half_max  # the 0.9 rows really do exit later
    _, rec = _train(trades)
    assert rec["exit_date_max"] == half_max.date().isoformat()


def test_main_inputs_end_to_end(tmp_path):
    out = tmp_path / "inputs_2025.parquet"
    rc = fold.main(["inputs", "--year", "2025", "--out", str(out)], wiring=_wiring(), root=tmp_path)
    assert rc == 0
    assert len(pd.read_parquet(out)) == 7
    receipt = json.loads((tmp_path / "inputs_2025.parquet.receipt.json").read_text())
    assert receipt["inputs_sha256"] == fold.file_sha256(out)


def test_publish_pair_receipt_replace_failure_leaves_nothing(tmp_path, monkeypatch):
    out = tmp_path / "a.bin"
    rec = tmp_path / "a.bin.receipt.json"
    real_replace = os.replace

    def failing_replace(src, dst):
        if os.fspath(dst) == os.fspath(rec):
            raise OSError("simulated receipt replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(fold.os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated"):
        fold._publish_pair(out, rec, lambda p: Path(p).write_bytes(b"payload"), {"kind": "t"}, "sha")
    assert not out.exists()
    assert not rec.exists()
    assert list(tmp_path.iterdir()) == []


def test_publish_pair_success_leaves_both_and_no_tmp(tmp_path, capsys):
    out = tmp_path / "a.bin"
    rec = tmp_path / "a.bin.receipt.json"
    text = fold._publish_pair(out, rec, lambda p: Path(p).write_bytes(b"payload"), {"kind": "t"}, "sha")
    assert out.read_bytes() == b"payload"
    receipt = json.loads(rec.read_text())
    assert receipt["sha"] == fold.file_sha256(out)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.bin", "a.bin.receipt.json"]
    assert json.loads(text) == receipt
    assert capsys.readouterr().out == ""  # the caller prints the receipt, not the publisher


def test_main_fit_refuses_out_under_data_models(tmp_path):
    out = tmp_path / "data" / "models" / "x.joblib"
    with pytest.raises(SystemExit, match="refused"):
        fold.main(["fit", "--train-through", "2024", "--out", str(out)], wiring=_wiring(), root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_main_fit_creates_nested_missing_output_dir(tmp_path):
    out = tmp_path / "a" / "b" / "gate.joblib"
    rc = fold.main(["fit", "--train-through", "2024", "--out", str(out)], wiring=_wiring(), root=tmp_path)
    assert rc == 0
    assert out.exists()
    assert (tmp_path / "a" / "b" / "gate.joblib.receipt.json").exists()


def test_main_fit_refuses_receipt_equal_to_out_and_writes_nothing(tmp_path):
    out = tmp_path / "nested" / "gate.joblib"
    with pytest.raises(SystemExit, match="refused"):
        fold.main(["fit", "--train-through", "2024", "--out", str(out), "--receipt", str(out)],
                  wiring=_wiring(), root=tmp_path)
    assert list(tmp_path.iterdir()) == []  # refused before any directory is created or file written
