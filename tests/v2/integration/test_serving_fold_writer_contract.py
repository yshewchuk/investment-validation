"""Serving-fold writers must produce artifacts that public joblib can mmap."""
# packages: engine.v2.features, engine.v2.models
from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from engine.data.features import tier4
from tools import prepare_phase4_tier4_caches as prepare

FOLD = pd.Timestamp("2026-09-01")
SNAPSHOT = "a" * 64


@pytest.fixture
def model():
    return tier4.FeatureModel(
        model_id="uncompressed-writer-test",
        produces="pred_iv_crush_30",
        features=("x", "z"),
        target="y",
        fit=lambda *_args: None,
        prepare=lambda panel: panel,
        interval_floor=None,
    )


@pytest.fixture
def served(model):
    values = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [4.0, -1.0]])
    estimator = LinearRegression().fit(values, values @ np.array([1.5, -0.75]) + 0.25)
    return tier4.ServingModel(
        estimator=estimator,
        model_id=model.model_id,
        fold_start=FOLD,
        tier3_snapshot=SNAPSHOT,
        features=model.features,
        interval_floor=model.interval_floor,
        pool_pred=np.linspace(-2.0, 3.0, 512),
        pool_res=np.linspace(-0.75, 0.25, 512),
    )


def _metadata(model):
    return {
        "model_id": model.model_id,
        "fold_start": str(FOLD.date()),
        "tier3_snapshot": SNAPSHOT,
        "features": list(model.features),
    }


def _assert_mappable_roundtrip(path, served, metadata):
    # Use joblib's public reader rather than its private compression detector.
    # A successfully loaded ndarray alone is insufficient: compressed files
    # load successfully but warn and silently ignore mmap_mode.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stored = joblib.load(path, mmap_mode="r")
    assert not caught, [str(item.message) for item in caught]
    assert set(stored) == set(metadata) | {"estimator", "pool_pred", "pool_res"}
    assert {key: stored[key] for key in metadata} == metadata

    for name in ("pool_pred", "pool_res"):
        actual = stored[name]
        assert isinstance(actual, np.memmap), name
        assert actual.mode == "r"
        assert not actual.flags.writeable
        assert Path(actual.filename) == path
        np.testing.assert_array_equal(actual, getattr(served, name))

    restored = stored["estimator"]
    assert type(restored) is type(served.estimator)
    assert isinstance(restored.coef_, np.memmap)
    assert joblib.hash(restored, coerce_mmap=True) == joblib.hash(served.estimator)
    samples = np.array([[0.0, 0.0], [2.5, -1.0], [-3.0, 4.0]])
    np.testing.assert_array_equal(restored.predict(samples), served.estimator.predict(samples))


@pytest.mark.parametrize("writer", ["serving_model", "upgrade_one", "build_one"])
@pytest.mark.parametrize("suffix", [".joblib", ".gz", ".xz"])
def test_serving_fold_writers_are_explicitly_uncompressed_and_mappable(
    tmp_path, monkeypatch, model, served, writer, suffix
):
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
    canonical_path = tier4._serving_path(model.model_id, FOLD, SNAPSHOT)
    path = canonical_path.with_suffix(suffix)
    if suffix != ".joblib":
        # Retain the real default-path case, then exercise an override whose
        # suffix would enable joblib compression even with compress=0.
        monkeypatch.setattr(tier4, "_serving_path", lambda *_args: path)
    metadata = _metadata(model)
    panel = pd.DataFrame({"x": [1.0], "z": [2.0], "y": [-0.5]})

    if writer == "upgrade_one":
        # Backfilling a legacy compressed file must install an uncompressed
        # replacement and preserve unrelated metadata, not just core identity.
        metadata["legacy_metadata"] = {"source": "fixture", "revision": 7}
        joblib.dump({**metadata, "estimator": served.estimator}, path, compress=3)
        path.chmod(0o640)
        target = prepare.CacheTarget(path, FOLD, prepare._sha256(path))

    dump = Mock(wraps=joblib.dump)
    monkeypatch.setattr(joblib, "dump", dump)
    if writer == "serving_model":
        monkeypatch.setattr(tier4.store, "file_sha256", lambda _path: SNAPSHOT)
        monkeypatch.setattr(tier4, "training_frames", lambda *_args: (panel, panel))
        monkeypatch.setattr(tier4, "fit_fold", lambda *_args: served.estimator)
        monkeypatch.setattr(
            tier4, "_pool_before", lambda *_args, **_kwargs: (served.pool_pred, served.pool_res)
        )
        result = tier4.serving_model(FOLD, panel=panel, model=model)
        assert result.estimator is served.estimator
    elif writer == "upgrade_one":
        count = prepare.upgrade_one(
            target,
            model=model,
            snapshot=SNAPSHOT,
            panel_loader=lambda: panel,
            pool_builder=lambda *_args: (served.pool_pred, served.pool_res),
        )
        assert count == served.pool_pred.size
        assert path.stat().st_mode & 0o777 == 0o640
    else:
        count = prepare.build_one(
            FOLD,
            model=model,
            snapshot=SNAPSHOT,
            cache_dir=tmp_path,
            panel_loader=lambda: panel,
            fit_builder=lambda *_args: served,
        )
        assert count == served.pool_pred.size

    dump.assert_called_once()
    # Omission currently happens to default to zero, but the persistent writer
    # contract requires an explicit argument rather than that library default.
    assert "compress" in dump.call_args.kwargs
    assert dump.call_args.kwargs["compress"] == 0
    destination = dump.call_args.args[1]
    assert not isinstance(destination, (str, Path)), "a suffix can override compress=0"
    assert "b" in destination.mode
    assert destination.closed
    _assert_mappable_roundtrip(path, served, metadata)
    assert list(tmp_path.iterdir()) == [path], "temporary siblings must be removed"
