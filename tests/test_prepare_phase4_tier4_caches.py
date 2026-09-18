from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from engine.data.features import tier4
from tools import prepare_phase4_tier4_caches as prepare

MODEL = tier4.FeatureModel(
    model_id="implied-test",
    produces="pred_im_t1_d14",
    features=("x",),
    target="y",
    fit=lambda *_args: None,
    prepare=lambda panel: panel,
)
SNAPSHOT = "a" * 64
FOLD = pd.Timestamp("2026-09-01")


def _cache(directory: Path, *, pools: bool = False, partial: bool = False) -> Path:
    path = directory / tier4._serving_path(MODEL.model_id, FOLD, SNAPSHOT).name
    stored = {
        "estimator": {"weights": [1.0]},
        "model_id": MODEL.model_id,
        "fold_start": str(FOLD.date()),
        "tier3_snapshot": SNAPSHOT,
        "features": list(MODEL.features),
    }
    if pools:
        stored["pool_pred"] = np.array([1.0, 2.0])
        stored["pool_res"] = np.array([0.1, -0.2])
    elif partial:
        stored["pool_pred"] = np.array([1.0])
    joblib.dump(stored, path)
    return path


def test_discovery_selects_only_existing_old_matching_cache(tmp_path, monkeypatch):
    path = _cache(tmp_path)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)

    targets, missing = prepare.discover_targets(
        [FOLD, pd.Timestamp("2026-10-01")],
        cache_dir=tmp_path,
        model=MODEL,
        snapshot=SNAPSHOT,
    )

    assert targets == [prepare.CacheTarget(path, FOLD, prepare._sha256(path))]
    assert missing == [pd.Timestamp("2026-10-01")]


def test_discovery_skips_already_upgraded_cache(tmp_path, monkeypatch):
    _cache(tmp_path, pools=True)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)

    targets, missing = prepare.discover_targets(
        [FOLD], cache_dir=tmp_path, model=MODEL, snapshot=SNAPSHOT
    )

    assert targets == []
    assert missing == []


def test_discovery_refuses_partial_pool(tmp_path, monkeypatch):
    _cache(tmp_path, partial=True)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)

    with pytest.raises(prepare.CachePreparationError, match="partial residual pool"):
        prepare.discover_targets([FOLD], cache_dir=tmp_path, model=MODEL, snapshot=SNAPSHOT)


def test_upgrade_is_atomic_and_preserves_estimator_and_metadata(tmp_path, monkeypatch):
    path = _cache(tmp_path)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
    original = joblib.load(path)
    target = prepare.CacheTarget(path, FOLD, prepare._sha256(path))

    count = prepare.upgrade_one(
        target,
        model=MODEL,
        snapshot=SNAPSHOT,
        panel_loader=lambda: pd.DataFrame({"x": [1.0], "y": [2.0]}),
        pool_builder=lambda *_args: (
            np.array([2.0, 3.0]),
            np.array([-0.25, 0.5]),
        ),
    )

    assert count == 2
    upgraded = joblib.load(path)
    assert joblib.hash(upgraded["estimator"]) == joblib.hash(original["estimator"])
    assert upgraded["model_id"] == original["model_id"]
    assert upgraded["fold_start"] == original["fold_start"]
    assert upgraded["tier3_snapshot"] == original["tier3_snapshot"]
    assert upgraded["features"] == original["features"]
    assert np.array_equal(upgraded["pool_pred"], np.array([2.0, 3.0]))
    assert np.array_equal(upgraded["pool_res"], np.array([-0.25, 0.5]))
    assert not list(tmp_path.glob(".*.tmp"))


def test_upgrade_refuses_a_race_before_replacement(tmp_path, monkeypatch):
    path = _cache(tmp_path)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
    target = prepare.CacheTarget(path, FOLD, "0" * 64)

    with pytest.raises(prepare.CachePreparationError, match="changed after discovery"):
        prepare.upgrade_one(
            target,
            model=MODEL,
            snapshot=SNAPSHOT,
            panel_loader=lambda: pd.DataFrame(),
            pool_builder=lambda *_args: (np.array([]), np.array([])),
        )


def test_fold_must_be_month_boundary():
    with pytest.raises(prepare.CachePreparationError, match="fold boundary"):
        prepare._fold("2026-09-17")


def test_fold_must_not_be_missing():
    with pytest.raises(prepare.CachePreparationError, match="must not be missing"):
        prepare._fold(None)
