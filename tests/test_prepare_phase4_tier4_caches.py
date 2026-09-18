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
#: A second, non-implied producer shaped like the real `iv_crush`
#: (tier4.py:401 iv_crush_feature_model): a different model_id, a different
#: feature list, a SIGNED target with `interval_floor=None`. Exercises the
#: preparer's generic model handling rather than only the implied_t1 shape.
OTHER_MODEL = tier4.FeatureModel(
    model_id="iv-crush-test",
    produces="pred_iv_crush_30",
    features=("z", "w"),
    target="crush",
    fit=lambda *_args: None,
    prepare=lambda panel: panel,
    interval_floor=None,
)
SNAPSHOT = "a" * 64
FOLD = pd.Timestamp("2026-09-01")


def _cache(
    directory: Path,
    *,
    model: tier4.FeatureModel = MODEL,
    pools: bool = False,
    partial: bool = False,
    bad_features: bool = False,
) -> Path:
    path = directory / tier4._serving_path(model.model_id, FOLD, SNAPSHOT).name
    stored = {
        "estimator": {"weights": [1.0]},
        "model_id": model.model_id,
        "fold_start": str(FOLD.date()),
        "tier3_snapshot": SNAPSHOT,
        "features": list(MODEL.features if bad_features else model.features),
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


# -- --model: mapping, default, selection -----------------------------------


def test_default_model_is_implied_t1():
    assert prepare.DEFAULT_MODEL == "implied_t1"
    assert prepare.MODEL_CHOICES[prepare.DEFAULT_MODEL] == "pred_im_t1_d14"


def test_model_choices_cover_every_tier4_producer():
    # Same producer set Scorer._serving/_crush_forecast can reach: the
    # `_chooser_frame` loop (pred_abs_move via its own forecast call,
    # pred_im_t1_d14, pred_runup_abs_move_d14) plus `_crush_forecast`
    # (pred_iv_crush_30).
    assert set(prepare.MODEL_CHOICES.values()) == set(tier4.PRODUCES)


def test_selected_models_default_dedupes_and_expands_all():
    assert prepare._selected_models([]) == [prepare.DEFAULT_MODEL]
    assert prepare._selected_models(["size", "size", "iv_crush"]) == ["size", "iv_crush"]
    assert prepare._selected_models(["size", "all"]) == list(prepare.MODEL_CHOICES)


def test_resolve_model_maps_cli_name_to_produces_like_scorer_serving(monkeypatch):
    seen = {}

    def fake_feature_model(produces, registry=None):
        seen["produces"] = produces
        return OTHER_MODEL

    monkeypatch.setattr(tier4, "feature_model", fake_feature_model)

    resolved = prepare._resolve_model("iv_crush")

    assert seen["produces"] == "pred_iv_crush_30"
    assert resolved is OTHER_MODEL


def test_resolve_model_rejects_unknown_name():
    with pytest.raises(prepare.CachePreparationError, match="not a known --model"):
        prepare._resolve_model("bogus")


def test_main_rejects_unknown_model_before_any_real_work():
    # argparse's `choices` check fires during parse_args, before main() ever
    # reaches phase4_required_folds or store.file_sha256(paths.PANEL) — so
    # this never touches real data.
    with pytest.raises(SystemExit):
        prepare.main(["--model", "bogus", "--dry-run"])


# -- generic handling of a non-implied model ---------------------------------


def test_discovery_selects_a_non_implied_old_matching_cache(tmp_path, monkeypatch):
    path = _cache(tmp_path, model=OTHER_MODEL)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)

    targets, missing = prepare.discover_targets(
        [FOLD], cache_dir=tmp_path, model=OTHER_MODEL, snapshot=SNAPSHOT
    )

    assert targets == [prepare.CacheTarget(path, FOLD, prepare._sha256(path))]
    assert missing == []


def test_discovery_refuses_identity_mismatch_for_non_implied_model(tmp_path, monkeypatch):
    # File lives at OTHER_MODEL's path (model_id/fold/snapshot), but its
    # stored `features` belong to a different model — exactly the corruption
    # `_validate_identity` exists to catch, now exercised on a non-implied
    # producer instead of only implied_t1.
    _cache(tmp_path, model=OTHER_MODEL, bad_features=True)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)

    with pytest.raises(prepare.CachePreparationError, match="identity mismatch"):
        prepare.discover_targets(
            [FOLD], cache_dir=tmp_path, model=OTHER_MODEL, snapshot=SNAPSHOT
        )


def test_upgrade_succeeds_for_a_non_implied_model_and_payload_is_unchanged(
    tmp_path, monkeypatch
):
    path = _cache(tmp_path, model=OTHER_MODEL)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
    original = joblib.load(path)
    target = prepare.CacheTarget(path, FOLD, prepare._sha256(path))

    count = prepare.upgrade_one(
        target,
        model=OTHER_MODEL,
        snapshot=SNAPSHOT,
        panel_loader=lambda: pd.DataFrame({"z": [1.0], "w": [2.0], "crush": [0.1]}),
        pool_builder=lambda *_args: (np.array([5.0]), np.array([-1.0])),
    )

    assert count == 1
    upgraded = joblib.load(path)
    assert joblib.hash(upgraded["estimator"]) == joblib.hash(original["estimator"])
    assert upgraded["model_id"] == OTHER_MODEL.model_id == original["model_id"]
    assert upgraded["fold_start"] == original["fold_start"]
    assert upgraded["tier3_snapshot"] == original["tier3_snapshot"]
    assert upgraded["features"] == original["features"] == list(OTHER_MODEL.features)
    assert np.array_equal(upgraded["pool_pred"], np.array([5.0]))
    assert np.array_equal(upgraded["pool_res"], np.array([-1.0]))
    assert not list(tmp_path.glob(".*.tmp"))
