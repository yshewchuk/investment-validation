"""Serving-fold writers must produce artifacts that public joblib can mmap."""
# packages: engine.v2.features, engine.v2.models
from __future__ import annotations

import io
import warnings
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from engine.data.features import tier4
from engine.v2.models import AdapterError, ModelBinding
from engine.v2.models.adapters import (
    ReadOnlyArtifact,
    Tier4ServingFoldAdapter,
    _ServingFold,
)
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


def _wire_serving_model(tmp_path, monkeypatch, model, served):
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
    monkeypatch.setattr(tier4.store, "file_sha256", lambda _path: SNAPSHOT)
    monkeypatch.setattr(tier4, "training_frames", lambda *_args: (None, None))
    monkeypatch.setattr(tier4, "fit_fold", lambda *_args: served.estimator)
    monkeypatch.setattr(
        tier4, "_pool_before", lambda *_args, **_kwargs: (served.pool_pred, served.pool_res)
    )


def test_fresh_serving_model_cache_declares_the_last_training_date(
    tmp_path, monkeypatch, model, served
):
    panel = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "z": [0.5, -1.5, 2.5],
            "y": [0.25, -0.75, 1.25],
            "date": pd.to_datetime(["2026-07-15", "2026-08-20", "2026-10-01"]),
        }
    )
    _wire_serving_model(tmp_path, monkeypatch, model, served)
    # The trainable rows actually fit for this fold: every dated row before it.
    monkeypatch.setattr(tier4, "training_frames", lambda *_args: (panel, panel))

    tier4.serving_model(FOLD, panel=panel, model=model, cache=True)

    stored = joblib.load(tmp_path / tier4._serving_path(model.model_id, FOLD, SNAPSHOT).name)
    assert stored["training_cutoff"] == "2026-08-20"
    assert stored["training_cutoff"] < str(FOLD.date())


def test_serving_model_cache_write_failure_leaves_no_partial_artifact(
    tmp_path, monkeypatch, model, served
):
    _wire_serving_model(tmp_path, monkeypatch, model, served)
    panel = pd.DataFrame()
    monkeypatch.setattr(tier4, "training_frames", lambda *_args: (panel, panel))

    def failing_dump(_header, handle, **_kwargs):
        handle.write(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(joblib, "dump", failing_dump)
    with pytest.raises(OSError, match="simulated write failure"):
        tier4.serving_model(FOLD, panel=panel, model=model, cache=True)

    path = tier4._serving_path(model.model_id, FOLD, SNAPSHOT)
    assert not path.exists(), "failure must not publish the partial bytes"
    assert list(tmp_path.iterdir()) == [], "no temporary sibling remains"


def test_serving_model_cache_hit_carries_the_cutoff_and_a_legacy_header_serves_none(
    tmp_path, monkeypatch, model, served
):
    panel = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "z": [0.5, -1.5, 2.5],
            "y": [0.25, -0.75, 1.25],
            "date": pd.to_datetime(["2026-07-15", "2026-08-20", "2026-10-01"]),
        }
    )
    _wire_serving_model(tmp_path, monkeypatch, model, served)
    monkeypatch.setattr(tier4, "training_frames", lambda *_args: (panel, panel))

    fresh = tier4.serving_model(FOLD, panel=panel, model=model, cache=True)
    path = tmp_path / tier4._serving_path(model.model_id, FOLD, SNAPSHOT).name

    hit = tier4.serving_model(FOLD, panel=panel, model=model, cache=True)
    assert fresh.training_cutoff == hit.training_cutoff == "2026-08-20"
    assert hit.training_cutoff == joblib.load(path)["training_cutoff"]

    # A legacy header with no declared cutoff serves None, never a guess.
    stored = joblib.load(path)
    stored.pop("training_cutoff", None)
    joblib.dump(stored, path)
    legacy = tier4.serving_model(FOLD, panel=pd.DataFrame(), model=model, cache=True)
    assert legacy.training_cutoff is None


def test_serving_model_cache_hit_refuses_a_malformed_training_cutoff(
    tmp_path, monkeypatch, model, served
):
    _wire_serving_model(tmp_path, monkeypatch, model, served)
    path = tier4._serving_path(model.model_id, FOLD, SNAPSHOT)
    joblib.dump(
        {
            "estimator": served.estimator,
            "model_id": served.model_id,
            "fold_start": str(FOLD.date()),
            "tier3_snapshot": SNAPSHOT,
            "features": list(served.features),
            "pool_pred": served.pool_pred,
            "pool_res": served.pool_res,
            "training_cutoff": ["2026-08-20"],
        },
        path,
    )
    with pytest.raises(tier4.Tier4Error, match="^invalid serving fold training cutoff$"):
        tier4.serving_model(FOLD, panel=pd.DataFrame(), model=model, cache=True)


@pytest.mark.parametrize("training_cutoff", [str(FOLD.date()), "2026-09-02"])
def test_serving_model_cache_hit_refuses_cutoff_at_or_after_fold(
    tmp_path, monkeypatch, model, served, training_cutoff
):
    _wire_serving_model(tmp_path, monkeypatch, model, served)
    path = tier4._serving_path(model.model_id, FOLD, SNAPSHOT)
    joblib.dump(
        {
            "estimator": served.estimator,
            "model_id": served.model_id,
            "fold_start": str(FOLD.date()),
            "tier3_snapshot": SNAPSHOT,
            "features": list(served.features),
            "pool_pred": served.pool_pred,
            "pool_res": served.pool_res,
            "training_cutoff": training_cutoff,
        },
        path,
    )
    with pytest.raises(tier4.Tier4Error, match="^invalid serving fold training cutoff$"):
        tier4.serving_model(FOLD, panel=pd.DataFrame(), model=model, cache=True)


@pytest.mark.parametrize(
    "training_cutoff",
    [["2026-08-20"], date(2026, 8, 20), datetime(2026, 8, 20)],
)
def test_direct_adapter_metadata_refuses_a_mutable_training_cutoff(
    model, served, training_cutoff
):
    binding = ModelBinding(
        binding_id="direct-cutoff",
        model_id=model.model_id,
        role="size",
        strategy_id="*",
        decision_clock_id="entry-close",
        adapter="tier4-serving-fold.v1",
        feature_order=tuple(model.features),
        output_names=(model.produces,),
        members=(),
    )
    header = {
        "estimator": served.estimator,
        "model_id": model.model_id,
        "fold_start": str(FOLD.date()),
        "tier3_snapshot": SNAPSHOT,
        "features": list(model.features),
        "pool_pred": served.pool_pred,
        "pool_res": served.pool_res,
        "training_cutoff": training_cutoff,
    }
    stream = io.BytesIO()
    joblib.dump(header, stream, compress=0)
    with pytest.raises(AdapterError, match="^invalid serving fold training cutoff$"):
        Tier4ServingFoldAdapter().load({"estimator": stream.getvalue()}, binding)


def _direct_binding(model, binding_id):
    return ModelBinding(
        binding_id=binding_id,
        model_id=model.model_id,
        role="size",
        strategy_id="*",
        decision_clock_id="entry-close",
        adapter="tier4-serving-fold.v1",
        feature_order=tuple(model.features),
        output_names=(model.produces,),
        members=(),
    )


_OMIT = object()


def _direct_header(model, served, *, fold_start=_OMIT, training_cutoff=None):
    header = {
        "model_id": model.model_id,
        "tier3_snapshot": SNAPSHOT,
        "features": list(model.features),
        "pool_pred": served.pool_pred,
        "pool_res": served.pool_res,
    }
    if fold_start is not _OMIT:
        header["fold_start"] = fold_start
    if training_cutoff is not None:
        header["training_cutoff"] = training_cutoff
    return header


@pytest.mark.parametrize("fold_start", [pytest.param(_OMIT, id="missing"), None])
def test_direct_adapter_load_refuses_declared_cutoff_without_fold_start(
    model, served, fold_start
):
    binding = _direct_binding(model, "direct-cutoff-missing-fold")
    header = {
        "estimator": served.estimator,
        **_direct_header(
            model, served, fold_start=fold_start, training_cutoff="2026-08-20"
        ),
    }
    stream = io.BytesIO()
    joblib.dump(header, stream, compress=0)
    with pytest.raises(AdapterError, match="^invalid serving fold training cutoff$"):
        Tier4ServingFoldAdapter().load({"estimator": stream.getvalue()}, binding)


@pytest.mark.parametrize("fold_start", [pytest.param(_OMIT, id="missing"), None])
def test_direct_adapter_predict_refuses_declared_cutoff_without_fold_start(
    model, served, fold_start
):
    binding = _direct_binding(model, "direct-predict-cutoff-missing-fold")
    header = _direct_header(
        model, served, fold_start=fold_start, training_cutoff="2026-08-20"
    )
    artifact = _ServingFold(ReadOnlyArtifact(served.estimator), header)
    with pytest.raises(AdapterError, match="^invalid serving fold training cutoff$"):
        Tier4ServingFoldAdapter().predict(artifact, [[0.0, 0.0]], binding)


def test_direct_adapter_predict_refuses_cutoff_at_or_after_fold(model, served):
    binding = _direct_binding(model, "direct-predict-cutoff-after-fold")
    header = _direct_header(
        model, served, fold_start=str(FOLD.date()), training_cutoff=str(FOLD.date())
    )
    artifact = _ServingFold(ReadOnlyArtifact(served.estimator), header)
    with pytest.raises(AdapterError, match="^invalid serving fold training cutoff$"):
        Tier4ServingFoldAdapter().predict(artifact, [[0.0, 0.0]], binding)


def test_direct_adapter_accepts_legacy_cutoffless_header(model, served):
    binding = _direct_binding(model, "direct-legacy")
    header = {
        "estimator": served.estimator,
        **_direct_header(model, served, fold_start=str(FOLD.date())),
    }
    stream = io.BytesIO()
    joblib.dump(header, stream, compress=0)
    artifact = Tier4ServingFoldAdapter().load({"estimator": stream.getvalue()}, binding)
    result = Tier4ServingFoldAdapter().predict(artifact, [[0.0, 0.0]], binding)
    assert len(result) == 1


def test_build_one_emits_the_declared_cutoff_and_omits_when_none(
    tmp_path, monkeypatch, model, served
):
    panel = pd.DataFrame({"x": [1.0], "z": [2.0], "y": [-0.5]})
    filename = tier4._serving_path(model.model_id, FOLD, SNAPSHOT).name

    declared_dir = tmp_path / "declared"
    monkeypatch.setattr(tier4, "SERVING_DIR", declared_dir)
    prepare.build_one(
        FOLD,
        model=model,
        snapshot=SNAPSHOT,
        cache_dir=declared_dir,
        panel_loader=lambda: panel,
        fit_builder=lambda *_args: replace(served, training_cutoff="2026-08-15"),
    )
    assert joblib.load(declared_dir / filename)["training_cutoff"] == "2026-08-15"

    omitted_dir = tmp_path / "omitted"
    monkeypatch.setattr(tier4, "SERVING_DIR", omitted_dir)
    prepare.build_one(
        FOLD,
        model=model,
        snapshot=SNAPSHOT,
        cache_dir=omitted_dir,
        panel_loader=lambda: panel,
        fit_builder=lambda *_args: served,
    )
    assert "training_cutoff" not in joblib.load(omitted_dir / filename)


def test_build_one_refuses_a_malformed_builder_cutoff(
    tmp_path, monkeypatch, model, served
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    panel = pd.DataFrame({"x": [1.0], "z": [2.0], "y": [-0.5]})
    with pytest.raises(
        prepare.CachePreparationError,
        match="^fit builder returned an invalid training cutoff$",
    ):
        prepare.build_one(
            FOLD,
            model=model,
            snapshot=SNAPSHOT,
            cache_dir=cache_dir,
            panel_loader=lambda: panel,
            fit_builder=lambda *_args: replace(served, training_cutoff=["2026-08-20"]),
        )
    assert list(cache_dir.iterdir()) == []


@pytest.mark.parametrize("training_cutoff", [str(FOLD.date()), "2026-09-02"])
def test_build_one_refuses_cutoff_at_or_after_fold(
    tmp_path, monkeypatch, model, served, training_cutoff
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    panel = pd.DataFrame({"x": [1.0], "z": [2.0], "y": [-0.5]})
    with pytest.raises(
        prepare.CachePreparationError,
        match="^fit builder returned an invalid training cutoff$",
    ):
        prepare.build_one(
            FOLD,
            model=model,
            snapshot=SNAPSHOT,
            cache_dir=cache_dir,
            panel_loader=lambda: panel,
            fit_builder=lambda *_args: replace(served, training_cutoff=training_cutoff),
        )
    assert list(cache_dir.iterdir()) == []


def test_upgrade_one_keeps_only_a_cutoff_the_original_declared(
    tmp_path, monkeypatch, model, served
):
    metadata = _metadata(model)
    filename = tier4._serving_path(model.model_id, FOLD, SNAPSHOT).name

    declared_dir = tmp_path / "declared"
    monkeypatch.setattr(tier4, "SERVING_DIR", declared_dir)
    declared = declared_dir / filename
    declared_dir.mkdir()
    joblib.dump({**metadata, "estimator": served.estimator, "training_cutoff": "2026-08-01"}, declared)
    prepare.upgrade_one(
        prepare.CacheTarget(declared, FOLD, prepare._sha256(declared)),
        model=model,
        snapshot=SNAPSHOT,
        panel_loader=lambda: pd.DataFrame(),
        pool_builder=lambda *_args: (served.pool_pred, served.pool_res),
    )
    assert joblib.load(declared)["training_cutoff"] == "2026-08-01"

    legacy_dir = tmp_path / "legacy"
    monkeypatch.setattr(tier4, "SERVING_DIR", legacy_dir)
    legacy = legacy_dir / filename
    legacy_dir.mkdir()
    joblib.dump({**metadata, "estimator": served.estimator}, legacy)
    prepare.upgrade_one(
        prepare.CacheTarget(legacy, FOLD, prepare._sha256(legacy)),
        model=model,
        snapshot=SNAPSHOT,
        panel_loader=lambda: pd.DataFrame(),
        pool_builder=lambda *_args: (served.pool_pred, served.pool_res),
    )
    assert "training_cutoff" not in joblib.load(legacy)


@pytest.mark.parametrize("training_cutoff", [str(FOLD.date()), "2026-09-02"])
def test_direct_adapter_metadata_refuses_cutoff_at_or_after_fold(
    model, served, training_cutoff
):
    binding = ModelBinding(
        binding_id="direct-cutoff",
        model_id=model.model_id,
        role="size",
        strategy_id="*",
        decision_clock_id="entry-close",
        adapter="tier4-serving-fold.v1",
        feature_order=tuple(model.features),
        output_names=(model.produces,),
        members=(),
    )
    header = {
        "estimator": served.estimator,
        "model_id": model.model_id,
        "fold_start": str(FOLD.date()),
        "tier3_snapshot": SNAPSHOT,
        "features": list(model.features),
        "pool_pred": served.pool_pred,
        "pool_res": served.pool_res,
        "training_cutoff": training_cutoff,
    }
    stream = io.BytesIO()
    joblib.dump(header, stream, compress=0)
    with pytest.raises(AdapterError, match="^invalid serving fold training cutoff$"):
        Tier4ServingFoldAdapter().load({"estimator": stream.getvalue()}, binding)


@pytest.mark.parametrize(
    "training_cutoff", [["2026-08-20"], str(FOLD.date()), "2026-09-02"]
)
def test_upgrade_one_refuses_an_invalid_existing_cutoff_before_doing_work(
    tmp_path, monkeypatch, model, served, training_cutoff
):
    metadata = _metadata(model)
    filename = tier4._serving_path(model.model_id, FOLD, SNAPSHOT).name
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(tier4, "SERVING_DIR", cache_dir)
    path = cache_dir / filename
    joblib.dump(
        {**metadata, "estimator": served.estimator, "training_cutoff": training_cutoff},
        path,
    )
    original_bytes = path.read_bytes()

    panel_loader = Mock()
    pool_builder = Mock(return_value=(served.pool_pred, served.pool_res))

    with pytest.raises(
        prepare.CachePreparationError,
        match="^cache has an invalid training cutoff$",
    ):
        prepare.upgrade_one(
            prepare.CacheTarget(path, FOLD, prepare._sha256(path)),
            model=model,
            snapshot=SNAPSHOT,
            panel_loader=panel_loader,
            pool_builder=pool_builder,
        )

    panel_loader.assert_not_called()
    pool_builder.assert_not_called()
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("training_cutoff", [str(FOLD.date()), "2026-09-02"])
def test_classifier_refuses_invalid_optional_cutoff(
    tmp_path, monkeypatch, model, served, training_cutoff
):
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
    path = tmp_path / tier4._serving_path(model.model_id, FOLD, SNAPSHOT).name
    joblib.dump(
        {**_metadata(model), "estimator": served.estimator,
         "pool_pred": served.pool_pred, "pool_res": served.pool_res,
         "training_cutoff": training_cutoff},
        path,
    )
    with pytest.raises(
        prepare.CachePreparationError, match="^cache has an invalid training cutoff$"
    ):
        prepare._classify_fold(FOLD, directory=tmp_path, model=model, snapshot=SNAPSHOT)


def test_read_serving_header_equal_cutoff_and_legacy(tmp_path, monkeypatch, model, served):
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)
    metadata = _metadata(model)
    path = tmp_path / tier4._serving_path(model.model_id, FOLD, SNAPSHOT).name
    joblib.dump(
        {**metadata, "estimator": served.estimator, "pool_pred": served.pool_pred,
         "pool_res": served.pool_res, "training_cutoff": str(FOLD.date())},
        path,
    )
    assert tier4.read_serving_header(
        path, expected_sha256_hex=prepare._sha256(path), max_bytes=1_000_000
    ) is None
    legacy = tmp_path / "legacy" / path.name
    legacy.parent.mkdir()
    joblib.dump(
        {**metadata, "estimator": served.estimator, "pool_pred": served.pool_pred,
         "pool_res": served.pool_res},
        legacy,
    )
    header = tier4.read_serving_header(
        legacy, expected_sha256_hex=prepare._sha256(legacy), max_bytes=1_000_000
    )
    assert header is not None and "training_cutoff" not in header
