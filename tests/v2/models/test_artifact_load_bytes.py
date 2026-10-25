"""``load_bytes`` verification for the payoff and recalibration loaders."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import engine.v2.models.payoff_artifact as payoff_artifact_module
import engine.v2.models.recalibration_artifact as recalibration_artifact_module
from engine.v2.models.payoff_artifact import (
    PayoffArtifactError,
    PayoffArtifactLoader,
    PayoffArtifactRef,
    make_payoff_line_artifact,
    serialize_payoff_artifact,
)
from engine.v2.models.recalibration_artifact import (
    RecalibrationArtifactError,
    RecalibrationArtifactLoader,
    RecalibrationArtifactRef,
    serialize_recalibration_artifact,
)
from engine.v2.models.training.recalibration import build_recalibration_map_artifact

_LINE_FIT = {
    "intercept": 0.05, "slope": 0.004, "resid_sd": 0.01, "n": 250,
    "r": 0.42, "residuals": [0.01, -0.02, 0.03, -0.01],
}

CUTOFF = "2020-09-01"


def _pairs(n=900, seed=5) -> pd.DataFrame:
    """Mixed strategies/alphas, post-cutoff rows, NaNs and tied raw_win values."""
    rng = np.random.default_rng(seed)
    raw = np.round(rng.uniform(0.0, 1.0, n), 2)
    frame = pd.DataFrame({
        "strategy": np.where(rng.random(n) < 0.8, "STR-THRU", "STR-RUNUP"),
        "fill_alpha": np.where(rng.random(n) < 0.8, 0.5, 0.25),
        "event_id": [f"E{i}" for i in range(n)],
        "ticker": "AAA",
        "event_date": pd.Timestamp("2019-01-01"),
        "exit_date": pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 800, n), "D"),
        "raw_win": raw,
        "outcome": (rng.uniform(0, 1, n) < 0.2 + 0.5 * raw).astype(float),
    })
    frame.loc[:6, "raw_win"] = np.nan
    frame.loc[7:9, "outcome"] = np.nan
    return frame


def test_loader_load_bytes_refuses_a_hash_mismatch(tmp_path):
    artifact = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
    )
    raw = serialize_payoff_artifact(artifact)
    loader = PayoffArtifactLoader(tmp_path)
    with pytest.raises(PayoffArtifactError, match="hash mismatch"):
        loader.load_bytes(raw, "sha256:" + "0" * 64)


def test_loader_load_bytes_loads_good_bytes(tmp_path):
    artifact = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
    )
    raw = serialize_payoff_artifact(artifact)
    loader = PayoffArtifactLoader(tmp_path)
    loaded = loader.load_bytes(raw, artifact.content_hash)
    assert loaded == artifact
    assert loader.cache_size == 1


def test_recalibration_loader_load_bytes_refuses_a_hash_mismatch(tmp_path):
    artifact = build_recalibration_map_artifact(_pairs(), strategy="STR-THRU", alpha=0.5,
                                                before=CUTOFF)
    raw = serialize_recalibration_artifact(artifact)
    loader = RecalibrationArtifactLoader(tmp_path)
    with pytest.raises(RecalibrationArtifactError, match="hash mismatch"):
        loader.load_bytes(raw, "sha256:" + "0" * 64)


def test_recalibration_loader_load_bytes_loads_good_bytes(tmp_path):
    artifact = build_recalibration_map_artifact(_pairs(), strategy="STR-THRU", alpha=0.5,
                                                before=CUTOFF)
    raw = serialize_recalibration_artifact(artifact)
    loader = RecalibrationArtifactLoader(tmp_path)
    loaded = loader.load_bytes(raw, artifact.content_hash)
    assert loaded == artifact
    assert loader.cache_size == 1


def test_loader_load_bytes_reuses_the_cached_object(tmp_path):
    artifact = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
    )
    raw = serialize_payoff_artifact(artifact)
    loader = PayoffArtifactLoader(tmp_path)
    first = loader.load_bytes(raw, artifact.content_hash)
    second = loader.load_bytes(raw, artifact.content_hash)
    assert first is second
    assert loader.cache_size == 1


def test_recalibration_loader_load_bytes_reuses_the_cached_object(tmp_path):
    artifact = build_recalibration_map_artifact(_pairs(), strategy="STR-THRU", alpha=0.5,
                                                before=CUTOFF)
    raw = serialize_recalibration_artifact(artifact)
    loader = RecalibrationArtifactLoader(tmp_path)
    first = loader.load_bytes(raw, artifact.content_hash)
    second = loader.load_bytes(raw, artifact.content_hash)
    assert first is second
    assert loader.cache_size == 1


def test_loader_load_bytes_checks_hash_before_cache(tmp_path):
    good = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
    )
    other = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.25,
    )
    loader = PayoffArtifactLoader(tmp_path)
    assert loader.load_bytes(serialize_payoff_artifact(good), good.content_hash) == good
    assert loader.cache_size == 1
    with pytest.raises(PayoffArtifactError, match="hash mismatch"):
        loader.load_bytes(serialize_payoff_artifact(other), good.content_hash)


def test_recalibration_loader_load_bytes_checks_hash_before_cache(tmp_path):
    good = build_recalibration_map_artifact(_pairs(), strategy="STR-THRU", alpha=0.5,
                                            before=CUTOFF)
    other = build_recalibration_map_artifact(_pairs(), strategy="STR-THRU", alpha=0.25,
                                             before=CUTOFF)
    loader = RecalibrationArtifactLoader(tmp_path)
    assert loader.load_bytes(serialize_recalibration_artifact(good), good.content_hash) == good
    assert loader.cache_size == 1
    with pytest.raises(RecalibrationArtifactError, match="hash mismatch"):
        loader.load_bytes(serialize_recalibration_artifact(other), good.content_hash)


def test_loader_load_bytes_refuses_non_bytes(tmp_path):
    loader = PayoffArtifactLoader(tmp_path)
    with pytest.raises(PayoffArtifactError, match="payoff artifact bytes must be bytes"):
        loader.load_bytes("not bytes", "sha256:" + "0" * 64)


def test_recalibration_loader_load_bytes_refuses_non_bytes(tmp_path):
    loader = RecalibrationArtifactLoader(tmp_path)
    with pytest.raises(RecalibrationArtifactError,
                       match="recalibration artifact bytes must be bytes"):
        loader.load_bytes("not bytes", "sha256:" + "0" * 64)


def test_loader_load_bytes_refuses_oversize_bytes(tmp_path, monkeypatch):
    artifact = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
    )
    raw = serialize_payoff_artifact(artifact)
    monkeypatch.setattr(payoff_artifact_module, "MAX_ARTIFACT_BYTES", len(raw) - 1)
    loader = PayoffArtifactLoader(tmp_path)
    with pytest.raises(PayoffArtifactError, match="payoff artifact exceeds byte limit"):
        loader.load_bytes(raw, artifact.content_hash)


def test_recalibration_loader_load_bytes_refuses_oversize_bytes(tmp_path, monkeypatch):
    artifact = build_recalibration_map_artifact(_pairs(), strategy="STR-THRU", alpha=0.5,
                                                before=CUTOFF)
    raw = serialize_recalibration_artifact(artifact)
    monkeypatch.setattr(recalibration_artifact_module, "MAX_ARTIFACT_BYTES", len(raw) - 1)
    loader = RecalibrationArtifactLoader(tmp_path)
    with pytest.raises(RecalibrationArtifactError,
                       match="recalibration artifact exceeds byte limit"):
        loader.load_bytes(raw, artifact.content_hash)


def _write_payoff(tmp_path, artifact) -> PayoffArtifactRef:
    raw = serialize_payoff_artifact(artifact)
    (tmp_path / "artifact.json").write_bytes(raw)
    return PayoffArtifactRef(path="artifact.json", content_hash=artifact.content_hash)


def _write_recalibration(tmp_path, artifact) -> RecalibrationArtifactRef:
    raw = serialize_recalibration_artifact(artifact)
    (tmp_path / "artifact.json").write_bytes(raw)
    return RecalibrationArtifactRef(path="artifact.json", content_hash=artifact.content_hash)


def test_loader_load_refuses_oversize_file(tmp_path, monkeypatch):
    artifact = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
    )
    ref = _write_payoff(tmp_path, artifact)
    monkeypatch.setattr(payoff_artifact_module, "MAX_ARTIFACT_BYTES",
                        len(serialize_payoff_artifact(artifact)) - 1)
    with pytest.raises(PayoffArtifactError, match="missing payoff artifact: artifact.json"):
        PayoffArtifactLoader(tmp_path).load(ref)


def test_loader_load_loads_normal_file(tmp_path):
    artifact = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
    )
    ref = _write_payoff(tmp_path, artifact)
    loaded = PayoffArtifactLoader(tmp_path).load(ref)
    assert loaded == artifact


def test_recalibration_loader_load_refuses_oversize_file(tmp_path, monkeypatch):
    artifact = build_recalibration_map_artifact(_pairs(), strategy="STR-THRU", alpha=0.5,
                                                before=CUTOFF)
    ref = _write_recalibration(tmp_path, artifact)
    monkeypatch.setattr(payoff_artifact_module, "MAX_ARTIFACT_BYTES",
                        len(serialize_recalibration_artifact(artifact)) - 1)
    with pytest.raises(RecalibrationArtifactError,
                       match="missing recalibration artifact: artifact.json"):
        RecalibrationArtifactLoader(tmp_path).load(ref)


def test_recalibration_loader_load_loads_normal_file(tmp_path):
    artifact = build_recalibration_map_artifact(_pairs(), strategy="STR-THRU", alpha=0.5,
                                                before=CUTOFF)
    ref = _write_recalibration(tmp_path, artifact)
    loaded = RecalibrationArtifactLoader(tmp_path).load(ref)
    assert loaded == artifact
