"""The frozen DYN-SV chooser analog pool (P5-4): builder, artifact, loader.

The expected side is the real legacy loader ``Scorer._chooser_analog_pool``
reading the same rows from a parquet file, so the builder's filter and the
artifact's stored order are proved against legacy, array for array.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

import engine.score as score_mod
from engine.score import CHOOSER_ANALOG_POOL, Scorer
from engine.v2.models import no_fit_guard
from engine.v2.models.chooser_analog_pool import (
    CHOOSER_ANALOG_DIMS,
    CHOOSER_ANALOG_K,
    ChooserAnalogPoolError,
    make_chooser_analog_pool_artifact,
)
from engine.v2.models.frozen_release import member_kind
from engine.v2.models.frozen_state import (
    FrozenStateError,
    FrozenStateLoader,
    FrozenStateRef,
    serialize_frozen_state,
)
from engine.v2.models.lineage import DataDependency, Lineage
from engine.v2.models.training.chooser_pool import build_chooser_analog_pool_artifact
from engine.v2.scoring.native_chooser_features import analog_arrays

LINEAGE = Lineage(data=(DataDependency(table="exp137.candidates",
                                       end_exclusive="2030-01-01"),))
MENU = ("TWIN-P", "TWIN-P5", "CND-PS", "BFLY-P", "BFLY-P5", "RAMP7", "CTR5")


def pool_frame(seed: int = 3, n: int = 240) -> pd.DataFrame:
    """Prior menu candidates shaped like ``tools/build_chooser_pool.py``'s
    output, with the rows legacy must drop (NaT exit, NaN P&L, a non-finite
    dimension) and an infinite P&L legacy keeps."""
    rng = np.random.default_rng(seed)
    entry = pd.Timestamp("2023-01-02") + pd.to_timedelta(rng.integers(0, 1200, n), unit="D")
    frame = pd.DataFrame({
        "strategy": rng.choice(MENU[:4], n),
        "event_id": [f"e{i}" for i in range(n)],
        "ticker": [f"T{i % 17}" for i in range(n)],
        "entry_date": entry,
        "exit_date": entry + pd.to_timedelta(rng.integers(1, 30, n), unit="D"),
        "pnl": rng.normal(0.0, 150.0, n),
        "exp_pnl_sim": rng.normal(0.0, 40.0, n),
        "width_over_forecast": rng.uniform(0.2, 2.0, n),
        "n_legs": rng.choice([3.0, 5.0, 7.0], n),
        "anchor_over_spot": rng.uniform(0.95, 1.02, n),
        "rel_spread": rng.uniform(0.01, 0.4, n),
    })
    frame.loc[3, "exit_date"] = pd.NaT
    frame.loc[5, "pnl"] = np.nan
    frame.loc[7, "rel_spread"] = np.nan
    frame.loc[9, "width_over_forecast"] = np.inf
    frame.loc[11, "pnl"] = np.inf
    return frame.sort_values(["strategy", "entry_date"]).reset_index(drop=True)


def legacy_pool(tmp_path, monkeypatch, frame: pd.DataFrame):
    """``Scorer._chooser_analog_pool`` over ``frame`` written as the pool file."""
    frame.to_parquet(tmp_path / CHOOSER_ANALOG_POOL, index=False)
    monkeypatch.setattr(score_mod.paths, "FEATURES", tmp_path)
    scorer = object.__new__(Scorer)
    scorer._chooser_pool = score_mod._UNSET
    return Scorer._chooser_analog_pool(scorer)


def build(frame: pd.DataFrame, cutoff="2030-01-01"):
    return build_chooser_analog_pool_artifact(
        frame.to_dict("records"), pool_id="exp137.menu7.alpha50", cutoff=cutoff,
        lineage=LINEAGE)


def test_constants_are_the_legacy_ones():
    assert CHOOSER_ANALOG_DIMS == Scorer._CHOOSER_ANALOG_DIMS
    assert CHOOSER_ANALOG_K == Scorer._CHOOSER_ANALOG_K


def test_builder_equals_the_legacy_loader_array_for_array(tmp_path, monkeypatch):
    frame = pool_frame()
    legacy = legacy_pool(tmp_path, monkeypatch, frame)
    artifact = build(frame)
    assert sorted(legacy) == [name for name, _ in artifact.strategies]
    for strategy, (X, y, closed) in legacy.items():
        nX, ny, nclosed = analog_arrays(artifact, strategy)
        assert nX.tobytes() == X.tobytes() and ny.tobytes() == y.tobytes()
        assert np.array_equal(nclosed.astype("datetime64[ns]"), closed)
    assert np.isinf(np.concatenate([analog_arrays(artifact, s)[1] for s in legacy])).any()


def test_order_is_state_a_reordered_source_is_a_different_pool():
    frame = pool_frame()
    assert build(frame).content_hash != build(frame.iloc[::-1]).content_hash


def test_serialized_bytes_hash_to_the_content_hash_and_load_verified(tmp_path):
    artifact = build(pool_frame())
    data = serialize_frozen_state(artifact)
    assert "sha256:" + hashlib.sha256(data).hexdigest() == artifact.content_hash
    (tmp_path / "pool.json").write_bytes(data)
    loaded = FrozenStateLoader(tmp_path).load(
        FrozenStateRef(path="pool.json", content_hash=artifact.content_hash))
    assert loaded == artifact
    assert member_kind(loaded) == "residual"
    tampered = data.replace(b'"pool_id":"exp137', b'"pool_id":"exp999')
    (tmp_path / "bad.json").write_bytes(tampered)
    with pytest.raises(FrozenStateError, match="hash mismatch"):
        FrozenStateLoader(tmp_path).load(
            FrozenStateRef(path="bad.json", content_hash=artifact.content_hash))


def test_causal_cutoff_is_the_key_and_is_enforced():
    frame = pool_frame()
    cutoff = "2024-06-01"
    artifact = build(frame, cutoff=cutoff)
    assert artifact.key == ("exp137.menu7.alpha50", cutoff)
    days = [row[0] for _, rows in artifact.strategies for row in rows]
    assert days and max(days) < cutoff
    with pytest.raises(ChooserAnalogPoolError, match="cutoff"):
        make_chooser_analog_pool_artifact(
            pool_id="p", cutoff=cutoff, lineage=LINEAGE,
            strategies={"TWIN-P": [("2024-06-01", 1.0, 1.0, 1.0, 5.0, 1.0, 0.1)]})


@pytest.mark.parametrize("row, message", [
    (("2024-01-02", float("nan"), 1.0, 1.0, 5.0, 1.0, 0.1), "NaN"),
    (("2024-01-02", 1.0, float("inf"), 1.0, 5.0, 1.0, 0.1), "finite"),
    (("2024-01-02", 1.0, 1.0), "columns"),
])
def test_malformed_rows_are_refused_not_dropped(row, message):
    with pytest.raises(ChooserAnalogPoolError, match=message):
        make_chooser_analog_pool_artifact(pool_id="p", cutoff=None, lineage=LINEAGE,
                                          strategies={"TWIN-P": [row]})


def test_builder_refuses_a_time_of_day_and_undeclared_lineage():
    frame = pool_frame().head(5).copy()
    frame.loc[0, "exit_date"] = pd.Timestamp("2024-01-02 16:00")
    with pytest.raises(ValueError, match="time of day"):
        build(frame)
    with pytest.raises(ValueError, match="lineage"):
        build_chooser_analog_pool_artifact([], pool_id="p", cutoff=None, lineage=Lineage())


def test_builder_is_a_training_step_refused_under_the_no_fit_guard():
    with no_fit_guard(), pytest.raises(Exception, match="fit|Fit"):
        build(pool_frame())
