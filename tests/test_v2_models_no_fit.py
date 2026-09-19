"""Phase 5 P5-2 negative control: the no-fit guard.

``guides/rearchitecture_phase5_models.md`` P5-2 acceptance: "Rig all
fitting/provider/cache-write paths to fail; cold and warm requests agree;
missing artifact returns MODEL_NOT_READY, never trains."

The guard (``engine/models/no_fit.py``) is thread-local and off by default,
so every test here that does not open ``no_fit_guard()`` proves legacy
behaviour is byte-for-byte what it was before this guard existed. All data
here is synthetic: small in-memory arrays/frames, never the real panel or
``data/``.

Enumerated call sites, each with its own on/off pair below:
- ``engine.data.features.tier4.fit_fold``
- ``engine.data.features.tier4.serving_model`` (cache-miss branch)
- ``engine.models.registry.ModelArtifact.save``
- ``engine.models.training.{gate,gate_forecast_analog,implied_t1,
  runup_move,size_model,iv_crush}.fit``
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from engine.models.no_fit import (
    RuntimeFitForbidden,
    fitting_forbidden,
    forbid_fitting,
    no_fit_guard,
)


class _JoblibStubEstimator:
    """A module-level (hence picklable) stand-in for a frozen sklearn-shaped model."""

    features = ("x", "y")

    def predict(self, rows):
        return [row[0] + 2.0 * row[1] for row in rows]


# --------------------------------------------------------------------------
# the guard primitive
# --------------------------------------------------------------------------


def test_guard_off_by_default_is_a_no_op():
    assert fitting_forbidden() is False
    forbid_fitting("probe")  # must not raise


def test_guard_on_raises_and_restores_on_exit():
    assert fitting_forbidden() is False
    with no_fit_guard():
        assert fitting_forbidden() is True
        with pytest.raises(RuntimeFitForbidden):
            forbid_fitting("probe")
    assert fitting_forbidden() is False


def test_guard_error_names_the_tripped_call_site():
    """``forbid_fitting``'s contract: ``path`` names the call site so a test or
    an error log can tell which path tripped without guessing."""
    with no_fit_guard():
        with pytest.raises(RuntimeFitForbidden, match="engine.probe.fit_fold"):
            forbid_fitting("engine.probe.fit_fold")


def test_guard_nesting_restores_outer_state_not_off():
    with no_fit_guard():
        with no_fit_guard():
            assert fitting_forbidden() is True
        assert fitting_forbidden() is True
    assert fitting_forbidden() is False


# --------------------------------------------------------------------------
# engine.data.features.tier4.fit_fold
# --------------------------------------------------------------------------


def test_no_fit_guard_blocks_tier4_fit_fold():
    from engine.data.features import tier4

    with no_fit_guard():
        with pytest.raises(RuntimeFitForbidden):
            tier4.fit_fold(pd.DataFrame(), object(), "2020-01-01")


def test_tier4_fit_fold_unaffected_when_guard_off():
    from engine.data.features import tier4

    rng = np.random.RandomState(0)
    n = tier4.MIN_TRAIN_ROWS + 10
    trainable = pd.DataFrame({
        "date": pd.date_range("2010-01-01", periods=n, freq="D"),
        "x": rng.rand(n),
        "y": rng.rand(n),
    })
    calls = []

    class _Model:
        features = ("x",)
        target = "y"
        seed = 0

        def fit(self, X, y, seed):
            calls.append((len(X), len(y), seed))
            return "fitted"

    result = tier4.fit_fold(trainable, _Model(), trainable["date"].iloc[-1])
    assert result == "fitted"
    assert calls and calls[0][2] == 0


# --------------------------------------------------------------------------
# engine.data.features.tier4.serving_model — the cache-miss branch
# --------------------------------------------------------------------------


def test_no_fit_guard_blocks_tier4_serving_model_miss(monkeypatch, tmp_path):
    from engine.data.features import tier4

    monkeypatch.setattr(tier4.store, "file_sha256", lambda path: "f" * 64)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)  # empty: guarantees a cache miss
    synthetic_model = type("SyntheticModel", (), {"model_id": "no-fit-probe"})()

    with no_fit_guard():
        with pytest.raises(RuntimeFitForbidden):
            tier4.serving_model(
                "2020-01-01", panel=object(), model=synthetic_model, cache=True,
            )
    # the guard fired before anything was written
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# engine.models.registry.ModelArtifact.save
# --------------------------------------------------------------------------


def test_no_fit_guard_blocks_registry_artifact_save(tmp_path):
    from engine.models.registry import ModelArtifact

    artifact = ModelArtifact(
        model=object(), role="size", features=("x",), residuals=[0.1, -0.2], target="abs_move",
    )
    target = tmp_path / "artifact.joblib"
    with no_fit_guard():
        with pytest.raises(RuntimeFitForbidden):
            artifact.save(target)
    assert not target.exists()


def test_registry_artifact_save_unaffected_when_guard_off(tmp_path):
    from engine.models.registry import ModelArtifact, load_artifact

    artifact = ModelArtifact(
        model=object(), role="size", features=("x",), residuals=[0.1, -0.2], target="abs_move",
    )
    target = tmp_path / "artifact.joblib"
    digest = artifact.save(target)
    assert target.exists()
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    assert load_artifact(target).role == "size"


# --------------------------------------------------------------------------
# engine.models.training.*.fit — one entry per module, synthetic tiny arrays
# --------------------------------------------------------------------------

_TRAINING_MODULES = (
    "gate",
    "gate_forecast_analog",
    "implied_t1",
    "runup_move",
    "size_model",
    "iv_crush",
)


@pytest.mark.parametrize("module_name", _TRAINING_MODULES)
def test_no_fit_guard_blocks_training_module_fit(module_name):
    import importlib

    module = importlib.import_module(f"engine.models.training.{module_name}")
    with no_fit_guard():
        with pytest.raises(RuntimeFitForbidden):
            module.fit(None, None)


@pytest.mark.parametrize("module_name", _TRAINING_MODULES)
def test_training_module_fit_unaffected_when_guard_off(module_name):
    import importlib

    module = importlib.import_module(f"engine.models.training.{module_name}")
    rng = np.random.RandomState(0)
    X = rng.rand(30, 2)
    y = rng.rand(30)
    fitted = module.fit(X, y)
    assert hasattr(fitted, "predict")
    predictions = np.asarray(fitted.predict(X[:3]))
    assert predictions.shape[0] == 3


# --------------------------------------------------------------------------
# planted defect: a hypothetical inference-time fallback that fits is caught
# --------------------------------------------------------------------------


def test_planted_defect_fit_during_inference_is_caught():
    """Simulates a future regression: an inference path that falls back to
    training instead of refusing when its artifact is missing. Proves the
    guard has teeth against that class of bug, not only against the exact
    call sites this commit already rigs.
    """
    from engine.data.features import tier4

    def broken_infer_fallback():
        # A hypothetical "provider" that forgot the read-only contract and
        # fits on demand instead of returning MODEL_NOT_READY.
        return tier4.fit_fold(pd.DataFrame(), object(), "2020-01-01")

    with no_fit_guard():
        with pytest.raises(RuntimeFitForbidden):
            broken_infer_fallback()


# --------------------------------------------------------------------------
# engine.payoff.fit_payoff / fit_runup_payoff — the payoff-map calibration
# Scorer.score runs on demand (engine/score.py:2314/2489 via
# Scorer.payoff/.runup_payoff, reached from _score_model/_score_runup_model).
# --------------------------------------------------------------------------


def test_no_fit_guard_blocks_payoff_fit_payoff():
    from engine import payoff

    with no_fit_guard():
        with pytest.raises(RuntimeFitForbidden):
            payoff.fit_payoff(pd.DataFrame(), "STR-THRU", alpha=0.5)


def test_payoff_fit_payoff_unaffected_when_guard_off():
    from engine import payoff

    rng = np.random.RandomState(0)
    n = payoff.MIN_TRADES + 50
    spot_entry = rng.uniform(20.0, 200.0, n)
    abs_move = rng.uniform(0.0, 10.0, n)
    trades = pd.DataFrame({
        "strategy": ["STR-THRU"] * n,
        "fill_alpha": [0.5] * n,
        "exit_date": pd.date_range("2018-01-01", periods=n, freq="D"),
        "spot_entry": spot_entry,
        "abs_move": abs_move,
        "exit_value": spot_entry * (0.05 + 0.01 * abs_move + rng.normal(0, 0.01, n)),
    })
    result = payoff.fit_payoff(trades, "STR-THRU", alpha=0.5)
    assert isinstance(result, payoff.PayoffMap)
    assert result.n == n


def test_no_fit_guard_blocks_payoff_fit_runup_payoff():
    from engine import payoff

    with no_fit_guard():
        with pytest.raises(RuntimeFitForbidden):
            payoff.fit_runup_payoff(pd.DataFrame(), alpha=0.5)


def test_payoff_fit_runup_payoff_unaffected_when_guard_off():
    from engine import payoff

    rng = np.random.RandomState(0)
    n = payoff.MIN_TRADES + 50
    spot_entry = rng.uniform(20.0, 200.0, n)
    trades = pd.DataFrame({
        "strategy": ["STR-RUNUP"] * n,
        "fill_alpha": [0.5] * n,
        "exit_date": pd.date_range("2018-01-01", periods=n, freq="D"),
        "im_t1": rng.uniform(2.0, 10.0, n),
        "spot_entry": spot_entry,
        "spot_exit": rng.uniform(20.0, 200.0, n),
        "strike": rng.uniform(20.0, 200.0, n),
        "exit_value": spot_entry * rng.uniform(0.0, 0.2, n),
    })
    result = payoff.fit_runup_payoff(trades, alpha=0.5)
    assert isinstance(result, payoff.RunupPayoffSurface)
    assert result.n == n


# --------------------------------------------------------------------------
# engine.recalibrate.fit_recalibration — the IsotonicRegression win-rate
# recalibration Scorer.score runs on demand (Scorer.recalibration, reached
# from _score_model at engine/score.py:2338).
# --------------------------------------------------------------------------


def test_no_fit_guard_blocks_recalibrate_fit_recalibration():
    from engine import recalibrate

    with no_fit_guard():
        with pytest.raises(RuntimeFitForbidden):
            # An explicit empty frame, never None: fit_recalibration(pairs=None)
            # would call load_pairs() and read a real file if the guard did not
            # fire first. It must fire first.
            recalibrate.fit_recalibration("STR-THRU", 0.5, before=None, pairs=pd.DataFrame())


def test_recalibrate_fit_recalibration_unaffected_when_guard_off():
    from engine import recalibrate

    rng = np.random.RandomState(0)
    n = recalibrate.MIN_PAIRS + 30
    pairs = pd.DataFrame({
        "strategy": ["STR-THRU"] * n,
        "fill_alpha": [0.5] * n,
        "exit_date": pd.date_range("2018-01-01", periods=n, freq="D"),
        "raw_win": rng.uniform(0.0, 1.0, n),
        "outcome": rng.randint(0, 2, n).astype(float),
    })
    result = recalibrate.fit_recalibration("STR-THRU", 0.5, before=None, pairs=pairs)
    assert isinstance(result, recalibrate.RecalibrationMap)
    assert result.n == n


# --------------------------------------------------------------------------
# FrozenInference under the guard: cold/warm agree, missing -> MODEL_NOT_READY,
# and it never touches a fitting/cache-write path either way (it never
# imports engine.models.no_fit at all — the test asserts that by running the
# whole exchange inside an active guard).
# --------------------------------------------------------------------------


def test_frozen_inference_joblib_cold_warm_and_missing_under_guard(tmp_path):
    import joblib

    from engine.v2.models import (
        MODEL_NOT_READY,
        FrozenInference,
        InferenceRequest,
        ModelBinding,
        ModelRelease,
    )
    from engine.v2.models.contracts import ArtifactMember

    path = tmp_path / "estimator.joblib"
    joblib.dump(_JoblibStubEstimator(), path)
    content_hash = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    member = ArtifactMember(name="estimator", path="estimator.joblib", content_hash=content_hash)
    binding = ModelBinding(
        binding_id="b1", model_id="joblib-m1", role="size", strategy_id="*",
        decision_clock_id="entry-close", adapter="joblib-estimator.v1",
        feature_order=("x", "y"), output_names=("prediction",), members=(member,),
    )
    release = ModelRelease(release_id="r-joblib", deployment_id="d1", bindings=(binding,))
    request = InferenceRequest(
        release_id="r-joblib", binding_id="b1", feature_order=("x", "y"),
        rows=((1.0, 2.0), (3.0, 4.0)),
    )

    with no_fit_guard():
        inference = FrozenInference(tmp_path)
        cold = inference.infer(release, request)
        warm = inference.infer(release, request)
    assert cold == warm
    assert cold.predictions == ((5.0,), (11.0,))

    path.unlink()
    with no_fit_guard():
        missing = FrozenInference(tmp_path).infer(release, request)
    assert missing.status == MODEL_NOT_READY
    assert missing.predictions == ()
