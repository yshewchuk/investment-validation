"""R4-16: forecast and gate recipes that name frozen release bindings.

A ``SourceBundle`` recipe may be ``{"binding_id": ..., "output": ...}``
instead of inline linear coefficients. The native forecast/gate stages then
run the binding through ``FrozenInference`` (hash-verified members, the
registered ``joblib-estimator.v1`` adapter), never a refit.

The model families are the ones the legacy registry actually serves
(``engine/models/registry.json``; ``engine/models/training``):

* size: ``BlendModel`` of OLS and a scaled MLP
* implied_t1, iv_crush and both gates: ``HistGradientBoostingRegressor``
* runup_move: ``LogTargetRegressor`` over an HGBR

Each is trained here on synthetic data, wrapped in the legacy
``ModelArtifact`` and written with joblib as the registry stores it. The
expected side is the legacy scorer's own call, ``registry.load_artifact(path)
.predict(X)[0]`` (engine/score.py:2275 and :3225), and equality is exact.
"""
from __future__ import annotations

import hashlib
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from engine.data.features import tier4
from engine.models import registry
from engine.models.training.common import BlendModel
from engine.models.training.runup_move import LogTargetRegressor
from engine.v2.contracts import ScoreRequest
from engine.v2.models import (
    ArtifactMember,
    FrozenInference,
    ModelBinding,
    ModelRelease,
    no_fit_guard,
)
from engine.v2.scoring import application
from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs

FEATURES = ("mean_prior_abs_move", "iv30", "signed_streak", "mcap_log")
CONTEXT = {"ticker": "AAA", "event_date": "2026-09-16", "entry_date": "2026-09-16",
           "exit_date": "2026-09-16", "expiry": "2026-09-18", "spot": 100.0}
QUOTES = {
    ("C", 100.0, "2026-09-18"): {"bid": 1.95, "ask": 2.05},
    ("P", 100.0, "2026-09-18"): {"bid": 1.95, "ask": 2.05},
}
ROWS = (
    {"mean_prior_abs_move": 6.2, "iv30": 55.0, "signed_streak": 2.0, "mcap_log": 23.1},
    {"mean_prior_abs_move": 3.1, "iv30": 31.5, "signed_streak": -1.0, "mcap_log": 25.4},
    {"mean_prior_abs_move": 9.7, "iv30": 88.0, "signed_streak": 0.0, "mcap_log": 21.9},
)
FAMILIES = ("size", "implied_t1", "iv_crush", "runup_move", "gate")

# joblib's own NumPy-2.5 deprecation noise on every load; not ours.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _data(seed: int):
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.uniform(1, 12, 400), rng.uniform(20, 120, 400),
        rng.integers(-4, 5, 400).astype(float), rng.uniform(20, 27, 400),
    ])
    y = 0.6 * X[:, 0] + 0.04 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(0, 1, 400)
    return X, y


def _model(role: str, seed: int):
    X, y = _data(seed)
    hgbr = dict(max_iter=40, learning_rate=0.1, max_leaf_nodes=8, random_state=seed)
    if role == "size":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nn = make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(8, 4), max_iter=60, random_state=seed)).fit(X, y)
        return BlendModel(models=(LinearRegression().fit(X, y), nn))
    if role == "runup_move":
        target = np.log1p(np.maximum(y, 0.0))
        return LogTargetRegressor(HistGradientBoostingRegressor(**hgbr).fit(X, target))
    if role == "gate":
        return HistGradientBoostingRegressor(**hgbr).fit(X, (y > 5.0).astype(float))
    return HistGradientBoostingRegressor(**hgbr).fit(X, y)


def _write(root, role: str, seed: int = 1) -> tuple[str, str]:
    artifact = registry.ModelArtifact(
        model=_model(role, seed), role=role, features=FEATURES,
        residuals=np.zeros(3), target="synthetic")
    path = root / f"{role}_{seed}.joblib"
    joblib.dump(artifact, path, compress=3)
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return path.name, digest


OUTPUT_NAMES = {"size": "pred_abs_move", "implied_t1": "pred_implied_t1",
                "iv_crush": "pred_iv_crush_30", "runup_move": "pred_runup_abs_move_d14",
                "gate": "gate_score"}


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp("frozen_recipes")
    bindings = []
    for role in FAMILIES:
        name, digest = _write(root, role)
        bindings.append(ModelBinding(
            binding_id=f"b-{role}", model_id=f"{role}_synthetic", role=role,
            strategy_id="*", decision_clock_id="legacy.entry_close.v1",
            adapter="joblib-estimator.v1", feature_order=FEATURES,
            output_names=(OUTPUT_NAMES[role],),
            members=(ArtifactMember(name="estimator", path=name, content_hash=digest),),
        ))
    for role in ("size", "iv_crush"):
        name, digest = _write_fold(root, role)
        bindings.append(ModelBinding(
            binding_id=f"b-{role}-fold", model_id=f"{role}_synthetic@202609", role=role,
            strategy_id="*", decision_clock_id="legacy.entry_close.v1",
            adapter="tier4-serving-fold.v1", feature_order=FEATURES,
            output_names=(OUTPUT_NAMES[role],),
            members=(ArtifactMember(name="estimator", path=name, content_hash=digest),),
        ))
    release = ModelRelease(release_id="rel-synthetic", deployment_id="dep-1",
                           bindings=tuple(bindings))
    return root, release


def _write_fold(root, role: str, seed: int = 3) -> tuple[str, str]:
    """A Tier-4 serving-fold cache, in the dict form tier4.serving_model writes."""
    path = root / f"{role}_fold_{seed}.joblib"
    joblib.dump({"estimator": _model(role, seed), "model_id": f"{role}_synthetic",
                 "fold_start": "2026-09-01", "tier3_snapshot": "abc123",
                 "features": list(FEATURES), "pool_pred": np.zeros(3),
                 "pool_res": np.zeros(3)}, path)
    return path.name, "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_fold(root, role: str, row: dict, seed: int = 3) -> float:
    """tier4.serving_model's cache-hit object, predicting as the scorer does
    (engine/score.py ``float(served.predict(features)[0])``)."""
    stored = joblib.load(root / f"{role}_fold_{seed}.joblib")
    served = tier4.ServingModel(
        estimator=stored["estimator"], model_id=stored["model_id"],
        fold_start=pd.Timestamp(stored["fold_start"]),
        tier3_snapshot=stored["tier3_snapshot"], features=tuple(stored["features"]))
    return float(served.predict(pd.DataFrame([row]))[0])


def _legacy(root, role: str, row: dict, seed: int = 1) -> float:
    """engine/score.py's own call over the registry's own loader."""
    artifact = registry.load_artifact(root / f"{role}_{seed}.joblib")
    X = pd.DataFrame([row])[list(artifact.features)].to_numpy(dtype=float)
    return float(artifact.predict(X)[0])


def _request(strategy: str) -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-frozen-recipe", calendar_revision="cal-1", strategy_version=strategy,
        deployment_id="dep-1", decision_clock_id="entry-close",
        requested_decision_at="2026-09-16", snapshot_id="snap-1", mode="replay",
        fill_model={"alpha": 0.5},
    )


def _bundle(frozen, row, *, strategy="STR-THRU", recipes=None, gate=None,
            inference=True, **overrides) -> SourceBundle:
    root, release = frozen
    recipes = recipes or ({
        "driver_prediction": {"binding_id": "b-size"},
        "forecast_abs_move": {"binding_id": "b-size"},
        "pred_iv_crush": {"binding_id": "b-iv_crush"},
    } if strategy == "STR-THRU" else {
        "driver_prediction": {"binding_id": "b-implied_t1"},
        "runup_move_prediction": {"binding_id": "b-runup_move"},
    })
    base: dict = dict(
        source_ref="frozen-recipe-bundle", strategy=strategy, context=dict(CONTEXT),
        raw_quotes=QUOTES, feature_vector=dict(row),
        feature_missing_mask={name: False for name in row},
        model_identity={"driver": {"model_id": "synthetic"}},
        forecast_recipes=recipes,
        model_artifact_refs={output: f"binding:{recipe['binding_id']}"
                             for output, recipe in recipes.items()},
        residual_recipe={}, analog_recipe={},
        gate_recipe=gate or {"binding_id": "b-gate", "threshold": 0.5},
        frozen_inference=FrozenInference(root) if inference else None,
        model_release=release if inference else None,
    )
    base.update(overrides)
    return SourceBundle(**base)


def _score(bundle: SourceBundle, strategy: str = "STR-THRU"):
    seen = {}
    record = application.score_one(
        _request(strategy), build_native_score_inputs(bundle),
        observer=lambda item: seen.setdefault(item.receipt.stage, item.output_document),
    )
    return record, seen["forecast"], seen.get("gate", {})


@pytest.mark.parametrize("row", ROWS)
def test_str_thru_size_iv_crush_and_gate_equal_legacy_exactly(frozen, row):
    root, _ = frozen
    record, forecast, gate = _score(_bundle(frozen, row))
    size = _legacy(root, "size", row)
    assert forecast["driver_prediction"] == size
    assert forecast["forecast_abs_move"] == size
    assert forecast["pred_iv_crush"] == _legacy(root, "iv_crush", row)
    assert gate["gate_score"] == _legacy(root, "gate", row)
    assert gate["gate_pass"] == (gate["gate_score"] >= 0.5)
    assert not any(code.startswith(("INVALID_", "MODEL_NOT_READY", "MISSING_FORECAST"))
                   for code in record.reason_codes)


@pytest.mark.parametrize("row", ROWS)
def test_str_runup_implied_t1_and_log_target_runup_equal_legacy_exactly(frozen, row):
    root, _ = frozen
    _, forecast, _ = _score(_bundle(frozen, row, strategy="STR-RUNUP"), "STR-RUNUP")
    assert forecast["driver_prediction"] == _legacy(root, "implied_t1", row)
    # The native D14 scale, as the model stage expects it (R4-17).
    assert forecast["runup_move_prediction"] == _legacy(root, "runup_move", row)


@pytest.mark.parametrize("row", ROWS)
def test_tier4_serving_folds_equal_legacy_serving_model_exactly(frozen, row):
    """The size and crush forecasts legacy actually serves come from Tier-4
    fold caches, not the full-refit champion files."""
    root, _ = frozen
    _, forecast, _ = _score(_bundle(frozen, row, recipes={
        "driver_prediction": {"binding_id": "b-size-fold"},
        "forecast_abs_move": {"binding_id": "b-size-fold"},
        "pred_iv_crush": {"binding_id": "b-iv_crush-fold"},
    }))
    assert forecast["forecast_abs_move"] == _legacy_fold(root, "size", row)
    assert forecast["pred_iv_crush"] == _legacy_fold(root, "iv_crush", row)
    assert forecast["forecast_abs_move"] != _legacy(root, "size", row)  # distinct artifacts


def test_frozen_recipes_score_under_the_no_fit_guard(frozen):
    root, _ = frozen
    bundle = _bundle(frozen, ROWS[0])
    with no_fit_guard():
        _, forecast, gate = _score(bundle)
    assert forecast["forecast_abs_move"] == _legacy(root, "size", ROWS[0])
    assert gate["gate_score"] == _legacy(root, "gate", ROWS[0])


def test_planted_defect_a_different_artifact_is_detected(frozen, tmp_path):
    """Negative control: the same family retrained on other data under the
    same binding id gives a different number, so the comparison can fail."""
    root, release = frozen
    name, digest = _write(tmp_path, "size", seed=7)
    swapped = ModelRelease(release_id=release.release_id, deployment_id="dep-1", bindings=tuple(
        ModelBinding(**{**binding.__dict__, "members": (
            ArtifactMember(name="estimator", path=name, content_hash=digest),)})
        if binding.role == "size" else binding for binding in release.bindings))
    bundle = _bundle(frozen, ROWS[0], frozen_inference=FrozenInference(tmp_path),
                     model_release=swapped,
                     recipes={"driver_prediction": {"binding_id": "b-size"}},
                     gate={"model": {"intercept": 0.0, "coefficients": {}}, "threshold": 0.0})
    _, forecast, _ = _score(bundle)
    assert forecast["driver_prediction"] != _legacy(root, "size", ROWS[0])


def test_tampered_member_refuses_model_not_ready(frozen, tmp_path):
    root, release = frozen
    (tmp_path / "size_1.joblib").write_bytes((root / "size_1.joblib").read_bytes() + b"x")
    bundle = _bundle(frozen, ROWS[0], frozen_inference=FrozenInference(tmp_path))
    record, forecast, _ = _score(bundle)
    assert "MODEL_NOT_READY" in record.reason_codes
    assert "ARTIFACT_INVALID" in record.reason_codes
    assert forecast.get("forecast_abs_move") is None


def test_declared_but_unresolved_recipes_refuse_and_never_fall_back(frozen):
    record, forecast, gate = _score(_bundle(frozen, ROWS[0], inference=False))
    assert "MODEL_NOT_READY" in record.reason_codes
    assert forecast.get("driver_prediction") is None
    assert gate.get("gate_score") is None
    missing = _bundle(frozen, ROWS[0], recipes={
        "driver_prediction": {"binding_id": "b-absent"}})
    record, forecast, _ = _score(missing)
    assert {"MODEL_NOT_READY", "BINDING_NOT_FOUND"} <= set(record.reason_codes)
    assert forecast.get("driver_prediction") is None


def test_missing_feature_refuses_without_a_number(frozen):
    row = {key: value for key, value in ROWS[0].items() if key != "iv30"}
    record, forecast, _ = _score(_bundle(frozen, row))
    assert "MISSING_FEATURES" in record.reason_codes
    assert forecast.get("forecast_abs_move") is None


def test_malformed_frozen_recipes_are_refused_at_build(frozen):
    with pytest.raises(ValueError, match="role"):
        build_native_score_inputs(_bundle(frozen, ROWS[0], recipes={
            "driver_prediction": {"binding_id": "b-iv_crush"}}))
    with pytest.raises(ValueError, match="role"):
        build_native_score_inputs(_bundle(
            frozen, ROWS[0], gate={"binding_id": "b-size", "threshold": 0.5}))
    with pytest.raises(ValueError, match="not in"):
        build_native_score_inputs(_bundle(frozen, ROWS[0], recipes={
            "driver_prediction": {"binding_id": "b-size", "output": "nope"}}))
    with pytest.raises(ValueError, match="unsupported recipe fields"):
        build_native_score_inputs(_bundle(frozen, ROWS[0], recipes={
            "driver_prediction": {"binding_id": "b-size", "intercept": 1.0}}))
    with pytest.raises(ValueError, match="both a binding and a linear model"):
        build_native_score_inputs(_bundle(frozen, ROWS[0], gate={
            "binding_id": "b-gate", "threshold": 0.5,
            "model": {"intercept": 0.0, "coefficients": {}}}))


def test_receipts_bind_binding_identity_not_object_addresses(frozen):
    root, release = frozen
    first = build_native_score_inputs(_bundle(frozen, ROWS[0]))
    second = build_native_score_inputs(_bundle(
        frozen, ROWS[0], frozen_inference=FrozenInference(root)))
    assert first.stage_receipts == second.stage_receipts
    other = ModelRelease(release_id="rel-other", deployment_id="dep-1",
                         bindings=release.bindings)
    third = build_native_score_inputs(_bundle(frozen, ROWS[0], model_release=other))
    assert first.stage_receipts != third.stage_receipts
