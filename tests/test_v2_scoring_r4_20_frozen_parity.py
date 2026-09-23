"""R4-20 gaps 3, 4 and 5: forecast-analog gate columns, legacy missing-feature
semantics and the DYN-SV chooser, against the legacy scorer's own code.

The expected side is always the real legacy method, run on a scorer shell
(``object.__new__(Scorer)``) whose model lookups return the same synthetic
artifacts the native side loads through ``FrozenInference``:

* gap 3 -- ``Scorer._score_gate`` (which calls the real
  ``_gate_feature_frame``/``_forecast_for_gate`` and a real
  ``tier4.ServingModel``): the forecast-analog gate's derived columns are
  derived natively from declared inputs, and ``gate_score`` is equal bit for
  bit; ``forecast_interval`` equals ``tier4.interval_for`` directly;
* gap 4 -- a non-finite frozen-model feature: legacy flags
  MISSING_FEATURES for a champion model or gate and silently serves NaN for a
  Tier-4 fold (sizing declines NO_FORECAST, the simulation is undetermined);
* gap 5 -- ``Scorer._score_chooser``: the chooser champion through a frozen
  recipe (``dyn_sv_chooser_v1_1``'s family, a ``MeanEnsemble``), no refit,
  equal to legacy's prediction, declining with the advisory
  CHOOSER_MISSING_FEATURES.
"""
from __future__ import annotations

import hashlib
import importlib
import math
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from engine import pnl_sim
from engine.data.features import tier4
from engine.models import registry
from engine.score import DYNAMIC_MENU, Scorer
from engine.v2.contracts import ScoreRequest
from engine.v2.models import (
    ArtifactMember,
    FrozenInference,
    ModelBinding,
    ModelRelease,
    no_fit_guard,
)
from engine.v2.scoring import application
from engine.v2.scoring.frozen_executor import FrozenStageExecutor, FrozenStageRefusal
from engine.v2.scoring.native_gate_features import (
    GATE_ANALOG_COLUMNS,
    GATE_FORECAST_COLUMNS,
    forecast_interval,
    gate_forecast_columns,
)
from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs
from engine.v2.scoring.stages import flags_refuse

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning"),
              pytest.mark.filterwarnings("ignore::RuntimeWarning")]

EVENT = "2026-09-16"
FOLD_FEATURES = ("mean_prior_abs_move", "iv30", "signed_streak", "mcap_log")
GATE_FEATURES = ("mean_prior_abs_move", "iv30", "im",
                 *GATE_FORECAST_COLUMNS, *GATE_ANALOG_COLUMNS)
CHOOSER_FEATURES = ("mean_prior_abs_move", "iv30", "spy_vol20", "exp_pnl_sim",
                    "exp_pnl_sim_select", "is_twin_p", "is_ctr5", "wide_market",
                    "quote_repaired")
ROW = {"mean_prior_abs_move": 6.2, "iv30": 55.0, "signed_streak": 2.0,
       "mcap_log": 23.1, "im": 5.5, "spy_vol20": 14.0}
STR_THRU_CONTEXT = {"ticker": "AAA", "event_date": EVENT, "entry_date": EVENT,
                    "exit_date": EVENT, "expiry": "2026-09-18", "spot": 100.0}
STR_THRU_QUOTES = {
    ("C", 100.0, "2026-09-18"): {"bid": 1.95, "ask": 2.05},
    ("P", 100.0, "2026-09-18"): {"bid": 1.95, "ask": 2.05},
}
TWIN_CONTEXT = {"ticker": "AAA", "event_date": EVENT, "entry_date": EVENT,
                "exit_date": "2026-09-09", "expiry": "2026-09-18", "spot": 100.0}
TWIN_QUOTES = {("P", strike, "2026-09-18"): {"bid": mid - 0.05, "ask": mid + 0.05}
               for strike, mid in ((84.0, 0.2), (92.0, 0.6), (96.0, 1.2), (100.0, 2.5),
                                   (104.0, 4.8), (108.0, 8.3), (116.0, 16.2))}
ANALOG = dict(
    analog_recipe={
        "bucket_dimensions": ("mcap_bucket", "moneyness_band", "dte_band", "implied_tercile"),
        "widening_order": ("moneyness_band", "dte_band", "implied_tercile"),
        "min_analogs": 2, "alpha": 0.5, "bootstrap_draws": 0, "bootstrap_seed": 0,
        "ci_quantiles": (0.1, 0.9),
    },
    analog_source_rows=tuple(
        {"row_id": f"a{i}", "mcap_bucket": "large", "moneyness_band": "atm",
         "dte_band": "30-45", "implied_tercile": "mid", "realized_return": value}
        for i, value in enumerate((0.10, 0.20, -0.05))),
    analog_query={"mcap_bucket": "large", "moneyness_band": "atm",
                  "dte_band": "30-45", "implied_tercile": "mid"},
)


def _hgbr(n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (400, n_features)) * 3.0 + 5.0
    y = X @ np.linspace(0.1, 0.5, n_features) + rng.normal(0.0, 1.0, 400)
    return HistGradientBoostingRegressor(max_iter=30, max_leaf_nodes=8,
                                         random_state=seed).fit(X, y)


def _ensemble(*seeds):
    """A ``MeanEnsemble`` of the CURRENT ``engine.models.ensemble`` class:
    other suites reload that module, and pickling an instance of a stale
    class object fails (joblib checks class identity by import path)."""
    ensemble = importlib.import_module("engine.models.ensemble").MeanEnsemble
    return ensemble([_hgbr(len(CHOOSER_FEATURES), seed) for seed in seeds])


def _artifact(model, role, features):
    """``registry.ModelArtifact`` resolved at call time, for the same reason."""
    cls = importlib.import_module("engine.models.registry").ModelArtifact
    return cls(model=model, role=role, features=features, residuals=np.zeros(3),
               target="synthetic")


def _digest(path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _pool(n: int, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pred = rng.uniform(1.0, 14.0, n)
    return pred, rng.normal(0.0, 2.0 + 0.2 * pred, n)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp("r4_20")
    joblib.dump(_artifact(_hgbr(len(GATE_FEATURES), 2), "gate", GATE_FEATURES), root / "gate.joblib")
    joblib.dump(_artifact(_hgbr(len(FOLD_FEATURES), 5), "size", FOLD_FEATURES), root / "size.joblib")
    ensemble = _ensemble(7, 8)
    joblib.dump(_artifact(ensemble, "chooser", CHOOSER_FEATURES), root / "chooser.joblib")
    pred, res = _pool(3000)
    for role, seed in (("size", 3), ("iv_crush", 4)):
        joblib.dump({"estimator": _hgbr(len(FOLD_FEATURES), seed),
                     "model_id": f"{role}_synthetic", "fold_start": "2026-09-01",
                     "tier3_snapshot": "abc123", "features": list(FOLD_FEATURES),
                     "pool_pred": pred, "pool_res": res}, root / f"{role}_fold.joblib")

    def binding(binding_id, role, name, adapter, order, output):
        return ModelBinding(
            binding_id=binding_id, model_id=f"{role}_synthetic", role=role,
            strategy_id="*", decision_clock_id="legacy.entry_close.v1", adapter=adapter,
            feature_order=order, output_names=(output,),
            members=(ArtifactMember(name="estimator", path=name,
                                    content_hash=_digest(root / name)),))

    release = ModelRelease(release_id="rel-r4-20", deployment_id="dep-1", bindings=(
        binding("b-gate", "gate", "gate.joblib", "joblib-estimator.v1",
                GATE_FEATURES, "gate_score"),
        binding("b-size", "size", "size.joblib", "joblib-estimator.v1",
                FOLD_FEATURES, "pred_abs_move"),
        binding("b-size-fold", "size", "size_fold.joblib", "tier4-serving-fold.v1",
                FOLD_FEATURES, "pred_abs_move"),
        binding("b-crush-fold", "iv_crush", "iv_crush_fold.joblib", "tier4-serving-fold.v1",
                FOLD_FEATURES, "pred_iv_crush_30"),
        binding("b-chooser", "chooser", "chooser.joblib", "joblib-estimator.v1",
                CHOOSER_FEATURES, "chooser_score"),
    ))
    return root, release


def _request(strategy: str) -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-r4-20", calendar_revision="cal-1", strategy_version=strategy,
        deployment_id="dep-1", decision_clock_id="entry-close",
        requested_decision_at=EVENT, snapshot_id="snap-1", mode="replay",
        fill_model={"alpha": 0.5},
    )


def _score(bundle: SourceBundle, strategy: str):
    seen = {}
    record = application.score_one(
        _request(strategy), build_native_score_inputs(bundle),
        observer=lambda item: seen.setdefault(item.receipt.stage, item.output_document))
    return record, seen


# -- legacy expected side ------------------------------------------------------


class _Result:
    """The ScoreResult attributes the legacy gate/chooser methods touch."""

    def __init__(self, **values):
        self.flags: list[str] = []
        self.detail = ""
        self.model_versions: dict = {}
        self.gate_score = self.gate_threshold = self.gate_pass = None
        self.chooser_score = None
        self.structure_spec = None
        self.event_date = pd.Timestamp(EVENT)
        self.as_of = pd.Timestamp(EVENT)
        self.__dict__.update(values)

    def flag(self, name):
        if name not in self.flags:
            self.flags.append(name)


def _served(root, pool) -> tier4.ServingModel:
    stored = joblib.load(root / "size_fold.joblib")
    return tier4.ServingModel(
        estimator=stored["estimator"], model_id=stored["model_id"],
        fold_start=pd.Timestamp(stored["fold_start"]),
        tier3_snapshot=stored["tier3_snapshot"], features=tuple(stored["features"]),
        interval_floor=0.0, pool_pred=pool[0], pool_res=pool[1])


def _legacy_gate(root, row, analog, pool) -> _Result:
    """engine/score.py ``Scorer._score_gate`` on a shell scorer."""
    scorer = object.__new__(Scorer)
    artifact = registry.load_artifact(root / "gate.joblib")
    entry = SimpleNamespace(id="gate_synthetic", threshold=0.5, decision_offset=None)
    scorer.model = lambda *args, **kwargs: (entry, artifact)
    scorer._gate_in_domain = lambda request, features: True
    scorer._serving = lambda fold, produces="pred_abs_move": _served(root, pool)
    result = _Result(exp_pnl_analog=analog["exp_pnl_analog"],
                     win_analog=analog["win_analog"], n_analogs=analog["n_analogs"])
    Scorer._score_gate(scorer, SimpleNamespace(strategy="STR-THRU"), result,
                       pd.DataFrame([row]))
    return result


def _legacy_chooser(root, frame: dict) -> _Result:
    """engine/score.py ``Scorer._score_chooser`` on a shell scorer."""
    scorer = object.__new__(Scorer)
    artifact = registry.load_artifact(root / "chooser.joblib")
    entry = SimpleNamespace(id="dyn_sv_chooser_v1_1", decision_offset=None)
    scorer.model = lambda *args, **kwargs: (entry, artifact)
    scorer._chooser_frame = lambda request, result, features, wanted: dict(frame)
    result = _Result()
    Scorer._score_chooser(scorer, SimpleNamespace(strategy="TWIN-P"), result,
                          pd.DataFrame([ROW]))
    return result


# -- native bundles --------------------------------------------------------------


def _gate_bundle(frozen, row, *, pool=_pool(3000), forecast=True, inference=True,
                 **overrides) -> SourceBundle:
    root, release = frozen
    gate = {"binding_id": "b-gate", "threshold": 0.5}
    if forecast:
        gate["forecast"] = {"binding_id": "b-size-fold"}
    base: dict = dict(
        source_ref="r4-20-gate", strategy="STR-THRU", context=dict(STR_THRU_CONTEXT),
        raw_quotes=STR_THRU_QUOTES, feature_vector=dict(row),
        feature_missing_mask={name: False for name in row},
        model_identity={"driver": {"model_id": "synthetic"}},
        forecast_recipes={"driver_prediction": {"intercept": 6.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:driver"},
        residual_recipe={}, gate_recipe=gate,
        gate_forecast_pool={} if pool is None else {
            "predictions": tuple(pool[0]), "residuals": tuple(pool[1]),
            "interval_floor": 0.0},
        frozen_inference=FrozenInference(root) if inference else None,
        model_release=release if inference else None,
        **ANALOG,
    )
    base.update(overrides)
    return SourceBundle(**base)


def _analog_outputs(record) -> dict:
    values = record.resolved_request
    return {name: values.get(name) for name in ("exp_pnl_analog", "win_analog", "n_analogs")}


# == gap 3: the forecast-analog gate's derived columns ===========================


@pytest.mark.parametrize("n", [3000, 600, 120])
@pytest.mark.parametrize("floor", [0.0, None])
def test_forecast_interval_equals_tier4_interval_for(n, floor):
    pred, res = _pool(n, seed=n)
    res[::97] = np.nan  # the flat pool filters non-finite, the buckets pairs
    queries = [-2.0, 0.5, 3.3, 7.0, 13.9, 40.0, float("nan")]
    native = forecast_interval(queries, pred, res, floor=floor)
    legacy = tier4.interval_for(np.asarray(queries), pred, res, floor=floor)
    for got, want in zip(native, legacy):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("n", [3000, 600, 120])
def test_gate_forecast_columns_equal_legacy_forecast_for_gate(frozen, n):
    root, _ = frozen
    pool = _pool(n)
    scorer = object.__new__(Scorer)
    scorer._serving = lambda fold, produces="pred_abs_move": _served(root, pool)
    legacy = Scorer._forecast_for_gate(scorer, None, _Result(), pd.DataFrame([ROW]))
    served = float(_served(root, pool).predict(pd.DataFrame([ROW]))[0])
    native = gate_forecast_columns(
        served, {"predictions": pool[0], "residuals": pool[1], "interval_floor": 0.0},
        ROW["im"])
    assert native.keys() == legacy.keys()
    for name in legacy:
        assert (native[name] == legacy[name]
                or (math.isnan(native[name]) and math.isnan(legacy[name]))), name


def test_forecast_analog_gate_equals_legacy_bit_for_bit(frozen):
    root, _ = frozen
    record, seen = _score(_gate_bundle(frozen, ROW), "STR-THRU")
    analog = _analog_outputs(record)
    assert analog["n_analogs"] == 3
    legacy = _legacy_gate(root, ROW, analog, _pool(3000))
    assert legacy.gate_score is not None, legacy.flags
    assert seen["gate"]["gate_score"] == legacy.gate_score
    assert seen["gate"]["gate_pass"] == legacy.gate_pass
    assert "MISSING_FEATURES" not in record.reason_codes


def test_flat_pool_gate_equals_legacy(frozen):
    root, _ = frozen
    pool = _pool(600)
    record, seen = _score(_gate_bundle(frozen, ROW, pool=pool), "STR-THRU")
    legacy = _legacy_gate(root, ROW, _analog_outputs(record), pool)
    assert legacy.gate_score is not None, legacy.flags
    assert seen["gate"]["gate_score"] == legacy.gate_score


@pytest.mark.parametrize("case", ["thin_pool", "no_implied", "fold_feature_nan"])
def test_gate_declines_missing_features_exactly_where_legacy_does(frozen, case):
    root, _ = frozen
    row, pool = dict(ROW), _pool(3000)
    if case == "thin_pool":
        pool = _pool(120)  # band NaN -> the gate cannot run
    elif case == "no_implied":
        row["im"] = float("nan")  # forecast_edge NaN (and im itself)
    else:
        row["signed_streak"] = float("inf")  # fold serves NaN: every forecast column NaN
    record, seen = _score(_gate_bundle(frozen, row, pool=pool), "STR-THRU")
    legacy = _legacy_gate(root, row, _analog_outputs(record), pool)
    assert legacy.gate_score is None and legacy.flags == ["MISSING_FEATURES"]
    assert seen["gate"].get("gate_score") is None
    assert "MISSING_FEATURES" in record.reason_codes
    assert not any(code.startswith(("INVALID_", "MISSING_GATE")) for code in record.reason_codes)


def test_undeclared_forecast_source_refuses_by_name(frozen):
    record, seen = _score(_gate_bundle(frozen, ROW, forecast=False), "STR-THRU")
    assert "MISSING_GATE_INPUT:forecast" in record.reason_codes
    assert seen["gate"].get("gate_score") is None
    record, _ = _score(_gate_bundle(frozen, ROW, pool=None), "STR-THRU")
    assert "MISSING_GATE_INPUT:forecast_pool" in record.reason_codes


@pytest.mark.parametrize("declared", [
    {"analog_n": 17.0},  # group still needed: legacy overwrites the declared value
    {"analog_mean": 9.0, "analog_win_rate": 0.9, "analog_n": 17.0},  # group not needed
    {"pred_abs_move_sd": 99.0},
])
def test_declared_base_columns_follow_legacy_group_rule(frozen, declared):
    """Legacy derives a group when the frame lacks any column of it, and then
    writes the whole group; a frame carrying every column keeps its own."""
    root, _ = frozen
    row = {**ROW, **declared}
    record, seen = _score(_gate_bundle(frozen, row), "STR-THRU")
    legacy = _legacy_gate(root, row, _analog_outputs(record), _pool(3000))
    assert legacy.gate_score is not None
    assert seen["gate"]["gate_score"] == legacy.gate_score


def test_planted_defects_in_gate_derivation_are_caught(frozen):
    root, _ = frozen
    record, seen = _score(_gate_bundle(frozen, ROW), "STR-THRU")
    analog = _analog_outputs(record)
    pred, res = _pool(3000)
    # A perturbed fold pool moves the band the column comparator checks ...
    scorer = object.__new__(Scorer)
    scorer._serving = lambda fold, produces="pred_abs_move": _served(root, (pred, res + 1.5))
    legacy = Scorer._forecast_for_gate(scorer, None, _Result(), pd.DataFrame([ROW]))
    served = float(_served(root, (pred, res)).predict(pd.DataFrame([ROW]))[0])
    native = gate_forecast_columns(
        served, {"predictions": pred, "residuals": res, "interval_floor": 0.0}, ROW["im"])
    assert native["pred_abs_move_p10"] != legacy["pred_abs_move_p10"]
    # ... and a perturbed analog output moves the gate score itself.
    other = {**analog, "exp_pnl_analog": analog["exp_pnl_analog"] + 6.0}
    assert _legacy_gate(root, ROW, other, _pool(3000)).gate_score != seen["gate"]["gate_score"]


def test_gate_derivation_runs_under_the_no_fit_guard(frozen):
    root, _ = frozen
    bundle = _gate_bundle(frozen, ROW)
    with no_fit_guard():
        record, seen = _score(bundle, "STR-THRU")
    assert seen["gate"]["gate_score"] == _legacy_gate(
        root, ROW, _analog_outputs(record), _pool(3000)).gate_score


# == gap 4: non-finite frozen-model features ====================================


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None])
def test_non_finite_feature_is_missing_features_naming_every_column(frozen, bad):
    root, release = frozen
    executor = FrozenStageExecutor(inference=FrozenInference(root), release=release,
                                   binding_id="b-size")
    with pytest.raises(FrozenStageRefusal) as error:
        executor.predict({**ROW, "iv30": bad, "mcap_log": float("nan")})
    assert error.value.code == "MISSING_FEATURES"
    assert error.value.reason_codes == ("MISSING_FEATURES",)
    assert error.value.missing_features == ("iv30", "mcap_log")
    with pytest.raises(FrozenStageRefusal) as error:
        executor.predict({**ROW, "iv30": "fifty"})
    assert error.value.code == "INVALID_FEATURE"


def test_champion_driver_non_finite_feature_flags_only_missing_features(frozen):
    """Legacy ``_score_model``: MISSING_FEATURES and an early return -- no
    second forecast-output refusal."""
    row = {**ROW, "iv30": float("inf")}
    record, seen = _score(_gate_bundle(frozen, row, forecast_recipes={
        "driver_prediction": {"binding_id": "b-size"}},
        model_artifact_refs={"driver_prediction": "binding:b-size"}), "STR-THRU")
    assert seen["forecast"].get("driver_prediction") is None
    assert "MISSING_FEATURES" in record.reason_codes
    assert not any(code.startswith(("INVALID_FEATURE", "MISSING_FORECAST_OUTPUT"))
                   for code in record.reason_codes)
    # Same readiness consequence as the legacy flag: the row is refused.
    assert flags_refuse(("MISSING_FEATURES",)) and record.validation_status == "refused"


def _twin_bundle(frozen, row, *, recipes=None, chooser=True, inference=True,
                 **overrides) -> SourceBundle:
    root, release = frozen
    pool = pnl_sim.ResidualPool(_paired_history())
    recipes = recipes or {
        "forecast_abs_move": {"intercept": 6.0, "coefficients": {}},
        "pred_iv_crush": {"intercept": -20.0, "coefficients": {}},
    }
    base: dict = dict(
        source_ref="r4-20-twin", strategy="TWIN-P", context=dict(TWIN_CONTEXT),
        raw_quotes=TWIN_QUOTES, feature_vector=dict(row),
        feature_missing_mask={name: False for name in row},
        model_identity={"size": {"model_id": "synthetic"}},
        forecast_recipes=recipes,
        model_artifact_refs={name: f"sha256:{name}" for name in recipes},
        residual_recipe={"mode": "planned_exit", "pre_iv30": 40.0},
        paired_residual_rows=pool.documented_population(pool.before(pd.Timestamp(EVENT))),
        analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0},
        chooser_recipe={"binding_id": "b-chooser"} if chooser else {},
        frozen_inference=FrozenInference(root) if inference else None,
        model_release=release if inference else None,
    )
    base.update(overrides)
    return SourceBundle(**base)


def _paired_history() -> pd.DataFrame:
    rng = np.random.default_rng(29)
    days = pd.date_range("2025-03-01", periods=500, freq="D")
    return pd.DataFrame([{"event_date": day, "ticker": f"T{index % 7}",
                          "pred_abs_move": 3.0 + (index * 7919 % 400) / 50.0,
                          "err_move": float(rng.normal(0.0, 3.0)),
                          "err_crush": float(rng.normal(-5.0, 12.0))}
                         for index, day in enumerate(days)])


def test_fold_sized_forecast_non_finite_feature_declines_no_forecast(frozen):
    """Legacy ``_size_from_forecast``: ``ServingModel.predict`` answers NaN
    with no flag, and sizing declines NO_FORECAST."""
    row = {**ROW, "iv30": float("nan")}
    record, seen = _score(_twin_bundle(frozen, row, chooser=False, recipes={
        "forecast_abs_move": {"binding_id": "b-size-fold"},
        "pred_iv_crush": {"intercept": -20.0, "coefficients": {}}}), "TWIN-P")
    assert seen["forecast"].get("forecast_abs_move") is None
    assert "NO_FORECAST" in record.reason_codes
    assert "MISSING_FEATURES" not in record.reason_codes
    assert not any(code.startswith("MISSING_FORECAST_OUTPUT") for code in record.reason_codes)


def test_fold_crush_non_finite_feature_leaves_the_simulation_undetermined(frozen):
    """Legacy ``_crush_forecast`` -> ``None`` -> ``expected_pnl`` returns
    ``None`` silently: no exp_pnl_sim, and no refusal for it."""
    row = {**ROW, "mcap_log": float("-inf")}
    record, seen = _score(_twin_bundle(frozen, row, chooser=False, recipes={
        "forecast_abs_move": {"intercept": 6.0, "coefficients": {}},
        "pred_iv_crush": {"binding_id": "b-crush-fold"}}), "TWIN-P")
    assert seen["forecast"].get("pred_iv_crush") is None
    assert record.resolved_request.get("exp_pnl_sim") is None
    assert not any(code.startswith(("MISSING_SIMULATION_INPUT", "MISSING_FORECAST_OUTPUT",
                                    "MISSING_FEATURES")) for code in record.reason_codes)
    # Control: the same bundle with finite features does simulate.
    finite, _ = _score(_twin_bundle(frozen, ROW, chooser=False, recipes={
        "forecast_abs_move": {"intercept": 6.0, "coefficients": {}},
        "pred_iv_crush": {"binding_id": "b-crush-fold"}}), "TWIN-P")
    assert finite.resolved_request.get("exp_pnl_sim") is not None


# == gap 3, sizing side: the served fold's own band ============================


def test_size_fold_band_equals_forecast_interval_on_the_declared_pool(frozen):
    """The forecast stage derives ``forecast_p10``/``_p90``/``_sd`` from the
    size fold's own declared pool -- the same helper and arguments legacy
    ``_size_from_forecast``'s ``served.interval`` runs."""
    pool = _pool(3000)
    declared = {"predictions": tuple(pool[0]), "residuals": tuple(pool[1]),
                "interval_floor": 0.0}
    _record, seen = _score(
        _twin_bundle(frozen, ROW, chooser=False, forecast_pool=declared), "TWIN-P")
    forecast = seen["forecast"]["forecast_abs_move"]
    p10, p90, sd, _ = forecast_interval(
        [forecast], declared["predictions"], declared["residuals"],
        floor=declared["interval_floor"])
    assert seen["forecast"]["forecast_p10"] == float(p10[0])
    assert seen["forecast"]["forecast_p90"] == float(p90[0])
    assert seen["forecast"]["forecast_sd"] == float(sd[0])


def test_undeclared_size_pool_leaves_the_band_absent(frozen):
    """No declared pool: the band is simply absent (legacy never fabricates
    one) -- not NaN-filled, and the forecast itself is still produced."""
    _record, seen = _score(_twin_bundle(frozen, ROW, chooser=False), "TWIN-P")
    assert "forecast_abs_move" in seen["forecast"]
    for name in ("forecast_p10", "forecast_p90", "forecast_sd"):
        assert name not in seen["forecast"]


# == gap 5: the DYN-SV chooser through a frozen recipe ============================


def _legacy_chooser_frame(record, row) -> dict:
    """``_chooser_frame``'s columns for this synthetic champion, by legacy's
    own formulas (engine/score.py:3415-3424, :3486-3490)."""
    sim = record.resolved_request.get("exp_pnl_sim")
    frame = {name: row.get(name, float("nan")) for name in ("mean_prior_abs_move",
                                                             "iv30", "spy_vol20")}
    frame["exp_pnl_sim"] = float("nan") if sim is None else float(sim)
    frame["exp_pnl_sim_select"] = frame["exp_pnl_sim"]
    for member in DYNAMIC_MENU:
        frame[f"is_{member.lower().replace('-', '_')}"] = 1.0 if member == "TWIN-P" else 0.0
    frame["quote_repaired"] = 0.0
    frame["wide_market"] = 1.0 if "WIDE_MARKET" in record.reason_codes else 0.0
    return frame


def test_chooser_frozen_recipe_equals_legacy_score_chooser(frozen):
    root, _ = frozen
    record, seen = _score(_twin_bundle(frozen, ROW), "TWIN-P")
    assert record.resolved_request.get("exp_pnl_sim") is not None, record.reason_codes
    legacy = _legacy_chooser(root, _legacy_chooser_frame(record, ROW))
    assert legacy.chooser_score is not None, legacy.flags
    assert seen["chooser"]["chooser_score"] == legacy.chooser_score
    assert record.forecasts["chooser_score"] == legacy.chooser_score


def test_chooser_runs_under_the_no_fit_guard(frozen):
    root, _ = frozen
    bundle = _twin_bundle(frozen, ROW)
    with no_fit_guard():
        record, seen = _score(bundle, "TWIN-P")
    assert seen["chooser"]["chooser_score"] == _legacy_chooser(
        root, _legacy_chooser_frame(record, ROW)).chooser_score


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_chooser_declines_with_the_advisory_flag_like_legacy(frozen, bad):
    root, _ = frozen
    row = {**ROW, "spy_vol20": bad}
    record, seen = _score(_twin_bundle(frozen, row), "TWIN-P")
    legacy = _legacy_chooser(root, _legacy_chooser_frame(record, row))
    assert legacy.chooser_score is None and legacy.flags == ["CHOOSER_MISSING_FEATURES"]
    assert seen["chooser"].get("chooser_score") is None
    assert "CHOOSER_MISSING_FEATURES" in record.reason_codes
    assert "MISSING_FEATURES" not in record.reason_codes
    assert not flags_refuse(("CHOOSER_MISSING_FEATURES",))


def test_chooser_absent_feature_declines_too(frozen):
    row = {key: value for key, value in ROW.items() if key != "spy_vol20"}
    record, seen = _score(_twin_bundle(frozen, row), "TWIN-P")
    assert seen["chooser"].get("chooser_score") is None
    assert "CHOOSER_MISSING_FEATURES" in record.reason_codes


def test_chooser_planted_defect_different_artifact_is_detected(frozen, tmp_path):
    root, release = frozen
    other = _ensemble(17, 18)
    joblib.dump(_artifact(other, "chooser", CHOOSER_FEATURES), tmp_path / "chooser.joblib")
    for name in ("size_fold.joblib", "iv_crush_fold.joblib", "gate.joblib", "size.joblib"):
        (tmp_path / name).write_bytes((root / name).read_bytes())
    swapped = ModelRelease(release_id=release.release_id, deployment_id="dep-1", bindings=tuple(
        ModelBinding(**{**binding.__dict__, "members": (ArtifactMember(
            name="estimator", path="chooser.joblib",
            content_hash=_digest(tmp_path / "chooser.joblib")),)})
        if binding.role == "chooser" else binding for binding in release.bindings))
    record, seen = _score(_twin_bundle(frozen, ROW, frozen_inference=FrozenInference(tmp_path),
                                       model_release=swapped), "TWIN-P")
    legacy = _legacy_chooser(root, _legacy_chooser_frame(record, ROW))
    assert seen["chooser"]["chooser_score"] != legacy.chooser_score


def test_chooser_declared_but_unresolved_refuses_model_not_ready(frozen):
    record, seen = _score(_twin_bundle(frozen, ROW, inference=False), "TWIN-P")
    assert "MODEL_NOT_READY" in record.reason_codes
    assert seen["chooser"].get("chooser_score") is None


def test_malformed_chooser_recipes_are_refused_at_build(frozen):
    with pytest.raises(ValueError, match="not a DYN-SV menu candidate"):
        build_native_score_inputs(_gate_bundle(frozen, ROW,
                                               chooser_recipe={"binding_id": "b-chooser"}))
    with pytest.raises(ValueError, match="role"):
        build_native_score_inputs(_twin_bundle(frozen, ROW,
                                               chooser_recipe={"binding_id": "b-gate"}))
    with pytest.raises(ValueError, match="unsupported recipe fields"):
        build_native_score_inputs(_twin_bundle(
            frozen, ROW, chooser_recipe={"binding_id": "b-chooser", "intercept": 1.0}))
