"""The DYN-SV chooser's 67-column vector derived natively, against legacy.

Expected side: the real legacy ``Scorer._chooser_frame`` (and through it
``_live_chain_depth``/``_chain_depth``, ``_n_admissible_for``,
``_chooser_analogs`` over the real ``_chooser_analog_pool`` parquet loader,
``_chooser_schematics`` and real ``tier4.ServingModel`` folds), run on a
scorer shell. Actual side: ``native_chooser.derive_chooser_columns`` over the
block ``build_native_score_inputs`` builds from a ``SourceBundle`` that
declares only primitive features, frozen state and source rows.

* one test per column group, over randomized synthetic cases and the edge
  cases legacy special-cases, each compared bit for bit (NaN where legacy
  has NaN);
* the full path: ``application.score_one`` on a menu candidate with a
  67-feature chooser champion; the vector the stage hands the frozen
  executor equals legacy's frame for all 67 columns, and ``chooser_score``
  equals legacy ``Scorer._score_chooser`` with its real ``_chooser_frame``;
* planted defects (a reordered pool, a perturbed fold pool, another
  admissible table, a wrong key) that each comparator catches.
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

import engine.score as score_mod
from engine.data.features import tier4
from engine.models import registry
from engine.score import (
    CHOOSER_ANALOG_POOL,
    DYNAMIC_MENU,
    Scorer,
    _mean_relative_spread,
)
from engine.v2.contracts import ScoreRequest
from engine.v2.models import ArtifactMember, FrozenInference, ModelBinding, ModelRelease
from engine.v2.models.admissible_table import (
    legacy_n_admissible_table,
    make_admissible_depth_table,
)
from engine.v2.models.lineage import DataDependency, Lineage
from engine.v2.models.training.chooser_pool import build_chooser_analog_pool_artifact
from engine.v2.scoring import application
from engine.v2.scoring.native_chooser import derive_chooser_columns
from engine.v2.scoring.native_chooser_features import (
    ANALOG_COLUMNS,
    BREAKEVEN_COLUMNS,
    PRODUCER_COLUMNS,
    SHAPE_COLUMNS,
)
from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning"),
              pytest.mark.filterwarnings("ignore::RuntimeWarning")]

NAN = float("nan")
EVENT = "2026-09-16"
EXPIRY = "2026-09-18"
FOLD_FEATURES = ("mean_prior_abs_move", "iv30", "signed_streak", "mcap_log")
PRIMITIVES = ("mean_prior_abs_move", "ema12r_abs", "signed_streak",
              "mean_prior_or_implied", "mcap_log", "or_implied", "or_rvol30",
              "spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol20",
              "spy_vol5", "spy_vol60", "spy_vol252", "spy_vol20_rel252", "dte_entry")
GROUPS = {
    "direct": ("exp_pnl_sim", "exp_pnl_sim_select", "entry_cost_pct", "quote_repaired",
               "wide_market", *(f"is_{m.lower().replace('-', '_')}" for m in DYNAMIC_MENU)),
    "size_band": ("pred_abs_move", "pred_abs_move_sd", "pred_abs_move_p10",
                  "pred_abs_move_p90", "pred_abs_move_resid_n", "tier4_pred_abs_move_sd"),
    "geometry": ("half_width_pct_spot", "width_over_forecast", "anchor_over_spot",
                 "n_legs", "rel_spread"),
    "n_admissible": ("n_admissible",),
    "knn_analog": ANALOG_COLUMNS,
    "schematics": (*BREAKEVEN_COLUMNS, *SHAPE_COLUMNS),
    "producers": tuple(c for cols in PRODUCER_COLUMNS.values() for c in cols),
    "forecast_edge": ("tier4_forecast_edge",),
    "primitives": PRIMITIVES,
}
CHOOSER_FEATURES = tuple(sorted(c for cols in GROUPS.values() for c in cols))
LINEAGE = Lineage(data=(DataDependency(table="synthetic.menu", end_exclusive="2030-01-01"),))
POOL_ID = "synthetic.menu7"


def same(a: float, b: float) -> bool:
    """Bit-for-bit float equality, NaN equal to NaN."""
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)


def diff(legacy: dict, native: dict, names) -> dict:
    return {n: (legacy.get(n, "absent"), native.get(n, "absent")) for n in names
            if n not in legacy or n not in native or not same(legacy[n], native[n])}


# -- frozen synthetic models ---------------------------------------------------


def _hgbr(n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (400, n_features)) * 3.0 + 5.0
    y = X @ np.linspace(0.1, 0.5, n_features) + rng.normal(0.0, 1.0, 400)
    return HistGradientBoostingRegressor(max_iter=30, max_leaf_nodes=8,
                                         random_state=seed).fit(X, y)


def _digest(path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _pool(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pred = rng.uniform(1.0, 14.0, n)
    return pred, rng.normal(0.0, 2.0 + 0.2 * pred, n)


FOLDS = {  # output -> (role, estimator seed, pool, interval floor)
    "pred_im_t1_d14": ("implied_t1", 21, _pool(2600, 5), 0.0),
    "pred_runup_abs_move_d14": ("runup_move", 22, _pool(900, 6), None),
}
SIZE_POOL = _pool(3000, 11)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp("chooser67")
    ensemble = importlib.import_module("engine.models.ensemble").MeanEnsemble
    artifact_cls = importlib.import_module("engine.models.registry").ModelArtifact
    joblib.dump(artifact_cls(model=ensemble([_hgbr(len(CHOOSER_FEATURES), s) for s in (7, 8)]),
                             role="chooser", features=CHOOSER_FEATURES,
                             residuals=np.zeros(3), target="synthetic"),
                root / "chooser.joblib")
    bindings = [ModelBinding(
        binding_id="b-chooser", model_id="chooser_synthetic", role="chooser",
        strategy_id="*", decision_clock_id="legacy.entry_close.v1",
        adapter="joblib-estimator.v1", feature_order=CHOOSER_FEATURES,
        output_names=("chooser_score",),
        members=(ArtifactMember(name="estimator", path="chooser.joblib",
                                content_hash=_digest(root / "chooser.joblib")),))]
    for output, (role, seed, pool, _floor) in FOLDS.items():
        name = f"{role}_fold.joblib"
        joblib.dump({"estimator": _hgbr(len(FOLD_FEATURES), seed),
                     "model_id": f"{role}_synthetic", "fold_start": "2026-09-01",
                     "tier3_snapshot": "abc123", "features": list(FOLD_FEATURES),
                     "pool_pred": pool[0], "pool_res": pool[1]}, root / name)
        bindings.append(ModelBinding(
            binding_id=f"b-{role}", model_id=f"{role}_synthetic", role=role,
            strategy_id="*", decision_clock_id="legacy.entry_close.v1",
            adapter="tier4-serving-fold.v1", feature_order=FOLD_FEATURES,
            output_names=(output,),
            members=(ArtifactMember(name="estimator", path=name,
                                    content_hash=_digest(root / name)),)))
    return root, ModelRelease(release_id="rel-chooser67", deployment_id="dep-1",
                              bindings=tuple(bindings))


# -- the chooser analog population ---------------------------------------------


def pool_frame(seed: int = 3, n: int = 1400, ties: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entry = pd.Timestamp("2024-01-02") + pd.to_timedelta(rng.integers(0, 980, n), unit="D")
    frame = pd.DataFrame({
        "strategy": rng.choice(DYNAMIC_MENU, n),
        "entry_date": entry,
        "exit_date": entry + pd.to_timedelta(rng.integers(1, 20, n), unit="D"),
        "pnl": rng.normal(0.0, 150.0, n),
        "exp_pnl_sim": rng.normal(0.0, 1.0, n),
        "width_over_forecast": rng.uniform(0.2, 2.0, n),
        "n_legs": rng.choice([3.0, 5.0, 7.0], n),
        "anchor_over_spot": rng.uniform(0.95, 1.02, n),
        "rel_spread": rng.uniform(0.01, 0.4, n),
    })
    if ties:  # identical geometry, distinct P&L: which K win is decided by ORDER
        for column in ("exp_pnl_sim", "width_over_forecast", "anchor_over_spot",
                       "rel_spread"):
            frame[column] = 0.5
        frame["n_legs"] = 7.0
    frame.loc[4, "pnl"] = np.nan
    frame.loc[6, "rel_spread"] = np.nan
    return frame.sort_values(["strategy", "entry_date"]).reset_index(drop=True)


def analog_artifact(frame):
    return build_chooser_analog_pool_artifact(
        frame.to_dict("records"), pool_id=POOL_ID, cutoff="2030-01-01", lineage=LINEAGE)


# -- synthetic cases ---------------------------------------------------------------


def make_case(seed: int, **overrides) -> dict:
    """One menu candidate: a priced put ladder on a listed chain, the sizing
    forecast, the primitive features and the dates."""
    rng = np.random.default_rng(seed)
    spot = float(rng.uniform(20.0, 400.0))
    step = float(rng.choice([s for s in (0.5, 1.0, 2.5, 5.0) if s <= spot / 15.0]))
    anchor = float(np.floor(spot / step) * step)
    grid = anchor + step * np.arange(-12, 13)
    quotes = {}
    for strike in grid:
        mid = max(anchor - strike, 0.0) + float(rng.uniform(0.05, 3.0))
        spread = 0.0 if rng.random() < 0.15 else float(rng.uniform(0.01, 0.3))
        bid = 0.0 if rng.random() < 0.2 else max(mid - spread, 0.0)
        quotes[("P", float(strike), EXPIRY)] = {"bid": bid, "ask": bid + spread + 0.05}
        quotes[("C", float(strike), EXPIRY)] = {"bid": 1.0, "ask": 1.1}
        quotes[("P", float(strike), "2026-10-16")] = {"bid": 1.0, "ask": 1.2}
    offsets = sorted(set(int(k) for k in rng.integers(1, 6, 3)))
    legs = [("atm", anchor, "sell" if rng.random() < 0.6 else "buy",
             float(rng.choice([0.0, 1.0, 2.0])))]
    for i, k in enumerate(offsets, start=1):
        side = "buy" if rng.random() < 0.5 else "sell"
        for name, strike in ((f"up{i}", anchor + k * step), (f"dn{i}", anchor - k * step)):
            legs.append((name, strike, side, float(rng.choice([1.0, 2.0]))))
    priced = []
    for name, strike, side, qty in legs:
        quote = quotes[("P", float(strike), EXPIRY)]
        priced.append({"name": name, "right": "P", "side": side, "quantity": qty,
                       "strike": float(strike), "expiry": EXPIRY,
                       "bid": quote["bid"], "ask": quote["ask"]})
    features = {name: float(rng.normal(1.0, 0.5)) for name in PRIMITIVES}
    features.update({"iv30": float(rng.uniform(20, 90)), "dte_entry": 2.0,
                     "mcap_log": float(rng.uniform(20, 26))})
    case = {
        "strategy": str(rng.choice(DYNAMIC_MENU)), "spot": spot, "legs": priced,
        "quotes": quotes, "entry_cost": float(rng.uniform(-3.0, 6.0)),
        "forecast_abs_move": float(rng.uniform(1.5, 13.0)),
        "exp_pnl_sim": float(rng.normal(0.0, 1.0)), "features": features,
        "flags": ["WIDE_MARKET"] if rng.random() < 0.3 else [],
        "entry_date": str(rng.choice(["2024-01-20", "2025-06-03", EVENT])),
    }
    case.update(overrides)
    return case


EDGE_CASES = {
    "nan_forecast": dict(forecast_abs_move=NAN),
    "zero_forecast": dict(forecast_abs_move=0.0),
    "no_sim": dict(exp_pnl_sim=None),
    "no_cost": dict(entry_cost=None),
    "cheap_cost": dict(entry_cost=0.03),
    "zero_spot": dict(spot=0.0),
    "early_entry": dict(entry_date="2024-01-05"),
    "no_or_implied": dict(features_drop=("or_implied",)),
    "no_fold_feature": dict(features_drop=("iv30",)),
    "nan_fold_feature": dict(features_nan=("signed_streak",)),
}


def edge_case(name: str) -> dict:
    overrides = dict(EDGE_CASES[name])
    drop, nans = overrides.pop("features_drop", ()), overrides.pop("features_nan", ())
    case = make_case(101 + len(name), **overrides)
    for column in drop:
        case["features"].pop(column)
    for column in nans:
        case["features"][column] = NAN
    return case


# -- the legacy expected side ------------------------------------------------------


def _served(output: str | None, pools=None) -> tier4.ServingModel:
    if output is None:
        pred, res = SIZE_POOL if pools is None else pools
        return tier4.ServingModel(estimator=None, model_id="size_synthetic",
                                  fold_start=pd.Timestamp("2026-09-01"),
                                  tier3_snapshot="abc123", features=FOLD_FEATURES,
                                  interval_floor=0.0, pool_pred=pred, pool_res=res)
    role, seed, pool, floor = FOLDS[output]
    return tier4.ServingModel(estimator=_hgbr(len(FOLD_FEATURES), seed),
                              model_id=f"{role}_synthetic",
                              fold_start=pd.Timestamp("2026-09-01"), tier3_snapshot="abc123",
                              features=FOLD_FEATURES, interval_floor=floor,
                              pool_pred=pool[0], pool_res=pool[1])


class _Result:
    def __init__(self, case: dict):
        legs = case["legs"]
        self.flags = list(case["flags"])
        self.detail = ""
        self.model_versions: dict = {}
        self.chooser_score = None
        self.structure_spec = None
        self.strategy = case["strategy"]
        self.legs = [{"name": leg["name"], "side": leg["side"], "right": leg["right"],
                      "qty": leg["quantity"], "strike": leg["strike"],
                      "expiry": leg["expiry"], "bid": leg["bid"], "ask": leg["ask"]}
                     for leg in legs]
        self.strike = float(legs[0]["strike"]) if legs else None
        self.expiry = pd.Timestamp(legs[0]["expiry"]) if legs else None
        self.spot = case["spot"]
        self.entry_cost = case["entry_cost"]
        self.forecast_abs_move = case["forecast_abs_move"]
        self.exp_pnl_sim = case["exp_pnl_sim"]
        self.dte_entry = case["features"].get("dte_entry")
        self.rel_spread = _mean_relative_spread(SimpleNamespace(
            legs=[SimpleNamespace(bid=leg["bid"], ask=leg["ask"]) for leg in legs]))
        self.entry_date = pd.Timestamp(case["entry_date"])
        self.event_date = pd.Timestamp(EVENT)
        self.as_of = pd.Timestamp(EVENT)
        self._entry_rows = pd.DataFrame([
            {"right": right, "strike": strike, "expiry": pd.Timestamp(expiry), **quote}
            for (right, strike, expiry), quote in case["quotes"].items()])

    def flag(self, name):
        if name not in self.flags:
            self.flags.append(name)


def legacy_features(case) -> pd.DataFrame:
    """The legacy feature frame. ``entry_cost_pct`` is written the way
    ``Scorer._entry_features`` writes it (engine/score.py:2100), because that
    method needs a full feature context to run."""
    built = pd.DataFrame([{**case["features"], "entry_cost": case["entry_cost"],
                           "spot_entry": case["spot"]}])
    built["entry_cost_pct"] = (
        pd.to_numeric(built["entry_cost"], errors="coerce")
        / pd.to_numeric(built["spot_entry"], errors="coerce") * 100.0
    )
    return built


def legacy_scorer(tmp_path, monkeypatch, *, frame=None, size_pool=None):
    if frame is not None:
        frame.to_parquet(tmp_path / CHOOSER_ANALOG_POOL, index=False)
    monkeypatch.setattr(score_mod.paths, "FEATURES", tmp_path)
    scorer = object.__new__(Scorer)
    scorer._chooser_pool = score_mod._UNSET
    scorer._serving = lambda fold, produces="pred_abs_move": _served(
        None if produces == "pred_abs_move" else produces, size_pool)
    scorer._regime_extra = lambda request, result, name: NAN
    return scorer


def legacy_frame(scorer, case) -> dict:
    return Scorer._chooser_frame(scorer, SimpleNamespace(strategy=case["strategy"]),
                                 _Result(case), legacy_features(case), CHOOSER_FEATURES)


# -- the native side -----------------------------------------------------------------


def bundle(frozen, case, *, analog=None, table=None, size_pool=SIZE_POOL,
           producers=True, recipe_extra=None) -> SourceBundle:
    root, release = frozen
    recipe = {"binding_id": "b-chooser"}
    if producers:
        recipe["producers"] = {output: {"binding_id": f"b-{role}"}
                               for output, (role, *_rest) in FOLDS.items()}
    if analog is not None:
        recipe["analog_pool"] = {"pool_id": analog.pool_id, "cutoff": analog.cutoff}
    if table is not None:
        recipe["admissible_table"] = {"table_id": table.table_id, "version": table.version}
    recipe.update(recipe_extra or {})
    pools = {output: {"predictions": tuple(pool[0]), "residuals": tuple(pool[1]),
                      "interval_floor": floor}
             for output, (_role, _seed, pool, floor) in FOLDS.items()}
    if size_pool is not None:
        pools["pred_abs_move"] = {"predictions": tuple(size_pool[0]),
                                  "residuals": tuple(size_pool[1]), "interval_floor": 0.0}
    return SourceBundle(
        source_ref="chooser67", strategy=case["strategy"],
        context={"ticker": "AAA", "event_date": EVENT, "entry_date": case["entry_date"],
                 "exit_date": "2026-09-09", "expiry": EXPIRY, "spot": case["spot"] or 100.0},
        raw_quotes=case["quotes"], feature_vector=dict(case["features"]),
        feature_missing_mask={name: False for name in case["features"]},
        model_identity={"size": {"model_id": "synthetic"}},
        forecast_recipes={"forecast_abs_move": {"intercept": 6.0, "coefficients": {}},
                          "pred_iv_crush": {"intercept": -20.0, "coefficients": {}}},
        model_artifact_refs={"forecast_abs_move": "sha256:a", "pred_iv_crush": "sha256:b"},
        residual_recipe={}, analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0},
        chooser_recipe=recipe, chooser_fold_pools=pools,
        chooser_analog_pool=analog, chooser_admissible_table=table,
        frozen_inference=FrozenInference(root), model_release=release,
    )


def native_frame(frozen, case, **declared) -> tuple[dict | None, list[str]]:
    inputs = build_native_score_inputs(bundle(frozen, case, **declared))
    values = {"legs": tuple(case["legs"]), "spot": case["spot"],
              "entry_cost": case["entry_cost"],
              "forecast_abs_move": case["forecast_abs_move"],
              "exp_pnl_sim": case["exp_pnl_sim"]}
    facts = {**case["features"], "entry_date": case["entry_date"]}
    flags = list(case["flags"])
    derived = derive_chooser_columns(inputs.chooser, facts, case["strategy"], values,
                                     inputs.context["quotes"], flags)
    if derived is None:
        return None, flags
    return {**derived, **{n: case["features"].get(n, NAN) for n in PRIMITIVES}}, flags


# -- per-group proofs ------------------------------------------------------------------


@pytest.fixture(scope="module")
def analog():
    return analog_artifact(pool_frame())


CASES = [*(f"seed{seed}" for seed in range(12)), *EDGE_CASES]


def _case(name: str) -> dict:
    return make_case(int(name[4:])) if name.startswith("seed") else edge_case(name)


def test_the_groups_cover_legacy_chooser_frame_exactly(frozen, analog, tmp_path, monkeypatch):
    scorer = legacy_scorer(tmp_path, monkeypatch, frame=pool_frame())
    legacy = legacy_frame(scorer, make_case(0))
    assert len(CHOOSER_FEATURES) == 67
    assert set(legacy) == set(CHOOSER_FEATURES)


@pytest.mark.parametrize("group", [g for g in GROUPS if g != "primitives"])
@pytest.mark.parametrize("name", CASES)
def test_each_column_group_equals_legacy_bit_for_bit(frozen, analog, tmp_path, monkeypatch,
                                                     group, name):
    case = _case(name)
    scorer = legacy_scorer(tmp_path, monkeypatch, frame=pool_frame())
    legacy = legacy_frame(scorer, case)
    native, flags = native_frame(frozen, case, analog=analog,
                                 table=legacy_n_admissible_table())
    assert native is not None, flags
    assert diff(legacy, native, GROUPS[group]) == {}


def test_the_cases_exercise_both_sides_of_every_group(frozen, analog, tmp_path, monkeypatch):
    """Control: across the cases each group has finite values somewhere and
    (except the always-defined ones) NaN somewhere, so equality is not
    vacuous NaN-equals-NaN."""
    scorer = legacy_scorer(tmp_path, monkeypatch, frame=pool_frame())
    frames = [legacy_frame(scorer, _case(name)) for name in CASES]
    for group, names in GROUPS.items():
        for column in names:
            assert any(math.isfinite(f[column]) for f in frames), column
    for column in ("analog_mean", "breakeven_down_room_forecast",
                   "pred_im_t1_d14", "pred_abs_move_p10", "width_over_forecast"):
        assert any(math.isnan(f[column]) for f in frames), column
    depths = {f["n_admissible"] for f in frames}
    assert len(depths) > 2  # several depth buckets, not only the fallback


def test_knn_block_keeps_legacy_order_on_ties(frozen, tmp_path, monkeypatch):
    """With identical geometry, WHICH 25 rows are nearest is decided by
    position (``argpartition``): the frozen order reproduces legacy, and a
    reordered pool (planted defect) does not."""
    ties = pool_frame(ties=True)
    case = make_case(4, exp_pnl_sim=0.5, entry_date=EVENT)
    scorer = legacy_scorer(tmp_path, monkeypatch, frame=ties)
    legacy = legacy_frame(scorer, case)
    assert math.isfinite(legacy["analog_mean"])
    native, _ = native_frame(frozen, case, analog=analog_artifact(ties),
                             table=legacy_n_admissible_table())
    assert diff(legacy, native, ANALOG_COLUMNS) == {}
    shuffled = analog_artifact(ties.sample(frac=1.0, random_state=1))
    planted, _ = native_frame(frozen, case, analog=shuffled,
                              table=legacy_n_admissible_table())
    assert diff(legacy, planted, ANALOG_COLUMNS) != {}


def test_undeclared_state_leaves_legacy_missing_state_nan(frozen, tmp_path, monkeypatch):
    """No pool file / no fold in legacy == nothing declared natively: the
    same columns are NaN (and the chooser then declines)."""
    case = make_case(2, entry_date=EVENT)
    scorer = legacy_scorer(tmp_path, monkeypatch, frame=None)
    scorer._serving = lambda fold, produces="pred_abs_move": (_ for _ in ()).throw(
        FileNotFoundError(produces))
    legacy = legacy_frame(scorer, case)
    native, _ = native_frame(frozen, case, size_pool=None, producers=False,
                             table=legacy_n_admissible_table())
    groups = ("knn_analog", "producers", "size_band", "n_admissible", "schematics")
    names = [n for g in groups for n in GROUPS[g]]
    assert all(math.isnan(legacy[n]) for n in (*ANALOG_COLUMNS, "pred_abs_move_sd"))
    assert diff(legacy, native, names) == {}


@pytest.mark.parametrize("planted", ["size_pool", "admissible_table", "producer_pool"])
def test_planted_defects_are_caught(frozen, analog, tmp_path, monkeypatch, planted):
    case = make_case(5, entry_date=EVENT)
    scorer = legacy_scorer(tmp_path, monkeypatch, frame=pool_frame())
    legacy = legacy_frame(scorer, case)
    kwargs = dict(analog=analog, table=legacy_n_admissible_table())
    if planted == "size_pool":
        pred, res = SIZE_POOL
        kwargs["size_pool"] = (pred, res + 0.25)
        group = "size_band"
    elif planted == "admissible_table":
        base = legacy_n_admissible_table()
        kwargs["table"] = make_admissible_depth_table(
            table_id=base.table_id, version="v2-planted",
            breakpoints=[(f, v + 1.0) for f, v in base.breakpoints],
            fallback=base.fallback + 1.0, provenance="planted", lineage=base.lineage)
        group = "n_admissible"
    else:
        original = FOLDS["pred_im_t1_d14"]
        FOLDS["pred_im_t1_d14"] = (*original[:2], (original[2][0], original[2][1] * 1.5),
                                   original[3])
        try:
            native, _ = native_frame(frozen, case, **kwargs)
        finally:
            FOLDS["pred_im_t1_d14"] = original
        assert diff(legacy, native, GROUPS["producers"]) != {}
        return
    native, _ = native_frame(frozen, case, **kwargs)
    assert diff(legacy, native, GROUPS[group]) != {}


@pytest.mark.parametrize("recipe_extra, declared", [
    ({"analog_pool": {"pool_id": POOL_ID, "cutoff": "2029-01-01"}}, "analog"),
    ({"analog_pool": {"pool_id": POOL_ID, "cutoff": "2030-01-01",
                      "content_hash": "sha256:" + "0" * 64}}, "analog"),
    ({"analog_pool": {"pool_id": POOL_ID, "cutoff": "2030-01-01"}}, "none"),
    ({"admissible_table": {"table_id": "dyn_sv.n_admissible_by_depth",
                           "version": "v9"}}, "table"),
])
def test_wrong_or_missing_frozen_state_is_model_not_ready(frozen, analog, recipe_extra,
                                                          declared):
    case = make_case(6)
    kwargs = {"analog": analog} if declared == "analog" else {}
    if declared == "table":
        kwargs["table"] = legacy_n_admissible_table()
    native, flags = native_frame(frozen, case, recipe_extra=recipe_extra, **kwargs)
    assert native is None and "MODEL_NOT_READY" in flags


def test_malformed_chooser_declarations_are_refused_at_build(frozen, analog):
    case = make_case(7)
    with pytest.raises(ValueError, match="unsupported outputs"):
        build_native_score_inputs(bundle(frozen, case, recipe_extra={
            "producers": {"pred_iv_crush_30": {"binding_id": "b-implied_t1"}}}))
    with pytest.raises(ValueError, match="role"):
        build_native_score_inputs(bundle(frozen, case, recipe_extra={
            "producers": {"pred_im_t1_d14": {"binding_id": "b-runup_move"}}}))
    plain = bundle(frozen, case)
    with pytest.raises(ValueError, match="ChooserAnalogPoolArtifact"):
        build_native_score_inputs(SourceBundle(**{
            **plain.__dict__, "chooser_analog_pool": legacy_n_admissible_table()}))
    loose = bundle(frozen, case, analog=analog)
    with pytest.raises(ValueError, match="without a chooser_recipe"):
        build_native_score_inputs(SourceBundle(**{**loose.__dict__, "chooser_recipe": {}}))


# -- the full vector through the scoring stage, and the chooser score --------------------


class _Spy:
    """Records the facts the chooser stage hands the frozen executor."""

    def __init__(self, executor):
        self.executor, self.seen = executor, None

    def predict(self, facts):
        self.seen = dict(facts)
        return self.executor.predict(facts)

    def __str__(self):
        return str(self.executor)


def _request(strategy: str) -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-chooser67", calendar_revision="cal-1", strategy_version=strategy,
        deployment_id="dep-1", decision_clock_id="entry-close",
        requested_decision_at=EVENT, snapshot_id="snap-1", mode="replay",
        fill_model={"alpha": 0.5})


def _legacy_case_from_record(case: dict, record) -> dict:
    values = record.resolved_request
    legs = [dict(leg) for leg in values["legs"]]
    return {**case, "legs": legs, "spot": values["spot"],
            "entry_cost": values["entry_cost"],
            "forecast_abs_move": values["forecast_abs_move"],
            "exp_pnl_sim": values.get("exp_pnl_sim"),
            "flags": [c for c in record.reason_codes if c == "WIDE_MARKET"]}


@pytest.mark.parametrize("strategy, complete", [
    ("TWIN-P", True), ("TWIN-P5", True), ("CTR5", False), ("BFLY-P", False)])
def test_full_vector_and_chooser_score_equal_legacy(frozen, analog, tmp_path, monkeypatch,
                                                    strategy, complete):
    """``complete``: this case's vector is all finite, so both sides score;
    otherwise (no breakeven on one side) both decline the same way."""
    root, _ = frozen
    spot = 100.0
    quotes = {("P", float(k), EXPIRY): {"bid": max(spot - k, 0.0) + 0.4,
                                         "ask": max(spot - k, 0.0) + 0.5}
              for k in range(70, 131)}
    case = make_case(9, strategy=strategy, spot=spot, quotes=quotes, entry_date=EVENT)
    case["features"].update({"iv30": 55.0, "signed_streak": 2.0})
    source = bundle(frozen, case, analog=analog, table=legacy_n_admissible_table())
    source = SourceBundle(**{**source.__dict__, "residual_recipe": {
        "mode": "planned_exit", "pre_iv30": 40.0},
        "paired_residual_rows": tuple(
            {"event_date": f"2025-{1 + i % 12:02d}-{1 + i % 27:02d}", "pred_abs_move": 3.0 + i % 9,
             "err_move": math.sin(i) * 3.0, "err_crush": -5.0 + math.cos(i) * 10.0}
            for i in range(400))})
    inputs = build_native_score_inputs(source)
    spy = _Spy(inputs.chooser["executors"]["chooser_score"])
    inputs.chooser["executors"]["chooser_score"] = spy
    record = application.score_one(_request(strategy), inputs)
    assert spy.seen is not None, record.reason_codes
    native_vector = {name: spy.seen[name] for name in CHOOSER_FEATURES}

    legacy_case = _legacy_case_from_record(case, record)
    scorer = legacy_scorer(tmp_path, monkeypatch, frame=pool_frame())
    assert diff(legacy_frame(scorer, legacy_case), native_vector, CHOOSER_FEATURES) == {}
    nonfinite = [n for n, v in native_vector.items() if not math.isfinite(v)]
    assert (nonfinite == []) is complete, nonfinite

    artifact = registry.load_artifact(root / "chooser.joblib")
    scorer.model = lambda *a, **k: (SimpleNamespace(id="dyn_sv_chooser_v1_1"), artifact)
    result = _Result(legacy_case)
    Scorer._score_chooser(scorer, SimpleNamespace(strategy=strategy), result,
                          legacy_features(legacy_case))
    if complete:
        assert result.chooser_score is not None, result.flags
        assert record.forecasts["chooser_score"] == result.chooser_score
    else:
        assert result.chooser_score is None and "CHOOSER_MISSING_FEATURES" in result.flags
        assert record.forecasts.get("chooser_score") is None
        assert "CHOOSER_MISSING_FEATURES" in record.reason_codes


def test_a_strict_capture_without_producer_role_rows_still_scores(
        frozen, analog, tmp_path, monkeypatch):
    """Phase 4 capture regression (the 2026-09-24 fresh-replay defect).

    A strict trace carries ``features.role_model_inputs`` rows only for the
    roles the row itself served: a forecast-sized menu row (CND-PS, BFLY-P5,
    CTR5, RAMP7, every DYN-SV member) registers its sizing fold and the
    chooser's 17 primitives, never the chooser's ``implied_t1``/``runup_move``
    producer folds -- those folds are recorded at the chooser site and their
    rows merged into ``model_inputs`` (``_merge_chooser_rows``). The frozen
    chooser stage must then serve those folds from the full runtime facts --
    legacy ``Scorer._chooser_frame`` feeds them the same feature frame -- and
    score; wiping the runtime features to an absent role row left the
    producer columns NaN and declined every such row with a native-only
    CHOOSER_MISSING_FEATURES while legacy carried a ``chooser_score``.
    """
    root, _ = frozen
    spot = 100.0
    quotes = {("P", float(k), EXPIRY): {"bid": max(spot - k, 0.0) + 0.4,
                                         "ask": max(spot - k, 0.0) + 0.5}
              for k in range(70, 131)}
    case = make_case(9, strategy="TWIN-P", spot=spot, quotes=quotes, entry_date=EVENT)
    case["features"].update({"iv30": 55.0, "signed_streak": 2.0})
    source = bundle(frozen, case, analog=analog, table=legacy_n_admissible_table())
    source = SourceBundle(**{**source.__dict__, "residual_recipe": {
        "mode": "planned_exit", "pre_iv30": 40.0},
        "paired_residual_rows": tuple(
            {"event_date": f"2025-{1 + i % 12:02d}-{1 + i % 27:02d}",
             "pred_abs_move": 3.0 + i % 9, "err_move": math.sin(i) * 3.0,
             "err_crush": -5.0 + math.cos(i) * 10.0}
            for i in range(400))})
    inputs = build_native_score_inputs(source)
    # What such a trace's role capture holds: the sizing fold's row (aliased
    # role) and the chooser's 17 primitives -- no implied_t1/runup_move rows.
    inputs.features["role_model_inputs"] = {
        "size": {name: case["features"][name] for name in FOLD_FEATURES},
        "chooser": {name: case["features"][name] for name in PRIMITIVES},
    }
    spy = _Spy(inputs.chooser["executors"]["chooser_score"])
    inputs.chooser["executors"]["chooser_score"] = spy
    record = application.score_one(_request("TWIN-P"), inputs)
    assert "CHOOSER_MISSING_FEATURES" not in record.reason_codes, record.reason_codes
    assert spy.seen is not None, record.reason_codes
    vector = {name: spy.seen[name] for name in CHOOSER_FEATURES}
    producer = tuple(c for cols in PRODUCER_COLUMNS.values() for c in cols)
    assert all(math.isfinite(vector[c]) for c in producer), producer

    legacy_case = _legacy_case_from_record(case, record)
    scorer = legacy_scorer(tmp_path, monkeypatch, frame=pool_frame())
    assert diff(legacy_frame(scorer, legacy_case), vector, CHOOSER_FEATURES) == {}
    artifact = registry.load_artifact(root / "chooser.joblib")
    scorer.model = lambda *a, **k: (SimpleNamespace(id="dyn_sv_chooser_v1_1"), artifact)
    result = _Result(legacy_case)
    Scorer._score_chooser(scorer, SimpleNamespace(strategy="TWIN-P"), result,
                          legacy_features(legacy_case))
    assert result.chooser_score is not None, result.flags
    assert record.forecasts["chooser_score"] == result.chooser_score
