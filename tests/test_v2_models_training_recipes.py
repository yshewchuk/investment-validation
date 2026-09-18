"""P5-3: current dataset/training recipes, receipts, resumable job, no-fit.

Acceptance (``guides/rearchitecture_phase5_models.md`` P5-3 row):
training-membership and label-availability receipts per fold; a planted
future label or upstream in-sample leakage fails; an interrupted multi-fold
job resumes without losing or redoing completed folds. Plus: legacy is the
spec (recipes are cross-checked against the legacy symbols and reproduce
legacy's folds and fits exactly), and nothing here runs under the no-fit
guard that rigs every scoring path.

Synthetic in-memory frames and ``tmp_path`` only; never ``data/``.
"""
from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.models.no_fit import RuntimeFitForbidden, no_fit_guard
from engine.v2.models.training import (
    OWNER_P5_4,
    RecipeKey,
    TrainingRefused,
    UnsupportedEstimator,
    current_recipes,
    dataset_fingerprint,
    fit_recipe_estimator,
    fold_receipts,
    plan_folds,
    prepare_dataset,
    receipt_issues,
    recipe_fingerprint,
    run_training_job,
)

ROOT = Path(__file__).resolve().parents[1]
RECIPES = current_recipes()


def _recipe(role, strategy="*", output="champion"):
    return RECIPES[RecipeKey(role, strategy, output)]


def _codes(issues):
    return {i.code for i in issues}


# --------------------------------------------------------------------------
# recipes are keyed by the inventory's roles and mirror the legacy symbols
# --------------------------------------------------------------------------


def _registry_champions():
    models = json.loads((ROOT / "engine/models/registry.json").read_text())["models"]
    rows = models.values() if isinstance(models, dict) else models
    return {(r["role"], r["strategy"]): r for r in rows if r.get("champion")}


def test_every_champion_binding_has_a_champion_recipe_with_its_feature_order():
    champions = _registry_champions()
    champion_recipes = {(k.role, k.strategy): r for k, r in RECIPES.items() if k.output == "champion"}
    assert set(champion_recipes) == set(champions)
    for key, row in champions.items():
        recipe = champion_recipes[key]
        assert recipe.output_id == row["id"]
        assert recipe.features == tuple(row["features"])
        if key == ("chooser", "DYN-SV"):  # the registry records prose for this target
            from engine.models.training import chooser
            assert recipe.target.column == chooser.TARGET
        else:
            assert recipe.target.column == row["target"]
        assert recipe.clock_id == "legacy.entry_close.v1"


def test_every_tier4_producer_has_a_monthly_recipe_matching_tier4_constants():
    from engine.data.features import tier4

    monthly = {r.produces: r for k, r in RECIPES.items() if k.output == "tier4_monthly"}
    assert set(monthly) == set(tier4.PRODUCES)
    for recipe in monthly.values():
        assert recipe.folds.kind == "monthly_cutoff"
        assert pd.Timestamp(recipe.folds.first_fold) == tier4.FIRST_FOLD
        assert recipe.folds.min_train_rows == tier4.MIN_TRAIN_ROWS
        assert recipe.residuals.min_pool == tier4.MIN_RESIDUALS
        # Separate outputs: the champion recipe of the same model is a different identity.
        champion = RECIPES[dataclasses.replace(recipe.key, output="champion")]
        assert champion.output_id == recipe.output_id
        assert recipe_fingerprint(champion) != recipe_fingerprint(recipe)


def test_recipe_constants_mirror_legacy_modules():
    from engine import payoff, recalibrate
    from engine.data.features import tier4
    from engine.models.training import (
        common,
        gate,
        gate_forecast_analog,
        implied_t1,
        iv_crush,
        runup_move,
        size_model,
    )

    size = _recipe("size")
    assert size.features == size_model.FEATURES and size.target.column == size_model.TARGET
    assert {(m.column, (m.lo, m.hi)) for m in size.value_masks} == set(size_model.BOUNDS.items())
    assert size.filters[0].value == size_model.MIN_PRIOR
    assert size.estimator.seeds == (common.SEED,)
    assert size.folds.first_test_year == 2013  # size_model.train default
    assert _recipe("implied_t1").features == implied_t1.FEATURES
    assert _recipe("implied_t1").folds.first_test_year == 2015
    runup = _recipe("runup_move")
    assert runup.features == runup_move.FEATURES and runup.target.column == runup_move.TARGET
    assert runup.folds.first_test_year == 2018 and runup.target.transform == "log1p_clip0"
    assert any(f.column == "days_before_print" and f.value == runup_move.HORIZON for f in runup.filters)
    years = next(f.value for f in runup.filters if f.op == "year_between")
    assert range(years[0], years[1] + 1) == tier4.IM_T1_YEARS
    crush = _recipe("iv_crush")
    assert crush.features == iv_crush.FEATURES and crush.target.column == iv_crush.TARGET
    assert crush.label.max_days_after_cutoff == iv_crush.MAX_GAP_DAYS
    thru, rn = _recipe("gate", "STR-THRU"), _recipe("gate", "STR-RUNUP")
    assert thru.features == gate_forecast_analog.FEATURES and rn.features == gate.FEATURES
    for g in (thru, rn):
        assert g.folds.first_test_year == 2020
        assert g.threshold.top_fraction == gate.TOP_FRACTION
        assert next(f.value for f in g.filters if f.column == "fill_alpha") == gate.GATE_ALPHA
    assert [d.produces for d in thru.upstream if d.lineage == "tier4_monthly_oos"] == ["pred_abs_move"]
    assert not rn.upstream
    line = _recipe("payoff_line", "STR-THRU", "calibration")
    assert line.features[0] == payoff.PAYOFF_DRIVER["STR-THRU"]
    assert _recipe("payoff_line", "STR-RUNUP", "calibration").features[0] == payoff.PAYOFF_DRIVER["STR-RUNUP"]
    assert line.estimator.params["min_trades"] == payoff.MIN_TRADES
    assert line.estimator.params["max_residuals"] == payoff.MAX_RESIDUALS
    assert line.estimator.seeds == (payoff.RESIDUAL_SEED,)
    recal = _recipe("recalibration_map", "STR-THRU", "calibration")
    assert recal.estimator.params["min_pairs"] == recalibrate.MIN_PAIRS
    owners = {k.role: r.fit_owner for k, r in RECIPES.items() if k.output == "calibration"}
    assert owners["recalibration_map"] == OWNER_P5_4  # no frozen recalibration builder yet
    assert owners["payoff_line"] == owners["payoff_surface"] != OWNER_P5_4


def _exp169():
    tree = ast.parse((ROOT / "experiments/EXP-169_menu7prime_confirmation/run.py").read_text())
    consts, calls = {}, {}
    for node in tree.body:  # module level only: main() overrides these for a smoke mode
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in (
                "MLPRegressor", "QuantileTransformer"):
            calls[node.func.id] = {k.arg: k.value for k in node.keywords}
    return consts, calls


def test_chooser_recipe_matches_exp169_fit_head_literals():
    consts, calls = _exp169()
    chooser = _recipe("chooser", "DYN-SV")
    assert chooser.estimator.seeds == consts["SEEDS"]
    assert chooser.folds.first_test_year == consts["FIRST_TEST_YEAR"]
    assert chooser.folds.min_train_rows == consts["MIN_FIT_ROWS"]
    assert any(f.op == "group_nunique_ge" and f.value[1] == consts["MIN_OFFERED"] for f in chooser.filters)
    mlp = {k: ast.literal_eval(v) for k, v in calls["MLPRegressor"].items() if k != "random_state"}
    assert dict(chooser.estimator.params["mlp"], hidden_layer_sizes=tuple(
        chooser.estimator.params["mlp"]["hidden_layer_sizes"])) == mlp
    assert ast.literal_eval(calls["QuantileTransformer"]["output_distribution"]) == "normal"


# --------------------------------------------------------------------------
# native estimators predict bit-identically to the legacy fit()
# --------------------------------------------------------------------------


def _xy(n=400, k=None, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    y = np.abs(X[:, 0] * 2.0 + X[:, 1] + rng.normal(size=n)) + 0.1
    return X, y


@pytest.mark.parametrize("key,legacy", [
    (("size", "*"), "engine.models.training.size_model"),
    (("implied_t1", "*"), "engine.models.training.implied_t1"),
    (("runup_move", "*"), "engine.models.training.runup_move"),
    (("iv_crush", "*"), "engine.models.training.iv_crush"),
    (("gate", "STR-RUNUP"), "engine.models.training.gate"),
    (("gate", "STR-THRU"), "engine.models.training.gate_forecast_analog"),
])
def test_native_estimator_predicts_exactly_like_legacy_fit(key, legacy):
    import importlib

    recipe = _recipe(*key)
    module = importlib.import_module(legacy)
    X, y = _xy(k=6)
    native = fit_recipe_estimator(recipe, X, y)
    old = module.fit(X, y, recipe.estimator.seeds[0])
    assert np.array_equal(np.asarray(native.predict(X), float).ravel(),
                          np.asarray(old.predict(X), float).ravel())


def test_chooser_ensemble_uses_the_served_mean_ensemble_arithmetic():
    from engine.models.ensemble import MeanEnsemble

    chooser = _recipe("chooser", "DYN-SV")
    fast = dataclasses.replace(chooser, estimator=dataclasses.replace(
        chooser.estimator, params={**chooser.estimator.params,
                                   "mlp": {**chooser.estimator.params["mlp"], "max_iter": 5}}))
    X, y = _xy(n=300, k=4)
    with pytest.warns(Warning):  # 5 iterations does not converge; speed only
        native = fit_recipe_estimator(fast, X, y)
    assert len(native.models) == 5
    assert [m[-1].random_state for m in native.models] == list(chooser.estimator.seeds)
    assert np.array_equal(native.predict(X), MeanEnsemble(native.models).predict(X))


# --------------------------------------------------------------------------
# membership and folds reproduce legacy walk_forward / tier4 exactly
# --------------------------------------------------------------------------


def _synthetic(n=3000, years=(2012, 2021), features=("f1", "f2", "f3"), seed=7, target="y"):
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp(f"{years[0]}-01-01"), pd.Timestamp(f"{years[1]}-12-31")
    days = rng.integers(0, (end - start).days + 1, size=n)
    dates = (start + pd.to_timedelta(np.sort(days), unit="D")).normalize()
    frame = pd.DataFrame({f: rng.normal(size=n) for f in features})
    frame["ticker"] = [f"T{i % 97}" for i in range(n)]
    frame["date"] = dates
    frame["event_date"] = dates
    frame["event_id"] = [f"E{i}" for i in range(n)]
    frame["year"] = dates.year
    frame[target] = frame[features[0]] * 2 + rng.normal(size=n)
    frame.loc[rng.choice(n, 40, replace=False), features[1]] = np.nan
    frame.loc[rng.choice(n, 40, replace=False), target] = np.nan
    frame["label_available_at"] = dates + pd.Timedelta(days=1)
    return frame


def _small(recipe, features=("f1", "f2", "f3"), target="y", min_rows=None, **extra):
    folds = recipe.folds if min_rows is None else dataclasses.replace(recipe.folds, min_train_rows=min_rows)
    return dataclasses.replace(
        recipe, features=features, target=dataclasses.replace(recipe.target, column=target),
        filters=(), value_masks=(), key_columns=("ticker", "date"), membership_time_column="date",
        year_column="year", folds=folds, upstream=(), **extra)


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, X, y, seed):
        self.calls.append((np.array(X), np.array(y)))

        class _Const:
            def predict(self, X):
                return np.zeros(len(X))
        return _Const()


def test_expanding_year_folds_equal_legacy_walk_forward_and_fit_final():
    from engine.models.training.common import fit_final, walk_forward

    recipe = _small(_recipe("size"), min_rows=500)
    recipe = dataclasses.replace(recipe, folds=dataclasses.replace(recipe.folds, first_test_year=2014))
    frame = _synthetic()
    rec = _Recorder()
    walk_forward(frame, recipe.features, "y", rec, first_test_year=2014, min_train_rows=500)
    fit_final(frame, recipe.features, "y", rec)
    prepared = prepare_dataset(recipe, frame)
    plans = [p for p in plan_folds(recipe, prepared) if p.skipped is None]
    assert len(plans) == len(rec.calls)
    for plan, (X, y) in zip(plans, rec.calls):
        mine = prepared.frame.iloc[plan.train]
        assert np.array_equal(mine[list(recipe.features)].to_numpy(float), X)
        assert np.array_equal(mine["y"].to_numpy(float), y)


def test_monthly_folds_equal_legacy_tier4_build_producer():
    from engine.data.features import tier4

    recipe = _small(_recipe("size", output="tier4_monthly"))
    frame = _synthetic(n=2500, years=(2011, 2014))
    rec = _Recorder()
    model = tier4.FeatureModel(model_id="p53-synthetic-monthly", produces="pred_abs_move",
                               features=recipe.features, target="y", fit=rec, prepare=lambda p: p)
    tier4.build_producer(frame, model, log=lambda _m: None)
    prepared = prepare_dataset(recipe, frame)
    plans = [p for p in plan_folds(recipe, prepared) if p.skipped is None]
    assert len(plans) == len(rec.calls) > 10
    for plan, (X, y) in zip(plans, rec.calls):
        mine = prepared.frame.iloc[plan.train]
        # Same rows; legacy keeps the prepared frame's order too.
        assert np.array_equal(mine[list(recipe.features)].to_numpy(float), X)
        assert np.array_equal(mine["y"].to_numpy(float), y)


def test_size_membership_equals_legacy_prepare():
    from engine.models.training import size_model

    recipe = _recipe("size")
    rng = np.random.default_rng(1)
    n = 300
    panel = pd.DataFrame({f: rng.normal(size=n) for f in recipe.features})
    panel["or_implied"] = rng.uniform(-5, 70, size=n)
    panel["or_rvol30"] = rng.uniform(-5, 800, size=n)
    panel["abs_move"] = rng.uniform(-5, 250, size=n)
    panel["n_prior"] = rng.integers(0, 10, size=n).astype(float)
    panel.loc[:5, "n_prior"] = np.nan
    panel["ticker"], panel["date"] = "T", pd.Timestamp("2020-01-02")
    panel["year"] = 2020
    legacy = size_model.prepare(panel)
    mine = prepare_dataset(recipe, panel).frame
    pd.testing.assert_frame_equal(legacy[list(recipe.features) + ["abs_move"]],
                                  mine[list(recipe.features) + ["abs_move"]])


# --------------------------------------------------------------------------
# receipts and their negative controls
# --------------------------------------------------------------------------


def _fold_receipts(recipe, frame, fold_id=None):
    prepared = prepare_dataset(recipe, frame)
    fp = dataset_fingerprint(recipe, prepared)
    plans = [p for p in plan_folds(recipe, prepared) if p.skipped is None]
    plan = plans[-2] if fold_id is None else next(p for p in plans if p.fold_id == fold_id)
    return prepared, plan, fold_receipts(recipe, prepared, plan, dataset_fp=fp)


def test_clean_fold_has_both_receipts_and_no_issues():
    recipe = _small(_recipe("size"), min_rows=200)
    _, plan, (membership, label) = _fold_receipts(recipe, _synthetic())
    assert membership["schema_version"] == "p5.training_membership_receipt.v1"
    assert label["schema_version"] == "p5.label_availability_receipt.v1"
    assert membership["n_train"] == len(plan.train) and membership["n_members_at_or_after_cutoff"] == 0
    assert label["n_labels_beyond_allowance"] == 0
    assert receipt_issues(recipe, membership, label) == ()


def test_planted_future_label_fails_the_check():
    recipe = _small(_recipe("size"), min_rows=200)
    frame = _synthetic()
    prepared, plan, _ = _fold_receipts(recipe, frame, "wf-2019")
    victim = prepared.frame.iloc[plan.train[0]]["event_id"]
    frame.loc[frame["event_id"] == victim, "label_available_at"] = pd.Timestamp("2019-03-01")
    _, _, (membership, label) = _fold_receipts(recipe, frame, "wf-2019")
    issues = receipt_issues(recipe, membership, label)
    assert _codes(issues) == {"FUTURE_LABEL"}
    assert issues[0].path == "$.folds.wf-2019.label"


def test_label_inside_legacy_tolerance_is_counted_not_refused():
    recipe = _small(_recipe("size"), min_rows=200)
    frame = _synthetic()
    prepared, plan, _ = _fold_receipts(recipe, frame, "wf-2019")
    victim = prepared.frame.iloc[plan.train[-1]]["event_id"]
    frame.loc[frame["event_id"] == victim, "label_available_at"] = pd.Timestamp("2019-01-03")
    _, _, (membership, label) = _fold_receipts(recipe, frame, "wf-2019")
    assert label["n_labels_after_cutoff"] >= 1 and label["n_labels_beyond_allowance"] == 0
    assert receipt_issues(recipe, membership, label) == ()


def test_strict_label_rule_refuses_a_label_on_the_cutoff():
    recipe = _small(_recipe("runup_move"), min_rows=200, label=dataclasses.replace(
        _recipe("runup_move").label, time_column="label_available_at"))
    assert recipe.label.max_days_after_cutoff == 0
    frame = _synthetic()
    prepared, plan, _ = _fold_receipts(recipe, frame, "wf-2019")
    victim = prepared.frame.iloc[plan.train[-1]]["event_id"]
    frame.loc[frame["event_id"] == victim, "label_available_at"] = pd.Timestamp("2019-01-01")
    _, _, (membership, label) = _fold_receipts(recipe, frame, "wf-2019")
    assert _codes(receipt_issues(recipe, membership, label)) == {"FUTURE_LABEL"}


def test_future_member_and_missing_label_time_fail():
    recipe = _small(_recipe("size"), min_rows=200)
    frame = _synthetic()
    prepared, plan, _ = _fold_receipts(recipe, frame, "wf-2019")
    leaked = dataclasses.replace(plan, train=np.concatenate([plan.train, plan.test[:1]]))
    fp = dataset_fingerprint(recipe, prepared)
    membership, label = fold_receipts(recipe, prepared, leaked, dataset_fp=fp)
    assert "FUTURE_MEMBER" in _codes(receipt_issues(recipe, membership, label))
    no_labels = frame.drop(columns=["label_available_at"])
    _, _, (membership, label) = _fold_receipts(recipe, no_labels, "wf-2019")
    assert _codes(receipt_issues(recipe, membership, label)) == {"LABEL_TIME_MISSING"}


def _gate_frame(recipe):
    frame = _synthetic(n=2500, years=(2016, 2022), features=recipe.features, target="ret")
    frame["strategy"], frame["provenance"], frame["fill_alpha"] = "STR-THRU", "engine.replay", 0.5
    frame["entry_date"] = frame["event_date"]
    frame["exit_date"] = frame["event_date"] + pd.Timedelta(days=1)
    frame["pred_abs_move_fold_start"] = frame["event_date"].dt.to_period("M").dt.start_time
    frame["pred_abs_move_model_id"] = "size_v1_4"
    for col in recipe.features:
        frame[col] = frame[col].fillna(0.0)
    return frame


def _gate_issues(recipe, frame):
    prepared = prepare_dataset(recipe, frame)
    plan = next(p for p in plan_folds(recipe, prepared) if p.fold_id == "wf-2021")
    membership, label = fold_receipts(recipe, prepared, plan,
                                      dataset_fp=dataset_fingerprint(recipe, prepared))
    return prepared, plan, label, receipt_issues(recipe, membership, label)


def test_upstream_in_sample_leakage_fails_the_check():
    recipe = _recipe("gate", "STR-THRU")
    frame = _gate_frame(recipe)
    prepared, plan, label, issues = _gate_issues(recipe, frame)
    assert issues == ()
    tier4_dep = next(d for d in label["upstream"] if d["produces"] == "pred_abs_move")
    assert tier4_dep["n_verified"] == len(plan.train)
    analog = next(d for d in label["upstream"] if d["produces"] == "bucket_analog")
    assert analog["n_verified"] == 0 and analog["n_unverified"] == len(plan.train)

    victim = prepared.frame.iloc[plan.train[3]]["event_id"]
    leaked = frame.copy()
    # A forecast from a fold that started after the row: that model trained on it.
    leaked.loc[leaked["event_id"] == victim, "pred_abs_move_fold_start"] = pd.Timestamp("2021-06-01")
    assert _codes(_gate_issues(recipe, leaked)[3]) == {"UPSTREAM_IN_SAMPLE"}

    refit = frame.copy()  # the full-refit champion's in-sample prediction carries no fold
    refit.loc[refit["event_id"] == victim, "pred_abs_move_fold_start"] = pd.NaT
    assert _codes(_gate_issues(recipe, refit)[3]) == {"UPSTREAM_LINEAGE_MISSING"}

    other = frame.copy()
    other.loc[other["event_id"] == victim, "pred_abs_move_model_id"] = "size_v1_3"
    assert _codes(_gate_issues(recipe, other)[3]) == {"UPSTREAM_MODEL_MISMATCH"}

    bare = frame.drop(columns=["pred_abs_move_fold_start", "pred_abs_move_model_id"])
    assert _codes(_gate_issues(recipe, bare)[3]) == {"UPSTREAM_LINEAGE_MISSING"}


# --------------------------------------------------------------------------
# the job: receipts per fold, refusal, resume
# --------------------------------------------------------------------------


class _CountingFit:
    def __init__(self, fail_on=None):
        self.n = 0
        self.fail_on = fail_on

    def __call__(self, recipe, X, y):
        self.n += 1
        if self.fail_on is not None and self.n == self.fail_on:
            raise KeyboardInterrupt("simulated interruption mid-job")
        from sklearn.linear_model import LinearRegression

        return LinearRegression().fit(X, y)


def _job_recipe():
    base = _small(_recipe("size"), min_rows=200)
    return dataclasses.replace(base, folds=dataclasses.replace(base.folds, first_test_year=2015))


def _files(out: Path) -> dict:
    return {str(p.relative_to(out)): p.read_bytes() for p in sorted((out / "folds").rglob("*")) if p.is_file()}


def test_job_writes_both_receipts_for_every_fold(tmp_path):
    recipe = _job_recipe()
    result = run_training_job(recipe, _synthetic(), tmp_path / "job", fit=_CountingFit())
    fold_ids = [o.fold_id for o in result.outcomes]
    assert fold_ids[-1] == "full-refit" and len(fold_ids) == 8
    for fold_id in fold_ids:
        fdir = tmp_path / "job" / "folds" / fold_id
        membership = json.loads((fdir / "membership_receipt.json").read_text())
        label = json.loads((fdir / "label_availability_receipt.json").read_text())
        assert membership["fold_id"] == label["fold_id"] == fold_id
        assert membership["recipe_fingerprint"] == recipe_fingerprint(recipe)
        assert (fdir / "estimator.joblib").is_file() and (fdir / "COMPLETE.json").is_file()
    assert (tmp_path / "job" / "summary.json").is_file()


def test_job_fits_with_the_native_estimator(tmp_path):
    recipe = dataclasses.replace(_small(_recipe("gate", "STR-RUNUP"), min_rows=200),
                                 threshold=_recipe("gate", "STR-RUNUP").threshold)
    recipe = dataclasses.replace(recipe, folds=dataclasses.replace(recipe.folds, first_test_year=2020))
    frame = _synthetic()
    frame["exit_date"] = frame["event_date"]
    result = run_training_job(recipe, frame, tmp_path / "gate")
    assert result.count("fitted") == 3  # 2020, 2021, full refit
    summary = json.loads((tmp_path / "gate" / "summary.json").read_text())
    assert summary["threshold_n"] > 0


def test_interrupted_job_resumes_without_losing_or_redoing_completed_folds(tmp_path):
    recipe, frame, out = _job_recipe(), _synthetic(), tmp_path / "job"
    with pytest.raises(KeyboardInterrupt):
        run_training_job(recipe, frame, out, fit=_CountingFit(fail_on=4))
    names = sorted(p.name for p in (out / "folds").iterdir())
    assert [n for n in names if ".partial-" not in n] == ["wf-2015", "wf-2016", "wf-2017"]
    assert [n.split(".")[0] for n in names if ".partial-" in n] == ["wf-2018"]  # never published
    before = {k: v for k, v in _files(out).items() if ".partial-" not in k}

    fit = _CountingFit()
    result = run_training_job(recipe, frame, out, fit=fit)
    assert result.count("resumed") == 3 and result.count("fitted") == 5
    assert fit.n == 5  # only the folds that were not complete
    after = _files(out)
    assert {k: after[k] for k in before} == before  # completed artifacts byte-identical
    assert not list((out / "folds").glob("*.partial-*"))

    again = _CountingFit()
    assert run_training_job(recipe, frame, out, fit=again).count("resumed") == 8 and again.n == 0


def test_resume_refuses_a_different_dataset_and_a_corrupted_fold(tmp_path):
    recipe, frame, out = _job_recipe(), _synthetic(), tmp_path / "job"
    run_training_job(recipe, frame, out, fit=_CountingFit())
    changed = frame.copy()
    changed.loc[0, "f1"] = 99.0
    with pytest.raises(TrainingRefused) as err:
        run_training_job(recipe, changed, out, fit=_CountingFit())
    assert _codes(err.value.issues) == {"RESUME_MISMATCH"}

    (out / "folds" / "wf-2016" / "estimator.joblib").write_bytes(b"corrupt")
    fit = _CountingFit()
    with pytest.raises(TrainingRefused) as err:
        run_training_job(recipe, frame, out, fit=fit)
    assert _codes(err.value.issues) == {"ARTIFACT_CORRUPT"} and fit.n == 0


def test_job_refuses_a_planted_future_label_and_keeps_earlier_folds(tmp_path):
    recipe, frame, out = _job_recipe(), _synthetic(), tmp_path / "job"
    victim = frame[(frame["year"] == 2017) & frame["y"].notna() & frame["f2"].notna()].index[0]
    frame.loc[victim, "label_available_at"] = pd.Timestamp("2018-06-01")
    with pytest.raises(TrainingRefused) as err:
        run_training_job(recipe, frame, out, fit=_CountingFit())
    assert _codes(err.value.issues) == {"FUTURE_LABEL"}
    assert err.value.issues[0].path.startswith("$.folds.wf-2018")
    assert sorted(p.name for p in (out / "folds").iterdir()) == ["wf-2015", "wf-2016", "wf-2017"]


def _payoff_trades(n=900, seed=5):
    rng = np.random.default_rng(seed)
    exits = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")
    trades = pd.DataFrame({
        "event_id": [f"E{i}" for i in range(n)],
        "strategy": np.where(rng.random(n) < 0.9, "STR-THRU", "STR-RUNUP"),
        "fill_alpha": np.where(rng.random(n) < 0.8, 0.5, 0.25),
        "abs_move": rng.uniform(0, 10, n), "im_t1": rng.uniform(2, 12, n),
        "spot_entry": rng.uniform(20, 200, n), "spot_exit": rng.uniform(20, 200, n),
        "strike": rng.uniform(20, 200, n), "exit_value": rng.uniform(0, 30, n), "exit_date": exits,
    })
    trades.loc[:4, "spot_entry"] = 0.0      # legacy's spot > 0 guard
    trades.loc[5:9, "abs_move"] = np.nan    # and its finite guard
    return trades


def _legacy_rows(trades, strategy, driver, alpha):
    rows = trades[(trades["strategy"] == strategy) & np.isclose(trades["fill_alpha"], alpha)]
    return [{"driver": r[driver], "spot_entry": r["spot_entry"], "spot_exit": r["spot_exit"],
             "strike": r["strike"], "exit_value": r["exit_value"],
             "exit_date": r["exit_date"].strftime("%Y-%m-%d")} for _, r in rows.iterrows()]


def test_payoff_line_recipe_fits_the_p5_4_frozen_artifact(tmp_path):
    from engine import payoff
    from engine.v2.models.payoff_artifact import serialize_payoff_artifact
    from engine.v2.models.training.payoff import build_payoff_line_artifact

    recipe = _recipe("payoff_line", "STR-THRU", "calibration")
    trades = _payoff_trades()
    with pytest.raises(UnsupportedEstimator, match="fill alpha"):
        run_training_job(recipe, trades, tmp_path / "noalpha", cutoffs=("2021-01-01",))
    result = run_training_job(recipe, trades, tmp_path / "fit", cutoffs=("2021-01-01",), alpha=0.5)
    assert [o.status for o in result.outcomes] == ["fitted"]
    fdir = tmp_path / "fit/folds/cut-2021-01-01"
    expected = build_payoff_line_artifact(_legacy_rows(trades, "STR-THRU", "abs_move", 0.5),
                                          strategy="STR-THRU", driver="abs_move", alpha=0.5,
                                          before="2021-01-01")
    assert (fdir / "payoff_artifact.json").read_bytes() == serialize_payoff_artifact(expected)
    legacy = payoff.fit_payoff(trades, "STR-THRU", alpha=0.5, before="2021-01-01")
    membership = json.loads((fdir / "membership_receipt.json").read_text())
    assert membership["n_train"] == legacy.n == expected.n  # members == rows the fit keeps
    assert (expected.slope, expected.intercept) == (legacy.slope, legacy.intercept)


def test_payoff_surface_recipe_fits_the_p5_4_frozen_artifact(tmp_path):
    from engine.v2.models.payoff_artifact import serialize_payoff_artifact
    from engine.v2.models.training.payoff import build_payoff_surface_artifact

    recipe = _recipe("payoff_surface", "STR-RUNUP", "calibration")
    trades = _payoff_trades(n=4000)
    run_training_job(recipe, trades, tmp_path / "fit", cutoffs=("2021-06-01",), alpha=0.5)
    fdir = tmp_path / "fit/folds/cut-2021-06-01"
    expected = build_payoff_surface_artifact(_legacy_rows(trades, "STR-RUNUP", "im_t1", 0.5),
                                             alpha=0.5, before="2021-06-01")
    assert (fdir / "payoff_artifact.json").read_bytes() == serialize_payoff_artifact(expected)
    membership = json.loads((fdir / "membership_receipt.json").read_text())
    assert membership["n_train"] == expected.n


def test_payoff_fold_under_min_trades_is_skipped_and_recalibration_stays_a_seam(tmp_path):
    trades = _payoff_trades()
    line = _recipe("payoff_line", "STR-THRU", "calibration")
    result = run_training_job(line, trades, tmp_path / "early", cutoffs=("2020-02-01",), alpha=0.5)
    assert [o.status for o in result.outcomes] == ["skipped"]  # legacy PayoffError
    assert not (tmp_path / "early/folds/cut-2020-02-01/payoff_artifact.json").exists()

    recal = _recipe("recalibration_map", "STR-THRU", "calibration")
    pairs = trades.assign(raw_win=0.5, outcome=1.0)
    with pytest.raises(UnsupportedEstimator):
        run_training_job(recal, pairs, tmp_path / "fit", cutoffs=("2021-01-01",))
    result = run_training_job(recal, pairs, tmp_path / "plan", plan_only=True,
                              cutoffs=("2021-01-01",), alpha=0.5)
    assert [o.status for o in result.outcomes] == ["planned"]
    membership = json.loads((tmp_path / "plan/folds/cut-2021-01-01/membership_receipt.json").read_text())
    keep = (pairs["strategy"] == "STR-THRU") & np.isclose(pairs["fill_alpha"], 0.5)
    assert membership["n_train"] == int((keep & (pairs["exit_date"] < "2021-01-01")).sum())
    assert membership["n_members_at_or_after_cutoff"] == 0


# --------------------------------------------------------------------------
# nothing here runs inside scoring
# --------------------------------------------------------------------------


def test_training_job_trips_the_scoring_no_fit_guard_before_touching_disk(tmp_path):
    recipe = _job_recipe()
    with no_fit_guard():
        with pytest.raises(RuntimeFitForbidden, match="run_training_job"):
            run_training_job(recipe, _synthetic(), tmp_path / "job", fit=_CountingFit())
        with pytest.raises(RuntimeFitForbidden, match="fit_recipe_estimator"):
            fit_recipe_estimator(_recipe("iv_crush"), *_xy(k=41))
    assert not (tmp_path / "job").exists()


def test_no_scoring_or_legacy_module_imports_the_training_package():
    from checks.import_layers import build_graph

    files = {str(p.relative_to(ROOT)): p.read_bytes() for p in (ROOT / "engine").rglob("*.py")}
    edges = build_graph(files).edges
    importers = {e.importer for e in edges if e.imported.startswith("engine.v2.models.training")}
    assert importers and all(i.startswith("engine.v2.models.training") for i in importers)


def test_training_job_tool_lists_every_recipe_and_bounds_label_dates(capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("p53_tool", ROOT / "tools/phase5_training_job.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert tool.main(["--list"]) == 0
    listed = [line.split()[0] for line in capsys.readouterr().out.splitlines()]
    assert listed == [k.label() for k in RECIPES]
    friday_amc = tool._next_session(["2021-12-31", "2021-12-29"])
    assert list(friday_amc.dt.strftime("%Y-%m-%d")) == ["2022-01-03", "2021-12-30"]
