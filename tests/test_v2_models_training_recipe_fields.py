"""P5-3 mutation-survivor cluster, Part 1: four consumer-test families.

``engine/v2/models/training/recipes.py``'s five ``_*_recipes`` builders
(``_size_recipes``, ``_decision_recipes``, ``_crush_recipes``,
``_gate_recipes``, ``_chooser_recipes``) build the 11 non-calibration
recipes. A prior mutation analysis found 809 survivors in these functions.
This file drives each field through its REAL consumer on a synthetic
``pd.DataFrame`` and asserts the consumer's observable OUTPUT -- never a
field read back off the ``TrainingRecipe`` itself, which would be a
change-detector that kills mutants while proving only that the literal
still says what it says (see ``recipes.py``'s own account of the
``ACTION_NAMES`` drift this exact failure mode caused).

Four families, each driving one real consumer:

1. ``prepare_dataset`` -- kills ``key_columns``, ``membership_time_column``,
   ``year_column``, ``filters``, ``value_masks``.
2. ``plan_folds`` -- kills ``folds.kind``, ``min_train_rows``,
   ``first_test_year``, ``first_fold``, ``full_refit``.
3. ``fold_receipts`` / ``receipt_issues`` -- kills ``label.*`` and all
   ``upstream[i].*`` fields except ``upstream[i].source`` (never read by
   the receipt logic; that one is Part 2's).
4. ``fit_recipe_estimator`` / ``job._summary`` -- kills ``estimator.kind``,
   ``estimator.seeds`` and ``threshold.*``.

The 11 recipes are discovered programmatically from ``current_recipes()``
(never a hand-copied key list): every key whose ``output`` is not
``"calibration"`` -- the calibration surfaces come from a separate module
(``.calibration``) and carry none of this cluster's survivors.

Synthetic in-memory frames only; never ``data/``.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.v2.models.training import (
    RecipeKey,
    current_recipes,
    dataset_fingerprint,
    fit_recipe_estimator,
    fold_receipts,
    plan_folds,
    prepare_dataset,
    receipt_issues,
    run_training_job,
)
from engine.v2.models.training.job import _summary

RECIPES = {k: r for k, r in current_recipes().items() if k.output != "calibration"}
assert len(RECIPES) == 11, f"expected 11 non-calibration recipes, found {len(RECIPES)}"


def _recipe(role, strategy="*", output="champion"):
    return RECIPES[RecipeKey(role, strategy, output)]


def _feature_frame(features, n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f: rng.normal(loc=1.0, scale=0.5, size=n) for f in features})


def _row(**cols):
    return pd.DataFrame([cols])


# ==========================================================================
# Family 1: prepare_dataset -- key_columns, membership_time_column,
# year_column, filters, value_masks
# ==========================================================================


def _size_frame(role_output):
    """size champion/tier4_monthly share filters/masks: n_prior>=4 (RowFilter
    'ge'); or_implied in [0,60], or_rvol30 in [0,700], abs_move in [0,200]
    (ValueMask, inclusive bounds)."""
    recipe = _recipe("size", output=role_output)
    n = 40
    frame = _feature_frame(recipe.features, n, seed=1)
    frame["or_implied"], frame["or_rvol30"], frame["abs_move"] = 20.0, 100.0, 5.0
    frame["n_prior"] = 10.0
    frame["ticker"] = [f"T{i}" for i in range(n)]
    frame["date"] = pd.Timestamp("2016-01-01") + pd.to_timedelta(np.arange(n), unit="D")
    frame["year"] = frame["date"].dt.year
    frame["label_available_at"] = frame["date"] + pd.Timedelta(days=1)
    frame["tag"] = [f"bulk-{i}" for i in range(n)]

    def probe(tag, **overrides):
        base = frame.iloc[[0]].copy()
        base["tag"] = tag
        base["ticker"] = tag
        for k, v in overrides.items():
            base[k] = v
        return base

    probes = pd.concat([
        probe("filter-below", n_prior=3.0),          # ge 4: excluded
        probe("filter-at", n_prior=4.0),              # ge 4: boundary, kept
        probe("mask-implied-lo-below", or_implied=-0.001),
        probe("mask-implied-lo-at", or_implied=0.0),
        probe("mask-implied-hi-at", or_implied=60.0),
        probe("mask-implied-hi-above", or_implied=60.001),
        probe("mask-rvol-lo-below", or_rvol30=-0.001),
        probe("mask-rvol-hi-above", or_rvol30=700.001),
        probe("mask-target-lo-below", abs_move=-0.001),
        probe("mask-target-hi-above", abs_move=200.001),
    ], ignore_index=True)
    combined = pd.concat([frame, probes], ignore_index=True)
    dropped = {"filter-below"}
    masked_to_nan = {
        "mask-implied-lo-below": "or_implied", "mask-implied-hi-above": "or_implied",
        "mask-rvol-lo-below": "or_rvol30", "mask-rvol-hi-above": "or_rvol30",
        "mask-target-lo-below": "abs_move", "mask-target-hi-above": "abs_move",
    }
    preserved = {
        "filter-at": [], "mask-implied-lo-at": ["or_implied"], "mask-implied-hi-at": ["or_implied"],
    }
    return combined, dropped, masked_to_nan, preserved


def _decision_frame(role, role_output):
    """implied_t1/runup_move: session notna, year in [2017,2026] (inclusive).
    runup_move filters days_before_print isclose 14 on BOTH outputs;
    implied_t1 only on tier4_monthly."""
    recipe = _recipe(role, output=role_output)
    n = 40
    frame = _feature_frame(recipe.features, n, seed=2)
    frame["session"] = "regular"
    frame["year"] = 2020
    frame["days_before_print"] = 14.0
    frame["ticker"] = [f"T{i}" for i in range(n)]
    frame["event_date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n), unit="D")
    frame["last_pre_print"] = frame["event_date"]
    target_col = recipe.target.column
    frame[target_col] = 1.0
    frame["tag"] = [f"bulk-{i}" for i in range(n)]

    def probe(tag, **overrides):
        base = frame.iloc[[0]].copy()
        base["tag"], base["ticker"] = tag, tag
        for k, v in overrides.items():
            base[k] = v
        return base

    probes = pd.concat([
        probe("filter-no-session", session=None),
        probe("filter-year-below", year=2016),
        probe("filter-year-at-lo", year=2017),
        probe("filter-year-at-hi", year=2026),
        probe("filter-year-above", year=2027),
        probe("filter-day14-off", days_before_print=13.0),
        probe("filter-day14-on", days_before_print=14.0),
    ], ignore_index=True)
    combined = pd.concat([frame, probes], ignore_index=True)
    day14_applies = role_output == "tier4_monthly" or role == "runup_move"
    dropped = {"filter-no-session", "filter-year-below", "filter-year-above"}
    if day14_applies:
        dropped = dropped | {"filter-day14-off"}
    kept = {"filter-year-at-lo", "filter-year-at-hi", "filter-day14-on"}
    return combined, dropped, kept


def _crush_frame(role_output):
    """iv_crush: no filters, no value masks -- every complete row survives."""
    recipe = _recipe("iv_crush", output=role_output)
    n = 30
    frame = _feature_frame(recipe.features, n, seed=3)
    frame["crush_pct_iv30"] = 5.0
    frame["ticker"] = [f"T{i}" for i in range(n)]
    frame["date"] = pd.Timestamp("2016-01-01") + pd.to_timedelta(np.arange(n), unit="D")
    frame["year"] = frame["date"].dt.year
    frame["label_available_at"] = frame["date"] + pd.Timedelta(days=1)
    frame["tag"] = [f"bulk-{i}" for i in range(n)]
    return frame


def _gate_frame(strategy):
    recipe = _recipe("gate", strategy)
    n = 30
    frame = _feature_frame(recipe.features, n, seed=4)
    frame["strategy"] = strategy
    frame["provenance"] = "engine.replay"
    frame["fill_alpha"] = 0.5
    frame["event_date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n), unit="D")
    frame["entry_date"] = frame["event_date"]
    frame["decision_date"] = frame["event_date"]
    frame["exit_date"] = frame["event_date"] + pd.Timedelta(days=1)
    frame["year"] = frame["event_date"].dt.year
    frame["ret"] = 0.05
    frame["event_id"] = [f"E{i}" for i in range(n)]
    frame["tag"] = [f"bulk-{i}" for i in range(n)]

    def probe(tag, **overrides):
        base = frame.iloc[[0]].copy()
        base["tag"], base["event_id"] = tag, tag
        for k, v in overrides.items():
            base[k] = v
        return base

    probes = pd.concat([
        probe("filter-wrong-strategy", strategy="STR-OTHER" if strategy != "STR-OTHER" else "X"),
        probe("filter-wrong-provenance", provenance="manual"),
        probe("filter-wrong-alpha", fill_alpha=0.25),
        probe("filter-alpha-at", fill_alpha=0.5),
        probe("filter-decision-before-entry",
              decision_date=frame["event_date"].iloc[0] - pd.Timedelta(days=1)),
        probe("filter-decision-at-entry", decision_date=frame["event_date"].iloc[0]),
    ], ignore_index=True)
    combined = pd.concat([frame, probes], ignore_index=True)
    dropped = {"filter-wrong-strategy", "filter-wrong-provenance", "filter-wrong-alpha",
              "filter-decision-before-entry"}
    kept = {"filter-alpha-at", "filter-decision-at-entry"}
    return combined, dropped, kept


def _chooser_frame():
    recipe = _recipe("chooser", "DYN-SV")
    n = 30
    frame = _feature_frame(recipe.features, n, seed=5)
    frame["fill_alpha"] = 0.5
    frame["event_date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n), unit="D")
    frame["exit_date"] = frame["event_date"] + pd.Timedelta(days=1)
    frame["year"] = frame["event_date"].dt.year
    frame["dev_target"] = 0.01
    # Every bulk row is its own event with 2 distinct strategies offered
    # (group_nunique_ge(strategy, 2) passes).
    frame["event_id"] = [f"E{i}" for i in range(n)]
    frame["strategy"] = ["STR-A" if i % 2 == 0 else "STR-B" for i in range(n)]
    frame["candidate_id"] = [f"C{i}" for i in range(n)]
    frame["tag"] = [f"bulk-{i}" for i in range(n)]

    # A dedicated event with only one distinct strategy offered (dropped),
    # and one with two distinct strategies (kept) -- both alpha-clean.
    lone = frame.iloc[[0]].copy()
    lone["tag"], lone["candidate_id"], lone["event_id"] = "filter-lone-strategy", "CLONE", "ELONE"
    lone["strategy"] = "STR-A"

    paired_a = frame.iloc[[0]].copy()
    paired_a["tag"], paired_a["candidate_id"], paired_a["event_id"] = "filter-paired-a", "CPA", "EPAIR"
    paired_a["strategy"] = "STR-A"
    paired_b = frame.iloc[[0]].copy()
    paired_b["tag"], paired_b["candidate_id"], paired_b["event_id"] = "filter-paired-b", "CPB", "EPAIR"
    paired_b["strategy"] = "STR-B"

    wrong_alpha = frame.iloc[[0]].copy()
    wrong_alpha["tag"] = "filter-wrong-alpha"
    wrong_alpha["candidate_id"], wrong_alpha["event_id"] = "CWA", "EWA2"
    wrong_alpha["fill_alpha"] = 0.3
    wrong_alpha_b = wrong_alpha.copy()
    wrong_alpha_b["tag"], wrong_alpha_b["candidate_id"] = "filter-wrong-alpha-b", "CWA2"
    wrong_alpha_b["strategy"] = "STR-B"

    combined = pd.concat([frame, lone, paired_a, paired_b, wrong_alpha, wrong_alpha_b],
                         ignore_index=True)
    dropped = {"filter-lone-strategy", "filter-wrong-alpha", "filter-wrong-alpha-b"}
    kept = {"filter-paired-a", "filter-paired-b"}
    return combined, dropped, kept


SIZE_CASES = [("size", o) for o in ("champion", "tier4_monthly")]
DECISION_CASES = [(role, o) for role in ("implied_t1", "runup_move") for o in ("champion", "tier4_monthly")]
CRUSH_CASES = [("iv_crush", o) for o in ("champion", "tier4_monthly")]


@pytest.mark.parametrize("role,output", SIZE_CASES)
def test_prepare_dataset_size_filters_and_value_masks(role, output):
    recipe = _recipe(role, output=output)
    frame, dropped, masked_to_nan, preserved = _size_frame(output)
    prepared = prepare_dataset(recipe, frame)
    survivors = set(prepared.frame["tag"])
    assert dropped.isdisjoint(survivors)
    assert survivors == set(frame["tag"]) - dropped

    def value_at(tag, col):
        row = prepared.frame[prepared.frame["tag"] == tag].iloc[0]
        return row[col]

    for tag, col in masked_to_nan.items():
        assert np.isnan(value_at(tag, col)), f"{tag}: {col} should have been masked to NaN"
    for tag, cols in preserved.items():
        for col in cols:
            assert not np.isnan(value_at(tag, col)), f"{tag}: {col} should NOT have been masked"
    # boundary "at" rows are complete (mask bounds are inclusive)
    idx = prepared.frame.index[prepared.frame["tag"] == "mask-implied-hi-at"][0]
    assert bool(prepared.features_complete[idx])


@pytest.mark.parametrize("role,output", DECISION_CASES)
def test_prepare_dataset_decision_filters(role, output):
    recipe = _recipe(role, output=output)
    frame, dropped, kept = _decision_frame(role, output)
    prepared = prepare_dataset(recipe, frame)
    survivors = set(prepared.frame["tag"])
    assert dropped.isdisjoint(survivors), f"{role}:{output} kept a row it should have dropped"
    assert kept.issubset(survivors), f"{role}:{output} dropped a row it should have kept"


@pytest.mark.parametrize("role,output", CRUSH_CASES)
def test_prepare_dataset_crush_has_no_filters_or_masks(role, output):
    recipe = _recipe(role, output=output)
    frame = _crush_frame(output)
    prepared = prepare_dataset(recipe, frame)
    # No filters/masks declared: every row survives untouched.
    assert len(prepared.frame) == len(frame)
    assert bool(prepared.complete.all())


@pytest.mark.parametrize("strategy", ["STR-THRU", "STR-RUNUP"])
def test_prepare_dataset_gate_filters(strategy):
    recipe = _recipe("gate", strategy)
    frame, dropped, kept = _gate_frame(strategy)
    prepared = prepare_dataset(recipe, frame)
    survivors = set(prepared.frame["tag"])
    assert dropped.isdisjoint(survivors)
    assert kept.issubset(survivors)


def test_prepare_dataset_chooser_filters():
    recipe = _recipe("chooser", "DYN-SV")
    frame, dropped, kept = _chooser_frame()
    prepared = prepare_dataset(recipe, frame)
    survivors = set(prepared.frame["tag"])
    assert dropped.isdisjoint(survivors)
    assert kept.issubset(survivors)


def test_prepare_dataset_key_columns_must_be_exactly_ticker_and_date_for_size():
    """size's real key_columns are ('ticker', 'date'); prepare_dataset's
    'needed' check requires both BY NAME. A hardcoded frame that has every
    other required column but not the real key name must refuse -- this
    would fail to refuse (and so fail this test) if key_columns drifted to
    a name our frame does not carry."""
    from engine.v2.models.training import RecipeDataError

    recipe = _recipe("size")
    frame, *_ = _size_frame("champion")
    assert prepare_dataset(recipe, frame) is not None  # sanity: full frame accepted
    with pytest.raises(RecipeDataError):
        prepare_dataset(recipe, frame.drop(columns=["ticker"]))
    with pytest.raises(RecipeDataError):
        prepare_dataset(recipe, frame.drop(columns=["date"]))
    with pytest.raises(RecipeDataError):
        prepare_dataset(recipe, frame.drop(columns=["year"]))


def test_prepare_dataset_key_columns_must_be_exactly_event_id_for_gate():
    from engine.v2.models.training import RecipeDataError

    recipe = _recipe("gate", "STR-THRU")
    frame, *_ = _gate_frame("STR-THRU")
    assert prepare_dataset(recipe, frame) is not None
    with pytest.raises(RecipeDataError):
        prepare_dataset(recipe, frame.drop(columns=["event_id"]))
    with pytest.raises(RecipeDataError):
        prepare_dataset(recipe, frame.drop(columns=["event_date"]))


def test_prepare_dataset_key_columns_must_be_exactly_candidate_id_for_chooser():
    from engine.v2.models.training import RecipeDataError

    recipe = _recipe("chooser", "DYN-SV")
    frame, *_ = _chooser_frame()
    assert prepare_dataset(recipe, frame) is not None
    with pytest.raises(RecipeDataError):
        prepare_dataset(recipe, frame.drop(columns=["candidate_id"]))


def test_prepare_dataset_key_columns_must_be_exactly_days_before_print_for_decision():
    """implied_t1/runup_move key_columns are (ticker, event_date,
    days_before_print) -- the third component only shows up here, not in
    the size/gate/chooser cases above."""
    from engine.v2.models.training import RecipeDataError

    recipe = _recipe("implied_t1")
    frame, *_ = _decision_frame("implied_t1", "champion")
    assert prepare_dataset(recipe, frame) is not None
    with pytest.raises(RecipeDataError):
        prepare_dataset(recipe, frame.drop(columns=["days_before_print"]))
    with pytest.raises(RecipeDataError):
        prepare_dataset(recipe, frame.drop(columns=["event_date"]))


# ==========================================================================
# Family 2: plan_folds -- folds.kind, min_train_rows, first_test_year,
# first_fold, full_refit
#
# filters/value_masks are stripped here (dataclasses.replace) -- they are
# Family 1's target and already killed there; leaving them in would make a
# filters mutation and a folds mutation indistinguishable through the same
# frame. membership_time_column/year_column/key_columns are read off the
# recipe only to PLUMB valid input (never asserted); folds.* stays real.
# ==========================================================================


def _stripped(recipe):
    return dataclasses.replace(recipe, filters=(), value_masks=())


def _years_frame(recipe, years, n_per_year=40, seed=0):
    parts = []
    for y in years:
        dates = pd.Timestamp(f"{y}-03-01") + pd.to_timedelta(np.arange(n_per_year), unit="D")
        cols = {recipe.target.column: 1.0, recipe.membership_time_column: dates}
        if recipe.year_column:
            cols[recipe.year_column] = dates.year
        for kc in recipe.key_columns:
            # A key column that is ALSO a feature (e.g. implied_t1/runup_move's
            # days_before_print) already got a finite random value from
            # _feature_frame; overwriting it with a non-numeric tag would
            # break completeness. Only tag key columns that are not features.
            if kc not in cols and kc not in recipe.features:
                cols[kc] = [f"{kc}{y}_{j}" for j in range(n_per_year)]
        parts.append(_bulk(recipe.features, n_per_year, seed=seed + y, **cols))
    return pd.concat(parts, ignore_index=True)


def _months_frame(recipe, months, n_per_month=40, seed=0):
    parts = []
    for i, m in enumerate(months):
        dates = m + pd.to_timedelta(np.arange(n_per_month) % 25, unit="D")
        cols = {recipe.target.column: 1.0, recipe.membership_time_column: dates}
        if recipe.year_column:
            cols[recipe.year_column] = dates.year
        for kc in recipe.key_columns:
            if kc not in cols and kc not in recipe.features:
                cols[kc] = [f"{kc}{i}_{j}" for j in range(n_per_month)]
        parts.append(_bulk(recipe.features, n_per_month, seed=1000 + seed + i, **cols))
    return pd.concat(parts, ignore_index=True)


def _bulk(features, n, seed, **cols):
    frame = _feature_frame(features, n, seed=seed)
    for k, v in cols.items():
        frame[k] = v
    return frame


EXPANDING_YEAR_ROLES = [
    ("size", "*", "champion", 2013), ("implied_t1", "*", "champion", 2015),
    ("runup_move", "*", "champion", 2018), ("iv_crush", "*", "champion", 2013),
    ("gate", "STR-THRU", "champion", 2020), ("gate", "STR-RUNUP", "champion", 2020),
    ("chooser", "DYN-SV", "champion", 2020),
]


@pytest.mark.parametrize("role,strategy,output,fty", EXPANDING_YEAR_ROLES,
                         ids=[f"{r}:{s}" for r, s, _o, _f in EXPANDING_YEAR_ROLES])
def test_plan_folds_expanding_year_kind_first_test_year_and_full_refit(role, strategy, output, fty):
    """Every champion recipe: kind='expanding_year', a wf-<year> fold per
    complete year from first_test_year, then one full-refit fold (full_refit
    is True for all 7). first_test_year is checked at its REAL, per-recipe
    value -- not shrunk or overridden."""
    recipe = _stripped(_recipe(role, strategy, output))
    small = dataclasses.replace(recipe, folds=dataclasses.replace(recipe.folds, min_train_rows=30))
    years = [fty - 2, fty - 1, fty, fty + 1]
    frame = _years_frame(small, years, n_per_year=40)
    prepared = prepare_dataset(small, frame)
    plans = plan_folds(small, prepared)
    ids = [p.fold_id for p in plans]
    assert f"wf-{fty - 1}" not in ids and f"wf-{fty - 2}" not in ids, (
        f"{role}:{strategy}:{output}: a year below first_test_year={fty} produced a fold")
    assert ids[:-1] == [f"wf-{fty}", f"wf-{fty + 1}"]
    assert ids[-1] == "full-refit"
    walk_forward = [p for p in plans if p.kind == "walk_forward"]
    assert len(walk_forward) == 2 and all(p.skipped is None for p in walk_forward)
    assert plans[-1].kind == "full_refit"


MONTHLY_ROLES = ["size", "implied_t1", "runup_move", "iv_crush"]


@pytest.mark.parametrize("role", MONTHLY_ROLES)
def test_plan_folds_monthly_cutoff_kind_and_first_fold(role):
    """Every tier4_monthly recipe: kind='monthly_cutoff', one m-<month>
    fold per month from first_fold (2013-01-01, shared by all four), never
    a full-refit fold (monthly plans never append one)."""
    recipe = _stripped(_recipe(role, output="tier4_monthly"))
    small = dataclasses.replace(recipe, folds=dataclasses.replace(recipe.folds, min_train_rows=30))
    months = [pd.Timestamp("2012-10-01"), pd.Timestamp("2012-11-01"), pd.Timestamp("2012-12-01"),
             pd.Timestamp("2013-01-01"), pd.Timestamp("2013-02-01")]
    frame = _months_frame(small, months, n_per_month=40)
    prepared = prepare_dataset(small, frame)
    plans = plan_folds(small, prepared)
    ids = [p.fold_id for p in plans]
    assert "m-2012-10" not in ids and "m-2012-11" not in ids and "m-2012-12" not in ids, (
        f"{role}: a month before first_fold=2013-01-01 produced a fold")
    assert ids == ["m-2013-01", "m-2013-02"]
    assert all(p.kind == "monthly" and p.skipped is None for p in plans)
    assert "full-refit" not in ids


def test_plan_folds_min_train_rows_boundary_is_exactly_500_not_499():
    """The one literal this cluster shares across all 11 recipes
    (``_MIN_TRAIN_ROWS = 500`` in recipes.py, referenced by name at every
    FoldScheme call site) -- one representative recipe (size champion) at
    its REAL, unshrunk min_train_rows and first_test_year kills the shared
    definition for all of them."""
    recipe = _stripped(_recipe("size", output="champion"))  # real: min_train_rows=500, first_test_year=2013

    def frame_with_prior_count(n_prior_complete):
        prior_dates = pd.Timestamp("2012-01-01") + pd.to_timedelta(np.arange(n_prior_complete) % 300, unit="D")
        prior = _bulk(recipe.features, n_prior_complete, seed=42,
                     **{recipe.target.column: 1.0, recipe.membership_time_column: prior_dates,
                        recipe.year_column: 2012, "ticker": [f"P{i}" for i in range(n_prior_complete)]})
        test_row = _bulk(recipe.features, 1, seed=43,
                        **{recipe.target.column: 1.0,
                           recipe.membership_time_column: pd.Series([pd.Timestamp("2013-06-01")]),
                           recipe.year_column: 2013, "ticker": ["TEST"]})
        return pd.concat([prior, test_row], ignore_index=True)

    under = frame_with_prior_count(499)
    prepared = prepare_dataset(recipe, under)
    plan = next(p for p in plan_folds(recipe, prepared) if p.fold_id == "wf-2013")
    assert plan.skipped is not None and len(plan.train) == 499

    at = frame_with_prior_count(500)
    prepared = prepare_dataset(recipe, at)
    plan = next(p for p in plan_folds(recipe, prepared) if p.fold_id == "wf-2013")
    assert plan.skipped is None and len(plan.train) == 500


# ==========================================================================
# Family 3: fold_receipts / receipt_issues -- label.* and upstream[i].*
# (produces, columns, lineage, model_id; upstream[i].source is never read
# by this logic and is Part 2's).
# ==========================================================================

FTY_MAP = {
    ("size", "*"): 2013, ("implied_t1", "*"): 2015, ("runup_move", "*"): 2018,
    ("iv_crush", "*"): 2013, ("gate", "STR-THRU"): 2020, ("gate", "STR-RUNUP"): 2020,
    ("chooser", "DYN-SV"): 2020,
}

LABEL_CASES = [
    ("size", "*", "champion", "label_available_at", 5),
    ("size", "*", "tier4_monthly", "label_available_at", 5),
    ("implied_t1", "*", "champion", "last_pre_print", 0),
    ("implied_t1", "*", "tier4_monthly", "last_pre_print", 0),
    ("runup_move", "*", "champion", "last_pre_print", 0),
    ("runup_move", "*", "tier4_monthly", "last_pre_print", 0),
    ("iv_crush", "*", "champion", "label_available_at", 5),
    ("iv_crush", "*", "tier4_monthly", "label_available_at", 5),
    ("gate", "STR-THRU", "champion", "exit_date", 5),
    ("gate", "STR-RUNUP", "champion", "exit_date", 0),
    ("chooser", "DYN-SV", "champion", "exit_date", None),
]


def _wf_fold(recipe, fty, n_per_year=60):
    small = dataclasses.replace(recipe, folds=dataclasses.replace(recipe.folds, min_train_rows=10))
    frame = _years_frame(small, [fty - 1, fty], n_per_year=n_per_year)
    frame[small.label.time_column] = frame[small.membership_time_column] + pd.Timedelta(days=1)
    prepared = prepare_dataset(small, frame)
    plan = next(p for p in plan_folds(small, prepared) if p.fold_id == f"wf-{fty}")
    return small, prepared, plan


def _monthly_fold(recipe, n_per_month=60):
    small = dataclasses.replace(recipe, folds=dataclasses.replace(recipe.folds, min_train_rows=10))
    months = [pd.Timestamp("2012-12-01"), pd.Timestamp("2013-01-01")]
    frame = _months_frame(small, months, n_per_month=n_per_month)
    frame[small.label.time_column] = frame[small.membership_time_column] + pd.Timedelta(days=1)
    prepared = prepare_dataset(small, frame)
    plan = next(p for p in plan_folds(small, prepared) if p.fold_id == "m-2013-01")
    return small, prepared, plan


@pytest.mark.parametrize("role,strategy,output,time_col,max_days", LABEL_CASES,
                         ids=[f"{r}:{s}:{o}" for r, s, o, _tc, _md in LABEL_CASES])
def test_fold_receipts_label_rule(role, strategy, output, time_col, max_days):
    """label.time_column and label.max_days_after_cutoff, through the real
    fold_receipts/receipt_issues consumer: a clean fold has no issues, and
    pushing one training row's label past the allowance raises FUTURE_LABEL
    -- except where the recipe's real rule is unbounded (chooser), where
    even a label years in the future must NOT be refused."""
    recipe = _stripped(_recipe(role, strategy, output))
    if output == "tier4_monthly":
        small, prepared, plan = _monthly_fold(recipe)
    else:
        small, prepared, plan = _wf_fold(recipe, FTY_MAP[(role, strategy)])
    assert len(plan.train) > 0
    fp = dataset_fingerprint(small, prepared)
    membership, label = fold_receipts(small, prepared, plan, dataset_fp=fp)
    assert label["label_time_column"] == time_col
    assert label["max_days_after_cutoff"] == max_days
    assert label["label_time_column_present"] is True
    assert label["n_label_time_missing"] == 0
    # gate:STR-THRU/chooser also carry upstream deps we have not populated
    # here (Family 3's dedicated upstream test does that); only the LABEL
    # codes are this test's concern.
    label_codes = {"FUTURE_LABEL", "LABEL_TIME_MISSING"}
    assert not [i for i in receipt_issues(small, membership, label) if i.code in label_codes]

    victim_pos = int(plan.train[0])
    violated = prepared.frame.copy()
    if max_days is None:
        violated.loc[victim_pos, time_col] = plan.cutoff + pd.Timedelta(days=20 * 365)
        _, label2 = fold_receipts(small, dataclasses.replace(prepared, frame=violated), plan, dataset_fp=fp)
        assert label2["n_labels_beyond_allowance"] is None
        assert label2["allowance"] == "unbounded (legacy sets no label-availability bound)"
    else:
        violated.loc[victim_pos, time_col] = plan.cutoff + pd.Timedelta(days=max_days + 1)
        membership2, label2 = fold_receipts(small, dataclasses.replace(prepared, frame=violated), plan,
                                            dataset_fp=fp)
        issues = receipt_issues(small, membership2, label2)
        assert any(i.code == "FUTURE_LABEL" for i in issues), (
            f"{role}:{strategy}:{output}: a label {max_days + 1}d after cutoff should refuse "
            f"(max_days_after_cutoff={max_days})")
        # Just INSIDE the allowance (one day short of the refusal limit)
        # must NOT refuse on FUTURE_LABEL.
        inside = prepared.frame.copy()
        inside.loc[victim_pos, time_col] = plan.cutoff + pd.Timedelta(days=max_days - 1)
        membership3, label3 = fold_receipts(small, dataclasses.replace(prepared, frame=inside), plan,
                                            dataset_fp=fp)
        issues3 = [i for i in receipt_issues(small, membership3, label3) if i.code in label_codes]
        assert not issues3, f"{role}:{strategy}:{output}: a label INSIDE the allowance was refused"


def _registry_champion_id(role):
    import json

    models = json.loads((Path(__file__).resolve().parents[1] / "engine/models/registry.json").read_text())["models"]
    rows = models.values() if isinstance(models, dict) else models
    return next(r["id"] for r in rows if r.get("champion") and r["role"] == role)


# (produces, columns, lineage, model_id_role or None for unrecorded)
GATE_THRU_UPSTREAM = [
    ("pred_abs_move", ("pred_abs_move", "pred_abs_move_p10", "pred_abs_move_p90", "pred_abs_move_sd",
                       "forecast_edge"), "tier4_monthly_oos", "size"),
    ("bucket_analog", ("analog_mean", "analog_win_rate", "analog_n"), "unrecorded", None),
]
CHOOSER_UPSTREAM = [
    ("pred_abs_move", ("pred_abs_move_p10", "pred_abs_move_p90", "tier4_pred_abs_move_sd",
                       "pred_abs_move_resid_n", "tier4_forecast_edge"), "tier4_monthly_oos", "size"),
    ("pred_im_t1_d14", ("pred_im_t1_d14", "pred_im_t1_d14_p10", "pred_im_t1_d14_p90"),
     "tier4_monthly_oos", "implied_t1"),
    ("pred_runup_abs_move_d14", ("pred_runup_abs_move_d14", "pred_runup_abs_move_d14_p10",
                                 "pred_runup_abs_move_d14_p90", "pred_runup_abs_move_d14_sd"),
     "tier4_monthly_oos", "runup_move"),
    ("candidate_forecast", ("pred_abs_move", "pred_abs_move_sd", "exp_pnl_sim", "exp_pnl_sim_select"),
     "unrecorded", None),
    ("causal_analog", ("analog_mean", "analog_win_rate", "analog_p10", "analog_p90", "analog_n"),
     "unrecorded", None),
]

UPSTREAM_CASES = ([("gate", "STR-THRU", d) for d in GATE_THRU_UPSTREAM] +
                  [("chooser", "DYN-SV", d) for d in CHOOSER_UPSTREAM])


@pytest.mark.parametrize("role,strategy,dep", UPSTREAM_CASES,
                         ids=[f"{r}:{s}:{d[0]}" for r, s, d in UPSTREAM_CASES])
def test_fold_receipts_upstream_dependency_fields(role, strategy, dep):
    """Every one of the 7 UpstreamDependency instances (gate:STR-THRU's 2,
    chooser's 5): produces, columns, lineage and model_id, through the real
    fold_receipts consumer. A correctly-populated frame must show every
    training row verified (tier4_monthly_oos) or unverified-by-design
    (unrecorded) -- a wrong produces/columns/lineage/model_id literal makes
    the frame's real columns not line up with what the recipe expects, and
    the count drops below len(train)."""
    produces, columns, lineage, model_id_role = dep
    recipe = _stripped(_recipe(role, strategy))
    small, prepared, plan = _wf_fold(recipe, FTY_MAP[(role, strategy)])
    frame = prepared.frame.copy()
    if lineage == "tier4_monthly_oos":
        member_time = pd.to_datetime(frame[small.membership_time_column])
        frame[f"{produces}_fold_start"] = member_time - pd.Timedelta(days=30)
        frame[f"{produces}_model_id"] = _registry_champion_id(model_id_role)
    prepared2 = dataclasses.replace(prepared, frame=frame)
    fp = dataset_fingerprint(small, prepared2)
    _, label = fold_receipts(small, prepared2, plan, dataset_fp=fp)
    entry = next(d for d in label["upstream"] if d["produces"] == produces)
    assert entry["lineage"] == lineage
    assert entry["columns"] == list(columns)
    n_train = len(plan.train)
    if lineage == "tier4_monthly_oos":
        assert entry["lineage_columns_present"] is True
        assert entry["expected_model_id"] == _registry_champion_id(model_id_role)
        assert entry["n_verified"] == n_train
        assert entry["n_lineage_missing"] == 0
        assert entry["n_in_sample"] == 0
        assert entry["n_model_mismatch"] == 0
    else:
        assert entry["n_verified"] == 0
        assert entry["n_unverified"] == n_train


# ==========================================================================
# Family 4: fit_recipe_estimator / job._summary -- estimator.kind,
# estimator.seeds, threshold.*
# ==========================================================================


def _xy(n=400, k=6, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    y = np.abs(X[:, 0] * 2.0 + X[:, 1] + rng.normal(size=n)) + 0.1
    return X, y


# Real seed for every single-seed recipe (registry.json's per-model "seed"
# is unset for all of these, so recipes.py falls back to LEGACY_SEED =
# 20260829). Hardcoded independently of recipe.estimator.seeds -- if that
# field drifted, fit_recipe_estimator would use the drifted seed while this
# constant stays put, and the legacy comparison below would stop matching.
REAL_LEGACY_SEED = 20260829

ESTIMATOR_LEGACY_MODULE = [
    ("size", "*", "champion", "engine.models.training.size_model"),
    ("size", "*", "tier4_monthly", "engine.models.training.size_model"),
    ("implied_t1", "*", "champion", "engine.models.training.implied_t1"),
    ("implied_t1", "*", "tier4_monthly", "engine.models.training.implied_t1"),
    ("runup_move", "*", "champion", "engine.models.training.runup_move"),
    ("runup_move", "*", "tier4_monthly", "engine.models.training.runup_move"),
    ("iv_crush", "*", "champion", "engine.models.training.iv_crush"),
    ("iv_crush", "*", "tier4_monthly", "engine.models.training.iv_crush"),
    ("gate", "STR-RUNUP", "champion", "engine.models.training.gate"),
    ("gate", "STR-THRU", "champion", "engine.models.training.gate_forecast_analog"),
]


@pytest.mark.parametrize("role,strategy,output,legacy_module", ESTIMATOR_LEGACY_MODULE,
                         ids=[f"{r}:{s}:{o}" for r, s, o, _m in ESTIMATOR_LEGACY_MODULE])
def test_fit_recipe_estimator_matches_legacy_fit_for_every_recipe(role, strategy, output, legacy_module):
    """estimator.kind and estimator.seeds, for all 10 single-seed recipes
    (chooser's 5-seed ensemble is separate, below): fit_recipe_estimator's
    prediction must equal the named legacy module's fit(X, y, seed) at the
    REAL, hardcoded seed -- not a seed read back off the recipe, which would
    make a seed mutation invisible (native and the comparison would drift
    together)."""
    import importlib

    recipe = _recipe(role, strategy, output)
    module = importlib.import_module(legacy_module)
    X, y = _xy()
    native = fit_recipe_estimator(recipe, X, y)
    legacy = module.fit(X, y, REAL_LEGACY_SEED)
    assert np.array_equal(np.asarray(native.predict(X), float).ravel(),
                          np.asarray(legacy.predict(X), float).ravel())


def test_fit_recipe_estimator_chooser_ensemble_seeds_and_kind():
    """chooser's estimator.kind='quantile_mlp_ensemble' with 5 seeds
    (20260908..20260912, EXP-169's SEEDS -- hardcoded, matching
    test_chooser_recipe_matches_exp169_fit_head_literals' independent
    source, not read back off the recipe)."""
    from engine.v2.models.training.estimators import SeedMeanEnsemble

    real_seeds = (20260908, 20260909, 20260910, 20260911, 20260912)
    recipe = _recipe("chooser", "DYN-SV")
    fast = dataclasses.replace(recipe, estimator=dataclasses.replace(
        recipe.estimator, params={**recipe.estimator.params,
                                  "mlp": {**recipe.estimator.params["mlp"], "max_iter": 5}}))
    X, y = _xy(n=300, k=4)
    with pytest.warns(Warning):  # 5 iterations does not converge; speed only
        native = fit_recipe_estimator(fast, X, y)
    assert isinstance(native, SeedMeanEnsemble)
    assert len(native.models) == 5
    assert [m[-1].random_state for m in native.models] == list(real_seeds)


# --------------------------------------------------------------------------
# job._summary -- threshold.kind, threshold.top_fraction
# --------------------------------------------------------------------------

REAL_TOP_FRACTION = 0.20  # gate.TOP_FRACTION -- hardcoded independently


class _IdentityModel:
    def predict(self, X):
        return np.asarray(X, dtype=float)[:, 0]


class _IdentityFit:
    """Ignores y; predicts the row's first feature. Lets the test compute
    the expected across-fold quantile by hand from values it chose. A
    module-level (not nested) class -- run_training_job's fold artifact is
    joblib-pickled, which cannot serialize a local class."""

    def __call__(self, recipe, X, y):
        return _IdentityModel()


@pytest.mark.parametrize("strategy", ["STR-THRU", "STR-RUNUP"])
def test_job_summary_threshold_matches_gate_top_fraction_quantile(tmp_path, strategy):
    """threshold.kind='oos_top_fraction_quantile' and
    threshold.top_fraction=0.20, through the real job._summary consumer:
    the written summary.json's 'threshold' must equal
    np.quantile(all OOS test predictions, 1 - 0.20), computed independently
    here from values this test controls (an identity-fit stub), not from
    recipe.threshold itself."""
    # upstream is Family 3's target (already killed there); stripped here so
    # this test's minimal frame does not also need fold-lineage columns.
    recipe = dataclasses.replace(_stripped(_recipe("gate", strategy)), upstream=())
    small = dataclasses.replace(recipe, folds=dataclasses.replace(recipe.folds, min_train_rows=10))
    frame = _years_frame(small, [2019, 2020, 2021], n_per_year=50, seed=500)
    frame[small.label.time_column] = frame[small.membership_time_column]
    # The stub predicts feature 0; give every row a distinct, known value so
    # the across-fold quantile is computable by hand.
    rng = np.random.default_rng(11)
    frame[small.features[0]] = rng.uniform(0, 1000, size=len(frame))

    result = run_training_job(small, frame, tmp_path / "gate", fit=_IdentityFit())
    summary = _summary(small, tmp_path / "gate", result.outcomes)
    assert summary["recipe_id"] == small.recipe_id

    import pandas as pd

    from engine.v2.models.training import fold_store as store

    test_preds = []
    for outcome in result.outcomes:
        path = tmp_path / "gate" / "folds" / outcome.fold_id / store.PREDICTIONS_FILE
        if path.is_file():
            test_preds.append(pd.read_parquet(path)["pred"].to_numpy(dtype=float))
    expected_values = np.concatenate(test_preds)
    expected = float(np.quantile(expected_values, 1.0 - REAL_TOP_FRACTION))
    assert summary["threshold_n"] == expected_values.size
    assert summary["threshold"] == pytest.approx(expected)


@pytest.mark.parametrize("key", list(RECIPES), ids=lambda k: k.label())
def test_job_summary_omits_threshold_for_non_threshold_recipes(key, tmp_path):
    """Every recipe except the 2 gate ones has threshold=None; _summary
    must not compute or write a 'threshold' key for them (the
    'recipe.threshold is not None' branch in job._summary)."""
    if key.role == "gate":
        pytest.skip("gate recipes DO carry a threshold; covered above")
    recipe = RECIPES[key]
    summary = _summary(recipe, tmp_path, ())
    assert "threshold" not in summary and "threshold_n" not in summary
