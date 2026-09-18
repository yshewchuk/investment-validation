"""The current training recipes, extracted from legacy as declarative data.

P5-3 (``guides/rearchitecture_phase5_models.md``): every champion the
registry serves, every Tier-4 monthly producer and every live-refit
calibration surface gets one :class:`TrainingRecipe` recording its
membership, target, missing mask, transforms, folds, seeds and upstream
out-of-sample dependencies. **Legacy behaviour is the spec.** Each field
below cites the legacy symbol it was read from (``legacy_refs``), and
``tests/test_v2_models_training_recipes.py`` cross-checks the constants
against those symbols so a legacy edit that is not mirrored here fails.

Keys follow the P5-1 inventory: ``(role, strategy)`` of a registry binding
(``engine.v2.models.inventory.current_release_inventory``) plus an
``output`` that keeps the two separate outputs the parity rules name apart
— ``champion`` (expanding-year walk-forward evidence plus the full refit
that serves) and ``tier4_monthly`` (the monthly causal folds whose
predictions are the stored Tier-4 columns). Calibration surfaces use the
``NON_MODEL_STATE_ITEMS`` names and ``output="calibration"``; their fit
belongs to P5-4 (``fit_owner``), so the training job here refuses to fit
them and only produces their membership/label receipts.

Feature order is read from the git-tracked ``engine/models/registry.json``
(the frozen manifest of what each champion was fit on) as plain JSON; this
module imports no legacy code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from engine.v2.foundation import content_hash

__all__ = [
    "CLOCK_ID",
    "LEGACY_SEED",
    "OWNER_TRAINING_JOB",
    "OWNER_P5_4",
    "RECIPE_V1",
    "EstimatorSpec",
    "FoldScheme",
    "LabelRule",
    "RecipeKey",
    "ResidualRule",
    "RowFilter",
    "TargetSpec",
    "ThresholdRule",
    "TrainingRecipe",
    "UpstreamDependency",
    "ValueMask",
    "current_recipes",
    "recipe_fingerprint",
]

RECIPE_V1 = "p5.training_recipe.v1"

#: ``engine.models.training.common.SEED``.
LEGACY_SEED = 20260829

#: The only clock any current binding uses (``inventory.KNOWN_CLOCK_IDS``).
CLOCK_ID = "legacy.entry_close.v1"

#: Who may fit a recipe. The P5-4 surfaces are receipt-only here.
OWNER_TRAINING_JOB = "engine.v2.models.training.job"
OWNER_P5_4 = "P5-4 frozen payoff/recalibration artifact"

_REGISTRY_JSON = Path(__file__).resolve().parents[3] / "models" / "registry.json"


# --------------------------------------------------------------------------
# recipe vocabulary
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeKey:
    role: str
    strategy: str
    output: str  # "champion" | "tier4_monthly" | "calibration"

    def label(self) -> str:
        return f"{self.role}:{self.strategy}:{self.output}"


@dataclass(frozen=True)
class RowFilter:
    """One membership rule. ``op`` is one of :data:`FILTER_OPS`."""

    column: str
    op: str
    value: Any = None


FILTER_OPS = ("eq", "isclose", "ge", "notna", "year_between", "not_before_column",
              "group_nunique_ge")


@dataclass(frozen=True)
class ValueMask:
    """Values outside the closed ``[lo, hi]`` become missing (legacy BOUNDS)."""

    column: str
    lo: float
    hi: float


@dataclass(frozen=True)
class TargetSpec:
    column: str
    definition: str
    signed: bool
    #: Applied inside the estimator; predictions come back in target units
    #: (``log1p_clip0``) or stay in the transformed space (``quantile_normal``).
    transform: str | None = None


@dataclass(frozen=True)
class LabelRule:
    """When a row's label became observable, and legacy's tolerance for it.

    ``time_column`` is the dataset column holding the label-availability
    date. ``max_days_after_cutoff`` is how far past a fold cutoff legacy's
    own membership rule lets a label land (a print on the last day of the
    fold is realized at the next session); ``None`` means legacy imposes no
    bound, which the receipt then reports rather than enforces.
    """

    time_column: str
    definition: str
    max_days_after_cutoff: int | None


@dataclass(frozen=True)
class UpstreamDependency:
    """An input column that is another model's prediction.

    ``lineage="tier4_monthly_oos"``: the dataset carries the Tier-4 group's
    ``<produces>_fold_start``/``<produces>_model_id`` columns, and every
    training row's upstream fold must start at or before the row's own
    membership time (so the upstream model never trained on the row).
    ``lineage="unrecorded"``: the source table keeps no lineage; receipts
    count the rows as unverified instead of passing them.
    """

    produces: str
    columns: tuple[str, ...]
    lineage: str
    model_id: str | None = None
    source: str = ""

    @property
    def fold_column(self) -> str:
        return f"{self.produces}_fold_start"

    @property
    def model_id_column(self) -> str:
        return f"{self.produces}_model_id"


@dataclass(frozen=True)
class FoldScheme:
    """``expanding_year``: train year < Y, test year == Y, from
    ``first_test_year``; skipped below ``min_train_rows``; plus the full
    refit when ``full_refit``. ``monthly_cutoff``: train ``time < month
    start``, test the month's scorable rows, from ``first_fold``.
    ``request_cutoff``: one fold per scoring cutoff (calibration)."""

    kind: str
    min_train_rows: int
    first_test_year: int | None = None
    first_fold: str | None = None
    full_refit: bool = False


@dataclass(frozen=True)
class EstimatorSpec:
    kind: str
    params: Mapping[str, Any]
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class ResidualRule:
    kind: str  # "walk_forward_oos" | "tier4_earlier_folds" | "payoff_fit_residuals" | "none"
    bucket_deciles: int | None = None
    bucket_min_pool: int | None = None
    min_pool: int | None = None


@dataclass(frozen=True)
class ThresholdRule:
    kind: str  # "oos_top_fraction_quantile"
    top_fraction: float


@dataclass(frozen=True)
class TrainingRecipe:
    key: RecipeKey
    recipe_id: str
    output_id: str | None
    produces: str | None
    clock_id: str
    dataset_source: str
    key_columns: tuple[str, ...]
    membership_time_column: str
    year_column: str | None
    filters: tuple[RowFilter, ...]
    value_masks: tuple[ValueMask, ...]
    target: TargetSpec
    features: tuple[str, ...]
    missing_mask: str
    label: LabelRule
    folds: FoldScheme
    estimator: EstimatorSpec
    upstream: tuple[UpstreamDependency, ...] = ()
    residuals: ResidualRule = ResidualRule("none")
    threshold: ThresholdRule | None = None
    fit_owner: str = OWNER_TRAINING_JOB
    legacy_refs: tuple[str, ...] = ()
    notes: str = ""
    schema_version: str = field(default=RECIPE_V1)

    def as_dict(self) -> dict:
        return _plain(asdict(self))


#: Missing-mask policies: legacy ``walk_forward``/``fit_final``/Tier-4
#: ``training_frames`` drop any row with a non-finite feature or target.
MISSING_DROP_INCOMPLETE = "drop_rows_with_nonfinite_feature_or_target"


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def recipe_fingerprint(recipe: TrainingRecipe) -> str:
    """Content hash over every field; any recipe change is a new identity."""
    return content_hash(recipe.as_dict())


# --------------------------------------------------------------------------
# the extracted estimators (hyperparameters copied from the legacy fit())
# --------------------------------------------------------------------------

#: size_model.fit: OLS + StandardScaler/MLPRegressor, equal-weight BlendModel.
_SIZE_ESTIMATOR = {
    "mlp": {"hidden_layer_sizes": [64, 32], "max_iter": 400, "early_stopping": True,
            "n_iter_no_change": 15},
    "scale_mlp_inputs": True,
    "blend": "equal_weight_mean(ols, mlp)",
}
#: implied_t1.fit.
_IMPLIED_T1_HGB = {"max_iter": 300, "learning_rate": 0.06, "max_leaf_nodes": 31,
                   "min_samples_leaf": 40, "l2_regularization": 1.0,
                   "early_stopping": True, "n_iter_no_change": 20}
#: runup_move.fit and gate.fit share these.
_RUNUP_GATE_HGB = {"max_iter": 250, "learning_rate": 0.05, "max_leaf_nodes": 15,
                   "min_samples_leaf": 60, "l2_regularization": 2.0,
                   "early_stopping": True, "n_iter_no_change": 20}
#: iv_crush.fit: sklearn defaults apart from these two.
_IV_CRUSH_HGB = {"learning_rate": 0.06, "max_iter": 300}
#: EXP-169 fit_head (the promoted chooser's configuration).
_CHOOSER_SEEDS = (20260908, 20260909, 20260910, 20260911, 20260912)
_CHOOSER_ESTIMATOR = {
    "target_quantile": {"output_distribution": "normal", "n_quantiles_cap": 1000},
    "mlp": {"hidden_layer_sizes": [64, 32], "max_iter": 800, "early_stopping": True,
            "validation_fraction": 0.15, "n_iter_no_change": 50},
    "scale_mlp_inputs": True,
    "ensemble": "mean over seeds",
}

#: tier4.FIRST_FOLD / MIN_TRAIN_ROWS; walk_forward min_train_rows.
_TIER4_FIRST_FOLD = "2013-01-01"
_MIN_TRAIN_ROWS = 500
#: tier4.MIN_RESIDUALS; registry.bucket_residuals defaults.
_TIER4_MIN_RESIDUALS = 250

#: size_model.BOUNDS / MIN_PRIOR.
_SIZE_BOUNDS = (ValueMask("or_implied", 0.0, 60.0), ValueMask("or_rvol30", 0.0, 700.0),
                ValueMask("abs_move", 0.0, 200.0))
_SIZE_MIN_PRIOR = 4
#: tier4.IM_T1_YEARS == train_all._events_with_session's default years.
_EVENT_YEARS = (2017, 2026)
#: gate.GATE_ALPHA / TOP_FRACTION.
_GATE_ALPHA = 0.5
_GATE_TOP_FRACTION = 0.20

#: A print on the fold's last day is realized at the next session: a weekend
#: plus a holiday is at most 5 calendar days (``iv_crush.MAX_GAP_DAYS``).
_POST_PRINT_DAYS = 5


def _registry_entries(path: Path | None) -> dict[str, dict]:
    data = json.loads(Path(path or _REGISTRY_JSON).read_text())
    models = data["models"]
    rows = models.values() if isinstance(models, dict) else models
    return {row["id"]: row for row in rows}


def _champions(entries: Mapping[str, dict]) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for row in entries.values():
        if row.get("champion"):
            out[(row["role"], row["strategy"])] = row
    return out


def _tier4(produces: str, columns: tuple[str, ...], model_id: str) -> UpstreamDependency:
    return UpstreamDependency(
        produces=produces, columns=columns, lineage="tier4_monthly_oos", model_id=model_id,
        source="data/features/tier4_forecasts.parquet (engine.data.features.tier4.build_producer)",
    )


def current_recipes(registry_path: Path | None = None) -> dict[RecipeKey, TrainingRecipe]:
    """Every current recipe, keyed by inventory role/strategy/output."""
    from .calibration import calibration_recipes

    champ = _champions(_registry_entries(registry_path))

    def entry(role: str, strategy: str = "*") -> dict:
        try:
            return champ[(role, strategy)]
        except KeyError:
            raise KeyError(f"registry has no champion for {role}:{strategy}") from None

    recipes = [*_size_recipes(entry), *_decision_recipes(entry), *_crush_recipes(entry),
               *_gate_recipes(entry), *_chooser_recipes(entry, champ), *calibration_recipes()]
    out = {r.key: r for r in recipes}
    if len(out) != len(recipes):  # pragma: no cover - construction invariant
        raise ValueError("duplicate recipe keys")
    return out


def _size_recipes(entry) -> list[TrainingRecipe]:
    """size_model.train / tier4.size_feature_model."""
    recipes: list[TrainingRecipe] = []
    size = entry("size")
    size_common = dict(
        dataset_source="Tier-3 panel (engine.features.load_panel) through "
                       "engine.models.training.size_model.prepare",
        key_columns=("ticker", "date"), membership_time_column="date", year_column="year",
        filters=(RowFilter("n_prior", "ge", _SIZE_MIN_PRIOR),),
        value_masks=_SIZE_BOUNDS,
        target=TargetSpec("abs_move", "absolute earnings move across the print, percent of spot",
                          signed=False),
        features=tuple(size["features"]), missing_mask=MISSING_DROP_INCOMPLETE,
        label=LabelRule("label_available_at", "post-print close of the event",
                        _POST_PRINT_DAYS),
        estimator=EstimatorSpec("ols_mlp_blend", _SIZE_ESTIMATOR, (int(size.get("seed") or LEGACY_SEED),)),
    )
    recipes.append(TrainingRecipe(
        key=RecipeKey("size", "*", "champion"), recipe_id="size.champion.v1",
        output_id=size["id"], produces=size.get("produces"), clock_id=CLOCK_ID,
        folds=FoldScheme("expanding_year", _MIN_TRAIN_ROWS, first_test_year=2013, full_refit=True),
        residuals=ResidualRule("walk_forward_oos", bucket_deciles=10, bucket_min_pool=250),
        legacy_refs=("engine.models.training.size_model.train", "engine.models.training.common.walk_forward",
                     "engine.models.training.common.fit_final", "engine.models.training.train_all.train_size",
                     "engine.models.registry.bucket_residuals"),
        **size_common,
    ))
    recipes.append(TrainingRecipe(
        key=RecipeKey("size", "*", "tier4_monthly"), recipe_id="size.tier4_monthly.v1",
        output_id=size["id"], produces="pred_abs_move", clock_id=CLOCK_ID,
        folds=FoldScheme("monthly_cutoff", _MIN_TRAIN_ROWS, first_fold=_TIER4_FIRST_FOLD),
        residuals=ResidualRule("tier4_earlier_folds", min_pool=_TIER4_MIN_RESIDUALS),
        legacy_refs=("engine.data.features.tier4.size_feature_model", "engine.data.features.tier4.build_producer",
                     "engine.data.features.tier4.fit_fold"),
        **size_common,
    ))
    return recipes


def _decision_recipes(entry) -> list[TrainingRecipe]:
    """implied_t1 and runup_move: the session-aware T-14 decision frame."""
    recipes: list[TrainingRecipe] = []
    decision_filters = (RowFilter("session", "notna"), RowFilter("year", "year_between", _EVENT_YEARS))
    im = entry("implied_t1")
    runup = entry("runup_move")
    for role, row, target, estimator, first_test, refs, notes in (
        ("implied_t1", im,
         TargetSpec("im_t1", "quoted implied move at the last pre-print close", signed=False),
         EstimatorSpec("hgb", _IMPLIED_T1_HGB, (int(im.get("seed") or LEGACY_SEED),)),
         2015, ("engine.models.training.implied_t1.build_dataset", "engine.models.training.implied_t1.train",
                "engine.data.features.tier4.im_t1_feature_model"),
         "champion pools DECISION_DAYS (25,20,15,14,10,7,5,3,2); the Tier-4 column uses day 14 only"),
        ("runup_move", runup,
         TargetSpec("runup_abs_move_d14", "100*|log(spot at last pre-print close / spot at T-14)|",
                    signed=False, transform="log1p_clip0"),
         EstimatorSpec("hgb", _RUNUP_GATE_HGB, (int(runup.get("seed") or LEGACY_SEED),)),
         2018, ("engine.models.training.runup_move.build_dataset", "engine.models.training.runup_move.train",
                "engine.data.features.tier4.runup_move_feature_model"),
         "horizon 14 trading days; direction not modeled"),
    ):
        day14 = (RowFilter("days_before_print", "isclose", 14),)
        common = dict(
            key_columns=("ticker", "event_date", "days_before_print"),
            membership_time_column="event_date", year_column="year",
            value_masks=(), target=target,
            features=tuple(row["features"]), missing_mask=MISSING_DROP_INCOMPLETE,
            label=LabelRule("last_pre_print", "the last pre-print close (the exit close)", 0),
            estimator=estimator, notes=notes,
        )
        buckets = role == "runup_move"
        recipes.append(TrainingRecipe(
            key=RecipeKey(role, "*", "champion"), recipe_id=f"{role}.champion.v1",
            output_id=row["id"], produces=row.get("produces"), clock_id=CLOCK_ID,
            dataset_source=f"earnings_events {_EVENT_YEARS[0]}-{_EVENT_YEARS[1]} with session, "
                           f"through {refs[0]}",
            folds=FoldScheme("expanding_year", _MIN_TRAIN_ROWS, first_test_year=first_test, full_refit=True),
            residuals=ResidualRule("walk_forward_oos", bucket_deciles=10 if buckets else None,
                                   bucket_min_pool=250 if buckets else None),
            legacy_refs=refs[:2] + ("engine.models.training.train_all.train_" + role,),
            # implied_t1 pools every DECISION_DAY; runup_move builds day 14 only.
            filters=decision_filters + (day14 if role == "runup_move" else ()),
            **common,
        ))
        recipes.append(TrainingRecipe(
            key=RecipeKey(role, "*", "tier4_monthly"), recipe_id=f"{role}.tier4_monthly.v1",
            output_id=row["id"], produces=row.get("produces"), clock_id=CLOCK_ID,
            dataset_source=f"earnings_events {_EVENT_YEARS[0]}-{_EVENT_YEARS[1]} with session, decision "
                           f"day 14 only, through {refs[0]}",
            folds=FoldScheme("monthly_cutoff", _MIN_TRAIN_ROWS, first_fold=_TIER4_FIRST_FOLD),
            residuals=ResidualRule("tier4_earlier_folds", min_pool=_TIER4_MIN_RESIDUALS),
            legacy_refs=(refs[2], "engine.data.features.tier4.build_producer",
                         "engine.data.features.tier4.fit_fold"),
            filters=decision_filters + day14,  # tier4.IM_T1_HORIZON / runup_move.HORIZON
            **common,
        ))
    return recipes


def _crush_recipes(entry) -> list[TrainingRecipe]:
    """iv_crush.prepare / train and tier4.iv_crush_feature_model."""
    recipes: list[TrainingRecipe] = []
    crush = entry("iv_crush")
    crush_common = dict(
        dataset_source="Tier-3 panel inner-joined on the realized crush "
                       "(engine.models.training.iv_crush.prepare / crush_frame)",
        key_columns=("ticker", "date"), membership_time_column="date", year_column="year",
        filters=(), value_masks=(),
        target=TargetSpec("crush_pct_iv30", "100*(post-print iv30 / pre-print iv30 - 1); "
                          "pre/post sessions at most 5 calendar days apart", signed=True),
        features=tuple(crush["features"]), missing_mask=MISSING_DROP_INCOMPLETE,
        label=LabelRule("label_available_at", "post-print close (next quoted session)",
                        _POST_PRINT_DAYS),
        estimator=EstimatorSpec("hgb", _IV_CRUSH_HGB, (int(crush.get("seed") or LEGACY_SEED),)),
    )
    recipes.append(TrainingRecipe(
        key=RecipeKey("iv_crush", "*", "champion"), recipe_id="iv_crush.champion.v1",
        output_id=crush["id"], produces=crush.get("produces"), clock_id=CLOCK_ID,
        folds=FoldScheme("expanding_year", _MIN_TRAIN_ROWS, first_test_year=2013, full_refit=True),
        residuals=ResidualRule("walk_forward_oos"),
        legacy_refs=("engine.models.training.iv_crush.prepare", "engine.models.training.iv_crush.train",
                     "engine.models.training.train_all.train_iv_crush"),
        **crush_common,
    ))
    recipes.append(TrainingRecipe(
        key=RecipeKey("iv_crush", "*", "tier4_monthly"), recipe_id="iv_crush.tier4_monthly.v1",
        output_id=crush["id"], produces=crush.get("produces"), clock_id=CLOCK_ID,
        folds=FoldScheme("monthly_cutoff", _MIN_TRAIN_ROWS, first_fold=_TIER4_FIRST_FOLD),
        residuals=ResidualRule("tier4_earlier_folds", min_pool=_TIER4_MIN_RESIDUALS),
        legacy_refs=("engine.data.features.tier4.iv_crush_feature_model",
                     "engine.data.features.tier4.build_producer", "engine.data.features.tier4.fit_fold"),
        notes="signed target: Tier-4 interval_floor=None",
        **crush_common,
    ))
    return recipes


def _gate_recipes(entry) -> list[TrainingRecipe]:
    """train_all.train_gate / train_gate_forecast_analog."""
    recipes: list[TrainingRecipe] = []
    size_id = entry("size")["id"]
    for strategy in ("STR-THRU", "STR-RUNUP"):
        gate = entry("gate", strategy)
        upstream: tuple[UpstreamDependency, ...] = ()
        refs = ["engine.models.training.train_all._engine_trades",
                "engine.models.training.gate.build_dataset", "engine.models.training.gate.train",
                "engine.models.training.gate.choose_threshold"]
        if "pred_abs_move" in gate["features"]:
            upstream = (
                _tier4("pred_abs_move", ("pred_abs_move", "pred_abs_move_p10", "pred_abs_move_p90",
                                         "pred_abs_move_sd", "forecast_edge"), size_id),
                UpstreamDependency("bucket_analog", ("analog_mean", "analog_win_rate", "analog_n"),
                                   lineage="unrecorded",
                                   source="engine.analogs.AnalogMatcher via "
                                          "gate_forecast_analog._attach_analogs (live-refit population)"),
            )
            refs += ["engine.models.training.gate_forecast_analog.build_dataset",
                     "engine.models.training.train_all.train_gate_forecast_analog"]
        # STR-THRU exits one session after the print; STR-RUNUP at the last
        # pre-print close (structures: exit_offset 1 / 0).
        lag = _POST_PRINT_DAYS if strategy == "STR-THRU" else 0
        recipes.append(TrainingRecipe(
            key=RecipeKey("gate", strategy, "champion"), recipe_id=f"gate.{strategy}.champion.v1",
            output_id=gate["id"], produces=gate.get("produces"), clock_id=CLOCK_ID,
            dataset_source=f"trades table, engine.replay {strategy} rows, D0 decisions",
            key_columns=("event_id",), membership_time_column="event_date", year_column="year",
            filters=(RowFilter("strategy", "eq", strategy),
                     RowFilter("provenance", "eq", "engine.replay"),
                     RowFilter("decision_date", "not_before_column", "entry_date"),
                     RowFilter("fill_alpha", "isclose", _GATE_ALPHA)),
            value_masks=(),
            target=TargetSpec("ret", "realized per-trade return at mid fills (alpha 0.5)", signed=True),
            features=tuple(gate["features"]), missing_mask=MISSING_DROP_INCOMPLETE,
            label=LabelRule("exit_date", "trade exit close", lag),
            folds=FoldScheme("expanding_year", _MIN_TRAIN_ROWS, first_test_year=2020, full_refit=True),
            estimator=EstimatorSpec("hgb", _RUNUP_GATE_HGB, (int(gate.get("seed") or LEGACY_SEED),)),
            upstream=upstream,
            residuals=ResidualRule("walk_forward_oos"),
            threshold=ThresholdRule("oos_top_fraction_quantile", _GATE_TOP_FRACTION),
            legacy_refs=tuple(refs),
        ))
    return recipes


def _chooser_recipes(entry, champ) -> list[TrainingRecipe]:
    """EXP-169 generate/fit_head over chooser.build_dataset."""
    recipes: list[TrainingRecipe] = []
    chooser = entry("chooser", "DYN-SV")
    tier4_ids = {row.get("produces"): row["id"] for row in champ.values() if row.get("produces")}
    chooser_upstream = (
        _tier4("pred_abs_move", ("pred_abs_move_p10", "pred_abs_move_p90", "tier4_pred_abs_move_sd",
                                 "pred_abs_move_resid_n", "tier4_forecast_edge"), tier4_ids["pred_abs_move"]),
        _tier4("pred_im_t1_d14", ("pred_im_t1_d14", "pred_im_t1_d14_p10", "pred_im_t1_d14_p90"),
               tier4_ids["pred_im_t1_d14"]),
        _tier4("pred_runup_abs_move_d14", ("pred_runup_abs_move_d14", "pred_runup_abs_move_d14_p10",
                                           "pred_runup_abs_move_d14_p90", "pred_runup_abs_move_d14_sd"),
               tier4_ids["pred_runup_abs_move_d14"]),
        UpstreamDependency("candidate_forecast", ("pred_abs_move", "pred_abs_move_sd", "exp_pnl_sim",
                                                  "exp_pnl_sim_select"), lineage="unrecorded",
                           source="EXP-161 candidates table columns (no fold lineage stored)"),
        UpstreamDependency("causal_analog", ("analog_mean", "analog_win_rate", "analog_p10", "analog_p90",
                                             "analog_n"), lineage="unrecorded",
                           source="EXP-161 base.add_causal_analogs"),
    )
    recipes.append(TrainingRecipe(
        key=RecipeKey("chooser", "DYN-SV", "champion"), recipe_id="chooser.DYN-SV.champion.v1",
        output_id=chooser["id"], produces=chooser.get("produces"), clock_id=CLOCK_ID,
        dataset_source="EXP-169 menu7-prime candidates through engine.models.training.chooser.build_dataset",
        key_columns=("candidate_id",), membership_time_column="event_date", year_column="year",
        filters=(RowFilter("fill_alpha", "isclose", 0.5),
                 RowFilter("event_id", "group_nunique_ge", ("strategy", 2))),
        value_masks=(),
        target=TargetSpec("dev_target", "realized pnl minus the mean over the structures offered "
                          "on the same event", signed=True, transform="quantile_normal"),
        features=tuple(chooser["features"]), missing_mask=MISSING_DROP_INCOMPLETE,
        label=LabelRule("exit_date", "candidate exit close (menu structures; legacy sets no bound)", None),
        folds=FoldScheme("expanding_year", _MIN_TRAIN_ROWS, first_test_year=2020, full_refit=True),
        estimator=EstimatorSpec("quantile_mlp_ensemble", _CHOOSER_ESTIMATOR, _CHOOSER_SEEDS),
        upstream=chooser_upstream,
        residuals=ResidualRule("none"),
        legacy_refs=("engine.models.training.chooser.build_dataset",
                     "experiments/EXP-169_menu7prime_confirmation/run.py:generate",
                     "experiments/EXP-169_menu7prime_confirmation/run.py:fit_head"),
        notes="Walk-forward folds are EXP-169's generate(). The deployment full refit "
              "('all exit-complete menu7-prime events, 2018-2026') has no code in the repo; "
              "the full_refit fold here is fit on every complete row, which is NOT proven "
              "to be the registered artifact's membership.",
    ))
    return recipes
