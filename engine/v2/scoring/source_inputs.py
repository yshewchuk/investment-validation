"""Build bounded native scoring inputs from answer-free source material."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping, Sequence

from engine.v2.domain.generation import DISABLED, STRATEGIES
from engine.v2.models.admissible_table import AdmissibleDepthTable
from engine.v2.models.analog_artifact import BoardAnalogPoolArtifact
from engine.v2.models.chooser_analog_pool import ChooserAnalogPoolArtifact
from engine.v2.models.contracts import ModelBinding, ModelRelease
from engine.v2.models.loader import FrozenInference
from engine.v2.models.payoff_artifact import PayoffLineArtifact, PayoffSurfaceArtifact
from engine.v2.models.recalibration_artifact import RecalibrationMapArtifact
from engine.v2.models.residual_artifact import (
    DriverResidualPoolArtifact,
    PairedResidualPoolArtifact,
)
from engine.v2.scoring.frozen_executor import (
    FrozenRecipeExecutor,
    FrozenStageExecutor,
)
from engine.v2.scoring.native_analog import (
    BUCKET_RECIPE_SCHEMA,
    FROZEN_ANALOG_RECIPE_FIELDS,
    bucket_population_hash,
)
from engine.v2.scoring.stages import (
    NativeScoreInputs,
    StageReceipt,
    receipt,
)

__all__ = ["SourceBundle", "build_native_score_inputs", "fold_pool"]

_ANSWER_FIELDS = frozenset({
    "legs", "selected_legs", "selected_contracts", "resolved_legs",
    "entry_cost", "structure_width", "model_artifact_ids", "forecasts",
    "driver_prediction", "forecast_abs_move", "forecast_p10",
    "forecast_p90", "forecast_sd", "runup_move_prediction",
    "runup_move_raw_d14", "runup_move_raw_d14_p10",
    "runup_move_raw_d14_p90", "runup_move_raw_d14_sd", "runup_move_p10",
    "runup_move_p90", "runup_move_sd", "pred_iv_crush",
    "pred_iv_crush_30", "model_fair_pct", "exp_pnl_sim", "win_sim",
    "sim_p10", "sim_p90", "pool_n", "exp_pnl_analog", "win_analog",
    "ci_low", "ci_high", "n_analogs", "gate_score", "gate_threshold",
    "gate_pass", "gate_decision", "chooser_score", "chooser_candidates",
    "chooser_selection", "chosen_strategy", "chosen_margin", "menu_size",
    "financial_diagnostics", "entry_cost_pct", "model_vs_market",
    "fair_premium_pct", "premium_vs_fair", "cost_over_width",
    "terminal_payoff", "exp_pnl_model", "win_model", "resolved_request",
    "validation_status", "readiness", "reason_codes", "warnings",
})
_FORECAST_OUTPUTS = frozenset({
    "driver_prediction", "forecast_abs_move", "runup_move_prediction",
    "pred_iv_crush", "pred_iv_crush_30", "model_fair_pct",
})
_LINEAR_RECIPE_FIELDS = frozenset({"intercept", "coefficients"})
# A forecast/gate recipe that names a frozen release binding instead of
# carrying inline coefficients (R4-16). ``output`` names the binding's own
# output when it has several; the stage runs it through FrozenInference (the
# binding's registered adapter over hash-verified members), never a refit.
_FROZEN_RECIPE_FIELDS = frozenset({"binding_id", "output"})
# Which release roles may own each native output. A recipe wired to a
# binding of another role is a malformed bundle and is refused at build time.
# The feature roles and the gate are the legacy champions actually served
# (engine/models/registry.json, engine.v2.models.inventory): size =
# BlendModel(OLS, scaled MLP); implied_t1, iv_crush and gate =
# HistGradientBoostingRegressor; runup_move = LogTargetRegressor(HGBR).
_FROZEN_OUTPUT_ROLES = {
    "driver_prediction": frozenset({"driver", "size", "implied_t1"}),
    "forecast_abs_move": frozenset({"size"}),
    "runup_move_prediction": frozenset({"runup_move"}),
    "pred_iv_crush": frozenset({"iv_crush"}),
    "pred_iv_crush_30": frozenset({"iv_crush"}),
    "model_fair_pct": frozenset({"fair_value"}),
    "gate_score": frozenset({"gate"}),
    # The forecast-analog gate's own ``pred_abs_move`` (R4-20 gap 3): the
    # Tier-4 size fold legacy ``Scorer._forecast_for_gate`` serves.
    "pred_abs_move": frozenset({"size"}),
    # The DYN-SV chooser champion (R4-20 gap 5), ``dyn_sv_chooser_v1_1``.
    "chooser_score": frozenset({"chooser"}),
    # The chooser's other Tier-4 producer columns (legacy
    # ``Scorer._chooser_frame``): the implied_t1 and runup_move serving folds.
    "pred_im_t1_d14": frozenset({"implied_t1"}),
    "pred_runup_abs_move_d14": frozenset({"runup_move"}),
}
_RESIDUAL_RECIPE_FIELDS = frozenset({
    "mode", "terminal_spots", "weights", "capital_at_risk", "pnl_cutoff",
    "population_ref", "recipe_id", "draw_count", "seed",
})
# The legacy planned-exit simulation (engine/pnl_sim.expected_pnl as called by
# engine/score.py Scorer._expectation), declared through residual_recipe with
# mode="planned_exit" (R4-10). Every field is a raw fact or a recipe
# constant: pre_iv30 is the pre-print vol level, dte_exit the days left at the
# planned exit, event_date the causal cutoff of the paired pool, draws the
# simulation size (pnl_sim.DRAWS by default). Each is optional: omitted, the
# stage reads pre_iv30/event_date from the context and derives dte_exit from
# expiry - exit_date, as the legacy caller does.
_PLANNED_EXIT_RECIPE_FIELDS = frozenset({
    "mode", "pre_iv30", "dte_exit", "event_date", "draws", "pnl_cutoff",
    "population_ref", "recipe_id",
})
# One compatibility-path paired-pool row: a PRIOR event's size forecast and
# its paired (move, crush) errors (legacy ResidualPool's columns).
_PAIRED_ROW_FIELDS = frozenset({
    "event_date", "ticker", "pred_abs_move", "err_move", "err_crush",
})
_ANALOG_RECIPE_FIELDS = frozenset({
    "recipe_id", "population_ref", "distance_metric", "neighbors",
})
# Fields describing the legacy bucket-analog recipe (the PRODUCTION analog
# construction; see native_analog.LegacyBucketRecipe). population_hash is
# deliberately excluded here: it is derived from analog_source_rows by the
# builder, never supplied by the caller, so a bundle cannot assert an
# unverified answer for its own population.
_BUCKET_ANALOG_RECIPE_FIELDS = frozenset({
    "bucket_dimensions", "widening_order", "min_analogs", "alpha",
    "bootstrap_draws", "bootstrap_seed", "ci_quantiles",
})
_GATE_RECIPE_FIELDS = frozenset({
    "model", "threshold", "recipe_id", "artifact_ref", "artifact_hashes",
    # R4-20 gap 3: the size-fold binding the gate's derived forecast columns
    # come from, as {"binding_id"[, "output"]}.
    "forecast",
}) | _FROZEN_RECIPE_FIELDS
# The served size fold's held-out pool, the band's source (legacy
# ``ServingModel.pool_pred``/``pool_res``/``interval_floor``).
_GATE_FORECAST_POOL_FIELDS = frozenset({"predictions", "residuals", "interval_floor"})
# engine/score.py DYNAMIC_MENU: the only strategies legacy ever asks the
# chooser to score (``Scorer.score``: ``if request.strategy in DYNAMIC_MENU``).
_CHOOSER_STRATEGIES = frozenset({
    "TWIN-P", "TWIN-P5", "CND-PS", "BFLY-P", "BFLY-P5", "RAMP7", "CTR5",
})
# Payoff-calibration/model-layer recipe fields. ``before`` is the decision's
# evidence cutoff (a raw fact, matched against each row's own exit_date by
# the native stage -- see stages.py's ``_execute_model``/``native_payoff.
# fit_payoff_line``), never a
# calculated answer. ``min_trades``/``max_residuals``/``residual_seed`` mirror
# engine/payoff.py's MIN_TRADES/MAX_RESIDUALS/RESIDUAL_SEED; ``draw_count``
# mirrors engine/score.py's MODEL_DRAWS; ``seed`` overrides the derived
# seed for tests that need a fixed one (parity checks).
_PAYOFF_RECIPE_FIELDS = frozenset({
    "before", "min_trades", "max_residuals", "residual_seed", "draw_count",
    "seed",
})
# The frozen-artifact path (P5-4). `min_trades`/`max_residuals`/
# `residual_seed` describe HOW to fit and drop out, since the line/surface is
# already fitted -- but `before` stays: it is the request's own causal
# cutoff (what the inline fit WOULD have used), and the model stage checks
# it against the artifact's own `.cutoff` (plus strategy and the request's
# resolved fill alpha) before trusting it -- a wrong-fold artifact is
# MODEL_NOT_READY, never a silently-wrong number (coordinator decision,
# 2026-09-18: the stage verifies the full causal key itself rather than
# trusting release selection alone). Mutually exclusive with
# payoff_recipe/payoff_source_rows (the source-rows compatibility path); see
# `_model_block`.
_PAYOFF_ARTIFACT_RECIPE_FIELDS = frozenset({"before", "draw_count", "seed"})
_MODEL_RESIDUAL_RECIPE_FIELDS = frozenset({"deciles", "min_pool"})
# Frozen residual pools (P5-4). Each declared slot names the causal key the
# request expects; ``content_hash`` optionally pins the exact artifact the
# release holds. The stage refuses (MODEL_NOT_READY) on any disagreement.
_DRIVER_RESIDUAL_SLOTS = frozenset({"driver", "runup_move"})
_DRIVER_RESIDUAL_KEY_FIELDS = frozenset({"role", "model_id", "fold", "content_hash"})
_PAIRED_RESIDUAL_RECIPE_FIELDS = frozenset({
    "move_model_id", "crush_model_id", "cutoff", "content_hash", "draws",
    "pre_iv30", "dte_exit",
})
_SIZE_STRATEGIES = frozenset({
    "TWIN-P", "TWIN-P5", "CND-PS", "BFLY-P", "BFLY-P5", "RAMP7",
    "CTR5",
})
_STRATEGY_FORECAST_OUTPUTS = {
    "STR-THRU": frozenset({"driver_prediction"}),
    "STR-RUNUP": frozenset({
        "driver_prediction", "runup_move_prediction",
    }),
    **{
        strategy: frozenset({"forecast_abs_move"})
        for strategy in _SIZE_STRATEGIES
    },
}
_SUPPORTED_STRATEGIES = frozenset(_STRATEGY_FORECAST_OUTPUTS)


@dataclass(frozen=True, kw_only=True)
class SourceBundle:
    """Source-only inputs for one bounded native strategy execution.

    Recipes describe calculations. They do not carry calculated forecasts,
    selected contracts, prices, simulation summaries, or decisions.
    """

    source_ref: str
    # Raw, source-owned facts about this row: ticker, strategy, the resolved
    # entry/exit/quote dates, spot, strike, ... Never a calculated scoring
    # answer (checked below via ``_ANSWER_FIELDS``). One entry worth calling
    # out explicitly, on the same non-circular footing as ``gate_forecast_pool``
    # below: ``calendar_observed_through`` -- the calendar's own horizon fact
    # (``engine.calendar.TradingCalendar.observed_through``, the last date
    # backed by real observed price history), constant for every row scored
    # against one calendar instance and carried here from
    # ``engine/score.py``'s own ``self.calendar.observed_through`` at capture
    # time. It is never legacy's PROJECTED_CALENDAR verdict -- that is a
    # per-row calculated answer and would be circular -- only the calendar
    # fact ``_check_projected_calendar`` (engine/v2/scoring/stages.py) needs
    # to reproduce ``calendar.is_projected(exit_date)`` independently.
    context: Mapping[str, Any]
    raw_quotes: Mapping[Any, Mapping[str, Any]]
    feature_vector: Mapping[str, Any]
    feature_missing_mask: Mapping[str, bool]
    model_identity: Mapping[str, Any]
    forecast_recipes: Mapping[str, Mapping[str, Any]]
    model_artifact_refs: Mapping[str, str]
    residual_recipe: Mapping[str, Any]
    analog_recipe: Mapping[str, Any]
    gate_recipe: Mapping[str, Any]
    driver_name: str = "abs_move"
    strategy: str = "STR-THRU"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Causal bucket-analog population, source-only: each row is a PRIOR
    # event's bucket membership plus its realized outcome. Empty by default,
    # meaning no analog population was supplied — the analog stage is then
    # genuinely not applicable, not silently skipped despite a request.
    analog_source_rows: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    # This request's own bucket membership (no outcome, no legacy answer).
    analog_query: Mapping[str, Any] = field(default_factory=dict)
    # Frozen board analog matcher (P5-4, engine.v2.models.analog_artifact):
    # a verified, already-loaded BoardAnalogPoolArtifact -- the causal
    # (strategy, alpha, cutoff) slice of the FULL-universe analog population.
    # Declared by a non-empty recipe (native_analog.FROZEN_ANALOG_RECIPE_FIELDS:
    # the request's evidence ``cutoff``, an optional ``content_hash`` release
    # pin, and legacy match()'s own arguments) or a supplied artifact. The
    # analog stage then reads the artifact after a full causal-key check and
    # never rebuilds a pool; a missing or mismatched artifact is
    # MODEL_NOT_READY. ``analog_query`` then carries mcap_bucket, dte_band,
    # moneyness_band and the RAW implied_ratio (the stage buckets it on the
    # artifact's frozen edges). Mutually exclusive with analog_recipe/
    # analog_source_rows, the declared-rows compatibility path.
    analog_artifact_recipe: Mapping[str, Any] = field(default_factory=dict)
    analog_artifact: "BoardAnalogPoolArtifact | None" = None
    # Payoff-calibration/model layer (exp_pnl_model, win_model), mirroring
    # the analog fields above exactly: a recipe describes the fit, the rows
    # are real PRIOR trades' own outcomes (driver, spot_entry, exit_value,
    # exit_date), never this row's own answer. Empty by default -- the model
    # stage stays not-applicable until a caller opts in.
    payoff_recipe: Mapping[str, Any] = field(default_factory=dict)
    payoff_source_rows: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    # Frozen payoff-calibration artifact (P5-4): a verified, already-loaded
    # PayoffLineArtifact/PayoffSurfaceArtifact (engine.v2.models.
    # payoff_artifact). When declared (a non-empty payoff_artifact_recipe, or
    # payoff_artifact itself set), the model stage reads its coefficients and
    # residuals directly and never fits -- mutually exclusive with
    # payoff_recipe/payoff_source_rows, the source-rows compatibility path
    # kept for bundles built before an artifact existed (Phase 4 captures).
    # A declared-but-unresolved request (payoff_artifact left None, or one of
    # the wrong kind/strategy for this bundle) is MODEL_NOT_READY at
    # execution -- this builder does not fall back to fitting.
    payoff_artifact_recipe: Mapping[str, Any] = field(default_factory=dict)
    payoff_artifact: "PayoffLineArtifact | PayoffSurfaceArtifact | None" = None
    # The champion driver model's own held-out (prediction, residual) pairs
    # -- a fixed, artifact-owned population (see native_payoff.py), not this
    # row's own answer either.
    model_residual_recipe: Mapping[str, Any] = field(default_factory=dict)
    model_residual_rows: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    # STR-RUNUP's second driver (legacy's ``runup_move`` champion): its own
    # held-out (prediction, residual) pairs, at the model's native D14 scale
    # -- distinct from ``model_residual_rows``, which for STR-RUNUP carries
    # the FIRST driver's (``implied_t1``) pool. Ignored by every strategy
    # with only one driver.
    runup_move_residual_rows: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    # Frozen win-rate recalibration map (P5-4, legacy Scorer.recalibration):
    # a verified, already-loaded RecalibrationMapArtifact. Declared by
    # ``recalibration_declared=True`` or a non-None artifact; the model stage
    # then checks its full causal key (strategy, fill alpha, the payoff
    # recipe's ``before``) and applies it to win_model -- or refuses with
    # MODEL_NOT_READY when it is missing or mismatched. Undeclared (the
    # default, and every Phase 4 capture) leaves win_model exactly as before.
    recalibration_declared: bool = False
    recalibration_artifact: "RecalibrationMapArtifact | None" = None
    # Frozen driver residual pools (P5-4, engine.v2.models.residual_artifact),
    # keyed by slot ("driver"; STR-RUNUP also "runup_move"). Declared by a
    # non-empty recipe (slot -> expected {role, model_id, fold[, content_hash]})
    # or non-empty artifacts; mutually exclusive with model_residual_rows/
    # runup_move_residual_rows/model_residual_recipe, which stay the
    # compatibility path for bundles that declare neither (Phase 4 captures).
    model_residual_artifact_recipe: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    model_residual_artifacts: Mapping[str, "DriverResidualPoolArtifact | None"] = field(
        default_factory=dict)
    # Frozen paired (err_move, err_crush) pool for the planned-exit simulation
    # (P5-4). Declared by a non-empty recipe or a supplied artifact; the
    # simulation then runs planned-exit from the artifact alone, after a full
    # causal-key check -- never from request rows or a context-scoped rebuild.
    paired_residual_recipe: Mapping[str, Any] = field(default_factory=dict)
    paired_residual_artifact: "PairedResidualPoolArtifact | None" = None
    # The planned-exit simulation's COMPATIBILITY path (R4-10): the paired
    # pool as source rows (event_date, pred_abs_move, err_move, err_crush,
    # optional ticker) -- prior events' residuals, e.g. a Phase 4 capture's
    # recorded residual_population, in that recorded order. Mutually
    # exclusive with the frozen paired artifact above.
    paired_residual_rows: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    # Frozen inference for recipes that name a release binding (R4-16): the
    # verified loader and the release holding the bindings. A binding recipe
    # without them is carried as declared-but-unresolved and refuses
    # MODEL_NOT_READY at execution; it never falls back to another model.
    frozen_inference: "FrozenInference | None" = None
    model_release: "ModelRelease | None" = None
    # R4-20 gap 3: the served size fold's held-out pool behind the gate's
    # ``pred_abs_move_p10``/``_p90``/``_sd`` -- ``{"predictions",
    # "residuals"[, "interval_floor"]}``, the fold's own ``pool_pred``/
    # ``pool_res`` in stored order (a model-owned population, like
    # ``model_residual_rows``; never this row's answer). Empty: undeclared.
    gate_forecast_pool: Mapping[str, Any] = field(default_factory=dict)
    # R4-20 gap 5: the DYN-SV chooser champion as a frozen release binding,
    # ``{"binding_id"[, "output"]}``, for a DYNAMIC_MENU candidate. Empty:
    # no chooser ranking is requested (legacy with no chooser champion).
    chooser_recipe: Mapping[str, Any] = field(default_factory=dict)
    # The chooser's derived columns (R4-20 remaining gap (a); see
    # ``native_chooser``). ``chooser_fold_pools``: the served Tier-4 folds'
    # held-out pools by output (``pred_abs_move``, ``pred_im_t1_d14``,
    # ``pred_runup_abs_move_d14``), each shaped like ``gate_forecast_pool``.
    # ``chooser_analog_pool``/``chooser_admissible_table``: the frozen k-NN
    # population and n_admissible table, checked against the keys
    # ``chooser_recipe["analog_pool"]``/``["admissible_table"]`` declare.
    chooser_fold_pools: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    chooser_analog_pool: "ChooserAnalogPoolArtifact | None" = None
    chooser_admissible_table: "AdmissibleDepthTable | None" = None
    # R4-20 (c): a forecast legacy read from the STORED Tier-4 table instead
    # of serving a fold (``Scorer._crush_forecast`` prefers the stored
    # ``pred_iv_crush_30`` whenever the table holds a non-NaN value for the
    # event). This carries a REFERENCE to that cell, never the cell:
    # ``{"pred_iv_crush_30": {"row": {"table", "table_sha256", "column",
    # "ticker", "event_date", "model_id", "fold_start"}, "row_hash"}}``.
    # ``row_hash`` is the content hash of ``{"row", "value"}`` -- a one-way
    # check that the value a resolver reads back is the one legacy read, and
    # not a way to recover it. Native must look the cell up for itself; see
    # ``checks/phase4_stored_forecasts.py``. A resolved stored value wins
    # over any recipe for the same output, as in legacy.
    stored_forecast_refs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def _answer_paths(value: Any, path: str) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_name = str(key)
            child_path = f"{path}.{key_name}"
            if key_name in _ANSWER_FIELDS:
                found.append(child_path)
            found.extend(_answer_paths(child, child_path))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            found.extend(_answer_paths(child, f"{path}[{index}]"))
    return found


def _reject_answers(name: str, values: Any) -> None:
    forbidden = sorted(_answer_paths(values, name))
    if forbidden:
        raise ValueError(f"{name} contains calculated answer fields: {forbidden}")


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _linear_recipe(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    extra = sorted(set(value) - _LINEAR_RECIPE_FIELDS)
    if extra:
        raise ValueError(f"{name} has unsupported recipe fields: {extra}")
    coefficients = value.get("coefficients", {})
    if not isinstance(coefficients, Mapping):
        raise ValueError(f"{name}.coefficients must be a mapping")
    return {
        "intercept": _finite_number(value.get("intercept", 0.0), f"{name}.intercept"),
        "coefficients": {
            str(feature): _finite_number(coefficient, f"{name}.{feature}")
            for feature, coefficient in coefficients.items()
        },
    }


#: Outputs legacy may read from the stored Tier-4 table (``_crush_forecast``).
_STORED_FORECAST_OUTPUTS = frozenset({"pred_iv_crush_30"})
#: A stored-forecast REFERENCE, and the whole of one. There is deliberately
#: no ``value`` member: a captured value would be an output of the system
#: under test supplied back to it as an input, which is the circularity
#: ``tools/phase4_release_assembler.py::_reject_answers`` exists to stop.
#: ``_bounded_recipe`` refuses any other member, so ``value`` cannot be
#: smuggled in beside the reference either (tests/test_phase4_stored_forecast_refs.py).
STORED_REF_FIELDS = frozenset({"row", "row_hash"})
#: The row identity: which table, which vintage of it, which column, which
#: key, and which producer model wrote the cell. Every one of these is an
#: ADDRESS or a provenance label -- none of them is, or bounds, the number
#: stored at that address.
STORED_ROW_FIELDS = frozenset({
    "table", "table_sha256", "column", "ticker", "event_date",
    "model_id", "fold_start",
})
_STORED_ROW_REQUIRED_TEXT = ("table", "table_sha256", "column", "ticker", "event_date")


def stored_forecast_row_hash(row: Mapping[str, Any], value: float) -> str:
    """The identity a stored forecast declaration carries: its source row
    and value, content-hashed together.

    This is a VERIFICATION aid and cannot stand in for the lookup: it is a
    one-way content hash, so a resolver has to produce the value by reading
    the table before it can check anything against this.
    """
    from engine.v2.foundation import content_hash

    return content_hash({"row": dict(row), "value": float(value)})


def _stored_forecast_row(name: str, output: str, raw: Any) -> dict[str, Any]:
    """The address half of one reference: exactly its identity fields."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name}.row must be a mapping")
    row = _bounded_recipe(f"{name}.row", raw, STORED_ROW_FIELDS)
    if set(row) != STORED_ROW_FIELDS:
        raise ValueError(f"{name}.row must name {sorted(STORED_ROW_FIELDS)}")
    for key in _STORED_ROW_REQUIRED_TEXT:
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"{name}.row.{key} must be a non-empty string")
    if row["column"] != output:
        raise ValueError(f"{name}.row.column must name {output!r}")
    for key in ("model_id", "fold_start"):
        if row[key] is not None and not isinstance(row[key], str):
            raise ValueError(f"{name}.row.{key} must be a string or null")
    return {key: row[key] for key in sorted(row)}


def _stored_forecast_ref(name: str, output: str, declared: Any) -> dict[str, Any]:
    """One reference: an address plus its one-way row hash, and nothing else.

    The "nothing else" is the point -- see :data:`STORED_REF_FIELDS`.
    """
    if not isinstance(declared, Mapping):
        raise ValueError(f"{name} must be a mapping")
    config = _bounded_recipe(name, declared, STORED_REF_FIELDS)
    if set(config) != STORED_REF_FIELDS:
        raise ValueError(f"{name} must name {sorted(STORED_REF_FIELDS)}")
    row_hash = config["row_hash"]
    if not isinstance(row_hash, str) or not row_hash.strip():
        raise ValueError(f"{name}.row_hash must be a non-empty string")
    return {
        "row": _stored_forecast_row(name, output, config.get("row")),
        "row_hash": row_hash,
    }


def _stored_forecast_refs(bundle: SourceBundle) -> dict[str, dict[str, Any]]:
    """Validate every declared stored-forecast reference.

    Refuses anything that is not exactly an address plus its one-way row
    hash, so this path cannot become the hole in the answer rule.
    """
    unknown = sorted(set(bundle.stored_forecast_refs) - _STORED_FORECAST_OUTPUTS)
    if unknown:
        raise ValueError(f"unsupported stored forecast outputs: {unknown}")
    return {
        str(output): _stored_forecast_ref(
            f"stored_forecast_refs.{output}", str(output), declared)
        for output, declared in bundle.stored_forecast_refs.items()
    }


def _forecast_block(bundle: SourceBundle, strategy: str) -> dict[str, Any]:
    unknown = sorted(set(bundle.forecast_recipes) - _FORECAST_OUTPUTS)
    if unknown:
        raise ValueError(f"unsupported forecast recipe outputs: {unknown}")
    missing = sorted(
        _STRATEGY_FORECAST_OUTPUTS[strategy] - set(bundle.forecast_recipes)
    )
    if missing:
        raise ValueError(
            f"{strategy} requires forecast recipes for: {missing}"
        )
    missing_refs = sorted(
        set(bundle.forecast_recipes) - set(bundle.model_artifact_refs)
    )
    if missing_refs:
        raise ValueError(f"forecast recipes lack artifact refs: {missing_refs}")
    refs = {
        str(role): str(reference)
        for role, reference in bundle.model_artifact_refs.items()
    }
    if any(not reference.strip() for reference in refs.values()):
        raise ValueError("model artifact refs must be non-empty")
    models: dict[str, Any] = {}
    executors: dict[str, FrozenRecipeExecutor] = {}
    for output, recipe in bundle.forecast_recipes.items():
        name = str(output)
        if _is_frozen_recipe(recipe):
            executors[name] = _frozen_recipe_executor(bundle, name, recipe)
        else:
            models[name] = _linear_recipe(name, recipe)
    block: dict[str, Any] = {
        "driver_name": str(bundle.driver_name),
        "models": models,
        "artifact_hashes": tuple(refs[output] for output in bundle.forecast_recipes),
        "model_artifact_refs": refs,
    }
    if executors:
        block["executors"] = executors
    refs = _stored_forecast_refs(bundle)
    if refs:
        block["stored_refs"] = refs
    return block


def _is_frozen_recipe(recipe: Any) -> bool:
    return isinstance(recipe, Mapping) and "binding_id" in recipe


def _frozen_recipe_executor(
    bundle: SourceBundle, target: str, recipe: Mapping[str, Any],
) -> FrozenRecipeExecutor:
    """Resolve one binding-named recipe (R4-16) into a frozen executor.

    The binding, its role and its output are checked here when a release is
    supplied, because a recipe wired to the wrong model is a malformed bundle.
    A missing release or inference is not malformed: the request is declared
    but unresolved, and the stage refuses it MODEL_NOT_READY.
    """
    config = _bounded_recipe(f"{target} recipe", recipe, _FROZEN_RECIPE_FIELDS)
    binding_id = str(config.get("binding_id") or "").strip()
    if not binding_id:
        raise ValueError(f"{target} recipe binding_id must be non-empty")
    inference, release = bundle.frozen_inference, bundle.model_release
    if inference is not None and not isinstance(inference, FrozenInference):
        raise ValueError("frozen_inference must be a FrozenInference")
    if release is not None and not isinstance(release, ModelRelease):
        raise ValueError("model_release must be a ModelRelease")
    if inference is None or release is None:
        return FrozenRecipeExecutor(target=target, binding_id=binding_id,
                                    source=None, executor=None, binding=None)
    matches = [item for item in release.bindings if item.binding_id == binding_id]
    binding = matches[0] if len(matches) == 1 else None
    if binding is None:
        # Not resolvable in this release: FrozenStageExecutor refuses it
        # (BINDING_NOT_FOUND/DUPLICATE_BINDING) at execution.
        return FrozenRecipeExecutor(
            target=target, binding_id=binding_id, source=str(config.get("output") or target),
            executor=FrozenStageExecutor(inference=inference, release=release,
                                         binding_id=binding_id),
            binding=None,
        )
    return FrozenRecipeExecutor(
        target=target, binding_id=binding_id,
        source=_binding_output(binding, target, config.get("output")),
        executor=FrozenStageExecutor(inference=inference, release=release,
                                     binding_id=binding_id),
        binding=binding,
    )


def _binding_output(binding: ModelBinding, target: str, declared: Any) -> str:
    role = str(binding.role).split(":", 1)[0]
    if role not in _FROZEN_OUTPUT_ROLES[target]:
        raise ValueError(
            f"{target} recipe names binding {binding.binding_id} of role {role!r}"
        )
    names = tuple(binding.output_names)
    if declared is not None:
        if str(declared) not in names:
            raise ValueError(f"{target} recipe output {declared!r} is not in {names}")
        return str(declared)
    if target in names:
        return target
    if len(names) == 1:
        return names[0]
    raise ValueError(f"{target} recipe must name one of the binding outputs {names}")


def _quote_block(raw: Mapping[Any, Mapping[str, Any]]) -> dict[Any, dict[str, float]]:
    quotes: dict[Any, dict[str, float]] = {}
    for contract, quote in raw.items():
        if not isinstance(quote, Mapping):
            raise ValueError("each raw quote must be a mapping")
        extra = sorted(set(quote) - {"bid", "ask"})
        if extra:
            raise ValueError(f"raw quote contains unsupported fields: {extra}")
        if "bid" not in quote or "ask" not in quote:
            raise ValueError("raw quotes require bid and ask")
        quotes[contract] = {
            "bid": _finite_number(quote["bid"], "quote.bid"),
            "ask": _finite_number(quote["ask"], "quote.ask"),
        }
    if not quotes:
        raise ValueError("raw quote map must not be empty")
    return quotes


def _bounded_recipe(
    name: str,
    values: Mapping[str, Any],
    allowed: frozenset[str],
) -> dict[str, Any]:
    extra = sorted(set(values) - allowed)
    if extra:
        raise ValueError(f"{name} has unsupported recipe fields: {extra}")
    return dict(values)


def _gate_forecast_members(bundle: SourceBundle, gate: dict[str, Any]) -> dict[str, Any]:
    """The gate's derived-forecast inputs (R4-20 gap 3), popped off ``gate``.

    ``forecast`` names the size-fold binding; ``gate_forecast_pool`` is that
    fold's pool. Both are optional -- a gate naming no forecast column needs
    neither, and one that does but lacks them is refused at execution.
    """
    members: dict[str, Any] = {}
    forecast = gate.pop("forecast", None)
    if forecast is not None:
        if not _is_frozen_recipe(forecast):
            raise ValueError("gate_recipe.forecast must name a frozen binding")
        executor = _frozen_recipe_executor(bundle, "pred_abs_move", forecast)
        members["forecast_executor"] = executor
        members["forecast_recipe"] = str(executor)
    pool = fold_pool("gate_forecast_pool", bundle.gate_forecast_pool)
    if pool:
        members["forecast_pool"] = pool
    return members


def fold_pool(name: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """A served Tier-4 fold's held-out pool (``pool_pred``/``pool_res``/
    ``interval_floor``) as declared source rows; ``{}`` when undeclared."""
    pool = _bounded_recipe(name, values, _GATE_FORECAST_POOL_FIELDS)
    if not pool:
        return {}
    predictions = tuple(float(value) for value in pool.get("predictions", ()))
    residuals = tuple(float(value) for value in pool.get("residuals", ()))
    if len(predictions) != len(residuals):
        raise ValueError(f"{name} predictions and residuals differ in length")
    floor = pool.get("interval_floor", 0.0)
    return {
        "predictions": predictions, "residuals": residuals,
        "interval_floor": None if floor is None else _finite_number(
            floor, f"{name}.interval_floor"),
    }


def _gate_block(bundle: SourceBundle) -> dict[str, Any]:
    gate = _bounded_recipe("gate_recipe", bundle.gate_recipe, _GATE_RECIPE_FIELDS)
    gate.update(_gate_forecast_members(bundle, gate))
    if _is_frozen_recipe(gate):
        if gate.get("model") is not None:
            raise ValueError("gate_recipe cannot name both a binding and a linear model")
        frozen = {name: gate.pop(name) for name in _FROZEN_RECIPE_FIELDS if name in gate}
        gate["executors"] = {
            "gate_score": _frozen_recipe_executor(bundle, "gate_score", frozen),
        }
        gate["binding_id"] = frozen["binding_id"]
        gate["threshold"] = _finite_number(
            gate.get("threshold"), "gate_recipe.threshold",
        )
        return gate
    model = gate.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("gate_recipe requires a linear model")
    gate["model"] = _linear_recipe("gate_recipe.model", model)
    gate["threshold"] = _finite_number(
        gate.get("threshold"), "gate_recipe.threshold",
    )
    return gate


def _chooser_block(bundle: SourceBundle, strategy: str) -> dict[str, Any]:
    """The DYN-SV chooser champion and its derived-column inputs
    (``chooser_inputs.chooser_block``)."""
    from engine.v2.scoring.chooser_inputs import chooser_block

    return chooser_block(bundle, strategy)


def _analog_block(bundle: SourceBundle) -> dict[str, Any]:
    """Express the legacy bucket-analog recipe, with inputs, when sourced.

    The recipe-vs-rows distinction is the one that matters, not rows alone:
    an empty ``analog_recipe`` means nothing was requested, so the block is
    genuinely not-applicable (``recipe: None``). A non-empty ``analog_recipe``
    is a positive request; if its source population (``analog_source_rows``)
    is absent, the declared (possibly incomplete) recipe is still carried so
    the execution stage reports MISSING_ANALOG_INPUT itself -- this builder
    must not reclassify a declared-but-unfed recipe as not-applicable, and
    must not fabricate rows to satisfy it.
    """
    frozen = _bounded_recipe(
        "analog_artifact_recipe", bundle.analog_artifact_recipe,
        FROZEN_ANALOG_RECIPE_FIELDS,
    )
    if frozen or bundle.analog_artifact is not None:
        return _frozen_analog_block(bundle, frozen)
    config = _bounded_recipe(
        "analog_recipe", bundle.analog_recipe,
        _ANALOG_RECIPE_FIELDS | _BUCKET_ANALOG_RECIPE_FIELDS,
    )
    if not config:
        return {"recipe": None}
    if not bundle.analog_source_rows:
        return {"recipe": config}
    missing = sorted(_BUCKET_ANALOG_RECIPE_FIELDS - set(config))
    if missing:
        raise ValueError(f"analog_recipe requires bucket fields: {missing}")
    _reject_answers("analog_source_rows", bundle.analog_source_rows)
    _reject_answers("analog_query", bundle.analog_query)
    dimensions = tuple(str(name) for name in config["bucket_dimensions"])
    population_hash = bucket_population_hash(bundle.analog_source_rows, dimensions)
    recipe = {
        "schema_version": BUCKET_RECIPE_SCHEMA,
        "bucket_dimensions": dimensions,
        "widening_order": tuple(str(name) for name in config["widening_order"]),
        "min_analogs": config["min_analogs"],
        "alpha": config["alpha"],
        "bootstrap_draws": config["bootstrap_draws"],
        "bootstrap_seed": config["bootstrap_seed"],
        "ci_quantiles": tuple(config["ci_quantiles"]),
        "population_hash": population_hash,
    }
    return {
        "recipe": recipe,
        "source_rows": [dict(row) for row in bundle.analog_source_rows],
        "query_features": dict(bundle.analog_query),
    }


def _frozen_analog_block(bundle: SourceBundle, recipe: dict[str, Any]) -> dict[str, Any]:
    """The P5-4 frozen board analog matcher branch of ``_analog_block``.

    The artifact is carried as given -- ``None`` included, so the stage
    refuses MODEL_NOT_READY rather than this builder dropping the request or
    falling back to declared rows.
    """
    if bundle.analog_recipe or bundle.analog_source_rows:
        raise ValueError("analog_source_rows/analog_recipe (the compatibility path) cannot "
                         "combine with analog_artifact_recipe/analog_artifact")
    artifact = bundle.analog_artifact
    if artifact is not None and not isinstance(artifact, BoardAnalogPoolArtifact):
        raise ValueError("analog_artifact must be a BoardAnalogPoolArtifact")
    _reject_answers("analog_artifact_recipe", recipe)
    _reject_answers("analog_query", bundle.analog_query)
    return {
        "analog_artifact": artifact,
        "analog_artifact_recipe": recipe,
        "query_features": dict(bundle.analog_query),
    }


def _artifact_model_block(
    bundle: SourceBundle, artifact_recipe: Mapping[str, Any], recipe: Mapping[str, Any],
) -> dict[str, Any]:
    """The P5-4 frozen-artifact branch of ``_model_block`` -- a positive
    request for the artifact path (mutually exclusive with the source-rows
    compatibility recipe/rows).
    """
    if bundle.payoff_artifact is not None and not isinstance(
        bundle.payoff_artifact, (PayoffLineArtifact, PayoffSurfaceArtifact),
    ):
        raise ValueError(
            "payoff_artifact must be a PayoffLineArtifact or PayoffSurfaceArtifact"
        )
    if recipe or bundle.payoff_source_rows:
        raise ValueError(
            "payoff_artifact_recipe/payoff_artifact cannot combine with "
            "payoff_recipe/payoff_source_rows (the source-rows compatibility path)"
        )
    return {
        "payoff_recipe": dict(artifact_recipe),
        "payoff_artifact": bundle.payoff_artifact,
        **_residual_members(bundle),
    }


def _compatibility_model_block(bundle: SourceBundle, recipe: Mapping[str, Any]) -> dict[str, Any]:
    """The source-rows compatibility branch of ``_model_block``, unchanged
    from before the P5-4 artifact path existed (Phase 4 captures)."""
    if not recipe:
        if (bundle.payoff_source_rows or bundle.model_residual_rows
                or bundle.model_residual_recipe
                or bundle.runup_move_residual_rows
                or _driver_artifacts_declared(bundle)):
            raise ValueError(
                "payoff_source_rows/model_residual_*/runup_move_residual_rows "
                "supplied without a payoff_recipe"
            )
        return {}
    _reject_answers("payoff_source_rows", bundle.payoff_source_rows)
    return {
        "payoff_recipe": recipe,
        "payoff_source_rows": [dict(row) for row in bundle.payoff_source_rows],
        **_residual_members(bundle),
    }


def _driver_artifacts_declared(bundle: SourceBundle) -> bool:
    return bool(bundle.model_residual_artifact_recipe or bundle.model_residual_artifacts)


def _driver_artifact_members(bundle: SourceBundle) -> dict[str, Any]:
    """The P5-4 frozen driver-pool members of the model block."""
    if (bundle.model_residual_rows or bundle.runup_move_residual_rows
            or bundle.model_residual_recipe):
        raise ValueError(
            "model_residual_artifact_recipe/model_residual_artifacts cannot combine "
            "with model_residual_rows/runup_move_residual_rows/model_residual_recipe"
        )
    unknown = sorted(
        (set(bundle.model_residual_artifact_recipe) | set(bundle.model_residual_artifacts))
        - _DRIVER_RESIDUAL_SLOTS
    )
    if unknown:
        raise ValueError(f"unknown model residual artifact slots: {unknown}")
    for slot, artifact in bundle.model_residual_artifacts.items():
        if artifact is not None and not isinstance(artifact, DriverResidualPoolArtifact):
            raise ValueError(f"model_residual_artifacts[{slot}] must be a DriverResidualPoolArtifact")
    keys = {
        str(slot): _bounded_recipe(
            f"model_residual_artifact_recipe.{slot}", expected, _DRIVER_RESIDUAL_KEY_FIELDS,
        )
        for slot, expected in bundle.model_residual_artifact_recipe.items()
    }
    return {
        "model_residual_artifact_recipe": keys,
        "model_residual_artifacts": dict(bundle.model_residual_artifacts),
    }


def _residual_members(bundle: SourceBundle) -> dict[str, Any]:
    """The driver residual members of either model block: frozen artifacts
    when declared (P5-4), else the request-supplied rows exactly as before."""
    if _driver_artifacts_declared(bundle):
        return _driver_artifact_members(bundle)
    _reject_answers("model_residual_rows", bundle.model_residual_rows)
    _reject_answers("runup_move_residual_rows", bundle.runup_move_residual_rows)
    residual_recipe = _bounded_recipe(
        "model_residual_recipe", bundle.model_residual_recipe,
        _MODEL_RESIDUAL_RECIPE_FIELDS,
    )
    return {
        "model_residual_recipe": residual_recipe,
        "model_residual_rows": [dict(row) for row in bundle.model_residual_rows],
        "runup_move_residual_rows": [
            dict(row) for row in bundle.runup_move_residual_rows
        ],
    }


def _simulation_block(bundle: SourceBundle) -> dict[str, Any]:
    """The simulation recipe; with a declared frozen paired pool (P5-4), a
    planned-exit simulation over that artifact alone.

    Undeclared (empty ``paired_residual_recipe`` and no artifact): the
    bounded ``residual_recipe`` exactly as before. Declared: the recipe's
    key fields become the causal key the stage checks the artifact against;
    a terminal-spot recipe cannot be combined with it (two simulations for
    one row), and a missing artifact is carried as ``None`` so the stage
    refuses MODEL_NOT_READY rather than this builder dropping the request.
    """
    planned = bundle.residual_recipe.get("mode") == "planned_exit"
    simulation = _bounded_recipe(
        "residual_recipe", bundle.residual_recipe,
        _PLANNED_EXIT_RECIPE_FIELDS if planned else _RESIDUAL_RECIPE_FIELDS,
    )
    recipe = _bounded_recipe(
        "paired_residual_recipe", bundle.paired_residual_recipe,
        _PAIRED_RESIDUAL_RECIPE_FIELDS,
    )
    artifact = bundle.paired_residual_artifact
    rows = bundle.paired_residual_rows
    artifact_declared = bool(recipe) or artifact is not None
    if not artifact_declared and not rows and not planned:
        return simulation
    if simulation.get("terminal_spots") is not None or simulation.get("mode") not in (
        None, "planned_exit",
    ):
        raise ValueError("paired_residual_* declares a planned-exit simulation; "
                         "residual_recipe cannot also declare a terminal one")
    if artifact_declared and rows:
        raise ValueError("paired_residual_rows (the compatibility path) cannot "
                         "combine with paired_residual_recipe/paired_residual_artifact")
    if not artifact_declared:
        return {**simulation, "mode": "planned_exit", **_paired_rows_member(rows)}
    if artifact is not None and not isinstance(artifact, PairedResidualPoolArtifact):
        raise ValueError("paired_residual_artifact must be a PairedResidualPoolArtifact")
    key = {name: recipe[name] for name in ("move_model_id", "crush_model_id", "cutoff",
                                           "content_hash") if name in recipe}
    extras = {name: recipe[name] for name in ("draws", "pre_iv30", "dte_exit") if name in recipe}
    return {
        **simulation, **extras, "mode": "planned_exit",
        "paired_residual_key": key, "paired_residual_artifact": artifact,
    }


def _paired_rows_member(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The compatibility-path pool, as declared rows in their given order.

    Empty rows under a planned-exit recipe are a declared-but-unfed request:
    no ``residuals`` member is emitted, so the stage refuses
    MISSING_SIMULATION_INPUT:residuals itself rather than this builder
    inventing a pool or dropping the request.
    """
    if not rows:
        return {}
    _reject_answers("paired_residual_rows", rows)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("each paired residual row must be a mapping")
        extra = sorted(set(row) - _PAIRED_ROW_FIELDS)
        if extra:
            raise ValueError(f"paired_residual_rows[{index}] has unsupported fields: {extra}")
    return {"residuals": [dict(row) for row in rows]}


def _model_block(bundle: SourceBundle) -> dict[str, Any]:
    """Express the payoff-calibration/model-layer recipe, mirroring ``_analog_block``.

    An empty ``payoff_recipe`` AND an undeclared ``payoff_artifact_recipe``
    together mean nothing was requested: the model stage stays not-applicable
    (``{}``), matching every bundle built before either field existed. A
    non-empty ``payoff_artifact_recipe`` (or a supplied ``payoff_artifact``)
    is a positive request for the P5-4 frozen-artifact path
    (``_artifact_model_block``) -- mutually exclusive with the source-rows
    COMPATIBILITY PATH (``_compatibility_model_block``) that fits inline at
    execution time and stays in place, unchanged, for bundles that declare
    neither artifact field (Phase 4 captures). Either declared recipe
    carries through even when it cannot yet be satisfied, so the native
    model stage reports its own refusal (NO_PAYOFF_MAP or MODEL_NOT_READY)
    rather than this builder silently downgrading a real request to
    not-applicable.
    """
    artifact_recipe = _bounded_recipe(
        "payoff_artifact_recipe", bundle.payoff_artifact_recipe,
        _PAYOFF_ARTIFACT_RECIPE_FIELDS,
    )
    recipe = _bounded_recipe(
        "payoff_recipe", bundle.payoff_recipe, _PAYOFF_RECIPE_FIELDS,
    )
    if bool(artifact_recipe) or bundle.payoff_artifact is not None:
        block = _artifact_model_block(bundle, artifact_recipe, recipe)
    else:
        block = _compatibility_model_block(bundle, recipe)
    return _with_recalibration_block(bundle, block)


def _with_recalibration_block(bundle: SourceBundle, block: dict[str, Any]) -> dict[str, Any]:
    """Add the P5-4 recalibration declaration to a model block.

    The key ``recalibration_artifact`` is present (even as ``None``) only
    when declared, exactly like ``payoff_artifact``; an undeclared bundle's
    block is returned unchanged.
    """
    declared = bool(bundle.recalibration_declared) or bundle.recalibration_artifact is not None
    if not declared:
        return block
    artifact = bundle.recalibration_artifact
    if artifact is not None and not isinstance(artifact, RecalibrationMapArtifact):
        raise ValueError("recalibration_artifact must be a RecalibrationMapArtifact")
    if not block:
        raise ValueError("recalibration_artifact declared without a payoff recipe")
    return {**block, "recalibration_artifact": artifact}


def _declaration_receipts(
    source_ref: str,
    strategy: str,
    context: Mapping[str, Any],
    features: Mapping[str, Any],
    forecast: Mapping[str, Any],
    quotes: Mapping[Any, Any],
    model: Mapping[str, Any],
    analogs: Mapping[str, Any],
    simulation: Mapping[str, Any],
    gate: Mapping[str, Any],
    chooser: Mapping[str, Any],
) -> tuple[StageReceipt, ...]:
    pending = {"execution": "native-runtime"}
    declarations = (
        ("resolve_context", {"source_ref": source_ref}, context),
        ("features", context, features),
        ("forecast", features, forecast),
        ("geometry", {"strategy": strategy, "context": context}, pending),
        ("pricing", {"raw_quotes": quotes}, pending),
        ("model", {"recipe": model}, pending),
        ("analogs", {"recipe": analogs}, pending),
        ("simulation", {"recipe": simulation}, pending),
        ("gate", {"recipe": gate}, pending),
        ("chooser", {"recipe": chooser}, pending),
        ("serialization", {"source_ref": source_ref}, pending),
    )
    return tuple(receipt(stage, inputs, output)
                 for stage, inputs, output in declarations)


def build_native_score_inputs(bundle: SourceBundle) -> NativeScoreInputs:
    """Translate a bounded source bundle into executable native stage inputs."""
    if not isinstance(bundle, SourceBundle):
        raise TypeError("source input builder requires SourceBundle")
    if not bundle.source_ref.strip():
        raise ValueError("source_ref must be non-empty")
    strategy = str(bundle.strategy)
    if strategy not in _SUPPORTED_STRATEGIES:
        if strategy in DISABLED:
            detail = DISABLED[strategy]
        elif strategy in STRATEGIES:
            detail = "UNSUPPORTED_SOURCE_CONTRACT"
        else:
            detail = "UNKNOWN_STRATEGY"
        raise ValueError(
            f"source input builder does not support {strategy}: {detail}"
        )
    for name, values in (
        ("context", bundle.context),
        ("feature_vector", bundle.feature_vector),
        ("feature_missing_mask", bundle.feature_missing_mask),
        ("model_identity", bundle.model_identity),
        ("metadata", bundle.metadata),
    ):
        _reject_answers(name, values)

    quotes = _quote_block(bundle.raw_quotes)
    context = {**dict(bundle.context), "strategy": strategy, "quotes": quotes}
    features = {
        "model_inputs": dict(bundle.feature_vector),
        "missing_mask": dict(bundle.feature_missing_mask),
        "model_identity": dict(bundle.model_identity),
        "source_metadata": dict(bundle.metadata),
    }
    forecast = _forecast_block(bundle, strategy)
    simulation = _simulation_block(bundle)
    model = _model_block(bundle)
    analogs = _analog_block(bundle)
    gate = _gate_block(bundle)
    chooser = _chooser_block(bundle, strategy)
    receipts = _declaration_receipts(
        bundle.source_ref, strategy, context, features, forecast, quotes,
        model, analogs, simulation, gate, chooser,
    )
    return NativeScoreInputs(
        context=context,
        features=features,
        forecast=forecast,
        geometry=None,
        pricing=None,
        model=model,
        analogs=analogs,
        simulation=simulation,
        gate=gate,
        chooser=chooser,
        diagnostics={},
        source_ref=bundle.source_ref,
        stage_receipts=receipts,
    )
