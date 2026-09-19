"""Build bounded native scoring inputs from answer-free source material."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping, Sequence

from engine.v2.domain.generation import DISABLED, STRATEGIES
from engine.v2.models.payoff_artifact import PayoffLineArtifact, PayoffSurfaceArtifact
from engine.v2.models.recalibration_artifact import RecalibrationMapArtifact
from engine.v2.models.residual_artifact import (
    DriverResidualPoolArtifact,
    PairedResidualPoolArtifact,
)
from engine.v2.scoring.native_analog import (
    BUCKET_RECIPE_SCHEMA,
    bucket_population_hash,
)
from engine.v2.scoring.stages import (
    NativeScoreInputs,
    StageReceipt,
    receipt,
)

__all__ = ["SourceBundle", "build_native_score_inputs"]

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
_RESIDUAL_RECIPE_FIELDS = frozenset({
    "mode", "terminal_spots", "weights", "capital_at_risk", "pnl_cutoff",
    "population_ref", "recipe_id", "draw_count", "seed",
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
    return {
        "driver_name": str(bundle.driver_name),
        "models": {
            str(output): _linear_recipe(str(output), recipe)
            for output, recipe in bundle.forecast_recipes.items()
        },
        "artifact_hashes": tuple(refs[output] for output in bundle.forecast_recipes),
        "model_artifact_refs": refs,
    }


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


def _gate_block(values: Mapping[str, Any]) -> dict[str, Any]:
    gate = _bounded_recipe("gate_recipe", values, _GATE_RECIPE_FIELDS)
    model = gate.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("gate_recipe requires a linear model")
    gate["model"] = _linear_recipe("gate_recipe.model", model)
    gate["threshold"] = _finite_number(
        gate.get("threshold"), "gate_recipe.threshold",
    )
    return gate


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
    simulation = _bounded_recipe(
        "residual_recipe", bundle.residual_recipe, _RESIDUAL_RECIPE_FIELDS,
    )
    recipe = _bounded_recipe(
        "paired_residual_recipe", bundle.paired_residual_recipe,
        _PAIRED_RESIDUAL_RECIPE_FIELDS,
    )
    artifact = bundle.paired_residual_artifact
    if not recipe and artifact is None:
        return simulation
    if artifact is not None and not isinstance(artifact, PairedResidualPoolArtifact):
        raise ValueError("paired_residual_artifact must be a PairedResidualPoolArtifact")
    if simulation.get("terminal_spots") is not None or simulation.get("mode") not in (
        None, "planned_exit",
    ):
        raise ValueError("paired_residual_* declares a planned-exit simulation; "
                         "residual_recipe cannot also declare a terminal one")
    key = {name: recipe[name] for name in ("move_model_id", "crush_model_id", "cutoff",
                                           "content_hash") if name in recipe}
    extras = {name: recipe[name] for name in ("draws", "pre_iv30", "dte_exit") if name in recipe}
    return {
        **simulation, **extras, "mode": "planned_exit",
        "paired_residual_key": key, "paired_residual_artifact": artifact,
    }


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
        ("chooser", {}, pending),
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
    gate = _gate_block(bundle.gate_recipe)
    receipts = _declaration_receipts(
        bundle.source_ref, strategy, context, features, forecast, quotes,
        model, analogs, simulation, gate,
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
        chooser={},
        diagnostics={},
        source_ref=bundle.source_ref,
        stage_receipts=receipts,
    )
