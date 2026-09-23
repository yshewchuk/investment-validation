"""Explicit inputs and receipts for the native scoring execution graph."""
from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from math import isfinite, log
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from scipy.stats import norm

from engine.v2.domain.generation import (
    DISABLED,
    Geometry,
    Pricing,
    generate,
    has_resolvable_expiry,
    price,
)
from engine.v2.domain.valuation import terminal_payoff
from engine.v2.foundation import content_hash, to_document
from engine.v2.scoring import financial

STAGE_NAMES = (
    "resolve_context", "features", "forecast", "geometry", "pricing",
    "model", "analogs", "simulation", "gate", "chooser", "diagnostics",
    "serialization",
)


@dataclass(frozen=True)
class StageReceipt:
    stage: str
    input_hash: str
    output_hash: str
    owner: str = "engine.v2.scoring"


@dataclass(frozen=True)
class StageObservation:
    """Stable documents and receipt emitted when one native stage completes."""

    input_document: Any
    output_document: Any
    receipt: StageReceipt


StageObserver = Callable[[StageObservation], None]


@dataclass(frozen=True)
class NativeScoreInputs:
    """All stage outputs required before a canonical record can be emitted."""

    context: Mapping[str, Any]
    features: Mapping[str, Any]
    forecast: Mapping[str, Any]
    geometry: Geometry | None
    pricing: Pricing | None
    analogs: Mapping[str, Any]
    simulation: Mapping[str, Any]
    gate: Mapping[str, Any]
    chooser: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    source_ref: str
    stage_receipts: tuple[StageReceipt, ...]
    #: The payoff-calibration/model layer (exp_pnl_model, win_model), added
    #: after every other block already had a caller. Defaulted and exempted
    #: from the required-receipt check below (like diagnostics) so the many
    #: existing NativeScoreInputs call sites that predate this field, and
    #: that hard-code their own declared stage_receipts tuples, keep working
    #: unchanged -- the "model" stage is still always EXECUTED (see
    #: assemble_native_values), just not required to be pre-DECLARED.
    model: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_ref.strip():
            raise ValueError("native score inputs require source_ref")
        stages = tuple(receipt.stage for receipt in self.stage_receipts)
        seen = set(stages)
        required = set(STAGE_NAMES) - {"diagnostics", "model"}
        missing = sorted(required - seen)
        if missing:
            raise ValueError(f"missing native stage receipts: {missing}")
        unknown = sorted(seen - set(STAGE_NAMES))
        if unknown:
            raise ValueError(f"unknown native stage receipts: {unknown}")
        duplicates = sorted(stage for stage in seen if stages.count(stage) > 1)
        if duplicates:
            raise ValueError(f"duplicate native stage receipts: {duplicates}")

    @classmethod
    def from_legacy_fields(cls, fields: Mapping[str, Any]) -> "NativeScoreInputs":
        """Create a migration input with explicit stage ownership.

        This compatibility constructor is only for old callers.  New scoring
        code should build the stage blocks directly and provide real receipts.
        """
        values = dict(fields)
        blocks = {name: values for name in ("context", "features", "forecast",
                                             "model", "analogs", "simulation",
                                             "gate", "chooser", "diagnostics")}
        receipts = tuple(StageReceipt(
            stage,
            content_hash({"stage": stage, "values": values}),
            content_hash({"stage": stage, "values": values}),
        ) for stage in STAGE_NAMES if stage != "diagnostics")
        return cls(**blocks, geometry=None, pricing=None,
                   source_ref="compatibility-input",
                   stage_receipts=receipts)


def receipt(stage: str, inputs: Any, output: Any) -> StageReceipt:
    if stage not in STAGE_NAMES:
        raise ValueError(f"unknown scoring stage: {stage}")
    return StageReceipt(stage, content_hash(inputs), content_hash(output))


def _emit_stage(
    executed: list[StageReceipt],
    stage: str,
    input_document: Any,
    output_document: Any,
    observer: StageObserver | None,
) -> None:
    input_document = to_document(input_document)
    output_document = to_document(output_document)
    item = receipt(stage, input_document, output_document)
    executed.append(item)
    if observer is not None:
        observer(StageObservation(
            deepcopy(input_document), deepcopy(output_document), item,
        ))


_PRICING_OUTPUTS = frozenset({
    "entry_cost", "fill", "legs", "model_artifact_ids", "selected_contracts",
    "structure_width",
})
_FORECAST_OUTPUTS = frozenset({
    "driver_prediction", "forecast_abs_move", "runup_move_prediction",
    "pred_iv_crush", "pred_iv_crush_30", "model_fair_pct",
    "forecast_p10", "forecast_p90", "forecast_sd",
})
_SIMULATION_OUTPUTS = frozenset({
    "exp_pnl_sim", "win_sim", "sim_p10", "sim_p90", "pool_n",
})
_GATE_OUTPUTS = frozenset({"gate_score", "gate_threshold", "gate_pass"})
_ANALOG_OUTPUTS = frozenset({
    "exp_pnl_analog", "win_analog", "ci_low", "ci_high", "n_analogs",
})
#: engine/score.py:2210/2214-2216 -- the payoff-calibration/model layer.
_MODEL_OUTPUTS = frozenset({"exp_pnl_model", "win_model"})
_FINANCIAL_OUTPUTS = frozenset({
    "entry_cost_pct", "model_vs_market", "fair_premium_pct",
    "premium_vs_fair", "cost_over_width", "terminal_payoff",
    "planned_exit", "payoff_horizon", "payoff_refusal",
    "exp_pnl_model", "win_model", "exp_pnl_analog", "win_analog",
    "implied_move", "driver_name", "payoff", "strike",
    "entry_date", "exit_date", "expiry",
})
_OWNED_OUTPUTS = (
    _PRICING_OUTPUTS | _FORECAST_OUTPUTS | _SIMULATION_OUTPUTS
    | _GATE_OUTPUTS | _ANALOG_OUTPUTS | _MODEL_OUTPUTS
)
_DIAGNOSTIC_PROTECTED = _OWNED_OUTPUTS | _FINANCIAL_OUTPUTS
_ROLE_OUTPUTS = {
    "driver": ("driver_prediction",),
    "size": ("forecast_abs_move",),
    "implied_t1": ("driver_prediction",),
    "runup_move": ("runup_move_prediction",),
    "iv_crush": ("pred_iv_crush", "pred_iv_crush_30"),
    "fair_value": ("model_fair_pct",),
}
# Executor objects are identified in receipts by their str() identity (the
# gate's ``forecast_recipe``), never deep-copied into an observer document.
_INTERNAL_STAGE_FIELDS = frozenset({"executors", "forecast_executor"})
# Source-only feature fields: part of the features block (and so of its
# hashes and receipts), never merged into the record's values. The strict
# capture's per-role model rows (``role_model_inputs``) are what frozen
# inference was fed for each binding; they are inputs, not record fields.
_SOURCE_ONLY_FEATURE_FIELDS = frozenset({"role_model_inputs"})
_STRATEGY_FORECAST_ROLES = {
    "STR-THRU": ("driver",),
    "STR-RUNUP": ("implied_t1", "runup_move"),
    "TWIN-P": ("size",),
    "TWIN-P5": ("size",),
    "CND-PS": ("size",),
    "BFLY-P": ("size",),
    "BFLY-P5": ("size",),
    "RAMP7": ("size",),
    "CTR5": ("size",),
}
_SIM_DRAWS = 4000
_SIM_MIN_POOL = 250
_SIM_MIN_VOL = 0.01
_SIM_MIN_SPOT_FRACTION = 1e-4

# -- Legacy actionable-flag derivation (R4-9) --------------------------------
# Values mirror engine/score.py's reference constants exactly. Not imported
# from there: engine/v2 has no dependency on legacy engine/ code, so the
# thresholds are reproduced as literals and cited by file:line instead.
ATM_TOLERANCE_PCT = 2.0     # engine/score.py:204
WIDE_MARKET_RATIO = 0.5     # engine/fills.py:25
BAD_QUOTE_COST_PCT = 30.0   # engine/fills.py:34
GATE_MCAP_FLOOR = 1e9       # engine/score.py:199

# -- Model layer (payoff calibration -> exp_pnl_model/win_model) ------------
# engine/payoff.py:79-82 PAYOFF_DRIVER: which strategies have a calibrated
# payoff at all. STR-THRU is a single-driver LINE (``_execute_model`` below);
# STR-RUNUP (R4-17) is the two-driver SURFACE ``engine/payoff.py``'s
# ``RunupPayoffSurface`` fits (implied T-1 move + exit moneyness), executed
# by ``_execute_runup_model``. Every other strategy (the bounded native
# builder declares no forecast recipe for their drivers) refuses with
# NO_PAYOFF_MAP exactly as legacy's driver_for() would for a strategy absent
# from its dict.
_PAYOFF_DRIVER_STRATEGIES = frozenset({"STR-THRU", "STR-RUNUP"})
_PAYOFF_DRIVER_FIELD = {"STR-THRU": "driver_prediction"}

# -- Advisory-vs-refusal flag taxonomy ---------------------------------------
# Whether a reason code the scorer emits merely ANNOTATES a row that still
# carries a score, or WITHHOLDS the row's own numbers (a refusal). Legacy
# (engine/score.py) distinguishes these; the native path did not, so any
# single flag refused the whole row regardless of what it meant. See
# application.py's row_status, the one place this taxonomy is consulted.
#
# This is a closed allow-list, not a denial-list: every code below is
# ADVISORY because legacy evidence says so; every other code -- including
# every native-only engineering-invariant code emitted elsewhere in this
# module (the `INVALID_*`, `UNOWNED_*`, `UNKNOWN_*`, `NONFINITE_*`,
# `UNSUPPORTED_*`, `INSUFFICIENT_*` and most `MISSING_*` codes below, none of
# which have a legacy counterpart or corpus evidence) -- refuses by omission.
# That is the conservative default the task requires, not an oversight.
#
# Evidence is measured from the frozen corpus (fixtures/tier0/*/pairs/*.json,
# 58 records, 52 carrying flags; "scored" = legacy's own
# `ScoreResult.scored` property, `exp_pnl_model is not None or
# exp_pnl_analog is not None`) plus the cited engine/score.py source lines.
ADVISORY_FLAGS = frozenset({
    # engine/score.py:1337 (`calendar.is_projected`) -- a raw calendar fact
    # about the resolved exit date vs the observed-through horizon, not a
    # data defect. Corpus: 33 seen / 27 scored (21 by the task's own count).
    "PROJECTED_CALENDAR",
    # engine/score.py:1687 -- an older chain substituted for the requested
    # date (`_fresh_quote_date`). Corpus: 28 seen / 25 scored (21 by the
    # task's own count).
    "STALE_QUOTE",
    # engine/fills.py:150 `FillModel.is_wide` -- a fill-quality note on a
    # leg that already priced, not a pricing failure. Corpus: 22 seen /
    # 19 scored.
    "WIDE_MARKET",
    # engine/score.py:1803-1807 -- moneyness vs ATM tolerance; the strike
    # resolved and priced, it is simply away from the money. Corpus:
    # 10 seen / 7 scored (6 by the task's own count).
    "EXTRAPOLATED",
    # engine/score.py:3082-3085; engine/SCORING.md states the design intent
    # outright: "the result carries LAYER_DISAGREE and both numbers
    # survive." Corpus: 2 seen / 2 scored.
    # User decision, 2026-09-18: LAYER_DISAGREE stays warning-only (advisory,
    # not refusing). Recorded here per that decision, not re-derived.
    "LAYER_DISAGREE",
    # engine/score.py:3178, with the FLAGS docstring at :159-166 stating
    # outright: "the row is fully scored and priced; only the champion
    # ranking is absent" -- deliberately distinguished there from
    # MISSING_FEATURES, which withholds the row's own numbers. Corpus:
    # 20 seen / 20 scored.
    "CHOOSER_MISSING_FEATURES",
    # engine/score.py:2079/2191/2366 -- fires inside `_score_model` /
    # `_score_runup_model` and returns early, but `Scorer.score` (:1448,
    # :1464) calls `_score_analogs` UNCONDITIONALLY afterward regardless:
    # the analog layer is independent and still sets `exp_pnl_analog`.
    # Corpus: 35 seen / 29 scored (via the analog layer) -- legacy
    # demonstrably still scores most rows carrying it.
    "NO_PAYOFF_MAP",
    # engine/analogs.py:600 -- `thin = returns.size < MIN_ANALOGS` is set
    # ALONGSIDE, not instead of, the real mean/median/win_rate for any
    # nonzero analog count; comment there: "thin is reported instead and
    # the dashboard renders it as low-confidence." Only the degenerate
    # zero-analog case (`_empty()`, analogs.py:207) withholds a value, and
    # that path stamps `thin=True` too -- the corpus's 6 THIN_ANALOGS
    # occurrences all land on that zero-analog edge case (0 scored there),
    # which is not representative of the flag's general, designed
    # behaviour; classified advisory on the source-code evidence.
    "THIN_ANALOGS",
})


def flags_refuse(flags: Iterable[str]) -> bool:
    """True if any of ``flags`` withholds the row's own score.

    Every code not in :data:`ADVISORY_FLAGS` refuses -- an unclassified or
    native-only engineering code is conservative-refused by omission, never
    advisory by accident.
    """
    return any(flag not in ADVISORY_FLAGS for flag in flags)


def _merge_stage(values: dict[str, Any], block: Mapping[str, Any]) -> None:
    values.update({key: value for key, value in block.items()
                   if key not in _OWNED_OUTPUTS
                   and key not in _INTERNAL_STAGE_FIELDS
                   and key not in _SOURCE_ONLY_FEATURE_FIELDS
                   and key != "flags"})


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _add_flag(flags: list[str], value: Any) -> None:
    code = str(value)
    if code and code not in flags:
        flags.append(code)


def _add_executor_refusal(flags: list[str], error: Exception,
                          fallback: str) -> None:
    reasons = tuple(getattr(error, "reason_codes", ()) or ())
    if reasons:
        for reason in reasons:
            _add_flag(flags, reason)
    else:
        _add_flag(flags, fallback)


def _to_day(value: Any) -> np.datetime64 | None:
    """Parse a date-like value the way ``_remaining_dte`` does (see below)."""
    if value is None:
        return None
    try:
        day = np.datetime64(str(value)[:10])
    except ValueError:
        return None
    return None if np.isnat(day) else day


def _check_projected_calendar(values: Mapping[str, Any], flags: list[str]) -> None:
    """PROJECTED_CALENDAR -- engine/score.py:1337 (``calendar.is_projected``).

    Mirrors ``TradingCalendar.is_projected`` (engine/calendar.py:241):
    ``exit_date.normalize() > observed_through``, strict. Both sides are raw
    calendar facts -- the resolved exit date, and ``calendar_observed_through``
    (the last date backed by real observed price history, past which the
    calendar is rule-projected) -- never a calculated scoring answer.
    """
    exit_day = _to_day(values.get("exit_date"))
    observed_day = _to_day(values.get("calendar_observed_through"))
    if exit_day is None or observed_day is None:
        return
    if exit_day > observed_day:
        _add_flag(flags, "PROJECTED_CALENDAR")


def _check_stale_quote(values: Mapping[str, Any], flags: list[str]) -> None:
    """STALE_QUOTE -- engine/score.py:1687 (fallback chain substitution).

    Legacy sets this when no chain exists for the requested decision date and
    an older chain, within a caller bound, is substituted instead
    (``_fresh_quote_date``, engine/score.py:1802). The source-owned signature
    of that substitution is that the raw quotes actually used were observed
    strictly before the requested entry date: ``quote_date`` (the date the
    supplied raw quotes were observed) and ``entry_date`` (the requested
    decision date) are both raw facts, never calculated answers. This
    reproduces the trigger. It does NOT reproduce the session-count age or the
    caller's max-age bound: those require a calendar object, which a bounded
    SourceBundle does not carry -- native cannot currently tell an in-bound
    substitution from an out-of-bound one, only that a substitution occurred.
    """
    quote_day = _to_day(values.get("quote_date"))
    entry_day = _to_day(values.get("entry_date"))
    if quote_day is None or entry_day is None:
        return
    if quote_day < entry_day:
        _add_flag(flags, "STALE_QUOTE")


def _check_wide_market(pricing: Pricing, flags: list[str]) -> None:
    """WIDE_MARKET -- engine/score.py:1789 (``priced.any_wide_market``).

    Mirrors ``FillModel.is_wide`` (engine/fills.py:150):
    ``(ask - bid) / mid > WIDE_MARKET_RATIO``, with a zero-or-negative mid
    counting as wide (no market at all). Reads the already-priced legs'
    bid/ask; does not change pricing.
    """
    if pricing.refusal is not None:
        return
    for leg in pricing.legs:
        mid = (leg.bid + leg.ask) / 2.0
        wide = True if mid <= 0.0 else (leg.ask - leg.bid) / mid > WIDE_MARKET_RATIO
        if wide:
            _add_flag(flags, "WIDE_MARKET")
            return


def _check_bad_quote(pricing: Pricing, flags: list[str]) -> None:
    """BAD_QUOTE -- engine/score.py ``_price_entry`` (EXP-117).

    Mirrors legacy exactly: once the structure priced, and only when the
    spot is nonzero and the cost exists, ``entry_cost / spot * 100 >
    BAD_QUOTE_COST_PCT`` flags the row. Legacy then leaves before the model,
    analog, gate and chooser layers (``Scorer.score``'s BAD_QUOTE exit);
    :func:`_append_late_stages` withholds the same layers.
    """
    if pricing.refusal is not None:
        return
    spot = _finite(pricing.spot)
    cost = _finite(pricing.entry_cost)
    if not spot or cost is None:
        return
    if cost / spot * 100.0 > BAD_QUOTE_COST_PCT:
        _add_flag(flags, "BAD_QUOTE")


#: The late stages legacy never reaches on a BAD_QUOTE row, and the output
#: native records for each of them instead of executing it.
_BAD_QUOTE_WITHHELD = {"withheld": "BAD_QUOTE"}
#: Forecast outputs a BAD_QUOTE row does not carry: legacy serves every
#: forecast role but ``size`` from the layers after its BAD_QUOTE exit.
_BAD_QUOTE_WITHHELD_FIELDS = ("driver_name",) + tuple(sorted({
    output for role, outputs in _ROLE_OUTPUTS.items() if role != "size"
    for output in outputs
}))


def _check_extrapolated(geometry: Geometry, pricing: Pricing,
                        flags: list[str]) -> None:
    """EXTRAPOLATED -- engine/score.py:1803-1807 (moneyness vs ATM tolerance).

    Mirrors legacy exactly: ``moneyness = abs(strike / spot - 1) * 100``,
    ``extrapolated = moneyness > ATM_TOLERANCE_PCT``. Legacy substitutes
    ``request.strike`` for the resolved strike when the caller pinned one;
    native's ``geometry.legs[0].strike`` already reflects a pinned strike
    (``generate()`` forces the caller-supplied value into the leg -- see
    ``engine/v2/domain/generation/structures.py``'s STR-THRU/STR-RUNUP
    branch), so no separate override is needed here. Only evaluated once
    pricing has actually succeeded, matching where legacy computes it.
    """
    if pricing.refusal is not None or not geometry.legs:
        return
    spot = geometry.spot
    if not isfinite(spot) or spot == 0.0:
        return
    strike = geometry.legs[0].strike
    moneyness = abs(float(strike) / spot - 1.0) * 100.0
    if moneyness > ATM_TOLERANCE_PCT:
        _add_flag(flags, "EXTRAPOLATED")


def _gate_in_domain(inputs: NativeScoreInputs, values: Mapping[str, Any]) -> bool:
    """Mirrors engine/score.py:2475 ``_gate_in_domain`` (mcap floor clause).

    Legacy's computed-moves clause was removed 2026-09-06 (see the docstring
    at that line); only the market-cap floor remains live. Reads ``mcap_log``
    from the model's own feature vector (never a legacy answer field). A
    missing or non-finite ``mcap_log`` is in-domain by construction, matching
    legacy's silent fall-through for the same cases.
    """
    mcap_log = _finite(_facts(inputs, values).get("mcap_log"))
    if mcap_log is None:
        return True
    return mcap_log >= log(GATE_MCAP_FLOOR)


def _facts(inputs: NativeScoreInputs, values: Mapping[str, Any]) -> dict[str, Any]:
    """Lowest precedence first: ``source_features`` (the captured base
    feature columns -- im/im_d1/im_d5/im_d10/iv10/iv30/... --
    ``tools/capture_tier0_corpus.py``'s ``features = {"model_inputs": ...,
    "source_features": ...}``) is the raw legacy-features-DataFrame
    equivalent (R4-20 gap: it was never merged, so the frozen gate champion
    ``gate_midfill_str_thru_forecast_analog`` could never see its own
    ``im_d*`` inputs). It is merged FIRST so context, the rest of
    ``inputs.features``, the flattened driver ``model_inputs``, and finally
    execution-owned ``values`` can each still override it -- a source
    snapshot must never shadow an owned output or a context fact.
    """
    facts: dict[str, Any] = {}
    source_features = inputs.features.get("source_features")
    if isinstance(source_features, Mapping):
        facts.update(source_features)
    facts.update(inputs.context)
    facts.update(inputs.features)
    model_inputs = inputs.features.get("model_inputs")
    if isinstance(model_inputs, Mapping):
        facts.update(model_inputs)
    facts.update(values)
    return facts


def _linear(spec: Mapping[str, Any], facts: Mapping[str, Any],
            owner: str) -> float:
    value = _finite(spec.get("intercept", 0.0))
    coefficients = spec.get("coefficients", {})
    if value is None or not isinstance(coefficients, Mapping):
        raise ValueError(f"INVALID_{owner.upper()}_MODEL")
    for feature, raw_coefficient in coefficients.items():
        coefficient = _finite(raw_coefficient)
        feature_value = _finite(facts.get(feature))
        if coefficient is None:
            raise ValueError(f"INVALID_{owner.upper()}_MODEL")
        if feature_value is None:
            raise ValueError(f"MISSING_{owner.upper()}_INPUT:{feature}")
        value += coefficient * feature_value
    if not isfinite(value):
        raise ValueError(f"INVALID_{owner.upper()}_OUTPUT")
    return value


def _required_forecast_roles(
    inputs: NativeScoreInputs,
    strategy: str | None,
) -> tuple[str, ...]:
    source = strategy
    if source is None and inputs.geometry is not None:
        source = inputs.geometry.strategy
    if source is None:
        source = inputs.context.get("strategy")
    name = str(source or "").split("@", 1)[0]
    roles = list(_STRATEGY_FORECAST_ROLES.get(name, ()))
    raw = inputs.forecast.get("required_roles") or ()
    declared = [str(raw)] if isinstance(raw, str) else [str(item) for item in raw]
    for role in declared:
        if role not in roles:
            roles.append(role)
    if _is_planned(inputs.simulation):
        for role in ("size", "iv_crush"):
            if role not in roles:
                roles.append(role)
    return tuple(roles)


def _is_planned(block: Mapping[str, Any]) -> bool:
    """A planned-exit simulation: declared by mode, raw residual rows, or
    (P5-4) a frozen paired residual pool artifact."""
    from engine.v2.scoring.native_residuals import PAIRED_ARTIFACT_FIELD

    return (
        block.get("mode") == "planned_exit"
        or block.get("residuals") is not None
        or PAIRED_ARTIFACT_FIELD in block
    )


def _validate_forecast_roles(inputs: NativeScoreInputs,
                             output: Mapping[str, Any],
                             flags: list[str],
                             strategy: str | None,
                             invalid_fields: set[str]) -> None:
    for role in _required_forecast_roles(inputs, strategy):
        fields = _ROLE_OUTPUTS.get(role)
        if fields is None:
            _add_flag(flags, f"UNKNOWN_FORECAST_ROLE:{role}")
        elif any(output.get(field) is not None for field in fields):
            continue
        elif not any(field in invalid_fields for field in fields):
            _add_flag(flags, f"MISSING_FORECAST_OUTPUT:{role}")


def _execute_frozen_forecast(
    block: Mapping[str, Any],
    output: dict[str, Any],
    invalid_fields: set[str],
    flags: list[str],
) -> bool:
    frozen = block.get("frozen_outputs")
    if frozen is None:
        return False
    if not isinstance(frozen, Mapping) or not block.get("artifact_hashes"):
        _add_flag(flags, "INVALID_FROZEN_INFERENCE")
        return False
    for field, raw in frozen.items():
        if field not in _FORECAST_OUTPUTS:
            _add_flag(flags, f"UNKNOWN_FROZEN_OUTPUT:{field}")
            continue
        if raw is None:
            continue
        value = _finite(raw)
        if value is None:
            invalid_fields.add(field)
            _add_flag(flags, f"NONFINITE_FORECAST_OUTPUT:{field}")
        else:
            output[field] = value
    return True


#: The adapter legacy's Tier-4 fold caches are served through. Legacy
#: ``tier4.ServingModel.predict`` answers NaN -- silently, no flag -- for a row
#: with a missing or non-finite feature (engine/data/features/tier4.py), and
#: every caller treats that NaN as "no forecast": sizing declines NO_FORECAST
#: (``Scorer._size_from_forecast``), the crush forecast becomes ``None`` and
#: the simulation is undetermined (``Scorer._crush_forecast`` ->
#: ``pnl_sim.expected_pnl`` returns ``None``), the gate's forecast columns
#: stay NaN (``Scorer._forecast_for_gate``).
TIER4_FOLD_ADAPTER = "tier4-serving-fold.v1"
#: Forecast outputs whose absence was an undetermined fold forecast (see
#: above), published so later stages decline the way legacy does.
_UNDETERMINED_FIELD = "forecast_undetermined"


def _missing_feature_refusal(exc: Exception) -> bool:
    return "MISSING_FEATURES" in tuple(getattr(exc, "reason_codes", ()) or ())


def _execute_forecast_executor(
    executor, field, facts, output, invalid_fields, flags, undetermined,
) -> None:
    predict = getattr(executor, "predict", None)
    if not callable(predict):
        _add_flag(flags, "INVALID_FORECAST_EXECUTOR")
        return
    try:
        result = predict(facts)
        raw = result.get(field) if isinstance(result, Mapping) else result
        if isinstance(result, Mapping) and raw is None and len(result) == 1:
            raw = next(iter(result.values()))
        value = _finite(raw)
        if value is None:
            invalid_fields.add(field)
            _add_flag(flags, f"NONFINITE_FORECAST_OUTPUT:{field}")
            return
        output[field] = value
    except (TypeError, ValueError, KeyError) as exc:
        if _missing_feature_refusal(exc):
            # R4-20 gap 4: a missing/non-finite model feature is legacy's
            # MISSING_FEATURES for a champion model (``_score_model``), and a
            # silent NaN for a Tier-4 fold (see TIER4_FOLD_ADAPTER). Either
            # way the output is accounted for, not a second
            # MISSING_FORECAST_OUTPUT refusal legacy never raises.
            invalid_fields.add(field)
            if getattr(executor, "adapter", None) == TIER4_FOLD_ADAPTER:
                undetermined.add(field)
                return
        elif not tuple(getattr(exc, "reason_codes", ()) or ()):
            invalid_fields.add(field)
        _add_executor_refusal(flags, exc, f"INVALID_FORECAST_EXECUTOR:{field}")


def _execute_forecast_model(spec, field, facts, output, invalid_fields, flags):
    if not isinstance(spec, Mapping):
        _add_flag(flags, f"UNOWNED_FORECAST_OUTPUT:{field}")
        return
    try:
        output[field] = _linear(spec, facts, "forecast")
    except ValueError as exc:
        invalid_fields.add(field)
        _add_flag(flags, exc)


def _execute_local_forecast(
    inputs: NativeScoreInputs,
    block: Mapping[str, Any],
    values: Mapping[str, Any],
    output: dict[str, Any],
    invalid_fields: set[str],
    flags: list[str],
    undetermined: set[str],
) -> bool:
    models = block.get("models", {})
    if not isinstance(models, Mapping):
        _add_flag(flags, "INVALID_FORECAST_MODELS")
        return False
    declared = False
    facts = _facts(inputs, values)
    executors = block.get("executors", {})
    if not isinstance(executors, Mapping):
        _add_flag(flags, "INVALID_FORECAST_EXECUTORS")
        executors = {}
    # ``stored``: a stored Tier-4 cell THIS PROCESS resolved from the table
    # the capture pointed it at (``checks/phase4_stored_forecasts.py``).
    # ``stored_refs``: the capture's address for that cell, carrying no
    # value. A reference that nobody resolved must refuse, never fall
    # through to a fold or a model -- that fallback would compare a
    # DIFFERENT forecast path against legacy's table read and call it parity.
    stored = block.get("stored") or {}
    stored_refs = block.get("stored_refs") or {}
    for field in _FORECAST_OUTPUTS:
        if field in stored:
            # Legacy ``Scorer._crush_forecast``: a stored Tier-4 value for the
            # event wins, and no fold is served.
            declared = True
            output[field] = float(stored[field]["value"])
            continue
        if field in stored_refs:
            declared = True
            invalid_fields.add(field)
            _add_flag(flags, f"UNRESOLVED_STORED_FORECAST:{field}")
            continue
        executor = executors.get(field)
        if executor is not None:
            declared = True
            _execute_forecast_executor(
                executor, field, facts, output, invalid_fields, flags,
                undetermined,
            )
            continue
        spec = models.get(field, block.get(field))
        if spec is None:
            continue
        declared = True
        _execute_forecast_model(spec, field, facts, output, invalid_fields, flags)
    return declared


def _execute_forecast(inputs: NativeScoreInputs, values: dict[str, Any],
                      flags: list[str], strategy: str | None) -> dict[str, Any]:
    block = inputs.forecast
    output: dict[str, Any] = {}
    invalid_fields: set[str] = set()
    if block.get("driver_name") is not None:
        output["driver_name"] = str(block["driver_name"])
    undetermined: set[str] = set()
    frozen_declared = _execute_frozen_forecast(
        block, output, invalid_fields, flags,
    )
    local_declared = _execute_local_forecast(
        inputs, block, values, output, invalid_fields, flags, undetermined,
    )
    _validate_forecast_roles(
        inputs, output, flags, strategy, invalid_fields,
    )
    if not (frozen_declared or local_declared):
        _add_flag(flags, "MISSING_FORECAST_INPUT")
    if "forecast_abs_move" in undetermined and "size" in _required_forecast_roles(
        inputs, strategy,
    ):
        # Legacy ``_size_from_forecast``: a NaN fold forecast cannot size the
        # structure, and the row declines NO_FORECAST.
        _add_flag(flags, "NO_FORECAST")
    if undetermined:
        output[_UNDETERMINED_FIELD] = tuple(sorted(undetermined))
    values.update(output)
    return output


def _initial_values(
    inputs: NativeScoreInputs,
    flags: list[str],
    compatibility: bool,
    strategy: str | None,
    observer: StageObserver | None,
) -> tuple[dict[str, Any], list[StageReceipt], Any]:
    values: dict[str, Any] = {}
    blocks = (inputs.context, inputs.features, inputs.forecast, inputs.analogs,
              inputs.simulation, inputs.gate, inputs.chooser, inputs.diagnostics)
    legacy_cost = next((block.get("entry_cost") for block in blocks
                        if block.get("entry_cost") is not None), None)
    executed: list[StageReceipt] = []
    prior: Any = {"source_ref": inputs.source_ref}
    for stage, block in (("resolve_context", inputs.context), ("features", inputs.features)):
        _merge_stage(values, block)
        output = {key: value for key, value in block.items()
                  if key not in _OWNED_OUTPUTS and key != "flags"}
        _emit_stage(executed, stage, prior, output, observer)
        # A SNAPSHOT, not a live reference: ``values`` is a single mutable
        # dict threaded through resolve_context -> features -> forecast, and
        # the forecast stage mutates it (``_execute_forecast``'s
        # ``values.update(output)``) before this "prior" is ever consumed by
        # ``_emit_stage``. An aliased dict here means the NEXT stage's
        # declared "input" quietly includes whatever the stage before it
        # goes on to compute -- for forecast, its own freshly-produced
        # OWNED output. That is invisible when a stage's own block is
        # byte-identical across two runs (the leaked content is identical
        # too), but a stored-forecast reference that resolves at replay and
        # not at capture (``checks/phase4_stored_forecasts.py``) makes the
        # leaked content differ, and the divergence is misreported as the
        # forecast stage's INPUT rather than its OUTPUT.
        prior = {"prior": executed[-1].output_hash, "values": dict(values)}
    if compatibility:
        values.update(inputs.forecast)
        forecast_output = dict(inputs.forecast)
    else:
        forecast_output = _execute_forecast(
            inputs, values, flags, strategy,
        )
    _emit_stage(executed, "forecast", prior, forecast_output, observer)
    return values, executed, legacy_cost


def _strategy_name(inputs: NativeScoreInputs, strategy: str | None,
                   values: Mapping[str, Any]) -> str:
    source = strategy or (inputs.geometry.strategy if inputs.geometry else values.get("strategy"))
    name = str(source or "").split("@", 1)[0]
    if not name:
        raise ValueError("native geometry stage requires a strategy")
    if inputs.geometry is not None and inputs.geometry.strategy != name:
        raise ValueError("request and geometry strategies disagree")
    return name


def _resolve_geometry(inputs: NativeScoreInputs, name: str,
                      values: dict[str, Any]):
    geometry_inputs = dict(values)
    if name not in {"CAL-P", "CND-P"}:
        spot = _finite(geometry_inputs.get("spot"))
        if spot is None or spot <= 0.0:
            return geometry_inputs, Geometry(name, 0.0, 0.0, (), "MISSING_SPOT")
        if (geometry_inputs.get("expiry") is None
                and geometry_inputs.get("post_event_expiry") is None
                and not has_resolvable_expiry(name, geometry_inputs, spot)):
            # A row legacy never priced (NO_CHAIN, COARSE_LADDER, NO_FORECAST
            # at sizing) carries no captured `expiry`, but every such row
            # still carries `quotes`/`event_date`/`exit_date`/`session` --
            # `has_resolvable_expiry` is the same native selection `generate`
            # performs below, so this refuses only when that selection also
            # finds nothing, not merely because legacy never priced the row.
            return geometry_inputs, Geometry(name, spot, 0.0, (), "MISSING_EXPIRY")
    if inputs.geometry is not None:
        geometry_inputs["resolved_legs"] = tuple(vars(leg) for leg in inputs.geometry.legs)
        geometry_inputs["width"] = inputs.geometry.width
    try:
        if name == "DYN-SV":
            geometry = Geometry(
                name, float(values["spot"]), 0.0, (), "DYNAMIC_CHOOSER",
            )
        else:
            geometry = generate(name, geometry_inputs)
    except Exception as exc:
        if inputs.geometry is not None or geometry_inputs.get("resolved_legs"):
            raise
        geometry = Geometry(name, 0.0, 0.0, (), str(exc))
    if geometry.refusal is None and not geometry.legs:
        geometry = Geometry(
            name, geometry.spot, geometry.width, (), "MISSING_CONTRACTS",
        )
    return geometry_inputs, geometry


def _quote_map(inputs: NativeScoreInputs, name: str) -> dict:
    declared = inputs.context.get("quotes")
    if isinstance(declared, Mapping):
        return dict(declared)
    declared = inputs.features.get("quotes")
    if isinstance(declared, Mapping):
        return dict(declared)
    if inputs.pricing is not None:
        quotes = {(leg.right, leg.strike, leg.expiry): {"bid": leg.bid, "ask": leg.ask}
                  for leg in inputs.pricing.legs}
    else:
        quotes = {}
    return quotes


def _resolve_pricing(inputs: NativeScoreInputs, name: str, geometry: Geometry,
                     quotes: Mapping, alpha: float, compatibility: Any):
    if inputs.pricing is not None and inputs.pricing.strategy != name:
        raise ValueError("request and pricing strategies disagree")
    if geometry.refusal:
        return Pricing(name, geometry.spot, 0.0, (), geometry.refusal)
    if not quotes and inputs.source_ref == "compatibility-input":
        return Pricing(name, geometry.spot, float(compatibility or 0.0), (), None)
    if not quotes:
        return Pricing(name, geometry.spot, 0.0, (), "MISSING_PRICING_INPUT")
    try:
        return price(geometry, quotes, alpha)
    except Exception as exc:
        return Pricing(name, geometry.spot, 0.0, (), str(exc))


def _publish_pricing(values: dict[str, Any], geometry: Geometry,
                     pricing: Pricing, alpha: float) -> None:
    legs = pricing.legs or geometry.legs
    values.update({
        "spot": geometry.spot,
        "structure_width": geometry.width,
        "entry_cost": (
            pricing.entry_cost if pricing.refusal is None else None
        ),
        "fill": alpha,
        "legs": tuple(vars(leg) for leg in legs),
        "selected_contracts": tuple(vars(leg) for leg in geometry.legs),
        # engine/score.py:2342 -- ``result.strike = float(priced.legs[0].strike)``,
        # set only once pricing has actually resolved (legacy never reaches
        # that assignment on a refusal, and a caller-pinned strike in
        # ``context`` is not enough on its own: an unpinned request resolves
        # its strike here, in geometry/pricing, not before). Same leg
        # ``_check_extrapolated`` above already reads off ``geometry`` --
        # this is the one place that fact becomes a published stage output,
        # which ``financial.py``'s payoff-surface reader needs.
        "strike": (float(legs[0].strike)
                   if pricing.refusal is None and legs else None),
    })


def _merge_diagnostics(values: dict[str, Any],
                       block: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        key: value for key, value in block.items()
        if key not in _DIAGNOSTIC_PROTECTED and key != "flags"
    }
    values.update(output)
    return output


def _simulation_spots(block: Mapping[str, Any],
                      flags: list[str]) -> tuple[float, ...] | None:
    raw_spots = block.get("terminal_spots")
    if raw_spots is None:
        declared = False
        for field in _SIMULATION_OUTPUTS:
            if block.get(field) is not None:
                declared = True
                _add_flag(flags, f"UNOWNED_SIMULATION_OUTPUT:{field}")
        if not declared:
            _add_flag(flags, "MISSING_SIMULATION_INPUT")
        return None
    try:
        spots = tuple(float(item) for item in raw_spots)
    except (TypeError, ValueError):
        _add_flag(flags, "INVALID_SIMULATION_SPOTS")
        return None
    if not spots or not all(isfinite(item) and item >= 0.0 for item in spots):
        _add_flag(flags, "INVALID_SIMULATION_SPOTS")
        return None
    return spots


def _simulation_weights(block: Mapping[str, Any], count: int,
                        flags: list[str]) -> tuple[float, ...] | None:
    raw_weights = block.get("weights")
    try:
        weights = (tuple(1.0 for _ in range(count)) if raw_weights is None
                   else tuple(float(item) for item in raw_weights))
    except (TypeError, ValueError):
        _add_flag(flags, "INVALID_SIMULATION_WEIGHTS")
        return None
    total_weight = sum(weights)
    if (len(weights) != count or total_weight <= 0.0
            or not all(isfinite(item) and item >= 0.0 for item in weights)):
        _add_flag(flags, "INVALID_SIMULATION_WEIGHTS")
        return None
    return weights


def _simulation_cutoff(block: Mapping[str, Any], output: dict[str, Any],
                       flags: list[str]) -> None:
    if block.get("pnl_cutoff") is None:
        return
    cutoff = _finite(block["pnl_cutoff"])
    if cutoff is None:
        _add_flag(flags, "INVALID_PNL_CUTOFF")
    else:
        output["pnl_cutoff"] = cutoff


def _remaining_dte(values: Mapping[str, Any]) -> float | None:
    expiry = values.get("expiry")
    exit_date = values.get("exit_date")
    if expiry is None or exit_date is None:
        return None
    try:
        difference = (
            np.datetime64(str(expiry)[:10])
            - np.datetime64(str(exit_date)[:10])
        )
        return float(difference / np.timedelta64(1, "D"))
    except ValueError:
        return None


def _black_scholes_put(spot: np.ndarray, strike: float, years: float,
                       vol: np.ndarray) -> np.ndarray:
    volatility = np.maximum(np.asarray(vol, dtype=float), _SIM_MIN_VOL)
    if years <= 0.0:
        return np.maximum(float(strike) - spot, 0.0)
    sigma_time = volatility * np.sqrt(years)
    d1 = (
        np.log(spot / float(strike))
        + 0.5 * volatility ** 2 * years
    ) / sigma_time
    return (
        float(strike) * norm.cdf(-(d1 - sigma_time))
        - spot * norm.cdf(-d1)
    )


def _residual_arrays(raw: Any) -> tuple[np.ndarray, ...]:
    """The compatibility-path paired pool, as legacy ``ResidualPool`` holds it.

    Legacy drops a row only where ``event_date``, ``pred_abs_move``,
    ``err_move`` or ``err_crush`` is MISSING (``history.dropna(subset=...)``,
    engine/pnl_sim.py ``ResidualPool.__init__``): NaN/NaT/None. A +/-inf row
    stays in the pool, counts toward ``pool_n`` and the decile edges, and can
    be drawn -- and the simulation then carries it through the same numpy
    arithmetic legacy does (R4-20 gap 1). So this drops NaN only, never inf.
    """
    rows: list[tuple[np.datetime64, float, float, float]] = []
    for item in raw or ():
        if not isinstance(item, Mapping):
            continue
        try:
            event_date = np.datetime64(str(item["event_date"]))
            prediction = float(item["pred_abs_move"])
            move = float(item["err_move"])
            crush = float(item["err_crush"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isnat(event_date) or any(
            value != value for value in (prediction, move, crush)
        ):
            continue
        rows.append((event_date, prediction, move, crush))
    rows.sort(key=lambda item: item[0])
    if not rows:
        empty = np.empty(0, dtype=float)
        return np.empty(0, dtype="datetime64[D]"), empty, empty, empty
    return (
        np.asarray([item[0] for item in rows]),
        np.asarray([item[1] for item in rows], dtype=float),
        np.asarray([item[2] for item in rows], dtype=float),
        np.asarray([item[3] for item in rows], dtype=float),
    )


def _simulation_residual_arrays(
    block: Mapping[str, Any], flags: list[str],
) -> tuple[np.ndarray, ...] | None:
    """The paired pool: the frozen artifact when declared (P5-4, never
    rebuilt, MODEL_NOT_READY on a missing/mismatched one), else the
    request-supplied rows exactly as before."""
    from engine.v2.scoring import native_residuals

    if native_residuals.PAIRED_ARTIFACT_FIELD not in block:
        return _residual_arrays(block.get("residuals"))
    arrays, flag = native_residuals.paired_arrays_from_artifact(block)
    if flag is not None:
        _add_flag(flags, flag)
    return arrays


def _draw_residuals(block: Mapping[str, Any], event_date: Any,
                    prediction: float, key: str,
                    flags: list[str]) -> tuple[Any, ...] | None:
    arrays = _simulation_residual_arrays(block, flags)
    if arrays is None:
        return None
    dates, predicted, move, crush = arrays
    cutoff = _event_day(event_date)
    if cutoff is None:
        _add_flag(flags, "INVALID_SIMULATION_EVENT_DATE")
        return None
    end = int(np.searchsorted(dates, cutoff, side="left"))
    if end < _SIM_MIN_POOL:
        _add_flag(flags, "INSUFFICIENT_SIMULATION_POOL")
        return None
    edges = np.quantile(
        predicted[:end], np.linspace(0.0, 1.0, 11)[1:-1],
    )
    bucket = int(np.searchsorted(edges, prediction, side="right"))
    rows = np.flatnonzero(
        np.searchsorted(edges, predicted[:end], side="right") == bucket
    )
    if rows.size < _SIM_MIN_POOL:
        rows = np.arange(end)
    try:
        draws = int(block.get("draws", _SIM_DRAWS))
    except (TypeError, ValueError):
        draws = 0
    if draws <= 0:
        _add_flag(flags, "INVALID_SIMULATION_DRAWS")
        return None
    seed = int.from_bytes(
        hashlib.sha256(planned_exit_seed_material(key, cutoff).encode()).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    chosen = rows[rng.integers(0, rows.size, size=draws)]
    return move[chosen], crush[chosen], rng, end


def _event_day(event_date: Any) -> np.datetime64 | None:
    """The event's calendar day: the legacy caller normalizes the event date
    to midnight (engine/score.py sets ``result.event_date =
    pd.Timestamp(...).normalize()``), so both the pool cutoff and the seed
    are functions of the day alone, whatever form the date arrives in."""
    try:
        day = np.datetime64(str(event_date)).astype("datetime64[D]")
    except ValueError:
        return None
    return None if np.isnat(day) else day


def planned_exit_seed_material(key: str, event_day: np.datetime64) -> str:
    """The planned-exit simulation's seed material (R4-10).

    The recipe seeds from ``f"{strategy}|{event_date}"`` where the event
    date is the normalized timestamp the scorer holds, whose string form is
    ``YYYY-MM-DD 00:00:00`` (engine/pnl_sim.py ``expected_pnl``, called with
    ``event_date=result.event_date`` from engine/score.py
    ``Scorer._expectation``). Deriving it from the day, not from whatever
    string the context carries, makes the draws a function of the event and
    not of its date's spelling.
    """
    return f"{key}|{np.datetime_as_string(event_day, unit='D')} 00:00:00"


def _is_planned_artifact(block: Mapping[str, Any]) -> bool:
    from engine.v2.scoring.native_residuals import PAIRED_ARTIFACT_FIELD

    return PAIRED_ARTIFACT_FIELD in block


def _planned_parameters(block: Mapping[str, Any], values: Mapping[str, Any],
                        flags: list[str]) -> tuple[Any, ...] | None:
    parameters = {
        "spot": values.get("spot"),
        "entry_cost": values.get("entry_cost"),
        "pre_iv30": block.get("pre_iv30", values.get("pre_iv30")),
        "pred_abs_move": values.get("forecast_abs_move"),
        "pred_iv_crush": values.get(
            "pred_iv_crush", values.get("pred_iv_crush_30"),
        ),
        "dte_exit": block.get("dte_exit", _remaining_dte(values)),
    }
    missing = [
        name for name, value in parameters.items() if _finite(value) is None
    ]
    undetermined = set(values.get(_UNDETERMINED_FIELD) or ())
    silent = {
        "pred_abs_move": "forecast_abs_move" in undetermined,
        "pred_iv_crush": bool({"pred_iv_crush", "pred_iv_crush_30"} & undetermined),
    }
    if missing and all(silent.get(name, False) for name in missing):
        # Legacy ``expected_pnl`` returns ``None`` (NONFINITE_INPUT) without
        # a flag when a fold forecast came back NaN: the simulation is
        # undetermined, the row is not refused for it (R4-20 gap 4).
        return None
    event_date = block.get("event_date", values.get("event_date"))
    if event_date is None:
        missing.append("event_date")
    if block.get("residuals") is None and not _is_planned_artifact(block):
        missing.append("residuals")
    for name in missing:
        _add_flag(flags, f"MISSING_SIMULATION_INPUT:{name}")
    if missing:
        return None
    numeric = tuple(float(parameters[name]) for name in (
        "spot", "entry_cost", "pre_iv30", "pred_abs_move",
        "pred_iv_crush", "dte_exit",
    ))
    if numeric[2] <= 0.0 or numeric[1] <= 0.0 or numeric[5] < 0.0:
        _add_flag(flags, "INVALID_SIMULATION_INPUT")
        return None
    return (*numeric, event_date)


def _planned_exit_simulation(
    block: Mapping[str, Any],
    values: Mapping[str, Any],
    pricing: Pricing,
    key: str,
    flags: list[str],
) -> dict[str, Any]:
    parameters = _planned_parameters(block, values, flags)
    if parameters is None or not pricing.legs:
        if not pricing.legs:
            _add_flag(flags, "MISSING_SIMULATION_INPUT:exit_legs")
        return {}
    spot, entry_cost, pre_iv30, move_forecast, crush_forecast, dte, event_date = parameters
    sampled = _draw_residuals(
        block, event_date, move_forecast, key, flags,
    )
    if sampled is None:
        return {}
    err_move, err_crush, rng, pool_n = sampled
    move = np.maximum(move_forecast + err_move, 0.0)
    crush = crush_forecast + err_crush
    sign = rng.choice((-1.0, 1.0), size=move.size)
    spot_exit = spot * np.maximum(
        1.0 + sign * move / 100.0, _SIM_MIN_SPOT_FRACTION,
    )
    vol_exit = (pre_iv30 / 100.0) * (1.0 + crush / 100.0)
    value = np.zeros(move.size)
    for leg in pricing.legs:
        if not isfinite(float(leg.strike)) or float(leg.quantity) == 0.0:
            continue
        side = 1.0 if str(leg.side).lower() == "buy" else -1.0
        value += (
            side * float(leg.quantity)
            * _black_scholes_put(
                spot_exit, float(leg.strike), dte / 365.0, vol_exit,
            )
        )
    returns = (value - entry_cost) / entry_cost
    return {
        "exp_pnl_sim": float(np.mean(returns)),
        "win_sim": float(np.mean(returns > 0.0)),
        "sim_p10": float(np.quantile(returns, 0.10)),
        "sim_p90": float(np.quantile(returns, 0.90)),
        "pool_n": int(pool_n),
    }


def _execute_simulation(
    inputs: NativeScoreInputs,
    values: dict[str, Any],
    geometry: Geometry,
    pricing: Pricing,
    flags: list[str],
) -> dict[str, Any]:
    block = inputs.simulation
    if block.get("mode") == "not_applicable":
        return {}
    output: dict[str, Any] = {}
    if _is_planned(block):
        output = _planned_exit_simulation(
            block, values, pricing, geometry.strategy, flags,
        )
        _simulation_cutoff(block, output, flags)
        values.update(output)
        return output
    spots = _simulation_spots(block, flags)
    if spots is None:
        return output
    if geometry.refusal or pricing.refusal:
        _add_flag(flags, geometry.refusal or pricing.refusal)
        return output
    weights = _simulation_weights(block, len(spots), flags)
    if weights is None:
        return output
    capital = _finite(block.get("capital_at_risk", abs(pricing.entry_cost)))
    if capital is None or capital <= 0.0:
        _add_flag(flags, "INVALID_SIMULATION_CAPITAL")
        return output
    legs = tuple(vars(leg) for leg in pricing.legs)
    returns = tuple(
        (terminal_payoff(legs, spot) - pricing.entry_cost) / capital
        for spot in spots
    )
    total_weight = sum(weights)
    output["exp_pnl_sim"] = sum(
        weight * result for weight, result in zip(weights, returns)
    ) / total_weight
    output["win_sim"] = sum(
        weight for weight, result in zip(weights, returns) if result > 0.0
    ) / total_weight
    _simulation_cutoff(block, output, flags)
    values.update(output)
    return output


def _model_seed(inputs: NativeScoreInputs, values: Mapping[str, Any]) -> int:
    """Deterministic RNG seed, byte-identical to legacy's bootstrap seed.

    Mirrors engine/score.py's ``sha256(f"{self.snapshot}|{request.key()}")``
    (Scorer._score_model / _score_runup_model, ~line 2891/3057) exactly,
    from the REQUEST-level identity facts captured into ``inputs.context``
    (see engine/score.py's unconditional capture_source_bundle call, ~line
    1847, and the ``requested_structure_params`` merge right after forecast
    sizing). No fallback to ``source_ref``: a bundle missing a required
    identity fact is a capture defect, and seeding on ``source_ref`` instead
    would silently draw a DIFFERENT Monte Carlo path than legacy drew for
    the same trade -- worse than refusing.
    """
    from engine.v2.scoring.identity import bootstrap_seed, score_request_key

    context = inputs.context
    if "snapshot" not in context:
        raise KeyError("_model_seed: missing identity field 'snapshot'")
    snapshot = context["snapshot"]
    key = score_request_key(context)
    return bootstrap_seed(snapshot, key)


def _model_driver_and_cost(
    name: str,
    block: Mapping[str, Any],
    values: Mapping[str, Any],
    flags: list[str],
) -> tuple[float, float, float] | None:
    """Resolve the driver/spot/entry_cost a payoff line would be pushed
    through, or add the refusing flag and return ``None``.

    ``None`` here always means a flag was already added by this function --
    callers just propagate it as "nothing more to do".
    """
    if name not in _PAYOFF_DRIVER_STRATEGIES:
        _add_flag(flags, "NO_PAYOFF_MAP")
        return None
    driver_field = _PAYOFF_DRIVER_FIELD[name]
    driver = _finite(values.get(driver_field))
    spot = _finite(values.get("spot"))
    cost = _finite(values.get("entry_cost"))
    if driver is None:
        _add_flag(flags, f"MISSING_MODEL_INPUT:{driver_field}")
        return None
    if spot is None or cost is None or cost <= 0.0:
        # No entry cost, no denominator -- engine/score.py:2180-2188's own
        # guard. A row already refused for the missing chain keeps that
        # reason rather than picking up a second, misleading one.
        if "COARSE_LADDER" not in flags:
            _add_flag(flags, "NO_CHAIN")
        return None
    return driver, spot, cost


def _artifact_key_mismatch(
    artifact: Any, expected_type: type, name: str, alpha: float | None, cutoff: Any,
) -> bool:
    """True unless ``artifact`` is the right kind AND its full causal key
    (strategy, alpha, cutoff) equals the one the inline fit would have used
    for THIS request.

    Coordinator decision, 2026-09-18: choosing the right artifact is the
    release's job, but the stage must not simply trust that choice -- a
    wrong-fold artifact (wrong alpha, or a LATER cutoff than this request's
    own, i.e. a leaked future fold) must not silently produce a
    plausible-looking number. ``alpha`` is ``None`` when the pricing stage
    never resolved a fill alpha (should not happen once pricing has run, but
    a missing identity component can never match, not "match by omission").
    """
    from engine.v2.models.payoff_artifact import payoff_artifact_key

    if alpha is None or not isinstance(artifact, expected_type):
        return True
    return artifact.key != payoff_artifact_key(name, alpha, cutoff)


def _line_fit_from_artifact(
    artifact: Any, name: str, alpha: float | None, cutoff: Any, flags: list[str],
) -> dict[str, Any] | None:
    """Read a frozen ``PayoffLineArtifact``'s line/residuals -- never fits (P5-4).

    ``None`` means MODEL_NOT_READY was already added: no artifact, the wrong
    kind (a surface artifact on a single-driver strategy), or a causal key
    (strategy, alpha, cutoff) that disagrees with this request's own --
    including a LATER cutoff, which would leak a future fold. There is no
    fallback to fitting from source rows here -- that is the separate,
    mutually exclusive compatibility path in ``_model_fit_and_pool``.
    """
    from engine.v2.models.payoff_artifact import PayoffLineArtifact

    if _artifact_key_mismatch(artifact, PayoffLineArtifact, name, alpha, cutoff):
        _add_flag(flags, "MODEL_NOT_READY")
        return None
    return {
        "intercept": artifact.intercept,
        "slope": artifact.slope,
        "resid_sd": artifact.resid_sd,
        "n": artifact.n,
        "r": artifact.r,
        "residuals": np.asarray(artifact.residuals, dtype=float),
    }


def _surface_fit_from_artifact(
    artifact: Any, name: str, alpha: float | None, cutoff: Any, flags: list[str],
) -> dict[str, Any] | None:
    """Read a frozen ``PayoffSurfaceArtifact``'s coefficients/residuals -- never
    fits (P5-4). Same full-key check as ``_line_fit_from_artifact``."""
    from engine.v2.models.payoff_artifact import PayoffSurfaceArtifact

    if _artifact_key_mismatch(artifact, PayoffSurfaceArtifact, name, alpha, cutoff):
        _add_flag(flags, "MODEL_NOT_READY")
        return None
    return {
        "coefficients": artifact.coefficients,
        "resid_sd": artifact.resid_sd,
        "n": artifact.n,
        "r": artifact.r,
        "residuals": np.asarray(artifact.residuals, dtype=float),
    }


def _driver_pool(
    block: Mapping[str, Any], slot: str, rows_field: str, prediction: float,
    flags: list[str],
) -> np.ndarray | None:
    """One driver's residual pool: the frozen artifact when the bundle
    declared one (P5-4 -- a key-checked lookup, MODEL_NOT_READY on a
    missing/mismatched artifact, never a rebuild from rows), else the
    request-supplied rows bucketed exactly as before. ``None`` means a flag
    was already added."""
    from engine.v2.scoring import native_payoff, native_residuals

    if native_residuals.RESIDUAL_ARTIFACTS_FIELD in block:
        pool, flag = native_residuals.driver_pool_from_artifact(block, slot, prediction)
        if flag is not None:
            _add_flag(flags, flag)
        return pool
    residual_recipe = block.get("model_residual_recipe") or {}
    pool = native_payoff.driver_residual_pool(
        block.get(rows_field), prediction,
        deciles=int(residual_recipe.get("deciles", native_payoff.DECILES)),
        min_pool=int(residual_recipe.get("min_pool", native_payoff.MIN_POOL)),
    )
    if pool is None or pool.size == 0:
        _add_flag(flags, "MISSING_MODEL_RESIDUALS")
        return None
    return pool


def _model_fit_and_pool(
    block: Mapping[str, Any],
    recipe: Mapping[str, Any],
    driver: float,
    name: str,
    values: Mapping[str, Any],
    flags: list[str],
) -> tuple[dict[str, Any], np.ndarray] | None:
    """The causal payoff line (frozen artifact or inline fit) and the
    driver's own residual pool.

    ``block["payoff_artifact"]`` present (even if ``None``) means the bundle
    declared the P5-4 frozen-artifact path (``source_inputs._model_block``);
    its absence means the source-rows compatibility path applies unchanged.
    ``None`` return means a flag (NO_PAYOFF_MAP, MODEL_NOT_READY or
    MISSING_MODEL_RESIDUALS) was already added.
    """
    from engine.v2.scoring import native_payoff

    if "payoff_artifact" in block:
        alpha = _finite(values.get("fill"))
        fit = _line_fit_from_artifact(
            block.get("payoff_artifact"), name, alpha, recipe.get("before"), flags,
        )
    else:
        rows = block.get("payoff_source_rows")
        if not isinstance(rows, (list, tuple)):
            _add_flag(flags, "NO_PAYOFF_MAP")
            return None
        fit = native_payoff.fit_payoff_line(
            rows,
            before=recipe.get("before"),
            min_trades=int(recipe.get("min_trades", native_payoff.MIN_TRADES)),
            max_residuals=int(recipe.get("max_residuals", native_payoff.MAX_RESIDUALS)),
            residual_seed=int(recipe.get("residual_seed", native_payoff.RESIDUAL_SEED)),
        )
        if fit is None:
            _add_flag(flags, "NO_PAYOFF_MAP")
    if fit is None:
        return None
    driver_pool = _driver_pool(block, "driver", "model_residual_rows", driver, flags)
    if driver_pool is None:
        return None
    return fit, driver_pool


def _runup_model_inputs(
    values: Mapping[str, Any],
    facts: Mapping[str, Any],
    flags: list[str],
) -> tuple[float, float, float, float, float, float] | None:
    """Resolve the two drivers, spot, strike, entry_cost and horizon a
    RunupPayoffSurface would be pushed through, or add the refusing flag
    and return ``None`` -- mirrors ``_model_driver_and_cost`` for the
    two-driver case (engine/score.py:2358-2413).
    """
    point_implied = _finite(values.get("driver_prediction"))
    point_move_d14 = _finite(values.get("runup_move_prediction"))
    days = _finite(facts.get("days_before_print"))
    if point_implied is None:
        _add_flag(flags, "MISSING_MODEL_INPUT:driver_prediction")
        return None
    if point_move_d14 is None:
        _add_flag(flags, "MISSING_MODEL_INPUT:runup_move_prediction")
        return None
    if days is None or days < 0.0:
        # engine/score.py:2363-2366 -- invalid days_before_print refuses
        # before the model ever runs.
        _add_flag(flags, "MISSING_MODEL_INPUT:days_before_print")
        return None
    spot = _finite(values.get("spot"))
    strike = _finite(values.get("strike"))
    cost = _finite(values.get("entry_cost"))
    if spot is None or strike is None or cost is None or cost <= 0.0:
        # engine/score.py:2405-2413's own guard -- no chain, no denominator,
        # no resolved strike. A row already refused for the missing chain
        # keeps that reason rather than picking up a second, misleading one.
        if "COARSE_LADDER" not in flags:
            _add_flag(flags, "NO_CHAIN")
        return None
    return point_implied, point_move_d14, spot, strike, cost, days


def _runup_fit_and_pool(
    block: Mapping[str, Any],
    recipe: Mapping[str, Any],
    point_implied: float,
    point_move_d14: float,
    values: Mapping[str, Any],
    flags: list[str],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray] | None:
    """The causal ``RunupPayoffSurface`` (frozen artifact or inline fit) and
    both drivers' own residual pools.

    ``block["payoff_artifact"]`` present (even if ``None``) means the bundle
    declared the P5-4 frozen-artifact path; its absence means the
    source-rows compatibility path applies unchanged. ``None`` return means
    a flag (NO_PAYOFF_MAP, MODEL_NOT_READY or MISSING_MODEL_RESIDUALS) was
    already added.
    """
    from engine.v2.scoring import native_payoff

    if "payoff_artifact" in block:
        alpha = _finite(values.get("fill"))
        fit = _surface_fit_from_artifact(
            block.get("payoff_artifact"), "STR-RUNUP", alpha, recipe.get("before"),
            flags,
        )
    else:
        rows = block.get("payoff_source_rows")
        if not isinstance(rows, (list, tuple)):
            _add_flag(flags, "NO_PAYOFF_MAP")
            return None
        fit = native_payoff.fit_runup_payoff_surface(
            rows,
            before=recipe.get("before"),
            min_trades=int(recipe.get("min_trades", native_payoff.MIN_TRADES)),
            max_residuals=int(recipe.get("max_residuals", native_payoff.MAX_RESIDUALS)),
            residual_seed=int(recipe.get("residual_seed", native_payoff.RESIDUAL_SEED)),
        )
        if fit is None:
            _add_flag(flags, "NO_PAYOFF_MAP")
    if fit is None:
        return None
    implied_pool = _driver_pool(
        block, "driver", "model_residual_rows", point_implied, flags,
    )
    if implied_pool is None:
        return None
    move_pool = _driver_pool(
        block, "runup_move", "runup_move_residual_rows", point_move_d14, flags,
    )
    if move_pool is None:
        return None
    return fit, implied_pool, move_pool


def _execute_runup_model(
    inputs: NativeScoreInputs,
    block: Mapping[str, Any],
    recipe: Mapping[str, Any],
    values: dict[str, Any],
    flags: list[str],
) -> dict[str, Any]:
    """STR-RUNUP's two-driver payoff-calibration/model layer (R4-17).

    Mirrors engine/score.py:2270-2440 (``_score_runup_model``) using
    ``native_payoff``'s pure re-derivation of ``engine/payoff.py``'s
    ``fit_runup_payoff``/``RunupPayoffSurface``/``simulate_runup_returns``.
    ``driver_prediction`` (legacy's ``point_implied``) and
    ``runup_move_prediction`` are read exactly as the forecast stage left
    them -- i.e. at the model's own D14 scale, since nothing upstream of
    this stage applies ``scale_runup_move`` in the native/local path
    (unlike the frozen-inference path's own ``application._runup_frozen_output``,
    which produces the row's PUBLISHED, already-scaled forecast field of the
    same name for a different purpose). This stage applies the scale itself,
    exactly once, when building the Monte Carlo move draws.
    """
    from engine.v2.scoring import native_payoff

    facts = _facts(inputs, values)
    resolved = _runup_model_inputs(values, facts, flags)
    if resolved is None:
        return {}
    point_implied, point_move_d14, spot, strike, cost, days = resolved
    if "recalibration_artifact" in block:
        # Legacy never recalibrates STR-RUNUP (engine/score.py:2508-2513).
        _add_flag(flags, "UNSUPPORTED_RECALIBRATION")
        return {}
    fitted = _runup_fit_and_pool(
        block, recipe, point_implied, point_move_d14, values, flags,
    )
    if fitted is None:
        return {}
    fit, implied_pool, move_pool = fitted
    draws = int(recipe.get("draw_count", native_payoff.MODEL_DRAWS))
    seed = recipe.get("seed")
    seed = int(seed) if seed is not None else _model_seed(inputs, values)
    returns = native_payoff.simulate_runup_model_returns(
        point_implied, point_move_d14, implied_pool, move_pool,
        fit["coefficients"], fit["residuals"], spot, strike, cost, days,
        draws, np.random.default_rng(seed),
    )
    output = {
        "exp_pnl_model": float(np.mean(returns)),
        "win_model": float(np.mean(returns > 0.0)),
        "payoff": _surface_payoff_document(fit),
    }
    values.update(output)
    return output


def _line_payoff_document(fit: Mapping[str, Any]) -> dict[str, Any]:
    """engine/payoff.py:157-167 (``PayoffLine.as_dict``) -- the shape
    ``financial.py::_fair_premium`` reads for the single-driver line case.
    Publishes the fit the model stage already produced instead of a
    separately captured legacy number. Rounded to 8 dp to match
    ``PayoffLine.as_dict`` (engine/payoff.py:157-165): the score record's
    ``fair_premium_pct``/``premium_vs_fair`` are computed from legacy's
    ROUNDED intercept/slope, and the Phase 4 comparator policy
    SCORE_RECORD_V1 is exact, so publishing full precision here would drift
    from legacy's own recorded numbers."""
    return {
        "intercept": round(float(fit["intercept"]), 8),
        "slope": round(float(fit["slope"]), 8),
    }


def _surface_payoff_document(fit: Mapping[str, Any]) -> dict[str, Any]:
    """engine/payoff.py:260-278 (``RunupPayoffSurface.as_dict``) -- the
    shape ``financial.py::_runup_fair_premium`` reads: ``kind`` picks the
    surface branch, ``coefficients`` keyed by RUNUP_TERMS."""
    from engine.v2.scoring import native_payoff

    return {
        "kind": "runup_payoff_surface",
        "coefficients": {
            name: round(float(value), 8)
            for name, value in zip(native_payoff.RUNUP_TERMS, fit["coefficients"])
        },
    }


def _execute_model(
    inputs: NativeScoreInputs,
    name: str,
    values: dict[str, Any],
    flags: list[str],
) -> dict[str, Any]:
    """Payoff-calibration/model layer: exp_pnl_model, win_model.

    Mirrors engine/score.py:2067-2218 (``_score_model``) using
    ``native_payoff``'s pure re-derivation of ``engine/payoff.py``'s
    ``fit_payoff``/``simulate_returns`` and
    ``engine/models/registry.py``'s ``bucket_residuals``/``residual_pool``
    (engine/v2 has no dependency on legacy ``engine/`` code -- see
    ATM_TOLERANCE_PCT above). NO_PAYOFF_MAP fires exactly where legacy fires
    it: a strategy absent from PAYOFF_DRIVER (engine/payoff.py:79-93,
    ``driver_for``), or too few closed trades before the decision cutoff to
    fit a line (engine/payoff.py:332-337, ``PayoffError``). STR-RUNUP
    (R4-17) is dispatched to ``_execute_runup_model``'s two-driver surface
    instead of the single-driver line below.
    """
    from engine.v2.scoring import native_payoff

    block = inputs.model
    recipe = block.get("payoff_recipe")
    if recipe is None:
        if name not in _PAYOFF_DRIVER_STRATEGIES and name not in DISABLED:
            # Legacy's site-1 NO_PAYOFF_MAP (engine/score.py:2799,
            # ``PAYOFF_DRIVER.get(strategy) is None``): a strategy with no
            # registered payoff driver never reaches the rest of the model
            # stage, so capture never records a ``payoff_recipe`` for it
            # either. This must fire from the strategy's identity alone,
            # before any output-anomaly check, exactly as legacy does --
            # never reachable via ``_model_driver_and_cost`` below, since
            # that path requires a recipe to have been declared at all.
            #
            # ``name not in DISABLED``: a SEPARATE, narrower legacy gate.
            # DISABLED_STRATEGIES (engine/score.py:117-127) returns from
            # Scorer.score before site-1 (or _price_entry/_score_model) ever
            # runs, flagging only UNVALIDATED_STRUCTURE. ``DISABLED`` is the
            # same registry domain.generation.structures already uses to
            # refuse geometry for these strategies -- reused by reference,
            # not a new list. Deliberately NOT "geometry refused for any
            # reason": a FORECAST_SIZED strategy whose geometry refuses for
            # an unrelated capture gap still reaches _score_model in legacy
            # and DOES get NO_PAYOFF_MAP there (see the commit that added
            # this line for the measurement).
            _add_flag(flags, "NO_PAYOFF_MAP")
        elif any(block.get(field_name) is not None for field_name in _MODEL_OUTPUTS):
            _add_flag(flags, "UNOWNED_MODEL_OUTPUT")
        return {}
    if not isinstance(recipe, Mapping):
        _add_flag(flags, "INVALID_PAYOFF_RECIPE")
        return {}
    if name == "STR-RUNUP":
        return _execute_runup_model(inputs, block, recipe, values, flags)
    driver_cost = _model_driver_and_cost(name, block, values, flags)
    if driver_cost is None:
        return {}
    driver, spot, cost = driver_cost
    fitted = _model_fit_and_pool(block, recipe, driver, name, values, flags)
    if fitted is None:
        return {}
    fit, driver_pool = fitted
    draws = int(recipe.get("draw_count", native_payoff.MODEL_DRAWS))
    seed = recipe.get("seed")
    seed = int(seed) if seed is not None else _model_seed(inputs, values)
    returns = native_payoff.simulate_model_returns(
        driver, driver_pool, fit["residuals"], fit["intercept"], fit["slope"],
        spot, cost, draws, np.random.default_rng(seed),
    )
    output = {
        "exp_pnl_model": float(np.mean(returns)),
        "win_model": float(np.mean(returns > 0.0)),
        "payoff": _line_payoff_document(fit),
    }
    output = _recalibrated_output(block, recipe, name, values, output, flags)
    values.update(output)
    return output


def _recalibrated_output(
    block: Mapping[str, Any],
    recipe: Mapping[str, Any],
    name: str,
    values: Mapping[str, Any],
    output: dict[str, Any],
    flags: list[str],
) -> dict[str, Any]:
    """Apply a declared frozen recalibration map to ``win_model`` (P5-4).

    Mirrors engine/score.py:2336-2341: ``win_model`` is the raw Monte Carlo
    win rate pushed through ``Scorer.recalibration(strategy, alpha,
    evidence_cutoff)``. Undeclared (no ``recalibration_artifact`` key in the
    block) returns ``output`` untouched -- the path every bundle took before
    this artifact existed. Declared: the artifact must be a
    ``RecalibrationMapArtifact`` whose full causal key (strategy, alpha at
    4dp, the payoff recipe's ``before``) equals this request's, else
    MODEL_NOT_READY and no model numbers. Never fits. Legacy keeps STR-RUNUP
    raw (engine/score.py:2508-2513), so this is only reached from the
    single-driver line path.
    """
    if "recalibration_artifact" not in block:
        return output
    from engine.v2.models.recalibration_artifact import (
        RecalibrationMapArtifact,
        recalibration_artifact_key,
    )

    artifact = block.get("recalibration_artifact")
    alpha = _finite(values.get("fill"))
    if (alpha is None or not isinstance(artifact, RecalibrationMapArtifact)
            or artifact.key != recalibration_artifact_key(name, alpha, recipe.get("before"))):
        _add_flag(flags, "MODEL_NOT_READY")
        return {}
    return {**output, "win_model": artifact.transform(output["win_model"])}


#: The analog block field that declares the frozen board analog matcher.
ANALOG_ARTIFACT_FIELD = "analog_artifact"
ANALOG_ARTIFACT_RECIPE_FIELD = "analog_artifact_recipe"


def _execute_frozen_analogs(
    block: Mapping[str, Any],
    strategy: str | None,
    values: dict[str, Any],
    flags: list[str],
) -> dict[str, Any]:
    from engine.v2.scoring.native_analog import evaluate_frozen_analogs

    if block.get("source_rows") is not None:
        _add_flag(flags, "MODEL_NOT_READY")  # two populations declared: never pick one
        return {}
    try:
        result, flag = evaluate_frozen_analogs(
            artifact=block.get(ANALOG_ARTIFACT_FIELD),
            recipe=block.get(ANALOG_ARTIFACT_RECIPE_FIELD) or {},
            query_features=block.get("query_features"),
            strategy=strategy,
            alpha=_finite(values.get("fill")),
        )
    except (TypeError, ValueError) as exc:
        _add_flag(flags, str(exc))
        return {}
    if flag is not None:
        _add_flag(flags, flag)
        return {}
    if result.thin:
        _add_flag(flags, "THIN_ANALOGS")
    output = {
        "exp_pnl_analog": result.exp_pnl_analog,
        "win_analog": result.win_analog,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "n_analogs": result.n_analogs,
    }
    values.update(output)
    return output


def _analog_identity(block: Mapping[str, Any]) -> dict[str, Any] | None:
    """The frozen analog artifact's identity for the stage receipt, or
    ``None`` on the undeclared/declared-rows paths (receipts unchanged)."""
    if ANALOG_ARTIFACT_FIELD not in block:
        return None
    artifact = block.get(ANALOG_ARTIFACT_FIELD)
    return {"artifact": None if artifact is None else str(artifact),
            "recipe": dict(block.get(ANALOG_ARTIFACT_RECIPE_FIELD) or {}),
            "query_features": dict(block.get("query_features") or {})}


def _execute_analogs(
    inputs: NativeScoreInputs,
    values: dict[str, Any],
    flags: list[str],
    strategy: str | None = None,
) -> dict[str, Any]:
    """Calculate analog summaries from a hash-bound source population.

    A block carrying ``analog_artifact`` (even ``None``) declared the P5-4
    frozen board analog matcher (``source_inputs._analog_block``): the pool
    is read from that artifact after a full causal-key check against this
    request's strategy, fill alpha and evidence cutoff, and a missing or
    mismatched artifact is MODEL_NOT_READY -- never a rebuild, never the
    declared-rows path below, which stays the compatibility path.
    """
    block = inputs.analogs
    if block.get("mode") == "not_applicable":
        return {}
    if ANALOG_ARTIFACT_FIELD in block:
        return _execute_frozen_analogs(block, strategy, values, flags)
    recipe = block.get("recipe")
    source_rows = block.get("source_rows")
    query_features = block.get("query_features")
    if recipe is None and source_rows is None and query_features is None:
        # Genuinely not-applicable: no recipe and no inputs at all. This row
        # never asked for an analog calculation, so silence here is correct
        # and unchanged.
        if any(block.get(field) is not None for field in _ANALOG_OUTPUTS):
            _add_flag(flags, "UNOWNED_ANALOG_OUTPUT")
        return {}
    # From here a recipe (or a stray input) is present: the row DID ask for
    # an analog calculation. Missing required inputs must be reported, not
    # silently skipped, so every remaining path below flags before returning.
    if not isinstance(recipe, Mapping) or not isinstance(source_rows, (list, tuple)):
        _add_flag(flags, "MISSING_ANALOG_INPUT")
        return {}
    if not isinstance(query_features, Mapping):
        _add_flag(flags, "MISSING_ANALOG_INPUT")
        return {}
    try:
        from engine.v2.scoring.native_analog import evaluate_analogs

        result = evaluate_analogs(
            source_rows=source_rows,
            query_features=query_features,
            recipe=recipe,
        )
    except (TypeError, ValueError) as exc:
        _add_flag(flags, str(exc))
        return {}
    if result.thin:
        _add_flag(flags, "THIN_ANALOGS")
    output = {
        "exp_pnl_analog": result.exp_pnl_analog,
        "win_analog": result.win_analog,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "n_analogs": result.n_analogs,
    }
    values.update(output)
    return output


def _gate_result(score: float | None, threshold: Any,
                 flags: list[str]) -> dict[str, Any]:
    if score is None:
        _add_flag(flags, "INVALID_GATE_SCORE")
        return {}
    value = _finite(threshold)
    if value is None:
        _add_flag(flags, "MISSING_GATE_THRESHOLD")
        return {}
    return {
        "gate_score": score,
        "gate_threshold": value,
        "gate_pass": score >= value,
    }


def _gate_facts(inputs: NativeScoreInputs, values: Mapping[str, Any],
                block: Mapping[str, Any], requested: tuple[str, ...] | None,
                flags: list[str]) -> dict[str, Any] | None:
    """The gate's inputs: ``_facts`` plus the derived columns a gate names
    (R4-20 gap 3; engine/score.py ``Scorer._gate_feature_frame``).

    Like legacy, a GROUP (the five forecast columns, the three analog
    columns) is derived when the gate names at least one of its columns that
    the base feature vector does not carry -- and then legacy writes EVERY
    column of the group, overwriting a base value of the same name. So does
    this. ``requested`` is the gate's feature list, or ``None`` when the
    executor cannot say (then the analog group is derived, and the forecast
    group when declared).
    ``None`` is returned after flagging when a named forecast column has no
    declared source, or the forecast binding cannot be served.
    """
    from engine.v2.scoring import native_gate_features as derived

    facts = _facts(inputs, values)
    base = inputs.features.get("model_inputs")
    base = base if isinstance(base, Mapping) else {}

    def wanted(names: tuple[str, ...]) -> list[str]:
        return [name for name in names if name not in base
                and (requested is None or name in requested)]

    if wanted(derived.GATE_ANALOG_COLUMNS):
        facts.update(derived.gate_analog_columns(values))
    forecast = wanted(derived.GATE_FORECAST_COLUMNS)
    executor = block.get("forecast_executor")
    if not forecast or (requested is None and executor is None):
        return facts
    if executor is None:
        _add_flag(flags, "MISSING_GATE_INPUT:forecast")
        return None
    pool = block.get("forecast_pool")
    band = {"pred_abs_move_p10", "pred_abs_move_p90", "pred_abs_move_sd"}
    if pool is None and requested is not None and band & set(requested):
        _add_flag(flags, "MISSING_GATE_INPUT:forecast_pool")
        return None
    try:
        served = _finite(executor.predict(facts).get("pred_abs_move"))
    except (TypeError, ValueError, KeyError) as exc:
        if not _missing_feature_refusal(exc):
            _add_executor_refusal(flags, exc, "INVALID_GATE_FORECAST")
            return None
        # A fold row with a missing/non-finite feature serves NaN (legacy
        # ``ServingModel.predict``): every forecast column stays NaN.
        served = None
    facts.update(derived.gate_forecast_columns(served, pool, facts.get("im")))
    return facts


def _execute_gate_executor(inputs: NativeScoreInputs,
                           values: Mapping[str, Any],
                           block: Mapping[str, Any],
                           flags: list[str]) -> dict[str, Any]:
    executors = block.get("executors")
    if not isinstance(executors, Mapping):
        _add_flag(flags, "INVALID_GATE_EXECUTORS")
        return {}
    executor = executors.get("gate_score")
    if executor is None or not callable(getattr(executor, "predict", None)):
        _add_flag(flags, "INVALID_GATE_EXECUTOR")
        return {}
    order = getattr(executor, "feature_order", None)
    requested = tuple(order) if order else None
    facts = _gate_facts(inputs, values, block, requested, flags)
    if facts is None:
        return {}
    try:
        result = executor.predict(facts)
        if isinstance(result, Mapping):
            score = result.get("gate_score")
            if score is None and len(result) == 1:
                score = next(iter(result.values()))
        else:
            score = result
        score = _finite(score)
    except (TypeError, ValueError, KeyError) as exc:
        _add_executor_refusal(flags, exc, "INVALID_GATE_EXECUTOR")
        return {}
    return _gate_result(score, block.get("threshold"), flags)


def _execute_gate_model(inputs: NativeScoreInputs,
                        values: Mapping[str, Any],
                        block: Mapping[str, Any],
                        flags: list[str]) -> dict[str, Any]:
    model = block.get("model")
    if not isinstance(model, Mapping):
        _add_flag(flags, "INVALID_GATE_MODEL")
        return {}
    coefficients = model.get("coefficients")
    requested = tuple(coefficients) if isinstance(coefficients, Mapping) else ()
    facts = _gate_facts(inputs, values, block, requested, flags)
    if facts is None:
        return {}
    try:
        score = _linear(model, facts, "gate")
    except ValueError as exc:
        _add_flag(flags, str(exc))
        return {}
    return _gate_result(score, block.get("threshold"), flags)


def _execute_gate(inputs: NativeScoreInputs, name: str,
                  values: dict[str, Any], flags: list[str]) -> dict[str, Any]:
    block = inputs.gate
    if block.get("mode") == "not_applicable":
        return {}
    if block.get("mode") == "entry_rule":
        # Legacy ``_apply_entry_rule``: reached with no gate champion, before
        # (and without) the model gate's domain check. An unpriced row never
        # reaches legacy's gate, so it gets no verdict and no flag here.
        from engine.v2.scoring.native_entry_rule import execute_entry_rule

        if values.get("entry_cost") is None:
            return {}

        rule_flags: list[str] = []
        output = execute_entry_rule(
            block, strategy=name, event_date=inputs.context.get("event_date"),
            legs=values.get("legs") or (), exp_pnl_sim=values.get("exp_pnl_sim"),
            flags=rule_flags)
        for flag in rule_flags:
            _add_flag(flags, flag)
        values.update(output)
        return output
    if block.get("frozen_score") is not None:
        _add_flag(flags, "UNSUPPORTED_FROZEN_GATE")
        return {}
    if not _gate_in_domain(inputs, values):
        _add_flag(flags, "OUT_OF_DOMAIN")
        return {}
    if block.get("executors") is not None:
        output = _execute_gate_executor(inputs, values, block, flags)
    elif block.get("model") is not None:
        output = _execute_gate_model(inputs, values, block, flags)
    elif block.get("recipe") is not None:
        _add_flag(flags, f"UNSUPPORTED_GATE_RECIPE:{block['recipe']}")
        return {}
    else:
        output = {}
        for field in _GATE_OUTPUTS:
            if block.get(field) is not None:
                _add_flag(flags, f"UNOWNED_GATE_OUTPUT:{field}")
        if not any(block.get(field) is not None for field in _GATE_OUTPUTS):
            _add_flag(flags, "MISSING_GATE_INPUT")
            return output
    values.update(output)
    return output


def _execute_chooser(inputs: NativeScoreInputs, name: str,
                     values: dict[str, Any], flags: list[str]) -> dict[str, Any]:
    """The DYN-SV chooser champion over this candidate (R4-20 gap 5).

    engine/score.py ``Scorer._score_chooser``: the champion's features are
    read by name (``_chooser_frame``); any missing or non-finite one
    declines the ranking with the ADVISORY ``CHOOSER_MISSING_FEATURES`` --
    the row stays scored, only ``chooser_score`` is absent. Otherwise
    ``chooser_score`` is the frozen binding's prediction (no refit). The
    columns ``_chooser_frame`` computes are derived by ``native_chooser``
    from the pass and the block's declared frozen state; the primitive ones
    come from the declared feature vector, which also wins for any derived
    column it declares (the compatibility path).
    """
    from engine.v2.scoring.native_chooser import derive_chooser_columns

    block = inputs.chooser
    executors = block.get("executors")
    if executors is None:
        output = {key: value for key, value in block.items()
                  if key not in _OWNED_OUTPUTS and key != "flags"}
        values.update(output)
        return output
    executor = executors.get("chooser_score") if isinstance(executors, Mapping) else None
    if executor is None or not callable(getattr(executor, "predict", None)):
        _add_flag(flags, "INVALID_CHOOSER_EXECUTOR")
        return {}
    facts = _facts(inputs, values)
    base = inputs.features.get("model_inputs")
    base = base if isinstance(base, Mapping) else {}
    derived = derive_chooser_columns(block, facts, name, values,
                                     _quote_map(inputs, name), flags)
    if derived is None:
        return {}
    facts.update({key: value for key, value in derived.items() if key not in base})
    try:
        score = _finite(executor.predict(facts).get("chooser_score"))
    except (TypeError, ValueError, KeyError) as exc:
        if _missing_feature_refusal(exc):
            _add_flag(flags, "CHOOSER_MISSING_FEATURES")
        else:
            _add_executor_refusal(flags, exc, "INVALID_CHOOSER_EXECUTOR")
        return {}
    if score is None:
        _add_flag(flags, "CHOOSER_MISSING_FEATURES")
        return {}
    output = {"chooser_score": score}
    values.update(output)
    return output


def _simulation_identity(block: Mapping[str, Any]) -> Mapping[str, Any]:
    """The simulation block as documented in its receipt: a frozen artifact
    appears as its ``schema:content_hash`` identity, not its rows (P5-4)."""
    from engine.v2.scoring.native_residuals import identity_view

    return identity_view(block)


def _withhold_bad_quote_stages(
    inputs: NativeScoreInputs,
    values: dict[str, Any],
    executed: list[StageReceipt],
    observer: StageObserver | None,
) -> None:
    """Legacy's BAD_QUOTE exit: no model, analog, gate or chooser numbers,
    and none of the forecasts only those layers serve (every role but the
    pre-pricing ``size``). The row keeps its pricing and features."""
    for field in _BAD_QUOTE_WITHHELD_FIELDS:
        values.pop(field, None)
    for stage in ("model", "analogs", "simulation", "gate", "chooser"):
        _emit_stage(
            executed, stage, {"prior": executed[-1].output_hash},
            dict(_BAD_QUOTE_WITHHELD), observer,
        )
    _emit_stage(
        executed, "diagnostics", {"prior": executed[-1].output_hash},
        _merge_diagnostics(values, inputs.diagnostics), observer,
    )


def _append_late_stages(
    inputs: NativeScoreInputs,
    values: dict[str, Any],
    executed: list[StageReceipt],
    geometry: Geometry,
    pricing: Pricing,
    compatibility: bool,
    flags: list[str],
    observer: StageObserver | None,
) -> None:
    blocks = (inputs.context, inputs.features, inputs.forecast, inputs.model,
              inputs.analogs, inputs.simulation, inputs.gate, inputs.chooser,
              inputs.diagnostics)
    if compatibility:
        for stage, block in (
            ("model", inputs.model), ("analogs", inputs.analogs),
            ("simulation", inputs.simulation),
            ("gate", inputs.gate), ("chooser", inputs.chooser),
            ("diagnostics", inputs.diagnostics),
        ):
            values.update(block)
            _emit_stage(
                executed, stage, {"prior": executed[-1].output_hash}, block,
                observer,
            )
    elif "BAD_QUOTE" in flags:
        _withhold_bad_quote_stages(inputs, values, executed, observer)
    else:
        model_output = _execute_model(inputs, geometry.strategy, values, flags)
        _emit_stage(
            executed, "model", {"prior": executed[-1].output_hash},
            model_output, observer,
        )
        analog_output = _execute_analogs(inputs, values, flags, geometry.strategy)
        analog_inputs: dict[str, Any] = {"prior": executed[-1].output_hash}
        analog_identity = _analog_identity(inputs.analogs)
        if analog_identity is not None:
            analog_inputs["inputs"] = analog_identity
        _emit_stage(executed, "analogs", analog_inputs, analog_output, observer)
        simulation = _execute_simulation(
            inputs, values, geometry, pricing, flags,
        )
        _emit_stage(
            executed, "simulation",
            {"prior": executed[-1].output_hash,
             "inputs": _simulation_identity(inputs.simulation),
             "entry_cost": pricing.entry_cost},
            simulation, observer,
        )
        gate = _execute_gate(inputs, geometry.strategy, values, flags)
        _emit_stage(
            executed, "gate",
            {"prior": executed[-1].output_hash,
             "inputs": {key: value for key, value in inputs.gate.items()
                        if key not in _INTERNAL_STAGE_FIELDS},
             "simulation": simulation},
            gate, observer,
        )
        chooser = _execute_chooser(inputs, geometry.strategy, values, flags)
        chooser_inputs: dict[str, Any] = {"prior": executed[-1].output_hash}
        if isinstance(inputs.chooser.get("executors"), Mapping):
            # A frozen chooser recipe is identified by its binding (the
            # executor's str()), exactly like the gate's.
            from engine.v2.scoring.native_chooser import identity_view

            chooser_inputs["inputs"] = {
                "binding_id": inputs.chooser.get("binding_id"),
                "executors": {key: str(value) for key, value
                              in inputs.chooser["executors"].items()},
                **identity_view(inputs.chooser),
            }
        _emit_stage(executed, "chooser", chooser_inputs, chooser, observer)
        _emit_stage(
            executed, "diagnostics", {"prior": executed[-1].output_hash},
            _merge_diagnostics(values, inputs.diagnostics), observer,
        )
    for block in blocks:
        for item in block.get("flags") or ():
            _add_flag(flags, item)


def assemble_native_values(inputs: NativeScoreInputs, *, strategy: str | None = None,
                           fill_model: Mapping[str, Any] | None = None,
                           observer: StageObserver | None = None) -> dict[str, Any]:
    """Execute declared stages and return execution-owned values."""
    if not isinstance(inputs, NativeScoreInputs):
        raise TypeError("native stage assembly requires NativeScoreInputs")
    is_compatibility = inputs.source_ref == "compatibility-input"
    flags: list[str] = []
    values, executed, legacy_cost = _initial_values(
        inputs, flags, is_compatibility, strategy, observer,
    )
    _check_projected_calendar(values, flags)
    _check_stale_quote(values, flags)
    name = _strategy_name(inputs, strategy, values)
    geometry_inputs, geometry = _resolve_geometry(inputs, name, values)
    _emit_stage(executed, "geometry", geometry_inputs, geometry, observer)
    alpha = _finite((fill_model or {}).get("alpha", 0.5))
    if alpha is None or not 0.0 <= alpha <= 1.0:
        alpha = 0.5
        _add_flag(flags, "INVALID_FILL_ALPHA")
    quotes = _quote_map(inputs, name)
    pricing = _resolve_pricing(
        inputs, name, geometry, quotes, alpha, legacy_cost,
    )
    _emit_stage(
        executed, "pricing",
        {"geometry": geometry, "quotes": quotes, "fill_alpha": alpha},
        pricing, observer,
    )
    _publish_pricing(values, geometry, pricing, alpha)
    # R4-20 gap 2: the STR-THRU/STR-RUNUP gate champions name
    # ``entry_cost_pct`` in their frozen ``feature_order``, but legacy
    # computes it in ``Scorer._features`` (engine/score.py:1934,
    # ~2562-2564) BEFORE ``Scorer._score_gate`` (engine/score.py:2030).
    # Native used to only produce it post-hoc in
    # ``financial.financial_diagnostics`` (application.py:174), long after
    # the gate stage in ``_append_late_stages`` has already run -- so the
    # gate's frozen executor always raised MISSING_FEATURES. Publish it here,
    # from the same shared formula ``financial.entry_cost_pct`` uses, so it
    # is in ``values`` (and therefore ``_facts``) before the gate executes.
    values["entry_cost_pct"] = financial.entry_cost_pct(
        values.get("entry_cost"), values.get("spot"),
    )
    _check_wide_market(pricing, flags)
    _check_bad_quote(pricing, flags)
    _check_extrapolated(geometry, pricing, flags)
    _append_late_stages(
        inputs, values, executed, geometry, pricing, is_compatibility, flags,
        observer,
    )
    for refusal in (geometry.refusal, pricing.refusal):
        if refusal:
            _add_flag(flags, refusal)
    _publish_pricing(values, geometry, pricing, alpha)
    values.update({"flags": tuple(flags),
                   "native_source_ref": inputs.source_ref})
    _emit_stage(executed, "serialization", values, values, observer)
    values["native_stage_receipts"] = tuple({"stage": item.stage,
        "input_hash": item.input_hash, "output_hash": item.output_hash,
        "owner": item.owner} for item in executed)
    return values


__all__ = [
    "ADVISORY_FLAGS", "NativeScoreInputs", "STAGE_NAMES", "StageObservation",
    "StageObserver", "StageReceipt",
    "assemble_native_values", "flags_refuse", "receipt",
]
