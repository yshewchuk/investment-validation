"""Explicit inputs and receipts for the native scoring execution graph."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import numpy as np
from scipy.stats import norm

from engine.v2.domain.generation import Geometry, Pricing, generate, price
from engine.v2.domain.valuation import terminal_payoff
from engine.v2.foundation import content_hash

STAGE_NAMES = (
    "resolve_context", "features", "forecast", "geometry", "pricing",
    "analogs", "simulation", "gate", "chooser", "diagnostics", "serialization",
)


@dataclass(frozen=True)
class StageReceipt:
    stage: str
    input_hash: str
    output_hash: str
    owner: str = "engine.v2.scoring"


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

    def __post_init__(self) -> None:
        if not self.source_ref.strip():
            raise ValueError("native score inputs require source_ref")
        stages = tuple(receipt.stage for receipt in self.stage_receipts)
        seen = set(stages)
        required = set(STAGE_NAMES) - {"diagnostics"}
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
                                             "analogs", "simulation", "gate",
                                             "chooser", "diagnostics")}
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
    | _GATE_OUTPUTS | _ANALOG_OUTPUTS
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
_INTERNAL_STAGE_FIELDS = frozenset({"executors"})
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


def _merge_stage(values: dict[str, Any], block: Mapping[str, Any]) -> None:
    values.update({key: value for key, value in block.items()
                   if key not in _OWNED_OUTPUTS
                   and key not in _INTERNAL_STAGE_FIELDS
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


def _facts(inputs: NativeScoreInputs, values: Mapping[str, Any]) -> dict[str, Any]:
    facts = dict(inputs.context)
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
    planned = (
        inputs.simulation.get("mode") == "planned_exit"
        or inputs.simulation.get("residuals") is not None
    )
    if planned:
        for role in ("size", "iv_crush"):
            if role not in roles:
                roles.append(role)
    return tuple(roles)


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


def _execute_forecast_executor(
    executor, field, facts, output, invalid_fields, flags,
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
        if not tuple(getattr(exc, "reason_codes", ()) or ()):
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
    for field in _FORECAST_OUTPUTS:
        executor = executors.get(field)
        if executor is not None:
            declared = True
            _execute_forecast_executor(
                executor, field, facts, output, invalid_fields, flags,
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
    frozen_declared = _execute_frozen_forecast(
        block, output, invalid_fields, flags,
    )
    local_declared = _execute_local_forecast(
        inputs, block, values, output, invalid_fields, flags,
    )
    _validate_forecast_roles(
        inputs, output, flags, strategy, invalid_fields,
    )
    if not (frozen_declared or local_declared):
        _add_flag(flags, "MISSING_FORECAST_INPUT")
    values.update(output)
    return output


def _initial_values(
    inputs: NativeScoreInputs,
    flags: list[str],
    compatibility: bool,
    strategy: str | None,
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
        executed.append(receipt(stage, prior, output))
        prior = {"prior": executed[-1].output_hash, "values": values}
    if compatibility:
        values.update(inputs.forecast)
        forecast_output = dict(inputs.forecast)
    else:
        forecast_output = _execute_forecast(
            inputs, values, flags, strategy,
        )
    executed.append(receipt("forecast", prior, forecast_output))
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
                and geometry_inputs.get("post_event_expiry") is None):
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
        if np.isnat(event_date) or not all(
            isfinite(value) for value in (prediction, move, crush)
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


def _draw_residuals(block: Mapping[str, Any], event_date: Any,
                    prediction: float, key: str,
                    flags: list[str]) -> tuple[Any, ...] | None:
    dates, predicted, move, crush = _residual_arrays(block.get("residuals"))
    try:
        cutoff = np.datetime64(str(event_date))
    except ValueError:
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
        hashlib.sha256(f"{key}|{event_date}".encode()).digest()[:8], "big",
    )
    rng = np.random.default_rng(seed)
    chosen = rows[rng.integers(0, rows.size, size=draws)]
    return move[chosen], crush[chosen], rng, end


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
    event_date = block.get("event_date", values.get("event_date"))
    if event_date is None:
        missing.append("event_date")
    if block.get("residuals") is None:
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
    output: dict[str, Any] = {}
    planned = (
        block.get("mode") == "planned_exit"
        or block.get("residuals") is not None
    )
    if planned:
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


def _execute_analogs(
    inputs: NativeScoreInputs,
    values: dict[str, Any],
    flags: list[str],
) -> dict[str, Any]:
    """Calculate analog summaries from a hash-bound source population."""
    block = inputs.analogs
    recipe = block.get("recipe")
    source_rows = block.get("source_rows")
    query_features = block.get("query_features")
    if recipe is None and source_rows is None and query_features is None:
        if any(block.get(field) is not None for field in _ANALOG_OUTPUTS):
            _add_flag(flags, "UNOWNED_ANALOG_OUTPUT")
        return {}
    if source_rows is None and query_features is None:
        return {}
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
    try:
        result = executor.predict(_facts(inputs, values))
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
    try:
        score = _linear(model, _facts(inputs, values), "gate")
    except ValueError as exc:
        _add_flag(flags, str(exc))
        return {}
    return _gate_result(score, block.get("threshold"), flags)


def _execute_gate(inputs: NativeScoreInputs, name: str,
                  values: dict[str, Any], flags: list[str]) -> dict[str, Any]:
    block = inputs.gate
    if block.get("frozen_score") is not None:
        _add_flag(flags, "UNSUPPORTED_FROZEN_GATE")
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


def _append_late_stages(
    inputs: NativeScoreInputs,
    values: dict[str, Any],
    executed: list[StageReceipt],
    geometry: Geometry,
    pricing: Pricing,
    compatibility: bool,
    flags: list[str],
) -> None:
    blocks = (inputs.context, inputs.features, inputs.forecast, inputs.analogs,
              inputs.simulation, inputs.gate, inputs.chooser, inputs.diagnostics)
    if compatibility:
        for stage, block in (
            ("analogs", inputs.analogs), ("simulation", inputs.simulation),
            ("gate", inputs.gate), ("chooser", inputs.chooser),
            ("diagnostics", inputs.diagnostics),
        ):
            values.update(block)
            executed.append(receipt(
                stage, {"prior": executed[-1].output_hash}, block,
            ))
    else:
        analog_output = _execute_analogs(inputs, values, flags)
        executed.append(receipt(
            "analogs", {"prior": executed[-1].output_hash}, analog_output,
        ))
        simulation = _execute_simulation(
            inputs, values, geometry, pricing, flags,
        )
        executed.append(receipt(
            "simulation",
            {"prior": executed[-1].output_hash, "inputs": inputs.simulation,
             "entry_cost": pricing.entry_cost},
            simulation,
        ))
        gate = _execute_gate(inputs, geometry.strategy, values, flags)
        executed.append(receipt(
            "gate",
            {"prior": executed[-1].output_hash,
             "inputs": {key: value for key, value in inputs.gate.items()
                        if key not in _INTERNAL_STAGE_FIELDS},
             "simulation": simulation},
            gate,
        ))
        for stage, block in (
            ("chooser", inputs.chooser), ("diagnostics", inputs.diagnostics),
        ):
            output = (
                _merge_diagnostics(values, block)
                if stage == "diagnostics"
                else {
                    key: value for key, value in block.items()
                    if key not in _OWNED_OUTPUTS and key != "flags"
                }
            )
            if stage != "diagnostics":
                values.update(output)
            executed.append(receipt(
                stage, {"prior": executed[-1].output_hash}, output,
            ))
    for block in blocks:
        for item in block.get("flags") or ():
            _add_flag(flags, item)


def assemble_native_values(inputs: NativeScoreInputs, *, strategy: str | None = None,
                           fill_model: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Execute declared stages and return execution-owned values."""
    if not isinstance(inputs, NativeScoreInputs):
        raise TypeError("native stage assembly requires NativeScoreInputs")
    is_compatibility = inputs.source_ref == "compatibility-input"
    flags: list[str] = []
    values, executed, legacy_cost = _initial_values(
        inputs, flags, is_compatibility, strategy,
    )
    name = _strategy_name(inputs, strategy, values)
    geometry_inputs, geometry = _resolve_geometry(inputs, name, values)
    executed.append(receipt("geometry", geometry_inputs, geometry))
    alpha = _finite((fill_model or {}).get("alpha", 0.5))
    if alpha is None or not 0.0 <= alpha <= 1.0:
        alpha = 0.5
        _add_flag(flags, "INVALID_FILL_ALPHA")
    quotes = _quote_map(inputs, name)
    pricing = _resolve_pricing(
        inputs, name, geometry, quotes, alpha, legacy_cost,
    )
    executed.append(receipt("pricing", {"geometry": geometry, "quotes": quotes,
                                         "fill_alpha": alpha}, pricing))
    _publish_pricing(values, geometry, pricing, alpha)
    _append_late_stages(
        inputs, values, executed, geometry, pricing, is_compatibility, flags,
    )
    for refusal in (geometry.refusal, pricing.refusal):
        if refusal:
            _add_flag(flags, refusal)
    _publish_pricing(values, geometry, pricing, alpha)
    values.update({"flags": tuple(flags),
                   "native_source_ref": inputs.source_ref})
    executed.append(receipt("serialization", values, values))
    values["native_stage_receipts"] = tuple({"stage": item.stage,
        "input_hash": item.input_hash, "output_hash": item.output_hash,
        "owner": item.owner} for item in executed)
    return values


__all__ = [
    "NativeScoreInputs", "STAGE_NAMES", "StageReceipt",
    "assemble_native_values", "receipt",
]
