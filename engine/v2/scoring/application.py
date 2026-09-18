"""Shared single, event and batch Phase 4 scoring application."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from math import isfinite
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from engine.v2.contracts import ReplayReceipt, ScoreBatch, ScoreRecord, ScoreRequest
from engine.v2.features import default_feature_registry
from engine.v2.foundation import from_document, to_document
from engine.v2.registry import DYNAMIC_MENU, default_registry

from .financial import financial_diagnostics
from .frozen_executor import FrozenStageExecutor
from .identity import dependency_hash, request_hash, with_score_id
from .stages import NativeScoreInputs, StageObserver, assemble_native_values, flags_refuse

__all__ = ["replay", "score_batch", "score_event", "score_frozen", "score_many", "score_one"]

_FEATURE_REGISTRY = default_feature_registry()
_FROZEN_ROLE_OUTPUTS = {
    "driver": ("driver_prediction",),
    "size": ("forecast_abs_move",),
    "implied_t1": ("driver_prediction",),
    "runup_move": ("runup_move_prediction",),
    "iv_crush": ("pred_iv_crush", "pred_iv_crush_30"),
    "fair_value": ("model_fair_pct",),
}
_FROZEN_OUTPUTS = frozenset(
    output
    for outputs in _FROZEN_ROLE_OUTPUTS.values()
    for output in outputs
)
_RUNUP_BASE_DAYS = 14.0
_RUNUP_RAW_NAMES = (
    "prediction", "pred_runup_abs_move_d14", "runup_move_d14",
    "runup_move_raw_d14",
)
_RUNUP_FINAL_NAMES = frozenset({
    "runup_move_prediction", "runup_move_p10", "runup_move_p90",
    "runup_move_sd",
})
_RUNUP_DERIVED_FIELDS = frozenset({
    "runup_move_raw_d14", "runup_move_raw_d14_p10",
    "runup_move_raw_d14_p90", "runup_move_raw_d14_sd",
    "runup_move_prediction", "runup_move_p10", "runup_move_p90",
    "runup_move_sd", "runup_move_days", "runup_move_scale",
    "runup_move_provenance",
})


class _CanonicalFrozenExecutor:
    """Map one verified artifact output into its canonical stage field."""

    def __init__(self, executor, source, target, *, scale=1.0, floor_zero=False):
        self._executor = executor
        self._source = source
        self._target = target
        self._scale = scale
        self._floor_zero = floor_zero

    def predict(self, features):
        outputs = self._executor.predict(features)
        value = float(outputs[self._source])
        if self._floor_zero:
            value = max(value, 0.0)
        return {self._target: value * self._scale}


def _runup_executor_spec(names, days):
    if any(name in _RUNUP_FINAL_NAMES for name in names):
        return ()
    source = next((name for name in _RUNUP_RAW_NAMES if name in names), None)
    source = source or (names[0] if len(names) == 1 else None)
    if source is None or days is None or days < 0.0:
        return ()
    return (("runup_move_prediction", source, days / _RUNUP_BASE_DAYS, True),)


def _ordinary_executor_specs(names, targets):
    specs = []
    for target in targets:
        source = target if target in names else None
        if source is None and len(names) == 1 and target == targets[0]:
            source = names[0]
        if source is not None:
            specs.append((target, source, 1.0, False))
    return tuple(specs)


def _canonical_executor_specs(binding, days):
    """Return canonical target, artifact source and transform declarations."""
    role = str(getattr(binding, "role", "")).split(":", 1)[0]
    names = tuple(getattr(binding, "output_names", ()) or ())
    if role == "runup_move":
        return _runup_executor_spec(names, days)
    return _ordinary_executor_specs(names, _FROZEN_ROLE_OUTPUTS.get(role, ()))


def _value_fields(values: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: values.get(name) for name in names}


def _feature_fields(values: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    features = dict(values.get("model_inputs") or {})
    return features, {name: value is None for name, value in features.items()}


def _chooser_selection(values: Mapping[str, Any]) -> dict[str, Any] | None:
    if values.get("chosen_strategy") is None:
        return None
    return _value_fields(values, ("chosen_strategy", "chosen_margin", "menu_size"))


# Legacy parity: ``ScoreResult.scored`` (engine/score.py:847-848) is
# ``exp_pnl_model is not None or exp_pnl_analog is not None`` -- NOT
# ``exp_pnl_sim``, which is a third, independent layer (engine/score.py:685,
# NO_PAYOFF_MAP only stops the model layer, THIN_ANALOGS's zero-analog case
# only empties the analog layer). v2 carries the identical two field names
# verbatim in ``values``/``legacy_fields`` -- confirmed at
# engine/v2/serving/bridge.py:317 (``row.get("exp_pnl_model") is not None or
# row.get("exp_pnl_analog") is not None``), the serving-layer twin of this
# same rule. This is the scoring-layer version: a row with neither number,
# and no already-refusing flag, must not be silently marked "scored".
_SCORE_NUMBER_FIELDS = ("exp_pnl_model", "exp_pnl_analog")


def _has_score_number(values: Mapping[str, Any]) -> bool:
    for name in _SCORE_NUMBER_FIELDS:
        value = values.get(name)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(number):
            return True
    return False


def _record_payload(request: ScoreRequest, values: Mapping[str, Any],
                    legacy_fields: Mapping[str, Any]) -> dict[str, Any]:
    features, null_masks = _feature_fields(values)
    reasons = tuple(values.get("flags") or ())
    if not flags_refuse(reasons) and not _has_score_number(values):
        # Neither layer produced a number and nothing else already refuses
        # this row: name the refusal instead of letting it pass silently as
        # "scored" with no numbers (the defect this guards against). No
        # legacy flag covers this case -- NO_PAYOFF_MAP/THIN_ANALOGS/etc. are
        # per-layer, not "both layers empty" -- so this is a native-only code
        # and is deliberately absent from ADVISORY_FLAGS: it always refuses.
        reasons = (*reasons, "NO_SCORE")
    diagnostics = financial_diagnostics(values)
    forecasts = _value_fields(values, ("driver_prediction", "forecast_abs_move",
                                        "runup_move_prediction", "exp_pnl_sim",
                                        "chooser_score"))
    forecasts.update({key: values[key] for key in (
        "runup_move_raw_d14", "runup_move_days", "runup_move_scale",
        "runup_move_provenance",
    ) if key in values})
    uncertainty = _value_fields(
        values, ("model_p10", "model_p90", "forecast_p10", "forecast_p90"),
    )
    uncertainty.update({key: values[key] for key in (
        "runup_move_raw_d14_p10", "runup_move_raw_d14_p90",
        "runup_move_raw_d14_sd", "runup_move_p10", "runup_move_p90",
        "runup_move_sd",
    ) if key in values})
    return dict(
        score_id="pending",
        canonical_request=to_document(request),
        resolved_request={key: value for key, value in legacy_fields.items()
                          if key != "scorer"},
        event_ref={"event_id": request.event_id, "event_revision": request.event_revision},
        clock_id=request.decision_clock_id,
        snapshot_ref=request.snapshot_id,
        dependency_hash=dependency_hash({"refs": request.dependency_refs}),
        model_artifact_ids=tuple(values.get("_model_artifact_ids") or request.model_artifact_refs),
        selected_contracts=tuple(values.get("selected_contracts") or values.get("legs") or ()),
        legs=tuple(values.get("legs") or ()),
        entry_exit_plan=_value_fields(values, ("entry_date", "exit_date", "quote_date", "expiry")),
        quote_provenance=_value_fields(values, ("quote_date", "quote_age_sessions", "fill")),
        forecasts=forecasts,
        uncertainty=uncertainty,
        residual_state_ref=request.residual_state_ref,
        analog_state_ref=request.analog_state_ref,
        payoff_state_ref=request.calibration_state_ref,
        feature_values=features,
        null_masks=null_masks,
        feature_lineage_refs=tuple(request.dependency_refs),
        gate_terms={name: values.get(name) for name in
                    ("gate_score", "gate_threshold", "gate_pass")},
        chooser_candidates=(),
        chooser_selection=_chooser_selection(values),
        financial_diagnostics=diagnostics,
        requested_payoff_views=(),
        validation_status="refused" if flags_refuse(reasons) else "scored",
        readiness="refused" if flags_refuse(reasons) else "ready",
        reason_codes=reasons,
        warnings=tuple(values.get("detail", "").split("; ")) if values.get("detail") else (),
        evidence_refs=tuple(request.dependency_refs),
        dependency_manifest_ref=request.snapshot_id,
    )


def _record(request: ScoreRequest, legacy_result, legacy_fields: Mapping[str, Any]) -> ScoreRecord:
    values = legacy_result.as_dict()
    return _record_values(request, values, legacy_fields)


def _record_values(request: ScoreRequest, values: Mapping[str, Any],
                   legacy_fields: Mapping[str, Any]) -> ScoreRecord:
    return with_score_id(ScoreRecord(**_record_payload(request, values, legacy_fields)))


def _apply_frozen_predictions(values: dict[str, Any], result, release,
                              inference_request) -> None:
    binding_id = getattr(result, "binding_id", None) or getattr(
        inference_request, "binding_id", None)
    binding = next((item for item in getattr(release, "bindings", ())
                    if item.binding_id == binding_id), None)
    names = tuple(getattr(result, "output_names", ()) or
                  (getattr(binding, "output_names", ()) if binding else ()))
    predictions = tuple(result.predictions[0])
    role = str(getattr(binding, "role", ""))
    targets = {
        "size": "forecast_abs_move", "implied_t1": "driver_prediction",
        "runup_move": "runup_move_prediction", "gate": "gate_score",
        "chooser": "chooser_score",
    }
    target = targets.get(role, targets.get(role.split(":", 1)[0]))
    allowed = {"driver_prediction", "forecast_abs_move", "runup_move_prediction",
               "gate_score", "chooser_score"}
    for index, prediction in enumerate(predictions):
        named = names[index] if index < len(names) else None
        output = named if named in allowed else target if len(predictions) == 1 else None
        if output is None and len(predictions) == 1 and not role and not names:
            output = "driver_prediction"
        if output is None:
            raise ValueError(f"unmapped frozen inference output for role {role!r}")
        values[output] = prediction


def _frozen_outputs(result, binding) -> dict[str, float]:
    if getattr(result, "status", None) != "READY":
        return {}
    names = tuple(getattr(result, "output_names", ()) or ())
    predictions = tuple(getattr(result, "predictions", ()) or ())
    row = tuple(predictions[0]) if predictions else ()
    role = str(getattr(binding, "role", ""))
    role_outputs = _FROZEN_ROLE_OUTPUTS.get(role.split(":", 1)[0], ())
    target = role_outputs[0] if role_outputs else None
    outputs = {}
    for index, value in enumerate(row):
        name = names[index] if index < len(names) else None
        output = name if name in _FROZEN_OUTPUTS else target if len(row) == 1 else None
        if output is not None:
            outputs[output] = value
    return outputs


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _metadata(sources, names):
    for source in sources:
        for name in names:
            value = (source.get(name) if isinstance(source, Mapping)
                     else getattr(source, name, None))
            if value is not None:
                return value
    return None


def _runup_days(request: ScoreRequest, fields: Mapping[str, Any],
                base: NativeScoreInputs) -> float | None:
    model_inputs = base.features.get("model_inputs")
    sources = (
        request.geometry_override or {}, request.contract_override or {}, fields,
        base.context, base.features,
        model_inputs if isinstance(model_inputs, Mapping) else {},
    )
    return _finite_number(_metadata(sources, ("days_before_print",)))


def _runup_interval(result, named: Mapping[str, Any]) -> tuple[float | None, ...]:
    direct = tuple(_finite_number(named.get(name)) for name in ("p10", "p90", "sd"))
    if any(value is not None for value in direct):
        return direct
    interval = _metadata(
        (result,), ("prediction_interval", "prediction_intervals", "interval", "intervals"),
    )
    if isinstance(interval, (tuple, list)) and len(interval) == 1:
        interval = interval[0]
    if not isinstance(interval, (tuple, list)):
        return (None, None, None)
    values = tuple(_finite_number(value) for value in interval[:3])
    return (*values, *(None for _ in range(3 - len(values))))


def _member_hashes(binding, tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        member.content_hash
        for member in (getattr(binding, "members", ()) or ())
        if any(token in str(getattr(member, "name", "")).lower() for token in tokens)
    )


def _runup_prediction_parts(result, binding):
    if getattr(result, "status", None) != "READY":
        return None, {}, ()
    names = tuple(getattr(result, "output_names", ()) or
                  getattr(binding, "output_names", ()) or ())
    predictions = tuple(getattr(result, "predictions", ()) or ())
    row = tuple(predictions[0]) if predictions else ()
    if any(name in _RUNUP_FINAL_NAMES for name in names):
        return None, {}, ("PRETRANSFORMED_FROZEN_OUTPUT:runup_move",)
    named = dict(zip(names, row))
    raw = next((_finite_number(named[name]) for name in _RUNUP_RAW_NAMES
                if name in named), None)
    if raw is None and len(row) == 1 and (not names or names[0] not in _RUNUP_FINAL_NAMES):
        raw = _finite_number(row[0])
    if raw is None:
        return None, {}, ("NONFINITE_FORECAST_OUTPUT:runup_move_prediction",)
    return raw, named, ()


def _runup_state(result, binding, inference_request, raw, named, days, scale):
    raw_p10, raw_p90, raw_sd = _runup_interval(result, named)
    state: dict[str, Any] = {
        "runup_move_raw_d14": raw,
        "runup_move_prediction": max(raw, 0.0) * scale,
        "runup_move_days": days,
        "runup_move_scale": scale,
    }
    for key, value in (
        ("runup_move_raw_d14_p10", raw_p10),
        ("runup_move_raw_d14_p90", raw_p90),
        ("runup_move_raw_d14_sd", raw_sd),
        ("runup_move_p10", None if raw_p10 is None else max(raw_p10, 0.0) * scale),
        ("runup_move_p90", None if raw_p90 is None else max(raw_p90, 0.0) * scale),
        ("runup_move_sd", None if raw_sd is None else raw_sd * scale),
    ):
        if value is not None:
            state[key] = value
    sources = (result, inference_request, binding)
    state["runup_move_provenance"] = {
        "release_id": getattr(result, "release_id", None),
        "binding_id": getattr(binding, "binding_id", None),
        "model_id": getattr(result, "model_id", None),
        "fold": _metadata(sources, ("fold_start", "forecast_fold", "fold_id")),
        "calibration_ref": _metadata(
            sources, ("calibration_ref", "calibration_state_ref", "calibration_id"),
        ),
        "artifact_hashes": tuple(getattr(result, "artifact_hashes", ()) or ()),
        "interval_artifact_hashes": _member_hashes(binding, ("residual", "interval")),
        "calibration_artifact_hashes": _member_hashes(binding, ("calibration",)),
        "raw_horizon_days": _RUNUP_BASE_DAYS,
        "final_horizon_days": days,
    }
    return state


def _runup_frozen_output(result, binding, inference_request,
                         days: float | None) -> tuple[dict[str, float], dict[str, Any], tuple[str, ...]]:
    raw, named, errors = _runup_prediction_parts(result, binding)
    if raw is None:
        return {}, {}, errors
    if days is None or days < 0.0:
        return {}, {}, ("INVALID_RUNUP_HORIZON",)
    scale = days / _RUNUP_BASE_DAYS
    state = _runup_state(result, binding, inference_request, raw, named, days, scale)
    return {"runup_move_prediction": state["runup_move_prediction"]}, state, ()


def _frozen_role_outputs(binding) -> frozenset[str]:
    """Return every local recipe owned by one requested frozen binding."""
    role = str(getattr(binding, "role", "")).split(":", 1)[0]
    outputs = set(_FROZEN_ROLE_OUTPUTS.get(role, ()))
    outputs.update(
        name for name in (getattr(binding, "output_names", ()) or ())
        if name in _FROZEN_OUTPUTS
    )
    return frozenset(outputs)


def _without_frozen_recipes(base_forecast, bindings) -> dict[str, Any]:
    owned_forecast = _FROZEN_OUTPUTS | {
        "forecast_p10", "forecast_p90", "forecast_sd",
    } | _RUNUP_DERIVED_FIELDS
    forecast = {
        key: value for key, value in base_forecast.items()
        if key not in owned_forecast
    }
    frozen_recipe_outputs = frozenset().union(
        *(_frozen_role_outputs(binding) for binding in bindings),
    )
    models = forecast.get("models")
    if isinstance(models, Mapping):
        forecast["models"] = {
            name: spec for name, spec in models.items()
            if name not in frozen_recipe_outputs
        }
    return forecast


def _without_runup_derived(block: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in block.items()
        if key not in _RUNUP_DERIVED_FIELDS
    }


def _frozen_binding(release, result, inference_request):
    binding_id = getattr(inference_request, "binding_id", None)
    binding = next((item for item in getattr(release, "bindings", ())
                    if item.binding_id == binding_id), None)
    if binding is not None:
        return binding
    return SimpleNamespace(
        binding_id=getattr(result, "binding_id", "frozen-binding"),
        role="implied_t1",
        output_names=tuple(getattr(result, "output_names", ()) or ()),
    )


def _frozen_result_state(result, binding, inference_request, days):
    role = str(getattr(binding, "role", ""))
    role_name = role.split(":", 1)[0]
    if role_name == "runup_move":
        outputs, state, role_flags = _runup_frozen_output(
            result, binding, inference_request, days,
        )
    else:
        outputs = _frozen_outputs(result, binding)
        state, role_flags = {}, ()
    gate_result = (result, binding) if role_name == "gate" else None
    return (
        role_name, outputs, state, tuple(role_flags),
        tuple(getattr(result, "artifact_hashes", ()) or ()), gate_result,
    )


def _collect_frozen_results(results, bindings, inference_requests, days):
    outputs = {}
    state = {}
    flags = []
    artifact_hashes = []
    required_roles = []
    gate_result = None
    for result, binding, inference_request in zip(
        results, bindings, inference_requests, strict=True,
    ):
        role_name, role_outputs, result_state, role_flags, hashes, current_gate = (
            _frozen_result_state(result, binding, inference_request, days)
        )
        outputs.update(role_outputs)
        state.update(result_state)
        flags.extend(role_flags)
        flags.extend(getattr(result, "reason_codes", ()) or ())
        artifact_hashes.extend(hashes)
        if role_name in {"driver", "size", "implied_t1", "runup_move", "iv_crush"}:
            required_roles.append(role_name)
        if current_gate is not None:
            gate_result = current_gate
    return outputs, state, flags, artifact_hashes, required_roles, gate_result


def _frozen_stage_executors(bindings, inference, release, days):
    executors = {}
    owned = set()
    for binding in bindings:
        role = str(getattr(binding, "role", "")).split(":", 1)[0]
        if role in _FROZEN_ROLE_OUTPUTS and hasattr(binding, "feature_order"):
            owned.update(_frozen_role_outputs(binding))
            frozen_executor = FrozenStageExecutor(
                inference=inference,
                release=release,
                binding_id=binding.binding_id,
            )
            for target, source_name, scale, floor_zero in (
                _canonical_executor_specs(binding, days)
            ):
                executors[target] = _CanonicalFrozenExecutor(
                    frozen_executor,
                    source_name,
                    target,
                    scale=scale,
                    floor_zero=floor_zero,
                )
    return executors, owned


def _frozen_forecast_inputs(base, bindings, outputs, artifact_hashes,
                            required_roles, results, inference, release, days):
    forecast = _without_frozen_recipes(base.forecast, bindings)
    forecast.update({
        "frozen_outputs": outputs,
        "artifact_hashes": tuple(dict.fromkeys(artifact_hashes)),
        "binding_id": bindings[0].binding_id,
        "binding_ids": tuple(binding.binding_id for binding in bindings),
        "model_id": getattr(results[0], "model_id", None),
        "required_roles": tuple(dict.fromkeys(required_roles)),
    })
    if inference is not None:
        executors, executor_owned = _frozen_stage_executors(
            bindings, inference, release, days,
        )
        frozen_outputs = {
            name: value for name, value in outputs.items()
            if name not in executor_owned
        }
        if frozen_outputs:
            forecast["frozen_outputs"] = frozen_outputs
        elif executor_owned:
            forecast.pop("frozen_outputs", None)
        if executors:
            forecast["executors"] = executors
    return forecast


def _frozen_gate_inputs(base, bindings, gate_result,
                        artifact_hashes, inference, release):
    # gate_threshold is answer-bearing (engine/v2/scoring/source_inputs.py
    # _ANSWER_FIELDS). The frozen/acceptance path takes it only from the
    # source-built NativeScoreInputs (base.gate) already folded into `gate`
    # below -- never from a caller-supplied fields mapping, which may carry
    # the legacy answer this run is being compared against.
    gate = dict(base.gate)
    if gate_result is not None:
        result, _ = gate_result
        row = tuple(getattr(result, "predictions", ()) or ())
        if row:
            gate.update({
                "frozen_score": row[0][0],
                "artifact_hashes": tuple(dict.fromkeys(artifact_hashes)),
            })
    if inference is not None:
        gate_bindings = [binding for binding in bindings
                         if str(getattr(binding, "role", "")).split(":", 1)[0] == "gate"]
        if gate_bindings and hasattr(gate_bindings[0], "feature_order"):
            binding = gate_bindings[0]
            gate.pop("frozen_score", None)
            names = tuple(getattr(binding, "output_names", ()) or ())
            source_name = "gate_score" if "gate_score" in names else (
                names[0] if len(names) == 1 else None
            )
            if source_name is not None:
                executor = FrozenStageExecutor(
                    inference=inference,
                    release=release,
                    binding_id=binding.binding_id,
                )
                gate["executors"] = {
                    "gate_score": _CanonicalFrozenExecutor(
                        executor, source_name, "gate_score",
                    ),
                }
    return _without_runup_derived(gate)


def _frozen_native_inputs(fields: Mapping[str, Any], results, bindings,
                          inference_requests, request, release,
                          inference=None) -> NativeScoreInputs:
    supplied = fields.get("_native_inputs")
    if not isinstance(supplied, NativeScoreInputs):
        # score_frozen is the acceptance/frozen path: its native inputs must
        # come from source material (SourceBundle / build_native_score_inputs,
        # or an explicitly assembled NativeScoreInputs), never be reconstructed
        # from the legacy fields a parity run is comparing against. Matches
        # the score_one contract at application.py:627 ("score_one requires
        # NativeScoreInputs; use the explicit legacy adapter for
        # comparisons"): NativeScoreInputs.from_legacy_fields remains for
        # non-acceptance legacy callers (e.g. checks/phase4_real.py's
        # _native() helper feeding score_one/score_many directly), but is
        # unreachable from this function.
        offending = sorted(key for key in fields if key not in {"_native_inputs", "scorer"})
        raise TypeError(
            "score_frozen requires fields['_native_inputs'] as a NativeScoreInputs "
            "built from source material; refusing to source frozen/acceptance "
            f"inputs from legacy answer fields {offending}; build inputs via "
            "SourceBundle/build_native_score_inputs (or assemble NativeScoreInputs "
            "explicitly) and pass it as fields['_native_inputs']"
        )
    base = supplied
    context_names = ("ticker", "strategy", "event_date",
                     "entry_date", "exit_date", "expiry",
                     "spot", "session", "as_of")
    context = {key: fields[key] for key in context_names if key in fields}
    context.setdefault("strategy", request.strategy_version)
    days = _runup_days(request, fields, base)
    context = {key: value for key, value in {**base.context, **context}.items()
               if key not in _RUNUP_DERIVED_FIELDS}
    features = {key: value for key, value in base.features.items()
                if key not in _RUNUP_DERIVED_FIELDS}
    features.update({key: fields[key] for key in ("model_inputs", "implied_move",
                                                   "spot", "pre_iv30")
                     if key in fields})
    if days is not None:
        features["days_before_print"] = days
    outputs = {}
    outputs, result_state, flags, artifact_hashes, required_roles, gate_result = (
        _collect_frozen_results(results, bindings, inference_requests, days)
    )
    context.update(result_state)
    release_id = getattr(results[0], "release_id", request.deployment_id)
    binding_ids = tuple(binding.binding_id for binding in bindings)
    source = f"frozen:{release_id}:{','.join(binding_ids)}"
    forecast = _frozen_forecast_inputs(
        base, bindings, outputs, artifact_hashes, required_roles,
        results, inference, release, days,
    )
    gate = _frozen_gate_inputs(
        base, bindings, gate_result, artifact_hashes, inference, release,
    )
    return replace(
        base,
        context={**context, "flags": flags},
        features=features,
        forecast=forecast,
        analogs=_without_runup_derived(base.analogs),
        simulation=_without_runup_derived(base.simulation),
        gate=gate,
        chooser=_without_runup_derived(base.chooser),
        diagnostics=_without_runup_derived(base.diagnostics),
        source_ref=source,
    )


def score_frozen(request: ScoreRequest, inference, release, inference_request,
                 fields: Mapping[str, Any], *, observer: StageObserver | None = None) -> ScoreRecord:
    """Run verified inference through the canonical native scoring graph."""
    requests = (tuple(inference_request) if isinstance(inference_request, (tuple, list))
                else (inference_request,))
    results = tuple(inference.infer(release, item) for item in requests)
    bindings = tuple(
        _frozen_binding(release, result, item)
        for result, item in zip(results, requests, strict=True)
    )
    inputs = _frozen_native_inputs(
        fields, results, bindings, requests, request, release, inference,
    )
    record = score_one(request, inputs, observer=observer)
    artifact_hashes = tuple(dict.fromkeys(
        hash_value
        for result in results
        for hash_value in (getattr(result, "artifact_hashes", ()) or ())
    ))
    release_ids = tuple(dict.fromkeys(
        getattr(result, "release_id", request.deployment_id)
        for result in results
    ))
    binding_ids = tuple(binding.binding_id for binding in bindings)
    record = replace(record, model_artifact_ids=artifact_hashes,
                     evidence_refs=tuple(dict.fromkeys((*request.dependency_refs,
                                                         *release_ids, *binding_ids))))
    reasons = tuple(
        reason
        for result in results
        for reason in (getattr(result, "reason_codes", ()) or ())
    )
    if any(getattr(result, "status", None) != "READY" for result in results):
        record = replace(record, reason_codes=tuple(dict.fromkeys((*record.reason_codes, *reasons))),
                         validation_status="refused", readiness="refused")
    return with_score_id(record)


def score_one(request: ScoreRequest, inputs: NativeScoreInputs, *,
              observer: StageObserver | None = None) -> ScoreRecord:
    """Emit one record after all explicitly owned native stages completed."""
    if not isinstance(inputs, NativeScoreInputs):
        raise TypeError("score_one requires NativeScoreInputs; use the explicit legacy adapter for comparisons")
    inputs = _with_request_overrides(request, inputs)
    values = assemble_native_values(
        inputs, strategy=request.strategy_version, fill_model=request.fill_model,
        observer=observer,
    )
    values["_model_artifact_ids"] = tuple(request.model_artifact_refs)
    return _record_values(request, values, values)


def score_many(requests: Iterable[tuple[ScoreRequest, NativeScoreInputs]]) -> tuple[ScoreRecord, ...]:
    """Batch scoring uses the same one-request kernel and preserves order."""
    return tuple(score_one(request, fields) for request, fields in requests)


def _with_request_overrides(request: ScoreRequest,
                            inputs: NativeScoreInputs) -> NativeScoreInputs:
    """Route request-owned geometry through native generation and pricing."""
    if request.contract_override is None and request.geometry_override is None:
        return inputs
    geometry_override = dict(request.geometry_override or {})
    contract_override = dict(request.contract_override or {})
    for alias in ("legs", "contracts", "selected_contracts"):
        if alias in contract_override and "resolved_legs" not in contract_override:
            contract_override["resolved_legs"] = contract_override.pop(alias)
    overrides = {**geometry_override, **contract_override}
    return replace(
        inputs,
        context={**inputs.context, **overrides},
        # Features merge after context in the native graph. Repeating the
        # request values here gives the explicit request final precedence.
        features={**inputs.features, **overrides},
        forecast={**inputs.forecast, **overrides},
        # A pre-resolved geometry would otherwise overwrite the request fields.
        # Pricing remains the quote inventory and is recomputed by the stage.
        geometry=None,
    )


def score_batch(batch: ScoreBatch, fields_by_request: Mapping[Any, NativeScoreInputs]) -> tuple[ScoreRecord, ...]:
    """Score a declared batch while preserving request order and identity."""
    event_counts = Counter(request.event_id for request in batch.requests)
    pair_counts = Counter((request.event_id, request.strategy_version)
                          for request in batch.requests)

    def inputs_for(request: ScoreRequest) -> NativeScoreInputs:
        identity = request_hash(request)
        if identity in fields_by_request:
            return fields_by_request[identity]
        pair = (request.event_id, request.strategy_version)
        if pair in fields_by_request:
            if pair_counts[pair] > 1:
                raise KeyError(
                    f"ambiguous event/strategy batch key {pair!r}; "
                    f"use request hash {identity!r}"
                )
            return fields_by_request[pair]
        if request.event_id in fields_by_request:
            if event_counts[request.event_id] > 1:
                raise KeyError(
                    f"ambiguous legacy event-only batch key {request.event_id!r}; "
                    f"use request hash {identity!r}"
                )
            return fields_by_request[request.event_id]
        raise KeyError(
            f"missing batch inputs for request hash {identity!r} "
            f"or event/strategy {pair!r}"
        )

    return score_many((request, inputs_for(request)) for request in batch.requests)


def replay(score_id: str, records: Mapping[str, ScoreRecord], legacy_fields: Mapping[str, Any]) -> tuple[ScoreRecord, ReplayReceipt]:
    """Replay one stored request and return its receipt alongside the record."""
    previous = records[score_id]
    request = from_document(ScoreRequest, previous.canonical_request)
    result = score_one(request, legacy_fields)
    receipt = ReplayReceipt(
        score_id=score_id, request_hash=previous.request_hash,
        replay_score_id=result.score_id,
        status="replayed" if result.score_id == score_id else "refused",
        validation_receipt_ref=result.payload_hash,
    )
    return result, receipt


def score_event(event_request: ScoreRequest, strategies: Iterable[tuple[str, Mapping[str, Any]]]):
    """Score all strategies for one event through the same batch kernel."""
    registry = default_registry()
    requests = []
    for strategy, fields in strategies:
        spec = registry.strategy(strategy)
        if strategy == "DYN-SV":
            menu = fields.get("menu")
            if (not isinstance(menu, Mapping)
                    or tuple(menu) != DYNAMIC_MENU):
                raise ValueError("direct DYN-SV requests must provide complete menu fields")
            candidates = tuple(
                score_many(((replace(event_request, strategy_version=member), member_fields),))
                for member, member_fields in menu.items()
            )
            flat = tuple(item for group in candidates for item in group)
            if flat:
                requests.append((_choose_dynamic(event_request, flat), None))
            continue
        requests.append((replace(event_request, strategy_version=spec.strategy_id), fields))
    return tuple(item if fields is None else score_one(item, fields)
                 for item, fields in requests)


def _choose_dynamic(request: ScoreRequest, candidates: tuple[ScoreRecord, ...]) -> ScoreRecord:
    def metric(record: ScoreRecord, name: str) -> float | None:
        try:
            value = float(record.forecasts.get(name))
        except (TypeError, ValueError):
            return None
        return value if isfinite(value) else None

    simulated = tuple(record for record in candidates
                      if metric(record, "exp_pnl_sim") is not None)
    key = ("chooser_score" if any(metric(record, "chooser_score") is not None
                                  for record in simulated) else "exp_pnl_sim")
    eligible = [(metric(record, key), key, record) for record in simulated
                if metric(record, key) is not None]
    if not eligible:
        base = candidates[0]
        selection = {"status": "no_eligible_candidate", "menu_size": len(candidates)}
        return with_score_id(replace(base, canonical_request={**base.canonical_request,
                        "strategy_version": "DYN-SV"},
                       chooser_candidates=tuple({
                           "strategy": r.canonical_request.get("strategy_version"),
                           "score_id": r.score_id,
                       } for r in candidates), chooser_selection=selection,
                       validation_status="refused", reason_codes=tuple(dict.fromkeys(
                           (*base.reason_codes, "NO_CHOOSER_CANDIDATE")))))
    eligible.sort(key=lambda item: item[0], reverse=True)
    best_value, key, best = eligible[0]
    tied = [item for item in eligible if item[0] == best_value]
    selection = {
        "status": "tie" if len(tied) > 1 else "selected",
        "strategy": best.canonical_request.get("strategy_version"),
        "ranking_key": key,
        "value": best_value,
        "menu_size": len(eligible),
    }
    return with_score_id(replace(best, canonical_request={**best.canonical_request,
                    "strategy_version": "DYN-SV"},
                   chooser_candidates=tuple({
                       "strategy": r.canonical_request.get("strategy_version"),
                       "score_id": r.score_id,
                       "eligible": any(r is item[2] for item in eligible),
                   } for r in candidates), chooser_selection=selection))
