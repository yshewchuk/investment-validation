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
from .identity import dependency_hash, request_hash, with_score_id
from .stages import NativeScoreInputs, assemble_native_values

__all__ = ["replay", "score_batch", "score_event", "score_frozen", "score_many", "score_one"]

_FEATURE_REGISTRY = default_feature_registry()


def _value_fields(values: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: values.get(name) for name in names}


def _feature_fields(values: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    features = dict(values.get("model_inputs") or {})
    return features, {name: value is None for name, value in features.items()}


def _chooser_selection(values: Mapping[str, Any]) -> dict[str, Any] | None:
    if values.get("chosen_strategy") is None:
        return None
    return _value_fields(values, ("chosen_strategy", "chosen_margin", "menu_size"))


def _record_payload(request: ScoreRequest, values: Mapping[str, Any],
                    legacy_fields: Mapping[str, Any]) -> dict[str, Any]:
    features, null_masks = _feature_fields(values)
    reasons = tuple(values.get("flags") or ())
    diagnostics = financial_diagnostics(values)
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
        forecasts=_value_fields(values, ("driver_prediction", "forecast_abs_move",
                                         "runup_move_prediction", "exp_pnl_sim", "chooser_score")),
        uncertainty=_value_fields(values, ("model_p10", "model_p90", "forecast_p10", "forecast_p90")),
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
        validation_status="refused" if reasons else "scored",
        readiness="refused" if reasons else "ready",
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
    names = tuple(getattr(result, "output_names", ()) or ())
    predictions = tuple(getattr(result, "predictions", ()) or ())
    row = tuple(predictions[0]) if predictions else ()
    role_targets = {
        "size": "forecast_abs_move", "implied_t1": "driver_prediction",
        "runup_move": "runup_move_prediction", "iv_crush": "pred_iv_crush",
        "fair_value": "model_fair_pct",
    }
    allowed = {
        "driver_prediction", "forecast_abs_move", "runup_move_prediction",
        "pred_iv_crush", "pred_iv_crush_30", "model_fair_pct",
    }
    role = str(getattr(binding, "role", ""))
    target = role_targets.get(role.split(":", 1)[0])
    outputs = {}
    for index, value in enumerate(row):
        name = names[index] if index < len(names) else None
        output = name if name in allowed else target if len(row) == 1 else None
        if output is not None:
            outputs[output] = value
    return outputs


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


def _frozen_native_inputs(fields: Mapping[str, Any], results, bindings, request) -> NativeScoreInputs:
    supplied = fields.get("_native_inputs")
    if isinstance(supplied, NativeScoreInputs):
        base = supplied
    else:
        raw = {key: value for key, value in fields.items()
               if key not in {"_native_inputs", "scorer"}}
        base = NativeScoreInputs.from_legacy_fields(raw)
    context_names = ("ticker", "strategy", "event_date",
                     "entry_date", "exit_date", "expiry",
                     "spot", "session", "as_of")
    context = {key: fields[key] for key in context_names if key in fields}
    context.setdefault("strategy", request.strategy_version)
    features = dict(base.features)
    features.update({key: fields[key] for key in ("model_inputs", "implied_move",
                                                   "spot", "pre_iv30")
                     if key in fields})
    outputs = {}
    flags = []
    artifact_hashes = []
    required_roles = []
    gate_result = None
    for result, binding in zip(results, bindings, strict=True):
        outputs.update(_frozen_outputs(result, binding))
        flags.extend(getattr(result, "reason_codes", ()) or ())
        artifact_hashes.extend(getattr(result, "artifact_hashes", ()) or ())
        role = str(getattr(binding, "role", ""))
        if role in {"size", "implied_t1", "runup_move", "iv_crush"}:
            required_roles.append(role)
        if role == "gate":
            gate_result = (result, binding)
    release_id = getattr(results[0], "release_id", request.deployment_id)
    binding_ids = tuple(binding.binding_id for binding in bindings)
    source = f"frozen:{release_id}:{','.join(binding_ids)}"
    owned_forecast = {"driver_prediction", "forecast_abs_move", "runup_move_prediction",
                      "pred_iv_crush", "pred_iv_crush_30", "model_fair_pct",
                      "forecast_p10", "forecast_p90", "forecast_sd"}
    forecast = {key: value for key, value in base.forecast.items()
                if key not in owned_forecast}
    forecast.update({
        "frozen_outputs": outputs,
        "artifact_hashes": tuple(dict.fromkeys(artifact_hashes)),
        "binding_id": binding_ids[0],
        "binding_ids": binding_ids,
        "model_id": getattr(results[0], "model_id", None),
        "required_roles": tuple(dict.fromkeys(required_roles)),
    })
    gate = dict(base.gate)
    if gate_result is not None:
        result, _ = gate_result
        row = tuple(getattr(result, "predictions", ()) or ())
        if row:
            gate.update({
                "frozen_score": row[0][0],
                "artifact_hashes": tuple(dict.fromkeys(artifact_hashes)),
                "threshold": fields.get("gate_threshold"),
            })
    return replace(base, context={**base.context, **context, "flags": flags},
                   features=features, forecast=forecast, gate=gate, source_ref=source)


def score_frozen(request: ScoreRequest, inference, release, inference_request,
                 fields: Mapping[str, Any]) -> ScoreRecord:
    """Run verified inference through the canonical native scoring graph."""
    requests = (tuple(inference_request) if isinstance(inference_request, (tuple, list))
                else (inference_request,))
    results = tuple(inference.infer(release, item) for item in requests)
    bindings = tuple(
        _frozen_binding(release, result, item)
        for result, item in zip(results, requests, strict=True)
    )
    inputs = _frozen_native_inputs(fields, results, bindings, request)
    record = score_one(request, inputs)
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


def score_one(request: ScoreRequest, inputs: NativeScoreInputs) -> ScoreRecord:
    """Emit one record after all explicitly owned native stages completed."""
    if not isinstance(inputs, NativeScoreInputs):
        raise TypeError("score_one requires NativeScoreInputs; use the explicit legacy adapter for comparisons")
    inputs = _with_request_overrides(request, inputs)
    values = assemble_native_values(
        inputs, strategy=request.strategy_version, fill_model=request.fill_model,
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
