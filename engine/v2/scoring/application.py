"""Shared single, event and batch Phase 4 scoring application."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping

from engine.v2.contracts import ReplayReceipt, ScoreBatch, ScoreRecord, ScoreRequest
from engine.v2.features import default_feature_registry
from engine.v2.foundation import from_document, to_document
from engine.v2.registry import DYNAMIC_MENU, default_registry

from .financial import financial_diagnostics
from .identity import dependency_hash, with_score_id

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
        model_artifact_ids=tuple(request.model_artifact_refs),
        selected_contracts=tuple(values.get("legs") or ()),
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


def score_frozen(request: ScoreRequest, inference, release, inference_request,
                 fields: Mapping[str, Any]) -> ScoreRecord:
    """Score the model stage from an explicitly verified frozen release."""
    result = inference.infer(release, inference_request)
    values = dict(fields)
    values.setdefault("model_inputs", {})
    if result.status != "READY":
        values["flags"] = tuple(result.reason_codes)
        values["detail"] = result.detail or "model release is not ready"
    else:
        values["driver_prediction"] = result.predictions[0][0]
        values["forecast_abs_move"] = result.predictions[0][0]
        values["model_artifact_ids"] = result.artifact_hashes
        values["flags"] = ()
        values["detail"] = ""
    return _record_values(request, values, fields)


def score_one(request: ScoreRequest, legacy_fields: Mapping[str, Any]) -> ScoreRecord:
    """Score one native stage payload through the shared canonical application."""
    values = legacy_fields.get("native_values", legacy_fields)
    if not isinstance(values, Mapping):
        raise TypeError("native_values must be a mapping")
    return _record_values(request, values, legacy_fields)


def score_many(requests: Iterable[tuple[ScoreRequest, Mapping[str, Any]]]) -> tuple[ScoreRecord, ...]:
    """Batch scoring uses the same one-request kernel and preserves order."""
    return tuple(score_one(request, fields) for request, fields in requests)


def score_batch(batch: ScoreBatch, fields_by_request: Mapping[str, Mapping[str, Any]]) -> tuple[ScoreRecord, ...]:
    """Score a declared batch while preserving request order and identity."""
    return score_many((request, fields_by_request[request.event_id]) for request in batch.requests)


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
            return (_choose_dynamic(event_request, flat),) if flat else ()
        requests.append((replace(event_request, strategy_version=spec.strategy_id), fields))
    return score_many(requests)


def _choose_dynamic(request: ScoreRequest, candidates: tuple[ScoreRecord, ...]) -> ScoreRecord:
    eligible = []
    for record in candidates:
        value = record.forecasts.get("chooser_score")
        key = "chooser_score"
        if value is None:
            value = record.forecasts.get("exp_pnl_sim")
            key = "exp_pnl_sim"
        if value is not None and record.validation_status != "refused":
            eligible.append((float(value), key, record))
    if not eligible:
        base = candidates[0]
        selection = {"status": "no_eligible_candidate", "menu_size": len(candidates)}
        return with_score_id(replace(base, canonical_request={**base.canonical_request,
                        "strategy_version": "DYN-SV"},
                       chooser_candidates=tuple({
                           "strategy": r.canonical_request.get("strategy_version"),
                           "score_id": r.score_id,
                       } for r in candidates), chooser_selection=selection,
                       validation_status="refused", reason_codes=("NO_CHOOSER_CANDIDATE",)))
    eligible.sort(key=lambda item: item[0], reverse=True)
    best_value, key, best = eligible[0]
    tied = [item for item in eligible if item[0] == best_value]
    selection = {
        "status": "tie" if len(tied) > 1 else "selected",
        "strategy": best.canonical_request.get("strategy_version"),
        "ranking_key": key,
        "value": best_value,
        "menu_size": len(candidates),
    }
    return with_score_id(replace(best, canonical_request={**best.canonical_request,
                    "strategy_version": "DYN-SV"},
                   chooser_candidates=tuple({
                       "strategy": r.canonical_request.get("strategy_version"),
                       "score_id": r.score_id,
                       "eligible": any(r is item[2] for item in eligible),
                   } for r in candidates), chooser_selection=selection))
