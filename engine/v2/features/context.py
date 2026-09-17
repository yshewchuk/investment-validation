"""Causal feature context planning and immutable frame construction."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from engine.v2.contracts import FeatureFrame, FeatureRequest
from engine.v2.foundation import content_hash

from .recipes import FeatureRegistry, default_feature_registry

__all__ = ["FeatureContextPlanner", "FeatureContextError"]


class FeatureContextError(ValueError):
    """A feature request violates its registered population or cutoff."""


def _causal_timestamp(value: Any, label: str) -> datetime:
    if value is None or not str(value).strip():
        raise FeatureContextError(f"{label} timestamp is required")
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError as exc:
        raise FeatureContextError(f"{label} timestamp is invalid") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise FeatureContextError(f"{label} timestamp must be timezone-aware")
    return stamp.astimezone(timezone.utc)


@dataclass(frozen=True)
class FeatureContextPlanner:
    registry: FeatureRegistry

    @classmethod
    def default(cls) -> "FeatureContextPlanner":
        return cls(default_feature_registry())

    def request(
        self,
        *,
        event_refs: Iterable[Mapping[str, Any]],
        snapshot_ref: str,
        recipe_refs: Iterable[str],
        visible_event_ids: Iterable[str] = (),
        analog_population_ref: str | None = None,
    ) -> FeatureRequest:
        events = tuple(dict(item) for item in event_refs)
        recipes = tuple(recipe_refs)
        for recipe_id in recipes:
            self.registry.get(recipe_id)
        visible = set(visible_event_ids)
        contexts = tuple({
            "event_id": item.get("event_id"),
            "decision_at": item.get("decision_at"),
            "visible": item.get("event_id") in visible,
            "analog_population_ref": analog_population_ref,
        } for item in events)
        return FeatureRequest(
            event_refs=events,
            decision_contexts=contexts,
            snapshot_ref=snapshot_ref,
            feature_recipe_refs=recipes,
        )

    def frame(
        self,
        request: FeatureRequest,
        rows: Iterable[Mapping[str, Any]],
        *,
        ordered_columns: Iterable[str],
        coverage_receipt_ref: str,
    ) -> FeatureFrame:
        columns = tuple(ordered_columns)
        values = tuple({column: row.get(column) for column in columns} | {
            key: value for key, value in row.items() if key not in columns
        } for row in rows)
        contexts = {item.get("event_id"): item for item in request.decision_contexts}
        for index, row in enumerate(values):
            event_id = row.get("event_id")
            context = contexts.get(event_id)
            if context is None:
                raise FeatureContextError(f"row {index} is outside requested events")
            observed = _causal_timestamp(row.get("observed_at"), f"row {index} observed_at")
            cutoff = _causal_timestamp(context.get("decision_at"), f"row {index} decision cutoff")
            if observed > cutoff:
                raise FeatureContextError(f"row {index} observes after decision cutoff")
        null_masks = tuple({column: row.get(column) is None for column in columns} for row in values)
        values_hash = content_hash({"columns": columns, "rows": values})
        return FeatureFrame(
            frame_ref=content_hash({"request": request.snapshot_ref, "values": values_hash}),
            schema_ref=content_hash(columns),
            row_keys_ref=content_hash(tuple(row.get("event_id") for row in values)),
            ordered_columns=columns,
            recipe_refs=request.feature_recipe_refs,
            dependency_refs=(request.snapshot_ref, coverage_receipt_ref),
            values_hash=values_hash,
            null_mask_hash=content_hash(null_masks),
            lineage_refs=request.feature_recipe_refs,
            coverage_receipt_ref=coverage_receipt_ref,
            causal_audit_receipt_ref=content_hash({"request": request, "cutoff": True}),
            values=values,
        )
