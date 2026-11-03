"""Frozen in-memory fit transport; from_recipe applies the guards, and public adapters also guard before recipe validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.v2.foundation.iso_dates import validate_train_event_dates
from engine.v2.models.no_fit import forbid_fitting as forbid_v2_fitting

from .legacy_adapter import forbid_fitting as forbid_legacy_fitting


def check_fitting_allowed(path: str) -> None:
    """Apply both legacy and v2 fitting guards in fixed order for the given path."""
    forbid_legacy_fitting(path)
    forbid_v2_fitting(path)


@dataclass(frozen=True)
class FitInputs:
    """Transport that stores fields as given; use from_recipe for guard and column checks."""

    train_features: Any
    train_labels: Any
    prediction_features: Any
    columns: list[str] | tuple[str, ...]
    train_event_dates: list[str] | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        validate_train_event_dates(self.train_event_dates, self.train_features)

    @classmethod
    def from_recipe(cls, recipe, train_features, train_labels, prediction_features, columns,
                    *, column_error,
                    guard_path="FitInputs.from_recipe") -> FitInputs:
        """Check guards and ordered columns, then store the caller objects as given."""
        check_fitting_allowed(guard_path)
        if not isinstance(columns, (list, tuple)) or list(columns) != recipe["features"]:
            raise ValueError(column_error)
        return cls(train_features, train_labels, prediction_features, columns)

    def validate_row_count(self, train_rows, label_rows, row_error) -> None:
        """Validate rows after matrix conversion and matrix-shape/infinity checks."""
        if train_rows != label_rows:
            raise ValueError(row_error)