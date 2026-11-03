"""Frozen in-memory fit transport guarded once at construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.v2.models.no_fit import forbid_fitting as forbid_v2_fitting

from .legacy_adapter import forbid_fitting as forbid_legacy_fitting


def check_fitting_allowed(path: str) -> None:
    """Apply both legacy and v2 fitting guards in fixed order for the given path."""
    forbid_legacy_fitting(path)
    forbid_v2_fitting(path)


def _is_selected_matrix(matrix, width: int) -> bool:
    """Check structural nonempty 2D width without reading numeric values."""
    shape = getattr(matrix, "shape", None)
    if shape is not None:
        try:
            rows, columns = shape
            return rows > 0 and columns == width
        except (TypeError, ValueError):
            return False
    if not isinstance(matrix, (list, tuple)) or not matrix:
        return False
    return all(isinstance(row, (list, tuple)) and len(row) == width for row in matrix)


@dataclass(frozen=True)
class FitInputs:
    """Caller-admitted arrays stored as given; no recipe, errors or conversions."""

    train_features: Any
    train_labels: Any
    prediction_features: Any
    columns: list[str] | tuple[str, ...]

    @classmethod
    def from_recipe(cls, recipe, train_features, train_labels, prediction_features, columns,
                    *, column_error, row_error,
                    guard_path="FitInputs.from_recipe") -> FitInputs:
        """Validate ordered recipe columns and train row counts, then store as given."""
        check_fitting_allowed(guard_path)
        if not isinstance(columns, (list, tuple)) or list(columns) != recipe["features"]:
            raise ValueError(column_error)
        if (_is_selected_matrix(train_features, len(columns))
                and _is_selected_matrix(prediction_features, len(columns))):
            try:
                rows_features = len(train_features)
                rows_labels = len(train_labels)
            except (TypeError, AttributeError):
                pass
            else:
                if rows_features != rows_labels:
                    raise ValueError(row_error)
        return cls(train_features, train_labels, prediction_features, columns)