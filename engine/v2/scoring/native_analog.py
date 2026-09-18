"""Deterministic nearest-neighbor analog scoring from source-only rows."""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from typing import Any, Mapping, Sequence

from engine.v2.foundation import content_hash

__all__ = [
    "AnalogRecipe",
    "AnalogRefusal",
    "AnalogResult",
    "evaluate_analogs",
    "source_population_hash",
]

RECIPE_SCHEMA = "native_analog_recipe.v1.0"
POPULATION_SCHEMA = "native_analog_population.v1.0"

_ANSWER_FIELDS = frozenset({
    "ci_high",
    "ci_low",
    "exp_pnl_analog",
    "n_analogs",
    "selected_neighbors",
    "win_analog",
})
_ROW_FIELDS = frozenset({"row_id", "features", "realized_pnl"})
_RECIPE_FIELDS = frozenset({
    "schema_version",
    "feature_names",
    "neighbors",
    "population_hash",
})


class AnalogRefusal(ValueError):
    """A stable refusal from the bounded native analog calculation."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, kw_only=True)
class AnalogRecipe:
    """Answer-free instructions for one nearest-neighbor calculation."""

    feature_names: tuple[str, ...]
    neighbors: int
    population_hash: str
    schema_version: str = RECIPE_SCHEMA


@dataclass(frozen=True, kw_only=True)
class AnalogResult:
    exp_pnl_analog: float
    win_analog: float
    ci_low: float
    ci_high: float
    n_analogs: int


def _refuse(code: str, detail: str) -> None:
    raise AnalogRefusal(code, detail)


def _reject_answers(value: Any, location: str) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(str(key) for key in value if str(key) in _ANSWER_FIELDS)
        if forbidden:
            _refuse(
                "PRECOMPUTED_ANALOG_ANSWER",
                f"{location} contains answer fields: {forbidden}",
            )
        for key, child in value.items():
            _reject_answers(child, f"{location}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _reject_answers(child, f"{location}[{index}]")


def _finite(value: Any, location: str) -> float:
    if isinstance(value, bool):
        _refuse("INVALID_ANALOG_INPUT", f"{location} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError):
        _refuse("INVALID_ANALOG_INPUT", f"{location} must be finite")
    if not isfinite(number):
        _refuse("INVALID_ANALOG_INPUT", f"{location} must be finite")
    return number


def _normalize_rows(
    source_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if not source_rows:
        _refuse("MISSING_ANALOG_POPULATION", "source population is empty")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(source_rows):
        if not isinstance(row, Mapping):
            _refuse("INVALID_ANALOG_SOURCE", f"row {index} must be a mapping")
        _reject_answers(row, f"source_rows[{index}]")
        unsupported = sorted(set(row) - _ROW_FIELDS)
        missing = sorted(_ROW_FIELDS - set(row))
        if unsupported or missing:
            _refuse(
                "INVALID_ANALOG_SOURCE",
                f"row {index} has missing={missing} unsupported={unsupported}",
            )
        row_id = str(row["row_id"]).strip()
        if not row_id:
            _refuse("INVALID_ANALOG_SOURCE", f"row {index} has an empty row_id")
        if row_id in seen:
            _refuse("INVALID_ANALOG_SOURCE", f"duplicate row_id: {row_id}")
        seen.add(row_id)
        features = row["features"]
        if not isinstance(features, Mapping):
            _refuse(
                "INVALID_ANALOG_SOURCE",
                f"row {row_id} features must be a mapping",
            )
        normalized.append({
            "row_id": row_id,
            "features": {
                str(name): _finite(value, f"row {row_id} feature {name}")
                for name, value in features.items()
            },
            "realized_pnl": _finite(
                row["realized_pnl"], f"row {row_id} realized_pnl",
            ),
        })
    return tuple(sorted(normalized, key=lambda row: row["row_id"]))


def source_population_hash(
    source_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Hash normalized source rows independently of their input ordering."""
    rows = _normalize_rows(source_rows)
    return content_hash({"schema_version": POPULATION_SCHEMA, "rows": rows})


def _normalize_recipe(recipe: AnalogRecipe | Mapping[str, Any]) -> AnalogRecipe:
    if isinstance(recipe, Mapping):
        _reject_answers(recipe, "recipe")
        unsupported = sorted(set(recipe) - _RECIPE_FIELDS)
        missing = sorted(
            {"feature_names", "neighbors", "population_hash"} - set(recipe)
        )
        if unsupported or missing:
            _refuse(
                "INVALID_ANALOG_RECIPE",
                f"recipe has missing={missing} unsupported={unsupported}",
            )
        normalized = AnalogRecipe(
            feature_names=tuple(str(name) for name in recipe["feature_names"]),
            neighbors=recipe["neighbors"],
            population_hash=str(recipe["population_hash"]),
            schema_version=str(recipe.get("schema_version", RECIPE_SCHEMA)),
        )
    elif isinstance(recipe, AnalogRecipe):
        normalized = recipe
    else:
        _refuse("INVALID_ANALOG_RECIPE", "recipe must be a mapping or AnalogRecipe")

    if normalized.schema_version != RECIPE_SCHEMA:
        _refuse(
            "INVALID_ANALOG_RECIPE",
            f"unsupported schema_version: {normalized.schema_version}",
        )
    if not normalized.feature_names:
        _refuse("INVALID_ANALOG_RECIPE", "feature_names must not be empty")
    if any(not name.strip() for name in normalized.feature_names):
        _refuse("INVALID_ANALOG_RECIPE", "feature names must be non-empty")
    if len(set(normalized.feature_names)) != len(normalized.feature_names):
        _refuse("INVALID_ANALOG_RECIPE", "feature_names must be unique")
    if isinstance(normalized.neighbors, bool) or not isinstance(
        normalized.neighbors, int
    ) or normalized.neighbors <= 0:
        _refuse("INVALID_ANALOG_RECIPE", "neighbors must be a positive integer")
    if not normalized.population_hash.startswith("sha256:"):
        _refuse("INVALID_ANALOG_RECIPE", "population_hash must be a content hash")
    return normalized


def evaluate_analogs(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    query_features: Mapping[str, Any],
    recipe: AnalogRecipe | Mapping[str, Any],
) -> AnalogResult:
    """Select deterministic nearest neighbors and summarize realized P&L."""
    normalized_recipe = _normalize_recipe(recipe)
    rows = _normalize_rows(source_rows)
    actual_hash = content_hash({
        "schema_version": POPULATION_SCHEMA,
        "rows": rows,
    })
    if actual_hash != normalized_recipe.population_hash:
        _refuse(
            "ANALOG_POPULATION_CORRUPT",
            "source population does not match recipe population_hash",
        )
    if not isinstance(query_features, Mapping):
        _refuse("INVALID_ANALOG_INPUT", "query_features must be a mapping")
    _reject_answers(query_features, "query_features")
    query: dict[str, float] = {}
    for name in normalized_recipe.feature_names:
        if name not in query_features:
            _refuse("MISSING_ANALOG_FEATURE", f"query is missing feature {name}")
        query[name] = _finite(query_features[name], f"query feature {name}")

    ranked: list[tuple[float, str, float]] = []
    for row in rows:
        features = row["features"]
        missing = [
            name for name in normalized_recipe.feature_names
            if name not in features
        ]
        if missing:
            _refuse(
                "MISSING_ANALOG_FEATURE",
                f"row {row['row_id']} is missing features {missing}",
            )
        squared_distance = fsum(
            (features[name] - query[name]) ** 2
            for name in normalized_recipe.feature_names
        )
        ranked.append((
            squared_distance,
            row["row_id"],
            row["realized_pnl"],
        ))
    ranked.sort(key=lambda item: (item[0], item[1]))
    outcomes = [
        item[2] for item in ranked[:normalized_recipe.neighbors]
    ]
    count = len(outcomes)
    mean = fsum(outcomes) / count
    win_rate = fsum(1.0 for value in outcomes if value > 0.0) / count
    if count == 1:
        ci_low = ci_high = mean
    else:
        variance = fsum((value - mean) ** 2 for value in outcomes) / (count - 1)
        margin = 1.96 * sqrt(variance / count)
        ci_low, ci_high = mean - margin, mean + margin
    return AnalogResult(
        exp_pnl_analog=mean,
        win_analog=win_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        n_analogs=count,
    )
