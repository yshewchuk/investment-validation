"""Deterministic analog scoring from answer-free source populations."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from typing import Any, Mapping, Sequence

import numpy as np

from engine.v2.foundation import content_hash

__all__ = [
    "AnalogRecipe",
    "AnalogRefusal",
    "AnalogResult",
    "LegacyBucketRecipe",
    "FROZEN_ANALOG_QUERY_FIELDS",
    "FROZEN_ANALOG_RECIPE_FIELDS",
    "bucket_population_hash",
    "evaluate_analogs",
    "evaluate_frozen_analogs",
    "frozen_key_mismatch",
    "implied_tercile",
    "legacy_bucket_bootstrap_seed",
    "source_population_hash",
]

RECIPE_SCHEMA = "native_analog_recipe.v1.0"
POPULATION_SCHEMA = "native_analog_population.v1.0"
BUCKET_RECIPE_SCHEMA = "legacy_bucket_analog_recipe.v1.0"
BUCKET_POPULATION_SCHEMA = "legacy_bucket_analog_population.v1.0"
LEGACY_BUCKET_DIMENSIONS = (
    "mcap_bucket",
    "moneyness_band",
    "dte_band",
    "implied_tercile",
)
LEGACY_WIDENING_ORDER = (
    "moneyness_band",
    "dte_band",
    "implied_tercile",
)

_ANSWER_FIELDS = frozenset({
    "ci_high",
    "ci_low",
    "exp_pnl_analog",
    "n_analogs",
    "selected_neighbors",
    "win_analog",
})
_BUCKET_ANSWER_FIELDS = _ANSWER_FIELDS | frozenset({
    "analog_mean",
    "analog_n",
    "analog_widened",
    "analog_win_rate",
    "contributing_row_ids",
    "dropped",
    "mean",
    "median",
    "p10",
    "p90",
    "population_row_ids",
    "selected_row_ids",
    "thin",
    "unavailable",
    "widened",
    "win_rate",
})
_ROW_FIELDS = frozenset({"row_id", "features", "realized_pnl"})
_RECIPE_FIELDS = frozenset({
    "schema_version",
    "feature_names",
    "neighbors",
    "population_hash",
})
_BUCKET_RECIPE_FIELDS = frozenset({
    "schema_version",
    "bucket_dimensions",
    "widening_order",
    "min_analogs",
    "alpha",
    "bootstrap_draws",
    "bootstrap_seed",
    "ci_quantiles",
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
class LegacyBucketRecipe:
    """Executable legacy bucket-matching instructions without saved answers.

    ``source_rows`` supplied at execution time are the causal population for
    this request, already restricted to the requested strategy and fill alpha.
    The population hash binds that filtering boundary without copying a legacy
    summary into the native input.
    """

    bucket_dimensions: tuple[str, ...]
    widening_order: tuple[str, ...]
    min_analogs: int
    alpha: float
    bootstrap_draws: int
    bootstrap_seed: int
    ci_quantiles: tuple[float, float]
    population_hash: str
    schema_version: str = BUCKET_RECIPE_SCHEMA


@dataclass(frozen=True, kw_only=True)
class AnalogResult:
    exp_pnl_analog: float | None
    win_analog: float | None
    ci_low: float | None
    ci_high: float | None
    n_analogs: int
    median: float | None = None
    p10: float | None = None
    p90: float | None = None
    widened: int = 0
    dropped: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()
    thin: bool = False
    population_row_ids: tuple[str, ...] = ()
    selected_row_ids: tuple[str, ...] = ()
    contributing_row_ids: tuple[str, ...] = ()


def _refuse(code: str, detail: str) -> None:
    raise AnalogRefusal(code, detail)


def _reject_answers(
    value: Any,
    location: str,
    answer_fields: frozenset[str] = _ANSWER_FIELDS,
) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(str(key) for key in value if str(key) in answer_fields)
        if forbidden:
            _refuse(
                "PRECOMPUTED_ANALOG_ANSWER",
                f"{location} contains answer fields: {forbidden}",
            )
        for key, child in value.items():
            _reject_answers(child, f"{location}.{key}", answer_fields)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _reject_answers(child, f"{location}[{index}]", answer_fields)


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


def _bucket_value(value: Any, location: str) -> str | int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        _refuse("INVALID_ANALOG_SOURCE", f"{location} must be a scalar bucket")
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    _refuse("INVALID_ANALOG_SOURCE", f"{location} must be a scalar bucket")


def _normalize_bucket_rows(
    source_rows: Sequence[Mapping[str, Any]],
    bucket_dimensions: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    dimensions = tuple(bucket_dimensions)
    expected = frozenset(("row_id", "realized_return", *dimensions))
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(source_rows):
        if not isinstance(row, Mapping):
            _refuse("INVALID_ANALOG_SOURCE", f"row {index} must be a mapping")
        _reject_answers(
            row, f"source_rows[{index}]", _BUCKET_ANSWER_FIELDS,
        )
        unsupported = sorted(set(row) - expected)
        missing = sorted(expected - set(row))
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
        realized = row["realized_return"]
        normalized_return = (
            None
            if realized is None
            else _finite(realized, f"row {row_id} realized_return")
        )
        normalized.append({
            "row_id": row_id,
            **{
                dimension: _bucket_value(
                    row[dimension], f"row {row_id} bucket {dimension}",
                )
                for dimension in dimensions
            },
            "realized_return": normalized_return,
        })
    return tuple(sorted(normalized, key=lambda row: row["row_id"]))


def bucket_population_hash(
    source_rows: Sequence[Mapping[str, Any]],
    bucket_dimensions: Sequence[str] = LEGACY_BUCKET_DIMENSIONS,
) -> str:
    """Hash a normalized causal bucket population independent of row order."""
    dimensions = tuple(str(value) for value in bucket_dimensions)
    rows = _normalize_bucket_rows(source_rows, dimensions)
    return content_hash({
        "schema_version": BUCKET_POPULATION_SCHEMA,
        "bucket_dimensions": dimensions,
        "rows": rows,
    })


def _normalize_recipe(recipe: AnalogRecipe | Mapping[str, Any]) -> AnalogRecipe:
    if isinstance(recipe, Mapping):
        _reject_answers(recipe, "recipe", _BUCKET_ANSWER_FIELDS)
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


def _string_tuple(value: Any, location: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _refuse("INVALID_ANALOG_RECIPE", f"{location} must be a sequence")
    return tuple(str(item) for item in value)


def _bucket_recipe_from_mapping(recipe: Mapping[str, Any]) -> LegacyBucketRecipe:
    _reject_answers(recipe, "recipe")
    unsupported = sorted(set(recipe) - _BUCKET_RECIPE_FIELDS)
    missing = sorted((_BUCKET_RECIPE_FIELDS - {"schema_version"}) - set(recipe))
    if unsupported or missing:
        _refuse(
            "INVALID_ANALOG_RECIPE",
            f"recipe has missing={missing} unsupported={unsupported}",
        )
    quantiles = recipe["ci_quantiles"]
    if isinstance(quantiles, (str, bytes)) or not isinstance(quantiles, Sequence):
        _refuse("INVALID_ANALOG_RECIPE", "ci_quantiles must be a sequence")
    return LegacyBucketRecipe(
        bucket_dimensions=_string_tuple(
            recipe["bucket_dimensions"], "bucket_dimensions",
        ),
        widening_order=_string_tuple(
            recipe["widening_order"], "widening_order",
        ),
        min_analogs=recipe["min_analogs"],
        alpha=recipe["alpha"],
        bootstrap_draws=recipe["bootstrap_draws"],
        bootstrap_seed=recipe["bootstrap_seed"],
        ci_quantiles=tuple(quantiles),
        population_hash=str(recipe["population_hash"]),
        schema_version=str(recipe.get("schema_version", BUCKET_RECIPE_SCHEMA)),
    )


def _validate_bucket_dimensions(normalized: LegacyBucketRecipe) -> None:
    dimensions = normalized.bucket_dimensions
    if not dimensions or any(not name.strip() for name in dimensions):
        _refuse("INVALID_ANALOG_RECIPE", "bucket dimensions must be non-empty")
    if len(set(dimensions)) != len(dimensions):
        _refuse("INVALID_ANALOG_RECIPE", "bucket dimensions must be unique")
    widening = normalized.widening_order
    if len(set(widening)) != len(widening) or any(
        name not in dimensions for name in widening
    ):
        _refuse(
            "INVALID_ANALOG_RECIPE",
            "widening_order must contain unique bucket dimensions",
        )


def _validate_bucket_sampling(normalized: LegacyBucketRecipe) -> None:
    if isinstance(normalized.min_analogs, bool) or not isinstance(
        normalized.min_analogs, int
    ) or normalized.min_analogs <= 0:
        _refuse("INVALID_ANALOG_RECIPE", "min_analogs must be a positive integer")
    if isinstance(normalized.bootstrap_draws, bool) or not isinstance(
        normalized.bootstrap_draws, int
    ) or normalized.bootstrap_draws < 0:
        _refuse(
            "INVALID_ANALOG_RECIPE",
            "bootstrap_draws must be a non-negative integer",
        )
    if isinstance(normalized.bootstrap_seed, bool) or not isinstance(
        normalized.bootstrap_seed, int
    ) or not 0 <= normalized.bootstrap_seed < 2**64:
        _refuse(
            "INVALID_ANALOG_RECIPE",
            "bootstrap_seed must be an unsigned 64-bit integer",
        )


def _validate_bucket_statistics(
    normalized: LegacyBucketRecipe,
) -> tuple[float, tuple[float, float]]:
    alpha = _finite(normalized.alpha, "recipe alpha")
    if not 0.0 <= alpha <= 1.0:
        _refuse("INVALID_ANALOG_RECIPE", "alpha must be between zero and one")
    if len(normalized.ci_quantiles) != 2:
        _refuse("INVALID_ANALOG_RECIPE", "ci_quantiles must contain two values")
    low = _finite(normalized.ci_quantiles[0], "recipe ci_quantiles[0]")
    high = _finite(normalized.ci_quantiles[1], "recipe ci_quantiles[1]")
    if not 0.0 <= low <= high <= 1.0:
        _refuse("INVALID_ANALOG_RECIPE", "ci_quantiles are outside [0, 1]")
    if not normalized.population_hash.startswith("sha256:"):
        _refuse("INVALID_ANALOG_RECIPE", "population_hash must be a content hash")
    return alpha, (low, high)


def _validate_bucket_recipe(
    normalized: LegacyBucketRecipe,
) -> LegacyBucketRecipe:
    if normalized.schema_version != BUCKET_RECIPE_SCHEMA:
        _refuse(
            "INVALID_ANALOG_RECIPE",
            f"unsupported schema_version: {normalized.schema_version}",
        )
    _validate_bucket_dimensions(normalized)
    _validate_bucket_sampling(normalized)
    alpha, quantiles = _validate_bucket_statistics(normalized)
    return LegacyBucketRecipe(
        bucket_dimensions=normalized.bucket_dimensions,
        widening_order=normalized.widening_order,
        min_analogs=normalized.min_analogs,
        alpha=alpha,
        bootstrap_draws=normalized.bootstrap_draws,
        bootstrap_seed=normalized.bootstrap_seed,
        ci_quantiles=quantiles,
        population_hash=normalized.population_hash,
    )


def _normalize_bucket_recipe(
    recipe: LegacyBucketRecipe | Mapping[str, Any],
) -> LegacyBucketRecipe:
    if isinstance(recipe, Mapping):
        normalized = _bucket_recipe_from_mapping(recipe)
    elif isinstance(recipe, LegacyBucketRecipe):
        normalized = recipe
    else:
        _refuse(
            "INVALID_ANALOG_RECIPE",
            "bucket recipe must be a mapping or LegacyBucketRecipe",
        )
    return _validate_bucket_recipe(normalized)


def legacy_bucket_bootstrap_seed(
    *,
    snapshot: str,
    strategy: str,
    alpha: float,
    buckets: Mapping[str, Any],
    request_key: str,
) -> int:
    """Derive the legacy bootstrap seed from its frozen request identity."""
    payload = "|".join(
        [
            str(snapshot),
            str(strategy),
            f"{float(alpha):.4f}",
            str(request_key),
        ]
        + [f"{key}={buckets.get(key)}" for key in sorted(buckets)]
    )
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def _summarize_returns(
    returns: np.ndarray,
    *,
    min_analogs: int,
    bootstrap_draws: int,
    bootstrap_seed: int,
    ci_quantiles: tuple[float, float],
    dropped: tuple[str, ...],
    unavailable: tuple[str, ...],
    all_unavailable: bool,
    population_ids: tuple[str, ...],
    selected_ids: tuple[str, ...],
    contributing_ids: tuple[str, ...],
) -> AnalogResult:
    """Legacy ``AnalogMatcher._summarize`` over already-sorted finite returns.

    One arithmetic path for the declared-rows recipe and the frozen
    artifact: sorted before the bootstrap (it samples by index), the
    bootstrap seeded from the request identity, quantiles as legacy.
    """
    if returns.size == 0:
        return AnalogResult(
            exp_pnl_analog=None,
            win_analog=None,
            ci_low=None,
            ci_high=None,
            n_analogs=0,
            widened=0 if all_unavailable else len(dropped),
            dropped=dropped,
            unavailable=unavailable,
            thin=True,
            population_row_ids=population_ids,
            selected_row_ids=selected_ids,
            contributing_row_ids=contributing_ids,
        )
    thin = returns.size < min_analogs
    ci_low = ci_high = None
    if not thin and bootstrap_draws:
        rng = np.random.default_rng(bootstrap_seed)
        draws = rng.choice(
            returns,
            size=(bootstrap_draws, returns.size),
            replace=True,
        )
        means = draws.mean(axis=1)
        ci_low, ci_high = (
            float(np.quantile(means, ci_quantiles[0])),
            float(np.quantile(means, ci_quantiles[1])),
        )
    return AnalogResult(
        exp_pnl_analog=float(returns.mean()),
        win_analog=float((returns > 0).mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        n_analogs=int(returns.size),
        median=float(np.median(returns)),
        p10=float(np.quantile(returns, 0.10)),
        p90=float(np.quantile(returns, 0.90)),
        widened=len(dropped),
        dropped=dropped,
        unavailable=unavailable,
        thin=thin,
        population_row_ids=population_ids,
        selected_row_ids=selected_ids,
        contributing_row_ids=contributing_ids,
    )


def _summarize_bucket_match(
    *,
    rows: tuple[dict[str, Any], ...],
    selected: tuple[dict[str, Any], ...],
    recipe: LegacyBucketRecipe,
    dropped: tuple[str, ...],
    unavailable: tuple[str, ...],
) -> AnalogResult:
    contributing = tuple(
        row for row in selected if row["realized_return"] is not None
    )
    returns = np.sort(np.asarray(
        [row["realized_return"] for row in contributing], dtype=float,
    ))
    return _summarize_returns(
        returns,
        min_analogs=recipe.min_analogs,
        bootstrap_draws=recipe.bootstrap_draws,
        bootstrap_seed=recipe.bootstrap_seed,
        ci_quantiles=recipe.ci_quantiles,
        dropped=dropped,
        unavailable=unavailable,
        all_unavailable=len(unavailable) == len(recipe.bucket_dimensions),
        population_ids=tuple(row["row_id"] for row in rows),
        selected_ids=tuple(row["row_id"] for row in selected),
        contributing_ids=tuple(row["row_id"] for row in contributing),
    )


def _evaluate_bucket_analogs(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    query_features: Mapping[str, Any],
    recipe: LegacyBucketRecipe | Mapping[str, Any],
) -> AnalogResult:
    normalized_recipe = _normalize_bucket_recipe(recipe)
    rows = _normalize_bucket_rows(
        source_rows, normalized_recipe.bucket_dimensions,
    )
    actual_hash = content_hash({
        "schema_version": BUCKET_POPULATION_SCHEMA,
        "bucket_dimensions": normalized_recipe.bucket_dimensions,
        "rows": rows,
    })
    if actual_hash != normalized_recipe.population_hash:
        _refuse(
            "ANALOG_POPULATION_CORRUPT",
            "source population does not match recipe population_hash",
        )
    if not isinstance(query_features, Mapping):
        _refuse("INVALID_ANALOG_INPUT", "query_features must be a mapping")
    _reject_answers(query_features, "query_features", _BUCKET_ANSWER_FIELDS)
    dimensions = normalized_recipe.bucket_dimensions
    unsupported = sorted(set(query_features) - set(dimensions))
    missing = sorted(set(dimensions) - set(query_features))
    if unsupported or missing:
        _refuse(
            "INVALID_ANALOG_INPUT",
            f"query has missing={missing} unsupported={unsupported}",
        )
    query = {
        dimension: _bucket_value(
            query_features[dimension], f"query bucket {dimension}",
        )
        for dimension in dimensions
    }
    unavailable = tuple(
        dimension for dimension in dimensions if query[dimension] is None
    )
    if len(unavailable) == len(dimensions):
        return _summarize_bucket_match(
            rows=rows,
            selected=(),
            recipe=normalized_recipe,
            dropped=unavailable,
            unavailable=unavailable,
        )

    active = [dimension for dimension in dimensions if dimension not in unavailable]
    dropped = list(unavailable)
    while True:
        selected = tuple(
            row
            for row in rows
            if all(row[dimension] == query[dimension] for dimension in active)
        )
        remaining = [
            dimension
            for dimension in normalized_recipe.widening_order
            if dimension not in dropped
        ]
        if len(selected) >= normalized_recipe.min_analogs or not remaining:
            return _summarize_bucket_match(
                rows=rows,
                selected=selected,
                recipe=normalized_recipe,
                dropped=tuple(dropped),
                unavailable=unavailable,
            )
        dimension = remaining[0]
        active.remove(dimension)
        dropped.append(dimension)


def _evaluate_nearest_analogs(
    source_rows: Sequence[Mapping[str, Any]],
    query_features: Mapping[str, Any],
    recipe: AnalogRecipe | Mapping[str, Any],
) -> AnalogResult:
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


def _recipe_schema(recipe: Any) -> Any:
    if isinstance(recipe, (AnalogRecipe, LegacyBucketRecipe)):
        return recipe.schema_version
    if isinstance(recipe, Mapping):
        return recipe.get("schema_version", RECIPE_SCHEMA)
    return None


def evaluate_analogs(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    query_features: Mapping[str, Any],
    recipe: AnalogRecipe | LegacyBucketRecipe | Mapping[str, Any],
) -> AnalogResult:
    """Execute the versioned analog recipe against source-only rows."""
    if _recipe_schema(recipe) == BUCKET_RECIPE_SCHEMA:
        return _evaluate_bucket_analogs(
            source_rows=source_rows,
            query_features=query_features,
            recipe=recipe,
        )
    return _evaluate_nearest_analogs(source_rows, query_features, recipe)


# --------------------------------------------------------------------------
# the frozen board analog matcher (P5-4)
# --------------------------------------------------------------------------

#: Fields of a frozen-path recipe (``source_inputs.SourceBundle
#: .analog_artifact_recipe``). ``cutoff`` (the request's evidence cutoff) and
#: the optional ``content_hash`` (the release pin) are the artifact key the
#: stage checks; the rest is legacy ``AnalogMatcher.match``'s own arguments:
#: ``min_analogs``, ``bootstrap_draws`` (legacy ``bootstrap``), the CI
#: quantiles, and the two request-identity parts of the legacy bootstrap seed
#: (``seed_snapshot`` = the matcher's ``snapshot``, ``request_key``).
FROZEN_ANALOG_RECIPE_FIELDS = frozenset({
    "cutoff", "content_hash", "min_analogs", "bootstrap_draws", "ci_quantiles",
    "seed_snapshot", "request_key",
})
FROZEN_ANALOG_QUERY_FIELDS = ("mcap_bucket", "dte_band", "moneyness_band", "implied_ratio")
TERCILE_LABELS = ("low", "mid", "high")
MODEL_NOT_READY = "MODEL_NOT_READY"
_FROZEN_CACHE: dict[str, tuple[Any, ...]] = {}
_FROZEN_CACHE_LIMIT = 8


def implied_tercile(ratio: float | None, edges: Sequence[float]) -> str | None:
    """Legacy ``engine.analogs._bucket`` for one implied ratio.

    Half-open upward (``searchsorted(side="right")``), clipped to the three
    labels, and ``None`` for a missing or non-finite ratio.
    """
    if ratio is None or not isfinite(float(ratio)):
        return None
    index = int(np.searchsorted(np.asarray(edges, dtype=float), float(ratio), side="right"))
    return TERCILE_LABELS[min(max(index, 0), len(TERCILE_LABELS) - 1)]


def _frozen_arrays(artifact: Any) -> tuple[Any, ...]:
    """``(ids, {dimension: labels}, returns)`` of a frozen pool, cached by hash.

    Read-only arrays in the artifact's own order: a request can neither pay
    the conversion twice nor mutate what the next request reads.
    """
    cached = _FROZEN_CACHE.get(artifact.content_hash)
    if cached is not None:
        return cached
    from engine.v2.models.analog_artifact import BOARD_ANALOG_COLUMNS

    rows = artifact.rows
    column = {name: index for index, name in enumerate(BOARD_ANALOG_COLUMNS)}
    ids = np.asarray([row[0] for row in rows], dtype=object)
    labels = {
        dimension: np.asarray([row[column[dimension]] for row in rows], dtype=object)
        for dimension in LEGACY_BUCKET_DIMENSIONS
    }
    returns = np.asarray(
        [np.nan if row[column["ret"]] is None else row[column["ret"]] for row in rows],
        dtype=float,
    )
    for array in (ids, returns, *labels.values()):
        array.setflags(write=False)
    cached = (ids, labels, returns)
    if len(_FROZEN_CACHE) >= _FROZEN_CACHE_LIMIT:
        _FROZEN_CACHE.pop(next(iter(_FROZEN_CACHE)))
    _FROZEN_CACHE[artifact.content_hash] = cached
    return cached


def _frozen_recipe(recipe: Any) -> dict[str, Any]:
    if not isinstance(recipe, Mapping):
        _refuse("INVALID_ANALOG_RECIPE", "frozen analog recipe must be a mapping")
    _reject_answers(recipe, "analog_artifact_recipe", _BUCKET_ANSWER_FIELDS)
    unsupported = sorted(set(recipe) - FROZEN_ANALOG_RECIPE_FIELDS)
    missing = sorted((FROZEN_ANALOG_RECIPE_FIELDS - {"content_hash"}) - set(recipe))
    if unsupported or missing:
        _refuse("INVALID_ANALOG_RECIPE",
                f"frozen recipe has missing={missing} unsupported={unsupported}")
    quantiles = recipe["ci_quantiles"]
    if isinstance(quantiles, (str, bytes)) or not isinstance(quantiles, Sequence):
        _refuse("INVALID_ANALOG_RECIPE", "ci_quantiles must be a sequence")
    checked = _validate_bucket_recipe(LegacyBucketRecipe(
        bucket_dimensions=LEGACY_BUCKET_DIMENSIONS, widening_order=LEGACY_WIDENING_ORDER,
        min_analogs=recipe["min_analogs"], alpha=0.5,
        bootstrap_draws=recipe["bootstrap_draws"], bootstrap_seed=0,
        ci_quantiles=tuple(quantiles), population_hash="sha256:frozen",
    ))
    return {"min_analogs": checked.min_analogs, "bootstrap_draws": checked.bootstrap_draws,
            "ci_quantiles": checked.ci_quantiles,
            "seed_snapshot": str(recipe["seed_snapshot"]),
            "request_key": str(recipe["request_key"])}


def _frozen_query(query: Any) -> dict[str, Any]:
    if not isinstance(query, Mapping):
        _refuse("INVALID_ANALOG_INPUT", "query_features must be a mapping")
    _reject_answers(query, "query_features", _BUCKET_ANSWER_FIELDS)
    unsupported = sorted(set(query) - set(FROZEN_ANALOG_QUERY_FIELDS))
    missing = sorted(set(FROZEN_ANALOG_QUERY_FIELDS) - set(query))
    if unsupported or missing:
        _refuse("INVALID_ANALOG_INPUT",
                f"frozen query has missing={missing} unsupported={unsupported}")
    out: dict[str, Any] = {}
    for dimension in ("mcap_bucket", "dte_band", "moneyness_band"):
        value = query[dimension]
        if value is not None and not isinstance(value, str):
            _refuse("INVALID_ANALOG_INPUT", f"query bucket {dimension} must be a label")
        out[dimension] = value
    ratio = query["implied_ratio"]
    out["implied_ratio"] = None if ratio is None else _finite(ratio, "query implied_ratio")
    return out


def frozen_key_mismatch(artifact: Any, recipe: Any, strategy: Any, alpha: Any) -> bool:
    """True unless ``artifact`` is the board analog pool for THIS request.

    The full causal key ``(strategy, alpha at 4dp, cutoff)`` is the request's
    own -- strategy and fill alpha from the resolved request, the evidence
    cutoff from the recipe -- and a release pin (``content_hash``), when
    given, must match too. A missing key part never matches.
    """
    from engine.v2.models.analog_artifact import BoardAnalogPoolArtifact, board_analog_pool_key

    if (not isinstance(artifact, BoardAnalogPoolArtifact) or not isinstance(recipe, Mapping)
            or strategy is None or alpha is None or "cutoff" not in recipe):
        return True
    try:
        expected = board_analog_pool_key(strategy, alpha, recipe["cutoff"])
    except (TypeError, ValueError):
        return True
    if artifact.key != expected:
        return True
    pinned = recipe.get("content_hash")
    return pinned is not None and pinned != artifact.content_hash


def evaluate_frozen_analogs(
    *,
    artifact: Any,
    recipe: Mapping[str, Any],
    query_features: Mapping[str, Any],
    strategy: str | None,
    alpha: float | None,
) -> tuple[AnalogResult | None, str | None]:
    """Legacy ``AnalogMatcher.match`` read from a frozen causal pool.

    ``(result, None)``, or ``(None, "MODEL_NOT_READY")`` when the artifact is
    missing or its key/pin disagrees with this request -- never a rebuild,
    never a fallback to declared rows. The request's implied ratio is
    bucketed on the artifact's own frozen edges (legacy's causal
    re-bucketing), and the bootstrap seed is derived from the re-bucketed
    query exactly as legacy ``_seed`` derives it.
    """
    if frozen_key_mismatch(artifact, recipe, strategy, alpha):
        return None, MODEL_NOT_READY
    config = _frozen_recipe(recipe)
    buckets = _frozen_query(query_features)
    buckets["implied_tercile"] = implied_tercile(buckets["implied_ratio"],
                                                 artifact.request_edges)
    seed = legacy_bucket_bootstrap_seed(
        snapshot=config["seed_snapshot"], strategy=str(strategy), alpha=float(alpha),
        buckets=buckets, request_key=config["request_key"],
    )
    ids, labels, returns = _frozen_arrays(artifact)
    unavailable = tuple(d for d in ("mcap_bucket", *LEGACY_WIDENING_ORDER)
                        if buckets.get(d) is None)
    common = dict(min_analogs=config["min_analogs"],
                  bootstrap_draws=config["bootstrap_draws"], bootstrap_seed=seed,
                  ci_quantiles=config["ci_quantiles"], unavailable=unavailable,
                  population_ids=())
    if len(unavailable) == len(LEGACY_BUCKET_DIMENSIONS):
        return _summarize_returns(
            np.empty(0), dropped=unavailable, all_unavailable=True,
            selected_ids=(), contributing_ids=(), **common), None
    active = [d for d in ("mcap_bucket", *LEGACY_WIDENING_ORDER) if d not in unavailable]
    dropped = list(unavailable)
    while True:
        mask = np.ones(len(ids), dtype=bool)
        for dimension in active:
            mask &= labels[dimension] == buckets[dimension]
        remaining = [d for d in LEGACY_WIDENING_ORDER if d not in dropped]
        if int(mask.sum()) >= config["min_analogs"] or not remaining:
            selected = returns[mask]
            finite = np.isfinite(selected)
            return _summarize_returns(
                np.sort(selected[finite]), dropped=tuple(dropped), all_unavailable=False,
                selected_ids=tuple(ids[mask]), contributing_ids=tuple(ids[mask][finite]),
                **common), None
        dimension = remaining[0]
        active.remove(dimension)
        dropped.append(dimension)
