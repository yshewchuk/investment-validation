"""Spline-specific ElasticNet fit helpers kept out of the main fitter."""
from __future__ import annotations

import resource
import sys
import time

from engine.v2.data.errors import fail
from engine.v2.models.magnitude_elasticnet_recipe import (
    validate_elasticnet_recipe,
)
from engine.v2.models.magnitude_elasticnet_spline_recipe import (
    spline_coefficient_layout,
    validate_elasticnet_spline_recipe,
)

_SPLINE_FITS = 0


def log_spline_fit(recipe, elapsed):
    global _SPLINE_FITS
    _SPLINE_FITS += 1
    if _SPLINE_FITS == 1 or _SPLINE_FITS % 100 == 0:
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        n_knots = recipe["preprocessing"]["spline"]["n_knots"]
        inputs = recipe["preprocessing"]["output_features"]
        sys.stderr.write(
            f"[spline-fit] fit={_SPLINE_FITS} spline_knots={n_knots} inputs={inputs} "
            f"elapsed={elapsed:.2f}s rss_mb={rss_mb:.1f}\n")


def is_spline(recipe):
    return recipe["schema_version"].startswith("magnitude_elasticnet_spline")


def select_validator(recipe):
    return (validate_elasticnet_spline_recipe
            if type(recipe) is dict
            and recipe.get("schema_version") == "magnitude_elasticnet_spline_recipe.v1"
            else validate_elasticnet_recipe)


def coefficient_pairs(recipe, columns):
    if is_spline(recipe):
        return spline_coefficient_layout(recipe)
    return [(kind, column) for kind in ("value", "missing_indicator") for column in columns]


def expected_width(recipe, columns):
    return (recipe["preprocessing"]["output_features"] if is_spline(recipe)
            else 2 * len(columns))


def convergence_error(recipe):
    if is_spline(recipe):
        return fail("EXPERIMENT_VARIANT_FAILED",
                    "ElasticNet convergence failed; no retry permitted")
    return ValueError("ElasticNet convergence failed; no retry permitted")


def timed_fit(model, train, labels):
    started = time.perf_counter()
    model.fit(train, labels)
    return time.perf_counter() - started
