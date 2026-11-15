"""Fixed L/LP derived-feature lists and the pure derive_columns step.

No caller yet by design: slices 2-4 of guides/size_log_product_features_design.md
(section 8) add the HMOSXL/HMOSXLP recipes, request v12 and the fit-path call at
the panel-to-matrix seam; this slice only fixes the lists and refusal semantics.
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np

from engine.v2.data.errors import fail

LOG_COLUMNS = (
    "mean_prior_abs_move", "ema2_prior_abs_move", "ema4_prior_abs_move",
    "ema8_prior_abs_move", "ema12_prior_abs_move", "implied_move",
    "iv10", "iv30", "exern_iv10", "exern_iv30", "fwd90_30",
    "fexern90_30", "rvol30", "spy_vol5", "spy_vol20", "spy_vol60",
    "season_heat_mean_abs_5", "season_heat_mean_abs_20",
    "season_heat_ratio_20_252", "tail_max_abs", "tail_std_abs",
    "tail_p75_abs", "tail_p90_abs", "tail_share_gt10", "tail_max4_abs",
    "tail_std_over_mean", "tail_max_over_mean", "iv10_30", "iv_rv",
    "ex_ratio", "imp_chg5", "imp_chg10", "iv30_chg5", "iv30_chg20",
    "iv10_chg5", "own_rvol5", "own_rvol10", "own_rvol60", "own_absret1",
)
PRODUCT_BASE = (
    "implied_move", "mean_prior_abs_move", "ema4_prior_abs_move",
    "tail_p90_abs", "tail_max_abs", "rvol30", "iv30",
    "season_heat_mean_abs_20",
)
PRODUCT_PAIRS = tuple(combinations(PRODUCT_BASE, 2))
SET_NAMES = ("HMOSXL", "HMOSXLP")


def derived_names(set_name: str) -> tuple[str, ...]:
    """Return the output column names for a derived-feature set."""
    if set_name not in SET_NAMES:
        raise fail("CONTRACT_MISMATCH", f"unknown derived-feature set {set_name!r}")
    names = tuple(f"log1p_{column}" for column in LOG_COLUMNS)
    if set_name == "HMOSXLP":
        names += tuple(f"{left}_x_{right}" for left, right in PRODUCT_PAIRS)
    return names


def _source(columns: Mapping[str, np.ndarray], name: str, expected: int | None) -> np.ndarray:
    """Validate and return one float64 source column."""
    if name not in columns:
        raise fail("CONTRACT_MISMATCH", f"derived source {name} is missing")
    values = columns[name]
    if not isinstance(values, np.ndarray) or values.ndim != 1 or values.dtype.kind != "f":
        raise fail("CONTRACT_MISMATCH", f"derived source {name} must be a 1-D float array")
    if expected is not None and values.shape[0] != expected:
        raise fail("CONTRACT_MISMATCH",
                   f"derived source {name} length {values.shape[0]} does not match {expected}")
    if np.isinf(values).any():
        raise fail("CONTRACT_MISMATCH", f"derived source {name} contains a non-null infinite value")
    return np.asarray(values, dtype=np.float64)


def derive_columns(columns: Mapping[str, np.ndarray], set_name: str) -> dict[str, np.ndarray]:
    """Return log1p and optional product columns for the given set."""
    names = derived_names(set_name)
    sources: dict[str, np.ndarray] = {}
    expected = None
    for name in LOG_COLUMNS:
        sources[name] = _source(columns, name, expected)
        if expected is None:
            expected = sources[name].shape[0]
    output: dict[str, np.ndarray] = {}
    for name in LOG_COLUMNS:
        values = sources[name]
        if np.any(values[~np.isnan(values)] < 0):
            raise fail("CONTRACT_MISMATCH", f"derived source {name} contains a negative value")
        output[f"log1p_{name}"] = np.log1p(values)
    if set_name == "HMOSXLP":
        with np.errstate(over="ignore", invalid="ignore"):
            for left, right in PRODUCT_PAIRS:
                product = sources[left] * sources[right]
                if np.any(np.isfinite(sources[left]) & np.isfinite(sources[right])
                          & ~np.isfinite(product)):
                    raise fail("EXPERIMENT_VARIANT_FAILED",
                               f"derived product {left}_x_{right} is not finite")
                output[f"{left}_x_{right}"] = product
    return {name: output[name] for name in names}
