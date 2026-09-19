"""Fit win-rate recalibration-map artifacts from causal pairs (P5-4).

The training-side builder for :mod:`engine.v2.models.recalibration_artifact`,
the recalibration twin of :mod:`.payoff`. :func:`fit_recalibration_map` is
legacy ``engine.recalibrate.fit_recalibration`` statement for statement --
the same strategy/``np.isclose`` alpha filter, ``exit_date < before`` causal
cut, ``dropna`` on ``raw_win``/``outcome``, ``min_pairs`` floor and
``IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")`` -- so the
frozen artifact carries the thresholds legacy would fit on the same pairs,
bit for bit (``tests/test_v2_models_recalibration_artifact.py`` fits both and
compares). It is re-derived here rather than imported because v2 may not
import legacy numerics (``checks/import_layers.py``); the P5-3 native
estimators follow the same rule.

Layer 6 (``engine.v2.models.training``): nothing in scoring, features or
``engine.v2.models`` imports this module.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from engine.v2.models.no_fit import forbid_fitting as forbid_v2_fitting
from engine.v2.models.recalibration_artifact import (
    RecalibrationMapArtifact,
    make_recalibration_map_artifact,
)

from .legacy_adapter import forbid_fitting

__all__ = ["MIN_PAIRS", "build_recalibration_map_artifact", "fit_recalibration_map"]

#: ``engine.recalibrate.MIN_PAIRS``: below this many closed pairs a map is noise.
MIN_PAIRS = 120


def _eligible(pairs: pd.DataFrame, strategy: str, alpha: float, before: Any) -> pd.DataFrame:
    stamp = pd.Timestamp(before).normalize() if before is not None else None
    rows = pairs[
        (pairs["strategy"] == strategy)
        & np.isclose(pairs["fill_alpha"].astype(float), float(alpha))
    ]
    if stamp is not None:
        rows = rows[pd.to_datetime(rows["exit_date"]) < stamp]
    return rows.dropna(subset=["raw_win", "outcome"])


def fit_recalibration_map(
    pairs: pd.DataFrame, strategy: str, alpha: float, *, before: Any,
    min_pairs: int = MIN_PAIRS,
) -> dict[str, Any] | None:
    """Legacy ``fit_recalibration``'s map as plain fields, or its ``None``.

    Both no-fit switches are checked first: the legacy one every guarded
    scoring path sets, and the v2-native one.
    """
    forbid_fitting("engine.v2.models.training.recalibration.fit_recalibration_map")
    forbid_v2_fitting("engine.v2.models.training.recalibration.fit_recalibration_map")
    if pairs.empty:
        return None
    rows = _eligible(pairs, strategy, alpha, before)
    if len(rows) < min_pairs:
        return None

    from sklearn.isotonic import IsotonicRegression

    x = rows["raw_win"].to_numpy(dtype=float)
    y = rows["outcome"].to_numpy(dtype=float)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(x, y)
    return {
        "n": int(len(rows)),
        "base_rate": float(y.mean()),
        "x_thresholds": np.asarray(iso.X_thresholds_, dtype=float),
        "y_thresholds": np.asarray(iso.y_thresholds_, dtype=float),
    }


def _window(pairs: pd.DataFrame, strategy: str, alpha: float, before: Any):
    if pairs.empty:
        return None
    dates = pd.to_datetime(_eligible(pairs, strategy, alpha, before)["exit_date"]).dropna()
    if dates.empty:
        return None
    return str(dates.min().date()), str(dates.max().date())


def build_recalibration_map_artifact(
    pairs: pd.DataFrame, *, strategy: str, alpha: float, before: Any = None,
    min_pairs: int = MIN_PAIRS,
) -> RecalibrationMapArtifact:
    """Fit and freeze one map; below ``min_pairs`` freeze legacy's "no map".

    Never ``None``: a cutoff with too few closed pairs still gets an
    artifact (``fitted=False``) so scoring can tell "legacy ships the raw
    probability here" from "this fold was never built" (MODEL_NOT_READY).
    """
    fit = fit_recalibration_map(pairs, strategy, alpha, before=before, min_pairs=min_pairs)
    window = _window(pairs, strategy, alpha, before) if fit is not None else None
    return make_recalibration_map_artifact(
        fit, strategy=strategy, alpha=alpha, cutoff=before, min_pairs=min_pairs, window=window,
    )
