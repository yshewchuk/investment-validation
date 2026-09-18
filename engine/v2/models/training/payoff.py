"""Fit payoff-calibration artifacts from causal source rows (P5-4).

The only production callers of ``native_payoff.fit_payoff_line``/
``fit_runup_payoff_surface`` that are allowed to exist: this package is
layer 6 (``engine.v2.models.training``), whose one job vs. layer 3
(``engine.v2.models``) is "fit anything" -- ``engine/v2/models`` itself must
not (``checks/layer_map.py``). Reuses ``native_payoff``'s fitting math
UNCHANGED (no new arithmetic here), which is what makes the artifact this
produces bit-identical to today's inline scoring-time fit on the same rows
and cutoff -- see ``tests/test_v2_models_training_payoff.py``'s parity tests.

Layering: layer 6 may import layer 5 (``engine.v2.scoring``) and layer 3
(``engine.v2.models``), both strictly lower (``checks/layer_map.py``); the
reverse is forbidden, which is exactly the "no training import can point
upward into a score request" rule in
``guides/rearchitecture_phase5_models.md``. Neither ``engine.v2.scoring`` nor
``engine.v2.models`` imports this module.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from engine.v2.models.payoff_artifact import (
    PayoffLineArtifact,
    PayoffSurfaceArtifact,
    make_payoff_line_artifact,
    make_payoff_surface_artifact,
)
from engine.v2.scoring import native_payoff

__all__ = ["build_payoff_line_artifact", "build_payoff_surface_artifact"]


def _causal_window(
    rows: Sequence[Mapping[str, Any]] | None, before: Any,
) -> tuple[str | None, str | None]:
    """Earliest/latest ``exit_date`` among rows that would actually enter the fit.

    Mirrors the exact causal filter ``native_payoff.fit_payoff_line``/
    ``fit_runup_payoff_surface`` apply (``exit_date < before``) so the
    recorded provenance window describes what was really fitted, not the
    raw row set handed in -- a row on or after ``before`` never enters this
    window either, the same as it never enters the fit.
    """
    cutoff = None if before is None else str(before)[:10]
    dates: list[str] = []
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        raw = row.get("exit_date")
        if raw is None:
            continue
        text = str(raw)[:10]
        if cutoff is not None and not (text < cutoff):
            continue
        dates.append(text)
    if not dates:
        return None, None
    return min(dates), max(dates)


def build_payoff_line_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    strategy: str,
    driver: str,
    alpha: float,
    before: Any = None,
    min_trades: int = native_payoff.MIN_TRADES,
    max_residuals: int = native_payoff.MAX_RESIDUALS,
    residual_seed: int = native_payoff.RESIDUAL_SEED,
) -> PayoffLineArtifact | None:
    """Fit and freeze one single-driver payoff line (STR-THRU and peers).

    ``None`` when ``native_payoff.fit_payoff_line`` itself returns ``None``
    -- too few closed trades before ``before`` to calibrate, exactly as
    legacy's ``PayoffError``/today's ``NO_PAYOFF_MAP`` path. Causal: rows
    dated on or after ``before`` are excluded by ``fit_payoff_line`` itself,
    never reaching the fit or the recorded provenance window.
    """
    fit = native_payoff.fit_payoff_line(
        rows, before=before, min_trades=min_trades,
        max_residuals=max_residuals, residual_seed=residual_seed,
    )
    if fit is None:
        return None
    window = _causal_window(rows, before)
    return make_payoff_line_artifact(
        fit, strategy=strategy, driver=driver, alpha=alpha, cutoff=before,
        window=window,
    )


def build_payoff_surface_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    strategy: str = "STR-RUNUP",
    alpha: float,
    before: Any = None,
    min_trades: int = native_payoff.MIN_TRADES,
    max_residuals: int = native_payoff.MAX_RESIDUALS,
    residual_seed: int = native_payoff.RESIDUAL_SEED,
) -> PayoffSurfaceArtifact | None:
    """Fit and freeze the STR-RUNUP two-driver exit-value surface (R4-17)."""
    fit = native_payoff.fit_runup_payoff_surface(
        rows, before=before, min_trades=min_trades,
        max_residuals=max_residuals, residual_seed=residual_seed,
    )
    if fit is None:
        return None
    window = _causal_window(rows, before)
    return make_payoff_surface_artifact(
        fit, strategy=strategy, alpha=alpha, cutoff=before, window=window,
    )
