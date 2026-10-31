"""Query and declaration helpers for the nightly frozen board analog matcher (#681).

Pure arithmetic, no I/O. The bucket edges and the implied-ratio rule are the
legacy ``engine.analogs`` ones (``MCAP_EDGES``, ``DTE_BANDS``,
``MONEYNESS_BANDS``; ``engine.score.Scorer._score_analogs``), restated here
because this module imports no legacy code.
"""
from __future__ import annotations

from math import exp, isfinite
from typing import Any, Mapping, Sequence

MCAP_EDGES = (1e9, 1e10)
MCAP_LABELS = ("<1B", "1-10B", ">=10B")
DTE_BANDS = ((1, 3), (4, 10), (11, 25), (26, 45))
DTE_LABELS = ("1-3", "4-10", "11-25", "26-45")
MONEYNESS_EDGES = (2.0, 5.0)
MONEYNESS_LABELS = ("ATM", "2-5%", ">5%")
#: Legacy ``AnalogMatcher.match`` defaults (``MIN_ANALOGS``, ``bootstrap``,
#: the 5%/95% quantiles of ``_summarize``).
MIN_ANALOGS = 30
BOOTSTRAP_DRAWS = 2000
CI_QUANTILES = (0.05, 0.95)


def _finite_or_none(value: Any) -> float | None:
    """``value`` as a finite float, or ``None`` (bool and non-numbers are None)."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _label(value: float | None, edges: Sequence[float], labels: Sequence[str]) -> str | None:
    """Legacy ``_bucket``: half-open upward, ``None`` for a missing value."""
    if value is None:
        return None
    return labels[sum(1 for edge in edges if edge <= value)]


def analog_query_base(panel_row: Mapping[str, Any]) -> dict[str, Any]:
    """The bundle-time part of the query: ``mcap_bucket`` and ``implied_ratio``.

    ``mcap_bucket`` buckets ``exp(mcap_log)``; ``implied_ratio`` is
    ``entry_im / mean_prior_or_implied`` (``entry_im`` is the decision-date
    implied move, not the model feature ``im``) when both are finite and the
    divisor is non-zero, else ``None``.
    """
    log = _finite_or_none(panel_row.get("mcap_log"))
    mcap = None
    if log is not None:
        try:
            mcap = exp(log)
        except OverflowError:
            mcap = None
    implied = _finite_or_none(panel_row.get("entry_im"))
    prior = _finite_or_none(panel_row.get("mean_prior_or_implied"))
    ratio = implied / prior if implied is not None and prior else None
    return {"mcap_bucket": _label(mcap, MCAP_EDGES, MCAP_LABELS),
            "implied_ratio": ratio if ratio is not None and isfinite(ratio) else None}


def complete_analog_query(query: Any, values: Mapping[str, Any]) -> Any:
    """``query`` plus the priced-leg bands, for a query lacking those keys.

    ``dte_band`` comes from ``values["dte_entry"]``; ``moneyness_band`` from
    ``abs(strike / spot - 1) * 100``. An absent, non-finite or (strike, spot)
    non-positive input gives ``None`` (the matcher then drops that dimension).
    A key already present is kept, and a non-mapping is returned unchanged.
    """
    if not isinstance(query, Mapping):
        return query
    out = dict(query)
    if "dte_band" not in out:
        dte = _finite_or_none(values.get("dte_entry"))
        out["dte_band"] = next(
            (label for (low, high), label in zip(DTE_BANDS, DTE_LABELS)
             if dte is not None and low <= dte <= high), None)
    if "moneyness_band" not in out:
        strike = _finite_or_none(values.get("strike"))
        spot = _finite_or_none(values.get("spot"))
        usable = strike is not None and spot is not None and strike > 0 and spot > 0
        pct = abs(strike / spot - 1.0) * 100.0 if usable else None
        if pct is not None and not isfinite(pct):
            pct = None  # an overflowed ratio is unavailable, never the ">5%" bucket
        out["moneyness_band"] = _label(pct, MONEYNESS_EDGES, MONEYNESS_LABELS)
    return out


class AmbiguousAnalogPool(ValueError):
    """More than one frozen pool carries the row's (strategy, alpha, cutoff) key."""


def analog_declarations(
    artifacts: Sequence[Any], strategy: str, alpha: float, cutoff: str,
    panel_row: Mapping[str, Any], snapshot: str, request_key: str,
) -> dict[str, Any]:
    """The SourceBundle analog fields for the pool matching this row's key.

    The pool is picked by ``board_analog_pool_key(strategy, alpha, cutoff)``
    and pinned by its ``content_hash``. No matching pool gives ``{}`` (the
    analog block stays undeclared); more than one raises
    :class:`AmbiguousAnalogPool` (never a first-wins pick).
    """
    # Lazy: engine.v2.ops imports this module and keeps engine.v2.models off its top-level import chain.
    from engine.v2.models.analog_artifact import board_analog_pool_key

    key = board_analog_pool_key(strategy, alpha, cutoff)
    matches = [artifact for artifact in artifacts if artifact.key == key]
    if len(matches) > 1:
        raise AmbiguousAnalogPool("more than one analog pool matches the row key")
    if not matches:
        return {}
    artifact = matches[0]
    return {
        "analog_artifact": artifact,
        "analog_artifact_recipe": {
            "cutoff": cutoff, "content_hash": artifact.content_hash,
            "min_analogs": MIN_ANALOGS, "bootstrap_draws": BOOTSTRAP_DRAWS,
            "ci_quantiles": CI_QUANTILES, "seed_snapshot": snapshot,
            "request_key": request_key,
        },
        "analog_query": analog_query_base(panel_row),
    }
