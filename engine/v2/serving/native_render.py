"""Native ``ScoreRecord`` -> flat serving display row for the board projection.

The native scoring kernel (:mod:`engine.v2.scoring.application`) emits a
:class:`~engine.v2.contracts.ScoreRecord`; the serving projection
(:mod:`engine.v2.serving.projections`) reads a flat display row through
``display.get(...)`` exactly as it reads one produced by
``engine.v2.serving.bridge`` from a legacy ``score.json`` row. This module is
that translator: it copies already-computed values into the flat field names
the projection checks -- no financial arithmetic, no filesystem, no clock,
mirroring ``bridge.py``'s own rule.

Layer discipline (guide §4.2): ``engine/v2/serving`` is layer 7, so this file
imports only layer-0 contracts, layer-5 scoring stages and the stdlib. It
never imports ``engine.v2.ops`` (a layer-7 peer) or legacy ``engine.*``; the
accidental-import proof is ``checks/import_layers.py``, not a runtime check.

Field sourcing follows the real record shape. ``ScoreRecord.resolved_request``
holds the native kernel's own ``values`` mapping verbatim
(``application.score_one`` records it as both ``values`` and ``legacy_fields``),
so the stage-owned outputs -- ``entry_cost``, ``exp_pnl_model`` /
``exp_pnl_analog`` / ``exp_pnl_sim``, ``implied_move``, ``detail`` and
``n_analogs`` -- are copied from there (``financial_diagnostics`` / ``forecasts``
are consulted first where the record carries the name there, per the display
mapping contract). Identity fields read their own named record attribute
(``strategy``, ``gate_pass``, the chooser selection).

Every name in :data:`NATIVE_NEVER_COMPUTED` is deliberately ABSENT from the
row rather than present as ``None``: a ``display.get`` reader cannot tell the
two apart, and the row's own ``forecasts``/``uncertainty`` failing loudly on a
non-``None`` value is what keeps the "explicitly missing" guarantee true the
day a stage starts computing the band (G3).
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from engine.v2.contracts import ScoreRecord
from engine.v2.scoring.stages import (
    DISPLAY_ANALOG_FIELDS,
    StageObservation,
    analog_display_fields,
)

__all__ = [
    "NATIVE_NEVER_COMPUTED",
    "join_analog_fields_by_request",
    "native_display_row",
    "native_row_key",
]

#: The forecast/driver band v2 does not compute today (memory
#: ``native-forecast-band-missing``); a row advertises these as missing by
#: omitting the key, never by inventing a value.
NATIVE_NEVER_COMPUTED = (
    "forecast_p10", "forecast_p90", "forecast_sd", "driver_p10", "driver_p90",
)


def _reject_computed_band(record: ScoreRecord) -> None:
    """Refuse a record that already carries a value for a never-computed band.

    Only a non-``None`` value counts as computed: ``uncertainty`` always carries
    the ``forecast_p10``/``forecast_p90``/``forecast_sd`` keys (with ``None``
    values) because ``application._record_payload`` projects them by name, and a
    key-presence test here would refuse every real record. A future stage that
    starts filling one is a hard failure -- silently continuing would let the
    row report a computed band as missing (G3).
    """
    computed = sorted(
        name for name in NATIVE_NEVER_COMPUTED
        if record.forecasts.get(name) is not None
        or record.uncertainty.get(name) is not None
    )
    if computed:
        raise ValueError(
            f"native ScoreRecord already carries computed never-computed "
            f"field(s): {computed}")


def _strategy(resolved: Mapping[str, Any]) -> Any:
    """The resolved strategy name: the explicit version when recorded, else the
    native kernel's own ``strategy`` context fact (what ``score_one`` stores)."""
    strategy = resolved.get("strategy_version")
    if strategy is None:
        strategy = resolved.get("strategy")
    return strategy


def _field(primary: Mapping[str, Any], name: str,
           fallback: Mapping[str, Any] | None = None) -> Any:
    """``primary.get(name)`` when the key is there, else ``fallback.get(name)``.

    Key membership, not truthiness: a present ``None`` is a real "computed to
    nothing" answer and must not fall through to another mapping.
    """
    if name in primary:
        return primary.get(name)
    if fallback is None:
        return None
    return fallback.get(name)


def native_display_row(
    record: ScoreRecord, *, analog_fields: Mapping[str, tuple] | None = None,
) -> dict:
    """Build the flat display row for one native :class:`ScoreRecord`.

    ``analog_fields`` is the S9H display channel
    (:func:`~engine.v2.scoring.stages.analog_display_fields`) for this record's
    own ``score_one`` call. ``None`` is a first-class input -- every non-analog
    strategy and every pre-S9H caller passes it -- and omits ``n_analogs``:
    absent means the analog stage did not run, an int ``0`` means it ran and
    matched nothing (memory ``n-analogs-int-default-mismatch``), and this
    function never collapses the two.
    """
    _reject_computed_band(record)
    resolved = record.resolved_request
    chooser = record.chooser_selection or {}
    warnings = list(record.warnings)
    reasons = list(record.reason_codes)
    row: dict[str, Any] = {
        "strategy": _strategy(resolved),
        "gate_pass": record.gate_terms.get("gate_pass"),
        "detail": _field(record.gate_terms, "detail", resolved),
        "driver_prediction": record.forecasts.get("driver_prediction"),
        "implied_move": _field(record.forecasts, "implied_move", resolved),
        "entry_cost": _field(record.financial_diagnostics, "entry_cost", resolved),
        "exp_pnl_model": _field(record.financial_diagnostics, "exp_pnl_model", resolved),
        "exp_pnl_analog": _field(record.financial_diagnostics, "exp_pnl_analog", resolved),
        "exp_pnl_sim": _field(record.financial_diagnostics, "exp_pnl_sim", resolved),
        "chosen_strategy": chooser.get("chosen_strategy"),
        "chosen_margin": chooser.get("chosen_margin"),
        "menu_size": chooser.get("menu_size"),
        "flags": warnings + reasons,
        "warnings": warnings,
        "reason_codes": reasons,
    }
    analog = dict(analog_fields) if analog_fields else {}
    if analog_fields:
        n_analogs = analog.get("n_analogs", resolved.get("n_analogs"))
        if n_analogs is not None:
            row["n_analogs"] = n_analogs
    for name in DISPLAY_ANALOG_FIELDS:
        row[name] = tuple(analog.get(name, ()))
    return row


def native_row_key(record: ScoreRecord) -> str:
    """The native twin of ``bridge._population_key`` for one scored record.

    ``"|".join(ticker, strategy, event_date)`` -- no strike/expiry -- so a
    native row lands in the same ``rendered_index``/``bundle_rows_by_ticker``
    join ``bridge._bridge_one`` performs. ``ticker``/``event_date`` come
    strictly from ``record.event_ref``: a missing field is a malformed record
    (upstream never resolved the event), not an ordinary unmapped one, so it
    raises a named ``KeyError`` rather than inventing ``""``/``"unknown"`` and
    silently colliding two events.
    """
    event_ref = record.event_ref
    for name in ("ticker", "event_date"):
        if name not in event_ref:
            raise KeyError(f"native ScoreRecord.event_ref missing {name!r}")
    strategy = _strategy(record.resolved_request)
    return "|".join((
        str(event_ref["ticker"]),
        "" if strategy is None else str(strategy),
        str(event_ref["event_date"]),
    ))


def join_analog_fields_by_request(
    observations_by_request: Mapping[str, Iterable[StageObservation]],
) -> dict[str, dict]:
    """Shape one analog display mapping per request's own observations.

    Thin wrapper over :func:`~engine.v2.scoring.stages.analog_display_fields`.
    The caller keys ``observations_by_request`` by the same key
    :func:`native_row_key` derives, so one ``score_one(..., observer=...)``
    call's observations join back onto that request's own row and never
    another's -- no cross-row matching lives here.
    """
    return {
        request_key: analog_display_fields(observations)
        for request_key, observations in observations_by_request.items()
    }
