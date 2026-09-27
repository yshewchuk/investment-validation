"""Per-night SourceBundle assembler for one (ticker, event) from already-staged inputs."""
from __future__ import annotations

import math
import numbers
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.v2.scoring.source_inputs import SourceBundle, _reject_answers

__all__ = [
    "NightlySourceBundleRefusal",
    "assemble_nightly_source_bundle",
    "quote_domain_map",
    "validated_as_of",
]

_EMPTY_QUOTE_STATUSES = frozenset({"empty", "not_reached"})

_CALENDAR_REQUIRED_FIELDS = (
    "ticker", "event_date", "entry_date", "exit_date", "expiry", "spot",
    "calendar_observed_through",
)

_MISSING = object()


class NightlySourceBundleRefusal(ValueError):
    """Explicit refusal to assemble a per-night SourceBundle from staged
    inputs."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def validated_as_of(value: Any, *, label: str = "as_of") -> pd.Timestamp:
    """Validate and normalize one staged date.

    Raises ``NightlySourceBundleRefusal`` (``MISSING_STAGED_INPUT`` when the
    value is ``None``, else ``INVALID_DATE``) for NaT, an unparseable value,
    a bare number/bool that would be misread as epoch time, or a
    timezone-aware timestamp; only a timezone-naive date is supported. An
    independent copy of
    ``engine.v2.ops.native_board_universe._validated_as_of``'s checks.
    """
    if value is None:
        raise NightlySourceBundleRefusal("MISSING_STAGED_INPUT", f"{label} must not be None")
    if isinstance(value, bool) or isinstance(value, (numbers.Number, np.number)):
        raise NightlySourceBundleRefusal(
            "INVALID_DATE",
            f"{label} must be a date/timestamp, not a bare number ({value!r}); a numeric "
            f"value would be misread as epoch time rather than a calendar date",
        )
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise NightlySourceBundleRefusal(
            "INVALID_DATE", f"{label} could not be parsed as a timestamp: {exc}") from exc
    if pd.isna(ts):
        raise NightlySourceBundleRefusal("INVALID_DATE", f"{label} must not be NaT")
    if ts.tzinfo is not None:
        raise NightlySourceBundleRefusal(
            "INVALID_DATE",
            f"{label} is timezone-aware; only a timezone-naive value is supported",
        )
    return ts


def quote_domain_map(rows: Any, quote_status: Any = None) -> dict[str, dict[str, float]]:
    """The native quote map for one event's Tier-1 quote domain.

    An empty domain is accepted only when the caller names WHY it is empty
    (no chain found, or pricing never reached, i.e. quote_status in
    _EMPTY_QUOTE_STATUSES); an empty domain with no recorded status is a
    refusal, not a silent gap. Extracted, behavior-identical (messages and
    checks unchanged), from tools/capture_tier0_corpus.py's _quote_map.
    """
    if quote_status in _EMPTY_QUOTE_STATUSES:
        if rows != []:
            raise NightlySourceBundleRefusal(
                "INVALID_QUOTE_DOMAIN",
                f"quote_status {quote_status} but quote_domain is not empty",
            )
        return {}
    if quote_status not in (None, "recorded", "priced"):
        raise NightlySourceBundleRefusal(
            "INVALID_QUOTE_DOMAIN", f"unknown quote_status {quote_status!r}")
    if not isinstance(rows, list) or not rows:
        raise NightlySourceBundleRefusal(
            "MISSING_STAGED_INPUT", "source_inputs.quote_domain is empty")
    quotes: dict[str, dict[str, float]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise NightlySourceBundleRefusal(
                "INVALID_QUOTE_DOMAIN", f"quote_domain[{index}] is not an object")
        right, strike, expiry, bid, ask = _contract_quote(row, index)
        key = f"{right}:{strike}:{expiry}"
        quote = {"bid": bid, "ask": ask}
        if key in quotes and quotes[key] != quote:
            raise NightlySourceBundleRefusal(
                "INVALID_QUOTE_DOMAIN", f"conflicting source quote: {key}")
        quotes[key] = quote
    return quotes


def _contract_quote(
    row: Mapping[str, Any], index: int,
) -> tuple[str, float, str, float, float]:
    """One quote row's normalized (right, strike, expiry, bid, ask)."""
    try:
        right = str(row["right"]).upper()
        right = {"CALL": "C", "PUT": "P"}.get(right, right)
        strike = float(row["strike"])
        expiry = str(pd.Timestamp(row["expiry"]).date())
        bid = float(row["bid"])
        ask = float(row["ask"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NightlySourceBundleRefusal(
            "INVALID_QUOTE_DOMAIN",
            f"quote_domain[{index}] lacks a complete contract quote",
        ) from exc
    if right not in {"C", "P"} or not all(
        math.isfinite(value) for value in (strike, bid, ask)
    ) or bid < 0.0 or ask < bid:
        raise NightlySourceBundleRefusal(
            "INVALID_QUOTE_DOMAIN", f"quote_domain[{index}] is invalid")
    return right, strike, expiry, bid, ask


def _staged_observed_at(quote_rows: Any) -> list[tuple[int, Any]]:
    """Every ``observed_at`` a staged quote row owns, with its row index."""
    found: list[tuple[int, Any]] = []
    if quote_rows is None or not isinstance(quote_rows, Sequence):
        return found
    for index, row in enumerate(quote_rows):
        if isinstance(row, Mapping) and "observed_at" in row:
            found.append((index, row["observed_at"]))
    return found


def _checked_against_as_of(
    calendar_row: Mapping[str, Any],
    panel_row: Mapping[str, Any],
    tier4_row: Mapping[str, Any],
    observed: Sequence[tuple[int, Any]],
    as_of_ts: pd.Timestamp,
) -> None:
    """Refuse any staged observation dated strictly after ``as_of``."""
    staged = [
        ("calendar_row.calendar_observed_through", calendar_row["calendar_observed_through"]),
        ("panel_row.observed_at", panel_row["observed_at"]),
        ("tier4_row.observed_at", tier4_row["observed_at"]),
        *((f"quote_rows[{index}].observed_at", value) for index, value in observed),
    ]
    for label, value in staged:
        ts = validated_as_of(value, label=label)
        if ts > as_of_ts:
            raise NightlySourceBundleRefusal(
                "POST_AS_OF_ROW", f"{label} ({ts}) is after as_of ({as_of_ts})")


def _feature_is_missing(value: Any) -> bool:
    """Whether a projected feature column counts as missing."""
    return (
        value is _MISSING
        or value is None
        or (isinstance(value, float) and math.isnan(value))
    )


def _require_staged_inputs_present(
    calendar_row: Mapping[str, Any],
    panel_row: Mapping[str, Any],
    tier4_row: Mapping[str, Any],
    quote_rows: Sequence[Mapping[str, Any]] | None,
) -> None:
    """Refuse a wholly-missing staged input before any date/quote checks."""
    if not isinstance(calendar_row, Mapping):
        raise NightlySourceBundleRefusal("MISSING_STAGED_INPUT", "calendar_row is missing")
    missing = sorted(k for k in _CALENDAR_REQUIRED_FIELDS if k not in calendar_row)
    if missing:
        raise NightlySourceBundleRefusal(
            "MISSING_STAGED_INPUT", f"calendar_row is missing {missing}")
    for name, row in (("panel_row", panel_row), ("tier4_row", tier4_row)):
        if row is None or not isinstance(row, Mapping):
            raise NightlySourceBundleRefusal("MISSING_STAGED_INPUT", f"{name} is missing")
        if "observed_at" not in row:
            raise NightlySourceBundleRefusal(
                "MISSING_STAGED_INPUT", f"{name} is missing observed_at")
    if quote_rows is None:
        raise NightlySourceBundleRefusal("MISSING_STAGED_INPUT", "quote_rows is missing")


def _project_features(
    tier4_row: Mapping[str, Any],
    panel_row: Mapping[str, Any],
    feature_names: Sequence[str],
) -> tuple[dict[str, float], dict[str, bool]]:
    """Project ``feature_names`` from the Tier-4 row, falling back to the panel."""
    feature_vector: dict[str, float] = {}
    feature_missing_mask: dict[str, bool] = {}
    for name in sorted(feature_names):
        value = tier4_row.get(name, _MISSING)
        if value is _MISSING:
            value = panel_row.get(name, _MISSING)
        if _feature_is_missing(value):
            feature_missing_mask[name] = True
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise NightlySourceBundleRefusal(
                "INVALID_FEATURE_VALUE",
                f"{name} is present but not a finite number: {value!r}",
            ) from exc
        if not math.isfinite(number):
            raise NightlySourceBundleRefusal(
                "INVALID_FEATURE_VALUE",
                f"{name} is present but not a finite number: {value!r}",
            )
        feature_vector[name] = number
        feature_missing_mask[name] = False
    return feature_vector, feature_missing_mask


def assemble_nightly_source_bundle(
    *,
    source_ref: str,
    strategy: str = "STR-THRU",
    as_of: Any,
    calendar_row: Mapping[str, Any],
    panel_row: Mapping[str, Any],
    tier4_row: Mapping[str, Any],
    quote_rows: Sequence[Mapping[str, Any]],
    quote_status: Any = None,
    feature_names: Sequence[str],
    driver_name: str = "abs_move",
    model_identity: Mapping[str, Any] | None = None,
    model_artifact_refs: Mapping[str, str] | None = None,
    forecast_recipes: Mapping[str, Mapping[str, Any]] | None = None,
    residual_recipe: Mapping[str, Any] | None = None,
    analog_recipe: Mapping[str, Any] | None = None,
    gate_recipe: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SourceBundle:
    """Assemble one (ticker, event)'s SourceBundle from already-staged rows.

    Builds context, raw_quotes, feature_vector and feature_missing_mask from
    the calendar/panel/Tier-4/quote rows, refusing a missing staged input, a
    malformed quote, a non-finite feature, or any observation after as_of.
    model_identity, model_artifact_refs and every recipe are caller-supplied
    pass-through (the {} default means "not yet declared"). No I/O is done.
    """
    _require_staged_inputs_present(calendar_row, panel_row, tier4_row, quote_rows)
    observed = _staged_observed_at(quote_rows)
    as_of_ts = validated_as_of(as_of, label="as_of")
    _checked_against_as_of(calendar_row, panel_row, tier4_row, observed, as_of_ts)
    raw_quotes = quote_domain_map(quote_rows, quote_status)
    context = {k: calendar_row[k] for k in sorted(_CALENDAR_REQUIRED_FIELDS)}
    feature_vector, feature_missing_mask = _project_features(
        tier4_row, panel_row, feature_names)
    _reject_answers("context", context)
    _reject_answers("feature_vector", feature_vector)
    return SourceBundle(
        source_ref=source_ref,
        context=context,
        raw_quotes=raw_quotes,
        feature_vector=feature_vector,
        feature_missing_mask=feature_missing_mask,
        model_identity=dict(model_identity or {}),
        forecast_recipes=dict(forecast_recipes or {}),
        model_artifact_refs=dict(model_artifact_refs or {}),
        residual_recipe=dict(residual_recipe or {}),
        analog_recipe=dict(analog_recipe or {}),
        gate_recipe=dict(gate_recipe or {}),
        driver_name=driver_name,
        strategy=strategy,
        metadata=dict(metadata or {}),
    )
