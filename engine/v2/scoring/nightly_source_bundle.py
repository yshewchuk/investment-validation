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

# The real legacy panel's own row-level date column (engine/features.py's
# _KEY_COLUMNS: ("ticker", "k", "date", "quarter", "year", "mcap_asof")) --
# NOT "observed_at", which panel.parquet has never carried.
_PANEL_DATE_COLUMN = "date"

# The realized-outcome columns legacy excludes from every panel feature set
# (engine/features.py:70, OUTCOME_COLUMNS = ("move", "abs_move")): "the
# realized outcome of the event being scored... move / abs_move are the
# answer." An independent copy, not an import -- this package never imports
# legacy `engine.*` outside compatibility.py's one lazy call (see
# ARCHITECTURE.md, Native vs. legacy independence). Kept in sync by
# tests/test_v2_scoring_nightly_source_bundle.py::
# test_panel_outcome_columns_matches_legacy, which imports the live legacy
# constant and asserts equality, so the two cannot silently drift apart.
_PANEL_OUTCOME_COLUMNS = frozenset({"move", "abs_move"})

# Tier-4 stamp/band/metadata columns (data/features/tier4_forecasts.parquet's
# real schema): every metric stamps its own "<metric>_fold_start" and
# "<metric>_model_id", plus a single top-level "tier3_snapshot"; the
# pred_iv_crush_30 family additionally carries "_p10"/"_p90"/"_sd"/
# "_resid_n" band columns that are refused as a whole prefix, not only their
# fold_start/model_id suffixes. None of these are legitimate model features:
# they describe HOW/WHEN a forecast was produced, or (for pred_iv_crush_30)
# are the crush forecast's own family, never an input to anything else.
_TIER4_STAMP_SUFFIXES = ("_fold_start", "_model_id")
_TIER4_STAMP_NAMES = frozenset({"tier3_snapshot"})
_TIER4_CRUSH_STAMP_PREFIX = "pred_iv_crush_30"

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


def _staged_observed_at(quote_rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, Any]]:
    """Every quote row's own ``observed_at``, with its row index.

    Every quote row must carry one -- a row without it would reach
    ``raw_quotes`` never checked against ``as_of`` at all, not merely
    unchecked-and-flagged, so this refuses rather than skips it. Quotes are
    the one staged input that genuinely carries ``observed_at`` (the panel
    and Tier-4 do not; see ``_checked_against_as_of``/``_project_features``).
    """
    found: list[tuple[int, Any]] = []
    for index, row in enumerate(quote_rows):
        if not isinstance(row, Mapping) or "observed_at" not in row:
            raise NightlySourceBundleRefusal(
                "MISSING_STAGED_INPUT", f"quote_rows[{index}] is missing observed_at")
        found.append((index, row["observed_at"]))
    return found


def _checked_against_as_of(
    calendar_row: Mapping[str, Any],
    panel_row: Mapping[str, Any],
    observed: Sequence[tuple[int, Any]],
    as_of_ts: pd.Timestamp,
) -> None:
    """Refuse any staged observation dated strictly after ``as_of``.

    Tier-4 has no row-level date to check here: ``panel.parquet`` and
    ``tier4_forecasts.parquet`` neither one carries ``observed_at``.
    ``panel_row`` carries its own real date column instead
    (``_PANEL_DATE_COLUMN``, "date" -- legacy's own panel key column,
    engine/features.py's ``_KEY_COLUMNS``). Tier-4 carries no single
    row-level date at all; each metric column stamps its own
    "<metric>_fold_start" instead, checked per used feature in
    ``_project_features``, not here.
    """
    staged = [
        ("calendar_row.calendar_observed_through", calendar_row["calendar_observed_through"]),
        (f"panel_row.{_PANEL_DATE_COLUMN}", panel_row[_PANEL_DATE_COLUMN]),
        *((f"quote_rows[{index}].observed_at", value) for index, value in observed),
    ]
    for label, value in staged:
        ts = validated_as_of(value, label=label)
        if ts > as_of_ts:
            raise NightlySourceBundleRefusal(
                "POST_AS_OF_ROW", f"{label} ({ts}) is after as_of ({as_of_ts})")


def _leaked_feature_reason(name: str, driver_name: str) -> str | None:
    """Why ``name`` may never be projected as a feature, or ``None`` if clean."""
    if name in _PANEL_OUTCOME_COLUMNS:
        return f"{name} is a realized panel outcome column (legacy OUTCOME_COLUMNS)"
    if name == driver_name:
        return f"{name} equals driver_name; the driver being forecast cannot be its own feature"
    if name in _TIER4_STAMP_NAMES:
        return f"{name} is a Tier-4 metadata stamp, not a feature"
    if name.endswith(_TIER4_STAMP_SUFFIXES):
        return f"{name} is a Tier-4 producer stamp (fold_start/model_id), not a feature"
    if name.startswith(_TIER4_CRUSH_STAMP_PREFIX):
        return f"{name} is a pred_iv_crush_30 stamp/band column, not a feature"
    return None


def _reject_leaked_feature_names(feature_names: Sequence[str], driver_name: str) -> None:
    """Refuse any requested feature name that is a known leakage class.

    Checked on the NAMES alone, before any row is read: a realized panel
    outcome (legacy's own OUTCOME_COLUMNS: "move"/"abs_move"), the
    configured ``driver_name`` itself, or a Tier-4 stamp/band/metadata
    column ("*_fold_start", "*_model_id", "tier3_snapshot",
    "pred_iv_crush_30*"). Distinct from ``source_inputs._ANSWER_FIELDS``
    (checked later, on the assembled ``feature_vector``'s keys): that
    denylist is calculated SCORING outputs; this one is calculated/realized
    SOURCE-TABLE columns that were never scoring outputs at all.
    """
    leaked = sorted(
        f"{name} ({reason})"
        for name in feature_names
        for reason in (_leaked_feature_reason(name, driver_name),)
        if reason is not None
    )
    if leaked:
        raise NightlySourceBundleRefusal("LEAKED_FEATURE_NAME", "; ".join(leaked))


def _validated_feature_names(feature_names: Any) -> tuple[str, ...]:
    """Validate ``feature_names``: a non-string sequence of non-empty ``str``,
    with no duplicates."""
    if isinstance(feature_names, (str, bytes)) or not isinstance(feature_names, Sequence):
        raise NightlySourceBundleRefusal(
            "INVALID_FEATURE_NAMES",
            f"feature_names must be a non-string sequence of feature names, "
            f"got {type(feature_names).__name__}",
        )
    names = list(feature_names)
    bad = [n for n in names if not isinstance(n, str) or not n.strip()]
    if bad:
        raise NightlySourceBundleRefusal(
            "INVALID_FEATURE_NAMES",
            f"feature_names entries must be non-empty str: {bad!r}",
        )
    seen: set[str] = set()
    dupes = sorted({n for n in names if n in seen or seen.add(n)})
    if dupes:
        raise NightlySourceBundleRefusal(
            "INVALID_FEATURE_NAMES", f"feature_names has duplicates: {dupes}")
    return tuple(names)


def _validated_spot(spot: Any) -> float:
    """``calendar_row.spot`` must be a finite, strictly positive number."""
    try:
        value = float(spot)
    except (TypeError, ValueError) as exc:
        raise NightlySourceBundleRefusal(
            "INVALID_SPOT",
            f"calendar_row.spot must be a finite positive number, got {spot!r}",
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise NightlySourceBundleRefusal(
            "INVALID_SPOT",
            f"calendar_row.spot must be a finite positive number, got {spot!r}",
        )
    return value


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
    if panel_row is None or not isinstance(panel_row, Mapping):
        raise NightlySourceBundleRefusal("MISSING_STAGED_INPUT", "panel_row is missing")
    if _PANEL_DATE_COLUMN not in panel_row:
        raise NightlySourceBundleRefusal(
            "MISSING_STAGED_INPUT", f"panel_row is missing {_PANEL_DATE_COLUMN!r}")
    if tier4_row is None or not isinstance(tier4_row, Mapping):
        raise NightlySourceBundleRefusal("MISSING_STAGED_INPUT", "tier4_row is missing")
    if (
        quote_rows is None
        or isinstance(quote_rows, (str, bytes))
        or not isinstance(quote_rows, Sequence)
    ):
        raise NightlySourceBundleRefusal("MISSING_STAGED_INPUT", "quote_rows is missing")


def _project_features(
    tier4_row: Mapping[str, Any],
    panel_row: Mapping[str, Any],
    feature_names: Sequence[str],
    as_of_ts: pd.Timestamp,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Project ``feature_names`` from the Tier-4 row, falling back to the panel.

    Every resolved value is passed through EXACTLY as staged -- ``None``,
    NaN (Python/NumPy/pandas, any flavor), +/-inf, or a non-numeric string
    included, unmodified and unconverted. Classifying a value as missing vs.
    invalid is the real consumer's job (``FrozenStageExecutor._row``,
    frozen_executor.py; ``application._feature_fields``, application.py),
    never this assembler's: this function only decides WHICH staged row a
    name resolves from, and whether that provenance is allowed to be used at
    all -- not what the resolved value itself means.

    A name resolved from ``tier4_row`` is allowed only when that metric's
    own ``"{name}_fold_start"`` is staged and ``<= as_of`` (the real Tier-4
    stamp contract: ``tier4_forecasts.parquet`` has no row-level
    ``observed_at``; every metric stamps its own ``fold_start`` instead). A
    used Tier-4 value with no ``fold_start`` staged at all is refused
    outright, not silently treated as missing -- the caller must be able to
    prove *when* that value was fit before native scoring may see it.

    ``feature_missing_mask`` is a pure PRESENCE fact (the name was found in
    neither row), never a value-quality judgement -- and exists only
    because ``SourceBundle``'s dataclass shape requires the field; no real
    consumer today reads it once it reaches ``NativeScoreInputs.features
    ["missing_mask"]`` (``application._feature_fields`` recomputes its own
    ``null_masks`` independently from ``model_inputs``).
    """
    feature_vector: dict[str, Any] = {}
    feature_missing_mask: dict[str, bool] = {}
    for name in sorted(feature_names):
        if name in tier4_row:
            fold_start = tier4_row.get(f"{name}_fold_start", _MISSING)
            if fold_start is _MISSING:
                raise NightlySourceBundleRefusal(
                    "MISSING_STAGED_INPUT",
                    f"tier4_row is missing {name}_fold_start, needed to use {name}",
                )
            fold_ts = validated_as_of(fold_start, label=f"tier4_row.{name}_fold_start")
            if fold_ts > as_of_ts:
                raise NightlySourceBundleRefusal(
                    "POST_AS_OF_ROW",
                    f"tier4_row.{name}_fold_start ({fold_ts}) is after as_of ({as_of_ts})",
                )
            feature_vector[name] = tier4_row[name]
            feature_missing_mask[name] = False
        elif name in panel_row:
            feature_vector[name] = panel_row[name]
            feature_missing_mask[name] = False
        else:
            feature_missing_mask[name] = True
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
    malformed quote, a leaked feature name, a non-finite/non-positive spot,
    or any observation after as_of. model_identity, model_artifact_refs and
    every recipe are caller-supplied pass-through (the {} default means "not
    yet declared"). No I/O is done.
    """
    feature_names = _validated_feature_names(feature_names)
    _require_staged_inputs_present(calendar_row, panel_row, tier4_row, quote_rows)
    _validated_spot(calendar_row["spot"])
    _reject_leaked_feature_names(feature_names, driver_name)
    observed = _staged_observed_at(quote_rows)
    as_of_ts = validated_as_of(as_of, label="as_of")
    _checked_against_as_of(calendar_row, panel_row, observed, as_of_ts)
    raw_quotes = quote_domain_map(list(quote_rows), quote_status)
    context = {k: calendar_row[k] for k in sorted(_CALENDAR_REQUIRED_FIELDS)}
    feature_vector, feature_missing_mask = _project_features(
        tier4_row, panel_row, feature_names, as_of_ts)
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
