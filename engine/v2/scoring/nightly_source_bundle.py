"""Per-night SourceBundle assembler for one (ticker, event) from already-staged inputs."""
from __future__ import annotations

import math
import numbers
from datetime import date, datetime
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
# tests/v2/scoring/test_nightly_source_bundle.py::
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

# A band column's own fold_start is its BASE metric's fold_start: the real
# schema has no "<metric>_p10_fold_start" etc. -- only one fold_start per
# metric (data/features/tier4_forecasts.parquet: pred_abs_move_fold_start,
# pred_im_t1_d14_fold_start, pred_runup_abs_move_d14_fold_start,
# pred_iv_crush_30_fold_start; the crush family is refused as a feature name
# entirely by _leaked_feature_reason, so it never reaches this lookup).
_TIER4_BAND_SUFFIXES = ("_p10", "_p90", "_sd", "_resid_n")

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


def quote_domain_map(rows: Any, quote_status: Any = None) -> dict[str, dict[str, Any]]:
    """The native quote map for one event's Tier-1 quote domain.

    An empty domain is accepted only when the caller names WHY it is empty
    (no chain found, or pricing never reached, i.e. quote_status in
    _EMPTY_QUOTE_STATUSES); an empty domain with no recorded status is a
    refusal, not a silent gap. Shared with tools/capture_tier0_corpus.py,
    which imports it (it was extracted from that tool's former _quote_map and
    has since gained observed_date). Returns {"right:strike:expiry": {"bid":
    float, "ask": float, "observed_date": "YYYY-MM-DD"}}; observed_date is the
    quote row's observed_at calendar date and is omitted when the row has none
    or it is unparseable. A repeated contract must repeat bid/ask exactly (else
    INVALID_QUOTE_DOMAIN) and keeps the earliest observed_date. Pricing reads
    the first leg's observed_date to measure dte_entry. Row keys other than
    right/strike/expiry/bid/ask/observed_at -- notably the staged
    ``chain_spot`` ``nightly_quote_rows`` adds -- are ignored here and never
    reach the returned domain; ``_staged_chain_spot`` is what reads them.
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
    quotes: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise NightlySourceBundleRefusal(
                "INVALID_QUOTE_DOMAIN", f"quote_domain[{index}] is not an object")
        right, strike, expiry, bid, ask = _contract_quote(row, index)
        key = f"{right}:{strike}:{expiry}"
        observed_date = _observed_date(row.get("observed_at"))
        quote: dict[str, Any] = {"bid": bid, "ask": ask}
        if observed_date is not None:
            quote["observed_date"] = observed_date
        if key in quotes:
            _merge_duplicate_quote(quotes[key], quote, key)
            continue
        quotes[key] = quote
    return quotes


def _observed_date(value: Any) -> str | None:
    """One quote row's observation date as an ISO string, or None when absent,
    NaT/NaN, empty, unparseable or not a date/ISO string."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        if value != value:  # NaT
            return None
        return (value.date() if isinstance(value, datetime) else value).isoformat()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            ts = pd.Timestamp(stripped)
        except (ValueError, TypeError):
            return None
        if ts != ts:  # NaT
            return None
        return ts.date().isoformat()
    return None


def _merge_duplicate_quote(existing: dict[str, Any], quote: dict[str, Any],
                           key: str) -> None:
    """Merge a duplicate key's quote into ``existing``: bid/ask must match,
    and the earlier observed_date wins so pricing can measure dte_entry."""
    if (existing["bid"], existing["ask"]) != (quote["bid"], quote["ask"]):
        raise NightlySourceBundleRefusal(
            "INVALID_QUOTE_DOMAIN", f"conflicting source quote: {key}")
    existing_date = existing.get("observed_date")
    new_date = quote.get("observed_date")
    if existing_date is None and new_date is not None:
        existing["observed_date"] = new_date
    elif existing_date is not None and new_date is not None:
        existing["observed_date"] = min(existing_date, new_date)


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


def _staged_chain_spot(quote_rows: Sequence[Mapping[str, Any]]) -> float | None:
    """The ``chain_spot`` the staged quote rows carry, or ``None`` when none do.

    ``nightly_quote_rows.scan_quote_rows`` stages the chain's own ``spot`` (the
    legacy ``entry_cost_pct`` denominator) onto every surviving quote row; this
    is the one place the bundle decides what that optional row key means. It is
    a presence-and-consistency rule, never a computation: nothing here reads the
    value. When any row carries the key, every row must carry the SAME finite
    positive number -- a partially-present, inconsistent, ``None``, NaN,
    non-finite, non-numeric or bool spot is ``INVALID_QUOTE_DOMAIN`` (the same
    class as an unusable bid/ask), so the caller gets no bundle -- whichever
    order the rows come in. When no row carries it, ``None`` means "omit the
    context key": callers that stage their own quote rows
    (``tools/capture_tier0_corpus.py``) are unchanged. Called only after
    ``_staged_observed_at`` has refused every row that is not a Mapping, so the
    row key can be read directly.
    """
    found: float | None = None
    missing_index: int | None = None
    for index, row in enumerate(quote_rows):
        if "chain_spot" not in row:
            if missing_index is None:
                missing_index = index
            continue
        value = row["chain_spot"]
        try:
            spot = None if isinstance(value, bool) else float(value)
        except (TypeError, ValueError):
            spot = None
        if spot is None or not math.isfinite(spot) or spot <= 0.0:
            raise NightlySourceBundleRefusal(
                "INVALID_QUOTE_DOMAIN", "quote rows carry an unusable chain_spot")
        if found is None:
            found = spot
        elif spot != found:
            raise NightlySourceBundleRefusal(
                "INVALID_QUOTE_DOMAIN", "quote rows disagree on chain_spot")
    if found is not None and missing_index is not None:
        raise NightlySourceBundleRefusal(
            "INVALID_QUOTE_DOMAIN", f"quote_rows[{missing_index}] is missing chain_spot")
    return found


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


def _checked_panel_row_event(
    calendar_row: Mapping[str, Any],
    panel_row: Mapping[str, Any],
) -> None:
    """The panel row must describe the SAME event being scored.

    In the real legacy panel, a row's own ``_PANEL_DATE_COLUMN`` ("date")
    is the EVENT date, never an observation/decision date --
    ``engine/features.py::live_features`` builds its synthetic row with
    ``"date": event_date`` (line ~753), and every persisted
    ``panel.parquet`` row is likewise keyed by its own event date. A prior
    version of this module compared ``panel_row.date`` against ``as_of``
    as if it were an observation timestamp; that refused every real
    upcoming-event row outright, since a real panel row's ``date`` always
    equals its own (future, relative to ``as_of``) event date -- the only
    rows that check ever passed were rows for a DIFFERENT, already-past
    event. The one invariant this module can actually verify from the data
    it is given is that the staged ``panel_row`` names the SAME event
    ``calendar_row`` does.
    """
    event_ts = validated_as_of(calendar_row["event_date"], label="calendar_row.event_date")
    panel_date_ts = validated_as_of(
        panel_row[_PANEL_DATE_COLUMN], label=f"panel_row.{_PANEL_DATE_COLUMN}")
    if panel_date_ts != event_ts:
        raise NightlySourceBundleRefusal(
            "PANEL_ROW_WRONG_EVENT",
            f"panel_row.{_PANEL_DATE_COLUMN} ({panel_date_ts}) does not match "
            f"calendar_row.event_date ({event_ts})",
        )


def _checked_against_as_of(
    calendar_row: Mapping[str, Any],
    observed: Sequence[tuple[int, Any]],
    panel_anchor: Any,
    as_of_ts: pd.Timestamp,
) -> None:
    """Refuse any staged observation dated strictly after ``as_of``.

    ``calendar_row.calendar_observed_through``, the caller-declared
    ``panel_anchor``, and each quote row's own ``observed_at`` are checked
    here. The legacy panel's own ``_PANEL_DATE_COLUMN`` ("date") is the
    EVENT date, not an observation date (see ``_checked_panel_row_event``);
    Tier-4 carries no row-level date at all -- each metric column stamps
    its own "<metric>_fold_start" instead, checked per used feature in
    ``_project_features``, not here. Neither belongs in an "as of this
    moment" comparison against ``as_of``.

    ``panel_anchor`` (issue #53) is the caller-declared upper bound on when
    every one of ``panel_row``'s market-state feature values was actually
    observed -- distinct from ``panel_row["date"]``, which is the EVENT
    date (see ``_checked_panel_row_event``), and from ``as_of`` itself.
    ``panel_row`` (a plain ``name -> value`` mapping) never carries this
    anchor as one of its own columns: for a row built by
    ``engine/features.py::live_features``, the real per-feature anchor
    (``FeatureVector.feature_as_of``, ``engine/audit.py``) lives on the
    ``FeatureVector`` wrapper the caller flattens into ``panel_row``, not
    inside the flattened values themselves -- the caller passes
    ``FeatureVector.as_of`` instead, the decision date ``live_features``
    already validated (via its own ``assert_causal(vector)`` call) as
    ``>=`` every one of that vector's per-feature stamps. For a persisted
    ``panel.parquet`` row, the true anchor
    (``regime_asof``/``runup_asof``/``orats_asof``, ``ANCHOR_COLUMNS``) is
    dropped before the file is written (safe only because those all equal
    ``date`` for a HISTORICAL row); the caller passes ``panel_row["date"]``
    itself instead -- a safe, if looser, upper bound. This module cannot
    derive the anchor from ``panel_row`` alone, so it is a required,
    caller-declared argument, checked here the same way
    ``calendar_observed_through`` and quote ``observed_at`` already are --
    never inferred or recomputed, and never compared against
    ``panel_row["date"]``/the event date (that causal ordering is
    ``live_features``'s own ``assert_decision_causal``, not a second check
    here that could disagree with it).
    """
    staged = [
        ("calendar_row.calendar_observed_through", calendar_row["calendar_observed_through"]),
        ("panel_row.anchor", panel_anchor),
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


def _tier4_fold_start_key(name: str) -> str:
    """The real fold_start column that stamps ``name`` in
    ``tier4_forecasts.parquet``.

    A band column (``<metric>_p10``/``_p90``/``_sd``/``_resid_n``) is
    stamped by its BASE metric's own fold_start -- there is no
    ``<metric>_p10_fold_start`` column in the real schema, only one
    fold_start per metric.
    """
    for suffix in _TIER4_BAND_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)] + "_fold_start"
    return f"{name}_fold_start"


def _is_null_forecast_value(value: Any) -> bool:
    """Whether ``value`` is legacy's "no forecast" null.

    Used ONLY to decide whether the fold_start gate below applies -- never
    to build ``feature_missing_mask``, which stays a pure presence fact.
    Measured against the real ``tier4_forecasts.parquet``: 108,320 of
    199,973 ``pred_abs_move`` rows are NULL, always paired with a NULL
    ``pred_abs_move_fold_start`` (zero counterexamples) -- a null forecast
    was never fit, so requiring its fold_start would refuse every real
    null row outright, exactly the rows the Tier-4 fallback-to-panel path
    exists to let through as "missing" rather than "unusable input."
    """
    return (
        value is None
        or value is pd.NA
        or (isinstance(value, (float, np.floating)) and math.isnan(value))
    )


#: Names a frozen model may list in its feature_order that the scorer itself
#: computes: days_to_print by engine/v2/scoring/stages.py
#: _stage_days_to_print (before the forecast); dte_entry, measured from the
#: first priced leg's own quote date, and entry_cost_pct by
#: engine/v2/scoring/stages.py _check_pricing_boundary (after pricing). They
#: are not projected from tier4/panel rows. Only entry_cost_pct is also an
#: answer field that source_inputs forbids (projecting it would make every row
#: refuse); days_to_print/dte_entry are not in that denylist, but the scorer
#: computes them (dte_entry from the first priced leg's quote date) and they are
#: never projected from source inputs.
_SCORER_COMPUTED_FEATURES = frozenset({
    "entry_cost_pct", "days_to_print", "dte_entry",
})


def _project_features(
    tier4_row: Mapping[str, Any],
    panel_row: Mapping[str, Any],
    feature_names: Sequence[str],
    as_of_ts: pd.Timestamp,
    calendar_features: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Project calendar-owned features first, then Tier-4 with panel fallback.

    Every resolved Tier-4/panel value passes through EXACTLY as staged -- ``None``,
    NaN (Python/NumPy/pandas, any flavor), +/-inf, or a non-numeric string
    included, unmodified and unconverted. Classifying a value as missing vs.
    invalid is the real consumer's job (``FrozenStageExecutor._row``,
    frozen_executor.py; ``application._feature_fields``, application.py),
    never this assembler's: this function only decides WHICH staged row a
    name resolves from, and whether that provenance is allowed to be used at
    all -- not what the resolved value itself means.

    A NON-NULL value resolved from ``tier4_row`` is allowed only when that
    metric's own fold_start (``_tier4_fold_start_key``) is staged and
    ``<= as_of`` (the real Tier-4 stamp contract: ``tier4_forecasts.parquet``
    has no row-level ``observed_at``; every metric stamps its own
    ``fold_start`` instead). A used, non-null Tier-4 value with no
    fold_start staged at all is refused outright, not silently treated as
    missing -- the caller must be able to prove *when* that value was fit
    before native scoring may see it. A NULL Tier-4 value (legacy's own
    "no forecast") skips this gate entirely, fold_start present or not,
    NaT/None or dated whenever -- a null forecast carries no information to
    leak, and its fold_start is itself always null in the real data.

    ``feature_missing_mask`` is a pure PRESENCE fact (the name was found in
    neither row), never a value-quality judgement -- and exists only
    because ``SourceBundle``'s dataclass shape requires the field; no real
    consumer today reads it once it reaches ``NativeScoreInputs.features
    ["missing_mask"]`` (``application._feature_fields`` recomputes its own
    ``null_masks`` independently from ``model_inputs``).
    """
    feature_vector: dict[str, Any] = {}
    feature_missing_mask: dict[str, bool] = {}
    calendar_features = calendar_features or {}
    for name in sorted(feature_names):
        if name in _SCORER_COMPUTED_FEATURES:
            continue
        if name in calendar_features:
            feature_vector[name] = calendar_features[name]
            feature_missing_mask[name] = False
        elif name in tier4_row:
            value = tier4_row[name]
            if not _is_null_forecast_value(value):
                fold_key = _tier4_fold_start_key(name)
                fold_start = tier4_row.get(fold_key, _MISSING)
                if fold_start is _MISSING:
                    raise NightlySourceBundleRefusal(
                        "MISSING_STAGED_INPUT",
                        f"tier4_row is missing {fold_key}, needed to use {name}",
                    )
                fold_ts = validated_as_of(fold_start, label=f"tier4_row.{fold_key}")
                if fold_ts > as_of_ts:
                    raise NightlySourceBundleRefusal(
                        "POST_AS_OF_ROW",
                        f"tier4_row.{fold_key} ({fold_ts}) is after as_of ({as_of_ts})",
                    )
            feature_vector[name] = value
            feature_missing_mask[name] = False
        elif name in panel_row:
            feature_vector[name] = panel_row[name]
            feature_missing_mask[name] = False
        else:
            feature_missing_mask[name] = True
    return feature_vector, feature_missing_mask


def _runup_calendar_features(strategy: str, calendar_row: Mapping[str, Any]) -> dict[str, int]:
    """RUNUP timing is calendar-owned, never a panel/Tier-4 fallback."""
    if strategy != "STR-RUNUP":
        return {}
    value = calendar_row.get("days_before_print", _MISSING)
    if value is _MISSING:
        raise NightlySourceBundleRefusal(
            "MISSING_STAGED_INPUT", "calendar_row is missing days_before_print")
    try:
        valid = (not isinstance(value, bool) and isinstance(value, numbers.Real)
                 and math.isfinite(value) and value >= 0 and value == int(value))
    except (ValueError, OverflowError):
        valid = False
    if not valid:
        raise NightlySourceBundleRefusal(
            "INVALID_RUNUP_HORIZON", "days_before_print must be a nonnegative session count")
    return {"days_before_print": int(value)}


def _staged_context(calendar_row: Mapping[str, Any], calendar_features: Mapping[str, int],
                    as_of_ts: pd.Timestamp, quote_rows: Sequence[Mapping[str, Any]]
                    ) -> dict[str, Any]:
    """Stage calendar-owned context, the validated decision date and the chain spot.

    ``chain_spot`` joins the context only when the staged quote rows carry it
    (``_staged_chain_spot`` decides presence, consistency and the refusal); the
    rows are Mappings by then, because ``_staged_observed_at`` ran first in
    ``assemble_nightly_source_bundle``. Nothing here reads the value: geometry
    and strike selection keep using ``calendar_row["spot"]``, the close.
    """
    context = {k: calendar_row[k] for k in sorted(_CALENDAR_REQUIRED_FIELDS)}
    context.update(calendar_features)
    # The decision date: stages._stage_days_to_print measures days_to_print from it.
    context["as_of"] = str(as_of_ts.date())
    chain_spot = _staged_chain_spot(quote_rows)
    if chain_spot is not None:
        context["chain_spot"] = chain_spot
    return context


def assemble_nightly_source_bundle(
    *,
    source_ref: str,
    strategy: str = "STR-THRU",
    as_of: Any,
    calendar_row: Mapping[str, Any],
    panel_row: Mapping[str, Any],
    panel_anchor: Any,
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
    quote_max_age_sessions: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> SourceBundle:
    """Assemble one (ticker, event)'s SourceBundle from already-staged rows.

    Builds context, raw_quotes, feature_vector and feature_missing_mask from
    the calendar/panel/Tier-4/quote rows, refusing a missing staged input, a
    malformed quote, a leaked feature name, a non-finite/non-positive spot,
    a panel_anchor after as_of, or any observation after as_of. model_identity,
    model_artifact_refs and every recipe are caller-supplied pass-through (the
    {} default means "not yet declared"). context carries the caller's
    quote_max_age_sessions unchanged (key omitted when None, kept when zero)
    plus quote_date and quote_latest_date -- the earliest and latest validated
    quote observed_at dates (YYYY-MM-DD) -- only when quote rows exist, so a
    later put observation cannot hide behind an earlier call. context carries
    ``chain_spot`` only when the quote rows carry it (see ``_staged_context``). No I/O.
    """
    feature_names = _validated_feature_names(feature_names)
    _require_staged_inputs_present(calendar_row, panel_row, tier4_row, quote_rows)
    _validated_spot(calendar_row["spot"])
    _reject_leaked_feature_names(feature_names, driver_name)
    _checked_panel_row_event(calendar_row, panel_row)
    observed = _staged_observed_at(quote_rows)
    as_of_ts = validated_as_of(as_of, label="as_of")
    _checked_against_as_of(calendar_row, observed, panel_anchor, as_of_ts)
    raw_quotes = quote_domain_map(list(quote_rows), quote_status)
    calendar_features = _runup_calendar_features(strategy, calendar_row)
    context = _staged_context(calendar_row, calendar_features, as_of_ts, quote_rows)
    if quote_max_age_sessions is not None:
        context["quote_max_age_sessions"] = quote_max_age_sessions
    if observed:
        # Observation provenance for stages._check_stale_quote (issue #169):
        # every observed_at already passed _checked_against_as_of (all <=
        # as_of, tz-naive). Earliest is the conservative age date; latest is
        # the newest observation, so a later put cannot hide behind an earlier
        # call. An allowed empty quote domain carries neither date.
        quote_days = [
            validated_as_of(value, label=f"quote_rows[{index}].observed_at")
            for index, value in observed]
        context["quote_date"] = str(min(quote_days).date())
        context["quote_latest_date"] = str(max(quote_days).date())
    feature_vector, feature_missing_mask = _project_features(
        tier4_row, panel_row, feature_names, as_of_ts, calendar_features)
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
