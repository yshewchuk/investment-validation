"""Cutover PR-3 batch assembler.

Turns one ``ScoringReleaseBinding`` (PR-1, ``engine/v2/scoring/
release_bindings.py``) plus per-event staged inputs (PR-2, ``engine/v2/
scoring/nightly_source_bundle.py``) into ``{BoardRequest: (ScoreRequest,
NativeScoreInputs)}``, admitting only registered strategies (currently
``STR-THRU``, ``STR-RUNUP`` and the gate-less entry-rule ``TWIN-P``,
``TWIN-P5``, ``BFLY-P``, ``BFLY-P5``, ``CTR5``, ``RAMP7`` and ``CND-PS``):
batch-level malformed input raises, every per-row gap is a collected
:class:`NativeScoreBatchRowRefusal`, and ``run_native_score_batch_worker``
scores the assembled batch under ``no_fit_guard``. Production reaches this
worker via the supervisor's native shadow sidecar -- see
``engine/v2/ops/ARCHITECTURE.md`` for the worker contract.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import pandas as pd

from engine.v2.contracts import ScoreBatch, ScoreRequest
from engine.v2.foundation import canonical_json, content_hash, to_document
from engine.v2.models.payoff_artifact import PayoffLineArtifact
from engine.v2.models.recalibration_artifact import RecalibrationMapArtifact
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_entry_rule_inputs import entry_rule_declarations
from engine.v2.ops.native_runup_inputs import (
    latest_artifacts_at_or_before,
    runup_model_declarations,
)
from engine.v2.ops.native_score_batch_types import (
    NativeScoreBatchRowRefusal,
    _event_date_identity,
)
from engine.v2.scoring.application import score_batch
from engine.v2.scoring.gate_fold_selection import select_gate_forecast_source
from engine.v2.scoring.identity import request_hash, score_request_key
from engine.v2.scoring.native_analog_query import AmbiguousAnalogPool, analog_declarations
from engine.v2.scoring.native_gate_features import GATE_ANALOG_COLUMNS, GATE_FORECAST_COLUMNS
from engine.v2.scoring.nightly_source_bundle import (
    NightlySourceBundleRefusal,
    assemble_nightly_source_bundle,
    validated_as_of,
)
from engine.v2.scoring.release_bindings import (
    ModelIdentity,
    ModelNotReady,
    ScoringReleaseBinding,
    select_chooser_state,
)
from engine.v2.scoring.source_inputs import (
    AnswerFieldsInSource,
    SourceBundle,
    build_native_score_inputs,
)
from engine.v2.scoring.stages import NativeScoreInputs

__all__ = [
    "NativeScoreBatchRowRefusal",
    "NightlyEventInputs",
    "assemble_score_batch_inputs",
    "run_native_score_batch_worker",
]

_SHADOW_FILL_ALPHA = 0.5  # engine.fills.MID.alpha; do not import engine.fills
                          # (legacy) from this v2 module -- this is a shadow
                          # default, documented in ARCHITECTURE.md, not a
                          # legacy-derived fact.

_CHOOSER_PRIMITIVES = (
    "mean_prior_abs_move", "ema12r_abs", "signed_streak",
    "mean_prior_or_implied", "mcap_log", "or_implied", "or_rvol30",
    "spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol20",
    "spy_vol5", "spy_vol60", "spy_vol252", "spy_vol20_rel252",
)

_SCORER_COMPUTED_FEATURES = frozenset({
    "entry_cost_pct", "days_to_print", "dte_entry",
})

@dataclass(frozen=True, slots=True)
class _DriverSpec:
    """One forecast output and its exact release role, then optional alias."""

    output: str
    role: str
    alias: str | None = None


@dataclass(frozen=True, slots=True)
class _StrategySpec:
    """Ordered drivers and release/bundle keys; registration alone admits rows.

    A spec with both ``gate_role`` and ``gate_policy_key`` ``None`` is gate-less
    (the entry-rule strategies); ``entry_rule`` selects that contract. Missing
    size/iv_crush role, clock mismatch, or no/ambiguous paired pool or trailing
    cutoff refuses only that row; missing/non-finite mcap leaves the mcap term
    undetermined; missing panel ``iv30`` is a stage flag. No retries, caches or fallbacks.
    """

    drivers: tuple[_DriverSpec, ...]
    bundle_strategy: str
    entry_rule: bool = False
    gate_role: str | None = None
    gate_policy_key: str | None = None
    driver_name: str | None = None
    chooser: bool = False

    def __post_init__(self) -> None:
        if (not isinstance(self.drivers, tuple) or not self.drivers
                or any(not isinstance(driver, _DriverSpec) for driver in self.drivers)):
            raise ValueError("strategy drivers must be a nonempty tuple of driver specs")
        if len({driver.output for driver in self.drivers}) != len(self.drivers):
            raise ValueError("strategy driver outputs must be unique")
        if (self.gate_role is None) != (self.gate_policy_key is None):
            raise ValueError("gate_role and gate_policy_key must be both None or both set")
        if self.gate_role is None and not self.entry_rule:
            raise ValueError("gate-less strategies must use the entry_rule path")
        if self.entry_rule:
            if self.gate_role is not None:
                raise ValueError("entry_rule strategies must be gate-less")
            if len(self.drivers) != 2 or (
                    self.drivers[0].output != "forecast_abs_move"
                    or self.drivers[1].output != "pred_iv_crush_30"):
                raise ValueError(
                    "entry_rule drivers must be forecast_abs_move then pred_iv_crush_30")


@dataclass(frozen=True, slots=True)
class _ResolvedDriver:
    output: str
    identity_key: str
    identity: ModelIdentity


# RUNUP's forecast declarations; frozen model-state requirements are separate.
_RUNUP_SPEC = _StrategySpec(
    drivers=(_DriverSpec("driver_prediction", "driver:STR-RUNUP", "implied_t1:*"),
             _DriverSpec("runup_move_prediction", "runup_move:*")),
    gate_role="gate:STR-RUNUP", gate_policy_key="STR-RUNUP",
    bundle_strategy="STR-RUNUP", driver_name="im_t1")

# Entry-rule strategies: gate-less, size then iv_crush drivers.
_ENTRY_RULE_SPECS = {
    name: _StrategySpec(
        drivers=(_DriverSpec("forecast_abs_move", "size:*"),
                 _DriverSpec("pred_iv_crush_30", "iv_crush:*")),
        bundle_strategy=name, entry_rule=True, chooser=True)
    for name in ("TWIN-P", "TWIN-P5", "BFLY-P", "BFLY-P5", "CTR5", "RAMP7", "CND-PS")
}

# Code-owned admission, not a runtime registration API or chooser contract.
_STRATEGY_REGISTRY: Mapping[str, _StrategySpec] = MappingProxyType({
    "STR-THRU": _StrategySpec(
        drivers=(_DriverSpec("driver_prediction", "driver:STR-THRU", "size:*"),),
        gate_role="gate:STR-THRU", gate_policy_key="STR-THRU",
        bundle_strategy="STR-THRU"),
    "STR-RUNUP": _RUNUP_SPEC,
    **_ENTRY_RULE_SPECS,
})


@dataclass(frozen=True, slots=True)
class NightlyEventInputs:
    """One (ticker, event, strategy)'s already-staged inputs for batch assembly."""

    key: BoardRequest
    calendar_row: Mapping[str, Any]
    panel_row: Mapping[str, Any]
    #: Issue #53's fix (PR #67): the caller-declared upper bound on when
    #: every one of panel_row's market-state feature values was actually
    #: observed -- FeatureVector.as_of for a forward event, panel_row["date"]
    #: for an already-realized one. Passed straight through to
    #: assemble_nightly_source_bundle's own required panel_anchor parameter;
    #: this module derives nothing about it itself.
    panel_anchor: Any
    tier4_row: Mapping[str, Any]
    quote_rows: Sequence[Mapping[str, Any]]
    quote_status: Any = None
    quote_max_age_sessions: Any = None


def _board_request_key(key: BoardRequest) -> str:
    """The canonical, JSON-object-key-safe string identity for one row:
    ``f"{ticker}|{strategy}|{event_date_identity}|{session}"``. The
    ``event_date_identity`` component (the strict :func:`
    _event_date_identity` form -- ``YYYY-MM-DD`` for a midnight/day event,
    canonical naive ISO datetime for an intraday one) can never contain the
    ``"|"`` join delimiter, but ``ticker``/``strategy``/``session`` are
    free-text-shaped inputs this module does not control at the source.
    Raises a :class:`NativeScoreBatchRowRefusal` (code
    ``INVALID_KEY_FIELD``) the moment ``"|"`` appears in any of those three
    fields, BEFORE the row is ever encoded into ``records.json``'s/
    ``refusals.json``'s own keys -- this makes the encoding a true
    bijection for every row that does get a canonical key.
    """
    for field_name, value in (
        ("ticker", key.ticker), ("strategy", key.strategy), ("session", key.session),
    ):
        if "|" in value:
            raise NativeScoreBatchRowRefusal(
                key, "INVALID_KEY_FIELD",
                f"{field_name} contains the canonical key delimiter")
    return f"{key.ticker}|{key.strategy}|{_event_date_identity(key.event_date)}|{key.session}"


def _iso(value: Any) -> str | None:
    """One date-shaped value as an ISO date string, ``None`` passed through."""
    if value is None:
        return None
    return str(pd.Timestamp(value).date())


def _matched_decision_clock(
    model_identity: Mapping[str, ModelIdentity],
    spec: _StrategySpec,
) -> tuple[str | None, str | None,
           tuple[tuple[_ResolvedDriver, ...], ModelIdentity | None] | None]:
    """Resolve every declared driver and gate, then require one decision clock.

    Returns ``(code, detail, None)`` for the first missing role (drivers in
    declaration order, then gate) or clock mismatch; otherwise returns
    ``(None, None, (drivers, gate_identity))``. A gate-less spec returns
    ``gate_identity=None`` after requiring every driver to share the first
    driver's clock. Each exact role wins over its optional alias. Actual
    identity keys and every forecast output survive; missing roles or
    conflicting clocks never produce a partial declaration. The caller
    attaches the row key to each refusal.
    """
    drivers = []
    for declaration in spec.drivers:
        identity_key = declaration.role
        identity = model_identity.get(identity_key)
        if identity is None and declaration.alias is not None:
            identity_key = declaration.alias
            identity = model_identity.get(identity_key)
        if identity is None:
            return ("RELEASE_MISSING_ROLE", declaration.role, None)
        drivers.append(_ResolvedDriver(declaration.output, identity_key, identity))
    if spec.gate_role is None:
        reference_clock = drivers[0].identity.decision_clock_id
        for declaration, driver in zip(spec.drivers, drivers, strict=True):
            if driver.identity.decision_clock_id != reference_clock:
                return ("AMBIGUOUS_DECISION_CLOCK",
                        f"{declaration.role}={driver.identity.decision_clock_id!r} "
                        f"{spec.drivers[0].role}={reference_clock!r}", None)
        return (None, None, (tuple(drivers), None))
    gate_identity = model_identity.get(spec.gate_role)
    if gate_identity is None:
        return ("RELEASE_MISSING_ROLE", spec.gate_role, None)
    for declaration, driver in zip(spec.drivers, drivers, strict=True):
        if driver.identity.decision_clock_id != gate_identity.decision_clock_id:
            return ("AMBIGUOUS_DECISION_CLOCK",
                    f"{declaration.role}={driver.identity.decision_clock_id!r} "
                    f"{spec.gate_role}={gate_identity.decision_clock_id!r}", None)
    return (None, None, (tuple(drivers), gate_identity))


def _feature_names_or_refusal(
    feature_names: Sequence[str], *identities: Any,
) -> tuple[Sequence[str], tuple[str, str] | None]:
    """Explicit ``feature_names`` win. Only ``None`` or an empty list/tuple
    derive: the sorted, de-duplicated union of the identities' ``feature_order``
    minus the stage-derived gate columns (projecting those would suppress their
    native derivation), or a ``(code, fixed_detail)`` refusal when an identity
    has no ``feature_order``. Any other shape (``0``, ``False``, ``{}``, ``""``,
    a non-str element) is passed through unchanged, so the bundle assembler
    refuses it ``INVALID_FEATURE_NAMES`` exactly as before."""
    if feature_names is None:
        feature_names = ()
    if feature_names or not isinstance(feature_names, (list, tuple)):
        return feature_names, None
    if not all(identity.feature_order for identity in identities):
        return (), ("RELEASE_MISSING_FEATURE_ORDER",
                    "driver or gate identity has no feature_order")
    stage_owned = {*GATE_FORECAST_COLUMNS, *GATE_ANALOG_COLUMNS}
    return tuple(sorted(
        {name for identity in identities for name in identity.feature_order} - stage_owned)), None


def _identity_context(as_of: Any, snapshot_id: str,
                       calendar_row: Mapping[str, Any], *,
                       quote_max_age_sessions: Any = None) -> dict[str, Any]:
    """The MC-seed identity fields ``stages._model_seed`` needs, merged into
    ``bundle.context`` before ``build_native_score_inputs`` -- see
    ARCHITECTURE.md's MC-seed note for the shadow defaults' provenance."""
    return {
        "snapshot": snapshot_id,
        "requested_as_of": _iso(as_of),
        "requested_event_date": _iso(calendar_row["event_date"]),
        "requested_strike": None,
        "requested_expiry": _iso(calendar_row["expiry"]),
        "fill_alpha": _SHADOW_FILL_ALPHA,
        "variant": None,
        "decision_offset": None,
        "quote_max_age_sessions": quote_max_age_sessions,
        "chain_as_of": _iso(as_of),
    }


def _calendar_row_problem(key: BoardRequest, calendar_row: Any) -> tuple[str, str] | None:
    """``None`` if ``calendar_row`` is a well-formed mapping whose own
    ``ticker``/``event_date``/``expiry``/``entry_date`` are parseable and
    match ``key``, else ``(code, fixed_detail)`` for a per-row refusal.
    Checked first, before every other per-row check (CodeRabbit rounds 2-5,
    PR #66):

    - Not a mapping at all (e.g. a null ``calendar_row`` in ``events.json``)
      -- ``CALENDAR_ROW_INVALID``. Nothing upstream of this module checks
      this; calling ``.get`` on a non-mapping would otherwise raise
      ``AttributeError`` and abort the whole batch before
      ``assemble_nightly_source_bundle`` gets a chance to refuse it.
    - An unparseable non-null ``event_date``/``expiry``/``entry_date`` --
      ``CALENDAR_ROW_INVALID``. ``assemble_nightly_source_bundle`` never
      parses ``event_date``/``expiry`` itself (it copies both straight into
      ``context``), and this module's own ``_identity_context`` parses
      ``expiry`` later, outside every try/except in
      :func:`_assemble_one_event` -- an unparseable value must be caught
      here, before that point, not there. ``entry_date`` is the causal
      cutoff :func:`_row_cutoff` later derives from
      :func:`validated_as_of`; validating it here with the SAME helper
      returns the row-level ``CALENDAR_ROW_INVALID`` instead of the
      ``INVALID_DATE`` the cutoff helper would otherwise surface. A null
      ``entry_date`` is left to downstream cutoff construction and its
      ``MISSING_STAGED_INPUT`` refusal, including STR-THRU's required exact-
      cutoff analog selection; only malformed non-null values are caught here.
    - ``ticker``/``event_date`` not matching ``key`` -- ``CALENDAR_ROW_KEY_
      MISMATCH``. Neither this module nor ``assemble_nightly_source_bundle``
      (which only checks ``panel_row`` against ``calendar_row``, never
      against the caller's ``BoardRequest``) verifies this elsewhere, and
      ``ScoreRequest`` carries no ticker/event_date of its own -- a
      caller-side pairing bug would otherwise pass silently and could
      produce a duplicate ``request_hash`` (see
      :func:`assemble_score_batch_inputs`) that corrupts a different row.

    Every detail here is a FIXED string, never the raw staged value
    (CodeRabbit round 4, CWE-209) -- ``refusals.json`` is a published
    output of a successful attempt.
    """
    if not isinstance(calendar_row, Mapping):
        return "CALENDAR_ROW_INVALID", "calendar row is missing or not an object"
    try:
        calendar_event_date = _iso(calendar_row.get("event_date"))
        _iso(calendar_row.get("expiry"))
        entry_date = calendar_row.get("entry_date")
        if entry_date is not None:
            validated_as_of(entry_date, label="calendar_row.entry_date")
    except (TypeError, ValueError):
        return "CALENDAR_ROW_INVALID", "calendar row has an unparseable date field"
    if calendar_row.get("ticker") != key.ticker or calendar_event_date != _iso(key.event_date):
        return "CALENDAR_ROW_KEY_MISMATCH", "calendar row does not match the request key"
    return None


def _row_cutoff(as_of: Any, calendar_row: Mapping[str, Any]) -> str:
    """One row's causal cutoff -- no later than both ``as_of`` and its entry.

    Legacy's ``evidence_cutoff`` is ``min(decision date, entry date)``
    (``tools/phase5_calibration_keys.py``); STR-THRU's decision date is
    ``as_of``, so both are validated here exactly as the nightly bundle
    validates ``as_of``. A malformed value raises ``NightlySourceBundleRefusal``
    (re-wrapped per-row by the caller, like every other staged-date refusal).
    """
    as_of_ts = validated_as_of(as_of, label="as_of")
    entry_ts = validated_as_of(calendar_row.get("entry_date"), label="calendar_row.entry_date")
    return min(as_of_ts, entry_ts).date().isoformat()


def _mcap_usd(panel_row: Mapping[str, Any]) -> float | None:
    """The row's log-market-cap as dollars, or ``None`` when undetermined.

    ``mcap_log`` must be a finite real number (never a ``bool`` or a string);
    an absent, non-finite or overflowing value leaves the mcap term
    undetermined rather than inventing a floor.
    """
    value = panel_row.get("mcap_log")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        if not math.isfinite(value):
            return None
        return math.exp(value)
    except (OverflowError, TypeError):
        return None


def _pre_iv30(panel_row: Mapping[str, Any]) -> float | None:
    """The row's positive pre-event 30-day implied vol (panel ``iv30``), or ``None`` when absent."""
    value = panel_row.get("iv30")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        if not math.isfinite(value) or value <= 0:
            return None
    except (OverflowError, TypeError):
        return None
    return float(value)


def _entry_rule_declarations_or_refusal(
    binding: ScoringReleaseBinding,
    *,
    strategy: str,
    entry_rule_ids: tuple[ModelIdentity, ModelIdentity],
    as_of: Any,
    calendar_row: Mapping[str, Any],
    panel_row: Mapping[str, Any],
) -> dict[str, Any]:
    """One entry-rule row's paired-pool, cutoff and gate declarations.

    ``entry_rule_declarations`` owns the exact-key checks (unique paired pool,
    event month's trailing cutoff, usable mcap); this only adds the planned-
    exit recipe and, when the row carries one, its ``iv30`` (recipe key ``pre_iv30``). Its
    ``NightlySourceBundleRefusal`` is re-wrapped per-row by the caller.
    """
    size_identity, crush_identity = entry_rule_ids
    declarations = entry_rule_declarations(
        binding, strategy=strategy, size_identity=size_identity,
        crush_identity=crush_identity,
        cutoff=_row_cutoff(as_of, calendar_row),
        event_date=calendar_row["event_date"],
        mcap_usd=_mcap_usd(panel_row))
    declarations["residual_recipe"] = {"mode": "planned_exit"}
    pre_iv30 = _pre_iv30(panel_row)
    if pre_iv30 is not None:
        declarations["paired_residual_recipe"] = {
            **declarations["paired_residual_recipe"], "pre_iv30": pre_iv30}
    return declarations


def _release_payoff_declarations(
    binding: ScoringReleaseBinding,
    strategy: str,
    artifacts: Sequence[Any],
    as_of: Any,
    calendar_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the latest causal payoff and recalibration inputs for one row."""
    if not artifacts and strategy != "STR-THRU":
        return {}
    cutoff = _row_cutoff(as_of, calendar_row)
    lines = tuple(artifact for artifact in artifacts
                  if isinstance(artifact, PayoffLineArtifact))
    matches = latest_artifacts_at_or_before(
        lines, cutoff=cutoff, fill_alpha=_SHADOW_FILL_ALPHA)
    if len(matches) > 1:
        raise NightlySourceBundleRefusal(
            "MODEL_NOT_READY", "release carries duplicate latest eligible payoff lines")
    if not matches:
        raise NightlySourceBundleRefusal(
            "MODEL_NOT_READY", "release lacks an eligible payoff line")
    payoff = matches[0]
    declarations: dict[str, Any] = {
        "payoff_artifact": payoff,
        "payoff_artifact_recipe": {"before": cutoff, "bound_cutoff": payoff.cutoff},
    }
    if strategy != "STR-THRU":
        return declarations
    recalibration = tuple(
        artifact for artifact in binding.recalibration_artifacts.get(strategy, ())
        if isinstance(artifact, RecalibrationMapArtifact))
    recalibration_matches = latest_artifacts_at_or_before(
        recalibration, cutoff=cutoff, fill_alpha=_SHADOW_FILL_ALPHA)
    if len(recalibration_matches) > 1:
        raise NightlySourceBundleRefusal(
            "MODEL_NOT_READY", "release carries duplicate latest eligible recalibration maps")
    if not recalibration_matches:
        raise NightlySourceBundleRefusal(
            "MODEL_NOT_READY", "release lacks an eligible recalibration map")
    recalibration_artifact = recalibration_matches[0]
    declarations["recalibration_declared"] = True
    declarations["recalibration_artifact"] = recalibration_artifact
    declarations["payoff_artifact_recipe"]["recalibration_bound_cutoff"] = (
        recalibration_artifact.cutoff)
    return declarations


def _release_artifact_declarations(
    binding: ScoringReleaseBinding,
    strategy: str,
    driver_identity: Any,
    as_of: Any,
    calendar_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Declare frozen driver, payoff and recalibration inputs for one row.

    Artifact data cutoffs establish eligibility and ordering; computed_at is
    never consulted. No request/event/panel rows or fitting provide fallback.
    """
    pool = binding.driver_residual_artifacts.get(driver_identity.role)
    artifacts = binding.payoff_artifacts.get(strategy) or ()
    if pool is None and not artifacts and strategy != "STR-THRU":
        return {}
    declarations: dict[str, Any] = {}
    if pool is not None:
        if pool.model_id != driver_identity.model_id:
            raise NightlySourceBundleRefusal(
                "MODEL_NOT_READY",
                "driver residual pool model_id disagrees with the resolved driver identity")
        declarations["model_residual_artifacts"] = {"driver": pool}
        declarations["model_residual_artifact_recipe"] = {"driver": {
            "role": pool.role, "model_id": pool.model_id, "fold": pool.fold,
            "content_hash": pool.content_hash}}
    declarations.update(_release_payoff_declarations(
        binding, strategy, artifacts, as_of, calendar_row))
    return declarations


_RELEASE_ARTIFACT_REFUSAL_LABELS: Mapping[tuple[str, str], str] = {
    ("MODEL_NOT_READY",
     "driver residual pool model_id disagrees with the resolved driver identity"):
        "driver residual pool identity mismatch",
    ("MODEL_NOT_READY", "release carries duplicate exact-cutoff payoff artifacts"):
        "duplicate exact-cutoff payoff artifact",
    ("MODEL_NOT_READY", "release lacks the exact-cutoff payoff artifact"):
        "exact-cutoff payoff artifact missing",
    ("MODEL_NOT_READY", "release lacks the exact-cutoff analog pool"):
        "exact-cutoff analog pool missing",
    ("MODEL_NOT_READY", "one exact paired residual pool is required"):
        "paired residual pool missing",
    ("MODEL_NOT_READY", "more than one exact paired residual pool"):
        "duplicate paired residual pool",
    ("MODEL_NOT_READY", "one exact trailing cutoff artifact is required"):
        "trailing cutoff artifact missing",
    ("MODEL_NOT_READY", "more than one exact trailing cutoff artifact"):
        "duplicate trailing cutoff artifact",
    ("MODEL_NOT_READY", "event date is not a valid trailing cutoff month"):
        "invalid trailing cutoff month",
    ("MODEL_NOT_READY", "RUNUP champion residual pool is missing or unusable"):
        "RUNUP champion pool missing or unusable",
    ("MODEL_NOT_READY", "RUNUP requires one exact-key payoff surface"):
        "RUNUP payoff surface missing",
    ("MODEL_NOT_READY", "RUNUP has more than one exact-key payoff surface"):
        "duplicate RUNUP payoff surface",
    ("MODEL_NOT_READY", "RUNUP payoff surface is numerically unusable"):
        "RUNUP payoff surface unusable",
    ("MODEL_NOT_READY", "chooser state is missing or unusable"): "DYN-SV chooser state missing or unusable",
}


def _nightly_source_bundle_refusal_detail(exc: NightlySourceBundleRefusal) -> str:
    """The safe published detail for one source-bundle refusal.

    Only the thirteen internal release-artifact causes above keep a fixed
    distinguishing label; every other refusal publishes just its code. The
    raw ``exc.detail`` is never interpolated or concatenated into the text.
    """
    fixed_label = None
    if isinstance(exc.detail, str):
        fixed_label = _RELEASE_ARTIFACT_REFUSAL_LABELS.get((exc.code, exc.detail))
    if fixed_label is not None:
        return f"nightly_source_bundle refused: {exc.code}: {fixed_label}"
    return f"nightly_source_bundle refused: {exc.code}"


def _with_release_artifacts(
    bundle: SourceBundle,
    *,
    binding: ScoringReleaseBinding,
    key: BoardRequest,
    strategy: str,
    drivers: tuple[_ResolvedDriver, ...],
    as_of: Any,
    snapshot_id: str,
    calendar_row: Mapping[str, Any],
    quote_max_age_sessions: Any,
    panel_row: Mapping[str, Any],
    entry_rule_ids: tuple[ModelIdentity, ModelIdentity] | None = None,
) -> SourceBundle | NativeScoreBatchRowRefusal:
    """Declare the release-backed artifacts on a bundle, or re-wrap a refusal.

    Split out of :func:`_assemble_one_event` to keep it under its line budget.
    The cutoff's own date refusal (:func:`_row_cutoff`) is a source-bundle
    refusal like any other and is re-wrapped with the row's key. An entry-rule
    row declares its paired pool, trailing cutoff, planned-exit recipe and gate
    recipe from :func:`_entry_rule_declarations_or_refusal` instead of the
    driver-residual/payoff path. STR-THRU requires the board analog pool for the
    exact ``(strategy, fill alpha, cutoff)`` key; absence refuses the row. Other
    strategies keep their optional analog behavior: the frozen board analog
    pool, when the release carries one for this row's ``(strategy, fill alpha,
    cutoff)``, is declared with the NATIVE snapshot id and the row's native
    request key as its bootstrap-seed identity (#681); otherwise the analog
    block stays undeclared and remains not-applicable.
    """
    context = {**bundle.context,
               **_identity_context(as_of, snapshot_id, calendar_row,
                                   quote_max_age_sessions=quote_max_age_sessions)}
    try:
        if entry_rule_ids is not None:
            declarations = _entry_rule_declarations_or_refusal(
                binding, strategy=strategy, entry_rule_ids=entry_rule_ids,
                as_of=as_of, calendar_row=calendar_row, panel_row=panel_row)
        elif strategy == "STR-RUNUP":
            declarations = runup_model_declarations(
                binding, {driver.output: driver.identity for driver in drivers},
                cutoff=_row_cutoff(as_of, calendar_row), fill_alpha=_SHADOW_FILL_ALPHA)
        else:
            declarations = _release_artifact_declarations(
                binding, strategy, drivers[0].identity, as_of, calendar_row)
        analogs = binding.analog_artifacts.get(strategy) or ()
        if analogs or strategy == "STR-THRU":
            analog_declaration = analog_declarations(
                analogs, strategy, _SHADOW_FILL_ALPHA, _row_cutoff(as_of, calendar_row),
                panel_row, snapshot_id,
                score_request_key({**context, "ticker": key.ticker, "strategy": strategy}))
            if not analog_declaration and strategy == "STR-THRU":
                raise NightlySourceBundleRefusal(
                    "MODEL_NOT_READY", "release lacks the exact-cutoff analog pool")
            declarations.update(analog_declaration)
        for artifact_name, context_name in (
            ("payoff_artifact", "payoff_artifact_cutoff"),
            ("recalibration_artifact", "recalibration_artifact_cutoff"),
        ):
            artifact = declarations.get(artifact_name)
            if artifact is not None:
                context[context_name] = artifact.cutoff
    except NightlySourceBundleRefusal as exc:
        return NativeScoreBatchRowRefusal(
            key, exc.code, _nightly_source_bundle_refusal_detail(exc))
    except AmbiguousAnalogPool:
        return NativeScoreBatchRowRefusal(
            key, "MODEL_NOT_READY", "release carries more than one analog pool for the row key")
    return replace(
        bundle, **declarations, context=context,
        model_release=binding.model_release, frozen_inference=binding.frozen_inference,
    )


def _input_build_refusal(key: BoardRequest, exc: ValueError) -> NativeScoreBatchRowRefusal:
    """NATIVE_INPUT_BUILD_FAILED with the exception class and, for
    AnswerFieldsInSource, fixed-vocabulary field names; never str(exc) (CWE-209)."""
    detail = f"native input build failed: {type(exc).__name__}"
    if isinstance(exc, AnswerFieldsInSource):
        detail += f" {exc.fields}"
    return NativeScoreBatchRowRefusal(key, "NATIVE_INPUT_BUILD_FAILED", detail)


def _bundle_or_refusal(
    key: BoardRequest,
    event: NightlyEventInputs,
    *,
    binding: ScoringReleaseBinding,
    spec: _StrategySpec,
    drivers: tuple[_ResolvedDriver, ...],
    gate_identity: ModelIdentity | None,
    gate_policy: Mapping[str, Mapping[str, Any]],
    as_of: Any,
    feature_names: Sequence[str],
    driver_name: str,
    gate_source_cache: dict[tuple[int, str], Any],
) -> SourceBundle | NativeScoreBatchRowRefusal:
    """``assemble_nightly_source_bundle``'s result, or a re-wrapped refusal.

    Split out of :func:`_assemble_one_event` purely to keep it under its
    line budget. ``exc.detail`` is never published (CodeRabbit round 5,
    CWE-209): it can itself echo staged input (e.g. ``quote_domain_map``
    embeds an invalid ``quote_status`` string into some refusals) --
    ``exc.code`` is a closed, module-controlled vocabulary and is safe to
    publish unchanged; the free-text detail is not. A gate-less spec declares
    neither the gate identity nor a gate recipe.
    """
    model_identity = {driver.identity_key: to_document(driver.identity)
                      for driver in drivers}
    gate_recipe = None
    if gate_identity is not None:
        model_identity[spec.gate_role] = to_document(gate_identity)
        gate_recipe = {**gate_policy[spec.gate_policy_key],
                       "binding_id": gate_identity.binding_id}
    try:
        bundle = assemble_nightly_source_bundle(
            source_ref=f"native-score-batch:{key.ticker}:{_iso(key.event_date)}:{key.strategy}",
            strategy=spec.bundle_strategy, as_of=as_of,
            calendar_row=event.calendar_row, panel_row=event.panel_row,
            panel_anchor=event.panel_anchor,
            tier4_row=event.tier4_row, quote_rows=event.quote_rows,
            quote_status=event.quote_status, feature_names=feature_names,
            quote_max_age_sessions=event.quote_max_age_sessions,
            driver_name=spec.driver_name if spec.driver_name is not None else driver_name,
            model_identity=model_identity,
            model_artifact_refs={driver.output: driver.identity.artifact_hash for driver in drivers},
            forecast_recipes={driver.output: {"binding_id": driver.identity.binding_id}
                              for driver in drivers},
            gate_recipe=gate_recipe,
        )
    except NightlySourceBundleRefusal as exc:
        return NativeScoreBatchRowRefusal(
            key, exc.code, f"nightly_source_bundle refused: {exc.code}")
    except AnswerFieldsInSource as exc:
        # One row carrying a calculated answer field refuses that row only (#630).
        return _input_build_refusal(key, exc)
    if key.strategy == "STR-THRU":
        event_id = event.calendar_row.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            return NativeScoreBatchRowRefusal(
                key, "MISSING_STAGED_INPUT", "calendar_row missing event_id")
        source = _gate_source_or_refusal(
            key, binding, key.event_date, as_of, gate_source_cache)
        if isinstance(source, NativeScoreBatchRowRefusal):
            return source
        return replace(
            bundle, gate_forecast_source=source,
            forecast_pool={
                "predictions": source.pool_pred,
                "residuals": source.pool_res,
                "interval_floor": source.interval_floor,
            },
        )
    return bundle


def _gate_source_or_refusal(
    key: BoardRequest,
    binding: ScoringReleaseBinding,
    event_date: Any,
    as_of: Any,
    gate_source_cache: dict[tuple[int, str], Any],
) -> Any | NativeScoreBatchRowRefusal:
    """The gate forecast source for ``key``, or its fixed-detail refusal.

    ``ModelNotReady`` from :func:`select_gate_forecast_source` is mapped to
    ``MODEL_NOT_READY`` with a fixed safe detail, and that exception text is
    never copied into the refusal (CWE-209).
    """
    causal_month = min(
        pd.Timestamp(event_date).strftime("%Y-%m"),
        pd.Timestamp(as_of).strftime("%Y-%m"),
    )
    cache_key = (id(binding), causal_month)
    if cache_key in gate_source_cache:
        return gate_source_cache[cache_key]
    try:
        source = select_gate_forecast_source(binding, event_date, as_of)
    except ModelNotReady:
        return NativeScoreBatchRowRefusal(
            key, "MODEL_NOT_READY",
            "causal size fold unavailable for gate forecast")
    gate_source_cache[cache_key] = source
    return source


def _declared_chooser_artifact_refs(recipe: Mapping) -> list[str]:
    refs: list[str] = []
    for key in ("analog_pool", "admissible_table"):
        entry = recipe.get(key)
        if isinstance(entry, Mapping):
            content_hash = entry.get("content_hash")
            if isinstance(content_hash, str):
                refs.append(content_hash)
    artifact_refs = recipe.get("artifact_refs")
    if isinstance(artifact_refs, (list, tuple)):
        refs.extend(ref for ref in artifact_refs if isinstance(ref, str))
    return refs


def _chooser_artifact_refs(bundle) -> tuple[str, ...]:
    recipe = getattr(bundle, "chooser_recipe", None)
    refs: list[str] = []
    if isinstance(recipe, Mapping) and recipe:
        chosen_binding_ids = set()
        top_level_binding_id = recipe.get("binding_id")
        if isinstance(top_level_binding_id, str):
            chosen_binding_ids.add(top_level_binding_id)
        producers = recipe.get("producers")
        if isinstance(producers, Mapping):
            for producer in producers.values():
                if isinstance(producer, Mapping):
                    binding_id = producer.get("binding_id")
                    if isinstance(binding_id, str):
                        chosen_binding_ids.add(binding_id)
        for binding in bundle.model_release.bindings:
            if binding.binding_id in chosen_binding_ids:
                for member in binding.members:
                    if member.name == "estimator":
                        refs.append(member.content_hash)
        refs.extend(_declared_chooser_artifact_refs(recipe))
    return tuple(refs)


def _score_artifact_refs(drivers, gate_identity: ModelIdentity | None, bundle) -> tuple[str, ...]:
    refs = [driver.identity.artifact_hash for driver in drivers]
    if gate_identity is not None:
        refs.append(gate_identity.artifact_hash)
    gate_source = bundle.gate_forecast_source
    if gate_source is not None:
        refs.append(gate_source.artifact_hash)
    refs.extend(_chooser_artifact_refs(bundle))
    return tuple(dict.fromkeys(refs))


def _gate_policy_and_feature_names(
    key: BoardRequest,
    spec: _StrategySpec,
    gate_policy: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
    drivers: Sequence[_ResolvedDriver],
    gate_identity: ModelIdentity | None,
) -> tuple[Sequence[str], NativeScoreBatchRowRefusal | None]:
    """Gate-policy check plus explicit/derived feature names for one row.

    Returns ``(feature_names, None)`` when the row may proceed, or
    ``((), refusal)`` with the same ``GATE_POLICY_NOT_STAGED`` or
    ``RELEASE_MISSING_FEATURE_ORDER`` refusal the caller would build itself.
    """
    if spec.gate_policy_key is not None and spec.gate_policy_key not in gate_policy:
        return (), NativeScoreBatchRowRefusal(
            key, "GATE_POLICY_NOT_STAGED",
            f"no gate policy staged for {key.strategy!r}")
    identities = [driver.identity for driver in drivers]
    if gate_identity is not None:
        identities.append(gate_identity)
    feature_names, problem = _feature_names_or_refusal(feature_names, *identities)
    if problem is not None:
        return (), NativeScoreBatchRowRefusal(key, *problem)
    return feature_names, None


_CHOOSER_STATE_FAILURE_DETAIL = "chooser state is missing or unusable"

_CHOOSER_REFUSAL_CAUSE_LABELS: Mapping[str, str] = {
    "chooser_analog_pool": "chooser analog pool missing or unusable",
    "admissible_table:dyn_sv": "chooser admissible table missing or unusable",
    "tier4_folds:size": "chooser size fold missing or unusable",
    "phase5_release.json": "chooser release catalog missing or unusable",
    "chooser_training_cutoff": "chooser fold training cutoff provenance missing or unusable",
    "chooser_champion": "chooser champion binding missing or unusable",
    "chooser_producer": "chooser producer binding missing or unusable",
    "chooser state": "chooser state missing or unusable",
}


def _chooser_bundle_or_refusal(
    bundle: Any, key: BoardRequest, *, binding: ScoringReleaseBinding,
    event_date: Any, as_of: Any, fold_cache: dict[tuple[int, str], Any],
) -> Any | NativeScoreBatchRowRefusal:
    try:
        state = select_chooser_state(binding, event_date, as_of, fold_cache=fold_cache)
        chooser = binding.model_identity.get("chooser:DYN-SV")
        implied = state.producer_bindings.get("pred_im_t1_d14")
        runup = state.producer_bindings.get("pred_runup_abs_move_d14")
        if (chooser is None
                or not isinstance(getattr(chooser, "binding_id", None), str)
                or not chooser.binding_id):
            raise ModelNotReady("chooser_champion", _CHOOSER_STATE_FAILURE_DETAIL)
        if (implied is None or runup is None
                or not isinstance(getattr(implied, "binding_id", None), str)
                or not implied.binding_id
                or not isinstance(getattr(runup, "binding_id", None), str)
                or not runup.binding_id):
            raise ModelNotReady("chooser_producer", _CHOOSER_STATE_FAILURE_DETAIL)
        bindings = list(bundle.model_release.bindings)
        known_ids = {item.binding_id for item in bindings}
        for selected in (implied, runup):
            if selected.binding_id not in known_ids:
                bindings.append(selected)
                known_ids.add(selected.binding_id)
        model_release = replace(bundle.model_release, bindings=tuple(bindings))
        return replace(bundle, chooser_recipe={
            "binding_id": chooser.binding_id,
            "artifact_refs": list(state.artifact_refs),
            "fold_provenance": state.fold_provenance,
            "producers": {
                "pred_im_t1_d14": {"binding_id": implied.binding_id},
                "pred_runup_abs_move_d14": {"binding_id": runup.binding_id},
            },
            "analog_pool": {
                "pool_id": state.analog_pool.pool_id,
                "cutoff": state.analog_pool.cutoff,
                "content_hash": state.analog_pool.content_hash,
            },
            "admissible_table": {
                "table_id": state.admissible_table.table_id,
                "version": state.admissible_table.version,
                "content_hash": state.admissible_table.content_hash,
            },
        }, chooser_fold_pools=state.fold_pools,
           chooser_analog_pool=state.analog_pool,
           chooser_admissible_table=state.admissible_table,
           model_release=model_release)
    except ModelNotReady as exc:
        label = _CHOOSER_REFUSAL_CAUSE_LABELS.get(
            exc.member_id, "chooser state missing or unusable")
        return NativeScoreBatchRowRefusal(key, "MODEL_NOT_READY", "DYN-SV " + label)


def _apply_declared_chooser(
    bundle: Any, key: BoardRequest, spec: _StrategySpec, event: NightlyEventInputs,
    *, binding: ScoringReleaseBinding, as_of: Any,
    fold_cache: dict[tuple[int, str], Any],
) -> Any | NativeScoreBatchRowRefusal:
    if not spec.chooser:
        return bundle
    return _chooser_bundle_or_refusal(
        bundle, key, binding=binding,
        event_date=event.calendar_row.get("event_date"), as_of=as_of,
        fold_cache=fold_cache)
def _valid_explicit_feature_names(feature_names: Sequence[str]) -> bool:
    """Whether names are a duplicate-free non-string sequence of non-empty str."""
    if isinstance(feature_names, (str, bytes)) or not isinstance(feature_names, Sequence):
        return False
    if any(not isinstance(name, str) or not name.strip() for name in feature_names):
        return False
    return len(set(feature_names)) == len(feature_names)


def _project_member_features(
    key: BoardRequest,
    event: NightlyEventInputs,
    binding: ScoringReleaseBinding,
    feature_names: Sequence[str],
) -> Sequence[str] | NativeScoreBatchRowRefusal:
    """Project required staged chooser/producer features for one DYN-SV row.

    Every other supported strategy keeps its explicit/derived ``feature_names``
    unchanged. Malformed explicit names are returned unchanged so the bundle's
    own ``INVALID_FEATURE_NAMES`` validation still refuses them before
    projection. Scorer-computed and stage-owned gate columns are never
    projected or required from staged rows.
    """
    if not _valid_explicit_feature_names(feature_names):
        return feature_names
    if key.strategy != "DYN-SV":
        return feature_names
    references = binding.chooser_fold_references
    if not references:
        return NativeScoreBatchRowRefusal(
            key, "RELEASE_MISSING_FEATURE_ORDER",
            "chooser producer references are missing")
    projected = set(_CHOOSER_PRIMITIVES) - _SCORER_COMPUTED_FEATURES
    for reference in references.values():
        if not reference.release_binding.feature_order:
            return NativeScoreBatchRowRefusal(
                key, "RELEASE_MISSING_FEATURE_ORDER",
                "chooser producer identity has no feature_order")
        projected.update(reference.release_binding.feature_order)
    projected -= _SCORER_COMPUTED_FEATURES
    projected -= set(GATE_FORECAST_COLUMNS)
    projected -= set(GATE_ANALOG_COLUMNS)
    ordered = tuple(sorted(projected.union(feature_names)))
    for name in projected:
        if name not in event.tier4_row and name not in event.panel_row:
            return NativeScoreBatchRowRefusal(
                key, "MISSING_STAGED_INPUT",
                "required member feature is missing from staged rows")
    return ordered


def _assemble_one_event(
    event: NightlyEventInputs,
    *,
    binding: ScoringReleaseBinding,
    gate_policy: Mapping[str, Mapping[str, Any]],
    as_of: Any,
    snapshot_id: str,
    calendar_revision: str,
    feature_names: Sequence[str],
    driver_name: str,
    gate_source_cache: dict[tuple[int, str], Any],
) -> tuple[ScoreRequest, NativeScoreInputs] | NativeScoreBatchRowRefusal:
    """Assemble one event's request and native inputs, or its refusal."""
    key = event.key
    try:
        _board_request_key(key)
    except NativeScoreBatchRowRefusal as exc:
        return exc
    strategy = key.strategy
    problem = _calendar_row_problem(key, event.calendar_row)
    if problem is not None:
        return NativeScoreBatchRowRefusal(key, *problem)
    spec = _STRATEGY_REGISTRY.get(strategy)
    if spec is None:
        return NativeScoreBatchRowRefusal(
            key, "UNSUPPORTED_STRATEGY",
            f"only {', '.join(_STRATEGY_REGISTRY)} is supported, got {strategy!r}")
    code, detail, pair = _matched_decision_clock(binding.model_identity, spec)
    if code is not None:
        return NativeScoreBatchRowRefusal(key, code, detail)
    drivers, gate_identity = pair
    driver_identity = drivers[0].identity
    feature_names, refusal = _gate_policy_and_feature_names(
        key, spec, gate_policy, feature_names, drivers, gate_identity)
    if refusal is not None:
        return refusal
    feature_names = _project_member_features(key, event, binding, feature_names)
    if isinstance(feature_names, NativeScoreBatchRowRefusal):
        return feature_names
    bundle = _bundle_or_refusal(
        key, event, binding=binding, spec=spec, drivers=drivers,
        gate_identity=gate_identity, gate_policy=gate_policy, as_of=as_of,
        feature_names=feature_names, driver_name=driver_name,
        gate_source_cache=gate_source_cache)
    if isinstance(bundle, NativeScoreBatchRowRefusal):
        return bundle
    event_id = event.calendar_row.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        return NativeScoreBatchRowRefusal(
            key, "MISSING_STAGED_INPUT", "calendar_row missing event_id")
    bundle = _with_release_artifacts(
        bundle, binding=binding, key=key, strategy=spec.bundle_strategy,
        drivers=drivers, as_of=as_of, snapshot_id=snapshot_id,
        calendar_row=event.calendar_row,
        quote_max_age_sessions=event.quote_max_age_sessions,
        panel_row=event.panel_row,
        entry_rule_ids=((drivers[0].identity, drivers[1].identity)
                        if spec.entry_rule else None))
    if isinstance(bundle, NativeScoreBatchRowRefusal):
        return bundle
    bundle = _apply_declared_chooser(
        bundle, key, spec, event, binding=binding, as_of=as_of,
        fold_cache=gate_source_cache)
    if isinstance(bundle, NativeScoreBatchRowRefusal):
        return bundle
    try:
        native_inputs = build_native_score_inputs(bundle)
    except ValueError as exc:
        # CWE-209: never str(exc) (refusals.json is published). Only the class
        # and, for AnswerFieldsInSource, fixed-vocabulary answer-field names.
        return _input_build_refusal(key, exc)
    request = ScoreRequest(
        event_id=str(event_id), calendar_revision=str(calendar_revision),
        strategy_version=strategy, deployment_id=binding.model_release.deployment_id,
        decision_clock_id=driver_identity.decision_clock_id,
        requested_decision_at=_iso(as_of) or "", snapshot_id=str(snapshot_id),
        mode="shadow", fill_model={"alpha": _SHADOW_FILL_ALPHA},
        model_artifact_refs=_score_artifact_refs(drivers, gate_identity, bundle),
    )
    return (request, native_inputs)


def _checked_event_input(event: Any) -> NightlyEventInputs:
    """Validate one batch event item, or raise -- the per-event half of
    :func:`_checked_batch_arguments`'s batch-level checks, split out purely to
    keep that function under its complexity budget. A valid item is returned
    unchanged (never copied or re-wrapped). ``quote_max_age_sessions`` accepts
    ``None`` or a non-negative ``int`` (a ``bool`` is rejected despite its
    ``int`` subclassing), exactly mirroring :func:`_event_inputs_from_document`'s
    own decode guard; every other bound raises, never becoming a per-row
    refusal."""
    if not isinstance(event, NightlyEventInputs):
        raise TypeError("events must be a sequence of NightlyEventInputs")
    bound = event.quote_max_age_sessions
    if bound is not None and (
            isinstance(bound, bool)
            or not isinstance(bound, int)
            or bound < 0):
        raise ValueError("quote_max_age_sessions must be null or a non-negative integer")
    return event


def _checked_batch_arguments(
    *,
    binding: Any,
    snapshot_id: Any,
    calendar_revision: Any,
    as_of: Any,
    events: Sequence[Any],
) -> tuple[tuple[NightlyEventInputs, ...], pd.Timestamp]:
    """Validate every batch-level (shared-by-every-row) argument to
    :func:`assemble_score_batch_inputs`, or raise.

    Split out purely to keep :func:`assemble_score_batch_inputs` under its
    line budget -- see that function's own docstring for why each of these
    checks exists and raises rather than becoming a per-row refusal.
    """
    if not isinstance(binding, ScoringReleaseBinding):
        raise TypeError("binding must be a ScoringReleaseBinding")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("snapshot_id must be a non-empty string")
    if not isinstance(calendar_revision, str) or not calendar_revision:
        raise ValueError("calendar_revision must be a non-empty string")
    try:
        normalized_as_of = validated_as_of(as_of, label="as_of")
    except NightlySourceBundleRefusal as exc:
        raise ValueError(f"invalid as_of: {exc.code}: {exc.detail}") from exc
    events = tuple(events)
    if any(not isinstance(event, NightlyEventInputs) for event in events):
        raise TypeError("events must be a sequence of NightlyEventInputs")
    events = tuple(_checked_event_input(event) for event in events)
    keys = [event.key for event in events]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate BoardRequest key in events")
    return events, normalized_as_of


def _assemble_validated_events(
    events: tuple[NightlyEventInputs, ...],
    *,
    as_of: Any,
    snapshot_id: str,
    calendar_revision: str,
    binding: ScoringReleaseBinding,
    feature_names: Sequence[str],
    gate_policy: Mapping[str, Mapping[str, Any]],
    driver_name: str,
) -> tuple[dict[BoardRequest, tuple[ScoreRequest, NativeScoreInputs]],
           tuple[NativeScoreBatchRowRefusal, ...]]:
    """Assemble checked rows and enforce request-hash uniqueness."""
    results: dict[BoardRequest, tuple[ScoreRequest, NativeScoreInputs]] = {}
    refusals: list[NativeScoreBatchRowRefusal] = []
    seen_hashes: dict[str, BoardRequest] = {}
    gate_source_cache: dict[tuple[int, str], Any] = {}
    for event in events:
        outcome = _assemble_one_event(
            event, binding=binding, gate_policy=gate_policy, as_of=as_of,
            snapshot_id=snapshot_id, calendar_revision=calendar_revision,
            feature_names=feature_names, driver_name=driver_name,
            gate_source_cache=gate_source_cache,
        )
        if isinstance(outcome, NativeScoreBatchRowRefusal):
            refusals.append(outcome)
            continue
        request, _ = outcome
        row_hash = request_hash(request)
        if row_hash in seen_hashes:
            raise ValueError(
                f"duplicate request_hash {row_hash} for {event.key!r} and "
                f"{seen_hashes[row_hash]!r}")
        seen_hashes[row_hash] = event.key
        results[event.key] = outcome
    return results, tuple(refusals)


def assemble_score_batch_inputs(
    *,
    as_of: Any,
    snapshot_id: str,
    calendar_revision: str,
    binding: ScoringReleaseBinding,
    events: Sequence[NightlyEventInputs],
    feature_names: Sequence[str],
    gate_policy: Mapping[str, Mapping[str, Any]] | None = None,
    driver_name: str = "abs_move",
) -> tuple[dict[BoardRequest, tuple[ScoreRequest, NativeScoreInputs]],
           tuple[NativeScoreBatchRowRefusal, ...]]:
    """Assemble ``{BoardRequest: (ScoreRequest, NativeScoreInputs)}`` for a batch.

    Batch-level malformed input -- a non-``ScoringReleaseBinding`` ``binding``,
    a non-``NightlyEventInputs`` item in ``events``, or two events sharing one
    ``BoardRequest`` key -- raises (the whole call is meaningless); every
    per-row gap is a collected :class:`NativeScoreBatchRowRefusal` instead, so
    one bad row never sinks the batch: ``UNSUPPORTED_STRATEGY`` (not
    registered), ``RELEASE_MISSING_ROLE``/``AMBIGUOUS_DECISION_CLOCK`` (the
    release's declared driver/gate identities for the strategy),
    ``RELEASE_MISSING_FEATURE_ORDER`` (empty ``feature_names`` and an
    identity without ``feature_order``), ``GATE_POLICY_NOT_
    STAGED`` (no caller-supplied threshold for the strategy -- gate thresholds
    are not part of the release binding), any re-wrapped
    ``NightlySourceBundleRefusal`` from the per-event bundle assembly,
    ``MISSING_STAGED_INPUT`` (no ``event_id`` in the calendar row),
    ``CALENDAR_ROW_INVALID`` (``calendar_row`` is not a mapping, or its
    ``event_date``/``expiry``/``entry_date`` do not parse), ``CALENDAR_ROW_KEY_MISMATCH``
    (``calendar_row``'s own ``ticker``/``event_date`` does not match its
    ``NightlyEventInputs.key``) and ``NATIVE_INPUT_BUILD_FAILED`` (a
    ``ValueError`` from ``build_native_score_inputs``). Per-row assembly
    itself lives in :func:`_assemble_one_event`.

    A colliding ``request_hash`` across two DIFFERENT ``BoardRequest`` keys
    (CodeRabbit round 2, PR #66) also raises ``ValueError`` rather than
    becoming a per-row refusal: ``ScoreRequest`` carries no ticker/event_date
    of its own, so two distinct rows whose ``ScoreRequest`` fields coincide
    (most plausibly a duplicated ``calendar_row["event_id"]``, since that is
    the one genuinely per-event field) would otherwise silently collide in
    ``run_native_score_batch_worker``'s ``fields_by_request`` map -- one row
    clobbering the other's inputs with no visible refusal for either.
    Nothing here can say which of the two rows is "the bad one", so this is
    batch-level, not a row-level refusal.

    ``as_of``/``snapshot_id``/``calendar_revision`` are batch-level, shared-
    by-every-row arguments validated once, here, before any row is
    attempted (Opus gate, PR #66; see :func:`_checked_batch_arguments`): a
    bad ``as_of`` would otherwise only surface once ``assemble_nightly_
    source_bundle`` re-validates it inside each row, refusing every row
    individually while the attempt still reports success, and a bad
    ``snapshot_id``/``calendar_revision`` would otherwise flow straight
    into every ``ScoreRequest`` via ``str()``.
    """
    events, as_of = _checked_batch_arguments(
        binding=binding, snapshot_id=snapshot_id, calendar_revision=calendar_revision,
        as_of=as_of, events=events)
    gate_policy = dict(gate_policy or {})
    return _assemble_validated_events(
        events, as_of=as_of, snapshot_id=snapshot_id,
        calendar_revision=calendar_revision, binding=binding,
        feature_names=feature_names, gate_policy=gate_policy,
        driver_name=driver_name,
    )


def _event_inputs_from_document(doc: Mapping[str, Any]) -> NightlyEventInputs:
    """Decode one ``events.json`` array item into :class:`NightlyEventInputs`.

    Date-shaped values are passed straight through as plain ISO strings --
    ``assemble_nightly_source_bundle`` already accepts them (exactly as
    ``tests/v2/scoring/test_nightly_source_bundle.py``'s own fixtures do).
    """
    raw_event_date = doc["key"]["event_date"]
    _event_date_identity(raw_event_date)
    quote_max_age_sessions = doc.get("quote_max_age_sessions")
    if quote_max_age_sessions is not None and (
            isinstance(quote_max_age_sessions, bool)
            or not isinstance(quote_max_age_sessions, int)
            or quote_max_age_sessions < 0):
        raise ValueError("quote_max_age_sessions must be null or a non-negative integer")
    key = BoardRequest(
        ticker=str(doc["key"]["ticker"]), strategy=str(doc["key"]["strategy"]),
        event_date=pd.Timestamp(raw_event_date),
        session=str(doc["key"]["session"]))
    return NightlyEventInputs(
        key=key, calendar_row=doc["calendar_row"], panel_row=doc["panel_row"],
        panel_anchor=doc["panel_anchor"],
        tier4_row=doc["tier4_row"], quote_rows=doc["quote_rows"],
        quote_status=doc.get("quote_status"),
        quote_max_age_sessions=quote_max_age_sessions)


def _keyed_by_board_request(items: Any) -> dict[str, Any]:
    """Key an iterable of ``(BoardRequest, value)`` pairs by
    :func:`_board_request_key`, raising ``ValueError`` on a duplicate
    canonical identity instead of silently letting the later pair overwrite
    the earlier one.

    ``_event_date_identity`` preserves the instant's naive wall-clock
    fields (midnight/day keeps the legacy ``YYYY-MM-DD`` form, intraday
    keeps its full datetime form, issue #356), so two rows that differ only
    by time of day have DISTINCT identities here; what this guard still
    catches is two rows sharing ONE identity (equal instants, however
    expressed) -- a plain ``ValueError`` batch-level failure, matching this
    module's existing ``duplicate request_hash`` batch-level check in
    :func:`assemble_score_batch_inputs`, never a per-row refusal: nothing
    here can say which of the two colliding rows is "the bad one".
    """
    keyed: dict[str, Any] = {}
    for key, value in items:
        canonical_key = _board_request_key(key)
        if canonical_key in keyed:
            raise ValueError(f"duplicate canonical BoardRequest key: {canonical_key}")
        keyed[canonical_key] = value
    return keyed


#: A producer refusal's detail is upstream, pipeline-generated text (the
#: same trust level as events.json, not external/adversarial input), but
#: refusals.json is a published artifact -- cap it defensively so one
#: malformed producer detail can never dump an unbounded blob into it.
_MAX_PRODUCER_DETAIL_LENGTH = 500

#: The only ``event_date`` wire shapes ``producer_refusals.json`` may carry
#: -- exactly what this module's own ``as_document()``/``_board_request_key``
#: write through :func:`_event_date_identity`: the legacy ``YYYY-MM-DD`` day
#: form for a midnight/day event, and the full canonical naive ISO datetime
#: form for an intraday one. Relative strings like "now"/"today" parse fine
#: under ``pd.Timestamp`` but resolve to wall-clock time, which would make a
#: merged refusal's identity depend on when the gate runs instead of on the
#: document's content; offset/Z strings would normalize an instant onto a
#: different naive wall clock. Both are rejected (issue #356) -- a decoder
#: identity must equal the producer's own, never a truncation of it.


def _validated_producer_refusal_fields(index: int, item: Any) -> tuple[Mapping[str, Any], str, str]:
    """Validate one ``producer_refusals.json`` item's shape and field types
    -- presence, then type/value -- and return ``(raw_key, code, detail)``
    ready for ``BoardRequest`` construction. Raises ``ValueError`` (a
    fixed, index-naming message) for any problem; this is the stage split
    out of :func:`_decode_producer_refusals` to keep that function's own
    complexity within budget.
    """
    if not isinstance(item, Mapping):
        raise ValueError(
            f"producer_refusals.json refusal at index {index} must be an object")
    raw_key = item.get("key")
    if not isinstance(raw_key, Mapping):
        raise ValueError(
            f"producer_refusals.json refusal at index {index} has no \"key\" object")
    for field_name in ("ticker", "strategy", "event_date", "session"):
        if field_name not in raw_key:
            raise ValueError(
                f"producer_refusals.json refusal at index {index} key is "
                f"missing {field_name!r}")
    if "code" not in item:
        raise ValueError(
            f"producer_refusals.json refusal at index {index} is missing \"code\"")
    if "detail" not in item:
        raise ValueError(
            f"producer_refusals.json refusal at index {index} is missing \"detail\"")
    for field_name in ("ticker", "strategy", "event_date", "session"):
        if not isinstance(raw_key[field_name], str) or not raw_key[field_name]:
            raise ValueError(
                f"producer_refusals.json refusal at index {index} key "
                f"{field_name!r} must be a non-empty string")
    if not isinstance(item["code"], str) or not item["code"]:
        raise ValueError(
            f"producer_refusals.json refusal at index {index} \"code\" "
            f"must be a non-empty string")
    if not isinstance(item["detail"], str):
        raise ValueError(
            f"producer_refusals.json refusal at index {index} \"detail\" "
            f"must be a string")
    event_date_error = (
        f"producer_refusals.json refusal at index {index} key "
        f"\"event_date\" must be a canonical YYYY-MM-DD date or canonical "
        f"naive ISO datetime string")
    try:
        # Strict round trip through the SAME formatter this module's own
        # keys/documents encode with: only strings that are already their
        # own canonical identity survive (issue #356).
        _event_date_identity(raw_key["event_date"])
    except ValueError:
        raise ValueError(event_date_error) from None
    return raw_key, str(item["code"]), str(item["detail"])


def _decode_producer_refusals(doc: Mapping[str, Any]) -> tuple[NativeScoreBatchRowRefusal, ...]:
    """Decode ``producer_refusals.json``'s v1.0 document into the same typed
    per-row refusal shape ``run_native_score_batch_worker``'s own per-row
    loop already produces, so merging it reuses ``_native_score_batch_documents``'s
    existing records/refusals collision check UNCHANGED -- see that
    function's own docstring. A malformed document (wrong/missing
    ``schema_version``, a non-list ``"refusals"``, or a malformed item) is a
    whole-call ``ValueError``, matching this module's existing
    "events.json must be a JSON array" discipline for malformed caller
    input -- never a silently-dropped item.
    """
    if (not isinstance(doc, Mapping)
            or doc.get("schema_version") != "native_score_batch_producer_refusals.v1.0"):
        raise ValueError(
            "producer_refusals.json must be a "
            "native_score_batch_producer_refusals.v1.0 document")
    items = doc.get("refusals")
    if not isinstance(items, list):
        raise ValueError("producer_refusals.json's \"refusals\" must be a list")
    decoded = []
    for index, item in enumerate(items):
        raw_key, code, detail = _validated_producer_refusal_fields(index, item)
        key = BoardRequest(
            ticker=str(raw_key["ticker"]), strategy=str(raw_key["strategy"]),
            event_date=pd.Timestamp(raw_key["event_date"]), session=str(raw_key["session"]))
        if code == "INVALID_KEY_FIELD" and not any(
                "|" in value for value in (key.ticker, key.strategy, key.session)):
            raise ValueError(
                "producer refusal claims INVALID_KEY_FIELD for an encodable key")
        detail = detail[:_MAX_PRODUCER_DETAIL_LENGTH]
        decoded.append(NativeScoreBatchRowRefusal(key, code, detail))
    return tuple(decoded)


def _native_score_batch_documents(
    keys_in_order: Sequence[BoardRequest],
    records: Sequence[Any],
    refusals: Sequence[NativeScoreBatchRowRefusal],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the v2.0 ``records.json``/``refusals.json`` documents.

    Split out of :func:`run_native_score_batch_worker` purely to keep that
    function under its line budget -- see that function's docstring for the
    full contract.
    """
    # score_batch/score_many preserve request order and identity (see their
    # own docstrings), and the caller builds batch.requests from this SAME
    # assembled.values() iteration that keys_in_order comes from -- so
    # zipping keys_in_order (assembled's own dict-order keys) against
    # records is a safe, exact positional pairing within this one call.
    # This is NOT the events.json-vs-records.json positional pairing the
    # PR-7a design names as broken (that hazard is about reconstructing
    # order from the ORIGINAL per-event array after refusals have been
    # dropped; here nothing has been dropped or reordered between the two
    # zipped sequences).
    #
    # Issue #53 (the post-as_of panel-feature anchor gap) is fixed
    # upstream (#67): assemble_nightly_source_bundle's own required
    # panel_anchor parameter refuses POST_AS_OF_ROW per-row before a
    # bundle is ever built here. Nothing populates known_gaps today;
    # the key stays in the schema for a future gap this module might
    # need to flag.
    records_by_key = _keyed_by_board_request(
        (key, to_document(record))
        for key, record in zip(keys_in_order, records, strict=True)
    )
    unkeyable_refusals: list[dict[str, Any]] = [
        # No safe canonical string exists for these rows (that is what
        # "INVALID_KEY_FIELD" means) -- keep their raw structured key,
        # never a joined string that could collide or fail to re-parse.
        refusal.as_document() for refusal in refusals
        if refusal.code == "INVALID_KEY_FIELD"
    ]
    keyed_refusals = _keyed_by_board_request(
        (refusal.key, {"code": refusal.code, "detail": refusal.detail})
        for refusal in refusals if refusal.code != "INVALID_KEY_FIELD"
    )
    overlap = set(records_by_key) & set(keyed_refusals)
    if overlap:
        # The same duplicate-identity collision _keyed_by_board_request
        # already catches WITHIN one dict can also happen ACROSS the two:
        # one row with one identity succeeded (a record) while another row
        # with that SAME identity failed (a refusal), and neither dict's own
        # internal check can see the other dict at all.
        raise ValueError(
            "canonical BoardRequest key used by both a record and a "
            f"refusal: {sorted(overlap)!r}")
    records_document = {
        "schema_version": "native_score_batch_records.v2.0",
        "authoritative": False,
        "known_gaps": [],
        "records": records_by_key,
    }
    refusals_document = {
        "schema_version": "native_score_batch_refusals.v2.0",
        "refusals": keyed_refusals,
        "unkeyable_refusals": unkeyable_refusals,
    }
    return records_document, refusals_document


def _worker_event_documents(parameters: Mapping[str, Any], root: Path):
    events_path = root / "events.json"
    producer_refusals_path = root / "producer_refusals.json"
    from engine.v2.ops.errors import fail

    producer_mode = parameters.get("producer_mode", "staged")
    if producer_mode not in ("staged", "snapshot"):
        raise fail("VALIDATION_FAILED", "native score batch producer_mode is invalid")
    production_mode = producer_mode == "snapshot"
    has_catalog_path = bool(parameters.get("catalog_path"))
    has_objects_root = bool(parameters.get("objects_root"))
    if production_mode and has_catalog_path != has_objects_root:
        raise fail(
            "VALIDATION_FAILED",
            "native score batch production requires catalog_path and objects_root together")
    if production_mode and not has_catalog_path:
        raise fail(
            "VALIDATION_FAILED",
            "native score batch snapshot production requires catalog_path and objects_root",
            details={"missing": ("catalog_path", "objects_root")})
    if not production_mode:
        if not events_path.exists():
            raise FileNotFoundError(events_path)
        events_doc = json.loads(events_path.read_text(encoding="utf-8"))
        if not isinstance(events_doc, list):
            raise ValueError("events.json must be a JSON array")
        producer_doc = None
        if producer_refusals_path.exists():
            producer_doc = json.loads(producer_refusals_path.read_text(encoding="utf-8"))
        return events_doc, producer_doc, parameters["calendar_revision"]

    from engine.v2.data.repository import Repository
    from engine.v2.foundation import ArtifactStore
    from engine.v2.ops.catalog import connect
    from engine.v2.ops.nightly_raw_row_producer import build_native_score_batch_events

    conn = connect(parameters["catalog_path"], must_exist=True)
    try:
        # One snapshot is pinned for the whole build: memoise its resolution and
        # table membership instead of re-reading the catalog on every scan.
        repository = Repository(
            conn, ArtifactStore(parameters["objects_root"]), pin_snapshots=True)
        snapshot = repository.resolve(parameters["snapshot_id"])
        events_doc, producer_doc = build_native_score_batch_events(
            repository, snapshot, as_of=parameters["as_of"],
            horizon_days=parameters.get("horizon_days", 35),
            tickers=parameters.get("tickers") or None)
        calendar_revision = snapshot.table_versions[
            "earnings_events"].dataset_version_id
        events_path.write_text(canonical_json(events_doc), encoding="utf-8")
        producer_refusals_path.write_text(canonical_json(producer_doc), encoding="utf-8")
        return events_doc, producer_doc, calendar_revision
    finally:
        conn.close()


def run_native_score_batch_worker(parameters: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """The ``native_score_batch`` job kind's worker entrypoint.

    In production the worker builds the event and refusal documents from the
    pinned snapshot in its own worker process, while staged ``events.json``
    remains supported for direct callers. It resolves the release binding
    once from ``parameters["release_root"]``, assembles the batch with
    :func:`assemble_score_batch_inputs`, scores it under
    ``engine.v2.models.no_fit.no_fit_guard``, and writes ``records.json`` /
    ``refusals.json`` into ``root`` -- see ARCHITECTURE.md's
    ``native_score_batch.py`` section for the full contract.
    """
    from engine.v2.models.no_fit import no_fit_guard
    from engine.v2.scoring.release_bindings import resolve_gate_policy, resolve_release_binding

    as_of = parameters["as_of"]
    snapshot_id = parameters["snapshot_id"]
    binding = resolve_release_binding(parameters["release_root"])
    events_doc, producer_doc, calendar_revision = _worker_event_documents(parameters, root)
    events = tuple(_event_inputs_from_document(item) for item in events_doc)
    raw_feature_names = parameters.get("feature_names")
    if raw_feature_names is None:
        raw_feature_names = ()
    if not isinstance(raw_feature_names, (list, tuple)):
        raise ValueError("feature_names must be a list or tuple of strings")
    assembled, refusals = assemble_score_batch_inputs(
        as_of=as_of, snapshot_id=snapshot_id,
        calendar_revision=calendar_revision, binding=binding,
        events=events, feature_names=tuple(raw_feature_names),
        gate_policy=(parameters.get("gate_policy")
                     or resolve_gate_policy(binding, parameters["release_root"])),
    )
    if producer_doc is not None:
        refusals = refusals + _decode_producer_refusals(producer_doc)
    seen_unkeyable_identities: set[tuple[str, str, str, str]] = set()
    for refusal in refusals:
        if refusal.code != "INVALID_KEY_FIELD":
            continue
        identity = (refusal.key.ticker, refusal.key.strategy,
                    _event_date_identity(refusal.key.event_date), refusal.key.session)
        if identity in seen_unkeyable_identities:
            raise ValueError(
                f"duplicate unkeyable refusal identity: {identity!r}")
        seen_unkeyable_identities.add(identity)
    fields_by_request = {request_hash(request): inputs
                         for request, inputs in assembled.values()}
    batch_id = content_hash({
        "as_of": as_of, "snapshot_id": snapshot_id,
        "calendar_revision": calendar_revision,
    })
    keys_in_order = tuple(assembled.keys())
    batch = ScoreBatch(batch_id=batch_id, requests=tuple(r for r, _ in assembled.values()),
                       population_ref=snapshot_id)
    with no_fit_guard():
        records = score_batch(batch, fields_by_request)
    records_document, refusals_document = _native_score_batch_documents(
        keys_in_order, records, refusals)
    (root / "records.json").write_text(json.dumps(records_document, sort_keys=True,
                                                  separators=(",", ":")))
    (root / "refusals.json").write_text(json.dumps(
        refusals_document, sort_keys=True, separators=(",", ":")))
    return {
        "outputs": [
            {"name": "records", "path": "records.json",
             "schema": "native_score_batch_records.v2.0"},
            {"name": "refusals", "path": "refusals.json",
             "schema": "native_score_batch_refusals.v2.0"},
        ],
        "completed_ids": list(parameters["expected_ids"]),
        "no_work": not parameters["expected_ids"],
        "refused": [refusal.code for refusal in refusals],
    }
