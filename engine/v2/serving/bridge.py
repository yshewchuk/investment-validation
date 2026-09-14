"""Offline mapping of a verified score artifact to a verified render bundle.

Rearchitecture phase-3 guide §5.3 (P3-1a): "the serving contracts and the
offline legacy score bridge (mapping, identity and population validation)".

Pure, in-memory joins over already-decoded documents. No filesystem, no
network, no clock, and **no financial arithmetic** — every value that reaches
a :class:`~engine.v2.contracts.LegacyScoreBridge` is copied verbatim from the
score job's ``rows``/``ladder`` or from an already-rendered board row; the one
numeric operation this module performs is rounding a copy to the renderer's
own documented display precision so the two copies can be compared
(``dashboard/render.py`` ``BUNDLE_PRECISION`` = 6), never a new derived
quantity.

**No legacy imports, no ``engine.v2.ops`` import** (a peer at layer 7).
``score.json`` rows carry no ``event_id`` — confirmed at
``engine/score.py`` (``_EVENT_KEY`` docstring, ~line 3196) and
``engine/v2/ops/legacy_adapter.py::_population_key`` (~line 234), both of
which key an event by ``(ticker, event_date)`` only. Event identity here
always comes from the caller's ``event_refs``, resolved earlier through the
Phase 2 repository (guide §5.3 point 3); this module never invents one. A row
missing even ``ticker``/``event_date`` is a malformed score artifact, not an
ordinary unresolved mapping, and raises rather than producing a soft finding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from engine.v2.contracts import (
    EventRef,
    Finding,
    LegacyScoreBridge,
    ProjectionFindings,
    RowIdentity,
)
from engine.v2.foundation import content_hash

__all__ = ["LEGACY_DISPLAY_MAPPING_V1", "DisplayFieldSpec", "build_bridges"]


@dataclass(frozen=True)
class DisplayFieldSpec:
    """One checked display-field mapping (guide §5.3 point 6).

    ``source_path``: for a copied field, the engine-record key it comes from
    (always the same name as ``field`` in this renderer). For a ``derived``
    field the renderer computes itself, the engine field(s) it was computed
    from, for documentation only — this bridge never recomputes it or checks
    its value against those sources, per "consume the already-rendered
    values; never recompute them".

    ``exact``: True for ``dashboard/render.py`` ``REPLAY_INPUT_FIELDS`` —
    kept at full precision in the rendered row — and False for every other
    numeric field, which the renderer rounds to ``BUNDLE_PRECISION`` (6).
    """

    field: str
    source_path: str
    unit: str
    nullable: bool
    derived: bool
    exact: bool


#: ``dashboard/render.py`` ``REPLAY_INPUT_FIELDS``: kept at full precision in
#: the rendered row because the self-check reconstructs a request from them.
_EXACT_FIELDS = frozenset({
    "structure_params", "requested_strike", "strike", "strike_offset",
    "fill", "quote_max_age_sessions",
})

#: Fields ``compact_row`` computes itself rather than copying from the engine
#: record, mapped to the engine field(s) they were computed from (prose, for
#: the mapping spec only).
_DERIVED_SOURCES: dict[str, str] = {
    "row_id": "ticker+strategy+event_date+strike",
    "payoff_curve": "legs",
    "cost_over_width": "entry_cost+structure_width",
    "entry_cost_pct": "entry_cost+spot",
    "model_fair_pct": "model_versions+model_inputs",
    "premium_vs_fair": "entry_cost_pct+model_fair_pct",
    "model_vs_market": "driver_prediction+implied_move+driver_name",
    "scored": "exp_pnl_model+exp_pnl_analog",
    "rank": "board ranking across rows",
    "digest": "the full engine record",
}

#: Always present and never null on a real rendered row.
_NOT_NULLABLE = frozenset({"row_id", "ticker", "strategy", "event_date", "as_of",
                            "digest", "scored"})

#: Display unit per field, for the mapping spec's own documentation. Not used
#: for value conversion — this module never converts a unit, only compares a
#: copy to its source at the renderer's own declared precision.
_UNITS: dict[str, str] = {
    "row_id": "identity", "ticker": "identity", "strategy": "identity",
    "as_of": "date", "event_date": "date", "session": "string",
    "entry_date": "date", "exit_date": "date", "expiry": "date",
    "quote_date": "date", "forecast_fold": "date", "model_input_as_of": "date",
    "chain_last_obs": "date",
    "strike": "usd", "requested_strike": "usd", "spot": "usd",
    "entry_cost": "usd", "structure_width": "usd",
    "strike_offset": "ratio", "cost_over_width": "ratio",
    "premium_vs_fair": "ratio", "model_vs_market": "ratio", "rel_spread": "ratio",
    "entry_cost_pct": "percent", "model_fair_pct": "percent",
    "exp_pnl_model": "percent_of_spot", "exp_pnl_analog": "percent_of_spot",
    "exp_pnl_sim": "percent_of_spot", "model_p10": "percent_of_spot",
    "model_p90": "percent_of_spot", "ci_low": "percent_of_spot",
    "ci_high": "percent_of_spot", "forecast_abs_move": "percent_of_spot",
    "forecast_p10": "percent_of_spot", "forecast_p90": "percent_of_spot",
    "forecast_sd": "percent_of_spot", "driver_prediction": "percent_of_spot",
    "driver_p10": "percent_of_spot", "driver_p90": "percent_of_spot",
    "runup_move_prediction": "percent_of_spot", "runup_move_p10": "percent_of_spot",
    "runup_move_p90": "percent_of_spot", "implied_move": "percent_of_spot",
    "implied_move_at_entry": "percent_of_spot", "gate_score": "raw_score",
    "gate_threshold": "raw_score", "chosen_margin": "raw_score",
    "win_model": "probability", "win_analog": "probability", "win_sim": "probability",
    "n_analogs": "count", "analog_widened": "count", "menu_size": "count",
    "dte_entry": "days", "quote_age_sessions": "days",
    "quote_max_age_sessions": "days", "chain_age_days": "days",
    "runup_move_days": "days", "runup_move_scale": "ratio", "rank": "count",
    "gate_pass": "bool", "extrapolated": "bool", "scored": "bool",
    "forecast_model": "string", "chosen_strategy": "string", "driver_name": "string",
    "detail": "string", "digest": "hash",
    "legs": "object", "structure_params": "object", "model_versions": "object",
    "payoff_curve": "object", "flags": "list", "fill": "ratio",
}

#: ``dashboard/render.py`` ``_BOARD_FIELDS`` — ``compact_row``'s real output
#: keys, transcribed (this module may not import ``engine.dashboard``, a
#: legacy package). A test in ``tests/test_v2_serving_bridge.py`` asserts a
#: synthetic ``compact_row``-shaped dict's keys are all covered here.
_BOARD_FIELD_NAMES: tuple[str, ...] = (
    "row_id", "ticker", "strategy", "as_of", "event_date", "session",
    "entry_date", "exit_date", "strike", "strike_offset", "expiry",
    "quote_date", "quote_age_sessions", "quote_max_age_sessions", "requested_strike",
    "dte_entry", "spot", "entry_cost", "entry_cost_pct",
    "model_fair_pct", "premium_vs_fair",
    "exp_pnl_model", "win_model", "model_p10", "model_p90",
    "exp_pnl_analog", "win_analog", "ci_low", "ci_high",
    "n_analogs", "analog_widened",
    "gate_score", "gate_threshold", "gate_pass",
    "forecast_abs_move", "forecast_p10", "forecast_p90", "forecast_sd",
    "forecast_model", "forecast_fold", "structure_params",
    "structure_width", "cost_over_width", "rel_spread",
    "legs", "payoff_curve",
    "exp_pnl_sim", "win_sim",
    "chosen_strategy", "chosen_margin", "menu_size",
    "extrapolated", "flags", "model_versions",
    "driver_name", "driver_prediction", "driver_p10", "driver_p90",
    "runup_move_prediction", "runup_move_p10", "runup_move_p90",
    "runup_move_days", "runup_move_scale",
    "implied_move", "implied_move_at_entry", "model_vs_market", "model_input_as_of",
    "chain_last_obs", "chain_age_days",
    "scored", "rank", "fill", "detail", "digest",
)

LEGACY_DISPLAY_MAPPING_V1: tuple[DisplayFieldSpec, ...] = tuple(
    DisplayFieldSpec(
        field=name, source_path=_DERIVED_SOURCES.get(name, name),
        unit=_UNITS.get(name, "raw"), nullable=name not in _NOT_NULLABLE,
        derived=name in _DERIVED_SOURCES, exact=name in _EXACT_FIELDS,
    )
    for name in _BOARD_FIELD_NAMES
)


# --------------------------------------------------------------------------
# join keys and identity
# --------------------------------------------------------------------------


def _key_str(value: Any) -> str:
    return "" if value is None else str(value)


def _row_identity(row: Mapping[str, Any]) -> RowIdentity:
    return RowIdentity(
        ticker=str(row.get("ticker")), event_date=str(row.get("event_date")),
        strategy=str(row.get("strategy") or ""),
        expiry=_key_str(row.get("expiry")) or None,
        strike_key=_key_str(row.get("strike")) or None,
        discriminator=_key_str(row.get("strike_offset")) or None,
    )


def _join_key(row: Mapping[str, Any], event_ref: EventRef) -> tuple:
    """Guide §5.3 point 4: event revision, strategy, expiry, exact strike,
    plus ``strike_offset`` as the ladder discriminator. DYN-SV needs no
    special case: its own rows carry ``strategy == "DYN-SV"``, already
    distinct from the structure it chose."""
    return (
        event_ref.event_id, event_ref.calendar_revision,
        str(row.get("strategy")), _key_str(row.get("expiry")),
        _key_str(row.get("strike")), _key_str(row.get("strike_offset")),
    )


def _resolve_event(row: Mapping[str, Any],
                    event_refs: Mapping[tuple[str, str], EventRef | None]
                    ) -> tuple[EventRef | None, Finding | None]:
    """``event_refs`` is ``{(ticker, event_date): EventRef | None}``, mapped
    earlier through the Phase 2 repository. A missing key is unmapped; an
    explicit ``None`` value is an ambiguous mapping the caller could not
    resolve. Both refuse the row rather than invent an identity."""
    ticker, event_date = row.get("ticker"), row.get("event_date")
    if ticker is None or event_date is None:
        raise ValueError(
            "score/rendered row has no ticker/event_date -- score rows carry "
            "no event_id (engine/score.py _EVENT_KEY docstring; "
            "engine/v2/ops/legacy_adapter.py::_population_key); this is a "
            "malformed score artifact, not an ordinary unresolved mapping.")
    lookup_key = (str(ticker), str(event_date))
    if lookup_key not in event_refs:
        return None, Finding(
            code="EVENT_UNMAPPED", category="unresolved_event",
            message="no Phase 2 event/calendar mapping for this ticker+event_date",
            row=_row_identity(row))
    event_ref = event_refs[lookup_key]
    if event_ref is None:
        return None, Finding(
            code="EVENT_AMBIGUOUS", category="unresolved_event",
            message="Phase 2 event/calendar mapping was ambiguous for this ticker+event_date",
            row=_row_identity(row))
    return event_ref, None


def _index_rendered(bundle_rows_by_ticker: Mapping[str, list[dict]],
                     event_refs) -> tuple[dict[tuple, dict], list[Finding]]:
    """Every rendered row keyed by its join key. A key seen twice cannot be
    matched one-to-one (guide §5.3 point 4), so both copies are pulled from
    the index rather than guessed between."""
    index: dict[tuple, dict] = {}
    seen: set[tuple] = set()
    findings: list[Finding] = []
    for rows in bundle_rows_by_ticker.values():
        for row in rows:
            event_ref, finding = _resolve_event(row, event_refs)
            if finding is not None:
                findings.append(finding)
                continue
            key = _join_key(row, event_ref)
            if key in seen:
                index.pop(key, None)
                findings.append(Finding(
                    code="RENDERED_ROW_DUPLICATE_KEY", category="duplicate",
                    message="more than one rendered row shares this join key",
                    row=_row_identity(row)))
                continue
            seen.add(key)
            index[key] = row
    return index, findings


# --------------------------------------------------------------------------
# display-value comparison
# --------------------------------------------------------------------------


def _round6(value: Any) -> Any:
    """The renderer's own ``BUNDLE_PRECISION``, applied to a COPY for
    comparison only -- never a new derived quantity."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [_round6(v) for v in value]
    if isinstance(value, dict):
        return {k: _round6(v) for k, v in value.items()}
    return value


def _values_match(source_value: Any, display_value: Any, *, exact: bool) -> bool:
    if exact:
        return source_value == display_value
    return _round6(source_value) == display_value


def _compare_one_field(spec: DisplayFieldSpec, engine_record: Mapping[str, Any],
                        display_record: Mapping[str, Any],
                        row_identity: RowIdentity) -> list[Finding]:
    source_value = engine_record.get(spec.source_path)
    display_present = spec.field in display_record
    display_value = display_record.get(spec.field)
    if source_value is None:
        if display_present and display_value is not None:
            return [Finding(code="NULL_TURNED_TO_VALUE", category="null_mask",
                             message=f"{spec.field} is null in the engine record "
                                     "but not null in the display record",
                             row=row_identity, field_name=spec.field)]
        return []
    if not display_present or display_value is None:
        return [Finding(code="VALUE_DROPPED", category="null_mask",
                         message=f"{spec.field} has a value in the engine record "
                                 "but is null or absent in the display record",
                         row=row_identity, field_name=spec.field)]
    if not _values_match(source_value, display_value, exact=spec.exact):
        return [Finding(code="VALUE_MISMATCH", category="value",
                         message=f"{spec.field} display value does not match its "
                                 "engine source",
                         row=row_identity, field_name=spec.field)]
    return []


def _compare_display(engine_record: Mapping[str, Any], display_record: Mapping[str, Any],
                      row_identity: RowIdentity) -> list[Finding]:
    findings: list[Finding] = []
    for spec in LEGACY_DISPLAY_MAPPING_V1:
        if spec.derived or spec.source_path not in engine_record:
            continue
        findings.extend(_compare_one_field(spec, engine_record, display_record, row_identity))
    return findings


def _model_ref_finding(row: Mapping[str, Any],
                        model_registry_artifact_refs: tuple[str, ...]) -> Finding | None:
    scored = row.get("exp_pnl_model") is not None or row.get("exp_pnl_analog") is not None
    if scored and not model_registry_artifact_refs:
        return Finding(code="MODEL_REF_MISSING", category="model_ref",
                       message="row was scored but no model registry artifact ref was pinned",
                       row=_row_identity(row))
    return None


# --------------------------------------------------------------------------
# per-row identity (legacy_row_id / source_row_key), reimplemented locally
# --------------------------------------------------------------------------


def _legacy_row_id(row: Mapping[str, Any]) -> str:
    """Mirrors ``engine.v2.ops.decision_replay.score_row_id`` (string join,
    no arithmetic) without importing ``engine.v2.ops``, a peer package this
    layer may not import. Main-board rows already carry ``row_id``; ladder
    rows do not (``strike_ladder`` never adds one), so this covers them."""
    return "|".join(str(row.get(k, "")) for k in
                     ("ticker", "strategy", "event_date", "strike", "expiry"))


def _population_key(row: Mapping[str, Any]) -> str:
    """Mirrors ``engine.v2.ops.legacy_adapter._population_key``: the planned
    population is keyed before strike/expiry exist."""
    return "|".join(str(row.get(k, "")) for k in ("ticker", "strategy", "event_date"))


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def _build_bridge(row: Mapping[str, Any], display_row: Mapping[str, Any], event_ref: EventRef,
                   *, score_batch_ref: str, snapshot_ref: str,
                   model_registry_artifact_refs: tuple[str, ...],
                   request_provenance_refs: tuple[str, ...]) -> LegacyScoreBridge:
    legacy_row_id = str(row.get("row_id") or _legacy_row_id(row))
    # Guide §5.3 point 5: "hash the full source record and pinned score-
    # batch/request dependencies" -- every pinned ref that could change what
    # this score MEANS (batch, request provenance, model registry, snapshot)
    # folds into the identity; nothing operational (a wall-clock write time,
    # a retry count) is ever a parameter of this call, so none can leak in.
    score_id = content_hash({
        "engine_record": row, "score_batch_ref": score_batch_ref,
        "request_provenance_refs": list(request_provenance_refs),
        "model_registry_artifact_refs": list(model_registry_artifact_refs),
        "snapshot_ref": snapshot_ref,
    })
    return LegacyScoreBridge(
        score_id=score_id, event_ref=event_ref, clock_id=str(row.get("as_of") or ""),
        legacy_row_id=legacy_row_id, score_batch_ref=score_batch_ref,
        source_row_key=_population_key(row), source_record_hash=content_hash(row),
        request_provenance_refs=tuple(request_provenance_refs), snapshot_ref=snapshot_ref,
        model_registry_artifact_refs=tuple(model_registry_artifact_refs),
        engine_record=dict(row), display_record=dict(display_row),
    )


def _bridge_one(row: Mapping[str, Any], rendered_index: dict, used_keys: set,
                 event_refs, seen_source_keys: set, *, score_batch_ref: str,
                 snapshot_ref: str, model_registry_artifact_refs: tuple[str, ...],
                 request_provenance_refs: tuple[str, ...]
                 ) -> tuple[LegacyScoreBridge | None, list[Finding]]:
    event_ref, finding = _resolve_event(row, event_refs)
    if finding is not None:
        return None, [finding]
    key = _join_key(row, event_ref)
    if key in seen_source_keys:
        return None, [Finding(code="SOURCE_ROW_DUPLICATE_KEY", category="duplicate",
                              message="more than one source row shares this join key",
                              row=_row_identity(row))]
    seen_source_keys.add(key)
    display_row = rendered_index.get(key)
    if display_row is None:
        return None, [Finding(code="ROW_NOT_RENDERED", category="missing",
                              message="a planned/scored row has no matching rendered row",
                              row=_row_identity(row))]
    used_keys.add(key)
    row_identity = _row_identity(row)
    findings = _compare_display(row, display_row, row_identity)
    model_finding = _model_ref_finding(row, model_registry_artifact_refs)
    if model_finding is not None:
        findings.append(model_finding)
    bridge = _build_bridge(row, display_row, event_ref, score_batch_ref=score_batch_ref,
                           snapshot_ref=snapshot_ref,
                           model_registry_artifact_refs=model_registry_artifact_refs,
                           request_provenance_refs=request_provenance_refs)
    return bridge, findings


def _planned_population_findings(planned: tuple[str, ...],
                                  source_main: list[dict]) -> list[Finding]:
    observed = {_population_key(row) for row in source_main}
    missing = sorted(set(planned) - observed)
    unplanned = sorted(observed - set(planned))
    findings = [Finding(code="PLANNED_ROW_MISSING", category="missing",
                        message=f"planned population key {key!r} has no scored row",
                        details={"population_key": key}) for key in missing]
    findings += [Finding(code="SCORED_ROW_UNPLANNED", category="unplanned",
                         message=f"scored row key {key!r} was not in the planned population",
                         details={"population_key": key}) for key in unplanned]
    return findings


def _unplanned_rendered_findings(rendered_index: dict, used_keys: set) -> list[Finding]:
    return [
        Finding(code="RENDERED_ROW_UNPLANNED", category="unplanned",
                message="a rendered row has no matching source row",
                row=_row_identity(row))
        for key, row in rendered_index.items() if key not in used_keys
    ]


def _findings_result(planned: tuple[str, ...], bundle_rows_by_ticker: Mapping[str, list[dict]],
                      matched: int, findings: list[Finding]) -> ProjectionFindings:
    rendered_main = sum(1 for rows in bundle_rows_by_ticker.values()
                        for row in rows if row.get("strike_offset") is None)
    rendered_ladder = sum(1 for rows in bundle_rows_by_ticker.values()
                          for row in rows if row.get("strike_offset") is not None)
    ok = matched > 0 and not findings
    return ProjectionFindings(
        planned_population=len(planned), rendered_main_population=rendered_main,
        rendered_ladder_population=rendered_ladder, matched_population=matched,
        compared_population=matched, findings=tuple(findings), ok=ok)


def build_bridges(score_doc: Mapping[str, Any], bundle_rows_by_ticker: Mapping[str, list[dict]],
                   event_refs: Mapping[tuple[str, str], EventRef | None], *,
                   score_batch_ref: str, snapshot_ref: str,
                   model_registry_artifact_refs: tuple[str, ...],
                   request_provenance_refs: tuple[str, ...]
                   ) -> tuple[list[LegacyScoreBridge], ProjectionFindings]:
    """Guide §5.3: join a verified ``score.json`` to a verified rendered bundle.

    ``event_refs`` is ``{(ticker, event_date): EventRef | None}``, resolved
    earlier through the Phase 2 repository; ``None`` marks an ambiguous
    mapping, a missing key an unmapped one, and either refuses the affected
    rows. The planned population comes from ``score_doc["expected_population"]``,
    never from whichever rows happened to render. Main-board and ladder rows
    are counted separately in the returned funnel.
    """
    planned = tuple(score_doc.get("expected_population") or ())
    source_main = list(score_doc.get("rows") or [])
    source_ladder = list(score_doc.get("ladder") or [])
    rendered_index, findings = _index_rendered(bundle_rows_by_ticker, event_refs)
    used_keys: set[tuple] = set()
    seen_source_keys: set[tuple] = set()
    bridges: list[LegacyScoreBridge] = []
    for row in source_main + source_ladder:
        bridge, row_findings = _bridge_one(
            row, rendered_index, used_keys, event_refs, seen_source_keys,
            score_batch_ref=score_batch_ref, snapshot_ref=snapshot_ref,
            model_registry_artifact_refs=model_registry_artifact_refs,
            request_provenance_refs=request_provenance_refs)
        findings.extend(row_findings)
        if bridge is not None:
            bridges.append(bridge)
    findings.extend(_planned_population_findings(planned, source_main))
    findings.extend(_unplanned_rendered_findings(rendered_index, used_keys))
    return bridges, _findings_result(planned, bundle_rows_by_ticker, len(bridges), findings)
