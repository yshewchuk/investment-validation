"""Protected earnings-magnitude labels and a separate full-session forecast clock."""
from __future__ import annotations

from types import SimpleNamespace

from engine.v2.data import errors
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_V5_CONTRACT
from engine.v2.foundation import content_hash, market_calendar
from engine.v2.foundation.experiment_holdouts import _calendar_date
from engine.v2.research.event_history import _CONTEXT, _plans, skipped_reason
from engine.v2.research.experiment_population import load_population
from engine.v2.research.prediction_inputs import _parse_target


def magnitude_recipe():
    """Fresh fixed recipe bytes; no caller-selected price basis or hidden scaling."""
    return {"schema_version": "earnings_magnitude.v1", "units": "percentage_points",
            "target": "absolute_earnings_close_to_close_return",
            "price_basis": "split_neutral_non_dividend_adjusted",
            "estimand": "conditional_mean", "primary_loss": "pooled_rmse",
            "secondary_diagnostics": ["mae", "quantile_calibration"]}


def forecast_cutoff(event_date, session, horizon, calendar):
    """Source-session ceiling, not proof of intraday source/schedule availability."""
    try:
        day = _calendar_date(event_date)
        if type(horizon) is not int or horizon not in (5, 3, 1):
            raise ValueError
        if not isinstance(calendar, market_calendar.CalendarSessions):
            raise ValueError
        observed = _calendar_date(calendar.observed_through)
        key = SimpleNamespace(event_date=day, session=session, strategy="STR-RUNUP")
        anchor = market_calendar.planned_exit_date(key, calendar)
        index = calendar.days.index(anchor) - horizon
        if index < 0 or anchor > observed.isoformat() or calendar.observed_through not in calendar.days:
            raise ValueError
        ceiling = calendar.days[index]
        # The native rule calendar detects missing interior sessions without weekday subtraction.
        complete = market_calendar.build_calendar_sessions((ceiling,), event_through=day)
        if anchor != market_calendar.planned_exit_date(key, complete):
            raise ValueError
        expected = tuple(d for d in complete.days if d <= anchor)
        if calendar.days[index:index + horizon + 1] != expected:
            raise ValueError
    except (ValueError, TypeError, OverflowError, AttributeError):
        raise errors.fail("CONTRACT_MISMATCH", "invalid or insufficient full-session forecast calendar") from None
    identity = {"schema_version": "earnings_forecast_cutoff.v1", "event_date": day.isoformat(),
        "session": session, "horizon_sessions": horizon, "pre_announcement_close": anchor,
        "source_session_ceiling": ceiling, "calendar_id": content_hash({
            "days": list(calendar.days), "observed_through": calendar.observed_through})}
    return {**identity, "cutoff_id": content_hash(identity)}


def _label_before(rows, event, cutoff):
    """Check the recorded clock before consuming its numeric outcome."""
    try:
        if len(rows) != 1 or _calendar_date(rows[0]["available_as_of_date"]) >= cutoff:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise errors.fail("EXPERIMENT_VARIANT_FAILED", "magnitude label is not available before cutoff") from None
    return _parse_target(rows, event.event_date.date())


def _label_cutoff(labels_before):
    """Require an explicit canonical label clock, independent of forecast clocks."""
    try:
        return _calendar_date(labels_before)
    except (ValueError, TypeError, OverflowError):
        raise errors.fail("CONTRACT_MISMATCH", "an explicit label-availability cutoff is required") from None


def _candidate_plans(repository, snapshot, as_of_month, purpose, event_ids, cutoff=None):
    """Admit the complete population and physical membership before any label opens."""
    population = load_population(repository, snapshot, as_of_month=as_of_month,
                                 purpose=purpose, event_ids=event_ids)
    eligible = population.event_date.dt.date < cutoff if cutoff is not None else population.event_id.notna()
    if event_ids is not None and not eligible.all():
        raise errors.fail("CONTRACT_MISMATCH", "requested magnitude events must precede label cutoff")
    rows = population[eligible].sort_values(["event_date", "event_id"]).reset_index(drop=True)
    if rows.empty:
        raise errors.fail("POPULATION_COLLAPSED", "no magnitude events precede label cutoff")
    contract = repository.table_contract(snapshot, "computed_moves")
    if contract not in (COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_V5_CONTRACT):
        raise errors.fail("CONTRACT_MISMATCH", "magnitude labels require native computed moves")
    return population, eligible, rows, _plans(repository, snapshot, rows, contract)


def _object_input(event, record):
    """Detach the exact object membership consumed or planned for one event."""
    return {"event_id": event.event_id, "fragment_id": record.fragment_id,
            "object_id": record.object_ref.object_id, "content_hash": record.object_ref.content_hash}


def load_magnitude_targets(repository, snapshot, *, as_of_month, labels_before,
                           purpose="selection", event_ids=None, omit_not_computed=False):
    """Admit metadata and every physical object before reading any magnitude label."""
    if type(omit_not_computed) is not bool:
        raise errors.fail("CONTRACT_MISMATCH", "omit_not_computed must be a boolean")
    cutoff = _label_cutoff(labels_before)
    population, eligible, rows, plans = _candidate_plans(
        repository, snapshot, as_of_month, purpose, event_ids, cutoff)
    signed, available, consumed, omitted, positions = [], [], [], [], []
    for index, (event, request, record) in enumerate(plans):
        decoded = [row for batch in repository.scan(request, table_name="computed_moves")
                   for row in batch.to_pylist()]
        consumed.append(_object_input(event, record))
        if (omit_not_computed and len(decoded) == 1
                and type(decoded[0].get("skipped")) is bool and decoded[0]["skipped"]
                and decoded[0].get("event_date") == event.event_date.date().isoformat()):
            omitted.append({"event_id": event.event_id, "reason": skipped_reason(decoded[0])})
            continue
        value, _, known = _label_before(decoded, event, cutoff)
        signed.append(value)
        available.append(known)
        positions.append(index)
    candidates = rows
    rows = candidates.iloc[positions].reset_index(drop=True)
    rows["realized_move_pct"], rows["absolute_move_pct"] = signed, [abs(value) for value in signed]
    rows["target_available_on"] = available
    rows["target_object_id"] = [consumed[index]["object_id"] for index in positions]
    recipe = magnitude_recipe()
    identity = {"schema_version": "research_magnitude_targets.v1",
        **{name: candidates.iloc[0][name] for name in _CONTEXT}, "purpose": purpose,
        "computed_moves_version": snapshot.table_versions["computed_moves"].dataset_version_id,
        "labels_before": cutoff.isoformat(), "target_recipe": recipe,
        "target_recipe_id": content_hash(recipe), "consumed_inputs": consumed,
        "excluded_future_events": sorted(population.loc[~eligible, "event_id"].tolist())}
    if omitted:
        identity["omitted_labels"] = sorted(omitted, key=lambda item: item["event_id"])
    rows.attrs.update(identity, inputs_id=content_hash(identity))
    return rows


def preflight_magnitude_targets(repository, snapshot, *, as_of_month, event_ids):
    """Prove an explicit run-wide participant union without opening outcome objects."""
    if event_ids is None:
        raise errors.fail("CONTRACT_MISMATCH", "magnitude preflight requires explicit participants")
    _, _, rows, plans = _candidate_plans(repository, snapshot, as_of_month, "selection", event_ids)
    identity = {"schema_version": "research_magnitude_preflight.v1",
        **{name: rows.iloc[0][name] for name in _CONTEXT}, "purpose": "selection",
        "computed_moves_version": snapshot.table_versions["computed_moves"].dataset_version_id,
        "event_ids": rows.event_id.tolist(),
        "planned_inputs": [_object_input(event, record) for event, _, record in plans]}
    return {**identity, "preflight_id": content_hash(identity)}


def _training_label(rows, event, cutoff):
    """Qualify the recorded availability before parsing any training numeric value."""
    day = event.event_date.date()
    if (len(rows) != 1 or rows[0]["event_date"] != day.isoformat()
            or type(rows[0]["skipped"]) is not bool):
        raise errors.fail("EXPERIMENT_VARIANT_FAILED", "magnitude training label identity is invalid")
    row = rows[0]
    if row["skipped"]:
        return None, skipped_reason(row)
    if row["available_as_of_date"] is None:
        return None, "unavailable"
    try:
        available = _calendar_date(row["available_as_of_date"])
        if available <= day:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise errors.fail("EXPERIMENT_VARIANT_FAILED", "magnitude training availability is invalid") from None
    if available >= cutoff:
        return None, "not_available_before_fit"
    return _parse_target(rows, day), None


def qualify_magnitude_training_targets(repository, snapshot, *, as_of_month, labels_before, event_ids):
    """Return qualified magnitudes with all opened objects and explicit omissions bound."""
    cutoff = _label_cutoff(labels_before)
    if event_ids is None:
        raise errors.fail("CONTRACT_MISMATCH", "magnitude training requires explicit candidates")
    _, _, candidates, plans = _candidate_plans(
        repository, snapshot, as_of_month, "selection", event_ids, cutoff)
    positions, targets, read_inputs, omitted = [], [], [], []
    for index, (event, request, record) in enumerate(plans):
        decoded = [row for batch in repository.scan(request, table_name="computed_moves")
                   for row in batch.to_pylist()]
        target, reason = _training_label(decoded, event, cutoff)
        read_inputs.append(_object_input(event, record))
        if reason is not None:
            omitted.append({"event_id": event.event_id, "reason": reason})
        else:
            positions.append(index)
            targets.append(target)
    recipe = magnitude_recipe()
    rows = candidates.iloc[positions].copy()
    identity = {"schema_version": "research_magnitude_training_targets.v1",
        **{name: candidates.iloc[0][name] for name in _CONTEXT}, "purpose": "selection",
        "computed_moves_version": snapshot.table_versions["computed_moves"].dataset_version_id,
        "labels_before": cutoff.isoformat(), "target_recipe": recipe, "target_recipe_id": content_hash(recipe),
        "candidate_event_ids": candidates.event_id.tolist(), "qualified_event_ids": rows.event_id.tolist(),
        "read_inputs": read_inputs, "omitted_labels": omitted}
    if rows.empty:
        raise errors.fail("POPULATION_COLLAPSED", "no magnitude training labels precede the fit boundary",
                          details={"qualification": identity, "inputs_id": content_hash(identity)})
    rows["realized_move_pct"] = [target[0] for target in targets]
    rows["absolute_move_pct"] = [abs(target[0]) for target in targets]
    rows["target_available_on"] = [target[2] for target in targets]
    rows["target_object_id"] = [read_inputs[index]["object_id"] for index in positions]
    rows.attrs.update(identity, inputs_id=content_hash(identity))
    return rows.reset_index(drop=True)
