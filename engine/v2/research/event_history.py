"""Holdout-admitted historical features over physically isolated native outcomes."""
from __future__ import annotations

from engine.v2.data import errors
from engine.v2.data import query as query_planner
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT
from engine.v2.features.panel_math import history_features
from engine.v2.foundation import content_hash
from engine.v2.foundation.experiment_holdouts import _calendar_date
from engine.v2.research.experiment_population import load_population
from engine.v2.research.prediction_inputs import _parse_target, _target_filters, _target_query

_CONTEXT = ("snapshot_id", "holdout_as_of_month", "random_membership_version",
            "rolling_membership_version", "population_use")


def _window(history_start, decision_session):
    """Normalize explicit calendar days without silently selecting a cutoff."""
    try:
        start, decision = _calendar_date(history_start), _calendar_date(decision_session)
        if start > decision:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise errors.fail("CONTRACT_MISMATCH", "an ordered calendar-day history window is required") from None
    return start, decision


def _plans(repository, snapshot, candidates, contract):
    """Shared history/target guard: prove all object membership before any opens."""
    records = repository.fragment_records(snapshot, "computed_moves")
    plans = []
    for event in candidates.itertuples(index=False):
        keys, interval = _target_filters(event)
        population = query_planner.plan_scan_population(
            contract, records, key_filter=keys, time_interval=interval)
        day = event.event_date.date().isoformat()
        key = (event.ticker, day)
        if len(population.records) != 1 or population.row_count != 1:
            raise errors.fail("HOLDOUT_ACCESS_DENIED", "research requires event-isolated outcome fragments")
        record = population.records[0]
        if (record.primary_key_min != key or record.primary_key_max != key
                or record.partition_key != event.ticker
                or record.time_min != day or record.time_max != day):
            raise errors.fail("HOLDOUT_ACCESS_DENIED", "research outcome isolation cannot be proven")
        request = _target_query(repository, snapshot, contract, event,
                                population_bound=population.row_count)
        plans.append((event, request, record))
    return plans


def _historical_target(rows, event, decision):
    """Omit unavailable decoded labels before numeric validation or arithmetic."""
    if len(rows) != 1 or rows[0]["event_date"] != event.event_date.date().isoformat():
        raise errors.fail("EXPERIMENT_VARIANT_FAILED", "historical outcome identity is invalid")
    row = rows[0]
    if row["skipped"] is True:
        return None, "skipped"
    if row["available_as_of_date"] is None:
        return None, "unavailable"
    try:
        available = _calendar_date(row["available_as_of_date"])
        if available <= event.event_date.date():
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise errors.fail("EXPERIMENT_VARIANT_FAILED", "historical target availability is invalid") from None
    if available > decision:
        return None, "unavailable"
    return _parse_target(rows, event.event_date.date()), None


def load_event_history(repository, snapshot, *, event_id, as_of_month,
                       decision_session, history_start, purpose="selection"):
    """One canonical target's causal history, with pins and explicit missingness."""
    start, decision = _window(history_start, decision_session)
    target = load_population(repository, snapshot, as_of_month=as_of_month,
                             purpose=purpose, event_ids=[event_id]).iloc[0]
    event_day = target.event_date.date()
    if decision > event_day:
        raise errors.fail("CONTRACT_MISMATCH", "history decision cannot follow the target event")
    population = load_population(repository, snapshot, as_of_month=as_of_month, purpose=purpose)
    candidates = population[(population.ticker == target.ticker)
        & (population.event_date.dt.date >= start)
        & (population.event_date.dt.date < min(event_day, decision))]
    candidates = candidates.sort_values(["event_date", "event_id"])
    contract = repository.table_contract(snapshot, "computed_moves")
    if contract != COMPUTED_MOVES_CONTRACT:
        raise errors.fail("CONTRACT_MISMATCH", "history requires the native computed-moves contract")
    plans = _plans(repository, snapshot, candidates, contract)
    history, omitted, moves = [], [], []
    for event, request, record in plans:
        rows = [row for batch in repository.scan(request, table_name="computed_moves")
                for row in batch.to_pylist()]
        outcome, reason = _historical_target(rows, event, decision)
        if reason:
            omitted.append({"event_id": event.event_id, "reason": reason})
            continue
        moves.append(outcome[0])
        history.append({"event_id": event.event_id, "event_date": event.event_date.date().isoformat(),
            "target_available_on": outcome[2], "fragment_id": record.fragment_id,
            "object_id": record.object_ref.object_id, "content_hash": record.object_ref.content_hash})
    context = {name: target[name] for name in _CONTEXT}
    context.update(event_id=event_id, ticker=target.ticker, event_date=event_day.isoformat(),
        history_start=start.isoformat(), decision_session=decision.isoformat(), purpose=purpose,
        computed_moves_version=snapshot.table_versions["computed_moves"].dataset_version_id)
    read_inputs = [{"event_id": event.event_id, "fragment_id": record.fragment_id,
        "object_id": record.object_ref.object_id, "content_hash": record.object_ref.content_hash}
        for event, _, record in plans]
    identity = {"schema_version": "research_event_history.v2", **context,
                "history": history, "omitted_history": omitted, "read_inputs": read_inputs}
    return {**identity, "history_id": content_hash(identity),
            "history_status": "available" if history else "empty",
            "features": history_features(moves, [abs(value) for value in moves]),
            "holdout_exclusions": population.attrs["holdout_exclusions"]}
