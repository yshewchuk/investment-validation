"""Holdout-admitted historical features over physically isolated native outcomes."""
from __future__ import annotations

from engine.v2.data import errors
from engine.v2.data import query as query_planner
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT
from engine.v2.features.panel_math import history_features
from engine.v2.foundation import content_hash
from engine.v2.foundation.experiment_holdouts import calendar_date
from engine.v2.research.experiment_population import load_population
from engine.v2.research.prediction_inputs import _parse_target, _target_filters, _target_query
from engine.v2.research.progress import Progress, stage

_CONTEXT = ("snapshot_id", "holdout_as_of_month", "random_membership_version",
            "rolling_membership_version", "population_use")
POPULATION_CONTEXT_COLUMNS = _CONTEXT


def plan_event_targets(repository, snapshot, candidates, contract, *, columns=None):
    """Prove singleton object membership for already holdout-admitted candidates.

    Metadata only; callers still own population admission and payload validation.
    """
    with stage("plan event targets"):
        try:
            if not {"ticker", "event_date"}.issubset(candidates.columns):
                raise ValueError
            for event in candidates.itertuples(index=False):
                if (not isinstance(event.ticker, str) or not event.ticker.strip()
                        or event.ticker != event.ticker.strip()
                        or calendar_date(event.event_date) != event.event_date.date()):
                    raise ValueError
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            raise errors.fail("CONTRACT_MISMATCH", "event target candidates are malformed") from None
        return _plans(repository, snapshot, candidates, contract, columns=columns)


def _window(history_start, decision_session):
    """Normalize explicit calendar days without silently selecting a cutoff."""
    try:
        start, decision = calendar_date(history_start), calendar_date(decision_session)
        if start > decision:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise errors.fail("CONTRACT_MISMATCH", "an ordered calendar-day history window is required") from None
    return start, decision


def _canonical_interval(record):
    """Index only time bounds with the planner's canonical calendar ordering."""
    lo, hi = record.time_min, record.time_max
    try:
        if (not all(isinstance(value, str) and value[:4] >= "1000"
                    and calendar_date(value).isoformat() == value for value in (lo, hi)) or lo > hi):
            return None
    except (ValueError, TypeError, OverflowError):
        return None
    return lo, hi


def _interval_node(entries):
    """Balanced centered intervals; input is already sorted by lower bound."""
    if not entries:
        return None
    center = entries[len(entries) // 2][0]
    left, middle, right = [], [], []
    for entry in entries:
        (left if entry[1] < center else right if entry[0] > center else middle).append(entry)
    by_end = sorted(middle, key=lambda entry: entry[1])
    return center, _interval_node(left), _interval_node(right), middle, by_end


def _point_entries(node, day):
    """Yield every inclusive interval overlap without scanning unrelated ranges."""
    if node is None:
        return
    center, left, right, by_start, by_end = node
    if day < center:
        for entry in by_start:
            if entry[0] > day:
                break
            yield entry
        yield from _point_entries(left, day)
    elif day > center:
        for entry in reversed(by_end):
            if entry[1] < day:
                break
            yield entry
        yield from _point_entries(right, day)
    else:
        yield from by_start


def _record_index(records, contract=COMPUTED_MOVES_CONTRACT):
    """One metadata pass; invalid keys survive even partition/time pruning."""
    partitions, invalid = {}, []
    for ordinal, record in enumerate(records):
        try:
            query_planner.fragment_key_bounds(record, contract)
        except errors.DataError:
            # Defer refusal to the planner: query validation and earlier
            # membership failures must still win in their original order.
            invalid.append((ordinal, record))
            continue
        entries, fallback, all_records = partitions.setdefault(record.partition_key, ([], [], []))
        all_records.append((ordinal, record))
        bounds = _canonical_interval(record)
        if bounds is None:
            fallback.append((ordinal, record))
        else:
            entries.append((*bounds, ordinal, record))
    return ({key: (_interval_node(sorted(entries, key=lambda entry: entry[0])), fallback, all_records)
             for key, (entries, fallback, all_records) in partitions.items()}, invalid)


def _indexed_records(index, ticker, day):
    """Restore manifest order before the unchanged planner validates survivors."""
    partitions, invalid = index
    node, fallback, all_records = partitions.get(ticker, (None, (), ()))
    if day[:4] < "1000":  # Platform timestamp normalization may not zero-pad these years.
        selected = [*invalid, *all_records]
    else:
        selected = [*invalid, *fallback,
                    *((ordinal, record) for _, _, ordinal, record in _point_entries(node, day))]
    return [record for _, record in sorted(selected, key=lambda entry: entry[0])]


def _plans(repository, snapshot, candidates, contract, *, columns=None):
    """Shared history/target guard: prove all object membership before any opens."""
    projection = {} if columns is None else {"columns": columns}
    index = _record_index(repository.fragment_records(snapshot, contract.table_name), contract)
    plans = []
    progress = Progress("plan history candidates", len(candidates))
    for event in candidates.itertuples(index=False):
        keys, interval = _target_filters(event)
        day = event.event_date.date().isoformat()
        population = query_planner.plan_scan_population(
            contract, _indexed_records(index, event.ticker, day), key_filter=keys, time_interval=interval)
        key = (event.ticker, day)
        if len(population.records) != 1 or population.row_count != 1:
            raise errors.fail("HOLDOUT_ACCESS_DENIED", "research requires event-isolated outcome fragments")
        record = population.records[0]
        if (record.primary_key_min != key or record.primary_key_max != key
                or record.partition_key != event.ticker
                or record.time_min != day or record.time_max != day):
            raise errors.fail("HOLDOUT_ACCESS_DENIED", "research outcome isolation cannot be proven")
        request = _target_query(repository, snapshot, contract, event,
                                population_bound=population.row_count, **projection)
        plans.append((event, request, record))
        progress.tick()
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
        available = calendar_date(row["available_as_of_date"])
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

    def read_rows(event, request):
        """Scan one outcome; the caller validates it before requesting the next."""
        return [row for batch in repository.scan(request, table_name="computed_moves")
                for row in batch.to_pylist()]
    return _assemble_history(snapshot, target, population, plans, start=start, decision=decision,
                             purpose=purpose, read_rows=read_rows)


def _assemble_history(snapshot, target, population, plans, *, start, decision, purpose, read_rows):
    """Shared arithmetic/provenance over admitted, ordered, physically preflighted plans."""
    event_id, event_day = target.event_id, target.event_date.date()
    history, omitted, moves = [], [], []
    for event, request, record in plans:
        rows = read_rows(event, request)
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
