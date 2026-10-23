"""Holdout-admitted historical features over physically isolated native outcomes."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

from engine.v2.contracts.data import KeyPredicate, TimeInterval
from engine.v2.data import errors
from engine.v2.data import query as query_planner
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_V5_CONTRACT
from engine.v2.features.panel_math import history_features
from engine.v2.foundation import content_hash
from engine.v2.foundation.experiment_holdouts import calendar_date
from engine.v2.research.experiment_population import load_population
from engine.v2.research.prediction_inputs import _parse_target, _target_filters, _target_query
from engine.v2.research.progress import Progress, decision, stage

_CONTEXT = ("snapshot_id", "holdout_as_of_month", "random_membership_version",
            "rolling_membership_version", "population_use")
POPULATION_CONTEXT_COLUMNS = _CONTEXT

_MAX_OPEN_RUN_STREAMS = 32  # interleaved tickers keep their streams; the oldest is closed beyond this


def plan_event_targets(repository, snapshot, candidates, contract, *, columns=None):
    """Prove singleton object membership for already holdout-admitted candidates.

    Metadata only; callers still own population admission and payload validation.
    """
    with stage("plan event targets"):
        _check_candidates(candidates)
        return _plans(repository, snapshot, candidates, contract, columns=columns)


def _check_candidates(candidates):
    """Refuse malformed candidate frames before any planning."""
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
    index = _record_index(repository.fragment_records(snapshot, contract.table_name), contract)
    return _plans_from_index(index, repository, snapshot, candidates, contract, columns)


def plan_with_reader(repository, snapshot, candidates, contract):
    """`_plans` plus a run-batched ``read_rows(event, request)`` that shares the same record index."""
    _check_candidates(candidates)
    if ("event_id" not in candidates.columns or candidates.event_id.isna().any()
            or candidates.event_id.duplicated().any()):
        raise errors.fail("CONTRACT_MISMATCH", "event target candidates are malformed")
    index = _record_index(repository.fragment_records(snapshot, contract.table_name), contract)
    plans = _plans_from_index(index, repository, snapshot, candidates, contract, None)
    return plans, _run_reader(repository, contract, plans, index)


def _plans_from_index(index, repository, snapshot, candidates, contract, columns):
    projection = {} if columns is None else {"columns": columns}
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


def _runs(group, others):
    """Cut date-ordered plans of one ticker where another fragment may lie between neighbours.

    ``others`` holds the (lo, hi) day interval of every non-candidate fragment of the ticker, or
    None for an unreadable one; None cuts every candidate into its own run."""
    runs, current = [], []
    for plan in group:
        if current:
            lo, hi = current[-1][2].time_min, plan[2].time_min
            if any(o is None or (o[0] <= hi and o[1] >= lo) for o in others):
                runs.append(current)
                current = []
        current.append(plan)
    if current:
        runs.append(current)
    return runs


def _check_planned(planned, event, day, request):
    """The reader serves only the exact preflighted event, day and request."""
    if (planned is None or planned[0].ticker != event.ticker or planned[2].time_min != day
            or planned[1] != request):
        raise errors.fail("HOLDOUT_ACCESS_DENIED", "event or request was not preflighted for this reader")


def _run_reader(repository, contract, plans, index):
    """Run-batched candidate reads: one streaming scan per run, opened when first needed; refuses unless membership is exact."""
    by_ticker, run_of, ticker_records, rows_by_event, plan_of = {}, {}, {}, {}, {}
    for plan in plans:
        by_ticker.setdefault(plan[0].ticker, []).append(plan)
        plan_of[plan[0].event_id] = plan
    partitions, invalid = index
    for ticker, group in by_ticker.items():
        group.sort(key=lambda plan: plan[2].time_min)
        ids = {plan[2].fragment_id for plan in group}
        records = [record for _, record in partitions.get(ticker, (None, (), ()))[2]]
        ticker_records[ticker] = [*(record for _, record in invalid), *records]
        others = [_canonical_interval(record) for record in records if record.fragment_id not in ids]
        for run in _runs(group, others):
            for plan in run:
                run_of[plan[0].event_id] = run
    decision("history batched reads", f"{len(plans)} candidates in {len({id(r) for r in run_of.values()})} "
             f"runs over {len(by_ticker)} tickers")

    def open_run(run):
        """Prove the run's exact membership, then stream its rows one per batch."""
        ticker = run[0][0].ticker
        keys = (KeyPredicate(column="ticker", operator="eq", values=(ticker,)),)
        interval = TimeInterval(column="event_date", start_inclusive=run[0][2].time_min,
            end_exclusive=(date.fromisoformat(run[-1][2].time_max) + timedelta(days=1)).isoformat())
        expected = {plan[2].fragment_id for plan in run}
        population = query_planner.plan_scan_population(
            contract, ticker_records[ticker], key_filter=keys, time_interval=interval)
        if ({record.fragment_id for record in population.records} != expected
                or population.row_count != len(expected)):
            raise errors.fail("HOLDOUT_ACCESS_DENIED", "research outcome isolation cannot be proven")
        request = replace(run[0][1], key_filter=keys, time_interval=interval,
            max_batch_rows=1, max_result_rows=len(expected))
        scan = repository.scan(request, table_name=contract.table_name)
        return scan, (row for batch in scan for row in batch.to_pylist())

    streams = {}  # id(run) -> [row iterator, rows by day, last day seen, scan generator]

    def read_rows(event, request):
        """Advance the run's stream only up to this candidate's day; later fragments stay unopened."""
        planned = plan_of.get(getattr(event, "event_id", None))
        if planned is None:
            raise errors.fail("HOLDOUT_ACCESS_DENIED", "event or request was not preflighted for this reader")
        day = event.event_date.date().isoformat()
        _check_planned(planned, event, day, request)
        if event.event_id in rows_by_event:
            return rows_by_event[event.event_id]
        run = run_of[event.event_id]
        state = streams.get(id(run))
        if state is None:
            while len(streams) >= _MAX_OPEN_RUN_STREAMS:  # close the oldest; its tail re-scans if asked for again
                streams.pop(next(iter(streams)))[3].close()
            scan, rows = open_run(run)
            state = streams[id(run)] = [rows, {}, "", scan]
        rows, by_day, _, scan = state
        try:
            while rows is not None and day not in by_day and not (state[2] and day < state[2]):
                row = next(rows, None)
                if row is None:
                    break
                by_day.setdefault(row["event_date"], []).append(row)
                state[2] = row["event_date"]
            if rows is not None and (day == run[-1][2].time_min or state[2] >= run[-1][2].time_min):
                rows.close()  # last candidate of the run read: release the open object
                scan.close()
                state[0] = None
        except BaseException:
            streams.pop(id(run), None)  # a retry re-scans the run from its start
            scan.close()
            raise
        rows_by_event[event.event_id] = by_day.get(day, [])
        return rows_by_event[event.event_id]
    return read_rows


def skipped_reason(row):
    """The writer's reason for a skipped row; "skipped" for v4 or reason-less rows."""
    reason = row.get("not_computed_reason")
    return reason if isinstance(reason, str) and reason else "skipped"


def _historical_target(rows, event, decision):
    """Omit unavailable decoded labels before numeric validation or arithmetic."""
    if len(rows) != 1 or rows[0]["event_date"] != event.event_date.date().isoformat():
        raise errors.fail("EXPERIMENT_VARIANT_FAILED", "historical outcome identity is invalid")
    row = rows[0]
    if row["skipped"] is True:
        return None, skipped_reason(row)
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
    if contract not in (COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_V5_CONTRACT):
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
