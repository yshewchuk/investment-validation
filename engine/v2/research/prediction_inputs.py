"""Pinned positive-move targets with availability metadata, never model outputs."""
from __future__ import annotations

import math
from datetime import date, timedelta

from engine.v2.contracts.data import DataQuery, KeyPredicate, TimeInterval
from engine.v2.data import errors
from engine.v2.research.experiment_population import load_population


def _target_filters(event):
    """Shared exact ticker/day selection for planning and query construction."""
    day = event.event_date.date()
    keys = (KeyPredicate(column="ticker", operator="eq", values=(event.ticker,)),)
    interval = TimeInterval(column="event_date", start_inclusive=day.isoformat(),
                            end_exclusive=(day + timedelta(days=1)).isoformat())
    return keys, interval


def _target_query(repository, snapshot, contract, event, *, population_bound=None,
                  columns=("event_date", "realized_move_pct", "available_as_of_date", "skipped")):
    """The shared exact ticker/day query; predicates alone do not isolate bytes."""
    version = snapshot.table_versions[contract.table_name]
    keys, interval = _target_filters(event)
    bound = population_bound
    if bound is None:
        bound = repository.scan_population_bound(snapshot.snapshot_id, table_name=contract.table_name,
            table_contract_ref=version.table_contract_ref, key_filter=keys, time_interval=interval)
    return DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=version.table_contract_ref,
        columns=columns, key_filter=keys,
        time_interval=interval, order_by=contract.primary_key,
        max_batch_rows=min(contract.maximum_batch_rows, max(1, min(bound, 2))),
        max_result_rows=min(bound, 2))


def _parse_target(rows, day):
    """Validate one admitted outcome and its recorded availability."""
    try:
        if (len(rows) != 1 or rows[0]["skipped"] is not False
                or rows[0]["event_date"] != day.isoformat()):
            raise ValueError
        value = rows[0]["realized_move_pct"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError
        availability = rows[0]["available_as_of_date"]
        available_on = date.fromisoformat(availability)
        if availability != available_on.isoformat() or available_on <= day:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise errors.fail("EXPERIMENT_VARIANT_FAILED", "prediction target or availability is invalid") from None
    return value, int(value > 0), available_on.isoformat()


def _target(repository, snapshot, contract, event):
    query = _target_query(repository, snapshot, contract, event)
    rows = [row for batch in repository.scan(query, table_name="computed_moves")
            for row in batch.to_pylist()]
    return _parse_target(rows, event.event_date.date())


def load_prediction_targets(repository, snapshot, *, as_of_month, purpose="selection",
                            event_ids=None):
    """Complete eligible labels; fold consumers must enforce target availability."""
    rows = load_population(repository, snapshot, as_of_month=as_of_month,
                           purpose=purpose, event_ids=event_ids)
    contract = repository.table_contract(snapshot, "computed_moves")
    targets = [_target(repository, snapshot, contract, event) for event in rows.itertuples(index=False)]
    for index, name in enumerate(("realized_move_pct", "positive_move", "target_available_on")):
        rows[name] = [target[index] for target in targets]
    return rows
