"""Pinned positive-move targets with availability metadata, never model outputs."""
from __future__ import annotations

import math
from datetime import date, timedelta

from engine.v2.contracts.data import DataQuery, KeyPredicate, TimeInterval
from engine.v2.data import errors
from engine.v2.research.experiment_population import load_population


def _target(repository, snapshot, contract, event):
    day = event.event_date.date()
    version = snapshot.table_versions["computed_moves"]
    keys = (KeyPredicate(column="ticker", operator="eq", values=(event.ticker,)),)
    interval = TimeInterval(column="event_date", start_inclusive=day.isoformat(),
                            end_exclusive=(day + timedelta(days=1)).isoformat())
    bound = repository.scan_population_bound(snapshot.snapshot_id, table_name="computed_moves",
        table_contract_ref=version.table_contract_ref, key_filter=keys, time_interval=interval)
    query = DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=version.table_contract_ref,
        columns=("event_date", "realized_move_pct", "available_as_of_date", "skipped"), key_filter=keys,
        time_interval=interval, order_by=contract.primary_key,
        max_batch_rows=min(contract.maximum_batch_rows, max(1, min(bound, 2))),
        max_result_rows=min(bound, 2))
    rows = [row for batch in repository.scan(query, table_name="computed_moves")
            for row in batch.to_pylist()]
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
