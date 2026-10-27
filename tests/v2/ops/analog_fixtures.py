"""Synthetic exact-key board analog pools for native score assembly tests."""
from dataclasses import replace
from datetime import date

from engine.v2.models.analog_artifact import (
    board_analog_pool_key,
    make_board_analog_pool_artifact,
)
from engine.v2.models.lineage import Lineage


def with_synthetic_exact_analogs(binding, events, *, as_of):
    """Add exact pools only to fixture releases with no analog family at all.

    Releases with any explicitly staged analog family are left untouched, so
    tests for wrong or missing exact keys continue to exercise refusals.
    """
    if binding.analog_artifacts:
        return binding
    analogs = {}
    for event in events:
        entry_date = event.calendar_row.get("entry_date")
        if not isinstance(entry_date, str):
            continue
        try:
            cutoff = min(date.fromisoformat(as_of), date.fromisoformat(entry_date)).isoformat()
        except ValueError:
            continue
        strategy = event.key.strategy
        key = board_analog_pool_key(strategy, 0.5, cutoff)
        pool = make_board_analog_pool_artifact(
            strategy=strategy, alpha=0.5, cutoff=cutoff,
            population_edges=(0.9, 1.1), causal_edges=(0.9, 1.1),
            lineage=Lineage(),
            rows=[("r1", "2025-01-01", "2025-02-01", "1-10B", "4-10", "ATM", "mid", 0.01)],
        )
        if pool.key != key:
            raise AssertionError("synthetic analog pool key did not match its request")
        analogs.setdefault(strategy, []).append(pool)
    return replace(binding, analog_artifacts={key: tuple(value) for key, value in analogs.items()})
