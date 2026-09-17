"""The fixed EOD input inventory for incremental refresh planning.

This module is deliberately pure.  It records the read contract that Phase 2
consumers already use so an incremental producer cannot quietly cover only the
first table it happens to implement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from engine.v2.data.errors import fail

__all__ = [
    "EODDataset",
    "CURRENT_EOD_INVENTORY",
    "inventory_by_name",
    "inventory_document",
    "validate_inventory",
]


@dataclass(frozen=True, kw_only=True)
class EODDataset:
    table_name: str
    source: str
    primary_key: tuple[str, ...]
    partition_columns: tuple[str, ...]
    units: str
    coverage_denominator: str
    revision_priority: tuple[str, ...]
    finality_semantics: str
    dependencies: tuple[str, ...]
    consumers: tuple[str, ...]
    read_scope: str


CURRENT_EOD_INVENTORY: tuple[EODDataset, ...] = (
    EODDataset(
        table_name="securities", source="ORATS/reference", primary_key=("ticker", "year"),
        partition_columns=("year",), units="declared table contract",
        coverage_denominator="security universe", revision_priority=("source", "finality", "ordinal"),
        finality_semantics="eod_final.v1", dependencies=("symbol identity",),
        consumers=("chain lookup", "scoring"), read_scope="whole table"),
    EODDataset(
        table_name="earnings_events", source="ORATS/Nasdaq/yfinance", primary_key=("event_id",),
        partition_columns=("year",), units="event session", coverage_denominator="event universe",
        revision_priority=("source", "finality", "ordinal"), finality_semantics="calendar_reconciled.v1",
        dependencies=("event identity", "trading calendar"),
        consumers=("scoring", "settlement", "features"), read_scope="whole table"),
    EODDataset(
        table_name="daily_market", source="ORATS", primary_key=("ticker", "date"),
        partition_columns=("year",), units="declared table contract", coverage_denominator="ticker-session",
        revision_priority=("source", "finality", "ordinal"), finality_semantics="eod_final.v1",
        dependencies=("rolling features", "finality"), consumers=("scoring", "features"),
        read_scope="whole table"),
    EODDataset(
        table_name="option_chains", source="ORATS", primary_key=("ticker", "obs_date", "expiry", "strike", "right"),
        partition_columns=("year",), units="declared quote units", coverage_denominator="expected contracts",
        revision_priority=("source", "finality", "ordinal"), finality_semantics="eod_final.v1",
        dependencies=("security identity", "event identity"), consumers=("trade construction", "settlement"),
        read_scope="ticker/year bounded"),
    EODDataset(
        table_name="option_daily", source="Polygon", primary_key=("contract_ticker", "obs_date"),
        partition_columns=("year",), units="USD traded prices", coverage_denominator="observed contracts",
        revision_priority=("source", "finality", "ordinal"), finality_semantics="provider_final.v1",
        dependencies=("option contract identity",), consumers=("fill quality", "settlement"),
        read_scope="trade window bounded"),
    EODDataset(
        table_name="trades", source="strategy ledger", primary_key=("trade_id",),
        partition_columns=("year",), units="declared trade contract", coverage_denominator="recorded trades",
        revision_priority=("source", "finality", "ordinal"), finality_semantics="append_only.v1",
        dependencies=("event identity", "chain identity"), consumers=("scoring", "settlement"),
        read_scope="whole table"),
    EODDataset(
        table_name="feature_panel", source="feature pipeline", primary_key=("ticker", "date"),
        partition_columns=("logical panel",), units="declared feature contract", coverage_denominator="feature universe",
        revision_priority=("source", "finality", "ordinal"), finality_semantics="derived_snapshot.v1",
        dependencies=("daily_market", "earnings_events", "option_chains"),
        consumers=("model scoring", "replay"), read_scope="whole file"),
    EODDataset(
        table_name="tier4_forecasts", source="forecast pipeline", primary_key=("ticker", "event_date"),
        partition_columns=("logical forecast",), units="declared forecast contract", coverage_denominator="forecast universe",
        revision_priority=("source", "finality", "ordinal"), finality_semantics="derived_snapshot.v1",
        dependencies=("feature_panel", "daily_market", "earnings_events"),
        consumers=("model scoring", "replay"), read_scope="whole file"),
)


def validate_inventory(inventory: tuple[EODDataset, ...] = CURRENT_EOD_INVENTORY) -> tuple[EODDataset, ...]:
    """Validate the complete fixed inventory before a refresh plan is admitted."""
    names = [item.table_name for item in inventory]
    expected = {item.table_name for item in CURRENT_EOD_INVENTORY}
    if len(names) != len(set(names)):
        raise fail("IDENTITY_CONFLICT", "EOD inventory contains duplicate table identities")
    if set(names) != expected:
        raise fail("UNSUPPORTED_CONTRACT", "EOD inventory is incomplete")
    for item in inventory:
        if (not item.source or not item.primary_key or not item.partition_columns
                or not item.units or not item.coverage_denominator
                or not item.revision_priority or not item.finality_semantics
                or not item.consumers):
            raise fail("UNSUPPORTED_CONTRACT", "EOD inventory entry is incomplete")
    return tuple(inventory)


def inventory_by_name(inventory: tuple[EODDataset, ...] = CURRENT_EOD_INVENTORY) -> Mapping[str, EODDataset]:
    validate_inventory(inventory)
    return {item.table_name: item for item in inventory}


def inventory_document(inventory: tuple[EODDataset, ...] = CURRENT_EOD_INVENTORY) -> tuple[dict, ...]:
    validate_inventory(inventory)
    return tuple({
        "table": item.table_name,
        "source": item.source,
        "key": ",".join(item.primary_key),
        "partition": ",".join(item.partition_columns),
        "units": item.units,
        "coverage_denominator": item.coverage_denominator,
        "revision_priority": ",".join(item.revision_priority),
        "finality": item.finality_semantics,
        "dependencies": list(item.dependencies),
        "consumers": list(item.consumers),
        "read_scope": item.read_scope,
    } for item in inventory)
