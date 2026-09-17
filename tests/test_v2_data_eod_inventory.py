import pytest

from engine.v2.data.eod_inventory import (
    CURRENT_EOD_INVENTORY,
    inventory_by_name,
    inventory_document,
    validate_inventory,
)
from engine.v2.data.errors import DataError


def test_current_eod_inventory_is_complete_and_declares_read_contracts():
    names = {item.table_name for item in CURRENT_EOD_INVENTORY}
    assert names == {
        "securities", "earnings_events", "daily_market", "option_chains",
        "option_daily", "trades", "feature_panel", "tier4_forecasts",
    }
    assert all(item.primary_key and item.partition_columns for item in CURRENT_EOD_INVENTORY)
    assert all(item.coverage_denominator and item.dependencies for item in CURRENT_EOD_INVENTORY)
    assert set(inventory_by_name()) == names
    assert len(inventory_document()) == len(names)


def test_inventory_rejects_missing_and_duplicate_dataset():
    with pytest.raises(DataError) as missing:
        validate_inventory(CURRENT_EOD_INVENTORY[:-1])
    assert missing.value.code == "UNSUPPORTED_CONTRACT"
    with pytest.raises(DataError) as duplicate:
        validate_inventory(CURRENT_EOD_INVENTORY + (CURRENT_EOD_INVENTORY[0],))
    assert duplicate.value.code == "IDENTITY_CONFLICT"
