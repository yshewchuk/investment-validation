"""Spec s4b Change 1: the new ``computed_moves`` table contract."""
from __future__ import annotations

from engine.v2.data.computed_moves_table import (
    COMPUTED_MOVES_CONTRACT,
    COMPUTED_MOVES_TABLE_NAME,
    _build,
)
from engine.v2.data.price_history_table import PRICE_HISTORY_CONTRACT


def test_contract_id_and_table_name():
    assert COMPUTED_MOVES_CONTRACT.contract_id == "computed_moves.v4"
    assert COMPUTED_MOVES_CONTRACT.semantic_version == "4.0.0"
    assert COMPUTED_MOVES_CONTRACT.table_name == "computed_moves"
    assert COMPUTED_MOVES_TABLE_NAME == "computed_moves"


def test_definition_hash_is_deterministic():
    first = COMPUTED_MOVES_CONTRACT.definition_hash
    second = _build().definition_hash
    assert first == second
    assert first.startswith("sha256:")
    assert len(first) == 71
    # distinct definition from the other natively-built contract
    assert first != PRICE_HISTORY_CONTRACT.definition_hash


def test_primary_key_columns_are_declared():
    """Negative control: this contract bypasses ``legacy_mapping``'s
    ``_check_declared_subset``, so a typo in ``primary_key``/
    ``partition_columns`` is caught by nothing else."""
    declared = {column.name for column in COMPUTED_MOVES_CONTRACT.columns}
    for name in (*COMPUTED_MOVES_CONTRACT.primary_key,
                 *COMPUTED_MOVES_CONTRACT.partition_columns):
        assert name in declared, name


def test_available_as_of_date_column_is_nullable_and_documented():
    columns = {column.name: column for column in COMPUTED_MOVES_CONTRACT.columns}
    column = columns["available_as_of_date"]
    assert column.nullable is True
    assert "not available to any decision" in column.null_policy


def test_v5_columns_match_the_store_fragment_schema_and_drop_quarter_ordinal():
    from engine.v2.data.computed_moves_table import COMPUTED_MOVES_V5_CONTRACT
    from engine.v2.ops.computed_moves_store import _ARROW_SCHEMA, FRAGMENT_COLUMNS

    names = tuple(column.name for column in COMPUTED_MOVES_V5_CONTRACT.columns)
    assert names == FRAGMENT_COLUMNS == tuple(_ARROW_SCHEMA.names)
    assert "quarter_ordinal" not in names
    columns = {column.name: column for column in COMPUTED_MOVES_V5_CONTRACT.columns}
    assert columns["mean_prior_or_implied"].nullable is True
    for name in ("event_ordinal", "admitted", "n_prior_implied"):
        assert columns[name].nullable is False
