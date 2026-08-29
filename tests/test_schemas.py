"""Tier-2 schema contracts: coercion, enforcement, and what must fail."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.data.schemas import (
    CONVENTIONS,
    SCHEMAS,
    SOURCE_PRIORITY,
    SchemaError,
    assert_schema,
    coerce,
    empty_frame,
)


def _minimal_chain_frame(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA"] * n,
            "obs_date": pd.to_datetime(["2024-05-01"] * n),
            "year": [2024] * n,
            "expiry": pd.to_datetime(["2024-05-17"] * n),
            "dte": [16] * n,
            "strike": [100.0 + i for i in range(n)],
            "right": (["C", "P"] * n)[:n],
            "bid": [1.0] * n,
            "ask": [1.2] * n,
        }
    )


class TestSchemaDefinitions:
    def test_all_five_tables_are_declared(self):
        assert set(SCHEMAS) == {
            "securities",
            "earnings_events",
            "daily_market",
            "option_chains",
            "trades",
        }

    @pytest.mark.parametrize("name", sorted(SCHEMAS))
    def test_every_table_has_a_primary_key_drawn_from_its_columns(self, name):
        schema = SCHEMAS[name]
        assert schema.primary_key
        assert set(schema.primary_key) <= set(schema.names)

    @pytest.mark.parametrize("name", sorted(SCHEMAS))
    def test_every_partitioned_table_carries_its_partition_column(self, name):
        schema = SCHEMAS[name]
        if schema.partition_by:
            assert schema.partition_by in schema.names

    @pytest.mark.parametrize("name", sorted(SCHEMAS))
    def test_empty_frame_satisfies_its_own_schema(self, name):
        assert_schema(empty_frame(name), name)

    def test_conventions_and_source_priority_are_documented(self):
        # These are decisions, not facts; the point of writing them down is that
        # every consumer makes the same one.
        assert "orats_mktcap_units" in CONVENTIONS
        assert "option_chains" in SOURCE_PRIORITY


class TestCoercion:
    def test_strings_are_cast_to_declared_types(self):
        raw = _minimal_chain_frame()
        raw["strike"] = raw["strike"].astype(str)
        raw["dte"] = raw["dte"].astype(str)
        out = coerce(raw, "option_chains")
        assert out["strike"].dtype == "float64"
        assert str(out["dte"].dtype) == "Int64"

    def test_missing_nullable_columns_are_filled_not_fabricated(self):
        out = coerce(_minimal_chain_frame(), "option_chains")
        assert "iv" in out.columns
        assert out["iv"].isna().all()

    def test_missing_required_columns_raise(self):
        raw = _minimal_chain_frame().drop(columns=["strike"])
        with pytest.raises(SchemaError, match="missing required columns"):
            coerce(raw, "option_chains")

    def test_columns_come_back_in_schema_order(self):
        shuffled = _minimal_chain_frame()[
            ["strike", "ticker", "right", "bid", "ask", "dte", "expiry", "obs_date", "year"]
        ]
        out = coerce(shuffled, "option_chains")
        assert list(out.columns) == list(SCHEMAS["option_chains"].names)

    def test_unknown_columns_are_dropped_by_default(self):
        raw = _minimal_chain_frame().assign(scratch=1)
        assert "scratch" not in coerce(raw, "option_chains").columns

    def test_unknown_columns_survive_when_explicitly_allowed(self):
        raw = _minimal_chain_frame().assign(scratch=1)
        assert "scratch" in coerce(raw, "option_chains", allow_extra=True).columns

    def test_integers_are_nullable_so_a_gap_never_becomes_zero(self):
        raw = _minimal_chain_frame()
        raw.loc[0, "dte"] = None
        out = coerce(raw, "option_chains")
        # Nullable Int64, not plain int64: a plain int cast would turn a missing
        # DTE into 0, which reads as "expires today" rather than "unknown".
        assert str(out["dte"].dtype) == "Int64"
        assert pd.isna(out.loc[0, "dte"])

    def test_datetimes_are_pinned_to_nanoseconds(self):
        out = coerce(_minimal_chain_frame(), "option_chains")
        assert str(out["obs_date"].dtype) == "datetime64[ns]"


class TestAssertSchema:
    def test_a_clean_frame_passes(self):
        assert_schema(coerce(_minimal_chain_frame(), "option_chains"), "option_chains")

    def test_wrong_dtype_is_caught(self):
        out = coerce(_minimal_chain_frame(), "option_chains")
        out["strike"] = out["strike"].astype(str)
        with pytest.raises(SchemaError, match="dtype"):
            assert_schema(out, "option_chains")

    def test_null_in_a_non_nullable_column_is_caught(self):
        out = coerce(_minimal_chain_frame(), "option_chains")
        out.loc[0, "ticker"] = None
        with pytest.raises(SchemaError, match="non-nullable"):
            assert_schema(out, "option_chains")

    def test_duplicate_primary_keys_are_caught(self):
        raw = _minimal_chain_frame()
        raw.loc[1, "strike"] = raw.loc[0, "strike"]
        raw.loc[1, "right"] = raw.loc[0, "right"]
        out = coerce(raw, "option_chains")
        with pytest.raises(SchemaError, match="duplicate row"):
            assert_schema(out, "option_chains")

    def test_key_check_can_be_deferred_for_streamed_batches(self):
        raw = _minimal_chain_frame()
        raw.loc[1, "strike"] = raw.loc[0, "strike"]
        raw.loc[1, "right"] = raw.loc[0, "right"]
        out = coerce(raw, "option_chains")
        assert_schema(out, "option_chains", check_keys=False)  # must not raise

    def test_unknown_table_is_rejected(self):
        with pytest.raises(SchemaError, match="unknown table"):
            assert_schema(pd.DataFrame(), "not_a_table")

    def test_missing_column_is_reported_by_name(self):
        out = coerce(_minimal_chain_frame(), "option_chains").drop(columns=["mid"])
        with pytest.raises(SchemaError, match=r"missing columns \['mid'\]"):
            assert_schema(out, "option_chains")
