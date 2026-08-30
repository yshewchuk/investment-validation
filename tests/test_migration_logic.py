"""Negative controls for the migration test.

The migration test is the load-bearing check in Phase 0: it is what licenses the
claim that the new pipeline did not change a number the verdicts rest on. Its
design — declare known deltas, fail on anything else — has an obvious failure
mode: if a predicate is too permissive, or the comparison silently skips a
column, the test passes *vacuously* and proves nothing.

So these tests mostly assert that it FAILS when it should. A green migration
test only means something if red is reachable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from checks.phase0_migration import (
    COLUMN_ALIASES,
    KNOWN_DELTAS,
    TOLERANCES,
    compare,
    verify_mcap_delta,
)

LOG1000 = float(np.log(1000))


def legacy_frame(**overrides) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": pd.to_datetime(["2010-05-05", "2024-05-05"]),
            "move": [1.5, -2.5],
            "abs_move": [1.5, 2.5],
            "or_implied": [5.0, 6.0],
            "or_mcap_log": [np.log(3e6), np.log(4e10)],
        }
    )
    for key, value in overrides.items():
        frame[key] = value
    return frame


def new_frame(**overrides) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": pd.to_datetime(["2010-05-05", "2024-05-05"]),
            "move": [1.5, -2.5],
            "abs_move": [1.5, 2.5],
            "or_implied": [5.0, 6.0],
            # Billions-era row is corrected upward by exactly log(1000).
            "mcap_log": [np.log(3e6) + LOG1000, np.log(4e10)],
            "mcap_asof": pd.to_datetime(["2010-05-04", "2024-05-04"]),
        }
    )
    for key, value in overrides.items():
        frame[key] = value
    return frame


class TestItPassesWhenItShould:
    def test_identical_panels_reconcile(self):
        result = compare(legacy_frame(), new_frame())
        assert result.ok
        assert result.matched_rows == 2
        assert result.legacy_only == 0 and result.new_only == 0

    def test_the_mcap_correction_is_recognised(self):
        result = compare(legacy_frame(), new_frame())
        mcap = next(c for c in result.columns if c.column == "or_mcap_log")
        assert mcap.n_unexplained == 0
        assert "mcap-era-correction" in mcap.deltas

    def test_differences_within_tolerance_are_not_flagged(self):
        # Implied/IV fields carry a 0.1 tolerance.
        new = new_frame()
        new.loc[0, "or_implied"] = 5.05
        assert compare(legacy_frame(), new).ok


class TestItFailsWhenItShould:
    """The point of the whole design: an unexplained change must be caught."""

    def test_an_unexplained_numeric_change_fails(self):
        new = new_frame()
        new.loc[0, "move"] = 99.0
        result = compare(legacy_frame(), new)
        assert not result.ok
        bad = next(c for c in result.columns if c.column == "move")
        assert bad.n_unexplained == 1

    def test_a_change_just_outside_tolerance_fails(self):
        new = new_frame()
        new.loc[0, "or_implied"] = 5.0 + TOLERANCES["or_implied"] * 2
        assert not compare(legacy_frame(), new).ok

    def test_a_lost_row_fails(self):
        result = compare(legacy_frame(), new_frame().iloc[:1])
        assert not result.ok
        assert result.legacy_only == 1

    def test_an_invented_row_fails(self):
        extra = pd.concat([new_frame(), new_frame().iloc[:1].assign(ticker="CCC")])
        result = compare(legacy_frame(), extra)
        assert not result.ok
        assert result.new_only == 1

    def test_a_missing_column_fails(self):
        result = compare(legacy_frame(), new_frame().drop(columns=["move"]))
        assert not result.ok
        assert "move" in result.missing_columns

    def test_a_null_flip_with_no_declared_delta_fails(self):
        # `move` has no delta covering it, so turning a value into a null is a
        # regression, not an improvement.
        new = new_frame()
        new.loc[0, "move"] = np.nan
        result = compare(legacy_frame(), new)
        assert not result.ok
        assert next(c for c in result.columns if c.column == "move").n_unexplained == 1


class TestDeltasAreBoundedNotBlanket:
    """A declared delta must not become a blanket excuse for its column."""

    def test_the_mcap_delta_does_not_excuse_a_wrong_correction(self):
        # Right column, right rows — but the wrong magnitude. The equality
        # check is what catches this; the predicate alone would not.
        new = new_frame()
        new.loc[0, "mcap_log"] = np.log(3e6) + LOG1000 + 1.0
        check = verify_mcap_delta(legacy_frame(), new)
        assert not check["exactly_as_declared"]

    def test_the_mcap_delta_does_not_excuse_a_post_boundary_change(self):
        # The 2024 row's cap was observed after the boundary and must match.
        new = new_frame()
        new.loc[1, "mcap_log"] = np.log(4e10) + 0.5
        check = verify_mcap_delta(legacy_frame(), new)
        assert not check["exactly_as_declared"]
        assert check["after_delta_max_abs"] > 0.1

    def test_the_correct_correction_verifies_exactly(self):
        check = verify_mcap_delta(legacy_frame(), new_frame())
        assert check["exactly_as_declared"]
        assert check["before_delta_median"] == pytest.approx(LOG1000)
        assert check["after_delta_max_abs"] == pytest.approx(0.0, abs=1e-9)

    def test_the_era_is_keyed_on_the_observation_not_the_event(self):
        # Five real events land on 2017-06-28 and read a 2017-06-27 cap, which
        # is still a billions-era value. Keying on the event date would call
        # those unexplained regressions.
        legacy = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "date": pd.to_datetime(["2017-06-28"]),
                "or_mcap_log": [np.log(3e6)],
            }
        )
        new = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "date": pd.to_datetime(["2017-06-28"]),
                "mcap_log": [np.log(3e6) + LOG1000],
                "mcap_asof": pd.to_datetime(["2017-06-27"]),
            }
        )
        assert verify_mcap_delta(legacy, new)["exactly_as_declared"]


class TestRegistryIntegrity:
    def test_every_declared_delta_has_a_reason_and_a_predicate(self):
        for delta in KNOWN_DELTAS:
            assert delta.columns, f"{delta.name} declares no columns"
            assert len(delta.reason) > 80, f"{delta.name} needs a real explanation"
            assert callable(delta.predicate)

    def test_delta_names_are_unique(self):
        names = [d.name for d in KNOWN_DELTAS]
        assert len(set(names)) == len(names)

    def test_no_delta_silently_covers_a_price_or_move_column(self):
        # Deltas exist for ORATS-sourced state and market cap. If one ever
        # covered a realized move or an implied move, the migration test would
        # stop protecting the numbers the verdicts are actually made of.
        protected = {"move", "abs_move", "implied_move", "k", "n_prior", "year"}
        for delta in KNOWN_DELTAS:
            assert not (set(delta.columns) & protected), (
                f"{delta.name} would excuse changes in {set(delta.columns) & protected}"
            )

    def test_the_alias_maps_the_renamed_column(self):
        assert COLUMN_ALIASES["or_mcap_log"] == "mcap_log"
