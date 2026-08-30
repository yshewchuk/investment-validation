"""The leak auditor. If these pass and a leak still gets through, the auditor is
being bypassed, not failing — which is why the scorer calls it on every path.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.audit import (
    FeatureVector,
    LeakError,
    assert_causal,
    assert_decision_causal,
    audit_frame,
)


def make_vector(**kwargs) -> FeatureVector:
    base = dict(
        ticker="TEST",
        as_of=pd.Timestamp("2024-05-01"),
        values={"a": 1.0, "b": 2.0},
        feature_as_of={
            "a": pd.Timestamp("2024-04-30"),
            "b": pd.Timestamp("2024-05-01"),
        },
        event_date=pd.Timestamp("2024-05-02"),
    )
    base.update(kwargs)
    return FeatureVector(**base)


class TestFeatureVector:
    def test_normalizes_timestamps(self):
        vector = make_vector(as_of="2024-05-01 15:59:00")
        assert vector.as_of == pd.Timestamp("2024-05-01")

    def test_rejects_a_stamp_for_an_absent_feature(self):
        with pytest.raises(ValueError, match="carry no value"):
            make_vector(feature_as_of={"ghost": pd.Timestamp("2024-01-01")})

    def test_unstamped_feature_defaults_to_the_decision_date(self):
        vector = make_vector(
            values={"a": 1.0, "c": 3.0},
            feature_as_of={"a": pd.Timestamp("2024-04-30")},
        )
        assert vector.stamp("c") == vector.as_of

    def test_assert_complete_names_unstamped_features(self):
        vector = make_vector(
            values={"a": 1.0, "c": 3.0},
            feature_as_of={"a": pd.Timestamp("2024-04-30")},
        )
        with pytest.raises(LeakError, match="c"):
            vector.assert_complete(["a", "c"])

    def test_vector_orders_by_requested_names(self):
        vector = make_vector()
        assert vector.vector(["b", "a"]).tolist() == [[2.0, 1.0]]

    def test_vector_raises_on_a_missing_name(self):
        with pytest.raises(KeyError, match="missing"):
            make_vector().vector(["a", "nope"])

    def test_missing_reports_nan_as_missing(self):
        vector = make_vector(values={"a": 1.0, "b": float("nan")})
        assert vector.missing(["a", "b"]) == ["b"]

    def test_missing_reports_an_absent_name(self):
        assert make_vector().missing(["a", "absent"]) == ["absent"]

    def test_is_immutable(self):
        vector = make_vector()
        with pytest.raises(Exception):
            vector.as_of = pd.Timestamp("2030-01-01")


class TestAssertCausal:
    def test_passes_when_every_stamp_precedes_the_decision(self):
        assert_causal(make_vector())

    def test_equality_is_allowed(self):
        """A feature read at the close we trade at is known to us at that close."""
        vector = make_vector(
            feature_as_of={"a": pd.Timestamp("2024-05-01"), "b": pd.Timestamp("2024-05-01")}
        )
        assert_causal(vector)

    def test_the_leak_poison_test(self):
        """Guide acceptance test 3: shift one stamp past as_of → must RAISE."""
        poisoned = make_vector().with_stamps(a=pd.Timestamp("2024-05-02"))
        with pytest.raises(LeakError, match="observed after"):
            assert_causal(poisoned)

    def test_one_day_is_enough_to_fail(self):
        poisoned = make_vector().with_stamps(b=pd.Timestamp("2024-05-02"))
        with pytest.raises(LeakError):
            assert_causal(poisoned)

    def test_names_every_offending_feature(self):
        poisoned = make_vector().with_stamps(
            a=pd.Timestamp("2024-06-01"), b=pd.Timestamp("2024-07-01")
        )
        with pytest.raises(LeakError) as exc:
            assert_causal(poisoned)
        assert "a @ 2024-06-01" in str(exc.value)
        assert "b @ 2024-07-01" in str(exc.value)

    def test_explicit_cutoff_catches_a_reused_vector(self):
        """A vector built for a later decision must fail against an earlier one."""
        vector = make_vector()
        with pytest.raises(LeakError):
            assert_causal(vector, as_of=pd.Timestamp("2024-04-29"))

    def test_rejects_a_non_vector(self):
        with pytest.raises(TypeError):
            assert_causal({"a": 1.0})


class TestDecisionCausality:
    """A vector can be internally consistent and still be assembled too late."""

    class FakeCalendar:
        def last_pre_print(self, event_date, session):
            event_date = pd.Timestamp(event_date)
            return event_date - pd.Timedelta(days=1) if session == "BMO" else event_date

    def test_bmo_forbids_deciding_on_the_event_date(self):
        with pytest.raises(LeakError, match="after the last information-free"):
            assert_decision_causal(
                "2024-05-02", "2024-05-02", "BMO", calendar=self.FakeCalendar()
            )

    def test_amc_allows_the_event_date_close(self):
        assert_decision_causal(
            "2024-05-02", "2024-05-02", "AMC", calendar=self.FakeCalendar()
        )

    def test_earlier_decisions_always_pass(self):
        assert_decision_causal(
            "2024-04-01", "2024-05-02", "BMO", calendar=self.FakeCalendar()
        )


class TestAuditFrame:
    def test_passes_a_clean_frame(self):
        frame = pd.DataFrame(
            {
                "as_of": pd.to_datetime(["2024-05-01", "2024-05-02"]),
                "f_stamp": pd.to_datetime(["2024-04-30", "2024-05-02"]),
            }
        )
        audit_frame(frame, "as_of", ["f_stamp"])

    def test_counts_offending_rows(self):
        frame = pd.DataFrame(
            {
                "as_of": pd.to_datetime(["2024-05-01", "2024-05-02"]),
                "f_stamp": pd.to_datetime(["2024-05-05", "2024-05-06"]),
            }
        )
        with pytest.raises(LeakError, match="2 row"):
            audit_frame(frame, "as_of", ["f_stamp"])

    def test_missing_stamp_column_is_an_error(self):
        frame = pd.DataFrame({"as_of": pd.to_datetime(["2024-05-01"])})
        with pytest.raises(KeyError):
            audit_frame(frame, "as_of", ["absent"])
