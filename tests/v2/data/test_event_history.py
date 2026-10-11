"""Event-history primitives on the real path (design #651)."""
import math

import numpy as np
import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.event_history import (
    AsofSeries,
    SequencedEvent,
    event_sequence,
    strict_prior_mean,
)

DAYS = ["2020-01-10", "2020-04-10", "2020-07-10", "2020-10-09", "2021-01-08", "2021-04-09"]


def _rows(*items):
    return [dict(date=d, src_iv=s, implied_move=v) for d, s, v in items]


def test_warmup_counts_every_event_including_null_move_ones():
    seq = event_sequence(DAYS, warmup=4)
    assert [e.ordinal for e in seq] == [0, 1, 2, 3, 4, 5]
    assert [e.admitted for e in seq] == [False] * 4 + [True] * 2


def test_warmup_is_declared_not_fixed():
    assert [e.admitted for e in event_sequence(DAYS, warmup=1)] == [False] + [True] * 5
    assert all(e.admitted for e in event_sequence(DAYS, warmup=0))


@pytest.mark.parametrize("days, kwargs", [
    (["2020-02-01", "2020-01-01"], {}),
    (["2020-01-01", "2020-01-01"], {}),
    (DAYS, {"first_known": "2019-01-01"}),
    ([], {"first_known": "2019-01-01"}),
])
def test_event_sequence_refuses_bad_lists(days, kwargs):
    with pytest.raises(DataError) as err:
        event_sequence(days, warmup=4, **kwargs)
    assert err.value.code == "CONTRACT_MISMATCH"


def test_event_sequence_refuses_negative_warmup():
    with pytest.raises(DataError):
        event_sequence(DAYS, warmup=-1)


def test_anchor_is_strictly_before_and_src_iv_only():
    rows = _rows(("2020-01-01", 0.5, 4.0), ("2020-01-08", 0.5, 5.0),
                 ("2020-01-09", None, 99.0), ("2020-01-10", 0.5, 7.0))
    series = AsofSeries(rows, eligible=("src_iv",), value="implied_move", on_missing="null")
    assert series.at("2020-01-10") == 5.0  # same-day row excluded; non-src_iv row ignored
    assert series.at("2020-01-11") == 7.0


def test_latest_src_iv_row_with_null_value_gives_null_not_older_quote():
    rows = _rows(("2020-01-01", 0.5, 4.0), ("2020-01-08", 0.5, None))
    series = AsofSeries(rows, eligible=("src_iv",), value="implied_move", on_missing="null")
    assert series.at("2020-01-10") is None
    refusing = AsofSeries(rows, eligible=("src_iv",), value="implied_move", on_missing="refuse")
    with pytest.raises(DataError) as err:
        refusing.at("2020-01-10")
    assert err.value.code == "CONTRACT_MISMATCH"


def test_nan_value_is_a_miss():
    rows = _rows(("2020-01-01", 0.5, math.nan))
    series = AsofSeries(rows, eligible=("src_iv",), value="implied_move", on_missing="null")
    assert series.at("2020-02-01") is None


def test_no_prior_row_follows_on_missing():
    rows = _rows(("2020-01-10", 0.5, 4.0))
    assert AsofSeries(rows, eligible=("src_iv",), value="implied_move",
                      on_missing="null").at("2020-01-10") is None
    with pytest.raises(DataError):
        AsofSeries(rows, eligible=("src_iv",), value="implied_move",
                   on_missing="refuse").at("2020-01-10")


def test_max_staleness_makes_an_old_anchor_a_miss():
    rows = _rows(("2020-01-01", 0.5, 4.0))
    series = AsofSeries(rows, eligible=("src_iv",), value="implied_move",
                        on_missing="null", max_staleness_days=5)
    assert series.at("2020-01-04") == 4.0
    assert series.at("2020-02-01") is None


def test_asof_series_refuses_bad_declarations_and_rows():
    good = _rows(("2020-01-01", 0.5, 4.0))
    with pytest.raises(DataError):
        AsofSeries(good, eligible=("src_iv",), value="implied_move", on_missing="default")
    with pytest.raises(DataError):
        AsofSeries(good, eligible=("missing",), value="implied_move", on_missing="null")
    with pytest.raises(DataError):
        AsofSeries(_rows(("2020-01-02", 0.5, 1.0), ("2020-01-01", 0.5, 1.0)),
                   eligible=("src_iv",), value="implied_move", on_missing="null")


def test_provenance_columns_never_change_the_answer():
    base = _rows(("2020-01-01", 0.5, 4.0), ("2020-01-08", 0.5, 5.0))
    noisy = [dict(r, retrieved_at=f"2030-0{i + 1}-01", computed_at="1999-01-01")
             for i, r in enumerate(base)]
    a = AsofSeries(base, eligible=("src_iv",), value="implied_move", on_missing="null")
    b = AsofSeries(noisy, eligible=("src_iv",), value="implied_move", on_missing="null")
    assert a.at("2020-01-10") == b.at("2020-01-10") == 5.0


def test_strict_prior_mean_excludes_current_and_warmup_events():
    seq = event_sequence(DAYS, warmup=4)
    values = [10.0, 20.0, 30.0, 40.0, 2.0, 4.0]
    out = strict_prior_mean(seq, values, population="admitted")
    # warm-up events neither get a value nor contribute; event 4 has no admitted prior
    assert out == ((None, 0),) * 5 + ((2.0, 1),)
    allpop = strict_prior_mean(seq, values, population="all")
    assert allpop[0] == (None, 0)
    assert allpop[4] == (25.0, 4)  # strictly prior: its own 2.0 is excluded
    assert allpop[5] == (pytest.approx(20.4), 5)


def test_strict_prior_mean_skips_null_contributions():
    seq = event_sequence(DAYS, warmup=0)
    out = strict_prior_mean(seq, [1.0, None, math.nan, 3.0, 5.0, 7.0], population="all")
    assert out[3] == (1.0, 1)
    assert out[4] == (2.0, 2)


def test_strict_prior_mean_refuses_bad_inputs():
    seq = event_sequence(DAYS, warmup=4)
    with pytest.raises(DataError):
        strict_prior_mean(seq, [1.0], population="all")
    with pytest.raises(DataError):
        strict_prior_mean(seq, [1.0] * 6, population="recent")
    with pytest.raises(DataError):
        strict_prior_mean(seq[1:], [1.0] * 5, population="all")


def test_eligible_nan_row_is_ineligible_not_an_anchor():
    rows = _rows(("2020-01-01", 0.5, 4.0), ("2020-01-08", math.nan, 9.0))
    series = AsofSeries(rows, eligible=("src_iv",), value="implied_move", on_missing="null")
    assert series.at("2020-01-10") == 4.0


def test_missing_value_column_refuses():
    with pytest.raises(DataError):
        AsofSeries([dict(date="2020-01-01", src_iv=0.5)], eligible=("src_iv",),
                   value="implied_move", on_missing="null")


def test_non_iso_dates_and_non_numeric_values_refuse():
    series = AsofSeries(_rows(("2020-01-01", 0.5, 4.0)), eligible=("src_iv",),
                        value="implied_move", on_missing="null")
    with pytest.raises(DataError):
        series.at("not-a-date")
    with pytest.raises(DataError):
        AsofSeries(_rows(("01/02/2020", 0.5, 4.0)), eligible=("src_iv",),
                   value="implied_move", on_missing="null")
    bad = AsofSeries(_rows(("2020-01-01", 0.5, "abc")), eligible=("src_iv",),
                     value="implied_move", on_missing="null")
    with pytest.raises(DataError):
        bad.at("2020-02-01")


def test_matching_first_known_is_accepted():
    assert len(event_sequence(DAYS, warmup=4, first_known=DAYS[0])) == 6


@pytest.mark.parametrize("bad", ["20200110", "2020-W02-5", "2020-1-10", "2020-01-10T00:00"])
def test_non_canonical_dates_refuse_everywhere(bad):
    series = AsofSeries(_rows(("2020-01-11", 0.5, 4.0)), eligible=("src_iv",),
                        value="implied_move", on_missing="null")
    with pytest.raises(DataError):
        series.at(bad)
    with pytest.raises(DataError):
        AsofSeries(_rows((bad, 0.5, 4.0)), eligible=("src_iv",),
                   value="implied_move", on_missing="null")
    with pytest.raises(DataError):
        event_sequence([bad], warmup=0)
    with pytest.raises(DataError):
        event_sequence(DAYS, warmup=0, first_known=bad)


@pytest.mark.parametrize("nan", [np.float32("nan"), np.float64("nan"), np.float16("nan")])
def test_numpy_nan_is_null_everywhere(nan):
    rows = _rows(("2020-01-01", 0.5, 4.0), ("2020-01-08", nan, 9.0))
    series = AsofSeries(rows, eligible=("src_iv",), value="implied_move", on_missing="null")
    assert series.at("2020-01-10") == 4.0
    nan_value = AsofSeries(_rows(("2020-01-01", 0.5, nan)), eligible=("src_iv",),
                           value="implied_move", on_missing="null")
    assert nan_value.at("2020-02-01") is None
    out = strict_prior_mean(event_sequence(DAYS, warmup=0), [1.0, nan, 3.0, 5.0, 7.0, 9.0],
                            population="all")
    assert out[2] == (1.0, 1)


def _series(**kwargs):
    base = dict(eligible=("src_iv",), value="implied_move", on_missing="null")
    return AsofSeries(_rows(("2020-01-01", 0.5, 4.0)), **{**base, **kwargs})


@pytest.mark.parametrize("warmup", [1.5, math.nan, True, "4", None, -1])
def test_bad_warmup_refuses_with_code(warmup):
    with pytest.raises(DataError) as err:
        event_sequence(DAYS, warmup=warmup)
    assert err.value.code == "CONTRACT_MISMATCH"


def test_numpy_integer_warmup_is_accepted():
    assert event_sequence(DAYS, warmup=np.int64(4))[4].admitted


@pytest.mark.parametrize("stale", [1.5, math.nan, True, "5", -1])
def test_bad_staleness_refuses_with_code(stale):
    with pytest.raises(DataError) as err:
        _series(max_staleness_days=stale)
    assert err.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("kwargs", [
    {"eligible": "src_iv"}, {"eligible": None}, {"eligible": (1,)}, {"value": None},
    {"on_missing": None},
])
def test_bad_declarations_refuse_with_code(kwargs):
    with pytest.raises(DataError) as err:
        _series(**kwargs)
    assert err.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("rows", [None, 5, "abc", {"date": "2020-01-01"}, [1], [None]])
def test_bad_rows_refuse_with_code(rows):
    with pytest.raises(DataError) as err:
        AsofSeries(rows, eligible=("src_iv",), value="implied_move", on_missing="null")
    assert err.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("bad", [None, 20200110, b"2020-01-10"])
def test_bad_lookup_date_type_refuses_with_code(bad):
    with pytest.raises(DataError) as err:
        _series().at(bad)
    assert err.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("bad", ["abc", True, math.inf, 10**400, [1]])
def test_non_numeric_anchor_value_refuses_with_code(bad):
    series = AsofSeries(_rows(("2020-01-01", 0.5, bad)), eligible=("src_iv",),
                        value="implied_move", on_missing="null")
    with pytest.raises(DataError) as err:
        series.at("2020-02-01")
    assert err.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("bad", ["abc", True, math.inf, 10**400, [1]])
def test_non_numeric_contribution_refuses_with_code(bad):
    seq = event_sequence(DAYS, warmup=0)
    with pytest.raises(DataError) as err:
        strict_prior_mean(seq, [bad, 1.0, 1.0, 1.0, 1.0, 1.0], population="all")
    assert err.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("dates", [None, 5, "2020-01-10", [None], [20200110]])
def test_bad_event_dates_refuse_with_code(dates):
    with pytest.raises(DataError) as err:
        event_sequence(dates, warmup=0)
    assert err.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("events, values", [
    (None, [1.0]), (5, [1.0]), ("ab", [1.0]), ([1, 2], [1.0, 2.0]),
    (event_sequence(DAYS, warmup=0), None), (event_sequence(DAYS, warmup=0), 5),
])
def test_bad_aggregate_inputs_refuse_with_code(events, values):
    with pytest.raises(DataError) as err:
        strict_prior_mean(events, values, population="all")
    assert err.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("bad", ["1.5", b"1", np.array([1.0]), np.bool_(True), True, [1.0],
                                 {"a": 1}, 1 + 2j, object()])
def test_numeric_inputs_are_an_allow_list(bad):
    series = AsofSeries(_rows(("2020-01-01", 0.5, bad)), eligible=("src_iv",),
                        value="implied_move", on_missing="null")
    with pytest.raises(DataError) as err:
        series.at("2020-02-01")
    assert err.value.code == "CONTRACT_MISMATCH"
    with pytest.raises(DataError) as err:
        strict_prior_mean(event_sequence(DAYS, warmup=0), [bad] + [1.0] * 5, population="all")
    assert err.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("good", [3, 3.0, np.int32(3), np.float32(3.0), np.float64(3.0)])
def test_numeric_allow_list_members_are_accepted(good):
    series = AsofSeries(_rows(("2020-01-01", 0.5, good)), eligible=("src_iv",),
                        value="implied_move", on_missing="null")
    assert series.at("2020-02-01") == 3.0


@pytest.mark.parametrize("kwargs", [
    {"event_date": "20200110", "ordinal": 0, "admitted": True},
    {"event_date": None, "ordinal": 0, "admitted": True},
    {"event_date": "2020-01-10", "ordinal": -1, "admitted": True},
    {"event_date": "2020-01-10", "ordinal": 1.5, "admitted": True},
    {"event_date": "2020-01-10", "ordinal": True, "admitted": True},
    {"event_date": "2020-01-10", "ordinal": 0, "admitted": 1},
    {"event_date": "2020-01-10", "ordinal": 0, "admitted": None},
])
def test_sequenced_event_validates_on_construction(kwargs):
    with pytest.raises(DataError) as err:
        SequencedEvent(**kwargs)
    assert err.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("policy", [None, 5, ["null"], b"null", "NULL", "default", {"null"}])
def test_policy_declarations_check_type_then_membership(policy):
    with pytest.raises(DataError) as err:
        _series(on_missing=policy)
    assert err.value.code == "CONTRACT_MISMATCH"
    with pytest.raises(DataError) as err:
        strict_prior_mean(event_sequence(DAYS, warmup=0), [1.0] * 6, population=policy)
    assert err.value.code == "CONTRACT_MISMATCH"


def test_mean_does_not_overflow_on_large_finite_values():
    seq = event_sequence(DAYS, warmup=0)
    out = strict_prior_mean(seq, [1e308, 1e308, 0.0, 1.0, 1.0, 1.0], population="all")
    assert out[2] == (pytest.approx(1e308), 2)
    assert all(m is None or math.isfinite(m) for m, _ in out)
    mixed = strict_prior_mean(seq, [1e308, -1e308, 0.0, 1.0, 1.0, 1.0], population="all")
    assert mixed[2] == (pytest.approx(0.0, abs=1e292), 2)


@pytest.mark.parametrize("population", ["admitted", "all"])
@pytest.mark.parametrize("dates", [
    ["2020-03-01", "2020-02-01", "2020-04-01"],
    ["2020-01-01", "2020-01-01", "2020-04-01"],
])
def test_hand_built_events_must_be_strictly_ascending(population, dates):
    events = [SequencedEvent(day, i, True) for i, day in enumerate(dates)]
    with pytest.raises(DataError) as err:
        strict_prior_mean(events, [1.0, 2.0, 3.0], population=population)
    assert err.value.code == "CONTRACT_MISMATCH"