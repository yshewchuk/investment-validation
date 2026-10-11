"""Event-history primitives on the real path (design #651).

Every refusal asserts the exact code, message and details; every computed output
is compared to a complete literal expected value.
"""
import contextlib
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
ISO = "dates must be ISO YYYY-MM-DD strings"
CANON = "dates must be canonical YYYY-MM-DD strings"
DATE_MSG = (ISO, CANON)  # which one fires for compact/week forms depends on the Python version
NUMERIC = "a numeric value is required"
FINITE = "a finite numeric value is required"
ON_MISSING_MSG = "on_missing must be one of ('null', 'refuse')"
POPULATION_MSG = "population must be one of ('admitted', 'all')"
WARMUP_MSG = "warmup must be a non-negative integer"
ASCENDING_MSG = "event dates must be strictly ascending"
FIRST_MSG = "event list does not start at the ticker's first known event"
NO_OBS_MSG = "no usable observation before the event"


@contextlib.contextmanager
def _mismatch(message, **details):
    """A refusal is exactly CONTRACT_MISMATCH with the given message and details."""
    with pytest.raises(DataError) as err:
        yield
    problem = err.value.problem
    assert problem.code == "CONTRACT_MISMATCH"
    assert problem.details == details
    assert problem.message in (message if isinstance(message, tuple) else (message,))


def _rows(*items):
    return [dict(date=d, src_iv=s, implied_move=v) for d, s, v in items]


def _asof(rows, **kwargs):
    base = dict(eligible=("src_iv",), value="implied_move", on_missing="null")
    return AsofSeries(rows, **{**base, **kwargs})


def _series(**kwargs):
    return _asof(_rows(("2020-01-01", 0.5, 4.0)), **kwargs)


def _expected(admitted):
    """Plain (event_date, ordinal, admitted) literals built from the DAYS literal."""
    return [(day, i, flag) for i, (day, flag) in enumerate(zip(DAYS, admitted))]


def _fields(events):
    return [(e.event_date, e.ordinal, e.admitted) for e in events]


def _assert_means(out, expected):
    """Counts exact; means to a 1e-9 relative tolerance with NO absolute slack."""
    assert len(out) == len(expected)
    for (mean, n), (exp_mean, exp_n) in zip(out, expected):
        assert n == exp_n
        if exp_mean is None:
            assert mean is None
        else:
            assert mean == pytest.approx(exp_mean, rel=1e-9, abs=0)


# ---- event_sequence ------------------------------------------------------------------

def test_event_sequence_full_expected_tuple_with_warmup_4():
    assert _fields(event_sequence(DAYS, warmup=4)) == [
        ("2020-01-10", 0, False), ("2020-04-10", 1, False),
        ("2020-07-10", 2, False), ("2020-10-09", 3, False),
        ("2021-01-08", 4, True), ("2021-04-09", 5, True),
    ]


@pytest.mark.parametrize("warmup, admitted", [
    (0, [True] * 6),
    (1, [False] + [True] * 5),
    (2, [False] * 2 + [True] * 4),
    (4, [False] * 4 + [True] * 2),
    (6, [False] * 6),
    (np.int64(4), [False] * 4 + [True] * 2),
    (np.int32(2), [False] * 2 + [True] * 4),
    (np.uint8(0), [True] * 6),
])
def test_warmup_is_declared_and_admission_vector_is_complete(warmup, admitted):
    assert _fields(event_sequence(DAYS, warmup=warmup)) == _expected(admitted)


def test_event_sequence_empty_and_tuple_inputs():
    assert _fields(event_sequence([], warmup=0)) == []
    assert _fields(event_sequence(tuple(DAYS), warmup=4, first_known="2020-01-10")) == [
        ("2020-01-10", 0, False), ("2020-04-10", 1, False),
        ("2020-07-10", 2, False), ("2020-10-09", 3, False),
        ("2021-01-08", 4, True), ("2021-04-09", 5, True),
    ]


@pytest.mark.parametrize("warmup", [1.5, 0.5, math.nan, True, False, "4", None, -1, -5])
def test_bad_warmup_refuses(warmup):
    with _mismatch(WARMUP_MSG):
        event_sequence(DAYS, warmup=warmup)


@pytest.mark.parametrize("days, message", [
    (["2020-02-01", "2020-01-01"], ASCENDING_MSG),
    (["2020-01-01", "2020-01-01"], ASCENDING_MSG),
])
def test_event_sequence_refuses_unsorted_or_duplicate(days, message):
    with _mismatch(message):
        event_sequence(days, warmup=4)


@pytest.mark.parametrize("days, first_known", [(DAYS, "2019-01-01"), ([], "2019-01-01"),
                                               (DAYS, "2020-01-11")])
def test_event_sequence_refuses_a_list_not_starting_at_first_known(days, first_known):
    with _mismatch(FIRST_MSG):
        event_sequence(days, warmup=4, first_known=first_known)


@pytest.mark.parametrize("dates", [None, 5, "2020-01-10", b"2020-01-10", range(3)])
def test_event_dates_container_must_be_list_or_tuple(dates):
    with _mismatch("event_dates must be a list or tuple"):
        event_sequence(dates, warmup=0)


@pytest.mark.parametrize("dates", [[None], [20200110], [b"2020-01-10"]])
def test_event_dates_items_must_be_strings(dates):
    with _mismatch(ISO):
        event_sequence(dates, warmup=0)


@pytest.mark.parametrize("bad", ["20200110", "2020-W02-5", "2020-1-10", "2020-01-10T00:00"])
def test_non_canonical_dates_refuse_in_every_date_input(bad):
    series = _series()
    with _mismatch(DATE_MSG):
        series.at(bad)
    with _mismatch(DATE_MSG):
        _asof(_rows((bad, 0.5, 4.0)))
    with _mismatch(DATE_MSG):
        event_sequence([bad], warmup=0)
    with _mismatch(DATE_MSG):
        event_sequence(DAYS, warmup=0, first_known=bad)


# ---- SequencedEvent ------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, message", [
    ({"event_date": "20200110", "ordinal": 0, "admitted": True}, DATE_MSG),
    ({"event_date": "2020-1-10", "ordinal": 0, "admitted": True}, DATE_MSG),
    ({"event_date": None, "ordinal": 0, "admitted": True}, ISO),
    ({"event_date": "2020-01-10", "ordinal": -1, "admitted": True},
     "ordinal must be a non-negative integer"),
    ({"event_date": "2020-01-10", "ordinal": 1.5, "admitted": True},
     "ordinal must be a non-negative integer"),
    ({"event_date": "2020-01-10", "ordinal": True, "admitted": True},
     "ordinal must be a non-negative integer"),
    ({"event_date": "2020-01-10", "ordinal": 0, "admitted": 1}, "admitted must be a bool"),
    ({"event_date": "2020-01-10", "ordinal": 0, "admitted": None}, "admitted must be a bool"),
])
def test_sequenced_event_validates_on_construction(kwargs, message):
    with _mismatch(message):
        SequencedEvent(**kwargs)


# ---- AsofSeries: lookup --------------------------------------------------------------

def test_anchor_is_strictly_before_and_src_iv_only():
    rows = _rows(("2020-01-01", 0.5, 4.0), ("2020-01-08", 0.5, 5.0),
                 ("2020-01-09", None, 99.0), ("2020-01-10", 0.5, 7.0))
    series = _asof(rows)
    assert series.at("2020-01-10") == 5.0  # same-day row excluded; non-src_iv row ignored
    assert series.at("2020-01-11") == 7.0
    assert series.at("2020-01-09") == 5.0
    assert series.at("2020-01-02") == 4.0


def test_latest_src_iv_row_with_null_value_gives_null_not_older_quote():
    rows = _rows(("2020-01-01", 0.5, 4.0), ("2020-01-08", 0.5, None))
    assert _asof(rows).at("2020-01-10") is None
    with _mismatch(NO_OBS_MSG):
        _asof(rows, on_missing="refuse").at("2020-01-10")


def test_nan_value_is_a_miss():
    assert _asof(_rows(("2020-01-01", 0.5, math.nan))).at("2020-02-01") is None


def test_no_prior_row_follows_on_missing():
    rows = _rows(("2020-01-10", 0.5, 4.0))
    assert _asof(rows).at("2020-01-10") is None
    assert _asof(rows).at("2020-01-05") is None
    with _mismatch(NO_OBS_MSG):
        _asof(rows, on_missing="refuse").at("2020-01-10")
    with _mismatch(NO_OBS_MSG):
        _asof([], on_missing="refuse").at("2020-01-10")


@pytest.mark.parametrize("max_age, expected", [(9, None), (10, 4.0), (11, 4.0)])
def test_staleness_boundary_is_inclusive_at_exact_age(max_age, expected):
    # anchor 2020-01-01, event 2020-01-11: age is exactly 10 days
    assert _series(max_staleness_days=max_age).at("2020-01-11") == expected


def test_max_staleness_makes_an_old_anchor_a_miss():
    series = _series(max_staleness_days=5)
    assert series.at("2020-01-04") == 4.0
    assert series.at("2020-02-01") is None


def test_stale_anchor_refuses_when_declared_and_zero_staleness_is_a_miss():
    with _mismatch(NO_OBS_MSG):
        _series(max_staleness_days=1, on_missing="refuse").at("2020-01-11")
    with _mismatch(NO_OBS_MSG):
        _series(max_staleness_days=0, on_missing="refuse").at("2020-01-02")
    assert _series(max_staleness_days=0).at("2020-01-02") is None
    assert _series(max_staleness_days=1).at("2020-01-02") == 4.0


def test_eligible_nan_row_is_ineligible_not_an_anchor():
    rows = _rows(("2020-01-01", 0.5, 4.0), ("2020-01-08", math.nan, 9.0))
    assert _asof(rows).at("2020-01-10") == 4.0


def test_every_eligible_field_must_be_present():
    rows = [dict(date="2020-01-01", a=1.0, b=1.0, v=4.0),
            dict(date="2020-01-08", a=1.0, b=None, v=9.0),
            dict(date="2020-01-09", a=None, b=1.0, v=11.0)]
    series = AsofSeries(rows, eligible=("a", "b"), value="v", on_missing="null")
    assert series.at("2020-01-10") == 4.0
    assert series.at("2020-01-09") == 4.0


def test_null_values_on_any_row_are_still_accepted():
    rows = _rows(("2020-01-01", None, None), ("2020-01-05", 0.5, math.nan),
                 ("2020-01-06", 0.5, 4.0))
    series = _asof(rows)
    assert series.at("2020-01-10") == 4.0
    assert series.at("2020-01-06") is None
    assert series.at("2020-01-05") is None


def test_provenance_columns_never_change_the_answer():
    base = _rows(("2020-01-01", 0.5, 4.0), ("2020-01-08", 0.5, 5.0))
    noisy = [dict(r, retrieved_at=f"2030-0{i + 1}-01", computed_at="1999-01-01")
             for i, r in enumerate(base)]
    assert [_asof(base).at("2020-01-10"), _asof(noisy).at("2020-01-10")] == [5.0, 5.0]
    assert [_asof(base).at("2020-01-05"), _asof(noisy).at("2020-01-05")] == [4.0, 4.0]


@pytest.mark.parametrize("nan", [np.float32("nan"), np.float64("nan"), np.float16("nan")])
def test_numpy_nan_is_null_everywhere(nan):
    rows = _rows(("2020-01-01", 0.5, 4.0), ("2020-01-08", nan, 9.0))
    assert _asof(rows).at("2020-01-10") == 4.0
    assert _asof(_rows(("2020-01-01", 0.5, nan))).at("2020-02-01") is None
    out = strict_prior_mean(event_sequence(DAYS, warmup=0), [1.0, nan, 3.0, 5.0, 7.0, 9.0],
                            population="all")
    _assert_means(out, [(None, 0), (1.0, 1), (1.0, 1), (2.0, 2), (3.0, 3), (4.0, 4)])


GOOD_NUMBERS = [
    (3, 3.0), (2.5, 2.5), (-7, -7.0), (np.int32(-3), -3.0), (np.int64(5), 5.0),
    (np.float32(-2.5), -2.5), (np.float64(0.125), 0.125), (np.float16(1.5), 1.5),
    (np.float32(3.0), 3.0),
]


@pytest.mark.parametrize("given, expected", GOOD_NUMBERS)
def test_numeric_allow_list_members_are_accepted_on_the_lookup_path(given, expected):
    assert _asof(_rows(("2020-01-01", 0.5, given))).at("2020-02-01") == expected


@pytest.mark.parametrize("given, e", GOOD_NUMBERS)
def test_numeric_allow_list_members_are_accepted_on_the_aggregate_path(given, e):
    out = strict_prior_mean(event_sequence(DAYS, warmup=0), [given, given, 0, 0.0, 0, 0.0],
                            population="all")
    _assert_means(out, [(None, 0), (e, 1), (e, 2), (e * 2 / 3, 3), (e / 2, 4), (e * 2 / 5, 5)])


# ---- AsofSeries: construction refusals -----------------------------------------------

@pytest.mark.parametrize("policy", [None, 5, ["null"], b"null", "NULL", "default", {"null"}])
def test_policy_declarations_check_type_then_membership(policy):
    with _mismatch(ON_MISSING_MSG):
        _series(on_missing=policy)
    with _mismatch(POPULATION_MSG):
        strict_prior_mean(event_sequence(DAYS, warmup=0), [1.0] * 6, population=policy)


@pytest.mark.parametrize("stale", [1.5, math.nan, True, "5", -1, -30])
def test_bad_staleness_refuses(stale):
    with _mismatch("max_staleness_days must be a non-negative integer"):
        _series(max_staleness_days=stale)


@pytest.mark.parametrize("eligible, message", [
    ("src_iv", "eligible must be a list or tuple"),
    (None, "eligible must be a list or tuple"),
    (range(2), "eligible must be a list or tuple"),
    ((1,), "eligible must be a sequence of column names"),
    (("src_iv", None), "eligible must be a sequence of column names"),
])
def test_bad_eligible_declaration_refuses(eligible, message):
    with _mismatch(message):
        _series(eligible=eligible)


@pytest.mark.parametrize("value", [None, 5, b"implied_move"])
def test_bad_value_declaration_refuses(value):
    with _mismatch("value must be a column name"):
        _series(value=value)


@pytest.mark.parametrize("rows, message", [
    (None, "rows must be a list or tuple"),
    (5, "rows must be a list or tuple"),
    ("abc", "rows must be a list or tuple"),
    ({"date": "2020-01-01"}, "rows must be a list or tuple"),
    ([1], "each row must be a mapping"),
    ([None], "each row must be a mapping"),
    ([["2020-01-01"]], "each row must be a mapping"),
])
def test_bad_rows_refuse(rows, message):
    with _mismatch(message):
        _asof(rows)


@pytest.mark.parametrize("rows", [
    _rows(("2020-01-02", 0.5, 1.0), ("2020-01-01", 0.5, 1.0)),
    _rows(("2020-01-01", 0.5, 1.0), ("2020-01-01", 0.5, 2.0)),
])
def test_rows_must_be_strictly_date_ascending(rows):
    with _mismatch("rows must be strictly date-ascending"):
        _asof(rows)


@pytest.mark.parametrize("missing", ["date", "src_iv", "implied_move"])
def test_missing_date_eligibility_or_value_column_refuses_with_its_name(missing):
    row = dict(date="2020-01-01", src_iv=0.5, implied_move=4.0)
    del row[missing]
    with _mismatch("rows lack a required column", column=missing):
        _asof([row])


def test_missing_second_eligibility_column_names_that_column():
    with _mismatch("rows lack a required column", column="ghost"):
        _asof(_rows(("2020-01-01", 0.5, 4.0)), eligible=("src_iv", "ghost"))


@pytest.mark.parametrize("bad", ["not-a-date", "01/02/2020", "2020-13-01"])
def test_bad_lookup_and_row_date_strings_refuse(bad):
    with _mismatch(ISO):
        _series().at(bad)
    with _mismatch(ISO):
        _asof(_rows((bad, 0.5, 4.0)))


@pytest.mark.parametrize("bad", [None, 20200110, b"2020-01-10"])
def test_bad_lookup_date_type_refuses(bad):
    with _mismatch(ISO):
        _series().at(bad)


NON_NUMERIC = [
    ("abc", NUMERIC), ("1.5", NUMERIC), (b"1", NUMERIC), (True, NUMERIC),
    (np.bool_(True), NUMERIC), ([1], NUMERIC), ([1.0], NUMERIC), ({"a": 1}, NUMERIC),
    (1 + 2j, NUMERIC), (np.array([1.0]), NUMERIC), (object(), NUMERIC),
    (math.inf, FINITE), (-math.inf, FINITE), (10**400, FINITE),
]


@pytest.mark.parametrize("bad, message", NON_NUMERIC)
def test_non_numeric_values_refuse_on_the_lookup_path(bad, message):
    with _mismatch(message):
        _asof(_rows(("2020-01-01", 0.5, bad)))


@pytest.mark.parametrize("bad, message", NON_NUMERIC)
def test_non_numeric_values_refuse_on_the_aggregate_path(bad, message):
    seq = event_sequence(DAYS, warmup=0)
    with _mismatch(message):
        strict_prior_mean(seq, [bad, 1.0, 1.0, 1.0, 1.0, 1.0], population="all")
    with _mismatch(message):
        strict_prior_mean(seq, [1.0, 1.0, 1.0, 1.0, 1.0, bad], population="all")


@pytest.mark.parametrize("bad, message", [("abc", NUMERIC), (math.inf, FINITE),
                                          ([1], NUMERIC), (True, NUMERIC)])
def test_malformed_value_is_refused_on_ineligible_and_non_selected_rows(bad, message):
    ineligible = _rows(("2020-01-01", 0.5, 4.0), ("2020-01-05", None, bad))
    non_selected = _rows(("2020-01-01", 0.5, bad), ("2020-01-05", 0.5, 4.0))
    for rows in (ineligible, non_selected):
        with _mismatch(message):
            _asof(rows)


# ---- strict_prior_mean ---------------------------------------------------------------

def test_strict_prior_mean_excludes_current_and_warmup_events():
    seq = event_sequence(DAYS, warmup=4)
    values = [10.0, 20.0, 30.0, 40.0, 2.0, 4.0]
    admitted = strict_prior_mean(seq, values, population="admitted")
    _assert_means(admitted, [(None, 0)] * 5 + [(2.0, 1)])
    everything = strict_prior_mean(seq, values, population="all")
    _assert_means(everything, [(None, 0), (10.0, 1), (15.0, 2), (20.0, 3), (25.0, 4),
                               (20.4, 5)])


def test_strict_prior_mean_skips_null_contributions():
    seq = event_sequence(DAYS, warmup=0)
    out = strict_prior_mean(seq, [1.0, None, math.nan, 3.0, 5.0, 7.0], population="all")
    _assert_means(out, [(None, 0), (1.0, 1), (1.0, 1), (1.0, 1), (2.0, 2), (3.0, 3)])


def test_admitted_population_means_over_several_admitted_events():
    seq = event_sequence(DAYS, warmup=2)
    out = strict_prior_mean(seq, [100.0, 200.0, 2.0, 4.0, 6.0, 8.0], population="admitted")
    _assert_means(out, [(None, 0), (None, 0), (None, 0), (2.0, 1), (3.0, 2), (4.0, 3)])


def test_mean_does_not_overflow_on_large_finite_values():
    seq = event_sequence(DAYS, warmup=0)
    big = strict_prior_mean(seq, [1e308, 1e308, 0.0, 1.0, 1.0, 1.0], population="all")
    _assert_means(big, [(None, 0), (1e308, 1), (1e308, 2), (6.666666666666667e307, 3),
                        (5e307, 4), (4e307, 5)])
    mixed = strict_prior_mean(seq, [1e308, -1e308, 0.0, 1.0, 1.0, 1.0], population="all")
    _assert_means(mixed, [(None, 0), (1e308, 1), (0.0, 2), (0.0, 3), (0.25, 4), (0.4, 5)])


def test_tuples_are_accepted_as_containers():
    seq = event_sequence(tuple(DAYS), warmup=0)
    out = strict_prior_mean(seq, tuple([1.0] * 6), population="all")
    _assert_means(out, [(None, 0), (1.0, 1), (1.0, 2), (1.0, 3), (1.0, 4), (1.0, 5)])
    series = AsofSeries(tuple(_rows(("2020-01-01", 0.5, 4.0))), eligible=("src_iv",),
                        value="implied_move", on_missing="null")
    assert series.at("2020-01-02") == 4.0


@pytest.mark.parametrize("values, message", [
    ([1.0], "events and values differ in length"),
    ([1.0] * 7, "events and values differ in length"),
])
def test_length_mismatch_refuses(values, message):
    with _mismatch(message):
        strict_prior_mean(event_sequence(DAYS, warmup=4), values, population="all")


@pytest.mark.parametrize("events", [event_sequence(DAYS, warmup=4)[1:],
                                    event_sequence(DAYS, warmup=4)[::-1]])
def test_events_must_be_the_full_ordinal_sequence(events):
    with _mismatch("events must be the full sequence in ordinal order"):
        strict_prior_mean(events, [1.0] * len(events), population="all")


@pytest.mark.parametrize("events, values, message", [
    (None, [1.0], "events must be a list or tuple"),
    (5, [1.0], "events must be a list or tuple"),
    ("ab", [1.0], "events must be a list or tuple"),
    ([1, 2], [1.0, 2.0], "events must be SequencedEvent items"),
    ([None], [1.0], "events must be SequencedEvent items"),
    (event_sequence(DAYS, warmup=0), None, "values must be a list or tuple"),
    (event_sequence(DAYS, warmup=0), 5, "values must be a list or tuple"),
])
def test_bad_aggregate_inputs_refuse(events, values, message):
    with _mismatch(message):
        strict_prior_mean(events, values, population="all")


@pytest.mark.parametrize("population", ["admitted", "all"])
@pytest.mark.parametrize("dates", [
    ["2020-03-01", "2020-02-01", "2020-04-01"],
    ["2020-01-01", "2020-01-01", "2020-04-01"],
])
def test_hand_built_events_must_be_strictly_ascending(population, dates):
    events = [SequencedEvent(day, i, True) for i, day in enumerate(dates)]
    with _mismatch("event dates must be strictly ascending in ordinal order"):
        strict_prior_mean(events, [1.0, 2.0, 3.0], population=population)


@pytest.mark.parametrize("bad, message", [("abc", NUMERIC), (math.inf, FINITE),
                                          ([1], NUMERIC), (True, NUMERIC)])
def test_warmup_event_values_are_validated_under_admitted(bad, message):
    seq = event_sequence(DAYS, warmup=4)
    with _mismatch(message):
        strict_prior_mean(seq, [bad, 1.0, 1.0, 1.0, 1.0, 1.0], population="admitted")


NOT_LIST_OR_TUPLE = [b"\x01\x03", "2020-01-10", range(3), np.array([1.0, 2.0]), {1.0, 2.0},
                     (x for x in [1.0])]


@pytest.mark.parametrize("bad", NOT_LIST_OR_TUPLE)
def test_every_container_input_allows_only_list_or_tuple(bad):
    seq = event_sequence(DAYS, warmup=0)
    with _mismatch("event_dates must be a list or tuple"):
        event_sequence(bad, warmup=0)
    with _mismatch("rows must be a list or tuple"):
        _asof(bad)
    with _mismatch("eligible must be a list or tuple"):
        _series(eligible=bad)
    with _mismatch("events must be a list or tuple"):
        strict_prior_mean(bad, [1.0] * 6, population="all")
    with _mismatch("values must be a list or tuple"):
        strict_prior_mean(seq, bad, population="all")


@pytest.mark.parametrize("call, message", [
    (lambda: event_sequence(DAYS), WARMUP_MSG),
    (lambda: AsofSeries(_rows(("2020-01-01", 0.5, 4.0)), eligible=("src_iv",),
                        value="implied_move"), ON_MISSING_MSG),
    (lambda: AsofSeries(_rows(("2020-01-01", 0.5, 4.0)), value="implied_move",
                        on_missing="null"), "eligible must be a list or tuple"),
    (lambda: AsofSeries(_rows(("2020-01-01", 0.5, 4.0)), eligible=("src_iv",),
                        on_missing="null"), "value must be a column name"),
    (lambda: strict_prior_mean(event_sequence(DAYS, warmup=0), [1.0] * 6), POPULATION_MSG),
])
def test_omitting_a_mandatory_declaration_refuses_with_no_silent_default(call, message):
    with _mismatch(message):
        call()