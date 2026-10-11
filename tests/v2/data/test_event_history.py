"""Event-history primitives on the real path (design #651)."""
import math

import numpy as np
import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.event_history import AsofSeries, event_sequence, strict_prior_mean

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