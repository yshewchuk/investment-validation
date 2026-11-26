"""Tests for EXP-190 Q4a2 baselines B1 and B2 (experiments/size_quantile_baselines.py). Synthetic fixtures only."""
import numpy as np
import pandas as pd
import pytest

from experiments import size_distribution_report as v14
from experiments.size_quantile_baselines import (
    MIN_IMPLIED_DONORS,
    band_quantiles,
    implied_quantiles,
)
from experiments.size_quantile_metrics import Rules

LEVELS = Rules().levels
QCOLS = [f"q{round(a * 100)}" for a in LEVELS]
EVENT = "2024-06-15"
B1_COLS = QCOLS + ["bucket", "n_rows", "reason"]
B2_COLS = QCOLS + ["n_rows", "n_dropped", "reason"]


def day_of(offset, start="2010-01-01"):
    return (pd.Timestamp(start) + pd.Timedelta(days=int(offset))).strftime("%Y-%m-%d")


def rows_of(pred, y, *, prefix="b", start="2010-01-01", avail=None):
    """(id, event_date, availability, model value, abs_move) rows; donor i is dated start + i days."""
    return [(f"{prefix}{i:04d}", day_of(i, start), avail or day_of(i, start), p, t)
            for i, (p, t) in enumerate(zip(pred, y))]


def b1_rows(n, *, base=0.5, ratio=None, prefix="b", start="2010-01-01", avail=None):
    i = np.arange(n)
    pred = base + 0.01 * i
    y = pred * (ratio if ratio is not None else 0.6 + 0.08 * ((i * 37) % 23))
    return rows_of(pred, y, prefix=prefix, start=start, avail=avail)


def b2_rows(n, *, ratio=None, prefix="d"):
    """Valid donors dated across the 12 months before 2024-06-01, each labelled on its own date."""
    rows = []
    for k in range(n):
        implied = 2.0 + 0.01 * k
        y = implied * (ratio if ratio is not None else 0.5 + 0.02 * (k % 30))
        day = day_of(k % 364, "2023-06-01")
        rows.append((f"{prefix}{k:04d}", day, day, implied, y))
    return rows


def frames(rows, model):
    ids, dates, avail, x, y = (list(col) for col in zip(*rows))
    table = pd.DataFrame({"event_date": pd.to_datetime(dates), model: x}, index=ids)
    labels = pd.DataFrame({"abs_move": y, "target_available_on": pd.to_datetime(avail)}, index=ids)
    return table, labels


def events(ids, dates, values, model):
    return pd.DataFrame({"event_date": pd.to_datetime(dates), model: list(values)}, index=ids)


def arrays(rows):
    """(model value, abs_move) arrays of the given rows, which must all be valid."""
    return np.array([r[3] for r in rows], dtype=float), np.array([r[4] for r in rows], dtype=float)


def band_oracle(pred, y, f):
    """Hand-rolled v14 band: edges on the pool pred, the forecast's bucket, the members' ratio quantiles."""
    edges = np.quantile(pred, [0.2, 0.4, 0.6, 0.8])
    member = np.digitize(pred, edges) == np.digitize(f, edges)
    ratio = y[member] / np.maximum(pred[member], v14.RATIO_FLOOR)
    return max(f, v14.RATIO_FLOOR) * np.quantile(ratio, LEVELS), int(member.sum())


def implied_oracle(implied, y, f):
    return f * np.quantile(y / implied, LEVELS)


def _dup(frame):
    frame = frame.copy()
    ids = list(frame.index)
    ids[1] = ids[0]
    frame.index = ids
    return frame


# ---------------------------------------------------------------- B1


@pytest.mark.parametrize("kind", ["below_floor", "mid", "edge", "far"])
def test_b1_matches_v14_band_metrics(monkeypatch, kind):
    rows = b1_rows(600, base=0.01)
    pred, y = arrays(rows)
    edges = np.quantile(pred, [0.2, 0.4, 0.6, 0.8])
    f = {"below_floor": 0.01, "mid": 1.3, "edge": edges[1], "far": 50.0}[kind]
    table, labels = frames(rows, "pred_move")
    got = band_quantiles(table, labels, events(["ev"], [EVENT], [f], "pred_move")).iloc[0]
    ids = np.array([r[0] for r in rows], dtype=object)
    pool = [{1: (pred, y, ids)}]
    scored = {1: (np.array([f]), np.array([1000.0]), np.array(["ev"], dtype=object))}

    def run(taus):
        monkeypatch.setattr(v14, "TAUS", taus)
        entry = v14.band_metrics(pool, scored, [1])["per_horizon"][0]
        assert entry["status"] == "ok"
        return entry

    # With y = 1000 above the band, the pinball loss at level t is t * (1000 - q_t).
    base = run((0.1, 0.5, 0.9))
    loss10 = base["pinball_mean"] * 3 - base["pinball_p50"] - base["pinball_p90"]
    want = {0.10: 1000.0 - loss10 / 0.1,
            0.50: 1000.0 - base["pinball_p50"] / 0.5,
            0.90: 1000.0 - base["pinball_p90"] / 0.9}
    for level in (0.25, 0.75):
        wide = run((0.1, 0.5, 0.9, level))
        loss = wide["pinball_mean"] * 4 - loss10 - wide["pinball_p50"] - wide["pinball_p90"]
        want[level] = 1000.0 - loss / level
    for level, value in want.items():
        np.testing.assert_allclose(got[f"q{round(level * 100)}"], value, rtol=1e-9)


@pytest.mark.parametrize("case", ["n99", "n100", "tied"])
def test_b1_pool_too_small_matches_v14_status(case):
    if case == "tied":
        pred = np.r_[0.01 * np.arange(1, 20), np.ones(581)]  # 600 donors, but ties leave buckets empty
    else:
        pred = 1.0 + 0.1 * np.arange(99 if case == "n99" else 100)
    y = pred * 1.2
    rows = rows_of(pred, y)
    table, labels = frames(rows, "pred_move")
    out = band_quantiles(table, labels, events(["ev"], [EVENT], [2.0], "pred_move")).iloc[0]
    ids = np.array([r[0] for r in rows], dtype=object)
    scored = {1: (np.array([2.0]), np.array([1000.0]), np.array(["ev"], dtype=object))}
    status = v14.band_metrics([{1: (pred, y, ids)}], scored, [1])["status"]
    if status == "pool_too_small":
        assert out.reason == "pool_too_small"
    else:
        assert status == "ok" and out.reason is None
    expected = {"n99": "pool_too_small", "n100": "ok", "tied": "pool_too_small"}[case]
    assert status == expected


def test_b1_month_cut_and_pool_shared_within_month():
    base = b1_rows(200, avail="2024-05-31")
    late = [("x1", "2010-01-01", "2024-06-01", 2.0, 1e6),
            ("x2", "2010-01-01", "2024-06-05", 2.0, 1e6),
            ("x3", "2010-01-01", None, 2.0, 1e6)]
    table, labels = frames(base + late, "pred_move")
    ev = events(["early", "late_in_month", "july"], ["2024-06-10", "2024-06-28", "2024-07-02"],
                [2.0, 2.0, 2.0], "pred_move")
    out = band_quantiles(table, labels, ev)
    want_june, n_june = band_oracle(*arrays(base), 2.0)
    for eid in ("early", "late_in_month"):
        assert out.loc[eid, "n_rows"] == n_june
        np.testing.assert_allclose(out.loc[eid, QCOLS].to_numpy(float), want_june)
    # From July, x1 and x2 (available in June) join the pool; x3 (null availability) never does.
    want_july, n_july = band_oracle(*arrays(base + late[:2]), 2.0)
    assert out.loc["july", "n_rows"] == n_july
    np.testing.assert_allclose(out.loc["july", QCOLS].to_numpy(float), want_july)


def test_b1_full_history_uses_both_eras():
    rows = b1_rows(200, prefix="o", start="2010-01-01") + b1_rows(200, prefix="n", start="2023-01-01")
    table, labels = frames(rows, "pred_move")
    out = band_quantiles(table, labels, events(["ev"], [EVENT], [2.0], "pred_move")).iloc[0]
    want, n = band_oracle(*arrays(rows), 2.0)
    assert out.reason is None and out.n_rows == n
    np.testing.assert_allclose(out[QCOLS].to_numpy(float), want)


def test_b1_invalid_donors_dropped_and_logged(capsys):
    good = b1_rows(200)
    bad = [("bad_pred", "2010-01-01", "2010-01-01", None, 1.0),
           ("bad_nan", "2010-01-01", "2010-01-01", np.nan, 1.0),
           ("bad_y", "2010-01-01", "2010-01-01", 1.0, np.inf)]
    table, labels = frames(good + bad, "pred_move")
    out = band_quantiles(table, labels, events(["ev"], [EVENT], [2.0], "pred_move")).iloc[0]
    want, n = band_oracle(*arrays(good), 2.0)
    assert out.reason is None and out.n_rows == n
    np.testing.assert_allclose(out[QCOLS].to_numpy(float), want)
    assert "invalid_donors=3" in capsys.readouterr().err


def test_b1_forecast_missing_and_empty_pool():
    table, labels = frames(b1_rows(200), "pred_move")
    ev = events(["nan", "none", "ok"], [EVENT] * 3, [np.nan, None, 2.0], "pred_move")
    out = band_quantiles(table, labels, ev)
    for eid in ("nan", "none"):
        assert out.loc[eid, "reason"] == "forecast_missing"
        assert out.loc[eid, QCOLS].isna().all() and np.isnan(out.loc[eid, "bucket"])
        assert out.loc[eid, "n_rows"] == 0
    assert out.loc["ok", "reason"] is None
    table, labels = frames(b1_rows(200, avail="2025-01-01"), "pred_move")
    empty = band_quantiles(table, labels, events(["ev"], [EVENT], [2.0], "pred_move")).iloc[0]
    assert empty.reason == "pool_too_small" and empty[QCOLS].isna().all() and empty.n_rows == 0


def test_b1_non_finite_output_blanks_quantiles():
    table, labels = frames(b1_rows(200, ratio=11.0), "pred_move")
    row = band_quantiles(table, labels, events(["ev"], [EVENT], [1e308], "pred_move")).iloc[0]
    assert row.reason == "non_finite_output"
    assert row[QCOLS].isna().all() and np.isnan(row.bucket) and row.n_rows == 0


# ---------------------------------------------------------------- B2


def test_b2_happy_path_matches_oracle():
    rows = b2_rows(600)
    table, labels = frames(rows, "implied_move")
    out = implied_quantiles(table, labels, events(["ev"], [EVENT], [2.0], "implied_move")).iloc[0]
    implied, y = arrays(rows)
    assert out.reason is None and out.n_rows == 600 and out.n_dropped == 0
    np.testing.assert_allclose(out[QCOLS].to_numpy(float), implied_oracle(implied, y, 2.0))


@pytest.mark.parametrize("count, expected", [(MIN_IMPLIED_DONORS - 1, "pool_too_small"),
                                             (MIN_IMPLIED_DONORS, None)])
def test_b2_threshold_is_500(count, expected):
    table, labels = frames(b2_rows(count), "implied_move")
    out = implied_quantiles(table, labels, events(["ev"], [EVENT], [2.0], "implied_move")).iloc[0]
    assert out.reason == expected
    assert out.n_rows == (0 if expected else count)


@pytest.mark.parametrize("day, avail", [("2023-05-31", "2023-05-31"),   # one day before the window
                                        ("2024-06-03", "2024-06-03"),   # dated in June 2024
                                        ("2024-05-10", "2024-06-01")])  # in window, label not before the month
def test_b2_excluded_donors_do_not_count(day, avail):
    base = b2_rows(500)
    table, labels = frames(base + [("edge", day, avail, 2.0, 1e6)], "implied_move")
    out = implied_quantiles(table, labels, events(["ev"], [EVENT], [2.0], "implied_move")).iloc[0]
    implied, y = arrays(base)
    assert out.reason is None and out.n_rows == 500 and out.n_dropped == 0
    np.testing.assert_allclose(out[QCOLS].to_numpy(float), implied_oracle(implied, y, 2.0))


def test_b2_window_start_is_included():
    rows = b2_rows(499) + [("start", "2023-06-01", "2023-06-01", 5.0, 4.0)]
    table, labels = frames(rows, "implied_move")
    out = implied_quantiles(table, labels, events(["ev"], [EVENT], [2.0], "implied_move")).iloc[0]
    implied, y = arrays(rows)
    assert out.reason is None and out.n_rows == 500
    np.testing.assert_allclose(out[QCOLS].to_numpy(float), implied_oracle(implied, y, 2.0))


BAD_IMPLIED = [("bad0", None), ("bad1", np.nan), ("bad2", 0.0), ("bad3", -1.0), ("bad4", np.inf)]


def _bad_rows(day="2023-09-01"):
    return [(name, day, day, value, 1.0) for name, value in BAD_IMPLIED]


def test_b2_invalid_donor_implied_dropped_and_counted():
    good = b2_rows(500)
    table, labels = frames(good + _bad_rows(), "implied_move")
    out = implied_quantiles(table, labels, events(["ev"], [EVENT], [2.0], "implied_move")).iloc[0]
    implied, y = arrays(good)
    assert out.reason is None and out.n_rows == 500 and out.n_dropped == 5
    np.testing.assert_allclose(out[QCOLS].to_numpy(float), implied_oracle(implied, y, 2.0))


def test_b2_invalid_donors_do_not_count_toward_threshold():
    table, labels = frames(b2_rows(499) + _bad_rows(), "implied_move")
    out = implied_quantiles(table, labels, events(["ev"], [EVENT], [2.0], "implied_move")).iloc[0]
    assert out.reason == "pool_too_small" and out.n_rows == 0 and out.n_dropped == 5


def test_b2_invalid_donor_outside_window_not_counted():
    rows = b2_rows(500) + [("old", "2020-01-01", "2020-01-01", None, 1.0)]
    table, labels = frames(rows, "implied_move")
    out = implied_quantiles(table, labels, events(["ev"], [EVENT], [2.0], "implied_move")).iloc[0]
    assert out.reason is None and out.n_rows == 500 and out.n_dropped == 0


def test_b2_invalid_label_donor_dropped_and_logged(capsys):
    rows = b2_rows(500) + [("lbl", "2023-09-01", "2023-09-01", 2.0, np.nan)]
    table, labels = frames(rows, "implied_move")
    out = implied_quantiles(table, labels, events(["ev"], [EVENT], [2.0], "implied_move")).iloc[0]
    assert out.reason is None and out.n_rows == 500 and out.n_dropped == 0
    assert "invalid_label_donors=1" in capsys.readouterr().err


def test_b2_log_counts_distinct_donors_across_events(capsys):
    table, labels = frames(b2_rows(500) + _bad_rows(), "implied_move")
    implied_quantiles(table, labels, events(["a", "b"], [EVENT] * 2, [2.0, 3.0], "implied_move"))
    err = capsys.readouterr().err
    assert "invalid_implied_donors=5" in err and "invalid_implied_donors=10" not in err


@pytest.mark.parametrize("bad", [None, np.nan, 0.0, -1.0, np.inf])
def test_b2_event_invalid_implied_move(bad):
    rows = b2_rows(500) + [("bad", "2023-09-01", "2023-09-01", None, 1.0)]
    table, labels = frames(rows, "implied_move")
    out = implied_quantiles(table, labels, events(["ev"], [EVENT], [bad], "implied_move")).iloc[0]
    assert out.reason == "implied_move_invalid"
    assert out.n_rows == 0 and out.n_dropped == 1
    assert out[QCOLS].isna().all()


def test_b2_non_finite_output_blanks_quantiles():
    table, labels = frames(b2_rows(500, ratio=11.0), "implied_move")
    row = implied_quantiles(table, labels, events(["ev"], [EVENT], [1e308], "implied_move")).iloc[0]
    assert row.reason == "non_finite_output"
    assert row[QCOLS].isna().all() and row.n_rows == 0 and row.n_dropped == 0


# ---------------------------------------------------------------- shared B1 / B2 contract

SHARED = [(band_quantiles, "pred_move", "B1", B1_COLS, lambda: b1_rows(200)),
          (implied_quantiles, "implied_move", "B2", B2_COLS, lambda: b2_rows(500))]
SHARED_IDS = ["B1", "B2"]


@pytest.mark.parametrize("fn, model, name, cols, rows_fn", SHARED, ids=SHARED_IDS)
def test_shared_order_columns_and_single_call_match(fn, model, name, cols, rows_fn):
    table, labels = frames(rows_fn(), model)
    ids, values = ["z", "a", "m"], [2.0, 3.0, 1.5]
    out = fn(table, labels, events(ids, [EVENT] * 3, values, model))
    assert out.index.tolist() == ids
    assert list(out.columns) == cols
    for eid, value in zip(ids, values):
        single = fn(table, labels, events([eid], [EVENT], [value], model))
        pd.testing.assert_frame_equal(out.loc[[eid]], single)


@pytest.mark.parametrize("fn, model, name, cols, rows_fn", SHARED, ids=SHARED_IDS)
def test_shared_empty_events_keep_columns(fn, model, name, cols, rows_fn):
    table, labels = frames(rows_fn(), model)
    empty = pd.DataFrame({"event_date": pd.Series([], dtype="datetime64[ns]"),
                          model: pd.Series([], dtype=float)})
    out = fn(table, labels, empty)
    assert out.empty and list(out.columns) == cols


@pytest.mark.parametrize("fn, model, name, cols, rows_fn", SHARED, ids=SHARED_IDS)
def test_shared_shuffled_donors_same_result(fn, model, name, cols, rows_fn):
    rows = rows_fn()
    ev = events(["ev"], [EVENT], [2.0], model)
    base = fn(*frames(rows, model), ev)
    order = np.random.default_rng(3).permutation(len(rows))
    shuffled = fn(*frames([rows[k] for k in order], model), ev)
    pd.testing.assert_frame_equal(base, shuffled)


@pytest.mark.parametrize("fn, model, name, cols, rows_fn", SHARED, ids=SHARED_IDS)
def test_shared_duplicate_index_raises(fn, model, name, cols, rows_fn):
    table, labels = frames(rows_fn(), model)
    ev = events(["ev", "ev2"], [EVENT] * 2, [2.0, 2.0], model)
    with pytest.raises(ValueError):
        fn(_dup(table), labels, ev)
    with pytest.raises(ValueError):
        fn(table, _dup(labels), ev)
    with pytest.raises(ValueError):
        fn(table, labels, _dup(ev))


@pytest.mark.parametrize("fn, model, name, cols, rows_fn", SHARED, ids=SHARED_IDS)
def test_shared_null_event_date_raises(fn, model, name, cols, rows_fn):
    table, labels = frames(rows_fn(), model)
    with pytest.raises(ValueError):
        fn(table, labels, events(["ev"], [pd.NaT], [2.0], model))


@pytest.mark.parametrize("fn, model, name, cols, rows_fn", SHARED, ids=SHARED_IDS)
def test_shared_null_donor_event_date_raises(fn, model, name, cols, rows_fn):
    rows = rows_fn()
    rid, _, avail, x, y = rows[0]
    rows[0] = (rid, None, avail, x, y)
    table, labels = frames(rows, model)
    with pytest.raises(ValueError):
        fn(table, labels, events(["ev"], [EVENT], [2.0], model))


@pytest.mark.parametrize("fn, model, name, cols, rows_fn", SHARED, ids=SHARED_IDS)
def test_shared_logging_goes_to_stderr_only(capsys, fn, model, name, cols, rows_fn):
    table, labels = frames(rows_fn(), model)
    fn(table, labels, events(["ev"], [EVENT], [2.0], model))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"quantile {name}" in captured.err and "scored=1" in captured.err
