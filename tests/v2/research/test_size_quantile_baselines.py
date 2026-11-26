"""Tests for EXP-190 Q4a1 baseline B0 (experiments/size_quantile_baselines.py). Synthetic fixtures only."""
import numpy as np
import pandas as pd
import pytest

from engine.pnl_sim import ResidualPool
from experiments.size_pnl_pool import build_pool
from experiments.size_quantile_baselines import pool_quantiles
from experiments.size_quantile_metrics import Rules

LEVELS = Rules().levels
QCOLS = [f"q{round(a * 100)}" for a in LEVELS]
EVENT_DATE = "2024-06-03"
EVENT_COLS = QCOLS + ["bucket", "fallback", "n_rows", "reason"]


def donors(n, *, start="2010-01-01", pred0=1.0):
    """n donors dated consecutive days from ``start``; pred rises by 0.01 and errors span about [-5, 5)."""
    i = np.arange(n)
    pred = pred0 + 0.01 * i
    err = ((i * 37) % 101) / 10 - 5.0
    dates = (pd.Timestamp(start) + pd.to_timedelta(i, unit="D")).to_numpy()
    return dates, pred, pred + err


def concat(*parts):
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def bucket_donors(n=3000):
    """Deciles of pred are contiguous blocks of 300, and the error depends strongly on the decile."""
    i = np.arange(n)
    pred = 0.5 + 0.003 * i
    err = 3.0 * (i // 300) - 13.5 + 0.1 * (i % 5)
    dates = (pd.Timestamp("2010-01-01") + pd.to_timedelta(i, unit="D")).to_numpy()
    return dates, pred, pred + err


def frames(dates, pred, abs_move, crush=None):
    """(table, labels) indexed by generated event ids; pred_crush is 0 and crush defaults to 0."""
    ids = [f"e{i:06d}" for i in range(len(pred))]
    table = pd.DataFrame({"event_date": pd.to_datetime(np.asarray(dates)),
                          "pred_move": np.asarray(pred, dtype=float), "pred_crush": 0.0}, index=ids)
    crush = np.zeros(len(ids)) if crush is None else np.asarray(crush, dtype=float)
    labels = pd.DataFrame({"abs_move": np.asarray(abs_move, dtype=float), "crush_pct_iv30": crush},
                          index=ids)
    return table, labels


def events_frame(ids, dates, pred):
    return pd.DataFrame({"event_date": pd.to_datetime(np.asarray(dates)),
                         "pred_move": np.asarray(pred, dtype=float)}, index=ids)


def score(table, labels, f, date=EVENT_DATE):
    return pool_quantiles(table, labels, events_frame(["ev"], [date], [f])).iloc[0]


def oracle(pred, err, f, *, levels=LEVELS):
    """Hand-computed bucket path: decile edges of pred, the forecast's bucket, and its members."""
    edges = np.quantile(pred, np.linspace(0, 1, 11)[1:-1])
    bucket = int(np.searchsorted(edges, f, "right"))
    members = np.searchsorted(edges, pred, "right") == bucket
    return np.quantile(np.maximum(f + err[members], 0.0), levels), bucket, int(members.sum())


def with_decoys(dates, pred, abs_move):
    """Append a same-day and a later donor with wild errors; neither may enter the pool for the event."""
    extra = np.array(["2024-06-03", "2024-06-03", "2025-01-01"], dtype="datetime64[ns]")
    return (np.concatenate([dates, extra]), np.concatenate([pred, [1.0, 1.0, 1.0]]),
            np.concatenate([abs_move, [1e6, -1e6, 1e6]]))


def test_threshold_is_strictly_earlier_donors():
    dates, pred, abs_move = with_decoys(*donors(249))
    table, labels = frames(dates, pred, abs_move)
    row = score(table, labels, 2.0)
    assert row.reason == "refused_draw"
    assert row[QCOLS].isna().all() and np.isnan(row.bucket)
    assert not row.fallback and row.n_rows == 0

    dates, pred, abs_move = with_decoys(*donors(250))
    table, labels = frames(dates, pred, abs_move)
    row = score(table, labels, 2.0)
    _, p, a = donors(250)
    assert row.reason is None
    assert row.fallback and row.n_rows == 250
    np.testing.assert_allclose(row[QCOLS].to_numpy(float),
                               np.quantile(np.maximum(2.0 + (a - p), 0), LEVELS))


def test_fallback_uses_all_earlier_donors():
    dates, pred, abs_move = donors(600)
    table, labels = frames(dates, pred, abs_move)
    row = score(table, labels, 2.0)
    err = abs_move - pred
    edges = np.quantile(pred, np.linspace(0, 1, 11)[1:-1])
    assert row.fallback and row.n_rows == 600 and row.reason is None
    assert row.bucket == np.searchsorted(edges, 2.0, "right")
    np.testing.assert_allclose(row[QCOLS].to_numpy(float), np.quantile(np.maximum(2.0 + err, 0), LEVELS))


def test_bucket_path_matches_oracle_and_edge_goes_up():
    dates, pred, abs_move = bucket_donors()
    table, labels = frames(dates, pred, abs_move)
    err = abs_move - pred
    edges = np.quantile(pred, np.linspace(0, 1, 11)[1:-1])
    for f in (4.2, edges[3]):
        row = score(table, labels, f)
        want, bucket, n = oracle(pred, err, f)
        assert not row.fallback
        assert row.bucket == bucket and row.n_rows == n
        np.testing.assert_allclose(row[QCOLS].to_numpy(float), want)
    assert row.bucket == 4


def test_floor_at_zero_never_negative():
    dates, pred, abs_move = donors(600)
    table, labels = frames(dates, pred, pred - 50.0)
    row = score(table, labels, 2.0)
    assert row.reason is None
    assert (row[QCOLS].to_numpy(float) == 0.0).all()


def test_full_history_uses_donors_before_2023():
    old = donors(300, start="2015-01-01")
    recent = donors(300, start="2023-01-01")
    dates, pred, abs_move = concat(old, recent)
    table, labels = frames(dates, pred, abs_move)
    row = score(table, labels, 2.0)
    err = abs_move - pred
    assert row.n_rows == 600 and row.fallback
    np.testing.assert_allclose(row[QCOLS].to_numpy(float), np.quantile(np.maximum(2.0 + err, 0), LEVELS))


def test_donor_with_nan_crush_is_dropped():
    dates, pred, abs_move = donors(600)
    abs_move = abs_move.copy()
    abs_move[-1] = 1e6
    crush = np.zeros(600)
    crush[-1] = np.nan
    table, labels = frames(dates, pred, abs_move, crush=crush)
    row = score(table, labels, 2.0)
    err = abs_move[:-1] - pred[:-1]
    assert row.n_rows == 599
    np.testing.assert_allclose(row[QCOLS].to_numpy(float), np.quantile(np.maximum(2.0 + err, 0), LEVELS))


@pytest.mark.parametrize("missing", [np.nan, None])
def test_forecast_missing_is_not_drawn(missing):
    dates, pred, abs_move = donors(600)
    table, labels = frames(dates, pred, abs_move)
    events = pd.DataFrame({"event_date": pd.to_datetime([EVENT_DATE] * 2),
                           "pred_move": pd.Series([missing, 2.0], index=["a", "b"], dtype=object)}, index=["a", "b"])
    out = pool_quantiles(table, labels, events)
    assert out.loc["a", "reason"] == "forecast_missing"
    assert out.loc["a", QCOLS].isna().all() and np.isnan(out.loc["a", "bucket"])
    assert out.loc["a", "n_rows"] == 0 and not out.loc["a", "fallback"]
    assert out.loc["b", "reason"] is None


def test_empty_events_keep_columns():
    dates, pred, abs_move = donors(600)
    table, labels = frames(dates, pred, abs_move)
    events = pd.DataFrame({"event_date": pd.Series([], dtype="datetime64[ns]"),
                           "pred_move": pd.Series([], dtype=float)})
    out = pool_quantiles(table, labels, events)
    assert out.empty
    assert list(out.columns) == EVENT_COLS


def test_output_order_follows_events_and_matches_single_calls():
    dates, pred, abs_move = donors(600)
    table, labels = frames(dates, pred, abs_move)
    ids, forecasts = ["z", "a", "m"], [2.0, 3.5, 1.2]
    out = pool_quantiles(table, labels, events_frame(ids, [EVENT_DATE] * 3, forecasts))
    assert out.index.tolist() == ids
    for eid, f in zip(ids, forecasts):
        single = pool_quantiles(table, labels, events_frame([eid], [EVENT_DATE], [f]))
        pd.testing.assert_frame_equal(out.loc[[eid]], single)


def test_shuffled_donor_rows_do_not_change_result():
    dates, pred, abs_move = bucket_donors()
    table, labels = frames(dates, pred, abs_move)
    events = events_frame(["ev"], [EVENT_DATE], [4.2])
    base = pool_quantiles(table, labels, events)
    order = np.random.default_rng(3).permutation(len(table))
    shuffled = pool_quantiles(table.iloc[order], labels.iloc[order], events)
    pd.testing.assert_frame_equal(base, shuffled)


def _with_dup_index(frame):
    frame = frame.copy()
    ids = list(frame.index)
    ids[1] = ids[0]
    frame.index = pd.Index(ids)
    return frame


def test_duplicate_index_raises_in_each_frame():
    dates, pred, abs_move = donors(600)
    table, labels = frames(dates, pred, abs_move)
    events = events_frame(["ev", "ev2"], [EVENT_DATE] * 2, [2.0, 2.0])
    with pytest.raises(ValueError):
        pool_quantiles(_with_dup_index(table), labels, events)
    with pytest.raises(ValueError):
        pool_quantiles(table, _with_dup_index(labels), events)
    with pytest.raises(ValueError):
        pool_quantiles(table, labels, _with_dup_index(events))


def test_null_event_date_raises():
    dates, pred, abs_move = donors(600)
    table, labels = frames(dates, pred, abs_move)
    events = events_frame(["ev"], [pd.NaT], [2.0])
    with pytest.raises(ValueError):
        pool_quantiles(table, labels, events)


def test_non_finite_output_blanks_quantiles_but_keeps_bucket():
    # Every donor shares pred 5.0 (bucket 9, all 250 rows). Top 26 errors are +inf, so the 0.9 quantile is not finite.
    n = 250
    dates = (pd.Timestamp("2010-01-01") + pd.to_timedelta(np.arange(n), unit="D")).to_numpy()
    pred = np.full(n, 5.0)
    abs_move = np.concatenate([np.full(n - 26, 1.0), np.full(26, np.inf)])
    table, labels = frames(dates, pred, abs_move)
    row = score(table, labels, 6.0)
    assert row.reason == "non_finite_output"
    assert row[QCOLS].isna().all()
    assert row.bucket == 9 and not row.fallback
    assert row.n_rows == n


def test_logging_goes_to_stderr_only(capsys):
    dates, pred, abs_move = donors(600)
    table, labels = frames(dates, pred, abs_move)
    pool_quantiles(table, labels, events_frame(["ev"], [EVENT_DATE], [2.0]))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "quantile B0" in captured.err and "scored=1" in captured.err


def test_sampled_rows_lie_in_oracle_members():
    dates, pred, abs_move = bucket_donors()
    table, labels = frames(dates, pred, abs_move)
    err = abs_move - pred
    f = 4.2
    edges = np.quantile(pred, np.linspace(0, 1, 11)[1:-1])
    members = np.searchsorted(edges, pred, "right") == np.searchsorted(edges, f, "right")
    pool = build_pool(table, labels, floor=None)
    move, _ = pool.draw(pd.Timestamp(EVENT_DATE), f, 20000, np.random.default_rng(1))
    assert len(np.unique(move)) > 1
    assert np.isin(move, err[members]).all()


def _draw_evidence(table, labels, f, date=EVENT_DATE):
    ev = {}
    build_pool(table, labels, floor=None).draw(pd.Timestamp(date), f, 0, np.random.default_rng(0), evidence=ev)
    return ev


@pytest.mark.parametrize("kind", ["bucket", "fallback"])
def test_labels_match_draw_own_evidence(kind):
    if kind == "bucket":
        dates, pred, abs_move = bucket_donors()
        edges = np.quantile(pred, np.linspace(0, 1, 11)[1:-1])
        forecasts = [4.2, edges[3], 0.0, 100.0]
    else:
        dates, pred, abs_move = donors(600)
        forecasts = [2.0, 0.0, 100.0]
    table, labels = frames(dates, pred, abs_move)
    ids = [f"f{i}" for i in range(len(forecasts))]
    out = pool_quantiles(table, labels, events_frame(ids, [EVENT_DATE] * len(ids), forecasts))
    for eid, f in zip(ids, forecasts):
        ev = _draw_evidence(table, labels, f)
        row = out.loc[eid]
        assert row.bucket == ev["bucket_index"]
        assert row.fallback == ev["fallback_used"]
        want = len(ev["fallback_indices"]) if ev["fallback_used"] else len(ev["eligible_indices"])
        assert row.n_rows == want


def test_pool_quantiles_never_requests_evidence(monkeypatch):
    calls = []
    original = ResidualPool.draw

    def recording(self, cutoff, prediction, n, rng, *, evidence=None):
        calls.append(evidence)
        return original(self, cutoff, prediction, n, rng, evidence=evidence)

    monkeypatch.setattr(ResidualPool, "draw", recording)
    dates, pred, abs_move = bucket_donors()
    table, labels = frames(dates, pred, abs_move)
    events = events_frame(["a", "b", "c"], [EVENT_DATE] * 3, [4.2, 0.0, np.nan])
    pool_quantiles(table, labels, events)
    assert len(calls) == 2
    assert all(ev is None for ev in calls)


def test_guard_raises_when_draw_disagrees(monkeypatch):
    original = ResidualPool.draw

    def one_short(self, cutoff, prediction, n, rng, *, evidence=None):
        move, crush = original(self, cutoff, prediction, n, rng, evidence=evidence)
        return move[:-1], crush[:-1]

    monkeypatch.setattr(ResidualPool, "draw", one_short)
    dates, pred, abs_move = donors(600)
    table, labels = frames(dates, pred, abs_move)
    with pytest.raises(RuntimeError):
        score(table, labels, 2.0)
