"""Tests for EXP-190 Q4a1 baseline B0 (experiments/size_quantile_baselines.py). Synthetic fixtures only."""
import numpy as np
import pandas as pd
import pytest

from engine.pnl_sim import MIN_POOL, ResidualPool
from engine.v2.ops.errors import OpsError
from engine.v2.research.magnitude_inputs import forecast_cutoff
from experiments.size_pnl_pool import build_pool
from experiments.size_quantile_baselines import lead1_decision_dates, pool_quantiles
from experiments.size_quantile_metrics import Rules
from tests.v2.research.magnitude_trial_support import calendar_sessions

LEVELS = Rules().levels
QCOLS = [f"q{round(a * 100)}" for a in LEVELS]
EVENT_DATE = "2024-06-03"
DECISION_DATE = "2024-06-02"
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


def plus_days(dates, days):
    """ISO text ``days`` after each date."""
    return [(pd.Timestamp(d) + pd.Timedelta(days=days)).strftime("%Y-%m-%d") for d in np.asarray(dates)]


def frames(dates, pred, abs_move, crush=None):
    """(table, labels) indexed by generated event ids; pred_crush is 0, crush defaults to 0, and each label is
    available the day after its donor's event date."""
    ids = [f"e{i:06d}" for i in range(len(pred))]
    table = pd.DataFrame({"event_date": pd.to_datetime(np.asarray(dates)),
                          "pred_move": np.asarray(pred, dtype=float), "pred_crush": 0.0}, index=ids)
    crush = np.zeros(len(ids)) if crush is None else np.asarray(crush, dtype=float)
    labels = pd.DataFrame({"abs_move": np.asarray(abs_move, dtype=float), "crush_pct_iv30": crush,
                           "available": plus_days(dates, 1)}, index=ids)
    return table, labels


def causal_frames(dates, pred, abs_move, available):
    """Like frames, with each label's available_as_of_date given explicitly (ISO text)."""
    table, labels = frames(dates, pred, abs_move)
    return table, labels.assign(available=list(available))


def events_frame(ids, dates, pred):
    """Events with event_date, decision_date (the day before the event) and pred_move."""
    when = pd.to_datetime(np.asarray(dates))
    return pd.DataFrame({"event_date": when, "decision_date": when - pd.Timedelta(days=1),
                         "pred_move": np.asarray(pred, dtype=float)}, index=ids)


def score(table, labels, f, date=EVENT_DATE):
    return pool_quantiles(table, labels, events_frame(["ev"], [date], [f])).iloc[0]


def oracle(pred, err, f, *, levels=LEVELS):
    """Hand-computed bucket path: decile edges of pred, the forecast's bucket, and its members."""
    edges = np.quantile(pred, np.linspace(0, 1, 11)[1:-1])
    bucket = int(np.searchsorted(edges, f, "right"))
    members = np.searchsorted(edges, pred, "right") == bucket
    return np.quantile(np.maximum(f + err[members], 0.0), levels), bucket, int(members.sum())


def causal_oracle(pred, err, f):
    """oracle() with ResidualPool's fallback: every eligible donor when the forecast's bucket holds under MIN_POOL."""
    quant, bucket, members = oracle(pred, err, f)
    if members < MIN_POOL:
        return np.quantile(np.maximum(f + err, 0.0), LEVELS), bucket, err.size
    return quant, bucket, members


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
                           "decision_date": pd.to_datetime([DECISION_DATE] * 2),
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
                           "decision_date": pd.Series([], dtype="datetime64[ns]"),
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
    assert "quantile B0" in captured.err and "scored=1" in captured.err and "pools=1" in captured.err


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


# ---------- causal donors: available_as_of_date <= the lead-1 decision date ----------

def test_donor_available_on_the_decision_date_counts_and_one_day_later_does_not():
    base = donors(250)
    on_day = (np.array(["2024-05-01"], dtype="datetime64[ns]"), np.array([1.0]), np.array([31.0]))
    late = (np.array(["2024-06-01"], dtype="datetime64[ns]"), np.array([1.0]), np.array([-39.0]))
    dates, pred, abs_move = concat(base, on_day, late)
    available = plus_days(base[0], 1) + ["2024-06-02", "2024-06-03"]
    table, labels = causal_frames(dates, pred, abs_move, available)
    row = score(table, labels, 2.0)
    err = np.concatenate([base[2] - base[1], [30.0]])
    assert row.reason is None and row.n_rows == 251
    np.testing.assert_allclose(row[QCOLS].to_numpy(float), np.quantile(np.maximum(2.0 + err, 0), LEVELS))


@pytest.mark.parametrize("eligible, scored", [(249, False), (250, True)])
def test_min_pool_threshold_counts_eligible_donors_only(eligible, scored):
    base = donors(eligible)
    late = (np.array(["2024-05-30"], dtype="datetime64[ns]"), np.array([1.0]), np.array([1001.0]))
    dates, pred, abs_move = concat(base, late)
    table, labels = causal_frames(dates, pred, abs_move, plus_days(base[0], 1) + ["2024-06-03"])
    row = score(table, labels, 2.0)
    if scored:
        err = base[2] - base[1]
        assert row.reason is None and row.n_rows == eligible == 250
        np.testing.assert_allclose(row[QCOLS].to_numpy(float), np.quantile(np.maximum(2.0 + err, 0), LEVELS))
    else:
        assert row.reason == "refused_draw" and row.n_rows == 0


def test_each_target_uses_its_own_eligible_donors_whatever_the_input_order():
    old = donors(600)
    new = donors(300, start="2024-01-01")
    dates, pred, abs_move = concat(old, new)
    available = plus_days(old[0], 1) + plus_days(new[0], 1)
    table, labels = causal_frames(dates, pred, abs_move, available)
    err = abs_move - pred
    avail = pd.to_datetime(available).to_numpy()

    def want(event_date, f):
        cutoff = pd.Timestamp(event_date).to_datetime64()
        decision_day = (pd.Timestamp(event_date) - pd.Timedelta(days=1)).to_datetime64()
        keep = (avail <= decision_day) & (dates < cutoff)
        return causal_oracle(pred[keep], err[keep], f)

    ids, days, forecasts = ["late", "early"], ["2024-12-02", "2024-03-02"], [2.0, 3.0]
    out = pool_quantiles(table, labels, events_frame(ids, days, forecasts))
    assert out.index.tolist() == ids
    for eid, day, f in zip(ids, days, forecasts):
        quant, bucket, n = want(day, f)
        row = out.loc[eid]
        assert row.reason is None and row.n_rows == n and row.bucket == bucket
        np.testing.assert_allclose(row[QCOLS].to_numpy(float), quant)
    reversed_out = pool_quantiles(table, labels, events_frame(ids[::-1], days[::-1], forecasts[::-1]))
    pd.testing.assert_frame_equal(reversed_out.loc[ids], out)


def test_bucket_edges_come_only_from_eligible_donors():
    base_dates, base_pred, base_abs = bucket_donors()
    late_dates = (pd.Timestamp("2024-05-01") + pd.to_timedelta(np.arange(200) % 30, unit="D")).to_numpy()
    dates = np.concatenate([base_dates, late_dates])
    pred = np.concatenate([base_pred, np.full(200, 4.2)])
    abs_move = np.concatenate([base_abs, np.full(200, 104.2)])  # the late donors would carry error +100 at 4.2
    available = plus_days(base_dates, 1) + ["2024-06-03"] * 200
    table, labels = causal_frames(dates, pred, abs_move, available)
    row = pool_quantiles(table, labels, events_frame(["ev"], [EVENT_DATE], [4.2])).iloc[0]
    want, bucket, n = oracle(base_pred, base_abs - base_pred, 4.2)
    assert row.reason is None and not row.fallback
    assert row.bucket == bucket and row.n_rows == n
    np.testing.assert_allclose(row[QCOLS].to_numpy(float), want)


@pytest.mark.parametrize("break_it, message", [
    ("no_decision_column", "events must have a decision_date column"),
    ("null_decision", "events decision_date must not be null or unparseable"),
    ("decision_on_event_date", "events decision_date must be strictly before event_date"),
    ("no_available_column", "labels must have an available column"),
    ("null_available", "labels available must not be null or unparseable"),
    ("unparseable_decision", "events decision_date must not be null or unparseable"),
    ("unparseable_available", "labels available must not be null or unparseable"),
])
def test_missing_null_or_non_causal_dates_raise(break_it, message):
    dates, pred, abs_move = donors(600)
    table, labels = frames(dates, pred, abs_move)
    events = events_frame(["ev"], [EVENT_DATE], [2.0])
    if break_it == "no_decision_column":
        events = events.drop(columns="decision_date")
    elif break_it == "null_decision":
        events["decision_date"] = pd.NaT
    elif break_it == "decision_on_event_date":
        events["decision_date"] = events["event_date"]
    elif break_it == "no_available_column":
        labels = labels.drop(columns="available")
    elif break_it == "null_available":
        labels["available"] = None
    elif break_it == "unparseable_decision":
        events["decision_date"] = "not-a-date"
    elif break_it == "unparseable_available":
        labels["available"] = "not-a-date"
    with pytest.raises(ValueError, match=message):
        pool_quantiles(table, labels, events)


def test_events_with_the_same_eligible_labels_share_one_pool(capsys):
    # Donors dated 2010-2011 are all available well before 2024-06. Five late donors dated 2024-06-05 become available
    # 2024-06-06: after "a" (decision 06-02) and "b" (decision 06-03), before "c" (decision 06-09).
    # So "a" and "b" admit the same 600 labels and share one pool; "c" admits 605 labels and builds a second pool.
    base = donors(600)
    late = (np.array(["2024-06-05"] * 5, dtype="datetime64[ns]"), np.full(5, 1.0), np.full(5, 2.0))
    dates, pred, abs_move = concat(base, late)
    table, labels = causal_frames(dates, pred, abs_move, plus_days(base[0], 1) + ["2024-06-06"] * 5)
    events = events_frame(["a", "b", "c"], ["2024-06-03", "2024-06-04", "2024-06-10"], [2.0, 2.0, 2.0])
    out = pool_quantiles(table, labels, events)
    err = capsys.readouterr().err
    assert "events=3 scored=3" in err and "pools=2" in err
    assert all(reason is None for reason in out["reason"])
    # Ten buckets of about 60 donors each are under MIN_POOL, so every event draws its whole eligible pool.
    assert out["fallback"].all()
    # a and b: the 600 base donors (all dated before their events). c: those 600 plus the 5 late donors.
    assert out.loc["a", "n_rows"] == 600
    assert out.loc["b", "n_rows"] == 600
    assert out.loc["c", "n_rows"] == 605


def _target(event_id, event_date, forecasts):
    return {"event_id": event_id, "event_date": event_date, "forecasts": forecasts}


def test_lead1_decision_dates_come_from_forecast_cutoff_output():
    # A weekday calendar with no holidays in the window. AMC 2024-06-12: the session before the event.
    # BMO 2024-06-14: two sessions before the event (06-13 is one, 06-12 is two).
    calendar = calendar_sessions()
    events = [("amc", "2024-06-12", "AMC"), ("bmo", "2024-06-14", "BMO")]
    targets = [_target(eid, day, [forecast_cutoff(day, session, horizon, calendar) for horizon in (1, 3, 5)])
               for eid, day, session in events]
    assert all("decision_date" not in f for target in targets for f in target["forecasts"])
    out = lead1_decision_dates(targets, pd.Index(["amc", "bmo"]))
    assert out.tolist() == [pd.Timestamp("2024-06-11"), pd.Timestamp("2024-06-12")]


def test_lead1_decision_date_is_the_horizon_one_entry():
    targets = [_target("a", "2024-06-03", [{"horizon_sessions": 5, "source_session_ceiling": "2024-05-24"},
                                           {"horizon_sessions": 3, "source_session_ceiling": "2024-05-29"},
                                           {"horizon_sessions": 1, "source_session_ceiling": "2024-06-02"}])]
    out = lead1_decision_dates(targets, pd.Index(["a"]))
    assert out.index.tolist() == ["a"]
    assert pd.api.types.is_datetime64_dtype(out)
    assert out.iloc[0] == pd.Timestamp("2024-06-02")


@pytest.mark.parametrize("case", ["no_lead1", "two_lead1", "missing_id", "bad_text", "not_before_event",
                                  "decision_date_only"])
def test_lead1_decision_date_refusals(case):
    good = {"horizon_sessions": 1, "source_session_ceiling": "2024-06-02"}
    targets = {
        "no_lead1": [_target("a", "2024-06-03", [{"horizon_sessions": 3, "source_session_ceiling": "2024-05-29"}])],
        "two_lead1": [_target("a", "2024-06-03", [good, dict(good)])],
        "missing_id": [_target("b", "2024-06-03", [good])],
        "bad_text": [_target("a", "2024-06-03", [{"horizon_sessions": 1, "source_session_ceiling": "2024-02-30"}])],
        "not_before_event": [_target("a", "2024-06-03", [{"horizon_sessions": 1, "source_session_ceiling": "2024-06-03"}])],
        "decision_date_only": [_target("a", "2024-06-03", [{"horizon_sessions": 1, "decision_date": "2024-06-02"}])],
    }[case]
    with pytest.raises(OpsError) as exc:
        lead1_decision_dates(targets, pd.Index(["a"]))
    assert exc.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.problem.details.get("reason") == "decision_date_invalid"
    assert "decision_date_invalid" in exc.value.problem.message
    assert "event a " in exc.value.problem.message
