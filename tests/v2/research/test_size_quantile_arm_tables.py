"""Tests for EXP-190 Q5b part 1 (experiments/size_quantile_arm_tables.py).

Synthetic fixtures only. Oracles are plain loops or a sorted-group bootstrap written here, not the module's code.
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from engine.v2.ops.errors import OpsError
from experiments import size_quantile_arm_tables as tables
from experiments.size_pnl_contrast import SEED
from experiments.size_quantile_baselines import pool_quantiles
from experiments.size_quantile_calibration import calibrate

LV = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
QCOLS = ["q10", "q25", "q40", "q50", "q60", "q75", "q90"]
QUARTERS = ["2023Q3", "2023Q4"] + [f"2024Q{k}" for k in range(1, 5)] + [f"2025Q{k}" for k in range(1, 5)]
TICKERS = ["AAA", "BBB", "CCC"]
R = 200  # resamples, small for speed
OFF = np.array([-0.5, -0.2, 0.0, 0.1, 0.3, 0.6, 1.0])
OFFS = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
ROOT = Path(__file__).resolve().parents[3]
SHAPES = {"11": [0.0, 0.5, 0.6, 0.8, 0.9, 1.5, 2.0],  # covers p25-p75 and p10-p90 of y = 1
          "01": [0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0],  # covers p10-p90 only
          "00": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}  # covers neither
BOTH, P25, P10, NONE, TIE = (5, 3), (5, 0), (4, 4), (0, 0), (3, 0)  # (n11, n01) out of ten events


def make_events(per_quarter=4, drop=()):
    """Admitted events in the ten quarters, ``per_quarter`` each, tickers cycling; quarters in ``drop`` absent."""
    rows = []
    for qi, q in enumerate(QUARTERS):
        if q in drop:
            continue
        start = pd.Period(q, freq="Q").start_time
        for i in range(per_quarter):
            rows.append({"id": f"{q}-{i}", "event_date": start + pd.Timedelta(days=i),
                         "ticker": TICKERS[(qi + i) % 3], "y": 1.0 + 0.1 * i + 0.05 * qi, "m0": 1.5, "m3": 2.0})
    return pd.DataFrame(rows).set_index("id")


def qframe(ids, q, reason=None):
    """Seven QCOLS and a reason column (None = scored), indexed by ids."""
    out = pd.DataFrame(np.asarray(q, float).reshape(len(ids), 7), index=list(ids), columns=QCOLS)
    out["reason"] = pd.Series([None] * len(ids) if reason is None else list(reason), index=out.index, dtype=object)
    return out


def quarter_of(events):
    return events["event_date"].dt.to_period("Q").astype(str).to_numpy()


def perfect_vs_shift(events, bad=()):
    """Arm = the realised move (zero pinball); B0 = move + 1. Quarters in ``bad`` get arm = B0 (no gain)."""
    y = events["y"].to_numpy(float)
    arm = np.repeat(y[:, None], 7, axis=1)
    b0 = arm + 1.0
    hit = np.array([q in bad for q in quarter_of(events)], dtype=bool)
    arm[hit] = b0[hit]
    return qframe(events.index, arm), qframe(events.index, b0)


def mark(frame, ids, reason):
    out = frame.copy()
    out.loc[list(ids), "reason"] = reason
    return out


def row_of(table, slice_name, label):
    return table.set_index(["slice", "label"]).loc[(slice_name, label)]


def pin_loops(y, q):
    """Pinball per event (rows) and level (columns), by plain loops."""
    return [[max(a * (yy - qq), (a - 1) * (yy - qq)) for a, qq in zip(LV, row)] for yy, row in zip(y, q)]


def oracle_interval(d, labels, resamples=R):
    """Cluster bootstrap written here: sorted groups, one draw of group indices, 5th and 95th percentiles."""
    keys = sorted(set(labels))
    sums = np.array([d[labels == k].sum() for k in keys])
    counts = np.array([(labels == k).sum() for k in keys])
    idx = np.random.default_rng(SEED).integers(0, len(keys), size=(resamples, len(keys)))
    stats = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return tuple(np.percentile(stats, [5, 95]))


def shape_frame(events, types):
    """Ten events per quarter: the first n11 take shape "11", the next n01 take "01", the rest "00"."""
    code = {}
    for q in QUARTERS:
        n11, n01 = types[q]
        for i in range(10):
            code[f"{q}-{i}"] = "11" if i < n11 else ("01" if i < n11 + n01 else "00")
    return qframe(events.index, [SHAPES[code[i]] for i in events.index])


def calib_result(arm_spec, b0_spec=(5, 3)):
    """Calibration block for y = 1 everywhere; arm_spec lists ten (n11, n01) pairs, one per quarter."""
    events = make_events(per_quarter=10)
    events["y"] = 1.0
    arm = shape_frame(events, dict(zip(QUARTERS, arm_spec)))
    b0 = shape_frame(events, {q: b0_spec for q in QUARTERS})
    return tables.evaluate_arm("S", arm, b0, events, resamples=R)["calibration"]


def tiny_frames():
    events = pd.DataFrame({"event_date": pd.to_datetime(["2023-07-05", "2023-08-10"]), "ticker": ["AAA", "BBB"],
                           "y": [1.0, 2.0], "m0": [1.5, 1.5], "m3": [2.2, np.nan]}, index=["e1", "e2"])
    q_arm = [[0.0, 1.0, 1.2, 1.3, 1.4, 1.6, 2.0], [0.5, 0.9, 1.0, 1.1, 1.2, 1.7, 2.0]]
    arm = qframe(["e1", "e2"], q_arm)
    arm["crps"] = [0.2, 0.4]
    b0 = qframe(["e1", "e2"], np.array(q_arm) + 0.5)
    raw = pd.DataFrame([q_arm[0], [0.5, 0.9, 1.0, 1.1, 1.2, 1.7, 1.6]], index=["e1", "e2"], columns=QCOLS)
    return events, arm, b0, raw


# ---------- a. row metrics against hand values ----------

def test_row_metrics_match_hand_values():
    events, arm, b0, raw = tiny_frames()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # empty slices must not warn
        res = tables.evaluate_arm("S", arm, b0, events, raw=raw, resamples=R)
    t = res["tables"]
    assert list(zip(t["slice"], t["label"])) == (
        [("pooled", "pooled"), ("period", "2023H2"), ("period", "2024"), ("period", "2025")]
        + [("quarter", q) for q in QUARTERS])
    y, q, qb = [1.0, 2.0], arm[QCOLS].to_numpy(float), b0[QCOLS].to_numpy(float)
    pin, pinb = pin_loops(y, q), pin_loops(y, qb)
    pooled = row_of(t, "pooled", "pooled")
    assert pooled["n"] == 2
    for k, a in enumerate(LV):
        assert pooled[f"pinball_p{round(a * 100)}"] == pytest.approx(np.mean([r[k] for r in pin]))
    assert pooled["mean_pinball"] == pytest.approx(np.mean(pin))
    assert pooled["b0_mean_pinball"] == pytest.approx(np.mean(pinb))
    # Coverage: e1 sits exactly on its q25 and e2 exactly on its q90 (both inclusive).
    assert np.mean([yy >= r[1] and yy <= r[5] for yy, r in zip(y, q)]) == 0.5
    assert pooled["cover_p25_p75"] == pytest.approx(0.5)
    assert pooled["cover_p10_p90"] == pytest.approx(1.0)
    assert pooled["le_p25"] == pytest.approx(0.5)
    assert pooled["le_p90"] == pytest.approx(1.0)
    assert pooled["b0_cover_p25_p75"] == pytest.approx(0.5)
    assert pooled["b0_cover_p10_p90"] == pytest.approx(1.0)
    assert pooled["median_width"] == pytest.approx(0.7)  # widths 0.6 and 0.8
    assert pooled["mae_p50"] == pytest.approx(0.6)
    assert (pooled["mae_m0"], pooled["n_m0"]) == (pytest.approx(0.5), 2)
    assert (pooled["mae_m3"], pooled["n_m3"]) == (pytest.approx(1.2), 1)  # the NaN m3 is not counted
    assert (pooled["crps"], pooled["n_crps"]) == (pytest.approx(0.3), 2)
    assert pooled["crossing_rate"] == pytest.approx(0.5)  # e2's raw row is out of order
    assert row_of(t, "quarter", "2023Q3")["n"] == 2
    empty = row_of(t, "quarter", "2023Q4")
    assert empty["n"] == 0 and empty["n_m0"] == 0 and empty["n_crps"] == 0
    assert np.isnan(empty["mean_pinball"]) and np.isnan(empty["crps"]) and np.isnan(empty["crossing_rate"])


# ---------- b. scored share ----------

def test_share_counts_all_admitted_events_and_floor_is_inclusive():
    events = make_events(per_quarter=5)  # 20 admitted events in 2024
    arm, b0 = perfect_vs_shift(events)
    ids = [i for i in events.index if i.startswith("2024")]
    # The arm misses two 2024 events and B0 misses two others: 16 of 20 paired.
    res = tables.evaluate_arm("S", mark(arm, ids[:2], "refused_draw"), mark(b0, ids[2:4], "pool_too_small"),
                              events, resamples=R)
    share = res["share"].set_index("period")
    assert list(share.index) == ["2023H2", "2024", "2025", "pooled"]
    assert (share.loc["2024", "admitted"], share.loc["2024", "paired"]) == (20, 16)
    assert share.loc["2024", "share"] == 0.8 and not bool(share.loc["2024", "below_floor"])
    assert res["status"] == "testable" and res["not_testable_reasons"] == []

    # 15 of 20: just under the floor.
    res = tables.evaluate_arm("S", mark(arm, ids[:3], "refused_draw"), mark(b0, ids[3:5], "pool_too_small"),
                              events, resamples=R)
    share = res["share"].set_index("period")
    assert share.loc["2024", "share"] == 0.75 and bool(share.loc["2024", "below_floor"])
    assert res["status"] == "not_testable"
    assert res["not_testable_reasons"] == ["2024:scored_share_below_floor"]
    assert res["accuracy"]["passed"] is None and res["calibration"]["passed"] is None
    assert len(res["tables"]) == 14


# ---------- c. drops ----------

def test_drops_group_by_year_and_both_reasons():
    events = make_events(per_quarter=5)
    arm, b0 = perfect_vs_shift(events)
    arm = arm.drop(["2024Q1-0", "2024Q3-0", "2023Q3-0"])  # no_output on the arm side
    b0 = b0.drop(["2024Q3-0"])
    arm = mark(arm, ["2024Q1-1", "2024Q1-2"], "refused_draw")  # reason with finite quantiles: unscored
    b0 = mark(b0, ["2024Q2-0"], "pool_too_small")
    res = tables.evaluate_arm("S", arm, b0, events, resamples=R)
    got = [tuple(r) for r in res["drops"].itertuples(index=False)]
    assert got == [(2023, "no_output", "scored", 1),
                   (2024, "no_output", "no_output", 1),
                   (2024, "no_output", "scored", 1),
                   (2024, "refused_draw", "scored", 2),
                   (2024, "scored", "pool_too_small", 1)]


# ---------- c2. Q4 option B: a NaN in-period y is label_missing: in the denominator, never paired ----------

def test_nan_in_period_y_stays_admitted_and_is_label_missing_on_both_sides():
    events = make_events(per_quarter=4)
    arm, b0 = perfect_vs_shift(events)  # quantiles come from the true y, so they are finite before the label drops
    events.loc[["2024Q1-0", "2024Q1-1"], "y"] = np.nan
    res = tables.evaluate_arm("S", arm, b0, events, resamples=R)
    share = res["share"].set_index("period")
    assert (share.loc["2024", "admitted"], share.loc["2024", "paired"]) == (16, 14)
    assert "2024Q1-0" not in res["paired_ids"] and "2024Q1-1" not in res["paired_ids"]
    assert (2024, "label_missing", "label_missing", 2) in [tuple(r) for r in res["drops"].itertuples(index=False)]
    assert res["status"] == "testable"


def test_b0_crps_is_the_mean_and_count_of_finite_paired_b0_crps():
    events, arm, b0 = tiny_frames()[:3]
    b0 = b0.assign(crps=[0.1, np.nan])
    res = tables.evaluate_arm("S", arm, b0, events, resamples=R)
    pooled = row_of(res["tables"], "pooled", "pooled")
    assert (pooled["b0_crps"], pooled["b0_n_crps"]) == (pytest.approx(0.1), 1)
    empty = row_of(res["tables"], "quarter", "2023Q4")
    assert np.isnan(empty["b0_crps"]) and empty["b0_n_crps"] == 0


# ---------- d. outside periods; a period with no pairs ----------

def test_events_outside_the_periods_are_ignored():
    events = make_events(per_quarter=4)
    arm, b0 = perfect_vs_shift(events)
    base = tables.evaluate_arm("S", arm, b0, events, resamples=R)
    extra = pd.DataFrame({"event_date": pd.to_datetime(["2022-06-01", "2026-01-05"]), "ticker": [None, None],
                          "y": [np.nan, np.nan], "m0": [np.nan, np.nan], "m3": [np.nan, np.nan]},
                         index=["x1", "x2"])
    nan_rows = qframe(["x1", "x2"], np.full((2, 7), np.nan), ["pool_too_small"] * 2)
    res = tables.evaluate_arm("S", pd.concat([arm, nan_rows]), pd.concat([b0, nan_rows]),
                              pd.concat([events, extra]), resamples=R)
    pd.testing.assert_frame_equal(res["tables"], base["tables"])
    pd.testing.assert_frame_equal(res["share"], base["share"])
    pd.testing.assert_frame_equal(res["drops"], base["drops"])


def test_period_without_paired_events_is_named():
    events = make_events(per_quarter=4)
    arm, b0 = perfect_vs_shift(events)
    ids = [i for i in events.index if i.startswith("2024")]
    res = tables.evaluate_arm("S", mark(arm, ids, "no_output"), b0, events, resamples=R)
    assert res["status"] == "not_testable"
    assert res["not_testable_reasons"] == ["2024:period_no_paired_events", "2024:scored_share_below_floor"]
    assert res["accuracy"]["periods_below"]["2024"] is False


def test_period_with_no_admitted_events_reports_only_no_pairs():
    events = make_events(per_quarter=4, drop=[q for q in QUARTERS if q.startswith("2025")])
    arm, b0 = perfect_vs_shift(events)
    res = tables.evaluate_arm("S", arm, b0, events, resamples=R)
    assert res["status"] == "not_testable"
    assert res["not_testable_reasons"] == ["2025:period_no_paired_events"]
    assert np.isnan(res["share"].set_index("period").loc["2025", "share"])


# ---------- e. accuracy (rule 1) ----------

def test_accuracy_passes_when_arm_is_better_everywhere():
    events = make_events(per_quarter=4)
    arm, b0 = perfect_vs_shift(events)
    res = tables.evaluate_arm("S", arm, b0, events, resamples=R)
    acc = res["accuracy"]
    assert acc["passed"] is True and acc["quarters_below"] == 10
    assert all(acc["periods_below"].values())
    assert acc["quarter_interval"][1] < 0 and acc["ticker_interval"][1] < 0
    assert res["drops"].empty and list(res["drops"].columns) == ["year", "arm_reason", "b0_reason", "n"]


ALL_PERIODS = {"2023H2": True, "2024": True, "2025": True}


@pytest.mark.parametrize("bad, count, passed, periods", [
    (("2023Q3", "2025Q4"), 8, True, ALL_PERIODS),
    (("2023Q3", "2024Q1", "2025Q4"), 7, False, ALL_PERIODS),
    # Both 2023H2 quarters tie, so that period is not below even though 8 quarters are.
    (("2023Q3", "2023Q4"), 8, False, {"2023H2": False, "2024": True, "2025": True}),
])
def test_quarter_count_threshold_is_eight(bad, count, passed, periods):
    events = make_events(per_quarter=4)
    arm, b0 = perfect_vs_shift(events, bad=bad)
    acc = tables.evaluate_arm("S", arm, b0, events, resamples=R)["accuracy"]
    assert acc["quarters_below"] == count and acc["passed"] is passed
    assert dict(acc["periods_below"]) == periods


def test_equal_pinball_everywhere_fails_because_the_test_is_strict():
    events = make_events(per_quarter=4)
    arm, b0 = perfect_vs_shift(events, bad=QUARTERS)
    acc = tables.evaluate_arm("S", arm, b0, events, resamples=R)["accuracy"]
    assert acc["quarters_below"] == 0 and not any(acc["periods_below"].values()) and acc["passed"] is False


@pytest.mark.parametrize("drop, count, passed", [
    (("2023Q3", "2025Q4"), 8, True),
    (("2023Q3", "2024Q1", "2025Q4"), 7, False),
])
def test_missing_quarters_do_not_count_but_the_requirement_stays_eight(drop, count, passed):
    events = make_events(per_quarter=4, drop=drop)
    arm, b0 = perfect_vs_shift(events)
    acc = tables.evaluate_arm("S", arm, b0, events, resamples=R)["accuracy"]
    assert acc["quarters_below"] == count and acc["passed"] is passed


def test_ticker_rule_fails_alone_when_quarters_pass():
    # Per event, arm pinball minus B0 pinball = -0.5 * gain (mean of 1 - a over LV is 0.5).
    # AAA and BBB (gain +0.5) give -0.25 each; CCC (gain -0.8) gives +0.4.
    # Every quarter holds two of each ticker, so each quarter's mean is (-0.25 - 0.25 + 0.4) / 3 < 0.
    # Ticker means are -0.25, -0.25, +0.4 with 20 events each; the bootstrap stat is (-0.75 + 0.65c) / 3,
    # where c is the number of CCC draws among three, so its upper 95th percentile is positive.
    events = make_events(per_quarter=6)
    y = events["y"].to_numpy(float)
    gain = np.where(events["ticker"].to_numpy() == "CCC", -0.8, 0.5)
    arm = qframe(events.index, np.repeat((y + 1.0 - gain)[:, None], 7, axis=1))
    b0 = qframe(events.index, np.repeat((y + 1.0)[:, None], 7, axis=1))
    acc = tables.evaluate_arm("S", arm, b0, events, resamples=R)["accuracy"]
    assert all(acc["periods_below"].values())
    assert acc["quarters_below"] >= 8
    assert acc["quarter_interval"][1] < 0
    assert acc["ticker_interval"][1] > 0
    assert acc["passed"] is False


def test_intervals_match_an_independent_bootstrap():
    events = make_events(per_quarter=4)
    y = events["y"].to_numpy(float)
    wiggle = (np.arange(len(y)) % 5 - 2) * 0.3
    arm = qframe(events.index, y[:, None] + wiggle[:, None] + OFF)
    b0 = qframe(events.index, y[:, None] + 1.0 + np.zeros(7))
    res = tables.evaluate_arm("S", arm, b0, events, resamples=R)
    arm_d = np.array([np.mean(r) for r in pin_loops(y, arm[QCOLS].to_numpy(float))])
    b0_d = np.array([np.mean(r) for r in pin_loops(y, b0[QCOLS].to_numpy(float))])
    d = arm_d - b0_d
    assert res["accuracy"]["quarter_interval"] == pytest.approx(oracle_interval(d, quarter_of(events)))
    assert res["accuracy"]["ticker_interval"] == pytest.approx(oracle_interval(d, events["ticker"].to_numpy()))


# ---------- f. calibration (rule 3) ----------

def test_calibration_counts_each_band_independently():
    cal = calib_result([BOTH] * 5 + [NONE] * 5)
    assert cal["quarters_no_worse"] == {"p25_p75": 5, "p10_p90": 5} and cal["passed"] is True
    cal = calib_result([BOTH] * 4 + [P25] + [NONE] * 5)
    assert cal["quarters_no_worse"] == {"p25_p75": 5, "p10_p90": 4} and cal["passed"] is False
    cal = calib_result([BOTH] * 4 + [P10] + [NONE] * 5)
    assert cal["quarters_no_worse"] == {"p25_p75": 4, "p10_p90": 5} and cal["passed"] is False


def test_five_quarters_pass_and_four_fail():
    assert calib_result([BOTH] * 5 + [NONE] * 5)["passed"] is True
    assert calib_result([BOTH] * 4 + [NONE] * 6)["passed"] is False


def test_tie_on_the_other_side_of_nominal_counts_as_no_worse():
    # B0 covers 0.7 of p25-p75 (distance 0.2). An arm covering 0.3 is the same distance on the other side.
    cal = calib_result([TIE] * 10, b0_spec=(7, 0))
    assert cal["quarters_no_worse"] == {"p25_p75": 10, "p10_p90": 0} and cal["passed"] is False


# ---------- g. one ticker ----------

def test_one_ticker_population_names_the_bootstrap_refusal():
    events = make_events(per_quarter=4)
    events["ticker"] = "AAA"
    arm, b0 = perfect_vs_shift(events)
    res = tables.evaluate_arm("S", arm, b0, events, resamples=R)
    assert res["status"] == "not_testable"
    assert res["not_testable_reasons"] == ["bootstrap_too_few_groups"]
    assert np.isnan(res["accuracy"]["ticker_interval"]).all()
    assert not np.isnan(res["accuracy"]["quarter_interval"]).any()
    assert res["accuracy"]["passed"] is None and res["calibration"]["passed"] is None


# ---------- h. validation refusals ----------

def _edit(frame, row, column, value):
    out = frame.copy()
    out.loc[row, column] = value
    return out


# Each case maps (events, arm, b0) to (arm, b0, events, raw).
CASES = {
    "events missing a column": lambda e, a, b: (a, b, e.drop(columns="m3"), None),
    "duplicate arm index": lambda e, a, b: (pd.concat([a, a.iloc[:1]]), b, e, None),
    "duplicate events index": lambda e, a, b: (a, b, pd.concat([e, e.iloc[:1]]), None),
    "arm missing reason": lambda e, a, b: (a.drop(columns="reason"), b, e, None),
    "b0 missing a quantile": lambda e, a, b: (a, b.drop(columns="q50"), e, None),
    "arm id not in events": lambda e, a, b: (a, b, e.iloc[1:], None),
    "null event_date": lambda e, a, b: (a, b, _edit(e, e.index[0], "event_date", pd.NaT), None),
    "null in-period ticker": lambda e, a, b: (a, b, _edit(e, e.index[0], "ticker", None), None),
    "negative in-period y": lambda e, a, b: (a, b, _edit(e, e.index[0], "y", -1.0), None),
    "scored row with nan quantile": lambda e, a, b: (_edit(a, e.index[0], "q50", np.nan), b, e, None),
    "raw missing a paired id": lambda e, a, b: (a, b, e, a[QCOLS].iloc[1:]),
    "raw with a nan value": lambda e, a, b: (a, b, e, _edit(a[QCOLS], e.index[0], "q25", np.nan)),
    "b0 duplicate index": lambda e, a, b: (a, pd.concat([b, b.iloc[:1]]), e, None),
    "raw duplicate index": lambda e, a, b: (a, b, e, pd.concat([a[QCOLS], a[QCOLS].iloc[:1]])),
    "b0 id not in events": lambda e, a, b: (a, b.rename(index={e.index[0]: "ghost"}), e, None),
    "in-period inf y": lambda e, a, b: (a, b, _edit(e, e.index[0], "y", np.inf), None),
    "scored arm row with inf quantile": lambda e, a, b: (_edit(a, e.index[0], "q50", np.inf), b, e, None),
    "scored b0 row with nan quantile": lambda e, a, b: (a, _edit(b, e.index[0], "q50", np.nan), e, None),
    "raw with an inf value": lambda e, a, b: (a, b, e, _edit(a[QCOLS], e.index[0], "q25", np.inf)),
    "raw missing a quantile column": lambda e, a, b: (a, b, e, a[QCOLS].drop(columns="q50")),
    "arm missing a quantile column": lambda e, a, b: (a.drop(columns="q50"), b, e, None),
    "b0 missing reason": lambda e, a, b: (a, b.drop(columns="reason"), e, None),
}


MESSAGES = {
    "events missing a column": "events, arm, b0 and raw need their columns",
    "duplicate arm index": "indexes must be unique",
    "duplicate events index": "indexes must be unique",
    "arm missing reason": "events, arm, b0 and raw need their columns",
    "b0 missing a quantile": "events, arm, b0 and raw need their columns",
    "arm id not in events": "arm and b0 ids must all be in events",
    "null event_date": "event_date must not be null",
    "null in-period ticker": "in-period events need a ticker and non-negative, non-infinite y",
    "negative in-period y": "in-period events need a ticker and non-negative, non-infinite y",
    "scored row with nan quantile": "scored rows need seven finite quantiles",
    "raw missing a paired id": "raw needs seven finite values for every paired event",
    "raw with a nan value": "raw needs seven finite values for every paired event",
    "b0 duplicate index": "indexes must be unique",
    "raw duplicate index": "indexes must be unique",
    "b0 id not in events": "arm and b0 ids must all be in events",
    "in-period inf y": "in-period events need a ticker and non-negative, non-infinite y",
    "scored arm row with inf quantile": "scored rows need seven finite quantiles",
    "scored b0 row with nan quantile": "scored rows need seven finite quantiles",
    "raw with an inf value": "raw needs seven finite values for every paired event",
    "raw missing a quantile column": "events, arm, b0 and raw need their columns",
    "arm missing a quantile column": "events, arm, b0 and raw need their columns",
    "b0 missing reason": "events, arm, b0 and raw need their columns",
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_each_refusal_raises_value_error(case):
    events = make_events(per_quarter=4)
    arm, b0 = perfect_vs_shift(events)
    a, b, e, raw = CASES[case](events, arm, b0)
    with pytest.raises(ValueError, match=re.escape(MESSAGES[case])):
        tables.evaluate_arm("S", a, b, e, raw=raw, resamples=R)


def _population(kind):
    """Four events per quarter in the ten quarters ("multi"), or one ticker in one quarter ("one_group")."""
    if kind == "multi":
        events = make_events(per_quarter=4)
    else:
        events = make_events(per_quarter=4, drop=tuple(QUARTERS[1:]))
        events["ticker"] = "AAA"
    arm, b0 = perfect_vs_shift(events)
    return events, arm, b0


POPULATIONS = ["multi", "one_group"]


@pytest.mark.parametrize("kind", POPULATIONS)
@pytest.mark.parametrize("resamples", [0, -1, 2.5, True])
def test_bad_resamples_is_refused(kind, resamples):
    events, arm, b0 = _population(kind)
    with pytest.raises(ValueError, match=re.escape("resamples must be an integer >= 1")):
        tables.evaluate_arm("S", arm, b0, events, resamples=resamples)


@pytest.mark.parametrize("kind", POPULATIONS)
@pytest.mark.parametrize("seed", [-1, None, 1.5, True])
def test_bad_seed_is_refused(kind, seed):
    events, arm, b0 = _population(kind)
    with pytest.raises(ValueError, match=re.escape("seed must be a non-negative integer")):
        tables.evaluate_arm("S", arm, b0, events, resamples=R, seed=seed)


def test_numpy_integer_controls_are_accepted():
    events, arm, b0 = _population("multi")
    base = tables.evaluate_arm("S", arm, b0, events, resamples=R, seed=3)
    got = tables.evaluate_arm("S", arm, b0, events, resamples=np.int64(R), seed=np.int64(3))
    assert got["status"] == "testable"
    assert got["accuracy"]["quarter_interval"] == pytest.approx(base["accuracy"]["quarter_interval"])
    assert got["accuracy"]["ticker_interval"] == pytest.approx(base["accuracy"]["ticker_interval"])


@pytest.mark.parametrize("where", ["m0", "m3", "crps"])
def test_non_numeric_value_is_refused_before_metrics(monkeypatch, where):
    def boom(*args, **kwargs):
        raise AssertionError("metrics ran before validation")

    monkeypatch.setattr(tables, "pinball", boom)
    events, arm, b0 = _population("multi")
    first = events.index[0]
    if where == "crps":
        arm = _edit(arm.assign(crps=0.5).astype({"crps": object}), first, "crps", "abc")
    else:
        events = _edit(events.astype({where: object}), first, where, "abc")
    with pytest.raises(ValueError):
        tables.evaluate_arm("S", arm, b0, events, resamples=R)


# ---------- i. memory ----------

def test_memory_error_becomes_resource_limit_exceeded(monkeypatch):
    def boom(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(tables, "pinball", boom)
    events = make_events(per_quarter=4)
    arm, b0 = perfect_vs_shift(events)
    with pytest.raises(OpsError) as exc:
        tables.evaluate_arm("S", arm, b0, events, resamples=R)
    assert exc.value.problem.code == "RESOURCE_LIMIT_EXCEEDED"


# ---------- j. the constant matches the spec ----------

def test_min_scored_share_matches_spec_yaml():
    text = (ROOT / "experiments" / "EXP-190_size_quantile_move" / "spec.yaml").read_text(encoding="utf-8")
    assert tables.MIN_SCORED_SHARE == yaml.safe_load(text)["stage_a"]["min_scored_share"]


# ---------- k. walking skeleton: real B0 and real calibrate ----------

def skeleton():
    """2000 donors dated 2022-01 to 2023-06 and 65 targets from 2023-07, with real B0 and real calibrate."""
    i = np.arange(2000)
    d_pred = 6.0 + 0.001 * i
    d_date = pd.Timestamp("2022-01-01") + pd.to_timedelta(i % 540, unit="D")
    d_y = d_pred + ((i * 37) % 101) / 10 - 5.0
    j = np.arange(65)
    t_pred = np.where(j % 7 == 0, np.nan, 2.0 + 0.01 * j)  # every seventh target has no forecast
    t_date = pd.Timestamp("2023-07-01") + pd.to_timedelta(15 * j, unit="D")
    t_y = 2.0 + 0.01 * j + ((j * 13) % 21) / 10 - 1.0
    ids = [f"d{k:04d}" for k in i] + [f"t{k:03d}" for k in j]
    dates = pd.DatetimeIndex(np.concatenate([d_date.to_numpy(), t_date.to_numpy()]))
    pred = np.concatenate([d_pred, t_pred])
    y = np.concatenate([d_y, t_y])
    events = pd.DataFrame({"event_date": dates, "ticker": ["DNR"] * 2000 + [TICKERS[k % 3] for k in j],
                           "y": y, "m0": pred, "m3": pred + 0.1}, index=ids)
    table = pd.DataFrame({"event_date": d_date, "pred_move": d_pred, "pred_crush": 0.0}, index=ids[:2000])
    available = (d_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d").to_numpy()
    labels = pd.DataFrame({"abs_move": d_y, "crush_pct_iv30": 0.0, "available": available}, index=ids[:2000])
    targets = pd.DataFrame({"event_date": dates, "decision_date": dates - pd.Timedelta(days=1), "pred_move": pred},
                           index=ids)
    b0 = pool_quantiles(table, labels, targets)
    label_end = (dates + pd.Timedelta(days=5)).strftime("%Y-%m-%d").to_numpy()
    qhat = pd.DataFrame(pred[:, None] + OFFS, index=ids, columns=QCOLS)
    arms, _, _ = calibrate(qhat, pd.DataFrame({"event_date": dates, "y": y, "label_end": label_end}, index=ids))
    return events, arms["Q"], b0


def test_walking_skeleton_paired_count_is_the_reason_null_intersection():
    events, arm, b0 = skeleton()
    res = tables.evaluate_arm("Q", arm, b0, events, resamples=R)
    in_period = pd.to_datetime(events["event_date"]).between("2023-07-01", "2025-12-31").to_numpy()
    both_scored = arm["reason"].isna().to_numpy() & b0["reason"].isna().to_numpy()
    expected = int((in_period & both_scored).sum())
    assert expected > 0
    assert int(res["share"].set_index("period").loc["pooled", "paired"]) == expected
    assert int(row_of(res["tables"], "pooled", "pooled")["n"]) == expected


# ---------- l. result shape, inputs unchanged, seed and resamples reach the bootstrap ----------

@pytest.mark.parametrize("case", ["testable", "not_testable"])
def test_result_has_no_combined_verdict_and_inputs_are_unchanged(case):
    if case == "testable":
        events, arm, b0 = _population("multi")
        raw = arm[QCOLS].copy()
    else:
        events, arm, b0, raw = tiny_frames()
    before = [f.copy(deep=True) for f in (events, arm, b0, raw)]
    res = tables.evaluate_arm("S", arm, b0, events, raw=raw, resamples=R)
    assert res["status"] == case
    assert set(res) == {"arm", "status", "not_testable_reasons", "share", "drops", "paired_ids", "tables",
                        "accuracy", "calibration"}
    for frame, saved in zip((events, arm, b0, raw), before):
        pd.testing.assert_frame_equal(frame, saved)


def test_resamples_and_seed_reach_the_bootstrap(monkeypatch):
    calls = []
    original = tables.bootstrap_interval

    def recording(*args, **kwargs):
        calls.append((kwargs["resamples"], kwargs["seed"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(tables, "bootstrap_interval", recording)
    events, arm, b0 = _population("multi")
    res = tables.evaluate_arm("S", arm, b0, events, resamples=37, seed=7)
    assert res["status"] == "testable"
    assert calls == [(37, 7), (37, 7)]


def test_seed_changes_the_intervals_on_a_noisy_population():
    events = make_events(per_quarter=4)
    y = events["y"].to_numpy(float)
    wiggle = (np.arange(len(y)) % 5 - 2) * 0.3
    arm = qframe(events.index, y[:, None] + wiggle[:, None] + OFF)
    b0 = qframe(events.index, y[:, None] + 1.0 + np.zeros(7))
    assert SEED != 7
    default = tables.evaluate_arm("S", arm, b0, events, resamples=R)["accuracy"]
    seeded = tables.evaluate_arm("S", arm, b0, events, resamples=R, seed=7)["accuracy"]
    assert any(not np.array_equal(seeded[k], default[k]) for k in ("quarter_interval", "ticker_interval"))


# ---------- stdout stays empty ----------

def test_logging_goes_to_stderr_only(capsys):
    events = make_events(per_quarter=4)
    arm, b0 = perfect_vs_shift(events)
    tables.evaluate_arm("S", arm, b0, events, resamples=R)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[quantile arm tables S] admitted=40 paired=40 status=testable reasons=[]" in captured.err
    assert "[quantile arm tables S] accuracy passed=True quarters_below=10" in captured.err
