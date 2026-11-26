"""Tests for EXP-190 Q5b part 2 (experiments/size_quantile_arm_verdict.py).

Synthetic fixtures only. Expected values come from plain loops, scipy, np.polyfit and a hand-written bootstrap,
never from the module's helpers.
"""
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata, spearmanr

from engine.v2.ops.errors import OpsError
from experiments import size_quantile_arm_verdict as verdict_mod
from experiments.size_pnl_contrast import RESAMPLES, SEED
from experiments.size_quantile_arm_verdict import judge_arm, stage_a_verdict

QCOLS = ["q10", "q25", "q40", "q50", "q60", "q75", "q90"]
QUARTERS = ["2023Q3", "2023Q4"] + [f"2024Q{k}" for k in range(1, 5)] + [f"2025Q{k}" for k in range(1, 5)]
TICKERS = ["AAA", "BBB", "CCC"]
R = 200
N = 40
G = np.array([4 * (i // 4) + 3 - i % 4 for i in range(N)])  # block-reversed permutation of 0..39, rho about 0.98
P = (7 * np.arange(N)) % N  # another permutation of 0..39 (gcd(7, 40) = 1)
E = 0.05 * (np.arange(N) + 1)  # distinct, increasing e within a quarter
W_BASE = 0.5 + 0.02 * G
HQ = np.linspace(0.5, 1.5, 600)  # history with several q50 bins
HY = HQ + 0.3 * np.sin(np.arange(600))
CONST_Y = 0.5 + (np.arange(600) % 37) * 0.05


def period_of(qu):
    return "2023H2" if qu.startswith("2023") else qu[:4]


def hist(q50, y, labels):
    return pd.DataFrame({"q50": q50, "y": y, "label_available": pd.to_datetime(labels)})


def const_hist(labels=None):
    """Constant q50, so one L bin: L is the IQR of CONST_Y for every event."""
    return hist(np.full(600, 1.0), CONST_Y, ["2022-12-01"] * 600 if labels is None else labels)


def arm_frame(ids, q50, W):
    cols = [q50 - W / 2 - 0.3, q50 - W / 2, q50 - W / 10, q50, q50 + W / 10, q50 + W / 2, q50 + W / 2 + 0.3]
    out = pd.DataFrame(np.column_stack(cols), index=ids, columns=QCOLS)
    out["reason"] = None
    return out


def b0_frame(ids, y, bucket):
    out = pd.DataFrame(np.repeat((y + 100.0)[:, None], 7, axis=1), index=ids, columns=QCOLS)  # far from y
    out["reason"] = None
    out["bucket"] = bucket
    return out


def panel(q50_of=None, w_of=None, bucket_of=None, y_of=None, tickers=TICKERS):
    """40 events per quarter, all dated in the quarter's first month. Arm q50 and W per event; y = q50 + E."""
    rows = []
    for qi, qu in enumerate(QUARTERS):
        start = pd.Period(qu, freq="Q").start_time
        for i in range(N):
            q50 = 1.0 + 0.05 * qi if q50_of is None else q50_of(qu, i)
            rows.append({"id": f"{qu}-{i:02d}", "event_date": start + pd.Timedelta(days=i % 28),
                         "ticker": tickers[(i + qi) % len(tickers)],
                         "y": q50 + E[i] if y_of is None else y_of(qu, i, q50), "m0": 1.5, "m3": 2.0,
                         "q50": q50, "W": W_BASE[i] if w_of is None else w_of(qu, i),
                         "bucket": 2.0 if bucket_of is None else bucket_of(qu, i)})
    df = pd.DataFrame(rows).set_index("id")
    events = df[["event_date", "ticker", "y", "m0", "m3"]]
    arm = arm_frame(df.index, df["q50"].to_numpy(float), df["W"].to_numpy(float))
    b0 = b0_frame(df.index, df["y"].to_numpy(float), df["bucket"].to_numpy(float))
    return events, arm, b0


def arrays(events, arm):
    qu = np.array([i[:6] for i in events.index])
    ids = np.asarray(events.index)
    y = events["y"].to_numpy(float)
    q = arm[QCOLS].to_numpy(float)
    return qu, ids, y, q, q[:, 5] - q[:, 1], np.abs(y - q[:, 3])


def oracle_partial(score, e, q50):
    """Partial Spearman by plain rank residualisation on the rank of q50 (np.polyfit); centring if q50 is constant."""
    rs, re, rq = rankdata(score), rankdata(e), rankdata(q50)
    if np.ptp(rq) == 0:
        return float(np.corrcoef(rs - rs.mean(), re - re.mean())[0, 1])
    rs = rs - np.polyval(np.polyfit(rq, rs, 1), rq)
    re = re - np.polyval(np.polyfit(rq, re, 1), rq)
    return float(np.corrcoef(rs, re)[0, 1])


def oracle_contrib(score, e, q50):
    """Per-event contributions n * rs * re / sqrt(sum rs^2 * sum re^2) for one cell."""
    rs, re, rq = rankdata(score), rankdata(e), rankdata(q50)
    if np.ptp(rq) == 0:
        rs, re = rs - rs.mean(), re - re.mean()
    else:
        rs = rs - np.polyval(np.polyfit(rq, rs, 1), rq)
        re = re - np.polyval(np.polyfit(rq, re, 1), rq)
    return len(score) * rs * re / np.sqrt(np.sum(rs ** 2) * np.sum(re ** 2))


def tercile_labels(score, ids):
    """Thirds of the (score, id) order by np.array_split, labels 0 (bottom) to 2 (top)."""
    lab = np.empty(len(score), int)
    order = np.array(sorted(range(len(score)), key=lambda i: (score[i], ids[i])))
    for k, part in enumerate(np.array_split(order, 3)):
        lab[part] = k
    return lab


def oracle_boot(d, groups, resamples=R, seed=SEED):
    """Ticker bootstrap written here: sorted groups, one draw of (resamples, g) indices, 5th and 95th percentiles."""
    keys = sorted(set(groups))
    sums = np.array([d[groups == k].sum() for k in keys])
    counts = np.array([(groups == k).sum() for k in keys])
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(resamples, len(keys)))
    stats = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return tuple(np.percentile(stats, [5, 95]))


def oracle_L(hq, hy, cur):
    """Level-only width by plain loops: equal-count bins of earlier q50, edge values to the lower bin."""
    edges = np.unique(np.quantile(hq, np.linspace(0, 1, min(20, len(hq) // 100) + 1)))[1:-1]
    bin_of = [sum(edge < x for edge in edges) for x in hq]
    out = []
    for x in cur:
        b = sum(edge < x for edge in edges)
        vals = [hy[j] for j in range(len(hq)) if bin_of[j] == b]
        out.append(np.quantile(vals, 0.75) - np.quantile(vals, 0.25))
    return np.array(out)


def expected_terciles(per, lab, y, e, q10, q25, q75, q90):
    out = {}
    for p in ("2023H2", "2024", "2025"):
        for k in range(3):
            m = (per == p) & (lab == k)
            if m.any():
                yy = y[m]
                out[(p, k)] = {"n": int(m.sum()), "mean_e": e[m].mean(),
                               "iqr_y": np.quantile(yy, 0.75) - np.quantile(yy, 0.25),
                               "cover_p25_p75": np.mean((yy >= q25[m]) & (yy <= q75[m])),
                               "cover_p10_p90": np.mean((yy >= q10[m]) & (yy <= q90[m]))}
    return out


def assert_terciles(got, exp):
    got = got.set_index(["period", "tercile"])
    assert len(got) == len(exp)
    for key, row in exp.items():
        for col, v in row.items():
            assert got.loc[key, col] == pytest.approx(v)


def oracle_resid(score, q50):
    """Residual of rank(score) on rank(q50) with an intercept (polyfit); centred if q50 is constant."""
    rs, rq = rankdata(score), rankdata(q50)
    if np.ptp(rq) == 0:
        return rs - rs.mean()
    return rs - np.polyval(np.polyfit(rq, rs, 1), rq)


def oracle_terc(score, q50, ids):
    """Tercile labels of the within-cell residual rank, ties broken by id (cell_scores orders by residual)."""
    return tercile_labels(oracle_resid(score, q50), ids)


PERIOD_MEMBERS = (("2023H2", ["2023Q3", "2023Q4"]), ("2024", [f"2024Q{k}" for k in range(1, 5)]),
                  ("2025", [f"2025Q{k}" for k in range(1, 5)]))
ROW_COLS = ["quarter", "period", "ticker", "bucket", "y", *QCOLS, *[f"b0_{c}" for c in QCOLS], "W", "L", "e",
            "level_reason", "zero_median", "used", "W_contribution", "W_tercile", "L_contribution", "L_tercile"]


def weighted_panel():
    """Like panel(), but 2024Q1 holds two B0 cells of 50 and 30 events. W and e are distinct within each quarter."""
    rows = []
    for qi, qu in enumerate(QUARTERS):
        start = pd.Period(qu, freq="Q").start_time
        layout = [(2.0, 50), (3.0, 30)] if qu == "2024Q1" else [(2.0, N)]
        j = 0
        for bucket, count in layout:
            for _ in range(count):
                q50 = 1.0 + 0.05 * qi
                rows.append({"id": f"{qu}-{j:02d}", "event_date": start + pd.Timedelta(days=j % 28),
                             "ticker": TICKERS[(j + qi) % len(TICKERS)], "y": q50 + 0.01 * ((11 * j) % 80 + 1),
                             "m0": 1.5, "m3": 2.0, "q50": q50, "W": 0.5 + 0.01 * ((7 * j) % 80),
                             "bucket": bucket})
                j += 1
    df = pd.DataFrame(rows).set_index("id")
    events = df[["event_date", "ticker", "y", "m0", "m3"]]
    arm = arm_frame(df.index, df["q50"].to_numpy(float), df["W"].to_numpy(float))
    b0 = b0_frame(df.index, df["y"].to_numpy(float), df["bucket"].to_numpy(float))
    return events, arm, b0


def zero_median_fixture():
    """q50 varies inside each cell; two 2024Q1 events have q50 0 (zero median). Level width varies with q50."""
    def q50_of(qu, i):
        return 0.0 if (qu == "2024Q1" and i < 2) else 1.0 + P[i] / 128

    def y_of(qu, i, q50):
        return E[i] if q50 == 0.0 else q50 + E[i]

    events, arm, b0 = panel(q50_of=q50_of, y_of=y_of)
    res = judge_arm("Q", arm, b0, events, hist(HQ, HY, ["2022-12-01"] * 600), resamples=R)
    return events, arm, b0, res


# ---------- 1. walking skeleton ----------

def test_walking_skeleton_passes_all_rules_with_oracle_values():
    events, arm, b0 = panel()
    res = judge_arm("Q", arm, b0, events, const_hist(), resamples=R)
    assert res["status"] == "testable" and res["not_testable_reasons"] == []
    assert res["verdict"] == "pass"
    cond = res["conditional"]
    assert cond["passed"] is True
    assert res["accuracy"]["passed"] is True and res["calibration"]["passed"] is True

    qu, ids, y, q, W, e = arrays(events, arm)
    per = np.array([period_of(x) for x in qu])
    rho = {x: spearmanr(W[qu == x], e[qu == x])[0] for x in QUARTERS}
    lab = np.empty(len(W), int)
    for x in QUARTERS:
        lab[qu == x] = tercile_labels(W[qu == x], ids[qu == x])
    diff = {x: e[(qu == x) & (lab == 2)].mean() - e[(qu == x) & (lab == 0)].mean() for x in QUARTERS}
    periods = cond["periods"].set_index("period")
    for p, members in (("2023H2", ["2023Q3", "2023Q4"]), ("2024", [f"2024Q{k}" for k in range(1, 5)]),
                       ("2025", [f"2025Q{k}" for k in range(1, 5)])):
        assert periods.loc[p, "W_spearman"] == pytest.approx(np.mean([rho[x] for x in members]))
        assert periods.loc[p, "W_tercile_diff"] == pytest.approx(np.mean([diff[x] for x in members]))
        assert periods.loc[p, "L_spearman"] == 0 and periods.loc[p, "L_tercile_diff"] == 0
    assert periods.loc["2024", "W_spearman"] < 0.999  # a non-perfect relation, so the values differ from 1

    iqr = np.quantile(CONST_Y, 0.75) - np.quantile(CONST_Y, 0.25)  # const_hist() uses CONST_Y as history y
    assert np.allclose(cond["rows"]["L"].to_numpy(float), iqr)
    arm_q = arm.loc[events.index]
    assert_terciles(cond["terciles"], expected_terciles(
        per, lab, y, e, arm_q["q10"].to_numpy(), arm_q["q25"].to_numpy(), arm_q["q75"].to_numpy(),
        arm_q["q90"].to_numpy()))


# ---------- 2. q50 varies inside a cell ----------

def test_partial_spearman_residualises_q50_ranks_with_polyfit():
    events, arm, b0 = panel(q50_of=lambda qu, i: 1.0 + 0.01 * P[i])
    res = judge_arm("Q", arm, b0, events, const_hist(), resamples=R)
    qu, _, _, q, W, e = arrays(events, arm)
    rho = {x: oracle_partial(W[qu == x], e[qu == x], q[qu == x, 3]) for x in QUARTERS}
    periods = res["conditional"]["periods"].set_index("period")
    for p, members in (("2023H2", ["2023Q3", "2023Q4"]), ("2024", [f"2024Q{k}" for k in range(1, 5)]),
                       ("2025", [f"2025Q{k}" for k in range(1, 5)])):
        assert periods.loc[p, "W_spearman"] == pytest.approx(np.mean([rho[x] for x in members]))


def test_period_values_weight_cells_by_n_not_per_cell_means():
    events, arm, b0 = weighted_panel()
    res = judge_arm("Q", arm, b0, events, const_hist(), resamples=R)
    qu, ids, y, q, W, e = arrays(events, arm)
    bucket = b0["bucket"].to_numpy(float)
    cells = [("2024Q1", 2.0), ("2024Q1", 3.0), ("2024Q2", 2.0), ("2024Q3", 2.0), ("2024Q4", 2.0)]
    n, rho, dif = [], [], []
    for x, b in cells:
        m = (qu == x) & (bucket == b)
        lab = tercile_labels(W[m], ids[m])
        n.append(int(m.sum()))
        rho.append(oracle_partial(W[m], e[m], q[m, 3]))
        dif.append(e[m][lab == 2].mean() - e[m][lab == 0].mean())
    assert n[0] == 50 and n[1] == 30
    w_rho, w_dif = np.average(rho, weights=n), np.average(dif, weights=n)
    u_rho, u_dif = np.mean(rho), np.mean(dif)
    periods = res["conditional"]["periods"].set_index("period")
    assert periods.loc["2024", "W_spearman"] == pytest.approx(w_rho)
    assert periods.loc["2024", "W_tercile_diff"] == pytest.approx(w_dif)
    assert u_rho != pytest.approx(w_rho) and u_dif != pytest.approx(w_dif)


# ---------- 3. rule 2 negative controls ----------

def test_negative_control_w_anticorrelated_in_one_period_fails_only_rule_two():
    def w_of(qu, i):
        return 0.5 + 0.02 * (N - 1 - G[i]) if qu.startswith("2024") else W_BASE[i]

    events, arm, b0 = panel(w_of=w_of)
    res = judge_arm("Q", arm, b0, events, const_hist(), resamples=R)
    periods = res["conditional"]["periods"].set_index("period")
    assert periods.loc["2024", "W_spearman"] < 0
    assert periods.loc["2023H2", "W_spearman"] > 0 and periods.loc["2025", "W_spearman"] > 0
    assert res["conditional"]["passed"] is False
    assert res["accuracy"]["passed"] is True and res["calibration"]["passed"] is True
    assert res["verdict"] == "fail"


def test_negative_control_l_matching_w_in_rank_fails_by_strictness():
    hist_df = hist(HQ, HY, ["2022-12-01"] * 600)
    q50_of = lambda qu, i: 1.0 + P[i] / 128  # noqa: E731  exact binary fractions
    cur = np.array([q50_of(qu, i) for qu in QUARTERS for i in range(N)])
    l_all = oracle_L(HQ, HY, cur)
    w_all = 1.0 + rankdata(l_all, method="dense") / 64  # exact, and ordered like L with the same ties
    events, arm, b0 = panel(q50_of=q50_of, w_of=lambda qu, i: w_all[QUARTERS.index(qu) * N + i])
    res = judge_arm("Q", arm, b0, events, hist_df, resamples=R)
    arm_q = arm.loc[events.index]
    assert np.array_equal(arm_q["q75"].to_numpy(float) - arm_q["q25"].to_numpy(float), w_all)
    assert np.allclose(res["conditional"]["rows"]["L"].to_numpy(float), l_all)
    periods = res["conditional"]["periods"].set_index("period")
    for p in ("2023H2", "2024", "2025"):
        assert periods.loc[p, "W_spearman"] == pytest.approx(periods.loc[p, "L_spearman"])
        assert periods.loc[p, "W_tercile_diff"] == pytest.approx(periods.loc[p, "L_tercile_diff"])
    assert res["conditional"]["passed"] is False
    assert res["accuracy"]["passed"] is True and res["calibration"]["passed"] is True
    assert res["verdict"] == "fail"


# ---------- 4. pooled intervals ----------

def test_pooled_intervals_match_a_hand_written_ticker_bootstrap():
    events, arm, b0 = panel()
    res = judge_arm("Q", arm, b0, events, const_hist(), resamples=R)
    qu, _, _, q, W, e = arrays(events, arm)
    d = np.concatenate([oracle_contrib(W[qu == x], e[qu == x], q[qu == x, 3]) for x in QUARTERS])
    groups = events["ticker"].to_numpy()
    cond = res["conditional"]
    assert cond["W_interval"] == pytest.approx(oracle_boot(d, groups))
    assert cond["diff_interval"] == pytest.approx(oracle_boot(d, groups))  # L contributes zero here


def test_pooled_intervals_with_nonzero_l_contributions():
    hist_df = hist(HQ, HY, ["2022-12-01"] * 600)  # 600 earlier labels: no burn-in
    q50_of = lambda qu, i: 1.0 + P[i] / 128  # noqa: E731  q50 varies inside each cell
    events, arm, b0 = panel(q50_of=q50_of, bucket_of=lambda qu, i: float(QUARTERS.index(qu)))  # one cell per quarter
    res = judge_arm("Q", arm, b0, events, hist_df, resamples=R)
    qu, _, _, q, W, e = arrays(events, arm)
    cur = q[:, 3]
    L = oracle_L(HQ, HY, cur)
    assert np.allclose(res["conditional"]["rows"]["L"].to_numpy(float), L)
    dW = np.concatenate([oracle_contrib(W[qu == x], e[qu == x], cur[qu == x]) for x in QUARTERS])
    dL = np.concatenate([oracle_contrib(L[qu == x], e[qu == x], cur[qu == x]) for x in QUARTERS])
    groups = events["ticker"].to_numpy()
    cond = res["conditional"]
    assert cond["W_interval"] == pytest.approx(oracle_boot(dW, groups))
    assert cond["diff_interval"] == pytest.approx(oracle_boot(dW - dL, groups))
    assert not np.allclose(np.array(cond["diff_interval"]), np.array(cond["W_interval"]))
    periods = cond["periods"].set_index("period")
    l_rho = {x: oracle_partial(L[qu == x], e[qu == x], cur[qu == x]) for x in QUARTERS}
    for p, members in (("2023H2", ["2023Q3", "2023Q4"]), ("2024", [f"2024Q{k}" for k in range(1, 5)]),
                       ("2025", [f"2025Q{k}" for k in range(1, 5)])):
        assert periods.loc[p, "L_spearman"] == pytest.approx(np.mean([l_rho[x] for x in members]))
    assert any(abs(periods.loc[p, "L_spearman"]) > 1e-6 for p in ("2023H2", "2024", "2025"))


def test_every_period_passing_with_a_pooled_low_end_at_or_below_zero_fails(monkeypatch):
    def fake(*args, **kwargs):
        return (0.1, -0.01, 0.5)  # only the pooled interval is made to straddle zero

    monkeypatch.setattr(verdict_mod, "bootstrap_interval", fake)
    events, arm, b0 = panel()
    res = judge_arm("Q", arm, b0, events, const_hist(), resamples=R)
    periods = res["conditional"]["periods"]
    assert (periods[["W_spearman", "W_tercile_diff"]] > 0).all().all()
    assert res["conditional"]["passed"] is False and res["verdict"] == "fail"


def test_seed_reaches_the_pooled_bootstrap():
    events, arm, b0 = panel()
    qu, _, _, q, W, e = arrays(events, arm)
    d = np.concatenate([oracle_contrib(W[qu == x], e[qu == x], q[qu == x, 3]) for x in QUARTERS])
    # Tickers by within-quarter index i: sort the 40 positions by mean contribution, cut into 8 unequal blocks.
    d_by_i = d.reshape(len(QUARTERS), N).mean(axis=0)
    tick_of_i = np.empty(N, dtype=object)
    block = np.repeat(np.arange(8), [2, 3, 4, 5, 6, 6, 7, 7])
    tick_of_i[np.argsort(d_by_i, kind="stable")] = [f"T{k}" for k in block]
    events = events.copy()
    events["ticker"] = np.tile(tick_of_i, len(QUARTERS))
    groups = events["ticker"].to_numpy()
    res7 = judge_arm("Q", arm, b0, events, const_hist(), resamples=R, seed=7)
    res188 = judge_arm("Q", arm, b0, events, const_hist(), resamples=R)
    w7, w188 = res7["conditional"]["W_interval"], res188["conditional"]["W_interval"]
    assert w7 == pytest.approx(oracle_boot(d, groups, seed=7))
    assert w188 == pytest.approx(oracle_boot(d, groups))
    assert not np.allclose(np.array(w7), np.array(w188))


# ---------- 5. level-width burn-in ----------

def test_burn_in_removes_events_from_conditional_tests_only():
    events, arm, b0 = panel()
    labels = ["2022-12-01"] * 400 + ["2023-09-01"] * 200  # July 2023 has only 400 earlier labels
    res = judge_arm("Q", arm, b0, events, const_hist(labels), resamples=R)
    rows = res["conditional"]["rows"]
    assert (rows.loc[rows["quarter"] == "2023Q3", "level_reason"] == "level_width_burn_in").all()
    assert (rows.loc[rows["quarter"] == "2023Q4", "level_reason"] == "").all()
    assert res["conditional"]["skips"]["level_width_burn_in"] == 40
    assert res["tables"].set_index(["slice", "label"]).loc[("pooled", "pooled"), "n"] == 400
    assert res["status"] == "testable"


def test_period_with_all_events_burned_in_is_not_testable():
    events, arm, b0 = panel()
    labels = ["2022-12-01"] * 400 + ["2024-01-01"] * 200  # no label before July 2023 besides the first 400
    res = judge_arm("Q", arm, b0, events, const_hist(labels), resamples=R)
    assert "2023H2:no_usable_cell" in res["not_testable_reasons"]
    assert res["status"] == "not_testable" and res["verdict"] == "not_testable"
    assert res["conditional"]["passed"] is None
    assert res["accuracy"]["passed"] is None and res["calibration"]["passed"] is None


# ---------- 6. causality of the month start ----------

@pytest.mark.parametrize("early, burned", [(450, 40), (500, 0)])
def test_label_on_the_month_start_is_not_used(early, burned):
    events, arm, b0 = panel()
    labels = ["2022-12-01"] * early + ["2023-07-01"] * (600 - early)
    res = judge_arm("Q", arm, b0, events, const_hist(labels), resamples=R)
    assert res["conditional"]["skips"]["level_width_burn_in"] == burned


# ---------- 7. cell skips and one-ticker population ----------

def test_small_and_constant_e_cells_are_counted_in_skips():
    events, arm, b0 = panel(
        bucket_of=lambda qu, i: 3.0 if (qu == "2024Q1" and i < 10) else (4.0 if qu == "2024Q2" else 2.0),
        y_of=lambda qu, i, q50: q50 + 0.5 if qu == "2024Q2" else q50 + E[i])
    res = judge_arm("Q", arm, b0, events, const_hist(), resamples=R)
    skips = res["conditional"]["skips"]
    assert (skips["cells_cell_small"], skips["events_cell_small"]) == (1, 10)
    assert (skips["cells_constant_e"], skips["events_constant_e"]) == (1, 40)
    assert (skips["cells_zero_e_residual"], skips["events_zero_e_residual"]) == (0, 0)


def test_one_ticker_population_names_the_conditional_bootstrap_refusal():
    events, arm, b0 = panel(tickers=["AAA"])
    res = judge_arm("Q", arm, b0, events, const_hist(), resamples=R)
    assert "conditional_bootstrap_too_few_groups" in res["not_testable_reasons"]
    assert res["conditional"]["passed"] is None and res["verdict"] == "not_testable"
    assert np.isnan(res["conditional"]["W_interval"]).all()


# ---------- 8. zero q50 ----------

def test_zero_median_is_excluded_from_ratio_only_and_warns_nothing():
    base_q50 = lambda qu, i: 1.0 + 0.05 * QUARTERS.index(qu)  # noqa: E731
    events, arm, b0 = panel(q50_of=lambda qu, i: 0.0 if (qu == "2024Q1" and i < 2) else base_q50(qu, i),
                            y_of=lambda qu, i, q50: E[i] if q50 == 0.0 else q50 + E[i])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = judge_arm("Q", arm, b0, events, const_hist(), resamples=R)
    cond = res["conditional"]
    assert cond["skips"]["zero_median"] == 2
    qu, _, _, q, W, e = arrays(events, arm)
    ratio_mean = {}
    for x in QUARTERS:
        m = (qu == x) & (q[:, 3] > 1e-9)
        ratio_mean[x] = (m.sum(), oracle_partial(W[m] / q[m, 3], e[m], q[m, 3]))
    weighted = sum(ratio_mean[x][0] * ratio_mean[x][1] for x in ("2024Q1", "2024Q2", "2024Q3", "2024Q4"))
    assert cond["ratio"]["periods"].set_index("period").loc["2024", "spearman"] == pytest.approx(
        weighted / sum(ratio_mean[x][0] for x in ("2024Q1", "2024Q2", "2024Q3", "2024Q4")))
    rho_w = [oracle_partial(W[qu == x], e[qu == x], q[qu == x, 3]) for x in ("2024Q1", "2024Q2", "2024Q3", "2024Q4")]
    assert cond["periods"].set_index("period").loc["2024", "W_spearman"] == pytest.approx(np.mean(rho_w))


def test_ratio_periods_interval_and_terciles_match_oracles():
    events, arm, b0, res = zero_median_fixture()
    qu, ids, y, q, W, e = arrays(events, arm)
    cur = q[:, 3]
    zero = cur <= 1e-9
    keep = ~zero
    ratio = W / np.where(zero, 1.0, cur)
    ratio_n, ratio_rho, ratio_dif = {}, {}, {}
    lab, d = np.full(len(W), -1), np.full(len(W), np.nan)
    for x in QUARTERS:
        m = (qu == x) & keep
        ratio_n[x] = int(m.sum())
        ratio_rho[x] = oracle_partial(ratio[m], e[m], cur[m])
        lab[m] = oracle_terc(ratio[m], cur[m], ids[m])
        d[m] = oracle_contrib(ratio[m], e[m], cur[m])
        ratio_dif[x] = e[m][lab[m] == 2].mean() - e[m][lab[m] == 0].mean()
    cond = res["conditional"]
    rp = cond["ratio"]["periods"].set_index("period")
    for p, members in PERIOD_MEMBERS:
        ns = [ratio_n[x] for x in members]
        assert rp.loc[p, "spearman"] == pytest.approx(np.average([ratio_rho[x] for x in members], weights=ns))
        assert rp.loc[p, "tercile_diff"] == pytest.approx(np.average([ratio_dif[x] for x in members], weights=ns))
    assert cond["ratio"]["interval"] == pytest.approx(oracle_boot(d[keep], events["ticker"].to_numpy()[keep]))
    per = np.array([period_of(x) for x in qu])
    assert_terciles(cond["ratio"]["terciles"], expected_terciles(
        per[keep], lab[keep], y[keep], e[keep], q[keep, 0], q[keep, 1], q[keep, 5], q[keep, 6]))
    assert rp.loc["2024", "spearman"] != pytest.approx(cond["periods"].set_index("period").loc["2024", "W_spearman"])


def test_raw_rows_match_oracles():
    events, arm, b0, res = zero_median_fixture()
    qu, ids, y, q, W, e = arrays(events, arm)
    cur = q[:, 3]
    zero = cur <= 1e-9
    L = oracle_L(HQ, HY, cur)
    n = len(W)
    wc, wt, lc, lt = (np.full(n, np.nan) for _ in range(4))
    for x in QUARTERS:
        m = qu == x
        wc[m] = oracle_contrib(W[m], e[m], cur[m])
        wt[m] = oracle_terc(W[m], cur[m], ids[m])
        lc[m] = oracle_contrib(L[m], e[m], cur[m])
        lt[m] = oracle_terc(L[m], cur[m], ids[m])
    rows = res["conditional"]["rows"]
    assert list(rows.columns) == ROW_COLS
    assert rows.index.tolist() == list(res["paired_ids"])
    assert rows["quarter"].tolist() == qu.tolist()
    assert rows["period"].tolist() == [period_of(x) for x in qu]
    assert rows["ticker"].tolist() == events["ticker"].tolist()
    np.testing.assert_allclose(rows["bucket"].to_numpy(float), b0["bucket"].to_numpy(float))
    np.testing.assert_allclose(rows["y"].to_numpy(float), y)
    np.testing.assert_allclose(rows[QCOLS].to_numpy(float), q)
    np.testing.assert_allclose(rows[[f"b0_{c}" for c in QCOLS]].to_numpy(float), b0[QCOLS].to_numpy(float))
    np.testing.assert_allclose(rows["W"].to_numpy(float), W)
    np.testing.assert_allclose(rows["L"].to_numpy(float), L)
    np.testing.assert_allclose(rows["e"].to_numpy(float), e)
    assert rows["zero_median"].tolist() == zero.tolist()
    assert rows["used"].all() and (rows["level_reason"] == "").all()
    np.testing.assert_allclose(rows["W_contribution"].to_numpy(float), wc)
    np.testing.assert_allclose(rows["W_tercile"].to_numpy(float), wt)
    np.testing.assert_allclose(rows["L_contribution"].to_numpy(float), lc)
    np.testing.assert_allclose(rows["L_tercile"].to_numpy(float), lt)


# ---------- 9. refusals ----------

@pytest.mark.parametrize("case, message", [
    ("history_list", "history must be a DataFrame with q50, y and label_available columns"),
    ("history_missing", "history must be a DataFrame with q50, y and label_available columns"),
    ("b0_no_bucket", "b0 must be a DataFrame with a bucket column"),
])
def test_input_refusals_come_before_evaluate_arm(monkeypatch, case, message):
    def boom(*args, **kwargs):
        raise AssertionError("evaluate_arm ran before validation")

    monkeypatch.setattr(verdict_mod, "evaluate_arm", boom)
    events, arm, b0 = panel()
    history, b = const_hist(), b0
    if case == "history_list":
        history = [[1.0]]
    elif case == "history_missing":
        history = history.drop(columns="label_available")
    else:
        b = b0.drop(columns="bucket")
    with pytest.raises(ValueError, match=message):
        judge_arm("Q", arm, b, events, history, resamples=R)


def test_nan_bucket_on_a_paired_event_is_refused():
    events, arm, b0 = panel()
    b0.loc[events.index[0], "bucket"] = np.nan
    with pytest.raises(ValueError, match="paired events need a finite B0 bucket"):
        judge_arm("Q", arm, b0, events, const_hist(), resamples=R)


def test_memory_error_becomes_resource_limit_exceeded(monkeypatch):
    def boom(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(verdict_mod, "level_width", boom)
    events, arm, b0 = panel()
    with pytest.raises(OpsError) as exc:
        judge_arm("Q", arm, b0, events, const_hist(), resamples=R)
    assert exc.value.problem.code == "RESOURCE_LIMIT_EXCEEDED"


def test_seed_and_resamples_defaults_are_the_spec_values():
    params = inspect.signature(judge_arm).parameters
    assert params["seed"].default == 188 == SEED
    assert params["resamples"].default == RESAMPLES


# ---------- 10. verdict only for Q and Q_raw ----------

@pytest.mark.parametrize("name", ["B1", "S"])
def test_other_arm_names_get_no_verdict(name):
    events, arm, b0 = panel()
    res = judge_arm(name, arm, b0, events, const_hist(), resamples=R)
    assert res["verdict"] is None
    assert res["conditional"]["passed"] is True


def test_q_raw_run_gets_a_pass_verdict():
    events, arm, b0 = panel()
    res = judge_arm("Q_raw", arm, b0, events, const_hist(), resamples=R)
    assert res["verdict"] == "pass"


# ---------- 11. Stage A truth table ----------

def jr(arm, verdict, reasons=()):
    return {"arm": arm, "verdict": verdict, "not_testable_reasons": list(reasons)}


@pytest.mark.parametrize("q, q_raw, status, arms, reasons, stage_b", [
    (jr("Q", "pass"), jr("Q_raw", "pass"), "passed", {"Q": "pass", "Q_raw": "pass"}, [], ["Q", "Q_raw"]),
    (jr("Q", "pass"), jr("Q_raw", "fail"), "passed", {"Q": "pass", "Q_raw": "fail"}, [], ["Q"]),
    (jr("Q", "fail"), jr("Q_raw", "pass"), "passed", {"Q": "fail", "Q_raw": "pass"}, [], ["Q_raw"]),
    (jr("Q", "fail"), jr("Q_raw", "fail"), "not_passed", {"Q": "fail", "Q_raw": "fail"}, [], []),
    (jr("Q", "fail"), jr("Q_raw", "not_testable", ["2024:period_no_paired_events"]), "not_passed",
     {"Q": "fail", "Q_raw": "not_testable"}, ["Q_raw:2024:period_no_paired_events"], []),
    (None, jr("Q_raw", "pass"), "passed", {"Q": "not_testable", "Q_raw": "pass"}, ["Q:calibration_failed"], ["Q_raw"]),
    (None, jr("Q_raw", "fail"), "not_passed", {"Q": "not_testable", "Q_raw": "fail"}, ["Q:calibration_failed"], []),
    (jr("Q", "pass"), None, "not_testable", {"Q": "pass", "Q_raw": "not_testable"}, ["Q_raw:fit_failed"], []),
    (jr("Q", "pass"), jr("Q_raw", "not_testable", ["2024:period_no_paired_events"]), "passed",
     {"Q": "pass", "Q_raw": "not_testable"}, ["Q_raw:2024:period_no_paired_events"], ["Q"]),
    (jr("Q", "not_testable", ["2023H2:no_usable_cell"]), jr("Q_raw", "pass"), "passed",
     {"Q": "not_testable", "Q_raw": "pass"}, ["Q:2023H2:no_usable_cell"], ["Q_raw"]),
])
def test_stage_a_truth_table(q, q_raw, status, arms, reasons, stage_b):
    out = stage_a_verdict(q, q_raw)
    assert out == {"status": status, "reasons": reasons, "arms": arms, "stage_b_arms": stage_b}


@pytest.mark.parametrize("q, q_raw", [
    (jr("Q_raw", "pass"), jr("Q_raw", "pass")),
    (jr("Q", "pass"), jr("Q", "pass")),
    (jr("Q", None), jr("Q_raw", "pass")),
])
def test_stage_a_refuses_wrong_arm_names_and_missing_verdicts(q, q_raw):
    with pytest.raises(ValueError, match="must be a judge_arm result"):
        stage_a_verdict(q, q_raw)
