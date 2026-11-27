"""EXP-190 Q3b calibration tests: synthetic deterministic fixtures, plain-numpy oracles, no data reads.

Fixture counts are fixed by construction (see the comments on each builder), so they can be checked by hand.
"""
import numpy as np
import pandas as pd
import pytest

from engine.v2.ops.errors import OpsError
from experiments import size_quantile_calibration as calib
from experiments.size_quantile_metrics import crps_quantile

LEVELS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
T_MONTH = pd.Timestamp("2020-02-01")
N_B, N_NEG = 500, 100
NEG = [-12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0]
# Offsets cycle with i % 3: a wide, a narrow and an exactly-0.5 spread (all multiples of 1/8, so exact in binary).
OFF_HI = ([0, .25, .5, .75, 1.0, 1.25, 1.5],
          [0, .125, .25, .375, .5, .625, .75],
          [0, .125, .25, .375, .5, .625, .875])
OFF_LO = ([0, .5, 1.0, 1.5, 2.0, 2.5, 3.0],
          [0, .125, .25, .375, .5, .625, .75],
          [0, .125, .25, .5, .625, .75, .875])
T_ROWS = (([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], 1.0),
          ([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.5], 4.0),
          ([-3.0, -2.5, -2.0, -1.0, 0.0, 1.0, 2.0], 0.0),
          ([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0], 2.5))
U_ROWS = (([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], 3.0),
          ([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 1.0))


def test_memory_error_inside_calibrate_becomes_resource_limit_exceeded(monkeypatch):
    def boom(*args, **kwargs):
        raise MemoryError

    # "d" (2020-01-10) is in the 12-month window of "a" (2020-02-15), so available_before is reached.
    monkeypatch.setattr(calib, "available_before", boom)
    qhat = [[float(k) for k in range(7)]] * 2
    predictions = pd.DataFrame(qhat, index=["d", "a"], columns=calib.QCOLS)
    events = pd.DataFrame({"event_date": ["2020-01-10", "2020-02-15"], "y": [1.0, np.nan],
                           "label_end": ["2020-01-20", None]}, index=["d", "a"])
    with pytest.raises(OpsError) as exc:
        calib.calibrate(predictions, events)
    assert exc.value.problem.code == "RESOURCE_LIMIT_EXCEEDED"


# ---------- helpers: builders and independent oracles ----------

def _qrow(i):
    """Sorted, positive, row-varying prediction row (multiples of 1/8)."""
    base = 1.0 + 0.125 * (i % 8)
    return [base + k for k in range(7)]


def _shape(i, offsets):
    base = 8.0 + 0.125 * (i % 8)
    return [base + o for o in offsets[i % 3]]


def _chain_rows(n_hi=50, n_lo=50, later=False):
    """Rows (id, event_date, label_end, qhat, y).

    B: 500 constant rows in Jun 2019 with y = v + 1, so every calibration error is exactly 1 and the first
       calibrated month (S, Jan 2020) has Q = qhat + 1 exactly (the 500 calibration rows come from B alone).
    S: n_hi rows with y = Q90 + 1 (donors above), n_lo rows with y = Q10 / 2 (donors below), and 100
       negative-prediction rows with y = 0 (never donors: y is not strictly beyond their floored Q10 or Q90).
       S's tails are thin (B has no Q quantiles), so S is unscored but its Q quantiles donate to T.
    T: the scored target (Feb 2020), with negative predictions. U (optional, Mar 2020) is for causality.
    """
    rows = []
    for i in range(N_B):
        v = 1.0 + 0.5 * (i % 11)
        rows.append((f"b{i:03d}", f"2019-06-{1 + i % 28:02d}", "2019-07-05", [v] * 7, v + 1.0))
    for i in range(n_hi):
        q = _shape(i, OFF_HI)
        rows.append((f"s_hi{i:03d}", "2020-01-05", "2020-01-10", q, q[6] + 2.0))
    for i in range(n_lo):
        q = _shape(i, OFF_LO)
        rows.append((f"s_lo{i:03d}", "2020-01-05", "2020-01-10", q, (q[0] + 1.0) / 2.0))
    for i in range(N_NEG):
        rows.append((f"s_neg{i:03d}", "2020-01-05", "2020-01-10", list(NEG), 0.0))
    for j, (q, y) in enumerate(T_ROWS):
        rows.append((f"t{j}", "2020-02-05", "2020-02-10", list(q), y))
    if later:
        for j, (q, y) in enumerate(U_ROWS):
            rows.append((f"u{j}", "2020-03-05", "2020-03-10", list(q), y))
    return rows


def _frames(rows):
    ids = [r[0] for r in rows]
    preds = pd.DataFrame([r[3] for r in rows], index=ids, columns=calib.QCOLS)
    events = pd.DataFrame({"event_date": [r[1] for r in rows], "y": [r[4] for r in rows],
                           "label_end": [r[2] for r in rows]}, index=ids)
    return preds, events


def _run(n_hi=50, n_lo=50, later=False):
    rows = _chain_rows(n_hi, n_lo, later)
    arms, pools, report = calib.calibrate(*_frames(rows))
    return rows, arms, pools, report


def _pick(rows, prefix):
    sel = [r for r in rows if r[0].startswith(prefix)]
    return [r[0] for r in sel], np.array([r[3] for r in sel], float), np.array([r[4] for r in sel], float)


def _oracle_delta(err):
    return np.array([np.quantile(err[:, k], LEVELS[k]) for k in range(7)])


def _oracle_q(qhat, calib_qhat, calib_y):
    """Q for prediction rows: signed errors on the unfloored sorted calibration rows, then shift, sort, floor."""
    qhat = np.sort(np.asarray(qhat, float), axis=1)
    err = np.asarray(calib_y, float)[:, None] - np.sort(np.asarray(calib_qhat, float), axis=1)
    return np.maximum(np.sort(qhat + _oracle_delta(err), axis=1), 0.0)


def _oracle_pools(q, y):
    """(d_lo, d_hi) from donor quantiles q (floored, sorted) and realised y: strict tails, width floor 0.5."""
    hi = [(yy - qq[6]) / max(qq[6] - qq[3], 0.5) for qq, yy in zip(q, y) if yy > qq[6]]
    lo = [(yy - qq[0]) / max(qq[3] - qq[0], 0.5) for qq, yy in zip(q, y) if yy < qq[0]]
    return np.sort(np.array(lo, float)), np.sort(np.array(hi, float))


def _eval(u, x, t):
    """Value at t of the piecewise-linear curve through (u, x), using the first non-degenerate segment."""
    for i in range(len(u) - 1):
        if u[i] <= t <= u[i + 1] and u[i + 1] > u[i]:
            return x[i] + (t - u[i]) / (u[i + 1] - u[i]) * (x[i + 1] - x[i])
    raise AssertionError(f"no segment contains {t}")


def _dense_crps(y, q, d_lo, d_hi, n=400001):
    """CRPS = 2 * integral of (1{Q(u) > y} - u)(Q(u) - y) du for the true floored inverse CDF, on a fine grid."""
    q, d_lo, d_hi = (np.asarray(v, float) for v in (q, d_lo, d_hi))
    u = np.linspace(0.0, 1.0, n)
    s_lo, s_hi = max(q[3] - q[0], 0.5), max(q[6] - q[3], 0.5)
    lower = np.maximum(0.0, q[0] + s_lo * np.quantile(d_lo, np.clip(u / 0.1, 0.0, 1.0)))
    upper = q[6] + s_hi * np.quantile(d_hi, np.clip((u - 0.9) / 0.1, 0.0, 1.0))
    middle = np.interp(u, LEVELS, q)
    quantile = np.where(u <= 0.1, lower, np.where(u >= 0.9, upper, middle))
    f = 2.0 * ((quantile > y).astype(float) - u) * (quantile - y)
    h = u[1] - u[0]
    return h * (f.sum() - 0.5 * (f[0] + f[-1]))


# ---------- 1. predictions_from_receipts ----------

def _receipt(configuration, ids, raw, status="ok", levels=LEVELS):
    return {"status": status, "levels": list(levels), "configuration": list(configuration), "event_ids": ids,
            "raw": {f"{a}x{b}": m for (a, b), m in raw.items()}}


def test_predictions_from_receipts_uses_each_years_own_configuration_sorted_and_unfloored():
    chosen_2020 = [[1.5, -2.0, 0.5, 3.0, -1.0, 2.0, 0.0], [4.0, 3.5, -3.0, 0.25, 1.0, -0.5, 2.5],
                   [-0.75, 1.25, 0.0, -2.5, 3.0, 0.5, -1.5]]
    chosen_2021 = [[-1.0, -4.0, 2.0, 0.0, 1.0, -2.0, 3.0], [0.5, -0.5, 2.5, -3.0, 1.5, 0.0, -1.0]]
    receipts = {
        2021: _receipt((1, 1), ["b1", "b2"], {(1, 1): chosen_2021,
                                                (2, 2): (np.array(chosen_2021) + 20.0).tolist()}),
        2020: _receipt((2, 2), ["a1", "a2", "a3"], {(2, 2): chosen_2020,
                                                      (1, 1): (np.array(chosen_2020) + 10.0).tolist()}),
    }
    out = calib.predictions_from_receipts(receipts)
    expected = np.vstack([np.sort(np.array(chosen_2020), axis=1), np.sort(np.array(chosen_2021), axis=1)])
    assert list(out.index) == ["a1", "a2", "a3", "b1", "b2"]
    assert list(out.columns) == calib.QCOLS
    np.testing.assert_array_equal(out.to_numpy(), expected)
    assert (out.to_numpy() < 0).any()


def test_predictions_from_receipts_refuses_bad_receipts():
    raw = {(1, 1): [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]}
    with pytest.raises(ValueError):
        calib.predictions_from_receipts({})
    with pytest.raises(ValueError):
        calib.predictions_from_receipts({2020: _receipt((1, 1), ["a1"], raw, status="fit_failed")})
    with pytest.raises(ValueError):
        calib.predictions_from_receipts({2020: _receipt((1, 1), ["a1"], raw,
                                                        levels=[0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.85])})
    with pytest.raises(ValueError):
        calib.predictions_from_receipts({2020: _receipt((1, 1), ["a1"], raw),
                                         2021: _receipt((1, 1), ["a1"], raw)})


# ---------- 2. Q delta oracle with negative qhat ----------

def test_q_delta_uses_unfloored_errors_then_sorts_then_floors():
    rows, arms, _, _ = _run()
    b_ids, qb, yb = _pick(rows, "b")
    _, qs, ys = _pick(rows, "s")
    t_ids, qt, _ = _pick(rows, "t")
    calib_q, calib_y = np.vstack([qb, qs]), np.concatenate([yb, ys])
    expected = _oracle_q(qt, calib_q, calib_y)
    np.testing.assert_allclose(arms["Q"].loc[t_ids, calib.QCOLS].to_numpy(), expected, rtol=0, atol=1e-9)
    assert arms["Q"].loc[t_ids, "reason"].isna().all()
    assert (arms["Q"].loc[t_ids, "month"] == T_MONTH).all()
    # Flooring the predictions before taking errors gives a different answer (the 100 negative rows dominate).
    delta_floor = _oracle_delta(calib_y[:, None] - np.maximum(np.sort(calib_q, axis=1), 0.0))
    floor_first = np.maximum(np.sort(qt + delta_floor, axis=1), 0.0)
    assert np.abs(floor_first - expected).max() > 1e-3


# ---------- 3. window and label availability at the 499/500 boundary ----------

@pytest.mark.parametrize("extra_date, extra_label, expected", [
    ("2020-03-01", "2020-03-10", None),                  # exactly m - 12 months: in pool, 500 rows: calibrated
    ("2020-02-29", "2020-03-10", "calibration_burn_in"),  # one day before the window: out, 499 rows
    ("2020-06-01", "2021-02-27", None),                  # label_end + 1 = 2021-02-28 < m: available, 500 rows
    ("2020-06-01", "2021-02-28", "calibration_burn_in"),  # label_end + 1 = m: not available, 499 rows
    ("2020-06-01", "2021-03-01", "calibration_burn_in"),  # label_end = m: not available, 499 rows
    ("2021-03-01", "2021-02-27", "calibration_burn_in"),  # dated in month m: never in its own pool, 499 rows
])
def test_pool_window_and_label_availability_at_the_499_500_boundary(extra_date, extra_label, expected):
    rows = [(f"c{i:03d}", "2020-03-02", "2020-03-10", _qrow(i), 1.0) for i in range(499)]
    rows.append(("x", extra_date, extra_label, _qrow(0), 1.0))
    rows.append(("t", "2021-03-15", "2021-03-20", _qrow(0), 1.0))
    arms, pools, _ = calib.calibrate(*_frames(rows))
    # The window and label rules decide the calibration rows: 500 gives reason None, 499 is burn-in.
    got = arms["Q"].loc["t", "reason"]
    if expected is None:  # a scored row's reason is NaN in pandas, not None
        assert pd.isna(got)
    else:
        assert got == expected
    # Every donor has y = 1.0, below its own Q90, so the month has no upper-tail donor: thin (Q3). The quantiles keep,
    # and only the pool (CRPS) is absent.
    assert ("Q", pd.Timestamp("2021-03-01")) not in pools


# ---------- 4. causality ----------

def test_later_labels_do_not_change_earlier_quantiles_or_pools():
    rows, arms, pools, _ = _run(later=True)
    new_y = {"t0": 0.0, "t1": 0.2, "t2": 9.0, "t3": 4.0, "u0": 0.5, "u1": 6.0}
    moved = [(i, d, lab, q, new_y.get(i, y)) for i, d, lab, q, y in rows]
    arms2, pools2, _ = calib.calibrate(*_frames(moved))
    early = [r[0] for r in rows if not r[0].startswith("u")]
    for arm in ("Q", "Q_raw"):
        pd.testing.assert_frame_equal(arms2[arm].loc[early], arms[arm].loc[early])

    def keep(key):
        return key[1] <= T_MONTH

    assert {k for k in pools if keep(k)} == {k for k in pools2 if keep(k)}
    for key in pools:
        if keep(key):
            np.testing.assert_array_equal(pools2[key][0], pools[key][0])
            np.testing.assert_array_equal(pools2[key][1], pools[key][1])


# ---------- 5. tail thinness and thin donors ----------

@pytest.mark.parametrize("n_hi, n_lo, pooled", [(29, 50, False), (50, 29, False), (30, 30, True)])
def test_thin_tail_below_30_donors_keeps_quantiles_and_drops_only_the_pool(n_hi, n_lo, pooled):
    rows, arms, pools, _ = _run(n_hi, n_lo)
    b_ids, s_ids, t_ids = _pick(rows, "b")[0], _pick(rows, "s")[0], _pick(rows, "t")[0]
    assert (arms["Q"].loc[b_ids, "reason"] == "calibration_burn_in").all()
    assert arms["Q"].loc[s_ids, "reason"].isna().all()  # S is thin, is still scored, and its quantiles donate
    got = arms["Q"].loc[t_ids]
    assert got[calib.QCOLS].notna().all().all() and got["reason"].isna().all()  # thin or not: quantiles kept
    if pooled:
        assert len(pools[("Q", T_MONTH)][0]) == n_lo and len(pools[("Q", T_MONTH)][1]) == n_hi
    else:
        assert ("Q", T_MONTH) not in pools


def test_thin_months_are_logged_once_per_arm_not_per_event(capsys):
    rows = _chain_rows(29, 50)
    calib.calibrate(*_frames(rows))
    err = capsys.readouterr().err
    # Q's thin months: 2020-01 (S, donors all burn-in) and 2020-02 (T). Two, counted, not one line per event.
    q_lines = [line for line in err.splitlines() if "arm=Q events=" in line]
    assert len(q_lines) == 1
    assert q_lines[0].endswith("thin_months=2")
    assert "tail_pool_thin" not in err


# ---------- 6. donor pools: oracle per arm, strict tails, width floor ----------

def test_q_arm_donor_pools_match_oracle_with_width_floor():
    rows, _, pools, _ = _run()
    _, qb, yb = _pick(rows, "b")
    _, qs, ys = _pick(rows, "s")
    d_lo, d_hi = _oracle_pools(_oracle_q(qs, qb, yb), ys)
    got_lo, got_hi = pools[("Q", T_MONTH)]
    np.testing.assert_allclose(got_lo, d_lo, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got_hi, d_hi, rtol=1e-12, atol=1e-12)
    assert len(d_hi) == 50 and len(d_lo) == 50
    # Narrow (0.375) and exactly-0.5 hi donors both divide by the 0.5 floor: 17 + 16 = 33 donors at 1 / 0.5 = 2.
    assert np.isclose(got_hi, 2.0).sum() == 33


def _raw_rows():
    """200 pool rows (Jun 2019) with fixed donor kinds, then 3 targets (Jan 2020).

    60 hi rows (y above Q90), 60 lo rows (y below Q10), 40 mid rows and 20 + 20 edge rows (y exactly at
    Q90 or Q10, never donors). Predictions include negatives (floored for Q_raw) and narrow spreads (floored).
    """
    rng = np.random.default_rng(190)
    kinds = np.array(["hi"] * 60 + ["lo"] * 60 + ["mid"] * 40 + ["edge90"] * 20 + ["edge10"] * 20)
    n = len(kinds)
    base = rng.uniform(-1.0, 4.0, n)
    base[kinds == "lo"] = rng.uniform(1.0, 4.0, int((kinds == "lo").sum()))
    spread = np.where(np.arange(n) % 4 == 0, 0.05, 1.0)
    qhat = base[:, None] + spread[:, None] * np.sort(rng.uniform(0.0, 1.0, (n, 7)), axis=1)
    qr = np.maximum(qhat, 0.0)
    y = np.zeros(n)
    hi, lo, mid = kinds == "hi", kinds == "lo", kinds == "mid"
    y[hi] = qr[hi, 6] + rng.uniform(0.1, 5.0, int(hi.sum()))
    y[lo] = qr[lo, 0] * rng.uniform(0.0, 0.9, int(lo.sum()))
    y[mid] = qr[mid, 3]
    y[kinds == "edge90"] = qr[kinds == "edge90", 6]
    y[kinds == "edge10"] = qr[kinds == "edge10", 0]
    rows = [(f"p{i:03d}", "2019-06-15", "2019-07-01", qhat[i].tolist(), float(y[i])) for i in range(n)]
    targets = (([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 2.0), ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 3.0),
               ([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], 0.2))
    rows += [(f"t{j}", "2020-01-10", "2020-01-15", q, yy) for j, (q, yy) in enumerate(targets)]
    return rows


def test_q_raw_donor_pools_match_oracle_and_q_stays_burned_in():
    rows = _raw_rows()
    arms, pools, _ = calib.calibrate(*_frames(rows))
    p = [r for r in rows if r[0].startswith("p")]
    d_lo, d_hi = _oracle_pools(np.maximum(np.array([r[3] for r in p]), 0.0), np.array([r[4] for r in p]))
    assert len(d_lo) == 60 and len(d_hi) == 60
    got_lo, got_hi = pools[("Q_raw", pd.Timestamp("2020-01-01"))]
    np.testing.assert_allclose(got_lo, d_lo, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got_hi, d_hi, rtol=1e-12, atol=1e-12)
    assert ("Q", pd.Timestamp("2020-01-01")) not in pools
    assert ("Q_raw", pd.Timestamp("2019-06-01")) not in pools
    assert (arms["Q"].loc[["t0", "t1", "t2"], "reason"] == "calibration_burn_in").all()


# ---------- 7. quantile_points ----------

Q_CROSS = [2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
D_LO_CROSS = [-10.0, -3.0, -0.5]
D_HI_CROSS = [0.5, 1.0, 3.0]
Q_WIDE = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0]
D_LO_WIDE = [-4.0, -2.5, -1.0, -0.25]
D_HI_WIDE = [0.5, 1.0, 2.0, 4.0, 7.0]


def test_zero_crossing_adds_exact_breakpoint_and_floors_the_function():
    u, x = calib.quantile_points(Q_CROSS, D_LO_CROSS, D_HI_CROSS)
    # Lower line at nodes 0, .05, .1 is 2 + 1 * (-10, -3, -0.5) = (-8, -1, 1.5): zero crossing at u = .07.
    exp_u = [0.0, 0.05, 0.07, 0.1, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.9, 0.95, 1.0]
    exp_x = [0.0, 0.0, 0.0, 1.5, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5, 5.25, 6.0, 9.0]
    np.testing.assert_allclose(u, exp_u, rtol=0, atol=1e-12)
    np.testing.assert_allclose(x, exp_x, rtol=0, atol=1e-12)
    assert np.all(np.diff(u) >= 0) and np.all(np.diff(x) >= 0)


def test_crps_of_flat_quantiles_with_width_floor_matches_hand_value():
    # Q(u) = 9.5 on [0, 0.1), 10 on [0.1, 0.9], 10.5 on (0.9, 1]. For y = 10:
    # CRPS = 2 * (int_0^0.1 0.5u du + int_0.9^1 0.5(1-u) du) = 2 * (0.0025 + 0.0025) = 0.01.
    u, x = calib.quantile_points([10.0] * 7, [-1.0, -1.0], [1.0, 1.0])
    assert crps_quantile(10.0, u, x) == pytest.approx(0.01, abs=1e-12)


@pytest.mark.parametrize("y", [0.0, 1.0, 2.2, 3.7, 9.5])
def test_points_give_the_floored_crps_of_the_true_function(y):
    u, x = calib.quantile_points(Q_CROSS, D_LO_CROSS, D_HI_CROSS)
    assert crps_quantile(y, u, x) == pytest.approx(_dense_crps(y, Q_CROSS, D_LO_CROSS, D_HI_CROSS), abs=1e-4)


def test_points_endpoints_knots_and_jump_duplicates_without_crossing():
    u, x = calib.quantile_points(Q_WIDE, D_LO_WIDE, D_HI_WIDE)
    assert u[0] == 0.0 and u[-1] == 1.0
    assert np.all(np.diff(u) >= 0) and np.all(np.diff(x) >= 0)
    for level, value in zip(LEVELS, Q_WIDE):
        assert np.any((u == level) & (x == value))  # every knot is present, exactly
    assert np.sum(u == 0.1) == 2 and np.sum(u == 0.9) == 2  # jumps keep the tail end point and the knot
    # Lower nodes are exactly q10 + s_lo * donors, upper nodes q90 + s_hi * donors.
    np.testing.assert_allclose(x[:len(D_LO_WIDE)], Q_WIDE[0] + (Q_WIDE[3] - Q_WIDE[0]) * np.array(D_LO_WIDE))
    upper = x[-len(D_HI_WIDE):]
    np.testing.assert_allclose(upper, Q_WIDE[6] + (Q_WIDE[6] - Q_WIDE[3]) * np.array(D_HI_WIDE))


def test_points_match_np_quantile_at_interior_levels_in_both_tails():
    q, d_lo, d_hi = np.array(Q_WIDE), np.array(D_LO_WIDE), np.array(D_HI_WIDE)
    u, x = calib.quantile_points(q, d_lo, d_hi)
    s_lo, s_hi = q[3] - q[0], q[6] - q[3]
    for t in (0.02, 0.05, 0.093):
        assert _eval(u, x, t) == pytest.approx(q[0] + s_lo * np.quantile(d_lo, t / 0.1), abs=1e-9)
    for t in (0.93, 0.97):
        assert _eval(u, x, t) == pytest.approx(q[6] + s_hi * np.quantile(d_hi, (t - 0.9) / 0.1), abs=1e-9)


def test_event_floor_of_half_applies_when_event_widths_are_zero():
    u, x = calib.quantile_points([5.0] * 7, [-3.0, -2.0, -1.0], [1.0, 2.0, 3.0])
    exp_u = [0.0, 0.05, 0.1] + LEVELS + [0.9, 0.95, 1.0]
    exp_x = [3.5, 4.0, 4.5] + [5.0] * 7 + [5.5, 6.0, 6.5]  # 5 + 0.5 * donor
    np.testing.assert_allclose(u, exp_u, rtol=0, atol=1e-12)
    np.testing.assert_allclose(x, exp_x, rtol=0, atol=1e-12)


def test_quantile_points_refuses_fewer_than_two_donors_per_side():
    with pytest.raises(ValueError):
        calib.quantile_points([1.0] * 7, [-1.0], [1.0, 2.0])


# ---------- 8. non-finite rows, refusals, floor report, logging ----------

def test_non_finite_prediction_is_flagged_never_counted_and_takes_precedence():
    rows = [(f"c{i:03d}", "2020-03-02", "2020-03-10", _qrow(i), 1.0) for i in range(499)]
    rows.append(("nan", "2020-03-02", "2020-03-10", [np.nan] * 7, 1.0))
    rows.append(("t", "2021-03-15", "2021-03-20", _qrow(0), 1.0))
    arms, _, _ = calib.calibrate(*_frames(rows))
    for arm in ("Q", "Q_raw"):
        assert arms[arm].loc["nan", "reason"] == "non_finite_output"
        assert arms[arm].loc["nan", calib.QCOLS].isna().all()
    # The NaN row has a finite y and an available label, but it is not one of the 500 calibration rows.
    assert arms["Q"].loc["t", "reason"] == "calibration_burn_in"


def test_extra_events_outside_the_predictions_are_ignored():
    preds = pd.DataFrame([_qrow(0)], index=["a"], columns=calib.QCOLS)
    events = pd.DataFrame({"event_date": ["2020-03-01", None], "y": [1.0, -1.0],
                           "label_end": ["2020-03-05", None]}, index=["a", "zzz"])
    arms, _, _ = calib.calibrate(preds, events)
    assert arms["Q"].loc["a", "reason"] == "calibration_burn_in"


@pytest.mark.parametrize("pred_ids, ev_ids, dates, labels, y", [
    (["a", "a"], ["a"], ["2020-01-10"], ["2020-01-20"], [1.0]),              # duplicate prediction id
    (["a"], ["a", "a"], ["2020-01-10"] * 2, ["2020-01-20"] * 2, [1.0, 1.0]),  # duplicate event id
    (["a", "z"], ["a"], ["2020-01-10"], ["2020-01-20"], [1.0]),              # prediction id missing from events
    (["a"], ["a"], [None], ["2020-01-20"], [1.0]),                           # null event_date
    (["a"], ["a"], ["2020-01-10"], ["2020-01-20"], [np.inf]),                # infinite y
    (["a"], ["a"], ["2020-01-10"], ["2020-01-20"], [-1.0]),                  # negative y
    (["a"], ["a"], ["2020-01-10"], [None], [1.0]),                           # finite y with null label_end
])
def test_refusals_on_identity_dates_and_labels(pred_ids, ev_ids, dates, labels, y):
    preds = pd.DataFrame([_qrow(0)] * len(pred_ids), index=pred_ids, columns=calib.QCOLS)
    events = pd.DataFrame({"event_date": dates, "y": y, "label_end": labels}, index=ev_ids)
    with pytest.raises(ValueError):
        calib.calibrate(preds, events)


@pytest.mark.parametrize("drop_pred, drop_event", [
    (["q40"], []),
    ([], ["event_date"]),
    ([], ["y"]),
    ([], ["label_end"]),
])
def test_missing_columns_raise_value_error(drop_pred, drop_event):
    preds = pd.DataFrame([_qrow(0)], index=["a"], columns=calib.QCOLS).drop(columns=drop_pred)
    events = pd.DataFrame({"event_date": ["2020-01-10"], "y": [1.0], "label_end": ["2020-01-20"]},
                          index=["a"]).drop(columns=drop_event)
    with pytest.raises(ValueError):
        calib.calibrate(preds, events)


def test_malformed_label_end_text_raises_value_error():
    preds = pd.DataFrame([_qrow(0), _qrow(1)], index=["x", "t"], columns=calib.QCOLS)
    events = pd.DataFrame({"event_date": ["2020-01-10", "2020-03-01"], "y": [1.0, np.nan],
                           "label_end": ["not-a-date", None]}, index=["x", "t"])
    with pytest.raises(ValueError):
        calib.calibrate(preds, events)  # x is in the window of month 2020-03, so its label is parsed


@pytest.mark.parametrize("label_end", [None, float("nan"), "not-a-date", "2020-13-01", 20200110,
                                       pd.Timestamp("2020-01-10")])
def test_finite_y_row_in_no_donor_window_still_needs_an_iso_label_end(label_end):
    preds = pd.DataFrame([_qrow(0)], index=["a"], columns=calib.QCOLS)  # the only event: no later month uses it
    events = pd.DataFrame({"event_date": ["2020-01-10"], "y": [1.0],
                           "label_end": np.array([label_end], dtype=object)}, index=["a"])
    with pytest.raises(ValueError, match="label_end"):
        calib.calibrate(preds, events)


def test_nan_y_row_keeps_any_label_end():
    preds = pd.DataFrame([_qrow(0)], index=["a"], columns=calib.QCOLS)
    events = pd.DataFrame({"event_date": ["2020-01-10"], "y": [np.nan],
                           "label_end": np.array(["not-a-date"], dtype=object)}, index=["a"])
    arms, _, _ = calib.calibrate(preds, events)  # label_end is never read: the event has no donors
    assert arms["Q"].loc["a", "reason"] == "calibration_burn_in"


def test_floor_report_counts_and_shares_on_the_chain():
    rows, _, _, report = _run()
    _, qb, yb = _pick(rows, "b")
    _, qs, ys = _pick(rows, "s")
    t_ids, qt, _ = _pick(rows, "t")
    # S (thin, but scored now: its quantiles keep) and T are both scored Q events of 2020: 200 + 4.
    tq = np.vstack([_oracle_q(qs, qb, yb), _oracle_q(qt, np.vstack([qb, qs]), np.concatenate([yb, ys]))])

    q_rows = report[report["arm"] == "Q"].set_index("year")
    assert list(q_rows.index) == [2019, 2020]
    r = q_rows.loc[2020]
    assert r["events"] == len(tq) == 204
    assert r["event_hi_bound"] == int(np.sum(tq[:, 6] - tq[:, 3] < 0.5))
    assert r["event_lo_bound"] == int(np.sum(tq[:, 3] - tq[:, 0] < 0.5))
    # Donors are the 50 S-hi and 50 S-lo rows; 17 of each have a spread below 0.5 (the narrow pattern).
    assert (r["donors_hi"], r["donors_hi_bound"], r["donors_lo"], r["donors_lo_bound"]) == (50, 17, 50, 17)
    assert r["donor_hi_share"] == pytest.approx(17 / 50)
    assert r["donor_lo_share"] == pytest.approx(17 / 50)
    assert r["event_hi_share"] == pytest.approx(r["event_hi_bound"] / 204)
    z = q_rows.loc[2019]
    assert z["events"] == 0 and z["donors_hi"] == 0 and z["donors_lo"] == 0
    assert np.isnan(z["event_hi_share"]) and np.isnan(z["donor_lo_share"])
    # Q_raw donors also include the 500 B rows (y = v + 1 above a constant Q_raw = v, spread 0).
    raw_row = report[report["arm"] == "Q_raw"].set_index("year").loc[2020]
    assert (raw_row["donors_hi"], raw_row["donors_hi_bound"]) == (550, 517)
    assert (raw_row["donors_lo"], raw_row["donors_lo_bound"]) == (50, 17)


def test_floor_lines_are_logged_one_per_report_row_on_stderr_only(capsys):
    rows = _chain_rows()
    _, _, report = calib.calibrate(*_frames(rows))
    captured = capsys.readouterr()
    floor = [line for line in captured.err.splitlines() if line.startswith("  [quantile calibration] floor ")]
    assert len(floor) == len(report)
    assert any(line.startswith("  [quantile calibration] floor arm=Q year=2020 events=204 ") for line in floor)
    assert captured.out == ""


# ---------- 9. score_crps: a thin tail drops only the CRPS (Q3), for Q (calibrated pools) and B0 (own pools) ----------

CRPS_Q = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]  # spread 3 on each side, so the 0.5 floor does not bind


def _crps_frames():
    """Donors: 30 with y = 15 (above Q90 = 11, so (15 - 11) / 3 = 4/3) and 30 with y = 0 (below Q10 = 5, so
    (0 - 5) / 3 = -5/3), all dated Jan 2023. Every y is non-negative. t1 (Jul 2023) is in the 2023-07 pool of 30 + 30
    donors. t2 (Jun 2024) and t3 (Dec 2025) have no donor in their 12 months, so they are thin. t4 is in t1's month but
    has no label."""
    rows = [(f"d{i:02d}", "2023-01-15", "2023-01-20", 15.0 if i < 30 else 0.0) for i in range(60)]
    rows += [("t1", "2023-07-05", "2023-07-10", 7.0), ("t2", "2024-06-05", "2024-06-10", 7.0),
             ("t3", "2025-12-05", "2025-12-10", 7.0), ("t4", "2023-07-06", "2023-07-10", np.nan)]
    events = pd.DataFrame({"event_date": [r[1] for r in rows], "label_end": [r[2] for r in rows],
                           "y": [r[3] for r in rows]}, index=[r[0] for r in rows])
    frame = pd.DataFrame([CRPS_Q] * len(rows), index=events.index, columns=calib.QCOLS)
    frame["reason"] = None
    return events, frame


def test_thin_tails_change_only_the_crps_for_q_and_b0_with_exclusions_per_arm_and_year(capsys):
    events, frame = _crps_frames()
    # Q takes calibrate's pool, given here for 2023-07 only; B0 builds its own pools from the same rule.
    calibrated = {("Q", pd.Timestamp("2023-07-01")): (np.full(30, -5 / 3), np.full(30, 4 / 3))}
    out, exclusions, floors = calib.score_crps({"Q": frame, "B0": frame}, events, calibrated)
    want = _dense_crps(7.0, CRPS_Q, np.full(30, -5 / 3), np.full(30, 4 / 3))
    for name in ("Q", "B0"):
        got = out[name]
        pd.testing.assert_frame_equal(got.drop(columns="crps"), frame)  # quantiles and reasons are untouched
        assert list(got.index[got["crps"].notna()]) == ["t1"]  # the donors' and thin events' CRPS is NaN
        assert got.loc["t1", "crps"] == pytest.approx(want, abs=1e-4)
    # Scored: finite y and reason None (t4 has no label, so it is not counted). Excluded: scored with no pool.
    got = {(r["arm"], r["year"]): (r["scored"], r["crps_excluded"]) for r in exclusions}
    assert got == {(arm, year): pair for arm in ("Q", "B0")
                   for year, pair in ((2023, (1, 0)), (2024, (1, 1)), (2025, (1, 1)))}
    assert all(r["arm"] == "B0" for r in floors)
    rows = {r["year"]: r for r in floors}
    assert (rows[2023]["events"], rows[2024]["events"], rows[2025]["events"]) == (62, 1, 1)
    assert (rows[2023]["donors_hi"], rows[2023]["donors_lo"]) == (30, 30)
    # Thin months with finite rows: 2023-01 (no donor dated in its 12 months), 2024-06 and 2025-12, for both frames.
    assert "[stage A CRPS] arm=B0 scored=63 thin_months=3" in capsys.readouterr().err


def _set(events, id_, column, value):
    events = events.copy()
    events.at[id_, column] = value
    return events


DATE_MSG = "event_date must be set; y must be finite or NaN and non-negative"
LABEL_MSG = "a finite y needs a label_end that is an ISO date"
INDEX_MSG = "predictions and events must have unique indexes"


@pytest.mark.parametrize("name", ["Q", "B0"])
@pytest.mark.parametrize("mutate, message", [
    (lambda e: _set(e, "t1", "y", -1.0), DATE_MSG),                  # negative y
    (lambda e: _set(e, "t1", "y", np.inf), DATE_MSG),                # +inf y
    (lambda e: pd.concat([e, e.loc[["t1"]]]), INDEX_MSG),            # duplicate events index
    (lambda e: _set(e, "t1", "event_date", None), DATE_MSG),         # null event_date
    (lambda e: _set(e, "t1", "label_end", None), LABEL_MSG),         # finite y, null label_end
    (lambda e: _set(e, "t1", "label_end", "not-a-date"), LABEL_MSG),  # finite y, non-ISO label_end
    (lambda e: _set(e, "t3", "label_end", None), LABEL_MSG),         # t3 is finite too: refused, not skipped
    (lambda e: _set(e, "t3", "label_end", "2025-13-10"), LABEL_MSG),  # same row, non-ISO label_end
])
def test_score_crps_refuses_what_calibrate_refuses(name, mutate, message):
    events, frame = _crps_frames()
    with pytest.raises(ValueError, match=message):
        calib.score_crps({name: frame}, mutate(events), {})


@pytest.mark.parametrize("drop", [["q40"], ["reason"]])
def test_score_crps_frame_missing_a_column_raises_value_error(drop):
    events, frame = _crps_frames()
    with pytest.raises(ValueError, match="needs the seven QCOLS and reason"):
        calib.score_crps({"B1": frame.drop(columns=drop)}, events, {})


def test_thin_tail_month_keeps_its_quantile_values_and_gets_no_crps():
    rows = _chain_rows(29, 50)
    preds, events = _frames(rows)
    arms, pools, _ = calib.calibrate(preds, events)
    _, qb, yb = _pick(rows, "b")
    _, qs, ys = _pick(rows, "s")
    t_ids, qt, _ = _pick(rows, "t")
    expected = _oracle_q(qt, np.vstack([qb, qs]), np.concatenate([yb, ys]))
    np.testing.assert_allclose(arms["Q"].loc[t_ids, calib.QCOLS].to_numpy(), expected, rtol=0, atol=1e-9)
    assert arms["Q"].loc[t_ids, "reason"].isna().all()
    assert ("Q", T_MONTH) not in pools  # thin: the quantiles are kept, only the pool is absent
    out, _, _ = calib.score_crps({"Q": arms["Q"]}, events, pools)
    assert out["Q"].loc[t_ids, "crps"].isna().all()


def test_b0_floor_row_counts_bound_events_by_hand():
    events, frame = _crps_frames()
    frame = frame.copy()
    frame.loc["t1", calib.QCOLS] = [6.0, 6.5, 7.0, 7.0, 7.0, 7.1, 7.25]  # q90 - q50 = 0.25: the upper floor binds
    _, _, floors = calib.score_crps({"B0": frame}, events, {})
    rows = {r["year"]: r for r in floors}
    # 2023: 60 donors, t1 and t4 (t4 has finite quantiles but no label, and the floor row counts quantiles only).
    assert rows[2023]["events"] == 62
    assert (rows[2023]["event_hi_bound"], rows[2023]["event_lo_bound"]) == (1, 0)
    assert rows[2023]["event_hi_share"] == pytest.approx(1 / 62)
    assert (rows[2024]["events"], rows[2024]["event_hi_bound"]) == (1, 0)
