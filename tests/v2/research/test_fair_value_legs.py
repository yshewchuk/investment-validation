"""Reference tests for fill model F slice F3: fit set, smile, vol source, leg values.

Synthetic inputs only (guides/fair_value_fill_design.md, sections 3, 4, 7, 10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments import fair_value_legs as legs
from experiments import fair_value_pricer as fvp

OBS, EXP, EXP2 = pd.Timestamp("2026-11-25"), pd.Timestamp("2026-12-25"), pd.Timestamp("2027-01-22")
T = 30 / 365  # EXP is 30 calendar days out
SPOT, RATE = 100.0, 0.04
BASE = np.arange(85.0, 120.1, 5.0)  # 8 strikes: fit set is 4 puts and 4 calls
QCOLS = ["strike", "right", "bid", "ask", "iv", "spot", "quote_repaired", "rate"]


def iv_curve(K):
    return 0.25 + 0.002 * (K - 100.0) + 0.0004 * (K - 100.0) ** 2


def smv_of(K):
    return 0.40 + 0.001 * (K - 100.0)  # clearly apart from iv_curve


def quad_iv(K):
    """Exact quadratic in k = ln(K/SPOT)/sqrt(T), the fit variable."""
    k = np.log(np.asarray(K, dtype=float) / SPOT) / np.sqrt(T)
    return 0.25 - 0.03 * k + 0.05 * k * k


def smile_rows(strikes, iv_of, bid=0.5, ask=1.5):
    return [(float(K), r, bid, ask, iv_of(K), SPOT, False, RATE) for K in strikes for r in ("C", "P")]


def quotes_frame(rows, expiry=EXP):
    df = pd.DataFrame(rows, columns=QCOLS)
    df.insert(0, "ticker", "XYZ")
    df.insert(1, "obs_date", OBS)
    df.insert(2, "expiry", expiry)
    return df


def vols_frame(strikes=(), smv=(), expiry=EXP):
    """Vol table with the re-ingest's extra columns, which must be ignored."""
    df = pd.DataFrame({"strike": np.asarray(strikes, float), "smvVol": np.asarray(smv, float)})
    df["callMidIv"], df["putMidIv"], df["stockPrice"] = 0.9, 0.9, 1.0
    df.insert(0, "ticker", "XYZ")
    df.insert(1, "obs_date", OBS)
    df.insert(2, "expiry", expiry)
    return df


def assert_unavailable(rows):
    """Rows with no fair value: V and vol NaN, no clamp side."""
    assert rows.V.isna().all() and rows.vol.isna().all()
    assert all(s is None for s in rows.clamp_side)


def _fit_expected(q, weighted=True):
    """Independent lstsq on the fit set (puts K<=SPOT, calls K>SPOT), weights 1/max(rel, 0.02), k clipped."""
    strike, right = q["strike"].to_numpy(dtype=float), q["right"].to_numpy()
    bid, ask, iv = (q[name].to_numpy(dtype=float) for name in ("bid", "ask", "iv"))
    k = np.log(strike / SPOT) / np.sqrt(T)
    fit = ((right == "P") & (strike <= SPOT)) | ((right == "C") & (strike > SPOT))
    w = 1.0 / np.maximum((ask - bid) / ((bid + ask) / 2.0), 0.02) if weighted else np.ones(len(k))
    design = np.column_stack([k**2, k, np.ones_like(k)])
    coef = np.linalg.lstsq(design[fit] * w[fit, None], iv[fit] * w[fit], rcond=None)[0]
    kc = np.clip(k, k[fit].min(), k[fit].max())
    return coef[0] * kc**2 + coef[1] * kc + coef[2]


BASE_NO_SPOT = [85.0, 90.0, 95.0, 105.0, 110.0, 115.0, 120.0]  # no strike at spot, so probes can add one


def _smile_base():
    return [(K, r, 0.5, 1.5, quad_iv(K), SPOT, False, RATE) for K in BASE_NO_SPOT for r in ("C", "P")]


def test_smile_recovers_noisy_curve_from_bid_ask_ivs():
    rng = np.random.default_rng(20261125)
    rows = []
    for K in np.arange(70.0, 130.1, 2.5):  # 25 strikes
        k = np.log(K / SPOT) / np.sqrt(T)
        true = 0.25 - 0.03 * k + 0.05 * k * k
        e_bid, e_ask = rng.normal(0.0, 0.002, 2)
        stored = ((true + e_bid) + (true + e_ask)) / 2.0  # stored iv = midpoint of bid and ask ivs
        half = rng.uniform(0.02, 0.25)  # relative spread 0.04 to 0.5
        rows += [(K, r, 1 - half, 1 + half, stored, SPOT, False, RATE) for r in ("C", "P")]
    out = legs.leg_values(quotes_frame(rows), vols_frame())
    assert len(out) == 50 and set(out.source) == {"smile"} and set(out.status) == {"ok"}
    k = np.log(out.strike.to_numpy() / SPOT) / np.sqrt(T)
    np.testing.assert_allclose(out.vol.to_numpy(), 0.25 - 0.03 * k + 0.05 * k * k, rtol=0, atol=0.005)


def test_fit_smile_matches_independent_weighted_least_squares():
    strike = 80.0 + 5.0 * np.arange(8)
    k = np.array([-0.9, -0.6, -0.3, -0.1, 0.2, 0.5, 0.8, 1.1])
    iv = 0.2 + 0.05 * k + 0.1 * k**2
    iv[2] += 0.03  # off-curve point
    rel_wide = np.full(8, 0.10)
    rel_wide[2] = 0.40
    rel_tight = np.full(8, 0.10)
    rel_tight[2] = 0.005  # below the 0.02 floor
    fits = []
    for rel in (rel_wide, rel_tight):
        w = 1.0 / np.maximum(rel, 0.02)
        design = np.column_stack([k**2, k, np.ones_like(k)])
        expected = np.linalg.lstsq(design * w[:, None], iv * w, rcond=None)[0]
        coef, k_min, k_max = legs.fit_smile(strike, k, iv, rel)
        np.testing.assert_allclose(coef, expected, rtol=1e-9, atol=1e-12)
        assert (k_min, k_max) == (-0.9, 1.1)
        fits.append(coef)
    assert not np.allclose(fits[0], fits[1], atol=1e-6)


def test_edge_value_holds_outside_fitted_range():
    rows = smile_rows(np.arange(90.0, 110.1, 2.5), quad_iv)
    rows += [(70.0, "C", 0.5, 1.5, 0.9, SPOT, False, RATE), (80.0, "C", 0.5, 1.5, 0.9, SPOT, False, RATE),
             (120.0, "P", 0.5, 1.5, 0.9, SPOT, False, RATE), (130.0, "P", 0.5, 1.5, 0.9, SPOT, False, RATE)]
    out = legs.leg_values(quotes_frame(rows), vols_frame())
    assert set(out.source) == {"smile"}
    K, vol = out.strike.to_numpy(), out.vol.to_numpy()
    k_lo, k_hi = np.log(0.9) / np.sqrt(T), np.log(1.1) / np.sqrt(T)  # k at strikes 90 and 110
    edge_lo = 0.25 - 0.03 * k_lo + 0.05 * k_lo**2
    edge_hi = 0.25 - 0.03 * k_hi + 0.05 * k_hi**2
    inside = (K >= 90.0) & (K <= 110.0)
    np.testing.assert_allclose(vol[inside], quad_iv(K[inside]), rtol=0, atol=1e-9)
    np.testing.assert_allclose(vol[K < 90.0], edge_lo, rtol=0, atol=1e-9)  # 70 and 80 hold the k(90) value
    np.testing.assert_allclose(vol[K > 110.0], edge_hi, rtol=0, atol=1e-9)  # 120 and 130 hold the k(110) value
    assert abs(edge_lo - edge_hi) > 0.01  # a constant vol would fail


def test_spread_weights_through_leg_values():
    rows = []
    for K in BASE:
        for r in ("C", "P"):
            bid, ask, iv = 0.95, 1.05, quad_iv(K)
            if (K, r) == (90.0, "P"):
                bid, ask, iv = 0.5, 1.5, quad_iv(K) + 0.05  # wide spread, off the curve
            if (K, r) == (110.0, "C"):
                bid, ask = 0.995, 1.005  # relative spread 0.01, below the 0.02 floor
            rows.append((K, r, bid, ask, iv, SPOT, False, RATE))
    q = quotes_frame(rows)
    out = legs.leg_values(q, vols_frame())
    assert set(out.source) == {"smile"}
    expected = _fit_expected(q)
    np.testing.assert_allclose(out.vol.to_numpy(), expected, rtol=0, atol=1e-8)
    assert np.abs(expected - _fit_expected(q, weighted=False)).max() > 1e-3


def test_fit_smile_none_cases_and_failed_fit_per_row():
    ones = np.ones
    assert legs.fit_smile(np.arange(4.0), np.array([-0.5, -0.2, 0.2, 0.5]), ones(4) * 0.25, ones(4) * 0.1) is None
    strike6 = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 4.0])  # four distinct strikes among six rows
    k6 = np.array([-0.5, -0.5, -0.2, 0.2, 0.5, 0.5])
    assert legs.fit_smile(strike6, k6, ones(6) * 0.25, ones(6) * 0.1) is None
    kneg = np.linspace(-1.0, -0.1, 5)  # one side only
    assert legs.fit_smile(np.arange(5.0), kneg, ones(5) * 0.25, ones(5) * 0.1) is None
    kq = np.linspace(-1.0, 1.0, 9)
    assert legs.fit_smile(np.arange(9.0), kq, kq**2 - 0.01, ones(9) * 0.1) is None  # interior min -0.01
    assert legs.fit_smile(np.arange(9.0), kq, kq**2 + 0.01, ones(9) * 0.1) is not None
    rows = [(K, "P", 0.5, 1.5, 0.25, SPOT, False, RATE) for K in (80.0, 85.0, 90.0, 95.0, 97.5)]
    rows += [(K, "C", 0.5, 1.5, 0.25, SPOT, False, RATE) for K in (85.0, 90.0, 95.0)]  # all k < 0
    out = legs.leg_values(quotes_frame(rows), vols_frame())
    assert set(out.source) == {"smile"} and set(out.status) == {"no_fair_value"}
    assert_unavailable(out)


def test_smvvol_shares_vol_across_rights_and_ignores_extra_columns():
    q = quotes_frame(smile_rows(BASE, lambda K: 0.25))
    v = vols_frame(BASE, [smv_of(K) for K in BASE])
    out = legs.leg_values(q, v)
    assert set(out.source) == {"smvvol"} and set(out.status) == {"ok"}
    np.testing.assert_array_equal(out.vol.to_numpy(), [smv_of(K) for K in out.strike])
    plain = v[["ticker", "obs_date", "expiry", "strike", "smvVol"]]
    pd.testing.assert_frame_equal(out, legs.leg_values(q, plain))


def test_row_outside_fit_without_smv_is_no_fair_value_rest_ok():
    rows = smile_rows(BASE, lambda K: 0.25) + [(80.0, "C", 0.5, 1.5, 0.25, SPOT, False, RATE)]
    out = legs.leg_values(quotes_frame(rows), vols_frame(BASE, [smv_of(K) for K in BASE]))
    assert set(out.source) == {"smvvol"}
    assert out.status.iloc[:-1].eq("ok").all() and out.status.iloc[-1] == "no_fair_value"
    np.testing.assert_array_equal(out.vol.to_numpy()[:-1], [smv_of(K) for K in out.strike.iloc[:-1]])
    assert_unavailable(out.iloc[[-1]])


@pytest.mark.parametrize("bad", [None, 0.0, np.nan, np.inf])
def test_one_missing_fit_vol_makes_every_row_smile(bad):
    strikes = [K for K in BASE if K != 95.0] if bad is None else list(BASE)
    smv = [smv_of(K) if K != 95.0 else bad for K in strikes]
    q = quotes_frame(smile_rows(BASE, iv_curve))
    out = legs.leg_values(q, vols_frame(strikes, smv))
    assert set(out.source) == {"smile"} and set(out.status) == {"ok"}
    pd.testing.assert_frame_equal(out, legs.leg_values(q, vols_frame()))
    vol = out.vol.to_numpy()
    assert np.ptp(vol) > 0.01
    assert (np.abs(vol - smv_of(out.strike.to_numpy())) > 1e-6).all()


@pytest.mark.parametrize("K, right, bid, ask, inside", [
    (100.0, "P", 0.5, 1.5, True),   # put at spot: in the fit set
    (100.0, "C", 0.5, 1.5, False),  # call at spot: out
    (80.0, "P", 0.0, 1.0, True),    # zero bid, positive ask: in
    (80.0, "P", 1.0, 1.0, True),    # locked quote, bid == ask > 0, not repaired: in
])
def test_fit_admission_boundaries(K, right, bid, ask, inside):
    base = legs.leg_values(quotes_frame(_smile_base()), vols_frame()).vol.to_numpy()
    probe = (K, right, bid, ask, quad_iv(K) + 0.1, SPOT, False, RATE)  # wild iv, off the curve
    out = legs.leg_values(quotes_frame(_smile_base() + [probe]), vols_frame()).vol.to_numpy()
    if inside:
        assert np.abs(out[:-1] - base).max() > 1e-4  # the probe moves the fit
    else:
        np.testing.assert_array_equal(out[:-1], base)  # the probe does not move the fit


def test_exactly_five_strikes_with_both_sides_fit():
    picks = ((85.0, "P"), (90.0, "P"), (95.0, "P"), (105.0, "C"), (110.0, "C"))
    out = legs.leg_values(quotes_frame([(K, r, 0.5, 1.5, quad_iv(K), SPOT, False, RATE) for K, r in picks]),
                          vols_frame())
    strike = out.strike.to_numpy()
    assert legs.fit_smile(strike, np.log(strike / SPOT) / np.sqrt(T), quad_iv(strike), np.full(5, 1.0)) is not None
    assert set(out.source) == {"smile"} and set(out.status) == {"ok"}
    np.testing.assert_allclose(out.vol.to_numpy(), quad_iv(strike), rtol=0, atol=1e-9)


def test_fitter_failures_return_none(monkeypatch):
    strike = np.array([85.0, 90.0, 95.0, 105.0, 110.0, 115.0])
    args = (strike, np.log(strike / SPOT) / np.sqrt(T), quad_iv(strike), np.full(6, 0.1))
    assert legs.fit_smile(*args) is not None

    def singular(*_, **__):
        raise np.linalg.LinAlgError("singular matrix")

    monkeypatch.setattr(legs.np, "polyfit", singular)
    assert legs.fit_smile(*args) is None
    monkeypatch.setattr(legs.np, "polyfit", lambda *_, **__: np.array([np.nan, 0.0, 0.2]))
    assert legs.fit_smile(*args) is None


def test_quadratic_non_positive_at_range_end_is_none():
    k, strike, ones = np.linspace(-0.5, 1.0, 6), np.arange(6.0), np.full(6, 0.1)
    assert legs.fit_smile(strike, k, 0.2 - 0.1 * k, ones) is not None  # linear, 0.1 at k_max
    assert legs.fit_smile(strike, k, 0.2 - 0.25 * k, ones) is None  # linear, -0.05 at k_max


EXCLUDED = [
    (60.0, "C", 0.5, 1.5, 2.0, SPOT, False, RATE),      # in the money, wild iv
    (92.5, "P", 1.0, 1.0, 3.0, SPOT, True, RATE),       # repaired
    (97.5, "P", 0.5, 1.5, 0.0, SPOT, False, RATE),      # iv zero
    (102.5, "C", 0.5, 1.5, np.nan, SPOT, False, RATE),  # iv NaN
    (87.5, "P", 0.0, 0.0, 0.3, SPOT, False, RATE),      # bid = ask = 0
    (77.5, "P", 2.0, 1.0, 5.0, SPOT, False, RATE),      # bid > ask, wild iv
    (112.5, "C", 0.5, 1.5, 5.0, 0.0, False, RATE),      # spot zero: not in the fit set although strike > spot
]


def test_fit_set_exclusions_do_not_move_the_fit():
    base_rows = smile_rows(BASE, iv_curve)
    out_base = legs.leg_values(quotes_frame(base_rows), vols_frame())
    out_var = legs.leg_values(quotes_frame(base_rows + EXCLUDED), vols_frame())
    assert set(out_base.source) == {"smile"} and np.isfinite(out_base.vol.to_numpy()).all()
    np.testing.assert_array_equal(out_var.vol.to_numpy()[:len(base_rows)], out_base.vol.to_numpy())
    # With smvVol, rows outside the fit set (and a repaired one) with no vol do not force the smile.
    rows = smile_rows(BASE, lambda K: 0.25)
    out = legs.leg_values(quotes_frame(rows + EXCLUDED[:2]), vols_frame(BASE, [smv_of(K) for K in BASE]))
    n = len(rows)
    assert set(out.source) == {"smvvol"}
    np.testing.assert_array_equal(out.vol.to_numpy()[:n], [smv_of(K) for K in out.strike.iloc[:n]])
    assert out.status.iloc[n:].tolist() == ["no_fair_value", "no_fair_value"]
    assert_unavailable(out.iloc[n:])


def test_expired_rows_are_intrinsic_without_vol_or_rate():
    rows = [
        (90.0, "C", 9.0, 11.0, np.nan, SPOT, False, np.nan),  # intrinsic 10, inside
        (110.0, "P", 4.0, 6.0, np.nan, SPOT, False, np.nan),  # intrinsic 10, clamps to ask 6
        (95.0, "C", 6.0, 7.0, np.nan, SPOT, False, np.nan),   # intrinsic 5, clamps to bid 6
        (95.0, "P", 0.0, 0.5, np.nan, SPOT, False, np.nan),   # intrinsic 0, inside
        (100.0, "C", 0.0, 1.0, np.nan, 0.0, False, np.nan),   # bad spot
    ]
    out = legs.leg_values(quotes_frame(rows, expiry=OBS), vols_frame(expiry=OBS))
    assert set(out.source) == {"intrinsic"}
    assert out.status.tolist() == ["ok", "ok", "ok", "ok", "no_fair_value"]
    np.testing.assert_array_equal(out.V.to_numpy()[:4], [10.0, 6.0, 6.0, 0.0])
    assert out.clamp_side.tolist() == ["none", "ask", "bid", "none", None]
    for c in ("status", "source", "clamp_side"):
        assert out[c].dtype == object
    assert out.clamp_side.iloc[-1] is None
    assert out.vol.isna().all() and out.rate.isna().all()
    assert_unavailable(out.iloc[[-1]])


def test_nat_expiry_has_no_source_and_no_fair_value():
    rows = [(100.0, "C", 0.5, 1.5, 0.25, SPOT, False, RATE)]
    out = legs.leg_values(quotes_frame(rows, expiry=pd.NaT), vols_frame(expiry=pd.NaT))
    assert out.source.iloc[0] is None
    assert out.status.iloc[0] == "no_fair_value"
    assert_unavailable(out)


UNPRICEABLE = [(np.nan, 1.5, False), (0.5, np.inf, False), (-0.1, 1.0, False),
               (0.5, 0.0, False), (1.0, 0.5, False), (0.0, 0.0, True)]


@pytest.mark.parametrize("bid, ask, repaired", UNPRICEABLE)
def test_each_unpriceable_condition(bid, ask, repaired):
    rows = smile_rows(BASE, iv_curve) + [(77.5, "P", bid, ask, 0.25, SPOT, repaired, RATE)]
    out = legs.leg_values(quotes_frame(rows), vols_frame())
    assert out.status.iloc[-1] == "unpriceable" and np.isnan(out.V.iloc[-1])
    assert_unavailable(out.iloc[[-1]])


def test_status_precedence_and_usable_zero_bid():
    rows = smile_rows(BASE, lambda K: 0.25) + [
        (77.5, "P", np.nan, 1.5, 0.25, SPOT, False, np.nan),  # bad quote beats NaN rate
        (72.5, "P", 0.5, 1.5, 0.25, 0.0, False, np.nan),      # NaN rate beats bad spot
        (122.5, "C", 0.0, 1.0, 0.25, SPOT, False, RATE),      # zero bid, positive ask: usable
    ]
    out = legs.leg_values(quotes_frame(rows), vols_frame())
    assert out.status.iloc[-3:].tolist() == ["unpriceable", "rate_unknown", "ok"]
    assert_unavailable(out.iloc[-3:-1])  # the unpriceable and the rate_unknown rows


def test_values_equal_pricer_and_clamp_to_effective_quote():
    strikes = (95.0, 100.0, 105.0)
    model, rows = {}, []
    for K in strikes:
        for right, price in (("P", fvp.american_put), ("C", fvp.european_call)):
            m = float(price(SPOT, K, T, 0.30, RATE))
            model[(K, right)] = m
            rows.append((K, right, 0.5 * m, 1.5 * m, 0.25, SPOT, False, RATE))
    rows[0] = (95.0, "P", 0.5 * model[(95.0, "P")], 0.9 * model[(95.0, "P")], 0.25, SPOT, False, RATE)
    rows[5] = (105.0, "C", 1.1 * model[(105.0, "C")], 1.5 * model[(105.0, "C")], 0.25, SPOT, False, RATE)
    out = legs.leg_values(quotes_frame(rows), vols_frame(strikes, [0.30, 0.30, 0.30]))
    assert set(out.source) == {"smvvol"} and set(out.status) == {"ok"}
    np.testing.assert_array_equal(out.vol.to_numpy(), 0.30)
    V = out.V.to_numpy()
    expected = np.array([model[(K, r)] for K in strikes for r in ("P", "C")])
    np.testing.assert_array_equal(V[[1, 2, 3, 4]], expected[[1, 2, 3, 4]])
    assert V[0] == rows[0][3] and V[5] == rows[5][2]  # clamped to ask and to bid
    assert out.clamp_side.tolist() == ["ask", "none", "none", "none", "none", "bid"]


def test_repaired_rows_use_effective_quote():
    lo, hi = legs.effective_quote([2.0, 1.0], [2.0, 3.0], [True, False])
    np.testing.assert_array_equal(lo, [1.5, 1.0])
    np.testing.assert_array_equal(hi, [2.5, 3.0])
    rows = [
        (90.0, "C", 4.0, 4.0, np.nan, SPOT, True, np.nan),  # P = 4: [3, 5]; intrinsic 10 clamps to 5
        (90.0, "P", 4.0, 4.0, np.nan, SPOT, True, np.nan),  # intrinsic 0 clamps to 3
    ]
    out = legs.leg_values(quotes_frame(rows, expiry=OBS), vols_frame(expiry=OBS))
    assert out.status.tolist() == ["ok", "ok"]
    np.testing.assert_array_equal(out.V.to_numpy(), [5.0, 3.0])
    assert out.clamp_side.tolist() == ["ask", "bid"]
    np.testing.assert_array_equal(out.eff_bid.to_numpy(), [3.0, 3.0])
    np.testing.assert_array_equal(out.eff_ask.to_numpy(), [5.0, 5.0])


def test_output_columns_and_passthrough():
    rows = [
        (100.0, "C", 0.5, 1.5, 0.25, SPOT, False, 0.031),
        (90.0, "P", 4.0, 4.0, np.nan, SPOT, True, 0.047),  # repaired: effective [3, 5]
        (110.0, "C", 0.9, 1.1, 0.25, SPOT, False, 0.052),
    ]
    q = quotes_frame(rows)
    out = legs.leg_values(q, vols_frame())
    assert list(out.columns) == ["ticker", "obs_date", "expiry", "strike", "right", "status", "V", "vol",
                                 "source", "rate", "clamp_side", "repaired", "eff_bid", "eff_ask"]
    for name in ("ticker", "obs_date", "expiry", "strike", "right", "rate"):
        assert out[name].tolist() == q[name].tolist(), name
    assert out.repaired.tolist() == q.quote_repaired.tolist() == [False, True, False]
    np.testing.assert_allclose(out.eff_bid.to_numpy(), [0.5, 3.0, 0.9], rtol=0, atol=1e-12)
    np.testing.assert_allclose(out.eff_ask.to_numpy(), [1.5, 5.0, 1.1], rtol=0, atol=1e-12)


def _two_expiries():
    q1 = quotes_frame(smile_rows(BASE, iv_curve), expiry=EXP)
    q2 = quotes_frame(smile_rows(BASE, lambda K: 0.25), expiry=EXP2)
    v = vols_frame(BASE, [smv_of(K) for K in BASE], expiry=EXP2)
    return pd.concat([q1, q2], ignore_index=True), v


def test_order_independent_separable_index_kept_and_not_mutated():
    q, v = _two_expiries()
    q0, v0 = q.copy(deep=True), v.copy(deep=True)
    out = legs.leg_values(q, v)
    assert set(out.source) == {"smile", "smvvol"}
    pd.testing.assert_frame_equal(q, q0)
    pd.testing.assert_frame_equal(v, v0)
    rng = np.random.default_rng(11)
    shuffled = legs.leg_values(q.iloc[rng.permutation(len(q))], v.iloc[rng.permutation(len(v))])
    pd.testing.assert_frame_equal(out.set_index(legs.QUOTE_KEY).sort_index(),
                                  shuffled.set_index(legs.QUOTE_KEY).sort_index(), check_exact=True)
    alone = legs.leg_values(q[q["expiry"] == EXP2], v)
    assert len(alone) == 16
    pd.testing.assert_frame_equal(alone.reset_index(drop=True),
                                  out[out["expiry"] == EXP2].reset_index(drop=True), check_exact=True)
    dup = q.copy()
    dup.index = np.arange(len(q)) // 2  # non-unique index
    dup_out = legs.leg_values(dup, v)
    pd.testing.assert_index_equal(dup_out.index, dup.index)
    assert dup_out.strike.tolist() == dup.strike.tolist()
    pd.testing.assert_frame_equal(dup_out.reset_index(drop=True), out.reset_index(drop=True), check_exact=True)


def _base():
    return quotes_frame(smile_rows(BASE, iv_curve)), vols_frame()


def _refusal_case(mutate):
    def make():
        q, v = _base()
        return mutate(q, v)

    return make


def _object_repaired(value):
    def mutate(q, v):
        q["quote_repaired"] = q["quote_repaired"].astype(object)
        q.loc[q.index[0], "quote_repaired"] = value
        return q, v

    return mutate


CASES = [
    (_refusal_case(lambda q, v: (q.drop(columns=["bid"]), v)), "column_missing"),
    (_refusal_case(lambda q, v: (q, v.drop(columns=["smvVol"]))), "column_missing"),
    (_refusal_case(lambda q, v: (pd.concat([q, q.iloc[[0]]], ignore_index=True), v)), "duplicate_quote_key"),
    (_refusal_case(lambda q, v: (q, vols_frame([85.0, 85.0], [np.nan, 0.3]))), "duplicate_vol_key"),
    (_refusal_case(lambda q, v: (q.assign(right=q["right"].where(q.index != 0, "X")), v)), "right_invalid"),
    (_refusal_case(lambda q, v: (q.assign(right=q["right"].where(q.index != 0, None)), v)), "right_invalid"),
    (_refusal_case(_object_repaired(None)), "quote_repaired_invalid"),
    (_refusal_case(_object_repaired(1)), "quote_repaired_invalid"),
]


@pytest.mark.parametrize("make, reason", CASES)
def test_refusals_carry_code_and_reason(make, reason):
    q, v = make()
    with pytest.raises(legs.FairValueRefusal) as info:
        legs.leg_values(q, v)
    assert info.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert info.value.details["reason"] == reason
    assert "EXPERIMENT_IDENTITY_CONFLICT" in str(info.value) and reason in str(info.value)


def test_column_missing_names_table_and_columns_and_boolean_accepted():
    q, v = _base()
    with pytest.raises(legs.FairValueRefusal) as info:
        legs.leg_values(q.drop(columns=["bid", "spot"]), v)
    assert info.value.details == {"reason": "column_missing", "table": "quotes", "columns": ["bid", "spot"]}
    with pytest.raises(legs.FairValueRefusal) as info:
        legs.leg_values(q, v.drop(columns=["smvVol"]))
    assert info.value.details == {"reason": "column_missing", "table": "vols", "columns": ["smvVol"]}
    q["quote_repaired"] = q["quote_repaired"].astype("boolean")  # nullable, no nulls: accepted
    assert legs.leg_values(q, v).repaired.tolist() == [False] * len(q)


def test_empty_input_keeps_columns_and_stdout_stays_empty(capsys):
    q, v = _base()
    empty = legs.leg_values(q.iloc[0:0], v.iloc[0:0])
    assert list(empty.columns) == legs.OUT_COLUMNS and len(empty) == 0
    capsys.readouterr()
    out = legs.leg_values(q, v)
    captured = capsys.readouterr()
    assert captured.out == "" and "[fv_legs]" in captured.err
    assert len(out) == len(q)


def test_published_anchor_through_leg_values():
    """Independent anchor: S=K=100, r=5%, vol=20%, T=1. Expected values are hard-coded, not from fvp."""
    exp_1y = pd.Timestamp("2027-11-25")  # OBS + 365 calendar days, so T = 1.0
    rows = [
        (100.0, "C", 0.5, 20.0, 0.9, SPOT, False, 0.05),
        (100.0, "P", 0.5, 20.0, 0.9, SPOT, False, 0.05),
    ]
    out = legs.leg_values(quotes_frame(rows, expiry=exp_1y), vols_frame([100.0], [0.2], expiry=exp_1y))
    assert out.source.tolist() == ["smvvol", "smvvol"]  # strike 100 == spot: put is in the fit set
    assert out.status.tolist() == ["ok", "ok"]
    assert out.clamp_side.tolist() == ["none", "none"]
    assert out.vol.tolist() == pytest.approx([0.2, 0.2])  # smvVol, not the stored 0.9
    V = dict(zip(out.right, out.V))
    assert V["C"] == pytest.approx(10.4506, abs=1e-4)  # Black-Scholes textbook value
    assert V["P"] == pytest.approx(6.090, abs=0.01)  # published American put value
