"""Reference tests for fill model F slice F4a: concession, sensitivities, per-quote leg fills.

Synthetic inputs only (guides/fair_value_fill_design.md, sections 5, 7, 10). Expiry == obs_date gives T = 0, so V is
the intrinsic value clamped to [bid, ask]; spot is per row. Expected numbers are hand-computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments import fair_value_fills as ff
from experiments import fair_value_legs as fvl
from experiments.fair_value_legs import FairValueRefusal

OBS, EXP = pd.Timestamp("2026-11-25"), pd.Timestamp("2026-12-25")
NAN = float("nan")


def near(x, tol=1e-9):
    return pytest.approx(x, abs=tol)


def row(K, right, bid, ask, spot=101.0, expiry=OBS, rep=False, oi=1000, bsz=50, asz=50, rate=0.04):
    """One quote row. Expiry OBS gives T = 0 (V = intrinsic clamped to [bid, ask])."""
    return {"ticker": "XYZ", "obs_date": OBS, "expiry": expiry, "strike": float(K), "right": right,
            "bid": bid, "ask": ask, "iv": 0.3, "spot": spot, "quote_repaired": rep, "rate": rate,
            "open_interest": oi, "bid_size": bsz, "ask_size": asz}


def no_vols():
    """Empty vol table with the VOL_COLUMNS, built the same way as the leg tests."""
    df = pd.DataFrame({"strike": np.asarray((), float), "smvVol": np.asarray((), float)})
    df.insert(0, "ticker", "XYZ")
    df.insert(1, "obs_date", OBS)
    df.insert(2, "expiry", OBS)
    return df


def fills(*rows):
    return ff.leg_fills(pd.DataFrame(list(rows)), no_vols())


# (bid, ask, bid_size, ask_size, OI, c_buy, c_sell, fill_buy, fill_sell) for a call at spot 101, K 100: V = mid = 1
CONCESSION = [
    (1.0, 1.0, 50, 50, 1000, 0.25, 0.25, 1.0, 1.0),           # s = 0
    (0.975, 1.025, 50, 50, 1000, 0.30, 0.30, 1.0075, 0.9925),  # s = 0.05
    (0.9, 1.1, 50, 50, 1000, 0.45, 0.45, 1.045, 0.955),        # s = 0.20
    (0.75, 1.25, 50, 50, 1000, 0.75, 0.75, 1.1875, 0.8125),    # s = 0.50
    (0.5, 1.5, 50, 50, 1000, 0.75, 0.75, 1.375, 0.625),        # s = 1.00, capped at SPREAD_SCALE
    (0.75, 1.25, 9, 9, 99, 1.0, 1.0, 1.25, 0.75),              # s = 0.50, both thin: c = 1, the largest possible sum
    (0.5, 1.5, 9, 9, 99, 1.0, 1.0, 1.5, 0.5),                  # s = 1.00, both thin: c = 1, the largest possible sum
    (0.9, 1.1, 10, 9, 100, 0.60, 0.45, 1.06, 0.955),           # ask thin, bid not; OI 100 not thin
    (0.9, 1.1, 0, 0, 1000, 0.60, 0.60, 1.06, 0.94),            # zero sizes are known and thin
    (0.9, 1.1, 50, 50, 99, 0.55, 0.55, 1.055, 0.945),          # OI 99 adds 0.10
]


@pytest.mark.parametrize("bid,ask,bsz,asz,oi,cb,cs,fb,fs", CONCESSION)
def test_concession_known_answers(bid, ask, bsz, asz, oi, cb, cs, fb, fs):
    out = fills(row(100, "C", bid, ask, spot=101.0, bsz=bsz, asz=asz, oi=oi)).iloc[0]
    assert out.V == near(1.0)
    assert out.c_buy == near(cb) and out.c_sell == near(cs)
    assert out.fill_buy == near(fb) and out.fill_sell == near(fs)


def test_unknown_liquidity_sensitivities_and_flags(capsys):
    out = fills(row(100, "C", 0.9, 1.1, bsz=None, asz=pd.NA, oi=None)).iloc[0]
    assert out.c_buy == near(0.575) and out.c_sell == near(0.575)
    assert out.fill_buy == near(1.0575) and out.fill_sell == near(0.9425)
    assert out.fill_buy_unknown0 == near(1.045) and out.fill_sell_unknown0 == near(0.955)
    assert out.fill_buy_unknown1 == near(1.07) and out.fill_sell_unknown1 == near(0.93)
    assert (out.thin_size_buy, out.thin_size_sell, out.thin_oi) == (0.5, 0.5, 0.5)
    assert (out.unknown_ask_size, out.unknown_bid_size, out.unknown_oi) == (True, True, True)
    assert "bid_size=1 ask_size=1 open_interest=1" in capsys.readouterr().err
    zero = fills(row(100, "C", 0.9, 1.1, bsz=0, asz=0, oi=0)).iloc[0]
    assert zero.c_buy == near(0.70) and zero.c_sell == near(0.70)
    assert zero.fill_buy_unknown0 == near(1.07) and zero.fill_buy_unknown1 == near(1.07)
    assert (zero.thin_size_buy, zero.thin_size_sell, zero.thin_oi) == (1.0, 1.0, 1.0)
    assert not (zero.unknown_ask_size or zero.unknown_bid_size or zero.unknown_oi)


def test_one_sided_unknown_size_ask_side_known():
    # bid size unknown, ask size 50: the unknown bid side takes the 0.10 penalty on the buy side
    out = fills(row(100, "C", 0.9, 1.1, bsz=None, asz=50, oi=1000)).iloc[0]
    assert out.V == near(1.0)
    assert out.c_buy == near(0.45) and out.c_sell == near(0.525)
    assert out.fill_buy == near(1.045) and out.fill_sell == near(0.9475)
    assert out.fill_buy_unknown0 == near(1.045) and out.fill_sell_unknown0 == near(0.955)
    assert out.fill_buy_unknown1 == near(1.045) and out.fill_sell_unknown1 == near(0.94)
    assert out.fill_buy_midfloor == near(1.045) and out.fill_sell_midfloor == near(0.9475)
    assert out.unknown_bid_size and not out.unknown_ask_size


def test_one_sided_unknown_size_bid_side_known():
    # mirror of the case above: ask size unknown, bid size 50
    out = fills(row(100, "C", 0.9, 1.1, bsz=50, asz=None, oi=1000)).iloc[0]
    assert out.V == near(1.0)
    assert out.c_buy == near(0.525) and out.c_sell == near(0.45)
    assert out.fill_buy == near(1.0525) and out.fill_sell == near(0.955)
    assert out.fill_buy_unknown0 == near(1.045) and out.fill_sell_unknown0 == near(0.955)
    assert out.fill_buy_unknown1 == near(1.06) and out.fill_sell_unknown1 == near(0.955)
    assert out.unknown_ask_size and not out.unknown_bid_size


SENSITIVITY_FILLS = ("fill_buy_unknown0", "fill_sell_unknown0", "fill_buy_unknown1", "fill_sell_unknown1",
                     "fill_buy_midfloor", "fill_sell_midfloor")


def test_unpriceable_crossed_quote_is_nan():
    # finite spread but crossed quote (bid above ask): status unpriceable, all price-derived columns NaN
    out = fills(row(100, "C", 1.2, 0.8, spot=101.0)).iloc[0]
    assert out.status == "unpriceable"
    for c in ("s", "c_buy", "c_sell", "fill_buy", "fill_sell") + SENSITIVITY_FILLS:
        assert pd.isna(out[c]), c


def test_no_fair_value_row_keeps_concession_and_nan_fills():
    # spot NaN with expiry == obs_date: no fair value, but the liquid sizes still give c = 0.45
    out = fills(row(100, "C", 0.9, 1.1, spot=NAN)).iloc[0]
    assert out.status == "no_fair_value"
    assert out.c_buy == near(0.45) and out.c_sell == near(0.45)
    for c in ("fill_buy", "fill_sell") + SENSITIVITY_FILLS:
        assert pd.isna(out[c]), c


def test_v_away_from_mid_and_midfloor_floors_both_sides():
    out = fills(row(100, "C", 0.9, 1.1, spot=100.9),   # V = bid: buy below mid
                row(100, "P", 0.9, 1.1, spot=98.9))    # V = ask: sell above mid
    call, put = out.iloc[0], out.iloc[1]
    assert call.V == near(0.9) and put.V == near(1.1)
    assert call.fill_buy == near(0.99) and call.fill_buy_midfloor == near(1.0)
    assert call.fill_sell == near(0.9) and call.fill_sell_midfloor == near(0.9)
    assert put.fill_sell == near(1.01) and put.fill_sell_midfloor == near(1.0)
    assert put.fill_buy == near(1.1) and put.fill_buy_midfloor == near(1.1)


def test_repaired_quote_uses_effective_quote_and_unpriceable_repaired_is_nan():
    out = fills(row(100, "C", 2.0, 2.0, spot=102.0, rep=True)).iloc[0]
    assert (out.eff_bid, out.eff_ask, out.V) == (near(1.5), near(2.5), near(2.0))
    assert out.s == near(0.5) and out.c_buy == near(0.75) and out.c_sell == near(0.75)
    assert out.fill_buy == near(2.375) and out.fill_sell == near(1.625)
    zero = fills(row(100, "C", 0.0, 0.0, spot=102.0, rep=True)).iloc[0]
    assert zero.status == "unpriceable"
    assert all(pd.isna(zero[c]) for c in ("s", "c_buy", "c_sell", "fill_buy", "fill_sell", "fill_buy_midfloor"))


def test_missing_liquidity_column_refuses_before_pricing(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("leg_values must not run")

    monkeypatch.setattr(fvl, "leg_values", boom)
    quotes = pd.DataFrame([row(100, "C", 0.9, 1.1)]).drop(columns=["open_interest", "ask_size"])
    with pytest.raises(FairValueRefusal) as err:
        ff.leg_fills(quotes, no_vols())
    assert err.value.details == {"reason": "column_missing", "table": "quotes",
                                 "columns": ["ask_size", "open_interest"]}
    assert err.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert str(err.value) == f"EXPERIMENT_IDENTITY_CONFLICT: {err.value.details['reason']}"


def test_row_order_independent_and_inputs_not_mutated():
    quotes = pd.DataFrame([
        row(100, "C", 0.9, 1.1, spot=101.0), row(100, "P", 0.75, 1.25, spot=99.0, bsz=None),
        row(95, "C", 5.9, 6.1, spot=101.0, oi=50), row(105, "P", 0.9, 1.1, spot=101.0, expiry=EXP, rate=NAN),
    ])
    before = quotes.copy()
    out = ff.leg_fills(quotes, no_vols())
    pd.testing.assert_frame_equal(quotes, before)
    shuffled = ff.leg_fills(quotes.sample(frac=1, random_state=7), no_vols())
    pd.testing.assert_frame_equal(shuffled.sort_index(), out.sort_index())
    # hand-computed rows, looked up by original index in the shuffled output
    # row 0: call K100 spot 101, quote 0.9/1.1, all liquidity known: c 0.45
    assert shuffled.loc[0, "c_buy"] == near(0.45)
    assert shuffled.loc[0, "fill_buy"] == near(1.045) and shuffled.loc[0, "fill_sell"] == near(0.955)
    # row 1: put K100 spot 99, quote 0.75/1.25, bid size unknown: c_buy 0.75 (sell side has unknown term)
    assert shuffled.loc[1, "c_buy"] == near(0.75)
    assert shuffled.loc[1, "fill_buy"] == near(1.1875) and shuffled.loc[1, "fill_sell"] == near(0.79375)
    # row 2: call K95 spot 101, quote 5.9/6.1 (V 6), OI 50 thin: c 23/60
    assert shuffled.loc[2, "c_buy"] == near(23 / 60)
    assert shuffled.loc[2, "fill_buy"] == near(6.0383333333, tol=1e-9)
    assert shuffled.loc[2, "fill_sell"] == near(5.9616666667, tol=1e-9)
    # row 3: put K105, T > 0 with NaN rate: status rate_unknown, fills NaN, c 0.45 from liquid sizes
    assert shuffled.loc[3, "status"] == "rate_unknown"
    assert pd.isna(shuffled.loc[3, "fill_buy"]) and pd.isna(shuffled.loc[3, "fill_sell"])
    assert shuffled.loc[3, "c_buy"] == near(0.45) and shuffled.loc[3, "c_sell"] == near(0.45)
    assert list(out.columns) == list(fvl.OUT_COLUMNS) + ff.FILL_COLUMNS
