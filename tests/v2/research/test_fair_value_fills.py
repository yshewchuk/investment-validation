"""Reference tests for fill model F slice F4: concession, sensitivities, structure entry and exit.

Synthetic inputs only (guides/fair_value_fill_design.md, sections 5-7, 10). Expiry == obs_date gives T = 0, so V is
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


def legs_of(*rows, sides, qtys):
    return fills(*rows).assign(side=list(sides), qty=list(qtys))


def base_legs():
    return legs_of(row(100, "C", 0.9, 1.1, spot=101.0), row(100, "P", 0.9, 1.1, spot=99.0),
                   sides=["buy", "sell"], qtys=[1.0, 1.0])


# (bid, ask, bid_size, ask_size, OI, c_buy, c_sell, fill_buy, fill_sell) for a call at spot 101, K 100: V = mid = 1
CONCESSION = [
    (1.0, 1.0, 50, 50, 1000, 0.25, 0.25, 1.0, 1.0),           # s = 0
    (0.975, 1.025, 50, 50, 1000, 0.30, 0.30, 1.0075, 0.9925),  # s = 0.05
    (0.9, 1.1, 50, 50, 1000, 0.45, 0.45, 1.045, 0.955),        # s = 0.20
    (0.75, 1.25, 50, 50, 1000, 0.75, 0.75, 1.1875, 0.8125),    # s = 0.50
    (0.5, 1.5, 50, 50, 1000, 0.75, 0.75, 1.375, 0.625),        # s = 1.00, capped at SPREAD_SCALE
    (0.75, 1.25, 9, 9, 99, 1.0, 1.0, 1.25, 0.75),              # s = 0.50, both thin: cap at 1
    (0.5, 1.5, 9, 9, 99, 1.0, 1.0, 1.5, 0.5),                  # s = 1.00, both thin: cap at 1
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
    entry = structure_spread(row(100, "C", 2.0, 2.0, spot=102.0, rep=True))
    assert entry.spread_sum == near(1.0)
    zero = fills(row(100, "C", 0.0, 0.0, spot=102.0, rep=True)).iloc[0]
    assert zero.status == "unpriceable"
    assert all(pd.isna(zero[c]) for c in ("s", "c_buy", "c_sell", "fill_buy", "fill_sell", "fill_buy_midfloor"))


def structure_spread(quote):
    return ff.structure_entry(legs_of(quote, sides=["buy"], qtys=[1.0]))


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


def test_entry_known_answer_qty_weighted_debits():
    call = row(100, "C", 0.9, 1.1, spot=101.0, asz=None)  # buy 2: unknown ask size, bid 50, OI 1000
    put = row(100, "P", 0.9, 1.1, spot=99.0)              # sell 1
    res = ff.structure_entry(legs_of(call, put, sides=["buy", "sell"], qtys=[2, 1]))
    assert res.outcome == "enter" and res.skip_reason is None
    assert res.debit_mid == near(1.0) and res.debit_fair == near(1.0)
    assert res.debit_F == near(1.15) and res.debit_F >= res.debit_fair
    assert res.spread_sum == near(0.6)
    assert res.sensitivity_debits["unknown0"] == near(1.135)
    assert res.sensitivity_debits["unknown1"] == near(1.165)
    assert res.sensitivity_debits["midfloor"] == near(1.15)
    buy_flags, sell_flags = res.leg_flags
    assert (buy_flags["thin_size"], buy_flags["unknown_size"], buy_flags["thin_oi"]) == (0.5, True, 0.0)
    assert (sell_flags["thin_size"], sell_flags["unknown_size"]) == (0.0, False)
    assert not any(f["exit_fair_mid"] for f in res.leg_flags)


def test_guard_first_match_and_each_reason():
    # fill_over_fair and spread_over_debit both hold; the first one wins.
    res = ff.structure_entry(legs_of(row(100, "C", 0.8, 1.2, spot=101.0), row(100, "P", 0.5, 0.7, spot=99.4),
                                     sides=["buy", "sell"], qtys=[1.0, 1.0]))
    assert res.outcome == "skip" and res.skip_reason == "fill_over_fair"
    assert res.debit_mid == near(0.4) and res.debit_fair == near(0.4)
    assert res.debit_F == near(0.588333, tol=1e-5) and res.spread_sum == near(0.6)
    # fill_over_fair case: call buy fills 1.13; put sell fills 13/24 (c = 7/12); no unknown or thin terms, so
    # all three sensitivity debits equal debit_F
    assert res.debit_mid == near(0.4) and res.debit_fair == near(0.4) and res.spread_sum == near(0.6)
    assert res.debit_F == near(0.588333, tol=1e-5)
    assert res.sensitivity_debits["unknown0"] == near(0.588333, tol=1e-5)
    assert res.sensitivity_debits["unknown1"] == near(0.588333, tol=1e-5)
    assert res.sensitivity_debits["midfloor"] == near(0.588333, tol=1e-5)

    # spread_over_debit alone: debit_fair 1, debit_F about 1.227, spread_sum 1.4
    res = ff.structure_entry(legs_of(row(100, "C", 9.65, 10.35, spot=110.0), row(100, "P", 8.65, 9.35, spot=91.0),
                                     sides=["buy", "sell"], qtys=[1.0, 1.0]))
    assert res.skip_reason == "spread_over_debit"
    assert res.debit_fair == near(1.0) and res.debit_F == near(1.226722, tol=1e-5)
    assert res.spread_sum == near(1.4)

    # credit structure: debit_mid = 1 - 5 = -4
    res = ff.structure_entry(legs_of(row(100, "P", 0.9, 1.1, spot=99.0), row(100, "C", 4.9, 5.1, spot=105.0),
                                     sides=["buy", "sell"], qtys=[1.0, 1.0]))
    assert res.skip_reason == "not_a_debit" and res.debit_mid == near(-4.0)

    # only debit_fair is at or below the threshold: V = bid = 0, mid 0.1
    res = ff.structure_entry(legs_of(row(100, "C", 0.0, 0.2, spot=99.0), sides=["buy"], qtys=[1.0]))
    assert res.skip_reason == "not_a_debit" and res.debit_mid == near(0.1) and res.debit_fair == near(0.0)

    # only debit_mid is at or below the threshold: buy V = 1, sell put V = 0 with mid 1
    res = ff.structure_entry(legs_of(row(100, "C", 0.9, 1.1, spot=101.0), row(100, "P", 0.0, 2.0, spot=101.0),
                                     sides=["buy", "sell"], qtys=[1.0, 1.0]))
    assert res.skip_reason == "not_a_debit" and res.debit_mid == near(0.0) and res.debit_fair == near(1.0)


def test_guard_boundary_is_strict():
    # buy call K 100 at spot 105 with quote 0/2: V = 2, debit_F = 2 = spread_sum, so the structure enters
    res = ff.structure_entry(legs_of(row(100, "C", 0.0, 2.0, spot=105.0), sides=["buy"], qtys=[1.0]))
    assert res.outcome == "enter" and res.skip_reason is None
    assert res.debit_F == res.spread_sum == near(2.0)
    assert res.debit_mid == near(1.0) and res.debit_fair == near(2.0)
    assert res.sensitivity_debits["unknown0"] == near(2.0)
    assert res.sensitivity_debits["unknown1"] == near(2.0)
    assert res.sensitivity_debits["midfloor"] == near(2.0)


def test_guard_exact_float_boundaries():
    m = ff.MIN_MEANINGFUL_COST
    n = np.nextafter(m, 1)
    assert ff._guard(1.0, 1.0, 1.25, 1.0) is None  # fill diff exactly 0.25 * debit_fair is not a skip
    assert ff._guard(1.0, 1.0, np.nextafter(1.25, 2), 1.0) == "fill_over_fair"
    assert ff._guard(1.0, 1.0, 1.0, np.nextafter(1.0, 2)) == "spread_over_debit"
    assert ff._guard(1.0, 1.0, 1.0, 1.0) is None
    assert ff._guard(m, 1.0, 1.0, 0.0) == "not_a_debit"
    assert ff._guard(1.0, m, 1.0, 0.0) == "not_a_debit"
    assert ff._guard(n, n, n, 0.0) is None


def test_undetermined_entry_when_a_leg_is_rate_unknown():
    ok = row(100, "C", 0.9, 1.1, spot=101.0)
    rate_unknown = row(95, "P", 0.9, 1.1, spot=101.0, expiry=EXP, rate=NAN)
    res = ff.structure_entry(legs_of(ok, rate_unknown, sides=["buy", "sell"], qtys=[1.0, 1.0]))
    assert res.outcome == "undetermined" and res.skip_reason is None
    assert res.debit_mid is None and res.debit_fair is None and res.debit_F is None and res.spread_sum is None
    assert res.sensitivity_debits == {v: None for v in ff.VARIANTS}
    assert [f["status"] for f in res.leg_flags] == ["ok", "rate_unknown"]


def test_exit_mirrors_entry_on_the_same_rows():
    legs = legs_of(
        row(100, "C", 0.9, 1.1, spot=101.0, asz=None, bsz=5, oi=50),  # buy 2: ask unknown, bid thin, OI thin
        row(100, "P", 0.75, 1.25, spot=99.0, bsz=None, asz=9),        # sell 1: bid unknown, ask thin
        row(95, "C", 5.9, 6.1, spot=101.0, oi=100),                   # buy 1: OI 100 is not thin
        sides=["buy", "sell", "buy"], qtys=[2, 1, 1])
    entry = ff.structure_entry(legs)
    assert entry.leg_flags[0]["thin_size"] == 0.5 and entry.leg_flags[0]["unknown_size"] is True
    mirror = ff.structure_exit(legs.assign(side=["sell", "buy", "sell"]))
    assert mirror.outcome == "ok"
    # entry: buy 2 @ 1.0625 (c 0.625), sell 1 @ 0.79375 (c 0.825), buy 1 @ 6.0283333 (c 17/60)
    assert entry.debit_F == near(7.3595833333, tol=1e-9)
    assert mirror.proceeds_F == near(-entry.debit_F)
    for v in ff.VARIANTS:
        assert mirror.sensitivity_proceeds[v] == near(-entry.sensitivity_debits[v])
    same = ff.structure_exit(legs)  # opening buy at exit hits the bid: flags are the bid-side ones
    assert same.leg_flags[0]["thin_size"] == 1.0 and same.leg_flags[0]["unknown_size"] is False


def test_exit_fair_mid_fallback_for_rate_unknown_leg():
    leg = legs_of(row(100, "C", 0.9, 1.1, spot=101.0, expiry=EXP, rate=NAN), sides=["buy"], qtys=[1.0])
    ex = ff.structure_exit(leg)
    assert ex.outcome == "ok" and ex.exit_fair_mid is True and ex.exit_negative is False
    assert ex.proceeds_F == near(0.955)  # V = mid 1, sell at 1 - 0.45 * 0.1
    assert ex.leg_flags[0]["exit_fair_mid"] is True and ex.leg_flags[0]["status"] == "rate_unknown"
    assert ex.sensitivity_proceeds["unknown0"] == near(0.955)
    assert ex.sensitivity_proceeds["unknown1"] == near(0.955)
    assert ex.sensitivity_proceeds["midfloor"] == near(0.955)


def test_exit_fair_mid_fallback_with_unknown_liquidity_sensitivities():
    leg = legs_of(row(100, "C", 0.9, 1.1, spot=101.0, expiry=EXP, rate=NAN, bsz=None, asz=None, oi=None),
                  sides=["buy"], qtys=[1.0])
    ex = ff.structure_exit(leg)
    assert ex.outcome == "ok" and ex.exit_fair_mid is True
    assert ex.proceeds_F == near(0.9425)  # V = mid 1, sell at 1 - 0.575 * 0.1
    assert ex.sensitivity_proceeds["unknown0"] == near(0.955)  # sell at 1 - 0.45 * 0.1
    assert ex.sensitivity_proceeds["unknown1"] == near(0.93)  # sell at 1 - 0.70 * 0.1
    assert ex.sensitivity_proceeds["midfloor"] == near(0.9425)  # min(sell fill 0.9425, mid 1)


def test_exit_fair_mid_fallback_for_no_fair_value_ignores_stored_fills():
    base = legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["sell"], qtys=[1.0])
    stale = base.assign(status="no_fair_value", V=NAN, fill_buy=9.99, fill_sell=9.99)
    ex = ff.structure_exit(stale)
    assert ex.proceeds_F == near(-1.045)  # opening sell buys back at 1 + 0.45 * 0.1
    assert ex.exit_negative is True and ex.exit_fair_mid is True
    assert ex.leg_flags[0]["status"] == "no_fair_value"
    assert ex.sensitivity_proceeds["unknown0"] == near(-1.045)
    assert ex.sensitivity_proceeds["unknown1"] == near(-1.045)
    assert ex.sensitivity_proceeds["midfloor"] == near(-1.045)


def test_exit_unpriceable_leg_gives_none_and_reports_fallback():
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0), row(95, "P", 0.0, 0.0, spot=101.0),
                   row(105, "C", 0.9, 1.1, spot=101.0, expiry=EXP, rate=NAN),
                   sides=["buy", "buy", "buy"], qtys=[1.0, 1.0, 1.0])
    ex = ff.structure_exit(legs)
    assert ex.outcome == "unpriceable" and ex.proceeds_F is None and ex.exit_negative is False
    assert ex.sensitivity_proceeds == {v: None for v in ff.VARIANTS}
    assert ex.exit_fair_mid is True
    assert [f["status"] for f in ex.leg_flags] == ["ok", "unpriceable", "rate_unknown"]


def test_negative_proceeds_kept_and_return_below_minus_one():
    ex = ff.structure_exit(legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["sell"], qtys=[1.0]))
    assert ex.proceeds_F == near(-1.045) and ex.exit_negative is True and ex.exit_fair_mid is False
    assert ff.return_on_debit(ex.proceeds_F, 1.0) == near(-2.045)
    assert ff.return_on_debit(2.0, 1.6) == near(0.25)


def test_exit_has_no_guard_for_a_wide_quote():
    ex = ff.structure_exit(legs_of(row(100, "C", 0.0, 2.0, spot=105.0), sides=["buy"], qtys=[1.0]))
    assert ex.outcome == "ok" and ex.proceeds_F == near(0.5)  # sell at 2 - 0.75 * 2
    assert ex.exit_fair_mid is False and ex.exit_negative is False
    for v in ff.VARIANTS:
        assert ex.sensitivity_proceeds[v] == near(0.5)
    assert ex.leg_flags[0]["exit_fair_mid"] is False and ex.leg_flags[0]["thin_size"] == 0.0
    assert ex.leg_flags[0]["unknown_size"] is False


@pytest.mark.parametrize("call", [ff.structure_entry, ff.structure_exit])
@pytest.mark.parametrize("bad", [
    lambda d: d.assign(side=["long", "sell"]),
    lambda d: d.assign(qty=[0.0, 1.0]),
    lambda d: d.assign(qty=[None, 1.0]),
    lambda d: d.assign(status=[NAN, "ok"]),
    lambda d: d.assign(qty=[-1.0, 1.0]),
    lambda d: d.assign(qty=[float("inf"), 1.0]),
    lambda d: d.assign(qty=[NAN, 1.0]),
    lambda d: d.assign(qty=pd.Series(["x", 1.0], dtype=object)),
    lambda d: d.assign(status=["bogus", "ok"]),
    lambda d: d.assign(side=[None, "sell"]),
    lambda d: d.iloc[:0],
])
def test_leg_invalid_refusals(call, bad):
    with pytest.raises(FairValueRefusal) as err:
        call(bad(base_legs()))
    assert err.value.details["reason"] == "leg_invalid"
    assert err.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert str(err.value) == f"EXPERIMENT_IDENTITY_CONFLICT: {err.value.details['reason']}"


@pytest.mark.parametrize("call", [ff.structure_entry, ff.structure_exit])
def test_missing_legs_column_refuses(call):
    with pytest.raises(FairValueRefusal) as err:
        call(base_legs().drop(columns=["side"]))
    assert err.value.details == {"reason": "column_missing", "table": "legs", "columns": ["side"]}
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
    assert list(out.columns) == list(fvl.OUT_COLUMNS) + ff.FILL_COLUMNS
    legs = out.assign(side=["buy", "sell", "buy", "sell"], qty=[1.0, 1.0, 2.0, 1.0])
    snapshot = legs.copy()
    ff.structure_entry(legs)
    ff.structure_exit(legs)
    pd.testing.assert_frame_equal(legs, snapshot)
