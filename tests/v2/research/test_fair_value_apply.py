"""Tests for fill model F slice F6c part 2: model A constants applied to leg fills, structure entry and exit.

Synthetic inputs only (ticker XYZ). The constants document is built here with made-up values (not the fitted ones) and
its id is computed here, so the oracle does not use the functions under test. Expected numbers are hand-written.
"""

import copy
import hashlib
import json

import pandas as pd
import pytest

from experiments import fair_value_apply as fva, fair_value_fills as ff, fair_value_legs as fvl, fair_value_run as run
from experiments.fair_value_legs import FairValueRefusal

OBS, EXP = pd.Timestamp("2026-11-25"), pd.Timestamp("2026-12-25")
NAN = float("nan")
PRIMARY = "primary"
CONSTANTS = {"C_BASE": 0.30, "C_SPREAD": 0.40, "C_THIN_SIZE": 0.20, "C_THIN_OI": 0.12, "SPREAD_SCALE": 0.50,
             "THIN_SIZE_BELOW": 10, "THIN_OI_BELOW": 100, "UNKNOWN_THIN": 0.5, "GUARD_FILL_FRACTION": 0.25}
SPREAD050_CONSTANTS = {**CONSTANTS, "C_SPREAD": 0.5}
FILLS = [c for c in ff.FILL_COLUMNS if c.startswith("fill_")]
PRICE_COLUMNS = ["s", "c_buy", "c_sell"] + FILLS


def near(x, tol=1e-9):
    return pytest.approx(x, abs=tol)


def doc(**overrides):
    """A fresh document with a self-consistent id: sha256 of the canonical body, computed here by hand."""
    body = {"anchor": "mid", "constants": {**CONSTANTS, **overrides},
            "sensitivity_variants": {"spread050": {"C_SPREAD": 0.5}}}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"constants_id": digest, **body}


DOC = doc()
PIN = DOC["constants_id"]


def row(K, right, bid, ask, spot=101.0, expiry=OBS, rep=False, oi=1000, bsz=50, asz=50, rate=0.04):
    """One quote row. Expiry OBS gives T = 0 (V = intrinsic clamped to [bid, ask])."""
    return {"ticker": "XYZ", "obs_date": OBS, "expiry": expiry, "strike": float(K), "right": right, "bid": bid, "ask": ask,
            "iv": 0.3, "spot": spot, "quote_repaired": rep, "rate": rate, "open_interest": oi, "bid_size": bsz,
            "ask_size": asz, "borrow30": NAN, "borrow2yr": NAN, "divAmt": 0.0, "divDate": pd.NaT}


def no_vols():
    df = pd.DataFrame({"strike": pd.Series((), dtype=float), "smvVol": pd.Series((), dtype=float)})
    df.insert(0, "ticker", "XYZ")
    df.insert(1, "obs_date", OBS)
    df.insert(2, "expiry", OBS)
    return df


def apply(*rows, variant=PRIMARY):
    return fva.leg_fills(pd.DataFrame(list(rows)), no_vols(), DOC, PIN, variant)


def legs_of(*rows, sides, qtys, variant=PRIMARY):
    return apply(*rows, variant=variant).assign(side=list(sides), qty=list(qtys))


def exit_of(legs, variant=PRIMARY):
    return fva.structure_exit(legs, DOC, PIN, variant)


def refusal(fn, *args):
    with pytest.raises(FairValueRefusal) as exc:
        fn(*args)
    return exc.value.details


def load_reason(document, pin=PIN, variant=PRIMARY):
    return refusal(fva.load_constants, document, pin, variant)["reason"]


# --- 1. load_constants --------------------------------------------------------------------------------------------------

def test_primary_returns_the_nine_constants():
    assert fva.load_constants(doc(), PIN, PRIMARY) == CONSTANTS


def test_spread050_changes_only_c_spread():
    out = fva.load_constants(doc(), PIN, "spread050")
    assert out == SPREAD050_CONSTANTS
    assert {k for k in out if out[k] != CONSTANTS[k]} == {"C_SPREAD"}


def test_input_document_is_not_mutated():
    d = doc()
    snapshot = copy.deepcopy(d)
    fva.load_constants(d, PIN, "spread050")
    fva.load_constants(d, PIN, PRIMARY)
    assert d == snapshot


# --- 2. constants_invalid ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("document", [None, [doc()]], ids=["none", "list"])
def test_constants_invalid_when_not_a_dict(document):
    assert load_reason(document) == "constants_invalid"


@pytest.mark.parametrize("edit", [lambda d: d.pop("constants"), lambda d: d.update(constants=[1, 2])],
                         ids=["constants_missing", "constants_not_a_dict"])
def test_constants_invalid_when_constants_is_missing_or_not_a_dict(edit):
    d = doc()
    edit(d)
    assert load_reason(d) == "constants_invalid"


MISSING = object()


@pytest.mark.parametrize("value", [MISSING, True, -0.1, NAN, float("inf"), 10**400, "0.3"],
                         ids=["missing", "true", "negative", "nan", "inf", "huge_int", "string"])
def test_constants_invalid_for_a_bad_fitted_value(value):
    d = doc()
    if value is MISSING:
        del d["constants"]["C_BASE"]
    else:
        d["constants"]["C_BASE"] = value
    assert load_reason(d) == "constants_invalid"


def test_constants_invalid_for_a_changed_fixed_constant():
    assert load_reason(doc(THIN_OI_BELOW=99)) == "constants_invalid"


def test_constants_invalid_for_an_extra_constant_key():
    assert load_reason(doc(EXTRA_KEY=1.0)) == "constants_invalid"


def test_constants_invalid_for_a_changed_anchor():
    d = doc()
    d["anchor"] = "spot"
    assert load_reason(d) == "constants_invalid"


def test_constants_invalid_for_changed_sensitivity_variants():
    d = doc()
    d["sensitivity_variants"] = {"spread050": {"C_SPREAD": 0.6}}
    assert load_reason(d) == "constants_invalid"


def test_constants_invalid_for_an_extra_top_level_key():
    d = doc()
    d["note"] = "x"
    assert load_reason(d) == "constants_invalid"


# --- 3. constants_id_mismatch -------------------------------------------------------------------------------------------

def test_constants_id_mismatch_for_a_wrong_caller_pin():
    assert load_reason(doc(), pin="0" * 64) == "constants_id_mismatch"


def test_constants_id_mismatch_for_a_wrong_document_id():
    d = doc()
    d["constants_id"] = "0" * 64
    assert load_reason(d) == "constants_id_mismatch"


def test_constants_id_mismatch_for_a_missing_document_id():
    d = doc()
    del d["constants_id"]
    assert load_reason(d) == "constants_id_mismatch"


def test_constants_id_mismatch_for_a_fitted_value_changed_without_recomputing_the_id():
    d = doc()
    d["constants"]["C_BASE"] = 0.31
    assert load_reason(d, pin=PIN) == "constants_id_mismatch"


# --- 4. variant_unknown ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["spread025", None, pd.NA, ["primary"]], ids=["unknown_name", "none", "pd_na", "list"])
def test_variant_unknown(variant):
    assert load_reason(doc(), variant=variant) == "variant_unknown"


# --- 5. refusal order --------------------------------------------------------------------------------------------------

def test_invalid_document_wins_over_pin_and_variant():
    assert load_reason(None, pin="0" * 64, variant="spread025") == "constants_invalid"


def test_pin_mismatch_wins_over_variant():
    assert load_reason(doc(), pin="0" * 64, variant="spread025") == "constants_id_mismatch"


def test_leg_fills_refuses_bad_document_before_pricing(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("pricing must not run")

    monkeypatch.setattr(fvl, "leg_values", boom)
    quotes = pd.DataFrame([row(100, "C", 0.9, 1.1)]).drop(columns=["open_interest"])
    assert refusal(fva.leg_fills, quotes, no_vols(), None, PIN, PRIMARY) == {"reason": "constants_invalid"}


def test_structure_exit_refuses_bad_document_before_binding():
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["buy"], qtys=[1.0]).drop(columns=["constants_id"])
    assert refusal(fva.structure_exit, legs, None, PIN, PRIMARY)["reason"] == "constants_invalid"


def test_structure_exit_missing_leg_columns_refuse_before_binding():
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["buy"], qtys=[1.0]).drop(columns=["side", "constants_id", "variant"])
    assert refusal(exit_of, legs)["reason"] == "column_missing"


# --- 6. leg_fills, hand-computed (spot 101, K 100 call, T = 0: V = mid = 1) ---------------------------------------------

def test_known_answer_6a():
    out = apply(row(100, "C", 0.9, 1.1, spot=101.0)).iloc[0]
    # mid 1, half .1, s .2, spread term .4: c = .30 + .40 * .4 = .46; buy 1 + .46 * .1, sell 1 - .46 * .1
    assert out.c_buy == near(0.46) and out.c_sell == near(0.46)
    assert out.fill_buy == near(1.046) and out.fill_sell == near(0.954)


def test_known_answer_6a_spread050():
    out = apply(row(100, "C", 0.9, 1.1, spot=101.0), variant="spread050").iloc[0]
    # c = .30 + .5 * .4 = .50; buy 1 + .05, sell 1 - .05
    assert out.c_buy == near(0.50) and out.fill_buy == near(1.05) and out.fill_sell == near(0.95)
    assert out.variant == "spread050"


def test_known_answer_6b_thin_ask_size_and_thin_oi():
    out = apply(row(100, "C", 0.9, 1.1, spot=101.0, asz=5, oi=50)).iloc[0]
    # ask size 5 is thin (+.20) and OI 50 is thin (+.12): c_buy = .46 + .20 + .12 = .78; the sell side takes only OI: .46 + .12 = .58
    assert out.c_buy == near(0.78) and out.fill_buy == near(1.078)
    assert out.c_sell == near(0.58) and out.fill_sell == near(0.942)


def test_known_answer_6c_unknown_ask_size():
    out = apply(row(100, "C", 0.9, 1.1, spot=101.0, asz=NAN)).iloc[0]
    # unknown ask size is thin 0.5: c_buy = .46 + .20 * .5 = .56; unknown0 (0) gives .46; unknown1 (1) gives .66
    assert out.c_buy == near(0.56) and out.fill_buy == near(1.056)
    assert out.fill_buy_unknown0 == near(1.046) and out.fill_buy_unknown1 == near(1.066)
    assert out.fill_sell == near(0.954)
    assert out.fill_sell_unknown0 == near(0.954) and out.fill_sell_unknown1 == near(0.954)


def test_known_answer_6c_unknown_open_interest_and_bid_size():
    oi = apply(row(100, "C", 0.9, 1.1, spot=101.0, oi=NAN)).iloc[0]
    # unknown OI is thin 0.5: c = .46 + .12 * .5 = .52 on both sides; unknown0 .46; unknown1 .58
    assert oi.c_buy == near(0.52) and oi.fill_buy == near(1.052) and oi.fill_sell == near(0.948)
    assert oi.fill_buy_unknown0 == near(1.046) and oi.fill_sell_unknown0 == near(0.954)
    assert oi.fill_buy_unknown1 == near(1.058) and oi.fill_sell_unknown1 == near(0.942)
    bid = apply(row(100, "C", 0.9, 1.1, spot=101.0, bsz=NAN)).iloc[0]
    # unknown bid size is thin 0.5: c_sell = .46 + .20 * .5 = .56; the buy side does not move
    assert bid.c_sell == near(0.56) and bid.fill_sell == near(0.944) and bid.fill_buy == near(1.046)
    assert bid.fill_sell_unknown0 == near(0.954) and bid.fill_sell_unknown1 == near(0.934)
    assert bid.fill_buy_unknown1 == near(1.046)


def test_known_answer_6d_capped_at_one():
    out = apply(row(100, "C", 0.5, 1.5, spot=101.0, asz=5, bsz=5, oi=50)).iloc[0]
    # s 1 gives term 1: .30 + .40 + .20 + .12 = 1.02 on each side, capped at 1; buy 1.5, sell .5
    assert out.c_buy == near(1.0) and out.c_sell == near(1.0)
    assert out.fill_buy == near(1.5) and out.fill_sell == near(0.5)


def test_known_answer_6e_repaired_quote():
    out = apply(row(100, "C", 2.0, 2.0, spot=102.0, rep=True)).iloc[0]
    # effective quote 1.5 / 2.5: mid 2, half .5, s .5, term 1, c = .30 + .40 = .70; buy 2 + .35, sell 2 - .35
    assert (out.eff_bid, out.eff_ask) == (near(1.5), near(2.5))
    assert out.s == near(0.5) and out.c_buy == near(0.70) and out.c_sell == near(0.70)
    assert out.fill_buy == near(2.35) and out.fill_sell == near(1.65)


def test_v_is_not_used_6f():
    at_mid = apply(row(100, "C", 0.9, 1.1, spot=101.0)).iloc[0]
    away = apply(row(100, "C", 0.9, 1.1, spot=100.0)).iloc[0]  # V clamps to the bid 0.9
    assert at_mid.V == near(1.0) and away.V == near(0.9)
    assert away.fill_buy == near(1.046) and away.fill_sell == near(0.954)


def test_unpriceable_quote_is_all_nan_6g():
    out = apply(row(100, "C", 1.2, 0.8, spot=101.0)).iloc[0]
    assert out.status == "unpriceable"
    assert all(pd.isna(out[c]) for c in PRICE_COLUMNS)


@pytest.mark.parametrize("quote,status", [
    (row(95, "P", 0.9, 1.1, spot=101.0, expiry=EXP, rate=NAN), "rate_unknown"),
    (row(100, "C", 0.9, 1.1, spot=NAN), "no_fair_value"),
], ids=["rate_unknown", "no_fair_value"])
def test_fallback_statuses_keep_concession_and_nan_fills_6g(quote, status):
    out = apply(quote).iloc[0]
    assert out.status == status
    assert out.c_buy == near(0.46) and out.c_sell == near(0.46)  # s .2, liquid sizes: .30 + .40 * .4
    assert all(pd.isna(out[c]) for c in FILLS)


def test_midfloor_equals_base_and_other_columns_match_f_6h():
    quotes = pd.DataFrame([row(100, "C", 0.9, 1.1, spot=101.0), row(100, "P", 0.9, 1.1, spot=99.0, asz=5)])
    out = fva.leg_fills(quotes, no_vols(), DOC, PIN, "spread050")
    base = ff.leg_fills(quotes, no_vols())
    assert (out.fill_buy_midfloor == out.fill_buy).all() and (out.fill_sell_midfloor == out.fill_sell).all()
    assert list(out.columns) == list(base.columns) + ["constants_id", "variant"]
    assert (out.constants_id == PIN).all() and (out.variant == "spread050").all()
    same = [c for c in base.columns if not c.startswith("fill_") and c not in ("c_buy", "c_sell")]
    pd.testing.assert_frame_equal(out[same], base[same])


# --- 7. structure_entry -------------------------------------------------------------------------------------------------

def test_entry_enters_single_buy_leg():
    res = fva.structure_entry(legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["buy"], qtys=[1.0]))
    assert res.outcome == "enter" and res.skip_reason is None
    assert res.debit_mid == near(1.0) and res.debit_fair == near(1.0) and res.debit_F == near(1.046)


def test_entry_fill_over_mid_hand_values():
    # sell row: bid .72 ask .88, spot 100.8: mid .8, half .08, s .2, c .46, sell fill .8 - .46 * .08 = .7632
    # the sell leg is a put so its strike key differs from the call; its V clamps to the bid, which the guard does not read
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0), row(100, "P", 0.72, 0.88, spot=100.8),
                   sides=["buy", "sell"], qtys=[1.0, 1.0])
    res = fva.structure_entry(legs)
    # debit_mid 1 - .8 = .2; debit 1.046 - .7632 = .2828; .2828 - .2 = .0828 > .25 * .2 = .05
    assert res.outcome == "skip" and res.skip_reason == "fill_over_mid"
    assert res.debit_mid == near(0.2) and res.debit_F == near(0.2828)


def test_entry_boundary_is_strict():
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["buy"], qtys=[1.0])
    at_edge = fva.structure_entry(legs.assign(fill_buy=1.25))  # debit 1.25 - 1.0 = .25, not greater than .25
    assert at_edge.outcome == "enter" and at_edge.skip_reason is None
    over = fva.structure_entry(legs.assign(fill_buy=1.2500001))
    assert over.outcome == "skip" and over.skip_reason == "fill_over_mid"


def test_entry_enters_when_only_debit_minus_fair_is_large():
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["buy"], qtys=[1.0]).assign(V=0.5)
    # debit_F 1.046 - V .5 = .546 > .125 would trip F's rule; over mid it is only .046 <= .25: enter
    res = fva.structure_entry(legs)
    assert res.outcome == "enter" and res.skip_reason is None


def test_entry_skips_fill_over_mid_when_v_is_high():
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["buy"], qtys=[1.0]).assign(V=1.2, fill_buy=1.3)
    # F: 1.3 - 1.2 = .1 <= .3 passes; ours: 1.3 - 1.0 = .3 > .25 over the mid
    res = fva.structure_entry(legs)
    assert res.outcome == "skip" and res.skip_reason == "fill_over_mid"


def test_not_a_debit_from_debit_mid_wins_over_fill_over_mid():
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["buy"], qtys=[1.0]).assign(eff_bid=0.0, eff_ask=0.0)
    res = fva.structure_entry(legs)
    assert res.skip_reason == "not_a_debit" and res.debit_mid == near(0.0)


def test_not_a_debit_from_debit_fair_wins_over_fill_over_mid():
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["buy"], qtys=[1.0]).assign(V=0.0, fill_buy=1.5)
    res = fva.structure_entry(legs)
    # fill over mid would hold (1.5 - 1.0 = .5 > .25): not_a_debit wins
    assert res.skip_reason == "not_a_debit" and res.debit_fair == near(0.0)


def test_spread_over_debit():
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["buy"], qtys=[1.0]).assign(eff_bid=0.5, eff_ask=1.5, fill_buy=0.9)
    # debit_mid 1, debit .9 (not over mid), spread_sum 1.0 > .9
    res = fva.structure_entry(legs)
    assert res.skip_reason == "spread_over_debit" and res.spread_sum == near(1.0)


def test_non_ok_leg_is_undetermined():
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0), row(95, "P", 0.9, 1.1, spot=101.0, expiry=EXP, rate=NAN),
                   sides=["buy", "sell"], qtys=[1.0, 1.0])
    res = fva.structure_entry(legs)
    assert res.outcome == "undetermined" and res.skip_reason is None and res.debit_F is None and res.debit_mid is None


# --- 8. structure_exit --------------------------------------------------------------------------------------------------

def test_long_leg_proceeds_are_the_sell_fill():
    ex = exit_of(legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["buy"], qtys=[1.0]))
    assert ex.outcome == "ok" and ex.proceeds_F == near(0.954)  # 1 - .46 * .1


def test_short_leg_proceeds_are_minus_the_buy_fill():
    ex = exit_of(legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["sell"], qtys=[1.0]))
    assert ex.proceeds_F == near(-1.046)  # buy back at 1 + .46 * .1


def test_spread050_long_leg():
    ex = exit_of(legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["buy"], qtys=[1.0], variant="spread050"), variant="spread050")
    assert ex.proceeds_F == near(0.95)  # 1 - .5 * .1


def test_two_legs_with_quantity_two():
    # call K100 spot 101 bought x2 (sell fill .954); put K100 spot 99 sold x1 (buy fill 1.046, V clamps to 1)
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0), row(100, "P", 0.9, 1.1, spot=99.0), sides=["buy", "sell"], qtys=[2.0, 1.0])
    # 2 * .954 - 1.046 = .862
    assert exit_of(legs).proceeds_F == near(0.862)


def test_negative_proceeds_are_flagged_and_kept():
    ex = exit_of(legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["sell"], qtys=[1.0]))
    assert ex.proceeds_F == near(-1.046) and ex.exit_negative is True


def test_unpriceable_exit_reports_none():
    legs = legs_of(row(100, "C", 1.2, 0.8, spot=101.0), sides=["buy"], qtys=[1.0])
    ex = exit_of(legs)
    assert ex.outcome == "unpriceable" and ex.proceeds_F is None and ex.exit_negative is False
    assert ex.sensitivity_proceeds == {v: None for v in ff.VARIANTS}


def test_rate_unknown_leg_with_known_sizes_falls_back_to_mid():
    ex = exit_of(legs_of(row(100, "C", 0.9, 1.1, spot=101.0, expiry=EXP, rate=NAN), sides=["buy"], qtys=[1.0]))
    # V = mid 1, sell at 1 - .46 * .1
    assert ex.outcome == "ok" and ex.exit_fair_mid is True and ex.proceeds_F == near(0.954)


def test_rate_unknown_leg_with_nan_bid_size_has_hand_sensitivities():
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0, expiry=EXP, rate=NAN, bsz=NAN), sides=["buy"], qtys=[1.0])
    ex = exit_of(legs)
    # sell c .30 + .40 * .4 + .20 * .5 = .56: 1 - .056 = .944; unknown0 (c .46) .954; unknown1 (c .66) .934
    assert ex.proceeds_F == near(0.944) and ex.exit_fair_mid is True
    assert ex.sensitivity_proceeds["unknown0"] == near(0.954)
    assert ex.sensitivity_proceeds["unknown1"] == near(0.934)
    assert ex.sensitivity_proceeds["midfloor"] == near(0.944)


def test_rate_unknown_leg_under_spread050_uses_half():
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0, expiry=EXP, rate=NAN), sides=["buy"], qtys=[1.0], variant="spread050")
    ex = exit_of(legs, variant="spread050")
    assert ex.proceeds_F == near(0.95)  # 1 - .5 * .1


# --- 9. binding refusals -------------------------------------------------------------------------------------------------

def test_dropped_binding_column_is_unbound():
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["buy"], qtys=[1.0])
    assert refusal(exit_of, legs.drop(columns=["constants_id"])) == {"reason": "legs_unbound", "columns": ["constants_id"]}
    assert refusal(exit_of, legs.drop(columns=["variant"])) == {"reason": "legs_unbound", "columns": ["variant"]}


def test_null_binding_row_is_unbound():
    f_table = ff.leg_fills(pd.DataFrame([row(100, "C", 0.9, 1.1, spot=101.0)]), no_vols())
    a_table = apply(row(100, "C", 0.9, 1.1, spot=101.0))
    legs = pd.concat([f_table, a_table], ignore_index=True).assign(side=["buy", "buy"], qty=[1.0, 1.0])
    assert refusal(exit_of, legs) == {"reason": "legs_unbound", "columns": ["constants_id", "variant"]}


def test_wrong_id_in_one_row_is_a_mismatch():
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0), row(100, "P", 0.9, 1.1, spot=99.0), sides=["buy", "sell"], qtys=[1.0, 1.0])
    legs.loc[1, "constants_id"] = "0" * 64
    assert refusal(exit_of, legs)["reason"] == "legs_constants_mismatch"


def test_spread050_priced_then_exited_with_primary_is_a_mismatch():
    legs = legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["buy"], qtys=[1.0], variant="spread050")
    assert refusal(exit_of, legs)["reason"] == "legs_constants_mismatch"


def test_mixed_variants_across_rows_is_a_mismatch():
    legs = pd.concat([legs_of(row(100, "C", 0.9, 1.1, spot=101.0), sides=["buy"], qtys=[1.0]),
                      legs_of(row(100, "P", 0.9, 1.1, spot=99.0), sides=["sell"], qtys=[1.0], variant="spread050")], ignore_index=True)
    assert refusal(exit_of, legs)["reason"] == "legs_constants_mismatch"


def test_unpriceable_unbound_leg_still_refuses():
    legs = legs_of(row(100, "C", 1.2, 0.8, spot=101.0), sides=["buy"], qtys=[1.0]).drop(columns=["variant"])
    assert refusal(exit_of, legs)["reason"] == "legs_unbound"


# --- 10. isolation from F ----------------------------------------------------------------------------------------------

def test_fvf_id_and_source_files_are_unchanged():
    assert run.fvf_id() == "735f3e6cb2fa65c303599152cfa6fead2d7950e9bcb367117d50ed25faffc0b7"
    assert "fair_value_apply.py" not in run.SOURCE_FILES


def test_f_output_is_unchanged_by_model_a_calls():
    quotes = pd.DataFrame([row(100, "C", 0.9, 1.1, spot=101.0), row(100, "P", 0.9, 1.1, spot=99.0)])
    snapshot = quotes.copy()
    before = ff.leg_fills(quotes, no_vols())
    legs = fva.leg_fills(quotes, no_vols(), DOC, PIN, "spread050").assign(side=["buy", "sell"], qty=[1.0, 1.0])
    exit_of(legs, variant="spread050")
    fva.structure_entry(legs)
    after = ff.leg_fills(quotes, no_vols())
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    assert after.iloc[0].fill_buy == near(1.045) and after.iloc[0].fill_sell == near(0.955)  # V 1, c .25 + .5 * .4 = .45
    pd.testing.assert_frame_equal(quotes, snapshot)
