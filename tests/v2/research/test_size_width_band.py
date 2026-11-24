import pytest

import experiments.size_pnl_receipts as receipts
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_band import (PHIS, WINDOW, WINDOW_KEYS, adopt, band, band_range, build_bands, choose_phi,
                                         frozen_bands, rho_ladder, tier, write_bands)


def _curve(rets, *, unscored=()):
    """A tuning ``scores`` dict from {k: ret}; ``unscored`` ks are flagged not_scored."""
    return {str(k): {"ret": r, "se": 0.1, "trades": 400, "not_scored": k in unscored} for k, r in rets.items()}


# the design curve: peak 1.25 (ret 1.0)
CURVE = _curve({0.5: 0.0, 0.67: 0.2, 0.75: 0.4, 1.0: 0.8, 1.25: 1.0, 1.5: 0.9, 1.75: 0.5, 2.0: 0.2, 2.5: -0.1, 3.0: -0.3})
NEG = _curve({0.5: -0.2, 1.0: 0.0, 2.0: -0.5})


# ---- bands ----------------------------------------------------------------

def test_band_at_phi_090():
    # thr 0.9. Right: 1.5 (0.9 >= 0.9) passes, 1.75 (0.5) fails -> hi = 1.5 + 0.25*(0.9-0.9)/(0.9-0.5) = 1.5.
    # Left: 1.0 (0.8) fails -> lo = 1.25 + (1.0-1.25)*(1.0-0.9)/(1.0-0.8) = 1.125.
    out = band(CURVE, 0.90)
    assert out["peak_k"] == 1.25
    assert out["lo"] == pytest.approx(1.125)
    assert out["hi"] == pytest.approx(1.5)
    assert out["flags"] == []


def test_band_at_phi_080():
    # thr 0.8. Left: 1.0 (0.8 >= 0.8) passes, 0.75 (0.4) fails -> lo = 1.0 + (0.75-1.0)*(0.8-0.8)/(0.8-0.4) = 1.0.
    # Right: 1.5 (0.9) passes, 1.75 (0.5) fails -> hi = 1.5 + 0.25*(0.9-0.8)/(0.9-0.5) = 1.5625.
    out = band(CURVE, 0.80)
    assert out["lo"] == pytest.approx(1.0)
    assert out["hi"] == pytest.approx(1.5625)
    assert out["flags"] == []


def test_band_at_phi_095():
    # thr 0.95. Left: 1.0 (0.8) fails -> lo = 1.25 + (1.0-1.25)*(1.0-0.95)/(1.0-0.8) = 1.1875.
    # Right: 1.5 (0.9) fails -> hi = 1.25 + 0.25*(1.0-0.95)/(1.0-0.9) = 1.375.
    out = band(CURVE, 0.95)
    assert out["lo"] == pytest.approx(1.1875)
    assert out["hi"] == pytest.approx(1.375)


def test_band_open_when_the_peak_is_at_an_edge():
    rising = _curve({0.5: 0.1, 1.0: 0.2, 2.0: 0.5, 3.0: 1.0})
    out = band(rising, 0.80)  # hi runs off the grid at 3.0; lo: 2.0 (0.5) fails -> 3.0 + (2.0-3.0)*(1.0-0.8)/(1.0-0.5) = 2.6
    assert out["hi"] == 3.0 and out["lo"] == pytest.approx(2.6)
    assert out["flags"] == ["band_open"]
    falling = _curve({0.5: 1.0, 1.0: 0.5, 2.0: 0.2, 3.0: 0.1})  # lo runs off at 0.5; hi: 1.0 (0.5) fails -> 0.5 + 0.5*0.2/0.5 = 0.7
    out = band(falling, 0.80)
    assert out["lo"] == 0.5 and out["hi"] == pytest.approx(0.7)
    assert out["flags"] == ["band_open"]


def test_no_positive_peak_when_no_return_is_positive():
    assert band(NEG, 0.9) == {"lo": None, "hi": None, "peak_k": None, "flags": ["no_positive_peak"]}


def test_no_positive_peak_when_every_point_is_not_scored():
    all_off = _curve({0.5: 0.3, 1.0: 0.6, 2.0: 0.2}, unscored={0.5, 1.0, 2.0})
    assert band(all_off, 0.9) == {"lo": None, "hi": None, "peak_k": None, "flags": ["no_positive_peak"]}


def test_not_scored_neighbour_ends_the_band_without_interpolation():
    # peak 1.0 (ret 1.0); 1.25 is not scored, so hi stops at 1.0 (not open); lo: 0.75 (0.5) fails -> 1.0 + (-0.25)*0.2/0.5 = 0.9
    curve = _curve({0.75: 0.5, 1.0: 1.0, 1.25: 0.95, 1.5: 0.9}, unscored={1.25})
    out = band(curve, 0.80)
    assert out["hi"] == 1.0
    assert out["lo"] == pytest.approx(0.9)
    assert out["flags"] == []


def test_peak_tie_goes_to_the_k_nearest_one_then_the_smaller():
    assert band(_curve({0.9: 1.0, 1.1: 1.0}), 0.9)["peak_k"] == 0.9
    assert band(_curve({0.5: 1.0, 1.25: 1.0}), 0.9)["peak_k"] == 1.25


def test_peak_tie_at_equal_distance_from_one_takes_the_smaller_k():
    # 0.75 and 1.25 both sit 0.25 from 1 and tie at ret 1.0; 1.0 is lower, so the tie is between the two outer points
    curve = _curve({0.5: 0.2, 0.75: 1.0, 1.0: 0.5, 1.25: 1.0, 2.0: 0.1})
    assert band(curve, 0.9)["peak_k"] == 0.75
    assert band(dict(reversed(list(curve.items()))), 0.9)["peak_k"] == 0.75  # insertion order does not matter


# ---- build_bands and the receipt -------------------------------------------

def test_build_bands_primary_and_labels():
    only_b = build_bands({n: {"B": CURVE} for n in NAMES})
    assert all(entry["primary"] == "B" and set(entry["curves"]) == {"B"} for entry in only_b.values())
    with_c = build_bands({n: {"B": CURVE, "C": CURVE, "anchor": CURVE} for n in NAMES})
    assert all(entry["primary"] == "C" and set(entry["curves"]) == {"B", "C", "anchor"} for entry in with_c.values())
    assert set(with_c[NAMES[0]]["curves"]["C"]) == {str(p) for p in PHIS}


def test_build_bands_refuses_a_missing_structure():
    curves = {n: {"B": CURVE} for n in NAMES[1:]}
    with pytest.raises(ValueError):
        build_bands(curves)


def test_build_bands_requires_b_even_with_c():
    with pytest.raises(ValueError):
        build_bands({n: {"C": CURVE} for n in NAMES})
    with pytest.raises(ValueError):
        build_bands({n: {"C": CURVE, "anchor": CURVE} for n in NAMES})


def test_receipt_roundtrip_returns_the_body(tmp_path):
    bands = build_bands({n: {"B": CURVE} for n in NAMES})
    write_bands(tmp_path, bands, "h1")
    assert frozen_bands(tmp_path, "h1") == bands


def test_missing_receipt_refuses(tmp_path):
    with pytest.raises(Refusal) as exc:
        frozen_bands(tmp_path, "h1")
    assert exc.value.details["reason"] == "tuning_receipt_missing"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_other_inputs_hash_refuses(tmp_path):
    write_bands(tmp_path, build_bands({n: {"B": CURVE} for n in NAMES}), "h1")
    with pytest.raises(Refusal) as exc:
        frozen_bands(tmp_path, "h2")
    assert exc.value.details["reason"] == "tuning_receipt_changed"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_wrong_arm_refuses(tmp_path):
    bands = build_bands({n: {"B": CURVE} for n in NAMES})
    receipts.write_receipt(tmp_path / "receipts" / "tuning_band.json",
                           {"experiment": "EXP-189", "arm": "wrong", "inputs_hash": "h1", "window": WINDOW, "body": bands})
    with pytest.raises(Refusal) as exc:
        frozen_bands(tmp_path, "h1")
    assert exc.value.details["reason"] == "tuning_receipt_changed"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_wrong_experiment_refuses(tmp_path):
    bands = build_bands({n: {"B": CURVE} for n in NAMES})
    receipts.write_receipt(tmp_path / "receipts" / "tuning_band.json",
                           {"experiment": "EXP-188", "arm": "band", "inputs_hash": "h1", "window": WINDOW, "body": bands})
    with pytest.raises(Refusal) as exc:
        frozen_bands(tmp_path, "h1")
    assert exc.value.details["reason"] == "tuning_receipt_changed"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_wrong_window_refuses(tmp_path):
    bands = build_bands({n: {"B": CURVE} for n in NAMES})
    receipts.write_receipt(tmp_path / "receipts" / "tuning_band.json",
                           {"experiment": "EXP-189", "arm": "band", "inputs_hash": "h1", "window": "2025", "body": bands})
    with pytest.raises(Refusal) as exc:
        frozen_bands(tmp_path, "h1")
    assert exc.value.details["reason"] == "tuning_receipt_changed"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_body_missing_a_structure_refuses(tmp_path):
    bands = build_bands({n: {"B": CURVE} for n in NAMES})
    del bands[NAMES[0]]
    receipts.write_receipt(tmp_path / "receipts" / "tuning_band.json",
                           {"experiment": "EXP-189", "arm": "band", "inputs_hash": "h1", "window": WINDOW, "body": bands})
    with pytest.raises(Refusal) as exc:
        frozen_bands(tmp_path, "h1")
    assert exc.value.details["reason"] == "tuning_receipt_changed"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_receipt_cannot_be_rewritten_with_a_different_body(tmp_path):
    write_bands(tmp_path, build_bands({n: {"B": CURVE} for n in NAMES}), "h1")
    other = build_bands({n: {"B": NEG} for n in NAMES})
    with pytest.raises(Refusal) as exc:
        write_bands(tmp_path, other, "h1")
    assert exc.value.details["reason"] == "tuning_receipt_changed"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_write_bands_refuses_a_malformed_body(tmp_path):
    bands = build_bands({n: {"B": CURVE} for n in NAMES})
    del bands[NAMES[0]]
    for body in ({}, bands):
        with pytest.raises(ValueError, match="tuning_band body is malformed"):
            write_bands(tmp_path, body, "h1")
        assert not (tmp_path / "receipts" / "tuning_band.json").exists()


def test_band_range_returns_lo_hi_or_none():
    bands = build_bands({n: {"B": CURVE} for n in NAMES})
    assert band_range(bands, NAMES[0], 0.90) == pytest.approx((1.125, 1.5))
    bands = build_bands({n: {"B": NEG} for n in NAMES})
    assert band_range(bands, NAMES[0], 0.90) is None


def test_tier_boundaries_and_unknowns():
    assert tier(10e9) == "ge_10B" and tier(9.99e9) == "1B_10B"
    assert tier(1e9) == "1B_10B" and tier(999e6) == "lt_1B"
    assert tier(float("nan")) == "unknown" and tier(None) == "unknown"


# ---- rho ladder ------------------------------------------------------------

def test_rho_ladder_nearest_one_first():
    assert rho_ladder(0.8, 1.2, 1.0) == [1.05, 0.95, 1.1, 0.9, 1.15, 0.85, 1.2, 0.8]


def test_rho_ladder_divides_the_band_by_k():
    ladder = rho_ladder(1.0, 3.0, 2.0)  # rho in [0.5, 1.5]: 0.5..1.5 in steps of 0.05, minus 1.0
    assert len(ladder) == 20 and 1.0 not in ladder
    assert ladder[:2] == [1.05, 0.95]
    assert all(0.5 <= r <= 1.5 for r in ladder)


def test_rho_ladder_empty_without_a_step_inside():
    assert rho_ladder(0.97, 1.03, 1.0) == []


def test_rho_ladder_refuses_non_positive_k():
    for k in (0.0, -1.0):
        with pytest.raises(ValueError):
            rho_ladder(0.8, 1.2, k)


def test_rho_ladder_has_no_duplicates_and_no_one():
    ladder = rho_ladder(0.5, 1.5, 1.0)
    assert len(ladder) == len(set(ladder)) and 1.0 not in ladder


# ---- phi choice and adoption ----------------------------------------------

def _rec(recovered, rec_ret=0.5, taken_ret=0.4):
    return {"recovered": recovered, "recovered_ret": rec_ret, "taken_ret": taken_ret}


def test_choose_phi_takes_the_smallest_qualifying_phi():
    assert choose_phi({0.80: _rec(100), 0.90: _rec(100), 0.95: _rec(100)}) == {"phi": 0.80, "flags": []}


def test_choose_phi_floor_is_100():
    assert choose_phi({0.80: _rec(99), 0.90: _rec(100)}) == {"phi": 0.90, "flags": []}


def test_choose_phi_equal_rets_pass():
    assert choose_phi({0.80: _rec(100, rec_ret=0.5, taken_ret=0.5)}) == {"phi": 0.80, "flags": []}


def test_choose_phi_none_rets_fail():
    assert choose_phi({0.80: _rec(500, rec_ret=None), 0.90: _rec(500, taken_ret=None)}) == \
        {"phi": None, "flags": ["insufficient_recovery"]}


def test_choose_phi_none_qualify():
    assert choose_phi({0.80: _rec(10), 0.90: _rec(500, rec_ret=0.1, taken_ret=0.4)}) == \
        {"phi": None, "flags": ["insufficient_recovery"]}


ACCOUNTS = {w: {"fallback": 1.0, "base": 0.5} for w in WINDOW_KEYS}
ALPHA = {"fallback": 0.30, "base": 0.30}


def test_adopt_when_every_window_beats_and_alpha_holds():
    assert adopt(0.90, ACCOUNTS, ALPHA) == {"adopt": True, "phi": 0.90, "beats": True, "alpha_ok": True, "reason": None}


def test_adopt_refuses_a_window_that_only_ties():
    accounts = {**ACCOUNTS, "2024": {"fallback": 0.5, "base": 0.5}}
    assert adopt(0.90, accounts, ALPHA)["adopt"] is False


def test_alpha_margin_is_inclusive_at_005():
    assert adopt(0.90, ACCOUNTS, {"fallback": 0.35, "base": 0.30})["alpha_ok"] is True
    assert adopt(0.90, ACCOUNTS, {"fallback": 0.36, "base": 0.30})["alpha_ok"] is False
    assert adopt(0.90, ACCOUNTS, {"fallback": 0.36, "base": 0.30})["adopt"] is False


def test_alpha_none_refuses():
    assert adopt(0.90, ACCOUNTS, {"fallback": None, "base": 0.30})["adopt"] is False


def test_phi_none_refuses_with_reason():
    assert adopt(None, ACCOUNTS, ALPHA) == {"adopt": False, "phi": None, "beats": False, "alpha_ok": False,
                                            "reason": "insufficient_recovery"}


def test_missing_window_key_is_a_key_error():
    accounts = {w: ACCOUNTS[w] for w in ("2025", "2024")}
    with pytest.raises(KeyError):
        adopt(0.90, accounts, ALPHA)
