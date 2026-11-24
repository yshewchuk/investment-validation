import pandas as pd
import pytest

from experiments import size_pnl_ladder as lad
from experiments import size_pnl_shortvol as sv
from experiments import size_width_fallback as fbm
from experiments.size_pnl_pool import build_pool
from experiments.size_pnl_shortvol import Refusal
from experiments.size_width_account import simulate_account
from experiments.size_width_band import PHIS, rho_ladder, write_bands
from experiments.size_width_fallback import FallbackSweep, accounts, floor_ok, recover, recovery, split_trades, t9
from experiments.size_width_sweep import ALPHA, E_ARM, Sweep, scaled
from tests.v2.research.test_size_pnl_shortvol import _tables, chunk_of, events_for, put_legs, rec

DRAW = 200
NAME = "CND-PS"
PRED = 6.0
# Hand oracle for the coarse-ladder scene (spot 100, strikes 60..140 step 10, mid 1 + 0.01*(k-100)^2, spread 0.2,
# every exit quote locked at (1.0, 1.0) on rung 1, alpha 0.5). At the recovered rho 2.55 the traded legs, one contract
# each, are BUY 80 and BUY 120 (mid 5.0) and SELL 90 and SELL 110 (mid 2.0); the atm leg is a reference leg (qty 0).
# Entry: 2 * 5.0 - 2 * 2.0 = 6.0 per share. Exit: sell the wings 2 * 1.0, buy back the inners 2 * 1.0, so 0.0.
# ret = 0.0 / 6.0 - 1 = -1.0. pnl_dollars_mid = -6.0 * 100 * lots(6.0); lots(6.0) = floor(10_000 / 600) = floor(16.67) = 16.
ENTRY_PER_SHARE = 6.0
RET_RECOVERED = -1.0
PNL_MID = -9600.0  # -6.0 * 100 * 16
SECURED_PER_CONTRACT = 20_000.0  # two short puts at 90 and 110, strike * 100 each
COARSE = range(60, 141, 10)  # CND-PS up1 (100 + 3 at k=1) snaps onto atm: ladder_too_coarse
WIDE = (0.5, 3.0)
WINDOW = (pd.Timestamp("2024-06-01"), pd.Timestamp("2024-07-01"))
PRIOR = [f"P{i}" for i in range(6)]
CUR = [f"C{i}" for i in range(6)]
POOL = build_pool(*_tables(), floor=None)


def _leg(side, strike, qty=1.0):
    return {"name": "x", "strike": float(strike), "side": side, "qty": qty, "idx": 0}


def _coarse(pred=PRED):
    chunk = chunk_of(("A", {}), strikes=COARSE)
    events = events_for("A", pred=pred)
    return chunk, events, sv.simulate(chunk, events, POOL, draws=DRAW)


def _frame(chunk, events, sims):
    return sv.score_events(chunk, scaled(events, 1.0), sims, NAME, alpha=ALPHA)


def _small_window(monkeypatch):
    real = sv.trailing_cutoff
    monkeypatch.setattr(sv, "trailing_cutoff",
                        lambda history, as_of, **kw: real(history, as_of, **{**kw, "min_window": 1}))


def _scene():
    chunk = chunk_of(*[(e, {}) for e in PRIOR + CUR], strikes=COARSE)
    prior = events_for(*PRIOR, date="2024-05-10")
    cur = events_for(*CUR, date="2024-06-10")
    sessions = {e: (pd.Timestamp("2024-05-15"), pd.Timestamp("2024-05-25")) for e in PRIOR}
    sessions.update({e: (pd.Timestamp("2024-06-11"), pd.Timestamp("2024-06-21")) for e in CUR})
    return chunk, prior, cur, sessions


MCAP_BY_TIER = {"ge_10B": 20e9, "1B_10B": 3e9, "lt_1B": 5e8}
# disjoint id groups per tier; the ids are the PRIOR / CUR ids of _scene, split across the three market-cap tiers
PRIOR_BY_TIER = {"ge_10B": PRIOR[:3], "1B_10B": PRIOR[3:5], "lt_1B": PRIOR[5:]}
CUR_BY_TIER = {"ge_10B": CUR[:3], "1B_10B": CUR[3:5], "lt_1B": CUR[5:]}


def _tiered_scene():
    """_scene with the same chunk, dates and sessions, but events of mixed market caps: one events_for call per tier."""
    chunk = chunk_of(*[(e, {}) for e in PRIOR + CUR], strikes=COARSE)
    prior = pd.concat([events_for(*PRIOR_BY_TIER[t], date="2024-05-10", mcap=MCAP_BY_TIER[t]) for t in MCAP_BY_TIER],
                      ignore_index=True)
    cur = pd.concat([events_for(*CUR_BY_TIER[t], date="2024-06-10", mcap=MCAP_BY_TIER[t]) for t in MCAP_BY_TIER],
                    ignore_index=True)
    sessions = {e: (pd.Timestamp("2024-05-15"), pd.Timestamp("2024-05-25")) for e in PRIOR}
    sessions.update({e: (pd.Timestamp("2024-06-11"), pd.Timestamp("2024-06-21")) for e in CUR})
    return chunk, prior, cur, sessions


def _run(sweep, scene):
    chunk, prior, cur, sessions = scene
    sweep.add_month(chunk, prior, POOL, output=False, sessions=sessions, draws=DRAW)
    sweep.add_month(chunk, cur, POOL, output=True, sessions=sessions, draws=DRAW)
    return sweep


def _factors():
    return {n: 1.0 for n in sv.NAMES}


def _body(bands):
    """A valid frozen tuning_band body: {name: (lo, hi) or None}; None is a no_positive_peak structure. Same band at every phi."""
    out = {}
    for n in sv.NAMES:
        if bands[n] is None:
            cell = {"lo": None, "hi": None, "peak_k": None, "flags": ["no_positive_peak"]}
        else:
            lo, hi = bands[n]
            cell = {"lo": lo, "hi": hi, "peak_k": min(max(1.0, lo), hi), "flags": []}
        out[n] = {"primary": "B", "curves": {"B": {str(p): cell for p in PHIS}}}
    return out


NARROW = (0.9, 1.1)


def _phi_body(name, bands_by_phi):
    """A valid frozen body where ``name`` has its own (lo, hi) at every phi in ``bands_by_phi``; the other structures are no_positive_peak."""
    body = _body({n: None for n in sv.NAMES})
    cells = {str(p): {"lo": lo, "hi": hi, "peak_k": min(max(1.0, lo), hi), "flags": []} for p, (lo, hi) in bands_by_phi.items()}
    body[name] = {"primary": "B", "curves": {"B": cells}}
    return body


def _fb(tmp_path, band=WIDE):
    """A FallbackSweep whose only band is NAME's ``band``, read back from the tuning_band receipt written under ``tmp_path``."""
    bands = {n: None for n in sv.NAMES}
    bands[NAME] = band
    write_bands(tmp_path, _body(bands), "h1")
    return FallbackSweep(_factors(), tmp_path, "h1", 0.9)


# ---- the floor re-check ----------------------------------------------------

def test_floor_ok_on_hand_built_positions():
    fly = [_leg(sv.BUY, 90), _leg(sv.SELL, 100, 2.0), _leg(sv.BUY, 110)]  # 90 - 200 + 110 = 0 at S = 0
    assert floor_ok(fly) is True
    flipped = [_leg(sv.SELL, 90), _leg(sv.BUY, 100, 2.0), _leg(sv.SELL, 110)]  # -10 at S = 100
    assert floor_ok(flipped) is False
    assert floor_ok([_leg(sv.BUY, 100)]) is True
    assert floor_ok([_leg(sv.SELL, 100)]) is False


# ---- recover ----------------------------------------------------------------

def test_recover_takes_the_nearest_feasible_rho_on_the_coarse_ladder():
    chunk, events, sims = _coarse()
    frame = _frame(chunk, events, sims)
    assert list(frame["reason"]) == ["ladder_too_coarse"]
    out, found = recover(chunk, events, sims, NAME, frame, 1.0, rho_ladder(*WIDE, 1.0))
    assert found == {"A": 2.55}  # 2.5 ties up2 onto up1 (110); from 2.55 up1 -> 110 and up2 -> 120
    legs, _ = sv.resolve(chunk, 0, NAME, PRED * 2.55)
    assert {leg["name"]: leg["strike"] for leg in legs} == {"atm": 100.0, "up1": 110.0, "up2": 120.0,
                                                            "dn1": 90.0, "dn2": 80.0}
    row = out.iloc[0]
    assert row["status"] == "scored" and row["k"] == 2.55
    # long 80 and 120 at mid 5.0, short 90 and 110 at mid 2.0: debit 2 * 5 - 2 * 2 = 6.0
    assert row["entry_cost"] == pytest.approx(6.0)
    direct = sv.score_events(chunk, scaled(events, 2.55), sims, NAME, alpha=ALPHA).iloc[0]
    for col in ("entry_cost", "exp_pnl_sim", "width"):
        assert row[col] == pytest.approx(direct[col])


def test_recover_leaves_the_event_declined_when_the_band_holds_no_feasible_rho():
    chunk, events, sims = _coarse()
    out, found = recover(chunk, events, sims, NAME, _frame(chunk, events, sims), 1.0, rho_ladder(0.9, 1.1, 1.0))
    assert found == {"A": None}
    assert out.iloc[0]["reason"] == "ladder_too_coarse" and out.iloc[0]["k"] == 1.0


def test_a_row_declined_for_another_reason_is_never_retried():
    chunk = chunk_of(("A", {}), strikes=COARSE)
    events = events_for("A", pred=50.0)  # width 0.25 > WIDTH_MAX: width_declined
    sims = sv.simulate(chunk, events, POOL, draws=DRAW)
    frame = _frame(chunk, events, sims)
    assert list(frame["reason"]) == ["width_declined"]
    out, found = recover(chunk, events, sims, NAME, frame, 1.0, rho_ladder(*WIDE, 1.0))
    assert found == {}
    assert out.iloc[0]["reason"] == "width_declined" and out.iloc[0]["k"] == 1.0


def test_a_trial_failing_the_floor_is_skipped(monkeypatch):
    chunk, events, sims = _coarse()
    frame, seen = _frame(chunk, events, sims), []
    real = fbm.floor_ok

    def first_fails(legs):
        seen.append(1)
        return len(seen) > 1 and real(legs)

    monkeypatch.setattr(fbm, "floor_ok", first_fails)
    _out, found = recover(chunk, events, sims, NAME, frame, 1.0, rho_ladder(*WIDE, 1.0))
    assert found == {"A": 2.6}  # 2.55 fails the floor; 2.6 is the next feasible rho in ladder order


def test_recover_never_looks_at_exp_pnl_sim(monkeypatch):
    chunk, events, sims = _coarse()
    frame = _frame(chunk, events, sims)
    real = sv.score_events

    def tilted(chunk_, events_, sims_, name, **kw):  # the farther rho (pred 15.6) scores higher
        out = real(chunk_, events_, sims_, name, **kw)
        out["exp_pnl_sim"] = 99.0 if float(events_["pred_move"].iloc[0]) > 15.45 else 0.0
        return out

    monkeypatch.setattr(sv, "score_events", tilted)
    out, found = recover(chunk, events, sims, NAME, frame, 1.0, rho_ladder(*WIDE, 1.0))
    assert found == {"A": 2.55}
    assert out.iloc[0]["exp_pnl_sim"] == 0.0


# ---- FallbackSweep ----------------------------------------------------------

def test_fallback_run_takes_the_recovered_events_at_their_effective_k(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    fb = _run(_fb(tmp_path), _scene())
    rows = fb.rows([(NAME, E_ARM)], WINDOW)
    assert not rows.empty
    assert (rows["k"] == 2.55).all()
    assert (rows["entry_date"] == pd.Timestamp("2024-06-11")).all()
    assert (rows["exit_date"] == pd.Timestamp("2024-06-21")).all()
    assert {t["event_id"] for t in fb.trials} == set(PRIOR + CUR)
    assert all(t["rho"] == pytest.approx(2.55) for t in fb.trials)
    hist = pd.concat(fb.scored[(NAME, E_ARM)], ignore_index=True)
    assert set(hist["status"]) == {"scored"}


def test_without_a_band_nothing_is_recovered_or_taken(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    fb = _run(_fb(tmp_path, band=None), _scene())
    assert fb.rows([(NAME, E_ARM)], WINDOW).empty
    assert fb.trials == []


def test_factors_are_validated(tmp_path):
    write_bands(tmp_path, _body({n: None for n in sv.NAMES}), "h1")
    with pytest.raises(ValueError):
        FallbackSweep({n: 1.0 for n in sv.NAMES if n != NAME}, tmp_path, "h1", 0.9)
    with pytest.raises(ValueError):
        FallbackSweep({**_factors(), NAME: 1.1}, tmp_path, "h1", 0.9)


def test_a_malformed_body_is_never_published(tmp_path):
    body = _body({n: WIDE for n in sv.NAMES})
    del body[NAME]
    with pytest.raises(ValueError, match="malformed"):
        write_bands(tmp_path, body, "h1")


def test_no_receipt_refuses_the_sweep(tmp_path):
    with pytest.raises(Refusal) as err:
        FallbackSweep(_factors(), tmp_path, "h1", 0.9)
    assert err.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert err.value.details["reason"] == "tuning_receipt_missing"


def test_a_receipt_for_other_inputs_refuses_the_sweep(tmp_path):
    write_bands(tmp_path, _body({n: WIDE for n in sv.NAMES}), "h1")
    with pytest.raises(Refusal) as err:
        FallbackSweep(_factors(), tmp_path, "h2", 0.9)
    assert err.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert err.value.details["reason"] == "tuning_receipt_changed"


def test_phi_outside_phis_is_refused(tmp_path):
    write_bands(tmp_path, _body({n: WIDE for n in sv.NAMES}), "h1")
    with pytest.raises(ValueError, match="phi"):
        FallbackSweep(_factors(), tmp_path, "h1", 0.5)


def test_no_positive_peak_structure_gets_no_band_and_no_trials(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    fb = _run(_fb(tmp_path, band=None), _scene())
    assert fb.bands[NAME] is None
    assert fb.trials == []


def test_valid_receipt_gives_the_band_range(tmp_path):
    assert _fb(tmp_path).bands[NAME] == WIDE


def test_phi_selects_its_own_band_and_its_own_recovery(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    by_phi = {0.80: WIDE, 0.90: WIDE, 0.95: NARROW}
    write_bands(tmp_path / "p080", _phi_body(NAME, by_phi), "h1")
    write_bands(tmp_path / "p095", _phi_body(NAME, by_phi), "h1")
    fb80 = FallbackSweep(_factors(), tmp_path / "p080", "h1", 0.8)
    fb95 = FallbackSweep(_factors(), tmp_path / "p095", "h1", 0.95)
    assert fb80.bands[NAME] == (0.5, 3.0)
    assert fb95.bands[NAME] == (0.9, 1.1)
    fb80, fb95 = _run(fb80, _scene()), _run(fb95, _scene())
    # phi 0.8 holds the wide band: the coarse events are recovered at rho 2.55
    assert {t["event_id"] for t in fb80.trials} == set(PRIOR + CUR)
    assert all(t["rho"] == pytest.approx(2.55) for t in fb80.trials)
    assert len(fb80.rows([(NAME, E_ARM)], WINDOW)) == 3
    # phi 0.95 holds the narrow band, which has no feasible rho on this ladder: every trial stays infeasible
    assert {t["event_id"] for t in fb95.trials} == set(PRIOR + CUR)
    assert all(t["rho"] is None for t in fb95.trials)
    assert fb95.rows([(NAME, E_ARM)], WINDOW).empty


def test_a_non_unit_factor_recovers_at_the_first_feasible_rho():
    # k = 1.25, band (0.5, 3.0). rho_ladder(0.5, 3.0, 1.25) spans rho in [0.5/1.25, 3.0/1.25] = [0.4, 2.4] in multiples of 0.05.
    # The width multiplier k * rho must be >= 2.55 (the k = 1 recovery at 2.55 is the first feasible rung there), so
    # rho >= 2.55 / 1.25 = 2.04. 2.04 is not a multiple of 0.05; ladder order is nearest 1 first, and every rung below
    # 2.05 (k * rho <= 2.5) is infeasible, so the first feasible rung is rho = 2.05, with k * rho = 2.5625.
    chunk, events, sims = _coarse()
    k = 1.25
    frame = sv.score_events(chunk, scaled(events, k), sims, NAME, alpha=ALPHA)
    assert list(frame["reason"]) == ["ladder_too_coarse"]
    out, found = recover(chunk, events, sims, NAME, frame, k, rho_ladder(*WIDE, k))
    assert found == {"A": 2.05}
    recovered = out[out["event_id"].astype(str).isin([e for e, rho in found.items() if rho is not None])]
    assert len(recovered) == 1 and recovered["status"].tolist() == ["scored"]
    assert recovered["k"].tolist() == pytest.approx([k * 2.05])


# ---- recovery, T9, accounts ---------------------------------------------------

def test_recovery_counts_the_recovered_trades_and_no_others(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    fb = _run(_fb(tmp_path), _scene())
    base = _run(Sweep(), _scene())
    rows = fb.rows([(NAME, E_ARM)], WINDOW)
    assert not rows.empty
    trades = split_trades(fb, WINDOW)
    mine = trades[trades["structure"] == NAME]
    assert set(mine["recovered"]) == {True} and set(mine["tier"]) == {"ge_10B"}
    # hand oracle (see the module constants): three taken CUR trades, each entry 6.0, exit 0.0, ret -1.0, pnl -9600.0.
    # The gate takes 3 of the 6 CUR events (C0, C2, C4); the other three CUR events are scored but not taken by the
    # trailing cutoff, so they never enter WINDOW's trades.
    assert len(mine) == 3
    assert mine["entry_cost"].tolist() == pytest.approx([ENTRY_PER_SHARE] * len(mine))
    assert mine["ret"].tolist() == pytest.approx([RET_RECOVERED] * len(mine))
    assert mine["pnl_dollars_mid"].tolist() == pytest.approx([PNL_MID] * len(mine))
    others = trades[trades["structure"] != NAME]
    assert not others.empty and not others["recovered"].astype(bool).any()
    # taken_ret is the mean ret of ALL no-fallback trades of the window, pooled over structures, from the raw base rows
    base_rows = base.rows([(n, 1.0) for n in sv.NAMES], WINDOW)
    assert not base_rows.empty
    raw_taken_ret = float((base_rows["exit_value_mid"] / base_rows["entry_cost"] - 1.0).mean())
    assert pd.notna(raw_taken_ret)
    got = recovery(fb, base, WINDOW)
    assert got["recovered"] == 3 == len(rows)  # no other structure contributes a recovered trade
    assert got["recovered_ret"] == pytest.approx(RET_RECOVERED)
    assert got["taken_ret"] == pytest.approx(raw_taken_ret)


def test_an_empty_fallback_sweep_gives_unique_columns_and_no_recovery(tmp_path):
    fb, base = _fb(tmp_path), Sweep()
    frame = split_trades(fb, WINDOW)
    assert frame.empty and frame.columns.is_unique
    assert recovery(fb, base, WINDOW) == {"recovered": 0, "recovered_ret": None, "taken_ret": None}
    with pytest.raises(ValueError, match="base"):
        recovery(fb, None, WINDOW)


def test_t9_counts_by_hand(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    fb = _run(_fb(tmp_path), _scene())
    base = _run(Sweep(), _scene())
    rows = t9(fb, base, 0.9, WINDOW)
    assert len(rows) == 1
    row = rows[0]
    assert (row["phi"], row["structure"], row["tier"]) == (0.9, NAME, "ge_10B")
    assert (row["lo"], row["hi"]) == WIDE
    assert row["declined"] == 6 and row["recovered"] == 6 and row["fallback_infeasible"] == 0
    assert row["share"] == pytest.approx(1.0)
    assert row["rho"] == {"2.55": 6}
    # hand counts: the gate takes 3 of the 6 CUR events (C0, C2, C4), which exit 2024-06-21, inside WINDOW; the other
    # three CUR events are scored but not taken by the trailing cutoff; the PRIOR events exit 2024-05-25, outside it
    assert row["recovered_trades"] == 3 == len(fb.rows([(NAME, E_ARM)], WINDOW))
    assert row["recovered_ret"] == pytest.approx(RET_RECOVERED)
    assert row["recovered_pnl"] == pytest.approx(PNL_MID)
    # the taken stats come from the no-fallback base, which takes no NAME trade in WINDOW (see the accounts test)
    assert row["taken_trades"] == 0 and row["taken_ret"] is None
    assert row["taken_trades"] == len(base.rows([(NAME, 1.0)], WINDOW))


# ---- T9 taken side with a non-zero base baseline ------------------------------
# Dense ladder (strikes 80..120 step 1): CND-PS at PRED 6.0 resolves at k = 1 with legs 94 / 97 / 103 / 106.
# Entry at mid: BUY 94 and 106 at 1.36, SELL 97 and 103 at 1.09, so cost = 2 * 1.36 - 2 * 1.09 = 0.54 per share.
# Exit locked at (1.0, 1.0): 2 * 1.0 - 2 * 1.0 = 0.0, so ret = 0.0 / 0.54 - 1 = -1.0.
# lots(0.54) = floor(10_000 / (0.54 * 100)) = floor(10_000 / 54) = floor(185.18) = 185.
# pnl_dollars_mid = -0.54 * 100 * 185 = -9990.0.
DENSE = range(80, 121)
DENSE_PRIOR = [f"D{i}" for i in range(6)]
DENSE_CUR = [f"E{i}" for i in range(6)]
DENSE_PNL_MID = -9990.0  # -0.54 * 100 * 185
# N = the number of E events the no-fallback base takes for NAME in WINDOW. From a run: base takes E2..E5, each with
# entry_cost 0.54, exit_value_mid 0.0, pnl_dollars_mid -9990.0 (matching DENSE_PNL_MID).
DENSE_TAKEN_N = 4


def _mixed_scene():
    """D / E events on the dense ladder, P / C events on the coarse ladder, all in one chunk; mcap 20e9 (ge_10B)."""
    strikes = {**{e: DENSE for e in DENSE_PRIOR + DENSE_CUR}, **{e: COARSE for e in PRIOR + CUR}}
    ids = DENSE_PRIOR + PRIOR + DENSE_CUR + CUR
    chunk = lad._pack([rec(e, 100.0, put_legs(strikes=strikes[e])) for e in ids], {})
    prior = events_for(*DENSE_PRIOR, *PRIOR, date="2024-05-10")
    cur = events_for(*DENSE_CUR, *CUR, date="2024-06-10")
    sessions = {e: (pd.Timestamp("2024-05-15"), pd.Timestamp("2024-05-25")) for e in DENSE_PRIOR + PRIOR}
    sessions.update({e: (pd.Timestamp("2024-06-11"), pd.Timestamp("2024-06-21")) for e in DENSE_CUR + CUR})
    return chunk, prior, cur, sessions


def test_t9_taken_side_has_a_nonzero_base_baseline(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    assert DENSE_TAKEN_N is not None, "set DENSE_TAKEN_N from one run of base.rows([(NAME, 1.0)], WINDOW)"
    scene = _mixed_scene()
    fb = _run(_fb(tmp_path), scene)
    base = _run(Sweep(), scene)
    # the base takes NAME only on the dense E events: the coarse P / C events are declined without fallback
    base_rows = base.rows([(NAME, 1.0)], WINDOW)
    assert len(base_rows) == DENSE_TAKEN_N > 0
    assert set(base_rows["event_id"]) == {"E2", "E3", "E4", "E5"}
    assert base_rows["entry_cost"].tolist() == pytest.approx([0.54] * 4)
    assert (base_rows["exit_value_mid"] == 0.0).all()
    assert base_rows["pnl_dollars_mid"].tolist() == pytest.approx([DENSE_PNL_MID] * DENSE_TAKEN_N)
    rows = [r for r in t9(fb, base, 0.9, WINDOW) if r["structure"] == NAME]
    assert len(rows) == 1
    row = rows[0]
    # taken side: from the base sweep, non-zero count, hand values
    assert row["taken_trades"] == DENSE_TAKEN_N
    assert row["taken_ret"] == pytest.approx(-1.0)
    assert row["taken_pnl"] == pytest.approx(DENSE_PNL_MID)
    # recovered side: the fallback RECOVERS the coarse C events (trials) but the gate does not TAKE them
    # (the dense events' scores raise the cutoff), so recovered has 0 trades
    assert {t["event_id"] for t in fb.trials if t["rho"] is not None} >= set(CUR)
    assert row["recovered"] == 6 and row["declined"] == 6
    assert row["recovered_trades"] == 0 and row["recovered_ret"] is None


def test_t9_groups_the_trials_by_market_cap_tier(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    fb = _run(_fb(tmp_path), _tiered_scene())
    base = _run(Sweep(), _tiered_scene())
    rows = t9(fb, base, 0.9, WINDOW)
    # one row per (structure, tier): exactly the three expected tier labels, one row each, all for NAME
    assert len(rows) == 3
    assert sorted(r["tier"] for r in rows) == sorted(MCAP_BY_TIER)
    assert all((r["phi"], r["structure"]) == (0.9, NAME) for r in rows)
    by_tier = {r["tier"]: r for r in rows}
    taken = fb.rows([(NAME, E_ARM)], WINDOW)
    taken_ids = set(taken["event_id"].astype(str)) if not taken.empty else set()
    base_rows = base.rows([(NAME, 1.0)], WINDOW)  # the no-fallback taken trades, split by the hand-built tier ids
    total = 0
    for t, ids in CUR_BY_TIER.items():
        row = by_tier[t]
        n = len(ids)  # trials in WINDOW are the CUR events; PRIOR exits outside it
        total += n
        assert row["declined"] == n and row["recovered"] == n and row["fallback_infeasible"] == 0
        assert row["rho"] == {"2.55": n}
        assert (row["lo"], row["hi"]) == WIDE
        taken_here = taken_ids & set(ids)
        if taken_here:  # the universe and spread gate may leave a tier untaken; only check tiers with taken trades
            assert row["recovered_trades"] == len(taken_here)
        base_here = base_rows[base_rows["event_id"].astype(str).isin(ids)]
        assert row["taken_trades"] == len(base_here)
    # a tier-less grouping would give one row with declined == total; the per-tier rows must sum to every trial
    assert sum(r["declined"] for r in rows) == total == len(CUR)


def test_t9_counts_the_fallback_infeasible_events(tmp_path):
    fb = _run(_fb(tmp_path, (0.9, 1.1)), _scene())  # the narrow band holds no feasible rho on this ladder
    base = _run(Sweep(), _scene())
    row = t9(fb, base, 0.9, WINDOW)[0]
    assert row["declined"] == 6 and row["recovered"] == 0 and row["fallback_infeasible"] == 6
    assert row["share"] == 0.0 and row["rho"] == {} and row["recovered_trades"] == 0


def test_t9_taken_comes_from_the_base_sweep_and_not_from_the_band(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    base = _run(Sweep(), _tiered_scene())
    wide = _run(_fb(tmp_path / "wide", WIDE), _tiered_scene())
    narrow = _run(_fb(tmp_path / "narrow", NARROW), _tiered_scene())  # same base, different band: different fb, same taken baseline
    # hand-built expectation: the no-fallback trades of NAME in WINDOW, counted per tier through the CUR ids of each tier
    base_rows = base.rows([(NAME, 1.0)], WINDOW)
    taken_of = lambda rows: {(r["tier"], r["taken_trades"], r["taken_ret"], r["taken_pnl"]) for r in rows}
    for fb in (wide, narrow):
        rows = {r["tier"]: r for r in t9(fb, base, 0.9, WINDOW)}
        assert set(rows) == set(MCAP_BY_TIER)
        for t, ids in CUR_BY_TIER.items():
            exp = base_rows[base_rows["event_id"].astype(str).isin(ids)]
            assert rows[t]["taken_trades"] == len(exp)
            if exp.empty:
                assert rows[t]["taken_ret"] is None and rows[t]["taken_pnl"] is None
            else:
                assert rows[t]["taken_ret"] == pytest.approx(float((exp["exit_value_mid"] / exp["entry_cost"] - 1.0).mean()))
                assert rows[t]["taken_pnl"] == pytest.approx(float(exp["pnl_dollars_mid"].mean()))
    # the band changes the recovered side only: the taken side is identical for both fallback sweeps
    assert taken_of(t9(wide, base, 0.9, WINDOW)) == taken_of(t9(narrow, base, 0.9, WINDOW))


def test_t9_refuses_a_missing_base(tmp_path):
    with pytest.raises(ValueError, match="t9 needs the no-fallback base sweep"):
        t9(_fb(tmp_path), None, 0.9, WINDOW)


def test_t9_refuses_a_phi_that_does_not_match_the_sweep(tmp_path):
    fb = _fb(tmp_path)  # built at phi 0.9
    base = _run(Sweep(), _scene())
    with pytest.raises(ValueError, match="does not match"):
        t9(fb, base, 0.8, WINDOW)


def test_accounts_base_takes_six_and_fallback_adds_the_recovered_trades(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    fb = _run(_fb(tmp_path), _scene())
    base = _run(Sweep(), _scene())
    got = accounts(fb, base, {"jun": WINDOW})["jun"]
    base_rows = base.rows([(n, 1.0) for n in sv.NAMES], WINDOW)
    fb_rows = fb.rows([(n, E_ARM) for n in sv.NAMES], WINDOW)
    assert got["base"] == pytest.approx(simulate_account(base_rows, "mid")["total_return"])
    assert got["fallback"] == pytest.approx(simulate_account(fb_rows, "mid")["total_return"])
    pairs = lambda rows: set(zip(rows["structure"], rows["event_id"].astype(str)))
    added = pairs(fb_rows) - pairs(base_rows)
    recovered = {(NAME, str(e)) for e in fb.rows([(NAME, E_ARM)], WINDOW)["event_id"]}
    assert added == recovered and added
    assert not any(s == NAME for s, _ in pairs(base_rows))


def test_accounts_hand_oracle_for_the_three_recovered_trades(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    fb = _run(_fb(tmp_path), _scene())
    mine = fb.rows([(NAME, E_ARM)], WINDOW)
    assert len(mine) == 3
    assert (mine["entry_date"] == pd.Timestamp("2024-06-11")).all()
    assert (mine["exit_date"] == pd.Timestamp("2024-06-21")).all()
    assert mine["secured_per_contract"].tolist() == pytest.approx([SECURED_PER_CONTRACT] * len(mine))
    # Hand walk of margin.simulate under POLICY (start 200,000, cap 0.66, target_share 0.25, expensive_first).
    # Each trade: cost 6.0 * 100 = 600 per contract, value 0.0 (exit_value_mid), secured 20,000 per contract.
    # Trade n (n = 1..3) sees equity 200,000 (cash plus cost at book), headroom 132,000 - 20,000 * (n - 1).
    # want = max(0.25 * headroom / 20,000, 1) is 1.65, 1.4, 1.15, so each trade is funded at floor(want) = 1 contract.
    # The cap never binds: 3 * 20,000 = 60,000 <= 132,000.
    # The account pays 3 * -600 = -1,800 and nothing returns before 2024-06-21, so final = 200,000 - 1,800 = 198,200.
    got = simulate_account(mine, "mid")
    assert got["wanted"] == 3 and got["funded"] == 3 and got["peak_open"] == 3
    assert got["final_equity"] == pytest.approx(198_200.0)
    assert got["total_return"] == pytest.approx(-0.009)
    assert got["max_drawdown"] == pytest.approx(-0.009)  # one exit date: curve [200,000, 198,200]
