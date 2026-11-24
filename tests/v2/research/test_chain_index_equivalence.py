"""Equivalence battery: ``engine.chain_index`` twin against ``engine.structures.price_structure``.

The legacy path is the oracle and is never modified. Every chain here is
synthetic and seeded; nothing reads ``data/`` or any store. Each comparison
checks outcomes (values, types, flags and exception types and messages), not
just the happy path, and runs with ``ENGINE_CHAIN_INDEX_VERIFY=1`` throughout.
"""
from __future__ import annotations

import dataclasses
import gc
from datetime import date

import numpy as np
import pandas as pd
import pytest

from engine import chain_index as ci
from engine.chain_index import ChainSnapshotMutated, price_structure_indexed
from engine.fills import BEST, MID, WORST, FillModel
from engine.structures import (
    BUY,
    CALL,
    PUT,
    SELL,
    STRUCTURES,
    ChainSnapshot,
    ExpirySelector,
    LadderTooCoarse,
    LegSpec,
    StrikeSelector,
    Structure,
    StructureError,
    ctr5,
    price_structure,
    put_butterfly,
    put_butterfly_wide,
    put_calendar,
    put_condor,
    put_condor_strike,
    ramp7,
    straddle_through,
    twin_peak,
    twin_peak_5,
)

OBS = pd.Timestamp("2026-03-09")
EVENT = pd.Timestamp("2026-03-10")
DEFAULT_EXPIRIES = (
    (date(2026, 3, 10), 1),
    (date(2026, 3, 17), 8),
    (date(2026, 3, 24), 15),
    (date(2026, 4, 14), 36),
)
FILLS = {"WORST": WORST, "MID": MID, "BEST": BEST}
SESSIONS = ("AMC", "BMO", None)
ATM = StrikeSelector("atm")
FIRST_POST = ExpirySelector(kind="first_post_event")


@pytest.fixture(autouse=True)
def _verify_mode_on(monkeypatch):
    monkeypatch.setenv("ENGINE_CHAIN_INDEX_VERIFY", "1")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_chain(seed=0, spot=100.0, spacing=1.0, n_strikes=41, expiries=DEFAULT_EXPIRIES,
               rights="CP", delta=True, expiry_unit="ns", strikes=None):
    """Synthetic option chain. Strikes centre on ``spot`` unless ``strikes`` is given."""
    if strikes is None:
        base = round(spot / spacing) * spacing
        strikes = base + (np.arange(n_strikes) - n_strikes // 2) * spacing
    strikes = np.asarray(strikes, dtype=float)
    rng = np.random.default_rng(seed)
    call_delta = 1.0 / (1.0 + np.exp((strikes - spot) / (0.05 * spot)))
    frames = []
    for exp_date, dte in expiries:
        for right in rights:
            n = strikes.size
            bid = rng.uniform(0.05, 20.0, n)
            ask = bid + rng.uniform(0.02, 1.5, n)
            bid = np.where(rng.random(n) < 0.04, 0.0, bid)  # a few zero bids
            frames.append(pd.DataFrame({
                "expiry": np.full(n, np.datetime64(exp_date.isoformat(), expiry_unit)),
                "dte": np.full(n, int(dte), dtype=np.int64),
                "strike": strikes.copy(),
                "right": np.full(n, right, dtype=object),
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2.0,
                "iv": rng.uniform(0.15, 0.8, n),
                "delta": call_delta if right == "C" else call_delta - 1.0,
                "spot": np.full(n, float(spot)),
            }))
    out = pd.concat(frames, ignore_index=True)
    if not delta:
        out = out.drop(columns="delta")
    return out


def snap(rows, session="AMC", spot=None):
    return ChainSnapshot(
        ticker="T", obs_date=OBS, event_date=EVENT, rows=rows,
        spot=100.0 if spot is None else spot, session=session,
    )


def outcome(fn):
    try:
        return ("ok", fn())
    except Exception as exc:  # the battery compares every failure, whatever its type
        return ("err", type(exc), str(exc))


def _same(legacy, twin):
    assert legacy[0] == twin[0], f"status differs: legacy={legacy!r} twin={twin!r}"
    if legacy[0] == "err":
        assert legacy[1] is twin[1], f"exception type: {legacy[1]} vs {twin[1]}"
        assert legacy[2] == twin[2], f"message: {legacy[2]!r} vs {twin[2]!r}"
        return
    a, b = legacy[1], twin[1]
    assert type(a) is type(b)
    assert a.to_dict() == b.to_dict()
    assert repr(a) == repr(b)
    for field in dataclasses.fields(a):
        assert type(getattr(a, field.name)) is type(getattr(b, field.name)), field.name
    assert len(a.legs) == len(b.legs)
    for la, lb in zip(a.legs, b.legs):
        for field in dataclasses.fields(la):
            assert type(getattr(la, field.name)) is type(getattr(lb, field.name)), (
                la.name, field.name)
        assert la.expiry.unit == lb.expiry.unit, la.name


def assert_same(structure, snapshot_factory, fill, **kw):
    """Legacy and twin must agree on a cold call and on a warm repeat on the same snapshot."""
    closing = kw.get("closing", False)
    pin = kw.get("pin")
    legacy_snap = snapshot_factory()
    twin_snap = snapshot_factory()
    legacy = outcome(lambda: price_structure(structure, legacy_snap, fill, pin=pin, closing=closing))
    cold = outcome(lambda: price_structure_indexed(structure, twin_snap, fill, pin=pin, closing=closing))
    _same(legacy, cold)
    warm = outcome(lambda: price_structure_indexed(structure, twin_snap, fill, pin=pin, closing=closing))
    _same(legacy, warm)
    return legacy


def _leg(name, expiry=FIRST_POST, strike=ATM, side=BUY, right=PUT):
    return LegSpec(name, right, side, expiry, strike)


def _struct(*legs):
    return Structure(name="ERR", legs=tuple(legs), entry_offset=0, exit_offset=1)


def _fixed(day):
    return ExpirySelector(kind="fixed", expiry=day)


def _reorder(rows, order):
    if order == "reversed":
        return rows.iloc[::-1].reset_index(drop=True)
    if order == "shuffled":
        return rows.sample(frac=1.0, random_state=7).reset_index(drop=True)
    return rows


def _concat(*frames):
    return pd.concat(frames, ignore_index=True)


A_DAY = date(2026, 3, 10)  # dte 1
B_DAY = date(2026, 3, 17)  # dte 8


def _two_expiry_chain(a_strikes, b_strikes, rights="P"):
    a = make_chain(expiries=((A_DAY, 1),), strikes=a_strikes, rights=rights, delta=False)
    b = make_chain(expiries=((B_DAY, 8),), strikes=b_strikes, rights=rights, delta=False)
    return _concat(a, b)


def _selected_rows(snapshot, leg):
    """Rows matching the priced leg's expiry, right and strike."""
    rows = snapshot.rows
    mask = (
        (rows["expiry"] == leg.expiry).to_numpy()
        & (rows["right"] == leg.right).to_numpy()
        & np.isclose(rows["strike"].to_numpy(), leg.strike)
    )
    return rows[mask]


def _bump_selected_bid(snapshot, first_price):
    """Raise the bid of the first leg's selected row by half its spread; return the leg name."""
    leg = first_price.legs[0]
    sel = _selected_rows(snapshot, leg)
    assert len(sel) == 1, f"expected one row for leg {leg.name}, got {len(sel)}"
    i = sel.index[0]
    rows = snapshot.rows
    rows.loc[i, "bid"] += (rows.loc[i, "ask"] - rows.loc[i, "bid"]) / 2
    return leg.name


# --------------------------------------------------------------------------
# 1. registry structures x sessions x fills x chain shapes
# --------------------------------------------------------------------------

CHAIN_VARIANTS = {
    "s1-cp-delta": dict(spacing=1.0, rights="CP", delta=True),
    "s1-cp-nodelta": dict(spacing=1.0, rights="CP", delta=False),
    "s1-p-delta": dict(spacing=1.0, rights="P", delta=True),
    "s1-p-nodelta": dict(spacing=1.0, rights="P", delta=False),
    "s5-cp-delta": dict(spacing=5.0, rights="CP", delta=True),
    "s5-cp-nodelta": dict(spacing=5.0, rights="CP", delta=False),
    "s5-p-delta": dict(spacing=5.0, rights="P", delta=True),
    "s5-p-nodelta": dict(spacing=5.0, rights="P", delta=False),
}


@pytest.mark.parametrize("variant", list(CHAIN_VARIANTS))
@pytest.mark.parametrize("session", SESSIONS, ids=str)
@pytest.mark.parametrize("fill_name", list(FILLS))
@pytest.mark.parametrize("key", list(STRUCTURES))
def test_registry_structures_match_legacy(key, fill_name, session, variant):
    rows = make_chain(**CHAIN_VARIANTS[variant])
    assert_same(STRUCTURES[key](), lambda: snap(rows.copy(), session), FILLS[fill_name])


def test_battery_is_not_vacuous():
    rows = make_chain(rights="CP", delta=True)
    priced = snap(rows.copy(), "AMC")
    assert outcome(lambda: price_structure_indexed(put_condor(), priced, MID))[0] == "ok"
    assert outcome(lambda: price_structure_indexed(straddle_through(), snap(rows.copy(), "BMO"), MID))[0] == "ok"
    # The index actually serves this call; it is not silently falling back to legacy.
    assert ci._select_index(straddle_through(), priced) is not None


# --------------------------------------------------------------------------
# 2. width and steps sweeps over short-vol builders and spacings
# --------------------------------------------------------------------------

SWEEP_CASES = []
for _builder in (put_condor_strike, put_butterfly, put_butterfly_wide, ramp7, ctr5, twin_peak, twin_peak_5):
    _kws = [{"width_moneyness": w} for w in (0.01, 0.02, 0.05, 0.1)]
    _kws += [{"steps": s} for s in (1, 2, 4)]
    for _kw in _kws:
        _label = _builder.__name__ + "-" + "-".join(f"{k}{v}" for k, v in _kw.items())
        SWEEP_CASES.append(pytest.param(_builder, _kw, id=_label))
SWEEP_CASES += [
    pytest.param(put_condor_strike, {"inner": 2, "outer": 3}, id="put_condor_strike-inner2-outer3"),
    pytest.param(put_condor_strike, {"width_moneyness": 0.02, "inner": 2, "outer": 3},
                 id="put_condor_strike-w0.02-inner2-outer3"),
    pytest.param(put_butterfly_wide, {"inner": 2}, id="put_butterfly_wide-inner2"),
    pytest.param(twin_peak_5, {"wing_multiple": 2}, id="twin_peak_5-m2"),
    pytest.param(twin_peak_5, {"wing_multiple": 3, "width_moneyness": 0.05}, id="twin_peak_5-m3-w0.05"),
]


@pytest.mark.parametrize("spacing", [0.5, 1.0, 5.0, 10.0])
@pytest.mark.parametrize("builder,kw", SWEEP_CASES)
def test_short_vol_width_and_steps_sweeps_match_legacy(builder, kw, spacing):
    rows = make_chain(spacing=spacing, rights="CP", delta=True)
    assert_same(builder(**kw), lambda: snap(rows.copy(), "AMC"), MID)


def test_coarse_ladder_raises_ladder_too_coarse_in_both_paths():
    rows = make_chain(spacing=10.0, rights="CP", delta=True)
    legacy = assert_same(put_condor_strike(width_moneyness=0.01), lambda: snap(rows.copy(), "AMC"), MID)
    assert legacy[0] == "err" and legacy[1] is LadderTooCoarse


# --------------------------------------------------------------------------
# 3. pinned closes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("structure,session", [
    (straddle_through(), "AMC"),
    (put_condor(), "AMC"),
    (twin_peak_5(), "AMC"),
    (ctr5(), "AMC"),
    (put_butterfly(), "AMC"),
    (put_calendar(), "BMO"),
])
def test_pinned_close_on_other_spot_matches_legacy(structure, session):
    entry = price_structure(structure, snap(make_chain(spot=100.0), session), MID)
    chain_b = make_chain(spot=103.0, seed=1)
    assert_same(structure, lambda: snap(chain_b.copy(), session, spot=103.0), MID,
                pin=entry.legs, closing=True)


def test_pin_expiry_absent_on_other_chain():
    entry = price_structure(straddle_through(), snap(make_chain(), "AMC"), MID)  # expires 03-17
    chain_b = make_chain(expiries=((A_DAY, 1), (date(2026, 3, 24), 15), (date(2026, 4, 14), 36)))
    legacy = assert_same(straddle_through(), lambda: snap(chain_b.copy(), "AMC"), MID, pin=entry.legs)
    assert legacy[0] == "err" and legacy[1] is StructureError
    assert "absent from the" in legacy[2]


def test_pin_strike_unlisted_on_other_chain():
    entry = price_structure(straddle_through(), snap(make_chain(), "AMC"), MID)  # strike 100
    odd_strikes = np.arange(81.0, 121.0, 2.0)  # 100 is not listed
    chain_b = make_chain(strikes=odd_strikes)
    legacy = assert_same(straddle_through(), lambda: snap(chain_b.copy(), "AMC", spot=101.0), MID,
                         pin=entry.legs)
    assert legacy[0] == "err" and legacy[1] is StructureError
    assert "no C at strike" in legacy[2]


def test_pin_naming_no_leg_is_ignored():
    chain = make_chain()
    entry = price_structure(put_condor(), snap(chain.copy(), "AMC"), MID)
    ghost = tuple(dataclasses.replace(leg, name="nobody") for leg in entry.legs)
    assert_same(put_condor(), lambda: snap(chain.copy(), "AMC"), MID, pin=ghost)


def test_duplicate_pin_name_last_wins():
    chain = make_chain()
    structure = straddle_through()
    entry = price_structure(structure, snap(chain.copy(), "AMC"), MID)
    moved = dataclasses.replace(entry.legs[0], strike=entry.legs[0].strike + 1.0)
    pin = (*entry.legs, moved)
    legacy = assert_same(structure, lambda: snap(chain.copy(), "AMC"), MID, pin=pin)
    assert legacy[0] == "ok"
    assert legacy[1].legs[0].strike == entry.legs[0].strike + 1.0


def test_closing_without_pin_matches_legacy():
    assert_same(straddle_through(), lambda: snap(make_chain()), MID, closing=True)
    assert_same(put_condor(), lambda: snap(make_chain(rights="P", delta=False)), MID, closing=True)


# --------------------------------------------------------------------------
# 4. refusals: every one must match the legacy error type and message
# --------------------------------------------------------------------------

_PUTS = dict(rights="P", delta=False)

REFUSALS = [
    ("expiry-no-survivor", lambda: _struct(_leg("a", ExpirySelector(
        kind="nearest_dte", target_dte=10, max_dte=0))), lambda: make_chain(),
     "AMC", None, StructureError, "no expiry survives"),
    ("expiry-amc-no-post-event", straddle_through, lambda: make_chain(expiries=((A_DAY, 1),)),
     "AMC", None, StructureError, "no expiry after the AMC event date"),
    ("expiry-bmo-no-post-event", straddle_through, lambda: make_chain(expiries=((date(2026, 3, 9), 0),)),
     "BMO", None, StructureError, "no expiry on or after the event date"),
    ("expiry-no-dte-at-least", STRUCTURES["STR-RUNUP"],
     lambda: make_chain(expiries=((A_DAY, 1), (B_DAY, 8), (date(2026, 3, 24), 15))),
     "AMC", None, StructureError, "no expiry with dte >= 30"),
    ("expiry-fixed-absent", lambda: _struct(_leg("a", _fixed(date(2026, 5, 1)))),
     lambda: make_chain(), "AMC", None, StructureError, "absent from this chain"),
    ("strike-bracket-above", lambda: _struct(_leg("a", strike=StrikeSelector("bracket", side="above"))),
     lambda: make_chain(), "AMC", 1000.0, StructureError, "no listed strike above spot"),
    ("strike-bracket-below", lambda: _struct(_leg("a", strike=StrikeSelector("bracket", side="below"))),
     lambda: make_chain(), "AMC", 1.0, StructureError, "no listed strike below spot"),
    ("strike-bracket-steps", lambda: _struct(_leg("a", strike=StrikeSelector("bracket", side="below", steps=1000))),
     lambda: make_chain(), "AMC", None, StructureError, "runs off the ladder"),
    ("strike-grid-anchor-off-grid", lambda: _struct(
        _leg("a", _fixed(A_DAY), StrikeSelector("fixed", strike=99.0)),
        _leg("b", _fixed(B_DAY), StrikeSelector("grid_step", ref="a", steps=1))),
     lambda: _two_expiry_chain([98.0, 99.0, 100.0, 101.0, 102.0], [98.0, 100.0, 102.0]),
     "AMC", None, StructureError, "is not on this expiry's grid"),
    ("strike-grid-run-off", lambda: _struct(
        _leg("a", strike=ATM), _leg("b", strike=StrikeSelector("grid_step", ref="a", steps=1000))),
     lambda: make_chain(**_PUTS), "AMC", None, StructureError, "runs off the ladder"),
    ("strike-offset-none-above", lambda: _struct(
        _leg("a", strike=StrikeSelector("fixed", strike=120.0)),
        _leg("b", strike=StrikeSelector("offset_from", ref="a", moneyness=0.01))),
     lambda: make_chain(**_PUTS), "AMC", None, StructureError, "no listed strike above"),
    ("strike-offset-none-below", lambda: _struct(
        _leg("a", strike=StrikeSelector("fixed", strike=80.0)),
        _leg("b", strike=StrikeSelector("offset_from", ref="a", moneyness=-0.01))),
     lambda: make_chain(**_PUTS), "AMC", None, StructureError, "no listed strike below"),
    ("strike-mirror-unlisted", lambda: _struct(
        _leg("a", strike=ATM),
        _leg("b", strike=StrikeSelector("fixed", strike=104.0)),
        _leg("c", strike=StrikeSelector("mirror", ref="b", about="a"))),
     lambda: make_chain(strikes=[100.0, 101.0, 104.0], **_PUTS), "AMC", None, StructureError,
     "is not listed at this expiry"),
    ("strike-same-as-absent", lambda: _struct(
        _leg("a", _fixed(A_DAY), ATM),
        _leg("b", _fixed(B_DAY), StrikeSelector("same_as", ref="a"))),
     lambda: _two_expiry_chain([100.0], [101.0, 102.0, 103.0]), "AMC", None, StructureError,
     "absent at this expiry"),
    ("strike-fixed-absent", lambda: _struct(_leg("a", strike=StrikeSelector("fixed", strike=150.0))),
     lambda: make_chain(**_PUTS), "AMC", None, StructureError, "absent at this expiry"),
    ("strike-delta-no-column", lambda: _struct(_leg("a", strike=StrikeSelector("delta", target_delta=0.25))),
     lambda: make_chain(**_PUTS), "AMC", None, StructureError, "needs a delta column"),
    ("right-calls-only-put-leg", lambda: _struct(_leg("a", right=PUT)),
     lambda: make_chain(rights="C", delta=False), "AMC", None, StructureError, "no P at strike"),
    ("right-no-listed-call", lambda: _struct(_leg("a", right=CALL)),
     lambda: make_chain(**_PUTS), "AMC", None, StructureError, "no C at strike"),
    ("quote-crossed", straddle_through,
     lambda: make_chain().assign(bid=lambda d: d["ask"] + 1.0),
     "AMC", None, ValueError, "crossed quote"),
    ("quote-negative-bid", straddle_through,
     lambda: make_chain().assign(bid=-1.0), "AMC", None, ValueError, "negative bid"),
    ("quote-negative-ask", straddle_through,
     lambda: make_chain().assign(bid=-2.0, ask=-1.0), "AMC", None, ValueError, "negative"),
    ("quote-nan", straddle_through,
     lambda: make_chain().assign(bid=np.nan), "AMC", None, ValueError, "NaN in bid/ask"),
    ("quote-error-before-structure-error", lambda: _struct(
        _leg("a", strike=ATM), _leg("b", strike=StrikeSelector("fixed", strike=150.0))),
     lambda: make_chain().assign(bid=lambda d: d["ask"] + 1.0),
     "AMC", None, ValueError, "crossed quote"),
    ("quote-error-before-collision", lambda: _struct(
        _leg("a", strike=ATM), _leg("b", strike=StrikeSelector("same_as", ref="a"))),
     lambda: make_chain(**_PUTS).assign(bid=lambda d: d["ask"] + 1.0),
     "AMC", None, ValueError, "crossed quote"),
]


@pytest.mark.parametrize("label,make_structure,make_rows,session,spot,exc,text", REFUSALS,
                         ids=[case[0] for case in REFUSALS])
def test_refusals_match_legacy(label, make_structure, make_rows, session, spot, exc, text):
    rows = make_rows()
    legacy = assert_same(make_structure(), lambda: snap(rows.copy(), session, spot=spot), MID)
    assert legacy[0] == "err", label
    assert legacy[1] is exc, label
    assert text in legacy[2], label


def test_missing_spot_refused_by_both_paths():
    rows = make_chain().drop(columns="spot")
    legacy = assert_same(straddle_through(),
                         lambda: ChainSnapshot(ticker="T", obs_date=OBS, event_date=EVENT,
                                               rows=rows, spot=None, session="AMC"), MID)
    assert legacy[0] == "err" and legacy[1] is StructureError
    assert "no spot price" in legacy[2]


def test_missing_column_refused_at_snapshot_construction():
    rows = make_chain().drop(columns="bid")
    out = outcome(lambda: ChainSnapshot(ticker="T", obs_date=OBS, event_date=EVENT,
                                        rows=rows, spot=100.0, session="AMC"))
    assert out[0] == "err" and out[1] is StructureError
    assert "missing columns" in out[2]


# --------------------------------------------------------------------------
# 5. isclose edges, duplicates, ties
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spot,strikes,target,inside", [
    (100000.0, [99990.0, 100000.0, 100010.0], 100000.0 + 0.9, True),
    (100000.0, [99990.0, 100000.0, 100010.0], 100000.0 + 1.2, False),
    (100.0, [99.0, 100.0, 101.0], 100.0 + 0.0009, True),
    (100.0, [99.0, 100.0, 101.0], 100.0 + 0.0011, False),
])
def test_isclose_boundary_matches_legacy(spot, strikes, target, inside):
    rows = make_chain(spot=spot, strikes=strikes, rights="P", delta=False)
    structure = _struct(_leg("a", strike=StrikeSelector("fixed", strike=target)))
    legacy = assert_same(structure, lambda: snap(rows.copy(), "AMC", spot=spot), MID)
    assert (legacy[0] == "ok") is inside


def test_two_listed_strikes_inside_one_tolerance_first_row_wins():
    rows = make_chain(strikes=[100.0005, 100.0], rights="P", delta=False)
    structure = _struct(_leg("a", strike=StrikeSelector("fixed", strike=100.0003)))
    legacy = assert_same(structure, lambda: snap(rows.copy(), "AMC"), MID)
    assert legacy[0] == "ok"


def test_duplicate_contract_rows_first_row_wins():
    rows = make_chain(rights="P", delta=False)
    doubled = _concat(rows, rows.assign(bid=rows["bid"] * 0.5))
    legacy = assert_same(_struct(_leg("a")), lambda: snap(doubled.copy(), "AMC"), MID)
    assert legacy[0] == "ok"


@pytest.mark.parametrize("order", ["asis", "reversed", "shuffled"])
def test_atm_tie_resolves_identically_in_any_row_order(order):
    rows = _reorder(make_chain(spot=100.5, strikes=[100.0, 101.0], rights="CP", delta=False), order)
    structure = _struct(_leg("c", right=CALL))
    assert_same(structure, lambda: snap(rows.copy(), "AMC", spot=100.5), MID)


@pytest.mark.parametrize("order", ["asis", "reversed", "shuffled"])
def test_moneyness_and_delta_ties_match_legacy(order):
    rows = make_chain(spot=100.5, strikes=[100.0, 101.0], rights="P", delta=True)
    rows["delta"] = -0.5
    rows = _reorder(rows, order)
    for selector in (StrikeSelector("moneyness", moneyness=1.0),
                     StrikeSelector("delta", target_delta=0.5)):
        assert_same(_struct(_leg("p", strike=selector)), lambda: snap(rows.copy(), "AMC", spot=100.5), MID)


# --------------------------------------------------------------------------
# 6. ineligible shapes fall back to legacy; eligible ones are served
# --------------------------------------------------------------------------

SHAPES = {
    "nan-strike": lambda r: r.assign(strike=r.strike.where(r.index != 0, np.nan)),
    "float32-strike": lambda r: r.assign(strike=r.strike.astype("float32")),
    "int64-strike": lambda r: r.assign(strike=r.strike.astype("int64")),
    "tz-aware-expiry": lambda r: r.assign(expiry=r.expiry.dt.tz_localize("UTC")),
    "two-dte-one-expiry": lambda r: _concat(r, r.iloc[[0]].assign(dte=int(r.dte.iloc[0]) + 1)),
    "shared-dte": lambda r: r.assign(dte=r.dte.where(r.dte != 8, 1)),
    "empty": lambda r: r.iloc[:0],
    "duplicate-labels": lambda r: r.set_axis([0] * len(r)),
    "nan-dte": lambda r: r.assign(dte=r.dte.astype(float).where(r.index != 0, np.nan)),
    "string-right": lambda r: r.assign(right=r.right.astype("string")),
    "nat-expiry": lambda r: r.assign(expiry=r.expiry.where(r.index != 0, pd.NaT)),
    "expiry-unit-s": lambda r: r.assign(expiry=r.expiry.astype("datetime64[s]")),
    "expiry-unit-us": lambda r: r.assign(expiry=r.expiry.astype("datetime64[us]")),
}


@pytest.mark.parametrize("structure_fn", [straddle_through, put_condor, twin_peak_5])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_ineligible_and_edge_shapes_match_legacy(shape, structure_fn):
    rows = SHAPES[shape](make_chain())
    assert_same(structure_fn(), lambda: snap(rows.copy(), "AMC"), MID)


@pytest.mark.parametrize("unit", ["ns", "us", "s"])
def test_expiry_unit_of_returned_timestamp_matches_legacy(unit):
    rows = make_chain(expiry_unit=unit)
    legacy = assert_same(straddle_through(), lambda: snap(rows.copy(), "AMC"), MID)
    assert legacy[0] == "ok"  # _same already compares leg.expiry.unit against legacy's


INDEX_ELIGIBILITY = [
    ("int64-strike", True),
    ("float32-strike", True),
    ("nan-strike", True),
    ("string-right", True),
    ("expiry-unit-s", True),
    ("expiry-unit-us", True),
    ("expiry-unit-ns", True),
    ("tz-aware-expiry", False),
    ("nat-expiry", False),
    ("nan-dte", False),
    ("two-dte-one-expiry", False),
    ("shared-dte", False),
    ("empty", False),
]


def _eligibility_rows(shape):
    if shape == "expiry-unit-ns":
        return make_chain(expiry_unit="ns")
    return SHAPES[shape](make_chain())


@pytest.mark.parametrize("shape,eligible", INDEX_ELIGIBILITY, ids=[s for s, _ in INDEX_ELIGIBILITY])
def test_index_eligibility(shape, eligible):
    rows = _eligibility_rows(shape)
    assert_same(straddle_through(), lambda: snap(rows.copy(), "AMC"), MID)
    priced = snap(rows.copy(), "AMC")
    index = ci._index_for(priced)
    assert (index is not None) is eligible, f"{shape}: index built={index is not None}"


# --------------------------------------------------------------------------
# 7. nearest_dte, first_dte_at_least, first_post_event
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["asis", "reversed"])
def test_nearest_dte_tie_prefers_longer_dated_in_both_row_orders(order):
    rows = _reorder(make_chain(expiries=((B_DAY, 8), (date(2026, 3, 21), 12)), **_PUTS), order)
    structure = _struct(_leg("a", ExpirySelector(kind="nearest_dte", target_dte=10)))
    legacy = assert_same(structure, lambda: snap(rows.copy(), "AMC"), MID)
    assert legacy[0] == "ok"
    assert legacy[1].legs[0].expiry == pd.Timestamp("2026-03-21")


@pytest.mark.parametrize("target", [1, 8, 9, 15, 36, 37])
def test_first_dte_at_least_matches_legacy(target):
    rows = make_chain(**_PUTS)
    structure = _struct(_leg("a", ExpirySelector(kind="first_dte_at_least", target_dte=target)))
    assert_same(structure, lambda: snap(rows.copy(), "AMC"), MID)


@pytest.mark.parametrize("session", SESSIONS, ids=str)
def test_first_post_event_with_expiry_on_event_date(session):
    rows = make_chain(**_PUTS)
    legacy = assert_same(_struct(_leg("a")), lambda: snap(rows.copy(), session), MID)
    assert legacy[0] == "ok"
    expected = pd.Timestamp("2026-03-17") if session == "AMC" else pd.Timestamp("2026-03-10")
    assert legacy[1].legs[0].expiry == expected


def _straddle_on(selector):
    return _struct(_leg("c", selector, right=CALL), _leg("p", selector, right=PUT))


@pytest.mark.parametrize("session", SESSIONS, ids=str)
@pytest.mark.parametrize("selector", [
    pytest.param(ExpirySelector(kind="first_dte_at_least", target_dte=1, min_dte=8), id="first-dte-min8"),
    pytest.param(ExpirySelector("nearest_dte", target_dte=3, max_dte=8), id="nearest-t3-max8"),
    pytest.param(ExpirySelector(kind="first_post_event", min_dte=10), id="post-event-min10"),
    pytest.param(ExpirySelector(kind="first_post_event", min_dte=500), id="min500-empties"),
])
def test_min_max_dte_filters_match_legacy(selector, session):
    rows = make_chain(rights="CP", delta=False)  # dtes 1, 8, 15, 36
    assert_same(_straddle_on(selector), lambda: snap(rows.copy(), session), MID)


@pytest.mark.parametrize("expiries,target,expected_dte", [
    (DEFAULT_EXPIRIES, 14, 15),  # gaps 13, 6, 1, 22: an interior expiry, neither first nor last
    (DEFAULT_EXPIRIES, 11, 8),   # gaps 10, 3, 4, 25: no tie possible at an integer target
    (((B_DAY, 8), (date(2026, 3, 21), 14)), 11, 14),  # gaps 3 and 3: a real tie, goes long
])
def test_nearest_dte_is_not_first_or_last_and_ties_go_long(expiries, target, expected_dte):
    rows = make_chain(expiries=expiries, **_PUTS)
    structure = _struct(_leg("a", ExpirySelector(kind="nearest_dte", target_dte=target)))
    legacy = assert_same(structure, lambda: snap(rows.copy(), "AMC"), MID)
    assert legacy[0] == "ok"
    dte_of = {pd.Timestamp(day): dte for day, dte in expiries}
    assert dte_of[legacy[1].legs[0].expiry] == expected_dte


@pytest.mark.parametrize("selector", [
    pytest.param(ExpirySelector(kind="nearest_dte", target_dte="3"), id="target-str"),
    pytest.param(ExpirySelector(kind="first_post_event", min_dte=float("nan")), id="min-nan"),
    pytest.param(ExpirySelector(kind="first_post_event", max_dte="9"), id="max-str"),
])
def test_non_numeric_or_non_finite_bounds_fall_back_to_legacy(selector):
    rows = make_chain(rights="CP", delta=False)
    structure = _straddle_on(selector)
    assert ci._call_eligible(structure, snap(rows.copy(), "AMC")) is False
    assert_same(structure, lambda: snap(rows.copy(), "AMC"), MID)


# --------------------------------------------------------------------------
# 8. seeded property test
# --------------------------------------------------------------------------

PROPERTY_KEYS = ["CND-PS", "BFLY-P", "BFLY-P5", "RAMP7", "CTR5"]


def _random_case(rng):
    spacing = float(rng.choice([0.5, 1.0, 2.5, 5.0, 10.0]))
    n_exp = int(rng.integers(1, 7))
    dtes = sorted(int(d) for d in rng.choice(np.arange(0, 61), size=n_exp, replace=False))
    expiries = tuple(((OBS + pd.Timedelta(days=d)).date(), d) for d in dtes)
    n_strikes = int(rng.integers(9, 42))
    base = round(100.0 / spacing) * spacing
    strikes = base + (np.arange(n_strikes) - n_strikes // 2) * spacing
    strikes = strikes[(strikes > 0) & (rng.random(n_strikes) > 0.1)]  # some strikes missing
    rights = "CP" if rng.random() < 0.6 else "P"
    rows = make_chain(seed=int(rng.integers(1 << 30)), expiries=expiries, strikes=strikes,
                      rights=rights, delta=bool(rng.random() < 0.5))
    n = len(rows)
    hit = rng.random(n) < 0.01  # about 1% of rows get a bad quote
    mode = rng.integers(0, 3, n)
    bid = rows["bid"].to_numpy(copy=True)
    ask = rows["ask"].to_numpy(copy=True)
    bid[hit & (mode == 0)] = ask[hit & (mode == 0)] + 1.0  # crossed
    bid[hit & (mode == 1)] = np.nan
    bid[hit & (mode == 2)] = -1.0  # negative
    rows["bid"] = bid
    rows = rows.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    session = ("AMC", "BMO", None)[int(rng.integers(3))]
    fill = FillModel(float(rng.uniform(0.0, 1.0)))
    return rows, session, fill


@pytest.mark.parametrize("seed", range(200))
def test_random_chains_match_legacy(seed):
    rows, session, fill = _random_case(np.random.default_rng(seed))
    for key in PROPERTY_KEYS:
        assert_same(STRUCTURES[key](), lambda: snap(rows.copy(), session), fill)


# --------------------------------------------------------------------------
# 9. call-order independence
# --------------------------------------------------------------------------


def test_call_order_does_not_change_results():
    rows = make_chain(spacing=1.0, rights="CP", delta=True)
    keys = list(STRUCTURES)
    legacy = {k: outcome(lambda k=k: price_structure(STRUCTURES[k](), snap(rows.copy(), "BMO"), MID))
              for k in keys}
    shared = snap(rows.copy(), "BMO")
    orders = [keys, keys[::-1], [keys[0]] + keys[1:] + [keys[0]]]
    for order in orders:
        for k in order:
            twin = outcome(lambda k=k: price_structure_indexed(STRUCTURES[k](), shared, MID))
            _same(legacy[k], twin)


# --------------------------------------------------------------------------
# 10. verify mode
# --------------------------------------------------------------------------


def test_mutation_is_not_checked_with_verify_disabled(monkeypatch):
    monkeypatch.setenv("ENGINE_CHAIN_INDEX_VERIFY", "0")
    priced = snap(make_chain(rights="CP", delta=True), "AMC")
    first = price_structure_indexed(straddle_through(), priced, MID)
    name = _bump_selected_bid(priced, first)
    new_bid = _selected_rows(priced, first.legs[0])["bid"].iloc[0]
    second = price_structure_indexed(straddle_through(), priced, MID)  # no ChainSnapshotMutated
    assert second.leg(name).bid != first.leg(name).bid
    assert second.leg(name).bid == new_bid
    # bid/ask/dte/delta are read live and strike/right/expiry/dte are checksummed; the bid
    # change must still price exactly as legacy does.
    assert second.to_dict() == price_structure(straddle_through(), priced, MID).to_dict()


@pytest.mark.parametrize("column", ["expiry", "dte"])
def test_verify_mode_raises_on_in_place_expiry_or_dte_mutation(column):
    priced = snap(make_chain(rights="CP", delta=True), "AMC")
    assert price_structure_indexed(straddle_through(), priced, MID).legs
    first = priced.rows.index[0]
    if column == "expiry":
        priced.rows.loc[first, "expiry"] = pd.Timestamp("2026-04-15")
    else:
        priced.rows.loc[first, "dte"] += 1
    with pytest.raises(ChainSnapshotMutated) as excinfo:
        price_structure_indexed(straddle_through(), priced, MID)
    assert str(excinfo.value) == "chain snapshot rows changed after the index was built"


def test_verify_mode_does_not_checksum_bid():
    priced = snap(make_chain(rights="CP", delta=True), "AMC")
    first = price_structure_indexed(straddle_through(), priced, MID)
    assert first.legs
    name = _bump_selected_bid(priced, first)
    new_bid = _selected_rows(priced, first.legs[0])["bid"].iloc[0]
    # Documented: bid/ask/dte/delta are read live and strike/right/expiry/dte are checksummed,
    # so a bid change is not refused.
    second = price_structure_indexed(straddle_through(), priced, MID)
    assert second.legs
    assert second.leg(name).bid != first.leg(name).bid
    assert second.leg(name).bid == new_bid
    assert second.to_dict() == price_structure(straddle_through(), priced, MID).to_dict()


def test_expiry_mutation_is_checked_by_default(monkeypatch):
    monkeypatch.delenv("ENGINE_CHAIN_INDEX_VERIFY", raising=False)
    priced = snap(make_chain(rights="CP", delta=True), "AMC")
    price_structure_indexed(straddle_through(), priced, MID)
    priced.rows.loc[priced.rows.index[0], "expiry"] = pd.Timestamp("2026-04-15")
    with pytest.raises(ChainSnapshotMutated) as excinfo:
        price_structure_indexed(straddle_through(), priced, MID)
    assert str(excinfo.value) == "chain snapshot rows changed after the index was built"


# --------------------------------------------------------------------------
# 11. routing: expiry selection really goes through the index
# --------------------------------------------------------------------------


def test_price_structure_indexed_routes_expiry_through_index(monkeypatch):
    calls = {"n": 0}
    original = ci._ChainIndex._expiry_rule

    def counted(self, sel, event_date, session):
        calls["n"] += 1
        return original(self, sel, event_date, session)

    monkeypatch.setattr(ci._ChainIndex, "_expiry_rule", counted)
    priced = snap(make_chain(spacing=1.0, rights="CP", delta=True), "AMC")
    price_structure_indexed(put_condor_strike(), priced, MID)
    first = calls["n"]
    assert first > 0
    price_structure_indexed(put_condor_strike(), priced, MID)
    assert calls["n"] == first  # identical second call on the same snapshot: memo hit


# --------------------------------------------------------------------------
# 12. planted defect: the harness must be able to fail
# --------------------------------------------------------------------------


def test_planted_expiry_defect_is_detected(monkeypatch):
    original = ci._ChainIndex._expiry_rule

    def last_expiry(self, sel, event_date, session):
        result = original(self, sel, event_date, session)
        return self.ts[-1] if result is not None else result

    monkeypatch.setattr(ci._ChainIndex, "_expiry_rule", last_expiry)
    rows = make_chain()  # four distinct expiries
    with pytest.raises(AssertionError):
        # Fresh snapshot, so the index and its memo are built under the planted rule.
        assert_same(_struct(_leg("a")), lambda: snap(rows.copy(), "AMC"), MID)


# --------------------------------------------------------------------------
# 13. registry lifetime
# --------------------------------------------------------------------------


def test_registry_entry_dropped_when_snapshot_dies():
    priced = snap(make_chain(rights="P", delta=False), "AMC")
    price_structure_indexed(_struct(_leg("a")), priced, MID)
    key = id(priced)
    assert key in ci._INDEXES
    del priced
    gc.collect()
    assert key not in ci._INDEXES


# --------------------------------------------------------------------------
# 14. strike selection and contract lookup
# --------------------------------------------------------------------------


def _S(kind, **kw):
    return StrikeSelector(kind, **kw)


def _kind_cases():
    cases = []

    def add(label, build):
        cases.append((label, build))

    add("atm-put", lambda: _struct(_leg("a", strike=_S("atm"), right=PUT)))
    add("atm-call", lambda: _struct(_leg("a", strike=_S("atm"), right=CALL)))
    for m in (0.97, 1.0, 1.03, 0.0, 50.0):
        for r in (PUT, CALL):
            add(f"moneyness-{m}-{r}", lambda m=m, r=r: _struct(
                _leg("a", strike=_S("moneyness", moneyness=m), right=r)))
    for d in (0.1, 0.25, 0.5):
        for r in (PUT, CALL):
            add(f"delta-{d}-{r}", lambda d=d, r=r: _struct(
                _leg("a", strike=_S("delta", target_delta=d), right=r)))
    for k in (100.0, 100.0 + 1e-9, 100.0 + 1e-4, 123456.0, 99.5):
        for r in (PUT, CALL):
            add(f"fixed-{k!r}-{r}", lambda k=k, r=r: _struct(
                _leg("a", strike=_S("fixed", strike=k), right=r)))
    for side in ("below", "above"):
        for steps in (None, 0, 1, -1, 3, -3, 500, -500):
            add(f"bracket-{side}-{steps}", lambda side=side, steps=steps: _struct(
                _leg("a", strike=_S("bracket", side=side, steps=steps),
                     right=PUT if side == "below" else CALL)))
    for s in (1, -1, 2, -2, 500, -500):
        add(f"grid-step-{s}", lambda s=s: _struct(
            _leg("a", strike=ATM, right=PUT),
            _leg("b", strike=_S("grid_step", ref="a", steps=s), right=CALL)))
    for m in (0.03, -0.03, 0.0001, 5.0, -5.0):
        add(f"offset-from-{m}", lambda m=m: _struct(
            _leg("a", strike=ATM, right=PUT),
            _leg("b", strike=_S("offset_from", ref="a", moneyness=m), right=CALL)))
    add("mirror-bracket-below-listed", lambda: _struct(
        _leg("a", strike=ATM, right=PUT),
        _leg("b", strike=_S("bracket", side="below"), right=CALL),
        _leg("c", strike=_S("mirror", ref="a", about="b"), right=PUT)))
    add("mirror-above-listed", lambda: _struct(
        _leg("a", strike=ATM, right=PUT),
        _leg("b", strike=_S("bracket", side="above"), right=CALL),
        _leg("c", strike=_S("mirror", ref="a", about="b"), right=CALL)))
    add("mirror-unlisted", lambda: _struct(
        _leg("a", strike=_S("moneyness", moneyness=1.03), right=PUT),
        _leg("b", strike=ATM, right=CALL),
        _leg("c", strike=_S("mirror", ref="a", about="b"), right=PUT)))
    add("same-as-call-of-put", lambda: _struct(
        _leg("a", strike=ATM, right=PUT),
        _leg("b", strike=_S("same_as", ref="a"), right=CALL)))
    add("same-as-put-of-call", lambda: _struct(
        _leg("a", strike=ATM, right=CALL),
        _leg("b", strike=_S("same_as", ref="a"), right=PUT)))
    add("same-as-unlisted-anchor", lambda: _struct(
        _leg("a", strike=_S("fixed", strike=99.5), right=PUT),
        _leg("b", strike=_S("same_as", ref="a"), right=CALL)))
    add("qty0-reference-leg", lambda: _struct(
        LegSpec("a", PUT, BUY, FIRST_POST, ATM, 0.0),
        _leg("b", strike=_S("grid_step", ref="a", steps=1), right=CALL)))
    add("two-legs-same-contract", lambda: _struct(
        _leg("a", strike=ATM, right=PUT),
        _leg("b", strike=_S("same_as", ref="a"), right=PUT)))
    add("two-legs-same-as-both", lambda: _struct(
        _leg("a", strike=ATM, right=PUT),
        _leg("b", strike=_S("same_as", ref="a"), right=PUT),
        _leg("c", strike=_S("same_as", ref="a"), right=CALL)))
    return cases


KIND_CASES = _kind_cases()


@pytest.mark.parametrize("label,build", KIND_CASES, ids=[case[0] for case in KIND_CASES])
@pytest.mark.parametrize("spacing", [0.5, 1.0, 5.0])
@pytest.mark.parametrize("rights", ["CP", "P"])
@pytest.mark.parametrize("delta", [True, False])
@pytest.mark.parametrize("session", ["AMC", "BMO"])
@pytest.mark.parametrize("fill_name", ["MID", "WORST"])
def test_every_strike_kind_matches_legacy(label, build, spacing, rights, delta, session, fill_name):
    rows = make_chain(spacing=spacing, rights=rights, delta=delta, n_strikes=21)
    assert_same(build(), lambda: snap(rows.copy(), session), FILLS[fill_name])


DANGLING_CASES = [
    ("grid-step-ref-missing", lambda: _struct(_leg("a", strike=_S("grid_step", ref="zz", steps=1)))),
    ("offset-from-ref-missing", lambda: _struct(
        _leg("a", strike=_S("offset_from", ref="zz", moneyness=0.03)))),
    ("same-as-ref-missing", lambda: _struct(_leg("a", strike=_S("same_as", ref="zz")))),
    ("mirror-ref-missing", lambda: _struct(
        _leg("a", strike=ATM), _leg("b", strike=_S("mirror", ref="zz", about="a")))),
    ("mirror-about-missing", lambda: _struct(
        _leg("a", strike=ATM), _leg("b", strike=_S("mirror", ref="a", about="zz")))),
    ("forward-reference", lambda: _struct(
        _leg("a", strike=_S("same_as", ref="b")), _leg("b", strike=ATM))),
]


@pytest.mark.parametrize("label,build", DANGLING_CASES, ids=[case[0] for case in DANGLING_CASES])
def test_refs_to_missing_or_later_legs_refused_at_construction(label, build):
    # A dangling or forward reference never reaches pricing: Structure refuses it first.
    with pytest.raises(ValueError, match="is not a leg|resolves later"):
        build()


TWO_A = [98.0, 99.0, 100.0, 101.0, 102.0]
TWO_B = [98.0, 100.0, 102.0]


def _same_as_pair():
    return _struct(
        _leg("a", _fixed(A_DAY), _S("fixed", strike=100.0), right=PUT),
        _leg("b", _fixed(B_DAY), _S("same_as", ref="a"), right=PUT),
    )


def _grid_pair(anchor):
    return _struct(
        _leg("a", _fixed(A_DAY), _S("fixed", strike=anchor), right=PUT),
        _leg("b", _fixed(B_DAY), _S("grid_step", ref="a", steps=1), right=PUT),
    )


def _mirror_pair():
    return _struct(
        _leg("a", _fixed(A_DAY), _S("fixed", strike=100.0), right=PUT),
        _leg("c", _fixed(B_DAY), _S("fixed", strike=101.0), right=PUT),
        _leg("b", _fixed(B_DAY), _S("mirror", ref="a", about="c"), right=PUT),
    )


def _fixed_pair():
    return _struct(
        _leg("a", _fixed(A_DAY), _S("fixed", strike=100.0), right=PUT),
        _leg("b", _fixed(B_DAY), _S("fixed", strike=100.0), right=PUT),
    )


TWO_EXPIRY_CASES = [
    ("same-as-listed-on-both", _same_as_pair, TWO_A, TWO_B),
    ("same-as-absent-on-b", _same_as_pair, TWO_A, [98.0, 99.0, 101.0]),
    ("grid-step-anchor-off-b-grid", lambda: _grid_pair(99.0), TWO_A, TWO_B),
    ("grid-step-anchor-on-b-grid", lambda: _grid_pair(99.0), TWO_A, [98.0, 99.0, 100.0, 101.0]),
    ("mirror-listed-on-b", _mirror_pair, TWO_A, [100.0, 101.0, 102.0]),
    ("mirror-unlisted-on-b", _mirror_pair, TWO_A, [100.0, 101.0, 103.0]),
    ("fixed-on-one-expiry-only", _fixed_pair, TWO_A, [98.0, 102.0]),
    ("fixed-on-both-expiries", _fixed_pair, TWO_A, TWO_B),
]


@pytest.mark.parametrize("label,build,a_strikes,b_strikes", TWO_EXPIRY_CASES,
                         ids=[case[0] for case in TWO_EXPIRY_CASES])
def test_strike_across_two_expiries(label, build, a_strikes, b_strikes):
    rows = _two_expiry_chain(a_strikes, b_strikes)
    assert_same(build(), lambda: snap(rows.copy(), "AMC"), MID)


def test_missing_right_falls_back_to_all_rights_then_lookup_fails():
    rows = _two_expiry_chain([100.0], [100.0])  # puts only
    structure = _struct(_leg("a", _fixed(A_DAY), _S("atm"), right=CALL))
    legacy = assert_same(structure, lambda: snap(rows.copy(), "AMC"), MID)
    assert legacy[0] == "err" and legacy[1] is StructureError
    assert "no C at strike" in legacy[2]


DELTA_KINDS = ("all-nan", "some-nan", "tie", "float32", "object")


def _delta_chain(kind):
    if kind == "tie":
        rows = make_chain(strikes=[99.0, 101.0], rights="P", delta=True)
        rows["delta"] = -0.25  # every row ties on |delta|
        return rows
    rows = make_chain(rights="CP", delta=True)
    if kind == "all-nan":
        rows["delta"] = np.nan
    elif kind == "some-nan":
        rows.loc[rows.index[::3], "delta"] = np.nan
    elif kind == "float32":
        rows["delta"] = rows["delta"].astype("float32")
    elif kind == "object":
        rows["delta"] = rows["delta"].astype(object)
    return rows


@pytest.mark.parametrize("kind", DELTA_KINDS)
@pytest.mark.parametrize("target", [0.1, 0.25, 0.5])
@pytest.mark.parametrize("right", [PUT, CALL])
@pytest.mark.parametrize("order", ["asis", "reversed", "shuffled"])
def test_delta_edge_cases_match_legacy(kind, target, right, order):
    rows = _reorder(_delta_chain(kind), order)
    structure = _struct(_leg("a", strike=_S("delta", target_delta=target), right=right))
    legacy = assert_same(structure, lambda: snap(rows.copy(), "AMC"), MID)
    if kind == "tie" and right == PUT:
        assert legacy[0] == "ok"
        first_put = float(rows.loc[rows["right"] == PUT, "strike"].iloc[0])
        assert legacy[1].legs[0].strike == first_put  # first row in row order wins


def _dup_tie_chain():
    base = make_chain(spot=100.5, strikes=[100.0, 101.0], rights="CP", delta=True)
    return _concat(base, base.assign(bid=base["bid"] * 0.5)), 100.5


def _tol_pair_chain():
    base = make_chain(spot=100.0, strikes=[100.0, 100.0000005, 101.0], rights="CP", delta=True)
    return _concat(base, base.assign(bid=base["bid"] * 0.5)), 100.0


ROW_ORDER_CHAINS = {"dup-tie": _dup_tie_chain, "tol-pair": _tol_pair_chain}

ROW_ORDER_SELECTIONS = {
    "atm": lambda: _struct(_leg("a", strike=ATM, right=CALL)),
    "moneyness": lambda: _struct(_leg("a", strike=_S("moneyness", moneyness=1.0), right=PUT)),
    "delta": lambda: _struct(_leg("a", strike=_S("delta", target_delta=0.5), right=CALL)),
    "fixed": lambda: _struct(_leg("a", strike=_S("fixed", strike=100.0), right=PUT)),
    "bracket": lambda: _struct(_leg("a", strike=_S("bracket", side="below"), right=CALL)),
    "same_as": lambda: _struct(
        _leg("a", strike=ATM, right=CALL),
        _leg("b", strike=_S("same_as", ref="a"), right=PUT)),
}


@pytest.mark.parametrize("order", ["asis", "reversed", "shuffled"])
@pytest.mark.parametrize("chain_name", list(ROW_ORDER_CHAINS))
@pytest.mark.parametrize("selection", list(ROW_ORDER_SELECTIONS))
def test_first_row_in_row_order_wins(order, chain_name, selection):
    chain, spot = ROW_ORDER_CHAINS[chain_name]()
    rows = _reorder(chain, order)
    legacy = assert_same(ROW_ORDER_SELECTIONS[selection](), lambda: snap(rows.copy(), "AMC", spot=spot), MID)
    if legacy[0] == "ok":
        for leg in legacy[1].legs:
            hit = rows[(rows["expiry"] == leg.expiry) & (rows["right"] == leg.right)
                       & np.isclose(rows["strike"], leg.strike)]
            assert leg.bid == float(hit["bid"].iloc[0]), (leg.name, order)


def _obj_with(series, value):
    arr = series.to_numpy(dtype=object, copy=True)
    arr[0] = value
    return arr


UNINDEXABLE_STRIKE_SHAPES = {
    "float32-strike": lambda r: r.assign(strike=r["strike"].astype("float32")),
    "nan-strike": lambda r: r.assign(
        strike=np.where(np.arange(len(r)) == 0, np.nan, r["strike"].to_numpy(float))),
    "inf-strike": lambda r: r.assign(
        strike=np.where(np.arange(len(r)) == 0, np.inf, r["strike"].to_numpy(float))),
    "right-none": lambda r: r.assign(right=_obj_with(r["right"], None)),
    "right-nan": lambda r: r.assign(right=_obj_with(r["right"], np.nan)),
    "non-unique-index": lambda r: pd.concat([r, r]),
}

INDEXABLE_SHAPES = {
    "int64-strike": lambda r: r.assign(strike=r["strike"].astype("int64")),
    "right-lowercase": lambda r: r.assign(right=r["right"].str.lower()),
}

SHAPE_STRUCTURES = [
    ("straddle", straddle_through),
    ("condor", put_condor),
    ("one-put", lambda: _struct(_leg("a", right=PUT))),
]


@pytest.mark.parametrize("shape", list(UNINDEXABLE_STRIKE_SHAPES))
@pytest.mark.parametrize("sname,structure_fn", SHAPE_STRUCTURES, ids=[s[0] for s in SHAPE_STRUCTURES])
def test_unindexable_strike_shapes_fall_back_identically(shape, sname, structure_fn):
    rows = UNINDEXABLE_STRIKE_SHAPES[shape](make_chain(rights="CP", delta=True))
    index = ci._index_for(snap(rows.copy(), "AMC"))
    assert index is not None and index.sd is None, shape
    assert_same(structure_fn(), lambda: snap(rows.copy(), "AMC"), MID)


@pytest.mark.parametrize("shape", list(INDEXABLE_SHAPES))
@pytest.mark.parametrize("sname,structure_fn", SHAPE_STRUCTURES, ids=[s[0] for s in SHAPE_STRUCTURES])
def test_indexable_strike_shapes_are_served(shape, sname, structure_fn):
    rows = INDEXABLE_SHAPES[shape](make_chain(rights="CP", delta=True))
    index = ci._index_for(snap(rows.copy(), "AMC"))
    assert index is not None and index.sd is not None, shape
    assert_same(structure_fn(), lambda: snap(rows.copy(), "AMC"), MID)


def test_normal_chain_is_indexed_with_strike_data():
    priced = snap(make_chain(rights="CP", delta=True), "AMC")
    index = ci._index_for(priced)
    assert index is not None and index.sd is not None


def _tol_pin_case(label, pin_strike, listed):
    return pytest.param(pin_strike, listed, id=label)


PIN_STRIKE_CASES = [
    _tol_pin_case("absent", 100.0, [99.0, 101.0]),
    _tol_pin_case("nan", float("nan"), [99.0, 100.0, 101.0]),
    _tol_pin_case("inf", float("inf"), [99.0, 100.0, 101.0]),
    _tol_pin_case("rel-edge-minus", 100.0 * (1 + 1e-5) - 1e-9, [100.0]),
    _tol_pin_case("rel-edge-plus", 100.0 * (1 + 1e-5) + 1e-9, [100.0]),
    _tol_pin_case("abs-edge-minus", 100.0 + 1e-8 - 1e-9, [100.0]),
    _tol_pin_case("abs-edge-plus", 100.0 + 1e-8 + 1e-9, [100.0]),
    _tol_pin_case("rel-outside", 100.0 * (1 + 1e-5) + 1e-7, [100.0]),
]


@pytest.mark.parametrize("pin_strike,listed", PIN_STRIKE_CASES)
def test_pinned_strike_against_other_chain(pin_strike, listed):
    structure = _struct(_leg("a", right=PUT))
    entry = price_structure(structure, snap(make_chain(rights="P", delta=False), "AMC"), MID)
    pin = (dataclasses.replace(entry.legs[0], strike=pin_strike),)
    chain_b = make_chain(strikes=listed, rights="P", delta=False)
    assert_same(structure, lambda: snap(chain_b.copy(), "AMC"), MID, pin=pin, closing=True)


@pytest.mark.parametrize("bad_expiry", [pd.Timestamp("2026-05-01"), pd.Timestamp("2026-03-18")])
def test_pinned_expiry_absent_on_other_chain_edges(bad_expiry):
    structure = _struct(_leg("a", right=PUT))
    entry = price_structure(structure, snap(make_chain(rights="P", delta=False), "AMC"), MID)
    pin = (dataclasses.replace(entry.legs[0], expiry=bad_expiry),)
    legacy = assert_same(structure, lambda: snap(make_chain(rights="P", delta=False), "AMC"), MID,
                         pin=pin, closing=True)
    assert legacy[0] == "err" and legacy[1] is StructureError
    assert "absent from the" in legacy[2]


@pytest.mark.parametrize("bad_expiry", [
    pd.Timestamp("2026-03-17", tz="UTC"),
    date(2026, 3, 17),
    "2026-03-17",
])
def test_pin_with_unsupported_expiry_goes_whole_to_legacy(bad_expiry):
    structure = _struct(_leg("a", right=PUT))
    chain = make_chain(rights="P", delta=False)
    entry = price_structure(structure, snap(chain.copy(), "AMC"), MID)
    pin = (dataclasses.replace(entry.legs[0], expiry=bad_expiry),)
    assert ci._select_index(structure, snap(chain.copy(), "AMC"), pin) is None
    assert_same(structure, lambda: snap(chain.copy(), "AMC"), MID, pin=pin)


def _pin_ref_structure(kind):
    a = _leg("a", strike=ATM, right=PUT)
    if kind == "same_as":
        return _struct(a, _leg("b", strike=_S("same_as", ref="a"), right=CALL))
    if kind == "grid_step":
        return _struct(a, _leg("b", strike=_S("grid_step", ref="a", steps=1), right=CALL))
    if kind == "offset_from":
        return _struct(a, _leg("b", strike=_S("offset_from", ref="a", moneyness=0.03), right=CALL))
    return _struct(
        a,
        _leg("c", strike=_S("fixed", strike=101.0), right=CALL),
        _leg("b", strike=_S("mirror", ref="a", about="c"), right=PUT),
    )


@pytest.mark.parametrize("kind", ["same_as", "grid_step", "offset_from", "mirror"])
def test_pin_on_leg_a_while_leg_b_references_it(kind):
    structure = _pin_ref_structure(kind)
    entry = price_structure(structure, snap(make_chain(rights="CP", delta=True), "AMC"), MID)
    pin = tuple(leg for leg in entry.legs if leg.name == "a")
    chain_b = make_chain(spot=103.0, seed=1, rights="CP", delta=True)
    assert_same(structure, lambda: snap(chain_b.copy(), "AMC", spot=103.0), MID,
                pin=pin, closing=True)


WINDOW_K = (0.0, 1e-9, 1.0, 100.0, 1e5)


def _window_offsets(K):
    tol = 1e-8 + 1e-5 * abs(K)
    return {
        "zero": 0.0,
        "edge-in-plus": tol * (1 - 1e-12),
        "edge-in-minus": -tol * (1 - 1e-12),
        "edge-out-plus": tol * (1 + 1e-12),
        "edge-out-minus": -tol * (1 + 1e-12),
        "two-tol-plus": 2 * tol,
        "two-tol-minus": -2 * tol,
    }


@pytest.mark.parametrize("K", WINDOW_K, ids=repr)
@pytest.mark.parametrize("dname", list(_window_offsets(1.0)))
def test_isclose_window_boundaries(K, dname):
    d = _window_offsets(K)[dname]
    structure = _struct(_leg("a", strike=_S("fixed", strike=K), right=PUT))
    rows = make_chain(strikes=[K + d, K + 1.0, K - 1.0], rights="P", delta=False)
    legacy = assert_same(structure, lambda: snap(rows.copy(), "AMC"), MID)
    if dname == "zero":
        assert legacy[0] == "ok"
    # Pinned lookup of the same strike on the same listed chain, closing.
    entry_rows = make_chain(strikes=[K], rights="P", delta=False)
    entry = price_structure(structure, snap(entry_rows, "AMC"), MID)
    pin = (dataclasses.replace(entry.legs[0], strike=K),)
    assert_same(structure, lambda: snap(rows.copy(), "AMC"), MID, pin=pin, closing=True)


RANDOM_EXPIRY_POOL = (
    (A_DAY, 1),
    (B_DAY, 8),
    (date(2026, 3, 24), 15),
    (date(2026, 4, 14), 36),
)
REF_KINDS = ("grid_step", "offset_from", "mirror", "same_as")
RANDOM_KINDS = ("atm", "moneyness", "delta", "fixed", "bracket") + REF_KINDS


def _random_strike_case(seed):
    rng = np.random.default_rng(seed)
    spacing = float(rng.choice([0.01, 0.5, 1.0, 2.5, 5.0]))
    n_strikes = int(rng.integers(9, 31))
    base = round(100.0 / spacing) * spacing
    strikes = base + (np.arange(n_strikes) - n_strikes // 2) * spacing
    strikes = strikes[(strikes > 0) & (rng.random(n_strikes) > 0.1)]
    n_exp = int(rng.integers(1, 4))
    idx = sorted(rng.choice(len(RANDOM_EXPIRY_POOL), size=n_exp, replace=False).tolist())
    expiries = tuple(RANDOM_EXPIRY_POOL[i] for i in idx)
    rights = "CP" if rng.random() < 0.6 else "P"
    rows = make_chain(seed=int(rng.integers(1 << 30)), expiries=expiries, strikes=strikes,
                      rights=rights, delta=bool(rng.random() < 0.7))
    if rng.random() < 0.3:  # duplicate contract rows
        dup = rows.sample(n=int(rng.integers(1, 6)), random_state=int(rng.integers(1 << 30)))
        rows = _concat(rows, dup)
    if "delta" in rows.columns:  # sprinkle NaN deltas
        delta = rows["delta"].to_numpy(copy=True)
        delta[rng.random(len(delta)) < 0.05] = np.nan
        rows["delta"] = delta
    rows = rows.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    spot = float(rng.choice([99.0, 100.0, 100.5, 103.0]))
    session = SESSIONS[int(rng.integers(len(SESSIONS)))]
    fill = list(FILLS.values())[int(rng.integers(len(FILLS)))]
    closing = bool(rng.random() < 0.3)
    legs = _random_legs(rng, strikes)
    return rows, spot, session, fill, closing, legs


def _random_legs(rng, strikes):
    legs, names = [], []
    for i in range(int(rng.integers(1, 5))):
        right = CALL if rng.random() < 0.5 else PUT
        side = BUY if rng.random() < 0.5 else SELL
        qty = float(rng.choice([1.0, 1.0, 2.0, 0.0]))
        if rng.random() < 0.8:
            expiry = FIRST_POST
        else:
            expiry = ExpirySelector("nearest_dte", target_dte=int(rng.choice([3, 10, 20, 40])))
        pool = [k for k in RANDOM_KINDS if names or k not in REF_KINDS]
        kind = pool[int(rng.integers(len(pool)))]

        def ref():
            if rng.random() < 0.05:  # sometimes dangling
                return "ghost"
            return names[int(rng.integers(len(names)))]

        if kind == "atm":
            sel = _S("atm")
        elif kind == "moneyness":
            sel = _S("moneyness", moneyness=float(rng.choice([0.95, 0.98, 1.0, 1.02, 1.05])))
        elif kind == "delta":
            sel = _S("delta", target_delta=float(rng.choice([0.1, 0.25, 0.5])))
        elif kind == "fixed":
            sel = _S("fixed", strike=float(rng.choice(strikes)) + float(rng.choice([0.0, 0.0, 0.5])))
        elif kind == "bracket":
            steps = [None, 0, 1, -1][int(rng.integers(4))]
            sel = _S("bracket", side=("below", "above")[int(rng.integers(2))], steps=steps)
        elif kind == "grid_step":
            sel = _S("grid_step", ref=ref(), steps=int(rng.integers(-2, 3)))
        elif kind == "offset_from":
            sel = _S("offset_from", ref=ref(), moneyness=float(rng.choice([0.01, -0.01, 0.03])))
        elif kind == "mirror":
            sel = _S("mirror", ref=ref(), about=ref())
        else:
            sel = _S("same_as", ref=ref())
        legs.append(LegSpec(f"l{i}", right, side, expiry, sel, qty))
        names.append(f"l{i}")
    return legs


def _has_dangling(legs):
    names = {leg.name for leg in legs}
    return any(ref not in names for leg in legs for ref in leg.strike.refs)


@pytest.mark.parametrize("seed", range(300))
def test_random_strike_cases_match_legacy(seed):
    rows, spot, session, fill, closing, legs = _random_strike_case(seed)
    if _has_dangling(legs):
        with pytest.raises(ValueError, match="is not a leg"):
            _struct(*legs)
        return
    assert_same(_struct(*legs), lambda: snap(rows.copy(), session, spot=spot), fill, closing=closing)


def test_random_strike_cases_are_not_vacuous():
    counts = {"ok": 0, "err": 0, "dangling": 0}
    for seed in range(300):
        rows, spot, session, fill, closing, legs = _random_strike_case(seed)
        if _has_dangling(legs):
            counts["dangling"] += 1
            continue
        structure = _struct(*legs)
        got = outcome(lambda: price_structure(structure, snap(rows.copy(), session, spot=spot),
                                              fill, closing=closing))
        counts[got[0]] += 1
    assert counts["ok"] > 20 and counts["err"] > 20, counts


def _every_kind_structure():
    return _struct(
        _leg("a", strike=_S("atm"), right=PUT),
        _leg("b", strike=_S("moneyness", moneyness=1.03), right=CALL),
        _leg("c", strike=_S("delta", target_delta=0.25), right=CALL),
        _leg("d", strike=_S("fixed", strike=95.0), right=PUT),
        _leg("e", strike=_S("bracket", side="below"), right=CALL),
        _leg("f", strike=_S("grid_step", ref="a", steps=2), right=PUT),
        _leg("g", strike=_S("offset_from", ref="d", moneyness=0.02), right=PUT),
        _leg("h", strike=_S("mirror", ref="d", about="a"), right=PUT),
        _leg("i", strike=_S("same_as", ref="c"), right=PUT),
    )


def test_strike_selection_and_lookup_do_not_use_dataframe_select(monkeypatch):
    rows = make_chain(spacing=1.0, rights="CP", delta=True)
    structure = _every_kind_structure()
    legacy = outcome(lambda: price_structure(structure, snap(rows.copy(), "AMC"), MID))
    assert legacy[0] == "ok"

    def forbidden(self, *args, **kwargs):
        raise AssertionError("StrikeSelector.select ran on the index path")

    monkeypatch.setattr(StrikeSelector, "select", forbidden)
    twin = outcome(lambda: price_structure_indexed(structure, snap(rows.copy(), "AMC"), MID))
    _same(legacy, twin)


def test_fallback_path_uses_legacy_select(monkeypatch):
    rows = make_chain(rights="P", delta=False)
    rows["strike"] = rows["strike"].astype("float32")
    priced = snap(rows, "AMC")
    index = ci._index_for(priced)
    assert index is not None and index.sd is None

    def forbidden(self, *args, **kwargs):
        raise AssertionError("legacy select reached from the fallback")

    monkeypatch.setattr(StrikeSelector, "select", forbidden)
    with pytest.raises(AssertionError, match="legacy select reached"):
        price_structure_indexed(_struct(_leg("a", strike=ATM, right=PUT)), priced, MID)


@pytest.mark.parametrize("mutation", ["strike", "right", "index"])
def test_verify_mode_raises_on_strike_or_right_mutation(mutation):
    priced = snap(make_chain(rights="CP", delta=True), "AMC")
    assert price_structure_indexed(straddle_through(), priced, MID).legs
    rows = priced.rows
    first = rows.index[0]
    if mutation == "strike":
        rows.loc[first, "strike"] += 1.0
    elif mutation == "right":
        rows.loc[first, "right"] = "P" if rows.loc[first, "right"] == "C" else "C"
    else:
        rows.index = pd.Index([0] * len(rows))  # reorder by making the index non-unique
    with pytest.raises(ChainSnapshotMutated) as excinfo:
        price_structure_indexed(straddle_through(), priced, MID)
    assert str(excinfo.value) == "chain snapshot rows changed after the index was built"


def test_strike_or_right_mutation_not_checked_with_verify_disabled(monkeypatch):
    """With verify off, a strike edit made after the index is built is not checksummed.

    Scope, stated honestly: this test checks only that the call does not raise
    ChainSnapshotMutated and does not surface an AssertionError. It does not assert
    that the priced result equals legacy. The edited row (row 0) is not one the
    straddle selects, so the outcome is not expected to depend on the edit.
    """
    monkeypatch.setenv("ENGINE_CHAIN_INDEX_VERIFY", "0")
    priced = snap(make_chain(rights="CP", delta=True), "AMC")
    assert price_structure_indexed(straddle_through(), priced, MID).legs
    priced.rows.loc[priced.rows.index[0], "strike"] += 1.0
    out = outcome(lambda: price_structure_indexed(straddle_through(), priced, MID))
    assert out[0] == "ok" or out[1] is not ChainSnapshotMutated
    assert out[1] is not AssertionError


def test_ask_edit_is_read_live_and_priced_as_legacy():
    priced = snap(make_chain(rights="CP", delta=True), "AMC")
    first = price_structure_indexed(straddle_through(), priced, MID)
    name = first.legs[0].name
    sel = _selected_rows(priced, first.legs[0])
    assert len(sel) == 1
    priced.rows.loc[sel.index[0], "ask"] += 0.5
    second = price_structure_indexed(straddle_through(), priced, MID)  # no ChainSnapshotMutated
    assert second.leg(name).ask != first.leg(name).ask
    assert second.to_dict() == price_structure(straddle_through(), priced, MID).to_dict()


def test_delta_edit_on_delta_leg_is_read_live():
    structure = _struct(_leg("a", strike=_S("delta", target_delta=0.25), right=CALL))
    priced = snap(make_chain(rights="CP", delta=True), "AMC")
    first = price_structure_indexed(structure, priced, MID)
    leg = first.legs[0]
    rows = priced.rows
    mask = (rows["right"] == CALL) & (rows["expiry"] == leg.expiry) & (rows["strike"] == 110.0)
    rows.loc[mask, "delta"] = 0.25  # the 110 call now carries exactly the target delta
    second = price_structure_indexed(structure, priced, MID)
    assert second.legs[0].strike == 110.0
    assert second.to_dict() == price_structure(structure, priced, MID).to_dict()


@pytest.mark.parametrize("key", list(STRUCTURES))
def test_registry_structures_use_strike_index(key):
    rows = make_chain(spacing=1.0, rights="CP", delta=True)
    index = ci._index_for(snap(rows.copy(), "AMC"))
    assert index is not None and index.sd is not None
    opening = outcome(lambda: price_structure(STRUCTURES[key](), snap(rows.copy(), "AMC"), MID))
    pin = opening[1].legs if opening[0] == "ok" else None
    assert_same(STRUCTURES[key](), lambda: snap(rows.copy(), "AMC"), MID, closing=True)
    assert_same(STRUCTURES[key](), lambda: snap(rows.copy(), "AMC"), MID, pin=pin, closing=True)


def test_planted_select_strike_defect_is_detected(monkeypatch):
    original = ci._StrikeData.select_strike

    def neighbour(self, sel, e, right, spot, resolved):
        chosen = original(self, sel, e, right, spot, resolved)
        grid = self.group(e, None).grid
        i = int(np.searchsorted(grid, chosen))
        j = i + 1 if i + 1 < grid.size else i - 1
        return float(grid[j])

    monkeypatch.setattr(ci._StrikeData, "select_strike", neighbour)
    with pytest.raises(AssertionError):
        assert_same(straddle_through(), lambda: snap(make_chain(rights="CP", delta=True), "AMC"), MID)


def test_planted_find_defect_detected_on_duplicate_rows(monkeypatch):
    def last_match(grp, strike):
        hit = np.flatnonzero(np.isclose(grp.st, strike))
        return int(hit[-1]) if hit.size else None

    monkeypatch.setattr(ci._StrikeData, "find", staticmethod(last_match))
    base = make_chain(rights="P", delta=False)
    doubled = _concat(base, base.assign(bid=base["bid"] * 0.5))
    with pytest.raises(AssertionError):
        assert_same(_struct(_leg("a")), lambda: snap(doubled.copy(), "AMC"), MID)


def test_planted_lookup_none_detected(monkeypatch):
    monkeypatch.setattr(ci._StrikeData, "lookup", lambda self, e, right, strike: None)
    with pytest.raises(AssertionError):
        assert_same(straddle_through(), lambda: snap(make_chain(rights="CP", delta=True), "AMC"), MID)


# --------------------------------------------------------------------------
# 15. non-finite spot and selector fields; live dte reads; call/put strike sets
# --------------------------------------------------------------------------

_NAN = float("nan")
_INF = float("inf")


def _cp_snapshot():
    return snap(make_chain(rights="CP", delta=True), "AMC")


def _pair_with_strike(strike):
    return _struct(_leg("c", strike=strike, right=CALL), _leg("p", strike=strike, right=PUT))


def _whole_to_legacy(make_structure, make_snapshot, fill=MID):
    """Legacy and twin each build their own structure and snapshot inside ``outcome``.

    A refusal at any construction or pricing step is compared as a whole: status,
    exception type and message. Successful outcomes are compared with ``_same``,
    cold and warm.
    """
    legacy = outcome(lambda: price_structure(make_structure(), make_snapshot(), fill))
    built = outcome(lambda: (make_structure(), make_snapshot()))
    if built[0] == "err":
        _same(legacy, built)
        return legacy
    structure, snapshot = built[1]
    cold = outcome(lambda: price_structure_indexed(structure, snapshot, fill))
    _same(legacy, cold)
    warm = outcome(lambda: price_structure_indexed(structure, snapshot, fill))
    _same(legacy, warm)
    return legacy


NONFINITE_CASES = [
    pytest.param(straddle_through,
                 lambda: snap(make_chain(rights="CP", delta=True), "AMC", spot=_NAN),
                 id="spot-nan"),
    pytest.param(straddle_through,
                 lambda: snap(make_chain(rights="CP", delta=True), "AMC", spot=_INF),
                 id="spot-inf"),
    pytest.param(straddle_through,
                 lambda: ChainSnapshot(ticker="T", obs_date=OBS, event_date=EVENT,
                                       rows=make_chain(rights="CP", delta=True).drop(columns="spot"),
                                       spot=None, session="AMC"),
                 id="spot-missing"),
    pytest.param(lambda: _pair_with_strike(_S("moneyness", moneyness=_NAN)), _cp_snapshot,
                 id="moneyness-nan"),
    pytest.param(lambda: _pair_with_strike(_S("delta", target_delta=_INF)), _cp_snapshot,
                 id="target-delta-inf"),
    pytest.param(lambda: _struct(
                     _leg("a", strike=ATM, right=PUT),
                     _leg("b", strike=_S("grid_step", ref="a", steps=2.5), right=CALL)),
                 _cp_snapshot, id="grid-step-steps-2.5"),
    pytest.param(lambda: _pair_with_strike(_S("fixed", strike=_NAN)), _cp_snapshot,
                 id="fixed-strike-nan"),
]


@pytest.mark.parametrize("make_structure,make_snapshot", NONFINITE_CASES)
def test_nonfinite_spot_and_selector_fields_go_whole_to_legacy(make_structure, make_snapshot):
    """Non-finite or missing spot, and non-finite or non-integral selector fields.

    Each case must end in the same outcome on both paths: the same values, or the
    same exception type and message.
    """
    _whole_to_legacy(make_structure, make_snapshot)


def test_dte_edit_is_read_live_with_verify_off(monkeypatch):
    monkeypatch.setenv("ENGINE_CHAIN_INDEX_VERIFY", "0")
    # Fixed selection on the last expiry (dte 36), so the dte edit cannot move the expiry choice.
    structure = _straddle_on(_fixed(date(2026, 4, 14)))
    priced = snap(make_chain(rights="CP", delta=True), "AMC")
    first = price_structure_indexed(structure, priced, MID)
    leg = first.legs[0]
    sel = _selected_rows(priced, leg)
    assert len(sel) == 1, f"expected one row for leg {leg.name}, got {len(sel)}"
    i = sel.index[0]
    before = int(priced.rows.loc[i, "dte"])
    priced.rows.loc[i, "dte"] += 100  # one row of the selected contract only
    second = price_structure_indexed(structure, priced, MID)  # no ChainSnapshotMutated
    assert second.leg(leg.name).dte == before + 100
    assert second.to_dict() == price_structure(structure, priced, MID).to_dict()


def _call_put_chain():
    """Calls list 90..110 step 5; puts list 92..112 step 4. One expiry; 11 rows."""
    calls = make_chain(expiries=((B_DAY, 8),), strikes=np.arange(90.0, 111.0, 5.0),
                       rights="C", delta=True)
    puts = make_chain(expiries=((B_DAY, 8),), strikes=np.arange(92.0, 113.0, 4.0),
                      rights="P", delta=True)
    return _concat(calls, puts)


CALL_PUT_CASES = [
    ("atm-call", lambda: _struct(_leg("a", strike=ATM, right=CALL))),
    ("atm-put", lambda: _struct(_leg("a", strike=ATM, right=PUT))),
    ("bracket-below-put", lambda: _struct(_leg("a", strike=_S("bracket", side="below"), right=PUT))),
    ("bracket-above-call", lambda: _struct(_leg("a", strike=_S("bracket", side="above"), right=CALL))),
    ("grid-step-call-from-put", lambda: _struct(
        _leg("a", strike=ATM, right=PUT),
        _leg("b", strike=_S("grid_step", ref="a", steps=1), right=CALL))),
    ("grid-step-put-from-call", lambda: _struct(
        _leg("a", strike=ATM, right=CALL),
        _leg("b", strike=_S("grid_step", ref="a", steps=1), right=PUT))),
    ("offset-call-from-put", lambda: _struct(
        _leg("a", strike=ATM, right=PUT),
        _leg("b", strike=_S("offset_from", ref="a", moneyness=0.03), right=CALL))),
    ("offset-put-from-call", lambda: _struct(
        _leg("a", strike=ATM, right=CALL),
        _leg("b", strike=_S("offset_from", ref="a", moneyness=-0.03), right=PUT))),
    ("mirror-put", lambda: _struct(
        _leg("a", strike=_S("fixed", strike=96.0), right=PUT),
        _leg("b", strike=_S("fixed", strike=104.0), right=PUT),
        _leg("c", strike=_S("mirror", ref="a", about="b"), right=PUT))),
    ("mirror-call", lambda: _struct(
        _leg("a", strike=_S("fixed", strike=90.0), right=CALL),
        _leg("b", strike=_S("fixed", strike=100.0), right=CALL),
        _leg("c", strike=_S("mirror", ref="a", about="b"), right=CALL))),
    ("same-as-call-of-put", lambda: _struct(
        _leg("a", strike=ATM, right=PUT),
        _leg("b", strike=_S("same_as", ref="a"), right=CALL))),
    ("same-as-put-of-call", lambda: _struct(
        _leg("a", strike=ATM, right=CALL),
        _leg("b", strike=_S("same_as", ref="a"), right=PUT))),
    ("same-as-put-absent-on-calls", lambda: _struct(
        _leg("a", strike=_S("fixed", strike=92.0), right=PUT),
        _leg("b", strike=_S("same_as", ref="a"), right=CALL))),
    ("fixed-call-only", lambda: _struct(_leg("a", strike=_S("fixed", strike=90.0), right=CALL))),
    ("fixed-put-only", lambda: _struct(_leg("a", strike=_S("fixed", strike=92.0), right=PUT))),
    ("delta-call", lambda: _struct(_leg("a", strike=_S("delta", target_delta=0.25), right=CALL))),
    ("delta-put", lambda: _struct(_leg("a", strike=_S("delta", target_delta=0.25), right=PUT))),
    ("all-kinds-valid-on-both-rights", lambda: _struct(
        _leg("a", strike=ATM, right=CALL),
        _leg("b", strike=ATM, right=PUT),
        _leg("c", strike=_S("grid_step", ref="a", steps=1), right=CALL),
        _leg("d", strike=_S("grid_step", ref="b", steps=1), right=PUT),
        _leg("e", strike=_S("offset_from", ref="a", moneyness=0.03), right=CALL),
        _leg("f", strike=_S("offset_from", ref="b", moneyness=-0.03), right=PUT),
        _leg("g", strike=_S("same_as", ref="b"), right=CALL),
        _leg("h", strike=_S("same_as", ref="a"), right=PUT),
        _leg("i", strike=_S("delta", target_delta=0.25), right=CALL),
        _leg("j", strike=_S("delta", target_delta=0.25), right=PUT))),
]


@pytest.mark.parametrize("fill_name", ["MID", "WORST"])
@pytest.mark.parametrize("label,build", CALL_PUT_CASES, ids=[case[0] for case in CALL_PUT_CASES])
def test_call_and_put_strike_sets_differ(label, build, fill_name):
    rows = _call_put_chain()
    assert_same(build(), lambda: snap(rows.copy(), "AMC"), FILLS[fill_name])
