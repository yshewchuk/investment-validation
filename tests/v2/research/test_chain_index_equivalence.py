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
    # Only expiry/dte are checksummed; the bid change must still price exactly as legacy does.
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
    # Documented: only expiry and dte are checksummed, so a bid change is not refused.
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
