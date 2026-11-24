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
from engine.chain_index import (
    ChainSnapshotMutated,
    price_resolution,
    price_structure_indexed,
    resolve_structure,
)
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
    """Legacy and twin must agree cold, warm, and through resolve/price_resolution."""
    closing = kw.get("closing", False)
    pin = kw.get("pin")
    legacy_snap = snapshot_factory()
    twin_snap = snapshot_factory()
    legacy = outcome(lambda: price_structure(structure, legacy_snap, fill, pin=pin, closing=closing))
    cold = outcome(lambda: price_structure_indexed(structure, twin_snap, fill, pin=pin, closing=closing))
    _same(legacy, cold)
    warm = outcome(lambda: price_structure_indexed(structure, twin_snap, fill, pin=pin, closing=closing))
    _same(legacy, warm)
    if legacy[0] == "ok":
        resolution = resolve_structure(structure, twin_snap, pin=pin)
        _same(legacy, ("ok", price_resolution(resolution, twin_snap, fill, closing=closing)))
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
    assert ci._select_index(straddle_through(), priced, None, 100.0) is not None


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
# 3. pinned closes and resolution pins
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


@pytest.mark.parametrize("structure", [straddle_through(), put_condor(), ctr5()])
def test_resolution_pin_equals_legacy_legs_pin(structure):
    chain = make_chain()
    entry = price_structure(structure, snap(chain.copy(), "AMC"), MID)
    legacy = outcome(lambda: price_structure(structure, snap(chain.copy(), "AMC"), MID, pin=entry.legs))
    twin_snap = snap(chain.copy(), "AMC")
    resolution = resolve_structure(structure, twin_snap)
    twin = outcome(lambda: price_structure_indexed(structure, twin_snap, MID, pin=resolution))
    _same(legacy, twin)


def test_resolution_pin_for_other_ticker_or_structure_raises_value_error_first():
    chain = make_chain()
    resolution = resolve_structure(straddle_through(), snap(chain.copy(), "AMC"))
    # No spot anywhere: a StructureError would appear if the identity check ran later.
    other_ticker = ChainSnapshot(ticker="U", obs_date=OBS, event_date=EVENT,
                                 rows=chain.drop(columns="spot"), spot=None, session="AMC")
    with pytest.raises(ValueError):
        price_structure_indexed(straddle_through(), other_ticker, MID, pin=resolution)
    with pytest.raises(ValueError):
        resolve_structure(straddle_through(), other_ticker, pin=resolution)
    with pytest.raises(ValueError):
        price_structure_indexed(put_condor(), snap(chain.copy(), "AMC"), MID, pin=resolution)


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


def test_index_serves_eligible_shapes_and_refuses_ineligible_ones():
    base = make_chain()
    int_strike = snap(SHAPES["int64-strike"](base))
    unit_s = snap(make_chain(expiry_unit="s"))
    nan_strike = snap(SHAPES["nan-strike"](base))
    tz_aware = snap(SHAPES["tz-aware-expiry"](base))
    dup_dte = snap(SHAPES["two-dte-one-expiry"](base))
    assert ci._index_for(int_strike) is not None
    assert ci._index_for(unit_s) is not None
    assert ci._index_for(nan_strike) is None
    assert ci._index_for(tz_aware) is None
    assert ci._index_for(dup_dte) is None


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


def test_verify_mode_raises_on_in_place_mutation():
    priced = snap(make_chain(rights="CP", delta=True), "AMC")
    assert price_structure_indexed(straddle_through(), priced, MID).legs
    priced.rows.loc[priced.rows.index[0], "bid"] += 1.0
    with pytest.raises(ChainSnapshotMutated):
        price_structure_indexed(straddle_through(), priced, MID)


def test_mutation_is_not_checked_without_verify_mode(monkeypatch):
    monkeypatch.delenv("ENGINE_CHAIN_INDEX_VERIFY", raising=False)
    priced = snap(make_chain(rights="CP", delta=True), "AMC")
    price_structure_indexed(straddle_through(), priced, MID)
    priced.rows.loc[priced.rows.index[0], "bid"] += 1.0
    price_structure_indexed(straddle_through(), priced, MID)  # no ChainSnapshotMutated


# --------------------------------------------------------------------------
# 11. resolutions
# --------------------------------------------------------------------------


def test_price_resolution_requires_the_snapshot_it_was_resolved_on():
    chain = make_chain(rights="CP", delta=True)
    first = snap(chain.copy(), "AMC")
    second = snap(chain.copy(), "AMC")
    resolution = resolve_structure(straddle_through(), first)
    with pytest.raises(ValueError):
        price_resolution(resolution, second, MID)


def test_resolution_legs_carry_opening_side_and_closing_flips_them():
    structure = put_condor()
    chain = make_chain(rights="P", delta=False)
    priced = snap(chain.copy(), "AMC")
    resolution = resolve_structure(structure, priced)
    assert [leg.side for leg in resolution.legs] == [spec.side for spec in structure.legs]
    closing = price_resolution(resolution, priced, MID, closing=True)
    assert closing.legs[0].side != resolution.legs[0].side
    entry = price_structure(structure, snap(chain.copy(), "AMC"), MID)
    legacy_close = outcome(lambda: price_structure(structure, snap(chain.copy(), "AMC"), MID,
                                                   pin=entry.legs, closing=True))
    _same(legacy_close, ("ok", closing))


# --------------------------------------------------------------------------
# 12. planted defect: the harness must be able to fail
# --------------------------------------------------------------------------


def test_planted_atm_defect_is_detected(monkeypatch):
    original = ci._ChainIndex.select_strike

    def shifted(self, sel, *args, **kwargs):
        result = original(self, sel, *args, **kwargs)
        return result + 0.01 if sel.kind == "atm" else result

    monkeypatch.setattr(ci._ChainIndex, "select_strike", shifted)
    rows = make_chain()
    with pytest.raises(AssertionError):
        assert_same(straddle_through(), lambda: snap(rows.copy(), "AMC"), MID)


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
