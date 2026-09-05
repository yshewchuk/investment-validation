#!/usr/bin/env python3
"""The complete family of symmetric all-put structures the rules admit.

EXP-126 compared three hand-written shapes. This module does not write shapes
down at all — it **enumerates** them from the four properties the program
already requires of a twin peak, and then lets the arithmetic say which ones
exist.

A candidate is a set of listed strikes, mirror-symmetric in DOLLARS about an
anchor strike ``A``, carrying signed contract counts ``q``:

    strikes   A,  A +/- d1,  A +/- d2,  ...   (d1 < d2 < ... , in dollars)
    contracts q0, q1 (both),  q2 (both),  ...

and it is admissible when all four of these hold:

1. **the tails cancel** — ``q0 + 2*sum(q_t) = 0``. Below the lowest strike the
   position settles to ``sum(q_i) * (K_i - S)``, whose ``S`` term vanishes on
   the sum and whose constant vanishes on the symmetry. This is what makes the
   deep-ITM tail exactly flat rather than approximately flat.
2. **the floor is zero** — payoff >= 0 at every strike. The payoff is piecewise
   linear with breakpoints only at the strikes and is zero outside them, so
   nonnegativity at the strikes is nonnegativity everywhere. With (1) it is
   what makes max loss equal to the debit, which is the entire defined-risk
   claim; without it "defined risk" is a sentence rather than a property.
3. **it is a debit** — implied by (2): a payoff that is nonnegative everywhere
   and positive somewhere cannot be bought for a credit. Stated separately
   because every metric downstream (``ret = pnl/cost``, the P&L gate, sizing)
   is quoted on a positive debit.
4. **it fits in eight contracts** — ``|q0| + 2*sum|q_t| <= 8``, TWIN-P's own
   size. Sixteen spread crossings round trip is already the structure's
   dominant execution risk and this experiment is not the place to raise it.

Scaled duplicates are removed (``gcd(|q|) = 1``): doubling every leg doubles
the debit and the P&L and leaves ``ret`` unchanged, so it is the same trade.

**What the enumeration finds, and it is the reason to read this file before
the report.** On 3, 4, 5 and 7 strikes there are exactly eight families, and
only two of them are twin-peaked:

    N=3  q=(-2, 1)          centre-peaked   the long put butterfly
    N=4  q=(0, -1, 1)       centre-peaked   the long put condor (CND-P's shape)
    N=5  q=(-4, 1, 1)       centre-peaked
    N=5  q=(-2, -1, 2)      centre-peaked
    N=5  q=(2, -2, 1)       TWIN-PEAKED     TWIN-P5, for every wing >= 2
    N=7  q=(-2, -1, 1, 1)   centre-peaked
    N=7  q=(-2, 1, -1, 1)   centre-peaked
    N=7  q=(2, -1, -1, 1)   TWIN-PEAKED     TWIN-P at d = (1, 2, 4)

Both incumbents fall out of the enumeration rather than being inserted into
it, which is the check that the four properties are the right four.

The consequence is registered in spec.yaml and it cuts against the motivating
intuition: **four strikes and three strikes cannot express a twin peak at
all.** Dropping from five strikes to four does not buy a cheaper version of
the same thesis; it buys a different thesis — a tent that wants a quiet print.
That is a fact about the payoff algebra, not a finding, and it is a unit test
(``test_family.py``) rather than a paragraph.

Six-strike families exist (``q0 = 0``, three pairs) and are excluded by
registration: the user's set is 7/5/4/3, and every extra family widens the
max-statistic this experiment is already worried about.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "MAX_CONTRACTS", "MAX_ABS_QTY", "STRIKE_COUNTS", "MAX_LADDER_POS",
    "ANCHOR_OFFSETS", "Family", "FAMILIES", "FAMILY_BY_KEY",
    "payoff_profile", "offsets_admissible", "patterns", "Pattern",
    "to_structure", "reduce_offsets",
]

#: Registered in spec.yaml before this ran. TWIN-P's own contract count.
MAX_CONTRACTS = 8

#: No single leg may carry more than this. It only ever binds together with
#: MAX_CONTRACTS, and exists so the enumeration terminates on a stated bound
#: rather than on the contract cap's interaction with the tail-cancel equation.
MAX_ABS_QTY = 4

#: The strike counts this experiment registers. Six is enumerable and excluded.
STRIKE_COUNTS = (3, 4, 5, 7)

#: How far along the ticker's own listed ladder a wing may sit. The width rules
#: bind long before this on any normal ladder; it is a termination bound, and
#: the run reports how often it is what stopped the enumeration.
MAX_LADDER_POS = 20

#: Where the axis of symmetry may sit, in ladder positions above the listed
#: strike at or below spot. The user's rule: ATM, ATM+1, ATM+2. Downward
#: shifts are excluded by registration — EXP-123 measured the put structure
#: decaying fastest on UP moves (exit/debit 0.14 deep-up against 0.36
#: deep-down), so room is bought above, and a downward shift is the opposite
#: of the one lever EXP-124 found.
ANCHOR_OFFSETS = (0, 1, 2)


@dataclass(frozen=True)
class Family:
    """One quantity vector: the shape, before any spacing is chosen.

    ``q0`` is the anchor's signed contract count (positive = long puts) and
    ``tail`` is the count carried at BOTH ``A + d_t`` and ``A - d_t``.
    """

    key: str
    q0: int
    tail: tuple[int, ...]
    n_strikes: int
    #: True when some admissible spacing puts the payoff maximum away from the
    #: anchor. Computed, never asserted.
    twin_peaked: bool
    label: str

    @property
    def tiers(self) -> int:
        return len(self.tail)

    @property
    def contracts(self) -> int:
        return abs(self.q0) + 2 * sum(abs(q) for q in self.tail)

    def quantities(self, offsets: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        """``(signed offsets in ladder steps, signed contracts)``, anchor first.

        The offsets are returned in LADDER POSITIONS relative to the anchor —
        positive up, negative down. Turning them into dollars is the caller's
        job because only the caller knows the ticker's ladder.
        """
        pos = [0] if self.q0 else []
        qty = [self.q0] if self.q0 else []
        for t, d in enumerate(offsets):
            pos += [d, -d]
            qty += [self.tail[t], self.tail[t]]
        return np.array(pos, dtype=int), np.array(qty, dtype=float)


def payoff_profile(q0: int, tail: tuple[int, ...], offsets) -> np.ndarray:
    """Terminal payoff at every strike, in the same units as ``offsets``.

    Puts only: ``V(S) = sum_i q_i * max(K_i - S, 0)``, evaluated at each strike
    with the anchor at zero. Returns values ordered by strike, ascending.
    """
    offsets = tuple(float(d) for d in offsets)
    strikes = np.array(sorted([0.0] + list(offsets) + [-d for d in offsets]))
    qty = np.zeros(strikes.size)
    qty[np.isclose(strikes, 0.0)] = q0
    for t, d in enumerate(offsets):
        qty[np.isclose(strikes, d)] = tail[t]
        qty[np.isclose(strikes, -d)] = tail[t]
    return (qty[None, :] * np.maximum(strikes[None, :] - strikes[:, None], 0.0)).sum(axis=1)


def offsets_admissible(q0: int, tail: tuple[int, ...], offsets) -> bool:
    """Property 2: the floor is zero, and the structure is not identically zero."""
    values = payoff_profile(q0, tail, offsets)
    return bool((values >= -1e-9).all() and (values > 1e-9).any())


def _enumerate_families() -> list[Family]:
    """Every quantity vector satisfying properties 1, 2 and 4, for N in 3/4/5/7."""
    found: list[Family] = []
    span = range(-MAX_ABS_QTY, MAX_ABS_QTY + 1)
    for tiers in (1, 2, 3):
        for q0 in span:
            for tail in itertools.product(span, repeat=tiers):
                # A zero in the tail is a structure with fewer tiers, and it
                # would be enumerated twice — once here and once at its own
                # tier count — so it is skipped rather than deduplicated later.
                if any(q == 0 for q in tail):
                    continue
                if q0 + 2 * sum(tail) != 0:               # (1) tails cancel
                    continue
                contracts = abs(q0) + 2 * sum(abs(q) for q in tail)
                if not 0 < contracts <= MAX_CONTRACTS:    # (4) eight contracts
                    continue
                if math.gcd(abs(q0), *[abs(q) for q in tail]) != 1:
                    continue
                n_strikes = (1 if q0 else 0) + 2 * tiers
                if n_strikes not in STRIKE_COUNTS:
                    continue
                # (2) is a property of the shape AND the spacing, so a family
                # survives if ANY integer spacing gives it a zero floor. The
                # per-candidate check happens again at build time.
                shapes = [
                    ds for ds in itertools.combinations(range(1, MAX_LADDER_POS + 1), tiers)
                    if offsets_admissible(q0, tail, ds)
                ]
                if not shapes:
                    continue
                twin = any(
                    payoff_profile(q0, tail, ds).max()
                    > payoff_profile(q0, tail, ds)[len(ds)] + 1e-9
                    for ds in shapes
                ) if q0 else any(
                    # No anchor strike: the "centre" value is the payoff at the
                    # axis, which sits between the two innermost strikes.
                    payoff_profile(q0, tail, ds).max()
                    > _value_at_axis(q0, tail, ds) + 1e-9
                    for ds in shapes
                )
                found.append(Family(
                    key=f"N{n_strikes}q{'_'.join(str(q) for q in (q0,) + tail)}",
                    q0=q0, tail=tuple(tail), n_strikes=n_strikes,
                    twin_peaked=bool(twin), label="",
                ))
    return found


def _value_at_axis(q0: int, tail: tuple[int, ...], offsets) -> float:
    """Payoff at the axis of symmetry, for families with no anchor contract."""
    strikes = [0.0] + [float(d) for d in offsets] + [-float(d) for d in offsets]
    qty = [q0] + [tail[t] for t in range(len(offsets))] * 2
    qty = [q0] + [tail[t] for t in range(len(offsets))] + [tail[t] for t in range(len(offsets))]
    return float(sum(q * max(k - 0.0, 0.0) for q, k in zip(qty, strikes)))


#: Human labels, attached after enumeration so the names cannot influence which
#: shapes exist. Anything unnamed is a shape the program had not written down.
_LABELS = {
    "N3q-2_1": "long put butterfly — centre peak, zero at +/-d1",
    "N4q0_-1_1": "long put condor — flat top between +/-d1, zero at +/-d2",
    "N5q-4_1_1": "wide butterfly on five strikes — centre peak, two ramps",
    "N5q-2_-1_2": "centre-peaked five, doubled wings",
    "N5q2_-2_1": "TWIN-P5 — twin peaks at +/-d1, wings +/-d2 (d2 >= 2*d1)",
    "N7q-2_-1_1_1": "centre-peaked seven, stepped ramp",
    "N7q-2_1_-1_1": "centre-peaked seven, notched",
    "N7q2_-1_-1_1": "TWIN-P — plateau d1..d2, wings +/-d3 (TWIN-P at 1,2,4)",
}

FAMILIES: tuple[Family, ...] = tuple(
    Family(f.key, f.q0, f.tail, f.n_strikes, f.twin_peaked,
           _LABELS.get(f.key, "unnamed — enumerated, never written down"))
    for f in sorted(_enumerate_families(), key=lambda f: (f.n_strikes, f.key))
)

FAMILY_BY_KEY = {f.key: f for f in FAMILIES}


def reduce_offsets(offsets: tuple[int, ...]) -> tuple[int, ...]:
    """Offsets divided by their gcd — the scale-free SHAPE of a candidate.

    ``(2, 4, 8)`` and ``(1, 2, 4)`` are the same tent at two widths, and the
    tallies the experiment reports are per shape, so they must collapse.
    """
    g = math.gcd(*offsets) if len(offsets) > 1 else offsets[0]
    return tuple(int(d // g) for d in offsets)


@dataclass(frozen=True)
class Pattern:
    """A family at one spacing and one anchor — everything but the ticker.

    A ``Pattern`` is still scale-free in dollars: its offsets are LADDER
    POSITIONS, so what it costs and whether it is even listed depends on the
    event it is resolved against.
    """

    family: Family
    offsets: tuple[int, ...]
    anchor_offset: int

    @property
    def shape_key(self) -> str:
        """Identity for the tallies: family plus the reduced spacing ratio."""
        return f"{self.family.key}@{':'.join(str(d) for d in reduce_offsets(self.offsets))}"

    @property
    def key(self) -> str:
        return (f"{self.family.key}@{':'.join(str(d) for d in self.offsets)}"
                f"+{self.anchor_offset}")


def patterns(max_pos: int = MAX_LADDER_POS,
             anchor_offsets=ANCHOR_OFFSETS) -> tuple[Pattern, ...]:
    """Every (family, spacing, anchor) that exists before an event is seen.

    The zero-floor check is deliberately NOT applied here, and the reason is a
    trap worth stating: offsets are LADDER POSITIONS and the floor is a
    condition on DOLLARS. On a uniform ladder the two agree and pruning here
    would be free; on a ladder that widens away from the money they do not.
    ``(2, -2, 1)`` at positions ``(2, 3)`` fails the floor on a uniform ladder
    (3 < 2x2) and passes it on a ladder listing 2.5-dollar strikes near the
    money and 10-dollar strikes outside, where those positions are 5 and 20
    dollars out. Pruning on positions would silently delete real candidates
    from exactly the tickers whose ladders this experiment exists to
    accommodate.

    So every combination is emitted and ``build.py`` re-checks the floor per
    event on the strikes the chain actually lists. The cost is ~2,000 extra
    rows in a mask that is computed with two gathers.
    """
    out: list[Pattern] = []
    for family in FAMILIES:
        for offsets in itertools.combinations(range(1, max_pos + 1), family.tiers):
            for anchor in anchor_offsets:
                out.append(Pattern(family, tuple(offsets), int(anchor)))
    return tuple(out)


# --------------------------------------------------------------------------
# the equivalence bridge
# --------------------------------------------------------------------------


def to_structure(pattern: Pattern, *, entry_offset: int = 0, exit_offset: int = 1):
    """The same candidate as an ``engine.structures.Structure``.

    This exists for ONE purpose: to let the run price a sample of candidates
    through ``engine.structures.price_structure`` and assert that the fast
    linear pricer in ``build.py`` agrees to floating point. The program's
    load-bearing rule is that there is exactly one pricing path; a second one
    is only tolerable while it is continuously proved to be the first one.

    Families with no anchor contract (the four-strike condor) cannot be built:
    ``LegSpec`` requires a positive qty, and the mirror selector needs a leg at
    the axis to mirror about. Those return ``None`` and the run reports the
    share of the check that had to be skipped rather than quietly narrowing it.
    """
    from engine.structures import PUT, BUY, SELL, ExpirySelector, LegSpec, StrikeSelector, Structure

    if pattern.family.q0 == 0:
        return None
    expiry = ExpirySelector(kind="first_post_event")
    side = BUY if pattern.family.q0 > 0 else SELL
    legs = [LegSpec("atm", PUT, side, expiry,
                    StrikeSelector("bracket", side="below",
                                   steps=pattern.anchor_offset or None),
                    qty=float(abs(pattern.family.q0)))]
    for t, d in enumerate(pattern.offsets):
        q = pattern.family.tail[t]
        leg_side = BUY if q > 0 else SELL
        legs.append(LegSpec(f"up{t+1}", PUT, leg_side, expiry,
                            StrikeSelector("grid_step", ref="atm", steps=int(d)),
                            qty=float(abs(q))))
        # Mirrored in DOLLARS about the anchor and required to be listed —
        # the same discipline twin_peak() uses, and the reason the tails
        # cancel exactly rather than nearly.
        legs.append(LegSpec(f"dn{t+1}", PUT, leg_side, expiry,
                            StrikeSelector("mirror", ref=f"up{t+1}", about="atm"),
                            qty=float(abs(q))))
    return Structure(
        name="EXP133",
        description=f"enumerated symmetric put structure {pattern.key}",
        legs=tuple(legs),
        entry_offset=entry_offset,
        exit_offset=exit_offset,
        params={"family": pattern.family.key, "offsets": list(pattern.offsets),
                "anchor_offset": pattern.anchor_offset},
    )


if __name__ == "__main__":
    print(f"{len(FAMILIES)} families on {STRIKE_COUNTS} strikes, "
          f"<= {MAX_CONTRACTS} contracts\n")
    for f in FAMILIES:
        ds = (1,) if f.tiers == 1 else ((1, 3) if f.tiers == 2 else (1, 2, 4))
        while not offsets_admissible(f.q0, f.tail, ds):
            ds = tuple(d * 2 if i == len(ds) - 1 else d for i, d in enumerate(ds))
        prof = payoff_profile(f.q0, f.tail, ds)
        print(f"  N={f.n_strikes}  q=({f.q0},{','.join(str(q) for q in f.tail)})"
              f"  {'TWIN' if f.twin_peaked else 'centre'}  d={ds}  payoff={list(prof)}")
        print(f"        {f.label}")
    pats = patterns()
    print(f"\n{len(pats):,} (family, spacing, anchor) patterns at "
          f"max ladder position {MAX_LADDER_POS}")
