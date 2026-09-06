# Two legs on one strike — the coarse-ladder collision

*Found 2026-09-06 by reading the live board, not by a test. Fixed in
`engine/structures.py::price_structure`.*

## What was on the board

KEN, event 2026-09-07, CND-PS. Spot 68.98, a $5 strike ladder, and the
structure sized at `width_moneyness = 0.0146`:

```
atm   buy  0.0  P 65      <- reference leg, no contracts
up1   sell 1.0  P 70
dn1   sell 1.0  P 60
up2   buy  1.0  P 70      <- same contract as up1
dn2   buy  1.0  P 60      <- same contract as dn1
```

Sell 70 and buy 70. Sell 60 and buy 60. A position of exactly nothing, priced
at a net debit of exactly nothing, and ranked against real structures.

ABM, event 2026-09-08, is the same defect wearing a different face. Its BFLY-P5
resolved `up1` and `up2` both to P 50 and `dn1`/`dn2` both to P 40 — which is
not a wash, because both are buys. What it is instead is **exactly two of the
same event's BFLY-P**: same three strikes, cost 3.95 against 1.975. A
three-strike butterfly at double size, carrying a five-strike structure's
`peak_multiple = 4`.

## Why it happened

`StrikeSelector("offset_from")` already guards against collapse, and its own
comment says so:

> Excluding the anchor keeps a spacing smaller than one grid step from snapping
> back onto it and collapsing two of the condor's strikes into one.

**That guard is one-dimensional.** It excludes the anchor's strike and nothing
else. `_symmetric_put_ladder` places `up1` at `moneyness * 1` and `up2` at
`moneyness * 2` as two INDEPENDENT offsets from the same anchor — neither
selector knows the other exists. On a ladder coarser than the inner width both
snap to the first listed rung above the anchor, and `dn1`/`dn2` mirror the
collision downward.

So the families at risk are exactly the ones that place two offsets from one
anchor: **CND-PS and BFLY-P5**. TWIN-P5 escapes because its wings are placed by
`mirror`, which refuses a strike that is not listed and cannot land on an
occupied one.

## How wide it was

On the 2026-09-06 board, **70 of 1,648 rows**:

| strategy | rows | shape |
|---|---:|---|
| CND-PS | 30 | complete wash — every contract cancels, net debit 0 |
| BFLY-P5 | 30 | exactly 2x the same event's BFLY-P, all 30 |
| DYN-SV | 10 | the chooser picked a collided BFLY-P5 |

**None of the 70 passed the entry gate, so nothing would have been traded.**
That is the materiality answer and it is the only reassuring number here.

The unreassuring one is the chooser. DYN-SV ranks on simulated expected P&L,
and in all 10 cases the collided BFLY-P5 **outranked** the real BFLY-P it was a
doubled copy of — `+0.0318` against `-0.0088` on DSGX, `+0.0539` against
`+0.0249` on TEN, and so on for all ten. A position identical up to scale
should score identically on a return measure; it did not, because
`peak_multiple` still said 4. The phantom structure was systematically
preferred to the real one.

## The fix

`price_structure` now refuses any resolution in which two legs land on the same
`(right, strike, expiry)`. Refusal rather than adjustment is the house style
already set by the `mirror` selector, which raises rather than snapping a wing
to the nearest listed strike — and for the same reason: the geometry is what
the defined-risk claim rests on, so a ladder that cannot carry the shape has
not produced a cheaper version of it.

Stated as the requirement rather than the mechanism: **every two legs of the
same right and expiry must be at least one strike width apart.** That is the
same condition as contract-distinctness here, because every selector returns a
LISTED strike and `price_structure` refuses a leg with no matching chain row —
so two distinct leg strikes are one grid rung apart by construction.

Reference legs (`REFERENCE_QTY`, the zero-quantity anchor CND-PS carries so its
mirrors are exact) are included in the check. A reference leg sitting on a
traded strike means the symmetry it exists to define has already collapsed.

## What the tests could not have caught, and now can

The unit tests passed throughout. They price these families against
`condor_rows` — a uniform $2.50 grid on spot 101 — which is fine enough that no
width in the sweep ever collides. The defect needs a ladder coarse relative to
the width, which is an ordinary condition on a $9 stock or any $5-ladder name,
and no fixture had one.

Two things went in:

- `TestCoarseLadderCollision` reproduces KEN's actual ladder — $5 rungs under
  spot 68.98 — and asserts the refusal, the message naming both legs, and that
  the same structure still prices on a $1 ladder. It also covers BFLY-P5's
  non-netting collapse specifically, because a test written against "the legs
  cancel" would pass that row.
- `TestEveryStructureKeepsItsLegsAStrikeApart` sweeps **all nine** structures
  against five ladder granularities x five widths and asserts the distance
  property on every priced result. With the guard removed it fails 16 cases,
  all CND-PS and BFLY-P5 — which is also the evidence that the other seven
  families were never exposed.

The second is the one that generalises: it states the requirement over the
registry rather than over the two families that happened to be noticed.

## Were the experiments affected? No — and the reason is structural

The fix is in `engine/structures.py`, the ONE pricing path, so the question is
real: a defect there is not confined to the dashboard. It is confined anyway,
by a difference in how the two paths size a structure.

**The board sizes in dollars.** It sets `width_moneyness` per event from the
forecast, which switches `_symmetric_put_ladder` from `grid_step` to
`offset_from` — and it is only in that mode that two legs are placed by
independent dollar offsets from one anchor:

| family | `width_moneyness=None` | width set |
|---|---|---|
| CND-PS | `grid_step` x2 | **`offset_from` x2 off `atm`** |
| BFLY-P5 | `grid_step` x2 | **`offset_from` x2 off `atm`** |
| TWIN-P, TWIN-P5 | `grid_step` x1 | `offset_from` x1, rest `mirror` |
| CND-P | `offset_from` x1 | (takes no `width_moneyness`) |

Only CND-PS and BFLY-P5 have two, and only with a width set. Every other family
places at most one offset and mirrors the rest, and `mirror` refuses a strike
that is not listed.

**The experiments size in ladder positions.** EXP-134, 137, 138, 139 and 141 all
`import family` from EXP-133, whose `to_structure` builds every candidate from
`bracket` + `grid_step` + `mirror` and takes its offsets from
`itertools.combinations`, which yields strictly increasing distinct integers.
Distinct positions map to distinct rungs. The collision cannot be expressed.

Checked rather than reasoned:

- All **12,600** enumerated patterns: 0 request two legs at one ladder
  position, 0 request a leg at the anchor's own position, and the selectors
  across all 12,030 buildable ones are `bracket` 12,030 / `grid_step` 34,260 /
  `mirror` 34,260 — **`offset_from` appears zero times.**
- The four families the experiments replay with a dollar width — TWIN-P,
  TWIN-P5 at m=3 and m=2, CND-P — priced across 5 ladder granularities x 6
  spots x 7 widths, 840 combinations: **0 collisions.**
- Every experiment that passes `width_moneyness` passes it to `twin_peak` or
  `twin_peak_5` (EXP-125, 126, 133, 134, 137, 138, 139). No experiment names
  CND-PS or BFLY-P5 at all — those two were added to the registry on
  2026-09-06 for forward tracking and have never been backtested.

So no published experimental result moves. CND-PS and BFLY-P5 were only ever
priced by the board, which is exactly where the 70 rows were found.

## COARSE_LADDER, because NO_CHAIN was the wrong thing to say

A refused row used to be flagged `NO_CHAIN`, and for these 70 rows that is not
merely imprecise, it is actionable and wrong: `NO_CHAIN` tells a reader to go
and re-pull quotes, and the UI even renders the age of the newest chain beside
it so they can judge whether a refresh helps. Here the chain is present and
correct. Nothing about a refresh changes the answer — only a denser ladder or a
wider structure does.

So the collision now raises `LadderTooCoarse(StructureError)`. A subclass
rather than a message, because three callers have to tell it apart and a string
comparison is not a contract. It stays a `StructureError`, so `UNSCORABLE` and
every existing `except` still catch it and nothing that used to be handled
stops being handled.

Three places consume the type:

- `Scorer._price` flags `COARSE_LADDER` and — new — writes the exception
  message to `detail`. That branch previously set no detail at all, so a refused
  row reached the board saying nothing about why it was refused.
- `unscorable_result`, the shared placeholder the dashboard self-check
  re-scores through, picks the flag from the exception type so both paths reach
  the same one from the same failure.
- `app.js` renders it as `COARSE_LADDER (strikes too far apart)` with the
  colliding legs on hover, alongside the existing `NO_CHAIN (newest Nd old)`.

**And a second NO_CHAIN, further down, that the first fix missed.** The
expected-P&L step flags `NO_CHAIN` whenever there is no entry cost — which is
of course true of a row whose pricing was refused. So the first version of this
change produced rows carrying BOTH flags, and the board would still have said
NO_CHAIN. A row that was already refused now keeps the reason it was refused
for. The test that caught it is the one asserting `"NO_CHAIN" not in flags`
rather than only `"COARSE_LADDER" in flags` — worth remembering that asserting
the presence of the new thing would have passed.
