# EXP-133 — searching every structure the ladder can carry

*2026-09-06. Exploratory. Nothing promoted, nothing on the board changed.
Artifacts: `experiments/EXP-133_every_symmetric_put_structure_the_ladder/`.*

> **This guide was rewritten after its first version was wrong.** The first
> version told the user the search failed because it maximised a ratio with an
> unreal denominator. That is a symptom. The cause is that the search finds
> places where the vendor's MID surface violates no-arbitrage and books the
> violation as free money. The first version was also written on a build
> carrying two bugs of mine, one of which was load-bearing for its conclusion.
> Both are fixed and the run below is the corrected one. What the earlier
> version got right and wrong is recorded in §6, because a guide that quietly
> replaces its own conclusion is worse than one that never had it.

## 1. The question

Can each event pick its own structure — and its own width — instead of trading
one fixed shape? And do four- and three-strike structures widen the tradeable
universe, the way five strikes did when TWIN-P5 replaced TWIN-P?

## 2. The algebra, before any data

Enumerate every all-put structure that is mirror-symmetric about a listed
strike, sums to zero contracts, has a zero payoff floor, and fits in eight
contracts. On 3/4/5/7 strikes there are **exactly eight families**:

| strikes | contracts | payoff | what it is |
|---|---|---|---|
| 3 | (−2, 1) | centre | the long put butterfly |
| 4 | (0, −1, 1) | centre | the long put condor — CND-P's shape |
| 5 | (−4, 1, 1) | centre | wide butterfly on five strikes |
| 5 | (−2, −1, 2) | centre | centre-peaked five, doubled wings |
| **5** | **(2, −2, 1)** | **twin** | **TWIN-P5**, at every wing ≥ 2× the peak spacing |
| 7 | (−2, −1, 1, 1) | centre | centre-peaked seven, stepped ramp |
| 7 | (−2, 1, −1, 1) | centre | centre-peaked seven, notched |
| **7** | **(2, −1, −1, 1)** | **twin** | **TWIN-P**, at 1:2:4 |

Both incumbents fall out of the enumeration rather than being written into it,
which is the check that those four properties are the right four.

**Only two families are twin-peaked and neither has fewer than five strikes.**
Four strikes and three cannot express a twin peak at all. The granularity
saving is real; what the saved strikes buy is a tent that wants a quiet print —
the opposite thesis. This is payoff algebra, not a finding, and it is pinned by
`test_family.py`.

## 3. What the search actually found

23,583 events, 12,600 candidate geometries each, 2018–2026. Five of six
registered criteria pass; `defined_risk_holds` fails. **None of that should be
read as a result**, for the reason in §4.

| arm | n | tickers | mean | on capital | Sharpe | years+ | centre-peaked |
|---|---:|---:|---:|---:|---:|---:|---:|
| best_all (primary) | 1,430 | 425 | +34.8% | +28.3% | 4.47 | 9/9 | **81%** |
| best_twin_only | 1,138 | 368 | +27.6% | +17.8% | 3.58 | 9/9 | 0% |
| best_twin_p5 | 1,139 | 364 | +20.3% | +11.3% | 2.57 | 9/9 | 0% |
| random_pick | 1,446 | 523 | +5.8% | −1.0% | 1.13 | 8/9 | 87% |
| incumbent TWIN-P5 | 572 | 324 | +10.5% | +6.2% | 2.04 | 9/9 | 0% |

## 4. Why the numbers cannot be believed

**The search is a no-arbitrage-violation detector.** A put curve must be
non-decreasing in strike, convex, and have every vertical worth no more than
its strike width. A vendor MID surface is a smoothed estimate and violates all
three routinely in thin strikes — harmless until something searches over
combinations of those strikes, at which point every violation is free money.

The gradient across arms is the mechanism, because every arm meets the same
surface and only some are able to go looking:

| arm | freedom | entry surface violates no-arb |
|---|---|---:|
| incumbent | none — fixed shape and width | **1.0%** |
| random_pick | random admissible candidate | 22.8% |
| best_twin_p5 | width + anchor, one family | 43.6% |
| **best_all** | everything | **44.6%** |

The clearest single case, on the pre-fix build: FDX 2024-12-19, seven strikes,
guaranteed non-negative payoff with a **maximum of $12.50**, priced by the mids
at **$0.025**. Two convexity violations in that put curve, and the structure the
search picked is exactly the combination that monetises both.

**The 71 defined-risk failures are the same disease at the exit.** The terminal
payoff floor of every violating trade is exactly `0.000000` — the structures are
sound. But 100% of the 71 sit on an EXIT mid surface that violates
no-arbitrage, and 0 of the 741 consistent ones do. PBR 2024-03-07 quotes the
17.50 put *below* the 17.00 put and prices a structure worth between $0 and
$1.50 at **−$2.09**.

**The chooser also reads model error.** Spearman ρ = +0.134 (p = 4e-7) between
an event's candidate count and its simulated-minus-realized return. Optimism
grows with how wide the search is allowed to be.

**And the winning book is not a twin-peak book.** 81% of the primary's trades
are centre-peaked. The thesis inverted underneath the search.

## 5. What was solid

- **The pricer.** 5,460 comparisons across 139 events and all nine years
  against `engine.structures.price_structure`: max difference **5e-14**. The
  expected-P&L shortcut agrees with `engine.pnl_sim.expected_pnl` to 3e-14.
- **The incumbent arm is really TWIN-P5.** Against EXP-126's stored artifact:
  3,975 vs 4,086 events after the same spread filter, and **90.1% of shared
  rows bit-identical**. The residual is EXP-126 bucketing spacing to 0.1% of
  spot.
- **Monte Carlo is not the problem.** Split-half selection: draws 1–2,000 chose,
  draws 2,001–4,000 scored, and the gap is ~1.8pp on a four-figure return.
- **The width rules bind, not the ladder.** 48.5% of events have no admissible
  candidate at all. TWIN-P's constraint was the ladder; this one's is the
  forecast's error band.

## 6. The two bugs, and what the first version of this guide got wrong

**Bug 1 — `zero_cost` applied at mid only.** `engine.replay.replay_one` refuses
an event across *every* alpha when one alpha prices it at or below
`MIN_MEANINGFUL_COST`. This build applied that at mid alone. Two consequences:
the alpha sweep compared 12,437 rows at mid against 5,188 at best fill, making
`breakeven_alpha` an interpolation between two different books; and sub-penny
debits survived. **This was load-bearing.** Fixing it moved the minimum entry
debit from $0.005 to $0.125 and the median from $0.18 to $1.67, and flipped two
registered criteria from FAIL to PASS. The first version of this guide reported
a falsification that was partly my own bug.

**Bug 2 — the equivalence receipt sampled one event** while describing itself as
"a sampled cross-check". Now 139 events across every year.

**What the first version got right:** the enumeration and its consequence for
3- and 4-strike structures; that the incumbent was faithful; that the headline
per-trade numbers were artifacts; that `random_pick` was a necessary null.

**What it got wrong:** the causal story (denominator, not arbitrage
violations); the falsification verdict (contaminated by bug 1); and the
post-hoc remedy — it swept a floor on spread-to-debit, which correlates with
the disease but is not it.

## 7. What this changed

Nothing on the board. TWIN-P5 remains the `twin-peak` champion and no new
structure was registered in `STRUCTURES`.

Three findings carried into **EXP-134**, which registers them as constraints:

1. **A no-arbitrage filter on the entry curve** — checkable at the decision from
   the entry chain alone.
2. **The exit convention is a choice, not an assumption.** Closing into a curve
   that violates no-arbitrage is not a trade anyone would make.
3. **Zero-bid legs are valued at `ask/2` and nothing filters the exit.** Worth
   ~20% of the incumbent's measured edge; a program-level issue, not this
   experiment's.

And one measurement that outlives the experiment: **TWIN-P5's entries are clean
at 1.0% but 30.8% of its EXITS violate no-arbitrage.** Every published
twin-peak result — EXP-125, 126, 129, 131 and the promotion — is marked out
through those quotes on roughly a third of its trades.

## 8. Disclosure

`LEDGER.csv` carries seven RAN rows for EXP-133 dated 2026-09-05 from a
pipeline smoke test on 2018 alone, written before the full universe had ever
been priced. They are left in place — the ledger is append-only by discipline —
and `run.py` now takes `--no-ledger`. The full-universe evaluation reused those
spec hashes and wrote no new rows, so **the ledger's numbers for EXP-133 are the
smoke test's; `REPORT.md` has the real ones.**
