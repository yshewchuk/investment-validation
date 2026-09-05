# EXP-133 — searching every structure the ladder can carry

*2026-09-05. Exploratory. The registered primary was **falsified**; nothing is
promoted and nothing on the live board changes. Artifacts:
`experiments/EXP-133_every_symmetric_put_structure_the_ladder/`.*

## The question

TWIN-P5 replaced TWIN-P because five strikes bought the same peak for less and
quadrupled the traded universe. The natural next step: keep going. Fewer
strikes need less ladder granularity, so four and three should fit more tickers
still; and picking the structure *per event* — at a width chosen per event
rather than by one sizing formula — should collect P&L a single fixed shape
leaves behind.

So: enumerate every structure the rules admit, price all of them on every
event, let each event pick the one with the highest simulated expected P&L, and
gate the result with EXP-131's trailing six-month top-20% rule.

## Answer 1 — the algebra, before any data

Enumerating every all-put structure that is mirror-symmetric about a listed
strike, whose contracts sum to zero, whose payoff floor is zero, and which fits
in eight contracts, on 3/4/5/7 strikes, gives **exactly eight families**:

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

**Only two families are twin-peaked, and neither has fewer than five strikes.**
Dropping from five strikes to four does not buy a cheaper twin peak — it buys a
*different thesis*, a tent that pays most at a dead-flat print. This is a fact
about the payoff algebra, not a finding, and it holds before any price is
quoted. It is pinned by `test_family.py`; the reasoning is in `family.py`.

That answers the motivating intuition on its own terms: the granularity saving
is real, but what the saved strikes buy is not a twin peak.

## Answer 2 — the search was falsified, three times over

23,583 events priced, 12,600 candidate patterns each, 2018–2026. Three of the
six registered acceptance criteria fail:

| criterion | result | |
|---|---|---|
| more_names | **PASS** | 1,506 trades on 486 tickers against the incumbent's 572 / 324 |
| universe_floor | **PASS** | ≥ 250 traded events |
| still_pays | **PASS** | breakeven α 0.274 |
| **beats_the_incumbent** | **FAIL** | CAGR −100% against +40.8% |
| **choosing_beats_not_choosing** | **FAIL** | a uniformly random admissible candidate beats it |
| **defined_risk_holds** | **FAIL** | 4.5% of chosen trades lose more than their debit |

The chooser DOES widen the universe 2.6×, exactly as hoped. Everything else
about it is an artifact, and the artifact has one cause.

### The cause: a ratio with an unreal denominator

The rule maximises `(E[exit value] − cost) / cost`, which is unbounded as the
debit goes to zero. An eight-leg structure whose longs and shorts nearly cancel
can net to almost anything, so the argmax walks straight into the cheapest net
debit the ladder can produce. Median chosen debit **$0.18** against the
incumbent's **$5.60**; 17.7% of picks net under $0.25 across eight legs against
0.0% of the incumbent's.

This is EXP-126's `choose_rr` failure arriving through a better criterion.
Ranking by reward:risk was cheapness-seeking; ranking by simulated expected
*return* is cheapness-seeking too, because the denominator is the same.

**The Monte Carlo is not the problem.** Draws 1–2,000 chose the candidate and
draws 2,001–4,000 supplied the number the gate ranked: the gap is +1.77pp on
+1,338%. Four thousand draws is plenty and the winner's curse in the *estimate*
is negligible.

**Nor is Black-Scholes badly wrong.** Realized tracked simulated closely in
every wing-distance bucket (+1,178% realized against +1,249% simulated at 5–10%
of spot). The registered consistency test does fire — Spearman ρ = +0.082,
p = 0.0014 between an event's candidate count and its (simulated − realized)
gap, so optimism *does* grow with the number of candidates — but the effect is
small next to the denominator problem.

### The filter that could not see it

The worst trade names the mechanism. MDT, 2024-05-23: the seven-strike twin
peak at $1 spacing, max payoff $2.00 at expiry, entry mid debit **half a cent**.
Six of its seven legs quote inside 5%. The seventh — the $90 wing — quotes
2.85 / 4.85, two dollars wide on a $3.85 mid, and that single leg's mid produced
the half-cent net. Its **mean relative leg spread is 11.4%**, comfortably inside
the registered 25% filter.

A mean over legs cannot see that the NET is a small difference of large numbers.
The quantity that can is **spread-to-debit**: total entry half-spread divided by
the net debit — how uncertain the price is, in units of the price.

| | median mean-leg spread | median spread-to-debit | picks where uncertainty > price |
|---|---:|---:|---:|
| the chooser | 0.147 | **1.31** | **57.6%** |
| TWIN-P5 incumbent | 0.148 | **0.24** | 2.4% |

Identical on the filter the program uses; 5× apart on the one it does not have.
This applies to TWIN-P and TWIN-P5 as well — their spread-to-debit p95 is 0.81
and max 7.43 — so it is not only a fact about this search.

### And defined risk is a claim about expiry

Max loss is the debit *at expiry*: the contracts sum to zero over exactly
mirrored strikes, so the deep-ITM tail is flat and the floor is zero. The exit
is not at expiry — median 9 DTE remain — and it is priced by selling the longs
and buying back the shorts at real quotes, which can net below zero. That
happens on 0.3% of the incumbent's trades and **4.5%** of the chooser's, worst
case −318× the debit. The search finds the events where the claim stops holding,
because those are the events where mid is least real.

## Answer 3 — post-hoc: what survives once the price is knowable

*Written after the primary ran, in response to what it did. Not registered, not
promotable, reported as a sweep rather than a chosen level. Full table in
`REPORT.md` §8.5.12.*

Capping spread-to-debit and re-running the same chooser:

| cap | n | median debit | mean/trade | on capital | mean at α=0.25 | Sharpe | years+ | centre-peaked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none (registered) | 1,506 | $0.18 | +1188% | +170% | −123% | 4.22 | 9/9 | 37% |
| ≤ 4× | 1,478 | $0.60 | +112% | +83% | −87% | 8.68 | 9/9 | 54% |
| ≤ 2× | 1,465 | $0.95 | +65% | +48% | −63% | 6.55 | 9/9 | 68% |
| ≤ 1× | 1,434 | $1.57 | +35% | +28% | −42% | 4.52 | 9/9 | 80% |
| ≤ 0.5× | 1,305 | $2.47 | +15% | +11% | −29% | 2.11 | 9/9 | 88% |
| ≤ 0.25× | 1,066 | $2.97 | **+6.4%** | **+3.2%** | −20% | **0.82** | 7/9 | **94%** |
| *TWIN-P5 incumbent* | *572* | *$5.60* | *+10.5%* | *+6.2%* | *−26%* | *2.04* | *9/9* | *0%* |

Every column is a monotone function of how unknowable the entry price is. At
0.25× — roughly where the incumbent already lives — the search delivers **+6.4%
per trade against the incumbent's +10.5%, +3.2% on capital against +6.2%,
Sharpe 0.82 against 2.04, and 7/9 years against 9/9** — while trading 1.9× as
many events.

And look at the last column. As the price becomes real the chooser migrates from
37% centre-peaked to **94%**. At the only cap where the money is believable, the
"twin-peak search" is buying condors and butterflies almost exclusively.

**The whole apparent advantage of searching was a function of price
uncertainty. The twin-peak thesis survives; the search does not.**

## What this changes

**Nothing on the board.** TWIN-P5 stays the `twin-peak` champion, unchanged, and
no new structure was registered in `STRUCTURES`.

**One thing worth carrying forward, and it is not the search.** The program has
no filter on spread-to-debit, and `engine.fills.MIN_MEANINGFUL_COST` (1e-6) was
calibrated on CND-P's four legs — the eight-leg noise band is orders of
magnitude higher. Every multi-leg debit structure the program prices is exposed
to a net debit that is a small difference of large numbers, and the mean-leg
spread filter cannot see it. That is a candidate for the entry rules regardless
of what happens to structure search.

**Two things the search did establish, cleanly:**

- The width rules bite far harder than the ladder does. The median event lists
  610 candidate geometries and **47.3% of events have none at all** that both
  span the predicted move and stay inside three SDs — the binding constraint
  has moved from "the ticker lists the strikes" (TWIN-P's problem) to "the
  forecast's error band is wide enough that almost nothing fits".
- Given the choice, the chooser goes NARROW — median width 1.62× the predicted
  move against the incumbent's 2.87× — and pays for it: 24.8% of prints land
  beyond a wing against the incumbent's 7.2%, which matches EXP-126's published
  7.6% for TWIN-P5 and is the check that this run's incumbent arm is the real
  one.

## What a successor should register before it runs

1. A **spread-to-debit cap** as an entry filter, pre-registered at a level
   chosen from 2018–2022 alone and tested on 2023–2026, the way EXP-131 chose
   its window. Not from this run's curve.
2. A **selection objective that is not a ratio on the debit** — expected P&L in
   dollars per unit of *risk*, where risk is the debit but the ranking is not
   divided by it — so that cheapness is not the thing being maximised.
3. **A much smaller candidate set.** The consistency test fired at ~940
   candidates per traded event. Any successor should register the candidate
   count as a treatment, not let it float.

## Two disclosures

- **LEDGER.csv carries seven RAN rows for EXP-133 dated 2026-09-05 that came
  from a pipeline smoke test on 2018 alone**, written before the full universe
  had ever been priced. They are left in place rather than removed — the ledger
  is append-only by discipline — and `run.py` now takes `--no-ledger` so it
  cannot happen again. The full-universe evaluation reused those spec hashes and
  therefore did not write new rows; the numbers in the ledger are the smoke
  test's, and the numbers in `REPORT.md` are the real ones.
- `tests/test_score.py::TestStructureParams::test_priced_score_carries_reproducible_rule_and_request_params`
  fails on the working tree as found (uncommitted ledger-settlement work). It
  is unrelated to this experiment and was not touched. Everything else passes:
  1,556 repo tests plus 42 new ones in `test_family.py`.
