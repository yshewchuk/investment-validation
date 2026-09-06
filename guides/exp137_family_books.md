# EXP-137 — one book per family: which structures are actually alive

*2026-09-06. Exploratory. All four registered criteria pass; **one of my own
registered predictions was falsified.** Nothing promoted. Artifacts:
`experiments/EXP-137_one_book_per_family_which_enumerated_str/`.*

## 1. Why this had to be run

EXP-134's book splits 42.7% condor, 17.8% wide-five butterfly, 14.5% TWIN-P5,
and so on. Those shares say which family **wins an argmax against seven
rivals**. They do not say which family can carry a book, and the two questions
have different answers: a family that is second-best on every single event
shows a share near zero and may be a perfectly good strategy.

So each family gets its own complete book here — on every event it can carry,
it trades, at the width and anchor its own rules pick. Same universe, same
filters, same gate machinery, same conditional exit, same margin. Eight
strategies, not one chooser.

## 2. The like-for-like comparison

The **common universe** — 4,641 events on which all eight families are
tradeable at once — so every row below covers identical events and identical
count. Ungated, so this is the unselected population: no gate, no chooser.

| family | payoff | mean | median | **on capital** | Sharpe | years+ |
|---|---|---:|---:|---:|---:|---:|
| `(0,−1,1)` long put condor | centre | +12.9% | +18.6% | **+7.0%** | 4.18 | 9/9 |
| `(2,−1,−1,1)` **TWIN-P** | **twin** | +10.5% | +20.8% | **+6.6%** | 3.44 | 9/9 |
| `(2,−2,1)` **TWIN-P5** | **twin** | +10.3% | +15.0% | +4.9% | 3.13 | 9/9 |
| `(−2,1,−1,1)` notched seven | centre | +8.2% | +13.6% | +4.1% | 2.58 | 9/9 |
| `(−2,−1,2)` centre five, doubled wings | centre | +6.1% | +10.3% | +1.9% | 1.78 | 9/9 |
| `(−4,1,1)` wide butterfly on five | centre | +6.0% | +4.4% | +1.5% | 1.58 | 7/9 |
| `(−2,−1,1,1)` stepped-ramp seven | centre | +4.6% | +3.4% | **−0.1%** | 1.23 | 7/9 |
| `(−2,1)` the butterfly | centre | +4.5% | +6.2% | **−0.3%** | 1.38 | 7/9 |

Zero defined-risk failures in any of the eight, at any alpha.

## 3. Two families are dead

**`(−2,−1,1,1)` stepped-ramp seven** and **`(−2,1)` the three-strike
butterfly** both return *negative on capital* on identical events, and both are
positive in only 7 of 9 years. Their positive per-trade means (+4.6%, +4.5%)
are the giveaway: mean above zero while capital return is below it means the
edge sits in the cheapest contracts and vanishes once dollars are counted.

The butterfly is the most-offered family of all — admissible on 11,678 events,
more than any other, because three strikes fit almost any ladder. Availability
and viability are unrelated.

**Recommendation: exclude both from any future chooser.** That also shrinks the
max-statistic the chooser is exposed to, which is the separate problem EXP-134
left open.

## 4. My registered prediction about the notched seven was wrong

EXP-137's spec states, before the run:

> *"Prior expectation, stated so it cannot be claimed afterwards: this family
> is dead and the filter already killed it."*

That was about `(−2,1,−1,1)`, the notched seven, on the grounds that three of
EXP-133's four most extreme arbitrage-violation trades were this family and it
earned only +0.5% in EXP-134.

**It is not dead.** On its own book it returns +8.2% per trade, **+4.1% on
capital**, Sharpe 2.58, positive in 9 of 9 years — fourth of eight and
comfortably alive. Its +0.5% in EXP-134 was precisely the
conditional-on-winning-an-argmax artifact this experiment was built to detect. I
predicted the artifact and then got its direction backwards: winning an argmax
was making this family look *worse*, not better, because the events where it
beat seven rivals were the events where its estimate was most inflated and its
realized result least.

That is the clearest single vindication of running the experiment at all, and
it is the reason argmax share must never be read as evidence about a family.

## 5. TWIN-P beats TWIN-P5 on a like-for-like universe

On identical events with the width chosen per event, **TWIN-P (seven strikes)
returns +6.6% on capital against TWIN-P5's +4.9%**, with a higher Sharpe (3.44
vs 3.13) and the same 9/9 years. Gated, the gap widens: +17.8% against +7.1%.

EXP-126 replaced TWIN-P with TWIN-P5, and its stated reason was universe size —
seven strikes reached 90 events where five reached 393. That reason is
unaffected here (TWIN-P is offered on 6,815 events, the fewest of any family;
TWIN-P5 on 8,771). But *per event it can actually trade*, the seven-strike
shape is the better structure, and EXP-126 never measured it with a
per-event width because it sized both from one formula.

This does not overturn the promotion — a narrower universe was a legitimate
reason to prefer TWIN-P5, and nothing here is a confirmatory test. It does mean
the trade-off should be re-examined with dynamic width on both sides.

## 6. What the gate does

Gated to the trailing six-month top 20%, on the common universe, all eight
families improve and the ordering is broadly preserved: condor +22.2% on
capital, TWIN-P +17.8%, notched seven and stepped-ramp seven +10.7%, TWIN-P5
+7.1%, butterfly +6.8%. Both dead families turn positive under the gate,
which is what a gate is for — but a family that needs the gate to be positive
at all is a gate result, not a structure result, and neither should be
registered on that basis.

## 7. Caveats

- **Eight arms on one universe is eight chances for one to look good.** Every
  arm is in the ledger, none is promotable on this run.
- **Six of the eight are centre-peaked and want a quiet print.** This compares
  two theses, not eight variants of one, and the tables are grouped by payoff
  shape for that reason.
- **The common universe is a selected slice** — 6,689 of 23,583 events, biased
  toward tickers with dense ladders, since a seven-strike structure has to fit.
- **Still a chooser inside each family.** Width and anchor are chosen per event
  by simulated expected P&L, so each book carries a smaller version of
  EXP-134's max-statistic problem.
- Pricer verified: 136 events across all nine years, max difference 5e-14.

## 8. What follows

1. **Drop `(−2,−1,1,1)` and `(−2,1)`** from the candidate set.
2. **Re-open TWIN-P vs TWIN-P5 with dynamic width on both** — a like-for-like
   confirmatory test EXP-126 could not run.
3. The condor family remains the strongest single book and is CND-P's shape,
   already in `STRUCTURES` — so the promotable object, if any, is a generalised
   CND-P with a per-event width, not a new structure.
