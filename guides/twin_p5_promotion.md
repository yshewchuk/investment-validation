# TWIN-P5 replaces TWIN-P — the decision, and what it cost

*2026-09-04. Decided by the user; recorded by EXP-126. Rollback is one edit to
`engine/models/structures.json`.*

## What changed

The board's twin-peak structure is now **TWIN-P5** with `wing_multiple=3` —
five listed strikes instead of seven, the same eight contracts, the same all-put
single-expiry construction, the same zero floor. TWIN-P is **superseded**, not
disabled: it stays in `STRUCTURES`, it keeps its entry rule, and EXP-123/124/125
still replay against it.

## Why

EXP-126 priced all three twin-peak shapes over one event universe of 25,123
events, sized per event by the same walk-forward OOS forecast, gated by the same
three arithmetic terms.

| | TWIN-P (seven) | **TWIN-P5 (wing 3)** |
|---|---:|---:|
| CAGR | 6.14% | **12.98%** |
| Sharpe (trade) | 0.85 | **1.10** |
| OOS years positive | 6/9 | **9/9** |
| crisis regimes positive | 2/4 | **4/4** |
| trades / tickers | 90 / 79 | **393 / 217** |
| return on capital | +5.52% | **+7.77%** |
| prints beyond a wing | 16.7% | **7.6%** |
| max drawdown | **9.8%** | 14.3% |
| breakeven alpha | **0.444** | 0.467 |

`promote.decide` returns all-green on the current rules.

## What this promotion is NOT

**It is not a confirmatory result.** EXP-126's registered primary was the
per-event chooser (`choose_rr`), and it was **falsified** — it traded 3,673
events for a CAGR of −0.52% and a 79% drawdown, because ranking shapes by
reward:risk is a cheapness-seeking rule that lands on the worst shape 92% of the
time. TWIN-P5 was a **grid cell**. The size forecast placing its strikes is
walk-forward out-of-sample, but *the choice of wing multiple saw all nine years*.
That is post-hoc selection over five arms, and it is the reason the promotion
note exists rather than a line in a changelog.

**It is worse on two measures.** Max drawdown deepens (14.3% against 9.8%) and
breakeven alpha rises to 0.467 — it needs 46.7% of the spread where TWIN-P
needed 44.4%, against a mid assumption of 50%. Tightening the traded universe to
markets quoted inside 15% keeps the edge (240 trades, +8.05% on capital, still
9/9 years) but does not fix the fill margin (0.461). At a 10% cap the edge
degrades badly (4/9 years on 116 trades) — so some of it does live where mid is
optimistic.

**The forward test is the missing evidence.** Live prints are the one dataset
the shape choice cannot have contaminated.

## Two things this decision changed underneath it

**The promotion measure.** Rule (a) compared challengers on **mean per trade**
and Sharpe. Mean per trade is now gone from the decision, replaced by CAGR,
Sharpe, and share of OOS years positive. EXP-126 is why: the chooser arm
reported mean **+0.49%**, which reads as profitable, while the book it managed
lost 0.5% a year and drew down 79%. A measure that calls that outcome positive
cannot be the one a promotion turns on. Max drawdown now WARNs rather than
blocks — a structure trading 4.4x as often carries a deeper absolute drawdown at
the same fractional sizing for reasons unrelated to edge quality, and Sharpe plus
the MC P(loss) rule already price risk.

**Structures got a champion role.** There wasn't one. `ROLES` covers models —
`size`, `implied_t1`, `gate` — one champion per `(strategy, role)`, and a
structure has no artifact to hash, no features, no training window. The first
attempt at this promotion put TWIN-P into `DISABLED_STRATEGIES`, which took it
off the board and stamped it `UNVALIDATED_STRUCTURE` — the opposite of the truth
for the most-measured structure in the program. `engine/structure_registry.py`
replaces that improvisation: structures that compete for the same events with
the same forecast form a **family**, exactly one member is live, and the rest are
`SUPERSEDED` — a flag that says which shape won and points at the evidence.

## How to roll this back

Edit `engine/models/structures.json` so the `twin-peak` champion is `TWIN-P`,
or call `promote_structure("twin-peak", "TWIN-P", evidence="...")`. Nothing else
has to move: TWIN-P never left `STRUCTURES`, its entry rule is intact, and its
forecast sizing rule is unchanged.

## Open

- **EXP-127**, confirmatory: TWIN-P5 as the registered primary, with the 0.45
  breakeven bar and a ≤15% tight-market requirement pre-registered.
- **Forward test** (Phase 5): commit TWIN-P5 to the live protocol so evidence
  accumulates on prints the shape choice never saw.
- The `zero_cost` abort in `engine/replay.py` still deletes a whole event across
  every fill alpha when one alpha prices at ≤$0. Measured as immaterial to the
  traded universe (0 of 50 and 0 of 44 deleted events would have passed the
  filters for the seven- and wide-five-strike shapes; 10 of 124 for the tight
  one), so it was left alone deliberately.
