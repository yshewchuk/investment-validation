# EXP-134 — the structure search under three constraints the account has

*2026-09-06. All four registered acceptance criteria pass. Nothing is promoted:
see §5 for the three reasons that would be premature. Artifacts:
`experiments/EXP-134_priced_right_funded_and_held_structure_s/`.*

## 1. What changed from EXP-133

EXP-133 searched 12,600 symmetric put structures per event and its primary
looked like it won. It was finding places where the vendor MID surface violates
no-arbitrage and booking the violation as free money — 44.6% of its traded
entries sat on an inconsistent put curve, against 1.0% for the fixed-shape
incumbent. See `guides/exp133_structure_search.md`.

EXP-134 keeps the candidate set, the width rules, the gate and the objective
**exactly as they were** — the enumeration and pricing code is imported from
EXP-133, not reimplemented — and adds three constraints the real account has
and the backtest did not:

**PRICED RIGHT.** A candidate whose own strikes carry a monotonicity, slope or
convexity violation is not priced. Decided at the entry close from the entry
chain alone, so it is a rule and not hindsight. A candidate is judged on the
sub-curve it trades: a violation four strikes away is somebody else's problem.

**FUNDED.** Every short put is cash-secured in full at `strike × 100` with no
spread offset, held until the position closes, and total secured across all
open positions may not exceed 50% of current marked equity. The account starts
at $100,000 and capacity travels with it.

**HELD.** The exit is not assumed. The registered rule closes at the first
post-print close when that curve is arithmetically consistent, and holds to
expiry when it is not — because closing into a broken quote is not a trade
anyone would make, and the enumerated payoff floor is zero, so settlement is
guaranteed non-negative.

## 2. The result

| arm (conditional exit) | wanted | funded | tickers | mean | on capital | Sharpe | years+ | loses>debit | $100k → |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **best_all** (primary) | 1,428 | **675** | 281 | +31.0% | **+29.5%** | 2.85 | 9/9 | **0** | **$182,842** |
| best_twin_only | 1,125 | 463 | 197 | +26.2% | +25.4% | 2.12 | 9/9 | 0 | $130,404 |
| best_twin_p5 | 1,124 | 451 | 192 | +24.5% | +24.6% | 1.91 | 8/9 | 0 | $131,841 |
| random_pick | 1,435 | 592 | 298 | +8.9% | +8.4% | 1.12 | 7/9 | 0 | $138,106 |
| incumbent TWIN-P5 | 560 | 228 | 153 | +11.3% | +12.2% | 1.51 | 9/9 | 0 | $112,410 |

All four registered criteria pass: beats the incumbent, beats the random null,
defined risk holds, universe floor cleared.

## 3. Each constraint did real work

**The no-arbitrage filter** removes 28.6% of admissible candidates. Entry
violations on the traded book go **44.6% → 0.00%** across every arm. The
zero-bid exit sensitivity that was worth 10–12% of the headline in EXP-133 is
now 1.3pp (+31.0% → +29.7% mean).

**The conditional exit** holds 45% of trades to expiry — exactly the share whose
post-print curve violated no-arbitrage. Defined-risk failures go to zero. The
`close_x1` arm, run on the identical book, still shows 21–28 of them, so the
rule is doing the work and not the luck:

| exit convention | best_all on capital | loses>debit | incumbent on capital | loses>debit |
|---|---:|---:|---:|---:|
| close at x+1 | +32.0% | 21 | +10.7% | 1 |
| **conditional** | **+29.5%** | **0** | **+12.2%** | **0** |
| hold to expiry | +31.1% | 0 | +13.1% | 0 |

**The margin constraint** refuses **53%** of the trades the strategy wanted.
Median $23,000 secured per contract, median 2 contracts, peak concurrency 3.
**100% of position sizes were set by margin**, not by the 5%-of-equity rule the
program has used since Phase 2 — that rule never binds once puts must be cash
secured.

## 4. Where the margin actually bites — and it is not concurrency

Of the 753 refused trades:

| cause | n | median underlying |
|---|---:|---:|
| one contract exceeds 50% of equity even with an empty account | 369 (49%) | **$306** |
| would have fit, but open positions had consumed the headroom | 384 (51%) | $92 |

Half is structural: you cannot cash-secure four puts on a $306 stock with
$100,000, at any width. Ticker price is an entry filter nobody wrote down.

And the account is **idle**, not congested:

```
average positions open at any moment          0.25
total position-days                            764  over a 3,109-day span
secured at entry as a share of the cap         median 0%, p90 54%
funded trades per year                          75
```

Concurrency binds only in earnings clusters. Between them the account sits
empty. That reframes the levers: the gate admits the top 20% by construction
(1,428 wanted from ~7,000 gateable) and EXP-131 chose 20% for *volume
stability*, explicitly not for returns — with an account idle three quarters of
the time that choice now costs capital efficiency it did not cost before.

A post-hoc funding sweep (`results/funding_sweep_*.csv`, not a registered arm)
found raising the cap to 66% and doubling the account are both clearly positive
— together 10.73% CAGR against the registered 7.35% — while sizing at a share
of headroom to "leave room for concurrency" is strongly negative (3.17% alone),
because it trades away size you certainly have for concurrency that mostly
never arrives. Registered properly as **EXP-136**.

## 5. Three reasons not to promote this

**The book is 78% centre-peaked.** The winner is not a better twin peak. It is
mostly condors and butterflies, which want a *quiet* print — the opposite
thesis. The original hypothesis is supported (1,130 three-strike and 1,599
four-strike trades chosen, so fewer strikes do widen the universe) but what the
saved strikes buy is a different strategy. `best_twin_only`, restricted to the
two twin-peaked families, still returns +25.4% on capital against the
incumbent's +12.2% — so **per-event width and anchor choice is worth roughly
double on its own**, separately from the shape switch, and that is the finding
most worth pursuing.

**The chooser still reads model error.** Spearman ρ = +0.096, p = 0.0122
between an event's candidate count and its simulated-minus-realized return.
Mean simulated +52.0% against realized +31.0% — a 21pp optimism gap. Smaller
than EXP-133's, not gone. The objective is still a maximum over ~900 candidates
per traded event.

**In dollars it is modest.** $100k → $182,842 over nine years is ~6.9%/yr,
because the account is unemployed most of the time. Note `random_pick` ends at
$138,106 despite worse per-trade numbers, purely by funding more trades: final
equity is confounded by trade count, which is why the registered criteria use
return on capital.

## 6. What was verified

- **Pricer:** 5,250 comparisons across 136 events and every year against
  `engine.structures.price_structure`, max difference **5e-14**. Expected P&L
  against `engine.pnl_sim.expected_pnl` to 2.6e-14.
- **Defined risk:** zero failures under the registered exit, at every alpha.
- **Alpha grid:** one sample throughout, inheriting EXP-133's `zero_cost` fix.
- **Entry no-arbitrage:** 0.00% violations on every arm's traded book.

## 7. Open

- **EXP-136** — the funding policy, registered rather than swept post-hoc.
- **The gate's 20%** is now a capital-efficiency question, not only a volume
  one, and changing it after seeing §4 would be post-hoc selection. It needs
  its own registration.
- **Exit-side quote quality for the existing board.** TWIN-P5's entries are
  clean at 1.0% but 30.8% of its exits violate no-arbitrage. Every published
  twin-peak result is marked out through those quotes on roughly a third of its
  trades. EXP-134 stops assuming it away going forward; it does not fix the
  past.
