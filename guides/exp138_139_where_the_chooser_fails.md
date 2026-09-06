# EXP-138 / EXP-139 — three suspects eliminated, and the chooser still can't rank

*2026-09-06. Both exploratory-to-confirmatory, both largely negative, nothing
promoted. Artifacts under `experiments/EXP-138_a_skewed_move_weibull_scaled_simulation/`
and `experiments/EXP-139_why_the_simulator_is_optimistic_market_c/`.*

## 1. The question these two answer

EXP-134's chooser picks the structure with the highest simulated expected P&L.
EXP-137 measured how often that pick is right: **24.7%** against a 12.5%
chance baseline, per-event rank correlation +0.137, and only **17% of the
spread** between a random family and a perfect one.

So the ranking has real but weak skill. These two experiments ask why, by
eliminating candidates.

## 2. Not the move distribution (EXP-138)

`engine.pnl_sim` draws the move additively and clips: `max(pred + err, 0)`.
Measured on the 87,121-event pool that is right in location and wrong in shape
— **2.74% of draws at exactly zero against 0.56% realized**, skew +3.43 against
+5.00. And the error has a direction: a spike at zero move sits at the CENTRE
of the payoff, where every centre-peaked family pays most and both twin-peaked
families dip.

Replacing it with a scale-family draw — `pred x Weibull(c=1.1666,
scale=1.0978)` — removes the atom and restores the skew. The construction is
sound: the ratio realized/predicted is scale-stable across a 5.5x range of
predicted move, Weibull fits it to within **1.3pp of probability mass** in every
band these structures pay over, and the pairing with the crush residual is
preserved by drawing the same pool row and reshaping only the marginal.

| move model | top-1 | centre share | chooser return |
|---|---:|---:|---:|
| additive (incumbent) | 24.7% | 63.2% | +16.24% |
| **weibull** (primary) | **24.5%** | 65.2% | +15.96% |
| ratio (empirical) | 24.8% | 64.9% | +16.20% |
| gamma | 24.6% | 66.0% | +15.73% |

**Falsified.** All four are inside noise of each other. The centre share moved
the *wrong way*. The clip is real and immaterial.

Two things fell out. Lognormal was rejected on evidence, not taste — fitted in
log space where near-zero moves have huge leverage, it gives a p99 of 8.88
against a realized 3.99, strips 9pp from the 1-2x band and quadruples the
beyond-the-wings tail. And the empirical `ratio` arm — pure resampling, no fit
— matched or beat both fitted curves, exactly as the spec registered it might.

## 3. Not Black-Scholes (a stage-0 check)

Fed the **realized** move and **realized** crush, the pricer reproduces the
actually quoted exit to a median 1-2 cents — **0.4% to 1.2% of the debit**
across all eight families, spread under 0.8pp. Against a 6-12pp
miscalibration that is a tenth of the problem, and it is not differentially
biased by family: the two twin-peaked families sit mid-pack at 1.1% and 1.2%.

## 4. Not the calibration either (EXP-139)

The simulator over-predicts **every** family, by +5.9pp to +12.3pp. A uniform
positive bias is what a too-WIDE outcome distribution produces — every family
here has a bounded payoff with a zero floor, so fattening the distribution
lifts them all, most where the payoff is widest relative to the debit.

And there was a mechanism: the residual pool is **87,121 events of which only
27.5% clear the $10B floor the strategy trades**. Market cap carries shape
information beyond the prediction level — KS against the smallest bucket rises
monotonically to 0.074 (p = 1.9e-8), and the >$200B bucket puts **33.9%** of
its mass in the 1-2x band against **26.9%** for <$2B, a 7pp swing into exactly
where the twin peaks pay most.

| arm | top-1 | mean gap | **gap spread** | twin share | chooser return | **spread captured** |
|---|---:|---:|---:|---:|---:|---:|
| uncorrected | 24.7% | +9.7pp | 6.5pp | 36.8% | +16.24% | **17.0%** |
| **cap** (primary) | **25.2%** | +8.3pp | **4.0pp** | 40.8% | +15.94% | 15.9% |
| debias | 24.3% | **−0.4pp** | **0.5pp** | 47.4% | +15.89% | 16.3% |
| both | 24.5% | −0.4pp | 0.4pp | 46.4% | +15.87% | 15.8% |

The last column is the payoff measure that matters: the share of the
random-to-oracle return spread the chooser actually captures,
`(chooser − random) / (oracle − random)`, which normalises out the fact that
cap conditioning also moves the oracle and random baselines. **Doing nothing
captures the most.** Every adjustment lands below the uncorrected 17.0%,
including the primary. So the top-1 improvement to 25.2% does not survive
translation into money: the extra hits are on events where being right is
worth less.

Cap conditioning was applied to **94.0%** of draws, so this is not a treatment
that failed to reach its universe.

**Two of four criteria pass.** It does improve the mechanism — mean optimism
+9.7 → +8.3pp, and the spread *between* families, which is what an argmax
actually ranks on, nearly halves from 6.5pp to 4.0pp. Top-1 rises to 25.2%,
best of the four, and the twin share moves 36.8% → 40.8%. But `less_optimistic`
required every family to improve and TWIN-P went the wrong way (+5.9 → +6.1pp)
— it was already the best-calibrated family and size-matching had nothing to
give it — and chooser return slips 16.24% → 15.94%.

**The de-bias arm is the sharper result.** Subtracting a causal per-family
correction annihilates the optimism (+9.7 → −0.4pp) and collapses the spread to
0.5pp — and the hit rate gets **worse**, 24.3%. Near-perfect calibration of the
expectation buys nothing. It also overshoots the twin share to 47.4% against
the 36.7% that actually wins.

## 5. What is left

Three suspects eliminated on evidence, and the hit rate never moves out of
24-25%:

| suspect | verdict | evidence |
|---|---|---|
| the move distribution's shape | **not it** | four draws, 0.3pp apart |
| Black-Scholes repricing | **not it** | 1-2 cents on realized inputs |
| the level and family-spread of the expectation | **not it** | spread 6.5 → 0.5pp, hit rate falls |

What survives is that the *ranking* is limited by the point forecast and by
irreducible noise, not by anything built on top of it. EXP-137 already measured
**33.6% of events with the top two families within 5pp** — unrankable by any
model — and realized return is a single draw, so even a perfect expected-value
model could not approach 100%.

A correction that also belongs on the record: the premise that the chooser
skews centre-peaked is **false at the marginal**. It predicts twin-peaked on
36.8% of events and twin-peaked wins 36.7%. The 78% centre share in EXP-134 is
a property of its **gated** book, so the trailing top-20% gate contributes
roughly 15pp of centre skew on top of the chooser. That is a separate,
untested mechanism and the more promising place to look next.

## 6. What to keep

- **Cap-conditioning the residual pool** is right on correctness grounds, but
  that is the *only* ground it is right on: the pool genuinely is not the
  traded universe, the fix reaches 94% of draws, and it halves the family
  spread the argmax ranks on — and it still captures less of the
  random-to-oracle spread (15.9%) than leaving the pool alone (17.0%). So it
  is a correctness fix that costs money on this sample, not an improvement.
  It is a candidate for `engine.pnl_sim`, which would move every published
  gate number and so needs its own registration; that registration has to
  carry the 1.1pp capture cost as a known debit, not bury it.
- **The empirical `ratio` draw** removes the atom at zero at no cost and with
  no fitted parameter. Also a candidate for the engine.
- Neither is promoted here. On the one measure that pays — spread captured —
  the uncorrected chooser beats all three treatments, so nothing in EXP-139
  earns a change to the live path.

## 7. Disclosures

- **EXP-138's first build was killed by the OOM killer** after 33 minutes,
  having accumulated 1.45M rows with ~1-2KB `legs` blobs in memory before
  writing. Rebuilt with per-year parquet sharding; both experiments now shard.
- **EXP-139's runner initially produced no REPORT.md and no ledger row** —
  the same defect corrected in EXP-136 and written into AGENTS.md hours
  earlier. Fixed by putting the primary's best-of-family book through
  `engine.evaluate`.
- EXP-139 ran throttled (`nice -n 19`, `ionice -c 3`, BLAS pinned to one
  thread) alongside EXP-135's ORATS pull, which finished its 379 repair jobs
  independently.
