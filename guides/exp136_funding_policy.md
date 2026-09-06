# EXP-136 — funding the book

*2026-09-06. Exploratory. **The registered primary was falsified.** Nothing is
promoted. Artifacts:
`experiments/EXP-136_funding_the_book_account_size_the_secure/`.*

## 1. What was tested

EXP-134 established that a cash-secured account is the binding constraint on
this strategy: it refused 53% of the trades the strategy wanted, set 100% of
the position sizes, and averaged **0.25 open positions** — idle roughly three
quarters of the time. $100,000 became $182,842 over nine years, ~7.35%/yr.

Four changes to the funding policy were proposed together:

| | from | to |
|---|---|---|
| account | $100,000 | $200,000 |
| secured cap | 50% of equity | 66% |
| sizing | 5% of equity | aim at 1/4 of current headroom |
| fill order | chronological | most expensive first within a date |

The strategy itself is held **completely fixed** — same candidates, same
no-arbitrage filter, same objective, same gate, same exit convention, same
1,428 wanted trades. Only funding moves.

## 2. Result: the bundle is worse than two of its own parts

`best_all | conditional`, identical book in every row:

| policy | funded | median contracts | avg open | final | **CAGR** |
|---|---:|---:|---:|---:|---:|
| registered EXP-134 ($100k, 50%, 5%, chrono) | 675 | 2 | 0.25 | $182,842 | 7.35% |
| + account $200k only | 797 | 3 | 0.29 | $390,529 | 8.18% |
| + cap 66% only | 727 | 3 | 0.26 | $229,057 | 10.23% |
| + size at 1/4 headroom only | 773 | **1** | 0.28 | $130,467 | **3.17%** |
| + most expensive first only | 710 | 1 | 0.26 | $179,039 | 7.08% |
| **$200k + 66%** | 843 | **4** | 0.31 | $476,214 | **10.73%** |
| $200k + 66% + expensive first | 939 | 2 | 0.34 | $413,116 | 8.90% |
| $200k + 66% + expensive first + 1/2 headroom | 1,151 | 1 | 0.41 | $330,632 | 6.08% |
| **PRIMARY — all four** | **1,124** | **1** | 0.40 | $284,742 | **4.24%** |

Acceptance: `funds_more` PASS, `no_new_defined_risk_failures` PASS,
**`beats_the_registered_policy` FAIL** — 4.24% against 7.35%.

## 3. Why: reserving headroom is backwards for an idle account

The intuition behind sizing to a quarter of the headroom is to leave room for
concurrency. **The concurrency being reserved for mostly does not arrive.** The
account averages a quarter of one open position and its median secured-at-entry
is 0% of the cap — usually nothing else is open at all. Under-sizing therefore
trades away size that is *certain* for trades that are *hypothetical*: the rule
funds far more events (78.7% against 47.3%) at **one contract instead of four**,
and the extra count does not pay for the lost size.

Sweeping the aim to 1/2 confirms the direction rather than rescuing it: 6.08%,
still below the registered 7.35%. The change is wrong in kind, not in
magnitude.

Meanwhile the two changes that simply give the account *more room* both work,
and they compose: cap 66% alone is +2.9pp, $200k alone is +0.8pp, together
+3.4pp. Raising the cap is the larger of the two because it applies to every
trade, where a bigger account mainly rescues the expensive ones.

**Fill order is close to neutral** on CAGR (7.08% alone, and it costs 1.8pp when
added to $200k + 66%) though it does improve the funded *mix* — return on
capital is highest of any cell at 33.3%. It fills expensive names first, which
then consume the headroom and shrink everything after them.

## 4. It is not an artifact of one arm or one exit

CAGR %, conditional exit:

| policy | best_all | best_twin_only | incumbent |
|---|---:|---:|---:|
| registered EXP-134 | 7.35 | 3.17 | 1.41 |
| cap 66% only | 10.23 | 5.31 | 2.14 |
| **$200k + 66%** | **10.73** | **5.33** | 2.46 |
| quarter headroom only | 3.17 | 1.91 | 1.01 |
| PRIMARY all four | 4.24 | 1.99 | 1.07 |

The ordering is the same in all three. Across the three exit conventions the
mean CAGR is 6.17% for `$200k + 66%`, 3.98% registered, 2.43% for the primary —
same ordering again. The best cell is `$200k + 66%` for two arms and
`$200k + 66% + expensive first` for the incumbent.

## 5. What this is not

**A bigger account is not a better strategy.** Doubling the account raises CAGR
by funding trades that were previously refused — 369 refusals that could not
fund one contract against an empty account fall to 49. That is a *capacity*
result. The per-trade edge is untouched: each trade's return on its own debit
is fixed by EXP-134's book and no funding policy can move it.

**These cells are not independent samples.** Every one reuses the same 1,428
wanted trades, so the differences carry no error bar from this design. They are
reported as a deterministic comparison and never with a p-value.

**And this was not a clean test.** A post-hoc sweep of these same policies was
run on this same book before the spec was written, at the user's request, and
its numbers are in `known_before_registration`. The experiment was registered
anyway — to put the comparison in the ledger, to run it across arms and exits
rather than one cell, and because the policy had been asked for. Registering a
hypothesis you expect to lose is the point of registering it, but no result
here is independent confirmation of anything.

## 6. Recommendation

Adopt **$200,000 at a 66% secured cap, sized as EXP-134 sized it, filled
chronologically** — 10.73% against 7.35%, best or near-best on every arm and
every exit convention. Drop the headroom-share sizing rule; it is the single
most damaging change tested. Fill order is a coin flip: neutral on CAGR, better
on capital efficiency, worse on size.

The larger lever is still untouched and still needs its own registration: the
gate admits the top 20% by construction, EXP-131 chose that for volume
stability rather than returns, and with an account idle three quarters of the
time that choice now costs capital efficiency it did not cost before.
