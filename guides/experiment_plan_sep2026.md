# Experiment Execution Plan — Sep 2026

**Version:** 1.1 · **Date:** 2026-08-30
**v1.1:** corrected the sequencing — C and D were placed after the Sep-1 pull as
"work for while it runs". They depend on nothing in it, and each controls a line
item of an oversubscribed budget, so all three experiments now precede the pull. · **Owner:** YS + Claude
**Covers:** the fetch-store bridge verification, EXP-102 (CAL-P risk mechanics),
EXP-105 (STR-THRU validation), EXP-107 (STR-RUNUP validation).
**Anchor:** the ORATS quota resets **Sep 1**. Two of these four run before it and
one of them should change how the quota is spent.

---

## 0. What changed since the plan was written

Two findings reorder the backlog.

**CAL-P is not data-blocked.** `EARNINGS_VOL_PROGRAM_PLAN.md` records CAL-P at
n=359 and schedules the exact-spec backtest as "post-Sep-1 pull". But
`engine.build_trades` has already priced **4,736 CAL-P events across 2018–2026**
on real chains, and the structure that produced them is the exact spec:

```
front_put  SELL  first_post_event, max_dte=7   ATM
back_put   BUY   first_dte_at_least dte>=20    same strike
entry_offset 0 (last pre-print close)   exit_offset +1 (first post-print close)
```

Put legs, both opened together shortly before the print, held through it, closed
together after. Backlog items #1 and #2 are runnable today at zero quota cost.

**The margin of safety on the mid-fill assumption is thinner than the program
assumes.** Measured on the engine's own trade set, at the alpha grid:

| Strategy | n | mid mean/trade | win | **breakeven alpha** | margin vs mid | negative years |
|---|---:|---:|---:|---:|---:|---|
| STR-THRU  | 17,666 | +2.70% | 38.8% | **0.448** | 5.2 pp | 2020, 2021 |
| STR-RUNUP |  5,311 | +1.02% | 38.0% | **0.478** | 2.2 pp | 2020, 2021, 2023, 2025 |
| CAL-P     |  4,736 | +2.76% | 53.0% | **0.475** | 2.5 pp | 2023, 2025, 2026 |

Breakeven alpha is the fraction of the bid-ask spread you must capture to make
zero. Mid is 0.50. **STR-RUNUP stops making money if you capture 47.8% of the
spread instead of 50%.** That is a 2.2-percentage-point margin on the single
assumption the plan names as its #1 risk. Every experiment below reports this
number as a headline, and Phase 5's α̂ measurement is what will settle it.

Note also that STR-RUNUP's +1.02%/trade on the engine trade set is well below
the +3.9%/trade the plan cites from EXP-048, and its registered gate's own eval
records a **negative** base mean return (−1.24%) over its 2020–2026 OOS window.
Reconciling those three numbers is a pre-registered output of EXP-107, not a
footnote.

---

## 1. Sequence and dependencies

```
  now ──► A1. fetch bridge (offline)          no quota
       ──► §6 shared pieces                   no quota   ── the real critical path
                    │
                    ├──► B. EXP-102  CAL-P risk mechanics   no new data
                    ├──► C. EXP-105  STR-THRU validation    no new data
                    └──► D. EXP-107  STR-RUNUP validation   no new data
                                    │
                                    ▼   all three inform the allocation
  when ready ──► A2. one live call + rebuild ──► release the 16,000-call pull
                                                        │  (runs for days)
                                                        ▼
  after ─► EXP-101 CAL-P exact-spec on the enlarged sample
        ─► EXP-103 STR-RUNUP entry-day sweep (needs T−j coverage; 11.6% today)
        ─► EXP-104 moneyness edge-decay (runnable now; unblocks EXTRAPOLATED)
```

**All three experiments run before the pull, and none of them waits on it.**
B, C and D read only trades that are already priced — 4,736 / 17,666 / 5,311
events on real chains — against gates that are already trained and registered.
Nothing in them needs a byte of new data.

**Because the budget is oversubscribed, each one controls a line item.** The
pull plan maps purposes onto dates (`{"exit": exit_date, "entry": entry_date,
"t14": runup_date}`), so:

| purpose | calls | share | whose sample it enlarges | informed by |
|---|---:|---:|---|---|
| `exit`  | 9,349 | 58% | STR-THRU + CAL-P exits | B, C |
| `t14`   | 3,535 | 22% | **STR-RUNUP only** | D |
| `entry` | 3,116 | 19% | all three | B, C, D |

The plan prints *"truncated at the budget; re-run next cycle for the rest"* —
what is skipped this cycle waits a month. So allocation is a real decision, and
spending 22% of it enlarging STR-RUNUP — the weakest of the three at +1.02%/trade,
a 2.2 pp margin, four negative years of nine, and a base return three sources
disagree on by 5 percentage points — before validating it is the wrong order.

**Sep 1 is not a deadline.** The quota resets on the 1st and runs to the 30th;
spending on Sep 4 costs nothing but calendar, and Q3 season is six weeks out. The
plan anchors on Sep 1 because that is when quota becomes *available*, not because
it must be spent that day. A few days is cheap insurance against mis-allocating
16,000 calls that then wait a month.

**The binding constraint is engineering time on the §6 shared pieces**, not data
and not quota. If time forces a subset, run them in budget order: C (58%),
D (22%), B — noting that B is also the cheapest, being descriptive.

**Numbering.** EXP-102 and EXP-105 keep the meanings `guides/phase2` reserved for
them. EXP-103 (entry-day sweep), EXP-104 (moneyness) and EXP-106 (1–10B slice)
stay reserved and unrun, so the STR-RUNUP validation takes **EXP-107** rather
than squatting on a reserved id.

---

## 2. Workstream A — the fetch-store bridge

**The risk.** `data/raw/fetch/` is empty: no body written by
`engine.data.fetch.Fetcher` has ever been read by a normalizer. If the live
ORATS envelope differs from the assumed `{"data": [...]}`,
`fetch_store.orats_rows` is **tolerant by design** and returns an empty list.
The failure mode is therefore: 16,000 calls succeed, raw lands in Tier 1,
`rebuild --table chains` produces zero rows, and nothing raises. You would find
out by noticing coverage did not move.

### A1 — offline proof (today, no quota)

1. Reconstruct a faithful raw body: legacy files decode to
   `{entry_date, tickers, rows}` where `rows` carry ORATS' own field names
   (`ticker`, `tradeDate`, `expirDate`, `strike`, `stockPrice`, `callBidPrice`,
   …), so `{"data": rows}` is byte-faithful to what the API returns.
2. Drive the **real** `Fetcher` with a stub adapter returning that body — the
   pattern `tests/test_fetch.py` already uses — so the genuine persist path runs:
   cache key, sha naming, gzip, meta sidecar, fetch log. No network.
3. Run `python3 -m engine.data.rebuild --table chains` and assert the rows reach
   Tier 2 with correct provenance.
4. Land it as an acceptance check (`checks/phase0_checks.py`), not a one-off
   script — the bridge is a standing contract, not a one-time question.

**Exit:** a fetch-store body demonstrably becomes Tier-2 chain rows.

### A2 — live probe (Sep 1, one call)

Offline proves the plumbing; it cannot prove the live envelope. Before releasing
the 16,000-call plan, spend **one** call: a single `/hist/strikes` request
through the wrapper, then `rebuild --table chains`, then confirm rows landed.
One call out of 16,000 to de-risk the other 15,999.

### A3 — harden the tolerance (with A1)

`orats_rows` returning `[]` on an unrecognized payload is a reasonable shrug
when data is free and wrong when a byte costs quota and the failure is silent.
It should be counted and flagged the way `validate.py` quarantines a bad file.

---

## 3. Workstream B — EXP-102, CAL-P risk mechanics

**Type:** descriptive measurement. No promotion target, no champion comparison.

**Hypothesis (pre-registered):** the CAL-P structure is defined-risk — realized
loss never exceeds the net debit — and early-assignment exposure on the short
front put is low enough to trade.

**A two-query probe already suggests both halves are false.** Reported here so
the experiment is pre-registered against a stated expectation rather than
discovering one:

```
losses exceeding the net debit : 80 of 4,736 (1.7%)
worst realized trade           : -3.87x the debit
  NFE 2024-08-09  paid 0.375, closing COST 1.075  (exit_value negative)
short put ITM at post-print close : 49.8%
  ITM by >5%  19.5%    ITM by >10%  8.3%
front-leg DTE at entry            : mostly 2-4, not 1
```

At **mid** fills, not worst-case. A negative exit value means buying back the
front put costs more than selling the back put brings in.

### Required outputs

1. **Max-loss distribution** vs net debit, per year and per mcap bucket. Every
   trade with `ret < -1.0` inspected individually and classified: real loss,
   crossed-quote artifact, or stale exit chain. The count of each is the result.
2. **Assignment exposure**: frequency and depth of the short put trading ITM at
   the post-print close, and separately at the front expiry (pin risk).
3. **The `zero_cost` selection.** `replay_one` drops an event when the structure
   prices at a credit at *any* alpha. Because the net debit falls as alpha rises,
   the surviving universe is "calendars still a debit at the best fill" — a
   systematic exclusion of the cheapest calendars. Recover the count from
   `build_trades --strategy CAL-P --dry-run` and report it beside the headline.
   The 4,736 is conditioned on it.
4. **Tail injection** — mandatory (`has_short_leg: true`), and the checklist FAILs
   without it. See §6.
5. Breakeven alpha, per-year table, regime replays, deployment.

### What would falsify the defined-risk claim

It is already falsified unless the 80 exceedances turn out to be data artifacts.
The decision-relevant question is therefore **how much** worse than the debit the
tail is, and whether the ~50% ITM rate is survivable given the plan's own
mitigation ("close both legs promptly post-print").

### Consequence

Feeds directly into the Sep-1 spend: if the tail is real and unbounded, the
put-side chain budget moves to exit-chain coverage.

---

## 4. Workstream C — EXP-105, STR-THRU validation

**Runs now. Needs no new data.** Controls the 9,349-call `exit` line (58%).

**Type:** confirmatory. This re-validates a known result on a rebuilt pipeline;
it is not discovery, and the spec says so. Its value is (a) the harness's first
use on a strategy with real evidence, (b) numbers on the engine's own trade set
rather than the differently-specified legacy one, (c) breakeven alpha and
deployment, which have never been computed for it.

**Hypothesis (pre-registered):** the registered champion gate
`gate_midfill_str_thru` at its stored threshold delivers, on the full
engine-replayed universe under independent walk-forward, an OOS mean return per
trade materially above the ungated base — of the order the registry's own
training evaluation records (base +2.92%, gated +7.34%, lift +4.42pp).

**Primary spec:** gate = registered champion at its stored threshold; walk-forward
expanding by calendar year, `min_train_years: 2`; mid fills headline with the
full alpha grid; universe = all 17,666 engine-replayed STR-THRU events 2018–2026;
`equity_mode: cashflow`; `max_deployed_fraction: 1.0`.

### Required outputs

1. OOS mean, Sharpe, win rate, profit factor — the canonical block — with the
   **unselected universe reported beside the gated set** (the anti-selection
   guard; already structural in `evaluate`).
2. **Breakeven alpha.** Baseline to beat: 0.448 ungated. Does the gate improve
   the margin of safety, or only the mean? A gate that lifts return while leaving
   breakeven at 0.45 has not reduced the program's #1 risk.
3. **Gate-lift reconciliation**: independent walk-forward lift vs the +4.42pp the
   registry records from training. A material shortfall means the gate is fitted
   to the legacy specification rather than to the exposure.
4. **Calibration block** — requires `predict_proba` (§6). Without it rule (f) is
   a permanent WARN and the calibration gate never applies.
5. Stress battery: 2018Q4, 2020-02..04, 2022, the 10 worst earnings weeks,
   IV-regime split. The plan asserts 2022/2024 carry the curve — quantify it.
6. **Deployment**: peak deployed / equity and worst cash, now that
   `build_equity` reports them. At 133 max concurrency, "5% per trade" is not 5%
   of anything.

### Falsification

The gate fails validation if its independent walk-forward OOS lift is not
materially positive, or if it lifts the mean without improving breakeven alpha.

---

## 5. Workstream D — EXP-107, STR-RUNUP validation

**Runs now. Needs no new data.** Controls the 3,535-call `t14` line (22%) outright — `t14` resolves to `runup_date`, which no other strategy uses.

Same shape as EXP-105 — deliberately, so the two are comparable through
`METRIC_KEYS` — with one extra pre-registered obligation.

**Hypothesis (pre-registered):** the registered champion gate
`gate_midfill_str_runup` at its stored threshold delivers positive OOS mean
return on the full engine-replayed universe, and the exposure it selects from is
materially better than the ungated base.

**Primary spec:** as EXP-105, on 5,311 STR-RUNUP events 2018–2026, entry fixed at
`entry_offset = -14` (the current champion configuration; the entry-day sweep is
EXP-103 and is data-blocked at 11.6% T−14 coverage).

### The reconciliation, pre-registered as a required output

Three numbers describe the same strategy and disagree:

| Source | base mean / trade |
|---|---:|
| Plan, citing EXP-048 (legacy S3 trade set) | **+3.9%** |
| Engine replay, all 5,311 events, mid fills | **+1.02%** |
| Registry gate eval, 2020–2026 OOS window | **−1.24%** |

Explain the gap before interpreting anything else. Candidate causes, all
checkable: different universe (legacy S3 was scoped; engine is unselected),
different window (EXP-048 spans 2017–2025; the gate's eval starts 2020),
different structure (legacy S3 entered on a calendar T−14, the engine on a
trading-day T−14), and coverage bias — 5,311 events is ~13% of the calendar, and
a T−14 chain exists mostly for liquid names.

### Additional required outputs

- **Coverage bias.** Report the priced share against the full planned universe
  and compare the mcap distribution of priced vs unpriced events. If T−14 chains
  exist mainly for large caps, the strategy is validated only there.
- Breakeven alpha. Baseline 0.478 ungated — a **2.2 pp** margin. If the gate does
  not widen this, STR-RUNUP is not a candidate for capital regardless of its mean.
- Four negative years of nine (2020, 2021, 2023, 2025) — per-year table and the
  MC sizing curve carry the lumpiness.

### Falsification

STR-RUNUP fails validation if the gated OOS mean is not positive, or if the
reconciliation shows the +3.9% figure came from a universe the engine's spec
cannot reproduce.

---

## 6. Shared implementation work

These are the critical path for C and D and the mandatory gate for B. None
exists today.

| Piece | Needed by | Why it is not trivial |
|---|---|---|
| **`Gate.predict_proba`** | C, D | The registered gates predict a *return*, not P(win). `calibration_block` needs a probability. Fit a monotone map (isotonic, as `engine.recalibrate` already does) from predicted return to realized win, **inside the walk-forward fold on train years only**. Without it, rule (f) WARNs — and WARN passes. |
| **`tail_shock`** | B (mandatory) | "Double the worst 1% of realized moves and re-price." CAL-P deliberately has no payoff map, so the shock must re-price both legs from chains at a synthetic post-print spot. This is real work and the checklist FAILs a short-leg spec without it. |
| **`repricer`** | B, C, D | Slippage (±1 day) and stale-date stress. Shift entry/exit by one trading day, re-run `replay_one` where the adjacent chain exists, return the frame with a `coverage` attr. Never fabricate a missing chain. |
| **`spy_daily`** | B, C, D | Regime replays and the IV-regime split. Available from `daily_market` / the cached index series. |

Build them once, in `experiments/lib.py` or a shared `experiments/common.py`, so
all three experiments use one implementation — the same reasoning that keeps
`engine.replay` the only pricing path.

---

## 7. Standing constraints these runs inherit

- **Pre-registration is enforced.** Scaffold with `new_experiment.py`; the primary
  spec's hash must match its PLANNED ledger row or `evaluate` refuses the OOS
  stage. To change a hypothesis after seeing results, scaffold a new experiment.
- **Headline numbers are walk-forward OOS only.**
- **Every result at worst / mid / best plus breakeven alpha.**
- **The unselected universe appears beside every gated statistic.**
- **An experiment without a REPORT.md does not exist.**
- **Confirmatory, not discovery.** ~50 prior experiments have touched this data
  and the data moat is fixed — no new history is coming. C and D are re-validations
  of known results on a rebuilt pipeline, and their specs must say so, so the
  ledger records them honestly.

---

## 8. What this plan does not claim

- It does not resolve the **sizing/leverage** question. `_compound` walks trades
  sequentially regardless of dates, so MC P(loss) cannot see simultaneous
  exposure — twenty consecutive −50% trades at 5% cost 40%, twenty *simultaneous*
  ones cost 50% in one step. The deployment metrics make the leverage visible;
  they do not make the MC date-aware. That decision belongs with the 6% → 15%
  question at the Phase 5 go/no-go memo.
- It does not improve any model. C and D validate what is registered.
- It does not unblock the dashboard. Zero forward events until the pull lands.

---

*Nothing here is financial advice. All figures are historical simulations at
documented fill assumptions; the breakeven alphas above are the reason Phase 5
exists.*
