# Earnings-Vol Trading Program — Engine

Implementation of `EARNINGS_VOL_PROGRAM_PLAN.md`. This repository holds the
**code**: the shared engine, the three-tier data architecture, and the checks
that decide whether a phase is done. Market data, research findings, reports and
the prediction ledger deliberately live elsewhere — see `RECOVERY.md`.

**Status: Phases 0, 1 and 4 complete. Phase 2's harness is complete and verified
(the EXP-050 regression reproduces through it); Phase 2's exit criteria are
not met until the backlog experiments (EXP-101+) have run through the harness
— backlog #1 waits on the Sep-1 ORATS pulls. Phase 4 (verification & reporting)
is built: one legible report format, the append-only prediction ledger, and
audit receipts — see `reports/phase4_reporting.md`. The ledger holds no forward
rows yet because the calendar ends 2026-08-27; it starts accruing when the Sep-1
pull refreshes the forward calendar.** Phases 3, 5 and 6 are specified in
`guides/` — Phase 3 (dashboard) is next, and reads `ledger/health.json`.

Build order is **0 → 1 → 2 → 4 → 3 → 5 → 6** (the plan's sequencing table):
the phases are numbered in the order they were specified, not built, and Phase
3's nightly job is what writes the Phase 4 ledger.

---

## Quick start

```bash
pip install --break-system-packages pyarrow pytest coverage   # non-system deps
cp .env.example .env                                  # fill from the password manager

python3 -m engine.data.rebuild                        # Tier 1 → Tier 2 → Tier 3
python3 checks/phase0_checks.py                       # data-foundation checks

python3 -m engine.build_trades                        # replay the structures (~35 min)
python3 -m engine.models.training.train_all           # train + register champions
python3 checks/phase1_checks.py                       # scoring-engine checks

python3 experiments/new_experiment.py --title ... --hypothesis ...   # scaffold EXP-101+
python3 experiments/EXP-NNN_slug/run.py               # evaluate through the harness
python3 experiments/promote.py EXP-NNN --champion-metrics ...        # promote or refuse
python3 checks/phase2_checks.py                       # experiment-framework checks
```

```python
from engine.score import score, score_calendar
score("AAPL", "STR-THRU", as_of="2026-01-28")   # one event
score_calendar(horizon_days=21)                  # the upcoming board
```

---

## Architecture

```
engine/
  paths.py            logical name → physical path registry
  fills.py            FillModel: the worst→best execution interpolation
  calendar.py         earnings calendar + session-aware trading-day math
  structures.py       trade structures → leg lists → the one pricing path
  audit.py            leak discipline, asserted on every scoring path
  features.py         as-of feature vectors (panel path and live path)
  replay.py           structures × real chains → priced trades
  payoff.py           predicted quantity → exit value, empirically calibrated
  analogs.py          matched historical trades + bootstrap intervals
  score.py            the Phase 1 scoring API
  calibrate.py        reliability curves, Brier scores, decile tables
  models/             registry + champions (artifacts live in data/models/)
  evaluate.py         Phase 2: backtest → walk-forward → MC → stress → metrics
  report.py           Phase 4: the one report generator every phase emits through
  ledger.py           Phase 4: append-only prediction ledger + outcome scorer
  data/
    fetch.py          Tier-1 wrapper: cache-first, throttled, quota-guarded
    throttle.py       per-source pacing, backoff, quota floor, Polygon lock
    sources/          one adapter per API (orats, polygon, oquants, yfinance)
    schemas.py        Tier-2 column specs + assert_schema
    store.py          partitioned Parquet store (csv.gz fallback)
    normalize/        Tier-1 → Tier-2 normalizers; every unit trap fixed here
    validate.py       ingestion battery + quarantine
    features/         Tier-3 causal panel
    coverage.py       coverage analysis + the data audit
    rebuild.py        the orchestrator
    manifest.py       generated MANIFEST.md + snapshot hash
    pulls/            quota-spending plans (--dry-run, then --confirm)
experiments/          EXP-101+ scaffolding, append-only LEDGER.csv, promote.py
checks/               acceptance tests, migration test, repo hygiene
tests/                unit suite (pytest)
```

`engine/SCORING.md` is the Phase 1 map: the two estimation layers, the leak
boundary that separates them, and the three places the Phase 1 guide's
assumptions did not survive contact with Phase 0's output.

### The three tiers

| Tier | Location | Rule |
|---|---|---|
| **1 — raw** | `data/raw/` | Append-only. Every byte ever fetched, kept verbatim forever, before any parsing. Quotas make refetching expensive; a parser bug must never cost a re-pull. |
| **2 — curated** | `data/curated/` | One normalized cross-source schema, Parquet partitioned by year. Rebuildable from Tier 1 **with no network access**. |
| **3 — features** | `data/features/` | The causal panel and derived matrices. Rebuildable from Tier 2, versioned by a snapshot hash that every report pins. |

The grandfathered research trees (`earnings_predictions/`, `bt/`,
`polygon_cache/`) are registered in `engine/paths.py` and are **read-only**;
`paths.assert_writable()` makes writing to them raise.

---

## The rules this code enforces

**Execution quality is the program.** Worst-case fills turn every earnings-vol
exposure negative and mid fills turn them positive, so no function computes P&L
without an explicit `FillModel`. Results are reported at worst / mid / best plus
the breakeven alpha.

**Real prices only.** ORATS chain bid/ask (validated to ±2–3% against Polygon
real trades) is the sole P&L source. oquants model-fitted marks are refused at
the adapter, not merely discouraged.

**Leak discipline.** Every panel feature reads the last observation *strictly
before* the print. BMO and AMC resolve to different dates, and
`engine/calendar.py` owns that arithmetic so no consumer has to remember it.

**Nothing is silently dropped.** Rows that fail validation are excluded and
counted; files that fail structurally are flagged in `data/raw/quarantine/`. The
raw bytes are never moved or deleted.

**Determinism.** Same inputs → identical bytes. Asserted by the `determinism`
acceptance check.

---

## Verification

| Command | What it proves |
|---|---|
| `python3 -m pytest tests -q` | Unit behaviour of every pure component |
| `python3 checks/phase0_checks.py --only test_policy` | Every module is covered by unit tests or a *declared* acceptance check |
| `python3 checks/phase0_checks.py` | All 14 Phase-0 acceptance checks |
| `python3 checks/phase0_migration.py` | The rebuilt panel reproduces the legacy master panel |
| `python3 checks/phase0_audit.py` | Coverage + price-sanity battery → `reports/phase0_data_audit.md` |
| `python3 checks/phase1_checks.py` | All 14 Phase-1 acceptance checks |
| `python3 checks/phase1_replay.py` | The scorer reproduces the replayed trades' pricing to 1e-6, and the live feature path reproduces the panel path to 1e-9 |
| `python3 checks/phase1_calibration.py` | Predicted win rates vs realized, out of sample |
| `python3 checks/phase2_checks.py` | The Phase-2 suite, incl. the load-bearing EXP-050 harness regression |
| `python3 checks/phase4_checks.py` | All 17 Phase-4 checks: golden report, regeneration, ledger append-only, leak poison, and the format guards |
| `python3 checks/phase4_report.py --checks-json …` | Phase 4's own evidence report |
| `python3 -m engine.ledger snapshot --as-of YYYY-MM-DD` | Freeze today's scored board into the prediction ledger |
| `python3 -m engine.ledger score --through YYYY-MM-DD` | Resolve predictions whose events have passed; recompute calibration at 50 new rows |
| `python3 -m engine.ledger status` | What the ledger holds |
| `python3 checks/repo_hygiene.py --all` | No secret, data file, or oversize blob is tracked |

Testing strategy — the two layers, what each is for, and the known thin spots —
is written up in `tests/README.md`, and enforced by the `test_policy` check
rather than left as an aspiration.

The migration test is the load-bearing one. Every verdict in the plan rests on
`events_with_orats_sum.csv`; the test reconciles all 115,500 rows and fails on
any difference not covered by a **declared, independently verified** delta.

Phase 4 adds the format guards: `numbers_preserved` re-renders every committed
experiment report from its saved artifacts and asserts the new document holds
the old one's numbers, so a presentation change can never quietly move a value.
It found one number that WAS wrong: the capacity line printed relative spread
100× too small (`+0.3649%` for a 36.5% spread), because the old formatter
appended `%` without scaling.

The Phase-2 equivalent is `harness_regression`: the EXP-050 equity curve (GBM
top-20% gate, walk-forward, 531 trades) must reproduce through
`engine.evaluate` before any new experiment runs. It surfaced one stale number
in the original report — the 5%-sizing row — root-caused in
`reports/phase2_exp050_regression.md`.

---

## Data findings worth knowing

Four things the pipeline turned up that change how the existing data should be
read. All are documented in full in `reports/phase0_data_audit.md` and
`reports/phase0_migration.md`.

1. **ORATS `mktCap` has three unit eras, not two.** Billions before
   2017-06-28, millions to 2026-03-10, thousands after. The legacy panel knew
   only about the second boundary, so its `or_mcap_log` is understated by
   `log(1000)` on every event before 2017-06-28 — roughly half the sample, and
   a step discontinuity in a champion-model feature. Corrected in Tier 2.

2. **The legacy panel carries ORATS FLT_MAX sentinels as feature values.**
   Missing data is encoded as ~3.4e38 and reached the panel scaled by the unit
   multipliers (up to −3.4e40). Masked at normalization.

3. **`spy_vol20` is simple returns with `ddof=1`, not log returns.** The live
   predictor computed it as log returns with `ddof=0` — a systematic ~2.6%
   difference in one of the nine model features between training and live
   scoring. The panel definition is now pinned and reproduced exactly.

4. **Crossed quotes carry real weight in the published numbers.** 0.076% of
   chain rows quote a bid above the ask — a stale bid on a leg that just became
   worthless, concentrated in the biggest movers. Handling matters more than the
   frequency suggests: excluding those rows drops whole trades and pulls the S2
   base exposure from +3.5% to +1.8%, while pricing them at a crossed mid pushes
   it to +3.7%. Tier 2 repairs them to `min(bid, ask)` and flags them
   (`quote_repaired`), giving +3.48%.

---

## Operating constraints

- **ORATS**: 20,000 calls/month, resets on the 1st. The quota guard refuses to
  spend below a 3,000-call live-operations reserve. Any plan spending >500 calls
  requires a `--dry-run` first and an explicit `--confirm`.
- **Polygon**: ~10 req/min on price endpoints; one process at a time (enforced
  by a lockfile). The adapter shells out to `curl` because fresh-process
  `urllib` 401s with a valid key.
- **WSL2**: the host sleeps. Long jobs must be resumable and must log at least
  once a minute; never assume a background job survived — check its output.

`AGENTS.md` holds the full operational playbook and is binding.
