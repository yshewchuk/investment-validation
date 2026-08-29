# Earnings-Vol Trading Program — Engine

Implementation of `EARNINGS_VOL_PROGRAM_PLAN.md`. This repository holds the
**code**: the shared engine, the three-tier data architecture, and the checks
that decide whether a phase is done. Market data, research findings, reports and
the prediction ledger deliberately live elsewhere — see `RECOVERY.md`.

**Status: Phase 0 complete.** Phases 1–6 are specified in `guides/`.

---

## Quick start

```bash
pip install --break-system-packages pyarrow pytest    # the two non-system deps
cp .env.example .env                                  # fill from the password manager

python3 -m engine.data.rebuild                        # Tier 1 → Tier 2 → Tier 3
python3 checks/phase0_checks.py                       # all acceptance checks
```

---

## Architecture

```
engine/
  paths.py            logical name → physical path registry
  fills.py            FillModel: the worst→best execution interpolation
  calendar.py         earnings calendar + session-aware trading-day math
  structures.py       trade structures → leg lists → the one pricing path
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
checks/               acceptance tests, migration test, repo hygiene
tests/                unit suite (pytest)
```

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
| `python3 checks/phase0_checks.py` | The 11 Phase-0 acceptance tests |
| `python3 checks/phase0_migration.py` | The rebuilt panel reproduces the legacy master panel |
| `python3 checks/phase0_audit.py` | Coverage + price-sanity battery → `reports/phase0_data_audit.md` |
| `python3 checks/repo_hygiene.py --all` | No secret, data file, or oversize blob is tracked |

The migration test is the load-bearing one. Every verdict in the plan rests on
`events_with_orats_sum.csv`; the test reconciles all 115,500 rows and fails on
any difference not covered by a **declared, independently verified** delta.

---

## Data findings worth knowing

Three things the pipeline turned up that change how the existing data should be
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
