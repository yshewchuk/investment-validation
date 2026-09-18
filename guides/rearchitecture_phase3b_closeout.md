# Phase 3B closeout: acceptance-evidence record

Written 2026-09-18 against branch `codex/phase4-remediation-track-a`. Modeled
on [`rearchitecture_phase2_closeout.md`](rearchitecture_phase2_closeout.md).
Companion to [the 3B implementation plan](rearchitecture_phase3_incremental_data.md)
and the verified gap list in
[`rearchitecture_phase3b_phase4_remaining.md`](rearchitecture_phase3b_phase4_remaining.md)
(items R3B-1 through R3B-8).

**This record closes five items — R3B-1, R3B-2, R3B-4, R3B-5, R3B-8 — the
acceptance-evidence and documentation slice. It does not close Phase 3B as a
whole.** R3B-3 (nightly integration unwired), R3B-6 (`STALE_EXPECTATION`
unregistered) and R3B-7 (a failing finality-coverage test) remain open and are
out of scope for this record; see "What remains" below. Per
`rearchitecture_phase3_incremental_data.md`, "Phase 3 as a whole closes only
when 3A and 3B have separate completed records" — this is 3B's record for the
work it covers, not a claim that 3B is fully wired into the nightly.

## What this record certifies

The Phase 3B acceptance gate (`checks/rearchitecture_phase3b.py`) passes over
the retained run `run-1789640509`
(`/tmp/phase3b-acceptance-final/run-1789640509`, read-only): **PASS, 0
findings, 8/8 subjects, 24/24 negative controls.** Reproduce with:

```bash
python3 checks/rearchitecture_phase3b.py \
    --evidence /tmp/phase3b-acceptance-final/run-1789640509/evidence.json \
    --artifact-root /tmp/phase3b-acceptance-final/run-1789640509 --json
```

The full evidence — per-table populations, manifests, no-op counters, changed
partitions, full-rebuild comparisons, negative controls, runtime/RSS/cache/
contention counters, actual failures and retained snapshot refs, as required
by the 3B plan's "final report" paragraph — is generated from that run by
`checks/phase3b_report.py` into `reports/phase3b_acceptance/report.md`
(gitignored locally, reaches the private mirror as `.md` per
`tools/private_mirror.py`'s `("reports", "**/*.md")` include). It is written
through `engine.report.Report`, Convention 6, like every other phase report.

## Known test-suite state at close

**The acceptance gate PASS above and one known-red test in the wider v2
suite are both true at the same time.** The gate only checks the 8
P3B01-P3B08 subjects; it says nothing about the rest of the suite. Verified
2026-09-18 in the main checkout (real `data/` present, so this is not a
worktree/missing-data artifact):

- `tests/test_v2_data_generic_incremental.py::test_generic_earnings_events_candidate_commits_atomically`,
  `tests/test_v2_data_incremental_persistence.py::test_frozen_curated_append_correction_tombstone_and_noop_replay`,
  and `tests/test_v2_data_incremental_tables.py::test_frozen_curated_tables_use_the_same_merge_rules`
  **pass** with real data present. (They fail with `FileNotFoundError` in a
  plain git worktree, because `data/` is gitignored and no worktree carries
  it — an environment artifact, not a defect.)
- `tests/test_v2_ops_nightly_completion.py::test_action_finality_writes_a_coverage_output_from_monkeypatched_frames`
  is **genuinely red**: `covered_tickers: []` where `['AAA']` is expected.
  This is R3B-7 (below), assigned separately and not fixed by this record.

So: Phase 3B's acceptance-evidence slice closes with the 8/8 gate PASS
**and** one known-red test in the wider suite, named above. Neither fact is
omitted for the other.

## Accepted limitation (supervisor decision, 2026-09-18)

**The gate result above is bounded-scope evidence, not production
acceptance.** `checks/phase3b_real.py:_rows(path, limit=64)` reads at most 64
rows from ONE curated partition per table:

| table | rows read by this run | rows in the real table |
|---|---:|---:|
| `daily_market` | 64 (of one `year=2007` partition, 14,684 rows) | ~9.1M |
| `option_chains` | 64 (of one `year=2017` partition, 15,786 rows) | ~28.7M |
| `securities` | 64 (of one `year=2017` partition, 1,784 rows) | ~223k |
| `trades` | 64 (of one `year=2018` partition, 89,371 rows) | ~527k |
| `option_daily`, `earnings_events`, `feature_panel`, `tier4_forecasts` | 64 or fewer, one partition/whole-file each | see run receipt |

The supervisor decided to **accept this bounded scope as a recorded
limitation** rather than require a real-scale, full-partition run before
closing this slice of 3B, and to **drop the `"production_acceptance": true`**
claim the prior hand-authored `reports/phase3b_acceptance/report.json`
carried (nothing in the repository ever wrote that file or that field — it
was hand-authored, and no receipt in this repository supports a claim of
full-table, production-scale acceptance). Raising the 64-row limit, or
attempting a real-scale run, was explicitly out of scope for this
remediation and was not attempted.

Any later reader who needs production-scale evidence must run a fresh
acceptance pass at real partition scale with a measured EOD-scale resource
profile (the original, un-taken option for R3B-1) and record a superseding
receipt — not edit this one's claims.

## R3B-2: whole-file rewrite on `feature_panel`/`tier4_forecasts`

**Finding: correct semantics, not a defect.** The run receipt's
`changed_partitions` for `feature_panel` and `tier4_forecasts` both read
`["__whole__"]` (199,884-row tables) on every correction, in contrast to the
six Tier-2 tables, which report a single `year=` partition. This is because
`checks/phase3b_real.py:_partition()` returns the literal `"__whole__"`
whenever `contract.partition_columns` is empty, and both tables are declared
that way by `engine/v2/data/legacy_mapping.py`
(`_build_feature_panel_contract`, `_build_tier4_contract`, both calling
`_finalize(..., partition_columns=(), ...)`).

That declaration is not an oversight — it is the Phase 2 decision recorded in
`rearchitecture_phase2_data_access.md` §3.3/§5.1: `feature_panel` and
`tier4_forecasts` are each declared **one logical partition**, because their
physical layout is still the legacy single-file artifact
(`engine.paths.PANEL`, `engine.paths.TIER4`), not a `year=`-partitioned
directory tree like the six Tier-2 tables. A correction to either file has no
finer physical partition to target, so a whole-file rewrite on every
correction is the only semantics the current physical layout supports — and
is therefore correct, not a bug to fix. No code change follows from this
finding. If a future phase re-partitions these two derived tables physically
(splitting `panel.parquet`/`tier4_forecasts.parquet` by year, say), this
decision should be revisited alongside that change, not before it.

## R3B-4: runbook

A Phase 3B append/correction/recovery section was added to
[`rearchitecture_phase3_runbook.md`](rearchitecture_phase3_runbook.md#7-phase-3b--incremental-appendcorrectionrecovery-2026-09-18)
with the commands actually executed for this record and their real output
shape. Commands not executed against a real-scale run are marked UNVERIFIED,
per that file's existing convention.

## R3B-5: report

`reports/phase3b_acceptance/report.json` (hand-authored, `production_acceptance:
true`, no code wrote it) is superseded by `reports/phase3b_acceptance/report.md`,
generated by `checks/phase3b_report.py` from the run receipt and the gate
result. The `.md` extension is deliberate: `tools/private_mirror.py`'s
`INCLUDE` only mirrors `("reports", "**/*.md")`, so a JSON-only report never
reached the private mirror. `tests/test_checks_phase3b_report.py` asserts the
generated file is Markdown, is matched by the mirror's `collect()`, and never
claims `production_acceptance`.

## What remains (not part of this record)

| ID | Item | Why it is not here |
|---|---|---|
| R3B-3 | `incremental_refresh` is registered but no nightly plan or CLI path submits it (`plans.py` `NIGHTLY_GRAPH` "refresh" still runs the legacy adapter) | Requires wiring a supervised nightly candidate — implementation work, out of this remediation's scope (deliverables were R3B-1/2/4/5/8 only) |
| R3B-6 | `STALE_EXPECTATION` is raised (`engine/v2/data/repository.py:585`) but unregistered against `DATA_FAILURE_CODES`; fails on `main` | Excluded from this task's brief; a separate fix |
| R3B-7 | `test_action_finality_writes_a_coverage_output_from_monkeypatched_frames` fails on `main` (`covered_tickers: []` vs expected `['AAA']`) | Excluded from this task's brief; a separate fix |

Phase 3B is **not** fully closed while R3B-3, R3B-6 and R3B-7 remain open.
This record closes the acceptance-evidence and documentation slice honestly:
the gate passes on bounded-scope evidence, the scope limitation and the
whole-file-rewrite semantics are recorded rather than papered over, and the
report and runbook now carry what the 3B plan requires of them.
