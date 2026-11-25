# EXP-189 runbook (production short-vol width tuning, inner, outer and report stages)

Operating steps for `experiments/EXP-189_short_vol_width_tuning/run.py`. Design: `guides/size_width_tuning_design.md`
(the contract). Only the supervisor runs these steps: one heavy job at a time, never beside another heavy run
(a heavy run in progress means wait). Agents never run them; their tests use synthetic fixtures only.

## Before any run

1. EXP-188's inputs are ready: the same `--expected` JSON (`plan_id`, `registration_id`, `seal_id`, `panel_id`),
   `--sources experiments.size_pnl_sources:build_sources` and `--sources-config <config.json>` as the EXP-188 runbook
   (`guides/size_pnl_runbook.md`). EXP-189 reads them through `size_pnl_driver.Sources` unchanged.
2. The spec is registered: one PLANNED `EXP-189` ledger row, plus one per entry of the spec's `amendments`, the last
   with the spec's hash (`python3 -m experiments.size_width_prereg --check`). Without it every stage refuses
   `EXPERIMENT_IDENTITY_CONFLICT` (`planned_row_missing`) before reading anything. A changed `spec.yaml` needs a new
   PLANNED row first.
3. Use a fresh `--out-dir` under `experiments/EXP-189_*/results/`. Receipts are immutable; a rerun with the same inputs
   replays read-only and changed inputs refuse (`tuning_receipt_changed`). Never delete a receipt to force a rerun.

## Smoke run (about 50 events)

Write a JSON array of about 50 admitted event ids (at most 1 MiB) to `<ids.json>`. Use a fresh `--out-dir` under `/tmp`.

```
python3 tools/bounded_run.py --heavy --cores 8 -- python3 experiments/EXP-189_short_vol_width_tuning/run.py \
  --stage inner --event-ids <ids.json> --no-ledger --expected <expected.json> \
  --sources experiments.size_pnl_sources:build_sources --sources-config <config.json> --out-dir <smoke-dir>
python3 tools/bounded_run.py --heavy --cores 8 -- python3 experiments/EXP-189_short_vol_width_tuning/run.py \
  --stage outer --event-ids <ids.json> --no-ledger --expected <expected.json> \
  --sources experiments.size_pnl_sources:build_sources --sources-config <config.json> --out-dir <smoke-dir>
python3 experiments/EXP-189_short_vol_width_tuning/run.py --stage report --no-ledger --expected <expected.json> \
  --sources experiments.size_pnl_sources:build_sources --out-dir <smoke-dir>
```

Under 300 inner events no error model is fitted (`c_status` is `smoke_no_fake`): tuning C is skipped and tuned arms
A, B, D and E run with every unit `k_defaulted` (fewer than 300 trades). The ids are receipted as `scope`, so a smoke
receipt cannot feed a full run (`tuning_receipt_changed`). Record RSS and wall time from stderr before the full run.

## Full run

```
python3 tools/bounded_run.py --heavy --cores 8 -- python3 experiments/EXP-189_short_vol_width_tuning/run.py \
  --stage inner --no-ledger --expected <expected.json> \
  --sources experiments.size_pnl_sources:build_sources --sources-config <config.json> --out-dir <out-dir>
python3 tools/bounded_run.py --heavy --cores 8 -- python3 experiments/EXP-189_short_vol_width_tuning/run.py \
  --stage outer --no-ledger --expected <expected.json> \
  --sources experiments.size_pnl_sources:build_sources --sources-config <config.json> --out-dir <out-dir>
```

- **Inner** (2023H2-2024, burn-in from 2023-01): one sweep over the grid and arm E, the error model fit, tuning C
  (`N = 20` draws), the anchor, tunings A, B, D. It writes `receipts/inputs_inner.json`, `tuning_A..E.json`
  (`tuning_C.json` unless the fake is not calibrated: `fake_not_calibrated`) and `inner_summary.json`. It holds every
  inner quarter chunk in memory (the fake and anchor re-use them): watch RSS on the smoke run first. The EXP-189 code
  emits no memory refusal of its own: over its limit, `bounded_run` stops the process, and `inner_summary.json` (written
  last) is then absent, so the outer and report stages refuse.
- **Outer** (2025): refuses before any chunk is built unless `inner_summary.json` and every arm's receipt exist and match
  (`tuning_receipt_missing`, `tuning_receipt_changed`). It warms the history from 2024-07, writes each 2025 month to
  `months/<period>.parquet` before it reads that month's labels (R6), and writes `outer_summary.json`.
- **Progress** goes to stderr: per month `done/total`, rate per minute, ETA, RSS; stage start and end with elapsed time.
  stdout is one JSON summary. A refusal prints `{"refused", "message", "reason"}` and exits 2.

## Report

`--stage report` reads no sources (`--sources` is required but never imported) and changes no receipt. It first
checks that the receipts belong to one driver run: the two summaries and the two inputs receipts agree on stage,
`inputs_hash` and scope (the summaries also on the smoke flag), the outer arms match the inner `c_status`, both inputs
receipts carry one identity, and every arm's tuning receipt passes `size_width_tuning.frozen`. A mismatch refuses
`tuning_receipt_changed`; an `--expected` value that differs from the receipted identity refuses `expected_mismatch`;
nothing is written. It needs the inner and
outer receipts and writes `<out-dir>/REPORT.md` (T1-T7, 2025 tables titled "2025 NOT blind for widths") and
`<out-dir>/proposed_rule.json` (the proposed board rule: divisors `divisor / k`, or `no_change`, or the E rule). In an
A-D rule, `flatness_se` is always T3's flatness at `k = 1` on the primary per-structure curve (arm C's when C is
calibrated, else arm B's), whichever arm is recommended: it is not the recommended arm's own tuning curve. T8 and
T9 (tolerance bands, fallback) enter through `assemble(..., w5_tables=)`. W5's modules are merged, but the driver does
not run them yet (slice W6b-3), so until then the report says T8 and T9 are not available. Do not start the recorded
run before W6b-3: the `tuning_band` receipt and fallback arm belong to it. The recommendation is evidence for the user's separate board decision, not a validated result.

## Pricing seam

`size_width_driver.PRICING` (default `experiments.size_pnl_shortvol`) is the one place a faster strike resolver
(`engine/chain_index.py`; routed in slice W6b-3) plugs in; it must keep `simulate`, `score_events`, `decide`, `score` and `secured`
bit-identical to `size_pnl_shortvol`.
