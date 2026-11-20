# EXP-188 runbook (SIZE decision-level P&L re-score, inner stage)

Operating steps for `experiments/EXP-188_size_decision_pnl_rescore/run.py --stage inner`. Design:
`guides/size_decision_pnl_design.md`. Only the supervisor runs these steps (one heavy job at a time, never
beside the v14 run, container 12 GiB). Agents never run them.

The outer stage (2025 decisions, warm-up, `decisions.parquet`), the `--event-ids` smoke and the decisions
output land in S9b (#2660). Until then `--stage` accepts `inner` only.

## Before any run

1. The v14 **rerun** (not the killed `v2full14` run) has completed and been collected. Record its `plan_id`,
   `registration_id`, driver-seal id and `panel_id` in a JSON file (`--expected`).
2. The spec is registered: exactly one PLANNED `EXP-188` ledger row whose `spec_hash` equals the spec's
   (`python3 -m experiments.size_pnl_prereg --check`). Without it the runner refuses
   `EXPERIMENT_IDENTITY_CONFLICT` (`planned_row_missing`) before reading anything.
3. A `--sources MODULE:FUNC` callable builds `size_pnl_driver.Sources` from the parsed args (catalog, store
   root, v14 run, pinned snapshot, production table hash). The store adapter is not part of this slice.

## Real run

```
python3 tools/bounded_run.py --heavy --cores 8 -- python3 experiments/EXP-188_size_decision_pnl_rescore/run.py \
  --stage inner --no-ledger --expected <expected.json> --sources <module>:<func> --out-dir <out-dir>
```

Use an `--out-dir` under `experiments/EXP-188_*/results/`. The run writes `receipts/inputs_inner.json`, one
`receipts/tuning_<model>.json` per ready model, and `receipts/inner.json`. Receipts are immutable: a rerun with
the same inputs replays read-only; changed inputs refuse `EXPERIMENT_IDENTITY_CONFLICT`. Never delete a receipt
to force a rerun; use a fresh out-dir and a fresh experiment. Without `BOUNDED_RUN_NESTED=1` the runner refuses
`not_bounded`.

## Exit codes and artifacts

Exit 0 prints a JSON summary on stdout. Exit 2 prints `{"refused": code, "message": ..., "reason": ...}`;
`reason` is the machine-readable cause (for example `sources_invalid`, `expected_mismatch`, `not_bounded`,
`no_ready_models`, `key_not_admitted`). Progress goes to stderr: stage start and end with elapsed time and RSS,
quarters done with rate and ETA, and per-model tuning lines. The synthetic calibration arm
(`guides/size_pnl_synthetic_calibration_design.md`) is a separate slice that reuses `size_pnl_driver.prepare`
and `model_events`.
