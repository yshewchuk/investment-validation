# EXP-188 runbook (SIZE decision-level P&L re-score)

Operating steps for `experiments/EXP-188_size_decision_pnl_rescore/run.py`. Design:
`guides/size_decision_pnl_design.md`. Only the supervisor runs these steps (one heavy job at a time, never
beside the v14 run, container 12 GiB). Agents never run them.

## Before any run

1. The v14 **rerun** (not the killed `v2full14` run) has completed and been collected. Record its `plan_id`,
   `registration_id`, driver-seal id and `panel_id` in a JSON file (`--expected`).
2. The spec is registered: exactly one PLANNED `EXP-188` ledger row whose `spec_hash` equals the spec's
   (`python3 experiments/size_pnl_prereg.py --check`). Without it the runner refuses
   `EXPERIMENT_IDENTITY_CONFLICT` (`planned_row_missing`) before reading anything.
3. A `--sources MODULE:FUNC` callable builds `size_pnl_driver.Sources` from the parsed args (catalog, store
   root, v14 run, pinned snapshot, production table hash). The store adapter is not part of this slice.

## Smoke (50 tickers, never recorded)

Pick the event ids of about 50 tickers inside the admitted population, one id per line, into a file under
`/tmp`. Then run both stages with `--event-ids` and a scratch `--out-dir` under `/tmp`:

```
python3 tools/bounded_run.py --heavy --cores 8 -- python3 experiments/EXP-188_size_decision_pnl_rescore/run.py \
  --stage inner --no-ledger --expected /tmp/exp188_expected.json --sources <module>:<func> \
  --event-ids /tmp/exp188_smoke_ids.txt --out-dir /tmp/exp188_smoke
```

then the same with `--stage outer`. The smoke records wall time and peak RSS (target well under 4 GiB, about
tens of minutes for the full run). A subset never widens admission: an id outside the admitted population
refuses `HOLDOUT_ACCESS_DENIED`. Delete the scratch directory afterwards.

## Real run

Same command without `--event-ids`, `--out-dir` under `experiments/EXP-188_*/results/`. `inner` first; it
writes `receipts/inputs.json`, one `receipts/tuning_<model>.json` per ready model and `receipts/inner.json`.
`outer` refuses (`tuning_receipt_missing`) unless all of them exist. Receipts are immutable: a rerun with the
same inputs replays read-only; changed inputs refuse `EXPERIMENT_IDENTITY_CONFLICT`. Never delete a receipt to
force a rerun; use a fresh out-dir and a fresh experiment.

The outer stage replays 2024 H2 as warm-up (history only, no rows) so the trailing cutoff exists in
2025-01, decides each 2025 month before reading that month's labels, and writes `decisions.parquet` plus
`receipts/outer.json` (the parquet hash). Without `BOUNDED_RUN_NESTED=1` the runner refuses `not_bounded`.

## Exit codes and artifacts

Exit 0 prints a JSON summary on stdout; exit 2 prints `{"refused": code, "message": ...}`. Progress goes to
stderr (stage start and end with elapsed time and RSS, quarters and months done with rate and ETA, skip
reasons). `REPORT.md` is built from `decisions.parquet` by `size_pnl_report` in a later slice. The synthetic
calibration arm (`guides/size_pnl_synthetic_calibration_design.md`) is a separate slice that reuses
`size_pnl_driver.prepare` and `model_events`.
