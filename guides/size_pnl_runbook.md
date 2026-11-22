# EXP-188 runbook (SIZE decision-level P&L re-score, inner and outer stages)

Operating steps for `experiments/EXP-188_size_decision_pnl_rescore/run.py --stage inner|outer`. Design:
`guides/size_decision_pnl_design.md`. Only the supervisor runs these steps (one heavy job at a time, never
beside the v14 run, container 12 GiB). Agents never run them.

`--stage` is `inner` (2023-24 tuning) then `outer` (2025 decisions). The v14 2025 refit predictions are bound by
`size_pnl_outer_inputs` (design section 12, #2658). A ready model with no 2025 rows still refuses
`outer_predictions_unbound`.

## Before any run

1. The v14 **rerun** (not the killed `v2full14` run) has completed and been collected. Record its `plan_id`,
   `registration_id`, driver-seal id and `panel_id` in a JSON file (`--expected`). `panel_id` is the plan's candidate `source_binding_id` (the one value every `plan.candidates[i].source_binding_id` shares); the factory derives it from the sealed plan and `bind_run` refuses `panel_id_mismatch` if `--expected` differs.
2. The spec is registered: exactly one PLANNED `EXP-188` ledger row whose `spec_hash` equals the spec's
   (`python3 -m experiments.size_pnl_prereg --check`). Without it the runner refuses
   `EXPERIMENT_IDENTITY_CONFLICT` (`planned_row_missing`) before reading anything.
3. `--sources experiments.size_pnl_sources:build_sources --sources-config <config.json>` builds
   `size_pnl_driver.Sources` (design section 13: catalog, store root, v14 spec/request/plan-ref, pinned snapshot,
   production hash). `production_hash` must equal the `tier4_forecasts` dataset-version manifest hash of the pinned
   snapshot, else `production_hash_mismatch` before `size_pnl_tables` reads any table. A missing table, column or event refuses
   `source_table_missing`; a non-finite, FLT_MAX or non-USD-scale `mcap_usd` refuses `source_value_invalid`.

## Real run

```
python3 tools/bounded_run.py --heavy --cores 8 -- python3 experiments/EXP-188_size_decision_pnl_rescore/run.py \
  --stage inner --no-ledger --expected <expected.json> --sources <module>:<func> --sources-config <config.json> --out-dir <out-dir>
```

Use an `--out-dir` under `experiments/EXP-188_*/results/`. The run writes `receipts/inputs_inner.json`, one
`receipts/tuning_<model>.json` per ready model, and `receipts/inner.json`. Receipts are immutable: a rerun with
the same inputs replays read-only; changed inputs refuse `EXPERIMENT_IDENTITY_CONFLICT`. Never delete a receipt
to force a rerun; use a fresh out-dir and a fresh experiment. Without `BOUNDED_RUN_NESTED=1` the runner refuses
`not_bounded`.

## Smoke run (50 tickers)

Write a JSON array of about 50 tickers' admitted event ids to `<ids.json>` (at most 1 MiB) and run both stages
with the same file and a fresh `--out-dir` under `/tmp` (smoke output is disposable):

```
python3 tools/bounded_run.py --heavy --cores 8 -- python3 experiments/EXP-188_size_decision_pnl_rescore/run.py \
  --stage inner --event-ids <ids.json> --no-ledger --expected <expected.json> \
  --sources experiments.size_pnl_sources:build_sources --sources-config <config.json> --out-dir <smoke-dir>
```

Here `<config.json>` holds `catalog`, `store_root`, `spec`, `request`, `plan_ref`, `snapshot_id`, `production_hash`
(design section 13); a real run uses the same `--sources` and `--sources-config`.

Then repeat with `--stage outer` and the same `--event-ids`. The ids are receipted as `scope`; a full-population
`inner.json` cannot feed a smoke outer run (or the reverse): `scope_mismatch`. An id outside the admitted
population refuses `key_not_admitted`; an empty array refuses `empty_population`. Record the smoke's RSS and wall
time from stderr before scheduling the full run.

## Outer stage

Run `--stage outer` without `--event-ids`, against the same `--out-dir` as the completed inner run. It refuses
before any chunk is built unless `inner.json` and every ready model's tuning receipt exist and match this run
(`inner_receipt_mismatch`, `scope_mismatch`, `inner_inputs_mismatch`, `inner_tables_mismatch`). It warms the
history from 2024-07, decides each 2025 month before reading that month's labels, and writes
`receipts/inputs_outer.json`, `receipts/outer.json` and `decisions.parquet` (model, book, arm or rule, structure,
event, status, reason, marks, P&L). Rerunning with identical inputs keeps `decisions.parquet` untouched; a changed
input refuses `EXPERIMENT_IDENTITY_CONFLICT`. A symlinked or oversized `decisions.parquet` refuses
`decisions_artifact_invalid`.

## Verify the outer binder (after v14 completes and is collected)

Not yet verified on the real store. Before the full outer run, run the 50-ticker smoke `--stage outer
--event-ids <ids.json> --no-ledger` with a fresh `/tmp` out-dir (same `--expected`, `--sources`). On stderr
`[pnl_outer]` every roster model must be `bound` or excluded with a reason (`outer_missing`, `outer_failed`,
`partial`, `component_outer_*`), the bound count must match the v14 `refit_units` for 2025, and no
`outer_forecasts_hash_mismatch` or `outer_identity_mismatch` may appear. Record the result in #2658, then
schedule the full run.

## Exit codes and artifacts

Exit 0 prints a JSON summary on stdout. Exit 2 prints `{"refused": code, "message": ..., "reason": ...}`;
`reason` is the machine-readable cause (for example `sources_invalid`, `expected_mismatch`, `not_bounded`,
`no_ready_models`, `key_not_admitted`). Progress goes to stderr: stage start and end with elapsed time and RSS,
quarters done with rate and ETA, and per-model tuning lines. The synthetic calibration arm
(`guides/size_pnl_synthetic_calibration_design.md`) is a separate slice that will get its own gated helper and reuses `model_events`.
