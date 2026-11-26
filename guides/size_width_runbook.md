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
   replays read-only and changed inputs refuse (first on the stage's inputs receipt, `inputs_receipt_changed`; the outer preflight against the inner receipts uses `tuning_receipt_changed`). Never delete a receipt to force a rerun.

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

The ids limit only the events priced. Pools, the error-model fit and the fake use the whole history, with a full run's
exclusions (facts included); the fake draws price only the smoke months. `smoke_no_fake` appears only when
that history has too few pairs. A stage that prices no event in any month refuses `smoke_priced_nothing`. Expect few
or no trades: the trade bar needs 100 earlier scored rows per structure from the smoke's own sweep (`thin_history`), so
tuned arms are `k_defaulted` (fewer than 300 trades). The ids are receipted as `scope`, so a smoke
receipt cannot feed a full run (`inputs_receipt_changed` or `tuning_receipt_changed`). Record RSS and wall time from stderr before the full run.

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
  last) is then absent, so the outer and report stages refuse. After the tunings it writes `receipts/tuning_band.json`
  and runs the fallback on each distinct fixed-factor arm (production, A-D) at three `phi` (at most 15 extra sweeps,
  one more simulation per month).
- **Outer** (2025): refuses before any chunk is built unless `inner_summary.json`, every arm's receipt and
  `tuning_band.json` exist and match (`tuning_receipt_missing`, `tuning_receipt_changed`). It warms the history from
  2024-07, writes each 2025 month to `months/<period>.parquet` before it reads that month's labels (R6), then makes a
  second pass over the months for the fallback on k = 1 and the recommended arm (at most 6 sweeps), and writes
  `outer_summary.json`.
- **Progress** goes to stderr: per month `done/total`, rate per minute, ETA, RSS; stage start and end with elapsed time.
  stdout is one JSON summary. A refusal prints `{"refused", "message", "reason"}` and exits 2.

## Arm E per-event choices (`e_choices/`)

Both stages write `<out-dir>/e_choices/<period>.parquet` for every output month. The inner stage (2023H2-2024) has no
`months/` files and has already loaded its labels. The outer stage (2025) writes each choice month after that month's
`months/` rows and before it reads that month's labels. The files are kept for later analysis only: no gate, objective, tuning or account reads it.

- **Rows:** one per simulated candidate event and structure, whatever the decision. An event with no simulation is
  dropped from the sweep (as from `months/`), so it has no row.
- **Choice:** `event_id`, `event_date`, `structure`, `k` (chosen; the grid value nearest 1 when no `k` is scored),
  `exp_pnl_sim` (at the chosen `k`), `exp_pnl_sim_k1`, `status` and `reason` (the gate's decision on arm E).
- **Marks:** `entry_cost` is the entry mark (debit per share) of any candidate that priced. The rest is filled for
  taken rows only, and is null otherwise: `exit_value`, `exit_value_mid`, `pnl_dollars`, `pnl_dollars_mid`, `lots`,
  `debit_dollars` (`entry_cost` x 100 x `lots`), `secured_per_contract`, and for each alpha in {0.00, 0.25, 0.50, 0.75,
  1.00} `entry_a*` and `exit_a*` (per share) and `pnl_a*` (dollars), with entry repriced at that alpha (the breakeven
  ladder's marks).
- **Receipts:** `receipts/e_choices_inner.json` and `receipts/e_choices_outer.json` hold stage, `inputs_hash`, scope, and per
  month the file's sha256, rows and taken count. Files and receipts are immutable like `months/`: a rerun with other
  bytes refuses (`month_freeze_changed`; the receipt, `tuning_receipt_changed`). A stage that stops before its
  receipt leaves unbound month files: do not analyse them. Existing outputs and receipts are unchanged by this.

## Report

`--stage report` reads no sources (`--sources` is required but never imported) and changes no receipt. It first
checks that the receipts belong to one driver run: the two summaries and the two inputs receipts agree on stage,
`inputs_hash` and scope (the summaries also on the smoke flag), each summary's digest matches its full inputs receipt,
the outer arms match the inner `c_status`, both inputs
receipts carry one identity, and every arm's tuning receipt passes `size_width_tuning.frozen`. A mismatch refuses
`tuning_receipt_changed`; an `--expected` value that differs from the receipted identity refuses `expected_mismatch`;
nothing is written. It needs the inner and
outer receipts and writes `<out-dir>/REPORT.md` (T1-T7, 2025 tables titled "2025 NOT blind for widths") and
`<out-dir>/proposed_rule.json` (the proposed board rule: divisors `divisor / k`, or `no_change`, or the E rule). In an
A-D rule, `flatness_se` is always T3's flatness at `k = 1` on the primary per-structure curve (arm C's when C is
calibrated, else arm B's), whichever arm is recommended: it is not the recommended arm's own tuning curve. T8 and
T9 (realised widths, tolerance bands, the fallback at each `phi`, the adoption answer) follow, after `tuning_band.json`
is checked against the inner summary. The recommendation and adoption answer are evidence for the user's separate board
decision, not a validated result. A short T5b section states the `e_choices` row and taken counts from their
receipts (no analysis; a receipt bound to other inputs or scope refuses `tuning_receipt_changed`). A run directory
with neither receipt has no T5b.

## Pricing and fill-model seams

`size_width_driver.PRICING` is `size_width_pricing.PRICING`: `size_pnl_shortvol`'s own code on
`engine.chain_index.price_structure_indexed`, with results identical to `price_structure`'s.

Its marks come from a `size_width_fills.Fills` object, `size_width_pricing.load(fills=...)`: `entry(priced)` gives the
entry cost per share and relative spread (the gate's entry cost, the `exp_pnl_sim` entry-cost input and the realised
entry mark), and `exit_value(chunk, traded, alpha, mid)` the exit mark. The default, `CurrentFills`, calls
`size_pnl_shortvol`'s functions. Strike resolution, the rung-3 zero floor and the simulation's Black-Scholes exit
repricing are not marks and stay outside the seam. No command selects another model; receipts do not record one, so a
change that does must bind its identity in the inputs receipt.

## Amendment 1 run (fair-value fills, model F)

Design: Amendment 1 (A1.9; re-split note under A1.10). Outputs are post-hoc. The steps above apply, with these
additions. F is selected only by the `spec.yaml` `amendments` entry that carries `fill_model` (`fill_model.id: "F"`).

1. **Wait to register** until W7a1 to W7b2, W7d and W7e are merged. Before W7d no code reads `fill_model`, and the guard
   checks only rows and hashes: a row registered early would let current marks run as primary under an F spec. From the
   W7e merge until registration every stage refuses `planned_row_ambiguous`, including a rerun of the original.
2. **Bindings, before registration, no outcome read.** `fill_model` must match:
   - `fvf_id`: `python3 -c "from experiments import fair_value_run as r; print(r.fvf_id())"`.
   - `reingest.manifest_sha256`: `sha256sum` of `manifest.json` under `reingest.path` (repo-relative).
   - `vol_scale`: the manifest names no `smvVol` unit; 1.0 (decimal) is assumed. F's load refuses `vol_unit` when the
     scaled median is outside (0.05, 5).
   - Quote snapshot: the sources config's `snapshot_id`.

   A mismatch stops the run. Changing a binding or rule needs a new amendment, hash and PLANNED row.
3. **Pins.** The spec cannot hold table hashes (the manifest lists none). Before the smoke, write one JSON file,
   `manifest_sha256` plus `files` (path relative to `reingest.path` to SHA-256), covering every quote and rate month the
   run reads. Keep it unchanged for the full run: the inputs receipts bind every hash. The pins do not prove that the
   tables and manifest came from one re-ingest run. W7d documents in `experiments/README.md` how the F inputs reach
   `run.py`.
4. **Register** (supervisor only): a second EXP-189 PLANNED row with the new hash
   (`python3 -c "from experiments import size_width_prereg as p; print(p.load_spec()[1])"`); the original row stays and
   the new one is last. Then `python3 -m experiments.size_width_prereg --check`. No agent writes
   `experiments/LEDGER.csv`.
5. **Smoke, 50 tickers:** all three stages, with the smoke commands above (`--no-ledger`, fresh `/tmp` out-dir). The ids
   are all admitted events of the 50 plan tickers with the smallest `sha256("<ticker>|exp189-a1")` (W7d supplies them;
   no outcome is used). Before the full run, record per stage: wall time, peak RSS, F load and pricing time, quote rows
   priced and their rate, entry coverage, and a full-run estimate. Entry status, vol-source and liquidity coverage must
   print before any outcome is read. No smoke result tunes F or a rule.
6. **Full rerun** of inner, outer and report in a fresh out-dir, one heavy job at a time. Reuse and delete no earlier
   receipt, month file or tuning. F's refusals (A1.7) leave no partial report.
7. **Report again once W7c2 lands**, on the same out-dir. An earlier `REPORT.md` may lack A1.1's labels and F's tables
   and diagnostics, so it is not the Amendment 1 report.
