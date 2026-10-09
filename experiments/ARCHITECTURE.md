# Experiments architecture

## Native gate-variant pricing

`v2_gate_variant.price_variant(repository, snapshot, spec, *, strategy,
as_of_month, event_ids=None)` resolves one immutable single-arm plan and
freshly prices its eligible canonical events from that snapshot's option
chains and calendar. It accepts only a finite `fill` economics declaration;
the alpha passed to native replay is the resolved plan's own value. Returned
stage inputs bind the strategy, alpha, snapshot, event population and holdout
context. No prepriced trade frame or previous experiment output is accepted.
The source population uses `research.experiment_population.load_population`
with `purpose="sweep"`; raw quotes are read only after exclusion succeeds.
This is a pricing prerequisite, with no gate fitting or sweep lifecycle.

`python -m experiments.v2_gate_variant --no-ledger` is the smoke entrypoint.
It pins the current scope head once, opens the existing catalog read-only,
and prints execution evidence. It writes no report, artifact or ledger row.
It is not an admitted supervisor runner or a promotion-authorizing evaluation.

| Condition | Outcome |
| --- | --- |
| Unknown runner/strategy, external input files, invalid fill, or other economics | `INVALID_EXPERIMENT_SPEC` before population or quote reads. |
| Missing or invalid pin/context, or an explicit excluded event | Preserve the shared population reader's typed refusal; no pricing result or report. |
| Replay cannot price every eligible event | `EXPERIMENT_VARIANT_FAILED`; no partial result is returned. Native data integrity refusals propagate. |
| Repeated call or failure | No cache, internal retry, transaction or publication; recompute from the same pin and never retain a partial result. |
| CLI omits `--no-ledger` | Refuse before opening the catalog; recorded execution is unavailable. |

## Evaluation recording and promotion receipts

| Condition | Outcome |
| --- | --- |
| `record_evaluation_result` has no `run_dir` or its results are not a `Mapping` | Raise `LedgerError` before appending a `ran` row. |
| `publish_receipt=False` | Append the `ran` row; skip artifact validation, finalization, and receipt publication. |
| Receipt publication is enabled and the artifact matches the run and spec | Append the row, finalize the checklist and `recording_mode`, then publish a receipt for the exact final bytes. |
| Validation or receipt publication fails after the append | Keep the `ran` row; the row alone cannot authorize promotion. |
| Promotion consumes an artifact | Require matching artifact, requested spec, receipt, run ID, digest, and qualifying ledger row; otherwise refuse with `PROMOTION_LEDGER_RECEIPT_MISSING`. |
| An `EvalResult` is recorded | Use its own `run_dir` for its metrics artifact, including for grid arms. |
