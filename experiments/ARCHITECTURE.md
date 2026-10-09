# Experiments architecture

## Evaluation recording and promotion receipts

`experiments.lib.record_evaluation` appends a `ran` row before finalizing the
metrics artifact. Receipt publication is enabled by default: the recorder
requires the artifact's `run_id` and embedded `spec_hash` to match the
evaluated run and spec, recomputes the checklist against the ledger, marks the
artifact `recorded`, and writes a receipt bound to the finalized artifact's
exact bytes. Promotion validates the artifact, requested spec, receipt, and
qualifying ledger row before using the metrics.

The ledger append is the commit point. A later validation or receipt-write
failure does not undo that row; the row alone cannot authorize promotion.
With `publish_receipt=False`, the recorder uses ledger-only mode: it appends
the `ran` row and returns without validating or finalizing metrics or
publishing a receipt. `record_evaluation_result` records an `EvalResult` from
its own `run_dir`, so grid arms bind the metrics artifact written in that arm
directory.
