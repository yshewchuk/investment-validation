# STR-THRU distribution research

[DESIGN.md](DESIGN.md) defines the four approved implementation bundles and scientific controls. [ARCHITECTURE.md](ARCHITECTURE.md) documents Bundle C's fitting/results and causal residual boundary; public modules remain in the parent `experiments` package.

- A supplies magnitude histogram math and frozen payoff controls.
- B supplies separate native ownership, source/fold admission, raw-pair budgets and claims.
- C adds admitted numeric fit receipts and earlier-OOF residual controls; raw probabilities remain explicitly uncalibrated.
- D's bounded runner, complete attempts, economics and portfolio reports are not implemented here.

Focused synthetic checks live in `tests/v2/research/test_distribution_*.py`. No real data, provider, final-holdout, deployment or automatic run authority follows from these APIs. Shared platform owners retain source eligibility, transport and holdout policy.

## Fixed CAS worker

Both in-process fit and fixed child first call `validate_fit_output` to validate raw numeric results against the recipe, ordered matrices/labels/events and the shared result contract. It returns ordered rows; identity mismatches retain `EXPERIMENT_IDENTITY_CONFLICT` and histogram checks retain their native data error. `publish_fit_output` then receives these supported rows, a shared empirical/unresolved tail and diagnostics. Forecasts are event-key sorted; excess order/duplicates remain intact. It uses existing CAS sections and bounded-cadence stderr progress; it does not validate admission, grant a retry or write the completion journal. Native CAS failures propagate; a failed later publication can leave complete unreferenced CAS objects.

The installed `python -m experiments.distribution_worker` is an internal protocol, not a public experiment CLI. A launching adapter must supply its private header path and explicit resident-memory/CPU/affinity limits (the shared VmRSS watchdog, not RLIMIT_AS), enforce wall/stdout/exit bounds and validate results before journal completion. Before OS calls, the child requires positive bounded plain-integer memory/CPU limits and a plain-integer `cpu_id` in `[0, 2**31 - 1)`. An unavailable CPU still returns sanitized `RESOURCE_UNAVAILABLE`. After containment, it requires a mapping header and validates its supplied recipe through the existing canonical recipe owner before reading recipe fields. It then checks plain declared horizon, unique string event axes/present projected rows and a nonempty store-root string; these are protocol shape checks, not source eligibility or path-security proofs. It caps its header/output document using the existing record limit, reads admitted feature/training shards only, uses the installed fixed native pair and writes a result document to CAS. stdout contains one compact artifact descriptor or sanitized typed failure; unexpected exceptions map to `RESOURCE_UNAVAILABLE`. No catalog, provider, fresh outcome, holdout policy or production path. This slice supplies no general launcher, automatic retry, experiment attempt or claim of calibrated probabilities.
