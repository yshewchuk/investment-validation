# Performance review: repeated work and safe reuse

Gate and CodeRabbit reviews apply these criteria to changed code and affected
callers, using the existing scope and finding-severity rules in `AGENTS.md` and
`CONTRIBUTING.md`. Performance is not a blanket reason to block a PR.

- Trace the actual consumer: legacy nightly, native nightly/scoring, experiment
  admission/recording/replay, or model fitting. Identify the repeated operation,
  its caller/loop and plausible input size or call count before raising a finding.
  Look for repeated full-inventory scans or identical pinned reads; rebuilding
  holiday calendars; repeated sorting, JSON normalization or array conversion;
  and re-verifying the same durable evidence within one response. Check whether
  existing request-local reuse already removes the cost.
- Prefer a small, semantics-preserving reduction in repeated work over a new
  cache or redesign. Use the standard-tool rule in `ARCHITECTURE.md` §6. Reuse
  only where input identity and lifetime justify it: same-call verified evidence is not
  permission to skip fresh integrity checks on later replay or recovery calls.
  Preserve admission, provenance, caller limits, refusals and scientific results.
- For caching, verify actual immutable identity, not just a caller-declared hash;
  separate snapshots, sessions and requests as required. Check entry/byte bounds,
  retained-object memory, mutation safety, eviction and concurrency. Do not
  replace required fresh reads with stale evidence or cache transient failures.
  Include cold-start validation/allocation costs and realistic reuse frequency;
  a warm hit alone does not establish a net benefit.
- Support a finding with a concrete call path plus avoidable work/counts,
  complexity or a representative benchmark. Label measured versus estimated
  savings, inputs and cold/warm conditions; do not extrapolate a helper speedup
  into a whole-nightly or whole-experiment claim without workload evidence.
  Lack of a benchmark alone is not a defect, and speculative micro-optimizations
  or unrelated redesigns are not required fixes. Rate likelihood and impact by
  demonstrated workload consequence under the existing gate tiers; do not
  invent a new performance severity threshold.
- For an optimization fix, look for independent output/refusal oracles and
  deterministic read/call-count tests that fail when the redundant work returns.
  Where applicable, cover changed inputs, cold/warm use, eviction and concurrent
  misses, plus fresh checks on separate replay/recovery calls. Timing evidence
  supplements these tests; fragile wall-clock assertions are not required.
