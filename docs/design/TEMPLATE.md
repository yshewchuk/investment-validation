# Design: <title>

Status: draft | design-approved | implemented
Spec: <link to the spec (PR body `<details><summary>Spec</summary>` or a
scratchpad/spec file path)>

<!--
This document is public once the PR opens. No strategy thresholds, gate-logic
numbers, edge figures, or local filesystem paths.
See docs/ARCHITECTURE.md for the layer table, import-direction rules,
production entrypoints and invariants this template cites.
-->

## Context

The problem this change solves, in a few sentences, plus the spec link above.
State what breaks or stays missing if this change does not happen.

## Decisions

For each decision this design makes:

- **Decision:** <what is being decided>
  - **Options considered:** <at least two, including "do nothing">
  - **Choice:** <which one>
  - **Why:** <the reason this option won, not just a restatement of the choice>

## Layers and modules touched

List every `engine/v2/**` (or legacy `engine/**`) package this change adds
to or edits, its layer number from ARCHITECTURE.md §2, and the direction of
any new import edge. If a new edge crosses layers, state which side is
lower and confirm the edge points down only. If a new edge reaches legacy
from v2, name the adapter module and state whether `checks/legacy_adapters.json`
needs a new entry.

| Package | Layer | New/changed | Imports (new edges) |
|---|---|---|---|

## Production call path

Entrypoint → ... → new code. Name the actual entrypoint from ARCHITECTURE.md
§4 (legacy nightly step, v2 nightly `GRAPH` stage, ops CLI subcommand,
serving route, etc.) and every hop between it and the new code. If nothing
in production reaches this change yet, say so explicitly and explain why
(e.g. "behind a flag until Phase N", "exercised only by `ops rescore`,
which is deliberately manual").

## Changed interfaces and their callers

For each changed function, class or contract: its signature/shape before and
after, and every caller — found by grep, including callers outside this
diff — with what changes for each one.

## Failure semantics

Using the 4c R1–R6 template:

- **Missing input:** what happens when a required input is absent or
  unusable. Cite the typed refusal/withheld status used, never a silent
  default.
- **Cache:** what is cached, its invalidation trigger, and staleness bound.
- **Retry:** what is safe to retry, what is not, and why.
- **Transaction:** what is atomic, what commits it, and what a partial
  transaction leaves behind.
- **Partial write:** what a crash mid-write leaves on disk/in the catalog,
  and how a later run detects and recovers from it.
- **Idempotency:** what makes a re-run safe — a dedupe key, a fenced
  attempt, an append-only write — and what would break it.

## Invariants touched

Cite the specific ARCHITECTURE.md §5 invariant(s) this change touches (native
vs. legacy provenance, typed refusal, no parity-only modes, shared parity
normalisation, snapshot/root isolation, stated failure semantics, no local
paths/raw exception text in published output, `--no-ledger` for experiment
smoke runs, legacy stays frozen until cutover) and how this change satisfies
each one.

## Test plan

One row per acceptance criterion in the spec:

| Acceptance criterion | Test | How this test could fail |
|---|---|---|

## Out of scope

What this change deliberately does not do, and why — including anything the
spec mentioned that is deferred to a later change.

## Changed during implementation

Filled in only if the implementation deviates from the sections above. Each
deviation: what changed, why, and which section above it invalidates. Left
empty (or omitted) if implementation matched the design exactly.
