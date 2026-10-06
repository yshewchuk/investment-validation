# Package test selection by layer

## Decision

PR test selection follows the enforced package layout. For each changed path,
resolve its package by directory using `checks/layer_map.py`; select tests in
that package and the reverse closure of packages that import it along allowed
layer-map edges. Run tests from `tests/v2/<package>/` for every selected
package. There is no component map and no import-graph or dynamic-load
failsafe. The allowed-import direction is importer → dependency; selection
walks it in reverse so a changed dependency selects its package and every
allowed importer above it.

## Path policy

| Changed path | Selection |
|---|---|
| `engine/v2/<package>/...` | The package named by the longest matching directory in `checks/layer_map.py`. |
| `tests/v2/<package>/...` | The matching package, including package-local helpers. |
| `tests/v2/integration/...` | Always run a changed integration test itself; also run it when any declared package changes. A test declares packages on one `# packages: ...` line. |
| `checks/`, `tools/`, `ui/`, root configuration, or `.github/` | Full suite: these paths are outside the layer map and can affect any package. This includes the selector and workflow themselves. |
| Root-level `tests/` files or helpers outside `tests/v2/` | Full suite until moved into a package folder or declared integration folder. In particular, move `tests/ops_support.py` to `tests/v2/ops/` and `tests/data_scan_support.py` to `tests/v2/data/`. |
| Markdown under `guides/` or `ARCHITECTURE.md` files | The always-run documentation/meta tests. |
| Any other path | Full suite, with the unmapped path and reason printed. |

Always add the layer/import-boundary checks, repository hygiene checks and
architecture-document/link/budget checks to every PR selection. A change to
`checks/`, `tools/`, `ui/` or configuration still runs the full suite in
addition to this set. Documentation paths are recognized explicitly; they do
not rely on a heuristic about which code may read a Markdown string.

## Test layout and declarations

Move package tests from `tests/test_*.py` to `tests/v2/<package>/`, mirroring
the package boundaries in `engine/v2/`. The file path determines its package;
the selector does not infer ownership from imports. Keep a test with its
package even when it imports shared lower-layer packages. Put a test that
asserts behavior across package boundaries in `tests/v2/integration/` and
declare every participating package on exactly one `# packages: ...` line.
The declaration check rejects unknown package names, duplicate entries,
missing declarations and declarations on non-integration tests. An invalid or
stale declaration triggers the full suite and a visible selector error so
CI cannot silently narrow around it.

Moves are path-only renames with zero added lines wherever possible. Move
shared helpers with their owning package (`ops_support.py` to `ops`,
`data_scan_support.py` to `data`). Each move is a separate package PR. Tests
that depend on more than one package belong in integration even if one
package currently provides most of their imports.

## Selector and workflow boundary

Add a small selector module under `checks/`, with focused tests of path
mapping, reverse closure, integration declarations, unmapped paths and
failure behavior, plus a cache regression proving stale selected paths cannot
override the current base/head diff. `.github/workflows/tests.yml` supplies the changed paths
and consumes the selected test directories or an explicit full-suite result.
It does not call or modify `tools/mutation_pilot.py select-tests`; that
selector remains untouched for its existing mutation-testing role. If the
selector process crashes or emits invalid output, the workflow prints the
error and runs `tests/` in full. Pushes to `main`, scheduled runs and manual
dispatch continue to run `tests/` in full as the post-merge backstop.

## Evidence and validation before code

The owner-provided `test-selection-hubs.md` reports that 34 of 40 recent PRs
selected more than half of the suite, with fan-out often coming from selector
safety nets. It identifies real broad reach from contracts, foundation and
data. The supplied package simulation puts dashboard/research/diagnosis/
serving at roughly 2–7% of tests and contracts/foundation/data at roughly
68–74%; the latter are likely to remain broad under a package rule. It also
reports 26 tests with dynamic loading and non-import couplings that a static
package edge cannot see. Running the supplied `pkggraph.py` and `pkgsim.py`
on this worktree printed a 161-file median (40%), compared with the brief's
approximate 150 (39%); treat both as estimates until the fixed replay below.

Before the first code slice, replay the last 40 merged PRs against their
changed paths and recorded CI results. For each PR, record the base/head,
selected test files, tests that actually failed in CI, and whether each
failure is covered by the package rule. Report every PR where a real CI
failure would have been missed, including the failing test and the changed
path that failed to select it. The acceptance bar is zero known misses across
all 40 PRs. If CI history cannot establish the actual failure set for a PR,
mark that replay inconclusive and keep that path class on full-suite
selection; unknown evidence does not count as a pass. No code slice starts
until the replay report meets this bar or the rule is adjusted and replayed.

## Failure semantics (R1–R6)

| Case | Contract |
|---|---|
| **R1 — unmapped path** | Select `tests/` in full and print the path and the reason it is outside the declared path policy. Never treat an empty selection as success. |
| **R2 — deleted or renamed file** | Classify changed paths from the base-to-head diff even when a path no longer exists in the checkout. Treat a rename as deletion plus addition; if either side is unmapped, run the full suite. |
| **R3 — moved test** | A package-test move selects both old and new package closures. A changed integration test selects itself even if none of its declared packages changed; a move into or out of integration also validates its declaration and selects the declared packages. An invalid move falls back to the full suite. |
| **R4 — stale integration declaration** | Validate each one-line declaration against `checks/layer_map.py` and the test location. Missing, duplicate, unknown or misplaced declarations select the full suite and fail the declaration check. |
| **R5 — selector crash or bad output** | The workflow reports the failure and runs `tests/` in full. A selector error must never leave pytest with a partial or empty target list. |
| **R6 — CI cache** | Recompute selection from the current base/head diff on every PR run. Do not restore selected paths from a cache; dependency caches must not supply or override the selection result. |

## Scope

This design changes no runtime behavior, test placement or workflow in this
PR. It does not change the mutation-testing selector, infer non-import
couplings, or promise that a package graph detects every fixture, data-file,
subprocess or dynamic-load dependency. The full main-branch suite remains the
backstop for those gaps. Findings outside this test-selection contract are
separate work.

## Follow-up slices (estimated added code / test / documentation lines)

The first slice builds and tests the selector without moving tests. Later
slices move one package folder per PR; pure renames add zero lines. Estimates
include any import rewiring needed to keep each slice green. Start with the
lowest-risk, narrow-closure packages and finish with shared lower layers.

| Order | Slice | Code / tests / docs |
|---:|---|---:|
| 1 | Selector module, selector tests and checks; no moves | 35 / 55 / 15 |
| 2 | `dashboard` | 0 / 10 / 5 |
| 3 | `research` | 0 / 15 / 5 |
| 4 | `diagnosis` | 0 / 20 / 5 |
| 5 | `serving` | 0 / 25 / 5 |
| 6 | `ops` and `ops_support.py` | 0 / 30 / 5 |
| 7 | `ledger` | 0 / 20 / 5 |
| 8 | `parity` | 0 / 15 / 5 |
| 9 | `models/training` | 0 / 20 / 5 |
| 10 | `scoring` | 0 / 25 / 5 |
| 11 | `models` | 0 / 25 / 5 |
| 12 | `domain/generation` | 0 / 15 / 5 |
| 13 | `domain/valuation` | 0 / 15 / 5 |
| 14 | `domain/scenarios` | 0 / 15 / 5 |
| 15 | `domain/simulation` | 0 / 10 / 5 |
| 16 | `registry` | 0 / 15 / 5 |
| 17 | `features` | 0 / 25 / 5 |
| 18 | `evaluation` | 0 / 10 / 5 |
| 19 | `data` and `data_scan_support.py` | 0 / 30 / 5 |
| 20 | `foundation` | 0 / 25 / 5 |
| 21 | `contracts` | 0 / 20 / 5 |
| 22 | Remaining cross-package integration moves, one owning package per PR | 0 / 20 / 5 |

Contracts, foundation and data are last because their real reverse closures
are broad; if the replay shows that their package rule selects nearly the
whole suite, leave those tests in place rather than moving them for layout
alone. Every follow-up slice stays within the code-PR size limits and updates
this design or the architecture contract only when its behavior differs.
