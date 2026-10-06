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

## Test-layout ratchet

Use a blocking, checked-in budget with the same absolute, no-exemptions,
no-grandfathering convention as `checks/code_budgets.py`. Slice 1 adds
`checks/test_layout_budget.txt`; it records the exact count of the unmoved set,
initially **403**. The unmoved set is the current root-level
`tests/test_*.py` paths. The check derives membership from that path pattern
and requires the recorded count to equal the current set size. Its budget may
only decrease from the PR base to its head.

This ratchet does not change the merged path policy: root-level tests and
helpers, `tools/` and `checks/` continue to select the full suite. The current
33-of-40 fallback rate is accepted while the layout improves incrementally.

Add the ratchet check to the always-run meta set. It fails when a new
`test_*.py` file is added outside `tests/v2/<package>/` or
`tests/v2/integration/`, when the unmoved set grows, when a package directory
is not a package in `checks/layer_map.py`, or when the budget is stale. A test
removed from the unmoved set must lower the budget in the same PR; a move that
does not update the budget therefore fails. A `git mv` with no content edits
costs zero added lines. Git reporting a move as delete plus add does not
change accounting: validate both paths, count the root removal, and require
the approved destination and matching budget decrease.

### Touch it, move it

Recommendation, subject to the gate and user decision: encourage a PR that
edits an unmoved test to move it into its package folder in the same PR when
the move remains reviewable. This is not required. Moving while editing can
reduce future full-suite fallbacks, but combining a rename and behavioral
change can make review harder. Keep the test at root while it imports an
unmoved shared helper; move the helper with its owning package, then move the
test. In particular, move `tests/ops_support.py` to `tests/v2/ops/` and
`tests/data_scan_support.py` to `tests/v2/data/`. A test and its helper may
move together when the package ownership is clear.

Concurrent PRs can edit or move the same file and conflict on the old path,
new path, or imports. Coordinate by letting the first change merge, then rebase
and preserve both changes; the gate and user decide whether the combined move
is still reviewable.

### Ratchet check failure semantics (R1–R6)

| Case | Contract |
|---|---|
| **R1 — new test outside the layout** | Fail for a newly added test file outside `tests/v2/<package>/` or `tests/v2/integration/`; never silently add it to the unmoved set. |
| **R2 — unmoved set grows** | Fail if the root-level set grows. A new root test that replaces a different root test is also rejected by R1, even when the set size stays constant. |
| **R3 — stale or increased budget** | Fail unless the file's count equals the current root-level set and does not increase from base to head. Moving a root test requires the budget to shrink in that PR. |
| **R4 — unknown package** | Fail if a `tests/v2/<package>/` directory names no package in `checks/layer_map.py`. |
| **R5 — rename appears as delete plus add** | Apply the same path checks and budget accounting without relying on Git rename detection: root deletion, approved destination, and one lower budget count. A path-only `git mv` adds zero lines. |
| **R6 — revert restores a moved test** | Fail if a later PR restores that file at root or increases the budget. Move it forward again; the budget never grows. |

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

## Evidence and acceptance before code

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

The replay parsed **467 CI failure occurrences with zero misses** across the
13 PRs whose CI history is recoverable. **Twenty-seven PRs are inconclusive**
because their runs were cancelled or logs expired; that unknown history cannot
be recovered. The original all-40 acceptance bar was not met.

**Supervisor's judgment call, for the user to confirm at review:** with no
shadow mode, acceptance before code is zero misses on recoverable evidence,
with `main` running the full suite after every merge as the backstop and the
ratchet check itself as the guard against silent narrowing. This is a changed
acceptance bar, not a claim that the former bar was met.

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
| 1 | Selector module, selector tests, ratchet check and initial unmoved-set budget; no moves | 55 / 70 / 15, plus a one-line count file |
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

Start with the narrowest reverse closures: dashboard, research, diagnosis and
serving, then follow the listed schedule. Slices 2–22 remain scheduled work
and the opportunistic path described above. Contracts, foundation and data
have broad closures and may never need to move if package selection remains
broad; do not move them for layout alone. Slice 1 is estimated at 55
non-test code lines, 70 test lines and 15 documentation lines, plus the
one-line initial count file, within the small-PR limits. Every follow-up slice
updates this design or the architecture contract only when its behavior
differs.
