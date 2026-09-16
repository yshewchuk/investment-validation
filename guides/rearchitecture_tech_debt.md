# Rearchitecture tech debt

Deferral rule, aligned 2026-09-16: work required by a migration phase is
finished in that phase. Optional optimizations here are owned by
[Phase 8C](rearchitecture_phase8_post_cutover.md), after Phase 7 cutover.
Nothing here may be a hidden prerequisite of Phases 3–7. If measurements make
an item necessary earlier, move it into that phase and update the
[delivery plan](rearchitecture_delivery_plan.md). Historical Phase 7 live
references in entries below now mean Phase 8D; after-Phase-6 optimization
references mean Phase 8C, not an extra pre-cutover gate.

Each entry: what was deferred, why it can wait, the risk while it waits, and
what should trigger revisiting it.

| ID | Item | Origin | Why deferrable | Risk while deferred | Revisit when |
|---|---|---|---|---|---|
| TD-1 | Process-local object-verification cache keyed by `(device, inode, size, mtime_ns)` (Phase 2 guide §8.2 step 6). | P2-4 planning | The repository re-hashes every object it opens. Correctness is unchanged; only speed is lost. | Slower scans of large `option_chains` fragments. | A read path measurably exceeds its time budget, or live latency matters (Phase 7). |
| TD-2 | `TimeInterval` start < end compares canonical strings. | P2-1a | Correct for dates, naive timestamps, and UTC timestamps produced by `format_timestamp`. | A non-UTC offset written by hand would compare wrongly. | Any contract accepts caller-supplied offsets other than UTC. |
| TD-3 | `artifact_check`'s `CheckParameters` carries an optional `input_bindings` field used only by the cache-identity tests. | P2-5 B1a | A test seam on a production kind; no production caller sets it. | A production caller could start depending on it. | A non-legacy production kind exists that those tests can use instead. |
| TD-4 | Code shaped to fit the eight-module import fan-out budget. `engine/v2/data/legacy_adapter.py` builds contracts in two passes and hard-codes three legacy paths (tier-0 tested against `engine.paths`). `engine/v2/data/objects.py` uses `os`-only equivalents of `stat`/`uuid`/`math`. `objects.py` and `documents.py` import sibling modules as `from . import errors, time_formats`, which the budget counts as one import. | P2-1b, P2-2b | Behaviour is correct and tested; the cost is readability. | Future edits keep contorting around the budget. | The fan-out budget is revisited, or these modules are split. |
| TD-7 | Two coverage ratchets both measure `engine.v2.data`. The Phase 1 suite (`phase1_coverage_suite.v2`) includes `tests/test_v2_data_*.py` as an interim fix, and the Phase 2 script measures the package too. | P2-5 B1c, task 8 | Both baselines only rise, so nothing is hidden. | Every data change needs two baseline refreshes. | The Phase 2 gate closes: drop the data tests from the Phase 1 suite. |
| TD-8 | `export_generation(conn, root, *, generation, purposes=None)` exports every purpose by default. The nightly export stage passes `legacy_import` and `shadow`. | P2-5 task 5 | The only production caller filters correctly. | A new caller that forgets the filter mixes `research_reconstruction` rows into legacy ledger files. | A second caller appears; then make `purposes` required. |
| TD-9 | `tests/ops_support.py::TEST_POLICY` copies CPU and thread counts from `DEFAULT_POLICY`, because environment refs are derived from `DEFAULT_POLICY`. | P2-5 test fix | Tests stay memory-independent and env refs still match. | A production CPU change silently changes test env refs too. | Env refs become derived from the injected policy. |
| TD-10 | Push key/time filtering into the Arrow scanner instead of evaluating predicates over each projected bounded batch in Python. | Sep-13 review of `97e2a5c`; explicit Phase 2 §8.2/§12.3 scope disposition | Batch-local filtering can preserve exact results, finite limits, and the public Arrow batch interface. Hidden predicate columns and materialization bounds are mandatory Phase 2 fixes, not part of this debt. | More CPU and decoded rows than a selective Arrow scan; no permission to exceed the admitted memory/time budget. | After Phase 6 when profiling shows predicate evaluation dominates, or Phase 7 latency needs it. If an earlier phase cannot meet its measured budget, move this item into that phase. |
| TD-11 | Avoid constructing a full legacy Scorer solely to supply render inputs; consume verified minimal panel/trade/registry inputs through a reviewed adapter or the later native serving projection. | Sep-13 review of `97e2a5c`, `_action_render` | The existing path is acceptable once Phase 2 measures and admits its actual peak and proves render parity. Native scoring/render separation is already scheduled in Phase 4; this entry covers only additional optimization if the compatibility path remains. | Extra startup, memory, and I/O per bundle; the old 2 GiB reservation is not evidence of safety. | After Phase 6 if this path remains and profiling justifies it. Close as superseded if Phase 4 removes it; move it into an earlier phase if resource limits make it necessary. |
| TD-12 | `legacy_selfcheck` re-scores against a bound `score.json` without checking that its producing job's `implementation_ref` matches the selfcheck worker's code snapshot. | Sep-15 Tier-4 determinism investigation (`74d70bb`) | In the nightly DAG, score and selfcheck bind artifacts from the same submitted plan and frozen snapshot, and job identity includes the implementation ref, so a cross-code pairing can't arise through the supervisor. | Manual or diagnostic selfcheck runs against an older score artifact report spurious forecast mismatches (8 mismatches seen on attempt-14 outputs), which looks like nondeterminism. | A path that reuses score artifacts across code versions appears; then refuse with a stale-input finding. |

## Tracked verification items (not debt)

The scheduled Tier-4 cache-miss, render-memory, chooser replay, and fresh
Phase 0/1/2 evidence work now lives in
[Phase 2 §12.2, P2-C01/P2-C02](rearchitecture_phase2_data_access.md#122-sep-13-review-closeout-fixes-owned-by-phase-2).
It is required closeout work, not debt. Resource-heavy/private checks run
sequentially under the existing authorization and resource policy; old
code-bound receipts cannot establish parity after implementation changes.

The rest of the Sep-13 review is assigned in
[Phase 2 §12.3](rearchitecture_phase2_data_access.md#123-explicit-later-owners),
[Phase 3 launch §5.5](rearchitecture_phase3_parity_launch.md#55-review-follow-through-for-repeatable-updates-and-live-health),
and [system design §12.1](system_rearchitecture.md#121-sep-13-review-follow-through).
Those phase requirements must not be treated as optional entries on this list.
The delivery plan and current phase guides supersede historical phase labels
in origin notes; they do not change the technical reason each item was deferred.
