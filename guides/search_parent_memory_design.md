# Search parent releases its data before dispatch (design)

Status: design only. No code in this PR. Scope rules: no edit to SIZE search modules or to the `experiments/size_panel.py` source closure in this PR; no heavy job; nothing here touches the ledger.

Standard tool check: nothing off the shelf covers this. The levers are plain Python/glibc (`gc.collect`, `malloc_trim`) and the repo's own spawn pool, so the design reuses the existing `ProcessPoolExecutor` spawn path and `experiments/memory_guard.py`; it adds no library.

## 1. Problem and what the diagnosis could and could not establish

User report (2026-11-18): in a SIZE full-universe search the parent process sits at about 7-8 GiB for the whole search, which forces `workers=1` on a 12 GiB box. This section is by code reading only. **I could not measure**, so sizes below are unknown; the table lists what is held, by whom, and when. Two things the brief assumed turned out different and the design follows the code:

1. **Per-trial inputs are not published before dispatch.** The first trial that needs a cohort admits it (`SearchTrials._cohort_inputs` -> `admit_fold_inputs` -> `size_search_binding.fold_inputs` -> `read_stored_panel`) and publishes it (`_start`: `publish_inputs`, then the `cohort:` and `inputs:` records). With `workers>1` that is the worker (`_init_trial_worker` builds its own `SearchTrials`; `_published` is per instance, so a worker re-admits a cohort a sibling already published). With `workers=1` (in-process serial) it is the **parent**. So the parent only holds a panel because the serial path makes it do the worker's job.
2. **In a fresh parallel run the parent does not materialize inner inputs** until the outer refit stage. On a **resume** it does earlier: each claimed trial runs `trials.reconcile` (`_result_evidence` -> `self._cohort(admitted)` -> uncached `load_inputs`, one cohort at a time), and each `completed_outer` replay runs `execute_outer_fold`, both before the pool starts and in both modes; a pool of one does not remove them (release point (a) below covers them). So if the parent is 7-8 GiB with `workers>1`, the cause is something other than the panel memo (see H3, H4 and the measurement note), and `workers=1` being the only workable setting is consistent with the serial path, where the parent is the worker.

### 1.1 Holders in the parent (trace)

| Id | Object | Where | Lifetime today | Needed by |
|---|---|---|---|---|
| H1 | Process-global `_PANEL_MEMO`: one admitted panel (`context`, `rows`, feature snapshot) | `size_search_binding.py` only; IV's `regression_search_binding.py` imports `read_stored_panel` from it, so both share this one memo | Until process exit (replaced only by another key). `read_stored_panel` returns `copy.deepcopy(panel)`, so a read peaks at about 2x the panel, then 1x stays resident | `fold_inputs` for each fresh cohort in the serial path, each refit, and `size_comparison_run.execute` |
| H2 | `SearchTrials._memo` (one cohort's admitted inputs) and `_published`; `_start` returns a deep copy (`detach=True`) | `magnitude_search_trials.py` | Instance lifetime: whole serial loop; a new instance per refit unit | The trial about to run. The fit child streams shards from the store and never needs the parent's copy |
| H3 | Hydrated registration document: v1.7 hydrates plan events, panel target events, metadata and fragment lists from CAS shards into one in-memory `registration.document` | `native_registration.load_registration_document` | Whole process; every worker hydrates its own copy too (`_init_trial_worker`) | Candidate execution, limits and fold lookups. Most of it is not needed after plan admission |
| H4 | Full plan rebuild on every admission: `read_search_plan` -> `_prepare` rebuilds the complete plan and compares it to the stored one; `SearchTrials._admit` calls it per trial and `execute_outer_fold` per unit | `magnitude_search_plan.py` | Transient per call, but repeated thousands of times; large transient dicts fragment the heap | Identity check; the result is the same every call |
| H5 | `freeze()` working set: per-cohort history rows (`history_rows_cache`), 32 MiB terminal cache, all selections | `magnitude_search_trials._freeze` | Duration of freeze. **`execute_outer_fold` calls `_native_selection` -> `trials.freeze()` once per refit unit**, so freeze runs again for every refit | Selection check against the stored `selection` record |
| H6 | Refit inputs: `_execute_admitted` keeps the full fold `inputs` until `_replay` returns | `magnitude_search_outer.py` | One unit | Evidence checks; the child streams shards |
| H7 | `registration`, `repository` (pinned fragment lists only, not data), `conn`, `request` | driver | Whole process | Cheap; keep |

Not holders: the fit child (streams manifests, own RSS watchdog), `collect_outer_results` results (small relative to the above).

### 1.2 RSS is not live data

`size_search` prints `ru_maxrss`, the **high-water mark**. A 7-8 GiB figure can be a transient peak (panel read plus its deepcopy, plan rebuild, hydration) that dropping references will not return to the OS: CPython arenas and glibc heap fragmentation keep freed pages. The parent's memory watchdog samples **current** VmRSS (`memory_guard.rss_bytes`), so that is the number that must fall. Consequence: *reference release alone is not trusted*. Heavy panel work moves to a process that exits (section 3.2), and reference release (`del`, `gc.collect()`, `malloc_trim(0)`) is the cheap supplement, verified by measurement before any claim.

## 2. What each phase actually needs

| Phase | Needs the panel / history? | Needs |
|---|---|---|
| seal | No (plan binding only) | registration, plan |
| admission (`_admit_driver`, roster, claims) | No | registration compact fields, plan, journal reads |
| materialize + publish cohorts | **Yes**, one cohort at a time | panel, registration events |
| trial dispatch/fit | No in the parent; the worker needs one cohort's manifest | journal, store |
| freeze | One cohort's stored rows at a time (from the store) | terminals, stored cohorts |
| outer refit | **Yes** (fold inputs, comparison) | panel, registration |
| collect | No | terminal/outer records |

## 3. Design

### 3.1 Principles

- **Identity is untouched.** No change to trial, candidate, plan, cohort or inputs ids, to journal slot names, or to forecasts (bit-identical). Every slice carries a before/after equality test on the journal and the collected report hash for a fixture search.
- **Fail closed, never retry.** No new retry; claims already written stay written.
- **One shared layer.** All changes land in `v2_magnitude_search_run.py`, `magnitude_search_trials.py`, `magnitude_search_outer.py` and the two bindings' memo accessors; SIZE and IV both reach them through `size_search._run` -> `run_search`.

### 3.2 Release points

1. **Process isolation for the serial path (proposed main change; the default is open decision 1).** With containment on, `workers=1` runs through the same one-worker spawn pool as `workers>1` (a pool of one), so the parent never materializes inputs. The worker already runs inside its 4 GiB child cap and per-process RSS watchdog, a path exercised for `workers>1`. The parent drops from "panel + cohort" to "registration + plan + records". Library callers and tests keep today's in-process serial path (`run_search(..., isolate=False)` default; the SIZE and IV CLIs pass `True`), because in-process fakes cannot cross `spawn`.
2. **Reference release with measurement** for what the parent still holds. A single helper `release_parent_caches()` clears the shared `_PANEL_MEMO` (one memo serves SIZE and IV), `SearchTrials._memo`/`_published`, runs `gc.collect()` and `ctypes` `malloc_trim(0)` when available (an absent libc symbol is a logged no-op, never an error). Called at: (a) after the claimed-trial and completed-outer replays (which load a cohort or panel in the parent on resume) and before the pool is created, (b) after pool shutdown and before `freeze`, (c) at the end of the refit stage (not between refit units: the memo exists to share one panel across units, and releasing per unit would re-read it per unit), (d) before `collect`.
3. **Refits stay in the parent** for now. They are sequential and run after the pool is gone, so the parent holds one panel at a time and releases at (c). If the smoke shows the refit stage still exceeding the box, refits move to the one-worker pool as a follow-up slice.
4. **Cheap fixes to H4/H5 are separate and optional** (section 6): memoize the plan rebuild per `(plan_ref, request)` within one `run_search` call, and share one verified `freeze` across the refit units of one stage. Both are speed-ups that weaken a repeated independent re-verification, so each needs the user's call (open decision 3).

### 3.3 How references are dropped

`release_parent_caches()` only clears module and instance containers (`_PANEL_MEMO.clear()`; `trials._memo.clear()`; `trials._published.clear()`), then `gc.collect()`. Callers must not hold their own handle: a test that, after release, no live object of the panel row container remains (`gc.get_objects` filter) catches accidental retention such as a closure over `inputs`. No semantic cache is lost: `_PANEL_MEMO` and `_memo` are re-derivable from the store and sources and are never persisted.

### 3.4 Memory budget semantics

- The request's `limits.parent.memory_bytes` stays the parent's RSS watchdog cap, and the preflight rule `workers * child <= parent` stays as is. With a small real parent the rule is conservative: 3 workers x 4 GiB needs a 12 GiB ceiling in the request even though the parent uses 1-2 GiB. Changing the rule would loosen admission or change request identity, so this design does not; operators set a high parent ceiling for the next request (open decision 2).
- Preflight adds a `decision()` log line with host `MemAvailable` and the projected family size (`parent_target + workers*child`). It refuses nothing new; host admission stays with `bounded_run` (per-process caps, not host reservations, as the memory ceiling contract says).
- The parent logs VmRSS and `ru_maxrss` at stage boundaries (progress-logging convention, bounded cadence).

### 3.5 Failure semantics (R1-R6)

- **R1 Triggers.** Worker: sampled VmRSS above the child cap (unchanged) or an OOM kill. Parent: sampled VmRSS above `parent.memory_bytes`, or the box OOM killer. Release: any exception in `gc` or `malloc_trim`.
- **R2 Behaviour.** Worker watchdog: `RESOURCE_LIMIT_EXCEEDED` receipt, unfinished claim kept, no freeze (existing). Worker killed: `BrokenProcessPool` -> `_abort_executor` -> `RESOURCE_UNAVAILABLE` (existing). With a pool of one for the serial path the same two outcomes cover serial trials. Parent watchdog: exit 2 with `RESOURCE_LIMIT_EXCEEDED`, pool aborted by the existing `except BaseException` path.
- **R3 Fail closed.** `malloc_trim` unavailable, `gc` failure or a retained-reference check failing: release logs the reason and continues; no refusal, no identity effect. A pool of one that cannot start refuses before any claim and never falls back to in-process silently.
- **R4 Overshoot.** Watchdog sampling is 50 ms, so one interval can exceed the cap; unchanged. More workers raise the chance that the box OOM killer, not a watchdog, ends the run; the smoke measures the family peak and the decision log records `MemAvailable`.
- **R5 Mid-search restart.** Nothing new is durable: no new record kind, no ordering change. A restart after a parent or worker death hits the existing rule: a `start:` without an `execution-result:` refuses `EXPERIMENT_IDENTITY_CONFLICT` (claimed trials cannot restart), so only a run killed mid-trial needs a fresh run id; trials with a recorded result reconcile and resume as today. A smaller parent lowers parent-OOM risk; it does not make a killed trial resumable.
- **R6 Scope.** Per process, no host reservation, no per-sample logging.

### 3.6 Source-closure ordering constraint (important)

SIZE registrations bind `source_closure` of `experiments/size_search.py`, and `verify_installation` refuses a binding whose closure differs from the installed tree; the driver seal binds the closure of `v2_magnitude_search_run.py`, and journal records bind the trials/outer closures. So **any code slice changes the closure and makes an existing SIZE run (including the in-flight v14 run, its `report`, and resumes) refuse `EXPERIMENT_IDENTITY_CONFLICT` on the merged tree.** Rule: no code slice merges until v14 has finished and its report is collected, or the v14 report is run from a pinned checkout; the next SIZE run re-preflights and re-registers anyway. The `experiments/size_panel.py` closure is not edited: the release helper lives in the two binding modules and the driver, which import `size_panel` but are not in its closure, and a test asserts its closure hash is unchanged. The bindings are in the `size_search.py` closure; that is the ordering constraint above, not a violation of the size_panel rule.

### 3.7 Contract text (moves into `experiments/ARCHITECTURE.md` with the first code slice)

`experiments/ARCHITECTURE.md` is about 23,500 words against a 10,500-word ceiling (`checks/architecture_doc_budgets.py`); a doc over the ceiling may only shrink, so it cannot grow in this PR. The contract lives here until a PR that nets a shrink of that file (a compression, or a slice whose text replaces more words than it adds) carries it. Each code slice that changes failure semantics must update the contract in the same PR, so P2 or P3 pairs with that shrink. Proposed contract (items marked proposed depend on open decision 1) for the shared search layer (`run_search`, `SearchTrials`, `execute_outer_fold`, the shared panel memo), covering SIZE and IV:

- Identity, journal slot names and forecasts are unchanged by release.
- The parent drops its panel memo and cohort memo before the pool starts, after it shuts down, at the end of the refit stage and before `collect`; proposed (decision 1): contained CLIs run `workers=1` as a pool of one; library callers keep in-process serial.
- `limits.parent.memory_bytes` stays the parent watchdog cap and `workers * child <= parent` is unchanged; preflight logs `MemAvailable` and refuses nothing new.
- A failed release logs and continues; a pool of one that cannot start refuses before any claim (missing CPU affinity `RESOURCE_UNAVAILABLE`; worker/memory bound violations `INVALID_EXPERIMENT_SPEC`, as `_parallel_preflight` does today; a spawn failure is a new preflight check and refuses `RESOURCE_UNAVAILABLE`); worker/parent exhaustion, OOM kill and restart keep existing outcomes (claim retained, no freeze, no retry).
- Any code slice changes the closures SIZE registrations and driver seals bind; slices merge only after the in-flight run is reported.

## 4. Other drivers

| Driver | Same pattern? | Disposition |
|---|---|---|
| SIZE (`size_search*`) | Yes (H1-H6) | This design |
| IV crush (`iv_crush_run`, `iv_crush_search_report`, `regression_search_binding`) | Yes: same `_run` -> `run_search`; shares SIZE's single `_PANEL_MEMO` through `read_stored_panel` | Same slices (one memo clear covers both) |
| `v2_magnitude_search_run` CLI | Same loop; refuses qualified SIZE/IV, so legacy magnitude only | Covered |
| `distribution_*` (cli, run, trials, worker) | Imports only `_read_document`/`_render` from the search run; own bounded subprocess child, no panel memo, no `run_search` | Not the same pattern; parent holdings not audited in depth. Issue to audit once P1 data exists |
| EXP-188 (`size_pnl_*`, `EXP-188_.../run.py`) | No: consumes stored v14 predictions, never calls `run_search`/`SearchTrials`; its design (`guides/size_pnl_synthetic_calibration_design.md`) already runs one heavy job, alone, after v14 | No change |

## 5. Measurement plan (only after the v14 run finishes; one heavy job at a time; `--no-ledger`)

50-ticker smoke through the contained SIZE CLI (`python -m experiments.size_search ... --no-ledger`), same request, fresh run id, `bounded_run` with the usual cpu-set, before (current `main`) and after each slice:

1. Parent **current** VmRSS and `ru_maxrss` at: start, after registration, after admission/roster, after claim replays, before pool, mid-dispatch, after pool shutdown, after `freeze`, per refit unit, after collect.
2. Peak family RSS (parent + workers) from `/proc` sampling at 1 Hz.
3. Wall time per stage and total, for `workers=1,2,3`.
4. Journal and report equality before and after for the same inputs.
5. A holder breakdown at each boundary (sizes of `_PANEL_MEMO`, `_memo`, hydrated registration, plan) with a bounded deep-size helper, so "parent is X GiB because Y" is evidence.

Pass: parent current VmRSS after pool creation <= 2 GiB with `workers>=2` on the smoke, scaling checked against panel-size ratio before extrapolating to the full universe; identical journal and report; `workers=3` wall time below `workers=1`. Failing the first criterion means H3/H4 dominate and P4 moves up.

## 6. Split into code PRs (each merges only after the v14 report is collected; estimates are code / test / doc words, code and test in added lines)

- **P1 Measure.** Stage-boundary VmRSS/`ru_maxrss` and holder-size logging in `run_search`, refit and collect. No behaviour change. Produces the baseline. About 100 / 80 / 0.
- **P2 Release helper.** `release_parent_caches()` plus a clear of the shared memo, called at the four points of 3.2, including before claimed-trial replay; retention and identity tests. About 90 / 150 / the contract text (net shrink, 3.7).
- **P3 Isolated serial path (needs decision 1).** `run_search(isolate=...)`; CLIs pass `True`; pool of one; parity test (isolated vs in-process on a tiny fixture: identical journal). About 120 / 180 / contract delta.
- **P4 (conditional on smoke) Hydration and plan admission.** Compact registration for the parent after plan admission; memoize plan rebuild per call. Only if H3/H4 measure large. About 150 / 150 / 0.
- **P5 (conditional) Refits through the pool** if the refit stage still exceeds the box after P2. About 150 / 150 / 0.
- **Issues (not slices):** share one verified `freeze` across refit units; audit the distribution drivers' parent holdings after P1.

## 7. Open decisions for the user

1. **Isolate by default for the contained CLIs?** `workers=1` becomes a pool of one (CLIs only). Cost: one spawn per run, and a killed worker kills the run. Alternative: keep in-process serial and rely on reference release, which section 1.2 says cannot be trusted to return RSS.
2. **Meaning of `parent.memory_bytes`.** Keep it as the parent watchdog cap and have operators set it at or above `workers * child` (no identity change), or add a separate family ceiling in a new request version (identity change). Recommendation: keep.
3. **Verification versus speed for H4/H5.** Whether to memoize the plan rebuild and share one `freeze` across refit units, each dropping a repeated independent re-verification. Recommendation: defer to issues unless the smoke shows them dominant.
4. **Merge ordering.** Confirm no code slice merges until v14 and its report are done (3.6), accepting that v14 cannot be resumed or re-reported on the merged tree.
