# Tools

## Purpose

Standalone operator tools. `bounded_run.py` is the wrapper every heavy job runs under: CPU pinning, memory, swap and box-floor limits, test slots and the heavy reservation. Its module docstring is the flag reference; `engine/v2/ops/ARCHITECTURE.md` describes it as an executor adapter. `bench_chain_index.py` is a synthetic-only benchmark of `price_structure` against the chain-index twin (`--tiny`, `--reps`, `--no-profile`); it reads no data and CI runs only a tiny smoke of it.

## `orats_smoothed_vol.py`: experiment-side smoothed vol

The production chain normalizer drops ORATS `smvVol`; this operator tool re-reads the raw cache into separate tables for fair-value fill experiments. No production path or `size_panel.py` closure file imports it; it writes no ledger row. Protected directories are the raw, curated and feature stores and `--raw-root`.

Output root: `--out`, default `data/experiment/orats_smoothed_vol/`. Each table is split into `year=YYYY/month=MM.parquet`, sorted by its unique key.

| Output | Source, key and values |
|---|---|
| `quotes/` | `hist/strikes`. Key `(ticker, obs_date, expiry, strike)`, built as the chain normalizer builds it from `tradeDate` and `expirDate`; join to `option_chains` on these four (both sides of a strike share the row). Values `smvVol`, `callMidIv`, `putMidIv`, `stockPrice`, as vendor-reported: sentinels become NaN, zeros stay zero. |
| `rates/` | `hist/cores`. Key `(ticker, obs_date)`. Values `iRate5wk`, `iRateLt` and the options-model inputs `borrow30`, `borrow2yr`, `annIdiv`, `annActDiv`, `divAmt`, `divDate`, `assetType` (vendor names; `borrow2yr` is ORATS's `borrow2y`). `divDate` is a date (only `YYYY-MM-DD` in range; anything else: null), the rest are numbers, with sentinels NaN and zeros kept. |
| `manifest.json` | Per-table entry, skip and row counts, `duplicates_dropped`, `conflicting_keys` (duplicated keys that differ: for `quotes/`, in the quote values; for `rates/`, in the two rates only), `conflicting_keys_all_columns` (differ in any written value) and `non_null` (per value field, over written rows). Top-level `units` names the unit of each `rates/` field. Written last. |

No unit is converted. `units` records `percent` for the rates and borrow fields (the `hist/cores` convention; ORATS's Summaries endpoint, not read here, uses decimals), `dollars_per_share` for `annIdiv`, `annActDiv` and `divAmt` (our reading, unconfirmed), `code` for `assetType` (0 ETB_NO_DIV, 1 HTB, 2 HTB_DIV_PAYING, 3 ETB_DIV_PAYING, 4 INDEX, 5 ETF) and `date` for `divDate`. The labels are fixed, not detected. Match coverage against `option_chains` is not reported yet. `dividend_fields` in `manifest.json` counts, for each raw `hist/cores` column whose name contains `div`, the non-null vendor values before any parsing, dedupe or skip rule, so it includes values the output writes as null or never writes; use `non_null` for written values. Presence does not establish meaning.

| Condition | Outcome |
|---|---|
| Several rows share a key | The latest `fetched_at` wins (a missing or unparseable stamp sorts oldest), then the greatest cache key, then the later body row. The winner is kept whole, even where its values are NaN. |
| A payload has no non-null `smvVol` (quotes) or no non-null rate (rates) | Skipped and counted, so it cannot override a pull that had them. The other `rates/` fields do not count: a payload with only those is skipped, and one without them can win a key, leaving them null. |
| A row has an unusable key | Dropped and counted. |
| A body is unreadable (over 256 MiB, either form) or its shape unrecognized | Counted and logged. Outputs and the manifest are still written; exit 1. Nothing is quarantined. |
| A run stops after preflight | No manifest exists. The next run reuses `--work-dir` (default under `/tmp`) entries only if body, parsing code, marker and shards are unchanged; others are reparsed. A shard that changes before its merge fails the run. |
| A run completes | Its shards and markers are removed; other files in `--work-dir` remain. A rerun rebuilds; the same cache and library versions give identical outputs. |
| A metadata sidecar exceeds 1 MiB | The run fails on the shared listing's `ValueError`, naming the file, before writing anything. |
| An output or work directory equals, contains or sits inside a protected one or each other; no `hist/strikes` entry | Refused with a message before writing anything. |
| `--work-dir` cannot be created | The run fails and any previous manifest remains. |
| Two runs share an output or work directory | Unsupported: there is no lock. |

Memory holds one body or one month. The shared listing (`iter_cached`) skips unparseable sidecars uncounted; a read error fails the run.

## Mutation snapshot-worker budget

The mutation driver limits concurrent synthetic snapshot-worker reservations, not whole-run RSS. It neither provisions a larger runner nor changes time budgets.

| Condition | Outcome |
|---|---|
| A caller runs the `data_incremental` mutation shard | At most one mutmut child runs, even when the requested or configured parallelism is higher. |
| A caller runs another mutation shard | Requested or configured child parallelism is unchanged. |
| The snapshot-worker fixture's imported `data.incremental` module has actual generated-code instrumentation | Only its `io_fetch` memory reservation rises to 1 GiB; every other resource-policy field is preserved. |
| The fixture uses authored source, including environment flags or paths that merely resemble instrumentation | Its existing 384-MiB allowance remains; shared test and production policies are unchanged. |
| Observed worker-family memory exceeds the selected reservation | The existing watchdog rejects it as `RESOURCE_LIMIT_EXCEEDED`; admission and enforcement are not bypassed. |
| A mutation time budget expires and the child exits zero within the SIGINT grace period | The driver returns `124`; incremental results are incomplete without a tool error, while full runs fail. Nonzero child exits during grace are preserved instead. |
| The SIGINT grace period expires and requires SIGKILL | The driver returns `137`; both modes fail. Export is attempted for readable partial state; corrupt state can fail export. |

## `bounded_run.py`: disk-used ceiling (#1119)

Opt-in guard against a heavy job filling the host disk (WSL's virtual disk never shrinks, so growth past its largest size takes host space). The ceiling is an absolute used-bytes figure, `--max-disk-used-gb G` or, when the flag is absent, the environment variable `BOUNDED_RUN_MAX_DISK_USED_GB`; no value is committed to the repository. Unset means off and nothing below applies. A value that is not a positive finite number is a usage error (exit 2).

| Condition | Outcome |
|---|---|
| what is sampled | `statvfs` used bytes `(f_blocks - f_bfree) * f_frsize` on the filesystem of bounded_run's working directory plus each `--disk-path` (repeatable, at most 8), de-duplicated by `st_dev`. One syscall per distinct filesystem per watchdog tick (`--poll-s`); no directory walk, so the cost is bounded by the path cap. |
| a sample exceeds the ceiling | the tree is killed through the memory-limit kill path (SIGTERM, then SIGKILL after `--kill-grace-s`); after SIGTERM the group is waited for up to 30 s (`SIGNAL_WAIT_S`) and then SIGKILLed, so no descendant outlives the guard; stdout carries a `DISK LIMIT` line naming the path, used GB and ceiling GB, and the exit code is 76 (`DISK_LIMIT_EXIT`), distinct from 137 (memory) and 75 (resource wait). |
| `statvfs` unreadable at startup | with `--heavy`: refuse to start, print `DISK LIMIT unreadable` and exit 69 (`EX_UNAVAILABLE`) before taking the heavy reservation. Without `--heavy`: print a warning and run with the disk check off. |
| `statvfs` unreadable mid-run | one tick is skipped and the skip is logged once per unreadable streak; with `--heavy`, 5 consecutive failed ticks kill the tree as a disk breach (fail closed); without `--heavy` the check stays off for that tick only. |
| started already above the ceiling | the first tick breaches and kills the job (exit 76). |
| heartbeat | the once-a-minute line adds `disk used <G>G/<ceiling>G` when the check is on. |
| resource-measurement record (`tools/v2_resource_measurement.py`, caller of bounded_run) | `killed` is true for exit 76 as well as 137; `kill_reason` is `DISK LIMIT` only for exit 76, when a terminal breach line (`DISK LIMIT:` or `DISK LIMIT unreadable for N ticks`) appears in the output, and `None` otherwise (an earlier `DISK LIMIT unreadable:` warning does not count); exit 137 reports the memory strings only (match priority BOX FLOOR, SWAP, CAP), never `DISK LIMIT`. |

## `bounded_run.py` dependencies

Callers are `tools/v2_resource_measurement.py`, `tools/land.py`, `tools/oc_check.py`, `checks/rearchitecture_phase2_render_parity.py` and the `ops/systemd/native-nightly-trigger.service` unit; depends only on the stdlib and Linux `/proc`, `os.statvfs`.

## `mutation_pilot.py run`: leftover-process sweep

`cmd_run` makes the pilot a Linux child subreaper (`prctl(PR_SET_CHILD_SUBREAPER)`) before launching mutmut, so a test worker orphaned when mutmut kills a timed-out mutant reparents to the pilot instead of init. After mutmut returns (normally, after the time-budget stop, or on an exception) `_reap_descendants` SIGKILLs every descendant it finds, so a leftover process normally cannot keep the CI step's stdout pipe open. This is best effort, not a guarantee: a process that survives SIGKILL, or whose parent was not reparented to the pilot, can still hold the pipe (see the table).

| Condition | Outcome |
|---|---|
| mutmut exits normally, by budget stop (124) or by exception | the sweep runs in `finally`; the mutmut exit code is returned unchanged and the exception still propagates. |
| subreaper unavailable (non-Linux or `prctl` fails) | `child subreaper: no` is printed; only descendants still attached to the process tree are found, orphans reparented to init are not. Off Linux the sweep is a silent no-op. |
| descendants found | one stderr line `killed N leftover descendant process(es)` plus a bounded number of `pid=… cmd=…` lines with a truncated command; if more were killed than lines shown, a final `... and N more` line follows; zombies are not counted. |
| a descendant forks during the sweep | the sweep rescans until no unsignalled live descendant remains or a bounded deadline passes. |
| a process survives SIGKILL | logged as `did not exit after SIGKILL` and abandoned; the wait is bounded. |
| any error inside the sweep | logged as `descendant sweep failed`, returns 0, never raises and never changes the exit code. |

No mutant is excluded or skipped by the sweep.

## Mutation stats diagnostics

Mutation diagnostics are optional evidence, never authority over a backend result or score. The mutation workflow calls `mutation_pilot.py`, which configures pytest to load `mutation_stats_diagnostics.py`. That plugin consumes pytest's stats-phase reports (including captured setup/call/teardown and collection output, plus internal errors) and writes at most 64 KiB per module to a sidecar. The driver clears stale evidence before launch and consumes the sidecar afterward, exposing at most 64 KiB on stderr after a failed backend. CI attempts to retain fresh evidence before result export; neither collection nor delivery reruns tests or enables per-mutant debug output.

| Condition | Outcome |
|---|---|
| No stats failure, or a mutant-phase report | No diagnostic report is added; ordinary backend behavior is unchanged. |
| Stale sidecar cannot be cleared | Evidence is disabled for this run with a best-effort warning; the backend still runs, old bytes are not echoed, and CI excludes files older than the run start. |
| Sidecar creation/write or its warning fails | Best-effort evidence is abandoned; pytest's original outcome remains authoritative. |
| Failure echo cannot be read/written, or CI cannot copy evidence | The backend exit remains unchanged and result export is still attempted. An export failure remains a failure, not a fabricated result. |
| Evidence exceeds its byte budget | The retained evidence is truncated, not the failure outcome. |

Authored-source import/size policies consume the uninstrumented work snapshot selected by the driver; behavioral tests still exercise instrumented imports. Operator details and artifact names are in [the testing guide](../tests/README.md).
