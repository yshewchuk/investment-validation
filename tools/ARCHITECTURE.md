# Tools

## Purpose

Standalone operator tools. `bounded_run.py` is the wrapper every heavy job runs under: CPU pinning, memory, swap and box-floor limits, test slots and the heavy reservation. Its module docstring is the flag reference; `engine/v2/ops/ARCHITECTURE.md` describes it as an executor adapter.

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
