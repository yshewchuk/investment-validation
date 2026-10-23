# Tools

## Purpose

Standalone operator tools. `bounded_run.py` is the wrapper every heavy job runs under: CPU pinning, memory, swap and box-floor limits, test slots and the heavy reservation. Its module docstring is the flag reference; `engine/v2/ops/ARCHITECTURE.md` describes it as an executor adapter.

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

## Dependencies

Callers are `tools/v2_resource_measurement.py`, `tools/land.py` and `tools/oc_check.py`; depends only on the stdlib and Linux `/proc`, `os.statvfs`.

## Mutation stats diagnostics (#581)

Mutation diagnostics are optional evidence, never authority over a backend result or score. The mutation driver consumes pytest's stats-phase failures (including captured setup/call/teardown and collection output, plus internal errors). It exposes at most 64 KiB per module as a fresh sidecar and at most 64 KiB on stderr after a failed backend. CI attempts to retain fresh evidence before result export; neither collection nor delivery reruns tests or enables per-mutant debug output.

| Condition | Outcome |
|---|---|
| No stats failure, or a mutant-phase report | No diagnostic report is added; ordinary backend behavior is unchanged. |
| Stale sidecar cannot be cleared | Evidence is disabled for this run with a best-effort warning; the backend still runs, old bytes are not echoed, and CI excludes files older than the run start. |
| Sidecar creation/write or its warning fails | Best-effort evidence is abandoned; pytest's original outcome remains authoritative. |
| Failure echo cannot be read/written, or CI cannot copy evidence | The backend exit remains unchanged and result export is still attempted. An export failure remains a failure, not a fabricated result. |
| Evidence exceeds its byte budget | The retained evidence is truncated, not the failure outcome. |

Authored-source import/size policies consume the uninstrumented work snapshot selected by the driver; behavioral tests still exercise instrumented imports. Production sources, selected tests and dependency allowances are unchanged. Operator details and artifact names are in [the testing guide](../tests/README.md).
