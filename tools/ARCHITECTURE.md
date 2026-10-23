# Tools

## Purpose

Standalone operator tools. `bounded_run.py` is the wrapper every heavy job runs under: CPU pinning, memory, swap and box-floor limits, test slots and the heavy reservation. Its module docstring is the flag reference; `engine/v2/ops/ARCHITECTURE.md` describes it as an executor adapter.

## Mutation snapshot-worker budget

The mutation driver limits concurrent synthetic snapshot-worker reservations, not whole-run RSS. It neither provisions a larger runner nor changes time budgets.

| Condition | Outcome |
|---|---|
| A caller runs the `data_incremental` mutation shard | At most one mutmut child runs, even when the requested or configured parallelism is higher. |
| A caller runs another mutation shard | Requested or configured child parallelism is unchanged. |
| The snapshot-worker fixture's imported data module has actual generated-code instrumentation | Only its `io_fetch` memory reservation rises to 1 GiB; every other resource-policy field is preserved. |
| The fixture uses authored source, including environment flags or paths that merely resemble instrumentation | Its existing 384-MiB allowance remains; shared test and production policies are unchanged. |
| Observed worker-family memory exceeds the selected reservation | The existing watchdog rejects it as `RESOURCE_LIMIT_EXCEEDED`; admission and enforcement are not bypassed. |

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
