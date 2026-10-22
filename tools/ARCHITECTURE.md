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

## `mutation_pilot.py run`: leftover-process sweep (#1115)

`cmd_run` makes the pilot a Linux child subreaper (`prctl(PR_SET_CHILD_SUBREAPER)`) before launching mutmut, so a test worker orphaned when mutmut kills a timed-out mutant reparents to the pilot instead of init. After mutmut returns (normally, after the time-budget stop, or on an exception) `_reap_descendants` SIGKILLs every remaining descendant, so none keeps the CI step's stdout pipe open.

| Condition | Outcome |
|---|---|
| mutmut exits normally, by budget stop (124) or by exception | the sweep runs in `finally`; the mutmut exit code is returned unchanged and the exception still propagates. |
| subreaper unavailable (non-Linux or `prctl` fails) | `child subreaper: no` is printed; only descendants still attached to the process tree are found, orphans reparented to init are not. Off Linux the sweep is a silent no-op. |
| descendants found | one stderr line `killed N leftover descendant process(es)` plus at most 20 `pid=… cmd=…` lines (command cut to 200 characters, read from the first 4096 bytes of `/proc/<pid>/cmdline`); zombies are not counted. |
| a descendant forks during the sweep | the sweep rescans until no unsignalled live descendant remains or the 5 s deadline passes. |
| a process survives SIGKILL | logged as `did not exit after SIGKILL` and abandoned; the wait is bounded at about 5 s. |
| any error inside the sweep | logged as `descendant sweep failed`, returns 0, never raises and never changes the exit code. |

No mutant is excluded or skipped by the sweep.

## Dependencies

Callers are `tools/v2_resource_measurement.py`, `tools/land.py` and `tools/oc_check.py`; depends only on the stdlib and Linux `/proc`, `os.statvfs`.
