"""Track kernel identities, including children which change process group.

Unobserved daemonization cannot be proved safe by polling. Unknown launch and
recovery reservations stay quarantined instead of being guessed dead.
"""
from __future__ import annotations

import os
import signal
from pathlib import Path

from engine.v2.contracts import ProcessIdentity


def process_info(pid: int, boot_id: str, proc: Path = Path("/proc")) -> tuple:
    """``(identity, ppid, state, rss_bytes, session)`` for one live pid.

    ``session`` (stat field 6, index 3 after the comm field) is kept outside
    the ``ProcessIdentity`` contract: it is a B1 ownership-proof input, not
    part of the kernel identity itself.
    """
    raw = (proc / str(pid) / "stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    identity = ProcessIdentity(boot_id=boot_id, pid=pid, start_ticks=int(fields[19]),
                               process_group=int(fields[2]))
    return (identity, int(fields[1]), fields[0], int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
            int(fields[3]))


def process_table(boot_id: str, proc: Path = Path("/proc")) -> dict:
    result = {}
    for path in proc.iterdir():
        if path.name.isdecimal():
            try:
                row = process_info(int(path.name), boot_id, proc)
                result[row[0].pid] = row
            except (OSError, ValueError, IndexError):
                continue
    return result


def observe(identities: tuple[ProcessIdentity, ...], boot_id: str, *, table=None) -> tuple:
    table = process_table(boot_id) if table is None else table
    known = {(p.pid, p.start_ticks): p for p in identities if p.boot_id == boot_id}
    live = {pid for (pid, ticks) in known if pid in table and table[pid][0].start_ticks == ticks}
    while True:
        children = {pid for pid, row in table.items() if row[1] in live}
        if children <= live:
            break
        live |= children
    for pid in live:
        identity = table[pid][0]
        known[(pid, identity.start_ticks)] = identity
    alive = [table[pid] for pid in live if table[pid][2] != "Z"]
    return tuple(known.values()), tuple(row[0] for row in alive), sum(row[3] for row in alive)


def is_alive(identity: ProcessIdentity, boot_id: str, proc: Path = Path("/proc")) -> bool:
    """True iff this exact kernel identity (never a pid-reuse impostor) is live."""
    try:
        current, _, state, _, _ = process_info(identity.pid, boot_id, proc)
    except (ProcessLookupError, FileNotFoundError):
        return False
    return state != "Z" and (current.boot_id, current.pid, current.start_ticks) == (
        identity.boot_id, identity.pid, identity.start_ticks)


def signal_owned(identities: tuple[ProcessIdentity, ...], boot_id: str, *, hard=False) -> None:
    for identity in reversed(identities):
        try:
            if is_alive(identity, boot_id):
                os.kill(identity.pid, signal.SIGKILL if hard else signal.SIGTERM)
        except (ProcessLookupError, FileNotFoundError):
            continue


def _environ_contains(pid: int, marker: str, proc: Path) -> bool | None:
    """True/False when the environ is readable; ``None`` when it cannot be read at all."""
    try:
        data = (proc / str(pid) / "environ").read_bytes()
    except OSError:
        return None
    return marker.encode() in data


def _owner_uid(pid: int, proc: Path) -> int | None:
    """The pid's owning uid, or ``None`` when even that cannot be read."""
    try:
        return (proc / str(pid)).stat().st_uid
    except OSError:
        return None


def _ancestor_pids(pid: int, table: dict) -> set[int]:
    """Every live pid on this process's own ancestor chain, traced through
    ``table``'s ppid links (including this process itself).

    A live process we are descended from can never be the escaper of an
    attempt we are now checking: our own lineage predates any attempt we
    launched. It is also unreadable for a structural reason, not a security
    one: Yama's restricted ptrace mode (``ptrace_scope=1``, the kernel
    default on any real, non-root host including a GitHub Actions runner)
    only lets a process trace its own descendants, never an ancestor, so a
    same-uid ancestor's ``/proc/<pid>/environ`` is permanently unreadable to
    us regardless of the same-uid rule above. Without this exclusion,
    find_owners() always finds at least one "blocker" -- the CI runner's own
    long-lived same-uid ancestors (its job shell, the actions worker, ...) --
    and recovery can never prove an attempt dead on a real runner.
    """
    seen: set[int] = set()
    current = pid
    while current in table and current not in seen:
        seen.add(current)
        current = table[current][1]  # ppid
    return seen


def _descendant_pids(roots: set[int], table: dict) -> set[int]:
    """Every pid transitively descended from any of ``roots`` (each root
    included), traced forward through ``table``'s ``ppid`` links.

    ``ppid`` comes from ``/proc/<pid>/stat``, which needs no special
    permission to read (unlike ``environ``), so this walk is exact and cheap
    regardless of privilege or Yama's ptrace restriction.
    """
    live = set(roots)
    while True:
        children = {pid for pid, row in table.items() if row[1] in live}
        if children <= live:
            return live
        live |= children


def find_owners(boot_id: str, *, launch_pid: int | None, launch_start_ticks: int | None,
                marker: str, table: dict | None = None, proc: Path = Path("/proc")) -> tuple:
    """Live, non-zombie processes that ownership proof (b)/(c) cannot exclude.

    (b): a live session still carrying ``launch_pid``, started at or after the
    launch identity's own start — this is a normal, un-``setsid``'d
    descendant. (c): a live process whose ``/proc/<pid>/environ`` carries
    ``marker`` (the attempt id) — this catches a ``setsid()`` escaper, whose
    session no longer matches (b), and the no-recorded-identity crash case,
    where there is no ``launch_pid`` to compare against at all. An environ
    that cannot be read never excludes its process (§ B1 house rule: unknown
    stays quarantined rather than guessed dead) -- UNLESS the process belongs
    to a different uid than this one: a fork/exec descendant of our own
    launch always keeps our uid (barring a setuid binary, which the legacy
    worker tree never runs), so a foreign-uid process categorically cannot be
    our escaper regardless of whether its environ happens to be readable.

    A live ancestor of the process running this check is excluded the same way: ptrace's
    restricted mode denies reading an ancestor's environ regardless of uid, and our own lineage
    predates any attempt we are now checking, so it can never be that attempt's escaper.

    None of this is enough on a real, busy, non-root host: Yama's restricted
    ptrace mode (``ptrace_scope=1``, Ubuntu's default, including on a GitHub
    Actions runner) only lets a process read the environ of its own
    descendants — never an unrelated process, even one at the very same uid.
    On a shared runner there are always other same-uid processes that are
    neither our ancestor nor any part of this attempt's lineage: other
    pytest-xdist workers' own child processes during a parallel test run, or
    the runner's other same-uid helpers. § B1's "unknown stays quarantined"
    rule, applied indiscriminately to every same-uid pid on the box, made
    every one of those permanently unreadable-and-unexcluded, so recovery
    could never prove an attempt dead outside a quiet, single-process host.
    (c) is therefore scoped to processes that could structurally BE this
    attempt's escaper: live descendants of the process performing this check
    (a setsid()'d child of ours we spawned and are still tracing) or of
    ``launch_pid`` (a setsid()'d descendant of the recorded launch, traced by
    ``ppid`` rather than session, which survives the setsid() call even
    though the session check (b) does not). A pid outside both subtrees
    cannot be a fork/exec descendant of either, so it cannot be our escaper
    regardless of whether its environ happens to be readable — this is what
    lets recovery prove "dead" on a shared, non-root CI runner.

    This does leave one case genuinely unprovable without root: a launch that
    crashed before its identity was ever recorded (``launch_pid`` is
    ``None``) AND whose escaper is not a descendant of the process doing the
    check either (for example, after a supervisor restart, checked from a
    brand-new process that never spawned it). There is no structural link to
    it at all, and no non-root way to read an arbitrary unrelated process's
    environ to look for the marker directly. That attempt stays quarantined
    forever rather than being guessed dead — a documented refusal, not a
    silent wrong answer; only a root check (which bypasses ptrace_scope
    entirely) can settle it, since ``ops reconcile`` runs this identical
    proof and has no force flag.
    """
    table = process_table(boot_id, proc) if table is None else table
    self_uid = os.getuid()
    ancestors = _ancestor_pids(os.getpid(), table)
    candidates = _descendant_pids(
        {os.getpid()} | ({launch_pid} if launch_pid is not None else set()), table)
    blockers = []
    for pid, (identity, _, state, _, session) in table.items():
        if state == "Z":
            continue
        if pid in ancestors:
            continue
        session_owns = (launch_pid is not None and session == launch_pid
                        and identity.start_ticks >= launch_start_ticks)
        if session_owns:
            blockers.append((pid, identity.start_ticks))
            continue
        if pid not in candidates:
            continue
        owner_uid = _owner_uid(pid, proc)
        if owner_uid is not None and owner_uid != self_uid:
            continue
        if _environ_contains(pid, marker, proc) is not False:
            blockers.append((pid, identity.start_ticks))
    return tuple(sorted(blockers))
