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
    stays quarantined rather than guessed dead).
    """
    table = process_table(boot_id, proc) if table is None else table
    blockers = []
    for pid, (identity, _, state, _, session) in table.items():
        if state == "Z":
            continue
        session_owns = (launch_pid is not None and session == launch_pid
                        and identity.start_ticks >= launch_start_ticks)
        if session_owns or _environ_contains(pid, marker, proc) is not False:
            blockers.append((pid, identity.start_ticks))
    return tuple(sorted(blockers))
