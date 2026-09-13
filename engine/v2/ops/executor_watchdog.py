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
    raw = (proc / str(pid) / "stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    identity = ProcessIdentity(boot_id=boot_id, pid=pid, start_ticks=int(fields[19]),
                               process_group=int(fields[2]))
    return identity, int(fields[1]), fields[0], int(fields[21]) * os.sysconf("SC_PAGE_SIZE")


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


def signal_owned(identities: tuple[ProcessIdentity, ...], boot_id: str, *, hard=False) -> None:
    for identity in reversed(identities):
        try:
            current, _, state, _ = process_info(identity.pid, boot_id)
            same = (current.boot_id, current.pid, current.start_ticks) == (
                identity.boot_id, identity.pid, identity.start_ticks)
            if same and state != "Z":
                os.kill(identity.pid, signal.SIGKILL if hard else signal.SIGTERM)
        except (ProcessLookupError, FileNotFoundError):
            continue
