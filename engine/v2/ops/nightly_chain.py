"""Snapshot chain for the nightly entry point (slice 4 of #564).

Contract in ``engine/v2/ops/ARCHITECTURE.md`` ("Nightly snapshot chain"). A leaf: not yet
called by ``nightly_trigger`` and it imports no CLI or store. Each step runs against the exact
snapshot its predecessor committed, behind a step receipt, so a restart never repeats it.
"""
from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from engine.v2.ops.errors import fail
from engine.v2.ops.nightly_receipts import Effect, begin_step, complete_step
from engine.v2.ops.nightly_session import SessionIdentity

__all__ = ["ChainResult", "ChainStep", "run_snapshot_chain"]


@dataclass(frozen=True)
class ChainStep:
    name: str
    effect: Callable[[str], Effect]  # predecessor snapshot -> the effect the receipt binds
    run: Callable[[str], None]  # perform the effect atomically on exactly that predecessor
    successor: Callable[[str], str | None]  # snapshot the effect committed, else None


@dataclass(frozen=True)
class ChainResult:
    final_snapshot: str
    executed: tuple[str, ...]
    skipped: tuple[str, ...]


def _digest(name: str, predecessor: str) -> str:
    return hashlib.sha256(f"{name}\0{predecessor}".encode()).hexdigest()


def _committed(step: ChainStep, predecessor: str) -> str | None:
    found = step.successor(predecessor)
    if found is not None and (not isinstance(found, str) or not found.strip()):
        raise fail("INTEGRITY_FAILED", "a chain step reported a malformed successor",
                   details={"step": step.name})
    return found


def run_snapshot_chain(root: Path, identity: SessionIdentity, generation: int,
                       start_snapshot: str, steps: Sequence[ChainStep], *,
                       conn: sqlite3.Connection | None = None) -> ChainResult:
    """Run ``steps`` serially from ``start_snapshot``; return the last committed snapshot."""
    steps = list(steps)
    names = [step.name for step in steps]
    if (not isinstance(start_snapshot, str) or not start_snapshot.strip() or not names
            or len(set(names)) != len(names)):
        raise fail("INVALID_REQUEST", "the chain needs a start snapshot and unique steps")
    predecessor = start_snapshot
    executed: list[str] = []
    skipped: list[str] = []
    for step in steps:
        receipt = begin_step(root, identity, generation, step.name,
                             _digest(step.name, predecessor), step.effect(predecessor))
        found = _committed(step, predecessor)
        if found is None:
            if receipt.state == "succeeded":
                raise fail("INTEGRITY_FAILED", "a completed step's commit is gone",
                           details={"step": step.name})
            step.run(predecessor)
            found = _committed(step, predecessor)
            if found is None:
                raise fail("DEPENDENCY_FAILED", "a chain step finished without committing",
                           details={"step": step.name})
            executed.append(step.name)
        else:
            skipped.append(step.name)
        complete_step(root, identity, generation, step.name, conn=conn)
        predecessor = found
    return ChainResult(predecessor, tuple(executed), tuple(skipped))