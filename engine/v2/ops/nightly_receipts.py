"""Step receipts and effect reconciliation for the nightly entry point (slice 2 of #564).

Contract in ``engine/v2/ops/ARCHITECTURE.md`` ("Nightly step receipts"). A leaf: not yet
called by ``nightly_trigger``, and it runs no effect itself; it records intents, proves
effects and decides what a restarted run may do.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

from engine.v2.ops.errors import fail
from engine.v2.ops.nightly_session import (
    STATE_DIR,
    Generation,
    SessionIdentity,
    atomic_write,
    load_session,
)

__all__ = ["Effect", "Reconciliation", "StepReceipt", "begin_step", "complete_step",
           "reconcile_step"]

SCHEMA = "nightly_receipts.v1"
KINDS = ("catalog_job", "artifact", "external")


@dataclass(frozen=True)
class Effect:
    kind: str  # "catalog_job" | "artifact" | "external"
    ref: str = ""  # job id, or artifact path; free label for "external"
    sha256: str = ""  # artifact content hash; empty otherwise


@dataclass(frozen=True)
class StepReceipt:
    step: str
    state: str  # "intent" | "succeeded"
    request_digest: str
    effect: Effect


@dataclass(frozen=True)
class Reconciliation:
    outcome: str  # "completed" | "not_started"
    receipt: StepReceipt | None


def _valid(step: str, digest: str, effect: Effect) -> bool:
    return bool(isinstance(step, str) and step.strip() and isinstance(digest, str)
                and digest.strip() and isinstance(effect, Effect) and effect.kind in KINDS
                and isinstance(effect.ref, str) and isinstance(effect.sha256, str)
                and (effect.kind == "external" or effect.ref)
                and (effect.kind != "artifact" or (effect.sha256 and Path(effect.ref).is_absolute()))
                and (effect.kind == "artifact" or not effect.sha256))


def _active(root: Path, identity: SessionIdentity, generation: int) -> Generation:
    state = load_session(root, identity)
    if (type(generation) is not int or state is None or state.active.generation != generation
            or state.active.status != "started"):
        raise fail("CHECKPOINT_INCOMPATIBLE", "generation is not the active started one")
    return state.active


def _encode(run_id: str, steps: dict[str, StepReceipt]) -> str:
    return json.dumps({"schema": SCHEMA, "run_id": run_id, "steps": {
        name: {"state": r.state, "request_digest": r.request_digest,
               "effect": {"kind": r.effect.kind, "ref": r.effect.ref,
                          "sha256": r.effect.sha256}} for name, r in steps.items()}},
        sort_keys=True)


def _decode(text: str, run_id: str) -> dict[str, StepReceipt]:
    try:
        doc = json.loads(text)
        if not isinstance(doc, dict):
            raise ValueError("not an object")
        if doc.get("schema") != SCHEMA:
            raise fail("CHECKPOINT_INCOMPATIBLE",
                       "receipts have another schema; move the file aside")
        if doc["run_id"] != run_id:
            raise ValueError("run_id")
        steps = {name: StepReceipt(name, s["state"], s["request_digest"], Effect(**s["effect"]))
                 for name, s in doc["steps"].items()}
        if not all(r.state in ("intent", "succeeded") and _valid(r.step, r.request_digest, r.effect)
                   for r in steps.values()):
            raise ValueError("step")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise fail("INTEGRITY_FAILED", "step receipts are unreadable") from exc
    return steps


def _update(root: Path, identity: SessionIdentity, generation: int, change):
    """``change(steps) -> (result, new steps | None)`` under the per-run file lock."""
    gen = _active(root, identity, generation)
    path = Path(root).joinpath(*STATE_DIR, f"{gen.run_id}.receipts.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            steps = _decode(path.read_text(), gen.run_id)
        except FileNotFoundError:
            steps = {}
        except (OSError, UnicodeDecodeError) as exc:
            raise fail("INTEGRITY_FAILED", "step receipts are unreadable") from exc
        result, new = change(steps)
        if new is not None:
            atomic_write(path, _encode(gen.run_id, new))
        return result


def _probe(effect: Effect, conn: sqlite3.Connection | None) -> str:
    """``holds`` | ``absent`` | ``unproven`` for the effect itself, never for the clock."""
    if effect.kind == "catalog_job":
        if conn is None:
            raise fail("INVALID_REQUEST", "a catalog_job effect needs the catalog connection")
        row = conn.execute("SELECT state FROM jobs WHERE job_id = ?", (effect.ref,)).fetchone()
        return "absent" if row is None else "holds" if row[0] == "succeeded" else "unproven"
    if effect.kind == "artifact":
        try:
            data = Path(effect.ref).read_bytes()
        except FileNotFoundError:
            return "absent"
        except OSError:
            return "unproven"
        return "holds" if hashlib.sha256(data).hexdigest() == effect.sha256 else "unproven"
    return "unproven"  # external: nothing to probe; only the caller's receipt can bind it


def begin_step(root: Path, identity: SessionIdentity, generation: int, step: str,
               request_digest: str, effect: Effect) -> StepReceipt:
    """Record the intent before the effect. Repeating it returns the existing receipt."""
    if not _valid(step, request_digest, effect):
        raise fail("INVALID_REQUEST", "step, request digest or effect is malformed")

    def change(steps):
        cur = steps.get(step)
        if cur is None:
            rec = StepReceipt(step, "intent", request_digest, effect)
            return rec, {**steps, step: rec}
        if cur.request_digest != request_digest or cur.effect != effect:
            raise fail("IDEMPOTENCY_CONFLICT",
                       "step was recorded with another request; changed inputs need a rerun")
        return cur, None
    return _update(root, identity, generation, change)


def complete_step(root: Path, identity: SessionIdentity, generation: int, step: str, *,
                  conn: sqlite3.Connection | None = None) -> StepReceipt:
    """Bind a finished effect to its step: prove it (an ``external`` effect is the caller's
    assertion), then mark the step ``succeeded``. Idempotent."""
    def change(steps):
        cur = steps.get(step)
        if cur is None:
            raise fail("INVALID_REQUEST", "step has no recorded intent")
        if cur.state == "succeeded":
            return cur, None
        if cur.effect.kind != "external" and _probe(cur.effect, conn) != "holds":
            raise fail("INVALID_REQUEST", "the declared effect is not complete")
        done = replace(cur, state="succeeded")
        return done, {**steps, step: done}
    return _update(root, identity, generation, change)


def reconcile_step(root: Path, identity: SessionIdentity, generation: int, step: str,
                   request_digest: str, *, conn: sqlite3.Connection | None = None
                   ) -> Reconciliation:
    """Decide ``completed`` / ``not_started``; an effect that cannot be proven is uncertain
    and raises ``CHECKPOINT_INCOMPATIBLE`` without writing, so it is never repeated."""
    def change(steps):
        cur = steps.get(step)
        if cur is None:
            return Reconciliation("not_started", None), None
        if cur.request_digest != request_digest:
            raise fail("IDEMPOTENCY_CONFLICT",
                       "step was recorded with another request; changed inputs need a rerun")
        if cur.state == "succeeded":
            if cur.effect.kind != "external" and _probe(cur.effect, conn) != "holds":
                raise fail("INTEGRITY_FAILED", "a recorded effect no longer holds")
            return Reconciliation("completed", cur), None
        seen = _probe(cur.effect, conn)
        if seen == "holds":
            done = replace(cur, state="succeeded")
            return Reconciliation("completed", done), {**steps, step: done}
        if seen == "absent":
            return Reconciliation("not_started", cur), None
        raise fail("CHECKPOINT_INCOMPATIBLE",
                   "effect outcome cannot be proven; repair in a new generation",
                   details={"step": step, "kind": cur.effect.kind})
    return _update(root, identity, generation, change)