#!/usr/bin/env python3
"""P6-6 controlled-failure drill: crash one real production entrypoint at a
named point, prove the durable state it left behind is exactly the recoverable
shape it claims, then let the SAME production code finish on retry.

Two scenarios, each reusing a ``fault=`` hook that already exists in
production code -- nothing here re-implements the crash, the recovery or the
release machinery:

* ``after-good-release`` -- ``engine.v2.ops.publication.publish_local`` swaps
  the ``CURRENT`` pointer atomically, then acknowledges the release in the
  catalog. Its ``fault("pointer_before_ack")`` hook fires in that window. The
  drill stages and crash-publishes a second release ``R1`` over the published
  fixture's live ``R0``, then asserts: the pointer already names ``R1``, every
  byte of ``R1`` verifies against its manifest, and ``published_at`` is still
  NULL. Retrying the same call with no fault detects the pointer and only
  acknowledges it.
* ``before-any-release`` -- ``engine.v2.ops.backup.run_backup`` fsyncs and
  renames the manifest, then acknowledges its outbox effect. Its
  ``fault("after_manifest_before_ack")`` hook fires in that window. The drill
  asserts the manifest bytes are already valid, the effect was not delivered,
  and the separate publication target's ``CURRENT`` never moved; then, exactly
  as the production coordinator (``effects_graph.backup_effect``) does on any
  failure, it returns the still-claimed effect to ``pending`` with
  ``outbox.fail_effect`` (never waits out a lease) and proves an immediate
  retry completes against the identical manifest.

The fixture is real production code end to end: this tool imports
``checks.rearchitecture_phase6_restore_drill``'s ``_build_fixture`` /
``_build_published_fixture`` rather than inventing a second fake.

``--against-real-candidate`` never mutates the real catalog, backup or target:
the catalog is restored from the backup into an isolated scratch copy, the
target is copied wholesale into scratch first, and only the append-only,
content-addressed object store receives the drill's own new objects.

Exit codes: 0 on a PASS receipt, 1 on a FAIL receipt (the drill ran and found
an unexpected surviving state -- a verdict mismatch never raises), 2 when the
setup cannot be trusted (scratch root already exists, a real candidate is
half-specified, restore fails, or a scenario precondition is absent).

Usage::

    python3 tools/v2_controlled_failure_drill.py \\
        --scenario after-good-release --scratch-root /tmp/cfd \\
        --artifact-root evidence/
    python3 tools/v2_controlled_failure_drill.py \\
        --scenario before-any-release --scratch-root /tmp/cfd \\
        --against-real-candidate cat.sqlite --store-root store \\
        --target public --backup backups/shadow
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase6_restore_drill import (  # noqa: E402
    _all_gates,
    _build_fixture,
    _build_published_fixture,
    _fenced_claim,
)
from engine.v2.foundation import ArtifactStore, Clock, SystemClock, format_timestamp  # noqa: E402
from engine.v2.ops import publication  # noqa: E402
from engine.v2.ops.backup import prepare_backup, restore_backup, run_backup  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.ops.errors import OpsError  # noqa: E402
from engine.v2.ops.fingerprints import file_hash  # noqa: E402
from engine.v2.ops.outbox import fail_effect  # noqa: E402

__all__ = ["OpsFailureDrillError", "run_drill", "main"]

SCHEMA_VERSION = "controlled_failure_drill_receipt.v1.0"
POINTER_FAULT = "pointer_before_ack"
MANIFEST_FAULT = "after_manifest_before_ack"
CAPABILITIES = {
    "after-good-release": ["nightly-publish", "nightly-backup"],
    "before-any-release": ["nightly-backup"],
}


class OpsFailureDrillError(RuntimeError):
    """The drill cannot trust its setup: the scratch root already exists, a
    real candidate is only half-specified, the restore failed, or a scenario
    precondition (a live release to crash over) is absent. Never raised for a
    verdict mismatch -- that is a FAIL receipt."""


class _InjectedCrash(RuntimeError):
    """The drill's own crash, so an engine RuntimeError is never swallowed."""


class _Fault:
    """The named crash point: raises only when the production code reaches it."""

    def __init__(self, point: str) -> None:
        self.point = point
        self.fired = False

    def __call__(self, point: str) -> None:
        if point == self.point:
            self.fired = True
            raise _InjectedCrash(f"controlled failure injected at {self.point}")


@dataclass
class _Deployment:
    """One candidate to rehearse against: an isolated catalog, store and
    publication target, plus where this run's own backup target lives."""

    conn_factory: Callable[[], sqlite3.Connection]
    store: ArtifactStore
    target: Path
    clock: Clock
    backup_root: Path


def _manifest_is_consistent(manifest: dict, root: Path, key: str) -> bool:
    """The same byte-level checks ``run_backup`` itself makes: the manifest
    names the key and schema, and every file it hashes (the database, and each
    artifact copy) still hashes to the recorded digest."""
    if manifest.get("schema_version") != "catalog_backup.v1.0":
        return False
    if manifest.get("backup_key") != key:
        return False
    database = root / (key + ".sqlite")
    if not database.is_file() or file_hash(database) != manifest.get("database"):
        return False
    for name, info in manifest.get("artifacts", {}).items():
        path = root / "artifacts" / name
        if not path.is_file() or file_hash(path) != info.get("content_hash"):
            return False
    return True


def _run_after_good_release(deployment: _Deployment) -> dict:
    """Crash ``publish_local`` after the pointer swap, before the ack."""
    conn = deployment.conn_factory()
    try:
        target, store, clock = deployment.target, deployment.store, deployment.clock
        live = publication.current(target)
        if live is None:
            raise OpsFailureDrillError(
                f"after-good-release needs a live release to crash over; {target} has "
                "no CURRENT pointer")
        files = {"index.html": store.publish_bytes(
            b"<html>controlled-failure R1</html>", schema_ref="release_file.v1.0")}
        occurrence = format_timestamp(clock.now())
        claim = _fenced_claim(conn, clock, key="controlled-failure-r1")
        publication.stage_release(conn, store, "R1", occurrence, files,
                                  expected_current=live,
                                  gates=_all_gates(store, "R1", occurrence, files),
                                  clock=clock, claim=claim)
        fault = _Fault(POINTER_FAULT)
        try:
            publication.publish_local(conn, claim, store, target, "R1", scope="shadow",
                                      clock=clock, fault=fault)
        except _InjectedCrash:
            pass
        if not fault.fired:
            raise OpsFailureDrillError(
                "publish_local returned without reaching pointer_before_ack; the "
                "drill will not report on an unexercised crash window")

        crash_current = publication.current(target)
        try:
            publication._verify_files(target / "releases" / "R1", files)
            bytes_verified = True
        except Exception:  # noqa: BLE001 - a byte mismatch is a FAIL, not a raise
            bytes_verified = False
        crash_row = conn.execute(
            "SELECT published_at FROM releases WHERE release_id='R1'").fetchone()
        crash_published_at = None if crash_row is None else crash_row["published_at"]

        retry = publication.publish_local(conn, claim, store, target, "R1",
                                          scope="shadow", clock=clock)
        retry_row = conn.execute(
            "SELECT published_at FROM releases WHERE release_id='R1'").fetchone()
        retry_published_at = None if retry_row is None else retry_row["published_at"]
        final_current = publication.current(target)
    finally:
        conn.close()

    verdict = "PASS" if (fault.fired and crash_current == "R1" and bytes_verified
                         and crash_published_at is None and retry["delivered"]
                         and retry_published_at is not None
                         and final_current == "R1") else "FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": "after-good-release",
        "fault_point": POINTER_FAULT,
        "injected_exception_raised": fault.fired,
        "state_after_crash": {
            "current": crash_current,
            "release_bytes_verified": bytes_verified,
            "published_at": crash_published_at,
        },
        "retry": {
            "delivered": retry["delivered"],
            "published_at": retry_published_at,
            "current": final_current,
        },
        "verdict": verdict,
        "capabilities_covered": CAPABILITIES["after-good-release"],
    }


def _run_before_any_release(deployment: _Deployment) -> dict:
    """Crash ``run_backup`` after the manifest rename, before the ack."""
    conn = deployment.conn_factory()
    key = "controlled-failure-backup"
    try:
        target, store, clock = deployment.target, deployment.store, deployment.clock
        current_before = publication.current(target)
        prepare_backup(conn, key, {}, clock=clock)
        fault = _Fault(MANIFEST_FAULT)
        try:
            run_backup(conn, key=key, owner="controlled-failure-drill",
                       target=deployment.backup_root, clock=clock, store=store, fault=fault)
        except _InjectedCrash:
            pass
        if not fault.fired:
            raise OpsFailureDrillError(
                "run_backup returned without reaching after_manifest_before_ack; the "
                "drill will not report on an unexercised crash window")

        manifest_path = deployment.backup_root / (key + ".manifest.json")
        manifest_present = manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text()) if manifest_present else None
        manifest_verified = manifest_present and _manifest_is_consistent(
            manifest, deployment.backup_root, key)
        row = conn.execute(
            "SELECT effect_id, claim_token, claimed_by, state FROM outbox "
            "WHERE kind='backup' AND logical_key=?", (key,)).fetchone()
        if row is None:
            raise OpsFailureDrillError("the crashed backup left no outbox row")
        state_at_crash = row["state"]
        # The production coordinator (effects_graph.backup_effect) returns a
        # failed effect to pending rather than waiting out its lease, so an
        # immediate retry can claim it. Apply that same recovery, then retry.
        fail_effect(conn, row["effect_id"], {"error": "controlled_failure_drill"},
                    owner=row["claimed_by"], claim_token=row["claim_token"], clock=clock)
        state_after_recovery = conn.execute(
            "SELECT state FROM outbox WHERE effect_id=?",
            (row["effect_id"],)).fetchone()["state"]
        current_after_crash = publication.current(target)

        retry_manifest = run_backup(conn, key=key, owner="controlled-failure-drill",
                                    target=deployment.backup_root, clock=clock, store=store)
        state_after_retry = conn.execute(
            "SELECT state FROM outbox WHERE effect_id=?",
            (row["effect_id"],)).fetchone()["state"]
        current_final = publication.current(target)
    finally:
        conn.close()

    verdict = "PASS" if (fault.fired and manifest_present and manifest_verified
                         and state_at_crash != "delivered"
                         and state_after_recovery == "pending"
                         and current_after_crash == current_before
                         and retry_manifest == manifest
                         and state_after_retry == "delivered"
                         and current_final == current_before) else "FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": "before-any-release",
        "fault_point": MANIFEST_FAULT,
        "injected_exception_raised": fault.fired,
        "state_after_crash": {
            "manifest_present": manifest_present,
            "manifest_verified": manifest_verified,
            "outbox_state": state_at_crash,
            "outbox_state_after_recovery": state_after_recovery,
            "current": current_after_crash,
        },
        "retry": {
            "manifest_matches": retry_manifest == manifest,
            "outbox_state": state_after_retry,
            "current": current_final,
        },
        "verdict": verdict,
        "capabilities_covered": CAPABILITIES["before-any-release"],
    }


def _scratch_deployment(scenario: str, scratch_root: Path) -> _Deployment:
    clock = SystemClock()
    if scenario == "after-good-release":
        _, _, _, _, store_root, target, conn_factory = \
            _build_published_fixture(scratch_root)
    else:
        _build_fixture(scratch_root)
        catalog_path, store_root = scratch_root / "ops.sqlite", scratch_root / "objects"
        target = scratch_root / "publication"
        conn_factory = partial(open_catalog, catalog_path, clock=clock)
    return _Deployment(conn_factory=conn_factory, store=ArtifactStore(store_root),
                       target=target, clock=clock,
                       backup_root=scratch_root / "controlled-failure-backup")


def _probe_candidate_catalog(catalog: Path, scenario: str) -> None:
    """Read-only sanity check of the real candidate catalog. Never a write:
    ``mode=ro`` is the whole point, so this cannot touch the live deployment."""
    uri = "file:" + catalog.resolve().as_posix() + "?mode=ro"
    try:
        probe = sqlite3.connect(uri, uri=True)
        try:
            published = probe.execute(
                "SELECT COUNT(*) FROM releases WHERE published_at IS NOT NULL").fetchone()[0]
        finally:
            probe.close()
    except sqlite3.Error as exc:
        raise OpsFailureDrillError(
            f"real candidate catalog {catalog} is not a readable ops catalog: {exc}") from exc
    if scenario == "after-good-release" and published < 1:
        raise OpsFailureDrillError(
            f"real candidate catalog {catalog} has no published release, so "
            "after-good-release has nothing to crash over")


def _candidate_deployment(scenario: str, scratch_root: Path,
                          candidate: tuple[Path, Path, Path, Path]) -> _Deployment:
    catalog, store_root, real_target, backup = candidate
    if not catalog.is_file():
        raise OpsFailureDrillError(f"real candidate catalog {catalog} is not a file")
    if not backup.is_dir():
        raise OpsFailureDrillError(f"backup {backup} is not a directory")
    _probe_candidate_catalog(catalog, scenario)
    try:
        restored = restore_backup(backup, scratch_root / "copy")
    except Exception as exc:  # noqa: BLE001 - any restore refusal is a setup refusal
        raise OpsFailureDrillError(f"restore_backup failed: {exc}") from exc
    target = scratch_root / "target"
    if real_target.exists():
        if real_target.is_symlink() or not real_target.is_dir():
            raise OpsFailureDrillError(
                f"publication target {real_target} is not a real directory")
        shutil.copytree(real_target, target)
    clock = SystemClock()
    return _Deployment(
        conn_factory=partial(open_catalog, restored / "ops.sqlite", clock=clock),
        store=ArtifactStore(store_root), target=target, clock=clock,
        backup_root=scratch_root / "controlled-failure-backup")


def _candidate(catalog, store_root, target, backup):
    given = (catalog, store_root, target, backup)
    if all(item is None for item in given):
        return None
    if any(item is None for item in given):
        raise OpsFailureDrillError(
            "--against-real-candidate requires --store-root, --target and --backup "
            "together")
    return Path(catalog), Path(store_root), Path(target), Path(backup)


def _write_evidence(receipt: dict, artifact_root, *, real_candidate: bool) -> None:
    """The receipt as durable evidence: under the repo's
    ``reports/phase6_evidence/controlled_failure/`` only when run
    ``--against-real-candidate``; scratch/fixture runs write nowhere but
    ``--artifact-root`` (when given) -- never the real repo tree."""
    name = f"controlled_failure_{receipt['scenario']}_receipt.json"
    payload = json.dumps(receipt, indent=2, sort_keys=True)
    if real_candidate:
        evidence_dir = ROOT / "reports" / "phase6_evidence" / "controlled_failure"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / name).write_text(payload)
    if artifact_root is not None:
        artifact_root = Path(artifact_root)
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / name).write_text(payload)


def run_drill(*, scenario: str, scratch_root: Path | str, artifact_root=None,
              candidate_catalog=None, candidate_store_root=None, candidate_target=None,
              candidate_backup=None) -> dict:
    scratch_root = Path(scratch_root)
    if scratch_root.exists():
        raise OpsFailureDrillError(
            f"scratch root {scratch_root} already exists; refusing to reuse a directory "
            "the drill did not create")
    candidate = _candidate(candidate_catalog, candidate_store_root, candidate_target,
                           candidate_backup)
    scratch_root.mkdir(parents=True)
    try:
        if candidate is None:
            deployment = _scratch_deployment(scenario, scratch_root)
        else:
            deployment = _candidate_deployment(scenario, scratch_root, candidate)
        if scenario == "after-good-release":
            receipt = _run_after_good_release(deployment)
        else:
            receipt = _run_before_any_release(deployment)
    except OpsFailureDrillError:
        raise
    except (OpsError, sqlite3.Error, OSError, ValueError) as exc:
        raise OpsFailureDrillError(f"{scenario} refused: {exc}") from exc
    _write_evidence(receipt, artifact_root, real_candidate=candidate is not None)
    return receipt


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True,
                        choices=("after-good-release", "before-any-release"))
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--against-real-candidate", type=Path, default=None,
                        help="the real candidate's catalog database (read-only)")
    parser.add_argument("--store-root", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--backup", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        receipt = run_drill(
            scenario=args.scenario, scratch_root=args.scratch_root,
            artifact_root=args.artifact_root,
            candidate_catalog=args.against_real_candidate,
            candidate_store_root=args.store_root, candidate_target=args.target,
            candidate_backup=args.backup)
    except OpsFailureDrillError as exc:
        print(f"controlled-failure-drill: refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
