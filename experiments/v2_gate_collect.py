"""Collect a sealed native gate sweep, optionally waiting for its supervisor."""
import argparse
import errno
import json
import math
import sqlite3
import stat
from pathlib import Path
from time import monotonic, sleep

from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactError, ArtifactStore
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.submission import MAX_JOB_FAILURE_BYTES, get_job
from experiments.native_gate_worker import read_native_staged_input
from experiments.native_sweep_collection import TERMINAL, collect_sweep


def run_sweep(conn, store, repository, document, *, code_root, timeout_seconds=3600,
              **kwargs):
    """Observe the ordinary supervisor; never start, stop or take over workers."""
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise fail("INVALID_EXPERIMENT_SPEC", "sweep timeout must be positive and finite")
    result = collect_sweep(conn, store, repository, document, code_root=code_root, **kwargs)
    if result["index_ref"] is not None:
        return result
    job_ids = [arm["job_id"] for arm in result["index"]["variants"]]
    deadline = monotonic() + timeout_seconds
    try:
        while not all(get_job(conn, job_id, receipt_decode_failure_code="EXPERIMENT_IDENTITY_CONFLICT",
                             max_failure_bytes=MAX_JOB_FAILURE_BYTES).state
                      in TERMINAL for job_id in job_ids):
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            sleep(min(1, remaining))
    except sqlite3.IntegrityError:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep observation catalog constraint failed") from None
    except (OSError, sqlite3.Error):
        raise fail("RESOURCE_UNAVAILABLE", "sweep observation was interrupted") from None
    return collect_sweep(conn, store, repository, document, code_root=code_root, **kwargs)


def _roster(path):
    try:
        document = json.loads(read_native_staged_input(Path(path)))
        if not isinstance(document, dict):
            raise ValueError
        return document
    except OSError as error:
        if error.errno not in {errno.ENOENT, errno.ENOTDIR, errno.EACCES, errno.EPERM,
                               errno.ELOOP, errno.EINVAL, errno.ENAMETOOLONG}:
            raise
        raise fail("INVALID_EXPERIMENT_SPEC", "sweep roster requires a readable regular path") from None
    except (ValueError, TypeError, RecursionError):
        raise fail("INVALID_EXPERIMENT_SPEC", "sweep roster requires bounded JSON") from None


def _catalog(path):
    try:
        metadata = Path(path).stat()
    except (FileNotFoundError, NotADirectoryError):
        raise fail("INVALID_EXPERIMENT_SPEC", "sweep catalog must name an existing regular file") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise fail("INVALID_EXPERIMENT_SPEC", "sweep catalog must name an existing regular file")
    return connect(path, must_exist=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "run"))
    for name in ("roster", "catalog", "store-root", "as-of-month", "code-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--scope", default="shadow")
    parser.add_argument("--event-id", action="append")
    parser.add_argument("--purpose", default="sweep")
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    args, conn = parser.parse_args(argv), None
    try:
        if args.purpose != "sweep":
            raise fail("HOLDOUT_ACCESS_DENIED", "gate sweep collection has no final-read authority")
        if args.no_ledger == (args.ledger is not None):
            raise fail("INVALID_EXPERIMENT_SPEC", "choose --no-ledger or --ledger")
        document = _roster(args.roster)
        conn = _catalog(args.catalog)
        store = ArtifactStore(args.store_root)
        action = run_sweep if args.action == "run" else collect_sweep
        result = action(conn, store, Repository(conn, store, pin_snapshots=True), document,
            code_root=Path(args.code_root), as_of_month=args.as_of_month, scope=args.scope,
            event_ids=args.event_id, no_ledger=args.no_ledger, ledger_path=args.ledger,
            **({"timeout_seconds": args.timeout_seconds} if args.action == "run" else {}))
        print(json.dumps(result, sort_keys=True))
        return 0 if result["index"]["complete"] else 2
    except (OpsError, DataError, ArtifactError, OSError, sqlite3.Error) as error:
        print(json.dumps({"refused": error.code if isinstance(error, (OpsError, DataError, ArtifactError)) else
                          "RESOURCE_UNAVAILABLE"}))
        return 2
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
