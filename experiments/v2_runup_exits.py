"""Print a read-only, pinned STR-RUNUP daily-exit comparison as JSON."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import stat
import sys
from contextlib import redirect_stdout
from pathlib import Path

from engine.v2.data.documents import reject_duplicate_keys
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, canonical_json
from engine.v2.ops.errors import OpsError, fail
from experiments.runup_exit_report import compare_runup_exits, validate_request


def _request(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise fail("INVALID_EXPERIMENT_SPEC", "comparison request must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(1024 * 1024 + 1)
    finally:
        os.close(descriptor)
    try:
        if len(data) > 1024 * 1024:
            raise ValueError
        return validate_request(json.loads(data, object_pairs_hook=reject_duplicate_keys))
    except (ValueError, UnicodeError, RecursionError):
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid comparison JSON request") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--as-of-month", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--no-ledger", action="store_true", help="accepted no-op; comparison never writes a ledger")
    args = parser.parse_args(argv)
    try:
        request = _request(args.request)
        connection = sqlite3.connect(args.catalog.resolve().as_uri() + "?mode=ro",
                                     uri=True, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            repository = Repository(connection, ArtifactStore(args.store_root), pin_snapshots=True)
            with redirect_stdout(sys.stderr):
                result = compare_runup_exits(repository, repository.resolve(args.snapshot_id),
                                            as_of_month=args.as_of_month, document=request)
        finally:
            connection.close()
    except (DataError, OpsError) as exc:
        print(canonical_json({"status": "refused", "refused": exc.code,
                              "details": exc.problem.details}))
        return 2
    except (OSError, sqlite3.Error):
        print(canonical_json({"status": "refused", "refused": "INPUT_CHANGED"}))
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
