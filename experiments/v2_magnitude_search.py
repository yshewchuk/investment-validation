"""Prepare or verify a bounded nested magnitude search plan; never execute trials."""
import argparse
import errno
import json
import os
import sqlite3
import stat
from pathlib import Path

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    DocumentError,
    from_document,
    to_document,
)
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import experiment_spec_from_document
from experiments.magnitude_search_plan import read_search_plan, seal_search_plan
from experiments.native_registration import (
    MAX_NATIVE_EVIDENCE_BYTES,
    read_native_registration,
    valid_registration_read_budget,
)

MAX_METADATA_BYTES = 1024 * 1024


def _read_document(path):
    """Bound the opened regular file before parsing; never block on a FIFO."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_METADATA_BYTES:
                raise fail("INVALID_EXPERIMENT_SPEC", "search metadata requires bounded regular files")
            raw = bytearray()
            with os.fdopen(descriptor, "rb", buffering=0, closefd=False) as stream:
                while len(raw) <= MAX_METADATA_BYTES:
                    chunk = stream.read(MAX_METADATA_BYTES + 1 - len(raw))
                    if not chunk:
                        break
                    raw.extend(chunk)
            if len(raw) > MAX_METADATA_BYTES:
                raise fail("INVALID_EXPERIMENT_SPEC", "search metadata exceeds its byte limit")
        finally:
            os.close(descriptor)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise fail("INVALID_EXPERIMENT_SPEC", "search metadata must not follow leaf symlinks") from None
        raise
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError):
        raise fail("INVALID_EXPERIMENT_SPEC", "search planning request could not be completed") from None


def main(argv=None):
    """Local metadata-artifact lifecycle; no repository reads or experiment attempts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    for name in ("spec", "catalog", "store-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--plan-ref")
    parser.add_argument("--no-ledger", action="store_true", help="accepted no-op; planning never writes a ledger")
    parser.add_argument("--max-registration-bytes", type=int, default=MAX_NATIVE_EVIDENCE_BYTES)
    args = parser.parse_args(argv)
    if not valid_registration_read_budget(args.max_registration_bytes):
        parser.error("--max-registration-bytes must be a positive SQLite-safe byte budget")
    if (args.action == "verify") != (args.plan_ref is not None):
        parser.error("verify requires --plan-ref in addition to --request; prepare accepts no --plan-ref")
    conn = None
    try:
        if not Path(args.store_root).is_dir():
            raise fail("RESOURCE_UNAVAILABLE", "search requires an existing artifact store")
        spec_document = _read_document(args.spec)
        try:
            spec = experiment_spec_from_document(spec_document)
        except TypeError:
            raise fail("INVALID_EXPERIMENT_SPEC", "search planning request could not be completed") from None
        document = _read_document(args.request)
        try:
            reference = None if args.plan_ref is None else from_document(
                ArtifactRef, _read_document(args.plan_ref))
        except DocumentError:
            raise fail("INVALID_EXPERIMENT_SPEC", "search planning request could not be completed") from None
        conn = connect(args.catalog, must_exist=True)
        store = ArtifactStore(args.store_root)
        registration = read_native_registration(conn, store, spec, max_evidence_bytes=args.max_registration_bytes)
        if args.action == "prepare":
            sealed = seal_search_plan(conn, store, registration, document)
            result = {"plan_ref": sealed["plan_ref"], "plan_id": sealed["plan"]["plan_id"],
                      "candidate_count": sealed["plan"]["candidate_count"],
                      "planned_fit_count": sealed["plan"]["planned_fit_count"]}
        else:
            plan = read_search_plan(conn, store, registration, document, reference)
            result = {"verified": plan["plan_id"]}
    except OpsError as error:
        print(json.dumps({"refused": to_document(error.problem)}, sort_keys=True))
        return 2
    except (ArtifactError, OSError, sqlite3.Error):
        problem = fail("RESOURCE_UNAVAILABLE", "search planning request could not be completed").problem
        print(json.dumps({"refused": to_document(problem)}, sort_keys=True))
        return 2
    finally:
        if conn is not None:
            conn.close()
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
