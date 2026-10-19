"""Explicitly build or read native stored research-history features."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactError, ArtifactStore, SystemClock, to_document
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import environment_identity, source_closure
from engine.v2.research import magnitude_history_store as history


def context(repository, snapshot, **request):
    """Fingerprint this installed producer, never caller-provided implementation."""
    root = Path(__file__).resolve().parents[1]
    sources = source_closure(root, ["experiments/magnitude_history_store.py"])
    producer = {"source_closure": sources, "environment": environment_identity(1)}
    return history.build_context(repository, snapshot, producer=producer, **request)


def build(repository, snapshot, **request):
    """Construct features once, under the supported native producer's identity."""
    binding = context(repository, snapshot, **request)
    output = history.materialize(repository, snapshot, binding)
    return {"context": binding, "feature_snapshot": to_document(output)}


def read(repository, snapshot, feature_snapshot, **request):
    """Reconstruct admission and return only the original committed stored rows."""
    return history.read_stored(repository, feature_snapshot, context(repository, snapshot, **request))


def _execute(args):
    """Close the native catalog before returning a complete CLI result."""
    conn = open_catalog(args.catalog, clock=SystemClock())
    try:
        # Pinned: decode each table's catalog fragment membership once, not on every scan (#953).
        repository = Repository(conn, ArtifactStore(args.store_root), pin_snapshots=True)
        snapshot = repository.resolve(args.source_snapshot_id)
        request = dict(as_of_month=args.as_of_month, evaluation_years=args.evaluation_years,
                       training_start=args.training_start, history_start=args.history_start, event_ids=args.event_id)
        return build(repository, snapshot, **request) if args.action == "build" else read(
            repository, snapshot, repository.resolve(args.feature_snapshot_id), **request)
    finally:
        conn.close()


def main(argv=None):
    """Local rebuild/read command; never providers, experiment evaluation or repair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "read"))
    for name in ("catalog", "store-root", "source-snapshot-id", "as-of-month", "training-start", "history-start"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--evaluation-years", nargs="+", required=True)
    parser.add_argument("--event-id", action="append", default=None)
    parser.add_argument("--feature-snapshot-id")
    args = parser.parse_args(argv)
    if (args.action == "read") != (args.feature_snapshot_id is not None):
        parser.error("read requires --feature-snapshot-id; build does not accept it")
    try:
        if not Path(args.catalog).is_file():
            parser.error("--catalog must name an existing native catalog file")
        if not Path(args.store_root).is_dir():
            raise ArtifactError("UNSAFE_PATH", "store root must be an existing directory")
        result = _execute(args)
    except (DataError, OpsError) as exc:
        print(json.dumps({"refused": exc.code, "message": str(exc)}, sort_keys=True))
        return 2
    except ArtifactError as exc:
        print(json.dumps({"refused": exc.code}, sort_keys=True))
        return 2
    except (OSError, sqlite3.Error):
        print(json.dumps({"refused": "INPUT_CHANGED"}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
