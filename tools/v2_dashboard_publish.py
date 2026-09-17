"""Submit a retained-input Phase 3 serving publication for normal supervision.

Example (the supervisor performs the publication):
``/usr/bin/python3 tools/v2_dashboard_publish.py --root /private/ops
--source-publication-job job_SOURCE --projection-binding art_BINDING
--operation-id rollback-001``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.v2.foundation import ArtifactStore, SystemClock, to_document
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.publication_submit import submit_retained_publication
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-publication-job", required=True)
    parser.add_argument("--projection-binding", required=True)
    parser.add_argument("--operation-id", required=True)
    args = parser.parse_args(argv)
    clock = SystemClock()
    conn = open_catalog(args.root / "catalog.sqlite", clock=clock)
    try:
        receipt = submit_retained_publication(
            conn, ArtifactStore(args.root), registry=registry(),
            policy=NamespacePolicy({"operator": frozenset({"shadow", "smoke"})}), clock=clock,
            source_job_id=args.source_publication_job, projection_binding_ref=args.projection_binding,
            operation_id=args.operation_id)
    finally:
        conn.close()
    print(json.dumps(to_document(receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
