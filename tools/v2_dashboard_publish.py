"""Submit a retained-input Phase 3 serving publication for normal supervision.

Example (the supervisor performs the publication):
``/usr/bin/python3 tools/v2_dashboard_publish.py --root /private/ops
--source-publication-job job_SOURCE --projection-binding art_BINDING
--operation-id rollback-001``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.foundation import ArtifactStore, SystemClock, to_document
from engine.v2.foundation import content_hash
from engine.v2.ops.effects_graph import _generation_ref
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.publication_submit import submit_retained_publication
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy
from engine.v2.ops.supervisor import Service


def _run_sequence(conn, root, store_root, clock, items, log_path):
    policy = NamespacePolicy({"operator": frozenset({"shadow", "smoke"})})
    store = ArtifactStore(root)
    service = Service(conn, root, registry(), __import__("engine.v2.ops.profiles", fromlist=["DEFAULT_POLICY"]).DEFAULT_POLICY,
                      clock=clock, code_source=ROOT, store_root=store_root)
    service.start()
    records = []
    try:
        for label, source, binding, operation in items:
            receipt = submit_retained_publication(conn, store, registry=registry(), policy=policy, clock=clock,
                source_job_id=source, projection_binding_ref=binding, operation_id=operation)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                service.tick()
                row = conn.execute("SELECT state FROM jobs WHERE job_id=?", (receipt.job_id,)).fetchone()
                if row and row[0] in ("succeeded", "failed", "blocked", "cancelled"):
                    break
                time.sleep(0.1)
            if row is None or row[0] != "succeeded":
                detail = conn.execute("SELECT state,failure_json,queue_reason_json FROM jobs WHERE job_id=?", (receipt.job_id,)).fetchone()
                raise RuntimeError("sequence publication did not succeed: " + str(dict(detail) if detail else None))
            job = conn.execute("SELECT spec_json FROM jobs WHERE job_id=?", (receipt.job_id,)).fetchone()
            spec = json.loads(job["spec_json"])
            finality_id = spec["parameters"]["input_bindings"]["finality.json"]
            session = json.loads(store.read_verified(artifact(conn, store, finality_id)))["date"]
            scope = spec["parameters"].get("effect_scope") or spec["output_namespace"]
            generation = _generation_ref(SimpleNamespace(spec=SimpleNamespace(parameters=spec["parameters"])))
            release_id = "rel" + content_hash([scope, session, generation]).split(":")[1][:24]
            release = conn.execute("SELECT release_id,manifest_json,published_at,delivered_at FROM releases "
                                   "WHERE release_id=? AND delivered_at IS NOT NULL", (release_id,)).fetchone()
            if release is None:
                raise RuntimeError("sequence publication has no durable delivered release")
            projection = json.loads(store.read_verified(artifact(conn, store, binding)))["projection_release_id"]
            records.append({"label": label, "publication_job_id": receipt.job_id,
                            "projection_release_id": projection, "ops_release_id": release["release_id"],
                            "published_at": release["published_at"], "delivered_at": release["delivered_at"]})
            print(json.dumps({"event": "publication_sequence_progress", "label": label,
                            "job_id": receipt.job_id, "release_id": release["release_id"]}), flush=True)
    finally:
        service.close()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"update": records[:2], "rollback": records[2]}, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--store-root", type=Path, default=None)
    parser.add_argument("--source-publication-job", required=True)
    parser.add_argument("--projection-binding", required=True)
    parser.add_argument("--operation-id", required=True)
    parser.add_argument("--verify-sequence", action="store_true")
    parser.add_argument("--b-source-publication-job")
    parser.add_argument("--b-projection-binding")
    parser.add_argument("--sequence-log", type=Path)
    args = parser.parse_args(argv)
    clock = SystemClock()
    conn = open_catalog(args.catalog or args.root / "catalog.sqlite", clock=clock)
    try:
        if args.verify_sequence:
            if not args.b_source_publication_job or not args.b_projection_binding or not args.sequence_log:
                parser.error("--verify-sequence needs B source, B binding and --sequence-log")
            if args.store_root is None:
                parser.error("--verify-sequence requires --store-root")
            _run_sequence(conn, args.root, args.store_root, clock, (("a", args.source_publication_job,
                args.projection_binding, args.operation_id), ("b", args.b_source_publication_job,
                args.b_projection_binding, args.operation_id + "-b"), ("rollback", args.source_publication_job,
                args.projection_binding, args.operation_id + "-rollback")), args.sequence_log)
            return 0
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
