#!/usr/bin/env python3
"""Fresh-process adapter score runner used by the private canary.

A6: this used to call ``legacy_action`` in-process, which never exercised the
executor, the worker subprocess or staging materialization — exactly the path
that hid A1-A5. It now submits one real ``legacy_score_requests`` job to a
temporary catalog and runs it through the real ``Service``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_manifest(root: Path) -> dict:
    """Convert the prepared root's ``INPUT_MANIFEST.json`` into a
    ``LegacyInputManifest`` document, adding ``score_requests.json`` itself —
    it is written after that manifest and so is never listed inside it."""
    from engine.v2.contracts import LegacyFileRef, LegacyInputManifest
    from engine.v2.foundation import to_document

    manifest_path = root / "INPUT_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError("prepared private root lacks INPUT_MANIFEST.json")
    payload = json.loads(manifest_path.read_text())
    refs = [LegacyFileRef(path=item["path"], content_hash=item["artifact_id"],
                          byte_size=int(item["bytes"])) for item in payload["files"]]
    requests_file = root / "score_requests.json"
    if not requests_file.is_file():
        raise RuntimeError("prepared private root lacks score_requests.json")
    digest = hashlib.sha256(requests_file.read_bytes()).hexdigest()
    refs.append(LegacyFileRef(path="score_requests.json", content_hash="sha256:" + digest,
                              byte_size=requests_file.stat().st_size))
    manifest = LegacyInputManifest(
        manifest_id="phase1_canary", file_refs=tuple(refs), table_contract_refs=(),
        registry_and_model_refs=(), calendar_ref=None, selected_session="",
        finality_receipt_refs=(), knowledge_mode_by_table={}, availability_evidence_refs=(),
        read_set_complete=True, capture_implementation_ref="phase1_canary_adapter.v1")
    return to_document(manifest)


def build_request(manifest_ref):
    """The ``SubmitRequest`` a canary run makes — no execution here."""
    from engine.v2.contracts import JobSpec, SubmitRequest
    from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
    from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
    from engine.v2.foundation import content_hash

    profile = profile_named(DEFAULT_POLICY, "legacy_score")
    job = JobSpec(
        kind="legacy_score_requests",
        implementation_ref=content_hash(worker_source_manifest(ROOT)),
        spec_hash=None,
        environment_ref=content_hash(environment_identity(profile.thread_count or profile.cpu_count)),
        parameters={"expected_ids": ("legacy_score_requests",),
                    "requests_path": "legacy/score_requests.json",
                    "year_start": 2007, "year_end": 2030,
                    "input_bindings": {"legacy_manifest.json": manifest_ref}},
        input_refs=(manifest_ref,), dependency_job_ids=(), output_namespace="shadow",
        resource_class="legacy_score", retry_policy_ref="bounded",
        checkpoint_contract_ref="legacy_action.v1.0")
    return SubmitRequest(namespace="shadow", idempotency_key="phase1_canary_adapter",
                         principal="operator", job=job)


def run_via_supervisor(root: Path, output: Path) -> None:
    from engine.v2.foundation import ArtifactStore, SystemClock
    from engine.v2.ops.bootstrap import open_catalog
    from engine.v2.ops.catalog import transaction
    from engine.v2.ops.checkpoints import artifact as load_artifact, register_artifact
    from engine.v2.ops.profiles import DEFAULT_POLICY
    from engine.v2.ops.stages import registry
    from engine.v2.ops.submission import NamespacePolicy, submit
    from engine.v2.ops.supervisor import Service, serve

    catalog_root = root / "ops_canary"
    catalog_root.mkdir(parents=True, exist_ok=True)
    clock = SystemClock()
    conn = open_catalog(catalog_root / "catalog.sqlite", clock=clock)
    try:
        store = ArtifactStore(catalog_root)
        manifest_ref = store.publish_bytes(
            json.dumps(build_manifest(root), sort_keys=True).encode(),
            schema_ref="legacy_input_manifest.v1.0")
        with transaction(conn):
            register_artifact(conn, manifest_ref, None, clock)
        policy = NamespacePolicy({"operator": frozenset({"shadow", "smoke"})})
        receipt = submit(conn, registry(), policy,
                         build_request(manifest_ref.artifact_id), clock=clock)
        service = Service(conn, catalog_root, registry(), DEFAULT_POLICY, clock=clock,
                          code_source=ROOT, store_root=root)
        serve(service, once=True)
        row = conn.execute("SELECT state, failure_json FROM jobs WHERE job_id=?",
                           (receipt.job_id,)).fetchone()
        if row is None or row[0] != "succeeded":
            raise RuntimeError(f"legacy_score_requests did not succeed: {row}")
        artifact_id = conn.execute(
            "SELECT ao.artifact_id FROM attempts a JOIN attempt_outputs ao "
            "ON ao.attempt_id=a.attempt_id WHERE a.job_id=? AND a.state='succeeded' "
            "ORDER BY a.attempt_number DESC LIMIT 1", (receipt.job_id,)).fetchone()[0]
        ref = load_artifact(conn, store, artifact_id)
        output.write_bytes(store.read_verified(ref))
    finally:
        conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_via_supervisor(args.requests.parent, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
