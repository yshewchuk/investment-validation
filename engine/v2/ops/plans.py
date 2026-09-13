"""Immutable, reviewable plans. Planning never authorizes an omitted effect."""
from __future__ import annotations

import json
from pathlib import Path

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import ArtifactStore, content_hash
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest

NIGHTLY_GRAPH = {
    "refresh": (), "finality": ("refresh",), "features": ("finality",),
    "score": ("features",), "decision_validation": ("score",),
    "decision_commit": ("decision_validation",), "settlement": ("finality",),
    "model_evidence": ("features",), "export": ("decision_commit",),
    "projection": ("export", "model_evidence"), "selfcheck": ("projection",),
    "engineering": (), "publication": ("selfcheck", "engineering"),
    "delivery": ("publication",), "backup": ("decision_commit",),
}


def check_plan(source_root, expected_ids):
    manifest = worker_source_manifest(Path(source_root))
    return {"schema_version": "operations_plan.v1.0", "kind": "artifact_check", "mode": "shadow",
            "source_manifest": manifest, "implementation_ref": content_hash(manifest),
            "environment_ref": content_hash(environment_identity()), "effects": ["private_artifacts"],
            "parameters": {"expected_ids": list(expected_ids)}, "resource_class": "delivery",
            "input_refs": [], "blocked_prerequisites": []}


def nightly_plan(source_root, session, *, mode="shadow", manifest_ref=None,
                 tickers=(), year_start=2024, year_end=2026):
    from datetime import date
    date.fromisoformat(session)
    if mode != "shadow":
        raise fail("INVALID_REQUEST", "production cutover has not been activated")
    plan = check_plan(source_root, [])
    plan.update(kind="nightly", session=session, graph=NIGHTLY_GRAPH,
                plan_hash=content_hash({"session": session, "manifest": manifest_ref,
                                         "tickers": list(tickers), "year_start": year_start,
                                         "year_end": year_end,
                                         "implementation": plan["implementation_ref"]}),
                input_manifest_ref=manifest_ref, tickers=list(tickers),
                year_start=year_start, year_end=year_end,
                effects=["private_shadow_artifacts", "copy_only_decision_authority"],
                blocked_prerequisites=[] if manifest_ref else [
                    "frozen_legacy_input_manifest", "adapter_parity_receipt"])
    return plan


def save_plan(conn, root, document, *, clock):
    store = ArtifactStore(root)
    ref = store.publish_bytes(json.dumps(document, sort_keys=True).encode(), schema_ref="operations_plan.v1.0")
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return ref


def request_from_plan(plan, key):
    if plan.get("schema_version") != "operations_plan.v1.0" or plan.get("blocked_prerequisites"):
        raise fail("INVALID_REQUEST", "plan has unsupported schema or blocked prerequisites")
    if plan["kind"] != "artifact_check" or plan["effects"] != ["private_artifacts"]:
        raise fail("INVALID_REQUEST", "plan kind is not enabled for submission")
    job = JobSpec(kind=plan["kind"], implementation_ref=plan["implementation_ref"],
                  spec_hash=plan.get("spec_hash"), environment_ref=plan["environment_ref"],
                  parameters=plan["parameters"], input_refs=tuple(plan["input_refs"]),
                  output_namespace=plan["mode"], resource_class=plan["resource_class"],
                  retry_policy_ref="bounded", checkpoint_contract_ref="receipt.v1.0")
    return SubmitRequest(namespace=plan["mode"], idempotency_key=key, principal="operator", job=job)
