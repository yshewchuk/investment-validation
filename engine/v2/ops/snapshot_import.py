"""§7/§10 planning glue and the two pure workers — task brief P2-7/Task7b.

Two job kinds live here, both coordinator-validated (like ``decision_evidence``
and ``ledger_export`` before them; ``engine/v2/ops/snapshot_promotion.py``
holds their coordinator-side effects, promotion and rollback):

* ``snapshot_import``: streams the pinned legacy read set, in a fresh worker
  process, into per-file :class:`~engine.v2.data.objects.FragmentInspection`
  candidates (:func:`worker_snapshot_import`). The worker never touches the
  catalog and never publishes an ``ArtifactStore`` object — it only proves
  what the staged bytes measure out to. Judgement call (task brief decision
  1): the worker's per-fragment ``logical_content_hash`` is trusted as-is by
  the coordinator; a *multi*-fragment partition's own combined hash is still
  recomputed there, by ``objects.partition_logical_hash`` over the freshly
  published objects — the one check that is genuinely expensive to skip
  (§6 invariant 6) and genuinely cheap to redo once, after publish.
* ``legacy_rebuild_candidate``: runs the real legacy rebuild entrypoint
  (``engine.data.rebuild.rebuild``, via ``ops.legacy_adapter.run_legacy_rebuild``)
  rooted at a private candidate directory. Its own coordinator effect
  (``snapshot_promotion.legacy_rebuild_candidate_effect``) is what actually
  refuses a run that touched anything outside that root — the worker itself
  only records what it measures, it is never trusted to police its own writes.

Progress (task brief decision 3): no worker-side, mid-run progress channel
exists today (``ops.supervisor.Service._progress`` only ever reports a
generic per-tick "worker observed" heartbeat while a subprocess is alive; a
worker sends its result once, at exit, over one pipe). Per-partition progress
is therefore recorded in the worker's own output (``inspections.json``'s
``partitions_completed``, in commit order) rather than through a channel that
does not exist — this module's own judgement call, following the task brief's
explicit fallback.

Admission (task brief decision 3): the estimated read-set bytes vs. the
attempt's scratch budget is already enforced generically, before any worker
launches, by ``ops.supervisor.Service._populate_legacy_staging`` for every
kind declaring a ``("legacy_store", "read")`` domain — ``snapshot_import``
gets this for free by declaring that domain in ``stages.py``; no new
admission code is needed here.

Layer: ``engine.v2.ops`` (layer 2), so it may import ``engine.v2.data``
freely (downward). The one legacy-touching call (``run_legacy_rebuild``) is
delegated to ``engine.v2.ops.legacy_adapter`` — the package's sole audited
subprocess boundary for this purpose (``checks/import_layers.py``'s
"undeclared-process-edge"; this module never calls ``subprocess`` itself).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from engine.v2.contracts import JobSpec, SnapshotImportRequest, SubmitRequest, TableContract
from engine.v2.data import documents
from engine.v2.data import legacy_adapter as data_legacy_adapter
from engine.v2.data.import_snapshot import ImportPlan, request_hash
from engine.v2.data.objects import inspect_staged_file
from engine.v2.foundation import canonical_json, content_hash, from_document, to_document
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import artifact, register_artifact
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.ops.legacy_adapter import run_legacy_rebuild
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.submission import submit

__all__ = [
    "protected_paths_hash",
    "save_import_plan",
    "submit_candidate_rebuild",
    "submit_import",
    "worker_legacy_rebuild_candidate",
    "worker_snapshot_import",
]

_YEAR_IN_PATH = re.compile(r"/year=(\d{4})/")


# --------------------------------------------------------------------------
# planning glue: publish an ImportPlan, then submit it through the supervisor
# --------------------------------------------------------------------------


def _publish_and_register(store, conn, document, schema_ref, *, clock):
    ref = store.publish_bytes(canonical_json(document).encode("utf-8"), schema_ref=schema_ref)
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return ref


def save_import_plan(conn, store, plan: ImportPlan, *, clock):
    """Publish an :class:`ImportPlan`'s three documents and one small artifact
    naming them all (§7.1 point 4: "hash the request and submit it").
    """
    manifest_ref = _publish_and_register(store, conn, to_document(plan.legacy_input_manifest),
                                         "legacy_input_manifest.v1.0", clock=clock)
    request_ref = _publish_and_register(store, conn, to_document(plan.snapshot_import_request),
                                        "snapshot_import_request.v1.0", clock=clock)
    mapping_ref = _publish_and_register(store, conn, data_legacy_adapter.build_legacy_mapping(),
                                        "legacy_table_mapping.v1.0", clock=clock)
    document = {
        "schema_version": "snapshot_import_plan.v1.0", "scope": plan.snapshot_import_request.scope,
        "manifest_ref": manifest_ref.artifact_id, "request_ref": request_ref.artifact_id,
        "mapping_ref": mapping_ref.artifact_id,
        "request_hash": request_hash(plan.snapshot_import_request),
    }
    return _publish_and_register(store, conn, document, "snapshot_import_plan.v1.0", clock=clock)


def submit_import(conn, store, plan_artifact_id, *, registry, policy, clock,
                  idempotency_key, repo_root, namespace="shadow"):
    """Build the ``snapshot_import`` ``JobSpec`` from a saved plan and submit it.

    The worker receives no mutable-current alias (§7.1 point 4): it is bound
    the plan's three *immutable* artifacts only, never a live catalog handle
    or the legacy store path directly (that arrives separately, through the
    Phase 1 read-set pin/staging copy every ``("legacy_store", "read")`` kind
    already gets).
    """
    plan_ref = artifact(conn, store, plan_artifact_id)
    plan = json.loads(store.read_verified(plan_ref))
    bindings = {"legacy_manifest.json": plan["manifest_ref"],
               "snapshot_import_request.json": plan["request_ref"],
               "legacy_table_mapping.json": plan["mapping_ref"]}
    profile = profile_named(DEFAULT_POLICY, "legacy_rebuild")
    job = JobSpec(
        kind="snapshot_import",
        implementation_ref=content_hash(worker_source_manifest(repo_root)),
        spec_hash=None,
        environment_ref=content_hash(environment_identity(profile.thread_count or profile.cpu_count)),
        parameters={"expected_ids": ("snapshot_import",), "input_bindings": bindings},
        input_refs=tuple(bindings.values()), dependency_job_ids=(), output_namespace=namespace,
        resource_class="legacy_rebuild", retry_policy_ref="bounded",
        checkpoint_contract_ref="snapshot_import_inspections.v1.0")
    return submit(conn, registry, policy, SubmitRequest(
        namespace=namespace, idempotency_key=idempotency_key, principal="operator", job=job),
        clock=clock)


def submit_candidate_rebuild(conn, store, *, candidate_root, protected_paths, protected_before_hash,
                             tables, sample, registry, policy, clock, idempotency_key, repo_root,
                             namespace="shadow"):
    """Build and submit the ``legacy_rebuild_candidate`` job (§10 point 1)."""
    profile = profile_named(DEFAULT_POLICY, "legacy_rebuild")
    job = JobSpec(
        kind="legacy_rebuild_candidate",
        implementation_ref=content_hash(worker_source_manifest(repo_root)),
        spec_hash=None,
        environment_ref=content_hash(environment_identity(profile.thread_count or profile.cpu_count)),
        parameters={"expected_ids": ("legacy_rebuild_candidate",), "candidate_root": str(candidate_root),
                    "protected_paths": tuple(str(p) for p in protected_paths),
                    "protected_before_hash": protected_before_hash, "tables": tuple(tables or ()),
                    "sample": sample},
        input_refs=(), dependency_job_ids=(), output_namespace=namespace,
        resource_class="legacy_rebuild", retry_policy_ref="bounded",
        checkpoint_contract_ref="legacy_rebuild_candidate.v1.0")
    return submit(conn, registry, policy, SubmitRequest(
        namespace=namespace, idempotency_key=idempotency_key, principal="operator", job=job),
        clock=clock)


# --------------------------------------------------------------------------
# worker: snapshot_import
# --------------------------------------------------------------------------


def _partition_key_for(relative_path: str) -> str:
    match = _YEAR_IN_PATH.search("/" + relative_path)
    return match.group(1) if match else "all"


def _group_by_partition(refs):
    groups: list[tuple[str, list]] = []
    for ref in refs:
        key = _partition_key_for(ref.path)
        if groups and groups[-1][0] == key:
            groups[-1][1].append(ref)
        else:
            groups.append((key, [ref]))
    return groups


def _inspect_table(legacy_root: Path, table: str, contract: TableContract, contract_ref,
                   refs) -> tuple[list[dict], list[str]]:
    partitions_out = []
    completed = []
    for partition_key, group in _group_by_partition(refs):
        fragments = []
        for file_ref in group:
            inspection = inspect_staged_file(
                legacy_root / file_ref.path, contract, contract_ref, partition_key,
                expected_content_hash=file_ref.content_hash, expected_byte_size=file_ref.byte_size)
            fragments.append({
                "path": file_ref.path, "content_hash": file_ref.content_hash,
                "byte_size": file_ref.byte_size, "row_count": inspection.row_count,
                "logical_content_hash": inspection.logical_content_hash,
                "primary_key_min": list(inspection.primary_key_min),
                "primary_key_max": list(inspection.primary_key_max),
                "time_min": inspection.time_min, "time_max": inspection.time_max})
        partitions_out.append({"partition_key": partition_key, "fragments": fragments})
        completed.append(f"{table}:{partition_key}")
    return partitions_out, completed


def worker_snapshot_import(parameters, root: Path) -> dict:
    """Stream every staged legacy file into a per-fragment inspection.

    Reads the bound ``snapshot_import_request.json``/``legacy_table_mapping.json``
    (materialized by the executor before launch, like any other
    ``input_bindings`` entry) and the pinned read set under ``root/legacy``.
    Writes ``inspections.json``, this kind's sole declared output.
    """
    request = from_document(SnapshotImportRequest,
                            json.loads((root / "snapshot_import_request.json").read_text()))
    mapping = json.loads((root / "legacy_table_mapping.json").read_text())
    legacy_root = root / "legacy"

    tables_out: dict[str, list[dict]] = {}
    partitions_completed: list[str] = []
    for table, refs in request.table_sources.items():
        contract = documents.decode_document(TableContract, mapping["tables"][table])
        contract_ref = request.table_contract_refs[table]
        partitions, completed = _inspect_table(legacy_root, table, contract, contract_ref, refs)
        tables_out[table] = partitions
        partitions_completed.extend(completed)

    snapshot_path = legacy_root / request.legacy_snapshot_source_ref.path
    snapshot_payload = json.loads(snapshot_path.read_text())
    document = {
        "schema_version": "snapshot_import_inspections.v1.0",
        "request_hash": request_hash(request), "tables": tables_out,
        "partitions_completed": partitions_completed,
        "legacy_snapshot_top_level_keys": (sorted(snapshot_payload)
                                           if isinstance(snapshot_payload, dict) else None),
    }
    (root / "inspections.json").write_text(json.dumps(document, sort_keys=True))
    return {"outputs": [{"name": "snapshot_import", "path": "inspections.json",
                        "schema": "snapshot_import_inspections.v1.0"}],
            "completed_ids": ["snapshot_import"], "no_work": False}


# --------------------------------------------------------------------------
# worker: legacy_rebuild_candidate
# --------------------------------------------------------------------------


def _hash_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _walk_entries(base: Path) -> list[tuple[str, str]]:
    if not base.exists():
        return [(str(base), "absent")]
    if base.is_file():
        return [(str(base), _hash_bytes(base))]
    return [(str(child), _hash_bytes(child)) for child in sorted(base.rglob("*")) if child.is_file()]


def protected_paths_hash(paths) -> str:
    """A deterministic content fingerprint of every file under ``paths``.

    Pure and side-effect-free (reads only): used identically before a
    candidate rebuild is submitted (pinning "before") and by
    ``snapshot_promotion.legacy_rebuild_candidate_effect`` after it finishes
    ("after", compared to the pin) — the same function on both sides is what
    makes the comparison mean something (§10 point 1's write-confinement
    proof).
    """
    entries: list[tuple[str, str]] = []
    for base in sorted((Path(p) for p in paths), key=str):
        entries.extend(_walk_entries(base))
    return content_hash(sorted(entries))


def worker_legacy_rebuild_candidate(parameters, root: Path) -> dict:
    """Run the real legacy rebuild, rooted at a private candidate directory.

    Records before/after fingerprints of the candidate root and every
    declared protected path as an audit trail; the coordinator effect is
    what actually decides pass/fail (this worker is never trusted to police
    its own writes — see the module docstring).
    """
    candidate_root = Path(parameters["candidate_root"])
    protected = [Path(p) for p in parameters.get("protected_paths", ())]
    before = protected_paths_hash(protected)
    report = run_legacy_rebuild(candidate_root, Path.cwd(),
                                tables=tuple(parameters.get("tables") or ()) or None,
                                sample=parameters.get("sample"))
    after = protected_paths_hash(protected)
    document = {
        "schema_version": "legacy_rebuild_candidate.v1.0", "report": report,
        "candidate_root": str(candidate_root), "candidate_root_hash": protected_paths_hash([candidate_root]),
        "protected_before_hash": before, "protected_after_hash": after,
    }
    (root / "candidate_report.json").write_text(json.dumps(document, sort_keys=True, default=str))
    return {"outputs": [{"name": "legacy_rebuild_candidate", "path": "candidate_report.json",
                        "schema": "legacy_rebuild_candidate.v1.0"}],
            "completed_ids": ["legacy_rebuild_candidate"], "no_work": False}
