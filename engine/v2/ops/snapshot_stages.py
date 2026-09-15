"""Supervisor half of snapshot-backed legacy stages — P2-6 §9.3, D13.

Three launch modes, decided from the claimed ``JobSpec`` alone:

* ``legacy`` — the Phase 1 path, unchanged: ``_pin_read_set`` plus a private
  ``staging/legacy`` copy under the cooperative barrier.
* ``materialize`` — kind ``legacy_materialize``. Before launch the bound
  ``SnapshotRef`` and ``LegacyMaterializationRequest`` are decoded and
  validated (request built for exactly that snapshot, the snapshot still
  resolves to the same committed manifest, request hash covers the content,
  ``read_plan_complete`` holds), and the scratch estimate must fit. After the
  worker, :func:`materialize_effect` re-verifies every byte of the root against
  the worker's manifest, checks that manifest against the request's own
  layout (pinned refs, SNAPSHOT bytes, whole-table object copies), and refuses
  a manifest that differs from one already committed for the same request.
* ``snapshot`` — ``input_mode == "snapshot"`` on a declared kind
  (``stages.SNAPSHOT_BACKED_KINDS``). Same request validation, then the bound
  manifest must be one a ``legacy_materialize`` attempt committed for this
  request, and the root must match it byte for byte (0444 files, 0555
  directories, no links, no extras). No read pin, no staging copy; the worker
  gets that root and nothing else. At finish :func:`confirm_attempt` requires
  the attempt's recorded bindings to name the same three artifacts and the
  same request hash, and the root's stat fingerprint to be unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from engine.v2.contracts import LegacyMaterializationRequest, SnapshotRef
from engine.v2.data import tier4_coverage
from engine.v2.data.documents import decode_document
from engine.v2.data.errors import DataError
from engine.v2.data.legacy_materialization import (
    LEGACY_SCORE_READ_PLAN_V1,
    TABLE_OUTPUT_KIND,
    panel_object_ref,
    parse_pinned_ref,
    px_expected_paths,
    read_plan_complete,
)
from engine.v2.data.reference_inputs import (
    LEGACY_SNAPSHOT_PATH,
    REGISTRY_PATH,
    TIER4_SERVING_DIR,
    champion_entries,
)
from engine.v2.data.repository import Repository
from engine.v2.foundation import CONTENT_HASH_PREFIX, content_hash, to_document
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.input_bindings import recorded_bindings, resolve_bindings
from engine.v2.ops.snapshot_roots import (
    MANIFEST_SCHEMA_REF,
    materialization_root,
    stat_fingerprint,
    verify_root,
)
from engine.v2.ops.stages import SNAPSHOT_BINDINGS

__all__ = ["MATERIALIZE_KIND", "MANIFEST_OUTPUT", "SnapshotLaunch", "cache_inputs",
           "committed_manifest_ids", "confirm_attempt", "launch_mode", "materialize_effect",
           "prepare_launch", "request_from_artifact", "request_mismatches",
           "snapshot_cache_inputs", "validated_request"]

#: The snapshot-backed kinds P2-C02's launch refusal covers. ``legacy_score_
#: requests`` is excluded: its planned population lives in per-request rows
#: under ``requests_path`` inside the (not-yet-launched) staging root, not in
#: ``JobSpec.parameters`` the way ``expected_population``/``session`` are for
#: the other two -- out of scope for this task (task brief decision 2).
_COVERAGE_KINDS = frozenset({"legacy_score", "legacy_decision_replay"})

MATERIALIZE_KIND = "legacy_materialize"
MANIFEST_OUTPUT = "materialization_manifest"
_SINGLE_FILE_PATHS = {"feature_panel": "data/features/panel.parquet",
                      "tier4_forecasts": "data/features/tier4_forecasts.parquet"}
_COMMITTED_MANIFESTS = """
SELECT DISTINCT o.artifact_id FROM attempt_input_bindings b
JOIN attempts a ON a.attempt_id = b.attempt_id AND a.state = 'succeeded'
JOIN jobs j ON j.job_id = a.job_id AND j.kind = 'legacy_materialize'
JOIN attempt_outputs o ON o.attempt_id = a.attempt_id AND o.name = 'materialization_manifest'
WHERE b.name = 'materialization_request.json' AND b.artifact_id = ?
ORDER BY o.artifact_id
"""


@dataclass(frozen=True)
class SnapshotLaunch:
    """What the supervisor verified before launch, kept until finish."""

    mode: str
    root: Path
    snapshot_artifact_id: str
    request_artifact_id: str
    request_hash: str
    snapshot_manifest_hash: str
    manifest_artifact_id: str = ""
    manifest_content_hash: str = ""
    fingerprint: dict = field(default_factory=dict)
    envelope_extra: dict = field(default_factory=dict)

    @property
    def worker_legacy_root(self):
        """Only a snapshot-backed stage's worker is rooted at the materialization."""
        return self.root if self.mode == "snapshot" else None


def launch_mode(spec) -> str:
    if spec.kind == MATERIALIZE_KIND:
        return "materialize"
    if (spec.parameters or {}).get("input_mode") == "snapshot":
        return "snapshot"
    if spec.kind == "legacy_finality" and _has_finality_cross_check_bindings(spec):
        return "finality_check"
    return "legacy"


def _has_finality_cross_check_bindings(spec) -> bool:
    """Last read-set gap fix (2026-09-15): ``legacy_finality`` never declares
    ``input_mode="snapshot"`` (``stages.input_mode_problems`` refuses it -- no
    declared read plan for the kind), but a snapshot-mode plan graph still
    binds it the same three ``SNAPSHOT_BINDINGS`` read-only, so its worker can
    cross-check its own barrier read against this run's committed
    materialization (``legacy_adapter._action_finality``). A legacy-mode plan
    leaves these unbound, so this is false and the kind launches as plain
    ``"legacy"``, exactly as before this fix."""
    bindings = (spec.parameters or {}).get("input_bindings") or {}
    return all(name in bindings for name in SNAPSHOT_BINDINGS)


def _catalog_path(conn) -> str:
    for row in conn.execute("PRAGMA database_list").fetchall():
        if row[1] == "main" and row[2]:
            return row[2]
    raise fail("RESOURCE_UNAVAILABLE", "the operations catalog has no file path")


def _read_json(conn, store, artifact_id):
    return json.loads(store.read_verified(artifact(conn, store, artifact_id)))


def _decode(cls, document):
    try:
        return decode_document(cls, document)
    except Exception:
        raise fail("INPUT_CHANGED", "bound snapshot document is malformed",
                   details={"contract": cls.__name__}) from None


def _bound(resolved, name):
    item = resolved.get(name)
    if item is None:
        raise fail("INPUT_CHANGED", "snapshot-backed input is not bound", details={"name": name})
    return item


def request_from_artifact(conn, store, artifact_id) -> LegacyMaterializationRequest:
    request = _decode(LegacyMaterializationRequest, _read_json(conn, store, artifact_id))
    document = to_document(request)
    del document["request_hash"]
    if content_hash(document) != request.request_hash:
        raise fail("INPUT_CHANGED", "materialization request hash does not cover its content")
    return request


def validated_request(conn, store, snapshot_artifact_id, request_artifact_id):
    """(SnapshotRef, request) — refused unless the request was built for exactly
    this committed snapshot and its read plan is provably complete."""
    snapshot = _decode(SnapshotRef, _read_json(conn, store, snapshot_artifact_id))
    request = request_from_artifact(conn, store, request_artifact_id)
    if request.snapshot_ref != snapshot:
        raise fail("INPUT_CHANGED", "materialization request was built for another snapshot")
    repository = Repository(conn, store)
    try:
        if repository.resolve(snapshot.snapshot_id) != snapshot:
            raise fail("INPUT_CHANGED", "bound SnapshotRef differs from the committed snapshot")
        complete = read_plan_complete(request, repository)
    except DataError as exc:
        raise fail("INPUT_CHANGED", "bound snapshot cannot be resolved",
                   details={"data_code": exc.code}) from None
    if not complete:
        raise fail("INPUT_CHANGED", "materialization read plan is not complete")
    return snapshot, request


def committed_manifest_ids(conn, request_artifact_id) -> list[str]:
    return [row[0] for row in conn.execute(_COMMITTED_MANIFESTS, (request_artifact_id,))]


def manifest_files(document, request) -> dict:
    if not isinstance(document, dict) or document.get("schema_version") != MANIFEST_SCHEMA_REF:
        raise fail("INPUT_CHANGED", "materialization manifest has an unsupported schema")
    claimed = (document.get("request_hash"), document.get("snapshot_id"),
               document.get("snapshot_manifest_hash"))
    if claimed != (request.request_hash, request.snapshot_ref.snapshot_id,
                   request.snapshot_ref.manifest_hash):
        raise fail("INPUT_CHANGED", "materialization manifest belongs to another request")
    files = document.get("files")
    if not isinstance(files, dict) or not files or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in files.items()):
        raise fail("INPUT_CHANGED", "materialization manifest is malformed")
    return files


def _effective_session(conn, store, resolved, parameters) -> str:
    """The ``as_of`` the worker will actually score at -- never the bare
    requested ``session``.

    P2-C03: ``legacy_score``/``legacy_decision_replay`` now bind ``finality.
    json`` and resolve the FINALITY-WALKED-BACK session from it
    (``legacy_actions._action_score``/``_action_decision_replay``), which can
    be strictly earlier than ``parameters["session"]``. Since
    ``tier4.serving_fold`` takes ``min(event_fold, decision_fold(as_of))``, a
    coverage check built on the later requested date can compute a fold the
    real run never asks for -- a false refusal, not merely an imprecise one.
    Falls back to the requested session only when no ``finality.json``
    binding is present (a caller that predates that binding); a bound one
    that fails to resolve refuses here exactly as it would inside the worker,
    just earlier.
    """
    requested = parameters.get("session")
    finality_item = resolved.get("finality.json")
    if finality_item is None:
        return requested
    from engine.v2.ops.session_resolution import resolve_effective_session

    finality = _read_json(conn, store, finality_item.artifact_id)
    return resolve_effective_session(finality, requested)


def _population_pairs(parameters, as_of: str) -> tuple[tuple[str, str], ...]:
    """``(event_date, as_of)`` pairs from a ``legacy_score``/``legacy_decision_
    replay`` job's own ``LegacyParameters`` -- ``expected_population`` keys
    (``ticker|strategy|event_date``, A2's own planned-population shape) each
    paired with the job's own effective session (:func:`_effective_session`)."""
    pairs = []
    for key in parameters.get("expected_population") or ():
        parts = str(key).split("|")
        if len(parts) == 3 and parts[2]:
            pairs.append((parts[2], as_of))
    return tuple(pairs)


def _pinned_tier4_refs(request: LegacyMaterializationRequest) -> dict[str, str]:
    prefix = TIER4_SERVING_DIR + "/"
    out = {}
    for ref in request.registry_and_model_refs:
        path, digest = parse_pinned_ref(ref)
        if path.startswith(prefix):
            out[path] = digest
    return out


def _check_tier4_coverage(conn, store, claim, request: LegacyMaterializationRequest, root: Path,
                          resolved) -> None:
    """P2-C02: refuse a snapshot-backed launch whose pinned Tier-4 serving
    caches do not cover what its planned population needs. A genuinely
    multi-fragment ``feature_panel`` has no one predictable panel hash
    (``panel_object_ref`` returns ``None``) -- the same case
    ``legacy_materialization._check_tier4_cache_refs`` already treats as
    vacuously satisfied, for the identical reason."""
    if claim.spec.kind not in _COVERAGE_KINDS:
        return
    parameters = claim.spec.parameters or {}
    as_of = _effective_session(conn, store, resolved, parameters)
    population = _population_pairs(parameters, as_of)
    if not population:
        return
    panel_ref = panel_object_ref(Repository(conn, store), request.snapshot_ref)
    if panel_ref is None:
        return
    panel_sha = panel_ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)
    try:
        registry_models = tier4_coverage.champion_producer_models(champion_entries(root / REGISTRY_PATH))
    except DataError as exc:
        raise fail("INPUT_CHANGED", "materialized model registry is not the reviewed shape",
                   details={"data_code": exc.code}) from None
    required = tier4_coverage.required_serving_triples(population, registry_models, panel_sha)
    missing = tier4_coverage.missing_triples(required, _pinned_tier4_refs(request), store,
                                             registry_models=registry_models)
    if missing:
        raise fail("TIER4_CACHE_MISSING",
                   "pinned Tier-4 serving caches do not cover the planned population",
                   details={"missing": missing})


def _check_scratch(claim):
    needed = int((claim.spec.parameters or {}).get("scratch_estimate_bytes", 0))
    if needed > claim.resources.scratch_limit_bytes:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "materialization exceeds the scratch budget",
                   details={"needed_bytes": needed,
                            "scratch_limit_bytes": claim.resources.scratch_limit_bytes})


def prepare_launch(conn, store, claim, *, base):
    """Validate before launch; ``None`` for a barrier-path (legacy) attempt."""
    mode = launch_mode(claim.spec)
    if mode == "legacy":
        return None
    resolved = resolve_bindings(conn, store, claim.spec)
    snapshot_item = _bound(resolved, "snapshot_ref.json")
    request_item = _bound(resolved, "materialization_request.json")
    snapshot, request = validated_request(conn, store, snapshot_item.artifact_id,
                                          request_item.artifact_id)
    common = dict(root=materialization_root(base, request.request_hash),
                  snapshot_artifact_id=snapshot_item.artifact_id,
                  request_artifact_id=request_item.artifact_id,
                  request_hash=request.request_hash, snapshot_manifest_hash=snapshot.manifest_hash)
    if mode == "materialize":
        _check_scratch(claim)
        extra = {"materialization": {"catalog_path": _catalog_path(conn),
                                     "store_root": str(store.root), "base": str(base)}}
        return SnapshotLaunch(mode=mode, envelope_extra=extra, **common)
    manifest_item = _bound(resolved, "materialization_manifest.json")
    if manifest_item.artifact_id not in committed_manifest_ids(conn, request_item.artifact_id):
        raise fail("INPUT_CHANGED", "materialization manifest was not committed for this request")
    files = manifest_files(_read_json(conn, store, manifest_item.artifact_id), request)
    fingerprint = verify_root(common["root"], files)
    if mode == "finality_check":
        # Last read-set gap fix (2026-09-15): ``legacy_finality`` stays
        # rooted at its own barrier legacy tree (``worker_legacy_root`` is
        # ``None`` for every mode but ``"snapshot"``) -- this branch only
        # verifies the SAME materialization score/render already trust and
        # exposes its path to the worker via ``envelope_extra``, for
        # ``_action_finality``'s own read-only content cross-check. No tier4
        # coverage check: that guards the scoring population's own serving
        # caches, which finality never reads.
        return SnapshotLaunch(mode=mode, manifest_artifact_id=manifest_item.artifact_id,
                              manifest_content_hash=manifest_item.content_hash,
                              fingerprint=fingerprint,
                              envelope_extra={"finality_cross_check":
                                             {"materialization_root": str(common["root"])}},
                              **common)
    _check_tier4_coverage(conn, store, claim, request, common["root"], resolved)
    return SnapshotLaunch(mode=mode, manifest_artifact_id=manifest_item.artifact_id,
                          manifest_content_hash=manifest_item.content_hash,
                          fingerprint=fingerprint, **common)


def snapshot_cache_inputs(base_inputs, *, snapshot_manifest_hash, request_hash,
                          manifest_content_hash):
    """§9.3 item 5: the checkpoint ``inputs`` of a snapshot-backed stage."""
    return content_hash({"bindings": base_inputs, "snapshot_manifest_hash": snapshot_manifest_hash,
                         "materialization_request_hash": request_hash,
                         "materialization_manifest_hash": manifest_content_hash})


def cache_inputs(base_inputs, launch):
    # Last read-set gap fix (2026-09-15): a checkpoint reused from an earlier
    # ``finality_check`` attempt must not skip re-running the cross-check
    # against a materialization that has since changed identity, so its
    # cache identity folds in the same three hashes a real "snapshot" stage's
    # does.
    if launch is None or launch.mode not in ("snapshot", "finality_check"):
        return base_inputs
    return snapshot_cache_inputs(base_inputs, snapshot_manifest_hash=launch.snapshot_manifest_hash,
                                 request_hash=launch.request_hash,
                                 manifest_content_hash=launch.manifest_content_hash)


def confirm_attempt(conn, store, claim, launch):
    """§9.3 item 4: the same verified binding, recorded for this attempt, before admission.

    ``finality_check`` (last read-set gap fix, 2026-09-15) gets the same
    re-verification as ``"snapshot"`` -- the three bindings recorded for this
    attempt still name the exact materialization the worker's own cross-check
    ran against, and the root has not changed since -- ``legacy_finality``'s
    ``worker_legacy_root``/staged legacy tree stay untouched either way
    (only ``"snapshot"`` mode roots the worker there)."""
    if launch_mode(claim.spec) not in ("snapshot", "finality_check"):
        return
    if launch is None or launch.mode not in ("snapshot", "finality_check"):
        raise fail("INPUT_CHANGED", "snapshot-backed attempt has no verified launch")
    recorded = recorded_bindings(conn, claim.attempt_id)
    observed = tuple(getattr(recorded.get(name), "artifact_id", None) for name in SNAPSHOT_BINDINGS)
    if observed != (launch.snapshot_artifact_id, launch.request_artifact_id,
                    launch.manifest_artifact_id):
        raise fail("INPUT_CHANGED", "recorded snapshot bindings differ from the verified launch")
    if request_from_artifact(conn, store, launch.request_artifact_id).request_hash \
            != launch.request_hash:
        raise fail("INPUT_CHANGED", "recorded materialization request hash changed")
    try:
        unchanged = stat_fingerprint(launch.root) == launch.fingerprint
    except OpsError:
        unchanged = False
    if not unchanged:
        raise fail("INPUT_CHANGED", "materialization root changed during the attempt")


def _curated_copies(table, records):
    out, ordinals = {}, {}
    for record in records:
        year = int(record.partition_key)
        ordinal = ordinals.get(year, 0)
        ordinals[year] = ordinal + 1
        out[f"data/curated/{table}/year={year}/part-{ordinal:04d}.parquet"] = \
            record.object_ref.content_hash
    return out


def _expected_layout(repository, request):
    """(exact ``{path: hash}``, required paths of unknown hash, free prefixes)."""
    exact = {LEGACY_SNAPSHOT_PATH: request.legacy_snapshot_object_ref.content_hash}
    for ref in (*request.registry_and_model_refs, *request.calendar_refs):
        path, digest = parse_pinned_ref(ref)
        exact[path] = digest
    required, prefixes = set(), []
    for table in request.table_queries:
        whole = LEGACY_SCORE_READ_PLAN_V1["tables"][table]["scope"] == "whole_table"
        records = repository.fragment_records(request.snapshot_ref, table)
        if TABLE_OUTPUT_KIND[table] == "single_file":
            if whole and len(records) == 1:
                exact[_SINGLE_FILE_PATHS[table]] = records[0].object_ref.content_hash
            else:
                required.add(_SINGLE_FILE_PATHS[table])
        elif whole:
            exact.update(_curated_copies(table, records))
        else:
            prefixes.append(f"data/curated/{table}/")
    # task brief 2026-09-15: ``px_<T>.csv`` files are not object-store copies
    # (no pinned hash to check exactly), and which tickers get one depends on
    # live ``price_history`` content, not the request's static shape -- so
    # ``px_expected_paths`` asks the repository the same question
    # ``materialize_price_series`` itself answers. Every one of these MUST be
    # declared (a missing one is "wrong" below the same way a missing
    # required single-file path is); no other px path may appear.
    required |= px_expected_paths(repository, request)
    return exact, required, tuple(prefixes)


def request_mismatches(repository, request, files) -> list[str]:
    """Paths where a manifest disagrees with what its request provably writes."""
    exact, required, prefixes = _expected_layout(repository, request)
    wrong = [path for path, digest in sorted(exact.items()) if files.get(path) != digest]
    wrong += sorted(path for path in required if path not in files)
    wrong += sorted(path for path in files if path not in exact and path not in required
                    and not path.startswith(prefixes))
    return wrong


def _named(refs, name):
    for candidate, ref in refs:
        if candidate == name:
            return ref
    raise fail("VALIDATION_FAILED", "required effect artifact is missing")


def materialize_effect(conn, store, claim, refs, launch, *, keepalive=None):
    """Admit a ``legacy_materialize`` manifest only once every byte is re-verified."""
    if launch is None or launch.mode != "materialize":
        raise fail("VALIDATION_FAILED", "materialization launch was not verified")
    manifest_ref = _named(refs, MANIFEST_OUTPUT)
    request = request_from_artifact(conn, store, launch.request_artifact_id)
    files = manifest_files(json.loads(store.read_verified(manifest_ref)), request)
    prior = committed_manifest_ids(conn, launch.request_artifact_id)
    if prior and prior != [manifest_ref.artifact_id]:
        raise fail("INPUT_CHANGED", "materialization root no longer matches its committed manifest")
    verify_root(launch.root, files, keepalive=keepalive)
    wrong = request_mismatches(Repository(conn, store), request, files)
    if wrong:
        raise fail("VALIDATION_FAILED", "materialization manifest does not match its request",
                   details={"paths": wrong[:5]})
    return None, ()
