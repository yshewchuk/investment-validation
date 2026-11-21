"""EXP-188 real-store Sources adapter, slice 1 (design section 13).

Slice 1 opens the catalog read-only, the artifact store, the sealed plan, the selection record, the admission and the chain index, and derives the roster inputs from the plan. Table-backed fields (`fetch`, `facts`) refuse `source_table_missing` until slice 2.
"""
import functools
import json
import os
import sqlite3
import stat
import sys
from pathlib import Path

import pandas as pd

from engine.v2.contracts import ArtifactRef
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, DocumentError, from_document, to_document
from engine.v2.ops.errors import fail
from engine.v2.ops.experiments import experiment_spec_from_document
from experiments import size_pnl_driver, size_pnl_inputs, size_pnl_ladder, size_pnl_readers
from experiments.magnitude_search_plan import read_search_plan
from experiments.magnitude_search_records import read_record
from experiments.native_registration import read_native_registration

_KEYS = ("catalog", "store_root", "spec", "request", "plan_ref", "snapshot_id", "production_hash")
_LIMIT = 1 << 20
MAX_OBJECT_BYTES = 1 << 28
# same caps as magnitude_search_plan._BOUNDS_CATALOG_LIMITS (the repo's bounded label-bounds admission)
MAX_SNAPSHOT_TABLES = 256
MAX_SNAPSHOT_FRAGMENTS = 65_536
MAX_SNAPSHOT_EVIDENCE_BYTES = 64 * 1024 * 1024
_BLEND_NAMES = {"M4": "size_b1_v1", "M5": "size_b2_v1"}
_INVALID, _CONFLICT = "INVALID_EXPERIMENT_SPEC", "EXPERIMENT_IDENTITY_CONFLICT"


def _refuse(reason, message):
    return fail(_INVALID, message, details={"reason": reason})


def _read_json(path, reason):
    """Read a regular, non-symlink file of at most _LIMIT bytes and parse it as JSON. The path never reaches a message."""
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > _LIMIT:
                raise ValueError("not a regular file of at most 1 MiB")
            data = handle.read(_LIMIT + 1)
        if len(data) > _LIMIT:
            raise ValueError("oversize")
        return json.loads(data.decode("utf-8"))
    except (OSError, ValueError):
        raise _refuse(reason, "input is unreadable or not valid JSON") from None


def _config(path):
    if path is None:
        raise _refuse("sources_config_invalid", "--sources-config is required")
    doc = _read_json(path, "sources_config_invalid")
    if not isinstance(doc, dict) or set(doc) != set(_KEYS) or not all(isinstance(v, str) and v.strip() for v in doc.values()):
        raise _refuse("sources_config_invalid", "sources config must map exactly the required keys to non-blank strings")
    return doc


def derive_roster_inputs(plan):
    """Blend ids from the plan's named blends, and the unique extra-refit candidate per pair role. Ambiguity omits the role."""
    blends = plan.get("blends", [])
    blend_ids = {}
    for role, name in _BLEND_NAMES.items():
        matches = [b["blend_id"] for b in blends if b.get("name") == name]
        if len(matches) == 1:
            blend_ids[role] = matches[0]
    extra = set(plan["selection"]["extra_refits"])
    pool = [c for c in plan["candidates"] if c["candidate_id"] in extra]
    pair_candidates = {}
    for role in size_pnl_inputs.PAIR_ROLES:
        required = size_pnl_inputs.PAIR_TRANSFORM.get(role)
        matches = [c["candidate_id"] for c in pool if size_pnl_inputs._recipe_matches(c["recipe"], role)
                   and (required is None or size_pnl_inputs._transform(c) == required)]
        if len(matches) == 1:
            pair_candidates[role] = matches[0]
    return pair_candidates, blend_ids


def _open(cfg):
    if not Path(cfg["store_root"]).is_dir():
        raise fail("RESOURCE_UNAVAILABLE", "existing artifact store required", details={"reason": "store_missing"})
    try:
        conn = sqlite3.connect(Path(cfg["catalog"]).resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None)
    except sqlite3.Error:
        raise fail("RESOURCE_UNAVAILABLE", "catalog unavailable", details={"reason": "catalog_unavailable"}) from None
    conn.row_factory = sqlite3.Row
    try:
        return conn, ArtifactStore(cfg["store_root"])
    except BaseException:
        conn.close()
        raise


def _bound_snapshot(conn, snapshot_id):
    """Refuse a pinned snapshot whose catalog metadata exceeds the table, fragment or evidence caps. Read-only counts only."""
    try:
        tables = conn.execute("SELECT COUNT(*) FROM data_snapshot_tables WHERE snapshot_id = ?", (snapshot_id,)).fetchone()[0]
        fragments = conn.execute(
            "SELECT COUNT(*) FROM data_version_fragments vf JOIN data_snapshot_tables st ON st.dataset_version_id = vf.dataset_version_id "
            "WHERE st.snapshot_id = ?", (snapshot_id,)).fetchone()[0]
        evidence = conn.execute(
            "SELECT COALESCE(SUM(LENGTH(CAST(dv.evidence_json AS BLOB)) + LENGTH(CAST(dv.partition_logical_hashes_json AS BLOB))), 0) "
            "FROM data_dataset_versions dv JOIN data_snapshot_tables st ON st.dataset_version_id = dv.dataset_version_id "
            "WHERE st.snapshot_id = ?", (snapshot_id,)).fetchone()[0]
        fragment_bytes = conn.execute(
            "SELECT COALESCE(SUM(LENGTH(CAST(COALESCE(f.key_bounds_json,'') AS BLOB)) + LENGTH(CAST(COALESCE(f.time_bounds_json,'') AS BLOB)) "
            "+ LENGTH(CAST(COALESCE(f.input_receipt_refs_json,'') AS BLOB))), 0) "
            "FROM data_version_fragments vf JOIN data_fragments f ON f.fragment_id = vf.fragment_id "
            "JOIN data_snapshot_tables st ON st.dataset_version_id = vf.dataset_version_id "
            "WHERE st.snapshot_id = ?", (snapshot_id,)).fetchone()[0]
        finality_bytes = conn.execute(
            "SELECT COALESCE(SUM(LENGTH(CAST(finality_receipt_refs_json AS BLOB))), 0) FROM data_snapshots WHERE snapshot_id = ?",
            (snapshot_id,)).fetchone()[0]
    except sqlite3.Error:
        raise fail("RESOURCE_UNAVAILABLE", "catalog unavailable", details={"reason": "catalog_unavailable"}) from None
    evidence += fragment_bytes
    evidence += finality_bytes
    if tables > MAX_SNAPSHOT_TABLES or fragments > MAX_SNAPSHOT_FRAGMENTS or evidence > MAX_SNAPSHOT_EVIDENCE_BYTES:
        raise fail(_CONFLICT, "snapshot catalog metadata is too large", details={"reason": "snapshot_metadata_too_large"})


def _snapshot(repository, snapshot_id):
    try:
        return repository.resolve(snapshot_id)
    except DataError as exc:
        if exc.code != "SNAPSHOT_NOT_FOUND":
            raise
        raise fail(_CONFLICT, "pinned snapshot cannot be resolved", details={"reason": "snapshot_missing"}) from None
    except (ValueError, KeyError):
        raise fail(_CONFLICT, "pinned snapshot cannot be resolved", details={"reason": "snapshot_missing"}) from None


def _snapshot_mismatch():
    return fail(_CONFLICT, "the configured snapshot is not the registration's pinned snapshot", details={"reason": "snapshot_mismatch"})


def _table_missing(*_a, **_k):
    raise fail(_CONFLICT, "source tables arrive in slice 2", details={"reason": "source_table_missing"})


def assemble(*, plan, read, selection, expected, panel_id, admit, admitted, snapshot_id, load_index):
    pair_candidates, blend_ids = derive_roster_inputs(plan)
    return size_pnl_driver.Sources(
        plan=plan, read=read, selection=selection, pair_candidates=pair_candidates, blend_ids=blend_ids,
        expected=expected, panel_id=panel_id, admit=admit, admitted=admitted, fetch=_table_missing,
        production_hash="", production_table_hash="", plan_frame=pd.DataFrame({"event_id": [], "exit_date": []}),
        load_index=load_index, facts=_table_missing, snapshot_id=snapshot_id)


def build_sources(args):
    """Public factory for `run.py --sources`. The connection stays open for the run's `read` closure; it closes on any refusal."""
    cfg = _config(getattr(args, "sources_config", None))
    expected = getattr(args, "expected_identities", None)
    if not (isinstance(expected, dict) and all(isinstance(expected.get(k), str) and expected[k].strip()
                                               for k in ("plan_id", "registration_id", "seal_id", "panel_id"))):
        raise _refuse("expected_invalid", "expected identities must hold plan_id, registration_id, seal_id and panel_id")
    try:
        spec = experiment_spec_from_document(_read_json(cfg["spec"], "input_unreadable"))
        ref = from_document(ArtifactRef, _read_json(cfg["plan_ref"], "input_unreadable"))
    except (DocumentError, TypeError):
        raise _refuse("input_unreadable", "a SIZE input document is invalid") from None
    request = _read_json(cfg["request"], "input_unreadable")
    conn, store = _open(cfg)
    try:
        registration = read_native_registration(conn, store, spec)
        pinned = registration.document.get("snapshot")
        if not (isinstance(pinned, dict) and pinned.get("snapshot_id") == cfg["snapshot_id"]):
            raise _snapshot_mismatch()
        repository = Repository(conn, store, pin_snapshots=True)
        _bound_snapshot(conn, cfg["snapshot_id"])
        snapshot = _snapshot(repository, cfg["snapshot_id"])
        if to_document(snapshot) != pinned:
            raise _snapshot_mismatch()
        plan = read_search_plan(conn, store, registration, request, ref)
        read = functools.partial(read_record, conn, store, registration.run_id)
        selection = read("selection")
        if not (isinstance(selection, dict) and isinstance(selection.get("selections"), list)):
            raise fail(_CONFLICT, "the selection record is missing", details={"reason": "selection_missing"})
        month = registration.document["holdouts"]["holdout_as_of_month"]
        admit = size_pnl_inputs.population_admit(repository, snapshot, month, max_object_bytes=MAX_OBJECT_BYTES)
        admitted = size_pnl_readers.admit_events(repository, snapshot, month, max_object_bytes=MAX_OBJECT_BYTES)
        load_index = size_pnl_ladder.default_load_index(repository, snapshot)
        sources = assemble(plan=plan, read=read, selection=selection, expected=expected, panel_id=expected["panel_id"],
                           admit=admit, admitted=admitted, snapshot_id=cfg["snapshot_id"], load_index=load_index)
    except BaseException:
        conn.close()
        raise
    print(f"[pnl_sources] opened snapshot={cfg['snapshot_id']} admitted={len(admitted['events'])}", file=sys.stderr)
    return sources
